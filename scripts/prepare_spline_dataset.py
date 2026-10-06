#!/usr/bin/env python3
"""Prepare split-preserving spline caches with explicit per-split caps."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import pyvista as pv
from data.spline_targets import contract_spline_graph, crop_dense_paths, fit_spline, resample
from scripts.audit_synthetic_mri_grid import SourceGraph, patch_voxel_cell_bounds


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_patient(task):
    source_root, graph_root, output_root, patient, rows, config, controls = task
    source_root, graph_root, output_root = Path(source_root), Path(graph_root), Path(output_root)
    # Original full MRI volumes may have been removed after materialization.
    # The generator required identical raw/seg affines; verify this transform
    # again against every stored VTP before constructing new spline geometry.
    image = nib.load(source_root/'seg'/f'{patient}.nii.gz')
    affine = image.affine
    assert image.header.get_xyzt_units()[0] == 'mm'
    source = SourceGraph.from_directory(graph_root/patient).transformed(np.linalg.inv(affine))
    crop_size, pad = np.array(config['crop_size']), np.array(config['pad'])
    shape = np.array(config['patch_size'])
    metric = affine[:3, :3] @ np.diag(shape)
    inverse = np.linalg.inv(metric)
    accepted, excluded = [], []
    for row in rows:
        sample, split = row['sample_id'], row['split']
        start = np.array([int(row['start_'+a]) for a in ('d', 'h', 'w')])
        stored_bounds = np.column_stack((start, start+crop_size-1))
        sparse = source.crop(stored_bounds)
        stored = pv.read(source_root/'new_patches_boundary'/split/'vtp'/f'{sample}_graph.vtp')
        assert stored.n_points == len(sparse.positions) == int(row['node_count']), sample
        assert len(sparse.edges) == int(row['edge_count']), sample
        assert np.allclose(stored.points, (sparse.positions-start+pad)/shape, atol=2e-5, rtol=0), sample
        bounds = patch_voxel_cell_bounds(start, crop_size)
        dense = crop_dense_paths(source, bounds, retain_tangent_contacts=True,
                                 boundary_policy='last_inside_sample')
        graph, reason = contract_spline_graph((dense.positions-start+pad)/shape, dense.edges,
                                             closed_components=dense.closed_components)
        if reason:
            excluded.append({'sample_id': sample, 'patient_id': patient, 'split': split, 'reason': reason})
            continue
        ids = sorted(graph.anchors)
        remap = {node: i for i, node in enumerate(ids)}
        nodes = np.asarray([graph.anchors[i] for i in ids]).reshape(-1, 3)
        edges, fitted, curves, guides, paths, offsets = [], [], [], [], [], [0]
        for branch in graph.branches:
            guide = branch.polyline @ metric.T
            a, b = branch.start, branch.end
            if tuple(guide[0]) > tuple(guide[-1]):
                guide = guide[::-1].copy(); a, b = b, a
            c, curve = fit_spline(guide, controls, 64)
            edges.append((remap[a], remap[b]))
            fitted.append(c @ inverse.T); curves.append(curve @ inverse.T)
            guides.append(resample(guide, 64) @ inverse.T)
            paths.append(guide @ inverse.T); offsets.append(offsets[-1]+len(guide))
        edges = np.asarray(edges, dtype='int64').reshape(-1, 2)
        leaf = output_root/split
        for directory in ('raw', 'seg', 'vtp', 'spline'):
            (leaf/directory).mkdir(parents=True, exist_ok=True)
        for directory, suffix in (('raw', 'data'), ('seg', 'seg')):
            src = source_root/'new_patches_boundary'/split/directory/f'{sample}_{suffix}.nii.gz'
            assert nib.load(src).shape == tuple(shape), sample
            dest = leaf/directory/src.name
            if not dest.exists():
                dest.symlink_to(src)
        mesh = pv.PolyData(nodes)
        if len(edges):
            mesh.lines = np.column_stack((np.full(len(edges), 2), edges)).ravel()
        # Empty and isolated-node patches remain part of the evaluated population.
        mesh.save(leaf/'vtp'/f'{sample}_graph.vtp')
        np.savez_compressed(leaf/'spline'/f'{sample}.npz', nodes=nodes.astype('float32'), edges=edges,
                            controls=np.asarray(fitted, dtype='float32').reshape(-1, controls, 3),
                            curves=np.asarray(curves, dtype='float32').reshape(-1, 64, 3),
                            guides=np.asarray(guides, dtype='float32').reshape(-1, 64, 3), metric=metric.astype('float32'),
                            full_paths=(np.concatenate(paths) if paths else np.empty((0, 3))).astype('float32'),
                            path_offsets=np.asarray(offsets), boundary=dense.boundary[ids])
        accepted.append({**row, 'anchor_count': len(nodes), 'branch_count': len(edges),
                         'spline_tangent_contacts': dense.tangent_contacts,
                         'snapped_endpoints': dense.snapped_endpoints,
                         'sparse_sample_fallbacks': dense.sparse_sample_fallbacks})
    result = {'patient_id': patient, 'accepted': accepted, 'excluded': excluded,
              'source_affine': affine.tolist(), 'graph_sha256': digest(graph_root/patient/'graph.vvg')}
    (output_root/'patients'/f'{patient}.json').write_text(json.dumps(result)+'\n')
    print(f'patient={patient} accepted={len(accepted)} excluded={len(excluded)}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI'))
    parser.add_argument('--graph-root', type=Path, help='Preserved full-graph snapshot; default: SOURCE/graphs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-limit', type=int, default=10000)
    parser.add_argument('--val-limit', type=int)
    parser.add_argument('--test-limit', type=int)
    parser.add_argument('--controls', type=int, default=8)
    parser.add_argument('--seed', type=int, default=364505)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--initial-manifest', type=Path, required=True)
    parser.add_argument('--pilot-available-subjects', action='store_true',
                        help='Pilot only: select from subjects with complete full-path source graphs')
    parser.add_argument('--available-subjects-only', action='store_true',
                        help='Use the user-authorized subset with complete original full graphs')
    args = parser.parse_args()
    args.graph_root = (args.graph_root or args.source/'graphs').resolve()
    patches = args.source/'new_patches_boundary'
    config = json.loads((patches/'generation_config.json').read_text())
    assert config['format_version'] == 3 and config['patch_size'] == [64, 64, 64]
    with (patches/'patch_index.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    populations = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val', 'test')}
    original_counts = {s: len(rs) for s, rs in populations.items()}
    availability = {patient: {name: (args.graph_root/patient/name).is_file()
                    for name in ('nodes.csv', 'edges.csv', 'graph.vvg')}
                    for patient in {r['patient_id'] for r in rows}}
    missing = sorted([p for p, files in availability.items() if not all(files.values())], key=int)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'source-availability.json').write_text(json.dumps(availability, indent=2)+'\n')
    if args.pilot_available_subjects or args.available_subjects_only:
        print('Source graph availability restricts membership; excluded subjects recorded in manifest', flush=True)
        populations = {s: [r for r in rs if r['patient_id'] not in missing] for s, rs in populations.items()}
    elif missing:
        raise FileNotFoundError(f'Full source graphs missing for subjects {missing}; complete test preparation cannot proceed')
    subjects = {s: {r['patient_id'] for r in rs} for s, rs in populations.items()}
    assert not (subjects['train'] & subjects['val'] or subjects['train'] & subjects['test'] or subjects['val'] & subjects['test'])
    prior = json.loads(args.initial_manifest.read_text())
    exposure = {}
    for split in ('train', 'validation'):
        ids = set().union(*(set(item['sample_ids']) for item in prior[split]))
        exposed_subjects = {str(int(sample.split('_')[1])) for sample in ids}
        overlap = sorted(exposed_subjects & subjects['test'])
        assert not overlap, f'Pretrained checkpoint exposed to test subjects: {overlap}'
        exposure[split] = {'samples': len(ids), 'test_subject_overlap': overlap}
    selected = []
    for split, limit in (('train', args.train_limit), ('val', args.val_limit), ('test', args.test_limit)):
        ranked = sorted(populations[split], key=lambda row: hashlib.sha256(
            f"{args.seed}\0{split}\0{row['sample_id']}".encode()).digest())
        selected.extend(ranked if limit is None else ranked[:limit])
    graph_hashes = {p: {name: digest(args.graph_root/p/name)
                       for name in ('nodes.csv', 'edges.csv', 'graph.vvg')}
                    for p in sorted(set().union(*subjects.values()), key=int)}
    plan = {'controls': args.controls, 'seed': args.seed, 'source': str(args.source), 'retain_tangent_contacts': True,
            'graph_root': str(args.graph_root), 'source_graph_hashes': graph_hashes,
            'pilot_available_subjects': args.pilot_available_subjects,
            'available_subjects_only': args.available_subjects_only,
            'boundary_policy': 'last_inside_sample_voxel_cells_with_sparse_component_exact_fallback',
            'index_sha256': digest(patches/'patch_index.csv'), 'initial_manifest_sha256': digest(args.initial_manifest),
            'selected_ids': sorted(r['sample_id'] for r in selected)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'patients').mkdir(exist_ok=True)
    plan_path = args.output/'preparation-plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, 'Output belongs to another preparation plan'
    else:
        plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    grouped = defaultdict(list)
    for row in selected:
        grouped[row['patient_id']].append(row)
    tasks, results = [], []
    for patient, patient_rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        cached = args.output/'patients'/f'{patient}.json'
        if cached.exists():
            results.append(json.loads(cached.read_text()))
        else:
            tasks.append((str(args.source), str(args.graph_root), str(args.output), patient, patient_rows, config, args.controls))
    if args.workers == 1:
        results.extend(map(prepare_patient, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results.extend(pool.map(prepare_patient, tasks))
    accepted = sorted([r for result in results for r in result['accepted']], key=lambda r: r['sample_id'])
    excluded = [r for result in results for r in result['excluded']]
    for split in populations:
        assert any(r['split'] == split for r in accepted), f'No compatible {split} samples'
    with (args.output/'patch_index.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(accepted[0])); writer.writeheader(); writer.writerows(accepted)
    manifest = {'source': str(args.source), 'controls': args.controls, 'samples': 64, 'seed': args.seed,
                'graph_root': str(args.graph_root), 'source_graph_hashes': graph_hashes,
                'selection': 'Seeded hash ranking within original splits before compatibility checks; caps recorded explicitly; no image/shape quality filtering.',
                'limits': {'train': args.train_limit, 'val': args.val_limit, 'test': args.test_limit},
                'split_policy': 'Original subject-disjoint source splits preserved. Empty/isolated nodes and singleton boundary contacts retained.',
                'guide': 'Original generator paths are uncertain references, not perfect anatomical ground truth.',
                'pilot_available_subjects': args.pilot_available_subjects,
                'available_subjects_only': args.available_subjects_only,
                'excluded_source_subjects': {s: sorted({r['patient_id'] for r in rows
                    if r['split'] == s and r['patient_id'] in missing}, key=int) for s in populations},
                'available_subject_counts': {s: len(subjects[s]) for s in populations},
                'boundary_policy': plan['boundary_policy'],
                'affine_source': 'Original full segmentation NIfTI header; generator checked raw/seg affine equality; all stored VTP correspondences rechecked.',
                'boundary_counts': {s: {key: sum(int(r[key]) for r in accepted if r['split'] == s)
                    for key in ('spline_tangent_contacts', 'snapped_endpoints', 'sparse_sample_fallbacks')} for s in populations},
                'crop_config': config, 'initial_checkpoint_exposure': exposure,
                'source_counts': original_counts,
                'available_counts': {s: len(rs) for s, rs in populations.items()},
                'requested_counts': dict(Counter(r['split'] for r in selected)),
                'accepted_counts': dict(Counter(r['split'] for r in accepted)),
                'excluded_counts': {s: dict(Counter(r['reason'] for r in excluded if r['split'] == s)) for s in populations},
                'excluded': excluded, 'patients': {r['patient_id']: {'affine': r['source_affine'], 'graph_sha256': r['graph_sha256']} for r in results},
                'index_sha256': plan['index_sha256'], 'initial_manifest_sha256': plan['initial_manifest_sha256']}
    (args.output/'generation_config.json').write_text(json.dumps(config, indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ('source_counts', 'requested_counts', 'accepted_counts', 'excluded_counts')}, indent=2))


if __name__ == '__main__':
    main()
