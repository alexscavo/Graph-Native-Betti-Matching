#!/usr/bin/env python3
"""Prepare four complete SyntheticMRI branch targets before training.

The val folder deliberately mirrors train for overfit monitoring, never a
held-out validation claim. Source dataset files are read-only.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import pyvista as pv
from data.spline_targets import crop_dense_paths, fit_spline, resample
from metrics.branch_connectivity import contract_degree_two
from metrics.representation_geometry import point_to_polyline_distance, polyline_length, sample_polyline_by_arclength
from scripts.audit_synthetic_mri_grid import SourceGraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI'))
    parser.add_argument('--output', type=Path, default=Path('experiments/spline_learning/targets_k8'))
    parser.add_argument('--controls', type=int, default=8)
    parser.add_argument('--count', type=int, default=4)
    args = parser.parse_args()
    patches = args.source/'new_patches_boundary'
    config = json.loads((patches/'generation_config.json').read_text())
    assert config['format_version'] == 3
    crop_size, pad = np.array(config['crop_size']), np.array(config['pad'])
    rows = list(csv.DictReader((patches/'patch_index.csv').open()))
    candidates = [r for r in rows if r['split']=='train' and 20<=int(r['node_count'])<=40 and int(r['foreground_voxels'])>1500]
    sources, selected, excluded, oracle = {}, [], [], []
    for row in candidates:
        patient, sample = row['patient_id'], row['sample_id']
        if patient not in sources:
            image = nib.load(args.source/'raw'/f'{patient}.nii.gz')
            assert image.header.get_xyzt_units()[0]=='mm'
            affine = image.affine
            sources[patient] = (affine, SourceGraph.from_directory(args.source/'graphs'/patient).transformed(np.linalg.inv(affine)))
        affine, source = sources[patient]
        start = np.array([int(row['start_'+a]) for a in ('d','h','w')])
        bounds = np.column_stack((start, start+crop_size-1))
        sparse = source.crop(bounds)
        stored = pv.read(patches/'train/vtp'/f'{sample}_graph.vtp')
        assert stored.n_points==len(sparse.positions)==int(row['node_count'])
        assert len(sparse.edges)==int(row['edge_count'])
        assert np.allclose(stored.points, (sparse.positions-start+pad)/64, atol=2e-5, rtol=0)
        dense = crop_dense_paths(source, bounds)
        metric = affine[:3, :3] @ np.diag([64.]*3)
        normalized = (dense.positions-start+pad)/64
        graph = contract_degree_two(normalized, dense.edges)
        pairs = [tuple(sorted((b.start,b.end))) for b in graph.branches]
        reason = ('closed_branch' if any(a==b or a<0 for a,b in pairs) else
                  'parallel_branches' if len(set(pairs))!=len(pairs) else
                  'tangent_contact' if dense.tangent_contacts else
                  'capacity_or_empty' if len(graph.anchors)>120 or not pairs else None)
        if reason:
            excluded.append(dict(sample=sample, reason=reason)); continue
        ids = sorted(graph.anchors)
        remap = {i:j for j,i in enumerate(ids)}
        nodes = np.array([graph.anchors[i] for i in ids])
        edges, controls, curves, guides, paths, offsets = [], [], [], [], [], [0]
        for k, branch in enumerate(graph.branches):
            guide = branch.polyline @ metric.T
            a,b = branch.start, branch.end
            if tuple(guide[0])>tuple(guide[-1]):
                guide=guide[::-1].copy(); a,b=b,a
            c,curve = fit_spline(guide, args.controls, 64)
            inverse = np.linalg.inv(metric)
            controls.append(c @ inverse.T); curves.append(curve @ inverse.T)
            guides.append(resample(guide,64) @ inverse.T)
            edges.append((remap[a],remap[b])); paths.append(guide @ inverse.T); offsets.append(offsets[-1]+len(guide))
            dense_fit = fit_spline(guide,args.controls,801)[1]
            forward = point_to_polyline_distance(sample_polyline_by_arclength(dense_fit,.1), guide)
            backward = point_to_polyline_distance(sample_polyline_by_arclength(guide,.1), dense_fit)
            chord = np.linalg.norm(guide[-1]-guide[0])
            oracle.append(dict(sample=sample,branch=k,controls=args.controls,
                               residual_mm=float((forward.mean()+backward.mean())/2),
                               p95_mm=float(np.percentile(np.r_[forward,backward],95)),
                               guide_tortuosity=polyline_length(guide)/chord,
                               fit_tortuosity=polyline_length(dense_fit)/chord))
        raw = patches/'train/raw'/f'{sample}_data.nii.gz'
        seg = patches/'train/seg'/f'{sample}_seg.nii.gz'
        assert nib.load(raw).shape==nib.load(seg).shape==(64,64,64)
        assert int((nib.load(seg).get_fdata()>.5).sum())==int(row['foreground_voxels'])
        for split in ('train','val'):
            leaf = args.output/split
            for directory in ('raw','seg','vtp','spline'):(leaf/directory).mkdir(parents=True,exist_ok=True)
            for src,directory in ((raw,'raw'),(seg,'seg')):
                destination=leaf/directory/src.name
                if not destination.exists():destination.symlink_to(src)
            mesh=pv.PolyData(nodes)
            mesh.lines=np.column_stack((np.full(len(edges),2),np.array(edges))).ravel()
            mesh.save(leaf/'vtp'/f'{sample}_graph.vtp')
            np.savez_compressed(leaf/'spline'/f'{sample}.npz', nodes=nodes.astype('float32'),edges=np.array(edges,dtype='int64'),
                                controls=np.array(controls,dtype='float32'),curves=np.array(curves,dtype='float32'),
                                guides=np.array(guides,dtype='float32'),metric=metric.astype('float32'),
                                full_paths=np.concatenate(paths).astype('float32'),path_offsets=np.array(offsets),
                                boundary=dense.boundary[ids])
        selected.append(dict(sample=sample,source_split='train',source_row=row,source_affine=affine.tolist(),
                             anchors=len(nodes),branches=len(edges),beta0=graph.betti_0,beta1=graph.betti_1,
                             hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (raw,seg,args.source/'graphs'/patient/'graph.vvg')}))
        if len(selected)==args.count:break
    assert len(selected)==args.count, 'Not enough compatible candidates'
    manifest=dict(controls=args.controls,samples=64,source=str(args.source),selection='First train rows with 20–40 stored nodes and >1500 foreground voxels, rejecting unsupported topology before fitting.',
                  monitoring='val is the same four train patches; overfit diagnostic, NOT held-out validation.',
                  guide='Original full generator paths; uncertain image references, no smoothing.',
                  crop_config=config,selected=selected,excluded=excluded)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (args.output/'oracle.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(oracle[0]));writer.writeheader();writer.writerows(oracle)
    print(json.dumps(dict(selected=[r['sample'] for r in selected],excluded=excluded,branches=len(oracle),mean_oracle_residual_mm=np.mean([r['residual_mm'] for r in oracle])),indent=2))


if __name__=='__main__':main()
