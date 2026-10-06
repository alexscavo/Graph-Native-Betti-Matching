#!/usr/bin/env python3
"""Check an archive restoration before replacing the working dataset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    patches = args.root/'new_patches_boundary'
    config = patches/'generation_config.json'
    index = patches/'patch_index.csv'
    assert config.read_bytes() == (args.snapshot/'generation_config.json').read_bytes(), 'Generation configuration differs from the saved source copy'
    assert hashlib.sha256(index.read_bytes()).digest() == hashlib.sha256((args.snapshot/'source-patch-index.csv').read_bytes()).digest(), 'Patch index differs from the saved source copy'
    rows = list(csv.DictReader(index.open()))

    def check(row):
        sample, split = row['sample_id'], row['split']
        paths = [patches/split/d/f'{sample}_{suffix}' for d, suffix in
                 (('raw', 'data.nii.gz'), ('seg', 'seg.nii.gz'), ('vtp', 'graph.vtp'))]
        return [str(p.relative_to(args.root)) for p in paths if not p.is_file() or p.stat().st_size == 0]

    with ThreadPoolExecutor(max_workers=8) as pool:
        missing = [name for names in pool.map(check, rows) for name in names]
    assert not missing, f'Missing/empty prepared files: {len(missing)}; first examples: {missing[:10]}'
    subjects = {r['patient_id'] for r in rows}
    affines = {}
    for patient in sorted(subjects, key=int):
        image = nib.load(args.root/'seg'/f'{patient}.nii.gz')
        assert image.header.get_xyzt_units()[0] == 'mm', patient
        assert np.isfinite(image.affine).all() and abs(np.linalg.det(image.affine[:3, :3])) > 0, patient
        affines[patient] = image.affine.tolist()
    known = json.loads((args.snapshot/'verified-pilot-affines.json').read_text())
    for patient, metadata in known.items():
        assert np.allclose(affines[patient], metadata['affine'], atol=1e-8, rtol=0), patient
    # Read one actual MRI/mask pair per split. The complete inventory above
    # checks existence and size, not every voxel payload.
    for split in ('train', 'val', 'test'):
        row = next(r for r in rows if r['split'] == split)
        for directory, suffix in (('raw', 'data'), ('seg', 'seg')):
            image = nib.load(patches/split/directory/f"{row['sample_id']}_{suffix}.nii.gz")
            data = np.asarray(image.dataobj)
            assert data.shape == (64, 64, 64) and np.isfinite(data).all(), row['sample_id']
    complete_graphs = sorted([p for p in subjects if all((args.root/'graphs'/p/name).is_file()
                              for name in ('nodes.csv', 'edges.csv', 'graph.vvg'))], key=int)
    report = {'archive': str(args.archive), 'status': 'verified',
              'patch_counts': {s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')},
              'checked_patch_files': len(rows)*3, 'segmentation_headers': len(affines),
              'complete_graph_subjects': complete_graphs, 'full_volume_affines': affines,
              'checks': 'Exact saved config/index match, all indexed triplets present/nonempty, all segmentation headers valid, pilot affines agree, one MRI/mask payload per split readable.'}
    (args.root/'restore-verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'full_volume_affines'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
