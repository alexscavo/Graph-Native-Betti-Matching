#!/usr/bin/env python3
"""Build a separate Catmull–Rom cache from the unchanged paired full guides."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from data.spline_targets import resample
from scripts.catmull_rom import catmull_rom


def convert_target(cache, controls=8, samples=64):
    result = {key: cache[key].copy() for key in cache.files}
    metric = result['metric'].astype(float)
    inverse = np.linalg.inv(metric)
    fitted, curves = [], []
    for i, edge in enumerate(result['edges']):
        path = result['full_paths'][result['path_offsets'][i]:result['path_offsets'][i+1]] @ metric.T
        points = resample(path, controls) @ inverse.T
        # Preserve graph endpoints exactly, including existing boundary rules.
        points[[0, -1]] = result['nodes'][edge]
        points = points.astype('float32')
        curve = (catmull_rom(points.astype(float) @ metric.T, np.linspace(0, 1, samples)) @ inverse.T).astype('float32')
        curve[[0, -1]] = points[[0, -1]]
        fitted.append(points); curves.append(curve)
    result['controls'] = np.asarray(fitted, dtype='float32').reshape(-1, controls, 3)
    result['curves'] = np.asarray(curves, dtype='float32').reshape(-1, samples, 3)
    result['family'] = np.asarray('catmull_rom')
    return result


def convert_one(job):
    source, destination = map(Path, job)
    with np.load(source, allow_pickle=False) as cache:
        result = convert_target(cache)
        assert np.array_equal(result['nodes'], cache['nodes'])
        assert np.array_equal(result['edges'], cache['edges'])
        assert np.array_equal(result['full_paths'], cache['full_paths'])
        assert np.isfinite(result['controls']).all() and np.isfinite(result['curves']).all()
        # Keep a small representation-capacity summary; no model evaluation.
        delta = (result['curves']-result['guides']) @ result['metric'].T
        error = np.linalg.norm(delta, axis=-1)
    with destination.open('xb') as stream:
        np.savez_compressed(stream, **result)
    return float(error.sum()), int(error.size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    manifest_path = args.comparison/'comparison-manifest.json'
    original = json.loads(manifest_path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(family='catmull_rom', controls=8, samples=64, alpha=.5,
                  target_construction='Equal physical arc-length samples of original full guide; reflected endpoint ghosts.',
                  comparison_manifest=str(manifest_path),
                  comparison_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  boundary_policy=original['boundary_policy'], counts={}, parameterwise_guide_distance_mm={})
    (args.output/'patch_index.csv').symlink_to(args.comparison/'spline/patch_index.csv')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split, key in [('train','train_ids'), ('val','validation_ids'), ('test','test_ids')]:
            leaf = args.output/split
            (leaf/'spline').mkdir(parents=True)
            for directory in ('raw','seg','vtp'):
                (leaf/directory).symlink_to(args.comparison/'spline'/split/directory, target_is_directory=True)
            ids = original[key]
            jobs = [(str(args.comparison/'spline'/split/'spline'/f'{sample}.npz'), str(leaf/'spline'/f'{sample}.npz')) for sample in ids]
            total, count = 0., 0
            for i, (error, n) in enumerate(pool.map(convert_one, jobs, chunksize=16), 1):
                total += error; count += n
                if i % 1000 == 0:
                    print(f'{split}: {i}/{len(ids)}', flush=True)
            result['counts'][split] = len(ids)
            result['parameterwise_guide_distance_mm'][split] = total/max(count, 1)
    assert result['counts'] == original['counts']
    (args.output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
