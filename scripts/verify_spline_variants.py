#!/usr/bin/env python3
"""Compare fresh decoded splines with uncertain original full-path guides."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics.representation_geometry import (
    point_to_polyline_distance, sample_polyline_by_arclength, tortuosity,
)


def shape_metrics(curve, guide):
    forward = point_to_polyline_distance(sample_polyline_by_arclength(curve, .1), guide)
    backward = point_to_polyline_distance(sample_polyline_by_arclength(guide, .1), curve)
    return {
        'symmetric_guide_distance_mm': float((forward.mean()+backward.mean())/2),
        'tortuosity_absolute_error': float(abs(tortuosity(curve)-tortuosity(guide))),
    }


def verify(run, targets):
    records = json.loads((run/'evaluation_four_patches/predictions.json').read_text())
    branches = []
    for record in records:
        sample = record['source_sample_id']
        with np.load(targets/'val/spline'/f'{sample}.npz') as target:
            metric = target['metric'].astype(float)
            nodes = np.asarray(record['nodes_dhw']) @ metric.T
            gt_nodes = target['nodes'] @ metric.T
            gt_ids, pred_ids = linear_sum_assignment(np.linalg.norm(gt_nodes[:, None]-nodes[None], axis=-1))
            mapping = dict(zip(pred_ids, gt_ids))
            edges = {tuple(sorted(edge)): i for i, edge in enumerate(target['edges'])}
            assert len(nodes) == len(gt_nodes), 'This small diagnostic requires complete node recovery'
            recovered = set()
            for edge, controls in zip(record['edges'], record['spline_controls']):
                identity = tuple(sorted(mapping[node] for node in edge))
                index = edges[identity]
                assert index not in recovered
                recovered.add(index)
                offsets = target['path_offsets']
                guide = target['full_paths'][offsets[index]:offsets[index+1]] @ metric.T
                controls = np.asarray(controls) @ metric.T
                k = len(controls)
                knots = np.r_[np.zeros(4), np.arange(1, k-3)/(k-3), np.ones(4)]
                t = np.linspace(0, 1, 801)
                curve = BSpline(knots, controls, 3)(t)
                chord = np.stack((controls[0], controls[-1]))
                oracle = BSpline(knots, target['controls'][index] @ metric.T, 3)(t)
                branches.append({'sample': sample, 'branch': index,
                                 'prediction': shape_metrics(curve, guide),
                                 'chord': shape_metrics(chord, guide),
                                 'oracle': shape_metrics(oracle, guide)})
            assert len(recovered) == len(edges), 'Missing decoded edges'
    summary = {kind: {key: float(np.mean([row[kind][key] for row in branches]))
                      for key in branches[0][kind]} for kind in ('prediction', 'chord', 'oracle')}
    return {'branch_count': len(branches), 'summary': summary, 'branches': branches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--targets', type=Path, default=Path('experiments/spline_learning/targets_k8'))
    args = parser.parse_args()
    results = {'protocol': 'Four training patches, equal branch means; 801 spline parameter samples, '
               '0.1 mm arc-length sampling, exact point-to-polyline distances. Original paths are '
               'uncertain references. Tortuosity is length divided by endpoint chord length.', 'runs': {}}
    for name in ('v0_pretrained', 'v0_scratch', 'v1_pretrained'):
        result = verify(args.run_root/name, args.targets)
        results['runs'][name] = result
        print(name, json.dumps(result['summary']))
    (args.run_root/'guide_verification.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
