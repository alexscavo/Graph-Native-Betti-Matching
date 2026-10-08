#!/usr/bin/env python3
"""Diagnostic raw node/link decomposition; does not replace benchmark metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True, help='Validation spline-cache directory')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    records = json.loads(a.predictions.read_text())
    result = {'samples': len(records), 'protocol': 'Diagnostic role-agnostic raw-node assignment; no degree-2 contraction. 2mm is a sensitivity check, not a new benchmark threshold.', 'thresholds': {}}
    for threshold in (1., 2.):
        counts = dict(pred_nodes=0, target_nodes=0, matched_nodes=0, target_edges=0,
                      recoverable_edges=0, correct_edges=0, candidate_edges_between_matched_nodes=0)
        for r in records:
            with np.load(a.targets / (r['source_sample_id'] + '.npz')) as gt:
                nodes = np.asarray(r['nodes_dhw']).reshape(-1, 3) @ gt['metric'].T
                truth = gt['nodes'] @ gt['metric'].T
                edges = {tuple(sorted(e)) for e in gt['edges'].tolist()}
            distances = np.linalg.norm(nodes[:, None] - truth[None], axis=-1)
            row, col = linear_sum_assignment(np.where(distances <= threshold, distances, 1e6))
            mapping = {int(i): int(j) for i, j in zip(row, col) if distances[i, j] <= threshold}
            recovered = set(mapping.values())
            possible = {e for e in edges if set(e) <= recovered}
            links = {tuple(sorted((mapping[i], mapping[j]))) for i, j in r['edges'] if i in mapping and j in mapping}
            for key, value in dict(pred_nodes=len(nodes), target_nodes=len(truth), matched_nodes=len(mapping),
                                   target_edges=len(edges), recoverable_edges=len(possible), correct_edges=len(edges & links),
                                   candidate_edges_between_matched_nodes=len(links)).items():
                counts[key] += value
        def ratio(x, y):
            return counts[x] / counts[y] if counts[y] else None
        result['thresholds'][str(threshold)] = {**counts,
            'node_recall': ratio('matched_nodes', 'target_nodes'),
            'edge_recoverability': ratio('recoverable_edges', 'target_edges'),
            'conditional_edge_recall': ratio('correct_edges', 'recoverable_edges'),
            'conditional_edge_precision': ratio('correct_edges', 'candidate_edges_between_matched_nodes')}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
