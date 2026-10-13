#!/usr/bin/env python3
"""Saved-prediction diagnostic: endpoint recovery, relation distance, fragmentation.

No checkpoint inference or benchmark changes. Labels use positional node matching,
never predicted degree. Candidate-pair rates condition on both nodes being matched.
"""
import argparse
from collections import deque
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_vessel_core import ARMS, RUN_ROOT, records, reference
from metrics.vessel_graph import VesselGraph
from metrics.representation_geometry import (
    polyline_length, sample_polyline_by_arclength, point_to_polyline_distance,
)


def match_nodes(prediction, truth, tolerance):
    distances = np.linalg.norm(prediction[:, None] - truth[None], axis=-1)
    penalty = (min(distances.shape) + 1) * (tolerance + 1)
    rows, cols = linear_sum_assignment(np.where(distances <= tolerance, distances, penalty))
    return {int(i): int(j) for i, j in zip(rows, cols) if distances[i, j] <= tolerance}


def route(adjacency, start, end, forbidden=()):
    parent = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            nodes, edges = [end], []
            while parent[node] is not None:
                node, edge = parent[node]
                nodes.append(node); edges.append(edge)
            return nodes[::-1], edges[::-1]
        for other, edge in adjacency[node]:
            if other in parent or (other != end and other in forbidden):
                continue
            parent[other] = node, edge
            queue.append(other)
    return None


def aligned_path(prediction, path, guide, tolerance):
    # One deterministic shortest-hop path is checked, not an exhaustive path search.
    nodes, edges = path
    pieces = []
    for start, edge in zip(nodes, edges):
        curve = prediction.curves_mm[edge]
        pieces.append(curve if prediction.edges[edge, 0] == start else curve[::-1])
    curve = np.concatenate(pieces)
    a = point_to_polyline_distance(sample_polyline_by_arclength(curve, .5), guide)
    b = point_to_polyline_distance(sample_polyline_by_arclength(guide, .5), curve)
    return bool(max(np.percentile(a, 95), np.percentile(b, 95)) <= tolerance)


def bin_index(value, cuts):
    return int(np.searchsorted(cuts, value, side='right'))


def diagnose(prediction, truth, tolerance, arc_cuts, chord_cuts):
    mapping = match_nodes(prediction.nodes_mm, truth.nodes_mm, tolerance)
    inverse = {j: i for i, j in mapping.items()}
    gt_links = {tuple(sorted(map(int, edge))) for edge in truth.edges}
    pred_links = {tuple(sorted(map(int, edge))) for edge in prediction.edges}
    adjacency = [[] for _ in prediction.nodes_mm]
    for edge, (a, b) in enumerate(prediction.edges):
        adjacency[a].append((int(b), edge)); adjacency[b].append((int(a), edge))
    # A chain may not silently pass another matched GT anchor.
    forbidden = {i for i, links in enumerate(adjacency) if len(links) != 2} | set(mapping)
    edge_rows = []
    for index, ((a, b), guide) in enumerate(zip(truth.edges, truth.curves_mm)):
        length = polyline_length(guide)
        chord = float(np.linalg.norm(truth.nodes_mm[a] - truth.nodes_mm[b]))
        eligible = int(a) in inverse and int(b) in inverse
        direct = eligible and tuple(sorted((inverse[a], inverse[b]))) in pred_links
        multi = chain = aligned = False
        # Closed/parallel GT branches are counted separately: binary adjacency
        # cannot establish their multiplicity. They are excluded from this audit.
        pair = tuple(sorted((int(a), int(b))))
        multiplicity = sum(tuple(sorted(map(int, e))) == pair for e in truth.edges)
        supported = a != b and multiplicity == 1
        if supported and eligible and not direct:
            path = route(adjacency, inverse[a], inverse[b])
            multi = path is not None
            chain_path = route(adjacency, inverse[a], inverse[b], forbidden)
            chain = chain_path is not None
            if multi:
                aligned = aligned_path(prediction, chain_path or path, guide, tolerance)
        edge_rows.append(dict(target_edge=index, arc_mm=length, chord_mm=chord,
            arc_bin=bin_index(length, arc_cuts), chord_bin=bin_index(chord, chord_cuts),
            supported=bool(supported), endpoints_recovered=bool(eligible), direct=bool(direct),
            indirect_path=multi, degree2_chain=chain, indirect_path_guide_aligned=aligned))
    pairs = [dict(positive_candidates=0, positive_selected=0,
                  negative_candidates=0, negative_selected=0) for _ in range(4)]
    matched = sorted(mapping)
    # Pair distance uses GT node positions, so model localization errors cannot
    # move the same anatomical pair between distance bins.
    for offset, i in enumerate(matched):
        for j in matched[offset + 1:]:
            a, b = mapping[i], mapping[j]
            bucket = pairs[bin_index(np.linalg.norm(truth.nodes_mm[a] - truth.nodes_mm[b]), chord_cuts)]
            label = 'positive' if tuple(sorted((a, b))) in gt_links else 'negative'
            bucket[label + '_candidates'] += 1
            bucket[label + '_selected'] += int((i, j) in pred_links)
    counts = dict(predicted_nodes=len(prediction.nodes_mm), target_nodes=len(truth.nodes_mm),
        matched_nodes=len(mapping), predicted_edges=len(prediction.edges), target_edges=len(truth.edges),
        predicted_edges_with_unmatched_endpoint=sum(a not in mapping or b not in mapping for a, b in prediction.edges),
        excluded_closed_or_parallel_targets=sum(not r['supported'] for r in edge_rows))
    return counts, edge_rows, pairs


def ratio(a, b):
    return a / b if b else None


def summarize(counts, rows, pairs):
    eligible = [r for r in rows if r['supported'] and r['endpoints_recovered']]
    supported = [r for r in rows if r['supported']]
    direct = sum(r['direct'] for r in eligible)
    out = dict(counts, supported_target_edges=len(supported), recoverable_edges=len(eligible),
        correct_direct_edges=direct, missing_direct_edges=len(eligible) - direct,
        node_recall=ratio(counts['matched_nodes'], counts['target_nodes']),
        node_precision=ratio(counts['matched_nodes'], counts['predicted_nodes']),
        endpoint_pair_recovery=ratio(len(eligible), len(supported)),
        conditional_direct_recall=ratio(direct, len(eligible)))
    for key in ('indirect_path', 'degree2_chain', 'indirect_path_guide_aligned'):
        out[key] = sum(r[key] for r in eligible)
    for key in ('arc_bin', 'chord_bin'):
        out[key] = []
        for index in range(4):
            population = [r for r in supported if r[key] == index]
            found = [r for r in population if r['endpoints_recovered']]
            hit = sum(r['direct'] for r in found)
            out[key].append(dict(bin=index, targets=len(population), recoverable=len(found), direct=hit,
                endpoint_pair_recovery=ratio(len(found), len(population)),
                conditional_direct_recall=ratio(hit, len(found))))
    out['candidate_pairs_by_gt_chord'] = [dict(p,
        true_positive_rate=ratio(p['positive_selected'], p['positive_candidates']),
        false_positive_rate=ratio(p['negative_selected'], p['negative_candidates']),
        precision=ratio(p['positive_selected'], p['positive_selected'] + p['negative_selected'])) for p in pairs]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=RUN_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-samples', type=int)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error('--max-samples must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    prior = {arm: json.loads((args.run_root/'full_validation'/source/'summary.json').read_text())
             for arm, source in ARMS.items()}
    first = next(iter(prior.values()))['provenance']
    for value in prior.values():
        assert value['provenance']['sample_ids'] == first['sample_ids']
        assert value['provenance']['dataset_root'] == first['dataset_root']
    ids = first['sample_ids'][:args.max_samples] if args.max_samples else first['sample_ids']
    assert len(ids) == len(set(ids))
    target_root = Path(first['dataset_root'])/'val/spline'
    arc, chord = [], []
    for name in ids:
        gt, _ = reference(target_root/(name+'.npz'))
        arc.extend(map(polyline_length, gt.curves_mm))
        chord.extend(np.linalg.norm(gt.nodes_mm[gt.edges[:, 0]] - gt.nodes_mm[gt.edges[:, 1]], axis=1))
    arc_cuts = np.quantile(arc, [.25, .5, .75]).tolist()
    chord_cuts = np.quantile(chord, [.25, .5, .75]).tolist()
    result = dict(protocol='branch-relations-v1', samples=len(ids),
        sample_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        gt_arc_quartile_cuts_mm=arc_cuts, gt_chord_quartile_cuts_mm=chord_cuts,
        matching_tolerances_mm=[1., 2.], inference_performed=False,
        dataset_root=str(target_root), source_run=str(args.run_root),
        checkpoints={arm: value['provenance']['checkpoint'] for arm, value in prior.items()}, models={},
        notes=['Role-agnostic maximum-cardinality positional matching.',
               'Conditional pair rates include only pairs of matched nodes; unmatched predictions remain explicitly counted.',
               'Closed/parallel GT branches excluded from direct-branch recall; candidate binary adjacency is still evaluated.',
               'Indirect guide alignment checks one deterministic shortest-hop route; it is descriptive, not anatomical proof.',
               'Guide alignment: bidirectional sampled HD95 <= matching tolerance, spacing 0.5mm.',
               'Quartiles derive solely from common GT; sensitivity/causality and patch correlation are not controlled.'])
    started = time.monotonic()
    for arm, source in ARMS.items():
        processed = 0
        all_rows = {t: [] for t in (1., 2.)}
        totals = {t: {} for t in all_rows}
        pair_totals = {t: [dict(positive_candidates=0, positive_selected=0, negative_candidates=0, negative_selected=0)
                           for _ in range(4)] for t in all_rows}
        for n, record in enumerate(records(args.run_root/'full_validation'/source/'validation-000000/predictions.json')):
            if n >= len(ids): break
            assert record['source_sample_id'] == ids[n]
            processed += 1
            gt, metric = reference(target_root/(ids[n]+'.npz'))
            predicted = VesselGraph.from_prediction(record, metric, curved=arm!='straight_final')
            for tolerance in all_rows:
                counts, rows, pairs = diagnose(predicted, gt, tolerance, arc_cuts, chord_cuts)
                for key, value in counts.items(): totals[tolerance][key] = totals[tolerance].get(key, 0) + value
                for i, p in enumerate(pairs):
                    for key, value in p.items(): pair_totals[tolerance][i][key] += value
                all_rows[tolerance].extend(dict(sample_id=ids[n], **r) for r in rows)
            if (n+1) % 200 == 0:
                print(json.dumps(dict(arm=arm, completed=n+1, elapsed_seconds=time.monotonic()-started)), flush=True)
        assert processed == len(ids)
        result['models'][arm] = {str(t): summarize(totals[t], all_rows[t], pair_totals[t]) for t in all_rows}
        for t, rows in all_rows.items():
            with (args.output/f'{arm}-{t:g}mm-branches.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
                writer.writeheader(); writer.writerows(rows)
    result['elapsed_seconds'] = time.monotonic()-started
    (args.output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
