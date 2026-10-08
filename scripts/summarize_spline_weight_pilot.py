#!/usr/bin/env python3
"""Verify completed pairing and summarize the bounded spline-weight trial."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arms', nargs=2, default=['original', 'reduced'])
    args = p.parse_args()
    runs = {arm: json.loads((args.root / arm / 'summary.json').read_text()) for arm in args.arms}
    a, b = (runs[k]['provenance'] for k in args.arms)
    for key in ('initial_state_sha256', 'batch_order_sha256', 'base_config_sha256', 'probe_ids',
                'updates', 'schedule_updates', 'warmup_updates', 'bn_recalibration_batches'):
        assert a[key] == b[key], (key, a[key], b[key])
    for key in ('initial_checkpoint', 'initial_epoch', 'schedule_start_update', 'spline_ramp_updates'):
        assert a.get(key) == b.get(key), key
    assert a['updates'] == 5000 and a['weight'] != b['weight']
    if args.arms == ['original', 'reduced']:
        assert abs(a['weight'] / b['weight'] - 10) < 1e-12
    membership = [(args.root / arm / 'dataset-manifest.json').read_bytes() for arm in runs]
    assert membership[0] == membership[1], 'Different patch membership'
    rows = []
    for arm, run in runs.items():
        f = run['final']; assert f['update'] == 5000
        rows.append({'arm': arm, 'spline_weight': run['provenance']['weight'],
                     'updates': f['update'], 'train_node_loss': f['training']['nodes'],
                     'train_edge_loss': f['training']['edges'],
                     'train_spline_curve_mm': f['training']['spline_curve_mm'],
                     'validation_node_loss': f['validation']['nodes'],
                     'validation_spline_curve_mm': f['validation']['spline_curve_mm'],
                     **f['sensitivity'], **f['geometry']})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    provenance = {'verified_pairing': True, 'membership_sha256': hashlib.sha256(membership[0]).hexdigest(),
                  'shared_initial_state_sha256': a['initial_state_sha256'],
                  'shared_batch_order_sha256': a['batch_order_sha256'],
                  'source_root': str(args.root), 'results': rows,
                  'scope': '128 validation patches; no held-out test evaluation; one seed and 512 training patches.'}
    (args.output / 'comparison.json').write_text(json.dumps(provenance, indent=2) + '\n')
    fields = [('train_node_loss', 'Training node loss'), ('validation_node_loss', 'Validation node loss'),
              ('validation_spline_curve_mm', 'Matched-positive curve error (mm)'),
              ('native_node_f1', 'Native node F1'), ('native_edge_f1', 'Native edge F1'),
              ('common_anchor_f1', 'Common anchor F1'), ('common_branch_f1', 'Common branch F1'),
              ('common_curve_f1', 'Common curve F1'), ('node_pair_max_delta', 'Probe node input sensitivity'),
              ('transformer_memory_pair_mae', 'Probe transformer input sensitivity')]
    if all('same_endpoint_chord_common_curve_f1' in row for row in rows):
        fields.append(('same_endpoint_chord_common_curve_f1', 'Same-endpoint chord curve F1'))
    lines = [f"| Metric | {args.arms[0]} (weight {a['weight']:.4g}) | {args.arms[1]} (weight {b['weight']:.4g}) |", '| --- | ---: | ---: |']
    for key, label in fields:
        lines.append('| ' + label + ' | ' + ' | '.join(f'{r[key]:.6g}' if r[key] is not None else 'N/A' for r in rows) + ' |')
    (args.output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('Verified matching initialization, dataset membership, schedule, and training-batch order.')


if __name__ == '__main__':
    main()
