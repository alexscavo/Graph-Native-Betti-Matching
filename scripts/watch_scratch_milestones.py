#!/usr/bin/env python3
"""Collate existing epoch-100/200 diagnostics without interrupting GPU training."""
import argparse
import base64
import csv
import html
import io
import json
from pathlib import Path
import time


ARMS = ('graph_control', 'detached_spline')


def history(path):
    if not path.exists(): return []
    rows = []
    for line in path.read_text().splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass  # Training may still be appending its last line.
    return rows


def compare(rows, baselines):
    result = []
    for arm in ARMS:
        row = rows[arm]; geo = row['geometry']; sensitivity = row['sensitivity']
        current = sensitivity['transformer_memory_pair_mae']
        ratio = current / max(baselines[arm]['sensitivity']['transformer_memory_pair_mae'], 1e-12)
        # Screening threshold against the previously collapsed run (~3e-6).
        # A positive flag requires scientific review; it is not a mechanism diagnosis.
        flag = current < 1e-3 and ratio < .01
        entry = dict(arm=arm, epoch=row['epoch'], node_f1=geo['native_node_f1'],
            edge_f1=geo['native_edge_f1'], branch_f1=geo['common_branch_f1'],
            curve_f1=geo['common_curve_f1'], chord_curve_f1=geo['same_endpoint_chord_common_curve_f1'],
            memory_sensitivity=current, sensitivity_relative_to_epoch10=ratio,
            token_sensitivity=sensitivity['token_pair_mae'],
            node_sensitivity=sensitivity['node_pair_max_delta'], collapse_screen_flag=flag)
        for split in ('train', 'val'):
            matched = row[split+'_matched_geometry']['all_pairs']
            for key in ('branches', 'guide_curve_mm', 'guide_chord_mm',
                        'endpoint_neutral_guide_curve_mm', 'endpoint_neutral_guide_relative_improvement'):
                entry[split+'_'+key] = matched[key]
            entry[split+'_curvature_groups'] = row[split+'_matched_geometry'].get('curvature_groups', {})
        entry['train_branch_f1'] = row['train_geometry']['common_branch_f1']
        entry['train_curve_f1'] = row['train_geometry']['common_curve_f1']
        result.append(entry)
    return result


def atomic(path, text):
    temp = path.with_suffix(path.suffix+'.tmp'); temp.write_text(text); temp.replace(path)


def report(root, epoch):
    stage = 'epoch150' if epoch <= 150 else 'epoch300'
    sources = {arm: root/arm/stage for arm in ARMS}
    rows = {}; baselines = {}
    for arm in ARMS:
        initial = history(root/arm/'epoch150/history.jsonl')
        candidates = [r for r in history(sources[arm]/'history.jsonl') if r['update'] == epoch*2500]
        if not initial or not candidates: return False
        row = candidates[-1]
        if not all(k in row for k in ('train_matched_geometry','val_matched_geometry','train_geometry')):
            return False
        rows[arm] = row; baselines[arm] = initial[0]
    for filename in ('dataset-manifest.json', 'evaluation-sample-ids.json'):
        assert json.loads((sources[ARMS[0]]/filename).read_text()) == json.loads((sources[ARMS[1]]/filename).read_text())
    comparisons = compare(rows, baselines)
    destination = root/'milestone_checks'/f'epoch{epoch}'
    destination.mkdir(parents=True, exist_ok=True)
    provenance = dict(epoch=epoch, updates=epoch*2500, train_monitor_samples=512,
        validation_samples=128, paired_membership_verified=True,
        source_directories={a:str(p) for a,p in sources.items()},
        collapse_screen='memory MAE < 0.001 AND < 1% of own epoch-10 value; heuristic requiring review',
        checkpoint_selection='exact epoch, independent of best-checkpoint selection',
        tests='existing decoded graph/chord evaluation, train/val GT-matched geometry, fixed image-sensitivity probe',
        bn='saved statistics; no recalibration',
        ground_truth='uncertain centerline guides; matched geometry separate from decoded graph metrics')
    atomic(destination/'comparison.json',json.dumps(dict(provenance=provenance, comparisons=comparisons,
                                                       source_rows=rows),indent=2)+'\n')
    columns = ['arm','node_f1','edge_f1','branch_f1','curve_f1','chord_curve_f1',
               'memory_sensitivity','val_endpoint_neutral_guide_relative_improvement','collapse_screen_flag']
    buffer = io.StringIO(); writer = csv.DictWriter(buffer,fieldnames=columns)
    writer.writeheader(); writer.writerows([{k:r[k] for k in columns} for r in comparisons])
    atomic(destination/'comparison.csv',buffer.getvalue())
    table = '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    for row in comparisons:
        table += '| '+' | '.join(f'{row[k]:.6g}' if isinstance(row[k],float) else str(row[k]) for k in columns)+' |\n'
    flags = [r['arm'] for r in comparisons if r['collapse_screen_flag']]
    conclusion = ('LOW IMAGE SENSITIVITY FLAG: '+', '.join(flags)+'. Inspect detection and neighboring checkpoints.'
                  if flags else 'Neither run triggers the image-sensitivity collapse screen at this epoch.')
    note = ('512 fixed training / 128 fixed validation cases; single seed. Shape gains compare curves with '
            'identical-endpoint chords. Collapse screening is a heuristic; poor graph metrics alone do not establish collapse. '
            'Train/validation geometry and curvature strata are included in comparison.json. No automatic training cancellation.')
    atomic(destination/'comparison.md',f'# Epoch {epoch} paired diagnostic\n\n{conclusion}\n\n{note}\n\n'+table)
    panels = []
    for arm in ARMS:
        pictures = []
        for image in sorted((sources[arm]/f'validation-{epoch*2500:06d}').glob('*.png')):
            uri = 'data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()
            pictures.append(f'<p>{html.escape(image.stem)}</p><img src="{uri}">')
        panels.append('<section><h2>'+arm+'</h2>'+''.join(pictures)+'</section>')
    document = ('<!doctype html><meta charset="utf-8"><title>Scratch epoch '+str(epoch)+'</title>'
        '<style>body{font:15px system-ui;margin:24px}main{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:100%}pre{overflow:auto}</style>'
        f'<h1>Epoch {epoch}</h1><p>{html.escape(conclusion)}</p><p>{html.escape(note)}</p>'
        '<pre>'+html.escape(table)+'</pre><main>'+''.join(panels)+'</main>')
    atomic(destination/'comparison.html',document)
    print(json.dumps(dict(epoch=epoch,report=str(destination),collapse_flags=flags)),flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/scratch_extend300_20261012'))
    parser.add_argument('--epochs',type=int,nargs='+',default=[100,200])
    parser.add_argument('--interval',type=int,default=300)
    parser.add_argument('--once',action='store_true')
    args = parser.parse_args()
    pending = set(args.epochs)
    deadline = time.monotonic()+48*3600
    while pending:
        for epoch in sorted(pending.copy()):
            if report(args.root,epoch): pending.remove(epoch)
        if not pending or args.once: return
        if time.monotonic() > deadline: raise SystemExit('48-hour deadline: milestones not reached; inspect Slurm jobs.')
        time.sleep(args.interval)


if __name__ == '__main__': main()
