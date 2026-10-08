#!/usr/bin/env python3
"""Build auditable tables from completed held-out summaries; no new evaluation."""
import argparse,csv,hashlib,json
from pathlib import Path

DEFAULT_ROOT=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning')
ARMS=[('old_graph','Straight edges'),('spline_k8','B-spline, 8 controls'),('adaptive_degree2','Adaptive degree-2')]
COMMON=[('common_anchor_f1','Anchor F1 ↑'),('common_branch_f1','Branch F1 ↑'),('common_branch_coverage','Branch coverage ↑'),
        ('common_curve_precision','Whole-curve precision ↑'),('common_curve_recall','Whole-curve recall ↑'),('common_curve_f1','Whole-curve F1 ↑'),
        ('common_guide_distance_mm','Matched guide distance, mm ↓'),('common_tortuosity_error','Matched tortuosity error ↓'),
        ('common_betti0_abs_error_mean','Absolute component-count error (β₀) ↓'),('common_betti1_abs_error_mean','Absolute loop-count error (β₁) ↓'),
        ('common_anchor_count_abs_error_mean','Topological-node count error ↓')]
COUNTS=[('native_predicted_nodes_mean','Raw predicted nodes'),('native_predicted_degree2_mean','Predicted degree-2 nodes'),
        ('common_predicted_anchors_mean','Topological nodes (includes isolates)'),('common_predicted_junctions_mean','Junctions'),
        ('common_predicted_endpoints_mean','Endpoints (includes crop boundaries)'),('common_predicted_isolates_mean','Isolated nodes'),
        ('common_predicted_branches_mean','Contracted branches'),('representation_internal_controls_mean','Internal spline-control points')]


def display(key,value):
    if value is None:return 'N/A'
    if key=='common_branch_coverage':return f'{100*value:.3f}%'
    if 'f1' in key or 'precision' in key or 'recall' in key:return f'{value:.6f}'
    return f'{value:.5f}'


def table(metrics,rows,source='common'):
    lines=['| Metric | Straight edges | B-spline, 8 controls | Adaptive degree-2 |','|---|---:|---:|---:|']
    for key,label in metrics:
        lines.append('| '+label+' | '+' | '.join(display(key,row[source].get(key)) for row in rows)+' |')
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--report',type=Path,default=Path('../graph-representation/docs/scratch-comparison-results.md'))
    parser.add_argument('--output',type=Path,default=Path('experiments/spline_learning/metadata/comparison'))
    args=parser.parse_args();shared_path=args.root/'scratch_comparison_10k_300ep/comparison-manifest.json'
    shared=json.loads(shared_path.read_text());assert shared['counts']=={'train':10000,'val':1952,'test':13071}
    rows=[];provenance={'shared_manifest':str(shared_path),'shared_manifest_sha256':hashlib.sha256(shared_path.read_bytes()).hexdigest(),'sources':[]}
    for arm,label in ARMS:
        run=args.root/'runs'/f'scratch_{arm}_10k_300ep'
        status=json.loads((run/'training-status.json').read_text());assert status['epoch']==300 and status['iteration']==750000
        history=[json.loads(line) for line in (run/'validation-losses.jsonl').read_text().splitlines()]
        selected=min(history,key=lambda r:r['checkpoint_total'])['epoch']
        for selection in ('best','final'):
            path=run/'test'/selection;files=[path/'summary.json',path/'common-v2-summary.json']
            native,common=[json.loads(f.read_text()) for f in files]
            assert native['samples']==common['samples']==13071
            assert common['protocol_version']=='shared-guides-v2'
            assert (common['anchor_threshold_mm'],common['curve_threshold_mm'],common['sample_spacing_mm'])==(1.,1.,.25)
            with (path/'common-v2-per-patch.csv').open() as f:ids=sorted(r['sample_id'] for r in csv.DictReader(f))
            assert ids==shared['test_ids'],(arm,selection,'test population mismatch')
            rows.append({'arm':arm,'label':label,'selection':selection,'epoch':selected if selection=='best' else 300,'native':native,'common':common})
            provenance['sources'].extend({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files)
    args.output.mkdir(parents=True,exist_ok=True);flat=[]
    for row in rows:
        native=row['native'];common=row['common']
        flat.append({'representation':row['label'],'selection':row['selection'],'checkpoint_epoch':row['epoch'],'test_patches':common['samples'],
                     **{key:common.get(key) for key,_ in COMMON+COUNTS},'native_node_f1':native['node_f1'],'native_edge_f1':native['edge_f1'],
                     'native_target_nodes':native['target_nodes'],'native_target_edges':native['target_edges']})
    with (args.output/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    parts=['# Three scratch trainings: completed held-out comparison',
           'All three train from scratch for 300 epochs on the same 10,000 vessel patches, with 1,952 validation and 13,071 held-out test patches. All use 120 object queries, batch 4, the same seed and shared optimizer/schedule. The spline arm adds its head and fixed geometry loss weight 1.4739989612791453.',
           'The source guide paths and masks remain uncertain references; agreement with them does not establish anatomical accuracy.',
           '## Checkpoint selection',
           '| Representation | Validation-loss-selected epoch | Final epoch |\n|---|---:|---:|\n'+'\n'.join(f"| {r['label']} | {r['epoch']} | 300 |" for r in rows if r['selection']=='best'),
           '“Best” means the lowest weighted validation graph loss (boxes/class/cardinality/nodes/edges, weights 3/4/0.8/2/6), not highest reconstruction F1. No checkpoint or threshold was selected using test scores. Final and selected checkpoints are both reported; these measurements do not justify retrospective test-based model selection.',
           '## Shared evaluation protocol',
           'Degree-two chains are contracted while retaining their full predicted geometry. All arms use the same target endpoint topology and full guides, physical matching at 1 mm, and curve sampling every 0.25 mm. Anchor matching is role-aware (junction, endpoint, isolate); virtual pure-cycle anchors are used only for topology. Whole-curve precision/recall/F1 include missing and extra geometry. Matched guide distance and tortuosity apply only to recovered branches and must be read with coverage. Dataset-level anchor/branch F1 are micro counts; curve precision/recall use pooled arc-length samples. Count/error entries are patch means.',
           '## Final checkpoints, epoch 300',table(COMMON,[r for r in rows if r['selection']=='final']),
           '## Validation-loss-selected checkpoints',table(COMMON,[r for r in rows if r['selection']=='best']),
           '## Predicted object and geometry counts: final checkpoints',table(COUNTS,[r for r in rows if r['selection']=='final']),
           'Shared reference means: 21.39255 topological anchors, 5.65710 junctions, 15.70056 endpoints and 16.37189 branches per patch. Spline control points are geometry parameters, not object tokens or anatomical junctions. A predicted degree-two node is contracted regardless of the intended native node role; false degree-two structure remains visible in raw counts and native diagnostics. Count agreement alone does not imply correct connectivity.',
           '## Native diagnostics: final checkpoints',
           table([('node_f1','Native node F1 ↑'),('edge_f1','Native edge F1 ↑'),('target_nodes','Native target nodes / patch'),('target_edges','Native target edges / patch')],[r for r in rows if r['selection']=='final'],source='native'),
           '## Native diagnostics: validation-loss-selected checkpoints',
           table([('node_f1','Native node F1 ↑'),('edge_f1','Native edge F1 ↑')],[r for r in rows if r['selection']=='best'],source='native'),
           'Native node/edge metrics use each arm’s own targets. The adaptive arm has extra degree-two target nodes/edges; native scores are therefore diagnostics, not the representation-independent headline comparison.',
           '## Interpretation and limitations',
           '- All three current runs show weak physical topology recovery under the fixed 1 mm protocol. Small conditional shape errors describe very few recovered branches.',
           '- The final adaptive model has higher whole-curve F1 (0.24711 versus 0.23051 for straight edges), while straight edges have higher anchor and Branch F1. No single representation wins every relevant quantity, and no statistical-significance claim is made.',
           '- The current spline run is degenerate: node/branch counts are constant across all 13,071 patches and no branches are recovered. Saved prediction checks across two distinct test subjects showed identical edge indices and coordinate differences of only approximately 7.6e-6 mm. Coordinates are not bit-identical. This is a warning about this training/evaluation outcome, not proof that spline representation learning is infeasible.',
           '- Spline guide/tortuosity errors are N/A because zero branches match, not because the geometry is perfect. Its small loop-count error is also trivial: predicting essentially no cycles does not imply correct vessels.',
           '- Final checkpoints outperform loss-selected checkpoints on several measures. The current validation-loss selection is poorly aligned with reconstruction metrics; a future validation-only topology/geometry selection protocol must be predefined.',
           '- This is a single-seed comparison using correlated patches from 14 test subjects and the common eligible subset. Subject-level uncertainty estimates and repeated seeds remain necessary for paper-level claims.',
           '## Next steps',
           'Diagnose the near-input-independent spline predictions before launching another large spline run or deciding between spline families. Check the input dependence, shared-head gradients, and scratch-versus-finetuning geometry-loss calibration. Keep these current results as the reproducible baseline. Validate any revised checkpoint-selection policy on validation subjects only, then repeat controlled runs and common held-out scoring.',
           '## Files and reproducibility',
           f'Full-precision CSV: `{args.output.resolve()/"comparison.csv"}`. Source checksums: `{args.output.resolve()/"provenance.json"}`.',
           f'Underlying results: `{args.root}/runs/scratch_{{old_graph,spline_k8,adaptive_degree2}}_10k_300ep/test/{{best,final}}/`.',
           'Regenerate from the representation repository: `python scripts/build_scratch_comparison_tables.py`. This reads existing results and changes no dataset, checkpoint or metric source files.']
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text('\n\n'.join(parts)+'\n')
    print('Created',args.report.resolve());print('Created',args.output.resolve()/'comparison.csv')

if __name__=='__main__':main()
