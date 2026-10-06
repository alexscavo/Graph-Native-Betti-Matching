#!/usr/bin/env python3
"""Record actual four-patch results and the explicit V1 learning gate."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--gate',action='store_true')
    args=p.parse_args()
    summaries={path.parent.name:json.loads(path.read_text()) for path in sorted(args.run_root.glob('v*/summary.json'))}
    passes={name: r['final']['spline_curve_mm']<=.5 and r['final']['spline_curve_mm']<=.5*r['initial']['spline_curve_mm'] for name,r in summaries.items() if name.startswith('v0_')}
    choice=('pretrained' if passes.get('v0_pretrained') else 'scratch' if passes.get('v0_scratch') else 'none')
    if args.gate:print(choice);return
    output=dict(runs=summaries,v0_learning_gate=passes,v1_initialization=choice,
                gate='Final mean corresponding fitted-curve point error <=0.5 mm AND <=50% of initial value; four train patches only.',
                interpretation='Matched-positive geometry diagnostics; decoded graph views must be inspected separately. No held-out performance claim.')
    (args.run_root/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    project=Path(__file__).resolve().parents[2]
    docs=project/'graph-representation/docs'
    lines=['# Spline learning: generated run results','',f'Run directory: `{args.run_root.resolve()}`','',
           '**Four training patches only. These are overfit diagnostics, not held-out validation results.**','',
           '| Run | Updates | Fixed spline weight | Initial curve error (mm) | Final curve error (mm) | Final endpoint error (mm) |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,r in summaries.items():
        lines.append(f"| {name} | {r['steps']} | {r['spline_weight']:.5g} | {r['initial']['spline_curve_mm']:.4f} | {r['final']['spline_curve_mm']:.4f} | {r['final']['spline_endpoint_mm']:.4f} |")
    lines+=['', 'Curve error means mean Euclidean deviation at corresponding spline parameter samples against the fitted target, over matched positive edges. It is not closest-centerline distance or anatomical accuracy.', '',
            f'V0 gate results: `{passes}`. Selected V1 initialization: `{choice}`.',
            'Gate: at least 50% reduction and final mean curve error ≤0.5 mm. This is an exploratory learning check, not a benchmark threshold.', '',
            '## Interactive outputs','']
    for name in summaries:
        html=args.run_root/name/'predictions.html'
        lines.append(f'- [{name}: targets versus predictions]({html.resolve()}). Switch matched-pair geometry and decoded-graph modes.')
    lines+=['','## Next steps','',
            '- Inspect decoded topology separately from matched-positive geometry.',
            '- If V0 fails, inspect endpoint errors, target alignment and optimization; do not conclude the representation is invalid.',
            '- If V0 succeeds, compare V1 from the same original initialization and inspect its predicted curves.',
            '- Then evaluate held-out subjects and more tortuous branches; compare control counts and residual scaling.',
            '- Generalize augmentation and support for loops/parallel branches before dataset-wide experiments.','']
    docs.mkdir(parents=True,exist_ok=True)
    (docs/'spline-learning-results.md').write_text('\n'.join(lines))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
