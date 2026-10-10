#!/usr/bin/env python3
"""Select a full-validation candidate and prepare a bounded stability continuation."""
import argparse
import json
import math
from pathlib import Path
import yaml


def select_candidate(summaries, reference, settings):
    assert reference['provenance']['weights_unchanged'] and reference['common']['samples'] == 1952
    eligible = []
    decisions = {}
    for arm, result in summaries.items():
        assert result['provenance']['sample_ids'] == reference['provenance']['sample_ids']
        for key in ('evaluation_protocol', 'input', 'dataset_root'):
            assert result['provenance'][key] == reference['provenance'][key], key
        assert result['provenance']['weights_unchanged'] and result['common']['samples'] == 1952
        curve = result['common']['common_curve_f1']
        node = result['native']['node_f1']
        shape = result['matched']['all_pairs']['endpoint_neutral_guide_curve_mm']
        branch = result['common']['common_branch_f1']
        edge = result['native']['edge_f1']
        assert all(math.isfinite(v) for v in (curve, node, shape, branch, edge))
        passes = (curve >= reference['common']['common_curve_f1'] - settings['curve_f1_margin']
                  and node >= reference['native']['node_f1'] - settings['node_f1_margin']
                  and shape <= reference['matched']['all_pairs']['endpoint_neutral_guide_curve_mm'] * settings['shape_error_ratio'])
        decisions[arm] = dict(eligible=passes, branch_f1=branch, curve_f1=curve, node_f1=node,
                              edge_f1=edge, endpoint_neutral_mm=shape)
        if passes:
            eligible.append((branch, curve, edge, arm))
    if not eligible:
        raise RuntimeError('No full-validation candidate passed geometry/node preservation guards; do not extend.')
    return max(eligible)[-1], decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/spline_local_stability.yaml'))
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    evaluation = yaml.safe_load(Path(settings['evaluation_config']).read_text())
    reference = json.loads(Path(settings['reference_summary']).read_text())
    summaries = {a: json.loads((Path(evaluation['output_root'])/a/'summary.json').read_text())
                 for a in evaluation['checkpoints']}
    selected, decisions = select_candidate(summaries, reference, settings)
    output = Path(settings['output_root'])
    output.mkdir(parents=True, exist_ok=False)
    config = {k: settings[k] for k in ('seed','updates','detector_lr','head_lr','monitor_interval','geometry_steps','checkpoint_steps')}
    config.update(source_run=str(Path(evaluation['checkpoints'][selected]).parent), source_head_updates=25000,
                  output_root=str(output/'training'), tracking_group='spline-local-stability-20261010',
                  resume_optimizer=True, curvature_breakdown=True, arms=[selected])
    (output/'training-config.yaml').write_text(yaml.safe_dump(config))
    (output/'selected-arm.txt').write_text(selected+'\n')
    (output/'selection.json').write_text(json.dumps(dict(selected=selected, candidates=decisions, settings=settings,
        rule='Eligible geometry/node preservation; maximum branch F1, ties by curve F1 then edge F1. Single-seed validation selection.'),indent=2)+'\n')
    final = dict(evaluation, output_root=str(output/'full-validation'), tracking_group='spline-local-stability-final-20261010',
                 checkpoints={'selected_before': evaluation['checkpoints'][selected],
                              'selected_after': str(output/'training'/selected/'model.pt')})
    # Reuse the completed selected baseline; final array task 1 evaluates only the extension.
    final_root = output/'full-validation'
    final_root.mkdir()
    (final_root/'selected_before').symlink_to(Path(evaluation['output_root'])/selected, target_is_directory=True)
    (output/'final-evaluation-config.yaml').write_text(yaml.safe_dump(final, sort_keys=False))
    print(selected, flush=True)


if __name__ == '__main__':
    main()
