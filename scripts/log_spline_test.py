#!/usr/bin/env python3
"""Attach final test metrics and evaluated-population coverage to the W&B run."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs import load_config
from training.tracking import build_tracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.run/'resolved-config.yaml')
    manifest = json.loads((Path(config['data']['datasets']['synthetic_mri']['root'])/'manifest.json').read_text())
    summary = json.loads((args.run/'test/summary.json').read_text())
    population = {'source_test_patches': manifest['source_counts']['test'],
                  'available_test_patches': manifest.get('available_counts', manifest['source_counts'])['test'],
                  'excluded_source_subjects': manifest.get('excluded_source_subjects', {}).get('test', []),
                  'compatible_test_patches': manifest['accepted_counts']['test'],
                  'evaluated_test_patches': summary['samples'],
                  'excluded_by_reason': manifest['excluded_counts']['test']}
    assert summary['samples'] == population['compatible_test_patches']
    (args.run/'test/population.json').write_text(json.dumps(population, indent=2)+'\n')
    tracker = build_tracker(config, args.run, resume=True, launch_metadata={'purpose': 'Final test only after validation checkpoint selection'})
    if tracker:
        tracker.run.log({'test/'+k: v for k, v in summary.items() if isinstance(v, (int, float))})
        tracker.run.summary['test/population'] = population
        tracker.finish()


if __name__ == '__main__':
    main()
