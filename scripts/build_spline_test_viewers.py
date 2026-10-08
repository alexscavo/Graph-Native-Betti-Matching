#!/usr/bin/env python3
"""Export fixed or explicitly selected held-out predictions to offline 3D viewers."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs
from vessel_mesh import mask_mesh


def prediction_records(path):
    """Read the large exported JSON array without loading it all into memory."""
    decoder = json.JSONDecoder()
    with path.open() as handle:
        assert handle.read(1) == '['
        buffer = ''
        while True:
            buffer = buffer.lstrip(' \n\r\t,')
            if buffer.startswith(']'):
                return
            try:
                record, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = handle.read(1024*1024)
                if not chunk:
                    raise ValueError('Incomplete prediction array')
                buffer += chunk
                continue
            buffer = buffer[end:]
            yield record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', choices=('best','final'), help='Checkpoint test subdirectory; omit for historical flat test outputs')
    parser.add_argument('--sample-id', help='Export one explicit sample, without quality-based selection')
    args = parser.parse_args()
    test_dir = args.run/'test'/args.selection if args.selection else args.run/'test'
    groups = defaultdict(list)
    with (test_dir/'per-patch-metrics.csv').open() as handle:
        for row in csv.DictReader(handle):
            sample = row['source_sample_id']
            groups[int(sample.split('_')[1])].append(sample)
    subjects = sorted(groups)[:2]
    selected = [args.sample_id] if args.sample_id else [sorted(groups[p])[len(groups[p])//2] for p in subjects]
    assert all(sample in {s for ids in groups.values() for s in ids} for sample in selected)
    found = {}
    for row in prediction_records(test_dir/'predictions.json'):
        if row['source_sample_id'] in selected:
            found[row['source_sample_id']] = row
        if len(found) == len(selected):
            break
    assert set(found) == set(selected)
    args.output.mkdir(parents=True, exist_ok=True)
    template = (Path(__file__).parent/'spline_test_predictions.html').read_text()
    script = get_plotlyjs()
    metadata = json.loads((test_dir/'metadata.json').read_text())
    provenance = {'selection': ('Explicit sample ID; same fixed example used for earlier held-out viewers, no quality selection.' if args.sample_id else 'Median eligible sample ID within each of the two lowest-ID test subjects; no prediction-quality selection.'),
                  'checkpoint_selection': args.selection or 'historical validation-selected',
                  'bn_calibration_batches': metadata['bn_calibration_batches'],
                  'checkpoint': json.loads((test_dir/'metadata.json').read_text())['checkpoint'],
                  'samples': []}
    for sample in selected:
        prediction = found[sample]
        with np.load(args.targets/'test/spline'/f'{sample}.npz') as target:
            data = {k: target[k].tolist() for k in ('nodes', 'edges', 'controls', 'curves', 'metric')}
            paths, offsets = target['full_paths'], target['path_offsets']
            guides = [paths[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
        np.testing.assert_allclose(prediction['spline_metric'], data['metric'], atol=1e-6)
        assert len(prediction['edges']) == len(prediction['spline_controls']) == len(prediction['spline_curves'])
        if prediction['edges']:
            endpoints = np.asarray(prediction['nodes_dhw'])[np.asarray(prediction['edges'])]
            np.testing.assert_allclose(np.asarray(prediction['spline_controls'])[:, [0, -1]], endpoints, atol=1e-6)
        page_data = {'sample': sample, 'metric': data['metric'], 'target_nodes': data['nodes'],
                     'target_edges': data['edges'], 'target_curves': data['curves'], 'guides': guides,
                     'prediction': prediction,
                     'mesh': mask_mesh(args.targets/'test/seg'/f'{sample}_seg.nii.gz')}
        example_template = template
        if args.selection=='final':
            example_template=example_template.replace('validation-selected checkpoint used in the final test','final epoch-300 checkpoint used in this test')
        page = example_template.replace('__PLOTLY__', script).replace('__DATA__', json.dumps(page_data, separators=(',', ':')))
        filename = f'{sample}.html'
        (args.output/filename).write_text(page)
        # Compact data supports a static inspection and reproducible checks.
        (args.output/f'{sample}.json').write_text(json.dumps(page_data, separators=(',', ':')))
        provenance['samples'].append({'sample_id': sample, 'html': filename,
                                     'predicted_nodes': len(prediction['nodes_dhw']),
                                     'predicted_edges': len(prediction['edges']),
                                     'target_nodes': len(data['nodes']), 'target_edges': len(data['edges']),
                                     'metrics': prediction['metrics']})
        print('Saved', args.output/filename, flush=True)
    (args.output/'selection.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
