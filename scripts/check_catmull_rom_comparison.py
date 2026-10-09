#!/usr/bin/env python3
"""Verify paired splits, unchanged optimization, and common random initialization."""
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from configs import load_config
from data.loaders import discover_synthetic_mri, build_evaluation_loader
from models import build_model


def main():
    root = Path(os.environ['SYNTHETIC_MRI_DATASET'])
    manifest = json.loads((root/'manifest.json').read_text())
    comparison_path = Path(manifest['comparison_manifest'])
    assert hashlib.sha256(comparison_path.read_bytes()).hexdigest() == manifest['comparison_manifest_sha256']
    comparison = json.loads(comparison_path.read_text())
    config = load_config('configs/scratch_comparison_catmull_rom_300ep.yaml')
    baseline = load_config('configs/scratch_comparison_spline_300ep.yaml')
    for key in ('training', 'loss', 'evaluation', 'optimizer', 'scheduler'):
        assert config.get(key) == baseline.get(key), key
    assert config['model']['spline']['controls'] == 8
    assert config['training']['epochs'] == 300
    assert config['loss']['spline']['mode'] == 'controls'
    assert not config['model']['spline']['curve_evidence']['enabled']
    for split, key in [('train','train_ids'), ('val','validation_ids'), ('test','test_ids')]:
        ids = sorted(record.sample_id for record in discover_synthetic_mri(root, split))
        assert ids == comparison[key], split
        assert sorted(p.stem for p in (root/split/'spline').glob('*.npz')) == ids
    # Compare learned tensors; family-specific fixed buffers intentionally differ.
    hashes = []
    counts = []
    for cfg in (baseline, config):
        torch.manual_seed(cfg['experiment']['seed'])
        model = build_model(cfg)
        hashes.append({name: hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
                       for name, value in model.named_parameters()})
        counts.append(sum(value.numel() for value in model.parameters()))
        del model
    assert hashes[0] == hashes[1], 'Learned initial tensors differ'
    assert counts[0] == counts[1]
    config['runtime']['workers'] = 0
    loader = build_evaluation_loader(config, dataset_name='synthetic_mri', split='val', max_samples=1)
    batch = next(iter(loader))
    assert batch[4][0]['controls'].shape[1:] == (8,3)
    print(json.dumps(dict(passed=True, counts=manifest['counts'], learned_parameters=counts[1],
                          identical_initial_learned_tensors=True, unchanged_optimization=True)), flush=True)


if __name__ == '__main__':
    main()
