#!/usr/bin/env python3
"""Small validation-only BN differential; does not save or modify checkpoints."""
import argparse
import gc
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from data.loaders import build_data_loaders, build_evaluation_loader
from models import build_model
from models.checkpoint import load_legacy_model_checkpoint
from training.engine import _move_graph_batch
from training.losses.criterion import build_criterion


def seed():
    random.seed(364505)
    np.random.seed(364505)
    torch.manual_seed(364505)
    torch.cuda.manual_seed_all(364505)


def attention_probe(args):
    """Locate lost input sensitivity; reset offsets only in disposable memory."""
    from models.ops.modules.ms_deform_attn import MSDeformAttn
    results = {}
    for arm in ('spline_k8', 'old_graph'):
        seed()
        run = args.runs / f'scratch_{arm}_10k_300ep'
        config = yaml.safe_load((run / 'resolved-config.yaml').read_text())
        config['runtime']['workers'] = 0
        config['data']['validation_batch_size'] = 4
        loader = build_evaluation_loader(config, dataset_name='synthetic_mri', split='val', max_samples=4)
        volumes, _ = _move_graph_batch(next(iter(loader)), 'cuda', config['training']['input'], True)
        model = build_model(config).cuda().eval()
        assert load_legacy_model_checkpoint(model, run / 'models/latest_checkpoint.pt').compatible
        layers = {}; handles = []
        def hook(name):
            def observe(m, inputs, output):
                q, reference, memory, shapes = inputs[:4]
                n, length, _ = q.shape
                offsets = m.sampling_offsets(q).view(n, length, m.n_heads, m.n_levels, m.n_points, 3)
                size = shapes[:, [2, 1, 0]]
                locations = reference[:, :, None, :, None] + offsets / size[None, None, None, :, None]
                margin = .5 / size[None, None, None, :, None]
                supported = ((locations > -margin) & (locations < 1 + margin)).all(-1)
                weights = m.attention_weights(q).view(n, length, m.n_heads, -1).softmax(-1).view_as(supported)
                layers[name] = {'supported_sample_fraction': float(supported.float().mean()),
                                'supported_attention_mass': float((weights * supported).sum((-1, -2)).mean()),
                                'memory_pair_mae': float((memory[0] - memory[1]).abs().mean()),
                                'output_pair_mae': float((output[0] - output[1]).abs().mean()),
                                'location_min': float(locations.min()), 'location_max': float(locations.max())}
            return observe
        for name, m in model.named_modules():
            if isinstance(m, MSDeformAttn):
                handles.append(m.register_forward_hook(hook(name)))
        results[arm] = {}
        with torch.no_grad():
            for mode in (('saved', 'zero_cross_attention_offsets') if arm == 'spline_k8' else ('saved',)):
                if mode != 'saved':
                    for name, m in model.named_modules():
                        if isinstance(m, MSDeformAttn) and 'cross_attn' in name:
                            m.sampling_offsets.weight.zero_(); m.sampling_offsets.bias.zero_()
                layers.clear()
                tokens, pred, _ = model(volumes)
                results[arm][mode] = {'node_pair_max_delta': float((pred['pred_nodes'][0] - pred['pred_nodes'][1]).abs().max()),
                                      'layers': dict(layers)}
        for h in handles:
            h.remove()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({arm: results[arm]}), flush=True)
        del model, loader, volumes, tokens, pred
        gc.collect(); torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--attention-only', action='store_true')
    args = p.parse_args()
    if args.attention_only:
        attention_probe(args)
        return
    results = {}
    for arm in ('spline_k8', 'old_graph', 'adaptive_degree2'):
        seed()
        run = args.runs / f'scratch_{arm}_10k_300ep'
        config = yaml.safe_load((run / 'resolved-config.yaml').read_text())
        config['runtime']['workers'] = 0
        config['data']['validation_batch_size'] = 4
        loader = build_evaluation_loader(config, dataset_name='synthetic_mri', split='val', max_samples=8)
        batches = [_move_graph_batch(b, 'cuda', config['training']['input'], True) for b in loader]
        model = build_model(config).cuda()
        report = load_legacy_model_checkpoint(model, run / 'models/latest_checkpoint.pt')
        assert report.compatible, report
        criterion = build_criterion(config, model).cuda()
        criterion.validation = True
        bn = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        result = {'checkpoint': str(run / 'models/latest_checkpoint.pt'),
                  'samples': [r.sample_id for r in loader.dataset.records],
                  'strict_compatible': report.compatible, 'bn_layers': len(bn),
                  'bn_variance_range': [min(float(m.running_var.min()) for m in bn), max(float(m.running_var.max()) for m in bn)],
                  'bn_batches_range': [min(int(m.num_batches_tracked) for m in bn), max(int(m.num_batches_tracked) for m in bn)],
                  'input_pair_mae': float((batches[0][0][0] - batches[0][0][1]).abs().mean()), 'modes': {}}
        for mode in ('saved_running_stats', 'batch_stats'):
            seed()
            model.eval()
            if mode == 'batch_stats':
                # Only BN changes; dropout and decoder stay in evaluation mode.
                for m in bn:
                    m.train()
            rows = []
            with torch.no_grad():
                for volumes, targets in batches:
                    tokens, pred, features = model(volumes)
                    losses = criterion(tokens, pred, targets)
                    rows.append({'losses': {k: float(v) for k, v in losses.items() if torch.is_tensor(v) and v.numel() == 1},
                                 'node_pair_max_delta': float((pred['pred_nodes'][0] - pred['pred_nodes'][1]).abs().max()),
                                 'token_pair_mae': float((tokens[0] - tokens[1]).abs().mean()),
                                 'feature_pair_mae': float((features[0] - features[1]).abs().mean()),
                                 'feature_abs_mean': float(features.abs().mean())})
            result['modes'][mode] = rows
        results[arm] = result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({arm: result}), flush=True)
        del model, criterion, batches, loader, tokens, pred, features, losses, bn
        gc.collect()
        torch.cuda.empty_cache()

    # One training batch, no optimizer step: compare the actual gradient scales
    # at initialization, including the coordinate and image-feature paths.
    seed()
    config = yaml.safe_load((args.runs / 'scratch_spline_k8_10k_300ep/resolved-config.yaml').read_text())
    config['runtime']['workers'] = 0
    config['data']['datasets']['synthetic_mri']['train_samples'] = 4
    config['data']['datasets']['synthetic_mri']['validation_samples'] = 4
    loader, _ = build_data_loaders(config)
    volumes, targets = _move_graph_batch(next(iter(loader)), 'cuda', config['training']['input'], True)
    seed()
    model = build_model(config).cuda().train()
    criterion = build_criterion(config, model).cuda()
    tokens, pred, features = model(volumes)
    losses = criterion(tokens, pred, targets)
    spline = criterion.weights['spline'] * losses['spline']
    graph = losses['total'] - spline
    tensors = (tokens, pred['pred_nodes'], features)
    a = torch.autograd.grad(graph, tensors, retain_graph=True)
    b = torch.autograd.grad(spline, tensors)
    gradients = {}
    for name, g, s in zip(('tokens', 'node_coordinates', 'image_features'), a, b):
        gradients[name] = {'graph_norm': float(g.norm()), 'weighted_spline_norm': float(s.norm()),
                           'spline_over_graph': float(s.norm() / g.norm().clamp_min(1e-12)),
                           'cosine': float((g.flatten() @ s.flatten()) / (g.norm() * s.norm()).clamp_min(1e-12))}
    results['initial_training_batch_gradients'] = {'graph_loss': float(graph), 'weighted_spline_loss': float(spline),
                                                  'gradients': gradients, 'sample_count': len(volumes)}
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({'initial_training_batch_gradients': results['initial_training_batch_gradients']}), flush=True)


if __name__ == '__main__':
    main()
