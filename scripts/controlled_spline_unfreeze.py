#!/usr/bin/env python3
"""Bounded comparison of geometry gradient routes through a pretrained detector."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from configs import validate_config
from data.loaders import build_data_loaders, build_evaluation_loader, dataset_sample_manifest
from models import build_model
from models.checkpoint import load_checkpoint_file
from models.matcher import build_matcher
from training.engine import _move_graph_batch, evaluate_loss
from training.losses.criterion import GraphCriterion, build_criterion
from training.tracking import build_tracker
from scripts.train_spline_weight_pilot import seed, monitoring_rng, sensitivity, geometry, write_json
from scripts.frozen_spline_diagnostic import training_evaluation_loader, matched_geometry, detector_digest


class RoutedCriterion(GraphCriterion):
    """Change backward routes only; graph losses and spline forward values are identical."""
    def __init__(self, *args, route, **kwargs):
        super().__init__(*args, **kwargs)
        assert route in ('graph_only', 'tokens_only', 'joint')
        self.route = route

    def loss_spline(self, tokens, predictions, targets, assignments):
        if self.route == 'graph_only':
            tokens = tokens.detach()
        if self.route != 'joint':
            predictions = dict(predictions, pred_nodes=predictions['pred_nodes'].detach())
        return super().loss_spline(tokens, predictions, targets, assignments)


def configure_unfreezing(model):
    assert model.spline_head.curve_evidence is None
    # encoder is the CNN; decoder.encoder is the transformer encoder.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith('encoder.'))
    model.eval()  # Fixed BN statistics and no dropout in every comparison arm.


def frozen_state_digest(model):
    state = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    state.update(dict(model.named_buffers()))
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def gradient_diagnostic(model, losses, weight):
    groups = {'transformer_encoder': 'decoder.encoder.', 'transformer_decoder': 'decoder.decoder.',
              'coordinates': 'coord_embed.', 'projection': 'input_proj.', 'queries': 'query_embed.',
              'relation': 'relation_embed.'}
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and not n.startswith('spline_head.')]
    graph = losses['total'] - weight * losses['spline']
    gg = torch.autograd.grad(graph, [p for _, p in named], retain_graph=True, allow_unused=True)
    sg = torch.autograd.grad(weight * losses['spline'], [p for _, p in named], retain_graph=True, allow_unused=True)
    result = {}
    for group, prefix in groups.items():
        a2 = b2 = dot = 0.
        for (name, _), a, b in zip(named, gg, sg):
            if not name.startswith(prefix):
                continue
            if a is not None: a2 += float(a.detach().square().sum())
            if b is not None: b2 += float(b.detach().square().sum())
            if a is not None and b is not None: dot += float((a.detach()*b.detach()).sum())
        result[group] = dict(graph_norm=a2**.5, weighted_spline_norm=b2**.5,
                             cosine=dot/(a2*b2)**.5 if a2*b2 > 0 else None)
    return result


def assignment_change(current, baseline):
    assert current.keys() == baseline.keys()
    total = changed = 0
    for sample, mapping in baseline.items():
        keys = set(mapping) | set(current[sample])
        total += len(keys)
        changed += sum(mapping.get(k) != current[sample].get(k) for k in keys)
    return changed / total if total else 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/spline_controlled_unfreeze.yaml'))
    parser.add_argument('--arm', choices=['graph_only', 'tokens_only', 'joint'], required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    pilot = yaml.safe_load(args.config.read_text())
    source = Path(pilot['source_run'])
    source_summary = json.loads((source / 'summary.json').read_text())
    checkpoint = load_checkpoint_file(source / 'model.pt')
    assert checkpoint['cumulative_head_updates'] == 15000
    config = copy.deepcopy(checkpoint['config'])
    assert config['loss']['spline']['target'] == 'residual'
    assert config['model']['spline']['token_inputs'] == 'full'
    assert config['data']['batch_size'] == 4 and not config['data']['train_augmentation']
    assert config['model']['matcher']['type'] == 'hungarian'
    assert not config.get('domain_adaptation', {}).get('enabled', False)
    config['experiment'].update(seed=pilot['seed'], name='spline_unfreeze_' + args.arm)
    config['runtime']['workers'] = 4
    config['tracking'].update(enabled=not args.smoke, mode='offline', group='spline-unfreeze-20261009')
    validate_config(config)
    output = Path(pilot['output_root']) / (('smoke-' if args.smoke else '') + args.arm)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'resolved-config.yaml').write_text(yaml.safe_dump(config))
    write_json(output / 'pilot-config.json', pilot)
    seed(pilot['seed'])
    train, _ = build_data_loaders(config)
    membership = dataset_sample_manifest(train.dataset)
    prior = json.loads((source / 'dataset-manifest.json').read_text())
    assert membership == prior['train'] and len(membership[0]['sample_ids']) == 10000
    ids = json.loads((source / 'evaluation-sample-ids.json').read_text())
    assert len(ids['train']) == 512 and len(ids['val']) == 128
    if args.smoke:
        ids = {k: v[:4] for k, v in ids.items()}
    write_json(output / 'evaluation-sample-ids.json', ids)
    write_json(output / 'dataset-manifest.json', prior)
    train_eval = training_evaluation_loader(train, config, ids['train'])
    val = build_evaluation_loader(config, dataset_name='synthetic_mri', split='val', sample_ids=ids['val'])
    probe_ids = [ids['val'][0], ids['val'][-1], ids['val'][1], ids['val'][-2]]
    probe = next(iter(build_evaluation_loader(config, dataset_name='synthetic_mri', split='val', sample_ids=probe_ids)))
    seed(pilot['seed'])
    model = build_model(config)
    model.load_state_dict(checkpoint['net'], strict=True)
    initial = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        initial.update(name.encode()); initial.update(value.numpy().tobytes())
    del checkpoint
    model = model.cuda()
    configure_unfreezing(model)
    frozen_digest = frozen_state_digest(model)
    criterion = RoutedCriterion(config, build_matcher(config), model.relation_embed,
                                spline_head=model.spline_head, route=args.arm).cuda()
    val_config = copy.deepcopy(config)
    val_config['loss']['edge']['balancing']['mode'] = 'none'
    val_config['loss']['edge']['candidates']['positive_cap'] = 40
    val_criterion = build_criterion(val_config, model, validation=True).cuda()
    settings = config['training']['optimizer']
    optimizer = torch.optim.AdamW([
        {'params': [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('spline_head.')], 'lr': pilot['detector_lr']},
        {'params': list(model.spline_head.parameters()), 'lr': pilot['head_lr']}],
        weight_decay=settings['weight_decay'], eps=settings['epsilon'], betas=tuple(settings['betas']))
    updates = 3 if args.smoke else pilot['updates']
    provenance = dict(arm=args.arm, source_run=str(source), source_head_updates=15000,
                      initial_state_sha256=initial.hexdigest(), frozen_state_sha256=frozen_digest,
                      initial_detector_sha256=detector_digest(model), updates=updates, seed=pilot['seed'],
                      detector_lr=pilot['detector_lr'], head_lr=pilot['head_lr'],
                      optimizer_restart=True, scheduler='constant', model_mode='eval throughout; gradients enabled',
                      probe_ids=probe_ids, spline_weight=criterion.weights['spline'],
                      trainable_parameters=[n for n,p in model.named_parameters() if p.requires_grad])
    write_json(output / 'provenance.json', provenance)
    tracker = build_tracker(config, output, resume=False, launch_metadata=provenance)
    history = []; assignments = {}; last_assignments = {}
    started = time.monotonic()

    def monitor(step):
        with monitoring_rng():
            assert frozen_state_digest(model) == frozen_digest, 'CNN parameters or buffers changed'
            row = dict(update=step, elapsed_seconds=time.monotonic()-started,
                       validation=evaluate_loss(model, val_criterion, val, config, 'cuda'),
                       sensitivity=sensitivity(model, probe, config))
            if step in pilot['geometry_steps'] or (args.smoke and step == updates):
                for split, loader in [('train', train_eval), ('val', val)]:
                    row['train_geometry' if split=='train' else 'geometry'] = geometry(
                        model, loader, config, output, step, tracker if split=='val' else None,
                        step//len(train), True, split=split)
                    report = matched_geometry(model, val_criterion, loader, config, 'cuda', include_assignments=True)
                    write_json(output / f'{split}-matched-geometry-{step:06d}.json', report)
                    current = report.pop('assignments'); report.pop('rows')
                    assignments.setdefault(split, current)
                    report['assignment_changed_fraction_from_initial'] = assignment_change(current, assignments[split])
                    report['assignment_changed_fraction_from_previous'] = assignment_change(current, last_assignments.get(split, current))
                    last_assignments[split] = current
                    row[split+'_matched_geometry'] = report
            row['frozen_state_unchanged'] = True
            if step == 0 and not args.smoke:
                for key in ('geometry', 'train_geometry'):
                    for metric, value in row[key].items():
                        previous = source_summary['final'][key][metric]
                        # Thresholded curve F1 can move by ~1e-6 under float32
                        # inference drift; this is a diagnostic tolerance only.
                        assert (value == previous or
                                (value is not None and previous is not None and abs(value-previous) < 1e-5)), (key, metric)
        history.append(row)
        with (output / 'history.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        if tracker:
            tracker.log_validation(row['validation'], iteration=step, epoch=step//len(train))
            tracker.log_metrics(row['sensitivity'], iteration=step, epoch=step//len(train))
            if 'val_matched_geometry' in row:
                tracker.log_metrics({f'{s}_matched/{k}': v for s in ('train','val')
                    for k,v in row[s+'_matched_geometry']['all_pairs'].items() if v is not None}, iteration=step, epoch=step//len(train))
        print(json.dumps(row), flush=True)

    try:
        monitor(0)
        seed(pilot['seed'])
        iterator = iter(train); order = hashlib.sha256()
        for step in range(1, updates+1):
            try: batch = next(iterator)
            except StopIteration:
                iterator = iter(train); batch = next(iterator)
            for nodes in batch[2]: order.update(nodes.numpy().tobytes())
            volumes, targets = _move_graph_batch(batch, 'cuda', config['training']['input'], True)
            model.eval(); optimizer.zero_grad(set_to_none=True)
            criterion.set_training_progress(1, 0.)  # Identical fixed graph-loss settings across arms.
            tokens, predictions, _ = model(volumes)
            losses = criterion(tokens, predictions, targets)
            assert torch.isfinite(losses['total']), 'Nonfinite total loss'
            if step in (1, 1000, updates):
                diagnostic = gradient_diagnostic(model, losses, criterion.weights['spline'])
                if args.arm == 'graph_only':
                    assert all(v['weighted_spline_norm'] == 0 for v in diagnostic.values())
                if args.arm == 'tokens_only':
                    assert diagnostic['coordinates']['weighted_spline_norm'] == 0
                with (output / 'gradients.jsonl').open('a') as f:
                    f.write(json.dumps(dict(update=step, groups=diagnostic))+'\n')
            losses['total'].backward()
            assert all(p.grad is None for p in model.encoder.parameters())
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if step == 1:
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.spline_head.parameters())
            optimizer.step()
            if tracker and step % 100 == 0:
                tracker.log_training({k: float(v.detach()) for k,v in losses.items() if torch.is_tensor(v) and v.numel()==1},
                                     iteration=step, epoch=1+(step-1)//len(train), learning_rate=pilot['detector_lr'])
            if step % pilot['monitor_interval'] == 0 or step == updates:
                monitor(step)
        provenance.update(batch_order_sha256=order.hexdigest(), elapsed_seconds=time.monotonic()-started,
                          final_detector_sha256=detector_digest(model))
        write_json(output / 'provenance.json', provenance)
        if not args.smoke:
            torch.save(dict(net=model.state_dict(), config=config, steps=updates,
                            optimizer=optimizer.state_dict(), provenance=provenance), output / 'model.pt')
        write_json(output / 'summary.json', dict(arm=args.arm, provenance=provenance, initial=history[0], final=history[-1]))
        if tracker: tracker.finish()
    except BaseException:
        if tracker: tracker.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
