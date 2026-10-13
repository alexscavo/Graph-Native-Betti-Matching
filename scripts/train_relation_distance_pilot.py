#!/usr/bin/env python3
"""Controlled relation-only and node-head pilots from one final spline checkpoint."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from configs import validate_config
from data.loaders import build_data_loaders, build_evaluation_loader, dataset_sample_manifest
from models import build_model
from models.checkpoint import load_checkpoint_file
from models.matcher import build_matcher
from training.engine import _move_graph_batch, evaluate_loss
from training.losses.criterion import build_criterion
from training.losses.relation_distance import DistanceRelationCriterion
from training.tracking import build_tracker
from scripts.train_spline_weight_pilot import seed, monitoring_rng, geometry, write_json
from scripts.frozen_spline_diagnostic import training_evaluation_loader
from scripts.controlled_spline_unfreeze import frozen_state_digest
from scripts.evaluate_vessel_core import records, reference
from scripts.diagnose_branch_relations import diagnose, summarize
from metrics.vessel_graph import VesselGraph
from metrics.vessel_suite import evaluate, aggregate


def configure_heads(model, node_learning):
    prefixes = ('relation_embed.', 'class_embed.', 'coord_embed.') if node_learning else ('relation_embed.',)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    model.eval()  # All BN buffers and dropout behavior fixed, including node arms.


def prediction_diagnostics(path, root, split, baseline, *, core):
    totals = {t: {} for t in (1., 2.)}
    branches = {t: [] for t in totals}
    pairs = {t: [dict(positive_candidates=0, positive_selected=0, negative_candidates=0, negative_selected=0)
                 for _ in range(4)] for t in totals}
    core_rows = []
    for record in records(path):
        name = record['source_sample_id']
        gt, metric = reference(root/split/'spline'/(name+'.npz'))
        predicted = VesselGraph.from_prediction(record, metric)
        for t in totals:
            counts, rows, candidates = diagnose(predicted, gt, t,
                baseline['gt_arc_quartile_cuts_mm'], baseline['gt_chord_quartile_cuts_mm'])
            for key, value in counts.items(): totals[t][key] = totals[t].get(key, 0) + value
            branches[t].extend(dict(sample_id=name, **r) for r in rows)
            for i, candidate in enumerate(candidates):
                for key, value in candidate.items(): pairs[t][i][key] += value
        if core: core_rows.append(dict(sample_id=name, **evaluate(predicted, gt)))
    result = {str(t): summarize(totals[t], branches[t], pairs[t]) for t in totals}
    write_json(path.parent/'relation-diagnostic.json', result)
    if core:
        write_json(path.parent/'vessel-core-per-patch.json', core_rows)
        write_json(path.parent/'vessel-core-summary.json', aggregate(core_rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/relation_distance_pilot.yaml'))
    parser.add_argument('--arm', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    pilot = yaml.safe_load(args.config.read_text())
    assert args.arm in pilot['arms']
    arm = pilot['arms'][args.arm]
    checkpoint = load_checkpoint_file(pilot['checkpoint'])
    config = copy.deepcopy(checkpoint['config'])
    assert config['model']['matcher']['type'] == 'hungarian'
    assert not config['model']['node_type']['enabled']
    assert not config['data']['train_augmentation'] and config['data']['batch_size'] == 4
    assert not config['loss']['degree']['enabled']
    assert all(not config['topology'][name]['enabled'] for name in ('betti_h0', 'betti_h1'))
    config['experiment'].update(seed=pilot['seed'], name='relation_distance_' + args.arm)
    config['loss']['weights']['spline'] = 0.
    config['runtime']['workers'] = 4
    config['tracking'].update(enabled=not args.smoke, mode='offline', group=pilot['tracking_group'],
        tags=['gnbm', 'spline', 'relation-distance', args.arm])
    validate_config(config)
    output = Path(pilot['output_root']) / (('smoke-' if args.smoke else '') + args.arm)
    output.mkdir(parents=True, exist_ok=False)
    (output/'resolved-config.yaml').write_text(yaml.safe_dump(config))
    write_json(output/'pilot-config.json', pilot)
    baseline = json.loads(Path(pilot['baseline_diagnostic']).read_text())
    membership_source = Path(pilot['evaluation_membership_run'])
    ids = json.loads((membership_source/'evaluation-sample-ids.json').read_text())
    population = json.loads(Path(pilot['population_manifest']).read_text())
    seed(pilot['seed'])
    train, _ = build_data_loaders(config)
    membership = dataset_sample_manifest(train.dataset)
    train_ids = membership[0]['sample_ids']
    assert len(train_ids) == 10000 and set(train_ids) == set(population['train_ids'])
    assert len(ids['train']) == 512 and len(ids['val']) == 128
    assert set(ids['train']) <= set(train_ids)
    assert set(ids['val']) <= set(population['validation_ids'])
    assert {s.split('_')[1] for s in train_ids}.isdisjoint({s.split('_')[1] for s in population['validation_ids']})
    full_ids = sorted(population['validation_ids'])
    assert len(full_ids) == 1952
    if args.smoke:
        ids = {key: value[:4] for key, value in ids.items()}
        full_ids = ids['val']
    write_json(output/'evaluation-sample-ids.json', dict(ids, full_val=full_ids))
    write_json(output/'dataset-manifest.json', {'train': membership})
    train_eval = training_evaluation_loader(train, config, ids['train'])
    val = build_evaluation_loader(config, dataset_name='synthetic_mri', split='val', sample_ids=ids['val'])
    full_val = build_evaluation_loader(config, dataset_name='synthetic_mri', split='val', sample_ids=full_ids)
    seed(pilot['seed'])
    model = build_model(config)
    model.load_state_dict(checkpoint['net'], strict=True)
    del checkpoint
    assert model.spline_head.curve_evidence is None
    initial = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        initial.update(name.encode()); initial.update(value.numpy().tobytes())
    model.cuda()
    configure_heads(model, arm['node_learning'])
    frozen_digest = frozen_state_digest(model)
    criterion = DistanceRelationCriterion(config, build_matcher(config), model.relation_embed,
        spline_head=model.spline_head, radius_mm=pilot['nearby_radius_mm'],
        factor=pilot['nearby_negative_factor'] if arm['near_weighting'] else 1.).cuda()
    val_criterion = build_criterion(config, model, validation=True).cuda()
    settings = config['training']['optimizer']
    groups = [{'params': list(model.relation_embed.parameters()), 'lr': pilot['relation_lr']}]
    if arm['node_learning']:
        groups.append({'params': list(model.class_embed.parameters()) + list(model.coord_embed.parameters()),
                       'lr': pilot['node_lr']})
    optimizer = torch.optim.AdamW(groups, weight_decay=settings['weight_decay'],
        eps=settings['epsilon'], betas=tuple(settings['betas']))
    updates = 3 if args.smoke else pilot['updates']
    provenance = dict(arm=args.arm, checkpoint=pilot['checkpoint'], initial_state_sha256=initial.hexdigest(),
        frozen_state_sha256=frozen_digest, seed=pilot['seed'], updates=updates, **arm,
        trainable_parameters=[n for n, p in model.named_parameters() if p.requires_grad],
        relation_lr=pilot['relation_lr'], node_lr=pilot['node_lr'] if arm['node_learning'] else None,
        nearby_radius_mm=pilot['nearby_radius_mm'], nearby_negative_factor=criterion.factor,
        optimizer='fresh AdamW; constant LR; all modules remain in eval mode',
        loss_weights=criterion.weights, train_samples=len(train_ids), validation_samples=len(full_ids),
        gt_arc_quartile_cuts_mm=baseline['gt_arc_quartile_cuts_mm'],
        gt_chord_quartile_cuts_mm=baseline['gt_chord_quartile_cuts_mm'],
        runtime_versions=dict(torch=torch.__version__, numpy=np.__version__))
    write_json(output/'provenance.json', provenance)
    tracker = build_tracker(config, output, resume=False, launch_metadata=provenance)
    started = time.monotonic(); history = []
    root = Path(config['data']['datasets']['synthetic_mri']['root'])
    # Repeated input verifies frozen representations/nodes in the actual model.
    probe = next(iter(val))
    probe_volumes, _ = _move_graph_batch(probe, 'cuda', config['training']['input'], True)
    with torch.no_grad():
        probe_tokens, probe_predictions, _ = model(probe_volumes)
        reference_nodes = probe_predictions['pred_nodes'].clone()
        reference_logits = probe_predictions['pred_logits'].clone()
        reference_tokens = probe_tokens.clone()
    del probe_predictions, probe_tokens

    def monitor(step):
        final = step == updates
        with monitoring_rng():
            model.eval()
            assert frozen_state_digest(model) == frozen_digest, 'Frozen parameters/buffers changed'
            with torch.no_grad():
                tokens, predictions, _ = model(probe_volumes)
            assert torch.equal(tokens, reference_tokens), 'Frozen transformer representation changed'
            if not arm['node_learning']:
                assert torch.equal(predictions['pred_nodes'], reference_nodes)
                assert torch.equal(predictions['pred_logits'], reference_logits)
            elif final:
                assert not torch.equal(predictions['pred_nodes'], reference_nodes), 'Node head did not update'
            row = dict(update=step, validation=evaluate_loss(model, val_criterion, val, config, 'cuda'),
                       frozen_state_unchanged=True, tokens_unchanged=True)
            loader = full_val if final else val
            row['validation_samples'] = len(loader.dataset)
            row['geometry'] = geometry(model, loader, config, output, step, tracker, step//len(train), True)
            row['relation_diagnostic'] = prediction_diagnostics(output/f'validation-{step:06d}'/'predictions.json',
                root, 'val', baseline, core=final)
            if step == 0 or final:
                row['train_geometry'] = geometry(model, train_eval, config, output, step, None, step//len(train), True, split='train')
                row['train_relation_diagnostic'] = prediction_diagnostics(output/f'train-evaluation-{step:06d}'/'predictions.json',
                    root, 'train', baseline, core=False)
            if final:
                row['core_metrics'] = json.loads((output/f'validation-{step:06d}'/'vessel-core-summary.json').read_text())
        row['elapsed_seconds'] = time.monotonic()-started
        history.append(row)
        with (output/'history.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        if tracker:
            tracker.log_validation(row['validation'], iteration=step, epoch=step//len(train))
            for tolerance, report in row['relation_diagnostic'].items():
                tracker.log_metrics({f'relation_audit/{tolerance}mm/{key}': report[key]
                    for key in ('node_recall', 'node_precision', 'endpoint_pair_recovery', 'conditional_direct_recall')
                    if report[key] is not None}, iteration=step, epoch=step//len(train))
            if final:
                tracker.log_metrics({f'core/{key}': row['core_metrics'][key]
                    for key in ('curve_f1_at_1mm', 'branch_f1', 'junction_f1', 'apls_3d', 'topo_3d_f1', 'betti0_error_mean', 'betti1_error_mean')},
                    iteration=step, epoch=step//len(train))
        print(json.dumps(dict(arm=args.arm, update=step, elapsed_seconds=row['elapsed_seconds'],
                              validation_samples=row['validation_samples'], geometry=row['geometry'])), flush=True)

    try:
        monitor(0)
        seed(pilot['seed'])
        iterator = iter(train); order = hashlib.sha256(); pair_counts = {}
        for step in range(1, updates+1):
            try: batch = next(iterator)
            except StopIteration: iterator = iter(train); batch = next(iterator)
            for nodes in batch[2]: order.update(nodes.numpy().tobytes())
            volumes, targets = _move_graph_batch(batch, 'cuda', config['training']['input'], True)
            model.eval(); optimizer.zero_grad(set_to_none=True)
            tokens, predictions, _ = model(volumes)
            losses = criterion(tokens, predictions, targets)
            assert torch.isfinite(losses['total'])
            losses['total'].backward()
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad: assert parameter.grad is None, name
                elif parameter.grad is not None: assert torch.isfinite(parameter.grad).all(), name
            if step == 1:
                prefixes = ('relation_embed.', 'class_embed.', 'coord_embed.') if arm['node_learning'] else ('relation_embed.',)
                for prefix in prefixes:
                    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.startswith(prefix)), prefix
            optimizer.step()
            for key, value in criterion.last_pair_counts.items(): pair_counts[key] = pair_counts.get(key, 0) + value
            if tracker and step % 100 == 0:
                tracker.log_training({k: float(v.detach()) for k, v in losses.items() if torch.is_tensor(v) and v.numel()==1},
                    iteration=step, epoch=step//len(train), learning_rate=pilot['relation_lr'])
            if step % pilot['monitor_interval'] == 0 or step == updates:
                monitor(step)
        provenance.update(batch_order_sha256=order.hexdigest(), training_pair_counts=pair_counts,
            elapsed_seconds=time.monotonic()-started, final_frozen_state_sha256=frozen_state_digest(model))
        write_json(output/'provenance.json', provenance)
        if not args.smoke:
            torch.save(dict(net=model.state_dict(), config=config, steps=updates,
                            optimizer=optimizer.state_dict(), provenance=provenance), output/'model.pt')
        write_json(output/'summary.json', dict(provenance=provenance, initial=history[0], final=history[-1]))
        if tracker: tracker.finish()
    except BaseException:
        if tracker: tracker.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
