#!/usr/bin/env python3
"""Paired, bounded spline-weight trial using existing training/evaluation code."""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from configs import validate_config
from data.loaders import build_data_loaders, build_evaluation_loader, dataset_sample_manifest
from metrics.common_vessel_metrics import evaluate_common_vessels, aggregate_common_vessels
from models import build_model
from models.checkpoint import load_legacy_model_checkpoint, load_checkpoint_file
from training.engine import train_step, evaluate_loss, _move_graph_batch
from training.evaluation import evaluate_model
from training.evaluation.spline_visualization import draw_projections
from training.losses.criterion import build_criterion
from training.optim import build_optimizer, build_scheduler
from training.tracking import build_tracker


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


@contextmanager
def monitoring_rng():
    """Monitoring must not change the training random stream."""
    py, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        seed(9871)
        try:
            yield
        finally:
            random.setstate(py); np.random.set_state(np_state)


@torch.no_grad()
def sensitivity(model, batch, config):
    model.eval()
    volumes, _ = _move_graph_batch(batch, 'cuda', config['training']['input'], True)
    memory = []
    hook = model.decoder.encoder.register_forward_hook(lambda m, a, out: memory.append(out.detach()))
    try:
        tokens, predictions, features = model(volumes)
    finally:
        hook.remove()
    return {'node_pair_max_delta': float((predictions['pred_nodes'][0] - predictions['pred_nodes'][1]).abs().max()),
            'token_pair_mae': float((tokens[0] - tokens[1]).abs().mean()),
            'cnn_pair_mae': float((features[0] - features[1]).abs().mean()),
            'transformer_memory_pair_mae': float((memory[0][0] - memory[0][1]).abs().mean())}


def geometry(model, loader, config, output, step, tracker, subset_epoch, score_chords=False, split='val'):
    destination = output / (f'validation-{step:06d}' if split == 'val' else f'test-{step:06d}' if split == 'test' else f'train-evaluation-{step:06d}')
    native, _ = evaluate_model(model, loader, config, 'cuda', output_dir=destination)
    records = json.loads((destination / 'predictions.json').read_text())
    assert [r['source_sample_id'] for r in records] == [r.sample_id for r in loader.dataset.records]
    root = Path(config['data']['datasets']['synthetic_mri']['root'])
    common_rows = [dict(sample_id=r['source_sample_id'], **evaluate_common_vessels(
        {'nodes': r['nodes_dhw'], 'edges': r['edges'], 'spline_controls': r['spline_controls'],
         'spline_family': r.get('spline_family', 'bspline')},
        root / split / 'spline' / (r['source_sample_id'] + '.npz'))) for r in records]
    common = aggregate_common_vessels(common_rows)
    if score_chords:
        chord_rows = [dict(sample_id=r['source_sample_id'], **evaluate_common_vessels(
            {'nodes': r['nodes_dhw'], 'edges': r['edges']},
            root / split / 'spline' / (r['source_sample_id'] + '.npz'))) for r in records]
        chord_summary = aggregate_common_vessels(chord_rows)
        write_json(destination / 'same-endpoint-chords-summary.json', chord_summary)
        common['same_endpoint_chord_common_curve_f1'] = chord_summary['common_curve_f1']
    write_json(destination / 'common-v2-summary.json', common)
    write_json(destination / 'common-v2-per-patch.json', common_rows)
    paths = []
    # Two fixed examples from different subjects, without quality selection.
    for index in (0, len(records) - 1):
        image, mask, nodes, _, guides, _ = loader.dataset[index]
        record = records[index]
        path = destination / (record['source_sample_id'] + '.png')
        draw_projections(image[0].squeeze().numpy(), mask[0].squeeze().numpy(), nodes[0].numpy(),
                         guides[0]['guides'].numpy(),
                         {'nodes': record['nodes_dhw'], 'spline_curves': record['spline_curves']}, path,
                         f"{config['experiment']['name']} · update {step} · {record['source_sample_id']}")
        paths.append(path)
    if tracker:
        tracker.log_validation_images(paths, epoch=subset_epoch)
        tracker.log_metrics({**native, **common}, iteration=step, epoch=subset_epoch)
    return {'native_node_f1': native['node_f1'], 'native_edge_f1': native['edge_f1'],
            **({'same_endpoint_chord_common_curve_f1': common['same_endpoint_chord_common_curve_f1']} if score_chords else {}),
            **{k: common[k] for k in ('common_anchor_f1', 'common_branch_f1', 'common_curve_f1',
                                      'common_guide_distance_mm', 'common_branch_coverage')}}



def assert_scratch_trainability(model):
    # Legacy relation cross-attention is stored for checkpoint compatibility but
    # deliberately absent from forward(). All active parameters must train.
    expected = set()
    for name, module in model.named_modules():
        if getattr(module, 'rln_attn', False):
            for child in ('cross_attn2', 'norm12'):
                expected.update(f'{name}.{child}.{n}' for n, _ in getattr(module, child).named_parameters())
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert frozen == expected, ('Unexpected scratch trainability', sorted(frozen ^ expected))
    return sorted(frozen)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/spline_weight_pilot.yaml', type=Path)
    p.add_argument('--arm', required=True)
    p.add_argument('--smoke', action='store_true', help='Three updates and four validation cases; separate output, no checkpoint')
    args = p.parse_args()
    pilot = yaml.safe_load(args.config.read_text())
    frozen = bool(pilot.get('freeze_detector', False))
    scratch_geometry = bool(pilot.get('scratch_geometry', False))
    if scratch_geometry:
        assert not frozen and not pilot.get('initial_weights') and not pilot.get('continuation_checkpoint')
        assert not pilot.get('schedule_start_update', 0)
    continuation = bool(pilot.get('continuation_checkpoint'))
    if continuation:
        assert frozen and pilot.get('reference_run')
    if args.arm not in pilot['arms']:
        p.error('Arm must be one of ' + ', '.join(pilot['arms']))
    config = yaml.safe_load(Path(pilot['base_config']).read_text())
    config['experiment'].update(seed=pilot['seed'], name=pilot.get('name_prefix', 'spline_weight_pilot') + '_' + args.arm)
    config['runtime']['workers'] = 4
    settings = config['data']['datasets']['synthetic_mri']
    settings.update(train_samples=pilot['train_samples'], validation_samples=pilot['validation_samples'],
                    sample_cap_selection='seeded_random', sample_cap_seed=pilot['seed'])
    config['loss']['weights']['spline'] = pilot['arms'][args.arm]
    variant = pilot.get('variants', {}).get(args.arm, {})
    if variant:
        assert frozen or scratch_geometry, 'Variants require frozen or explicit scratch diagnostic'
        config['loss']['spline']['target'] = variant['loss_target']
        config['model']['spline']['token_inputs'] = variant['token_inputs']
    if pilot.get('local_geometry_variants'):
        assert (frozen and continuation) or scratch_geometry
        config['model']['spline']['local_geometry'] = dict(
            enabled=True, channels=64, samples=32, **pilot['local_geometry_variants'][args.arm])
    if 'image_evidence_scales' in pilot:
        config['model']['spline']['curve_evidence'] = dict(
            enabled=True, channels=8, samples=32, chunk_size=256,
            scale=pilot['image_evidence_scales'][args.arm])
    config['tracking'].update(enabled=not args.smoke, mode='offline', group=pilot.get('tracking_group', 'spline-weight-pilot-20261008'),
                              tags=['gnbm', 'spline', 'weight-pilot', args.arm])
    assert config['data']['batch_size'] == 4 and not config['data']['train_augmentation']
    assert config['training']['epochs'] == 300 and config['training']['warmup_epochs'] == 1
    validate_config(config)
    output = Path(pilot['output_root']) / (('smoke-' if args.smoke else '') + args.arm)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'resolved-config.yaml').write_text(yaml.safe_dump(config))
    write_json(output / 'pilot-config.json', pilot)
    seed(pilot['seed'])
    train, val = build_data_loaders(config)
    membership = {'train': dataset_sample_manifest(train.dataset), 'val': dataset_sample_manifest(val.dataset)}
    train_ids = membership['train'][0]['sample_ids']; val_ids = membership['val'][0]['sample_ids']
    original = json.loads(Path(pilot['population_manifest']).read_text())
    assert len(train_ids) == pilot['train_samples'] and len(val_ids) == pilot['validation_samples']
    assert set(train_ids) <= set(original['train_ids']) and set(val_ids) <= set(original['validation_ids'])
    train_subjects = {s.split('_')[1] for s in train_ids}; val_subjects = {s.split('_')[1] for s in val_ids}
    assert train_subjects.isdisjoint(val_subjects) and len(train_subjects) >= 50 and len(val_subjects) >= 10
    if frozen:
        prior_manifest = json.loads(Path(pilot['pretraining_manifest']).read_text())
        prior_train = {s for leaf in prior_manifest['train'] for s in leaf['sample_ids']}
        assert set(train_ids) <= prior_train
        assert val_subjects.isdisjoint({s.split('_')[1] for s in prior_train})
    write_json(output / 'dataset-manifest.json', membership)
    expanded_training = pilot.get('reference_comparison', 'paired') == 'expanded_training'
    assert pilot.get('reference_comparison', 'paired') in ('paired', 'expanded_training')
    train_eval_ids = train_ids
    if pilot.get('evaluation_membership_run'):
        assert (frozen or scratch_geometry) and not expanded_training
        evaluation_ids = json.loads((Path(pilot['evaluation_membership_run']) / 'evaluation-sample-ids.json').read_text())
        train_eval_ids = evaluation_ids['train']
        assert set(train_eval_ids) <= set(train_ids)
        assert evaluation_ids['val'] == val_ids
    if expanded_training:
        assert frozen and pilot.get('reference_run')
        reference_membership = json.loads((Path(pilot['reference_run']) / 'dataset-manifest.json').read_text())
        train_eval_ids = reference_membership['train'][0]['sample_ids']
        assert set(train_eval_ids) < set(train_ids), 'Expected a strict expansion of training membership'
        assert set(train_ids) == set(original['train_ids']), 'Expected the existing full 10k population'
        assert membership['val'] == reference_membership['val'], 'Validation membership/order changed'
    write_json(output / 'evaluation-sample-ids.json', {
        'train': train_eval_ids[:4] if args.smoke else train_eval_ids,
        'val': val_ids[:4] if args.smoke else val_ids})
    # Evaluation needs concrete records, rather than a ConcatDataset wrapper.
    if args.smoke:
        val_ids = val_ids[:4]
    val = build_evaluation_loader(config, dataset_name='synthetic_mri', split='val', sample_ids=val_ids)
    train_eval = None
    if frozen or scratch_geometry:
        from scripts.frozen_spline_diagnostic import training_evaluation_loader
        train_eval = training_evaluation_loader(train, config, train_eval_ids[:4] if args.smoke else train_eval_ids)
    probe_ids = [val_ids[0], val_ids[-1], val_ids[1], val_ids[-2]]
    probe = next(iter(build_evaluation_loader(config, dataset_name='synthetic_mri', split='val', sample_ids=probe_ids)))
    seed(pilot['seed'])
    model = build_model(config)
    initial_checkpoint = pilot.get('initial_weights')
    if initial_checkpoint:
        checkpoint = load_checkpoint_file(initial_checkpoint)
        assert int(checkpoint['epoch']) == int(pilot['initial_epoch'])
        del checkpoint
        report = load_legacy_model_checkpoint(model, initial_checkpoint, allow_missing_spline_head=True)
        assert report.missing and all(k.startswith('spline_head.') for k in report.missing)
    if continuation:
        checkpoint = load_checkpoint_file(pilot['continuation_checkpoint'])
        assert checkpoint.get('cumulative_head_updates', checkpoint['steps']) == pilot['continuation_start_update']
        for section in ('model', 'loss'):
            compared = copy.deepcopy(config[section])
            if section == 'model' and pilot.get('local_geometry_variants'):
                compared['spline'].pop('local_geometry')
            assert checkpoint['config'][section] == compared, ('Continuation config differs', section)
        report = model.load_state_dict(checkpoint['net'], strict=False)
        expected_missing = [n for n in model.state_dict() if n.startswith('spline_head.local_geometry.')]
        assert set(report.missing_keys) == set(expected_missing) and not report.unexpected_keys, report
        del checkpoint
    initial = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        initial.update(name.encode()); initial.update(value.numpy().tobytes())
    model = model.cuda()
    frozen_digest = None
    if frozen:
        from scripts.frozen_spline_diagnostic import freeze_detector, detector_digest, frozen_step, matched_geometry
        assert initial_checkpoint and config['model']['matcher']['type'] == 'hungarian'
        assert config['loss']['spline']['mode'] == 'controls'
        freeze_detector(model)
        frozen_digest = detector_digest(model)
    criterion = build_criterion(config, model).cuda()
    if scratch_geometry:
        from scripts.controlled_spline_unfreeze import RoutedCriterion
        from scripts.frozen_spline_diagnostic import matched_geometry
        from models.matcher import build_matcher
        scratch_inactive_parameters = assert_scratch_trainability(model)
        criterion = RoutedCriterion(config, build_matcher(config), model.relation_embed,
                                    spline_head=model.spline_head, route=pilot.get('scratch_gradient_route', 'tokens_only')).cuda()
    val_config = copy.deepcopy(config)
    val_config['loss']['edge']['balancing']['mode'] = 'none'
    val_config['loss']['edge']['candidates']['positive_cap'] = 40
    val_criterion = build_criterion(val_config, model, validation=True).cuda()
    optimizer = build_optimizer(config, model)
    scheduler = (torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.) if frozen
                 else build_scheduler(config, optimizer, pilot['schedule_iterations_per_epoch']))
    schedule_start = int(pilot.get('schedule_start_update', 0))
    if schedule_start:
        # New AdamW moments in BOTH continuation arms; keep the LR trajectory
        # at the source graph checkpoint's update, rather than restarting it.
        scheduler.step(schedule_start)
    updates = 3 if args.smoke else pilot['updates']
    provenance = {'initial_state_sha256': initial.hexdigest(), 'train_subjects': len(train_subjects),
                  'validation_subjects': len(val_subjects), 'weight': pilot['arms'][args.arm],
                  'updates': updates, 'schedule_updates': 750000, 'warmup_updates': 2500,
                  'bn_recalibration_batches': 0, 'probe_ids': probe_ids,
                  'initial_checkpoint': initial_checkpoint,
                  'initial_epoch': pilot.get('initial_epoch', 0),
                  'schedule_start_update': schedule_start,
                  'spline_ramp_updates': int(pilot.get('spline_ramp_updates', 0)),
                  'image_evidence_scale': pilot.get('image_evidence_scales', {}).get(args.arm),
                  'base_config_sha256': hashlib.sha256(Path(pilot['base_config']).read_bytes()).hexdigest()}
    if scratch_geometry:
        provenance.update(initialization='random_all_modules_no_checkpoint',
                          gradient_route=pilot.get('scratch_gradient_route', 'tokens_only'),
                          inactive_legacy_parameters=scratch_inactive_parameters,
                          bn_mode='train with normal training updates; eval with saved statistics',
                          training_samples=len(train_ids), training_evaluation_samples=len(train_eval_ids))
    if frozen:
        assert schedule_start == 0 and not pilot.get('spline_ramp_updates', 0)
        provenance.update(freeze_detector=True,detector_sha256=frozen_digest,
                          trainable_parameters=[n for n,p in model.named_parameters() if p.requires_grad],
                          schedule_updates=None,warmup_updates=0,learning_rate=config['training']['optimizer']['base_lr'],
                          scheduler='constant head-only LR',bn_mode='eval throughout',
                          pretraining_manifest=pilot['pretraining_manifest'])
        provenance.update(loss_target=config['loss']['spline'].get('target', 'absolute'),
                          token_inputs=config['model']['spline'].get('token_inputs', 'full'))
    if continuation:
        provenance.update(continuation_checkpoint=pilot['continuation_checkpoint'],
                          continuation_start_update=pilot['continuation_start_update'],
                          optimizer_restart=True, training_shuffle_restart=True)
    if pilot.get('local_geometry_variants'):
        provenance['local_geometry'] = config['model']['spline']['local_geometry']
    reference = None
    source_geometry_reference = None
    if pilot.get('local_geometry_variants') and continuation:
        source_geometry_reference = json.loads((Path(pilot['reference_run'])/'summary.json').read_text())['final']
    if pilot.get('reference_run'):
        reference_path = Path(pilot['reference_run'])
        if args.smoke:
            reference_path = reference_path.parent / ('smoke-' + reference_path.name)
        reference = json.loads((reference_path / 'summary.json').read_text())
        for key in ('initial_state_sha256', 'detector_sha256', 'weight', 'updates', 'learning_rate',
                    'initial_checkpoint', 'base_config_sha256', 'scheduler', 'bn_mode', 'probe_ids'):
            if continuation and key in ('initial_state_sha256', 'updates'):
                continue
            assert provenance[key] == reference['provenance'][key], ('Reference differs', key)
        if expanded_training or continuation:
            for key in ('loss_target', 'token_inputs'):
                assert provenance[key] == reference['provenance'][key], ('Reference differs', key)
        if not expanded_training:
            assert membership == json.loads((reference_path / 'dataset-manifest.json').read_text())
        provenance['reference_run'] = str(reference_path)
        provenance['reference_comparison'] = ('weight_continuation' if continuation else
                                              'expanded_training' if expanded_training else 'paired')
        provenance['training_samples'] = len(train_ids)
        provenance['training_evaluation_samples'] = len(train_eval_ids[:4] if args.smoke else train_eval_ids)
    scratch_reference = None
    if pilot.get('scratch_reference_run'):
        assert scratch_geometry
        reference_path = Path(pilot['scratch_reference_run'])
        if args.smoke:
            reference_path = reference_path.parent / ('smoke-' + reference_path.name)
        scratch_reference = json.loads((reference_path / 'summary.json').read_text())
        for key in ('initial_state_sha256', 'initial_checkpoint', 'updates', 'base_config_sha256',
                    'probe_ids', 'schedule_updates', 'warmup_updates', 'bn_mode'):
            assert provenance[key] == scratch_reference['provenance'][key], ('Scratch reference differs', key)
        assert membership == json.loads((reference_path / 'dataset-manifest.json').read_text())
        assert json.loads((output/'evaluation-sample-ids.json').read_text()) == json.loads(
            (reference_path/'evaluation-sample-ids.json').read_text())
        reference_config = yaml.safe_load((reference_path/'resolved-config.yaml').read_text())
        for section in ('model', 'data', 'training', 'evaluation'):
            assert config[section] == reference_config[section], ('Scratch config differs', section)
        lhs, rhs = copy.deepcopy(config['loss']), copy.deepcopy(reference_config['loss'])
        lhs['weights'].pop('spline'); rhs['weights'].pop('spline')
        assert lhs == rhs
        provenance['scratch_reference_run'] = str(reference_path)
    write_json(output / 'provenance.json', provenance)
    tracker = build_tracker(config, output, resume=False, launch_metadata=provenance)
    seed(pilot['seed'])
    history = []; sums = {}; count = 0
    started = time.monotonic()
    fixed_assignments = {}

    def monitor(step):
        with monitoring_rng():
            row = {'update': step, 'elapsed_seconds': time.monotonic() - started,
                   'effective_spline_weight': criterion.weights['spline'],
                   'validation': evaluate_loss(model, val_criterion, val, config, 'cuda'),
                   'sensitivity': sensitivity(model, probe, config)}
            if continuation:
                row['cumulative_head_updates'] = pilot['continuation_start_update'] + step
            if count:
                row['training'] = {k: v / count for k, v in sums.items()}
            if step in pilot['geometry_steps'] or (args.smoke and step == updates):
                row['geometry'] = geometry(model, val, config, output, step, tracker, step // len(train), pilot.get('score_chords', False))
                if frozen or scratch_geometry:
                    if frozen:
                        assert detector_digest(model) == frozen_digest, 'Frozen detector parameters or buffers changed'
                    row['train_geometry'] = geometry(model, train_eval, config, output, step, None, step // len(train), True, split='train')
                    for label,loader in [('train',train_eval),('val',val)]:
                        report = matched_geometry(model,val_criterion,loader,config,'cuda',
                                                  curvature_breakdown=bool(pilot.get('local_geometry_variants')))
                        write_json(output / f'{label}-matched-geometry-{step:06d}.json',report)
                        report.pop('rows')
                        row[label+'_matched_geometry'] = report
                        if frozen:
                            fixed_assignments.setdefault(label,report['assignment_sha256'])
                            assert report['assignment_sha256'] == fixed_assignments[label], 'Frozen matching changed'
                    if frozen:
                        row['detector_unchanged'] = True
                    if source_geometry_reference is not None and step == 0 and not args.smoke:
                        for label in ('geometry', 'train_geometry'):
                            for key, value in row[label].items():
                                previous = source_geometry_reference[label][key]
                                assert value == previous or (value is not None and previous is not None and
                                    abs(value-previous) < 1e-5), ('Initial local geometry changed source prediction', label, key)
                    if reference is not None:
                        for label in ('train', 'val'):
                            assert fixed_assignments[label] == reference['initial'][label+'_matched_geometry']['assignment_sha256']
        history.append(row)
        with (output / 'history.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        if tracker:
            tracker.log_validation(row['validation'], iteration=step, epoch=step // len(train))
            tracker.log_metrics(row['sensitivity'], iteration=step, epoch=step // len(train))
            if (frozen or scratch_geometry) and 'train_matched_geometry' in row:
                tracker.log_metrics({f'{split}_matched/{key}':value for split in ('train','val')
                                     for key,value in row[split+'_matched_geometry']['all_pairs'].items() if value is not None},
                                    iteration=step,epoch=step // len(train))
        print(json.dumps(row), flush=True)

    try:
        if pilot.get('spline_ramp_updates', 0):
            criterion.weights['spline'] = 0.
        monitor(0)
        iterator = iter(train)
        order_digest = hashlib.sha256()
        for step in range(1, updates + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train); batch = next(iterator)
            # Hash actual target batches to verify identical ordering between arms.
            for nodes in batch[2]:
                order_digest.update(nodes.numpy().tobytes())
            ramp = int(pilot.get('spline_ramp_updates', 0))
            criterion.weights['spline'] = pilot['arms'][args.arm] * (min(1., step / ramp) if ramp else 1.)
            step_function = frozen_step if frozen else train_step
            losses = step_function(model, criterion, optimizer, scheduler, batch, config, 'cuda',
                                epoch=1 + (schedule_start + step - 1) // pilot['schedule_iterations_per_epoch'],
                                iteration=schedule_start + step, total_iterations=750000)
            scalars = {k: float(v.detach()) for k, v in losses.items() if torch.is_tensor(v) and v.numel() == 1}
            for k, v in scalars.items():
                sums[k] = sums.get(k, 0.) + v
            count += 1
            if step == 1:
                gradients = [p.grad for p in model.spline_head.parameters() if p.grad is not None]
                assert gradients and all(torch.isfinite(g).all() for g in gradients)
                if criterion.weights['spline']:
                    assert any(g.abs().sum() > 0 for g in gradients)
                else:
                    assert all(g.abs().sum() == 0 for g in gradients)
                local = model.spline_head.local_geometry
                if local is not None:
                    local_grads = [p.grad for p in local.parameters() if p.grad is not None]
                    assert local_grads and all(torch.isfinite(g).all() for g in local_grads)
                    if local.scale and criterion.weights['spline']:
                        assert local.scorer[-1].bias.grad.abs().sum() > 0
                    else:
                        assert all(g.abs().sum() == 0 for g in local_grads)
                if scratch_geometry:
                    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                print('First forward/loss/backward/update passed', flush=True)
            if tracker and step % 100 == 0:
                scalars['effective_spline_weight'] = criterion.weights['spline']
                tracker.log_training(scalars, iteration=step, epoch=1 + (step - 1) // len(train),
                                     learning_rate=optimizer.param_groups[0]['lr'])
            if step % pilot['monitor_interval'] == 0 or step == updates:
                monitor(step); sums = {}; count = 0
            if not args.smoke and step in pilot.get('checkpoint_steps', []) and step != updates:
                torch.save({'net': model.state_dict(), 'config': config, 'steps': step,
                            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()},
                           output / f'model-{step:06d}.pt')
        if scratch_geometry and not pilot['arms'][args.arm]:
            # Zero-loss control must still decode exact chords after AdamW updates.
            for layer in (model.spline_head.network[-1], model.spline_head.local_geometry.scorer[-1]):
                assert torch.count_nonzero(layer.weight) == 0 and torch.count_nonzero(layer.bias) == 0
        provenance['batch_order_sha256'] = order_digest.hexdigest()
        if reference is not None:
            if expanded_training or continuation:
                provenance['reference_fixed_conditions_verified'] = True
                provenance['reference_batch_order_comparable'] = False
            else:
                assert provenance['batch_order_sha256'] == reference['provenance']['batch_order_sha256'], 'Reference batch order differs'
                provenance['reference_pairing_verified'] = True
        if scratch_reference is not None:
            assert provenance['batch_order_sha256'] == scratch_reference['provenance']['batch_order_sha256']
            provenance['scratch_pairing_verified'] = True
        provenance['elapsed_seconds'] = time.monotonic() - started
        write_json(output / 'provenance.json', provenance)
        if not args.smoke:
            torch.save({'net': model.state_dict(), 'config': config, 'steps': updates,
                        'cumulative_head_updates': pilot.get('continuation_start_update', 0) + updates,
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, output / 'model.pt')
        write_json(output / 'summary.json', {'arm': args.arm, 'provenance': provenance, 'initial': history[0], 'final': history[-1]})
        if tracker:
            tracker.finish()
    except BaseException:
        if tracker:
            tracker.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
