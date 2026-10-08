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


def geometry(model, loader, config, output, step, tracker, subset_epoch):
    destination = output / f'validation-{step:06d}'
    native, _ = evaluate_model(model, loader, config, 'cuda', output_dir=destination)
    records = json.loads((destination / 'predictions.json').read_text())
    assert [r['source_sample_id'] for r in records] == [r.sample_id for r in loader.dataset.records]
    root = Path(config['data']['datasets']['synthetic_mri']['root'])
    common_rows = [dict(sample_id=r['source_sample_id'], **evaluate_common_vessels(
        {'nodes': r['nodes_dhw'], 'edges': r['edges'], 'spline_controls': r['spline_controls']},
        root / 'val/spline' / (r['source_sample_id'] + '.npz'))) for r in records]
    common = aggregate_common_vessels(common_rows)
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
            **{k: common[k] for k in ('common_anchor_f1', 'common_branch_f1', 'common_curve_f1',
                                      'common_guide_distance_mm', 'common_branch_coverage')}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/spline_weight_pilot.yaml', type=Path)
    p.add_argument('--arm', required=True, choices=('original', 'reduced'))
    p.add_argument('--smoke', action='store_true', help='Three updates and four validation cases; separate output, no checkpoint')
    args = p.parse_args()
    pilot = yaml.safe_load(args.config.read_text())
    config = yaml.safe_load(Path(pilot['base_config']).read_text())
    config['experiment'].update(seed=pilot['seed'], name='spline_weight_pilot_' + args.arm)
    config['runtime']['workers'] = 4
    settings = config['data']['datasets']['synthetic_mri']
    settings.update(train_samples=pilot['train_samples'], validation_samples=pilot['validation_samples'],
                    sample_cap_selection='seeded_random', sample_cap_seed=pilot['seed'])
    config['loss']['weights']['spline'] = pilot['arms'][args.arm]
    config['tracking'].update(enabled=not args.smoke, mode='offline', group='spline-weight-pilot-20261008',
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
    assert len(train_ids) == 512 and len(val_ids) == 128
    assert set(train_ids) <= set(original['train_ids']) and set(val_ids) <= set(original['validation_ids'])
    train_subjects = {s.split('_')[1] for s in train_ids}; val_subjects = {s.split('_')[1] for s in val_ids}
    assert train_subjects.isdisjoint(val_subjects) and len(train_subjects) >= 50 and len(val_subjects) >= 10
    write_json(output / 'dataset-manifest.json', membership)
    # Evaluation needs concrete records, rather than a ConcatDataset wrapper.
    if args.smoke:
        val_ids = val_ids[:4]
    val = build_evaluation_loader(config, dataset_name='synthetic_mri', split='val', sample_ids=val_ids)
    probe_ids = [val_ids[0], val_ids[-1], val_ids[1], val_ids[-2]]
    probe = next(iter(build_evaluation_loader(config, dataset_name='synthetic_mri', split='val', sample_ids=probe_ids)))
    seed(pilot['seed'])
    model = build_model(config)
    initial = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        initial.update(name.encode()); initial.update(value.numpy().tobytes())
    model = model.cuda()
    criterion = build_criterion(config, model).cuda()
    val_config = copy.deepcopy(config)
    val_config['loss']['edge']['balancing']['mode'] = 'none'
    val_config['loss']['edge']['candidates']['positive_cap'] = 40
    val_criterion = build_criterion(val_config, model, validation=True).cuda()
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, pilot['schedule_iterations_per_epoch'])
    updates = 3 if args.smoke else pilot['updates']
    provenance = {'initial_state_sha256': initial.hexdigest(), 'train_subjects': len(train_subjects),
                  'validation_subjects': len(val_subjects), 'weight': pilot['arms'][args.arm],
                  'updates': updates, 'schedule_updates': 750000, 'warmup_updates': 2500,
                  'bn_recalibration_batches': 0, 'probe_ids': probe_ids,
                  'base_config_sha256': hashlib.sha256(Path(pilot['base_config']).read_bytes()).hexdigest()}
    write_json(output / 'provenance.json', provenance)
    tracker = build_tracker(config, output, resume=False, launch_metadata=provenance)
    seed(pilot['seed'])
    history = []; sums = {}; count = 0
    started = time.monotonic()

    def monitor(step):
        with monitoring_rng():
            row = {'update': step, 'elapsed_seconds': time.monotonic() - started,
                   'validation': evaluate_loss(model, val_criterion, val, config, 'cuda'),
                   'sensitivity': sensitivity(model, probe, config)}
            if count:
                row['training'] = {k: v / count for k, v in sums.items()}
            if step in pilot['geometry_steps'] or (args.smoke and step == updates):
                row['geometry'] = geometry(model, val, config, output, step, tracker, step // len(train))
        history.append(row)
        with (output / 'history.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        if tracker:
            tracker.log_validation(row['validation'], iteration=step, epoch=step // len(train))
            tracker.log_metrics(row['sensitivity'], iteration=step, epoch=step // len(train))
        print(json.dumps(row), flush=True)

    try:
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
            losses = train_step(model, criterion, optimizer, scheduler, batch, config, 'cuda',
                                epoch=1 + (step - 1) // pilot['schedule_iterations_per_epoch'],
                                iteration=step, total_iterations=750000)
            scalars = {k: float(v.detach()) for k, v in losses.items() if torch.is_tensor(v) and v.numel() == 1}
            for k, v in scalars.items():
                sums[k] = sums.get(k, 0.) + v
            count += 1
            if step == 1:
                gradients = [p.grad for p in model.spline_head.parameters() if p.grad is not None]
                assert gradients and all(torch.isfinite(g).all() for g in gradients) and any(g.abs().sum() > 0 for g in gradients)
                print('First forward/loss/backward/update passed', flush=True)
            if tracker and step % 100 == 0:
                tracker.log_training(scalars, iteration=step, epoch=1 + (step - 1) // len(train),
                                     learning_rate=optimizer.param_groups[0]['lr'])
            if step % pilot['monitor_interval'] == 0 or step == updates:
                monitor(step); sums = {}; count = 0
        provenance['batch_order_sha256'] = order_digest.hexdigest()
        provenance['elapsed_seconds'] = time.monotonic() - started
        write_json(output / 'provenance.json', provenance)
        if not args.smoke:
            torch.save({'net': model.state_dict(), 'config': config, 'steps': updates}, output / 'model.pt')
        write_json(output / 'summary.json', {'arm': args.arm, 'provenance': provenance, 'initial': history[0], 'final': history[-1]})
        if tracker:
            tracker.finish()
    except BaseException:
        if tracker:
            tracker.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
