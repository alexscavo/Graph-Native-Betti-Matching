#!/usr/bin/env python3
"""Continue the paired scratch pilots without resetting AdamW or the LR schedule."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
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
from training.engine import train_step, evaluate_loss
from training.losses.criterion import build_criterion
from training.optim import build_optimizer, build_scheduler
from training.tracking import build_tracker
from scripts.controlled_spline_unfreeze import RoutedCriterion
from scripts.frozen_spline_diagnostic import training_evaluation_loader, matched_geometry
from scripts.train_spline_weight_pilot import (
    seed, monitoring_rng, sensitivity, geometry, write_json, assert_scratch_trainability,
)


def restore_optimizer(checkpoint, optimizer, scheduler):
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    assert scheduler.last_epoch == checkpoint['steps']
    assert scheduler.get_last_lr() == [g['lr'] for g in optimizer.param_groups]


def capture_rng(loader):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                loader=loader.generator.get_state())


def restore_rng(state, loader):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])
    loader.generator.set_state(state['loader'])


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/spline_scratch_extend300.yaml'))
    parser.add_argument('--arm', required=True)
    parser.add_argument('--stage', type=int, choices=(150, 300), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plan = yaml.safe_load(args.config.read_text())
    arm = plan['arms'][args.arm]
    root = Path(plan['output_root']) / args.arm
    previous = Path(arm['source_run']) if args.stage == 150 else root / 'epoch150'
    checkpoint = load_checkpoint_file(previous / 'model.pt')
    start = checkpoint['steps']
    assert start == (25000 if args.stage == 150 else 375000)
    config = copy.deepcopy(checkpoint['config'])
    config['experiment']['name'] = f'scratch_extend300_{args.arm}_epoch{args.stage}'
    config['tracking'].update(enabled=not args.smoke, mode='offline', group=plan['tracking_group'])
    assert config['training']['epochs'] == 300 and config['data']['batch_size'] == 4
    assert not config['data']['train_augmentation']
    assert config['loss']['weights']['spline'] == arm['weight']
    validate_config(config)
    output = root / (('smoke-' if args.smoke else '') + f'epoch{args.stage}')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'resolved-config.yaml').write_text(yaml.safe_dump(config))
    write_json(output / 'extension-config.json', plan)
    seed(config['experiment']['seed'])
    train, _ = build_data_loaders(config)
    assert len(train) == 2500
    membership = json.loads((previous / 'dataset-manifest.json').read_text())
    assert dataset_sample_manifest(train.dataset) == membership['train']
    ids = json.loads((previous / 'evaluation-sample-ids.json').read_text())
    assert len(ids['train']) == 512 and len(ids['val']) == 128
    write_json(output / 'dataset-manifest.json', membership)
    write_json(output / 'evaluation-sample-ids.json', ids)
    val_ids = ids['val'][:4] if args.smoke else ids['val']
    val = build_evaluation_loader(config, dataset_name='synthetic_mri', split='val', sample_ids=val_ids)
    assert dataset_sample_manifest(val.dataset)[0]['sample_ids'] == val_ids
    train_eval = training_evaluation_loader(train, config, ids['train'][:4] if args.smoke else ids['train'])
    probe_ids = [ids['val'][0], ids['val'][-1], ids['val'][1], ids['val'][-2]]
    probe = next(iter(build_evaluation_loader(config, dataset_name='synthetic_mri', split='val', sample_ids=probe_ids)))
    model = build_model(config).cuda()
    assert_scratch_trainability(model)
    model.load_state_dict(checkpoint['net'], strict=True)
    criterion = RoutedCriterion(config, build_matcher(config), model.relation_embed,
                                spline_head=model.spline_head, route=arm['route']).cuda()
    val_config = copy.deepcopy(config)
    val_config['loss']['edge']['balancing']['mode'] = 'none'
    val_config['loss']['edge']['candidates']['positive_cap'] = 40
    val_criterion = build_criterion(val_config, model, validation=True).cuda()
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, len(train))
    restore_optimizer(checkpoint, optimizer, scheduler)
    best_score = tuple(checkpoint.get('best_score', (-1., -1.)))
    best_update = checkpoint.get('best_update')
    if args.stage == 300 and not args.smoke:
        shutil.copyfile(previous / 'best.pt', output / 'best.pt')
    provenance = dict(source_checkpoint=str(previous / 'model.pt'), start_update=start,
        target_update=args.stage * len(train), optimizer_restored=True, scheduler_restored=True,
        start_lr=optimizer.param_groups[0]['lr'], gradient_route=arm['route'], spline_weight=arm['weight'],
        training_samples=10000, validation_samples=128, train_evaluation_samples=512,
        rng_restored='rng' in checkpoint, shuffle_restart='rng' not in checkpoint,
        best_selection='maximum validation common_branch_f1, then native_edge_f1; includes source checkpoint',
        bn_mode='train during updates; saved running statistics during evaluation')
    write_json(output / 'provenance.json', provenance)
    tracker = build_tracker(config, output, resume=False, launch_metadata=provenance)
    # Old pilots did not save random states. Both arms restart the same stream.
    # Subsequent stages resume at an exhausted epoch boundary, including sampler state.
    if 'rng' in checkpoint:
        restore_rng(checkpoint['rng'], train)
    else:
        seed(config['experiment']['seed'])
        train.generator.manual_seed(config['experiment']['seed'])
    del checkpoint
    started = time.monotonic()
    history = []; sums = {}; count = 0
    order_digest = hashlib.sha256()
    target = start + 3 if args.smoke else args.stage * len(train)

    def payload(step):
        return dict(net=model.state_dict(), config=config, steps=step, epoch=step // len(train),
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    rng=capture_rng(train), best_score=best_score, best_update=best_update)

    def monitor(step, detailed=False):
        nonlocal best_score, best_update
        with monitoring_rng():
            row = dict(update=step, epoch=step / len(train), elapsed_seconds=time.monotonic()-started,
                       learning_rate=optimizer.param_groups[0]['lr'],
                       validation=evaluate_loss(model, val_criterion, val, config, 'cuda'),
                       sensitivity=sensitivity(model, probe, config),
                       geometry=geometry(model, val, config, output, step, tracker, step // len(train), True))
            if count:
                row['training'] = {k: v/count for k,v in sums.items()}
            if detailed:
                row['train_geometry'] = geometry(model, train_eval, config, output, step, None,
                                                  step // len(train), True, split='train')
                for label, loader in (('train', train_eval), ('val', val)):
                    report = matched_geometry(model, val_criterion, loader, config, 'cuda', curvature_breakdown=True)
                    write_json(output / f'{label}-matched-geometry-{step:06d}.json', report)
                    report.pop('rows')
                    row[label+'_matched_geometry'] = report
        score = (row['geometry']['common_branch_f1'], row['geometry']['native_edge_f1'])
        if score > best_score:
            best_score, best_update = score, step
            if not args.smoke:
                atomic_checkpoint(output / 'best.pt', payload(step))
        row.update(best_update=best_update, best_score=best_score)
        history.append(row)
        with (output / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if tracker:
            tracker.log_validation(row['validation'], iteration=step, epoch=step // len(train))
            tracker.log_metrics(row['sensitivity'], iteration=step, epoch=step // len(train))
            if detailed:
                tracker.log_metrics({f'{split}_matched/{key}': value for split in ('train','val')
                    for key,value in row[split+'_matched_geometry']['all_pairs'].items() if value is not None},
                    iteration=step, epoch=step // len(train))
        print(json.dumps(row), flush=True)

    try:
        if not args.smoke:
            monitor(start)
        step = start
        for epoch in range(start // len(train) + 1, args.stage + 1):
            # Exhaust the loader before saving RNG: stage boundaries match full epochs.
            for batch in train:
                step += 1
                for nodes in batch[2]: order_digest.update(nodes.numpy().tobytes())
                losses = train_step(model, criterion, optimizer, scheduler, batch, config, 'cuda',
                                    epoch=epoch, iteration=step, total_iterations=750000)
                scalars = {k: float(v.detach()) for k,v in losses.items() if torch.is_tensor(v) and v.numel()==1}
                for k,v in scalars.items(): sums[k] = sums.get(k, 0.) + v
                count += 1
                if step == start + 1:
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
                    gradients = [p.grad for p in model.spline_head.parameters() if p.grad is not None]
                    assert gradients
                    assert any(g.abs().sum()>0 for g in gradients) == bool(arm['weight'])
                    print('Resume forward/backward/AdamW passed; LR:', optimizer.param_groups[0]['lr'], flush=True)
                if tracker and step % 100 == 0:
                    tracker.log_training(scalars, iteration=step, epoch=epoch, learning_rate=optimizer.param_groups[0]['lr'])
                if args.smoke and step == target: break
            if args.smoke or epoch % plan['monitor_epochs'] == 0 or step == target:
                monitor(step, detailed=args.smoke or epoch % plan['detailed_epochs'] == 0 or step == target)
                if not args.smoke: atomic_checkpoint(output / 'latest.pt', payload(step))
                sums = {}; count = 0
            if step == target: break
        assert step == target and scheduler.last_epoch == target
        if not args.smoke: atomic_checkpoint(output / 'model.pt', payload(step))
        provenance.update(elapsed_seconds=time.monotonic()-started, end_update=step,
                          batch_order_sha256=order_digest.hexdigest())
        write_json(output / 'provenance.json', provenance)
        write_json(output / 'summary.json', dict(arm=args.arm, provenance=provenance,
                                                initial=history[0], final=history[-1]))
        if tracker: tracker.finish()
    except BaseException:
        if tracker: tracker.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
