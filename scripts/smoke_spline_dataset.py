#!/usr/bin/env python3
"""Three real-batch updates, validation loss, projections and W&B image upload."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from configs import load_config
from data.loaders import build_data_loaders, build_evaluation_loader
from models import build_model
from models.checkpoint import load_legacy_model_checkpoint
from training.engine import train_step, evaluate_loss
from training.losses.criterion import build_criterion
from training.optim import build_optimizer, build_scheduler
from training.tracking import build_tracker
from training.evaluation.spline_visualization import save_validation_projections
from training.evaluation.evaluator import evaluate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/finetune_spline_v0_10k.yaml')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial-weights', type=Path, required=True)
    args = parser.parse_args()
    config = copy.deepcopy(load_config(args.config))
    config['data']['datasets']['synthetic_mri']['root'] = str(args.data_root)
    config['experiment']['name'] += '_smoke'
    torch.manual_seed(config['experiment']['seed'])
    train, _ = build_data_loaders(config)
    validation = build_evaluation_loader(config, dataset_name='synthetic_mri', split='val', max_samples=4)
    model = build_model(config).cuda()
    load_legacy_model_checkpoint(model, args.initial_weights, allow_missing_spline_head=True)
    criterion = build_criterion(config, model).cuda()
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, len(train))
    args.output.mkdir(parents=True, exist_ok=True)
    tracker = build_tracker(config, args.output, resume=False, launch_metadata={'purpose': 'Three-batch smoke, not results'})
    try:
        for step, batch in enumerate(train, start=1):
            losses = train_step(model, criterion, optimizer, scheduler, batch, config, 'cuda',
                                epoch=1, iteration=step, total_iterations=50*len(train))
            gradient = sum(float(p.grad.abs().sum()) for p in model.spline_head.parameters() if p.grad is not None)
            assert torch.isfinite(losses['total']) and gradient > 0
            print(f'smoke step={step} total={float(losses["total"])} spline_gradient={gradient}', flush=True)
            if step == 3:
                break
        validation_loss = evaluate_loss(model, criterion, validation, config, 'cuda')
        assert all(torch.isfinite(torch.tensor(value)) for value in validation_loss.values())
        paths = save_validation_projections(model, config, 'cuda', args.output, 0, 4)
        if tracker:
            tracker.log_validation(validation_loss, iteration=3, epoch=0)
            tracker.log_validation_images(paths, epoch=0)
        summary, _ = evaluate_model(model, validation, config, 'cuda', output_dir=args.output/'validation-evaluation')
        assert summary['samples'] == 4
        (args.output/'smoke.json').write_text(json.dumps({'passed': True, 'validation_loss': validation_loss,
                                                       'projection_paths': [str(p) for p in paths]}, indent=2)+'\n')
    except BaseException:
        if tracker:
            tracker.finish(exit_code=1)
        raise
    if tracker:
        tracker.finish()


if __name__ == '__main__':
    main()
