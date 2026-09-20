#!/usr/bin/env python3
"""Benchmark 120 versus 192 object queries on identical real IXI patches."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs import load_config, validate_config
from data.loaders import build_data_loaders
from data.loaders.common import image_graph_collate
from data.loaders.discovery import discover_synthetic_mri
from data.loaders.synthetic_mri import SyntheticMRIDataset
from models import build_model
from training import build_criterion, build_optimizer, build_scheduler
from training.engine import train_step


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def benchmark_config(base: dict, dataset: Path, queries: int, batch_size: int, workers: int) -> dict:
    config = copy.deepcopy(base)
    config["runtime"]["device"] = "cuda"
    config["runtime"]["distributed"] = False
    config["runtime"]["workers"] = int(workers)
    config["tracking"]["enabled"] = False
    config["data"]["batch_size"] = int(batch_size)
    config["data"]["validation_batch_size"] = int(batch_size)
    config["data"]["train_augmentation"] = False
    config["data"]["datasets"] = {
        "synthetic_mri": {
            "role": "target",
            "root": str(dataset.resolve()),
            "train_samples": None,
            "validation_samples": None,
            "coordinate_space_on_disk": "normalized",
            # The patch generator already applies IXI intensity normalization.
            # No additional centering is needed for a compute benchmark.
            "foreground_mean": 0.0,
            "sample_cap_selection": "first",
            "sample_cap_seed": int(config["experiment"]["seed"]),
        }
    }
    config["model"]["decoder"]["object_queries"] = int(queries)
    config["training"]["epochs"] = 1
    config["training"]["warmup_epochs"] = 0
    config["training"]["checkpoint"]["policy"] = "none"
    config["evaluation"]["training_metrics"]["enabled"] = False
    for name in ("betti_h0", "betti_h1"):
        config["topology"][name]["enabled"] = False
        config["topology"][name]["log_only"] = True
        config["topology"][name]["weight"] = 0.0
    validate_config(config)
    return config


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def run_case(
    base: dict,
    dataset: Path,
    *,
    queries: int,
    batch_size: int,
    workers: int,
    warmup_steps: int,
    measured_steps: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    seed_everything(seed)
    config = benchmark_config(base, dataset, queries, batch_size, workers)
    train_loader, _ = build_data_loaders(config)
    device = torch.device("cuda")
    model = build_model(config).to(device)
    criterion = build_criterion(config, model).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, warmup_steps + measured_steps)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    query_parameters = model.query_embed.weight.numel()
    iterator = iter(train_loader)

    for step in range(warmup_steps):
        batch, iterator = next_batch(iterator, train_loader)
        train_step(
            model,
            criterion,
            optimizer,
            scheduler,
            batch,
            config,
            device,
            epoch=0,
            iteration=step + 1,
            total_iterations=warmup_steps + measured_steps,
        )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    rows = []
    for step in range(measured_steps):
        wait_started = time.perf_counter()
        batch, iterator = next_batch(iterator, train_loader)
        data_wait = time.perf_counter() - wait_started
        node_counts = [int(len(nodes)) for nodes in batch[2]]
        edge_counts = [int(len(edges)) for edges in batch[3]]
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        losses = train_step(
            model,
            criterion,
            optimizer,
            scheduler,
            batch,
            config,
            device,
            epoch=0,
            iteration=warmup_steps + step + 1,
            total_iterations=warmup_steps + measured_steps,
        )
        torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - started
        rows.append({
            "queries": queries,
            "step": step + 1,
            "batch_size": len(node_counts),
            "data_wait_seconds": data_wait,
            "train_step_seconds": step_seconds,
            "samples_per_second": len(node_counts) / step_seconds,
            "maximum_target_nodes": max(node_counts),
            "mean_target_nodes": statistics.fmean(node_counts),
            "maximum_target_edges": max(edge_counts),
            "loss_total": float(losses["total"].detach().cpu()),
        })

    summary = {
        "queries": queries,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "query_embedding_parameters": query_parameters,
        "train_step_seconds": distribution([row["train_step_seconds"] for row in rows]),
        "samples_per_second": distribution([row["samples_per_second"] for row in rows]),
        "data_wait_seconds": distribution([row["data_wait_seconds"] for row in rows]),
        "maximum_target_nodes": max(row["maximum_target_nodes"] for row in rows),
        "maximum_target_edges": max(row["maximum_target_edges"] for row in rows),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }
    del iterator, train_loader, scheduler, optimizer, criterion, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return summary, rows


def overflow_smoke(
    base: dict,
    root: Path,
    manifest_path: Path,
    *,
    seed: int,
) -> dict:
    records = discover_synthetic_mri(root, "train", allow_direct=False)
    with manifest_path.open(newline="") as stream:
        node_manifest = {
            row["sample_id"]: int(row["node_count"]) for row in csv.DictReader(stream)
        }
    compatible = [
        record
        for record in records
        if 120 < node_manifest.get(record.sample_id, -1) <= 192
    ]
    if not compatible:
        raise ValueError("overflow set has no IXI patch with 121..192 graph nodes")
    compatible.sort(
        key=lambda record: (-node_manifest[record.sample_id], record.sample_id)
    )
    selected = compatible[0]
    seed_everything(seed)
    config = benchmark_config(base, root, 192, 1, 0)
    dataset = SyntheticMRIDataset(
        [selected],
        image_size=tuple(config["data"]["image_size"]),
        foreground_mean=0.0,
        coordinate_space="normalized",
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=image_graph_collate)
    batch = next(iter(loader))
    target_nodes = int(len(batch[2][0]))
    target_edges = int(len(batch[3][0]))
    if target_nodes <= 120:
        raise ValueError(f"overflow smoke selected only {target_nodes} nodes")
    device = torch.device("cuda")
    model = build_model(config).to(device)
    criterion = build_criterion(config, model).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, 1)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    losses = train_step(
        model,
        criterion,
        optimizer,
        scheduler,
        batch,
        config,
        device,
        epoch=0,
        iteration=1,
        total_iterations=1,
    )
    torch.cuda.synchronize(device)
    result = {
        "sample_id": selected.sample_id,
        "target_nodes": target_nodes,
        "target_edges": target_edges,
        "queries": 192,
        "train_step_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "loss_total": float(losses["total"].detach().cpu()),
        "success": True,
    }
    del scheduler, optimizer, criterion, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def plot(summaries: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(item["queries"]) for item in summaries]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    panels = (
        ("train_step_seconds", "mean", "Mean training-step time (s)"),
        ("samples_per_second", "mean", "Throughput (samples/s)"),
        ("peak_allocated_gib", None, "Peak allocated GPU memory (GiB)"),
    )
    for axis, (name, nested, ylabel) in zip(axes, panels):
        values = [item[name][nested] if nested else item[name] for item in summaries]
        bars = axis.bar(labels, values, color=("#4c78a8", "#59a14f"))
        axis.set(xlabel="Object queries", ylabel=ylabel)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3g}",
                ha="center",
                va="bottom",
            )
    figure.suptitle("IXI-only 120 vs 192 object-query training cost")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--overflow-dataset", type=Path, required=True)
    parser.add_argument("--overflow-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=364505)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.warmup_steps < 1 or args.measured_steps < 1:
        parser.error("warmup and measured steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    # These base-config placeholders are replaced below, but must be resolvable
    # while loading the inherited YAML configuration.
    environment.setdefault("GNBM_OUTPUT_DIR", str(args.output.resolve()))
    environment.setdefault("SYNTHETIC_MRI_DATASET", str(args.dataset.resolve()))
    base = load_config(args.config, environment=environment)
    summaries, steps = [], []
    for queries in (120, 192):
        summary, rows = run_case(
            base,
            args.dataset,
            queries=queries,
            batch_size=args.batch_size,
            workers=args.workers,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
            seed=args.seed,
        )
        summaries.append(summary)
        steps.extend(rows)
        print(json.dumps(summary, sort_keys=True), flush=True)
    baseline, candidate = summaries
    comparison = {
        "time_increase_fraction": (
            candidate["train_step_seconds"]["mean"]
            / baseline["train_step_seconds"]["mean"]
            - 1.0
        ),
        "throughput_change_fraction": (
            candidate["samples_per_second"]["mean"]
            / baseline["samples_per_second"]["mean"]
            - 1.0
        ),
        "peak_allocated_increase_gib": (
            candidate["peak_allocated_gib"] - baseline["peak_allocated_gib"]
        ),
        "peak_allocated_increase_fraction": (
            candidate["peak_allocated_gib"] / baseline["peak_allocated_gib"] - 1.0
        ),
        "parameter_increase": candidate["parameters"] - baseline["parameters"],
        "parameter_increase_fraction": (
            candidate["parameters"] / baseline["parameters"] - 1.0
        ),
    }
    payload = {
        "schema_version": 1,
        "dataset": "IXI only",
        "config": str(args.config.resolve()),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "cases": summaries,
        "comparison": comparison,
        "overflow_smoke": overflow_smoke(
            base,
            args.overflow_dataset,
            args.overflow_manifest,
            seed=args.seed,
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with (args.output / "steps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(steps[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(steps)
    plot(summaries, args.output / "comparison.png")
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)
    print(json.dumps(payload["overflow_smoke"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
