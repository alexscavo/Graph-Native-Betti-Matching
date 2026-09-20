#!/usr/bin/env python3
"""Select deterministic paired and overflow IXI patch benchmark datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def evenly_spaced(rows: list[dict], limit: int) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (int(row["node_count"]), row["sample_id"]))
    if len(ordered) <= limit:
        return ordered
    indices = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    if len(set(indices)) != limit:
        raise AssertionError("rank selection produced duplicate indices")
    return [ordered[index] for index in indices]


def patch_paths(root: Path, row: dict) -> tuple[Path, Path, Path]:
    split, sample_id = row["split"], row["sample_id"]
    return (
        root / split / "raw" / f"{sample_id}_data.nii.gz",
        root / split / "seg" / f"{sample_id}_seg.nii.gz",
        root / split / "vtp" / f"{sample_id}_graph.vtp",
    )


def materialize(root: Path, output: Path, rows: Iterable[dict], *, force_split: str | None = None) -> list[dict]:
    written = []
    for row in rows:
        split = force_split or row["split"]
        sample_id = row["sample_id"]
        destinations = (
            output / split / "raw" / f"{sample_id}_data.nii.gz",
            output / split / "seg" / f"{sample_id}_seg.nii.gz",
            output / split / "vtp" / f"{sample_id}_graph.vtp",
        )
        for directory in (destination.parent for destination in destinations):
            directory.mkdir(parents=True, exist_ok=True)
        for source, destination in zip(patch_paths(root, row), destinations):
            if destination.exists():
                if not os.path.samefile(source, destination):
                    raise FileExistsError(destination)
            else:
                os.link(source, destination)
        written.append({**row, "benchmark_split": split})
    return written


def distribution(rows: list[dict]) -> dict:
    values = np.asarray([int(row["node_count"]) for row in rows], dtype=np.int64)
    return {
        "patches": int(len(values)),
        "node_mean": float(values.mean()) if len(values) else 0.0,
        "node_median": float(np.median(values)) if len(values) else 0.0,
        "node_p95": float(np.percentile(values, 95)) if len(values) else 0.0,
        "node_max": int(values.max()) if len(values) else 0,
        "over_120": int(np.count_nonzero(values > 120)),
        "over_192": int(np.count_nonzero(values > 192)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(all_rows: list[dict], paired: list[dict], overflow: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    bins = np.arange(0, max(int(row["node_count"]) for row in all_rows) + 6, 5)
    for rows, label, colour in (
        (all_rows, "all extracted IXI patches", "#4c78a8"),
        (paired, "paired benchmark (≤120)", "#59a14f"),
    ):
        axes[0].hist(
            [int(row["node_count"]) for row in rows],
            bins=bins,
            histtype="step",
            linewidth=2,
            label=label,
            color=colour,
        )
    axes[0].axvline(120, color="#c0392b", linestyle="--", label="120 queries")
    axes[0].set(xlabel="Graph nodes per 64³ patch", ylabel="Patch count", yscale="log")
    axes[0].legend(fontsize=8)

    labels = ["all", "paired", "overflow"]
    summaries = [distribution(all_rows), distribution(paired), distribution(overflow)]
    axes[1].bar(labels, [item["node_max"] for item in summaries], color=("#4c78a8", "#59a14f", "#e15759"))
    axes[1].axhline(120, color="#c0392b", linestyle="--")
    axes[1].axhline(192, color="#9467bd", linestyle="--")
    axes[1].set(ylabel="Maximum graph nodes")
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle("IXI-only object-query benchmark patch selection")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--paired-output", type=Path, required=True)
    parser.add_argument("--overflow-output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    with (args.patch_root / "patch_index.csv").open(newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    limits = {
        "train": args.train_samples,
        "val": args.validation_samples,
        "test": args.test_samples,
    }
    paired_source = [row for row in all_rows if int(row["node_count"]) <= 120]
    paired_selected = [
        row
        for split, limit in limits.items()
        for row in evenly_spaced(
            [item for item in paired_source if item["split"] == split], limit
        )
    ]
    overflow_source = [row for row in all_rows if int(row["node_count"]) > 120]
    if not overflow_source:
        raise ValueError("no >120-node IXI patches were extracted")
    paired_written = materialize(args.patch_root, args.paired_output, paired_selected)
    overflow_written = materialize(
        args.patch_root,
        args.overflow_output,
        sorted(overflow_source, key=lambda row: (-int(row["node_count"]), row["sample_id"])),
        force_split="train",
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / "paired_patches.csv", paired_written)
    write_csv(args.report_dir / "overflow_patches.csv", overflow_written)
    summary = {
        "schema_version": 1,
        "dataset": "IXI only",
        "source_patch_root": str(args.patch_root.resolve()),
        "paired_output": str(args.paired_output.resolve()),
        "overflow_output": str(args.overflow_output.resolve()),
        "all": distribution(all_rows),
        "paired": distribution(paired_written),
        "overflow": distribution(overflow_written),
        "paired_by_split": {
            split: distribution([row for row in paired_written if row["benchmark_split"] == split])
            for split in ("train", "val", "test")
        },
        "maximum_overflow_sample": overflow_written[0]["sample_id"],
        "maximum_192_compatible_sample": max(
            (row for row in overflow_written if int(row["node_count"]) <= 192),
            key=lambda row: (int(row["node_count"]), row["sample_id"]),
        )["sample_id"],
    }
    (args.report_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    plot(all_rows, paired_written, overflow_written, args.report_dir / "patch_selection.png")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
