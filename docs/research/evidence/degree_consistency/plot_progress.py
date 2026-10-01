#!/usr/bin/env python3
"""Plot IXI degree-consistency training against the matched CE baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASELINE = "ixi_q256_types_300ep_20260924"
CANDIDATE = "ixi_q256_degree_consistency_300ep_20261001"


def records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "progress.png")
    args = parser.parse_args()
    candidate = records(args.runs_root / CANDIDATE / "validation-metrics.jsonl")
    baseline = records(args.runs_root / BASELINE / "validation-metrics.jsonl")
    degree = records(args.runs_root / CANDIDATE / "training-losses.jsonl")
    if not candidate or not baseline:
        raise ValueError("Both runs need validation-metrics.jsonl")
    last_epoch = max(row["epoch"] for row in candidate)
    baseline = [row for row in baseline if row["epoch"] <= last_epoch]

    fig, axes = plt.subplots(2, 4, figsize=(18, 7), constrained_layout=True)
    panels = (
        ("node_f1", "Node F1", None),
        ("edge_f1", "Edge F1", None),
        ("branch_f1", "Branch F1", None),
        ("predicted_nodes", "Predicted nodes / patch", "target_nodes"),
        ("predicted_edges", "Predicted edges / patch", "target_edges"),
        ("predicted_beta0", "Components β₀ / patch", "target_beta0"),
        ("predicted_beta1", "Predicted cycles β₁ / patch", "target_beta1"),
    )
    for ax, (key, title, target) in zip(axes.flat, panels):
        ax.plot([r["epoch"] for r in baseline], [r[key] for r in baseline],
                "o-", color="#4c78a8", label="Cross-entropy")
        ax.plot([r["epoch"] for r in candidate], [r[key] for r in candidate],
                "o-", color="#f58518", label="Degree + unmatched + class 5")
        if target:
            ax.axhline(candidate[-1][target], color="#54a24b", linestyle="--",
                       label="GT mean")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    ax = axes.flat[-1]
    ax.plot([r["epoch"] for r in degree], [r["degree"] for r in degree],
            color="#b279a2", label="Raw degree loss")
    ax.plot([r["epoch"] for r in degree], [r["degree_weighted"] for r in degree],
            color="#e45756", label="Weighted contribution")
    ax.set_title("Degree loss during training")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    axes.flat[3].legend(fontsize=8)
    fig.suptitle(
        f"IXI validation, 2,961 patches · matched comparison through epoch {last_epoch}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
