#!/usr/bin/env python3
"""Plot paired IXI test results for the three 300-epoch node-type models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


RUNS = (
    ("Cross-entropy", "ixi_q256_types_300ep_20260924"),
    ("Object focal", "ixi_q256_object_focal_300ep_20260925"),
    ("Object + type focal", "ixi_q256_object_and_type_focal_300ep_20260925"),
)
METRICS = (
    "node_f1", "node_mAP", "node_mAR", "edge_f1", "edge_mAP", "edge_mAR",
    "edge_precision", "edge_recall", "branch_f1", "predicted_nodes",
    "predicted_edges", "predicted_beta1", "target_nodes", "target_edges",
    "target_beta1", "node_count_absolute_error", "edge_count_absolute_error",
    "beta1_absolute_error", "samples",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    rows = []
    for label, run in RUNS:
        for decoder, suffix in (
            ("Plain", "evaluation/test_best_branch_f1/summary.json"),
            ("Degree 0.5", "evaluation/test_degree_penalty/penalty_0.5/summary.json"),
        ):
            path = args.runs_root / run / suffix
            values = json.loads(path.read_text(encoding="utf-8"))
            if values["samples"] != 2384:
                raise ValueError(f"Expected all 2384 IXI test patches: {path}")
            rows.append({"model": label, "decoder": decoder, "run": run,
                         **{key: values[key] for key in METRICS}})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("model", "decoder", "run", *METRICS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    by_pair = {(row["model"], row["decoder"]): row for row in rows}
    names = [label for label, _ in RUNS]
    x = np.arange(len(names))
    width = 0.34
    colors = {"Plain": "#4c78a8", "Degree 0.5": "#f58518"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = (
        (axes[0, 0], "node_f1", "Node F1", None),
        (axes[0, 1], "edge_f1", "Edge F1", None),
        (axes[1, 0], "predicted_edges", "Predicted edges / patch", "target_edges"),
        (axes[1, 1], "predicted_beta1", "Predicted cycles β₁ / patch", "target_beta1"),
    )
    for ax, key, title, target_key in panels:
        for decoder, offset in (("Plain", -width / 2), ("Degree 0.5", width / 2)):
            values = [by_pair[(name, decoder)][key] for name in names]
            ax.bar(x + offset, values, width, label=decoder, color=colors[decoder])
        if target_key:
            target = rows[0][target_key]
            ax.axhline(target, color="#54a24b", linestyle="--", linewidth=1.7,
                       label=f"GT mean ({target:.2f})")
        ax.set_title(title)
        ax.set_xticks(x, names)
        ax.tick_params(axis="x", labelrotation=12)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.legend(
        handles=[
            Patch(color=colors["Plain"], label="Plain"),
            Patch(color=colors["Degree 0.5"], label="Degree penalty 0.5"),
            Line2D([0], [0], color="#54a24b", linestyle="--", label="GT mean"),
        ],
        loc="outside lower center",
        ncol=3,
    )
    fig.suptitle("IXI held-out test · 2,384 patches · epoch-290 checkpoints", fontsize=14)
    fig.savefig(args.output_dir / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
