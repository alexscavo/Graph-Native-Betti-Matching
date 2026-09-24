"""Summarize the frozen IXI validation soft-degree versus threshold sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="IXI validation: frozen type model, 2,961 patches")
    args = parser.parse_args()

    records = []
    for path in sorted(args.sweep_dir.glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["samples"] != 2961:
            raise ValueError(f"Incomplete validation: {path}")
        label = path.parent.name
        records.append({
            "method": "soft type degree" if label.startswith("penalty_") else "plain threshold",
            "setting": label.split("_", maxsplit=2)[-1],
            "edge_count": summary["predicted_edges"],
            "edge_precision": summary["edge_precision"],
            "edge_recall": summary["edge_recall"],
            "edge_f1": summary["edge_f1"],
            "branch_f1": summary["branch_f1"],
            "beta1_error": summary["beta1_absolute_error"],
            "edge_count_error": summary["edge_count_absolute_error"],
            "target_edges": summary["target_edges"],
        })
    if not records:
        raise ValueError(f"No completed sweep outputs in {args.sweep_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "soft_degree_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    fig, axes = plt.subplots(3, 2, figsize=(11, 11), layout="constrained")
    colors = {"soft type degree": "#1f77b4", "plain threshold": "#ff7f0e"}
    for method in colors:
        rows = [row for row in records if row["method"] == method]
        for axis, key, ylabel in (
            (axes[0, 0], "edge_precision", "Edge precision"),
            (axes[0, 1], "edge_recall", "Edge recall"),
            (axes[1, 0], "edge_f1", "Edge F1"),
            (axes[1, 1], "branch_f1", "Physical Branch F1"),
            (axes[2, 0], "beta1_error", "Mean cycle-count error |β₁|"),
            (axes[2, 1], "edge_count_error", "Mean absolute edge-count error"),
        ):
            axis.plot([r["edge_count"] for r in rows], [r[key] for r in rows],
                      "o-", color=colors[method], label=method)
            for row in rows:
                axis.annotate(row["setting"], (row["edge_count"], row[key]),
                              xytext=(4, 3), textcoords="offset points", fontsize=8)
            axis.set(xlabel="Mean predicted edges / patch", ylabel=ylabel)
            axis.axvline(records[0]["target_edges"], color="black", linestyle="--",
                         linewidth=1, label="GT edge count")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(args.title)
    fig.savefig(args.output_dir / "soft_degree_sweep.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
