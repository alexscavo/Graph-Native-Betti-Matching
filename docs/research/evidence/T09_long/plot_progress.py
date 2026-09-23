"""Plot the live IXI 100-epoch run from its own logs and JSONL records."""

from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "runs" / "ixi_q256_100ep_20260923"
LOG = Path("/lustre/fswork/projects/rech/vnc/upz25mj/logs/graph-native-betti-matching/gnbm-train-93539.out")
OUT = Path(__file__).resolve().parent


def rows(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    performance = rows(RUN / "performance.jsonl")
    validation = rows(RUN / "validation-metrics.jsonl")
    log = LOG.read_text() if LOG.is_file() else ""
    training_loss = [(int(epoch), float(value)) for epoch, value in
                     re.findall(r"^epoch=(\d+)/\d+ total=([0-9.eE+-]+)", log, re.M)]
    validation_loss = [(int(epoch), float(value)) for epoch, value in
                       re.findall(r"^validation epoch=(\d+) total=([0-9.eE+-]+)", log, re.M)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for points, label in ((training_loss, "train loss"), (validation_loss, "validation loss")):
        if points:
            axes[0, 0].plot(*zip(*points), marker="o", ms=2, label=label)
    axes[0, 0].set_title("Optimization")
    if training_loss or validation_loss:
        axes[0, 0].legend()
    for key, label in (("branch_f1", "Branch F1"), ("node_f1", "Node F1"),
                       ("edge_f1", "Edge F1")):
        points = [(row["epoch"], row[key]) for row in validation
                  if row.get(key) is not None]
        if points:
            axes[0, 1].plot(*zip(*points), marker="o", label=label)
    axes[0, 1].set_title("Held-out IXI graph metrics")
    axes[0, 1].set_ylim(0, 1)
    if validation:
        axes[0, 1].legend()
    if performance:
        axes[1, 0].plot([row["epoch"] for row in performance],
                        [row["samples_per_second"] for row in performance], marker=".")
        axes[1, 1].plot([row["epoch"] for row in performance],
                        [row["peak_allocated_gib"] for row in performance], marker=".")
    axes[1, 0].set_title("Training throughput (samples/s)")
    axes[1, 1].set_title("Peak allocated GPU memory (GiB)")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=.25)
    fig.suptitle("IXI-only adaptive-GT training, 256 queries, two H100s")
    fig.savefig(OUT / "progress.png", dpi=160)
    plt.close(fig)
    (OUT / "progress_summary.json").write_text(json.dumps({
        "training_epochs_logged": len(training_loss),
        "validation_loss_epochs_logged": len(validation_loss),
        "graph_metric_epochs_logged": len(validation),
        "performance_epochs_logged": len(performance),
        "latest_epoch": max((epoch for epoch, _ in training_loss), default=0),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
