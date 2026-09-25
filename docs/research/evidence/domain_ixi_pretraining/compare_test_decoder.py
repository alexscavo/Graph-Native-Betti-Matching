"""Create a plain-vs-soft-degree-penalty test comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_q256_types_300ep_20260924/evaluation")
OUT = Path(__file__).resolve().parent
PLAIN = ROOT / "test_best_branch_f1/summary.json"
PENALTY = ROOT / "test_degree_penalty/penalty_0.5/summary.json"


METRICS = (
    ("node_f1", "Node F1", False),
    ("edge_f1", "Edge F1", False),
    ("edge_precision", "Edge precision", False),
    ("edge_recall", "Edge recall", False),
    ("branch_f1", "Branch F1", False),
    ("edge_count_absolute_error", "Edge-count error", True),
    ("beta1_absolute_error", "β₁ error", True),
    ("predicted_edges", "Predicted edges", True),
    ("predicted_beta1", "Predicted β₁", True),
)


def main():
    plain = json.loads(PLAIN.read_text(encoding="utf-8"))
    penalty = json.loads(PENALTY.read_text(encoding="utf-8"))
    rows = []
    for key, label, lower_is_better in METRICS:
        rows.append({
            "metric": key,
            "label": label,
            "plain": plain[key],
            "penalty_0p5": penalty[key],
            "delta": penalty[key] - plain[key],
            "improvement_direction": "decrease" if lower_is_better else "increase",
        })
    payload = {
        "samples": plain["samples"],
        "checkpoint": "ixi_q256_types_300ep_20260924/models/best_metric_checkpoint.pt",
        "plain_decoder": "test_best_branch_f1",
        "soft_degree_penalty": 0.5,
        "penalty_decoder": "test_degree_penalty/penalty_0.5",
        "metrics": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_decoder_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    labels = [row["label"] for row in rows]
    plain_values = [row["plain"] for row in rows]
    penalty_values = [row["penalty_0p5"] for row in rows]
    # Normalize each metric by the plain value so metrics with different units
    # can be compared without hiding the absolute values in the JSON.
    ratios = [
        [1.0, row["penalty_0p5"] / row["plain"] if row["plain"] else 0.0]
        for row in rows
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].barh(labels, plain_values, color="#90a4ae", label="Plain")
    axes[0].barh(labels, penalty_values, left=plain_values, color="#1565c0", alpha=0.85,
                 label="Penalty 0.5 (stacked; not additive)")
    axes[0].set_title("Absolute test metrics")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].barh(labels, [value[0] for value in ratios], color="#90a4ae", label="Plain = 1")
    axes[1].barh(labels, [value[1] for value in ratios], color="#ef6c00", alpha=0.9,
                 label="Penalty / plain")
    axes[1].axvline(1.0, color="black", linewidth=0.8)
    axes[1].set_title("Relative change (higher is not always better)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle("IXI test: plain decoder vs soft degree penalty 0.5")
    fig.savefig(OUT / "test_decoder_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
