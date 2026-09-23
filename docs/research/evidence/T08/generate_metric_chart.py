"""Regenerate the Phase-4 diagnostic score table and chart (synthetic controls)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tests.vessel.test_metric_failure_matrix import evaluated_case


def main() -> None:
    parallel = np.array([[.2, .4, .5], [.8, .4, .5],
                         [.2, .42, .5], [.8, .42, .5]])
    parallel_edges = [(0, 1), (2, 3)]
    straight = np.array([[.2, .5, .5], [.8, .5, .5]])
    bent = np.array([[.2, .5, .5], [.8, .5, .5], [.5, .58, .5]])
    scores = {
        "separate_vessels": evaluated_case(parallel, parallel_edges, parallel, parallel_edges),
        "false_fusion": evaluated_case(parallel, parallel_edges + [(0, 2)], parallel, parallel_edges),
        "straight_branch": evaluated_case(straight, [(0, 1)], straight, [(0, 1)]),
        "bent_branch": evaluated_case(bent, [(0, 2), (2, 1)], straight, [(0, 1)]),
    }
    fields = ("branch_f1", "node_mAP", "edge_mAP", "smd",
              "beta0_absolute_error", "beta1_absolute_error", "node_fp", "edge_fp")
    compact = {case: {field: float(values[field]) for field in fields}
               for case, values in scores.items()}
    output = Path(__file__).resolve().parent
    (output / "metric_scores.json").write_text(json.dumps(compact, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    pairs = (("separate_vessels", "false_fusion"),
             ("straight_branch", "bent_branch"))
    labels = ("Branch F1", "Node mAP", "Edge mAP")
    for axis, pair, title in zip(axes, pairs, ("False vessel fusion", "Bent geometry")):
        for offset, case in zip((-.18, .18), pair):
            axis.bar(np.arange(3) + offset,
                     [compact[case][field] for field in ("branch_f1", "node_mAP", "edge_mAP")],
                     width=.34, label=case.replace("_", " "))
        axis.set_xticks(range(3), labels)
        axis.set_ylim(0, 1.08)
        axis.set_title(title)
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=.25)
    fig.suptitle("Metric behavior on controlled graph errors (not model predictions)")
    fig.savefig(output / "metric_failure_matrix.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
