#!/usr/bin/env python3
"""Check Branch F1 against saved real graphs and controlled real-graph breaks."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.branch_connectivity import branch_connectivity_f1
from scripts.audit_synthetic_mri_grid import SourceGraph
from scripts.study_full_volume_representations import IXI, TOPBRAIN, write_csv
from scripts.visualize_saved_adaptive_graph import graph_arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = (
        ("IXI122", IXI / "mra/train/IXI122-Guys-0773-MRA_IXI-Guys/graphs"),
        ("TopBrain-CT-001", TOPBRAIN / "ct/train/topcow_ct_001/graphs"),
    )
    rows = []
    for subject, root in pairs:
        adaptive, dense = (graph_arrays(SourceGraph.from_directory(root / name))[:2]
                           for name in ("adaptive", "dense"))
        nodes, edges = adaptive
        reference_nodes, reference_edges = dense
        cases = {
            "adaptive vs dense": (nodes, edges),
            "adaptive, one edge broken": (nodes, edges[1:]),
        }
        for name, (pred_nodes, pred_edges) in cases.items():
            result = branch_connectivity_f1(pred_nodes, pred_edges, reference_nodes,
                                            reference_edges, threshold_mm=.5)
            rows.append({"subject": subject, "case": name, **result})
        if rows[-2]["f1"] != 1 or rows[-1]["f1"] >= 1:
            raise ValueError(f"real-data metric smoke check failed: {subject}")
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "real_graph_checks.csv", rows)
    (args.output / "real_graph_checks.json").write_text(json.dumps(rows, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for i, name in enumerate(("adaptive vs dense", "adaptive, one edge broken")):
        values = [r["f1"] for r in rows if r["case"] == name]
        x = np.arange(len(pairs)) + (i - .5) * .37
        ax.bar(x, values, width=.36, color=("#398974", "#a56355")[i], label=name)
    ax.set(xticks=np.arange(len(pairs)), xticklabels=[subject for subject, _ in pairs],
           ylabel="Branch connectivity F1 (zoomed)", ylim=(.985, 1.004),
           title="Real saved graphs; a controlled break is NOT a model prediction")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(args.output / "real_graph_checks.png", dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
