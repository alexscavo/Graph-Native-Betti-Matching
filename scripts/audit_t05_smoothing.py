#!/usr/bin/env python3
"""Decompose raw-dense to smoothed to straight-graph vessel-length change."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.representation_geometry import polyline_length
from scripts.audit_synthetic_mri_grid import SourceGraph
from scripts.study_full_volume_representations import IXI, TOPBRAIN, write_csv


def length_decomposition(raw_mm: float, smoothed_mm: float, adaptive_mm: float) -> dict:
    if raw_mm <= 0 or smoothed_mm <= 0:
        raise ValueError("both reference lengths must be positive")
    return {
        "raw_dense_mm": raw_mm, "smoothed_mm": smoothed_mm, "adaptive_straight_mm": adaptive_mm,
        "raw_to_smoothed_fraction": (smoothed_mm - raw_mm) / raw_mm,
        "smoothed_to_adaptive_fraction": (adaptive_mm - smoothed_mm) / raw_mm,
        "raw_to_adaptive_fraction": (adaptive_mm - raw_mm) / raw_mm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    with (args.report_dir / "volumes.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    rows = []
    for row in records:
        if row["representation"] != "adaptive":
            continue
        root = IXI if row["dataset"] == "ixi" else TOPBRAIN
        graph_dir = root / row["modality"] / row["split"] / row["subject"] / "graphs"
        smoothed = SourceGraph.from_directory(graph_dir / "junction_only")
        smoothed_mm = sum(polyline_length(line) for line in smoothed.edge_polylines)
        measurement = length_decomposition(float(row["dense_length_mm"]), smoothed_mm,
                                           float(row["physical_length_mm"]))
        if not np.isclose(sum(measurement[key] for key in (
                "raw_to_smoothed_fraction", "smoothed_to_adaptive_fraction")),
                measurement["raw_to_adaptive_fraction"], atol=1e-10):
            raise ValueError("non-additive length decomposition")
        rows.append({"subject": row["subject"], "stratum": row["stratum"],
                     "split": row["split"], **measurement})
    write_csv(args.report_dir / "length_decomposition.csv", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["stratum"]].append(row)
    summary = {group: {key: float(np.mean([row[key] for row in members]))
              for key in ("raw_to_smoothed_fraction", "smoothed_to_adaptive_fraction",
                          "raw_to_adaptive_fraction")}
              for group, members in sorted(groups.items())}
    (args.report_dir / "length_decomposition.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(summary)
    x = np.arange(len(names))
    smooth = np.array([summary[name]["raw_to_smoothed_fraction"] for name in names]) * -100
    compact = np.array([summary[name]["smoothed_to_adaptive_fraction"] for name in names]) * -100
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x, smooth, label="Raw voxel path → smoothed path", color="#a88d70")
    ax.bar(x, compact, bottom=smooth, label="Smoothed path → straight adaptive edges", color="#398974")
    ax.set(xticks=x, xticklabels=names, ylabel="Mean loss of raw dense length (%)",
           title="T05: distinguish smoothing from graph-edge simplification")
    ax.set_ylim(0, max(0, *(smooth + compact)) * 1.35)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(args.report_dir / "length_decomposition.png", dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
