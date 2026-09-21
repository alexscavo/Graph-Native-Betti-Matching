#!/usr/bin/env python3
"""Per-volume T05 rate–distortion scatter and robust modality/site summaries."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.study_full_volume_representations import write_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    with (args.report_dir / "volumes.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_subject = defaultdict(dict)
    for row in rows:
        by_subject[row["subject"]][row["representation"]] = row
    if any(set(items) != {"dense", "adaptive", "junction_only"} for items in by_subject.values()):
        raise ValueError("incomplete representation family")
    fields = ("nodes", "degree_2", "curve_f1_0.5mm", "curve_acd_mm", "curve_hd95_mm",
              "length_weighted_absolute_branch_length_error_fraction", "junction_angle_mae_deg")
    groups = defaultdict(list)
    for triple in by_subject.values():
        for row in triple.values():
            groups[(row["stratum"], row["representation"])].append(row)
    summary = []
    for (stratum, representation), members in sorted(groups.items()):
        result = {"stratum": stratum, "representation": representation, "subjects": len(members)}
        for field in fields:
            values = np.asarray([float(row[field]) for row in members])
            values = values[np.isfinite(values)]
            result[field + "_median"] = float(np.median(values)) if len(values) else float("nan")
            result[field + "_p95"] = float(np.percentile(values, 95)) if len(values) else float("nan")
        summary.append(result)
    write_csv(args.report_dir / "strata_robust.csv", summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"junction_only": ("#a56355", "o"), "adaptive": ("#398974", "^")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for stratum, row in groups.items():
        site, representation = stratum
        if representation == "dense":
            continue
        color, marker = styles[representation]
        axes[0].scatter([int(record["nodes"]) for record in row],
                        [float(record["curve_f1_0.5mm"]) for record in row],
                        color=color, marker=marker, s=17, alpha=.45,
                        label=representation if site == "IXI-Guys" else None)
        axes[1].scatter([float(record["nodes_per_dense_mm"]) for record in row],
                        [float(record["curve_acd_mm"]) for record in row],
                        color=color, marker=marker, s=17, alpha=.45)
    axes[0].set(xlabel="Full-volume nodes", ylabel="Branch-paired Curve F1 at 0.5 mm")
    axes[1].set(xlabel="Nodes / raw-dense vessel length (mm⁻¹)", ylabel="Branch-paired ACD (mm)")
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(f"T05: per-volume rate–distortion, {len(by_subject)} real volumes")
    fig.savefig(args.report_dir / "rate_distortion.png", dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
