#!/usr/bin/env python3
"""Compare two selected graph policies from saved real-data audit summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text())


def policy_record(path: Path, policy: str) -> dict:
    payload = load_json(path)
    try:
        return payload["policies"][policy]
    except KeyError as error:
        raise ValueError(f"policy {policy!r} is absent from {path}") from error


def single_record(path: Path, *, policy: str | None = None) -> dict:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected a list of summary records in {path}")
    if policy is not None:
        matches = [row for row in payload if row.get("policy") == policy]
        if len(matches) == 1:
            return matches[0]
    if len(payload) == 1:
        return payload[0]
    raise ValueError(f"could not select policy {policy!r} from {path}")


def collect(label: str, sweep: Path, policy: str, geometry: Path, overflow: Path) -> dict:
    counts = policy_record(sweep, policy)
    shape = single_record(geometry)
    capacity = single_record(overflow, policy=policy)
    return {
        "label": label,
        "policy": policy,
        "subjects": counts["topology_subjects"],
        "topology_preserved_subjects": counts["topology_preserved_subjects"],
        "full_degree_2_nodes": counts["full_degree_2_total"],
        "patch_node_p95": counts["patch_nodes"]["p95"],
        "patch_node_p99": counts["patch_nodes"]["p99"],
        "patch_node_max": counts["patch_nodes"]["max"],
        "patches_over_120": capacity["capacity_exceedances"]["120"],
        "patches_over_160": capacity["capacity_exceedances"]["160"],
        "patches_over_192": capacity["capacity_exceedances"]["192"],
        "p95_edge_error_mm": shape["maximum_error_mm"]["p95"],
        "maximum_edge_error_mm": shape["maximum_error_mm"]["max"],
        "maximum_error_over_radius": shape["max_error_over_median_radius"]["max"],
        "length_weighted_shortening_fraction": shape["length_weighted_relative_length_error"],
        "mean_chord_containment_fraction": shape["containment_mean"],
        "edges_below_95pct_containment": shape["edges_below_95pct_containment"],
    }


def collect_strata(label: str, sweep: Path, policy: str) -> list[dict]:
    payload = load_json(sweep)
    rows = []
    for stratum, policies in sorted(payload["strata"].items()):
        if policy not in policies:
            raise ValueError(f"policy {policy!r} is absent from stratum {stratum!r} in {sweep}")
        nodes = policies[policy]["patch_nodes"]
        rows.append({
            "label": label,
            "stratum": stratum,
            "patches": nodes["count"],
            "mean_patch_nodes": nodes["mean"],
            "p95_patch_nodes": nodes["p95"],
            "p99_patch_nodes": nodes["p99"],
            "maximum_patch_nodes": nodes["max"],
            "patches_over_70": nodes["over_70"],
            "patches_over_120": nodes["over_120"],
        })
    return rows


def plot(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [row["label"] for row in rows]
    colours = ("#4c78a8", "#59a14f")
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = (
        ("full_degree_2_nodes", "Full-volume degree-2 nodes", 1.0),
        ("patch_node_p95", "P95 nodes / nonempty 64³ patch", 1.0),
        ("patches_over_120", "Patches exceeding 120 nodes", 1.0),
        ("p95_edge_error_mm", "P95 edge maximum error (mm)", 1.0),
        ("maximum_edge_error_mm", "Worst edge maximum error (mm)", 1.0),
        ("length_weighted_shortening_fraction", "Length-weighted shortening (%)", 100.0),
    )
    x = np.arange(len(rows))
    for axis, (key, ylabel, scale) in zip(axes.flat, panels):
        values = [scale * row[key] for row in rows]
        bars = axis.bar(x, values, color=colours[: len(rows)])
        axis.set(xticks=x, xticklabels=labels, ylabel=ylabel)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle(
        "Selected global graph policy: complexity and centerline fidelity on 14 real volumes"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_strata(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(dict.fromkeys(row["label"] for row in rows))
    strata = list(dict.fromkeys(row["stratum"] for row in rows))
    colours = ("#4c78a8", "#59a14f")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    metrics = (
        ("mean_patch_nodes", "Mean nodes / nonempty 64³ patch"),
        ("p95_patch_nodes", "P95 nodes / nonempty 64³ patch"),
        ("patches_over_70", "Patches exceeding 70 nodes"),
    )
    x = np.arange(len(strata))
    width = 0.36
    by_key = {(row["label"], row["stratum"]): row for row in rows}
    for axis, (metric, ylabel) in zip(axes, metrics):
        for index, label in enumerate(labels):
            values = [by_key[(label, stratum)][metric] for stratum in strata]
            axis.bar(
                x + (index - (len(labels) - 1) / 2) * width,
                values,
                width,
                label=label,
                color=colours[index],
            )
        axis.set(xticks=x, xticklabels=strata, ylabel=ylabel)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend()
    figure.suptitle("Cross-dataset exact-crop complexity (same 14-volume comparison)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("baseline", "candidate"):
        parser.add_argument(f"--{prefix}-label", required=True)
        parser.add_argument(f"--{prefix}-sweep", type=Path, required=True)
        parser.add_argument(f"--{prefix}-policy", required=True)
        parser.add_argument(f"--{prefix}-geometry", type=Path, required=True)
        parser.add_argument(f"--{prefix}-overflow", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        collect(
            getattr(args, f"{prefix}_label"),
            getattr(args, f"{prefix}_sweep"),
            getattr(args, f"{prefix}_policy"),
            getattr(args, f"{prefix}_geometry"),
            getattr(args, f"{prefix}_overflow"),
        )
        for prefix in ("baseline", "candidate")
    ]
    baseline, candidate = rows
    strata = [
        row
        for prefix in ("baseline", "candidate")
        for row in collect_strata(
            getattr(args, f"{prefix}_label"),
            getattr(args, f"{prefix}_sweep"),
            getattr(args, f"{prefix}_policy"),
        )
    ]
    comparison = {
        "policies": rows,
        "strata": strata,
        "candidate_change_vs_baseline": {
            "full_degree_2_nodes_fraction": (
                candidate["full_degree_2_nodes"] / baseline["full_degree_2_nodes"] - 1.0
            ),
            "patches_over_120_fraction": (
                candidate["patches_over_120"] / baseline["patches_over_120"] - 1.0
            ),
            "p95_edge_error_fraction": (
                candidate["p95_edge_error_mm"] / baseline["p95_edge_error_mm"] - 1.0
            ),
            "maximum_edge_error_fraction": (
                candidate["maximum_edge_error_mm"] / baseline["maximum_edge_error_mm"] - 1.0
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "strata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(strata[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(strata)
    plot(rows, args.output_dir / "comparison.png")
    plot_strata(strata, args.output_dir / "strata_comparison.png")
    print(json.dumps(comparison["candidate_change_vs_baseline"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
