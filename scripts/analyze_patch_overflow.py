#!/usr/bin/env python3
"""Decompose real patch node overflows without changing the graph representation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_synthetic_mri_grid import (
    SourceGraph,
    complete_grid_positions,
    patch_voxel_cell_bounds,
)


def full_precision_source(directory: Path, inverse_affine: np.ndarray) -> SourceGraph:
    source = SourceGraph.from_directory(directory)
    payload = json.loads((directory / "graph.vvg").read_text())
    precise = {
        int(item["id"]): np.asarray(item["pos"], dtype=np.float64)
        for item in payload["graph"].get("nodes", ())
    }
    if precise:
        source = SourceGraph(
            nodes={node_id: precise.get(node_id, point) for node_id, point in source.nodes.items()},
            edges=source.edges,
            centerlines=source.centerlines,
        )
    return source.transformed(inverse_affine)


def summarize(rows: list[dict], policy: str) -> dict:
    selected = [row for row in rows if row["policy"] == policy and row["edges"] > 0]
    overflow = [row for row in selected if row["nodes"] > 120]
    array = lambda key, group=selected: np.asarray([row[key] for row in group], dtype=np.int64)
    required = array("topology_nodes") + array("boundary_nodes")
    result = {
        "policy": policy,
        "nonempty_patches": len(selected),
        "patches_over_120": len(overflow),
        "node_p95": float(np.percentile(array("nodes"), 95)),
        "node_p99": float(np.percentile(array("nodes"), 99)),
        "node_max": int(array("nodes").max()),
        "topology_plus_boundary_p99": float(np.percentile(required, 99)),
        "topology_plus_boundary_max": int(required.max()),
        "capacity_exceedances": {
            str(capacity): int(np.count_nonzero(array("nodes") > capacity))
            for capacity in (120, 160, 192, 256)
        },
        "without_degree2_token_exceedances": int(np.count_nonzero(required > 120)),
    }
    if overflow:
        result["overflow_mean_composition"] = {
            key: float(np.mean([row[key] for row in overflow]))
            for key in ("topology_nodes", "degree2_nodes", "boundary_nodes")
        }
        result["overflow_max_composition"] = {
            key: int(max(row[key] for row in overflow))
            for key in ("topology_nodes", "degree2_nodes", "boundary_nodes")
        }
    return result


def plot(rows: list[dict], summaries: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    policies = [summary["policy"] for summary in summaries]
    labels = [policy.replace("radius_", "radius ").replace("_c095", "") for policy in policies]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    bins = np.arange(0, max(row["nodes"] for row in rows) + 6, 5)
    for policy, label in zip(policies, labels):
        values = [row["nodes"] for row in rows if row["policy"] == policy and row["edges"] > 0]
        axes[0].hist(values, bins=bins, histtype="step", linewidth=2, label=label)
    axes[0].axvline(120, color="#c0392b", linestyle="--", label="120 tokens")
    axes[0].set(xlabel="Nodes per nonempty patch", ylabel="Patch count", yscale="log")
    axes[0].legend()

    x = np.arange(len(policies))
    bottom = np.zeros(len(policies))
    for key, label, colour in (
        ("topology_nodes", "topological", "#f28e2b"),
        ("degree2_nodes", "degree-2 geometry", "#59a14f"),
        ("boundary_nodes", "patch boundary", "#4e79a7"),
    ):
        values = np.asarray([s["overflow_mean_composition"][key] for s in summaries])
        axes[1].bar(x, values, bottom=bottom, label=label, color=colour)
        bottom += values
    axes[1].axhline(120, color="#c0392b", linestyle="--")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Mean nodes in patches >120")
    axes[1].legend(fontsize=8)

    capacities = (120, 160, 192)
    width = 0.25
    for index, capacity in enumerate(capacities):
        axes[2].bar(
            x + (index - 1) * width,
            [s["capacity_exceedances"][str(capacity)] for s in summaries],
            width,
            label=f"> {capacity}",
        )
    axes[2].set(xticks=x, xticklabels=labels, ylabel="Overflowing patches")
    axes[2].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Exact-crop node-capacity analysis; representation unchanged")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policy", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict] = []
    for marker_path in sorted(args.root.rglob("complete.json")):
        marker = json.loads(marker_path.read_text())
        image = nib.load(marker["label"])
        shape = tuple(int(value) for value in marker["shape"])
        starts = complete_grid_positions(shape, (64, 64, 64), (40, 40, 40))
        for policy in args.policy:
            graph_dir = marker_path.parent / "graphs" / policy
            if not graph_dir.is_dir():
                continue
            source = full_precision_source(graph_dir, np.linalg.inv(image.affine))
            full_degree = {node_id: 0 for node_id in source.nodes}
            for left, right in source.edges:
                full_degree[left] += 1
                full_degree[right] += 1
            node_ids = source.node_ids
            node_positions = source.node_positions
            for patch_index, start in enumerate(starts):
                bounds = patch_voxel_cell_bounds(start, (64, 64, 64))
                inside = np.logical_and(
                    node_positions >= bounds[:, 0], node_positions <= bounds[:, 1]
                ).all(axis=1)
                inside_ids = node_ids[inside]
                topology = sum(full_degree[int(node)] != 2 for node in inside_ids)
                degree2 = sum(full_degree[int(node)] == 2 for node in inside_ids)
                cropped = source.crop(bounds)
                boundary = int(np.count_nonzero(cropped.boundary_intersections))
                rows.append({
                    "policy": policy,
                    "dataset": marker["dataset"],
                    "subject": marker["subject"],
                    "patch_index": patch_index,
                    "start_x": start[0], "start_y": start[1], "start_z": start[2],
                    "nodes": len(cropped.positions), "edges": cropped.edge_count,
                    "topology_nodes": topology,
                    "degree2_nodes": degree2,
                    "boundary_nodes": boundary,
                })
    summaries = [summarize(rows, policy) for policy in args.policy]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (args.output_dir / "patches.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, summaries, args.output_dir / "overflow_analysis.png")
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
