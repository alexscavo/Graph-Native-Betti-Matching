#!/usr/bin/env python3
"""Audit straight model edges against saved real-data centerline geometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_synthetic_mri_grid import SourceGraph, world_to_voxel
from scripts.ixi_vessel_graph import chord_mask_fraction


def parse_candidate(value: str) -> tuple[str, Path, str]:
    parts = value.split("=", 1)
    if len(parts) != 2 or ":" not in parts[1]:
        raise argparse.ArgumentTypeError("expected NAME=ROOT:POLICY")
    root, policy = parts[1].rsplit(":", 1)
    return parts[0], Path(root), policy


def point_segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = end - start
    squared = float(direction @ direction)
    if squared <= 1e-16:
        return np.linalg.norm(points - start, axis=1)
    fraction = np.clip(((points - start) @ direction) / squared, 0.0, 1.0)
    return np.linalg.norm(points - (start + fraction[:, None] * direction), axis=1)


def distributions(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def audit_candidate(name: str, root: Path, policy: str) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    markers = sorted(root.rglob("complete.json"))
    for marker_path in markers:
        marker = json.loads(marker_path.read_text())
        graph_dir = marker_path.parent / "graphs" / policy
        if not graph_dir.is_dir():
            continue
        image = nib.load(marker["label"])
        mask = np.asanyarray(image.dataobj) > 0
        spacing = np.sqrt((image.affine[:3, :3] ** 2).sum(axis=0))
        radius_map = distance_transform_edt(mask, sampling=spacing)
        world = SourceGraph.from_directory(graph_dir)
        # Legacy nodes.csv artifacts were rounded to six decimals, while graph.vvg
        # retained full-precision nodes. Use the latter for a fair geometry audit.
        payload = json.loads((graph_dir / "graph.vvg").read_text())
        precise = {
            int(item["id"]): np.asarray(item["pos"], dtype=np.float64)
            for item in payload["graph"].get("nodes", ())
        }
        if precise:
            world = SourceGraph(
                nodes={node_id: precise.get(node_id, position) for node_id, position in world.nodes.items()},
                edges=world.edges,
                centerlines=world.centerlines,
            )
        voxel = world.transformed(np.linalg.inv(image.affine))
        for edge_index, (world_line, voxel_line) in enumerate(
            zip(world.edge_polylines, voxel.edge_polylines)
        ):
            start, end = world_line[0], world_line[-1]
            errors = point_segment_distances(world_line, start, end)
            dense_length = float(np.linalg.norm(np.diff(world_line, axis=0), axis=1).sum())
            chord_length = float(np.linalg.norm(end - start))
            indices = np.clip(
                np.floor(voxel_line + 0.5).astype(np.int64),
                0,
                np.asarray(mask.shape) - 1,
            )
            radii = radius_map[indices[:, 0], indices[:, 1], indices[:, 2]]
            positive = radii[radii > 0]
            median_radius = float(np.median(positive)) if len(positive) else float("nan")
            containment = chord_mask_fraction(voxel_line[0], voxel_line[-1], mask)
            rows.append({
                "candidate": name,
                "dataset": marker["dataset"],
                "subject": marker["subject"],
                "edge_index": edge_index,
                "dense_samples": len(world_line),
                "dense_length_mm": dense_length,
                "chord_length_mm": chord_length,
                "relative_length_error": ((dense_length - chord_length) / dense_length) if dense_length else 0.0,
                "mean_approximation_error_mm": float(errors.mean()),
                "max_approximation_error_mm": float(errors.max()),
                "median_radius_mm": median_radius,
                "max_error_over_median_radius": float(errors.max() / median_radius) if median_radius > 0 else float("nan"),
                "containment_fraction": containment,
            })
    if not rows:
        raise ValueError(f"no graphs found for {name}: {root}:{policy}")
    length = np.asarray([row["dense_length_mm"] for row in rows])
    weights = length / length.sum()
    max_error = np.asarray([row["max_approximation_error_mm"] for row in rows])
    normalized = np.asarray([row["max_error_over_median_radius"] for row in rows])
    containment = np.asarray([row["containment_fraction"] for row in rows])
    relative_length = np.asarray([row["relative_length_error"] for row in rows])
    summary = {
        "candidate": name,
        "subjects": len({row["subject"] for row in rows}),
        "edges": len(rows),
        "maximum_error_mm": distributions(max_error),
        "max_error_over_median_radius": distributions(normalized[np.isfinite(normalized)]),
        "length_weighted_relative_length_error": float(weights @ relative_length),
        "containment_mean": float(containment.mean()),
        "containment_p05": float(np.percentile(containment, 5)),
        "edges_below_95pct_containment": int((containment < 0.95 - 1e-12).sum()),
        "fully_contained_edges": int((containment == 1.0).sum()),
    }
    return rows, summary


def plot(summaries: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["candidate"] for row in summaries]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].bar(labels, [row["maximum_error_mm"]["p95"] for row in summaries], color="#4c78a8")
    axes[0].set_ylabel("P95 edge maximum error (mm)")
    axes[1].bar(labels, [100 * row["length_weighted_relative_length_error"] for row in summaries], color="#f58518")
    axes[1].set_ylabel("Length-weighted shortening (%)")
    axes[2].bar(labels, [100 * row["containment_mean"] for row in summaries], color="#54a24b")
    axes[2].set_ylabel("Mean chord containment (%)")
    axes[2].set_ylim(95, 100.1)
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Real-data model-edge geometry audit")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True,
                        help="NAME=ROOT:POLICY (repeatable)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    all_rows, summaries = [], []
    for candidate in args.candidate:
        rows, summary = audit_candidate(*candidate)
        all_rows.extend(rows)
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (args.output_dir / "edges.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    plot(summaries, args.output_dir / "geometry_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
