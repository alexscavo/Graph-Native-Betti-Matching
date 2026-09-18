#!/usr/bin/env python3
"""Run compact-graph policies while extracting each dense centerline only once."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import nibabel as nib
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_synthetic_mri_grid import SourceGraph, complete_grid_positions, patch_voxel_cell_bounds
from scripts.ixi_vessel_graph import derive_graph_from_dense, extract_dense_centerline, write_source_graph


POLICIES = [
    {"name": "fixed_2v_c100", "representation": "adaptive", "tolerance_voxels": 2.0, "adaptive_radius_fraction": 0.0, "minimum_chord_fraction": 1.00},
    {"name": "fixed_2p5v_c099", "representation": "adaptive", "tolerance_voxels": 2.5, "adaptive_radius_fraction": 0.0, "minimum_chord_fraction": 0.99},
    {"name": "fixed_3v_c099", "representation": "adaptive", "tolerance_voxels": 3.0, "adaptive_radius_fraction": 0.0, "minimum_chord_fraction": 0.99},
    {"name": "fixed_3v_c098", "representation": "adaptive", "tolerance_voxels": 3.0, "adaptive_radius_fraction": 0.0, "minimum_chord_fraction": 0.98},
    {"name": "fixed_4v_c098", "representation": "adaptive", "tolerance_voxels": 4.0, "adaptive_radius_fraction": 0.0, "minimum_chord_fraction": 0.98},
    {"name": "radius_0p25x_c099", "representation": "adaptive", "tolerance_voxels": 0.0, "adaptive_radius_fraction": 0.25, "minimum_chord_fraction": 0.99},
    {"name": "radius_0p5x_c099", "representation": "adaptive", "tolerance_voxels": 0.0, "adaptive_radius_fraction": 0.50, "minimum_chord_fraction": 0.99},
    {"name": "radius_0p75x_c098", "representation": "adaptive", "tolerance_voxels": 0.0, "adaptive_radius_fraction": 0.75, "minimum_chord_fraction": 0.98},
]
SCHEMA_VERSION = 1


def config(args) -> dict:
    return {"schema_version": SCHEMA_VERSION, "policies": POLICIES,
            "centerline_backend": "legacy", "spur_length": args.spur_length,
            "max_junction_extent_mm": args.max_junction_extent_mm,
            "smooth_iterations": args.smooth_iterations, "smooth_alpha": args.smooth_alpha,
            "patch_size_voxels": 64, "patch_stride_voxels": 40,
            "patch_bounds": "exact voxel-cell faces [-0.5, size-0.5]"}


def distribution(values: np.ndarray) -> dict:
    if not len(values):
        return {"mean": 0.0, "p95": 0.0, "p99": 0.0, "max": 0, "over_70": 0, "over_120": 0}
    return {"mean": float(values.mean()), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "max": int(values.max()),
            "over_70": int((values > 70).sum()), "over_120": int((values > 120).sum())}


def process(row: dict[str, str], args) -> str:
    target = args.output / row["dataset"] / row["modality"] / row["split"] / row["subject"]
    marker = target / "complete.json"
    configuration = config(args)
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:16]
    expected = ["dense", *(policy["name"] for policy in POLICIES)]
    if marker.is_file() and not args.force:
        old = json.loads(marker.read_text())
        complete = old.get("configuration_fingerprint") == fingerprint
        complete &= all(all((target / "graphs" / name / filename).is_file()
                            for filename in ("nodes.csv", "edges.csv", "graph.vvg")) for name in expected)
        if complete: return "skipped"
    started = time.monotonic()
    image, label = nib.load(row["image"]), nib.load(row["label"])
    if image.shape != label.shape or not np.allclose(image.affine, label.affine, atol=1e-4):
        raise ValueError("image/label geometry mismatch")
    mask = np.asanyarray(label.dataobj) > 0
    spacing = np.sqrt((label.affine[:3, :3] ** 2).sum(axis=0))
    dense = extract_dense_centerline(mask, spacing=spacing, centerline_backend="legacy",
                                     spur_length=args.spur_length,
                                     max_junction_extent_mm=args.max_junction_extent_mm)
    graphs = {"dense": dense.graph}
    for policy in POLICIES:
        options = {key: value for key, value in policy.items() if key not in {"name", "tolerance_voxels"}}
        options["adaptive_tolerance_mm"] = policy["tolerance_voxels"] * float(spacing.min())
        graphs[policy["name"]] = derive_graph_from_dense(
            dense, smooth_iterations=args.smooth_iterations, smooth_alpha=args.smooth_alpha,
            **options)
    for name, graph in graphs.items(): write_source_graph(target / "graphs" / name, graph, label.affine, label.shape)
    starts = complete_grid_positions(label.shape, (64, 64, 64), (40, 40, 40))
    patch_metrics, patch_rows = {}, []
    with tempfile.TemporaryDirectory(prefix="graph-sweep-crop-") as temporary:
        for name, graph in graphs.items():
            directory = Path(temporary) / name
            write_source_graph(directory, graph, np.eye(4), label.shape)
            source = SourceGraph.from_directory(directory)
            counts = []
            for patch_index, start in enumerate(starts):
                cropped = source.crop(patch_voxel_cell_bounds(start, (64, 64, 64)))
                nodes, edges = len(cropped.positions), cropped.edge_count
                counts.append((nodes, edges))
                patch_rows.append({"policy": name, "patch_index": patch_index,
                                   "start_x": start[0], "start_y": start[1], "start_z": start[2],
                                   "nodes": nodes, "edges": edges})
            counts_array = np.asarray(counts, dtype=np.int64)
            nonempty = counts_array[:, 1] > 0
            patch_metrics[name] = {"patches": len(counts), "nonempty_patches": int(nonempty.sum()),
                                   "nodes_nonempty": distribution(counts_array[nonempty, 0]),
                                   "edges_nonempty": distribution(counts_array[nonempty, 1])}
    target.mkdir(parents=True, exist_ok=True)
    patch_path = target / "patch_counts.csv"; patch_temporary = patch_path.with_suffix(".tmp")
    with patch_temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(patch_rows[0])); writer.writeheader(); writer.writerows(patch_rows)
    os.replace(patch_temporary, patch_path)
    payload = {**{key: row[key] for key in ("index", "dataset", "modality", "split", "subject", "image", "label")},
               "configuration": configuration, "configuration_fingerprint": fingerprint,
               "dense_extractions": 1, "shape": list(label.shape), "spacing_mm": spacing.tolist(),
               "foreground_voxels": int(mask.sum()), "elapsed_seconds": time.monotonic() - started,
               "patch_audit": patch_metrics,
               "completed_at": datetime.now(timezone.utc).isoformat(),
               "representations": {name: {"nodes": graph.node_count, "edges": graph.edge_count,
                  "degree_2": int((graph.node_degrees == 2).sum()), "betti_0": graph.betti()[0],
                  "betti_1": graph.betti()[1]} for name, graph in graphs.items()}}
    target.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".tmp"); temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, marker)
    return "done"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=int(os.getenv("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--num-shards", type=int, default=int(os.getenv("REAL_SWEEP_NUM_SHARDS", "1")))
    parser.add_argument("--force", action="store_true"); parser.add_argument("--spur-length", type=int, default=4)
    parser.add_argument("--max-junction-extent-mm", type=float, default=1.5)
    parser.add_argument("--smooth-iterations", type=int, default=5); parser.add_argument("--smooth-alpha", type=float, default=0.5)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards: parser.error("invalid shard")
    with args.manifest.open(newline="") as stream: rows = list(csv.DictReader(stream))
    assigned = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard]
    failures = []
    for row in assigned:
        try: print(row["subject"], process(row, args), flush=True)
        except Exception as error: failures.append(row["subject"]); print(row["subject"], "FAILED", repr(error), flush=True)
    if failures: raise SystemExit(f"failed: {', '.join(failures)}")


if __name__ == "__main__": main()
