#!/usr/bin/env python3
"""Derive stronger adaptive policies from saved dense graphs without reskeletonizing."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import time

import nibabel as nib
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_synthetic_mri_grid import (
    SourceGraph,
    complete_grid_positions,
    patch_voxel_cell_bounds,
    world_to_voxel,
)
from scripts.ixi_vessel_graph import (
    DenseCenterline,
    VesselGraph,
    derive_graph_from_dense,
    write_source_graph,
)
from scripts.sweep_real_graph_representations import distribution


POLICIES = [
    {"name": "fixed_4v_c095", "tolerance_voxels": 4.0, "radius_fraction": 0.0},
    {"name": "fixed_5v_c095", "tolerance_voxels": 5.0, "radius_fraction": 0.0},
    {"name": "fixed_6v_c095", "tolerance_voxels": 6.0, "radius_fraction": 0.0},
    {"name": "radius_1x_c095", "tolerance_voxels": 0.0, "radius_fraction": 1.0},
    {"name": "radius_1p5x_c095", "tolerance_voxels": 0.0, "radius_fraction": 1.5},
    {"name": "radius_2x_c095", "tolerance_voxels": 0.0, "radius_fraction": 2.0},
]


def load_dense(directory: Path, affine: np.ndarray, mask: np.ndarray, spacing: np.ndarray) -> DenseCenterline:
    source = SourceGraph.from_directory(directory)
    node_ids = list(source.nodes)
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    # Older stage-1 nodes.csv files used six decimal places, which can move a
    # reloaded world-coordinate anchor across a voxel-cell face. graph.vvg keeps
    # full precision, so use its node payload when resuming from those artifacts.
    payload = json.loads((directory / "graph.vvg").read_text())
    precise_nodes = {
        int(item["id"]): np.asarray(item["pos"], dtype=np.float64)
        for item in payload["graph"].get("nodes", ())
    }
    world_positions = np.asarray(
        [precise_nodes.get(node_id, source.nodes[node_id]) for node_id in node_ids]
    )
    positions = world_to_voxel(world_positions, affine)
    edges = [(index[left], index[right]) for left, right in source.edges]
    degrees = np.zeros(len(positions), dtype=np.int64)
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    centerlines = [world_to_voxel(polyline, affine) for polyline in source.edge_polylines]
    graph = VesselGraph(
        node_positions=positions,
        node_degrees=degrees,
        edges=edges,
        centerlines=centerlines,
        node_radii=np.zeros(len(positions), dtype=np.float64),
    )
    return DenseCenterline(graph=graph, segmentation=mask, spacing=spacing,
                           provenance={"source": "saved_stage1_dense_graph"})


def process(row: dict[str, str], args) -> str:
    relative = Path(row["dataset"]) / row["modality"] / row["split"] / row["subject"]
    source_subject = args.stage1_root / relative
    target = args.output / relative
    marker = target / "complete.json"
    if marker.is_file() and not args.force:
        return "skipped"
    if marker.is_file() and args.force:
        # Invalidate the previous checkpoint before overwriting any artifacts. If
        # this process is interrupted, a later non-force resume must rerun the
        # subject instead of accepting a stale completion record.
        marker.unlink()
    started = time.monotonic()
    label = nib.load(row["label"])
    mask = np.asanyarray(label.dataobj) > 0
    spacing = np.sqrt((label.affine[:3, :3] ** 2).sum(axis=0))
    dense = load_dense(source_subject / "graphs" / "dense", label.affine, mask, spacing)
    graphs = {}
    for policy in POLICIES:
        graphs[policy["name"]] = derive_graph_from_dense(
            dense,
            representation="adaptive",
            adaptive_tolerance_mm=policy["tolerance_voxels"] * float(spacing.min()),
            adaptive_radius_fraction=policy["radius_fraction"],
            minimum_chord_fraction=0.95,
        )
    for name, graph in graphs.items():
        write_source_graph(target / "graphs" / name, graph, label.affine, label.shape)

    starts = complete_grid_positions(label.shape, (64, 64, 64), (40, 40, 40))
    patch_metrics, patch_rows = {}, []
    with tempfile.TemporaryDirectory(prefix="graph-stage2-crop-") as temporary:
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
            array = np.asarray(counts, dtype=np.int64)
            nonempty = array[:, 1] > 0
            patch_metrics[name] = {
                "patches": len(array), "nonempty_patches": int(nonempty.sum()),
                "nodes_nonempty": distribution(array[nonempty, 0]),
                "edges_nonempty": distribution(array[nonempty, 1]),
            }
    target.mkdir(parents=True, exist_ok=True)
    with (target / "patch_counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(patch_rows[0]))
        writer.writeheader(); writer.writerows(patch_rows)
    dense_record = json.loads((source_subject / "complete.json").read_text())["representations"]["dense"]
    payload = {
        **{key: row[key] for key in ("index", "dataset", "modality", "split", "subject", "image", "label")},
        "configuration": {"policies": POLICIES, "minimum_chord_fraction": 0.95,
                          "dense_source": str((source_subject / "graphs" / "dense").resolve())},
        "dense_extractions": 0,
        "shape": list(label.shape), "spacing_mm": spacing.tolist(),
        "foreground_voxels": int(mask.sum()), "patch_audit": patch_metrics,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "representations": {
            "dense": dense_record,
            **{name: {"nodes": graph.node_count, "edges": graph.edge_count,
                      "degree_2": int((graph.node_degrees == 2).sum()),
                      "betti_0": graph.betti()[0], "betti_1": graph.betti()[1]}
               for name, graph in graphs.items()},
        },
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return "done"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject", action="append", dest="subjects",
                        help="Process only this subject (repeatable).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    with args.manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.subjects:
        requested = set(args.subjects)
        rows = [row for row in rows if row["subject"] in requested]
    failed = []
    for row in rows:
        stage1_marker = args.stage1_root / row["dataset"] / row["modality"] / row["split"] / row["subject"] / "complete.json"
        if not stage1_marker.is_file():
            print(row["subject"], "omitted: no stage-1 result", flush=True)
            continue
        try:
            print(row["subject"], process(row, args), flush=True)
        except Exception as error:
            failed.append(row["subject"])
            print(row["subject"], "FAILED", repr(error), flush=True)
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
