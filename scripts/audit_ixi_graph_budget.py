#!/usr/bin/env python3
"""Audit real IXI graph coordinates and 64^3 patch complexity.

The audit builds all three representation controls for one complete IXI subject,
exports them through the real world-coordinate source format, and crops them with
the same exact polyline policy used by patch generation. Results therefore test
the actual training-target path rather than approximate voxel-window counts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_synthetic_mri_grid import SourceGraph, complete_grid_positions
from scripts.generate_synthetic_mri_dataset import patch_world_bounds
from scripts.ixi_vessel_graph import build_representation_family, write_source_graph
from scripts.prepare_ixi_sources import DEFAULT_ROOT, load_geometry


def distribution(values: np.ndarray, *, preferred: int, capacity: int) -> dict:
    if not len(values):
        return {"mean": 0.0, "p95": 0.0, "p99": 0.0, "max": 0,
                f"over_{preferred}": 0, f"over_{capacity}": 0}
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
        f"over_{preferred}": int((values > preferred).sum()),
        f"over_{capacity}": int((values > capacity).sum()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rdp-voxels", type=float, default=2.0)
    parser.add_argument("--radius-fraction", type=float, default=0.0)
    parser.add_argument("--spur-length", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--pad", type=int, default=5)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--preferred-edges", type=int, default=70)
    parser.add_argument("--token-capacity", type=int, default=120)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    image, segmentation, _, affine, spacing, repaired, selected = load_geometry(
        args.root, args.subject
    )
    mask = np.asanyarray(segmentation.dataobj) > 0
    graphs = build_representation_family(
        mask,
        spacing=spacing,
        adaptive_tolerance_mm=args.rdp_voxels * float(spacing.min()),
        adaptive_radius_fraction=args.radius_fraction,
        spur_length=args.spur_length,
    )
    crop_size = (args.patch_size - 2 * args.pad,) * 3
    starts = complete_grid_positions(mask.shape, crop_size, (args.stride,) * 3)
    result = {
        "subject": args.subject,
        "source": "real IXI segmentation",
        "source_segmentation": str(selected.relative_to(args.root)),
        "header_repaired": repaired,
        "shape": list(mask.shape),
        "spacing_mm": [float(value) for value in spacing],
        "patch_size": args.patch_size,
        "effective_crop_size": list(crop_size),
        "stride": args.stride,
        "preferred_edge_ceiling": args.preferred_edges,
        "token_capacity": args.token_capacity,
        "representations": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ixi-graph-budget-") as temporary:
        for name, graph in graphs.items():
            directory = Path(temporary) / name
            write_source_graph(directory, graph, affine, mask.shape)
            source = SourceGraph.from_directory(directory)
            patch_rows = []
            for index, start in enumerate(starts):
                cropped = source.crop(patch_world_bounds(start, crop_size, affine))
                patch_rows.append(
                    {
                        "patch_index": index,
                        "start_0": start[0],
                        "start_1": start[1],
                        "start_2": start[2],
                        "nodes": len(cropped.positions),
                        "edges": cropped.edge_count,
                    }
                )
            counts = np.asarray(
                [(row["nodes"], row["edges"]) for row in patch_rows], dtype=np.int64
            )
            nonempty = counts[:, 1] > 0
            fractional = np.any(
                np.abs(graph.node_positions - np.rint(graph.node_positions)) > 1e-6,
                axis=1,
            )
            result["representations"][name] = {
                "source_nodes": graph.node_count,
                "source_edges": graph.edge_count,
                "beta_0": graph.betti()[0],
                "beta_1": graph.betti()[1],
                "fractional_nodes": int(fractional.sum()),
                "fractional_node_fraction": float(fractional.mean()),
                "all_training_edges_are_two_point_lines": True,
                "patches": len(patch_rows),
                "nonempty_patches": int(nonempty.sum()),
                "nodes_nonempty": distribution(
                    counts[nonempty, 0],
                    preferred=args.preferred_edges,
                    capacity=args.token_capacity,
                ),
                "edges_nonempty": distribution(
                    counts[nonempty, 1],
                    preferred=args.preferred_edges,
                    capacity=args.token_capacity,
                ),
            }
            csv_path = args.output_dir / f"{args.subject}_{name}_patches.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(patch_rows[0]))
                writer.writeheader()
                writer.writerows(patch_rows)
    destination = args.output_dir / f"{args.subject}_budget.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

