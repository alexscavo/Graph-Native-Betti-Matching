#!/usr/bin/env python3
"""Compare exact and inherited crop policies for one saved real-data graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.crop_comparison import compare_cropped_graphs
from metrics.representation_geometry import (
    average_centerline_distance,
    curve_precision_recall_f1,
    hd95,
)
from scripts.audit_synthetic_mri_grid import (
    CenterlineEdge,
    SourceGraph,
    voxel_to_world,
    world_to_voxel,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    bounds_group = parser.add_mutually_exclusive_group(required=True)
    bounds_group.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="closed crop bounds in the same physical coordinates as the graph",
    )
    bounds_group.add_argument(
        "--voxel-bounds",
        type=float,
        nargs=6,
        metavar=("IMIN", "IMAX", "JMIN", "JMAX", "KMIN", "KMAX"),
        help="closed voxel bounds; requires --geometry-image",
    )
    parser.add_argument(
        "--geometry-image",
        type=Path,
        help="NIfTI defining the world-to-voxel transform for --voxel-bounds",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-spacing", type=float, default=0.25)
    parser.add_argument("--curve-tolerance", type=float, default=0.5)
    parser.add_argument("--correspondence-tolerance", type=float, default=2.0)
    parser.add_argument("--preferred-edge-budget", type=int, default=70)
    parser.add_argument("--hard-token-budget", type=int, default=120)
    args = parser.parse_args()

    graph = SourceGraph.from_directory(args.graph_dir)
    if args.voxel_bounds is not None:
        if args.geometry_image is None:
            parser.error("--voxel-bounds requires --geometry-image")
        affine = nib.load(str(args.geometry_image)).affine
        graph = SourceGraph(
            nodes={
                node_id: world_to_voxel(position[None], affine)[0]
                for node_id, position in graph.nodes.items()
            },
            edges=graph.edges,
            centerlines=[
                CenterlineEdge(
                    node1=edge.node1,
                    node2=edge.node2,
                    positions=tuple(world_to_voxel(np.asarray(edge.positions), affine)),
                )
                for edge in graph.centerlines
            ],
        )
        bounds = np.asarray(args.voxel_bounds, dtype=np.float64).reshape(3, 2)
        coordinate_system = "voxel"
    else:
        bounds = np.asarray(args.bounds, dtype=np.float64).reshape(3, 2)
        coordinate_system = "graph_physical"
    exact = graph.crop(bounds)
    inherited = graph.crop_inherited(bounds)
    report = compare_cropped_graphs(
        exact.positions,
        exact.edges,
        inherited.positions,
        inherited.edges,
        bounds,
        sample_spacing=args.sample_spacing,
        curve_tolerance=args.curve_tolerance,
        correspondence_tolerance=args.correspondence_tolerance,
        preferred_edge_budget=args.preferred_edge_budget,
        hard_token_budget=args.hard_token_budget,
    )
    if args.voxel_bounds is not None:
        exact_world = voxel_to_world(exact.positions, affine)
        inherited_world = voxel_to_world(inherited.positions, affine)
        exact_curves = [exact_world[edge] for edge in exact.edges]
        inherited_curves = [inherited_world[edge] for edge in inherited.edges]
        if exact_curves and inherited_curves:
            # Use the smallest voxel-axis scale to avoid under-sampling physical
            # curves when an affine is anisotropic or oblique.
            physical_spacing = args.sample_spacing * float(
                np.linalg.norm(affine[:3, :3], axis=0).min()
            )
            curve = curve_precision_recall_f1(
                inherited_curves,
                exact_curves,
                args.curve_tolerance,
                physical_spacing,
            )
            report["geometry_physical_mm"] = {
                "curve_precision": curve["precision"],
                "curve_recall": curve["recall"],
                "curve_f1": curve["f1"],
                "average_centerline_distance": average_centerline_distance(
                    inherited_curves, exact_curves, physical_spacing
                ),
                "hd95": hd95(inherited_curves, exact_curves, physical_spacing),
                "sample_spacing_mm": physical_spacing,
                "curve_tolerance_mm": args.curve_tolerance,
            }
    report["graph_dir"] = str(args.graph_dir.resolve())
    report["bounds"] = bounds.tolist()
    report["coordinate_system"] = coordinate_system
    if args.geometry_image:
        report["geometry_image"] = str(args.geometry_image.resolve())
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
