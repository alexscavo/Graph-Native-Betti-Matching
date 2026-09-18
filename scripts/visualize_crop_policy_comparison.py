#!/usr/bin/env python3
"""Visualize exact versus inherited graph cropping on a real segmentation patch."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_synthetic_mri_grid import CenterlineEdge, SourceGraph, world_to_voxel


def _degrees(node_count: int, edges: np.ndarray) -> np.ndarray:
    return np.bincount(edges.ravel(), minlength=node_count) if len(edges) else np.zeros(node_count, dtype=int)


def _matched_inherited_terminals(exact, inherited, tolerance: float) -> np.ndarray:
    exact_ids = np.flatnonzero(exact.boundary_intersections)
    candidates = np.flatnonzero(_degrees(len(inherited.positions), inherited.edges) <= 1)
    pairs = []
    for left in exact_ids:
        for right in candidates:
            pairs.append((float(np.linalg.norm(exact.positions[left] - inherited.positions[right])), int(left), int(right)))
    used_left, used_right = set(), set()
    for distance, left, right in sorted(pairs):
        if distance > tolerance:
            break
        if left not in used_left and right not in used_right:
            used_left.add(left)
            used_right.add(right)
    return np.asarray(sorted(used_right), dtype=np.int64)


def _edge_xyz(graph) -> tuple[list[float | None], list[float | None], list[float | None]]:
    xyz = [[], [], []]
    for left, right in graph.edges:
        points = graph.positions[[left, right]]
        for axis in range(3):
            xyz[axis].extend((float(points[0, axis]), float(points[1, axis]), None))
    return xyz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--voxel-bounds", type=float, nargs=6, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--correspondence-tolerance", type=float, default=3.0)
    parser.add_argument("--mesh-step", type=int, default=2)
    args = parser.parse_args()

    image = nib.load(str(args.segmentation))
    mask = np.asanyarray(image.dataobj) > 0
    bounds = np.asarray(args.voxel_bounds, dtype=np.float64).reshape(3, 2)
    starts = np.rint(bounds[:, 0] + 0.5).astype(int)
    stops = np.rint(bounds[:, 1] + 0.5).astype(int)
    patch = mask[tuple(slice(int(a), int(b)) for a, b in zip(starts, stops))]
    vertices, faces, _, _ = marching_cubes(
        patch.astype(np.float32), 0.5, step_size=args.mesh_step
    )
    vertices += starts

    world = SourceGraph.from_directory(args.graph_dir)
    graph = SourceGraph(
        nodes={node: world_to_voxel(point[None], image.affine)[0] for node, point in world.nodes.items()},
        edges=world.edges,
        centerlines=[
            CenterlineEdge(
                edge.node1,
                edge.node2,
                tuple(world_to_voxel(np.asarray(edge.positions), image.affine)),
            )
            for edge in world.centerlines
        ],
    )
    exact = graph.crop(bounds)
    inherited = graph.crop_inherited(bounds)
    exact_ids = np.flatnonzero(exact.boundary_intersections)
    inherited_ids = _matched_inherited_terminals(exact, inherited, args.correspondence_tolerance)

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 8))
    for index, (title, cropped, ids, color, label) in enumerate(
        (
            ("Exact cell-face clipping", exact, exact_ids, "#d62728", "interpolated face intersections"),
            ("Inherited last-inside samples", inherited, inherited_ids, "#ff7f0e", "matched last-inside endpoints"),
        ),
        start=1,
    ):
        axis = fig.add_subplot(1, 2, index, projection="3d")
        axis.plot_trisurf(*vertices.T, triangles=faces, color="#9aa0a6", alpha=0.12, linewidth=0)
        for left, right in cropped.edges:
            points = cropped.positions[[left, right]]
            axis.plot(*points.T, color="#276fbf", linewidth=1.1, alpha=0.9)
        if len(ids):
            axis.scatter(*cropped.positions[ids].T, color=color, s=26, depthshade=False, label=label)
        axis.set_title(f"{title}\n{len(cropped.positions)} nodes, {len(cropped.edges)} edges")
        axis.set_xlabel("voxel i")
        axis.set_ylabel("voxel j")
        axis.set_zlabel("voxel k")
        axis.set_box_aspect(np.ptp(bounds, axis=1))
        axis.legend(loc="upper right", fontsize=8)
    fig.suptitle("Real vessel segmentation: crop-boundary policy comparison")
    fig.tight_layout()
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Exact cell-face clipping", "Inherited last-inside samples"),
    )
    for column, (cropped, ids, color, label) in enumerate(
        ((exact, exact_ids, "#d62728", "exact intersections"), (inherited, inherited_ids, "#ff7f0e", "last-inside endpoints")),
        start=1,
    ):
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color="#9aa0a6", opacity=0.12, name="segmentation", showlegend=column == 1,
            ), row=1, col=column,
        )
        x, y, z = _edge_xyz(cropped)
        figure.add_trace(
            go.Scatter3d(x=x, y=y, z=z, mode="lines", line={"color": "#276fbf", "width": 3}, name="graph edges", showlegend=column == 1),
            row=1, col=column,
        )
        points = cropped.positions[ids]
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0] if len(points) else [], y=points[:, 1] if len(points) else [], z=points[:, 2] if len(points) else [],
                mode="markers", marker={"color": color, "size": 5}, name=label,
            ), row=1, col=column,
        )
    figure.update_layout(
        title="Real segmentation and crop-boundary correspondence",
        height=760,
        width=1450,
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
    )
    figure.write_html(args.output_stem.with_suffix(".html"), include_plotlyjs=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
