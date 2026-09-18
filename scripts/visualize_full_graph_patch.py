#!/usr/bin/env python3
"""Crop a saved full-volume graph using existing centerline samples and visualize it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_synthetic_mri_grid import (
    CenterlineEdge,
    SourceGraph,
    voxel_to_world,
    world_to_voxel,
)
from scripts.ixi_graph_report_3d import densest_window


def graph_topology(node_count: int, edges: np.ndarray) -> dict[str, int]:
    """Return elementary topology counts, including nodes left isolated by cropping."""
    degree = np.zeros(node_count, dtype=np.int64)
    parent = np.arange(node_count, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for left, right in edges:
        left, right = int(left), int(right)
        degree[left] += 1
        degree[right] += 1
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left
    components = len({find(node) for node in range(node_count)})
    return {
        "components": int(components),
        "cycle_rank": int(len(edges) - node_count + components),
        "isolated_nodes": int(np.count_nonzero(degree == 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject")
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--geometry-image", type=Path, default=None)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--mesh-step", type=int, default=2)
    args = parser.parse_args()

    segmentation_image = nib.load(str(args.segmentation))
    geometry_image = nib.load(str(args.geometry_image)) if args.geometry_image else segmentation_image
    if segmentation_image.shape != geometry_image.shape:
        parser.error("segmentation and geometry image shapes differ")
    mask = np.asanyarray(segmentation_image.dataobj) > 0
    half = args.patch_size // 2
    window = densest_window(mask, half, half)
    start = np.asarray([item.start for item in window], dtype=np.int64)
    patch = mask[window]
    voxel_bounds = np.stack((start, start + np.asarray(patch.shape) - 1), axis=1)

    source_world = SourceGraph.from_directory(args.graph_dir)
    source = SourceGraph(
        nodes={
            node_id: world_to_voxel(position[None], geometry_image.affine)[0]
            for node_id, position in source_world.nodes.items()
        },
        edges=source_world.edges,
        centerlines=[
            CenterlineEdge(
                node1=edge.node1,
                node2=edge.node2,
                positions=tuple(
                    world_to_voxel(np.asarray(edge.positions), geometry_image.affine)
                ),
            )
            for edge in source_world.centerlines
        ],
    )
    # Requested policy: existing samples only. No interpolated box-face node.
    cropped_voxel = source.crop_inherited(voxel_bounds)
    cropped_positions_world = voxel_to_world(cropped_voxel.positions, geometry_image.affine)
    topology = graph_topology(len(cropped_voxel.positions), cropped_voxel.edges)

    from skimage.measure import marching_cubes
    vertices_voxel, faces, _, _ = marching_cubes(
        patch.astype(np.float32), level=0.5, step_size=args.mesh_step
    )
    vertices_world = voxel_to_world(vertices_voxel + start, geometry_image.affine)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"patch64_{args.subject}_last_sample"
    summary = {
        "subject": args.subject,
        "source_graph": str(args.graph_dir.resolve()),
        "segmentation": str(args.segmentation.resolve()),
        "policy": "full_volume_extraction_then_crop_last_existing_centerline_sample",
        "creates_interpolated_boundary_nodes": False,
        "patch_shape": list(patch.shape),
        "start_voxel": start.tolist(),
        "voxel_bounds": voxel_bounds.tolist(),
        "vessel_voxels": int(patch.sum()),
        "nodes": int(len(cropped_voxel.positions)),
        "edges": int(cropped_voxel.edge_count),
        **topology,
    }
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")

    import plotly.graph_objects as go
    figure = go.Figure(go.Mesh3d(
        x=vertices_world[:, 0], y=vertices_world[:, 1], z=vertices_world[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color="#9fb3c8",
        opacity=0.18, name="segmentation", hoverinfo="skip",
    ))
    x, y, z = [], [], []
    for left, right in cropped_voxel.edges:
        line = cropped_positions_world[[left, right]]
        x.extend([line[0, 0], line[1, 0], None])
        y.extend([line[0, 1], line[1, 1], None])
        z.extend([line[0, 2], line[1, 2], None])
    figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name="cropped graph",
                                  line={"color": "#c0392b", "width": 4}))
    if len(cropped_positions_world):
        figure.add_trace(go.Scatter3d(
            x=cropped_positions_world[:, 0], y=cropped_positions_world[:, 1], z=cropped_positions_world[:, 2],
            mode="markers", name="existing graph nodes / boundary samples",
            marker={"color": "#1f77b4", "size": 3},
        ))
    figure.update_layout(
        title=f"{args.subject}: full graph cropped to 64³ (last existing sample)",
        scene={"aspectmode": "data", "xaxis_title": "x (mm)",
               "yaxis_title": "y (mm)", "zaxis_title": "z (mm)"},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )
    figure.write_html(stem.with_suffix(".html"), include_plotlyjs=True)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    static = plt.figure(figsize=(8, 7), constrained_layout=True)
    axis = static.add_subplot(111, projection="3d")
    stride = max(1, len(faces) // 8_000)
    axis.add_collection3d(Poly3DCollection(
        vertices_world[faces[::stride]], facecolor="#9fb3c8", edgecolor="none", alpha=0.08
    ))
    for left, right in cropped_voxel.edges:
        line = cropped_positions_world[[left, right]]
        axis.plot(line[:, 0], line[:, 1], line[:, 2], color="#c0392b", linewidth=1)
    if len(cropped_positions_world):
        axis.scatter(*cropped_positions_world.T, c="#1f77b4", s=6, depthshade=False)
    extent = vertices_world.max(axis=0) - vertices_world.min(axis=0)
    axis.set_box_aspect(np.maximum(extent, 1e-6))
    axis.set_title(f"{args.subject}: full graph → 64³ last-sample crop")
    axis.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
    axis.view_init(elev=24, azim=-58)
    static.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(static)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
