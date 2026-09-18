#!/usr/bin/env python3
"""Visualize the adaptive model graph with every graph-node type visible."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_synthetic_mri_grid import SourceGraph, voxel_to_world


NODE_STYLES = (
    ("isolated", lambda degree: degree == 0, "#d627b0", 18),
    ("termination (degree 1)", lambda degree: degree == 1, "#1f77b4", 14),
    ("adaptive subdivision (degree 2)", lambda degree: degree == 2, "#2ca02c", 9),
    ("junction (degree >= 3)", lambda degree: degree >= 3, "#ff7f0e", 18),
)


def graph_arrays(source: SourceGraph):
    node_ids = list(source.nodes)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    positions = np.asarray([source.nodes[node_id] for node_id in node_ids], dtype=float)
    edges = np.asarray(
        [(node_index[left], node_index[right]) for left, right in source.edges], dtype=int
    ).reshape(-1, 2)
    degree = np.bincount(edges.ravel(), minlength=len(positions)) if len(edges) else np.zeros(len(positions), dtype=int)
    return positions, edges, degree


def line_coordinates(lines):
    x, y, z = [], [], []
    for line in lines:
        x.extend([*line[:, 0], None])
        y.extend([*line[:, 1], None])
        z.extend([*line[:, 2], None])
    return x, y, z


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject")
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--geometry-image", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-step", type=int, default=2)
    args = parser.parse_args()

    source = SourceGraph.from_directory(args.graph_dir)
    positions, edges, degree = graph_arrays(source)
    straight = [positions[edge] for edge in edges]

    segmentation_image = nib.load(str(args.segmentation))
    geometry_image = nib.load(str(args.geometry_image or args.segmentation))
    if segmentation_image.shape != geometry_image.shape:
        raise ValueError("segmentation and geometry image shapes differ")
    mask = np.asanyarray(segmentation_image.dataobj) > 0
    from skimage.measure import marching_cubes
    vertices, faces, _, _ = marching_cubes(
        mask.astype(np.float32), level=0.5, step_size=args.mesh_step
    )
    vertices = voxel_to_world(vertices, geometry_image.affine)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"{args.subject}_adaptive_all_nodes"

    import plotly.graph_objects as go
    figure = go.Figure(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color="#aebdcc",
        opacity=0.10, name="segmentation", hoverinfo="skip",
    ))
    x, y, z = line_coordinates(straight)
    figure.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="lines", name="graph edges",
        line={"color": "#555555", "width": 2}, hoverinfo="skip",
    ))
    counts = {}
    for label, selector, colour, size in NODE_STYLES:
        selected = selector(degree)
        counts[label] = int(selected.sum())
        if not selected.any():
            continue
        figure.add_trace(go.Scatter3d(
            x=positions[selected, 0], y=positions[selected, 1], z=positions[selected, 2],
            mode="markers", name=f"{label}: {selected.sum()}",
            marker={"color": colour, "size": max(2, size / 3)},
            customdata=degree[selected],
            hovertemplate="degree %{customdata}<br>(%{x:.2f}, %{y:.2f}, %{z:.2f}) mm<extra></extra>",
        ))
    figure.update_layout(
        title=f"{args.subject}: adaptive graph — all node types",
        scene={"aspectmode": "data", "xaxis_title": "x (mm)", "yaxis_title": "y (mm)", "zaxis_title": "z (mm)"},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )
    figure.write_html(stem.with_suffix(".html"), include_plotlyjs=True)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    static = plt.figure(figsize=(10, 8), constrained_layout=True)
    axis = static.add_subplot(111, projection="3d")
    stride = max(1, len(faces) // 12_000)
    axis.add_collection3d(Poly3DCollection(
        vertices[faces[::stride]], facecolor="#aebdcc", edgecolor="none", alpha=0.035
    ))
    for line in straight:
        axis.plot(*line.T, color="#555555", linewidth=0.55, alpha=0.75)
    for label, selector, node_colour, size in NODE_STYLES:
        selected = selector(degree)
        if selected.any():
            axis.scatter(*positions[selected].T, color=node_colour, s=size,
                         depthshade=False, label=f"{label}: {selected.sum()}")
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    axis.set_box_aspect(np.maximum(extent, 1e-6))
    axis.set_title("Adaptive graph used by the model")
    axis.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
    axis.view_init(elev=24, azim=-58)
    axis.legend(loc="upper left", fontsize=8)
    static.suptitle(
        f"{args.subject}: adaptive graph ({len(positions)} nodes, {len(edges)} edges)", fontsize=16
    )
    static.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(static)
    print({"subject": args.subject, "nodes": len(positions), "edges": len(edges), **counts})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
