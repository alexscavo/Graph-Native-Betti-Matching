#!/usr/bin/env python3
"""Create interactive HTML and PNG views of a saved vessel graph and mask."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

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


def patch_graph(path: Path, shape: tuple[int, ...]):
    """Read our ASCII VTP patch target without importing the training stack."""
    piece = ET.parse(path).getroot().find(".//Piece")
    if piece is None:
        raise ValueError(f"VTP has no PolyData piece: {path}")
    points = piece.find("./Points/DataArray")
    connectivity = piece.find("./Lines/DataArray[@Name='connectivity']")
    offsets = piece.find("./Lines/DataArray[@Name='offsets']")
    boundary = piece.find("./PointData/DataArray[@Name='is_patch_boundary_intersection']")
    if points is None or connectivity is None or offsets is None:
        raise ValueError(f"VTP is missing points or lines: {path}")
    normalized = np.fromstring(points.text or "", sep=" ").reshape(-1, 3)
    edges = np.fromstring(connectivity.text or "", sep=" ", dtype=int).reshape(-1, 2)
    ends = np.fromstring(offsets.text or "", sep=" ", dtype=int)
    if len(ends) != len(edges) or not np.array_equal(ends, 2 * np.arange(1, len(edges) + 1)):
        raise ValueError(f"VTP contains non-pair line cells: {path}")
    flags = (np.fromstring(boundary.text or "", sep=" ", dtype=int).astype(bool)
             if boundary is not None else np.zeros(len(normalized), dtype=bool))
    if len(flags) != len(normalized):
        raise ValueError(f"VTP boundary flags do not match points: {path}")
    degree = np.bincount(edges.ravel(), minlength=len(normalized))
    return normalized * np.asarray(shape), edges, degree, flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject")
    graph_input = parser.add_mutually_exclusive_group(required=True)
    graph_input.add_argument("--graph-dir", type=Path, help="saved full-volume graph directory")
    graph_input.add_argument("--patch-vtp", type=Path, help="normalized patch graph target")
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--geometry-image", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-step", type=int, default=2)
    args = parser.parse_args()

    segmentation_image = nib.load(str(args.segmentation))
    mask = np.asanyarray(segmentation_image.dataobj) > 0
    if not mask.any():
        raise ValueError(f"segmentation contains no foreground: {args.segmentation}")
    if args.patch_vtp:
        positions, edges, degree, boundary = patch_graph(args.patch_vtp, mask.shape)
        label = "patch voxel"
        kind = "patch"
    else:
        source = SourceGraph.from_directory(args.graph_dir)
        positions, edges, degree = graph_arrays(source)
        boundary = np.zeros(len(positions), dtype=bool)
        geometry_image = nib.load(str(args.geometry_image or args.segmentation))
        if segmentation_image.shape != geometry_image.shape:
            raise ValueError("segmentation and geometry image shapes differ")
        label = "mm"
        kind = "adaptive"
    straight = [positions[edge] for edge in edges]
    from skimage.measure import marching_cubes
    vertices, faces, _, _ = marching_cubes(
        mask.astype(np.float32), level=0.5, step_size=args.mesh_step
    )
    if not args.patch_vtp:
        vertices = voxel_to_world(vertices, geometry_image.affine)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"{args.subject}_{kind}_all_nodes"

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
    styles = (("patch boundary", boundary, "#d627b0", 18),) if args.patch_vtp else ()
    styles += tuple((name, selector(degree) & ~boundary, colour, size)
                    for name, selector, colour, size in NODE_STYLES)
    for name, selected, colour, size in styles:
        counts[name] = int(selected.sum())
        if not selected.any():
            continue
        figure.add_trace(go.Scatter3d(
            x=positions[selected, 0], y=positions[selected, 1], z=positions[selected, 2],
            mode="markers", name=f"{name}: {selected.sum()}",
            marker={"color": colour, "size": max(2, size / 3)},
            customdata=degree[selected],
            hovertemplate=f"degree %{{customdata}}<br>(%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}}) {label}<extra></extra>",
        ))
    figure.update_layout(
        title=f"{args.subject}: {kind} graph — all node types",
        scene={"aspectmode": "data", "xaxis_title": f"x ({label})", "yaxis_title": f"y ({label})", "zaxis_title": f"z ({label})"},
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
    for name, selected, node_colour, size in styles:
        if selected.any():
            axis.scatter(*positions[selected].T, color=node_colour, s=size,
                         depthshade=False, label=f"{name}: {selected.sum()}")
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    axis.set_box_aspect(np.maximum(extent, 1e-6))
    axis.set_title(f"{kind.title()} graph used by the model")
    axis.set(xlabel=f"x ({label})", ylabel=f"y ({label})", zlabel=f"z ({label})")
    axis.view_init(elev=24, azim=-58)
    axis.legend(loc="upper left", fontsize=8)
    static.suptitle(
        f"{args.subject}: {kind} graph ({len(positions)} nodes, {len(edges)} edges)", fontsize=16
    )
    static.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(static)
    print({"subject": args.subject, "nodes": len(positions), "edges": len(edges), **counts})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
