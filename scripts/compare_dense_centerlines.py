#!/usr/bin/env python3
"""Compare dense-centerline variants against a binary vessel segmentation.

The input centerlines are Voreen-style ``graph.vvg`` files whose coordinates
are in the segmentation's world coordinate system.  The report is deliberately
extractor-independent: it can compare the legacy/current output with any
refactored Vedo-style backend without importing either implementation.

Outputs are ``summary.json``, ``metrics.csv``, ``comparison.png`` and, unless
disabled, an interactive ``comparison.html`` with toggleable 3-D centerlines.
Containment is measured on densely resampled line segments, not only at the
stored samples, so a shortcut through background cannot pass unnoticed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


def load_vvg(path: Path) -> dict:
    payload = json.loads(path.read_text())
    graph = payload.get("graph", payload)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    polylines = []
    for edge in edges:
        samples = edge.get("skeletonVoxels", edge.get("points", []))
        points = [sample.get("pos", sample) for sample in samples]
        if len(points) >= 2:
            polylines.append(np.asarray(points, dtype=np.float64))
    return {"nodes": nodes, "edges": edges, "polylines": polylines}


def resample_polyline(points: np.ndarray, step_mm: float) -> np.ndarray:
    pieces = []
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        length = float(np.linalg.norm(end - start))
        count = max(2, int(math.ceil(length / step_mm)) + 1)
        sample = np.linspace(start, end, count)
        pieces.append(sample if index == 0 else sample[1:])
    return np.concatenate(pieces, axis=0) if pieces else points.copy()


def world_to_voxel(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[points, np.ones(len(points))]
    return (homogeneous @ np.linalg.inv(affine).T)[:, :3]


def containment(points_world: np.ndarray, mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    coordinates = world_to_voxel(points_world, affine)
    inside = np.zeros(len(coordinates), dtype=bool)
    epsilon = 1e-9
    for dz in (-epsilon, epsilon):
        for dy in (-epsilon, epsilon):
            for dx in (-epsilon, epsilon):
                indices = np.floor(coordinates + 0.5 + (dz, dy, dx)).astype(np.int64)
                valid = np.logical_and(indices >= 0, indices < np.asarray(mask.shape)).all(axis=1)
                if valid.any():
                    selected = indices[valid]
                    inside[valid] |= mask[selected[:, 0], selected[:, 1], selected[:, 2]]
    return inside


def turn_angles(polylines: Iterable[np.ndarray]) -> np.ndarray:
    values = []
    for points in polylines:
        if len(points) < 3:
            continue
        before, after = np.diff(points, axis=0)[:-1], np.diff(points, axis=0)[1:]
        denominator = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
        valid = denominator > 1e-12
        cosine = np.divide(
            np.einsum("ij,ij->i", before, after), denominator,
            out=np.ones_like(denominator), where=valid,
        )
        values.extend(np.degrees(np.arccos(np.clip(cosine[valid], -1.0, 1.0))))
    return np.asarray(values, dtype=np.float64)


def betti(nodes: list[dict], edges: list[dict]) -> tuple[int, int] | tuple[None, None]:
    if not nodes:
        return (0, 0) if not edges else (None, None)
    identifiers = [int(node.get("id", index)) for index, node in enumerate(nodes)]
    parent = {identifier: identifier for identifier in identifiers}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    valid_edges = 0
    for edge in edges:
        left = int(edge.get("node1", edge.get("node1id", -1)))
        right = int(edge.get("node2", edge.get("node2id", -1)))
        if left not in parent or right not in parent:
            return None, None
        valid_edges += 1
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left
    beta0 = len({find(identifier) for identifier in identifiers})
    return beta0, valid_edges - len(nodes) + beta0


def measure(name: str, path: Path, mask: np.ndarray, affine: np.ndarray, step_mm: float) -> tuple[dict, dict]:
    graph = load_vvg(path)
    resampled = [resample_polyline(line, step_mm) for line in graph["polylines"]]
    inside = np.concatenate([containment(line, mask, affine) for line in resampled]) if resampled else np.empty(0, bool)
    all_resampled = np.concatenate(resampled) if resampled else np.empty((0, 3))
    graph["outside_world"] = all_resampled[~inside]
    angles = turn_angles(graph["polylines"])
    lengths = [float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()) for line in graph["polylines"]]
    beta0, beta1 = betti(graph["nodes"], graph["edges"])
    metrics = {
        "variant": name,
        "source": str(path.resolve()),
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "polyline_count": len(graph["polylines"]),
        "stored_sample_count": int(sum(len(line) for line in graph["polylines"])),
        "resampled_point_count": int(len(inside)),
        "length_mm": float(sum(lengths)),
        "containment_fraction": float(inside.mean()) if len(inside) else None,
        "outside_point_count": int((~inside).sum()),
        "turn_angle_median_deg": float(np.median(angles)) if len(angles) else None,
        "turn_angle_p95_deg": float(np.percentile(angles, 95)) if len(angles) else None,
        "betti_0": beta0,
        "betti_1": beta1,
    }
    return metrics, graph


def write_outputs(results: list[dict], graphs: dict[str, dict], output: Path, segmentation: Path, step_mm: float, html: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = list(results[0])
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(results)
    (output / "summary.json").write_text(json.dumps({
        "segmentation": str(segmentation.resolve()), "resample_step_mm": step_mm,
        "variants": results,
    }, indent=2) + "\n")

    import matplotlib.pyplot as plt
    names = [row["variant"] for row in results]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    panels = (
        ("containment_fraction", "Continuous-path containment", "fraction"),
        ("turn_angle_median_deg", "Median stored-sample turn", "degrees"),
        ("stored_sample_count", "Dense centerline samples", "count"),
        ("edge_count", "Topological edges", "count"),
    )
    colours = plt.cm.Set2(np.linspace(0, 1, len(results)))
    for axis, (key, title, ylabel) in zip(axes.flat, panels):
        values = [np.nan if row[key] is None else row[key] for row in results]
        bars = axis.bar(names, values, color=colours)
        axis.set(title=title, ylabel=ylabel)
        axis.tick_params(axis="x", rotation=15)
        axis.bar_label(bars, fmt="%.3g", padding=2)
        if key == "containment_fraction": axis.set_ylim(0, 1.05)
    figure.suptitle("Dense-centerline comparison against segmentation")
    figure.savefig(output / "comparison.png", dpi=170)
    plt.close(figure)

    if html:
        try:
            import plotly.graph_objects as go
        except ImportError:
            return
        figure3d = go.Figure()
        palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
        for variant_index, (name, graph) in enumerate(graphs.items()):
            x, y, z = [], [], []
            for line in graph["polylines"]:
                x.extend(line[:, 0].tolist() + [None]); y.extend(line[:, 1].tolist() + [None]); z.extend(line[:, 2].tolist() + [None])
            figure3d.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="lines", name=name,
                line={"width": 3, "color": palette[variant_index % len(palette)]},
                visible=True if variant_index == 0 else "legendonly",
            ))
            outside = graph["outside_world"]
            if len(outside):
                # Subsample display only; the numeric metric always uses every point.
                stride = max(1, int(math.ceil(len(outside) / 10_000)))
                outside = outside[::stride]
                figure3d.add_trace(go.Scatter3d(
                    x=outside[:, 0], y=outside[:, 1], z=outside[:, 2], mode="markers",
                    name=f"{name}: outside lumen ({len(graph['outside_world'])})",
                    marker={"size": 2.5, "color": "#D00000", "opacity": 0.8},
                    visible=True if variant_index == 0 else "legendonly",
                ))
        figure3d.update_layout(
            title="Dense centerlines (world coordinates; click legend to toggle)",
            scene={"aspectmode": "data", "xaxis_title": "x (mm)", "yaxis_title": "y (mm)", "zaxis_title": "z (mm)"},
            margin={"l": 0, "r": 0, "t": 45, "b": 0},
        )
        figure3d.write_html(output / "comparison.html", include_plotlyjs=True)


def parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("variant must have the form NAME=/path/to/graph.vvg")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("variant name and path must be non-empty")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--geometry-image", type=Path, default=None, help="Image supplying the trusted affine when the segmentation header is stripped")
    parser.add_argument("--variant", action="append", type=parse_variant, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-mm", type=float, default=None, help="Containment sampling step (default: half minimum voxel spacing)")
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()
    if len(args.variant) < 2:
        parser.error("at least two --variant values are required")
    image = nib.load(str(args.segmentation))
    mask = np.asanyarray(image.dataobj) > 0
    geometry = nib.load(str(args.geometry_image)) if args.geometry_image else image
    if geometry.shape != image.shape:
        parser.error("--geometry-image and --segmentation must have the same shape")
    affine = geometry.affine
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    step_mm = args.step_mm or float(spacing.min() / 2.0)
    if step_mm <= 0:
        parser.error("--step-mm must be positive")
    results, graphs = [], {}
    for name, path in args.variant:
        result, graph = measure(name, path, mask, affine, step_mm)
        results.append(result); graphs[name] = graph
    write_outputs(results, graphs, args.output, args.segmentation, step_mm, not args.no_html)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
