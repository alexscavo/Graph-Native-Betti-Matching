#!/usr/bin/env python3
"""Self-contained interactive 3D report for an extracted IXI vessel graph.

Renders the vessel surface mesh together with the extracted graph so the two can
be inspected against each other: centerlines, junctions, terminations and the
optional curvature-driven degree-2 nodes are separate toggleable layers.

All three Phase-1 graph representations are embedded: junction-only, adaptive,
and the immutable dense reference. They can be compared in the browser without
re-running extraction.

A whole 512x512x100 volume produces a mesh far too large to embed, so a dense
sub-volume is selected by default; pass --full to override at your own risk.
"""

from __future__ import annotations

import argparse
import html as html_lib
from pathlib import Path
import sys

import numpy as np
import nibabel as nib

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ixi_vessel_graph import (  # noqa: E402
    build_representation_family,
    chord_mask_fraction,
)
from scripts.prepare_ixi_sources import segmentation_path  # noqa: E402


DEFAULT_ROOT = Path("/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/IXI_dataset")


def load_subject(root: Path, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return segmentation, affine and voxel spacing, repairing a stripped header."""

    segmentation = nib.load(str(segmentation_path(root, subject)))
    affine = segmentation.affine
    mask_path = root / "Brain_masks" / f"{subject}-MRA_mask.nii.gz"
    zooms = tuple(round(float(z), 4) for z in segmentation.header.get_zooms())
    if zooms == (1.0, 1.0, 1.0) and mask_path.is_file():
        brain_mask = nib.load(str(mask_path))
        if brain_mask.shape == segmentation.shape:
            affine = brain_mask.affine
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return np.asanyarray(segmentation.dataobj) > 0, affine, spacing


def densest_window(
    volume: np.ndarray, half: int, half_z: int, step: int = 12
) -> tuple[slice, slice, slice]:
    """Locate the sub-volume holding the most vessel; the median position is empty."""

    best, chosen = -1, None
    for z in range(half_z, max(half_z + 1, volume.shape[2] - half_z), step):
        for y in range(half, max(half + 1, volume.shape[0] - half), step * 2):
            for x in range(half, max(half + 1, volume.shape[1] - half), step * 2):
                candidate = (
                    slice(y - half, y + half),
                    slice(x - half, x + half),
                    slice(z - half_z, z + half_z),
                )
                count = int(volume[candidate].sum())
                if count > best:
                    best, chosen = count, candidate
    return chosen


def loop_edges(graph) -> tuple[set[int], int]:
    """Return the indices of every edge lying on a cycle, and the cycle count.

    A spanning tree is built first; each remaining edge closes one independent
    cycle, whose member edges are that edge plus the tree path between its
    endpoints. Edges on no cycle form the acyclic skeleton of the graph.
    """

    import collections

    parent = list(range(graph.node_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    tree, closing = [], []
    for index, (left, right) in enumerate(graph.edges):
        root_l, root_r = find(left), find(right)
        if root_l != root_r:
            parent[root_r] = root_l
            tree.append(index)
        else:
            closing.append(index)

    adjacency = collections.defaultdict(list)
    for index in tree:
        left, right = graph.edges[index]
        adjacency[left].append((right, index))
        adjacency[right].append((left, index))

    members: set[int] = set()
    for index in closing:
        source, target = graph.edges[index]
        previous = {source: None}
        queue = collections.deque([source])
        while queue:
            node = queue.popleft()
            if node == target:
                break
            for neighbour, edge in adjacency[node]:
                if neighbour not in previous:
                    previous[neighbour] = (node, edge)
                    queue.append(neighbour)
        members.add(index)
        if target in previous:
            cursor = target
            while previous[cursor] is not None:
                node, edge = previous[cursor]
                members.add(edge)
                cursor = node
    return members, len(closing)


def graph_traces(graph, spacing, label, colours, visible):
    import plotly.graph_objects as go

    traces = []
    # The ground truth is nodes plus STRAIGHT node-to-node edges: write_vtp_graph
    # emits one two-point line per edge. The dense centerline polyline only serves
    # to place degree-2 nodes and to find patch-face crossings, so drawing it here
    # would show voxel staircase that is not part of the target.
    positions = graph.node_positions * spacing
    on_loop, cycle_count = loop_edges(graph)

    # ALL edges in one trace, so hiding the highlight never removes an edge.
    x, y, z = [], [], []
    for left, right in graph.edges:
        x.extend([positions[left, 0], positions[right, 0], None])
        y.extend([positions[left, 1], positions[right, 1], None])
        z.extend([positions[left, 2], positions[right, 2], None])
    traces.append(
        go.Scatter3d(
            x=x, y=y, z=z, mode="lines", name=f"{label}: edges ({graph.edge_count})",
            line=dict(color=colours["line"], width=4), hoverinfo="skip",
            visible=visible, legendgroup=label,
        )
    )

    # Optional highlight drawn ON TOP of the full edge set, not instead of part of it.
    if on_loop:
        x, y, z = [], [], []
        for index in sorted(on_loop):
            left, right = graph.edges[index]
            x.extend([positions[left, 0], positions[right, 0], None])
            y.extend([positions[left, 1], positions[right, 1], None])
            z.extend([positions[left, 2], positions[right, 2], None])
        traces.append(
            go.Scatter3d(
                x=x, y=y, z=z, mode="lines",
                name=f"{label}: highlight loops ({cycle_count} cycles, {len(on_loop)} edges)",
                line=dict(color="#e5007d", width=9), hoverinfo="skip",
                visible=visible, legendgroup=label,
            )
        )

    # Dense skeleton centerline. NOT part of the ground truth -- the GT is the
    # straight node-to-node edges above -- but shown as an optional reference so the
    # chord can be compared against the path it approximates. Hidden by default.
    cx, cy, cz = [], [], []
    for polyline in graph.centerlines:
        pts = np.asarray(polyline) * spacing
        cx.extend(pts[:, 0].tolist() + [None])
        cy.extend(pts[:, 1].tolist() + [None])
        cz.extend(pts[:, 2].tolist() + [None])
    traces.append(
        go.Scatter3d(
            x=cx, y=cy, z=cz, mode="lines",
            name=f"{label}: centerline (reference, not GT)",
            line=dict(color="#00b3a4", width=3, dash="dot"), hoverinfo="skip",
            visible="legendonly", legendgroup=label,
        )
    )

    degrees = graph.node_degrees
    for selector, colour, kind, size in (
        (degrees >= 3, colours["junction"], "junction", 5.5),
        (degrees == 1, colours["termination"], "termination", 4.5),
        (degrees == 2, colours["degree2"], "degree-2 (curvature)", 3.0),
    ):
        if not selector.any():
            continue
        index = np.flatnonzero(selector)
        traces.append(
            go.Scatter3d(
                x=positions[selector, 0], y=positions[selector, 1], z=positions[selector, 2],
                mode="markers", name=f"{label}: {kind} ({int(selector.sum())})",
                marker=dict(size=size, color=colour, opacity=0.95),
                customdata=np.stack(
                    [index, degrees[selector], graph.node_radii[selector]], axis=1
                ),
                hovertemplate=(
                    "node %{customdata[0]}<br>degree %{customdata[1]}"
                    "<br>radius %{customdata[2]:.2f} mm"
                    "<br>(%{x:.1f}, %{y:.1f}, %{z:.1f}) mm<extra></extra>"
                ),
                visible=visible, legendgroup=label,
            )
        )
    return traces


def build_report(args: argparse.Namespace) -> Path:
    import plotly.graph_objects as go
    from skimage.measure import marching_cubes

    volume, _, spacing = load_subject(args.root, args.subject)
    if args.full:
        window = (slice(None), slice(None), slice(None))
    else:
        window = densest_window(volume, args.half, args.half_z)
    cropped = volume[window]
    print(f"region {cropped.shape}, {int(cropped.sum())} vessel voxels")

    vertices, faces, _, _ = marching_cubes(
        cropped.astype(np.float32), level=0.5, spacing=tuple(spacing), step_size=args.mesh_step
    )
    print(f"mesh: {len(vertices)} vertices, {len(faces)} faces")

    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="#9fb3c8", opacity=args.mesh_opacity, name="vessel surface",
            showlegend=True, hoverinfo="skip", flatshading=False,
        )
    )

    tolerance = args.rdp_voxels * float(spacing.min())
    graphs = build_representation_family(
        cropped,
        spacing=spacing,
        adaptive_tolerance_mm=tolerance,
        adaptive_radius_fraction=args.radius_fraction,
        spur_length=args.spur_length,
    )
    styles = {
        "adaptive": ({"line": "#c0392b", "junction": "#ff7f0e", "termination": "#1f77b4", "degree2": "#2ca02c"}, True),
        "junction_only": ({"line": "#8e44ad", "junction": "#e67e22", "termination": "#2980b9", "degree2": "#27ae60"}, "legendonly"),
        "dense": ({"line": "#008c95", "junction": "#f39c12", "termination": "#2471a3", "degree2": "#00a6a6"}, "legendonly"),
    }
    for name in ("adaptive", "junction_only", "dense"):
        colours, visible = styles[name]
        for trace in graph_traces(
            graphs[name], spacing, name.replace("_", " "), colours, visible
        ):
            figure.add_trace(trace)

    extent = np.asarray(cropped.shape) * spacing
    figure.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)",
        ),
        legend=dict(itemsizing="constant", groupclick="toggleitem"),
        margin=dict(l=0, r=0, t=30, b=0), height=820,
        title=f"{args.subject} — vessel mesh and extracted graph",
    )

    graph_summary = {}
    for name, graph in graphs.items():
        beta_0, beta_1 = graph.betti()
        containment = np.asarray(
            [chord_mask_fraction(line[0], line[-1], cropped) for line in graph.centerlines],
            dtype=np.float64,
        )
        graph_summary[name] = {
            "nodes": graph.node_count,
            "edges": graph.edge_count,
            "degree_2": int((graph.node_degrees == 2).sum()),
            "beta_0": beta_0,
            "beta_1": beta_1,
            "loop_edges": len(loop_edges(graph)[0]),
            "minimum_edge_chord_containment": float(containment.min()) if len(containment) else 1.0,
            "mean_edge_chord_containment": float(containment.mean()) if len(containment) else 1.0,
            "off_lumen_edge_chords": int(np.count_nonzero(containment < 1.0)),
        }
    rows = [
        ("subject", args.subject),
        ("region (voxels)", " x ".join(str(s) for s in cropped.shape)),
        ("region (mm)", " x ".join(f"{e:.1f}" for e in extent)),
        ("voxel spacing (mm)", " x ".join(f"{s:.3f}" for s in spacing)),
        ("vessel voxels", f"{int(cropped.sum()):,}"),
        ("mesh faces", f"{len(faces):,}"),
        ("RDP tolerance", f"{args.rdp_voxels} vox = {tolerance:.3f} mm"),
        ("radius-adaptive", f"{args.radius_fraction}x local radius" if args.radius_fraction>0 else "off"),
        ("spur pruning", f"{args.spur_length} voxels"),
    ]
    for name in ("junction_only", "adaptive", "dense"):
        item = graph_summary[name]
        label = name.replace("_", " ")
        rows.extend(
            [
                (f"{label}: nodes / edges", f"{item['nodes']} / {item['edges']}"),
                (f"{label}: degree-2", item["degree_2"]),
                (f"{label}: beta-0 / beta-1", f"{item['beta_0']} / {item['beta_1']}"),
            ]
        )
    table = "".join(
        f"<tr><th>{html_lib.escape(k)}</th><td>{html_lib.escape(str(v))}</td></tr>"
        for k, v in rows
    )
    plot = figure.to_html(
        full_html=False, include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>IXI vessel graph — {html_lib.escape(args.subject)}</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:1.2rem;color:#1c1c1c;background:#fff}}
 table{{border-collapse:collapse;margin:0 0 1rem}}
 th,td{{text-align:left;border:1px solid #d5d5d5;padding:.3rem .6rem;font-size:.86rem}}
 th{{background:#f4f6f8;font-weight:600}}
 p{{max-width:62rem;font-size:.9rem;color:#444}}
</style></head><body>
<h2>IXI vessel graph — {html_lib.escape(args.subject)}</h2>
<table>{table}</table>
<p>Drag to rotate, scroll to zoom, right-drag to pan. Click legend entries to toggle
layers. The adaptive representation starts visible. Enable the junction-only and
dense groups to compare them against the same vessel surface. The dense graph is
the raw post-topology-cleanup reference before smoothing and RDP; it is an
evaluation control, not the selected training target. The magenta
<em>highlight loops</em> layer overlays edges lying on cycles. Hover any node for
its id, degree and local vessel radius.</p>
{plot}
</body></html>"""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"graph3d_{args.subject}.html"
    destination.write_text(document)
    import json
    summary = {
        "subject": args.subject,
        "source": "real IXI segmentation",
        "window": [
            [item.start, item.stop] if item.start is not None else [0, int(size)]
            for item, size in zip(window, volume.shape)
        ],
        "shape_voxels": list(cropped.shape),
        "spacing_mm": [float(value) for value in spacing],
        "vessel_voxels": int(cropped.sum()),
        "rdp_tolerance_mm": tolerance,
        "radius_fraction": args.radius_fraction,
        "spur_length_voxels": args.spur_length,
        "representations": graph_summary,
    }
    destination.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    # A compact static decision aid accompanies the interactive 3-D report.
    import matplotlib.pyplot as plt
    names = ["junction_only", "adaptive", "dense"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for axis, key, title in (
        (axes[0], "nodes", "Nodes"),
        (axes[1], "edges", "Edges"),
        (axes[2], "minimum_edge_chord_containment", "Worst chord containment"),
    ):
        values = [graph_summary[name][key] for name in names]
        bars = axis.bar([name.replace("_", "\n") for name in names], values,
                        color=("#8e44ad", "#c0392b", "#008c95"))
        axis.set_title(title)
        axis.bar_label(bars, fmt="%.3g", padding=2)
        if key.endswith("containment"):
            axis.set_ylim(0, 1.08)
    fig.suptitle(f"{args.subject}: three representations from one dense centerline")
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    plt.close(fig)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/ixi_qc"),
    )
    parser.add_argument("--half", type=int, default=45, help="in-plane half-width of the region")
    parser.add_argument("--half-z", type=int, default=18, help="slice half-depth of the region")
    parser.add_argument("--rdp-voxels", type=float, default=2.0)
    parser.add_argument("--radius-fraction", type=float, default=0.0,
                        help="bound chord deviation by this fraction of local vessel radius")
    parser.add_argument("--spur-length", type=int, default=4)
    parser.add_argument("--mesh-step", type=int, default=1, help="marching-cubes decimation")
    parser.add_argument("--mesh-opacity", type=float, default=0.35)
    parser.add_argument("--full", action="store_true", help="whole volume; produces a very large file")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    destination = build_report(args)
    print(f"Wrote interactive report: {destination} ({destination.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
