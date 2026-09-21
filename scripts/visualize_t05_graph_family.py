#!/usr/bin/env python3
"""Graph-only interactive and static T05 real-volume representation comparison."""

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_synthetic_mri_grid import SourceGraph
from scripts.study_full_volume_representations import REPRESENTATIONS, graph_topology


def lines(graph: SourceGraph) -> tuple[list, list, list]:
    x, y, z = [], [], []
    for left, right in graph.edges:
        a, b = graph.nodes[left], graph.nodes[right]
        x.extend((a[0], b[0], None))
        y.extend((a[1], b[1], None))
        z.extend((a[2], b[2], None))
    return x, y, z


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="path stem for PNG + HTML")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    graphs = {name: SourceGraph.from_directory(args.graph_dir / name) for name in REPRESENTATIONS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = go.Figure()
    static = plt.figure(figsize=(17, 6), constrained_layout=True)
    for index, (name, graph) in enumerate(graphs.items()):
        x, y, z = lines(graph)
        figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", visible=index == 1,
            name=f"{name}: edges", line={"color": "#485766", "width": 2}, hoverinfo="skip"))
        degree = {node: 0 for node in graph.nodes}
        for left, right in graph.edges:
            degree[left] += 1
            degree[right] += 1
        for caption, condition, color, size in (
            ("endpoints", lambda d: d == 1, "#2171b5", 4),
            ("degree-2", lambda d: d == 2, "#30a25d", 2),
            ("junctions", lambda d: d >= 3, "#e47737", 5),
        ):
            points = np.asarray([graph.nodes[node] for node in graph.nodes if condition(degree[node])], dtype=float).reshape(-1, 3)
            figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode="markers", visible=index == 1, name=f"{name}: {caption} ({len(points)})",
                marker={"size": size, "color": color, "opacity": .85}))
        ax = static.add_subplot(1, 3, index + 1, projection="3d")
        for left, right in graph.edges:
            a, b = graph.nodes[left], graph.nodes[right]
            ax.plot((a[0], b[0]), (a[1], b[1]), (a[2], b[2]), color="#485766", linewidth=.45)
        for caption, condition, color, size in (
            ("endpoint", lambda d: d == 1, "#2171b5", 3),
            ("degree-2", lambda d: d == 2, "#30a25d", 1),
            ("junction", lambda d: d >= 3, "#e47737", 5),
        ):
            if name == "dense" and caption == "degree-2":
                continue  # reference points would hide the graph itself
            points = np.asarray([graph.nodes[node] for node in graph.nodes if condition(degree[node])], dtype=float).reshape(-1, 3)
            if len(points):
                ax.scatter(*points.T, color=color, s=size, label=caption, depthshade=False)
        positions = np.stack(list(graph.nodes.values()))
        ax.set_box_aspect(np.maximum(np.ptp(positions, axis=0), .01))
        ax.view_init(elev=23, azim=-55)
        ax.set(title=f"{name.replace('_', ' ')}\n{len(graph.nodes):,} nodes; {len(graph.edges):,} edges; β₁={graph_topology(graph)['cycle_rank']}",
               xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
        if index == 1:
            ax.legend(loc="upper left", fontsize=7)
    buttons = []
    for index, name in enumerate(REPRESENTATIONS):
        buttons.append({"label": name, "method": "update", "args": [{"visible":
            [j // 4 == index for j in range(4 * len(REPRESENTATIONS))]},
            {"title": f"{args.graph_dir.parent.name}: {name} — straight graph edges and node types"}]})
    figure.update_layout(
        title=f"{args.graph_dir.parent.name}: adaptive — straight graph edges and node types",
        updatemenus=[{"buttons": buttons, "direction": "down", "x": .02, "y": .99}],
        scene={"aspectmode": "data", "xaxis_title": "x (mm)", "yaxis_title": "y (mm)", "zaxis_title": "z (mm)"},
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
    )
    figure.write_html(args.output.with_suffix(".html"), include_plotlyjs=True)
    static.suptitle(f"{args.graph_dir.parent.name}: model-visible graph geometry vs dense reference")
    static.savefig(args.output.with_suffix(".png"), dpi=150)
    plt.close(static)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
