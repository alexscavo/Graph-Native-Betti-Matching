#!/usr/bin/env python3
"""T05: paired, physical-space representation study on saved full-volume graphs.

Every straight edge of the exported graph is evaluated, not the hidden VVG
centerline stored along a simplified edge. The unsmoothed dense graph is the
reference; branch correspondence uses the preserved endpoint/junction IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.representation_geometry import (
    point_to_polyline_distance, polyline_length, sample_polyline_by_arclength,
)
from scripts.audit_synthetic_mri_grid import SourceGraph

IXI = Path("/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/IXI/vascular_graphs/optimal_radius_0p75x_c095")
TOPBRAIN = Path("/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/TopBrain_Data_Release_Batches1n2_081425/vascular_graphs/optimal_radius_0p75x_c095")
REPRESENTATIONS = ("junction_only", "adaptive", "dense")
TOLERANCES_MM = (0.5, 1.0, 2.0)


def read_manifest(root: Path) -> list[dict]:
    with (root / "extraction_manifest.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def graph_branches(graph: SourceGraph, anchors: set[int]) -> dict[tuple[int, int], list[np.ndarray]]:
    """Contract internal degree-2 vertices; retain each branch's ordered points."""

    adjacency: dict[int, list[tuple[int, int]]] = {node: [] for node in graph.nodes}
    for index, (left, right) in enumerate(graph.edges):
        if left not in adjacency or right not in adjacency:
            raise ValueError("invalid source edge")
        adjacency[left].append((right, index))
        adjacency[right].append((left, index))
    if not anchors <= graph.nodes.keys():
        raise ValueError("graph lost a topological anchor")
    visited: set[int] = set()
    branches: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for start in sorted(anchors):
        for neighbor, edge_index in adjacency[start]:
            if edge_index in visited:
                continue
            path = [start, neighbor]
            visited.add(edge_index)
            current = neighbor
            while current not in anchors:
                onward = [(node, index) for node, index in adjacency[current] if index not in visited]
                if len(adjacency[current]) != 2 or len(onward) != 1:
                    raise ValueError(f"unexpected internal node degree at {current}")
                following, index = onward[0]
                path.append(following)
                visited.add(index)
                current = following
            key = tuple(sorted((start, current)))
            if start != key[0]:
                path.reverse()
            branches[key].append(np.asarray([graph.nodes[node] for node in path], dtype=np.float64))
    if len(visited) != len(graph.edges):
        raise ValueError("unmatched source edges (possible anchorless cycle)")
    return branches


def graph_topology(graph: SourceGraph) -> dict[str, int]:
    parent = {node: node for node in graph.nodes}

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    degree = Counter()
    for left, right in graph.edges:
        degree[left] += 1
        degree[right] += 1
        a, b = root(left), root(right)
        parent[a] = b
    components = len({root(node) for node in graph.nodes})
    return {
        "nodes": len(graph.nodes), "edges": len(graph.edges),
        "degree_2": sum(degree[node] == 2 for node in graph.nodes),
        "endpoints": sum(degree[node] == 1 for node in graph.nodes),
        "junctions": sum(degree[node] >= 3 for node in graph.nodes),
        "high_degree": sum(degree[node] > 3 for node in graph.nodes),
        "components": components,
        "cycle_rank": len(graph.edges) - len(graph.nodes) + components,
    }


def pair_branches(
    reference: dict[tuple[int, int], list[np.ndarray]],
    candidate: dict[tuple[int, int], list[np.ndarray]],
) -> dict[tuple[int, int], list[np.ndarray]]:
    """Resolve parallel branches by minimum one-to-one spatial discrepancy."""

    if {key: len(value) for key, value in candidate.items()} != {
        key: len(value) for key, value in reference.items()
    }:
        raise ValueError("different topological branch multiplicities")
    paired = {}
    for key, refs in reference.items():
        curves = candidate[key]
        if len(refs) == 1:
            paired[key] = curves
            continue
        costs = np.asarray([[float(np.mean(point_to_polyline_distance(
            sample_polyline_by_arclength(path, 1.0), target)))
            for path in refs] for target in curves])
        source_indices, ref_indices = linear_sum_assignment(costs)
        paired[key] = [curves[int(source_indices[np.where(ref_indices == i)[0][0]])]
                       for i in range(len(refs))]
    return paired


def junction_angles(branches: dict[tuple[int, int], list[np.ndarray]]) -> dict[int, np.ndarray]:
    """Sorted unordered angles among outgoing branch tangents at each junction."""

    outgoing: dict[int, list[np.ndarray]] = defaultdict(list)
    for (left, right), paths in branches.items():
        for path in paths:
            if left == right:
                continue  # self-returning branch has no unique arm assignment
            outgoing[left].append(path[1] - path[0])
            outgoing[right].append(path[-2] - path[-1])
    result = {}
    for node, tangents in outgoing.items():
        valid = [v / np.linalg.norm(v) for v in tangents if np.linalg.norm(v) > 1e-9]
        if len(valid) >= 3:
            result[node] = np.sort(np.asarray([
                np.degrees(np.arccos(np.clip(float(a @ b), -1, 1)))
                for i, a in enumerate(valid) for b in valid[i + 1:]
            ]))
    return result


def compare_graphs(graphs: dict[str, SourceGraph], spacing_mm: float = 0.5) -> tuple[list[dict], list[dict]]:
    if spacing_mm <= 0:
        raise ValueError("sampling step must be positive")
    junction = graphs["junction_only"]
    anchors = set(junction.nodes)
    branches = {name: graph_branches(graphs[name], anchors) for name in REPRESENTATIONS}
    keys = set(branches["dense"])
    for name in REPRESENTATIONS:
        branches[name] = pair_branches(branches["dense"], branches[name])
    reference_angles = junction_angles(branches["dense"])
    rows, branch_rows = [], []
    dense_topology = graph_topology(graphs["dense"])
    for name in REPRESENTATIONS:
        topology = graph_topology(graphs[name])
        if (topology["components"], topology["cycle_rank"]) != (
            dense_topology["components"], dense_topology["cycle_rank"]
        ):
            raise ValueError("topology changed relative to the dense graph")
        source_to_ref, ref_to_source = [], []
        lengths, reference_lengths, tort_weighted_errors, angle_errors = [], [], [], []
        tort_reference_length = 0.0
        closed = 0
        candidate_angles = junction_angles(branches[name])
        for anchor in reference_angles:
            if anchor in candidate_angles and len(reference_angles[anchor]) == len(candidate_angles[anchor]):
                # Sorted angle lists are a descriptor, not a vessel-arm matching.
                angle_errors.extend(abs(reference_angles[anchor] - candidate_angles[anchor]))
        for key in sorted(keys):
          for branch_index, (reference, candidate) in enumerate(zip(branches["dense"][key], branches[name][key])):
            if not np.allclose(reference[[0, -1]], candidate[[0, -1]], atol=1e-5, rtol=0):
                raise ValueError(f"anchor positions changed on branch {key}")
            ref_length, length = polyline_length(reference), polyline_length(candidate)
            reference_lengths.append(ref_length)
            lengths.append(length)
            chord = float(np.linalg.norm(reference[-1] - reference[0]))
            if chord < 1e-6:
                closed += 1
                tort_error = float("nan")
            else:
                tort_error = abs(length - ref_length) / chord
                tort_weighted_errors.append(ref_length * tort_error)
                tort_reference_length += ref_length
            reference_samples = sample_polyline_by_arclength(reference, spacing_mm)
            candidate_samples = sample_polyline_by_arclength(candidate, spacing_mm)
            source_to_ref.extend(point_to_polyline_distance(candidate_samples, reference))
            ref_to_source.extend(point_to_polyline_distance(reference_samples, candidate))
            branch_rows.append({
                "representation": name, "anchor_a": key[0], "anchor_b": key[1], "parallel_branch_index": branch_index,
                "dense_length_mm": ref_length, "length_mm": length,
                "absolute_length_error_mm": abs(length - ref_length),
                "absolute_arc_chord_tortuosity_error": tort_error,
                "reference_samples": len(reference_samples), "representation_samples": len(candidate_samples),
            })
        forward, backward = np.asarray(source_to_ref), np.asarray(ref_to_source)
        total_reference_length = sum(reference_lengths)
        result = {
            "representation": name, **topology, "branches": sum(map(len, branches[name].values())),
            "physical_length_mm": sum(lengths), "dense_length_mm": total_reference_length,
            "relative_length_error": (sum(lengths) - total_reference_length) / total_reference_length,
            "nodes_per_dense_mm": topology["nodes"] / total_reference_length,
            "junctions_per_dense_mm": topology["junctions"] / total_reference_length,
            "length_weighted_absolute_branch_length_error_fraction": sum(
                abs(a - b) for a, b in zip(lengths, reference_lengths)
            ) / total_reference_length,
            "length_weighted_tortuosity_error": (
                sum(tort_weighted_errors) / tort_reference_length
                if tort_reference_length else float("nan")
            ),
            "closed_or_degenerate_branches": closed,
            "junction_angle_mae_deg": float(np.mean(angle_errors)) if angle_errors else float("nan"),
            "curve_acd_mm": float((forward.mean() + backward.mean()) / 2),
            "curve_hd95_mm": float(max(np.percentile(forward, 95), np.percentile(backward, 95))),
        }
        for tolerance in TOLERANCES_MM:
            precision = float(np.mean(forward <= tolerance))
            recall = float(np.mean(backward <= tolerance))
            result[f"curve_precision_{tolerance:g}mm"] = precision
            result[f"curve_recall_{tolerance:g}mm"] = recall
            result[f"curve_f1_{tolerance:g}mm"] = (
                2 * precision * recall / (precision + recall) if precision + recall else 0.0
            )
        rows.append(result)
    return rows, branch_rows


def plot_summary(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ("junction_only", "adaptive")
    groups = sorted({row["stratum"] for row in rows})
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    panels = (
        ("nodes", "Mean full-volume nodes", 1),
        ("curve_f1_0.5mm", "Mean branch-paired Curve F1 (0.5 mm)", 1),
        ("curve_acd_mm", "Mean branch-paired ACD (mm)", 1),
        ("length_weighted_absolute_branch_length_error_fraction", "Mean absolute vessel-length error (%)", 100),
    )
    x = np.arange(len(groups))
    for ax, (field, title, scale) in zip(axes.flat, panels):
        for offset, name in enumerate(labels):
            values = [np.mean([r[field] for r in rows if r["stratum"] == group and r["representation"] == name]) * scale for group in groups]
            ax.bar(x + (offset - .5) * .37, values, width=.36,
                   label=name.replace("_", " "), color=("#a56355", "#398974")[offset])
        ax.set(xticks=x, xticklabels=groups, title=title)
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend()
    figure.suptitle("T05: real full-volume graphs; dense graph is the reference")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-stratum", type=int, default=0,
                        help="0 = exhaustive 220-volume comparison; otherwise sorted subset per site/modality")
    parser.add_argument("--sample-spacing-mm", type=float, default=0.5)
    parser.add_argument("--start-index", type=int, default=0, help="inclusive index in the selected manifest inventory")
    parser.add_argument("--stop-index", type=int, default=None, help="exclusive index in the selected manifest inventory")
    args = parser.parse_args()
    if args.start_index < 0 or (args.stop_index is not None and args.stop_index <= args.start_index):
        parser.error("expected 0 <= start-index < stop-index")
    rows, branch_rows, selected = [], [], []
    inventory = []
    for root in (IXI, TOPBRAIN):
        groups = defaultdict(list)
        for item in read_manifest(root):
            site = item["subject"].split("-")[1] if item["dataset"] == "ixi" else item["modality"].upper()
            stratum = f"IXI-{site}" if item["dataset"] == "ixi" else f"TopBrain-{site}"
            groups[stratum].append(item)
        for stratum, members in sorted(groups.items()):
            # Include low/high in-plane resolution and spread across patients;
            # do not take the first N alphabetically from a single site.
            ordered = sorted(members, key=lambda row: (float(row["spacing_x"]), row["subject"]))
            if args.max_per_stratum and len(ordered) > args.max_per_stratum:
                positions = np.linspace(0, len(ordered) - 1, args.max_per_stratum)
                ordered = [ordered[int(round(index))] for index in positions]
            inventory.extend((root, stratum, item) for item in ordered)
    for current_index, (root, stratum, item) in enumerate(inventory):
        if current_index < args.start_index or (args.stop_index is not None and current_index >= args.stop_index):
            continue
        site = stratum.split("-", 1)[1]
        graph_dir = root / item["modality"] / item["split"] / item["subject"] / "graphs"
        graphs = {name: SourceGraph.from_directory(graph_dir / name) for name in REPRESENTATIONS}
        report, detailed = compare_graphs(graphs, args.sample_spacing_mm)
        for record in report:
            record.update(dataset=item["dataset"], modality=item["modality"],
                          site=site, stratum=stratum, subject=item["subject"],
                          split=item["split"], spacing_x_mm=float(item["spacing_x"]),
                          spacing_y_mm=float(item["spacing_y"]), spacing_z_mm=float(item["spacing_z"]))
        rows.extend(report)
        for record in detailed:
            record.update(subject=item["subject"], stratum=stratum)
        branch_rows.extend(detailed)
        selected.append({"subject": item["subject"], "stratum": stratum, "split": item["split"],
                         "spacing_mm": [float(item[key]) for key in ("spacing_x", "spacing_y", "spacing_z")]})
        print(f"{current_index + 1}: {item['subject']} ({stratum})", flush=True)
    if not selected:
        raise ValueError("no volumes selected in the specified index range")
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "volumes.csv", rows)
    write_csv(args.output / "branches.csv", branch_rows)
    (args.output / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    summary = {group: {name: {
        key: float(np.mean([r[key] for r in rows if r["stratum"] == group and r["representation"] == name]))
        for key in ("nodes", "edges", "degree_2", "curve_f1_0.5mm", "curve_acd_mm", "curve_hd95_mm",
                    "length_weighted_absolute_branch_length_error_fraction", "length_weighted_tortuosity_error")
    } for name in REPRESENTATIONS} for group in sorted({r["stratum"] for r in rows})}
    (args.output / "summary.json").write_text(json.dumps({"subjects": len(selected), "sample_spacing_mm": args.sample_spacing_mm,
        "sampling_note": "Branch-paired, arc-uniform samples to exact continuous target polylines; endpoints repeated per branch",
        "strata": summary}, indent=2) + "\n")
    plot_summary(rows, args.output / "comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
