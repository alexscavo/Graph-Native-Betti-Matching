"""Metrics for comparing exact and inherited graph crop policies.

All positions and bounds must use the same physical coordinate system.  The
functions are deliberately independent of the source graph loader so they can
also evaluate saved VTP graphs and real-data crops.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from metrics.representation_geometry import (
    average_centerline_distance,
    curve_precision_recall_f1,
    hd95,
)


def _graph_arrays(
    positions: np.ndarray | Sequence[Sequence[float]],
    edges: np.ndarray | Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1] < 1 or not np.isfinite(positions).all():
        raise ValueError("positions must be a finite (n, dimensions) array")
    if edges.size == 0:
        edges = edges.reshape(0, 2)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (n_edges, 2)")
    if edges.size and (edges.min() < 0 or edges.max() >= len(positions)):
        raise ValueError("an edge endpoint is outside the positions array")
    return positions, edges


def graph_topology_summary(
    positions: np.ndarray | Sequence[Sequence[float]],
    edges: np.ndarray | Sequence[Sequence[int]],
) -> dict[str, int]:
    """Return graph counts, connected components, cycle rank and degree counts."""

    positions, edges = _graph_arrays(positions, edges)
    node_count = len(positions)
    parent = np.arange(node_count, dtype=np.int64)
    degree = np.zeros(node_count, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for left, right in edges:
        left, right = int(left), int(right)
        degree[left] += 1
        degree[right] += 1
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    components = len({find(node) for node in range(node_count)})
    return {
        "nodes": node_count,
        "edges": int(len(edges)),
        "components": int(components),
        "cycle_rank": int(len(edges) - node_count + components),
        "isolated_nodes": int(np.count_nonzero(degree == 0)),
        "terminal_nodes": int(np.count_nonzero(degree == 1)),
        "degree_2_nodes": int(np.count_nonzero(degree == 2)),
        "junction_nodes": int(np.count_nonzero(degree >= 3)),
    }


def distance_to_box_boundary(positions: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Return each in-box point's shortest distance to an axis-aligned box face."""

    positions = np.asarray(positions, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    if positions.ndim != 2 or bounds.shape != (positions.shape[1], 2):
        raise ValueError("bounds must have shape (position_dimensions, 2)")
    if np.any(bounds[:, 1] < bounds[:, 0]):
        raise ValueError("each upper bound must be greater than or equal to its lower bound")
    if len(positions) == 0:
        return np.empty(0, dtype=np.float64)
    face_distances = np.minimum(
        positions - bounds[:, 0], bounds[:, 1] - positions
    )
    # Negative values make points outside the crop explicit rather than silently
    # treating them as valid boundary points.
    return face_distances.min(axis=1)


def _degrees(node_count: int, edges: np.ndarray) -> np.ndarray:
    return np.bincount(edges.ravel(), minlength=node_count) if len(edges) else np.zeros(node_count, dtype=int)


def boundary_correspondence(
    exact_positions: np.ndarray,
    exact_edges: np.ndarray,
    inherited_positions: np.ndarray,
    inherited_edges: np.ndarray,
    bounds: np.ndarray,
    *,
    boundary_tolerance: float = 1e-6,
    correspondence_tolerance: float = 2.0,
) -> dict[str, float | int]:
    """Compare exact face intersections with inherited terminal samples.

    Exact crop boundary contacts are nodes on a box face.  Candidate inherited
    contacts are degree-zero/one nodes: these include last-inside samples and
    retain tangent contacts.  One-to-one pairs are selected greedily by physical
    distance, then thresholded by ``correspondence_tolerance``.
    """

    exact_positions, exact_edges = _graph_arrays(exact_positions, exact_edges)
    inherited_positions, inherited_edges = _graph_arrays(inherited_positions, inherited_edges)
    bounds = np.asarray(bounds, dtype=np.float64)
    if exact_positions.shape[1] != inherited_positions.shape[1]:
        raise ValueError("both graphs must use the same dimensionality")
    if boundary_tolerance < 0 or correspondence_tolerance < 0:
        raise ValueError("tolerances must be non-negative")

    exact_gap = distance_to_box_boundary(exact_positions, bounds)
    exact_ids = np.flatnonzero(np.abs(exact_gap) <= boundary_tolerance)
    inherited_degree = _degrees(len(inherited_positions), inherited_edges)
    inherited_ids = np.flatnonzero(inherited_degree <= 1)

    pairs: list[tuple[float, int, int]] = []
    for exact_id in exact_ids:
        if len(inherited_ids):
            distances = np.linalg.norm(
                inherited_positions[inherited_ids] - exact_positions[exact_id], axis=1
            )
            pairs.extend(
                (float(distance), int(exact_id), int(inherited_id))
                for distance, inherited_id in zip(distances, inherited_ids)
            )
    used_exact: set[int] = set()
    used_inherited: set[int] = set()
    matched_distances = []
    for distance, exact_id, inherited_id in sorted(pairs):
        if distance > correspondence_tolerance:
            break
        if exact_id not in used_exact and inherited_id not in used_inherited:
            used_exact.add(exact_id)
            used_inherited.add(inherited_id)
            matched_distances.append(distance)

    exact_count = len(exact_ids)
    inherited_gaps = distance_to_box_boundary(inherited_positions[inherited_ids], bounds)
    matched_inherited_ids = np.asarray(sorted(used_inherited), dtype=np.int64)
    matched_gaps = distance_to_box_boundary(
        inherited_positions[matched_inherited_ids], bounds
    )
    return {
        "exact_boundary_contacts": int(exact_count),
        "inherited_terminal_candidates": int(len(inherited_ids)),
        "matched_boundary_contacts": int(len(matched_distances)),
        "missing_boundary_contacts": int(exact_count - len(matched_distances)),
        "boundary_recall": float(len(matched_distances) / exact_count) if exact_count else 1.0,
        "mean_correspondence_distance": float(np.mean(matched_distances)) if matched_distances else float("nan"),
        "max_correspondence_distance": float(np.max(matched_distances)) if matched_distances else float("nan"),
        "mean_matched_inherited_face_gap": float(np.mean(matched_gaps)) if len(matched_gaps) else float("nan"),
        "max_matched_inherited_face_gap": float(np.max(matched_gaps)) if len(matched_gaps) else float("nan"),
        "mean_inherited_terminal_face_gap": float(np.mean(inherited_gaps)) if len(inherited_gaps) else float("nan"),
        "max_inherited_terminal_face_gap": float(np.max(inherited_gaps)) if len(inherited_gaps) else float("nan"),
    }


def _edge_polylines(positions: np.ndarray, edges: np.ndarray) -> list[np.ndarray]:
    return [positions[np.asarray(edge, dtype=np.int64)] for edge in edges]


def compare_cropped_graphs(
    exact_positions: np.ndarray,
    exact_edges: np.ndarray,
    inherited_positions: np.ndarray,
    inherited_edges: np.ndarray,
    bounds: np.ndarray,
    *,
    sample_spacing: float = 0.25,
    curve_tolerance: float = 0.5,
    boundary_tolerance: float = 1e-6,
    correspondence_tolerance: float = 2.0,
    preferred_edge_budget: int = 70,
    hard_token_budget: int = 120,
) -> dict[str, object]:
    """Return a complete exact-versus-inherited crop comparison."""

    exact_positions, exact_edges = _graph_arrays(exact_positions, exact_edges)
    inherited_positions, inherited_edges = _graph_arrays(inherited_positions, inherited_edges)
    exact_topology = graph_topology_summary(exact_positions, exact_edges)
    inherited_topology = graph_topology_summary(inherited_positions, inherited_edges)
    boundary = boundary_correspondence(
        exact_positions,
        exact_edges,
        inherited_positions,
        inherited_edges,
        bounds,
        boundary_tolerance=boundary_tolerance,
        correspondence_tolerance=correspondence_tolerance,
    )

    exact_curves = _edge_polylines(exact_positions, exact_edges)
    inherited_curves = _edge_polylines(inherited_positions, inherited_edges)
    if exact_curves and inherited_curves:
        curve = curve_precision_recall_f1(
            inherited_curves, exact_curves, curve_tolerance, sample_spacing
        )
        geometry = {
            "curve_precision": curve["precision"],
            "curve_recall": curve["recall"],
            "curve_f1": curve["f1"],
            "average_centerline_distance": average_centerline_distance(
                inherited_curves, exact_curves, sample_spacing
            ),
            "hd95": hd95(inherited_curves, exact_curves, sample_spacing),
        }
    else:
        geometry = {
            name: float("nan")
            for name in (
                "curve_precision",
                "curve_recall",
                "curve_f1",
                "average_centerline_distance",
                "hd95",
            )
        }

    topology_delta = {
        key: int(inherited_topology[key] - exact_topology[key])
        for key in ("nodes", "edges", "components", "cycle_rank", "isolated_nodes")
    }
    token_counts = {
        "exact_nodes": exact_topology["nodes"],
        "exact_edges": exact_topology["edges"],
        "inherited_nodes": inherited_topology["nodes"],
        "inherited_edges": inherited_topology["edges"],
        "exact_exceeds_preferred_edges": bool(exact_topology["edges"] > preferred_edge_budget),
        "inherited_exceeds_preferred_edges": bool(inherited_topology["edges"] > preferred_edge_budget),
        "exact_exceeds_hard_node_tokens": bool(exact_topology["nodes"] > hard_token_budget),
        "inherited_exceeds_hard_node_tokens": bool(inherited_topology["nodes"] > hard_token_budget),
    }
    return {
        "boundary": boundary,
        "geometry": geometry,
        "exact_topology": exact_topology,
        "inherited_topology": inherited_topology,
        "topology_delta_inherited_minus_exact": topology_delta,
        "tokens": token_counts,
        "settings": {
            "sample_spacing": float(sample_spacing),
            "curve_tolerance": float(curve_tolerance),
            "boundary_tolerance": float(boundary_tolerance),
            "correspondence_tolerance": float(correspondence_tolerance),
            "preferred_edge_budget": int(preferred_edge_budget),
            "hard_token_budget": int(hard_token_budget),
        },
    }
