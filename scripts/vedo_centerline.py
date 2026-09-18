#!/usr/bin/env python3
"""Reusable Vedo refinement stage for dense vascular centerlines.

The historical ``vedo_extractor_batch.py`` calls its dense sample network a
"graph".  In this repository that object is the centerline intermediate: it is
produced after skeletonization and before branch reduction or graph-budget
compression.  This module retains that ordering while removing Windows paths,
the Slicer flip and batch-only I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx
import numpy as np
from scipy.ndimage import center_of_mass, distance_transform_edt, label
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from skimage.morphology import skeletonize


@dataclass
class VedoCenterline:
    positions: np.ndarray
    edges: list[tuple[int, int]]
    radii_mm: np.ndarray
    raw_skeleton_voxels: int
    removed_triangle_edges: int = 0
    removed_local_redundant_edges: int = 0
    removed_invalid_edges: int = 0
    removed_small_nodes: int = 0
    orphan_candidates: int = 0
    orphan_nodes_added: int = 0
    provenance: dict[str, object] = field(default_factory=dict)


def _angle(center: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    left, right = first - center, second - center
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 1e-12:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(left, right) / denominator, -1, 1))))


def _inside_chord(start: np.ndarray, end: np.ndarray, mask: np.ndarray) -> bool:
    count = max(2, int(np.linalg.norm(end - start) * 4) + 1)
    samples = start + np.linspace(0, 1, count)[:, None] * (end - start)
    inside = np.zeros(count, dtype=bool)
    for dz in (-1e-9, 1e-9):
        for dy in (-1e-9, 1e-9):
            for dx in (-1e-9, 1e-9):
                index = np.floor(samples + 0.5 + (dz, dy, dx)).astype(int)
                valid = np.logical_and(index >= 0, index < np.asarray(mask.shape)).all(axis=1)
                selected = index[valid]
                inside[valid] |= mask[selected[:, 0], selected[:, 1], selected[:, 2]]
    return bool(inside.all())


def _surface(mask: np.ndarray, spacing: np.ndarray):
    """Build the physical-space surface on which Vedo evaluates geodesics."""

    import vedo

    vertices, faces, _, _ = marching_cubes(
        mask.astype(np.float32), level=0.5, spacing=tuple(spacing)
    )
    return vedo.Mesh([vertices, faces]).clean()


def _prune_local_redundancy(graph: nx.Graph, positions: dict[int, np.ndarray]) -> int:
    """Remove digital shortcuts with an alternative path in the same 3³ cell."""

    removed = 0
    edges = sorted(
        graph.edges,
        key=lambda edge: -float(np.linalg.norm(positions[edge[0]] - positions[edge[1]])),
    )
    for left, right in edges:
        if not graph.has_edge(left, right):
            continue
        lower = np.minimum(positions[left], positions[right]) - 1
        upper = np.maximum(positions[left], positions[right]) + 1

        def local(node: int) -> bool:
            return bool(np.logical_and(positions[node] >= lower, positions[node] <= upper).all())

        frontier = {node for node in graph.neighbors(left) if node != right and local(node)}
        reached = set(frontier) | {left}
        redundant = right in frontier
        depth = 0
        while not redundant and frontier and depth < 2:
            following = {
                node for current in frontier for node in graph.neighbors(current)
                if node not in reached and local(node)
            }
            redundant = right in following
            reached |= following
            frontier = following
            depth += 1
        if redundant:
            graph.remove_edge(left, right)
            removed += 1
    return removed


def _geodesic_length(mesh, start_voxel: np.ndarray, end_voxel: np.ndarray,
                     spacing: np.ndarray) -> float:
    start = start_voxel * spacing
    end = end_voxel * spacing
    start_id = mesh.closest_point(start, return_point_id=True)
    end_id = mesh.closest_point(end, return_point_id=True)
    path = mesh.geodesic(start_id, end_id)
    points = np.asarray(path.vertices)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def extract_vedo_centerline(
    segmentation: np.ndarray,
    spacing=(1.0, 1.0, 1.0),
    *,
    smooth_iterations: int = 4,
    smooth_alpha: float = 0.8,
    orphan_distance_mm: float = 2.5,
    orphan_merge_mm: float = 3.0,
    orphan_min_angle_deg: float = 30.0,
    minimum_component_nodes: int = 7,
    geodesic_ratio_limit: float = 2.0,
    use_orphans: bool = True,
    use_geodesic_validation: bool = True,
) -> VedoCenterline:
    """Run skeletonization followed by Vedo dense-centerline refinement."""

    mask = np.asarray(segmentation, dtype=bool)
    spacing = np.asarray(spacing, dtype=np.float64)
    skeleton = skeletonize(mask)
    skeleton_points = np.argwhere(skeleton).astype(np.float64)
    graph = nx.Graph()
    graph.add_nodes_from((index, {"pos": point}) for index, point in enumerate(skeleton_points))
    index_of = {tuple(point.astype(int)): index for index, point in enumerate(skeleton_points)}
    for index, point in enumerate(skeleton_points.astype(int)):
        for offset in np.ndindex(3, 3, 3):
            delta = np.asarray(offset) - 1
            if not delta.any():
                continue
            neighbour = index_of.get(tuple(point + delta))
            if neighbour is not None and neighbour > index:
                graph.add_edge(index, neighbour)

    mesh = _surface(mask, spacing) if (use_orphans or use_geodesic_validation) else None
    orphan_candidates = orphan_added = 0
    if use_orphans and len(skeleton_points):
        distance_to_skeleton = distance_transform_edt(~skeleton, sampling=spacing)
        orphan_mask = mask & (distance_to_skeleton > orphan_distance_mm)
        zones, zone_count = label(orphan_mask)
        candidates = np.asarray([
            center_of_mass(orphan_mask, zones, zone) for zone in range(1, zone_count + 1)
        ], dtype=np.float64).reshape(-1, 3)
        orphan_candidates = len(candidates)
        if len(candidates):
            physical = candidates * spacing
            merge_graph = nx.Graph()
            merge_graph.add_nodes_from(range(len(candidates)))
            merge_graph.add_edges_from(cKDTree(physical).query_pairs(orphan_merge_mm))
            retained = []
            for component in nx.connected_components(merge_graph):
                indices = np.asarray(sorted(component))
                centre = physical[indices].mean(axis=0)
                retained.append(int(indices[np.argmin(np.linalg.norm(physical[indices] - centre, axis=1))]))
            candidates = candidates[np.asarray(retained)]

        tree = cKDTree(skeleton_points * spacing)
        for candidate in candidates:
            count = min(5, len(skeleton_points))
            distances, neighbours = tree.query(candidate * spacing, k=count)
            neighbours = np.atleast_1d(neighbours)
            valid = []
            for neighbour in neighbours:
                try:
                    geodesic = _geodesic_length(mesh, candidate, skeleton_points[int(neighbour)], spacing)
                except Exception:
                    continue
                if geodesic > 0:
                    valid.append((geodesic, int(neighbour)))
            bridge = any(
                _angle(candidate, skeleton_points[left[1]], skeleton_points[right[1]])
                > orphan_min_angle_deg
                for left, right in combinations(valid, 2)
            )
            if valid and not bridge:
                node = graph.number_of_nodes()
                graph.add_node(node, pos=candidate)
                graph.add_edge(node, min(valid)[1])
                orphan_added += 1

    # Match the useful local cleanup in the historical script without enumerating
    # arbitrary-size cliques. Remove the longest edge of every voxel triangle.
    triangle_edges: set[tuple[int, int]] = set()
    for first in list(graph.nodes):
        neighbours = set(graph.neighbors(first))
        for second in neighbours:
            if second <= first:
                continue
            for third in neighbours & set(graph.neighbors(second)):
                if third <= second:
                    continue
                pairs = ((first, second), (first, third), (second, third))
                longest = max(
                    pairs,
                    key=lambda edge: np.linalg.norm(
                        (graph.nodes[edge[0]]["pos"] - graph.nodes[edge[1]]["pos"]) * spacing
                    ),
                )
                triangle_edges.add(tuple(sorted(longest)))
    graph.remove_edges_from(triangle_edges)
    redundant_edges = _prune_local_redundancy(
        graph, {node: graph.nodes[node]["pos"] for node in graph.nodes}
    )

    invalid_edges = []
    if use_geodesic_validation and mesh is not None:
        for left, right in list(graph.edges):
            start, end = graph.nodes[left]["pos"], graph.nodes[right]["pos"]
            if _inside_chord(start, end, mask):
                continue
            euclidean = float(np.linalg.norm((end - start) * spacing))
            try:
                geodesic = _geodesic_length(mesh, start, end, spacing)
            except Exception:
                geodesic = float("inf")
            if euclidean <= 1e-12 or geodesic / euclidean > geodesic_ratio_limit:
                invalid_edges.append((left, right))
        graph.remove_edges_from(invalid_edges)

    removed_small = []
    if minimum_component_nodes > 0:
        for component in nx.connected_components(graph):
            if len(component) < minimum_component_nodes:
                removed_small.extend(component)
        graph.remove_nodes_from(removed_small)

    # Vedo-stage sub-voxel refinement. Endpoints are pinned, and a proposed move
    # is accepted only if every incident edge remains inside the segmentation.
    positions = {node: graph.nodes[node]["pos"].copy() for node in graph.nodes}
    for _ in range(smooth_iterations):
        for node in graph.nodes:
            neighbours = list(graph.neighbors(node))
            if len(neighbours) <= 1:
                continue
            candidate = (1 - smooth_alpha) * positions[node] + smooth_alpha * np.mean(
                [positions[other] for other in neighbours], axis=0
            )
            voxel = np.floor(candidate + 0.5).astype(int)
            if np.logical_and(voxel >= 0, voxel < np.asarray(mask.shape)).all() and mask[tuple(voxel)]:
                if all(_inside_chord(candidate, positions[other], mask) for other in neighbours):
                    # Gauss-Seidel update: later neighbour checks see this move,
                    # so simultaneous individually-valid moves cannot combine
                    # into an invalid final edge.
                    positions[node] = candidate

    ordered = sorted(graph.nodes)
    remap = {node: index for index, node in enumerate(ordered)}
    output_positions = np.asarray([positions[node] for node in ordered], dtype=np.float64)
    output_edges = [(remap[left], remap[right]) for left, right in graph.edges]
    radius_map = distance_transform_edt(mask, sampling=spacing)
    voxels = np.clip(np.floor(output_positions + 0.5).astype(int), 0, np.asarray(mask.shape) - 1)
    radii = radius_map[voxels[:, 0], voxels[:, 1], voxels[:, 2]] if len(voxels) else np.empty(0)
    return VedoCenterline(
        positions=output_positions,
        edges=output_edges,
        radii_mm=np.asarray(radii),
        raw_skeleton_voxels=len(skeleton_points),
        removed_triangle_edges=len(triangle_edges),
        removed_local_redundant_edges=redundant_edges,
        removed_invalid_edges=len(invalid_edges),
        removed_small_nodes=len(removed_small),
        orphan_candidates=orphan_candidates,
        orphan_nodes_added=orphan_added,
        provenance={"backend": "vedo_surface_geodesic_dense_centerline", "vedo_stage": True},
    )
