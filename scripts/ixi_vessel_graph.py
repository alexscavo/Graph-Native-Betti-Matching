#!/usr/bin/env python3
"""Extract a sparse vessel graph from a binary segmentation volume.

The output matches the inherited Voreen source format consumed by
``audit_synthetic_mri_grid.SourceGraph``: ``nodes.csv``, ``edges.csv`` and
``graph.vvg``, with every position expressed in **world coordinates** via the
volume affine, exactly as the SyntheticMRI source graphs are.

Pipeline, in order:

1. ``skimage`` 3D skeletonization of the binary mask.
2. 26-neighbourhood voxel adjacency over the skeleton.
3. Spur pruning -- terminal branches shorter than ``spur_length`` voxels are
   dropped, iteratively. Skeletonization noise produces short hairs at vessel
   boundaries; without this the junction count roughly doubles.
4. Junction clustering -- 26-adjacent degree>=3 voxels collapse to one logical
   node. A single anatomical bifurcation is usually several adjacent skeleton
   voxels of degree 3+, and treating each as its own node is what makes a naive
   reduction disagree with a mature vessel-graph tool.
5. Branch extraction -- polylines between anchors (junctions and terminations).
6. Optional degree-2 nodes at high curvature, chosen by Douglas-Peucker so the
   retained polyline never deviates from the true centerline by more than
   ``rdp_tolerance_mm``.

Steps 3 and 4 carry as much weight as step 6: with them the node-count
distribution reproduces the reference vessel graphs, without them it does not.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# Offsets grouped by distance: face (1), edge-diagonal (sqrt 2), corner (sqrt 3).
# Order matters -- see _skeleton_adjacency.
NEIGHBOUR_OFFSETS = tuple(
    sorted(
        (
            (dz, dy, dx)
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dz, dy, dx) != (0, 0, 0)
        ),
        key=lambda o: abs(o[0]) + abs(o[1]) + abs(o[2]),
    )
)


@dataclass
class VesselGraph:
    """A sparse vessel graph in voxel coordinates of its source volume."""

    node_positions: np.ndarray                 # (N, 3) float voxel coordinates
    node_degrees: np.ndarray                   # (N,) int
    edges: list[tuple[int, int]]               # node index pairs
    centerlines: list[np.ndarray]              # per-edge (M, 3) voxel polylines
    node_radii: np.ndarray                     # (N,) float, in voxels
    skeleton_voxels: int = 0
    pruned_voxels: int = 0
    ring_count: int = 0

    @property
    def node_count(self) -> int:
        return int(len(self.node_positions))

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def betti(self) -> tuple[int, int]:
        """Return (beta_0, beta_1) of the undirected multigraph."""
        parent = list(range(self.node_count))

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for left, right in self.edges:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root
        components = len({find(i) for i in range(self.node_count)}) if self.node_count else 0
        beta_1 = self.edge_count - self.node_count + components
        return int(components), int(beta_1)


@dataclass(frozen=True)
class DenseCenterline:
    """Immutable, high-detail intermediate shared by every graph representation."""

    graph: VesselGraph
    segmentation: np.ndarray = field(repr=False)
    spacing: np.ndarray
    provenance: dict[str, object] = field(default_factory=dict)


def _prune_local_redundancy(
    points: np.ndarray, neighbours: list[set[int]]
) -> int:
    """Drop edges that are shortcuts across a locally-connected neighbourhood.

    A digital curve skeleton is 26-connected, so a diagonal run makes every triple
    of consecutive voxels a triangle and any two-voxel-thick patch a small clique.
    Those artifacts are pure quantisation: on a real subject, raw 26-adjacency
    reports beta_1 = 732 where the skeleton's true value is 161.

    An edge is removed only when its endpoints remain joined by a path of length
    two or three whose intermediate voxels all lie inside the joint 3x3x3
    neighbourhood of the endpoints. Such an edge is redundant by construction, so
    removal cannot disconnect anything or change global topology -- it only
    collapses the local clique. Longest edges are considered first so that face
    connections survive in preference to diagonals.
    """

    removed = 0
    edges = sorted(
        {(min(i, j), max(i, j)) for i, group in enumerate(neighbours) for j in group},
        key=lambda e: -float(np.linalg.norm(points[e[0]] - points[e[1]])),
    )
    for left, right in edges:
        if right not in neighbours[left]:
            continue
        lower = np.minimum(points[left], points[right]) - 1
        upper = np.maximum(points[left], points[right]) + 1

        def local(voxel: int) -> bool:
            position = points[voxel]
            return bool(np.all(position >= lower) and np.all(position <= upper))

        frontier = {k for k in neighbours[left] if k != right and local(k)}
        reached = set(frontier)
        redundant = any(right in neighbours[k] for k in frontier)
        depth = 0
        while not redundant and frontier and depth < 2:
            frontier = {
                n
                for k in frontier
                for n in neighbours[k]
                if n not in reached and n != left and local(n)
            }
            redundant = any(right in neighbours[k] for k in frontier)
            reached |= frontier
            depth += 1
        if redundant:
            neighbours[left].discard(right)
            neighbours[right].discard(left)
            removed += 1
    return removed


def _skeleton_adjacency(skeleton: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Return skeleton voxel coordinates and a topology-faithful adjacency."""

    points = np.argwhere(skeleton)
    if not len(points):
        return points, []
    index_of = {(int(z), int(y), int(x)): i for i, (z, y, x) in enumerate(points)}
    neighbours: list[set[int]] = [set() for _ in points]
    for dz, dy, dx in NEIGHBOUR_OFFSETS:
        for i, (z, y, x) in enumerate(points):
            j = index_of.get((int(z) + dz, int(y) + dy, int(x) + dx))
            if j is not None and j > i:
                neighbours[i].add(j)
                neighbours[j].add(i)
    _prune_local_redundancy(points, neighbours)
    return points, [sorted(group) for group in neighbours]


def _prune_spurs(adjacency: list[list[int]], spur_length: int) -> set[int]:
    """Iteratively drop terminal branches of at most ``spur_length`` voxels."""

    if spur_length <= 0:
        return set()
    dead: set[int] = set()
    changed = True
    while changed:
        changed = False
        degree = [
            0 if i in dead else sum(1 for n in neigh if n not in dead)
            for i, neigh in enumerate(adjacency)
        ]
        for start in range(len(adjacency)):
            if start in dead or degree[start] != 1:
                continue
            path = [start]
            previous, current = None, start
            while True:
                nxt = [n for n in adjacency[current] if n != previous and n not in dead]
                if len(nxt) != 1:
                    break
                previous, current = current, nxt[0]
                path.append(current)
                if degree[current] != 2 or len(path) > spur_length + 1:
                    break
            # Only remove a hair that terminates into a junction; a short free
            # floating fragment is a component decision, not a spur decision.
            if len(path) <= spur_length + 1 and degree[path[-1]] >= 3:
                dead.update(path[:-1])
                changed = True
    return dead


def _cluster_junctions(
    adjacency: list[list[int]],
    dead: set[int],
    points: np.ndarray | None = None,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    max_extent_mm: float = 1.5,
) -> tuple[dict[int, int], list[list[int]]]:
    """Collapse adjacent degree>=3 voxels into shared cluster ids, bounded in extent.

    A real bifurcation occupies one to three adjacent skeleton voxels. But merging
    every 26-adjacent branch voxel unconditionally fuses DISTINCT junctions wherever
    two vessels touch: measured on a real subject, one cluster reached 20 voxels
    spanning 4.79 mm with 19 branches attached. Collapsing that to a single node
    invents a hub joining branches that were never mutually connected, and
    manufactures cycles around it.

    A cluster therefore only absorbs a voxel while its spatial extent stays within
    ``max_extent_mm``; beyond that the junctions stay separate, joined by short
    edges, which is what the skeleton actually says.
    """

    degree = {
        i: sum(1 for n in neigh if n not in dead)
        for i, neigh in enumerate(adjacency)
        if i not in dead
    }
    junction = {i for i, d in degree.items() if d >= 3}
    spacing_array = np.asarray(spacing, dtype=np.float64)

    members: dict[int, list[int]] = {i: [i] for i in junction}
    parent = {i: i for i in junction}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in sorted(junction):
        for j in adjacency[i]:
            if j not in junction:
                continue
            root_i, root_j = find(i), find(j)
            if root_i == root_j:
                continue
            if points is not None and max_extent_mm > 0:
                combined = points[np.asarray(members[root_i] + members[root_j])]
                span = float(
                    np.linalg.norm(
                        (combined.max(axis=0) - combined.min(axis=0)) * spacing_array
                    )
                )
                if span > max_extent_mm:
                    continue  # keep these bifurcations separate
            parent[root_j] = root_i
            members[root_i].extend(members[root_j])
            members.pop(root_j, None)

    clusters: dict[int, list[int]] = {}
    for i in junction:
        clusters.setdefault(find(i), []).append(i)
    cluster_of = {
        voxel: cid
        for cid, (_, group) in enumerate(sorted(clusters.items()))
        for voxel in group
    }
    ordered = [group for _, group in sorted(clusters.items())]
    return cluster_of, ordered


def smooth_polyline(
    polyline: np.ndarray,
    mask: np.ndarray,
    *,
    iterations: int = 5,
    alpha: float = 0.5,
) -> np.ndarray:
    """Laplacian-smooth a centerline, keeping every point inside the mask.

    A digital skeleton steps between voxel centres, so its direction changes in
    coarse increments -- measured median turn angle 45 degrees, against 6.5 degrees
    for the reference SyntheticMRI centerlines. Douglas-Peucker selects nodes by
    deviation from a chord, so on a raw staircase it samples lattice corners rather
    than anatomical bends. Smoothing first makes curvature-driven node placement
    respond to the vessel instead of the quantisation.

    Endpoints are pinned: they carry the junction and termination positions, which
    are topological anchors and must not drift. Any interior point that smoothing
    would push outside the segmentation is reverted, so the centerline never leaves
    the vessel it describes.
    """

    if len(polyline) < 3 or iterations <= 0:
        return polyline
    extent = np.asarray(mask.shape) - 1
    current = np.array(polyline, dtype=np.float64, copy=True)
    for _ in range(iterations):
        candidate = current.copy()
        candidate[1:-1] = (1.0 - alpha) * current[1:-1] + alpha * 0.5 * (
            current[:-2] + current[2:]
        )
        voxel = np.clip(np.rint(candidate).astype(np.int64), 0, extent)
        outside = ~mask[voxel[:, 0], voxel[:, 1], voxel[:, 2]]
        candidate[outside] = current[outside]
        # A sample can remain inside while either adjacent straight segment cuts
        # through background. Reject such moves as well. Repeat because reverting
        # one sample changes the segment tested by its neighbour; the raw voxel
        # path is the finite fallback and is valid under the same sampling rule.
        changed = True
        while changed:
            changed = False
            reject = []
            for index in range(1, len(candidate) - 1):
                if np.array_equal(candidate[index], current[index]):
                    continue
                if (
                    chord_mask_fraction(candidate[index - 1], candidate[index], mask) < 1.0
                    or chord_mask_fraction(candidate[index], candidate[index + 1], mask) < 1.0
                ):
                    reject.append(index)
            if reject:
                candidate[np.asarray(reject)] = current[np.asarray(reject)]
                changed = True
        current = candidate
    return current


def rdp_keep_mask(polyline: np.ndarray, tolerance) -> np.ndarray:
    """Douglas-Peucker: mark points to retain so deviation stays within tolerance.

    ``tolerance`` may be a scalar, or a per-point array. The per-point form lets the
    bound follow the local vessel radius: a fixed bound large enough to keep the node
    count affordable (0.94 mm) exceeds the median vessel radius (0.80 mm), so a chord
    can satisfy it and still leave the lumen, which is what puts edges through
    background. Within a candidate segment the tightest local tolerance governs.
    """

    keep = np.zeros(len(polyline), dtype=bool)
    if len(polyline) == 0:
        return keep
    keep[0] = keep[-1] = True
    scalar = np.isscalar(tolerance)
    if len(polyline) < 3:
        return keep
    if scalar and tolerance <= 0:
        keep[:] = True
        return keep
    limits = None if scalar else np.asarray(tolerance, dtype=np.float64)
    stack = [(0, len(polyline) - 1)]
    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue
        anchor, tip = polyline[start], polyline[end]
        direction = tip - anchor
        length = float(np.linalg.norm(direction))
        segment = polyline[start + 1 : end]
        if length < 1e-9:
            distance = np.linalg.norm(segment - anchor, axis=1)
        else:
            distance = np.linalg.norm(np.cross(segment - anchor, direction), axis=1) / length
        limit = float(tolerance) if scalar else float(limits[start : end + 1].min())
        worst = int(np.argmax(distance))
        if float(distance[worst]) > limit:
            split = start + 1 + worst
            keep[split] = True
            stack.append((start, split))
            stack.append((split, end))
    return keep


def chord_mask_fraction(
    start: np.ndarray,
    end: np.ndarray,
    mask: np.ndarray,
    *,
    samples_per_voxel: float = 4.0,
) -> float:
    """Return the fraction of a straight voxel-coordinate chord inside ``mask``."""

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    count = max(2, int(np.linalg.norm(end - start) * samples_per_voxel) + 1)
    samples = start + np.linspace(0.0, 1.0, count)[:, None] * (end - start)
    extent = np.asarray(mask.shape, dtype=np.int64) - 1
    inside = np.zeros(count, dtype=bool)
    # Voxels are closed cells. At an exact face/edge/corner boundary, accept the
    # point when *any* incident cell is foreground. This avoids inventing an
    # outside point for a valid diagonal transition between touching voxels.
    epsilon = 1e-9
    for dz in (-epsilon, epsilon):
        for dy in (-epsilon, epsilon):
            for dx in (-epsilon, epsilon):
                indices = np.floor(samples + 0.5 + (dz, dy, dx)).astype(np.int64)
                valid = np.logical_and(indices >= 0, indices <= extent).all(axis=1)
                if valid.any():
                    selected = indices[valid]
                    inside[valid] |= mask[selected[:, 0], selected[:, 1], selected[:, 2]]
    return float(inside.mean())


def enforce_chord_containment(
    polyline: np.ndarray,
    keep: np.ndarray,
    mask: np.ndarray,
    *,
    minimum_fraction: float = 1.0,
) -> np.ndarray:
    """Add dense samples until every retained straight chord stays in the lumen.

    RDP bounds distance from a centreline but does not imply containment in a
    narrow or non-convex vessel. This deterministic refinement never removes an
    RDP point. For an invalid interval it inserts the interior dense sample with
    greatest deviation from the chord (or the midpoint for a degenerate tie),
    then rechecks the two smaller intervals. In the worst case every dense sample
    is retained, providing a finite, auditable fallback rather than an invalid
    training edge.
    """

    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError("minimum_fraction must lie in [0, 1]")
    refined = np.asarray(keep, dtype=bool).copy()
    refined[0] = refined[-1] = True
    pending = [
        (int(left), int(right))
        for left, right in zip(np.flatnonzero(refined)[:-1], np.flatnonzero(refined)[1:])
    ]
    while pending:
        start, end = pending.pop()
        if end - start < 2:
            continue
        if chord_mask_fraction(polyline[start], polyline[end], mask) >= minimum_fraction:
            continue
        anchor, tip = polyline[start], polyline[end]
        direction = tip - anchor
        length = float(np.linalg.norm(direction))
        interior = polyline[start + 1 : end]
        if length < 1e-9:
            distances = np.linalg.norm(interior - anchor, axis=1)
        else:
            distances = np.linalg.norm(
                np.cross(interior - anchor, direction), axis=1
            ) / length
        split = start + 1 + int(np.argmax(distances))
        if float(distances.max(initial=0.0)) <= 1e-12:
            split = (start + end) // 2
        refined[split] = True
        pending.append((start, split))
        pending.append((split, end))
    return refined


def _branch_polylines(
    adjacency: list[list[int]],
    dead: set[int],
    anchor_of: dict[int, int],
) -> tuple[list[tuple[int, int, list[int]]], list[list[int]]]:
    """Walk chains between anchors and return branches plus ordered pure rings."""

    branches: list[tuple[int, int, list[int]]] = []
    visited: set[tuple[int, int]] = set()
    for start in sorted(anchor_of):
        for first in adjacency[start]:
            if first in dead or (start, first) in visited:
                continue
            visited.add((start, first))
            path = [start, first]
            previous, current = start, first
            while current not in anchor_of:
                nxt = [
                    n for n in adjacency[current] if n != previous and n not in dead
                ]
                if len(nxt) != 1:
                    break
                previous, current = current, nxt[0]
                path.append(current)
            if current not in anchor_of:
                continue
            visited.add((path[-1], path[-2]))
            branches.append((anchor_of[start], anchor_of[current], path))

    # Components made entirely of degree-2 voxels have no anchor. Preserve their
    # cyclic order so the caller can introduce deterministic geometric nodes rather
    # than silently dropping a connected component and one beta-1 generator.
    rings: list[list[int]] = []
    seen = set(voxel for _, _, path in branches for voxel in path) | set(anchor_of) | dead
    for seed in range(len(adjacency)):
        if seed in seen or seed in dead:
            continue
        component = []
        stack = [seed]
        local = {seed}
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in adjacency[node]:
                if neighbour not in local and neighbour not in dead:
                    local.add(neighbour)
                    stack.append(neighbour)
        seen |= local
        if len(component) < 3 or any(
            sum(1 for neighbour in adjacency[node] if neighbour not in dead) != 2
            for node in component
        ):
            continue

        local = set(component)
        seed = min(local)
        ordered = [seed]
        previous: int | None = None
        current = seed
        while True:
            candidates = sorted(
                neighbour
                for neighbour in adjacency[current]
                if neighbour in local and neighbour != previous
            )
            if not candidates:
                ordered = []
                break
            following = candidates[0]
            if following == seed:
                break
            if following in ordered:
                ordered = []
                break
            ordered.append(following)
            previous, current = current, following
        if len(ordered) == len(local):
            rings.append(ordered)
    return branches, rings


def build_vessel_graph(
    segmentation: np.ndarray,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    spur_length: int = 4,
    rdp_tolerance_mm: float = 0.0,
    intermediate_nodes: bool = False,
    min_component_voxels: int = 0,
    max_artifact_self_loop: int = 4,
    smooth_iterations: int = 5,
    smooth_alpha: float = 0.5,
    max_junction_extent_mm: float = 1.5,
    radius_fraction: float = 0.0,
    min_tolerance_mm: float = 0.1,
    enforce_lumen_containment: bool = False,
    minimum_chord_fraction: float = 1.0,
) -> VesselGraph:
    """Reduce a binary segmentation to a sparse vessel graph.

    ``rdp_tolerance_mm`` is a physical bound, converted to voxels per axis using
    ``spacing`` so anisotropic volumes get a consistent geometric guarantee.
    It is ignored unless ``intermediate_nodes`` is set. With intermediate nodes
    enabled, a non-positive tolerance and no radius bound retain every dense
    centerline sample; this is the dense comparison representation.
    """

    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    mask = np.asarray(segmentation) > 0
    if not mask.any():
        empty = np.empty((0, 3), dtype=np.float64)
        return VesselGraph(empty, np.empty((0,), int), [], [], np.empty((0,)))

    skeleton = skeletonize(mask)
    points, adjacency = _skeleton_adjacency(skeleton)
    if not len(points):
        empty = np.empty((0, 3), dtype=np.float64)
        return VesselGraph(empty, np.empty((0,), int), [], [], np.empty((0,)))

    spacing_array = np.asarray(spacing, dtype=np.float64)
    dead = _prune_spurs(adjacency, spur_length)
    cluster_of, clusters = _cluster_junctions(
        adjacency, dead, points, spacing_array, max_junction_extent_mm
    )

    degree = {
        i: sum(1 for n in adjacency[i] if n not in dead)
        for i in range(len(points))
        if i not in dead
    }

    positions: list[np.ndarray] = []
    representative_voxels: list[int] = []
    anchor_of: dict[int, int] = {}
    for members in clusters:
        anchor = len(positions)
        # Use the cluster member NEAREST the centroid, not the centroid itself. For an
        # elongated or L-shaped junction the centroid can fall outside the vessel, which
        # put 2.6% of centerlines partly through background. A member is a skeleton
        # voxel, so it is inside the mask by construction.
        member_points = points[np.asarray(members)].astype(np.float64)
        centre = member_points.mean(axis=0)
        nearest = int(np.argmin(np.linalg.norm(member_points - centre, axis=1)))
        representative = members[nearest]
        positions.append(member_points[nearest])
        representative_voxels.append(representative)
        for voxel in members:
            anchor_of[voxel] = anchor
    for voxel, value in sorted(degree.items()):
        if value == 1:
            anchor_of[voxel] = len(positions)
            positions.append(points[voxel].astype(np.float64))
            representative_voxels.append(voxel)

    branches, rings = _branch_polylines(adjacency, dead, anchor_of)

    # A pure ring has no anatomical endpoint or junction. Introduce three
    # deterministic degree-2 geometric nodes, yielding a simple three-edge cycle.
    # This preserves the component and beta-1 while avoiding self-loops and parallel
    # edges, neither of which the downstream binary adjacency matrix can represent.
    for ring in rings:
        split = (0, len(ring) // 3, (2 * len(ring)) // 3)
        ring_nodes = []
        for index in split:
            ring_nodes.append(len(positions))
            positions.append(points[ring[index]].astype(np.float64))
            representative_voxels.append(ring[index])
        for part in range(3):
            start = split[part]
            end = split[(part + 1) % 3]
            if part < 2:
                path = ring[start : end + 1]
            else:
                path = ring[start:] + [ring[0]]
            branches.append(
                (ring_nodes[part], ring_nodes[(part + 1) % 3], path)
            )
    ring_count = len(rings)

    # Junction clustering collapses adjacent degree-3+ voxels into one node, which
    # turns a branch leaving and immediately re-entering the same cluster into a
    # self-loop. Those are quantisation artifacts (median length 2 voxels) and each
    # one adds a phantom cycle: on a real subject they inflated beta_1 from 162 to
    # 257. A genuine vascular loop leaves its junction and travels before returning,
    # so only short self-loops are dropped.
    if max_artifact_self_loop > 0:
        branches = [
            (node_a, node_b, path)
            for node_a, node_b, path in branches
            if node_a != node_b or len(path) > max_artifact_self_loop
        ]

    def route_inside_anchor(start: int, target: int, anchor: int) -> list[int]:
        """Shortest skeleton route inside one collapsed junction cluster."""

        if start == target:
            return [start]
        previous: dict[int, int | None] = {start: None}
        queue = deque([start])
        found = False
        while queue and not found:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if neighbour in previous or anchor_of.get(neighbour) != anchor:
                    continue
                previous[neighbour] = current
                queue.append(neighbour)
                if neighbour == target:
                    found = True
                    break
        if target not in previous:
            return [start, target]
        route = [target]
        while route[-1] != start:
            parent = previous[route[-1]]
            if parent is None:  # only possible for the start node
                break
            route.append(parent)
        return list(reversed(route))

    # Preserve the real skeleton route through each collapsed junction. Without
    # this prefix/suffix, replacing a branch endpoint by the shared representative
    # node creates a direct shortcut that can leave a non-convex junction lumen.
    routed_branches = []
    for node_a, node_b, path in branches:
        representative_a = representative_voxels[node_a]
        representative_b = representative_voxels[node_b]
        prefix = route_inside_anchor(representative_a, path[0], node_a)
        suffix = route_inside_anchor(path[-1], representative_b, node_b)
        routed_branches.append(
            (node_a, node_b, prefix[:-1] + path + suffix[1:])
        )
    branches = routed_branches

    tolerance_voxels = (
        float(rdp_tolerance_mm / max(float(spacing_array.min()), 1e-9))
        if rdp_tolerance_mm > 0
        else 0.0
    )

    radius_map = (
        distance_transform_edt(mask, sampling=spacing_array)
        if radius_fraction > 0
        else None
    )

    edges: list[tuple[int, int]] = []
    centerlines: list[np.ndarray] = []
    for node_a, node_b, path in branches:
        polyline = points[np.asarray(path)].astype(np.float64)
        polyline[0] = positions[node_a]
        polyline[-1] = positions[node_b]
        polyline = smooth_polyline(
            polyline, mask, iterations=smooth_iterations, alpha=smooth_alpha
        )
        if not intermediate_nodes or len(polyline) < 3:
            edges.append((node_a, node_b))
            centerlines.append(polyline)
            continue
        if radius_map is not None:
            # Bound deviation by a fraction of the LOCAL vessel radius so a chord can
            # never leave the lumen it describes, instead of by a fixed distance that
            # is too loose for thin vessels and needlessly tight for thick ones.
            voxel = np.clip(
                np.rint(polyline).astype(np.int64), 0, np.asarray(mask.shape) - 1
            )
            local = radius_map[voxel[:, 0], voxel[:, 1], voxel[:, 2]]
            limit = np.maximum(radius_fraction * local, min_tolerance_mm)
            if rdp_tolerance_mm > 0:
                limit = np.minimum(limit, rdp_tolerance_mm)
            keep = rdp_keep_mask(polyline * spacing_array, limit)
        else:
            keep = rdp_keep_mask(polyline * spacing_array, rdp_tolerance_mm)
        if enforce_lumen_containment:
            keep = enforce_chord_containment(
                polyline,
                keep,
                mask,
                minimum_fraction=minimum_chord_fraction,
            )
        split_indices = [i for i in range(1, len(polyline) - 1) if keep[i]]
        previous_index, previous_node = 0, node_a
        for index in split_indices:
            new_node = len(positions)
            positions.append(polyline[index])
            edges.append((previous_node, new_node))
            centerlines.append(polyline[previous_index : index + 1])
            previous_index, previous_node = index, new_node
        edges.append((previous_node, node_b))
        centerlines.append(polyline[previous_index:])

    if min_component_voxels > 0 and edges:
        edges, centerlines, positions = _drop_small_components(
            edges, centerlines, positions, min_component_voxels
        )

    # Degenerate edges: two nodes at the same position joined by a zero-length edge.
    # They carry no geometry and break any length-normalised quantity downstream.
    keep_edges, keep_lines = [], []
    for (left, right), polyline in zip(edges, centerlines):
        if left == right:
            keep_edges.append((left, right)); keep_lines.append(polyline); continue
        if np.linalg.norm(
            (np.asarray(positions[left]) - np.asarray(positions[right])) * spacing_array
        ) < 1e-6:
            continue
        keep_edges.append((left, right)); keep_lines.append(polyline)
    edges, centerlines = keep_edges, keep_lines

    node_positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    degrees = np.zeros(len(node_positions), dtype=np.int64)
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1

    distance_map = distance_transform_edt(mask, sampling=spacing_array)
    if len(node_positions):
        voxel_index = np.clip(
            np.rint(node_positions).astype(np.int64), 0, np.asarray(mask.shape) - 1
        )
        radii = distance_map[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]]
    else:
        radii = np.empty((0,), dtype=np.float64)

    return VesselGraph(
        node_positions=node_positions,
        node_degrees=degrees,
        edges=edges,
        centerlines=centerlines,
        node_radii=np.asarray(radii, dtype=np.float64),
        skeleton_voxels=int(len(points)),
        pruned_voxels=int(len(dead)),
        ring_count=ring_count,
    )


def extract_dense_centerline(
    segmentation: np.ndarray,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    **topology_options,
) -> DenseCenterline:
    """Extract topology once and retain every post-cleanup centerline sample."""

    options = dict(topology_options)
    options.pop("smooth_iterations", None)
    options.pop("smooth_alpha", None)
    graph = build_vessel_graph(
        segmentation,
        spacing=spacing,
        intermediate_nodes=True,
        rdp_tolerance_mm=0.0,
        radius_fraction=0.0,
        smooth_iterations=0,
        **options,
    )
    mask = np.asarray(segmentation, dtype=bool).copy()
    mask.setflags(write=False)
    spacing_array = np.asarray(spacing, dtype=np.float64).copy()
    spacing_array.setflags(write=False)
    return DenseCenterline(
        graph=graph,
        segmentation=mask,
        spacing=spacing_array,
        provenance={
            "backend": "skimage_skeletonize_3d",
            "skeleton_voxels": graph.skeleton_voxels,
            "pruned_voxels": graph.pruned_voxels,
            "topology_options": options,
        },
    )


def _dense_branch_paths(graph: VesselGraph) -> list[list[int]]:
    """Return maximal degree-2 chains, including deterministic pure cycles."""

    adjacency: list[list[int]] = [[] for _ in range(graph.node_count)]
    for left, right in graph.edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    anchors = {index for index, neighbours in enumerate(adjacency) if len(neighbours) != 2}
    visited: set[tuple[int, int]] = set()
    paths: list[list[int]] = []

    def mark(left: int, right: int) -> None:
        visited.add((min(left, right), max(left, right)))

    for start in sorted(anchors):
        for first in sorted(adjacency[start]):
            key = (min(start, first), max(start, first))
            if key in visited:
                continue
            mark(start, first)
            path = [start, first]
            previous, current = start, first
            while current not in anchors:
                following = next(item for item in adjacency[current] if item != previous)
                mark(current, following)
                path.append(following)
                previous, current = current, following
            paths.append(path)

    # Any unvisited edge belongs to a component made entirely of degree-2 nodes.
    for seed, neighbours in enumerate(adjacency):
        if not neighbours:
            continue
        first = neighbours[0]
        if (min(seed, first), max(seed, first)) in visited:
            continue
        cycle = [seed]
        previous, current = seed, first
        mark(seed, first)
        while current != seed:
            cycle.append(current)
            following = next(item for item in adjacency[current] if item != previous)
            mark(current, following)
            previous, current = current, following
        split = sorted({0, len(cycle) // 3, (2 * len(cycle)) // 3})
        for part, begin in enumerate(split):
            end = split[(part + 1) % len(split)]
            if part + 1 < len(split):
                paths.append(cycle[begin : end + 1])
            else:
                paths.append(cycle[begin:] + [cycle[0]])
    return paths


def derive_graph_from_dense(
    dense: DenseCenterline,
    representation: str,
    *,
    adaptive_tolerance_mm: float = 0.0,
    adaptive_radius_fraction: float = 0.0,
    min_tolerance_mm: float = 0.1,
    smooth_iterations: int = 5,
    smooth_alpha: float = 0.5,
    enforce_lumen_containment: bool = True,
    minimum_chord_fraction: float = 1.0,
) -> VesselGraph:
    """Derive one compact representation without repeating skeletonization."""

    if representation == "dense":
        return dense.graph
    if representation not in {"junction_only", "adaptive"}:
        raise ValueError(f"unknown representation: {representation}")
    from scipy.ndimage import distance_transform_edt

    source = dense.graph
    paths = _dense_branch_paths(source)
    anchors = sorted({path[0] for path in paths} | {path[-1] for path in paths})
    output_index = {source_index: index for index, source_index in enumerate(anchors)}
    positions = [source.node_positions[index].copy() for index in anchors]
    radii = [float(source.node_radii[index]) for index in anchors]
    edges: list[tuple[int, int]] = []
    centerlines: list[np.ndarray] = []
    radius_map = distance_transform_edt(dense.segmentation, sampling=dense.spacing)

    for path in paths:
        polyline = source.node_positions[np.asarray(path)].astype(np.float64, copy=True)
        polyline = smooth_polyline(
            polyline,
            dense.segmentation,
            iterations=smooth_iterations,
            alpha=smooth_alpha,
        )
        keep = np.zeros(len(polyline), dtype=bool)
        keep[0] = keep[-1] = True
        if representation == "adaptive" and len(polyline) > 2:
            if adaptive_radius_fraction > 0:
                voxel = np.clip(
                    np.floor(polyline + 0.5).astype(np.int64),
                    0,
                    np.asarray(dense.segmentation.shape) - 1,
                )
                local = radius_map[voxel[:, 0], voxel[:, 1], voxel[:, 2]]
                limit = np.maximum(adaptive_radius_fraction * local, min_tolerance_mm)
                if adaptive_tolerance_mm > 0:
                    limit = np.minimum(limit, adaptive_tolerance_mm)
            else:
                limit = adaptive_tolerance_mm
            keep = rdp_keep_mask(polyline * dense.spacing, limit)
            if enforce_lumen_containment:
                keep = enforce_chord_containment(
                    polyline, keep, dense.segmentation,
                    minimum_fraction=minimum_chord_fraction,
                )
        retained = np.flatnonzero(keep)
        previous_node = output_index[path[0]]
        previous_sample = 0
        for sample_index in retained[1:-1]:
            node = len(positions)
            positions.append(polyline[sample_index].copy())
            voxel = np.clip(
                np.floor(polyline[sample_index] + 0.5).astype(np.int64),
                0, np.asarray(dense.segmentation.shape) - 1,
            )
            radii.append(float(radius_map[tuple(voxel)]))
            edges.append((previous_node, node))
            centerlines.append(polyline[previous_sample : sample_index + 1])
            previous_node, previous_sample = node, int(sample_index)
        edges.append((previous_node, output_index[path[-1]]))
        centerlines.append(polyline[previous_sample:])

    node_positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    degrees = np.zeros(len(node_positions), dtype=np.int64)
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    return VesselGraph(
        node_positions=node_positions,
        node_degrees=degrees,
        edges=edges,
        centerlines=centerlines,
        node_radii=np.asarray(radii, dtype=np.float64),
        skeleton_voxels=source.skeleton_voxels,
        pruned_voxels=source.pruned_voxels,
        ring_count=source.ring_count,
    )


def build_representation_family(
    segmentation: np.ndarray,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    adaptive_tolerance_mm: float,
    adaptive_radius_fraction: float = 0.0,
    enforce_lumen_containment: bool = True,
    minimum_chord_fraction: float = 1.0,
    **shared_options,
) -> dict[str, VesselGraph]:
    """Build the three Phase-1 comparison representations.

    All variants use identical extraction and topology parameters. They differ
    only in the degree-2 geometry samples retained along each branch:

    - ``junction_only`` retains no degree-2 geometry nodes;
    - ``adaptive`` retains the minimum RDP-selected geometry nodes;
    - ``dense`` retains every sample on the raw post-topology-cleanup centerline,
      before smoothing or RDP simplification.

    Keeping this family in one API prevents comparison scripts from accidentally
    changing topology parameters between variants.
    """

    common = dict(shared_options)
    smooth_iterations = int(common.pop("smooth_iterations", 5))
    smooth_alpha = float(common.pop("smooth_alpha", 0.5))
    min_tolerance_mm = float(common.pop("min_tolerance_mm", 0.1))
    dense = extract_dense_centerline(segmentation, spacing=spacing, **common)
    derivation = dict(
        smooth_iterations=smooth_iterations,
        smooth_alpha=smooth_alpha,
        enforce_lumen_containment=enforce_lumen_containment,
        minimum_chord_fraction=minimum_chord_fraction,
    )
    return {
        "junction_only": derive_graph_from_dense(dense, "junction_only", **derivation),
        "adaptive": derive_graph_from_dense(
            dense,
            "adaptive",
            adaptive_tolerance_mm=adaptive_tolerance_mm,
            adaptive_radius_fraction=adaptive_radius_fraction,
            min_tolerance_mm=min_tolerance_mm,
            **derivation,
        ),
        "dense": dense.graph,
    }


def _drop_small_components(
    edges: list[tuple[int, int]],
    centerlines: list[np.ndarray],
    positions: list[np.ndarray],
    min_voxels: int,
) -> tuple[list[tuple[int, int]], list[np.ndarray], list[np.ndarray]]:
    """Remove connected components whose total centerline length is tiny."""

    parent = list(range(len(positions)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        root_l, root_r = find(left), find(right)
        if root_l != root_r:
            parent[root_r] = root_l

    size: dict[int, int] = {}
    for index, (left, _) in enumerate(edges):
        size[find(left)] = size.get(find(left), 0) + len(centerlines[index])

    keep_roots = {root for root, count in size.items() if count >= min_voxels}
    kept = [i for i, (left, _) in enumerate(edges) if find(left) in keep_roots]
    used: dict[int, int] = {}
    new_positions: list[np.ndarray] = []
    new_edges: list[tuple[int, int]] = []
    new_centerlines: list[np.ndarray] = []
    for index in kept:
        left, right = edges[index]
        for node in (left, right):
            if node not in used:
                used[node] = len(new_positions)
                new_positions.append(positions[node])
        new_edges.append((used[left], used[right]))
        new_centerlines.append(centerlines[index])
    return new_edges, new_centerlines, new_positions


def voxel_to_world(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    return (homogeneous @ np.asarray(affine, dtype=np.float64).T)[:, :3]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_source_graph(
    directory: Path, graph: VesselGraph, affine: np.ndarray, shape: Sequence[int]
) -> None:
    """Write nodes.csv / edges.csv / graph.vvg in world coordinates.

    ``SourceGraph.from_directory`` requires a one-to-one correspondence between
    CSV edges and VVG centerlines, so both are emitted from the same list in the
    same order.
    """

    directory.mkdir(parents=True, exist_ok=True)
    world_nodes = (
        voxel_to_world(graph.node_positions, affine)
        if graph.node_count
        else np.empty((0, 3))
    )
    extent = np.asarray(shape, dtype=np.float64) - 1.0
    at_border = (
        np.logical_or(graph.node_positions <= 0.5, graph.node_positions >= extent - 0.5).any(axis=1)
        if graph.node_count
        else np.empty((0,), dtype=bool)
    )

    node_rows = ["id;pos_x;pos_y;pos_z;degree;isAtSampleBorder"]
    for index, position in enumerate(world_nodes):
        node_rows.append(
            f"{index};{position[0]:.6f};{position[1]:.6f};{position[2]:.6f};"
            f"{int(graph.node_degrees[index])};{int(at_border[index])}"
        )
    _atomic_write(directory / "nodes.csv", "\n".join(node_rows) + "\n")

    edge_rows = [
        "id;node1id;node2id;length;distance;curveness;volume;avgCrossSection;"
        "minRadiusAvg;minRadiusStd;avgRadiusAvg;avgRadiusStd;maxRadiusAvg;maxRadiusStd;"
        "roundnessAvg;roundnessStd;node1_degree;node2_degree;num_voxels;hasNodeAtSampleBorder"
    ]
    vvg_edges = []
    for index, ((left, right), polyline) in enumerate(zip(graph.edges, graph.centerlines)):
        world_polyline = voxel_to_world(polyline, affine)
        steps = np.linalg.norm(np.diff(world_polyline, axis=0), axis=1)
        length = float(steps.sum())
        straight = float(np.linalg.norm(world_polyline[-1] - world_polyline[0]))
        curveness = length / straight if straight > 1e-9 else 1.0
        radius = float((graph.node_radii[left] + graph.node_radii[right]) / 2.0)
        border = int(bool(at_border[left] or at_border[right]))
        edge_rows.append(
            f"{index};{left};{right};{length:.6f};{straight:.6f};{curveness:.6f};"
            f"0;0;{radius:.6f};0;{radius:.6f};0;{radius:.6f};0;0;0;"
            f"{int(graph.node_degrees[left])};{int(graph.node_degrees[right])};"
            f"{len(polyline)};{border}"
        )
        vvg_edges.append(
            {
                "id": index,
                "node1": int(left),
                "node2": int(right),
                "skeletonVoxels": [
                    {"pos": [float(p[0]), float(p[1]), float(p[2])]}
                    for p in world_polyline
                ],
            }
        )
    _atomic_write(directory / "edges.csv", "\n".join(edge_rows) + "\n")

    incident: dict[int, list[int]] = {}
    for index, (left, right) in enumerate(graph.edges):
        incident.setdefault(left, []).append(index)
        incident.setdefault(right, []).append(index)
    payload = {
        "graph": {
            "nodes": [
                {
                    "id": index,
                    "edges": incident.get(index, []),
                    "pos": [float(p[0]), float(p[1]), float(p[2])],
                    "radius": float(graph.node_radii[index]),
                    "isAtSampleBorder": bool(at_border[index]),
                }
                for index, p in enumerate(world_nodes)
            ],
            "edges": vvg_edges,
        }
    }
    _atomic_write(directory / "graph.vvg", json.dumps(payload) + "\n")
