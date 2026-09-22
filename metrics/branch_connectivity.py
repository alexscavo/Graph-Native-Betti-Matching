"""Physical-space, degree-2-subdivision-invariant Branch Connectivity F1.

Inputs must already be world coordinates in millimetres. The graph is
contracted only for topology evaluation; each branch retains its polyline.
Pure degree-2 cycles become one virtual centroid anchor and a closed branch.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Branch:
    start: int
    end: int
    polyline: np.ndarray


@dataclass(frozen=True)
class ContractedGraph:
    anchors: dict[int, np.ndarray]
    roles: dict[int, str]
    branches: tuple[Branch, ...]
    betti_0: int
    betti_1: int


def contract_degree_two(nodes: np.ndarray, edges: np.ndarray) -> ContractedGraph:
    positions = np.asarray(nodes, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] not in (2, 3) or not np.isfinite(positions).all():
        raise ValueError("nodes must have finite shape (N,2) or (N,3)")
    links = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if links.size and (links.min() < 0 or links.max() >= len(positions)):
        raise ValueError("edge index outside node array")
    if np.any(links[:, 0] == links[:, 1]):
        raise ValueError("self-loops must be represented by a spatially embedded cycle")
    if len({tuple(sorted(pair)) for pair in links.tolist()}) != len(links):
        raise ValueError("duplicate source edges are ambiguous")
    adjacency = [[] for _ in positions]
    parent = list(range(len(positions)))

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for index, (a, b) in enumerate(links):
        a, b = int(a), int(b)
        adjacency[a].append((b, index))
        adjacency[b].append((a, index))
        parent[root(a)] = root(b)
    beta_0 = len({root(node) for node in range(len(positions))})
    beta_1 = len(links) - len(positions) + beta_0
    anchors = {i: positions[i].copy() for i, neighbors in enumerate(adjacency) if len(neighbors) != 2}
    roles = {i: ("isolated" if not adjacency[i] else "endpoint" if len(adjacency[i]) == 1 else "junction")
             for i in anchors}
    visited: set[int] = set()
    branches = []

    def trace(start: int, neighbor: int, edge_index: int):
        path = [start, neighbor]
        visited.add(edge_index)
        current = neighbor
        while current not in anchors and current != start:
            options = [(node, index) for node, index in adjacency[current] if index not in visited]
            if len(options) != 1:
                raise ValueError("invalid degree-2 chain")
            current, index = options[0]
            visited.add(index)
            path.append(current)
        return path

    for start in sorted(anchors):
        for neighbor, index in sorted(adjacency[start]):
            if index not in visited:
                path = trace(start, neighbor, index)
                branches.append(Branch(start, path[-1], positions[path].copy()))
    # Remaining edges lie in connected components with no degree != 2 node.
    for index, (a, b) in enumerate(links):
        if index in visited:
            continue
        path = trace(int(a), int(b), index)
        if path[-1] != a:
            raise ValueError("degree-2-only component is not a closed cycle")
        curve = positions[path].copy()
        lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        centroid = (np.average((curve[:-1] + curve[1:]) / 2, axis=0, weights=lengths)
                    if lengths.sum() else curve[:-1].mean(axis=0))
        virtual = -(int(min(path)) + 1)
        anchors[virtual] = centroid
        roles[virtual] = "cycle"
        branches.append(Branch(virtual, virtual, curve))
    if len(visited) != len(links):
        raise ValueError("not every graph edge was contracted")
    return ContractedGraph(anchors, roles, tuple(branches), beta_0, beta_1)


def match_anchors(predicted: ContractedGraph, target: ContractedGraph, threshold_mm: float) -> dict[int, int]:
    """Maximum-cardinality, then minimum-distance one-to-one role-aware match."""

    if not np.isfinite(threshold_mm) or threshold_mm < 0:
        raise ValueError("threshold_mm must be finite and non-negative")
    # Isolated detections have no contracted branches, and therefore cannot
    # contribute to a branch match. Skipping them avoids a potentially huge
    # Hungarian matrix for query-based models that retain many isolates.
    pred_ids = sorted(i for i in predicted.anchors if predicted.roles[i] != "isolated")
    gt_ids = sorted(i for i in target.anchors if target.roles[i] != "isolated")
    p, g = len(pred_ids), len(gt_ids)
    if not p or not g:
        return {}
    penalty = threshold_mm + 1.0
    cost = np.full((p + g, p + g), 0.0)
    cost[:p, :g] = penalty * 4
    cost[:p, g:] = penalty
    cost[p:, :g] = penalty
    for i, pred in enumerate(pred_ids):
        for j, gt in enumerate(gt_ids):
            if predicted.roles[pred] == target.roles[gt]:
                distance = float(np.linalg.norm(predicted.anchors[pred] - target.anchors[gt]))
                if distance <= threshold_mm:
                    cost[i, j] = distance
    row, column = linear_sum_assignment(cost)
    return {pred_ids[int(i)]: gt_ids[int(j)] for i, j in zip(row, column)
            if i < p and j < g and cost[i, j] <= threshold_mm}


def branch_connectivity_f1(pred_nodes, pred_edges, gt_nodes, gt_edges, *, threshold_mm: float) -> dict:
    """Count matched contracted branches, without requiring geometric overlap."""

    predicted = contract_degree_two(pred_nodes, pred_edges)
    target = contract_degree_two(gt_nodes, gt_edges)
    matched = match_anchors(predicted, target, threshold_mm)
    reference = Counter(tuple(sorted((branch.start, branch.end))) for branch in target.branches)
    observed = Counter(tuple(sorted((matched[branch.start], matched[branch.end])))
                       for branch in predicted.branches
                       if branch.start in matched and branch.end in matched)
    tp = sum((reference & observed).values())
    fp, fn = len(predicted.branches) - tp, len(target.branches) - tp
    precision = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if not fp else 0.0)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "matched_anchors": len(matched),
        "predicted_branches": len(predicted.branches), "gt_branches": len(target.branches),
        "predicted_betti_0": predicted.betti_0, "predicted_betti_1": predicted.betti_1,
        "gt_betti_0": target.betti_0, "gt_betti_1": target.betti_1,
    }
