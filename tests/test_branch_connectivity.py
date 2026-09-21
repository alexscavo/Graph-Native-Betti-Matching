"""Roadmap Phase 3/4: separate branch connectivity from geometric fidelity."""

import numpy as np
import pytest

from metrics.branch_connectivity import branch_connectivity_f1, contract_degree_two


def evaluate(pred, gt, tol=.25):
    return branch_connectivity_f1(*pred, *gt, threshold_mm=tol)


def test_subdivision_and_curved_trajectory_do_not_change_topology():
    gt = (np.array([[0., 0.], [4., 0.]]), [(0, 1)])
    pred = (np.array([[0., 0.], [4., 0.], [2., 3.]]), [(0, 2), (2, 1)])
    contracted = contract_degree_two(*pred)
    assert len(contracted.branches) == 1
    np.testing.assert_allclose(contracted.branches[0].polyline[1], [2, 3])
    assert evaluate(pred, gt)["f1"] == 1


def test_spatial_threshold_applies_in_physical_millimetres():
    gt = (np.array([[0., 0.], [4., 0.]]), [(0, 1)])
    shifted = (np.array([[.3, 0.], [4.3, 0.]]), [(0, 1)])
    assert evaluate(shifted, gt, tol=.25)["f1"] == 0
    assert evaluate(shifted, gt, tol=.4)["f1"] == 1


def test_wrong_pairing_fails_even_when_betti_numbers_match():
    nodes = np.array([[0., 0.], [0., 1.], [2., 0.], [2., 1.]])
    gt = (nodes, [(0, 2), (1, 3)])
    pred = (nodes, [(0, 1), (2, 3)])
    result = evaluate(pred, gt)
    assert (result["predicted_betti_0"], result["predicted_betti_1"]) == (
        result["gt_betti_0"], result["gt_betti_1"])
    assert (result["tp"], result["fp"], result["fn"]) == (0, 2, 2)


def test_missing_branch_and_extra_separate_component():
    gt = (np.array([[0., 0.], [2., 0.], [10., 0.], [12., 0.]]), [(0, 1), (2, 3)])
    missing = (gt[0], [(0, 1)])
    result = evaluate(missing, gt)
    assert result["tp"] == 1 and result["fn"] == 1
    assert result["precision"] == 1 and result["recall"] == .5
    assert evaluate(gt, missing)["fp"] == 1


def test_crossing_does_not_connect_without_a_graph_node():
    nodes = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
    gt = (nodes, [(0, 1), (2, 3)])
    assert evaluate(gt, gt)["f1"] == 1
    wrong = (np.vstack((nodes, [0., 0.])), [(0, 4), (4, 1), (2, 4), (4, 3)])
    assert evaluate(wrong, gt)["f1"] == 0


def test_degree_two_only_ring_remains_a_one_branch_cycle():
    base = (np.array([[0., 0.], [2., 0.], [1., 2.]]), [(0, 1), (1, 2), (2, 0)])
    divided = (np.array([[0., 0.], [2., 0.], [1., 2.], [1., 0.]]),
               [(0, 3), (3, 1), (1, 2), (2, 0)])
    result = evaluate(divided, base)
    assert result["f1"] == 1 and result["gt_betti_1"] == 1
    assert result["gt_branches"] == 1


def test_empty_graphs_have_explicit_perfect_empty_convention():
    empty = (np.empty((0, 3)), [])
    assert evaluate(empty, empty)["f1"] == 1
    edge = (np.array([[0., 0., 0.], [2., 0., 0.]]), [(0, 1)])
    assert evaluate(empty, edge)["f1"] == 0


def test_invalid_graphs_and_thresholds_are_rejected():
    graph = (np.array([[0., 0.], [1., 0.]]), [(0, 1)])
    with pytest.raises(ValueError, match="threshold"):
        evaluate(graph, graph, tol=-1)
    with pytest.raises(ValueError, match="duplicate"):
        contract_degree_two(graph[0], [(0, 1), (1, 0)])
