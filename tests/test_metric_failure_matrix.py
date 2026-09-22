"""Roadmap Phase 4: topology and physical geometry must not be conflated."""

import numpy as np

from metrics.branch_connectivity import branch_connectivity_f1
from metrics.representation_geometry import (
    average_centerline_distance, curve_precision_recall_f1, hd95,
)


def describe(predicted, reference, *, tolerance=.25):
    pred_nodes, pred_edges = predicted
    gt_nodes, gt_edges = reference
    pred_curves = [pred_nodes[[a, b]] for a, b in pred_edges]
    gt_curves = [gt_nodes[[a, b]] for a, b in gt_edges]
    score = branch_connectivity_f1(pred_nodes, pred_edges, gt_nodes, gt_edges,
                                   threshold_mm=tolerance)
    score["curve_f1"] = curve_precision_recall_f1(pred_curves, gt_curves, tolerance, .1)["f1"]
    score["acd_mm"] = average_centerline_distance(pred_curves, gt_curves, .1)
    score["hd95_mm"] = hd95(pred_curves, gt_curves, .1)
    return score


def test_extra_degree_two_nodes_and_shifted_interior_geometry():
    gt = (np.array([[0., 0.], [4., 0.]]), [(0, 1)])
    divided = (np.array([[0., 0.], [4., 0.], [2., 0.]]), [(0, 2), (2, 1)])
    curved = (np.array([[0., 0.], [4., 0.], [2., 2.]]), [(0, 2), (2, 1)])
    equal = describe(divided, gt)
    shifted = describe(curved, gt)
    assert equal["f1"] == equal["curve_f1"] == 1
    assert equal["acd_mm"] < 1e-8 and equal["hd95_mm"] < 1e-8
    assert len(divided[0]) > len(gt[0])
    assert shifted["f1"] == 1  # Same anchor connectivity despite geometry change.
    assert shifted["curve_f1"] < 1 and shifted["acd_mm"] > .25 and shifted["hd95_mm"] > .25


def test_small_break_can_escape_geometry_tolerance_but_not_topology():
    gt = (np.array([[0., 0.], [4., 0.]]), [(0, 1)])
    broken = (np.array([[0., 0.], [1.9, 0.], [2.1, 0.], [4., 0.]]), [(0, 1), (2, 3)])
    score = describe(broken, gt)
    assert score["f1"] < 1
    assert score["curve_f1"] == 1
    assert score["predicted_betti_0"] != score["gt_betti_0"]


def test_false_nearby_bridge_can_escape_geometry_and_betti_global_count():
    nodes = np.array([[0., 0.], [4., 0.], [0., .4], [4., .4]])
    gt = (nodes, [(0, 1), (2, 3)])
    bridged = (nodes, [(0, 1), (2, 3), (0, 2)])
    score = describe(bridged, gt, tolerance=.5)
    assert score["f1"] < 1
    assert score["curve_f1"] == 1  # Added bridge lies within geometry tolerance.


def test_same_betti_numbers_can_hide_wrong_branch_pairing():
    nodes = np.array([[0., 0.], [4., 0.], [0., 3.], [4., 3.]])
    gt = (nodes, [(0, 1), (2, 3)])
    wrong = (nodes, [(0, 2), (1, 3)])
    score = describe(wrong, gt)
    assert score["f1"] == 0 and score["curve_f1"] < 1
    assert (score["predicted_betti_0"], score["predicted_betti_1"]) == (
        score["gt_betti_0"], score["gt_betti_1"])
