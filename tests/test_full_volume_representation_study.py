"""Metric invariance and multi-branch corner cases for T05."""

import numpy as np
import pytest

from scripts.audit_synthetic_mri_grid import CenterlineEdge, SourceGraph
from scripts.study_full_volume_representations import compare_graphs, graph_branches, graph_topology, pair_branches
from scripts.audit_t05_smoothing import length_decomposition
from scripts.validate_t05_report import validate


def graph(nodes, edges):
    positions = {i: np.asarray(point, dtype=float) for i, point in nodes.items()}
    centerlines = [CenterlineEdge(left, right, tuple((positions[left], positions[right])))
                   for left, right in edges]
    return SourceGraph(positions, edges, centerlines)


def test_same_straight_geometry_with_extra_degree_two_node():
    anchor = graph({0: (0, 0, 0), 1: (2, 0, 0)}, [(0, 1)])
    subdivided = graph({0: (0, 0, 0), 1: (2, 0, 0), 2: (1, 0, 0)}, [(0, 2), (2, 1)])
    rows, branch_rows = compare_graphs({"junction_only": anchor, "adaptive": subdivided, "dense": subdivided})
    assert rows[1]["degree_2"] == 1
    assert rows[1]["curve_f1_0.5mm"] == 1
    assert rows[1]["cycle_rank"] == 0
    assert all(record["absolute_length_error_mm"] == 0 for record in branch_rows)


def test_bending_edge_loses_geometry_but_preserves_topology():
    chord = graph({0: (0, 0, 0), 1: (2, 0, 0)}, [(0, 1)])
    bend = graph({0: (0, 0, 0), 1: (2, 0, 0), 2: (1, 1, 0)}, [(0, 2), (2, 1)])
    rows, _ = compare_graphs({"junction_only": chord, "adaptive": bend, "dense": bend}, 0.25)
    assert rows[0]["curve_f1_0.5mm"] < rows[1]["curve_f1_0.5mm"]
    assert rows[0]["curve_f1_0.5mm"] <= rows[0]["curve_f1_1mm"] <= rows[0]["curve_f1_2mm"]
    assert rows[0]["curve_precision_0.5mm"] < 1
    assert rows[0]["curve_recall_0.5mm"] < 1
    assert rows[0]["curve_acd_mm"] > 0
    assert rows[0]["relative_length_error"] < 0
    assert all(row["cycle_rank"] == 0 for row in rows)


def test_parallel_branches_pair_by_trajectory_not_listing_order():
    a = graph({0: (0, 0, 0), 1: (2, 0, 0), 2: (1, 1, 0), 3: (1, -1, 0)},
              [(0, 2), (2, 1), (0, 3), (3, 1)])
    b = graph(a.nodes, [(0, 3), (3, 1), (0, 2), (2, 1)])
    reference = graph_branches(a, {0, 1})
    paired = pair_branches(reference, graph_branches(b, {0, 1}))
    for first, second in zip(reference[(0, 1)], paired[(0, 1)]):
        np.testing.assert_allclose(first, second)
    assert graph_topology(a)["cycle_rank"] == 1


def test_junction_only_self_loop_retained_and_degenerate_ratio_excluded():
    ring = graph({0: (0, 0, 0)}, [(0, 0)])
    expanded = graph({0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0)},
                     [(0, 1), (1, 2), (2, 0)])
    rows, _ = compare_graphs({"junction_only": ring, "adaptive": expanded, "dense": expanded})
    assert rows[0]["cycle_rank"] == 1
    assert rows[0]["closed_or_degenerate_branches"] == 1
    assert np.isnan(rows[0]["length_weighted_tortuosity_error"])


def test_missing_branch_is_not_silently_matched_to_other_vessel():
    a = graph({0: (0, 0, 0), 1: (1, 0, 0)}, [(0, 1)])
    b = graph(a.nodes, [])
    with pytest.raises(ValueError, match="topology changed|multiplicities"):
        compare_graphs({"junction_only": a, "adaptive": b, "dense": a})


def test_length_decomposition_uses_same_raw_reference_denominator():
    result = length_decomposition(10., 8., 7.5)
    assert result["raw_to_smoothed_fraction"] == pytest.approx(-.2)
    assert result["smoothed_to_adaptive_fraction"] == pytest.approx(-.05)
    assert result["raw_to_adaptive_fraction"] == pytest.approx(-.25)


def test_report_validation_rejects_missing_representation():
    with pytest.raises(ValueError, match="incomplete representation family"):
        validate([{"subject": "real_subject", "representation": "adaptive"}])
