import numpy as np
import pytest

from metrics.crop_comparison import (
    boundary_correspondence,
    compare_cropped_graphs,
    distance_to_box_boundary,
    graph_topology_summary,
)


BOUNDS = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])


def test_distance_to_box_boundary_is_physical_and_marks_outside_points():
    result = distance_to_box_boundary(
        np.array([[5, 5, 5], [0, 4, 4], [9.5, 2, 2], [11, 2, 2]], dtype=float),
        BOUNDS,
    )
    np.testing.assert_allclose(result, [5.0, 0.0, 0.5, -1.0])


def test_topology_summary_counts_isolates_components_and_cycles():
    positions = np.zeros((5, 3))
    edges = np.array([[0, 1], [1, 2], [2, 0], [3, 3]])
    summary = graph_topology_summary(positions, edges)
    assert summary == {
        "nodes": 5,
        "edges": 4,
        "components": 3,
        "cycle_rank": 2,
        "isolated_nodes": 1,
        "terminal_nodes": 0,
        "degree_2_nodes": 4,
        "junction_nodes": 0,
    }


def test_boundary_correspondence_measures_last_inside_sample_gap():
    exact_positions = np.array([[5, 5, 5], [10, 5, 5]], dtype=float)
    inherited_positions = np.array([[5, 5, 5], [9, 5, 5]], dtype=float)
    edge = np.array([[0, 1]])
    result = boundary_correspondence(
        exact_positions,
        edge,
        inherited_positions,
        edge,
        BOUNDS,
        correspondence_tolerance=1.5,
    )
    assert result["exact_boundary_contacts"] == 1
    assert result["matched_boundary_contacts"] == 1
    assert result["boundary_recall"] == 1.0
    assert result["mean_correspondence_distance"] == pytest.approx(1.0)
    assert result["mean_matched_inherited_face_gap"] == pytest.approx(1.0)
    # Both graph endpoints are candidates: their mean face gap is (5 + 1) / 2.
    assert result["mean_inherited_terminal_face_gap"] == pytest.approx(3.0)


def test_missing_crossing_is_visible_in_boundary_and_topology_metrics():
    exact_positions = np.array([[0, 5, 5], [10, 5, 5]], dtype=float)
    exact_edges = np.array([[0, 1]])
    inherited_positions = np.empty((0, 3))
    inherited_edges = np.empty((0, 2), dtype=int)
    report = compare_cropped_graphs(
        exact_positions,
        exact_edges,
        inherited_positions,
        inherited_edges,
        BOUNDS,
    )
    assert report["boundary"]["missing_boundary_contacts"] == 2
    assert report["topology_delta_inherited_minus_exact"]["edges"] == -1
    assert np.isnan(report["geometry"]["curve_f1"])


def test_complete_comparison_reports_geometry_topology_and_budgets():
    exact_positions = np.array([[0, 5, 5], [5, 5, 5], [10, 5, 5]], dtype=float)
    inherited_positions = np.array([[1, 5, 5], [5, 5, 5], [9, 5, 5]], dtype=float)
    edges = np.array([[0, 1], [1, 2]])
    report = compare_cropped_graphs(
        exact_positions,
        edges,
        inherited_positions,
        edges,
        BOUNDS,
        sample_spacing=0.25,
        curve_tolerance=0.1,
        correspondence_tolerance=1.1,
        preferred_edge_budget=1,
        hard_token_budget=2,
    )
    assert report["boundary"]["matched_boundary_contacts"] == 2
    assert report["geometry"]["average_centerline_distance"] > 0
    assert report["exact_topology"]["cycle_rank"] == 0
    assert report["topology_delta_inherited_minus_exact"]["components"] == 0
    assert report["tokens"]["exact_exceeds_preferred_edges"] is True
    assert report["tokens"]["inherited_exceeds_hard_node_tokens"] is True
