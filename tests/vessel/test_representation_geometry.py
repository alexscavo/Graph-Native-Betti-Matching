import numpy as np
import pytest

from metrics.representation_geometry import (
    average_centerline_distance,
    complexity_summary,
    curve_precision_recall_f1,
    hd95,
    point_to_polyline_distance,
    polyline_length,
    sample_polyline_by_arclength,
    tortuosity,
)


def test_point_distance_uses_segments_not_only_vertices():
    line = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    distances = point_to_polyline_distance([[5, 2, 0], [-1, 0, 0], [11, 0, 0]], line)
    np.testing.assert_allclose(distances, [2.0, 1.0, 1.0])


def test_arclength_sampling_includes_endpoints_and_bounds_spacing():
    bent = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    sampled = sample_polyline_by_arclength(bent, spacing=2.0)
    np.testing.assert_allclose(sampled[[0, -1]], bent[[0, -1]])
    assert np.linalg.norm(np.diff(sampled, axis=0), axis=1).max() <= 2.0


def test_curve_metrics_are_physical_and_sampling_density_invariant():
    sparse = [np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])]
    dense_offset = [np.column_stack((np.linspace(0, 10, 101), np.ones(101), np.zeros(101)))]
    scores = curve_precision_recall_f1(sparse, dense_offset, tolerance=1.0, sample_spacing=0.25)
    assert scores == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert average_centerline_distance(sparse, dense_offset, 0.25) == pytest.approx(1.0)
    assert hd95(sparse, dense_offset, 0.25) == pytest.approx(1.0)


def test_curve_precision_and_recall_are_directional():
    reference = [np.array([[0.0, 0.0], [10.0, 0.0]])]
    prediction = [np.array([[0.0, 0.0], [5.0, 0.0]])]
    scores = curve_precision_recall_f1(prediction, reference, 0.01, sample_spacing=1.0)
    assert scores["precision"] == 1.0
    assert scores["recall"] == pytest.approx(6 / 11)
    assert scores["f1"] == pytest.approx(2 * (6 / 11) / (1 + 6 / 11))


def test_length_and_tortuosity_in_physical_coordinates():
    branch = np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [4.0, 3.0, 0.0]])
    assert polyline_length(branch) == 7.0
    assert tortuosity(branch) == pytest.approx(7 / 5)
    assert tortuosity(np.array([[1.0, 1.0], [1.0, 1.0]])) == 1.0
    assert np.isinf(tortuosity(np.array([[0, 0], [1, 0], [0, 0]])))


def test_complexity_summary_counts_degree_and_normalizes_by_physical_length():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [9, 9, 9]])
    edges = [(0, 1), (1, 2)]
    summary = complexity_summary(nodes, edges, [np.array([[0, 0, 0], [4, 0, 0]])])
    assert summary == {
        "nodes": 4,
        "edges": 2,
        "degree_2_nodes": 1,
        "isolated_nodes": 1,
        "reference_length": 4.0,
        "nodes_per_reference_length": 1.0,
        "edges_per_reference_length": 0.5,
    }


@pytest.mark.parametrize("spacing", [0, -1, np.inf])
def test_invalid_sample_spacing_is_rejected(spacing):
    with pytest.raises(ValueError):
        sample_polyline_by_arclength([[0, 0], [1, 0]], spacing)
