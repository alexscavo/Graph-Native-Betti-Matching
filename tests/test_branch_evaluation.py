import csv

import numpy as np
import pytest

from metrics.branch_evaluation import (
    evaluate_maximal_branch,
    plot_branch_tradeoffs,
    summarize_branches,
    turning_curvature_proxy,
    write_branch_metrics_csv,
)


def test_turning_proxy_distinguishes_straight_and_right_angle():
    straight = np.array([[0, 0], [1, 0], [3, 0]], dtype=float)
    bent = np.array([[0, 0], [1, 0], [1, 1]], dtype=float)
    assert turning_curvature_proxy(straight)["total_turning_radians"] == 0.0
    assert turning_curvature_proxy(bent)["total_turning_radians"] == pytest.approx(np.pi / 2)
    assert turning_curvature_proxy(bent)["turning_radians_per_length"] == pytest.approx(np.pi / 4)


def test_branch_metrics_capture_compression_geometry_radius_and_containment():
    dense = np.array([[0, 0], [1, 1], [2, 0], [3, 0]], dtype=float)
    simplified = np.array([[0, 0], [2, 0], [3, 0]], dtype=float)
    result = evaluate_maximal_branch(
        dense,
        simplified,
        branch_id="b0",
        dense_radii=[0.5, 0.75, 1.0, 1.25],
        containment=lambda points: points[:, 0] <= 2.5,
        containment_spacing=0.5,
    )
    assert result["branch_id"] == "b0"
    assert result["retained_degree2_nodes"] == 1
    assert result["degree2_retention_fraction"] == 0.5
    assert result["max_approximation_error"] == pytest.approx(1.0)
    assert result["median_radius"] == pytest.approx(0.875)
    assert result["max_error_over_median_radius"] == pytest.approx(1 / 0.875)
    assert result["containment_fraction"] < 1.0
    assert result["fully_contained"] is False


def test_endpoints_must_match_to_define_same_maximal_branch():
    with pytest.raises(ValueError, match="endpoints"):
        evaluate_maximal_branch(
            np.array([[0, 0], [1, 0], [2, 0]]),
            np.array([[0, 0], [1, 0]]),
        )


def test_summary_is_length_weighted_and_counts_containment():
    short = evaluate_maximal_branch(
        np.array([[0, 0], [1, 0]]),
        np.array([[0, 0], [1, 0]]),
        containment=lambda points: np.ones(len(points), dtype=bool),
    )
    long = evaluate_maximal_branch(
        np.array([[0, 0], [0, 3], [4, 3]]),
        np.array([[0, 0], [4, 3]]),
        containment=lambda points: np.zeros(len(points), dtype=bool),
    )
    summary = summarize_branches([short, long])
    assert summary["branches"] == 2
    assert summary["total_dense_length"] == pytest.approx(8.0)
    assert summary["fully_contained_branches"] == 1
    assert summary["mean_containment_fraction"] == 0.5
    expected = (1 * short["relative_length_error"] + 7 * long["relative_length_error"]) / 8
    assert summary["length_weighted_relative_length_error"] == pytest.approx(expected)


def test_csv_and_plot_helpers_create_inspectable_artifacts(tmp_path):
    record = evaluate_maximal_branch(
        np.array([[0, 0], [1, 0], [2, 0]], dtype=float),
        np.array([[0, 0], [2, 0]], dtype=float),
        dense_radii=[1, 1, 1],
        containment=lambda points: np.ones(len(points), dtype=bool),
    )
    csv_path = tmp_path / "branches.csv"
    png_path = tmp_path / "tradeoffs.png"
    write_branch_metrics_csv([record], csv_path)
    plot_branch_tradeoffs([record], png_path)
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["retained_degree2_nodes"] == "0"
    assert png_path.stat().st_size > 1_000
