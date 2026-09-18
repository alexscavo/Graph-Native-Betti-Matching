"""Branch-level physical metrics for tuning adaptive graph simplification.

Each input pair represents one maximal branch between topological anchors.  The
dense polyline is the immutable reference and the simplified polyline contains
the retained graph control nodes. Coordinates and radii must be in physical
units (normally millimetres).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import csv
from pathlib import Path

import numpy as np

from metrics.representation_geometry import (
    point_to_polyline_distance,
    polyline_length,
    sample_polyline_by_arclength,
    tortuosity,
)


def _points(value: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] < 2 or len(result) < 2:
        raise ValueError(f"{name} must have shape (at least 2, dimensions >= 2)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite coordinates")
    return result


def turning_curvature_proxy(polyline: np.ndarray) -> dict[str, float]:
    """Return discrete turning statistics, including total radians per length.

    Zero-length segments are ignored. This is a stable tuning proxy, not a
    differential curvature estimator.
    """

    points = _points(polyline, "polyline")
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    vectors = vectors[lengths > 0]
    lengths = lengths[lengths > 0]
    if len(vectors) < 2:
        return {
            "total_turning_radians": 0.0,
            "mean_turning_radians": 0.0,
            "turning_radians_per_length": 0.0,
        }
    unit = vectors / lengths[:, None]
    cosine = np.clip(np.einsum("ij,ij->i", unit[:-1], unit[1:]), -1.0, 1.0)
    angles = np.arccos(cosine)
    total = float(angles.sum())
    length = float(lengths.sum())
    return {
        "total_turning_radians": total,
        "mean_turning_radians": float(angles.mean()),
        "turning_radians_per_length": total / length if length else 0.0,
    }


def evaluate_maximal_branch(
    dense_polyline: np.ndarray,
    simplified_polyline: np.ndarray,
    *,
    branch_id: str | int | None = None,
    dense_radii: np.ndarray | Sequence[float] | None = None,
    containment: Callable[[np.ndarray], np.ndarray] | None = None,
    containment_spacing: float = 0.25,
    approximation_spacing: float = 0.25,
) -> dict[str, float | int | str | None]:
    """Evaluate one simplified maximal branch against its dense reference."""

    dense = _points(dense_polyline, "dense_polyline")
    simplified = _points(simplified_polyline, "simplified_polyline")
    if dense.shape[1] != simplified.shape[1]:
        raise ValueError("dense and simplified polylines must have equal dimensionality")
    if not np.allclose(dense[[0, -1]], simplified[[0, -1]], atol=1e-7, rtol=0):
        raise ValueError("dense and simplified branch endpoints must match")

    dense_length = polyline_length(dense)
    simplified_length = polyline_length(simplified)
    # Include original vertices so sharp dense-reference extrema cannot fall
    # between uniform samples, and uniform samples so long sparse segments are
    # not underweighted.
    approximation_samples = np.concatenate(
        (dense, sample_polyline_by_arclength(dense, approximation_spacing)), axis=0
    )
    errors = point_to_polyline_distance(approximation_samples, simplified)
    dense_turning = turning_curvature_proxy(dense)
    simplified_turning = turning_curvature_proxy(simplified)
    dense_tortuosity = tortuosity(dense)
    simplified_tortuosity = tortuosity(simplified)
    dense_interior = max(0, len(dense) - 2)
    retained_degree2 = max(0, len(simplified) - 2)

    result: dict[str, float | int | str | None] = {
        "branch_id": branch_id,
        "dense_points": int(len(dense)),
        "simplified_points": int(len(simplified)),
        "retained_degree2_nodes": int(retained_degree2),
        "degree2_retention_fraction": (
            float(retained_degree2 / dense_interior) if dense_interior else 0.0
        ),
        "compression_ratio": float(len(dense) / len(simplified)),
        "dense_length": dense_length,
        "simplified_length": simplified_length,
        "relative_length_error": (
            abs(simplified_length - dense_length) / dense_length if dense_length else 0.0
        ),
        "dense_tortuosity": dense_tortuosity,
        "simplified_tortuosity": simplified_tortuosity,
        "absolute_tortuosity_error": abs(simplified_tortuosity - dense_tortuosity),
        "dense_turning_radians": dense_turning["total_turning_radians"],
        "simplified_turning_radians": simplified_turning["total_turning_radians"],
        "dense_turning_per_length": dense_turning["turning_radians_per_length"],
        "simplified_turning_per_length": simplified_turning["turning_radians_per_length"],
        "absolute_turning_per_length_error": abs(
            simplified_turning["turning_radians_per_length"]
            - dense_turning["turning_radians_per_length"]
        ),
        "mean_approximation_error": float(errors.mean()),
        "p95_approximation_error": float(np.percentile(errors, 95)),
        "max_approximation_error": float(errors.max()),
    }

    if dense_radii is not None:
        radii = np.asarray(dense_radii, dtype=np.float64)
        if radii.shape != (len(dense),) or not np.isfinite(radii).all() or np.any(radii < 0):
            raise ValueError("dense_radii must be one finite non-negative value per dense point")
        positive = radii[radii > 0]
        result.update(
            mean_radius=float(radii.mean()),
            median_radius=float(np.median(radii)),
            minimum_radius=float(radii.min()),
            maximum_radius=float(radii.max()),
            max_error_over_median_radius=(
                float(errors.max() / np.median(positive)) if len(positive) else float("nan")
            ),
        )
    else:
        result.update(
            mean_radius=float("nan"),
            median_radius=float("nan"),
            minimum_radius=float("nan"),
            maximum_radius=float("nan"),
            max_error_over_median_radius=float("nan"),
        )

    if containment is not None:
        samples = sample_polyline_by_arclength(simplified, containment_spacing)
        inside = np.asarray(containment(samples), dtype=bool)
        if inside.shape != (len(samples),):
            raise ValueError("containment callback must return one boolean per sample")
        result.update(
            containment_samples=int(len(samples)),
            containment_fraction=float(inside.mean()),
            outside_samples=int(np.count_nonzero(~inside)),
            fully_contained=bool(inside.all()),
        )
    else:
        result.update(
            containment_samples=0,
            containment_fraction=float("nan"),
            outside_samples=0,
            fully_contained=None,
        )
    return result


def summarize_branches(
    records: Iterable[Mapping[str, object]],
) -> dict[str, float | int]:
    """Aggregate branch records with length weighting where scientifically useful."""

    rows = list(records)
    if not rows:
        raise ValueError("at least one branch record is required")

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    lengths = values("dense_length")
    total_length = float(lengths.sum())
    weights = lengths / total_length if total_length > 0 else np.full(len(rows), 1 / len(rows))
    containment = values("containment_fraction")
    valid_containment = np.isfinite(containment)
    summary: dict[str, float | int] = {
        "branches": len(rows),
        "total_dense_length": total_length,
        "total_retained_degree2_nodes": int(sum(int(row["retained_degree2_nodes"]) for row in rows)),
        "median_retained_degree2_nodes": float(np.median(values("retained_degree2_nodes"))),
        "median_compression_ratio": float(np.median(values("compression_ratio"))),
        "length_weighted_relative_length_error": float(np.sum(weights * values("relative_length_error"))),
        "length_weighted_mean_approximation_error": float(np.sum(weights * values("mean_approximation_error"))),
        "p95_branch_max_approximation_error": float(np.percentile(values("max_approximation_error"), 95)),
        "maximum_approximation_error": float(values("max_approximation_error").max()),
        "fully_contained_branches": int(sum(row.get("fully_contained") is True for row in rows)),
    }
    summary["mean_containment_fraction"] = (
        float(containment[valid_containment].mean()) if valid_containment.any() else float("nan")
    )
    return summary


def write_branch_metrics_csv(records: Iterable[Mapping[str, object]], output: Path) -> None:
    """Write branch records with a stable union of fields."""

    rows = list(records)
    if not rows:
        raise ValueError("at least one branch record is required")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_branch_tradeoffs(
    records: Iterable[Mapping[str, object]], output: Path, *, title: str = "Branch simplification trade-offs"
) -> None:
    """Plot error/containment and node cost versus branch radius and curvature."""

    rows = list(records)
    if not rows:
        raise ValueError("at least one branch record is required")
    import matplotlib.pyplot as plt

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    radius = values("median_radius")
    curvature = values("dense_turning_per_length")
    nodes = values("retained_degree2_nodes")
    error = values("max_approximation_error")
    containment = values("containment_fraction")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    scatter = axes[0].scatter(radius, error, c=nodes, cmap="viridis", s=24)
    axes[0].set(xlabel="Median radius", ylabel="Maximum approximation error")
    figure.colorbar(scatter, ax=axes[0], label="Retained degree-2 nodes")
    axes[1].scatter(curvature, nodes, c=error, cmap="magma", s=24)
    axes[1].set(xlabel="Dense turning / length", ylabel="Retained degree-2 nodes")
    axes[2].scatter(nodes, containment, c=radius, cmap="plasma", s=24)
    axes[2].set(xlabel="Retained degree-2 nodes", ylabel="Containment fraction", ylim=(-0.02, 1.02))
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(title)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
