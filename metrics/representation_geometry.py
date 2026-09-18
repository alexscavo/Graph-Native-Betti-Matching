"""Physical-space metrics for compact vascular graph representations.

Polylines are arrays of shape ``(n_points, n_dimensions)``.  Coordinates are
assumed to already be in physical units (normally millimetres); this module
never applies voxel spacing implicitly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


Array = np.ndarray


def _polyline(polyline: Array | Sequence[Sequence[float]]) -> Array:
    points = np.asarray(polyline, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 1:
        raise ValueError("a polyline must have shape (n_points, n_dimensions)")
    if len(points) == 0:
        raise ValueError("a polyline must contain at least one point")
    if not np.isfinite(points).all():
        raise ValueError("polyline coordinates must be finite")
    return points


def _polylines(polylines: Iterable[Array]) -> list[Array]:
    curves = [_polyline(curve) for curve in polylines]
    if not curves:
        raise ValueError("at least one polyline is required")
    dimensions = {curve.shape[1] for curve in curves}
    if len(dimensions) != 1:
        raise ValueError("all polylines must have the same dimensionality")
    return curves


def polyline_length(polyline: Array | Sequence[Sequence[float]]) -> float:
    """Return path length in the coordinate system's physical units."""

    points = _polyline(polyline)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def tortuosity(polyline: Array | Sequence[Sequence[float]]) -> float:
    """Return path length divided by endpoint chord length.

    A zero-length path has tortuosity 1.  A non-zero closed path has infinite
    tortuosity because its endpoint chord has zero length.
    """

    points = _polyline(polyline)
    length = polyline_length(points)
    chord = float(np.linalg.norm(points[-1] - points[0]))
    if length == 0.0:
        return 1.0
    if chord == 0.0:
        return float("inf")
    return length / chord


def sample_polyline_by_arclength(
    polyline: Array | Sequence[Sequence[float]], spacing: float
) -> Array:
    """Sample a polyline uniformly by physical arc length, including endpoints.

    The returned interval is never larger than ``spacing``.  Repeated input
    points are ignored for interpolation.  A zero-length curve yields one
    point.
    """

    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be a positive finite number")
    points = _polyline(polyline)
    if len(points) == 1:
        return points.copy()

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, segment_lengths > 0]
    points = points[keep]
    if len(points) == 1:
        return points.copy()

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    length = cumulative[-1]
    count = max(1, int(np.ceil(length / spacing)))
    targets = np.linspace(0.0, length, count + 1)
    sampled = np.empty((len(targets), points.shape[1]), dtype=np.float64)
    for dimension in range(points.shape[1]):
        sampled[:, dimension] = np.interp(targets, cumulative, points[:, dimension])
    return sampled


def point_to_polyline_distance(
    points: Array | Sequence[Sequence[float]],
    polyline: Array | Sequence[Sequence[float]],
) -> Array:
    """Return exact Euclidean distance from each point to a continuous polyline."""

    query = np.asarray(points, dtype=np.float64)
    if query.ndim == 1:
        query = query[None, :]
    if query.ndim != 2 or len(query) == 0 or not np.isfinite(query).all():
        raise ValueError("points must be a non-empty finite (n, d) array")
    curve = _polyline(polyline)
    if query.shape[1] != curve.shape[1]:
        raise ValueError("points and polyline must have the same dimensionality")
    if len(curve) == 1:
        return np.linalg.norm(query - curve[0], axis=1)

    starts = curve[:-1]
    vectors = curve[1:] - starts
    squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
    valid = squared_lengths > 0
    if not valid.any():
        return np.linalg.norm(query - curve[0], axis=1)
    starts, vectors, squared_lengths = starts[valid], vectors[valid], squared_lengths[valid]

    # Evaluate in query chunks to bound the temporary (queries x segments x d).
    result = np.empty(len(query), dtype=np.float64)
    chunk_size = max(1, 1_000_000 // max(1, len(starts)))
    for offset in range(0, len(query), chunk_size):
        chunk = query[offset : offset + chunk_size]
        relative = chunk[:, None, :] - starts[None, :, :]
        projection = np.einsum("qsd,sd->qs", relative, vectors) / squared_lengths
        projection = np.clip(projection, 0.0, 1.0)
        residual = relative - projection[:, :, None] * vectors[None, :, :]
        result[offset : offset + len(chunk)] = np.sqrt(
            np.einsum("qsd,qsd->qs", residual, residual).min(axis=1)
        )
    return result


def _sample_curves(polylines: Iterable[Array], spacing: float) -> Array:
    curves = _polylines(polylines)
    return np.concatenate([sample_polyline_by_arclength(curve, spacing) for curve in curves])


def _distance_to_curves(points: Array, polylines: Iterable[Array]) -> Array:
    curves = _polylines(polylines)
    return np.min(
        np.stack([point_to_polyline_distance(points, curve) for curve in curves]), axis=0
    )


def directed_curve_distances(
    source: Iterable[Array], target: Iterable[Array], sample_spacing: float
) -> Array:
    """Distances from arc-length samples of ``source`` to continuous ``target``."""

    source_points = _sample_curves(source, sample_spacing)
    return _distance_to_curves(source_points, target)


def curve_precision_recall_f1(
    prediction: Iterable[Array],
    reference: Iterable[Array],
    tolerance: float,
    sample_spacing: float,
) -> dict[str, float]:
    """Compute length-uniform curve precision, recall and F1 in physical space."""

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative finite number")
    prediction = list(prediction)
    reference = list(reference)
    pred_to_ref = directed_curve_distances(prediction, reference, sample_spacing)
    ref_to_pred = directed_curve_distances(reference, prediction, sample_spacing)
    precision = float(np.mean(pred_to_ref <= tolerance))
    recall = float(np.mean(ref_to_pred <= tolerance))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def average_centerline_distance(
    first: Iterable[Array], second: Iterable[Array], sample_spacing: float
) -> float:
    """Return symmetric average centerline distance (ACD)."""

    first = list(first)
    second = list(second)
    forward = directed_curve_distances(first, second, sample_spacing)
    backward = directed_curve_distances(second, first, sample_spacing)
    return float(0.5 * (forward.mean() + backward.mean()))


def hd95(first: Iterable[Array], second: Iterable[Array], sample_spacing: float) -> float:
    """Return the max of the two directed 95th-percentile curve distances."""

    first = list(first)
    second = list(second)
    forward = directed_curve_distances(first, second, sample_spacing)
    backward = directed_curve_distances(second, first, sample_spacing)
    return float(max(np.percentile(forward, 95), np.percentile(backward, 95)))


def complexity_summary(
    nodes: Array | Sequence[Sequence[float]],
    edges: Sequence[Sequence[int]],
    reference_polylines: Iterable[Array] | None = None,
) -> dict[str, float | int]:
    """Summarize representation size and optional length-normalized complexity."""

    node_array = np.asarray(nodes, dtype=np.float64)
    if node_array.ndim != 2 or not np.isfinite(node_array).all():
        raise ValueError("nodes must be a finite (n, d) array")
    edge_array = np.asarray(edges, dtype=np.int64)
    if edge_array.size == 0:
        edge_array = edge_array.reshape(0, 2)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("edges must have shape (n_edges, 2)")
    if edge_array.size and (edge_array.min() < 0 or edge_array.max() >= len(node_array)):
        raise ValueError("edge endpoint index is outside the node array")

    degrees = np.bincount(edge_array.ravel(), minlength=len(node_array))
    result: dict[str, float | int] = {
        "nodes": int(len(node_array)),
        "edges": int(len(edge_array)),
        "degree_2_nodes": int(np.count_nonzero(degrees == 2)),
        "isolated_nodes": int(np.count_nonzero(degrees == 0)),
    }
    if reference_polylines is not None:
        reference_length = sum(polyline_length(curve) for curve in _polylines(reference_polylines))
        result["reference_length"] = float(reference_length)
        result["nodes_per_reference_length"] = (
            float(len(node_array) / reference_length) if reference_length > 0 else float("nan")
        )
        result["edges_per_reference_length"] = (
            float(len(edge_array) / reference_length) if reference_length > 0 else float("nan")
        )
    return result
