"""Exact original Vedo extraction followed by explicit mask-compatible repair."""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass as scipy_center_of_mass, distance_transform_edt
from scipy.spatial import cKDTree
from skimage.graph import route_through_array

from scripts.ixi_vessel_graph import DenseCenterline, VesselGraph, chord_mask_fraction


def _original_vedo(mask: np.ndarray, affine: np.ndarray, work: Path):
    from vedo.core.points import PointAlgorithms
    import vedo

    binary = work / "segmentation.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), binary)
    output = work / "original_output"
    output.mkdir()
    original_apply = PointAlgorithms.apply_transform

    def compatible_affine(obj, transform, *args, **kwargs):
        # Vedo 2026.6.1 rejects a Numpy matrix through a truth-value check.
        # Only wrap the same matrix in its documented transform class.
        if isinstance(transform, np.ndarray):
            transform = vedo.LinearTransform(transform)
        return original_apply(obj, transform, *args, **kwargs)

    PointAlgorithms.apply_transform = compatible_affine
    try:
        spec = importlib.util.spec_from_file_location(
            "unmodified_vedo_extractor_original", Path(__file__).with_name("vedo_extractor_original.py")
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        # The original calls center_of_mass once per orphan label, which
        # rescans the entire volume each time. SciPy accepts all labels in one
        # call and returns the same centroids in label order. Cache that result
        # without changing Vedo's orphan rules or the original source file.
        centroids = None

        def cached_center_of_mass(input_mask, labels, index):
            nonlocal centroids
            if centroids is None:
                centroids = scipy_center_of_mass(
                    input_mask, labels, list(range(1, int(labels.max()) + 1))
                )
            return centroids[int(index) - 1]

        module.center_of_mass = cached_center_of_mass
        module.process_single_case(str(binary), str(output))
    finally:
        PointAlgorithms.apply_transform = original_apply
    with (output / "vessel_data.pkl").open("rb") as stream:
        saved = pickle.load(stream)  # Written by this invocation of the original script.
    graph = saved["graph"]
    ids = sorted(graph.nodes)
    index = {node: position for position, node in enumerate(ids)}
    aligned = np.asarray([graph.nodes[node]["pos"] for node in ids], dtype=float)
    world = nib.affines.apply_affine(np.linalg.inv(module.SLICER_MATRIX), aligned)
    nodes = nib.affines.apply_affine(np.linalg.inv(affine), world)
    edges = np.asarray([(index[a], index[b]) for a, b in graph.edges()], dtype=int).reshape(-1, 2)
    return nodes, edges


def _route_inside_mask(start: np.ndarray, end: np.ndarray, mask: np.ndarray):
    shape = np.asarray(mask.shape)
    for margin in (4, 8, 16, 32):
        low = np.maximum(np.minimum(start, end) - margin, 0)
        high = np.minimum(np.maximum(start, end) + margin + 1, shape)
        local = mask[tuple(slice(int(a), int(b)) for a, b in zip(low, high))]
        cost = np.where(local, 1.0, np.inf)
        try:
            route, _ = route_through_array(cost, tuple(start - low), tuple(end - low),
                                           fully_connected=False, geometric=True)
        except (ValueError, RuntimeError):
            continue
        path = np.asarray(route, dtype=int) + low
        if all(mask[tuple(point)] for point in path):
            return path
    return None


def _graph(nodes: np.ndarray, edges: np.ndarray, radius_map: np.ndarray) -> VesselGraph:
    degrees = np.zeros(len(nodes), dtype=np.int64)
    pairs = [tuple(map(int, pair)) for pair in edges]
    for left, right in pairs:
        degrees[left] += 1
        degrees[right] += 1
    nearest = np.clip(np.rint(nodes).astype(int), 0, np.asarray(radius_map.shape) - 1)
    return VesselGraph(nodes, degrees, pairs, [nodes[[a, b]] for a, b in pairs],
                       radius_map[tuple(nearest.T)], skeleton_voxels=len(nodes))


def extract_original_vedo_centerline(
    segmentation: np.ndarray, *, spacing, affine: np.ndarray | None = None,
    minimum_chord_fraction: float = 0.95,
) -> DenseCenterline:
    """Preserve Vedo topology, repairing only connections incompatible with the mask.

    Nodes outside the segmentation are projected to the nearest foreground
    voxel. Adjacent chords below the requested mask containment are routed along a local
    six-connected foreground path; an edge without such a route is dropped.
    All counts are returned in ``DenseCenterline.provenance``.
    """
    mask = np.asarray(segmentation, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    if affine is None:
        affine = np.diag([*spacing, 1.0])
    affine = np.asarray(affine, dtype=float)
    with tempfile.TemporaryDirectory(prefix="gnbm-vedo-original-") as directory:
        nodes, edges = _original_vedo(mask, affine, Path(directory))
    if not len(nodes) or not len(edges):
        raise ValueError("original Vedo produced an empty centerline")
    original_count = len(nodes)
    original_edges = len(edges)
    nearest = np.rint(nodes).astype(int)
    bounds = np.asarray(mask.shape)
    legal = np.logical_and(nearest >= 0, nearest < bounds).all(axis=1)
    inside = np.zeros(len(nodes), dtype=bool)
    inside[legal] = mask[tuple(nearest[legal].T)]
    outside = np.flatnonzero(~inside)
    projected = nodes.copy()
    if len(outside):
        foreground = np.argwhere(mask)
        if not len(foreground):
            raise ValueError("segmentation has no foreground")
        _, closest = cKDTree(foreground * spacing).query(nodes[outside] * spacing)
        projected[outside] = foreground[closest]
    displacement = np.linalg.norm((projected[outside] - nodes[outside]) * spacing, axis=1)
    radius_map = distance_transform_edt(mask, sampling=spacing)
    node_list = list(projected)
    edge_list: list[tuple[int, int]] = []
    rerouted = inserted = 0
    dropped: list[list[int]] = []
    for raw_a, raw_b in edges:
        a, b = int(raw_a), int(raw_b)
        if chord_mask_fraction(projected[a], projected[b], mask) >= minimum_chord_fraction:
            edge_list.append((a, b))
            continue
        route = _route_inside_mask(np.rint(projected[a]).astype(int),
                                   np.rint(projected[b]).astype(int), mask)
        if route is None:
            dropped.append([a, b])
            continue
        rerouted += 1
        previous = a
        for point in route[1:-1]:
            target = len(node_list)
            node_list.append(point.astype(float))
            edge_list.append((previous, target))
            previous = target
            inserted += 1
        edge_list.append((previous, b))
    repaired_nodes = np.asarray(node_list)
    repaired_edges = np.asarray(edge_list, dtype=int).reshape(-1, 2)
    graph = _graph(repaired_nodes, repaired_edges, radius_map)
    if any(chord_mask_fraction(repaired_nodes[a], repaired_nodes[b], mask) < minimum_chord_fraction
           for a, b in graph.edges):
        raise RuntimeError("Vedo repair left an adjacent edge below the requested mask containment")
    frozen_mask = mask.copy()
    frozen_mask.setflags(write=False)
    return DenseCenterline(
        graph=graph, segmentation=frozen_mask, spacing=spacing,
        provenance={
            "backend": "vedo_original_mask_repaired",
            "repair_minimum_chord_fraction": minimum_chord_fraction,
            "original_nodes": original_count,
            "original_edges": original_edges,
            "projected_nodes": int(len(outside)),
            "projection_distance_median_mm": float(np.median(displacement)) if len(displacement) else 0.0,
            "projection_distance_max_mm": float(np.max(displacement)) if len(displacement) else 0.0,
            "rerouted_original_edges": rerouted,
            "inserted_degree_two_nodes": inserted,
            "dropped_edges_without_local_mask_route": len(dropped),
            "dropped_original_edge_ids": dropped,
        },
    )
