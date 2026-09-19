"""Regression tests for the IXI segmentation-to-graph extractor."""

from __future__ import annotations

import numpy as np
from unittest.mock import patch

from scripts.audit_synthetic_mri_grid import SourceGraph, discover_sources, world_to_voxel
from scripts.ixi_vessel_graph import (
    VesselGraph,
    adaptive_tolerance_profile,
    build_representation_family,
    build_vessel_graph,
    optimal_keep_mask,
    rdp_keep_mask,
    write_source_graph,
)
from scripts.prepare_ixi_sources import segmentation_path


def _square_ring_mask() -> np.ndarray:
    mask = np.zeros((5, 30, 30), dtype=np.uint8)
    mask[2, 5, 5:25] = 1
    mask[2, 24, 5:25] = 1
    mask[2, 5:25, 5] = 1
    mask[2, 5:25, 24] = 1
    return mask


def test_pure_degree_two_ring_is_preserved_as_simple_cycle():
    graph = build_vessel_graph(
        _square_ring_mask(),
        spur_length=0,
        smooth_iterations=0,
    )

    assert graph.ring_count == 1
    assert graph.node_count == 3
    assert graph.edge_count == 3
    assert graph.betti() == (1, 1)
    assert np.all(graph.node_degrees == 2)
    assert len({tuple(sorted(edge)) for edge in graph.edges}) == 3
    assert all(left != right for left, right in graph.edges)


def test_adaptive_subdivision_preserves_ring_topology():
    graph = build_vessel_graph(
        _square_ring_mask(),
        spacing=(0.5, 0.5, 0.8),
        spur_length=0,
        smooth_iterations=0,
        intermediate_nodes=True,
        rdp_tolerance_mm=0.1,
    )

    assert graph.ring_count == 1
    assert graph.node_count > 3
    assert graph.betti() == (1, 1)
    assert np.all(graph.node_degrees == 2)


def test_documented_preferred_segmentation_is_selected(tmp_path):
    segmentations = tmp_path / "segmentations"
    segmentations.mkdir()
    canonical = segmentations / "IXI638-HH-2786-MRA.nii.gz"
    preferred = segmentations / "IXI638-HH-2786-MRA.nii(1).gz"
    canonical.touch()
    preferred.touch()

    assert segmentation_path(tmp_path, "IXI638-HH-2786") == preferred


def test_canonical_segmentation_is_selected_for_other_subjects(tmp_path):
    segmentations = tmp_path / "segmentations"
    segmentations.mkdir()
    canonical = segmentations / "IXI002-Guys-0828-MRA.nii.gz"
    canonical.touch()

    assert segmentation_path(tmp_path, "IXI002-Guys-0828") == canonical


def test_representation_family_differs_only_by_degree_two_sampling():
    graphs = build_representation_family(
        _square_ring_mask(),
        spacing=(0.5, 0.5, 0.8),
        adaptive_tolerance_mm=0.8,
        spur_length=0,
        smooth_iterations=0,
    )

    assert set(graphs) == {"junction_only", "adaptive", "dense"}
    assert graphs["junction_only"].node_count < graphs["adaptive"].node_count
    assert graphs["adaptive"].node_count < graphs["dense"].node_count
    assert {graph.betti() for graph in graphs.values()} == {(1, 1)}
    assert {
        int((graph.node_degrees != 2).sum()) for graph in graphs.values()
    } == {0}


def test_source_discovery_defaults_to_adaptive_representation(tmp_path):
    for directory in ("raw", "seg", "graphs/adaptive/7", "graphs/dense/7"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw/7.nii.gz").touch()
    (tmp_path / "seg/7.nii.gz").touch()

    sources = discover_sources(tmp_path)

    assert sources["7"][2] == tmp_path / "graphs/adaptive/7"


def test_dense_reference_precedes_smoothing_and_simplification():
    from scipy.ndimage import binary_dilation

    # Give the sub-voxel curve room to move while remaining inside the lumen.
    mask = binary_dilation(_square_ring_mask(), iterations=2)
    graphs = build_representation_family(
        mask,
        spacing=(0.5, 0.5, 0.8),
        adaptive_tolerance_mm=0.8,
        spur_length=0,
        smooth_iterations=5,
        smooth_alpha=0.5,
    )

    dense_points = np.concatenate(graphs["dense"].centerlines)
    adaptive_points = np.concatenate(graphs["adaptive"].centerlines)
    assert np.allclose(dense_points, np.rint(dense_points))
    assert np.any(np.abs(adaptive_points - np.rint(adaptive_points)) > 1e-6)


def test_smoothing_safely_falls_back_in_a_one_voxel_lumen():
    graphs = build_representation_family(
        _square_ring_mask(),
        spacing=(0.5, 0.5, 0.8),
        adaptive_tolerance_mm=0.8,
        spur_length=0,
        smooth_iterations=5,
        smooth_alpha=0.5,
    )

    adaptive_points = np.concatenate(graphs["adaptive"].centerlines)
    assert np.allclose(adaptive_points, np.rint(adaptive_points))


def test_representation_family_skeletonizes_only_once():
    from skimage.morphology import skeletonize as real_skeletonize

    with patch("skimage.morphology.skeletonize", wraps=real_skeletonize) as mocked:
        build_representation_family(
            _square_ring_mask(),
            spacing=(0.5, 0.5, 0.8),
            adaptive_tolerance_mm=0.8,
            spur_length=0,
        )

    assert mocked.call_count == 1


def test_radius_aware_tolerance_is_not_capped_by_fixed_global_tolerance():
    thin = adaptive_tolerance_profile(
        np.full(5, 0.5), radius_fraction=0.8, minimum_tolerance_mm=0.1
    )
    thick = adaptive_tolerance_profile(
        np.full(5, 3.0), radius_fraction=0.8, minimum_tolerance_mm=0.1
    )

    np.testing.assert_allclose(thin, 0.4)
    np.testing.assert_allclose(thick, 2.4)
    # The former global value (0.8 mm) must not cap a 2.4-mm thick-vessel rule.
    assert thick.min() > 0.8


def test_radius_aware_rdp_retains_more_geometry_in_thin_than_thick_vessels():
    polyline = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.6, 0.0), (2.0, 0.0, 0.0),
         (3.0, 0.6, 0.0), (4.0, 0.0, 0.0))
    )
    thin_limit = adaptive_tolerance_profile(
        np.full(len(polyline), 0.5), radius_fraction=0.8
    )
    thick_limit = adaptive_tolerance_profile(
        np.full(len(polyline), 3.0), radius_fraction=0.8
    )

    thin_keep = rdp_keep_mask(polyline, thin_limit)
    thick_keep = rdp_keep_mask(polyline, thick_limit)

    assert thin_keep.sum() > thick_keep.sum()
    np.testing.assert_array_equal(thick_keep, (True, False, False, False, True))


def test_optimal_simplifier_uses_fewer_nodes_than_greedy_rdp_when_possible():
    polyline = np.asarray(
        ((1.0, 0.57551092, -0.58791274),
         (2.0, 0.09683346, -0.70991223),
         (3.0, 0.54560930, -0.19933186),
         (4.0, 0.11979837, -0.56365017),
         (5.0, -0.35889538, -0.91419311),
         (6.0, -0.50164940, -0.99328213))
    )

    greedy = rdp_keep_mask(polyline, 0.5)
    optimal = optimal_keep_mask(polyline, 0.5)

    assert greedy.sum() == 4
    assert optimal.sum() == 3


def test_optimal_simplifier_node_count_is_monotonic_with_tolerance():
    polyline = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.5, 0.0), (2.0, -0.4, 0.0),
         (3.0, 0.7, 0.0), (4.0, -0.2, 0.0), (5.0, 0.0, 0.0))
    )

    strict = optimal_keep_mask(polyline, 0.25)
    relaxed = optimal_keep_mask(polyline, 0.75)

    assert relaxed.sum() <= strict.sum()


def test_optimal_simplifier_measures_distance_to_finite_chord():
    # All points lie on the same infinite line, but the middle point overshoots
    # the finite start/end chord and must therefore remain at a tight tolerance.
    polyline = np.asarray(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    )

    keep = optimal_keep_mask(polyline, 0.1)

    np.testing.assert_array_equal(keep, (True, True, True))


def test_optimal_simplifier_enforces_lumen_containment():
    voxel_polyline = np.asarray(
        ((2.0, 2.0, 0.0), (2.0, 6.0, 0.0), (6.0, 6.0, 0.0),
         (6.0, 2.0, 0.0))
    )
    mask = np.zeros((9, 9, 3), dtype=bool)
    for left, right in zip(voxel_polyline[:-1], voxel_polyline[1:]):
        samples = left + np.linspace(0.0, 1.0, 20)[:, None] * (right - left)
        indices = np.rint(samples).astype(int)
        mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    unconstrained = optimal_keep_mask(voxel_polyline, 10.0)
    contained = optimal_keep_mask(
        voxel_polyline,
        10.0,
        voxel_polyline=voxel_polyline,
        mask=mask,
        minimum_chord_fraction=1.0,
    )

    assert unconstrained.sum() == 2
    assert contained.sum() > unconstrained.sum()


def test_optimal_representation_family_preserves_topology_and_never_exceeds_rdp():
    shared = dict(
        segmentation=_square_ring_mask(),
        spacing=(0.5, 0.5, 0.8),
        adaptive_tolerance_mm=0.8,
        spur_length=0,
        smooth_iterations=0,
        minimum_chord_fraction=1.0,
    )

    greedy = build_representation_family(**shared, simplification_method="rdp")
    optimal = build_representation_family(**shared, simplification_method="optimal")

    assert optimal["adaptive"].betti() == greedy["adaptive"].betti() == (1, 1)
    assert optimal["adaptive"].node_count <= greedy["adaptive"].node_count


def test_source_graph_node_serialization_preserves_voxel_cell_side(tmp_path):
    affine = np.asarray(
        ((-0.46875, 0.0, 0.0, 64.511681234567),
         (0.0, 0.46875, 0.0, -63.292368765432),
         (0.0, 0.0, 0.421, 4.690878123456),
         (0.0, 0.0, 0.0, 1.0))
    )
    positions = np.asarray(((50.75, 211.25, 34.5000003), (51.0, 212.0, 34.0)))
    graph = VesselGraph(
        node_positions=positions,
        node_degrees=np.asarray((1, 1)),
        edges=[(0, 1)],
        centerlines=[positions.copy()],
        node_radii=np.ones(2),
    )

    write_source_graph(tmp_path, graph, affine, (100, 250, 100))
    reloaded = SourceGraph.from_directory(tmp_path)
    round_trip = world_to_voxel(reloaded.node_positions, affine)

    np.testing.assert_allclose(round_trip, positions, atol=1e-9, rtol=0.0)
    np.testing.assert_array_equal(
        np.floor(round_trip + 0.5).astype(int),
        np.floor(positions + 0.5).astype(int),
    )
