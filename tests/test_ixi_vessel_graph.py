"""Regression tests for the IXI segmentation-to-graph extractor."""

from __future__ import annotations

import numpy as np

from scripts.audit_synthetic_mri_grid import discover_sources
from scripts.ixi_vessel_graph import build_representation_family, build_vessel_graph
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
    graphs = build_representation_family(
        _square_ring_mask(),
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
