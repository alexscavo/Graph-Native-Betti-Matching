import json
from pathlib import Path

import numpy as np

from scripts.compare_dense_centerlines import betti, containment, load_vvg, measure, resample_polyline, turn_angles


def write_vvg(path: Path, points, nodes=None, edge_nodes=(0, 1)):
    nodes = nodes or [{"id": 0}, {"id": 1}]
    path.write_text(json.dumps({"graph": {"nodes": nodes, "edges": [{
        "id": 0, "node1": edge_nodes[0], "node2": edge_nodes[1],
        "skeletonVoxels": [{"pos": point} for point in points],
    }]}}))


def test_containment_checks_between_stored_samples(tmp_path):
    # Endpoints lie in foreground, but the straight path between them does not.
    mask = np.zeros((7, 7, 7), dtype=bool)
    mask[1, 3, 3] = mask[5, 3, 3] = True
    line = resample_polyline(np.array([[1., 3., 3.], [5., 3., 3.]]), 0.5)
    inside = containment(line, mask, np.eye(4))
    assert inside[0] and inside[-1]
    assert inside.mean() < 0.5


def test_angles_and_betti():
    angles = turn_angles([np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])])
    assert np.allclose(angles, [90.])
    nodes = [{"id": i} for i in range(3)]
    edges = [{"node1": 0, "node2": 1}, {"node1": 1, "node2": 2}, {"node1": 2, "node2": 0}]
    assert betti(nodes, edges) == (1, 1)


def test_measure_vvg(tmp_path):
    path = tmp_path / "graph.vvg"
    write_vvg(path, [[1, 2, 2], [2, 2, 2], [3, 2, 2]])
    mask = np.ones((5, 5, 5), dtype=bool)
    result, graph = measure("candidate", path, mask, np.eye(4), 0.25)
    assert result["containment_fraction"] == 1.0
    assert result["stored_sample_count"] == 3
    assert result["betti_0"] == 1 and result["betti_1"] == 0
    assert len(load_vvg(path)["polylines"]) == len(graph["polylines"]) == 1
