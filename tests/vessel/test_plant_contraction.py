import numpy as np
from data.spline_targets import contract_spline_graph, fit_spline


def test_contraction_retains_guide_and_boundary_node():
    nodes=np.array([[0,0,0],[1,0,0],[2,1,0],[3,1,0],[4,1,0]],float)
    edges=np.array([[0,1],[1,2],[2,3],[3,4]])
    graph,reason=contract_spline_graph(nodes,edges,preserve_nodes=[3])
    assert reason is None
    assert set(graph.anchors)=={0,3,4}
    assert len(graph.branches)==2
    assert np.array_equal(graph.branches[0].polyline,nodes[:4])
    controls,curve=fit_spline(graph.branches[0].polyline)
    assert np.allclose(controls[[0,-1]],nodes[[0,3]])


def test_closed_component_not_silently_deleted():
    graph,reason=contract_spline_graph(np.array([[0,0,0],[1,0,0],[1,1,0]]),np.array([[0,1],[1,2],[2,0]]))
    assert reason=='closed_branch'
