from metrics.vessel_graph import VesselGraph
from scripts.diagnose_branch_relations import diagnose, summarize


def graph(nodes, edges):
    return VesselGraph.from_arrays(nodes, edges)


def test_near_false_connections_are_distinguished_from_long_true_relations():
    nodes = [[0, 0, 0], [0, 1, 0], [10, 0, 0], [10, 1, 0]]
    truth = graph(nodes, [[0, 2], [1, 3]])
    predicted = graph(nodes, [[0, 1], [2, 3]])
    counts, rows, pairs = diagnose(predicted, truth, .1, [2, 5, 8], [2, 5, 8])
    result = summarize(counts, rows, pairs)
    assert result['node_recall'] == 1
    assert result['endpoint_pair_recovery'] == 1
    assert result['conditional_direct_recall'] == 0
    assert pairs[0]['negative_candidates'] == pairs[0]['negative_selected'] == 2
    assert pairs[3]['positive_candidates'] == 2
    assert pairs[3]['positive_selected'] == 0


def test_fragmented_but_connected_branch_and_actual_break_are_separated():
    truth = graph([[0, 0, 0], [10, 0, 0]], [[0, 1]])
    chain = graph([[0, 0, 0], [5, 0, 0], [10, 0, 0]], [[0, 1], [1, 2]])
    _, rows, _ = diagnose(chain, truth, .1, [2, 5, 8], [2, 5, 8])
    assert not rows[0]['direct']
    assert rows[0]['indirect_path'] and rows[0]['degree2_chain']
    assert rows[0]['indirect_path_guide_aligned']
    broken = graph([[0, 0, 0], [4, 0, 0], [6, 0, 0], [10, 0, 0]], [[0, 1], [2, 3]])
    _, rows, _ = diagnose(broken, truth, .1, [2, 5, 8], [2, 5, 8])
    assert rows[0]['endpoints_recovered'] and not rows[0]['indirect_path']


def test_missing_endpoint_is_not_counted_as_a_relation_failure():
    truth = graph([[0, 0, 0], [10, 0, 0]], [[0, 1]])
    predicted = graph([[0, 0, 0], [8, 0, 0]], [[0, 1]])
    result = summarize(*diagnose(predicted, truth, 1., [2, 5, 8], [2, 5, 8]))
    assert result['node_recall'] == .5
    assert result['recoverable_edges'] == result['missing_direct_edges'] == 0
    assert result['conditional_direct_recall'] is None
    assert result['predicted_edges_with_unmatched_endpoint'] == 1
