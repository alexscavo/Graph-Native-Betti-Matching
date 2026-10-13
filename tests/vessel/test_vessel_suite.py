"""Small failure cases establish the meaning of the new physical metric protocol."""
import numpy as np
import pytest
from metrics.vessel_graph import VesselGraph,contract
from metrics.vessel_suite import evaluate,aggregate,Protocol


P=Protocol(curve_tolerances_mm=(.5,1.,2.),route_spacing_mm=.5)


def graph(nodes,edges,curves=None):return VesselGraph.from_arrays(nodes,edges,curves)


def test_identity_and_degree_two_subdivision():
    ref=graph([[0,0,0],[4,0,0]],[[0,1]])
    split=graph([[0,0,0],[4,0,0],[2,0,0]],[[0,2],[2,1]])
    a=evaluate(ref,ref,P);b=evaluate(split,ref,P)
    for key in ('curve_f1_at_1mm','branch_f1','anchor_f1','topo_3d_f1','apls_3d'):
        assert a[key]==pytest.approx(1) and b[key]==pytest.approx(a[key]),key
    assert b['average_centerline_distance_mm']==0 and b['hd95_mm']==0
    assert b['representation_nodes']>a['representation_nodes']


def test_break_preserves_overlap_but_damages_routes():
    ref=graph([[0,0,0],[8,0,0]],[[0,1]])
    broken=graph([[0,0,0],[3.9,0,0],[4.1,0,0],[8,0,0]],[[0,1],[2,3]])
    row=evaluate(broken,ref,P)
    assert row['curve_f1_at_1mm']==1
    assert row['branch_f1']==0 and row['apls_3d']<.9 and row['topo_3d_f1']<1
    assert row['betti0_error']==1


def test_curve_lengths_and_spline_geometry_are_used():
    controls=np.array([[0,0,0],[1,4,0],[3,4,0],[4,0,0]],dtype=float)
    record=dict(nodes_dhw=controls[[0,-1]],edges=[[0,1]],spline_controls=[controls])
    curved=VesselGraph.from_prediction(record,np.eye(3))
    ref=graph(controls[[0,-1]],[[0,1]],curved.curves_mm)
    chord=VesselGraph.from_prediction(record,np.eye(3),curved=False)
    a=evaluate(curved,ref,P);b=evaluate(chord,ref,P)
    assert a['curve_f1_at_1mm']==1 and a['apls_3d']==pytest.approx(1)
    assert b['curve_f1_at_1mm']<1 and b['apls_3d']<1 and b['hd95_mm']>1
    assert a['predicted_length_mm']>b['predicted_length_mm']
    assert curved.representation['internal_points']==2
    assert curved.representation['geometry_connectivity_bytes_float32_int32']==56


def test_isolates_cycles_and_multiedges_are_counted_consistently():
    isolate=graph([[0,0,0]],[])
    row=evaluate(isolate,isolate,P)
    assert row['anchor_tp']==1 and row['anchor_fp']==row['anchor_fn']==0
    circle=np.array([[0,0,0],[2,0,0],[2,2,0],[0,2,0],[0,0,0]])
    loop=graph([[0,0,0]],[[0,0]],[circle])
    dense=graph(circle[:-1],[[0,1],[1,2],[2,3],[3,0]])
    row=evaluate(loop,dense,P)
    assert row['branch_f1']==1 and row['anchor_fp']==row['anchor_fn']==0
    assert row['matched_tortuosity_error'] is None and row['matched_closed_tortuosity_excluded']==1
    assert row['betti1_error']==0 and contract(loop).betti_1==1
    ref=graph([[0,0,0],[4,0,0]],[[0,1]])
    duplicate=graph([[0,0,0],[4,0,0]],[[0,1],[0,1]])
    row=evaluate(duplicate,ref,P)
    assert row['curve_f1_at_1mm']==1 and row['betti1_error']==1 and row['branch_f1']<1


def test_centerline_connectivity_is_explicit_and_crossings_do_not_join():
    lines=[np.array([[0,0,0],[2,0,0]]),np.array([[1,-1,0],[1,1,0]])]
    bare=VesselGraph.from_centerlines(lines)
    assert evaluate(bare,bare,P)['topology_available'] is False
    connected=VesselGraph.from_centerlines(lines,point_ids=[[0,1],[2,3]])
    assert contract(connected).betti_0==2
    branch_lines=[np.array([[0,0,0],[1,0,0],[2,0,0]]),np.array([[0,0,0],[1,0,0],[1,1,0]])]
    mesh=VesselGraph.from_centerlines(branch_lines,point_ids=[[0,1,2],[0,1,3]])
    assert len(mesh.edges)==3 and contract(mesh).roles[1]=='junction'
    row=evaluate(mesh,mesh,P)
    assert row['junction_f1']==row['branch_f1']==1


def test_empty_prediction_cannot_escape_global_scores():
    ref=graph([[0,0,0],[4,0,0]],[[0,1]])
    empty=graph([],[]);row=evaluate(empty,ref,P)
    assert row['curve_f1_at_1mm']==row['apls_3d']==row['branch_f1']==0
    assert row['average_centerline_distance_mm'] is None and row['geometry_missing_prediction']
    summary=aggregate([dict(sample_id='perfect',**evaluate(ref,ref,P)),dict(sample_id='missing',**row)],P)
    assert summary['distance_valid_cases']==1 and summary['geometry_missing_prediction_mean']==.5
    assert summary['curve_f1_at_1mm']<1 and summary['branch_f1']<1


def test_false_bridge_preserves_geometry_but_damages_topology():
    nodes=[[0,0,0],[8,0,0],[0,.5,0],[8,.5,0]]
    ref=graph(nodes,[[0,1],[2,3]])
    bridge=graph(nodes,[[0,1],[2,3],[0,2]])
    row=evaluate(bridge,ref,P)
    assert row['curve_f1_at_1mm']==1 and row['betti0_error']==1
    assert row['topo_3d_precision']<1 and row['apls_3d']<1
