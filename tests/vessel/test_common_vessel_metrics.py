"""The common evaluation must not reward extra representation nodes."""
import numpy as np
from scipy.interpolate import BSpline
from metrics.common_vessel_metrics import evaluate_common_vessels,aggregate_common_vessels


def reference(path,points):
    np.savez(path,nodes=points[[0,-1]],edges=[[0,1]],metric=np.eye(3),full_paths=points,path_offsets=[0,len(points)])


def test_degree_two_subdivision_preserves_common_scores(tmp_path):
    path=tmp_path/'guide.npz';nodes=np.array([[0.,0,0],[1.,0,0]])
    reference(path,nodes)
    a=evaluate_common_vessels({'nodes':nodes,'edges':[[0,1]]},path)
    b=evaluate_common_vessels({'nodes':[[0,0,0],[.5,0,0],[1,0,0]],'edges':[[0,1],[1,2]]},path)
    for key in a:
        if key.startswith('common_'):assert np.allclose(a[key],b[key]),key
    assert a['native_predicted_nodes']==2 and b['native_predicted_nodes']==3
    assert aggregate_common_vessels([{'sample_id':'fixture',**b}])['common_anchor_f1']==1


def test_curvature_is_scored_against_full_guide(tmp_path):
    path=tmp_path/'guide.npz';controls=np.array([[0.,0,0],[.2,2.5,0],[.8,2.5,0],[1.,0,0]])
    curve=BSpline([0,0,0,0,1,1,1,1],controls,3)(np.linspace(0,1,257));reference(path,curve)
    straight=evaluate_common_vessels({'nodes':curve[[0,-1]],'edges':[[0,1]]},path)
    curved=evaluate_common_vessels({'nodes':curve[[0,-1]],'edges':[[0,1]],'spline_controls':[controls]},path)
    assert straight['common_branch_tp']==curved['common_branch_tp']==1
    assert curved['common_guide_distance_mm']<1e-10
    assert straight['common_guide_distance_mm']>curved['common_guide_distance_mm']
    assert aggregate_common_vessels([curved])['common_curve_f1']==1
    assert aggregate_common_vessels([straight])['common_curve_f1']<1


def test_missing_vessel_is_counted_in_global_geometry(tmp_path):
    path=tmp_path/'guide.npz';reference(path,np.array([[0.,0,0],[1.,0,0]]))
    row=evaluate_common_vessels({'nodes':[],'edges':[]},path)
    summary=aggregate_common_vessels([row])
    assert summary['common_branch_coverage']==0
    assert summary['common_curve_recall']==0 and summary['common_curve_f1']==0
