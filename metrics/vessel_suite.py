"""Versioned physical geometry/topology suite, independent of model tokens.

Legacy RelationFormer/common-v2 outputs are intentionally not modified.
"""
from dataclasses import dataclass,asdict
from collections import defaultdict,Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from .vessel_graph import contract
from .vessel_routes import route_graph,apls,topo
from .representation_geometry import directed_curve_distances,polyline_length,tortuosity


@dataclass(frozen=True)
class Protocol:
    version: str = 'vessel-core-v1'
    sample_spacing_mm: float = .25
    curve_tolerances_mm: tuple = (.5,1.,2.)
    anchor_tolerance_mm: float = 1.
    route_spacing_mm: float = .5
    route_snap_mm: float = 1.
    topo_radius_mm: float = 10.
    topo_max_seeds: int = 32
    apls_max_controls: int = 64

    def __post_init__(self):
        for value in (self.sample_spacing_mm,self.anchor_tolerance_mm,self.route_spacing_mm,
                      self.route_snap_mm,self.topo_radius_mm,*self.curve_tolerances_mm):
            if not np.isfinite(value) or value<=0:raise ValueError('physical spacings/tolerances must be positive')
        if self.topo_max_seeds<1 or self.apls_max_controls<2:raise ValueError('invalid route sampling budget')

    def metadata(self):return asdict(self)


def score_counts(tp,fp,fn):
    precision=tp/(tp+fp) if tp+fp else (1. if not fn else 0.)
    recall=tp/(tp+fn) if tp+fn else (1. if not fp else 0.)
    return dict(precision=precision,recall=recall,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,tp=tp,fp=fp,fn=fn)


def match(predicted,reference,tolerance,roles=None):
    p=sorted(i for i in predicted.anchors if roles is None or predicted.roles[i] in roles)
    g=sorted(i for i in reference.anchors if roles is None or reference.roles[i] in roles)
    if not p or not g:return {}
    distances=np.linalg.norm(np.array([predicted.anchors[i] for i in p])[:,None]-np.array([reference.anchors[i] for i in g])[None,:],axis=-1)
    eligible=np.array([[predicted.roles[i]==reference.roles[j] for j in g] for i in p])&(distances<=tolerance)
    penalty=(min(len(p),len(g))+1)*(tolerance+1.)
    rows,cols=linear_sum_assignment(np.where(eligible,distances,penalty))
    return {p[i]:g[j] for i,j in zip(rows,cols) if eligible[i,j]}


def tolerance_name(value):return f'{value:g}'.replace('.','p')+'mm'


def geometry(prediction_curves,reference_curves,protocol):
    pred=[c for c in prediction_curves if polyline_length(c)>0]
    gt=[c for c in reference_curves if polyline_length(c)>0]
    row=dict(geometry_missing_prediction=bool(gt and not pred),geometry_missing_reference=bool(pred and not gt))
    if pred and gt:
        forward=directed_curve_distances(pred,gt,protocol.sample_spacing_mm)
        backward=directed_curve_distances(gt,pred,protocol.sample_spacing_mm)
        row.update(pred_distance_sum_mm=float(forward.sum()),gt_distance_sum_mm=float(backward.sum()),
                   hd95_mm=float(max(np.percentile(forward,95),np.percentile(backward,95))))
    else:
        from .representation_geometry import sample_polyline_by_arclength
        forward=np.full(sum(len(sample_polyline_by_arclength(c,protocol.sample_spacing_mm)) for c in pred),np.inf)
        backward=np.full(sum(len(sample_polyline_by_arclength(c,protocol.sample_spacing_mm)) for c in gt),np.inf)
        row.update(pred_distance_sum_mm=None,gt_distance_sum_mm=None,hd95_mm=None if pred or gt else 0.)
    row.update(curve_pred_samples=len(forward),curve_gt_samples=len(backward))
    row['average_centerline_distance_mm']=float(.5*(forward.mean()+backward.mean())) if pred and gt else (None if pred or gt else 0.)
    for tolerance in protocol.curve_tolerances_mm:
        name=tolerance_name(tolerance); p=int((forward<=tolerance).sum());r=int((backward<=tolerance).sum())
        precision=p/len(forward) if len(forward) else (1. if not len(backward) else 0.)
        recall=r/len(backward) if len(backward) else (1. if not len(forward) else 0.)
        row.update({f'curve_pred_hits_at_{name}':p,f'curve_gt_hits_at_{name}':r,
            f'curve_precision_at_{name}':precision,f'curve_recall_at_{name}':recall,
            f'curve_f1_at_{name}':2*precision*recall/(precision+recall) if precision+recall else 0.})
    row.update(predicted_length_mm=sum(map(polyline_length,pred)),reference_length_mm=sum(map(polyline_length,gt)))
    row['length_absolute_error_mm']=abs(row['predicted_length_mm']-row['reference_length_mm'])
    return row


def evaluate(prediction,reference,protocol=Protocol()):
    p=contract(prediction) if prediction.nodes_mm is not None else None
    g=contract(reference) if reference.nodes_mm is not None else None
    row=geometry([b.polyline for b in p.branches] if p else prediction.curves_mm,
                 [b.polyline for b in g.branches] if g else reference.curves_mm,protocol)
    row.update({'representation_'+key:value for key,value in prediction.representation.items()})
    row['topology_available']=prediction.nodes_mm is not None and reference.nodes_mm is not None
    if not row['topology_available']:
        row.update(branch_f1=None,junction_f1=None,anchor_f1=None,apls_3d=None,topo_3d_f1=None,
                   betti0_error=None,betti1_error=None)
        return row
    all_mapping=match(p,g,protocol.anchor_tolerance_mm)
    # Real anchors exclude virtual cycle centroids on BOTH sides; isolates count.
    for label,roles in (('anchor',{'isolated','endpoint','junction'}),('junction',{'junction'}),('endpoint',{'endpoint'})):
        hits=sum(p.roles[i] in roles for i in all_mapping)
        npred=sum(r in roles for r in p.roles.values());ngt=sum(r in roles for r in g.roles.values())
        row.update({label+'_'+k:v for k,v in score_counts(hits,npred-hits,ngt-hits).items()})
    target=Counter(tuple(sorted((b.start,b.end))) for b in g.branches)
    observed=Counter(tuple(sorted((all_mapping[b.start],all_mapping[b.end]))) for b in p.branches if b.start in all_mapping and b.end in all_mapping)
    hits=sum((target&observed).values())
    row.update({'branch_'+k:v for k,v in score_counts(hits,len(p.branches)-hits,len(g.branches)-hits).items()})
    row.update(predicted_branches=len(p.branches),reference_branches=len(g.branches),
               predicted_betti0=p.betti_0,reference_betti0=g.betti_0,predicted_betti1=p.betti_1,reference_betti1=g.betti_1,
               betti0_error=abs(p.betti_0-g.betti_0),betti1_error=abs(p.betti_1-g.betti_1))
    # Matched shape diagnostics cannot silently exclude unmatched branches: expose count.
    available=defaultdict(list)
    for b in g.branches:available[tuple(sorted((b.start,b.end)))].append(b)
    tort=[];closed=0
    for b in p.branches:
        if b.start not in all_mapping or b.end not in all_mapping:continue
        key=tuple(sorted((all_mapping[b.start],all_mapping[b.end])))
        if not available[key]:continue
        other=available[key].pop(0);a,c=tortuosity(b.polyline),tortuosity(other.polyline)
        if not np.isfinite(a) or not np.isfinite(c):closed+=1
        else:tort.append(abs(a-c))
    row.update(matched_tortuosity_error=float(np.mean(tort)) if tort else None,
               matched_open_tortuosity_branches=len(tort),matched_closed_tortuosity_excluded=closed)
    pred_routes=route_graph(p,protocol.route_spacing_mm);gt_routes=route_graph(g,protocol.route_spacing_mm)
    row.update(apls(gt_routes,pred_routes,protocol.route_snap_mm,protocol.apls_max_controls))
    row.update(topo(pred_routes,gt_routes,protocol.route_snap_mm,protocol.topo_radius_mm,protocol.topo_max_seeds))
    return row


def aggregate(rows,protocol=Protocol()):
    if not rows:raise ValueError('no evaluated cases')
    result=dict(samples=len(rows),protocol=protocol.metadata(),topology_cases=sum(r['topology_available'] for r in rows))
    for key in rows[0]:
        if key=='sample_id' or isinstance(rows[0][key],str):continue
        values=[r[key] for r in rows if r.get(key) is not None and isinstance(r[key],(int,float,bool))]
        result[key+'_mean']=float(np.mean(values)) if values else None
    for label in ('anchor','junction','endpoint','branch'):
        selected=[r for r in rows if r['topology_available']]
        if selected:
            counts=[sum(r[label+'_'+k] for r in selected) for k in ('tp','fp','fn')]
            result.update({label+'_'+k:v for k,v in score_counts(*counts).items()})
    n=sum(r['curve_pred_samples'] for r in rows);m=sum(r['curve_gt_samples'] for r in rows)
    for tolerance in protocol.curve_tolerances_mm:
        name=tolerance_name(tolerance);p=sum(r[f'curve_pred_hits_at_{name}'] for r in rows);g=sum(r[f'curve_gt_hits_at_{name}'] for r in rows)
        precision=p/n if n else (1. if not m else 0.);recall=g/m if m else (1. if not n else 0.)
        result.update({f'curve_precision_at_{name}':precision,f'curve_recall_at_{name}':recall,
                       f'curve_f1_at_{name}':2*precision*recall/(precision+recall) if precision+recall else 0.})
    # Distance aggregation includes only cases with both geometries, with coverage explicit.
    valid=[r for r in rows if r['pred_distance_sum_mm'] is not None]
    result['distance_valid_cases']=len(valid)
    if valid:
        result['average_centerline_distance_mm']=.5*(sum(r['pred_distance_sum_mm'] for r in valid)/sum(r['curve_pred_samples'] for r in valid)+sum(r['gt_distance_sum_mm'] for r in valid)/sum(r['curve_gt_samples'] for r in valid))
    else:result['average_centerline_distance_mm']=None
    for key in ('apls_3d','topo_3d_f1'):
        values=[r[key] for r in rows if r.get(key) is not None]
        result[key]=float(np.mean(values)) if values else None
        result[key+'_valid_cases']=len(values)
    return result
