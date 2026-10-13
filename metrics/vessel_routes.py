"""Explicitly adapted 3D APLS and reachable-neighborhood TOPO; all units are mm."""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import linear_sum_assignment
from .representation_geometry import sample_polyline_by_arclength,polyline_length


@dataclass
class RouteGraph:
    points: np.ndarray
    segments: np.ndarray
    weights: np.ndarray
    anchors: np.ndarray

    def matrix(self):
        # Parallel edges contribute their shortest length, rather than a sum.
        weights={}
        for (a,b),w in zip(self.segments,self.weights):
            for key in ((int(a),int(b)),(int(b),int(a))):weights[key]=min(weights.get(key,np.inf),float(w))
        row=[k[0] for k in weights];col=[k[1] for k in weights]
        return csr_matrix((list(weights.values()),(row,col)),shape=(len(self.points),len(self.points)))


def route_graph(contracted,spacing):
    keys=sorted(contracted.anchors);lookup={a:i for i,a in enumerate(keys)}
    points=[np.asarray(contracted.anchors[a]).copy() for a in keys]
    segments=[];weights=[]
    for branch in contracted.branches:
        samples=sample_polyline_by_arclength(branch.polyline,spacing)
        if contracted.roles[branch.start]=='cycle':points[lookup[branch.start]]=branch.polyline[0].copy()
        ids=[lookup[branch.start]]
        for p in samples[1:-1]:ids.append(len(points));points.append(p)
        ids.append(lookup[branch.end])
        step=polyline_length(branch.polyline)/max(1,len(ids)-1)
        segments.extend(zip(ids[:-1],ids[1:]));weights.extend([step]*(len(ids)-1))
    dimension=next(iter(contracted.anchors.values())).shape[0] if keys else 3
    return RouteGraph(np.asarray(points).reshape(-1,dimension),np.asarray(segments,dtype=int).reshape(-1,2),
                      np.asarray(weights),np.arange(len(keys)))


def control_indices(graph,limit):
    if not len(graph.points):return np.array([],dtype=int)
    order=np.lexsort(graph.points.T[::-1])
    if len(order)<=limit:return order
    anchors=set(graph.anchors.tolist())
    first=[i for i in order if i in anchors][:limit]
    rest=[i for i in order if i not in anchors]
    remaining=limit-len(first)
    return np.array(first+[rest[i] for i in np.linspace(0,len(rest)-1,remaining,dtype=int)] if remaining else first,dtype=int)


def inject(graph,positions,tolerance):
    """Project controls to the closest sampled physical segment, then split it."""
    mapping=np.full(len(positions),-1,dtype=int)
    if not len(graph.points):return graph,mapping
    points=graph.points.tolist();cuts={}
    starts=graph.points[graph.segments[:,0]];vectors=graph.points[graph.segments[:,1]]-starts
    lengths=(vectors*vectors).sum(1)
    for i,p in enumerate(positions):
        distances=np.linalg.norm(graph.points-p,axis=1);node=int(distances.argmin())
        best=float(distances[node]);segment=None;fraction=0.
        if len(starts):
            t=np.clip(((p-starts)*vectors).sum(1)/np.maximum(lengths,1e-20),0,1)
            projected=starts+t[:,None]*vectors;d=np.linalg.norm(projected-p,axis=1);j=int(d.argmin())
            if d[j]<best-1e-10:best=float(d[j]);segment=j;fraction=float(t[j])
        if best>tolerance:continue
        if segment is None:mapping[i]=node;continue
        existing=cuts.setdefault(segment,{})
        key=round(fraction,12)
        if key not in existing:
            existing[key]=len(points);points.append((starts[segment]+fraction*vectors[segment]).tolist())
        mapping[i]=existing[key]
    edges=[];weights=[]
    for j,((a,b),weight) in enumerate(zip(graph.segments,graph.weights)):
        parts=[(0.,int(a)),*sorted(cuts.get(j,{}).items()),(1.,int(b))]
        for (t,u),(s,v) in zip(parts[:-1],parts[1:]):edges.append((u,v));weights.append((s-t)*weight)
    return RouteGraph(np.asarray(points),np.asarray(edges,dtype=int).reshape(-1,2),np.asarray(weights),graph.anchors),mapping


def apls_direction(source,target,tolerance,limit):
    indices=control_indices(source,limit)
    if len(indices)<2:return dict(score=None,pairs=0,penalty_sum=0.)
    source_dist=dijkstra(source.matrix(),directed=False,indices=indices)
    injected,mapping=inject(target,source.points[indices],tolerance)
    valid=mapping>=0;target_dist={}
    if valid.any():
        unique=np.unique(mapping[valid]);values=dijkstra(injected.matrix(),directed=False,indices=unique)
        target_dist=dict(zip(unique,values))
    penalty=0.;pairs=0
    for i in range(len(indices)):
        for j in range(i+1,len(indices)):
            length=source_dist[i,indices[j]]
            if not np.isfinite(length) or length<=1e-8:continue
            pairs+=1
            predicted=target_dist[mapping[i]][mapping[j]] if valid[i] and valid[j] else np.inf
            penalty+=min(1.,abs(predicted-length)/length) if np.isfinite(predicted) else 1.
    return dict(score=1-penalty/pairs if pairs else None,pairs=pairs,penalty_sum=penalty)


def apls(source,target,tolerance,limit):
    forward=apls_direction(source,target,tolerance,limit);backward=apls_direction(target,source,tolerance,limit)
    a,b=forward['score'],backward['score']
    if a is None and b is None:score=None
    elif a is None or b is None:score=0.
    else:score=2*a*b/(a+b) if a+b else 0.
    return dict(apls_3d=score,apls_gt_to_pred=forward['score'],apls_pred_to_gt=backward['score'],
                apls_gt_pairs=forward['pairs'],apls_pred_pairs=backward['pairs'])


def neighborhood_matches(first,second,tolerance):
    if not len(first) or not len(second):return 0
    # Maximum cardinality first. One inadmissible pair costs more than all distances.
    distances=np.linalg.norm(first[:,None]-second[None,:],axis=-1)
    penalty=(min(len(first),len(second))+1)*(tolerance+1.)
    rows,cols=linear_sum_assignment(np.where(distances<=tolerance,distances,penalty))
    return int((distances[rows,cols]<=tolerance).sum())


def topo_direction(source,target,tolerance,radius,limit):
    seeds=control_indices(source,limit);hits=total=0
    if not len(seeds):return hits,total
    sd=dijkstra(source.matrix(),directed=False,indices=seeds,limit=radius)
    td_cache={};matrix=target.matrix()
    for seed,distances in zip(seeds,sd):
        neighborhood=source.points[np.isfinite(distances)];total+=len(neighborhood)
        if not len(target.points):continue
        delta=np.linalg.norm(target.points-source.points[seed],axis=1);nearest=int(delta.argmin())
        if delta[nearest]>tolerance:continue
        if nearest not in td_cache:td_cache[nearest]=dijkstra(matrix,directed=False,indices=nearest,limit=radius)
        other=target.points[np.isfinite(td_cache[nearest])]
        hits+=neighborhood_matches(neighborhood,other,tolerance)
    return hits,total


def topo(predicted,reference,tolerance,radius,limit):
    p,n=topo_direction(predicted,reference,tolerance,radius,limit)
    r,m=topo_direction(reference,predicted,tolerance,radius,limit)
    precision=p/n if n else (1. if not m else 0.)
    recall=r/m if m else (1. if not n else 0.)
    return dict(topo_3d_precision=precision,topo_3d_recall=recall,
        topo_3d_f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
        topo_pred_hits=p,topo_pred_samples=n,topo_gt_hits=r,topo_gt_samples=m)
