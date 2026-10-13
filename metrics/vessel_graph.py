"""Physical vessel geometry with explicit connectivity and native storage counts."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline
from .branch_connectivity import Branch, ContractedGraph


@dataclass
class VesselGraph:
    nodes_mm: np.ndarray | None
    edges: np.ndarray | None
    curves_mm: tuple
    representation: dict

    @classmethod
    def from_arrays(cls, nodes_mm, edges, edge_polylines_mm=None, *, internal_controls=0):
        nodes = np.asarray(nodes_mm, dtype=float)
        if nodes.size == 0: nodes = nodes.reshape(0, 3)
        links = np.asarray(edges, dtype=int).reshape(-1, 2)
        if nodes.ndim != 2 or nodes.shape[1] not in (2, 3) or not np.isfinite(nodes).all():
            raise ValueError('finite 2D/3D physical coordinates required')
        if links.size and (links.min() < 0 or links.max() >= len(nodes)):
            raise ValueError('edge index outside nodes')
        curves = tuple(nodes[e].copy() for e in links) if edge_polylines_mm is None else tuple(
            np.asarray(c, dtype=float) for c in edge_polylines_mm)
        if len(curves) != len(links): raise ValueError('one curve per edge required')
        for edge, curve in zip(links, curves):
            if curve.ndim != 2 or curve.shape[1] != nodes.shape[1] or len(curve)<2 or not np.isfinite(curve).all():
                raise ValueError('invalid physical polyline')
            if not np.allclose(curve[[0,-1]], nodes[edge], atol=1e-5, rtol=0):
                raise ValueError('polyline endpoints disagree with graph connectivity')
        # Native representation, never the densified evaluation mesh. Endpoints are shared.
        extra = internal_controls if internal_controls else sum(max(0,len(c)-2) for c in curves)
        degrees = np.bincount(links.ravel(), minlength=len(nodes))
        scalars = nodes.size + nodes.shape[1]*extra
        offsets=len(curves)+1 if edge_polylines_mm is not None and extra and not internal_controls else 0
        return cls(nodes, links, curves, dict(nodes=len(nodes),edges=len(links),
            internal_points=int(extra),degree2_nodes=int((degrees==2).sum()),
            coordinate_scalars=int(scalars),index_scalars=int(links.size+offsets),
            geometry_connectivity_bytes_float32_int32=int(4*(scalars+links.size+offsets))))

    @classmethod
    def from_prediction(cls, record, metric, *, curved=True):
        nodes = np.asarray(record.get('nodes_dhw',record.get('nodes',[])),dtype=float).reshape(-1,3)
        edges = np.asarray(record['edges'],dtype=int).reshape(-1,2)
        controls = record.get('spline_controls') if curved else None
        if controls is None or not len(edges): return cls.from_arrays(nodes@metric.T, edges)
        curves=[]; count=0
        for control in controls:
            control=np.asarray(control,dtype=float); k=len(control); count+=k-2
            knots=np.r_[np.zeros(4),np.arange(1,k-3)/(k-3),np.ones(4)]
            if record.get('spline_family','bspline')!='bspline':
                raise ValueError('this adapter currently supports cubic B-splines; use polylines for other families')
            curves.append(BSpline(knots,control,3)(np.linspace(0,1,257))@metric.T)
        return cls.from_arrays(nodes@metric.T,edges,curves,internal_controls=count)

    @classmethod
    def from_centerlines(cls, polylines_mm, *, point_ids=None):
        """Bare polylines: geometry only. Shared point IDs explicitly supply topology.

        Coordinate coincidence/crossings never create connections implicitly.
        Dense centerline graphs can also use from_arrays(nodes_mm, edges).
        """
        curves=tuple(np.asarray(c,dtype=float) for c in polylines_mm)
        if any(c.ndim!=2 or c.shape[1] not in (2,3) or len(c)<2 or not np.isfinite(c).all() for c in curves):
            raise ValueError('finite centerline polylines required')
        if point_ids is None:
            scalars=sum(c.size for c in curves)
            offsets=len(curves)+1 if curves else 0
            return cls(None,None,curves,dict(nodes=None,edges=None,internal_points=None,
                degree2_nodes=None,coordinate_scalars=scalars,index_scalars=offsets,
                geometry_connectivity_bytes_float32_int32=4*(scalars+offsets)))
        if len(point_ids)!=len(curves): raise ValueError('one ID sequence per polyline required')
        mapping={};nodes=[];edges=[];seen=set()
        for curve, ids in zip(curves,point_ids):
            if len(ids)!=len(curve): raise ValueError('IDs must identify every polyline vertex')
            sequence=[]
            for position, identifier in zip(curve,ids):
                if identifier not in mapping:
                    mapping[identifier]=len(nodes);nodes.append(position)
                index=mapping[identifier]
                if not np.allclose(nodes[index],position,atol=1e-6,rtol=0):
                    raise ValueError('shared point ID has inconsistent coordinates')
                sequence.append(index)
            for a,b in zip(sequence[:-1],sequence[1:]):
                key=tuple(sorted((a,b)))
                # Shared mesh IDs identify one segment even in root-to-tip exports.
                if key not in seen:edges.append((a,b));seen.add(key)
        return cls.from_arrays(nodes,edges)


def contract(graph):
    """Contract degree-two chains, retaining distinct edge identities and loops.

    Separate from the legacy evaluator, whose historical behavior stays intact.
    """
    nodes,links=graph.nodes_mm,graph.edges
    if nodes is None: raise ValueError('topology requires explicit connectivity')
    adjacency=[[] for _ in nodes]; parent=list(range(len(nodes)))
    def root(n):
        while parent[n]!=n: parent[n]=parent[parent[n]];n=parent[n]
        return n
    for i,(a,b) in enumerate(links):
        adjacency[a].append((int(b),i));adjacency[b].append((int(a),i));parent[root(int(a))]=root(int(b))
    b0=len({root(i) for i in range(len(nodes))}); b1=len(links)-len(nodes)+b0
    anchors={i:nodes[i] for i,a in enumerate(adjacency) if len(a)!=2}
    roles={i:('isolated' if not adjacency[i] else 'endpoint' if len(adjacency[i])==1 else 'junction') for i in anchors}
    visited=set();branches=[]
    def trace(start,neighbor,index):
        pieces=[];current=start
        while True:
            visited.add(index); curve=graph.curves_mm[index]
            if links[index,0]!=current:curve=curve[::-1]
            pieces.append(curve if not pieces else curve[1:]);current=neighbor
            if current in anchors or current==start:break
            options=[(n,i) for n,i in adjacency[current] if i not in visited]
            if len(options)!=1:raise ValueError('invalid degree-two chain')
            neighbor,index=options[0]
        return current,np.concatenate(pieces)
    for a in sorted(anchors):
        for b,i in adjacency[a]:
            if i not in visited:
                end,curve=trace(a,b,i);branches.append(Branch(a,end,curve))
    for i,(a,b) in enumerate(links):
        if i in visited:continue
        end,curve=trace(int(a),int(b),i)
        if end!=a:raise ValueError('unclosed degree-two cycle')
        virtual=-(i+1);lengths=np.linalg.norm(np.diff(curve,axis=0),axis=1)
        anchors[virtual]=np.average((curve[:-1]+curve[1:])/2,axis=0,weights=lengths) if lengths.sum() else curve[0]
        roles[virtual]='cycle';branches.append(Branch(virtual,virtual,curve))
    return ContractedGraph(anchors,roles,tuple(branches),b0,b1)
