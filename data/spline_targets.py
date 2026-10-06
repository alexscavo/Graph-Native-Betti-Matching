"""Full-path crop and endpoint-constrained oracle fit for supplied vessel guides."""
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import BSpline


def contract_spline_graph(nodes, edges, max_anchors=120, *, closed_components=0):
    """Return the contracted target and its whole-patch exclusion reason."""
    from metrics.branch_connectivity import contract_degree_two
    links = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    # The generic contractor only accepts simple embedded graphs. Check the
    # current decoder's unsupported cases before invoking it: never deduplicate
    # records, which could silently erase a parallel vessel branch.
    if closed_components or np.any(links[:, 0] == links[:, 1]):
        return None, 'closed_branch'
    source_pairs = {tuple(sorted(pair)) for pair in links.tolist()}
    if len(source_pairs) != len(links):
        return None, 'parallel_branches'
    graph = contract_degree_two(nodes, edges)
    pairs = [tuple(sorted((b.start, b.end))) for b in graph.branches]
    reason = ('closed_branch' if any(a == b or a < 0 for a, b in pairs) else
              'parallel_branches' if len(set(pairs)) != len(pairs) else
              'object_capacity' if len(graph.anchors) > max_anchors else None)
    return graph, reason


def crop_dense_paths(source, bounds, *, retain_tangent_contacts=False, boundary_policy='exact'):
    if boundary_policy not in ('exact', 'last_inside_sample'):
        raise ValueError('Unknown boundary policy')
    ids = [int(i) for i in source.node_ids if source._inside(source.nodes[int(i)], bounds)]
    positions = [source.nodes[i].copy() for i in ids]
    mapping = {i: j for j, i in enumerate(ids)}
    boundary = [False]*len(ids)
    edges, contacts, snapped, fallbacks, closed = [], 0, 0, 0, 0

    def endpoint(point, edge):
        for i in (edge.node1, edge.node2):
            if i in mapping and np.allclose(point, source.nodes[i], atol=1e-7, rtol=0):
                return mapping[i]
        positions.append(point.copy()); boundary.append(True)
        return len(positions)-1

    for k in source._candidate_edge_indices(bounds):
        edge = source.edge_geometries[int(k)]
        polyline = source.edge_polylines[int(k)]
        for part in source._clip_polyline(polyline, bounds):
            if len(part) == 1:
                contacts += 1
                if retain_tangent_contacts:
                    endpoint(part[0], edge)
                continue
            if boundary_policy == 'last_inside_sample':
                # A clipped component retains original samples between its two
                # intersections. Trim only interpolated endpoints, never choose
                # a sample from another component of a re-entering vessel.
                first_original = np.any(np.all(np.isclose(polyline, part[0], atol=1e-8, rtol=0), axis=1))
                last_original = np.any(np.all(np.isclose(polyline, part[-1], atol=1e-8, rtol=0), axis=1))
                first = 0 if first_original else 1
                stop = len(part) if last_original else len(part)-1
                trimmed = part[first:stop]
                if len(trimmed) >= 2 and np.linalg.norm(trimmed[-1]-trimmed[0]) > 1e-10:
                    snapped += int(not first_original)+int(not last_original)
                    part = trimmed
                elif not first_original or not last_original:
                    # Preserve positive-length foreground-guided components when
                    # the sampling is too sparse for two existing endpoints.
                    fallbacks += 1
            left, right = endpoint(part[0], edge), endpoint(part[-1], edge)
            closed += int(left == right)
            chain = [left]
            for point in part[1:-1]:
                chain.append(len(positions)); positions.append(point.copy()); boundary.append(False)
            chain.append(right); edges.extend(zip(chain[:-1], chain[1:]))
    # Retain isolated original nodes: do not silently alter target topology.
    return SimpleNamespace(positions=np.asarray(positions).reshape(-1, 3),
                           edges=np.asarray(edges, dtype=int).reshape(-1, 2),
                           boundary=np.asarray(boundary), tangent_contacts=contacts,
                           snapped_endpoints=snapped, sparse_sample_fallbacks=fallbacks,
                           closed_components=closed)


def resample(points, samples):
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance)>1e-10]
    points, distance = points[keep], distance[keep]
    if distance[-1] <= 1e-10:
        raise ValueError('Degenerate guide')
    return np.column_stack([np.interp(np.linspace(0, distance[-1], samples), distance, points[:, i]) for i in range(3)])


def fit_spline(points, controls=8, samples=64):
    knots = np.r_[np.zeros(4), np.arange(1, controls-3)/(controls-3), np.ones(4)]
    guide = resample(points, 257)
    train_basis = BSpline(knots, np.eye(controls), 3)(np.linspace(0, 1, 257))
    target = guide-train_basis[:, :1]*guide[0]-train_basis[:, -1:]*guide[-1]
    internal = np.linalg.lstsq(train_basis[:, 1:-1], target, rcond=None)[0]
    fitted = np.vstack((guide[0], internal, guide[-1]))
    return fitted, BSpline(knots, fitted, 3)(np.linspace(0, 1, samples))
