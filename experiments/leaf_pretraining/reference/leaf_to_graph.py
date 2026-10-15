"""
leaf_to_graph.py

Convert a color-coded leaf-venation segmentation PNG (red = 1st order / midrib,
green = 2nd order veins, blue = 3rd+ order veins -- adjust COLOR_ORDER below if
your dataset uses a different convention) into a topology graph, exported as:

  - <name>_nodes.csv   : one row per graph node (endpoint or branch point).
                          x/y are in "plot" convention (y flipped so image-down
                          becomes plot-up), matching the .vtp below.
  - <name>_edges.csv   : one row per edge, incl. the full centerline polyline.
                          NOTE: the points_yx column is raw (row, col) PIXEL
                          coords (row increases downward, image convention) --
                          NOT flipped like nodes.csv/vtp. If you plot edges and
                          nodes together, flip edge points' row sign to match,
                          or use ax.invert_yaxis() when plotting points_yx alone.
  - <name>.vtp          : VTK PolyData (ASCII) with the true skeleton centerlines,
                           vein order stored as cell data, for QC / visualization
                           and directly usable if your loader expects .vtp.
                           Points are (x, -row, 0) -- y flipped, same as nodes.csv.

No external deps beyond numpy / scipy / scikit-image / networkx (all present in
your usual conda/pip science stack).

Usage:
    python leaf_to_graph.py input.png out_dir/ [--min-branch-len 3]
    python leaf_to_graph.py input.png out_dir/ --veins red,green   # drop blue/minor veins
    python leaf_to_graph.py input.png out_dir/ --veins red         # midrib only
"""
import sys, os, argparse, time
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import networkx as nx

# ---- adjust to your dataset's color->order convention ---------------------
COLOR_ORDER = {
    "red":   1,   # midrib / 1st order
    "green": 2,   # 2nd order (secondary) veins
    "blue":  3,   # 3rd order+ (minor/areole) veins
}
# ----------------------------------------------------------------------------

NEI8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def load_masks(path):
    img = np.array(Image.open(path).convert("RGB"))
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    masks = {
        "red":   (r > 128) & (g < 128) & (b < 128),
        "green": (g > 128) & (r < 128) & (b < 128),
        "blue":  (b > 128) & (r < 128) & (g < 128),
    }
    return img, masks


def build_skeleton_and_labels(masks, veins=None):
    """
    masks: dict color-name -> bool array (from load_masks)
    veins: iterable of color names to include (subset of masks.keys()).
           None / omitted = use all colors present.
    Skeletonization runs ONLY on the union of the selected colors, so
    excluded veins are treated as background -- they don't just get
    filtered out of the graph after the fact, they never influence the
    skeleton's topology (no dangling stubs where a dropped vein used to
    branch off).
    """
    veins = list(masks.keys()) if veins is None else list(veins)
    unknown = set(veins) - set(masks.keys())
    if unknown:
        raise ValueError(f"Unknown vein color(s) {unknown}; available: {list(masks.keys())}")

    union = np.zeros_like(next(iter(masks.values())))
    for c in veins:
        union |= masks[c]
    skel = skeletonize(union)

    order = np.zeros(skel.shape, dtype=np.uint8)
    for c in veins:
        order[skel & masks[c]] = COLOR_ORDER[c]

    # any skeleton pixel that (rarely, due to thinning at exact color boundary)
    # didn't land on a colored pixel gets the color of its nearest selected px
    missing = skel & (order == 0)
    if missing.any():
        colored = np.zeros(skel.shape, dtype=np.uint8)
        for c in veins:
            colored[masks[c]] = COLOR_ORDER[c]
        _, (iy, ix) = ndi.distance_transform_edt(colored == 0, return_indices=True)
        order[missing] = colored[iy[missing], ix[missing]]
    return skel, order


from skimage.measure import approximate_polygon


def merge_junction_clusters(G, radius=3.0):
    """
    In an 8-connected skeleton a single anatomical branch point usually thins to
    a small CLUSTER of adjacent pixels that each have degree >= 3, so one real
    junction becomes 2-4 nodes joined by ~1px edges (and often a fake triangle).
    This collapses each such cluster into one node at its centroid.

    Only edges shorter than `radius` whose BOTH endpoints are junctions
    (degree >= 3) are contracted, so genuinely short vein segments ending in a
    tip (degree 1) are never eaten.
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        # Merge a short edge unless one end is a genuine vein tip (degree 1).
        # Requiring degree >= 3 on both ends missed degenerate 2-px loops, where
        # a node carries two parallel edges to the same neighbour (degree 2).
        if u != v and d["length"] < radius and G.degree(u) >= 2 and G.degree(v) >= 2:
            H.add_edge(u, v)

    clusters = [c for c in nx.connected_components(H) if len(c) > 1]
    if not clusters:
        return G

    rep_of = {}          # old node id -> surviving node id
    new_pos = {}         # surviving node id -> (row, col) centroid
    for c in clusters:
        rep = min(c)
        ys = [G.nodes[n]["y"] for n in c]
        xs = [G.nodes[n]["x"] for n in c]
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        for n in c:
            rep_of[n] = rep
        new_pos[rep] = (cy, cx)

    G2 = nx.MultiGraph()
    for n, d in G.nodes(data=True):
        r = rep_of.get(n, n)
        if r not in G2:
            if r in new_pos:
                cy, cx = new_pos[r]
                G2.add_node(r, y=cy, x=cx, order=d["order"])
            else:
                G2.add_node(r, y=d["y"], x=d["x"], order=d["order"])

    for u, v, d in G.edges(data=True):
        ru, rv = rep_of.get(u, u), rep_of.get(v, v)
        if ru == rv:
            continue  # edge lived entirely inside one cluster -> gone
        pts = list(d["points"])
        # orient path to start at u (undirected iteration may hand back (u,v) flipped)
        uy, ux = G.nodes[u]["y"], G.nodes[u]["x"]
        vy, vx = G.nodes[v]["y"], G.nodes[v]["x"]
        if ((pts[0][0]-vy)**2 + (pts[0][1]-vx)**2) < ((pts[0][0]-uy)**2 + (pts[0][1]-ux)**2):
            pts = pts[::-1]
        if ru in new_pos:
            pts[0] = new_pos[ru]
        if rv in new_pos:
            pts[-1] = new_pos[rv]
        length = float(sum(np.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
                           for i in range(len(pts)-1)))
        G2.add_edge(ru, rv, points=pts, length=length, order=d["order"])
    return G2


def _max_dev_from_chain(pts, chain):
    """Max distance from every original point to the polyline `chain`."""
    best = np.full(len(pts), np.inf)
    for a, b in zip(chain[:-1], chain[1:]):
        ab = b - a
        L2 = float(ab[0]**2 + ab[1]**2)
        if L2 == 0:
            d = np.hypot(pts[:, 0]-a[0], pts[:, 1]-a[1])
        else:
            t = np.clip(((pts[:, 0]-a[0])*ab[0] + (pts[:, 1]-a[1])*ab[1]) / L2, 0.0, 1.0)
            px, py = a[0] + t*ab[0], a[1] + t*ab[1]
            d = np.hypot(pts[:, 0]-px, pts[:, 1]-py)
        best = np.minimum(best, d)
    return float(best.max())


def _prune_collinear(pts, chain, tolerance, angle_thresh):
    """
    Drop interior vertices whose turning angle exceeds `angle_thresh` degrees
    (i.e. nearly straight, carrying no shape information), but only when doing
    so keeps every original point within `tolerance` of the reduced chain.

    RDP bounds deviation but does not produce a minimal vertex set: a point kept
    early in the recursion can become near-collinear once its neighbours are
    refined by later splits. This removes those leftovers.
    """
    chain = [np.asarray(p, float) for p in chain]
    changed = True
    while changed and len(chain) > 2:
        changed = False
        i = 1
        while i < len(chain) - 1:
            a, p, b = chain[i-1], chain[i], chain[i+1]
            va, vb = a - p, b - p
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na == 0 or nb == 0:
                cand = chain[:i] + chain[i+1:]
                chain = cand; changed = True; continue
            cos = float(np.dot(va, vb) / (na * nb))
            ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
            if ang > angle_thresh:
                cand = chain[:i] + chain[i+1:]
                if _max_dev_from_chain(pts, np.array(cand)) <= tolerance:
                    chain = cand; changed = True; continue
            i += 1
    return np.array(chain)


def _prune_close(pts, chain, tolerance, min_spacing):
    """
    Drop interior vertices that sit closer than `min_spacing` to a neighbouring
    vertex, when doing so keeps every original point within `tolerance`.
    RDP bounds deviation but says nothing about spacing, so it can place a shape
    node a pixel or two from a junction where a vein bends sharply on exit.
    """
    chain = [np.asarray(p, float) for p in chain]
    changed = True
    while changed and len(chain) > 2:
        changed = False
        for i in range(1, len(chain) - 1):
            d_prev = float(np.hypot(*(chain[i] - chain[i-1])))
            d_next = float(np.hypot(*(chain[i] - chain[i+1])))
            if min(d_prev, d_next) < min_spacing:
                cand = chain[:i] + chain[i+1:]
                if _max_dev_from_chain(pts, np.array(cand)) <= tolerance:
                    chain = cand
                    changed = True
                    break
    return np.array(chain)


def densify_with_shape_nodes(G, tolerance, collinear_angle=170.0, min_spacing=0.0):
    """
    Insert extra degree-2 'shape' nodes along each edge wherever needed to
    keep a piecewise-straight approximation within `tolerance` pixels of the
    true skeleton centerline (Ramer-Douglas-Peucker simplification, via
    skimage.measure.approximate_polygon). Returns a NEW MultiGraph where
    every edge is a straight (p0, p1) segment -- curvy edges get split into
    several straight sub-edges through these new shape nodes.

    tolerance: max allowed perpendicular deviation (in pixels) between the
    simplified straight segments and the true centerline. Smaller = more
    shape nodes / tighter fit to curvature, larger = fewer nodes / coarser.
    """
    G2 = nx.MultiGraph()
    for n, d in G.nodes(data=True):
        G2.add_node(n, x=d["x"], y=d["y"], order=d["order"])
    next_id = (max(G.nodes()) + 1) if G.number_of_nodes() else 0

    for u, v, data in G.edges(data=True):
        pts = np.asarray(data["points"], dtype=float)  # (row, col), full pixel path
        # G is an UNDIRECTED MultiGraph: edges(data=True) may yield (u, v) in the
        # opposite order to how the edge was added, while `points` still runs in
        # the original traced direction. Orient `points` to actually start at u,
        # or the chain below wires the first shape node to the wrong endpoint
        # (producing long spurious edges across the image).
        uy, ux = G.nodes[u]["y"], G.nodes[u]["x"]
        vy, vx = G.nodes[v]["y"], G.nodes[v]["x"]
        d_u = (pts[0][0] - uy) ** 2 + (pts[0][1] - ux) ** 2
        d_v = (pts[0][0] - vy) ** 2 + (pts[0][1] - vx) ** 2
        if d_v < d_u:
            pts = pts[::-1]
        if len(pts) <= 2:
            G2.add_edge(u, v, order=data["order"], length=data["length"],
                        points=[tuple(pts[0]), tuple(pts[-1])])
            continue
        simplified = approximate_polygon(pts, tolerance=tolerance)  # keeps endpoints
        if collinear_angle and collinear_angle < 180 and len(simplified) > 2:
            simplified = _prune_collinear(pts, simplified, tolerance, collinear_angle)
        if min_spacing > 0 and len(simplified) > 2:
            simplified = _prune_close(pts, simplified, tolerance, min_spacing)
        chain = [u]
        for p in simplified[1:-1]:
            nid = next_id; next_id += 1
            G2.add_node(nid, x=float(p[1]), y=float(p[0]), order=int(data["order"]))
            chain.append(nid)
        chain.append(v)
        for a, b in zip(chain[:-1], chain[1:]):
            ay, ax_ = G2.nodes[a]["y"], G2.nodes[a]["x"]
            by, bx_ = G2.nodes[b]["y"], G2.nodes[b]["x"]
            seglen = float(np.hypot(by - ay, bx_ - ax_))
            G2.add_edge(a, b, order=int(data["order"]), length=seglen,
                        points=[(ay, ax_), (by, bx_)])
    return G2


def degree_map(skel):
    k = np.ones((3, 3), dtype=int)
    k[1, 1] = 0
    deg = ndi.convolve(skel.astype(int), k, mode="constant", cval=0)
    deg[~skel] = 0
    return deg


def trace_graph(skel, order):
    """Skeleton -> graph (nodes = endpoints/junctions, edges = pixel polylines)."""
    deg = degree_map(skel)
    skel_pts = set(map(tuple, np.argwhere(skel)))
    node_pts = set(map(tuple, np.argwhere(skel & ((deg == 1) | (deg >= 3)))))

    # isolated simple loops with no junction/endpoint: seed one node per loop
    lbl, n_comp = ndi.label(skel, structure=np.ones((3, 3)))
    for comp_id in range(1, n_comp + 1):
        comp_pts = set(map(tuple, np.argwhere(lbl == comp_id)))
        if not (comp_pts & node_pts):
            node_pts.add(next(iter(comp_pts)))

    G = nx.MultiGraph()
    node_id_of = {}
    for i, p in enumerate(sorted(node_pts)):
        node_id_of[p] = i
        G.add_node(i, y=int(p[0]), x=int(p[1]), order=int(order[p]))

    visited_dir = set()  # (from_pixel, to_pixel) directed steps already consumed

    def neighbors(p):
        y, x = p
        for dy, dx in NEI8:
            q = (y + dy, x + dx)
            if q in skel_pts:
                yield q

    for start in node_pts:
        for nb in list(neighbors(start)):
            step = (start, nb)
            if step in visited_dir:
                continue
            # walk from start, through nb, until next node
            path = [start, nb]
            visited_dir.add(step)
            prev, cur = start, nb
            while cur not in node_pts:
                nxts = [q for q in neighbors(cur) if q != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                visited_dir.add((cur, nxt))
                path.append(nxt)
                prev, cur = cur, nxt
            if cur in node_pts:
                visited_dir.add((path[-1], path[-2]))  # mark reverse consumed too
                u, v = node_id_of[start], node_id_of[cur]
                length = float(sum(np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
                                    for i in range(len(path)-1)))
                orders_on_path = [int(order[p]) for p in path]
                vein_order = max(set(orders_on_path), key=orders_on_path.count)  # mode
                G.add_edge(u, v, points=path, length=length, order=vein_order)
    return G


def write_csv(G, out_prefix):
    import csv
    with open(out_prefix + "_nodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "z", "degree", "vein_order", "node_type"])
        for n, d in G.nodes(data=True):
            deg = G.degree(n)
            # degree 1 = vein tip, degree 2 = inserted shape point (only
            # present if densify_with_shape_nodes was used), degree >=3 = branch
            node_type = "endpoint" if deg == 1 else ("shape" if deg == 2 else "branch")
            # flip y to match write_vtp()'s convention (image row -> plot "up")
            w.writerow([n, d["x"], -d["y"], 0.0, deg, d["order"], node_type])

    with open(out_prefix + "_edges.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "target", "vein_order", "length_px", "num_points", "points_yx"])
        for i, (u, v, d) in enumerate(G.edges(data=True)):
            pts_str = ";".join(f"{p[0]},{p[1]}" for p in d["points"])
            w.writerow([i, u, v, d["order"], round(d["length"], 3), len(d["points"]), pts_str])


def write_vtp(G, out_path):
    """Write full-resolution centerlines as an ASCII VTK PolyData (.vtp)."""
    points = []
    lines = []  # list of index-lists into `points`
    cell_order = []
    for u, v, d in G.edges(data=True):
        idx0 = len(points)
        for (y, x) in d["points"]:
            points.append((float(x), float(-y), 0.0))  # flip y so "up" is up
        n = len(d["points"])
        lines.append(list(range(idx0, idx0 + n)))
        cell_order.append(d["order"])

    n_pts = len(points)
    n_lines = len(lines)
    conn = []
    offsets = []
    running = 0
    for line in lines:
        conn.extend(line)
        running += len(line)
        offsets.append(running)

    with open(out_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <PolyData>\n')
        f.write(f'    <Piece NumberOfPoints="{n_pts}" NumberOfVerts="0" '
                f'NumberOfLines="{n_lines}" NumberOfStrips="0" NumberOfPolys="0">\n')
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        f.write(' '.join(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}" for p in points))
        f.write('\n        </DataArray>\n')
        f.write('      </Points>\n')
        f.write('      <Lines>\n')
        f.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        f.write(' '.join(str(c) for c in conn))
        f.write('\n        </DataArray>\n')
        f.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        f.write(' '.join(str(o) for o in offsets))
        f.write('\n        </DataArray>\n')
        f.write('      </Lines>\n')
        f.write('      <CellData Scalars="vein_order">\n')
        f.write('        <DataArray type="Int32" Name="vein_order" format="ascii">\n')
        f.write(' '.join(str(o) for o in cell_order))
        f.write('\n        </DataArray>\n')
        f.write('      </CellData>\n')
        f.write('    </Piece>\n')
        f.write('  </PolyData>\n')
        f.write('</VTKFile>\n')


def write_vtp_straight(G, out_path):
    """
    Write a SPARSE-topology .vtp: one straight 2-point line per edge, using
    only the endpoint node coordinates (no intermediate polyline points).
    This is the geometric equivalent of nodes.csv + edges.csv[source,target]
    -- use this if your model/loader expects (p0, p1) edges, not curves.
    """
    node_pts = {n: (float(d["x"]), float(-d["y"]), 0.0) for n, d in G.nodes(data=True)}

    points = []
    idx_of_node = {}
    for n, p in node_pts.items():
        idx_of_node[n] = len(points)
        points.append(p)

    lines = []
    cell_order = []
    for u, v, d in G.edges(data=True):
        lines.append([idx_of_node[u], idx_of_node[v]])
        cell_order.append(d["order"])

    n_pts = len(points)
    n_lines = len(lines)
    conn, offsets, running = [], [], 0
    for line in lines:
        conn.extend(line)
        running += len(line)
        offsets.append(running)

    with open(out_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <PolyData>\n')
        f.write(f'    <Piece NumberOfPoints="{n_pts}" NumberOfVerts="0" '
                f'NumberOfLines="{n_lines}" NumberOfStrips="0" NumberOfPolys="0">\n')
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        f.write(' '.join(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}" for p in points))
        f.write('\n        </DataArray>\n')
        f.write('      </Points>\n')
        f.write('      <Lines>\n')
        f.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        f.write(' '.join(str(c) for c in conn))
        f.write('\n        </DataArray>\n')
        f.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        f.write(' '.join(str(o) for o in offsets))
        f.write('\n        </DataArray>\n')
        f.write('      </Lines>\n')
        f.write('      <CellData Scalars="vein_order">\n')
        f.write('        <DataArray type="Int32" Name="vein_order" format="ascii">\n')
        f.write(' '.join(str(o) for o in cell_order))
        f.write('\n        </DataArray>\n')
        f.write('      </CellData>\n')
        f.write('    </Piece>\n')
        f.write('  </PolyData>\n')
        f.write('</VTKFile>\n')


def process(path, out_dir, min_branch_len=0.0, veins=None, straight_vtp=False,
            curvature_tolerance=None, merge_junctions=3.0,
            collinear_angle=170.0, min_spacing=0.0):
    name = os.path.splitext(os.path.basename(path))[0]
    if veins is not None and set(veins) != set(COLOR_ORDER.keys()):
        name += "_" + "-".join(sorted(veins, key=lambda c: COLOR_ORDER[c]))
    t0 = time.time()
    img, masks = load_masks(path)
    skel, order = build_skeleton_and_labels(masks, veins)
    G = trace_graph(skel, order)

    if min_branch_len > 0:
        # prune tiny spur edges attached to a degree-1 node (skeletonization artifacts)
        to_remove = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
                     if d["length"] < min_branch_len and (G.degree(u) == 1 or G.degree(v) == 1)]
        G.remove_edges_from(to_remove)
        G.remove_nodes_from(list(nx.isolates(G)))
        # Pruning a spur off a 3-way junction leaves that junction at degree 2,
        # i.e. a node sitting mid-vein splitting one vein into two edges.
        # Merge those back into a single edge.
        for n in [n for n in G.nodes() if G.degree(n) == 2]:
            (a, _, da), (b, _, db) = [(x, y, d) for x, y, d in G.edges(n, data=True)]
            if a == b:
                continue
            pa, pb = da["points"], db["points"]
            npix = (G.nodes[n]["y"], G.nodes[n]["x"])
            if pa[0] == npix:
                pa = pa[::-1]
            if pb[-1] == npix:
                pb = pb[::-1]
            G.remove_node(n)
            G.add_edge(a, b, points=pa + pb[1:],
                       length=da["length"] + db["length"],
                       order=da["order"])

    out_prefix = os.path.join(out_dir, name)

    if merge_junctions > 0:
        G = merge_junction_clusters(G, merge_junctions)

    write_vtp(G, out_prefix + ".vtp")  # always full-resolution, for visual QC

    G_out = G
    if curvature_tolerance is not None:
        G_out = densify_with_shape_nodes(G, curvature_tolerance, collinear_angle,
                                         min_spacing)

    write_csv(G_out, out_prefix)
    if straight_vtp:
        write_vtp_straight(G_out, out_prefix + "_straight.vtp")
    n_shape = sum(1 for n in G_out.nodes() if G_out.degree(n) == 2)
    print(f"{name}: {G_out.number_of_nodes()} nodes "
          f"({n_shape} shape) , {G_out.number_of_edges()} edges "
          f"({time.time()-t0:.1f}s)")
    return G_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("out_dir")
    ap.add_argument("--min-branch-len", type=float, default=3.0)
    ap.add_argument("--veins", type=str, default=None,
                     help="Comma-separated subset of colors to extract, e.g. "
                          "'red,green' to drop blue/minor veins. Default: all.")
    ap.add_argument("--straight-vtp", action="store_true",
                     help="Also write <name>_straight.vtp with 2-point (p0,p1) "
                          "edges instead of full polylines -- for models that "
                          "only support straight node-to-node edges.")
    ap.add_argument("--min-node-spacing", type=float, default=0.0,
                     help="Drop shape nodes closer than this many px to a "
                          "neighbouring node (e.g. crowding a junction), when it "
                          "does not break the curvature tolerance. 0 disables.")
    ap.add_argument("--collinear-angle", type=float, default=170.0,
                     help="Drop shape nodes whose turning angle exceeds this many "
                          "degrees (nearly straight => no shape info), when it does "
                          "not break the curvature tolerance. 180 disables.")
    ap.add_argument("--merge-junctions", type=float, default=3.0,
                     help="Collapse clusters of branch-point nodes closer than "
                          "this many px into a single node (fixes the 8-connectivity "
                          "junction artifact). Set 0 to disable.")
    ap.add_argument("--curvature-tolerance", type=float, default=None,
                     help="If set, insert degree-2 'shape' nodes along curved "
                          "edges (Douglas-Peucker) so the resulting straight "
                          "(p0,p1) edges stay within this many pixels of the "
                          "true centerline. Omit to keep only junction/endpoint "
                          "nodes (old behavior). Typical: 1.5-3.0.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    veins = args.veins.split(",") if args.veins else None
    process(args.input, args.out_dir, args.min_branch_len, veins=veins,
            straight_vtp=args.straight_vtp,
            curvature_tolerance=args.curvature_tolerance,
            merge_junctions=args.merge_junctions,
            collinear_angle=args.collinear_angle,
            min_spacing=args.min_node_spacing)
