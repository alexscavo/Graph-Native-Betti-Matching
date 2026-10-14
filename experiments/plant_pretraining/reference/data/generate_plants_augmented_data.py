#!/usr/bin/env python3
"""
Synthetic plant/tree generator (Option 3: synth from your own graph labels)

What it does
------------
For each existing annotation graph (nodes + edges):
  - Applies random geometric transforms (scale/rotate/shear/jitter)
  - Perturbs topology a bit:
      * branch length scaling / endpoint jitter
      * random pruning of some leaf branches
      * adds a few small offshoot branches
  - Renders a thin skeleton mask (seg)
  - Produces a "raw" image by thickening that mask using the SAME approach as in generate_plants_data.py:
        conv2d with a 3x3 all-ones kernel repeated several times, clip, scale to 0..255

Outputs (per split if you give a split CSV, or all graphs otherwise):
  out/{train,val,test}/
        seg/sample_000001_seg.png   (thin skeleton)
        raw/sample_000001_data.png  (thickened rendering)
        vtp/sample_000001_graph.vtp (graph as VTK PolyData lines)

Notes
-----
- The script does not require external datasets. It only uses your existing graphs (and optionally segs to get canvas size).
- Coordinates: graph JSON provides (x,y). Internally we keep (x,y) in pixels, but VTP points are saved as (y,x,0)
  to match the convention used in generate_plants_data.py (bounds are [ymin,ymax,xmin,xmax,...]).
"""

import os
import json
import math
import random
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import imageio.v2 as imageio
import pyvista
import torch
from torch.nn.functional import conv2d


# ----------------------------
# Shared helpers (compatible with your current pipeline)
# ----------------------------
def configure_threading(num_threads: int = 1):
    """
    Limit CPU thread usage to avoid oversubscription when running multiple workers.
    """
    n = max(1, int(num_threads))
    thread_env = {
        "OMP_NUM_THREADS": str(n),
        "MKL_NUM_THREADS": str(n),
        "OPENBLAS_NUM_THREADS": str(n),
        "NUMEXPR_NUM_THREADS": str(n),
        "VECLIB_MAXIMUM_THREADS": str(n),
        "BLIS_NUM_THREADS": str(n),
    }
    for k, v in thread_env.items():
        os.environ[k] = v
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        # Can fail if threads were already set by another import path.
        pass


def apply_sharding(sample_ids, shard_id: int, num_shards: int):
    if num_shards <= 1:
        return sample_ids
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards - 1}]")
    return sample_ids[shard_id::num_shards]


def split_evenly(items, num_parts: int):
    if num_parts <= 1:
        return [items]
    return [items[i::num_parts] for i in range(num_parts)]
def load_graph_from_json(file_path: Path):
    """
    Returns:
      nodes_xy: (N,2) float32, pixel coords (x,y)
      edges:    (E,2) int64, indices into nodes_xy
    """
    with open(file_path, "r") as f:
        ann = json.load(f)

    plant_img = ann["VineImage"][0]
    features = plant_img["VineFeature"][0]

    nodes_by_id = {}
    for feat in features:
        fid = int(feat["FeatureID"])
        nodes_by_id[fid] = tuple(feat["FeatureCoordinates"])  # (x,y)

    edges_by_id = []
    for feat in features:
        pid = feat.get("ParentID", None)
        if pid is None:
            continue
        pid = int(pid)
        cid = int(feat["FeatureID"])
        edges_by_id.append((pid, cid))

    feat_ids = sorted(nodes_by_id.keys())
    id_to_idx = {fid: idx for idx, fid in enumerate(feat_ids)}

    nodes_xy = np.array([nodes_by_id[fid] for fid in feat_ids], dtype=np.float32)
    edges_ix = np.array([[id_to_idx[pid], id_to_idx[cid]] for pid, cid in edges_by_id], dtype=np.int64)
    return nodes_xy, edges_ix


def vtp_lines_to_edges(vtk_lines: np.ndarray) -> np.ndarray:
    """
    Convert VTK 'lines' connectivity to edge list (E,2).
    Handles polylines: [n, p0, p1, ..., p(n-1), ...] -> edges between consecutive points.
    """
    arr = np.asarray(vtk_lines).ravel()
    edges = []
    i = 0
    L = len(arr)
    while i < L:
        n = int(arr[i])
        i += 1
        pts = arr[i:i + n].astype(np.int64)
        i += n
        if n >= 2:
            edges.append(np.stack([pts[:-1], pts[1:]], axis=1))
    if len(edges) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(edges, axis=0)


def load_graph_from_vtp(file_path: Path):
    """
    Returns:
      nodes_xy: (N,2) float32, pixel coords (x,y)
      edges:    (E,2) int64, indices into nodes_xy
    Note: VTP points are stored as (y,x,0) in this pipeline.
    """
    vtk_data = pyvista.read(str(file_path))
    pts = np.asarray(vtk_data.points, dtype=np.float32)
    nodes_xy = pts[:, [1, 0]]  # swap back to (x,y)
    edges_ix = vtp_lines_to_edges(vtk_data.lines)
    return nodes_xy, edges_ix


def list_sample_ids_from_seg(root_dir: Path, max_inputs: int = 0):
    """
    Build sample_ids by scanning root_dir/seg for *_seg.png.
    If max_inputs > 0, return only the first N (sorted by filename).
    """
    seg_dir = root_dir / "seg"
    if not seg_dir.exists():
        return []
    seg_files = sorted(seg_dir.glob("*_seg.png"))
    if max_inputs > 0:
        seg_files = seg_files[:max_inputs]
    sample_ids = [p.name.replace("_seg.png", "") for p in seg_files]
    return sample_ids


def resolve_paths(root_dir: Path, sample_id: str):
    """
    Same layout as your plants_3d2cut:
      raw:   raw/{sample_id}.jpg
      seg:   seg/{sample_id}_seg.png
      graph: graphs/{sample_id}_annotation.json (or graphs/{sample_id}.json)
    """
    raw_path = root_dir / "raw" / f"{sample_id}.jpg"
    seg_path = root_dir / "seg" / f"{sample_id}_seg.png"
    graph_dir = root_dir / "graphs"
    graph_path = graph_dir / f"{sample_id}_annotation.json"
    if not graph_path.exists():
        graph_path = graph_dir / f"{sample_id}.json"
    if not graph_path.exists():
        graph_path = graph_dir / f"{sample_id}_gt_graph.vtp"
    if not graph_path.exists():
        graph_path = graph_dir / f"{sample_id}.vtp"
    if not graph_path.exists():
        # if sample_id already includes a suffix/extension
        graph_path = graph_dir / sample_id
    if not graph_path.exists():
        # last resort: match any json/vtp with this prefix
        matches = list(graph_dir.glob(f"{sample_id}*.json"))
        if not matches:
            matches = list(graph_dir.glob(f"{sample_id}*.vtp"))
        if matches:
            graph_path = matches[0]
    return raw_path, seg_path, graph_path


# ----------------------------
# Geometry + topology ops (option 3)
# ----------------------------
def _adj_from_edges(n_nodes: int, edges: np.ndarray):
    adj = [[] for _ in range(n_nodes)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _leaf_nodes(n_nodes: int, edges: np.ndarray):
    adj = _adj_from_edges(n_nodes, edges)
    deg = np.array([len(x) for x in adj], dtype=np.int64)
    return np.where(deg == 1)[0], adj, deg


def keep_largest_component(nodes_xy: np.ndarray, edges: np.ndarray):
    """
    Keep only the largest connected component (by node count).
    """
    n = nodes_xy.shape[0]
    if n == 0 or edges.shape[0] == 0:
        return nodes_xy, edges

    adj = _adj_from_edges(n, edges)
    visited = np.zeros((n,), dtype=bool)
    comps = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = [i]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
                    comp.append(v)
        comps.append(comp)

    if not comps:
        return nodes_xy, edges

    comps.sort(key=len, reverse=True)
    keep_idx = np.array(comps[0], dtype=np.int64)
    keep_mask = np.zeros((n,), dtype=bool)
    keep_mask[keep_idx] = True

    old_to_new = -np.ones((n,), dtype=np.int64)
    old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)

    nodes_new = nodes_xy[keep_idx]
    edges_new = np.array(
        [[old_to_new[a], old_to_new[b]] for a, b in edges if keep_mask[a] and keep_mask[b]],
        dtype=np.int64,
    )
    return nodes_new, edges_new


def random_affine(nodes_xy: np.ndarray, canvas_w: int, canvas_h: int, rng: random.Random,
                  scale_range=(0.85, 1.15), rot_deg=20.0, shear=0.08, jitter=8.0):
    """
    Apply a random affine transform around the graph centroid, then translate to fit canvas.
    """
    pts = nodes_xy.astype(np.float32).copy()
    if pts.shape[0] == 0:
        return pts

    cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    pts[:, 0] -= cx
    pts[:, 1] -= cy

    s = rng.uniform(*scale_range)
    th = math.radians(rng.uniform(-rot_deg, rot_deg))
    shx = rng.uniform(-shear, shear)
    shy = rng.uniform(-shear, shear)

    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th),  math.cos(th)]], dtype=np.float32)
    Sh = np.array([[1.0, shx],
                   [shy, 1.0]], dtype=np.float32)

    A = (R @ Sh) * s
    pts = (A @ pts.T).T

    pts[:, 0] += canvas_w / 2.0 + rng.uniform(-jitter, jitter)
    pts[:, 1] += canvas_h / 2.0 + rng.uniform(-jitter, jitter)
    return pts


def perturb_endpoints(nodes_xy: np.ndarray, edges: np.ndarray, rng: random.Random,
                      length_scale_range=(0.85, 1.25), angle_jitter_deg=25.0, prob=0.6):
    """
    For a subset of leaf nodes:
      - scale the parent->leaf vector length
      - jitter its angle
    """
    pts = nodes_xy.astype(np.float32).copy()
    leafs, adj, _deg = _leaf_nodes(pts.shape[0], edges)

    for leaf in leafs.tolist():
        if rng.random() > prob:
            continue
        parent = adj[leaf][0]
        v = pts[leaf] - pts[parent]
        norm = float(np.linalg.norm(v)) + 1e-6
        v /= norm

        # angle jitter
        a = math.radians(rng.uniform(-angle_jitter_deg, angle_jitter_deg))
        ca, sa = math.cos(a), math.sin(a)
        v2 = np.array([ca * v[0] - sa * v[1], sa * v[0] + ca * v[1]], dtype=np.float32)

        # length scaling
        L = norm * rng.uniform(*length_scale_range)
        pts[leaf] = pts[parent] + v2 * L

    return pts


def prune_leaf_branches(nodes_xy: np.ndarray, edges: np.ndarray, rng: random.Random,
                        prune_prob=0.20, max_pruned=6):
    """
    Randomly removes some leaf edges/nodes.
    """
    pts = nodes_xy.astype(np.float32).copy()
    ed = edges.astype(np.int64).copy()

    pruned = 0
    while pruned < max_pruned:
        leafs, adj, _deg = _leaf_nodes(pts.shape[0], ed)
        if leafs.size == 0:
            break

        cand = leafs.tolist()
        rng.shuffle(cand)
        removed_any = False

        for leaf in cand:
            if rng.random() > prune_prob:
                continue
            parent = adj[leaf][0]

            # remove the edge (leaf,parent)
            mask = ~(((ed[:, 0] == leaf) & (ed[:, 1] == parent)) | ((ed[:, 0] == parent) & (ed[:, 1] == leaf)))
            if mask.sum() == ed.shape[0]:
                continue

            ed = ed[mask]
            pruned += 1
            removed_any = True
            if pruned >= max_pruned:
                break

        if not removed_any:
            break

    # drop isolated nodes and reindex
    adj = _adj_from_edges(pts.shape[0], ed)
    keep = np.array([len(a) > 0 for a in adj], dtype=bool)
    if keep.sum() >= 2:
        old_to_new = -np.ones((pts.shape[0],), dtype=np.int64)
        old_to_new[np.where(keep)[0]] = np.arange(int(keep.sum()), dtype=np.int64)
        pts2 = pts[keep]
        ed2 = np.array([[old_to_new[a], old_to_new[b]] for a, b in ed if keep[a] and keep[b]], dtype=np.int64)
        return pts2, ed2
    return pts, ed


def add_offshoots(nodes_xy: np.ndarray, edges: np.ndarray, rng: random.Random,
                  n_new_range=(0, 4), length_range=(12.0, 55.0), angle_range_deg=(20.0, 85.0)):
    """
    Add small new branches off random existing edges.
    """
    pts = nodes_xy.astype(np.float32).copy()
    ed = edges.astype(np.int64).copy()
    if ed.shape[0] == 0:
        return pts, ed

    n_new = rng.randint(int(n_new_range[0]), int(n_new_range[1]))
    for _ in range(n_new):
        eidx = rng.randrange(ed.shape[0])
        a, b = int(ed[eidx, 0]), int(ed[eidx, 1])

        p0, p1 = pts[a], pts[b]
        t = rng.uniform(0.2, 0.8)
        p = (1.0 - t) * p0 + t * p1

        v = p1 - p0
        v /= (np.linalg.norm(v) + 1e-6)

        sign = -1.0 if rng.random() < 0.5 else 1.0
        ang = math.radians(sign * rng.uniform(*angle_range_deg))

        ca, sa = math.cos(ang), math.sin(ang)
        v2 = np.array([ca * v[0] - sa * v[1], sa * v[0] + ca * v[1]], dtype=np.float32)

        L = rng.uniform(*length_range)
        new_pt = p + v2 * L

        new_idx = pts.shape[0]
        pts = np.vstack([pts, new_pt[None, :]])

        if np.linalg.norm(p - p0) < np.linalg.norm(p - p1):
            ed = np.vstack([ed, np.array([[a, new_idx]], dtype=np.int64)])
        else:
            ed = np.vstack([ed, np.array([[b, new_idx]], dtype=np.int64)])

    return pts, ed


def clip_to_canvas(nodes_xy: np.ndarray, edges: np.ndarray, w: int, h: int, margin: int = 2):
    """
    Clip edges to the canvas. If a node falls outside, clip the edge to the border
    and keep the resulting segment so connectivity is preserved.
    """
    pts = nodes_xy.astype(np.float32).copy()
    ed = edges.astype(np.int64).copy()

    xmin, ymin = float(margin), float(margin)
    xmax, ymax = float(w - 1 - margin), float(h - 1 - margin)

    def clip_segment_to_box(x0, y0, x1, y1):
        # Liang-Barsky clipping against axis-aligned box
        dx = x1 - x0
        dy = y1 - y0
        p = [-dx, dx, -dy, dy]
        q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]
        u1, u2 = 0.0, 1.0
        for pi, qi in zip(p, q):
            if pi == 0:
                if qi < 0:
                    return None
            else:
                t = qi / pi
                if pi < 0:
                    if t > u2:
                        return None
                    if t > u1:
                        u1 = t
                else:
                    if t < u1:
                        return None
                    if t < u2:
                        u2 = t
        nx0 = x0 + u1 * dx
        ny0 = y0 + u1 * dy
        nx1 = x0 + u2 * dx
        ny1 = y0 + u2 * dy
        return nx0, ny0, nx1, ny1

    new_nodes = []
    new_edges = []
    node_map = {}

    def get_node_idx(x, y):
        key = (round(float(x), 4), round(float(y), 4))
        if key in node_map:
            return node_map[key]
        idx = len(new_nodes)
        new_nodes.append([float(x), float(y)])
        node_map[key] = idx
        return idx

    for a, b in ed:
        x0, y0 = pts[a, 0], pts[a, 1]
        x1, y1 = pts[b, 0], pts[b, 1]
        clipped = clip_segment_to_box(x0, y0, x1, y1)
        if clipped is None:
            continue
        cx0, cy0, cx1, cy1 = clipped
        i0 = get_node_idx(cx0, cy0)
        i1 = get_node_idx(cx1, cy1)
        if i0 != i1:
            new_edges.append((i0, i1))

    if len(new_nodes) < 2 or len(new_edges) < 1:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.int64)

    return np.array(new_nodes, dtype=np.float32), np.array(new_edges, dtype=np.int64)


def translate_to_fit(nodes_xy: np.ndarray, w: int, h: int, margin: int = 2):
    """
    Translate (and if needed, uniformly scale) the graph to fit inside the canvas.
    """
    pts = nodes_xy.astype(np.float32).copy()
    if pts.shape[0] == 0:
        return pts

    xmin, ymin = float(margin), float(margin)
    xmax, ymax = float(w - 1 - margin), float(h - 1 - margin)

    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    bw = max(1e-6, x1 - x0)
    bh = max(1e-6, y1 - y0)

    avail_w = max(1e-6, xmax - xmin)
    avail_h = max(1e-6, ymax - ymin)

    # If too big, scale uniformly to fit
    scale = min(1.0, avail_w / bw, avail_h / bh)
    if scale < 1.0:
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        pts[:, 0] = (pts[:, 0] - cx) * scale + cx
        pts[:, 1] = (pts[:, 1] - cy) * scale + cy
        x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())

    # Translate to fit inside bounds
    dx = 0.0
    dy = 0.0
    if x0 < xmin:
        dx = xmin - x0
    elif x1 > xmax:
        dx = xmax - x1
    if y0 < ymin:
        dy = ymin - y0
    elif y1 > ymax:
        dy = ymax - y1

    pts[:, 0] += dx
    pts[:, 1] += dy
    return pts


# ----------------------------
# Rasterization + thickening (matches your approach)
# ----------------------------
def draw_line(mask: np.ndarray, x0: float, y0: float, x1: float, y1: float, value: int = 255):
    """
    Simple line rasterization by sampling along the segment.
    """
    h, w = mask.shape
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy))) + 1
    if steps <= 1:
        xi = int(round(x0))
        yi = int(round(y0))
        if 0 <= xi < w and 0 <= yi < h:
            mask[yi, xi] = value
        return

    xs = np.linspace(x0, x1, steps, dtype=np.float32)
    ys = np.linspace(y0, y1, steps, dtype=np.float32)
    xi = np.clip(np.round(xs).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(ys).astype(np.int32), 0, h - 1)
    mask[yi, xi] = value


def render_skeleton(nodes_xy: np.ndarray, edges: np.ndarray, w: int, h: int):
    seg = np.zeros((h, w), dtype=np.uint8)
    for a, b in edges.astype(np.int64):
        x0, y0 = float(nodes_xy[a, 0]), float(nodes_xy[a, 1])
        x1, y1 = float(nodes_xy[b, 0]), float(nodes_xy[b, 1])
        draw_line(seg, x0, y0, x1, y1, value=255)
    return seg


def thicken_like_generate_plants(seg_u8: np.ndarray, n_convs: int = 4):
    """
    Same thickening spirit as your patch_extract (4 conv passes by default).
    """
    x = torch.from_numpy(seg_u8).unsqueeze(0).unsqueeze(0).float() / 255.0
    weights = torch.tensor([[1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.]], dtype=torch.float32).view(1, 1, 3, 3)

    y = x
    for _ in range(int(n_convs)):
        y = conv2d(y, weight=weights, padding=1)

    y = y[0, 0]
    y[y > 1] = 1
    y = (y * 255.0).byte().cpu().numpy()
    return y


# ----------------------------
# Saving
# ----------------------------
def save_sample(out_dir: Path, idx: int, raw_u8: np.ndarray, seg_u8: np.ndarray,
                nodes_xy: np.ndarray, edges: np.ndarray, norm_padding: int = 0):
    out_dir_raw = out_dir / "raw"
    out_dir_seg = out_dir / "seg"
    out_dir_vtp = out_dir / "vtp"
    out_dir_raw.mkdir(parents=True, exist_ok=True)
    out_dir_seg.mkdir(parents=True, exist_ok=True)
    out_dir_vtp.mkdir(parents=True, exist_ok=True)

    raw_name = f"sample_{idx:06d}_data.png"
    seg_name = f"sample_{idx:06d}_seg.png"
    vtp_name = f"sample_{idx:06d}_graph.vtp"

    imageio.imwrite(str(out_dir_raw / raw_name), raw_u8)
    imageio.imwrite(str(out_dir_seg / seg_name), seg_u8)

    # Save graph VTP: points as (y,x,0) normalized to image size (+ optional padding)
    H, W = raw_u8.shape[:2]
    pad = float(max(0, norm_padding))
    denom_x = float(W + 2 * pad)
    denom_y = float(H + 2 * pad)
    nodes_yx = np.stack([nodes_xy[:, 1] + pad, nodes_xy[:, 0] + pad], axis=1).astype(np.float32)
    nodes_yx[:, 0] /= denom_y
    nodes_yx[:, 1] /= denom_x
    pts = np.concatenate([nodes_yx, np.zeros((nodes_yx.shape[0], 1), dtype=np.float32)], axis=1)

    mesh = pyvista.PolyData(pts)
    if edges.shape[0] > 0:
        lines = np.concatenate([2 * np.ones((edges.shape[0], 1), dtype=np.int32), edges.astype(np.int32)], axis=1)
        mesh.lines = lines.flatten()
    mesh.save(str(out_dir_vtp / vtp_name))


# ----------------------------
# Main loop
# ----------------------------
def process_ids(sample_ids, root_dir: Path, out_split_dir: Path, args, seed: int):
    rng = random.Random(seed)
    idx = 1
    stats = {
        "missing_graph": 0,
        "load_fail": 0,
        "too_few_nodes": 0,
        "clip_invalid": 0,
        "min_seg_sum": 0,
        "saved": 0,
    }

    if len(sample_ids) == 0:
        print(f"[prova_piante] No sample_ids found for {out_split_dir}")
        return

    for sid in tqdm(sample_ids, desc=f"[prova_piante] {out_split_dir.name}", unit="sample",
                    disable=(args.num_workers > 1)):
        _raw_path, seg_path, graph_path = resolve_paths(root_dir, sid)
        if not graph_path.exists():
            stats["missing_graph"] += 1
            continue

        try:
            if graph_path.suffix.lower() == ".vtp":
                nodes_xy, edges = load_graph_from_vtp(graph_path)
            else:
                nodes_xy, edges = load_graph_from_json(graph_path)
        except Exception:
            stats["load_fail"] += 1
            continue

        if nodes_xy.shape[0] < args.min_nodes or edges.shape[0] < args.min_edges:
            stats["too_few_nodes"] += 1
            continue

        # canvas size: prefer seg size
        if seg_path.exists():
            try:
                base_seg = imageio.imread(seg_path)
                if base_seg.ndim == 3:
                    base_seg = base_seg[:, :, 0]
                h, w = int(base_seg.shape[0]), int(base_seg.shape[1])
            except Exception:
                h, w = args.canvas_h, args.canvas_w
        else:
            h, w = args.canvas_h, args.canvas_w

        # If graph coords look normalized, scale to pixel space
        if nodes_xy.size > 0:
            n_min = float(nodes_xy.min())
            n_max = float(nodes_xy.max())
            if 0.0 <= n_min and n_max <= 1.5:
                nodes_xy = nodes_xy.copy()
                nodes_xy[:, 0] *= float(w - 1)
                nodes_xy[:, 1] *= float(h - 1)
            if os.getenv("DEBUG_GRAPH", "0") == "1":
                print(f"[prova_piante] {sid} nodes_xy range=({n_min:.4f},{n_max:.4f}) canvas=(H={h},W={w})")

        for _k in range(int(args.synth_per_image)):
            pts = random_affine(nodes_xy, w, h, rng,
                                scale_range=(args.scale_min, args.scale_max),
                                rot_deg=args.rot_deg,
                                shear=args.shear,
                                jitter=args.jitter)

            pts = perturb_endpoints(pts, edges, rng,
                                    length_scale_range=(args.len_scale_min, args.len_scale_max),
                                    angle_jitter_deg=args.endpoint_angle_deg,
                                    prob=args.endpoint_prob)

            pts2, ed2 = prune_leaf_branches(pts, edges, rng,
                                            prune_prob=args.prune_prob,
                                            max_pruned=args.max_pruned)
            if args.keep_connected:
                pts2, ed2 = keep_largest_component(pts2, ed2)

            pts3, ed3 = add_offshoots(pts2, ed2, rng,
                                      n_new_range=(0, args.max_offshoots),
                                      length_range=(args.offshoot_len_min, args.offshoot_len_max),
                                      angle_range_deg=(args.offshoot_ang_min, args.offshoot_ang_max))
            if args.keep_connected:
                pts3, ed3 = keep_largest_component(pts3, ed3)

            if args.clip_mode == "clip":
                pts4, ed4 = clip_to_canvas(pts3, ed3, w=w, h=h, margin=args.clip_margin)
            else:
                pts4 = translate_to_fit(pts3, w=w, h=h, margin=args.clip_margin)
                ed4 = ed3
            if args.keep_connected:
                pts4, ed4 = keep_largest_component(pts4, ed4)
            if pts4.shape[0] < args.min_nodes or ed4.shape[0] < args.min_edges:
                stats["too_few_nodes"] += 1
                continue
            if pts4.shape[0] < 2 or ed4.shape[0] < 1:
                stats["clip_invalid"] += 1
                continue

            seg_u8 = render_skeleton(pts4, ed4, w=w, h=h)
            if int(seg_u8.sum()) < args.min_seg_sum:
                stats["min_seg_sum"] += 1
                continue

            raw_u8 = thicken_like_generate_plants(seg_u8, n_convs=args.thicken_convs)

            save_sample(out_split_dir, idx, raw_u8, seg_u8, pts4, ed4, norm_padding=args.norm_padding)
            if args.debug:
                print(f"[prova_piante][debug] sample_{idx:06d} nodes={pts4.shape[0]} edges={ed4.shape[0]}")
            idx += 1
            stats["saved"] += 1
            if args.max_outputs > 0 and idx > args.max_outputs:
                print(f"[prova_piante] Reached max_outputs={args.max_outputs} for {out_split_dir}")
                print(f"[prova_piante] Stats: {stats}")
                return
    print(f"[prova_piante] Done {out_split_dir} | Stats: {stats}")
    if args.max_outputs > 0 and stats["saved"] < args.max_outputs:
        print(f"[prova_piante] Note: saved {stats['saved']} < max_outputs {args.max_outputs}.")
        if args.max_inputs > 0:
            print(f"[prova_piante] max_inputs={args.max_inputs} limits how many inputs are scanned.")


def _next_index(counter, max_outputs: int):
    if counter is None:
        return None
    with counter.get_lock():
        if max_outputs > 0 and counter.value >= max_outputs:
            return None
        counter.value += 1
        return int(counter.value)


def worker_process(ids_chunk, root_dir: Path, out_split_dir: Path, args, seed: int, counter):
    configure_threading(args.torch_threads)
    rng = random.Random(seed)
    stats = {
        "missing_graph": 0,
        "load_fail": 0,
        "too_few_nodes": 0,
        "clip_invalid": 0,
        "min_seg_sum": 0,
        "saved": 0,
    }
    if len(ids_chunk) == 0:
        return
    for sid in ids_chunk:
        _raw_path, seg_path, graph_path = resolve_paths(root_dir, sid)
        if not graph_path.exists():
            stats["missing_graph"] += 1
            continue
        try:
            if graph_path.suffix.lower() == ".vtp":
                nodes_xy, edges = load_graph_from_vtp(graph_path)
            else:
                nodes_xy, edges = load_graph_from_json(graph_path)
        except Exception:
            stats["load_fail"] += 1
            continue

        if nodes_xy.shape[0] < args.min_nodes or edges.shape[0] < args.min_edges:
            stats["too_few_nodes"] += 1
            continue

        if seg_path.exists():
            try:
                base_seg = imageio.imread(seg_path)
                if base_seg.ndim == 3:
                    base_seg = base_seg[:, :, 0]
                h, w = int(base_seg.shape[0]), int(base_seg.shape[1])
            except Exception:
                h, w = args.canvas_h, args.canvas_w
        else:
            h, w = args.canvas_h, args.canvas_w

        if nodes_xy.size > 0:
            n_min = float(nodes_xy.min())
            n_max = float(nodes_xy.max())
            if 0.0 <= n_min and n_max <= 1.5:
                nodes_xy = nodes_xy.copy()
                nodes_xy[:, 0] *= float(w - 1)
                nodes_xy[:, 1] *= float(h - 1)
            if os.getenv("DEBUG_GRAPH", "0") == "1":
                print(f"[prova_piante] {sid} nodes_xy range=({n_min:.4f},{n_max:.4f}) canvas=(H={h},W={w})")

        for _k in range(int(args.synth_per_image)):
            pts = random_affine(nodes_xy, w, h, rng,
                                scale_range=(args.scale_min, args.scale_max),
                                rot_deg=args.rot_deg,
                                shear=args.shear,
                                jitter=args.jitter)

            pts = perturb_endpoints(pts, edges, rng,
                                    length_scale_range=(args.len_scale_min, args.len_scale_max),
                                    angle_jitter_deg=args.endpoint_angle_deg,
                                    prob=args.endpoint_prob)

            pts2, ed2 = prune_leaf_branches(pts, edges, rng,
                                            prune_prob=args.prune_prob,
                                            max_pruned=args.max_pruned)
            if args.keep_connected:
                pts2, ed2 = keep_largest_component(pts2, ed2)

            pts3, ed3 = add_offshoots(pts2, ed2, rng,
                                      n_new_range=(0, args.max_offshoots),
                                      length_range=(args.offshoot_len_min, args.offshoot_len_max),
                                      angle_range_deg=(args.offshoot_ang_min, args.offshoot_ang_max))
            if args.keep_connected:
                pts3, ed3 = keep_largest_component(pts3, ed3)

            if args.clip_mode == "clip":
                pts4, ed4 = clip_to_canvas(pts3, ed3, w=w, h=h, margin=args.clip_margin)
            else:
                pts4 = translate_to_fit(pts3, w=w, h=h, margin=args.clip_margin)
                ed4 = ed3
            if args.keep_connected:
                pts4, ed4 = keep_largest_component(pts4, ed4)
            if pts4.shape[0] < args.min_nodes or ed4.shape[0] < args.min_edges:
                stats["too_few_nodes"] += 1
                continue
            if pts4.shape[0] < 2 or ed4.shape[0] < 1:
                stats["clip_invalid"] += 1
                continue

            seg_u8 = render_skeleton(pts4, ed4, w=w, h=h)
            if int(seg_u8.sum()) < args.min_seg_sum:
                stats["min_seg_sum"] += 1
                continue

            idx = _next_index(counter, args.max_outputs)
            if idx is None:
                print(f"[prova_piante] Reached max_outputs={args.max_outputs} for {out_split_dir}")
                return

            raw_u8 = thicken_like_generate_plants(seg_u8, n_convs=args.thicken_convs)
            save_sample(out_split_dir, idx, raw_u8, seg_u8, pts4, ed4, norm_padding=args.norm_padding)
            stats["saved"] += 1


def process_ids_parallel(sample_ids, root_dir: Path, out_split_dir: Path, args, seed: int):
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    chunks = split_evenly(sample_ids, args.num_workers)

    procs = []
    for wid, chunk in enumerate(chunks):
        seed_offset = (wid + 1) * 1000003
        p = ctx.Process(target=worker_process, args=(chunk, root_dir, out_split_dir, args, seed + seed_offset, counter))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker exited with code {p.exitcode}")


def main():
    p = ArgumentParser()
    p.add_argument("--source", required=True, help="Path to plants_3d2cut directory (contains raw/, seg/, graphs/)")
    p.add_argument("--target", required=True, help="Output directory (will create train/val/test or 'all')")
    p.add_argument("--split_csv", default=None, help="(ignored) split CSV is no longer used")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synth_per_image", type=int, default=3)
    p.add_argument("--max_inputs", type=int, default=0,
                   help="Limit number of input seg files to process (0 = no limit).")

    # canvas fallback if no seg exists
    p.add_argument("--canvas_w", type=int, default=672)
    p.add_argument("--canvas_h", type=int, default=504)

    # affine
    p.add_argument("--scale_min", type=float, default=0.85)
    p.add_argument("--scale_max", type=float, default=1.15)
    p.add_argument("--rot_deg", type=float, default=20.0)
    p.add_argument("--shear", type=float, default=0.08)
    p.add_argument("--jitter", type=float, default=8.0)

    # endpoint jitter
    p.add_argument("--endpoint_prob", type=float, default=0.60)
    p.add_argument("--len_scale_min", type=float, default=0.85)
    p.add_argument("--len_scale_max", type=float, default=1.25)
    p.add_argument("--endpoint_angle_deg", type=float, default=25.0)

    # pruning
    p.add_argument("--prune_prob", type=float, default=0.20)
    p.add_argument("--max_pruned", type=int, default=6)

    # offshoots
    p.add_argument("--max_offshoots", type=int, default=4)
    p.add_argument("--offshoot_len_min", type=float, default=12.0)
    p.add_argument("--offshoot_len_max", type=float, default=55.0)
    p.add_argument("--offshoot_ang_min", type=float, default=20.0)
    p.add_argument("--offshoot_ang_max", type=float, default=85.0)

    # rendering / filtering
    p.add_argument("--clip_margin", type=int, default=2)
    p.add_argument("--min_seg_sum", type=int, default=200)
    p.add_argument("--thicken_convs", type=int, default=4)
    p.add_argument("--clip_mode", choices=["clip", "fit"], default="clip",
                   help="clip: clip edges to canvas; fit: translate/scale to fit inside.")
    p.add_argument("--max_outputs", type=int, default=0,
                   help="Max number of samples to generate (0 = no limit).")
    p.add_argument("--debug", action="store_true",
                   help="Print number of nodes/edges for each generated sample.")
    p.add_argument("--no_keep_connected", action="store_false", dest="keep_connected",
                   help="Allow disconnected components (default keeps only the largest connected component).")
    p.add_argument("--min_nodes", type=int, default=4,
                   help="Skip samples with fewer nodes than this.")
    p.add_argument("--min_edges", type=int, default=3,
                   help="Skip samples with fewer edges than this.")
    p.add_argument("--norm_padding", type=int, default=0,
                   help="Padding (in pixels) to include when normalizing VTP coords.")
    p.add_argument("--torch_threads", type=int, default=1,
                   help="Number of Torch CPU threads per worker (use 1 to avoid oversubscription).")
    p.add_argument("--num_workers", type=int, default=1,
                   help="Number of worker processes to use (1 = single-process).")
    p.add_argument("--shard_id", type=int, default=0,
                   help="Shard index for manual sharding (0-based).")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Total number of shards for manual sharding.")

    args = p.parse_args([
        '--source', '/data/scavone/plants_3d2cut/patches/train',
        '--target', '/data/scavone/plants_3d2cut/prove',
        '--max_inputs', '1',
        '--synth_per_image', '20',
        '--seed', '12',
        '--min_nodes', '6',
        '--min_edges', '5',
        '--debug',
        '--clip_mode', 'fit',
    ])
    configure_threading(args.torch_threads)
    root_dir = Path(args.source)
    out_dir = Path(args.target)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split_csv:
        print("[prova_piante] split_csv is ignored; scanning root/seg instead.")

    ids = list_sample_ids_from_seg(root_dir, max_inputs=args.max_inputs)
    ids = apply_sharding(ids, args.shard_id, args.num_shards)
    split_out = out_dir / "all"
    if args.num_workers > 1:
        process_ids_parallel(ids, root_dir, split_out, args, seed=args.seed)
    else:
        process_ids(ids, root_dir, split_out, args, seed=args.seed)


if __name__ == "__main__":
    main()
