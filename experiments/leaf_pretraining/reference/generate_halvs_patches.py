"""
generate_halvs_patches.py

Cut HALVS leaf images + colour-coded vein masks into the SAME patch format as
Image-to-Graph-Vessel/3d/data/generate_plants_data.py (raw/seg/vtp triplets,
128x128 patches with 5px padding, overlap-based sliding window, pyvista mesh
clipping, per-split patch budgets) so the existing plants PyTorch dataloader
can be reused unchanged for HALVS.

WHAT IS REUSED VERBATIM FROM generate_plants_data.py
------------------------------------------------------
patch_size/pad, angle(), save_input(), prune_patch(), simplify_degree2_nodes(),
patch_extract(), resize_pair(), rescale_graph_xy(). These are copied as-is
(not imported cross-project) to keep this folder self-contained, per its own
README ("copy this folder to the cluster ... nothing else from the original
project is needed"). Do not tweak them without checking the plants script
stays in sync, since the whole point is identical patch semantics.

NOTE: patch_extract's "raw" output is NOT the photo. It computes an image
crop, then overwrites it with a smoothed/dilated version of the seg mask
(a torch conv2d blur applied 4x) before saving. That is upstream behavior,
kept unchanged here on purpose.

WHAT IS HALVS-SPECIFIC
-----------------------
Plants loads a hand-annotated graph JSON directly. HALVS has no such graph --
it must be derived from the RGB colour-coded mask (red=1st order midrib,
green=2nd order, blue=3rd order) by skeletonisation. That happens ONCE PER
LEAF AT FULL RESOLUTION (build_leaf_graph, reusing patchify.py's graph_for_crop
across the whole image instead of a crop) before any resizing, because
downscaling a mask before skeletonising shatters thin veins -- patchify.py's
own docstring documents this finding on this exact dataset. Only node
coordinates are rescaled afterwards, exactly like generate_plants_data.py
rescales its hand-annotated graph coordinates after resizing.

Usage:
    python generate_halvs_patches.py
    python generate_halvs_patches.py --source /path/to/HALVS --target out/
    python generate_halvs_patches.py --limit 3 --debug   # quick smoke test
"""
import math
import os
import random
import sys
import json
from pathlib import Path
from argparse import ArgumentParser

import imageio.v2 as imageio
import numpy as np
import pyvista
import torch
from torch.nn.functional import conv2d
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patchify import ORDER_OF, mask_to_labels, graph_for_crop  # noqa: E402


# ----------------------------
# Patch parameters (identical to generate_plants_data.py)
# ----------------------------
patch_size = [128, 128, 1]
pad = [5, 5, 0]

total_min = 1
total_max = 0

patch_budget = {"train": None, "val": None, "test": None}
patch_count = {"train": 0, "val": 0, "test": 0}

image_id = 1


def _pil_resample():
    try:
        return Image.Resampling.BILINEAR, Image.Resampling.NEAREST
    except AttributeError:
        return Image.BILINEAR, Image.NEAREST


def resize_pair(image: np.ndarray, seg: np.ndarray, out_w: int, out_h: int):
    bilinear, nearest = _pil_resample()
    img = Image.fromarray(image)
    sg = Image.fromarray(seg)
    img_r = np.array(img.resize((out_w, out_h), resample=bilinear))
    seg_r = np.array(sg.resize((out_w, out_h), resample=nearest))
    return img_r, seg_r


def rescale_graph_xy(nodes_xy: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """nodes_xy is (N,2) in (x,y)."""
    if nodes_xy is None:
        return np.zeros((0, 2), dtype=np.float32)
    if nodes_xy.size == 0:
        return nodes_xy.astype(np.float32).reshape(0, 2)
    out = nodes_xy.astype(np.float32).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out


# ----------------------------
# Geometry helpers (unchanged from generate_plants_data.py)
# ----------------------------
def angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(np.clip(dot_product, a_min=-1, a_max=1))


def save_input(path, idx, patch, patch_seg, patch_coord, patch_edge):
    """Save a single patch (image, seg, and graph) to disk."""
    global total_min
    global total_max

    imageio.imwrite(path + 'raw/sample_' + str(idx).zfill(6) + '_data.png', patch)
    imageio.imwrite(path + 'seg/sample_' + str(idx).zfill(6) + '_seg.png', patch_seg)

    patch_edge = np.concatenate((np.int32(2 * np.ones((patch_edge.shape[0], 1))), patch_edge), 1)
    mesh = pyvista.PolyData(patch_coord)

    cur_min = np.min(patch_coord[:, :2])
    cur_max = np.max(patch_coord[:, :2])
    if cur_min < total_min:
        total_min = cur_min
        print(total_min)
    if cur_max > total_max:
        total_max = cur_max
        print(total_max)

    mesh.lines = patch_edge.flatten()
    mesh.save(path + 'vtp/sample_' + str(idx).zfill(6) + '_graph.vtp')


def prune_patch(patch_coord_list, patch_edge_list):
    """Prune patch graphs by removing nearly collinear degree-2 nodes."""
    mod_patch_coord_list = []
    mod_patch_edge_list = []

    for coord, edge in zip(patch_coord_list, patch_edge_list):

        dist_adj = np.zeros((coord.shape[0], coord.shape[0]))
        dist_adj[edge[:, 0], edge[:, 1]] = np.sum((coord[edge[:, 0], :] - coord[edge[:, 1], :]) ** 2, 1)
        dist_adj[edge[:, 1], edge[:, 0]] = np.sum((coord[edge[:, 0], :] - coord[edge[:, 1], :]) ** 2, 1)

        start = True
        node_mask = np.ones(coord.shape[0], dtype=bool)
        while start:
            degree = (dist_adj > 0).sum(1)
            deg_2 = list(np.where(degree == 2)[0])
            if len(deg_2) == 0:
                start = False
            for n, idx in enumerate(deg_2):
                deg_2_neighbor = np.where(dist_adj[idx, :] > 0)[0]

                p1 = coord[idx, :]
                p2 = coord[deg_2_neighbor[0], :]
                p3 = coord[deg_2_neighbor[1], :]
                l1 = p2 - p1
                l2 = p3 - p1
                node_angle = angle(l1, l2) * 180 / math.pi
                if node_angle > 160:
                    node_mask[idx] = False
                    dist_adj[deg_2_neighbor[0], deg_2_neighbor[1]] = np.sum((p2 - p3) ** 2)
                    dist_adj[deg_2_neighbor[1], deg_2_neighbor[0]] = np.sum((p2 - p3) ** 2)

                    dist_adj[idx, deg_2_neighbor[0]] = 0.0
                    dist_adj[deg_2_neighbor[0], idx] = 0.0
                    dist_adj[idx, deg_2_neighbor[1]] = 0.0
                    dist_adj[deg_2_neighbor[1], idx] = 0.0
                    break
                elif n == len(deg_2) - 1:
                    start = False

        new_coord = coord[node_mask, :]
        new_dist_adj = dist_adj[np.ix_(node_mask, node_mask)]
        new_edge = np.array(np.where(np.triu(new_dist_adj) > 0)).T

        mod_patch_coord_list.append(new_coord)
        mod_patch_edge_list.append(new_edge)

    return mod_patch_coord_list, mod_patch_edge_list


def simplify_degree2_nodes(nodes_xyz, edges_idx, max_passes=10):
    """
    Collapse degree-2 nodes by connecting their two neighbors.
    Degree is computed from topology only (same logic as 2D preprocess).
    """
    if nodes_xyz.size == 0 or edges_idx.size == 0:
        return nodes_xyz, edges_idx, np.arange(nodes_xyz.shape[0], dtype=np.int64)

    N = nodes_xyz.shape[0]
    adj = [set() for _ in range(N)]
    for u, v in edges_idx:
        if u == v:
            continue
        u = int(u)
        v = int(v)
        adj[u].add(v)
        adj[v].add(u)

    alive = np.ones(N, dtype=bool)
    changed = True
    passes = 0

    while changed and passes < max_passes:
        changed = False
        passes += 1
        for n in range(N):
            if not alive[n] or len(adj[n]) != 2:
                continue
            u, v = tuple(adj[n])
            if not (alive[u] and alive[v]):
                continue

            adj[u].discard(n)
            adj[v].discard(n)
            adj[u].add(v)
            adj[v].add(u)
            adj[n].clear()
            alive[n] = False
            changed = True

    new_edges = set()
    for u in range(N):
        if not alive[u]:
            continue
        for v in adj[u]:
            if alive[v] and u < v:
                new_edges.add((u, v))
    if new_edges:
        new_edges = np.array(sorted(new_edges), dtype=np.int64)
    else:
        new_edges = np.zeros((0, 2), dtype=np.int64)

    old2new = -np.ones(N, dtype=np.int64)
    keep_idx = np.where(alive)[0]
    old2new[keep_idx] = np.arange(keep_idx.size, dtype=np.int64)
    nodes_new = nodes_xyz[keep_idx]
    if new_edges.size > 0:
        new_edges = np.stack([old2new[new_edges[:, 0]], old2new[new_edges[:, 1]]], axis=1)
    return nodes_new, new_edges, old2new


def patch_extract(save_path, image, seg, mesh, split_name, max_patches_for_image=None, device=None, overlap=0.48, debug=None, collapse_degree2=False, prune_collinear=False):
    """Extract patches from an image and save them, respecting per-split budgets."""
    global image_id, patch_budget, patch_count

    p_h, p_w, _ = patch_size
    pad_h, pad_w, _ = pad

    # effective (unpadded) patch size
    p_h = p_h - 2 * pad_h
    p_w = p_w - 2 * pad_w

    h, w, d = image.shape

    # --- tunable overlap: stride from overlap in [0, 1) ---
    S_h = p_h
    S_w = p_w

    stride_h = int(round(S_h * (1.0 - float(overlap))))
    stride_w = int(round(S_w * (1.0 - float(overlap))))

    stride_h = max(1, min(stride_h, S_h))
    stride_w = max(1, min(stride_w, S_w))

    margin = 5

    start_y_list = list(range(margin, h - margin - p_h + 1, stride_h))
    start_x_list = list(range(margin, w - margin - p_w + 1, stride_w))

    if len(start_y_list) == 0 or len(start_x_list) == 0:
        if debug is not None:
            debug["empty_start_list"] = debug.get("empty_start_list", 0) + 1
        return 0

    last_y = h - margin - p_h
    last_x = w - margin - p_w
    if start_y_list[-1] != last_y:
        start_y_list.append(last_y)
    if start_x_list[-1] != last_x:
        start_x_list.append(last_x)

    num_saved_from_image = 0

    for sy in start_y_list:
        for sx in start_x_list:
            # Global and per-image budget checks
            if patch_budget[split_name] is not None and patch_count[split_name] >= patch_budget[split_name]:
                return num_saved_from_image
            if max_patches_for_image is not None and num_saved_from_image >= max_patches_for_image:
                return num_saved_from_image

            start = np.array((sy, sx, 0))
            end = start + np.array(patch_size) - 1 - 2 * np.array(pad)

            patch = np.pad(
                image[start[0]:start[0] + p_h, start[1]:start[1] + p_w, :],
                ((pad_h, pad_h), (pad_w, pad_w), (0, 0))
            )

            patch_seg = np.pad(
                seg[start[0]:start[0] + p_h, start[1]:start[1] + p_w],
                ((pad_h, pad_h), (pad_w, pad_w))
            )

            # your original "augmentation" / smoothing on seg for patch_list
            input_ = torch.from_numpy(patch_seg).unsqueeze(0).unsqueeze(0).float() / 255.
            weights = torch.tensor([[1., 1., 1.],
                                    [1., 1., 1.],
                                    [1., 1., 1.]])
            weights = weights.view(1, 1, 3, 3).repeat(1, 1, 1, 1)

            x = conv2d(input_, weight=weights, padding=1)
            for _ in range(3):
                x = conv2d(x, weight=weights, padding=1)

            aug_img = x[0][0]
            aug_img[aug_img > 1] = 1
            aug_img *= 255
            patch = aug_img.byte().numpy()

            bounds = [start[0], end[0], start[1], end[1], -0.5, 0.5]
            clipped_mesh = mesh.clip_box(bounds, invert=False)
            patch_coordinates = np.float32(np.asarray(clipped_mesh.points))
            patch_edge = clipped_mesh.cells[np.sum(clipped_mesh.celltypes == 1) * 2:].reshape(-1, 3)

            patch_coord_ind = np.where(
                (np.prod(patch_coordinates >= start, 1) * np.prod(patch_coordinates <= end, 1)) > 0.0
            )
            patch_coordinates = patch_coordinates[patch_coord_ind[0], :]
            patch_edge = [tuple(l) for l in patch_edge[:, 1:] if l[0] in patch_coord_ind[0] and l[1] in patch_coord_ind[0]]

            if len(patch_edge) == 0:
                if debug is not None:
                    debug["no_patch_edges"] = debug.get("no_patch_edges", 0) + 1
                continue

            temp = np.array(patch_edge).flatten()
            temp = [np.where(patch_coord_ind[0] == ind_) for ind_ in temp]
            patch_edge = np.array(temp).reshape(-1, 2)

            if patch_coordinates.shape[0] < 2 or patch_edge.shape[0] < 1 or patch_coordinates.shape[0] > 80:
                if debug is not None:
                    debug["invalid_patch_graph"] = debug.get("invalid_patch_graph", 0) + 1
                continue

            patch_coordinates = (patch_coordinates - start + np.array(pad)) / np.array(patch_size)
            # prune_patch's collinearity test (angle > 160deg) is tuned for plants' hand-annotated
            # graphs, which carry no deliberate curvature nodes. Here shape nodes were placed by
            # densify_with_shape_nodes() specifically to keep curved veins within
            # --curvature-tolerance, and a coarse per-node angle test throws most of them away
            # regardless of --collapse_degree2 -- defeating the very feature config.json declares
            # active. Off by default; opt in with --prune-collinear.
            if prune_collinear:
                mod_patch_coord_list, mod_patch_edge_list = prune_patch([patch_coordinates], [patch_edge])
            else:
                mod_patch_coord_list, mod_patch_edge_list = [patch_coordinates], [patch_edge]

            for patch_coord, patch_edge_ in zip(mod_patch_coord_list, mod_patch_edge_list):
                if patch_seg.sum() <= 10:
                    if debug is not None:
                        debug["seg_sum_too_small"] = debug.get("seg_sum_too_small", 0) + 1
                    continue

                # Optional final topological simplification step.
                if collapse_degree2 and patch_edge_.shape[0] > 0:
                    patch_coord, patch_edge_, _ = simplify_degree2_nodes(patch_coord, patch_edge_)
                    if patch_coord.shape[0] < 2 or patch_edge_.shape[0] < 1:
                        if debug is not None:
                            debug["collapsed_graph_too_small"] = debug.get("collapsed_graph_too_small", 0) + 1
                        continue

                if patch_budget[split_name] is not None and patch_count[split_name] >= patch_budget[split_name]:
                    return num_saved_from_image
                if max_patches_for_image is not None and num_saved_from_image >= max_patches_for_image:
                    return num_saved_from_image

                save_input(save_path, image_id, patch, patch_seg, patch_coord, patch_edge_)
                image_id += 1
                patch_count[split_name] += 1
                num_saved_from_image += 1

    return num_saved_from_image


# ----------------------------
# HALVS-specific I/O and graph extraction
# ----------------------------
def find_leaf_ids(root_dir: Path, split: str):
    idir = root_dir / "annotation" / split / "img"
    mdir = root_dir / "annotation" / split / "mask"
    if not idir.is_dir():
        return [], idir, mdir
    names = sorted(n for n in os.listdir(idir) if n.lower().endswith(".png"))
    ids = [n for n in names if (mdir / n).exists()]
    return ids, idir, mdir


def build_leaf_graph(mask_rgb, veins, min_branch_len, merge_junctions,
                     curvature_tolerance, collinear_angle, min_node_spacing):
    """
    Skeletonise the FULL-RESOLUTION mask and trace it into a graph, exactly
    like patchify.py's graph_for_crop does per-crop -- reused here on the
    whole leaf so thin veins are never lost to a premature downscale.

    Returns (nodes_xy, edges_ix, seg) where nodes_xy is (N,2) float32 (x,y)
    pixel coords and edges_ix is (E,2) int64 indices into nodes_xy, matching
    the contract of generate_plants_data.py's load_graph_from_json(). seg is
    the binary 0/255 vein raster, filtered to the same selected vein orders
    as the graph (patchify.py's rationale: the raster and the graph must
    agree on what counts as a vein).
    """
    lab = mask_to_labels(mask_rgb)
    if veins is not None:
        keep = np.zeros_like(lab, dtype=bool)
        for name in veins:
            keep |= (lab == ORDER_OF[name])
        lab = np.where(keep, lab, 0).astype(np.uint8)

    seg = (lab > 0).astype(np.uint8) * 255

    G = graph_for_crop(lab, veins, min_branch_len, merge_junctions,
                       curvature_tolerance, collinear_angle, min_node_spacing)
    if G is None or G.number_of_nodes() == 0:
        return None, None, seg

    ids = sorted(G.nodes())
    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    nodes_xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in ids], dtype=np.float32)
    edges_ix = np.array([[id_to_idx[u], id_to_idx[v]] for u, v in G.edges()], dtype=np.int64)
    return nodes_xy, edges_ix, seg


# Per-shard patch-id offset: sample_NNNNNN numbering is a single running
# counter (see generate_plants_data.py), so N parallel shards writing into the
# SAME split folder would all start at 1 and clobber each other's files. Give
# each shard a private numbering block instead; 200k is far above what any
# realistic shard (even an unsharded full split) will produce.
SHARD_ID_STRIDE = 200_000


def prepare_split_folders(target_dir: Path, split_name: str, sharded: bool = False):
    split_path = target_dir / split_name
    if split_path.exists() and not sharded:
        raise Exception(f"{split_name} folder already exists: {split_path}")
    (split_path / "raw").mkdir(parents=True, exist_ok=sharded)
    (split_path / "seg").mkdir(parents=True, exist_ok=sharded)
    (split_path / "vtp").mkdir(parents=True, exist_ok=sharded)
    return str(split_path) + "/"


def run_split(split_name: str, leaf_ids, idir: Path, mdir: Path, save_path: str,
             overlap: float, args, debug: bool = False, manifest_rows=None, id_start=1):
    """Process all leaves for one split, mirroring generate_plants_data.py's run_split."""
    global image_id, patch_budget, patch_count

    image_id = id_start
    num_images = len(leaf_ids)

    dbg = {} if debug else None
    saved_total = 0

    def record_skip(n):
        if manifest_rows is not None:
            manifest_rows.append({
                "split": split_name, "leaf": Path(n).stem, "patches": 0,
                "first_sample_id": "", "last_sample_id": "",
            })

    for i, n in enumerate(tqdm(leaf_ids, desc=f"Preparing {split_name} data")):
        if patch_budget[split_name] is not None and patch_count[split_name] >= patch_budget[split_name]:
            break

        img_path = idir / n
        mask_path = mdir / n

        try:
            img = imageio.imread(img_path)
        except Exception:
            if dbg is not None:
                dbg["raw_read_fail"] = dbg.get("raw_read_fail", 0) + 1
            record_skip(n)
            continue
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]

        try:
            mask_rgb = imageio.imread(mask_path)
        except Exception:
            if dbg is not None:
                dbg["mask_read_fail"] = dbg.get("mask_read_fail", 0) + 1
            record_skip(n)
            continue
        if mask_rgb.ndim == 2:
            mask_rgb = np.stack([mask_rgb, mask_rgb, mask_rgb], axis=-1)
        elif mask_rgb.ndim == 3 and mask_rgb.shape[2] > 3:
            mask_rgb = mask_rgb[:, :, :3]

        if img.shape[:2] != mask_rgb.shape[:2]:
            if dbg is not None:
                dbg["size_mismatch"] = dbg.get("size_mismatch", 0) + 1
            record_skip(n)
            continue

        nodes_xy, edges, seg = build_leaf_graph(
            mask_rgb, args.veins, args.min_branch_len, args.merge_junctions,
            args.curvature_tolerance, args.collinear_angle, args.min_node_spacing)

        if nodes_xy is None or edges is None:
            if dbg is not None:
                dbg["graph_none"] = dbg.get("graph_none", 0) + 1
            record_skip(n)
            continue
        if len(nodes_xy) < 2 or len(edges) < 1:
            if dbg is not None:
                dbg["graph_too_small"] = dbg.get("graph_too_small", 0) + 1
            record_skip(n)
            continue

        # --- resize-first (to match generate_plants_data.py behavior) ---
        resize_w = args.resize_width
        resize_h = args.resize_height
        if (resize_w is None) != (resize_h is None):
            raise ValueError("Use both --resize_width and --resize_height together (or neither).")

        H, W = img.shape[0], img.shape[1]

        if resize_w is not None and resize_h is not None:
            out_w, out_h = int(resize_w), int(resize_h)
        else:
            f = float(args.downscale_factor)
            if f is None or f <= 1.0:
                out_w, out_h = W, H
            else:
                out_w = int(round(W / f))
                out_h = int(round(H / f))
                # don't let it shrink below patch size
                out_w = max(out_w, patch_size[1])
                out_h = max(out_h, patch_size[0])

        if out_w != W or out_h != H:
            sx = out_w / float(W)
            sy = out_h / float(H)

            img, seg = resize_pair(img, seg, out_w, out_h)
            nodes_xy = rescale_graph_xy(nodes_xy, sx, sy)

        # IMPORTANT: patch_extract expects points in (y, x, z) space because it uses bounds [ymin,ymax,xmin,xmax].
        nodes_yx = np.stack([nodes_xy[:, 1], nodes_xy[:, 0]], axis=1).astype(np.float32)

        patch_coord = np.concatenate((nodes_yx, np.zeros((nodes_yx.shape[0], 1), dtype=np.float32)), axis=1)
        mesh = pyvista.PolyData(patch_coord)

        edges = np.asarray(edges, dtype=np.int64)
        patch_edge = np.concatenate((np.int32(2 * np.ones((edges.shape[0], 1))), edges.astype(np.int32)), axis=1)
        mesh.lines = patch_edge.flatten()

        if patch_budget[split_name] is not None:
            remaining_images = max(num_images - i, 1)
            remaining_patches = patch_budget[split_name] - patch_count[split_name]
            max_for_this_image = max(0, math.ceil(remaining_patches / remaining_images))
        else:
            max_for_this_image = None

        leaf_start_id = image_id
        saved_from_image = patch_extract(
            save_path, img, seg, mesh,
            split_name=split_name,
            max_patches_for_image=max_for_this_image,
            overlap=overlap,
            debug=dbg,
            collapse_degree2=args.collapse_degree2,
            prune_collinear=args.prune_collinear,
        )
        saved_total += saved_from_image
        if manifest_rows is not None:
            manifest_rows.append({
                "split": split_name,
                "leaf": Path(n).stem,
                "patches": saved_from_image,
                "first_sample_id": leaf_start_id if saved_from_image else "",
                "last_sample_id": (image_id - 1) if saved_from_image else "",
            })

    if dbg is not None:
        print(f"[{split_name}] saved patches: {saved_total}")
        for k in sorted(dbg.keys()):
            print(f"[{split_name}] {k}: {dbg[k]}")


def generate_data(args):
    global patch_budget, patch_count

    root_dir = Path(args.source)
    target_dir = Path(args.target)
    overlap = args.overlap

    random.seed(args.seed)

    patch_budget["train"] = args.train_patches
    patch_budget["val"] = args.val_patches
    patch_budget["test"] = args.test_patches

    patch_count["train"] = 0
    patch_count["val"] = 0
    patch_count["test"] = 0

    target_dir.mkdir(parents=True, exist_ok=True)

    sharded = args.num_shards > 1
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"--shard-index must be in [0, {args.num_shards})")

    cfg = dict(vars(args))
    cfg["argv"] = " ".join(sys.argv)
    if args.shard_index == 0:
        with open(target_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)

    manifest_rows = []
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split_name in splits:
        leaf_ids, idir, mdir = find_leaf_ids(root_dir, split_name)
        if not leaf_ids:
            print(f"!! no leaves found for split '{split_name}' under {root_dir / 'annotation' / split_name}")
            continue
        if sharded:
            leaf_ids = leaf_ids[args.shard_index::args.num_shards]
        if args.limit:
            leaf_ids = leaf_ids[:args.limit]
        shard_tag = f" (shard {args.shard_index}/{args.num_shards})" if sharded else ""
        print(f"[{split_name}]{shard_tag} {len(leaf_ids)} leaves")
        save_path = prepare_split_folders(target_dir, split_name, sharded=sharded)
        run_split(split_name, leaf_ids, idir, mdir, save_path, overlap, args, debug=args.debug,
                  manifest_rows=manifest_rows, id_start=1 + args.shard_index * SHARD_ID_STRIDE)

    if manifest_rows:
        import csv
        man_name = f"patches_manifest.shard{args.shard_index:02d}.csv" if sharded else "patches_manifest.csv"
        with open(target_dir / man_name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["split", "leaf", "patches", "first_sample_id", "last_sample_id"])
            w.writeheader()
            w.writerows(manifest_rows)

        counts = np.array([r["patches"] for r in manifest_rows])
        nonzero = counts[counts > 0]
        partial_tag = f" [shard {args.shard_index}/{args.num_shards} only -- merge shard CSVs for the real total]" if sharded else ""
        print(f"\n[summary]{partial_tag} {len(manifest_rows)} leaves attempted, {len(nonzero)} produced patches, "
              f"{int(counts.sum())} patches total")
        if nonzero.size:
            print(f"[summary] patches/leaf (over leaves that produced any): "
                  f"min={nonzero.min()} med={int(np.median(nonzero))} mean={nonzero.mean():.1f} max={nonzero.max()}")
        for split_name in splits:
            sp = counts_by_split = np.array([r["patches"] for r in manifest_rows if r["split"] == split_name])
            if sp.size:
                print(f"[summary][{split_name}] {sp.size} leaves, {int(sp.sum())} patches total, "
                      f"{sp.sum()/sp.size:.1f} patches/leaf average (incl. 0-patch leaves)")


parser = ArgumentParser()
parser.add_argument('--source', default='/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/HALVS',
                    help='Path to HALVS_DATASET dir (contains annotation/{train,val}/{img,mask})')
parser.add_argument('--target', default='/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/HALVS/patches',
                    help='Path to output patches directory')
parser.add_argument('--splits', default='train,val', help='Comma-separated splits to process')
parser.add_argument('--seed', default=0, type=int, help='Random seed')

parser.add_argument('--num-shards', dest='num_shards', type=int, default=1,
                    help="split each split's leaf list into this many shards (round-robin) so "
                         "N processes can run concurrently, e.g. from a SLURM job that launches "
                         "--shard-index 0..num_shards-1 in parallel against the SAME --target. "
                         "Each shard gets a private sample-id numbering block (SHARD_ID_STRIDE) "
                         "so patch filenames never collide; only shard 0 writes config.json, and "
                         "each shard writes patches_manifest.shard<NN>.csv instead of the "
                         "unsharded patches_manifest.csv (concatenate after all shards finish). "
                         "NOTE: --train-patches/--val-patches/--test-patches budgets are NOT "
                         "coordinated across shards -- each shard would enforce the full budget "
                         "independently. Leave budgets uncapped (the default) when sharding.")
parser.add_argument('--shard-index', dest='shard_index', type=int, default=0,
                    help='which shard this process handles (0-indexed, < --num-shards).')

parser.add_argument("--downscale-factor", dest="downscale_factor", type=float, default=5.42,
                    help="Downscale factor before patching, applied AFTER the graph is traced at "
                         "full resolution (default 5.42: matches patchify.py's settled 640px crop "
                         "at 128px patches with 5px padding -> 118px effective content, "
                         "640/118~=5.42). Set <=1 to disable.")
parser.add_argument("--resize-width", dest="resize_width", type=int, default=None,
                    help="Override resized width (requires --resize-height).")
parser.add_argument("--resize-height", dest="resize_height", type=int, default=None,
                    help="Override resized height (requires --resize-width).")

parser.add_argument('--train-patches', dest='train_patches', type=int, default=None,
                    help='Cap on train split patch count. Default: no cap, take every valid patch.')
parser.add_argument('--val-patches', dest='val_patches', type=int, default=None,
                    help='Cap on val split patch count. Default: no cap.')
parser.add_argument('--test-patches', dest='test_patches', type=int, default=None,
                    help='Cap on test split patch count (only used if "test" is in --splits).')
parser.add_argument('--overlap', type=float, default=0.50, help='Fractional overlap between adjacent patches in [0, 1).')
parser.add_argument('--debug', action='store_true', help='Print per-split skip reasons and counts.')
parser.add_argument('--collapse_degree2', action='store_true',
                    help='Collapse degree-2 nodes after patch extraction (final graph step).')
parser.add_argument('--prune-collinear', dest='prune_collinear', action='store_true',
                    help="Run generate_plants_data.py's angle>160deg collinear-node pruning inside "
                         "patch_extract. OFF by default: it was tuned for plants' hand-annotated "
                         "graphs (no synthetic curvature nodes) and, left on, strips out most of the "
                         "shape nodes --curvature-tolerance just inserted, regardless of "
                         "--collapse_degree2. Leave off to actually get the curvature-preserving "
                         "shape nodes documented in README.md.")
parser.add_argument('--limit', type=int, default=None, help='Process at most N leaves per split (for quick tests).')

# forwarded to graph_for_crop / leaf_to_graph (settled config from patchify.py's README)
parser.add_argument('--veins', default='red,green',
                    help="Comma-separated vein orders to keep, e.g. 'red,green' (default) or "
                         "'red,green,blue'. Filters both the graph and the seg raster.")
parser.add_argument('--min-branch-len', dest='min_branch_len', type=float, default=8.0,
                    help="prune spur edges shorter than this (full-res px)")
parser.add_argument('--merge-junctions', dest='merge_junctions', type=float, default=6.0,
                    help="collapse branch-point clusters closer than this (full-res px)")
parser.add_argument('--curvature-tolerance', dest='curvature_tolerance', type=float, default=6.0,
                    help="RDP tolerance in full-resolution pixels for inserting degree-2 shape "
                         "nodes along curved veins. Set to a falsy value via code if you want to "
                         "disable (None keeps only junction/endpoint nodes).")
parser.add_argument('--collinear-angle', dest='collinear_angle', type=float, default=170.0)
parser.add_argument('--min-node-spacing', dest='min_node_spacing', type=float, default=0.0)


if __name__ == "__main__":
    args = parser.parse_args()
    args.veins = args.veins.split(",") if args.veins else None
    if args.veins:
        bad = set(args.veins) - set(ORDER_OF)
        if bad:
            parser.error(f"unknown vein colour(s) {bad}; choose from {list(ORDER_OF)}")
    generate_data(args)
