import math
import imageio
import pyvista
import numpy as np
import random
import os
import json
import csv
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from torch.nn.functional import conv2d




# ----------------------------
# Patch parameters (unchanged)
# ----------------------------
patch_size = [128, 128, 1]
pad = [5, 5, 0]

total_min = 1
total_max = 0

# Global patch budgets and counters per split
patch_budget = {"train": None, "val": None, "test": None}
patch_count = {"train": 0, "val": 0, "test": 0}

image_id = 1  # global running patch index per split

from PIL import Image  # add this

def _pil_resample():
    try:
        return Image.Resampling.BILINEAR, Image.Resampling.NEAREST
    except AttributeError:
        return Image.BILINEAR, Image.NEAREST

def resize_pair(image: np.ndarray, seg: np.ndarray, out_w: int, out_h: int):
    bilinear, nearest = _pil_resample()
    img = Image.fromarray(image)
    sg  = Image.fromarray(seg)
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


def load_graph_from_json(file_path: Path):
    """
    Returns:
      nodes_xy: (N,2) float32, pixel coords (x,y)
      edges:    (E,2) int64, indices into nodes_xy (not FeatureID)
    """
    
    # open the annotation file
    with open(file_path, 'r') as f:
        ann = json.load(f)
        
    # get the plant image annotations and features
    plant_img = ann["VineImage"][0]
    features = plant_img["VineFeature"][0]
    
    # nodes in ID space
    nodes_by_id = {}
    for feat in features:
        fid = int(feat["FeatureID"])
        nodes_by_id[fid] = tuple(feat["FeatureCoordinates"])
        
    # edges in ID space
    edges_by_id = []
    for feat in features:
        pid = feat.get("ParentID", None)    # ParentID
        if pid is None:
            continue
        
        pid = int(pid)
        cid = int(feat["FeatureID"])  # child ID
    
        edges_by_id.append((pid, cid))

    # convert node ids into indeces to have dense indeces between 0 - N-1
    feat_ids = sorted(nodes_by_id.keys())
    id_to_idx = {fid: idx for idx, fid in enumerate(feat_ids)}
    
    nodes_xy = np.array([nodes_by_id[fid] for fid in feat_ids], dtype=np.float32)
    edges_ix = np.array([[id_to_idx[pid], id_to_idx[cid]] for pid, cid in edges_by_id], dtype=np.int64)
    
    return nodes_xy, edges_ix

# ----------------------------
# Geometry helpers (unchanged)
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


def patch_extract(save_path, image, seg, mesh, split_name, max_patches_for_image=None, device=None, overlap=0.48, debug=None, collapse_degree2=False):
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
            mod_patch_coord_list, mod_patch_edge_list = prune_patch([patch_coordinates], [patch_edge])

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
# Plants-specific I/O
# ----------------------------
def resolve_paths(root_dir: Path, sample_id: str):
    """
    For a sample_id like 'Set00_IMG_3283', returns:
      raw:   plants_3d2cut/raw/Set00_IMG_3283.jpg
      seg:   plants_3d2cut/seg/Set00_IMG_3283_seg.png
      graph: plants_3d2cut/graphs/Set00_IMG_3283_annotation.json
    """
    raw_path = root_dir / "raw" / f"{sample_id}.jpg"
    seg_path = root_dir / "seg" / f"{sample_id}_seg.png"

    # Most common
    graph_path = root_dir / "graphs" / f"{sample_id}_annotation.json"

    # If sample_id already includes _annotation, allow that too
    if not graph_path.exists() and sample_id.endswith("_annotation"):
        graph_path = root_dir / "graphs" / f"{sample_id}.json"

    return raw_path, seg_path, graph_path



def read_split_csv(split_csv: Path):
    """
    Expects CSV columns:
      sample_id,split
    with split in {train,val,test} (case-insensitive).
    """
    splits = {"train": [], "val": [], "test": []}
    with split_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "sample_id" not in row or "split" not in row:
                continue
            sid = (row["sample_id"] or "").strip()
            sp = (row["split"] or "").strip().lower()
            if not sid:
                continue
            if sp in ("train", "training"):
                splits["train"].append(sid)
            elif sp in ("val", "valid", "validation"):
                splits["val"].append(sid)
            elif sp in ("test", "testing"):
                splits["test"].append(sid)
    return splits


def prepare_split_folders(target_dir: Path, split_name: str):
    split_path = target_dir / split_name
    if split_path.exists():
        raise Exception(f"{split_name} folder already exists: {split_path}")
    (split_path / "raw").mkdir(parents=True, exist_ok=False)
    (split_path / "seg").mkdir(parents=True, exist_ok=False)
    (split_path / "vtp").mkdir(parents=True, exist_ok=False)
    return str(split_path) + "/"


def run_split(split_name: str, sample_ids, root_dir: Path, save_path: str, overlap: float, args, debug: bool = False):
    """
    Process all samples for one split until its patch budget is met.
    """
    global image_id, patch_budget, patch_count

    image_id = 1
    num_images = len(sample_ids)

    dbg = {} if debug else None
    saved_total = 0
    for i, sid in enumerate(tqdm(sample_ids, desc=f"Preparing {split_name} data")):
        if patch_count[split_name] >= patch_budget[split_name]:
            break

        raw_path, seg_path, graph_path = resolve_paths(root_dir, sid)

        if not raw_path.exists():
            # skip quietly
            if dbg is not None:
                dbg["missing_raw"] = dbg.get("missing_raw", 0) + 1
            continue
        if not seg_path.exists():
            if dbg is not None:
                dbg["missing_seg"] = dbg.get("missing_seg", 0) + 1
            continue
        if not graph_path.exists():
            if dbg is not None:
                dbg["missing_graph"] = dbg.get("missing_graph", 0) + 1
            continue

        # Load raw RGB image
        try:
            img = imageio.imread(raw_path)
        except Exception:
            # In case of weird extensions or formats, skip
            if dbg is not None:
                dbg["raw_read_fail"] = dbg.get("raw_read_fail", 0) + 1
            continue

        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]

        # Load seg
        try:
            seg = imageio.imread(seg_path)
        except Exception:
            if dbg is not None:
                dbg["seg_read_fail"] = dbg.get("seg_read_fail", 0) + 1
            continue
        if seg.ndim == 3:
            seg = seg[:, :, 0]

        # Load graph
        try:
            nodes_xy, edges = load_graph_from_json(graph_path)
        except Exception:
            if dbg is not None:
                dbg["graph_read_fail"] = dbg.get("graph_read_fail", 0) + 1
            continue

        if nodes_xy is None or edges is None:
            if dbg is not None:
                dbg["graph_none"] = dbg.get("graph_none", 0) + 1
            continue
        if len(nodes_xy) < 2 or len(edges) < 1:
            if dbg is not None:
                dbg["graph_too_small"] = dbg.get("graph_too_small", 0) + 1
            continue
        
        # --- resize-first (to match preprocess_plants.py behavior) ---
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
                # don’t let it shrink below patch size
                out_w = max(out_w, patch_size[1])  # patch_size is [H,W,1]
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

        # edges are indices into nodes_xy
        edges = np.asarray(edges, dtype=np.int64)
        patch_edge = np.concatenate((np.int32(2 * np.ones((edges.shape[0], 1))), edges.astype(np.int32)), axis=1)
        mesh.lines = patch_edge.flatten()

        # Budget distribution (same idea as original)
        remaining_images = max(num_images - i, 1)
        remaining_patches = patch_budget[split_name] - patch_count[split_name]
        max_for_this_image = max(0, math.ceil(remaining_patches / remaining_images))

        saved_from_image = patch_extract(
            save_path, img, seg, mesh,
            split_name=split_name,
            max_patches_for_image=max_for_this_image,
            overlap=overlap,
            debug=dbg,
            collapse_degree2=args.collapse_degree2,
        )
        saved_total += saved_from_image

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

    splits = read_split_csv(Path(args.split_csv))

    # Create folders
    train_path = prepare_split_folders(target_dir, "train")
    val_path = prepare_split_folders(target_dir, "val")
    test_path = prepare_split_folders(target_dir, "test")

    # Run each split
    run_split("train", splits["train"], root_dir, train_path, overlap, args, debug=args.debug)
    run_split("val", splits["val"], root_dir, val_path, overlap, args, debug=args.debug)
    run_split("test", splits["test"], root_dir, test_path, overlap, args, debug=args.debug)


parser = ArgumentParser()
parser.add_argument('--source', required=True, help='Path to plants_3d2cut directory (contains raw/, seg/, graphs/)')
parser.add_argument('--target', required=True, help='Path to output patches directory')
parser.add_argument('--split_csv', required=True, type=str, help='CSV file with columns sample_id,split (train/val/test)')
parser.add_argument('--seed', default=0, type=int, help='Random seed')

parser.add_argument("--downscale_factor", type=float, default=6.0,
                    help="Downscale factor before patching (default 6: 4032x3024 -> 672x504). Set <=1 to disable.")
parser.add_argument("--resize_width", type=int, default=None, help="Override resized width (requires --resize_height).")
parser.add_argument("--resize_height", type=int, default=None, help="Override resized height (requires --resize_width).")

parser.add_argument('--train_patches', type=int, required=True, help='Target number of patches for training split')
parser.add_argument('--val_patches', type=int, required=True, help='Target number of patches for validation split')
parser.add_argument('--test_patches', type=int, required=True, help='Target number of patches for test split')
parser.add_argument('--overlap', type=float, default=0.50, help='Fractional overlap between adjacent patches in [0, 1).')
parser.add_argument('--debug', action='store_true', help='Print per-split skip reasons and counts.')
parser.add_argument('--collapse_degree2', action='store_true',
                    help='Collapse degree-2 nodes after patch extraction (final graph step).')


if __name__ == "__main__":
    args = parser.parse_args([
        '--source', '/data/scavone/plants_3d2cut',
        '--target', '/data/scavone/plants_3d2cut/patches_3d_wo_deg2',
        '--split_csv', '/data/scavone/plants_3d2cut/splits.csv',
        '--overlap', '0.35',
        '--train_patches', '25923',
        '--val_patches', '6421',
        '--test_patches', '6737',
        '--debug',
        '--downscale_factor', '6.0',
        '--collapse_degree2',
    ])
    generate_data(args)
