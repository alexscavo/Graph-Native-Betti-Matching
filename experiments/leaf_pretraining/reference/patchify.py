"""
patchify.py

Cut HALVS leaf images + their colour-coded vein annotations into paired square
patches, downscaled to a fixed network input size (default 128x128), with the
vein graph extracted at FULL resolution and node coordinates normalised to the
patch.

WHY THE GRAPH IS EXTRACTED BEFORE DOWNSCALING
---------------------------------------------
3rd-order veins are only 2-3 px wide in the source. Skeletonising a mask that
has already been squeezed into 128x128 shatters it: on a single 1024px crop,
extracting at full resolution gives 289 nodes / 386 edges / 3 connected
components, while downscaling first gives 488 (nearest) or 68 (lanczos)
components and loses essentially every branch point. So the mask is cropped,
the graph is traced at 1024px, and only the *coordinates* are rescaled. The
128x128 rasters are the network's input, never the source of the labels.

Because node coordinates are normalised by the patch size, they are scale-free:
multiply by 128 to draw on the downscaled patch, or by 1024 to draw on the
full-resolution crop.

LAYOUT PRODUCED
---------------
    out_dir/
      train/
        img/    <leaf>_r<row>c<col>.png     128x128 RGB  (network input)
        mask/   <leaf>_r<row>c<col>.png     128x128 RGB  (downscaled labels)
        mask_full/ <leaf>_r<row>c<col>.png  1024x1024 RGB (graph source, kept
                                            only with --keep-full-mask)
        graph/  <leaf>_r<row>c<col>_nodes.csv
                <leaf>_r<row>c<col>_edges.csv
      val/  ...
      patches.csv     one row per emitted patch

SETTLED CONFIGURATION (the defaults)
------------------------------------
    crop 640 -> 128 (5x)   red+green   prune 8 / merge 6 / RDP tol 6.0   cap 50 edges

Chosen against a hard model budget of <=50 edges per graph. Measured over the
150 annotated leaves that yields 7,116 patches with median 19 edges, p95 41,
max 50, and zero patches over budget.

Why these values, briefly:
  * crop 640    - 512 wastes the budget (median 17 edges), 1024 busts it 57% of
                  the time. 640 sits at median 27-28 with p95 near 50.
  * red+green   - a connected hierarchy (midrib -> secondaries). Adding blue
                  explodes the graph to ~142 edges; keeping red+blue while
                  dropping green shatters the tertiary net into fragments,
                  because 3rd-order veins attach to 2nd-order ones.
  * tol 6.0     - without shape nodes a curved vein becomes one straight chord,
                  missing the true centerline by up to ~8px at output scale.
  * --max-edges - a hard guarantee; the p95 tail estimate is noisy, so do not
                  rely on it alone.

Usage:
    python patchify.py DATASET_ROOT out_dir/               # settled config
    python patchify.py DATASET_ROOT out_dir/ --viz 8       # + QC overlays
    python patchify.py DATASET_ROOT out_dir/ --veins red,green,blue --crop 256
    python patchify.py DATASET_ROOT out_dir/ --no-graph    # rasters only
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leaf_to_graph import (  # noqa: E402
    build_skeleton_and_labels,
    trace_graph,
    merge_junction_clusters,
    densify_with_shape_nodes,
)

# label value -> (colour, vein order name); order value doubles as the label id
ORDER_OF = {"red": 1, "green": 2, "blue": 3}
RGB_OF = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}


# --------------------------------------------------------------------------
# mask <-> label-map helpers
# --------------------------------------------------------------------------
def mask_to_labels(arr):
    """RGB annotation -> uint8 label map (0 = background, 1/2/3 = vein order)."""
    r, g, b = arr[..., 0] > 128, arr[..., 1] > 128, arr[..., 2] > 128
    lab = np.zeros(arr.shape[:2], dtype=np.uint8)
    lab[b & ~r & ~g] = 3
    lab[g & ~r & ~b] = 2
    lab[r & ~g & ~b] = 1
    return lab


def labels_to_rgb(lab):
    out = np.zeros(lab.shape + (3,), dtype=np.uint8)
    for v, c in RGB_OF.items():
        out[lab == v] = c
    return out


def render_mask(lab, mode):
    """
    Label map -> the array actually written to disk.

    "binary" gives a single-channel 0/255 vein mask; the vein ORDER is not lost,
    it stays on every node and edge in the graph CSVs as `vein_order`. Use it
    when the raster is only a foreground target and the orders come from the
    graph. "color" keeps the red/green/blue order coding as an RGB image.
    """
    if mode == "binary":
        return (lab > 0).astype(np.uint8) * 255
    return labels_to_rgb(lab)


def downscale_labels(lab, out_size):
    """
    Downscale a label map so thin veins SURVIVE.

    Nearest-neighbour sampling drops a 2px vein entirely once the factor exceeds
    2, turning continuous veins into dotted lines. Instead each output cell asks
    "did ANY source pixel in this cell belong to order c?", lowest order winning
    ties, so a vein stays a connected line no matter how aggressive the factor.
    Implemented via a BOX (area-average) resize of each class mask followed by a
    >0 test, which handles non-integer scale factors exactly.
    """
    out = np.zeros((out_size, out_size), dtype=np.uint8)
    for v in (3, 2, 1):  # paint coarse orders last so the midrib wins ties
        m = (lab == v).astype(np.uint8) * 255
        if not m.any():
            continue
        small = np.array(Image.fromarray(m).resize((out_size, out_size), Image.BOX))
        out[small > 0] = v
    return out


# --------------------------------------------------------------------------
# patch grid
# --------------------------------------------------------------------------
def patch_boxes(W, H, crop, stride):
    """
    Top-left origins for a `crop`-sized square grid with the given stride.

    The final row/column is anchored flush against the right/bottom border
    rather than padded, so every pixel is covered and no synthetic background is
    invented. Images smaller than `crop` in a dimension yield a single patch
    clamped to that dimension (the caller sees the real crop size back).
    """
    c = min(crop, W, H)

    def starts(extent):
        if extent <= c:
            return [0]
        s = list(range(0, extent - c + 1, stride))
        if s[-1] != extent - c:
            s.append(extent - c)
        return s

    return c, [(x, y) for y in starts(H) for x in starts(W)]


# --------------------------------------------------------------------------
# graph extraction on one full-resolution crop
# --------------------------------------------------------------------------
def graph_for_crop(lab, veins, min_branch_len, merge_junctions,
                   curvature_tolerance, collinear_angle, min_spacing):
    masks = {name: (lab == o) for name, o in ORDER_OF.items()}
    sel = veins if veins else [n for n in ORDER_OF if masks[n].any()]
    sel = [n for n in sel if masks[n].any()]
    if not sel:
        return None

    skel, order = build_skeleton_and_labels(masks, sel)
    if not skel.any():
        return None
    G = trace_graph(skel, order)

    if min_branch_len > 0:
        import networkx as nx
        drop = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
                if d["length"] < min_branch_len and (G.degree(u) == 1 or G.degree(v) == 1)]
        G.remove_edges_from(drop)
        G.remove_nodes_from(list(nx.isolates(G)))
        # Pruning a spur off a 3-way junction leaves that junction at degree 2:
        # a node sitting mid-vein that splits one vein into two edges. Merge the
        # two edges back together, exactly as leaf_to_graph.process() does.
        for n in [n for n in G.nodes() if G.degree(n) == 2]:
            inc = [(x, y, d) for x, y, d in G.edges(n, data=True)]
            if len(inc) != 2:
                continue
            (a, _, da), (b, _, db) = inc
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
                       length=da["length"] + db["length"], order=da["order"])

    if merge_junctions > 0 and G.number_of_nodes():
        G = merge_junction_clusters(G, merge_junctions)
    if curvature_tolerance is not None and G.number_of_nodes():
        G = densify_with_shape_nodes(G, curvature_tolerance, collinear_angle, min_spacing)
    return G


def write_graph_csv(G, size, prefix, border_tol=1.5):
    """
    Node coordinates normalised by the patch size -> [0, 1], y DOWN so they index
    the image array directly (row = y * H). `is_boundary` flags degree-1 nodes
    sitting on the crop border: those are veins that ran out of frame, not real
    vein tips, and a loss on endpoints must be able to tell them apart.
    """
    n_boundary = 0
    with open(prefix + "_nodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "degree", "vein_order", "node_type", "is_boundary"])
        for n, d in G.nodes(data=True):
            deg = G.degree(n)
            ntype = "endpoint" if deg == 1 else ("shape" if deg == 2 else "branch")
            on_border = (d["x"] <= border_tol or d["y"] <= border_tol or
                         d["x"] >= size - 1 - border_tol or d["y"] >= size - 1 - border_tol)
            is_b = int(bool(on_border and deg == 1))
            n_boundary += is_b
            w.writerow([n, round(d["x"] / size, 6), round(d["y"] / size, 6),
                        deg, d["order"], ntype, is_b])

    with open(prefix + "_edges.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "target", "vein_order", "length"])
        for i, (u, v, d) in enumerate(G.edges(data=True)):
            w.writerow([i, u, v, d["order"], round(d["length"] / size, 6)])
    return n_boundary


# --------------------------------------------------------------------------
# per-image driver
# --------------------------------------------------------------------------
def process_pair(img_path, mask_path, split, out_root, args, rows, viz_budget):
    leaf = os.path.splitext(os.path.basename(img_path))[0]
    img = Image.open(img_path).convert("RGB")
    msk = Image.open(mask_path).convert("RGB")
    if img.size != msk.size:
        print(f"  !! {leaf}: image {img.size} != mask {msk.size}, skipped")
        return 0, 0
    W, H = img.size
    lab_full = mask_to_labels(np.array(msk))

    crop, boxes = patch_boxes(W, H, args.crop, args.stride)
    xs = sorted({b[0] for b in boxes})
    ys = sorted({b[1] for b in boxes})

    dirs = {k: os.path.join(out_root, split, k) for k in
            ("img", "mask", "mask_full", "graph", "viz")}
    for k, d in dirs.items():
        if k == "mask_full" and not args.keep_full_mask:
            continue
        if k == "viz" and viz_budget <= 0:
            continue
        os.makedirs(d, exist_ok=True)

    kept = skipped = 0
    for (x0, y0) in boxes:
        r, c = ys.index(y0), xs.index(x0)
        stem = f"{leaf}_r{r:02d}c{c:02d}"
        box = (x0, y0, x0 + crop, y0 + crop)
        lab = lab_full[y0:y0 + crop, x0:x0 + crop]

        # --veins must filter the RASTER too, not just the graph. Otherwise a
        # red+green build ships masks that still contain 3rd-order veins, so the
        # mask and the graph disagree about what a vein is - and the vein-fraction
        # filter would keep patches made entirely of orders we then discard.
        if args.veins is not None:
            keep = np.zeros_like(lab, dtype=bool)
            for name in args.veins:
                keep |= (lab == ORDER_OF[name])
            lab = np.where(keep, lab, 0).astype(np.uint8)

        vein_frac = float((lab > 0).mean())
        if vein_frac < args.min_vein_frac:
            skipped += 1
            continue

        G = None
        n_nodes = n_edges = n_branch = n_boundary = 0
        if not args.no_graph:
            # --max-edge-error is specified at OUTPUT resolution, so convert it
            # into the full-res pixels the RDP simplification actually works in.
            # Done per patch because `crop` is clamped on leaves smaller than it.
            tol = args.curvature_tolerance
            if tol is None and args.max_edge_error:
                tol = args.max_edge_error * crop / args.out_size
            G = graph_for_crop(lab, args.veins, args.min_branch_len,
                               args.merge_junctions, tol,
                               args.collinear_angle, args.min_node_spacing)
            if G is None or G.number_of_nodes() < args.min_nodes:
                skipped += 1
                continue
            if ((args.max_edges is not None and G.number_of_edges() > args.max_edges) or
                    (args.max_nodes is not None and G.number_of_nodes() > args.max_nodes)):
                skipped += 1
                continue
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            n_branch = sum(1 for n in G.nodes() if G.degree(n) >= 3)

        patch_img = img.crop(box).resize((args.out_size, args.out_size), Image.LANCZOS)
        patch_img.save(os.path.join(dirs["img"], stem + ".png"))

        lab_small = downscale_labels(lab, args.out_size)
        Image.fromarray(render_mask(lab_small, args.mask_mode)).save(
            os.path.join(dirs["mask"], stem + ".png"))

        if args.keep_full_mask:
            Image.fromarray(render_mask(lab, args.mask_mode)).save(
                os.path.join(dirs["mask_full"], stem + ".png"))

        if G is not None:
            n_boundary = write_graph_csv(G, crop, os.path.join(dirs["graph"], stem))

        rows.append(dict(
            split=split, leaf=leaf, patch=stem, row=r, col=c,
            src_x=x0, src_y=y0, crop=crop, out_size=args.out_size,
            scale=round(args.out_size / crop, 6), img_w=W, img_h=H,
            vein_frac=round(vein_frac, 6), nodes=n_nodes, edges=n_edges,
            branch=n_branch, boundary_nodes=n_boundary))
        kept += 1

        if viz_budget > 0 and G is not None:
            save_overlay(patch_img, lab_small, G, crop, args.out_size,
                         os.path.join(dirs["viz"], stem + ".png"), stem,
                         args.mask_mode)
            viz_budget -= 1

    return kept, skipped


# --------------------------------------------------------------------------
# QC overlay
# --------------------------------------------------------------------------
def save_overlay(patch_img, lab_small, G, crop, out_size, path, title,
                 mask_mode="binary"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    EDGE_C = {1: "#ff2020", 2: "#20e020", 3: "#4080ff"}
    scale = out_size / crop  # full-res graph coords -> downscaled pixels
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    # show the mask exactly as it is stored, so QC reflects the real artefact
    panels = [(np.array(patch_img), "photo (network input)"),
              (render_mask(lab_small, mask_mode), f"mask ({mask_mode})"),
              (np.array(patch_img) // 3, "graph on photo")]
    for ax, (bg, ttl) in zip(axes, panels):
        ax.imshow(bg, extent=[0, out_size, out_size, 0],
                  cmap=("gray" if bg.ndim == 2 else None))
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    ax = axes[2]
    for u, v, d in G.edges(data=True):
        a, b = G.nodes[u], G.nodes[v]
        ax.plot([a["x"] * scale, b["x"] * scale], [a["y"] * scale, b["y"] * scale],
                color=EDGE_C[d["order"]], lw=1.0, alpha=.95, solid_capstyle="round")
    # Shape nodes (degree 2) are the intermediate vertices that make a curved
    # vein a chain of short straight segments. Drawing them is the only way to
    # see that the smooth-looking curves really are piecewise linear.
    for key, deg_test, col, sz, mk in (("shape", lambda k: k == 2, "#ffffff", 3, "o"),
                                       ("branch", lambda k: k >= 3, "#ffcc00", 9, "o"),
                                       ("endpoint", lambda k: k == 1, "#ff00ff", 9, "s")):
        px = [G.nodes[n]["x"] * scale for n in G.nodes() if deg_test(G.degree(n))]
        py = [G.nodes[n]["y"] * scale for n in G.nodes() if deg_test(G.degree(n))]
        ax.scatter(px, py, s=sz, c=col, marker=mk, zorder=3, linewidths=0,
                   label=f"{key} ({len(px)})")
    ax.legend(fontsize=7, loc="upper right", framealpha=.8)
    ax.set_xlim(0, out_size), ax.set_ylim(out_size, 0)
    fig.suptitle(f"{title}   {crop}px crop -> {out_size}px   "
                 f"{G.number_of_nodes()} nodes / {G.number_of_edges()} edges", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_root",
                    help="HALVS_DATASET dir (containing annotation/{train,val}/{img,mask})")
    ap.add_argument("out_dir")
    ap.add_argument("--crop", type=int, default=640,
                    help="source patch size in ORIGINAL pixels (default 640). At "
                         "640 -> 128 with red+green veins the graph stays inside a "
                         "50-edge budget (median 19, p95 41).")
    ap.add_argument("--out-size", type=int, default=128,
                    help="network input size the patch is downscaled to (default 128)")
    ap.add_argument("--stride", type=int, default=None,
                    help="grid stride in original pixels (default crop//2 = 50%% overlap)")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--veins", default="red,green",
                    help="subset of vein orders to keep, e.g. 'red,green' (default) "
                         "or 'red,green,blue' for all. Filters BOTH the graph and the "
                         "saved mask. red+green is a connected hierarchy (midrib -> "
                         "secondaries); subsets that skip a level, like red+blue, "
                         "shatter into fragments because 3rd-order veins attach to "
                         "2nd-order ones.")
    ap.add_argument("--min-vein-frac", type=float, default=0.002,
                    help="skip patches whose vein pixels fall below this fraction "
                         "(drops off-leaf / background patches). Default 0.002")
    ap.add_argument("--min-nodes", type=int, default=4,
                    help="skip patches whose graph has fewer nodes than this")
    ap.add_argument("--max-edges", type=int, default=50,
                    help="drop patches whose graph exceeds this many edges. Use when "
                         "the model has a hard edge budget - pick a crop size whose "
                         "p95 already sits under the cap so this only trims a small "
                         "tail, otherwise it biases the set toward sparse regions.")
    ap.add_argument("--max-nodes", type=int, default=None,
                    help="same, for nodes")
    ap.add_argument("--no-graph", action="store_true",
                    help="only emit image/mask patches, skip graph extraction")
    ap.add_argument("--mask-mode", choices=("binary", "color"), default="binary",
                    help="'binary' (default) writes a single-channel 0/255 vein "
                         "mask; 'color' keeps the red/green/blue order coding. The "
                         "vein order is NOT lost in binary mode - it stays on every "
                         "node and edge in the graph CSVs as vein_order.")
    ap.add_argument("--keep-full-mask", action="store_true",
                    help="also write the full-resolution mask crop per patch")
    ap.add_argument("--viz", type=int, default=0,
                    help="write this many QC overlay figures per split")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N leaves per split (for quick tests)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split each split's leaf list into this many shards (round-robin) "
                         "so N processes can run concurrently, e.g. from a SLURM job that "
                         "launches --shard-index 0..num_shards-1 in parallel against the SAME "
                         "--out-dir. Patch filenames are already keyed by leaf+row+col so they "
                         "never collide across shards; only patches.csv and config.json need "
                         "shard-aware handling (see --shard-index).")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="which shard this process handles (0-indexed, < --num-shards). With "
                         "--num-shards > 1, writes patches.shard<NN>.csv instead of patches.csv "
                         "(concatenate them after all shards finish), and only shard 0 writes "
                         "config.json.")
    # forwarded to leaf_to_graph
    ap.add_argument("--min-branch-len", type=float, default=8.0,
                    help="prune spur edges shorter than this (full-res px)")
    ap.add_argument("--merge-junctions", type=float, default=6.0,
                    help="collapse branch-point clusters closer than this (full-res px)")
    ap.add_argument("--max-edge-error", type=float, default=None,
                    help="Insert degree-2 'shape' nodes so a straight edge never "
                         "deviates from the true vein centerline by more than this "
                         "many pixels AT OUTPUT RESOLUTION. Without it a curved vein "
                         "between two junctions becomes one straight chord, which on "
                         "these leaves misses by up to ~8px (6%% of the patch). "
                         "1.0 costs ~1.4x more nodes. Set 0 to disable.")
    ap.add_argument("--curvature-tolerance", type=float, default=6.0,
                    help="RDP tolerance in FULL-RESOLUTION pixels, exactly as "
                         "leaf_to_graph.py takes it (default 6.0). Takes precedence "
                         "over --max-edge-error. At crop 640 -> 128 this is 1.2px of "
                         "error at output resolution.")
    ap.add_argument("--collinear-angle", type=float, default=170.0)
    ap.add_argument("--min-node-spacing", type=float, default=0.0)
    args = ap.parse_args()

    if args.stride is None:
        args.stride = args.crop // 2
    args.veins = args.veins.split(",") if args.veins else None
    if args.veins:
        bad = set(args.veins) - set(ORDER_OF)
        if bad:
            ap.error(f"unknown vein colour(s) {bad}; choose from {list(ORDER_OF)}")
    if not (0 <= args.shard_index < args.num_shards):
        ap.error(f"--shard-index must be in [0, {args.num_shards})")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    t0 = time.time()

    # Provenance: a build is worthless if you cannot tell later which settings
    # produced it (e.g. whether shape nodes were inserted, or which orders were
    # kept). Dump the resolved config next to the data, including the derived
    # full-resolution RDP tolerance that --max-edge-error turns into.
    import json
    cfg = dict(vars(args))
    cfg["derived_curvature_tolerance_fullres_px"] = (
        args.curvature_tolerance if args.curvature_tolerance is not None
        else (args.max_edge_error * args.crop / args.out_size
              if args.max_edge_error else None))
    cfg["argv"] = " ".join(sys.argv)
    if args.shard_index == 0:
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print("config: " + " ".join(f"{k}={v}" for k, v in cfg.items()
                                    if k not in ("dataset_root", "out_dir", "argv")))

    for split in args.splits.split(","):
        idir = os.path.join(args.dataset_root, "annotation", split, "img")
        mdir = os.path.join(args.dataset_root, "annotation", split, "mask")
        if not os.path.isdir(idir):
            print(f"!! no such split dir: {idir}")
            continue
        names = sorted(n for n in os.listdir(idir) if n.lower().endswith(".png"))
        if args.num_shards > 1:
            names = names[args.shard_index::args.num_shards]
        if args.limit:
            names = names[:args.limit]
        shard_tag = f" (shard {args.shard_index}/{args.num_shards})" if args.num_shards > 1 else ""
        print(f"\n[{split}]{shard_tag} {len(names)} leaves")
        kept = skipped = 0
        viz_left = args.viz
        for i, n in enumerate(names, 1):
            mp = os.path.join(mdir, n)
            if not os.path.exists(mp):
                print(f"  !! {n}: no matching mask, skipped")
                continue
            k, s = process_pair(os.path.join(idir, n), mp, split, args.out_dir,
                                args, rows, viz_left)
            viz_left = max(0, viz_left - k)
            kept += k
            skipped += s
            print(f"  [{i}/{len(names)}] {n}: +{k} patches ({s} dropped)  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        print(f"[{split}] {kept} patches kept, {skipped} dropped")

    man_name = f"patches.shard{args.shard_index:02d}.csv" if args.num_shards > 1 else "patches.csv"
    man = os.path.join(args.out_dir, man_name)
    if rows:
        with open(man, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {len(rows)} patches + {man}  ({time.time()-t0:.0f}s)")
    if rows:
        nn = np.array([r["nodes"] for r in rows])
        print(f"nodes/patch: min={nn.min()} med={int(np.median(nn))} "
              f"mean={nn.mean():.0f} max={nn.max()}")
        print("NOTE: patches overlap 50%% - split train/val BY LEAF, not by patch.")


if __name__ == "__main__":
    main()
