#!/usr/bin/env python3
"""Audit real segmented vessel patches and graph-based training eligibility."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def audit(patch_roots: list[Path], output: Path, inspect_masks: bool = True) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Counter] = {}
    exceptions = []
    graph_on_empty = []
    for root in patch_roots:
        dataset = root.parent.parent.name
        counts[dataset] = Counter()
        with (root / "patch_index.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                fg = int(row["foreground_voxels"])
                nodes = int(row["node_count"])
                edges = int(row["edge_count"])
                if fg == 0 and (nodes or edges):
                    # Preserve this geometry error in QC; the curator will
                    # exclude the patch from every active split.
                    graph_on_empty.append({
                        "dataset": dataset, "sample_id": row["sample_id"],
                        "split": row["split"], "patient_id": row["patient_id"],
                        "node_count": nodes, "edge_count": edges,
                    })
                category = ("empty_mask" if fg == 0 else
                            "foreground_no_graph" if nodes == 0 or edges == 0 else
                            "graph_positive")
                counts[dataset][category] += 1
                if category != "foreground_no_graph":
                    continue
                example = {"dataset": dataset, "sample_id": row["sample_id"],
                           "split": row["split"], "foreground_voxels": fg,
                           "node_count": nodes, "edge_count": edges,
                           "patient_id": row["patient_id"]}
                if inspect_masks:
                    mask_path = root / row["split"] / "seg" / f"{row['sample_id']}_seg.nii.gz"
                    image = nib.load(mask_path)
                    mask = np.asanyarray(image.dataobj) > 0
                    if np.count_nonzero(mask) != fg:
                        raise ValueError(f"Foreground index disagrees with segmentation: {mask_path}")
                    coords = np.argwhere(mask)
                    nearest_face = np.minimum(coords, np.array(mask.shape) - 1 - coords)
                    example["max_interior_depth_voxels"] = int(nearest_face.min(axis=1).max())
                    example["bounding_box_min"] = coords.min(axis=0).tolist()
                    example["bounding_box_max"] = coords.max(axis=0).tolist()
                exceptions.append(example)

    combined = sum(counts.values(), Counter())
    payload = {"datasets": {name: dict(values) for name, values in counts.items()},
               "overall": dict(combined),
               "graph_on_empty_mask_count": len(graph_on_empty),
               "graph_on_empty_mask_examples": graph_on_empty[:20],
               "foreground_no_graph_interior_depth_at_least": {
                   str(depth): sum(item.get("max_interior_depth_voxels", -1) >= depth
                                   for item in exceptions) for depth in (1, 4, 8, 12)},
               "definition": "graph_positive requires foreground>0, nodes>0, and edges>0"}
    (output / "patch_eligibility.json").write_text(json.dumps(payload, indent=2) + "\n")
    if graph_on_empty:
        with (output / "graph_on_empty_mask.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(graph_on_empty[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(graph_on_empty)
    if exceptions:
        with (output / "foreground_without_graph.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(exceptions[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(exceptions)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels = list(counts)
    display_labels = ["TopBrain" if name.startswith("TopBrain_") else name for name in labels]
    bottom = np.zeros(len(labels))
    for category, color in (("graph_positive", "#177e89"),
                            ("foreground_no_graph", "#e9a23b"),
                            ("empty_mask", "#787c8a")):
        values = np.array([counts[label][category] for label in labels])
        ax.bar(display_labels, values, bottom=bottom, label=category.replace("_", " "), color=color)
        for pos, value in enumerate(values):
            if value:
                ax.text(pos, bottom[pos] + value / 2, f"{value:,}", ha="center", va="center",
                        fontsize=9 if value >= 500 else 8)
        bottom += values
    ax.set_ylabel("Real 64³ vessel patches")
    ax.set_title("Exhaustive sliding grid: training-target eligibility")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "patch_eligibility.png", dpi=170)
    plt.close(fig)
    if exceptions and inspect_masks:
        examples = (min(exceptions, key=lambda row: row["max_interior_depth_voxels"]),
                    max(exceptions, key=lambda row: row["max_interior_depth_voxels"]))
        roots_by_dataset = {root.parent.parent.name: root for root in patch_roots}
        figure, axes = plt.subplots(2, 2, figsize=(8, 8))
        for row_index, item in enumerate(examples):
            root = roots_by_dataset[item["dataset"]]
            path = root / item["split"] / "seg" / f"{item['sample_id']}_seg.nii.gz"
            mask = np.asanyarray(nib.load(path).dataobj) > 0
            for column, axis in enumerate((2, 1)):
                axes[row_index, column].imshow(mask.max(axis=axis), cmap="magma", vmin=0, vmax=1,
                                                origin="lower", interpolation="nearest")
                axes[row_index, column].set_title(
                    f"{item['sample_id']}  MIP axis {axis}\n"
                    f"{item['foreground_voxels']:,} mask voxels; interior depth "
                    f"{item['max_interior_depth_voxels']} voxels", fontsize=9)
                axes[row_index, column].set_xlabel("Patch voxel")
                axes[row_index, column].set_ylabel("Patch voxel")
        figure.suptitle("Real foreground crops with no graph edge: boundary vs deeper example")
        figure.tight_layout()
        figure.savefig(output / "foreground_without_graph_examples.png", dpi=170)
        plt.close(figure)
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-mask-inspection", action="store_true")
    args = parser.parse_args()
    audit(args.patch_root, args.output, inspect_masks=not args.skip_mask_inspection)


if __name__ == "__main__":
    main()
