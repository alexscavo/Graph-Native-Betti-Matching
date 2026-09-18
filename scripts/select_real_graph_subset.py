#!/usr/bin/env python3
"""Select a deterministic geometry/foreground-diverse real-data subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_ALLOCATIONS = {
    ("ixi", "mra", "train"): 6,
    ("topbrain", "mr", "train"): 4,
    ("topbrain", "ct", "train"): 3,
    ("topbrain", "ct", "test"): 2,
}


def descriptors(rows: list[dict[str, str]]) -> np.ndarray:
    values = []
    for row in rows:
        shape = np.asarray([float(row[f"shape_{axis}"]) for axis in "xyz"])
        spacing = np.asarray([float(row[f"spacing_{axis}"]) for axis in "xyz"])
        foreground = float(row["foreground_voxels"])
        voxels = float(np.prod(shape))
        values.append([
            np.log1p(voxels), np.log1p(foreground), np.log(max(foreground / voxels, 1e-12)),
            *np.log(spacing), np.log(spacing.max() / spacing.min()),
            np.log((shape * spacing).prod()),
        ])
    return np.asarray(values, dtype=np.float64)


def maximin(rows: list[dict[str, str]], count: int) -> list[int]:
    if count >= len(rows): return list(range(len(rows)))
    features = descriptors(rows)
    median = np.median(features, axis=0)
    scale = np.subtract(*np.percentile(features, [75, 25], axis=0))
    scale[scale < 1e-12] = 1.0
    normal = (features - median) / scale
    # Begin at the medoid, then repeatedly select the point farthest from its
    # closest selected point. Subject-name ordering resolves exact ties.
    selected = [int(np.argmin(np.linalg.norm(normal, axis=1)))]
    while len(selected) < count:
        distance = np.min(
            np.linalg.norm(normal[:, None, :] - normal[np.asarray(selected)][None, :, :], axis=2),
            axis=1,
        )
        distance[selected] = -1
        best = max(range(len(rows)), key=lambda i: (distance[i], rows[i]["subject"]))
        selected.append(best)
    return selected


def enrich(row: dict[str, str]) -> dict[str, str]:
    label = nib.load(row["label"])
    foreground = int(np.count_nonzero(np.asanyarray(label.dataobj)))
    voxels = int(np.prod(label.shape))
    return {**row, "foreground_voxels": str(foreground),
            "foreground_fraction": f"{foreground / voxels:.10g}"}


def select(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    enriched = [enrich(row) for row in rows]
    selected = []
    for stratum, count in DEFAULT_ALLOCATIONS.items():
        group = [row for row in enriched if tuple(row[key] for key in ("dataset", "modality", "split")) == stratum]
        if len(group) < count: raise ValueError(f"stratum {stratum} has {len(group)} rows, expected >= {count}")
        selected.extend(group[index] for index in maximin(group, count))
    selected.sort(key=lambda row: (row["dataset"], row["modality"], row["split"], row["subject"]))
    for index, row in enumerate(selected):
        row["source_index"], row["index"] = row["index"], str(index)
    return enriched, selected


def write_visual(all_rows, selected, path: Path) -> None:
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    def vector(rows, name): return np.asarray([float(row[name]) for row in rows])
    voxel_volume = lambda rows: np.prod(np.asarray([[float(row[f"spacing_{a}"]) for a in "xyz"] for row in rows]), axis=1)
    panels = [
        (voxel_volume, lambda rows: vector(rows, "foreground_fraction"), "Voxel volume (mm³)", "Foreground fraction"),
        (lambda rows: np.prod(np.asarray([[float(row[f"shape_{a}"]) for a in "xyz"] for row in rows]), axis=1), lambda rows: vector(rows, "foreground_voxels"), "Volume voxels", "Foreground voxels"),
        (lambda rows: vector(rows, "spacing_x"), lambda rows: vector(rows, "spacing_z"), "In-plane spacing (mm)", "Slice spacing (mm)"),
    ]
    for axis, (xfn, yfn, xlabel, ylabel) in zip(axes, panels):
        axis.scatter(xfn(all_rows), yfn(all_rows), s=13, alpha=.25, color="#777777", label="all 220")
        axis.scatter(xfn(selected), yfn(selected), s=42, color="#D55E00", edgecolor="black", linewidth=.4, label="selected 15")
        axis.set(xlabel=xlabel, ylabel=ylabel); axis.grid(alpha=.2)
    axes[0].legend(); figure.suptitle("Representative extraction subset: geometry and vessel-foreground coverage")
    figure.savefig(path, dpi=170); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="") as stream: rows = list(csv.DictReader(stream))
    all_rows, chosen = select(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    fields = list(chosen[0])
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(chosen)
    temporary.replace(args.output)
    summary = {"source_manifest": str(args.manifest.resolve()), "candidate_count": len(rows),
               "selected_count": len(chosen), "allocations": {"/".join(key): value for key, value in DEFAULT_ALLOCATIONS.items()},
               "subjects": [{key: row[key] for key in ("index", "source_index", "dataset", "modality", "split", "subject", "foreground_voxels", "foreground_fraction")} for row in chosen]}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_visual(all_rows, chosen, args.output.with_suffix(".png"))
    print(f"selected {len(chosen)} of {len(rows)} volumes -> {args.output}")


if __name__ == "__main__": main()
