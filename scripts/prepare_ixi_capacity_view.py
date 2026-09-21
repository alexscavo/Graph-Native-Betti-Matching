#!/usr/bin/env python3
"""Make a lightweight IXI-only paired benchmark view and inspect training counts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.make_ixi_query_benchmark_dataset import evenly_spaced, patch_paths


def prepare(root: Path, view: Path, report: Path, *, maximum_nodes: int = 192) -> dict:
    with (root / "patch_index.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    training_view = json.loads((root / "training_view.json").read_text())
    selected = []
    for split, limit in (("train", 512), ("val", 128)):
        candidates = [row for row in rows if row["split"] == split and
                      0 < int(row["node_count"]) <= maximum_nodes and
                      int(row["edge_count"]) > 0 and int(row["foreground_voxels"]) > 0]
        if len(candidates) < limit:
            raise ValueError(f"Only {len(candidates)} eligible {split} patches")
        selected.extend(evenly_spaced(candidates, limit))
    for row in selected:
        for source, part in zip(patch_paths(root, row), ("raw", "seg", "vtp")):
            if not source.is_file():
                raise FileNotFoundError(source)
            dest = view / row["split"] / part / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink():
                if dest.resolve() != source.resolve():
                    raise FileExistsError(dest)
            elif dest.exists():
                raise FileExistsError(dest)
            else:
                dest.symlink_to(source.resolve())
    report.mkdir(parents=True, exist_ok=True)
    with (report / "benchmark_selection.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    result = {
        "source": str(root), "paired_192_256_view": str(view),
        "paired_train": 512, "paired_val": 128,
        "maximum_paired_nodes": max(int(row["node_count"]) for row in selected),
        "grid_counts": training_view["grid_patches_by_split"],
        "active_train": training_view["active_train_patches"],
        "excluded_train": training_view["excluded_train_patches"],
        "overflow_train_over_192": sum(row["split"] == "train" and
                                        192 < int(row["node_count"]) <= 256 for row in rows),
    }
    (report / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.patch_root, args.view, args.report), indent=2))


if __name__ == "__main__":
    main()
