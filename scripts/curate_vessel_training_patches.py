#!/usr/bin/env python3
"""Archive ineligible vessel patches while retaining the exhaustive grid index.

By default only training patches are curated for backwards compatibility.
``--all-splits`` applies the same rule to validation and test patches.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re


SAMPLE_ID = re.compile(r"sample_[0-9]{6}_[0-9]{4,}")
REQUIRED = {"sample_id", "split", "foreground_voxels", "node_count", "edge_count"}


def read_index(root: Path) -> tuple[list[dict], list[str], str]:
    index = root / "patch_index.csv"
    digest = hashlib.sha256(index.read_bytes()).hexdigest()
    with index.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        if not REQUIRED.issubset(fields):
            raise ValueError(f"Missing required fields in {index}")
        rows = list(reader)
    keys = [(row["split"], row["sample_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate samples in {index}")
    if any(split not in {"train", "val", "test"} or not SAMPLE_ID.fullmatch(sample)
           for split, sample in keys):
        raise ValueError("Unexpected split or unsafe sample ID in patch index")
    return rows, fields, digest


def excluded(row: dict, *, all_splits: bool) -> bool:
    return (all_splits or row["split"] == "train") and (
        int(row["foreground_voxels"]) <= 0 or
        int(row["node_count"]) <= 0 or int(row["edge_count"]) <= 0
    )


def triplets(root: Path, split: str, sample_id: str) -> tuple[tuple[Path, Path], ...]:
    names = (f"{sample_id}_data.nii.gz", f"{sample_id}_seg.nii.gz", f"{sample_id}_graph.vtp")
    return tuple(
        (root / split / part / name, root / f"excluded_{split}" / part / name)
        for part, name in zip(("raw", "seg", "vtp"), names)
    )


def write_csv_atomic(path: Path, fields: list[str], rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def curate(root: Path, *, apply: bool, all_splits: bool = False) -> dict:
    rows, fields, digest = read_index(root)
    rejected = [row for row in rows if excluded(row, all_splits=all_splits)]
    retained = [row for row in rows if not excluded(row, all_splits=all_splits)]
    retained_train = [row for row in retained if row["split"] == "train"]
    reasons = Counter(
        "zero_foreground" if int(row["foreground_voxels"]) == 0 else "foreground_without_graph"
        for row in rejected
    )
    counts = dict(Counter(row["split"] for row in rows))
    active_counts = dict(Counter(row["split"] for row in retained))
    excluded_counts = dict(Counter(row["split"] for row in rejected))
    summary = {
        "schema_version": 1,
        "source_index_sha256": digest,
        "grid_patches_by_split": counts,
        "active_train_patches": len(retained_train),
        "excluded_train_patches": excluded_counts.get("train", 0),
        "active_patches_by_split": active_counts,
        "excluded_patches_by_split": excluded_counts,
        "excluded_reasons": dict(reasons),
        "validation_test_unchanged": not all_splits,
        "archive": {split: str(root / f"excluded_{split}") for split in ("train", "val", "test") if all_splits or split == "train"},
        "note": "Excluded triplets are archived, not deleted. Original patch_index.csv remains the full-grid audit."
    }
    moved = []
    for row in rejected:
        for original, archive in triplets(root, row["split"], row["sample_id"]):
            if original.is_file() and not archive.exists():
                moved.append((original, archive))
            elif not original.exists() and archive.is_file():
                continue  # resume a partially completed move
            else:
                raise ValueError(f"Missing or conflicting sample triplet: {original} / {archive}")
    summary["remaining_files_to_move"] = len(moved)
    if not apply:
        return summary
    for original, archive in moved:
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(original, archive)
    for row in retained:
        sample = row["sample_id"]
        for part, filename in (("raw", f"{sample}_data.nii.gz"),
                               ("seg", f"{sample}_seg.nii.gz"),
                               ("vtp", f"{sample}_graph.vtp")):
            path = root / row["split"] / part / filename
            if not path.is_file():
                raise FileNotFoundError(f"Retained patch missing: {path}")
    for split in (("train", "val", "test") if all_splits else ("train",)):
        write_csv_atomic(root / f"excluded_{split}_index.csv", fields,
                         [row for row in rejected if row["split"] == split])
        write_csv_atomic(root / f"active_{split}_index.csv", fields,
                         [row for row in retained if row["split"] == split])
    summary.pop("remaining_files_to_move")
    temporary = root / f".training_view.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, root / "training_view.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-root", type=Path, action="append", required=True)
    parser.add_argument("--apply", action="store_true", help="move only excluded training triplets")
    parser.add_argument("--all-splits", action="store_true", help="apply the same eligibility rule to train, val and test")
    args = parser.parse_args()
    for root in args.patch_root:
        print(json.dumps({"patch_root": str(root), **curate(root, apply=args.apply, all_splits=args.all_splits)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
