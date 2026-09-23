#!/usr/bin/env python3
"""Stage validated production vessel graphs for exact-boundary patch generation."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_vessel_graphs import (
    DATASET_DIRECTORIES, graph_policy_directory,
)

GRAPH_FILES = ("nodes.csv", "edges.csv", "graph.vvg")
EXPECTED_CONFIGURATION = {
    "schema_version": 3,
    "centerline_backend": "legacy",
    "rdp_voxels": 0.0,
    "radius_fraction": 0.75,
    "simplification_method": "optimal",
    "spur_length": 4,
    "max_junction_extent_mm": 1.5,
    "smooth_iterations": 5,
    "smooth_alpha": 0.5,
    "minimum_chord_fraction": 0.95,
}
SPLITS = ("train", "val", "test")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def link_exact(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if destination.is_symlink():
        if destination.resolve(strict=True) == source:
            return
        raise FileExistsError(f"stale link: {destination} -> {destination.resolve()}")
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.symlink_to(source, target_is_directory=source.is_dir())
    os.replace(temporary, destination)


def split_sizes(patient_count: int) -> dict[str, int]:
    train = int(math.floor(patient_count * 0.70))
    val = int(math.floor(patient_count * 0.15))
    return {"train": train, "val": val, "test": patient_count - train - val}


def _rounded_group_quotas(
    group_sizes: dict[tuple[str, str], int], targets: dict[str, int]
) -> dict[tuple[str, str], dict[str, int]]:
    """Proportionally allocate exact split totals with deterministic rounding."""

    total = sum(group_sizes.values())
    if total != sum(targets.values()):
        raise ValueError(f"group/target total mismatch: {total} != {sum(targets.values())}")
    raw = {
        (group, split): size * targets[split] / total
        for group, size in group_sizes.items()
        for split in SPLITS
    }
    quotas = {
        group: {split: int(math.floor(raw[group, split])) for split in SPLITS}
        for group in group_sizes
    }
    row_remaining = {
        group: size - sum(quotas[group].values())
        for group, size in group_sizes.items()
    }
    column_remaining = {
        split: targets[split] - sum(quotas[group][split] for group in group_sizes)
        for split in SPLITS
    }
    while sum(row_remaining.values()):
        candidates = [
            (raw[group, split] - math.floor(raw[group, split]), group, split)
            for group in group_sizes
            for split in SPLITS
            if row_remaining[group] and column_remaining[split]
        ]
        if not candidates:
            raise AssertionError("could not satisfy stratified split quotas")
        _, group, split = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        quotas[group][split] += 1
        row_remaining[group] -= 1
        column_remaining[split] -= 1
    if any(column_remaining.values()):
        raise AssertionError(f"unfilled split quotas: {column_remaining}")
    return quotas


def assign_splits(records: list[dict], seed: int) -> dict[str, str]:
    """Create an exact 70/15/15 patient split while retaining official tests."""

    targets = split_sizes(len(records))
    assignment = {
        record["patient_id"]: "test"
        for record in records
        if record["source_split"] == "test"
    }
    fixed = Counter(assignment.values())
    remaining_targets = {
        split: targets[split] - fixed.get(split, 0) for split in SPLITS
    }
    if any(value < 0 for value in remaining_targets.values()):
        raise ValueError(f"fixed source splits exceed requested totals: {fixed}")
    eligible: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        if record["patient_id"] in assignment:
            continue
        eligible.setdefault((record["dataset"], record["modality"]), []).append(record)
    quotas = _rounded_group_quotas(
        {group: len(items) for group, items in eligible.items()}, remaining_targets
    )
    for group, items in eligible.items():
        ordered = sorted(
            items,
            key=lambda item: hashlib.sha256(
                f"{seed}:{item['dataset']}:{item['modality']}:{item['subject']}".encode()
            ).hexdigest(),
        )
        offset = 0
        for split in SPLITS:
            count = quotas[group][split]
            for item in ordered[offset : offset + count]:
                assignment[item["patient_id"]] = split
            offset += count
        if offset != len(ordered):
            raise AssertionError(f"split allocation failed for {group}")
    if Counter(assignment.values()) != Counter(targets):
        raise AssertionError(
            f"split totals differ: {Counter(assignment.values())} != {Counter(targets)}"
        )
    return assignment


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    graph_location = parser.add_mutually_exclusive_group(required=True)
    graph_location.add_argument("--graph-root", type=Path, help="legacy combined graph root")
    graph_location.add_argument("--dataset-root", type=Path, help="dataset parent for direct graph output")
    parser.add_argument("--dataset", choices=tuple(DATASET_DIRECTORIES), help="stage only one dataset")
    parser.add_argument("--centerline-backend", choices=("legacy", "vedo_original"), default="legacy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_output = args.split_output or (args.output / "patient_split.csv")

    with args.manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.dataset_root is not None and args.dataset is None:
        parser.error("--dataset is required with --dataset-root")
    if args.dataset is not None:
        rows = [row for row in rows if row["dataset"] == args.dataset]
    if not rows:
        raise ValueError(f"empty manifest: {args.manifest}")

    expected_configuration = {
        **EXPECTED_CONFIGURATION, "centerline_backend": args.centerline_backend,
    }

    records = []
    for row in rows:
        index = int(row["index"])
        patient_id = f"{index:06d}"
        volume_root = (
            args.dataset_root / DATASET_DIRECTORIES[row["dataset"]]
            / "vascular_graphs" / graph_policy_directory(args.centerline_backend)
            / row["modality"] / row["split"] / row["subject"]
            if args.dataset_root is not None else
            args.graph_root / row["dataset"] / row["modality"] / row["split"] / row["subject"]
        )
        marker_path = volume_root / "complete.json"
        if not marker_path.is_file():
            raise FileNotFoundError(f"missing completion marker: {marker_path}")
        marker = json.loads(marker_path.read_text())
        for key in ("index", "dataset", "modality", "split", "subject", "image", "label"):
            if str(marker.get(key)) != str(row[key]):
                raise ValueError(
                    f"{row['subject']}: marker {key}={marker.get(key)!r}, manifest={row[key]!r}"
                )
        configuration = marker.get("configuration", {})
        mismatches = {
            key: (configuration.get(key), expected)
            for key, expected in expected_configuration.items()
            if configuration.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"{row['subject']}: wrong graph policy: {mismatches}")
        graph = volume_root / "graphs" / "adaptive"
        missing = [name for name in GRAPH_FILES if not (graph / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{row['subject']}: missing adaptive graph files {missing}")
        records.append(
            {
                "patient_id": patient_id,
                "manifest_index": index,
                "dataset": row["dataset"],
                "modality": row["modality"],
                "source_split": row["split"],
                "subject": row["subject"],
                "image": str(Path(row["image"]).resolve(strict=True)),
                "label": str(Path(row["label"]).resolve(strict=True)),
                "graph": str(graph.resolve(strict=True)),
                "completion_marker": str(marker_path.resolve(strict=True)),
                "configuration_fingerprint": marker["configuration_fingerprint"],
            }
        )

    patient_ids = [record["patient_id"] for record in records]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("manifest indices do not produce unique patient IDs")
    if split_output.is_file() and args.dataset_root is not None:
        from scripts.audit_synthetic_mri_grid import read_splits
        assignment = read_splits(split_output)
        if set(assignment) != set(patient_ids):
            raise ValueError(f"Existing patient split does not match staged subjects: {split_output}")
        if any(record["source_split"] == "test" and assignment[record["patient_id"]] != "test"
               for record in records):
            raise ValueError("Existing split violates official test subjects")
    else:
        assignment = assign_splits(records, args.seed)
    for directory in (
        args.output / "raw",
        args.output / "seg",
        args.output / "graphs" / "adaptive",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        patient_id = record["patient_id"]
        link_exact(Path(record["image"]), args.output / "raw" / f"{patient_id}.nii.gz")
        link_exact(Path(record["label"]), args.output / "seg" / f"{patient_id}.nii.gz")
        link_exact(
            Path(record["graph"]), args.output / "graphs" / "adaptive" / patient_id
        )
        record["patch_split"] = assignment[patient_id]

    expected_names = set(patient_ids)
    actual_names = {
        path.name for path in (args.output / "graphs" / "adaptive").iterdir()
    }
    if actual_names != expected_names:
        raise ValueError(
            f"staged source set mismatch: extra={sorted(actual_names - expected_names)}, "
            f"missing={sorted(expected_names - actual_names)}"
        )
    split_rows = [
        {"patient_id": record["patient_id"], "split": record["patch_split"]}
        for record in records
    ]
    # A pre-existing split was already validated above. Preserve its bytes:
    # patch reuse fingerprints the CSV, and changing only CRLF to LF would
    # otherwise reject an identical patient assignment.
    if not split_output.is_file():
        _write_csv(split_output, ("patient_id", "split"), split_rows)
    _write_csv(
        args.output / "source_manifest.csv",
        records[0].keys(),
        records,
    )
    payload = {
        "schema_version": 1,
        "graph_policy": expected_configuration,
        "source_manifest": str(args.manifest.resolve()),
        "subjects": records,
        "counts": {
            "volumes": len(records),
            "datasets": dict(Counter(record["dataset"] for record in records)),
            "modalities": dict(Counter(record["modality"] for record in records)),
            "patch_splits": dict(Counter(assignment.values())),
        },
    }
    atomic_write(
        args.output / "source_manifest.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(payload["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
