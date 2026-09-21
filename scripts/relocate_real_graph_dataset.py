#!/usr/bin/env python3
"""Move validated production vessel graphs and patches into dataset folders.

The operation is resumable: verify each copied graph byte-for-byte before
removing its original, and move each already-audited patch on the same fsn1
filesystem via rename. No original segmentation or image is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from collections import Counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_real_graph_patch_sources import EXPECTED_CONFIGURATION, GRAPH_FILES


POLICY = "optimal_radius_0p75x_c095"
REPRESENTATIONS = ("adaptive", "dense", "junction_only")
DATASET_DIRS = {
    "ixi": "IXI",
    "topbrain": "TopBrain_Data_Release_Batches1n2_081425",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_checked(source: Path, target: Path) -> None:
    if not source.is_file():
        if target.is_file():
            return
        raise FileNotFoundError(f"neither original nor relocated file exists: {source}, {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source_hash = digest(source)
    if target.is_file():
        if digest(target) != source_hash:
            raise ValueError(f"existing destination differs from source: {target}")
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        if digest(temporary) != source_hash:
            raise ValueError(f"copied graph checksum mismatch: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def move_checked(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        if target.exists():
            raise FileExistsError(f"patch exists in source and destination: {source}, {target}")
        os.replace(source, target)
    elif not target.is_file():
        raise FileNotFoundError(f"missing patch in source and destination: {source}, {target}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cleanup-originals", action="store_true")
    args = parser.parse_args()
    if not (args.evidence / "summary.json").is_file():
        raise FileNotFoundError("production audit must pass before relocation")
    qc = json.loads((args.evidence / "summary.json").read_text())
    with args.manifest.open(newline="") as stream:
        volumes = list(csv.DictReader(stream))
    with (args.patch_root / "patch_index.csv").open(newline="") as stream:
        patches = list(csv.DictReader(stream))
    original_summary = json.loads((args.patch_root / "generation_summary.json").read_text())
    if not qc.get("complete") or len(volumes) != 220 or qc["full_volumes"] != 220:
        raise ValueError("the audited full-volume graph inventory is incomplete")
    if (not original_summary.get("complete") or original_summary["completed_patients"] != 220
            or original_summary["patches_indexed"] != len(patches)
            or qc["patches"] != len(patches)):
        raise ValueError("the audited patch inventory is incomplete")

    dataset_roots = {
        key: args.dataset_root / directory for key, directory in DATASET_DIRS.items()
    }
    for root in dataset_roots.values():
        if not root.is_dir():
            raise FileNotFoundError(f"original dataset root is missing: {root}")

    records = json.loads((args.source_root / "source_manifest.json").read_text())["subjects"]
    by_patient = {record["patient_id"]: record for record in records}
    if len(records) != len(volumes):
        raise ValueError("source provenance / volume count mismatch")
    patch_groups: dict[str, list[dict]] = {key: [] for key in DATASET_DIRS}
    for patch in patches:
        patch_groups[by_patient[patch["patient_id"]]["dataset"]].append(patch)

    for position, volume in enumerate(volumes, 1):
        dataset = volume["dataset"]
        patient = f"{int(volume['index']):06d}"
        record = by_patient[patient]
        if record["subject"] != volume["subject"] or record["dataset"] != dataset:
            raise ValueError(f"source provenance mismatch for {patient}")
        relative = Path(dataset) / volume["modality"] / volume["split"] / volume["subject"]
        source = args.graph_root / relative
        target = (
            dataset_roots[dataset] / "vascular_graphs" / POLICY
            / volume["modality"] / volume["split"] / volume["subject"]
        )
        marker_source = source / "complete.json"
        if marker_source.is_file():
            marker = json.loads(marker_source.read_text())
            if any(marker["configuration"].get(key) != value for key, value in EXPECTED_CONFIGURATION.items()):
                raise ValueError(f"unexpected graph policy: {volume['subject']}")
        else:
            marker = json.loads((target / "complete.json").read_text())
        if str(marker["index"]) != volume["index"]:
            raise ValueError(f"graph marker index mismatch: {volume['subject']}")
        for name in REPRESENTATIONS:
            for filename in GRAPH_FILES:
                copy_checked(source / "graphs" / name / filename, target / "graphs" / name / filename)
        copy_checked(marker_source, target / "complete.json")
        record["graph"] = str((target / "graphs" / "adaptive").resolve())
        record["completion_marker"] = str((target / "complete.json").resolve())
        if position % 25 == 0 or position == len(volumes):
            print(f"verified/copied full-volume graphs {position}/{len(volumes)}", flush=True)

    # Finish the entire graph stage before moving any already-audited patch.
    for dataset, subset in patch_groups.items():
        output = dataset_roots[dataset] / "vascular_patches" / POLICY
        for position, patch in enumerate(subset, 1):
            split, sample = patch["split"], patch["sample_id"]
            for folder, suffix in (
                ("raw", "_data.nii.gz"),
                ("seg", "_seg.nii.gz"),
                ("vtp", "_graph.vtp"),
            ):
                filename = f"{sample}{suffix}"
                move_checked(
                    args.patch_root / split / folder / filename,
                    output / split / folder / filename,
                )
            if position % 2500 == 0 or position == len(subset):
                print(f"{dataset}: relocated patch triplets {position}/{len(subset)}", flush=True)

    # Copy small metadata to both self-contained patch datasets. Preserve the
    # original patient IDs so VTP files and source provenance remain joinable.
    with (args.patch_root / "patient_features.csv").open(newline="") as stream:
        feature_rows = list(csv.DictReader(stream))
    generation_config = json.loads((args.patch_root / "generation_config.json").read_text())
    for dataset, subset in patch_groups.items():
        root = dataset_roots[dataset]
        output = root / "vascular_patches" / POLICY
        patients = {record["patient_id"] for record in records if record["dataset"] == dataset}
        dataset_records = [record for record in records if record["dataset"] == dataset]
        for record in dataset_records:
            patient = record["patient_id"]
            for filename in (f".complete/{patient}.json", f".manifests/{patient}.csv"):
                move_checked(args.patch_root / filename, output / filename)
        write_csv(output / "patch_index.csv", list(subset[0]), subset)
        write_csv(
            output / "patient_features.csv", list(feature_rows[0]),
            [row for row in feature_rows if row["patient_id"] in patients],
        )
        write_csv(
            output / "patient_split.csv", ["patient_id", "split"],
            [{"patient_id": row["patient_id"], "split": row["patch_split"]} for row in dataset_records],
        )
        write_csv(
            root / "vascular_graphs" / POLICY / "extraction_manifest.csv",
            list(volumes[0]), [row for row in volumes if row["dataset"] == dataset],
        )
        write_json(output / "source_manifest.json", {
            "schema_version": 1,
            "graph_policy": EXPECTED_CONFIGURATION,
            "subjects": dataset_records,
        })
        write_json(output / "generation_config.json", generation_config)
        write_json(output / "generation_summary.json", {
            "complete": True,
            "dataset": dataset,
            "graph_policy": POLICY,
            "completed_patients": len(patients),
            "expected_patients": len(patients),
            "patches_indexed": len(subset),
            "original_generation_fingerprint": original_summary["fingerprint"],
            "node_count_max": max(int(row["node_count"]) for row in subset),
            "over_120_nodes": sum(int(row["node_count"]) > 120 for row in subset),
            "over_192_nodes": sum(int(row["node_count"]) > 192 for row in subset),
            "patch_splits": dict(Counter(row["split"] for row in subset)),
        })
        print(f"{dataset}: {len(patients)} subjects, {len(subset)} patches verified", flush=True)

    if args.cleanup_originals:
        # Destructive cleanup is restricted to the exact validated production
        # files listed above; original image/label datasets are never targeted.
        for volume in volumes:
            source = (
                args.graph_root / volume["dataset"] / volume["modality"]
                / volume["split"] / volume["subject"]
            )
            for name in REPRESENTATIONS:
                directory = source / "graphs" / name
                for filename in GRAPH_FILES:
                    (directory / filename).unlink(missing_ok=True)
                directory.rmdir()
            (source / "graphs").rmdir()
            (source / "complete.json").unlink(missing_ok=True)
            source.rmdir()
        for dataset in DATASET_DIRS:
            parent = args.graph_root / dataset
            for modality_dir in list(parent.iterdir()):
                for split_dir in list(modality_dir.iterdir()):
                    split_dir.rmdir()
                modality_dir.rmdir()
            parent.rmdir()
        for split in ("train", "val", "test"):
            for folder in ("raw", "seg", "vtp"):
                (args.patch_root / split / folder).rmdir()
            (args.patch_root / split).rmdir()
        (args.patch_root / ".complete").rmdir()
        (args.patch_root / ".manifests").rmdir()
        for filename in (
            "generation_config.json", "generation_summary.json", "patient_features.csv",
            "split_balance.csv", "patch_index.csv",
        ):
            (args.patch_root / filename).unlink(missing_ok=True)
        args.patch_root.rmdir()
        for folder in ("raw", "seg", "graphs/adaptive"):
            directory = args.source_root / folder
            for link in directory.iterdir():
                if not link.is_symlink():
                    raise ValueError(f"refusing to remove non-symlink source: {link}")
                link.unlink()
            directory.rmdir()
        (args.source_root / "graphs").rmdir()
        for filename in ("source_manifest.csv", "source_manifest.json", "patient_split.csv"):
            (args.source_root / filename).unlink(missing_ok=True)
        args.source_root.rmdir()
        print("removed only validated, relocated production originals/staging", flush=True)

    print(json.dumps({
        "datasets": {
            dataset: {
                "graphs": sum(row["dataset"] == dataset for row in volumes),
                "patches": len(patch_groups[dataset]),
                "graph_root": str(dataset_roots[dataset] / "vascular_graphs" / POLICY),
                "patch_root": str(dataset_roots[dataset] / "vascular_patches" / POLICY),
            }
            for dataset in DATASET_DIRS
        },
        "originals_cleaned": args.cleanup_originals,
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
