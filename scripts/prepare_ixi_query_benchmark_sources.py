#!/usr/bin/env python3
"""Assemble an IXI-only source tree from selected real-graph artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re


SUBJECT_PATTERN = re.compile(r"^IXI(\d+)")
OPTIMAL_POLICY = "optimal_radius_0p75x_c095"


def _link(source: Path, destination: Path) -> None:
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        raise FileExistsError(f"stale link: {destination} -> {destination.resolve()}")
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.symlink_to(source, target_is_directory=source.is_dir())
    os.replace(temporary, destination)


def _selected_graph(marker_path: Path, marker: dict) -> Path | None:
    named = marker_path.parent / "graphs" / OPTIMAL_POLICY
    if named.is_dir():
        return named
    adaptive = marker_path.parent / "graphs" / "adaptive"
    configuration = marker.get("configuration", {})
    if (
        adaptive.is_dir()
        and configuration.get("simplification_method") == "optimal"
        and float(configuration.get("radius_fraction", -1.0)) == 0.75
        and float(configuration.get("minimum_chord_fraction", -1.0)) == 0.95
    ):
        return adaptive
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates: dict[str, dict] = {}
    for root in args.graph_root:
        for marker_path in sorted(root.rglob("complete.json")):
            marker = json.loads(marker_path.read_text())
            if marker.get("dataset") != "ixi":
                continue
            match = SUBJECT_PATTERN.match(str(marker["subject"]))
            if not match:
                raise ValueError(f"cannot obtain numeric IXI id from {marker['subject']!r}")
            patient_id = str(int(match.group(1)))
            graph = _selected_graph(marker_path, marker)
            if graph is None:
                continue
            record = {
                "patient_id": patient_id,
                "subject": marker["subject"],
                "image": str(Path(marker["image"]).resolve()),
                "segmentation": str(Path(marker["label"]).resolve()),
                "graph": str(graph.resolve()),
                "marker": str(marker_path.resolve()),
            }
            previous = candidates.get(patient_id)
            if previous is not None and previous != record:
                raise ValueError(f"duplicate IXI id {patient_id}: {previous} vs {record}")
            candidates[patient_id] = record
    if len(candidates) < 7:
        raise ValueError(f"need at least 7 IXI subjects for nonempty train/val/test; found {len(candidates)}")

    for directory in (
        args.output / "raw",
        args.output / "seg",
        args.output / "graphs" / "adaptive",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    records = [candidates[key] for key in sorted(candidates, key=int)]
    for record in records:
        patient_id = record["patient_id"]
        _link(Path(record["image"]), args.output / "raw" / f"{patient_id}.nii.gz")
        _link(
            Path(record["segmentation"]),
            args.output / "seg" / f"{patient_id}.nii.gz",
        )
        _link(
            Path(record["graph"]),
            args.output / "graphs" / "adaptive" / patient_id,
        )
    payload = {
        "schema_version": 1,
        "dataset": "IXI only",
        "graph_policy": OPTIMAL_POLICY,
        "subjects": records,
    }
    (args.output / "source_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"subjects": len(records), "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
