#!/usr/bin/env python3
"""Validate a paired real-data report and record limits of its conclusions."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def validate(rows: list[dict]) -> dict:
    grouped = defaultdict(dict)
    for row in rows:
        key = row["subject"]
        representation = row["representation"]
        if representation in grouped[key]:
            raise ValueError(f"duplicate {representation} for {key}")
        grouped[key][representation] = row
    result = {"subjects": len(grouped), "adaptive_better_f1_count": 0,
              "adaptive_better_acd_count": 0, "topology_equal_count": 0,
              "adaptive_lower_node_count": 0, "adaptive_f1_0p5_range": [1.0, 0.0],
              "adaptive_f1_1mm_range": [1.0, 0.0], "groups": {}}
    strata = defaultdict(list)
    for subject, group in grouped.items():
        if set(group) != {"junction_only", "adaptive", "dense"}:
            raise ValueError(f"incomplete representation family for {subject}")
        compact, anchor, reference = (group[key] for key in ("adaptive", "junction_only", "dense"))
        for row in group.values():
            if not 0 <= float(row["curve_f1_0.5mm"]) <= 1:
                raise ValueError(f"invalid F1 on {subject}")
            if not np.isfinite(float(row["curve_acd_mm"])):
                raise ValueError(f"non-finite ACD on {subject}")
        if all((row["components"], row["cycle_rank"]) ==
               (reference["components"], reference["cycle_rank"]) for row in group.values()):
            result["topology_equal_count"] += 1
        result["adaptive_better_f1_count"] += float(compact["curve_f1_0.5mm"]) > float(anchor["curve_f1_0.5mm"])
        result["adaptive_better_acd_count"] += float(compact["curve_acd_mm"]) < float(anchor["curve_acd_mm"])
        result["adaptive_lower_node_count"] += int(compact["nodes"]) < int(reference["nodes"])
        for field, key in (("curve_f1_0.5mm", "adaptive_f1_0p5_range"),
                           ("curve_f1_1mm", "adaptive_f1_1mm_range")):
            value = float(compact[field])
            result[key][0] = min(result[key][0], value)
            result[key][1] = max(result[key][1], value)
        strata[compact["stratum"]].append(compact)
    for name, entries in sorted(strata.items()):
        result["groups"][name] = {
            "subjects": len(entries),
            "in_plane_spacing_mm": [min(float(r["spacing_x_mm"]) for r in entries),
                                    max(float(r["spacing_x_mm"]) for r in entries)],
            "mean_adaptive_f1_0p5": float(np.mean([float(r["curve_f1_0.5mm"]) for r in entries])),
            "mean_adaptive_f1_1mm": float(np.mean([float(r["curve_f1_1mm"]) for r in entries])),
            "mean_adaptive_acd_mm": float(np.mean([float(r["curve_acd_mm"]) for r in entries])),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--sampling-check", action="append", type=Path, default=[])
    args = parser.parse_args()
    rows = load_rows(args.report_dir / "volumes.csv")
    report = validate(rows)
    normal = {(row["subject"], row["representation"]): row for row in rows}
    sampling = []
    for path in args.sampling_check:
        for row in load_rows(path / "volumes.csv"):
            if row["representation"] != "adaptive":
                continue
            base = normal[(row["subject"], "adaptive")]
            sampling.append({"subject": row["subject"], "tested_spacing_mm": 0.25,
                             "baseline_spacing_mm": 0.5,
                             "curve_f1_0p5_absolute_change": abs(float(row["curve_f1_0.5mm"]) - float(base["curve_f1_0.5mm"])),
                             "acd_absolute_change_mm": abs(float(row["curve_acd_mm"]) - float(base["curve_acd_mm"]))})
    report["sampling_sensitivity"] = sampling
    (args.report_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
