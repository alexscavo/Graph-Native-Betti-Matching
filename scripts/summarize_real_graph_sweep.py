#!/usr/bin/env python3
"""Aggregate adaptive-policy full-graph and exact-patch sweep results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def distribution(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.int64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()) if len(array) else 0.0,
        "median": float(np.median(array)) if len(array) else 0.0,
        "p95": float(np.percentile(array, 95)) if len(array) else 0.0,
        "p99": float(np.percentile(array, 99)) if len(array) else 0.0,
        "max": int(array.max()) if len(array) else 0,
        "over_70": int(np.count_nonzero(array > 70)),
        "over_120": int(np.count_nonzero(array > 120)),
        "fraction_over_70": float(np.mean(array > 70)) if len(array) else 0.0,
        "fraction_over_120": float(np.mean(array > 120)) if len(array) else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    markers = sorted(args.root.glob("**/complete.json"))
    if not markers:
        parser.error(f"no complete.json files below {args.root}")
    patches: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"nodes": [], "edges": []})
    full: dict[str, list[dict]] = defaultdict(list)
    strata: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: {"nodes": [], "edges": []})
    )
    subjects = []
    for marker in markers:
        payload = json.loads(marker.read_text())
        subject = payload["subject"]
        stratum = f"{payload['dataset']}/{payload['modality']}"
        subjects.append({"subject": subject, "stratum": stratum})
        dense = payload["representations"]["dense"]
        with (marker.parent / "patch_counts.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                if int(row["edges"]) == 0 or row["policy"] == "dense":
                    continue
                policy = row["policy"]
                for metric in ("nodes", "edges"):
                    value = int(row[metric])
                    patches[policy][metric].append(value)
                    strata[stratum][policy][metric].append(value)
        for policy, record in payload["representations"].items():
            if policy == "dense":
                continue
            full[policy].append({
                **record,
                "dense_nodes": dense["nodes"],
                "dense_edges": dense["edges"],
                "dense_degree_2": dense["degree_2"],
                "topology_preserved": (
                    record["betti_0"] == dense["betti_0"]
                    and record["betti_1"] == dense["betti_1"]
                ),
            })

    policies = sorted(patches)
    summary = {
        "completed_subjects": len(markers),
        "subjects": subjects,
        "policies": {},
        "strata": {},
    }
    for policy in policies:
        records = full[policy]
        degree2 = np.asarray([row["degree_2"] for row in records], dtype=float)
        dense_degree2 = np.asarray([row["dense_degree_2"] for row in records], dtype=float)
        summary["policies"][policy] = {
            "topology_preserved_subjects": int(sum(row["topology_preserved"] for row in records)),
            "topology_subjects": len(records),
            "full_degree_2_total": int(degree2.sum()),
            "full_degree_2_mean": float(degree2.mean()),
            "degree_2_reduction_vs_dense": float(1.0 - degree2.sum() / dense_degree2.sum()),
            "patch_nodes": distribution(patches[policy]["nodes"]),
            "patch_edges": distribution(patches[policy]["edges"]),
        }
    for stratum, policy_values in strata.items():
        summary["strata"][stratum] = {
            policy: {
                "patch_nodes": distribution(values["nodes"]),
                "patch_edges": distribution(values["edges"]),
            }
            for policy, values in policy_values.items()
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    rows = []
    for policy, record in summary["policies"].items():
        rows.append({
            "policy": policy,
            "topology_preserved": f"{record['topology_preserved_subjects']}/{record['topology_subjects']}",
            "full_degree_2_total": record["full_degree_2_total"],
            "degree_2_reduction_vs_dense": record["degree_2_reduction_vs_dense"],
            **{f"nodes_{key}": value for key, value in record["patch_nodes"].items()},
            **{f"edges_{key}": value for key, value in record["patch_edges"].items()},
        })
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt
    labels = [policy.replace("fixed_", "F ").replace("radius_", "R ") for policy in policies]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    x = np.arange(len(policies))
    axes[0].bar(x, [summary["policies"][p]["patch_nodes"]["p95"] for p in policies], color="#4472C4")
    axes[0].axhline(120, color="#c0392b", linestyle="--", label="120-token capacity")
    axes[0].set(ylabel="P95 nodes / nonempty patch", xticks=x, xticklabels=labels)
    axes[0].legend()
    axes[1].bar(x, [summary["policies"][p]["patch_edges"]["p95"] for p in policies], color="#70AD47")
    axes[1].axhline(70, color="#e67e22", linestyle="--", label="preferred 70 edges")
    axes[1].set(ylabel="P95 edges / nonempty patch", xticks=x, xticklabels=labels)
    axes[1].legend()
    axes[2].bar(x, [summary["policies"][p]["full_degree_2_mean"] for p in policies], color="#A64D79")
    axes[2].set(ylabel="Mean full-volume degree-2 nodes", xticks=x, xticklabels=labels)
    for axis in axes:
        axis.tick_params(axis="x", rotation=55)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(f"Adaptive policy sweep: {len(markers)} real volumes, exact 64³ crops")
    figure.savefig(args.output_dir / "policy_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"completed_subjects": len(markers), "policies": policies}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
