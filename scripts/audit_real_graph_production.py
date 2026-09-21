#!/usr/bin/env python3
"""Validate full-volume graph and patch inventories and render production QC."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_real_graph_patch_sources import EXPECTED_CONFIGURATION, GRAPH_FILES


def describe(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.int64)
    return {
        "count": int(len(array)),
        "minimum": int(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": int(array.max()),
        "over_70": int(np.sum(array > 70)),
        "over_120": int(np.sum(array > 120)),
        "over_192": int(np.sum(array > 192)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    sources = json.loads(args.source_manifest.read_text())
    source_by_id = {item["patient_id"]: item for item in sources["subjects"]}
    if len(manifest) != len(source_by_id):
        raise ValueError("full-volume manifest and staged source counts differ")

    full_rows = []
    for row in manifest:
        root = args.graph_root / row["dataset"] / row["modality"] / row["split"] / row["subject"]
        marker = json.loads((root / "complete.json").read_text())
        if any(marker["configuration"].get(k) != v for k, v in EXPECTED_CONFIGURATION.items()):
            raise ValueError(f"wrong policy: {row['subject']}")
        graph = root / "graphs" / "adaptive"
        if any(not (graph / name).is_file() for name in GRAPH_FILES):
            raise FileNotFoundError(f"incomplete graph: {graph}")
        stats = marker["representations"]["adaptive"]
        full_rows.append(
            {
                "index": row["index"],
                "dataset": row["dataset"],
                "modality": row["modality"],
                "source_split": row["split"],
                "subject": row["subject"],
                "nodes": int(stats["nodes"]),
                "edges": int(stats["edges"]),
                "betti_0": int(stats["betti_0"]),
                "betti_1": int(stats["betti_1"]),
                "elapsed_seconds": float(marker["elapsed_seconds"]),
            }
        )

    summary_path = args.patch_root / "generation_summary.json"
    generation = json.loads(summary_path.read_text())
    if not generation.get("complete") or generation["completed_patients"] != len(manifest):
        raise ValueError(f"patch generation is incomplete: {generation}")
    with (args.patch_root / "patch_index.csv").open(newline="") as stream:
        patch_rows = list(csv.DictReader(stream))
    enriched = []
    missing_vtp = []
    for row in patch_rows:
        source = source_by_id[row["patient_id"]]
        vtp = args.patch_root / row["split"] / "vtp" / f"{row['sample_id']}_graph.vtp"
        if not vtp.is_file():
            missing_vtp.append(str(vtp))
        enriched.append(
            {
                **row,
                "dataset": source["dataset"],
                "modality": source["modality"],
                "subject": source["subject"],
            }
        )
    if missing_vtp:
        raise FileNotFoundError(f"missing {len(missing_vtp)} patch graphs; first={missing_vtp[0]}")

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in enriched:
        grouped[(row["dataset"], row["modality"], row["split"])].append(row)
    strata = []
    for key, rows in sorted(grouped.items()):
        nodes = [int(row["node_count"]) for row in rows]
        edges = [int(row["edge_count"]) for row in rows]
        node_stats, edge_stats = describe(nodes), describe(edges)
        strata.append(
            {
                "dataset": key[0],
                "modality": key[1],
                "split": key[2],
                "patches": len(rows),
                **{f"nodes_{name}": value for name, value in node_stats.items()},
                **{f"edges_{name}": value for name, value in edge_stats.items()},
            }
        )
    overflow = sorted(
        (
            {
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "modality": row["modality"],
                "subject": row["subject"],
                "split": row["split"],
                "node_count": int(row["node_count"]),
                "edge_count": int(row["edge_count"]),
            }
            for row in enriched
            if int(row["node_count"]) > 120
        ),
        key=lambda row: row["node_count"],
        reverse=True,
    )
    all_nodes = [int(row["node_count"]) for row in enriched]
    all_edges = [int(row["edge_count"]) for row in enriched]
    summary = {
        "complete": True,
        "full_volumes": len(full_rows),
        "full_volume_groups": {
            "/".join(key): count
            for key, count in sorted(
                Counter((row["dataset"], row["modality"], row["source_split"]) for row in full_rows).items()
            )
        },
        "patches": len(enriched),
        "patch_patients": len({row["patient_id"] for row in enriched}),
        "patch_nodes": describe(all_nodes),
        "patch_edges": describe(all_edges),
        "generation_summary": generation,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(args.output / "full_volume_inventory.csv", full_rows)
    write_csv(args.output / "patch_strata.csv", strata)
    write_csv(args.output / "patches_over_120_nodes.csv", overflow)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].hist([row["nodes"] for row in full_rows], bins=30, color="#3969ac")
    axes[0].set(title="Selected full-volume graph complexity", xlabel="Adaptive nodes", ylabel="Volumes")
    # Show the complete tail: a clipped 99.9-percentile histogram would hide
    # precisely the rare patches exceeding the training query capacity.
    upper = max(193, max(all_nodes) + 2)
    axes[1].hist(all_nodes, bins=np.arange(0, upper + 2, 2), color="#11a579")
    for value, color, label in ((70, "#e73f74", "70"), (120, "#f2b701", "120"), (192, "#7f3c8d", "192")):
        axes[1].axvline(value, color=color, linestyle="--", linewidth=1.5, label=label)
    axes[1].set(title="Exact-boundary patch node distribution", xlabel="Nodes per 64³ patch", ylabel="Patches", yscale="log")
    axes[1].legend(title="Capacity")
    figure.savefig(args.output / "production_graph_and_patch_qc.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
