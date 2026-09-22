"""Small, reproducible Phase-3 real IXI patch oracle/ablation check.

Only reads two graph-positive validation patches; no GPU, Slurm, or training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loaders.io import read_vtp_graph
from metrics.branch_connectivity import branch_connectivity_f1, contract_degree_two
from training.evaluation.metrics import evaluate_graph
from training.evaluation.physical_coordinates import PatchPhysicalCoordinates, to_world_mm


def samples(root: Path):
    patients = set()
    with (root / "patch_index.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if (row["split"] != "val" or row["patient_id"] in patients
                    or not 18 <= int(row["edge_count"]) <= 60):
                continue
            path = root / "val" / "vtp" / f'{row["sample_id"]}_graph.vtp'
            if path.is_file():
                patients.add(row["patient_id"])
                yield row, path
            if len(patients) == 2:
                break


def plot_graph(ax, nodes, edges, title, *, missing=None):
    for a, b in edges:
        segment = nodes[[a, b]]
        ax.plot(*segment.T, color="#246baf", linewidth=1.3, alpha=.78)
    if missing is not None:
        segment = nodes[list(missing)]
        ax.plot(*segment.T, color="#e34d42", linestyle="--", linewidth=3)
    topology = contract_degree_two(nodes, edges)
    for role, color in (("endpoint", "#1a9d61"), ("junction", "#ff9800"), ("isolated", "#b81740")):
        points = np.array([position for index, position in topology.anchors.items()
                           if topology.roles[index] == role]).reshape(-1, 3)
        if len(points):
            ax.scatter(*points.T, color=color, s=13 if role != "junction" else 25,
                       label=role, depthshade=False)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (mm)", fontsize=8, labelpad=-1)
    ax.set_ylabel("y (mm)", fontsize=8, labelpad=-1)
    ax.set_zlabel("z (mm)", fontsize=8, labelpad=-1)
    ax.tick_params(labelsize=7, pad=0)
    ax.set_box_aspect(np.maximum(np.ptp(nodes, axis=0), 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    physical = PatchPhysicalCoordinates()
    figure = plt.figure(figsize=(15, 13))
    for index, (row, path) in enumerate(samples(args.patch_root)):
        record = SimpleNamespace(sample_id=row["sample_id"], graph=path)
        transform = physical.transform(record)
        nodes_normalized, edges_tensor = read_vtp_graph(path)
        nodes = to_world_mm(nodes_normalized.numpy(), transform)
        edges = edges_tensor.numpy()
        clean = np.delete(edges, len(edges) // 2, axis=0)
        baseline = branch_connectivity_f1(nodes, edges, nodes, edges, threshold_mm=1.)
        ablation = branch_connectivity_f1(nodes, clean, nodes, edges, threshold_mm=1.)
        if ablation["f1"] >= baseline["f1"]:
            raise AssertionError(f'Edge deletion was not detected for {row["sample_id"]}')
        def evaluator_branch_f1(links):
            predicted = {"nodes": nodes_normalized, "edges": links,
                         "boxes": torch.cat((nodes_normalized, torch.full_like(nodes_normalized, .2)), dim=1),
                         "node_scores": torch.ones(len(nodes)), "edge_scores": torch.ones(len(links))}
            return evaluate_graph(predicted, nodes_normalized, edges_tensor,
                                  protocol={"branch_threshold_mm": 1., "smd_iterations": 2},
                                  world_transform=transform)["branch_f1"]
        integrated_oracle = evaluator_branch_f1(edges_tensor)
        integrated_ablation = evaluator_branch_f1(torch.as_tensor(clean.copy()))
        if not np.isclose(integrated_oracle, baseline["f1"]) or not np.isclose(integrated_ablation, ablation["f1"]):
            raise AssertionError("Model evaluator disagrees with standalone physical Branch F1")
        rows.append({"sample_id": row["sample_id"], "patient_id": row["patient_id"],
                     "normalized_patch_axis_scale_mm": np.linalg.norm(transform[:3, :3], axis=0).tolist(),
                     "nodes": len(nodes), "edges": len(edges), "removed_edge": edges[len(edges) // 2].tolist(),
                     "oracle": baseline, "edge_removed": ablation,
                     "integrated_evaluator_branch_f1": [integrated_oracle, integrated_ablation]})
        for side, graph_edges in enumerate((edges, clean)):
            ax = figure.add_subplot(2, 2, index * 2 + side + 1, projection="3d")
            plot_graph(ax, nodes, graph_edges,
                       f'{row["sample_id"]}: {"unchanged" if not side else "one edge removed"}\n'
                       f'Branch F1 = {baseline["f1"] if not side else ablation["f1"]:.3f}',
                       missing=edges[len(edges) // 2] if side else None)
    if len(rows) != 2:
        raise RuntimeError(f"Need two real validation patches; found {len(rows)}")
    figure.subplots_adjust(left=.04, right=.98, top=.94, bottom=.04, hspace=.40, wspace=.18)
    figure.savefig(args.output / "ixi_patch_edge_ablation.png", dpi=170)
    plt.close(figure)
    (args.output / "ixi_patch_edge_ablation.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps([{"sample_id": item["sample_id"],
                       "oracle_f1": item["oracle"]["f1"],
                       "edge_removed_f1": item["edge_removed"]["f1"]} for item in rows]))


if __name__ == "__main__":
    main()
