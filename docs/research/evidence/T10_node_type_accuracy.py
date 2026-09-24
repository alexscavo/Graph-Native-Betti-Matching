"""Check predicted patch-degree classes against real IXI validation graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


LABELS = ("isolated", "patch terminal", "degree 2", "junction")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--maximum-distance", type=float, default=0.05)
    args = parser.parse_args()

    records = json.loads(args.predictions.read_text(encoding="utf-8"))
    chosen = random.Random(364505).sample(records, min(args.samples, len(records)))
    confusion = np.zeros((4, 4), dtype=np.int64)
    predicted_histogram = np.zeros(4, dtype=np.int64)
    target_histogram = np.zeros(4, dtype=np.int64)
    matched_distances = []
    for record in chosen:
        source_id = record["source_sample_id"]
        graph = pv.read(args.dataset_root / "val" / "vtp" / f"{source_id}_graph.vtp")
        gt = np.asarray(graph.points, dtype=np.float32).reshape(-1, 3)
        pred = np.asarray(record["nodes_dhw"], dtype=np.float32).reshape(-1, 3)
        scores = np.asarray(record["node_type_scores"], dtype=np.float32).reshape(-1, 4)
        if len(pred) != len(scores):
            raise ValueError(f"Node/type count mismatch: {source_id}")
        lines = np.asarray(graph.lines, dtype=np.int64).reshape(-1, 3)
        edges = lines[:, 1:] if len(lines) else np.empty((0, 2), dtype=np.int64)
        degree = np.bincount(edges.reshape(-1), minlength=len(gt))
        target_type = np.minimum(degree, 3)
        predicted_type = scores.argmax(axis=1)
        target_histogram += np.bincount(target_type, minlength=4)
        predicted_histogram += np.bincount(predicted_type, minlength=4)
        if not len(gt) or not len(pred):
            continue
        distances = cdist(gt, pred)
        gt_indices, pred_indices = linear_sum_assignment(distances)
        for gt_index, pred_index in zip(gt_indices, pred_indices):
            distance = float(distances[gt_index, pred_index])
            if distance <= args.maximum_distance:
                confusion[target_type[gt_index], predicted_type[pred_index]] += 1
                matched_distances.append(distance)

    result = {
        "samples": len(chosen),
        "maximum_normalized_distance": args.maximum_distance,
        "matched_nodes": int(confusion.sum()),
        "target_nodes": int(target_histogram.sum()),
        "predicted_nodes": int(predicted_histogram.sum()),
        "matched_type_accuracy": float(np.trace(confusion) / confusion.sum()) if confusion.sum() else None,
        "target_type_counts": target_histogram.tolist(),
        "predicted_type_counts": predicted_histogram.tolist(),
        "confusion_gt_rows_predicted_columns": confusion.tolist(),
        "mean_matched_distance": float(np.mean(matched_distances)) if matched_distances else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "node_type_accuracy.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Matched nodes")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=LABELS,
           yticklabels=LABELS, xlabel="Predicted patch degree class",
           ylabel="GT patch degree class",
           title=f"IXI validation • {result['matched_nodes']:,} spatially matched nodes")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for row in range(4):
        for column in range(4):
            ax.text(column, row, str(confusion[row, column]), ha="center", va="center",
                    color="white" if confusion[row, column] > confusion.max() / 2 else "black")
    fig.savefig(args.output_dir / "node_type_confusion.png", dpi=170)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
