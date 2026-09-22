"""Evaluator and source-affine checks for physical branch connectivity."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from configs import load_config
from training.evaluation.metrics import evaluate_graph, summarize_metrics
from training.evaluation.physical_coordinates import PatchPhysicalCoordinates, to_world_mm


def test_only_ixi_recipe_opts_in_to_mm_branch_metric():
    root = Path(__file__).resolve().parents[1] / "configs"
    environment = {"SYNTHETIC_MRI_DATASET": "/synthetic", "GNBM_OUTPUT_DIR": "/outputs"}
    ixi = load_config(root / "finetune_ixi_vessels_q256.yaml", environment=environment)
    baseline = load_config(root / "finetune_synthetic_mri.yaml", environment=environment)
    assert ixi["evaluation"]["protocol"]["branch_threshold_mm"] == 1.
    assert "branch_threshold_mm" not in baseline["evaluation"]["protocol"]


def test_patch_provenance_restores_world_mm_not_identity_patch_spacing(tmp_path):
    patch_root = tmp_path / "patches"
    (patch_root / "val" / "vtp").mkdir(parents=True)
    source = tmp_path / "source.nii.gz"
    affine = np.array([[0, -0.5, 0, 11], [0.4, 0, 0, 22], [0, 0, 2, 33], [0, 0, 0, 1.]])
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), affine), source)
    (patch_root / "generation_config.json").write_text(json.dumps({"patch_size": [64, 64, 64], "pad": [1, 2, 3]}))
    (patch_root / "source_manifest.json").write_text(json.dumps({"subjects": [{"patient_id": "000001", "image": str(source)}]}))
    with (patch_root / "patch_index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "patient_id", "start_d", "start_h", "start_w"])
        writer.writeheader()
        writer.writerow({"sample_id": "sample_000001_0000", "patient_id": "000001", "start_d": 10, "start_h": 20, "start_w": 30})
    record = SimpleNamespace(sample_id="sample_000001_0000", graph=patch_root / "val" / "vtp" / "sample_000001_0000_graph.vtp")
    transform = PatchPhysicalCoordinates().transform(record)
    np.testing.assert_allclose(to_world_mm([[.5, .5, .5]], transform)[0], (affine @ [41, 50, 59, 1])[:3], atol=1e-5)
    with pytest.raises(ValueError, match="normalized"):
        PatchPhysicalCoordinates().transform(record, coordinate_space="voxel")


def test_branch_metric_opt_in_and_micro_average_across_patches():
    gt = torch.tensor([[0., 0., 0.], [.5, 0., 0.]])
    prediction = {"nodes": gt, "edges": torch.tensor([[0, 1]]),
                  "boxes": torch.cat((gt, torch.full_like(gt, .2)), dim=1),
                  "node_scores": torch.ones(2), "edge_scores": torch.ones(1)}
    missing = {**prediction, "edges": torch.empty((0, 2), dtype=torch.long), "edge_scores": torch.empty(0)}
    protocol = {"branch_threshold_mm": 1., "smd_iterations": 2}
    transform = np.diag([2., 2., 2., 1.])
    perfect = evaluate_graph(prediction, gt, torch.tensor([[0, 1]]), protocol=protocol, world_transform=transform)
    broken = evaluate_graph(missing, gt, torch.tensor([[0, 1]]), protocol=protocol, world_transform=transform)
    assert (perfect["branch_tp"], perfect["branch_f1"]) == (1, 1.)
    assert broken["branch_fn"] == 1
    summary = summarize_metrics([perfect, broken], folds=2)
    assert summary["branch_f1"] == pytest.approx(2 / 3)  # Micro, not patch-wise mean.
    assert summary["branch_tp_total"] == 1
    assert summary["branch_fn_total"] == 1
    assert evaluate_graph(prediction, gt, torch.tensor([[0, 1]]), protocol={"smd_iterations": 2}).get("branch_f1") is None
    with pytest.raises(ValueError, match="world transform"):
        evaluate_graph(prediction, gt, torch.tensor([[0, 1]]), protocol=protocol)
