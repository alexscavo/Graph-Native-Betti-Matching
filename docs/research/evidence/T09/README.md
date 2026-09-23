# IXI 256-query integration smoke (2026-09-23)

This is an integration test, **not** a convergence or anatomical-quality
experiment. It uses only real IXI Vedo-original adaptive-graph patches.

- Config: `configs/finetune_ixi_vessels_q256_smoke.yaml` (32 seeded training
  patches, 16 validation patches, batch size 2, one epoch, 256 object queries).
- Slurm training job `92364`: completed with exit 0 in 75 seconds on one H100.
  Sixteen optimizer steps; training loss 10.049695; validation loss 8.655972.
  Measured training portion 19.88 seconds, 1.61 samples/s, peak allocated
  GPU memory 2.208 GiB. These losses have no useful quality interpretation
  after one epoch.
- Outputs: `runs/ixi_q256_smoke_20260923/` contains the resolved config,
  exact train/validation sample manifest, status, performance JSONL, offline
  tracking, and checkpoints. The completed status reports epoch 1/1,
  iteration 16. The cluster log is
  `/lustre/fswork/projects/rech/vnc/upz25mj/logs/graph-native-betti-matching/gnbm-train-92364.out`.
- [Real validation-patch PNG](sample_000012_0062_patch_all_nodes.png) and
  [interactive 3D HTML](sample_000012_0062_patch_all_nodes.html) show the
  target graph and segmentation for one patch in the smoke validation set:
  33 nodes, 29 edges, two exact patch-boundary nodes.

The first preflight failed because exact patch-face nodes can have normalized
coordinates `-0.5/64` or `63.5/64`; the old augmentation validator allowed
only voxel-centre coordinates. `allow_patch_faces=True` now admits only that
half-voxel extension for the SyntheticMRI graph augmentations. The corrected
preflight read real train/validation triplets, and the full training epoch
loaded all 32 training samples without a coordinate exception.

The one-sample checkpoint evaluation (Slurm job `92694`, exit 0) is recorded
in [prediction_eval/summary.json](prediction_eval/summary.json), the
[static GT/prediction comparison](prediction_eval/plots/sample_000000.png),
and the [interactive 3D report](prediction_eval/sample_000012_0062__sample_000000.html).
The first attempt exposed a missing dataset-local source-manifest lookup for
physical-mm evaluation; the evaluator now follows `generation_config.json`'s
`source_root`, and a regression test covers that layout. On this one patch,
the one-epoch, random-initialized checkpoint predicts 30 isolated nodes and
zero edges versus 33 GT nodes and 29 GT edges; Branch F1 is 0. This is an
expected **failed-quality** example, not a scientifically meaningful model.
It confirms that inference, physical-coordinate metrics, and exports run.
