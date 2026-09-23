# IXI adaptive-GT 100-epoch run

Status at submission on 2026-09-23: Slurm job **93539** submitted; not yet a
completed training result. The run is IXI-only, using the Vedo-original
adaptive patch set. There are no plants, TopBrain patches, matcher changes,
loss changes, or architecture changes.

| Setting | Frozen value |
|---|---|
| Config | `configs/finetune_ixi_vessels_q256.yaml` |
| Train | 8,000 seeded IXI patches from 13,742 active training patches |
| Validation | all 2,961 active IXI validation patches |
| Split unit | source volume/patient; no patch sharing across splits |
| Object queries | 256 |
| Hardware | two H100 GPUs, batch 8 per GPU (global batch 16) |
| Requested time | 12 hours, `qos_gpu_h100-t3` |
| Epochs | 100 |
| Evaluation | validation loss and graph metrics every 10 epochs; Branch F1 in physical mm |
| Checkpoints | best validation, best Branch F1, and replace-in-place latest every two epochs |
| Initialization | random, fixed seed 364505; not resumed from the one-epoch smoke/pilot |

The one-GPU I/O/metric pilot was job `93034`: 512 train / 64 validation,
batch 8, one epoch, exit 0 in 89 seconds. Its measured training portion was
25.25 seconds for 512 samples (20.28 samples/s; 3.541 GiB peak allocated).
This is a **one-GPU timing anchor**, not a measured two-GPU throughput or a
convergence claim. The 12-hour request includes overhead for full validation
and checkpoints; if it is insufficient, submit the same config/run name with
`GNBM_AUTO_RESUME=1`, which restores the latest full-state checkpoint. The
pilot confirmed that Branch F1 appears in `validation-metrics.jsonl`; at one
epoch from scratch it was 0, as expected.

Live outputs: `runs/ixi_q256_100ep_20260923/`. Cluster stdout/stderr:
`/lustre/fswork/projects/rech/vnc/upz25mj/logs/graph-native-betti-matching/gnbm-train-93539.{out,err}`.
When metrics become available, generate the progress chart with
`python docs/research/evidence/T09_long/plot_progress.py`.
