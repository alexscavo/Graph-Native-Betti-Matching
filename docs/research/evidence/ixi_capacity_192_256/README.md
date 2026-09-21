# IXI capacity and training-crop audit (2026-09-21)

The dataset-local IXI `optimal_radius_0p75x_c095` 64³ grid has maximum
per-axis stride 40 (at least 37.5% overlap for adjacent crops). We archived
2,092 of its 15,828 **training** crop triplets: 1,966 have no mask foreground
and 126 have foreground but no graph nodes or edges. The 13,736 remaining
training triplets are graph-positive. All original validation and test crops
remain in place; the exhaustive `patch_index.csv` remains unchanged. The
dataset-local `training_view.json` and paired CSV indices record provenance.
The independent TopBrain training split has 7,379 retained, 11,124 archived.
Archived data is recoverable, not deleted and does not free storage.

For a single-dataset 5–10k training run, TopBrain already has 7,379 usable
training crops. The proposed IXI-only configuration uses a deterministic
8,000-crop **training** subset of the 13,736, not a new overlapping extraction;
validation and test remain exhaustive, including negative crops. Volume-level
patient assignments are not changed.

## Measured 192-versus-256 cost (IXI only)

One H100 job (`22067`) used 512 paired real IXI training crops with at most
192 nodes, 128 validation crops, batch size 8, five warm-up and 60 full
training steps **per capacity**, 256 first. Both cases used the same training
configuration and seeds, with only the number of object queries changed.
Topo loss and augmentation were disabled identically for the compute comparison.

| Measurement | 192 queries | 256 queries | Change |
|---|---:|---:|---:|
| Mean training step | 92.22 ms | 99.14 ms | +7.51% |
| 5%-trimmed training step | 83.88 ms | 90.40 ms | +7.77% |
| Peak allocated GPU memory | 3.286 GiB | 3.371 GiB | +0.085 GiB (+2.57%) |
| Mean throughput | 94.28/s | 87.84/s | −6.83% |

Separately, a **real IXI 239-node, 261-edge** crop completed a full
256-query training step. IXI has 23 training crops with >192 nodes (35 across
all splits); the maximum is 239. The paired timing samples are deliberately
within the 192-query limit so both cases perform the same work; the separate
overflow step verifies capacity. This is a short-run computational comparison,
not a convergence or accuracy experiment.

- [Training-crop count visualization](training_cleanup.png)
- [Measured cost visualization](comparison.png)
- [Full timings, peaks and overflow step](summary.json)
- [Paired selection manifest](benchmark_selection.csv)
- [All 120 timing steps](steps.csv)
- [Selection summary](selection.json)
- [Job output](slurm-22067.out) and [environment/warnings](slurm-22067.err)
