# TopBrain mixed-pretraining plus vessel fine-tuning time estimate

## Central estimate

For one model, expect about **9.5 hours of compute wall time** with the current
launcher layout: one H100 for 50 epochs of mixed pretraining, then four H100s
for 100 epochs of vessel-only fine-tuning, preserving global batch 32. A
reasonable planning range is **7.3–13.0 hours**, excluding Slurm queue time.
Requesting approximately 12 hours for pretraining and 8 hours for fine-tuning
would provide operational margin while keeping the jobs independently
resumable.

This is an extrapolation, not a completed end-to-end timing run. The empirical
anchor is the IXI-only 192-query measurement of 81.30 ms per full batch-8
training step on one H100. The range accounts for batch-32 scaling, four-GPU
DDP communication, mixed Plants/MRI loading and augmentation, validation,
metric evaluation, and checkpoint I/O.

## Patch inventory assumption

The current TopBrain manifest has 43 released training volumes: 25 MR and 18
CT. The production patch generator's defaults (`patch_size=64`, `pad=5`,
`maximum_stride=40`) yield an estimated 28,843 training patches from their
image dimensions. The seven released labeled CT test cases add 2,485 patches
and are treated as validation/test data, not training data, in this estimate.

If the zero-padding ablation used by the IXI query benchmark is selected
instead (`pad=0`, true 64-cubed crops), the same 43 TopBrain volumes produce
26,391 patches. That lowers the central fine-tuning estimate by only about 13
minutes. The final exact count depends on the patient split written during
TopBrain patch generation.

## What the current loader actually does

With `balance_source_target: true`, `compose_source_target()` constructs a
weighted sampler with `num_samples = 2 × len(Plants)`. Consequently, the full
28,843-patch vessel dataset is available to the sampler, but a pretraining
epoch contains exactly:

- 51,800 sample presentations;
- approximately 25,900 Plants draws;
- approximately 25,900 vessel draws;
- sampling with replacement; and
- 1,619 optimizer steps at global batch 32.

Therefore 50 pretraining epochs contain 2,590,000 sample presentations and
80,950 optimizer steps. The code does not traverse every vessel patch exactly
once per mixed epoch. Nevertheless, every vessel patch receives about 44.9
draws on average over 50 epochs, and the probability that an individual patch
is never sampled is negligible.

If literal one-pass concatenation is desired, set
`balance_source_target: false`. Each epoch would then contain 54,743 samples
and 1,711 steps. At the central timing assumption, this adds approximately 19
minutes across all 50 pretraining epochs.

## Fine-tuning calculation

Four-GPU DDP uses batch eight per GPU. `DistributedSampler` pads 28,843 samples
to 28,844 globally, or 7,211 per rank, producing 902 optimizer steps; the final
step contains only three samples per rank. One hundred epochs therefore contain
90,200 optimizer steps and 2,884,400 actual presentations. The trainer's
performance log reports the nominal batch capacity (28,864 per epoch), because
it multiplies batch count by the configured batch size and does not inspect the
short final batch; the timing calculation depends on steps and is unaffected.

The central estimate assumes 250 ms per one-H100 batch-32 pretraining step and
105 ms per four-H100 DDP batch-8-per-rank fine-tuning step:

| Stage | Steps | Training time | Estimated non-training overhead | Central wall time |
|---|---:|---:|---:|---:|
| Mixed pretraining, 1 H100 | 80,950 | 5.62 h | about 0.50 h | about 6.1 h |
| Vessel fine-tuning, 4 H100s | 90,200 | 2.63 h | about 0.75 h | about 3.4 h |
| Total | 171,150 | 8.25 h | about 1.25 h | **about 9.5 h** |

The overhead allowance covers the behavior implemented by `Trainer.fit()`:
validation loss every five epochs, graph metrics every five epochs, full
fine-tuning validation metrics, a replace-in-place recovery checkpoint every
two epochs, best-metric checkpoints, data loading, startup, and synchronization.
The full-validation graph metrics execute only on rank zero while the other
DDP ranks wait. A local real-IXI target benchmark measured the CPU part of
`evaluate_graph()` at about 16 ms per patch; random-model inference and relation
decoding can make it slower, which is included in the range.

## Alternative GPU layouts

At the same global batch of 32 and central step assumptions:

| Pretraining GPUs | Fine-tuning GPUs | Estimated wall time | Qualitative cost |
|---:|---:|---:|---|
| 1 | 1 | about 13.1 h | Lowest GPU-hours, longest wall time |
| 1 | 4 | about 9.5 h | Current launcher default and recommended balance |
| 4 | 4 | about 6.2 h | Shortest wall time, higher GPU-hours and queue cost |

These estimates assume all 150 requested epochs run. If fine-tuning retains
the current early-stopping logic, it can stop earlier; it should not be counted
on when reserving wall time.

## Evidence

- [`estimate.csv`](estimate.csv): machine-readable timing assumptions and
  totals.
- [`estimate.png`](estimate.png): stage and GPU-layout comparison.
- TopBrain volume shapes come from
  `docs/research/artifacts/real_graph_dataset/extraction_manifest.csv`.
- The empirical H100 anchor is in
  `../ixi_query_benchmark/final/summary.json`.
