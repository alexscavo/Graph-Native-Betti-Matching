# IXI-only 120-versus-192 object-query benchmark

This experiment measures the incremental training cost of increasing the
RelationFormer decoder from 120 to 192 object queries. It uses only real IXI
MRA data and the selected full-volume graph policy: constrained-optimal
simplification at 0.75 times local radius and 95% minimum lumen containment.

The completed outcome and interpretation are in [`RESULTS.md`](RESULTS.md).

## Data protocol

Seven IXI subjects are split by patient into train, validation, and test. Their
full-volume graphs are cropped into 64-cubed network inputs after graph
extraction, preserving exact graph/patch-boundary intersection nodes. The
existing `SyntheticMRIDataset` loader already supports the resulting NIfTI plus
VTP layout, so no loader fork or dataset-specific code is needed.

The paired timing dataset contains every available patch with at most 120
nodes, capped at 512/128/128 train/validation/test samples when necessary. The
same shuffled training batches and seed are used for the 120- and 192-query
cases. A separate overflow dataset contains every patch above 120 nodes; its
largest patch is exercised with a complete 192-query training step.

The retained benchmark outputs are under [`final/`](final/). The temporary
selection manifests and generated NIfTI/VTP patches were removed after the
benchmark was completed; the final logs, timings, checksums, and summary remain.

## What is measured

Each case performs five warm-up steps and 100 measured full training steps at
batch size eight on one H100. A step includes forward inference, Hungarian
matching, all enabled graph losses, backward propagation, AdamW, and the
learning-rate scheduler. The report records step-time distributions,
throughput, peak allocated/reserved GPU memory, parameter counts, target graph
sizes, losses, whole-job resource usage, and one-second GPU telemetry.

The definitive run executes 192 before 120 to countercheck an initial 30-step
run that used the opposite order. Reports include the mean, median, 5% trimmed
mean, P05/P95, and standard deviation so isolated scheduler or I/O stalls do
not determine the conclusion.

The comparison changes only `model.decoder.object_queries`. Topological losses
and augmentations are disabled identically in both cases, and no Plants or
TopBrain samples are configured.

## Reproduction

The original reproduction command used temporary staging under
`docs/research/artifacts/`. Those staging outputs were deliberately cleared;
future reruns should use the dataset-local vascular graph and patch folders.

```bash
.venv/bin/python scripts/prepare_ixi_query_benchmark_sources.py \
  --graph-root docs/research/artifacts/real_graph_sweep_optimal_0p75 \
  --graph-root docs/research/artifacts/ixi_query_benchmark_graphs \
  --output docs/research/artifacts/ixi_query_benchmark/sources

.venv/bin/python scripts/generate_synthetic_mri_dataset.py \
  --root docs/research/artifacts/ixi_query_benchmark/sources \
  --output-dir docs/research/artifacts/ixi_query_benchmark/all_patches \
  --split-output docs/research/artifacts/ixi_query_benchmark/sources/ixi_query_split.csv \
  --patch-size 64 64 64 --pad 0 0 0 --maximum-stride 40 40 40 --workers 7

.venv/bin/python scripts/make_ixi_query_benchmark_dataset.py \
  --patch-root docs/research/artifacts/ixi_query_benchmark/all_patches \
  --paired-output docs/research/artifacts/ixi_query_benchmark/paired \
  --overflow-output docs/research/artifacts/ixi_query_benchmark/overflow \
  --report-dir docs/research/evidence/ixi_query_benchmark/selection

bash cluster/jean_zay/submit_ixi_query_benchmark.sh
```

The submitter requests one H100 for 30 minutes in the development QoS to
minimize both queue delay and allocation cost.
