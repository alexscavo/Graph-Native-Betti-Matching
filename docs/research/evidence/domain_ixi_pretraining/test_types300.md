# IXI node-type model: held-out test evaluation

Job `170838` evaluated the completed 300-epoch IXI node-type model
(`ixi_q256_types_300ep_20260924`, best validation Branch-F1 checkpoint from
epoch 290) on all **2,384 active test patches**.

Summary: node F1 **0.5103**, edge F1 **0.3894**, edge precision **0.2620**,
edge recall **0.7582**, Branch F1 **0.00887**, and edge-count absolute error
**95.55**. The model predicts on average **65.64 nodes / 145.87 edges**, while
the targets contain **54.35 nodes / 50.39 edges**. Thus node detection is
moderate, but the graph is strongly over-connected: predicted beta-1 is
82.82 versus target beta-1 0.77.

These are the plain decoder results at the model's standard inference rule;
they are not evidence that the node-type head improves performance yet. The
matched no-node-type pretraining+fine-tuning control is being run separately.

Interactive paired reports (segmentation+GT beside segmentation+prediction):

- [Sparse patch: `sample_000003_0000`](../../../../../../../../experiments/gnbm/ixi_q256_types_300ep_20260924/evaluation/test_best_branch_f1/interactive_sample_000003_0000/sample_000003_0000__sample_000000.html)
- [Dense/over-connected patch: `sample_000104_0119`](../../../../../../../../experiments/gnbm/ixi_q256_types_300ep_20260924/evaluation/test_best_branch_f1/interactive_sample_000104_0119/sample_000104_0119__sample_001685.html)

The dense example has 168 GT nodes and 164 GT edges versus 228 predicted
nodes and 886 predicted edges. PNG snapshots are next to each HTML report.
The complete machine-readable outputs are `summary.json`,
`per-patch-metrics.csv`, and `predictions.json` in the evaluation directory.
