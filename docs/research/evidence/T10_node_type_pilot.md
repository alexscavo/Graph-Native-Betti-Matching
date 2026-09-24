# IXI node-type / unmatched-negative pilot (2026-09-24)

## Why this test

The IXI-only 300-epoch model overconnects vessels (held-out test: mean 179.76
predicted edges versus 50.39 target edges per patch). Its existing `class` loss
is binary object/background, not vessel-node type. In training, matched node
pairs receive edge supervision, but pairs involving unmatched predicted nodes
do not unless hard-negative mining is enabled. Inference considers all
predicted-node pairs.

An earlier hard-negative comparison also lowered the matched-pair limit from
9,999 to about 1,000. It cannot isolate hard-negative mining: at 30–40 nodes,
there are only 435–780 unordered node pairs, while dense patches exceed 1,000.
All new ablations **retain the 9,999 matched-pair limit**; the separate
unmatched hard-negative cap is 128 per graph.

## Real-data target distribution

We sampled 100 of the 13,742 IXI training-patch VTPs with seed 364505 and
counted degree directly from stored edges. Among 5,615 nodes: isolated 0
(0%), within-patch terminals 1,651 (29.40%), degree-2 3,214 (57.24%),
junctions (degree >=3) 750 (13.36%). Median patch size was 49 nodes, maximum
150; 7 sampled patches exceeded 120 nodes, none exceeded 192. This is a
descriptive sample, not a full-dataset frequency estimate. A patch terminal
may be a boundary-cut node, not an anatomical vessel endpoint.

## Controlled comparison

- Existing epoch-270 IXI checkpoint is the common **model-only initializer**.
  The optional four-class type head is new and randomly initialized.
- `configs/finetune_ixi_q256_control_10ep.yaml` and
  `configs/finetune_ixi_q256_types_10ep.yaml` use the same seeded 1,000 IXI
  training patches, complete validation split, 10 epochs, and optimizer
  schedule. Only the type head/loss differs. No plants or test patches enter
  this pilot.
- Compare validation node/edge/Branch F1, mean predicted nodes/edges, beta-1,
  and results by target-node-count strata. Evaluate the trained type model
  once without and once with optional type-degree caps. The cap is a decoder
  ablation, not evidence of learned edge quality on its own.
- Hard negatives remain **off** in both control and type pilot. They have
  separate configurations (`finetune_ixi_vessels_q256_hnm.yaml` and
  `finetune_ixi_vessels_q256_types_hnm.yaml`) for a later isolated test if
  justified by the node-type result.

A 10-epoch warm-start result is directional. It does not establish the final
quality after full training or plants pretraining. The held-out test split
must not be used to choose thresholds or variants.

## Status

Focused CPU tests: 27 passed. Both pilot configurations passed dataset
preflight with identical train and validation manifest hashes. Submitted
one-H100, 90-minute development jobs: control `140955`, node types `141009`.
At submission they were pending; no model-quality result is claimed yet.

## Completed 10-epoch warm-start comparison

Both jobs completed successfully in 6–7 minutes. At epoch 10, on all 2,961
IXI validation patches:

| Metric | Control | Type auxiliary loss |
|---|---:|---:|
| Predicted nodes / patch | 77.19 | 74.81 |
| Predicted edges / patch | 214.78 | 211.08 |
| Target edges / patch | 52.43 | 52.43 |
| Edge precision | 0.1662 | 0.1675 |
| Edge recall | 0.6811 | 0.6746 |
| Edge F1 | 0.2673 | 0.2684 |
| Physical Branch F1 | 0.00561 | 0.00587 |
| Mean absolute beta-1 error | 138.86 | 137.51 |

The type auxiliary loss alone gave only a very small difference and did not
solve overconnection. This does **not** test feeding type information into
edge decisions. The next validation-only ablation uses the frozen type-model
checkpoint and a probability-weighted *soft* degree preference; no hard cap
or additional training is involved. Patch terminals include boundary cuts.

The epoch-270 initializer was chosen by validation Branch F1 (0.006756 at
epoch 270 versus 0.006657 at epoch 300). Epoch 300 had slightly higher edge
F1 (0.3171 versus 0.3148), and both had severe edge overcount. To check
dependence on this checkpoint choice, matched 10-epoch control and type runs
from the **epoch-300 IXI-only** model were also submitted as jobs `141926`
and `141927`. Both use exactly the same manifest hashes as the epoch-270
pilots. These are not plants-pretrained runs.

## Real-data node-type diagnostic

On 200 seeded validation patches, nearest one-to-one spatial matching within
0.05 normalized units matched 3,904 of 11,119 GT nodes to predictions. Type
accuracy among those matched nodes was 59.63%. This conditional accuracy is
only a little above the degree-2 majority frequency among matched nodes; it
does not measure localization failures. Terminals and junctions are often
confused with degree-2 nodes. Across all predicted nodes in this sample,
68.0% were classified degree-2 and only 7.9% junction; GT proportions were
54.5% and 15.0%. This junction underprediction is particularly concerning
for degree-aware pruning. See
[confusion matrix](T10_node_type_results/node_type_confusion.png) and
[counts](T10_node_type_results/node_type_accuracy.json). The type head is
therefore not yet a reliable hard constraint; a soft test is appropriate.
