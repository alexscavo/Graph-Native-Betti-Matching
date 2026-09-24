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

Focused CPU tests: 34 passed. Both pilot configurations passed dataset
preflight with identical train and validation manifest hashes. The
one-H100 development jobs, control `140955` and node types `141009`, both
completed successfully.

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

## Epoch-300 initializer check

The epoch-270 initializer was selected only by a narrow original-run
validation Branch F1 advantage (0.006756 versus 0.006657 at epoch 300).
Four completed, matched 10-epoch IXI-only warm starts now test whether that
choice changes the type-head conclusion. They use the same seeded 1,000
training patches, all 2,961 validation patches, hyperparameters, and
decoding threshold. The optional type head is randomly initialized in each
type run; the control does not have it. No plants data or test patches enter.

| Initializer / model | Pred. nodes | Pred. edges | Edge F1 | Branch F1 | Mean abs. beta-1 error |
|---|---:|---:|---:|---:|---:|
| Epoch 270 / control | 77.19 | 214.78 | 0.26725 | 0.00561 | 138.86 |
| Epoch 270 / type loss | 74.81 | 211.08 | 0.26842 | 0.00587 | 137.51 |
| Epoch 300 / control | 78.16 | 218.67 | 0.26494 | 0.00569 | 141.77 |
| Epoch 300 / type loss | 76.87 | 214.49 | 0.26712 | 0.00560 | 138.87 |

The same target has 52.43 edges/patch. Epoch 300 is not better in this
matched pilot. Type supervision by itself makes only a small change from
either initializer, and does not cure overconnection. The difference between
270 and 300 is small; these short warm starts do not establish that 270 is a
universally superior training checkpoint. Jobs `141926` and `141927`
completed successfully.

## Frozen epoch-270 type model: soft degree-aware decoding

The frozen epoch-270-initialized type model was evaluated on the complete
validation set. A *soft* degree penalty uses predicted patch-local type
probabilities to lower the acceptance margin for edges that would exceed a
likely terminal or degree-2 node's expected degree. It never imposes a hard
cap. A plain edge-threshold sweep is the necessary non-type comparison:

| Decoder | Pred. edges | Edge F1 | Branch F1 | Mean abs. beta-1 error |
|---|---:|---:|---:|---:|
| Original threshold (penalty 0) | 211.08 | 0.2684 | 0.00587 | 137.51 |
| Plain threshold 0.65 | 125.33 | 0.3421 | 0.00536 | 56.30 |
| Soft penalty 0.5 | 115.16 | 0.3611 | 0.00658 | 44.93 |
| Soft penalty 1 | 76.82 | 0.4036 | 0.00395 | 13.25 |
| Soft penalty 2 | 57.80 | 0.4101 | 0.00067 | 3.15 |

The modest 0.5 penalty is promising: despite predicting slightly fewer
edges than plain threshold 0.65, it has better edge F1 and Branch F1. This
is evidence that type-based edge ordering may help, not proof that the
classifier is well calibrated. Stronger penalties improve edge F1 and
edge-count error but destroy branch continuity. Edge count or beta-1 error
alone is an unsafe selection criterion. The threshold 0.65 and penalty 0.5
points have similar, but not exactly matched, edge counts; a finer
count-matched threshold or paired per-patch analysis would strengthen the
claim.

See the [complete sweep data](T10_node_type_results/soft_degree_sweep.csv),
[metric plot](T10_node_type_results/soft_degree_sweep.png), and paired
segmentation/graph views for [sample 38, penalty 0.5](T10_node_type_results/views/penalty_0.5/sample_000038_0076__sample_000616.html)
and [sample 155, penalty 0.5](T10_node_type_results/views/penalty_0.5/sample_000155_0062__sample_002580.html).
These two validation samples were selected by GT node-count stratum and
seed, not by model performance. The HTML views synchronize the two cameras.

## Epoch-300 soft-decoder replication

The epoch-300-initialized type model's frozen checkpoint was evaluated with
the same penalty 0, 0.5, and 1 settings, plus plain threshold 0.65, on the
same complete validation split (job `142267`, completed):

| Decoder | Epoch-270 edge F1 / Branch F1 | Epoch-300 edge F1 / Branch F1 | Epoch-300 pred. edges |
|---|---:|---:|---:|
| Original threshold | 0.2684 / 0.00587 | 0.2671 / 0.00560 | 214.49 |
| Plain threshold 0.65 | 0.3421 / 0.00536 | 0.3405 / 0.00492 | 125.79 |
| Soft penalty 0.5 | 0.3611 / 0.00658 | 0.3594 / 0.00597 | 116.98 |
| Soft penalty 1 | 0.4036 / 0.00395 | 0.4031 / 0.00397 | 78.50 |

The qualitative trade-off replicates: the 300 initializer is slightly worse
at the modest penalty and does not change the conclusion. The type-aware
decoder appears more useful than type supervision alone, but the remaining
edge overcount (about 117 versus 52 target edges/patch at penalty 0.5) and
very low Branch F1 mean this is not yet a satisfactory graph predictor.
This is a validation-only comparison, not a held-out test result. See
[epoch-300 sweep data](T10_node_type_results/from300/soft_degree_sweep.csv)
and [plot](T10_node_type_results/from300/soft_degree_sweep.png). The same
preselected validation example is available as an
[interactive paired view](T10_node_type_results/from300/views/penalty_0.5/sample_000038_0076__sample_000616.html)
and [PNG](T10_node_type_results/from300/views/penalty_0.5/sample_000038_0076__sample_000616.png).
