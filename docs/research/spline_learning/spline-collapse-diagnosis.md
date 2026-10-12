# Scratch B-spline collapse diagnosis — 2026-10-08

## Scope and established result

Diagnose the completed three-arm, 300-epoch SyntheticMRI comparison before
changing the representation or launching another long run. The final models
have common whole-curve F1 of 0.23051 (straight), 0.00135 (B-spline), and
0.24711 (adaptive); see [full comparison](scratch-comparison-results.md).

The spline run failed to learn the training population. At epoch 300 its
training/validation node losses are 0.20171/0.20324, versus 0.00731/0.07341
for straight edges and 0.00758/0.08366 for adaptive. Spline node loss remains
around 0.21–0.24 from early training onward. This predates the epoch-171
filesystem interruption and resume. Both best and final checkpoints fail.
This differs from the baselines' large training/validation gap.

## Checks performed

1. Replayed saved predictions for the first two test patches. Spline graphs
   have identical edge indices and 38 nodes, with maximum normalized node
   coordinate change 1.79e-7. Both baselines change substantially. This is
   input-independent collapse, rather than a subtle curve-fitting error.
2. GPU job **762225**, eight fixed validation patches, batch size four,
   all three final checkpoints: compared saved BN running statistics with
   current batch statistics, keeping dropout and the rest of the model in
   evaluation mode. No weights/checkpoints were saved or changed.
3. One initial training batch: measured weighted spline versus ordinary graph
   gradient norms at shared tokens, node coordinates, and image features.
   No optimization step was performed.
4. GPU job **762449**, four fixed validation patches, spline and straight final
   checkpoints: recorded transformer attention support and feature dependence.
   For the spline model only, zeroed cross-attention offsets in disposable
   memory to sample inside the feature grid. This was an intervention to
   localize the failure, not a new evaluated model or a checkpoint repair.

| Model | Node loss, saved BN | Node loss, batch BN | Largest between-input node change, saved / batch BN |
| --- | ---: | ---: | ---: |
| Straight | 0.06868 | 0.07287 | 0.7461 / 0.8065 |
| B-spline | 0.18983 | 0.18983 | 1.79e-7 / 1.79e-7 |
| Adaptive | 0.08500 | 0.08566 | 0.7852 / 0.7802 |

All checkpoints load compatibly with no missing active tensors. Each has 89
BN layers and 750,000 recorded BN updates. Input differences are identical
across arms. The spline encoder still produces different image features,
but decoder token differences are only about 7e-8. BN mode does not restore
input sensitivity. These checks strongly disfavor stale inference BN
statistics as the explanation; they do not prove that normalization never
affected optimization during training.

### Transformer localization

For the first pair of validation patches, the spline model's projected CNN
feature mean absolute difference is **0.85446**. Across transformer encoder
layers, it falls to 0.00424, 0.000427, 1.72e-5, 3.12e-7, 4.23e-8, and finally
**3.42e-8** at encoder output. The straight model preserves a difference of
**0.87522** at encoder output. Thus the CNN remains input-sensitive, while
the spline model's transformer encoder nearly removes that dependence.

Most deformable-attention samples lack interpolation support. Spline decoder
cross-attention supported weight is approximately 1e-7, 0, 0, and 0.07436
across its four layers. However, the straight baseline also has poor sampling
coverage, including zero supported encoder-attention samples on this batch.
Its residual/FFN paths still preserve image dependence. Coverage alone is
therefore insufficient to identify the cause of the spline-specific collapse.

Zeroing spline cross-attention offsets restores supported weight to 1 in
every decoder layer but leaves node differences at **1.19e-7**. The incoming
transformer memory has already collapsed. This intervention does not repair
the prediction, and no modified weights were saved. No further diagnostic
tests are needed before choosing a controlled short optimization experiment.

The initial weighted spline/graph gradient ratios were **1.35 at tokens,
7.49 at node coordinates, and 1.24 at image features**. The coordinate
gradient cosine was +0.779, so this batch demonstrates scale imbalance,
not directly opposing coordinate gradients. It is one diagnostic batch,
not a population estimate or proof of the initiating cause.

## Likely optimization issue

The spline loss uses millimetres and backpropagates through predicted endpoint
coordinates as well as shared features. Early Hungarian assignments and
endpoints are inaccurate; much of this loss initially concerns localization,
not tortuosity. The fixed weight 1.474 was calibrated at shared tokens using
the original four training patches (both scratch and pretrained
initializations, in evaluation mode). That calibration did not constrain
the coordinate-gradient ratio during full-population training.

Four-patch overfitting demonstrated that the parameterization is learnable.
It does not establish that the same optimization settings work across
10,000 varied patches. Identifying the exact initiating cause would require
a controlled short training comparison; no causal claim from loss magnitude
alone is justified.

## Proposed next experiment

Use fixed training membership, seed, and disjoint validation. Keep the
existing held-out test set untouched during solution selection:

- First compare the current spline weight (1.474) with a tenfold smaller
  weight (0.1474) on a modest diverse training subset. Change only this
  variable; the smaller value is a diagnostic candidate, not an established
  optimum. Monitor graph learning and transformer input sensitivity alongside
  curve error. If both settings work on the small subset, expand the training
  population before drawing conclusions about the original collapse.
- If necessary, next test graph-learning warmup followed by a gradual
  spline-weight ramp. Calibrate on several training batches in training mode,
  checking coordinate and shared-feature gradients separately.
- If endpoint coupling remains unstable, compare temporarily detaching
  predicted endpoints from the spline loss while ordinary node losses still
  train them; retain predicted endpoints at inference.
- Gate a long run on improving training graph loss, input-sensitive decoded
  outputs, and validation topology/geometry. A short successful import or
  finite-gradient smoke alone cannot detect this failure.

These are proposals; the training objective and completed benchmark results
have not been changed. No new full training or dataset-wide test was launched.

## Artifacts

Reproducer: `Graph-Native-Betti-Matching-Representation/scripts/diagnose_spline_collapse.py`.
Raw BN/gradient measurements:
`Graph-Native-Betti-Matching-Representation/experiments/spline_learning/metadata/diagnosis/bn-differential.json`.
Attention measurements: the adjacent `attention-differential.json`.

[One actual epoch-300 prediction](../experiments/synthetic_mri_scratch_spline_prediction/results/sample_000020_0470.html)
and [its static projections](../experiments/synthetic_mri_scratch_spline_prediction/results/preview.png).
Download the HTML alone and open it locally. This fixed sample contains 38
predicted nodes/3 edges versus 22 reference nodes/17 branches. It was also
used in earlier viewers and was not selected for prediction quality.

Training/native spline diagnostics use eight-control fitted targets. Common
cross-representation metrics use the original full generator guide polylines
and shared endpoint topology. Both guides and segmentation are uncertain
references. All six original benchmark tests used saved BN statistics with
zero recalibration batches; the diagnostic above is separate.
