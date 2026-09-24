# Opt-in domain-adaptation experiment

## Status

Adversarial domain adaptation remains **off by default** in Graph-Native Betti
Matching. The controlled Plants-source/IXI-target experiment reintroduces it
only in `configs/pretrain_plants_ixi_domain_types_50ep.yaml`. The IXI-only
fine-tuning configuration disables it because it has no source/target mixture.

Source and target labels remain in the model-facing batch. They describe dataset
roles and may be used for sampling, filtering, and diagnostics; their presence
does not mean that adversarial domain adaptation is enabled.

## Why it was removed

The pre-focal, pre-Betti reference repository contained image-level and
instance-level domain discriminators and calculated a domain loss. However, its
committed 3D configurations enabled only `boxes`, `class`, `cards`, `nodes`, and
`edges`. The criterion constructed `total` only from that list, and the trainer
called `backward()` only on `total`. Consequently, the domain loss was logged but
did not update either discriminator or the graph-extraction network.

Reference implementation:

- model discriminators: <https://github.com/alexscavo/Vascular-Graph-Extraction/blob/main/3d/models/relationformer.py#L69-L71>
- domain-loss calculation: <https://github.com/alexscavo/Vascular-Graph-Extraction/blob/main/3d/training/losses.py#L749-L777>
- optimized-total dispatch: <https://github.com/alexscavo/Vascular-Graph-Extraction/blob/main/3d/training/losses.py#L850-L868>
- reference 3D loss list: <https://github.com/alexscavo/Vascular-Graph-Extraction/blob/main/3d/configs/mixed_plants_synth_HNS_3D.yaml#L138-L145>
- backward call: <https://github.com/alexscavo/Vascular-Graph-Extraction/blob/main/3d/training/trainer.py#L318-L345>

This was therefore dead training logic in the effective baseline, despite the
adversarial flags and the `domain_loss` curve. Removing it preserves
the optimized objective while reducing computation, memory use, checkpoint
size, and configuration ambiguity. Git history remains the source for the old
implementation.

## Intended objective

The archived design followed the domain-adversarial pattern used by
Domain-Adaptive Faster R-CNN rather than the core RelationFormer objective. A
gradient-reversal layer was applied before each domain discriminator. The
intended loss was

```text
L_domain = w_image * NLL(image_domain_prediction, domain)
         + w_graph * NLL(graph_domain_prediction, domain)
         + w_consistency * MSE(mean_image_prediction, graph_prediction)

L_total = L_graph + w_domain * L_domain
```

During backpropagation, gradient reversal multiplies the feature gradient from
`L_domain` by `-alpha`, encouraging domain-invariant features while the
discriminators learn to classify source versus target.

## Historical configurations

The effective GitHub baseline had adversarial flags set but omitted `domain`
from `TRAIN.LOSSES`, so `W_DOMAIN` had no effect.

A separate cluster run did activate the experiment with the following relevant
legacy settings:

```yaml
TRAIN:
  DOMAIN_LR: 0.00007
  DOMAIN_WEIGHTING: [0.05, 0.95]
  IMAGE_ADVERSARIAL: true
  GRAPH_ADVERSARIAL: true
  CONSISTENCY_REGULARIZATION: true
  ALPHA_COEFF: 1.0
  LOSSES: [boxes, class, cards, nodes, edges, domain]
  W_DOMAIN: 1.0
```

That run is an experimental variant, not evidence that domain adaptation was
part of the best committed baseline.

The old fields `UPSAMPLE_TARGET_DOMAIN` and `COMPUTE_TARGET_GRAPH_LOSS` were not
intrinsically adversarial. Their active replacements are:

```yaml
data:
  mixed_sampling:
    balance_source_target: true

loss:
  supervise_target_graphs: true
```

## Reactivation contract and current implementation

The experiment uses one explicit, self-contained configuration block. Its
implemented field names are:

```yaml
domain_adaptation:
  enabled: true
  image_adversarial: true
  graph_adversarial: true
  consistency_regularization: true
  alpha_coefficient: 1.0
  learning_rate: 0.00007
  hidden_size: 2048
  class_weights: [0.05, 0.95]

loss:
  enabled: [boxes, class, cardinality, nodes, edges, node_types, domain]
  supervise_target_graphs: true
  weights:
    domain: 1.0
```

`enabled: true` must have all of the following observable effects:

1. construct only the requested discriminator modules;
2. add their parameters to an explicit optimizer group;
3. calculate `L_domain`;
4. include `loss_weight * L_domain` in the tensor passed to `backward()`;
5. log both raw and weighted domain losses;
6. fail configuration validation if there is no source/target mixture.

`enabled: false` must construct no discriminators, perform no discriminator
forward passes, add no optimizer group, and emit no domain-loss metric. The loss
must never be enabled indirectly through an independent list of names.

The implementation follows the Image-to-Graph-Vessel reference's two
2048-wide discriminators, image-feature and flattened-decoder-token inputs,
two-pass gradient reversal, weighted image/instance NLL, consistency MSE,
logistic reversal schedule, and separate discriminator learning rate. The
target is sampled with replacement at the reference's expected 50% rate,
using a combined-dataset epoch length. Unlike the old committed baseline,
`domain` is present in `loss.enabled`, so its weighted term is actually
backpropagated. Both Plants and IXI graphs are supervised during pretraining.

`tests/test_domain_adaptation.py` verifies discriminator and feature gradients,
the optimized loss, disabled-state behavior, the sampler, and checkpoint
compatibility. A controlled ablation of image-only, graph-only, combined, and
combined-with-consistency remains future work; the current request is to
reproduce the reference's combined objective in one mixed run.
