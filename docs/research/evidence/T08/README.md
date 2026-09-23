# T08: controlled metric-failure checks

These are synthetic controls for the **metric implementation**, not evidence
that the extractor or a trained model is anatomically correct. Regenerate with
`python docs/research/evidence/T08/generate_metric_chart.py`; the definitions
and assertions are in `tests/vessel/test_metric_failure_matrix.py`.

The [chart](metric_failure_matrix.png) and [scores](metric_scores.json) show:

| Perturbation | Branch F1 | Node mAP | Edge mAP | SMD |
|---|---:|---:|---:|---:|
| False cross-link between two separate close vessels | 1.000 → 0.000 | 1.000 → 1.000 | 1.000 → 0.835 | 0 → 0.000031 |
| Bent interior path with the same two anchors | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 0.100 | 0 → 0.002276 |

SMD here uses the evaluator's **normalized patch coordinates**, so its raw
values are not millimetres. The short false link barely changes sampled curve
mass, while Branch F1 detects the connectivity error. Conversely, Branch F1
deliberately ignores interior bending; geometry/edge metrics react. Node mAP
alone detects neither mistake. Beta-1 error is zero in both cases. In the
false-link case beta-0 error is one, but the other tests show equal Betti
numbers can also hide wrong branch pairing.

The model evaluator's AP/AR protocol caps detections. The IXI 256-query
recipe therefore uses a 256-detection cap; the old 40-detection cap must not
be used as a headline score on dense vessel patches. Compare AP/AR only under
the same cap and protocol.
