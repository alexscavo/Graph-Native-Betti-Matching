# Early Phase 3 metric check

The standalone `metrics/branch_connectivity.py` accepts **world-coordinate
millimetres**, not model-normalized coordinates. It contracts degree-2 chains
without discarding their ordered points, matches topological anchors one-to-one
within a physical threshold (here 0.5 mm), then counts corresponding branches.
Branch trajectories are deliberately not part of the Branch F1 score. Multiple
branches between the same anchors retain multiplicity; pure degree-2 rings
are one virtual closed branch anchored at a length-weighted spatial centroid.

On two real saved full-volume graph families, adaptive vs dense scores 1.0,
while controlled deletion of one real adaptive edge lowers it. [Raw scores](real_graph_checks.csv)
and the [zoomed plot](real_graph_checks.png) show the results. This confirms
the implementation distinguishes a topological break; it is **not a model
accuracy measurement**. The deleted edge has a small effect on whole-volume
F1 because hundreds of other branches remain correct.

The focused failure-case unit tests live in `tests/test_branch_connectivity.py`.
The metric currently rejects direct self-loop edges; our normal adaptive and
dense products express rings as spatially embedded multi-edge cycles. A
junction-only full-volume export can contain self-loop edges and must not be
passed to this implementation without an explicit representation conversion.
The model evaluator now optionally computes physical Branch F1 on generated
vessel patches when `evaluation.protocol.branch_threshold_mm` is set. The
IXI-only configuration uses 1.0 mm. The conversion reconstructs world
coordinates from `patch_index.csv`, `generation_config.json`, the original
source NIfTI affine in `source_manifest.json`, and the normalized graph points.
The saved patch NIfTI headers have identity affines and must **not** be used
to infer physical spacing. Evaluation fails clearly when provenance is absent.
Default/legacy evaluation is unchanged when the metric is disabled. Isolated
predicted nodes are excluded from anchor matching since they contain no branch
(their counts remain available in the legacy diagnostics).
The dataset-level `branch_f1` is computed from summed TP/FP/FN (micro F1),
not by averaging per-patch scores. Two empty graphs have per-patch F1=1,
so micro aggregation avoids inflated dataset scores from empty validation
patches. Isolated-node false positives are assessed separately by node/count
metrics; Branch F1 alone does not penalize them.

In [two real IXI validation patches](ixi_patches/ixi_patch_edge_ablation.png)
from different volumes, self-comparison gives Branch F1 = 1.000, whereas
removing a single saved graph edge drops the score to 0.857 and 0.783. The
[per-patch JSON](ixi_patches/ixi_patch_edge_ablation.json) includes physical
axis scale and the deleted edge. This is a controlled oracle/ablation check,
**not a trained-model evaluation or independent anatomical ground truth**.
`scripts/validate_branch_metric_on_ixi_patches.py` regenerates the evidence.
Remaining Phase 4 multi-metric failure-case validation and eventual model
training/evaluation are tracked in `docs/research/IMPLEMENTATION_TRACKER.md`.
