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
Model integration must first convert normalized patch coordinates to physical
world coordinates. Remaining Phase 4 metric cases and end-to-end evaluation
are tracked in `docs/research/IMPLEMENTATION_TRACKER.md`.
