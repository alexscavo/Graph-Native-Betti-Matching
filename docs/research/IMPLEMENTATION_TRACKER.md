# Vascular graph roadmap implementation tracker

## Purpose

This is the living execution record for
[`vascular_graph_roadmap.md`](vascular_graph_roadmap.md). Update it whenever a
task changes state, a decision is made, or validation produces evidence.

Status vocabulary: `not started`, `in progress`, `blocked`, `complete`.

A task is not complete until it has automated checks and inspectable evidence.
For spatial or quantitative work, save both machine-readable data and a plot or
screenshot under `evidence/<task-id>/` so results can be independently reviewed.

## Current position

- Current phase: Phase 2 — representation-quality metrics and real-data policy selection.
- Next task: finish morphology metrics and the ordinary-volume policy audit,
  including confirmation that 192 queries cover the full distribution. Keep
  the 72-million-voxel `topcow_ct_017` outlier deferred.
- Training changes: prohibited until the representation study is complete.

## Ordered task list

| ID | Task | Roadmap phase | Status |
|---|---|---:|---|
| T01 | Preserve pure degree-2 rings and add topology regression tests | 1 | complete |
| T02 | Enforce the IXI638/IXI661 annotation overrides and record provenance | 1 | complete |
| T03 | Emit junction-only, adaptive, and dense comparison representations | 1 | complete |
| T04 | Preserve and name an immutable dense reference before simplification | 1 | complete |
| T05 | Implement representation geometry, morphology, and complexity metrics | 2 | in progress |
| T06 | Run the rate-distortion study across sites and acquisition resolutions | 2 | in progress |
| T07 | Implement degree-2 contraction, physical anchor matching, and Branch F1 | 3 | not started |
| T08 | Validate metrics on the prescribed handcrafted failure cases | 4 | not started |
| T09 | Generate/smoke-test IXI patches and retrain with optimization unchanged | 5 | not started |
| T10 | Benchmark 120 versus 192 object queries on paired real IXI patches | 5 | complete |

## Decision log

### D01 — Representation of an anchorless ring

- Date: 2026-09-18
- Status: accepted for implementation
- Decision: represent each pure degree-2 ring with three deterministic
  degree-2 nodes and three ordered arcs.
- Reason: one node plus a self-loop and two nodes plus parallel edges are not
  reliably representable by the downstream simple adjacency matrix. Three
  nodes form a simple cycle, preserve beta-0/beta-1, and remain compatible with
  adaptive RDP subdivision.

### D02 — Three explicit source representations

- Date: 2026-09-18
- Status: accepted for implementation
- Decision: emit `graphs/junction_only`, `graphs/adaptive`, and `graphs/dense`
  as separate graph products. Patch generation defaults to `adaptive` when this
  layout is detected.
- Reason: representation quality must be measured before selecting a training
  target. Keeping the products separate makes comparisons reproducible and
  prevents dense-reference geometry from being confused with model ground truth.

### D03 — Dataset-agnostic extraction constraint

- Date: 2026-09-18
- Status: active design constraint
- Decision: extractor policies should use physical units or local normalized
  quantities. Dataset-specific constants are provisional and must be tested
  across acquisition resolutions, sites, vessel calibres, and geometries.
- Consequence: the current fixed voxel-based RDP and spur defaults are baselines,
  not accepted final policies. T06 must explicitly test their transferability.

### D04 — Dense means pre-smoothing reference

- Date: 2026-09-18
- Status: accepted for implementation
- Decision: `dense` is the ordered centerline after topology cleanup but before
  smoothing and RDP. It is one of the three representations, not a fourth
  auxiliary graph.
- Reason: smoothing is an intentional geometric transformation. Its distortion
  must be visible in the representation study instead of being silently folded
  into the reference.

### D05 — Global constrained-optimal adaptive simplification

- Date: 2026-09-19
- Status: accepted for full-dataset validation
- Decision: derive the adaptive graph once per full volume with the exact
  minimum-segment path through each dense branch. A candidate chord is valid
  only when its physical deviation is no greater than `0.75 ×` the minimum
  local lumen radius on the interval and at least 95% of its samples lie in the
  segmentation. Crop this fixed graph afterward using exact boundary
  intersections; never re-simplify individual patches.
- Reason: the shortest path on the ordered dense samples is globally optimal
  for the stated constraints and has monotonic complexity as tolerance is
  relaxed. On 14 real volumes it uses fewer degree-2 nodes than greedy RDP while
  also reducing P95 and worst-case centerline error.

### D06 — Handle graph-token overflow with uniform capacity

- Date: 2026-09-19
- Status: provisional pending the ordinary-volume inventory audit
- Decision: retain the single global graph representation and use 192 object
  queries for the real-data experiment. The data pipeline and loss must fail
  explicitly when a target exceeds the configured capacity.
- Reason: 45 of 2,326 nonempty 64³ patches exceed 120 nodes under D05, but none
  exceeds 192. Topological plus exact-boundary nodes have a maximum of 81;
  geometry nodes explain every observed overflow. Patch-specific simplification
  would make the target representation depend on crop placement.

## Evidence log

### E01 — Pre-fix ring failure

A handcrafted square-ring mask produced 72 skeleton voxels and `ring_count=1`,
but exported zero nodes and zero edges, so `VesselGraph.betti()` returned
`(0, 0)` instead of `(1, 1)`. This establishes the T01 regression target.

### E02 — Ring regression result

The focused test suite verifies both the basic three-node ring and its adaptive
RDP subdivision. Both return `(beta_0, beta_1) = (1, 1)`; 2 tests passed. Data
and visualization: [`evidence/T01/`](evidence/T01/).

### E03 — Annotation override provenance

The 397-subject planning run selects `nii(1).gz` for IXI638 and IXI661 and
records both rows as `preferred_duplicate`; all focused extraction tests pass.
Data and visualization: [`evidence/T02/`](evidence/T02/).

### E04 — Three-representation source build

An end-to-end synthetic NIfTI regression build emitted nine graph files: the three
Voreen files for each of `junction_only`, `adaptive`, and `dense`. The example
retained identical topology `(beta_0, beta_1) = (1, 0)` while using 2, 4, and 39
nodes respectively. This validates mechanics only; it is not scientific evidence
for choosing a representation. Data and visualization: [`evidence/T03/`](evidence/T03/).

### E05 — Dense-reference immutability

The regression check verifies that dense-reference samples remain on the raw
voxel-centre centerline while adaptive path samples reflect smoothing. The
example contains 144 concatenated dense samples, all integer-valued, versus 78
adaptive path samples with sub-voxel coordinates. Data and visualization:
[`evidence/T04/`](evidence/T04/).

### E06 — Real IXI three-representation 3D QC

Three real IXI crops cover Guys, HH, and IOP plus both 0.469-mm and 0.264-mm
in-plane acquisition resolutions. Interactive HTML reports overlay all three
representations on the real vessel surface. In every crop, beta-0 and beta-1 are
identical across representations. Adaptive versus dense node counts are 141 vs
924, 110 vs 807, and 314 vs 2141. This supports compression without topology
change, but does not yet establish geometric fidelity or resolution invariance.
Data, regeneration commands, and overview visualization:
[`evidence/real_ixi/`](evidence/real_ixi/).

### E07 — Real IXI bad-loop and chord diagnosis

User 3D inspection exposed dense cycle tangles and off-vessel straight chords.
Across the three real crops, fixed 2-voxel RDP produced 4, 6, and 3 edges with
less than 50% chord containment. Radius-normalized RDP reduced these to 0, 2,
and 3 at radius fraction 0.6, but beta-1 was unchanged for every policy. Thus
radius adaptation improves geometry but cannot repair segmentation-derived
topology. Data and visualization:
[`evidence/real_ixi_loop_diagnosis/`](evidence/real_ixi_loop_diagnosis/).

Training-data generation remains paused until the containment contract and
raw-MRA-supported false-bridge validation in
[`loop_failure_mitigation.md`](loop_failure_mitigation.md) are implemented and
reviewed.

### E08 — Real IXI coordinate and patch-budget verification

On a real IXI002 crop, raw voxel-centre paths have a 45.0° median sample-to-sample
turn angle; constrained smoothing reduces this to 6.6°. Adaptive graph nodes are
not voxel restricted (28.4% sub-voxel in the crop; 46.5% in the whole-volume
audit), and every training VTP edge is encoded as a straight two-point line.
Evidence: [`evidence/real_ixi_geometry/`](evidence/real_ixi_geometry/).

The preliminary whole-volume IXI002 audit finds mean/p95/p99 patch edge counts of
21.7/63.0/78.6. Five of 175 non-empty patches exceed the preferred 70-edge
ceiling and one reaches 139, although no patch exceeds 120 nodes. This means the
typical patch is acceptable but the complexity tail requires topology cleaning
and/or a budget-aware representation policy. CPU Slurm array job `2141634` is
pending for the three-representation, three-subject audit. Evidence:
[`evidence/real_ixi_patch_budget/`](evidence/real_ixi_patch_budget/).

## Dense intermediate and lumen-safe compact geometry (2026-09-18)

The extractor now skeletonizes each segmentation once into an immutable dense
centerline intermediate and deterministically derives junction-only, adaptive,
and dense graphs from it. Junction representatives retain their real skeleton
routes through non-convex clusters; smoothing rejects moves whose adjacent
segments leave the mask; adaptive RDP subdivision then adds samples until every
straight training edge remains inside the segmented lumen.

On the real IXI002 densest crop, all representations retain β₀/β₁ = 5/37. The
adaptive graph has 244 nodes / 276 edges versus 983 / 1015 dense, with minimum
edge-chord lumen containment 1.0. Junction-only has 98 / 130 but minimum
containment 0.106, so it remains a topology-preserving ablation rather than the
recommended geometric target. See
[`evidence/real_ixi_containment/`](evidence/real_ixi_containment/).

The reusable evaluator `scripts/compare_dense_centerlines.py` emits CSV, JSON,
PNG and interactive HTML, and `metrics/representation_geometry.py` supplies
continuous-polyline distance, curve precision/recall/F1, ACD, HD95, length,
tortuosity and normalized graph-complexity metrics.

The oversized Slurm audit job 2141634 was cancelled before execution. The audit
request is now one CPU and 30 minutes; no replacement job has been submitted
while local real-crop validation is sufficient.

## Rejected Vedo dense-centerline backend (2026-09-18)

The historical Vedo extractor was reinterpreted correctly: its dense NetworkX
sample graph is a centerline intermediate, followed by our compact graph
extraction. A reusable backend implemented surface-geodesic validation, orphan
handling, constrained sub-voxel smoothing, local digital-cycle cleanup and the
usual junction/branch reduction.

Real IXI evidence rejected it. On the IXI002 crop, adaptive complexity worsened
from 244 nodes / 276 edges / β₁=37 to 363 / 411 / β₁=49, and visual tangles
remained. The legacy extractor was restored as the default; Vedo is available
only by explicit selection for reproducibility. See
[`evidence/real_ixi_vedo/`](evidence/real_ixi_vedo/).

## New IXI and TopBrain restart (2026-09-18)

The rejected Vedo evidence was removed at the user's request, and evaluation
restarted on newly supplied data using the better pre-Vedo (`legacy`) backend.
The first subjects are improved-segmentation IXI122 and multiclass TopBrain
MRA subject 001. Fresh interactive 3-D reports, static 3-D views, measurements
and densest-64³ patch counts are in
[`evidence/new_datasets_baseline/`](evidence/new_datasets_baseline/).

Both adaptive representations preserve dense β₀/β₁ and have complete lumen
containment. Densest 64³ adaptive patches contain 99 IXI edges and 79 TopBrain
edges: below the 120-token capacity but above the preferred 60–70 range. The
next dataset-level test must measure percentiles across subjects rather than
optimizing against these two worst-density patches.

Those initial counts were produced by cropping the segmentation before graph
extraction. The corrected evaluation now extracts all three representations on
the full volume and only then crops the saved adaptive graph. Cropping is done
in voxel coordinates (important for TopBrain's oblique affine) and uses
`SourceGraph.crop_inherited()`: a crossing edge terminates at its last existing
centerline sample inside the patch, without an interpolated box-face node.
Results and full-volume visualizations are in
[`evidence/full_volume_then_crop/`](evidence/full_volume_then_crop/).

## Exact boundary intersections selected (2026-09-18)

Exact voxel-cell-face clipping is now the production patch policy. Full graphs
are transformed into voxel coordinates and clipped against
`[start - 0.5, start + crop_size - 0.5]`, which fixes world-axis-box errors for
oblique affines and gives neighboring non-overlapping patches the same shared
face. Synthetic intersection nodes retain `source_edge_index` and an explicit
boundary flag in VTP, yielding deterministic cross-patch correspondence while
remaining compatible with the existing model reader.

The last-inside-sample policy remains an ablation. On real densest IXI122 and
TopBrain-MR-001 patches it produced identical topology and token counts, but
terminated crossing edges approximately 0.42 voxel inside the true face. Exact
clipping therefore strictly improves boundary geometry without increasing the
graph budget in these tests. Metrics, JSON, PNG, and interactive comparisons
are in [`evidence/full_volume_then_crop/crop_policy_comparison/`](evidence/full_volume_then_crop/crop_policy_comparison/).

The complete extraction inventory contains 220 paired volumes: 170 IXI MRA,
25 TopBrain MR, 18 TopBrain CT training, and 7 labeled TopBrain CT test cases.
The resumable CPU-array extractor skeletonizes every full segmentation once and
writes separate junction-only, adaptive, and dense graph products. Adaptive is
the selected training representation; the other two are retained for controlled
comparison. Each volume has an atomic, configuration-fingerprinted completion
marker.

The Jean Zay launcher uses one CPU and partition-default memory per volume,
a 20-minute walltime, and an array concurrency cap of 12. Explicit `--mem` is
not valid on Jean Zay and is intentionally omitted.

Full-dataset extraction was submitted as Slurm array job `2161189` with tasks
`0-219%12`. Outputs are written below
`docs/research/artifacts/real_graph_dataset/`; rerunning the launcher safely
skips volumes whose configuration-fingerprinted completion marker and nine
graph files are present.

Job `2161189` was cancelled while still pending, before any cluster task ran,
because the strict baseline produced too many degree-2 nodes in the inspected
patches. A deterministic 15-volume, eight-policy tuning sweep now precedes the
production extraction. Its design and subset evidence are in
[`evidence/adaptive_tuning/`](evidence/adaptive_tuning/).

The tuning sweep was submitted as Slurm array job `2161680` (`0-14%4`, one CPU,
30 minutes per selected volume). It derives all eight policies from one dense
extraction per volume and writes exact-crop patch-budget statistics. Production
extraction remains intentionally paused until these results select one global
policy.

Model-only full-volume adaptive views (straight edges plus every termination,
degree-2 subdivision, and junction; no centerline polylines) are recorded in
[`evidence/full_volume_then_crop/adaptive_all_nodes/`](evidence/full_volume_then_crop/adaptive_all_nodes/).

## Initial greedy-RDP policy selection, now superseded (2026-09-19)

The pending Slurm sweep was cancelled and reproduced locally, sequentially, on
14 real volumes (six IXI, four TopBrain CT, four TopBrain MR). The 72-million-
voxel `topcow_ct_017` outlier is intentionally deferred. All policies were
derived from saved full-volume dense graphs, then audited with exact 64³,
stride-40 clipping; no crop-first skeletonization was used.

The first sweep showed that fixed tolerances up to four voxels were not strong
enough. A second sweep tested fixed 4/5/6-voxel and radius-aware 1/1.5/2×
tolerances, all with a 95% minimum straight-chord lumen-containment constraint.
Every policy preserved full-volume β₀/β₁ on 14/14 subjects. A follow-up geometry
audit selected radius-1.5× rather than radius-2×: it lowers P95 maximum error
from 0.634 to 0.603 mm, length-weighted shortening from 1.89% to 1.80%, and the
worst error normalized by local radius from 1.77× to 1.50×. The cost is about
1% more full-volume edges; patch P95 changes from 99 to 100.5 nodes and the
number of patches above 120 changes from 59 to 57. Radius-1.5× was therefore the
initial global dataset-agnostic policy before the exact solver below.

An independent audit found zero radius-1.5× edges below the configured 95%
containment floor (13,527 edges total; mean containment 99.82%). Budget data and plots are in
[`evidence/adaptive_tuning/results_stage2_14/`](evidence/adaptive_tuning/results_stage2_14/),
geometry data and plots in
[`evidence/adaptive_tuning/geometry_14/`](evidence/adaptive_tuning/geometry_14/),
and graph-only PNG/interactive HTML views for IXI425 and TopBrain-MR-002 in
[`evidence/adaptive_tuning/selected_radius1p5x/`](evidence/adaptive_tuning/selected_radius1p5x/).

Patch overflow is not caused by unavoidable topology. For radius-1.5×, the 57
patches above 120 contain on average 32.2 topological nodes, 20.6 exact boundary
nodes, and 91.3 degree-2 geometry nodes; topology plus boundary has a maximum of
81 over all 2,326 nonempty patches. Without changing or locally simplifying the
representation, 192 object queries cover every observed patch (maximum 186),
whereas 160 still misses 14. The selected strategy is therefore a single
full-volume representation plus a 192-query real-data model configuration,
subject to confirmation on the complete dataset and a GPU memory benchmark.
Training now raises an explicit error when a target exceeds query capacity,
instead of letting Hungarian matching silently omit unmatched ground-truth
nodes. Evidence is in
[`evidence/adaptive_tuning/overflow_14/`](evidence/adaptive_tuning/overflow_14/).

The audit also exposed and fixed a serialization defect: six-decimal world
coordinates in `nodes.csv` could move a junction by approximately 1e-7 voxel
across a voxel-cell face after reloading. Node coordinates now use 12 decimal
places, legacy tuning inputs recover full-precision nodes from `graph.vvg`, and
a regression test checks preservation of voxel-cell side. Forced tuning runs
now invalidate stale completion markers before overwriting artifacts, so an
interruption cannot masquerade as a completed regeneration.

## Global optimal simplification selected (2026-09-19)

Greedy RDP followed by containment repair was replaced by a constrained
shortest-path solver over each full-volume dense branch. The solver retains the
fewest dense samples whose chords satisfy both the local physical-error profile
and the lumen-containment requirement. This is performed before cropping, so
all overlapping or neighboring patches inherit the same graph and matching
face-intersection nodes.

The final 14-volume comparison selected optimal radius-0.75× at 95% minimum
chord containment over the previous greedy radius-1.5× baseline. Both preserve
topology on 14/14 volumes and neither has an audited edge below 95%
containment. The selected policy changes:

- full-volume degree-2 nodes: 9,682 → 8,962 (-7.4%);
- P95 patch nodes: 100.5 → 95;
- patches above 120 nodes: 57 → 45 (-21.1%);
- P95 edge maximum error: 0.603 → 0.585 mm;
- worst edge maximum error: 2.736 → 1.913 mm (-30.1%);
- length-weighted shortening: 1.80% → 1.94%.

The last change is the measured cost: the exact optimizer uses its allowed
deviation more consistently, producing slightly more average shortening even
though its P95 and maximum errors improve. On the IXI068 calibration volume,
optimal radius-0.5× further improves geometry but exceeds the RDP baseline edge
count; radius-1× and 1.25× are more aggressive than desired. Data, a direct
comparison plot, overflow
decomposition, and interactive IXI/TopBrain graph viewers are in
[`evidence/adaptive_tuning/`](evidence/adaptive_tuning/).

Stratified patch counts improve for IXI MRA and TopBrain CT. TopBrain MR is
approximately neutral: mean nodes remain 16.1, P95 changes from 45 to 46, and
patches above 70 change from 4 to 7, with a maximum of 115. The policy is thus a
promising dataset-agnostic rule, not yet a completed transferability claim.

## Change log

- 2026-09-18: Created the tracker and organized the research documents under
  `docs/research/`.
- 2026-09-18: Completed T01 by exporting pure rings as simple three-edge cycles
  and adding topology regression coverage.
- 2026-09-18: Completed T02 by enforcing the two documented annotation
  overrides and recording the selected source and variant in both provenance
  outputs.
- 2026-09-18: Completed T03 by emitting three explicit representation products,
  retaining adaptive as the automatically selected patch-training target, and
  validating the layout end to end.
- 2026-09-18: Completed T04 by defining `dense` as the immutable centerline
  before smoothing and RDP, recording the policy in provenance, and bumping the
  representation schema so old completion markers cannot skip regeneration.
- 2026-09-19: Replaced greedy RDP plus repair with exact constrained branch
  simplification, selected radius-0.75× on 14 real volumes, and retained uniform
  192-query handling rather than patch-specific target changes.
- 2026-09-20: Prepared the IXI-only 120-versus-192 query benchmark. Seven
  patient-split real volumes yielded 884 64-cubed patches; 59 exceed 120 nodes,
  none exceeds 192, and the maximum is 177. Added a paired full-training-step
  benchmark, overflow smoke test, selection tables, and visualization under
  [`evidence/ixi_query_benchmark/`](evidence/ixi_query_benchmark/).
- 2026-09-20: Completed T10 on one H100 with 100 full training steps per case.
  Moving from 120 to 192 queries increased mean step time by 2.02%, the 5%-
  trimmed mean by 1.53%, and absolute peak allocation by 0.101 GiB (3.33%). The
  maximum real IXI patch (177 nodes) completed successfully. Raw steps, GPU
  telemetry, logs, and plots are recorded in
  [`evidence/ixi_query_benchmark/`](evidence/ixi_query_benchmark/).
- 2026-09-20: Estimated the proposed 50-epoch Plants+TopBrain pretraining and
  100-epoch TopBrain specialization schedule from the loader/sampler semantics,
  manifest-derived patch inventory, and measured H100 step cost. The current
  one-to-four-H100 launcher layout is approximately 9.5 compute hours, with a
  7.3–13.0-hour planning range before queue time. Assumptions and visualization
  are in [`evidence/topbrain_training_time_estimate/`](evidence/topbrain_training_time_estimate/).
