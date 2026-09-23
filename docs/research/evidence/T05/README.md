# Phase 2: representation-quality study (2026-09-21)

## Decision

The compact adaptive representation is an appropriate **provisional training
target** for the current extracted topology. Proceed to Phase 3 metric work;
do not claim that the segmentation-derived topology is anatomically correct
or that the extractor transfers to every dataset. No model was trained here.

We compared **three separately saved full-volume graph products**, not graphs
re-extracted independently in patches. Six volumes each from IXI Guys, HH and
IOP, and TopBrain CT and MR (30 total) were sampled deterministically across
available in-plane voxel spacings. Physical world coordinates are in mm. Dense
is the unsmoothed post-topology-cleanup voxel-centre reference, *not* manual
anatomical ground truth. Cross-representation branch matching uses inherited
topological anchor identities, with one-to-one spatial assignment for parallel
branches. Distances are sampled every 0.5 mm along each paired branch to its
counterpart's continuous straight-edge polyline, avoiding credit from a
nearby **different** vessel. The resulting Curve F1 is branch-paired, not the
unpaired network-wide Curve F1 proposed for future model evaluation.

Across all 30 subjects, adaptive has fewer nodes than dense, higher branch-
paired Curve F1 and lower ACD than junction-only, and the same extracted β₀/β₁
as dense. At a 0.5-mm tolerance, mean adaptive Curve F1 spans **0.757**
(TopBrain CT) to **0.893** (IXI Guys); junction-only spans 0.227–0.306.
Adaptive mean ACD is **0.271–0.366 mm**, vs 2.261–4.303 mm for junction-only.
At 1 mm, adaptive mean Curve F1 is 0.966–0.998 across the five strata. Mean
adaptive node counts are 550–1,352 per whole volume, vs 3,943–6,760 dense
(individual volumes and medians/P95 are in the CSVs). This supports the
compactness–geometry trade-off, not a claim of perfect centerline accuracy.

The apparent **14.7–16.1%** length loss from *raw* dense to adaptive is mostly
the change from a voxel-staircase path to the constrained smoothed path:
13.4–14.5 percentage points. Only another **1.4–2.2 percentage points** is
from turning that smoothed path into straight adaptive graph edges. Do not
attribute the whole length change to the simplification policy; neither the
raw voxel path nor its smoothed version is an independent manual standard.
Spline curvature and junction-angle error relative to a digital staircase
should not be reported as validated anatomical biomarkers.

The sampler is arc-uniform *within each branch*; repeated branch endpoints can
overweight very short branches in the reported curve F1/ACD. F1 at 0.5 mm
changed by 0.0038 on one IXI and 0.0056 on one TopBrain CT volume when the
sample interval was halved from 0.5 to 0.25 mm; ACD moved <0.007 mm. This
checks sampling sensitivity on **two**, not all 30, subjects. The sampled IXI
volumes have in-plane spacings around 0.46–0.49 mm; TopBrain CT spans
0.398–0.553 mm and TopBrain MR 0.297–0.352 mm. Dataset-agnostic transfer and
false-loop correctness need independent future validation. The present
topology equality is expected because all three graphs derive from the same
extraction; it does **not** validate segmentation quality.

## Inspectable evidence

- [Per-site comparison](stratified_30/comparison.png),
  [per-volume rate–distortion](stratified_30/rate_distortion.png),
  [length decomposition](stratified_30/length_decomposition.png).
- Real graph-only 3D views: [IXI122 interactive](ixi122_graphs.html),
  [IXI122 static](ixi122_graphs.png),
  [TopBrain CT-001 interactive](topbrain_ct_001_graphs.html),
  [TopBrain CT-001 static](topbrain_ct_001_graphs.png). The interactive control
  switches among the three graph files, with endpoint/degree-2/junction colors;
  no hidden VVG centerline is drawn as a training edge.
- [Per-volume measurements](stratified_30/volumes.csv),
  [per-branch measurements](stratified_30/branches.csv),
  [median/P95 by stratum](stratified_30/strata_robust.csv),
  [validation and physical-spacing coverage](stratified_30/validation.json),
  [selected subjects](stratified_30/selection.json),
  [smoothing-vs-edge-length components](stratified_30/length_decomposition.csv).
- [0.25-mm IXI check](sampling_ixi_hh_0p25/volumes.csv),
  [0.25-mm TopBrain check](sampling_topbrain_ct_0p25/volumes.csv).

The original study commands are retained below for provenance. The one-off
research scripts used by these commands were removed during repository cleanup;
the saved measurements and views above are the reviewable result.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/study_full_volume_representations.py --max-per-stratum 6 --output docs/research/evidence/T05/stratified_30
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/audit_t05_smoothing.py --report-dir docs/research/evidence/T05/stratified_30
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/plot_t05_rate_distortion.py --report-dir docs/research/evidence/T05/stratified_30
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/validate_t05_report.py --report-dir docs/research/evidence/T05/stratified_30 --sampling-check docs/research/evidence/T05/sampling_ixi_hh_0p25 --sampling-check docs/research/evidence/T05/sampling_topbrain_ct_0p25
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_full_volume_representation_study.py tests/test_representation_geometry.py tests/test_branch_evaluation.py
```

The initial five-subject pilot was removed after the 30-subject study
superseded it; all decisions here use the 30-subject study. An earlier unbounded local
220-volume process terminated before writing a complete report, so its
in-memory results are **not** evidence. No exhaustive 220-volume T05 claim is
made or needed for the next metric-implementation step.
