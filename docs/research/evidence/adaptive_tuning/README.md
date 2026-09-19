# Adaptive graph tuning on representative real volumes

The cancelled baseline array job `2161189` had not started. Tuning therefore
precedes full-dataset extraction, avoiding generation of 220 graph families
with parameters already suspected to overproduce degree-2 nodes.

The deterministic subset contains 15 of 220 paired volumes:

- 6 IXI MRA;
- 4 TopBrain MR;
- 3 TopBrain CT training; and
- 2 labeled TopBrain CT test volumes.

Selection uses maximin coverage of volume geometry, voxel spacing, foreground
count, and foreground fraction. `subset_manifest.png` shows the selected cases
against the complete inventory.

Each volume is skeletonized once. Eight adaptive policies are derived from the
same immutable dense graph:

| Policy | Geometric tolerance | Minimum chord containment |
|---|---:|---:|
| fixed baseline | 2.0 voxels | 1.00 |
| fixed mild | 2.5 voxels | 0.99 |
| fixed moderate | 3.0 voxels | 0.99 |
| fixed moderate relaxed | 3.0 voxels | 0.98 |
| fixed strong | 4.0 voxels | 0.98 |
| radius-aware strict | 0.25 × local radius | 0.99 |
| radius-aware moderate | 0.50 × local radius | 0.99 |
| radius-aware strong | 0.75 × local radius | 0.98 |

Every policy is audited using exact voxel-cell-face crops on a 64³, stride-40
grid. Per-volume results record full graph topology, degree-2 counts, and patch
node/edge mean, P95, P99, maximum, and counts above 70 and 120.

Radius-aware tolerance is an alternative to fixed tolerance, not a fixed-limit
modifier. This is required for the intended behavior: thick straight vessels
may use longer edges, while thin vessels receive a tighter geometric bound.

## Completed real-data decision (2026-09-19)

The sweep completed locally on 14 ordinary-sized volumes; the 72-million-voxel
`topcow_ct_017` outlier is intentionally deferred. A stronger second stage
tested fixed 4/5/6-voxel and radius-aware 1/1.5/2× policies at 95% minimum chord
containment. All candidates preserved topology on 14/14 subjects.

Radius-2× is selected for production. Fixed-5 has a marginally smaller patch
tail, but radius-2× controls errors in thin vessels much better: the maximum
edge error normalized by median local radius is 1.77× rather than 5.01×. Its
P95 edge maximum error is 0.634 mm, length-weighted shortening is 1.89%, and
none of 13,393 audited edges falls below 95% lumen containment. Exact 64³ crop
statistics are in `results_stage2_14/`, all-edge geometry evidence in
`geometry_14/`, and two graph-only real-data viewers in `selected_radius2x/`.
