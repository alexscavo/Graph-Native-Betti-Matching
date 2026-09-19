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

Radius-1.5× is selected for production. It improves on radius-2× with P95 edge
maximum error 0.603 mm, length-weighted shortening 1.80%, and maximum error
1.50 times the local median radius. None of 13,527 audited edges falls below
95% lumen containment. It adds only about 1% more edges than radius-2×, while
the observed patch maximum remains 186 nodes. Exact 64³ crop statistics are in
`results_stage2_14/`, all-edge geometry evidence in `geometry_radius1p5_14/`,
and two graph-only real-data viewers in `selected_radius1p5x/`.

`overflow_14/` shows why patches exceed 120 nodes. The maximum topological plus
exact-boundary requirement is only 81 nodes; degree-2 geometry nodes account
for every overflow. We do not simplify those patches independently. A uniform
192-query real-data configuration covers every observed patch, while keeping
one globally defined graph representation. The complete dataset must be
audited before 192 becomes a guaranteed final bound.
