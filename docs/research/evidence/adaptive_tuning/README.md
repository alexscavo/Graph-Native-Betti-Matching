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
