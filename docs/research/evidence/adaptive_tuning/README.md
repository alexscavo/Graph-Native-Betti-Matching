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

## Superseded greedy-RDP decision (2026-09-19)

The sweep completed locally on 14 ordinary-sized volumes; the 72-million-voxel
`topcow_ct_017` outlier is intentionally deferred. A stronger second stage
tested fixed 4/5/6-voxel and radius-aware 1/1.5/2× policies at 95% minimum chord
containment. All candidates preserved topology on 14/14 subjects.

Radius-1.5× was initially selected. It improves on radius-2× with P95 edge
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

## Final tuning-subset decision: constrained optimum at radius-0.75×

The greedy RDP policy is now superseded by an exact branch-level solver. For
each dense full-volume branch, sample indices form a directed acyclic graph.
An arc is allowed when the straight chord:

1. stays within the configured physical error of every skipped dense sample;
2. uses the minimum local tolerance across that interval; and
3. has at least 95% chord containment in the segmentation.

The shortest path from the branch start to end therefore retains the fewest
degree-2 samples possible under those constraints. It is deterministic, and
relaxing the tolerance cannot increase the optimal node count. This is not a
patch-budget simplifier: it runs once on the full-volume graph, followed by
unchanged exact boundary cropping.

The selected policy is `optimal_radius_0p75x_c095`. Across the 14 ordinary-sized
real volumes (6 IXI MRA, 4 TopBrain MR, 4 TopBrain CT), relative to greedy
`radius_1p5x_c095`:

| Metric | Greedy RDP 1.5× | Optimal 0.75× | Change |
|---|---:|---:|---:|
| Topology preserved | 14/14 | 14/14 | unchanged |
| Full-volume degree-2 nodes | 9,682 | 8,962 | -7.4% |
| P95 nodes per nonempty 64³ patch | 100.5 | 95.0 | -5.5% |
| Patches above 120 nodes | 57 | 45 | -21.1% |
| P95 edge maximum error | 0.603 mm | 0.585 mm | -3.1% |
| Worst edge maximum error | 2.736 mm | 1.913 mm | -30.1% |
| Worst error / local median radius | 1.496 | 0.750 | -49.9% |
| Length-weighted shortening | 1.798% | 1.938% | +0.140 pp |
| Mean chord containment | 99.817% | 99.652% | -0.165 pp |
| Edges below 95% containment | 0 | 0 | unchanged |

The selected policy deliberately accepts a small increase in average shortening
and a small decrease in mean containment. In exchange it reduces graph size and
improves the P95, normalized maximum, and absolute worst-case centerline errors.
On the IXI068 calibration volume, radius-0.5× improves fidelity further but
creates more edges than the RDP baseline, so it was not expanded to the 14-volume
sweep. Radius-1× and 1.25× use fewer nodes but allow more error than desired.
The stratified plot prevents the aggregate result from hiding a small exception:
IXI MRA and TopBrain CT complexity improve, while TopBrain MR is essentially
flat in mean nodes and its P95 rises from 45 to 46 (patches above 70 rise from 4
to 7). Its maximum remains 115, below 120. This is acceptable for the tuning
subset but is one reason the complete-dataset transfer check is still required.

Evidence:

- direct comparison: [`selected_policy_comparison/`](selected_policy_comparison/);
- rejected 0.5× calibration: [`optimal_0p5_one/`](optimal_0p5_one/);
- rejected 1×/1.25× summaries: [`results_optimal_14/`](results_optimal_14/)
  and [`geometry_optimal_14/`](geometry_optimal_14/);
- optimal policy counts: [`results_optimal_0p75_14/`](results_optimal_0p75_14/);
- all-edge geometry audit: [`geometry_optimal_0p75_14/`](geometry_optimal_0p75_14/);
- exact-crop overflow decomposition: [`overflow_optimal_0p75_14/`](overflow_optimal_0p75_14/);
- static and interactive graph-only viewers: [`selected_optimal_0p75/`](selected_optimal_0p75/).

The overflow result remains a capacity decision rather than a representation
change: 45/2,326 nonempty patches exceed 120 nodes, 7 exceed 160, and none
exceeds 192. Topological plus exact-boundary nodes never exceed 81. A uniform
192-query real-data configuration is therefore supported on this tuning subset,
subject to confirmation on the 219 ordinary-sized volumes. The 72-million-voxel
`topcow_ct_017` outlier remains explicitly deferred.
