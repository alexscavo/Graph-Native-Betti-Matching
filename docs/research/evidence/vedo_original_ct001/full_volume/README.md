# Full-volume CT-001: current versus original Vedo

## Direct comparison of final graph outputs

The answer to the production-pipeline question is in
`dataset_graph_vs_vedo_full.html` (full volume, selectable graphs and mesh) and
`dataset_graph_vs_vedo_focus.html` (the same saved graphs in a local mesh view).
`dataset_graph_vs_vedo_focus.png` is a quick side-by-side picture. The blue
graph is loaded **directly from the adaptive `nodes.csv` and `edges.csv` in
`/datasets/TopBrain.../vascular_graphs/optimal_radius_0p75x_c095/ct/train/topcow_ct_001`**;
its `complete.json` confirms the graph policy. It is the actual graph
produced for the dataset, not a raw centerline proxy. The orange graph starts
from the original Vedo full-volume centerline and uses the same adaptive graph
derivation settings after the documented mask repair below.

| Final graph on CT-001 | Saved graph in `/datasets` | Original Vedo + mask repair + same adaptive graph policy |
| --- | ---: | ---: |
| Nodes / edges | 581 / 649 | 646 / 684 |
| Components / cycle rank β₁ | 5 / 73 | 6 / 44 |
| Endpoints / junctions | 37 / 144 | 108 / 178 |
| Nodes outside segmentation | 0 | 0 |
| Edge samples outside segmentation | 69 / 12,269 | 55 / 11,802 |
| Total edge length | 2,004 mm | 1,915 mm |

The original Vedo centerline cannot go through the dataset's normal
95%-contained-chord graph policy unchanged. To make a final graph under that
policy, 78 Vedo points were moved to the nearest foreground voxel (median
movement 0.41 mm, maximum 0.60 mm). Of 57 adjacent edges that still crossed
too much background, 52 were rerouted through the mask using six-connected
voxel paths (maximum 10 steps), and 5 without a local in-mask route were
dropped. The repair inserted 94 dense degree-2 points before graph
simplification. Those operations are part of the **Vedo branch in this final
comparison**, not part of the unmodified original Vedo extractor. Details and
the five removed edge IDs are in `vedo_repair_trial.json`; the final Vedo graph
arrays are in `vedo_repaired_production_graph.npz`.

The Vedo branch yields a smoother-looking graph in the selected region and
fewer cycles overall. The extra endpoints and five dropped edges mean that a
lower cycle count cannot yet be called better vessel topology. The HTML views
let us judge specific suspected loops against the segmentation surface. The
numeric graph comparison is in `dataset_graph_vs_vedo.json`.

This is a **full-volume** comparison on TopBrain `topcow_ct_001.nii.gz`
(284 × 327 × 243 voxels, 79,177 foreground voxels). There is no patch or
extraction crop. The 24-mm local views are only camera/view selections from
the saved full-volume results. Job 50525 completed on 2026-09-22; see
`run-50525.log` and `comparison.json` for provenance and timings.

The two dense centerlines come from the same binary segmentation. `current`
uses the legacy centerline extraction and production topology cleanup;
`original_vedo` runs `vedo_extractor_original.py` via its
`process_single_case` with its original algorithm parameters. A compatibility
wrapper only converts the Numpy affine to the Vedo 2026.6.1 transform type.
The original script's Slicer X/Y display flip is undone before measurement
and graph derivation. Both use the same voxel-to-world affine and mask.

| Full-volume measure | Current | Original Vedo |
| --- | ---: | ---: |
| Dense nodes / edges | 3,161 / 3,229 | 3,171 / 3,212 |
| Connected components | 5 | 4 |
| Cycle rank β₁ | 73 | 45 |
| Endpoints | 37 | 102 |
| Junctions | 144 | 178 |
| Dense nodes outside segmentation | 0 | 78 |
| Dense edge samples outside segmentation | 67 / 18,235 (0.37%) | 366 / 15,858 (2.31%) |
| Dense length | 2,369 mm | 1,967 mm |
| Extraction time | 5.6 s | 190.3 s |

The original Vedo centerline is visibly smoother in the selected local
region, but smoothing is **not** evidence of better anatomical topology. It
has many more endpoints and leaves the vessel mask more often. Fewer cycles
could mean removal of false loops, removal of true loops, or both: this sample
has no centerline ground truth that resolves that question. Counts are
whole-volume, not local-view counts.

The downstream graph derivation was attempted with exactly the same adaptive
policy for each dense centerline: optimal simplification, radius fraction
0.75, minimum contained-chord fraction 0.95, and 5 smoothing iterations at
alpha 0.5. It produced the current production graph (581 nodes, 649 edges),
but failed on the original Vedo centerline with
`no valid simplification path; adjacent dense samples must remain feasible`.
This is consistent with its smoothed centerline and some adjacent chords
leaving the mask. A historical diagnostic applied the same graph policy to
both centerlines **with lumen containment disabled for both**. These were
never production-quality training graphs; their redundant viewers and arrays
were removed after the repaired final graph comparison became available:

| Diagnostic graph, containment off | Current | Original Vedo |
| --- | ---: | ---: |
| Nodes / edges | 500 / 568 | 567 / 608 |
| Cycle rank β₁ | 73 | 45 |
| Nodes outside segmentation | 0 | 38 |
| Edge samples outside segmentation | 344 / 12,101 | 604 / 11,643 |

Open `centerlines.html` for the **full-volume dense centerlines plus a
translucent decimated segmentation mesh** (20,528 mesh vertices and 41,478
triangles). The decisive comparison is
`dataset_graph_vs_vedo_full.html`, which shows the saved legacy graph beside
the Vedo-derived graph after mask repair, under the same graph policy.
`ct_mip_centerlines.png` overlays both dense centerlines on three local CT
maximum-intensity projections; a projection overlays different depths and
cannot by itself prove a 3D connection. Numeric centerline measurements
remain in `comparison.json`.

**Interpretation from this one case:** the original Vedo output is smoother,
but requires the documented repair to pass the normal graph derivation. The
result has fewer cycles and more endpoints. These counts alone do not tell us
which connections are anatomically correct. Review the final graph overlay
against the segmentation in the HTML views before choosing a pipeline.

## Geometry-controlled follow-up

The initial `centerlines.html` displayed the current **unsmoothed** dense
intermediate against Vedo's **already smoothed** dense result. This made the
geometry comparison unequal. The current graph pipeline already calls
`smooth_polyline` on each degree-2 branch before simplifying, so an additional
full-volume diagnostic applies that exact mask-constrained operation directly
to the current dense node coordinates. Junctions and endpoints stay fixed.

| Full-volume dense geometry | Current raw | Current constrained 5 × 0.5 | Current constrained 4 × 0.8 | Original Vedo |
| --- | ---: | ---: | ---: | ---: |
| Median turn at degree-2 nodes | 45.0° | 6.34° | 5.99° | 6.56° |
| 90th percentile turn | 63.61° | 15.43° | 14.28° | 14.63° |
| Nodes outside segmentation | 0 | 0 | 0 | 78 |
| Edge samples outside segmentation | 67 / 18,235 | 60 / 16,395 | 54 / 16,326 | 366 / 15,858 |
| Cycle rank β₁ | 73 | 73 | 73 | 45 |

See `current_smoothing_full.html` for the **full-volume mesh and all four
selectable geometries**, `current_smoothing_focus.png` for the same local area
side by side, and `current_smoothing.json` for the full metrics. The 5 × 0.5 case is
the smoothing already used by the current compact graph policy. The 4 × 0.8
case uses the original Vedo Laplacian settings but retains the current graph's
connectivity and applies mask constraints.

This supports using a smooth, mask-constrained centerline for geometry. It
does **not** settle the topology question: smoothing coordinates cannot
change β₁. Vedo's 45 cycles result from its different edge-pruning decisions,
which need anatomical review before treating the 28-cycle difference as an
improvement.
