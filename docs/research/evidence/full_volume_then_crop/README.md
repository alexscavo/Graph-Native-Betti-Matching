# Full-volume extraction followed by inherited patch cropping

This evidence corrects the operation order used in `new_datasets_baseline`:

1. skeletonize the complete vessel segmentation once with the legacy (pre-Vedo)
   backend;
2. derive and save the junction-only, adaptive, and dense full-volume graphs;
3. select the densest 64³ voxel window;
4. crop the saved adaptive graph in voxel coordinates with
   `SourceGraph.crop_inherited()`; and
5. transform the cropped geometry back to world coordinates for display.

The inherited crop uses the last already-existing centerline sample inside the
patch for a crossing edge. It does **not** interpolate a new node on the exact
patch face. This is the requested experimental policy. By contrast,
`generate_synthetic_mri_dataset.py` currently writes patches with
`SourceGraph.crop()`, which performs exact geometric clipping and creates a
box-face intersection node; it only computes the inherited result for its
comparison audit.

## Initial real-data cases

| Dataset / subject | Full adaptive nodes | Full adaptive edges | Full β₀ | Full β₁ | Densest 64³ crop nodes | Crop edges |
|---|---:|---:|---:|---:|---:|---:|
| Improved IXI / IXI122 | 1,630 | 1,720 | 14 | 104 | 99 | 104 |
| TopBrain MRA / 001 | 839 | 881 | 4 | 46 | 73 | 74 |

The JSON files beside each visualization record the exact source paths, voxel
bounds, topology, and isolated-node count. Self-contained interactive HTML is
generated locally but ignored by Git because each file bundles several MB of
Plotly JavaScript. Static PNG and JSON evidence are tracked.

These are deliberately density-maximizing patches, not dataset-wide
percentiles. TopBrain is near the preferred 60–70-edge range; IXI is below the
120-token hard capacity but above that preference. No parameter was tuned per
dataset in this comparison.
