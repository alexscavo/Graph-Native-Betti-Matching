# Fresh baseline on new IXI and TopBrain data

This folder starts a new real-data evaluation using the pre-Vedo (`legacy`)
centerline/graph pipeline, which performed better than the rejected Vedo
experiment.

## Inputs

- `IXI122-new`: improved binary IXI segmentation, native spacing
  0.469×0.469×0.800 mm.
- `TopBrain-MR-001`: union of all nonzero TopBrain anatomical vessel labels,
  native spacing 0.297×0.297×0.600 mm. The original multiclass labels remain
  available for subsequent class-aware topology checks.

The interactive HTML reports show the vessel surface and toggleable
junction-only, adaptive and dense representations. `*_overview3d.png` provides
a static view of the adaptive graph inside the segmentation; the other PNG
compares representation complexity and chord containment.

## Initial observations

- Both adaptive graphs have zero off-lumen straight edge chords and preserve the
  dense graph's β₀/β₁.
- The improved IXI crop is markedly less cycle-dense than the previous IXI
  failure example: β₁=10 in the comparable physical crop.
- The TopBrain crop also has β₁=10, but visually contains more tortuous and
  looped anatomy. Its multiclass labels should be used to distinguish expected
  Circle-of-Willis connections from extraction artifacts.
- In the densest 64³ patches, adaptive has 99 edges for IXI122 and 79 for
  TopBrain-MR-001. Both exceed the preferred 60–70 edge range, while remaining
  below the 120-token capacity. These are deliberately worst-density patches,
  not dataset percentiles.

See `patch64_summary.json` for exact windows and counts.
