# Real IXI lumen-containment evidence

This evidence uses the densest 90×90×36 crop of the real
`IXI002-Guys-0828` vessel segmentation. The three representations now derive
from one immutable dense centerline extraction; skeletonization is not repeated
between variants.

The static PNG is a compact decision plot and the HTML is an interactive 3-D
view with the vessel surface and toggleable graph layers. The JSON records the
exact crop, spacing, topology and containment measurements.

Key result: adaptive and dense representations preserve β₀/β₁ = 5/37 and every
straight edge chord is inside the segmented lumen (minimum containment 1.0).
Junction-only preserves topology but its long straight chords are intentionally
an ablation/control: its worst containment is 0.106, demonstrating why degree-2
geometry nodes are necessary.

Regenerate with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/ixi_graph_report_3d.py IXI002-Guys-0828 \
  --output-dir docs/research/evidence/real_ixi_containment --mesh-step 2
```
