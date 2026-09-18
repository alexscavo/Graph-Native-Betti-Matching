# Real IXI coordinate and smoothness verification

This check uses the real IXI002-Guys-0828 segmentation crop from the interactive
QC report.

- Graph topology and centreline initialization come from the segmentation.
- The raw dense centreline is voxel-centred by definition and is kept only as an
  evaluation reference.
- Constrained Laplacian smoothing reduced median consecutive-sample turn angle
  from 45.0° to 6.6° on this crop.
- 28.4% of adaptive graph nodes in this crop have sub-voxel coordinates.
- Every graph edge written to the training VTP is one straight two-point line;
  the dense/smoothed polylines are used for simplification and exact patch
  clipping, not written as zig-zag training edges.

The whole-volume IXI002 audit found 46.5% sub-voxel adaptive source nodes. The
difference from the crop percentage is expected because vessel content differs.

