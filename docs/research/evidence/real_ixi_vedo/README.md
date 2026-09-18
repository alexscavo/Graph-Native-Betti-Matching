# Rejected Vedo centerline experiment

Status: **rejected; never the default extractor**.

This experiment implemented the requested ordering
`segmentation → skeletonization → Vedo dense refinement → graph extraction`.
The historical Vedo script's NetworkX graph was treated correctly as a dense
centerline intermediate, not as the compact training graph.

Real IXI evaluation was materially worse than the existing baseline. On the
IXI002 90×90×36 crop, the Vedo-derived adaptive representation had 363 nodes,
411 edges and β₁=49. The previous baseline had 244 nodes, 276 edges and β₁=37
on the same crop. The result remained visually tangled despite constrained
sub-voxel smoothing and lumen-contained adaptive chords.

The reason is structural: Vedo does not replace the initial voxel skeleton in
the supplied method. It smooths and filters a dense 26-neighbour sample graph;
surface geodesics do not reliably distinguish true vascular cycles from local
digital cycles or segmentation fusions. Smoother coordinates therefore do not
imply better topology.

The PNG/JSON files are retained as negative evidence. Bundled interactive HTML
files remain local and ignored because they are several megabytes each.
