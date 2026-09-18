# Real IXI 64³ patch-budget audit

The committed IXI002 baseline was generated through the same world-coordinate
export and exact centreline clipping code used by dataset generation. It uses a
64³ patch with 5-voxel padding, hence an effective 54³ source crop.

Current fixed-2-voxel adaptive baseline over 175 non-empty patches:

- mean edges: 21.7;
- p95 edges: 63;
- p99 edges: 78.6;
- maximum edges: 139;
- patches above the preferred 70-edge ceiling: 5;
- patches above 120 edges: 1;
- maximum nodes: 114; no patch exceeded 120 nodes.

The typical patch meets the preferred range, but the high-complexity tail does
not. Slurm array job `2141634` was submitted to repeat the audit for all three
representations on Guys, HH, and high-resolution IOP subjects. Its outputs will
replace this preliminary single-subject summary when complete.

