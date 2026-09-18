# Exact versus inherited crop-policy metrics

These reports compare `SourceGraph.crop()` (exact polyline/voxel-cell-face
intersection) with `SourceGraph.crop_inherited()` (last existing centerline
sample inside the patch) on the two existing real-data densest 64³ windows.

Bounds use complete voxel cells: `[start - 0.5, start + 63.5]`. Boundary gaps
are therefore reported in voxel coordinates. Geometry is additionally reported
in physical millimetres after applying each image affine. Topology and token
counts use the actual cropped graphs.

| Subject | Boundary contacts matched | Mean last-sample face gap | ACD | HD95 | Exact / inherited nodes | Exact / inherited edges | β₁ delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| IXI122-new | 13 / 13 | 0.421 vox | 0.0034 mm | 0.0233 mm | 103 / 103 | 108 / 108 | 0 |
| TopBrain-MR-001 | 22 / 22 | 0.413 vox | 0.0060 mm | 0.0397 mm | 77 / 77 | 78 / 78 | 0 |

At these cell-face bounds, the inherited policy preserves topology and token
counts in both cases. Its endpoints are displaced inward by roughly 0.4 voxel
on average. Both densest patches still exceed the preferred 70-edge operating
range, although neither exceeds the 120-node hard query capacity.

The JSON files retain all settings, source paths, boundary matching, Curve F1,
ACD, HD95, topology, and budget flags. This is a two-case diagnostic rather
than a dataset-level conclusion.

Visual evidence:

- `IXI122-new.png` and interactive `IXI122-new.html`;
- `TopBrain-MR-001.png` and interactive `TopBrain-MR-001.html`.

Red points are exact interpolated cell-face intersections; orange points are
their matched inherited last-inside centerline samples. The translucent grey
surface is the real vessel segmentation.
