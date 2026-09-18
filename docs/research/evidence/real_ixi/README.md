# Real IXI three-representation evidence

These reports use real IXI vessel segmentations, not handcrafted masks.

| Subject | Site | Native in-plane spacing | Crop | Purpose |
|---|---|---:|---:|---|
| IXI002-Guys-0828 | Guys | 0.469 mm | 90×90×36 | standard-resolution example |
| IXI609-HH-2600 | HH | 0.469 mm | 90×90×36 | second site, standard resolution |
| IXI290-IOP-0874 | IOP | 0.264 mm | 160×160×36 | high-resolution example with comparable physical in-plane field of view |

Each local `graph3d_<subject>.html` is a self-contained interactive 3D viewer
with the vessel surface and toggleable junction-only, adaptive, and dense graph
layers. HTML files bundle Plotly and are intentionally ignored by Git; the
compact summaries and overview plot are tracked.

Regenerate them from the repository root:

```bash
python scripts/ixi_graph_report_3d.py IXI002-Guys-0828 \
  --output-dir docs/research/evidence/real_ixi --mesh-step 2
python scripts/ixi_graph_report_3d.py IXI609-HH-2600 \
  --output-dir docs/research/evidence/real_ixi --mesh-step 2
python scripts/ixi_graph_report_3d.py IXI290-IOP-0874 \
  --output-dir docs/research/evidence/real_ixi \
  --half 80 --half-z 18 --mesh-step 2
```

The crop extents, physical spacing, extraction parameters, and representation
counts are recorded in each `*.summary.json`. These are crop-level QC examples,
not yet the dataset-level rate-distortion study required by T06.

