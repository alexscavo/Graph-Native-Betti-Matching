# Adaptive graph with all model nodes

These views intentionally omit the stored centerline polylines. They show only
the graph representation consumed by the model: straight graph edges and every
graph node, overlaid on a faint vessel-segmentation surface.

Colors:

- blue: vessel termination, degree 1;
- green: adaptive geometric subdivision, degree 2;
- orange: junction, degree 3 or greater;
- magenta: isolated node, degree 0 (none in either case).

| Subject | Nodes | Edges | Terminations | Degree-2 subdivisions | Junctions | Isolated |
|---|---:|---:|---:|---:|---:|---:|
| IXI122 | 1,630 | 1,720 | 113 | 1,281 | 236 | 0 |
| TopBrain MR 001 | 839 | 881 | 52 | 657 | 130 | 0 |

The interactive HTML files allow node categories and the segmentation surface
to be toggled independently and show node degree and world coordinates on
hover. HTML is generated locally but ignored by Git due to the bundled Plotly
payload; PNG evidence is tracked.
