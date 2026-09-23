# Vascular graph research documents

This directory is the canonical home for the representation and evaluation
work that follows the ordered research roadmap.

## Documents

- [`vascular_graph_roadmap.md`](vascular_graph_roadmap.md) — ordered phases and
  the rule that later changes must be justified by measured failure modes.
- [`vascular_graph_representation.md`](vascular_graph_representation.md) — the
  representation rationale: topological anchors plus adaptive geometric
  control points in the near term, curve-valued edges in the longer term.
- [`vascular_graph_metrics.md`](vascular_graph_metrics.md) — metric definitions,
  validation protocol, and recommended implementation structure.
- [`IMPLEMENTATION_TRACKER.md`](IMPLEMENTATION_TRACKER.md) — live execution
  record for the roadmap, including decisions, evidence, and blockers.
- [`loop_failure_mitigation.md`](loop_failure_mitigation.md) — separates
  off-vessel geometry failures from false topological loops and defines the
  validation gate before either correction can enter training data.

Dataset-specific extraction decisions remain in
[`../IXI_DATASET.md`](../IXI_DATASET.md). Repository-wide operational guidance
remains in [`../repo-graph.md`](../repo-graph.md).

## Evidence to show the decisions

Start with the [implementation tracker](IMPLEMENTATION_TRACKER.md) for the
dated decision log. The following evidence folders contain the short
interpretation, numeric data, and inspectable plots or real-data views:

| Decision | Evidence to show | Scope |
| --- | --- | --- |
| Choose one full-volume adaptive graph policy and crop afterward | [14-volume policy comparison](evidence/adaptive_tuning/selected_policy_comparison/README.md), [real 3D graph views](evidence/adaptive_tuning/selected_optimal_0p75/) | Extraction-policy choice, before the Vedo backend change |
| Compare junction-only, adaptive, and dense representations | [30-volume Phase 2 study](evidence/T05/README.md) | Geometry relative to the extracted dense graph; not anatomical ground truth |
| Choose Vedo-original centerlines with documented mask repair | [CT-001 full-volume comparison](evidence/vedo_original_ct001/full_volume/README.md), [production validation](VEDO_PRODUCTION_RUN.md) | One detailed method comparison, then 220-volume production checks |
| Keep graph-positive patches and 256 object queries for IXI | [Production patch QC](evidence/vedo_production/final_validation.json), [IXI 192-versus-256 cost](evidence/ixi_capacity_192_256/README.md) | Current patch selection and measured compute cost |
| Use Branch F1 to detect broken or false connections | [Real IXI edge ablations](evidence/T07/README.md), [controlled metric failures](evidence/T08/README.md) | Metric sensitivity, not model or extractor accuracy |
| Check model integration and subsequent learning | [Real-patch smoke](evidence/T09/README.md), [100-epoch run record](evidence/T09_long/README.md) | The long run is in progress; inspect validation before claiming model quality |

The older [full-dataset inventory](evidence/full_dataset_production/) and
[120-versus-192 query benchmark](evidence/ixi_query_benchmark/RESULTS.md)
remain as historical evidence for why the initial capacity and extraction
decisions changed. The current production outputs live in the dataset folders;
research evidence contains only compact reports and representative viewers.
