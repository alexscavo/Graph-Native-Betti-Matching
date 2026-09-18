# Vascular graph roadmap implementation tracker

## Purpose

This is the living execution record for
[`vascular_graph_roadmap.md`](vascular_graph_roadmap.md). Update it whenever a
task changes state, a decision is made, or validation produces evidence.

Status vocabulary: `not started`, `in progress`, `blocked`, `complete`.

A task is not complete until it has automated checks and inspectable evidence.
For spatial or quantitative work, save both machine-readable data and a plot or
screenshot under `artifacts/<task-id>/` so results can be independently reviewed.

## Current position

- Current phase: Phase 1 — finalize the adaptive ground-truth representation.
- Next task: preserve and name the immutable dense reference before simplification.
- Training changes: prohibited until the representation study is complete.

## Ordered task list

| ID | Task | Roadmap phase | Status |
|---|---|---:|---|
| T01 | Preserve pure degree-2 rings and add topology regression tests | 1 | complete |
| T02 | Enforce the IXI638/IXI661 annotation overrides and record provenance | 1 | complete |
| T03 | Emit junction-only, adaptive, and dense comparison representations | 1 | complete |
| T04 | Preserve and name an immutable dense reference before simplification | 1 | not started |
| T05 | Implement representation geometry, morphology, and complexity metrics | 2 | not started |
| T06 | Run the rate-distortion study across sites and acquisition resolutions | 2 | not started |
| T07 | Implement degree-2 contraction, physical anchor matching, and Branch F1 | 3 | not started |
| T08 | Validate metrics on the prescribed handcrafted failure cases | 4 | not started |
| T09 | Generate/smoke-test IXI patches and retrain with optimization unchanged | 5 | not started |

## Decision log

### D01 — Representation of an anchorless ring

- Date: 2026-09-18
- Status: accepted for implementation
- Decision: represent each pure degree-2 ring with three deterministic
  degree-2 nodes and three ordered arcs.
- Reason: one node plus a self-loop and two nodes plus parallel edges are not
  reliably representable by the downstream simple adjacency matrix. Three
  nodes form a simple cycle, preserve beta-0/beta-1, and remain compatible with
  adaptive RDP subdivision.

### D02 — Three explicit source representations

- Date: 2026-09-18
- Status: accepted for implementation
- Decision: emit `graphs/junction_only`, `graphs/adaptive`, and `graphs/dense`
  as separate graph products. Patch generation defaults to `adaptive` when this
  layout is detected.
- Reason: representation quality must be measured before selecting a training
  target. Keeping the products separate makes comparisons reproducible and
  prevents dense-reference geometry from being confused with model ground truth.

### D03 — Dataset-agnostic extraction constraint

- Date: 2026-09-18
- Status: active design constraint
- Decision: extractor policies should use physical units or local normalized
  quantities. Dataset-specific constants are provisional and must be tested
  across acquisition resolutions, sites, vessel calibres, and geometries.
- Consequence: the current fixed voxel-based RDP and spur defaults are baselines,
  not accepted final policies. T06 must explicitly test their transferability.

## Evidence log

### E01 — Pre-fix ring failure

A handcrafted square-ring mask produced 72 skeleton voxels and `ring_count=1`,
but exported zero nodes and zero edges, so `VesselGraph.betti()` returned
`(0, 0)` instead of `(1, 1)`. This establishes the T01 regression target.

### E02 — Ring regression result

The focused test suite verifies both the basic three-node ring and its adaptive
RDP subdivision. Both return `(beta_0, beta_1) = (1, 1)`; 2 tests passed. Data
and visualization: [`evidence/T01/`](evidence/T01/).

### E03 — Annotation override provenance

The 397-subject planning run selects `nii(1).gz` for IXI638 and IXI661 and
records both rows as `preferred_duplicate`; all focused extraction tests pass.
Data and visualization: [`evidence/T02/`](evidence/T02/).

### E04 — Three-representation source build

An end-to-end synthetic NIfTI source build emitted nine graph files: the three
Voreen files for each of `junction_only`, `adaptive`, and `dense`. The example
retained identical topology `(beta_0, beta_1) = (1, 0)` while using 2, 4, and 39
nodes respectively. Data and visualization: [`evidence/T03/`](evidence/T03/).

## Change log

- 2026-09-18: Created the tracker and organized the research documents under
  `docs/research/`.
- 2026-09-18: Completed T01 by exporting pure rings as simple three-edge cycles
  and adding topology regression coverage.
- 2026-09-18: Completed T02 by enforcing the two documented annotation
  overrides and recording the selected source and variant in both provenance
  outputs.
- 2026-09-18: Completed T03 by emitting three explicit representation products,
  retaining adaptive as the automatically selected patch-training target, and
  validating the layout end to end.
