# IXI source-domain comparison, representation trained in both stages

Date: 2026-10-15. User authorized three data strategies, each with straight edges and B-splines during BOTH pretraining and fine-tuning. This differs from the preceding plant/leaf→SyntheticMRI experiments, which pretrained graph detection only and introduced spline loss during fine-tuning. Those existing runs are preserved.

## Six arms

| Array index | Arm | First stage | Second stage |
|---|---|---|---|
|0|ixi_straight|IXI, straight|IXI, straight|
|1|ixi_spline|IXI, B-spline|IXI, B-spline|
|2|plants_straight|Plants + IXI, straight|IXI, straight|
|3|plants_spline|Plants + IXI, B-spline|IXI, B-spline|
|4|synthetic_straight|SyntheticMRI + IXI, straight|IXI, straight|
|5|synthetic_spline|SyntheticMRI + IXI, B-spline|IXI, B-spline|

All first stages start from random weights; the base YAML supplies configuration, not pretrained parameters. Each second stage loads its own validation-selected first-stage checkpoint, INCLUDING learned spline parameters, with fresh optimizer/scheduler. The IXI-only second stage is continued training on the same population, not a domain transfer.

## Controlled training recipe

- First stage50epochs,50,000 replacement draws/epoch, batch32 (1563updates/epoch).
- Same10,000 compatible IXI training patches across all six arms, selected by seeded hash ranking; subject-disjoint original splits preserved.
- Mixed sources25,000 plants or25,000 SyntheticMRI patches. Equal expected source/target draws via target weight2.5. IXI-only uses50,000 IXI draws. This matches updates, not IXI exposure or exact GPU compute. Edge-geometry loss averages matched positives, so sample balance does not imply equal per-domain geometry gradient magnitude.
- Second stage up to100epochs on the same IXI training population; existing patience30/minimum30 policy and monitoring every5epochs retained.
- AdamW LR7e-5, decay1e-4, one-epoch warmup and existing polynomial schedule. Graph losses unchanged; spline residual-control weight1.4739989612791453. This weight is reused from SyntheticMRI, not newly calibrated to IXI.
- Mixed pretraining uses domain adaptation weight1.0, image/instance adversarial losses and consistency. IXI-only and all second stages disable domain adaptation.
- All arms use256object queries, following existing IXI configurations; original SyntheticMRI study used120. Node-type prediction remains disabled. All other architecture choices follow the prior controlled setup.
- Cubic B-splines have8total controls (6internal). Retain the tested detached gradient route: residual geometry learns from relation/object tokens and local image features without sending geometry gradients into the shared detector or predicted endpoints. Graph losses still train the detector. Thus “B-splines throughout” means geometry supervision in both stages, not fully coupled optimization.
- Straight controls retain a frozen zero-residual spline head for common geometry inference. Their curves are straight chords.
- Same approved quarter-turn/noise MRI augmentation; no zoom. Plants retain quarter-turns/flips and five-slice projection into padded64³ volumes.
- BN training statistics during updates, saved running statistics during evaluation. Offline W&B and fixed validation projections retained.

## Data and geometric supervision

New derived vascular caches are under `/datasets`, not the experiment output directory:

- IXI: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/IXI/transfer_splines_k8_20261015`.
- Auxiliary SyntheticMRI: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/syntheticMRI/transfer_source_k8_25k_20261015`.
- Plants reuse the existing25k contracted source and retained branch-path cache from the previous experiment.

IXI uses the existing `optimal_radius_0p75x_c095_vedo_original` sources/patches. Trace full stored paths before fitting, preserve junctions/endpoints and boundary anchors using the established closest-inside-centerline-sample policy, with exact intersection fallback for sparse clipped components. Straight and spline arms use identical nodes, edges, membership and full-path evaluation references. Intermediate degree-2 samples remain in curve guides, not object targets.

Unsupported parallel/closed branches and capacity overflow exclude the entire patch for EVERY arm; reasons are recorded. Original IXI active pool has13,742train/2,961val/2,384test patches, before this common compatibility check. All compatible validation/test patches are retained; final counts are only established by preparation. Failure to supply10k IXI or25k SyntheticMRI training patches blocks training. Source SyntheticMRI is limited to original training subjects with available full graphs; missing subjects are reported. Original source validation/test subjects are not imported as training data.

IXI graph paths are segmentation-derived uncertain references, not perfect anatomical truth. MRI metrics use original full-volume affines, never identity patch headers.

### Plant spline targets: previously missing integration

`PlantsDataset` optionally reads cached controls/curves/guides, verifies node/edge correspondence, swaps PNG axes, embeds curves at z=.5, applies the SAME sampled rotations/flips as the graph, and maps them through final padding. Controls are never clipped. Its returned geometry dictionary is supervised by the existing residual-control loss.

Plants have no physical millimetres. Define virtual isotropic voxel spacing as the median geometric-mean spacing across selected IXI TRAINING subjects. This scales the projected source geometry to the target loss's physical range; it is an explicit rendering assumption, not measured plant anatomy. Maintain a metric matrix through augmentation/padding so distances and canonical edge orientation remain consistent. The exact value is recorded in `monitoring.json` and each run's provenance. The source cache stays in original PNG units; only loading transforms it.

## Evaluation and execution gates

- Fixed hash-selected128validation and512training monitors, identical across arms; length-bin cutoffs computed on IXI training guides only.
- Checkpoint selection: validation common branch-F1, then native edge-F1. Full compatible validation evaluated for final and selected checkpoints, including common geometry and topology suite.
- After fine-tuning, evaluate the validation-selected checkpoint once on the fixed compatible IXI test set. Test outcomes never select checkpoints or adjust the setup.
- GPU smoke checks run three updates per stage/arm. They assert finite gradients, active domain classifiers in mixed runs, spline gradients in spline runs, and nonzero geometry loss gradients separately for source and target domains. Shared initialization hashes must agree across all six before production training starts.
- CPU smoke1125311 failed because the IXI source path required `graphs/adaptive/`; corrected without changing data. Replacement1125734 passed, accepting2train/2val/2test targets;3parallel-branch exclusions among24attempted candidates. No performance conclusion from this extraction check.
- Targeted coordinate tests:3passed (plant embedding and existing MRI spline rotation). All12stage configurations validated. GPU checks remain pending until full target preparation completes.

## Submitted jobs

Record: `experiments/ixi_transfer/submission-20261015.json`.

| Stage | Job |
|---|---|
|IXI target preparation|1126571|
|SyntheticMRI source preparation|1126575|
|GPU verification, six arms|1126576|
|Pretraining to25|1126577|
|Pretraining to50|1126578|
|Fine-tuning to50|1126579|
|Fine-tuning to100|1126580|
|Validation-selected held-out tests|1126581|

GPU arrays use indices0–5 with at most3concurrent tasks per array. Each task uses oneH100. `afterok` gates preparation/verification; `aftercorr` links the corresponding arm's later stages. Failed prerequisites cancel dependent jobs instead of launching an incomplete experiment.

Output: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/ixi_transfer_20261015/runs/`.
Config: `configs/ixi_transfer.yaml`. Runner: `scripts/train_ixi_transfer.py`; it reuses the existing loss, optimizer, matcher, augmentation, monitoring and metrics infrastructure. `source_mri` is an additional loader name for the existing vascular loader; `synthetic_mri` remains the internal target key but points to IXI here.

## Next research decision

First inspect full-population exclusions and GPU checks; then compare learning curves and final held-out metrics. This experiment tests auxiliary biological versus synthetic vascular sources under a fixed update budget. It does not isolate a single causal “biological alignment” mechanism or establish source-independent superiority. No new sweep or loss terms are authorized by this setup.

Submission check: both full preparation jobs are running; all GPU stages and tests are registered with the intended dependencies. The two training smoke targets were visually inspected in `previews/ixi-target-smoke.png`: retained paths align with the masks, but source-derived graphs do not explain every foreground fragment. This is a limitation of the existing extracted references, not evidence of anatomical completeness.
