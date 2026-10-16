# Spline experiment artifacts

| Entry | Purpose |
| --- | --- |
| `synthetic_mri_k8_10k_val2k` | Current prepared dataset: 9,709 train / 1,952 val / 13,071 test patches. Link to SCRATCH. |
| `runs` | Current training/evaluation outputs. Link to SCRATCH. |
| `logs` | Current preparation/training/test logs. |
| `metadata` | Exact sample membership, source configuration, graph hashes, exclusions, restoration checks and current job IDs. |

Historical pilots, four-patch proof experiments and emergency recovery records are archived outside projects:

`/lustre/fsstor/projects/rech/vnc/upz25mj/archives/2026-10-06/spline-prototypes-and-recovery-history.tar.gz`

They are useful for reproducing the initial V0/V1 comparison and debugging history, but are not inputs to the current training. Extract selected archive members into a temporary checkout when revisiting those experiments. Earlier image links in pilot caches may need repointing to the restored SyntheticMRI dataset.

Current code/configuration is committed in the representation branch. Scientific documentation lives in `docs/research/spline_learning/` in this repository.

## New paired scratch experiment

Both 300-epoch arms use the same 10,000 train / 1,952 validation / 13,071 test
patches and junction/termination/boundary targets. Dataset roots and full ID
manifest live in SCRATCH at `spline_learning/scratch_comparison_10k_300ep`.
Protocol: `docs/research/spline_learning/scratch-comparison-300ep.md`.

## Current navigation (2026-10-16)

The 9,709-patch dataset above is the earlier prepared cohort, retained for reproducibility. Main later comparisons use the 10,000-patch membership below it. These are distinct populations, not interchangeable caches.

See `docs/research/storage/retained-experiments.md` for retained checkpoints and `docs/research/storage/cleanup-20261016.md` for the verified smoke/abandoned-run archive. Completed scratch and source-transfer results supersede historical launch-status statements; no scientific result has been deleted merely because it is old.
