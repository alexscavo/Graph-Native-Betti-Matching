# Plants → IXI domain-adaptation run, 2026-09-24

The vessel target is **IXI**, not TopBrain. The 50-epoch pretraining stage
mixes 25,900 Plants patches (source, domain 0) with 8,000 IXI vessel patches
(target, domain 1). Weighted replacement sampling makes the expected domain
draws 50:50 over 33,900 draws per epoch. Both domains contribute graph losses;
the image/instance adversarial losses and consistency term also contribute to
the optimized total. The 100-epoch fine-tuning stage uses IXI only and no
domain discriminators. The degree-type head stays active in both stages.

The [smoke plot](smoke_evidence.png) and `smoke_summary.json` show the input
composition and measured one-H100 smoke-run throughput/memory. Draw counts in
the plot are *expectations*, not observed per-epoch counts. The batch-8 smoke
run passed training and validation; batch-32 is a separate memory/throughput
check before production becomes eligible.

Jobs and dependencies:

- `148568`: batch-8 smoke, **completed**.
- `148871`: batch-32 smoke, pending at last check.
- `149029`: 50-epoch pretraining, `afterok:148871`, 2 H100s, 40-hour limit.
- `149114`: 100-epoch IXI fine-tuning, `afterok:149029`, 2 H100s, 8-hour
  limit; initializes from the completed pretraining run's epoch-50 `latest`
  checkpoint and starts a fresh optimizer.

Run outputs and dataset manifests are under
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/`. Slurm logs are
under `/lustre/fswork/projects/rech/vnc/upz25mj/logs/graph-native-betti-matching/`.
W&B is in offline mode for reliable cluster execution; metrics are retained in
the run directory for later synchronization and local plotting.
