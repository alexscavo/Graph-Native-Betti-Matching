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
run passed training and validation. The batch-32 run also passed training and
validation with 10.28 GiB peak allocated GPU memory. Its one-step throughput
is not a reliable full-epoch speed estimate; the purpose was to check memory.

Jobs and dependencies:

- `148568`: batch-8 smoke, **completed**.
- `148871`: batch-32 smoke, **completed**.
- `149029`: 50-epoch pretraining, `afterok:148871`, 2 H100s, 40-hour limit.
- `149114`: 100-epoch IXI fine-tuning, `afterok:149029`, 2 H100s, 8-hour
  limit; initializes from the completed pretraining run's epoch-50 `latest`
  checkpoint and starts a fresh optimizer.

Run outputs and dataset manifests are under
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/`. Slurm logs are
under `/lustre/fswork/projects/rech/vnc/upz25mj/logs/graph-native-betti-matching/`.
W&B is in offline mode for reliable cluster execution; metrics are retained in
the run directory for later synchronization and local plotting.

## Matched node-type ablation

The control repeats **both** stages independently: 50 mixed pretraining epochs
from a fresh initialization, then 100 IXI-only fine-tuning epochs initialized
from that control's own epoch-50 checkpoint. Its only substantive differences
are `model.node_type.enabled: false` and removal of `node_types` from the
optimized losses. The same seed, dataset manifests, batch sizes, query count,
domain objective, optimizer, schedules, and validation/test protocol are kept.
The shared weights are initialized identically under that seed; the enabled
head is initialized without advancing the shared initialization RNG.

For the primary comparison, use the same ordinary edge decoder (degree-aware
penalty **off**) and the same held-out IXI test split. Select checkpoints and
any thresholds using validation only. Report Branch F1, node/edge precision
and recall, edge overconnection/topology errors, and GPU time/memory. Report
matched-node type accuracy for the head model separately; a later degree-aware
decoding comparison tests the *inference* contribution, not the head's training
contribution. One seed is a matched pilot, not a significance claim.
