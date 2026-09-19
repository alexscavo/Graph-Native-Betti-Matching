# Selected global graph policy comparison

This directory compares the former greedy RDP radius-1.5× policy with the
selected constrained-optimal radius-0.75× policy on the same 14 real volumes and
2,326 nonempty exact 64³ crops.

- `comparison.json` is the complete machine-readable result and relative change.
- `comparison.csv` is the two-row metric table.
- `comparison.png` visualizes graph complexity, patch capacity, centerline error,
  and the measured length-shortening trade-off.
- `strata.csv` and `strata_comparison.png` show patch complexity separately for
  IXI MRA, TopBrain CT, and TopBrain MR.

Regenerate from the repository root with:

```bash
.venv/bin/python scripts/compare_selected_graph_policies.py \
  --baseline-label 'greedy RDP 1.5×' \
  --baseline-sweep docs/research/evidence/adaptive_tuning/results_stage2_14/summary.json \
  --baseline-policy radius_1p5x_c095 \
  --baseline-geometry docs/research/evidence/adaptive_tuning/geometry_radius1p5_14/summary.json \
  --baseline-overflow docs/research/evidence/adaptive_tuning/overflow_14/summary.json \
  --candidate-label 'optimal 0.75×' \
  --candidate-sweep docs/research/evidence/adaptive_tuning/results_optimal_0p75_14/summary.json \
  --candidate-policy optimal_radius_0p75x_c095 \
  --candidate-geometry docs/research/evidence/adaptive_tuning/geometry_optimal_0p75_14/summary.json \
  --candidate-overflow docs/research/evidence/adaptive_tuning/overflow_optimal_0p75_14/summary.json \
  --output-dir docs/research/evidence/adaptive_tuning/selected_policy_comparison
```
