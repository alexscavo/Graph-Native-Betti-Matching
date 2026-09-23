# Result: cost of 192 object queries on IXI

## Conclusion

Using 192 rather than 120 object queries is inexpensive for the current
64-cubed RelationFormer workload and solves every capacity overflow observed in
this seven-volume IXI sample.

| Metric | 120 queries | 192 queries | Change |
|---|---:|---:|---:|
| Mean full training-step time | 79.69 ms | 81.30 ms | +2.02% |
| 5%-trimmed mean step time | 76.08 ms | 77.25 ms | +1.53% |
| Mean throughput | 104.14 samples/s | 103.03 samples/s | -1.06% |
| Absolute peak allocated GPU memory | 3.023 GiB | 3.124 GiB | +0.101 GiB (+3.33%) |
| Incremental step peak above baseline | 1.234 GiB | 1.333 GiB | +0.099 GiB (+8.05%) |
| Model parameters | 116,762,117 | 116,801,861 | +39,744 (+0.034%) |

The median step time changed by -1.86%, confirming that differences at this
scale are close to per-step scheduling noise. The mean increase was +2.02% in
both 100-step verification runs; the final clean run's trimmed estimate is
+1.53%. A reasonable planning estimate is therefore roughly 2% more compute
time and 0.10 GiB more peak H100 memory at batch size eight, not the 60% query-
count ratio.

The definitive Slurm job was `2187821`. It completed in 56 seconds and consumed
0.01556 allocated GPU-hours. The earlier short runs were useful diagnostics:
they exposed a Python loss-object lifetime artifact that made whichever case
ran second appear to use about 0.8 GiB more memory. The final benchmark releases
each loss graph immediately; its baseline allocations are matched to within
0.0014 GiB, so the reported memory delta is not an ordering artifact.

## Data and capacity result

Only IXI was configured. Seven patient-separated IXI volumes produced 884 real
64-cubed patches after full-volume extraction and exact graph/patch-boundary
clipping:

- 59 patches (6.67%) have more than 120 graph nodes;
- no patch has more than 192 nodes;
- the maximum is 177 nodes and 200 edges;
- the paired timing set contains 729 patches with at most 120 nodes;
- the separate 59-patch overflow set contains every observed 121-to-177-node
  case.

The maximum patch, `sample_000425_0069`, successfully completed a full
192-query training step in 100 ms. This verifies the intended behavior rather
than merely measuring smaller targets that both capacities could represent.

## Protocol

The existing production MRI loader was reused unchanged because the generated
NIfTI image/segmentation plus VTP graph layout already matches its interface.
Both query cases use the same seed, real IXI patches, batch size eight, model,
precision, and loss configuration. Each measurement includes forward
inference, Hungarian matching, graph losses, backward propagation, AdamW, and
the learning-rate scheduler. Augmentation and disabled topology losses are held
identical. The definitive run executes 192 first, then 120, with five warm-up
and 100 measured steps per case.

## Evidence

- The temporary seven-volume selection files were cleared during an earlier
  research-folder cleanup; the selection counts and scope are recorded above.
- [`final/comparison.png`](final/comparison.png): final time, throughput,
  and peak-memory comparison.
- [`final/summary.json`](final/summary.json): complete metrics and overflow
  smoke result.
- [`final/steps.csv`](final/steps.csv): all 200 measured training steps.
- [`final/gpu-telemetry.csv`](final/gpu-telemetry.csv): one-second H100
  utilization, memory, power, clock, and temperature log.
- [`final/system-info.txt`](final/system-info.txt) and
  [`final/job-summary.txt`](final/job-summary.txt): environment and Slurm
  resource record.
- [`final/benchmark.log`](final/benchmark.log) and
  [`final/time-and-errors.log`](final/time-and-errors.log): captured benchmark
  output and process resource log.

## Decision

Retain 192 object queries for the real-data experiment. For this IXI evidence,
the capacity benefit is substantial—every observed patch fits instead of 59
being invalid—while the measured time and absolute-memory costs are small. The
decision remains conditional on the planned full ordinary-volume inventory;
that audit must still confirm that no selected-policy patch exceeds 192.
