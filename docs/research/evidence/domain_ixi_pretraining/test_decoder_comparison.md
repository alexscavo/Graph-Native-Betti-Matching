# IXI test: plain decoder versus soft degree penalty 0.5

Both evaluations use the same frozen checkpoint, all 2,384 active IXI test
patches, and identical thresholds. Only `node_type_degree_penalty` changes.

| Metric | Plain | Penalty 0.5 | Change |
|---|---:|---:|---:|
| Node F1 | 0.5103 | 0.5103 | 0 |
| Node mAP | 0.0984 | 0.0984 | 0 |
| Node mAR | 0.2262 | 0.2262 | 0 |
| Edge F1 | 0.3894 | **0.4847** | **+0.0953** |
| Edge mAP | **0.1011** | 0.0937 | −0.0075 |
| Edge mAR | **0.2965** | 0.2613 | −0.0353 |
| Edge precision | 0.2620 | **0.3786** | **+0.1167** |
| Edge recall | 0.7582 | 0.6732 | −0.0850 |
| Branch F1 | 0.00887 | **0.01008** | **+0.00121** |
| Edge-count absolute error | 95.55 | **39.39** | **−56.16** |
| β₁ absolute error | 82.05 | **28.92** | **−53.13** |
| Predicted edges | 145.87 | **89.60** | **−56.26** |
| Predicted β₁ | 82.82 | **29.70** | **−53.13** |

The penalty leaves node predictions unchanged, as expected: it only changes
edge decoding using the predicted node types. It substantially reduces false
edges and improves edge F1/precision and topology error, at the cost of some
edge recall. This is strong evidence that the soft decoder is useful for the
over-connection problem, but it is still an inference-time result; it does not
yet prove that the node-type head improves training compared with the matched
no-head control.

Evidence:

- [Comparison plot](test_decoder_comparison.png)
- [Machine-readable comparison](test_decoder_comparison.json)
- [Plain dense-patch HTML](../../../../../../../../experiments/gnbm/ixi_q256_types_300ep_20260924/evaluation/test_best_branch_f1/interactive_sample_000104_0119/sample_000104_0119__sample_001685.html)
- [Penalty-0.5 dense-patch HTML](../../../../../../../../experiments/gnbm/ixi_q256_types_300ep_20260924/evaluation/test_degree_penalty/penalty_0.5/interactive_sample_000104_0119/sample_000104_0119__sample_001685.html)

The complete evaluation directories contain `summary.json`,
`per-patch-metrics.csv`, `predictions.json`, and static PNG samples.
