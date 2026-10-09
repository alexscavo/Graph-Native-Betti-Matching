# Full validation comparison

All 1,952 validation patches, identical protocols; no test evaluation or BN recalibration. Matched-positive geometry is separate from full decoded graph metrics. Centerline guides are uncertain. Single seed; validation supports model selection, not an unbiased final test claim.

| model | node_f1 | edge_f1 | branch_f1 | curve_f1 | chord_curve_f1 | matched_guide_mm | endpoint_neutral_mm |
|---|---|---|---|---|---|---|---|
| frozen_residual | 0.91041 | 0.86363 | 0.27625 | 0.68441 | 0.64702 | 0.89484 | 0.46094 |
| detached_head | 0.90910 | 0.87198 | 0.29183 | 0.67955 | 0.64762 | 0.89051 | 0.46107 |
| endpoints_detached | 0.90889 | 0.87096 | 0.29236 | 0.68296 | 0.64766 | 0.88814 | 0.45990 |
