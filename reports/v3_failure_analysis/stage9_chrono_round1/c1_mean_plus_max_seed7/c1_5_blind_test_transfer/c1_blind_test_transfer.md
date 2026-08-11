# C1 Blind-Test Threshold Transfer

**Verdict:** `TEST_DOES_NOT_CORROBORATE_C1`

Thresholds were selected on validation and transferred unchanged.
No threshold sweep or threshold selection was performed on test.

## Frozen thresholds

| Model | Graph threshold | Node threshold |
|---|---:|---:|
| A1 | 0.410 | 0.765 |
| C1 | 0.140 | 0.620 |

## Key metrics

| Metric | A1 | C1 | C1−A1 |
|---|---:|---:|---:|
| graph_f1 | 0.871904 | 0.833630 | -0.038273 |
| graph_recall | 0.936055 | 0.943517 | +0.007462 |
| graph_fpr | 0.246280 | 0.373469 | +0.127189 |
| normal_run_macro_fpr | 0.246280 | 0.373469 | +0.127189 |
| worst_normal_run_fpr | 0.990282 | 0.996356 | +0.006073 |
| attack_run_macro_recall | 0.936055 | 0.943517 | +0.007462 |
| strength20_macro_recall | 0.874886 | 0.922867 | +0.047981 |
| worst_attack_run_recall | 0.807470 | 0.802612 | -0.004859 |
| attack_only_node_f1 | 0.805525 | 0.803716 | -0.001809 |
| exact_localization | 0.406794 | 0.385753 | -0.021040 |
| top1_hit_rate | 0.760618 | 0.772721 | +0.012104 |
| top3_hit_rate | 0.976704 | 0.976747 | +0.000043 |

## Primary corroboration

- FAIL — `graph_f1_nonnegative_delta`
- FAIL — `normal_run_macro_fpr_improvement_at_least_0.020`
- FAIL — `worst_normal_run_fpr_improvement_at_least_0.050`

## Mandatory safeguards

- FAIL — `graph_f1_drop_no_worse_than_0.010`
- FAIL — `overall_graph_fpr_not_worse_than_plus_0.020`
- FAIL — `normal_run_macro_fpr_not_worse_than_plus_0.020`
- PASS — `worst_normal_run_fpr_not_worse_than_plus_0.050`
- PASS — `overall_graph_recall_drop_no_worse_than_0.020`
- PASS — `attack_run_macro_recall_drop_no_worse_than_0.020`
- PASS — `strength20_recall_drop_no_worse_than_0.030`
- PASS — `worst_attack_run_recall_drop_no_worse_than_0.050`
- PASS — `attack_only_node_f1_drop_no_worse_than_0.020`
- PASS — `exact_localization_drop_no_worse_than_0.030`
- PASS — `top1_drop_no_worse_than_0.020`
- PASS — `top3_drop_no_worse_than_0.010`

## Decision

Do not spend further V3 training budget on C1. Chrono-A1 remains the final V3 model.
