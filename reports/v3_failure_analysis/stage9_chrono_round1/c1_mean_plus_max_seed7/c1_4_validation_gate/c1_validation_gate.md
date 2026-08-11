# C1 Validation-Only Advancement Gate

**Verdict:** `PROMISING_ADVANCE_TO_BLIND_TEST`

No test inference or test threshold selection was performed.

## Frozen validation thresholds

| Model | Graph threshold | Node threshold |
|---|---:|---:|
| A1 | 0.410 | 0.765 |
| C1 | 0.140 | 0.620 |

## Key metrics

| Metric | A1 | C1 | C1−A1 |
|---|---:|---:|---:|
| graph_f1 | 0.958716 | 0.974935 | +0.016219 |
| graph_recall | 0.977494 | 0.987583 | +0.010089 |
| graph_fpr | 0.138779 | 0.086319 | -0.052460 |
| normal_run_macro_fpr | 0.138779 | 0.086319 | -0.052460 |
| worst_normal_run_fpr | 0.432736 | 0.190708 | -0.242029 |
| attack_run_macro_recall | 0.977494 | 0.987583 | +0.010089 |
| strength20_macro_recall | 0.959611 | 0.968519 | +0.008908 |
| worst_attack_run_recall | 0.942606 | 0.929548 | -0.013058 |
| attack_only_node_f1 | 0.905798 | 0.910246 | +0.004448 |
| exact_localization | 0.704963 | 0.718764 | +0.013800 |
| top1_hit_rate | 0.952256 | 0.962446 | +0.010190 |
| top3_hit_rate | 0.995107 | 0.995107 | +0.000000 |

## Primary improvements

- PASS — `graph_f1_improvement_at_least_0.010`
- FAIL — `strength20_macro_recall_improvement_at_least_0.030`
- FAIL — `worst_attack_run_recall_improvement_at_least_0.050`

## Mandatory safeguards

- PASS — `normal_run_macro_fpr_not_worse_than_plus_0.020`
- PASS — `worst_normal_run_fpr_not_worse_than_plus_0.050`
- PASS — `overall_graph_recall_drop_no_worse_than_0.020`
- PASS — `attack_run_macro_recall_drop_no_worse_than_0.020`
- PASS — `attack_only_node_f1_drop_no_worse_than_0.020`
- PASS — `exact_localization_drop_no_worse_than_0.030`
- PASS — `top1_drop_no_worse_than_0.020`
- PASS — `top3_drop_no_worse_than_0.010`

## Decision

Proceed to frozen-threshold blind-test transfer.
