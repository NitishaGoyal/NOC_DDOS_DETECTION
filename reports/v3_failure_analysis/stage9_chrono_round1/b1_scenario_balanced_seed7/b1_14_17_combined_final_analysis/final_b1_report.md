# Chrono-B1 Final Post-Training Report

## Formal verdict: **REJECT**

Only 3/10 frozen validation safeguards passed. The primary normal-run FPR objectives failed or the blind test materially contradicted the intended benefit.

## Final retained architecture

Chrono-A1 remains the retained Conv1D-TemporalGCN baseline. The scenario-balanced sampler is excluded.

## Next single-change experiment

Return to Chrono-A1 unchanged. The next single-change experiment should test an alternative graph readout while freezing the Conv1D encoder, GCN layers, node head, losses, splits, and evaluation protocol.

## Protocol integrity

- Prerequisite and alignment checks passed: **True**
- Thresholds were selected on validation only and transferred unchanged to test.
- Localization denominators contain attack-positive samples only.
- Paired bootstrap: 2000 repetitions, seed 7.

### Frozen thresholds

| Model | Graph threshold | Node threshold |
| --- | --- | --- |
| A1 | 0.410 | 0.765 |
| B1 | 0.080 | 0.545 |

## Overall A1 versus B1

| Split | Metric | A1 | B1 | B1−A1 |
| --- | --- | --- | --- | --- |
| val | graph_recall | 0.977494 | 0.940547 | -0.036947 |
| val | graph_f1 | 0.958716 | 0.919876 | -0.038840 |
| val | graph_fpr | 0.138779 | 0.234892 | +0.096113 |
| val | attack_only_node_f1 | 0.905798 | 0.863588 | -0.042210 |
| val | attack_only_exact_localization | 0.704963 | 0.568917 | -0.136046 |
| val | top1_hit_rate | 0.952256 | 0.945507 | -0.006748 |
| val | top3_hit_rate | 0.995107 | 0.996997 | +0.001890 |
| val | mean_reciprocal_rank | 0.973435 | 0.970475 | -0.002960 |
| test | graph_recall | 0.936055 | 0.952887 | +0.016832 |
| test | graph_f1 | 0.871904 | 0.842249 | -0.029655 |
| test | graph_fpr | 0.246280 | 0.361474 | +0.115194 |
| test | attack_only_node_f1 | 0.805525 | 0.792767 | -0.012758 |
| test | attack_only_exact_localization | 0.406794 | 0.429526 | +0.022732 |
| test | top1_hit_rate | 0.760618 | 0.713635 | -0.046983 |
| test | top3_hit_rate | 0.976704 | 0.956227 | -0.020476 |
| test | mean_reciprocal_rank | 0.868366 | 0.837153 | -0.031213 |

## Frozen safeguard table

| Criterion | Validation delta | Limit | Formal result | Test delta | Test corroboration |
| --- | --- | --- | --- | --- | --- |
| normal_run_macro_fpr_improvement_at_least_0.02 | +0.096113 | delta <= -0.02 | FAIL | +0.115194 | contradicts |
| worst_normal_run_fpr_improvement_at_least_0.05 | +0.124810 | delta <= -0.05 | FAIL | +0.006377 | contradicts |
| overall_graph_recall_drop_no_worse_than_0.02 | -0.036947 | delta >= -0.02 | FAIL | +0.016832 | supports |
| attack_run_macro_recall_drop_no_worse_than_0.02 | -0.036947 | delta >= -0.02 | FAIL | +0.016832 | supports |
| strength20_recall_drop_no_worse_than_0.03 | +0.015791 | delta >= -0.03 | PASS | +0.022978 | supports |
| worst_attack_run_recall_drop_no_worse_than_0.05 | -0.388703 | delta >= -0.05 | FAIL | -0.072275 | contradicts |
| attack_only_node_f1_drop_no_worse_than_0.02 | -0.042210 | delta >= -0.02 | FAIL | -0.012758 | supports |
| attack_only_exact_localization_drop_no_worse_than_0.03 | -0.136046 | delta >= -0.03 | FAIL | +0.022732 | supports |
| top1_drop_no_worse_than_0.02 | -0.006748 | delta >= -0.02 | PASS | -0.046983 | contradicts |
| top3_drop_no_worse_than_0.01 | +0.001890 | delta >= -0.01 | PASS | -0.020476 | contradicts |

## Run-level graph summary

| Split | Model | Normal macro FPR | Worst normal FPR | Attack macro recall | Worst attack recall | S20 recall | Hard-normal FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | A1 | 0.138779 | 0.432736 | 0.977494 | 0.942606 | 0.959611 | NA |
| val | B1 | 0.234892 | 0.557546 | 0.940547 | 0.553902 | 0.975402 | NA |
| test | A1 | 0.246280 | 0.990282 | 0.936055 | 0.807470 | 0.874886 | 0.990282 |
| test | B1 | 0.361474 | 0.996660 | 0.952887 | 0.735196 | 0.897864 | 0.996660 |

## Paired bootstrap confidence intervals

| Split | Metric | Point delta | 95% CI | Unit | CI excludes zero |
| --- | --- | --- | --- | --- | --- |
| val | graph_f1 | -0.038840 | [-0.041019, -0.036767] | sample | True |
| val | graph_recall | -0.036947 | [-0.040002, -0.033766] | sample | True |
| val | graph_fpr | +0.096113 | [+0.089460, +0.102918] | sample | True |
| val | attack_only_node_f1 | -0.042210 | [-0.044098, -0.040387] | attack-positive sample | True |
| val | attack_only_exact_localization | -0.136046 | [-0.141917, -0.130513] | attack-positive sample | True |
| val | top1_hit_rate | -0.006748 | [-0.009347, -0.003913] | attack-positive sample | True |
| val | top3_hit_rate | +0.001890 | [+0.001046, +0.002767] | attack-positive sample | True |
| val | mean_reciprocal_rank | -0.002960 | [-0.004394, -0.001457] | attack-positive sample | True |
| val | normal_run_macro_fpr | +0.096113 | [+0.006073, +0.187899] | run | True |
| val | attack_run_macro_recall | -0.036947 | [-0.135034, +0.018187] | run | False |
| test | graph_f1 | -0.029655 | [-0.032121, -0.027144] | sample | True |
| test | graph_recall | +0.016832 | [+0.013410, +0.020456] | sample | True |
| test | graph_fpr | +0.115194 | [+0.110524, +0.120059] | sample | True |
| test | attack_only_node_f1 | -0.012758 | [-0.015146, -0.010249] | attack-positive sample | True |
| test | attack_only_exact_localization | +0.022732 | [+0.016658, +0.028763] | attack-positive sample | True |
| test | top1_hit_rate | -0.046983 | [-0.051234, -0.042645] | attack-positive sample | True |
| test | top3_hit_rate | -0.020476 | [-0.023124, -0.017960] | attack-positive sample | True |
| test | mean_reciprocal_rank | -0.031213 | [-0.033661, -0.028796] | attack-positive sample | True |
| test | normal_run_macro_fpr | +0.115194 | [+0.007086, +0.291376] | run | True |
| test | attack_run_macro_recall | +0.016832 | [-0.018698, +0.050934] | run | False |

## Matched disagreement summary

| Split | Metric | Both correct | Both wrong | B1 fixes | B1 breaks | Net fixes |
| --- | --- | --- | --- | --- | --- | --- |
| val | graph_correctness | 36929 | 1471 | 1024 | 3385 | -2361 |
| val | exact_localization | 14973 | 6856 | 1888 | 5920 | -4032 |
| val | top1_localization | 27300 | 693 | 722 | 922 | -200 |
| val | top3_localization | 29438 | 35 | 110 | 54 | 56 |
| test | graph_correctness | 33325 | 5084 | 1256 | 3144 | -1888 |
| test | exact_localization | 7249 | 11022 | 2652 | 2128 | 524 |
| test | top1_localization | 15731 | 4799 | 719 | 1802 | -1083 |
| test | top3_localization | 21831 | 326 | 211 | 683 | -472 |

## Recommendation

Return to Chrono-A1 unchanged. The next single-change experiment should test an alternative graph readout while freezing the Conv1D encoder, GCN layers, node head, losses, splits, and evaluation protocol.

The next experiment must preserve the corrected chronological split and change only the named intervention.
