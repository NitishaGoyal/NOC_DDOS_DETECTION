# V3 Validation-Selected Graph-Threshold Transfer

Thresholds were selected using validation data only and then frozen before test evaluation. No model inference was rerun.

## Selection rule

A threshold must first satisfy both validation guardrails when possible:

- worst normal-run FPR ≤ 0.50;
- worst attack-run FNR ≤ 0.50.

Among thresholds satisfying both guardrails, selection maximizes:

```text
0.50 × graph F1
+ 0.20 × (1 − run-macro error rate)
+ 0.15 × (1 − worst normal-run FPR)
+ 0.15 × (1 − worst attack-run FNR)
```

If no threshold satisfies both guardrails, the script uses a validation-only minimax fallback. Test metrics never participate in threshold selection.

## Selected thresholds

| Model | Threshold | Selection mode | Eligible validation thresholds |
|---|---:|---|---:|
| Conv1D-GCN | 0.60 | guardrails_then_max_validation_score | 58 |
| TCN-Attention-GCN | 0.26 | guardrails_then_max_validation_score | 77 |
| TCN-MeanPool-GCN | 0.56 | guardrails_then_max_validation_score | 64 |
| TCN-MaxPool-GCN | 0.41 | guardrails_then_max_validation_score | 76 |

## Test transfer

| Model | Threshold | F1 change | FPR change | FNR change | Worst normal FPR change | Worst attack FNR change |
|---|---:|---:|---:|---:|---:|---:|
| Conv1D-GCN | 0.60 | -0.0129 | -0.0166 | +0.0341 | -0.0033 | +0.0780 |
| TCN-Attention-GCN | 0.26 | -0.0110 | +0.1154 | -0.0524 | +0.0796 | -0.1795 |
| TCN-MeanPool-GCN | 0.56 | -0.0051 | -0.0183 | +0.0205 | -0.0422 | +0.0534 |
| TCN-MaxPool-GCN | 0.41 | +0.0022 | +0.0299 | -0.0233 | +0.0541 | -0.0662 |

## Dominant failure runs

| Model | Run | True class | Operating point | Threshold | Error rate | Mean probability | Median probability |
|---|---|---:|---|---:|---:|---:|---:|
| Conv1D-GCN | `N-3-7-8-12-Pmixed-R18-V3` | 0 | reference_0_50 | 0.50 | 0.9848 | 0.9740 | 0.9988 |
| Conv1D-GCN | `N-3-7-8-12-Pmixed-R18-V3` | 0 | validation_selected | 0.60 | 0.9815 | 0.9740 | 0.9988 |
| Conv1D-GCN | `N-5-10-Pbursty-R51-A-12-S20-V3` | 1 | reference_0_50 | 0.50 | 0.2581 | 0.6699 | 0.7340 |
| Conv1D-GCN | `N-5-10-Pbursty-R51-A-12-S20-V3` | 1 | validation_selected | 0.60 | 0.3362 | 0.6699 | 0.7340 |
| TCN-Attention-GCN | `N-3-7-8-12-Pmixed-R18-V3` | 0 | reference_0_50 | 0.50 | 0.8986 | 0.8204 | 0.9116 |
| TCN-Attention-GCN | `N-3-7-8-12-Pmixed-R18-V3` | 0 | validation_selected | 0.26 | 0.9781 | 0.8204 | 0.9116 |
| TCN-Attention-GCN | `N-5-10-Pbursty-R51-A-12-S20-V3` | 1 | reference_0_50 | 0.50 | 0.4151 | 0.5698 | 0.6184 |
| TCN-Attention-GCN | `N-5-10-Pbursty-R51-A-12-S20-V3` | 1 | validation_selected | 0.26 | 0.2357 | 0.5698 | 0.6184 |
| TCN-MaxPool-GCN | `N-3-7-8-12-Pmixed-R18-V3` | 0 | reference_0_50 | 0.50 | 0.7953 | 0.7507 | 0.8762 |
| TCN-MaxPool-GCN | `N-3-7-8-12-Pmixed-R18-V3` | 0 | validation_selected | 0.41 | 0.8494 | 0.7507 | 0.8762 |
| TCN-MaxPool-GCN | `N-5-10-Pbursty-R51-A-12-S20-V3` | 1 | reference_0_50 | 0.50 | 0.7467 | 0.3031 | 0.1997 |
| TCN-MaxPool-GCN | `N-5-10-Pbursty-R51-A-12-S20-V3` | 1 | validation_selected | 0.41 | 0.6805 | 0.3031 | 0.1997 |
| TCN-MeanPool-GCN | `N-3-7-8-12-Pmixed-R18-V3` | 0 | reference_0_50 | 0.50 | 0.8050 | 0.7420 | 0.8251 |
| TCN-MeanPool-GCN | `N-3-7-8-12-Pmixed-R18-V3` | 0 | validation_selected | 0.56 | 0.7628 | 0.7420 | 0.8251 |
| TCN-MeanPool-GCN | `N-5-10-Pbursty-R51-A-12-S20-V3` | 1 | reference_0_50 | 0.50 | 0.7054 | 0.3386 | 0.2708 |
| TCN-MeanPool-GCN | `N-5-10-Pbursty-R51-A-12-S20-V3` | 1 | validation_selected | 0.56 | 0.7589 | 0.3386 | 0.2708 |

## Interpretation

- Validation-selected calibration improves the FPR/F1 operating trade-off for 1 of four models without using test labels for selection.
- At least one catastrophic run-level failure remains for 4 of four models after threshold transfer.
- Threshold transfer measures the calibration limit; it does not prove the cause of the remaining errors.
- If the hard normal run remains above threshold while the weak attack remains below it, the score ordering—not merely the threshold—is the central problem.
