# V5 P2-C0 Frozen Test-Cache Failure Analysis

## Integrity boundary

- Analysis source: frozen B6 prediction cache and pair-window manifest only.
- Model/checkpoint loaded: **no**.
- Original test tensors opened or test inference rerun: **no**.

## Primary diagnosis

- False-positive control windows: **2307**.
- False-negative attack windows: **35**.
- False positives are **98.51%** of graph errors.
- Pairs with at least one FP: **60/69**.
- Pairs with at least one FN: **10/69**.

The dominant failure is normal/control traffic being classified as attack.

## Highest false-positive pairs

| Rank | Pair | FP | Control FPR | Mean control score |
|---:|---|---:|---:|---:|
| 1 | `P2TE-K2-021` | 60 | 0.4138 | 0.3074 |
| 2 | `P2TE-K2-029` | 60 | 0.4138 | 0.3244 |
| 3 | `P2TE-K4-073` | 60 | 0.4138 | 0.3353 |
| 4 | `P2TE-K3-047` | 60 | 0.4082 | 0.2246 |
| 5 | `P2TE-K1-003` | 59 | 0.4126 | 0.3266 |
| 6 | `P2TE-K3-055` | 59 | 0.4126 | 0.3082 |
| 7 | `P2TE-K4-065` | 59 | 0.4126 | 0.2743 |
| 8 | `P2TE-K1-019` | 57 | 0.3986 | 0.2368 |
| 9 | `P2TE-K2-037` | 57 | 0.3931 | 0.2782 |
| 10 | `P2TE-K3-045` | 55 | 0.3901 | 0.2954 |

## K1-K4 attack behavior

| Count | Windows | Graph recall | Count accuracy | All-task exact | Source exact | Transit exact | Victim exact | Path exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K1 | 951 | 0.9800 | 1.0000 | 0.9264 | 0.9685 | 0.9716 | 0.9664 | 0.9506 |
| K2 | 815 | 0.9926 | 1.0000 | 0.6442 | 0.9018 | 0.7681 | 0.9558 | 0.7460 |
| K3 | 904 | 0.9923 | 1.0000 | 0.2777 | 0.7588 | 0.3960 | 0.9204 | 0.5055 |
| K4 | 740 | 0.9959 | 1.0000 | 0.1703 | 0.7054 | 0.2297 | 0.9014 | 0.3595 |

## Rule

Do not alter the P2 checkpoint or thresholds from this analysis. Use it only to design the next dataset/model revision.
