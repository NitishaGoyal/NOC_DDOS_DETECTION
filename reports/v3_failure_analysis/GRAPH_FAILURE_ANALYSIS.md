# V3 Graph Failure Analysis

This report analyses graph-level detection at the fixed threshold of 0.50.

## Overall test errors

| Model | FP | FN | Total errors | FPR | FNR |
|---|---:|---:|---:|---:|---:|
| Conv1D-GCN | 4023 | 3511 | 7534 | 0.2036 | 0.1523 |
| TCN-Attention-GCN | 4877 | 2078 | 6955 | 0.2468 | 0.0901 |
| TCN-MeanPool-GCN | 3574 | 3179 | 6753 | 0.1809 | 0.1379 |
| TCN-MaxPool-GCN | 3315 | 3258 | 6573 | 0.1678 | 0.1413 |

## Direct findings

- Across all four models: **15789 false-positive windows** and **12026 false-negative windows**.
- Dominant error type: **false positives**.
- Largest FP problem: **TCN-Attention-GCN** (4877 test FPs).
- Largest FN problem: **Conv1D-GCN** (3511 test FNs).

## Worst normal test runs

### Conv1D-GCN
- `N-3-7-8-12-Pmixed-R18-V3`: 3225/3293 error windows (97.94%); profile=mixed; active_cores=3-7-8-12; attackers=<none>; strength=<none>; longest streak=3028.
- `N-1-6-9-14-Pmixed-R17-V3`: 410/3293 error windows (12.45%); profile=mixed; active_cores=1-6-9-14; attackers=<none>; strength=<none>; longest streak=16.
- `N-2-13-Pstream-R15-V3`: 115/3293 error windows (3.49%); profile=stream; active_cores=2-13; attackers=<none>; strength=<none>; longest streak=15.
- Normal test runs with zero FPs: 0; above 50% FP: 1; above 80% FP: 1.

### TCN-Attention-GCN
- `N-3-7-8-12-Pmixed-R18-V3`: 2959/3293 error windows (89.86%); profile=mixed; active_cores=3-7-8-12; attackers=<none>; strength=<none>; longest streak=447.
- `N-2-13-Pstream-R15-V3`: 875/3293 error windows (26.57%); profile=stream; active_cores=2-13; attackers=<none>; strength=<none>; longest streak=23.
- `N-1-6-9-14-Pmixed-R17-V3`: 852/3293 error windows (25.87%); profile=mixed; active_cores=1-6-9-14; attackers=<none>; strength=<none>; longest streak=25.
- Normal test runs with zero FPs: 0; above 50% FP: 1; above 80% FP: 1.

### TCN-MeanPool-GCN
- `N-3-7-8-12-Pmixed-R18-V3`: 2651/3293 error windows (80.50%); profile=mixed; active_cores=3-7-8-12; attackers=<none>; strength=<none>; longest streak=88.
- `N-1-6-9-14-Pmixed-R17-V3`: 493/3293 error windows (14.97%); profile=mixed; active_cores=1-6-9-14; attackers=<none>; strength=<none>; longest streak=33.
- `N-2-13-Pstream-R15-V3`: 131/3293 error windows (3.98%); profile=stream; active_cores=2-13; attackers=<none>; strength=<none>; longest streak=12.
- Normal test runs with zero FPs: 0; above 50% FP: 1; above 80% FP: 1.

### TCN-MaxPool-GCN
- `N-3-7-8-12-Pmixed-R18-V3`: 2619/3293 error windows (79.53%); profile=mixed; active_cores=3-7-8-12; attackers=<none>; strength=<none>; longest streak=159.
- `N-1-6-9-14-Pmixed-R17-V3`: 373/3293 error windows (11.33%); profile=mixed; active_cores=1-6-9-14; attackers=<none>; strength=<none>; longest streak=18.
- `N-2-13-Pstream-R15-V3`: 108/3293 error windows (3.28%); profile=stream; active_cores=2-13; attackers=<none>; strength=<none>; longest streak=12.
- Normal test runs with zero FPs: 0; above 50% FP: 1; above 80% FP: 0.

## Worst attack test runs

### Conv1D-GCN
- `N-5-10-Pbursty-R51-A-12-S20-V3`: 1367/3293 error windows (41.51%); profile=bursty; active_cores=5-10; attackers=12; strength=20; longest streak=39.
- `N-1-6-9-14-Pmixed-R52-A-12-S20-V3`: 902/3293 error windows (27.39%); profile=mixed; active_cores=1-6-9-14; attackers=12; strength=20; longest streak=20.
- `N-2-13-Pstream-R50-A-12-S20-V3`: 613/3293 error windows (18.62%); profile=stream; active_cores=2-13; attackers=12; strength=20; longest streak=30.
- Attack test runs with zero FNs: 0; above 50% FN: 0.

### TCN-Attention-GCN
- `N-5-10-Pbursty-R51-A-12-S20-V3`: 1367/3293 error windows (41.51%); profile=bursty; active_cores=5-10; attackers=12; strength=20; longest streak=74.
- `N-2-13-Pstream-R50-A-12-S20-V3`: 162/3293 error windows (4.92%); profile=stream; active_cores=2-13; attackers=12; strength=20; longest streak=15.
- `N-5-10-Pbursty-R51-A-1-7-S63-V3`: 120/3293 error windows (3.64%); profile=bursty; active_cores=5-10; attackers=1-7; strength=63; longest streak=19.
- Attack test runs with zero FNs: 0; above 50% FN: 0.

### TCN-MeanPool-GCN
- `N-5-10-Pbursty-R51-A-12-S20-V3`: 2323/3293 error windows (70.54%); profile=bursty; active_cores=5-10; attackers=12; strength=20; longest streak=100.
- `N-2-13-Pstream-R50-A-12-S20-V3`: 271/3293 error windows (8.23%); profile=stream; active_cores=2-13; attackers=12; strength=20; longest streak=16.
- `N-1-6-9-14-Pmixed-R52-A-12-S20-V3`: 228/3293 error windows (6.92%); profile=mixed; active_cores=1-6-9-14; attackers=12; strength=20; longest streak=35.
- Attack test runs with zero FNs: 0; above 50% FN: 1.

### TCN-MaxPool-GCN
- `N-5-10-Pbursty-R51-A-12-S20-V3`: 2459/3293 error windows (74.67%); profile=bursty; active_cores=5-10; attackers=12; strength=20; longest streak=136.
- `N-2-13-Pstream-R50-A-12-S20-V3`: 240/3293 error windows (7.29%); profile=stream; active_cores=2-13; attackers=12; strength=20; longest streak=18.
- `N-1-6-9-14-Pmixed-R52-A-12-S20-V3`: 176/3293 error windows (5.34%); profile=mixed; active_cores=1-6-9-14; attackers=12; strength=20; longest streak=22.
- Attack test runs with zero FNs: 0; above 50% FN: 1.

## Error concentration

- **Conv1D-GCN:** FP: worst two runs contribute 90.4%; FN: worst two runs contribute 64.6%.
- **TCN-Attention-GCN:** FP: worst two runs contribute 78.6%; FN: worst two runs contribute 73.6%.
- **TCN-MeanPool-GCN:** FP: worst two runs contribute 88.0%; FN: worst two runs contribute 81.6%.
- **TCN-MaxPool-GCN:** FP: worst two runs contribute 90.3%; FN: worst two runs contribute 82.8%.

## Confidence and persistence

- Borderline errors: 17.9% of test graph-error windows.
- High-confidence errors: 53.3% of test graph-error windows.
- Longest test error streak: 3028 consecutive windows in `N-3-7-8-12-Pmixed-R18-V3` for Conv1D-GCN.
- The large number of high-confidence errors suggests that threshold adjustment alone is unlikely to solve the problem.

## Cross-model agreement

- All four models correct: 31039 test windows.
- All four models wrong: 2887 test windows.
- Conv1D correct while all TCN models are wrong: 698 windows.
- All TCN models correct while Conv1D is wrong: 1700 windows.
- Attention uniquely fails: 1402 windows.
- MaxPool uniquely succeeds: 554 windows.

## Current graph-level diagnosis

The mean validation-to-test FPR increase is 0.1376. The mean FNR increase is 0.0826. The current collapse is best described as **primarily a benign-distribution false-positive shift, with a secondary attack-generalization failure**.

This does not prove whether calibration, feature ambiguity, temporal behaviour, or representation shift is causal. The next controlled step is a threshold-transfer analysis.
