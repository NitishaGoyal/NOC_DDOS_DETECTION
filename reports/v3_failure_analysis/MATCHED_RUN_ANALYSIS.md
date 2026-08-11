# V3 Matched Dominant-Run Analysis

This stage compares run metadata, model probabilities, temporal behaviour, threshold crossings, error persistence and cross-model agreement. It does not load the original 24-feature tensor and does not rerun inference.

Only validation and test probabilities are available because Stage 3 did not export training predictions. Training-run feature analogues can be examined in Stage 7 using the original dataset.

## Exact and partial controls

- Hard normal target: `N-3-7-8-12-Pmixed-R18-V3`.
- Exact normal test control by class/profile/core count: `N-1-6-9-14-Pmixed-R17-V3`.
- Hard attack target: `N-5-10-Pbursty-R51-A-12-S20-V3`.
- Exact attack test controls by attacker count, attacker router and strength: `N-2-13-Pstream-R50-A-12-S20-V3`, `N-1-6-9-14-Pmixed-R52-A-12-S20-V3`.
- All remaining same-class validation and test runs are partial controls ranked by factor-match score.

## Hard normal versus primary control

### Conv1D-GCN
- Target `N-3-7-8-12-Pmixed-R18-V3`: median probability 0.9975, error rate 97.94%, longest error streak 3028.
- Control `N-1-6-9-14-Pmixed-R17-V3`: median probability 0.1346, error rate 12.45%, longest error streak 16.

### TCN-Attention-GCN
- Target `N-3-7-8-12-Pmixed-R18-V3`: median probability 0.9116, error rate 89.86%, longest error streak 447.
- Control `N-1-6-9-14-Pmixed-R17-V3`: median probability 0.3151, error rate 25.87%, longest error streak 25.

### TCN-MeanPool-GCN
- Target `N-3-7-8-12-Pmixed-R18-V3`: median probability 0.8251, error rate 80.50%, longest error streak 88.
- Control `N-1-6-9-14-Pmixed-R17-V3`: median probability 0.1211, error rate 14.97%, longest error streak 33.

### TCN-MaxPool-GCN
- Target `N-3-7-8-12-Pmixed-R18-V3`: median probability 0.8762, error rate 79.53%, longest error streak 159.
- Control `N-1-6-9-14-Pmixed-R17-V3`: median probability 0.0958, error rate 11.33%, longest error streak 18.

## Hard attack versus primary controls

### Conv1D-GCN
- Target `N-5-10-Pbursty-R51-A-12-S20-V3`: median probability 0.5795, error rate 41.51%, longest error streak 39.
- Control `N-2-13-Pstream-R50-A-12-S20-V3`: median probability 0.8571, error rate 18.62%, longest error streak 30.
- Control `N-1-6-9-14-Pmixed-R52-A-12-S20-V3`: median probability 0.7042, error rate 27.39%, longest error streak 20.

### TCN-Attention-GCN
- Target `N-5-10-Pbursty-R51-A-12-S20-V3`: median probability 0.6184, error rate 41.51%, longest error streak 74.
- Control `N-2-13-Pstream-R50-A-12-S20-V3`: median probability 0.9756, error rate 4.92%, longest error streak 15.
- Control `N-1-6-9-14-Pmixed-R52-A-12-S20-V3`: median probability 0.9891, error rate 3.49%, longest error streak 23.

### TCN-MeanPool-GCN
- Target `N-5-10-Pbursty-R51-A-12-S20-V3`: median probability 0.2708, error rate 70.54%, longest error streak 100.
- Control `N-2-13-Pstream-R50-A-12-S20-V3`: median probability 0.9781, error rate 8.23%, longest error streak 16.
- Control `N-1-6-9-14-Pmixed-R52-A-12-S20-V3`: median probability 0.9165, error rate 6.92%, longest error streak 35.

### TCN-MaxPool-GCN
- Target `N-5-10-Pbursty-R51-A-12-S20-V3`: median probability 0.1997, error rate 74.67%, longest error streak 136.
- Control `N-2-13-Pstream-R50-A-12-S20-V3`: median probability 0.9905, error rate 7.29%, longest error streak 18.
- Control `N-1-6-9-14-Pmixed-R52-A-12-S20-V3`: median probability 0.9740, error rate 5.34%, longest error streak 22.

## Direct Stage 6 findings

- The hard normal run has a mean four-model error rate of **87.0%** at threshold 0.50, versus **16.2%** for the exact mixed/four-core test control.
- Its middle-80% mean error rate is **89.9%**, so the failure is not confined to run startup or shutdown.
- The hard attack has a mean four-model error rate of **57.1%** and a middle-80% mean error rate of **58.1%**.
- All four models are simultaneously wrong on **63.9%** of hard-normal windows and **15.2%** of hard-attack windows at threshold 0.50.
- Model predictions disagree on **35.5%** of hard-normal windows and **75.7%** of hard-attack windows.

## What Stage 6 can establish

- Whether the target run is uniquely difficult relative to same-class controls.
- Whether the failure persists through steady-state epochs.
- Whether models fail at the same temporal locations.
- Whether profile, core count, exact placement, attacker count, attacker location or strength are sufficient metadata explanations.
- Which matched controls should be carried into Stage 7.

## What remains unproven until Stage 7

- Whether directional IFD saturation causes the overlap.
- Whether benign congestion and weak attacks are inseparable in the 24-feature space.
- Whether normalization or clipping destroys magnitude information.
- Which routers and directions produce the conflicting signals.
- Whether pooling suppresses a weak temporal feature that is present in the raw input.

## Recommended Stage 7 comparisons

### Hard normal controls
- `N-1-6-9-14-Pmixed-R17-V3`
- `N-2-6-10-14-Pmixed-R13-V3`
- `N-2-13-Pstream-R15-V3`

### Hard attack controls
- `N-2-13-Pstream-R50-A-12-S20-V3`
- `N-1-6-9-14-Pmixed-R52-A-12-S20-V3`
- `N-4-15-Pbursty-R41-A-11-S20-V3`

These controls are ranked using metadata match quality plus probability-distribution behaviour. Stage 7 should still inspect all exact controls, not only the top-ranked entries.
