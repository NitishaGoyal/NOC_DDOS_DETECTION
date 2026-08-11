# V5 P2-C1 Scenario and Root-Cause Interpretation

## Frozen conclusion

**The dominant P2 failure is broad false-positive activation on legitimate/control traffic.**

- False-positive windows: **2307**.
- False-negative windows: **35**.
- FP share of graph errors: **98.51%**.
- Pairs with at least one FP: **60 / 69**.
- Top-10 FP-pair concentration: **25.40%**.
- FP temporal form: **persistent_false_alarm_episodes**.

This means the problem is not limited to one attacker count or one isolated bad run. Legitimate traffic broadly overlaps the learned attack signature.

## What is already working

- Active attacker-count accuracy: **1.0000**.
- The count head should remain frozen as the baseline.
- Attack recall remains very high; the detector is not primarily missing attacks.

## Localization bottleneck

- Weakest role by mean supported per-node F1: **transit**.
- Weakest K category for graph recall: **K1**.
- Weakest K category for strict attack exactness: **K4**.

Localization should be improved after the legal/control false-positive problem, not before it.

## Required next dataset revision

1. Collect hard legitimate traffic: high-load, bursty, hotspot, phase-change, and memory-intensive controls.
2. Match each hard control with attacks using the same legal workload, mapping, traffic phase, and placement.
3. Preserve run-level chronological splitting and pair alignment.
4. Oversample legal windows corresponding to long false-positive episodes without copying any P2 test tensor into training.
5. Retain K1–K4 count labels and all four role labels.

## Required next model revision

Keep the 43,273-parameter P2 model as the frozen reference. In a new revision, test a graph decision mechanism that learns explicit normal-traffic evidence or enforces consistency between graph detection and localized source/path evidence. Do not optimize the count head.

## Integrity boundary

- C1 opened only C0 reports and CSV/JSON analyses.
- B6 prediction cache opened: **no**.
- Model/checkpoint loaded: **no**.
- Original test tensors accessed: **no**.
- Test inference rerun: **no**.
