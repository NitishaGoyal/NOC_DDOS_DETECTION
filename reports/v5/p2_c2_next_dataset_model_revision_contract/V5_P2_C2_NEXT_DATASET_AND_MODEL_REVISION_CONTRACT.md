# V5 P2-C2 Next Dataset and Model Revision Contract

## Frozen diagnosis

- P2 graph false positives: **2,307**.
- P2 graph false negatives: **35**.
- Pairs with false positives: **60/69**.
- False alarms are **broad** and occur in **persistent episodes**.
- Active attacker-count accuracy is **100%**.
- Weakest localization role: **transit**.
- Weakest graph-recall category: **K1**.
- Weakest strict exact category: **K4**.

The next revision is therefore a **fresh hard-legitimate-traffic dataset first**, followed by a zero-extra-parameter matched-pair graph-margin objective.

## Dataset

Working name: `V6_HARD_LEGITIMATE_MATCHED_CHRONO_GRAPH`.

Generate fresh legal/control runs covering sustained load, bursts, memory-controller hotspots, phase changes, synchronized benign sources, asymmetry, locality skew, and backpressure. Every control must be matched to attacks under the same workload, mapping, seed, phase, placement, and duration.

Do not copy P2 test tensors or windows into the new dataset.

## Model order

1. Retrain the unchanged 43,273-parameter B3-count4 model.
2. Add only a training-time matched ATTACK/CONTROL graph-margin loss.
3. Compare frozen margins 0.25, 0.5, and 1.0 using validation only.
4. Consider asymmetric graph loss or graph/node consistency only after the unchanged baseline and margin ablation.
5. Keep the four-class count head unchanged.

## Primary margin revision

For aligned ATTACK/CONTROL windows, train the graph logits so that:

`attack_logit >= control_logit + margin`

This adds no inference parameters and directly targets the broad control false-positive problem.

## Required split policy

- Split complete run pairs, never windows.
- Keep run ID, pair, seed, mapping, trace lineage, and scenario family disjoint across splits.
- Hold out complete scenario/workload groups.
- Do not inspect the new blind-test tensor contents before checkpoint and thresholds are frozen.

## Required experiment sequence

`D0 dataset generation -> D1 audits -> D2 unchanged B3 -> D3 matched-margin ablation -> D4 selection -> D5 threshold freeze -> D6 one-shot blind test`

## Integrity

- P2 prediction cache accessed: **no**.
- P2 checkpoint loaded: **no**.
- P2 test tensors accessed: **no**.
- P2 inference rerun: **no**.
