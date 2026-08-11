# V5 P2 L3-R1 Beam Completion-Feasibility Guard

## Failure classification

A1 R3 completed all **12,528** validation items with zero changed historical
outputs and zero high-precision candidate rejections. A2 then failed before
committing its first item because the narrow beam contained only partial states
that could not be extended to K4.

This is a beam-pruning feasibility bug, not a model, dataset, threshold, A1, or
test-split failure.

## Frozen correction

Before partial states are ordered and truncated to the beam width, a state must
have enough distinct unused higher-ID candidate sources to reach its target K.

- Top-B grid: **unchanged (1, 2, 4)**
- Width grid: **unchanged (4, 8, 16)**
- K grid: **unchanged K1-K4**
- Scores and route unions: **unchanged**
- Deterministic ties: **unchanged**
- Dead-end partial states: **removed before beam admission**

## Verification

- Tests: **17 / 17 PASS**
- Original L3 tests: **12**
- New completion-feasibility tests: **5**
- Adversarial high-source-score case across all nine configurations: **PASS**
- Narrowest B1/width4 determinism: **PASS**

## Preserved state

- A1 exact cache: **12,528 / 12,528**
- A1 R3 certification: **12,528 / 12,528**
- A2 committed cache: **0 / 12,528**
- A2 restart position: **0**
- P2 test directory enumerated: **false**
- P2 test tensors deserialized: **false**
