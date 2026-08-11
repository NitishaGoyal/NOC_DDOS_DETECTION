# V4-A1 Comprehensive Failure Analysis

Script version: `1.0.0`

## Integrity

- Analysis pass: **True**
- Saved predictions reused: **True**
- Model inference performed: **False**
- Test thresholds retuned: **False**
- Dataset modified: **False**

## Frozen operating point

- Graph threshold: `0.5399999999999999`
- Node threshold: `0.8799999999999999`

## Validation and test

- Validation graph F1/FPR: `0.8728` / `0.0963`
- Test graph F1/FPR: `0.8571` / `0.0887`
- Validation node F1/exact: `0.6721` / `0.5393`
- Test node F1/exact: `0.6657` / `0.5827`

## Cardinality and ranking

- Frozen test count accuracy: `0.6156`
- Frozen test exact localization: `0.5827`
- Oracle true-count top-k exact localization: `0.7479`
- Oracle gain: `0.1653`

## Spatial structure

- Corner node F1: `0.4957`
- Edge node F1: `0.7290`
- Interior node F1: `0.7136`

## Matched controls

- Matched attack runs: `608`
- Active-core shortcut fraction: `1.0000`
- Mean activity-delta top-k exact run fraction: `0.9655`

## Decision

- Publication-ready dataset: **False**
- A2 capacity lift status: **secondary_only_cardinality_or_set_decoding_is_the_larger_issue**
- Recommended next action: Review count and exact-set reports; test count-conditioned decoding before full A2 training.

The temporal observability audit is heuristic because saved traffic features do not identify malicious injection state separately from legitimate traffic.
