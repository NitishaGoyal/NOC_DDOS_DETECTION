# V4-A1 Comprehensive Failure Analysis

Script version: `1.0.0`

## Integrity

- Analysis pass: **True**
- Saved predictions reused: **True**
- Model inference performed: **False**
- Test thresholds retuned: **False**
- Dataset modified: **False**

## Frozen operating point

- Graph threshold: `0.6`
- Node threshold: `0.71`

## Validation and test

- Validation graph F1/FPR: `0.7715` / `0.0952`
- Test graph F1/FPR: `0.7571` / `0.0876`
- Validation node F1/exact: `0.4618` / `0.3789`
- Test node F1/exact: `0.4393` / `0.4067`

## Cardinality and ranking

- Frozen test count accuracy: `0.4566`
- Frozen test exact localization: `0.4067`
- Oracle true-count top-k exact localization: `0.5710`
- Oracle gain: `0.1643`

## Spatial structure

- Corner node F1: `0.2980`
- Edge node F1: `0.4816`
- Interior node F1: `0.5125`

## Matched controls

- Matched attack runs: `608`
- Active-core shortcut fraction: `1.0000`
- Mean activity-delta top-k exact run fraction: `0.9655`

## Decision

- Publication-ready dataset: **False**
- A2 capacity lift status: **secondary_only_cardinality_or_set_decoding_is_the_larger_issue**
- Recommended next action: Review count and exact-set reports; test count-conditioned decoding before full A2 training.

The temporal observability audit is heuristic because saved traffic features do not identify malicious injection state separately from legitimate traffic.
