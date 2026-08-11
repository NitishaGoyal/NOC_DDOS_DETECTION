# V4-A1 Comprehensive Failure Analysis

Script version: `1.0.0`

## Integrity

- Analysis pass: **True**
- Saved predictions reused: **True**
- Model inference performed: **False**
- Test thresholds retuned: **False**
- Dataset modified: **False**

## Frozen operating point

- Graph threshold: `0.4599999999999999`
- Node threshold: `0.7799999999999999`

## Validation and test

- Validation graph F1/FPR: `0.9055` / `0.0978`
- Test graph F1/FPR: `0.8932` / `0.1136`
- Validation node F1/exact: `0.7752` / `0.6603`
- Test node F1/exact: `0.7521` / `0.6645`

## Cardinality and ranking

- Frozen test count accuracy: `0.6883`
- Frozen test exact localization: `0.6645`
- Oracle true-count top-k exact localization: `0.8069`
- Oracle gain: `0.1424`

## Spatial structure

- Corner node F1: `0.5863`
- Edge node F1: `0.8072`
- Interior node F1: `0.8095`

## Matched controls

- Matched attack runs: `608`
- Active-core shortcut fraction: `1.0000`
- Mean activity-delta top-k exact run fraction: `0.9655`

## Decision

- Publication-ready dataset: **False**
- A2 capacity lift status: **useful_but_must_be_paired_with_cardinality_diagnostics**
- Recommended next action: Proceed with A2 only as a controlled diagnostic, while retaining count-conditioned analyses.

The temporal observability audit is heuristic because saved traffic features do not identify malicious injection state separately from legitimate traffic.
