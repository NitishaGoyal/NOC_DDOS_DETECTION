# V4 Consolidated Pretraining Diagnostic

- Generated: `2026-07-23T03:59:43.779997+00:00`
- Script version: `1.0.1`
- Dataset: `/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v4_all16_chrono_memmap`
- Final verdict: **V4_READY_ONLY_FOR_DIAGNOSTIC_BASELINE**
- Training authorized: **True**
- Authorization level: **diagnostic_only**

## Frozen prerequisite stages

- A0.0 pass: `True`
- A0.1 pass: `True`

## Embedded metadata alignment

- Source: `metadata.json["runs"]`
- Run count: `908`
- Mismatches: `0`
- Hard pass: `True`

## Family leakage

- `background_family_cross_split_count`: `0`
- `pair_cross_split_count`: `0`
- `matched_normal_cross_split_count`: `0`
- `near_duplicate_signature_cross_split_count`: `0`
- Hard pass: `True`

## Matched controls and active-core shortcut

- Attack runs: `608`
- Hard control failures: `0`
- Shortcut cases: `608`
- Shortcut rate: `1.000000`

## Feature-semantic audit

- Verdict: **FEATURE_AMBIGUITY_PRESENT_NO_LABEL_SEPARATION**
- Maximum graph-label rate difference: `0.100817`
- Maximum split rate difference: `0.088020`

## Attack-kind and victim metadata

- Attack-kind implementation aliases: `0`
- Victim metadata blocks attacker-only A1: `False`
- Victim metadata blocks victim head: `False`

## Authorization

- Development test status: `development blind test`
- Publication holdout status: `not established; independent family pool required`

### Allowed tasks

- Create the frozen V4-A1 Conv1D-TemporalGCN memmap training script.
- Run loader, one-batch, and short training preflights.
- Train seed 7 within the stated authorization level.

### Blocked tasks

- Treat diagnostic metrics as formal evidence.
- Use the development test for architecture or threshold tuning.
- Call the current test split an independent publication holdout.
- Train a victim head when victim metadata is blocked.

### Required remediation

- None before the authorized V4-A1 scope.

### Documented limitations

- Potential active-core or feature shortcut prevents formal scientific claims.

## Safety boundary

- Model training performed: **False**
- Model inference performed: **False**
- Threshold selection performed: **False**
- Test predictions accessed: **False**
- Full x.npy loaded into RAM: **False**
- Dataset directory modified: **False**

