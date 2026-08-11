# V4 Consolidated Pretraining Diagnostic

- Generated: `2026-07-23T03:53:47.875210+00:00`
- Script version: `1.0.0`
- Dataset: `/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v4_all16_chrono_memmap`
- Final verdict: **V4_REQUIRES_SPLIT_OR_METADATA_REBUILD**
- Training authorized: **False**
- Authorization level: **none**

## Frozen prerequisite stages

- A0.0 pass: `True`
- A0.1 pass: `True`

## Embedded metadata alignment

- Source: `metadata.json["runs"]`
- Run count: `908`
- Mismatches: `1277`
- Hard pass: `False`

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

- None until remediation.

### Blocked tasks

- Start V4-A1 training before required remediation.
- Call the current test split an independent publication holdout.
- Train a victim head when victim metadata is blocked.

### Required remediation

- Repair embedded metadata or metadata arrays so every run aligns exactly.

### Documented limitations

- None identified by this diagnostic.

## Safety boundary

- Model training performed: **False**
- Model inference performed: **False**
- Threshold selection performed: **False**
- Test predictions accessed: **False**
- Full x.npy loaded into RAM: **False**
- Dataset directory modified: **False**

