# V5 P0-A2 Feature and Normalization Contract

## Status

**Contract version:** `V5_P0_A2_D0`

**Primary feature variant:** `PRIMARY58`

## Frozen input variants

### ALL81_COMPATIBILITY

- Feature count: 81
- Uses the package exactly as delivered
- Includes the constant `status_flags` channel and stored standardized mask
  channels
- Compatibility baseline only

### DYNAMIC70

- Feature count: 70
- Indices: 0-29 and 31-70
- Excludes `status_flags` and stored mask channels
- Retains zero-variance P0 stall channels for schema-complete ablation

### PRIMARY58

- Feature count: 58
- Indices: 0-29, 31-55, 62, 64, 65
- Excludes status, twelve zero-variance training channels, and stored mask
  channels
- Receives topology-derived raw Boolean physical-port masks separately
- This is the primary P0 model input

## Normalization

The stored `x` tensor is already transformed. The loader must not apply
`log1p`, `normalization.pt`, or any newly fitted normalization.

## Recoverable mask metadata defect

Features 71-80 were named as physical-port validity masks but were not marked
inside `mask_features`. They therefore followed `log1p_then_standardize`
despite the metadata field `mask_transform='identity'`.

The original tensors remain immutable. The primary loader excludes those
stored mask channels and uses the audited topology-derived Boolean mask:

`/home/zira/research/projects/GNN-2d/reports/v5/p0_a2_1_mask_recoverability_audit/topology_derived_raw_physical_port_mask.pt`

## Test boundary

No validation or test performance was used to freeze this contract.

## Next stage

`V5_P0_A3_LOADER_CONTRACT_AND_SMOKE`
