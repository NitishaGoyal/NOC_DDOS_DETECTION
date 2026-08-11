# V5 P0-B0-R1 Identifier Quarantine

## Classification

`LABEL_BEARING_PROVENANCE_IDENTIFIERS_QUARANTINED`

The original B0 HOLD remains preserved. It is classified as a provenance-only
shortcut because `run_id` and `pair_id` directly encode source, victim, and
attacker count, while `run_id`, file naming, and `mode` encode attack/control
run identity.

The learned identifier probe and the direct mode baseline have the same graph
confusion matrix:

`{'tn': 342, 'fp': 46, 'fn': 0, 'tp': 296}`

Every matched attack/control pair has equal run length. Train and validation
contain no shared runs or pairs.

## Allowed learned inputs

- `PRIMARY58 x`
- `edge_index`
- topology-derived raw Boolean `physical_port_mask`

## Quarantined metadata

All identifiers, filenames, mode, seed, epoch/window indices, run length,
serialization position, and every parsed or hashed derivative are forbidden
as learned inputs.

## Next stage

`HOLD_REMEDIATE_B0_R1`
