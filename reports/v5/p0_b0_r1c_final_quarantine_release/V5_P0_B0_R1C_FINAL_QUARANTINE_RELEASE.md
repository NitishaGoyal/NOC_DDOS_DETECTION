# V5 P0-B0-R1C Final Quarantine Release

## Status

`PASS`

## Classification

`PROVENANCE_ONLY_SHORTCUT_QUARANTINED`

The original B0 and B0-R1 HOLD artifacts are preserved. B0-R1B demonstrated
that run length, run window count, within-run position, and all numeric
provenance combined do not cross the frozen validation shortcut thresholds.

## Allowed learned inputs

- `PRIMARY58 x`
- `edge_index`
- topology-derived raw Boolean `physical_port_mask`

## Forbidden learned inputs

All provenance identifiers, filenames, mode, seed, run/window timing metadata,
serialization position, and every parsed, hashed, embedded, or derived version
of those fields.

## Next stage

`V5_P0_B1_STATIC_FINAL_EPOCH_MLP`
