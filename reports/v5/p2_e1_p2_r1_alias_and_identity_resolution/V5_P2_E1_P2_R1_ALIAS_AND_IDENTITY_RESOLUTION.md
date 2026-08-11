# V5 P2 E1-P2-R1 Alias and Identity Resolution

## Append-only resolution

The original E1-P2 report remains unchanged. Its `exporter_ready=false` state
was caused by canonical export names that differ from the frozen loader names,
not by absent targets.

- Canonical `y_graph` is exported from loader key `y_attack`.
- Canonical `y_path` is exported from loader key `y_attack_path`.
- All other label keys are used without renaming.
- Stable item identity is reconstructed from the frozen dataset `_index`,
  deterministic validation sampler order, and pair-manifest `pair_key` join.
- The sampler-emitted item sequence must equal `0..12527` exactly.

## State

- Exporter ready: **true**
- Checkpoint tensors loaded by this correction: **false**
- Validation tensors opened: **false**
- Validation predictions exported: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
