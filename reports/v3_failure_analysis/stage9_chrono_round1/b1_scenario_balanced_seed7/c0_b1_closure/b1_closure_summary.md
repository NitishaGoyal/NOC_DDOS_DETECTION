# Chrono-B1 Closure

**Status:** CLOSED AND REJECTED

- Formal verdict: `reject`
- Retained architecture: `Chrono-A1 Conv1D-TemporalGCN`
- Scenario-balanced sampler retained: `False`
- Frozen artifact count: `61`
- Inventory bytes: `16426092`

## Scientific consequence

Chrono-B1 is closed as a rejected sampling-only ablation. Chrono-A1 remains the frozen control. The B1 sampler must not be reused for C1.

## Next experiment

Return to Chrono-A1 unchanged. The next single-change experiment should test an alternative graph readout while freezing the Conv1D encoder, GCN layers, node head, losses, splits, and evaluation protocol.

## Closure files

- `b1_closure_manifest.json`
- `b1_artifact_inventory_sha256.csv`
- `b1_closure_summary.md`
