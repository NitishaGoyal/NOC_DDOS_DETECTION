# V5 P2 Missing-Dependency Location Resolution R2

R1 correctly resolved byte-identical source mirrors, but its edge-artifact
filter admitted only `.pt`, `.pth`, and `.bin`. The frozen topology contract
stores the canonical static edge index as an `int64` NumPy `.npy` file.

R2 admits that exact format, prefers the frozen topology-contract path,
verifies the same frozen SHA-256, and materializes one R2-owned read-only
provenance copy. No dependency is reconstructed or edited.

## Resolved dependencies

- **loader_source**: `/home/zira/research/projects/GNN-2d/artifacts/v5/p2_preflight_missing_dependency_location_resolution_r2/resolved_dependencies/resolved_pair_aligned_loader.py` (`2725ff993f4f03ebee3d5ffb775b1fedd6b131a9c4c89ed8049f45b249c24ac2`)
- **b3_model_source**: `/home/zira/research/projects/GNN-2d/src/models/v5_p2_b3_conv1d_only_count4.py` (`56ee3207d039b60e8e3a898a689cd8e361247c86b7450a3a5e95f423ebe30def`)
- **edge_index_artifact**: `/home/zira/research/projects/GNN-2d/reports/v5/p2_g1a_r2a_canonical_static_topology_contract/V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy` (`f6b8050bc158de509b0ff1c5d1d7cb1ffe32c08b0f2287398270b1b891b57aff`)

## Scientific boundary

- Frozen dependency bytes changed: **false**
- Edge-index format corrected to canonical `.npy`: **true**
- Scientific contract changed: **false**
- Model/checkpoint changed: **false**
- Thresholds or decoders changed: **false**
- Dataset builder imported: **false**
- Dataset constructor invoked: **false**
- Test path resolved: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**

The original dependency-capture runner is now authorized to continue unchanged.
