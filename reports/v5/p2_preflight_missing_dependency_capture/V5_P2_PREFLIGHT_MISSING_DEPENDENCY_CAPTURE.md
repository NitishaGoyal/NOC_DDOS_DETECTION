# V5 P2 Missing Dependency Capture

## Status

- Status: **DEPENDENCY_BUNDLE_READY**
- Loader implementation resolved: **true**
- B3 reference-model source resolved: **true**
- Frozen edge-index artifact resolved: **true**
- Canonical edge format (`int64` NumPy `.npy`): **PASS**
- Complete directed 4×4 mesh contract: **PASS**
- Frozen physical-port mask resolved: **true**
- Route-library source resolved: **true**
- Task-D checkpoint strict load: **PASS**
- Synthetic Task-D forward pass: **PASS**
- Route count: **240**
- Exact MILP executed: **false**

## Frozen runtime interface

- Loader class: `V5P2PairAlignedPrimary58Dataset(root: 'str | Path', split: 'str', pair_manifest: 'str | Path', *, window: 'int' = 32, stride: 'int' = 8) -> 'None'`
- B3 class: `P2B3Conv1DOnlyCount4() -> 'None'`
- Task-D class: `P2TaskDGraphConvCount4(reference_b3: 'nn.Module', base_edge_index: 'torch.Tensor') -> 'None'`
- Checkpoint state key: `model_state_dict`
- Model parameters: **59785**
- State tensors: **49**
- State elements including buffers: **59881**

## Review bundle

- Path: `/home/zira/research/projects/GNN-2d/artifacts/v5/p2_preflight_missing_dependency_capture/V5_P2_PREFLIGHT_MISSING_DEPENDENCY_SOURCE_BUNDLE.zip`
- SHA-256: `8a3615fc73bb3ee056bd1d41acae85b2fdc5044403a39fbb96db60a2242e045f`

## Boundary

- Dataset module imported: **true**
- Dataset constructor invoked: **false**
- Validation dataset opened: **false**
- Test path resolved: **false**
- Test directory existence checked: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Concrete evaluator source frozen: **false**

The review bundle now contains every missing source dependency required to
construct the immutable Raw/A0/A1 one-shot evaluator.
