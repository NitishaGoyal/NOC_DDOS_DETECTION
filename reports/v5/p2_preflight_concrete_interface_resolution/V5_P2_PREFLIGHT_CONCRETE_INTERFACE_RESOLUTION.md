# V5 P2 Concrete Interface Resolution

## Status

- Status: **SOURCE_BUNDLE_READY**
- Candidate source files snapshotted: **23**
- Ranked relevant callables: **181**
- Loader callable candidates: **32**
- Evaluator callable candidates: **43**

## Highest-ranked loader callables

- Score **100** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/run_v5_p2_a4_loader_integration_tiny_overfit.py::module::main`
- Score **100** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/run_v5_p2_b0_r2_count_head_expansion_a4_repeat.py::module::main`
- Score **93** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/tune_v5_p2_b4_validation_thresholds.py::module::main`
- Score **70** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/train_v5_p2_b2_single_seed.py::module::validate`
- Score **68** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/audit_v5_p2_a3_loader_contract.py::module::main`
- Score **67** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/audit_v5_p2_b0_r3_corrected_nontest_label_shortcuts.py::module::main`
- Score **67** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/evaluate_v5_p2_b6_one_shot_test.py::module::evaluate_test_once`
- Score **63** — `/home/zira/research/projects/GNN-2d/scripts/v5/common/audit_v5_p0_a3_loader_contract_smoke.py::module::main`
- Score **63** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/audit_v5_p2_b0_nontest_label_shortcuts.py::module::main`
- Score **57** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/run_v5_p2_a4_loader_integration_tiny_overfit.py::module::evaluate`

## Highest-ranked evaluator callables

- Score **100** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/run_v5_p2_a4_loader_integration_tiny_overfit.py::module::main`
- Score **100** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/run_v5_p2_b0_r2_count_head_expansion_a4_repeat.py::module::main`
- Score **93** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/tune_v5_p2_b4_validation_thresholds.py::module::main`
- Score **70** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/train_v5_p2_b2_single_seed.py::module::validate`
- Score **68** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/audit_v5_p2_a3_loader_contract.py::module::main`
- Score **67** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/audit_v5_p2_b0_r3_corrected_nontest_label_shortcuts.py::module::main`
- Score **67** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/evaluate_v5_p2_b6_one_shot_test.py::module::evaluate_test_once`
- Score **63** — `/home/zira/research/projects/GNN-2d/scripts/v5/common/audit_v5_p0_a3_loader_contract_smoke.py::module::main`
- Score **63** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/audit_v5_p2_b0_nontest_label_shortcuts.py::module::main`
- Score **57** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/run_v5_p2_a4_loader_integration_tiny_overfit.py::module::evaluate`

## Frozen review bundle

- Path: `/home/zira/research/projects/GNN-2d/artifacts/v5/p2_preflight_concrete_interface_resolution/V5_P2_PREFLIGHT_CONCRETE_INTERFACE_SOURCE_BUNDLE.zip`
- SHA-256: `642cf0815f8e8de3b562678526bb0e9da01f8d0dae0e5a7167394c6644304618`

The bundle contains exact source snapshots and an AST-derived callable
inventory. It is the evidence used to construct and freeze the immutable
one-shot evaluator without guessing function names or argument contracts.

## Security boundary

- Dataset builder imported: **false**
- Dataset builder executed: **false**
- Test path resolved: **false**
- Test directory existence checked: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Concrete evaluator source frozen: **false**
- Ready for one-shot test: **false**
