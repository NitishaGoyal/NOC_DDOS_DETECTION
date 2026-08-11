# V5 P2 E1-P1 Compact Interface Resolution

## Compact summary

- Checkpoint SHA-256: `82314baedb4bf3969842abf9076679682369daae2e57db07d4b4aeaa8623f7cc`
- State-dict container: `model_state_dict`
- Candidate loader classes: `['V5P2PairAlignedPrimary58Dataset']`
- Candidate model classes: `['P2TaskDGraphConvCount4']`
- Detected batch keys: `['pair_id', 'pair_key', 'physical_port_mask', 'split', 'window_start', 'x', 'y_attacker_count', 'y_source', 'y_transit', 'y_victim']`
- Detected output keys: `['attack_logits', 'count_logits', 'path_logits', 'source_logits', 'transit_logits', 'victim_logits']`
- Validation tensors opened: **false**
- Test directory enumerated: **false**

## Checkpoint parameter prefixes

```json
{
  "attack_head": 2,
  "base_edge_index": 1,
  "count_head": 2,
  "graph1": 3,
  "graph2": 3,
  "graph_projection": 2,
  "input_projection": 2,
  "node_projection": 2,
  "path_head": 4,
  "source_head": 4,
  "temporal_blocks": 16,
  "transit_head": 4,
  "victim_head": 4
}
```

## Manifest interface

```json
{
  "bytes": 65038,
  "columns": [
    "split",
    "pair_key",
    "category",
    "pair_id",
    "attack_length",
    "control_length",
    "common_length",
    "window",
    "stride",
    "window_count_per_member",
    "first_window_start",
    "last_window_start",
    "last_window_end_exclusive",
    "attack_native_window_count",
    "control_native_window_count",
    "attack_windows_removed",
    "control_windows_removed",
    "identical_ordered_window_starts"
  ],
  "exists": true,
  "first_three_rows": [
    [
      "train",
      "P2TR-K1-001",
      "k1_local",
      "V5-P2-FINAL640-P2TR-K1-001-S5T5-SEED620001",
      "722",
      "722",
      "722",
      "32",
      "8",
      "87",
      "0",
      "688",
      "720",
      "87",
      "87",
      "0",
      "0",
      "True"
    ],
    [
      "train",
      "P2TR-K1-002",
      "k1_local",
      "V5-P2-FINAL640-P2TR-K1-002-S1T1-SEED620002",
      "707",
      "707",
      "707",
      "32",
      "8",
      "85",
      "0",
      "672",
      "704",
      "85",
      "85",
      "0",
      "0",
      "True"
    ],
    [
      "train",
      "P2TR-K1-003",
      "k1_local",
      "V5-P2-FINAL640-P2TR-K1-003-S9T9-SEED620003",
      "714",
      "714",
      "714",
      "32",
      "8",
      "86",
      "0",
      "680",
      "712",
      "86",
      "86",
      "0",
      "0",
      "True"
    ]
  ],
  "full_row_count_computed": false,
  "path": "/home/zira/research/projects/GNN-2d/reports/v5/p2_a1_r2_pair_aligned_window_contract/V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv",
  "sha256": "f42f40446d03161a6932ee060f6d8c859f894cb5075d1fc926ea161461413ab5"
}
```

## `scripts/v5/p2/train_v5_p2_task_d_full_multitask_single_run.py`

- SHA-256: `553dd206a8e6518b3a1e768c8f9c69633fcada421475b8ca75fda57c2fc13766`
- Constants:

```json
{
  "CANDIDATES": [
    "conv1d",
    "graphconv"
  ],
  "COUNT_CLASS_VALUES": [
    1,
    2,
    3,
    4
  ],
  "EXPECTED_B1_PROTOCOL_SHA": "0817ae7812f91e3c75260589acaf52bd8a74b07b2114a451b4773578efde4b60",
  "EXPECTED_PARAMS": {
    "conv1d": 43273,
    "graphconv": 59785
  },
  "EXPECTED_TRAIN_ITEMS": 70166,
  "EXPECTED_VALIDATION_ITEMS": 12528,
  "SEEDS": [
    107,
    117,
    127,
    137,
    147
  ],
  "STAGE": "V5_P2_TASK_D_FULL_MULTITASK_SINGLE_RUN"
}
```

- Classes:

```json
[]
```

### Paths Dictionary

Match line 63 (context starts 57):

```python
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def measure_latency(model, batch: dict[str, torch.Tensor], device: torch.device) -> float:
    x = batch["x"].to(device, non_blocking=(device.type == "cuda"))
    mask = batch["physical_port_mask"].to(device, non_blocking=(device.type == "cuda"))
    model.eval()
    warmup, repeats = 10, 40
    with torch.no_grad():
        for _ in range(warmup):
            model(x, mask)
        if device.type == "cuda":
```

Match line 89 (context starts 83):

```python
    parser.add_argument("--b1-dir", type=Path, required=True)
    parser.add_argument("--b0-r3-dir", type=Path, required=True)
    parser.add_argument("--promotion-dir", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--topology-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--b3-model-path", type=Path, required=True)
    parser.add_argument("--task-d-model-path", type=Path, required=True)
    parser.add_argument("--b2-training-script-path", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
```

Match line 90 (context starts 84):

```python
    parser.add_argument("--b0-r3-dir", type=Path, required=True)
    parser.add_argument("--promotion-dir", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--topology-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--b3-model-path", type=Path, required=True)
    parser.add_argument("--task-d-model-path", type=Path, required=True)
    parser.add_argument("--b2-training-script-path", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
```

Match line 91 (context starts 85):

```python
    parser.add_argument("--promotion-dir", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--topology-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--b3-model-path", type=Path, required=True)
    parser.add_argument("--task-d-model-path", type=Path, required=True)
    parser.add_argument("--b2-training-script-path", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
```

Match line 95 (context starts 89):

```python
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--b3-model-path", type=Path, required=True)
    parser.add_argument("--task-d-model-path", type=Path, required=True)
    parser.add_argument("--b2-training-script-path", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate = args.candidate
    seed = int(args.seed)
    if seed not in SEEDS:
```

Match line 107 (context starts 101):

```python
    if seed not in SEEDS:
        print(f"STOP: seed={seed} is not in {SEEDS}", file=sys.stderr)
        return 2

    resolved = {k: v.expanduser().resolve() if isinstance(v, Path) else v for k, v in vars(args).items()}
    root: Path = resolved["root"]
    model_dir: Path = resolved["model_dir"]
    report_dir: Path = resolved["report_dir"]
    if model_dir.exists() or report_dir.exists():
        print("STOP: model/report output already exists", file=sys.stderr)
        print(f"model_dir={model_dir}", file=sys.stderr)
        print(f"report_dir={report_dir}", file=sys.stderr)
        return 2
```

Match line 117 (context starts 111):

```python
        print(f"model_dir={model_dir}", file=sys.stderr)
        print(f"report_dir={report_dir}", file=sys.stderr)
        return 2
    model_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    paths = {
        "b1_protocol": resolved["b1_dir"] / "V5_P2_B1_TRAINING_PROTOCOL.json",
        "b1_report": resolved["b1_dir"] / "V5_P2_B1_TRAINING_PROTOCOL_LOCK.json",
        "b1_lock": resolved["b1_dir"] / "V5_P2_B1_TRAINING_PROTOCOL_LOCK_LOCK.json",
        "b0_report": resolved["b0_r3_dir"] / "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT.json",
        "b0_lock": resolved["b0_r3_dir"] / "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT_LOCK.json",
        "promotion_report": resolved["promotion_dir"] / "V5_P2_G123_GRAPH_OPERATOR_PROMOTION.json",
```

Match line 127 (context starts 121):

```python
        "b0_report": resolved["b0_r3_dir"] / "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT.json",
        "b0_lock": resolved["b0_r3_dir"] / "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT_LOCK.json",
        "promotion_report": resolved["promotion_dir"] / "V5_P2_G123_GRAPH_OPERATOR_PROMOTION.json",
        "promotion_lock": resolved["promotion_dir"] / "V5_P2_G123_GRAPH_OPERATOR_PROMOTION_LOCK.json",
        "preflight_report": resolved["preflight_dir"] / "V5_P2_TASK_D_FULL_MULTITASK_PREFLIGHT.json",
        "preflight_lock": resolved["preflight_dir"] / "V5_P2_TASK_D_FULL_MULTITASK_PREFLIGHT_LOCK.json",
        "edge": resolved["topology_dir"] / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
        "pair_manifest": resolved["pair_manifest"],
        "loader": resolved["loader_path"],
        "b3_model": resolved["b3_model_path"],
        "task_d_model": resolved["task_d_model_path"],
        "b2_train": resolved["b2_training_script_path"],
    }
```

Match line 128 (context starts 122):

```python
        "b0_lock": resolved["b0_r3_dir"] / "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT_LOCK.json",
        "promotion_report": resolved["promotion_dir"] / "V5_P2_G123_GRAPH_OPERATOR_PROMOTION.json",
        "promotion_lock": resolved["promotion_dir"] / "V5_P2_G123_GRAPH_OPERATOR_PROMOTION_LOCK.json",
        "preflight_report": resolved["preflight_dir"] / "V5_P2_TASK_D_FULL_MULTITASK_PREFLIGHT.json",
        "preflight_lock": resolved["preflight_dir"] / "V5_P2_TASK_D_FULL_MULTITASK_PREFLIGHT_LOCK.json",
        "edge": resolved["topology_dir"] / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
        "pair_manifest": resolved["pair_manifest"],
        "loader": resolved["loader_path"],
        "b3_model": resolved["b3_model_path"],
        "task_d_model": resolved["task_d_model_path"],
        "b2_train": resolved["b2_training_script_path"],
    }
    failures = [f"missing {name}: {path}" for name, path in paths.items() if not path.is_file()]
```

Match line 129 (context starts 123):

```python
        "promotion_report": resolved["promotion_dir"] / "V5_P2_G123_GRAPH_OPERATOR_PROMOTION.json",
        "promotion_lock": resolved["promotion_dir"] / "V5_P2_G123_GRAPH_OPERATOR_PROMOTION_LOCK.json",
        "preflight_report": resolved["preflight_dir"] / "V5_P2_TASK_D_FULL_MULTITASK_PREFLIGHT.json",
        "preflight_lock": resolved["preflight_dir"] / "V5_P2_TASK_D_FULL_MULTITASK_PREFLIGHT_LOCK.json",
        "edge": resolved["topology_dir"] / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
        "pair_manifest": resolved["pair_manifest"],
        "loader": resolved["loader_path"],
        "b3_model": resolved["b3_model_path"],
        "task_d_model": resolved["task_d_model_path"],
        "b2_train": resolved["b2_training_script_path"],
    }
    failures = [f"missing {name}: {path}" for name, path in paths.items() if not path.is_file()]
    warnings: list[str] = []
```

Match line 130 (context starts 124):

```python
        "promotion_lock": resolved["promotion_dir"] / "V5_P2_G123_GRAPH_OPERATOR_PROMOTION_LOCK.json",
        "preflight_report": resolved["preflight_dir"] / "V5_P2_TASK_D_FULL_MULTITASK_PREFLIGHT.json",
        "preflight_lock": resolved["preflight_dir"] / "V5_P2_TASK_D_FULL_MULTITASK_PREFLIGHT_LOCK.json",
        "edge": resolved["topology_dir"] / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
        "pair_manifest": resolved["pair_manifest"],
        "loader": resolved["loader_path"],
        "b3_model": resolved["b3_model_path"],
        "task_d_model": resolved["task_d_model_path"],
        "b2_train": resolved["b2_training_script_path"],
    }
    failures = [f"missing {name}: {path}" for name, path in paths.items() if not path.is_file()]
    warnings: list[str] = []
    if not root.is_dir():
```

Match line 131 (context starts 125):

```python
        "preflight_report": resolved["preflight_dir"] / "V5_P2_TASK_D_FULL_MULTITASK_PREFLIGHT.json",
        "preflight_lock": resolved["preflight_dir"] / "V5_P2_TASK_D_FULL_MULTITASK_PREFLIGHT_LOCK.json",
        "edge": resolved["topology_dir"] / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
        "pair_manifest": resolved["pair_manifest"],
        "loader": resolved["loader_path"],
        "b3_model": resolved["b3_model_path"],
        "task_d_model": resolved["task_d_model_path"],
        "b2_train": resolved["b2_training_script_path"],
    }
    failures = [f"missing {name}: {path}" for name, path in paths.items() if not path.is_file()]
    warnings: list[str] = []
    if not root.is_dir():
        failures.append(f"dataset root missing: {root}")
```

Match line 161 (context starts 155):

```python
    preflight_lock = json.loads(paths["preflight_lock"].read_text())

    if protocol.get("protocol_sha256") != EXPECTED_B1_PROTOCOL_SHA:
        failures.append("B1 protocol SHA changed")
    if b1_lock.get("protocol_file_sha256") != sha256_file(paths["b1_protocol"]):
        failures.append("B1 protocol file SHA mismatch")
    if b1_lock.get("model_sha256") != sha256_file(paths["b3_model"]):
        failures.append("B3 model SHA mismatch")
    if b1_lock.get("loader_sha256") != sha256_file(paths["loader"]):
        failures.append("loader SHA mismatch")
    if b0_lock.get("report_sha256") != sha256_file(paths["b0_report"]):
        failures.append("B0-R3 report SHA mismatch")
    if b0_lock.get("shortcut_block_count") != 0 or b0_lock.get("label_integrity_pass") is not True:
```

Match line 163 (context starts 157):

```python
    if protocol.get("protocol_sha256") != EXPECTED_B1_PROTOCOL_SHA:
        failures.append("B1 protocol SHA changed")
    if b1_lock.get("protocol_file_sha256") != sha256_file(paths["b1_protocol"]):
        failures.append("B1 protocol file SHA mismatch")
    if b1_lock.get("model_sha256") != sha256_file(paths["b3_model"]):
        failures.append("B3 model SHA mismatch")
    if b1_lock.get("loader_sha256") != sha256_file(paths["loader"]):
        failures.append("loader SHA mismatch")
    if b0_lock.get("report_sha256") != sha256_file(paths["b0_report"]):
        failures.append("B0-R3 report SHA mismatch")
    if b0_lock.get("shortcut_block_count") != 0 or b0_lock.get("label_integrity_pass") is not True:
        failures.append("B0-R3 label/shortcut audit is not clean")
    if promotion_lock.get("report_sha256") != sha256_file(paths["promotion_report"]):
```

Match line 192 (context starts 186):

```python
        (report_dir / f"{STAGE}_HOLD").write_text(f"{STAGE}_HOLD\n", encoding="utf-8")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    b2 = import_module(paths["b2_train"], f"taskd_b2_{candidate}_{seed}")
    loader_mod = import_module(paths["loader"], f"taskd_loader_{candidate}_{seed}")
    b3_mod = import_module(paths["b3_model"], f"taskd_b3_{candidate}_{seed}")
    td_mod = import_module(paths["task_d_model"], f"taskd_model_{candidate}_{seed}")

    b2.EXPECTED_SEEDS = SEEDS
    b2.EXPECTED_TRAIN_ITEMS = EXPECTED_TRAIN_ITEMS
    b2.EXPECTED_VALIDATION_ITEMS = EXPECTED_VALIDATION_ITEMS
```

Match line 201 (context starts 195):

```python

    b2.EXPECTED_SEEDS = SEEDS
    b2.EXPECTED_TRAIN_ITEMS = EXPECTED_TRAIN_ITEMS
    b2.EXPECTED_VALIDATION_ITEMS = EXPECTED_VALIDATION_ITEMS
    b2.set_seed(seed)

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation length={len(validation_dataset)}")
```

Match line 202 (context starts 196):

```python
    b2.EXPECTED_SEEDS = SEEDS
    b2.EXPECTED_TRAIN_ITEMS = EXPECTED_TRAIN_ITEMS
    b2.EXPECTED_VALIDATION_ITEMS = EXPECTED_VALIDATION_ITEMS
    b2.set_seed(seed)

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation length={len(validation_dataset)}")

```

Match line 203 (context starts 197):

```python
    b2.EXPECTED_TRAIN_ITEMS = EXPECTED_TRAIN_ITEMS
    b2.EXPECTED_VALIDATION_ITEMS = EXPECTED_VALIDATION_ITEMS
    b2.set_seed(seed)

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation length={len(validation_dataset)}")

    train_summary = b0_report["label_summaries"]["train"]
```

Match line 331 (context starts 325):

```python
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "b1_protocol_sha256": protocol["protocol_sha256"],
                "promotion_report_sha256": sha256_file(paths["promotion_report"]),
                "loader_sha256": sha256_file(paths["loader"]),
                "b3_model_sha256": sha256_file(paths["b3_model"]),
                "task_d_model_sha256": sha256_file(paths["task_d_model"]),
                "full_initial_state_sha256": full_initial_state_sha256,
                "common_initial_state_sha256": common_initial_state_sha256,
                "train_label_weight_manifest": {
                    "graph_positive_weight": graph_pos_weight_value,
```

Match line 332 (context starts 326):

```python
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "b1_protocol_sha256": protocol["protocol_sha256"],
                "promotion_report_sha256": sha256_file(paths["promotion_report"]),
                "loader_sha256": sha256_file(paths["loader"]),
                "b3_model_sha256": sha256_file(paths["b3_model"]),
                "task_d_model_sha256": sha256_file(paths["task_d_model"]),
                "full_initial_state_sha256": full_initial_state_sha256,
                "common_initial_state_sha256": common_initial_state_sha256,
                "train_label_weight_manifest": {
                    "graph_positive_weight": graph_pos_weight_value,
                    "role_positive_weights": role_weight_manifest,
```

Match line 333 (context starts 327):

```python
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "b1_protocol_sha256": protocol["protocol_sha256"],
                "promotion_report_sha256": sha256_file(paths["promotion_report"]),
                "loader_sha256": sha256_file(paths["loader"]),
                "b3_model_sha256": sha256_file(paths["b3_model"]),
                "task_d_model_sha256": sha256_file(paths["task_d_model"]),
                "full_initial_state_sha256": full_initial_state_sha256,
                "common_initial_state_sha256": common_initial_state_sha256,
                "train_label_weight_manifest": {
                    "graph_positive_weight": graph_pos_weight_value,
                    "role_positive_weights": role_weight_manifest,
                    "count_distribution": count_distribution,
```

Match line 413 (context starts 407):

```python
        "seed": seed,
        "architecture": {
            "name": model.architecture_name,
            "parameter_count": parameter_count,
            "full_initial_state_sha256": full_initial_state_sha256,
            "common_initial_state_sha256": common_initial_state_sha256,
            "b3_model_sha256": sha256_file(paths["b3_model"]),
            "task_d_model_sha256": sha256_file(paths["task_d_model"]),
            "loader_sha256": sha256_file(paths["loader"]),
            "b1_protocol_sha256": protocol["protocol_sha256"],
            "promotion_report_sha256": sha256_file(paths["promotion_report"]),
        },
        "data": {
```

Match line 414 (context starts 408):

```python
        "architecture": {
            "name": model.architecture_name,
            "parameter_count": parameter_count,
            "full_initial_state_sha256": full_initial_state_sha256,
            "common_initial_state_sha256": common_initial_state_sha256,
            "b3_model_sha256": sha256_file(paths["b3_model"]),
            "task_d_model_sha256": sha256_file(paths["task_d_model"]),
            "loader_sha256": sha256_file(paths["loader"]),
            "b1_protocol_sha256": protocol["protocol_sha256"],
            "promotion_report_sha256": sha256_file(paths["promotion_report"]),
        },
        "data": {
            "train_items": len(train_dataset), "validation_items": len(validation_dataset),
```

Match line 415 (context starts 409):

```python
            "name": model.architecture_name,
            "parameter_count": parameter_count,
            "full_initial_state_sha256": full_initial_state_sha256,
            "common_initial_state_sha256": common_initial_state_sha256,
            "b3_model_sha256": sha256_file(paths["b3_model"]),
            "task_d_model_sha256": sha256_file(paths["task_d_model"]),
            "loader_sha256": sha256_file(paths["loader"]),
            "b1_protocol_sha256": protocol["protocol_sha256"],
            "promotion_report_sha256": sha256_file(paths["promotion_report"]),
        },
        "data": {
            "train_items": len(train_dataset), "validation_items": len(validation_dataset),
            "pair_block_batch_size": 128, "item_batch_size": 256,
```

### Validation Dataset Constructor

Match line 203 (context starts 197):

```python
    b2.EXPECTED_TRAIN_ITEMS = EXPECTED_TRAIN_ITEMS
    b2.EXPECTED_VALIDATION_ITEMS = EXPECTED_VALIDATION_ITEMS
    b2.set_seed(seed)

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation length={len(validation_dataset)}")

    train_summary = b0_report["label_summaries"]["train"]
```

### Dataloader Constructor

Match line 240 (context starts 234):

```python
        raise RuntimeError("count distribution is not 1..4")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
    if candidate == "conv1d":
        model = reference_b3
```

Match line 241 (context starts 235):

```python

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
    if candidate == "conv1d":
        model = reference_b3
    else:
```

### Model Class

Match line 248 (context starts 242):

```python

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
    if candidate == "conv1d":
        model = reference_b3
    else:
        model = td_mod.P2TaskDGraphConvCount4(reference_b3, edge)
    model = model.to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != EXPECTED_PARAMS[candidate]:
        raise RuntimeError(f"parameter count={parameter_count}, expected {EXPECTED_PARAMS[candidate]}")

    full_initial_state_sha256 = b2.sha256_state_dict(model.state_dict())
```

### Checkpoint Load

Match line 326 (context starts 320):

```python
            best_validation_metrics = validation_metrics
            payload = {
                "stage": STAGE,
                "candidate": candidate,
                "seed": seed,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "b1_protocol_sha256": protocol["protocol_sha256"],
                "promotion_report_sha256": sha256_file(paths["promotion_report"]),
                "loader_sha256": sha256_file(paths["loader"]),
                "b3_model_sha256": sha256_file(paths["b3_model"]),
```

Match line 396 (context starts 390):

```python
            stopped_early = True
            print(f"early stopping at epoch {epoch}; patience reached 12")
            break

    if best_epoch is None or best_validation_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("no best checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    first_validation_batch = next(iter(validation_loader))
    latency_us = measure_latency(model, first_validation_batch, device)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    operation_proxy = td_mod.operation_count_proxy(candidate)

```

Match line 397 (context starts 391):

```python
            print(f"early stopping at epoch {epoch}; patience reached 12")
            break

    if best_epoch is None or best_validation_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("no best checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    first_validation_batch = next(iter(validation_loader))
    latency_us = measure_latency(model, first_validation_batch, device)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    operation_proxy = td_mod.operation_count_proxy(candidate)

    report = {
```

### Validation Iteration

Match line 398 (context starts 392):

```python
            break

    if best_epoch is None or best_validation_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("no best checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    first_validation_batch = next(iter(validation_loader))
    latency_us = measure_latency(model, first_validation_batch, device)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    operation_proxy = td_mod.operation_count_proxy(candidate)

    report = {
        "stage": STAGE,
```

### Batch Transfer

Match line 62 (context starts 56):

```python
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def measure_latency(model, batch: dict[str, torch.Tensor], device: torch.device) -> float:
    x = batch["x"].to(device, non_blocking=(device.type == "cuda"))
    mask = batch["physical_port_mask"].to(device, non_blocking=(device.type == "cuda"))
    model.eval()
    warmup, repeats = 10, 40
    with torch.no_grad():
        for _ in range(warmup):
            model(x, mask)
```

Match line 63 (context starts 57):

```python
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def measure_latency(model, batch: dict[str, torch.Tensor], device: torch.device) -> float:
    x = batch["x"].to(device, non_blocking=(device.type == "cuda"))
    mask = batch["physical_port_mask"].to(device, non_blocking=(device.type == "cuda"))
    model.eval()
    warmup, repeats = 10, 40
    with torch.no_grad():
        for _ in range(warmup):
            model(x, mask)
        if device.type == "cuda":
```

Match line 249 (context starts 243):

```python
    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
    if candidate == "conv1d":
        model = reference_b3
    else:
        model = td_mod.P2TaskDGraphConvCount4(reference_b3, edge)
    model = model.to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != EXPECTED_PARAMS[candidate]:
        raise RuntimeError(f"parameter count={parameter_count}, expected {EXPECTED_PARAMS[candidate]}")

    full_initial_state_sha256 = b2.sha256_state_dict(model.state_dict())
    if candidate == "conv1d":
```

### Stable Identity

Match line 423 (context starts 417):

```python
            "promotion_report_sha256": sha256_file(paths["promotion_report"]),
        },
        "data": {
            "train_items": len(train_dataset), "validation_items": len(validation_dataset),
            "pair_block_batch_size": 128, "item_batch_size": 256,
            "train_batches": len(train_loader), "validation_batches": len(validation_loader),
            "pair_keys_returned_to_model": False,
        },
        "label_weights": {
            "graph_positive_weight": graph_pos_weight_value,
            "role_positive_weights": role_weight_manifest,
            "count_distribution": count_distribution,
            "count_class_weights_applied": False,
```

### Expected Counts

Match line 25 (context starts 19):

```python

STAGE = "V5_P2_TASK_D_FULL_MULTITASK_SINGLE_RUN"
COMPLETE = f"{STAGE}_COMPLETE"
SEEDS = [107, 117, 127, 137, 147]
CANDIDATES = ["conv1d", "graphconv"]
EXPECTED_PARAMS = {"conv1d": 43_273, "graphconv": 59_785}
EXPECTED_TRAIN_ITEMS = 70_166
EXPECTED_VALIDATION_ITEMS = 12_528
EXPECTED_B1_PROTOCOL_SHA = "0817ae7812f91e3c75260589acaf52bd8a74b07b2114a451b4773578efde4b60"
COUNT_CLASS_VALUES = [1, 2, 3, 4]


def import_module(path: Path, name: str):
```

Match line 26 (context starts 20):

```python
STAGE = "V5_P2_TASK_D_FULL_MULTITASK_SINGLE_RUN"
COMPLETE = f"{STAGE}_COMPLETE"
SEEDS = [107, 117, 127, 137, 147]
CANDIDATES = ["conv1d", "graphconv"]
EXPECTED_PARAMS = {"conv1d": 43_273, "graphconv": 59_785}
EXPECTED_TRAIN_ITEMS = 70_166
EXPECTED_VALIDATION_ITEMS = 12_528
EXPECTED_B1_PROTOCOL_SHA = "0817ae7812f91e3c75260589acaf52bd8a74b07b2114a451b4773578efde4b60"
COUNT_CLASS_VALUES = [1, 2, 3, 4]


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
```

Match line 197 (context starts 191):

```python
    b2 = import_module(paths["b2_train"], f"taskd_b2_{candidate}_{seed}")
    loader_mod = import_module(paths["loader"], f"taskd_loader_{candidate}_{seed}")
    b3_mod = import_module(paths["b3_model"], f"taskd_b3_{candidate}_{seed}")
    td_mod = import_module(paths["task_d_model"], f"taskd_model_{candidate}_{seed}")

    b2.EXPECTED_SEEDS = SEEDS
    b2.EXPECTED_TRAIN_ITEMS = EXPECTED_TRAIN_ITEMS
    b2.EXPECTED_VALIDATION_ITEMS = EXPECTED_VALIDATION_ITEMS
    b2.set_seed(seed)

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
```

Match line 198 (context starts 192):

```python
    loader_mod = import_module(paths["loader"], f"taskd_loader_{candidate}_{seed}")
    b3_mod = import_module(paths["b3_model"], f"taskd_b3_{candidate}_{seed}")
    td_mod = import_module(paths["task_d_model"], f"taskd_model_{candidate}_{seed}")

    b2.EXPECTED_SEEDS = SEEDS
    b2.EXPECTED_TRAIN_ITEMS = EXPECTED_TRAIN_ITEMS
    b2.EXPECTED_VALIDATION_ITEMS = EXPECTED_VALIDATION_ITEMS
    b2.set_seed(seed)

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
```

Match line 204 (context starts 198):

```python
    b2.EXPECTED_VALIDATION_ITEMS = EXPECTED_VALIDATION_ITEMS
    b2.set_seed(seed)

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation length={len(validation_dataset)}")

    train_summary = b0_report["label_summaries"]["train"]
    graph_positive = int(train_summary["graph_positive"])
```

Match line 206 (context starts 200):

```python

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation length={len(validation_dataset)}")

    train_summary = b0_report["label_summaries"]["train"]
    graph_positive = int(train_summary["graph_positive"])
    graph_negative = int(train_summary["graph_negative"])
    graph_pos_weight_value = graph_negative / graph_positive
```

Match line 224 (context starts 218):

```python
        "path": "active_path_count_distribution",
    }
    role_weight_manifest: dict[str, Any] = {}
    for role, key in role_distribution_keys.items():
        distribution = b2.parse_distribution(train_summary[key])
        positives = b2.role_positive_count(distribution)
        negatives = EXPECTED_TRAIN_ITEMS * 16 - positives
        raw = negatives / positives
        role_weight_manifest[role] = {
            "positive_entries": positives,
            "negative_entries": negatives,
            "raw_positive_weight": raw,
            "clamped_positive_weight": min(20.0, max(1.0, raw)),
```

Match line 238 (context starts 232):

```python
    count_distribution = b2.parse_distribution(train_summary["active_count_distribution"])
    if sorted(count_distribution) != COUNT_CLASS_VALUES:
        raise RuntimeError("count distribution is not 1..4")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
```

Match line 239 (context starts 233):

```python
    if sorted(count_distribution) != COUNT_CLASS_VALUES:
        raise RuntimeError("count distribution is not 1..4")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
    if candidate == "conv1d":
```

Match line 295 (context starts 289):

```python

    print("===== V5 P2 TASK-D FULL MULTITASK SINGLE RUN =====")
    print("candidate:", candidate)
    print("seed:", seed)
    print("device:", device)
    print("train_items:", len(train_dataset))
    print("validation_items:", len(validation_dataset))
    print("parameter_count:", parameter_count)
    print("b1_protocol_sha256:", protocol["protocol_sha256"])
    print("threshold_tuning_performed: false")
    print("test_directory_enumerated: false")
    print("test_tensors_deserialized: false")

```

Match line 420 (context starts 414):

```python
            "task_d_model_sha256": sha256_file(paths["task_d_model"]),
            "loader_sha256": sha256_file(paths["loader"]),
            "b1_protocol_sha256": protocol["protocol_sha256"],
            "promotion_report_sha256": sha256_file(paths["promotion_report"]),
        },
        "data": {
            "train_items": len(train_dataset), "validation_items": len(validation_dataset),
            "pair_block_batch_size": 128, "item_batch_size": 256,
            "train_batches": len(train_loader), "validation_batches": len(validation_loader),
            "pair_keys_returned_to_model": False,
        },
        "label_weights": {
            "graph_positive_weight": graph_pos_weight_value,
```

Match line 421 (context starts 415):

```python
            "loader_sha256": sha256_file(paths["loader"]),
            "b1_protocol_sha256": protocol["protocol_sha256"],
            "promotion_report_sha256": sha256_file(paths["promotion_report"]),
        },
        "data": {
            "train_items": len(train_dataset), "validation_items": len(validation_dataset),
            "pair_block_batch_size": 128, "item_batch_size": 256,
            "train_batches": len(train_loader), "validation_batches": len(validation_loader),
            "pair_keys_returned_to_model": False,
        },
        "label_weights": {
            "graph_positive_weight": graph_pos_weight_value,
            "role_positive_weights": role_weight_manifest,
```

## `src/models/v5_p2_task_d_full_multitask_count4.py`

- SHA-256: `ccdfcb74c1ddab7f20b98c922444873ad76c204ec4b5fd6827d52fb716f11fd9`
- Constants:

```json
{}
```

- Classes:

```json
[
  {
    "line": 41,
    "methods": [
      {
        "args": [
          "self",
          "reference_b3",
          "base_edge_index"
        ],
        "line": 54,
        "name": "__init__"
      },
      {
        "args": [
          "self",
          "x",
          "physical_port_mask"
        ],
        "line": 86,
        "name": "encode_pre_graph"
      },
      {
        "args": [
          "self",
          "x",
          "physical_port_mask"
        ],
        "line": 115,
        "name": "forward"
      }
    ],
    "name": "P2TaskDGraphConvCount4"
  }
]
```

### Paths Dictionary

Match line 89 (context starts 83):

```python
                f"expected {self.expected_parameter_count}"
            )

    def encode_pre_graph(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (16, 58, 32):
            raise ValueError(f"x shape={tuple(x.shape)}, expected [B,16,58,32]")
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
```

Match line 94 (context starts 88):

```python
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (16, 58, 32):
            raise ValueError(f"x shape={tuple(x.shape)}, expected [B,16,58,32]")
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError("physical_port_mask must have shape [B,16,10]")

        batch_size, num_nodes, num_features, time_steps = x.shape
```

Match line 95 (context starts 89):

```python
        physical_port_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (16, 58, 32):
            raise ValueError(f"x shape={tuple(x.shape)}, expected [B,16,58,32]")
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError("physical_port_mask must have shape [B,16,10]")

        batch_size, num_nodes, num_features, time_steps = x.shape
        y = x.reshape(batch_size * num_nodes, num_features, time_steps)
```

Match line 96 (context starts 90):

```python
    ) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (16, 58, 32):
            raise ValueError(f"x shape={tuple(x.shape)}, expected [B,16,58,32]")
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError("physical_port_mask must have shape [B,16,10]")

        batch_size, num_nodes, num_features, time_steps = x.shape
        y = x.reshape(batch_size * num_nodes, num_features, time_steps)
        y = F.relu(self.input_projection(y))
```

Match line 98 (context starts 92):

```python
            raise ValueError(f"x shape={tuple(x.shape)}, expected [B,16,58,32]")
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError("physical_port_mask must have shape [B,16,10]")

        batch_size, num_nodes, num_features, time_steps = x.shape
        y = x.reshape(batch_size * num_nodes, num_features, time_steps)
        y = F.relu(self.input_projection(y))
        for block in self.temporal_blocks:
            y = block(y)
```

Match line 109 (context starts 103):

```python
        for block in self.temporal_blocks:
            y = block(y)
        temporal_embedding = y[:, :, -1].reshape(batch_size, num_nodes, 64)
        node_input = torch.cat(
            (
                temporal_embedding,
                physical_port_mask.to(dtype=temporal_embedding.dtype),
            ),
            dim=-1,
        )
        return F.relu(self.node_projection(node_input))

    def forward(
```

Match line 118 (context starts 112):

```python
        )
        return F.relu(self.node_projection(node_input))

    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_embedding = self.encode_pre_graph(x, physical_port_mask)
        batch_size, num_nodes, embedding_dim = node_embedding.shape
        flat = node_embedding.reshape(batch_size * num_nodes, embedding_dim)
        edges = batched_edge_index(
            self.base_edge_index,
```

Match line 120 (context starts 114):

```python

    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_embedding = self.encode_pre_graph(x, physical_port_mask)
        batch_size, num_nodes, embedding_dim = node_embedding.shape
        flat = node_embedding.reshape(batch_size * num_nodes, embedding_dim)
        edges = batched_edge_index(
            self.base_edge_index,
            batch_size,
            num_nodes,
```

### Getitem Return

Match line 168 (context starts 162):

```python
        return outputs


def common_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only parameters/buffers shared by both Task-D candidates."""
    excluded_prefixes = ("graph1.", "graph2.", "base_edge_index")
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith(excluded_prefixes)
    }


```

Match line 179 (context starts 173):

```python


def operation_count_proxy(candidate: str) -> dict[str, int]:
    # Linear/depthwise MAC proxy for one [16,58,32] item.
    base = 10_899_776
    if candidate == "conv1d":
        return {
            "total_linear_macs": base,
            "graph_linear_macs": 0,
            "graph_message_scalar_ops": 0,
        }
    if candidate == "graphconv":
        graph_linear = 2 * 2 * 16 * 64 * 64
```

Match line 186 (context starts 180):

```python
            "total_linear_macs": base,
            "graph_linear_macs": 0,
            "graph_message_scalar_ops": 0,
        }
    if candidate == "graphconv":
        graph_linear = 2 * 2 * 16 * 64 * 64
        return {
            "total_linear_macs": base + graph_linear,
            "graph_linear_macs": graph_linear,
            "graph_message_scalar_ops": 2 * 48 * 64,
        }
    raise ValueError(candidate)
```

### Model Class

Match line 41 (context starts 35):

```python
    target_graph = torch.div(result[1], num_nodes, rounding_mode="floor")
    if not torch.equal(source_graph, target_graph):
        raise RuntimeError("batched edge_index contains cross-graph edges")
    return result


class P2TaskDGraphConvCount4(nn.Module):
    """
    B3 temporal encoder + two frozen-width GraphConv layers + unchanged B3 heads.

    Common B3 modules are deep-copied from a seeded reference B3 instance so
    the Conv1D and GraphConv candidates begin with byte-identical common
    parameters for every matched seed.
```

Match line 115 (context starts 109):

```python
                physical_port_mask.to(dtype=temporal_embedding.dtype),
            ),
            dim=-1,
        )
        return F.relu(self.node_projection(node_input))

    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_embedding = self.encode_pre_graph(x, physical_port_mask)
        batch_size, num_nodes, embedding_dim = node_embedding.shape
```

### Model Output

Match line 133 (context starts 127):

```python
            flat.device,
        )
        flat = self.activation(self.graph1(flat, edges))
        flat = self.activation(self.graph2(flat, edges))
        node_embedding = flat.reshape(batch_size, num_nodes, embedding_dim)

        source_logits = self.source_head(node_embedding).squeeze(-1)
        transit_logits = self.transit_head(node_embedding).squeeze(-1)
        victim_logits = self.victim_head(node_embedding).squeeze(-1)
        path_logits = self.path_head(node_embedding).squeeze(-1)

        pooled = torch.cat(
            (node_embedding.mean(dim=1), node_embedding.amax(dim=1)),
```

Match line 134 (context starts 128):

```python
        )
        flat = self.activation(self.graph1(flat, edges))
        flat = self.activation(self.graph2(flat, edges))
        node_embedding = flat.reshape(batch_size, num_nodes, embedding_dim)

        source_logits = self.source_head(node_embedding).squeeze(-1)
        transit_logits = self.transit_head(node_embedding).squeeze(-1)
        victim_logits = self.victim_head(node_embedding).squeeze(-1)
        path_logits = self.path_head(node_embedding).squeeze(-1)

        pooled = torch.cat(
            (node_embedding.mean(dim=1), node_embedding.amax(dim=1)),
            dim=-1,
```

Match line 135 (context starts 129):

```python
        flat = self.activation(self.graph1(flat, edges))
        flat = self.activation(self.graph2(flat, edges))
        node_embedding = flat.reshape(batch_size, num_nodes, embedding_dim)

        source_logits = self.source_head(node_embedding).squeeze(-1)
        transit_logits = self.transit_head(node_embedding).squeeze(-1)
        victim_logits = self.victim_head(node_embedding).squeeze(-1)
        path_logits = self.path_head(node_embedding).squeeze(-1)

        pooled = torch.cat(
            (node_embedding.mean(dim=1), node_embedding.amax(dim=1)),
            dim=-1,
        )
```

Match line 136 (context starts 130):

```python
        flat = self.activation(self.graph2(flat, edges))
        node_embedding = flat.reshape(batch_size, num_nodes, embedding_dim)

        source_logits = self.source_head(node_embedding).squeeze(-1)
        transit_logits = self.transit_head(node_embedding).squeeze(-1)
        victim_logits = self.victim_head(node_embedding).squeeze(-1)
        path_logits = self.path_head(node_embedding).squeeze(-1)

        pooled = torch.cat(
            (node_embedding.mean(dim=1), node_embedding.amax(dim=1)),
            dim=-1,
        )
        graph_embedding = self.graph_projection(pooled)
```

Match line 144 (context starts 138):

```python
        pooled = torch.cat(
            (node_embedding.mean(dim=1), node_embedding.amax(dim=1)),
            dim=-1,
        )
        graph_embedding = self.graph_projection(pooled)
        outputs = {
            "attack_logits": self.attack_head(graph_embedding).squeeze(-1),
            "count_logits": self.count_head(graph_embedding),
            "source_logits": source_logits,
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
```

Match line 145 (context starts 139):

```python
            (node_embedding.mean(dim=1), node_embedding.amax(dim=1)),
            dim=-1,
        )
        graph_embedding = self.graph_projection(pooled)
        outputs = {
            "attack_logits": self.attack_head(graph_embedding).squeeze(-1),
            "count_logits": self.count_head(graph_embedding),
            "source_logits": source_logits,
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
        expected = {
```

Match line 146 (context starts 140):

```python
            dim=-1,
        )
        graph_embedding = self.graph_projection(pooled)
        outputs = {
            "attack_logits": self.attack_head(graph_embedding).squeeze(-1),
            "count_logits": self.count_head(graph_embedding),
            "source_logits": source_logits,
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
        expected = {
            "attack_logits": (batch_size,),
```

Match line 147 (context starts 141):

```python
        )
        graph_embedding = self.graph_projection(pooled)
        outputs = {
            "attack_logits": self.attack_head(graph_embedding).squeeze(-1),
            "count_logits": self.count_head(graph_embedding),
            "source_logits": source_logits,
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
```

Match line 148 (context starts 142):

```python
        graph_embedding = self.graph_projection(pooled)
        outputs = {
            "attack_logits": self.attack_head(graph_embedding).squeeze(-1),
            "count_logits": self.count_head(graph_embedding),
            "source_logits": source_logits,
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
```

Match line 149 (context starts 143):

```python
        outputs = {
            "attack_logits": self.attack_head(graph_embedding).squeeze(-1),
            "count_logits": self.count_head(graph_embedding),
            "source_logits": source_logits,
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
```

Match line 152 (context starts 146):

```python
            "source_logits": source_logits,
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
```

Match line 153 (context starts 147):

```python
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
        for key, shape in expected.items():
```

Match line 154 (context starts 148):

```python
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
        for key, shape in expected.items():
            if tuple(outputs[key].shape) != shape:
```

Match line 155 (context starts 149):

```python
            "path_logits": path_logits,
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
        for key, shape in expected.items():
            if tuple(outputs[key].shape) != shape:
                raise RuntimeError(f"{key} shape={tuple(outputs[key].shape)}, expected {shape}")
```

Match line 156 (context starts 150):

```python
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
        for key, shape in expected.items():
            if tuple(outputs[key].shape) != shape:
                raise RuntimeError(f"{key} shape={tuple(outputs[key].shape)}, expected {shape}")
        return outputs
```

Match line 157 (context starts 151):

```python
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
        for key, shape in expected.items():
            if tuple(outputs[key].shape) != shape:
                raise RuntimeError(f"{key} shape={tuple(outputs[key].shape)}, expected {shape}")
        return outputs

```

### Batch Transfer

Match line 25 (context starts 19):

```python
def batched_edge_index(
    base_edge_index: torch.Tensor,
    batch_size: int,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    edge = base_edge_index.to(device=device, dtype=torch.long)
    if tuple(edge.shape) != (2, 48):
        raise ValueError(f"base_edge_index shape={tuple(edge.shape)}, expected [2,48]")
    offsets = (
        torch.arange(batch_size, device=device, dtype=torch.long)
        .view(batch_size, 1, 1)
        * num_nodes
```

### Expected Counts

Match line 21 (context starts 15):

```python
else:
    _PYG_IMPORT_ERROR = None


def batched_edge_index(
    base_edge_index: torch.Tensor,
    batch_size: int,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    edge = base_edge_index.to(device=device, dtype=torch.long)
    if tuple(edge.shape) != (2, 48):
        raise ValueError(f"base_edge_index shape={tuple(edge.shape)}, expected [2,48]")
```

Match line 29 (context starts 23):

```python
    device: torch.device,
) -> torch.Tensor:
    edge = base_edge_index.to(device=device, dtype=torch.long)
    if tuple(edge.shape) != (2, 48):
        raise ValueError(f"base_edge_index shape={tuple(edge.shape)}, expected [2,48]")
    offsets = (
        torch.arange(batch_size, device=device, dtype=torch.long)
        .view(batch_size, 1, 1)
        * num_nodes
    )
    result = (edge.view(1, 2, 48) + offsets).permute(1, 0, 2).reshape(2, -1)
    source_graph = torch.div(result[0], num_nodes, rounding_mode="floor")
    target_graph = torch.div(result[1], num_nodes, rounding_mode="floor")
```

Match line 30 (context starts 24):

```python
) -> torch.Tensor:
    edge = base_edge_index.to(device=device, dtype=torch.long)
    if tuple(edge.shape) != (2, 48):
        raise ValueError(f"base_edge_index shape={tuple(edge.shape)}, expected [2,48]")
    offsets = (
        torch.arange(batch_size, device=device, dtype=torch.long)
        .view(batch_size, 1, 1)
        * num_nodes
    )
    result = (edge.view(1, 2, 48) + offsets).permute(1, 0, 2).reshape(2, -1)
    source_graph = torch.div(result[0], num_nodes, rounding_mode="floor")
    target_graph = torch.div(result[1], num_nodes, rounding_mode="floor")
    if not torch.equal(source_graph, target_graph):
```

Match line 100 (context starts 94):

```python
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError("physical_port_mask must have shape [B,16,10]")

        batch_size, num_nodes, num_features, time_steps = x.shape
        y = x.reshape(batch_size * num_nodes, num_features, time_steps)
        y = F.relu(self.input_projection(y))
        for block in self.temporal_blocks:
            y = block(y)
        temporal_embedding = y[:, :, -1].reshape(batch_size, num_nodes, 64)
        node_input = torch.cat(
```

Match line 101 (context starts 95):

```python
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError("physical_port_mask must have shape [B,16,10]")

        batch_size, num_nodes, num_features, time_steps = x.shape
        y = x.reshape(batch_size * num_nodes, num_features, time_steps)
        y = F.relu(self.input_projection(y))
        for block in self.temporal_blocks:
            y = block(y)
        temporal_embedding = y[:, :, -1].reshape(batch_size, num_nodes, 64)
        node_input = torch.cat(
            (
```

Match line 105 (context starts 99):

```python

        batch_size, num_nodes, num_features, time_steps = x.shape
        y = x.reshape(batch_size * num_nodes, num_features, time_steps)
        y = F.relu(self.input_projection(y))
        for block in self.temporal_blocks:
            y = block(y)
        temporal_embedding = y[:, :, -1].reshape(batch_size, num_nodes, 64)
        node_input = torch.cat(
            (
                temporal_embedding,
                physical_port_mask.to(dtype=temporal_embedding.dtype),
            ),
            dim=-1,
```

Match line 121 (context starts 115):

```python
    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_embedding = self.encode_pre_graph(x, physical_port_mask)
        batch_size, num_nodes, embedding_dim = node_embedding.shape
        flat = node_embedding.reshape(batch_size * num_nodes, embedding_dim)
        edges = batched_edge_index(
            self.base_edge_index,
            batch_size,
            num_nodes,
            flat.device,
```

Match line 122 (context starts 116):

```python
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_embedding = self.encode_pre_graph(x, physical_port_mask)
        batch_size, num_nodes, embedding_dim = node_embedding.shape
        flat = node_embedding.reshape(batch_size * num_nodes, embedding_dim)
        edges = batched_edge_index(
            self.base_edge_index,
            batch_size,
            num_nodes,
            flat.device,
        )
```

Match line 125 (context starts 119):

```python
    ) -> dict[str, torch.Tensor]:
        node_embedding = self.encode_pre_graph(x, physical_port_mask)
        batch_size, num_nodes, embedding_dim = node_embedding.shape
        flat = node_embedding.reshape(batch_size * num_nodes, embedding_dim)
        edges = batched_edge_index(
            self.base_edge_index,
            batch_size,
            num_nodes,
            flat.device,
        )
        flat = self.activation(self.graph1(flat, edges))
        flat = self.activation(self.graph2(flat, edges))
        node_embedding = flat.reshape(batch_size, num_nodes, embedding_dim)
```

Match line 131 (context starts 125):

```python
            batch_size,
            num_nodes,
            flat.device,
        )
        flat = self.activation(self.graph1(flat, edges))
        flat = self.activation(self.graph2(flat, edges))
        node_embedding = flat.reshape(batch_size, num_nodes, embedding_dim)

        source_logits = self.source_head(node_embedding).squeeze(-1)
        transit_logits = self.transit_head(node_embedding).squeeze(-1)
        victim_logits = self.victim_head(node_embedding).squeeze(-1)
        path_logits = self.path_head(node_embedding).squeeze(-1)

```

Match line 152 (context starts 146):

```python
            "source_logits": source_logits,
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
```

Match line 153 (context starts 147):

```python
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
        for key, shape in expected.items():
```

Match line 154 (context starts 148):

```python
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
        for key, shape in expected.items():
            if tuple(outputs[key].shape) != shape:
```

Match line 155 (context starts 149):

```python
            "path_logits": path_logits,
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
        for key, shape in expected.items():
            if tuple(outputs[key].shape) != shape:
                raise RuntimeError(f"{key} shape={tuple(outputs[key].shape)}, expected {shape}")
```

Match line 156 (context starts 150):

```python
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
        for key, shape in expected.items():
            if tuple(outputs[key].shape) != shape:
                raise RuntimeError(f"{key} shape={tuple(outputs[key].shape)}, expected {shape}")
        return outputs
```

Match line 157 (context starts 151):

```python
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
        for key, shape in expected.items():
            if tuple(outputs[key].shape) != shape:
                raise RuntimeError(f"{key} shape={tuple(outputs[key].shape)}, expected {shape}")
        return outputs

```

## `src/data/v5_p2_pair_aligned_primary58_dataset.py`

- SHA-256: `2725ff993f4f03ebee3d5ffb775b1fedd6b131a9c4c89ed8049f45b249c24ac2`
- Constants:

```json
{
  "AUTHORIZED_SPLITS": [
    "train",
    "validation"
  ],
  "MODEL_ITEM_KEYS": [
    "physical_port_mask",
    "role_mask",
    "x",
    "y_attack",
    "y_attack_path",
    "y_attacker_count",
    "y_source",
    "y_transit",
    "y_victim"
  ],
  "STRIDE": 8,
  "WINDOW": 32
}
```

- Classes:

```json
[
  {
    "line": 62,
    "methods": [],
    "name": "IndexEntry"
  },
  {
    "line": 137,
    "methods": [
      {
        "args": [
          "self",
          "root",
          "split",
          "pair_manifest"
        ],
        "line": 151,
        "name": "__init__"
      },
      {
        "args": [
          "path"
        ],
        "line": 318,
        "name": "_load_run"
      },
      {
        "args": [
          "self"
        ],
        "line": 328,
        "name": "__len__"
      },
      {
        "args": [
          "self"
        ],
        "line": 332,
        "name": "pair_count"
      },
      {
        "args": [
          "self",
          "index"
        ],
        "line": 335,
        "name": "__getitem__"
      }
    ],
    "name": "V5P2PairAlignedPrimary58Dataset"
  }
]
```

### Paths Dictionary

Match line 14 (context starts 8):

```python

The P2 test split is intentionally rejected. A separately locked test loader
must be created only after the P2 model, checkpoint, and thresholds are frozen.

Returned model/target dictionary:
    x                  float32 [16,58,32]
    physical_port_mask bool    [16,10]
    y_attack           float32 scalar
    y_attacker_count   int64   scalar
    y_source           float32 [16]
    y_transit          float32 [16]
    y_victim           float32 [16]
    y_attack_path      float32 [16]
```

Match line 39 (context starts 33):

```python
from typing import Any

import torch
from torch.utils.data import Dataset


PRIMARY58_INDICES = tuple(
    list(range(0, 30))
    + list(range(31, 56))
    + [62, 64, 65]
)
WINDOW = 32
STRIDE = 8
```

Match line 50 (context starts 44):

```python
WINDOW = 32
STRIDE = 8
AUTHORIZED_SPLITS = {"train", "validation"}

MODEL_ITEM_KEYS = {
    "x",
    "physical_port_mask",
    "y_attack",
    "y_attacker_count",
    "y_source",
    "y_transit",
    "y_victim",
    "y_attack_path",
```

Match line 81 (context starts 75):

```python
        raise ValueError(
            f"Invalid integer field {key!r} for "
            f"{row.get('split')}/{row.get('pair_key')}"
        ) from exc


def derive_corrected_physical_port_mask(
    topology: dict[str, Any],
) -> torch.Tensor:
    """
    Derive Boolean [16,10] in this exact order:
      input : local,north,east,south,west
      output: local,north,east,south,west
```

Match line 137 (context starts 131):

```python
            f"derived physical-port mask true count is "
            f"{int(mask.sum().item())}, expected 128"
        )
    return mask.contiguous()


class V5P2PairAlignedPrimary58Dataset(Dataset):
    """
    Pair-aligned P2 train/validation windows for frozen B3.

    Window-index construction comes exclusively from the frozen A1-R2 pair
    manifest. Both ATTACK and CONTROL members of a matched pair receive the
    same ordered starts inside:
```

Match line 155 (context starts 149):

```python
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        pair_manifest: str | Path,
        *,
        window: int = WINDOW,
        stride: int = STRIDE,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = str(split)
```

Match line 162 (context starts 156):

```python
        *,
        window: int = WINDOW,
        stride: int = STRIDE,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = str(split)
        self.pair_manifest = Path(pair_manifest).expanduser().resolve()
        self.window = int(window)
        self.stride = int(stride)

        if self.split not in AUTHORIZED_SPLITS:
            raise ValueError(
                "This audited loader authorizes only train and validation. "
```

Match line 181 (context starts 175):

```python
        if self.stride != STRIDE:
            raise ValueError(
                f"stride must remain frozen at {STRIDE}, got {self.stride}"
            )
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root missing: {self.root}")
        if not self.pair_manifest.is_file():
            raise FileNotFoundError(
                f"pair manifest missing: {self.pair_manifest}"
            )

        split_dir = self.root / "runs" / self.split
        if not split_dir.is_dir():
```

Match line 183 (context starts 177):

```python
                f"stride must remain frozen at {STRIDE}, got {self.stride}"
            )
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root missing: {self.root}")
        if not self.pair_manifest.is_file():
            raise FileNotFoundError(
                f"pair manifest missing: {self.pair_manifest}"
            )

        split_dir = self.root / "runs" / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"authorized split directory missing: {split_dir}"
```

Match line 199 (context starts 193):

```python
            self.root / "topology.pt",
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(topology, dict):
            raise TypeError("topology.pt payload must be a dictionary")
        self.physical_port_mask = (
            derive_corrected_physical_port_mask(topology)
        )

        self._index: list[IndexEntry] = []
        self._pair_count = 0

```

Match line 200 (context starts 194):

```python
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(topology, dict):
            raise TypeError("topology.pt payload must be a dictionary")
        self.physical_port_mask = (
            derive_corrected_physical_port_mask(topology)
        )

        self._index: list[IndexEntry] = []
        self._pair_count = 0

        with self.pair_manifest.open(
```

Match line 206 (context starts 200):

```python
            derive_corrected_physical_port_mask(topology)
        )

        self._index: list[IndexEntry] = []
        self._pair_count = 0

        with self.pair_manifest.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))

```

Match line 360 (context starts 354):

```python

        stop = entry.start + self.window
        x = (
            x_stored[
                entry.start:stop,
                :,
                PRIMARY58_INDICES,
            ]
            .permute(1, 2, 0)
            .contiguous()
        )

        if tuple(x.shape) != (16, 58, 32):
```

Match line 374 (context starts 368):

```python
                f"constructed x shape={tuple(x.shape)}, "
                "expected [16,58,32]"
            )

        item = {
            "x": x,
            "physical_port_mask": (
                self.physical_port_mask.clone()
            ),
            "y_attack": run["y_attack"][entry.target].clone(),
            "y_attacker_count": (
                run["y_attacker_count"][entry.target].clone()
            ),
```

Match line 375 (context starts 369):

```python
                "expected [16,58,32]"
            )

        item = {
            "x": x,
            "physical_port_mask": (
                self.physical_port_mask.clone()
            ),
            "y_attack": run["y_attack"][entry.target].clone(),
            "y_attacker_count": (
                run["y_attacker_count"][entry.target].clone()
            ),
            "y_source": run["y_source"][entry.target].clone(),
```

### Getitem Return

Match line 335 (context starts 329):

```python
        return len(self._index)

    @property
    def pair_count(self) -> int:
        return self._pair_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self._index[index]
        run = self._load_run(entry.file_path)

        x_stored = run.get("x")
        if (
            not isinstance(x_stored, torch.Tensor)
```

### Checkpoint Load

Match line 192 (context starts 186):

```python
        split_dir = self.root / "runs" / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"authorized split directory missing: {split_dir}"
            )

        topology = torch.load(
            self.root / "topology.pt",
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(topology, dict):
            raise TypeError("topology.pt payload must be a dictionary")
```

Match line 319 (context starts 313):

```python
                "pair-aligned dataset length must be even"
            )

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_run(path: Path) -> dict[str, Any]:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, dict):
            raise TypeError(f"run payload is not a dictionary: {path}")
```

### Stable Identity

Match line 64 (context starts 58):

```python
}


@dataclass(frozen=True)
class IndexEntry:
    file_path: Path
    pair_key: str
    mode: str
    start: int
    target: int
    common_length: int


```

Match line 77 (context starts 71):

```python
def _as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid integer field {key!r} for "
            f"{row.get('split')}/{row.get('pair_key')}"
        ) from exc


def derive_corrected_physical_port_mask(
    topology: dict[str, Any],
) -> torch.Tensor:
```

Match line 218 (context starts 212):

```python

        seen_pairs: set[str] = set()
        for row in rows:
            if row.get("split") != self.split:
                continue

            pair_key = row.get("pair_key", "")
            if not pair_key:
                raise ValueError("pair manifest row has empty pair_key")
            if pair_key in seen_pairs:
                raise ValueError(
                    f"duplicate pair manifest row: "
                    f"{self.split}/{pair_key}"
```

Match line 219 (context starts 213):

```python
        seen_pairs: set[str] = set()
        for row in rows:
            if row.get("split") != self.split:
                continue

            pair_key = row.get("pair_key", "")
            if not pair_key:
                raise ValueError("pair manifest row has empty pair_key")
            if pair_key in seen_pairs:
                raise ValueError(
                    f"duplicate pair manifest row: "
                    f"{self.split}/{pair_key}"
                )
```

Match line 220 (context starts 214):

```python
        for row in rows:
            if row.get("split") != self.split:
                continue

            pair_key = row.get("pair_key", "")
            if not pair_key:
                raise ValueError("pair manifest row has empty pair_key")
            if pair_key in seen_pairs:
                raise ValueError(
                    f"duplicate pair manifest row: "
                    f"{self.split}/{pair_key}"
                )
            seen_pairs.add(pair_key)
```

Match line 221 (context starts 215):

```python
            if row.get("split") != self.split:
                continue

            pair_key = row.get("pair_key", "")
            if not pair_key:
                raise ValueError("pair manifest row has empty pair_key")
            if pair_key in seen_pairs:
                raise ValueError(
                    f"duplicate pair manifest row: "
                    f"{self.split}/{pair_key}"
                )
            seen_pairs.add(pair_key)
            self._pair_count += 1
```

Match line 224 (context starts 218):

```python
            pair_key = row.get("pair_key", "")
            if not pair_key:
                raise ValueError("pair manifest row has empty pair_key")
            if pair_key in seen_pairs:
                raise ValueError(
                    f"duplicate pair manifest row: "
                    f"{self.split}/{pair_key}"
                )
            seen_pairs.add(pair_key)
            self._pair_count += 1

            common_length = _as_int(row, "common_length")
            window_count = _as_int(
```

Match line 226 (context starts 220):

```python
                raise ValueError("pair manifest row has empty pair_key")
            if pair_key in seen_pairs:
                raise ValueError(
                    f"duplicate pair manifest row: "
                    f"{self.split}/{pair_key}"
                )
            seen_pairs.add(pair_key)
            self._pair_count += 1

            common_length = _as_int(row, "common_length")
            window_count = _as_int(
                row,
                "window_count_per_member",
```

Match line 239 (context starts 233):

```python
            )
            manifest_window = _as_int(row, "window")
            manifest_stride = _as_int(row, "stride")

            if manifest_window != self.window:
                raise ValueError(
                    f"{self.split}/{pair_key}: manifest window="
                    f"{manifest_window}, expected {self.window}"
                )
            if manifest_stride != self.stride:
                raise ValueError(
                    f"{self.split}/{pair_key}: manifest stride="
                    f"{manifest_stride}, expected {self.stride}"
```

Match line 244 (context starts 238):

```python
                raise ValueError(
                    f"{self.split}/{pair_key}: manifest window="
                    f"{manifest_window}, expected {self.window}"
                )
            if manifest_stride != self.stride:
                raise ValueError(
                    f"{self.split}/{pair_key}: manifest stride="
                    f"{manifest_stride}, expected {self.stride}"
                )

            expected_window_count = (
                0
                if common_length < self.window
```

Match line 256 (context starts 250):

```python
                if common_length < self.window
                else 1
                + (common_length - self.window) // self.stride
            )
            if window_count != expected_window_count:
                raise ValueError(
                    f"{self.split}/{pair_key}: manifest window count="
                    f"{window_count}, expected {expected_window_count}"
                )

            attack_path = split_dir / f"{pair_key}_ATTACK.pt"
            control_path = split_dir / f"{pair_key}_CONTROL.pt"
            if not attack_path.is_file():
```

Match line 260 (context starts 254):

```python
            if window_count != expected_window_count:
                raise ValueError(
                    f"{self.split}/{pair_key}: manifest window count="
                    f"{window_count}, expected {expected_window_count}"
                )

            attack_path = split_dir / f"{pair_key}_ATTACK.pt"
            control_path = split_dir / f"{pair_key}_CONTROL.pt"
            if not attack_path.is_file():
                raise FileNotFoundError(
                    f"missing ATTACK tensor: {attack_path}"
                )
            if not control_path.is_file():
```

Match line 261 (context starts 255):

```python
                raise ValueError(
                    f"{self.split}/{pair_key}: manifest window count="
                    f"{window_count}, expected {expected_window_count}"
                )

            attack_path = split_dir / f"{pair_key}_ATTACK.pt"
            control_path = split_dir / f"{pair_key}_CONTROL.pt"
            if not attack_path.is_file():
                raise FileNotFoundError(
                    f"missing ATTACK tensor: {attack_path}"
                )
            if not control_path.is_file():
                raise FileNotFoundError(
```

Match line 277 (context starts 271):

```python
            for window_index in range(window_count):
                start = window_index * self.stride
                target = start + self.window - 1

                if target >= common_length:
                    raise ValueError(
                        f"{self.split}/{pair_key}: target {target} "
                        f"outside common length {common_length}"
                    )

                # Pair-major, start-major, ATTACK then CONTROL.
                self._index.append(
                    IndexEntry(
```

Match line 285 (context starts 279):

```python
                    )

                # Pair-major, start-major, ATTACK then CONTROL.
                self._index.append(
                    IndexEntry(
                        file_path=attack_path,
                        pair_key=pair_key,
                        mode="attack",
                        start=start,
                        target=target,
                        common_length=common_length,
                    )
                )
```

Match line 295 (context starts 289):

```python
                        common_length=common_length,
                    )
                )
                self._index.append(
                    IndexEntry(
                        file_path=control_path,
                        pair_key=pair_key,
                        mode="control",
                        start=start,
                        target=target,
                        common_length=common_length,
                    )
                )
```

## `scripts/v5/p2/aggregate_v5_p2_task_d_full_multitask_matrix.py`

- SHA-256: `660288f0b1ce47fece4406410bec7ada922970da18dbdbca879bff745ce78c79`
- Constants:

```json
{
  "CANDIDATES": [
    "conv1d",
    "graphconv"
  ],
  "SEEDS": [
    107,
    117,
    127,
    137,
    147
  ],
  "STAGE": "V5_P2_TASK_D_FULL_MULTITASK_MATRIX_AGGREGATION"
}
```

- Classes:

```json
[]
```

## `scripts/v5/p2/audit_v5_p2_a3_loader_contract.py`

- SHA-256: `09d646aa39d47b7e541d9cb5b167a49beda2dfa79bb27e935132e1156c9cbd86`
- Constants:

```json
{
  "EXPECTED_ITEM_KEYS": [
    "physical_port_mask",
    "role_mask",
    "x",
    "y_attack",
    "y_attack_path",
    "y_attacker_count",
    "y_source",
    "y_transit",
    "y_victim"
  ],
  "EXPECTED_TOTAL_ALIGNED_WINDOWS": 82694,
  "EXPECTED_TOTAL_PAIRS": 489,
  "EXPECTED_TRAIN_PAIRS": 415,
  "EXPECTED_VALIDATION_PAIRS": 74,
  "PROHIBITED_ITEM_KEYS": [
    "case_id",
    "category",
    "common_length",
    "edge_index",
    "epoch_id",
    "filename",
    "mode",
    "native_run_length",
    "pair_id",
    "router_coordinates",
    "run_id",
    "split",
    "window_end",
    "window_start",
    "window_target"
  ],
  "STAGE": "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT"
}
```

- Classes:

```json
[]
```

### Paths Dictionary

Match line 3 (context starts 1):

```python
#!/usr/bin/env python3
"""
V5 P2-A3 Pair-Aligned PRIMARY58 Loader Contract Audit

Installs/audits the replacement P2 train/validation dataset loader.

The audit:
- verifies A0, A1-R2, and A2-R2 locks and hashes;
- constructs TRAIN and VALIDATION datasets only;
```

Match line 14 (context starts 8):

```python
- verifies A0, A1-R2, and A2-R2 locks and hashes;
- constructs TRAIN and VALIDATION datasets only;
- validates all pair-aligned index entries without loading test;
- verifies total aligned-window count = 82,694;
- checks pair adjacency and identical ATTACK/CONTROL starts;
- loads deterministic TRAIN/VALIDATION sample items;
- proves returned x is an exact PRIMARY58 slice of stored standardized x;
- proves targets come from the final epoch of each window;
- proves only the approved item keys are returned;
- proves the corrected Boolean [16,10] mask matches A2-R2;
- proves split='test' is rejected before filesystem access.

No P2 test directory is enumerated or opened.
```

Match line 38 (context starts 32):

```python
from pathlib import Path
from typing import Any

import torch


STAGE = "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT"
COMPLETE = f"{STAGE}_COMPLETE"

EXPECTED_TOTAL_ALIGNED_WINDOWS = 82694
EXPECTED_TOTAL_PAIRS = 489
EXPECTED_TRAIN_PAIRS = 415
EXPECTED_VALIDATION_PAIRS = 74
```

Match line 48 (context starts 42):

```python
EXPECTED_TOTAL_PAIRS = 489
EXPECTED_TRAIN_PAIRS = 415
EXPECTED_VALIDATION_PAIRS = 74

EXPECTED_ITEM_KEYS = {
    "x",
    "physical_port_mask",
    "y_attack",
    "y_attacker_count",
    "y_source",
    "y_transit",
    "y_victim",
    "y_attack_path",
```

Match line 108 (context starts 102):

```python


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def import_loader(path: Path):
    spec = importlib.util.spec_from_file_location(
        "v5_p2_pair_aligned_primary58_dataset",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import loader from {path}")
```

Match line 110 (context starts 104):

```python
def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def import_loader(path: Path):
    spec = importlib.util.spec_from_file_location(
        "v5_p2_pair_aligned_primary58_dataset",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import loader from {path}")

    module = importlib.util.module_from_spec(spec)
```

Match line 114 (context starts 108):

```python
def import_loader(path: Path):
    spec = importlib.util.spec_from_file_location(
        "v5_p2_pair_aligned_primary58_dataset",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import loader from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

```

Match line 182 (context starts 176):

```python
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a0-dir", type=Path, required=True)
    parser.add_argument("--a1-r2-dir", type=Path, required=True)
    parser.add_argument("--a2-r2-dir", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root_input = args.root.expanduser()
    root = root_input.resolve()
    a0_dir = args.a0_dir.expanduser().resolve()
```

Match line 191 (context starts 185):

```python

    root_input = args.root.expanduser()
    root = root_input.resolve()
    a0_dir = args.a0_dir.expanduser().resolve()
    a1_r2_dir = args.a1_r2_dir.expanduser().resolve()
    a2_r2_dir = args.a2_r2_dir.expanduser().resolve()
    loader_path = args.loader_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)
```

Match line 202 (context starts 196):

```python
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    paths = {
        "a0_report": (
            a0_dir
            / "V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT.json"
        ),
        "a0_lock": (
            a0_dir
```

Match line 219 (context starts 213):

```python
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT.json"
        ),
        "a1_r2_lock": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT_LOCK.json"
        ),
        "pair_manifest": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
        ),
        "a2_r2_report": (
            a2_r2_dir
            / "V5_P2_A2_R2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT.json"
```

Match line 235 (context starts 229):

```python
            / "V5_P2_A2_R2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT_LOCK.json"
        ),
        "a2_r2_mask": (
            a2_r2_dir
            / "V5_P2_A2_R2_TOPOLOGY_DERIVED_BOOLEAN_PORT_MASK.pt"
        ),
        "loader": loader_path,
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

```

Match line 280 (context starts 274):

```python
    if a0_lock.get("report_sha256") != sha256_file(paths["a0_report"]):
        failures.append("A0 report SHA mismatch")
    if a1_lock.get("report_sha256") != sha256_file(paths["a1_r2_report"]):
        failures.append("A1-R2 report SHA mismatch")
    if a2_lock.get("report_sha256") != sha256_file(paths["a2_r2_report"]):
        failures.append("A2-R2 report SHA mismatch")
    if a1_lock.get("pair_manifest_sha256") != sha256_file(
        paths["pair_manifest"]
    ):
        failures.append("A1-R2 pair manifest SHA mismatch")
    if a2_lock.get("corrected_port_mask_sha256") != sha256_file(
        paths["a2_r2_mask"]
    ):
```

Match line 281 (context starts 275):

```python
        failures.append("A0 report SHA mismatch")
    if a1_lock.get("report_sha256") != sha256_file(paths["a1_r2_report"]):
        failures.append("A1-R2 report SHA mismatch")
    if a2_lock.get("report_sha256") != sha256_file(paths["a2_r2_report"]):
        failures.append("A2-R2 report SHA mismatch")
    if a1_lock.get("pair_manifest_sha256") != sha256_file(
        paths["pair_manifest"]
    ):
        failures.append("A1-R2 pair manifest SHA mismatch")
    if a2_lock.get("corrected_port_mask_sha256") != sha256_file(
        paths["a2_r2_mask"]
    ):
        failures.append("A2-R2 mask SHA mismatch")
```

Match line 293 (context starts 287):

```python
        failures.append("A2-R2 mask SHA mismatch")

    if a1_lock.get("aligned_windows") != EXPECTED_TOTAL_ALIGNED_WINDOWS:
        failures.append("A1-R2 aligned-window count is not 82,694")
    if a1_lock.get("pair_count") != EXPECTED_TOTAL_PAIRS:
        failures.append("A1-R2 pair count is not 489")
    if a2_lock.get("primary58_count") != 58:
        failures.append("A2-R2 PRIMARY58 count is not 58")
    if a2_lock.get("window") != 32 or a2_lock.get("stride") != 8:
        failures.append("A2-R2 window/stride contract changed")
    if a2_lock.get("second_normalization_forbidden") is not True:
        failures.append("A2-R2 does not forbid second normalization")
    if a2_lock.get("reference_dataset_loader_authorized") is not False:
```

Match line 294 (context starts 288):

```python

    if a1_lock.get("aligned_windows") != EXPECTED_TOTAL_ALIGNED_WINDOWS:
        failures.append("A1-R2 aligned-window count is not 82,694")
    if a1_lock.get("pair_count") != EXPECTED_TOTAL_PAIRS:
        failures.append("A1-R2 pair count is not 489")
    if a2_lock.get("primary58_count") != 58:
        failures.append("A2-R2 PRIMARY58 count is not 58")
    if a2_lock.get("window") != 32 or a2_lock.get("stride") != 8:
        failures.append("A2-R2 window/stride contract changed")
    if a2_lock.get("second_normalization_forbidden") is not True:
        failures.append("A2-R2 does not forbid second normalization")
    if a2_lock.get("reference_dataset_loader_authorized") is not False:
        failures.append("A2-R2 unexpectedly authorizes reference loader")
```

Match line 334 (context starts 328):

```python
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    module = import_loader(loader_path)
    DatasetClass = module.V5P2PairAlignedPrimary58Dataset

    # Must reject test before any test filesystem interaction.
    test_rejected = False
    try:
        DatasetClass(
```

Match line 335 (context starts 329):

```python
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    module = import_loader(loader_path)
    DatasetClass = module.V5P2PairAlignedPrimary58Dataset

    # Must reject test before any test filesystem interaction.
    test_rejected = False
    try:
        DatasetClass(
            root=root,
```

Match line 343 (context starts 337):

```python
    # Must reject test before any test filesystem interaction.
    test_rejected = False
    try:
        DatasetClass(
            root=root,
            split="test",
            pair_manifest=paths["pair_manifest"],
        )
    except ValueError:
        test_rejected = True
    except Exception as exc:
        failures.append(
            "split='test' raised the wrong exception type: "
```

Match line 360 (context starts 354):

```python

    datasets = {}
    for split in ("train", "validation"):
        datasets[split] = DatasetClass(
            root=root,
            split=split,
            pair_manifest=paths["pair_manifest"],
        )

    if datasets["train"].pair_count != EXPECTED_TRAIN_PAIRS:
        failures.append(
            f"train pair_count={datasets['train'].pair_count}, expected 415"
        )
```

Match line 511 (context starts 505):

```python
                    or tuple(x.shape) != (16, 58, 32)
                    or not x.is_contiguous()
                    or not bool(torch.isfinite(x).all().item())
                ):
                    failures.append(f"{context}: x contract failed")

                mask = item.get("physical_port_mask")
                if (
                    not isinstance(mask, torch.Tensor)
                    or mask.dtype != torch.bool
                    or tuple(mask.shape) != (16, 10)
                ):
                    failures.append(
```

Match line 518 (context starts 512):

```python
                if (
                    not isinstance(mask, torch.Tensor)
                    or mask.dtype != torch.bool
                    or tuple(mask.shape) != (16, 10)
                ):
                    failures.append(
                        f"{context}: physical_port_mask contract failed"
                    )
                elif isinstance(stored_mask, torch.Tensor) and not torch.equal(
                    mask,
                    stored_mask,
                ):
                    failures.append(
```

Match line 572 (context starts 566):

```python
                    weights_only=False,
                )
                expected_x = (
                    raw["x"][
                        entry.start:entry.start + 32,
                        :,
                        module.PRIMARY58_INDICES,
                    ]
                    .permute(1, 2, 0)
                    .contiguous()
                )
                if not torch.equal(x, expected_x):
                    failures.append(
```

Match line 579 (context starts 573):

```python
                    ]
                    .permute(1, 2, 0)
                    .contiguous()
                )
                if not torch.equal(x, expected_x):
                    failures.append(
                        f"{context}: x is not the exact stored PRIMARY58 slice"
                    )

                target_key_pairs = (
                    ("y_attack", "y_attack"),
                    ("y_attacker_count", "y_attacker_count"),
                    ("y_source", "y_source"),
```

### Checkpoint Load

Match line 380 (context starts 374):

```python
    if total_dataset_windows != EXPECTED_TOTAL_ALIGNED_WINDOWS:
        failures.append(
            f"loader aligned windows={total_dataset_windows}, "
            f"expected {EXPECTED_TOTAL_ALIGNED_WINDOWS}"
        )

    stored_mask_payload = torch.load(
        paths["a2_r2_mask"],
        map_location="cpu",
        weights_only=False,
    )
    stored_mask = stored_mask_payload.get("mask")
    if (
```

Match line 563 (context starts 557):

```python
                    (torch.uint8, torch.bool),
                    "role_mask",
                    failures,
                    context,
                )

                raw = torch.load(
                    entry.file_path,
                    map_location="cpu",
                    weights_only=False,
                )
                expected_x = (
                    raw["x"][
```

### Stable Identity

Match line 63 (context starts 57):

```python

PROHIBITED_ITEM_KEYS = {
    "edge_index",
    "router_coordinates",
    "epoch_id",
    "case_id",
    "pair_id",
    "run_id",
    "filename",
    "mode",
    "split",
    "category",
    "window_start",
```

Match line 64 (context starts 58):

```python
PROHIBITED_ITEM_KEYS = {
    "edge_index",
    "router_coordinates",
    "epoch_id",
    "case_id",
    "pair_id",
    "run_id",
    "filename",
    "mode",
    "split",
    "category",
    "window_start",
    "window_target",
```

Match line 69 (context starts 63):

```python
    "pair_id",
    "run_id",
    "filename",
    "mode",
    "split",
    "category",
    "window_start",
    "window_target",
    "window_end",
    "native_run_length",
    "common_length",
}

```

Match line 417 (context starts 411):

```python
            if entry.file_path.parent.name != split:
                failures.append(
                    f"{split}: entry points outside split: {entry.file_path}"
                )
            if entry.mode not in {"attack", "control"}:
                failures.append(
                    f"{split}/{entry.pair_key}: invalid mode {entry.mode}"
                )
            if entry.target != entry.start + 31:
                failures.append(
                    f"{split}/{entry.pair_key}: target/start mismatch"
                )
            if entry.start % 8:
```

Match line 421 (context starts 415):

```python
            if entry.mode not in {"attack", "control"}:
                failures.append(
                    f"{split}/{entry.pair_key}: invalid mode {entry.mode}"
                )
            if entry.target != entry.start + 31:
                failures.append(
                    f"{split}/{entry.pair_key}: target/start mismatch"
                )
            if entry.start % 8:
                failures.append(
                    f"{split}/{entry.pair_key}: start is not stride-aligned"
                )
            if entry.target >= entry.common_length:
```

Match line 425 (context starts 419):

```python
            if entry.target != entry.start + 31:
                failures.append(
                    f"{split}/{entry.pair_key}: target/start mismatch"
                )
            if entry.start % 8:
                failures.append(
                    f"{split}/{entry.pair_key}: start is not stride-aligned"
                )
            if entry.target >= entry.common_length:
                failures.append(
                    f"{split}/{entry.pair_key}: target exceeds common length"
                )

```

Match line 429 (context starts 423):

```python
            if entry.start % 8:
                failures.append(
                    f"{split}/{entry.pair_key}: start is not stride-aligned"
                )
            if entry.target >= entry.common_length:
                failures.append(
                    f"{split}/{entry.pair_key}: target exceeds common length"
                )

            pair_start_groups[
                (entry.pair_key, entry.start)
            ].append(entry)
            pair_counts[entry.pair_key] += 1
```

Match line 433 (context starts 427):

```python
            if entry.target >= entry.common_length:
                failures.append(
                    f"{split}/{entry.pair_key}: target exceeds common length"
                )

            pair_start_groups[
                (entry.pair_key, entry.start)
            ].append(entry)
            pair_counts[entry.pair_key] += 1

        for key, group in pair_start_groups.items():
            index_checks["pair_start_group_count"] += 1
            if len(group) != 2:
```

Match line 435 (context starts 429):

```python
                    f"{split}/{entry.pair_key}: target exceeds common length"
                )

            pair_start_groups[
                (entry.pair_key, entry.start)
            ].append(entry)
            pair_counts[entry.pair_key] += 1

        for key, group in pair_start_groups.items():
            index_checks["pair_start_group_count"] += 1
            if len(group) != 2:
                failures.append(
                    f"{split}/{key}: aligned start has {len(group)} members"
```

Match line 473 (context starts 467):

```python
                )
            if control_entry.mode != "control":
                failures.append(
                    f"{split} sample {base+1}: second member is not control"
                )
            if (
                attack_entry.pair_key != control_entry.pair_key
                or attack_entry.start != control_entry.start
                or attack_entry.target != control_entry.target
                or attack_entry.common_length != control_entry.common_length
            ):
                failures.append(
                    f"{split} sample base {base}: pair alignment mismatch"
```

Match line 486 (context starts 480):

```python
                )

            for item_index in (base, base + 1):
                entry = entries[item_index]
                item = dataset[item_index]
                context = (
                    f"{split}/{entry.pair_key}/"
                    f"{entry.mode}/start={entry.start}"
                )

                if set(item) != EXPECTED_ITEM_KEYS:
                    failures.append(
                        f"{context}: item keys={sorted(item)}, "
```

Match line 603 (context starts 597):

```python
                            f"{context}: {item_key} is not final-epoch target"
                        )

                sample_records.append(
                    {
                        "split": split,
                        "pair_key": entry.pair_key,
                        "mode": entry.mode,
                        "start": entry.start,
                        "target": entry.target,
                        "common_length": entry.common_length,
                        "x_shape": list(x.shape),
                        "x_dtype": str(x.dtype),
```

### Expected Counts

Match line 866 (context starts 860):

```python
        "contract_file_sha256": sha256_file(contract_path),
        "contract_sha256": contract["contract_sha256"],
        "loader_sha256": sha256_file(loader_path),
        "train_pairs": datasets["train"].pair_count,
        "validation_pairs": datasets["validation"].pair_count,
        "train_items": len(datasets["train"]),
        "validation_items": len(datasets["validation"]),
        "total_aligned_items": total_dataset_windows,
        "item_keys": sorted(EXPECTED_ITEM_KEYS),
        "x_shape": [16, 58, 32],
        "mask_shape": [16, 10],
        "test_constructor_rejected": test_rejected,
        "test_directory_enumerated": False,
```

Match line 892 (context starts 886):

```python
        "FREEZE_PAIR_ALIGNED_PRIMARY58_LOADER_AND_AUTHORIZE_A4"
    )
    print("loader_class: V5P2PairAlignedPrimary58Dataset")
    print("train_pairs:", datasets["train"].pair_count)
    print("validation_pairs:", datasets["validation"].pair_count)
    print("train_items:", len(datasets["train"]))
    print("validation_items:", len(datasets["validation"]))
    print("total_aligned_items:", total_dataset_windows)
    print("x_shape: [16, 58, 32]")
    print("x_dtype: float32")
    print("physical_port_mask_shape: [16, 10]")
    print("physical_port_mask_dtype: bool")
    print("second_normalization_performed: false")
```

## `scripts/v5/p2/freeze_v5_p2_a1_r2_pair_aligned_window_contract.py`

- SHA-256: `ba4ca3305661d7f585d4b696945fb0b2dcccbc2ebdc02b3990356e7b25c73266`
- Constants:

```json
{
  "EXPECTED": {
    "aligned_windows": 82694,
    "native_windows": 82708,
    "pair_count": 489,
    "removed_windows": 14,
    "train_pairs": 415,
    "unequal_length_pairs": 77,
    "unequal_native_window_pairs": 14,
    "validation_pairs": 74
  },
  "STAGE": "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT",
  "STRIDE": 8,
  "WINDOW": 32
}
```

- Classes:

```json
[]
```

### Paths Dictionary

Match line 139 (context starts 133):

```python
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    paths = {
        "a0_report": (
            a0_dir
            / "V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT.json"
        ),
        "a0_lock": (
            a0_dir
```

Match line 518 (context starts 512):

```python
        "reason": (
            "Remove ATTACK/CONTROL shortcut leakage through unequal "
            "run lengths, unequal window counts, and terminal-window "
            "position while discarding only 14 of 82,708 native "
            "non-test windows."
        ),
        "pair_manifest_file": str(manifest_path),
        "pair_manifest_file_sha256": sha256_file(manifest_path),
        "test_tensor_contents_accessed": False,
    }
    contract["contract_sha256"] = canonical_sha256(contract)

    contract_path = (
```

Match line 519 (context starts 513):

```python
            "Remove ATTACK/CONTROL shortcut leakage through unequal "
            "run lengths, unequal window counts, and terminal-window "
            "position while discarding only 14 of 82,708 native "
            "non-test windows."
        ),
        "pair_manifest_file": str(manifest_path),
        "pair_manifest_file_sha256": sha256_file(manifest_path),
        "test_tensor_contents_accessed": False,
    }
    contract["contract_sha256"] = canonical_sha256(contract)

    contract_path = (
        output_dir
```

Match line 661 (context starts 655):

```python
        "decision": (
            "FREEZE_PAIR_ALIGNED_WINDOW_CONTRACT_AND_AUTHORIZE_A2"
        ),
        "report_sha256": sha256_file(report_path),
        "contract_file_sha256": sha256_file(contract_path),
        "contract_sha256": contract["contract_sha256"],
        "pair_manifest_sha256": sha256_file(manifest_path),
        "window": WINDOW,
        "stride": STRIDE,
        **actuals,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "next_stage": (
```

### Stable Identity

Match line 105 (context starts 99):

```python
def as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid integer field {key!r} in pair row "
            f"{row.get('split')}/{row.get('pair_key')}"
        ) from exc


def window_count(length: int) -> int:
    if length < WINDOW:
        return 0
```

Match line 257 (context starts 251):

```python
        for failure in failures:
            print("FAIL:", failure)
        return 1

    required_columns = {
        "split",
        "pair_key",
        "category",
        "pair_id",
        "attack_length",
        "control_length",
        "attack_native_window_count",
        "control_native_window_count",
```

Match line 259 (context starts 253):

```python
        return 1

    required_columns = {
        "split",
        "pair_key",
        "category",
        "pair_id",
        "attack_length",
        "control_length",
        "attack_native_window_count",
        "control_native_window_count",
        "common_aligned_length",
        "common_aligned_window_count_per_member",
```

Match line 276 (context starts 270):

```python
    if missing_columns:
        failures.append(
            f"A1-R1 pair CSV missing columns: {missing_columns}"
        )

    seen: set[tuple[str, str]] = set()
    manifest_rows: list[dict[str, Any]] = []
    split_counts = Counter()
    unequal_length_count = 0
    unequal_window_count = 0
    native_total = 0
    aligned_total = 0
    removed_total = 0
```

Match line 286 (context starts 280):

```python
    native_total = 0
    aligned_total = 0
    removed_total = 0

    for row in rows:
        split = row.get("split", "")
        pair_key = row.get("pair_key", "")
        identity = (split, pair_key)

        if split not in {"train", "validation"}:
            failures.append(
                f"unexpected split {split!r} for pair {pair_key!r}"
            )
```

Match line 287 (context starts 281):

```python
    aligned_total = 0
    removed_total = 0

    for row in rows:
        split = row.get("split", "")
        pair_key = row.get("pair_key", "")
        identity = (split, pair_key)

        if split not in {"train", "validation"}:
            failures.append(
                f"unexpected split {split!r} for pair {pair_key!r}"
            )
            continue
```

Match line 291 (context starts 285):

```python
        split = row.get("split", "")
        pair_key = row.get("pair_key", "")
        identity = (split, pair_key)

        if split not in {"train", "validation"}:
            failures.append(
                f"unexpected split {split!r} for pair {pair_key!r}"
            )
            continue
        if identity in seen:
            failures.append(
                f"duplicate pair row: {split}/{pair_key}"
            )
```

Match line 296 (context starts 290):

```python
            failures.append(
                f"unexpected split {split!r} for pair {pair_key!r}"
            )
            continue
        if identity in seen:
            failures.append(
                f"duplicate pair row: {split}/{pair_key}"
            )
            continue
        seen.add(identity)
        split_counts[split] += 1

        attack_length = as_int(row, "attack_length")
```

Match line 338 (context starts 332):

```python
        expected_common_windows = window_count(
            expected_common_length
        )

        if common_length != expected_common_length:
            failures.append(
                f"{split}/{pair_key}: common_length={common_length}, "
                f"expected {expected_common_length}"
            )
        if attack_native != expected_attack_native:
            failures.append(
                f"{split}/{pair_key}: attack native windows mismatch"
            )
```

Match line 343 (context starts 337):

```python
            failures.append(
                f"{split}/{pair_key}: common_length={common_length}, "
                f"expected {expected_common_length}"
            )
        if attack_native != expected_attack_native:
            failures.append(
                f"{split}/{pair_key}: attack native windows mismatch"
            )
        if control_native != expected_control_native:
            failures.append(
                f"{split}/{pair_key}: control native windows mismatch"
            )
        if common_windows != expected_common_windows:
```

Match line 347 (context starts 341):

```python
        if attack_native != expected_attack_native:
            failures.append(
                f"{split}/{pair_key}: attack native windows mismatch"
            )
        if control_native != expected_control_native:
            failures.append(
                f"{split}/{pair_key}: control native windows mismatch"
            )
        if common_windows != expected_common_windows:
            failures.append(
                f"{split}/{pair_key}: aligned window count mismatch"
            )
        if attack_removed != attack_native - common_windows:
```

Match line 351 (context starts 345):

```python
        if control_native != expected_control_native:
            failures.append(
                f"{split}/{pair_key}: control native windows mismatch"
            )
        if common_windows != expected_common_windows:
            failures.append(
                f"{split}/{pair_key}: aligned window count mismatch"
            )
        if attack_removed != attack_native - common_windows:
            failures.append(
                f"{split}/{pair_key}: attack removed-window mismatch"
            )
        if control_removed != control_native - common_windows:
```

Match line 355 (context starts 349):

```python
        if common_windows != expected_common_windows:
            failures.append(
                f"{split}/{pair_key}: aligned window count mismatch"
            )
        if attack_removed != attack_native - common_windows:
            failures.append(
                f"{split}/{pair_key}: attack removed-window mismatch"
            )
        if control_removed != control_native - common_windows:
            failures.append(
                f"{split}/{pair_key}: control removed-window mismatch"
            )
        if attack_removed < 0 or control_removed < 0:
```

Match line 359 (context starts 353):

```python
        if attack_removed != attack_native - common_windows:
            failures.append(
                f"{split}/{pair_key}: attack removed-window mismatch"
            )
        if control_removed != control_native - common_windows:
            failures.append(
                f"{split}/{pair_key}: control removed-window mismatch"
            )
        if attack_removed < 0 or control_removed < 0:
            failures.append(
                f"{split}/{pair_key}: negative removed-window count"
            )

```

Match line 363 (context starts 357):

```python
        if control_removed != control_native - common_windows:
            failures.append(
                f"{split}/{pair_key}: control removed-window mismatch"
            )
        if attack_removed < 0 or control_removed < 0:
            failures.append(
                f"{split}/{pair_key}: negative removed-window count"
            )

        if attack_length != control_length:
            unequal_length_count += 1
        if attack_native != control_native:
            unequal_window_count += 1
```

Match line 386 (context starts 380):

```python
        last_end_exclusive = (
            last_start + WINDOW
            if last_start is not None
            else None
        )

        manifest_rows.append(
            {
                "split": split,
                "pair_key": pair_key,
                "category": row.get("category"),
                "pair_id": row.get("pair_id"),
                "attack_length": attack_length,
```

Match line 389 (context starts 383):

```python
            else None
        )

        manifest_rows.append(
            {
                "split": split,
                "pair_key": pair_key,
                "category": row.get("category"),
                "pair_id": row.get("pair_id"),
                "attack_length": attack_length,
                "control_length": control_length,
                "common_length": common_length,
                "window": WINDOW,
```

Match line 391 (context starts 385):

```python

        manifest_rows.append(
            {
                "split": split,
                "pair_key": pair_key,
                "category": row.get("category"),
                "pair_id": row.get("pair_id"),
                "attack_length": attack_length,
                "control_length": control_length,
                "common_length": common_length,
                "window": WINDOW,
                "stride": STRIDE,
                "window_count_per_member": common_windows,
```

Match line 398 (context starts 392):

```python
                "attack_length": attack_length,
                "control_length": control_length,
                "common_length": common_length,
                "window": WINDOW,
                "stride": STRIDE,
                "window_count_per_member": common_windows,
                "first_window_start": 0,
                "last_window_start": last_start,
                "last_window_end_exclusive": last_end_exclusive,
                "attack_native_window_count": attack_native,
                "control_native_window_count": control_native,
                "attack_windows_removed": attack_removed,
                "control_windows_removed": control_removed,
```

Match line 399 (context starts 393):

```python
                "control_length": control_length,
                "common_length": common_length,
                "window": WINDOW,
                "stride": STRIDE,
                "window_count_per_member": common_windows,
                "first_window_start": 0,
                "last_window_start": last_start,
                "last_window_end_exclusive": last_end_exclusive,
                "attack_native_window_count": attack_native,
                "control_native_window_count": control_native,
                "attack_windows_removed": attack_removed,
                "control_windows_removed": control_removed,
                "identical_ordered_window_starts": True,
```

Match line 405 (context starts 399):

```python
                "last_window_start": last_start,
                "last_window_end_exclusive": last_end_exclusive,
                "attack_native_window_count": attack_native,
                "control_native_window_count": control_native,
                "attack_windows_removed": attack_removed,
                "control_windows_removed": control_removed,
                "identical_ordered_window_starts": True,
            }
        )

    actuals = {
        "pair_count": len(manifest_rows),
        "train_pairs": split_counts["train"],
```

Match line 410 (context starts 404):

```python
                "control_windows_removed": control_removed,
                "identical_ordered_window_starts": True,
            }
        )

    actuals = {
        "pair_count": len(manifest_rows),
        "train_pairs": split_counts["train"],
        "validation_pairs": split_counts["validation"],
        "native_windows": native_total,
        "aligned_windows": aligned_total,
        "removed_windows": removed_total,
        "unequal_length_pairs": unequal_length_count,
```

Match line 430 (context starts 424):

```python

    if native_total - aligned_total != removed_total:
        failures.append(
            "native-aligned total does not equal removed total"
        )

    manifest_rows.sort(
        key=lambda item: (item["split"], item["pair_key"])
    )
    manifest_path = (
        output_dir
        / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
    )
```

Match line 431 (context starts 425):

```python
    if native_total - aligned_total != removed_total:
        failures.append(
            "native-aligned total does not equal removed total"
        )

    manifest_rows.sort(
        key=lambda item: (item["split"], item["pair_key"])
    )
    manifest_path = (
        output_dir
        / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
    )
    write_csv(manifest_path, manifest_rows)
```

## `scripts/v5/p2/preflight_v5_p2_task_d_full_multitask.py`

- SHA-256: `b7ef6c9a522ec6fd122b16fc17f220cc413f6e7e24333778d3a6885dbc347a42`
- Constants:

```json
{
  "EXPECTED_B1_PROTOCOL_SHA": "0817ae7812f91e3c75260589acaf52bd8a74b07b2114a451b4773578efde4b60",
  "EXPECTED_PARAMS": {
    "conv1d": 43273,
    "graphconv": 59785
  },
  "SEEDS": [
    107,
    117,
    127,
    137,
    147
  ],
  "STAGE": "V5_P2_TASK_D_FULL_MULTITASK_PREFLIGHT"
}
```

- Classes:

```json
[]
```

### Paths Dictionary

Match line 97 (context starts 91):

```python
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        print(f"STOP: output already exists: {out}")
        return 2
    out.mkdir(parents=True)

    paths = {
        "promotion_report": repo / "reports/v5/p2_g123_graph_operator_promotion/V5_P2_G123_GRAPH_OPERATOR_PROMOTION.json",
        "promotion_lock": repo / "reports/v5/p2_g123_graph_operator_promotion/V5_P2_G123_GRAPH_OPERATOR_PROMOTION_LOCK.json",
        "promotion_marker": repo / "reports/v5/p2_g123_graph_operator_promotion/V5_P2_G123_GRAPH_OPERATOR_PROMOTION_COMPLETE",
        "b1_protocol": repo / "reports/v5/p2_b1_training_protocol_lock/V5_P2_B1_TRAINING_PROTOCOL.json",
        "b1_report": repo / "reports/v5/p2_b1_training_protocol_lock/V5_P2_B1_TRAINING_PROTOCOL_LOCK.json",
        "b1_lock": repo / "reports/v5/p2_b1_training_protocol_lock/V5_P2_B1_TRAINING_PROTOCOL_LOCK_LOCK.json",
```

Match line 107 (context starts 101):

```python
        "b1_protocol": repo / "reports/v5/p2_b1_training_protocol_lock/V5_P2_B1_TRAINING_PROTOCOL.json",
        "b1_report": repo / "reports/v5/p2_b1_training_protocol_lock/V5_P2_B1_TRAINING_PROTOCOL_LOCK.json",
        "b1_lock": repo / "reports/v5/p2_b1_training_protocol_lock/V5_P2_B1_TRAINING_PROTOCOL_LOCK_LOCK.json",
        "g0_seed_policy": repo / "reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol/V5_P2_G0_TRAINING_SELECTION_AND_SEED_POLICY.json",
        "g0_operator_contracts": repo / "reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol/V5_P2_G0_GRAPH_OPERATOR_CONTRACTS.json",
        "test_policy": repo / "reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol/V5_P2_G0_P2_TEST_ACCESS_POLICY.json",
        "edge": repo / "reports/v5/p2_g1a_r2a_canonical_static_topology_contract/V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
        "b3_model": repo / "src/models/v5_p2_b3_conv1d_only_count4.py",
        "task_d_model": repo / "src/models/v5_p2_task_d_full_multitask_count4.py",
        "b2_train": repo / "scripts/v5/p2/train_v5_p2_b2_single_seed.py",
        "loader": repo / "src/data/v5_p2_pair_aligned_primary58_dataset.py",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
```

Match line 111 (context starts 105):

```python
        "g0_operator_contracts": repo / "reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol/V5_P2_G0_GRAPH_OPERATOR_CONTRACTS.json",
        "test_policy": repo / "reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol/V5_P2_G0_P2_TEST_ACCESS_POLICY.json",
        "edge": repo / "reports/v5/p2_g1a_r2a_canonical_static_topology_contract/V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
        "b3_model": repo / "src/models/v5_p2_b3_conv1d_only_count4.py",
        "task_d_model": repo / "src/models/v5_p2_task_d_full_multitask_count4.py",
        "b2_train": repo / "scripts/v5/p2/train_v5_p2_b2_single_seed.py",
        "loader": repo / "src/data/v5_p2_pair_aligned_primary58_dataset.py",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    failures: list[str] = []
    if missing:
        failures.extend(f"missing: {p}" for p in missing)

```

Match line 139 (context starts 133):

```python
        b1_report = json.loads(paths["b1_report"].read_text())
        b1_lock = json.loads(paths["b1_lock"].read_text())
        if protocol["protocol_sha256"] != EXPECTED_B1_PROTOCOL_SHA:
            raise RuntimeError("B1 protocol SHA changed")
        if b1_lock["protocol_file_sha256"] != sha256_file(paths["b1_protocol"]):
            raise RuntimeError("B1 protocol file SHA mismatch")
        if b1_lock["model_sha256"] != sha256_file(paths["b3_model"]):
            raise RuntimeError("B3 model SHA mismatch")
        if b1_lock["loader_sha256"] != sha256_file(paths["loader"]):
            raise RuntimeError("loader SHA mismatch")

        seed_policy = json.loads(paths["g0_seed_policy"].read_text())
        if seed_policy["training_base"]["finalist_stability_seeds"] != SEEDS:
```

Match line 141 (context starts 135):

```python
        if protocol["protocol_sha256"] != EXPECTED_B1_PROTOCOL_SHA:
            raise RuntimeError("B1 protocol SHA changed")
        if b1_lock["protocol_file_sha256"] != sha256_file(paths["b1_protocol"]):
            raise RuntimeError("B1 protocol file SHA mismatch")
        if b1_lock["model_sha256"] != sha256_file(paths["b3_model"]):
            raise RuntimeError("B3 model SHA mismatch")
        if b1_lock["loader_sha256"] != sha256_file(paths["loader"]):
            raise RuntimeError("loader SHA mismatch")

        seed_policy = json.loads(paths["g0_seed_policy"].read_text())
        if seed_policy["training_base"]["finalist_stability_seeds"] != SEEDS:
            raise RuntimeError("finalist seed set changed")
        operator_contracts = json.loads(paths["g0_operator_contracts"].read_text())
```

Match line 183 (context starts 177):

```python

        batch_size = 4
        x = torch.randn(batch_size, 16, 58, 32, device=device)
        mask = torch.ones(batch_size, 16, 10, dtype=torch.bool, device=device)
        batch = {
            "x": x,
            "physical_port_mask": mask,
            "y_attack": torch.tensor([0, 1, 1, 1], dtype=torch.float32, device=device),
            "y_attacker_count": torch.tensor([0, 1, 2, 4], dtype=torch.int64, device=device),
            "y_source": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_transit": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_victim": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_attack_path": torch.randint(0, 2, (batch_size, 16), device=device),
```

### Getitem Return

Match line 181 (context starts 175):

```python
        if sha256_state_dict(duplicate_b3.state_dict()) != conv_hash:
            raise RuntimeError("B3 deterministic initialization failed")

        batch_size = 4
        x = torch.randn(batch_size, 16, 58, 32, device=device)
        mask = torch.ones(batch_size, 16, 10, dtype=torch.bool, device=device)
        batch = {
            "x": x,
            "physical_port_mask": mask,
            "y_attack": torch.tensor([0, 1, 1, 1], dtype=torch.float32, device=device),
            "y_attacker_count": torch.tensor([0, 1, 2, 4], dtype=torch.int64, device=device),
            "y_source": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_transit": torch.randint(0, 2, (batch_size, 16), device=device),
```

### Model Class

Match line 166 (context starts 160):

```python

        # Matched-seed common initialization proof.
        set_seed(107)
        conv_reference = b3_mod.P2B3Conv1DOnlyCount4()
        set_seed(107)
        graph_reference = b3_mod.P2B3Conv1DOnlyCount4()
        graph_model = td_mod.P2TaskDGraphConvCount4(graph_reference, edge)
        conv_hash = sha256_state_dict(conv_reference.state_dict())
        graph_common_hash = sha256_state_dict(td_mod.common_state_dict(graph_model))
        if conv_hash != graph_common_hash:
            raise RuntimeError("matched-seed common initialization is not identical")

        # Independent duplicate B3 instantiation must be exact under the seed.
```

### Batch Transfer

Match line 200 (context starts 194):

```python
            "transit": torch.tensor(15.7433147902343, device=device),
            "victim": torch.tensor(20.0, device=device),
            "path": torch.tensor(7.375342240922689, device=device),
        }

        for candidate, model in (
            ("conv1d", conv_reference.to(device)),
            ("graphconv", graph_model.to(device)),
        ):
            params = sum(p.numel() for p in model.parameters())
            if params != EXPECTED_PARAMS[candidate]:
                raise RuntimeError(f"{candidate} params={params}, expected {EXPECTED_PARAMS[candidate]}")
            model.train()
```

Match line 201 (context starts 195):

```python
            "victim": torch.tensor(20.0, device=device),
            "path": torch.tensor(7.375342240922689, device=device),
        }

        for candidate, model in (
            ("conv1d", conv_reference.to(device)),
            ("graphconv", graph_model.to(device)),
        ):
            params = sum(p.numel() for p in model.parameters())
            if params != EXPECTED_PARAMS[candidate]:
                raise RuntimeError(f"{candidate} params={params}, expected {EXPECTED_PARAMS[candidate]}")
            model.train()
            outputs = model(x, mask)
```

### Expected Counts

Match line 178 (context starts 172):

```python
        # Independent duplicate B3 instantiation must be exact under the seed.
        set_seed(107)
        duplicate_b3 = b3_mod.P2B3Conv1DOnlyCount4()
        if sha256_state_dict(duplicate_b3.state_dict()) != conv_hash:
            raise RuntimeError("B3 deterministic initialization failed")

        batch_size = 4
        x = torch.randn(batch_size, 16, 58, 32, device=device)
        mask = torch.ones(batch_size, 16, 10, dtype=torch.bool, device=device)
        batch = {
            "x": x,
            "physical_port_mask": mask,
            "y_attack": torch.tensor([0, 1, 1, 1], dtype=torch.float32, device=device),
```

Match line 179 (context starts 173):

```python
        set_seed(107)
        duplicate_b3 = b3_mod.P2B3Conv1DOnlyCount4()
        if sha256_state_dict(duplicate_b3.state_dict()) != conv_hash:
            raise RuntimeError("B3 deterministic initialization failed")

        batch_size = 4
        x = torch.randn(batch_size, 16, 58, 32, device=device)
        mask = torch.ones(batch_size, 16, 10, dtype=torch.bool, device=device)
        batch = {
            "x": x,
            "physical_port_mask": mask,
            "y_attack": torch.tensor([0, 1, 1, 1], dtype=torch.float32, device=device),
            "y_attacker_count": torch.tensor([0, 1, 2, 4], dtype=torch.int64, device=device),
```

Match line 180 (context starts 174):

```python
        duplicate_b3 = b3_mod.P2B3Conv1DOnlyCount4()
        if sha256_state_dict(duplicate_b3.state_dict()) != conv_hash:
            raise RuntimeError("B3 deterministic initialization failed")

        batch_size = 4
        x = torch.randn(batch_size, 16, 58, 32, device=device)
        mask = torch.ones(batch_size, 16, 10, dtype=torch.bool, device=device)
        batch = {
            "x": x,
            "physical_port_mask": mask,
            "y_attack": torch.tensor([0, 1, 1, 1], dtype=torch.float32, device=device),
            "y_attacker_count": torch.tensor([0, 1, 2, 4], dtype=torch.int64, device=device),
            "y_source": torch.randint(0, 2, (batch_size, 16), device=device),
```

Match line 186 (context starts 180):

```python
        mask = torch.ones(batch_size, 16, 10, dtype=torch.bool, device=device)
        batch = {
            "x": x,
            "physical_port_mask": mask,
            "y_attack": torch.tensor([0, 1, 1, 1], dtype=torch.float32, device=device),
            "y_attacker_count": torch.tensor([0, 1, 2, 4], dtype=torch.int64, device=device),
            "y_source": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_transit": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_victim": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_attack_path": torch.randint(0, 2, (batch_size, 16), device=device),
        }
        graph_pos_weight = torch.tensor(2.45339108179939, device=device)
        role_weights = {
```

Match line 187 (context starts 181):

```python
        batch = {
            "x": x,
            "physical_port_mask": mask,
            "y_attack": torch.tensor([0, 1, 1, 1], dtype=torch.float32, device=device),
            "y_attacker_count": torch.tensor([0, 1, 2, 4], dtype=torch.int64, device=device),
            "y_source": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_transit": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_victim": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_attack_path": torch.randint(0, 2, (batch_size, 16), device=device),
        }
        graph_pos_weight = torch.tensor(2.45339108179939, device=device)
        role_weights = {
            "source": torch.tensor(20.0, device=device),
```

Match line 188 (context starts 182):

```python
            "x": x,
            "physical_port_mask": mask,
            "y_attack": torch.tensor([0, 1, 1, 1], dtype=torch.float32, device=device),
            "y_attacker_count": torch.tensor([0, 1, 2, 4], dtype=torch.int64, device=device),
            "y_source": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_transit": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_victim": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_attack_path": torch.randint(0, 2, (batch_size, 16), device=device),
        }
        graph_pos_weight = torch.tensor(2.45339108179939, device=device)
        role_weights = {
            "source": torch.tensor(20.0, device=device),
            "transit": torch.tensor(15.7433147902343, device=device),
```

Match line 189 (context starts 183):

```python
            "physical_port_mask": mask,
            "y_attack": torch.tensor([0, 1, 1, 1], dtype=torch.float32, device=device),
            "y_attacker_count": torch.tensor([0, 1, 2, 4], dtype=torch.int64, device=device),
            "y_source": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_transit": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_victim": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_attack_path": torch.randint(0, 2, (batch_size, 16), device=device),
        }
        graph_pos_weight = torch.tensor(2.45339108179939, device=device)
        role_weights = {
            "source": torch.tensor(20.0, device=device),
            "transit": torch.tensor(15.7433147902343, device=device),
            "victim": torch.tensor(20.0, device=device),
```

## `scripts/v5/p2/run_v5_p2_a4_loader_integration_tiny_overfit.py`

- SHA-256: `990bfefeaaa7a87607a13c99cd273734c941c6bbb9403fdc9c31deca4b154d4a`
- Constants:

```json
{
  "LEARNING_RATE": 0.01,
  "LOSS_WEIGHTS": {
    "attack": 1.0,
    "count": 0.5,
    "path": 0.5,
    "source": 1.0,
    "transit": 0.5,
    "victim": 0.75
  },
  "MAX_STEPS": 3000,
  "REQUIRED_STABLE_STEPS": 20,
  "SEED": 2404,
  "STAGE": "V5_P2_A4_LOADER_INTEGRATION_AND_TINY_OVERFIT"
}
```

- Classes:

```json
[]
```

### Paths Dictionary

Match line 237 (context starts 231):

```python
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a1-r2-dir", type=Path, required=True)
    parser.add_argument("--a2-r2-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root_input = args.root.expanduser()
    root = root_input.resolve()
```

Match line 238 (context starts 232):

```python
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a1-r2-dir", type=Path, required=True)
    parser.add_argument("--a2-r2-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root_input = args.root.expanduser()
    root = root_input.resolve()
    a1_r2_dir = args.a1_r2_dir.expanduser().resolve()
```

Match line 247 (context starts 241):

```python

    root_input = args.root.expanduser()
    root = root_input.resolve()
    a1_r2_dir = args.a1_r2_dir.expanduser().resolve()
    a2_r2_dir = args.a2_r2_dir.expanduser().resolve()
    a3_dir = args.a3_dir.expanduser().resolve()
    loader_path = args.loader_path.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
        return 2
```

Match line 248 (context starts 242):

```python
    root_input = args.root.expanduser()
    root = root_input.resolve()
    a1_r2_dir = args.a1_r2_dir.expanduser().resolve()
    a2_r2_dir = args.a2_r2_dir.expanduser().resolve()
    a3_dir = args.a3_dir.expanduser().resolve()
    loader_path = args.loader_path.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)
```

Match line 259 (context starts 253):

```python
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    paths = {
        "a1_r2_report": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT.json"
        ),
        "a1_r2_lock": (
            a1_r2_dir
```

Match line 268 (context starts 262):

```python
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT.json"
        ),
        "a1_r2_lock": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT_LOCK.json"
        ),
        "pair_manifest": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
        ),
        "a2_r2_report": (
            a2_r2_dir
            / "V5_P2_A2_R2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT.json"
```

Match line 282 (context starts 276):

```python
        "a2_r2_lock": (
            a2_r2_dir
            / "V5_P2_A2_R2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT_LOCK.json"
        ),
        "a3_report": (
            a3_dir
            / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT.json"
        ),
        "a3_lock": (
            a3_dir
            / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT_LOCK.json"
        ),
        "loader": loader_path,
```

Match line 286 (context starts 280):

```python
        "a3_report": (
            a3_dir
            / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT.json"
        ),
        "a3_lock": (
            a3_dir
            / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT_LOCK.json"
        ),
        "loader": loader_path,
        "model": model_path,
    }

    for name, path in paths.items():
```

Match line 288 (context starts 282):

```python
            / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT.json"
        ),
        "a3_lock": (
            a3_dir
            / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT_LOCK.json"
        ),
        "loader": loader_path,
        "model": model_path,
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")
```

Match line 289 (context starts 283):

```python
        ),
        "a3_lock": (
            a3_dir
            / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT_LOCK.json"
        ),
        "loader": loader_path,
        "model": model_path,
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

```

Match line 342 (context starts 336):

```python
    if a1_lock.get("report_sha256") != sha256_file(paths["a1_r2_report"]):
        failures.append("A1-R2 report SHA mismatch")
    if a2_lock.get("report_sha256") != sha256_file(paths["a2_r2_report"]):
        failures.append("A2-R2 report SHA mismatch")
    if a3_lock.get("report_sha256") != sha256_file(paths["a3_report"]):
        failures.append("A3 report SHA mismatch")
    if a3_lock.get("loader_sha256") != sha256_file(paths["loader"]):
        failures.append("A3 loader SHA mismatch")
    if a3_lock.get("total_aligned_items") != 82694:
        failures.append("A3 total aligned item count changed")
    if a3_lock.get("x_shape") != [16, 58, 32]:
        failures.append("A3 x shape changed")
    if a3_lock.get("mask_shape") != [16, 10]:
```

Match line 380 (context starts 374):

```python
            print("FAIL:", failure)
        return 1

    set_seed(SEED)

    loader_module = import_module(
        loader_path,
        "v5_p2_pair_aligned_primary58_dataset_a4",
    )
    model_module = import_module(
        model_path,
        "v5_frozen_b3_conv1d_only_a4",
    )
```

Match line 381 (context starts 375):

```python
        return 1

    set_seed(SEED)

    loader_module = import_module(
        loader_path,
        "v5_p2_pair_aligned_primary58_dataset_a4",
    )
    model_module = import_module(
        model_path,
        "v5_frozen_b3_conv1d_only_a4",
    )

```

Match line 384 (context starts 378):

```python

    loader_module = import_module(
        loader_path,
        "v5_p2_pair_aligned_primary58_dataset_a4",
    )
    model_module = import_module(
        model_path,
        "v5_frozen_b3_conv1d_only_a4",
    )

    DatasetClass = loader_module.V5P2PairAlignedPrimary58Dataset
    ModelClass = model_module.FrozenB3Conv1DOnly

```

Match line 388 (context starts 382):

```python
    )
    model_module = import_module(
        model_path,
        "v5_frozen_b3_conv1d_only_a4",
    )

    DatasetClass = loader_module.V5P2PairAlignedPrimary58Dataset
    ModelClass = model_module.FrozenB3Conv1DOnly

    train_dataset = DatasetClass(
        root=root,
        split="train",
        pair_manifest=paths["pair_manifest"],
```

Match line 394 (context starts 388):

```python
    DatasetClass = loader_module.V5P2PairAlignedPrimary58Dataset
    ModelClass = model_module.FrozenB3Conv1DOnly

    train_dataset = DatasetClass(
        root=root,
        split="train",
        pair_manifest=paths["pair_manifest"],
    )

    # Locate the final aligned ATTACK/CONTROL start for each pair.
    last_pair_base: dict[str, int] = {}
    for base in range(0, len(train_dataset._index), 2):
        attack_entry = train_dataset._index[base]
```

Match line 549 (context starts 543):

```python
        drop_last=False,
    )
    batch = next(iter(batch_loader))

    expected_batch_keys = {
        "x",
        "physical_port_mask",
        "y_attack",
        "y_attacker_count",
        "y_source",
        "y_transit",
        "y_victim",
        "y_attack_path",
```

Match line 571 (context starts 565):

```python
            f"collated x shape={tuple(batch['x'].shape)}"
        )
    if batch["x"].dtype != torch.float32:
        failures.append(
            f"collated x dtype={batch['x'].dtype}"
        )
    if tuple(batch["physical_port_mask"].shape) != (12, 16, 10):
        failures.append(
            "collated physical-port-mask shape mismatch"
        )
    if batch["physical_port_mask"].dtype != torch.bool:
        failures.append(
            "collated physical-port-mask dtype mismatch"
```

Match line 575 (context starts 569):

```python
            f"collated x dtype={batch['x'].dtype}"
        )
    if tuple(batch["physical_port_mask"].shape) != (12, 16, 10):
        failures.append(
            "collated physical-port-mask shape mismatch"
        )
    if batch["physical_port_mask"].dtype != torch.bool:
        failures.append(
            "collated physical-port-mask dtype mismatch"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
```

Match line 674 (context starts 668):

```python
    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            batch["x"],
            batch["physical_port_mask"],
        )

        attack_loss = F.binary_cross_entropy_with_logits(
            outputs["attack_logits"],
            batch["y_attack"].float(),
            pos_weight=graph_pos_weight,
```

Match line 755 (context starts 749):

```python
        optimizer.step()

        model.eval()
        with torch.no_grad():
            current_outputs = model(
                batch["x"],
                batch["physical_port_mask"],
            )
            metrics = evaluate(
                current_outputs,
                batch,
                count_targets,
                active_mask,
```

Match line 929 (context starts 923):

```python
            "selected_dataset_indices": selected_indices,
            "positive_raw_attacker_counts": positive_raw_counts,
            "count_class_mapping": {
                str(key): value
                for key, value in count_class_mapping.items()
            },
            "model_inputs": ["x", "physical_port_mask"],
            "targets": [
                "y_attack",
                "y_attacker_count",
                "y_source",
                "y_transit",
                "y_victim",
```

Match line 968 (context starts 962):

```python
            "trainable_parameter_count": (
                trainable_parameter_count
            ),
            "graph_message_passing": False,
            "edge_index_model_input": False,
            "x_shape": [12, 16, 58, 32],
            "physical_port_mask_shape": [12, 16, 10],
        },
        "tiny_subset": {
            "item_count": len(selected_indices),
            "active_attack_count": int(active_mask.sum().item()),
            "control_count": int((~active_mask).sum().item()),
            "selected_pairs": selected_pairs,
```

Match line 1031 (context starts 1025):

```python
                paths["a2_r2_lock"]
            ),
            "a3_report_sha256": sha256_file(
                paths["a3_report"]
            ),
            "a3_lock_sha256": sha256_file(paths["a3_lock"]),
            "loader_sha256": sha256_file(paths["loader"]),
            "model_sha256": sha256_file(paths["model"]),
        },
        "security_boundary": {
            "training_performed": True,
            "training_scope": "fixed 12-item train-only audit batch",
            "final_experiment_training_performed": False,
```

### Dataloader Constructor

Match line 538 (context starts 532):

```python
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    subset = Subset(train_dataset, selected_indices)
    batch_loader = DataLoader(
        subset,
        batch_size=len(subset),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
```

### Getitem Return

Match line 169 (context starts 163):

```python

    if equality.ndim == 1:
        exact_items = equality
    else:
        exact_items = equality.reshape(equality.shape[0], -1).all(dim=1)

    return {
        "element_accuracy": float(equality.float().mean().item()),
        "exact_item_accuracy": float(
            exact_items.float().mean().item()
        ),
        "all_exact": bool(equality.all().item()),
    }
```

Match line 222 (context starts 216):

```python
    all_exact = (
        graph["all_exact"]
        and count_all_exact
        and all(item["all_exact"] for item in roles.values())
    )

    return {
        "graph": graph,
        "count_active_accuracy": count_accuracy,
        "count_all_exact": count_all_exact,
        "roles": roles,
        "all_tasks_all_exact": all_exact,
    }
```

Match line 583 (context starts 577):

```python
            "collated physical-port-mask dtype mismatch"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    active_mask = batch["y_attack"] >= 0.5
    if int(active_mask.sum().item()) != 6:
```

### Model Output

Match line 185 (context starts 179):

```python
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    count_targets: torch.Tensor,
    active_mask: torch.Tensor,
) -> dict[str, Any]:
    graph = exact_binary_metrics(
        outputs["attack_logits"],
        batch["y_attack"],
    )

    active_count_prediction = outputs["count_logits"][
        active_mask
    ].argmax(dim=-1)
```

Match line 189 (context starts 183):

```python
) -> dict[str, Any]:
    graph = exact_binary_metrics(
        outputs["attack_logits"],
        batch["y_attack"],
    )

    active_count_prediction = outputs["count_logits"][
        active_mask
    ].argmax(dim=-1)
    active_count_truth = count_targets[active_mask]
    count_accuracy = float(
        (
            active_count_prediction == active_count_truth
```

Match line 206 (context starts 200):

```python
            active_count_prediction == active_count_truth
        ).all().item()
    )

    roles = {}
    for logical, output_key, target_key in (
        ("source", "source_logits", "y_source"),
        ("transit", "transit_logits", "y_transit"),
        ("victim", "victim_logits", "y_victim"),
        ("path", "path_logits", "y_attack_path"),
    ):
        roles[logical] = exact_binary_metrics(
            outputs[output_key],
```

Match line 207 (context starts 201):

```python
        ).all().item()
    )

    roles = {}
    for logical, output_key, target_key in (
        ("source", "source_logits", "y_source"),
        ("transit", "transit_logits", "y_transit"),
        ("victim", "victim_logits", "y_victim"),
        ("path", "path_logits", "y_attack_path"),
    ):
        roles[logical] = exact_binary_metrics(
            outputs[output_key],
            batch[target_key],
```

Match line 208 (context starts 202):

```python
    )

    roles = {}
    for logical, output_key, target_key in (
        ("source", "source_logits", "y_source"),
        ("transit", "transit_logits", "y_transit"),
        ("victim", "victim_logits", "y_victim"),
        ("path", "path_logits", "y_attack_path"),
    ):
        roles[logical] = exact_binary_metrics(
            outputs[output_key],
            batch[target_key],
        )
```

Match line 209 (context starts 203):

```python

    roles = {}
    for logical, output_key, target_key in (
        ("source", "source_logits", "y_source"),
        ("transit", "transit_logits", "y_transit"),
        ("victim", "victim_logits", "y_victim"),
        ("path", "path_logits", "y_attack_path"),
    ):
        roles[logical] = exact_binary_metrics(
            outputs[output_key],
            batch[target_key],
        )

```

Match line 678 (context starts 672):

```python
        outputs = model(
            batch["x"],
            batch["physical_port_mask"],
        )

        attack_loss = F.binary_cross_entropy_with_logits(
            outputs["attack_logits"],
            batch["y_attack"].float(),
            pos_weight=graph_pos_weight,
        )
        count_loss = F.cross_entropy(
            outputs["count_logits"][active_mask],
            count_targets[active_mask],
```

Match line 683 (context starts 677):

```python
        attack_loss = F.binary_cross_entropy_with_logits(
            outputs["attack_logits"],
            batch["y_attack"].float(),
            pos_weight=graph_pos_weight,
        )
        count_loss = F.cross_entropy(
            outputs["count_logits"][active_mask],
            count_targets[active_mask],
        )
        source_loss = F.binary_cross_entropy_with_logits(
            outputs["source_logits"],
            batch["y_source"].float(),
            pos_weight=role_pos_weights["source"],
```

Match line 687 (context starts 681):

```python
        )
        count_loss = F.cross_entropy(
            outputs["count_logits"][active_mask],
            count_targets[active_mask],
        )
        source_loss = F.binary_cross_entropy_with_logits(
            outputs["source_logits"],
            batch["y_source"].float(),
            pos_weight=role_pos_weights["source"],
        )
        transit_loss = F.binary_cross_entropy_with_logits(
            outputs["transit_logits"],
            batch["y_transit"].float(),
```

Match line 692 (context starts 686):

```python
        source_loss = F.binary_cross_entropy_with_logits(
            outputs["source_logits"],
            batch["y_source"].float(),
            pos_weight=role_pos_weights["source"],
        )
        transit_loss = F.binary_cross_entropy_with_logits(
            outputs["transit_logits"],
            batch["y_transit"].float(),
            pos_weight=role_pos_weights["transit"],
        )
        victim_loss = F.binary_cross_entropy_with_logits(
            outputs["victim_logits"],
            batch["y_victim"].float(),
```

Match line 697 (context starts 691):

```python
        transit_loss = F.binary_cross_entropy_with_logits(
            outputs["transit_logits"],
            batch["y_transit"].float(),
            pos_weight=role_pos_weights["transit"],
        )
        victim_loss = F.binary_cross_entropy_with_logits(
            outputs["victim_logits"],
            batch["y_victim"].float(),
            pos_weight=role_pos_weights["victim"],
        )
        path_loss = F.binary_cross_entropy_with_logits(
            outputs["path_logits"],
            batch["y_attack_path"].float(),
```

Match line 702 (context starts 696):

```python
        victim_loss = F.binary_cross_entropy_with_logits(
            outputs["victim_logits"],
            batch["y_victim"].float(),
            pos_weight=role_pos_weights["victim"],
        )
        path_loss = F.binary_cross_entropy_with_logits(
            outputs["path_logits"],
            batch["y_attack_path"].float(),
            pos_weight=role_pos_weights["path"],
        )

        loss = (
            LOSS_WEIGHTS["attack"] * attack_loss
```

Match line 768 (context starts 762):

```python
            )

            # Recompute the aggregate loss after the optimizer step.
            current_loss = (
                LOSS_WEIGHTS["attack"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["attack_logits"],
                    batch["y_attack"].float(),
                    pos_weight=graph_pos_weight,
                )
                + LOSS_WEIGHTS["count"]
                * F.cross_entropy(
                    current_outputs["count_logits"][active_mask],
```

Match line 774 (context starts 768):

```python
                    current_outputs["attack_logits"],
                    batch["y_attack"].float(),
                    pos_weight=graph_pos_weight,
                )
                + LOSS_WEIGHTS["count"]
                * F.cross_entropy(
                    current_outputs["count_logits"][active_mask],
                    count_targets[active_mask],
                )
                + LOSS_WEIGHTS["source"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["source_logits"],
                    batch["y_source"].float(),
```

Match line 779 (context starts 773):

```python
                * F.cross_entropy(
                    current_outputs["count_logits"][active_mask],
                    count_targets[active_mask],
                )
                + LOSS_WEIGHTS["source"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["source_logits"],
                    batch["y_source"].float(),
                    pos_weight=role_pos_weights["source"],
                )
                + LOSS_WEIGHTS["transit"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["transit_logits"],
```

Match line 785 (context starts 779):

```python
                    current_outputs["source_logits"],
                    batch["y_source"].float(),
                    pos_weight=role_pos_weights["source"],
                )
                + LOSS_WEIGHTS["transit"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["transit_logits"],
                    batch["y_transit"].float(),
                    pos_weight=role_pos_weights["transit"],
                )
                + LOSS_WEIGHTS["victim"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["victim_logits"],
```

Match line 791 (context starts 785):

```python
                    current_outputs["transit_logits"],
                    batch["y_transit"].float(),
                    pos_weight=role_pos_weights["transit"],
                )
                + LOSS_WEIGHTS["victim"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["victim_logits"],
                    batch["y_victim"].float(),
                    pos_weight=role_pos_weights["victim"],
                )
                + LOSS_WEIGHTS["path"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["path_logits"],
```

Match line 797 (context starts 791):

```python
                    current_outputs["victim_logits"],
                    batch["y_victim"].float(),
                    pos_weight=role_pos_weights["victim"],
                )
                + LOSS_WEIGHTS["path"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["path_logits"],
                    batch["y_attack_path"].float(),
                    pos_weight=role_pos_weights["path"],
                )
            )

        completed_step = step
```

### Batch Transfer

Match line 584 (context starts 578):

```python
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    active_mask = batch["y_attack"] >= 0.5
    if int(active_mask.sum().item()) != 6:
        failures.append(
```

Match line 585 (context starts 579):

```python

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    active_mask = batch["y_attack"] >= 0.5
    if int(active_mask.sum().item()) != 6:
        failures.append(
            f"active attack item count={int(active_mask.sum())}, expected 6"
```

Match line 608 (context starts 602):

```python
            active_mask
            & (batch["y_attacker_count"] == raw_count)
        ] = class_index
    if bool((count_targets[active_mask] < 0).any().item()):
        failures.append("one or more active count targets were not mapped")

    model = ModelClass().to(device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
```

### Stable Identity

Match line 134 (context starts 128):

```python
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def category_from_pair_key(pair_key: str) -> str | None:
    match = re.search(r"-K([124])-", pair_key)
    if match is None:
        return None
    return f"K{match.group(1)}"


```

Match line 135 (context starts 129):

```python
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def category_from_pair_key(pair_key: str) -> str | None:
    match = re.search(r"-K([124])-", pair_key)
    if match is None:
        return None
    return f"K{match.group(1)}"


def positive_weight(target: torch.Tensor) -> torch.Tensor:
```

Match line 405 (context starts 399):

```python
    for base in range(0, len(train_dataset._index), 2):
        attack_entry = train_dataset._index[base]
        control_entry = train_dataset._index[base + 1]
        if (
            attack_entry.mode != "attack"
            or control_entry.mode != "control"
            or attack_entry.pair_key != control_entry.pair_key
            or attack_entry.start != control_entry.start
        ):
            failures.append(
                f"train index pair adjacency failed at base={base}"
            )
            continue
```

Match line 412 (context starts 406):

```python
            or attack_entry.start != control_entry.start
        ):
            failures.append(
                f"train index pair adjacency failed at base={base}"
            )
            continue
        last_pair_base[attack_entry.pair_key] = base

    selected_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_count_by_category: dict[str, set[int]] = defaultdict(set)

    for pair_key in sorted(last_pair_base):
        category = category_from_pair_key(pair_key)
```

Match line 417 (context starts 411):

```python
            continue
        last_pair_base[attack_entry.pair_key] = base

    selected_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_count_by_category: dict[str, set[int]] = defaultdict(set)

    for pair_key in sorted(last_pair_base):
        category = category_from_pair_key(pair_key)
        if category not in {"K1", "K2", "K4"}:
            continue
        if len(selected_by_category[category]) >= 2:
            continue

```

Match line 418 (context starts 412):

```python
        last_pair_base[attack_entry.pair_key] = base

    selected_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_count_by_category: dict[str, set[int]] = defaultdict(set)

    for pair_key in sorted(last_pair_base):
        category = category_from_pair_key(pair_key)
        if category not in {"K1", "K2", "K4"}:
            continue
        if len(selected_by_category[category]) >= 2:
            continue

        base = last_pair_base[pair_key]
```

Match line 424 (context starts 418):

```python
        category = category_from_pair_key(pair_key)
        if category not in {"K1", "K2", "K4"}:
            continue
        if len(selected_by_category[category]) >= 2:
            continue

        base = last_pair_base[pair_key]
        attack_item = train_dataset[base]
        control_item = train_dataset[base + 1]

        attack_label = float(attack_item["y_attack"].item())
        control_label = float(control_item["y_attack"].item())
        raw_count = int(attack_item["y_attacker_count"].item())
```

Match line 436 (context starts 430):

```python
        raw_count = int(attack_item["y_attacker_count"].item())

        if attack_label < 0.5:
            continue
        if control_label >= 0.5:
            failures.append(
                f"{pair_key}: matched control is graph-positive"
            )
            continue
        if int(control_item["y_attacker_count"].item()) != 0:
            failures.append(
                f"{pair_key}: matched control attacker count is nonzero"
            )
```

Match line 441 (context starts 435):

```python
            failures.append(
                f"{pair_key}: matched control is graph-positive"
            )
            continue
        if int(control_item["y_attacker_count"].item()) != 0:
            failures.append(
                f"{pair_key}: matched control attacker count is nonzero"
            )
            continue
        if raw_count <= 0:
            failures.append(
                f"{pair_key}: active attack has non-positive count"
            )
```

Match line 446 (context starts 440):

```python
            failures.append(
                f"{pair_key}: matched control attacker count is nonzero"
            )
            continue
        if raw_count <= 0:
            failures.append(
                f"{pair_key}: active attack has non-positive count"
            )
            continue

        entry = train_dataset._index[base]
        selected_by_category[category].append(
            {
```

Match line 454 (context starts 448):

```python
            continue

        entry = train_dataset._index[base]
        selected_by_category[category].append(
            {
                "category": category,
                "pair_key": pair_key,
                "base_index": base,
                "attack_index": base,
                "control_index": base + 1,
                "window_start": entry.start,
                "window_target": entry.target,
                "common_length": entry.common_length,
```

Match line 458 (context starts 452):

```python
            {
                "category": category,
                "pair_key": pair_key,
                "base_index": base,
                "attack_index": base,
                "control_index": base + 1,
                "window_start": entry.start,
                "window_target": entry.target,
                "common_length": entry.common_length,
                "raw_positive_attacker_count": raw_count,
            }
        )
        raw_count_by_category[category].add(raw_count)
```

### Expected Counts

Match line 540 (context starts 534):

```python
            print("FAIL:", failure)
        return 1

    subset = Subset(train_dataset, selected_indices)
    batch_loader = DataLoader(
        subset,
        batch_size=len(subset),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    batch = next(iter(batch_loader))

```

## Current state

- E1 exporter implemented: **false**
- Validation predictions exported: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
