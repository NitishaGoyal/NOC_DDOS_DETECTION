# V5 P2 E1-P0 Validation Export Interface Preflight

## Status

- Preflight complete: **true**
- Validation tensors opened: **false**
- Validation predictions exported: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**

## Selected checkpoint

- Path: `/home/zira/research/projects/GNN-2d/reports/v5/p2_task_d_checkpoint_selection_freeze/selected_graphconv_checkpoint.pt`
- SHA-256: `82314baedb4bf3969842abf9076679682369daae2e57db07d4b4aeaa8623f7cc`
- Selected seed: **107**
- Selected epoch: **25**
- Checkpoint type: `dict`
- State-dict container: `model_state_dict`

### Checkpoint top-level keys

```json
[
  "b1_protocol_sha256",
  "b3_model_sha256",
  "candidate",
  "common_initial_state_sha256",
  "epoch",
  "full_initial_state_sha256",
  "loader_sha256",
  "model_state_dict",
  "optimizer_state_dict",
  "promotion_report_sha256",
  "scheduler_state_dict",
  "seed",
  "stage",
  "task_d_model_sha256",
  "test_tensors_deserialized",
  "threshold_tuning_performed",
  "train_label_weight_manifest",
  "validation_metrics"
]
```

## Pair-aligned manifest metadata

- Path: `/home/zira/research/projects/GNN-2d/reports/v5/p2_a1_r2_pair_aligned_window_contract/V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv`
- SHA-256: `f42f40446d03161a6932ee060f6d8c859f894cb5075d1fc926ea161461413ab5`
- Full row count computed: **false**
- Columns:

```json
[
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
]
```

## Source: `scripts/v5/p2/train_v5_p2_task_d_full_multitask_single_run.py`

- SHA-256: `553dd206a8e6518b3a1e768c8f9c69633fcada421475b8ca75fda57c2fc13766`
- Batch keys detected: `['physical_port_mask', 'x']`
- Output keys detected: `[]`

### Classes

```json
[]
```

### Functions

```json
[
  {
    "name": "import_module",
    "line": 31,
    "args": [
      "path",
      "name"
    ]
  },
  {
    "name": "sha256_file",
    "line": 47,
    "args": [
      "path"
    ]
  },
  {
    "name": "write_json",
    "line": 55,
    "args": [
      "path",
      "obj"
    ]
  },
  {
    "name": "measure_latency",
    "line": 61,
    "args": [
      "model",
      "batch",
      "device"
    ]
  },
  {
    "name": "main",
    "line": 80,
    "args": []
  }
]
```

### Evidence: Model Construction

Line 246 (context starts 243):

```python
    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
    if candidate == "conv1d":
        model = reference_b3
    else:
        model = td_mod.P2TaskDGraphConvCount4(reference_b3, edge)
    model = model.to(device)
```

Line 248 (context starts 245):

```python
    if candidate == "conv1d":
        model = reference_b3
    else:
        model = td_mod.P2TaskDGraphConvCount4(reference_b3, edge)
    model = model.to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != EXPECTED_PARAMS[candidate]:
```

Line 249 (context starts 246):

```python
        model = reference_b3
    else:
        model = td_mod.P2TaskDGraphConvCount4(reference_b3, edge)
    model = model.to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != EXPECTED_PARAMS[candidate]:
        raise RuntimeError(f"parameter count={parameter_count}, expected {EXPECTED_PARAMS[candidate]}")
```

Line 307 (context starts 304):

```python
        train_sampler.set_epoch(epoch)
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = b2.train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, device=device,
            graph_pos_weight=graph_pos_weight, role_pos_weights=role_pos_weights,
            gradient_clip=1.0,
        )
```

Line 312 (context starts 309):

```python
            gradient_clip=1.0,
        )
        validation_metrics = b2.validate(
            model=model, loader=validation_loader, device=device,
            graph_pos_weight=graph_pos_weight, role_pos_weights=role_pos_weights,
        )
        rank = b2.checkpoint_rank(validation_metrics, epoch)
```

### Evidence: Checkpoint Loading

Line 254 (context starts 251):

```python
    if parameter_count != EXPECTED_PARAMS[candidate]:
        raise RuntimeError(f"parameter count={parameter_count}, expected {EXPECTED_PARAMS[candidate]}")

    full_initial_state_sha256 = b2.sha256_state_dict(model.state_dict())
    if candidate == "conv1d":
        common_initial_state = OrderedDict((k, v) for k, v in model.state_dict().items())
    else:
```

Line 256 (context starts 253):

```python

    full_initial_state_sha256 = b2.sha256_state_dict(model.state_dict())
    if candidate == "conv1d":
        common_initial_state = OrderedDict((k, v) for k, v in model.state_dict().items())
    else:
        common_initial_state = OrderedDict(td_mod.common_state_dict(model).items())
    common_initial_state_sha256 = b2.sha256_state_dict(common_initial_state)
```

Line 258 (context starts 255):

```python
    if candidate == "conv1d":
        common_initial_state = OrderedDict((k, v) for k, v in model.state_dict().items())
    else:
        common_initial_state = OrderedDict(td_mod.common_state_dict(model).items())
    common_initial_state_sha256 = b2.sha256_state_dict(common_initial_state)

    optimizer = torch.optim.AdamW(
```

Line 259 (context starts 256):

```python
        common_initial_state = OrderedDict((k, v) for k, v in model.state_dict().items())
    else:
        common_initial_state = OrderedDict(td_mod.common_state_dict(model).items())
    common_initial_state_sha256 = b2.sha256_state_dict(common_initial_state)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4
```

Line 274 (context starts 271):

```python
        for role, details in role_weight_manifest.items()
    }

    checkpoint_path = model_dir / "best_checkpoint.pt"
    history_path = report_dir / f"V5_P2_TASK_D_{candidate.upper()}_SEED_{seed}_HISTORY.csv"
    progress_path = report_dir / f"V5_P2_TASK_D_{candidate.upper()}_SEED_{seed}_PROGRESS.json"
    start = time.time()
```

Line 281 (context starts 278):

```python
    best_rank = None
    best_epoch = None
    best_validation_metrics = None
    best_checkpoint_sha256 = None
    best_early_stop_score = float("-inf")
    patience_counter = 0
    history_rows: list[dict[str, Any]] = []
```

Line 315 (context starts 312):

```python
            model=model, loader=validation_loader, device=device,
            graph_pos_weight=graph_pos_weight, role_pos_weights=role_pos_weights,
        )
        rank = b2.checkpoint_rank(validation_metrics, epoch)
        is_best = best_rank is None or rank > best_rank
        if is_best:
            best_rank = rank
```

Line 326 (context starts 323):

```python
                "candidate": candidate,
                "seed": seed,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "b1_protocol_sha256": protocol["protocol_sha256"],
```

Line 327 (context starts 324):

```python
                "seed": seed,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "b1_protocol_sha256": protocol["protocol_sha256"],
                "promotion_report_sha256": sha256_file(paths["promotion_report"]),
```

Line 328 (context starts 325):

```python
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "b1_protocol_sha256": protocol["protocol_sha256"],
                "promotion_report_sha256": sha256_file(paths["promotion_report"]),
                "loader_sha256": sha256_file(paths["loader"]),
```

Line 346 (context starts 343):

```python
                "threshold_tuning_performed": False,
                "test_tensors_deserialized": False,
            }
            b2.atomic_torch_save(payload, checkpoint_path)
            best_checkpoint_sha256 = sha256_file(checkpoint_path)

        score = float(validation_metrics["selection_score"])
```

Line 347 (context starts 344):

```python
                "test_tensors_deserialized": False,
            }
            b2.atomic_torch_save(payload, checkpoint_path)
            best_checkpoint_sha256 = sha256_file(checkpoint_path)

        score = float(validation_metrics["selection_score"])
        if score > best_early_stop_score + 1e-4:
```

Line 361 (context starts 358):

```python
            epoch=epoch, learning_rate=current_lr, train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            elapsed_seconds=time.time() - epoch_start,
            is_best_checkpoint=is_best,
            early_stop_patience_counter=patience_counter,
        )
        row = {"candidate": candidate, "seed": seed, **row}
```

Line 370 (context starts 367):

```python
        write_json(progress_path, {
            "stage": STAGE, "status": "RUNNING", "candidate": candidate,
            "seed": seed, "completed_epoch": completed_epoch, "best_epoch": best_epoch,
            "best_checkpoint_sha256": best_checkpoint_sha256,
            "best_validation_metrics": best_validation_metrics,
            "current_validation_metrics": validation_metrics,
            "current_learning_rate_after_scheduler": float(optimizer.param_groups[0]["lr"]),
```

Line 394 (context starts 391):

```python
            print(f"early stopping at epoch {epoch}; patience reached 12")
            break

    if best_epoch is None or best_validation_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("no best checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
```

Line 395 (context starts 392):

```python
            break

    if best_epoch is None or best_validation_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("no best checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    first_validation_batch = next(iter(validation_loader))
```

Line 396 (context starts 393):

```python

    if best_epoch is None or best_validation_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("no best checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    first_validation_batch = next(iter(validation_loader))
    latency_us = measure_latency(model, first_validation_batch, device)
```

Line 397 (context starts 394):

```python
    if best_epoch is None or best_validation_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("no best checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    first_validation_batch = next(iter(validation_loader))
    latency_us = measure_latency(model, first_validation_batch, device)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
```

Line 442 (context starts 439):

```python
        },
        "best": {
            "epoch": best_epoch, "validation_metrics": best_validation_metrics,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "ranking_tuple": list(best_rank),
        },
```

Line 443 (context starts 440):

```python
        "best": {
            "epoch": best_epoch, "validation_metrics": best_validation_metrics,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "ranking_tuple": list(best_rank),
        },
        "deployment_proxies": {
```

Line 454 (context starts 451):

```python
        "artifacts": {
            "history_csv": [str(history_path), sha256_file(history_path)],
            "progress_json": [str(progress_path), sha256_file(progress_path)],
            "best_checkpoint": [str(checkpoint_path), sha256_file(checkpoint_path)],
        },
        "security_boundary": {
            "train_tensor_contents_accessed": True,
```

Line 476 (context starts 473):

```python
        "status": COMPLETE, "candidate": candidate, "seed": seed,
        "report_sha256": sha256_file(report_path),
        "history_sha256": sha256_file(history_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "common_initial_state_sha256": common_initial_state_sha256,
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_validation_metrics["selection_score"],
```

### Evidence: Dataset Or Loader

Line 18 (context starts 15):

```python

import numpy as np
import torch
from torch.utils.data import DataLoader

STAGE = "V5_P2_TASK_D_FULL_MULTITASK_SINGLE_RUN"
COMPLETE = f"{STAGE}_COMPLETE"
```

Line 137 (context starts 134):

```python
    failures = [f"missing {name}: {path}" for name, path in paths.items() if not path.is_file()]
    warnings: list[str] = []
    if not root.is_dir():
        failures.append(f"dataset root missing: {root}")
    if failures:
        write_json(report_dir / f"{STAGE}.json", {
            "stage": STAGE, "status": "HOLD", "candidate": candidate,
```

Line 201 (context starts 198):

```python
    b2.EXPECTED_VALIDATION_ITEMS = EXPECTED_VALIDATION_ITEMS
    b2.set_seed(seed)

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
```

Line 202 (context starts 199):

```python
    b2.set_seed(seed)

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
```

Line 203 (context starts 200):

```python

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
```

Line 204 (context starts 201):

```python
    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation length={len(validation_dataset)}")
```

Line 205 (context starts 202):

```python
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation length={len(validation_dataset)}")

```

Line 206 (context starts 203):

```python
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation length={len(validation_dataset)}")

    train_summary = b0_report["label_summaries"]["train"]
```

Line 207 (context starts 204):

```python
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation length={len(validation_dataset)}")

    train_summary = b0_report["label_summaries"]["train"]
    graph_positive = int(train_summary["graph_positive"])
```

Line 238 (context starts 235):

```python

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)
```

Line 239 (context starts 236):

```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

```

Line 240 (context starts 237):

```python
    pin_memory = device.type == "cuda"
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
```

Line 241 (context starts 238):

```python
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
```

Line 294 (context starts 291):

```python
    print("candidate:", candidate)
    print("seed:", seed)
    print("device:", device)
    print("train_items:", len(train_dataset))
    print("validation_items:", len(validation_dataset))
    print("parameter_count:", parameter_count)
    print("b1_protocol_sha256:", protocol["protocol_sha256"])
```

Line 295 (context starts 292):

```python
    print("seed:", seed)
    print("device:", device)
    print("train_items:", len(train_dataset))
    print("validation_items:", len(validation_dataset))
    print("parameter_count:", parameter_count)
    print("b1_protocol_sha256:", protocol["protocol_sha256"])
    print("threshold_tuning_performed: false")
```

Line 312 (context starts 309):

```python
            gradient_clip=1.0,
        )
        validation_metrics = b2.validate(
            model=model, loader=validation_loader, device=device,
            graph_pos_weight=graph_pos_weight, role_pos_weights=role_pos_weights,
        )
        rank = b2.checkpoint_rank(validation_metrics, epoch)
```

Line 398 (context starts 395):

```python
        raise RuntimeError("no best checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    first_validation_batch = next(iter(validation_loader))
    latency_us = measure_latency(model, first_validation_batch, device)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    operation_proxy = td_mod.operation_count_proxy(candidate)
```

Line 420 (context starts 417):

```python
            "promotion_report_sha256": sha256_file(paths["promotion_report"]),
        },
        "data": {
            "train_items": len(train_dataset), "validation_items": len(validation_dataset),
            "pair_block_batch_size": 128, "item_batch_size": 256,
            "train_batches": len(train_loader), "validation_batches": len(validation_loader),
            "pair_keys_returned_to_model": False,
```

Line 422 (context starts 419):

```python
        "data": {
            "train_items": len(train_dataset), "validation_items": len(validation_dataset),
            "pair_block_batch_size": 128, "item_batch_size": 256,
            "train_batches": len(train_loader), "validation_batches": len(validation_loader),
            "pair_keys_returned_to_model": False,
        },
        "label_weights": {
```

Line 464 (context starts 461):

```python
            "test_directory_enumerated": False,
            "test_tensor_files_opened": False,
            "test_tensors_deserialized": False,
            "test_dataset_constructed": False,
            "test_evaluation_performed": False,
        },
        "failures": [], "warnings": warnings,
```

### Evidence: Validation Split

Line 26 (context starts 23):

```python
CANDIDATES = ["conv1d", "graphconv"]
EXPECTED_PARAMS = {"conv1d": 43_273, "graphconv": 59_785}
EXPECTED_TRAIN_ITEMS = 70_166
EXPECTED_VALIDATION_ITEMS = 12_528
EXPECTED_B1_PROTOCOL_SHA = "0817ae7812f91e3c75260589acaf52bd8a74b07b2114a451b4773578efde4b60"
COUNT_CLASS_VALUES = [1, 2, 3, 4]

```

Line 198 (context starts 195):

```python

    b2.EXPECTED_SEEDS = SEEDS
    b2.EXPECTED_TRAIN_ITEMS = EXPECTED_TRAIN_ITEMS
    b2.EXPECTED_VALIDATION_ITEMS = EXPECTED_VALIDATION_ITEMS
    b2.set_seed(seed)

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
```

Line 203 (context starts 200):

```python

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
```

Line 206 (context starts 203):

```python
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation length={len(validation_dataset)}")

    train_summary = b0_report["label_summaries"]["train"]
```

Line 207 (context starts 204):

```python
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation length={len(validation_dataset)}")

    train_summary = b0_report["label_summaries"]["train"]
    graph_positive = int(train_summary["graph_positive"])
```

Line 239 (context starts 236):

```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

```

Line 241 (context starts 238):

```python
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
```

Line 280 (context starts 277):

```python
    start = time.time()
    best_rank = None
    best_epoch = None
    best_validation_metrics = None
    best_checkpoint_sha256 = None
    best_early_stop_score = float("-inf")
    patience_counter = 0
```

Line 295 (context starts 292):

```python
    print("seed:", seed)
    print("device:", device)
    print("train_items:", len(train_dataset))
    print("validation_items:", len(validation_dataset))
    print("parameter_count:", parameter_count)
    print("b1_protocol_sha256:", protocol["protocol_sha256"])
    print("threshold_tuning_performed: false")
```

Line 311 (context starts 308):

```python
            graph_pos_weight=graph_pos_weight, role_pos_weights=role_pos_weights,
            gradient_clip=1.0,
        )
        validation_metrics = b2.validate(
            model=model, loader=validation_loader, device=device,
            graph_pos_weight=graph_pos_weight, role_pos_weights=role_pos_weights,
        )
```

Line 312 (context starts 309):

```python
            gradient_clip=1.0,
        )
        validation_metrics = b2.validate(
            model=model, loader=validation_loader, device=device,
            graph_pos_weight=graph_pos_weight, role_pos_weights=role_pos_weights,
        )
        rank = b2.checkpoint_rank(validation_metrics, epoch)
```

Line 315 (context starts 312):

```python
            model=model, loader=validation_loader, device=device,
            graph_pos_weight=graph_pos_weight, role_pos_weights=role_pos_weights,
        )
        rank = b2.checkpoint_rank(validation_metrics, epoch)
        is_best = best_rank is None or rank > best_rank
        if is_best:
            best_rank = rank
```

Line 320 (context starts 317):

```python
        if is_best:
            best_rank = rank
            best_epoch = epoch
            best_validation_metrics = validation_metrics
            payload = {
                "stage": STAGE,
                "candidate": candidate,
```

Line 342 (context starts 339):

```python
                    "count_distribution": count_distribution,
                    "count_class_mapping": b2.COUNT_CLASS_MAPPING,
                },
                "validation_metrics": validation_metrics,
                "threshold_tuning_performed": False,
                "test_tensors_deserialized": False,
            }
```

Line 349 (context starts 346):

```python
            b2.atomic_torch_save(payload, checkpoint_path)
            best_checkpoint_sha256 = sha256_file(checkpoint_path)

        score = float(validation_metrics["selection_score"])
        if score > best_early_stop_score + 1e-4:
            best_early_stop_score = score
            patience_counter = 0
```

Line 359 (context starts 356):

```python
        completed_epoch = epoch
        row = b2.flatten_epoch_row(
            epoch=epoch, learning_rate=current_lr, train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            elapsed_seconds=time.time() - epoch_start,
            is_best_checkpoint=is_best,
            early_stop_patience_counter=patience_counter,
```

Line 371 (context starts 368):

```python
            "stage": STAGE, "status": "RUNNING", "candidate": candidate,
            "seed": seed, "completed_epoch": completed_epoch, "best_epoch": best_epoch,
            "best_checkpoint_sha256": best_checkpoint_sha256,
            "best_validation_metrics": best_validation_metrics,
            "current_validation_metrics": validation_metrics,
            "current_learning_rate_after_scheduler": float(optimizer.param_groups[0]["lr"]),
            "early_stop_patience_counter": patience_counter,
```

Line 372 (context starts 369):

```python
            "seed": seed, "completed_epoch": completed_epoch, "best_epoch": best_epoch,
            "best_checkpoint_sha256": best_checkpoint_sha256,
            "best_validation_metrics": best_validation_metrics,
            "current_validation_metrics": validation_metrics,
            "current_learning_rate_after_scheduler": float(optimizer.param_groups[0]["lr"]),
            "early_stop_patience_counter": patience_counter,
            "test_directory_enumerated": False, "test_tensors_deserialized": False,
```

Line 379 (context starts 376):

```python
        })
        print(
            f"candidate={candidate} seed={seed} epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} val_loss={validation_metrics['loss']:.6f} "
            f"score={score:.6f} g_auc={validation_metrics['graph']['auroc']:.6f} "
            f"g_ap={validation_metrics['graph']['average_precision']:.6f} "
            f"count_f1={validation_metrics['count_active']['macro_f1']:.6f} "
```

Line 380 (context starts 377):

```python
        print(
            f"candidate={candidate} seed={seed} epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} val_loss={validation_metrics['loss']:.6f} "
            f"score={score:.6f} g_auc={validation_metrics['graph']['auroc']:.6f} "
            f"g_ap={validation_metrics['graph']['average_precision']:.6f} "
            f"count_f1={validation_metrics['count_active']['macro_f1']:.6f} "
            f"src_ap={validation_metrics['roles']['source']['average_precision']:.6f} "
```

Line 381 (context starts 378):

```python
            f"candidate={candidate} seed={seed} epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} val_loss={validation_metrics['loss']:.6f} "
            f"score={score:.6f} g_auc={validation_metrics['graph']['auroc']:.6f} "
            f"g_ap={validation_metrics['graph']['average_precision']:.6f} "
            f"count_f1={validation_metrics['count_active']['macro_f1']:.6f} "
            f"src_ap={validation_metrics['roles']['source']['average_precision']:.6f} "
            f"tr_ap={validation_metrics['roles']['transit']['average_precision']:.6f} "
```

Line 382 (context starts 379):

```python
            f"train_loss={train_metrics['loss']:.6f} val_loss={validation_metrics['loss']:.6f} "
            f"score={score:.6f} g_auc={validation_metrics['graph']['auroc']:.6f} "
            f"g_ap={validation_metrics['graph']['average_precision']:.6f} "
            f"count_f1={validation_metrics['count_active']['macro_f1']:.6f} "
            f"src_ap={validation_metrics['roles']['source']['average_precision']:.6f} "
            f"tr_ap={validation_metrics['roles']['transit']['average_precision']:.6f} "
            f"vic_ap={validation_metrics['roles']['victim']['average_precision']:.6f} "
```

Line 383 (context starts 380):

```python
            f"score={score:.6f} g_auc={validation_metrics['graph']['auroc']:.6f} "
            f"g_ap={validation_metrics['graph']['average_precision']:.6f} "
            f"count_f1={validation_metrics['count_active']['macro_f1']:.6f} "
            f"src_ap={validation_metrics['roles']['source']['average_precision']:.6f} "
            f"tr_ap={validation_metrics['roles']['transit']['average_precision']:.6f} "
            f"vic_ap={validation_metrics['roles']['victim']['average_precision']:.6f} "
            f"path_ap={validation_metrics['roles']['path']['average_precision']:.6f} "
```

Line 384 (context starts 381):

```python
            f"g_ap={validation_metrics['graph']['average_precision']:.6f} "
            f"count_f1={validation_metrics['count_active']['macro_f1']:.6f} "
            f"src_ap={validation_metrics['roles']['source']['average_precision']:.6f} "
            f"tr_ap={validation_metrics['roles']['transit']['average_precision']:.6f} "
            f"vic_ap={validation_metrics['roles']['victim']['average_precision']:.6f} "
            f"path_ap={validation_metrics['roles']['path']['average_precision']:.6f} "
            f"lr={current_lr:.8g} best_epoch={best_epoch} patience={patience_counter}"
```

Line 385 (context starts 382):

```python
            f"count_f1={validation_metrics['count_active']['macro_f1']:.6f} "
            f"src_ap={validation_metrics['roles']['source']['average_precision']:.6f} "
            f"tr_ap={validation_metrics['roles']['transit']['average_precision']:.6f} "
            f"vic_ap={validation_metrics['roles']['victim']['average_precision']:.6f} "
            f"path_ap={validation_metrics['roles']['path']['average_precision']:.6f} "
            f"lr={current_lr:.8g} best_epoch={best_epoch} patience={patience_counter}"
        )
```

Line 386 (context starts 383):

```python
            f"src_ap={validation_metrics['roles']['source']['average_precision']:.6f} "
            f"tr_ap={validation_metrics['roles']['transit']['average_precision']:.6f} "
            f"vic_ap={validation_metrics['roles']['victim']['average_precision']:.6f} "
            f"path_ap={validation_metrics['roles']['path']['average_precision']:.6f} "
            f"lr={current_lr:.8g} best_epoch={best_epoch} patience={patience_counter}"
        )
        if epoch >= 15 and patience_counter >= 12:
```

Line 394 (context starts 391):

```python
            print(f"early stopping at epoch {epoch}; patience reached 12")
            break

    if best_epoch is None or best_validation_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("no best checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
```

Line 398 (context starts 395):

```python
        raise RuntimeError("no best checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    first_validation_batch = next(iter(validation_loader))
    latency_us = measure_latency(model, first_validation_batch, device)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    operation_proxy = td_mod.operation_count_proxy(candidate)
```

Line 399 (context starts 396):

```python
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    first_validation_batch = next(iter(validation_loader))
    latency_us = measure_latency(model, first_validation_batch, device)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    operation_proxy = td_mod.operation_count_proxy(candidate)

```

Line 420 (context starts 417):

```python
            "promotion_report_sha256": sha256_file(paths["promotion_report"]),
        },
        "data": {
            "train_items": len(train_dataset), "validation_items": len(validation_dataset),
            "pair_block_batch_size": 128, "item_batch_size": 256,
            "train_batches": len(train_loader), "validation_batches": len(validation_loader),
            "pair_keys_returned_to_model": False,
```

Line 422 (context starts 419):

```python
        "data": {
            "train_items": len(train_dataset), "validation_items": len(validation_dataset),
            "pair_block_batch_size": 128, "item_batch_size": 256,
            "train_batches": len(train_loader), "validation_batches": len(validation_loader),
            "pair_keys_returned_to_model": False,
        },
        "label_weights": {
```

Line 441 (context starts 438):

```python
            "total_elapsed_seconds": time.time() - start,
        },
        "best": {
            "epoch": best_epoch, "validation_metrics": best_validation_metrics,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "ranking_tuple": list(best_rank),
```

Line 458 (context starts 455):

```python
        },
        "security_boundary": {
            "train_tensor_contents_accessed": True,
            "validation_tensor_contents_accessed": True,
            "threshold_tuning_performed": False,
            "test_directory_existence_checked": False,
            "test_directory_enumerated": False,
```

Line 479 (context starts 476):

```python
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "common_initial_state_sha256": common_initial_state_sha256,
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_validation_metrics["selection_score"],
        "threshold_tuning_performed": False,
        "architecture_selected": False,
        "test_directory_enumerated": False,
```

Line 493 (context starts 490):

```python
    print("candidate:", candidate)
    print("seed:", seed)
    print("best_epoch:", best_epoch)
    print("best_validation_selection_score:", best_validation_metrics["selection_score"])
    print("inference_microseconds_per_item:", latency_us)
    print("threshold_tuning_performed: false")
    print("architecture_selected: false")
```

### Evidence: Batch Keys

Line 62 (context starts 59):

```python


def measure_latency(model, batch: dict[str, torch.Tensor], device: torch.device) -> float:
    x = batch["x"].to(device, non_blocking=(device.type == "cuda"))
    mask = batch["physical_port_mask"].to(device, non_blocking=(device.type == "cuda"))
    model.eval()
    warmup, repeats = 10, 40
```

Line 63 (context starts 60):

```python

def measure_latency(model, batch: dict[str, torch.Tensor], device: torch.device) -> float:
    x = batch["x"].to(device, non_blocking=(device.type == "cuda"))
    mask = batch["physical_port_mask"].to(device, non_blocking=(device.type == "cuda"))
    model.eval()
    warmup, repeats = 10, 40
    with torch.no_grad():
```

### Evidence: Stable Ids

Line 198 (context starts 195):

```python

    b2.EXPECTED_SEEDS = SEEDS
    b2.EXPECTED_TRAIN_ITEMS = EXPECTED_TRAIN_ITEMS
    b2.EXPECTED_VALIDATION_ITEMS = EXPECTED_VALIDATION_ITEMS
    b2.set_seed(seed)

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
```

Line 239 (context starts 236):

```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

```

Line 241 (context starts 238):

```python
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
```

Line 295 (context starts 292):

```python
    print("seed:", seed)
    print("device:", device)
    print("train_items:", len(train_dataset))
    print("validation_items:", len(validation_dataset))
    print("parameter_count:", parameter_count)
    print("b1_protocol_sha256:", protocol["protocol_sha256"])
    print("threshold_tuning_performed: false")
```

Line 420 (context starts 417):

```python
            "promotion_report_sha256": sha256_file(paths["promotion_report"]),
        },
        "data": {
            "train_items": len(train_dataset), "validation_items": len(validation_dataset),
            "pair_block_batch_size": 128, "item_batch_size": 256,
            "train_batches": len(train_loader), "validation_batches": len(validation_loader),
            "pair_keys_returned_to_model": False,
```

### Evidence: Physical Port Mask

Line 63 (context starts 60):

```python

def measure_latency(model, batch: dict[str, torch.Tensor], device: torch.device) -> float:
    x = batch["x"].to(device, non_blocking=(device.type == "cuda"))
    mask = batch["physical_port_mask"].to(device, non_blocking=(device.type == "cuda"))
    model.eval()
    warmup, repeats = 10, 40
    with torch.no_grad():
```

## Source: `src/models/v5_p2_task_d_full_multitask_count4.py`

- SHA-256: `ccdfcb74c1ddab7f20b98c922444873ad76c204ec4b5fd6827d52fb716f11fd9`
- Batch keys detected: `[]`
- Output keys detected: `['attack_logits', 'count_logits', 'path_logits', 'source_logits', 'transit_logits', 'victim_logits']`

### Classes

```json
[
  {
    "name": "P2TaskDGraphConvCount4",
    "line": 41,
    "methods": [
      {
        "name": "__init__",
        "line": 54,
        "args": [
          "self",
          "reference_b3",
          "base_edge_index"
        ]
      },
      {
        "name": "encode_pre_graph",
        "line": 86,
        "args": [
          "self",
          "x",
          "physical_port_mask"
        ]
      },
      {
        "name": "forward",
        "line": 115,
        "args": [
          "self",
          "x",
          "physical_port_mask"
        ]
      }
    ]
  }
]
```

### Functions

```json
[
  {
    "name": "batched_edge_index",
    "line": 19,
    "args": [
      "base_edge_index",
      "batch_size",
      "num_nodes",
      "device"
    ]
  },
  {
    "name": "common_state_dict",
    "line": 165,
    "args": [
      "model"
    ]
  },
  {
    "name": "operation_count_proxy",
    "line": 175,
    "args": [
      "candidate"
    ]
  }
]
```

### Evidence: Model Construction

Line 41 (context starts 38):

```python
    return result


class P2TaskDGraphConvCount4(nn.Module):
    """
    B3 temporal encoder + two frozen-width GraphConv layers + unchanged B3 heads.

```

### Evidence: Checkpoint Loading

Line 165 (context starts 162):

```python
        return outputs


def common_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only parameters/buffers shared by both Task-D candidates."""
    excluded_prefixes = ("graph1.", "graph2.", "base_edge_index")
    return {
```

Line 170 (context starts 167):

```python
    excluded_prefixes = ("graph1.", "graph2.", "base_edge_index")
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith(excluded_prefixes)
    }

```

### Evidence: Output Keys

Line 144 (context starts 141):

```python
        )
        graph_embedding = self.graph_projection(pooled)
        outputs = {
            "attack_logits": self.attack_head(graph_embedding).squeeze(-1),
            "count_logits": self.count_head(graph_embedding),
            "source_logits": source_logits,
            "transit_logits": transit_logits,
```

Line 145 (context starts 142):

```python
        graph_embedding = self.graph_projection(pooled)
        outputs = {
            "attack_logits": self.attack_head(graph_embedding).squeeze(-1),
            "count_logits": self.count_head(graph_embedding),
            "source_logits": source_logits,
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
```

Line 146 (context starts 143):

```python
        outputs = {
            "attack_logits": self.attack_head(graph_embedding).squeeze(-1),
            "count_logits": self.count_head(graph_embedding),
            "source_logits": source_logits,
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
            "path_logits": path_logits,
```

Line 147 (context starts 144):

```python
            "attack_logits": self.attack_head(graph_embedding).squeeze(-1),
            "count_logits": self.count_head(graph_embedding),
            "source_logits": source_logits,
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
```

Line 148 (context starts 145):

```python
            "count_logits": self.count_head(graph_embedding),
            "source_logits": source_logits,
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
        expected = {
```

Line 149 (context starts 146):

```python
            "source_logits": source_logits,
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
        expected = {
            "attack_logits": (batch_size,),
```

Line 152 (context starts 149):

```python
            "path_logits": path_logits,
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
```

Line 153 (context starts 150):

```python
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
```

Line 154 (context starts 151):

```python
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
```

Line 155 (context starts 152):

```python
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
```

Line 156 (context starts 153):

```python
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
        for key, shape in expected.items():
```

Line 157 (context starts 154):

```python
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
        for key, shape in expected.items():
            if tuple(outputs[key].shape) != shape:
```

### Evidence: Physical Port Mask

Line 89 (context starts 86):

```python
    def encode_pre_graph(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (16, 58, 32):
            raise ValueError(f"x shape={tuple(x.shape)}, expected [B,16,58,32]")
```

Line 94 (context starts 91):

```python
        if x.ndim != 4 or tuple(x.shape[1:]) != (16, 58, 32):
            raise ValueError(f"x shape={tuple(x.shape)}, expected [B,16,58,32]")
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
```

Line 95 (context starts 92):

```python
            raise ValueError(f"x shape={tuple(x.shape)}, expected [B,16,58,32]")
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError("physical_port_mask must have shape [B,16,10]")
```

Line 96 (context starts 93):

```python
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError("physical_port_mask must have shape [B,16,10]")

```

Line 98 (context starts 95):

```python
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError("physical_port_mask must have shape [B,16,10]")

        batch_size, num_nodes, num_features, time_steps = x.shape
        y = x.reshape(batch_size * num_nodes, num_features, time_steps)
```

Line 109 (context starts 106):

```python
        node_input = torch.cat(
            (
                temporal_embedding,
                physical_port_mask.to(dtype=temporal_embedding.dtype),
            ),
            dim=-1,
        )
```

Line 118 (context starts 115):

```python
    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_embedding = self.encode_pre_graph(x, physical_port_mask)
        batch_size, num_nodes, embedding_dim = node_embedding.shape
```

Line 120 (context starts 117):

```python
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_embedding = self.encode_pre_graph(x, physical_port_mask)
        batch_size, num_nodes, embedding_dim = node_embedding.shape
        flat = node_embedding.reshape(batch_size * num_nodes, embedding_dim)
        edges = batched_edge_index(
```

## Source: `src/data/v5_p2_pair_aligned_primary58_dataset.py`

- SHA-256: `2725ff993f4f03ebee3d5ffb775b1fedd6b131a9c4c89ed8049f45b249c24ac2`
- Batch keys detected: `[]`
- Output keys detected: `[]`

### Classes

```json
[
  {
    "name": "IndexEntry",
    "line": 62,
    "methods": []
  },
  {
    "name": "V5P2PairAlignedPrimary58Dataset",
    "line": 137,
    "methods": [
      {
        "name": "__init__",
        "line": 151,
        "args": [
          "self",
          "root",
          "split",
          "pair_manifest"
        ]
      },
      {
        "name": "_load_run",
        "line": 318,
        "args": [
          "path"
        ]
      },
      {
        "name": "__len__",
        "line": 328,
        "args": [
          "self"
        ]
      },
      {
        "name": "pair_count",
        "line": 332,
        "args": [
          "self"
        ]
      },
      {
        "name": "__getitem__",
        "line": 335,
        "args": [
          "self",
          "index"
        ]
      }
    ]
  }
]
```

### Functions

```json
[
  {
    "name": "_as_int",
    "line": 71,
    "args": [
      "row",
      "key"
    ]
  },
  {
    "name": "derive_corrected_physical_port_mask",
    "line": 81,
    "args": [
      "topology"
    ]
  }
]
```

### Evidence: Checkpoint Loading

Line 10 (context starts 7):

```python
    validation

The P2 test split is intentionally rejected. A separately locked test loader
must be created only after the P2 model, checkpoint, and thresholds are frozen.

Returned model/target dictionary:
    x                  float32 [16,58,32]
```

Line 192 (context starts 189):

```python
                f"authorized split directory missing: {split_dir}"
            )

        topology = torch.load(
            self.root / "topology.pt",
            map_location="cpu",
            weights_only=False,
```

Line 319 (context starts 316):

```python
    @staticmethod
    @lru_cache(maxsize=8)
    def _load_run(path: Path) -> dict[str, Any]:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
```

### Evidence: Dataset Or Loader

Line 3 (context starts 1):

```python
#!/usr/bin/env python3
"""
Leakage-safe P2 dataset loader for the frozen B3 Conv1D-only architecture.

Authorized splits in this module:
    train
```

Line 36 (context starts 33):

```python
from typing import Any

import torch
from torch.utils.data import Dataset


PRIMARY58_INDICES = tuple(
```

Line 137 (context starts 134):

```python
    return mask.contiguous()


class V5P2PairAlignedPrimary58Dataset(Dataset):
    """
    Pair-aligned P2 train/validation windows for frozen B3.

```

Line 180 (context starts 177):

```python
                f"stride must remain frozen at {STRIDE}, got {self.stride}"
            )
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root missing: {self.root}")
        if not self.pair_manifest.is_file():
            raise FileNotFoundError(
                f"pair manifest missing: {self.pair_manifest}"
```

Line 313 (context starts 310):

```python
            )
        if len(self._index) % 2 != 0:
            raise RuntimeError(
                "pair-aligned dataset length must be even"
            )

    @staticmethod
```

### Evidence: Validation Split

Line 7 (context starts 4):

```python

Authorized splits in this module:
    train
    validation

The P2 test split is intentionally rejected. A separately locked test loader
must be created only after the P2 model, checkpoint, and thresholds are frozen.
```

Line 46 (context starts 43):

```python
)
WINDOW = 32
STRIDE = 8
AUTHORIZED_SPLITS = {"train", "validation"}

MODEL_ITEM_KEYS = {
    "x",
```

Line 139 (context starts 136):

```python

class V5P2PairAlignedPrimary58Dataset(Dataset):
    """
    Pair-aligned P2 train/validation windows for frozen B3.

    Window-index construction comes exclusively from the frozen A1-R2 pair
    manifest. Both ATTACK and CONTROL members of a matched pair receive the
```

Line 168 (context starts 165):

```python

        if self.split not in AUTHORIZED_SPLITS:
            raise ValueError(
                "This audited loader authorizes only train and validation. "
                "The P2 test split remains locked."
            )
        if self.window != WINDOW:
```

### Evidence: Stable Ids

Line 252 (context starts 249):

```python
                0
                if common_length < self.window
                else 1
                + (common_length - self.window) // self.stride
            )
            if window_count != expected_window_count:
                raise ValueError(
```

Line 272 (context starts 269):

```python
                )

            for window_index in range(window_count):
                start = window_index * self.stride
                target = start + self.window - 1

                if target >= common_length:
```

### Evidence: Physical Port Mask

Line 14 (context starts 11):

```python

Returned model/target dictionary:
    x                  float32 [16,58,32]
    physical_port_mask bool    [16,10]
    y_attack           float32 scalar
    y_attacker_count   int64   scalar
    y_source           float32 [16]
```

Line 50 (context starts 47):

```python

MODEL_ITEM_KEYS = {
    "x",
    "physical_port_mask",
    "y_attack",
    "y_attacker_count",
    "y_source",
```

Line 81 (context starts 78):

```python
        ) from exc


def derive_corrected_physical_port_mask(
    topology: dict[str, Any],
) -> torch.Tensor:
    """
```

Line 199 (context starts 196):

```python
        )
        if not isinstance(topology, dict):
            raise TypeError("topology.pt payload must be a dictionary")
        self.physical_port_mask = (
            derive_corrected_physical_port_mask(topology)
        )

```

Line 200 (context starts 197):

```python
        if not isinstance(topology, dict):
            raise TypeError("topology.pt payload must be a dictionary")
        self.physical_port_mask = (
            derive_corrected_physical_port_mask(topology)
        )

        self._index: list[IndexEntry] = []
```

Line 374 (context starts 371):

```python

        item = {
            "x": x,
            "physical_port_mask": (
                self.physical_port_mask.clone()
            ),
            "y_attack": run["y_attack"][entry.target].clone(),
```

Line 375 (context starts 372):

```python
        item = {
            "x": x,
            "physical_port_mask": (
                self.physical_port_mask.clone()
            ),
            "y_attack": run["y_attack"][entry.target].clone(),
            "y_attacker_count": (
```

## Source: `scripts/v5/p2/aggregate_v5_p2_task_d_full_multitask_matrix.py`

- SHA-256: `660288f0b1ce47fece4406410bec7ada922970da18dbdbca879bff745ce78c79`
- Batch keys detected: `[]`
- Output keys detected: `[]`

### Classes

```json
[]
```

### Functions

```json
[
  {
    "name": "sha256_file",
    "line": 21,
    "args": [
      "path"
    ]
  },
  {
    "name": "write_json",
    "line": 29,
    "args": [
      "path",
      "obj"
    ]
  },
  {
    "name": "write_csv",
    "line": 35,
    "args": [
      "path",
      "rows"
    ]
  },
  {
    "name": "mean_std",
    "line": 49,
    "args": [
      "values"
    ]
  },
  {
    "name": "main",
    "line": 54,
    "args": []
  }
]
```

### Evidence: Checkpoint Loading

Line 92 (context starts 89):

```python
            lock = json.loads(lock_path.read_text())
            if lock["report_sha256"] != sha256_file(report_path):
                raise RuntimeError(f"report SHA mismatch: {candidate} seed {seed}")
            checkpoint_path = Path(report["best"]["checkpoint_path"])
            if not checkpoint_path.is_file():
                raise RuntimeError(f"checkpoint missing: {checkpoint_path}")
            if report["best"]["checkpoint_sha256"] != sha256_file(checkpoint_path):
```

Line 93 (context starts 90):

```python
            if lock["report_sha256"] != sha256_file(report_path):
                raise RuntimeError(f"report SHA mismatch: {candidate} seed {seed}")
            checkpoint_path = Path(report["best"]["checkpoint_path"])
            if not checkpoint_path.is_file():
                raise RuntimeError(f"checkpoint missing: {checkpoint_path}")
            if report["best"]["checkpoint_sha256"] != sha256_file(checkpoint_path):
                raise RuntimeError(f"checkpoint SHA mismatch: {candidate} seed {seed}")
```

Line 94 (context starts 91):

```python
                raise RuntimeError(f"report SHA mismatch: {candidate} seed {seed}")
            checkpoint_path = Path(report["best"]["checkpoint_path"])
            if not checkpoint_path.is_file():
                raise RuntimeError(f"checkpoint missing: {checkpoint_path}")
            if report["best"]["checkpoint_sha256"] != sha256_file(checkpoint_path):
                raise RuntimeError(f"checkpoint SHA mismatch: {candidate} seed {seed}")
            if report["security_boundary"]["test_tensors_deserialized"] is not False:
```

Line 95 (context starts 92):

```python
            checkpoint_path = Path(report["best"]["checkpoint_path"])
            if not checkpoint_path.is_file():
                raise RuntimeError(f"checkpoint missing: {checkpoint_path}")
            if report["best"]["checkpoint_sha256"] != sha256_file(checkpoint_path):
                raise RuntimeError(f"checkpoint SHA mismatch: {candidate} seed {seed}")
            if report["security_boundary"]["test_tensors_deserialized"] is not False:
                raise RuntimeError("test tensor boundary violated")
```

Line 96 (context starts 93):

```python
            if not checkpoint_path.is_file():
                raise RuntimeError(f"checkpoint missing: {checkpoint_path}")
            if report["best"]["checkpoint_sha256"] != sha256_file(checkpoint_path):
                raise RuntimeError(f"checkpoint SHA mismatch: {candidate} seed {seed}")
            if report["security_boundary"]["test_tensors_deserialized"] is not False:
                raise RuntimeError("test tensor boundary violated")
            if report["security_boundary"]["threshold_tuning_performed"] is not False:
```

Line 128 (context starts 125):

```python
                "inference_microseconds_per_item": report["deployment_proxies"]["inference_microseconds_per_item"],
                "peak_cuda_memory_bytes": report["deployment_proxies"]["peak_cuda_memory_bytes"],
                "common_initial_state_sha256": report["architecture"]["common_initial_state_sha256"],
                "checkpoint_sha256": report["best"]["checkpoint_sha256"],
            }
            per_seed.append(row)
            common_hashes[seed][candidate] = row["common_initial_state_sha256"]
```

### Evidence: Validation Split

Line 2 (context starts 1):

```python
#!/usr/bin/env python3
"""Aggregate the frozen 2-candidate x 5-seed Task-D validation matrix."""
from __future__ import annotations

import argparse
```

Line 102 (context starts 99):

```python
            if report["security_boundary"]["threshold_tuning_performed"] is not False:
                raise RuntimeError("thresholds were tuned during Task-D training")

            m = report["best"]["validation_metrics"]
            row = {
                "candidate": candidate,
                "seed": seed,
```

Line 108 (context starts 105):

```python
                "seed": seed,
                "best_epoch": report["best"]["epoch"],
                "selection_score": m["selection_score"],
                "validation_loss": m["loss"],
                "graph_auroc": m["graph"]["auroc"],
                "graph_average_precision": m["graph"]["average_precision"],
                "graph_fixed_0_5_f1": m["graph"]["fixed_0_5"]["f1"],
```

Line 139 (context starts 136):

```python
            raise RuntimeError(f"common matched-seed initialization differs for seed {seed}")

    metric_names = [
        "selection_score", "validation_loss", "graph_auroc", "graph_average_precision",
        "graph_fixed_0_5_f1", "count_active_macro_f1", "count_active_accuracy",
        "source_average_precision", "transit_average_precision", "victim_average_precision",
        "path_average_precision", "source_fixed_0_5_f1", "transit_fixed_0_5_f1",
```

Line 191 (context starts 188):

```python
        "seeds": SEEDS,
        "summary": summary,
        "paired_seed_deltas": paired,
        "provisional_validation_selection_score_leader": leader,
        "architecture_selected": False,
        "selection_deferred_to": "V5_P2_G4_GRAPH_OPERATOR_STABILITY_AND_SELECTION",
        "threshold_tuning_performed": False,
```

## Source: `scripts/v5/p2/audit_v5_p2_a3_loader_contract.py`

- SHA-256: `09d646aa39d47b7e541d9cb5b167a49beda2dfa79bb27e935132e1156c9cbd86`
- Batch keys detected: `[]`
- Output keys detected: `[]`

### Classes

```json
[]
```

### Functions

```json
[
  {
    "name": "sha256_file",
    "line": 77,
    "args": [
      "path",
      "chunk_size"
    ]
  },
  {
    "name": "canonical_sha256",
    "line": 85,
    "args": [
      "value"
    ]
  },
  {
    "name": "atomic_write",
    "line": 94,
    "args": [
      "path",
      "text"
    ]
  },
  {
    "name": "write_json",
    "line": 100,
    "args": [
      "path",
      "value"
    ]
  },
  {
    "name": "load_json",
    "line": 104,
    "args": [
      "path"
    ]
  },
  {
    "name": "import_loader",
    "line": 108,
    "args": [
      "path"
    ]
  },
  {
    "name": "deterministic_pair_sample_bases",
    "line": 122,
    "args": [
      "dataset"
    ]
  },
  {
    "name": "check_scalar_tensor",
    "line": 135,
    "args": [
      "value",
      "dtype",
      "name",
      "failures",
      "context"
    ]
  },
  {
    "name": "check_node_tensor",
    "line": 155,
    "args": [
      "value",
      "allowed_dtypes",
      "name",
      "failures",
      "context"
    ]
  },
  {
    "name": "main",
    "line": 176,
    "args": []
  }
]
```

### Evidence: Checkpoint Loading

Line 380 (context starts 377):

```python
            f"expected {EXPECTED_TOTAL_ALIGNED_WINDOWS}"
        )

    stored_mask_payload = torch.load(
        paths["a2_r2_mask"],
        map_location="cpu",
        weights_only=False,
```

Line 563 (context starts 560):

```python
                    context,
                )

                raw = torch.load(
                    entry.file_path,
                    map_location="cpu",
                    weights_only=False,
```

### Evidence: Dataset Or Loader

Line 5 (context starts 2):

```python
"""
V5 P2-A3 Pair-Aligned PRIMARY58 Loader Contract Audit

Installs/audits the replacement P2 train/validation dataset loader.

The audit:
- verifies A0, A1-R2, and A2-R2 locks and hashes;
```

Line 9 (context starts 6):

```python

The audit:
- verifies A0, A1-R2, and A2-R2 locks and hashes;
- constructs TRAIN and VALIDATION datasets only;
- validates all pair-aligned index entries without loading test;
- verifies total aligned-window count = 82,694;
- checks pair adjacency and identical ATTACK/CONTROL starts;
```

Line 110 (context starts 107):

```python

def import_loader(path: Path):
    spec = importlib.util.spec_from_file_location(
        "v5_p2_pair_aligned_primary58_dataset",
        path,
    )
    if spec is None or spec.loader is None:
```

Line 122 (context starts 119):

```python
    return module


def deterministic_pair_sample_bases(dataset) -> list[int]:
    length = len(dataset)
    if length < 2 or length % 2:
        raise ValueError("dataset length is not positive and even")
```

Line 123 (context starts 120):

```python


def deterministic_pair_sample_bases(dataset) -> list[int]:
    length = len(dataset)
    if length < 2 or length % 2:
        raise ValueError("dataset length is not positive and even")

```

Line 125 (context starts 122):

```python
def deterministic_pair_sample_bases(dataset) -> list[int]:
    length = len(dataset)
    if length < 2 or length % 2:
        raise ValueError("dataset length is not positive and even")

    first = 0
    middle = (length // 2) // 2 * 2
```

Line 205 (context starts 202):

```python
    paths = {
        "a0_report": (
            a0_dir
            / "V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT.json"
        ),
        "a0_lock": (
            a0_dir
```

Line 209 (context starts 206):

```python
        ),
        "a0_lock": (
            a0_dir
            / "V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT_LOCK.json"
        ),
        "a1_r2_report": (
            a1_r2_dir
```

Line 299 (context starts 296):

```python
        failures.append("A2-R2 window/stride contract changed")
    if a2_lock.get("second_normalization_forbidden") is not True:
        failures.append("A2-R2 does not forbid second normalization")
    if a2_lock.get("reference_dataset_loader_authorized") is not False:
        failures.append("A2-R2 unexpectedly authorizes reference loader")
    if a2_lock.get("graph_message_passing") is not False:
        failures.append("A2-R2 graph-message-passing flag changed")
```

Line 335 (context starts 332):

```python
        return 1

    module = import_loader(loader_path)
    DatasetClass = module.V5P2PairAlignedPrimary58Dataset

    # Must reject test before any test filesystem interaction.
    test_rejected = False
```

Line 340 (context starts 337):

```python
    # Must reject test before any test filesystem interaction.
    test_rejected = False
    try:
        DatasetClass(
            root=root,
            split="test",
            pair_manifest=paths["pair_manifest"],
```

Line 355 (context starts 352):

```python
    else:
        failures.append("split='test' was not rejected")

    datasets = {}
    for split in ("train", "validation"):
        datasets[split] = DatasetClass(
            root=root,
```

Line 357 (context starts 354):

```python

    datasets = {}
    for split in ("train", "validation"):
        datasets[split] = DatasetClass(
            root=root,
            split=split,
            pair_manifest=paths["pair_manifest"],
```

Line 363 (context starts 360):

```python
            pair_manifest=paths["pair_manifest"],
        )

    if datasets["train"].pair_count != EXPECTED_TRAIN_PAIRS:
        failures.append(
            f"train pair_count={datasets['train'].pair_count}, expected 415"
        )
```

Line 365 (context starts 362):

```python

    if datasets["train"].pair_count != EXPECTED_TRAIN_PAIRS:
        failures.append(
            f"train pair_count={datasets['train'].pair_count}, expected 415"
        )
    if datasets["validation"].pair_count != EXPECTED_VALIDATION_PAIRS:
        failures.append(
```

Line 367 (context starts 364):

```python
        failures.append(
            f"train pair_count={datasets['train'].pair_count}, expected 415"
        )
    if datasets["validation"].pair_count != EXPECTED_VALIDATION_PAIRS:
        failures.append(
            "validation pair_count="
            f"{datasets['validation'].pair_count}, expected 74"
```

Line 370 (context starts 367):

```python
    if datasets["validation"].pair_count != EXPECTED_VALIDATION_PAIRS:
        failures.append(
            "validation pair_count="
            f"{datasets['validation'].pair_count}, expected 74"
        )

    total_dataset_windows = sum(len(dataset) for dataset in datasets.values())
```

Line 373 (context starts 370):

```python
            f"{datasets['validation'].pair_count}, expected 74"
        )

    total_dataset_windows = sum(len(dataset) for dataset in datasets.values())
    if total_dataset_windows != EXPECTED_TOTAL_ALIGNED_WINDOWS:
        failures.append(
            f"loader aligned windows={total_dataset_windows}, "
```

Line 374 (context starts 371):

```python
        )

    total_dataset_windows = sum(len(dataset) for dataset in datasets.values())
    if total_dataset_windows != EXPECTED_TOTAL_ALIGNED_WINDOWS:
        failures.append(
            f"loader aligned windows={total_dataset_windows}, "
            f"expected {EXPECTED_TOTAL_ALIGNED_WINDOWS}"
```

Line 376 (context starts 373):

```python
    total_dataset_windows = sum(len(dataset) for dataset in datasets.values())
    if total_dataset_windows != EXPECTED_TOTAL_ALIGNED_WINDOWS:
        failures.append(
            f"loader aligned windows={total_dataset_windows}, "
            f"expected {EXPECTED_TOTAL_ALIGNED_WINDOWS}"
        )

```

Line 401 (context starts 398):

```python
    split_index_summaries = {}
    sample_records = []

    for split, dataset in datasets.items():
        entries = dataset._index
        index_checks["total_entries"] += len(entries)

```

Line 402 (context starts 399):

```python
    sample_records = []

    for split, dataset in datasets.items():
        entries = dataset._index
        index_checks["total_entries"] += len(entries)

        if len(entries) % 2:
```

Line 406 (context starts 403):

```python
        index_checks["total_entries"] += len(entries)

        if len(entries) % 2:
            failures.append(f"{split} dataset length is odd")

        pair_start_groups = defaultdict(list)
        pair_counts = Counter()
```

Line 459 (context starts 456):

```python
                    f"{split}/{key}: pair members do not share indices"
                )

        for base in deterministic_pair_sample_bases(dataset):
            attack_entry = entries[base]
            control_entry = entries[base + 1]
            index_checks["paired_adjacency_checks"] += 1
```

Line 484 (context starts 481):

```python

            for item_index in (base, base + 1):
                entry = entries[item_index]
                item = dataset[item_index]
                context = (
                    f"{split}/{entry.pair_key}/"
                    f"{entry.mode}/start={entry.start}"
```

Line 621 (context starts 618):

```python
                )

        split_index_summaries[split] = {
            "dataset_length": len(dataset),
            "pair_count": dataset.pair_count,
            "pair_start_group_count": len(pair_start_groups),
            "unique_pair_count": len(pair_counts),
```

Line 622 (context starts 619):

```python

        split_index_summaries[split] = {
            "dataset_length": len(dataset),
            "pair_count": dataset.pair_count,
            "pair_start_group_count": len(pair_start_groups),
            "unique_pair_count": len(pair_counts),
            "minimum_entries_per_pair": min(
```

Line 642 (context starts 639):

```python
        "contract_version": 1,
        "authorized_splits": ["train", "validation"],
        "test_split_authorized": False,
        "dataset_class": "V5P2PairAlignedPrimary58Dataset",
        "index_policy": {
            "source": (
                "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
```

Line 654 (context starts 651):

```python
            "stride": 8,
            "common_length": "min(T_attack,T_control)",
            "identical_pair_member_starts": True,
            "total_non_test_items": total_dataset_windows,
        },
        "returned_item": {
            "keys": sorted(EXPECTED_ITEM_KEYS),
```

Line 708 (context starts 705):

```python

## Authorized loader

`V5P2PairAlignedPrimary58Dataset`

Authorized splits:

```

Line 759 (context starts 756):

```python
## Counts

```text
train pairs          = {datasets["train"].pair_count}
validation pairs     = {datasets["validation"].pair_count}
total aligned items  = {total_dataset_windows}
```
```

Line 760 (context starts 757):

```python

```text
train pairs          = {datasets["train"].pair_count}
validation pairs     = {datasets["validation"].pair_count}
total aligned items  = {total_dataset_windows}
```

```

Line 761 (context starts 758):

```python
```text
train pairs          = {datasets["train"].pair_count}
validation pairs     = {datasets["validation"].pair_count}
total aligned items  = {total_dataset_windows}
```

## Contract SHA-256
```

Line 787 (context starts 784):

```python
            if not failures
            else "BLOCK_P2_A4"
        ),
        "dataset": {
            "link_path": str(root_input.absolute()),
            "resolved_root": str(root),
        },
```

Line 794 (context starts 791):

```python
        "loader": {
            "path": str(loader_path),
            "sha256": sha256_file(loader_path),
            "class": "V5P2PairAlignedPrimary58Dataset",
        },
        "split_index_summaries": split_index_summaries,
        "index_checks": index_checks,
```

Line 830 (context starts 827):

```python
            "test_tensor_files_opened": False,
            "test_tensor_bytes_read": False,
            "test_tensor_contents_accessed": False,
            "test_dataset_constructed": False,
            "test_windows_constructed": False,
        },
        "failures": failures,
```

Line 863 (context starts 860):

```python
        "contract_file_sha256": sha256_file(contract_path),
        "contract_sha256": contract["contract_sha256"],
        "loader_sha256": sha256_file(loader_path),
        "train_pairs": datasets["train"].pair_count,
        "validation_pairs": datasets["validation"].pair_count,
        "train_items": len(datasets["train"]),
        "validation_items": len(datasets["validation"]),
```

Line 864 (context starts 861):

```python
        "contract_sha256": contract["contract_sha256"],
        "loader_sha256": sha256_file(loader_path),
        "train_pairs": datasets["train"].pair_count,
        "validation_pairs": datasets["validation"].pair_count,
        "train_items": len(datasets["train"]),
        "validation_items": len(datasets["validation"]),
        "total_aligned_items": total_dataset_windows,
```

Line 865 (context starts 862):

```python
        "loader_sha256": sha256_file(loader_path),
        "train_pairs": datasets["train"].pair_count,
        "validation_pairs": datasets["validation"].pair_count,
        "train_items": len(datasets["train"]),
        "validation_items": len(datasets["validation"]),
        "total_aligned_items": total_dataset_windows,
        "item_keys": sorted(EXPECTED_ITEM_KEYS),
```

Line 866 (context starts 863):

```python
        "train_pairs": datasets["train"].pair_count,
        "validation_pairs": datasets["validation"].pair_count,
        "train_items": len(datasets["train"]),
        "validation_items": len(datasets["validation"]),
        "total_aligned_items": total_dataset_windows,
        "item_keys": sorted(EXPECTED_ITEM_KEYS),
        "x_shape": [16, 58, 32],
```

_Additional matches omitted: 7_

### Evidence: Validation Split

Line 5 (context starts 2):

```python
"""
V5 P2-A3 Pair-Aligned PRIMARY58 Loader Contract Audit

Installs/audits the replacement P2 train/validation dataset loader.

The audit:
- verifies A0, A1-R2, and A2-R2 locks and hashes;
```

Line 9 (context starts 6):

```python

The audit:
- verifies A0, A1-R2, and A2-R2 locks and hashes;
- constructs TRAIN and VALIDATION datasets only;
- validates all pair-aligned index entries without loading test;
- verifies total aligned-window count = 82,694;
- checks pair adjacency and identical ATTACK/CONTROL starts;
```

Line 13 (context starts 10):

```python
- validates all pair-aligned index entries without loading test;
- verifies total aligned-window count = 82,694;
- checks pair adjacency and identical ATTACK/CONTROL starts;
- loads deterministic TRAIN/VALIDATION sample items;
- proves returned x is an exact PRIMARY58 slice of stored standardized x;
- proves targets come from the final epoch of each window;
- proves only the approved item keys are returned;
```

Line 44 (context starts 41):

```python
EXPECTED_TOTAL_ALIGNED_WINDOWS = 82694
EXPECTED_TOTAL_PAIRS = 489
EXPECTED_TRAIN_PAIRS = 415
EXPECTED_VALIDATION_PAIRS = 74

EXPECTED_ITEM_KEYS = {
    "x",
```

Line 242 (context starts 239):

```python
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    for split in ("train", "validation"):
        if not (root / "runs" / split).is_dir():
            failures.append(f"missing runs/{split}")

```

Line 356 (context starts 353):

```python
        failures.append("split='test' was not rejected")

    datasets = {}
    for split in ("train", "validation"):
        datasets[split] = DatasetClass(
            root=root,
            split=split,
```

Line 367 (context starts 364):

```python
        failures.append(
            f"train pair_count={datasets['train'].pair_count}, expected 415"
        )
    if datasets["validation"].pair_count != EXPECTED_VALIDATION_PAIRS:
        failures.append(
            "validation pair_count="
            f"{datasets['validation'].pair_count}, expected 74"
```

Line 369 (context starts 366):

```python
        )
    if datasets["validation"].pair_count != EXPECTED_VALIDATION_PAIRS:
        failures.append(
            "validation pair_count="
            f"{datasets['validation'].pair_count}, expected 74"
        )

```

Line 370 (context starts 367):

```python
    if datasets["validation"].pair_count != EXPECTED_VALIDATION_PAIRS:
        failures.append(
            "validation pair_count="
            f"{datasets['validation'].pair_count}, expected 74"
        )

    total_dataset_windows = sum(len(dataset) for dataset in datasets.values())
```

Line 417 (context starts 414):

```python
                )
            if entry.mode not in {"attack", "control"}:
                failures.append(
                    f"{split}/{entry.pair_key}: invalid mode {entry.mode}"
                )
            if entry.target != entry.start + 31:
                failures.append(
```

Line 640 (context starts 637):

```python
            "V5_P2_PAIR_ALIGNED_PRIMARY58_FROZEN_B3_LOADER"
        ),
        "contract_version": 1,
        "authorized_splits": ["train", "validation"],
        "test_split_authorized": False,
        "dataset_class": "V5P2PairAlignedPrimary58Dataset",
        "index_policy": {
```

Line 714 (context starts 711):

```python

```text
train
validation
```

The constructor rejects `test`.
```

Line 760 (context starts 757):

```python

```text
train pairs          = {datasets["train"].pair_count}
validation pairs     = {datasets["validation"].pair_count}
total aligned items  = {total_dataset_windows}
```

```

Line 822 (context starts 819):

```python
        },
        "security_boundary": {
            "train_tensor_contents_accessed": True,
            "validation_tensor_contents_accessed": True,
            "test_constructor_attempted": True,
            "test_constructor_rejected_before_filesystem_access": test_rejected,
            "test_directory_existence_checked": False,
```

Line 864 (context starts 861):

```python
        "contract_sha256": contract["contract_sha256"],
        "loader_sha256": sha256_file(loader_path),
        "train_pairs": datasets["train"].pair_count,
        "validation_pairs": datasets["validation"].pair_count,
        "train_items": len(datasets["train"]),
        "validation_items": len(datasets["validation"]),
        "total_aligned_items": total_dataset_windows,
```

Line 866 (context starts 863):

```python
        "train_pairs": datasets["train"].pair_count,
        "validation_pairs": datasets["validation"].pair_count,
        "train_items": len(datasets["train"]),
        "validation_items": len(datasets["validation"]),
        "total_aligned_items": total_dataset_windows,
        "item_keys": sorted(EXPECTED_ITEM_KEYS),
        "x_shape": [16, 58, 32],
```

Line 890 (context starts 887):

```python
    )
    print("loader_class: V5P2PairAlignedPrimary58Dataset")
    print("train_pairs:", datasets["train"].pair_count)
    print("validation_pairs:", datasets["validation"].pair_count)
    print("train_items:", len(datasets["train"]))
    print("validation_items:", len(datasets["validation"]))
    print("total_aligned_items:", total_dataset_windows)
```

Line 892 (context starts 889):

```python
    print("train_pairs:", datasets["train"].pair_count)
    print("validation_pairs:", datasets["validation"].pair_count)
    print("train_items:", len(datasets["train"]))
    print("validation_items:", len(datasets["validation"]))
    print("total_aligned_items:", total_dataset_windows)
    print("x_shape: [16, 58, 32]")
    print("x_dtype: float32")
```

### Evidence: Stable Ids

Line 64 (context starts 61):

```python
    "epoch_id",
    "case_id",
    "pair_id",
    "run_id",
    "filename",
    "mode",
    "split",
```

Line 295 (context starts 292):

```python
        failures.append("A1-R2 pair count is not 489")
    if a2_lock.get("primary58_count") != 58:
        failures.append("A2-R2 PRIMARY58 count is not 58")
    if a2_lock.get("window") != 32 or a2_lock.get("stride") != 8:
        failures.append("A2-R2 window/stride contract changed")
    if a2_lock.get("second_normalization_forbidden") is not True:
        failures.append("A2-R2 does not forbid second normalization")
```

Line 296 (context starts 293):

```python
    if a2_lock.get("primary58_count") != 58:
        failures.append("A2-R2 PRIMARY58 count is not 58")
    if a2_lock.get("window") != 32 or a2_lock.get("stride") != 8:
        failures.append("A2-R2 window/stride contract changed")
    if a2_lock.get("second_normalization_forbidden") is not True:
        failures.append("A2-R2 does not forbid second normalization")
    if a2_lock.get("reference_dataset_loader_authorized") is not False:
```

Line 866 (context starts 863):

```python
        "train_pairs": datasets["train"].pair_count,
        "validation_pairs": datasets["validation"].pair_count,
        "train_items": len(datasets["train"]),
        "validation_items": len(datasets["validation"]),
        "total_aligned_items": total_dataset_windows,
        "item_keys": sorted(EXPECTED_ITEM_KEYS),
        "x_shape": [16, 58, 32],
```

Line 892 (context starts 889):

```python
    print("train_pairs:", datasets["train"].pair_count)
    print("validation_pairs:", datasets["validation"].pair_count)
    print("train_items:", len(datasets["train"]))
    print("validation_items:", len(datasets["validation"]))
    print("total_aligned_items:", total_dataset_windows)
    print("x_shape: [16, 58, 32]")
    print("x_dtype: float32")
```

### Evidence: Physical Port Mask

Line 48 (context starts 45):

```python

EXPECTED_ITEM_KEYS = {
    "x",
    "physical_port_mask",
    "y_attack",
    "y_attacker_count",
    "y_source",
```

Line 233 (context starts 230):

```python
        ),
        "a2_r2_mask": (
            a2_r2_dir
            / "V5_P2_A2_R2_TOPOLOGY_DERIVED_BOOLEAN_PORT_MASK.pt"
        ),
        "loader": loader_path,
    }
```

Line 284 (context starts 281):

```python
        paths["pair_manifest"]
    ):
        failures.append("A1-R2 pair manifest SHA mismatch")
    if a2_lock.get("corrected_port_mask_sha256") != sha256_file(
        paths["a2_r2_mask"]
    ):
        failures.append("A2-R2 mask SHA mismatch")
```

Line 511 (context starts 508):

```python
                ):
                    failures.append(f"{context}: x contract failed")

                mask = item.get("physical_port_mask")
                if (
                    not isinstance(mask, torch.Tensor)
                    or mask.dtype != torch.bool
```

Line 518 (context starts 515):

```python
                    or tuple(mask.shape) != (16, 10)
                ):
                    failures.append(
                        f"{context}: physical_port_mask contract failed"
                    )
                elif isinstance(stored_mask, torch.Tensor) and not torch.equal(
                    mask,
```

Line 664 (context starts 661):

```python
                "exact stored standardized PRIMARY58 slice; "
                "no normalization performed"
            ),
            "physical_port_mask_shape": [16, 10],
            "physical_port_mask_dtype": "torch.bool",
            "target_epoch": "start + window - 1",
        },
```

Line 665 (context starts 662):

```python
                "no normalization performed"
            ),
            "physical_port_mask_shape": [16, 10],
            "physical_port_mask_dtype": "torch.bool",
            "target_epoch": "start + window - 1",
        },
        "model_input_boundary": {
```

Line 669 (context starts 666):

```python
            "target_epoch": "start + window - 1",
        },
        "model_input_boundary": {
            "model_inputs": ["x", "physical_port_mask"],
            "targets_only": [
                "y_attack",
                "y_attacker_count",
```

Line 734 (context starts 731):

```python

```text
x                  float32 [16,58,32]
physical_port_mask bool    [16,10]
y_attack           float32 scalar
y_attacker_count   int64   scalar
y_source           float32 [16]
```

Line 744 (context starts 741):

```python
role_mask          uint8/bool [16]
```

Only `x` and `physical_port_mask` are model inputs. The rest are targets or
loss masks.

The loader returns no identifiers, metadata, coordinates, edge index, lengths,
```

Line 896 (context starts 893):

```python
    print("total_aligned_items:", total_dataset_windows)
    print("x_shape: [16, 58, 32]")
    print("x_dtype: float32")
    print("physical_port_mask_shape: [16, 10]")
    print("physical_port_mask_dtype: bool")
    print("second_normalization_performed: false")
    print("edge_index_returned: false")
```

Line 897 (context starts 894):

```python
    print("x_shape: [16, 58, 32]")
    print("x_dtype: float32")
    print("physical_port_mask_shape: [16, 10]")
    print("physical_port_mask_dtype: bool")
    print("second_normalization_performed: false")
    print("edge_index_returned: false")
    print("metadata_returned: false")
```

## Source: `scripts/v5/p2/freeze_v5_p2_a1_r2_pair_aligned_window_contract.py`

- SHA-256: `ba4ca3305661d7f585d4b696945fb0b2dcccbc2ebdc02b3990356e7b25c73266`
- Batch keys detected: `[]`
- Output keys detected: `[]`

### Classes

```json
[]
```

### Functions

```json
[
  {
    "name": "sha256_file",
    "line": 48,
    "args": [
      "path",
      "chunk_size"
    ]
  },
  {
    "name": "canonical_sha256",
    "line": 56,
    "args": [
      "value"
    ]
  },
  {
    "name": "atomic_write",
    "line": 65,
    "args": [
      "path",
      "text"
    ]
  },
  {
    "name": "write_json",
    "line": 71,
    "args": [
      "path",
      "value"
    ]
  },
  {
    "name": "write_csv",
    "line": 75,
    "args": [
      "path",
      "rows"
    ]
  },
  {
    "name": "load_json",
    "line": 90,
    "args": [
      "path"
    ]
  },
  {
    "name": "load_csv",
    "line": 94,
    "args": [
      "path"
    ]
  },
  {
    "name": "as_int",
    "line": 99,
    "args": [
      "row",
      "key"
    ]
  },
  {
    "name": "window_count",
    "line": 109,
    "args": [
      "length"
    ]
  },
  {
    "name": "main",
    "line": 115,
    "args": []
  }
]
```

### Evidence: Dataset Or Loader

Line 142 (context starts 139):

```python
    paths = {
        "a0_report": (
            a0_dir
            / "V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT.json"
        ),
        "a0_lock": (
            a0_dir
```

Line 146 (context starts 143):

```python
        ),
        "a0_lock": (
            a0_dir
            / "V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT_LOCK.json"
        ),
        "a1_report": (
            a1_dir
```

Line 179 (context starts 176):

```python
            failures.append(f"missing prerequisite {name}: {path}")

    if not root.is_dir():
        failures.append(f"dataset root is missing: {root}")

    documents: dict[str, Any] = {}
    rows: list[dict[str, str]] = []
```

Line 594 (context starts 591):

```python
            if not failures
            else "BLOCK_A2"
        ),
        "dataset": {
            "link_path": str(root_input.absolute()),
            "resolved_root": str(root),
        },
```

Line 631 (context starts 628):

```python
            "test_tensor_files_opened": False,
            "test_tensor_bytes_read": False,
            "test_tensor_contents_accessed": False,
            "test_dataset_constructed": False,
            "test_windows_constructed": False,
        },
        "failures": failures,
```

### Evidence: Validation Split

Line 7 (context starts 4):

```python

Freezes the remediation required by A1-R1:

For every TRAIN or VALIDATION ATTACK/CONTROL pair:
    common_length = min(T_attack, T_control)

Both members use the identical ordered window starts:
```

Line 39 (context starts 36):

```python
EXPECTED = {
    "pair_count": 489,
    "train_pairs": 415,
    "validation_pairs": 74,
    "native_windows": 82708,
    "aligned_windows": 82694,
    "removed_windows": 14,
```

Line 289 (context starts 286):

```python
        pair_key = row.get("pair_key", "")
        identity = (split, pair_key)

        if split not in {"train", "validation"}:
            failures.append(
                f"unexpected split {split!r} for pair {pair_key!r}"
            )
```

Line 412 (context starts 409):

```python
    actuals = {
        "pair_count": len(manifest_rows),
        "train_pairs": split_counts["train"],
        "validation_pairs": split_counts["validation"],
        "native_windows": native_total,
        "aligned_windows": aligned_total,
        "removed_windows": removed_total,
```

Line 443 (context starts 440):

```python
        "contract_name": "V5_P2_PAIR_ALIGNED_COMMON_PREFIX_WINDOWS",
        "contract_version": 1,
        "scope": {
            "splits": ["train", "validation"],
            "test_excluded": True,
            "matched_pair_unit": "ATTACK_CONTROL_PAIR",
        },
```

Line 534 (context starts 531):

```python

## Decision

Use a common-prefix window index for every TRAIN and VALIDATION matched pair.

```text
common_length = min(T_attack, T_control)
```

Line 625 (context starts 622):

```python
        "security_boundary": {
            "run_tensors_deserialized": False,
            "train_tensor_contents_accessed": False,
            "validation_tensor_contents_accessed": False,
            "test_directory_existence_checked": False,
            "test_directory_enumerated": False,
            "test_tensor_files_opened": False,
```

Line 685 (context starts 682):

```python
    print("stride:", STRIDE)
    print("pair_count:", actuals["pair_count"])
    print("train_pairs:", actuals["train_pairs"])
    print("validation_pairs:", actuals["validation_pairs"])
    print(
        "unequal_length_pairs:",
        actuals["unequal_length_pairs"],
```

### Evidence: Stable Ids

Line 12 (context starts 9):

```python

Both members use the identical ordered window starts:
    0, stride, 2*stride, ...
up to the last full window inside common_length.

This stage reads only the A1-R1 report and pair CSV. It does not deserialize
any run tensor and does not enumerate or open runs/test.
```

Line 112 (context starts 109):

```python
def window_count(length: int) -> int:
    if length < WINDOW:
        return 0
    return 1 + (length - WINDOW) // STRIDE


def main() -> int:
```

Line 276 (context starts 273):

```python
        )

    seen: set[tuple[str, str]] = set()
    manifest_rows: list[dict[str, Any]] = []
    split_counts = Counter()
    unequal_length_count = 0
    unequal_window_count = 0
```

Line 376 (context starts 373):

```python
        removed_total += attack_removed + control_removed

        last_start = (
            (common_windows - 1) * STRIDE
            if common_windows > 0
            else None
        )
```

Line 386 (context starts 383):

```python
            else None
        )

        manifest_rows.append(
            {
                "split": split,
                "pair_key": pair_key,
```

Line 410 (context starts 407):

```python
        )

    actuals = {
        "pair_count": len(manifest_rows),
        "train_pairs": split_counts["train"],
        "validation_pairs": split_counts["validation"],
        "native_windows": native_total,
```

Line 430 (context starts 427):

```python
            "native-aligned total does not equal removed total"
        )

    manifest_rows.sort(
        key=lambda item: (item["split"], item["pair_key"])
    )
    manifest_path = (
```

Line 437 (context starts 434):

```python
        output_dir
        / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
    )
    write_csv(manifest_path, manifest_rows)

    contract = {
        "contract_name": "V5_P2_PAIR_ALIGNED_COMMON_PREFIX_WINDOWS",
```

Line 453 (context starts 450):

```python
            "common_length_rule": "min(T_attack,T_control)",
            "window_start_rule": (
                "start_k = k*stride for "
                "k=0..floor((common_length-window)/stride)"
            ),
            "window_interval": "[start_k,start_k+window)",
            "identical_ordered_window_starts_for_pair_members": True,
```

Line 457 (context starts 454):

```python
            ),
            "window_interval": "[start_k,start_k+window)",
            "identical_ordered_window_starts_for_pair_members": True,
            "native_unaligned_windowing_forbidden": True,
            "source_tensors_modified": False,
            "source_tensors_truncated_on_disk": False,
            "loader_slices_common_prefix_only": True,
```

Line 471 (context starts 468):

```python
        "learned_input_prohibitions": [
            "case_id",
            "pair_id",
            "run_id",
            "filename",
            "split",
            "mode",
```

Line 534 (context starts 531):

```python

## Decision

Use a common-prefix window index for every TRAIN and VALIDATION matched pair.

```text
common_length = min(T_attack, T_control)
```

## Source: `scripts/v5/p2/preflight_v5_p2_task_d_full_multitask.py`

- SHA-256: `b7ef6c9a522ec6fd122b16fc17f220cc413f6e7e24333778d3a6885dbc347a42`
- Batch keys detected: `[]`
- Output keys detected: `[]`

### Classes

```json
[]
```

### Functions

```json
[
  {
    "name": "sha256_file",
    "line": 26,
    "args": [
      "path"
    ]
  },
  {
    "name": "sha256_state_dict",
    "line": 34,
    "args": [
      "state"
    ]
  },
  {
    "name": "import_module",
    "line": 41,
    "args": [
      "path",
      "name"
    ]
  },
  {
    "name": "write_json",
    "line": 50,
    "args": [
      "path",
      "obj"
    ]
  },
  {
    "name": "set_seed",
    "line": 56,
    "args": [
      "seed"
    ]
  },
  {
    "name": "validate_edge_index",
    "line": 67,
    "args": [
      "edge"
    ]
  },
  {
    "name": "main",
    "line": 85,
    "args": []
  }
]
```

### Evidence: Model Construction

Line 166 (context starts 163):

```python
        conv_reference = b3_mod.P2B3Conv1DOnlyCount4()
        set_seed(107)
        graph_reference = b3_mod.P2B3Conv1DOnlyCount4()
        graph_model = td_mod.P2TaskDGraphConvCount4(graph_reference, edge)
        conv_hash = sha256_state_dict(conv_reference.state_dict())
        graph_common_hash = sha256_state_dict(td_mod.common_state_dict(graph_model))
        if conv_hash != graph_common_hash:
```

### Evidence: Checkpoint Loading

Line 34 (context starts 31):

```python
    return h.hexdigest()


def sha256_state_dict(state: dict[str, torch.Tensor]) -> str:
    buf = io.BytesIO()
    cpu = OrderedDict((k, v.detach().cpu().contiguous()) for k, v in state.items())
    torch.save(cpu, buf)
```

Line 167 (context starts 164):

```python
        set_seed(107)
        graph_reference = b3_mod.P2B3Conv1DOnlyCount4()
        graph_model = td_mod.P2TaskDGraphConvCount4(graph_reference, edge)
        conv_hash = sha256_state_dict(conv_reference.state_dict())
        graph_common_hash = sha256_state_dict(td_mod.common_state_dict(graph_model))
        if conv_hash != graph_common_hash:
            raise RuntimeError("matched-seed common initialization is not identical")
```

Line 168 (context starts 165):

```python
        graph_reference = b3_mod.P2B3Conv1DOnlyCount4()
        graph_model = td_mod.P2TaskDGraphConvCount4(graph_reference, edge)
        conv_hash = sha256_state_dict(conv_reference.state_dict())
        graph_common_hash = sha256_state_dict(td_mod.common_state_dict(graph_model))
        if conv_hash != graph_common_hash:
            raise RuntimeError("matched-seed common initialization is not identical")

```

Line 175 (context starts 172):

```python
        # Independent duplicate B3 instantiation must be exact under the seed.
        set_seed(107)
        duplicate_b3 = b3_mod.P2B3Conv1DOnlyCount4()
        if sha256_state_dict(duplicate_b3.state_dict()) != conv_hash:
            raise RuntimeError("B3 deterministic initialization failed")

        batch_size = 4
```

### Evidence: Dataset Or Loader

Line 111 (context starts 108):

```python
        "b3_model": repo / "src/models/v5_p2_b3_conv1d_only_count4.py",
        "task_d_model": repo / "src/models/v5_p2_task_d_full_multitask_count4.py",
        "b2_train": repo / "scripts/v5/p2/train_v5_p2_b2_single_seed.py",
        "loader": repo / "src/data/v5_p2_pair_aligned_primary58_dataset.py",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    failures: list[str] = []
```

### Evidence: Physical Port Mask

Line 183 (context starts 180):

```python
        mask = torch.ones(batch_size, 16, 10, dtype=torch.bool, device=device)
        batch = {
            "x": x,
            "physical_port_mask": mask,
            "y_attack": torch.tensor([0, 1, 1, 1], dtype=torch.float32, device=device),
            "y_attacker_count": torch.tensor([0, 1, 2, 4], dtype=torch.int64, device=device),
            "y_source": torch.randint(0, 2, (batch_size, 16), device=device),
```

## Source: `scripts/v5/p2/run_v5_p2_a4_loader_integration_tiny_overfit.py`

- SHA-256: `990bfefeaaa7a87607a13c99cd273734c941c6bbb9403fdc9c31deca4b154d4a`
- Batch keys detected: `['physical_port_mask', 'x', 'y_attack', 'y_attack_path', 'y_attacker_count', 'y_source', 'y_transit', 'y_victim']`
- Output keys detected: `['attack_logits', 'count_logits', 'path_logits', 'source_logits', 'transit_logits', 'victim_logits']`

### Classes

```json
[]
```

### Functions

```json
[
  {
    "name": "sha256_file",
    "line": 65,
    "args": [
      "path",
      "chunk_size"
    ]
  },
  {
    "name": "canonical_sha256",
    "line": 73,
    "args": [
      "value"
    ]
  },
  {
    "name": "atomic_write",
    "line": 82,
    "args": [
      "path",
      "text"
    ]
  },
  {
    "name": "write_json",
    "line": 88,
    "args": [
      "path",
      "value"
    ]
  },
  {
    "name": "write_csv",
    "line": 92,
    "args": [
      "path",
      "rows"
    ]
  },
  {
    "name": "load_json",
    "line": 107,
    "args": [
      "path"
    ]
  },
  {
    "name": "import_module",
    "line": 111,
    "args": [
      "path",
      "module_name"
    ]
  },
  {
    "name": "set_seed",
    "line": 122,
    "args": [
      "seed"
    ]
  },
  {
    "name": "category_from_pair_key",
    "line": 134,
    "args": [
      "pair_key"
    ]
  },
  {
    "name": "positive_weight",
    "line": 141,
    "args": [
      "target"
    ]
  },
  {
    "name": "exact_binary_metrics",
    "line": 156,
    "args": [
      "logits",
      "target"
    ]
  },
  {
    "name": "evaluate",
    "line": 178,
    "args": [
      "outputs",
      "batch",
      "count_targets",
      "active_mask"
    ]
  },
  {
    "name": "main",
    "line": 231,
    "args": []
  }
]
```

### Evidence: Model Construction

Line 608 (context starts 605):

```python
    if bool((count_targets[active_mask] < 0).any().item()):
        failures.append("one or more active count targets were not mapped")

    model = ModelClass().to(device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
```

### Evidence: Dataset Or Loader

Line 13 (context starts 10):

```python
- selects two active matched pair-windows from each positive attacker-count
  category exposed by the train split;
- includes each matched CONTROL item at the identical window start;
- batches the 12 selected items through torch DataLoader;
- instantiates the frozen 43,208-parameter B3 Conv1D-only architecture;
- trains only on this tiny fixed batch until it memorizes all graph, count,
  source, transit, victim, and path targets;
```

Line 44 (context starts 41):

```python
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


STAGE = "V5_P2_A4_LOADER_INTEGRATION_AND_TINY_OVERFIT"
```

Line 381 (context starts 378):

```python

    loader_module = import_module(
        loader_path,
        "v5_p2_pair_aligned_primary58_dataset_a4",
    )
    model_module = import_module(
        model_path,
```

Line 388 (context starts 385):

```python
        "v5_frozen_b3_conv1d_only_a4",
    )

    DatasetClass = loader_module.V5P2PairAlignedPrimary58Dataset
    ModelClass = model_module.FrozenB3Conv1DOnly

    train_dataset = DatasetClass(
```

Line 391 (context starts 388):

```python
    DatasetClass = loader_module.V5P2PairAlignedPrimary58Dataset
    ModelClass = model_module.FrozenB3Conv1DOnly

    train_dataset = DatasetClass(
        root=root,
        split="train",
        pair_manifest=paths["pair_manifest"],
```

Line 399 (context starts 396):

```python

    # Locate the final aligned ATTACK/CONTROL start for each pair.
    last_pair_base: dict[str, int] = {}
    for base in range(0, len(train_dataset._index), 2):
        attack_entry = train_dataset._index[base]
        control_entry = train_dataset._index[base + 1]
        if (
```

Line 400 (context starts 397):

```python
    # Locate the final aligned ATTACK/CONTROL start for each pair.
    last_pair_base: dict[str, int] = {}
    for base in range(0, len(train_dataset._index), 2):
        attack_entry = train_dataset._index[base]
        control_entry = train_dataset._index[base + 1]
        if (
            attack_entry.mode != "attack"
```

Line 401 (context starts 398):

```python
    last_pair_base: dict[str, int] = {}
    for base in range(0, len(train_dataset._index), 2):
        attack_entry = train_dataset._index[base]
        control_entry = train_dataset._index[base + 1]
        if (
            attack_entry.mode != "attack"
            or control_entry.mode != "control"
```

Line 425 (context starts 422):

```python
            continue

        base = last_pair_base[pair_key]
        attack_item = train_dataset[base]
        control_item = train_dataset[base + 1]

        attack_label = float(attack_item["y_attack"].item())
```

Line 426 (context starts 423):

```python

        base = last_pair_base[pair_key]
        attack_item = train_dataset[base]
        control_item = train_dataset[base + 1]

        attack_label = float(attack_item["y_attack"].item())
        control_label = float(control_item["y_attack"].item())
```

Line 450 (context starts 447):

```python
            )
            continue

        entry = train_dataset._index[base]
        selected_by_category[category].append(
            {
                "category": category,
```

Line 537 (context starts 534):

```python
            print("FAIL:", failure)
        return 1

    subset = Subset(train_dataset, selected_indices)
    batch_loader = DataLoader(
        subset,
        batch_size=len(subset),
```

Line 538 (context starts 535):

```python
        return 1

    subset = Subset(train_dataset, selected_indices)
    batch_loader = DataLoader(
        subset,
        batch_size=len(subset),
        shuffle=False,
```

Line 923 (context starts 920):

```python
                "K1/K2/K4 category plus matched controls"
            ),
            "selected_pairs": selected_pairs,
            "selected_dataset_indices": selected_indices,
            "positive_raw_attacker_counts": positive_raw_counts,
            "count_class_mapping": {
                str(key): value
```

Line 955 (context starts 952):

```python
            "loader/model/loss integration test only; "
            "not final P2 training"
        ),
        "dataset": {
            "link_path": str(root_input.absolute()),
            "resolved_root": str(root),
        },
```

Line 1046 (context starts 1043):

```python
            "test_tensor_files_opened": False,
            "test_tensor_bytes_read": False,
            "test_tensor_contents_accessed": False,
            "test_dataset_constructed": False,
            "test_windows_constructed": False,
        },
        "failures": failures,
```

### Evidence: Validation Split

Line 22 (context starts 19):

```python
The resulting weights are disposable audit weights and are not saved or
authorized for any experiment.

No validation tensor is opened and no test directory is enumerated or opened.
"""

from __future__ import annotations
```

Line 306 (context starts 303):

```python
            "failures": failures,
            "warnings": warnings,
            "training_performed": False,
            "validation_tensor_contents_accessed": False,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
```

Line 366 (context starts 363):

```python
            "failures": failures,
            "warnings": warnings,
            "training_performed": False,
            "validation_tensor_contents_accessed": False,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
```

Line 526 (context starts 523):

```python
            "warnings": warnings,
            "selection": selected_pairs,
            "training_performed": False,
            "validation_tensor_contents_accessed": False,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
```

Line 635 (context starts 632):

```python
            "warnings": warnings,
            "selection": selected_pairs,
            "training_performed": False,
            "validation_tensor_contents_accessed": False,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
```

Line 1040 (context starts 1037):

```python
            "final_experiment_training_performed": False,
            "audit_weights_saved": False,
            "train_tensor_contents_accessed": True,
            "validation_tensor_contents_accessed": False,
            "test_directory_existence_checked": False,
            "test_directory_enumerated": False,
            "test_tensor_files_opened": False,
```

Line 1104 (context starts 1101):

```python
        "stable_exact_steps": stable_exact_steps,
        "audit_weights_saved": False,
        "audit_weights_authorized_for_reuse": False,
        "validation_tensor_contents_accessed": False,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "next_stage": (
```

Line 1144 (context starts 1141):

```python
    print("stable_exact_steps:", stable_exact_steps)
    print("audit_weights_saved: false")
    print("audit_weights_authorized_for_reuse: false")
    print("validation_tensor_contents_accessed: false")
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")
    print("failure_count:", len(failures))
```

### Evidence: Batch Keys

Line 186 (context starts 183):

```python
) -> dict[str, Any]:
    graph = exact_binary_metrics(
        outputs["attack_logits"],
        batch["y_attack"],
    )

    active_count_prediction = outputs["count_logits"][
```

Line 563 (context starts 560):

```python
            f"collated batch keys={sorted(batch)}, "
            f"expected {sorted(expected_batch_keys)}"
        )
    if tuple(batch["x"].shape) != (12, 16, 58, 32):
        failures.append(
            f"collated x shape={tuple(batch['x'].shape)}"
        )
```

Line 565 (context starts 562):

```python
        )
    if tuple(batch["x"].shape) != (12, 16, 58, 32):
        failures.append(
            f"collated x shape={tuple(batch['x'].shape)}"
        )
    if batch["x"].dtype != torch.float32:
        failures.append(
```

Line 567 (context starts 564):

```python
        failures.append(
            f"collated x shape={tuple(batch['x'].shape)}"
        )
    if batch["x"].dtype != torch.float32:
        failures.append(
            f"collated x dtype={batch['x'].dtype}"
        )
```

Line 569 (context starts 566):

```python
        )
    if batch["x"].dtype != torch.float32:
        failures.append(
            f"collated x dtype={batch['x'].dtype}"
        )
    if tuple(batch["physical_port_mask"].shape) != (12, 16, 10):
        failures.append(
```

Line 571 (context starts 568):

```python
        failures.append(
            f"collated x dtype={batch['x'].dtype}"
        )
    if tuple(batch["physical_port_mask"].shape) != (12, 16, 10):
        failures.append(
            "collated physical-port-mask shape mismatch"
        )
```

Line 575 (context starts 572):

```python
        failures.append(
            "collated physical-port-mask shape mismatch"
        )
    if batch["physical_port_mask"].dtype != torch.bool:
        failures.append(
            "collated physical-port-mask dtype mismatch"
        )
```

Line 588 (context starts 585):

```python
        for key, value in batch.items()
    }

    active_mask = batch["y_attack"] >= 0.5
    if int(active_mask.sum().item()) != 6:
        failures.append(
            f"active attack item count={int(active_mask.sum())}, expected 6"
```

Line 595 (context starts 592):

```python
        )

    count_targets = torch.full(
        batch["y_attacker_count"].shape,
        fill_value=-100,
        dtype=torch.int64,
        device=device,
```

Line 603 (context starts 600):

```python
    for raw_count, class_index in count_class_mapping.items():
        count_targets[
            active_mask
            & (batch["y_attacker_count"] == raw_count)
        ] = class_index
    if bool((count_targets[active_mask] < 0).any().item()):
        failures.append("one or more active count targets were not mapped")
```

Line 646 (context starts 643):

```python
            print("FAIL:", failure)
        return 1

    graph_pos_weight = positive_weight(batch["y_attack"])
    role_pos_weights = {
        "source": positive_weight(batch["y_source"]),
        "transit": positive_weight(batch["y_transit"]),
```

Line 648 (context starts 645):

```python

    graph_pos_weight = positive_weight(batch["y_attack"])
    role_pos_weights = {
        "source": positive_weight(batch["y_source"]),
        "transit": positive_weight(batch["y_transit"]),
        "victim": positive_weight(batch["y_victim"]),
        "path": positive_weight(batch["y_attack_path"]),
```

Line 649 (context starts 646):

```python
    graph_pos_weight = positive_weight(batch["y_attack"])
    role_pos_weights = {
        "source": positive_weight(batch["y_source"]),
        "transit": positive_weight(batch["y_transit"]),
        "victim": positive_weight(batch["y_victim"]),
        "path": positive_weight(batch["y_attack_path"]),
    }
```

Line 650 (context starts 647):

```python
    role_pos_weights = {
        "source": positive_weight(batch["y_source"]),
        "transit": positive_weight(batch["y_transit"]),
        "victim": positive_weight(batch["y_victim"]),
        "path": positive_weight(batch["y_attack_path"]),
    }

```

Line 651 (context starts 648):

```python
        "source": positive_weight(batch["y_source"]),
        "transit": positive_weight(batch["y_transit"]),
        "victim": positive_weight(batch["y_victim"]),
        "path": positive_weight(batch["y_attack_path"]),
    }

    optimizer = torch.optim.Adam(
```

Line 673 (context starts 670):

```python
        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            batch["x"],
            batch["physical_port_mask"],
        )

```

Line 674 (context starts 671):

```python

        outputs = model(
            batch["x"],
            batch["physical_port_mask"],
        )

        attack_loss = F.binary_cross_entropy_with_logits(
```

Line 679 (context starts 676):

```python

        attack_loss = F.binary_cross_entropy_with_logits(
            outputs["attack_logits"],
            batch["y_attack"].float(),
            pos_weight=graph_pos_weight,
        )
        count_loss = F.cross_entropy(
```

Line 688 (context starts 685):

```python
        )
        source_loss = F.binary_cross_entropy_with_logits(
            outputs["source_logits"],
            batch["y_source"].float(),
            pos_weight=role_pos_weights["source"],
        )
        transit_loss = F.binary_cross_entropy_with_logits(
```

Line 693 (context starts 690):

```python
        )
        transit_loss = F.binary_cross_entropy_with_logits(
            outputs["transit_logits"],
            batch["y_transit"].float(),
            pos_weight=role_pos_weights["transit"],
        )
        victim_loss = F.binary_cross_entropy_with_logits(
```

Line 698 (context starts 695):

```python
        )
        victim_loss = F.binary_cross_entropy_with_logits(
            outputs["victim_logits"],
            batch["y_victim"].float(),
            pos_weight=role_pos_weights["victim"],
        )
        path_loss = F.binary_cross_entropy_with_logits(
```

Line 703 (context starts 700):

```python
        )
        path_loss = F.binary_cross_entropy_with_logits(
            outputs["path_logits"],
            batch["y_attack_path"].float(),
            pos_weight=role_pos_weights["path"],
        )

```

Line 754 (context starts 751):

```python
        model.eval()
        with torch.no_grad():
            current_outputs = model(
                batch["x"],
                batch["physical_port_mask"],
            )
            metrics = evaluate(
```

Line 755 (context starts 752):

```python
        with torch.no_grad():
            current_outputs = model(
                batch["x"],
                batch["physical_port_mask"],
            )
            metrics = evaluate(
                current_outputs,
```

Line 769 (context starts 766):

```python
                LOSS_WEIGHTS["attack"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["attack_logits"],
                    batch["y_attack"].float(),
                    pos_weight=graph_pos_weight,
                )
                + LOSS_WEIGHTS["count"]
```

Line 780 (context starts 777):

```python
                + LOSS_WEIGHTS["source"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["source_logits"],
                    batch["y_source"].float(),
                    pos_weight=role_pos_weights["source"],
                )
                + LOSS_WEIGHTS["transit"]
```

Line 786 (context starts 783):

```python
                + LOSS_WEIGHTS["transit"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["transit_logits"],
                    batch["y_transit"].float(),
                    pos_weight=role_pos_weights["transit"],
                )
                + LOSS_WEIGHTS["victim"]
```

Line 792 (context starts 789):

```python
                + LOSS_WEIGHTS["victim"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["victim_logits"],
                    batch["y_victim"].float(),
                    pos_weight=role_pos_weights["victim"],
                )
                + LOSS_WEIGHTS["path"]
```

Line 798 (context starts 795):

```python
                + LOSS_WEIGHTS["path"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["path_logits"],
                    batch["y_attack_path"].float(),
                    pos_weight=role_pos_weights["path"],
                )
            )
```

### Evidence: Output Keys

Line 185 (context starts 182):

```python
    active_mask: torch.Tensor,
) -> dict[str, Any]:
    graph = exact_binary_metrics(
        outputs["attack_logits"],
        batch["y_attack"],
    )

```

Line 189 (context starts 186):

```python
        batch["y_attack"],
    )

    active_count_prediction = outputs["count_logits"][
        active_mask
    ].argmax(dim=-1)
    active_count_truth = count_targets[active_mask]
```

Line 678 (context starts 675):

```python
        )

        attack_loss = F.binary_cross_entropy_with_logits(
            outputs["attack_logits"],
            batch["y_attack"].float(),
            pos_weight=graph_pos_weight,
        )
```

Line 683 (context starts 680):

```python
            pos_weight=graph_pos_weight,
        )
        count_loss = F.cross_entropy(
            outputs["count_logits"][active_mask],
            count_targets[active_mask],
        )
        source_loss = F.binary_cross_entropy_with_logits(
```

Line 687 (context starts 684):

```python
            count_targets[active_mask],
        )
        source_loss = F.binary_cross_entropy_with_logits(
            outputs["source_logits"],
            batch["y_source"].float(),
            pos_weight=role_pos_weights["source"],
        )
```

Line 692 (context starts 689):

```python
            pos_weight=role_pos_weights["source"],
        )
        transit_loss = F.binary_cross_entropy_with_logits(
            outputs["transit_logits"],
            batch["y_transit"].float(),
            pos_weight=role_pos_weights["transit"],
        )
```

Line 697 (context starts 694):

```python
            pos_weight=role_pos_weights["transit"],
        )
        victim_loss = F.binary_cross_entropy_with_logits(
            outputs["victim_logits"],
            batch["y_victim"].float(),
            pos_weight=role_pos_weights["victim"],
        )
```

Line 702 (context starts 699):

```python
            pos_weight=role_pos_weights["victim"],
        )
        path_loss = F.binary_cross_entropy_with_logits(
            outputs["path_logits"],
            batch["y_attack_path"].float(),
            pos_weight=role_pos_weights["path"],
        )
```

Line 768 (context starts 765):

```python
            current_loss = (
                LOSS_WEIGHTS["attack"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["attack_logits"],
                    batch["y_attack"].float(),
                    pos_weight=graph_pos_weight,
                )
```

Line 774 (context starts 771):

```python
                )
                + LOSS_WEIGHTS["count"]
                * F.cross_entropy(
                    current_outputs["count_logits"][active_mask],
                    count_targets[active_mask],
                )
                + LOSS_WEIGHTS["source"]
```

Line 779 (context starts 776):

```python
                )
                + LOSS_WEIGHTS["source"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["source_logits"],
                    batch["y_source"].float(),
                    pos_weight=role_pos_weights["source"],
                )
```

Line 785 (context starts 782):

```python
                )
                + LOSS_WEIGHTS["transit"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["transit_logits"],
                    batch["y_transit"].float(),
                    pos_weight=role_pos_weights["transit"],
                )
```

Line 791 (context starts 788):

```python
                )
                + LOSS_WEIGHTS["victim"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["victim_logits"],
                    batch["y_victim"].float(),
                    pos_weight=role_pos_weights["victim"],
                )
```

Line 797 (context starts 794):

```python
                )
                + LOSS_WEIGHTS["path"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["path_logits"],
                    batch["y_attack_path"].float(),
                    pos_weight=role_pos_weights["path"],
                )
```

### Evidence: Stable Ids

Line 12 (context starts 9):

```python
- constructs the audited TRAIN loader only;
- selects two active matched pair-windows from each positive attacker-count
  category exposed by the train split;
- includes each matched CONTROL item at the identical window start;
- batches the 12 selected items through torch DataLoader;
- instantiates the frozen 43,208-parameter B3 Conv1D-only architecture;
- trains only on this tiny fixed batch until it memorizes all graph, count,
```

### Evidence: Physical Port Mask

Line 549 (context starts 546):

```python

    expected_batch_keys = {
        "x",
        "physical_port_mask",
        "y_attack",
        "y_attacker_count",
        "y_source",
```

Line 571 (context starts 568):

```python
        failures.append(
            f"collated x dtype={batch['x'].dtype}"
        )
    if tuple(batch["physical_port_mask"].shape) != (12, 16, 10):
        failures.append(
            "collated physical-port-mask shape mismatch"
        )
```

Line 575 (context starts 572):

```python
        failures.append(
            "collated physical-port-mask shape mismatch"
        )
    if batch["physical_port_mask"].dtype != torch.bool:
        failures.append(
            "collated physical-port-mask dtype mismatch"
        )
```

Line 674 (context starts 671):

```python

        outputs = model(
            batch["x"],
            batch["physical_port_mask"],
        )

        attack_loss = F.binary_cross_entropy_with_logits(
```

Line 755 (context starts 752):

```python
        with torch.no_grad():
            current_outputs = model(
                batch["x"],
                batch["physical_port_mask"],
            )
            metrics = evaluate(
                current_outputs,
```

Line 929 (context starts 926):

```python
                str(key): value
                for key, value in count_class_mapping.items()
            },
            "model_inputs": ["x", "physical_port_mask"],
            "targets": [
                "y_attack",
                "y_attacker_count",
```

Line 968 (context starts 965):

```python
            "graph_message_passing": False,
            "edge_index_model_input": False,
            "x_shape": [12, 16, 58, 32],
            "physical_port_mask_shape": [12, 16, 10],
        },
        "tiny_subset": {
            "item_count": len(selected_indices),
```

## Required immutable E1 arrays

```json
{
  "count_logits": {
    "dtype": "float32",
    "shape": "[N,4]"
  },
  "graph_logits": {
    "dtype": "float32",
    "shape": "[N]"
  },
  "manifest_row_index": {
    "dtype": "int64",
    "shape": "[N]"
  },
  "path_logits": {
    "dtype": "float32",
    "shape": "[N,16]"
  },
  "source_logits": {
    "dtype": "float32",
    "shape": "[N,16]"
  },
  "stable_item_id": {
    "dtype": "fixed-width Unicode or UTF-8 JSONL",
    "shape": "[N]"
  },
  "transit_logits": {
    "dtype": "float32",
    "shape": "[N,16]"
  },
  "victim_logits": {
    "dtype": "float32",
    "shape": "[N,16]"
  },
  "y_attacker_count": {
    "dtype": "int64",
    "shape": "[N]"
  },
  "y_graph": {
    "dtype": "uint8",
    "shape": "[N]"
  },
  "y_path": {
    "dtype": "uint8",
    "shape": "[N,16]"
  },
  "y_source": {
    "dtype": "uint8",
    "shape": "[N,16]"
  },
  "y_transit": {
    "dtype": "uint8",
    "shape": "[N,16]"
  },
  "y_victim": {
    "dtype": "uint8",
    "shape": "[N,16]"
  }
}
```

## Current state

- E1 exporter implemented: **false**
- Validation predictions exported: **false**
- Raw thresholds frozen: **false**
- Decoder selected/frozen: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
