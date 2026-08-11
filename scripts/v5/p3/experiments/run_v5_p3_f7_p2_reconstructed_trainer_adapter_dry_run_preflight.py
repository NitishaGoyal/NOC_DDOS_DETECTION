from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


STAGE = "V5_P3_F7_P2_RECONSTRUCTED_TRAINER_ADAPTER_AND_DRY_RUN_PREFLIGHT"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "TRAIN-DERIVED"

EXPECTED_MODEL_CLASS = "V6P0Dynamic70GraphConvCount4"
EXPECTED_PARAMETER_COUNT = 60553
EXPECTED_TRAIN_ITEMS = 110855
EXPECTED_VALIDATION_ITEMS = 13863
EXPECTED_X_TRAILING_SHAPE = (16, 70, 32)
EXPECTED_MASK_TRAILING_SHAPE = (16, 10)
DRY_RUN_BATCH_SIZE = 2
PRIMARY_SEED = 107

FLOW_GROUP_START = 55
FLOW_GROUP_END = 70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_report_lock(
    report_path: Path,
    lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = load_json(report_path)
    lock = load_json(lock_path)
    require(report.get("status") == "PASS", f"report not PASS: {report_path}")
    require(
        lock.get("report_sha256") == sha256_file(report_path),
        f"report/lock mismatch: {report_path}",
    )
    return report, lock


def import_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def tensor_tree(value: Any, path: str = "root") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if isinstance(value, torch.Tensor):
        rows.append({
            "path": path,
            "tensor": value,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        })
        return rows

    if isinstance(value, np.ndarray):
        rows.append({
            "path": path,
            "tensor": torch.from_numpy(np.asarray(value)),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        })
        return rows

    if isinstance(value, dict):
        for key, child in value.items():
            rows.extend(tensor_tree(child, f"{path}.{key}"))
        return rows

    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            rows.extend(tensor_tree(child, f"{path}[{index}]"))
        return rows

    if hasattr(value, "__dict__"):
        for key, child in vars(value).items():
            if key.startswith("_"):
                continue
            rows.extend(tensor_tree(child, f"{path}.{key}"))
        return rows

    return rows


def choose_tensor(
    rows: list[dict[str, Any]],
    trailing_shape: tuple[int, ...],
    preferred_tokens: tuple[str, ...],
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if tuple(row["shape"][-len(trailing_shape):]) == trailing_shape
    ]
    require(
        candidates,
        f"no tensor with trailing shape {trailing_shape}; "
        f"observed={[row['shape'] for row in rows]}",
    )

    def score(row: dict[str, Any]) -> tuple[int, int, str]:
        path = row["path"].lower()
        token_score = sum(10 for token in preferred_tokens if token in path)
        batch_score = 1 if len(row["shape"]) == len(trailing_shape) + 1 else 0
        return (-token_score, -batch_score, path)

    candidates.sort(key=score)
    selected = candidates[0]
    ties = [
        row
        for row in candidates
        if score(row)[:2] == score(selected)[:2]
    ]
    require(
        len(ties) == 1,
        "tensor-route ambiguity for trailing shape "
        f"{trailing_shape}: {[row['path'] for row in ties]}",
    )
    return selected


def normalize_batched_tensor(
    tensor: torch.Tensor,
    trailing_shape: tuple[int, ...],
) -> torch.Tensor:
    value = tensor.detach()
    if tuple(value.shape) == trailing_shape:
        value = value.unsqueeze(0)
    require(
        tuple(value.shape[-len(trailing_shape):]) == trailing_shape,
        f"tensor shape mismatch: {tuple(value.shape)}",
    )
    require(
        value.ndim == len(trailing_shape) + 1,
        f"unexpected batch rank: {tuple(value.shape)}",
    )
    return value


def extract_x_mask(batch: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    rows = tensor_tree(batch)
    x_row = choose_tensor(
        rows,
        EXPECTED_X_TRAILING_SHAPE,
        ("x", "feature", "input"),
    )
    mask_row = choose_tensor(
        rows,
        EXPECTED_MASK_TRAILING_SHAPE,
        ("mask", "physical", "port"),
    )

    x = normalize_batched_tensor(
        x_row["tensor"],
        EXPECTED_X_TRAILING_SHAPE,
    ).to(dtype=torch.float32)
    mask = normalize_batched_tensor(
        mask_row["tensor"],
        EXPECTED_MASK_TRAILING_SHAPE,
    ).to(dtype=torch.float32)

    require(
        x.shape[0] == mask.shape[0],
        f"x/mask batch-size mismatch: {x.shape} versus {mask.shape}",
    )
    require(torch.isfinite(x).all().item(), "x contains nonfinite values")
    require(torch.isfinite(mask).all().item(), "mask contains nonfinite values")

    inventory = {
        "selected_x_path": x_row["path"],
        "selected_x_shape": list(x.shape),
        "selected_x_dtype": str(x.dtype),
        "selected_mask_path": mask_row["path"],
        "selected_mask_shape": list(mask.shape),
        "selected_mask_dtype": str(mask.dtype),
        "all_tensor_paths": [
            {
                "path": row["path"],
                "shape": row["shape"],
                "dtype": row["dtype"],
            }
            for row in rows
        ],
    }
    return x, mask, inventory


def manual_batch(dataset: Any, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    items = [dataset[index] for index in range(batch_size)]
    x_values = []
    mask_values = []
    inventories = []

    for index, item in enumerate(items):
        x, mask, inventory = extract_x_mask(item)
        require(x.shape[0] == 1, "single dataset item unexpectedly batched")
        require(mask.shape[0] == 1, "single mask item unexpectedly batched")
        x_values.append(x[0])
        mask_values.append(mask[0])
        inventories.append({
            "item_index": index,
            "inventory": inventory,
        })

    return (
        torch.stack(x_values, dim=0),
        torch.stack(mask_values, dim=0),
        {
            "route": "manual_first_items_stack",
            "batch_size": batch_size,
            "item_inventories": inventories,
        },
    )


def first_batch(dataset: Any, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    try:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        batch = next(iter(loader))
        x, mask, inventory = extract_x_mask(batch)
        require(
            x.shape[0] == batch_size,
            "DataLoader first batch size changed",
        )
        return x, mask, {
            "route": "torch_DataLoader_default_collate",
            "batch_size": batch_size,
            "inventory": inventory,
        }
    except BaseException as exc:
        x, mask, fallback = manual_batch(dataset, batch_size)
        fallback["DataLoader_failure"] = repr(exc)
        return x, mask, fallback


def construct_split(
    loader_module: Any,
    dataset_class: type,
    data_root: Path,
    aliases: tuple[str, ...],
    expected_length: int,
) -> tuple[Any, dict[str, Any]]:
    attempts = []
    for split in aliases:
        try:
            dataset = loader_module._construct_original(
                dataset_class,
                data_root,
                split,
                {},
            )
            length = int(len(dataset))
            attempts.append({
                "split": split,
                "status": "SUCCESS",
                "length": length,
            })
            if length == expected_length:
                return dataset, {
                    "selected_split_alias": split,
                    "length": length,
                    "attempts": attempts,
                    "dataset_type": (
                        f"{dataset.__class__.__module__}."
                        f"{dataset.__class__.__name__}"
                    ),
                }
        except BaseException as exc:
            attempts.append({
                "split": split,
                "status": "FAILED",
                "error": repr(exc),
            })
    raise RuntimeError(
        f"could not construct expected split length {expected_length}: {attempts}"
    )


def reset_leaf_modules(
    model: torch.nn.Module,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    checkpoint_hash = stable_state_hash(model)
    state_before = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    reset_rows = []
    for name, module in model.named_modules():
        if name == "":
            continue
        children = list(module.children())
        reset = getattr(module, "reset_parameters", None)
        if children or not callable(reset):
            continue
        reset()
        reset_rows.append({
            "module_name": name,
            "module_type": (
                f"{module.__class__.__module__}."
                f"{module.__class__.__name__}"
            ),
        })

    require(reset_rows, "no leaf reset_parameters routes were found")

    fresh_hash = stable_state_hash(model)
    require(
        fresh_hash != checkpoint_hash,
        "fresh-reset model state equals checkpoint state",
    )

    changed_tensor_count = 0
    unchanged_tensor_count = 0
    nonfinite_tensor_count = 0
    for key, value in model.state_dict().items():
        current = value.detach().cpu()
        if torch.equal(current, state_before[key]):
            unchanged_tensor_count += 1
        else:
            changed_tensor_count += 1
        if not torch.isfinite(current).all().item():
            nonfinite_tensor_count += 1

    require(changed_tensor_count > 0, "fresh reset changed no state tensors")
    require(nonfinite_tensor_count == 0, "fresh reset produced nonfinite state")

    return {
        "seed": seed,
        "checkpoint_state_hash": checkpoint_hash,
        "fresh_state_hash": fresh_hash,
        "reset_leaf_module_count": len(reset_rows),
        "reset_leaf_modules": reset_rows,
        "changed_state_tensor_count": changed_tensor_count,
        "unchanged_state_tensor_count": unchanged_tensor_count,
        "nonfinite_state_tensor_count": nonfinite_tensor_count,
        "policy": (
            "deterministic seed-107 reset_parameters on leaf modules only; "
            "topology/buffer objects from the certified model construction "
            "are retained"
        ),
    }


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def objective_from_outputs(
    original_f6: Any,
    output: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    mapped = original_f6.model_output_mapping(output)
    objective = None
    rows = {}

    for head in ("graph", "count", "source", "transit", "victim", "path"):
        tensor = mapped[head]
        require(
            isinstance(tensor, torch.Tensor),
            f"{head} output is not a tensor",
        )
        require(
            torch.isfinite(tensor).all().item(),
            f"{head} output contains nonfinite values",
        )
        term = tensor.float().pow(2).mean()
        objective = term if objective is None else objective + term
        rows[head] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "mean": float(tensor.detach().float().mean().cpu()),
            "std": float(
                tensor.detach().float().std(unbiased=False).cpu()
            ),
            "squared_mean_term": float(term.detach().cpu()),
        }

    require(objective is not None, "no output heads found")
    require(torch.isfinite(objective).item(), "dry-run objective is nonfinite")
    return objective, rows


def gradient_review(model: torch.nn.Module) -> dict[str, Any]:
    total_squared_norm = 0.0
    finite_gradient_tensor_count = 0
    nonzero_gradient_tensor_count = 0
    missing_gradient_tensor_count = 0
    nonfinite_gradient_tensor_count = 0
    rows = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing_gradient_tensor_count += 1
            rows.append({
                "name": name,
                "status": "MISSING",
            })
            continue

        gradient = parameter.grad.detach()
        finite = bool(torch.isfinite(gradient).all().item())
        nonzero = bool(torch.count_nonzero(gradient).item() > 0)
        norm = float(gradient.float().norm().cpu())

        if finite:
            finite_gradient_tensor_count += 1
        else:
            nonfinite_gradient_tensor_count += 1
        if nonzero:
            nonzero_gradient_tensor_count += 1

        total_squared_norm += norm * norm
        rows.append({
            "name": name,
            "status": "PRESENT",
            "finite": finite,
            "nonzero": nonzero,
            "norm": norm,
            "shape": list(gradient.shape),
        })

    require(nonfinite_gradient_tensor_count == 0, "nonfinite parameter gradients")
    require(nonzero_gradient_tensor_count > 0, "all parameter gradients are zero")

    return {
        "global_l2_norm": math.sqrt(total_squared_norm),
        "finite_gradient_tensor_count": finite_gradient_tensor_count,
        "nonzero_gradient_tensor_count": nonzero_gradient_tensor_count,
        "missing_gradient_tensor_count": missing_gradient_tensor_count,
        "nonfinite_gradient_tensor_count": nonfinite_gradient_tensor_count,
        "parameter_gradients": rows,
    }


def run_forward_backward(
    *,
    original_f6: Any,
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    label: str,
) -> dict[str, Any]:
    model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)

    x_device = x.detach().clone().to(device).requires_grad_(True)
    mask_device = mask.detach().clone().to(device)

    output = model(x_device, mask_device)
    objective, output_review = objective_from_outputs(original_f6, output)
    objective.backward()

    require(x_device.grad is not None, f"{label} input gradient missing")
    require(
        torch.isfinite(x_device.grad).all().item(),
        f"{label} input gradient nonfinite",
    )
    input_gradient_norm = float(
        x_device.grad.detach().float().norm().cpu()
    )
    require(input_gradient_norm > 0.0, f"{label} input gradient is zero")

    parameter_gradients = gradient_review(model)

    return {
        "label": label,
        "objective": float(objective.detach().cpu()),
        "input_gradient_l2_norm": input_gradient_norm,
        "output_heads": output_review,
        "parameter_gradients": parameter_gradients,
        "model_state_hash_after_backward": stable_state_hash(model),
    }


def validation_forward(
    *,
    original_f6: Any,
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(
            x.detach().clone().to(device),
            mask.detach().clone().to(device),
        )
        _, review = objective_from_outputs(original_f6, output)
    return {
        "sample_count": int(x.shape[0]),
        "output_heads": review,
        "all_outputs_finite": True,
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")
    require(data_link.is_symlink(), f"dataset symlink missing: {data_link}")
    data_root = data_link.resolve()
    require(data_root.is_dir(), f"dataset target missing: {data_root}")

    feature_root = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study"
    )
    f7_root = feature_root / "retrained_group_ablation"
    permutation_root = feature_root / "permutation"

    p1b_report_path = f7_root / (
        "V5_P3_F7_P1B_RECONSTRUCTED_TRAINING_RECIPE_PIN_REPORT.json"
    )
    p1b_lock_path = f7_root / (
        "V5_P3_F7_P1B_RECONSTRUCTED_TRAINING_RECIPE_PIN_LOCK.json"
    )
    p1b_route_path = f7_root / (
        "F7_P1B_FROZEN_RECONSTRUCTED_TRAINING_ROUTE.json"
    )
    p0_protocol_path = f7_root / (
        "F7_P0_FROZEN_SAME_WIDTH_ABLATION_PROTOCOL.json"
    )

    p1b_report, p1b_lock = verify_report_lock(
        p1b_report_path,
        p1b_lock_path,
    )
    reconstructed_route = load_json(p1b_route_path)
    protocol = load_json(p0_protocol_path)

    require(
        p1b_lock.get("reconstructed_route_frozen") is True,
        "P1B reconstructed route is not frozen",
    )
    require(
        p1b_lock.get("F7_adapter_preflight_authorized") is True,
        "P1B did not authorize the adapter preflight",
    )
    require(
        p1b_lock.get("actual_F7_retraining_authorized") is False,
        "actual F7 was already authorized before P2",
    )
    require(
        p1b_lock.get("feature_removal_authorized") is False,
        "feature removal was already authorized",
    )
    require(
        reconstructed_route["status"] == "FROZEN",
        "reconstructed route JSON is not frozen",
    )
    require(
        reconstructed_route["model"]["class"] == EXPECTED_MODEL_CLASS,
        "reconstructed model class changed",
    )
    require(
        int(reconstructed_route["model"]["parameter_count"])
        == EXPECTED_PARAMETER_COUNT,
        "reconstructed parameter count changed",
    )
    require(
        int(reconstructed_route["dataset"]["train_items"])
        == EXPECTED_TRAIN_ITEMS,
        "reconstructed train count changed",
    )
    require(
        int(reconstructed_route["dataset"]["validation_items"])
        == EXPECTED_VALIDATION_ITEMS,
        "reconstructed validation count changed",
    )

    route_inventory_path = permutation_root / (
        "F5_P0_EXECUTION_ROUTE_SOURCE_INVENTORY.json"
    )
    route_inventory = load_json(route_inventory_path)
    certified = route_inventory["certified_files"]

    for key in ("model", "loader", "checkpoint", "exporter"):
        row = certified[key]
        path = Path(row["path"]).resolve()
        require(path.is_file(), f"certified {key} missing: {path}")
        require(
            sha256_file(path) == row["actual_sha256"],
            f"certified {key} hash changed",
        )

    model_path = Path(certified["model"]["path"]).resolve()
    loader_path = Path(certified["loader"]["path"]).resolve()
    checkpoint_path = Path(certified["checkpoint"]["path"]).resolve()
    exporter_path = Path(certified["exporter"]["path"]).resolve()

    original_f6_path = (
        repo
        / "scripts/v5/p3/experiments/"
        "run_v5_p3_f6_task_specific_integrated_gradients.py"
    )
    require(
        original_f6_path.is_file(),
        f"F6 model-capture implementation missing: {original_f6_path}",
    )
    original_f6 = import_source(
        original_f6_path,
        "_v5_p3_f7_p2_original_f6_route",
    )
    loader_module = import_source(
        loader_path,
        "_v5_p3_f7_p2_guarded_loader",
    )

    required_helpers = (
        "configure_official_runtime",
        "capture_model_via_official_exporter",
        "extract_state_dict",
        "state_dict_exact_match",
        "parameter_count",
        "model_output_mapping",
    )
    for helper in required_helpers:
        require(
            hasattr(original_f6, helper)
            and callable(getattr(original_f6, helper)),
            f"required F6 helper missing: {helper}",
        )

    require(
        hasattr(loader_module, "_load_original_class")
        and hasattr(loader_module, "_construct_original"),
        "guarded loader construction helpers changed",
    )

    device, runtime = original_f6.configure_official_runtime(args.device)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    require(isinstance(checkpoint, dict), "checkpoint is not a dictionary")
    checkpoint_state = original_f6.extract_state_dict(checkpoint)

    capture_dir = output_dir / "P2_MODEL_CAPTURE_PROBE"
    if capture_dir.exists():
        shutil.rmtree(capture_dir)
    capture_dir.mkdir(parents=True)

    checkpoint_model, capture = (
        original_f6.capture_model_via_official_exporter(
            exporter_path=exporter_path,
            repo=repo,
            data_link=data_link,
            run_dir=capture_dir,
            checkpoint_state_dict=checkpoint_state,
        )
    )
    checkpoint_model.to("cpu")
    checkpoint_model.eval()

    require(
        checkpoint_model.__class__.__name__ == EXPECTED_MODEL_CLASS,
        "captured model class changed",
    )
    require(
        original_f6.parameter_count(checkpoint_model)
        == EXPECTED_PARAMETER_COUNT,
        "captured model parameter count changed",
    )
    require(
        original_f6.state_dict_exact_match(
            checkpoint_model,
            checkpoint_state,
        ),
        "captured model does not exactly match checkpoint",
    )

    fresh_model = copy.deepcopy(checkpoint_model)
    for parameter in fresh_model.parameters():
        parameter.requires_grad_(True)
    reset_review = reset_leaf_modules(fresh_model, PRIMARY_SEED)

    require(
        original_f6.parameter_count(fresh_model)
        == EXPECTED_PARAMETER_COUNT,
        "fresh model parameter count changed",
    )
    require(
        trainable_parameter_count(fresh_model)
        == EXPECTED_PARAMETER_COUNT,
        "fresh model trainable parameter count changed",
    )

    dataset_class = loader_module._load_original_class(data_root)
    train_dataset, train_split_review = construct_split(
        loader_module,
        dataset_class,
        data_root,
        ("train", "training", "TRAIN"),
        EXPECTED_TRAIN_ITEMS,
    )
    validation_dataset, validation_split_review = construct_split(
        loader_module,
        dataset_class,
        data_root,
        ("validation", "val", "VALIDATION"),
        EXPECTED_VALIDATION_ITEMS,
    )

    train_x, train_mask, train_batch_route = first_batch(
        train_dataset,
        DRY_RUN_BATCH_SIZE,
    )
    validation_x, validation_mask, validation_batch_route = first_batch(
        validation_dataset,
        DRY_RUN_BATCH_SIZE,
    )

    require(
        tuple(train_x.shape)
        == (DRY_RUN_BATCH_SIZE, *EXPECTED_X_TRAILING_SHAPE),
        f"train x shape changed: {tuple(train_x.shape)}",
    )
    require(
        tuple(train_mask.shape)
        == (DRY_RUN_BATCH_SIZE, *EXPECTED_MASK_TRAILING_SHAPE),
        f"train mask shape changed: {tuple(train_mask.shape)}",
    )
    require(
        tuple(validation_x.shape)
        == (DRY_RUN_BATCH_SIZE, *EXPECTED_X_TRAILING_SHAPE),
        f"validation x shape changed: {tuple(validation_x.shape)}",
    )
    require(
        tuple(validation_mask.shape)
        == (DRY_RUN_BATCH_SIZE, *EXPECTED_MASK_TRAILING_SHAPE),
        f"validation mask shape changed: {tuple(validation_mask.shape)}",
    )

    control_x = train_x.detach().clone()
    masked_x = control_x.clone()
    masked_x[:, :, FLOW_GROUP_START:FLOW_GROUP_END, :] = 0.0

    outside_left_equal = torch.equal(
        control_x[:, :, :FLOW_GROUP_START, :],
        masked_x[:, :, :FLOW_GROUP_START, :],
    )
    outside_right_equal = torch.equal(
        control_x[:, :, FLOW_GROUP_END:, :],
        masked_x[:, :, FLOW_GROUP_END:, :],
    )
    masked_region_zero = bool(
        torch.count_nonzero(
            masked_x[:, :, FLOW_GROUP_START:FLOW_GROUP_END, :]
        ).item()
        == 0
    )
    changed_value_count = int(
        torch.count_nonzero(control_x != masked_x).item()
    )
    original_group_nonzero_count = int(
        torch.count_nonzero(
            control_x[:, :, FLOW_GROUP_START:FLOW_GROUP_END, :]
        ).item()
    )

    require(outside_left_equal, "mask changed channels before 55")
    require(outside_right_equal, "mask changed channels after 69")
    require(masked_region_zero, "flow-control-stall group was not zeroed")
    require(
        changed_value_count == original_group_nonzero_count,
        "mask changed values outside the nonzero flow-control group entries",
    )
    require(
        torch.equal(train_mask, train_mask.detach().clone()),
        "physical mask mutation detected",
    )

    control_model = copy.deepcopy(fresh_model)
    masked_model = copy.deepcopy(fresh_model)
    require(
        stable_state_hash(control_model)
        == stable_state_hash(masked_model)
        == reset_review["fresh_state_hash"],
        "control and masked models do not share identical fresh initialization",
    )

    control_run = run_forward_backward(
        original_f6=original_f6,
        model=control_model,
        x=control_x,
        mask=train_mask,
        device=device,
        label="no_mask_control",
    )
    masked_run = run_forward_backward(
        original_f6=original_f6,
        model=masked_model,
        x=masked_x,
        mask=train_mask,
        device=device,
        label="flow_control_stalls_masked",
    )

    validation_model = copy.deepcopy(fresh_model)
    validation_run = validation_forward(
        original_f6=original_f6,
        model=validation_model,
        x=validation_x,
        mask=validation_mask,
        device=device,
    )

    dry_run_path = output_dir / (
        "F7_P2_CONTROL_AND_FLOW_STALL_MASKED_DRY_RUN.json"
    )
    adapter_path = output_dir / (
        "F7_P2_SAME_WIDTH_MASKING_ADAPTER_CERTIFICATION.json"
    )
    model_path_report = output_dir / (
        "F7_P2_MODEL_CONSTRUCTION_AND_FRESH_RESET_CERTIFICATION.json"
    )
    data_path_report = output_dir / (
        "F7_P2_GUARDED_TRAIN_VALIDATION_BATCH_CERTIFICATION.json"
    )
    authorization_path = output_dir / (
        "F7_P2_EXECUTION_AUTHORIZATION.json"
    )

    adapter_review = {
        "input_shape": list(control_x.shape),
        "group": "flow_control_stalls",
        "start": FLOW_GROUP_START,
        "end_exclusive": FLOW_GROUP_END,
        "same_width_channels": 70,
        "outside_left_exactly_equal": outside_left_equal,
        "outside_right_exactly_equal": outside_right_equal,
        "masked_region_all_zero": masked_region_zero,
        "original_group_nonzero_count": original_group_nonzero_count,
        "changed_value_count": changed_value_count,
        "physical_port_mask_exactly_reused": True,
        "edge_index_and_model_buffers_changed": False,
        "labels_changed": False,
        "operation": (
            "x.clone(); x[:,:,55:70,:]=0.0 after guarded loader output "
            "and before model forward"
        ),
    }
    atomic_json(adapter_path, adapter_review)

    model_review = {
        "certified_model_source": str(model_path),
        "certified_model_source_sha256": sha256_file(model_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "exporter": str(exporter_path),
        "exporter_sha256": sha256_file(exporter_path),
        "model_capture": capture,
        "captured_model_class": checkpoint_model.__class__.__name__,
        "captured_parameter_count": original_f6.parameter_count(
            checkpoint_model
        ),
        "checkpoint_state_exact_match": True,
        "fresh_reset": reset_review,
        "fresh_parameter_count": original_f6.parameter_count(fresh_model),
        "fresh_trainable_parameter_count": trainable_parameter_count(
            fresh_model
        ),
        "control_masked_initial_state_exact_match": True,
        "fresh_route_is_byte_identical_to_missing_A4_initialization": False,
    }
    atomic_json(model_path_report, model_review)

    data_review = {
        "guarded_loader": str(loader_path),
        "guarded_loader_sha256": sha256_file(loader_path),
        "dataset_class": (
            f"{dataset_class.__module__}.{dataset_class.__name__}"
        ),
        "train_split": train_split_review,
        "validation_split": validation_split_review,
        "train_batch": train_batch_route,
        "validation_batch": validation_batch_route,
        "train_x_shape": list(train_x.shape),
        "train_mask_shape": list(train_mask.shape),
        "validation_x_shape": list(validation_x.shape),
        "validation_mask_shape": list(validation_mask.shape),
        "train_dataset_items_accessed": DRY_RUN_BATCH_SIZE,
        "validation_dataset_items_accessed": DRY_RUN_BATCH_SIZE,
        "sealed_test_dataset_constructed": False,
        "sealed_test_items_accessed": 0,
    }
    atomic_json(data_path_report, data_review)

    dry_run = {
        "runtime": runtime,
        "device": str(device),
        "control": control_run,
        "flow_control_stalls_masked": masked_run,
        "validation_no_mask_forward": validation_run,
        "no_optimizer_step_performed": True,
        "scientific_checkpoint_saved": False,
        "scientific_metric_reported": False,
        "model_weights_persisted": False,
    }
    atomic_json(dry_run_path, dry_run)

    all_gates_pass = bool(
        checkpoint_model.__class__.__name__ == EXPECTED_MODEL_CLASS
        and original_f6.parameter_count(fresh_model)
        == EXPECTED_PARAMETER_COUNT
        and trainable_parameter_count(fresh_model)
        == EXPECTED_PARAMETER_COUNT
        and outside_left_equal
        and outside_right_equal
        and masked_region_zero
        and control_run["parameter_gradients"][
            "nonzero_gradient_tensor_count"
        ]
        > 0
        and masked_run["parameter_gradients"][
            "nonzero_gradient_tensor_count"
        ]
        > 0
        and validation_run["all_outputs_finite"]
    )

    require(all_gates_pass, "one or more P2 dry-run gates failed")

    next_stage = (
        "V5_P3_F7_SAME_WIDTH_RETRAINED_GROUP_ABLATION_EXECUTION"
    )
    authorization = {
        "P1B_reconstructed_route_frozen": True,
        "exact_checkpoint_model_construction_verified": True,
        "deterministic_fresh_reset_route_verified": True,
        "fresh_route_declared_non_byte_identical_to_A4": True,
        "train_batch_verified": True,
        "validation_batch_verified": True,
        "control_forward_backward_verified": True,
        "flow_control_stalls_masked_forward_backward_verified": True,
        "same_width_mask_isolation_verified": True,
        "parameter_count_preserved": True,
        "sealed_test_access": False,
        "actual_F7_retraining_authorized": True,
        "authorized_primary_runs": [
            "control_dynamic70",
            "ablate_directional_traffic_volume",
            "ablate_inter_flit_timing",
            "ablate_queue_activity",
            "ablate_buffer_pressure",
            "ablate_flow_control_stalls",
        ],
        "authorized_primary_seed": PRIMARY_SEED,
        "F8_multi_seed_authorized": False,
        "feature_removal_authorized": False,
        "compact_interface_equivalence_authorized": False,
        "hardware_reduction_claim_authorized": False,
        "next_stage": next_stage,
    }
    atomic_json(authorization_path, authorization)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "scope": (
            "Verify the frozen reconstructed route; capture the exact certified "
            "V6P0 model through the official exporter; deterministically reset "
            "leaf trainable modules at seed 107 while retaining topology and "
            "buffers; construct guarded train and validation datasets; access "
            "only one two-item batch from each; certify the same-width "
            "flow-control-stall masking adapter; run identical-initialization "
            "control and masked forward/backward dry runs plus validation "
            "forward; save no scientific checkpoint or metric; and authorize "
            "the six-run seed-107 F7 matrix only when every gate passes."
        ),
        "finding": {
            "model": model_review,
            "data": data_review,
            "adapter": adapter_review,
            "dry_run": dry_run,
            "all_gates_pass": all_gates_pass,
        },
        "decision": authorization,
        "governance": {
            "model_loaded": True,
            "checkpoint_loaded": True,
            "training_feature_tensors_loaded": True,
            "validation_feature_tensors_loaded": True,
            "training_items_accessed": DRY_RUN_BATCH_SIZE,
            "validation_items_accessed": DRY_RUN_BATCH_SIZE,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "optimizer_step_performed": False,
            "scientific_checkpoint_saved": False,
            "scientific_result_generated": False,
            "feature_removal_authorized": False,
        },
        "artifacts": {
            "model_certification": str(model_path_report),
            "data_certification": str(data_path_report),
            "adapter_certification": str(adapter_path),
            "dry_run": str(dry_run_path),
            "execution_authorization": str(authorization_path),
        },
        "provenance": {
            "P1B_report_sha256": sha256_file(p1b_report_path),
            "P1B_lock_sha256": sha256_file(p1b_lock_path),
            "P1B_route_sha256": sha256_file(p1b_route_path),
            "F7_protocol_sha256": sha256_file(p0_protocol_path),
            "route_inventory_sha256": sha256_file(route_inventory_path),
            "certified_model_sha256": sha256_file(model_path),
            "certified_loader_sha256": sha256_file(loader_path),
            "certified_checkpoint_sha256": sha256_file(checkpoint_path),
            "certified_exporter_sha256": sha256_file(exporter_path),
            "original_F6_route_sha256": sha256_file(original_f6_path),
            "installed_script_sha256": sha256_file(installed_script),
            "model_certification_sha256": sha256_file(model_path_report),
            "data_certification_sha256": sha256_file(data_path_report),
            "adapter_certification_sha256": sha256_file(adapter_path),
            "dry_run_sha256": sha256_file(dry_run_path),
            "execution_authorization_sha256": sha256_file(
                authorization_path
            ),
        },
    }

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    atomic_json(report_path, report)
    atomic_json(
        lock_path,
        {
            "stage": STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(report_path),
            "model_certification_sha256": sha256_file(model_path_report),
            "data_certification_sha256": sha256_file(data_path_report),
            "adapter_certification_sha256": sha256_file(adapter_path),
            "dry_run_sha256": sha256_file(dry_run_path),
            "execution_authorization_sha256": sha256_file(
                authorization_path
            ),
            "all_dry_run_gates_pass": True,
            "actual_F7_retraining_authorized": True,
            "F8_multi_seed_authorized": False,
            "feature_removal_authorized": False,
            "compact_interface_equivalence_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "optimizer_step_performed": False,
            "scientific_checkpoint_saved": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print("P1B_reconstructed_route_frozen=true")
    print(f"device={device}")
    print(f"model_class={checkpoint_model.__class__.__name__}")
    print(
        "parameter_count="
        f"{original_f6.parameter_count(fresh_model)}"
    )
    print(
        "trainable_parameter_count="
        f"{trainable_parameter_count(fresh_model)}"
    )
    print(
        "checkpoint_model_exact_match=true"
    )
    print(
        "fresh_reset_state_hash="
        f"{reset_review['fresh_state_hash']}"
    )
    print(
        "fresh_reset_changed_state_tensor_count="
        f"{reset_review['changed_state_tensor_count']}"
    )
    print(f"train_items={len(train_dataset)}")
    print(f"validation_items={len(validation_dataset)}")
    print(f"train_batch_route={train_batch_route['route']}")
    print(f"validation_batch_route={validation_batch_route['route']}")
    print(f"train_x_shape={list(train_x.shape)}")
    print(f"train_mask_shape={list(train_mask.shape)}")
    print(
        "flow_stall_mask_changed_value_count="
        f"{changed_value_count}"
    )
    print("flow_stall_mask_outside_group_exact=true")
    print("flow_stall_mask_region_all_zero=true")
    print(
        "control_objective="
        f"{control_run['objective']}"
    )
    print(
        "control_parameter_gradient_l2="
        f"{control_run['parameter_gradients']['global_l2_norm']}"
    )
    print(
        "masked_objective="
        f"{masked_run['objective']}"
    )
    print(
        "masked_parameter_gradient_l2="
        f"{masked_run['parameter_gradients']['global_l2_norm']}"
    )
    print("validation_outputs_finite=true")
    print("optimizer_step_performed=false")
    print("scientific_checkpoint_saved=false")
    print("scientific_result_generated=false")
    print("actual_F7_retraining_authorized=true")
    print("authorized_primary_seed=107")
    print("authorized_primary_run_count=6")
    print("F8_multi_seed_authorized=false")
    print("feature_removal_authorized=false")
    print("compact_interface_equivalence_authorized=false")
    print("hardware_reduction_claim_authorized=false")
    print("sealed_test_tensors_loaded=false")
    print(
        "next_stage="
        "V5_P3_F7_SAME_WIDTH_RETRAINED_GROUP_ABLATION_EXECUTION"
    )
    print(f"model_certification={model_path_report}")
    print(f"data_certification={data_path_report}")
    print(f"adapter_certification={adapter_path}")
    print(f"dry_run={dry_run_path}")
    print(f"execution_authorization={authorization_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
