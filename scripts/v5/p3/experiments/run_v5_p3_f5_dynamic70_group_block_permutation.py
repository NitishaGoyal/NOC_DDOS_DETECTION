from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


STAGE = "V5_P3_F5_DYNAMIC70_GROUP_BLOCK_PERMUTATION"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "VALIDATION-EXPLORATORY"

EXPECTED_ITEMS = 13863
EXPECTED_X_SHAPE = (16, 70, 32)
EXPECTED_MASK_SHAPE = (16, 10)
EXPECTED_PARAMETER_COUNT = 60553
EXPECTED_EDGE_BUFFER_ELEMENTS = 96

GROUPS = {
    "directional_traffic_volume": (0, 10),
    "inter_flit_timing": (10, 30),
    "queue_activity": (30, 40),
    "buffer_pressure": (40, 55),
    "flow_control_stalls": (55, 70),
}

SEEDS = (
    5101,
    5102,
    5103,
    5104,
    5105,
    5106,
    5107,
    5108,
    5109,
    5110,
)

HEAD_WIDTHS = {
    "graph": 1,
    "count": 4,
    "source": 16,
    "transit": 16,
    "victim": 16,
    "path": 16,
}

LOGIT_FILE_NAMES = {
    "graph": "attack_logits.npy",
    "count": "count_logits.npy",
    "source": "source_logits.npy",
    "transit": "transit_logits.npy",
    "victim": "victim_logits.npy",
    "path": "path_logits.npy",
}

LABEL_KEYS = {
    "y_attack": ("y_attack", "attack_label", "y_graph", "graph_label"),
    "y_attacker_count": (
        "y_attacker_count",
        "attacker_count",
        "y_count",
        "count_label",
    ),
    "y_source": ("y_source", "source_label", "source_target"),
    "y_transit": ("y_transit", "transit_label", "transit_target"),
    "y_victim": ("y_victim", "victim_label", "victim_target"),
    "y_attack_path": (
        "y_attack_path",
        "y_path",
        "path_label",
        "path_target",
    ),
}

X_ALIASES = (
    "x",
    "features",
    "feature",
    "temporal_x",
    "temporal_features",
    "node_features",
)

MASK_ALIASES = (
    "physical_port_mask",
    "port_mask",
    "physical_mask",
)

METRIC_NAMES = (
    "selection_score",
    "graph_auroc",
    "graph_ap",
    "graph_f1_at_0_5",
    "graph_fpr_at_0_5",
    "count_active_macro_f1",
    "source_ap",
    "source_exact_active",
    "transit_ap",
    "transit_exact_active",
    "victim_ap",
    "victim_exact_active",
    "path_ap",
    "path_exact_active",
)

HIGHER_IS_BETTER = {
    metric for metric in METRIC_NAMES
    if metric != "graph_fpr_at_0_5"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--installed-script", required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--batch-size", type=int, default=0)
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
    if isinstance(value, set):
        return sorted(value)
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


def import_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_report_lock(
    report_path: Path,
    lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(report_path.is_file(), f"report missing: {report_path}")
    require(lock_path.is_file(), f"lock missing: {lock_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require(report.get("status") == "PASS", f"report not PASS: {report_path}")
    require(
        lock.get("report_sha256") == sha256_file(report_path),
        f"report/lock mismatch: {report_path}",
    )
    return report, lock


def choose_run_dir(output_root: Path, explicit: str) -> Path:
    runs_root = output_root / "f5_runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    if explicit:
        run_dir = Path(explicit).expanduser().resolve()
        require(run_dir.is_dir(), f"explicit run directory missing: {run_dir}")
        return run_dir

    active_pointer = output_root / "F5_ACTIVE_RUN_PATH.txt"
    if active_pointer.is_file():
        candidate = Path(
            active_pointer.read_text(encoding="utf-8").strip()
        ).expanduser().resolve()
        if candidate.is_dir() and not (
            candidate / f"{STAGE}_COMPLETE"
        ).is_file():
            return candidate

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_root / f"run_{timestamp}"
    require(not run_dir.exists(), f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    atomic_text(active_pointer, str(run_dir) + "\n")
    return run_dir


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def recursive_field_search(
    item: Any,
    aliases: tuple[str, ...],
    *,
    depth: int = 0,
) -> Any:
    if depth > 4:
        raise KeyError(aliases)

    normalized_aliases = {
        normalize_name(alias)
        for alias in aliases
    }

    if isinstance(item, Mapping):
        for key, value in item.items():
            if normalize_name(str(key)) in normalized_aliases:
                return value
        for key, value in item.items():
            if isinstance(value, (Mapping, list, tuple)) or hasattr(
                value, "__dict__"
            ):
                try:
                    return recursive_field_search(
                        value,
                        aliases,
                        depth=depth + 1,
                    )
                except KeyError:
                    pass

    if hasattr(item, "_fields"):
        for field in item._fields:
            if normalize_name(str(field)) in normalized_aliases:
                return getattr(item, field)

    for alias in aliases:
        if hasattr(item, alias):
            return getattr(item, alias)

    if hasattr(item, "__dict__"):
        dictionary = vars(item)
        for key, value in dictionary.items():
            if normalize_name(str(key)) in normalized_aliases:
                return value
        for value in dictionary.values():
            if isinstance(value, Mapping) or hasattr(value, "__dict__"):
                try:
                    return recursive_field_search(
                        value,
                        aliases,
                        depth=depth + 1,
                    )
                except KeyError:
                    pass

    raise KeyError(aliases)


def item_structure(item: Any, depth: int = 0) -> Any:
    if depth > 3:
        return type(item).__name__
    if isinstance(item, Mapping):
        return {
            str(key): item_structure(value, depth + 1)
            for key, value in item.items()
        }
    if isinstance(item, (list, tuple)):
        return [
            item_structure(value, depth + 1)
            for value in item
        ]
    if isinstance(item, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(item.shape),
            "dtype": str(item.dtype),
        }
    if isinstance(item, np.ndarray):
        return {
            "type": "numpy.ndarray",
            "shape": list(item.shape),
            "dtype": str(item.dtype),
        }
    if hasattr(item, "shape") and hasattr(item, "dtype"):
        return {
            "type": type(item).__name__,
            "shape": list(item.shape),
            "dtype": str(item.dtype),
        }
    if hasattr(item, "__dict__"):
        return {
            "type": type(item).__name__,
            "attributes": {
                key: item_structure(value, depth + 1)
                for key, value in vars(item).items()
                if not key.startswith("_")
            },
        }
    return {
        "type": type(item).__name__,
        "repr": repr(item)[:500],
    }


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def canonicalize_x(value: Any) -> np.ndarray:
    array = to_numpy(value)
    array = np.squeeze(array)
    require(array.ndim == 3, f"x must be rank 3, got {array.shape}")

    target = EXPECTED_X_SHAPE
    if tuple(array.shape) == target:
        result = array
    else:
        matches = []
        for axes in (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ):
            if tuple(array.shape[index] for index in axes) == target:
                matches.append(axes)
        require(
            len(matches) == 1,
            f"cannot uniquely canonicalize x shape {array.shape}",
        )
        result = np.transpose(array, matches[0])

    result = np.asarray(result, dtype=np.float32)
    require(
        result.shape == EXPECTED_X_SHAPE,
        f"canonical x shape mismatch: {result.shape}",
    )
    require(np.all(np.isfinite(result)), "x contains nonfinite values")
    return result


def canonicalize_mask(value: Any) -> np.ndarray:
    array = np.squeeze(to_numpy(value))
    require(array.ndim == 2, f"mask must be rank 2, got {array.shape}")

    if tuple(array.shape) == EXPECTED_MASK_SHAPE:
        result = array
    elif tuple(array.T.shape) == EXPECTED_MASK_SHAPE:
        result = array.T
    else:
        raise RuntimeError(f"unexpected physical mask shape: {array.shape}")

    result = np.asarray(result, dtype=np.float32)
    require(np.all(np.isfinite(result)), "physical mask contains nonfinite values")
    return result


def canonicalize_scalar(value: Any, name: str) -> int:
    array = np.squeeze(to_numpy(value))
    require(array.size == 1, f"{name} must be scalar, got {array.shape}")
    numeric = int(array.reshape(-1)[0])
    return numeric


def canonicalize_role(value: Any, name: str) -> np.ndarray:
    array = np.squeeze(to_numpy(value))
    require(array.size == 16, f"{name} must contain 16 values, got {array.shape}")
    result = np.asarray(array, dtype=np.uint8).reshape(16)
    require(
        np.all((result == 0) | (result == 1)),
        f"{name} is not binary",
    )
    return result


def extract_item(item: Any) -> dict[str, np.ndarray | int]:
    x = canonicalize_x(recursive_field_search(item, X_ALIASES))
    mask = canonicalize_mask(
        recursive_field_search(item, MASK_ALIASES)
    )

    result: dict[str, np.ndarray | int] = {
        "x": x,
        "physical_port_mask": mask,
    }
    for output_key, aliases in LABEL_KEYS.items():
        value = recursive_field_search(item, aliases)
        if output_key in ("y_attack", "y_attacker_count"):
            result[output_key] = canonicalize_scalar(value, output_key)
        else:
            result[output_key] = canonicalize_role(value, output_key)
    return result


def construct_validation_dataset(
    loader_module: Any,
    data_root: Path,
) -> tuple[Any, dict[str, Any]]:
    attempts = []

    if hasattr(loader_module, "_load_original_class") and hasattr(
        loader_module, "_construct_original"
    ):
        try:
            dataset_class = loader_module._load_original_class(data_root)
            for split in ("validation", "val", "VALIDATION"):
                try:
                    dataset = loader_module._construct_original(
                        dataset_class,
                        data_root,
                        split,
                        {},
                    )
                    length = len(dataset)
                    attempts.append({
                        "route": "_load_original_class/_construct_original",
                        "split": split,
                        "length": int(length),
                        "status": "SUCCESS",
                    })
                    if length == EXPECTED_ITEMS:
                        return dataset, {
                            "route": "_load_original_class/_construct_original",
                            "split": split,
                            "dataset_class": (
                                f"{dataset.__class__.__module__}."
                                f"{dataset.__class__.__name__}"
                            ),
                            "attempts": attempts,
                        }
                except BaseException as exc:
                    attempts.append({
                        "route": "_load_original_class/_construct_original",
                        "split": split,
                        "status": "FAILED",
                        "error": repr(exc),
                    })
        except BaseException as exc:
            attempts.append({
                "route": "_load_original_class",
                "status": "FAILED",
                "error": repr(exc),
            })

    candidate_classes = []
    for name, value in vars(loader_module).items():
        if not inspect.isclass(value):
            continue
        if not hasattr(value, "__getitem__") or not hasattr(value, "__len__"):
            continue
        candidate_classes.append((name, value))

    for name, candidate in candidate_classes:
        try:
            signature = inspect.signature(candidate)
        except BaseException:
            continue

        for split in ("validation", "val", "VALIDATION"):
            kwargs = {}
            unsupported = []
            for parameter in signature.parameters.values():
                if parameter.name == "self":
                    continue
                if parameter.default is not inspect.Parameter.empty:
                    continue

                normalized = normalize_name(parameter.name)
                if normalized in (
                    "dataset_root",
                    "data_root",
                    "root",
                    "path",
                    "data_dir",
                ):
                    kwargs[parameter.name] = data_root
                elif normalized in ("split", "partition", "subset"):
                    kwargs[parameter.name] = split
                else:
                    unsupported.append(parameter.name)

            if unsupported:
                continue

            try:
                dataset = candidate(**kwargs)
                length = len(dataset)
                attempts.append({
                    "route": f"class:{name}",
                    "split": split,
                    "length": int(length),
                    "status": "SUCCESS",
                })
                if length == EXPECTED_ITEMS:
                    return dataset, {
                        "route": f"class:{name}",
                        "split": split,
                        "dataset_class": (
                            f"{dataset.__class__.__module__}."
                            f"{dataset.__class__.__name__}"
                        ),
                        "attempts": attempts,
                    }
            except BaseException as exc:
                attempts.append({
                    "route": f"class:{name}",
                    "split": split,
                    "status": "FAILED",
                    "error": repr(exc),
                })

    raise RuntimeError(
        "could not construct a validation-only dataset of length "
        f"{EXPECTED_ITEMS}; attempts={attempts}"
    )


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "x": cache_dir / "x.npy",
        "physical_port_mask": cache_dir / "physical_port_mask.npy",
        "y_attack": cache_dir / "y_attack.npy",
        "y_attacker_count": cache_dir / "y_attacker_count.npy",
        "y_source": cache_dir / "y_source.npy",
        "y_transit": cache_dir / "y_transit.npy",
        "y_victim": cache_dir / "y_victim.npy",
        "y_attack_path": cache_dir / "y_attack_path.npy",
    }


def open_cache_arrays(
    cache_dir: Path,
    mode: str,
) -> dict[str, np.memmap]:
    paths = cache_paths(cache_dir)

    specifications = {
        "x": (np.float32, (EXPECTED_ITEMS, *EXPECTED_X_SHAPE)),
        "physical_port_mask": (
            np.float32,
            (EXPECTED_ITEMS, *EXPECTED_MASK_SHAPE),
        ),
        "y_attack": (np.int64, (EXPECTED_ITEMS,)),
        "y_attacker_count": (np.int64, (EXPECTED_ITEMS,)),
        "y_source": (np.uint8, (EXPECTED_ITEMS, 16)),
        "y_transit": (np.uint8, (EXPECTED_ITEMS, 16)),
        "y_victim": (np.uint8, (EXPECTED_ITEMS, 16)),
        "y_attack_path": (np.uint8, (EXPECTED_ITEMS, 16)),
    }

    arrays = {}
    for key, (dtype, shape) in specifications.items():
        arrays[key] = np.lib.format.open_memmap(
            paths[key],
            mode=mode,
            dtype=dtype,
            shape=shape,
        )
    return arrays


def build_or_resume_cache(
    dataset: Any,
    cache_dir: Path,
    dataset_provenance: dict[str, Any],
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    completion_path = cache_dir / "CACHE_COMPLETE.json"
    progress_path = cache_dir / "CACHE_PROGRESS.json"
    structure_path = cache_dir / "FIRST_ITEM_STRUCTURE.json"

    if completion_path.is_file():
        completion = json.loads(
            completion_path.read_text(encoding="utf-8")
        )
        require(
            completion.get("item_count") == EXPECTED_ITEMS,
            "completed cache item count mismatch",
        )
        for key, path in cache_paths(cache_dir).items():
            require(path.is_file(), f"completed cache missing {key}: {path}")
        return completion

    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        start = int(progress["written_items"])
        mode = "r+"
    else:
        start = 0
        mode = "w+"

    arrays = open_cache_arrays(cache_dir, mode)

    for index in range(start, EXPECTED_ITEMS):
        item = dataset[index]
        if index == 0 and not structure_path.is_file():
            atomic_json(structure_path, item_structure(item))

        extracted = extract_item(item)
        arrays["x"][index] = extracted["x"]
        arrays["physical_port_mask"][index] = extracted[
            "physical_port_mask"
        ]
        arrays["y_attack"][index] = extracted["y_attack"]
        arrays["y_attacker_count"][index] = extracted[
            "y_attacker_count"
        ]
        arrays["y_source"][index] = extracted["y_source"]
        arrays["y_transit"][index] = extracted["y_transit"]
        arrays["y_victim"][index] = extracted["y_victim"]
        arrays["y_attack_path"][index] = extracted["y_attack_path"]

        if (index + 1) % 100 == 0 or index + 1 == EXPECTED_ITEMS:
            for array in arrays.values():
                array.flush()
            atomic_json(
                progress_path,
                {
                    "written_items": index + 1,
                    "item_count": EXPECTED_ITEMS,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(
                f"cache_progress={index + 1}/{EXPECTED_ITEMS}",
                flush=True,
            )

    validation_summary = {
        "item_count": EXPECTED_ITEMS,
        "x_shape": [EXPECTED_ITEMS, *EXPECTED_X_SHAPE],
        "physical_port_mask_shape": [
            EXPECTED_ITEMS,
            *EXPECTED_MASK_SHAPE,
        ],
        "attack_positive_count": int(
            np.sum(np.asarray(arrays["y_attack"]) == 1)
        ),
        "attack_control_count": int(
            np.sum(np.asarray(arrays["y_attack"]) == 0)
        ),
        "attacker_count_values": sorted(
            int(value)
            for value in np.unique(arrays["y_attacker_count"])
        ),
        "mask_variation_count": int(
            np.sum(
                np.any(
                    np.asarray(arrays["physical_port_mask"])
                    != np.asarray(arrays["physical_port_mask"][0]),
                    axis=(1, 2),
                )
            )
        ),
        "dataset_provenance": dataset_provenance,
        "cache_files": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for key, path in cache_paths(cache_dir).items()
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(completion_path, validation_summary)
    return validation_summary


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            return value

    tensor_mapping = {
        key: value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }
    require(tensor_mapping, "checkpoint state_dict not found")
    return tensor_mapping


class _OfficialModelCaptured(BaseException):
    pass


def state_dict_exact_match(
    model: torch.nn.Module,
    checkpoint_state_dict: dict[str, torch.Tensor],
) -> bool:
    try:
        model_state = model.state_dict()
    except BaseException:
        return False

    if set(model_state) != set(checkpoint_state_dict):
        return False

    for key in model_state:
        left = model_state[key]
        right = checkpoint_state_dict[key]
        if not isinstance(left, torch.Tensor) or not isinstance(
            right, torch.Tensor
        ):
            return False
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        if not torch.equal(
            left.detach().cpu(),
            right.detach().cpu(),
        ):
            return False
    return True


def capture_model_via_official_exporter(
    *,
    exporter_path: Path,
    repo: Path,
    data_link: Path,
    run_dir: Path,
    checkpoint_state_dict: dict[str, torch.Tensor],
) -> tuple[torch.nn.Module, str, dict[str, Any]]:
    """
    Reuse the exact official D1 model-construction path without guessing
    constructor arguments.

    The official exporter is executed only until a local nn.Module exactly
    matches the frozen checkpoint state_dict. A trace hook then captures that
    fully constructed/loaded model and aborts before validation inference.
    """
    exporter_module = import_source(
        exporter_path,
        "_v5_p3_f5_r1_official_d1_exporter",
    )
    require(
        hasattr(exporter_module, "main")
        and callable(exporter_module.main),
        "official D1 exporter does not expose main()",
    )

    probe_dir = run_dir / "official_exporter_model_capture_probe"
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    probe_dir.mkdir(parents=True)

    captured: dict[str, Any] = {}
    exporter_resolved = str(exporter_path.resolve())

    def tracer(frame, event, arg):
        if event != "line":
            return tracer
        if str(Path(frame.f_code.co_filename).resolve()) != exporter_resolved:
            return tracer

        for local_name, value in tuple(frame.f_locals.items()):
            if not isinstance(value, torch.nn.Module):
                continue
            if parameter_count(value) != EXPECTED_PARAMETER_COUNT:
                continue
            if not state_dict_exact_match(
                value,
                checkpoint_state_dict,
            ):
                continue

            captured["model"] = value
            captured["class_name"] = value.__class__.__name__
            captured["local_name"] = local_name
            captured["line"] = int(frame.f_lineno)
            captured["function"] = frame.f_code.co_name
            raise _OfficialModelCaptured()

        return tracer

    old_argv = list(sys.argv)
    old_trace = sys.gettrace()
    exit_observation = None

    try:
        sys.argv = [
            str(exporter_path),
            "--repo",
            str(repo),
            "--data-link",
            str(data_link),
            "--output-dir",
            str(probe_dir),
            "--installed-script",
            str(exporter_path),
        ]
        sys.settrace(tracer)
        try:
            exporter_module.main()
            exit_observation = "main_returned_without_capture"
        except _OfficialModelCaptured:
            exit_observation = "captured_exact_checkpoint_model"
        except SystemExit as exc:
            exit_observation = f"system_exit_before_capture:{exc.code}"
    finally:
        sys.settrace(old_trace)
        sys.argv = old_argv

    require(
        "model" in captured,
        "official D1 exporter did not expose a model whose full state_dict "
        "exactly matches the frozen checkpoint; "
        f"observation={exit_observation}",
    )

    model = captured["model"]
    model.to("cpu")
    model.eval()
    require(
        state_dict_exact_match(model, checkpoint_state_dict),
        "captured official model no longer matches checkpoint after capture",
    )

    provenance = {
        "route": "official_D1_exporter_trace_capture",
        "exporter": str(exporter_path),
        "exporter_sha256": sha256_file(exporter_path),
        "captured_class": captured["class_name"],
        "captured_local_name": captured["local_name"],
        "captured_function": captured["function"],
        "captured_line": captured["line"],
        "exit_observation": exit_observation,
        "state_dict_exact_match": True,
        "parameter_count": parameter_count(model),
        "validation_inference_completed_by_probe": False,
        "sealed_test_access": False,
    }
    atomic_json(
        run_dir / "F5_R1_OFFICIAL_EXPORTER_MODEL_CAPTURE.json",
        provenance,
    )
    return model, captured["class_name"], provenance


def instantiate_model(
    *,
    model_module: Any,
    exporter_path: Path,
    repo: Path,
    data_link: Path,
    run_dir: Path,
    checkpoint_state_dict: dict[str, torch.Tensor],
) -> tuple[torch.nn.Module, str, dict[str, Any]]:
    class_name = "V6P0Dynamic70GraphConvCount4"
    direct_failure = None

    if hasattr(model_module, class_name):
        candidate = getattr(model_module, class_name)
        require(inspect.isclass(candidate), f"{class_name} is not a class")
        try:
            model = candidate()
            return model, class_name, {
                "route": "zero_argument_constructor",
                "constructor_signature": str(inspect.signature(candidate)),
            }
        except TypeError as exc:
            direct_failure = repr(exc)

    model, resolved_class, provenance = capture_model_via_official_exporter(
        exporter_path=exporter_path,
        repo=repo,
        data_link=data_link,
        run_dir=run_dir,
        checkpoint_state_dict=checkpoint_state_dict,
    )
    provenance["zero_argument_constructor_failure"] = direct_failure
    return model, resolved_class, provenance


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def model_output_mapping(output: Any) -> dict[str, torch.Tensor]:
    aliases = {
        "graph": (
            "attack_logits",
            "graph_logits",
            "graph",
            "attack",
            "y_attack",
        ),
        "count": (
            "count_logits",
            "attacker_count_logits",
            "count",
        ),
        "source": ("source_logits", "source"),
        "transit": ("transit_logits", "transit"),
        "victim": ("victim_logits", "victim"),
        "path": ("path_logits", "attack_path_logits", "path"),
    }

    result: dict[str, torch.Tensor] = {}

    if isinstance(output, Mapping):
        normalized = {
            normalize_name(str(key)): value
            for key, value in output.items()
        }
        for head, candidates in aliases.items():
            for candidate in candidates:
                normalized_candidate = normalize_name(candidate)
                if normalized_candidate in normalized:
                    result[head] = normalized[normalized_candidate]
                    break

    elif isinstance(output, (tuple, list)) and len(output) == 6:
        result = {
            head: value
            for head, value in zip(
                ("graph", "count", "source", "transit", "victim", "path"),
                output,
            )
        }

    else:
        for head, candidates in aliases.items():
            for candidate in candidates:
                if hasattr(output, candidate):
                    result[head] = getattr(output, candidate)
                    break

    require(
        set(result) == set(HEAD_WIDTHS),
        f"could not resolve all output heads; found={sorted(result)} "
        f"type={type(output).__name__}",
    )
    return result


def canonicalize_head(
    tensor: torch.Tensor,
    head: str,
    batch_size: int,
) -> np.ndarray:
    require(
        isinstance(tensor, torch.Tensor),
        f"{head} output is not a torch.Tensor",
    )
    array = tensor.detach().float().cpu().numpy()
    width = HEAD_WIDTHS[head]

    if head == "graph":
        array = np.asarray(array).reshape(batch_size, -1)
        require(
            array.shape[1] == 1,
            f"graph logit width mismatch: {array.shape}",
        )
        return array[:, 0].astype(np.float32, copy=False)

    array = np.asarray(array).reshape(batch_size, -1)
    require(
        array.shape[1] == width,
        f"{head} logit width mismatch: {array.shape}",
    )
    return array.astype(np.float32, copy=False)


def exporter_batch_size(exporter_path: Path) -> int | None:
    import ast

    text = exporter_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text, filename=str(exporter_path))
    constants: dict[str, int] = {}

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value_node = node.value
        if value_node is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except Exception:
            continue
        if not isinstance(value, int):
            continue

        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value

    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = ""
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        if function_name != "DataLoader":
            continue

        for keyword in node.keywords:
            if keyword.arg != "batch_size":
                continue
            try:
                value = ast.literal_eval(keyword.value)
            except Exception:
                if isinstance(keyword.value, ast.Name):
                    value = constants.get(keyword.value.id)
                else:
                    value = None
            if isinstance(value, int) and value > 0:
                candidates.append(value)

    unique = sorted(set(candidates))
    return unique[0] if len(unique) == 1 else None


def configure_runtime(device_argument: str) -> tuple[torch.device, dict[str, Any]]:
    if device_argument:
        device = torch.device(device_argument)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    random.seed(107)
    np.random.seed(107)
    torch.manual_seed(107)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(107)

    # F5-R3A recovered the exact official D1 runtime:
    # deterministic_algorithms_enabled=False
    # cudnn_deterministic=True
    # cudnn_benchmark=False
    # cuda_matmul_allow_tf32=False
    # cudnn_allow_tf32=True
    # float32_matmul_precision="highest"
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    runtime = {
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }

    if device.type == "cuda":
        runtime["cuda_device_name"] = torch.cuda.get_device_name(device)
    return device, runtime


def labels_from_cache(
    cache_arrays: dict[str, np.memmap],
) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(cache_arrays[key])
        for key in (
            "y_attack",
            "y_attacker_count",
            "y_source",
            "y_transit",
            "y_victim",
            "y_attack_path",
        )
    }


def unit_paths(unit_dir: Path) -> dict[str, Path]:
    return {
        "manifest": unit_dir / "UNIT_MANIFEST.json",
        "metrics": unit_dir / "METRICS.json",
        "complete": unit_dir / "UNIT_COMPLETE",
        "permutation": unit_dir / "permutation.npy",
    }


def open_logit_memmaps(
    unit_dir: Path,
    mode: str,
) -> dict[str, np.memmap]:
    arrays = {}
    for head, width in HEAD_WIDTHS.items():
        shape = (
            (EXPECTED_ITEMS,)
            if width == 1
            else (EXPECTED_ITEMS, width)
        )
        arrays[head] = np.lib.format.open_memmap(
            unit_dir / LOGIT_FILE_NAMES[head],
            mode=mode,
            dtype=np.float32,
            shape=shape,
        )
    return arrays


def load_unit_metrics(unit_dir: Path) -> dict[str, float]:
    paths = unit_paths(unit_dir)
    require(paths["complete"].is_file(), f"unit incomplete: {unit_dir}")
    require(paths["metrics"].is_file(), f"metrics missing: {unit_dir}")
    document = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    return {
        key: float(value)
        for key, value in document["metrics"].items()
    }


def execute_unit(
    *,
    unit_name: str,
    unit_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    cache_arrays: dict[str, np.memmap],
    labels: dict[str, np.ndarray],
    adapter_module: Any,
    group: str | None,
    seed: int | None,
) -> dict[str, float]:
    paths = unit_paths(unit_dir)
    if paths["complete"].is_file():
        return load_unit_metrics(unit_dir)

    unit_dir.mkdir(parents=True, exist_ok=True)
    logits = open_logit_memmaps(unit_dir, "w+")

    if group is None:
        permutation = None
        start_channel = None
        end_channel = None
        fixed_points = None
    else:
        require(seed is not None, "permutation seed missing")
        start_channel, end_channel = GROUPS[group]
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(EXPECTED_ITEMS).astype(np.int64)
        np.save(paths["permutation"], permutation, allow_pickle=False)
        fixed_points = int(
            np.sum(permutation == np.arange(EXPECTED_ITEMS))
        )

    atomic_json(
        paths["manifest"],
        {
            "unit_name": unit_name,
            "group": group,
            "seed": seed,
            "start_channel": start_channel,
            "end_channel_exclusive": end_channel,
            "fixed_points": fixed_points,
            "batch_size": batch_size,
            "item_count": EXPECTED_ITEMS,
            "status": "RUNNING",
            "started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, EXPECTED_ITEMS, batch_size):
            batch_end = min(batch_start + batch_size, EXPECTED_ITEMS)
            local_size = batch_end - batch_start

            x_batch = np.array(
                cache_arrays["x"][batch_start:batch_end],
                dtype=np.float32,
                copy=True,
            )

            if permutation is not None:
                donors = permutation[batch_start:batch_end]
                x_batch[
                    :,
                    :,
                    start_channel:end_channel,
                    :,
                ] = np.asarray(
                    cache_arrays["x"][
                        donors,
                        :,
                        start_channel:end_channel,
                        :,
                    ],
                    dtype=np.float32,
                )

            mask_batch = np.array(
                cache_arrays["physical_port_mask"][
                    batch_start:batch_end
                ],
                dtype=np.float32,
                copy=True,
            )

            x_tensor = torch.from_numpy(x_batch).to(
                device=device,
                dtype=torch.float32,
                non_blocking=False,
            )
            mask_tensor = torch.from_numpy(mask_batch).to(
                device=device,
                dtype=torch.float32,
                non_blocking=False,
            )

            output = model(x_tensor, mask_tensor)
            mapped = model_output_mapping(output)
            for head in HEAD_WIDTHS:
                logits[head][batch_start:batch_end] = canonicalize_head(
                    mapped[head],
                    head,
                    local_size,
                )

            if (
                batch_end == EXPECTED_ITEMS
                or batch_end % (batch_size * 20) == 0
            ):
                for array in logits.values():
                    array.flush()
                print(
                    f"unit_progress={unit_name}:{batch_end}/{EXPECTED_ITEMS}",
                    flush=True,
                )

    arrays_for_adapter = dict(labels)
    for head, key in (
        ("graph", "attack_logits"),
        ("count", "count_logits"),
        ("source", "source_logits"),
        ("transit", "transit_logits"),
        ("victim", "victim_logits"),
        ("path", "path_logits"),
    ):
        arrays_for_adapter[key] = np.asarray(logits[head])

    logit_keys = {
        "graph": "attack_logits",
        "count": "count_logits",
        "source": "source_logits",
        "transit": "transit_logits",
        "victim": "victim_logits",
        "path": "path_logits",
    }
    metrics = adapter_module.compute_metrics_from_arrays(
        arrays_for_adapter,
        logit_keys,
    )
    metrics = {
        key: float(metrics[key])
        for key in METRIC_NAMES
    }

    metrics_document = {
        "unit_name": unit_name,
        "group": group,
        "seed": seed,
        "fixed_points": fixed_points,
        "metrics": metrics,
        "logit_files": {
            head: {
                "path": str(unit_dir / LOGIT_FILE_NAMES[head]),
                "sha256": sha256_file(
                    unit_dir / LOGIT_FILE_NAMES[head]
                ),
            }
            for head in HEAD_WIDTHS
        },
        "permutation": (
            {
                "path": str(paths["permutation"]),
                "sha256": sha256_file(paths["permutation"]),
            }
            if permutation is not None
            else None
        ),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(paths["metrics"], metrics_document)
    atomic_json(
        paths["manifest"],
        {
            **json.loads(
                paths["manifest"].read_text(encoding="utf-8")
            ),
            "status": "PASS",
            "metrics_sha256": sha256_file(paths["metrics"]),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    atomic_text(paths["complete"], "UNIT_COMPLETE\n")
    return metrics


def t_critical_95(df: int) -> float:
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:
        # Conservative normal approximation fallback.
        return 1.96


def summary_statistics(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size == len(SEEDS), "summary requires ten repeats")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1))
    sem = std / math.sqrt(array.size)
    half = t_critical_95(array.size - 1) * sem

    try:
        from scipy.stats import ttest_1samp
        test = ttest_1samp(array, popmean=0.0)
        p_value = float(test.pvalue)
    except Exception:
        p_value = None

    return {
        "repeat_count": int(array.size),
        "values": [float(value) for value in array],
        "mean": mean,
        "standard_deviation": std,
        "standard_error": sem,
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean_ci95_lower": mean - half,
        "mean_ci95_upper": mean + half,
        "empirical_p2_5": float(np.quantile(array, 0.025)),
        "empirical_p97_5": float(np.quantile(array, 0.975)),
        "one_sample_ttest_two_sided_p": p_value,
        "positive_repeat_count": int(np.sum(array > 0)),
        "negative_repeat_count": int(np.sum(array < 0)),
        "zero_repeat_count": int(np.sum(array == 0)),
    }


def metric_importance(
    baseline: float,
    permuted: float,
    metric: str,
) -> float:
    if metric in HIGHER_IS_BETTER:
        return baseline - permuted
    return permuted - baseline


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(rows, f"cannot write empty CSV: {path}")
    keys = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def make_figures(
    output_dir: Path,
    summaries: dict[str, dict[str, dict[str, Any]]],
) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    groups = list(GROUPS)
    selection_means = [
        summaries[group]["selection_score"]["mean"]
        for group in groups
    ]
    selection_errors = [
        (
            summaries[group]["selection_score"]["mean_ci95_upper"]
            - summaries[group]["selection_score"]["mean"]
        )
        for group in groups
    ]

    selection_path = output_dir / "F5_SELECTION_SCORE_IMPORTANCE.png"
    figure = plt.figure(figsize=(11, 6))
    axis = figure.add_subplot(111)
    axis.bar(
        range(len(groups)),
        selection_means,
        yerr=selection_errors,
        capsize=4,
    )
    axis.axhline(0.0, linewidth=1)
    axis.set_xticks(range(len(groups)))
    axis.set_xticklabels(
        [group.replace("_", "\n") for group in groups],
        rotation=0,
    )
    axis.set_ylabel("Baseline − permuted selection score")
    axis.set_title("Dynamic70 macro-group permutation importance")
    figure.tight_layout()
    figure.savefig(selection_path, dpi=180)
    plt.close(figure)

    heatmap_metrics = [
        "selection_score",
        "graph_ap",
        "source_ap",
        "transit_ap",
        "victim_ap",
        "path_ap",
        "count_active_macro_f1",
        "graph_fpr_at_0_5",
    ]
    matrix = np.asarray(
        [
            [
                summaries[group][metric]["mean"]
                for group in groups
            ]
            for metric in heatmap_metrics
        ],
        dtype=np.float64,
    )

    heatmap_path = output_dir / "F5_TASKWISE_IMPORTANCE_HEATMAP.png"
    figure = plt.figure(figsize=(12, 7))
    axis = figure.add_subplot(111)
    image = axis.imshow(matrix, aspect="auto")
    axis.set_xticks(range(len(groups)))
    axis.set_xticklabels(
        [group.replace("_", "\n") for group in groups],
    )
    axis.set_yticks(range(len(heatmap_metrics)))
    axis.set_yticklabels(heatmap_metrics)
    axis.set_title("Mean permutation importance by task metric")
    figure.colorbar(image, ax=axis, label="Performance degradation")
    figure.tight_layout()
    figure.savefig(heatmap_path, dpi=180)
    plt.close(figure)

    return [selection_path, heatmap_path]


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_root = Path(args.output_root).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")
    require(data_link.is_symlink(), f"dataset symlink missing: {data_link}")
    data_root = data_link.resolve()
    require(data_root.is_dir(), f"dataset target missing: {data_root}")

    preflight_report_path = output_root / (
        "V5_P3_F5_P0_GROUP_BLOCK_PERMUTATION_PREFLIGHT_REPORT.json"
    )
    preflight_lock_path = output_root / (
        "V5_P3_F5_P0_GROUP_BLOCK_PERMUTATION_PREFLIGHT_LOCK.json"
    )
    route_path = output_root / "F5_P0_EXECUTION_ROUTE_SOURCE_INVENTORY.json"
    protocol_path = output_root / "F5_P0_GROUP_BLOCK_PERMUTATION_PROTOCOL.json"

    preflight_report, preflight_lock = verify_report_lock(
        preflight_report_path,
        preflight_lock_path,
    )
    require(
        preflight_lock.get("actual_F5_execution_authorized") is True,
        "F5-P0 did not authorize actual F5 execution",
    )
    require(
        preflight_lock.get("F6_authorized") is False,
        "F6 must remain unauthorized during F5",
    )
    require(route_path.is_file(), f"route inventory missing: {route_path}")
    require(protocol_path.is_file(), f"protocol missing: {protocol_path}")
    require(
        preflight_lock.get("route_inventory_sha256")
        == sha256_file(route_path),
        "F5-P0 route inventory hash mismatch",
    )
    require(
        preflight_lock.get("protocol_sha256")
        == sha256_file(protocol_path),
        "F5-P0 protocol hash mismatch",
    )

    r3a_report_path = output_root / (
        "V5_P3_F5_R3A_BASELINE_PROCESS_RUNTIME_REVIEW_REPORT.json"
    )
    r3a_lock_path = output_root / (
        "V5_P3_F5_R3A_BASELINE_PROCESS_RUNTIME_REVIEW_LOCK.json"
    )
    r3a_report, r3a_lock = verify_report_lock(
        r3a_report_path,
        r3a_lock_path,
    )
    require(
        r3a_lock.get("classification")
        == "F5_RUNTIME_CONFIGURATION_DRIFT_CONFIRMED",
        "F5-R3B expected confirmed runtime drift",
    )
    require(
        r3a_lock.get("runtime_drift_confirmed") is True,
        "F5-R3A did not confirm runtime drift",
    )
    require(
        r3a_lock.get("repair_authorized") is True,
        "F5-R3A did not authorize runtime repair",
    )
    require(
        r3a_lock.get("F5_permutation_units_authorized") is False,
        "permutation units must remain held until repaired baseline passes",
    )

    route = json.loads(route_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol.get("coverage_exact") is True, "group coverage not frozen")
    require(protocol.get("repeats") == 10, "repeat count changed")
    require(protocol.get("seeds") == list(SEEDS), "seed list changed")
    require(protocol.get("groups") == {
        group: {
            "start": start,
            "end_exclusive": end,
            "count": end - start,
        }
        for group, (start, end) in GROUPS.items()
    }, "macro-group contract changed")

    certified = route["certified_files"]
    source_paths = {}
    for label in ("model", "loader", "exporter", "checkpoint", "adapter"):
        row = certified[label]
        path = Path(row["path"]).resolve()
        require(path.is_file(), f"{label} missing: {path}")
        require(
            sha256_file(path) == row["actual_sha256"],
            f"{label} hash changed: {path}",
        )
        source_paths[label] = path

    run_dir = choose_run_dir(output_root, args.run_dir)
    cache_dir = run_dir / "validation_cache"
    units_dir = run_dir / "units"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    device, runtime = configure_runtime(args.device)
    derived_batch_size = exporter_batch_size(source_paths["exporter"])
    batch_size = (
        args.batch_size
        if args.batch_size > 0
        else derived_batch_size or 128
    )
    require(batch_size > 0, "invalid batch size")

    run_contract_path = run_dir / "F5_RUN_CONTRACT.json"
    runtime_override_path = run_dir / (
        "F5_R3B_OFFICIAL_D1_RUNTIME_RECOVERY_OVERRIDE.json"
    )
    previous_contract = None

    if run_contract_path.is_file():
        previous_contract = json.loads(
            run_contract_path.read_text(encoding="utf-8")
        )
        require(
            previous_contract["batch_size"] == batch_size,
            "resume batch size differs from frozen run contract",
        )
        require(
            previous_contract["device"] == str(device),
            "resume device differs from frozen run contract",
        )

        failed_runtime = previous_contract["runtime"]
        require(
            failed_runtime[
                "deterministic_algorithms_enabled"
            ] is True,
            "historical F5 run contract does not contain the diagnosed "
            "deterministic-algorithms setting",
        )
        require(
            failed_runtime["cudnn_allow_tf32"] is False,
            "historical F5 run contract does not contain the diagnosed "
            "cuDNN TF32-disabled setting",
        )
    else:
        atomic_json(
            run_contract_path,
            {
                "stage": STAGE,
                "campaign": CAMPAIGN,
                "classification": CLASSIFICATION,
                "run_directory": str(run_dir),
                "data_link": str(data_link),
                "data_root": str(data_root),
                "batch_size": batch_size,
                "batch_size_source": (
                    "CLI"
                    if args.batch_size > 0
                    else (
                        "official_D1_exporter"
                        if derived_batch_size is not None
                        else "fallback_128"
                    )
                ),
                "device": str(device),
                "runtime": runtime,
                "groups": GROUPS,
                "seeds": list(SEEDS),
                "source_files": {
                    label: {
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                    for label, path in source_paths.items()
                },
                "model_or_checkpoint_change": False,
                "threshold_change": False,
                "metric_formula_change": False,
                "sealed_test_access": False,
                "recovery": {
                    "stage": (
                        "V5_P3_F5_R1_OFFICIAL_EXPORTER_MODEL_"
                        "CONSTRUCTION_RECOVERY"
                    ),
                    "failure_classification": (
                        "zero_argument_model_constructor_mismatch"
                    ),
                },
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    official_runtime_expected = {
        "deterministic_algorithms_enabled": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "float32_matmul_precision": "highest",
    }
    observed_runtime = {
        key: runtime[key]
        for key in official_runtime_expected
    }
    require(
        observed_runtime == official_runtime_expected,
        "configured F5-R3B runtime does not exactly match the official "
        f"D1 runtime: {observed_runtime}",
    )

    runtime_override = {
        "stage": (
            "V5_P3_F5_R3B_OFFICIAL_RUNTIME_BASELINE_RECOVERY_"
            "AND_PERMUTATION_RESUME"
        ),
        "status": "FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authority": {
            "F5_R3A_report": str(r3a_report_path),
            "F5_R3A_report_sha256": sha256_file(r3a_report_path),
            "F5_R3A_lock": str(r3a_lock_path),
            "F5_R3A_lock_sha256": sha256_file(r3a_lock_path),
            "classification": r3a_lock["classification"],
            "runtime_drift_confirmed": True,
            "repair_authorized": True,
        },
        "historical_failed_runtime": (
            previous_contract["runtime"]
            if previous_contract is not None
            else None
        ),
        "official_recovered_runtime": runtime,
        "minimum_causal_subset": ["cudnn_allow_tf32"],
        "causal_change": {
            "cudnn_allow_tf32": {
                "failed_F5_value": False,
                "official_D1_value": True,
            }
        },
        "full_official_alignment": {
            "deterministic_algorithms_enabled": {
                "failed_F5_value": True,
                "official_D1_value": False,
            },
            "cudnn_allow_tf32": {
                "failed_F5_value": False,
                "official_D1_value": True,
            },
        },
        "historical_run_contract_modified": False,
        "old_baseline_overwritten": False,
        "new_baseline_directory": "units/baseline_official_runtime",
        "new_gate_artifact": (
            "results/"
            "F5_R3B_OFFICIAL_RUNTIME_BASELINE_REPRODUCTION_GATE.json"
        ),
        "sealed_test_access": False,
    }

    if runtime_override_path.is_file():
        existing_override = json.loads(
            runtime_override_path.read_text(encoding="utf-8")
        )
        require(
            existing_override["official_recovered_runtime"]
            == runtime_override["official_recovered_runtime"],
            "existing runtime override differs from current official runtime",
        )
        require(
            existing_override["minimum_causal_subset"]
            == ["cudnn_allow_tf32"],
            "existing runtime override causal subset changed",
        )
    else:
        atomic_json(runtime_override_path, runtime_override)

    loader_module = import_source(
        source_paths["loader"],
        "_v5_p3_f5_guarded_loader",
    )
    dataset, dataset_provenance = construct_validation_dataset(
        loader_module,
        data_root,
    )
    require(
        len(dataset) == EXPECTED_ITEMS,
        f"validation dataset length mismatch: {len(dataset)}",
    )

    completed_permutation_units_before_repair = sorted(
        str(path)
        for path in units_dir.glob("*/seed_*/UNIT_COMPLETE")
    )
    require(
        not completed_permutation_units_before_repair,
        "permutation units exist before the official-runtime baseline "
        f"repair: {completed_permutation_units_before_repair}",
    )

    cache_summary = build_or_resume_cache(
        dataset,
        cache_dir,
        dataset_provenance,
    )
    cache_arrays = open_cache_arrays(cache_dir, "r")
    labels = labels_from_cache(cache_arrays)

    checkpoint = torch.load(
        source_paths["checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    require(isinstance(checkpoint, dict), "checkpoint is not a dictionary")
    state_dict = extract_state_dict(checkpoint)

    edge_buffer_elements = sum(
        int(value.numel())
        for key, value in state_dict.items()
        if normalize_name(str(key)).endswith("edge_index")
        and list(value.shape) == [2, 48]
    )
    require(
        edge_buffer_elements == EXPECTED_EDGE_BUFFER_ELEMENTS,
        f"edge buffer mismatch: {edge_buffer_elements}",
    )

    model_module = import_source(
        source_paths["model"],
        "_v5_p3_f5_model",
    )
    model, model_class, model_construction = instantiate_model(
        model_module=model_module,
        exporter_path=source_paths["exporter"],
        repo=repo,
        data_link=data_link,
        run_dir=run_dir,
        checkpoint_state_dict=state_dict,
    )
    require(
        parameter_count(model) == EXPECTED_PARAMETER_COUNT,
        f"model parameter count mismatch: {parameter_count(model)}",
    )

    if not state_dict_exact_match(model, state_dict):
        load_result = model.load_state_dict(state_dict, strict=True)
        require(
            not load_result.missing_keys
            and not load_result.unexpected_keys,
            f"checkpoint load mismatch: {load_result}",
        )

    require(
        state_dict_exact_match(model, state_dict),
        "constructed model does not exactly match frozen checkpoint",
    )
    model.to(device)

    adapter_module = import_source(
        source_paths["adapter"],
        "_v5_p3_f5_metric_adapter",
    )
    require(
        hasattr(adapter_module, "compute_metrics_from_arrays"),
        "canonical adapter missing compute_metrics_from_arrays",
    )

    baseline_dir = units_dir / "baseline_official_runtime"
    baseline_metrics = execute_unit(
        unit_name="baseline_official_runtime",
        unit_dir=baseline_dir,
        model=model,
        device=device,
        batch_size=batch_size,
        cache_arrays=cache_arrays,
        labels=labels,
        adapter_module=adapter_module,
        group=None,
        seed=None,
    )

    metric_dir = output_root.parent / "metric_adapter"
    fresh_cert_path = metric_dir / (
        "F4M_R3A_FRESH_REPLAY_14_METRIC_CERTIFICATION.json"
    )
    require(
        fresh_cert_path.is_file(),
        f"F4M fresh certification missing: {fresh_cert_path}",
    )
    fresh_cert = json.loads(fresh_cert_path.read_text(encoding="utf-8"))
    certified_fresh_metrics = {
        key: float(value)
        for key, value in fresh_cert["metrics"].items()
        if key in METRIC_NAMES
    }

    baseline_gate = {}
    baseline_gate_pass = True
    for metric in METRIC_NAMES:
        difference = abs(
            baseline_metrics[metric] - certified_fresh_metrics[metric]
        )
        passed = difference <= 1e-6
        baseline_gate_pass &= passed
        baseline_gate[metric] = {
            "same_runtime_baseline": baseline_metrics[metric],
            "F4M_fresh_certified": certified_fresh_metrics[metric],
            "absolute_difference": difference,
            "tolerance": 1e-6,
            "pass": passed,
        }

    repaired_gate_path = results_dir / (
        "F5_R3B_OFFICIAL_RUNTIME_BASELINE_REPRODUCTION_GATE.json"
    )
    atomic_json(
        repaired_gate_path,
        {
            "stage": (
                "V5_P3_F5_R3B_OFFICIAL_RUNTIME_BASELINE_RECOVERY_"
                "AND_PERMUTATION_RESUME"
            ),
            "status": "PASS" if baseline_gate_pass else "FAIL",
            "runtime_override": str(runtime_override_path),
            "runtime_override_sha256": sha256_file(runtime_override_path),
            "baseline_unit": str(baseline_dir),
            "all_14_metrics_pass": baseline_gate_pass,
            "metrics": baseline_gate,
            "historical_failed_baseline_preserved": str(
                units_dir / "baseline"
            ),
            "historical_failed_gate_preserved": str(
                results_dir / "F5_BASELINE_REPRODUCTION_GATE.json"
            ),
        },
    )
    require(
        baseline_gate_pass,
        "official-runtime unpermuted baseline did not reproduce the "
        "F4M fresh-certified metric vector within 1e-6",
    )

    r3b_report_path = output_root / (
        "V5_P3_F5_R3B_OFFICIAL_RUNTIME_BASELINE_RECOVERY_REPORT.json"
    )
    r3b_lock_path = output_root / (
        "V5_P3_F5_R3B_OFFICIAL_RUNTIME_BASELINE_RECOVERY_LOCK.json"
    )
    r3b_complete_path = output_root / (
        "V5_P3_F5_R3B_OFFICIAL_RUNTIME_BASELINE_RECOVERY_COMPLETE"
    )
    r3b_report = {
        "stage": (
            "V5_P3_F5_R3B_OFFICIAL_RUNTIME_BASELINE_RECOVERY"
        ),
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": "VALIDATION-EXPLORATORY runtime recovery",
        "finding": {
            "confirmed_causal_runtime_field": "cudnn_allow_tf32",
            "historical_failed_value": False,
            "official_D1_value": True,
            "official_full_runtime": runtime,
            "baseline_all_14_metrics_pass": True,
        },
        "decision": {
            "F5_baseline_gate_complete": True,
            "F5_permutation_units_authorized": True,
            "F6_integrated_gradients_authorized": False,
            "F7_retraining_ablation_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F5_DYNAMIC70_GROUP_BLOCK_PERMUTATION_RESUME"
            ),
        },
        "governance": {
            "historical_run_contract_modified": False,
            "historical_failed_baseline_overwritten": False,
            "validation_cache_rebuilt": False,
            "model_weights_changed": False,
            "thresholds_changed": False,
            "metric_formulas_changed": False,
            "sealed_test_tensors_loaded": False,
        },
        "artifacts": {
            "runtime_override": str(runtime_override_path),
            "repaired_baseline_unit": str(baseline_dir),
            "repaired_baseline_gate": str(repaired_gate_path),
        },
        "provenance": {
            "F5_R3A_report_sha256": sha256_file(r3a_report_path),
            "F5_R3A_lock_sha256": sha256_file(r3a_lock_path),
            "runtime_override_sha256": sha256_file(runtime_override_path),
            "repaired_baseline_metrics_sha256": sha256_file(
                baseline_dir / "METRICS.json"
            ),
            "repaired_baseline_gate_sha256": sha256_file(
                repaired_gate_path
            ),
            "installed_script_sha256": sha256_file(installed_script),
        },
    }
    atomic_json(r3b_report_path, r3b_report)
    atomic_json(
        r3b_lock_path,
        {
            "stage": r3b_report["stage"],
            "status": "PASS",
            "report_sha256": sha256_file(r3b_report_path),
            "runtime_override_sha256": sha256_file(runtime_override_path),
            "repaired_baseline_metrics_sha256": sha256_file(
                baseline_dir / "METRICS.json"
            ),
            "repaired_baseline_gate_sha256": sha256_file(
                repaired_gate_path
            ),
            "F5_baseline_gate_complete": True,
            "F5_permutation_units_authorized": True,
            "F6_authorized": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(
        r3b_complete_path,
        "V5_P3_F5_R3B_OFFICIAL_RUNTIME_BASELINE_RECOVERY_COMPLETE\n",
    )

    print(
        "V5_P3_F5_R3B_OFFICIAL_RUNTIME_BASELINE_RECOVERY_COMPLETE",
        flush=True,
    )
    print("recovery_status=PASS", flush=True)
    print("confirmed_causal_runtime_field=cudnn_allow_tf32", flush=True)
    print("historical_failed_cudnn_allow_tf32=false", flush=True)
    print("official_D1_cudnn_allow_tf32=true", flush=True)
    print(
        "official_D1_deterministic_algorithms_enabled=false",
        flush=True,
    )
    print("repaired_baseline_all_14_metrics_pass=true", flush=True)
    print("F5_permutation_units_authorized=true", flush=True)
    print("F6_integrated_gradients_authorized=false", flush=True)
    print(
        "next_stage="
        "V5_P3_F5_DYNAMIC70_GROUP_BLOCK_PERMUTATION_RESUME",
        flush=True,
    )

    repeat_metrics: dict[str, dict[int, dict[str, float]]] = {
        group: {}
        for group in GROUPS
    }

    for group in GROUPS:
        for seed in SEEDS:
            unit_name = f"{group}__seed_{seed}"
            unit_dir = units_dir / group / f"seed_{seed}"
            print(
                f"starting_unit={unit_name}",
                flush=True,
            )
            metrics = execute_unit(
                unit_name=unit_name,
                unit_dir=unit_dir,
                model=model,
                device=device,
                batch_size=batch_size,
                cache_arrays=cache_arrays,
                labels=labels,
                adapter_module=adapter_module,
                group=group,
                seed=seed,
            )
            repeat_metrics[group][seed] = metrics
            print(
                f"completed_unit={unit_name}:"
                f"selection_score={metrics['selection_score']}",
                flush=True,
            )

    importance_values: dict[str, dict[str, list[float]]] = {
        group: {
            metric: []
            for metric in METRIC_NAMES
        }
        for group in GROUPS
    }
    repeat_rows = []

    for group in GROUPS:
        for seed in SEEDS:
            metrics = repeat_metrics[group][seed]
            for metric in METRIC_NAMES:
                importance = metric_importance(
                    baseline_metrics[metric],
                    metrics[metric],
                    metric,
                )
                importance_values[group][metric].append(importance)
                repeat_rows.append({
                    "group": group,
                    "seed": seed,
                    "metric": metric,
                    "baseline_value": baseline_metrics[metric],
                    "permuted_value": metrics[metric],
                    "importance": importance,
                    "importance_definition": (
                        "baseline_minus_permuted"
                        if metric in HIGHER_IS_BETTER
                        else "permuted_minus_baseline"
                    ),
                })

    summaries = {
        group: {
            metric: summary_statistics(
                importance_values[group][metric]
            )
            for metric in METRIC_NAMES
        }
        for group in GROUPS
    }

    ranking = sorted(
        (
            {
                "rank": 0,
                "group": group,
                "selection_score_importance_mean": summaries[group][
                    "selection_score"
                ]["mean"],
                "selection_score_importance_std": summaries[group][
                    "selection_score"
                ]["standard_deviation"],
                "selection_score_ci95_lower": summaries[group][
                    "selection_score"
                ]["mean_ci95_lower"],
                "selection_score_ci95_upper": summaries[group][
                    "selection_score"
                ]["mean_ci95_upper"],
            }
            for group in GROUPS
        ),
        key=lambda row: row["selection_score_importance_mean"],
        reverse=True,
    )
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index

    summary_rows = []
    for group in GROUPS:
        for metric in METRIC_NAMES:
            statistics = summaries[group][metric]
            summary_rows.append({
                "group": group,
                "metric": metric,
                "mean_importance": statistics["mean"],
                "standard_deviation": statistics["standard_deviation"],
                "standard_error": statistics["standard_error"],
                "median": statistics["median"],
                "minimum": statistics["minimum"],
                "maximum": statistics["maximum"],
                "mean_ci95_lower": statistics["mean_ci95_lower"],
                "mean_ci95_upper": statistics["mean_ci95_upper"],
                "positive_repeat_count": statistics[
                    "positive_repeat_count"
                ],
                "negative_repeat_count": statistics[
                    "negative_repeat_count"
                ],
                "one_sample_ttest_two_sided_p": statistics[
                    "one_sample_ttest_two_sided_p"
                ],
            })

    repeat_csv = results_dir / "F5_ALL_REPEAT_METRIC_DELTAS.csv"
    summary_csv = results_dir / "F5_GROUP_METRIC_SUMMARY.csv"
    ranking_csv = results_dir / "F5_SELECTION_SCORE_GROUP_RANKING.csv"
    write_csv(repeat_csv, repeat_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(ranking_csv, ranking)

    results_json = results_dir / "F5_GROUP_BLOCK_PERMUTATION_RESULTS.json"
    atomic_json(
        results_json,
        {
            "stage": STAGE,
            "classification": CLASSIFICATION,
            "baseline_metrics": baseline_metrics,
            "repeat_metrics": {
                group: {
                    str(seed): metrics
                    for seed, metrics in values.items()
                }
                for group, values in repeat_metrics.items()
            },
            "importance_definition": {
                "higher_is_better": "baseline - permuted",
                "graph_fpr_at_0_5": "permuted - baseline",
                "positive_value": "performance degradation after permutation",
            },
            "summaries": summaries,
            "selection_score_ranking": ranking,
        },
    )

    figure_paths = make_figures(results_dir, summaries)

    top_group = ranking[0]["group"]
    completion_report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "scope": (
            "Ten-repeat sample-axis permutation of each complete Dynamic70 "
            "macro-group block using the frozen seed-107 checkpoint and "
            "canonical 14-metric adapter."
        ),
        "execution": {
            "run_directory": str(run_dir),
            "validation_items": EXPECTED_ITEMS,
            "batch_size": batch_size,
            "device": str(device),
            "model_class": model_class,
            "model_construction": model_construction,
            "parameter_count": parameter_count(model),
            "edge_buffer_elements": edge_buffer_elements,
            "group_count": len(GROUPS),
            "repeat_count_per_group": len(SEEDS),
            "permuted_inference_units": len(GROUPS) * len(SEEDS),
            "baseline_reproduction_pass": baseline_gate_pass,
            "runtime_recovery": {
                "classification": (
                    "F5_RUNTIME_CONFIGURATION_DRIFT_CONFIRMED"
                ),
                "confirmed_causal_field": "cudnn_allow_tf32",
                "failed_value": False,
                "official_value": True,
                "official_runtime": runtime,
                "historical_failed_baseline_preserved": True,
            },
        },
        "protocol": {
            "groups": GROUPS,
            "seeds": list(SEEDS),
            "permutation_unit": "[16,C_group,32] complete group block",
            "permutation_axis": "validation sample",
            "labels_permuted": False,
            "physical_port_mask_permuted": False,
            "edge_index_permuted": False,
            "fixed_points_allowed": True,
        },
        "results": {
            "baseline_metrics": baseline_metrics,
            "selection_score_ranking": ranking,
            "top_selection_score_group": top_group,
            "summaries": summaries,
        },
        "decision": {
            "F5_complete": True,
            "F5_result_review_authorized": True,
            "F6_integrated_gradients_authorized": False,
            "F7_retraining_ablation_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F5R_GROUP_BLOCK_PERMUTATION_RESULT_REVIEW"
            ),
        },
        "governance": {
            "model_weights_changed": False,
            "checkpoint_changed": False,
            "thresholds_changed": False,
            "metric_formulas_changed": False,
            "training_tensors_loaded": False,
            "validation_tensors_loaded": True,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "results_are_independent_validation": False,
        },
        "artifacts": {
            "run_contract": str(run_contract_path),
            "cache_completion": str(cache_dir / "CACHE_COMPLETE.json"),
            "baseline_gate": str(repaired_gate_path),
            "runtime_recovery_override": str(runtime_override_path),
            "runtime_recovery_report": str(r3b_report_path),
            "runtime_recovery_lock": str(r3b_lock_path),
            "results_json": str(results_json),
            "repeat_csv": str(repeat_csv),
            "summary_csv": str(summary_csv),
            "ranking_csv": str(ranking_csv),
            "figures": [str(path) for path in figure_paths],
        },
        "provenance": {
            "F5_P0_report_sha256": sha256_file(preflight_report_path),
            "F5_R3A_report_sha256": sha256_file(r3a_report_path),
            "F5_R3A_lock_sha256": sha256_file(r3a_lock_path),
            "F5_P0_lock_sha256": sha256_file(preflight_lock_path),
            "route_inventory_sha256": sha256_file(route_path),
            "protocol_sha256": sha256_file(protocol_path),
            "model_sha256": sha256_file(source_paths["model"]),
            "loader_sha256": sha256_file(source_paths["loader"]),
            "checkpoint_sha256": sha256_file(source_paths["checkpoint"]),
            "adapter_sha256": sha256_file(source_paths["adapter"]),
            "installed_script_sha256": sha256_file(installed_script),
            "results_json_sha256": sha256_file(results_json),
            "repeat_csv_sha256": sha256_file(repeat_csv),
            "summary_csv_sha256": sha256_file(summary_csv),
            "ranking_csv_sha256": sha256_file(ranking_csv),
        },
    }

    report_path = run_dir / f"{STAGE}_REPORT.json"
    lock_path = run_dir / f"{STAGE}_LOCK.json"
    complete_path = run_dir / f"{STAGE}_COMPLETE"

    atomic_json(report_path, completion_report)
    atomic_json(
        lock_path,
        {
            "stage": STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(report_path),
            "run_contract_sha256": sha256_file(run_contract_path),
            "cache_completion_sha256": sha256_file(
                cache_dir / "CACHE_COMPLETE.json"
            ),
            "baseline_gate_sha256": sha256_file(
                repaired_gate_path
            ),
            "runtime_recovery_override_sha256": sha256_file(
                runtime_override_path
            ),
            "runtime_recovery_report_sha256": sha256_file(
                r3b_report_path
            ),
            "runtime_recovery_lock_sha256": sha256_file(
                r3b_lock_path
            ),
            "results_json_sha256": sha256_file(results_json),
            "repeat_csv_sha256": sha256_file(repeat_csv),
            "summary_csv_sha256": sha256_file(summary_csv),
            "ranking_csv_sha256": sha256_file(ranking_csv),
            "F5_complete": True,
            "F5R_authorized": True,
            "F6_authorized": False,
            "F7_authorized": False,
            "validation_tensors_loaded": True,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    canonical_report_path = output_root / f"{STAGE}_REPORT.json"
    canonical_lock_path = output_root / f"{STAGE}_LOCK.json"
    canonical_complete_path = output_root / f"{STAGE}_COMPLETE"
    atomic_json(
        canonical_report_path,
        {
            **completion_report,
            "run_report": str(report_path),
            "run_report_sha256": sha256_file(report_path),
            "run_lock": str(lock_path),
            "run_lock_sha256": sha256_file(lock_path),
        },
    )
    atomic_json(
        canonical_lock_path,
        {
            "stage": STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(canonical_report_path),
            "run_report_sha256": sha256_file(report_path),
            "run_lock_sha256": sha256_file(lock_path),
            "F5_complete": True,
            "F5R_authorized": True,
            "F6_authorized": False,
            "F7_authorized": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(canonical_complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print(f"run_directory={run_dir}")
    print(f"validation_items={EXPECTED_ITEMS}")
    print(f"batch_size={batch_size}")
    print(f"device={device}")
    print("runtime_recovery_applied=true")
    print("confirmed_causal_runtime_field=cudnn_allow_tf32")
    print("cudnn_allow_tf32=true")
    print("deterministic_algorithms_enabled=false")
    print(f"model_class={model_class}")
    print(f"model_construction_route={model_construction['route']}")
    print(
        "model_state_dict_exact_checkpoint_match="
        f"{str(state_dict_exact_match(model, state_dict)).lower()}"
    )
    print(f"parameter_count={parameter_count(model)}")
    print("baseline_reproduction_pass=true")
    print(f"group_count={len(GROUPS)}")
    print(f"repeat_count_per_group={len(SEEDS)}")
    print(f"permuted_inference_units={len(GROUPS) * len(SEEDS)}")
    for row in ranking:
        print(
            "selection_score_ranking="
            f"{row['rank']}:{row['group']}:"
            f"mean={row['selection_score_importance_mean']}:"
            f"std={row['selection_score_importance_std']}:"
            f"ci95=[{row['selection_score_ci95_lower']},"
            f"{row['selection_score_ci95_upper']}]"
        )
    print(f"top_selection_score_group={top_group}")
    print("F5_complete=true")
    print("F5_result_review_authorized=true")
    print("F6_integrated_gradients_authorized=false")
    print("F7_retraining_ablation_authorized=false")
    print("model_weights_changed=false")
    print("validation_tensors_loaded=true")
    print("sealed_test_tensors_loaded=false")
    print(
        "next_stage="
        "V5_P3_F5R_GROUP_BLOCK_PERMUTATION_RESULT_REVIEW"
    )
    print(f"results_json={results_json}")
    print(f"repeat_csv={repeat_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"ranking_csv={ranking_csv}")
    for path in figure_paths:
        print(f"figure={path}")
    print(f"report={canonical_report_path}")
    print(f"lock={canonical_lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
