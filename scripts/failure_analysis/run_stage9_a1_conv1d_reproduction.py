#!/usr/bin/env python3
"""
Run Stage 9 A1: exact Conv1D baseline reproduction.

This runner:
1. Verifies frozen original artifacts by SHA-256.
2. Performs dataset, split, model, class-weight, CUDA, and VRAM checks.
3. Prints the exact training command in --dry-run mode.
4. Launches the original training script without changing its architecture,
   sampler, losses, checkpoint rule, or hyperparameters.
5. Streams training output to both the terminal and a persistent log.
6. Verifies outputs and split identity.
7. Compares the reproduction with the original summary.
8. Writes JSON, Markdown, and SHA-256 records.

It never deletes or overwrites the original baseline or an existing A1 output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import py_compile
import shlex
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCRIPT_VERSION = "1.0.0"

EXPECTED_HASHES: dict[str, str] = {
    "training_script": "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b",
    "original_checkpoint": "3ef62871cb62616bcb74bd627e1f30446a1d9a065628f162b83a793259798c20",
    "original_summary": "df6c47cc9e927796460bbd129492c7437f9424baada15ba4d1f288266e39eba0",
    "original_history": "912fa171a025ffdc4c4cca1779146d8d82d7ff2402e0677bf9f8cb8fb545eade",
    "original_splits": "4a58d9fa6667de6d15c5d6563db0e7a0ec306382cd9816900f7098c80bbee8a2",
}

EXPECTED_DATASET_SHAPES: dict[str, tuple[int, ...]] = {
    "x": (233803, 16, 8, 24),
    "y_graph": (233803,),
    "y_node": (233803, 16),
    "edge_index": (2, 64),
}

EXPECTED_SPLIT_SIZES: dict[str, int] = {
    "train_idx": 148185,
    "val_idx": 42809,
    "test_idx": 42809,
}

EXPECTED_PARAMETER_COUNT = 882
EXPECTED_GRAPH_POS_WEIGHT = 0.45161290322580644
EXPECTED_NODE_POS_WEIGHT = 13.4
MINIMUM_FREE_VRAM_GIB = 6.0

METRIC_TOLERANCES: dict[tuple[str, str], float] = {
    ("val", "f1"): 0.005,
    ("test", "f1"): 0.010,
    ("val", "node_f1"): 0.010,
    ("test", "node_f1"): 0.010,
    ("val", "exact_localization"): 0.015,
    ("test", "exact_localization"): 0.020,
}


class Stage9A1Error(RuntimeError):
    """Raised when an A1 safety or validation requirement fails."""


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    dataset_root: Path
    training_script: Path
    original_dir: Path
    output_dir: Path
    stage9_root: Path
    log_file: Path
    status_file: Path
    tables_dir: Path
    freeze_dir: Path
    summary_output: Path
    report_output: Path
    output_hashes_file: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise Stage9A1Error(f"Expected a JSON object in {path}.")
    return data


def resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise Stage9A1Error(f"Missing {label}: {path}")


def ensure_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise Stage9A1Error(f"Missing {label}: {path}")


def path_is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def import_training_module(path: Path) -> ModuleType:
    module_name = "stage9_a1_frozen_train_temporal_gcn_v3"
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise Stage9A1Error(f"Could not import training script: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    return value


def first_present(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def nested_get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def metric_value(summary: Mapping[str, Any], split: str, metric: str) -> float:
    block = summary.get(split)
    if not isinstance(block, Mapping) or metric not in block:
        raise Stage9A1Error(f"Missing metric {split}.{metric} in summary.")
    return float(block[metric])


def inspect_dataset(dataset_root: Path) -> dict[str, Any]:
    files = {
        "x": dataset_root / "x.npy",
        "y_graph": dataset_root / "y_graph.npy",
        "y_node": dataset_root / "y_node.npy",
        "edge_index": dataset_root / "edge_index.npy",
    }
    for name, path in files.items():
        ensure_file(path, f"dataset array {name}")

    arrays = {
        "x": np.load(files["x"], mmap_mode="r", allow_pickle=False),
        "y_graph": np.load(files["y_graph"], mmap_mode="r", allow_pickle=False),
        "y_node": np.load(files["y_node"], mmap_mode="r", allow_pickle=False),
        "edge_index": np.load(files["edge_index"], allow_pickle=False),
    }

    checks: dict[str, Any] = {}
    for name, array in arrays.items():
        observed_shape = tuple(int(value) for value in array.shape)
        expected_shape = EXPECTED_DATASET_SHAPES[name]
        passed = observed_shape == expected_shape
        checks[name] = {
            "path": str(files[name]),
            "shape": observed_shape,
            "expected_shape": expected_shape,
            "dtype": str(array.dtype),
            "passed": passed,
        }
        if not passed:
            raise Stage9A1Error(
                f"Dataset shape mismatch for {name}: "
                f"observed={observed_shape}, expected={expected_shape}"
            )

    return {"arrays": arrays, "checks": checks}


def read_splits(path: Path) -> dict[str, np.ndarray]:
    ensure_file(path, "split archive")
    with np.load(path, allow_pickle=False) as archive:
        required = ("train_idx", "val_idx", "test_idx")
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise Stage9A1Error(f"Missing split arrays in {path}: {missing}")
        return {
            key: np.asarray(archive[key], dtype=np.int64)
            for key in required
        }


def verify_split_identity(
    generated: Sequence[np.ndarray],
    saved: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    names = ("train_idx", "val_idx", "test_idx")
    if len(generated) != 3:
        raise Stage9A1Error(
            f"make_v3_splits returned {len(generated)} arrays; expected 3."
        )

    results: dict[str, Any] = {}
    for name, new_array in zip(names, generated):
        generated_array = np.asarray(new_array, dtype=np.int64)
        saved_array = np.asarray(saved[name], dtype=np.int64)
        exact_equal = bool(np.array_equal(generated_array, saved_array))
        expected_size = EXPECTED_SPLIT_SIZES[name]
        size_ok = (
            generated_array.size == expected_size
            and saved_array.size == expected_size
        )
        results[name] = {
            "generated_size": int(generated_array.size),
            "saved_size": int(saved_array.size),
            "expected_size": expected_size,
            "size_ok": size_ok,
            "exact_equal": exact_equal,
        }
        if not size_ok or not exact_equal:
            raise Stage9A1Error(
                f"Split mismatch for {name}: "
                f"generated={generated_array.size}, "
                f"saved={saved_array.size}, "
                f"exact_equal={exact_equal}"
            )

    results["all_exact_equal"] = True
    return results


def verify_hashes(paths: Paths) -> dict[str, Any]:
    artifacts = {
        "training_script": paths.training_script,
        "original_checkpoint": paths.original_dir / "best_model.pt",
        "original_summary": paths.original_dir / "summary.json",
        "original_history": paths.original_dir / "history.json",
        "original_splits": paths.original_dir / "splits.npz",
    }

    results: dict[str, Any] = {}
    for name, path in artifacts.items():
        ensure_file(path, name)
        actual = sha256_file(path)
        expected = EXPECTED_HASHES[name]
        passed = actual == expected
        results[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "passed": passed,
        }
        if not passed:
            raise Stage9A1Error(
                f"Frozen artifact hash mismatch for {name}: {path}\n"
                f"expected={expected}\nactual={actual}"
            )
    return results


def verify_environment(minimum_free_vram_gib: float) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise Stage9A1Error(f"Could not import PyTorch: {exc}") from exc

    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        raise Stage9A1Error("CUDA is not available.")

    try:
        gpu_name = str(torch.cuda.get_device_name(0))
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    except Exception as exc:
        raise Stage9A1Error(f"Could not query CUDA device 0: {exc}") from exc

    free_gib = free_bytes / (1024**3)
    total_gib = total_bytes / (1024**3)
    if free_gib < minimum_free_vram_gib:
        raise Stage9A1Error(
            f"Insufficient free VRAM: {free_gib:.3f} GiB available; "
            f"{minimum_free_vram_gib:.3f} GiB required."
        )

    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": str(torch.__version__),
        "cuda_available": cuda_available,
        "gpu": gpu_name,
        "free_vram_gib": free_gib,
        "total_vram_gib": total_gib,
        "minimum_required_vram_gib": minimum_free_vram_gib,
        "passed": True,
    }


def verify_model_and_weights(
    module: ModuleType,
    arrays: Mapping[str, np.ndarray],
    original_splits: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if not hasattr(module, "TemporalGCN"):
        raise Stage9A1Error("Training script does not define TemporalGCN.")

    model = module.TemporalGCN(
        input_features=24,
        temporal_dim=8,
        gcn_hidden=16,
        gcn_out=8,
    )
    parameter_count = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    parameter_count_ok = parameter_count == EXPECTED_PARAMETER_COUNT
    if not parameter_count_ok:
        raise Stage9A1Error(
            f"Parameter count mismatch: observed={parameter_count}, "
            f"expected={EXPECTED_PARAMETER_COUNT}"
        )

    train_idx = original_splits["train_idx"]
    y_graph_train = np.asarray(arrays["y_graph"][train_idx], dtype=np.float32)
    y_node_train = np.asarray(arrays["y_node"][train_idx], dtype=np.float32)

    graph_pos = float(y_graph_train.sum())
    graph_neg = float(y_graph_train.size - graph_pos)
    node_pos = float(y_node_train.sum())
    node_neg = float(y_node_train.size - node_pos)

    graph_weight = graph_neg / max(graph_pos, 1.0)
    node_weight = node_neg / max(node_pos, 1.0)

    graph_weight_ok = bool(
        np.isclose(
            graph_weight,
            EXPECTED_GRAPH_POS_WEIGHT,
            rtol=0.0,
            atol=1e-12,
        )
    )
    node_weight_ok = bool(
        np.isclose(
            node_weight,
            EXPECTED_NODE_POS_WEIGHT,
            rtol=0.0,
            atol=1e-12,
        )
    )

    if not graph_weight_ok or not node_weight_ok:
        raise Stage9A1Error(
            "Class-weight mismatch: "
            f"graph={graph_weight} expected={EXPECTED_GRAPH_POS_WEIGHT}; "
            f"node={node_weight} expected={EXPECTED_NODE_POS_WEIGHT}"
        )

    return {
        "parameter_count": parameter_count,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "parameter_count_ok": parameter_count_ok,
        "graph_pos_weight": graph_weight,
        "expected_graph_pos_weight": EXPECTED_GRAPH_POS_WEIGHT,
        "graph_pos_weight_ok": graph_weight_ok,
        "node_pos_weight": node_weight,
        "expected_node_pos_weight": EXPECTED_NODE_POS_WEIGHT,
        "node_pos_weight_ok": node_weight_ok,
        "passed": True,
    }


def build_training_command(paths: Paths) -> list[str]:
    return [
        sys.executable,
        str(paths.training_script),
        "--data",
        str(paths.dataset_root),
        "--out-dir",
        str(paths.output_dir),
        "--split-mode",
        "v3",
        "--epochs",
        "100",
        "--patience",
        "15",
        "--min-delta",
        "1e-4",
        "--batch-size",
        "256",
        "--lr",
        "1e-3",
        "--weight-decay",
        "1e-4",
        "--temporal-dim",
        "8",
        "--gcn-hidden",
        "16",
        "--gcn-out",
        "8",
        "--node-loss-weight",
        "1.0",
        "--graph-threshold",
        "0.5",
        "--node-threshold",
        "0.5",
        "--seed",
        "7",
    ]


def stream_process(command: Sequence[str], cwd: Path, log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with log_file.open("x", encoding="utf-8", buffering=1) as log:
        header = [
            "===== STAGE 9 A1 TRAINING =====",
            f"started_at={utc_now()}",
            f"cwd={cwd}",
            f"command={shlex.join(command)}",
            "",
        ]
        for line in header:
            print(line)
            log.write(line + "\n")

        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        previous_handlers: dict[int, Any] = {}

        def forward_signal(signum: int, _frame: Any) -> None:
            if process.poll() is None:
                process.send_signal(signum)

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, forward_signal)

        try:
            if process.stdout is None:
                raise Stage9A1Error("Training subprocess stdout pipe is unavailable.")

            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)

            return_code = int(process.wait())
            footer = (
                f"\ntraining_exit_code={return_code}\n"
                f"finished_at={utc_now()}\n"
            )
            print(footer, end="")
            log.write(footer)
            return return_code
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def verify_expected_outputs(output_dir: Path) -> dict[str, str]:
    files = {
        "best_model": output_dir / "best_model.pt",
        "history": output_dir / "history.json",
        "summary": output_dir / "summary.json",
        "splits": output_dir / "splits.npz",
    }
    for name, path in files.items():
        ensure_file(path, f"A1 output {name}")
    return {name: str(path) for name, path in files.items()}


def compare_summaries(
    original: Mapping[str, Any],
    reproduction: Mapping[str, Any],
) -> dict[str, Any]:
    metric_results: list[dict[str, Any]] = []

    for (split, metric), tolerance in METRIC_TOLERANCES.items():
        old_value = metric_value(original, split, metric)
        new_value = metric_value(reproduction, split, metric)
        delta = new_value - old_value
        passed = abs(delta) <= tolerance
        metric_results.append(
            {
                "split": split,
                "metric": metric,
                "original": old_value,
                "a1": new_value,
                "delta": delta,
                "absolute_delta": abs(delta),
                "tolerance": tolerance,
                "passed": passed,
            }
        )

    old_class_weights = original.get("class_weights", {})
    new_class_weights = reproduction.get("class_weights", {})
    old_thresholds = original.get("thresholds", {})
    new_thresholds = reproduction.get("thresholds", {})

    model_name_old = first_present(original, ("model", "model_name"))
    model_name_new = first_present(reproduction, ("model", "model_name"))
    parameters_new = int(first_present(reproduction, ("parameters", "parameter_count"), -1))

    hard_checks = {
        "model_name_matches": model_name_new == model_name_old,
        "parameter_count_882": parameters_new == EXPECTED_PARAMETER_COUNT,
        "split_sizes_match": reproduction.get("split_sizes") == original.get("split_sizes"),
        "graph_pos_weight_matches": bool(
            np.isclose(
                float(new_class_weights.get("graph_pos_weight", math.nan)),
                float(old_class_weights.get("graph_pos_weight", math.nan)),
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "node_pos_weight_matches": bool(
            np.isclose(
                float(new_class_weights.get("node_pos_weight", math.nan)),
                float(old_class_weights.get("node_pos_weight", math.nan)),
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "graph_threshold_0_5": bool(
            np.isclose(
                float(new_thresholds.get("graph", math.nan)),
                0.5,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "node_threshold_0_5": bool(
            np.isclose(
                float(new_thresholds.get("node", math.nan)),
                0.5,
                rtol=0.0,
                atol=1e-12,
            )
        ),
    }

    hard_pass = all(hard_checks.values())
    metrics_pass = all(item["passed"] for item in metric_results)

    old_best_epoch = int(original.get("best_epoch", -1))
    new_best_epoch = int(reproduction.get("best_epoch", -1))

    if hard_pass and metrics_pass:
        verdict = "PASS — SUFFICIENTLY REPRODUCED AT SUMMARY LEVEL"
    elif hard_pass:
        verdict = (
            "PASS WITH POSSIBLE NONDETERMINISM — "
            "RUN-LEVEL DIAGNOSTICS REQUIRED"
        )
    else:
        verdict = "FAIL — PIPELINE OR CONFIGURATION MISMATCH"

    return {
        "hard_checks": hard_checks,
        "hard_checks_passed": hard_pass,
        "metric_results": metric_results,
        "all_metric_tolerances_passed": metrics_pass,
        "best_epoch_original": old_best_epoch,
        "best_epoch_a1": new_best_epoch,
        "best_epoch_delta": new_best_epoch - old_best_epoch,
        "provisional_verdict": verdict,
    }


def write_output_hashes(paths: Paths) -> dict[str, Any]:
    outputs = {
        "best_model.pt": paths.output_dir / "best_model.pt",
        "history.json": paths.output_dir / "history.json",
        "summary.json": paths.output_dir / "summary.json",
        "splits.npz": paths.output_dir / "splits.npz",
    }

    results: dict[str, Any] = {}
    lines: list[str] = []
    for name, path in outputs.items():
        digest = sha256_file(path)
        results[name] = {
            "path": str(path),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
        lines.append(f"{digest}  {path}")

    atomic_write_text(paths.output_hashes_file, "\n".join(lines) + "\n")
    return results


def summary_training_details(summary: Mapping[str, Any], history_length: int) -> dict[str, Any]:
    training = summary.get("training")
    if not isinstance(training, Mapping):
        training = {}

    return {
        "best_epoch": summary.get("best_epoch"),
        "best_val_score": summary.get("best_val_score"),
        "epochs_completed": first_present(
            training,
            ("epochs_completed", "completed_epochs"),
            history_length,
        ),
        "epochs_requested": first_present(
            training,
            ("epochs_requested", "max_epochs", "epochs"),
            None,
        ),
        "stopped_early": first_present(
            training,
            ("stopped_early", "early_stopping"),
            None,
        ),
        "stopped_epoch": first_present(
            training,
            ("stopped_epoch", "early_stop_epoch"),
            None,
        ),
    }


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def build_markdown_report(
    result: Mapping[str, Any],
    original_summary: Mapping[str, Any],
    a1_summary: Mapping[str, Any],
) -> str:
    comparison = result["comparison"]
    training_details = result["a1_training_details"]

    lines = [
        "# Stage 9 A1 Conv1D Baseline Reproduction",
        "",
        f"- Generated: `{result['generated_at']}`",
        f"- Script version: `{result['runner_version']}`",
        f"- Training exit code: `{result['training_exit_code']}`",
        f"- Provisional verdict: **{comparison['provisional_verdict']}**",
        "",
        "## Purpose",
        "",
        (
            "A1 reran the original Conv1D-TemporalGCN baseline while keeping "
            "the dataset, V3 split, architecture, optimizer, class weights, "
            "losses, checkpoint score, thresholds, and seed unchanged."
        ),
        "",
        "## Preflight",
        "",
        f"- Frozen artifacts matched: `{all(item['passed'] for item in result['hash_checks'].values())}`",
        f"- Dataset shapes matched: `{all(item['passed'] for item in result['dataset_checks'].values())}`",
        f"- Original V3 split matched generated split exactly: `{result['pretraining_split_checks']['all_exact_equal']}`",
        f"- Parameter count: `{result['model_and_class_weights']['parameter_count']}`",
        f"- Graph positive weight: `{result['model_and_class_weights']['graph_pos_weight']}`",
        f"- Node positive weight: `{result['model_and_class_weights']['node_pos_weight']}`",
        f"- CUDA device: `{result['environment']['gpu']}`",
        "",
        "## Training result",
        "",
        f"- Best epoch: `{training_details['best_epoch']}`",
        f"- Best validation score: `{training_details['best_val_score']}`",
        f"- Epochs completed: `{training_details['epochs_completed']}`",
        f"- Early stopping: `{training_details['stopped_early']}`",
        f"- Stopped epoch: `{training_details['stopped_epoch']}`",
        "",
        "## Metric comparison",
        "",
        "| Split | Metric | Original | A1 | Delta | Tolerance | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for item in comparison["metric_results"]:
        lines.append(
            "| {split} | {metric} | {original:.9f} | {a1:.9f} | "
            "{delta:+.9f} | ±{tolerance:.3f} | {passed} |".format(**item)
        )

    lines.extend(
        [
            "",
            "## Hard configuration checks",
            "",
            "| Check | Pass |",
            "|---|---:|",
        ]
    )
    for name, passed in comparison["hard_checks"].items():
        lines.append(f"| {markdown_escape(name)} | {passed} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )

    if comparison["provisional_verdict"].startswith("PASS —"):
        lines.append(
            "The baseline is sufficiently reproduced at summary level. "
            "The next gate is aligned probability export and run-level "
            "diagnostics before implementing scenario-balanced sampling."
        )
    elif comparison["provisional_verdict"].startswith("PASS WITH"):
        lines.append(
            "The pipeline configuration matches, but at least one summary "
            "metric exceeded its tolerance. Run-level diagnostics are required "
            "to determine whether this is ordinary nondeterminism or a deeper "
            "reproducibility issue."
        )
    else:
        lines.append(
            "A hard pipeline or configuration check failed. Do not implement "
            "the B1 scenario-balanced sampler until the mismatch is resolved."
        )

    lines.extend(
        [
            "",
            "## Next step",
            "",
            (
                "Freeze the A1 outputs, export aligned graph and node "
                "probabilities, evaluate the reference and validation-selected "
                "graph thresholds, and repeat the run-level/Stage 7E diagnostics."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def print_preflight(
    hash_checks: Mapping[str, Any],
    dataset_checks: Mapping[str, Any],
    split_checks: Mapping[str, Any],
    model_checks: Mapping[str, Any],
    environment: Mapping[str, Any],
    command: Sequence[str],
) -> None:
    print("===== FROZEN HASH CHECKS =====")
    for name, item in hash_checks.items():
        print(f"{name}: passed={item['passed']} sha256={item['actual_sha256']}")

    print("\n===== DATASET CHECKS =====")
    for name, item in dataset_checks.items():
        print(
            f"{name}: shape={tuple(item['shape'])} "
            f"dtype={item['dtype']} passed={item['passed']}"
        )

    print("\n===== SPLIT CHECKS =====")
    for name in ("train_idx", "val_idx", "test_idx"):
        item = split_checks[name]
        print(
            f"{name}: generated={item['generated_size']} "
            f"saved={item['saved_size']} "
            f"exact_equal={item['exact_equal']}"
        )
    print("all_exact_equal:", split_checks["all_exact_equal"])

    print("\n===== MODEL AND CLASS WEIGHTS =====")
    print("parameter_count:", model_checks["parameter_count"])
    print("graph_pos_weight:", model_checks["graph_pos_weight"])
    print("node_pos_weight:", model_checks["node_pos_weight"])

    print("\n===== ENVIRONMENT =====")
    for key, value in environment.items():
        print(f"{key}: {value}")

    print("\n===== EXACT A1 COMMAND =====")
    print(shlex.join(command))


def parse_args() -> argparse.Namespace:
    home = Path.home()
    default_repo = home / "research/projects/GNN-2d"
    default_dataset = (
        home
        / "tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3"
    )
    default_output = (
        default_repo
        / "models/v3/stage9_a1_conv1d_exact_reproduction_seed7"
    )
    default_log = (
        default_repo
        / "reports/v3_failure_analysis/stage9_round1/logs"
        / "49_stage9_a1_conv1d_exact_training.log"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Safely run and validate the exact Stage 9 A1 Conv1D baseline "
            "reproduction."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=str(default_repo),
        help=f"Repository root (default: {default_repo})",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(default_dataset),
        help=f"V3 dataset directory (default: {default_dataset})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output),
        help=f"New A1 model output directory (default: {default_output})",
    )
    parser.add_argument(
        "--log-file",
        default=str(default_log),
        help=f"Training log file (default: {default_log})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run all read-only preflight checks and print the exact training "
            "command without launching training or creating the A1 output."
        ),
    )
    return parser.parse_args()


def build_paths(args: argparse.Namespace) -> Paths:
    repo_root = resolve_path(args.repo_root)
    dataset_root = resolve_path(args.dataset_root)
    output_dir = resolve_path(args.output_dir)
    log_file = resolve_path(args.log_file)

    stage9_root = (
        repo_root / "reports/v3_failure_analysis/stage9_round1"
    )
    original_dir = (
        repo_root / "models/v3/temporal_gcn_ports_v3_builtin_split"
    )
    training_script = repo_root / "scripts/train_temporal_gcn_v3.py"

    return Paths(
        repo_root=repo_root,
        dataset_root=dataset_root,
        training_script=training_script,
        original_dir=original_dir,
        output_dir=output_dir,
        stage9_root=stage9_root,
        log_file=log_file,
        status_file=(
            stage9_root
            / "logs/49_stage9_a1_conv1d_exact_training_status.txt"
        ),
        tables_dir=stage9_root / "tables",
        freeze_dir=stage9_root / "baseline_freeze",
        summary_output=(
            stage9_root / "tables/stage9_a1_reproduction_summary.json"
        ),
        report_output=(
            stage9_root / "STAGE9_A1_REPRODUCTION_REPORT.md"
        ),
        output_hashes_file=(
            stage9_root
            / "baseline_freeze/stage9_a1_outputs_sha256.txt"
        ),
    )


def validate_paths(paths: Paths) -> None:
    ensure_dir(paths.repo_root, "repository root")
    ensure_dir(paths.dataset_root, "dataset root")
    ensure_file(paths.training_script, "training script")
    ensure_dir(paths.original_dir, "original baseline directory")

    if paths.output_dir.exists():
        raise Stage9A1Error(
            f"A1 output directory already exists: {paths.output_dir}"
        )

    if paths.output_dir.resolve() == paths.original_dir.resolve():
        raise Stage9A1Error(
            "A1 output directory must not equal the original baseline directory."
        )

    if path_is_within(paths.output_dir, paths.original_dir):
        raise Stage9A1Error(
            "A1 output directory must not be inside the original baseline directory."
        )

    if paths.log_file.exists():
        raise Stage9A1Error(
            f"Training log already exists: {paths.log_file}"
        )

    if paths.summary_output.exists():
        raise Stage9A1Error(
            f"Reproduction summary already exists: {paths.summary_output}"
        )

    if paths.report_output.exists():
        raise Stage9A1Error(
            f"Reproduction report already exists: {paths.report_output}"
        )

    if paths.output_hashes_file.exists():
        raise Stage9A1Error(
            f"A1 output hash file already exists: {paths.output_hashes_file}"
        )


def run() -> int:
    args = parse_args()
    paths = build_paths(args)

    validate_paths(paths)

    py_compile.compile(
        str(paths.training_script),
        doraise=True,
    )

    hash_checks = verify_hashes(paths)
    dataset_result = inspect_dataset(paths.dataset_root)
    arrays = dataset_result["arrays"]
    dataset_checks = dataset_result["checks"]

    training_module = import_training_module(paths.training_script)
    if not hasattr(training_module, "make_v3_splits"):
        raise Stage9A1Error(
            "Training script does not define make_v3_splits()."
        )

    original_splits = read_splits(paths.original_dir / "splits.npz")
    generated_splits = training_module.make_v3_splits(
        str(paths.dataset_root)
    )
    split_checks = verify_split_identity(
        generated_splits,
        original_splits,
    )

    model_checks = verify_model_and_weights(
        training_module,
        arrays,
        original_splits,
    )
    environment = verify_environment(MINIMUM_FREE_VRAM_GIB)
    command = build_training_command(paths)

    print_preflight(
        hash_checks,
        dataset_checks,
        split_checks,
        model_checks,
        environment,
        command,
    )

    if args.dry_run:
        print(
            "\nDRY RUN PASSED: training was not launched and the "
            "A1 output directory was not created."
        )
        return 0

    paths.log_file.parent.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths.freeze_dir.mkdir(parents=True, exist_ok=True)
    paths.stage9_root.mkdir(parents=True, exist_ok=True)

    training_exit_code = stream_process(
        command,
        cwd=paths.repo_root,
        log_file=paths.log_file,
    )

    status_payload = {
        "stage9_a1_exit_code": training_exit_code,
        "training_log": str(paths.log_file),
        "output_directory": str(paths.output_dir),
        "completed_at": utc_now(),
    }
    atomic_write_text(
        paths.status_file,
        "\n".join(f"{key}={value}" for key, value in status_payload.items())
        + "\n",
    )

    if training_exit_code != 0:
        raise Stage9A1Error(
            f"Stage 9 A1 training failed with exit code {training_exit_code}. "
            f"See {paths.log_file}"
        )

    expected_outputs = verify_expected_outputs(paths.output_dir)

    post_hash_checks = verify_hashes(paths)
    for name, item in post_hash_checks.items():
        if not item["passed"]:
            raise Stage9A1Error(
                f"Original artifact changed during A1: {name}"
            )

    new_splits = read_splits(paths.output_dir / "splits.npz")
    post_split_checks = verify_split_identity(
        (
            new_splits["train_idx"],
            new_splits["val_idx"],
            new_splits["test_idx"],
        ),
        original_splits,
    )

    original_summary = load_json(paths.original_dir / "summary.json")
    a1_summary = load_json(paths.output_dir / "summary.json")

    history_path = paths.output_dir / "history.json"
    with history_path.open("r", encoding="utf-8") as handle:
        history_data = json.load(handle)
    history_length = len(history_data) if isinstance(history_data, list) else 0

    comparison = compare_summaries(original_summary, a1_summary)
    output_hashes = write_output_hashes(paths)

    result: dict[str, Any] = {
        "runner_version": SCRIPT_VERSION,
        "generated_at": utc_now(),
        "training_exit_code": training_exit_code,
        "paths": {
            "repo_root": str(paths.repo_root),
            "dataset_root": str(paths.dataset_root),
            "training_script": str(paths.training_script),
            "original_baseline_directory": str(paths.original_dir),
            "a1_output_directory": str(paths.output_dir),
            "training_log": str(paths.log_file),
            "status_file": str(paths.status_file),
            "summary_json": str(paths.summary_output),
            "report_markdown": str(paths.report_output),
            "output_hashes": str(paths.output_hashes_file),
        },
        "training_command": command,
        "hash_checks": hash_checks,
        "posttraining_original_hash_checks": post_hash_checks,
        "preflight_checks": {
            "training_script_compiled": True,
            "cuda_available": environment["cuda_available"],
            "minimum_vram_met": (
                environment["free_vram_gib"]
                >= environment["minimum_required_vram_gib"]
            ),
            "dataset_shapes_match": all(
                item["passed"] for item in dataset_checks.values()
            ),
            "split_identity_match": split_checks["all_exact_equal"],
            "parameter_count_match": model_checks["parameter_count_ok"],
            "class_weights_match": (
                model_checks["graph_pos_weight_ok"]
                and model_checks["node_pos_weight_ok"]
            ),
        },
        "environment": environment,
        "dataset_checks": dataset_checks,
        "pretraining_split_checks": split_checks,
        "posttraining_split_checks": post_split_checks,
        "model_and_class_weights": model_checks,
        "expected_outputs": expected_outputs,
        "output_hashes": output_hashes,
        "original_summary": original_summary,
        "a1_summary": a1_summary,
        "a1_history_length": history_length,
        "a1_training_details": summary_training_details(
            a1_summary,
            history_length,
        ),
        "comparison": comparison,
        "provisional_verdict": comparison["provisional_verdict"],
    }

    atomic_write_json(
        paths.summary_output,
        as_jsonable(result),
    )
    atomic_write_text(
        paths.report_output,
        build_markdown_report(
            result,
            original_summary,
            a1_summary,
        ),
    )

    print("\n===== A1 IMMEDIATE SUMMARY =====")
    details = result["a1_training_details"]
    for key, value in details.items():
        print(f"{key}: {value}")

    print("\n===== ORIGINAL VERSUS A1 =====")
    for item in comparison["metric_results"]:
        print(
            f"{item['split']}.{item['metric']}: "
            f"original={item['original']:.9f} "
            f"a1={item['a1']:.9f} "
            f"delta={item['delta']:+.9f} "
            f"tolerance=±{item['tolerance']:.3f} "
            f"pass={item['passed']}"
        )

    print("\n===== HARD CONFIGURATION CHECKS =====")
    for name, passed in comparison["hard_checks"].items():
        print(f"{name}: {passed}")

    print("\nA1_PROVISIONAL_VERDICT:", comparison["provisional_verdict"])
    print("machine_readable_summary:", paths.summary_output)
    print("human_readable_report:", paths.report_output)
    print("output_hashes:", paths.output_hashes_file)

    if comparison["provisional_verdict"].startswith("FAIL"):
        return 2
    return 0


def main() -> None:
    try:
        exit_code = run()
    except KeyboardInterrupt:
        print("\nSTOP: interrupted by user.", file=sys.stderr)
        exit_code = 130
    except Stage9A1Error as exc:
        print(f"\nSTOP: {exc}", file=sys.stderr)
        exit_code = 1
    except Exception as exc:
        print(
            f"\nSTOP: unexpected {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        exit_code = 1
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
