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
import shutil
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


STAGE = "V5_P3_F6_TASK_SPECIFIC_INTEGRATED_GRADIENTS"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "VALIDATION-EXPLORATORY"

EXPECTED_ITEMS = 13863
EXPECTED_ACTIVE = 3822
EXPECTED_PARAMETER_COUNT = 60553
EXPECTED_X_SHAPE = (16, 70, 32)
EXPECTED_MASK_SHAPE = (16, 10)

SAMPLE_SELECTION_SEED = 6101
BOOTSTRAP_SEED = 6102
ITEMS_PER_COUNT = 128
EXPECTED_SELECTED_ITEMS = 512
DEFAULT_IG_POINTS = 64
FALLBACK_IG_POINTS = 128
SAMPLE_MICROBATCH_SIZE = 16
BOOTSTRAP_RESAMPLES = 1000

TASKS = (
    "graph",
    "count",
    "source",
    "transit",
    "victim",
    "path",
)

ROLE_LABEL_KEYS = {
    "source": "y_source",
    "transit": "y_transit",
    "victim": "y_victim",
    "path": "y_attack_path",
}

GROUPS = {
    "directional_traffic_volume": (0, 10),
    "inter_flit_timing": (10, 30),
    "queue_activity": (30, 40),
    "buffer_pressure": (40, 55),
    "flow_control_stalls": (55, 70),
}

HEAD_WIDTHS = {
    "graph": 1,
    "count": 4,
    "source": 16,
    "transit": 16,
    "victim": 16,
    "path": 16,
}

CONSTANT_ZERO_CHANNELS_FROM_F2R = {60, 65, 66, 67, 68, 69}

PORTS = ("local", "north", "east", "south", "west")
CHANNEL_NAMES = (
    [f"in_count_{port}" for port in PORTS]
    + [f"out_count_{port}" for port in PORTS]
    + [f"in_gap_sum_{port}" for port in PORTS]
    + [f"in_gap_count_{port}" for port in PORTS]
    + [f"out_gap_sum_{port}" for port in PORTS]
    + [f"out_gap_count_{port}" for port in PORTS]
    + [f"enqueue_count_{port}" for port in PORTS]
    + [f"dequeue_count_{port}" for port in PORTS]
    + [f"occupancy_cycle_sum_{port}" for port in PORTS]
    + [f"occupancy_max_{port}" for port in PORTS]
    + [f"occupancy_end_{port}" for port in PORTS]
    + [f"stall_no_free_vc_{port}" for port in PORTS]
    + [f"stall_no_credit_{port}" for port in PORTS]
    + [f"stall_ordering_{port}" for port in PORTS]
)
assert len(CHANNEL_NAMES) == 70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--installed-script", required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--tasks",
        default=",".join(TASKS),
        help="Comma-separated task subset. Re-running resumes completed work.",
    )
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


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(rows, f"cannot write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
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


def normalize_name(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            return value

    tensors = {
        key: value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }
    require(tensors, "checkpoint state_dict not found")
    return tensors


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
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        if not torch.equal(left.detach().cpu(), right.detach().cpu()):
            return False
    return True


class _OfficialModelCaptured(BaseException):
    pass


def capture_model_via_official_exporter(
    *,
    exporter_path: Path,
    repo: Path,
    data_link: Path,
    run_dir: Path,
    checkpoint_state_dict: dict[str, torch.Tensor],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    exporter = import_source(
        exporter_path,
        "_v5_p3_f6_official_d1_exporter",
    )
    require(
        hasattr(exporter, "main") and callable(exporter.main),
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
            if not state_dict_exact_match(value, checkpoint_state_dict):
                continue

            captured["model"] = value
            captured["class_name"] = value.__class__.__name__
            captured["local_name"] = local_name
            captured["function"] = frame.f_code.co_name
            captured["line"] = int(frame.f_lineno)
            raise _OfficialModelCaptured()

        return tracer

    old_argv = list(sys.argv)
    old_trace = sys.gettrace()
    observation = None

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
            exporter.main()
            observation = "official_exporter_returned_without_capture"
        except _OfficialModelCaptured:
            observation = "captured_exact_checkpoint_model"
        except SystemExit as exc:
            observation = f"system_exit_before_capture:{exc.code}"
    finally:
        sys.settrace(old_trace)
        sys.argv = old_argv

    require(
        "model" in captured,
        "official D1 exporter did not expose the exact checkpoint model; "
        f"observation={observation}",
    )

    model = captured["model"]
    model.to("cpu")
    model.eval()
    require(
        state_dict_exact_match(model, checkpoint_state_dict),
        "captured model changed after official-exporter capture",
    )

    provenance = {
        "route": "official_D1_exporter_trace_capture",
        "exporter": str(exporter_path),
        "exporter_sha256": sha256_file(exporter_path),
        "model_class": captured["class_name"],
        "captured_local_name": captured["local_name"],
        "captured_function": captured["function"],
        "captured_line": captured["line"],
        "observation": observation,
        "state_dict_exact_match": True,
        "parameter_count": parameter_count(model),
        "validation_inference_completed_by_probe": False,
        "sealed_test_access": False,
    }
    atomic_json(
        run_dir / "F6_OFFICIAL_EXPORTER_MODEL_CAPTURE.json",
        provenance,
    )
    return model, provenance


def configure_official_runtime(
    device_argument: str,
) -> tuple[torch.device, dict[str, Any]]:
    device = torch.device(
        device_argument
        if device_argument
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    random.seed(107)
    np.random.seed(107)
    torch.manual_seed(107)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(107)

    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")

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
                key = normalize_name(candidate)
                if key in normalized:
                    result[head] = normalized[key]
                    break
    elif isinstance(output, (tuple, list)) and len(output) == 6:
        result = {
            head: value
            for head, value in zip(TASKS, output)
        }
    else:
        for head, candidates in aliases.items():
            for candidate in candidates:
                if hasattr(output, candidate):
                    result[head] = getattr(output, candidate)
                    break

    require(
        set(result) == set(HEAD_WIDTHS),
        f"could not resolve six output heads; found={sorted(result)}",
    )
    return result


def canonical_head_tensor(
    tensor: torch.Tensor,
    head: str,
    batch_size: int,
) -> torch.Tensor:
    width = HEAD_WIDTHS[head]
    if head == "graph":
        value = tensor.reshape(batch_size, -1)
        require(value.shape[1] == 1, f"graph output shape mismatch: {value.shape}")
        return value[:, 0]
    value = tensor.reshape(batch_size, -1)
    require(
        value.shape[1] == width,
        f"{head} output shape mismatch: {value.shape}",
    )
    return value


def task_target(
    mapped: dict[str, torch.Tensor],
    task: str,
    labels: dict[str, torch.Tensor],
) -> torch.Tensor:
    batch_size = int(labels["count"].shape[0])

    if task == "graph":
        return canonical_head_tensor(
            mapped["graph"],
            "graph",
            batch_size,
        )

    if task == "count":
        logits = canonical_head_tensor(
            mapped["count"],
            "count",
            batch_size,
        )
        class_index = labels["count"].long() - 1
        require(
            bool(torch.all((class_index >= 0) & (class_index < 4))),
            "active attacker-count labels are not in 1..4",
        )
        true_logit = logits.gather(
            1,
            class_index[:, None],
        )[:, 0]
        other_mean = (logits.sum(dim=1) - true_logit) / 3.0
        return true_logit - other_mean

    require(task in ROLE_LABEL_KEYS, f"unknown task: {task}")
    logits = canonical_head_tensor(
        mapped[task],
        task,
        batch_size,
    )
    role = labels["role"].to(dtype=torch.bool)
    positive_count = role.sum(dim=1)
    negative_count = (~role).sum(dim=1)
    require(
        bool(torch.all(positive_count > 0)),
        f"{task} target contains an empty positive set",
    )
    require(
        bool(torch.all(negative_count > 0)),
        f"{task} target contains an empty negative set",
    )
    positive_mean = (
        (logits * role.to(logits.dtype)).sum(dim=1)
        / positive_count.to(logits.dtype)
    )
    negative_mean = (
        (logits * (~role).to(logits.dtype)).sum(dim=1)
        / negative_count.to(logits.dtype)
    )
    return positive_mean - negative_mean


def active_f5_run(permutation_root: Path) -> Path:
    pointer = permutation_root / "F5_ACTIVE_RUN_PATH.txt"
    require(pointer.is_file(), f"F5 active-run pointer missing: {pointer}")
    run_dir = Path(
        pointer.read_text(encoding="utf-8").strip()
    ).expanduser().resolve()
    require(run_dir.is_dir(), f"F5 run directory missing: {run_dir}")
    return run_dir


def active_f6_run(
    output_root: Path,
    explicit: str,
) -> Path:
    pointer = output_root / "F6_ACTIVE_RUN_PATH.txt"

    if explicit:
        run_dir = Path(explicit).expanduser().resolve()
        require(run_dir.is_dir(), f"explicit F6 run missing: {run_dir}")
        atomic_text(pointer, str(run_dir) + "\n")
        return run_dir

    if pointer.is_file():
        run_dir = Path(
            pointer.read_text(encoding="utf-8").strip()
        ).expanduser().resolve()
        require(run_dir.is_dir(), f"F6 active run missing: {run_dir}")
        return run_dir

    run_dir = (
        output_root
        / "f6_runs"
        / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    )
    run_dir.mkdir(parents=True)
    atomic_text(pointer, str(run_dir.resolve()) + "\n")
    return run_dir.resolve()


def open_cache(cache_dir: Path) -> dict[str, np.ndarray]:
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
        path = cache_dir / f"{key}.npy"
        require(path.is_file(), f"F5 validation cache array missing: {path}")
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        require(array.shape == shape, f"{key} shape changed: {array.shape}")
        require(array.dtype == dtype, f"{key} dtype changed: {array.dtype}")
        arrays[key] = array
    return arrays


def freeze_sample_selection(
    *,
    run_dir: Path,
    mapping_document: dict[str, Any],
    cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    selection_path = run_dir / "F6_SELECTED_ATTACK_CONTROL_ITEMS.json"
    selection_csv_path = run_dir / "F6_SELECTED_ATTACK_CONTROL_ITEMS.csv"

    mapping_rows = mapping_document["mapping"]
    require(len(mapping_rows) == EXPECTED_ACTIVE, "P3 mapping attack count changed")

    rows_by_attack = {
        int(row["attack_index"]): row
        for row in mapping_rows
    }
    require(
        len(rows_by_attack) == EXPECTED_ACTIVE,
        "P3 mapping contains duplicate attack indices",
    )

    active_indices = np.flatnonzero(
        np.asarray(cache["y_attack"]) == 1
    ).astype(np.int64)
    require(
        active_indices.size == EXPECTED_ACTIVE,
        "cached active validation count changed",
    )
    require(
        active_indices.tolist() == sorted(rows_by_attack),
        "P3 mapping does not cover exact cached active indices",
    )

    if selection_path.is_file():
        selection = load_json(selection_path)
        require(
            selection["selection_seed"] == SAMPLE_SELECTION_SEED,
            "selection seed changed",
        )
        require(
            selection["items_per_count"] == ITEMS_PER_COUNT,
            "items-per-count changed",
        )
        require(
            selection["selected_item_count"] == EXPECTED_SELECTED_ITEMS,
            "selected item count changed",
        )
        require(selection_csv_path.is_file(), "selection CSV missing")
        return selection

    rng = np.random.default_rng(SAMPLE_SELECTION_SEED)
    selected_rows = []

    for count in (1, 2, 3, 4):
        candidates = np.asarray(
            [
                attack_index
                for attack_index in active_indices.tolist()
                if int(cache["y_attacker_count"][attack_index]) == count
            ],
            dtype=np.int64,
        )
        require(
            candidates.size >= ITEMS_PER_COUNT,
            f"attacker-count {count} lacks 128 candidates",
        )
        chosen = np.sort(
            rng.choice(
                candidates,
                size=ITEMS_PER_COUNT,
                replace=False,
            )
        )
        for attack_index in chosen.tolist():
            mapping = rows_by_attack[attack_index]
            control_index = int(mapping["control_index"])
            require(
                int(cache["y_attack"][control_index]) == 0,
                "mapped control item is not benign",
            )
            require(
                int(cache["y_attacker_count"][attack_index]) == count,
                "mapped attack count changed",
            )
            selected_rows.append({
                "attack_index": attack_index,
                "control_index": control_index,
                "attacker_count": count,
                "pair_id": mapping.get("pair_id"),
                "window_start": mapping.get("window_start"),
            })

    selected_rows.sort(
        key=lambda row: (row["attacker_count"], row["attack_index"])
    )
    require(
        len(selected_rows) == EXPECTED_SELECTED_ITEMS,
        "selection size mismatch",
    )
    require(
        len({row["attack_index"] for row in selected_rows})
        == EXPECTED_SELECTED_ITEMS,
        "selected attack index duplicated",
    )
    require(
        len({row["control_index"] for row in selected_rows})
        == EXPECTED_SELECTED_ITEMS,
        "selected control index duplicated",
    )

    mask_mismatches = 0
    for row in selected_rows:
        if not np.array_equal(
            cache["physical_port_mask"][row["attack_index"]],
            cache["physical_port_mask"][row["control_index"]],
        ):
            mask_mismatches += 1

    selection = {
        "status": "FROZEN",
        "selection_seed": SAMPLE_SELECTION_SEED,
        "items_per_count": ITEMS_PER_COUNT,
        "selected_item_count": len(selected_rows),
        "attacker_count_strata": dict(
            Counter(row["attacker_count"] for row in selected_rows)
        ),
        "attack_control_mask_mismatch_count": mask_mismatches,
        "baseline_policy": (
            "matched control x with the attack item's physical-port mask"
        ),
        "rows": selected_rows,
    }
    atomic_json(selection_path, selection)
    write_csv(selection_csv_path, selected_rows)
    return selection


def task_rows(
    task: str,
    selection: dict[str, Any],
    cache: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = list(selection["rows"])
    if task not in ROLE_LABEL_KEYS:
        return rows

    label_key = ROLE_LABEL_KEYS[task]
    filtered = [
        row
        for row in rows
        if int(np.sum(cache[label_key][row["attack_index"]])) > 0
    ]
    require(filtered, f"{task} has no selected nonempty targets")
    return filtered


def labels_for_batch(
    task: str,
    attack_indices: np.ndarray,
    cache: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    labels = {
        "count": torch.from_numpy(
            np.array(
                cache["y_attacker_count"][attack_indices],
                dtype=np.int64,
                copy=True,
            )
        ).to(device=device),
    }
    if task in ROLE_LABEL_KEYS:
        labels["role"] = torch.from_numpy(
            np.array(
                cache[ROLE_LABEL_KEYS[task]][attack_indices],
                dtype=np.float32,
                copy=True,
            )
        ).to(device=device)
    return labels


def endpoint_targets(
    *,
    model: torch.nn.Module,
    task: str,
    x: torch.Tensor,
    mask: torch.Tensor,
    labels: dict[str, torch.Tensor],
) -> torch.Tensor:
    with torch.no_grad():
        mapped = model_output_mapping(model(x, mask))
        return task_target(mapped, task, labels).detach()


def compute_ig_batch(
    *,
    model: torch.nn.Module,
    task: str,
    attack_x: np.ndarray,
    control_x: np.ndarray,
    attack_mask: np.ndarray,
    labels: dict[str, torch.Tensor],
    device: torch.device,
    points: int,
) -> dict[str, np.ndarray]:
    require(points >= 2, "IG requires at least two integration points")

    x1 = torch.from_numpy(
        np.array(attack_x, dtype=np.float32, copy=True)
    ).to(device=device)
    x0 = torch.from_numpy(
        np.array(control_x, dtype=np.float32, copy=True)
    ).to(device=device)
    mask = torch.from_numpy(
        np.array(attack_mask, dtype=np.float32, copy=True)
    ).to(device=device)

    delta = x1 - x0
    target_attack = endpoint_targets(
        model=model,
        task=task,
        x=x1,
        mask=mask,
        labels=labels,
    )
    target_control = endpoint_targets(
        model=model,
        task=task,
        x=x0,
        mask=mask,
        labels=labels,
    )

    gradient_integral = torch.zeros_like(x1)
    alphas = torch.linspace(
        0.0,
        1.0,
        points,
        device=device,
        dtype=x1.dtype,
    )

    for step_index, alpha in enumerate(alphas):
        interpolated = (
            x0 + alpha * delta
        ).detach().requires_grad_(True)

        mapped = model_output_mapping(model(interpolated, mask))
        target = task_target(mapped, task, labels)

        gradient = torch.autograd.grad(
            target.sum(),
            interpolated,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

        weight = 0.5 if step_index in (0, points - 1) else 1.0
        gradient_integral.add_(gradient.detach(), alpha=weight)

    average_gradient = gradient_integral / float(points - 1)
    integrated_gradients = delta * average_gradient

    ig = integrated_gradients.detach().float().cpu().numpy()
    target_attack_np = target_attack.float().cpu().numpy()
    target_control_np = target_control.float().cpu().numpy()
    target_delta = target_attack_np - target_control_np
    ig_sum = ig.reshape(ig.shape[0], -1).sum(axis=1)
    absolute_error = np.abs(ig_sum - target_delta)
    relative_error = absolute_error / np.maximum(
        np.abs(target_delta),
        1e-6,
    )

    absolute_ig = np.abs(ig)
    channel_signed = ig.sum(axis=(1, 3))
    channel_absolute = absolute_ig.sum(axis=(1, 3))
    router_absolute = absolute_ig.sum(axis=(2, 3))
    time_absolute = absolute_ig.sum(axis=(1, 2))

    group_absolute = np.zeros((ig.shape[0], len(GROUPS)), dtype=np.float32)
    group_per_channel = np.zeros_like(group_absolute)
    for group_index, (group, (start, end)) in enumerate(GROUPS.items()):
        group_absolute[:, group_index] = absolute_ig[
            :, :, start:end, :
        ].sum(axis=(1, 2, 3))
        group_per_channel[:, group_index] = (
            group_absolute[:, group_index] / float(end - start)
        )

    total_absolute = absolute_ig.reshape(ig.shape[0], -1).sum(axis=1)
    safe_total = np.maximum(total_absolute, 1e-12)
    group_fraction = group_absolute / safe_total[:, None]
    channel_fraction = channel_absolute / safe_total[:, None]

    return {
        "integrated_gradients": ig.astype(np.float32, copy=False),
        "target_attack": target_attack_np.astype(np.float32, copy=False),
        "target_control": target_control_np.astype(np.float32, copy=False),
        "target_delta": target_delta.astype(np.float32, copy=False),
        "ig_sum": ig_sum.astype(np.float32, copy=False),
        "absolute_completeness_error": absolute_error.astype(
            np.float32,
            copy=False,
        ),
        "relative_completeness_error": relative_error.astype(
            np.float32,
            copy=False,
        ),
        "channel_signed": channel_signed.astype(np.float32, copy=False),
        "channel_absolute": channel_absolute.astype(np.float32, copy=False),
        "channel_fraction": channel_fraction.astype(np.float32, copy=False),
        "group_absolute": group_absolute,
        "group_per_channel": group_per_channel,
        "group_fraction": group_fraction.astype(np.float32, copy=False),
        "router_absolute": router_absolute.astype(np.float32, copy=False),
        "time_absolute": time_absolute.astype(np.float32, copy=False),
        "total_absolute": total_absolute.astype(np.float32, copy=False),
    }


def batch_paths(
    attempt_dir: Path,
    batch_number: int,
) -> dict[str, Path]:
    batch_dir = attempt_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    stem = f"batch_{batch_number:04d}"
    return {
        "npz": batch_dir / f"{stem}.npz",
        "manifest": batch_dir / f"{stem}.json",
        "complete": batch_dir / f"{stem}.complete",
    }


def load_completed_batch(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    require(paths["npz"].is_file(), f"batch NPZ missing: {paths['npz']}")
    require(
        paths["manifest"].is_file(),
        f"batch manifest missing: {paths['manifest']}",
    )
    manifest = load_json(paths["manifest"])
    require(
        manifest.get("status") == "PASS",
        f"batch not PASS: {paths['manifest']}",
    )
    require(
        manifest.get("npz_sha256") == sha256_file(paths["npz"]),
        f"batch hash mismatch: {paths['npz']}",
    )
    with np.load(paths["npz"], allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    require(array.shape[0] >= 2, "bootstrap requires at least two samples")

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(
        (resamples, array.shape[1]),
        dtype=np.float64,
    )
    for repeat in range(resamples):
        indices = rng.integers(
            0,
            array.shape[0],
            size=array.shape[0],
        )
        bootstrap_means[repeat] = array[indices].mean(axis=0)

    lower = np.quantile(bootstrap_means, 0.025, axis=0)
    upper = np.quantile(bootstrap_means, 0.975, axis=0)
    return lower, upper


def aggregate_attempt(
    *,
    task: str,
    points: int,
    attempt_dir: Path,
    batch_outputs: list[dict[str, np.ndarray]],
    task_rows_list: list[dict[str, Any]],
    bootstrap_seed: int,
) -> dict[str, Any]:
    keys = (
        "attack_indices",
        "control_indices",
        "target_attack",
        "target_control",
        "target_delta",
        "ig_sum",
        "absolute_completeness_error",
        "relative_completeness_error",
        "channel_signed",
        "channel_absolute",
        "channel_fraction",
        "group_absolute",
        "group_per_channel",
        "group_fraction",
        "router_absolute",
        "time_absolute",
        "total_absolute",
    )
    combined = {
        key: np.concatenate(
            [output[key] for output in batch_outputs],
            axis=0,
        )
        for key in keys
    }

    expected_attacks = np.asarray(
        [row["attack_index"] for row in task_rows_list],
        dtype=np.int64,
    )
    require(
        np.array_equal(combined["attack_indices"], expected_attacks),
        f"{task} aggregate attack-index order changed",
    )

    relative = combined["relative_completeness_error"]
    median_relative = float(np.median(relative))
    p95_relative = float(np.quantile(relative, 0.95))
    maximum_relative = float(np.max(relative))
    completeness_pass = bool(
        median_relative <= 0.02
        and p95_relative <= 0.05
    )

    channel_signed_mean = combined["channel_signed"].mean(axis=0)
    channel_absolute_mean = combined["channel_absolute"].mean(axis=0)
    channel_fraction_mean = combined["channel_fraction"].mean(axis=0)
    group_absolute_mean = combined["group_absolute"].mean(axis=0)
    group_per_channel_mean = combined["group_per_channel"].mean(axis=0)
    group_fraction_mean = combined["group_fraction"].mean(axis=0)
    router_absolute_mean = combined["router_absolute"].mean(axis=0)
    time_absolute_mean = combined["time_absolute"].mean(axis=0)

    channel_abs_lo, channel_abs_hi = bootstrap_mean_ci(
        combined["channel_absolute"],
        seed=bootstrap_seed + 1,
    )
    channel_fraction_lo, channel_fraction_hi = bootstrap_mean_ci(
        combined["channel_fraction"],
        seed=bootstrap_seed + 2,
    )
    group_abs_lo, group_abs_hi = bootstrap_mean_ci(
        combined["group_absolute"],
        seed=bootstrap_seed + 3,
    )
    group_per_channel_lo, group_per_channel_hi = bootstrap_mean_ci(
        combined["group_per_channel"],
        seed=bootstrap_seed + 4,
    )
    group_fraction_lo, group_fraction_hi = bootstrap_mean_ci(
        combined["group_fraction"],
        seed=bootstrap_seed + 5,
    )
    router_lo, router_hi = bootstrap_mean_ci(
        combined["router_absolute"],
        seed=bootstrap_seed + 6,
    )
    time_lo, time_hi = bootstrap_mean_ci(
        combined["time_absolute"],
        seed=bootstrap_seed + 7,
    )

    channel_rows = []
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        group_name = next(
            group
            for group, (start, end) in GROUPS.items()
            if start <= channel_index < end
        )
        channel_rows.append({
            "rank_by_mean_absolute": 0,
            "channel_index": channel_index,
            "channel_name": channel_name,
            "macro_group": group_name,
            "F2R_constant_zero": (
                channel_index in CONSTANT_ZERO_CHANNELS_FROM_F2R
            ),
            "mean_signed_attribution": float(
                channel_signed_mean[channel_index]
            ),
            "mean_absolute_attribution": float(
                channel_absolute_mean[channel_index]
            ),
            "mean_absolute_ci95_lower": float(
                channel_abs_lo[channel_index]
            ),
            "mean_absolute_ci95_upper": float(
                channel_abs_hi[channel_index]
            ),
            "mean_fraction_of_total_absolute": float(
                channel_fraction_mean[channel_index]
            ),
            "fraction_ci95_lower": float(
                channel_fraction_lo[channel_index]
            ),
            "fraction_ci95_upper": float(
                channel_fraction_hi[channel_index]
            ),
        })
    channel_rows.sort(
        key=lambda row: row["mean_absolute_attribution"],
        reverse=True,
    )
    for rank, row in enumerate(channel_rows, start=1):
        row["rank_by_mean_absolute"] = rank

    group_rows = []
    for group_index, (group, (start, end)) in enumerate(GROUPS.items()):
        group_rows.append({
            "rank_by_total_absolute": 0,
            "rank_by_per_channel_absolute": 0,
            "group": group,
            "start_channel": start,
            "end_channel_exclusive": end,
            "channel_count": end - start,
            "mean_total_absolute_attribution": float(
                group_absolute_mean[group_index]
            ),
            "total_absolute_ci95_lower": float(
                group_abs_lo[group_index]
            ),
            "total_absolute_ci95_upper": float(
                group_abs_hi[group_index]
            ),
            "mean_absolute_attribution_per_channel": float(
                group_per_channel_mean[group_index]
            ),
            "per_channel_ci95_lower": float(
                group_per_channel_lo[group_index]
            ),
            "per_channel_ci95_upper": float(
                group_per_channel_hi[group_index]
            ),
            "mean_fraction_of_total_absolute": float(
                group_fraction_mean[group_index]
            ),
            "fraction_ci95_lower": float(
                group_fraction_lo[group_index]
            ),
            "fraction_ci95_upper": float(
                group_fraction_hi[group_index]
            ),
        })

    total_order = sorted(
        range(len(group_rows)),
        key=lambda index: group_rows[index][
            "mean_total_absolute_attribution"
        ],
        reverse=True,
    )
    per_channel_order = sorted(
        range(len(group_rows)),
        key=lambda index: group_rows[index][
            "mean_absolute_attribution_per_channel"
        ],
        reverse=True,
    )
    for rank, index in enumerate(total_order, start=1):
        group_rows[index]["rank_by_total_absolute"] = rank
    for rank, index in enumerate(per_channel_order, start=1):
        group_rows[index]["rank_by_per_channel_absolute"] = rank
    group_rows.sort(key=lambda row: row["rank_by_total_absolute"])

    router_rows = [
        {
            "rank": 0,
            "router_index": index,
            "mean_absolute_attribution": float(
                router_absolute_mean[index]
            ),
            "ci95_lower": float(router_lo[index]),
            "ci95_upper": float(router_hi[index]),
        }
        for index in range(16)
    ]
    router_rows.sort(
        key=lambda row: row["mean_absolute_attribution"],
        reverse=True,
    )
    for rank, row in enumerate(router_rows, start=1):
        row["rank"] = rank

    time_rows = [
        {
            "rank": 0,
            "time_index": index,
            "mean_absolute_attribution": float(
                time_absolute_mean[index]
            ),
            "ci95_lower": float(time_lo[index]),
            "ci95_upper": float(time_hi[index]),
        }
        for index in range(32)
    ]
    time_rows.sort(
        key=lambda row: row["mean_absolute_attribution"],
        reverse=True,
    )
    for rank, row in enumerate(time_rows, start=1):
        row["rank"] = rank

    channel_csv = attempt_dir / "CHANNEL_ATTRIBUTION.csv"
    group_csv = attempt_dir / "GROUP_ATTRIBUTION.csv"
    router_csv = attempt_dir / "ROUTER_ATTRIBUTION.csv"
    time_csv = attempt_dir / "TIME_ATTRIBUTION.csv"
    sample_npz = attempt_dir / "PER_SAMPLE_AGGREGATES.npz"

    write_csv(channel_csv, channel_rows)
    write_csv(group_csv, group_rows)
    write_csv(router_csv, router_rows)
    write_csv(time_csv, time_rows)
    atomic_npz(
        sample_npz,
        {
            key: value
            for key, value in combined.items()
            if key != "integrated_gradients"
        },
    )

    result = {
        "task": task,
        "integration_points": points,
        "sample_count": int(relative.shape[0]),
        "completeness": {
            "median_relative_error": median_relative,
            "p95_relative_error": p95_relative,
            "maximum_relative_error": maximum_relative,
            "mean_absolute_error": float(
                np.mean(combined["absolute_completeness_error"])
            ),
            "pass": completeness_pass,
            "gate": {
                "median_relative_error_max": 0.02,
                "p95_relative_error_max": 0.05,
            },
        },
        "group_ranking_by_total_absolute": group_rows,
        "top_20_channels_by_absolute": channel_rows[:20],
        "top_8_routers_by_absolute": router_rows[:8],
        "top_10_time_steps_by_absolute": time_rows[:10],
        "artifacts": {
            "channel_csv": str(channel_csv),
            "group_csv": str(group_csv),
            "router_csv": str(router_csv),
            "time_csv": str(time_csv),
            "per_sample_aggregates": str(sample_npz),
        },
        "artifact_hashes": {
            "channel_csv_sha256": sha256_file(channel_csv),
            "group_csv_sha256": sha256_file(group_csv),
            "router_csv_sha256": sha256_file(router_csv),
            "time_csv_sha256": sha256_file(time_csv),
            "per_sample_aggregates_sha256": sha256_file(sample_npz),
        },
    }
    result_path = attempt_dir / "ATTEMPT_RESULT.json"
    atomic_json(result_path, result)
    atomic_text(
        attempt_dir / "ATTEMPT_COMPLETE",
        f"task={task}\npoints={points}\npass={str(completeness_pass).lower()}\n",
    )
    return result


def execute_attempt(
    *,
    task: str,
    points: int,
    task_rows_list: list[dict[str, Any]],
    task_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    cache: dict[str, np.ndarray],
    bootstrap_seed: int,
) -> dict[str, Any]:
    attempt_dir = task_dir / f"attempt_points_{points}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    result_path = attempt_dir / "ATTEMPT_RESULT.json"
    complete_path = attempt_dir / "ATTEMPT_COMPLETE"

    if complete_path.is_file() and result_path.is_file():
        return load_json(result_path)

    outputs = []
    total_batches = math.ceil(
        len(task_rows_list) / SAMPLE_MICROBATCH_SIZE
    )

    for batch_number, batch_start in enumerate(
        range(0, len(task_rows_list), SAMPLE_MICROBATCH_SIZE)
    ):
        batch_end = min(
            batch_start + SAMPLE_MICROBATCH_SIZE,
            len(task_rows_list),
        )
        paths = batch_paths(attempt_dir, batch_number)

        if paths["complete"].is_file():
            outputs.append(load_completed_batch(paths))
            print(
                f"IG_batch_resume={task}:points={points}:"
                f"{batch_number + 1}/{total_batches}",
                flush=True,
            )
            continue

        rows = task_rows_list[batch_start:batch_end]
        attack_indices = np.asarray(
            [row["attack_index"] for row in rows],
            dtype=np.int64,
        )
        control_indices = np.asarray(
            [row["control_index"] for row in rows],
            dtype=np.int64,
        )

        labels = labels_for_batch(
            task,
            attack_indices,
            cache,
            device,
        )
        result = compute_ig_batch(
            model=model,
            task=task,
            attack_x=np.asarray(cache["x"][attack_indices]),
            control_x=np.asarray(cache["x"][control_indices]),
            attack_mask=np.asarray(
                cache["physical_port_mask"][attack_indices]
            ),
            labels=labels,
            device=device,
            points=points,
        )
        result["attack_indices"] = attack_indices
        result["control_indices"] = control_indices

        atomic_npz(paths["npz"], result)
        manifest = {
            "status": "PASS",
            "task": task,
            "integration_points": points,
            "batch_number": batch_number,
            "batch_start": batch_start,
            "batch_end": batch_end,
            "sample_count": int(attack_indices.size),
            "attack_indices": attack_indices.tolist(),
            "control_indices": control_indices.tolist(),
            "npz": str(paths["npz"]),
            "npz_sha256": sha256_file(paths["npz"]),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(paths["manifest"], manifest)
        atomic_text(paths["complete"], "PASS\n")
        outputs.append(result)

        print(
            f"IG_batch_complete={task}:points={points}:"
            f"{batch_number + 1}/{total_batches}:"
            f"samples={batch_end}/{len(task_rows_list)}",
            flush=True,
        )

        del result
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return aggregate_attempt(
        task=task,
        points=points,
        attempt_dir=attempt_dir,
        batch_outputs=outputs,
        task_rows_list=task_rows_list,
        bootstrap_seed=bootstrap_seed,
    )


def execute_task(
    *,
    task: str,
    run_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    selection: dict[str, Any],
    cache: dict[str, np.ndarray],
    task_index: int,
) -> dict[str, Any]:
    task_dir = run_dir / "tasks" / task
    task_dir.mkdir(parents=True, exist_ok=True)
    task_complete = task_dir / "TASK_COMPLETE"
    final_path = task_dir / "TASK_RESULT.json"

    if task_complete.is_file() and final_path.is_file():
        return load_json(final_path)

    rows = task_rows(task, selection, cache)
    sample_manifest = {
        "task": task,
        "selected_sample_count": len(rows),
        "attack_indices": [row["attack_index"] for row in rows],
        "control_indices": [row["control_index"] for row in rows],
        "attacker_count_strata": dict(
            Counter(row["attacker_count"] for row in rows)
        ),
        "role_nonempty_filter_applied": task in ROLE_LABEL_KEYS,
    }
    atomic_json(task_dir / "TASK_SAMPLE_MANIFEST.json", sample_manifest)

    first = execute_attempt(
        task=task,
        points=DEFAULT_IG_POINTS,
        task_rows_list=rows,
        task_dir=task_dir,
        model=model,
        device=device,
        cache=cache,
        bootstrap_seed=BOOTSTRAP_SEED + task_index * 100,
    )

    if first["completeness"]["pass"]:
        selected_attempt = first
        fallback_used = False
    else:
        print(
            f"IG_completeness_retry={task}:"
            f"points={FALLBACK_IG_POINTS}",
            flush=True,
        )
        second = execute_attempt(
            task=task,
            points=FALLBACK_IG_POINTS,
            task_rows_list=rows,
            task_dir=task_dir,
            model=model,
            device=device,
            cache=cache,
            bootstrap_seed=BOOTSTRAP_SEED + task_index * 100 + 50,
        )
        require(
            second["completeness"]["pass"],
            f"{task} failed IG completeness at both 64 and 128 points",
        )
        selected_attempt = second
        fallback_used = True

    final = {
        "task": task,
        "status": "PASS",
        "sample_count": len(rows),
        "fallback_128_points_used": fallback_used,
        "selected_integration_points": selected_attempt[
            "integration_points"
        ],
        "selected_attempt": selected_attempt,
        "sample_manifest": str(task_dir / "TASK_SAMPLE_MANIFEST.json"),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(final_path, final)
    atomic_text(task_complete, "PASS\n")

    print(
        f"IG_task_complete={task}:"
        f"samples={len(rows)}:"
        f"points={final['selected_integration_points']}:"
        f"median_rel={selected_attempt['completeness']['median_relative_error']}:"
        f"p95_rel={selected_attempt['completeness']['p95_relative_error']}",
        flush=True,
    )
    return final


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_root = Path(args.output_root).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")
    require(data_link.is_symlink(), f"dataset symlink missing: {data_link}")
    require(data_link.resolve().is_dir(), "dataset target missing")

    requested_tasks = [
        task.strip()
        for task in args.tasks.split(",")
        if task.strip()
    ]
    require(requested_tasks, "no F6 tasks requested")
    require(
        all(task in TASKS for task in requested_tasks),
        f"unknown requested tasks: {requested_tasks}",
    )

    feature_root = repo / (
        "reports/v5/p3_experiments/f0_d70_feature_study"
    )
    ig_root = feature_root / "integrated_gradients"
    permutation_root = feature_root / "permutation"

    p3_report_path = ig_root / (
        "V5_P3_F6_P3_DATASET_BUILDER_IDENTITY_ROUTE_RECOVERY_REPORT.json"
    )
    p3_lock_path = ig_root / (
        "V5_P3_F6_P3_DATASET_BUILDER_IDENTITY_ROUTE_RECOVERY_LOCK.json"
    )
    p3_mapping_path = ig_root / (
        "F6_P3_FROZEN_ATTACK_TO_CONTROL_MAPPING.json"
    )
    p0_protocol_path = ig_root / (
        "F6_P0_TASK_SPECIFIC_INTEGRATED_GRADIENTS_PROTOCOL.json"
    )
    route_path = permutation_root / (
        "F5_P0_EXECUTION_ROUTE_SOURCE_INVENTORY.json"
    )

    p3_report, p3_lock = verify_report_lock(
        p3_report_path,
        p3_lock_path,
    )
    require(
        p3_lock.get("mapping_resolved") is True,
        "F6-P3 mapping is not resolved",
    )
    require(
        p3_lock.get("actual_F6_execution_authorized") is True,
        "F6-P3 did not authorize actual F6",
    )
    require(p3_lock.get("F7_authorized") is False, "F7 must remain held")
    require(p3_mapping_path.is_file(), "P3 frozen mapping missing")
    require(p0_protocol_path.is_file(), "F6 protocol missing")
    require(route_path.is_file(), "execution-route inventory missing")

    mapping_document = load_json(p3_mapping_path)
    protocol = load_json(p0_protocol_path)
    route = load_json(route_path)

    require(
        mapping_document["mapping_fingerprint_sha256"]
        == "8b6657a8287c899d97437ed3531f6d43b3cce327f0c45f743a9ae324365d5ae8",
        "P3 mapping fingerprint changed from certified terminal result",
    )
    require(
        protocol["baseline"]["type"]
        == "matched_control_feature_tensor",
        "F6 baseline protocol changed",
    )
    require(
        protocol["sample_selection"]["total_items"]
        == EXPECTED_SELECTED_ITEMS,
        "F6 selected sample count changed",
    )
    require(
        protocol["integrated_gradients"]["steps"]
        == DEFAULT_IG_POINTS,
        "F6 default integration points changed",
    )

    f5_run_dir = active_f5_run(permutation_root)
    cache_dir = f5_run_dir / "validation_cache"
    cache_complete_path = cache_dir / "CACHE_COMPLETE.json"
    require(cache_complete_path.is_file(), "F5 validation cache incomplete")
    cache_complete = load_json(cache_complete_path)
    require(
        cache_complete.get("item_count") == EXPECTED_ITEMS,
        "F5 cache item count changed",
    )
    cache = open_cache(cache_dir)

    run_dir = active_f6_run(output_root, args.run_dir)
    device, runtime = configure_official_runtime(args.device)

    certified = route["certified_files"]
    exporter_path = Path(certified["exporter"]["path"]).resolve()
    checkpoint_path = Path(certified["checkpoint"]["path"]).resolve()
    for label, path in (
        ("exporter", exporter_path),
        ("checkpoint", checkpoint_path),
    ):
        require(path.is_file(), f"{label} missing: {path}")
        require(
            sha256_file(path) == certified[label]["actual_sha256"],
            f"{label} hash changed",
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    require(isinstance(checkpoint, dict), "checkpoint is not a dictionary")
    state_dict = extract_state_dict(checkpoint)

    run_contract_path = run_dir / "F6_RUN_CONTRACT.json"
    contract = {
        "stage": STAGE,
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mapping": {
            "path": str(p3_mapping_path),
            "sha256": sha256_file(p3_mapping_path),
            "fingerprint": mapping_document[
                "mapping_fingerprint_sha256"
            ],
        },
        "protocol": {
            "path": str(p0_protocol_path),
            "sha256": sha256_file(p0_protocol_path),
        },
        "F5_validation_cache": {
            "run_dir": str(f5_run_dir),
            "cache_dir": str(cache_dir),
            "cache_complete_sha256": sha256_file(cache_complete_path),
        },
        "exporter": {
            "path": str(exporter_path),
            "sha256": sha256_file(exporter_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "runtime": runtime,
        "sample_selection_seed": SAMPLE_SELECTION_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "items_per_count": ITEMS_PER_COUNT,
        "default_IG_points": DEFAULT_IG_POINTS,
        "fallback_IG_points": FALLBACK_IG_POINTS,
        "sample_microbatch_size": SAMPLE_MICROBATCH_SIZE,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "tasks": list(TASKS),
        "task_target_policy": {
            "graph": "attack logit",
            "count": (
                "true count-class logit minus mean of the other three logits"
            ),
            "source_transit_victim_path": (
                "mean true-positive router logit minus "
                "mean true-negative router logit"
            ),
        },
        "sealed_test_access": False,
    }
    if run_contract_path.is_file():
        existing = load_json(run_contract_path)
        for key in (
            "mapping",
            "protocol",
            "F5_validation_cache",
            "exporter",
            "checkpoint",
            "runtime",
            "sample_selection_seed",
            "bootstrap_seed",
            "items_per_count",
            "default_IG_points",
            "fallback_IG_points",
            "sample_microbatch_size",
            "bootstrap_resamples",
            "tasks",
            "task_target_policy",
        ):
            require(
                existing[key] == contract[key],
                f"F6 resume contract changed at key: {key}",
            )
    else:
        atomic_json(run_contract_path, contract)

    selection = freeze_sample_selection(
        run_dir=run_dir,
        mapping_document=mapping_document,
        cache=cache,
    )

    model, model_capture = capture_model_via_official_exporter(
        exporter_path=exporter_path,
        repo=repo,
        data_link=data_link,
        run_dir=run_dir,
        checkpoint_state_dict=state_dict,
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    require(
        parameter_count(model) == EXPECTED_PARAMETER_COUNT,
        "captured model parameter count changed",
    )
    require(
        state_dict_exact_match(model, state_dict),
        "captured model does not exactly match checkpoint",
    )

    task_results = {}
    for task_index, task in enumerate(TASKS):
        if task not in requested_tasks:
            continue
        print(
            f"IG_task_start={task}:"
            f"default_points={DEFAULT_IG_POINTS}:"
            f"microbatch={SAMPLE_MICROBATCH_SIZE}",
            flush=True,
        )
        task_results[task] = execute_task(
            task=task,
            run_dir=run_dir,
            model=model,
            device=device,
            selection=selection,
            cache=cache,
            task_index=task_index,
        )

    all_complete = all(
        (run_dir / "tasks" / task / "TASK_COMPLETE").is_file()
        for task in TASKS
    )

    if not all_complete:
        print(f"{STAGE}_PARTIAL_COMPLETE")
        print("status=PARTIAL")
        print(f"run_directory={run_dir}")
        print(f"completed_tasks={sorted(task_results)}")
        print(
            "remaining_tasks="
            f"{[task for task in TASKS if not (run_dir / 'tasks' / task / 'TASK_COMPLETE').is_file()]}"
        )
        print("rerun_same_command_to_resume=true")
        print("F6R_result_review_authorized=false")
        print("F7_retraining_ablation_authorized=false")
        print("sealed_test_tensors_loaded=false")
        return 0

    task_results = {
        task: load_json(run_dir / "tasks" / task / "TASK_RESULT.json")
        for task in TASKS
    }

    multi_task_rows = []
    for task in TASKS:
        selected = task_results[task]["selected_attempt"]
        for group_row in selected["group_ranking_by_total_absolute"]:
            multi_task_rows.append({
                "task": task,
                "sample_count": task_results[task]["sample_count"],
                "integration_points": task_results[task][
                    "selected_integration_points"
                ],
                "group": group_row["group"],
                "rank_by_total_absolute": group_row[
                    "rank_by_total_absolute"
                ],
                "rank_by_per_channel_absolute": group_row[
                    "rank_by_per_channel_absolute"
                ],
                "mean_total_absolute_attribution": group_row[
                    "mean_total_absolute_attribution"
                ],
                "mean_absolute_attribution_per_channel": group_row[
                    "mean_absolute_attribution_per_channel"
                ],
                "mean_fraction_of_total_absolute": group_row[
                    "mean_fraction_of_total_absolute"
                ],
                "fraction_ci95_lower": group_row[
                    "fraction_ci95_lower"
                ],
                "fraction_ci95_upper": group_row[
                    "fraction_ci95_upper"
                ],
            })

    multi_task_csv = run_dir / "F6_MULTI_TASK_GROUP_ATTRIBUTION.csv"
    write_csv(multi_task_csv, multi_task_rows)

    summary = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": CLASSIFICATION,
        "run_directory": str(run_dir),
        "model_class": model_capture["model_class"],
        "parameter_count": parameter_count(model),
        "runtime": runtime,
        "selected_item_count": selection["selected_item_count"],
        "attacker_count_strata": selection["attacker_count_strata"],
        "task_results": {
            task: {
                "sample_count": result["sample_count"],
                "selected_integration_points": result[
                    "selected_integration_points"
                ],
                "fallback_128_points_used": result[
                    "fallback_128_points_used"
                ],
                "completeness": result["selected_attempt"][
                    "completeness"
                ],
                "group_ranking_by_total_absolute": result[
                    "selected_attempt"
                ]["group_ranking_by_total_absolute"],
                "top_20_channels_by_absolute": result[
                    "selected_attempt"
                ]["top_20_channels_by_absolute"],
            }
            for task, result in task_results.items()
        },
        "decision": {
            "F6_complete": True,
            "F6R_result_review_authorized": True,
            "F7_retraining_ablation_authorized": False,
            "feature_removal_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "compact_interface_equivalence_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F6R_TASK_SPECIFIC_INTEGRATED_GRADIENTS_RESULT_REVIEW"
            ),
        },
        "artifacts": {
            "run_contract": str(run_contract_path),
            "selection": str(
                run_dir / "F6_SELECTED_ATTACK_CONTROL_ITEMS.json"
            ),
            "model_capture": str(
                run_dir / "F6_OFFICIAL_EXPORTER_MODEL_CAPTURE.json"
            ),
            "multi_task_group_attribution_csv": str(multi_task_csv),
        },
        "provenance": {
            "F6_P3_report_sha256": sha256_file(p3_report_path),
            "F6_P3_lock_sha256": sha256_file(p3_lock_path),
            "F6_P3_mapping_sha256": sha256_file(p3_mapping_path),
            "F6_protocol_sha256": sha256_file(p0_protocol_path),
            "run_contract_sha256": sha256_file(run_contract_path),
            "selection_sha256": sha256_file(
                run_dir / "F6_SELECTED_ATTACK_CONTROL_ITEMS.json"
            ),
            "model_capture_sha256": sha256_file(
                run_dir / "F6_OFFICIAL_EXPORTER_MODEL_CAPTURE.json"
            ),
            "multi_task_group_attribution_csv_sha256": sha256_file(
                multi_task_csv
            ),
            "installed_script_sha256": sha256_file(installed_script),
        },
    }

    run_report_path = run_dir / f"{STAGE}_REPORT.json"
    run_lock_path = run_dir / f"{STAGE}_LOCK.json"
    canonical_report_path = output_root / f"{STAGE}_REPORT.json"
    canonical_lock_path = output_root / f"{STAGE}_LOCK.json"

    atomic_json(run_report_path, summary)
    run_lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(run_report_path),
        "F6_complete": True,
        "F6R_authorized": True,
        "F7_authorized": False,
        "feature_removal_authorized": False,
        "hardware_reduction_claim_authorized": False,
        "compact_interface_equivalence_authorized": False,
        "sealed_test_tensors_loaded": False,
    }
    atomic_json(run_lock_path, run_lock)

    canonical_summary = dict(summary)
    canonical_summary["run_report"] = str(run_report_path)
    canonical_summary["run_report_sha256"] = sha256_file(run_report_path)
    canonical_summary["run_lock"] = str(run_lock_path)
    canonical_summary["run_lock_sha256"] = sha256_file(run_lock_path)
    atomic_json(canonical_report_path, canonical_summary)
    atomic_json(
        canonical_lock_path,
        {
            **run_lock,
            "report_sha256": sha256_file(canonical_report_path),
            "run_report_sha256": sha256_file(run_report_path),
            "run_lock_sha256": sha256_file(run_lock_path),
        },
    )
    atomic_text(output_root / f"{STAGE}_COMPLETE", f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print(f"run_directory={run_dir}")
    print(f"model_class={model_capture['model_class']}")
    print(f"parameter_count={parameter_count(model)}")
    print(f"selected_item_count={selection['selected_item_count']}")
    print(f"attacker_count_strata={selection['attacker_count_strata']}")
    for task in TASKS:
        result = task_results[task]
        completeness = result["selected_attempt"]["completeness"]
        leader = result["selected_attempt"][
            "group_ranking_by_total_absolute"
        ][0]
        print(
            f"F6_task={task}:"
            f"samples={result['sample_count']}:"
            f"points={result['selected_integration_points']}:"
            f"fallback_128={str(result['fallback_128_points_used']).lower()}:"
            f"median_rel_error={completeness['median_relative_error']}:"
            f"p95_rel_error={completeness['p95_relative_error']}:"
            f"leading_group={leader['group']}:"
            f"leading_group_fraction={leader['mean_fraction_of_total_absolute']}"
        )
    print("F6R_result_review_authorized=true")
    print("F7_retraining_ablation_authorized=false")
    print("feature_removal_authorized=false")
    print("hardware_reduction_claim_authorized=false")
    print("compact_interface_equivalence_authorized=false")
    print("sealed_test_tensors_loaded=false")
    print(
        "next_stage="
        "V5_P3_F6R_TASK_SPECIFIC_INTEGRATED_GRADIENTS_RESULT_REVIEW"
    )
    print(f"multi_task_group_attribution_csv={multi_task_csv}")
    print(f"report={canonical_report_path}")
    print(f"lock={canonical_lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
