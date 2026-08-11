from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


STAGE = "V5_P3_F5_R3_OFFICIAL_D1_BATCH_AND_FORWARD_ROUTE_REPLAY"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"

EXPECTED_PARAMETER_COUNT = 60553

HEAD_FILES = {
    "graph": "attack_logits.npy",
    "count": "count_logits.npy",
    "source": "source_logits.npy",
    "transit": "transit_logits.npy",
    "victim": "victim_logits.npy",
    "path": "path_logits.npy",
}

FRESH_HEAD_KEYS = {
    "graph": "attack_logits",
    "count": "count_logits",
    "source": "source_logits",
    "transit": "transit_logits",
    "victim": "victim_logits",
    "path": "path_logits",
}

HEAD_WIDTHS = {
    "graph": 1,
    "count": 4,
    "source": 16,
    "transit": 16,
    "victim": 16,
    "path": 16,
}


class _OfficialBatchCaptured(BaseException):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
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


def import_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_name(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def active_run(output_root: Path, explicit: str) -> Path:
    if explicit:
        run_dir = Path(explicit).expanduser().resolve()
    else:
        pointer = output_root / "F5_ACTIVE_RUN_PATH.txt"
        require(pointer.is_file(), f"active-run pointer missing: {pointer}")
        run_dir = Path(
            pointer.read_text(encoding="utf-8").strip()
        ).expanduser().resolve()

    require(run_dir.is_dir(), f"F5 run directory missing: {run_dir}")
    return run_dir


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON artifact missing: {path}")
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


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


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

    if isinstance(output, dict):
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
        f"could not resolve six output heads; found={sorted(result)}",
    )
    return result


def canonicalize_head(
    tensor: torch.Tensor,
    head: str,
    batch_size: int,
) -> np.ndarray:
    require(isinstance(tensor, torch.Tensor), f"{head} output is not a tensor")
    array = tensor.detach().float().cpu().numpy()
    width = HEAD_WIDTHS[head]

    if head == "graph":
        array = np.asarray(array).reshape(batch_size, -1)
        require(array.shape[1] == 1, f"graph width mismatch: {array.shape}")
        return array[:, 0].astype(np.float32, copy=False)

    array = np.asarray(array).reshape(batch_size, -1)
    require(array.shape[1] == width, f"{head} width mismatch: {array.shape}")
    return array.astype(np.float32, copy=False)


def capture_forward_inputs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    x = None
    mask = None

    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
        x = args[0]
    if len(args) >= 2 and isinstance(args[1], torch.Tensor):
        mask = args[1]

    if x is None:
        for key in ("x", "features", "temporal_x"):
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                x = value
                break

    if mask is None:
        for key in ("physical_port_mask", "port_mask", "physical_mask"):
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                mask = value
                break

    require(x is not None, "official forward x tensor not captured")
    require(mask is not None, "official forward mask tensor not captured")
    return x, mask


def capture_official_first_batch(
    *,
    exporter_path: Path,
    repo: Path,
    data_link: Path,
    run_dir: Path,
    checkpoint_state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    exporter_module = import_source(
        exporter_path,
        "_v5_p3_f5_r3_official_exporter",
    )
    require(
        hasattr(exporter_module, "main")
        and callable(exporter_module.main),
        "official D1 exporter does not expose main()",
    )

    probe_dir = run_dir / "official_D1_first_batch_probe"
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    probe_dir.mkdir(parents=True)

    captured: dict[str, Any] = {
        "model": None,
        "x": None,
        "mask": None,
        "outputs": None,
        "hook_registered": False,
        "capture_event": None,
    }
    handles = []
    exporter_resolved = str(exporter_path.resolve())

    def pre_hook_with_kwargs(module, args, kwargs):
        x, mask = capture_forward_inputs(args, kwargs)
        captured["x"] = x.detach().cpu().clone()
        captured["mask"] = mask.detach().cpu().clone()

    def post_hook_with_kwargs(module, args, kwargs, output):
        batch_size = int(captured["x"].shape[0])
        mapped = model_output_mapping(output)
        captured["outputs"] = {
            head: canonicalize_head(mapped[head], head, batch_size)
            for head in HEAD_WIDTHS
        }
        captured["capture_event"] = "official_first_forward_complete"
        raise _OfficialBatchCaptured()

    def tracer(frame, event, arg):
        if event != "line":
            return tracer
        if str(Path(frame.f_code.co_filename).resolve()) != exporter_resolved:
            return tracer
        if captured["hook_registered"]:
            return tracer

        for local_name, value in tuple(frame.f_locals.items()):
            if not isinstance(value, torch.nn.Module):
                continue
            if parameter_count(value) != EXPECTED_PARAMETER_COUNT:
                continue
            if not state_dict_exact_match(value, checkpoint_state_dict):
                continue

            captured["model"] = value
            captured["model_local_name"] = local_name
            captured["model_capture_line"] = int(frame.f_lineno)
            captured["model_capture_function"] = frame.f_code.co_name
            captured["hook_registered"] = True

            try:
                pre_handle = value.register_forward_pre_hook(
                    pre_hook_with_kwargs,
                    with_kwargs=True,
                )
                post_handle = value.register_forward_hook(
                    post_hook_with_kwargs,
                    with_kwargs=True,
                )
            except TypeError:
                def pre_hook_legacy(module, args):
                    pre_hook_with_kwargs(module, args, {})

                def post_hook_legacy(module, args, output):
                    post_hook_with_kwargs(module, args, {}, output)

                pre_handle = value.register_forward_pre_hook(
                    pre_hook_legacy
                )
                post_handle = value.register_forward_hook(
                    post_hook_legacy
                )

            handles.extend([pre_handle, post_handle])
            break

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
            exporter_module.main()
            observation = "official_exporter_returned_without_capture"
        except _OfficialBatchCaptured:
            observation = "official_first_batch_captured"
        except SystemExit as exc:
            observation = f"system_exit_before_capture:{exc.code}"
    finally:
        sys.settrace(old_trace)
        sys.argv = old_argv
        for handle in handles:
            try:
                handle.remove()
            except BaseException:
                pass

    require(
        captured["model"] is not None,
        f"official model was not captured; observation={observation}",
    )
    require(
        captured["x"] is not None
        and captured["mask"] is not None
        and captured["outputs"] is not None,
        f"official first forward was not captured; observation={observation}",
    )

    model = captured["model"]
    model.eval()
    require(
        state_dict_exact_match(model, checkpoint_state_dict),
        "captured model state changed after first-batch abort",
    )

    return {
        **captured,
        "observation": observation,
        "probe_directory": str(probe_dir),
        "exporter_path": str(exporter_path),
        "exporter_sha256": sha256_file(exporter_path),
        "model_class": model.__class__.__name__,
        "parameter_count": parameter_count(model),
        "state_dict_exact_match": True,
    }


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": [int(item) for item in tensor.stride()],
        "contiguous": bool(tensor.is_contiguous()),
        "requires_grad": bool(tensor.requires_grad),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
        "mean": float(tensor.float().mean().item()),
    }


def array_comparison(
    reference: np.ndarray,
    current: np.ndarray,
) -> dict[str, Any]:
    shape_match = reference.shape == current.shape
    if not shape_match:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "current_shape": list(current.shape),
        }

    ref64 = np.asarray(reference, dtype=np.float64)
    cur64 = np.asarray(current, dtype=np.float64)
    absolute = np.abs(cur64 - ref64)

    return {
        "shape_match": True,
        "reference_shape": list(reference.shape),
        "current_shape": list(current.shape),
        "reference_dtype": str(reference.dtype),
        "current_dtype": str(current.dtype),
        "dtype_match": reference.dtype == current.dtype,
        "exact_equal": bool(np.array_equal(reference, current)),
        "max_abs_difference": float(np.max(absolute)),
        "mean_abs_difference": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(absolute * absolute))),
        "p99_abs_difference": float(np.quantile(absolute, 0.99)),
    }


def head_comparison(
    reference_heads: dict[str, np.ndarray],
    current_heads: dict[str, np.ndarray],
) -> dict[str, Any]:
    rows = {}
    for head in HEAD_WIDTHS:
        rows[head] = array_comparison(
            np.asarray(reference_heads[head]),
            np.asarray(current_heads[head]),
        )
    return rows


def comparison_is_close(
    rows: dict[str, dict[str, Any]],
    *,
    max_gate: float,
    mean_gate: float,
) -> bool:
    return all(
        row.get("shape_match") is True
        and row.get("max_abs_difference", float("inf")) <= max_gate
        and row.get("mean_abs_difference", float("inf")) <= mean_gate
        for row in rows.values()
    )


def comparison_is_substantial(
    rows: dict[str, dict[str, Any]],
) -> bool:
    return any(
        row.get("mean_abs_difference", 0.0) >= 1e-4
        or row.get("max_abs_difference", 0.0) >= 1e-3
        for row in rows.values()
    )


def run_model_batch(
    model: torch.nn.Module,
    x: np.ndarray,
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device

    x_array = np.array(x, copy=True)
    mask_array = np.array(mask, copy=True)

    x_tensor = torch.from_numpy(x_array).to(
        device=device,
        non_blocking=False,
    )
    mask_tensor = torch.from_numpy(mask_array).to(
        device=device,
        non_blocking=False,
    )

    model.eval()
    with torch.inference_mode():
        output = model(x_tensor, mask_tensor)

    mapped = model_output_mapping(output)
    batch_size = int(x_tensor.shape[0])
    return {
        head: canonicalize_head(mapped[head], head, batch_size)
        for head in HEAD_WIDTHS
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

    permutation_root = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/permutation"
    )
    metric_dir = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/metric_adapter"
    )

    run_dir = active_run(permutation_root, args.run_dir)
    baseline_dir = run_dir / "units/baseline"
    cache_dir = run_dir / "validation_cache"

    r2_report_path = permutation_root / (
        "V5_P3_F5_R2_BASELINE_REPRODUCTION_"
        "MISMATCH_DIAGNOSTIC_REPORT.json"
    )
    r2_lock_path = permutation_root / (
        "V5_P3_F5_R2_BASELINE_REPRODUCTION_"
        "MISMATCH_DIAGNOSTIC_LOCK.json"
    )
    p0_route_path = permutation_root / (
        "F5_P0_EXECUTION_ROUTE_SOURCE_INVENTORY.json"
    )
    run_contract_path = run_dir / "F5_RUN_CONTRACT.json"
    fresh_cert_path = metric_dir / (
        "F4M_R3A_FRESH_REPLAY_14_METRIC_CERTIFICATION.json"
    )

    r2_report, r2_lock = verify_report_lock(
        r2_report_path,
        r2_lock_path,
    )
    require(
        r2_lock.get("classification")
        == "INFERENCE_INPUT_OR_FORWARD_ROUTE_MISMATCH",
        "F5-R3 expected input/forward-route mismatch classification",
    )
    require(
        r2_lock.get("F5_permutation_units_authorized") is False,
        "permutation units must remain held",
    )

    route = load_json(p0_route_path)
    run_contract = load_json(run_contract_path)
    fresh_cert = load_json(fresh_cert_path)

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

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    require(isinstance(checkpoint, dict), "checkpoint is not a dictionary")
    state_dict = extract_state_dict(checkpoint)

    captured = capture_official_first_batch(
        exporter_path=exporter_path,
        repo=repo,
        data_link=data_link,
        run_dir=run_dir,
        checkpoint_state_dict=state_dict,
    )

    model = captured["model"]
    official_x_tensor = captured["x"]
    official_mask_tensor = captured["mask"]
    official_heads = captured["outputs"]

    official_batch_size = int(official_x_tensor.shape[0])
    f5_batch_size = int(run_contract["batch_size"])

    cache_x = np.load(
        cache_dir / "x.npy",
        allow_pickle=False,
        mmap_mode="r",
    )
    cache_mask = np.load(
        cache_dir / "physical_port_mask.npy",
        allow_pickle=False,
        mmap_mode="r",
    )

    require(
        cache_x.shape[0] >= max(official_batch_size, f5_batch_size),
        "cache is shorter than required replay batch",
    )

    official_x = official_x_tensor.numpy()
    official_mask = official_mask_tensor.numpy()

    cache_official_x = np.asarray(
        cache_x[:official_batch_size],
    )
    cache_official_mask = np.asarray(
        cache_mask[:official_batch_size],
    )

    input_comparison = {
        "x": array_comparison(official_x, cache_official_x),
        "physical_port_mask": array_comparison(
            official_mask,
            cache_official_mask,
        ),
        "official_x_tensor": tensor_summary(official_x_tensor),
        "official_mask_tensor": tensor_summary(official_mask_tensor),
        "official_batch_size": official_batch_size,
        "F5_batch_size": f5_batch_size,
        "batch_size_match": official_batch_size == f5_batch_size,
    }

    fresh_npz_path = Path(fresh_cert["npz"]).resolve()
    require(fresh_npz_path.is_file(), f"fresh NPZ missing: {fresh_npz_path}")
    require(
        sha256_file(fresh_npz_path) == fresh_cert["npz_sha256"],
        "F4 fresh NPZ hash changed",
    )

    with np.load(fresh_npz_path, allow_pickle=False) as archive:
        fresh_heads = {
            head: np.asarray(
                archive[key][:official_batch_size]
            )
            for head, key in FRESH_HEAD_KEYS.items()
        }

    baseline_heads = {
        head: np.load(
            baseline_dir / filename,
            allow_pickle=False,
            mmap_mode="r",
        )[:official_batch_size]
        for head, filename in HEAD_FILES.items()
    }

    official_vs_fresh = head_comparison(
        fresh_heads,
        official_heads,
    )
    baseline_vs_official = head_comparison(
        official_heads,
        baseline_heads,
    )

    official_batch_replay_heads = run_model_batch(
        model,
        cache_official_x,
        cache_official_mask,
    )
    official_batch_replay_vs_official = head_comparison(
        official_heads,
        official_batch_replay_heads,
    )

    f5_x = np.asarray(cache_x[:f5_batch_size])
    f5_mask = np.asarray(cache_mask[:f5_batch_size])
    f5_batch_replay_full = run_model_batch(
        model,
        f5_x,
        f5_mask,
    )
    f5_batch_replay_heads = {
        head: array[:official_batch_size]
        for head, array in f5_batch_replay_full.items()
    }

    f5_batch_replay_vs_official = head_comparison(
        official_heads,
        f5_batch_replay_heads,
    )
    f5_batch_replay_vs_baseline = head_comparison(
        baseline_heads,
        f5_batch_replay_heads,
    )

    inputs_exact = bool(
        input_comparison["x"]["exact_equal"]
        and input_comparison["physical_port_mask"]["exact_equal"]
    )
    official_matches_fresh = comparison_is_close(
        official_vs_fresh,
        max_gate=2e-5,
        mean_gate=5e-7,
    )
    official_batch_replay_matches = comparison_is_close(
        official_batch_replay_vs_official,
        max_gate=2e-5,
        mean_gate=5e-7,
    )
    f5_replay_matches_baseline = comparison_is_close(
        f5_batch_replay_vs_baseline,
        max_gate=2e-4,
        mean_gate=2e-5,
    )
    f5_batch_effect_substantial = comparison_is_substantial(
        f5_batch_replay_vs_official
    )

    reasons = []
    repair_authorized = False

    if not inputs_exact:
        classification = (
            "CACHE_PREPROCESSING_OR_CANONICALIZATION_MISMATCH"
        )
        reasons.append(
            "The official D1 model inputs do not exactly equal the cached "
            "first-batch features and/or physical-port mask."
        )
        next_stage = (
            "V5_P3_F5_R3A_OFFICIAL_BATCH_CACHE_REBUILD"
        )

    elif not official_matches_fresh:
        classification = (
            "CURRENT_OFFICIAL_D1_RUNTIME_DIVERGES_FROM_F4_FRESH"
        )
        reasons.append(
            "The current official D1 first-batch output does not reproduce "
            "the preserved F4 fresh output at the previously observed "
            "floating-point scale."
        )
        next_stage = (
            "V5_P3_F5_R3A_OFFICIAL_RUNTIME_PROVENANCE_REVIEW"
        )

    elif (
        official_batch_size != f5_batch_size
        and official_batch_replay_matches
        and f5_replay_matches_baseline
        and f5_batch_effect_substantial
    ):
        classification = (
            "BATCH_SIZE_DEPENDENT_NUMERIC_ROUTE_MISMATCH_CONFIRMED"
        )
        reasons.extend([
            "Official D1 and F5 used different inference batch sizes.",
            "Cached inputs exactly match the official D1 first batch.",
            "The official-size cache replay reproduces the official output.",
            "The F5-size replay reproduces the completed F5 baseline.",
            "Changing only batch size produces the substantial logit drift "
            "observed by F5-R2.",
        ])
        repair_authorized = True
        next_stage = (
            "V5_P3_F5_R3A_OFFICIAL_BATCH_SIZE_BASELINE_RECOVERY_"
            "AND_PERMUTATION_RESUME"
        )

    elif (
        official_batch_replay_matches
        and not f5_batch_effect_substantial
    ):
        classification = (
            "FORWARD_ROUTE_MATCHES_OFFICIAL_BATCH_BUT_BASELINE_PROCESS_DRIFT"
        )
        reasons.append(
            "The exact official-size cache replay matches the official D1 "
            "batch, but changing to the F5 batch size does not explain the "
            "stored baseline discrepancy."
        )
        next_stage = (
            "V5_P3_F5_R3A_BASELINE_PROCESS_RUNTIME_REVIEW"
        )

    else:
        classification = (
            "MIXED_BATCH_FORWARD_OR_RUNTIME_MISMATCH"
        )
        reasons.append(
            "The first-batch evidence does not isolate the mismatch to one "
            "high-confidence cause."
        )
        next_stage = (
            "V5_P3_F5_R3A_TARGETED_BATCH_FORWARD_REVIEW"
        )

    input_path = output_dir / (
        "F5_R3_OFFICIAL_D1_INPUT_VS_CACHE_COMPARISON.json"
    )
    output_path = output_dir / (
        "F5_R3_OFFICIAL_D1_AND_CONTROLLED_REPLAY_LOGIT_COMPARISON.json"
    )
    capture_path = output_dir / (
        "F5_R3_OFFICIAL_D1_FIRST_BATCH_CAPTURE_PROVENANCE.json"
    )
    decision_path = output_dir / (
        "F5_R3_BATCH_FORWARD_ROUTE_CLASSIFICATION.json"
    )

    capture_document = {
        key: value
        for key, value in captured.items()
        if key not in ("model", "x", "mask", "outputs")
    }
    capture_document.update({
        "official_batch_size": official_batch_size,
        "official_x": tensor_summary(official_x_tensor),
        "official_physical_port_mask": tensor_summary(
            official_mask_tensor
        ),
        "official_output_shapes": {
            head: list(array.shape)
            for head, array in official_heads.items()
        },
    })

    atomic_json(input_path, input_comparison)
    atomic_json(capture_path, capture_document)
    atomic_json(
        output_path,
        {
            "official_D1_vs_F4_fresh": official_vs_fresh,
            "stored_F5_baseline_vs_official_D1": baseline_vs_official,
            "official_batch_cache_replay_vs_official_D1": (
                official_batch_replay_vs_official
            ),
            "F5_batch_cache_replay_vs_official_D1": (
                f5_batch_replay_vs_official
            ),
            "F5_batch_cache_replay_vs_stored_F5_baseline": (
                f5_batch_replay_vs_baseline
            ),
            "gates": {
                "official_matches_F4_fresh": official_matches_fresh,
                "official_batch_cache_replay_matches_official": (
                    official_batch_replay_matches
                ),
                "F5_batch_cache_replay_matches_stored_baseline": (
                    f5_replay_matches_baseline
                ),
                "F5_batch_effect_substantial": (
                    f5_batch_effect_substantial
                ),
            },
        },
    )
    atomic_json(
        decision_path,
        {
            "classification": classification,
            "reasons": reasons,
            "official_batch_size": official_batch_size,
            "F5_batch_size": f5_batch_size,
            "inputs_exact": inputs_exact,
            "official_matches_F4_fresh": official_matches_fresh,
            "official_batch_cache_replay_matches_official": (
                official_batch_replay_matches
            ),
            "F5_batch_cache_replay_matches_stored_baseline": (
                f5_replay_matches_baseline
            ),
            "F5_batch_effect_substantial": f5_batch_effect_substantial,
            "repair_authorized": repair_authorized,
            "F5_permutation_units_authorized": False,
            "next_stage": next_stage,
        },
    )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": classification,
        "scope": (
            "Capture the official D1 first model batch, compare its exact "
            "features/mask/output against the F5 cache, F4 fresh output, and "
            "stored F5 baseline, then replay the same checkpoint at both the "
            "official and F5 batch sizes to isolate batch-dependent drift."
        ),
        "batch_route": {
            "official_D1_batch_size": official_batch_size,
            "F5_batch_size": f5_batch_size,
            "batch_size_match": official_batch_size == f5_batch_size,
        },
        "findings": {
            "inputs_exact": inputs_exact,
            "official_matches_F4_fresh": official_matches_fresh,
            "official_batch_cache_replay_matches_official": (
                official_batch_replay_matches
            ),
            "F5_batch_cache_replay_matches_stored_baseline": (
                f5_replay_matches_baseline
            ),
            "F5_batch_effect_substantial": f5_batch_effect_substantial,
            "classification_reasons": reasons,
        },
        "decision": {
            "F5_baseline_gate_complete": False,
            "repair_authorized": repair_authorized,
            "F5_permutation_units_authorized": False,
            "F6_integrated_gradients_authorized": False,
            "F7_retraining_ablation_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": next_stage,
        },
        "governance": {
            "official_D1_validation_first_batch_loaded": True,
            "cached_validation_features_loaded": True,
            "model_loaded": True,
            "checkpoint_loaded": True,
            "model_weights_changed": False,
            "thresholds_changed": False,
            "metric_formulas_changed": False,
            "permutation_units_executed": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
        },
        "artifacts": {
            "capture_provenance": str(capture_path),
            "input_comparison": str(input_path),
            "logit_comparison": str(output_path),
            "classification": str(decision_path),
        },
        "provenance": {
            "F5_R2_report_sha256": sha256_file(r2_report_path),
            "F5_R2_lock_sha256": sha256_file(r2_lock_path),
            "F5_P0_route_sha256": sha256_file(p0_route_path),
            "run_contract_sha256": sha256_file(run_contract_path),
            "fresh_certification_sha256": sha256_file(fresh_cert_path),
            "fresh_npz_sha256": sha256_file(fresh_npz_path),
            "exporter_sha256": sha256_file(exporter_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "installed_script_sha256": sha256_file(installed_script),
            "capture_provenance_sha256": sha256_file(capture_path),
            "input_comparison_sha256": sha256_file(input_path),
            "logit_comparison_sha256": sha256_file(output_path),
            "classification_sha256": sha256_file(decision_path),
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
            "capture_provenance_sha256": sha256_file(capture_path),
            "input_comparison_sha256": sha256_file(input_path),
            "logit_comparison_sha256": sha256_file(output_path),
            "classification_sha256": sha256_file(decision_path),
            "classification": classification,
            "repair_authorized": repair_authorized,
            "F5_permutation_units_authorized": False,
            "F6_authorized": False,
            "model_loaded": True,
            "checkpoint_loaded": True,
            "cached_validation_features_loaded": True,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"run_directory={run_dir}")
    print(f"official_D1_batch_size={official_batch_size}")
    print(f"F5_batch_size={f5_batch_size}")
    print(
        "batch_size_match="
        f"{str(official_batch_size == f5_batch_size).lower()}"
    )
    print(
        "official_x_exact_cache="
        f"{str(input_comparison['x']['exact_equal']).lower()}"
    )
    print(
        "official_mask_exact_cache="
        f"{str(input_comparison['physical_port_mask']['exact_equal']).lower()}"
    )
    print(f"inputs_exact={str(inputs_exact).lower()}")

    for label, comparisons in (
        ("official_vs_F4_fresh", official_vs_fresh),
        ("stored_F5_baseline_vs_official", baseline_vs_official),
        (
            "official_batch_cache_replay_vs_official",
            official_batch_replay_vs_official,
        ),
        (
            "F5_batch_cache_replay_vs_official",
            f5_batch_replay_vs_official,
        ),
        (
            "F5_batch_cache_replay_vs_stored_baseline",
            f5_batch_replay_vs_baseline,
        ),
    ):
        for head, row in comparisons.items():
            print(
                f"{label}_{head}="
                f"max_abs_diff={row.get('max_abs_difference')}:"
                f"mean_abs_diff={row.get('mean_abs_difference')}:"
                f"rmse={row.get('rmse')}"
            )

    print(
        "official_matches_F4_fresh="
        f"{str(official_matches_fresh).lower()}"
    )
    print(
        "official_batch_cache_replay_matches_official="
        f"{str(official_batch_replay_matches).lower()}"
    )
    print(
        "F5_batch_cache_replay_matches_stored_baseline="
        f"{str(f5_replay_matches_baseline).lower()}"
    )
    print(
        "F5_batch_effect_substantial="
        f"{str(f5_batch_effect_substantial).lower()}"
    )
    print(f"failure_classification={classification}")
    print(f"classification_reasons={reasons}")
    print(f"repair_authorized={str(repair_authorized).lower()}")
    print("F5_baseline_gate_complete=false")
    print("F5_permutation_units_authorized=false")
    print("F6_integrated_gradients_authorized=false")
    print("model_loaded=true")
    print("checkpoint_loaded=true")
    print("cached_validation_features_loaded=true")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={next_stage}")
    print(f"capture_provenance={capture_path}")
    print(f"input_comparison={input_path}")
    print(f"logit_comparison={output_path}")
    print(f"classification_report={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
