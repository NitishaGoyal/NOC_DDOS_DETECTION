from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


STAGE = "V5_P3_F5_R3A_BASELINE_PROCESS_RUNTIME_REVIEW"
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

RUNTIME_FIELDS = (
    "deterministic_algorithms_enabled",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "cuda_matmul_allow_tf32",
    "cudnn_allow_tf32",
    "float32_matmul_precision",
)


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


def current_runtime_state() -> dict[str, Any]:
    return {
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
        "grad_enabled": bool(torch.is_grad_enabled()),
        "inference_mode_enabled": bool(
            torch.is_inference_mode_enabled()
        ),
        "autocast_enabled": bool(torch.is_autocast_enabled()),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
    }


def apply_runtime_state(state: dict[str, Any]) -> None:
    torch.use_deterministic_algorithms(
        bool(state["deterministic_algorithms_enabled"]),
        warn_only=True,
    )
    torch.backends.cudnn.deterministic = bool(
        state["cudnn_deterministic"]
    )
    torch.backends.cudnn.benchmark = bool(
        state["cudnn_benchmark"]
    )
    torch.backends.cuda.matmul.allow_tf32 = bool(
        state["cuda_matmul_allow_tf32"]
    )
    torch.backends.cudnn.allow_tf32 = bool(
        state["cudnn_allow_tf32"]
    )
    torch.set_float32_matmul_precision(
        str(state["float32_matmul_precision"])
    )


@contextmanager
def runtime_context(state: dict[str, Any]):
    previous = current_runtime_state()
    apply_runtime_state(state)
    try:
        yield
    finally:
        apply_runtime_state(previous)


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
        "_v5_p3_f5_r3a_official_exporter",
    )
    require(
        hasattr(exporter_module, "main")
        and callable(exporter_module.main),
        "official D1 exporter does not expose main()",
    )

    probe_dir = run_dir / "official_D1_runtime_probe"
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    probe_dir.mkdir(parents=True)

    captured: dict[str, Any] = {
        "model": None,
        "x": None,
        "mask": None,
        "outputs": None,
        "runtime": None,
        "hook_registered": False,
    }
    handles = []
    exporter_resolved = str(exporter_path.resolve())

    def pre_hook_with_kwargs(module, args, kwargs):
        x, mask = capture_forward_inputs(args, kwargs)
        captured["x"] = x.detach().cpu().clone()
        captured["mask"] = mask.detach().cpu().clone()
        captured["runtime"] = current_runtime_state()

    def post_hook_with_kwargs(module, args, kwargs, output):
        batch_size = int(captured["x"].shape[0])
        mapped = model_output_mapping(output)
        captured["outputs"] = {
            head: canonicalize_head(mapped[head], head, batch_size)
            for head in HEAD_WIDTHS
        }
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

    require(captured["model"] is not None, "official model not captured")
    require(captured["x"] is not None, "official x not captured")
    require(captured["mask"] is not None, "official mask not captured")
    require(captured["outputs"] is not None, "official outputs not captured")
    require(captured["runtime"] is not None, "official runtime not captured")

    model = captured["model"]
    model.eval()
    require(
        state_dict_exact_match(model, checkpoint_state_dict),
        "captured official model no longer matches checkpoint",
    )

    captured["observation"] = observation
    captured["probe_directory"] = str(probe_dir)
    captured["exporter_path"] = str(exporter_path)
    captured["exporter_sha256"] = sha256_file(exporter_path)
    captured["model_class"] = model.__class__.__name__
    captured["parameter_count"] = parameter_count(model)
    captured["state_dict_exact_match"] = True
    return captured


def run_model_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    runtime: dict[str, Any],
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device

    x_tensor = x.detach().clone().to(
        device=device,
        non_blocking=False,
    )
    mask_tensor = mask.detach().clone().to(
        device=device,
        non_blocking=False,
    )

    model.eval()
    with runtime_context(runtime):
        with torch.inference_mode():
            output = model(x_tensor, mask_tensor)

    mapped = model_output_mapping(output)
    batch_size = int(x_tensor.shape[0])
    return {
        head: canonicalize_head(mapped[head], head, batch_size)
        for head in HEAD_WIDTHS
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
        "exact_equal": bool(np.array_equal(reference, current)),
        "max_abs_difference": float(np.max(absolute)),
        "mean_abs_difference": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(absolute * absolute))),
    }


def compare_heads(
    reference: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    return {
        head: array_comparison(reference[head], current[head])
        for head in HEAD_WIDTHS
    }


def aggregate_distance(
    comparison: dict[str, dict[str, Any]],
) -> dict[str, float]:
    max_abs = max(
        row["max_abs_difference"]
        for row in comparison.values()
    )
    mean_abs = float(
        np.mean(
            [
                row["mean_abs_difference"]
                for row in comparison.values()
            ]
        )
    )
    rmse = float(
        np.mean(
            [
                row["rmse"]
                for row in comparison.values()
            ]
        )
    )
    return {
        "maximum_head_max_abs_difference": max_abs,
        "mean_of_head_mean_abs_differences": mean_abs,
        "mean_of_head_rmse": rmse,
    }


def close_to_reference(
    comparison: dict[str, dict[str, Any]],
    *,
    max_gate: float = 2e-5,
    mean_gate: float = 5e-7,
) -> bool:
    return all(
        row["max_abs_difference"] <= max_gate
        and row["mean_abs_difference"] <= mean_gate
        for row in comparison.values()
    )


def normalized_runtime(
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: runtime[key]
        for key in RUNTIME_FIELDS
    }


def runtime_from_run_contract(
    run_contract: dict[str, Any],
) -> dict[str, Any]:
    runtime = run_contract["runtime"]
    return {
        key: runtime[key]
        for key in RUNTIME_FIELDS
    }


def differing_fields(
    official: dict[str, Any],
    f5: dict[str, Any],
) -> list[str]:
    return [
        field for field in RUNTIME_FIELDS
        if official[field] != f5[field]
    ]


def configuration_from_subset(
    official: dict[str, Any],
    f5: dict[str, Any],
    subset: tuple[str, ...],
) -> dict[str, Any]:
    result = dict(official)
    for field in subset:
        result[field] = f5[field]
    return result


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

    r3_report_path = permutation_root / (
        "V5_P3_F5_R3_OFFICIAL_D1_BATCH_AND_FORWARD_ROUTE_REPLAY_REPORT.json"
    )
    r3_lock_path = permutation_root / (
        "V5_P3_F5_R3_OFFICIAL_D1_BATCH_AND_FORWARD_ROUTE_REPLAY_LOCK.json"
    )
    p0_route_path = permutation_root / (
        "F5_P0_EXECUTION_ROUTE_SOURCE_INVENTORY.json"
    )
    run_contract_path = run_dir / "F5_RUN_CONTRACT.json"
    fresh_cert_path = metric_dir / (
        "F4M_R3A_FRESH_REPLAY_14_METRIC_CERTIFICATION.json"
    )

    r3_report, r3_lock = verify_report_lock(
        r3_report_path,
        r3_lock_path,
    )
    require(
        r3_lock.get("classification")
        == "FORWARD_ROUTE_MATCHES_OFFICIAL_BATCH_BUT_BASELINE_PROCESS_DRIFT",
        "F5-R3 did not authorize runtime-process review",
    )
    require(
        r3_lock.get("repair_authorized") is False,
        "F5-R3 unexpectedly authorized repair",
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
    x = captured["x"]
    mask = captured["mask"]
    official_heads = captured["outputs"]
    official_runtime = normalized_runtime(captured["runtime"])
    f5_runtime = runtime_from_run_contract(run_contract)

    baseline_heads = {
        head: np.asarray(
            np.load(
                baseline_dir / filename,
                allow_pickle=False,
                mmap_mode="r",
            )[: x.shape[0]]
        )
        for head, filename in HEAD_FILES.items()
    }

    fresh_npz_path = Path(fresh_cert["npz"]).resolve()
    require(fresh_npz_path.is_file(), f"fresh NPZ missing: {fresh_npz_path}")
    require(
        sha256_file(fresh_npz_path) == fresh_cert["npz_sha256"],
        "fresh NPZ hash changed",
    )
    with np.load(fresh_npz_path, allow_pickle=False) as archive:
        fresh_heads = {
            head: np.asarray(
                archive[key][: x.shape[0]]
            )
            for head, key in FRESH_HEAD_KEYS.items()
        }

    official_replay_heads = run_model_batch(
        model,
        x,
        mask,
        official_runtime,
    )
    f5_replay_heads = run_model_batch(
        model,
        x,
        mask,
        f5_runtime,
    )

    official_replay_vs_official = compare_heads(
        official_heads,
        official_replay_heads,
    )
    official_replay_vs_fresh = compare_heads(
        fresh_heads,
        official_replay_heads,
    )
    f5_replay_vs_baseline = compare_heads(
        baseline_heads,
        f5_replay_heads,
    )
    f5_replay_vs_official = compare_heads(
        official_heads,
        f5_replay_heads,
    )

    official_replay_pass = close_to_reference(
        official_replay_vs_official
    )
    f5_replay_matches_baseline = close_to_reference(
        f5_replay_vs_baseline,
        max_gate=2e-4,
        mean_gate=2e-5,
    )

    differences = differing_fields(official_runtime, f5_runtime)

    subset_rows = []
    for subset_size in range(len(differences) + 1):
        for subset in itertools.combinations(differences, subset_size):
            configuration = configuration_from_subset(
                official_runtime,
                f5_runtime,
                subset,
            )
            heads = run_model_batch(
                model,
                x,
                mask,
                configuration,
            )
            versus_official = compare_heads(
                official_heads,
                heads,
            )
            versus_baseline = compare_heads(
                baseline_heads,
                heads,
            )
            row = {
                "changed_fields_from_official": list(subset),
                "changed_field_count": len(subset),
                "configuration": configuration,
                "versus_official": versus_official,
                "versus_stored_F5_baseline": versus_baseline,
                "aggregate_vs_official": aggregate_distance(
                    versus_official
                ),
                "aggregate_vs_stored_F5_baseline": aggregate_distance(
                    versus_baseline
                ),
                "matches_official": close_to_reference(
                    versus_official
                ),
                "matches_stored_F5_baseline": close_to_reference(
                    versus_baseline,
                    max_gate=2e-4,
                    mean_gate=2e-5,
                ),
            }
            subset_rows.append(row)

    matching_baseline = [
        row for row in subset_rows
        if row["matches_stored_F5_baseline"]
    ]
    matching_baseline.sort(
        key=lambda row: (
            row["changed_field_count"],
            row["aggregate_vs_stored_F5_baseline"][
                "mean_of_head_mean_abs_differences"
            ],
        )
    )

    minimum_matching_subset = (
        matching_baseline[0]
        if matching_baseline
        else None
    )

    full_f5_subset = tuple(differences)
    full_f5_row = next(
        row for row in subset_rows
        if tuple(row["changed_fields_from_official"]) == full_f5_subset
    )

    runtime_drift_confirmed = bool(
        official_replay_pass
        and f5_replay_matches_baseline
        and not close_to_reference(f5_replay_vs_official)
        and minimum_matching_subset is not None
    )

    if runtime_drift_confirmed:
        classification = (
            "F5_RUNTIME_CONFIGURATION_DRIFT_CONFIRMED"
        )
        reasons = [
            "The official runtime replay reproduces the official D1 first "
            "batch at the small F4 numeric scale.",
            "The F5 run-contract runtime replay reproduces the stored F5 "
            "baseline.",
            "The official and F5 runtime configurations differ.",
            "A deterministic subset search identifies the smallest runtime "
            "field set that moves the output from the official result to the "
            "stored F5 baseline.",
        ]
        repair_authorized = True
        next_stage = (
            "V5_P3_F5_R3B_OFFICIAL_RUNTIME_BASELINE_RECOVERY_"
            "AND_PERMUTATION_RESUME"
        )
    else:
        classification = (
            "RUNTIME_CONFIGURATION_NOT_SUFFICIENT_TO_EXPLAIN_BASELINE"
        )
        reasons = [
            "The controlled official/F5 runtime replay did not reproduce both "
            "reference endpoints with sufficient confidence."
        ]
        repair_authorized = False
        next_stage = (
            "V5_P3_F5_R3B_PROCESS_STATE_AND_BACKEND_REVIEW"
        )

    runtime_path = output_dir / (
        "F5_R3A_OFFICIAL_VS_F5_RUNTIME_CONFIGURATION.json"
    )
    replay_path = output_dir / (
        "F5_R3A_OFFICIAL_AND_F5_RUNTIME_REPLAY_COMPARISON.json"
    )
    subset_path = output_dir / (
        "F5_R3A_RUNTIME_FIELD_SUBSET_SEARCH.json"
    )
    decision_path = output_dir / (
        "F5_R3A_RUNTIME_DRIFT_CLASSIFICATION.json"
    )

    atomic_json(
        runtime_path,
        {
            "official_runtime": official_runtime,
            "F5_run_contract_runtime": f5_runtime,
            "differing_fields": differences,
            "official_capture_full_runtime": captured["runtime"],
        },
    )
    atomic_json(
        replay_path,
        {
            "official_runtime_replay_vs_official": (
                official_replay_vs_official
            ),
            "official_runtime_replay_vs_F4_fresh": (
                official_replay_vs_fresh
            ),
            "F5_runtime_replay_vs_stored_F5_baseline": (
                f5_replay_vs_baseline
            ),
            "F5_runtime_replay_vs_official": (
                f5_replay_vs_official
            ),
            "official_runtime_replay_pass": official_replay_pass,
            "F5_runtime_replay_matches_stored_baseline": (
                f5_replay_matches_baseline
            ),
        },
    )
    atomic_json(
        subset_path,
        {
            "differing_fields": differences,
            "configuration_count": len(subset_rows),
            "configurations": subset_rows,
            "matching_stored_baseline_count": len(matching_baseline),
            "minimum_matching_subset": minimum_matching_subset,
            "full_F5_runtime_configuration": full_f5_row,
        },
    )
    atomic_json(
        decision_path,
        {
            "classification": classification,
            "reasons": reasons,
            "runtime_drift_confirmed": runtime_drift_confirmed,
            "minimum_matching_subset": (
                minimum_matching_subset[
                    "changed_fields_from_official"
                ]
                if minimum_matching_subset is not None
                else None
            ),
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
            "Capture the official D1 runtime state at first forward, replay "
            "the exact same checkpoint and first batch under both the official "
            "and frozen F5 runtime configurations, and exhaustively search all "
            "subsets of differing runtime fields."
        ),
        "runtime": {
            "official": official_runtime,
            "F5_run_contract": f5_runtime,
            "differing_fields": differences,
            "configuration_count_tested": len(subset_rows),
        },
        "findings": {
            "official_runtime_replay_pass": official_replay_pass,
            "F5_runtime_replay_matches_stored_baseline": (
                f5_replay_matches_baseline
            ),
            "runtime_drift_confirmed": runtime_drift_confirmed,
            "minimum_matching_subset": minimum_matching_subset,
            "classification_reasons": reasons,
        },
        "decision": {
            "F5_baseline_gate_complete": False,
            "official_runtime_repair_authorized": repair_authorized,
            "F5_permutation_units_authorized": False,
            "F6_integrated_gradients_authorized": False,
            "F7_retraining_ablation_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": next_stage,
        },
        "governance": {
            "model_loaded": True,
            "checkpoint_loaded": True,
            "official_D1_validation_first_batch_loaded": True,
            "cached_validation_features_loaded": False,
            "permutation_units_executed": False,
            "model_weights_changed": False,
            "thresholds_changed": False,
            "metric_formulas_changed": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
        },
        "artifacts": {
            "runtime_configuration": str(runtime_path),
            "runtime_replay": str(replay_path),
            "runtime_subset_search": str(subset_path),
            "classification": str(decision_path),
        },
        "provenance": {
            "F5_R3_report_sha256": sha256_file(r3_report_path),
            "F5_R3_lock_sha256": sha256_file(r3_lock_path),
            "F5_P0_route_sha256": sha256_file(p0_route_path),
            "run_contract_sha256": sha256_file(run_contract_path),
            "fresh_certification_sha256": sha256_file(fresh_cert_path),
            "fresh_npz_sha256": sha256_file(fresh_npz_path),
            "exporter_sha256": sha256_file(exporter_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "installed_script_sha256": sha256_file(installed_script),
            "runtime_configuration_sha256": sha256_file(runtime_path),
            "runtime_replay_sha256": sha256_file(replay_path),
            "runtime_subset_search_sha256": sha256_file(subset_path),
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
            "runtime_configuration_sha256": sha256_file(runtime_path),
            "runtime_replay_sha256": sha256_file(replay_path),
            "runtime_subset_search_sha256": sha256_file(subset_path),
            "classification_sha256": sha256_file(decision_path),
            "classification": classification,
            "runtime_drift_confirmed": runtime_drift_confirmed,
            "repair_authorized": repair_authorized,
            "F5_permutation_units_authorized": False,
            "F6_authorized": False,
            "model_loaded": True,
            "checkpoint_loaded": True,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"run_directory={run_dir}")
    print(f"official_runtime={official_runtime}")
    print(f"F5_run_contract_runtime={f5_runtime}")
    print(f"differing_runtime_fields={differences}")
    print(f"runtime_configuration_count_tested={len(subset_rows)}")
    print(
        "official_runtime_replay_pass="
        f"{str(official_replay_pass).lower()}"
    )
    print(
        "F5_runtime_replay_matches_stored_baseline="
        f"{str(f5_replay_matches_baseline).lower()}"
    )
    for label, comparison in (
        (
            "official_runtime_replay_vs_official",
            official_replay_vs_official,
        ),
        (
            "F5_runtime_replay_vs_stored_baseline",
            f5_replay_vs_baseline,
        ),
        (
            "F5_runtime_replay_vs_official",
            f5_replay_vs_official,
        ),
    ):
        aggregate = aggregate_distance(comparison)
        print(
            f"{label}="
            f"max_head_abs_diff="
            f"{aggregate['maximum_head_max_abs_difference']}:"
            f"mean_head_mae="
            f"{aggregate['mean_of_head_mean_abs_differences']}:"
            f"mean_head_rmse="
            f"{aggregate['mean_of_head_rmse']}"
        )

    if minimum_matching_subset is not None:
        print(
            "minimum_runtime_subset_matching_stored_baseline="
            f"{minimum_matching_subset['changed_fields_from_official']}"
        )
        print(
            "minimum_subset_vs_baseline="
            f"{minimum_matching_subset['aggregate_vs_stored_F5_baseline']}"
        )
        print(
            "minimum_subset_vs_official="
            f"{minimum_matching_subset['aggregate_vs_official']}"
        )
    else:
        print("minimum_runtime_subset_matching_stored_baseline=None")

    for field in differences:
        one_field = next(
            row for row in subset_rows
            if row["changed_fields_from_official"] == [field]
        )
        print(
            f"single_field_test_{field}="
            f"matches_official={one_field['matches_official']}:"
            f"matches_stored_baseline="
            f"{one_field['matches_stored_F5_baseline']}:"
            f"vs_official="
            f"{one_field['aggregate_vs_official']}:"
            f"vs_baseline="
            f"{one_field['aggregate_vs_stored_F5_baseline']}"
        )

    print(f"failure_classification={classification}")
    print(f"classification_reasons={reasons}")
    print(
        "runtime_drift_confirmed="
        f"{str(runtime_drift_confirmed).lower()}"
    )
    print(f"repair_authorized={str(repair_authorized).lower()}")
    print("F5_baseline_gate_complete=false")
    print("F5_permutation_units_authorized=false")
    print("F6_integrated_gradients_authorized=false")
    print("model_loaded=true")
    print("checkpoint_loaded=true")
    print("official_D1_validation_first_batch_loaded=true")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={next_stage}")
    print(f"runtime_configuration={runtime_path}")
    print(f"runtime_replay={replay_path}")
    print(f"runtime_subset_search={subset_path}")
    print(f"classification_report={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
