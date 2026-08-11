from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "V5_P3_F7_E3_R1_RUNTIME_ADAPTER_ENVIRONMENT_RECOVERY"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "ENGINEERING"

EXPECTED_SEED = 107
EXPECTED_PARAMETER_COUNT = 60553
EXPECTED_TRAIN_ITEMS = 110855
EXPECTED_VALIDATION_ITEMS = 13863
EXPECTED_BATCH_SIZE = 256
EXPECTED_TRAIN_BATCHES = 434
EXPECTED_VALIDATION_BATCHES = 55

WARMUP_STEPS = 3
MEASURE_STEPS = 12
TOTAL_STEPS = WARMUP_STEPS + MEASURE_STEPS
PROBE_EXIT_CODE = 87


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)
    if mode is not None:
        path.chmod(mode)


def line_offsets(text: str) -> list[int]:
    offsets = [0]
    for match in re.finditer(r"\n", text):
        offsets.append(match.end())
    return offsets


def absolute_span(node: ast.AST, offsets: list[int]) -> tuple[int, int]:
    require(
        hasattr(node, "lineno")
        and hasattr(node, "col_offset")
        and hasattr(node, "end_lineno")
        and hasattr(node, "end_col_offset"),
        f"node has no complete source span: {type(node).__name__}",
    )
    start = offsets[int(node.lineno) - 1] + int(node.col_offset)
    end = offsets[int(node.end_lineno) - 1] + int(node.end_col_offset)
    return start, end


def apply_replacements(
    text: str,
    replacements: list[tuple[int, int, str]],
) -> str:
    result = text
    for start, end, replacement in sorted(
        replacements,
        key=lambda row: row[0],
        reverse=True,
    ):
        result = result[:start] + replacement + result[end:]
    return result


def target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [
            name
            for child in node.elts
            for name in target_names(child)
        ]
    if isinstance(node, ast.Attribute):
        return [ast.unparse(node)]
    return []


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{call_name(node.value)}.{node.attr}"
    return ast.unparse(node)


def insertion_offset_after_future_imports(
    text: str,
    tree: ast.Module,
    offsets: list[int],
) -> int:
    insertion = 0
    body = list(tree.body)

    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            _, insertion = absolute_span(body[0], offsets)

    for node in body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            _, insertion = absolute_span(node, offsets)

    if insertion:
        newline = text.find("\n", insertion)
        if newline >= 0:
            return newline + 1
    return 0


def percentile(values: list[float], quantile: float) -> float:
    require(values, "cannot calculate percentile of empty values")
    require(0.0 <= quantile <= 1.0, "quantile outside [0,1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def extract_flag(command: list[str], flag: str) -> str:
    require(
        command.count(flag) == 1,
        f"command flag is not unique: {flag}",
    )
    index = command.index(flag)
    require(index + 1 < len(command), f"missing value for {flag}")
    return command[index + 1]


def create_cost_probe_trainer(
    trainer_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    text = trainer_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    tree = ast.parse(text)
    offsets = line_offsets(text)

    optimizer_assignments: list[ast.AST] = []
    optimizer_step_calls: list[ast.Call] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names: list[str] = []
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    names.extend(target_names(target))
            else:
                names.extend(target_names(node.target))
            if "optimizer" in names:
                optimizer_assignments.append(node)

        if isinstance(node, ast.Call):
            if call_name(node.func) == "optimizer.step":
                optimizer_step_calls.append(node)

    require(
        len(optimizer_assignments) == 1,
        "optimizer assignment is not unique: "
        f"{[(getattr(node, 'lineno', None), ast.unparse(node)) for node in optimizer_assignments]}",
    )
    require(
        len(optimizer_step_calls) == 1,
        "optimizer.step call is not unique: "
        f"{[(node.lineno, ast.unparse(node)) for node in optimizer_step_calls]}",
    )

    step_call = optimizer_step_calls[0]
    require(
        not step_call.args and not step_call.keywords,
        "optimizer.step call unexpectedly has arguments",
    )

    helper_source = r'''
_F7_E3_PROBE_STATE = {
    "initialized": False,
    "step_count": 0,
    "last_step_end": None,
    "measured_batch_seconds": [],
    "measured_optimizer_seconds": [],
    "gradient_tensor_counts": [],
    "gradient_l2_values": [],
    "all_gradients_finite": True,
    "initial_parameter_hash": None,
}


def _f7_e3_parameter_hash(optimizer):
    import hashlib as _hashlib

    _digest = _hashlib.sha256()
    for _group_index, _group in enumerate(optimizer.param_groups):
        _digest.update(str(_group_index).encode("utf-8"))
        for _parameter_index, _parameter in enumerate(_group["params"]):
            _tensor = _parameter.detach().cpu().contiguous()
            _digest.update(str(_parameter_index).encode("utf-8"))
            _digest.update(str(_tensor.dtype).encode("utf-8"))
            _digest.update(str(tuple(_tensor.shape)).encode("utf-8"))
            _digest.update(_tensor.numpy().tobytes())
    return _digest.hexdigest()


def _f7_e3_cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _f7_e3_write_probe_payload(optimizer, completed=False, failure=None):
    import json as _json
    import os as _os
    import resource as _resource
    import statistics as _statistics
    import time as _time
    from pathlib import Path as _Path

    _state = _F7_E3_PROBE_STATE
    _batch_seconds = list(_state["measured_batch_seconds"])
    _optimizer_seconds = list(_state["measured_optimizer_seconds"])
    _current_hash = _f7_e3_parameter_hash(optimizer)

    _payload = {
        "sentinel": (
            "F7_E3_BOUNDED_COST_PROBE_COMPLETE"
            if completed
            else "F7_E3_BOUNDED_COST_PROBE_PROGRESS"
        ),
        "created_unix": _time.time(),
        "completed": bool(completed),
        "failure": failure,
        "warmup_steps": int(_os.environ["F7_E3_WARMUP_STEPS"]),
        "measure_steps": int(_os.environ["F7_E3_MEASURE_STEPS"]),
        "optimizer_steps_performed": int(_state["step_count"]),
        "measured_batch_seconds": _batch_seconds,
        "measured_optimizer_seconds": _optimizer_seconds,
        "gradient_tensor_counts": list(
            _state["gradient_tensor_counts"]
        ),
        "gradient_l2_values": list(_state["gradient_l2_values"]),
        "all_gradients_finite": bool(
            _state["all_gradients_finite"]
        ),
        "initial_parameter_hash": _state["initial_parameter_hash"],
        "current_parameter_hash": _current_hash,
        "parameters_changed_after_disposable_steps": (
            _state["initial_parameter_hash"] != _current_hash
        ),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "cuda_current_device": (
            int(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "cuda_memory_allocated_bytes": (
            int(torch.cuda.memory_allocated())
            if torch.cuda.is_available()
            else None
        ),
        "cuda_memory_reserved_bytes": (
            int(torch.cuda.memory_reserved())
            if torch.cuda.is_available()
            else None
        ),
        "cuda_max_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None
        ),
        "cuda_max_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved())
            if torch.cuda.is_available()
            else None
        ),
        "process_max_rss_kib": int(
            _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        ),
    }

    if _batch_seconds:
        _payload["batch_seconds_mean"] = float(
            _statistics.fmean(_batch_seconds)
        )
        _payload["batch_seconds_median"] = float(
            _statistics.median(_batch_seconds)
        )
        _payload["batch_seconds_min"] = float(min(_batch_seconds))
        _payload["batch_seconds_max"] = float(max(_batch_seconds))

    if _optimizer_seconds:
        _payload["optimizer_seconds_mean"] = float(
            _statistics.fmean(_optimizer_seconds)
        )
        _payload["optimizer_seconds_median"] = float(
            _statistics.median(_optimizer_seconds)
        )

    _path = _Path(_os.environ["F7_E3_PROBE_JSON"])
    _path.parent.mkdir(parents=True, exist_ok=True)
    _temporary = _path.with_suffix(_path.suffix + ".tmp")
    _temporary.write_text(
        _json.dumps(_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _os.replace(_temporary, _path)


def _f7_e3_initialize_probe(optimizer):
    import os as _os
    import time as _time

    _state = _F7_E3_PROBE_STATE
    if _state["initialized"]:
        raise RuntimeError("F7 E3 probe initialized more than once")

    _warmup = int(_os.environ["F7_E3_WARMUP_STEPS"])
    _measure = int(_os.environ["F7_E3_MEASURE_STEPS"])
    if _warmup < 1:
        raise RuntimeError("F7 E3 warmup steps must be at least one")
    if _measure < 3:
        raise RuntimeError("F7 E3 measured steps must be at least three")

    _f7_e3_cuda_synchronize()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _state["initialized"] = True
    _state["last_step_end"] = _time.perf_counter()
    _state["initial_parameter_hash"] = _f7_e3_parameter_hash(
        optimizer
    )
    _f7_e3_write_probe_payload(optimizer, completed=False)


def _f7_e3_timed_optimizer_step(optimizer):
    import math as _math
    import os as _os
    import time as _time

    _state = _F7_E3_PROBE_STATE
    if not _state["initialized"]:
        raise RuntimeError("F7 E3 probe was not initialized")

    _warmup = int(_os.environ["F7_E3_WARMUP_STEPS"])
    _measure = int(_os.environ["F7_E3_MEASURE_STEPS"])
    _total = _warmup + _measure

    _gradient_square_sum = 0.0
    _gradient_tensor_count = 0
    _finite = True

    for _group in optimizer.param_groups:
        for _parameter in _group["params"]:
            if _parameter.grad is None:
                continue
            _gradient = _parameter.grad.detach()
            _gradient_tensor_count += 1
            _finite = (
                _finite
                and bool(torch.isfinite(_gradient).all().item())
            )
            _gradient_square_sum += float(
                torch.sum(
                    _gradient.float() * _gradient.float()
                ).item()
            )

    if _gradient_tensor_count <= 0:
        _f7_e3_write_probe_payload(
            optimizer,
            completed=False,
            failure="no gradients before optimizer step",
        )
        raise RuntimeError("no gradients before optimizer step")

    if not _finite:
        _state["all_gradients_finite"] = False
        _f7_e3_write_probe_payload(
            optimizer,
            completed=False,
            failure="non-finite gradient before optimizer step",
        )
        raise RuntimeError("non-finite gradient before optimizer step")

    _f7_e3_cuda_synchronize()
    _step_entry = _time.perf_counter()
    _batch_seconds = _step_entry - _state["last_step_end"]

    optimizer.step()

    _f7_e3_cuda_synchronize()
    _step_end = _time.perf_counter()
    _optimizer_seconds = _step_end - _step_entry

    _state["step_count"] += 1
    _step_number = int(_state["step_count"])

    if _step_number <= _warmup:
        if _step_number == _warmup and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    else:
        _state["measured_batch_seconds"].append(
            float(_batch_seconds + _optimizer_seconds)
        )
        _state["measured_optimizer_seconds"].append(
            float(_optimizer_seconds)
        )
        _state["gradient_tensor_counts"].append(
            int(_gradient_tensor_count)
        )
        _state["gradient_l2_values"].append(
            float(_math.sqrt(_gradient_square_sum))
        )

    _state["last_step_end"] = _step_end
    _f7_e3_write_probe_payload(optimizer, completed=False)

    if _step_number >= _total:
        _f7_e3_write_probe_payload(optimizer, completed=True)
        print("F7_E3_BOUNDED_COST_PROBE_COMPLETE", flush=True)
        raise SystemExit(87)
'''

    replacements: list[tuple[int, int, str]] = []

    insertion = insertion_offset_after_future_imports(
        text,
        tree,
        offsets,
    )
    replacements.append((insertion, insertion, helper_source))

    optimizer_assignment = optimizer_assignments[0]
    _, assignment_end = absolute_span(
        optimizer_assignment,
        offsets,
    )

    source_line_start = text.rfind("\n", 0, assignment_end) + 1
    prefix = text[source_line_start:assignment_end]
    indentation_match = re.match(r"[ \t]*", prefix)
    require(
        indentation_match is not None,
        "could not derive optimizer indentation",
    )
    indentation = indentation_match.group(0)

    replacements.append(
        (
            assignment_end,
            assignment_end,
            (
                "\n"
                + indentation
                + "_f7_e3_initialize_probe(optimizer)"
            ),
        )
    )

    step_start, step_end = absolute_span(step_call, offsets)
    replacements.append(
        (
            step_start,
            step_end,
            "_f7_e3_timed_optimizer_step(optimizer)",
        )
    )

    probe_text = apply_replacements(text, replacements)
    probe_tree = ast.parse(probe_text)
    compile(probe_text, str(output_path), "exec")

    remaining_direct_steps = [
        {
            "line": int(node.lineno),
            "source": ast.unparse(node),
        }
        for node in ast.walk(probe_tree)
        if isinstance(node, ast.Call)
        and call_name(node.func) == "optimizer.step"
    ]

    require(
        len(remaining_direct_steps) == 1,
        "probe trainer direct optimizer.step inventory changed: "
        f"{remaining_direct_steps}",
    )

    wrapper_calls = [
        node
        for node in ast.walk(probe_tree)
        if isinstance(node, ast.Call)
        and call_name(node.func) == "_f7_e3_timed_optimizer_step"
    ]
    require(
        len(wrapper_calls) == 1,
        "timed optimizer wrapper call is not unique",
    )

    atomic_text(output_path, probe_text, mode=0o555)

    return {
        "source_trainer": str(trainer_path),
        "source_trainer_sha256": sha256_file(trainer_path),
        "probe_trainer": str(output_path),
        "probe_trainer_sha256": sha256_file(output_path),
        "optimizer_assignment_line": int(
            optimizer_assignment.lineno
        ),
        "original_optimizer_step_line": int(step_call.lineno),
        "warmup_steps": WARMUP_STEPS,
        "measure_steps": MEASURE_STEPS,
        "total_disposable_optimizer_steps": TOTAL_STEPS,
        "direct_optimizer_step_inside_helper_count": 1,
        "timed_wrapper_call_count": 1,
    }


def synthetic_instrumentation_self_test(root: Path) -> dict[str, Any]:
    source = '''import torch


def main():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.001)
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = parameter.square().sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameter], max_norm=1.0)
        optimizer.step()
'''
    input_path = root / "synthetic_input.py"
    output_path = root / "synthetic_probe.py"
    atomic_text(input_path, source)

    result = create_cost_probe_trainer(input_path, output_path)
    compiled = compile(
        output_path.read_text(encoding="utf-8"),
        str(output_path),
        "exec",
    )
    require(compiled is not None, "synthetic probe compile failed")
    return result


def main() -> int:
    args = parse_args()

    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()

    require(repo.is_dir(), f"repository missing: {repo}")
    require(data_link.is_symlink(), f"dataset symlink missing: {data_link}")
    require(installed_script.is_file(), f"installed script missing: {installed_script}")

    canonical_e2_lock_path = output_dir / (
        "V5_P3_F7_E2_GENERATED_TRAINER_STATIC_AND_RUNTIME_PREFLIGHT_LOCK.json"
    )
    canonical_e2_lock = load_json(canonical_e2_lock_path)

    require(
        canonical_e2_lock.get("status") == "PASS",
        "canonical E2 lock is not PASS",
    )
    require(
        canonical_e2_lock.get("primary_matrix_execution_authorized")
        is True,
        "canonical E2 did not authorize the primary matrix",
    )
    require(
        canonical_e2_lock.get("actual_scientific_training_started")
        is False,
        "scientific training already marked as started",
    )
    require(
        canonical_e2_lock.get("actual_F7_primary_matrix_execution_started")
        is False,
        "primary matrix already marked as started",
    )
    require(
        canonical_e2_lock.get("sealed_test_tensors_loaded") is False,
        "canonical E2 indicates sealed-test access",
    )

    source_report_path = Path(
        canonical_e2_lock["source_report"]
    ).resolve()
    require(
        sha256_file(source_report_path)
        == canonical_e2_lock["source_report_sha256"],
        "canonical E2 source report SHA mismatch",
    )
    e2_report = load_json(source_report_path)
    require(e2_report.get("status") == "PASS", "E2 report is not PASS")

    trainer_path = Path(
        e2_report["artifacts"]["corrected_trainer"]
    ).resolve()
    launcher_path = Path(
        e2_report["artifacts"]["corrected_matrix_launcher"]
    ).resolve()

    require(trainer_path.is_file(), f"corrected trainer missing: {trainer_path}")
    require(launcher_path.is_file(), f"corrected launcher missing: {launcher_path}")
    require(
        sha256_file(trainer_path)
        == canonical_e2_lock["generated_trainer_sha256"],
        "corrected trainer SHA differs from canonical E2",
    )
    require(
        sha256_file(launcher_path)
        == canonical_e2_lock["matrix_launcher_sha256"],
        "corrected launcher SHA differs from canonical E2",
    )

    runtime = e2_report["finding"]["runtime"]
    require(
        runtime["parameter_count"] == EXPECTED_PARAMETER_COUNT,
        "E2 parameter count changed",
    )
    require(
        runtime["train_items"] == EXPECTED_TRAIN_ITEMS,
        "E2 train cardinality changed",
    )
    require(
        runtime["validation_items"] == EXPECTED_VALIDATION_ITEMS,
        "E2 validation cardinality changed",
    )
    require(
        runtime["effective_item_batch_size"] == EXPECTED_BATCH_SIZE,
        "E2 batch size changed",
    )
    require(
        runtime["train_batches"] == EXPECTED_TRAIN_BATCHES,
        "E2 train batch count changed",
    )
    require(
        runtime["validation_batches"] == EXPECTED_VALIDATION_BATCHES,
        "E2 validation batch count changed",
    )
    require(
        runtime["intercept"]["optimizer_step_performed"] is False,
        "E2 preflight unexpectedly performed optimizer step",
    )
    require(
        runtime["intercept"]["all_gradients_finite"] is True,
        "E2 preflight gradients were not finite",
    )

    prior_command = list(runtime["instrumented_command"])
    b1_dir = Path(extract_flag(prior_command, "--b1-dir")).resolve()
    b0_r3_dir = Path(extract_flag(prior_command, "--b0-r3-dir")).resolve()
    legacy_loader = Path(extract_flag(prior_command, "--loader-path")).resolve()
    legacy_model = Path(extract_flag(prior_command, "--model-path")).resolve()

    for required_path in (b1_dir, b0_r3_dir, legacy_loader, legacy_model):
        require(required_path.exists(), f"required compatibility path missing: {required_path}")

    stage_report = output_dir / f"{STAGE}_REPORT.json"
    stage_lock = output_dir / f"{STAGE}_LOCK.json"
    stage_complete = output_dir / f"{STAGE}_COMPLETE"
    require(
        not stage_report.exists()
        and not stage_lock.exists()
        and not stage_complete.exists(),
        "E3 stage artifacts already exist; append-only rerun refused",
    )

    scratch = output_dir / "F7_E3_R1_RUNTIME_ADAPTER_ENVIRONMENT_RECOVERY_SCRATCH"
    require(
        not scratch.exists(),
        f"E3 scratch already exists: {scratch}",
    )
    scratch.mkdir(parents=True)

    self_test_dir = scratch / "instrumentation_self_test"
    self_test_dir.mkdir()
    self_test = synthetic_instrumentation_self_test(self_test_dir)

    probe_trainer = scratch / (
        "run_v5_p3_f7_e3_r1_bounded_cost_probe_trainer.py"
    )
    instrumentation = create_cost_probe_trainer(
        trainer_path,
        probe_trainer,
    )

    run_dir = scratch / "control_dynamic70_cost_probe"
    run_dir.mkdir()
    model_dir = run_dir / "model"
    report_dir = run_dir / "report"
    require(not model_dir.exists(), "probe model dir unexpectedly exists")
    require(not report_dir.exists(), "probe report dir unexpectedly exists")

    probe_json = run_dir / "F7_E3_R1_BOUNDED_COST_PROBE_RESULT.json"

    environment = os.environ.copy()
    environment.update({
        "F7_REPO": str(repo),
        "F7_DATA_LINK": str(data_link),
        "F7_SEED": str(EXPECTED_SEED),
        "F7_RUN_DIR": str(run_dir),
        "F7_ABLATION_LABEL": "control_dynamic70",
        "F7_E3_PROBE_JSON": str(probe_json),
        "F7_E3_WARMUP_STEPS": str(WARMUP_STEPS),
        "F7_E3_MEASURE_STEPS": str(MEASURE_STEPS),
        "PYTHONPATH": (
            str(repo)
            + os.pathsep
            + environment.get("PYTHONPATH", "")
        ),
    })

    required_adapter_environment = {
        "F7_REPO": str(repo),
        "F7_DATA_LINK": str(data_link),
        "F7_SEED": str(EXPECTED_SEED),
        "F7_RUN_DIR": str(run_dir),
        "F7_ABLATION_LABEL": "control_dynamic70",
    }
    for key, expected_value in required_adapter_environment.items():
        require(
            environment.get(key) == expected_value,
            f"runtime adapter environment mismatch for {key}: "
            f"actual={environment.get(key)!r}; "
            f"expected={expected_value!r}",
        )

    require(
        Path(environment["F7_REPO"]).resolve() == repo,
        "F7_REPO does not resolve to the certified repository",
    )
    require(
        Path(environment["F7_DATA_LINK"]).expanduser().is_symlink(),
        "F7_DATA_LINK is not the guarded dataset symlink",
    )
    require(
        int(environment["F7_SEED"]) == EXPECTED_SEED,
        "F7_SEED changed before child launch",
    )
    require(
        Path(environment["F7_RUN_DIR"]).resolve() == run_dir,
        "F7_RUN_DIR changed before child launch",
    )

    command = [
        sys.executable,
        str(probe_trainer),
        "--root",
        str(repo),
        "--b1-dir",
        str(b1_dir),
        "--b0-r3-dir",
        str(b0_r3_dir),
        "--loader-path",
        str(legacy_loader),
        "--model-path",
        str(legacy_model),
        "--model-dir",
        str(model_dir),
        "--report-dir",
        str(report_dir),
        "--seed",
        str(EXPECTED_SEED),
    ]

    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    process_wall_seconds = time.perf_counter() - started

    console_path = run_dir / "console.log"
    atomic_text(
        console_path,
        process.stdout + "\n--- STDERR ---\n" + process.stderr,
    )

    require(
        process.returncode == PROBE_EXIT_CODE,
        "bounded cost probe did not reach planned exit; "
        f"returncode={process.returncode}; console={console_path}",
    )
    combined_console = process.stdout + process.stderr
    require(
        "F7_E3_BOUNDED_COST_PROBE_COMPLETE" in combined_console,
        "bounded cost probe completion sentinel missing",
    )
    require(probe_json.is_file(), "bounded cost probe JSON missing")

    result = load_json(probe_json)
    require(result["completed"] is True, "probe JSON is not complete")
    require(result["failure"] is None, f"probe recorded failure: {result['failure']}")
    require(
        result["warmup_steps"] == WARMUP_STEPS,
        "probe warmup count changed",
    )
    require(
        result["measure_steps"] == MEASURE_STEPS,
        "probe measurement count changed",
    )
    require(
        result["optimizer_steps_performed"] == TOTAL_STEPS,
        "probe optimizer-step count changed",
    )
    require(
        len(result["measured_batch_seconds"]) == MEASURE_STEPS,
        "measured batch timing count changed",
    )
    require(
        len(result["measured_optimizer_seconds"]) == MEASURE_STEPS,
        "measured optimizer timing count changed",
    )
    require(
        all(
            math.isfinite(value) and value > 0.0
            for value in result["measured_batch_seconds"]
        ),
        "invalid measured batch time",
    )
    require(
        all(
            math.isfinite(value) and value >= 0.0
            for value in result["measured_optimizer_seconds"]
        ),
        "invalid measured optimizer time",
    )
    require(
        result["all_gradients_finite"] is True,
        "probe encountered non-finite gradients",
    )
    require(
        all(count > 0 for count in result["gradient_tensor_counts"]),
        "probe measured a batch with no gradients",
    )
    require(
        result["parameters_changed_after_disposable_steps"] is True,
        "disposable optimizer steps did not change parameters",
    )
    require(
        result["cuda_available"] is True,
        "cost probe did not run on CUDA",
    )

    require(
        "test_directory_enumerated: false" in process.stdout,
        "probe did not certify test-directory non-enumeration",
    )
    require(
        "test_tensor_contents_accessed: false" in process.stdout,
        "probe did not certify sealed-test non-access",
    )
    require(
        f"train_items: {EXPECTED_TRAIN_ITEMS}" in process.stdout,
        "probe console train cardinality mismatch",
    )
    require(
        f"validation_items: {EXPECTED_VALIDATION_ITEMS}" in process.stdout,
        "probe console validation cardinality mismatch",
    )
    require(
        f"train_batches: {EXPECTED_TRAIN_BATCHES}" in process.stdout,
        "probe console train batch count mismatch",
    )
    require(
        f"validation_batches: {EXPECTED_VALIDATION_BATCHES}" in process.stdout,
        "probe console validation batch count mismatch",
    )
    require(
        f"parameter_count: {EXPECTED_PARAMETER_COUNT}" in process.stdout,
        "probe console parameter count mismatch",
    )

    checkpoint_suffixes = {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".bin",
    }
    checkpoint_files = [
        str(path)
        for path in run_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in checkpoint_suffixes
    ]
    require(
        not checkpoint_files,
        f"bounded cost probe created checkpoint-like files: {checkpoint_files}",
    )

    batch_seconds = [
        float(value)
        for value in result["measured_batch_seconds"]
    ]
    optimizer_seconds = [
        float(value)
        for value in result["measured_optimizer_seconds"]
    ]

    mean_batch = statistics.fmean(batch_seconds)
    median_batch = statistics.median(batch_seconds)
    p90_batch = percentile(batch_seconds, 0.90)
    min_batch = min(batch_seconds)
    max_batch = max(batch_seconds)
    std_batch = (
        statistics.stdev(batch_seconds)
        if len(batch_seconds) > 1
        else 0.0
    )

    mean_optimizer = statistics.fmean(optimizer_seconds)
    median_optimizer = statistics.median(optimizer_seconds)

    median_train_epoch_seconds = (
        median_batch * EXPECTED_TRAIN_BATCHES
    )
    p90_train_epoch_seconds = (
        p90_batch * EXPECTED_TRAIN_BATCHES
    )
    median_items_per_second = EXPECTED_BATCH_SIZE / median_batch

    estimates = {
        "scope": (
            "train-loop-only lower-bound estimates; validation, epoch-end "
            "metrics, scheduler, checkpointing, and process setup are excluded"
        ),
        "median_train_epoch_seconds": median_train_epoch_seconds,
        "p90_train_epoch_seconds": p90_train_epoch_seconds,
        "median_train_epoch_minutes": (
            median_train_epoch_seconds / 60.0
        ),
        "p90_train_epoch_minutes": (
            p90_train_epoch_seconds / 60.0
        ),
        "single_run_train_only_hours": {
            "15_epochs": median_train_epoch_seconds * 15 / 3600.0,
            "30_epochs": median_train_epoch_seconds * 30 / 3600.0,
            "100_epochs": median_train_epoch_seconds * 100 / 3600.0,
        },
        "six_run_matrix_train_only_hours": {
            "15_epochs_each": (
                median_train_epoch_seconds * 15 * 6 / 3600.0
            ),
            "30_epochs_each": (
                median_train_epoch_seconds * 30 * 6 / 3600.0
            ),
            "100_epochs_each": (
                median_train_epoch_seconds * 100 * 6 / 3600.0
            ),
        },
    }

    timing_summary = {
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURE_STEPS,
        "total_disposable_optimizer_steps": TOTAL_STEPS,
        "batch_seconds": {
            "raw": batch_seconds,
            "mean": mean_batch,
            "median": median_batch,
            "p90": p90_batch,
            "minimum": min_batch,
            "maximum": max_batch,
            "sample_stddev": std_batch,
        },
        "optimizer_seconds": {
            "raw": optimizer_seconds,
            "mean": mean_optimizer,
            "median": median_optimizer,
        },
        "median_items_per_second": median_items_per_second,
        "process_wall_seconds": process_wall_seconds,
        "cuda_device_name": result["cuda_device_name"],
        "cuda_max_memory_allocated_bytes": (
            result["cuda_max_memory_allocated_bytes"]
        ),
        "cuda_max_memory_reserved_bytes": (
            result["cuda_max_memory_reserved_bytes"]
        ),
        "process_max_rss_kib": result["process_max_rss_kib"],
    }

    decision = {
        "stage": STAGE,
        "status": "PASS",
        "classification": CLASSIFICATION,
        "bounded_cost_probe_complete": True,
        "recovery_classification": (
            "original_E3_child_launch_omitted_F7_REPO_F7_DATA_LINK_"
            "and_F7_SEED_required_by_runtime_adapter"
        ),
        "runtime_adapter_environment_verified": True,
        "disposable_optimizer_steps_performed": TOTAL_STEPS,
        "scientific_optimizer_steps_performed": 0,
        "scientific_checkpoint_saved": False,
        "primary_matrix_execution_started": False,
        "sealed_test_access": False,
        "timing_sample_sufficient_for_control_smoke": True,
        "control_run_smoke_authorized": True,
        "full_six_run_matrix_operationally_held": True,
        "next_stage": "V5_P3_F7_E4_CONTROL_RUN_SMOKE_GATE",
    }

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "scope": (
            "Run a disposable control-model training probe for three warm-up "
            "and twelve measured real optimizer steps on CUDA. Measure "
            "synchronized end-to-end training-batch time, optimizer-step time, "
            "finite gradients, peak CUDA memory, and parameter change. Exit "
            "mid-epoch before validation, scheduler, checkpointing, or any "
            "scientific result is produced."
        ),
        "finding": {
            "instrumentation": instrumentation,
            "self_test": self_test,
            "runtime_adapter_environment": {
                "required_keys": sorted(
                    required_adapter_environment
                ),
                "values": required_adapter_environment,
                "all_required_values_verified_before_child_launch": True,
            },
            "probe_result": result,
            "timing_summary": timing_summary,
            "train_only_estimates": estimates,
        },
        "decision": decision,
        "governance": {
            "runtime_adapter_environment_verified": True,
            "training_dataset_accessed": True,
            "validation_dataset_constructed_by_trainer": True,
            "sealed_test_access": False,
            "disposable_optimizer_steps_performed": TOTAL_STEPS,
            "scientific_training_started": False,
            "scientific_checkpoint_saved": False,
            "primary_matrix_execution_started": False,
            "F8_multi_seed_authorized": False,
            "feature_removal_authorized": False,
            "compact_interface_equivalence_authorized": False,
            "hardware_reduction_claim_authorized": False,
        },
        "artifacts": {
            "canonical_E2_lock": str(canonical_e2_lock_path),
            "E2_source_report": str(source_report_path),
            "source_trainer": str(trainer_path),
            "corrected_matrix_launcher": str(launcher_path),
            "probe_trainer": str(probe_trainer),
            "probe_result": str(probe_json),
            "console": str(console_path),
            "scratch": str(scratch),
        },
        "provenance": {
            "canonical_E2_lock_sha256": sha256_file(
                canonical_e2_lock_path
            ),
            "E2_source_report_sha256": sha256_file(
                source_report_path
            ),
            "source_trainer_sha256": sha256_file(trainer_path),
            "corrected_matrix_launcher_sha256": sha256_file(
                launcher_path
            ),
            "probe_trainer_sha256": sha256_file(probe_trainer),
            "probe_result_sha256": sha256_file(probe_json),
            "console_sha256": sha256_file(console_path),
            "installed_script_sha256": sha256_file(installed_script),
        },
    }

    atomic_json(stage_report, report)
    atomic_json(
        stage_lock,
        {
            "stage": STAGE,
            "status": "PASS",
            "report": str(stage_report),
            "report_sha256": sha256_file(stage_report),
            "canonical_E2_lock_sha256": sha256_file(
                canonical_e2_lock_path
            ),
            "source_trainer_sha256": sha256_file(trainer_path),
            "probe_trainer_sha256": sha256_file(probe_trainer),
            "probe_result_sha256": sha256_file(probe_json),
            "runtime_adapter_environment_verified": True,
            "runtime_adapter_environment_keys": sorted(
                required_adapter_environment
            ),
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURE_STEPS,
            "disposable_optimizer_steps_performed": TOTAL_STEPS,
            "scientific_optimizer_steps_performed": 0,
            "scientific_checkpoint_saved": False,
            "primary_matrix_execution_started": False,
            "sealed_test_access": False,
            "control_run_smoke_authorized": True,
            "full_six_run_matrix_operationally_held": True,
            "next_stage": "V5_P3_F7_E4_CONTROL_RUN_SMOKE_GATE",
        },
    )
    atomic_text(stage_complete, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print("runtime_adapter_environment_verified=true")
    print(
        "runtime_adapter_environment_keys="
        + ",".join(sorted(required_adapter_environment))
    )
    print(f"warmup_steps={WARMUP_STEPS}")
    print(f"measured_steps={MEASURE_STEPS}")
    print(f"disposable_optimizer_steps_performed={TOTAL_STEPS}")
    print("scientific_optimizer_steps_performed=0")
    print(f"cuda_device={result['cuda_device_name']}")
    print(f"batch_seconds_mean={mean_batch}")
    print(f"batch_seconds_median={median_batch}")
    print(f"batch_seconds_p90={p90_batch}")
    print(f"batch_seconds_min={min_batch}")
    print(f"batch_seconds_max={max_batch}")
    print(f"batch_seconds_sample_stddev={std_batch}")
    print(f"optimizer_seconds_mean={mean_optimizer}")
    print(f"optimizer_seconds_median={median_optimizer}")
    print(f"median_items_per_second={median_items_per_second}")
    print(
        "cuda_max_memory_allocated_bytes="
        f"{result['cuda_max_memory_allocated_bytes']}"
    )
    print(
        "cuda_max_memory_reserved_bytes="
        f"{result['cuda_max_memory_reserved_bytes']}"
    )
    print(
        "median_train_epoch_minutes="
        f"{estimates['median_train_epoch_minutes']}"
    )
    print(
        "p90_train_epoch_minutes="
        f"{estimates['p90_train_epoch_minutes']}"
    )
    print(
        "single_run_train_only_15_epochs_hours="
        f"{estimates['single_run_train_only_hours']['15_epochs']}"
    )
    print(
        "single_run_train_only_30_epochs_hours="
        f"{estimates['single_run_train_only_hours']['30_epochs']}"
    )
    print(
        "single_run_train_only_100_epochs_hours="
        f"{estimates['single_run_train_only_hours']['100_epochs']}"
    )
    print(
        "six_run_train_only_15_epochs_each_hours="
        f"{estimates['six_run_matrix_train_only_hours']['15_epochs_each']}"
    )
    print(
        "six_run_train_only_30_epochs_each_hours="
        f"{estimates['six_run_matrix_train_only_hours']['30_epochs_each']}"
    )
    print(
        "six_run_train_only_100_epochs_each_hours="
        f"{estimates['six_run_matrix_train_only_hours']['100_epochs_each']}"
    )
    print("train_only_estimates_exclude_validation_and_checkpointing=true")
    print("scientific_checkpoint_saved=false")
    print("primary_matrix_execution_started=false")
    print("sealed_test_access=false")
    print("control_run_smoke_authorized=true")
    print("full_six_run_matrix_operationally_held=true")
    print("next_stage=V5_P3_F7_E4_CONTROL_RUN_SMOKE_GATE")
    print(f"report={stage_report}")
    print(f"lock={stage_lock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
