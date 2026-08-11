from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


STAGE = "V5_P3_F7_M2_EPOCH16_SHUTDOWN_RESUME"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "TRAIN-DERIVED"

EXPECTED_SEED = 107
EXPECTED_LABEL = "ablate_buffer_pressure"
EXPECTED_PARAMETER_COUNT = 60553
EXPECTED_TRAIN_ITEMS = 110855
EXPECTED_VALIDATION_ITEMS = 13863
EXPECTED_TRAIN_BATCHES = 434
EXPECTED_VALIDATION_BATCHES = 55
EXPECTED_COMPLETED_EPOCH = 17
EXPECTED_RESUME_CHECKPOINT_EPOCH = 16
EXPECTED_BEST_SCORE = 0.7406281414244834
EXPECTED_SOURCE_CHECKPOINT_SHA256 = (
    "8f50b87cb11f692d0d0d15143951e62ca6c863a78b43372cf2f99018264b0fcd"
)
EXPECTED_TRAINER_SHA256 = (
    "8baeed5a7469c0a751ab0a614837d08d8d4058e1cd331a3128b9f5e3754e3faf"
)
EXPECTED_LAUNCHER_SHA256 = (
    "fa6e9690f55081ea3364a4a91c391ef08be52e38cfaddd43176051314e202f2a"
)
EXPECTED_ADAPTER_SHA256 = (
    "6e2b355d1c5f4f0e71c3eb0798b263aca2e936d11623d8526a20f6204feba059"
)


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
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON is not an object: {path}")
    return value


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


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_command(pid: int) -> str:
    path = Path(f"/proc/{pid}/cmdline")
    if not path.is_file():
        return ""
    return path.read_bytes().replace(b"\0", b" ").decode(
        "utf-8", errors="replace"
    ).strip()


def line_offsets(text: str) -> list[int]:
    offsets = [0]
    for index, character in enumerate(text):
        if character == "\n":
            offsets.append(index + 1)
    return offsets


def absolute_span(node: ast.AST, offsets: list[int]) -> tuple[int, int]:
    require(
        hasattr(node, "lineno")
        and hasattr(node, "col_offset")
        and hasattr(node, "end_lineno")
        and hasattr(node, "end_col_offset"),
        f"node has incomplete source span: {type(node).__name__}",
    )
    start = offsets[int(node.lineno) - 1] + int(node.col_offset)
    end = offsets[int(node.end_lineno) - 1] + int(node.end_col_offset)
    return start, end


def indentation_at(text: str, position: int) -> str:
    line_start = text.rfind("\n", 0, position) + 1
    prefix = text[line_start:position]
    return prefix[: len(prefix) - len(prefix.lstrip(" \t"))]


def apply_replacements(
    text: str,
    replacements: list[tuple[int, int, str]],
) -> str:
    output = text
    for start, end, replacement in sorted(
        replacements, key=lambda row: row[0], reverse=True
    ):
        output = output[:start] + replacement + output[end:]
    return output


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{call_name(node.value)}.{node.attr}"
    return ast.unparse(node)


def target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [
            name for child in node.elts for name in target_names(child)
        ]
    if isinstance(node, ast.Attribute):
        return [ast.unparse(node)]
    return []


def assignment_nodes(tree: ast.AST, target_name: str) -> list[ast.AST]:
    rows: list[ast.AST] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.extend(target_names(target))
        elif isinstance(node, ast.AnnAssign):
            names.extend(target_names(node.target))
        else:
            continue
        if target_name in names:
            rows.append(node)
    return rows


def extract_flag(command: list[str], flag: str) -> str:
    require(command.count(flag) == 1, f"flag not unique: {flag}")
    index = command.index(flag)
    require(index + 1 < len(command), f"missing value for {flag}")
    return command[index + 1]


def read_history(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"history missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, f"history is empty: {path}")
    return rows


def json_equivalent(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def create_resume_trainer(
    source_trainer: Path,
    resume_trainer: Path,
) -> dict[str, Any]:
    text = source_trainer.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text)
    offsets = line_offsets(text)

    main_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]
    require(len(main_functions) == 1, "module-level main is not unique")
    main_node = main_functions[0]

    epoch_loops: list[ast.For] = []
    for node in ast.walk(main_node):
        if not isinstance(node, ast.For):
            continue
        calls = [child for child in ast.walk(node) if isinstance(child, ast.Call)]
        if any(
            call_name(call.func).rsplit(".", 1)[-1] == "train_one_epoch"
            for call in calls
        ):
            epoch_loops.append(node)
    require(
        len(epoch_loops) == 1,
        "training epoch loop is not unique: "
        f"{[(node.lineno, ast.unparse(node.iter)) for node in epoch_loops]}",
    )
    epoch_loop = epoch_loops[0]
    require(
        isinstance(epoch_loop.iter, ast.Call)
        and call_name(epoch_loop.iter.func) == "range",
        "training epoch loop is not a range call",
    )
    require(
        len(epoch_loop.iter.args) == 2
        and isinstance(epoch_loop.iter.args[0], ast.Constant)
        and int(epoch_loop.iter.args[0].value) == 1
        and isinstance(epoch_loop.iter.args[1], ast.Constant)
        and int(epoch_loop.iter.args[1].value) == 101,
        f"unexpected frozen epoch range: {ast.unparse(epoch_loop.iter)}",
    )

    completed_assignments = assignment_nodes(main_node, "completed_epoch")
    completed_initializations = []
    for node in completed_assignments:
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        if (
            isinstance(value, ast.Constant)
            and isinstance(value.value, int)
            and int(value.value) == 0
        ):
            completed_initializations.append(node)
    require(
        len(completed_initializations) == 1,
        "completed_epoch zero initialization is not unique: "
        f"{[(getattr(node, 'lineno', None), ast.unparse(node)) for node in completed_assignments]}",
    )
    completed_assignment = completed_initializations[0]

    progress_write_calls: list[ast.Expr] = []
    for node in ast.walk(main_node):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if call_name(call.func) != "write_json" or len(call.args) < 2:
            continue
        if isinstance(call.args[0], ast.Name) and call.args[0].id == "progress_path":
            progress_write_calls.append(node)
    require(
        len(progress_write_calls) == 1,
        "write_json(progress_path, progress) is not unique",
    )
    progress_write = progress_write_calls[0]

    checkpoint_save_calls = [
        node
        for node in ast.walk(epoch_loop)
        if isinstance(node, ast.Call)
        and call_name(node.func) == "atomic_torch_save"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == "checkpoint_path"
    ]
    scheduler_step_calls = [
        node
        for node in ast.walk(epoch_loop)
        if isinstance(node, ast.Call)
        and call_name(node.func) == "scheduler.step"
    ]
    require(len(checkpoint_save_calls) == 1, "best checkpoint save is not unique")
    require(len(scheduler_step_calls) == 1, "scheduler.step is not unique")
    require(
        int(checkpoint_save_calls[0].lineno) < int(scheduler_step_calls[0].lineno),
        "checkpoint is no longer saved before scheduler.step; resume reconstruction changed",
    )

    setup = r'''
    # F7-M2 shutdown-resume state reconstruction.
    _F7_RESUME_SOURCE_CHECKPOINT = Path(
        os.environ["F7_M2_SOURCE_CHECKPOINT"]
    ).expanduser().resolve()
    _F7_RESUME_SOURCE_PROGRESS = Path(
        os.environ["F7_M2_SOURCE_PROGRESS"]
    ).expanduser().resolve()
    _F7_RESUME_SOURCE_HISTORY = Path(
        os.environ["F7_M2_SOURCE_HISTORY"]
    ).expanduser().resolve()
    _F7_RESUME_LAST_CHECKPOINT = Path(
        os.environ["F7_M2_LAST_CHECKPOINT"]
    ).expanduser().resolve()
    _F7_RESUME_CERTIFICATION = Path(
        os.environ["F7_M2_RUNTIME_CERTIFICATION"]
    ).expanduser().resolve()

    for _f7_path in (
        _F7_RESUME_SOURCE_CHECKPOINT,
        _F7_RESUME_SOURCE_PROGRESS,
        _F7_RESUME_SOURCE_HISTORY,
    ):
        if not _f7_path.is_file():
            raise RuntimeError(
                f"F7-M2 resume prerequisite missing: {_f7_path}"
            )

    _f7_progress = load_json(_F7_RESUME_SOURCE_PROGRESS)
    if int(_f7_progress["completed_epoch"]) != 17:
        raise RuntimeError("F7-M2 expected interrupted progress at epoch 17")
    if int(_f7_progress["best_epoch"]) != 16:
        raise RuntimeError("F7-M2 expected best epoch 16")
    if int(_f7_progress["early_stop_patience_counter"]) != 1:
        raise RuntimeError("F7-M2 expected epoch-17 patience 1")
    if _f7_progress.get("test_directory_enumerated") is not False:
        raise RuntimeError("F7-M2 source progress indicates test enumeration")
    if _f7_progress.get("test_tensor_contents_accessed") is not False:
        raise RuntimeError("F7-M2 source progress indicates sealed-test access")

    try:
        _f7_checkpoint = torch.load(
            _F7_RESUME_SOURCE_CHECKPOINT,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        _f7_checkpoint = torch.load(
            _F7_RESUME_SOURCE_CHECKPOINT,
            map_location=device,
        )

    _f7_required_checkpoint_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "validation_metrics",
        "initial_state_sha256",
        "test_tensor_contents_accessed",
    }
    _f7_missing_checkpoint_keys = sorted(
        _f7_required_checkpoint_keys - set(_f7_checkpoint)
    )
    if _f7_missing_checkpoint_keys:
        raise RuntimeError(
            "F7-M2 checkpoint keys missing: "
            f"{_f7_missing_checkpoint_keys}"
        )
    if int(_f7_checkpoint["epoch"]) != 16:
        raise RuntimeError("F7-M2 source checkpoint is not epoch 16")
    if _f7_checkpoint["test_tensor_contents_accessed"] is not False:
        raise RuntimeError("F7-M2 checkpoint indicates sealed-test access")
    if _f7_checkpoint["initial_state_sha256"] != initial_state_sha256:
        raise RuntimeError("F7-M2 fresh-model identity does not match checkpoint")

    _f7_best_metrics = _f7_progress["best_validation_metrics"]
    if abs(
        float(_f7_best_metrics["selection_score"])
        - 0.7406281414244834
    ) > 1e-15:
        raise RuntimeError("F7-M2 best selection score changed")
    if json.dumps(
        _f7_checkpoint["validation_metrics"],
        sort_keys=True,
        separators=(",", ":"),
    ) != json.dumps(
        _f7_best_metrics,
        sort_keys=True,
        separators=(",", ":"),
    ):
        raise RuntimeError(
            "F7-M2 checkpoint validation metrics differ from progress"
        )

    model.load_state_dict(
        _f7_checkpoint["model_state_dict"], strict=True
    )
    optimizer.load_state_dict(
        _f7_checkpoint["optimizer_state_dict"]
    )
    scheduler.load_state_dict(
        _f7_checkpoint["scheduler_state_dict"]
    )

    # The frozen trainer saves a best checkpoint before scheduler.step.
    # Replaying the epoch-16 scheduler observation reconstructs the exact
    # post-epoch-16 scheduler state used to begin epoch 17.
    scheduler.step(float(_f7_best_metrics["selection_score"]))

    _f7_stochastic_modules = [
        f"{name}:{module.__class__.__name__}"
        for name, module in model.named_modules()
        if isinstance(
            module,
            (
                torch.nn.Dropout,
                torch.nn.Dropout1d,
                torch.nn.Dropout2d,
                torch.nn.Dropout3d,
                torch.nn.AlphaDropout,
                torch.nn.FeatureAlphaDropout,
            ),
        )
    ]
    if _f7_stochastic_modules:
        raise RuntimeError(
            "F7-M2 refuses best-checkpoint resume because stochastic "
            f"modules are present without saved RNG state: {_f7_stochastic_modules}"
        )

    with _F7_RESUME_SOURCE_HISTORY.open(
        "r", encoding="utf-8", newline=""
    ) as _f7_handle:
        _f7_all_history_rows = list(csv.DictReader(_f7_handle))
    _f7_history_epochs = [
        int(row["epoch"]) for row in _f7_all_history_rows
    ]
    if _f7_history_epochs != list(range(1, 18)):
        raise RuntimeError(
            "F7-M2 interrupted history must contain epochs 1..17 exactly; "
            f"found {_f7_history_epochs}"
        )
    history_rows = [
        row
        for row in _f7_all_history_rows
        if int(row["epoch"]) <= 16
    ]
    _f7_prior_elapsed_seconds = sum(
        float(row["elapsed_seconds"]) for row in history_rows
    )

    best_rank = checkpoint_rank(_f7_best_metrics, 16)
    best_epoch = 16
    best_validation_metrics = _f7_best_metrics
    best_early_stop_score = float(
        _f7_best_metrics["selection_score"]
    )
    early_stop_patience_counter = 0
    completed_epoch = 16
    start_time = time.time() - _f7_prior_elapsed_seconds

    if checkpoint_path.exists():
        raise RuntimeError(
            f"F7-M2 clean resume checkpoint path already exists: {checkpoint_path}"
        )
    checkpoint_path.write_bytes(
        _F7_RESUME_SOURCE_CHECKPOINT.read_bytes()
    )
    best_checkpoint_sha256 = sha256_file(checkpoint_path)
    if best_checkpoint_sha256 != sha256_file(
        _F7_RESUME_SOURCE_CHECKPOINT
    ):
        raise RuntimeError("F7-M2 copied checkpoint hash mismatch")

    _f7_runtime_certification = {
        "stage": "V5_P3_F7_M2_EPOCH16_SHUTDOWN_RESUME",
        "status": "RESUME_STATE_RECONSTRUCTED",
        "source_checkpoint": str(_F7_RESUME_SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": sha256_file(
            _F7_RESUME_SOURCE_CHECKPOINT
        ),
        "source_checkpoint_epoch": 16,
        "interrupted_completed_epoch": 17,
        "resume_start_epoch": 17,
        "epoch_17_replayed": True,
        "model_state_loaded": True,
        "optimizer_state_loaded": True,
        "scheduler_state_loaded": True,
        "epoch_16_scheduler_observation_replayed": True,
        "best_rank_reconstructed": list(best_rank),
        "best_early_stop_score": best_early_stop_score,
        "early_stop_patience_at_resume_boundary": 0,
        "history_rows_retained": len(history_rows),
        "history_retained_through_epoch": 16,
        "stochastic_modules": _f7_stochastic_modules,
        "rng_state_available_in_source_checkpoint": False,
        "bitwise_uninterrupted_equivalence_claimed": False,
        "deterministic_epoch_shuffle_preserved": True,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
    }
    write_json(_F7_RESUME_CERTIFICATION, _f7_runtime_certification)

    print("F7_M2_RESUME_STATE_RECONSTRUCTED", flush=True)
    print("resume_source_epoch=16", flush=True)
    print("resume_start_epoch=17", flush=True)
    print("epoch_17_replayed=true", flush=True)
    print("model_optimizer_scheduler_loaded=true", flush=True)
    print("scheduler_epoch16_observation_replayed=true", flush=True)
    print("stochastic_module_count=0", flush=True)
    print("bitwise_uninterrupted_equivalence_claimed=false", flush=True)
'''

    last_checkpoint = r'''
        _f7_last_checkpoint_payload = {
            "stage": "V5_P3_F7_M2_EPOCH_BOUNDARY_LAST_CHECKPOINT",
            "seed": seed,
            "epoch": completed_epoch,
            "next_epoch": completed_epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_rank": best_rank,
            "best_epoch": best_epoch,
            "best_validation_metrics": best_validation_metrics,
            "best_checkpoint_sha256": best_checkpoint_sha256,
            "best_early_stop_score": best_early_stop_score,
            "early_stop_patience_counter": early_stop_patience_counter,
            "history_rows": history_rows,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "initial_state_sha256": initial_state_sha256,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
        atomic_torch_save(
            _f7_last_checkpoint_payload,
            _F7_RESUME_LAST_CHECKPOINT,
        )
'''

    replacements: list[tuple[int, int, str]] = []

    iter_start, iter_end = absolute_span(epoch_loop.iter, offsets)
    replacements.append((iter_start, iter_end, "range(17, 101)"))

    _, completed_end = absolute_span(completed_assignment, offsets)
    completed_indent = indentation_at(text, completed_end)
    setup_indented = textwrap.indent(textwrap.dedent(setup), completed_indent)
    replacements.append(
        (
            completed_end,
            completed_end,
            "\n" + setup_indented.rstrip("\n"),
        )
    )

    _, progress_end = absolute_span(progress_write, offsets)
    progress_indent = indentation_at(text, progress_end)
    last_indented = textwrap.indent(textwrap.dedent(last_checkpoint), progress_indent)
    replacements.append(
        (
            progress_end,
            progress_end,
            "\n" + last_indented.rstrip("\n"),
        )
    )

    output = apply_replacements(text, replacements)
    output = (
        "# AUTO-GENERATED BY V5-P3 F7-M2 SHUTDOWN RESUME.\n"
        "# Resumes ablate_buffer_pressure from durable best epoch 16, "
        "replays epoch 17, and writes an exact last checkpoint each epoch.\n"
        + output
    )
    output_tree = ast.parse(output)
    compile(output, str(resume_trainer), "exec")

    output_main = next(
        node
        for node in output_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    output_epoch_loops = [
        node
        for node in ast.walk(output_main)
        if isinstance(node, ast.For)
        and any(
            isinstance(child, ast.Call)
            and call_name(child.func).rsplit(".", 1)[-1] == "train_one_epoch"
            for child in ast.walk(node)
        )
    ]
    require(len(output_epoch_loops) == 1, "patched epoch loop is not unique")
    require(
        ast.unparse(output_epoch_loops[0].iter) == "range(17, 101)",
        "patched resume epoch range changed",
    )
    require(
        output.count("F7_M2_RESUME_STATE_RECONSTRUCTED") == 1,
        "resume setup sentinel count changed",
    )
    require(
        output.count("V5_P3_F7_M2_EPOCH_BOUNDARY_LAST_CHECKPOINT") == 1,
        "last-checkpoint injection count changed",
    )

    atomic_text(resume_trainer, output, mode=0o555)
    return {
        "source_trainer": str(source_trainer),
        "source_trainer_sha256": sha256_file(source_trainer),
        "resume_trainer": str(resume_trainer),
        "resume_trainer_sha256": sha256_file(resume_trainer),
        "source_epoch_range": "range(1, 101)",
        "resume_epoch_range": "range(17, 101)",
        "checkpoint_saved_before_scheduler_step_verified": True,
        "epoch_16_scheduler_observation_replay_required": True,
        "last_checkpoint_injected_after_progress_write": True,
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()

    require(repo.is_dir(), f"repository missing: {repo}")
    require(data_link.is_symlink(), f"dataset symlink missing: {data_link}")
    require(data_link.resolve().is_dir(), "dataset target missing")
    require(installed_script.is_file(), "installed recovery script missing")

    e2_lock_path = output_dir / (
        "V5_P3_F7_E2_GENERATED_TRAINER_STATIC_AND_RUNTIME_PREFLIGHT_LOCK.json"
    )
    e4_lock_path = output_dir / (
        "V5_P3_F7_E4_ONE_EPOCH_CONTROL_SMOKE_GATE_LOCK.json"
    )
    e2_lock = load_json(e2_lock_path)
    e4_lock = load_json(e4_lock_path)
    require(e2_lock.get("status") == "PASS", "E2 lock is not PASS")
    require(e4_lock.get("status") == "PASS", "E4 lock is not PASS")
    require(
        e4_lock.get("primary_seed107_six_run_matrix_release_authorized") is True,
        "E4 did not release the seed-107 matrix",
    )
    require(e2_lock.get("sealed_test_tensors_loaded") is False, "E2 test access")
    require(e4_lock.get("sealed_test_access") is False, "E4 test access")

    source_trainer = repo / (
        "scripts/v5/p3/experiments/"
        "run_v5_p3_f7_reconstructed_group_ablation_trainer_e2_r6_r2.py"
    )
    launcher = repo / (
        "scripts/v5/p3/experiments/"
        "run_v5_p3_f7_primary_seed107_matrix_e2_r6_r2.sh"
    )
    adapter = repo / "src/models/v5_p3_f7_r5_runtime_adapter.py"
    for path in (source_trainer, launcher, adapter):
        require(path.is_file(), f"required artifact missing: {path}")
    require(sha256_file(source_trainer) == EXPECTED_TRAINER_SHA256, "trainer hash changed")
    require(sha256_file(launcher) == EXPECTED_LAUNCHER_SHA256, "launcher hash changed")
    require(sha256_file(adapter) == EXPECTED_ADAPTER_SHA256, "adapter hash changed")
    require(
        e2_lock.get("generated_trainer_sha256") == EXPECTED_TRAINER_SHA256,
        "E2 trainer hash changed",
    )
    require(
        e2_lock.get("matrix_launcher_sha256") == EXPECTED_LAUNCHER_SHA256,
        "E2 launcher hash changed",
    )

    m1_pid_path = output_dir / "V5_P3_F7_PRIMARY_SEED107_MATRIX_M1_R1.pid"
    if m1_pid_path.is_file():
        raw_pid = m1_pid_path.read_text(encoding="utf-8").strip()
        require(raw_pid.isdigit(), "invalid M1 PID file")
        pid = int(raw_pid)
        if process_exists(pid):
            command = process_command(pid)
            require(
                str(launcher) not in command,
                f"matrix launcher is still active: pid={pid}; command={command}",
            )

    run_root = output_dir / "f7_runs/seed107"
    canonical_run = run_root / EXPECTED_LABEL
    canonical_model = canonical_run / "model"
    canonical_report = canonical_run / "report"
    canonical_console = canonical_run / "console.log"
    canonical_complete = canonical_report / "F7_RUN_COMPLETE"
    require(canonical_run.is_dir(), "interrupted canonical run missing")
    require(canonical_model.is_dir(), "interrupted model directory missing")
    require(canonical_report.is_dir(), "interrupted report directory missing")
    require(canonical_console.is_file(), "interrupted console missing")
    require(not canonical_complete.exists(), "canonical run is already complete")

    source_checkpoint = canonical_model / "v5_p2_b2_seed_107_best.pt"
    source_progress = canonical_report / "V5_P2_B2_SEED_107_PROGRESS.json"
    source_history = canonical_report / "V5_P2_B2_SEED_107_HISTORY.csv"
    for path in (source_checkpoint, source_progress, source_history):
        require(path.is_file(), f"interrupted artifact missing: {path}")
    require(
        sha256_file(source_checkpoint) == EXPECTED_SOURCE_CHECKPOINT_SHA256,
        "epoch-16 source checkpoint hash changed",
    )

    progress = load_json(source_progress)
    require(progress.get("status") == "RUNNING", "source progress is not RUNNING")
    require(int(progress["seed"]) == EXPECTED_SEED, "source seed changed")
    require(int(progress["completed_epoch"]) == EXPECTED_COMPLETED_EPOCH, "source completed epoch changed")
    require(int(progress["best_epoch"]) == EXPECTED_RESUME_CHECKPOINT_EPOCH, "source best epoch changed")
    require(int(progress["early_stop_patience_counter"]) == 1, "source patience changed")
    require(
        abs(float(progress["best_validation_metrics"]["selection_score"]) - EXPECTED_BEST_SCORE) <= 1e-15,
        "source best score changed",
    )
    require(progress.get("test_directory_enumerated") is False, "source test enumeration")
    require(progress.get("test_tensor_contents_accessed") is False, "source test access")
    require(
        progress.get("best_checkpoint_sha256") == EXPECTED_SOURCE_CHECKPOINT_SHA256,
        "progress checkpoint hash changed",
    )

    history = read_history(source_history)
    history_epochs = [int(row["epoch"]) for row in history]
    require(history_epochs == list(range(1, 18)), f"unexpected source history epochs: {history_epochs}")
    require(
        int(history[-1]["early_stop_patience_counter"]) == 1,
        "epoch-17 history patience changed",
    )

    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict), "source checkpoint is not a dictionary")
    required_checkpoint_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "validation_metrics",
        "initial_state_sha256",
        "test_tensor_contents_accessed",
    }
    require(
        required_checkpoint_keys.issubset(checkpoint),
        f"source checkpoint keys missing: {sorted(required_checkpoint_keys - set(checkpoint))}",
    )
    require(int(checkpoint["epoch"]) == 16, "source checkpoint epoch changed")
    require(checkpoint["test_tensor_contents_accessed"] is False, "checkpoint test access")
    require(
        json_equivalent(checkpoint["validation_metrics"], progress["best_validation_metrics"]),
        "checkpoint and progress best metrics differ",
    )

    e2_report_path = Path(e2_lock["source_report"]).resolve()
    e2_report = load_json(e2_report_path)
    runtime = e2_report["finding"]["runtime"]
    require(runtime["parameter_count"] == EXPECTED_PARAMETER_COUNT, "parameter count changed")
    require(runtime["train_items"] == EXPECTED_TRAIN_ITEMS, "train items changed")
    require(runtime["validation_items"] == EXPECTED_VALIDATION_ITEMS, "validation items changed")
    require(runtime["train_batches"] == EXPECTED_TRAIN_BATCHES, "train batches changed")
    require(runtime["validation_batches"] == EXPECTED_VALIDATION_BATCHES, "validation batches changed")
    prior_command = list(runtime["instrumented_command"])
    b1_dir = Path(extract_flag(prior_command, "--b1-dir")).resolve()
    b0_r3_dir = Path(extract_flag(prior_command, "--b0-r3-dir")).resolve()
    legacy_loader = Path(extract_flag(prior_command, "--loader-path")).resolve()
    legacy_model = Path(extract_flag(prior_command, "--model-path")).resolve()
    for path in (b1_dir, b0_r3_dir, legacy_loader, legacy_model):
        require(path.exists(), f"compatibility path missing: {path}")

    stage_report = output_dir / f"{STAGE}_REPORT.json"
    stage_lock = output_dir / f"{STAGE}_LOCK.json"
    stage_complete = output_dir / f"{STAGE}_COMPLETE"
    require(
        not stage_report.exists() and not stage_lock.exists() and not stage_complete.exists(),
        "M2 completion artifacts already exist",
    )

    recovery_root = canonical_run / "recovery/M2_RESUME_FROM_EPOCH16_AFTER_SHUTDOWN"
    require(not recovery_root.exists(), f"M2 recovery root already exists: {recovery_root}")
    recovery_root.mkdir(parents=True)
    resume_run = recovery_root / "run"
    resume_run.mkdir()
    resume_model = resume_run / "model"
    resume_report = resume_run / "report"
    resume_console = resume_run / "console.log"
    last_checkpoint = recovery_root / "F7_M2_LAST_EPOCH_BOUNDARY.pt"
    runtime_certification = recovery_root / "F7_M2_RUNTIME_RESUME_CERTIFICATION.json"
    resume_trainer = repo / (
        "scripts/v5/p3/experiments/"
        "run_v5_p3_f7_m2_resume_ablate_buffer_pressure_from_epoch16.py"
    )
    require(not resume_trainer.exists(), f"resume trainer already exists: {resume_trainer}")

    instrumentation = create_resume_trainer(source_trainer, resume_trainer)

    environment = os.environ.copy()
    environment.update({
        "F7_REPO": str(repo),
        "F7_DATA_LINK": str(data_link),
        "F7_SEED": str(EXPECTED_SEED),
        "F7_RUN_DIR": str(resume_run),
        "F7_ABLATION_LABEL": EXPECTED_LABEL,
        "F7_M2_SOURCE_CHECKPOINT": str(source_checkpoint),
        "F7_M2_SOURCE_PROGRESS": str(source_progress),
        "F7_M2_SOURCE_HISTORY": str(source_history),
        "F7_M2_LAST_CHECKPOINT": str(last_checkpoint),
        "F7_M2_RUNTIME_CERTIFICATION": str(runtime_certification),
        "PYTHONUNBUFFERED": "1",
    })
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(repo) + (os.pathsep + old_pythonpath if old_pythonpath else "")

    command = [
        str(repo / ".venv/bin/python"),
        "-u",
        str(resume_trainer),
        "--root", str(repo),
        "--b1-dir", str(b1_dir),
        "--b0-r3-dir", str(b0_r3_dir),
        "--loader-path", str(legacy_loader),
        "--model-path", str(legacy_model),
        "--model-dir", str(resume_model),
        "--report-dir", str(resume_report),
        "--seed", str(EXPECTED_SEED),
    ]

    launch_manifest = {
        "stage": STAGE,
        "status": "RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": CLASSIFICATION,
        "failure_cause": "laptop_shutdown_after_completed_epoch_17",
        "source_checkpoint_epoch": 16,
        "interrupted_completed_epoch": 17,
        "resume_start_epoch": 17,
        "epoch_17_replayed": True,
        "command": command,
        "environment_keys": sorted(
            key for key in environment if key.startswith("F7_")
        ),
        "instrumentation": instrumentation,
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_progress_sha256": sha256_file(source_progress),
        "source_history_sha256": sha256_file(source_history),
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
    }
    launch_manifest_path = recovery_root / "F7_M2_RESUME_LAUNCH_MANIFEST.json"
    atomic_json(launch_manifest_path, launch_manifest)

    started = time.time()
    with resume_console.open("xb") as console_handle:
        process = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=console_handle,
            stderr=subprocess.STDOUT,
            timeout=12 * 3600,
        )
    wall_seconds = time.time() - started

    require(
        process.returncode == 0,
        "M2 resumed trainer failed; "
        f"returncode={process.returncode}; console={resume_console}",
    )
    require(runtime_certification.is_file(), "runtime resume certification missing")
    require(last_checkpoint.is_file(), "epoch-boundary last checkpoint missing")

    resume_report_path = resume_report / "V5_P3_F7_SEED_107_TRAINING_REPORT.json"
    resume_lock_path = resume_report / "V5_P3_F7_SEED_107_LOCK.json"
    resume_history_path = resume_report / "V5_P2_B2_SEED_107_HISTORY.csv"
    resume_progress_path = resume_report / "V5_P2_B2_SEED_107_PROGRESS.json"
    resume_best_checkpoint = resume_model / "v5_p2_b2_seed_107_best.pt"
    for path in (
        resume_report_path,
        resume_lock_path,
        resume_history_path,
        resume_progress_path,
        resume_best_checkpoint,
    ):
        require(path.is_file(), f"resumed final artifact missing: {path}")

    completion_markers = sorted(resume_report.glob("*_COMPLETE"))
    require(completion_markers, "resumed trainer completion marker missing")

    final_report = load_json(resume_report_path)
    final_lock = load_json(resume_lock_path)
    final_progress = load_json(resume_progress_path)
    require(final_report.get("status") == "COMPLETE", "resumed report is not COMPLETE")
    require(str(final_lock.get("status", "")).endswith("_COMPLETE"), "resumed lock is not complete")
    require(final_progress.get("test_directory_enumerated") is False, "resumed progress test enumeration")
    require(final_progress.get("test_tensor_contents_accessed") is False, "resumed progress test access")
    require(final_report["security_boundary"]["test_tensor_contents_accessed"] is False, "resumed report test access")
    final_completed_epoch = int(final_report["training"]["completed_epoch"])
    require(final_completed_epoch >= 17, "resumed run did not complete epoch 17")
    require(int(final_lock["completed_epoch"]) == final_completed_epoch, "final completed epoch mismatch")

    final_history = read_history(resume_history_path)
    final_history_epochs = [int(row["epoch"]) for row in final_history]
    require(
        final_history_epochs == list(range(1, final_completed_epoch + 1)),
        f"resumed history is not contiguous: {final_history_epochs}",
    )
    require(len([epoch for epoch in final_history_epochs if epoch == 17]) == 1, "epoch 17 not represented exactly once")

    final_checkpoint = torch.load(resume_best_checkpoint, map_location="cpu", weights_only=False)
    require(isinstance(final_checkpoint, dict), "final best checkpoint is not a dictionary")
    require(final_checkpoint.get("test_tensor_contents_accessed") is False, "final best checkpoint test access")
    require(
        sha256_file(resume_best_checkpoint) == final_lock["checkpoint_sha256"],
        "final best checkpoint hash differs from lock",
    )

    final_last = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    require(int(final_last["epoch"]) == final_completed_epoch, "last checkpoint epoch mismatch")
    require(int(final_last["next_epoch"]) == final_completed_epoch + 1, "last checkpoint next epoch mismatch")
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "early_stop_patience_counter",
        "best_rank",
        "history_rows",
    ):
        require(key in final_last, f"last checkpoint key missing: {key}")
    require(final_last["test_tensor_contents_accessed"] is False, "last checkpoint test access")

    # Prepare canonical output copies before moving the interrupted snapshot.
    promote_model = canonical_run / "model.m2_promote_tmp"
    promote_report = canonical_run / "report.m2_promote_tmp"
    require(not promote_model.exists() and not promote_report.exists(), "promotion temp paths already exist")
    shutil.copytree(resume_model, promote_model)
    shutil.copytree(resume_report, promote_report)

    snapshot = canonical_run / "interrupted_epoch17_shutdown_snapshot"
    require(not snapshot.exists(), f"snapshot already exists: {snapshot}")
    snapshot.mkdir()

    for name in (
        "model",
        "report",
        "model_capture",
        "F7_MODEL_INITIALIZATION_CERTIFICATION.json",
        "console.log",
    ):
        source = canonical_run / name
        if source.exists():
            os.replace(source, snapshot / name)

    os.replace(promote_model, canonical_model)
    os.replace(promote_report, canonical_report)

    for name in (
        "model_capture",
        "F7_MODEL_INITIALIZATION_CERTIFICATION.json",
        "console.log",
    ):
        source = resume_run / name
        destination = canonical_run / name
        if source.is_dir():
            shutil.copytree(source, destination)
        elif source.is_file():
            shutil.copy2(source, destination)

    canonicalization = {
        "stage": STAGE,
        "status": "COMPLETE",
        "classification": CLASSIFICATION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "failure_cause": "laptop_shutdown_after_completed_epoch_17",
        "source_checkpoint_epoch": 16,
        "resume_start_epoch": 17,
        "epoch_17_replayed": True,
        "bitwise_uninterrupted_equivalence_claimed": False,
        "scientific_interpretation": (
            "Interrupted-and-resumed seed-107 same-width ablation. "
            "Model, optimizer and scheduler were restored from the durable "
            "epoch-16 best checkpoint; the epoch-16 scheduler observation "
            "was reconstructed; epoch 17 was replayed; all subsequent epochs "
            "used the frozen trainer and deterministic epoch shuffle."
        ),
        "interrupted_snapshot": str(snapshot),
        "resume_run": str(resume_run),
        "canonical_model": str(canonical_model),
        "canonical_report": str(canonical_report),
        "final_completed_epoch": final_completed_epoch,
        "final_best_epoch": int(final_lock["best_epoch"]),
        "final_best_selection_score": float(final_lock["best_validation_selection_score"]),
        "final_best_checkpoint": str(canonical_model / "v5_p2_b2_seed_107_best.pt"),
        "final_best_checkpoint_sha256": sha256_file(canonical_model / "v5_p2_b2_seed_107_best.pt"),
        "last_epoch_boundary_checkpoint": str(last_checkpoint),
        "last_epoch_boundary_checkpoint_sha256": sha256_file(last_checkpoint),
        "resume_wall_seconds": wall_seconds,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "sealed_test_access": False,
        "F8_multi_seed_authorized": False,
        "feature_removal_authorized": False,
    }
    canonicalization_path = canonical_report / "F7_M2_RESUME_CANONICALIZATION.json"
    atomic_json(canonicalization_path, canonicalization)
    atomic_text(canonical_complete, f"{STAGE}_COMPLETE\n")

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "finding": {
            "shutdown_interruption_confirmed_by_timeline": True,
            "source_completed_epoch": 17,
            "source_best_epoch": 16,
            "source_best_score": EXPECTED_BEST_SCORE,
            "resume_start_epoch": 17,
            "epoch_17_replayed": True,
            "instrumentation": instrumentation,
            "resume_wall_seconds": wall_seconds,
            "final_completed_epoch": final_completed_epoch,
            "final_best_epoch": int(final_lock["best_epoch"]),
            "final_best_selection_score": float(final_lock["best_validation_selection_score"]),
            "last_checkpoint_has_rng_state": True,
            "canonicalization": canonicalization,
        },
        "governance": {
            "training_and_validation_accessed": True,
            "sealed_test_access": False,
            "scientific_checkpoint_saved": True,
            "scientific_result_classification": "interrupted-and-resumed",
            "bitwise_uninterrupted_equivalence_claimed": False,
            "primary_matrix_execution_relaunch_authorized": True,
            "F8_multi_seed_authorized": False,
            "feature_removal_authorized": False,
            "compact_interface_equivalence_authorized": False,
            "hardware_reduction_claim_authorized": False,
        },
        "artifacts": {
            "source_checkpoint": str(source_checkpoint),
            "source_progress": str(source_progress),
            "source_history": str(source_history),
            "resume_trainer": str(resume_trainer),
            "resume_console": str(resume_console),
            "resume_run": str(resume_run),
            "interrupted_snapshot": str(snapshot),
            "canonical_run": str(canonical_run),
            "canonicalization": str(canonicalization_path),
            "F7_RUN_COMPLETE": str(canonical_complete),
            "last_epoch_boundary_checkpoint": str(last_checkpoint),
        },
        "provenance": {
            "E2_lock_sha256": sha256_file(e2_lock_path),
            "E4_lock_sha256": sha256_file(e4_lock_path),
            "source_trainer_sha256": sha256_file(source_trainer),
            "resume_trainer_sha256": sha256_file(resume_trainer),
            "source_checkpoint_sha256": EXPECTED_SOURCE_CHECKPOINT_SHA256,
            "final_checkpoint_sha256": sha256_file(canonical_model / "v5_p2_b2_seed_107_best.pt"),
            "last_checkpoint_sha256": sha256_file(last_checkpoint),
            "canonicalization_sha256": sha256_file(canonicalization_path),
            "installed_script_sha256": sha256_file(installed_script),
        },
        "next_stage": "V5_P3_F7_M2_REMAINING_FIVE_RUN_MATRIX_RELAUNCH_PREFLIGHT",
    }
    atomic_json(stage_report, report)
    atomic_json(
        stage_lock,
        {
            "stage": STAGE,
            "status": "COMPLETE",
            "report": str(stage_report),
            "report_sha256": sha256_file(stage_report),
            "canonical_run": str(canonical_run),
            "F7_RUN_COMPLETE": str(canonical_complete),
            "final_completed_epoch": final_completed_epoch,
            "final_best_epoch": int(final_lock["best_epoch"]),
            "final_best_selection_score": float(final_lock["best_validation_selection_score"]),
            "final_checkpoint_sha256": sha256_file(canonical_model / "v5_p2_b2_seed_107_best.pt"),
            "last_checkpoint_sha256": sha256_file(last_checkpoint),
            "sealed_test_access": False,
            "primary_matrix_execution_relaunch_authorized": True,
            "F8_multi_seed_authorized": False,
            "feature_removal_authorized": False,
            "next_stage": "V5_P3_F7_M2_REMAINING_FIVE_RUN_MATRIX_RELAUNCH_PREFLIGHT",
        },
    )
    atomic_text(stage_complete, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=COMPLETE")
    print(f"classification={CLASSIFICATION}")
    print("failure_cause=laptop_shutdown_after_completed_epoch_17")
    print("source_checkpoint_epoch=16")
    print("resume_start_epoch=17")
    print("epoch_17_replayed=true")
    print("bitwise_uninterrupted_equivalence_claimed=false")
    print(f"final_completed_epoch={final_completed_epoch}")
    print(f"final_best_epoch={int(final_lock['best_epoch'])}")
    print(f"final_best_selection_score={float(final_lock['best_validation_selection_score'])}")
    print(f"resume_wall_seconds={wall_seconds}")
    print("last_checkpoint_has_rng_state=true")
    print("canonical_run_promoted=true")
    print("F7_RUN_COMPLETE=true")
    print("sealed_test_access=false")
    print("primary_matrix_execution_relaunch_authorized=true")
    print("F8_multi_seed_authorized=false")
    print("feature_removal_authorized=false")
    print("next_stage=V5_P3_F7_M2_REMAINING_FIVE_RUN_MATRIX_RELAUNCH_PREFLIGHT")
    print(f"report={stage_report}")
    print(f"lock={stage_lock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
