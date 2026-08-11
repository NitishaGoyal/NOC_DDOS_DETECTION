from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


STAGE = "V5_P3_F7_E4_ONE_EPOCH_CONTROL_SMOKE_GATE"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "ENGINEERING"

EXPECTED_SEED = 107
EXPECTED_PARAMETER_COUNT = 60_553
EXPECTED_TRAIN_ITEMS = 110_855
EXPECTED_VALIDATION_ITEMS = 13_863
EXPECTED_BATCH_SIZE = 256
EXPECTED_TRAIN_BATCHES = 434
EXPECTED_VALIDATION_BATCHES = 55


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


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
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
        f"incomplete source span: {type(node).__name__}",
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


def indentation_at(text: str, position: int) -> str:
    line_start = text.rfind("\n", 0, position) + 1
    prefix = text[line_start:position]
    match = re.match(r"[ \t]*", prefix)
    require(match is not None, "could not derive indentation")
    return match.group(0)


def assignment_nodes(tree: ast.Module, target_name: str) -> list[ast.AST]:
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


def render_wrapper_call(
    wrapper_name: str,
    object_name: str,
    original_call: ast.Call,
) -> str:
    pieces = [object_name]
    pieces.extend(ast.unparse(arg) for arg in original_call.args)
    for keyword in original_call.keywords:
        if keyword.arg is None:
            pieces.append(f"**{ast.unparse(keyword.value)}")
        else:
            pieces.append(f"{keyword.arg}={ast.unparse(keyword.value)}")
    return f"{wrapper_name}({', '.join(pieces)})"


def create_one_epoch_smoke_trainer(
    trainer_path: Path,
    smoke_path: Path,
) -> dict[str, Any]:
    text = trainer_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text)
    offsets = line_offsets(text)

    main_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]
    require(len(main_nodes) == 1, "module-level main is not unique")
    main_node = main_nodes[0]

    epoch_loops: list[ast.For] = []
    for node in ast.walk(main_node):
        if not isinstance(node, ast.For):
            continue
        if any(
            isinstance(child, ast.Call)
            and call_name(child.func).rsplit(".", 1)[-1]
            == "train_one_epoch"
            for child in ast.walk(node)
        ):
            epoch_loops.append(node)

    require(
        len(epoch_loops) == 1,
        "epoch loop containing train_one_epoch is not unique: "
        f"{[(node.lineno, ast.unparse(node.target), ast.unparse(node.iter)) for node in epoch_loops]}",
    )
    epoch_loop = epoch_loops[0]
    require(epoch_loop.body, "epoch loop is empty")
    epoch_target = ast.unparse(epoch_loop.target)

    optimizer_steps = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and call_name(node.func) == "optimizer.step"
    ]
    scheduler_steps = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and call_name(node.func) == "scheduler.step"
    ]
    require(len(optimizer_steps) == 1, "optimizer.step is not unique")
    require(len(scheduler_steps) == 1, "scheduler.step is not unique")

    train_loader_assignments = assignment_nodes(tree, "train_loader")
    validation_loader_assignments = assignment_nodes(
        tree,
        "validation_loader",
    )
    require(
        len(train_loader_assignments) == 1,
        "train_loader assignment is not unique",
    )
    require(
        len(validation_loader_assignments) == 1,
        "validation_loader assignment is not unique",
    )

    parent_map: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent

    executable_main_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and call_name(node.func) == "main"
        ):
            continue
        parent = parent_map.get(node)
        if (
            isinstance(parent, ast.Call)
            and call_name(parent.func) == "SystemExit"
        ):
            executable_main_calls.append(node)
    require(
        len(executable_main_calls) == 1,
        "SystemExit(main()) call is not unique",
    )

    helper_source = r'''
_F7_E4_STATE = {
    "optimizer_steps": 0,
    "scheduler_steps": 0,
    "epoch_started": [],
    "epoch_completed": [],
    "loader_batches": {"train": 0, "validation": 0},
    "started_unix": None,
}


class _F7E4CountingLoader:
    def __init__(self, loader, role):
        self._loader = loader
        self._role = str(role)

    def __iter__(self):
        for batch in self._loader:
            _F7_E4_STATE["loader_batches"][self._role] += 1
            yield batch

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        return getattr(self._loader, name)


def _f7_e4_optimizer_step(optimizer, *args, **kwargs):
    result = optimizer.step(*args, **kwargs)
    _F7_E4_STATE["optimizer_steps"] += 1
    return result


def _f7_e4_scheduler_step(scheduler, *args, **kwargs):
    result = scheduler.step(*args, **kwargs)
    _F7_E4_STATE["scheduler_steps"] += 1
    return result


def _f7_e4_epoch_started(epoch):
    _F7_E4_STATE["epoch_started"].append(int(epoch))


def _f7_e4_epoch_completed(epoch):
    _F7_E4_STATE["epoch_completed"].append(int(epoch))


def _f7_e4_write_result(completed, return_value=None, failure=None):
    import json as _json
    import os as _os
    import time as _time
    from pathlib import Path as _Path

    payload = {
        "sentinel": (
            "F7_E4_ONE_EPOCH_CONTROL_SMOKE_COMPLETE"
            if completed
            else "F7_E4_ONE_EPOCH_CONTROL_SMOKE_FAILED"
        ),
        "completed": bool(completed),
        "return_value": return_value,
        "failure": failure,
        "optimizer_steps": int(_F7_E4_STATE["optimizer_steps"]),
        "scheduler_steps": int(_F7_E4_STATE["scheduler_steps"]),
        "epoch_started": list(_F7_E4_STATE["epoch_started"]),
        "epoch_completed": list(_F7_E4_STATE["epoch_completed"]),
        "loader_batches": dict(_F7_E4_STATE["loader_batches"]),
        "elapsed_seconds": (
            float(_time.time() - _F7_E4_STATE["started_unix"])
            if _F7_E4_STATE["started_unix"] is not None
            else None
        ),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
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
    }

    path = _Path(_os.environ["F7_E4_SMOKE_JSON"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        _json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _os.replace(temporary, path)


def _f7_e4_run_main(main_function):
    import time as _time

    _F7_E4_STATE["started_unix"] = _time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        return_value = main_function()
    except BaseException as error:
        _f7_e4_write_result(
            completed=False,
            failure=f"{type(error).__name__}: {error}",
        )
        raise
    else:
        _f7_e4_write_result(
            completed=True,
            return_value=return_value,
        )
        print("F7_E4_ONE_EPOCH_CONTROL_SMOKE_COMPLETE", flush=True)
        return return_value
'''

    replacements: list[tuple[int, int, str]] = []
    insertion = insertion_offset_after_future_imports(text, tree, offsets)
    replacements.append((insertion, insertion, helper_source))

    for assignment, variable, role in (
        (train_loader_assignments[0], "train_loader", "train"),
        (
            validation_loader_assignments[0],
            "validation_loader",
            "validation",
        ),
    ):
        _, end = absolute_span(assignment, offsets)
        indent = indentation_at(text, end)
        replacements.append(
            (
                end,
                end,
                "\n"
                + indent
                + f'{variable} = _F7E4CountingLoader({variable}, "{role}")',
            )
        )

    replacements.append(
        (
            *absolute_span(optimizer_steps[0], offsets),
            render_wrapper_call(
                "_f7_e4_optimizer_step",
                "optimizer",
                optimizer_steps[0],
            ),
        )
    )
    replacements.append(
        (
            *absolute_span(scheduler_steps[0], offsets),
            render_wrapper_call(
                "_f7_e4_scheduler_step",
                "scheduler",
                scheduler_steps[0],
            ),
        )
    )

    first_statement = epoch_loop.body[0]
    first_start, _ = absolute_span(first_statement, offsets)
    body_indent = indentation_at(text, first_start)
    replacements.append(
        (
            first_start,
            first_start,
            f"_f7_e4_epoch_started({epoch_target})\n{body_indent}",
        )
    )

    last_statement = epoch_loop.body[-1]
    _, last_end = absolute_span(last_statement, offsets)
    replacements.append(
        (
            last_end,
            last_end,
            "\n"
            + body_indent
            + f"_f7_e4_epoch_completed({epoch_target})"
            + "\n"
            + body_indent
            + "break",
        )
    )

    replacements.append(
        (
            *absolute_span(executable_main_calls[0], offsets),
            "_f7_e4_run_main(main)",
        )
    )

    smoke_text = apply_replacements(text, replacements)
    smoke_tree = ast.parse(smoke_text)
    compile(smoke_text, str(smoke_path), "exec")

    wrapper_optimizer_calls = [
        node
        for node in ast.walk(smoke_tree)
        if isinstance(node, ast.Call)
        and call_name(node.func) == "_f7_e4_optimizer_step"
    ]
    wrapper_scheduler_calls = [
        node
        for node in ast.walk(smoke_tree)
        if isinstance(node, ast.Call)
        and call_name(node.func) == "_f7_e4_scheduler_step"
    ]
    require(
        len(wrapper_optimizer_calls) == 1,
        "optimizer wrapper call is not unique",
    )
    require(
        len(wrapper_scheduler_calls) == 1,
        "scheduler wrapper call is not unique",
    )

    atomic_text(smoke_path, smoke_text, mode=0o555)
    return {
        "source_trainer": str(trainer_path),
        "source_trainer_sha256": sha256_file(trainer_path),
        "smoke_trainer": str(smoke_path),
        "smoke_trainer_sha256": sha256_file(smoke_path),
        "epoch_loop_line": int(epoch_loop.lineno),
        "epoch_target": epoch_target,
        "epoch_iterator": ast.unparse(epoch_loop.iter),
        "one_epoch_break_inserted_after_complete_original_epoch_body": True,
        "train_loader_wrapped": True,
        "validation_loader_wrapped": True,
        "optimizer_step_wrapped": True,
        "scheduler_step_wrapped": True,
        "post_finalization_main_wrapper_added": True,
    }


def tensor_inventory(value: Any) -> dict[str, Any]:
    tensors: list[torch.Tensor] = []

    def visit(node: Any) -> None:
        if isinstance(node, torch.Tensor):
            tensors.append(node)
        elif isinstance(node, dict):
            for child in node.values():
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)
    finite_flags = [
        bool(torch.isfinite(tensor).all().item())
        for tensor in tensors
        if tensor.is_floating_point() or tensor.is_complex()
    ]
    return {
        "tensor_count": len(tensors),
        "total_tensor_elements": sum(int(t.numel()) for t in tensors),
        "all_floating_tensors_finite": all(finite_flags),
    }


def audit_torch_loadable_files(model_dir: Path) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            errors.append({"path": str(path), "error": repr(error)})
            continue
        inventory = tensor_inventory(loaded)
        if inventory["tensor_count"] <= 0:
            errors.append({"path": str(path), "error": "no tensors"})
            continue
        require(
            inventory["all_floating_tensors_finite"] is True,
            f"non-finite checkpoint tensors: {path}",
        )
        audits.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                **inventory,
            }
        )
    require(
        audits,
        "no torch-loadable checkpoint with tensors was finalized; "
        f"load_errors={errors}",
    )
    return audits


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()

    require(repo.is_dir(), f"repository missing: {repo}")
    require(data_link.is_symlink(), f"dataset symlink missing: {data_link}")
    require(installed_script.is_file(), "installed E4 script missing")

    e2_lock_path = output_dir / (
        "V5_P3_F7_E2_GENERATED_TRAINER_STATIC_AND_RUNTIME_PREFLIGHT_LOCK.json"
    )
    e3_lock_path = output_dir / (
        "V5_P3_F7_E3_R1_RUNTIME_ADAPTER_ENVIRONMENT_RECOVERY_LOCK.json"
    )
    e2_lock = load_json(e2_lock_path)
    e3_lock = load_json(e3_lock_path)

    require(e2_lock.get("status") == "PASS", "E2 lock is not PASS")
    require(e3_lock.get("status") == "PASS", "E3 lock is not PASS")
    require(
        e3_lock.get("control_run_smoke_authorized") is True,
        "E3 did not authorize E4",
    )
    require(
        e3_lock.get("full_six_run_matrix_operationally_held") is True,
        "E3 full-matrix hold missing",
    )
    require(
        e3_lock.get("disposable_optimizer_steps_performed") == 15,
        "E3 disposable-step count changed",
    )
    require(
        e3_lock.get("scientific_optimizer_steps_performed") == 0,
        "E3 recorded scientific optimizer steps",
    )
    require(
        e3_lock.get("sealed_test_access") is False,
        "E3 recorded sealed-test access",
    )

    e2_report_path = Path(e2_lock["source_report"]).resolve()
    e2_report = load_json(e2_report_path)
    require(e2_report.get("status") == "PASS", "E2 report is not PASS")

    trainer_path = Path(
        e2_report["artifacts"]["corrected_trainer"]
    ).resolve()
    matrix_launcher = Path(
        e2_report["artifacts"]["corrected_matrix_launcher"]
    ).resolve()
    require(trainer_path.is_file(), "certified trainer missing")
    require(matrix_launcher.is_file(), "corrected matrix launcher missing")
    require(
        sha256_file(trainer_path) == e2_lock["generated_trainer_sha256"],
        "certified trainer SHA changed",
    )
    require(
        sha256_file(matrix_launcher) == e2_lock["matrix_launcher_sha256"],
        "matrix launcher SHA changed",
    )

    runtime = e2_report["finding"]["runtime"]
    require(runtime["parameter_count"] == EXPECTED_PARAMETER_COUNT, "parameter count changed")
    require(runtime["train_items"] == EXPECTED_TRAIN_ITEMS, "train count changed")
    require(runtime["validation_items"] == EXPECTED_VALIDATION_ITEMS, "validation count changed")
    require(runtime["effective_item_batch_size"] == EXPECTED_BATCH_SIZE, "batch size changed")
    require(runtime["train_batches"] == EXPECTED_TRAIN_BATCHES, "train batches changed")
    require(runtime["validation_batches"] == EXPECTED_VALIDATION_BATCHES, "validation batches changed")

    prior_command = list(runtime["instrumented_command"])
    b1_dir = Path(extract_flag(prior_command, "--b1-dir")).resolve()
    b0_r3_dir = Path(extract_flag(prior_command, "--b0-r3-dir")).resolve()
    legacy_loader = Path(extract_flag(prior_command, "--loader-path")).resolve()
    legacy_model = Path(extract_flag(prior_command, "--model-path")).resolve()
    for path in (b1_dir, b0_r3_dir, legacy_loader, legacy_model):
        require(path.exists(), f"compatibility path missing: {path}")

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"
    require(
        not report_path.exists()
        and not lock_path.exists()
        and not complete_path.exists(),
        "E4 stage already exists; append-only rerun refused",
    )

    scratch = output_dir / "F7_E4_ONE_EPOCH_CONTROL_SMOKE_GATE_SCRATCH"
    require(not scratch.exists(), f"E4 scratch already exists: {scratch}")
    scratch.mkdir(parents=True)

    smoke_trainer = scratch / "run_v5_p3_f7_e4_one_epoch_control_smoke_trainer.py"
    instrumentation = create_one_epoch_smoke_trainer(
        trainer_path,
        smoke_trainer,
    )

    run_dir = scratch / "control_dynamic70_one_epoch_smoke"
    run_dir.mkdir()
    model_dir = run_dir / "model"
    trainer_report_dir = run_dir / "report"
    smoke_json = run_dir / "F7_E4_ONE_EPOCH_CONTROL_SMOKE_RESULT.json"
    require(not model_dir.exists(), "model dir exists before smoke")
    require(not trainer_report_dir.exists(), "report dir exists before smoke")

    environment = os.environ.copy()
    required_environment = {
        "F7_REPO": str(repo),
        "F7_DATA_LINK": str(data_link),
        "F7_SEED": str(EXPECTED_SEED),
        "F7_RUN_DIR": str(run_dir),
        "F7_ABLATION_LABEL": "control_dynamic70",
        "F7_E4_SMOKE_JSON": str(smoke_json),
    }
    environment.update(required_environment)
    environment["PYTHONPATH"] = (
        str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    for key, expected in required_environment.items():
        require(environment.get(key) == expected, f"environment mismatch: {key}")

    command = [
        sys.executable,
        str(smoke_trainer),
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
        str(trainer_report_dir),
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
        timeout=3600,
    )
    wall_seconds = time.perf_counter() - started

    console_path = run_dir / "console.log"
    atomic_text(
        console_path,
        process.stdout + "\n--- STDERR ---\n" + process.stderr,
    )
    require(
        process.returncode == 0,
        "E4 child failed; "
        f"returncode={process.returncode}; console={console_path}",
    )
    require(
        "F7_E4_ONE_EPOCH_CONTROL_SMOKE_COMPLETE"
        in (process.stdout + process.stderr),
        "E4 completion sentinel missing",
    )
    require(smoke_json.is_file(), "E4 result JSON missing")

    result = load_json(smoke_json)
    require(result.get("completed") is True, "E4 result not complete")
    require(result.get("failure") is None, f"E4 recorded failure: {result.get('failure')}")
    require(result.get("optimizer_steps") == EXPECTED_TRAIN_BATCHES, "optimizer steps are not one train epoch")
    require(result.get("scheduler_steps") == 1, "scheduler did not step once")
    require(len(result.get("epoch_started", [])) == 1, "not exactly one epoch started")
    require(result.get("epoch_started") == result.get("epoch_completed"), "epoch did not complete")
    require(result["loader_batches"]["train"] == EXPECTED_TRAIN_BATCHES, "train loader not fully consumed")
    validation_batches = int(result["loader_batches"]["validation"])
    require(validation_batches >= EXPECTED_VALIDATION_BATCHES, "validation loader not fully consumed")
    require(validation_batches % EXPECTED_VALIDATION_BATCHES == 0, "validation count is not a whole pass")
    require(result.get("cuda_available") is True, "E4 did not run on CUDA")

    require("test_directory_enumerated: false" in process.stdout, "test-directory non-enumeration missing")
    require("test_tensor_contents_accessed: false" in process.stdout, "sealed-test non-access missing")
    require(f"train_items: {EXPECTED_TRAIN_ITEMS}" in process.stdout, "console train count mismatch")
    require(f"validation_items: {EXPECTED_VALIDATION_ITEMS}" in process.stdout, "console validation count mismatch")
    require(f"parameter_count: {EXPECTED_PARAMETER_COUNT}" in process.stdout, "console parameter count mismatch")

    require(model_dir.is_dir(), "trainer did not finalize model directory")
    require(trainer_report_dir.is_dir(), "trainer did not finalize report directory")
    checkpoint_audits = audit_torch_loadable_files(model_dir)

    report_files = sorted(
        path for path in trainer_report_dir.rglob("*") if path.is_file()
    )
    require(report_files, "trainer report directory is empty")
    json_audits: list[dict[str, Any]] = []
    for path in report_files:
        if path.suffix.lower() != ".json":
            continue
        parsed = load_json(path)
        require(parsed, f"empty trainer JSON: {path}")
        json_audits.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "top_level_keys": sorted(parsed),
            }
        )
    require(json_audits, "no valid trainer JSON report finalized")

    run_files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    decision = {
        "stage": STAGE,
        "status": "PASS",
        "classification": CLASSIFICATION,
        "one_complete_control_train_epoch_verified": True,
        "complete_validation_pass_verified": True,
        "scheduler_step_verified": True,
        "disposable_checkpoint_verified": True,
        "trainer_report_finalization_verified": True,
        "clean_process_exit_verified": True,
        "scientific_checkpoint_saved": False,
        "sealed_test_access": False,
        "primary_matrix_execution_started": False,
        "primary_seed107_six_run_matrix_release_authorized": True,
        "F8_multi_seed_authorized": False,
        "feature_removal_authorized": False,
        "compact_interface_equivalence_authorized": False,
        "hardware_reduction_claim_authorized": False,
        "next_stage": (
            "V5_P3_F7_PRIMARY_SEED107_SAME_WIDTH_RETRAINED_"
            "GROUP_ABLATION_MATRIX_EXECUTION"
        ),
    }

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "scope": (
            "One disposable seed-107 Dynamic70 control epoch through all "
            "434 training batches, at least one complete 55-batch validation "
            "pass, one scheduler step, checkpoint creation, report generation, "
            "and clean normal finalization."
        ),
        "finding": {
            "instrumentation": instrumentation,
            "runtime": result,
            "process_wall_seconds": wall_seconds,
            "checkpoint_audits": checkpoint_audits,
            "trainer_json_audits": json_audits,
            "run_file_inventory": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in run_files
            ],
        },
        "decision": decision,
        "governance": {
            "disposable_control_optimizer_steps": result["optimizer_steps"],
            "scientific_optimizer_steps": 0,
            "disposable_checkpoint_saved": True,
            "scientific_checkpoint_saved": False,
            "sealed_test_access": False,
            "primary_matrix_execution_started": False,
        },
        "artifacts": {
            "E2_lock": str(e2_lock_path),
            "E3_lock": str(e3_lock_path),
            "source_trainer": str(trainer_path),
            "source_matrix_launcher": str(matrix_launcher),
            "smoke_trainer": str(smoke_trainer),
            "smoke_result": str(smoke_json),
            "console": str(console_path),
            "run_dir": str(run_dir),
            "model_dir": str(model_dir),
            "trainer_report_dir": str(trainer_report_dir),
        },
        "provenance": {
            "E2_lock_sha256": sha256_file(e2_lock_path),
            "E3_lock_sha256": sha256_file(e3_lock_path),
            "source_trainer_sha256": sha256_file(trainer_path),
            "source_matrix_launcher_sha256": sha256_file(matrix_launcher),
            "smoke_trainer_sha256": sha256_file(smoke_trainer),
            "smoke_result_sha256": sha256_file(smoke_json),
            "console_sha256": sha256_file(console_path),
            "installed_script_sha256": sha256_file(installed_script),
        },
    }

    atomic_json(report_path, report)
    atomic_json(
        lock_path,
        {
            "stage": STAGE,
            "status": "PASS",
            "classification": CLASSIFICATION,
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
            "E2_lock_sha256": sha256_file(e2_lock_path),
            "E3_lock_sha256": sha256_file(e3_lock_path),
            "source_trainer_sha256": sha256_file(trainer_path),
            "source_matrix_launcher_sha256": sha256_file(matrix_launcher),
            "smoke_trainer_sha256": sha256_file(smoke_trainer),
            "smoke_result_sha256": sha256_file(smoke_json),
            "train_batches": result["loader_batches"]["train"],
            "validation_batches": validation_batches,
            "optimizer_steps": result["optimizer_steps"],
            "scheduler_steps": result["scheduler_steps"],
            "disposable_checkpoint_saved": True,
            "scientific_checkpoint_saved": False,
            "sealed_test_access": False,
            "primary_matrix_execution_started": False,
            "primary_seed107_six_run_matrix_release_authorized": True,
            "next_stage": decision["next_stage"],
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print("one_complete_control_train_epoch_verified=true")
    print(f"train_batches_consumed={result['loader_batches']['train']}")
    print(f"validation_batches_consumed={validation_batches}")
    print(f"validation_passes={validation_batches // EXPECTED_VALIDATION_BATCHES}")
    print(f"optimizer_steps={result['optimizer_steps']}")
    print(f"scheduler_steps={result['scheduler_steps']}")
    print(f"epoch_started={result['epoch_started']}")
    print(f"epoch_completed={result['epoch_completed']}")
    print(f"process_wall_seconds={wall_seconds}")
    print(f"cuda_max_memory_allocated_bytes={result['cuda_max_memory_allocated_bytes']}")
    print(f"cuda_max_memory_reserved_bytes={result['cuda_max_memory_reserved_bytes']}")
    print(f"disposable_checkpoint_count={len(checkpoint_audits)}")
    print(f"trainer_json_report_count={len(json_audits)}")
    print("disposable_checkpoint_verified=true")
    print("trainer_report_finalization_verified=true")
    print("clean_process_exit_verified=true")
    print("scientific_checkpoint_saved=false")
    print("sealed_test_access=false")
    print("primary_matrix_execution_started=false")
    print("primary_seed107_six_run_matrix_release_authorized=true")
    print("F8_multi_seed_authorized=false")
    print("feature_removal_authorized=false")
    print(
        "next_stage=V5_P3_F7_PRIMARY_SEED107_SAME_WIDTH_RETRAINED_"
        "GROUP_ABLATION_MATRIX_EXECUTION"
    )
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
