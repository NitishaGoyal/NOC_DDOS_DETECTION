from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "V5_P3_F7_M1_R1_PYTHONPATH_IMPORT_RECOVERY_AND_MATRIX_RESUME"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "TRAIN-DERIVED"

EXPECTED_TRAINER_SHA256 = (
    "8baeed5a7469c0a751ab0a614837d08d8d4058e1cd331a3128b9f5e3754e3faf"
)
EXPECTED_LAUNCHER_SHA256 = (
    "fa6e9690f55081ea3364a4a91c391ef08be52e38cfaddd43176051314e202f2a"
)
EXPECTED_ADAPTER_SHA256 = (
    "6e2b355d1c5f4f0e71c3eb0798b263aca2e936d11623d8526a20f6204feba059"
)
EXPECTED_LABELS = {
    "control_dynamic70",
    "ablate_directional_traffic_volume",
    "ablate_inter_flit_timing",
    "ablate_queue_activity",
    "ablate_buffer_pressure",
    "ablate_flow_control_stalls",
}
IMPORT_FAILURE_TEXT = "ModuleNotFoundError: No module named 'src'"


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
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.is_file():
        return ""
    return cmdline.read_bytes().replace(b"\0", b" ").decode(
        "utf-8",
        errors="replace",
    ).strip()


def parse_literal_assignment(
    launcher_text: str,
    variable: str,
) -> Path:
    prefix = variable + "="
    matching = [
        line.strip()
        for line in launcher_text.splitlines()
        if line.strip().startswith(prefix)
    ]
    require(
        len(matching) == 1,
        f"{variable} assignment is not unique: {matching}",
    )
    rhs = matching[0][len(prefix):]
    values = shlex.split(rhs)
    require(
        len(values) == 1,
        f"{variable} assignment is not one literal path: {rhs}",
    )
    value = values[0]
    require(
        "$" not in value,
        f"{variable} is not a literal path: {value}",
    )
    return Path(value).expanduser().resolve()


def tail_text(path: Path, max_lines: int = 120) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()
    return "\n".join(lines[-max_lines:])


def main() -> int:
    args = parse_args()

    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()

    require(repo.is_dir(), f"repository missing: {repo}")
    require(
        data_link.is_symlink(),
        f"guarded dataset symlink missing: {data_link}",
    )
    require(
        data_link.resolve().is_dir(),
        f"guarded dataset target missing: {data_link.resolve()}",
    )
    require(
        installed_script.is_file(),
        f"installed script missing: {installed_script}",
    )

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
        e2_lock.get("primary_matrix_execution_authorized") is True,
        "E2 did not authorize matrix execution",
    )
    require(
        e4_lock.get(
            "primary_seed107_six_run_matrix_release_authorized"
        ) is True,
        "E4 did not release the seed-107 matrix",
    )
    require(
        e2_lock.get("sealed_test_tensors_loaded") is False,
        "E2 records sealed-test access",
    )
    require(
        e4_lock.get("sealed_test_access") is False,
        "E4 records sealed-test access",
    )

    launcher = (
        repo
        / "scripts/v5/p3/experiments/"
        "run_v5_p3_f7_primary_seed107_matrix_e2_r6_r2.sh"
    )
    trainer = (
        repo
        / "scripts/v5/p3/experiments/"
        "run_v5_p3_f7_reconstructed_group_ablation_trainer_e2_r6_r2.py"
    )
    adapter = (
        repo
        / "src/models/v5_p3_f7_r5_runtime_adapter.py"
    )

    for path in (launcher, trainer, adapter):
        require(path.is_file(), f"required artifact missing: {path}")

    launcher_sha = sha256_file(launcher)
    trainer_sha = sha256_file(trainer)
    adapter_sha = sha256_file(adapter)

    require(
        launcher_sha == EXPECTED_LAUNCHER_SHA256,
        "launcher differs from certified hash",
    )
    require(
        launcher_sha == e2_lock.get("matrix_launcher_sha256"),
        "launcher differs from E2 lock",
    )
    require(
        launcher_sha
        == e4_lock.get("source_matrix_launcher_sha256"),
        "launcher differs from E4 lock",
    )
    require(
        trainer_sha == EXPECTED_TRAINER_SHA256,
        "trainer differs from certified hash",
    )
    require(
        trainer_sha == e2_lock.get("generated_trainer_sha256"),
        "trainer differs from E2 lock",
    )
    require(
        adapter_sha == EXPECTED_ADAPTER_SHA256,
        "runtime adapter differs from certified hash",
    )

    subprocess.run(["bash", "-n", str(launcher)], check=True)

    launcher_text = launcher.read_text(
        encoding="utf-8",
        errors="replace",
    )
    require(
        "PYTHONPATH" not in launcher_text,
        "certified launcher already contains PYTHONPATH; "
        "failure classification requires review",
    )
    require(
        str(trainer) in launcher_text,
        "launcher does not bind the certified trainer",
    )

    matrix_dir = parse_literal_assignment(
        launcher_text,
        "MATRIX_DIR",
    )
    run_root = parse_literal_assignment(
        launcher_text,
        "RUN_ROOT",
    )
    launcher_adapter = parse_literal_assignment(
        launcher_text,
        "ADAPTER",
    )
    require(matrix_dir.is_dir(), f"matrix directory missing: {matrix_dir}")
    require(
        launcher_adapter == adapter,
        "launcher adapter binding changed",
    )

    spec_files = sorted(matrix_dir.glob("*.json"))
    require(len(spec_files) == 6, f"expected six specs, found {len(spec_files)}")

    specs: list[dict[str, Any]] = []
    for spec_path in spec_files:
        spec = load_json(spec_path)
        label = spec.get("label")
        require(
            isinstance(label, str) and label,
            f"spec label missing: {spec_path}",
        )
        specs.append({
            "path": str(spec_path),
            "sha256": sha256_file(spec_path),
            "label": label,
        })

    labels_in_launch_order = [row["label"] for row in specs]
    require(
        set(labels_in_launch_order) == EXPECTED_LABELS,
        "matrix label set changed: "
        f"{labels_in_launch_order}",
    )

    # The launcher uses shell glob order, so the first run is determined by
    # spec filename sorting, not by the conceptual control-first listing.
    require(
        labels_in_launch_order[0] == "ablate_buffer_pressure",
        "observed first failed label no longer matches sorted spec order",
    )

    old_pid_path = output_dir / (
        "V5_P3_F7_PRIMARY_SEED107_MATRIX.pid"
    )
    old_master_log = output_dir / (
        "V5_P3_F7_PRIMARY_SEED107_MATRIX_MASTER.log"
    )
    stale_pid: int | None = None
    stale_pid_command = ""
    if old_pid_path.is_file():
        raw_pid = old_pid_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()
        require(raw_pid.isdigit(), f"invalid old PID file: {old_pid_path}")
        stale_pid = int(raw_pid)
        if process_exists(stale_pid):
            stale_pid_command = process_command(stale_pid)
            require(
                str(launcher) not in stale_pid_command,
                "previous matrix launcher is still active; "
                f"pid={stale_pid}; command={stale_pid_command}",
            )

    require(
        old_master_log.is_file(),
        f"original failed master log missing: {old_master_log}",
    )
    old_master_tail = tail_text(old_master_log, 160)
    require(
        IMPORT_FAILURE_TEXT in old_master_tail,
        "original master log does not contain the certified import failure",
    )
    require(
        "launching ablate_buffer_pressure" in old_master_tail,
        "original master log does not identify the first sorted label",
    )

    state_rows: list[dict[str, Any]] = []
    archived_failure_logs: list[dict[str, Any]] = []
    completed_labels: list[str] = []
    pending_labels: list[str] = []

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for label in labels_in_launch_order:
        run_dir = run_root / label
        model_dir = run_dir / "model"
        report_dir = run_dir / "report"
        complete_marker = report_dir / "F7_RUN_COMPLETE"
        console_log = run_dir / "console.log"

        if complete_marker.is_file():
            require(
                model_dir.is_dir() and report_dir.is_dir(),
                f"complete marker without model/report directories: {label}",
            )
            completed_labels.append(label)
            state = "complete"
        elif model_dir.exists() or report_dir.exists():
            raise RuntimeError(
                "partial scientific output exists and resume is refused: "
                f"label={label}; model_exists={model_dir.exists()}; "
                f"report_exists={report_dir.exists()}; run_dir={run_dir}"
            )
        else:
            pending_labels.append(label)
            state = "pending"

            if console_log.is_file():
                console_text = console_log.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                require(
                    IMPORT_FAILURE_TEXT in console_text,
                    "unexpected console exists in pending run: "
                    f"{console_log}",
                )
                require(
                    label == "ablate_buffer_pressure",
                    "import-failure console exists under unexpected label: "
                    f"{label}",
                )
                archive = run_dir / (
                    f"console.pre_pythonpath_import_failure.{timestamp}.log"
                )
                require(
                    not archive.exists(),
                    f"archive target already exists: {archive}",
                )
                os.replace(console_log, archive)
                archived_failure_logs.append({
                    "label": label,
                    "original": str(console_log),
                    "archive": str(archive),
                    "sha256": sha256_file(archive),
                })

            if run_dir.exists():
                allowed_files = {
                    Path(row["archive"]).resolve()
                    for row in archived_failure_logs
                    if row["label"] == label
                }
                actual_files = {
                    path.resolve()
                    for path in run_dir.rglob("*")
                    if path.is_file()
                }
                require(
                    actual_files == allowed_files,
                    "unexpected files remain in pending run directory: "
                    f"label={label}; files={sorted(map(str, actual_files))}",
                )

        state_rows.append({
            "label": label,
            "run_dir": str(run_dir),
            "model_dir_exists": model_dir.exists(),
            "report_dir_exists": report_dir.exists(),
            "complete_marker_exists": complete_marker.is_file(),
            "state": state,
        })

    require(
        pending_labels,
        "all six runs are already complete; resume launch is unnecessary",
    )

    # Verify the exact import path correction independently before launch.
    environment = os.environ.copy()
    prior_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(repo)
        + (os.pathsep + prior_pythonpath if prior_pythonpath else "")
    )
    environment["PYTHONUNBUFFERED"] = "1"

    import_test = subprocess.run(
        [
            str(repo / ".venv/bin/python"),
            "-c",
            (
                "import pathlib,src;"
                "import src.models.v5_p3_f7_r5_runtime_adapter as a;"
                "print('F7_SRC_IMPORT_PASS');"
                "print(pathlib.Path(src.__file__).resolve());"
                "print(pathlib.Path(a.__file__).resolve())"
            ),
        ],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    require(
        import_test.returncode == 0,
        "repository import-path self-test failed: "
        f"stdout={import_test.stdout}; stderr={import_test.stderr}",
    )
    require(
        "F7_SRC_IMPORT_PASS" in import_test.stdout,
        "repository import-path self-test sentinel missing",
    )
    require(
        str(adapter.resolve()) in import_test.stdout,
        "runtime adapter import resolved to an unexpected file",
    )

    report_path = output_dir / f"{STAGE}_LAUNCH_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LAUNCH_LOCK.json"
    complete_path = output_dir / f"{STAGE}_LAUNCHED"
    require(
        not report_path.exists()
        and not lock_path.exists()
        and not complete_path.exists(),
        "M1-R1 launch artifacts already exist; append-only rerun refused",
    )

    master_log = output_dir / (
        "V5_P3_F7_PRIMARY_SEED107_MATRIX_M1_R1_MASTER.log"
    )
    pid_path = output_dir / (
        "V5_P3_F7_PRIMARY_SEED107_MATRIX_M1_R1.pid"
    )
    require(
        not master_log.exists() and not pid_path.exists(),
        "M1-R1 master log or PID file already exists",
    )

    master_handle = master_log.open("xb")
    try:
        header = (
            "===== V5-P3 F7 M1-R1 PYTHONPATH IMPORT RECOVERY "
            "AND MATRIX RESUME =====\n"
            f"created_utc={datetime.now(timezone.utc).isoformat()}\n"
            f"repo={repo}\n"
            f"data_link={data_link}\n"
            f"launcher={launcher}\n"
            f"launcher_sha256={launcher_sha}\n"
            f"trainer_sha256={trainer_sha}\n"
            f"adapter_sha256={adapter_sha}\n"
            f"PYTHONPATH_prefix={repo}\n"
            f"completed_labels={completed_labels}\n"
            f"pending_labels={pending_labels}\n"
            f"launch_order={labels_in_launch_order}\n"
            "sealed_test_access=false\n"
            "F8_multi_seed_authorized=false\n"
            "feature_removal_authorized=false\n"
            "scientific_matrix_resume=true\n"
            "===============================================\n"
        )
        master_handle.write(header.encode("utf-8"))
        master_handle.flush()
        os.fsync(master_handle.fileno())

        process = subprocess.Popen(
            [
                "bash",
                str(launcher),
                str(repo),
                str(data_link),
            ],
            cwd=repo,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=master_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        master_handle.close()

    atomic_text(pid_path, f"{process.pid}\n")

    # Catch only immediate launch/import failures. A healthy first epoch will
    # remain active well beyond this bounded observation window.
    observation_seconds = 12
    deadline = time.monotonic() + observation_seconds
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            break
        time.sleep(1)

    returncode = process.poll()
    current_tail = tail_text(master_log, 180)

    require(
        IMPORT_FAILURE_TEXT not in current_tail,
        "M1-R1 still encountered the src import failure; "
        f"master_log={master_log}",
    )
    require(
        "F7_SRC_IMPORT_PASS" not in current_tail,
        "internal import self-test leaked into matrix output",
    )
    require(
        "launching ablate_buffer_pressure" in current_tail
        or any(
            f"skip complete run: {label}" in current_tail
            for label in completed_labels
        ),
        "matrix launcher did not begin/resume its spec loop",
    )

    if returncode is not None:
        raise RuntimeError(
            "matrix launcher exited during the immediate observation window; "
            f"returncode={returncode}; master_log={master_log}; "
            f"tail=\n{current_tail}"
        )

    require(
        process_exists(process.pid),
        "matrix process disappeared after launch observation",
    )

    report = {
        "stage": STAGE,
        "status": "LAUNCHED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "failure_classification": (
            "matrix_wrapper_omitted_repository_from_PYTHONPATH_so_"
            "certified_trainer_could_not_import_src_runtime_adapter"
        ),
        "finding": {
            "original_attempt": {
                "old_pid_file": str(old_pid_path),
                "old_pid": stale_pid,
                "old_pid_alive": (
                    process_exists(stale_pid)
                    if stale_pid is not None
                    else False
                ),
                "old_pid_command": stale_pid_command,
                "old_master_log": str(old_master_log),
                "old_master_log_sha256": sha256_file(old_master_log),
                "failure_before_dataset_construction": True,
                "failure_before_optimizer_step": True,
                "scientific_checkpoint_saved": False,
                "first_sorted_label": labels_in_launch_order[0],
            },
            "recovery": {
                "certified_launcher_modified": False,
                "repository_prepended_to_PYTHONPATH": True,
                "child_working_directory": str(repo),
                "runtime_adapter_import_self_test": {
                    "returncode": import_test.returncode,
                    "stdout": import_test.stdout,
                    "stderr": import_test.stderr,
                    "status": "PASS",
                },
                "archived_failure_logs": archived_failure_logs,
                "run_state_before_resume": state_rows,
                "completed_labels": completed_labels,
                "pending_labels": pending_labels,
                "launch_order_from_sorted_spec_files": (
                    labels_in_launch_order
                ),
            },
            "launch": {
                "pid": process.pid,
                "pid_alive_after_observation": True,
                "observation_seconds": observation_seconds,
                "master_log": str(master_log),
                "pid_file": str(pid_path),
                "master_log_tail": current_tail,
            },
        },
        "governance": {
            "primary_seed107_matrix_execution_resumed": True,
            "sealed_test_access": False,
            "F8_multi_seed_authorized": False,
            "feature_removal_authorized": False,
            "compact_interface_equivalence_authorized": False,
            "hardware_reduction_claim_authorized": False,
        },
        "artifacts": {
            "E2_lock": str(e2_lock_path),
            "E4_lock": str(e4_lock_path),
            "launcher": str(launcher),
            "trainer": str(trainer),
            "runtime_adapter": str(adapter),
            "matrix_dir": str(matrix_dir),
            "run_root": str(run_root),
            "master_log": str(master_log),
            "pid_file": str(pid_path),
        },
        "provenance": {
            "E2_lock_sha256": sha256_file(e2_lock_path),
            "E4_lock_sha256": sha256_file(e4_lock_path),
            "launcher_sha256": launcher_sha,
            "trainer_sha256": trainer_sha,
            "runtime_adapter_sha256": adapter_sha,
            "installed_script_sha256": sha256_file(installed_script),
            "specs": specs,
        },
    }

    atomic_json(report_path, report)
    atomic_json(
        lock_path,
        {
            "stage": STAGE,
            "status": "LAUNCHED",
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
            "pid": process.pid,
            "pid_file": str(pid_path),
            "master_log": str(master_log),
            "launcher_sha256": launcher_sha,
            "trainer_sha256": trainer_sha,
            "runtime_adapter_sha256": adapter_sha,
            "repository_prepended_to_PYTHONPATH": True,
            "certified_launcher_modified": False,
            "completed_labels_before_resume": completed_labels,
            "pending_labels_before_resume": pending_labels,
            "sealed_test_access": False,
            "F8_multi_seed_authorized": False,
            "feature_removal_authorized": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_LAUNCHED\n")

    print(f"{STAGE}_LAUNCHED")
    print("status=LAUNCHED")
    print(f"classification={CLASSIFICATION}")
    print(
        "failure_classification="
        "matrix_wrapper_omitted_repository_from_PYTHONPATH"
    )
    print("failure_before_dataset_construction=true")
    print("failure_before_optimizer_step=true")
    print("scientific_checkpoint_saved_by_failed_attempt=false")
    print("certified_launcher_modified=false")
    print("repository_prepended_to_PYTHONPATH=true")
    print("runtime_adapter_import_self_test=PASS")
    print(
        "launch_order="
        + ",".join(labels_in_launch_order)
    )
    print(
        "completed_labels_before_resume="
        + ",".join(completed_labels)
    )
    print(
        "pending_labels_before_resume="
        + ",".join(pending_labels)
    )
    print(f"matrix_pid={process.pid}")
    print("matrix_pid_alive=true")
    print(f"master_log={master_log}")
    print(f"pid_file={pid_path}")
    print("sealed_test_access=false")
    print("F8_multi_seed_authorized=false")
    print("feature_removal_authorized=false")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
