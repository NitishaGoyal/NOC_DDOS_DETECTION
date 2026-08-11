from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


STAGE = "V5_P3_F6_R2_COUNT_256_POINT_TARGETED_RETRY"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "VALIDATION-EXPLORATORY"

EXPECTED_SAMPLE_COUNT = 512
EXPECTED_PARAMETER_COUNT = 60553
TARGET_POINTS = 256
BOOTSTRAP_SEED_256 = 6302

EXPECTED_R1_CLASSIFICATION = (
    "COUNT_IG_DISCRETIZATION_NONCONVERGENCE_AT_128_POINTS"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    parser.add_argument("--run-dir", default="")
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


def active_run(output_root: Path, explicit: str) -> Path:
    if explicit:
        run_dir = Path(explicit).expanduser().resolve()
    else:
        pointer = output_root / "F6_ACTIVE_RUN_PATH.txt"
        require(pointer.is_file(), f"F6 active-run pointer missing: {pointer}")
        run_dir = Path(
            pointer.read_text(encoding="utf-8").strip()
        ).expanduser().resolve()

    require(run_dir.is_dir(), f"F6 run directory missing: {run_dir}")
    return run_dir


def load_attempt_arrays(attempt_dir: Path) -> dict[str, np.ndarray]:
    aggregate_path = attempt_dir / "PER_SAMPLE_AGGREGATES.npz"
    result_path = attempt_dir / "ATTEMPT_RESULT.json"
    complete_path = attempt_dir / "ATTEMPT_COMPLETE"

    require(aggregate_path.is_file(), f"aggregate NPZ missing: {aggregate_path}")
    require(result_path.is_file(), f"attempt result missing: {result_path}")
    require(complete_path.is_file(), f"attempt marker missing: {complete_path}")

    with np.load(aggregate_path, allow_pickle=False) as archive:
        arrays = {
            key: np.asarray(archive[key])
            for key in archive.files
        }
    arrays["_result"] = load_json(result_path)
    arrays["_aggregate_path"] = np.asarray(str(aggregate_path))
    arrays["_aggregate_sha256"] = np.asarray(sha256_file(aggregate_path))
    arrays["_result_path"] = np.asarray(str(result_path))
    arrays["_result_sha256"] = np.asarray(sha256_file(result_path))
    return arrays


def compare_attempts(
    attempt128: dict[str, np.ndarray],
    attempt256: dict[str, np.ndarray],
) -> dict[str, Any]:
    attack_exact = bool(
        np.array_equal(
            attempt128["attack_indices"],
            attempt256["attack_indices"],
        )
    )
    control_exact = bool(
        np.array_equal(
            attempt128["control_indices"],
            attempt256["control_indices"],
        )
    )

    target_difference = np.abs(
        np.asarray(attempt128["target_delta"], dtype=np.float64)
        - np.asarray(attempt256["target_delta"], dtype=np.float64)
    )
    target_max = float(np.max(target_difference))
    target_mean = float(np.mean(target_difference))

    rel128 = np.asarray(
        attempt128["relative_completeness_error"],
        dtype=np.float64,
    )
    rel256 = np.asarray(
        attempt256["relative_completeness_error"],
        dtype=np.float64,
    )
    abs128 = np.asarray(
        attempt128["absolute_completeness_error"],
        dtype=np.float64,
    )
    abs256 = np.asarray(
        attempt256["absolute_completeness_error"],
        dtype=np.float64,
    )

    p95_128 = float(np.quantile(rel128, 0.95))
    p95_256 = float(np.quantile(rel256, 0.95))
    median_128 = float(np.median(rel128))
    median_256 = float(np.median(rel256))

    return {
        "attack_indices_exact": attack_exact,
        "control_indices_exact": control_exact,
        "target_delta_maximum_absolute_difference": target_max,
        "target_delta_mean_absolute_difference": target_mean,
        "median_relative_error_128": median_128,
        "median_relative_error_256": median_256,
        "p95_relative_error_128": p95_128,
        "p95_relative_error_256": p95_256,
        "median_relative_improvement_fraction": (
            (median_128 - median_256) / max(median_128, 1e-15)
        ),
        "p95_relative_improvement_fraction": (
            (p95_128 - p95_256) / max(p95_128, 1e-15)
        ),
        "median_absolute_error_128": float(np.median(abs128)),
        "median_absolute_error_256": float(np.median(abs256)),
        "p95_absolute_error_128": float(np.quantile(abs128, 0.95)),
        "p95_absolute_error_256": float(np.quantile(abs256, 0.95)),
        "relative_error_improved_sample_count": int(np.sum(rel256 < rel128)),
        "relative_error_worsened_sample_count": int(np.sum(rel256 > rel128)),
        "relative_error_tie_count": int(np.sum(rel256 == rel128)),
        "relative_error_gt_0_02_at_256": int(np.sum(rel256 > 0.02)),
        "relative_error_gt_0_05_at_256": int(np.sum(rel256 > 0.05)),
        "relative_error_gt_0_10_at_256": int(np.sum(rel256 > 0.10)),
        "relative_error_gt_1_00_at_256": int(np.sum(rel256 > 1.00)),
    }


def load_checkpoint_state(
    original: Any,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    require(isinstance(checkpoint, dict), "checkpoint is not a dictionary")
    state_dict = original.extract_state_dict(checkpoint)
    return checkpoint, state_dict


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")
    require(data_link.is_symlink(), f"dataset symlink missing: {data_link}")
    require(data_link.resolve().is_dir(), "dataset target missing")

    ig_root = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/integrated_gradients"
    )
    permutation_root = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/permutation"
    )
    run_dir = active_run(ig_root, args.run_dir)

    r1_report_path = ig_root / (
        "V5_P3_F6_R1_COUNT_COMPLETENESS_FAILURE_DIAGNOSTIC_REPORT.json"
    )
    r1_lock_path = ig_root / (
        "V5_P3_F6_R1_COUNT_COMPLETENESS_FAILURE_DIAGNOSTIC_LOCK.json"
    )
    r1_report, r1_lock = verify_report_lock(
        r1_report_path,
        r1_lock_path,
    )

    require(
        r1_lock.get("failure_classification")
        == EXPECTED_R1_CLASSIFICATION,
        "R1 did not classify the failure as 128-point discretization "
        "nonconvergence",
    )
    require(
        r1_lock.get("count_task_complete") is False,
        "count task was unexpectedly completed after R1",
    )
    require(r1_lock.get("F6R_authorized") is False, "F6R must remain held")
    require(r1_lock.get("F7_authorized") is False, "F7 must remain held")

    original_script_path = (
        repo
        / "scripts/v5/p3/experiments/"
        "run_v5_p3_f6_task_specific_integrated_gradients.py"
    )
    require(
        original_script_path.is_file(),
        f"original F6 implementation missing: {original_script_path}",
    )
    original = import_source(
        original_script_path,
        "_v5_p3_f6_r2_original_execution",
    )

    required_functions = (
        "configure_official_runtime",
        "active_f5_run",
        "open_cache",
        "capture_model_via_official_exporter",
        "extract_state_dict",
        "state_dict_exact_match",
        "parameter_count",
        "task_rows",
        "execute_attempt",
    )
    for name in required_functions:
        require(
            hasattr(original, name) and callable(getattr(original, name)),
            f"original F6 function missing: {name}",
        )

    execute_signature = inspect.signature(original.execute_attempt)
    require(
        "points" in execute_signature.parameters,
        "original execute_attempt no longer accepts integration points",
    )

    run_contract_path = run_dir / "F6_RUN_CONTRACT.json"
    selection_path = run_dir / "F6_SELECTED_ATTACK_CONTROL_ITEMS.json"
    count_task_dir = run_dir / "tasks/count"
    count_task_complete_path = count_task_dir / "TASK_COMPLETE"
    count_task_result_path = count_task_dir / "TASK_RESULT.json"
    sample_manifest_path = count_task_dir / "TASK_SAMPLE_MANIFEST.json"

    require(run_contract_path.is_file(), "F6 run contract missing")
    require(selection_path.is_file(), "F6 selected-pair artifact missing")
    require(count_task_dir.is_dir(), "count task directory missing")
    require(sample_manifest_path.is_file(), "count sample manifest missing")
    require(
        not count_task_complete_path.exists(),
        "count task already has TASK_COMPLETE before R2",
    )

    contract = load_json(run_contract_path)
    selection = load_json(selection_path)
    require(
        selection["selected_item_count"] == EXPECTED_SAMPLE_COUNT,
        "count selected sample count changed",
    )
    require(
        contract["default_IG_points"] == 64,
        "original default point count changed",
    )
    require(
        contract["fallback_IG_points"] == 128,
        "original fallback point count changed",
    )

    attempt128_dir = count_task_dir / "attempt_points_128"
    attempt128 = load_attempt_arrays(attempt128_dir)

    device, runtime = original.configure_official_runtime(args.device)
    require(
        runtime == contract["runtime"],
        "current runtime differs from frozen F6 run contract",
    )

    f5_run_dir = original.active_f5_run(permutation_root)
    cache_dir = f5_run_dir / "validation_cache"
    require(
        str(cache_dir) == contract["F5_validation_cache"]["cache_dir"],
        "active F5 validation cache differs from F6 run contract",
    )
    cache = original.open_cache(cache_dir)

    exporter_path = Path(contract["exporter"]["path"]).resolve()
    checkpoint_path = Path(contract["checkpoint"]["path"]).resolve()
    require(exporter_path.is_file(), f"exporter missing: {exporter_path}")
    require(checkpoint_path.is_file(), f"checkpoint missing: {checkpoint_path}")
    require(
        sha256_file(exporter_path) == contract["exporter"]["sha256"],
        "exporter hash changed",
    )
    require(
        sha256_file(checkpoint_path) == contract["checkpoint"]["sha256"],
        "checkpoint hash changed",
    )

    _, state_dict = load_checkpoint_state(original, checkpoint_path)
    model, model_capture = original.capture_model_via_official_exporter(
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
        original.parameter_count(model) == EXPECTED_PARAMETER_COUNT,
        "captured model parameter count changed",
    )
    require(
        original.state_dict_exact_match(model, state_dict),
        "captured model does not exactly match the frozen checkpoint",
    )

    rows = original.task_rows(
        "count",
        selection,
        cache,
    )
    require(len(rows) == EXPECTED_SAMPLE_COUNT, "count task row count changed")

    print(
        "count_256_retry_start="
        f"samples={len(rows)}:"
        f"points={TARGET_POINTS}:"
        f"microbatch={contract['sample_microbatch_size']}",
        flush=True,
    )

    attempt256_result = original.execute_attempt(
        task="count",
        points=TARGET_POINTS,
        task_rows_list=rows,
        task_dir=count_task_dir,
        model=model,
        device=device,
        cache=cache,
        bootstrap_seed=BOOTSTRAP_SEED_256,
    )

    attempt256_dir = count_task_dir / f"attempt_points_{TARGET_POINTS}"
    attempt256 = load_attempt_arrays(attempt256_dir)
    comparison = compare_attempts(attempt128, attempt256)

    require(
        comparison["attack_indices_exact"] is True,
        "128-point and 256-point attack sample order differs",
    )
    require(
        comparison["control_indices_exact"] is True,
        "128-point and 256-point control sample order differs",
    )
    require(
        comparison["target_delta_maximum_absolute_difference"] <= 5e-5,
        "128-point and 256-point endpoint target deltas differ materially",
    )

    completeness = attempt256_result["completeness"]
    count_pass = bool(completeness["pass"])

    r2_attempt_path = output_dir / (
        "F6_R2_COUNT_128_TO_256_POINT_CONVERGENCE.json"
    )
    decision_path = output_dir / (
        "F6_R2_COUNT_256_POINT_RETRY_DECISION.json"
    )

    atomic_json(
        r2_attempt_path,
        {
            "attempt_128": {
                "result_path": str(attempt128["_result_path"].item()),
                "result_sha256": str(attempt128["_result_sha256"].item()),
                "aggregate_path": str(attempt128["_aggregate_path"].item()),
                "aggregate_sha256": str(
                    attempt128["_aggregate_sha256"].item()
                ),
                "completeness": attempt128["_result"].item()
                if isinstance(attempt128["_result"], np.ndarray)
                else attempt128["_result"],
            },
            "attempt_256": {
                "result_path": str(attempt256["_result_path"].item()),
                "result_sha256": str(attempt256["_result_sha256"].item()),
                "aggregate_path": str(attempt256["_aggregate_path"].item()),
                "aggregate_sha256": str(
                    attempt256["_aggregate_sha256"].item()
                ),
                "completeness": attempt256_result["completeness"],
            },
            "comparison": comparison,
        },
    )

    # Determine next route before creating any task completion marker.
    if count_pass:
        next_stage = "V5_P3_F6_REMAINING_TASKS_RESUME"
        other_tasks_authorized = True
        repair_status = "COUNT_256_POINT_COMPLETENESS_RECOVERED"
    else:
        p95_improvement = comparison[
            "p95_relative_improvement_fraction"
        ]
        if p95_improvement >= 0.25:
            next_stage = "V5_P3_F6_R3_COUNT_512_POINT_TARGETED_RETRY"
            repair_status = "COUNT_256_POINT_IMPROVED_BUT_GATE_STILL_FAILED"
        else:
            next_stage = (
                "V5_P3_F6_R3_COUNT_INTEGRATION_RULE_AND_RUNTIME_DIAGNOSTIC"
            )
            repair_status = "COUNT_256_POINT_NUMERICAL_PLATEAU"
        other_tasks_authorized = False

    if count_pass:
        task_result = {
            "task": "count",
            "status": "PASS",
            "sample_count": len(rows),
            "fallback_128_points_used": True,
            "recovery_256_points_used": True,
            "selected_integration_points": TARGET_POINTS,
            "selected_attempt": attempt256_result,
            "sample_manifest": str(sample_manifest_path),
            "recovery": {
                "stage": STAGE,
                "R1_classification": EXPECTED_R1_CLASSIFICATION,
                "R1_report": str(r1_report_path),
                "R1_report_sha256": sha256_file(r1_report_path),
                "R1_lock": str(r1_lock_path),
                "R1_lock_sha256": sha256_file(r1_lock_path),
                "attempt_128_preserved": True,
                "attempt_256": str(attempt256_dir),
                "frozen_completeness_gate_changed": False,
                "task_target_changed": False,
                "sample_selection_changed": False,
                "runtime_changed": False,
                "model_or_checkpoint_changed": False,
            },
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(count_task_result_path, task_result)
        atomic_text(count_task_complete_path, "PASS\n")

    atomic_json(
        decision_path,
        {
            "R1_classification": EXPECTED_R1_CLASSIFICATION,
            "targeted_retry_points": TARGET_POINTS,
            "count_256_completeness": completeness,
            "count_completeness_pass": count_pass,
            "comparison_128_to_256": comparison,
            "count_task_complete": count_pass,
            "other_F6_tasks_authorized_to_continue": other_tasks_authorized,
            "F6R_result_review_authorized": False,
            "F7_retraining_ablation_authorized": False,
            "feature_removal_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "sealed_test_access": False,
            "repair_status": repair_status,
            "next_stage": next_stage,
        },
    )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "scope": (
            "Execute only the frozen 512-sample count integrated-gradients "
            "task at 256 trapezoidal points after R1 established material "
            "64-to-128 convergence; preserve the 64/128 attempts; verify "
            "identical attack/control sample order and endpoint targets; "
            "apply the unchanged 2%-median/5%-p95 completeness gate; and "
            "authorize remaining F6 tasks only if the count task passes."
        ),
        "finding": {
            "R1_classification": EXPECTED_R1_CLASSIFICATION,
            "attempt_128_completeness": (
                attempt128["_result"]["completeness"]
            ),
            "attempt_256_completeness": completeness,
            "comparison_128_to_256": comparison,
            "count_completeness_pass": count_pass,
            "repair_status": repair_status,
        },
        "decision": {
            "F6_R2_complete": True,
            "count_task_complete": count_pass,
            "other_F6_tasks_authorized_to_continue": (
                other_tasks_authorized
            ),
            "F6R_result_review_authorized": False,
            "F7_retraining_ablation_authorized": False,
            "feature_removal_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": next_stage,
        },
        "governance": {
            "model_loaded": True,
            "checkpoint_loaded": True,
            "validation_feature_tensors_loaded": True,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "model_weights_changed": False,
            "runtime_changed": False,
            "sample_selection_changed": False,
            "count_target_changed": False,
            "completeness_gate_changed": False,
            "attempt_64_preserved": True,
            "attempt_128_preserved": True,
        },
        "artifacts": {
            "F6_run_directory": str(run_dir),
            "attempt_256_directory": str(attempt256_dir),
            "attempt_256_result": str(
                attempt256_dir / "ATTEMPT_RESULT.json"
            ),
            "attempt_256_aggregate": str(
                attempt256_dir / "PER_SAMPLE_AGGREGATES.npz"
            ),
            "count_task_result": (
                str(count_task_result_path)
                if count_task_result_path.is_file()
                else None
            ),
            "count_task_complete_marker": (
                str(count_task_complete_path)
                if count_task_complete_path.is_file()
                else None
            ),
            "convergence_review": str(r2_attempt_path),
            "decision": str(decision_path),
        },
        "provenance": {
            "R1_report_sha256": sha256_file(r1_report_path),
            "R1_lock_sha256": sha256_file(r1_lock_path),
            "F6_run_contract_sha256": sha256_file(run_contract_path),
            "F6_selection_sha256": sha256_file(selection_path),
            "original_F6_implementation_sha256": sha256_file(
                original_script_path
            ),
            "exporter_sha256": sha256_file(exporter_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "model_capture_sha256": sha256_file(
                run_dir / "F6_OFFICIAL_EXPORTER_MODEL_CAPTURE.json"
            ),
            "attempt_128_result_sha256": str(
                attempt128["_result_sha256"].item()
            ),
            "attempt_128_aggregate_sha256": str(
                attempt128["_aggregate_sha256"].item()
            ),
            "attempt_256_result_sha256": str(
                attempt256["_result_sha256"].item()
            ),
            "attempt_256_aggregate_sha256": str(
                attempt256["_aggregate_sha256"].item()
            ),
            "installed_script_sha256": sha256_file(installed_script),
            "convergence_review_sha256": sha256_file(r2_attempt_path),
            "decision_sha256": sha256_file(decision_path),
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
            "convergence_review_sha256": sha256_file(r2_attempt_path),
            "decision_sha256": sha256_file(decision_path),
            "count_completeness_pass": count_pass,
            "count_task_complete": count_pass,
            "other_F6_tasks_authorized_to_continue": (
                other_tasks_authorized
            ),
            "F6R_authorized": False,
            "F7_authorized": False,
            "feature_removal_authorized": False,
            "completeness_gate_changed": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print(f"run_directory={run_dir}")
    print(f"R1_classification={EXPECTED_R1_CLASSIFICATION}")
    print("attempt_64_preserved=true")
    print("attempt_128_preserved=true")
    print("targeted_retry_points=256")
    print("attack_indices_exact_128_vs_256=true")
    print("control_indices_exact_128_vs_256=true")
    print(
        "target_delta_max_abs_diff_128_vs_256="
        f"{comparison['target_delta_maximum_absolute_difference']}"
    )
    print(
        "attempt_256_completeness="
        f"median_relative_error={completeness['median_relative_error']}:"
        f"p95_relative_error={completeness['p95_relative_error']}:"
        f"maximum_relative_error={completeness['maximum_relative_error']}:"
        f"mean_absolute_error={completeness['mean_absolute_error']}:"
        f"pass={str(count_pass).lower()}"
    )
    print(
        "convergence_128_to_256="
        f"median_relative_improvement_fraction="
        f"{comparison['median_relative_improvement_fraction']}:"
        f"p95_relative_improvement_fraction="
        f"{comparison['p95_relative_improvement_fraction']}:"
        f"improved_samples="
        f"{comparison['relative_error_improved_sample_count']}:"
        f"worsened_samples="
        f"{comparison['relative_error_worsened_sample_count']}"
    )
    print(
        "attempt_256_failure_counts="
        f"gt_2pct={comparison['relative_error_gt_0_02_at_256']}:"
        f"gt_5pct={comparison['relative_error_gt_0_05_at_256']}:"
        f"gt_10pct={comparison['relative_error_gt_0_10_at_256']}:"
        f"gt_100pct={comparison['relative_error_gt_1_00_at_256']}"
    )
    print(f"repair_status={repair_status}")
    print(f"count_task_complete={str(count_pass).lower()}")
    print(
        "other_F6_tasks_authorized_to_continue="
        f"{str(other_tasks_authorized).lower()}"
    )
    print("F6R_result_review_authorized=false")
    print("F7_retraining_ablation_authorized=false")
    print("feature_removal_authorized=false")
    print("hardware_reduction_claim_authorized=false")
    print("completeness_gate_changed=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={next_stage}")
    print(f"attempt_256_directory={attempt256_dir}")
    print(f"convergence_review={r2_attempt_path}")
    print(f"decision={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
