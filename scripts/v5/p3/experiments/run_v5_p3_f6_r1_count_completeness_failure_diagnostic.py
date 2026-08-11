from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = "V5_P3_F6_R1_COUNT_COMPLETENESS_FAILURE_DIAGNOSTIC"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "VALIDATION-EXPLORATORY"

EXPECTED_SAMPLE_COUNT = 512
EXPECTED_BATCH_COUNT = 32
EXPECTED_BATCH_SIZE = 16
POINTS = (64, 128)

AGGREGATE_KEYS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"NPZ missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: np.asarray(archive[key])
            for key in archive.files
        }


def quantiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    require(array.size > 0, "cannot summarize an empty array")
    return {
        "minimum": float(np.min(array)),
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def compare_arrays(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any]:
    shape_match = left.shape == right.shape
    if not shape_match:
        return {
            "shape_match": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }

    if np.issubdtype(left.dtype, np.number) and np.issubdtype(
        right.dtype,
        np.number,
    ):
        difference = np.abs(
            np.asarray(left, dtype=np.float64)
            - np.asarray(right, dtype=np.float64)
        )
        return {
            "shape_match": True,
            "dtype_match": left.dtype == right.dtype,
            "exact_equal": bool(np.array_equal(left, right)),
            "maximum_absolute_difference": float(np.max(difference)),
            "mean_absolute_difference": float(np.mean(difference)),
            "rmse": float(np.sqrt(np.mean(difference * difference))),
        }

    return {
        "shape_match": True,
        "dtype_match": left.dtype == right.dtype,
        "exact_equal": bool(np.array_equal(left, right)),
    }


def reconstruct_attempt(attempt_dir: Path) -> dict[str, Any]:
    result_path = attempt_dir / "ATTEMPT_RESULT.json"
    complete_path = attempt_dir / "ATTEMPT_COMPLETE"
    aggregate_path = attempt_dir / "PER_SAMPLE_AGGREGATES.npz"
    batches_dir = attempt_dir / "batches"

    require(result_path.is_file(), f"attempt result missing: {result_path}")
    require(complete_path.is_file(), f"attempt complete marker missing: {complete_path}")
    require(aggregate_path.is_file(), f"aggregate NPZ missing: {aggregate_path}")
    require(batches_dir.is_dir(), f"batch directory missing: {batches_dir}")

    result = load_json(result_path)
    aggregate = load_npz(aggregate_path)

    batch_outputs = []
    batch_manifests = []

    for batch_number in range(EXPECTED_BATCH_COUNT):
        stem = f"batch_{batch_number:04d}"
        npz_path = batches_dir / f"{stem}.npz"
        manifest_path = batches_dir / f"{stem}.json"
        complete_batch_path = batches_dir / f"{stem}.complete"

        require(npz_path.is_file(), f"batch NPZ missing: {npz_path}")
        require(manifest_path.is_file(), f"batch manifest missing: {manifest_path}")
        require(
            complete_batch_path.is_file(),
            f"batch marker missing: {complete_batch_path}",
        )

        manifest = load_json(manifest_path)
        require(manifest.get("status") == "PASS", f"batch not PASS: {manifest_path}")
        require(
            manifest.get("npz_sha256") == sha256_file(npz_path),
            f"batch NPZ hash mismatch: {npz_path}",
        )
        require(
            int(manifest.get("sample_count")) == EXPECTED_BATCH_SIZE,
            f"batch sample count changed: {manifest_path}",
        )

        output = load_npz(npz_path)
        batch_outputs.append(output)
        batch_manifests.append(manifest)

    reconstructed = {
        key: np.concatenate(
            [batch[key] for batch in batch_outputs],
            axis=0,
        )
        for key in AGGREGATE_KEYS
    }

    require(
        reconstructed["attack_indices"].shape == (EXPECTED_SAMPLE_COUNT,),
        "reconstructed attack-index shape changed",
    )

    aggregate_comparison = {
        key: compare_arrays(reconstructed[key], aggregate[key])
        for key in AGGREGATE_KEYS
    }
    require(
        all(row.get("exact_equal") is True for row in aggregate_comparison.values()),
        f"stored aggregate differs from batch reconstruction: {attempt_dir}",
    )

    relative = reconstructed["relative_completeness_error"].astype(np.float64)
    absolute = reconstructed["absolute_completeness_error"].astype(np.float64)
    target_delta = reconstructed["target_delta"].astype(np.float64)
    ig_sum = reconstructed["ig_sum"].astype(np.float64)

    computed = {
        "median_relative_error": float(np.median(relative)),
        "p95_relative_error": float(np.quantile(relative, 0.95)),
        "maximum_relative_error": float(np.max(relative)),
        "mean_absolute_error": float(np.mean(absolute)),
    }
    recorded = result["completeness"]

    for key, value in computed.items():
        require(
            abs(float(recorded[key]) - value) <= 1e-8,
            f"recorded completeness mismatch at {attempt_dir}: {key}",
        )

    return {
        "attempt_dir": str(attempt_dir),
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "aggregate_path": str(aggregate_path),
        "aggregate_sha256": sha256_file(aggregate_path),
        "batch_count": len(batch_outputs),
        "batch_manifest_hashes": [
            {
                "path": manifest["npz"],
                "sha256": manifest["npz_sha256"],
            }
            for manifest in batch_manifests
        ],
        "result": result,
        "arrays": reconstructed,
        "aggregate_comparison": aggregate_comparison,
        "target_delta_absolute_quantiles": quantiles(np.abs(target_delta)),
        "absolute_error_quantiles": quantiles(absolute),
        "relative_error_quantiles": quantiles(relative),
        "ig_sum_absolute_quantiles": quantiles(np.abs(ig_sum)),
        "failure_counts": {
            "relative_error_gt_0_02": int(np.sum(relative > 0.02)),
            "relative_error_gt_0_05": int(np.sum(relative > 0.05)),
            "relative_error_gt_0_10": int(np.sum(relative > 0.10)),
            "relative_error_gt_1_00": int(np.sum(relative > 1.00)),
        },
    }


def attacker_count_lookup(selection: dict[str, Any]) -> dict[int, int]:
    rows = selection["rows"]
    return {
        int(row["attack_index"]): int(row["attacker_count"])
        for row in rows
    }


def class_breakdown(
    attempt: dict[str, Any],
    count_by_attack: dict[int, int],
) -> dict[str, Any]:
    arrays = attempt["arrays"]
    attack_indices = arrays["attack_indices"].astype(np.int64)
    relative = arrays["relative_completeness_error"].astype(np.float64)
    absolute = arrays["absolute_completeness_error"].astype(np.float64)
    target = np.abs(arrays["target_delta"].astype(np.float64))

    rows = {}
    for count in (1, 2, 3, 4):
        mask = np.asarray(
            [count_by_attack[int(index)] == count for index in attack_indices],
            dtype=bool,
        )
        require(int(np.sum(mask)) == 128, f"count-{count} sample count changed")
        rows[str(count)] = {
            "sample_count": int(np.sum(mask)),
            "target_delta_absolute_quantiles": quantiles(target[mask]),
            "absolute_error_quantiles": quantiles(absolute[mask]),
            "relative_error_quantiles": quantiles(relative[mask]),
            "relative_error_gt_0_02": int(np.sum(relative[mask] > 0.02)),
            "relative_error_gt_0_05": int(np.sum(relative[mask] > 0.05)),
        }
    return rows


def small_delta_review(attempt: dict[str, Any]) -> dict[str, Any]:
    arrays = attempt["arrays"]
    target = np.abs(arrays["target_delta"].astype(np.float64))
    absolute = arrays["absolute_completeness_error"].astype(np.float64)
    relative = arrays["relative_completeness_error"].astype(np.float64)

    thresholds = (
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        5e-2,
        1e-1,
    )
    rows = {}
    for threshold in thresholds:
        mask = target <= threshold
        rows[f"{threshold:.0e}"] = {
            "sample_count": int(np.sum(mask)),
            "fraction": float(np.mean(mask)),
            "relative_error_gt_0_02": int(np.sum(mask & (relative > 0.02))),
            "relative_error_gt_0_05": int(np.sum(mask & (relative > 0.05))),
            "absolute_error_mean": (
                float(np.mean(absolute[mask]))
                if np.any(mask)
                else None
            ),
            "relative_error_median": (
                float(np.median(relative[mask]))
                if np.any(mask)
                else None
            ),
        }

    failing_5 = relative > 0.05
    rows["five_percent_failure_concentration"] = {
        "failure_count": int(np.sum(failing_5)),
        "target_delta_absolute_quantiles": (
            quantiles(target[failing_5])
            if np.any(failing_5)
            else None
        ),
        "absolute_error_quantiles": (
            quantiles(absolute[failing_5])
            if np.any(failing_5)
            else None
        ),
    }
    return rows


def convergence_review(
    attempt64: dict[str, Any],
    attempt128: dict[str, Any],
) -> dict[str, Any]:
    a64 = attempt64["arrays"]
    a128 = attempt128["arrays"]

    identity = {
        key: compare_arrays(a64[key], a128[key])
        for key in (
            "attack_indices",
            "control_indices",
            "target_attack",
            "target_control",
            "target_delta",
        )
    }

    abs64 = a64["absolute_completeness_error"].astype(np.float64)
    abs128 = a128["absolute_completeness_error"].astype(np.float64)
    rel64 = a64["relative_completeness_error"].astype(np.float64)
    rel128 = a128["relative_completeness_error"].astype(np.float64)

    safe_abs64 = np.maximum(abs64, 1e-15)
    safe_rel64 = np.maximum(rel64, 1e-15)

    absolute_ratio = abs128 / safe_abs64
    relative_ratio = rel128 / safe_rel64

    return {
        "identity_comparison": identity,
        "absolute_error_change": {
            "median_64": float(np.median(abs64)),
            "median_128": float(np.median(abs128)),
            "p95_64": float(np.quantile(abs64, 0.95)),
            "p95_128": float(np.quantile(abs128, 0.95)),
            "mean_64": float(np.mean(abs64)),
            "mean_128": float(np.mean(abs128)),
            "median_128_over_64": float(np.median(absolute_ratio)),
            "improved_sample_count": int(np.sum(abs128 < abs64)),
            "worsened_sample_count": int(np.sum(abs128 > abs64)),
            "exact_tie_count": int(np.sum(abs128 == abs64)),
        },
        "relative_error_change": {
            "median_64": float(np.median(rel64)),
            "median_128": float(np.median(rel128)),
            "p95_64": float(np.quantile(rel64, 0.95)),
            "p95_128": float(np.quantile(rel128, 0.95)),
            "median_128_over_64": float(np.median(relative_ratio)),
            "improved_sample_count": int(np.sum(rel128 < rel64)),
            "worsened_sample_count": int(np.sum(rel128 > rel64)),
            "exact_tie_count": int(np.sum(rel128 == rel64)),
        },
        "absolute_error_difference_quantiles": quantiles(abs128 - abs64),
        "relative_error_difference_quantiles": quantiles(rel128 - rel64),
    }


def hybrid_gate_review(attempt: dict[str, Any]) -> dict[str, Any]:
    arrays = attempt["arrays"]
    target = np.abs(arrays["target_delta"].astype(np.float64))
    absolute = arrays["absolute_completeness_error"].astype(np.float64)

    target_scale = float(np.median(target))
    absolute_scale = float(np.quantile(absolute, 0.95))

    floors = {
        "fixed_1e_4": 1e-4,
        "fixed_1e_3": 1e-3,
        "one_percent_median_target": max(0.01 * target_scale, 1e-6),
        "five_times_p95_absolute_error": max(5.0 * absolute_scale, 1e-6),
    }

    rows = {}
    for name, floor in floors.items():
        normalized = absolute / np.maximum(target, floor)
        rows[name] = {
            "denominator_floor": float(floor),
            "median_normalized_error": float(np.median(normalized)),
            "p95_normalized_error": float(np.quantile(normalized, 0.95)),
            "maximum_normalized_error": float(np.max(normalized)),
            "pass_original_2pct_5pct_limits": bool(
                np.median(normalized) <= 0.02
                and np.quantile(normalized, 0.95) <= 0.05
            ),
        }
    return {
        "diagnostic_only": True,
        "gate_change_authorized": False,
        "target_scale_median": target_scale,
        "absolute_error_p95": absolute_scale,
        "candidate_normalizations": rows,
    }


def classify(
    attempt64: dict[str, Any],
    attempt128: dict[str, Any],
    convergence: dict[str, Any],
    small_delta128: dict[str, Any],
) -> tuple[str, list[str], str]:
    identity = convergence["identity_comparison"]
    endpoint_identity_ok = (
        identity["attack_indices"]["exact_equal"]
        and identity["control_indices"]["exact_equal"]
        and identity["target_delta"]["maximum_absolute_difference"] <= 5e-5
    )

    rel64 = attempt64["arrays"]["relative_completeness_error"].astype(np.float64)
    rel128 = attempt128["arrays"]["relative_completeness_error"].astype(np.float64)
    abs128 = attempt128["arrays"]["absolute_completeness_error"].astype(np.float64)
    target128 = np.abs(attempt128["arrays"]["target_delta"].astype(np.float64))

    p95_rel64 = float(np.quantile(rel64, 0.95))
    p95_rel128 = float(np.quantile(rel128, 0.95))
    median_rel64 = float(np.median(rel64))
    median_rel128 = float(np.median(rel128))
    p95_abs128 = float(np.quantile(abs128, 0.95))

    failing = rel128 > 0.05
    near_zero_limit = max(10.0 * p95_abs128, 1e-3)
    failure_near_zero_fraction = (
        float(np.mean(target128[failing] <= near_zero_limit))
        if np.any(failing)
        else 0.0
    )

    p95_improvement = (
        (p95_rel64 - p95_rel128) / max(p95_rel64, 1e-15)
    )
    median_improvement = (
        (median_rel64 - median_rel128) / max(median_rel64, 1e-15)
    )

    reasons = []

    if not endpoint_identity_ok:
        classification = "COUNT_ATTEMPT_ENDPOINT_OR_SAMPLE_IDENTITY_MISMATCH"
        reasons.append(
            "The 64-point and 128-point attempts do not preserve identical "
            "sample identities or endpoint targets within the diagnostic gate."
        )
        next_stage = "V5_P3_F6_R2_COUNT_ENDPOINT_REPLAY_AND_ARTIFACT_REPAIR"
        return classification, reasons, next_stage

    if (
        p95_abs128 <= 1e-3
        and failure_near_zero_fraction >= 0.80
    ):
        classification = (
            "COUNT_RELATIVE_COMPLETENESS_DENOMINATOR_INSTABILITY_"
            "ON_SMALL_TARGET_DELTAS"
        )
        reasons.extend([
            "The 128-point absolute completeness residual is small.",
            "At least 80% of samples failing the 5% relative gate have target "
            "differences no larger than a data-dependent near-zero threshold.",
            "The current max(abs(target_delta),1e-6) denominator can therefore "
            "magnify small absolute residuals into large relative errors.",
        ])
        next_stage = (
            "V5_P3_F6_R2_COUNT_HYBRID_ABSOLUTE_RELATIVE_"
            "COMPLETENESS_GATE_CERTIFICATION"
        )
        return classification, reasons, next_stage

    if (
        p95_improvement >= 0.25
        or median_improvement >= 0.25
    ):
        classification = "COUNT_IG_DISCRETIZATION_NONCONVERGENCE_AT_128_POINTS"
        reasons.extend([
            "The 128-point attempt materially improves completeness relative "
            "to 64 points but remains outside the frozen gate.",
            "This pattern is consistent with a path that requires a denser "
            "integration rule rather than a corrupted artifact.",
        ])
        next_stage = (
            "V5_P3_F6_R2_COUNT_256_POINT_TARGETED_RETRY"
        )
        return classification, reasons, next_stage

    classification = "COUNT_IG_NUMERICAL_OR_NONSMOOTH_PATH_PLATEAU"
    reasons.extend([
        "The 128-point attempt does not materially improve the failed "
        "completeness distribution relative to 64 points.",
        "The sample identities and endpoint targets remain aligned, so the "
        "remaining issue is a numerical/backend plateau or a highly "
        "nonsmooth count-target path.",
    ])
    next_stage = (
        "V5_P3_F6_R2_COUNT_INTEGRATION_RULE_AND_RUNTIME_DIAGNOSTIC"
    )
    return classification, reasons, next_stage


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")

    ig_root = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/integrated_gradients"
    )
    run_dir = active_run(ig_root, args.run_dir)

    p3_report_path = ig_root / (
        "V5_P3_F6_P3_DATASET_BUILDER_IDENTITY_ROUTE_RECOVERY_REPORT.json"
    )
    p3_lock_path = ig_root / (
        "V5_P3_F6_P3_DATASET_BUILDER_IDENTITY_ROUTE_RECOVERY_LOCK.json"
    )
    p3_report, p3_lock = verify_report_lock(p3_report_path, p3_lock_path)

    require(
        p3_lock.get("actual_F6_execution_authorized") is True,
        "F6-P3 did not authorize F6",
    )
    require(p3_lock.get("F7_authorized") is False, "F7 must remain held")

    run_contract_path = run_dir / "F6_RUN_CONTRACT.json"
    selection_path = run_dir / "F6_SELECTED_ATTACK_CONTROL_ITEMS.json"
    count_task_dir = run_dir / "tasks/count"
    count_task_complete = count_task_dir / "TASK_COMPLETE"

    require(run_contract_path.is_file(), "F6 run contract missing")
    require(selection_path.is_file(), "F6 frozen selection missing")
    require(count_task_dir.is_dir(), "count task directory missing")
    require(
        not count_task_complete.exists(),
        "count task unexpectedly marked complete",
    )

    contract = load_json(run_contract_path)
    selection = load_json(selection_path)

    require(
        selection["selected_item_count"] == EXPECTED_SAMPLE_COUNT,
        "F6 selected sample count changed",
    )
    require(
        selection["items_per_count"] == 128,
        "F6 selected items-per-count changed",
    )
    require(
        contract["default_IG_points"] == 64,
        "default IG point count changed",
    )
    require(
        contract["fallback_IG_points"] == 128,
        "fallback IG point count changed",
    )

    attempts = {
        points: reconstruct_attempt(
            count_task_dir / f"attempt_points_{points}"
        )
        for points in POINTS
    }

    count_by_attack = attacker_count_lookup(selection)
    class_reviews = {
        str(points): class_breakdown(attempts[points], count_by_attack)
        for points in POINTS
    }
    small_delta_reviews = {
        str(points): small_delta_review(attempts[points])
        for points in POINTS
    }
    convergence = convergence_review(attempts[64], attempts[128])
    hybrid_review = hybrid_gate_review(attempts[128])

    classification, reasons, next_stage = classify(
        attempts[64],
        attempts[128],
        convergence,
        small_delta_reviews["128"],
    )

    attempt_summary_path = output_dir / (
        "F6_R1_COUNT_64_AND_128_POINT_ATTEMPT_RECONSTRUCTION.json"
    )
    class_path = output_dir / (
        "F6_R1_COUNT_COMPLETENESS_BY_ATTACKER_COUNT.json"
    )
    convergence_path = output_dir / (
        "F6_R1_COUNT_64_TO_128_POINT_CONVERGENCE_REVIEW.json"
    )
    small_delta_path = output_dir / (
        "F6_R1_COUNT_SMALL_TARGET_DELTA_REVIEW.json"
    )
    hybrid_path = output_dir / (
        "F6_R1_COUNT_DIAGNOSTIC_HYBRID_GATE_REVIEW.json"
    )
    classification_path = output_dir / (
        "F6_R1_COUNT_COMPLETENESS_FAILURE_CLASSIFICATION.json"
    )

    atomic_json(
        attempt_summary_path,
        {
            str(points): {
                key: value
                for key, value in attempt.items()
                if key != "arrays"
            }
            for points, attempt in attempts.items()
        },
    )
    atomic_json(class_path, class_reviews)
    atomic_json(convergence_path, convergence)
    atomic_json(small_delta_path, small_delta_reviews)
    atomic_json(hybrid_path, hybrid_review)
    atomic_json(
        classification_path,
        {
            "classification": classification,
            "reasons": reasons,
            "repair_authorized": False,
            "count_task_complete": False,
            "other_F6_tasks_authorized_to_continue": False,
            "F6R_result_review_authorized": False,
            "F7_retraining_ablation_authorized": False,
            "feature_removal_authorized": False,
            "sealed_test_access": False,
            "next_stage": next_stage,
        },
    )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "failure_classification": classification,
        "scope": (
            "Reconstruct the completed 64-point and 128-point count IG "
            "attempts from all 64 hashed microbatch artifacts, verify stored "
            "aggregate identity, compare endpoint targets and selected sample "
            "order, measure convergence and attacker-count-specific error, "
            "test whether relative failure is concentrated on near-zero "
            "target deltas, and classify the next recovery route without "
            "loading the model, checkpoint, or validation features."
        ),
        "finding": {
            "attempt_64": {
                "completeness": attempts[64]["result"]["completeness"],
                "failure_counts": attempts[64]["failure_counts"],
            },
            "attempt_128": {
                "completeness": attempts[128]["result"]["completeness"],
                "failure_counts": attempts[128]["failure_counts"],
            },
            "convergence": convergence,
            "small_target_delta_review_128": small_delta_reviews["128"],
            "classification_reasons": reasons,
        },
        "decision": {
            "F6_R1_complete": True,
            "repair_authorized": False,
            "count_task_complete": False,
            "other_F6_tasks_authorized_to_continue": False,
            "F6R_result_review_authorized": False,
            "F7_retraining_ablation_authorized": False,
            "feature_removal_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": next_stage,
        },
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "validation_feature_tensors_loaded": False,
            "saved_validation_output_artifacts_loaded": True,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "model_weights_changed": False,
            "thresholds_changed": False,
            "IG_task_target_changed": False,
            "completeness_gate_changed": False,
        },
        "artifacts": {
            "attempt_reconstruction": str(attempt_summary_path),
            "attacker_count_review": str(class_path),
            "convergence_review": str(convergence_path),
            "small_target_delta_review": str(small_delta_path),
            "diagnostic_hybrid_gate_review": str(hybrid_path),
            "classification": str(classification_path),
        },
        "provenance": {
            "F6_P3_report_sha256": sha256_file(p3_report_path),
            "F6_P3_lock_sha256": sha256_file(p3_lock_path),
            "F6_run_contract_sha256": sha256_file(run_contract_path),
            "F6_selection_sha256": sha256_file(selection_path),
            "attempt_64_result_sha256": attempts[64]["result_sha256"],
            "attempt_64_aggregate_sha256": attempts[64]["aggregate_sha256"],
            "attempt_128_result_sha256": attempts[128]["result_sha256"],
            "attempt_128_aggregate_sha256": attempts[128]["aggregate_sha256"],
            "installed_script_sha256": sha256_file(installed_script),
            "attempt_reconstruction_sha256": sha256_file(attempt_summary_path),
            "attacker_count_review_sha256": sha256_file(class_path),
            "convergence_review_sha256": sha256_file(convergence_path),
            "small_target_delta_review_sha256": sha256_file(small_delta_path),
            "diagnostic_hybrid_gate_review_sha256": sha256_file(hybrid_path),
            "classification_sha256": sha256_file(classification_path),
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
            "attempt_reconstruction_sha256": sha256_file(attempt_summary_path),
            "attacker_count_review_sha256": sha256_file(class_path),
            "convergence_review_sha256": sha256_file(convergence_path),
            "small_target_delta_review_sha256": sha256_file(small_delta_path),
            "diagnostic_hybrid_gate_review_sha256": sha256_file(hybrid_path),
            "classification_sha256": sha256_file(classification_path),
            "failure_classification": classification,
            "repair_authorized": False,
            "count_task_complete": False,
            "F6R_authorized": False,
            "F7_authorized": False,
            "feature_removal_authorized": False,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "validation_feature_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    a64 = attempts[64]["result"]["completeness"]
    a128 = attempts[128]["result"]["completeness"]

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print(f"run_directory={run_dir}")
    print("count_attempt_64_batches_verified=32")
    print("count_attempt_128_batches_verified=32")
    print("stored_aggregates_exactly_reconstructed=true")
    print(
        "attempt_64_completeness="
        f"median_relative_error={a64['median_relative_error']}:"
        f"p95_relative_error={a64['p95_relative_error']}:"
        f"maximum_relative_error={a64['maximum_relative_error']}:"
        f"mean_absolute_error={a64['mean_absolute_error']}"
    )
    print(
        "attempt_128_completeness="
        f"median_relative_error={a128['median_relative_error']}:"
        f"p95_relative_error={a128['p95_relative_error']}:"
        f"maximum_relative_error={a128['maximum_relative_error']}:"
        f"mean_absolute_error={a128['mean_absolute_error']}"
    )
    identity = convergence["identity_comparison"]
    print(
        "attempt_identity="
        f"attack_indices_exact={identity['attack_indices']['exact_equal']}:"
        f"control_indices_exact={identity['control_indices']['exact_equal']}:"
        f"target_delta_max_abs_diff="
        f"{identity['target_delta']['maximum_absolute_difference']}"
    )
    print(
        "convergence_64_to_128="
        f"median_absolute_ratio="
        f"{convergence['absolute_error_change']['median_128_over_64']}:"
        f"median_relative_ratio="
        f"{convergence['relative_error_change']['median_128_over_64']}:"
        f"improved_samples="
        f"{convergence['relative_error_change']['improved_sample_count']}:"
        f"worsened_samples="
        f"{convergence['relative_error_change']['worsened_sample_count']}"
    )
    print(
        "attempt_128_failure_counts="
        f"{attempts[128]['failure_counts']}"
    )
    print(
        "attempt_128_target_delta_abs_quantiles="
        f"{attempts[128]['target_delta_absolute_quantiles']}"
    )
    print(
        "attempt_128_absolute_error_quantiles="
        f"{attempts[128]['absolute_error_quantiles']}"
    )
    for count in ("1", "2", "3", "4"):
        row = class_reviews["128"][count]
        print(
            f"count_class_{count}_128="
            f"median_rel={row['relative_error_quantiles']['median']}:"
            f"p95_rel={row['relative_error_quantiles']['q95']}:"
            f"p95_abs={row['absolute_error_quantiles']['q95']}:"
            f"fail_gt_5pct={row['relative_error_gt_0_05']}"
        )
    print(f"failure_classification={classification}")
    print(f"classification_reasons={reasons}")
    print("repair_authorized=false")
    print("count_task_complete=false")
    print("other_F6_tasks_authorized_to_continue=false")
    print("F6R_result_review_authorized=false")
    print("F7_retraining_ablation_authorized=false")
    print("feature_removal_authorized=false")
    print("model_loaded=false")
    print("checkpoint_loaded=false")
    print("validation_feature_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={next_stage}")
    print(f"attempt_reconstruction={attempt_summary_path}")
    print(f"attacker_count_review={class_path}")
    print(f"convergence_review={convergence_path}")
    print(f"small_target_delta_review={small_delta_path}")
    print(f"diagnostic_hybrid_gate_review={hybrid_path}")
    print(f"classification_report={classification_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
