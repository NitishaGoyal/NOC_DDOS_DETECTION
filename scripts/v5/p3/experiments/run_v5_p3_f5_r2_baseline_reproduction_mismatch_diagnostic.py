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


STAGE = "V5_P3_F5_R2_BASELINE_REPRODUCTION_MISMATCH_DIAGNOSTIC"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"

EXPECTED_ITEMS = 13863

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

LABEL_FILES = {
    "y_attack": "y_attack.npy",
    "y_attacker_count": "y_attacker_count.npy",
    "y_source": "y_source.npy",
    "y_transit": "y_transit.npy",
    "y_victim": "y_victim.npy",
    "y_attack_path": "y_attack_path.npy",
}

METRIC_NAMES = (
    "selection_score",
    "graph_auroc",
    "graph_ap",
    "graph_f1_at_0_5",
    "graph_fpr_at_0_5",
    "count_active_macro_f1",
    "source_ap",
    "source_exact_active",
    "transit_ap",
    "transit_exact_active",
    "victim_ap",
    "victim_exact_active",
    "path_ap",
    "path_exact_active",
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


def load_npy(path: Path) -> np.ndarray:
    require(path.is_file(), f"NumPy artifact missing: {path}")
    return np.load(path, allow_pickle=False, mmap_mode="r")


def array_summary(reference: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    shape_match = reference.shape == current.shape
    dtype_match = reference.dtype == current.dtype

    if not shape_match:
        return {
            "shape_match": False,
            "dtype_match": dtype_match,
            "reference_shape": list(reference.shape),
            "current_shape": list(current.shape),
        }

    ref64 = np.asarray(reference, dtype=np.float64)
    cur64 = np.asarray(current, dtype=np.float64)
    difference = cur64 - ref64
    absolute = np.abs(difference)

    flat_ref = ref64.reshape(-1)
    flat_cur = cur64.reshape(-1)

    if flat_ref.size >= 2 and np.std(flat_ref) > 0 and np.std(flat_cur) > 0:
        correlation = float(np.corrcoef(flat_ref, flat_cur)[0, 1])
    else:
        correlation = None

    if flat_ref.size >= 2 and np.var(flat_ref) > 0:
        slope = float(
            np.cov(flat_ref, flat_cur, ddof=0)[0, 1]
            / np.var(flat_ref)
        )
        intercept = float(np.mean(flat_cur) - slope * np.mean(flat_ref))
    else:
        slope = None
        intercept = None

    return {
        "shape_match": True,
        "dtype_match": dtype_match,
        "reference_shape": list(reference.shape),
        "current_shape": list(current.shape),
        "reference_dtype": str(reference.dtype),
        "current_dtype": str(current.dtype),
        "exact_equal": bool(np.array_equal(reference, current)),
        "max_abs_difference": float(np.max(absolute)),
        "mean_abs_difference": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "p99_abs_difference": float(np.quantile(absolute, 0.99)),
        "p999_abs_difference": float(np.quantile(absolute, 0.999)),
        "correlation": correlation,
        "affine_slope_current_vs_reference": slope,
        "affine_intercept_current_vs_reference": intercept,
    }


def binary_decision_summary(
    reference: np.ndarray,
    current: np.ndarray,
) -> dict[str, Any]:
    ref = np.asarray(reference).reshape(-1)
    cur = np.asarray(current).reshape(-1)
    require(ref.shape == cur.shape, "binary decision shape mismatch")
    ref_decision = ref >= 0.0
    cur_decision = cur >= 0.0
    flips = ref_decision != cur_decision
    return {
        "threshold": 0.0,
        "flip_count": int(np.sum(flips)),
        "flip_fraction": float(np.mean(flips)),
        "reference_positive_count": int(np.sum(ref_decision)),
        "current_positive_count": int(np.sum(cur_decision)),
    }


def count_argmax_summary(
    reference: np.ndarray,
    current: np.ndarray,
) -> dict[str, Any]:
    require(reference.shape == current.shape, "count shape mismatch")
    require(reference.ndim == 2, "count logits must be rank 2")
    ref_class = np.argmax(reference, axis=1)
    cur_class = np.argmax(current, axis=1)
    flips = ref_class != cur_class
    return {
        "flip_count": int(np.sum(flips)),
        "flip_fraction": float(np.mean(flips)),
        "reference_class_counts": {
            str(value): int(np.sum(ref_class == value))
            for value in np.unique(ref_class)
        },
        "current_class_counts": {
            str(value): int(np.sum(cur_class == value))
            for value in np.unique(cur_class)
        },
    }


def row_counter(array: np.ndarray) -> Counter[bytes]:
    contiguous = np.ascontiguousarray(array)
    if contiguous.ndim == 1:
        contiguous = contiguous.reshape(-1, 1)
    rows = contiguous.reshape(contiguous.shape[0], -1)
    return Counter(row.tobytes() for row in rows)


def label_comparison(
    cache_labels: dict[str, np.ndarray],
    fresh_labels: dict[str, np.ndarray],
) -> dict[str, Any]:
    rows = {}
    all_exact = True
    all_multiset = True

    for key in LABEL_FILES:
        cache = np.asarray(cache_labels[key])
        fresh = np.asarray(fresh_labels[key])
        shape_match = cache.shape == fresh.shape
        dtype_match = cache.dtype == fresh.dtype
        exact = bool(shape_match and np.array_equal(cache, fresh))
        multiset = bool(
            shape_match and row_counter(cache) == row_counter(fresh)
        )
        all_exact &= exact
        all_multiset &= multiset
        rows[key] = {
            "shape_match": shape_match,
            "dtype_match": dtype_match,
            "cache_shape": list(cache.shape),
            "fresh_shape": list(fresh.shape),
            "cache_dtype": str(cache.dtype),
            "fresh_dtype": str(fresh.dtype),
            "exact_equal": exact,
            "row_multiset_equal": multiset,
        }

    return {
        "all_label_arrays_exact": bool(all_exact),
        "all_label_row_multisets_equal": bool(all_multiset),
        "labels": rows,
    }


def combined_label_fingerprint(
    labels: dict[str, np.ndarray],
) -> np.ndarray:
    components = []
    for key in LABEL_FILES:
        array = np.asarray(labels[key])
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        components.append(array.reshape(array.shape[0], -1))
    combined = np.concatenate(components, axis=1)
    return np.ascontiguousarray(combined)


def role_cross_alignment(
    fresh_heads: dict[str, np.ndarray],
    baseline_heads: dict[str, np.ndarray],
) -> dict[str, Any]:
    roles = ("source", "transit", "victim", "path")
    matrix = {}
    best = {}

    for baseline_role in roles:
        comparisons = {}
        current = baseline_heads[baseline_role]
        for fresh_role in roles:
            reference = fresh_heads[fresh_role]
            require(
                reference.shape == current.shape,
                "role-head shape mismatch during cross alignment",
            )
            difference = (
                np.asarray(current, dtype=np.float64)
                - np.asarray(reference, dtype=np.float64)
            )
            mae = float(np.mean(np.abs(difference)))
            rmse = float(np.sqrt(np.mean(difference * difference)))
            flat_current = np.asarray(current, dtype=np.float64).reshape(-1)
            flat_reference = np.asarray(reference, dtype=np.float64).reshape(-1)
            correlation = (
                float(np.corrcoef(flat_current, flat_reference)[0, 1])
                if np.std(flat_current) > 0 and np.std(flat_reference) > 0
                else None
            )
            comparisons[fresh_role] = {
                "mean_abs_difference": mae,
                "rmse": rmse,
                "correlation": correlation,
            }

        ranked = sorted(
            comparisons.items(),
            key=lambda item: item[1]["mean_abs_difference"],
        )
        best_role, best_row = ranked[0]
        diagonal = comparisons[baseline_role]
        ratio = (
            diagonal["mean_abs_difference"]
            / best_row["mean_abs_difference"]
            if best_row["mean_abs_difference"] > 0
            else math.inf
        )
        matrix[baseline_role] = comparisons
        best[baseline_role] = {
            "best_fresh_role": best_role,
            "best_mean_abs_difference": best_row["mean_abs_difference"],
            "diagonal_mean_abs_difference": diagonal["mean_abs_difference"],
            "diagonal_to_best_ratio": ratio,
            "diagonal_is_best": best_role == baseline_role,
        }

    likely_mapping_problem = any(
        not row["diagonal_is_best"]
        and row["diagonal_to_best_ratio"] >= 5.0
        for row in best.values()
    )
    return {
        "matrix": matrix,
        "best_alignment": best,
        "likely_role_head_mapping_problem": likely_mapping_problem,
    }


def metric_comparison(
    baseline_metrics: dict[str, float],
    fresh_metrics: dict[str, float],
) -> dict[str, Any]:
    rows = {}
    all_within_1e6 = True
    maximum = 0.0

    for metric in METRIC_NAMES:
        require(metric in baseline_metrics, f"baseline metric missing: {metric}")
        require(metric in fresh_metrics, f"fresh metric missing: {metric}")
        baseline = float(baseline_metrics[metric])
        fresh = float(fresh_metrics[metric])
        difference = baseline - fresh
        absolute = abs(difference)
        maximum = max(maximum, absolute)
        passed = absolute <= 1e-6
        all_within_1e6 &= passed
        rows[metric] = {
            "baseline": baseline,
            "F4M_fresh": fresh,
            "baseline_minus_F4M_fresh": difference,
            "absolute_difference": absolute,
            "within_1e_6": passed,
        }

    return {
        "metrics": rows,
        "all_within_1e_6": bool(all_within_1e6),
        "maximum_absolute_metric_difference": maximum,
    }


def classify(
    *,
    labels: dict[str, Any],
    heads: dict[str, Any],
    role_cross: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[str, list[str], str]:
    reasons = []

    if not labels["all_label_arrays_exact"]:
        if labels["all_label_row_multisets_equal"]:
            classification = "SAMPLE_ORDER_OR_LABEL_EXTRACTION_MISMATCH"
            reasons.append(
                "Cached and fresh labels have equal row multisets but are not "
                "in identical order."
            )
        else:
            classification = "VALIDATION_LABEL_CONTENT_MISMATCH"
            reasons.append(
                "Cached and fresh validation labels differ in content."
            )
        return (
            classification,
            reasons,
            "V5_P3_F5_R3_VALIDATION_CACHE_LINEAGE_AND_ORDER_REPAIR",
        )

    if role_cross["likely_role_head_mapping_problem"]:
        reasons.append(
            "At least one role logit array aligns at least 5x better with a "
            "different fresh role head than with its nominal head."
        )
        return (
            "OUTPUT_HEAD_MAPPING_MISMATCH",
            reasons,
            "V5_P3_F5_R3_OUTPUT_HEAD_MAPPING_REPAIR",
        )

    head_rows = heads["heads"]
    maximum = max(
        row["numeric"]["max_abs_difference"]
        for row in head_rows.values()
        if row["numeric"].get("shape_match")
    )
    mean_max = max(
        row["numeric"]["mean_abs_difference"]
        for row in head_rows.values()
        if row["numeric"].get("shape_match")
    )
    minimum_correlation = min(
        row["numeric"]["correlation"]
        for row in head_rows.values()
        if row["numeric"].get("correlation") is not None
    )

    decision_flips = sum(
        row.get("decision", {}).get("flip_count", 0)
        for row in head_rows.values()
    )

    if (
        maximum <= 2e-5
        and mean_max <= 5e-7
        and minimum_correlation >= 0.999999
    ):
        reasons.append(
            "All labels are exact and all logit heads differ only at the "
            "small floating-point scale previously observed in F4."
        )
        if not metrics["all_within_1e_6"]:
            reasons.append(
                "The current F5 baseline gate is stricter than the observed "
                "metric behavior and needs a metric-level acceptance review."
            )
            return (
                "SMALL_NUMERIC_DRIFT_WITH_METRIC_GATE_MISMATCH",
                reasons,
                "V5_P3_F5_R3_BASELINE_METRIC_EQUIVALENCE_REVIEW",
            )

    if minimum_correlation < 0.99 or mean_max > 1e-3:
        reasons.append(
            "Labels are exact, but one or more logit heads have substantial "
            "numeric disagreement, indicating an input, batching, or forward "
            "route mismatch rather than ordinary backend drift."
        )
        return (
            "INFERENCE_INPUT_OR_FORWARD_ROUTE_MISMATCH",
            reasons,
            "V5_P3_F5_R3_OFFICIAL_D1_BATCH_AND_FORWARD_ROUTE_REPLAY",
        )

    if decision_flips > 0:
        reasons.append(
            "Labels are exact but the baseline changes threshold or argmax "
            "decisions relative to the F4 fresh replay."
        )
        return (
            "METRIC_RELEVANT_LOGIT_DIFFERENCE",
            reasons,
            "V5_P3_F5_R3_DECISION_DELTA_AND_RUNTIME_ROUTE_REVIEW",
        )

    reasons.append(
        "The mismatch does not fit a single high-confidence category from the "
        "available baseline and F4 output artifacts."
    )
    return (
        "MIXED_OR_UNRESOLVED_BASELINE_MISMATCH",
        reasons,
        "V5_P3_F5_R3_TARGETED_ROUTE_REVIEW",
    )


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")

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
    results_dir = run_dir / "results"

    baseline_metrics_path = baseline_dir / "METRICS.json"
    baseline_complete_path = baseline_dir / "UNIT_COMPLETE"
    baseline_gate_path = results_dir / "F5_BASELINE_REPRODUCTION_GATE.json"
    fresh_cert_path = metric_dir / (
        "F4M_R3A_FRESH_REPLAY_14_METRIC_CERTIFICATION.json"
    )
    r1_capture_path = run_dir / "F5_R1_OFFICIAL_EXPORTER_MODEL_CAPTURE.json"

    require(
        baseline_complete_path.is_file(),
        "F5 baseline unit did not complete",
    )
    require(
        baseline_metrics_path.is_file(),
        "F5 baseline metrics are missing",
    )
    require(
        baseline_gate_path.is_file(),
        "F5 baseline reproduction gate is missing",
    )
    require(
        fresh_cert_path.is_file(),
        "F4M fresh certification is missing",
    )
    require(
        r1_capture_path.is_file(),
        "F5-R1 model-capture provenance is missing",
    )

    baseline_document = load_json(baseline_metrics_path)
    baseline_gate = load_json(baseline_gate_path)
    fresh_cert = load_json(fresh_cert_path)
    r1_capture = load_json(r1_capture_path)

    require(
        r1_capture.get("state_dict_exact_match") is True,
        "F5-R1 model did not exactly match checkpoint",
    )
    require(
        baseline_gate.get("all_14_metrics_pass") is False,
        "F5-R2 expected a failed baseline gate",
    )

    fresh_npz_path = Path(fresh_cert["npz"]).resolve()
    require(fresh_npz_path.is_file(), f"fresh NPZ missing: {fresh_npz_path}")
    require(
        sha256_file(fresh_npz_path) == fresh_cert["npz_sha256"],
        "fresh NPZ hash changed",
    )

    baseline_heads = {
        head: load_npy(baseline_dir / filename)
        for head, filename in HEAD_FILES.items()
    }

    cache_labels = {
        key: load_npy(cache_dir / filename)
        for key, filename in LABEL_FILES.items()
    }

    with np.load(fresh_npz_path, allow_pickle=False) as fresh_archive:
        fresh_heads = {
            head: np.asarray(fresh_archive[key])
            for head, key in FRESH_HEAD_KEYS.items()
        }
        fresh_labels = {
            key: np.asarray(fresh_archive[key])
            for key in LABEL_FILES
        }

    for key, array in cache_labels.items():
        require(
            array.shape[0] == EXPECTED_ITEMS,
            f"cached label item count mismatch for {key}: {array.shape}",
        )
    for head, array in baseline_heads.items():
        require(
            array.shape[0] == EXPECTED_ITEMS,
            f"baseline logit item count mismatch for {head}: {array.shape}",
        )

    label_result = label_comparison(cache_labels, fresh_labels)

    combined_cache = combined_label_fingerprint(cache_labels)
    combined_fresh = combined_label_fingerprint(fresh_labels)
    combined_exact = bool(np.array_equal(combined_cache, combined_fresh))
    combined_multiset = bool(
        row_counter(combined_cache) == row_counter(combined_fresh)
    )
    label_result["combined_label_fingerprint"] = {
        "shape": list(combined_cache.shape),
        "exact_equal": combined_exact,
        "row_multiset_equal": combined_multiset,
        "unique_cache_rows": int(
            len(row_counter(combined_cache))
        ),
        "unique_fresh_rows": int(
            len(row_counter(combined_fresh))
        ),
    }

    head_rows = {}
    for head in HEAD_FILES:
        reference = fresh_heads[head]
        current = baseline_heads[head]
        numeric = array_summary(reference, current)
        row = {"numeric": numeric}

        if numeric.get("shape_match"):
            if head == "count":
                row["decision"] = count_argmax_summary(
                    reference,
                    current,
                )
            else:
                row["decision"] = binary_decision_summary(
                    reference,
                    current,
                )
        head_rows[head] = row

    head_result = {"heads": head_rows}
    role_cross = role_cross_alignment(fresh_heads, baseline_heads)

    baseline_metrics = {
        key: float(value)
        for key, value in baseline_document["metrics"].items()
        if key in METRIC_NAMES
    }
    fresh_metrics = {
        key: float(value)
        for key, value in fresh_cert["metrics"].items()
        if key in METRIC_NAMES
    }
    metric_result = metric_comparison(baseline_metrics, fresh_metrics)

    classification, reasons, next_stage = classify(
        labels=label_result,
        heads=head_result,
        role_cross=role_cross,
        metrics=metric_result,
    )

    label_path = output_dir / "F5_R2_CACHE_VS_F4_LABEL_COMPARISON.json"
    logit_path = output_dir / "F5_R2_BASELINE_VS_F4_LOGIT_COMPARISON.json"
    metric_path = output_dir / "F5_R2_BASELINE_METRIC_DELTA.json"
    decision_path = output_dir / "F5_R2_FAILURE_CLASSIFICATION.json"

    atomic_json(label_path, label_result)
    atomic_json(
        logit_path,
        {
            **head_result,
            "role_cross_alignment": role_cross,
        },
    )
    atomic_json(metric_path, metric_result)
    atomic_json(
        decision_path,
        {
            "classification": classification,
            "reasons": reasons,
            "next_stage": next_stage,
            "repair_authorized": False,
            "F5_permutation_units_authorized": False,
            "F6_integrated_gradients_authorized": False,
        },
    )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": classification,
        "scope": (
            "Diagnose the failed F5 same-runtime baseline gate using only the "
            "completed F5 baseline logits, cached validation labels, and the "
            "F4M fresh-certified validation-output NPZ."
        ),
        "input": {
            "run_directory": str(run_dir),
            "baseline_metrics": str(baseline_metrics_path),
            "baseline_gate": str(baseline_gate_path),
            "fresh_certification": str(fresh_cert_path),
            "fresh_npz": str(fresh_npz_path),
            "F5_R1_model_capture": str(r1_capture_path),
        },
        "findings": {
            "labels": label_result,
            "heads": head_result,
            "role_cross_alignment": role_cross,
            "metrics": metric_result,
            "classification_reasons": reasons,
        },
        "decision": {
            "F5_baseline_gate_complete": False,
            "F5_permutation_units_authorized": False,
            "repair_authorized": False,
            "F6_integrated_gradients_authorized": False,
            "F7_retraining_ablation_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": next_stage,
        },
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "validation_feature_tensors_loaded": False,
            "validation_cached_labels_loaded": True,
            "validation_output_artifact_payloads_loaded": True,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "model_weights_changed": False,
            "thresholds_changed": False,
            "metric_formulas_changed": False,
        },
        "artifacts": {
            "label_comparison": str(label_path),
            "logit_comparison": str(logit_path),
            "metric_delta": str(metric_path),
            "classification": str(decision_path),
        },
        "provenance": {
            "baseline_metrics_sha256": sha256_file(baseline_metrics_path),
            "baseline_gate_sha256": sha256_file(baseline_gate_path),
            "fresh_certification_sha256": sha256_file(fresh_cert_path),
            "fresh_npz_sha256": sha256_file(fresh_npz_path),
            "F5_R1_model_capture_sha256": sha256_file(r1_capture_path),
            "installed_script_sha256": sha256_file(installed_script),
            "label_comparison_sha256": sha256_file(label_path),
            "logit_comparison_sha256": sha256_file(logit_path),
            "metric_delta_sha256": sha256_file(metric_path),
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
            "label_comparison_sha256": sha256_file(label_path),
            "logit_comparison_sha256": sha256_file(logit_path),
            "metric_delta_sha256": sha256_file(metric_path),
            "classification_sha256": sha256_file(decision_path),
            "classification": classification,
            "F5_permutation_units_authorized": False,
            "F6_authorized": False,
            "validation_feature_tensors_loaded": False,
            "validation_cached_labels_loaded": True,
            "validation_output_artifact_payloads_loaded": True,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"run_directory={run_dir}")
    print(
        "all_label_arrays_exact="
        f"{str(label_result['all_label_arrays_exact']).lower()}"
    )
    print(
        "all_label_row_multisets_equal="
        f"{str(label_result['all_label_row_multisets_equal']).lower()}"
    )
    print(
        "combined_label_fingerprint_exact="
        f"{str(combined_exact).lower()}"
    )
    print(
        "combined_label_fingerprint_multiset_equal="
        f"{str(combined_multiset).lower()}"
    )

    for head, row in head_rows.items():
        numeric = row["numeric"]
        decision = row.get("decision", {})
        print(
            f"head_{head}="
            f"shape_match={numeric.get('shape_match')}:"
            f"dtype_match={numeric.get('dtype_match')}:"
            f"exact_equal={numeric.get('exact_equal')}:"
            f"max_abs_diff={numeric.get('max_abs_difference')}:"
            f"mean_abs_diff={numeric.get('mean_abs_difference')}:"
            f"rmse={numeric.get('rmse')}:"
            f"correlation={numeric.get('correlation')}:"
            f"decision_flips={decision.get('flip_count')}"
        )

    print(
        "likely_role_head_mapping_problem="
        f"{str(role_cross['likely_role_head_mapping_problem']).lower()}"
    )
    for role, row in role_cross["best_alignment"].items():
        print(
            f"role_alignment_{role}="
            f"best_fresh_role={row['best_fresh_role']}:"
            f"diagonal_is_best={row['diagonal_is_best']}:"
            f"diagonal_to_best_ratio={row['diagonal_to_best_ratio']}"
        )

    print(
        "all_metrics_within_1e_6="
        f"{str(metric_result['all_within_1e_6']).lower()}"
    )
    print(
        "maximum_absolute_metric_difference="
        f"{metric_result['maximum_absolute_metric_difference']}"
    )
    for metric, row in metric_result["metrics"].items():
        if not row["within_1e_6"]:
            print(
                f"failing_metric_{metric}="
                f"baseline={row['baseline']}:"
                f"F4M_fresh={row['F4M_fresh']}:"
                f"absolute_difference={row['absolute_difference']}"
            )

    print(f"failure_classification={classification}")
    print(f"classification_reasons={reasons}")
    print("F5_baseline_gate_complete=false")
    print("F5_permutation_units_authorized=false")
    print("repair_authorized=false")
    print("F6_integrated_gradients_authorized=false")
    print("model_loaded=false")
    print("checkpoint_loaded=false")
    print("validation_feature_tensors_loaded=false")
    print("validation_cached_labels_loaded=true")
    print("validation_output_artifact_payloads_loaded=true")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={next_stage}")
    print(f"label_comparison={label_path}")
    print(f"logit_comparison={logit_path}")
    print(f"metric_delta={metric_path}")
    print(f"classification_report={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
