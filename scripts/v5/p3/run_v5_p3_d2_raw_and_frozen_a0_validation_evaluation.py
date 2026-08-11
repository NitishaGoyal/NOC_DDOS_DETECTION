from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


STAGE = "V5_P3_D2_RAW_AND_FROZEN_A0_TRANCHE_A_VALIDATION_EVALUATION"
CAMPAIGN_LABEL = "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic"
EXPECTED_ITEMS = 13_863
EXPECTED_A0_THRESHOLDS = {
    "graph": 0.47174675035328983,
    "source": 0.94960549299285935,
    "transit": 0.85703332488359107,
    "victim": 0.82601148026998117,
    "path": 0.8339347466090468,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def import_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    return torch.sigmoid(
        torch.as_tensor(values, dtype=torch.float64)
    ).cpu().numpy()


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while (
            stop < len(values)
            and sorted_values[stop] == sorted_values[start]
        ):
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def binary_auroc(
    scores: np.ndarray,
    targets: np.ndarray,
) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    positive = targets == 1
    negative = targets == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = average_ranks(scores)
    return float(
        (
            ranks[positive].sum()
            - n_positive * (n_positive + 1) / 2.0
        )
        / (n_positive * n_negative)
    )


def binary_average_precision(
    scores: np.ndarray,
    targets: np.ndarray,
) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    positive_count = int((targets == 1).sum())
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered_targets = targets[order]
    cumulative_positive = np.cumsum(ordered_targets == 1)
    positions = np.arange(1, len(targets) + 1)
    precision = cumulative_positive / positions
    return float(
        precision[ordered_targets == 1].sum() / positive_count
    )


def binary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.uint8).reshape(-1)
    target = np.asarray(target, dtype=np.uint8).reshape(-1)
    tp = int(np.sum((prediction == 1) & (target == 1)))
    fp = int(np.sum((prediction == 1) & (target == 0)))
    fn = int(np.sum((prediction == 0) & (target == 1)))
    tn = int(np.sum((prediction == 0) & (target == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def multiclass_metrics_active(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    per_class = {}
    f1_values = []
    for class_value in (1, 2, 3, 4):
        pred_positive = prediction == class_value
        true_positive = target == class_value
        tp = int(np.sum(pred_positive & true_positive))
        fp = int(np.sum(pred_positive & ~true_positive))
        fn = int(np.sum(~pred_positive & true_positive))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[str(class_value)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(true_positive.sum()),
        }
        f1_values.append(f1)
    return {
        "accuracy": float(np.mean(prediction == target)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "items": int(target.size),
    }


def role_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    scores: np.ndarray,
    active_truth: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.uint8)
    target = np.asarray(target, dtype=np.uint8)
    flattened = binary_metrics(
        prediction.reshape(-1),
        target.reshape(-1),
    )
    exact_all = float(
        np.mean(np.all(prediction == target, axis=1))
    )
    exact_active = float(
        np.mean(
            np.all(
                prediction[active_truth] == target[active_truth],
                axis=1,
            )
        )
    )
    exact_inactive = float(
        np.mean(
            np.all(
                prediction[~active_truth] == target[~active_truth],
                axis=1,
            )
        )
    )
    predicted_cardinality = prediction.sum(axis=1)
    true_cardinality = target.sum(axis=1)
    return {
        "flattened": flattened,
        "average_precision_raw_logits": binary_average_precision(
            scores.reshape(-1),
            target.reshape(-1),
        ),
        "auroc_raw_logits": binary_auroc(
            scores.reshape(-1),
            target.reshape(-1),
        ),
        "exact_all": exact_all,
        "exact_active": exact_active,
        "exact_inactive": exact_inactive,
        "mean_predicted_cardinality_all": float(
            predicted_cardinality.mean()
        ),
        "mean_predicted_cardinality_active": float(
            predicted_cardinality[active_truth].mean()
        ),
        "mean_true_cardinality_active": float(
            true_cardinality[active_truth].mean()
        ),
    }


def strict_exactness(
    graph_prediction: np.ndarray,
    count_prediction: np.ndarray,
    role_predictions: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> dict[str, Any]:
    graph_truth = labels["graph"].astype(np.uint8)
    active_truth = graph_truth == 1
    graph_correct = graph_prediction == graph_truth

    roles_correct = np.ones(graph_truth.shape[0], dtype=bool)
    roles_empty = np.ones(graph_truth.shape[0], dtype=bool)
    for role in ("source", "transit", "victim", "path"):
        roles_correct &= np.all(
            role_predictions[role] == labels[role],
            axis=1,
        )
        roles_empty &= np.all(
            role_predictions[role] == 0,
            axis=1,
        )

    active_strict = (
        graph_correct
        & active_truth
        & (count_prediction == labels["count"])
        & roles_correct
    )
    inactive_strict = (
        graph_correct
        & (~active_truth)
        & roles_empty
    )
    strict = active_strict | inactive_strict

    graph_and_count_active = (
        graph_correct[active_truth]
        & (
            count_prediction[active_truth]
            == labels["count"][active_truth]
        )
    )
    return {
        "strict_all_task_exactness": float(strict.mean()),
        "strict_active_exactness": float(
            active_strict[active_truth].mean()
        ),
        "strict_inactive_exactness": float(
            inactive_strict[~active_truth].mean()
        ),
        "graph_and_count_exact_active": float(
            graph_and_count_active.mean()
        ),
        "strict_exact_items": int(strict.sum()),
        "strict_active_exact_items": int(active_strict.sum()),
        "strict_inactive_exact_items": int(inactive_strict.sum()),
    }


def consistency_audit(
    graph_prediction: np.ndarray,
    count_prediction: np.ndarray,
    role_predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    predicted_active = graph_prediction == 1
    source = role_predictions["source"].astype(bool)
    transit = role_predictions["transit"].astype(bool)
    victim = role_predictions["victim"].astype(bool)
    path = role_predictions["path"].astype(bool)

    source_count = source.sum(axis=1)
    count_consistent = source_count == count_prediction
    source_victim_disjoint = ~np.any(source & victim, axis=1)
    transit_endpoint_disjoint = ~np.any(
        transit & (source | victim),
        axis=1,
    )
    endpoints_inside_path = np.all(
        ~(source | victim) | path,
        axis=1,
    )
    transit_inside_path = np.all(~transit | path, axis=1)
    path_equals_union = np.all(
        path == (source | transit | victim),
        axis=1,
    )

    def rate(values: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(values[mask].mean())

    return {
        "predicted_active_items": int(predicted_active.sum()),
        "source_count_equals_predicted_count_on_predicted_active": rate(
            count_consistent,
            predicted_active,
        ),
        "source_victim_disjoint_on_predicted_active": rate(
            source_victim_disjoint,
            predicted_active,
        ),
        "transit_endpoint_disjoint_on_predicted_active": rate(
            transit_endpoint_disjoint,
            predicted_active,
        ),
        "source_and_victim_inside_path_on_predicted_active": rate(
            endpoints_inside_path,
            predicted_active,
        ),
        "transit_inside_path_on_predicted_active": rate(
            transit_inside_path,
            predicted_active,
        ),
        "path_equals_source_union_transit_union_victim_on_predicted_active": rate(
            path_equals_union,
            predicted_active,
        ),
        "fully_consistent_on_predicted_active": rate(
            count_consistent
            & source_victim_disjoint
            & transit_endpoint_disjoint
            & endpoints_inside_path
            & transit_inside_path
            & path_equals_union,
            predicted_active,
        ),
    }


def evaluate_policy(
    name: str,
    graph_threshold: float,
    role_thresholds: dict[str, float],
    arrays: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    probabilities = {
        "graph": stable_sigmoid(arrays["attack_logits"]),
        "source": stable_sigmoid(arrays["source_logits"]),
        "transit": stable_sigmoid(arrays["transit_logits"]),
        "victim": stable_sigmoid(arrays["victim_logits"]),
        "path": stable_sigmoid(arrays["path_logits"]),
    }
    graph_prediction = (
        probabilities["graph"] >= graph_threshold
    ).astype(np.uint8)
    count_prediction = (
        np.argmax(arrays["count_logits"], axis=1) + 1
    ).astype(np.int8)
    role_predictions = {
        role: (
            probabilities[role] >= role_thresholds[role]
        ).astype(np.uint8)
        for role in ("source", "transit", "victim", "path")
    }

    active_truth = labels["graph"] == 1
    metrics = {
        "policy": name,
        "thresholds": {
            "graph": graph_threshold,
            **role_thresholds,
        },
        "graph": {
            **binary_metrics(
                graph_prediction,
                labels["graph"],
            ),
            "auroc_raw_logits": binary_auroc(
                arrays["attack_logits"],
                labels["graph"],
            ),
            "average_precision_raw_logits": (
                binary_average_precision(
                    arrays["attack_logits"],
                    labels["graph"],
                )
            ),
        },
        "count_active": multiclass_metrics_active(
            count_prediction[active_truth],
            labels["count"][active_truth],
        ),
        "roles": {
            role: role_metrics(
                role_predictions[role],
                labels[role],
                arrays[f"{role}_logits"],
                active_truth,
            )
            for role in ("source", "transit", "victim", "path")
        },
        "strict": strict_exactness(
            graph_prediction,
            count_prediction,
            role_predictions,
            labels,
        ),
        "cross_head_consistency": consistency_audit(
            graph_prediction,
            count_prediction,
            role_predictions,
        ),
        "graph_gating_applied": False,
        "route_legality_enforced": False,
        "count_source_consistency_enforced": False,
    }
    outputs = {
        "graph_prediction": graph_prediction,
        "count_prediction": count_prediction,
        "source_prediction": role_predictions["source"],
        "transit_prediction": role_predictions["transit"],
        "victim_prediction": role_predictions["victim"],
        "path_prediction": role_predictions["path"],
    }
    return metrics, outputs


def main() -> int:
    args = parse_args()

    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(
        args.installed_script
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    raw_outputs_path = output_dir / (
        "V5_P3_D2_RAW_NEURAL_VALIDATION_OUTPUTS.npz"
    )
    a0_outputs_path = output_dir / (
        "V5_P3_D2_FROZEN_P2_A0_VALIDATION_OUTPUTS.npz"
    )
    comparison_path = output_dir / (
        "V5_P3_D2_RAW_VS_FROZEN_A0_COMPARISON.json"
    )
    complete_path = output_dir / f"{STAGE}_COMPLETE"
    hold_path = output_dir / f"{STAGE}_HOLD"

    d1_dir = repo / "reports/v5/p3_d1_immutable_validation_logit_export"
    d1_report_path = d1_dir / (
        "V5_P3_D1_IMMUTABLE_TRANCHE_A_VALIDATION_LOGIT_EXPORT_REPORT.json"
    )
    d1_lock_path = d1_dir / (
        "V5_P3_D1_IMMUTABLE_TRANCHE_A_VALIDATION_LOGIT_EXPORT_LOCK.json"
    )
    export_path = d1_dir / (
        "V5_P3_D1_TRANCHE_A_VALIDATION_LOGITS_AND_LABELS.npz"
    )
    p2_evaluator_path = (
        repo
        / "scripts/v5/p2/evaluate_v5_p2_final_raw_a0_a1_one_shot.py"
    )
    l5_contract_path = (
        repo
        / "reports/v5/p2_l5_final_pretest_freeze/"
        "V5_P2_ONE_SHOT_BLIND_EVALUATION_CONTRACT.json"
    )

    required = [
        d1_report_path,
        d1_lock_path,
        export_path,
        p2_evaluator_path,
        l5_contract_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")

    d1_report = json.loads(
        d1_report_path.read_text(encoding="utf-8")
    )
    d1_lock = json.loads(
        d1_lock_path.read_text(encoding="utf-8")
    )
    if d1_report.get("status") != "PASS":
        raise RuntimeError("D1 report is not PASS")
    if not d1_report.get("decision", {}).get(
        "raw_and_A0_evaluation_authorized"
    ):
        raise RuntimeError("D1 did not authorize Raw/A0 evaluation")
    if d1_lock.get("report_sha256") != sha256_file(
        d1_report_path
    ):
        raise RuntimeError("D1 report/lock SHA mismatch")
    if d1_lock.get("export_sha256") != sha256_file(export_path):
        raise RuntimeError("D1 export SHA mismatch")
    if d1_report.get("sealed_test", {}).get(
        "test_tensor_loaded"
    ):
        raise RuntimeError("D1 reports sealed-test access")

    p2_module = import_source(
        p2_evaluator_path,
        "_v5_p3_d2_p2_frozen_policy_source",
    )
    module_thresholds = dict(p2_module.A0_THRESHOLDS)
    for key, expected in EXPECTED_A0_THRESHOLDS.items():
        actual = float(module_thresholds[key])
        if actual != expected:
            raise RuntimeError(
                f"P2 evaluator A0 threshold changed for {key}: "
                f"{actual} != {expected}"
            )

    with np.load(export_path, allow_pickle=False) as loaded:
        arrays = {
            key: loaded[key].copy()
            for key in loaded.files
        }

    expected_shapes = {
        "attack_logits": (EXPECTED_ITEMS,),
        "count_logits": (EXPECTED_ITEMS, 4),
        "source_logits": (EXPECTED_ITEMS, 16),
        "transit_logits": (EXPECTED_ITEMS, 16),
        "victim_logits": (EXPECTED_ITEMS, 16),
        "path_logits": (EXPECTED_ITEMS, 16),
        "y_attack": (EXPECTED_ITEMS,),
        "y_attacker_count": (EXPECTED_ITEMS,),
        "y_source": (EXPECTED_ITEMS, 16),
        "y_transit": (EXPECTED_ITEMS, 16),
        "y_victim": (EXPECTED_ITEMS, 16),
        "y_attack_path": (EXPECTED_ITEMS, 16),
    }
    for key, shape in expected_shapes.items():
        if key not in arrays:
            raise RuntimeError(f"D1 export missing {key}")
        if arrays[key].shape != shape:
            raise RuntimeError(
                f"{key} shape={arrays[key].shape}, expected={shape}"
            )

    for key in (
        "attack_logits",
        "count_logits",
        "source_logits",
        "transit_logits",
        "victim_logits",
        "path_logits",
    ):
        if not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"non-finite values in {key}")

    labels = {
        "graph": arrays["y_attack"].astype(np.uint8),
        "count": arrays["y_attacker_count"].astype(np.int8),
        "source": arrays["y_source"].astype(np.uint8),
        "transit": arrays["y_transit"].astype(np.uint8),
        "victim": arrays["y_victim"].astype(np.uint8),
        "path": arrays["y_attack_path"].astype(np.uint8),
    }

    raw_metrics, raw_outputs = evaluate_policy(
        "Raw neural threshold-0.5 heads",
        graph_threshold=0.5,
        role_thresholds={
            "source": 0.5,
            "transit": 0.5,
            "victim": 0.5,
            "path": 0.5,
        },
        arrays=arrays,
        labels=labels,
    )
    a0_metrics, a0_outputs = evaluate_policy(
        "Frozen P2 A0 threshold transfer",
        graph_threshold=EXPECTED_A0_THRESHOLDS["graph"],
        role_thresholds={
            role: EXPECTED_A0_THRESHOLDS[role]
            for role in ("source", "transit", "victim", "path")
        },
        arrays=arrays,
        labels=labels,
    )

    # Raw graph metrics must reproduce D1/A5 because Raw uses threshold 0.5.
    d1_metric = d1_report["metric_certification"]
    raw_consistency = {
        "graph_f1_delta_vs_D1": abs(
            raw_metrics["graph"]["f1"]
            - float(
                d1_metric["graph_threshold_0p5"]["f1"]
            )
        ),
        "graph_fpr_delta_vs_D1": abs(
            raw_metrics["graph"]["fpr"]
            - float(
                d1_metric["graph_threshold_0p5"]["fpr"]
            )
        ),
        "graph_auroc_delta_vs_D1": abs(
            raw_metrics["graph"]["auroc_raw_logits"]
            - float(
                d1_metric["selection_components"][
                    "graph_auroc"
                ]
            )
        ),
        "graph_ap_delta_vs_D1": abs(
            raw_metrics["graph"][
                "average_precision_raw_logits"
            ]
            - float(
                d1_metric["selection_components"][
                    "graph_average_precision"
                ]
            )
        ),
    }
    raw_consistency["maximum_delta"] = max(
        raw_consistency.values()
    )
    raw_consistency["pass"] = (
        raw_consistency["maximum_delta"] <= 1e-9
    )
    if not raw_consistency["pass"]:
        raise RuntimeError(
            f"Raw metric reproduction failed: {raw_consistency}"
        )

    comparison = {
        "stage": STAGE,
        "campaign_label": CAMPAIGN_LABEL,
        "classification": (
            "A-validation preliminary frozen-policy transfer"
        ),
        "raw": raw_metrics,
        "A0_frozen_P2_transfer": a0_metrics,
        "deltas_A0_minus_Raw": {
            "graph_accuracy": (
                a0_metrics["graph"]["accuracy"]
                - raw_metrics["graph"]["accuracy"]
            ),
            "graph_f1": (
                a0_metrics["graph"]["f1"]
                - raw_metrics["graph"]["f1"]
            ),
            "graph_fpr": (
                a0_metrics["graph"]["fpr"]
                - raw_metrics["graph"]["fpr"]
            ),
            "strict_all_task_exactness": (
                a0_metrics["strict"][
                    "strict_all_task_exactness"
                ]
                - raw_metrics["strict"][
                    "strict_all_task_exactness"
                ]
            ),
            "strict_active_exactness": (
                a0_metrics["strict"][
                    "strict_active_exactness"
                ]
                - raw_metrics["strict"][
                    "strict_active_exactness"
                ]
            ),
            "strict_inactive_exactness": (
                a0_metrics["strict"][
                    "strict_inactive_exactness"
                ]
                - raw_metrics["strict"][
                    "strict_inactive_exactness"
                ]
            ),
            "source_exact_active": (
                a0_metrics["roles"]["source"]["exact_active"]
                - raw_metrics["roles"]["source"]["exact_active"]
            ),
            "transit_exact_active": (
                a0_metrics["roles"]["transit"]["exact_active"]
                - raw_metrics["roles"]["transit"]["exact_active"]
            ),
            "victim_exact_active": (
                a0_metrics["roles"]["victim"]["exact_active"]
                - raw_metrics["roles"]["victim"]["exact_active"]
            ),
            "path_exact_active": (
                a0_metrics["roles"]["path"]["exact_active"]
                - raw_metrics["roles"]["path"]["exact_active"]
            ),
        },
        "interpretation_boundary": {
            "thresholds_transferred_without_tuning": True,
            "A0_is_not_route_legal_decoder": True,
            "A0_does_not_enforce_cross_head_consistency": True,
            "A1_exact_not_executed": True,
            "sealed_test_not_accessed": True,
            "generalization_claim_authorized": False,
        },
    }
    atomic_json(comparison_path, comparison)
    atomic_npz(raw_outputs_path, raw_outputs)
    atomic_npz(a0_outputs_path, a0_outputs)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "classification": (
            "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic"
        ),
        "scope": (
            "Raw threshold-0.5 and frozen-P2-A0 threshold "
            "evaluation on the immutable D1 A-validation export."
        ),
        "policy_source": {
            "P2_evaluator_path": str(p2_evaluator_path),
            "P2_evaluator_sha256": sha256_file(
                p2_evaluator_path
            ),
            "L5_contract_path": str(l5_contract_path),
            "L5_contract_sha256": sha256_file(
                l5_contract_path
            ),
            "A0_thresholds": EXPECTED_A0_THRESHOLDS,
            "threshold_tuning_performed": False,
        },
        "raw": raw_metrics,
        "A0_frozen_P2_transfer": a0_metrics,
        "comparison_path": str(comparison_path),
        "comparison_sha256": sha256_file(comparison_path),
        "outputs": {
            "raw_path": str(raw_outputs_path),
            "raw_sha256": sha256_file(raw_outputs_path),
            "A0_path": str(a0_outputs_path),
            "A0_sha256": sha256_file(a0_outputs_path),
        },
        "raw_reproduction": raw_consistency,
        "decision": {
            "raw_and_A0_evaluation_complete": True,
            "A0_improves_strict_all_task_exactness": (
                a0_metrics["strict"][
                    "strict_all_task_exactness"
                ]
                > raw_metrics["strict"][
                    "strict_all_task_exactness"
                ]
            ),
            "A0_is_route_legal": False,
            "A1_exact_resumable_preflight_authorized": True,
            "A1_exact_full_evaluation_authorized": False,
            "beam_decoder_selected": False,
            "sealed_test_evaluation_authorized": False,
            "next_stage": (
                "V5_P3_D3_RESUMABLE_A1_EXACT_DECODER_"
                "CHUNK_PREFLIGHT"
            ),
        },
        "sealed_test": {
            "test_dataset_instantiated": False,
            "test_length_computed": False,
            "test_tensor_loaded": False,
            "evaluation_authorized": False,
        },
        "provenance": {
            "d1_report_sha256": sha256_file(
                d1_report_path
            ),
            "d1_lock_sha256": sha256_file(
                d1_lock_path
            ),
            "D1_export_sha256": sha256_file(export_path),
            "installed_script_sha256": sha256_file(
                installed_script
            ),
        },
        "model_replayed": False,
        "model_trained": False,
        "decoder_A1_executed": False,
        "threshold_tuning_performed": False,
        "certified_dataset_modified": False,
        "generalization_claim_authorized": False,
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(report_path),
        "comparison_sha256": sha256_file(comparison_path),
        "raw_outputs_sha256": sha256_file(raw_outputs_path),
        "A0_outputs_sha256": sha256_file(a0_outputs_path),
        "D1_export_sha256": sha256_file(export_path),
        "P2_evaluator_sha256": sha256_file(
            p2_evaluator_path
        ),
        "A0_thresholds": EXPECTED_A0_THRESHOLDS,
        "items": EXPECTED_ITEMS,
        "A1_exact_executed": False,
        "test_tensor_loaded": False,
        "certified_dataset_modified": False,
    }
    atomic_json(lock_path, lock)

    complete_path.write_text(
        f"{STAGE}_COMPLETE\n",
        encoding="utf-8",
    )
    if hold_path.exists():
        hold_path.unlink()

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign_label={CAMPAIGN_LABEL}")
    print(f"items={EXPECTED_ITEMS}")
    print(
        "raw_graph_accuracy="
        f"{raw_metrics['graph']['accuracy']:.8f}"
    )
    print(
        "raw_graph_f1="
        f"{raw_metrics['graph']['f1']:.8f}"
    )
    print(
        "raw_graph_fpr="
        f"{raw_metrics['graph']['fpr']:.8f}"
    )
    print(
        "raw_strict_all_task_exactness="
        f"{raw_metrics['strict']['strict_all_task_exactness']:.8f}"
    )
    print(
        "A0_graph_accuracy="
        f"{a0_metrics['graph']['accuracy']:.8f}"
    )
    print(
        "A0_graph_f1="
        f"{a0_metrics['graph']['f1']:.8f}"
    )
    print(
        "A0_graph_fpr="
        f"{a0_metrics['graph']['fpr']:.8f}"
    )
    print(
        "A0_strict_all_task_exactness="
        f"{a0_metrics['strict']['strict_all_task_exactness']:.8f}"
    )
    print(
        "strict_delta_A0_minus_Raw="
        f"{comparison['deltas_A0_minus_Raw']['strict_all_task_exactness']:.8f}"
    )
    for policy_name, metrics in (
        ("raw", raw_metrics),
        ("A0", a0_metrics),
    ):
        print(
            f"{policy_name}_source_exact_active="
            f"{metrics['roles']['source']['exact_active']:.8f}"
        )
        print(
            f"{policy_name}_transit_exact_active="
            f"{metrics['roles']['transit']['exact_active']:.8f}"
        )
        print(
            f"{policy_name}_victim_exact_active="
            f"{metrics['roles']['victim']['exact_active']:.8f}"
        )
        print(
            f"{policy_name}_path_exact_active="
            f"{metrics['roles']['path']['exact_active']:.8f}"
        )
        print(
            f"{policy_name}_fully_consistent_predicted_active="
            f"{metrics['cross_head_consistency']['fully_consistent_on_predicted_active']:.8f}"
        )
    print("threshold_tuning_performed=false")
    print("A0_route_legality_enforced=false")
    print("A1_exact_executed=false")
    print("beam_decoder_selected=false")
    print("test_dataset_instantiated=false")
    print("test_length_computed=false")
    print("test_tensor_loaded=false")
    print("A1_exact_resumable_preflight_authorized=true")
    print("A1_exact_full_evaluation_authorized=false")
    print("sealed_test_evaluation_authorized=false")
    print(
        "next_stage="
        "V5_P3_D3_RESUMABLE_A1_EXACT_DECODER_CHUNK_PREFLIGHT"
    )
    print(f"comparison={comparison_path}")
    print(f"raw_outputs={raw_outputs_path}")
    print(f"A0_outputs={a0_outputs_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
