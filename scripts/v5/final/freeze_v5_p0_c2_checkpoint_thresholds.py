#!/usr/bin/env python3
"""
V5 P0-C2 Final Checkpoint and Threshold Freeze

Loads only:
- the C1-selected checkpoint;
- the C1-saved validation predictions and targets;
- prerequisite JSON/lock artifacts.

It does not construct any dataset and does not access the P0 test split.

The fixed C0 threshold grid and tie-breaking rules are applied exactly:
    0.10, 0.15, ..., 0.90

Outputs:
- calibrated validation threshold sweeps;
- copied and hash-locked final checkpoint;
- immutable threshold manifest;
- validation metrics at frozen thresholds;
- C2 report, lock, and completion marker.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch


STAGE = "V5_P0_C2_FINAL_CHECKPOINT_THRESHOLD_FREEZE"
COMPLETE = f"{STAGE}_COMPLETE"
ROLES = ("source", "transit", "victim", "path")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def binary_metrics(
    truth: torch.Tensor,
    prediction: torch.Tensor,
) -> dict[str, Any]:
    truth = truth.detach().cpu().bool()
    prediction = prediction.detach().cpu().bool()

    tp = int((truth & prediction).sum().item())
    tn = int((~truth & ~prediction).sum().item())
    fp = int((~truth & prediction).sum().item())
    fn = int((truth & ~prediction).sum().item())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    f1 = (
        2.0 * precision * recall
        / max(1e-12, precision + recall)
    )

    return {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "balanced_accuracy": 0.5 * (recall + tnr),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / max(1, fp + tn),
        "tnr": tnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support_negative": tn + fp,
        "support_positive": tp + fn,
    }


def multiclass_metrics(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    class_count: int = 3,
) -> dict[str, Any]:
    truth = truth.detach().cpu().long()
    prediction = prediction.detach().cpu().long()

    confusion = torch.zeros(
        class_count,
        class_count,
        dtype=torch.long,
    )
    for actual, predicted in zip(
        truth.tolist(),
        prediction.tolist(),
    ):
        confusion[actual, predicted] += 1

    f1_values: list[float] = []
    per_class: dict[str, Any] = {}
    for label in range(class_count):
        tp = int(confusion[label, label].item())
        fp = int(confusion[:, label].sum().item()) - tp
        fn = int(confusion[label, :].sum().item()) - tp
        support = int(confusion[label, :].sum().item())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (
            2.0 * precision * recall
            / max(1e-12, precision + recall)
        )
        f1_values.append(f1)
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "accuracy": float(
            (truth == prediction).float().mean().item()
        ),
        "macro_f1": sum(f1_values) / class_count,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def node_metrics(
    truth: torch.Tensor,
    prediction: torch.Tensor,
) -> dict[str, Any]:
    truth = truth.detach().cpu().bool()
    prediction = prediction.detach().cpu().bool()

    tp = int((truth & prediction).sum().item())
    tn = int((~truth & ~prediction).sum().item())
    fp = int((~truth & prediction).sum().item())
    fn = int((truth & ~prediction).sum().item())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        2.0 * precision * recall
        / max(1e-12, precision + recall)
    )
    exact = float(
        (truth == prediction)
        .all(dim=1)
        .float()
        .mean()
        .item()
    )

    return {
        "node_accuracy": (
            (tp + tn) / max(1, tp + tn + fp + fn)
        ),
        "node_precision": precision,
        "node_recall": recall,
        "node_f1": f1,
        "exact_set": exact,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_entries": int(truth.sum().item()),
        "window_count": int(truth.shape[0]),
    }


def validate_prediction_payload(
    payload: dict[str, Any],
    failures: list[str],
) -> dict[str, torch.Tensor]:
    values = payload.get("predictions_and_targets")
    if not isinstance(values, dict):
        failures.append(
            "validation prediction artifact lacks predictions_and_targets"
        )
        return {}

    required = {
        "attack_truth",
        "attack_probability",
        "count_truth",
        "count_prediction",
    }
    for role in ROLES:
        required.add(f"{role}_truth")
        required.add(f"{role}_probability")

    missing = sorted(required - set(values))
    if missing:
        failures.append(
            f"validation prediction artifact missing keys: {missing}"
        )

    tensors: dict[str, torch.Tensor] = {}
    for key in required:
        value = values.get(key)
        if not isinstance(value, torch.Tensor):
            failures.append(f"{key} is not a Tensor")
        else:
            tensors[key] = value.detach().cpu()

    if failures:
        return tensors

    window_count = int(tensors["attack_truth"].shape[0])
    if tensors["attack_probability"].shape != (window_count,):
        failures.append("attack_probability shape mismatch")
    if tensors["count_truth"].shape != (window_count,):
        failures.append("count_truth shape mismatch")
    if tensors["count_prediction"].shape != (window_count,):
        failures.append("count_prediction shape mismatch")

    for role in ROLES:
        truth = tensors[f"{role}_truth"]
        probability = tensors[f"{role}_probability"]
        if truth.shape != (window_count, 16):
            failures.append(
                f"{role}_truth shape={tuple(truth.shape)}, "
                f"expected [{window_count},16]"
            )
        if probability.shape != (window_count, 16):
            failures.append(
                f"{role}_probability shape={tuple(probability.shape)}, "
                f"expected [{window_count},16]"
            )

    for key, value in tensors.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            failures.append(f"{key} contains NaN or Inf")

    for key in (
        "attack_probability",
        "source_probability",
        "transit_probability",
        "victim_probability",
        "path_probability",
    ):
        value = tensors.get(key)
        if value is not None:
            minimum = float(value.min().item())
            maximum = float(value.max().item())
            if minimum < 0.0 or maximum > 1.0:
                failures.append(
                    f"{key} outside [0,1]: min={minimum}, max={maximum}"
                )

    return tensors


def metrics_at_thresholds(
    tensors: dict[str, torch.Tensor],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    attack_truth = tensors["attack_truth"].bool()
    attack_prediction = (
        tensors["attack_probability"] >= thresholds["attack"]
    )
    count_truth = tensors["count_truth"].long()
    count_prediction = tensors["count_prediction"].long()
    attack_windows = attack_truth

    metrics: dict[str, Any] = {
        "graph": binary_metrics(
            attack_truth,
            attack_prediction,
        ),
        "count": multiclass_metrics(
            count_truth,
            count_prediction,
        ),
        "roles": {},
    }

    all_exact = (
        attack_truth == attack_prediction
    ) & (
        count_truth == count_prediction
    )

    for role in ROLES:
        truth = tensors[f"{role}_truth"].bool()
        prediction = (
            tensors[f"{role}_probability"] >= thresholds[role]
        )
        metrics["roles"][role] = {
            "all_windows": node_metrics(truth, prediction),
            "attack_windows": node_metrics(
                truth[attack_windows],
                prediction[attack_windows],
            ),
        }
        all_exact &= (truth == prediction).all(dim=1)

    metrics["all_tasks_exact"] = float(
        all_exact.float().mean().item()
    )
    metrics["window_count"] = int(attack_truth.shape[0])
    return metrics


def compare_metric(
    actual: float,
    expected: float,
    name: str,
    failures: list[str],
    tolerance: float = 1e-9,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        failures.append(
            f"C1 validation reproduction mismatch for {name}: "
            f"actual={actual}, expected={expected}"
        )


def select_attack_threshold(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    eligible = [
        row for row in rows
        if float(row["fpr"]) <= 0.15 + 1e-12
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                float(row["recall"]),
                float(row["f1"]),
                float(row["balanced_accuracy"]),
                -abs(float(row["threshold"]) - 0.5),
            ),
        )
        return selected, "eligible_fpr_le_0_15"

    selected = max(
        rows,
        key=lambda row: (
            -float(row["fpr"]),
            float(row["recall"]),
            float(row["f1"]),
            -abs(float(row["threshold"]) - 0.5),
        ),
    )
    return selected, "fallback_no_threshold_met_fpr_cap"


def select_role_threshold(
    role: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if role in ("source", "victim"):
        return max(
            rows,
            key=lambda row: (
                float(row["node_f1"]),
                float(row["exact_set"]),
                float(row["node_precision"]),
                -abs(float(row["threshold"]) - 0.5),
            ),
        )
    return max(
        rows,
        key=lambda row: (
            float(row["node_f1"]),
            float(row["exact_set"]),
            float(row["node_recall"]),
            -abs(float(row["threshold"]) - 0.5),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b8-dir", type=Path, required=True)
    parser.add_argument("--c0-dir", type=Path, required=True)
    parser.add_argument("--c1-report-dir", type=Path, required=True)
    parser.add_argument("--c1-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()

    b8_dir = args.b8_dir.expanduser().resolve()
    c0_dir = args.c0_dir.expanduser().resolve()
    c1_report_dir = args.c1_report_dir.expanduser().resolve()
    c1_model_dir = args.c1_model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output exists: {output_dir}", file=sys.stderr)
        return 2
    if model_dir.exists():
        print(f"STOP: model directory exists: {model_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    paths = {
        "b8_report": (
            b8_dir
            / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE.json"
        ),
        "b8_lock": (
            b8_dir
            / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE_LOCK.json"
        ),
        "b8_contract": (
            b8_dir
            / "V5_P0_B8_B3_CONV1D_ONLY_ARCHITECTURE_CONTRACT.json"
        ),
        "b8_marker": (
            b8_dir
            / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE_COMPLETE"
        ),
        "c0_report": (
            c0_dir / "V5_P0_C0_FINAL_B3_TRAINING_PROTOCOL_LOCK.json"
        ),
        "c0_lock": (
            c0_dir
            / "V5_P0_C0_FINAL_B3_TRAINING_PROTOCOL_LOCK_LOCK.json"
        ),
        "c0_protocol": (
            c0_dir / "V5_P0_C0_FINAL_B3_PROTOCOL.json"
        ),
        "c0_marker": (
            c0_dir
            / "V5_P0_C0_FINAL_B3_TRAINING_PROTOCOL_LOCK_COMPLETE"
        ),
        "c1_report": (
            c1_report_dir
            / "V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING.json"
        ),
        "c1_lock": (
            c1_report_dir
            / "V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING_LOCK.json"
        ),
        "c1_marker": (
            c1_report_dir
            / "V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING_COMPLETE"
        ),
        "selected_checkpoint": (
            c1_model_dir
            / "selected"
            / "selected_best_checkpoint.pt"
        ),
        "selected_predictions": (
            c1_model_dir
            / "selected"
            / "selected_validation_predictions_and_targets.pt"
        ),
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    b8_report: dict[str, Any] = {}
    b8_lock: dict[str, Any] = {}
    b8_contract: dict[str, Any] = {}
    c0_report: dict[str, Any] = {}
    c0_lock: dict[str, Any] = {}
    protocol: dict[str, Any] = {}
    c1_report: dict[str, Any] = {}
    c1_lock: dict[str, Any] = {}

    if not failures:
        b8_report = load_json(paths["b8_report"])
        b8_lock = load_json(paths["b8_lock"])
        b8_contract = load_json(paths["b8_contract"])
        c0_report = load_json(paths["c0_report"])
        c0_lock = load_json(paths["c0_lock"])
        protocol = load_json(paths["c0_protocol"])
        c1_report = load_json(paths["c1_report"])
        c1_lock = load_json(paths["c1_lock"])

        for label, report in (
            ("B8", b8_report),
            ("C0", c0_report),
            ("C1", c1_report),
        ):
            if report.get("status") != "COMPLETE":
                failures.append(
                    f"{label} status={report.get('status')!r}, "
                    "expected COMPLETE"
                )

        if b8_report.get("decision") != "FREEZE_B3_CONV1D_ONLY":
            failures.append("B8 did not freeze B3")
        if (
            c0_report.get("decision")
            != "LOCK_FINAL_P0_B3_TRAINING_PROTOCOL"
        ):
            failures.append("C0 final protocol is not locked")
        if (
            c1_report.get("decision")
            != "SELECT_ONE_B3_CHECKPOINT_FOR_C2_CALIBRATION"
        ):
            failures.append("C1 did not authorize C2 calibration")

        for label, report in (
            ("B8", b8_report),
            ("C0", c0_report),
            ("C1", c1_report),
        ):
            if report.get("test_split_accessed") is not False:
                failures.append(f"{label} does not certify untouched test")

        if b8_lock.get("report_sha256") != sha256_file(paths["b8_report"]):
            failures.append("B8 report SHA mismatch")
        if c0_lock.get("report_sha256") != sha256_file(paths["c0_report"]):
            failures.append("C0 report SHA mismatch")
        if c1_lock.get("report_sha256") != sha256_file(paths["c1_report"]):
            failures.append("C1 report SHA mismatch")

        if (
            b8_lock.get("contract_file_sha256")
            != sha256_file(paths["b8_contract"])
        ):
            failures.append("B8 contract file SHA mismatch")
        if (
            c0_lock.get("protocol_file_sha256")
            != sha256_file(paths["c0_protocol"])
        ):
            failures.append("C0 protocol file SHA mismatch")

        protocol_without_hash = dict(protocol)
        stored_protocol_hash = protocol_without_hash.pop(
            "protocol_sha256",
            None,
        )
        if stored_protocol_hash != canonical_sha256(protocol_without_hash):
            failures.append("C0 canonical protocol hash mismatch")
        if c0_lock.get("protocol_sha256") != stored_protocol_hash:
            failures.append("C0 lock protocol hash mismatch")
        if c1_lock.get("protocol_sha256") != stored_protocol_hash:
            failures.append("C1 lock protocol hash mismatch")

        architecture_hash = b8_contract.get(
            "architecture_contract_sha256"
        )
        if (
            c0_lock.get("architecture_contract_sha256")
            != architecture_hash
        ):
            failures.append("C0 architecture hash mismatch")
        if (
            c1_lock.get("architecture_contract_sha256")
            != architecture_hash
        ):
            failures.append("C1 architecture hash mismatch")

        selected = c1_report.get("selected_checkpoint", {})
        expected_checkpoint_sha = selected.get(
            "copied_checkpoint_sha256"
        )
        expected_predictions_sha = selected.get(
            "validation_predictions_sha256"
        )
        actual_checkpoint_sha = sha256_file(
            paths["selected_checkpoint"]
        )
        actual_predictions_sha = sha256_file(
            paths["selected_predictions"]
        )

        if expected_checkpoint_sha != actual_checkpoint_sha:
            failures.append("selected checkpoint SHA mismatch vs C1 report")
        if (
            c1_lock.get("selected_checkpoint_sha256")
            != actual_checkpoint_sha
        ):
            failures.append("selected checkpoint SHA mismatch vs C1 lock")
        if expected_predictions_sha != actual_predictions_sha:
            failures.append("selected predictions SHA mismatch vs C1 report")
        if (
            c1_lock.get("selected_validation_predictions_sha256")
            != actual_predictions_sha
        ):
            failures.append("selected predictions SHA mismatch vs C1 lock")

        if c1_lock.get("selected_seed") != 117:
            failures.append(
                f"C1 selected seed={c1_lock.get('selected_seed')!r}, "
                "expected 117"
            )
        if c1_lock.get("selected_best_epoch") != 101:
            failures.append(
                "C1 selected best epoch is not 101"
            )

        grid = (
            protocol.get("threshold_calibration", {})
            .get("candidate_thresholds")
        )
        expected_grid = [
            round(0.10 + 0.05 * index, 2)
            for index in range(17)
        ]
        if grid != expected_grid:
            failures.append("C0 threshold grid is not 0.10..0.90 step 0.05")

    if failures:
        hold = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "training_performed": False,
            "dataset_constructed": False,
            "test_split_constructed": False,
            "test_split_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", hold)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    checkpoint = torch.load(
        paths["selected_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    prediction_payload = torch.load(
        paths["selected_predictions"],
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("seed") != 117:
        failures.append("selected checkpoint internal seed is not 117")
    if checkpoint.get("epoch") != 101:
        failures.append("selected checkpoint internal epoch is not 101")
    if checkpoint.get("parameter_count") != 43208:
        failures.append("selected checkpoint parameter count is not 43,208")
    if (
        checkpoint.get("protocol_sha256")
        != protocol["protocol_sha256"]
    ):
        failures.append("selected checkpoint protocol hash mismatch")
    if (
        checkpoint.get("architecture_contract_sha256")
        != b8_contract["architecture_contract_sha256"]
    ):
        failures.append("selected checkpoint architecture hash mismatch")
    if checkpoint.get("test_split_accessed") is not False:
        failures.append("selected checkpoint does not certify untouched test")
    if prediction_payload.get("seed") != 117:
        failures.append("validation predictions internal seed is not 117")
    if prediction_payload.get("epoch") != 101:
        failures.append("validation predictions internal epoch is not 101")
    if prediction_payload.get("test_split_accessed") is not False:
        failures.append(
            "validation predictions do not certify untouched test"
        )

    tensors = validate_prediction_payload(
        prediction_payload,
        failures,
    )

    if failures:
        hold = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "training_performed": False,
            "dataset_constructed": False,
            "test_split_constructed": False,
            "test_split_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", hold)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    baseline_thresholds = {
        "attack": 0.5,
        "source": 0.5,
        "transit": 0.5,
        "victim": 0.5,
        "path": 0.5,
    }
    baseline_metrics = metrics_at_thresholds(
        tensors,
        baseline_thresholds,
    )

    expected_metrics = (
        c1_report["selected_checkpoint"]["validation_metrics"]
    )
    compare_metric(
        baseline_metrics["graph"]["balanced_accuracy"],
        expected_metrics["graph"]["balanced_accuracy"],
        "graph balanced accuracy",
        failures,
    )
    compare_metric(
        baseline_metrics["graph"]["recall"],
        expected_metrics["graph"]["recall"],
        "graph recall",
        failures,
    )
    compare_metric(
        baseline_metrics["graph"]["f1"],
        expected_metrics["graph"]["f1"],
        "graph F1",
        failures,
    )
    compare_metric(
        baseline_metrics["graph"]["fpr"],
        expected_metrics["graph"]["fpr"],
        "graph FPR",
        failures,
    )
    compare_metric(
        baseline_metrics["count"]["macro_f1"],
        expected_metrics["count"]["macro_f1"],
        "count macro F1",
        failures,
    )
    compare_metric(
        baseline_metrics["all_tasks_exact"],
        expected_metrics["all_tasks_exact"],
        "all-task exact",
        failures,
    )
    for role in ROLES:
        compare_metric(
            baseline_metrics["roles"][role]["attack_windows"]["node_f1"],
            expected_metrics["roles"][role]["attack_windows"]["node_f1"],
            f"{role} attack-window F1",
            failures,
        )

    if failures:
        hold = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "training_performed": False,
            "dataset_constructed": False,
            "test_split_constructed": False,
            "test_split_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", hold)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    grid = protocol["threshold_calibration"]["candidate_thresholds"]
    attack_rows: list[dict[str, Any]] = []
    attack_truth = tensors["attack_truth"].bool()

    for threshold in grid:
        metrics = binary_metrics(
            attack_truth,
            tensors["attack_probability"] >= threshold,
        )
        attack_rows.append(
            {
                "task": "attack",
                "threshold": threshold,
                "eligible_fpr_le_0_15": (
                    metrics["fpr"] <= 0.15 + 1e-12
                ),
                **metrics,
            }
        )

    selected_attack_row, attack_selection_mode = (
        select_attack_threshold(attack_rows)
    )
    selected_thresholds: dict[str, float] = {
        "attack": float(selected_attack_row["threshold"])
    }

    attack_windows = attack_truth
    role_rows: list[dict[str, Any]] = []
    role_selected_rows: dict[str, dict[str, Any]] = {}

    for role in ROLES:
        truth = tensors[f"{role}_truth"].bool()[attack_windows]
        probability = tensors[
            f"{role}_probability"
        ][attack_windows]

        rows_for_role: list[dict[str, Any]] = []
        for threshold in grid:
            metrics = node_metrics(
                truth,
                probability >= threshold,
            )
            row = {
                "task": role,
                "threshold": threshold,
                "calibration_scope": (
                    "ground_truth_validation_attack_windows"
                ),
                **metrics,
            }
            rows_for_role.append(row)
            role_rows.append(row)

        selected_row = select_role_threshold(
            role,
            rows_for_role,
        )
        role_selected_rows[role] = selected_row
        selected_thresholds[role] = float(
            selected_row["threshold"]
        )

    calibrated_metrics = metrics_at_thresholds(
        tensors,
        selected_thresholds,
    )

    for row in attack_rows:
        row["selected"] = (
            float(row["threshold"])
            == selected_thresholds["attack"]
        )
    for row in role_rows:
        row["selected"] = (
            float(row["threshold"])
            == selected_thresholds[str(row["task"])]
        )

    attack_sweep_path = (
        output_dir / "V5_P0_C2_ATTACK_THRESHOLD_SWEEP.csv"
    )
    role_sweep_path = (
        output_dir / "V5_P0_C2_ROLE_THRESHOLD_SWEEP.csv"
    )
    write_csv(attack_sweep_path, attack_rows)
    write_csv(role_sweep_path, role_rows)

    final_checkpoint_path = (
        model_dir / "final_locked_b3_checkpoint.pt"
    )
    frozen_predictions_path = (
        model_dir
        / "frozen_validation_predictions_and_targets.pt"
    )
    shutil.copy2(
        paths["selected_checkpoint"],
        final_checkpoint_path,
    )
    shutil.copy2(
        paths["selected_predictions"],
        frozen_predictions_path,
    )

    threshold_manifest = {
        "stage": STAGE,
        "status": "FROZEN",
        "checkpoint": {
            "seed": 117,
            "epoch": 101,
            "parameter_count": 43208,
            "source_path": str(paths["selected_checkpoint"]),
            "source_sha256": sha256_file(
                paths["selected_checkpoint"]
            ),
            "frozen_path": str(final_checkpoint_path),
            "frozen_sha256": sha256_file(
                final_checkpoint_path
            ),
        },
        "threshold_grid": grid,
        "frozen_thresholds": selected_thresholds,
        "attacker_count_rule": "argmax",
        "graph_gating_of_role_predictions": False,
        "selection_details": {
            "attack": {
                "mode": attack_selection_mode,
                "selected_row": selected_attack_row,
            },
            **{
                role: {
                    "selected_row": role_selected_rows[role],
                    "priority": (
                        [
                            "node_f1",
                            "exact_set",
                            "node_precision",
                            "closest_to_0.5",
                        ]
                        if role in ("source", "victim")
                        else [
                            "node_f1",
                            "exact_set",
                            "node_recall",
                            "closest_to_0.5",
                        ]
                    ),
                }
                for role in ROLES
            },
        },
        "validation_metrics_at_threshold_0_5": baseline_metrics,
        "validation_metrics_at_frozen_thresholds": (
            calibrated_metrics
        ),
        "architecture_contract_sha256": (
            b8_contract["architecture_contract_sha256"]
        ),
        "protocol_sha256": protocol["protocol_sha256"],
        "selected_validation_predictions_sha256": sha256_file(
            frozen_predictions_path
        ),
        "training_performed": False,
        "dataset_constructed": False,
        "test_split_constructed": False,
        "test_split_accessed": False,
    }
    threshold_manifest["threshold_manifest_sha256"] = (
        canonical_sha256(threshold_manifest)
    )

    manifest_path = (
        output_dir / "V5_P0_C2_FROZEN_CHECKPOINT_THRESHOLDS.json"
    )
    write_json(manifest_path, threshold_manifest)

    delta = {
        "graph_balanced_accuracy": (
            calibrated_metrics["graph"]["balanced_accuracy"]
            - baseline_metrics["graph"]["balanced_accuracy"]
        ),
        "graph_recall": (
            calibrated_metrics["graph"]["recall"]
            - baseline_metrics["graph"]["recall"]
        ),
        "graph_f1": (
            calibrated_metrics["graph"]["f1"]
            - baseline_metrics["graph"]["f1"]
        ),
        "graph_fpr": (
            calibrated_metrics["graph"]["fpr"]
            - baseline_metrics["graph"]["fpr"]
        ),
        "count_macro_f1": 0.0,
        "source_f1_attack": (
            calibrated_metrics["roles"]["source"]["attack_windows"]["node_f1"]
            - baseline_metrics["roles"]["source"]["attack_windows"]["node_f1"]
        ),
        "transit_f1_attack": (
            calibrated_metrics["roles"]["transit"]["attack_windows"]["node_f1"]
            - baseline_metrics["roles"]["transit"]["attack_windows"]["node_f1"]
        ),
        "victim_f1_attack": (
            calibrated_metrics["roles"]["victim"]["attack_windows"]["node_f1"]
            - baseline_metrics["roles"]["victim"]["attack_windows"]["node_f1"]
        ),
        "path_f1_attack": (
            calibrated_metrics["roles"]["path"]["attack_windows"]["node_f1"]
            - baseline_metrics["roles"]["path"]["attack_windows"]["node_f1"]
        ),
        "all_tasks_exact": (
            calibrated_metrics["all_tasks_exact"]
            - baseline_metrics["all_tasks_exact"]
        ),
    }

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "decision": "FREEZE_SELECTED_B3_CHECKPOINT_AND_THRESHOLDS",
        "selected_checkpoint": threshold_manifest["checkpoint"],
        "frozen_thresholds": selected_thresholds,
        "validation_metrics_at_threshold_0_5": baseline_metrics,
        "validation_metrics_at_frozen_thresholds": (
            calibrated_metrics
        ),
        "calibration_delta_vs_0_5": delta,
        "threshold_manifest_sha256": (
            threshold_manifest["threshold_manifest_sha256"]
        ),
        "artifacts": {
            "threshold_manifest": [
                str(manifest_path),
                sha256_file(manifest_path),
            ],
            "attack_threshold_sweep": [
                str(attack_sweep_path),
                sha256_file(attack_sweep_path),
            ],
            "role_threshold_sweep": [
                str(role_sweep_path),
                sha256_file(role_sweep_path),
            ],
            "final_locked_checkpoint": [
                str(final_checkpoint_path),
                sha256_file(final_checkpoint_path),
            ],
            "frozen_validation_predictions": [
                str(frozen_predictions_path),
                sha256_file(frozen_predictions_path),
            ],
        },
        "provenance": {
            "b8_report_sha256": sha256_file(paths["b8_report"]),
            "b8_lock_sha256": sha256_file(paths["b8_lock"]),
            "b8_contract_file_sha256": sha256_file(
                paths["b8_contract"]
            ),
            "c0_report_sha256": sha256_file(paths["c0_report"]),
            "c0_lock_sha256": sha256_file(paths["c0_lock"]),
            "c0_protocol_file_sha256": sha256_file(
                paths["c0_protocol"]
            ),
            "c1_report_sha256": sha256_file(paths["c1_report"]),
            "c1_lock_sha256": sha256_file(paths["c1_lock"]),
            "source_checkpoint_sha256": sha256_file(
                paths["selected_checkpoint"]
            ),
            "source_predictions_sha256": sha256_file(
                paths["selected_predictions"]
            ),
            "c2_script_sha256": sha256_file(Path(__file__)),
        },
        "training_performed": False,
        "dataset_constructed": False,
        "validation_predictions_reused": True,
        "test_split_constructed": False,
        "test_split_accessed": False,
        "test_performance_evaluated": False,
        "failures": failures,
        "warnings": warnings,
        "next_stage": (
            "V5_P0_C3_ONE_SHOT_LOCKED_TEST_EVALUATION"
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    lock = {
        "status": COMPLETE,
        "decision": "FREEZE_SELECTED_B3_CHECKPOINT_AND_THRESHOLDS",
        "report_sha256": sha256_file(report_path),
        "threshold_manifest_file_sha256": sha256_file(
            manifest_path
        ),
        "threshold_manifest_sha256": (
            threshold_manifest["threshold_manifest_sha256"]
        ),
        "final_checkpoint_sha256": sha256_file(
            final_checkpoint_path
        ),
        "frozen_validation_predictions_sha256": sha256_file(
            frozen_predictions_path
        ),
        "frozen_thresholds": selected_thresholds,
        "architecture_contract_sha256": (
            b8_contract["architecture_contract_sha256"]
        ),
        "protocol_sha256": protocol["protocol_sha256"],
        "c2_script_sha256": sha256_file(Path(__file__)),
        "training_performed": False,
        "dataset_constructed": False,
        "test_split_constructed": False,
        "test_split_accessed": False,
        "next_stage": (
            "V5_P0_C3_ONE_SHOT_LOCKED_TEST_EVALUATION"
        ),
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    base_graph = baseline_metrics["graph"]
    cal_graph = calibrated_metrics["graph"]

    print("===== V5 P0-C2 FINAL CHECKPOINT + THRESHOLD FREEZE =====")
    print("status: COMPLETE")
    print("selected_seed: 117")
    print("selected_epoch: 101")
    print("parameter_count: 43208")
    print(
        "final_checkpoint_sha256:",
        sha256_file(final_checkpoint_path),
    )
    print("frozen_thresholds:", selected_thresholds)
    print(
        "baseline_graph_balanced_accuracy:",
        f"{base_graph['balanced_accuracy']:.4f}",
    )
    print(
        "calibrated_graph_balanced_accuracy:",
        f"{cal_graph['balanced_accuracy']:.4f}",
    )
    print(
        "baseline_graph_recall:",
        f"{base_graph['recall']:.4f}",
    )
    print(
        "calibrated_graph_recall:",
        f"{cal_graph['recall']:.4f}",
    )
    print(
        "baseline_graph_f1:",
        f"{base_graph['f1']:.4f}",
    )
    print(
        "calibrated_graph_f1:",
        f"{cal_graph['f1']:.4f}",
    )
    print(
        "baseline_graph_fpr:",
        f"{base_graph['fpr']:.4f}",
    )
    print(
        "calibrated_graph_fpr:",
        f"{cal_graph['fpr']:.4f}",
    )
    print(
        "baseline_source_f1_attack:",
        f"{baseline_metrics['roles']['source']['attack_windows']['node_f1']:.4f}",
    )
    print(
        "calibrated_source_f1_attack:",
        f"{calibrated_metrics['roles']['source']['attack_windows']['node_f1']:.4f}",
    )
    print(
        "baseline_victim_f1_attack:",
        f"{baseline_metrics['roles']['victim']['attack_windows']['node_f1']:.4f}",
    )
    print(
        "calibrated_victim_f1_attack:",
        f"{calibrated_metrics['roles']['victim']['attack_windows']['node_f1']:.4f}",
    )
    print(
        "baseline_all_tasks_exact:",
        f"{baseline_metrics['all_tasks_exact']:.4f}",
    )
    print(
        "calibrated_all_tasks_exact:",
        f"{calibrated_metrics['all_tasks_exact']:.4f}",
    )
    print(
        "threshold_manifest_sha256:",
        threshold_manifest["threshold_manifest_sha256"],
    )
    print("training_performed: false")
    print("dataset_constructed: false")
    print("test_split_constructed: false")
    print("test_split_accessed: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print(
        "next_stage: V5_P0_C3_ONE_SHOT_LOCKED_TEST_EVALUATION"
    )
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
