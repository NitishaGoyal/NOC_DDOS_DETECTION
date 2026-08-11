#!/usr/bin/env python3
"""
V5 P2-B0-R3 Corrected Non-Test Label and Shortcut Audit

This stage performs a complete TRAIN/VALIDATION audit over the frozen
pair-aligned window population. It never enumerates or opens runs/test.

Checks
------
1. Every A1-R2 aligned target from all 489 non-test matched pairs.
2. Graph/count/source/transit/victim/path binary rules and role-mask bit-field consistency
   rules.
3. ATTACK/CONTROL balance and identical eligible window counts per pair.
4. Count-category mapping {K1:1, K2:2, K3:3, K4:4}.
5. Trivial and shortcut baselines:
   - always normal / always attack / train-majority graph
   - most-common active attacker count
   - most-common source router
   - highest observed traffic router
   - physical-mask-only role priors
   - epoch-position-only graph threshold
   - common-length-only graph threshold
   - quarantined status-flags-only graph threshold
   - quarantined mode-only graph baseline
   - quarantined category-only count baseline

Interpretation
--------------
Strong mask-only or epoch-only validation performance is a warning or blocker,
not a success. Status, mode, category, lengths, and window positions remain
forbidden learned inputs regardless of baseline performance.

The audit reads every train and validation run tensor sequentially, but does
not construct or access the P2 test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


STAGE = "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT"
COMPLETE = f"{STAGE}_COMPLETE"

WINDOW = 32
STRIDE = 8
PRIMARY58_INDICES = (
    list(range(0, 30))
    + list(range(31, 56))
    + [62, 64, 65]
)
EXPECTED_ITEM_KEYS = {
    "x",
    "physical_port_mask",
    "y_attack",
    "y_attacker_count",
    "y_source",
    "y_transit",
    "y_victim",
    "y_attack_path",
    "role_mask",
}

# Strong-shortcut gates.
EPOCH_POSITION_BLOCK_BA = 0.80
MASK_ROLE_BLOCK_AUROC = 0.90

# Warning levels.
EPOCH_POSITION_WARN_BA = 0.65
MASK_ROLE_WARN_AUROC = 0.75
COMMON_SOURCE_WARN_HIT1 = 0.50
TRAFFIC_SOURCE_WARN_HIT1 = 0.75


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


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid integer field {key!r} for "
            f"{row.get('split')}/{row.get('pair_key')}"
        ) from exc


def category_count(pair_key: str) -> int | None:
    match = re.search(r"-K([1-4])-", pair_key)
    return int(match.group(1)) if match else None


def is_binary_tensor(value: torch.Tensor) -> bool:
    if value.numel() == 0:
        return True
    return bool(((value == 0) | (value == 1)).all().item())


def confusion_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    y = truth.astype(np.int64).reshape(-1)
    p = prediction.astype(np.int64).reshape(-1)

    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())

    accuracy = (tp + tn) / max(1, len(y))
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * tpr / max(1e-12, precision + tpr)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "precision": precision,
        "recall": tpr,
        "f1": f1,
        "fpr": fp / max(1, fp + tn),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def binary_auroc(
    truth: np.ndarray,
    score: np.ndarray,
) -> float | None:
    y = truth.astype(np.int64).reshape(-1)
    s = score.astype(np.float64).reshape(-1)

    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())
    if positive_count == 0 or negative_count == 0:
        return None

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)

    start = 0
    while start < len(s):
        stop = start + 1
        while (
            stop < len(s)
            and sorted_scores[stop] == sorted_scores[start]
        ):
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_rank
        start = stop

    positive_rank_sum = ranks[y == 1].sum()
    return float(
        (
            positive_rank_sum
            - positive_count * (positive_count + 1) / 2
        )
        / (positive_count * negative_count)
    )


def average_precision(
    truth: np.ndarray,
    score: np.ndarray,
) -> float | None:
    y = truth.astype(np.int64).reshape(-1)
    s = score.astype(np.float64).reshape(-1)

    positive_count = int((y == 1).sum())
    if positive_count == 0:
        return None

    order = np.argsort(-s, kind="mergesort")
    sorted_truth = y[order]
    cumulative_positive = np.cumsum(sorted_truth)
    positions = np.arange(1, len(y) + 1)
    precision_at_rank = cumulative_positive / positions

    return float(
        (precision_at_rank * sorted_truth).sum() / positive_count
    )


def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)].astype(np.float64)
    if finite.size == 0:
        return np.array([0.0], dtype=np.float64)

    if finite.size <= 512:
        unique = np.unique(finite)
    else:
        quantiles = np.linspace(0.0, 1.0, 257)
        unique = np.unique(np.quantile(finite, quantiles))

    if unique.size == 1:
        epsilon = max(1e-6, abs(float(unique[0])) * 1e-6)
        return np.array(
            [
                unique[0] - epsilon,
                unique[0],
                unique[0] + epsilon,
            ],
            dtype=np.float64,
        )

    midpoints = 0.5 * (unique[:-1] + unique[1:])
    epsilon_low = max(1e-6, abs(float(unique[0])) * 1e-6)
    epsilon_high = max(1e-6, abs(float(unique[-1])) * 1e-6)

    return np.concatenate(
        (
            [unique[0] - epsilon_low],
            unique,
            midpoints,
            [unique[-1] + epsilon_high],
        )
    )


def fit_threshold(
    train_values: np.ndarray,
    train_truth: np.ndarray,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None

    for threshold in candidate_thresholds(train_values):
        for direction in ("ge", "le"):
            prediction = (
                train_values >= threshold
                if direction == "ge"
                else train_values <= threshold
            )
            metrics = confusion_metrics(train_truth, prediction)

            candidate = {
                "threshold": float(threshold),
                "direction": direction,
                "train_metrics": metrics,
            }
            score = (
                metrics["balanced_accuracy"],
                metrics["f1"],
                metrics["accuracy"],
                -abs(float(threshold)),
                1 if direction == "ge" else 0,
            )
            if best is None or score > best["_score"]:
                best = {**candidate, "_score": score}

    assert best is not None
    best.pop("_score")
    return best


def apply_threshold(
    values: np.ndarray,
    threshold: float,
    direction: str,
) -> np.ndarray:
    if direction == "ge":
        return values >= threshold
    if direction == "le":
        return values <= threshold
    raise ValueError(f"unknown direction: {direction}")


def source_single_prediction_metrics(
    truth: np.ndarray,
    predicted_node: np.ndarray,
) -> dict[str, Any]:
    true_matrix = truth.astype(bool)
    predicted = predicted_node.astype(np.int64)

    hit = true_matrix[
        np.arange(true_matrix.shape[0]),
        predicted,
    ]
    true_positive = int(hit.sum())
    false_positive = int(len(hit) - true_positive)
    total_true = int(true_matrix.sum())
    false_negative = total_true - true_positive

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "hit_at_1": float(hit.mean()),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "active_item_count": int(len(hit)),
        "true_source_count": total_true,
    }


def mask_signature(mask: torch.Tensor) -> np.ndarray:
    weights = (
        2
        ** torch.arange(
            mask.shape[1],
            dtype=torch.int64,
        )
    )
    return (
        mask.to(torch.int64) * weights.view(1, -1)
    ).sum(dim=1).cpu().numpy()


def role_mask_prior_baseline(
    train_truth: np.ndarray,
    validation_truth: np.ndarray,
    node_signatures: np.ndarray,
) -> dict[str, Any]:
    train_y = train_truth.astype(np.int64)
    validation_y = validation_truth.astype(np.int64)

    signature_rates: dict[int, float] = {}
    for signature in sorted(set(int(item) for item in node_signatures)):
        node_selector = node_signatures == signature
        signature_rates[signature] = float(
            train_y[:, node_selector].mean()
        )

    train_scores_by_node = np.array(
        [
            signature_rates[int(signature)]
            for signature in node_signatures
        ],
        dtype=np.float64,
    )
    validation_scores_by_node = train_scores_by_node.copy()

    train_scores = np.broadcast_to(
        train_scores_by_node,
        train_y.shape,
    ).reshape(-1)
    validation_scores = np.broadcast_to(
        validation_scores_by_node,
        validation_y.shape,
    ).reshape(-1)

    train_flat = train_y.reshape(-1)
    validation_flat = validation_y.reshape(-1)

    threshold_fit = fit_threshold(train_scores, train_flat)
    validation_prediction = apply_threshold(
        validation_scores,
        threshold_fit["threshold"],
        threshold_fit["direction"],
    )

    return {
        "signature_positive_rates": {
            str(key): value
            for key, value in signature_rates.items()
        },
        "threshold": threshold_fit["threshold"],
        "direction": threshold_fit["direction"],
        "train_metrics": threshold_fit["train_metrics"],
        "validation_metrics": confusion_metrics(
            validation_flat,
            validation_prediction,
        ),
        "validation_auroc": binary_auroc(
            validation_flat,
            validation_scores,
        ),
        "validation_average_precision": average_precision(
            validation_flat,
            validation_scores,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a0-dir", type=Path, required=True)
    parser.add_argument("--a1-r2-dir", type=Path, required=True)
    parser.add_argument("--a2-r2-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--b0-r1b-dir", type=Path, required=True)
    parser.add_argument("--b0-r2-dir", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root_input = args.root.expanduser()
    root = root_input.resolve()
    a0_dir = args.a0_dir.expanduser().resolve()
    a1_r2_dir = args.a1_r2_dir.expanduser().resolve()
    a2_r2_dir = args.a2_r2_dir.expanduser().resolve()
    a3_dir = args.a3_dir.expanduser().resolve()
    b0_r1b_dir = args.b0_r1b_dir.expanduser().resolve()
    b0_r2_dir = args.b0_r2_dir.expanduser().resolve()
    loader_path = args.loader_path.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(
            f"STOP: output already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []
    shortcut_blocks: list[str] = []
    shortcut_warnings: list[str] = []

    paths = {
        "a0_report": (
            a0_dir
            / "V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT.json"
        ),
        "a0_lock": (
            a0_dir
            / "V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT_LOCK.json"
        ),
        "a1_r2_report": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT.json"
        ),
        "a1_r2_lock": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT_LOCK.json"
        ),
        "pair_manifest": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
        ),
        "a2_r2_report": (
            a2_r2_dir
            / "V5_P2_A2_R2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT.json"
        ),
        "a2_r2_lock": (
            a2_r2_dir
            / "V5_P2_A2_R2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT_LOCK.json"
        ),
        "a2_r2_mask": (
            a2_r2_dir
            / "V5_P2_A2_R2_TOPOLOGY_DERIVED_BOOLEAN_PORT_MASK.pt"
        ),
        "a3_report": (
            a3_dir
            / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT.json"
        ),
        "a3_lock": (
            a3_dir
            / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT_LOCK.json"
        ),
        "b0_r1b_report": (
            b0_r1b_dir
            / "V5_P2_B0_R1B_K3_COUNT_AND_ROLE_MASK_SEMANTICS_AUDIT.json"
        ),
        "b0_r1b_lock": (
            b0_r1b_dir
            / "V5_P2_B0_R1B_K3_COUNT_AND_ROLE_MASK_SEMANTICS_AUDIT_LOCK.json"
        ),
        "b0_r2_report": (
            b0_r2_dir
            / "V5_P2_B0_R2_COUNT_HEAD_EXPANSION_AND_A4_REPEAT.json"
        ),
        "b0_r2_lock": (
            b0_r2_dir
            / "V5_P2_B0_R2_COUNT_HEAD_EXPANSION_AND_A4_REPEAT_LOCK.json"
        ),
        "loader": loader_path,
        "model": model_path,
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    for split in ("train", "validation"):
        if not (root / "runs" / split).is_dir():
            failures.append(f"missing runs/{split}")

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        return 1

    reports = {
        "A0": load_json(paths["a0_report"]),
        "A1-R2": load_json(paths["a1_r2_report"]),
        "A2-R2": load_json(paths["a2_r2_report"]),
        "A3": load_json(paths["a3_report"]),
        "B0-R1B": load_json(paths["b0_r1b_report"]),
        "B0-R2": load_json(paths["b0_r2_report"]),
    }
    locks = {
        "A0": load_json(paths["a0_lock"]),
        "A1-R2": load_json(paths["a1_r2_lock"]),
        "A2-R2": load_json(paths["a2_r2_lock"]),
        "A3": load_json(paths["a3_lock"]),
        "B0-R1B": load_json(paths["b0_r1b_lock"]),
        "B0-R2": load_json(paths["b0_r2_lock"]),
    }

    report_paths = {
        "A0": paths["a0_report"],
        "A1-R2": paths["a1_r2_report"],
        "A2-R2": paths["a2_r2_report"],
        "A3": paths["a3_report"],
        "B0-R1B": paths["b0_r1b_report"],
        "B0-R2": paths["b0_r2_report"],
    }

    for label in reports:
        if reports[label].get("status") != "COMPLETE":
            failures.append(f"{label} status is not COMPLETE")
        if (
            locks[label].get("report_sha256")
            != sha256_file(report_paths[label])
        ):
            failures.append(f"{label} report SHA mismatch")
        if reports[label].get("security_boundary", {}).get(
            "test_tensor_contents_accessed"
        ) is not False:
            failures.append(
                f"{label} does not certify untouched test tensors"
            )

    if locks["A1-R2"].get("aligned_windows") != 82694:
        failures.append("A1-R2 aligned-window count changed")
    if locks["A1-R2"].get("pair_count") != 489:
        failures.append("A1-R2 pair count changed")
    if (
        locks["A1-R2"].get("pair_manifest_sha256")
        != sha256_file(paths["pair_manifest"])
    ):
        failures.append("A1-R2 pair manifest SHA mismatch")
    if (
        locks["A2-R2"].get("corrected_port_mask_sha256")
        != sha256_file(paths["a2_r2_mask"])
    ):
        failures.append("A2-R2 port-mask SHA mismatch")
    if locks["A2-R2"].get("primary58_indices") != PRIMARY58_INDICES:
        failures.append("A2-R2 PRIMARY58 indices changed")
    if locks["A2-R2"].get(
        "second_normalization_forbidden"
    ) is not True:
        failures.append("second-normalization contract changed")
    if locks["A3"].get("loader_sha256") != sha256_file(paths["loader"]):
        failures.append("A3 loader SHA mismatch")
    if set(locks["A3"].get("item_keys", [])) != EXPECTED_ITEM_KEYS:
        failures.append("A3 loader item-key contract changed")
    expected_r1b_decision = (
        "RESOLVE_B0_HOLD_AS_K3_OMISSION_AND_"
        "ROLE_MASK_SEMANTIC_MISCLASSIFICATION_"
        "REQUIRE_COUNT_HEAD_3_TO_4_AND_REPEAT_A4"
    )
    if locks["B0-R1B"].get("decision") != expected_r1b_decision:
        failures.append("B0-R1B decision changed")
    if locks["B0-R1B"].get("observed_categories") != [1, 2, 3, 4]:
        failures.append("B0-R1B category contract changed")
    if locks["B0-R1B"].get("observed_active_counts") != [1, 2, 3, 4]:
        failures.append("B0-R1B count contract changed")
    if (
        locks["B0-R1B"].get("role_mask_semantics")
        != "bitfield3_source1_transit2_victim4"
    ):
        failures.append("B0-R1B role-mask contract changed")

    if locks["B0-R2"].get("all_tasks_all_exact") is not True:
        failures.append("B0-R2 did not lock exact four-count tiny overfit")
    if locks["B0-R2"].get("audit_weights_saved") is not False:
        failures.append("B0-R2 unexpectedly saved audit weights")
    if locks["B0-R2"].get("parameter_count") != 43273:
        failures.append("B0-R2 parameter count changed")
    if locks["B0-R2"].get("count_logits") != 4:
        failures.append("B0-R2 count-head width changed")
    if locks["B0-R2"].get("count_class_values") != [1, 2, 3, 4]:
        failures.append("B0-R2 count-class mapping changed")
    if (
        locks["B0-R2"].get("model_sha256")
        != sha256_file(paths["model"])
    ):
        failures.append("B0-R2 corrected-model SHA mismatch")

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    manifest_rows = load_csv(paths["pair_manifest"])
    mask_payload = torch.load(
        paths["a2_r2_mask"],
        map_location="cpu",
        weights_only=False,
    )
    physical_mask = mask_payload.get("mask")
    if (
        not isinstance(physical_mask, torch.Tensor)
        or physical_mask.dtype != torch.bool
        or tuple(physical_mask.shape) != (16, 10)
    ):
        failures.append("A2-R2 physical mask payload is invalid")

    split_data: dict[str, dict[str, list[np.ndarray]]] = {
        split: defaultdict(list)
        for split in ("train", "validation")
    }
    pair_audit_rows: list[dict[str, Any]] = []

    label_violation_counts = Counter()
    relation_counts = Counter()
    processed_pairs = Counter()
    processed_runs = Counter()
    processed_windows = Counter()

    print("===== V5 P2-B0-R3 CORRECTED NON-TEST LABEL AUDIT =====")
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")

    rows_by_split = {
        split: [
            row
            for row in manifest_rows
            if row.get("split") == split
        ]
        for split in ("train", "validation")
    }

    for split in ("train", "validation"):
        rows = rows_by_split[split]

        for row_index, row in enumerate(rows, start=1):
            pair_key = row.get("pair_key", "")
            common_length = as_int(row, "common_length")
            window_count = as_int(
                row,
                "window_count_per_member",
            )
            attack_length = as_int(row, "attack_length")
            control_length = as_int(row, "control_length")
            expected_category_count = category_count(pair_key)

            if expected_category_count not in {1, 2, 3, 4}:
                failures.append(
                    f"{split}/{pair_key}: cannot infer K1/K2/K3/K4"
                )
                continue

            target_indices = torch.arange(
                WINDOW - 1,
                common_length,
                STRIDE,
                dtype=torch.int64,
            )
            if int(target_indices.numel()) != window_count:
                failures.append(
                    f"{split}/{pair_key}: target count="
                    f"{int(target_indices.numel())}, "
                    f"manifest={window_count}"
                )
                continue

            pair_member_window_counts = {}
            pair_graph_positive = {}
            pair_inactive_attack_windows = 0
            pair_active_attack_windows = 0

            for mode in ("attack", "control"):
                suffix = mode.upper()
                path = (
                    root
                    / "runs"
                    / split
                    / f"{pair_key}_{suffix}.pt"
                )
                if not path.is_file():
                    failures.append(
                        f"missing {split}/{path.name}"
                    )
                    continue

                payload = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False,
                )
                processed_runs[split] += 1

                if not isinstance(payload, dict):
                    failures.append(
                        f"{split}/{path.name}: payload is not dict"
                    )
                    continue

                x = payload.get("x")
                if (
                    not isinstance(x, torch.Tensor)
                    or x.dtype != torch.float32
                    or x.ndim != 3
                    or tuple(x.shape[1:]) != (16, 81)
                    or not bool(torch.isfinite(x).all().item())
                ):
                    failures.append(
                        f"{split}/{path.name}: invalid x"
                    )
                    continue

                expected_native_length = (
                    attack_length if mode == "attack"
                    else control_length
                )
                if int(x.shape[0]) != expected_native_length:
                    failures.append(
                        f"{split}/{path.name}: native length="
                        f"{int(x.shape[0])}, manifest="
                        f"{expected_native_length}"
                    )

                logical = {
                    "graph": payload["y_attack"][target_indices],
                    "count": payload[
                        "y_attacker_count"
                    ][target_indices],
                    "source": payload["y_source"][target_indices],
                    "transit": payload["y_transit"][target_indices],
                    "victim": payload["y_victim"][target_indices],
                    "path": payload[
                        "y_attack_path"
                    ][target_indices],
                    "role_mask": payload[
                        "role_mask"
                    ][target_indices],
                }

                expected_shapes = {
                    "graph": (window_count,),
                    "count": (window_count,),
                    "source": (window_count, 16),
                    "transit": (window_count, 16),
                    "victim": (window_count, 16),
                    "path": (window_count, 16),
                    "role_mask": (window_count, 16),
                }
                for key, expected_shape in expected_shapes.items():
                    if tuple(logical[key].shape) != expected_shape:
                        failures.append(
                            f"{split}/{path.name}: {key} shape="
                            f"{tuple(logical[key].shape)}, "
                            f"expected {expected_shape}"
                        )

                for key in (
                    "graph",
                    "source",
                    "transit",
                    "victim",
                    "path",
                ):
                    if not is_binary_tensor(logical[key]):
                        label_violation_counts[
                            f"nonbinary_{key}"
                        ] += 1

                graph = logical["graph"].to(torch.int64)
                count = logical["count"].to(torch.int64)
                source = logical["source"].to(torch.int64)
                transit = logical["transit"].to(torch.int64)
                victim = logical["victim"].to(torch.int64)
                path_label = logical["path"].to(torch.int64)
                role_mask = logical["role_mask"].to(torch.int64)

                expected_role_mask = (
                    source
                    + 2 * transit
                    + 4 * victim
                )
                label_violation_counts[
                    "role_mask_out_of_range"
                ] += int(
                    (
                        (role_mask < 0)
                        | (role_mask > 7)
                    ).sum().item()
                )
                label_violation_counts[
                    "role_mask_bitfield_mismatch"
                ] += int(
                    (
                        role_mask
                        != expected_role_mask
                    ).sum().item()
                )

                allowed_counts = torch.tensor(
                    [0, 1, 2, 3, 4],
                    dtype=torch.int64,
                )
                valid_count = (
                    count.unsqueeze(-1) == allowed_counts
                ).any(dim=-1)
                label_violation_counts[
                    "invalid_count_value"
                ] += int((~valid_count).sum().item())

                active = graph == 1
                inactive = ~active

                label_violation_counts[
                    "graph_count_disagreement"
                ] += int(
                    (
                        active
                        != (count > 0)
                    ).sum().item()
                )
                label_violation_counts[
                    "graph_source_disagreement"
                ] += int(
                    (
                        active
                        != (source.sum(dim=1) > 0)
                    ).sum().item()
                )

                inactive_role_sum = (
                    source[inactive].sum()
                    + transit[inactive].sum()
                    + victim[inactive].sum()
                    + path_label[inactive].sum()
                    + role_mask[inactive].sum()
                )
                label_violation_counts[
                    "inactive_role_positive_entries"
                ] += int(inactive_role_sum.item())

                if bool(active.any().item()):
                    active_source_count = source[active].sum(dim=1)
                    active_count = count[active]
                    label_violation_counts[
                        "active_source_count_mismatch"
                    ] += int(
                        (
                            active_source_count != active_count
                        ).sum().item()
                    )
                    label_violation_counts[
                        "active_category_count_mismatch"
                    ] += int(
                        (
                            active_count
                            != expected_category_count
                        ).sum().item()
                    )
                    label_violation_counts[
                        "active_missing_victim"
                    ] += int(
                        (
                            victim[active].sum(dim=1) == 0
                        ).sum().item()
                    )
                    label_violation_counts[
                        "active_missing_path"
                    ] += int(
                        (
                            path_label[active].sum(dim=1) == 0
                        ).sum().item()
                    )
                    label_violation_counts[
                        "active_missing_role_mask"
                    ] += int(
                        (
                            role_mask[active].sum(dim=1) == 0
                        ).sum().item()
                    )

                if mode == "control":
                    control_total = (
                        graph.sum()
                        + count.sum()
                        + source.sum()
                        + transit.sum()
                        + victim.sum()
                        + path_label.sum()
                        + role_mask.sum()
                    )
                    label_violation_counts[
                        "control_positive_label_entries"
                    ] += int(control_total.item())

                union_roles = (
                    (source > 0)
                    | (transit > 0)
                    | (victim > 0)
                )
                relation_counts[
                    "role_mask_equals_source_plus_2transit_plus_4victim_items"
                ] += int(
                    (
                        role_mask == expected_role_mask
                    ).all(dim=1).sum().item()
                )
                relation_counts[
                    "path_equals_role_union_items"
                ] += int(
                    (
                        path_label.bool() == union_roles
                    ).all(dim=1).sum().item()
                )
                relation_counts[
                    "source_subset_of_path_items"
                ] += int(
                    (
                        (~source.bool())
                        | path_label.bool()
                    ).all(dim=1).sum().item()
                )
                relation_counts[
                    "victim_subset_of_path_items"
                ] += int(
                    (
                        (~victim.bool())
                        | path_label.bool()
                    ).all(dim=1).sum().item()
                )
                relation_counts["total_items"] += window_count

                targets_np = target_indices.cpu().numpy()
                progress = (
                    targets_np.astype(np.float64)
                    / max(1, common_length - 1)
                )
                common_lengths = np.full(
                    window_count,
                    common_length,
                    dtype=np.float64,
                )
                mode_values = np.full(
                    window_count,
                    1 if mode == "attack" else 0,
                    dtype=np.int64,
                )
                category_values = np.full(
                    window_count,
                    expected_category_count,
                    dtype=np.int64,
                )

                target_x = x[target_indices]
                status = target_x[:, :, 30]
                status_max = (
                    status.amax(dim=1).cpu().numpy()
                )
                status_mean = (
                    status.mean(dim=1).cpu().numpy()
                )
                traffic_score = target_x[:, :, 0:10].sum(dim=2)
                traffic_router = (
                    traffic_score.argmax(dim=1).cpu().numpy()
                )

                split_data[split]["graph"].append(
                    graph.cpu().numpy()
                )
                split_data[split]["count"].append(
                    count.cpu().numpy()
                )
                split_data[split]["source"].append(
                    source.cpu().numpy()
                )
                split_data[split]["transit"].append(
                    transit.cpu().numpy()
                )
                split_data[split]["victim"].append(
                    victim.cpu().numpy()
                )
                split_data[split]["path"].append(
                    path_label.cpu().numpy()
                )
                split_data[split]["role_mask"].append(
                    role_mask.cpu().numpy()
                )
                split_data[split]["mode"].append(mode_values)
                split_data[split]["category"].append(category_values)
                split_data[split]["progress"].append(progress)
                split_data[split]["common_length"].append(
                    common_lengths
                )
                split_data[split]["status_max"].append(status_max)
                split_data[split]["status_mean"].append(status_mean)
                split_data[split]["traffic_router"].append(
                    traffic_router
                )

                pair_member_window_counts[mode] = window_count
                pair_graph_positive[mode] = int(graph.sum().item())
                if mode == "attack":
                    pair_active_attack_windows = int(
                        active.sum().item()
                    )
                    pair_inactive_attack_windows = int(
                        inactive.sum().item()
                    )

                processed_windows[split] += window_count

            if (
                pair_member_window_counts.get("attack")
                != pair_member_window_counts.get("control")
            ):
                label_violation_counts[
                    "pair_member_window_count_mismatch"
                ] += 1

            pair_audit_rows.append(
                {
                    "split": split,
                    "pair_key": pair_key,
                    "category_count": expected_category_count,
                    "attack_length": attack_length,
                    "control_length": control_length,
                    "common_length": common_length,
                    "window_count_per_member": window_count,
                    "attack_graph_positive_windows": (
                        pair_graph_positive.get("attack")
                    ),
                    "control_graph_positive_windows": (
                        pair_graph_positive.get("control")
                    ),
                    "attack_active_windows": (
                        pair_active_attack_windows
                    ),
                    "attack_inactive_windows": (
                        pair_inactive_attack_windows
                    ),
                    "member_window_counts_equal": (
                        pair_member_window_counts.get("attack")
                        == pair_member_window_counts.get("control")
                    ),
                }
            )
            processed_pairs[split] += 1

            if row_index % 50 == 0 or row_index == len(rows):
                print(
                    f"{split}: audited {row_index}/{len(rows)} pairs"
                )

    # Turn per-member counts into total aligned item counts.
    expected_split_items = {
        "train": 70166,
        "validation": 12528,
    }
    for split in ("train", "validation"):
        if processed_pairs[split] != (
            415 if split == "train" else 74
        ):
            failures.append(
                f"{split} processed pair count="
                f"{processed_pairs[split]}"
            )
        if processed_runs[split] != (
            830 if split == "train" else 148
        ):
            failures.append(
                f"{split} processed run count="
                f"{processed_runs[split]}"
            )
        if processed_windows[split] != expected_split_items[split]:
            failures.append(
                f"{split} processed aligned items="
                f"{processed_windows[split]}, expected "
                f"{expected_split_items[split]}"
            )

    for key, count in label_violation_counts.items():
        if count != 0:
            failures.append(
                f"label integrity violation {key}={count}"
            )

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "validation"):
        arrays[split] = {}
        for key, parts in split_data[split].items():
            if not parts:
                failures.append(f"{split} has no collected {key}")
                continue
            arrays[split][key] = np.concatenate(parts, axis=0)

        lengths = {
            key: value.shape[0]
            for key, value in arrays[split].items()
        }
        if len(set(lengths.values())) != 1:
            failures.append(
                f"{split} collected array lengths disagree: {lengths}"
            )

    if failures:
        pair_csv_path = (
            output_dir
            / "V5_P2_B0_R3_PAIR_LABEL_AUDIT.csv"
        )
        write_csv(pair_csv_path, pair_audit_rows)

        report = {
            "stage": STAGE,
            "status": "HOLD",
            "decision": "BLOCK_P2_B1",
            "processed_pairs": dict(processed_pairs),
            "processed_runs": dict(processed_runs),
            "processed_aligned_items": dict(processed_windows),
            "label_violation_counts": dict(label_violation_counts),
            "relation_counts": dict(relation_counts),
            "failures": failures,
            "warnings": warnings,
            "security_boundary": {
                "train_tensor_contents_accessed": True,
                "validation_tensor_contents_accessed": True,
                "test_directory_enumerated": False,
                "test_tensor_contents_accessed": False,
            },
            "next_stage": None,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures[:100]:
            print("FAIL:", failure)
        print("failure_count:", len(failures))
        return 1

    train = arrays["train"]
    validation = arrays["validation"]

    train_active = train["graph"] == 1
    validation_active = validation["graph"] == 1

    # ------------------------------------------------------------------
    # Label summaries
    # ------------------------------------------------------------------
    label_summaries: dict[str, Any] = {}
    for split, data in arrays.items():
        active = data["graph"] == 1
        summary = {
            "items": int(data["graph"].shape[0]),
            "graph_negative": int((data["graph"] == 0).sum()),
            "graph_positive": int(active.sum()),
            "graph_positive_prevalence": float(active.mean()),
            "mode_control_items": int((data["mode"] == 0).sum()),
            "mode_attack_items": int((data["mode"] == 1).sum()),
            "active_count_distribution": {
                str(key): int(value)
                for key, value in zip(
                    *np.unique(
                        data["count"][active],
                        return_counts=True,
                    )
                )
            },
            "active_source_count_distribution": {
                str(key): int(value)
                for key, value in zip(
                    *np.unique(
                        data["source"][active].sum(axis=1),
                        return_counts=True,
                    )
                )
            },
            "active_victim_count_distribution": {
                str(key): int(value)
                for key, value in zip(
                    *np.unique(
                        data["victim"][active].sum(axis=1),
                        return_counts=True,
                    )
                )
            },
            "active_transit_count_distribution": {
                str(key): int(value)
                for key, value in zip(
                    *np.unique(
                        data["transit"][active].sum(axis=1),
                        return_counts=True,
                    )
                )
            },
            "active_path_count_distribution": {
                str(key): int(value)
                for key, value in zip(
                    *np.unique(
                        data["path"][active].sum(axis=1),
                        return_counts=True,
                    )
                )
            },
            "category_item_distribution": {
                str(key): int(value)
                for key, value in zip(
                    *np.unique(
                        data["category"],
                        return_counts=True,
                    )
                )
            },
            "role_mask_value_distribution": {
                str(key): int(value)
                for key, value in zip(
                    *np.unique(
                        data["role_mask"],
                        return_counts=True,
                    )
                )
            },
        }
        label_summaries[split] = summary

    # ------------------------------------------------------------------
    # Graph baselines
    # ------------------------------------------------------------------
    baseline_rows: list[dict[str, Any]] = []
    baseline_details: dict[str, Any] = {}

    def record_graph_baseline(
        name: str,
        prediction: np.ndarray,
        family: str,
        learned_input_status: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metrics = confusion_metrics(
            validation["graph"],
            prediction,
        )
        row = {
            "task": "graph",
            "baseline": name,
            "family": family,
            "learned_input_status": learned_input_status,
            **metrics,
        }
        baseline_rows.append(row)
        baseline_details[name] = {
            "metrics": metrics,
            "details": details or {},
        }
        return metrics

    always_normal = record_graph_baseline(
        "always_normal",
        np.zeros_like(validation["graph"]),
        "trivial",
        "none",
    )
    always_attack = record_graph_baseline(
        "always_attack",
        np.ones_like(validation["graph"]),
        "trivial",
        "none",
    )

    train_majority_label = int(
        train["graph"].mean() >= 0.5
    )
    train_majority = record_graph_baseline(
        "train_majority_graph",
        np.full_like(
            validation["graph"],
            train_majority_label,
        ),
        "trivial",
        "none",
        {"train_majority_label": train_majority_label},
    )

    mode_only = record_graph_baseline(
        "quarantined_mode_only",
        validation["mode"],
        "provenance_shortcut",
        "forbidden",
    )

    def threshold_graph_baseline(
        name: str,
        train_values: np.ndarray,
        validation_values: np.ndarray,
        family: str,
        learned_input_status: str,
    ) -> dict[str, Any]:
        fit = fit_threshold(
            train_values,
            train["graph"],
        )
        prediction = apply_threshold(
            validation_values,
            fit["threshold"],
            fit["direction"],
        )
        return record_graph_baseline(
            name,
            prediction,
            family,
            learned_input_status,
            {
                "threshold": fit["threshold"],
                "direction": fit["direction"],
                "train_metrics": fit["train_metrics"],
            },
        )

    epoch_position = threshold_graph_baseline(
        "forbidden_epoch_position_only",
        train["progress"],
        validation["progress"],
        "timing_shortcut",
        "forbidden",
    )
    common_length = threshold_graph_baseline(
        "forbidden_common_length_only",
        train["common_length"],
        validation["common_length"],
        "length_shortcut",
        "forbidden",
    )

    status_max_fit = fit_threshold(
        train["status_max"],
        train["graph"],
    )
    status_mean_fit = fit_threshold(
        train["status_mean"],
        train["graph"],
    )
    status_candidates = [
        ("max", status_max_fit),
        ("mean", status_mean_fit),
    ]
    status_summary_name, status_fit = max(
        status_candidates,
        key=lambda item: (
            item[1]["train_metrics"]["balanced_accuracy"],
            item[1]["train_metrics"]["f1"],
        ),
    )
    status_values = validation[
        "status_max" if status_summary_name == "max"
        else "status_mean"
    ]
    status_prediction = apply_threshold(
        status_values,
        status_fit["threshold"],
        status_fit["direction"],
    )
    status_only = record_graph_baseline(
        "quarantined_status_flags_only",
        status_prediction,
        "instrumentation_shortcut",
        "forbidden_and_excluded",
        {
            "summary": status_summary_name,
            "threshold": status_fit["threshold"],
            "direction": status_fit["direction"],
            "train_metrics": status_fit["train_metrics"],
        },
    )

    # ------------------------------------------------------------------
    # Count baselines, active windows only
    # ------------------------------------------------------------------
    train_active_counts = train["count"][train_active]
    validation_active_counts = validation["count"][
        validation_active
    ]
    count_values, count_frequencies = np.unique(
        train_active_counts,
        return_counts=True,
    )
    most_common_count = int(
        count_values[np.argmax(count_frequencies)]
    )
    count_prediction = np.full_like(
        validation_active_counts,
        most_common_count,
    )
    most_common_count_accuracy = float(
        (count_prediction == validation_active_counts).mean()
    )
    baseline_rows.append(
        {
            "task": "count_active",
            "baseline": "most_common_active_count",
            "family": "trivial",
            "learned_input_status": "none",
            "accuracy": most_common_count_accuracy,
            "predicted_count": most_common_count,
            "support": int(validation_active_counts.size),
        }
    )
    baseline_details["most_common_active_count"] = {
        "predicted_count": most_common_count,
        "validation_accuracy": most_common_count_accuracy,
    }

    category_count_accuracy = float(
        (
            validation["category"][validation_active]
            == validation_active_counts
        ).mean()
    )
    baseline_rows.append(
        {
            "task": "count_active",
            "baseline": "quarantined_category_only_count",
            "family": "provenance_shortcut",
            "learned_input_status": "forbidden",
            "accuracy": category_count_accuracy,
            "support": int(validation_active_counts.size),
        }
    )
    baseline_details["quarantined_category_only_count"] = {
        "validation_accuracy": category_count_accuracy,
    }

    # ------------------------------------------------------------------
    # Source trivial and traffic baselines, active windows only
    # ------------------------------------------------------------------
    train_active_source = train["source"][train_active]
    validation_active_source = validation["source"][
        validation_active
    ]

    train_source_frequency = train_active_source.sum(axis=0)
    most_common_source_node = int(
        np.argmax(train_source_frequency)
    )
    most_common_source_prediction = np.full(
        validation_active_source.shape[0],
        most_common_source_node,
        dtype=np.int64,
    )
    most_common_source_metrics = source_single_prediction_metrics(
        validation_active_source,
        most_common_source_prediction,
    )
    baseline_rows.append(
        {
            "task": "source_active",
            "baseline": "most_common_source_router",
            "family": "trivial_spatial_prior",
            "learned_input_status": (
                "router_identity_forbidden; audit_only"
            ),
            **most_common_source_metrics,
            "predicted_router": most_common_source_node,
        }
    )
    baseline_details["most_common_source_router"] = {
        "predicted_router": most_common_source_node,
        "train_source_frequency": (
            train_source_frequency.astype(int).tolist()
        ),
        "validation_metrics": most_common_source_metrics,
    }

    traffic_source_prediction = validation["traffic_router"][
        validation_active
    ]
    traffic_source_metrics = source_single_prediction_metrics(
        validation_active_source,
        traffic_source_prediction,
    )
    baseline_rows.append(
        {
            "task": "source_active",
            "baseline": "highest_final_epoch_traffic_router",
            "family": "observable_trivial_heuristic",
            "learned_input_status": "allowed_PRIMARY58",
            **traffic_source_metrics,
        }
    )
    baseline_details["highest_final_epoch_traffic_router"] = {
        "traffic_definition": (
            "argmax_router sum(final_epoch standardized "
            "in/out count channels 0..9)"
        ),
        "validation_metrics": traffic_source_metrics,
    }

    # ------------------------------------------------------------------
    # Physical-mask-only role priors, active windows only
    # ------------------------------------------------------------------
    signatures = mask_signature(physical_mask)
    mask_role_baselines: dict[str, Any] = {}

    for role in ("source", "transit", "victim", "path"):
        result = role_mask_prior_baseline(
            train[role][train_active],
            validation[role][validation_active],
            signatures,
        )
        mask_role_baselines[role] = result
        baseline_rows.append(
            {
                "task": f"{role}_active_node",
                "baseline": "physical_mask_only_role_prior",
                "family": "structural_shortcut",
                "learned_input_status": "allowed_topology_mask",
                "balanced_accuracy": (
                    result["validation_metrics"][
                        "balanced_accuracy"
                    ]
                ),
                "f1": result["validation_metrics"]["f1"],
                "precision": (
                    result["validation_metrics"]["precision"]
                ),
                "recall": result["validation_metrics"]["recall"],
                "fpr": result["validation_metrics"]["fpr"],
                "auroc": result["validation_auroc"],
                "average_precision": (
                    result["validation_average_precision"]
                ),
                "threshold": result["threshold"],
                "direction": result["direction"],
            }
        )

    baseline_details["physical_mask_only_role_priors"] = (
        mask_role_baselines
    )

    # ------------------------------------------------------------------
    # Shortcut interpretation
    # ------------------------------------------------------------------
    epoch_ba = epoch_position["balanced_accuracy"]
    if epoch_ba >= EPOCH_POSITION_BLOCK_BA:
        shortcut_blocks.append(
            "epoch-position-only graph balanced accuracy "
            f"{epoch_ba:.4f} >= {EPOCH_POSITION_BLOCK_BA:.2f}"
        )
    elif epoch_ba >= EPOCH_POSITION_WARN_BA:
        shortcut_warnings.append(
            "epoch-position-only graph balanced accuracy "
            f"{epoch_ba:.4f} >= {EPOCH_POSITION_WARN_BA:.2f}"
        )

    for role, result in mask_role_baselines.items():
        auc = result["validation_auroc"]
        if auc is None:
            continue
        if auc >= MASK_ROLE_BLOCK_AUROC:
            shortcut_blocks.append(
                f"physical-mask-only {role} AUROC "
                f"{auc:.4f} >= {MASK_ROLE_BLOCK_AUROC:.2f}"
            )
        elif auc >= MASK_ROLE_WARN_AUROC:
            shortcut_warnings.append(
                f"physical-mask-only {role} AUROC "
                f"{auc:.4f} >= {MASK_ROLE_WARN_AUROC:.2f}"
            )

    if (
        most_common_source_metrics["hit_at_1"]
        >= COMMON_SOURCE_WARN_HIT1
    ):
        shortcut_warnings.append(
            "most-common-source hit@1 "
            f"{most_common_source_metrics['hit_at_1']:.4f} >= "
            f"{COMMON_SOURCE_WARN_HIT1:.2f}"
        )

    if (
        traffic_source_metrics["hit_at_1"]
        >= TRAFFIC_SOURCE_WARN_HIT1
    ):
        shortcut_warnings.append(
            "highest-traffic-router source hit@1 "
            f"{traffic_source_metrics['hit_at_1']:.4f} >= "
            f"{TRAFFIC_SOURCE_WARN_HIT1:.2f}; treat as a required "
            "paper baseline, not leakage"
        )

    # Expected forbidden baselines are diagnostic only.
    diagnostic_notes = [
        (
            "Mode, category, status flags, run length, and epoch position "
            "remain forbidden learned inputs regardless of their scores."
        ),
        (
            "A strong highest-traffic-router result is an observable "
            "heuristic baseline and must be reported against the model."
        ),
        (
            "Physical mask is an authorized B3 input, so strong mask-only "
            "role AUROC indicates dataset spatial bias and is a blocker "
            "at the configured threshold."
        ),
        (
            "role_mask is the audited bit-field source + 2*transit + "
            "4*victim. It is bookkeeping only and is neither a model "
            "input nor an independently trained output."
        ),
        (
            "The P2 B3 architecture uses four active-count logits for "
            "raw attacker counts 1,2,3,4. Zero count remains represented "
            "by the graph normal/attack head."
        ),
    ]

    status = (
        "HOLD"
        if shortcut_blocks
        else "COMPLETE"
    )
    decision = (
        "BLOCK_P2_B1_FOR_SHORTCUT_REMEDIATION"
        if shortcut_blocks
        else "AUTHORIZE_P2_B1_TRAINING_PROTOCOL_LOCK"
    )
    next_stage = (
        None
        if shortcut_blocks
        else "V5_P2_B1_TRAINING_PROTOCOL_LOCK"
    )

    pair_csv_path = (
        output_dir
        / "V5_P2_B0_R3_PAIR_LABEL_AUDIT.csv"
    )
    baseline_csv_path = (
        output_dir
        / "V5_P2_B0_R3_TRIVIAL_AND_SHORTCUT_BASELINES.csv"
    )
    label_summary_path = (
        output_dir
        / "V5_P2_B0_R3_LABEL_SUMMARY.json"
    )

    write_csv(pair_csv_path, pair_audit_rows)
    write_csv(baseline_csv_path, baseline_rows)
    write_json(
        label_summary_path,
        {
            "splits": label_summaries,
            "label_violation_counts": dict(
                label_violation_counts
            ),
            "label_relation_diagnostics": {
                key: {
                    "matching_items": int(value),
                    "total_items": int(
                        relation_counts["total_items"]
                    ),
                    "fraction": (
                        value
                        / max(
                            1,
                            relation_counts["total_items"],
                        )
                    ),
                }
                for key, value in relation_counts.items()
                if key != "total_items"
            },
        },
    )

    report = {
        "stage": STAGE,
        "status": status,
        "decision": decision,
        "dataset": {
            "link_path": str(root_input.absolute()),
            "resolved_root": str(root),
        },
        "coverage": {
            "train_pairs": int(processed_pairs["train"]),
            "validation_pairs": int(
                processed_pairs["validation"]
            ),
            "total_pairs": int(sum(processed_pairs.values())),
            "train_runs": int(processed_runs["train"]),
            "validation_runs": int(
                processed_runs["validation"]
            ),
            "total_runs": int(sum(processed_runs.values())),
            "train_aligned_items": int(
                processed_windows["train"]
            ),
            "validation_aligned_items": int(
                processed_windows["validation"]
            ),
            "total_aligned_items": int(
                sum(processed_windows.values())
            ),
        },
        "corrected_contracts": {
            "count_categories": [1, 2, 3, 4],
            "active_count_values": [1, 2, 3, 4],
            "count_head_logits": 4,
            "corrected_model_parameter_count": 43273,
            "role_mask_semantics": (
                "source + 2*transit + 4*victim"
            ),
            "role_mask_learned_input": False,
            "role_mask_independent_model_output": False,
        },
        "label_integrity": {
            "violation_counts": dict(label_violation_counts),
            "all_zero": not any(label_violation_counts.values()),
            "relation_diagnostics": {
                key: {
                    "matching_items": int(value),
                    "total_items": int(
                        relation_counts["total_items"]
                    ),
                    "fraction": (
                        value
                        / max(
                            1,
                            relation_counts["total_items"],
                        )
                    ),
                }
                for key, value in relation_counts.items()
                if key != "total_items"
            },
        },
        "label_summaries": label_summaries,
        "baselines": baseline_details,
        "shortcut_policy": {
            "epoch_position_block_balanced_accuracy": (
                EPOCH_POSITION_BLOCK_BA
            ),
            "epoch_position_warning_balanced_accuracy": (
                EPOCH_POSITION_WARN_BA
            ),
            "mask_role_block_auroc": MASK_ROLE_BLOCK_AUROC,
            "mask_role_warning_auroc": MASK_ROLE_WARN_AUROC,
            "common_source_warning_hit_at_1": (
                COMMON_SOURCE_WARN_HIT1
            ),
            "traffic_source_warning_hit_at_1": (
                TRAFFIC_SOURCE_WARN_HIT1
            ),
        },
        "shortcut_blocks": shortcut_blocks,
        "shortcut_warnings": shortcut_warnings,
        "diagnostic_notes": diagnostic_notes,
        "artifacts": {
            "pair_label_audit_csv": [
                str(pair_csv_path),
                sha256_file(pair_csv_path),
            ],
            "baseline_csv": [
                str(baseline_csv_path),
                sha256_file(baseline_csv_path),
            ],
            "label_summary_json": [
                str(label_summary_path),
                sha256_file(label_summary_path),
            ],
        },
        "provenance": {
            "a0_report_sha256": sha256_file(paths["a0_report"]),
            "a0_lock_sha256": sha256_file(paths["a0_lock"]),
            "a1_r2_report_sha256": sha256_file(
                paths["a1_r2_report"]
            ),
            "a1_r2_lock_sha256": sha256_file(
                paths["a1_r2_lock"]
            ),
            "pair_manifest_sha256": sha256_file(
                paths["pair_manifest"]
            ),
            "a2_r2_report_sha256": sha256_file(
                paths["a2_r2_report"]
            ),
            "a2_r2_lock_sha256": sha256_file(
                paths["a2_r2_lock"]
            ),
            "a2_r2_mask_sha256": sha256_file(
                paths["a2_r2_mask"]
            ),
            "a3_report_sha256": sha256_file(
                paths["a3_report"]
            ),
            "a3_lock_sha256": sha256_file(paths["a3_lock"]),
            "b0_r1b_report_sha256": sha256_file(
                paths["b0_r1b_report"]
            ),
            "b0_r1b_lock_sha256": sha256_file(
                paths["b0_r1b_lock"]
            ),
            "b0_r2_report_sha256": sha256_file(
                paths["b0_r2_report"]
            ),
            "b0_r2_lock_sha256": sha256_file(
                paths["b0_r2_lock"]
            ),
            "loader_sha256": sha256_file(paths["loader"]),
            "corrected_model_sha256": sha256_file(paths["model"]),
        },
        "security_boundary": {
            "train_tensor_contents_accessed": True,
            "validation_tensor_contents_accessed": True,
            "test_directory_existence_checked": False,
            "test_directory_enumerated": False,
            "test_tensor_files_opened": False,
            "test_tensor_bytes_read": False,
            "test_tensor_contents_accessed": False,
            "test_dataset_constructed": False,
            "test_windows_constructed": False,
            "test_labels_read": False,
        },
        "failures": failures,
        "warnings": warnings + shortcut_warnings,
        "next_stage": next_stage,
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    if status == "HOLD":
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print("===== V5 P2-B0-R3 FINAL =====")
        print("status: HOLD")
        print("decision:", decision)
        print("label_integrity_pass: true")
        print("shortcut_block_count:", len(shortcut_blocks))
        for item in shortcut_blocks:
            print("SHORTCUT_BLOCK:", item)
        print("shortcut_warning_count:", len(shortcut_warnings))
        for item in shortcut_warnings:
            print("SHORTCUT_WARNING:", item)
        print("test_directory_enumerated: false")
        print("test_tensor_contents_accessed: false")
        print(f"{STAGE}_HOLD")
        return 1

    lock = {
        "status": COMPLETE,
        "decision": decision,
        "report_sha256": sha256_file(report_path),
        "pair_label_audit_sha256": sha256_file(pair_csv_path),
        "baseline_csv_sha256": sha256_file(baseline_csv_path),
        "label_summary_sha256": sha256_file(label_summary_path),
        "total_pairs": int(sum(processed_pairs.values())),
        "total_runs": int(sum(processed_runs.values())),
        "total_aligned_items": int(sum(processed_windows.values())),
        "label_integrity_pass": True,
        "label_violation_counts": dict(label_violation_counts),
        "count_categories": [1, 2, 3, 4],
        "active_count_values": [1, 2, 3, 4],
        "count_head_logits": 4,
        "corrected_model_parameter_count": 43273,
        "corrected_model_sha256": sha256_file(paths["model"]),
        "role_mask_semantics": (
            "bitfield3_source1_transit2_victim4"
        ),
        "shortcut_block_count": 0,
        "shortcut_warning_count": len(shortcut_warnings),
        "epoch_position_validation_balanced_accuracy": (
            epoch_position["balanced_accuracy"]
        ),
        "mask_role_validation_auroc": {
            role: result["validation_auroc"]
            for role, result in mask_role_baselines.items()
        },
        "most_common_source_validation_hit_at_1": (
            most_common_source_metrics["hit_at_1"]
        ),
        "highest_traffic_source_validation_hit_at_1": (
            traffic_source_metrics["hit_at_1"]
        ),
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "next_stage": next_stage,
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-B0-R3 FINAL =====")
    print("status: COMPLETE")
    print("decision:", decision)
    print("train_pairs:", processed_pairs["train"])
    print("validation_pairs:", processed_pairs["validation"])
    print("total_pairs:", sum(processed_pairs.values()))
    print("total_runs:", sum(processed_runs.values()))
    print("train_aligned_items:", processed_windows["train"])
    print(
        "validation_aligned_items:",
        processed_windows["validation"],
    )
    print("total_aligned_items:", sum(processed_windows.values()))
    print("label_integrity_pass: true")
    print("label_violation_count: 0")
    print("count_categories: [1, 2, 3, 4]")
    print("active_count_values: [1, 2, 3, 4]")
    print("count_head_logits: 4")
    print("corrected_model_parameter_count: 43273")
    print(
        "role_mask_semantics: "
        "bitfield3_source1_transit2_victim4"
    )
    print(
        "graph_validation_prevalence:",
        f"{label_summaries['validation']['graph_positive_prevalence']:.10g}",
    )
    print(
        "always_normal_validation_balanced_accuracy:",
        f"{always_normal['balanced_accuracy']:.10g}",
    )
    print(
        "epoch_position_validation_balanced_accuracy:",
        f"{epoch_position['balanced_accuracy']:.10g}",
    )
    print(
        "common_length_validation_balanced_accuracy:",
        f"{common_length['balanced_accuracy']:.10g}",
    )
    print(
        "quarantined_status_validation_balanced_accuracy:",
        f"{status_only['balanced_accuracy']:.10g}",
    )
    print(
        "quarantined_mode_validation_balanced_accuracy:",
        f"{mode_only['balanced_accuracy']:.10g}",
    )
    print(
        "most_common_count_validation_accuracy:",
        f"{most_common_count_accuracy:.10g}",
    )
    print(
        "quarantined_category_count_validation_accuracy:",
        f"{category_count_accuracy:.10g}",
    )
    print(
        "most_common_source_validation_hit_at_1:",
        f"{most_common_source_metrics['hit_at_1']:.10g}",
    )
    print(
        "highest_traffic_source_validation_hit_at_1:",
        f"{traffic_source_metrics['hit_at_1']:.10g}",
    )
    for role in ("source", "transit", "victim", "path"):
        print(
            f"mask_only_{role}_validation_auroc:",
            mask_role_baselines[role]["validation_auroc"],
        )
    print("shortcut_block_count: 0")
    print("shortcut_warning_count:", len(shortcut_warnings))
    for item in shortcut_warnings:
        print("SHORTCUT_WARNING:", item)
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")
    print("failure_count:", len(failures))
    print(
        "warning_count:",
        len(warnings) + len(shortcut_warnings),
    )
    print("next_stage:", next_stage)
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
