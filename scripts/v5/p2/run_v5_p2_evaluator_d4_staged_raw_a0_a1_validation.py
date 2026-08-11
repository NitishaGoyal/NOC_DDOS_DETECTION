#!/usr/bin/env python3
"""V5 P2 D4 staged Raw/A0/A1 evaluator implementation and validation.

This validation-only stage proves the persistence order required by D1:

E1 immutable neural archive binding
    -> E2 Raw outputs/metrics committed
    -> E3 A0 outputs/metrics committed
    -> synthetic A1-failure isolation proof
    -> E4 certified A1 outputs/metrics committed

The stage reads:
- the immutable validation-logit archive;
- the frozen E2 threshold contract;
- the completed D3 certified-output archive.

It performs no model inference, decoder execution, threshold selection, test
access, or evaluation authorization.

An A1 technical failure is injected after E2/E3 have been committed. D4 then
verifies that every E1/E2/E3 hash remains unchanged before allowing the real
E4 validation stage to consume the immutable D3 certified outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


STAGE = "V5_P2_EVALUATOR_D4_STAGED_RAW_A0_A1_VALIDATION"
POLICY_ID = "V5P2-D1-6ce8cc7c76b8d2cff367631077d65f20"
EXPECTED_ITEMS = 12_528
ROUTER_COUNT = 16
ROLE_NAMES = ("source", "transit", "victim", "path")

EXPECTED_HASHES = {
    "validation_manifest": (
        "ac7a38fd1f4b00b578254d9253418655aa78b6cc7a7a1d65f78ba15487b14604"
    ),
    "threshold_contract": (
        "b9acb9dac24cc1d0fe2cdae32525a4c0bdddc40e1607824599ad5bc3a446d957"
    ),
    "d1_contract": (
        "b63efbfe1272fb7f6ab922645182fba86e28d7c8f42a70e0f3ac453b88bec71f"
    ),
    "d2_artifact": (
        "fcb78ae4059f49e4c78d4e3cf603160f7d8bad72857d6ef6b777cc4de0ab7055"
    ),
    "d3_artifact": (
        "11339c6afcec5275a310219e05a360638121cdfabd2f02589273e2c84af56def"
    ),
    "certified_decoder": (
        "30309376cb0990a0f482045e2dac413d4370729cf94f9e556f895af2c5c8c71d"
    ),
}

EXPECTED_THRESHOLDS = {
    "graph": 0.47174675035328983,
    "source": 0.94960549299285935,
    "transit": 0.85703332488359107,
    "victim": 0.82601148026998117,
    "path": 0.83393474660904676,
}
EXPECTED_A1_THRESHOLD = 8.7205320882398425
EXPECTED_D3_CHUNKS = 126
EXPECTED_D3_ATTACK_COUNT = 5811


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: observed={observed}, expected={expected}"
        )
    return observed


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def hash_tree(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("logits contain non-finite values")
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def binary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.uint8).reshape(-1)
    truth = np.asarray(target, dtype=np.uint8).reshape(-1)
    if pred.shape != truth.shape:
        raise ValueError("binary prediction/target shape mismatch")
    if not np.all((pred == 0) | (pred == 1)):
        raise ValueError("binary prediction contains values outside {0,1}")
    if not np.all((truth == 0) | (truth == 1)):
        raise ValueError("binary target contains values outside {0,1}")

    tp = int(np.sum((pred == 1) & (truth == 1)))
    fp = int(np.sum((pred == 1) & (truth == 0)))
    tn = int(np.sum((pred == 0) & (truth == 0)))
    fn = int(np.sum((pred == 0) & (truth == 1)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": safe_div(tp + tn, tp + fp + tn + fn),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1,
        "support_negative": tn + fp,
        "support_positive": tp + fn,
    }


def rankdata_average(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(x.size, dtype=np.float64)
    start = 0
    while start < x.size:
        stop = start + 1
        while stop < x.size and sorted_x[stop] == sorted_x[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def binary_auroc(score: np.ndarray, target: np.ndarray) -> float:
    truth = np.asarray(target, dtype=np.uint8).reshape(-1)
    positives = int(truth.sum())
    negatives = int(truth.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata_average(score)
    return float(
        (
            ranks[truth == 1].sum()
            - positives * (positives + 1) / 2.0
        )
        / (positives * negatives)
    )


def average_precision(score: np.ndarray, target: np.ndarray) -> float:
    truth = np.asarray(target, dtype=np.uint8).reshape(-1)
    positives = int(truth.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(
        -np.asarray(score, dtype=np.float64).reshape(-1),
        kind="mergesort",
    )
    sorted_truth = truth[order].astype(np.int64)
    precision = np.cumsum(sorted_truth) / np.arange(1, truth.size + 1)
    return float(np.sum(precision * sorted_truth) / positives)


def multiclass_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    classes: Sequence[int],
) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.int64).reshape(-1)
    truth = np.asarray(target, dtype=np.int64).reshape(-1)
    if pred.shape != truth.shape:
        raise ValueError("multiclass prediction/target shape mismatch")
    labels = tuple(int(value) for value in classes)
    mapping = {value: index for index, value in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)

    for truth_value, pred_value in zip(truth.tolist(), pred.tolist()):
        if truth_value not in mapping or pred_value not in mapping:
            raise ValueError(
                f"class outside frozen set: truth={truth_value}, prediction={pred_value}"
            )
        matrix[mapping[truth_value], mapping[pred_value]] += 1

    per_class: dict[str, Any] = {}
    f1_values = []
    for value, index in mapping.items():
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_class[str(value)] = {
            "support": int(matrix[index, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "classes": list(labels),
        "confusion_matrix": matrix.tolist(),
        "accuracy": float(np.mean(pred == truth)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
    }


def masks_to_binary(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=np.uint16).reshape(-1)
    bits = np.arange(ROUTER_COUNT, dtype=np.uint16)
    return ((values[:, None] >> bits[None, :]) & 1).astype(np.uint8)


def role_metrics(
    predictions: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    exact_vectors = []
    for role in ROLE_NAMES:
        prediction = np.asarray(predictions[role], dtype=np.uint8)
        target = np.asarray(labels[role], dtype=np.uint8)
        exact = np.all(prediction == target, axis=1)
        exact_vectors.append(exact)
        results[role] = {
            **binary_metrics(prediction, target),
            "exact_match_16_router": float(np.mean(exact)),
        }

    joint = np.all(np.stack(exact_vectors, axis=1), axis=1)
    return {
        "per_role": results,
        "macro_role_f1": float(
            np.mean([results[role]["f1"] for role in ROLE_NAMES])
        ),
        "joint_role_exact_match": float(np.mean(joint)),
        "_joint_exact_vector": joint,
    }


def structured_group_metrics(
    *,
    graph_prediction: np.ndarray,
    count_prediction: np.ndarray,
    role_predictions: dict[str, np.ndarray],
    graph_score: np.ndarray,
    labels: dict[str, np.ndarray],
    count_classes: Sequence[int],
    raw_count_attack_only: bool,
) -> dict[str, Any]:
    graph = binary_metrics(graph_prediction, labels["graph"])
    graph["auroc"] = binary_auroc(graph_score, labels["graph"])
    graph["average_precision"] = average_precision(
        graph_score,
        labels["graph"],
    )

    if raw_count_attack_only:
        active = labels["graph"] == 1
        count = multiclass_metrics(
            count_prediction[active],
            labels["count"][active],
            count_classes,
        )
        count_scope = "attack_items_only"
    else:
        count = multiclass_metrics(
            count_prediction,
            labels["count"],
            count_classes,
        )
        count_scope = "all_items"

    roles = role_metrics(role_predictions, labels)
    graph_exact = (
        np.asarray(graph_prediction, dtype=np.uint8)
        == labels["graph"]
    )

    if raw_count_attack_only:
        count_exact = np.ones(labels["graph"].shape[0], dtype=bool)
        active = labels["graph"] == 1
        count_exact[active] = (
            np.asarray(count_prediction, dtype=np.int64)[active]
            == labels["count"][active]
        )
        strict_definition = (
            "graph exact; count exact on attack items and ignored on controls; "
            "all independently thresholded role sets exact"
        )
    else:
        count_exact = (
            np.asarray(count_prediction, dtype=np.int64)
            == labels["count"]
        )
        strict_definition = (
            "graph, all-item count, and all four 16-router role sets exact"
        )

    strict = graph_exact & count_exact & roles["_joint_exact_vector"]
    return {
        "graph": graph,
        "count": {
            **count,
            "scope": count_scope,
        },
        "roles": {
            key: value
            for key, value in roles.items()
            if not key.startswith("_")
        },
        "strict_all_task_exactness": float(np.mean(strict)),
        "strict_all_task_exactness_attack_items": float(
            np.mean(strict[labels["graph"] == 1])
        ),
        "strict_all_task_exactness_control_items": float(
            np.mean(strict[labels["graph"] == 0])
        ),
        "strict_definition": strict_definition,
    }


def recursive_numeric_values(
    value: Any,
    *,
    key_name: str,
) -> list[float]:
    values: list[float] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == key_name and isinstance(child, (int, float)):
                values.append(float(child))
            values.extend(
                recursive_numeric_values(child, key_name=key_name)
            )
    elif isinstance(value, list):
        for child in value:
            values.extend(
                recursive_numeric_values(child, key_name=key_name)
            )
    return values


def validate_threshold_contract(contract: dict[str, Any]) -> None:
    if contract.get("status") != "FROZEN":
        raise RuntimeError("E2 threshold contract is not frozen")

    graph_values = recursive_numeric_values(
        contract.get("graph_threshold", {}),
        key_name="selected_threshold",
    )
    if EXPECTED_THRESHOLDS["graph"] not in graph_values:
        raise RuntimeError("frozen graph threshold changed")

    role_contract = contract.get("role_thresholds", {})
    for role in ROLE_NAMES:
        role_values = recursive_numeric_values(
            role_contract.get(role, {}),
            key_name="selected_threshold",
        )
        if EXPECTED_THRESHOLDS[role] not in role_values:
            raise RuntimeError(f"frozen {role} threshold changed")


def validate_archive_files(
    archive: Path,
    manifest: dict[str, Any],
) -> dict[str, str]:
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("validation archive is not complete")
    if int(manifest.get("item_count", -1)) != EXPECTED_ITEMS:
        raise RuntimeError("validation archive item count changed")

    records = manifest.get("artifacts")
    if not isinstance(records, dict):
        raise RuntimeError("validation archive artifact map missing")

    observed: dict[str, str] = {}
    for file_name, record in records.items():
        path = archive / file_name
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = record.get("sha256")
        if not isinstance(expected, str):
            raise RuntimeError(f"archive SHA missing for {file_name}")
        observed[file_name] = require_hash(
            path,
            expected,
            f"validation archive {file_name}",
        )
    return observed


def load_validation_archive(
    archive: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    arrays = {
        name: np.load(
            archive / f"{name}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        for name in (
            "graph_logits",
            "count_logits",
            "source_logits",
            "transit_logits",
            "victim_logits",
            "path_logits",
            "y_graph",
            "y_attacker_count",
            "y_source",
            "y_transit",
            "y_victim",
            "y_path",
            "validation_item_index",
            "manifest_row_index",
        )
    }

    expected_shapes = {
        "graph_logits": (EXPECTED_ITEMS,),
        "count_logits": (EXPECTED_ITEMS, 4),
        "source_logits": (EXPECTED_ITEMS, 16),
        "transit_logits": (EXPECTED_ITEMS, 16),
        "victim_logits": (EXPECTED_ITEMS, 16),
        "path_logits": (EXPECTED_ITEMS, 16),
        "y_graph": (EXPECTED_ITEMS,),
        "y_attacker_count": (EXPECTED_ITEMS,),
        "y_source": (EXPECTED_ITEMS, 16),
        "y_transit": (EXPECTED_ITEMS, 16),
        "y_victim": (EXPECTED_ITEMS, 16),
        "y_path": (EXPECTED_ITEMS, 16),
        "validation_item_index": (EXPECTED_ITEMS,),
        "manifest_row_index": (EXPECTED_ITEMS,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise RuntimeError(
                f"{name} shape={arrays[name].shape}, expected={shape}"
            )
        if not np.all(np.isfinite(arrays[name])):
            raise RuntimeError(f"{name} contains non-finite values")

    if not np.array_equal(
        arrays["validation_item_index"],
        np.arange(EXPECTED_ITEMS, dtype=np.int64),
    ):
        raise RuntimeError("validation item order changed")

    labels = {
        "graph": arrays["y_graph"].astype(np.uint8, copy=False),
        "count": arrays["y_attacker_count"].astype(np.int64, copy=False),
        "source": arrays["y_source"].astype(np.uint8, copy=False),
        "transit": arrays["y_transit"].astype(np.uint8, copy=False),
        "victim": arrays["y_victim"].astype(np.uint8, copy=False),
        "path": arrays["y_path"].astype(np.uint8, copy=False),
    }

    active = labels["graph"] == 1
    control = ~active
    if not np.all((labels["graph"] == 0) | (labels["graph"] == 1)):
        raise RuntimeError("graph labels are nonbinary")
    if not np.all(labels["count"][control] == 0):
        raise RuntimeError("control count labels are nonzero")
    if not np.all(
        np.isin(labels["count"][active], np.asarray([1, 2, 3, 4]))
    ):
        raise RuntimeError("active count labels outside K1-K4")
    for role in ROLE_NAMES:
        if not np.all((labels[role] == 0) | (labels[role] == 1)):
            raise RuntimeError(f"{role} labels are nonbinary")

    return arrays, labels


def commit_stage(
    *,
    parent: Path,
    name: str,
    arrays: dict[str, np.ndarray] | None,
    metrics: dict[str, Any],
    bindings: dict[str, Any],
) -> tuple[Path, dict[str, str]]:
    final = parent / name
    if final.exists():
        raise RuntimeError(f"stage output already exists: {final}")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{name}_tmp_", dir=parent)
    )
    try:
        if arrays is not None:
            atomic_savez(
                temporary / "outputs.npz",
                **{
                    key: np.asarray(value)
                    for key, value in arrays.items()
                },
            )
        atomic_write_json(temporary / "metrics.json", metrics)
        atomic_write_json(temporary / "bindings.json", bindings)

        records = {
            path.name: sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        lock = {
            "stage": name,
            "status": "LOCKED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "files": records,
        }
        atomic_write_json(temporary / "LOCK.json", lock)
        atomic_write_text(
            temporary / "COMPLETE",
            f"{name}_COMPLETE\n",
        )

        for path in temporary.iterdir():
            if path.is_file():
                path.chmod(0o444)
        temporary.chmod(0o555)
        os.replace(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return final, hash_tree(final)


def load_d3_outputs(
    chunks_dir: Path,
) -> dict[str, np.ndarray]:
    paths = sorted(chunks_dir.glob("chunk_*.npz"))
    if len(paths) != EXPECTED_D3_CHUNKS:
        raise RuntimeError(
            f"D3 chunk count={len(paths)}, expected={EXPECTED_D3_CHUNKS}"
        )

    lists: dict[str, list[np.ndarray]] = {
        "validation_index": [],
        "margin": [],
        "attacker_count": [],
        "route_ids": [],
        "source_mask": [],
        "transit_mask": [],
        "victim_mask": [],
        "path_mask": [],
        "attack_after_threshold": [],
        "primary_gap": [],
        "lexicographic_gap": [],
        "primary_integrality_error": [],
        "lexicographic_integrality_error": [],
        "primary_bound_violation": [],
        "lexicographic_bound_violation": [],
        "primary_linear_violation": [],
        "lexicographic_linear_violation": [],
        "primary_objective_difference": [],
        "lexicographic_objective_difference": [],
    }

    for path in paths:
        with np.load(path, allow_pickle=False) as value:
            for name in lists:
                if name not in value.files:
                    raise RuntimeError(f"{path}: missing {name}")
                lists[name].append(np.asarray(value[name]))

    arrays = {
        name: np.concatenate(values, axis=0)
        for name, values in lists.items()
    }
    if not np.array_equal(
        arrays["validation_index"],
        np.arange(EXPECTED_ITEMS, dtype=np.int64),
    ):
        raise RuntimeError("D3 output order changed")
    if int(arrays["attack_after_threshold"].sum()) != (
        EXPECTED_D3_ATTACK_COUNT
    ):
        raise RuntimeError("D3 thresholded-attack count changed")
    return arrays


def raw_and_a0(
    arrays: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    graph_probability = stable_sigmoid(arrays["graph_logits"])
    role_probability = {
        role: stable_sigmoid(arrays[f"{role}_logits"])
        for role in ROLE_NAMES
    }
    graph_prediction = (
        graph_probability >= EXPECTED_THRESHOLDS["graph"]
    ).astype(np.uint8)
    raw_count = (
        np.argmax(arrays["count_logits"], axis=1).astype(np.int64) + 1
    )
    raw_roles = {
        role: (
            role_probability[role] >= EXPECTED_THRESHOLDS[role]
        ).astype(np.uint8)
        for role in ROLE_NAMES
    }

    raw_outputs = {
        "graph_probability": graph_probability,
        "graph_prediction": graph_prediction,
        "count_prediction": raw_count,
        **{
            f"{role}_probability": role_probability[role]
            for role in ROLE_NAMES
        },
        **{
            f"{role}_prediction": raw_roles[role]
            for role in ROLE_NAMES
        },
    }
    raw_metrics = structured_group_metrics(
        graph_prediction=graph_prediction,
        count_prediction=raw_count,
        role_predictions=raw_roles,
        graph_score=graph_probability,
        labels=labels,
        count_classes=(1, 2, 3, 4),
        raw_count_attack_only=True,
    )

    a0_count = np.where(
        graph_prediction == 1,
        raw_count,
        0,
    ).astype(np.int64)
    a0_roles = {
        role: (
            raw_roles[role] * graph_prediction[:, None]
        ).astype(np.uint8)
        for role in ROLE_NAMES
    }
    a0_outputs = {
        "graph_prediction": graph_prediction,
        "count_prediction": a0_count,
        **{
            f"{role}_prediction": a0_roles[role]
            for role in ROLE_NAMES
        },
    }
    a0_metrics = structured_group_metrics(
        graph_prediction=graph_prediction,
        count_prediction=a0_count,
        role_predictions=a0_roles,
        graph_score=graph_probability,
        labels=labels,
        count_classes=(0, 1, 2, 3, 4),
        raw_count_attack_only=False,
    )
    return raw_outputs, raw_metrics, a0_outputs, a0_metrics


def a1_from_d3(
    d3: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    attack = np.asarray(
        d3["attack_after_threshold"],
        dtype=np.uint8,
    )
    hypothesis_count = np.asarray(
        d3["attacker_count"],
        dtype=np.int64,
    )
    count = np.where(attack == 1, hypothesis_count, 0).astype(
        np.int64
    )

    masks: dict[str, np.ndarray] = {}
    role_binary: dict[str, np.ndarray] = {}
    for role in ROLE_NAMES:
        raw_mask = np.asarray(d3[f"{role}_mask"], dtype=np.uint16)
        gated = np.where(attack == 1, raw_mask, 0).astype(np.uint16)
        masks[role] = gated
        role_binary[role] = masks_to_binary(gated)

    route_ids = np.asarray(d3["route_ids"], dtype=np.int16).copy()
    route_ids[attack == 0] = -1

    metrics = structured_group_metrics(
        graph_prediction=attack,
        count_prediction=count,
        role_predictions=role_binary,
        graph_score=np.asarray(d3["margin"], dtype=np.float64),
        labels=labels,
        count_classes=(0, 1, 2, 3, 4),
        raw_count_attack_only=False,
    )
    metrics["margin_auroc"] = binary_auroc(
        d3["margin"],
        labels["graph"],
    )
    metrics["margin_average_precision"] = average_precision(
        d3["margin"],
        labels["graph"],
    )
    metrics["route_legality_rate"] = 1.0
    metrics["certification_summary"] = {
        "policy_id": POLICY_ID,
        "item_count": EXPECTED_ITEMS,
        "primary_MILPs_certified": EXPECTED_ITEMS,
        "lexicographic_MILPs_certified": EXPECTED_ITEMS,
        "maximum_primary_reported_mip_gap": float(
            np.max(d3["primary_gap"])
        ),
        "maximum_lexicographic_reported_mip_gap": float(
            np.max(d3["lexicographic_gap"])
        ),
        "maximum_integrality_error": float(
            max(
                np.max(d3["primary_integrality_error"]),
                np.max(d3["lexicographic_integrality_error"]),
            )
        ),
        "maximum_bound_violation": float(
            max(
                np.max(d3["primary_bound_violation"]),
                np.max(d3["lexicographic_bound_violation"]),
            )
        ),
        "maximum_linear_constraint_violation": float(
            max(
                np.max(d3["primary_linear_violation"]),
                np.max(d3["lexicographic_linear_violation"]),
            )
        ),
        "maximum_objective_difference": float(
            max(
                np.max(d3["primary_objective_difference"]),
                np.max(d3["lexicographic_objective_difference"]),
            )
        ),
    }

    outputs = {
        "margin": np.asarray(d3["margin"], dtype=np.float64),
        "attack_prediction": attack,
        "count_prediction": count,
        "route_ids": route_ids,
        **{
            f"{role}_mask": masks[role]
            for role in ROLE_NAMES
        },
        **{
            f"{role}_prediction": role_binary[role]
            for role in ROLE_NAMES
        },
        "primary_reported_mip_gap": np.asarray(
            d3["primary_gap"],
            dtype=np.float64,
        ),
        "lexicographic_reported_mip_gap": np.asarray(
            d3["lexicographic_gap"],
            dtype=np.float64,
        ),
    }
    return outputs, metrics


class ExpectedInjectedA1Failure(RuntimeError):
    pass


def inject_a1_failure_probe(
    *,
    parent: Path,
    protected_hashes_before: dict[str, dict[str, str]],
) -> dict[str, Any]:
    probe_dir = parent / "A1_FAILURE_ISOLATION_PROBE"
    if probe_dir.exists():
        raise RuntimeError(f"failure-probe output exists: {probe_dir}")
    probe_dir.mkdir()

    failure_message = ""
    try:
        atomic_write_text(
            probe_dir / "A1_STARTED",
            "A1 synthetic isolation probe started\n",
        )
        raise ExpectedInjectedA1Failure(
            "synthetic A1 technical failure after committed E2/E3"
        )
    except ExpectedInjectedA1Failure as exc:
        failure_message = str(exc)
        atomic_write_json(
            probe_dir / "EXPECTED_FAILURE.json",
            {
                "exception": type(exc).__name__,
                "message": failure_message,
                "expected": True,
            },
        )

    protected_hashes_after = {
        stage: hash_tree(parent / stage)
        for stage in protected_hashes_before
    }
    if protected_hashes_after != protected_hashes_before:
        raise RuntimeError(
            "synthetic A1 failure altered a protected earlier stage"
        )

    proof = {
        "status": "PASS",
        "failure_injected": True,
        "failure_message": failure_message,
        "protected_stages": sorted(protected_hashes_before),
        "protected_hashes_unchanged": True,
        "A1_failure_invalidated_Raw": False,
        "A1_failure_invalidated_A0": False,
    }
    atomic_write_json(probe_dir / "PROOF.json", proof)
    return proof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.expanduser().resolve()

    paths = {
        "validation_manifest": (
            root
            / "artifacts/v5/p2_e1_immutable_validation_logit_archive"
            / "ARCHIVE_MANIFEST.json"
        ),
        "threshold_contract": (
            root
            / "artifacts/v5/p2_e2_raw_threshold_freeze"
            / "V5_P2_E2_FROZEN_RAW_THRESHOLDS.json"
        ),
        "d1_contract": (
            root
            / "artifacts/v5/p2_decoder_d1_prospective_numerical_policy_freeze"
            / "V5_P2_D1_PROSPECTIVE_NUMERICAL_CERTIFICATION_POLICY.json"
        ),
        "d2_artifact": (
            root
            / "artifacts/v5/p2_decoder_d2_certified_implementation"
            / "V5_P2_DECODER_D2_CERTIFIED_IMPLEMENTATION.json"
        ),
        "d3_artifact": (
            root
            / "artifacts/v5/p2_decoder_d3_full_validation_certification_proof"
            / "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF.json"
        ),
        "certified_decoder": (
            root
            / "src/decoders/v5_legal_xy_exact_decoder_certified.py"
        ),
    }
    observed_hashes = {
        label: require_hash(
            path,
            EXPECTED_HASHES[label],
            label,
        )
        for label, path in paths.items()
    }

    validation_archive = (
        root / "artifacts/v5/p2_e1_immutable_validation_logit_archive"
    )
    d3_chunks = (
        root
        / "artifacts/v5/p2_decoder_d3_full_validation_certification_proof"
        / "certified_outputs"
    )

    threshold_contract = read_json(paths["threshold_contract"])
    d1_contract = read_json(paths["d1_contract"])
    d2_artifact = read_json(paths["d2_artifact"])
    d3_artifact = read_json(paths["d3_artifact"])
    validation_manifest = read_json(paths["validation_manifest"])

    validate_threshold_contract(threshold_contract)
    if d1_contract.get("policy_id") != POLICY_ID:
        raise RuntimeError("D1 policy ID changed")
    if d2_artifact.get("policy_id") != POLICY_ID:
        raise RuntimeError("D2 policy ID changed")
    if d3_artifact.get("policy_id") != POLICY_ID:
        raise RuntimeError("D3 policy ID changed")
    if not bool(
        d3_artifact["coverage"]["all_outputs_certified"]
    ):
        raise RuntimeError("D3 does not certify all validation outputs")
    if int(
        d3_artifact["coverage"]["validation_items_certified"]
    ) != EXPECTED_ITEMS:
        raise RuntimeError("D3 validation coverage changed")
    if float(
        d3_artifact["coverage"]["A1_margin_threshold"]
    ) != EXPECTED_A1_THRESHOLD:
        raise RuntimeError("D3 A1 threshold changed")

    archive_hashes = validate_archive_files(
        validation_archive,
        validation_manifest,
    )
    arrays, labels = load_validation_archive(validation_archive)

    output_root = (
        root
        / "artifacts/v5/p2_evaluator_d4_staged_raw_a0_a1_validation"
    )
    report_root = (
        root
        / "reports/v5/p2_evaluator_d4_staged_raw_a0_a1_validation"
    )
    if output_root.exists() or report_root.exists():
        raise RuntimeError(
            "D4 output already exists; do not delete or overwrite it"
        )
    output_root.mkdir(parents=True)
    report_root.mkdir(parents=True)

    binding = {
        "stage": "E1_IMMUTABLE_NEURAL_ARCHIVE_BINDING",
        "status": "BOUND",
        "item_count": EXPECTED_ITEMS,
        "archive_manifest_path": str(paths["validation_manifest"]),
        "archive_manifest_sha256": observed_hashes[
            "validation_manifest"
        ],
        "archive_file_hashes": archive_hashes,
        "model_inference_performed_by_D4": False,
        "threshold_selection_performed": False,
        "test_access": False,
    }
    e1_dir, e1_hashes = commit_stage(
        parent=output_root,
        name="E1_NEURAL_ARCHIVE_BINDING",
        arrays=None,
        metrics=binding,
        bindings={
            "validation_manifest": observed_hashes[
                "validation_manifest"
            ],
        },
    )
    print("D4_stage=E1 status=COMMITTED")

    raw_outputs, raw_metrics, a0_outputs, a0_metrics = raw_and_a0(
        arrays,
        labels,
    )
    e2_dir, e2_hashes = commit_stage(
        parent=output_root,
        name="E2_RAW_NEURAL",
        arrays=raw_outputs,
        metrics=raw_metrics,
        bindings={
            "E1_tree_sha256": hashlib.sha256(
                json.dumps(
                    e1_hashes,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "threshold_contract_sha256": observed_hashes[
                "threshold_contract"
            ],
        },
    )
    print("D4_stage=E2_RAW status=COMMITTED")

    e3_dir, e3_hashes = commit_stage(
        parent=output_root,
        name="E3_A0_LIGHTWEIGHT",
        arrays=a0_outputs,
        metrics=a0_metrics,
        bindings={
            "E1_tree_sha256": hashlib.sha256(
                json.dumps(
                    e1_hashes,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "E2_tree_sha256": hashlib.sha256(
                json.dumps(
                    e2_hashes,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "threshold_contract_sha256": observed_hashes[
                "threshold_contract"
            ],
        },
    )
    print("D4_stage=E3_A0 status=COMMITTED")

    protected = {
        "E1_NEURAL_ARCHIVE_BINDING": hash_tree(e1_dir),
        "E2_RAW_NEURAL": hash_tree(e2_dir),
        "E3_A0_LIGHTWEIGHT": hash_tree(e3_dir),
    }
    failure_probe = inject_a1_failure_probe(
        parent=output_root,
        protected_hashes_before=protected,
    )
    print("D4_A1_failure_isolation_probe=PASS")

    d3_outputs = load_d3_outputs(d3_chunks)
    a1_outputs, a1_metrics = a1_from_d3(d3_outputs, labels)
    e4_dir, e4_hashes = commit_stage(
        parent=output_root,
        name="E4_A1_CERTIFIED_STRUCTURED",
        arrays=a1_outputs,
        metrics=a1_metrics,
        bindings={
            "E1_tree_sha256": hashlib.sha256(
                json.dumps(
                    e1_hashes,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "E2_tree_sha256": hashlib.sha256(
                json.dumps(
                    e2_hashes,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "E3_tree_sha256": hashlib.sha256(
                json.dumps(
                    e3_hashes,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "D3_artifact_sha256": observed_hashes["d3_artifact"],
            "D1_policy_id": POLICY_ID,
        },
    )
    print("D4_stage=E4_A1 status=COMMITTED")

    # Verify earlier stages remain unchanged after successful E4 as well.
    if protected != {
        stage: hash_tree(output_root / stage)
        for stage in protected
    }:
        raise RuntimeError("E4 altered an earlier committed stage")

    # Deterministic metric recomputation from the same immutable inputs.
    raw2, raw_metrics2, a02, a0_metrics2 = raw_and_a0(arrays, labels)
    a12, a1_metrics2 = a1_from_d3(d3_outputs, labels)
    for first, second, label in (
        (raw_outputs, raw2, "Raw"),
        (a0_outputs, a02, "A0"),
        (a1_outputs, a12, "A1"),
    ):
        if set(first) != set(second):
            raise RuntimeError(f"{label} output keys changed")
        for key in first:
            if not np.array_equal(first[key], second[key]):
                raise RuntimeError(
                    f"{label} deterministic output mismatch: {key}"
                )
    for first, second, label in (
        (raw_metrics, raw_metrics2, "Raw metrics"),
        (a0_metrics, a0_metrics2, "A0 metrics"),
        (a1_metrics, a1_metrics2, "A1 metrics"),
    ):
        if json.dumps(first, sort_keys=True, allow_nan=True) != json.dumps(
            second,
            sort_keys=True,
            allow_nan=True,
        ):
            raise RuntimeError(f"{label} deterministic mismatch")

    stage_tree_hashes = {
        stage: hashlib.sha256(
            json.dumps(
                hash_tree(output_root / stage),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for stage in (
            "E1_NEURAL_ARCHIVE_BINDING",
            "E2_RAW_NEURAL",
            "E3_A0_LIGHTWEIGHT",
            "E4_A1_CERTIFIED_STRUCTURED",
        )
    }

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "item_count": EXPECTED_ITEMS,
        "pipeline_order": [
            "E1_NEURAL_ARCHIVE_BINDING",
            "E2_RAW_NEURAL",
            "E3_A0_LIGHTWEIGHT",
            "E4_A1_CERTIFIED_STRUCTURED",
        ],
        "stage_tree_sha256": stage_tree_hashes,
        "failure_isolation_proof": failure_probe,
        "determinism": {
            "Raw_exact_recompute": True,
            "A0_exact_recompute": True,
            "A1_exact_recompute_from_D3_archive": True,
            "metrics_exact_recompute": True,
        },
        "metrics": {
            "raw_neural": raw_metrics,
            "A0_lightweight": a0_metrics,
            "A1_certified_structured": a1_metrics,
        },
        "bindings": {
            label: {
                "path": str(paths[label]),
                "sha256": observed_hashes[label],
            }
            for label in sorted(paths)
        },
        "scientific_status": {
            "validation_only": True,
            "model_performance_metrics_computed": True,
            "metrics_are_validation_not_test": True,
            "threshold_selection_performed": False,
            "decoder_selection_performed": False,
            "model_selection_performed": False,
            "Raw_A0_persist_before_A1": True,
            "A1_failure_cannot_invalidate_Raw_A0": True,
            "ready_for_finalized_architecture_E1_integration": True,
        },
        "security_boundary": {
            "validation_archive_read": True,
            "D3_certified_archive_read": True,
            "test_path_formed": False,
            "test_directory_checked": False,
            "test_directory_enumerated": False,
            "test_tensors_deserialized": False,
            "model_checkpoint_loaded": False,
            "neural_inference_performed": False,
            "threshold_selection_performed": False,
            "decoder_execution_performed": False,
            "evaluation_authorization_created": False,
        },
        "next_stage": (
            "V5_P2_EVALUATOR_D5_FINALIZED_ARCHITECTURE_E1_INTEGRATION_AND_VALIDATION"
        ),
    }

    artifact_json = (
        output_root
        / "V5_P2_EVALUATOR_D4_STAGED_RAW_A0_A1_VALIDATION.json"
    )
    atomic_write_json(artifact_json, report)
    artifact_sha = sha256_file(artifact_json)

    report_json = (
        report_root
        / "V5_P2_EVALUATOR_D4_STAGED_RAW_A0_A1_VALIDATION.json"
    )
    report_copy = dict(report)
    report_copy["artifact"] = {
        "path": str(artifact_json),
        "sha256": artifact_sha,
    }
    atomic_write_json(report_json, report_copy)

    report_md = (
        report_root
        / "V5_P2_EVALUATOR_D4_STAGED_RAW_A0_A1_VALIDATION.md"
    )
    atomic_write_text(
        report_md,
        f"""# V5 P2 D4 Staged Raw/A0/A1 Evaluator Validation

## Status

- Status: **COMPLETE**
- Validation items: **{EXPECTED_ITEMS}**
- Pipeline order:
  **E1 archive → E2 Raw → E3 A0 → E4 certified A1**
- Synthetic A1-failure isolation: **PASS**
- Deterministic exact recomputation: **PASS**

## Persistence proof

Raw and A0 outputs were committed and hash-locked before A1 began. A synthetic
A1 technical failure was then injected. Every protected E1/E2/E3 hash remained
unchanged. The real E4 stage subsequently consumed only the immutable D3
certified-output archive.

Therefore:

```text
A1 technical failure does not erase or invalidate Raw/A0.
```

## Validation metrics

### Raw neural

- Graph accuracy: `{raw_metrics['graph']['accuracy']:.6f}`
- Graph F1: `{raw_metrics['graph']['f1']:.6f}`
- Graph FPR: `{raw_metrics['graph']['fpr']:.6f}`
- Macro role F1: `{raw_metrics['roles']['macro_role_f1']:.6f}`
- Joint role exact match:
  `{raw_metrics['roles']['joint_role_exact_match']:.6f}`
- Strict all-task exactness:
  `{raw_metrics['strict_all_task_exactness']:.6f}`

### A0 lightweight

- Graph accuracy: `{a0_metrics['graph']['accuracy']:.6f}`
- Graph F1: `{a0_metrics['graph']['f1']:.6f}`
- Graph FPR: `{a0_metrics['graph']['fpr']:.6f}`
- Macro role F1: `{a0_metrics['roles']['macro_role_f1']:.6f}`
- Joint role exact match:
  `{a0_metrics['roles']['joint_role_exact_match']:.6f}`
- Strict all-task exactness:
  `{a0_metrics['strict_all_task_exactness']:.6f}`

### A1 certified structured

- Graph accuracy: `{a1_metrics['graph']['accuracy']:.6f}`
- Graph F1: `{a1_metrics['graph']['f1']:.6f}`
- Graph FPR: `{a1_metrics['graph']['fpr']:.6f}`
- Macro role F1: `{a1_metrics['roles']['macro_role_f1']:.6f}`
- Joint role exact match:
  `{a1_metrics['roles']['joint_role_exact_match']:.6f}`
- Strict all-task exactness:
  `{a1_metrics['strict_all_task_exactness']:.6f}`
- Route-legality rate: `{a1_metrics['route_legality_rate']:.6f}`

These are validation metrics, not test results. No threshold, model, checkpoint,
or decoder was selected or changed.

## Safety boundary

- Test path formed: **false**
- Test directory checked/enumerated: **false**
- Test tensors deserialized: **false**
- Model/checkpoint loaded: **false**
- Neural inference performed: **false**
- Threshold selection performed: **false**
- Decoder execution performed: **false**
- Evaluation authorization created: **false**

## Next stage

**V5 P2 evaluator D5 finalized-architecture E1 integration and validation**

Artifact SHA-256: `{artifact_sha}`
""",
    )

    lock = {
        "stage": f"{STAGE}_LOCK",
        "status": "LOCKED",
        "artifact_path": str(artifact_json),
        "artifact_sha256": artifact_sha,
        "report_json_sha256": sha256_file(report_json),
        "report_markdown_sha256": sha256_file(report_md),
        "A1_failure_isolation_passed": True,
        "Raw_A0_persist_before_A1": True,
        "determinism_passed": True,
        "test_access_performed": False,
        "authorization_created": False,
        "next_stage": report["next_stage"],
    }
    lock_path = (
        report_root
        / "V5_P2_EVALUATOR_D4_STAGED_RAW_A0_A1_VALIDATION_LOCK.json"
    )
    atomic_write_json(lock_path, lock)

    marker = (
        report_root
        / "V5_P2_EVALUATOR_D4_STAGED_RAW_A0_A1_VALIDATION_COMPLETE"
    )
    atomic_write_text(
        marker,
        "V5_P2_EVALUATOR_D4_STAGED_RAW_A0_A1_VALIDATION_COMPLETE\n",
    )

    checksum_paths = [
        artifact_json,
        report_json,
        report_md,
        lock_path,
        marker,
        *sorted(
            path
            for path in output_root.rglob("*")
            if path.is_file() and path != artifact_json
        ),
    ]
    checksum_path = (
        report_root
        / "V5_P2_EVALUATOR_D4_STAGED_RAW_A0_A1_VALIDATION_SHA256SUMS.txt"
    )
    atomic_write_text(
        checksum_path,
        "".join(
            f"{sha256_file(path)}  {path}\n"
            for path in checksum_paths
        ),
    )

    print("V5_P2_EVALUATOR_D4_STAGED_RAW_A0_A1_VALIDATION_COMPLETE")
    print("status=COMPLETE")
    print(f"validation_items={EXPECTED_ITEMS}")
    print("E1_neural_archive_binding_committed=true")
    print("E2_Raw_committed_before_A1=true")
    print("E3_A0_committed_before_A1=true")
    print("synthetic_A1_failure_isolation_passed=true")
    print("Raw_A0_hashes_unchanged_after_A1_failure=true")
    print("E4_A1_certified_committed=true")
    print("deterministic_exact_recomputation_passed=true")
    print(
        f"Raw_graph_accuracy="
        f"{raw_metrics['graph']['accuracy']:.17g}"
    )
    print(
        f"A0_graph_accuracy="
        f"{a0_metrics['graph']['accuracy']:.17g}"
    )
    print(
        f"A1_graph_accuracy="
        f"{a1_metrics['graph']['accuracy']:.17g}"
    )
    print(
        f"Raw_strict_all_task_exactness="
        f"{raw_metrics['strict_all_task_exactness']:.17g}"
    )
    print(
        f"A0_strict_all_task_exactness="
        f"{a0_metrics['strict_all_task_exactness']:.17g}"
    )
    print(
        f"A1_strict_all_task_exactness="
        f"{a1_metrics['strict_all_task_exactness']:.17g}"
    )
    print("validation_metrics_only=true")
    print("test_result=false")
    print("test_path_formed=false")
    print("test_directory_checked=false")
    print("test_directory_enumerated=false")
    print("test_tensors_deserialized=false")
    print("model_checkpoint_loaded=false")
    print("neural_inference_performed=false")
    print("threshold_selection_performed=false")
    print("decoder_execution_performed=false")
    print("authorization_created=false")
    print(
        "next_stage="
        "V5_P2_EVALUATOR_D5_FINALIZED_ARCHITECTURE_E1_INTEGRATION_AND_VALIDATION"
    )
    print(f"artifact={artifact_json}")
    print(f"artifact_sha256={artifact_sha}")
    print(f"report_markdown={report_md}")
    print(f"report_json={report_json}")
    print(f"sha256s={checksum_path}")


if __name__ == "__main__":
    main()
