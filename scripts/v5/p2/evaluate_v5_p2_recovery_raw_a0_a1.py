#!/usr/bin/env python3
"""Prospectively amended V5 P2 Raw/A0/A1 recovery blind-test evaluator.

This recovery source is frozen before a new recovery authorization. It contains no training, checkpoint
selection, threshold search, decoder selection, or A2 execution.

Security sequence for the authorized blind run
----------------------------------------------
1. Verify frozen non-test prerequisites, evaluator source, authorization
   contract, and single-use authorization token.
2. Refuse any pre-existing output directory.
3. Create the immutable output guard directory.
4. Atomically consume the authorization token.
5. Write TEST_ACCESS_STARTED.
6. Only then resolve/check/enumerate ``data_root/runs/test`` using the pre-existing B6 filename-pairing rule.
7. Execute exactly one deterministic neural inference pass over every frozen
   pair-aligned test window.
8. Produce Raw neural, A0 lightweight, and A1 exact-structured outputs.
9. Write immutable prediction cache, primary metrics, lock, and completion
   marker. Successful reruns are forbidden.

A source-only self-test mode verifies model/checkpoint/decoder integration using
synthetic tensors and does not resolve or inspect a test path.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


STAGE = "V5_P2_RECOVERY_RAW_A0_A1_BLIND_EVALUATION"
COMPLETE = f"{STAGE}_COMPLETE"
ACCESS_STARTED = f"{STAGE}_TEST_ACCESS_STARTED"
IRREVERSIBLE_FAILURE = f"{STAGE}_IRREVERSIBLE_FAILURE"

SELECTED_SEED = 107
SELECTED_EPOCH = 25
EXPECTED_PARAMETER_COUNT = 59_785
EXPECTED_STATE_TENSOR_COUNT = 49
EXPECTED_STATE_NUMEL = 59_881

WINDOW = 32
STRIDE = 8
BATCH_SIZE = 128
EXPECTED_TEST_FILES = 138
EXPECTED_TEST_PAIRS = 69
ROUTER_COUNT = 16
ROLE_NAMES = ("source", "transit", "victim", "path")

PRIMARY58_INDICES = tuple(
    list(range(0, 30))
    + list(range(31, 56))
    + [62, 64, 65]
)

A0_THRESHOLDS = {
    "graph": 0.47174675035328983,
    "source": 0.94960549299285935,
    "transit": 0.85703332488359107,
    "victim": 0.82601148026998117,
    "path": 0.83393474660904676,
}
A1_MARGIN_THRESHOLD = 8.7205320882398425

EXPECTED_HASHES = {
    "l5_contract": "a9cd7b7d437e7e362b1bb942796df9773a603c90ab95e4b6647b9049114fc078",
    "checkpoint": "82314baedb4bf3969842abf9076679682369daae2e57db07d4b4aeaa8623f7cc",
    "task_d_model": "ccdfcb74c1ddab7f20b98c922444873ad76c204ec4b5fd6827d52fb716f11fd9",
    "b3_model": "56ee3207d039b60e8e3a898a689cd8e361247c86b7450a3a5e95f423ebe30def",
    "edge_index": "f6b8050bc158de509b0ff1c5d1d7cb1ffe32c08b0f2287398270b1b891b57aff",
    "physical_mask": "a93f81a9ce4315d0eef911f9b9c45529ceb9b2da726dd578a247b46d260e6a3a",
    "manifest": "f42f40446d03161a6932ee060f6d8c859f894cb5075d1fc926ea161461413ab5",
    "exact_decoder": "8da32b3ca3915365b8b01049a4ddc6a043583aa7193589ccf4af8116a4b0783c",
    "route_library": "3ae860817ddef7cdbc0360057c51dc86908337b40b1df4c7df8fc2c8b82ea0d7",
    "route_table_json": "19624d95a83114fba1e27e647aab59a0231c97d45053b598692ae62578b321b0",
    "dependency_interface": "92fdeecacb9e93fd52cd485a9a52f7886ab4c4ca9e0bd20dae3e6a4e622c9d76",
    "recovery_amendment": "b2be725493603328bf7ae56a7ba0b8d6bdb3dc5aadc9ffdc52e3c3fc3cfde095",
    "preexisting_b6_reference": "29b5efbbbbadaee55984e4f80f96fc1cb23be2c09a9c1eda8b212b674d0480f1",
    "f0_disposition": "47331afc6523474fbb98dcb44157be71fda85488da5a078be8c98f285db89b50",
}

TARGET_KEYS = {
    "source": "y_source",
    "transit": "y_transit",
    "victim": "y_victim",
    "path": "y_attack_path",
}
LOGIT_KEYS = {
    "source": "source_logits",
    "transit": "transit_logits",
    "victim": "victim_logits",
    "path": "path_logits",
}

REQUIRED_ITEM_KEYS = {
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
REQUIRED_OUTPUT_KEYS = {
    "attack_logits",
    "count_logits",
    "source_logits",
    "transit_logits",
    "victim_logits",
    "path_logits",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected JSON object")
    return value


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected={expected}, observed={observed}"
        )


def import_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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


@dataclass(frozen=True)
class TestIndexEntry:
    file_path: Path
    pair_key: str
    pair_ordinal: int
    mode: str
    start: int
    target: int
    common_length: int


def validate_run_payload(payload: Any, path: Path) -> int:
    if not isinstance(payload, dict):
        raise TypeError(f"run payload is not a dictionary: {path}")
    x = payload.get("x")
    if (
        not isinstance(x, torch.Tensor)
        or x.dtype != torch.float32
        or x.ndim != 3
        or tuple(x.shape[1:]) != (16, 81)
    ):
        raise ValueError(f"invalid x tensor in {path}")
    length = int(x.shape[0])

    for key in ("y_attack", "y_attacker_count"):
        value = payload.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 1
            or int(value.shape[0]) != length
        ):
            raise ValueError(f"invalid {key} in {path}")

    for key in (
        "y_source",
        "y_transit",
        "y_victim",
        "y_attack_path",
        "role_mask",
    ):
        value = payload.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or tuple(value.shape) != (length, 16)
        ):
            raise ValueError(f"invalid {key} in {path}")
    return length


def build_recovery_pair_index(
    test_dir: Path,
) -> list[dict[str, Path | str]]:
    files = sorted(
        path
        for path in test_dir.iterdir()
        if path.is_file() and path.suffix == ".pt"
    )
    if len(files) != EXPECTED_TEST_FILES:
        raise RuntimeError(
            f"test .pt file count={len(files)}, expected {EXPECTED_TEST_FILES}"
        )

    pairs: dict[str, dict[str, Path]] = {}
    for path in files:
        name = path.name
        if name.endswith("_ATTACK.pt"):
            pair_key = name[:-len("_ATTACK.pt")]
            mode = "attack"
        elif name.endswith("_CONTROL.pt"):
            pair_key = name[:-len("_CONTROL.pt")]
            mode = "control"
        else:
            raise RuntimeError(f"unexpected test tensor filename: {name}")
        if not pair_key:
            raise RuntimeError(f"empty pair key in test filename: {name}")
        pair = pairs.setdefault(pair_key, {})
        if mode in pair:
            raise RuntimeError(f"duplicate {mode} tensor for {pair_key}")
        pair[mode] = path

    if len(pairs) != EXPECTED_TEST_PAIRS:
        raise RuntimeError(
            f"test pair count={len(pairs)}, expected {EXPECTED_TEST_PAIRS}"
        )

    records: list[dict[str, Path | str]] = []
    for pair_key in sorted(pairs):
        pair = pairs[pair_key]
        if set(pair) != {"attack", "control"}:
            raise RuntimeError(f"incomplete test pair: {pair_key}")
        records.append(
            {
                "pair_key": pair_key,
                "attack": pair["attack"],
                "control": pair["control"],
            }
        )
    return records


class FrozenP2RecoveryTestDataset(Dataset):
    """B6-paired P2 recovery loader; instantiate only after authorization."""

    def __init__(
        self,
        *,
        data_root: Path,
        physical_port_mask: torch.Tensor,
    ) -> None:
        self.data_root = data_root
        self.physical_port_mask = (
            physical_port_mask.detach().cpu().bool().contiguous()
        )
        if tuple(self.physical_port_mask.shape) != (16, 10):
            raise ValueError("physical-port mask must be [16,10]")
        if int(self.physical_port_mask.sum()) != 128:
            raise ValueError("physical-port mask true count must be 128")
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"dataset root missing: {self.data_root}")

        test_dir = self.data_root / "runs" / "test"
        if not test_dir.is_dir():
            raise FileNotFoundError(f"test split missing: {test_dir}")

        pair_records = build_recovery_pair_index(test_dir)
        self._index: list[TestIndexEntry] = []
        self._pair_rows: list[dict[str, Any]] = []

        for pair_ordinal, record in enumerate(pair_records):
            pair_key = str(record["pair_key"])
            attack_path = Path(record["attack"])
            control_path = Path(record["control"])
            attack_run = self._load_run(attack_path)
            control_run = self._load_run(control_path)
            attack_length = validate_run_payload(attack_run, attack_path)
            control_length = validate_run_payload(control_run, control_path)
            common_length = min(attack_length, control_length)
            starts = list(range(0, common_length - WINDOW + 1, STRIDE))
            if not starts:
                raise RuntimeError(f"test pair {pair_key} produces no windows")

            self._pair_rows.append(
                {
                    "pair_ordinal": pair_ordinal,
                    "pair_key": pair_key,
                    "attack_length": attack_length,
                    "control_length": control_length,
                    "common_length": common_length,
                    "window_count_per_member": len(starts),
                    "aligned_item_count": 2 * len(starts),
                    "window": WINDOW,
                    "stride": STRIDE,
                    "attack_file_name": attack_path.name,
                    "control_file_name": control_path.name,
                    "pair_index_source": "preexisting_B6_filename_pairing_rule",
                }
            )
            for start in starts:
                target = start + WINDOW - 1
                for mode, file_path in (
                    ("attack", attack_path),
                    ("control", control_path),
                ):
                    self._index.append(
                        TestIndexEntry(
                            file_path=file_path,
                            pair_key=pair_key,
                            pair_ordinal=pair_ordinal,
                            mode=mode,
                            start=start,
                            target=target,
                            common_length=common_length,
                        )
                    )

        if len(self._pair_rows) != EXPECTED_TEST_PAIRS:
            raise RuntimeError("recovery pair count changed after indexing")
        if not self._index or len(self._index) % 2 != 0:
            raise RuntimeError("invalid pair-aligned recovery test index")

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_run(path: Path) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_run_payload(payload, path)
        return payload

    def __len__(self) -> int:
        return len(self._index)

    @property
    def pair_rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._pair_rows]

    def stable_metadata(self, index: int) -> dict[str, Any]:
        entry = self._index[index]
        return {
            "test_item_index": index,
            "pair_ordinal": entry.pair_ordinal,
            "stable_item_id": (
                f"recovery-test/{entry.pair_key}/{entry.mode}/"
                f"start={entry.start}/target={entry.target}"
            ),
            "split": "test",
            "pair_key": entry.pair_key,
            "pair_id": entry.pair_key,
            "category": "recovery_test",
            "mode": entry.mode,
            "start": entry.start,
            "target": entry.target,
            "common_length": entry.common_length,
            "file_name": entry.file_path.name,
            "pair_index_source": "preexisting_B6_filename_pairing_rule",
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self._index[index]
        run = self._load_run(entry.file_path)
        if int(run["x"].shape[0]) < entry.common_length:
            raise ValueError(
                f"{entry.file_path}: run shorter than frozen common length"
            )
        stop = entry.start + WINDOW
        x = (
            run["x"][entry.start:stop, :, PRIMARY58_INDICES]
            .permute(1, 2, 0)
            .contiguous()
        )
        if tuple(x.shape) != (16, 58, 32):
            raise RuntimeError(f"constructed x shape={tuple(x.shape)}")

        item = {
            "x": x,
            "physical_port_mask": self.physical_port_mask.clone(),
            "y_attack": run["y_attack"][entry.target].clone(),
            "y_attacker_count": run["y_attacker_count"][entry.target].clone(),
            "y_source": run["y_source"][entry.target].clone(),
            "y_transit": run["y_transit"][entry.target].clone(),
            "y_victim": run["y_victim"][entry.target].clone(),
            "y_attack_path": run["y_attack_path"][entry.target].clone(),
            "role_mask": run["role_mask"][entry.target].clone(),
        }
        if set(item) != REQUIRED_ITEM_KEYS:
            raise RuntimeError("recovery loader item-key contract changed")
        return item


def repository_paths(root: Path) -> dict[str, Path]:
    reports = root / "reports/v5"
    artifacts = root / "artifacts/v5"
    return {
        "l5_contract": (
            reports
            / "p2_l5_final_pretest_freeze"
            / "V5_P2_ONE_SHOT_BLIND_EVALUATION_CONTRACT.json"
        ),
        "checkpoint": (
            reports
            / "p2_task_d_checkpoint_selection_freeze"
            / "selected_graphconv_checkpoint.pt"
        ),
        "task_d_model": root / "src/models/v5_p2_task_d_full_multitask_count4.py",
        "b3_model": root / "src/models/v5_p2_b3_conv1d_only_count4.py",
        "edge_index": (
            reports
            / "p2_g1a_r2a_canonical_static_topology_contract"
            / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy"
        ),
        "physical_mask": (
            reports
            / "p2_a2_r2_feature_normalization_mask_contract"
            / "V5_P2_A2_R2_TOPOLOGY_DERIVED_BOOLEAN_PORT_MASK.pt"
        ),
        "manifest": (
            reports
            / "p2_a1_r2_pair_aligned_window_contract"
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
        ),
        "exact_decoder": root / "src/decoders/v5_legal_xy_exact_decoder.py",
        "route_library": root / "src/decoders/v5_xy_route_library.py",
        "route_table_json": artifacts / "decoders/xy4x4_route_table.json",
        "dependency_interface": (
            artifacts
            / "p2_preflight_missing_dependency_capture"
            / "V5_P2_PREFLIGHT_FROZEN_DEPENDENCY_INTERFACE.json"
        ),
        "recovery_amendment": (
            artifacts
            / "p2_prospective_recovery_protocol_amendment"
            / "V5_P2_PROSPECTIVE_RECOVERY_PROTOCOL_AMENDMENT.json"
        ),
        "preexisting_b6_reference": (
            root / "scripts/v5/p2/evaluate_v5_p2_b6_one_shot_test.py"
        ),
        "f0_disposition": (
            reports
            / "p2_final_f0_irreversible_failure_disposition"
            / "V5_P2_FINAL_F0_IRREVERSIBLE_FAILURE_DISPOSITION.json"
        ),
    }


def verify_frozen_repository(root: Path) -> dict[str, Path]:
    paths = repository_paths(root)
    for name, expected in EXPECTED_HASHES.items():
        require_hash(paths[name], expected)

    contract = load_json(paths["l5_contract"])
    if contract.get("status") != "FROZEN":
        raise RuntimeError("L5 one-shot contract is not frozen")
    if contract.get("official_output_groups") != [
        "raw_neural",
        "A0_lightweight",
        "A1_exact_structured",
    ]:
        raise RuntimeError("official output groups changed")
    if contract.get("A2", {}).get("official_execution_allowed") is not False:
        raise RuntimeError("A2 rejection changed")
    if contract.get("neural_execution", {}).get("single_inference_pass") is not True:
        raise RuntimeError("single-inference-pass contract changed")
    amendment = load_json(paths["recovery_amendment"])
    if amendment.get("status") != "FROZEN":
        raise RuntimeError("recovery protocol amendment is not frozen")
    if amendment.get("scope_of_change", {}).get("allowed_change_count") != 1:
        raise RuntimeError("recovery amendment scope changed")
    if amendment.get("recovery_test_pairing_contract", {}).get("expected_test_tensor_files") != 138:
        raise RuntimeError("recovery expected test-file count changed")
    if amendment.get("recovery_test_pairing_contract", {}).get("expected_complete_pairs") != 69:
        raise RuntimeError("recovery expected pair count changed")
    if amendment.get("security_boundary_at_amendment_freeze", {}).get("test_directory_enumerated") is not False:
        raise RuntimeError("recovery-amendment test boundary changed")
    f0 = load_json(paths["f0_disposition"])
    if f0.get("classification") != "PRE_INFERENCE_MANIFEST_WIRING_ABORT":
        raise RuntimeError("F0 failure classification changed")
    return paths


def load_edge_index(path: Path) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise TypeError("edge-index artifact is not an ndarray")
    if array.dtype != np.int64 or tuple(array.shape) != (2, 48):
        raise ValueError(
            f"edge-index contract changed: dtype={array.dtype}, shape={array.shape}"
        )
    tensor = torch.from_numpy(np.ascontiguousarray(array)).long()
    pairs = {(int(a), int(b)) for a, b in tensor.t().tolist()}
    if len(pairs) != 48:
        raise ValueError("edge index does not contain 48 unique directed edges")
    return tensor.contiguous()


def load_physical_mask(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("mask"), torch.Tensor):
        raise TypeError("physical-mask artifact must contain tensor key 'mask'")
    mask = payload["mask"].detach().cpu().bool().contiguous()
    if tuple(mask.shape) != (16, 10) or int(mask.sum()) != 128:
        raise ValueError("physical-mask contract changed")
    return mask


def checkpoint_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise TypeError("checkpoint model_state_dict is absent")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("checkpoint state contains non-tensors")
    return state


def build_frozen_model(
    *,
    paths: dict[str, Path],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    b3_module = import_source(paths["b3_model"], "v5_p2_final_b3")
    task_module = import_source(paths["task_d_model"], "v5_p2_final_task_d")
    edge = load_edge_index(paths["edge_index"])
    reference = b3_module.P2B3Conv1DOnlyCount4()
    model = task_module.P2TaskDGraphConvCount4(reference, edge).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"parameter count={parameter_count}, expected={EXPECTED_PARAMETER_COUNT}"
        )

    checkpoint = torch.load(
        paths["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint_state_dict(checkpoint)
    if len(state) != EXPECTED_STATE_TENSOR_COUNT:
        raise RuntimeError("checkpoint tensor count changed")
    state_numel = sum(int(value.numel()) for value in state.values())
    if state_numel != EXPECTED_STATE_NUMEL:
        raise RuntimeError("checkpoint total state elements changed")

    expected_metadata = {
        "candidate": "graphconv",
        "seed": SELECTED_SEED,
        "epoch": SELECTED_EPOCH,
        "threshold_tuning_performed": False,
        "test_tensors_deserialized": False,
    }
    for key, expected in expected_metadata.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(
                f"checkpoint metadata mismatch: {key}={checkpoint.get(key)!r}"
            )

    if checkpoint.get("b3_model_sha256") != EXPECTED_HASHES["b3_model"]:
        raise RuntimeError("checkpoint B3 source hash changed")
    if checkpoint.get("task_d_model_sha256") != EXPECTED_HASHES["task_d_model"]:
        raise RuntimeError("checkpoint Task-D source hash changed")

    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint load returned incompatible keys")
    model.eval()
    return model, {
        "parameter_count": parameter_count,
        "state_tensor_count": len(state),
        "state_total_numel": state_numel,
        "checkpoint_top_level_keys": sorted(str(key) for key in checkpoint),
    }


def run_source_only_self_test(root: Path) -> dict[str, Any]:
    paths = verify_frozen_repository(root)

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")

    model, model_summary = build_frozen_model(paths=paths, device=device)
    mask = load_physical_mask(paths["physical_mask"])
    x = torch.zeros((2, 16, 58, 32), dtype=torch.float32)
    batch_mask = mask.unsqueeze(0).expand(2, -1, -1).clone()
    with torch.inference_mode():
        first = model(x, batch_mask)
        second = model(x, batch_mask)
    if set(first) != REQUIRED_OUTPUT_KEYS:
        raise RuntimeError("synthetic output-key contract changed")
    expected_shapes = {
        "attack_logits": (2,),
        "count_logits": (2, 4),
        "source_logits": (2, 16),
        "transit_logits": (2, 16),
        "victim_logits": (2, 16),
        "path_logits": (2, 16),
    }
    for key, expected_shape in expected_shapes.items():
        if tuple(first[key].shape) != expected_shape:
            raise RuntimeError(f"{key} shape changed")
        if not torch.equal(first[key], second[key]):
            raise RuntimeError(f"deterministic repeated inference failed: {key}")
        if not bool(torch.isfinite(first[key]).all()):
            raise RuntimeError(f"non-finite synthetic output: {key}")

    route_module = import_source(
        paths["route_library"],
        "src.decoders.v5_xy_route_library",
    )
    exact_module = import_source(
        paths["exact_decoder"],
        "src.decoders.v5_legal_xy_exact_decoder",
    )
    if len(route_module.ROUTE_TABLE) != 240:
        raise RuntimeError("route-library count changed")
    hypothesis = exact_module.decode_best_attack_hypothesis(
        graph_logit=0.0,
        count_logits=[0.0, 0.0, 0.0, 0.0],
        source_logits=[0.0] * 16,
        transit_logits=[0.0] * 16,
        victim_logits=[0.0] * 16,
        path_logits=[0.0] * 16,
    )
    if not (
        hypothesis.optimality_proven
        and hypothesis.lexicographic_tie_break_certified
        and hypothesis.high_precision_semantic_face_certified
    ):
        raise RuntimeError("synthetic A1 exact-decoder certification failed")
    decoded = exact_module.apply_margin_threshold(
        hypothesis,
        A1_MARGIN_THRESHOLD,
    )
    if decoded.attack not in (0, 1):
        raise RuntimeError("invalid synthetic A1 attack output")

    metric_check = binary_metrics(
        np.asarray([0, 1, 1, 0], dtype=np.uint8),
        np.asarray([0, 1, 0, 0], dtype=np.uint8),
    )
    if metric_check["tp"] != 1 or metric_check["fp"] != 1:
        raise RuntimeError("binary metric self-test failed")

    def synthetic_run(length: int, attack: bool) -> dict[str, torch.Tensor]:
        x = torch.arange(length, dtype=torch.float32).view(length, 1, 1)
        x = x.expand(length, 16, 81).clone()
        y_attack = torch.full((length,), int(attack), dtype=torch.int64)
        y_count = torch.full((length,), 1 if attack else 0, dtype=torch.int64)
        source = torch.zeros((length, 16), dtype=torch.int64)
        transit = torch.zeros((length, 16), dtype=torch.int64)
        victim = torch.zeros((length, 16), dtype=torch.int64)
        path = torch.zeros((length, 16), dtype=torch.int64)
        if attack:
            source[:, 0] = 1
            transit[:, 1] = 1
            victim[:, 2] = 1
            path[:, :3] = 1
        role_mask = source + 2 * transit + 4 * victim
        return {
            "x": x,
            "y_attack": y_attack,
            "y_attacker_count": y_count,
            "y_source": source,
            "y_transit": transit,
            "y_victim": victim,
            "y_attack_path": path,
            "role_mask": role_mask,
        }

    with tempfile.TemporaryDirectory(prefix="v5_p2_recovery_loader_self_test_") as temporary:
        synthetic_root = Path(temporary)
        synthetic_test = synthetic_root / "runs" / "test"
        synthetic_test.mkdir(parents=True)
        attack_template = synthetic_root / "attack_template.pt"
        control_template = synthetic_root / "control_template.pt"
        torch.save(synthetic_run(40, True), attack_template)
        torch.save(synthetic_run(48, False), control_template)
        for ordinal in range(EXPECTED_TEST_PAIRS):
            key = f"pair{ordinal:03d}"
            os.link(attack_template, synthetic_test / f"{key}_ATTACK.pt")
            os.link(control_template, synthetic_test / f"{key}_CONTROL.pt")
        pair_records = build_recovery_pair_index(synthetic_test)
        if [str(row["pair_key"]) for row in pair_records] != [f"pair{i:03d}" for i in range(69)]:
            raise RuntimeError("synthetic recovery pair order changed")
        synthetic_dataset = FrozenP2RecoveryTestDataset(
            data_root=synthetic_root,
            physical_port_mask=mask,
        )
        if len(synthetic_dataset.pair_rows) != 69 or len(synthetic_dataset) != 276:
            raise RuntimeError("synthetic recovery index cardinality changed")
        first_metadata = [synthetic_dataset.stable_metadata(i) for i in range(4)]
        if [row["mode"] for row in first_metadata] != ["attack", "control", "attack", "control"]:
            raise RuntimeError("synthetic recovery item ordering changed")
        if [row["start"] for row in first_metadata] != [0, 0, 8, 8]:
            raise RuntimeError("synthetic recovery window starts changed")
        if tuple(synthetic_dataset[0]["x"].shape) != (16, 58, 32):
            raise RuntimeError("synthetic recovery item shape changed")

    return {
        "status": "PASS",
        "test_path_resolved": False,
        "test_directory_existence_checked": False,
        "test_directory_enumerated": False,
        "test_tensors_deserialized": False,
        "model": model_summary,
        "synthetic_forward_exact_repeat": True,
        "route_count": len(route_module.ROUTE_TABLE),
        "synthetic_A1_certified": True,
        "synthetic_recovery_pairing_passed": True,
        "synthetic_recovery_pair_count": 69,
        "synthetic_recovery_item_count": 276,
        "real_test_path_resolved": False,
        "solver_identity": dataclasses.asdict(hypothesis.solver_identity),
    }


def validate_authorization(
    *,
    contract_path: Path,
    token_path: Path,
    evaluator_sha256: str,
    data_root_text: str,
    output_dir_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not token_path.is_file():
        raise FileNotFoundError(token_path)
    contract = load_json(contract_path)
    token = load_json(token_path)

    if contract.get("stage") != "V5_P2_RECOVERY_ONE_SHOT_AUTHORIZATION":
        raise RuntimeError("recovery authorization contract stage mismatch")
    if contract.get("status") != "AUTHORIZED":
        raise RuntimeError("authorization contract is not AUTHORIZED")
    if contract.get("single_use") is not True:
        raise RuntimeError("authorization is not single-use")
    if contract.get("test_access_authorized") is not True:
        raise RuntimeError("test access is not authorized")
    if int(contract.get("evaluation_count", -1)) != 1:
        raise RuntimeError("authorization evaluation_count is not one")
    if contract.get("evaluator_source_sha256") != evaluator_sha256:
        raise RuntimeError("authorization is not bound to this evaluator source")
    if contract.get("checkpoint_sha256") != EXPECTED_HASHES["checkpoint"]:
        raise RuntimeError("authorization checkpoint hash changed")
    if contract.get("l5_contract_sha256") != EXPECTED_HASHES["l5_contract"]:
        raise RuntimeError("authorization L5 contract hash changed")
    if contract.get("recovery_protocol_amendment_sha256") != EXPECTED_HASHES["recovery_amendment"]:
        raise RuntimeError("recovery amendment authorization binding changed")
    if contract.get("original_authorization_id") != "V5P2-caaea91e4eb6772ace0dc1b43b7e28c4":
        raise RuntimeError("original authorization disclosure changed")
    if contract.get("recovery_after_pre_inference_abort") is not True:
        raise RuntimeError("recovery disclosure is absent")
    if contract.get("data_root_text_sha256") != sha256_text(data_root_text):
        raise RuntimeError("authorization data-root binding mismatch")
    if contract.get("output_dir_text_sha256") != sha256_text(output_dir_text):
        raise RuntimeError("authorization output-dir binding mismatch")
    if contract.get("authorization_token_sha256") != sha256_file(token_path):
        raise RuntimeError("authorization token hash mismatch")

    required_token_fields = {
        "authorization_id",
        "single_use",
        "test_access_authorized",
        "evaluator_source_sha256",
        "data_root_text_sha256",
        "output_dir_text_sha256",
        "recovery_protocol_amendment_sha256",
        "recovery_after_pre_inference_abort",
    }
    if not required_token_fields.issubset(token):
        raise RuntimeError("authorization token fields are incomplete")
    if token.get("single_use") is not True:
        raise RuntimeError("token is not single-use")
    if token.get("test_access_authorized") is not True:
        raise RuntimeError("token does not authorize test access")
    if token.get("evaluator_source_sha256") != evaluator_sha256:
        raise RuntimeError("token evaluator binding mismatch")
    if token.get("data_root_text_sha256") != sha256_text(data_root_text):
        raise RuntimeError("token data-root binding mismatch")
    if token.get("output_dir_text_sha256") != sha256_text(output_dir_text):
        raise RuntimeError("token output-dir binding mismatch")
    if token.get("recovery_protocol_amendment_sha256") != EXPECTED_HASHES["recovery_amendment"]:
        raise RuntimeError("token recovery-amendment binding mismatch")
    if token.get("recovery_after_pre_inference_abort") is not True:
        raise RuntimeError("token recovery disclosure is absent")
    if token.get("authorization_id") != contract.get("authorization_id"):
        raise RuntimeError("authorization ID mismatch")
    return contract, token


def consume_authorization_token(token_path: Path) -> Path:
    consumed = token_path.with_name(token_path.name + ".consumed")
    if consumed.exists():
        raise RuntimeError(f"authorization token already consumed: {consumed}")
    os.replace(token_path, consumed)
    consumed.chmod(0o444)
    return consumed


def save_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    if value.dtype == object:
        raise TypeError(f"object arrays are forbidden: {path.name}")
    np.save(path, value, allow_pickle=False)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": path.stat().st_size,
    }


def concatenate_chunks(
    chunks: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate(parts, axis=0)
        for key, parts in chunks.items()
    }


def perform_single_inference_pass(
    *,
    dataset: FrozenP2RecoveryTestDataset,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    chunks: dict[str, list[np.ndarray]] = {
        "graph_logits": [],
        "count_logits": [],
        "source_logits": [],
        "transit_logits": [],
        "victim_logits": [],
        "path_logits": [],
        "y_graph": [],
        "y_attacker_count": [],
        "y_source": [],
        "y_transit": [],
        "y_victim": [],
        "y_path": [],
        "role_mask": [],
    }
    metadata: list[dict[str, Any]] = []
    emitted = 0
    first_batch_repeat_verified = False

    model.eval()
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader):
            if set(batch) != REQUIRED_ITEM_KEYS:
                raise RuntimeError("test batch-key contract changed")
            x = batch["x"].to(device=device, dtype=torch.float32)
            mask = batch["physical_port_mask"].to(device=device)
            outputs = model(x, mask)
            if set(outputs) != REQUIRED_OUTPUT_KEYS:
                raise RuntimeError("model output-key contract changed")

            if not first_batch_repeat_verified:
                repeated = model(x, mask)
                for key in REQUIRED_OUTPUT_KEYS:
                    if not torch.equal(outputs[key], repeated[key]):
                        raise RuntimeError(
                            f"first-batch deterministic repeat failed: {key}"
                        )
                first_batch_repeat_verified = True

            batch_size = int(x.shape[0])
            indices = range(emitted, emitted + batch_size)
            metadata.extend(dataset.stable_metadata(index) for index in indices)

            chunks["graph_logits"].append(
                outputs["attack_logits"].cpu().numpy().astype(np.float32)
            )
            chunks["count_logits"].append(
                outputs["count_logits"].cpu().numpy().astype(np.float32)
            )
            for role in ROLE_NAMES:
                chunks[f"{role}_logits"].append(
                    outputs[LOGIT_KEYS[role]]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            chunks["y_graph"].append(
                batch["y_attack"].cpu().numpy().astype(np.uint8)
            )
            chunks["y_attacker_count"].append(
                batch["y_attacker_count"].cpu().numpy().astype(np.int64)
            )
            for role in ROLE_NAMES:
                chunks[f"y_{role}"].append(
                    batch[TARGET_KEYS[role]].cpu().numpy().astype(np.uint8)
                )
            chunks["role_mask"].append(
                batch["role_mask"].cpu().numpy().astype(np.uint8)
            )

            emitted += batch_size
            print(
                f"neural_inference_batch={batch_number + 1}/{len(loader)} "
                f"emitted={emitted}/{len(dataset)}",
                flush=True,
            )

    if emitted != len(dataset):
        raise RuntimeError("single-pass emitted count mismatch")
    if len(metadata) != len(dataset):
        raise RuntimeError("stable metadata count mismatch")
    stable_ids = [record["stable_item_id"] for record in metadata]
    if len(set(stable_ids)) != len(stable_ids):
        raise RuntimeError("test stable IDs are not unique")

    arrays = concatenate_chunks(chunks)
    n = len(dataset)
    expected_shapes = {
        "graph_logits": (n,),
        "count_logits": (n, 4),
        "source_logits": (n, 16),
        "transit_logits": (n, 16),
        "victim_logits": (n, 16),
        "path_logits": (n, 16),
        "y_graph": (n,),
        "y_attacker_count": (n,),
        "y_source": (n, 16),
        "y_transit": (n, 16),
        "y_victim": (n, 16),
        "y_path": (n, 16),
        "role_mask": (n, 16),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise RuntimeError(
                f"{name} shape={arrays[name].shape}, expected={expected}"
            )
        if not np.all(np.isfinite(arrays[name])):
            raise RuntimeError(f"{name} contains non-finite values")

    return arrays, metadata, {
        "item_count": n,
        "batch_count": len(loader),
        "pair_count": len(dataset.pair_rows),
        "run_count": 2 * len(dataset.pair_rows),
        "first_batch_exact_repeat_verified": first_batch_repeat_verified,
        "single_neural_inference_pass": True,
    }


def validate_labels(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    labels = {
        "graph": arrays["y_graph"].astype(np.uint8, copy=False),
        "count": arrays["y_attacker_count"].astype(np.int64, copy=False),
        **{
            role: arrays[f"y_{role}"].astype(np.uint8, copy=False)
            for role in ROLE_NAMES
        },
    }
    if not np.all((labels["graph"] == 0) | (labels["graph"] == 1)):
        raise RuntimeError("graph labels are nonbinary")
    active = labels["graph"] == 1
    control = ~active
    if not np.all(labels["count"][control] == 0):
        raise RuntimeError("control count labels are nonzero")
    if not np.all(
        np.isin(labels["count"][active], np.asarray([1, 2, 3, 4]))
    ):
        raise RuntimeError("active count labels are outside 1..4")
    for role in ROLE_NAMES:
        if not np.all((labels[role] == 0) | (labels[role] == 1)):
            raise RuntimeError(f"{role} labels are nonbinary")

    expected_role_mask = (
        labels["source"].astype(np.uint8)
        + 2 * labels["transit"].astype(np.uint8)
        + 4 * labels["victim"].astype(np.uint8)
    )
    mismatch_count = int(
        np.sum(arrays["role_mask"].astype(np.uint8) != expected_role_mask)
    )
    if mismatch_count != 0:
        raise RuntimeError(f"role-mask mismatch count={mismatch_count}")
    return labels


def produce_raw_and_a0(
    arrays: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    graph_probability = stable_sigmoid(arrays["graph_logits"])
    role_probability = {
        role: stable_sigmoid(arrays[f"{role}_logits"])
        for role in ROLE_NAMES
    }
    graph_prediction = (
        graph_probability >= A0_THRESHOLDS["graph"]
    ).astype(np.uint8)
    raw_count = (
        np.argmax(arrays["count_logits"], axis=1).astype(np.int64) + 1
    )
    raw_roles = {
        role: (
            role_probability[role] >= A0_THRESHOLDS[role]
        ).astype(np.uint8)
        for role in ROLE_NAMES
    }

    a0_count = np.where(graph_prediction == 1, raw_count, 0).astype(np.int64)
    a0_roles = {
        role: (
            raw_roles[role] * graph_prediction[:, None]
        ).astype(np.uint8)
        for role in ROLE_NAMES
    }

    raw = {
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
    a0 = {
        "graph_prediction": graph_prediction,
        "count_prediction": a0_count,
        **{
            f"{role}_prediction": a0_roles[role]
            for role in ROLE_NAMES
        },
    }
    metrics = {
        "raw_neural": structured_group_metrics(
            graph_prediction=graph_prediction,
            count_prediction=raw_count,
            role_predictions=raw_roles,
            graph_score=graph_probability,
            labels=labels,
            count_classes=(1, 2, 3, 4),
            raw_count_attack_only=True,
        ),
        "A0_lightweight": structured_group_metrics(
            graph_prediction=graph_prediction,
            count_prediction=a0_count,
            role_predictions=a0_roles,
            graph_score=graph_probability,
            labels=labels,
            count_classes=(0, 1, 2, 3, 4),
            raw_count_attack_only=False,
        ),
    }
    return raw, a0, metrics


def produce_a1(
    *,
    arrays: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    exact_module: Any,
    route_module: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n = arrays["graph_logits"].shape[0]
    margin = np.empty(n, dtype=np.float64)
    attack = np.empty(n, dtype=np.uint8)
    count = np.empty(n, dtype=np.int64)
    route_ids = np.full((n, 4), -1, dtype=np.int16)
    masks = {
        role: np.zeros(n, dtype=np.uint16)
        for role in ROLE_NAMES
    }
    solver = {
        "primary_mip_gap": np.empty(n, dtype=np.float64),
        "lexicographic_mip_gap": np.empty(n, dtype=np.float64),
        "primary_mip_node_count": np.empty(n, dtype=np.int64),
        "lexicographic_mip_node_count": np.empty(n, dtype=np.int64),
        "semantic_face_score_difference": np.empty(n, dtype=np.float64),
        "semantic_face_numerical_allowance": np.empty(n, dtype=np.float64),
        "high_precision_certified": np.empty(n, dtype=np.uint8),
        "rejected_candidates": np.empty(n, dtype=np.int64),
    }

    legal_items = 0
    for index in range(n):
        hypothesis = exact_module.decode_best_attack_hypothesis(
            graph_logit=float(arrays["graph_logits"][index]),
            count_logits=arrays["count_logits"][index].astype(np.float64),
            source_logits=arrays["source_logits"][index].astype(np.float64),
            transit_logits=arrays["transit_logits"][index].astype(np.float64),
            victim_logits=arrays["victim_logits"][index].astype(np.float64),
            path_logits=arrays["path_logits"][index].astype(np.float64),
        )
        if not (
            hypothesis.optimality_proven
            and hypothesis.lexicographic_tie_break_certified
            and hypothesis.high_precision_semantic_face_certified
        ):
            raise RuntimeError(f"A1 item {index} is not exactly certified")

        decoded = exact_module.apply_margin_threshold(
            hypothesis,
            A1_MARGIN_THRESHOLD,
        )
        margin[index] = hypothesis.margin
        attack[index] = decoded.attack
        count[index] = decoded.attacker_count
        if decoded.attack == 1:
            if len(decoded.route_ids) != decoded.attacker_count:
                raise RuntimeError(f"A1 route/count mismatch at item {index}")
            combined = route_module.combine_route_ids(
                decoded.route_ids,
                expected_k=decoded.attacker_count,
            )
            if (
                combined.source_mask != decoded.source_mask
                or combined.transit_mask != decoded.transit_mask
                or combined.victim_mask != decoded.victim_mask
                or combined.path_mask != decoded.path_mask
            ):
                raise RuntimeError(f"A1 mask legality mismatch at item {index}")
            route_ids[index, : len(decoded.route_ids)] = np.asarray(
                decoded.route_ids,
                dtype=np.int16,
            )
            masks["source"][index] = decoded.source_mask
            masks["transit"][index] = decoded.transit_mask
            masks["victim"][index] = decoded.victim_mask
            masks["path"][index] = decoded.path_mask
            legal_items += 1
        else:
            if (
                decoded.attacker_count != 0
                or decoded.route_ids
                or decoded.source_mask
                or decoded.transit_mask
                or decoded.victim_mask
                or decoded.path_mask
            ):
                raise RuntimeError(f"A1 normal output is nonempty at item {index}")

        solver["primary_mip_gap"][index] = hypothesis.primary_mip_gap
        solver["lexicographic_mip_gap"][index] = (
            hypothesis.lexicographic_mip_gap
        )
        solver["primary_mip_node_count"][index] = (
            hypothesis.primary_mip_node_count
        )
        solver["lexicographic_mip_node_count"][index] = (
            hypothesis.lexicographic_mip_node_count
        )
        solver["semantic_face_score_difference"][index] = (
            hypothesis.semantic_face_score_difference
        )
        solver["semantic_face_numerical_allowance"][index] = (
            hypothesis.semantic_face_numerical_allowance
        )
        solver["high_precision_certified"][index] = int(
            hypothesis.high_precision_semantic_face_certified
        )
        solver["rejected_candidates"][index] = (
            hypothesis.lexicographic_candidates_rejected_by_high_precision
        )

        if (index + 1) % 100 == 0 or index + 1 == n:
            print(f"A1_exact_decode={index + 1}/{n}", flush=True)

    role_binary = {
        role: masks_to_binary(masks[role])
        for role in ROLE_NAMES
    }
    metrics = structured_group_metrics(
        graph_prediction=attack,
        count_prediction=count,
        role_predictions=role_binary,
        graph_score=margin,
        labels=labels,
        count_classes=(0, 1, 2, 3, 4),
        raw_count_attack_only=False,
    )
    metrics["margin_auroc"] = binary_auroc(margin, labels["graph"])
    metrics["margin_average_precision"] = average_precision(
        margin,
        labels["graph"],
    )
    metrics["route_legality_rate"] = (
        1.0 if int(np.sum(attack)) == 0 else legal_items / int(np.sum(attack))
    )
    metrics["solver_certification_summary"] = {
        "item_count": n,
        "all_optimality_proven": True,
        "all_lexicographic_tie_break_certified": True,
        "all_high_precision_semantic_face_certified": bool(
            np.all(solver["high_precision_certified"] == 1)
        ),
        "max_primary_mip_gap": float(
            np.max(np.abs(solver["primary_mip_gap"]))
        ),
        "max_lexicographic_mip_gap": float(
            np.max(np.abs(solver["lexicographic_mip_gap"]))
        ),
        "max_semantic_face_score_difference": float(
            np.max(np.abs(solver["semantic_face_score_difference"]))
        ),
        "total_high_precision_rejected_candidates": int(
            np.sum(solver["rejected_candidates"])
        ),
    }

    outputs = {
        "margin": margin,
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
        **{
            f"solver_{name}": value
            for name, value in solver.items()
        },
    }
    return outputs, metrics


def write_prediction_archive(
    *,
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    metadata: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    raw: dict[str, np.ndarray],
    a0: dict[str, np.ndarray],
    a1: dict[str, np.ndarray],
    source_hashes: dict[str, str],
) -> tuple[Path, dict[str, Any]]:
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".prediction_cache_tmp_",
            dir=output_dir,
        )
    )
    final = output_dir / "prediction_cache"
    try:
        records: dict[str, Any] = {}
        groups = {
            "neural_and_labels": arrays,
            "raw_neural": raw,
            "A0_lightweight": a0,
            "A1_exact_structured": a1,
        }
        for group_name, group_arrays in groups.items():
            group_dir = temporary / group_name
            group_dir.mkdir()
            for name, value in sorted(group_arrays.items()):
                path = group_dir / f"{name}.npy"
                record = save_array(path, value)
                record["relative_path"] = str(path.relative_to(temporary))
                records[f"{group_name}/{name}.npy"] = record

        metadata_path = temporary / "stable_items.jsonl"
        with metadata_path.open("w", encoding="utf-8") as handle:
            for record in metadata:
                handle.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        records["stable_items.jsonl"] = {
            "relative_path": "stable_items.jsonl",
            "sha256": sha256_file(metadata_path),
            "records": len(metadata),
            "bytes": metadata_path.stat().st_size,
        }

        pair_path = temporary / "test_pair_window_manifest.csv"
        fields = list(pair_rows[0])
        with pair_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(pair_rows)
        records["test_pair_window_manifest.csv"] = {
            "relative_path": "test_pair_window_manifest.csv",
            "sha256": sha256_file(pair_path),
            "records": len(pair_rows),
            "bytes": pair_path.stat().st_size,
        }

        manifest = {
            "artifact": "V5_P2_FINAL_ONE_SHOT_PREDICTION_CACHE",
            "status": "COMPLETE",
            "official_output_groups": [
                "raw_neural",
                "A0_lightweight",
                "A1_exact_structured",
            ],
            "raw_A0_A1_kept_separate": True,
            "A2_executed": False,
            "single_neural_inference_pass": True,
            "item_count": len(metadata),
            "pair_count": len(pair_rows),
            "artifacts": records,
            "source_hashes": source_hashes,
        }
        manifest_path = temporary / "ARCHIVE_MANIFEST.json"
        write_json_atomic(manifest_path, manifest)
        for path in temporary.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        os.replace(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    canonical_manifest = final / "ARCHIVE_MANIFEST.json"
    return final, {
        "path": str(canonical_manifest),
        "sha256": sha256_file(canonical_manifest),
        "directory": str(final),
    }


def authorized_evaluation(
    *,
    repo_root: Path,
    data_root_text: str,
    output_dir_text: str,
    authorization_contract: Path,
    authorization_token: Path,
) -> int:
    repo_root = repo_root.expanduser().resolve()
    paths = verify_frozen_repository(repo_root)
    evaluator_path = Path(__file__).resolve()
    evaluator_sha = sha256_file(evaluator_path)

    # Preserve the exact user-authorized path strings. No test path is formed,
    # checked, or enumerated before token consumption.
    data_root_expanded = os.path.abspath(
        os.path.expanduser(data_root_text)
    )
    output_dir_expanded = os.path.abspath(
        os.path.expanduser(output_dir_text)
    )

    contract, token = validate_authorization(
        contract_path=authorization_contract.expanduser().resolve(),
        token_path=authorization_token.expanduser().resolve(),
        evaluator_sha256=evaluator_sha,
        data_root_text=data_root_expanded,
        output_dir_text=output_dir_expanded,
    )

    output_dir = Path(output_dir_expanded)
    if output_dir.exists():
        raise RuntimeError(
            f"one-shot output guard already exists: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(mode=0o755)

    try:
        consumed_token = consume_authorization_token(
            authorization_token.expanduser().resolve()
        )
    except Exception:
        # Authorization was not consumed, so it is safe to remove the empty
        # output guard and correct the pre-access issue.
        output_dir.rmdir()
        raise

    try:
        access_record = {
            "stage": STAGE,
            "event": "TEST_ACCESS_STARTED",
            "authorization_id": token["authorization_id"],
            "authorization_contract_sha256": sha256_file(
                authorization_contract.expanduser().resolve()
            ),
            "consumed_token_path": str(consumed_token),
            "consumed_token_sha256": sha256_file(consumed_token),
            "evaluator_source_sha256": evaluator_sha,
            "test_access_started": True,
            "test_evaluation_count": 1,
            "successful_rerun_authorized": False,
            "recovery_after_pre_inference_abort": True,
            "original_authorization_id": "V5P2-caaea91e4eb6772ace0dc1b43b7e28c4",
        }
        write_json_atomic(output_dir / ACCESS_STARTED, access_record)

        # First test-path operation occurs here, after token consumption.
        data_root = Path(data_root_expanded)
        mask = load_physical_mask(paths["physical_mask"])
        dataset = FrozenP2RecoveryTestDataset(
            data_root=data_root,
            physical_port_mask=mask,
        )

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.use_deterministic_algorithms(True)
        device = torch.device("cpu")
        model, model_summary = build_frozen_model(
            paths=paths,
            device=device,
        )

        arrays, metadata, inference_summary = perform_single_inference_pass(
            dataset=dataset,
            model=model,
            device=device,
        )
        labels = validate_labels(arrays)
        raw, a0, metrics = produce_raw_and_a0(arrays, labels)

        route_module = import_source(
            paths["route_library"],
            "src.decoders.v5_xy_route_library",
        )
        exact_module = import_source(
            paths["exact_decoder"],
            "src.decoders.v5_legal_xy_exact_decoder",
        )
        a1, a1_metrics = produce_a1(
            arrays=arrays,
            labels=labels,
            exact_module=exact_module,
            route_module=route_module,
        )
        metrics["A1_exact_structured"] = a1_metrics

        source_hashes = {
            name: sha256_file(path)
            for name, path in paths.items()
        }
        source_hashes["evaluator_source"] = evaluator_sha
        archive_dir, archive_record = write_prediction_archive(
            output_dir=output_dir,
            arrays=arrays,
            metadata=metadata,
            pair_rows=dataset.pair_rows,
            raw=raw,
            a0=a0,
            a1=a1,
            source_hashes=source_hashes,
        )

        report = {
            "stage": STAGE,
            "status": "COMPLETE",
            "decision": "V5_P2_RECOVERY_RAW_A0_A1_BLIND_TEST_RESULT_RECORDED",
            "authorization": {
                "authorization_id": token["authorization_id"],
                "authorization_contract_path": str(
                    authorization_contract.expanduser().resolve()
                ),
                "authorization_contract_sha256": sha256_file(
                    authorization_contract.expanduser().resolve()
                ),
                "consumed_token_path": str(consumed_token),
                "consumed_token_sha256": sha256_file(consumed_token),
                "authorization_consumed": True,
            },
            "frozen_system": {
                "architecture": (
                    "Causal Depthwise-Separable Conv1D Temporal Encoder "
                    "+ Two-Layer GraphConv"
                ),
                "model_class": "P2TaskDGraphConvCount4",
                "selected_seed": SELECTED_SEED,
                "selected_epoch": SELECTED_EPOCH,
                "checkpoint_sha256": EXPECTED_HASHES["checkpoint"],
                "A0_thresholds": A0_THRESHOLDS,
                "A1_margin_threshold": A1_MARGIN_THRESHOLD,
                "A2_official": False,
            },
            "dataset": {
                "item_count": inference_summary["item_count"],
                "batch_count": inference_summary["batch_count"],
                "pair_count": inference_summary["pair_count"],
                "run_count": inference_summary["run_count"],
                "non_test_manifest_sha256": EXPECTED_HASHES["manifest"],
                "pair_index_source": "preexisting_B6_filename_pairing_rule",
                "expected_test_tensor_files": EXPECTED_TEST_FILES,
                "expected_complete_pairs": EXPECTED_TEST_PAIRS,
                "stable_item_ids_unique": True,
            },
            "inference": {
                **inference_summary,
                "device": "cpu",
                "dtype": "float32",
                "amp": False,
                "deterministic_algorithms": True,
                "torch_num_threads": 1,
                "model_summary": model_summary,
            },
            "official_metrics": metrics,
            "prediction_cache": archive_record,
            "reporting": {
                "raw_A0_A1_kept_separate": True,
                "A1_never_overwrites_raw_or_A0": True,
                "A2_executed": False,
                "threshold_search_performed": False,
                "model_or_checkpoint_selection_performed": False,
                "test_driven_changes_allowed": False,
                "recovery_after_pre_inference_abort": True,
                "original_authorization_id": "V5P2-caaea91e4eb6772ace0dc1b43b7e28c4",
                "original_execution_completed": False,
                "original_test_result_obtained": False,
                "recovery_reported_separately": True,
            },
            "environment": {
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "torch_version": torch.__version__,
                "solver_identity": dataclasses.asdict(
                    exact_module.solver_identity()
                ),
            },
            "source_hashes": source_hashes,
            "security_boundary": {
                "authorization_consumed": True,
                "training_performed": False,
                "model_weights_changed": False,
                "checkpoint_selection_performed": False,
                "threshold_tuning_performed": False,
                "decoder_selection_performed": False,
                "test_directory_existence_checked": True,
                "test_directory_enumerated": True,
                "test_tensor_files_opened": True,
                "test_tensors_deserialized": True,
                "test_evaluation_performed": True,
                "test_evaluation_count": 1,
                "successful_rerun_authorized": False,
                "primary_report_written_before_error_analysis": True,
                "test_error_analysis_performed": False,
            },
            "next_stage": (
                "V5_P2_FINAL_RESULTS_ARCHIVE_AND_ERROR_ANALYSIS_FROM_"
                "FROZEN_PREDICTION_CACHE"
            ),
        }

        report_path = output_dir / f"{STAGE}.json"
        write_json_atomic(report_path, report)
        lock = {
            "stage": f"{STAGE}_LOCK",
            "status": "LOCKED",
            "decision": report["decision"],
            "report_sha256": sha256_file(report_path),
            "prediction_cache_manifest_sha256": archive_record["sha256"],
            "authorization_id": token["authorization_id"],
            "authorization_consumed": True,
            "test_evaluation_count": 1,
            "successful_rerun_authorized": False,
            "evaluator_source_sha256": evaluator_sha,
            "checkpoint_sha256": EXPECTED_HASHES["checkpoint"],
            "raw_A0_A1_kept_separate": True,
            "A2_executed": False,
            "next_stage": report["next_stage"],
        }
        lock_path = output_dir / f"{STAGE}_LOCK.json"
        write_json_atomic(lock_path, lock)
        atomic_write_text(output_dir / COMPLETE, COMPLETE + "\n")

        sums_path = output_dir / f"{STAGE}_SHA256SUMS.txt"
        with sums_path.open("w", encoding="utf-8") as handle:
            for path in (
                evaluator_path,
                authorization_contract.expanduser().resolve(),
                consumed_token,
                report_path,
                lock_path,
                output_dir / COMPLETE,
                archive_dir / "ARCHIVE_MANIFEST.json",
            ):
                handle.write(f"{sha256_file(path)}  {path}\n")

        for path in (
            output_dir / ACCESS_STARTED,
            report_path,
            lock_path,
            output_dir / COMPLETE,
            sums_path,
        ):
            path.chmod(0o444)

        print(COMPLETE)
        print("status=COMPLETE")
        print("decision=FINAL_P2_RAW_A0_A1_BLIND_TEST_RESULT_RECORDED")
        print(f"test_item_count={inference_summary['item_count']}")
        print(f"test_pair_count={inference_summary['pair_count']}")
        print(f"test_batch_count={inference_summary['batch_count']}")
        print(
            "raw_graph_balanced_accuracy="
            f"{metrics['raw_neural']['graph']['balanced_accuracy']:.12g}"
        )
        print(
            "A0_strict_all_task_exactness="
            f"{metrics['A0_lightweight']['strict_all_task_exactness']:.12g}"
        )
        print(
            "A1_strict_all_task_exactness="
            f"{metrics['A1_exact_structured']['strict_all_task_exactness']:.12g}"
        )
        print(
            "A1_margin_auroc="
            f"{metrics['A1_exact_structured']['margin_auroc']:.12g}"
        )
        print("authorization_consumed=true")
        print("test_evaluation_count=1")
        print("successful_rerun_authorized=false")
        print(f"prediction_cache={archive_dir}")
        print(f"report_json={report_path}")
        return 0

    except Exception as exc:
        failure = {
            "stage": STAGE,
            "status": "IRREVERSIBLE_FAILURE_AFTER_AUTHORIZATION_CONSUMPTION",
            "authorization_id": token.get("authorization_id"),
            "authorization_consumed": True,
            "test_access_started": True,
            "test_evaluation_may_be_partial": True,
            "successful_rerun_authorized": False,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(
            output_dir / f"{IRREVERSIBLE_FAILURE}.json",
            failure,
        )
        atomic_write_text(
            output_dir / IRREVERSIBLE_FAILURE,
            IRREVERSIBLE_FAILURE + "\n",
        )
        print(IRREVERSIBLE_FAILURE)
        print(f"FAIL: {type(exc).__name__}: {exc}")
        print(
            "Recovery authorization was consumed. Do not rerun or delete guard artifacts."
        )
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-only-self-test", action="store_true")
    parser.add_argument("--data-root")
    parser.add_argument("--authorization-contract", type=Path)
    parser.add_argument("--authorization-token", type=Path)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    if args.source_only_self_test:
        if any(
            value is not None
            for value in (
                args.data_root,
                args.authorization_contract,
                args.authorization_token,
                args.output_dir,
            )
        ):
            raise RuntimeError(
                "source-only self-test forbids data/auth/output arguments"
            )
        result = run_source_only_self_test(
            args.repo_root.expanduser().resolve()
        )
        print("V5_P2_RECOVERY_EVALUATOR_SOURCE_ONLY_SELF_TEST_PASS")
        for key, value in result.items():
            if isinstance(value, (str, int, float, bool)):
                print(f"{key}={str(value).lower() if isinstance(value, bool) else value}")
        return 0

    missing = [
        name
        for name, value in (
            ("--data-root", args.data_root),
            ("--authorization-contract", args.authorization_contract),
            ("--authorization-token", args.authorization_token),
            ("--output-dir", args.output_dir),
        )
        if value is None
    ]
    if missing:
        parser.error("missing authorized-run arguments: " + ", ".join(missing))

    return authorized_evaluation(
        repo_root=args.repo_root,
        data_root_text=str(args.data_root),
        output_dir_text=str(args.output_dir),
        authorization_contract=args.authorization_contract,
        authorization_token=args.authorization_token,
    )


if __name__ == "__main__":
    raise SystemExit(main())
