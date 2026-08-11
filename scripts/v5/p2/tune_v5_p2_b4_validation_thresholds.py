#!/usr/bin/env python3
"""
V5 P2-B4 Validation Threshold Tuning

Loads the B3-selected checkpoint and performs validation-only inference over
the audited pair-aligned PRIMARY58 loader. It freezes thresholds for:

    attack, source, transit, victim, path

The four-class attacker-count head continues to use argmax and has no tunable
threshold.

Threshold policy
----------------
Attack:
    maximize validation balanced accuracy
    tie-break by F1, accuracy, lower FPR, proximity to 0.5, lower threshold

Each role:
    maximize 0.60 * node F1
           + 0.40 * exact-set accuracy on true attack windows
    tie-break by active exact-set accuracy, node F1, recall, precision,
    lower FPR, proximity to 0.5, lower threshold

Search:
    coarse thresholds 0.01..0.99 with step 0.01
    fine thresholds +/-0.02 around the coarse winner with step 0.001

No model weights are changed. No training occurs. P2 test is neither
enumerated nor accessed, and test evaluation remains unauthorized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


STAGE = "V5_P2_B4_VALIDATION_THRESHOLD_TUNING"
COMPLETE = f"{STAGE}_COMPLETE"

EXPECTED_PROTOCOL_SHA = (
    "0817ae7812f91e3c75260589acaf52bd8a74b07b2114a451b4773578efde4b60"
)
EXPECTED_SELECTED_SEED = 127
EXPECTED_SELECTED_EPOCH = 59
EXPECTED_CHECKPOINT_SHA = (
    "7d4afae2214f67ecd9c65c6ff4614ba8238614234d7b07cdf1408b6d3efb07ef"
)
EXPECTED_PARAMETER_COUNT = 43_273
EXPECTED_VALIDATION_ITEMS = 12_528
COUNT_CLASS_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3}


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
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
    atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def import_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_deterministic_inference() -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class ValidationPairBlockBatchSampler(Sampler[list[int]]):
    """
    Deterministic validation sampler preserving ATTACK/CONTROL adjacency.

    It groups all starts of a run pair together for efficient audited-loader
    access and emits 128 aligned pair-window blocks (256 items) per batch.
    """

    def __init__(
        self,
        dataset,
        *,
        block_batch_size: int = 128,
    ) -> None:
        self.dataset = dataset
        self.block_batch_size = int(block_batch_size)

        groups: OrderedDict[str, list[int]] = OrderedDict()
        if len(dataset._index) % 2 != 0:
            raise ValueError("validation item count must be even")

        for base in range(0, len(dataset._index), 2):
            attack = dataset._index[base]
            control = dataset._index[base + 1]

            if (
                attack.mode != "attack"
                or control.mode != "control"
                or attack.pair_key != control.pair_key
                or attack.start != control.start
                or attack.target != control.target
            ):
                raise ValueError(
                    f"pair-block contract failed at base={base}"
                )

            groups.setdefault(
                attack.pair_key,
                [],
            ).append(base)

        self.blocks = [
            base
            for pair_key in groups
            for base in groups[pair_key]
        ]

        if len(self.blocks) * 2 != len(dataset):
            raise RuntimeError("pair blocks do not cover validation data")

    def __len__(self) -> int:
        return math.ceil(
            len(self.blocks) / self.block_batch_size
        )

    def __iter__(self) -> Iterator[list[int]]:
        for start in range(
            0,
            len(self.blocks),
            self.block_batch_size,
        ):
            item_indices: list[int] = []
            for base in self.blocks[
                start:start + self.block_batch_size
            ]:
                item_indices.extend((base, base + 1))
            yield item_indices


def binary_auroc(
    truth: np.ndarray,
    score: np.ndarray,
) -> float:
    y = truth.astype(np.int64).reshape(-1)
    s = score.astype(np.float64).reshape(-1)

    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC requires both binary classes")

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
) -> float:
    y = truth.astype(np.int64).reshape(-1)
    s = score.astype(np.float64).reshape(-1)

    positive_count = int((y == 1).sum())
    if positive_count == 0:
        raise ValueError("average precision requires positives")

    order = np.argsort(-s, kind="mergesort")
    sorted_truth = y[order]
    cumulative_positive = np.cumsum(sorted_truth)
    positions = np.arange(1, len(y) + 1)
    precision_at_rank = cumulative_positive / positions

    return float(
        (precision_at_rank * sorted_truth).sum()
        / positive_count
    )


def binary_threshold_metrics(
    truth: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y = truth.astype(np.int64).reshape(-1)
    prediction = (
        score.astype(np.float64).reshape(-1)
        >= threshold
    ).astype(np.int64)

    tn = int(((y == 0) & (prediction == 0)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    tp = int(((y == 1) & (prediction == 1)).sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = (
        2 * precision * recall
        / max(1e-12, precision + recall)
    )

    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(1, len(y)),
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / max(1, fp + tn),
        "specificity": specificity,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def role_threshold_metrics(
    truth: np.ndarray,
    score: np.ndarray,
    graph_truth: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y = truth.astype(np.int64)
    prediction = (
        score.astype(np.float64) >= threshold
    ).astype(np.int64)

    flat = binary_threshold_metrics(
        y.reshape(-1),
        score.reshape(-1),
        threshold,
    )

    equality = prediction == y
    exact_by_item = equality.all(axis=1)
    active = graph_truth.astype(bool)
    control = ~active

    active_exact = float(
        exact_by_item[active].mean()
    )
    control_exact = float(
        exact_by_item[control].mean()
    )
    all_exact = float(
        exact_by_item.mean()
    )

    objective = (
        0.60 * float(flat["f1"])
        + 0.40 * active_exact
    )

    return {
        **flat,
        "node_f1": float(flat["f1"]),
        "active_exact_set_accuracy": active_exact,
        "control_exact_zero_accuracy": control_exact,
        "all_item_exact_set_accuracy": all_exact,
        "selection_objective": objective,
    }


def coarse_thresholds() -> list[float]:
    return [
        round(value, 3)
        for value in np.arange(0.01, 1.00, 0.01)
    ]


def fine_thresholds(center: float) -> list[float]:
    lower = max(0.001, center - 0.020)
    upper = min(0.999, center + 0.020)
    values = np.arange(lower, upper + 0.0005, 0.001)
    return sorted(
        {
            round(float(value), 3)
            for value in values
        }
        | {round(float(center), 3), 0.5}
    )


def graph_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["balanced_accuracy"]),
        float(row["f1"]),
        float(row["accuracy"]),
        -float(row["fpr"]),
        -abs(float(row["threshold"]) - 0.5),
        -float(row["threshold"]),
    )


def role_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["selection_objective"]),
        float(row["active_exact_set_accuracy"]),
        float(row["node_f1"]),
        float(row["recall"]),
        float(row["precision"]),
        -float(row["fpr"]),
        -abs(float(row["threshold"]) - 0.5),
        -float(row["threshold"]),
    )


def sweep_graph(
    truth: np.ndarray,
    score: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coarse_rows = [
        binary_threshold_metrics(
            truth,
            score,
            threshold,
        )
        for threshold in coarse_thresholds()
    ]
    coarse_best = max(coarse_rows, key=graph_rank)

    fine_rows = [
        binary_threshold_metrics(
            truth,
            score,
            threshold,
        )
        for threshold in fine_thresholds(
            float(coarse_best["threshold"])
        )
    ]

    rows_by_threshold = {
        float(row["threshold"]): row
        for row in coarse_rows + fine_rows
    }
    rows = [
        rows_by_threshold[key]
        for key in sorted(rows_by_threshold)
    ]
    return max(rows, key=graph_rank), rows


def sweep_role(
    truth: np.ndarray,
    score: np.ndarray,
    graph_truth: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coarse_rows = [
        role_threshold_metrics(
            truth,
            score,
            graph_truth,
            threshold,
        )
        for threshold in coarse_thresholds()
    ]
    coarse_best = max(coarse_rows, key=role_rank)

    fine_rows = [
        role_threshold_metrics(
            truth,
            score,
            graph_truth,
            threshold,
        )
        for threshold in fine_thresholds(
            float(coarse_best["threshold"])
        )
    ]

    rows_by_threshold = {
        float(row["threshold"]): row
        for row in coarse_rows + fine_rows
    }
    rows = [
        rows_by_threshold[key]
        for key in sorted(rows_by_threshold)
    ]
    return max(rows, key=role_rank), rows


def multiclass_macro_f1(
    truth: np.ndarray,
    prediction: np.ndarray,
    class_count: int,
) -> float:
    y = truth.astype(np.int64).reshape(-1)
    p = prediction.astype(np.int64).reshape(-1)
    values = []

    for label in range(class_count):
        tp = int(((y == label) & (p == label)).sum())
        fp = int(((y != label) & (p == label)).sum())
        fn = int(((y == label) & (p != label)).sum())

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (
            2 * precision * recall
            / max(1e-12, precision + recall)
        )
        values.append(f1)

    return float(np.mean(values))


def end_to_end_metrics(
    *,
    graph_truth: np.ndarray,
    graph_score: np.ndarray,
    count_truth_raw: np.ndarray,
    count_prediction_class: np.ndarray,
    role_truth: dict[str, np.ndarray],
    role_score: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    graph_prediction = (
        graph_score >= thresholds["attack"]
    ).astype(np.int64)

    role_predictions = {}
    role_exact = {}
    for role in role_truth:
        prediction = (
            role_score[role] >= thresholds[role]
        ).astype(np.int64)

        # Deployment gating: a graph-normal prediction emits no roles.
        prediction[
            graph_prediction == 0,
            :
        ] = 0

        role_predictions[role] = prediction
        role_exact[role] = (
            prediction == role_truth[role]
        ).all(axis=1)

    active = graph_truth == 1
    count_truth_class = np.array(
        [
            COUNT_CLASS_MAPPING[int(value)]
            for value in count_truth_raw[active]
        ],
        dtype=np.int64,
    )
    count_exact_active = (
        count_prediction_class[active]
        == count_truth_class
    )

    all_task_exact = np.zeros(
        len(graph_truth),
        dtype=bool,
    )

    control = ~active
    all_task_exact[control] = (
        graph_prediction[control] == 0
    )
    for role in role_truth:
        all_task_exact[control] &= (
            role_predictions[role][control] == 0
        ).all(axis=1)

    active_indices = np.flatnonzero(active)
    active_exact = (
        graph_prediction[active] == 1
    ) & count_exact_active
    for role in role_truth:
        active_exact &= role_exact[role][active]
    all_task_exact[active_indices] = active_exact

    return {
        "graph_fixed_threshold": binary_threshold_metrics(
            graph_truth,
            graph_score,
            thresholds["attack"],
        ),
        "count_active_macro_f1": multiclass_macro_f1(
            count_truth_class,
            count_prediction_class[active],
            class_count=4,
        ),
        "count_active_accuracy": float(
            count_exact_active.mean()
        ),
        "role_graph_gated_exact_set_accuracy": {
            role: {
                "all_items": float(
                    role_exact[role].mean()
                ),
                "true_attack_items": float(
                    role_exact[role][active].mean()
                ),
                "true_control_items": float(
                    role_exact[role][control].mean()
                ),
            }
            for role in role_truth
        },
        "all_task_exact_accuracy": float(
            all_task_exact.mean()
        ),
        "all_task_exact_attack_accuracy": float(
            all_task_exact[active].mean()
        ),
        "all_task_exact_control_accuracy": float(
            all_task_exact[control].mean()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a1-r2-dir", type=Path, required=True)
    parser.add_argument("--b1-dir", type=Path, required=True)
    parser.add_argument("--b3-dir", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--model-source-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    a1_r2_dir = args.a1_r2_dir.expanduser().resolve()
    b1_dir = args.b1_dir.expanduser().resolve()
    b3_dir = args.b3_dir.expanduser().resolve()
    loader_path = args.loader_path.expanduser().resolve()
    model_source_path = args.model_source_path.expanduser().resolve()
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

    paths = {
        "pair_manifest": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
        ),
        "a1_r2_lock": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT_LOCK.json"
        ),
        "b1_report": (
            b1_dir
            / "V5_P2_B1_TRAINING_PROTOCOL_LOCK.json"
        ),
        "b1_lock": (
            b1_dir
            / "V5_P2_B1_TRAINING_PROTOCOL_LOCK_LOCK.json"
        ),
        "b1_protocol": (
            b1_dir
            / "V5_P2_B1_TRAINING_PROTOCOL.json"
        ),
        "b3_report": (
            b3_dir
            / "V5_P2_B3_SEED_AND_CHECKPOINT_SELECTION.json"
        ),
        "b3_lock": (
            b3_dir
            / "V5_P2_B3_SEED_AND_CHECKPOINT_SELECTION_LOCK.json"
        ),
        "b3_manifest": (
            b3_dir
            / "V5_P2_B3_SELECTED_CHECKPOINT_MANIFEST.json"
        ),
        "loader": loader_path,
        "model_source": model_source_path,
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    if not root.is_dir():
        failures.append(f"dataset root missing: {root}")
    if not (root / "runs" / "validation").is_dir():
        failures.append("runs/validation is missing")

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "validation_tensor_contents_accessed": False,
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

    a1_lock = load_json(paths["a1_r2_lock"])
    b1_report = load_json(paths["b1_report"])
    b1_lock = load_json(paths["b1_lock"])
    b1_protocol = load_json(paths["b1_protocol"])
    b3_report = load_json(paths["b3_report"])
    b3_lock = load_json(paths["b3_lock"])
    b3_manifest = load_json(paths["b3_manifest"])

    if (
        a1_lock.get("pair_manifest_sha256")
        != sha256_file(paths["pair_manifest"])
    ):
        failures.append("A1-R2 pair-manifest SHA mismatch")
    if a1_lock.get("aligned_windows") != 82_694:
        failures.append("A1-R2 aligned-window count changed")

    if b1_report.get("status") != "COMPLETE":
        failures.append("B1 status is not COMPLETE")
    if (
        b1_lock.get("report_sha256")
        != sha256_file(paths["b1_report"])
    ):
        failures.append("B1 report SHA mismatch")
    if (
        b1_lock.get("protocol_file_sha256")
        != sha256_file(paths["b1_protocol"])
    ):
        failures.append("B1 protocol-file SHA mismatch")
    if (
        b1_protocol.get("protocol_sha256")
        != EXPECTED_PROTOCOL_SHA
    ):
        failures.append("B1 protocol SHA changed")
    if b1_lock.get("test_evaluation_authorized") is not False:
        failures.append("B1 unexpectedly authorizes test")

    if b3_report.get("status") != "COMPLETE":
        failures.append("B3 status is not COMPLETE")
    if (
        b3_lock.get("report_sha256")
        != sha256_file(paths["b3_report"])
    ):
        failures.append("B3 report SHA mismatch")
    if (
        b3_lock.get("selection_manifest_sha256")
        != sha256_file(paths["b3_manifest"])
    ):
        failures.append("B3 selection-manifest SHA mismatch")
    if b3_lock.get("selected_seed") != EXPECTED_SELECTED_SEED:
        failures.append("B3 selected seed changed")
    if (
        b3_lock.get("selected_best_epoch")
        != EXPECTED_SELECTED_EPOCH
    ):
        failures.append("B3 selected epoch changed")
    if (
        b3_lock.get("selected_checkpoint_sha256")
        != EXPECTED_CHECKPOINT_SHA
    ):
        failures.append("B3 selected-checkpoint SHA changed")
    if b3_lock.get("selection_uses_validation_only") is not True:
        failures.append("B3 selection was not validation-only")
    if b3_lock.get("threshold_tuning_performed") is not False:
        failures.append("B3 already performed threshold tuning")
    if b3_lock.get("test_evaluation_authorized") is not False:
        failures.append("B3 unexpectedly authorizes test")

    if (
        b3_lock.get("loader_sha256")
        != sha256_file(loader_path)
    ):
        failures.append("B3 loader SHA mismatch")
    if (
        b3_lock.get("model_source_sha256")
        != sha256_file(model_source_path)
    ):
        failures.append("B3 model-source SHA mismatch")

    selected_checkpoint = Path(
        b3_lock["selected_checkpoint_path"]
    ).resolve()
    if not selected_checkpoint.is_file():
        failures.append(
            f"selected checkpoint missing: {selected_checkpoint}"
        )
    elif (
        sha256_file(selected_checkpoint)
        != EXPECTED_CHECKPOINT_SHA
    ):
        failures.append("selected checkpoint file SHA mismatch")

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "validation_tensor_contents_accessed": False,
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

    set_deterministic_inference()

    loader_module = import_module(
        loader_path,
        "v5_p2_loader_b4_thresholds",
    )
    model_module = import_module(
        model_source_path,
        "v5_p2_model_b4_thresholds",
    )

    DatasetClass = (
        loader_module
        .V5P2PairAlignedPrimary58Dataset
    )
    ModelClass = (
        model_module
        .P2B3Conv1DOnlyCount4
    )

    validation_dataset = DatasetClass(
        root=root,
        split="validation",
        pair_manifest=paths["pair_manifest"],
    )
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        failures.append(
            f"validation length={len(validation_dataset)}, "
            f"expected {EXPECTED_VALIDATION_ITEMS}"
        )

    sampler = ValidationPairBlockBatchSampler(
        validation_dataset,
        block_batch_size=128,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = ModelClass().to(device)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        failures.append(
            f"parameter count={parameter_count}, "
            f"expected {EXPECTED_PARAMETER_COUNT}"
        )

    checkpoint = torch.load(
        selected_checkpoint,
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("seed") != EXPECTED_SELECTED_SEED:
        failures.append("checkpoint seed changed")
    if checkpoint.get("epoch") != EXPECTED_SELECTED_EPOCH:
        failures.append("checkpoint epoch changed")
    if (
        checkpoint.get("protocol_sha256")
        != EXPECTED_PROTOCOL_SHA
    ):
        failures.append("checkpoint protocol SHA changed")
    if (
        checkpoint.get("loader_sha256")
        != sha256_file(loader_path)
    ):
        failures.append("checkpoint loader SHA mismatch")
    if (
        checkpoint.get("model_source_sha256")
        != sha256_file(model_source_path)
    ):
        failures.append("checkpoint model-source SHA mismatch")

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    graph_truth_parts = []
    graph_score_parts = []
    count_truth_parts = []
    count_prediction_parts = []

    role_truth_parts = {
        "source": [],
        "transit": [],
        "victim": [],
        "path": [],
    }
    role_score_parts = {
        "source": [],
        "transit": [],
        "victim": [],
        "path": [],
    }

    item_count = 0

    print("===== V5 P2-B4 VALIDATION INFERENCE =====")
    print("selected_seed:", EXPECTED_SELECTED_SEED)
    print("selected_epoch:", EXPECTED_SELECTED_EPOCH)
    print("device:", device)
    print("validation_items:", len(validation_dataset))
    print("validation_batches:", len(validation_loader))
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")

    with torch.no_grad():
        for batch_index, batch in enumerate(
            validation_loader,
            start=1,
        ):
            batch = {
                key: value.to(
                    device,
                    non_blocking=(device.type == "cuda"),
                )
                for key, value in batch.items()
            }

            outputs = model(
                batch["x"],
                batch["physical_port_mask"],
            )
            batch_size = int(batch["x"].shape[0])
            item_count += batch_size

            graph_truth_parts.append(
                batch["y_attack"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            graph_score_parts.append(
                torch.sigmoid(
                    outputs["attack_logits"]
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            count_truth_parts.append(
                batch["y_attacker_count"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int8)
            )
            count_prediction_parts.append(
                outputs["count_logits"]
                .argmax(dim=-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int8)
            )

            for role, output_key, target_key in (
                ("source", "source_logits", "y_source"),
                ("transit", "transit_logits", "y_transit"),
                ("victim", "victim_logits", "y_victim"),
                ("path", "path_logits", "y_attack_path"),
            ):
                role_truth_parts[role].append(
                    batch[target_key]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                role_score_parts[role].append(
                    torch.sigmoid(
                        outputs[output_key]
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            if (
                batch_index % 10 == 0
                or batch_index == len(validation_loader)
            ):
                print(
                    f"validation inference "
                    f"{batch_index}/{len(validation_loader)} batches"
                )

    if item_count != EXPECTED_VALIDATION_ITEMS:
        failures.append(
            f"inference item count={item_count}, "
            f"expected {EXPECTED_VALIDATION_ITEMS}"
        )

    graph_truth = np.concatenate(graph_truth_parts)
    graph_score = np.concatenate(graph_score_parts)
    count_truth_raw = np.concatenate(count_truth_parts)
    count_prediction_class = np.concatenate(
        count_prediction_parts
    )

    role_truth = {
        role: np.concatenate(parts, axis=0)
        for role, parts in role_truth_parts.items()
    }
    role_score = {
        role: np.concatenate(parts, axis=0)
        for role, parts in role_score_parts.items()
    }

    graph_auroc = binary_auroc(
        graph_truth,
        graph_score,
    )
    graph_ap = average_precision(
        graph_truth,
        graph_score,
    )

    selected_candidate = next(
        row
        for row in b3_manifest["ranked_candidates"]
        if int(row["seed"]) == EXPECTED_SELECTED_SEED
    )

    if (
        abs(
            graph_auroc
            - float(selected_candidate["graph_auroc"])
        )
        > 1e-8
    ):
        failures.append(
            "recomputed graph AUROC differs from B3 candidate"
        )
    if (
        abs(
            graph_ap
            - float(
                selected_candidate[
                    "graph_average_precision"
                ]
            )
        )
        > 1e-8
    ):
        failures.append(
            "recomputed graph AP differs from B3 candidate"
        )

    active = graph_truth == 1
    count_truth_class = np.array(
        [
            COUNT_CLASS_MAPPING[int(value)]
            for value in count_truth_raw[active]
        ],
        dtype=np.int64,
    )
    count_macro_f1 = multiclass_macro_f1(
        count_truth_class,
        count_prediction_class[active],
        class_count=4,
    )
    count_accuracy = float(
        (
            count_truth_class
            == count_prediction_class[active]
        ).mean()
    )

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "validation_tensor_contents_accessed": True,
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

    print("===== V5 P2-B4 THRESHOLD SWEEPS =====")

    graph_best, graph_rows = sweep_graph(
        graph_truth,
        graph_score,
    )

    role_best: dict[str, dict[str, Any]] = {}
    role_rows: dict[str, list[dict[str, Any]]] = {}
    for role in ("source", "transit", "victim", "path"):
        best, rows = sweep_role(
            role_truth[role],
            role_score[role],
            graph_truth,
        )
        role_best[role] = best
        role_rows[role] = rows
        print(
            f"{role}: threshold={best['threshold']:.3f} "
            f"objective={best['selection_objective']:.6f} "
            f"node_f1={best['node_f1']:.6f} "
            f"active_exact={best['active_exact_set_accuracy']:.6f}"
        )

    thresholds = {
        "attack": float(graph_best["threshold"]),
        "source": float(role_best["source"]["threshold"]),
        "transit": float(role_best["transit"]["threshold"]),
        "victim": float(role_best["victim"]["threshold"]),
        "path": float(role_best["path"]["threshold"]),
    }

    for name, threshold in thresholds.items():
        if threshold <= 0.002 or threshold >= 0.998:
            warnings.append(
                f"{name} threshold is near the search boundary: "
                f"{threshold}"
            )

    end_to_end = end_to_end_metrics(
        graph_truth=graph_truth,
        graph_score=graph_score,
        count_truth_raw=count_truth_raw,
        count_prediction_class=count_prediction_class,
        role_truth=role_truth,
        role_score=role_score,
        thresholds=thresholds,
    )

    cache_path = (
        output_dir
        / "V5_P2_B4_VALIDATION_PREDICTION_CACHE.npz"
    )
    np.savez_compressed(
        cache_path,
        graph_truth=graph_truth,
        graph_score=graph_score,
        count_truth_raw=count_truth_raw,
        count_prediction_class=count_prediction_class,
        source_truth=role_truth["source"],
        source_score=role_score["source"],
        transit_truth=role_truth["transit"],
        transit_score=role_score["transit"],
        victim_truth=role_truth["victim"],
        victim_score=role_score["victim"],
        path_truth=role_truth["path"],
        path_score=role_score["path"],
    )

    graph_sweep_path = (
        output_dir
        / "V5_P2_B4_ATTACK_THRESHOLD_SWEEP.csv"
    )
    write_csv(graph_sweep_path, graph_rows)

    role_sweep_paths = {}
    for role, rows in role_rows.items():
        path = (
            output_dir
            / f"V5_P2_B4_{role.upper()}_THRESHOLD_SWEEP.csv"
        )
        write_csv(path, rows)
        role_sweep_paths[role] = path

    threshold_manifest = {
        "contract_name": (
            "V5_P2_B4_VALIDATION_ONLY_THRESHOLD_MANIFEST"
        ),
        "contract_version": 1,
        "selected_seed": EXPECTED_SELECTED_SEED,
        "selected_epoch": EXPECTED_SELECTED_EPOCH,
        "selected_checkpoint_path": str(
            selected_checkpoint
        ),
        "selected_checkpoint_sha256": (
            sha256_file(selected_checkpoint)
        ),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "model_source_sha256": sha256_file(
            model_source_path
        ),
        "loader_sha256": sha256_file(loader_path),
        "thresholds": thresholds,
        "count_decision": {
            "method": "argmax",
            "class_mapping": {
                "0": 1,
                "1": 2,
                "2": 3,
                "3": 4,
            },
            "threshold": None,
        },
        "selection_policy": {
            "search": (
                "coarse 0.01..0.99 step 0.01; "
                "fine +/-0.02 step 0.001"
            ),
            "attack_objective": (
                "maximize balanced accuracy"
            ),
            "attack_tie_breakers": [
                "higher F1",
                "higher accuracy",
                "lower FPR",
                "threshold closer to 0.5",
                "lower threshold",
            ],
            "role_objective": (
                "0.60*node_F1 + "
                "0.40*exact_set_accuracy_on_true_attack_windows"
            ),
            "role_tie_breakers": [
                "higher active exact-set accuracy",
                "higher node F1",
                "higher recall",
                "higher precision",
                "lower FPR",
                "threshold closer to 0.5",
                "lower threshold",
            ],
            "role_selection_uses_true_graph_labels": True,
            "deployment_role_outputs_graph_gated": True,
        },
        "threshold_free_validation": {
            "graph_auroc": graph_auroc,
            "graph_average_precision": graph_ap,
            "count_active_macro_f1": count_macro_f1,
            "count_active_accuracy": count_accuracy,
            "role_average_precision": {
                role: average_precision(
                    role_truth[role].reshape(-1),
                    role_score[role].reshape(-1),
                )
                for role in role_truth
            },
        },
        "selected_threshold_metrics": {
            "attack": graph_best,
            **{
                role: role_best[role]
                for role in role_best
            },
        },
        "end_to_end_validation_metrics": end_to_end,
        "validation_prediction_cache_sha256": (
            sha256_file(cache_path)
        ),
        "threshold_tuning_split": "validation",
        "model_weights_changed": False,
        "training_performed": False,
        "test_evaluation_authorized": False,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
    }
    threshold_manifest["threshold_manifest_sha256"] = (
        canonical_sha256(threshold_manifest)
    )

    manifest_path = (
        output_dir
        / "V5_P2_B4_FROZEN_THRESHOLD_MANIFEST.json"
    )
    write_json(manifest_path, threshold_manifest)

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "decision": (
            "FREEZE_P2_VALIDATION_THRESHOLDS_AND_AUTHORIZE_"
            "B5_ONE_SHOT_TEST_AUTHORIZATION"
        ),
        "threshold_manifest": threshold_manifest,
        "artifacts": {
            "threshold_manifest": [
                str(manifest_path),
                sha256_file(manifest_path),
            ],
            "validation_prediction_cache": [
                str(cache_path),
                sha256_file(cache_path),
            ],
            "attack_sweep_csv": [
                str(graph_sweep_path),
                sha256_file(graph_sweep_path),
            ],
            "role_sweep_csv": {
                role: [
                    str(path),
                    sha256_file(path),
                ]
                for role, path in role_sweep_paths.items()
            },
        },
        "provenance": {
            name: sha256_file(path)
            for name, path in paths.items()
        },
        "security_boundary": {
            "training_performed": False,
            "model_weights_changed": False,
            "checkpoint_deserialized": True,
            "validation_dataset_constructed": True,
            "validation_tensor_contents_accessed": True,
            "threshold_tuning_split": "validation",
            "test_directory_existence_checked": False,
            "test_directory_enumerated": False,
            "test_tensor_files_opened": False,
            "test_tensor_contents_accessed": False,
            "test_dataset_constructed": False,
            "test_windows_constructed": False,
            "test_evaluation_performed": False,
            "test_evaluation_authorized": False,
        },
        "failures": failures,
        "warnings": warnings,
        "next_stage": (
            "V5_P2_B5_ONE_SHOT_TEST_AUTHORIZATION"
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    lock = {
        "status": COMPLETE,
        "decision": (
            "FREEZE_P2_VALIDATION_THRESHOLDS_AND_AUTHORIZE_"
            "B5_ONE_SHOT_TEST_AUTHORIZATION"
        ),
        "report_sha256": sha256_file(report_path),
        "threshold_manifest_file_sha256": (
            sha256_file(manifest_path)
        ),
        "threshold_manifest_sha256": (
            threshold_manifest[
                "threshold_manifest_sha256"
            ]
        ),
        "validation_prediction_cache_sha256": (
            sha256_file(cache_path)
        ),
        "selected_seed": EXPECTED_SELECTED_SEED,
        "selected_epoch": EXPECTED_SELECTED_EPOCH,
        "selected_checkpoint_sha256": (
            sha256_file(selected_checkpoint)
        ),
        "thresholds": thresholds,
        "count_decision": "argmax",
        "training_performed": False,
        "model_weights_changed": False,
        "validation_tensor_contents_accessed": True,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "test_evaluation_performed": False,
        "test_evaluation_authorized": False,
        "next_stage": (
            "V5_P2_B5_ONE_SHOT_TEST_AUTHORIZATION"
        ),
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(
        output_dir / f"{STAGE}_LOCK.json",
        lock,
    )
    atomic_write(
        output_dir / COMPLETE,
        COMPLETE + "\n",
    )

    print("===== V5 P2-B4 VALIDATION THRESHOLD TUNING =====")
    print("status: COMPLETE")
    print(
        "decision: "
        "FREEZE_P2_VALIDATION_THRESHOLDS_AND_AUTHORIZE_"
        "B5_ONE_SHOT_TEST_AUTHORIZATION"
    )
    print("selected_seed:", EXPECTED_SELECTED_SEED)
    print("selected_epoch:", EXPECTED_SELECTED_EPOCH)
    print(
        "selected_checkpoint_sha256:",
        sha256_file(selected_checkpoint),
    )
    print(
        "validation_graph_auroc:",
        graph_auroc,
    )
    print(
        "validation_graph_average_precision:",
        graph_ap,
    )
    print(
        "validation_count_active_macro_f1:",
        count_macro_f1,
    )
    print(
        "validation_count_active_accuracy:",
        count_accuracy,
    )
    for name in (
        "attack",
        "source",
        "transit",
        "victim",
        "path",
    ):
        print(f"{name}_threshold:", thresholds[name])
    print(
        "attack_balanced_accuracy:",
        graph_best["balanced_accuracy"],
    )
    print(
        "attack_f1:",
        graph_best["f1"],
    )
    for role in ("source", "transit", "victim", "path"):
        print(
            f"{role}_node_f1:",
            role_best[role]["node_f1"],
        )
        print(
            f"{role}_active_exact_set_accuracy:",
            role_best[role][
                "active_exact_set_accuracy"
            ],
        )
    print(
        "validation_all_task_exact_accuracy:",
        end_to_end["all_task_exact_accuracy"],
    )
    print("training_performed: false")
    print("model_weights_changed: false")
    print("validation_tensor_contents_accessed: true")
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")
    print("test_evaluation_performed: false")
    print("test_evaluation_authorized: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print(
        "threshold_manifest_sha256:",
        threshold_manifest[
            "threshold_manifest_sha256"
        ],
    )
    print(
        "next_stage: "
        "V5_P2_B5_ONE_SHOT_TEST_AUTHORIZATION"
    )
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
