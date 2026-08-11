#!/usr/bin/env python3
"""
V5 P2-B6 One-Shot Blind Test Evaluation

This is the first and only authorized P2 test evaluation.

Security sequence
-----------------
1. Validate B1, B3, B4, and B5 without checking or enumerating runs/test.
2. Create the immutable B6 output guard.
3. Atomically consume the B5 authorization token.
4. Write TEST_ACCESS_STARTED.
5. Only then enumerate and open P2 test tensors.
6. Stream every matched test pair exactly once through the frozen model.
7. Write the primary metrics and immutable prediction cache.
8. Write a completion record. A successful rerun is forbidden.

No training, checkpoint selection, threshold tuning, calibration, or weight
changes are performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


STAGE = "V5_P2_B6_ONE_SHOT_TEST_EVALUATION"
COMPLETE = f"{STAGE}_COMPLETE"
ACCESS_STARTED = f"{STAGE}_TEST_ACCESS_STARTED"
IRREVERSIBLE_FAILURE = f"{STAGE}_IRREVERSIBLE_FAILURE"

EXPECTED_PROTOCOL_SHA = (
    "0817ae7812f91e3c75260589acaf52bd8a74b07b2114a451b4773578efde4b60"
)
EXPECTED_SELECTED_SEED = 127
EXPECTED_SELECTED_EPOCH = 59
EXPECTED_CHECKPOINT_SHA = (
    "7d4afae2214f67ecd9c65c6ff4614ba8238614234d7b07cdf1408b6d3efb07ef"
)
EXPECTED_THRESHOLD_MANIFEST_SHA = (
    "64bb23f1d0f36ec5f09236c4939317cddc96260f0ed4695036c4e802cd3056a7"
)
EXPECTED_AUTHORIZATION_ID = (
    "95da82aebaf1b61b13ce4bf31346b08500037841369cfd3fe5dba0965dff24e2"
)
EXPECTED_THRESHOLDS = {
    "attack": 0.2,
    "source": 0.778,
    "transit": 0.824,
    "victim": 0.661,
    "path": 0.545,
}

EXPECTED_TEST_FILES = 138
EXPECTED_TEST_PAIRS = 69
WINDOW = 32
STRIDE = 8
PAIR_WINDOW_BATCH = 64
PARAMETER_COUNT = 43_273

ROLES = ("source", "transit", "victim", "path")
TARGET_KEYS = {
    "source": "y_source",
    "transit": "y_transit",
    "victim": "y_victim",
    "path": "y_attack_path",
}
OUTPUT_KEYS = {
    "source": "source_logits",
    "transit": "transit_logits",
    "victim": "victim_logits",
    "path": "path_logits",
}


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


def explicit_false(
    document: dict[str, Any],
    key: str,
) -> bool:
    boundary = document.get("security_boundary")
    if isinstance(boundary, dict) and key in boundary:
        return boundary[key] is False
    return document.get(key) is False


def create_exclusive_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def binary_auroc(
    truth: np.ndarray,
    score: np.ndarray,
) -> float:
    y = truth.astype(np.int64).reshape(-1)
    s = score.astype(np.float64).reshape(-1)

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both binary classes")

    order = np.argsort(s, kind="mergesort")
    sorted_score = s[order]
    ranks = np.empty(len(s), dtype=np.float64)

    start = 0
    while start < len(s):
        stop = start + 1
        while (
            stop < len(s)
            and sorted_score[stop] == sorted_score[start]
        ):
            stop += 1

        average_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_rank
        start = stop

    positive_rank_sum = ranks[y == 1].sum()
    return float(
        (
            positive_rank_sum
            - positives * (positives + 1) / 2
        )
        / (positives * negatives)
    )


def average_precision(
    truth: np.ndarray,
    score: np.ndarray,
) -> float:
    y = truth.astype(np.int64).reshape(-1)
    s = score.astype(np.float64).reshape(-1)

    positives = int((y == 1).sum())
    if positives == 0:
        raise ValueError(
            "average precision requires positive examples"
        )

    order = np.argsort(-s, kind="mergesort")
    sorted_truth = y[order]
    cumulative_positive = np.cumsum(sorted_truth)
    positions = np.arange(1, len(y) + 1)
    precision_at_rank = cumulative_positive / positions

    return float(
        (precision_at_rank * sorted_truth).sum()
        / positives
    )


def binary_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    y = truth.astype(bool).reshape(-1)
    p = prediction.astype(bool).reshape(-1)

    tp = int((y & p).sum())
    tn = int((~y & ~p).sum())
    fp = int((~y & p).sum())
    fn = int((y & ~p).sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = (
        2 * precision * recall
        / max(1e-12, precision + recall)
    )

    return {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / max(1, fp + tn),
        "specificity": specificity,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support_negative": tn + fp,
        "support_positive": tp + fn,
    }


def node_set_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    graph_truth: np.ndarray,
) -> dict[str, Any]:
    y = truth.astype(bool)
    p = prediction.astype(bool)

    flat = binary_metrics(
        y.reshape(-1),
        p.reshape(-1),
    )
    exact = (y == p).all(axis=1)
    active = graph_truth.astype(bool)
    control = ~active

    return {
        "node": flat,
        "exact_set_all_items": float(exact.mean()),
        "exact_set_attack_items": float(
            exact[active].mean()
        ),
        "exact_set_control_items": float(
            exact[control].mean()
        ),
        "positive_entries": int(y.sum()),
        "window_count": int(len(y)),
    }


def multiclass_metrics_raw(
    truth_raw: np.ndarray,
    prediction_raw: np.ndarray,
    classes: Iterable[int] = (1, 2, 3, 4),
) -> dict[str, Any]:
    y = truth_raw.astype(np.int64).reshape(-1)
    p = prediction_raw.astype(np.int64).reshape(-1)
    labels = list(classes)

    confusion = np.zeros(
        (len(labels), len(labels)),
        dtype=np.int64,
    )
    index = {
        label: position
        for position, label in enumerate(labels)
    }

    for truth_value, prediction_value in zip(y, p):
        confusion[
            index[int(truth_value)],
            index[int(prediction_value)],
        ] += 1

    per_class = {}
    f1_values = []
    for position, label in enumerate(labels):
        tp = int(confusion[position, position])
        fp = int(confusion[:, position].sum()) - tp
        fn = int(confusion[position, :].sum()) - tp
        support = int(confusion[position, :].sum())

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (
            2 * precision * recall
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
        "accuracy": float((y == p).mean()),
        "macro_f1": float(np.mean(f1_values)),
        "classes": labels,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def validate_run(
    payload: Any,
    path: Path,
) -> int:
    if not isinstance(payload, dict):
        raise TypeError(f"run payload is not a dict: {path}")

    x = payload.get("x")
    if (
        not isinstance(x, torch.Tensor)
        or x.dtype != torch.float32
        or x.ndim != 3
        or tuple(x.shape[1:]) != (16, 81)
    ):
        raise ValueError(f"invalid x tensor: {path}")

    length = int(x.shape[0])

    scalar_keys = (
        "y_attack",
        "y_attacker_count",
    )
    node_keys = (
        "y_source",
        "y_transit",
        "y_victim",
        "y_attack_path",
        "role_mask",
    )

    for key in scalar_keys:
        value = payload.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 1
            or int(value.shape[0]) != length
        ):
            raise ValueError(
                f"invalid {key} tensor in {path}"
            )

    for key in node_keys:
        value = payload.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or tuple(value.shape) != (length, 16)
        ):
            raise ValueError(
                f"invalid {key} tensor in {path}"
            )

    return length


def build_pair_index(
    test_dir: Path,
) -> list[dict[str, Path | str]]:
    files = sorted(
        path
        for path in test_dir.iterdir()
        if path.is_file() and path.suffix == ".pt"
    )

    if len(files) != EXPECTED_TEST_FILES:
        raise RuntimeError(
            f"test .pt file count={len(files)}, "
            f"expected {EXPECTED_TEST_FILES}"
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
            raise RuntimeError(
                f"unexpected test tensor filename: {name}"
            )

        if not pair_key:
            raise RuntimeError(
                f"empty pair key in test filename: {name}"
            )

        pair = pairs.setdefault(pair_key, {})
        if mode in pair:
            raise RuntimeError(
                f"duplicate {mode} tensor for {pair_key}"
            )
        pair[mode] = path

    if len(pairs) != EXPECTED_TEST_PAIRS:
        raise RuntimeError(
            f"test pair count={len(pairs)}, "
            f"expected {EXPECTED_TEST_PAIRS}"
        )

    records = []
    for pair_key in sorted(pairs):
        pair = pairs[pair_key]
        if set(pair) != {"attack", "control"}:
            raise RuntimeError(
                f"incomplete test pair: {pair_key}"
            )
        records.append(
            {
                "pair_key": pair_key,
                "attack": pair["attack"],
                "control": pair["control"],
            }
        )
    return records


def stack_pair_batch(
    *,
    attack_run: dict[str, Any],
    control_run: dict[str, Any],
    starts: list[int],
    primary_indices: tuple[int, ...],
    physical_port_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    x_items = []
    attack_targets = []
    count_targets = []
    role_targets = {
        role: []
        for role in ROLES
    }
    role_masks = []

    for start in starts:
        stop = start + WINDOW
        target = stop - 1

        for run in (attack_run, control_run):
            x = (
                run["x"][
                    start:stop,
                    :,
                    primary_indices,
                ]
                .permute(1, 2, 0)
                .contiguous()
            )
            if tuple(x.shape) != (16, 58, 32):
                raise RuntimeError(
                    f"constructed test x shape={tuple(x.shape)}"
                )

            x_items.append(x)
            attack_targets.append(
                run["y_attack"][target].clone()
            )
            count_targets.append(
                run["y_attacker_count"][target].clone()
            )
            for role in ROLES:
                role_targets[role].append(
                    run[TARGET_KEYS[role]][target].clone()
                )
            role_masks.append(
                run["role_mask"][target].clone()
            )

    return {
        "x": torch.stack(x_items, dim=0),
        "physical_port_mask": (
            physical_port_mask
            .unsqueeze(0)
            .expand(len(x_items), -1, -1)
            .clone()
        ),
        "y_attack": torch.stack(attack_targets),
        "y_attacker_count": torch.stack(count_targets),
        **{
            TARGET_KEYS[role]: torch.stack(
                role_targets[role],
                dim=0,
            )
            for role in ROLES
        },
        "role_mask": torch.stack(role_masks, dim=0),
    }


def concatenate(
    values: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate(parts, axis=0)
        for key, parts in values.items()
    }


def evaluate_test_once(
    *,
    pair_records: list[dict[str, Path | str]],
    model: torch.nn.Module,
    device: torch.device,
    thresholds: dict[str, float],
    primary_indices: tuple[int, ...],
    physical_port_mask: torch.Tensor,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    list[dict[str, Any]],
]:
    arrays: dict[str, list[np.ndarray]] = {
        "graph_truth": [],
        "graph_score": [],
        "graph_prediction": [],
        "count_truth_raw": [],
        "count_prediction_raw": [],
        "pair_ordinal": [],
        "window_start": [],
        "mode": [],
    }
    for role in ROLES:
        arrays[f"{role}_truth"] = []
        arrays[f"{role}_score"] = []
        arrays[f"{role}_prediction_ungated"] = []
        arrays[f"{role}_prediction_gated"] = []

    pair_manifest_rows: list[dict[str, Any]] = []
    total_forward_batches = 0
    total_windows = 0
    role_mask_mismatch_count = 0

    model.eval()

    with torch.inference_mode():
        for pair_ordinal, record in enumerate(
            pair_records
        ):
            pair_key = str(record["pair_key"])
            attack_path = Path(record["attack"])
            control_path = Path(record["control"])

            attack_run = torch.load(
                attack_path,
                map_location="cpu",
                weights_only=False,
            )
            control_run = torch.load(
                control_path,
                map_location="cpu",
                weights_only=False,
            )

            attack_length = validate_run(
                attack_run,
                attack_path,
            )
            control_length = validate_run(
                control_run,
                control_path,
            )
            common_length = min(
                attack_length,
                control_length,
            )
            starts = list(
                range(
                    0,
                    common_length - WINDOW + 1,
                    STRIDE,
                )
            )

            if not starts:
                raise RuntimeError(
                    f"test pair {pair_key} produces no windows"
                )

            pair_manifest_rows.append(
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
                }
            )

            for offset in range(
                0,
                len(starts),
                PAIR_WINDOW_BATCH,
            ):
                selected_starts = starts[
                    offset:offset + PAIR_WINDOW_BATCH
                ]
                batch = stack_pair_batch(
                    attack_run=attack_run,
                    control_run=control_run,
                    starts=selected_starts,
                    primary_indices=primary_indices,
                    physical_port_mask=physical_port_mask,
                )

                graph_truth_tensor = (
                    batch["y_attack"]
                    .to(torch.int64)
                )
                count_truth_tensor = (
                    batch["y_attacker_count"]
                    .to(torch.int64)
                )

                if bool(
                    (
                        ~torch.isin(
                            graph_truth_tensor,
                            torch.tensor(
                                [0, 1],
                                dtype=torch.int64,
                            ),
                        )
                    )
                    .any()
                    .item()
                ):
                    raise RuntimeError(
                        f"nonbinary graph label in {pair_key}"
                    )

                active = graph_truth_tensor == 1
                control = ~active
                if bool(
                    (
                        (count_truth_tensor[active] < 1)
                        | (count_truth_tensor[active] > 4)
                    ).any().item()
                ):
                    raise RuntimeError(
                        f"active count outside 1..4 in {pair_key}"
                    )
                if bool(
                    (count_truth_tensor[control] != 0)
                    .any()
                    .item()
                ):
                    raise RuntimeError(
                        f"control count is nonzero in {pair_key}"
                    )

                expected_role_mask = (
                    batch["y_source"].to(torch.int64)
                    + 2 * batch["y_transit"].to(torch.int64)
                    + 4 * batch["y_victim"].to(torch.int64)
                )
                role_mask_mismatch_count += int(
                    (
                        batch["role_mask"].to(torch.int64)
                        != expected_role_mask
                    )
                    .sum()
                    .item()
                )

                outputs = model(
                    batch["x"].to(
                        device=device,
                        dtype=torch.float32,
                    ),
                    batch["physical_port_mask"].to(
                        device=device,
                    ),
                )
                total_forward_batches += 1

                graph_score = (
                    torch.sigmoid(
                        outputs["attack_logits"]
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                graph_prediction = (
                    graph_score >= thresholds["attack"]
                ).astype(np.uint8)

                count_prediction_raw = (
                    outputs["count_logits"]
                    .argmax(dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int8)
                    + 1
                )

                graph_truth = (
                    graph_truth_tensor
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                count_truth_raw = (
                    count_truth_tensor
                    .cpu()
                    .numpy()
                    .astype(np.int8)
                )

                item_count = len(graph_truth)
                starts_repeated = np.repeat(
                    np.asarray(
                        selected_starts,
                        dtype=np.int32,
                    ),
                    2,
                )
                mode = np.tile(
                    np.asarray([1, 0], dtype=np.uint8),
                    len(selected_starts),
                )

                arrays["graph_truth"].append(
                    graph_truth
                )
                arrays["graph_score"].append(
                    graph_score
                )
                arrays["graph_prediction"].append(
                    graph_prediction
                )
                arrays["count_truth_raw"].append(
                    count_truth_raw
                )
                arrays["count_prediction_raw"].append(
                    count_prediction_raw
                )
                arrays["pair_ordinal"].append(
                    np.full(
                        item_count,
                        pair_ordinal,
                        dtype=np.int16,
                    )
                )
                arrays["window_start"].append(
                    starts_repeated
                )
                arrays["mode"].append(mode)

                for role in ROLES:
                    truth = (
                        batch[TARGET_KEYS[role]]
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )
                    score = (
                        torch.sigmoid(
                            outputs[OUTPUT_KEYS[role]]
                        )
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    prediction_ungated = (
                        score >= thresholds[role]
                    ).astype(np.uint8)
                    prediction_gated = (
                        prediction_ungated
                        * graph_prediction[:, None]
                    ).astype(np.uint8)

                    arrays[f"{role}_truth"].append(
                        truth
                    )
                    arrays[f"{role}_score"].append(
                        score
                    )
                    arrays[
                        f"{role}_prediction_ungated"
                    ].append(prediction_ungated)
                    arrays[
                        f"{role}_prediction_gated"
                    ].append(prediction_gated)

                total_windows += item_count

            del attack_run
            del control_run

            if (
                (pair_ordinal + 1) % 10 == 0
                or pair_ordinal + 1 == len(pair_records)
            ):
                print(
                    f"test evaluation "
                    f"{pair_ordinal + 1}/{len(pair_records)} pairs"
                )

    combined = concatenate(arrays)

    if role_mask_mismatch_count != 0:
        raise RuntimeError(
            f"test role_mask mismatch count="
            f"{role_mask_mismatch_count}"
        )

    graph_truth = combined["graph_truth"]
    graph_score = combined["graph_score"]
    graph_prediction = combined["graph_prediction"]
    active = graph_truth == 1
    control = ~active

    count_metrics = multiclass_metrics_raw(
        combined["count_truth_raw"][active],
        combined["count_prediction_raw"][active],
    )

    roles = {}
    for role in ROLES:
        truth = combined[f"{role}_truth"]
        score = combined[f"{role}_score"]

        roles[role] = {
            "threshold": thresholds[role],
            "auroc": binary_auroc(
                truth.reshape(-1),
                score.reshape(-1),
            ),
            "average_precision": average_precision(
                truth.reshape(-1),
                score.reshape(-1),
            ),
            "ungated": node_set_metrics(
                truth,
                combined[
                    f"{role}_prediction_ungated"
                ],
                graph_truth,
            ),
            "graph_gated": node_set_metrics(
                truth,
                combined[
                    f"{role}_prediction_gated"
                ],
                graph_truth,
            ),
        }

    graph_correct = (
        graph_prediction == graph_truth
    )
    count_correct = np.ones(
        len(graph_truth),
        dtype=bool,
    )
    count_correct[active] = (
        combined["count_prediction_raw"][active]
        == combined["count_truth_raw"][active]
    )

    all_task_exact = graph_correct & count_correct
    for role in ROLES:
        all_task_exact &= (
            combined[
                f"{role}_prediction_gated"
            ]
            == combined[f"{role}_truth"]
        ).all(axis=1)

    metrics = {
        "graph": {
            **binary_metrics(
                graph_truth,
                graph_prediction,
            ),
            "threshold": thresholds["attack"],
            "auroc": binary_auroc(
                graph_truth,
                graph_score,
            ),
            "average_precision": average_precision(
                graph_truth,
                graph_score,
            ),
        },
        "count_active": count_metrics,
        "roles": roles,
        "strict_end_to_end": {
            "all_task_exact_accuracy": float(
                all_task_exact.mean()
            ),
            "all_task_exact_attack_accuracy": float(
                all_task_exact[active].mean()
            ),
            "all_task_exact_control_accuracy": float(
                all_task_exact[control].mean()
            ),
            "definition": (
                "graph correct; active count correct; "
                "all graph-gated source/transit/victim/path "
                "sets exactly correct; count ignored for controls"
            ),
        },
        "test_item_count": int(total_windows),
        "test_pair_count": int(len(pair_records)),
        "test_run_count": int(2 * len(pair_records)),
        "forward_batch_count": int(
            total_forward_batches
        ),
        "graph_positive_items": int(active.sum()),
        "graph_negative_items": int(control.sum()),
        "role_mask_mismatch_count": int(
            role_mask_mismatch_count
        ),
    }
    return metrics, combined, pair_manifest_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--b1-dir", type=Path, required=True)
    parser.add_argument("--b3-dir", type=Path, required=True)
    parser.add_argument("--b4-dir", type=Path, required=True)
    parser.add_argument("--b5-dir", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument(
        "--model-source-path",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    b1_dir = args.b1_dir.expanduser().resolve()
    b3_dir = args.b3_dir.expanduser().resolve()
    b4_dir = args.b4_dir.expanduser().resolve()
    b5_dir = args.b5_dir.expanduser().resolve()
    loader_path = args.loader_path.expanduser().resolve()
    model_source_path = (
        args.model_source_path.expanduser().resolve()
    )
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(
            f"STOP: one-shot output already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2

    # No runs/test path is formed, checked, or enumerated before token use.
    if not root.is_dir():
        print(
            f"STOP: dataset root missing: {root}",
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    paths = {
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
        "b4_report": (
            b4_dir
            / "V5_P2_B4_VALIDATION_THRESHOLD_TUNING.json"
        ),
        "b4_lock": (
            b4_dir
            / "V5_P2_B4_VALIDATION_THRESHOLD_TUNING_LOCK.json"
        ),
        "b4_manifest": (
            b4_dir
            / "V5_P2_B4_FROZEN_THRESHOLD_MANIFEST.json"
        ),
        "b5_report": (
            b5_dir
            / "V5_P2_B5_ONE_SHOT_TEST_AUTHORIZATION.json"
        ),
        "b5_lock": (
            b5_dir
            / "V5_P2_B5_ONE_SHOT_TEST_AUTHORIZATION_LOCK.json"
        ),
        "b5_token": (
            b5_dir
            / "V5_P2_B5_ONE_SHOT_TEST_AUTHORIZATION_TOKEN.json"
        ),
        "loader": loader_path,
        "model_source": model_source_path,
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(
                f"missing prerequisite {name}: {path}"
            )

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "authorization_consumed": False,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
            "test_evaluation_performed": False,
        }
        write_json(
            output_dir / f"{STAGE}.json",
            report,
        )
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        return 1

    b1_report = load_json(paths["b1_report"])
    b1_lock = load_json(paths["b1_lock"])
    b1_protocol = load_json(paths["b1_protocol"])
    b3_report = load_json(paths["b3_report"])
    b3_lock = load_json(paths["b3_lock"])
    b4_report = load_json(paths["b4_report"])
    b4_lock = load_json(paths["b4_lock"])
    b4_manifest = load_json(paths["b4_manifest"])
    b5_report = load_json(paths["b5_report"])
    b5_lock = load_json(paths["b5_lock"])
    b5_token = load_json(paths["b5_token"])

    # B1 chain.
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
    if (
        b1_lock.get("loader_sha256")
        != sha256_file(loader_path)
    ):
        failures.append("B1 loader SHA mismatch")
    if (
        b1_lock.get("model_sha256")
        != sha256_file(model_source_path)
    ):
        failures.append("B1 model-source SHA mismatch")

    # B3 selected checkpoint.
    if b3_report.get("status") != "COMPLETE":
        failures.append("B3 status is not COMPLETE")
    if (
        b3_lock.get("report_sha256")
        != sha256_file(paths["b3_report"])
    ):
        failures.append("B3 report SHA mismatch")
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
        failures.append(
            "selected checkpoint file SHA mismatch"
        )

    # B4 frozen thresholds.
    if b4_report.get("status") != "COMPLETE":
        failures.append("B4 status is not COMPLETE")
    if (
        b4_lock.get("report_sha256")
        != sha256_file(paths["b4_report"])
    ):
        failures.append("B4 report SHA mismatch")
    if (
        b4_lock.get("threshold_manifest_file_sha256")
        != sha256_file(paths["b4_manifest"])
    ):
        failures.append("B4 manifest file SHA mismatch")
    if (
        b4_lock.get("threshold_manifest_sha256")
        != EXPECTED_THRESHOLD_MANIFEST_SHA
    ):
        failures.append(
            "B4 canonical threshold-manifest SHA changed"
        )
    if b4_lock.get("thresholds") != EXPECTED_THRESHOLDS:
        failures.append("B4 thresholds changed")
    if b4_lock.get("count_decision") != "argmax":
        failures.append("B4 count decision changed")

    manifest_core = dict(b4_manifest)
    stored_manifest_sha = manifest_core.pop(
        "threshold_manifest_sha256",
        None,
    )
    if stored_manifest_sha != EXPECTED_THRESHOLD_MANIFEST_SHA:
        failures.append(
            "B4 manifest stored canonical SHA changed"
        )
    if (
        canonical_sha256(manifest_core)
        != EXPECTED_THRESHOLD_MANIFEST_SHA
    ):
        failures.append(
            "B4 manifest canonical content changed"
        )

    # B5 authorization.
    if b5_report.get("status") != "COMPLETE":
        failures.append("B5 status is not COMPLETE")
    if (
        b5_lock.get("report_sha256")
        != sha256_file(paths["b5_report"])
    ):
        failures.append("B5 report SHA mismatch")
    if (
        b5_lock.get("authorization_token_sha256")
        != sha256_file(paths["b5_token"])
    ):
        failures.append("B5 token SHA mismatch")
    if (
        b5_lock.get("authorization_id")
        != EXPECTED_AUTHORIZATION_ID
    ):
        failures.append("B5 authorization ID changed")
    if (
        b5_token.get("authorization_id")
        != EXPECTED_AUTHORIZATION_ID
    ):
        failures.append("B5 token authorization ID changed")
    if (
        b5_token.get("authorized_evaluation_count")
        != 1
    ):
        failures.append(
            "B5 token does not authorize exactly one evaluation"
        )
    if (
        b5_token.get("authorized_stage")
        != STAGE
    ):
        failures.append(
            "B5 token authorizes a different stage"
        )
    if b5_token.get("authorization_consumed") is not False:
        failures.append("B5 token is already consumed")
    if b5_token.get("test_evaluation_performed") is not False:
        failures.append(
            "B5 token says test evaluation already occurred"
        )
    if (
        b5_token.get("selected_checkpoint_sha256")
        != EXPECTED_CHECKPOINT_SHA
    ):
        failures.append(
            "B5 token checkpoint SHA changed"
        )
    if (
        b5_token.get("threshold_manifest_sha256")
        != EXPECTED_THRESHOLD_MANIFEST_SHA
    ):
        failures.append(
            "B5 token threshold-manifest SHA changed"
        )
    if b5_token.get("thresholds") != EXPECTED_THRESHOLDS:
        failures.append("B5 token thresholds changed")

    token_core = dict(b5_token)
    token_core.pop("authorization_id", None)
    if (
        canonical_sha256(token_core)
        != EXPECTED_AUTHORIZATION_ID
    ):
        failures.append("B5 token canonical content changed")

    # All prior stages must certify untouched test.
    for label, document in (
        ("B1", b1_report),
        ("B3", b3_report),
        ("B4", b4_report),
        ("B5", b5_report),
    ):
        if not explicit_false(
            document,
            "test_tensor_contents_accessed",
        ):
            failures.append(
                f"{label} does not certify untouched test tensors"
            )

    consumed_marker = (
        b5_dir
        / "V5_P2_B5_ONE_SHOT_TEST_AUTHORIZATION_CONSUMED.json"
    )
    consumption_complete_marker = (
        b5_dir
        / "V5_P2_B5_ONE_SHOT_TEST_CONSUMPTION_COMPLETE.json"
    )
    if consumed_marker.exists():
        failures.append(
            "B5 authorization-consumed marker already exists"
        )
    if consumption_complete_marker.exists():
        failures.append(
            "B5 consumption-complete marker already exists"
        )

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "authorization_consumed": False,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
            "test_evaluation_performed": False,
        }
        write_json(
            output_dir / f"{STAGE}.json",
            report,
        )
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    # Irreversible authorization consumption occurs BEFORE any test check.
    consumption_record = {
        "stage": STAGE,
        "event": "AUTHORIZATION_CONSUMED",
        "authorization_id": EXPECTED_AUTHORIZATION_ID,
        "authorization_token_sha256": (
            sha256_file(paths["b5_token"])
        ),
        "selected_checkpoint_sha256": (
            EXPECTED_CHECKPOINT_SHA
        ),
        "threshold_manifest_sha256": (
            EXPECTED_THRESHOLD_MANIFEST_SHA
        ),
        "authorized_evaluation_count": 1,
        "consumed_evaluation_count": 1,
        "test_access_started": False,
        "rerun_allowed": False,
    }
    create_exclusive_json(
        consumed_marker,
        consumption_record,
    )

    access_record = {
        "stage": STAGE,
        "event": "TEST_ACCESS_STARTED",
        "authorization_id": EXPECTED_AUTHORIZATION_ID,
        "selected_seed": EXPECTED_SELECTED_SEED,
        "selected_epoch": EXPECTED_SELECTED_EPOCH,
        "selected_checkpoint_sha256": (
            EXPECTED_CHECKPOINT_SHA
        ),
        "threshold_manifest_sha256": (
            EXPECTED_THRESHOLD_MANIFEST_SHA
        ),
        "thresholds": EXPECTED_THRESHOLDS,
        "count_decision": "argmax",
        "test_evaluation_count": 1,
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "threshold_tuning_allowed": False,
        "rerun_allowed": False,
    }
    write_json(
        output_dir / f"{ACCESS_STARTED}.json",
        access_record,
    )
    atomic_write(
        output_dir / ACCESS_STARTED,
        ACCESS_STARTED + "\n",
    )

    print("===== V5 P2-B6 ONE-SHOT BLIND TEST =====")
    print("authorization_consumed: true")
    print("authorization_id:", EXPECTED_AUTHORIZATION_ID)
    print("selected_seed:", EXPECTED_SELECTED_SEED)
    print("selected_epoch:", EXPECTED_SELECTED_EPOCH)
    print(
        "selected_checkpoint_sha256:",
        EXPECTED_CHECKPOINT_SHA,
    )
    print(
        "threshold_manifest_sha256:",
        EXPECTED_THRESHOLD_MANIFEST_SHA,
    )
    print("test_evaluation_count: 1")
    print("rerun_allowed: false")

    try:
        # First runs/test path creation/check happens after consumption.
        test_dir = root / "runs" / "test"
        if not test_dir.is_dir():
            raise FileNotFoundError(
                f"test directory missing: {test_dir}"
            )

        pair_records = build_pair_index(test_dir)

        loader_module = import_module(
            loader_path,
            "v5_p2_audited_helpers_b6",
        )
        model_module = import_module(
            model_source_path,
            "v5_p2_model_b6",
        )

        primary_indices = tuple(
            loader_module.PRIMARY58_INDICES
        )
        if (
            len(primary_indices) != 58
            or primary_indices
            != tuple(
                list(range(0, 30))
                + list(range(31, 56))
                + [62, 64, 65]
            )
        ):
            raise RuntimeError(
                "audited PRIMARY58 indices changed"
            )

        topology = torch.load(
            root / "topology.pt",
            map_location="cpu",
            weights_only=False,
        )
        physical_port_mask = (
            loader_module
            .derive_corrected_physical_port_mask(
                topology
            )
        )
        if tuple(
            physical_port_mask.shape
        ) != (16, 10):
            raise RuntimeError(
                "physical-port mask shape changed"
            )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        model = (
            model_module
            .P2B3Conv1DOnlyCount4()
            .to(device)
        )
        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )
        if parameter_count != PARAMETER_COUNT:
            raise RuntimeError(
                f"parameter_count={parameter_count}, "
                f"expected {PARAMETER_COUNT}"
            )

        checkpoint = torch.load(
            selected_checkpoint,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("seed") != EXPECTED_SELECTED_SEED:
            raise RuntimeError("checkpoint seed changed")
        if checkpoint.get("epoch") != EXPECTED_SELECTED_EPOCH:
            raise RuntimeError("checkpoint epoch changed")
        if (
            checkpoint.get("protocol_sha256")
            != EXPECTED_PROTOCOL_SHA
        ):
            raise RuntimeError(
                "checkpoint protocol SHA changed"
            )
        if (
            checkpoint.get("loader_sha256")
            != sha256_file(loader_path)
        ):
            raise RuntimeError(
                "checkpoint loader SHA changed"
            )
        if (
            checkpoint.get("model_source_sha256")
            != sha256_file(model_source_path)
        ):
            raise RuntimeError(
                "checkpoint model-source SHA changed"
            )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        print("device:", device)
        print("test_run_count:", 2 * len(pair_records))
        print("test_pair_count:", len(pair_records))
        print("parameter_count:", parameter_count)
        for name, value in EXPECTED_THRESHOLDS.items():
            print(f"{name}_threshold:", value)
        print("count_decision: argmax")
        print("training_performed: false")
        print("threshold_tuning_performed: false")

        metrics, predictions, pair_manifest_rows = (
            evaluate_test_once(
                pair_records=pair_records,
                model=model,
                device=device,
                thresholds=EXPECTED_THRESHOLDS,
                primary_indices=primary_indices,
                physical_port_mask=physical_port_mask,
            )
        )

        prediction_cache_path = (
            output_dir
            / "V5_P2_B6_LOCKED_TEST_PREDICTIONS_AND_TARGETS.npz"
        )
        np.savez_compressed(
            prediction_cache_path,
            **predictions,
        )

        pair_manifest_path = (
            output_dir
            / "V5_P2_B6_TEST_PAIR_WINDOW_MANIFEST.csv"
        )
        write_csv(
            pair_manifest_path,
            pair_manifest_rows,
        )

        # Primary report is written before any error analysis.
        report = {
            "stage": STAGE,
            "status": "COMPLETE",
            "decision": "FINAL_P2_BLIND_TEST_RESULT_RECORDED",
            "architecture": {
                "name": (
                    "B3_CAUSAL_DEPTHWISE_SEPARABLE_"
                    "CONV1D_ONLY_P2_COUNT4"
                ),
                "parameter_count": PARAMETER_COUNT,
                "model_source_sha256": (
                    sha256_file(model_source_path)
                ),
                "loader_source_sha256": (
                    sha256_file(loader_path)
                ),
            },
            "checkpoint": {
                "seed": EXPECTED_SELECTED_SEED,
                "epoch": EXPECTED_SELECTED_EPOCH,
                "path": str(selected_checkpoint),
                "sha256": EXPECTED_CHECKPOINT_SHA,
            },
            "frozen_decisions": {
                "thresholds": EXPECTED_THRESHOLDS,
                "count": "four-class argmax, raw counts 1..4",
                "threshold_manifest_sha256": (
                    EXPECTED_THRESHOLD_MANIFEST_SHA
                ),
                "protocol_sha256": EXPECTED_PROTOCOL_SHA,
            },
            "test_metrics": metrics,
            "test_evaluation_count": 1,
            "artifacts": {
                "locked_test_predictions_and_targets": [
                    str(prediction_cache_path),
                    sha256_file(prediction_cache_path),
                ],
                "test_pair_window_manifest": [
                    str(pair_manifest_path),
                    sha256_file(pair_manifest_path),
                ],
            },
            "provenance": {
                name: sha256_file(path)
                for name, path in paths.items()
            },
            "security_boundary": {
                "authorization_id": (
                    EXPECTED_AUTHORIZATION_ID
                ),
                "authorization_consumed": True,
                "training_performed": False,
                "model_weights_changed": False,
                "checkpoint_selection_performed": False,
                "threshold_tuning_performed": False,
                "threshold_calibration_performed": False,
                "test_directory_enumerated": True,
                "test_tensor_contents_accessed": True,
                "test_evaluation_performed": True,
                "test_evaluation_count": 1,
                "successful_rerun_authorized": False,
                "primary_report_written_before_error_analysis": True,
                "test_error_analysis_performed": False,
            },
            "failures": [],
            "warnings": warnings,
            "next_stage": (
                "V5_P2_FINAL_RESULTS_ARCHIVE_AND_ERROR_ANALYSIS_FROM_"
                "FROZEN_PREDICTION_CACHE"
            ),
        }

        report_path = output_dir / f"{STAGE}.json"
        write_json(report_path, report)

        lock = {
            "status": COMPLETE,
            "decision": "FINAL_P2_BLIND_TEST_RESULT_RECORDED",
            "report_sha256": sha256_file(report_path),
            "authorization_id": EXPECTED_AUTHORIZATION_ID,
            "authorization_token_sha256": (
                sha256_file(paths["b5_token"])
            ),
            "authorization_consumed": True,
            "selected_seed": EXPECTED_SELECTED_SEED,
            "selected_epoch": EXPECTED_SELECTED_EPOCH,
            "selected_checkpoint_sha256": (
                EXPECTED_CHECKPOINT_SHA
            ),
            "threshold_manifest_sha256": (
                EXPECTED_THRESHOLD_MANIFEST_SHA
            ),
            "thresholds": EXPECTED_THRESHOLDS,
            "count_decision": "argmax",
            "test_prediction_cache_sha256": (
                sha256_file(prediction_cache_path)
            ),
            "test_pair_window_manifest_sha256": (
                sha256_file(pair_manifest_path)
            ),
            "test_item_count": metrics["test_item_count"],
            "test_pair_count": metrics["test_pair_count"],
            "test_run_count": metrics["test_run_count"],
            "test_evaluation_count": 1,
            "test_directory_enumerated": True,
            "test_tensor_contents_accessed": True,
            "test_evaluation_performed": True,
            "successful_rerun_authorized": False,
            "next_stage": (
                "V5_P2_FINAL_RESULTS_ARCHIVE_AND_ERROR_ANALYSIS_FROM_"
                "FROZEN_PREDICTION_CACHE"
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

        completion_record = {
            "stage": STAGE,
            "event": "AUTHORIZATION_CONSUMPTION_COMPLETE",
            "authorization_id": EXPECTED_AUTHORIZATION_ID,
            "authorization_consumed": True,
            "test_evaluation_count": 1,
            "test_result_report_path": str(report_path),
            "test_result_report_sha256": (
                sha256_file(report_path)
            ),
            "test_prediction_cache_sha256": (
                sha256_file(prediction_cache_path)
            ),
            "successful_rerun_authorized": False,
        }
        create_exclusive_json(
            consumption_complete_marker,
            completion_record,
        )

        graph = metrics["graph"]
        count = metrics["count_active"]
        end_to_end = metrics["strict_end_to_end"]

        print("===== V5 P2-B6 FINAL BLIND TEST RESULT =====")
        print("status: COMPLETE")
        print("decision: FINAL_P2_BLIND_TEST_RESULT_RECORDED")
        print("test_item_count:", metrics["test_item_count"])
        print("test_pair_count:", metrics["test_pair_count"])
        print("test_run_count:", metrics["test_run_count"])
        print("graph_accuracy:", graph["accuracy"])
        print(
            "graph_balanced_accuracy:",
            graph["balanced_accuracy"],
        )
        print("graph_precision:", graph["precision"])
        print("graph_recall:", graph["recall"])
        print("graph_f1:", graph["f1"])
        print("graph_fpr:", graph["fpr"])
        print("graph_auroc:", graph["auroc"])
        print(
            "graph_average_precision:",
            graph["average_precision"],
        )
        print(
            "count_active_accuracy:",
            count["accuracy"],
        )
        print(
            "count_active_macro_f1:",
            count["macro_f1"],
        )
        for role in ROLES:
            role_result = metrics["roles"][role]
            print(
                f"{role}_average_precision:",
                role_result["average_precision"],
            )
            print(
                f"{role}_gated_node_f1:",
                role_result["graph_gated"]["node"]["f1"],
            )
            print(
                f"{role}_gated_exact_set_attack:",
                role_result["graph_gated"][
                    "exact_set_attack_items"
                ],
            )
        print(
            "all_task_exact_accuracy:",
            end_to_end["all_task_exact_accuracy"],
        )
        print(
            "all_task_exact_attack_accuracy:",
            end_to_end[
                "all_task_exact_attack_accuracy"
            ],
        )
        print(
            "all_task_exact_control_accuracy:",
            end_to_end[
                "all_task_exact_control_accuracy"
            ],
        )
        print("authorization_consumed: true")
        print("test_evaluation_count: 1")
        print("successful_rerun_authorized: false")
        print("failure_count: 0")
        print("warning_count:", len(warnings))
        print(
            "next_stage: "
            "V5_P2_FINAL_RESULTS_ARCHIVE_AND_ERROR_ANALYSIS_FROM_"
            "FROZEN_PREDICTION_CACHE"
        )
        print(COMPLETE)
        return 0

    except Exception as exc:
        failure_record = {
            "stage": STAGE,
            "status": "IRREVERSIBLE_FAILURE_AFTER_AUTHORIZATION_CONSUMPTION",
            "authorization_id": EXPECTED_AUTHORIZATION_ID,
            "authorization_consumed": True,
            "test_access_started": True,
            "test_evaluation_may_be_partial": True,
            "successful_rerun_authorized": False,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(
            output_dir / f"{IRREVERSIBLE_FAILURE}.json",
            failure_record,
        )
        atomic_write(
            output_dir / IRREVERSIBLE_FAILURE,
            IRREVERSIBLE_FAILURE + "\n",
        )
        print(IRREVERSIBLE_FAILURE)
        print("FAIL:", type(exc).__name__, str(exc))
        print(
            "Authorization was consumed. Do not rerun or remove "
            "the guard artifacts."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
