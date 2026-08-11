#!/usr/bin/env python3
"""
V4-A4a-2B
Validation-Only Exact-Decoder Checkpoint Selection

Modes
-----
source-audit
    Verifies all frozen provenance, all 12 retained checkpoints, the selected
    A3.11 L2 temporal policy, the A3-H32 reference exact score, selection gates,
    temporal de-overlap arithmetic, and persistence semantics. It does not open
    dataset sample arrays or execute model inference.

select
    Runs validation-only inference for every retained A4a-2A checkpoint,
    applies the exact unique constrained slot decoder under the frozen A3.11
    temporal policy, evaluates the frozen safety/quality gates, and freezes at
    most one checkpoint.

Scientific boundary
-------------------
* Validation only.
* No threshold search.
* No development-test loader or prediction access.
* No post-selection parameter changes.
* If no checkpoint passes every frozen gate, the stage returns HOLD and does
  not authorize development-test transfer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from v4_a4a_slot_model import (
    A4aFrozenSlotModel,
    NUM_ROUTERS,
    count_parameters,
    exact_unique_constrained_decode,
)


EXPECTED_PRIMARY_TRAINER_SHA = (
    "c424a4877c39cdbf1df54dba89d15291"
    "3c7aef3d7a265930f21c9af4ec52f42d"
)
EXPECTED_ORIGINAL_TRAINER_SHA = (
    "e5f9a48e9f7ca873c351743df92e3bb0"
    "0d00a893333434c7fd0e04734e010525"
)
EXPECTED_MODEL_SOURCE_SHA = (
    "5ebcf80385b95f329faf935cad8039b64"
    "afd79f464903308eb07e674d78a1704"
)
EXPECTED_BASE_CHECKPOINT_SHA = (
    "f61c1add6c057f7f53dd34fb1f9f4e95"
    "b01cefd5053e0e42bc100c76dac7f923"
)
EXPECTED_SLOT_MODULE_SHA = (
    "97b56aabb3239dfc7543c241665cacf99"
    "8eb6b1a1f5f67f931cdeb70e75c111c"
)
EXPECTED_PREFLIGHT_SCRIPT_SHA = (
    "4f605c2f34dd985b138430f6f37343cd"
    "0a1284a816ebfbd89479707c58bc5208"
)

EXPECTED_TOTAL_PARAMETERS = 3189
EXPECTED_TRAINABLE_PARAMETERS = 1076
EXPECTED_CHECKPOINT_COUNT = 12
EXPECTED_DEOVERLAP_STRIDE = 8
EXPECTED_H32_HORIZON = 32
EXPECTED_SAMPLES_PER_RUN = 3293
EXPECTED_H32_DECISIONS_PER_FULL_RUN = 381

POPCOUNT16 = np.array(
    [int(value).bit_count() for value in range(1 << NUM_ROUTERS)],
    dtype=np.uint8,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if (
                isinstance(value, dict)
                and value
                and all(isinstance(v, torch.Tensor) for v in value.values())
            ):
                return value
    raise RuntimeError("checkpoint model state dictionary not found")


def recursive_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for current_key, child in value.items():
            if current_key == key:
                found.append(child)
            found.extend(recursive_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(recursive_values(child, key))
    return found


def require_unique_nested_value(value: Any, key: str) -> Any:
    found = recursive_values(value, key)
    unique = []
    for item in found:
        if item not in unique:
            unique.append(item)
    if len(unique) != 1:
        raise RuntimeError(
            f"expected exactly one unique value for {key}, found {unique}"
        )
    return unique[0]


def extract_selected_policy(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        selected = value.get("selected_policy")
        if isinstance(selected, dict):
            return selected

    candidates: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            required = {"mode", "horizon", "graph_method", "graph_threshold"}
            if required.issubset(node.keys()):
                candidates.append(node)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    if len(candidates) != 1:
        raise RuntimeError(
            "could not uniquely identify selected temporal policy; "
            f"candidate_count={len(candidates)}"
        )
    return candidates[0]


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output.astype(np.float32)


def bitmask_from_truth(y_node: np.ndarray) -> np.ndarray:
    truth = np.asarray(y_node, dtype=bool)
    if truth.ndim != 2 or truth.shape[1] != NUM_ROUTERS:
        raise ValueError(f"unexpected truth shape: {truth.shape}")
    weights = (1 << np.arange(NUM_ROUTERS, dtype=np.uint16))[None, :]
    return np.sum(truth.astype(np.uint16) * weights, axis=1, dtype=np.uint16)


def bitmask_from_sets(predicted_sets: Iterable[Iterable[int]]) -> np.ndarray:
    output = []
    for predicted in predicted_sets:
        mask = 0
        for router in predicted:
            router_value = int(router)
            if not 0 <= router_value < NUM_ROUTERS:
                raise ValueError(f"invalid decoded router: {router_value}")
            mask |= 1 << router_value
        output.append(mask)
    return np.asarray(output, dtype=np.uint16)


def contiguous_true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    array = np.asarray(values, dtype=bool)
    padded = np.concatenate(
        [np.array([False]), array, np.array([False])]
    )
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(start), int(end))
        for start, end in zip(changes[0::2], changes[1::2])
    ]


def stable_point_mask(
    values: np.ndarray,
    streak_required: int,
) -> np.ndarray:
    output = np.zeros(len(values), dtype=bool)
    for start, end in contiguous_true_runs(values):
        if end - start >= streak_required:
            output[start:end] = True
    return output


def first_stable_completion(
    values: np.ndarray,
    streak_required: int,
) -> int | None:
    for start, end in contiguous_true_runs(values):
        if end - start >= streak_required:
            return start + streak_required - 1
    return None


def maximum_router_streak(predicted_masks: np.ndarray) -> int:
    maximum = 0
    for router in range(NUM_ROUTERS):
        active = (predicted_masks & (1 << router)) != 0
        for start, end in contiguous_true_runs(active):
            maximum = max(maximum, end - start)
    return int(maximum)


def has_persistent_router(
    predicted_masks: np.ndarray,
    streak_required: int,
) -> bool:
    return maximum_router_streak(predicted_masks) >= streak_required


def mean_set_churn(predicted_masks: np.ndarray) -> float:
    if len(predicted_masks) <= 1:
        return 0.0
    values = np.asarray(predicted_masks, dtype=np.uint16)
    churn = []
    for left, right in zip(values[:-1], values[1:]):
        union = int(left | right)
        if union == 0:
            churn.append(0.0)
        else:
            intersection = int(left & right)
            churn.append(
                1.0
                - int(POPCOUNT16[intersection])
                / int(POPCOUNT16[union])
            )
    return float(np.mean(churn))


def rolling_mean(
    values: np.ndarray,
    horizon: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[0] < horizon:
        shape = (0,) + array.shape[1:]
        return np.empty(shape, dtype=np.float32)
    leading = np.zeros((1,) + array.shape[1:], dtype=np.float64)
    cumulative = np.concatenate(
        [leading, np.cumsum(array, axis=0, dtype=np.float64)],
        axis=0,
    )
    return (
        (cumulative[horizon:] - cumulative[:-horizon]) / horizon
    ).astype(np.float32)


def aggregate_run(
    graph_logits: np.ndarray,
    slot_logits: np.ndarray,
    end_epochs: np.ndarray,
    *,
    deoverlap_stride: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.arange(
        0,
        int(graph_logits.shape[0]),
        deoverlap_stride,
        dtype=np.int64,
    )
    deoverlap_graph = graph_logits[positions]
    deoverlap_slot = slot_logits[positions]
    deoverlap_end = end_epochs[positions]

    aggregate_graph = rolling_mean(deoverlap_graph, horizon)
    aggregate_slot = rolling_mean(deoverlap_slot, horizon)
    decision_end = deoverlap_end[horizon - 1 :].astype(np.int32, copy=False)

    if not (
        aggregate_graph.shape[0]
        == aggregate_slot.shape[0]
        == decision_end.shape[0]
    ):
        raise RuntimeError("temporal aggregation alignment failure")
    return aggregate_graph, aggregate_slot, decision_end


def decode_aggregated_slots(
    aggregate_slot_logits: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    predicted_masks: list[np.ndarray] = []
    for start in range(0, aggregate_slot_logits.shape[0], chunk_size):
        stop = min(start + chunk_size, aggregate_slot_logits.shape[0])
        tensor = torch.from_numpy(
            np.asarray(
                aggregate_slot_logits[start:stop],
                dtype=np.float32,
            )
        )
        decoded = exact_unique_constrained_decode(tensor)
        predicted_masks.append(
            bitmask_from_sets(decoded["predicted_sets"])
        )
    if not predicted_masks:
        return np.empty(0, dtype=np.uint16)
    return np.concatenate(predicted_masks)


def evaluate_decision_trace(
    *,
    checkpoint_epoch: int,
    checkpoint_sha256: str,
    validation_loss: float,
    run_index: np.ndarray,
    decision_ordinal: np.ndarray,
    decision_end_epoch: np.ndarray,
    true_masks: np.ndarray,
    predicted_masks: np.ndarray,
    graph_probabilities: np.ndarray,
    stable_streak: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_index = np.asarray(run_index, dtype=np.int32)
    true_masks = np.asarray(true_masks, dtype=np.uint16)
    predicted_masks = np.asarray(predicted_masks, dtype=np.uint16)

    attack_mask = true_masks != 0
    normal_mask = ~attack_mask
    exact = true_masks == predicted_masks
    nonempty = predicted_masks != 0

    intersection = np.bitwise_and(true_masks, predicted_masks)
    false_positive = np.bitwise_and(
        predicted_masks,
        np.bitwise_not(true_masks),
    )
    false_negative = np.bitwise_and(
        true_masks,
        np.bitwise_not(predicted_masks),
    )

    tp_all = int(POPCOUNT16[intersection].sum())
    fp_all = int(POPCOUNT16[false_positive].sum())
    fn_all = int(POPCOUNT16[false_negative].sum())

    tp_attack = int(POPCOUNT16[intersection[attack_mask]].sum())
    fp_attack = int(POPCOUNT16[false_positive[attack_mask]].sum())
    fn_attack = int(POPCOUNT16[false_negative[attack_mask]].sum())

    overall_precision = (
        tp_all / (tp_all + fp_all) if tp_all + fp_all else 1.0
    )
    overall_recall = (
        tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    )
    attack_precision = (
        tp_attack / (tp_attack + fp_attack)
        if tp_attack + fp_attack
        else 1.0
    )
    attack_recall = (
        tp_attack / (tp_attack + fn_attack)
        if tp_attack + fn_attack
        else 0.0
    )
    attack_f1 = (
        2.0 * attack_precision * attack_recall
        / (attack_precision + attack_recall)
        if attack_precision + attack_recall
        else 0.0
    )

    predicted_counts = POPCOUNT16[predicted_masks]
    true_counts = POPCOUNT16[true_masks]

    run_rows: list[dict[str, Any]] = []
    normal_run_count = 0
    normal_run_ever = 0
    normal_run_persistent = 0
    maximum_normal_streak = 0

    attack_run_count = 0
    attack_run_first_exact = 0
    attack_run_first_stable_exact = 0
    first_exact_times: list[int] = []
    first_stable_times: list[int] = []
    stable_exact_points = 0
    attack_points_for_stability = 0
    churn_values: list[float] = []

    unique_runs, starts = np.unique(run_index, return_index=True)
    order = np.argsort(starts)
    unique_runs = unique_runs[order]

    for run in unique_runs:
        positions = np.flatnonzero(run_index == run)
        truth_values = true_masks[positions]
        if not np.all(truth_values == truth_values[0]):
            raise RuntimeError(
                f"truth attacker set changes within run {int(run)}"
            )
        truth_mask = int(truth_values[0])
        run_pred = predicted_masks[positions]
        run_exact = exact[positions]
        run_nonempty = nonempty[positions]
        max_streak = maximum_router_streak(run_pred)
        churn = mean_set_churn(run_pred)
        churn_values.append(churn)

        row = {
            "checkpoint_epoch": checkpoint_epoch,
            "run_index": int(run),
            "decision_count": int(len(positions)),
            "true_attacker_count": int(POPCOUNT16[truth_mask]),
            "true_attacker_mask": truth_mask,
            "exact_fraction": float(np.mean(run_exact)),
            "nonempty_fraction": float(np.mean(run_nonempty)),
            "maximum_router_prediction_streak": max_streak,
            "mean_set_churn": churn,
            "first_decision_end_epoch": int(
                decision_end_epoch[positions[0]]
            ),
            "last_decision_end_epoch": int(
                decision_end_epoch[positions[-1]]
            ),
        }

        if truth_mask == 0:
            normal_run_count += 1
            ever = bool(np.any(run_nonempty))
            persistent = has_persistent_router(
                run_pred,
                stable_streak,
            )
            normal_run_ever += int(ever)
            normal_run_persistent += int(persistent)
            maximum_normal_streak = max(
                maximum_normal_streak,
                max_streak,
            )
            row.update(
                {
                    "run_role": "normal",
                    "ever_false_isolated": ever,
                    "persistent_false_isolated": persistent,
                    "stable_exact_fraction": None,
                    "first_exact_decision": None,
                    "first_stable_exact_decision": None,
                }
            )
        else:
            attack_run_count += 1
            exact_positions = np.flatnonzero(run_exact)
            stable_mask = stable_point_mask(
                run_exact,
                stable_streak,
            )
            stable_completion = first_stable_completion(
                run_exact,
                stable_streak,
            )

            any_exact = exact_positions.size > 0
            any_stable = stable_completion is not None
            attack_run_first_exact += int(any_exact)
            attack_run_first_stable_exact += int(any_stable)
            if any_exact:
                first_exact_times.append(int(exact_positions[0]))
            if stable_completion is not None:
                first_stable_times.append(int(stable_completion))

            stable_exact_points += int(stable_mask.sum())
            attack_points_for_stability += int(len(run_exact))

            row.update(
                {
                    "run_role": "attack",
                    "ever_false_isolated": None,
                    "persistent_false_isolated": None,
                    "stable_exact_fraction": float(
                        np.mean(stable_mask)
                    ),
                    "first_exact_decision": (
                        int(exact_positions[0])
                        if any_exact
                        else None
                    ),
                    "first_stable_exact_decision": stable_completion,
                }
            )
        run_rows.append(row)

    summary = {
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "training_validation_total_loss": validation_loss,
        "decision_count": int(len(true_masks)),
        "attack_decision_count": int(attack_mask.sum()),
        "normal_decision_count": int(normal_mask.sum()),
        "set_precision_overall": overall_precision,
        "set_recall_overall": overall_recall,
        "attack_set_precision": attack_precision,
        "attack_set_recall": attack_recall,
        "attack_set_f1": attack_f1,
        "exact_localization": float(np.mean(exact)),
        "attack_exact_localization": (
            float(np.mean(exact[attack_mask]))
            if np.any(attack_mask)
            else 0.0
        ),
        "candidate_coverage": (
            float(np.mean(nonempty[attack_mask]))
            if np.any(attack_mask)
            else 0.0
        ),
        "normal_false_isolation_rate": (
            float(np.mean(nonempty[normal_mask]))
            if np.any(normal_mask)
            else 0.0
        ),
        "count_accuracy": float(
            np.mean(predicted_counts == true_counts)
        ),
        "attack_count_accuracy": (
            float(
                np.mean(
                    predicted_counts[attack_mask]
                    == true_counts[attack_mask]
                )
            )
            if np.any(attack_mask)
            else 0.0
        ),
        "normal_run_count": normal_run_count,
        "normal_run_ever_isolated_fraction": (
            normal_run_ever / normal_run_count
            if normal_run_count
            else 0.0
        ),
        "normal_run_persistent_false_isolation_fraction": (
            normal_run_persistent / normal_run_count
            if normal_run_count
            else 0.0
        ),
        "maximum_false_isolation_streak": maximum_normal_streak,
        "attack_run_count": attack_run_count,
        "attack_run_first_exact_coverage": (
            attack_run_first_exact / attack_run_count
            if attack_run_count
            else 0.0
        ),
        "attack_run_first_stable_exact_coverage": (
            attack_run_first_stable_exact / attack_run_count
            if attack_run_count
            else 0.0
        ),
        "stable_attack_exact_point_rate": (
            stable_exact_points / attack_points_for_stability
            if attack_points_for_stability
            else 0.0
        ),
        "median_time_to_first_exact_decisions": (
            float(np.median(first_exact_times))
            if first_exact_times
            else None
        ),
        "median_time_to_stable_exact_decisions": (
            float(np.median(first_stable_times))
            if first_stable_times
            else None
        ),
        "mean_attack_set_churn": (
            float(np.mean(churn_values))
            if churn_values
            else 0.0
        ),
        "attack_node_tp": tp_attack,
        "attack_node_fp": fp_attack,
        "attack_node_fn": fn_attack,
        "all_node_tp": tp_all,
        "all_node_fp": fp_all,
        "all_node_fn": fn_all,
        "graph_probability_mean": float(
            np.mean(graph_probabilities)
        ),
        "graph_probability_attack_mean": (
            float(np.mean(graph_probabilities[attack_mask]))
            if np.any(attack_mask)
            else 0.0
        ),
        "graph_probability_normal_mean": (
            float(np.mean(graph_probabilities[normal_mask]))
            if np.any(normal_mask)
            else 0.0
        ),
    }
    return summary, run_rows


def verify_paths_and_provenance(
    args: argparse.Namespace,
) -> dict[str, Any]:
    required = [
        args.repo,
        args.data_dir,
        args.training_dir,
        args.primary_trainer,
        args.original_trainer,
        args.model_source,
        args.base_checkpoint,
        args.slot_module,
        args.preflight_script,
        args.preflight_dir
        / "V4_A4A_2B0_VALIDATION_SELECTION_PREFLIGHT_PASS",
        args.preflight_dir / "validation_selection_preflight.json",
        args.preflight_dir / "A4A_2B0_PREFLIGHT_LOCK.json",
        args.training_dir / "summary.json",
        args.training_dir / "validation_checkpoint_manifest.json",
        args.training_dir / "V4_A4A_2A_PRIMARY_TRAINING_PASS",
        args.full_training_contract,
        args.slot_contract,
        args.frozen_l2_policy,
        args.frozen_operational_policy,
        args.selected_validation_policy,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing required paths:\n  " + "\n  ".join(missing)
        )

    hashes = {
        "primary_trainer_sha256": sha256_file(args.primary_trainer),
        "original_trainer_sha256": sha256_file(args.original_trainer),
        "model_source_sha256": sha256_file(args.model_source),
        "base_checkpoint_sha256": sha256_file(
            args.base_checkpoint
        ),
        "slot_module_sha256": sha256_file(args.slot_module),
        "preflight_script_sha256": sha256_file(
            args.preflight_script
        ),
        "preflight_report_sha256": sha256_file(
            args.preflight_dir
            / "validation_selection_preflight.json"
        ),
        "training_summary_sha256": sha256_file(
            args.training_dir / "summary.json"
        ),
        "checkpoint_manifest_sha256": sha256_file(
            args.training_dir
            / "validation_checkpoint_manifest.json"
        ),
        "full_training_contract_sha256": sha256_file(
            args.full_training_contract
        ),
        "slot_contract_sha256": sha256_file(args.slot_contract),
        "frozen_l2_policy_sha256": sha256_file(
            args.frozen_l2_policy
        ),
        "frozen_operational_policy_sha256": sha256_file(
            args.frozen_operational_policy
        ),
        "selected_validation_policy_sha256": sha256_file(
            args.selected_validation_policy
        ),
        "selector_script_sha256": sha256_file(Path(__file__)),
    }

    expected = {
        "primary_trainer_sha256": EXPECTED_PRIMARY_TRAINER_SHA,
        "original_trainer_sha256": EXPECTED_ORIGINAL_TRAINER_SHA,
        "model_source_sha256": EXPECTED_MODEL_SOURCE_SHA,
        "base_checkpoint_sha256": EXPECTED_BASE_CHECKPOINT_SHA,
        "slot_module_sha256": EXPECTED_SLOT_MODULE_SHA,
        "preflight_script_sha256": EXPECTED_PREFLIGHT_SCRIPT_SHA,
    }
    mismatches = [
        f"{key}: {hashes[key]} != {value}"
        for key, value in expected.items()
        if hashes[key] != value
    ]
    if mismatches:
        raise RuntimeError(
            "provenance hash mismatch:\n  " + "\n  ".join(mismatches)
        )

    preflight = load_json(
        args.preflight_dir / "validation_selection_preflight.json"
    )
    preflight_lock = load_json(
        args.preflight_dir / "A4A_2B0_PREFLIGHT_LOCK.json"
    )
    summary = load_json(args.training_dir / "summary.json")
    manifest = load_json(
        args.training_dir / "validation_checkpoint_manifest.json"
    )
    full_contract = load_json(args.full_training_contract)
    slot_contract = load_json(args.slot_contract)
    l2_policy_payload = load_json(args.frozen_l2_policy)
    operational_payload = load_json(
        args.frozen_operational_policy
    )
    selected_validation_payload = load_json(
        args.selected_validation_policy
    )

    if preflight.get("status") != "PASS":
        raise RuntimeError("A4a-2B0 preflight is not PASS")
    if preflight.get("ready_to_build_a4a_2b_selector") is not True:
        raise RuntimeError("A4a-2B0 did not authorize selector build")
    if preflight_lock.get("status") != (
        "A4A_2B0_VALIDATION_SELECTION_PREFLIGHT_COMPLETE"
    ):
        raise RuntimeError("A4a-2B0 lock is incomplete")
    if preflight_lock.get("preflight_report_sha256") != hashes[
        "preflight_report_sha256"
    ]:
        raise RuntimeError("A4a-2B0 report hash differs from lock")

    if summary.get("status") != "PASS":
        raise RuntimeError("A4a-2A training summary is not PASS")
    if summary.get("development_test_accessed") is not False:
        raise RuntimeError("A4a-2A reports development-test access")
    if summary.get("validation_checkpoint_count") != EXPECTED_CHECKPOINT_COUNT:
        raise RuntimeError("unexpected validation checkpoint count")
    if len(manifest) != EXPECTED_CHECKPOINT_COUNT:
        raise RuntimeError("checkpoint manifest does not contain 12 entries")

    checkpoint_rows = []
    for row in manifest:
        path = args.training_dir / row["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != row["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint_rows.append(
            {
                "epoch": int(row["epoch"]),
                "path": path,
                "sha256": actual_sha,
                "bytes": int(row["bytes"]),
            }
        )
    checkpoint_rows.sort(key=lambda row: row["epoch"])

    selected_policy = extract_selected_policy(l2_policy_payload)
    stable_streak = int(
        require_unique_nested_value(
            selected_validation_payload,
            "stable_streak_decisions",
        )
    )
    deoverlap_stride = int(
        require_unique_nested_value(
            slot_contract,
            "deoverlap_stride_epochs",
        )
    )
    a3_h32_reference_exact = float(
        require_unique_nested_value(
            slot_contract,
            "a3_h32_reference_exact",
        )
    )

    validation_contract = full_contract[
        "checkpoint_and_validation_contract"
    ]
    gates = validation_contract["selection_hard_gates"]
    minimum_gain_pp = float(
        validation_contract[
            "minimum_meaningful_validation_exact_gain_percentage_points"
        ]
    )

    temporal_policy = {
        "policy_name": selected_policy.get(
            "policy_name",
            selected_policy.get("name", "P1_L2_CANDIDATE"),
        ),
        "mode": selected_policy["mode"],
        "horizon": int(selected_policy["horizon"]),
        "evidence_span_epochs": int(
            selected_policy.get(
                "evidence_span_epochs",
                int(selected_policy["horizon"])
                * deoverlap_stride,
            )
        ),
        "graph_method": selected_policy["graph_method"],
        "graph_threshold": float(
            selected_policy["graph_threshold"]
        ),
        "node_method": selected_policy.get(
            "node_method",
            "mean_logit",
        ),
        "stable_streak_decisions": stable_streak,
        "deoverlap_stride_epochs": deoverlap_stride,
    }

    if temporal_policy["mode"] != "deoverlap":
        raise RuntimeError(
            f"unsupported frozen mode: {temporal_policy['mode']}"
        )
    if temporal_policy["graph_method"] != "mean_logit":
        raise RuntimeError(
            "selector requires frozen mean_logit graph aggregation"
        )
    if temporal_policy["node_method"] != "mean_logit":
        raise RuntimeError(
            "selector requires frozen mean_logit node/slot aggregation"
        )
    if deoverlap_stride != EXPECTED_DEOVERLAP_STRIDE:
        raise RuntimeError(
            f"unexpected deoverlap stride: {deoverlap_stride}"
        )
    if temporal_policy["horizon"] != EXPECTED_H32_HORIZON:
        raise RuntimeError(
            "frozen L2 candidate is not H32: "
            f"horizon={temporal_policy['horizon']}"
        )
    if stable_streak != 4:
        raise RuntimeError(
            f"unexpected stable streak: {stable_streak}"
        )

    frozen_gates = {
        "attacker_set_precision_min": float(
            gates["attacker_set_precision_min"]
        ),
        "candidate_coverage_min": float(
            gates["candidate_coverage_min"]
        ),
        "persistent_normal_run_false_isolation_fraction": float(
            gates[
                "persistent_normal_run_false_isolation_fraction"
            ]
        ),
        "parameter_budget_pass": bool(
            gates["parameter_budget_pass"]
        ),
        "latency_budget_pass": bool(
            gates["latency_budget_pass"]
        ),
        "minimum_exact_improvement_percentage_points": (
            minimum_gain_pp
        ),
        "a3_h32_reference_exact": a3_h32_reference_exact,
        "minimum_required_attack_exact": (
            a3_h32_reference_exact + minimum_gain_pp / 100.0
        ),
    }

    return {
        "hashes": hashes,
        "checkpoint_rows": checkpoint_rows,
        "temporal_policy": temporal_policy,
        "frozen_gates": frozen_gates,
        "full_training_contract": full_contract,
        "slot_contract": slot_contract,
    }


def source_audit(
    args: argparse.Namespace,
    verified: dict[str, Any],
) -> int:
    if args.output_dir.exists():
        raise RuntimeError(
            f"source-audit output exists: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)

    temporal_policy = verified["temporal_policy"]
    horizon = int(temporal_policy["horizon"])
    stride = int(temporal_policy["deoverlap_stride_epochs"])
    deoverlap_count = len(
        np.arange(0, EXPECTED_SAMPLES_PER_RUN, stride)
    )
    decisions = deoverlap_count - horizon + 1

    synthetic_exact = np.array(
        [False, True, True, True, True, False, True],
        dtype=bool,
    )
    stable_mask = stable_point_mask(synthetic_exact, 4)
    stable_completion = first_stable_completion(
        synthetic_exact,
        4,
    )
    persistence_mask = np.array(
        [0, 1 << 3, 1 << 3, 1 << 3, 1 << 3, 0],
        dtype=np.uint16,
    )

    checks = {
        "checkpoint_count_is_12": (
            len(verified["checkpoint_rows"])
            == EXPECTED_CHECKPOINT_COUNT
        ),
        "checkpoint_epochs_unique": (
            len(
                {
                    row["epoch"]
                    for row in verified["checkpoint_rows"]
                }
            )
            == EXPECTED_CHECKPOINT_COUNT
        ),
        "h32_horizon": horizon == 32,
        "deoverlap_stride_8": stride == 8,
        "full_run_deoverlap_count_412": (
            deoverlap_count == 412
        ),
        "full_run_h32_decision_count_381": (
            decisions == EXPECTED_H32_DECISIONS_PER_FULL_RUN
        ),
        "stable_streak_is_4": (
            temporal_policy["stable_streak_decisions"] == 4
        ),
        "stable_point_mask_semantics": (
            stable_mask.tolist()
            == [False, True, True, True, True, False, False]
        ),
        "first_stable_completion_semantics": (
            stable_completion == 4
        ),
        "persistent_router_semantics": has_persistent_router(
            persistence_mask,
            4,
        ),
        "test_loader_constructed": False,
        "development_test_accessed": False,
    }

    pass_value = (
        all(
            checks[key] is True
            for key in (
                "checkpoint_count_is_12",
                "checkpoint_epochs_unique",
                "h32_horizon",
                "deoverlap_stride_8",
                "full_run_deoverlap_count_412",
                "full_run_h32_decision_count_381",
                "stable_streak_is_4",
                "stable_point_mask_semantics",
                "first_stable_completion_semantics",
                "persistent_router_semantics",
            )
        )
        and checks["test_loader_constructed"] is False
        and checks["development_test_accessed"] is False
    )
    if not pass_value:
        raise RuntimeError(f"source-audit checks failed: {checks}")

    audit = {
        "status": "PASS",
        "designation": (
            "V4-A4a-2B Validation Selector Source Audit"
        ),
        "temporal_policy": temporal_policy,
        "frozen_gates": verified["frozen_gates"],
        "checkpoint_inventory": [
            {
                "epoch": row["epoch"],
                "path": str(row["path"]),
                "sha256": row["sha256"],
                "bytes": row["bytes"],
            }
            for row in verified["checkpoint_rows"]
        ],
        "checks": checks,
        "dataset_samples_loaded": False,
        "model_inference_performed": False,
        "checkpoint_selection_performed": False,
        "threshold_search_performed": False,
        "validation_loader_constructed": False,
        "test_loader_constructed": False,
        "development_test_accessed": False,
        "ready_for_a4a_2b_validation_selection": True,
        "provenance": verified["hashes"],
    }
    audit_path = args.output_dir / "source_audit.json"
    atomic_json(audit_path, audit)

    lock = {
        "status": "A4A_2B_SELECTOR_SOURCE_AUDIT_COMPLETE",
        "source_audit_sha256": sha256_file(audit_path),
        "selector_script_sha256": verified["hashes"][
            "selector_script_sha256"
        ],
        "development_test_accessed": False,
    }
    atomic_json(
        args.output_dir / "A4A_2B_SOURCE_AUDIT_LOCK.json",
        lock,
    )
    (
        args.output_dir
        / "V4_A4A_2B_SELECTOR_SOURCE_AUDIT_PASS"
    ).write_text(
        "V4_A4A_2B_SELECTOR_SOURCE_AUDIT_PASS\n",
        encoding="utf-8",
    )

    print(json.dumps(audit, indent=2, sort_keys=True))
    print("V4_A4A_2B_SELECTOR_SOURCE_AUDIT_PASS")
    return 0


def validate_source_audit(
    source_audit_dir: Path,
    verified: dict[str, Any],
) -> None:
    marker = (
        source_audit_dir
        / "V4_A4A_2B_SELECTOR_SOURCE_AUDIT_PASS"
    )
    report_path = source_audit_dir / "source_audit.json"
    lock_path = source_audit_dir / "A4A_2B_SOURCE_AUDIT_LOCK.json"
    for path in (marker, report_path, lock_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"missing A4a-2B source-audit artifact: {path}"
            )
    report = load_json(report_path)
    lock = load_json(lock_path)
    if report.get("status") != "PASS":
        raise RuntimeError("A4a-2B source audit is not PASS")
    if report.get(
        "ready_for_a4a_2b_validation_selection"
    ) is not True:
        raise RuntimeError(
            "A4a-2B source audit does not authorize selection"
        )
    if lock.get("status") != (
        "A4A_2B_SELECTOR_SOURCE_AUDIT_COMPLETE"
    ):
        raise RuntimeError("A4a-2B source-audit lock is incomplete")
    if lock.get("source_audit_sha256") != sha256_file(
        report_path
    ):
        raise RuntimeError(
            "A4a-2B source-audit report hash differs from lock"
        )
    if lock.get("selector_script_sha256") != verified["hashes"][
        "selector_script_sha256"
    ]:
        raise RuntimeError(
            "selector script changed after source audit"
        )


def import_project_modules(args: argparse.Namespace):
    for root in (
        args.repo,
        args.repo / "scripts",
        args.original_trainer.parent,
        args.model_source.parent,
        args.slot_module.parent,
    ):
        text = str(root.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)

    trainer_module = load_module(
        args.original_trainer,
        "resolved_v4_a3_trainer_for_a4a_2b",
    )
    model_module = load_module(
        args.model_source,
        "resolved_v4_a3_model_for_a4a_2b",
    )
    return trainer_module, model_module


def make_model(
    model_module,
    base_checkpoint: dict[str, Any],
    device: torch.device,
) -> A4aFrozenSlotModel:
    base_model = model_module.A3SourcePreserveModel()
    base_model.load_state_dict(
        extract_state_dict(base_checkpoint),
        strict=True,
    )
    base_model.eval()
    model = A4aFrozenSlotModel(base_model).to(device)
    parameters = count_parameters(model)
    if parameters["total"] != EXPECTED_TOTAL_PARAMETERS:
        raise RuntimeError(
            f"unexpected total parameters: {parameters}"
        )
    if parameters["trainable"] != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"unexpected trainable parameters: {parameters}"
        )
    return model


def infer_validation_logits(
    *,
    model: A4aFrozenSlotModel,
    checkpoint_path: Path,
    loader,
    adjacency: torch.Tensor,
    physical_mask: torch.Tensor,
    validation_count: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(
        extract_state_dict(checkpoint),
        strict=True,
    )
    model.eval()

    graph_logits = np.empty(validation_count, dtype=np.float32)
    slot_logits = np.empty(
        (validation_count, 4, NUM_ROUTERS + 1),
        dtype=np.float32,
    )

    cursor = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_number, batch in enumerate(loader, start=1):
            x = batch["x"].to(
                device,
                non_blocking=True,
                dtype=torch.float32,
            )
            output = model(x, adjacency, physical_mask)
            batch_graph = (
                output["graph_logits"]
                .detach()
                .reshape(-1)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            batch_slot = (
                output["slot_logits"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            stop = cursor + int(x.shape[0])
            graph_logits[cursor:stop] = batch_graph
            slot_logits[cursor:stop] = batch_slot
            cursor = stop

            if batch_number == 1 or batch_number % 250 == 0:
                print(
                    f"  inference_batch={batch_number} "
                    f"samples={cursor}/{validation_count}",
                    flush=True,
                )

    if cursor != validation_count:
        raise RuntimeError(
            f"inference produced {cursor} samples, expected "
            f"{validation_count}"
        )
    return graph_logits, slot_logits, time.perf_counter() - started


def build_decision_trace(
    *,
    raw_graph_logits: np.ndarray,
    raw_slot_logits: np.ndarray,
    raw_run_index: np.ndarray,
    raw_end_epoch: np.ndarray,
    raw_true_masks: np.ndarray,
    temporal_policy: dict[str, Any],
) -> dict[str, np.ndarray]:
    horizon = int(temporal_policy["horizon"])
    stride = int(temporal_policy["deoverlap_stride_epochs"])
    graph_threshold = float(
        temporal_policy["graph_threshold"]
    )

    output_run = []
    output_ordinal = []
    output_end = []
    output_truth = []
    output_prediction = []
    output_graph_probability = []

    unique_runs, starts = np.unique(
        raw_run_index,
        return_index=True,
    )
    order = np.argsort(starts)
    unique_runs = unique_runs[order]

    for run in unique_runs:
        positions = np.flatnonzero(raw_run_index == run)
        run_truth = raw_true_masks[positions]
        if not np.all(run_truth == run_truth[0]):
            raise RuntimeError(
                f"truth labels vary within run {int(run)}"
            )

        aggregate_graph, aggregate_slot, decision_end = aggregate_run(
            raw_graph_logits[positions],
            raw_slot_logits[positions],
            raw_end_epoch[positions],
            deoverlap_stride=stride,
            horizon=horizon,
        )
        if aggregate_graph.shape[0] == 0:
            continue

        graph_probability = stable_sigmoid(aggregate_graph)
        predicted_masks = decode_aggregated_slots(
            aggregate_slot
        )
        predicted_masks = np.where(
            graph_probability >= graph_threshold,
            predicted_masks,
            np.uint16(0),
        ).astype(np.uint16, copy=False)

        count = len(predicted_masks)
        output_run.append(
            np.full(count, int(run), dtype=np.int32)
        )
        output_ordinal.append(
            np.arange(count, dtype=np.int32)
        )
        output_end.append(decision_end)
        output_truth.append(
            np.full(
                count,
                int(run_truth[0]),
                dtype=np.uint16,
            )
        )
        output_prediction.append(predicted_masks)
        output_graph_probability.append(graph_probability)

    return {
        "run_index": np.concatenate(output_run),
        "decision_ordinal": np.concatenate(output_ordinal),
        "decision_end_epoch": np.concatenate(output_end),
        "true_attacker_mask": np.concatenate(output_truth),
        "predicted_attacker_mask": np.concatenate(
            output_prediction
        ),
        "graph_probability": np.concatenate(
            output_graph_probability
        ),
    }


def apply_frozen_gates(
    summary: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    exact_gain_pp = 100.0 * (
        float(summary["attack_exact_localization"])
        - float(gates["a3_h32_reference_exact"])
    )
    checks = {
        "attacker_set_precision": (
            float(summary["attack_set_precision"])
            >= float(gates["attacker_set_precision_min"])
        ),
        "candidate_coverage": (
            float(summary["candidate_coverage"])
            >= float(gates["candidate_coverage_min"])
        ),
        "persistent_normal_run_false_isolation": (
            float(
                summary[
                    "normal_run_persistent_false_isolation_fraction"
                ]
            )
            <= float(
                gates[
                    "persistent_normal_run_false_isolation_fraction"
                ]
            )
        ),
        "minimum_exact_improvement": (
            exact_gain_pp
            >= float(
                gates[
                    "minimum_exact_improvement_percentage_points"
                ]
            )
        ),
        "parameter_budget": bool(gates["parameter_budget_pass"]),
        "latency_budget": bool(gates["latency_budget_pass"]),
    }
    return {
        "exact_gain_over_a3_h32_percentage_points": exact_gain_pp,
        "checks": checks,
        "all_gates_pass": all(checks.values()),
    }


def checkpoint_rank(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(summary["stable_attack_exact_point_rate"]),
        float(summary["attack_exact_localization"]),
        -float(summary["normal_false_isolation_rate"]),
        float(summary["attack_set_precision"]),
        float(summary["candidate_coverage"]),
        -float(summary["training_validation_total_loss"]),
        -int(summary["checkpoint_epoch"]),
    )


def diagnostic_rank(summary: dict[str, Any]) -> tuple[Any, ...]:
    gates = summary["gate_evaluation"]["checks"]
    return (
        sum(bool(value) for value in gates.values()),
        *checkpoint_rank(summary),
    )


def selection(args: argparse.Namespace, verified: dict[str, Any]) -> int:
    if args.source_audit_dir is None:
        raise ValueError(
            "--source-audit-dir is required for select mode"
        )
    validate_source_audit(args.source_audit_dir, verified)

    if args.output_dir.exists():
        raise RuntimeError(
            f"selection output exists: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)
    traces_dir = args.output_dir / "decision_traces"
    traces_dir.mkdir()
    per_run_dir = args.output_dir / "per_run"
    per_run_dir.mkdir()

    trainer_module, model_module = import_project_modules(args)

    metadata = trainer_module.load_metadata(args.data_dir)
    splits = trainer_module.load_split_indices(
        args.data_dir,
        metadata,
    )
    validation_indices = np.sort(
        np.asarray(splits["val"], dtype=np.int64)
    )
    validation_count = int(validation_indices.size)

    dataset = trainer_module.V4A3MemmapDataset(
        args.data_dir,
        validation_indices,
    )
    loader = trainer_module.build_loader(
        dataset,
        args.batch_size,
        False,
        args.num_workers,
        args.pin_memory,
        args.persistent_workers,
        args.prefetch_factor,
        7,
    )

    raw_run_index = np.asarray(
        np.load(
            args.data_dir / "run_index.npy",
            mmap_mode="r",
        )[validation_indices],
        dtype=np.int32,
    )
    raw_end_epoch = np.asarray(
        np.load(
            args.data_dir / "end_epoch.npy",
            mmap_mode="r",
        )[validation_indices],
        dtype=np.int32,
    )
    raw_y_node = np.asarray(
        np.load(
            args.data_dir / "y_node.npy",
            mmap_mode="r",
        )[validation_indices],
        dtype=bool,
    )
    raw_true_masks = bitmask_from_truth(raw_y_node)
    del raw_y_node

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        raise RuntimeError(
            "A4a-2B full validation selection requires CUDA"
        )

    base_checkpoint = torch.load(
        args.base_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = make_model(
        model_module,
        base_checkpoint,
        device,
    )
    adjacency = base_checkpoint["A_hat"].to(
        device=device,
        dtype=torch.float32,
    )
    physical_mask = base_checkpoint[
        "physical_valid_port_mask"
    ].to(
        device=device,
        dtype=torch.float32,
    )

    temporal_policy = verified["temporal_policy"]
    stable_streak = int(
        temporal_policy["stable_streak_decisions"]
    )
    gates = verified["frozen_gates"]

    checkpoint_summaries: list[dict[str, Any]] = []
    all_run_rows: list[dict[str, Any]] = []

    for position, checkpoint_row in enumerate(
        verified["checkpoint_rows"],
        start=1,
    ):
        epoch = int(checkpoint_row["epoch"])
        print(
            f"===== CHECKPOINT {position}/{EXPECTED_CHECKPOINT_COUNT} "
            f"EPOCH {epoch} =====",
            flush=True,
        )
        checkpoint = torch.load(
            checkpoint_row["path"],
            map_location="cpu",
            weights_only=False,
        )
        validation_loss = float(
            checkpoint["validation_losses"]["total"]
        )

        raw_graph_logits, raw_slot_logits, elapsed = (
            infer_validation_logits(
                model=model,
                checkpoint_path=checkpoint_row["path"],
                loader=loader,
                adjacency=adjacency,
                physical_mask=physical_mask,
                validation_count=validation_count,
                device=device,
            )
        )
        decision_trace = build_decision_trace(
            raw_graph_logits=raw_graph_logits,
            raw_slot_logits=raw_slot_logits,
            raw_run_index=raw_run_index,
            raw_end_epoch=raw_end_epoch,
            raw_true_masks=raw_true_masks,
            temporal_policy=temporal_policy,
        )
        del raw_graph_logits
        del raw_slot_logits

        summary, run_rows = evaluate_decision_trace(
            checkpoint_epoch=epoch,
            checkpoint_sha256=checkpoint_row["sha256"],
            validation_loss=validation_loss,
            run_index=decision_trace["run_index"],
            decision_ordinal=decision_trace[
                "decision_ordinal"
            ],
            decision_end_epoch=decision_trace[
                "decision_end_epoch"
            ],
            true_masks=decision_trace[
                "true_attacker_mask"
            ],
            predicted_masks=decision_trace[
                "predicted_attacker_mask"
            ],
            graph_probabilities=decision_trace[
                "graph_probability"
            ],
            stable_streak=stable_streak,
        )
        summary["inference_seconds"] = elapsed
        summary["gate_evaluation"] = apply_frozen_gates(
            summary,
            gates,
        )
        checkpoint_summaries.append(summary)
        all_run_rows.extend(run_rows)

        trace_path = (
            traces_dir / f"epoch_{epoch:03d}_validation_decisions.npz"
        )
        np.savez_compressed(
            trace_path,
            **decision_trace,
            checkpoint_epoch=np.asarray(epoch, dtype=np.int16),
            checkpoint_sha256=np.asarray(
                checkpoint_row["sha256"]
            ),
            graph_threshold=np.asarray(
                temporal_policy["graph_threshold"],
                dtype=np.float32,
            ),
            horizon=np.asarray(
                temporal_policy["horizon"],
                dtype=np.int16,
            ),
            deoverlap_stride=np.asarray(
                temporal_policy[
                    "deoverlap_stride_epochs"
                ],
                dtype=np.int16,
            ),
            stable_streak=np.asarray(
                stable_streak,
                dtype=np.int16,
            ),
        )
        summary["decision_trace_path"] = str(trace_path)
        summary["decision_trace_sha256"] = sha256_file(trace_path)

        per_run_path = (
            per_run_dir / f"epoch_{epoch:03d}_per_run.csv"
        )
        atomic_csv(per_run_path, run_rows)
        summary["per_run_metrics_path"] = str(per_run_path)
        summary["per_run_metrics_sha256"] = sha256_file(
            per_run_path
        )

        print(
            f"epoch={epoch:03d} "
            f"attack_exact={summary['attack_exact_localization']:.6f} "
            f"stable_exact={summary['stable_attack_exact_point_rate']:.6f} "
            f"precision={summary['attack_set_precision']:.6f} "
            f"coverage={summary['candidate_coverage']:.6f} "
            f"normal_persistent="
            f"{summary['normal_run_persistent_false_isolation_fraction']:.6f} "
            f"gain_pp="
            f"{summary['gate_evaluation']['exact_gain_over_a3_h32_percentage_points']:.3f} "
            f"all_gates={summary['gate_evaluation']['all_gates_pass']}",
            flush=True,
        )

    eligible = [
        row
        for row in checkpoint_summaries
        if row["gate_evaluation"]["all_gates_pass"]
    ]
    diagnostic_best = max(
        checkpoint_summaries,
        key=diagnostic_rank,
    )
    selected = max(eligible, key=checkpoint_rank) if eligible else None

    selection_status = (
        "CHECKPOINT_SELECTED"
        if selected is not None
        else "HOLD_NO_CHECKPOINT_PASSES_ALL_FROZEN_GATES"
    )

    frozen_checkpoint_path = None
    frozen_checkpoint_sha = None
    frozen_trace_path = None
    frozen_trace_sha = None

    if selected is not None:
        selected_checkpoint_row = next(
            row
            for row in verified["checkpoint_rows"]
            if row["epoch"] == selected["checkpoint_epoch"]
        )
        frozen_checkpoint_path = (
            args.output_dir / "frozen_selected_checkpoint.pt"
        )
        shutil.copy2(
            selected_checkpoint_row["path"],
            frozen_checkpoint_path,
        )
        frozen_checkpoint_sha = sha256_file(
            frozen_checkpoint_path
        )
        if frozen_checkpoint_sha != selected_checkpoint_row["sha256"]:
            raise RuntimeError(
                "frozen selected checkpoint hash changed during copy"
            )

        source_trace = Path(selected["decision_trace_path"])
        frozen_trace_path = (
            args.output_dir
            / "frozen_selected_validation_decisions.npz"
        )
        shutil.copy2(source_trace, frozen_trace_path)
        frozen_trace_sha = sha256_file(frozen_trace_path)
        if frozen_trace_sha != selected["decision_trace_sha256"]:
            raise RuntimeError(
                "frozen validation decision trace hash changed"
            )

    selection_report = {
        "status": "PASS" if selected is not None else "HOLD",
        "designation": (
            "V4-A4a-2B Validation-Only Exact-Decoder "
            "Checkpoint Selection"
        ),
        "selection_status": selection_status,
        "selection_split": "validation",
        "temporal_policy": temporal_policy,
        "frozen_gates": gates,
        "selection_rule": (
            "among checkpoints passing every frozen gate: maximize "
            "stable attack exact point rate, then attack exact "
            "localization, then minimize normal false isolation rate, "
            "then maximize precision and coverage, then minimize "
            "training validation loss, then prefer the earlier epoch"
        ),
        "checkpoint_count_evaluated": len(checkpoint_summaries),
        "checkpoint_summaries": checkpoint_summaries,
        "eligible_checkpoint_epochs": [
            row["checkpoint_epoch"] for row in eligible
        ],
        "selected_checkpoint": selected,
        "selected_checkpoint_source_path": (
            str(
                next(
                    row["path"]
                    for row in verified["checkpoint_rows"]
                    if (
                        selected is not None
                        and row["epoch"]
                        == selected["checkpoint_epoch"]
                    )
                )
            )
            if selected is not None
            else None
        ),
        "frozen_selected_checkpoint_path": (
            str(frozen_checkpoint_path)
            if frozen_checkpoint_path is not None
            else None
        ),
        "frozen_selected_checkpoint_sha256": (
            frozen_checkpoint_sha
        ),
        "frozen_selected_validation_decisions_path": (
            str(frozen_trace_path)
            if frozen_trace_path is not None
            else None
        ),
        "frozen_selected_validation_decisions_sha256": (
            frozen_trace_sha
        ),
        "diagnostic_best_checkpoint": diagnostic_best,
        "development_test_transfer_authorized": (
            selected is not None
        ),
        "parameter_changes_after_selection_allowed": False,
        "threshold_changes_after_selection_allowed": False,
        "validation_selection_performed": True,
        "threshold_search_performed": False,
        "test_loader_constructed": False,
        "test_evaluated": False,
        "development_test_accessed": False,
        "provenance": verified["hashes"],
    }

    report_path = args.output_dir / "selection_report.json"
    atomic_json(report_path, selection_report)

    csv_rows = []
    for row in checkpoint_summaries:
        flat = {
            key: value
            for key, value in row.items()
            if isinstance(value, (str, int, float, bool))
            or value is None
        }
        flat.update(
            {
                f"gate_{key}": value
                for key, value in row["gate_evaluation"][
                    "checks"
                ].items()
            }
        )
        flat["all_gates_pass"] = row["gate_evaluation"][
            "all_gates_pass"
        ]
        flat["exact_gain_over_a3_h32_percentage_points"] = (
            row["gate_evaluation"][
                "exact_gain_over_a3_h32_percentage_points"
            ]
        )
        csv_rows.append(flat)
    atomic_csv(
        args.output_dir / "checkpoint_selection_summary.csv",
        csv_rows,
    )
    atomic_csv(
        args.output_dir / "all_checkpoint_run_metrics.csv",
        all_run_rows,
    )

    lock = {
        "status": (
            "A4A_2B_VALIDATION_SELECTION_COMPLETE"
            if selected is not None
            else "A4A_2B_VALIDATION_SELECTION_HOLD"
        ),
        "selection_report_sha256": sha256_file(report_path),
        "selected_checkpoint_epoch": (
            selected["checkpoint_epoch"]
            if selected is not None
            else None
        ),
        "frozen_selected_checkpoint_sha256": (
            frozen_checkpoint_sha
        ),
        "frozen_selected_validation_decisions_sha256": (
            frozen_trace_sha
        ),
        "checkpoint_count_evaluated": len(
            checkpoint_summaries
        ),
        "threshold_search_performed": False,
        "test_loader_constructed": False,
        "development_test_accessed": False,
    }
    atomic_json(args.output_dir / "A4A_2B_LOCK.json", lock)

    if selected is not None:
        marker = "V4_A4A_2B_VALIDATION_SELECTION_PASS"
        return_code = 0
    else:
        marker = "V4_A4A_2B_VALIDATION_SELECTION_HOLD"
        return_code = 3

    (args.output_dir / marker).write_text(
        marker + "\n",
        encoding="utf-8",
    )

    print(json.dumps(selection_report, indent=2, sort_keys=True))
    print(marker)
    return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("source-audit", "select"),
        required=True,
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--primary-trainer", type=Path, required=True)
    parser.add_argument("--original-trainer", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--slot-module", type=Path, required=True)
    parser.add_argument("--preflight-script", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument(
        "--full-training-contract",
        type=Path,
        required=True,
    )
    parser.add_argument("--slot-contract", type=Path, required=True)
    parser.add_argument("--frozen-l2-policy", type=Path, required=True)
    parser.add_argument(
        "--frozen-operational-policy",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--selected-validation-policy",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-audit-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (
        args.num_workers == 0
        and args.persistent_workers
    ):
        raise ValueError(
            "persistent workers require num_workers > 0"
        )
    verified = verify_paths_and_provenance(args)
    if args.mode == "source-audit":
        if args.source_audit_dir is not None:
            raise ValueError(
                "--source-audit-dir is invalid in source-audit mode"
            )
        return source_audit(args, verified)
    return selection(args, verified)


if __name__ == "__main__":
    raise SystemExit(main())
