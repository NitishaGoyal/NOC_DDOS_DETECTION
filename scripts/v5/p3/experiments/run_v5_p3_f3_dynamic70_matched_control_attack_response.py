from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STAGE = (
    "V5_P3_F3_DYNAMIC70_TRAINING_ONLY_MATCHED_"
    "CONTROL_ATTACK_RESPONSE_ANALYSIS"
)
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
ALIGNMENT_COVERAGE_GATE = 0.99
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 607
EARLY_FRACTION = 0.25
ZERO_TOLERANCE = 1e-12
PROGRESS_INTERVAL = 2500

ROLE_NAMES = (
    "whole_graph",
    "source",
    "transit",
    "victim",
    "path_any",
    "path_non_endpoint",
    "non_path",
)
PHASE_NAMES = (
    "all_active",
    "early_quartile",
    "sustained_remaining",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def import_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def scalarize(value: Any) -> Any | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, bool, int, float, np.integer, np.floating)):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    try:
        array = tensor_to_numpy(value)
    except Exception:
        return None
    if array.size != 1:
        return None
    result = array.reshape(-1)[0]
    if isinstance(result, bytes):
        return result.decode("utf-8", errors="replace")
    if isinstance(result, np.generic):
        result = result.item()
    if isinstance(result, float) and not math.isfinite(result):
        return str(result)
    if isinstance(result, (str, bool, int, float)):
        return result
    return str(result)


def extract_first(sample: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in sample:
            return sample[key]
    raise KeyError(f"none of the required keys exist: {keys}")


def extract_scalar_with_key(
    sample: dict[str, Any],
    keys: tuple[str, ...],
) -> tuple[str, Any]:
    for key in keys:
        if key in sample:
            value = scalarize(sample[key])
            if value is not None:
                return key, value
    raise KeyError(f"none of the required scalar keys exist: {keys}")


def extract_pair_id(sample: dict[str, Any]) -> tuple[str, str]:
    key, value = extract_scalar_with_key(
        sample,
        (
            "pair_id",
            "matched_pair_id",
            "pair_uid",
            "pair_index",
            "pair_key",
        ),
    )
    return key, str(value)


def extract_attack_label(sample: dict[str, Any]) -> tuple[str, int]:
    key, value = extract_scalar_with_key(
        sample,
        (
            "y_attack",
            "y_graph",
            "attack_label",
            "graph_label",
        ),
    )
    return key, int(value)


def extract_count_label(sample: dict[str, Any]) -> tuple[str, int]:
    for key in (
        "y_attacker_count",
        "attacker_count",
        "y_count",
        "count_label",
    ):
        if key in sample:
            value = scalarize(sample[key])
            if value is not None:
                return key, int(value)
    return "", 0


def extract_bitmap(
    sample: dict[str, Any],
    keys: tuple[str, ...],
) -> np.ndarray:
    value = tensor_to_numpy(extract_first(sample, keys))
    value = np.asarray(value).reshape(-1)
    require(value.size == 16, f"expected 16-label bitmap for {keys}")
    return (value != 0).astype(bool)


def physical_mask_index(channel_name: str) -> int:
    direction_to_index = {
        "local": 0,
        "north": 1,
        "east": 2,
        "south": 3,
        "west": 4,
    }
    direction = channel_name.rsplit("_", 1)[-1]
    require(
        direction in direction_to_index,
        f"unknown direction in channel {channel_name}",
    )
    base = direction_to_index[direction]
    return 5 + base if channel_name.startswith("out_") else base


def canonical_scalar_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def sorted_unique(values: Iterable[Any]) -> list[Any]:
    unique = {}
    for value in values:
        unique[canonical_scalar_key(value)] = value
    return [unique[key] for key in sorted(unique)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def discover_run_candidates(
    pair_to_indices: dict[str, np.ndarray],
    attack_labels: np.ndarray,
    metadata: dict[str, list[Any]],
    excluded_keys: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for key, values in metadata.items():
        if key in excluded_keys:
            continue
        valid_pairs = 0
        pairs_with_two_values = 0
        attack_value_unique_pairs = 0
        global_unique = set()

        for pair_indices in pair_to_indices.values():
            pair_values = [
                values[int(index)]
                for index in pair_indices
                if values[int(index)] is not None
            ]
            unique_values = sorted_unique(pair_values)
            for value in unique_values:
                global_unique.add(canonical_scalar_key(value))
            if len(unique_values) != 2:
                continue
            pairs_with_two_values += 1

            positive_values = set()
            for index in pair_indices:
                index = int(index)
                value = values[index]
                if value is None:
                    continue
                if attack_labels[index] == 1:
                    positive_values.add(canonical_scalar_key(value))
            if len(positive_values) != 1:
                continue
            attack_value_unique_pairs += 1
            attack_value_key = next(iter(positive_values))
            other_values = {
                canonical_scalar_key(value)
                for value in unique_values
                if canonical_scalar_key(value) != attack_value_key
            }
            if len(other_values) != 1:
                continue

            attack_value_has_all_positives = True
            control_has_no_positives = True
            for index in pair_indices:
                index = int(index)
                value_key = canonical_scalar_key(values[index])
                if attack_labels[index] == 1 and value_key != attack_value_key:
                    attack_value_has_all_positives = False
                if attack_labels[index] == 1 and value_key in other_values:
                    control_has_no_positives = False
            if attack_value_has_all_positives and control_has_no_positives:
                valid_pairs += 1

        rows.append({
            "key": key,
            "pair_count": len(pair_to_indices),
            "pairs_with_two_values": pairs_with_two_values,
            "attack_value_unique_pairs": attack_value_unique_pairs,
            "valid_run_partition_pairs": valid_pairs,
            "valid_pair_fraction": (
                valid_pairs / len(pair_to_indices)
                if pair_to_indices else 0.0
            ),
            "global_unique_count": len(global_unique),
            "name_bonus": int(
                any(token in key.lower() for token in (
                    "run", "member", "role", "control", "attack", "trace"
                ))
            ),
        })

    rows.sort(
        key=lambda row: (
            row["valid_run_partition_pairs"],
            row["name_bonus"],
            -row["global_unique_count"],
        ),
        reverse=True,
    )
    return rows


def classify_pair_runs(
    pair_indices: np.ndarray,
    values: list[Any],
    attack_labels: np.ndarray,
) -> tuple[Any, Any] | None:
    unique_values = sorted_unique(
        values[int(index)] for index in pair_indices
    )
    if len(unique_values) != 2:
        return None
    positive_values = {
        canonical_scalar_key(values[int(index)])
        for index in pair_indices
        if attack_labels[int(index)] == 1
    }
    if len(positive_values) != 1:
        return None
    attack_key = next(iter(positive_values))
    attack_values = [
        value for value in unique_values
        if canonical_scalar_key(value) == attack_key
    ]
    control_values = [
        value for value in unique_values
        if canonical_scalar_key(value) != attack_key
    ]
    if len(attack_values) != 1 or len(control_values) != 1:
        return None
    return control_values[0], attack_values[0]


def evaluate_time_candidate(
    key: str,
    values: list[Any],
    pair_to_indices: dict[str, np.ndarray],
    run_values: list[Any],
    attack_labels: np.ndarray,
) -> dict[str, Any]:
    pair_coverage = 0
    matched_all = 0
    matched_active = 0
    duplicate_time_pairs = 0
    active_total = int(np.sum(attack_labels == 1))

    for pair_indices in pair_to_indices.values():
        classified = classify_pair_runs(
            pair_indices,
            run_values,
            attack_labels,
        )
        if classified is None:
            continue
        control_value, attack_value = classified
        control_map: dict[tuple[str, str], int] = {}
        attack_map: dict[tuple[str, str], int] = {}
        duplicate = False

        for index in pair_indices:
            index = int(index)
            time_value = values[index]
            if time_value is None:
                continue
            time_key = canonical_scalar_key(time_value)
            run_key = canonical_scalar_key(run_values[index])
            if run_key == canonical_scalar_key(control_value):
                if time_key in control_map:
                    duplicate = True
                control_map[time_key] = index
            elif run_key == canonical_scalar_key(attack_value):
                if time_key in attack_map:
                    duplicate = True
                attack_map[time_key] = index

        if duplicate:
            duplicate_time_pairs += 1
            continue
        common = set(control_map) & set(attack_map)
        if common:
            pair_coverage += 1
        matched_all += len(common)
        matched_active += sum(
            attack_labels[attack_map[time_key]] == 1
            for time_key in common
        )

    return {
        "time_key": key,
        "pair_coverage": pair_coverage,
        "pair_coverage_fraction": (
            pair_coverage / len(pair_to_indices)
            if pair_to_indices else 0.0
        ),
        "matched_all_windows": matched_all,
        "matched_active_windows": matched_active,
        "active_alignment_fraction": (
            matched_active / active_total if active_total else 0.0
        ),
        "duplicate_time_pairs": duplicate_time_pairs,
        "mode": "metadata_key",
    }


def build_ordinal_alignment(
    pair_to_indices: dict[str, np.ndarray],
    run_values: list[Any],
    attack_labels: np.ndarray,
) -> dict[str, Any]:
    matches: dict[str, list[tuple[int, int, int]]] = {}
    matched_all = 0
    matched_active = 0
    equal_length_pairs = 0
    pair_coverage = 0

    for pair_id, pair_indices in pair_to_indices.items():
        classified = classify_pair_runs(
            pair_indices,
            run_values,
            attack_labels,
        )
        if classified is None:
            continue
        control_value, attack_value = classified
        control_indices = sorted(
            int(index)
            for index in pair_indices
            if canonical_scalar_key(run_values[int(index)])
            == canonical_scalar_key(control_value)
        )
        attack_indices = sorted(
            int(index)
            for index in pair_indices
            if canonical_scalar_key(run_values[int(index)])
            == canonical_scalar_key(attack_value)
        )
        if len(control_indices) == len(attack_indices):
            equal_length_pairs += 1
        count = min(len(control_indices), len(attack_indices))
        pair_matches = [
            (control_indices[position], attack_indices[position], position)
            for position in range(count)
        ]
        matches[pair_id] = pair_matches
        if pair_matches:
            pair_coverage += 1
        matched_all += len(pair_matches)
        matched_active += sum(
            attack_labels[attack_index] == 1
            for _, attack_index, _ in pair_matches
        )

    active_total = int(np.sum(attack_labels == 1))
    return {
        "matches": matches,
        "summary": {
            "time_key": "__within_run_ordinal__",
            "pair_coverage": pair_coverage,
            "pair_coverage_fraction": (
                pair_coverage / len(pair_to_indices)
                if pair_to_indices else 0.0
            ),
            "matched_all_windows": matched_all,
            "matched_active_windows": matched_active,
            "active_alignment_fraction": (
                matched_active / active_total if active_total else 0.0
            ),
            "duplicate_time_pairs": 0,
            "equal_length_pairs": equal_length_pairs,
            "mode": "within_run_ordinal_fallback",
        },
    }


def build_metadata_alignment(
    pair_to_indices: dict[str, np.ndarray],
    run_values: list[Any],
    time_values: list[Any],
    attack_labels: np.ndarray,
) -> dict[str, list[tuple[int, int, Any]]]:
    matches: dict[str, list[tuple[int, int, Any]]] = {}
    for pair_id, pair_indices in pair_to_indices.items():
        classified = classify_pair_runs(
            pair_indices,
            run_values,
            attack_labels,
        )
        if classified is None:
            continue
        control_value, attack_value = classified
        control_map = {}
        attack_map = {}
        for index in pair_indices:
            index = int(index)
            value = time_values[index]
            if value is None:
                continue
            time_key = canonical_scalar_key(value)
            run_key = canonical_scalar_key(run_values[index])
            if run_key == canonical_scalar_key(control_value):
                require(
                    time_key not in control_map,
                    f"duplicate control time key in pair {pair_id}",
                )
                control_map[time_key] = (index, value)
            elif run_key == canonical_scalar_key(attack_value):
                require(
                    time_key not in attack_map,
                    f"duplicate attack time key in pair {pair_id}",
                )
                attack_map[time_key] = (index, value)

        common_keys = sorted(set(control_map) & set(attack_map))
        matches[pair_id] = [
            (
                control_map[key][0],
                attack_map[key][0],
                attack_map[key][1],
            )
            for key in common_keys
        ]
    return matches


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    repeats: int = BOOTSTRAP_REPEATS,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    chunk = 100
    means = []
    for start in range(0, repeats, chunk):
        current = min(chunk, repeats - start)
        indices = rng.integers(
            0,
            values.size,
            size=(current, values.size),
        )
        means.append(np.mean(values[indices], axis=1))
    distribution = np.concatenate(means)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(low), float(high)


def summarize_pair_values(
    signed: np.ndarray,
    absolute: np.ndarray,
    seed: int,
    bootstrap: bool,
) -> dict[str, Any]:
    signed = np.asarray(signed, dtype=np.float64)
    absolute = np.asarray(absolute, dtype=np.float64)
    valid = np.isfinite(signed) & np.isfinite(absolute)
    signed = signed[valid]
    absolute = absolute[valid]
    if signed.size == 0:
        return {
            "pair_count": 0,
            "mean_signed_effect": float("nan"),
            "median_signed_effect": float("nan"),
            "median_absolute_effect": float("nan"),
            "std_signed_effect": float("nan"),
            "paired_standardized_effect_dz": float("nan"),
            "positive_pair_fraction": float("nan"),
            "negative_pair_fraction": float("nan"),
            "zero_pair_fraction": float("nan"),
            "bootstrap_mean_ci_low": float("nan"),
            "bootstrap_mean_ci_high": float("nan"),
        }

    mean = float(np.mean(signed))
    std = float(np.std(signed, ddof=1)) if signed.size > 1 else 0.0
    if std > ZERO_TOLERANCE:
        dz = mean / std
    elif abs(mean) <= ZERO_TOLERANCE:
        dz = 0.0
    else:
        dz = float("nan")

    if bootstrap:
        ci_low, ci_high = bootstrap_mean_ci(signed, seed)
    else:
        ci_low = float("nan")
        ci_high = float("nan")

    return {
        "pair_count": int(signed.size),
        "mean_signed_effect": mean,
        "median_signed_effect": float(np.median(signed)),
        "median_absolute_effect": float(np.median(absolute)),
        "std_signed_effect": std,
        "paired_standardized_effect_dz": float(dz),
        "positive_pair_fraction": float(
            np.mean(signed > ZERO_TOLERANCE)
        ),
        "negative_pair_fraction": float(
            np.mean(signed < -ZERO_TOLERANCE)
        ),
        "zero_pair_fraction": float(
            np.mean(np.abs(signed) <= ZERO_TOLERANCE)
        ),
        "bootstrap_mean_ci_low": ci_low,
        "bootstrap_mean_ci_high": ci_high,
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")
    require(data_link.is_symlink(), f"dataset symlink missing: {data_link}")
    data_root = data_link.resolve()
    require(data_root.is_dir(), f"dataset target missing: {data_root}")

    study_dir = repo / "reports/v5/p3_experiments/f0_d70_feature_study"
    audit_dir = study_dir / "schema_audit"
    paired_dir = study_dir / "paired_analysis"
    paired_dir.mkdir(parents=True, exist_ok=True)

    f0_lock_path = study_dir / "V5_P3_F0_D70_FEATURE_STUDY_LOCK.json"
    group_contract_path = study_dir / "FEATURE_GROUP_CONTRACT.json"
    metric_contract_path = study_dir / "METRIC_CONTRACT.json"
    data_access_path = study_dir / "DATA_ACCESS_CONTRACT.json"
    f2_report_path = audit_dir / (
        "V5_P3_F2_DYNAMIC70_TRAINING_ONLY_DEGENERACY_"
        "AND_OBSERVABILITY_AUDIT_REPORT.json"
    )
    f2_lock_path = audit_dir / (
        "V5_P3_F2_DYNAMIC70_TRAINING_ONLY_DEGENERACY_"
        "AND_OBSERVABILITY_AUDIT_LOCK.json"
    )
    f2r_report_path = audit_dir / (
        "V5_P3_F2R_DYNAMIC70_DEGENERACY_FINDING_REVIEW_REPORT.json"
    )
    f2r_lock_path = audit_dir / (
        "V5_P3_F2R_DYNAMIC70_DEGENERACY_FINDING_REVIEW_LOCK.json"
    )
    f2r_amendment_path = audit_dir / (
        "F2R_PROTOCOL_AMENDMENT_CONSTANT_CHANNEL_HANDLING.json"
    )
    loader_path = repo / "src/data/v5_p3_tranche_a_guarded_loader.py"

    required = [
        f0_lock_path,
        group_contract_path,
        metric_contract_path,
        data_access_path,
        f2_report_path,
        f2_lock_path,
        f2r_report_path,
        f2r_lock_path,
        f2r_amendment_path,
        loader_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"required artifacts missing: {missing}")

    f0_lock = json.loads(f0_lock_path.read_text(encoding="utf-8"))
    group_contract = json.loads(
        group_contract_path.read_text(encoding="utf-8")
    )
    data_access = json.loads(data_access_path.read_text(encoding="utf-8"))
    f2_report = json.loads(f2_report_path.read_text(encoding="utf-8"))
    f2_lock = json.loads(f2_lock_path.read_text(encoding="utf-8"))
    f2r_report = json.loads(f2r_report_path.read_text(encoding="utf-8"))
    f2r_lock = json.loads(f2r_lock_path.read_text(encoding="utf-8"))
    f2r_amendment = json.loads(
        f2r_amendment_path.read_text(encoding="utf-8")
    )

    require(f0_lock.get("status") == "FROZEN", "F0 is not frozen")
    require(f2_report.get("status") == "PASS", "F2 is not PASS")
    require(
        f2_lock.get("report_sha256") == sha256_file(f2_report_path),
        "F2 report/lock mismatch",
    )
    require(f2r_report.get("status") == "PASS", "F2R is not PASS")
    require(
        f2r_lock.get("report_sha256") == sha256_file(f2r_report_path),
        "F2R report/lock mismatch",
    )
    require(
        f2r_lock.get("protocol_amendment_sha256")
        == sha256_file(f2r_amendment_path),
        "F2R protocol amendment mismatch",
    )
    require(
        f2r_lock.get("F3_authorized") is True,
        "F2R did not authorize F3",
    )
    require(
        f2r_lock.get("constant_channel_removal_authorized") is False,
        "constant-channel removal unexpectedly authorized",
    )
    require(
        data_access["F2_F3"].get("Tranche_A_train") is True,
        "F3 training access not authorized",
    )
    require(
        data_access["F2_F3"].get("Tranche_A_validation") is False,
        "F3 validation access must remain false",
    )

    channels = group_contract["channels"]
    channel_names = [row["name"] for row in channels]
    physical_indices = np.asarray(
        [physical_mask_index(name) for name in channel_names],
        dtype=np.int64,
    )

    loader_module = import_source(
        loader_path,
        "_v5_p3_f3_guarded_loader",
    )
    Dataset = loader_module.GuardedV5P3TrancheAPreliminaryDataset
    try:
        dataset = Dataset(data_root, "train", active_only=False)
    except TypeError:
        dataset = Dataset(
            data_root=data_root,
            split="train",
            active_only=False,
        )

    item_count = len(dataset)
    require(
        item_count == int(f2_report["dataset"]["items"]),
        "training item count differs from F2",
    )

    first = dataset[0]
    require(isinstance(first, dict), "guarded-loader sample is not a dict")
    pair_key, first_pair = extract_pair_id(first)
    attack_key, first_attack = extract_attack_label(first)
    count_key, first_count = extract_count_label(first)

    excluded_large = {
        "x",
        "features",
        "input",
        "physical_port_mask",
        "port_mask",
        "physical_mask",
        "edge_index",
        "y_source",
        "y_transit",
        "y_victim",
        "y_attack_path",
        "y_path",
    }
    scalar_keys = [
        key for key, value in first.items()
        if key not in excluded_large and scalarize(value) is not None
    ]
    metadata: dict[str, list[Any]] = {
        key: [None] * item_count
        for key in scalar_keys
    }
    pair_ids = [""] * item_count
    attack_labels = np.zeros(item_count, dtype=np.uint8)
    count_labels = np.zeros(item_count, dtype=np.int8)

    for index in range(item_count):
        sample = first if index == 0 else dataset[index]
        observed_pair_key, pair_id = extract_pair_id(sample)
        observed_attack_key, attack = extract_attack_label(sample)
        observed_count_key, count = extract_count_label(sample)
        require(observed_pair_key == pair_key, "pair key changed")
        require(observed_attack_key == attack_key, "attack-label key changed")
        if count_key:
            require(observed_count_key == count_key, "count-label key changed")
        pair_ids[index] = pair_id
        attack_labels[index] = attack
        count_labels[index] = count
        for key in scalar_keys:
            if key in sample:
                metadata[key][index] = scalarize(sample[key])

        if (
            (index + 1) % 10000 == 0
            or index + 1 == item_count
        ):
            print(
                f"F3_metadata_items={index + 1}/{item_count}",
                flush=True,
            )

    pair_to_list: dict[str, list[int]] = defaultdict(list)
    for index, pair_id in enumerate(pair_ids):
        pair_to_list[pair_id].append(index)
    pair_to_indices = {
        pair_id: np.asarray(indices, dtype=np.int64)
        for pair_id, indices in pair_to_list.items()
    }
    require(
        len(pair_to_indices) == 600,
        f"expected 600 matched pairs, found {len(pair_to_indices)}",
    )

    excluded_run_keys = {
        pair_key,
        attack_key,
        count_key,
        "y_attack",
        "y_graph",
        "attack_label",
        "graph_label",
        "y_attacker_count",
        "attacker_count",
        "y_count",
        "count_label",
    }
    run_candidates = discover_run_candidates(
        pair_to_indices,
        attack_labels,
        metadata,
        excluded_run_keys,
    )
    require(run_candidates, "no run/member metadata candidates found")
    best_run = run_candidates[0]
    require(
        best_run["valid_pair_fraction"] >= 0.99,
        "no scalar metadata key partitions at least 99% of pairs into "
        "one control run and one attack-bearing run",
    )
    run_key = best_run["key"]
    run_values = metadata[run_key]

    time_candidate_rows = []
    excluded_time_keys = excluded_run_keys | {run_key}
    for key, values in metadata.items():
        if key in excluded_time_keys:
            continue
        # A time key must vary within at least one pair/run.
        unique_global = len({
            canonical_scalar_key(value)
            for value in values
            if value is not None
        })
        if unique_global <= 8:
            continue
        row = evaluate_time_candidate(
            key,
            values,
            pair_to_indices,
            run_values,
            attack_labels,
        )
        time_candidate_rows.append(row)

    time_candidate_rows.sort(
        key=lambda row: (
            row["matched_active_windows"],
            row["pair_coverage"],
            -row["duplicate_time_pairs"],
            int(
                any(token in row["time_key"].lower() for token in (
                    "window", "frame", "epoch", "time", "start", "end", "index"
                ))
            ),
        ),
        reverse=True,
    )

    active_total = int(np.sum(attack_labels == 1))
    alignment_mode = None
    alignment_key = None
    alignment_summary = None
    matches = None

    if (
        time_candidate_rows
        and time_candidate_rows[0]["active_alignment_fraction"]
        >= ALIGNMENT_COVERAGE_GATE
        and time_candidate_rows[0]["duplicate_time_pairs"] == 0
    ):
        best_time = time_candidate_rows[0]
        alignment_mode = "metadata_key"
        alignment_key = best_time["time_key"]
        alignment_summary = best_time
        matches = build_metadata_alignment(
            pair_to_indices,
            run_values,
            metadata[alignment_key],
            attack_labels,
        )
    else:
        ordinal = build_ordinal_alignment(
            pair_to_indices,
            run_values,
            attack_labels,
        )
        require(
            ordinal["summary"]["active_alignment_fraction"]
            >= ALIGNMENT_COVERAGE_GATE,
            "neither metadata-key nor ordinal alignment reaches the "
            f"{ALIGNMENT_COVERAGE_GATE:.2%} active-window coverage gate",
        )
        alignment_mode = ordinal["summary"]["mode"]
        alignment_key = ordinal["summary"]["time_key"]
        alignment_summary = ordinal["summary"]
        matches = ordinal["matches"]

    require(matches is not None, "alignment construction failed")

    active_matches_by_pair: dict[str, list[tuple[int, int, Any]]] = {}
    for pair_id, pair_matches in matches.items():
        active_matches = [
            match
            for match in pair_matches
            if attack_labels[int(match[1])] == 1
        ]
        active_matches_by_pair[pair_id] = active_matches

    matched_active = sum(
        len(pair_matches)
        for pair_matches in active_matches_by_pair.values()
    )
    active_coverage = matched_active / active_total if active_total else 0.0
    require(
        active_coverage >= ALIGNMENT_COVERAGE_GATE,
        f"active aligned coverage {active_coverage:.6f} below gate",
    )

    pair_order = sorted(pair_to_indices)
    pair_position = {
        pair_id: position
        for position, pair_id in enumerate(pair_order)
    }
    pair_count = len(pair_order)
    role_count = len(ROLE_NAMES)
    phase_count = len(PHASE_NAMES)
    channel_count = len(channels)

    signed_sum = np.zeros(
        (pair_count, phase_count, role_count, channel_count),
        dtype=np.float64,
    )
    absolute_sum = np.zeros_like(signed_sum)
    value_count = np.zeros(
        (pair_count, phase_count, role_count, channel_count),
        dtype=np.int64,
    )
    pair_k = np.zeros(pair_count, dtype=np.int8)
    pair_shared_victim = np.zeros(pair_count, dtype=np.uint8)
    pair_active_windows = np.zeros(pair_count, dtype=np.int32)

    nonfinite_delta_count = 0
    physical_mask_mismatch_count = 0
    label_inconsistency_count = 0
    processed = 0

    for pair_id in pair_order:
        position = pair_position[pair_id]
        pair_matches = active_matches_by_pair[pair_id]
        pair_active_windows[position] = len(pair_matches)
        if not pair_matches:
            continue

        pair_matches = sorted(
            pair_matches,
            key=lambda row: canonical_scalar_key(row[2]),
        )
        early_count = max(
            1,
            int(math.ceil(EARLY_FRACTION * len(pair_matches))),
        )

        observed_counts = set()
        observed_shared = set()

        for local_index, (control_index, attack_index, _) in enumerate(
            pair_matches
        ):
            control_sample = dataset[int(control_index)]
            attack_sample = dataset[int(attack_index)]

            x_control = tensor_to_numpy(
                extract_first(
                    control_sample,
                    ("x", "features", "input"),
                )
            ).astype(np.float64, copy=False)
            x_attack = tensor_to_numpy(
                extract_first(
                    attack_sample,
                    ("x", "features", "input"),
                )
            ).astype(np.float64, copy=False)
            require(
                tuple(x_control.shape) == (16, 70, 32),
                "control x shape changed",
            )
            require(
                tuple(x_attack.shape) == (16, 70, 32),
                "attack x shape changed",
            )

            control_mask = tensor_to_numpy(
                extract_first(
                    control_sample,
                    (
                        "physical_port_mask",
                        "port_mask",
                        "physical_mask",
                    ),
                )
            ).astype(bool, copy=False)
            attack_mask = tensor_to_numpy(
                extract_first(
                    attack_sample,
                    (
                        "physical_port_mask",
                        "port_mask",
                        "physical_mask",
                    ),
                )
            ).astype(bool, copy=False)
            if not np.array_equal(control_mask, attack_mask):
                physical_mask_mismatch_count += 1
                raise RuntimeError(
                    f"physical mask mismatch in pair {pair_id}"
                )

            _, count_label = extract_count_label(attack_sample)
            observed_counts.add(count_label)

            source = extract_bitmap(
                attack_sample,
                ("y_source", "source_label"),
            )
            transit = extract_bitmap(
                attack_sample,
                ("y_transit", "transit_label"),
            )
            victim = extract_bitmap(
                attack_sample,
                ("y_victim", "victim_label"),
            )
            path = extract_bitmap(
                attack_sample,
                ("y_attack_path", "y_path", "path_label"),
            )
            observed_shared.add(
                bool(count_label > int(np.sum(victim)))
            )

            roles = np.stack(
                [
                    np.ones(16, dtype=bool),
                    source,
                    transit,
                    victim,
                    path,
                    path & ~(source | victim),
                    ~path,
                ],
                axis=0,
            )

            delta = np.transpose(
                x_attack - x_control,
                (1, 0, 2),
            )
            nonfinite_delta_count += int(
                np.sum(~np.isfinite(delta))
            )
            if not np.all(np.isfinite(delta)):
                raise RuntimeError(
                    f"nonfinite aligned delta in pair {pair_id}"
                )

            epoch_signed = np.sum(delta, axis=2)
            epoch_absolute = np.sum(np.abs(delta), axis=2)
            valid_router = attack_mask[:, physical_indices].T

            phase_indices = [0]
            phase_indices.append(
                1 if local_index < early_count else 2
            )
            for phase in phase_indices:
                for role_index, role_mask in enumerate(roles):
                    selected = valid_router & role_mask[None, :]
                    selected_count = (
                        np.sum(selected, axis=1).astype(np.int64) * 32
                    )
                    signed_sum[
                        position, phase, role_index
                    ] += np.sum(
                        np.where(selected, epoch_signed, 0.0),
                        axis=1,
                    )
                    absolute_sum[
                        position, phase, role_index
                    ] += np.sum(
                        np.where(selected, epoch_absolute, 0.0),
                        axis=1,
                    )
                    value_count[
                        position, phase, role_index
                    ] += selected_count

            processed += 1
            if (
                processed % PROGRESS_INTERVAL == 0
                or processed == matched_active
            ):
                print(
                    f"F3_aligned_active_windows={processed}/{matched_active}",
                    flush=True,
                )

        if len(observed_counts) != 1 or len(observed_shared) != 1:
            label_inconsistency_count += 1
            raise RuntimeError(
                f"pair {pair_id} has inconsistent active labels: "
                f"counts={observed_counts}, shared={observed_shared}"
            )
        pair_k[position] = int(next(iter(observed_counts)))
        pair_shared_victim[position] = int(
            next(iter(observed_shared))
        )

    signed_mean = np.divide(
        signed_sum,
        value_count,
        out=np.full_like(signed_sum, np.nan),
        where=value_count > 0,
    )
    absolute_mean = np.divide(
        absolute_sum,
        value_count,
        out=np.full_like(absolute_sum, np.nan),
        where=value_count > 0,
    )

    constant_indices = [
        int(row["dynamic70_index"])
        for row in f2r_report["constant_channels"]
    ]
    constant_max_abs = float(
        np.nanmax(
            np.abs(
                signed_mean[:, 0, :, constant_indices]
            )
        )
    )
    require(
        constant_max_abs <= ZERO_TOLERANCE,
        "F2R zero-signal channels produced nonzero matched response",
    )

    pair_effects_path = paired_dir / "F3_PAIR_LEVEL_EFFECTS.npz"
    atomic_npz(
        pair_effects_path,
        signed_sum=signed_sum,
        absolute_sum=absolute_sum,
        value_count=value_count,
        signed_mean=signed_mean,
        absolute_mean=absolute_mean,
        pair_k=pair_k,
        pair_shared_victim=pair_shared_victim,
        pair_active_windows=pair_active_windows,
        pair_ids=np.asarray(pair_order, dtype="U128"),
    )

    per_channel_rows = []
    seed_counter = BOOTSTRAP_SEED
    for phase_index, phase_name in enumerate(PHASE_NAMES):
        for role_index, role_name in enumerate(ROLE_NAMES):
            for channel_index, metadata_row in enumerate(channels):
                stats = summarize_pair_values(
                    signed_mean[
                        :, phase_index, role_index, channel_index
                    ],
                    absolute_mean[
                        :, phase_index, role_index, channel_index
                    ],
                    seed=seed_counter,
                    bootstrap=(phase_name == "all_active"),
                )
                seed_counter += 1
                per_channel_rows.append({
                    "phase": phase_name,
                    "router_role": role_name,
                    **metadata_row,
                    **stats,
                })

    group_rows = []
    k_rows = []
    shared_rows = []
    phase_rows = []
    seed_counter = BOOTSTRAP_SEED + 100000

    for group_name, indices in group_contract["macro_groups"].items():
        indices = np.asarray(indices, dtype=np.int64)
        group_signed_sum = np.sum(
            signed_sum[:, :, :, indices],
            axis=3,
        )
        group_abs_sum = np.sum(
            absolute_sum[:, :, :, indices],
            axis=3,
        )
        group_counts = np.sum(
            value_count[:, :, :, indices],
            axis=3,
        )
        group_signed = np.divide(
            group_signed_sum,
            group_counts,
            out=np.full_like(group_signed_sum, np.nan),
            where=group_counts > 0,
        )
        group_absolute = np.divide(
            group_abs_sum,
            group_counts,
            out=np.full_like(group_abs_sum, np.nan),
            where=group_counts > 0,
        )

        for phase_index, phase_name in enumerate(PHASE_NAMES):
            for role_index, role_name in enumerate(ROLE_NAMES):
                stats = summarize_pair_values(
                    group_signed[:, phase_index, role_index],
                    group_absolute[:, phase_index, role_index],
                    seed=seed_counter,
                    bootstrap=True,
                )
                seed_counter += 1
                row = {
                    "macro_group": group_name,
                    "channel_count": int(indices.size),
                    "phase": phase_name,
                    "router_role": role_name,
                    **stats,
                }
                group_rows.append(row)
                phase_rows.append(row)

                if phase_name == "all_active":
                    for k in (1, 2, 3, 4):
                        selector = pair_k == k
                        stats_k = summarize_pair_values(
                            group_signed[
                                selector, phase_index, role_index
                            ],
                            group_absolute[
                                selector, phase_index, role_index
                            ],
                            seed=seed_counter,
                            bootstrap=True,
                        )
                        seed_counter += 1
                        k_rows.append({
                            "macro_group": group_name,
                            "channel_count": int(indices.size),
                            "attacker_count": k,
                            "router_role": role_name,
                            **stats_k,
                        })

                    for shared_value, shared_name in (
                        (0, "distinct_or_nonshared_victims"),
                        (1, "shared_victim"),
                    ):
                        selector = pair_shared_victim == shared_value
                        stats_shared = summarize_pair_values(
                            group_signed[
                                selector, phase_index, role_index
                            ],
                            group_absolute[
                                selector, phase_index, role_index
                            ],
                            seed=seed_counter,
                            bootstrap=True,
                        )
                        seed_counter += 1
                        shared_rows.append({
                            "macro_group": group_name,
                            "channel_count": int(indices.size),
                            "victim_sharing": shared_name,
                            "router_role": role_name,
                            **stats_shared,
                        })

    channel_csv = paired_dir / "F3_PER_CHANNEL_ROLE_EFFECTS.csv"
    group_csv = paired_dir / "F3_PER_GROUP_ROLE_EFFECTS.csv"
    k_csv = paired_dir / "F3_ATTACKER_COUNT_GROUP_EFFECTS.csv"
    shared_csv = paired_dir / "F3_SHARED_VICTIM_GROUP_EFFECTS.csv"
    phase_csv = paired_dir / "F3_TEMPORAL_PHASE_GROUP_EFFECTS.csv"
    alignment_path = paired_dir / "F3_PAIR_ALIGNMENT_AUDIT.json"
    heatmap_png = paired_dir / "F3_ROLE_FEATURE_RESPONSE_HEATMAP.png"
    heatmap_pdf = paired_dir / "F3_ROLE_FEATURE_RESPONSE_HEATMAP.pdf"

    write_csv(channel_csv, per_channel_rows)
    write_csv(group_csv, group_rows)
    write_csv(k_csv, k_rows)
    write_csv(shared_csv, shared_rows)
    write_csv(phase_csv, phase_rows)

    atomic_json(
        alignment_path,
        {
            "stage": STAGE,
            "pair_key": pair_key,
            "attack_label_key": attack_key,
            "count_label_key": count_key,
            "scalar_metadata_keys": scalar_keys,
            "run_key": run_key,
            "run_candidate_rows": run_candidates,
            "time_candidate_rows": time_candidate_rows,
            "selected_alignment_mode": alignment_mode,
            "selected_alignment_key": alignment_key,
            "selected_alignment_summary": alignment_summary,
            "active_windows_total": active_total,
            "active_windows_aligned": matched_active,
            "active_alignment_fraction": active_coverage,
            "pairs": pair_count,
            "pairs_with_active_matches": int(
                np.sum(pair_active_windows > 0)
            ),
        },
    )

    figure_status = {
        "generated": False,
        "png": None,
        "pdf": None,
        "exception": None,
    }
    try:
        import matplotlib.pyplot as plt

        matrix = np.full((len(ROLE_NAMES), channel_count), np.nan)
        for role_index, role_name in enumerate(ROLE_NAMES):
            for channel_index in range(channel_count):
                values = signed_mean[
                    :, 0, role_index, channel_index
                ]
                matrix[role_index, channel_index] = np.nanmedian(values)

        fig, ax = plt.subplots(figsize=(18, 5))
        image = ax.imshow(matrix, aspect="auto")
        ax.set_xlabel("Dynamic70 channel index")
        ax.set_ylabel("Router role")
        ax.set_xticks(range(0, channel_count, 2))
        ax.set_xticklabels(
            [str(index) for index in range(0, channel_count, 2)],
            fontsize=7,
        )
        ax.set_yticks(range(len(ROLE_NAMES)))
        ax.set_yticklabels(ROLE_NAMES)
        fig.colorbar(image, ax=ax, label="Median matched attack-control response")
        fig.tight_layout()
        fig.savefig(heatmap_png, dpi=180)
        fig.savefig(heatmap_pdf)
        plt.close(fig)
        figure_status = {
            "generated": True,
            "png": str(heatmap_png),
            "pdf": str(heatmap_pdf),
            "exception": None,
        }
    except Exception as exc:
        figure_status["exception"] = repr(exc)

    f4_authorized = (
        active_coverage >= ALIGNMENT_COVERAGE_GATE
        and nonfinite_delta_count == 0
        and physical_mask_mismatch_count == 0
        and label_inconsistency_count == 0
        and constant_max_abs <= ZERO_TOLERANCE
    )
    next_stage = (
        "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_BASELINE_REPRODUCTION"
        if f4_authorized
        else "V5_P3_F3R_MATCHED_RESPONSE_ALIGNMENT_REVIEW"
    )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": "Tranche-A training-only matched-pair response study",
        "method": {
            "comparison": (
                "attack-bearing run minus matched control run at the same "
                "metadata time key or certified within-run ordinal"
            ),
            "active_windows_only": True,
            "physical_invalid_ports_excluded": True,
            "pair_cluster_unit": True,
            "early_fraction": EARLY_FRACTION,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "alignment": {
            "pair_key": pair_key,
            "run_key": run_key,
            "mode": alignment_mode,
            "time_key": alignment_key,
            "pairs": pair_count,
            "active_windows_total": active_total,
            "active_windows_aligned": matched_active,
            "active_alignment_fraction": active_coverage,
            "gate": ALIGNMENT_COVERAGE_GATE,
        },
        "integrity": {
            "nonfinite_delta_count": nonfinite_delta_count,
            "physical_mask_mismatch_count": physical_mask_mismatch_count,
            "label_inconsistency_count": label_inconsistency_count,
            "F2R_constant_channel_max_abs_response": constant_max_abs,
        },
        "artifacts": {
            "pair_alignment_audit": str(alignment_path),
            "pair_level_effects": str(pair_effects_path),
            "per_channel_role_effects": str(channel_csv),
            "per_group_role_effects": str(group_csv),
            "attacker_count_group_effects": str(k_csv),
            "shared_victim_group_effects": str(shared_csv),
            "temporal_phase_group_effects": str(phase_csv),
            "figure": figure_status,
        },
        "decision": {
            "F3_complete": True,
            "F4_authorized": f4_authorized,
            "validation_access_authorized_now": False,
            "sealed_test_access_authorized": False,
            "constant_channel_removal_authorized": False,
            "next_stage": next_stage,
        },
        "governance": {
            "model_loaded": False,
            "training_tensors_loaded": True,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
        },
        "provenance": {
            "F0_lock_sha256": sha256_file(f0_lock_path),
            "feature_group_contract_sha256": sha256_file(group_contract_path),
            "metric_contract_sha256": sha256_file(metric_contract_path),
            "data_access_contract_sha256": sha256_file(data_access_path),
            "F2_report_sha256": sha256_file(f2_report_path),
            "F2_lock_sha256": sha256_file(f2_lock_path),
            "F2R_report_sha256": sha256_file(f2r_report_path),
            "F2R_lock_sha256": sha256_file(f2r_lock_path),
            "F2R_amendment_sha256": sha256_file(f2r_amendment_path),
            "guarded_loader_sha256": sha256_file(loader_path),
            "installed_script_sha256": sha256_file(installed_script),
        },
    }

    report_path = paired_dir / f"{STAGE}_REPORT.json"
    lock_path = paired_dir / f"{STAGE}_LOCK.json"
    complete_path = paired_dir / f"{STAGE}_COMPLETE"
    atomic_json(report_path, report)
    atomic_json(
        lock_path,
        {
            "stage": STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(report_path),
            "pair_alignment_audit_sha256": sha256_file(alignment_path),
            "pair_level_effects_sha256": sha256_file(pair_effects_path),
            "per_channel_role_effects_sha256": sha256_file(channel_csv),
            "per_group_role_effects_sha256": sha256_file(group_csv),
            "attacker_count_group_effects_sha256": sha256_file(k_csv),
            "shared_victim_group_effects_sha256": sha256_file(shared_csv),
            "temporal_phase_group_effects_sha256": sha256_file(phase_csv),
            "active_alignment_fraction": active_coverage,
            "constant_channel_max_abs_response": constant_max_abs,
            "F4_authorized": f4_authorized,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign={CAMPAIGN}")
    print(f"pair_key={pair_key}")
    print(f"run_key={run_key}")
    print(f"alignment_mode={alignment_mode}")
    print(f"alignment_key={alignment_key}")
    print(f"pairs={pair_count}")
    print(f"active_windows_total={active_total}")
    print(f"active_windows_aligned={matched_active}")
    print(f"active_alignment_fraction={active_coverage:.8f}")
    print(f"alignment_coverage_gate={ALIGNMENT_COVERAGE_GATE:.8f}")
    print(f"nonfinite_delta_count={nonfinite_delta_count}")
    print(
        "physical_mask_mismatch_count="
        f"{physical_mask_mismatch_count}"
    )
    print(
        "label_inconsistency_count="
        f"{label_inconsistency_count}"
    )
    print(
        "F2R_constant_channel_max_abs_response="
        f"{constant_max_abs:.12g}"
    )
    print(f"figure_generated={str(figure_status['generated']).lower()}")
    print("model_loaded=false")
    print("training_tensors_loaded=true")
    print("validation_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print("constant_channel_removal_authorized=false")
    print(f"F4_authorized={str(f4_authorized).lower()}")
    print(f"next_stage={next_stage}")
    print(f"alignment_audit={alignment_path}")
    print(f"per_channel_effects={channel_csv}")
    print(f"per_group_effects={group_csv}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
