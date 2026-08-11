#!/usr/bin/env python3
"""
Chrono-B1 Stage B1.1: audit the Chrono-A1 training distribution.

Reads metadata arrays from the corrected chronological V3 dataset and writes:
- training_distribution_summary.json
- training_distribution_by_run.csv
- training_distribution_by_graph_class.csv
- training_distribution_by_profile.csv
- training_distribution_by_strength.csv
- training_distribution_by_attacker_count.csv
- training_distribution_by_active_cores.csv

No dataset files are modified.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REQUIRED_FILES = (
    "y_graph.npy",
    "run_id.npy",
    "split.npy",
    "profile.npy",
    "active_cores.npy",
    "strength.npy",
    "attackers.npy",
)

OPTIONAL_FILES = (
    "end_epoch.npy",
    "seed.npy",
)


def load_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path, mmap_mode="r", allow_pickle=True)


def scalar_to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_text(value: Any) -> str:
    value = scalar_to_python(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if value is None:
        return "NA"
    text = str(value).strip()
    return text if text else "NA"


def normalize_graph_label(value: Any) -> int:
    value = scalar_to_python(value)
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return int(round(float(value)))

    text = normalize_text(value).lower()
    if text in {"0", "normal", "benign", "no_attack", "no-attack"}:
        return 0
    if text in {"1", "attack", "malicious", "ddos", "dos"}:
        return 1
    raise ValueError(f"Cannot normalize graph label: {value!r}")


def normalize_strength(value: Any) -> str:
    value = scalar_to_python(value)
    if value is None:
        return "NA"

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "NA"
        if number.is_integer():
            return str(int(number))
        return f"{number:g}"

    text = str(value).strip().strip('"').strip("'")
    if not text or text.lower() in {"na", "nan", "none", "null"}:
        return "NA"

    try:
        number = float(text)
        if math.isfinite(number):
            return str(int(number)) if number.is_integer() else f"{number:g}"
    except ValueError:
        pass

    return text


def parse_sequence_like(value: Any) -> list[Any]:
    value = scalar_to_python(value)

    if value is None:
        return []

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (list, tuple, set)):
        return list(value)

    if isinstance(value, (int, np.integer)):
        number = int(value)
        return [] if number < 0 else [number]

    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number) or number < 0:
            return []
        return [int(number)]

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "none", "null", "[]", "()"}:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set, np.ndarray)):
            return list(parsed)
        if isinstance(parsed, (int, float)):
            return [parsed]
    except (ValueError, SyntaxError):
        pass

    for separator in ("-", ",", "_", " "):
        if separator in text:
            parts = [part.strip() for part in text.split(separator)]
            cleaned = []
            for part in parts:
                if not part:
                    continue
                try:
                    cleaned.append(int(float(part)))
                except ValueError:
                    cleaned.append(part)
            return cleaned

    try:
        return [int(float(text))]
    except ValueError:
        return [text]


def canonical_sequence(value: Any) -> str:
    items = parse_sequence_like(value)
    normalized: list[str] = []
    for item in items:
        item = scalar_to_python(item)
        try:
            number = int(float(item))
            normalized.append(str(number))
        except (TypeError, ValueError):
            text = normalize_text(item)
            if text != "NA":
                normalized.append(text)

    if not normalized:
        return "NA"

    def key_fn(token: str) -> tuple[int, Any]:
        try:
            return (0, int(token))
        except ValueError:
            return (1, token)

    normalized = sorted(dict.fromkeys(normalized), key=key_fn)
    return "-".join(normalized)


def attacker_count(value: Any, graph_label: int) -> int:
    if graph_label == 0:
        return 0
    items = parse_sequence_like(value)
    cleaned = []
    for item in items:
        text = normalize_text(item)
        if text.lower() not in {"na", "nan", "none", "null", "-1"}:
            cleaned.append(text)
    return len(cleaned)


def detect_train_mask(split: np.ndarray) -> tuple[np.ndarray, str, dict[str, int]]:
    values = np.asarray(split)
    unique, counts = np.unique(values, return_counts=True)
    distribution = {
        normalize_text(key): int(count)
        for key, count in zip(unique.tolist(), counts.tolist())
    }

    if values.dtype.kind in {"U", "S", "O"}:
        normalized = np.array(
            [normalize_text(v).lower() for v in values],
            dtype=object,
        )
        candidates = ("train", "training", "tr")
        for candidate in candidates:
            mask = normalized == candidate
            if mask.any():
                return mask, candidate, distribution

        if np.any(normalized == "0"):
            return normalized == "0", "0", distribution

        raise ValueError(
            "Could not identify training split from string labels. "
            f"Observed split counts: {distribution}"
        )

    numeric = values.astype(np.int64, copy=False)
    if np.any(numeric == 0):
        return numeric == 0, "0", distribution

    raise ValueError(
        "Could not identify training split. Expected numeric label 0 or "
        f"a train-like string. Observed split counts: {distribution}"
    )


def write_counter_csv(path: Path, key_name: str, counter: Counter[str]) -> None:
    total = sum(counter.values())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[key_name, "sample_count", "sample_fraction"],
        )
        writer.writeheader()
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow(
                {
                    key_name: key,
                    "sample_count": count,
                    "sample_fraction": count / total if total else 0.0,
                }
            )


def quantiles(values: Iterable[int]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
        "std": float(np.std(array)),
        "coefficient_of_variation": (
            float(np.std(array) / np.mean(array)) if np.mean(array) else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: output directory already exists and is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        filename.removesuffix(".npy"): load_array(data_dir / filename)
        for filename in REQUIRED_FILES
    }
    for filename in OPTIONAL_FILES:
        path = data_dir / filename
        if path.is_file():
            arrays[filename.removesuffix(".npy")] = np.load(
                path, mmap_mode="r", allow_pickle=True
            )

    lengths = {name: len(array) for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise SystemExit(f"STOP: metadata lengths differ: {lengths}")

    sample_count = next(iter(lengths.values()))
    train_mask, train_label, split_distribution = detect_train_mask(arrays["split"])
    train_indices = np.flatnonzero(train_mask)

    if train_indices.size == 0:
        raise SystemExit("STOP: training split contains zero samples")

    counters: dict[str, Counter[str]] = {
        "graph_class": Counter(),
        "profile": Counter(),
        "strength": Counter(),
        "attacker_count": Counter(),
        "active_cores": Counter(),
        "run_id": Counter(),
    }
    run_rows: dict[str, dict[str, Any]] = {}
    run_end_epochs: defaultdict[str, list[int]] = defaultdict(list)

    for index in train_indices.tolist():
        graph_label = normalize_graph_label(arrays["y_graph"][index])
        graph_class = "attack" if graph_label == 1 else "normal"
        profile = normalize_text(arrays["profile"][index]).lower()
        strength = normalize_strength(arrays["strength"][index])
        attackers_value = arrays["attackers"][index]
        count = attacker_count(attackers_value, graph_label)
        active = canonical_sequence(arrays["active_cores"][index])
        run_id = normalize_text(arrays["run_id"][index])

        counters["graph_class"][graph_class] += 1
        counters["profile"][profile] += 1
        counters["strength"][strength] += 1
        counters["attacker_count"][str(count)] += 1
        counters["active_cores"][active] += 1
        counters["run_id"][run_id] += 1

        if run_id not in run_rows:
            run_rows[run_id] = {
                "run_id": run_id,
                "graph_class": graph_class,
                "profile": profile,
                "strength": strength,
                "attacker_count": count,
                "active_cores": active,
                "attackers": canonical_sequence(attackers_value),
                "sample_count": 0,
            }

        row = run_rows[run_id]
        row["sample_count"] += 1

        consistency_fields = {
            "graph_class": graph_class,
            "profile": profile,
            "strength": strength,
            "attacker_count": count,
            "active_cores": active,
            "attackers": canonical_sequence(attackers_value),
        }
        for field, observed in consistency_fields.items():
            if row[field] != observed:
                row[field] = "MIXED"

        if "end_epoch" in arrays:
            try:
                run_end_epochs[run_id].append(int(arrays["end_epoch"][index]))
            except (TypeError, ValueError):
                pass

    run_class_counts = Counter(row["graph_class"] for row in run_rows.values())
    run_sample_counts = [int(row["sample_count"]) for row in run_rows.values()]

    for run_id, row in run_rows.items():
        epochs = sorted(run_end_epochs.get(run_id, []))
        if len(epochs) >= 2:
            diffs = np.diff(np.asarray(epochs, dtype=np.int64))
            row["end_epoch_min_step"] = int(np.min(diffs))
            row["end_epoch_median_step"] = float(np.median(diffs))
            row["end_epoch_max_step"] = int(np.max(diffs))
            row["adjacent_end_epoch_pairs"] = int(np.sum(diffs == 1))
            row["adjacent_end_epoch_pair_fraction"] = float(np.mean(diffs == 1))
        else:
            row["end_epoch_min_step"] = ""
            row["end_epoch_median_step"] = ""
            row["end_epoch_max_step"] = ""
            row["adjacent_end_epoch_pairs"] = ""
            row["adjacent_end_epoch_pair_fraction"] = ""

    run_sample_count_stats = quantiles(run_sample_counts)
    approximately_equal = (
        run_sample_count_stats.get("coefficient_of_variation", float("inf")) <= 0.05
        and (
            run_sample_count_stats.get("maximum", 0)
            - run_sample_count_stats.get("minimum", 0)
        )
        <= max(1.0, 0.05 * run_sample_count_stats.get("mean", 0))
    )

    normal_samples = counters["graph_class"].get("normal", 0)
    attack_samples = counters["graph_class"].get("attack", 0)
    normal_runs = run_class_counts.get("normal", 0)
    attack_runs = run_class_counts.get("attack", 0)

    summary = {
        "data_dir": str(data_dir),
        "total_dataset_samples": sample_count,
        "metadata_lengths": lengths,
        "split_distribution": split_distribution,
        "detected_train_label": train_label,
        "training_sample_count": int(train_indices.size),
        "training_run_count": len(run_rows),
        "training_class_samples": {
            "normal": normal_samples,
            "attack": attack_samples,
            "normal_fraction": normal_samples / train_indices.size,
            "attack_fraction": attack_samples / train_indices.size,
            "attack_to_normal_sample_ratio": (
                attack_samples / normal_samples if normal_samples else None
            ),
        },
        "training_class_runs": {
            "normal": normal_runs,
            "attack": attack_runs,
            "attack_to_normal_run_ratio": (
                attack_runs / normal_runs if normal_runs else None
            ),
        },
        "samples_per_run": run_sample_count_stats,
        "windows_per_run_approximately_equal": approximately_equal,
        "unique_group_counts": {
            "profiles": len(counters["profile"]),
            "strengths": len(counters["strength"]),
            "attacker_counts": len(counters["attacker_count"]),
            "active_core_configurations": len(counters["active_cores"]),
            "run_ids": len(counters["run_id"]),
        },
        "observed_distributions": {
            key: dict(sorted(counter.items()))
            for key, counter in counters.items()
            if key != "run_id"
        },
        "b1_interpretation": {
            "class_balance_needed": (
                abs(
                    counters["graph_class"].get("normal", 0) / train_indices.size
                    - 0.5
                )
                > 0.05
            ),
            "run_uniform_sampling_matters": not approximately_equal,
            "note": (
                "Final B1 hierarchy must be frozen after reviewing the generated "
                "CSV tables. This script does not modify the sampler or dataset."
            ),
        },
    }

    (output_dir / "training_distribution_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    write_counter_csv(
        output_dir / "training_distribution_by_graph_class.csv",
        "graph_class",
        counters["graph_class"],
    )
    write_counter_csv(
        output_dir / "training_distribution_by_profile.csv",
        "profile",
        counters["profile"],
    )
    write_counter_csv(
        output_dir / "training_distribution_by_strength.csv",
        "strength",
        counters["strength"],
    )
    write_counter_csv(
        output_dir / "training_distribution_by_attacker_count.csv",
        "attacker_count",
        counters["attacker_count"],
    )
    write_counter_csv(
        output_dir / "training_distribution_by_active_cores.csv",
        "active_cores",
        counters["active_cores"],
    )

    run_fields = [
        "run_id",
        "graph_class",
        "profile",
        "strength",
        "attacker_count",
        "active_cores",
        "attackers",
        "sample_count",
        "end_epoch_min_step",
        "end_epoch_median_step",
        "end_epoch_max_step",
        "adjacent_end_epoch_pairs",
        "adjacent_end_epoch_pair_fraction",
    ]
    with (output_dir / "training_distribution_by_run.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        for row in sorted(run_rows.values(), key=lambda item: item["run_id"]):
            writer.writerow({field: row.get(field, "") for field in run_fields})

    print("B1.1 TRAINING DISTRIBUTION AUDIT: PASS")
    print(f"training_samples={train_indices.size}")
    print(f"training_runs={len(run_rows)}")
    print(f"normal_samples={normal_samples}")
    print(f"attack_samples={attack_samples}")
    print(f"normal_runs={normal_runs}")
    print(f"attack_runs={attack_runs}")
    print(
        "attack_to_normal_sample_ratio="
        f"{summary['training_class_samples']['attack_to_normal_sample_ratio']:.6f}"
        if normal_samples
        else "attack_to_normal_sample_ratio=NA"
    )
    print(
        "attack_to_normal_run_ratio="
        f"{summary['training_class_runs']['attack_to_normal_run_ratio']:.6f}"
        if normal_runs
        else "attack_to_normal_run_ratio=NA"
    )
    print(
        "samples_per_run_min="
        f"{run_sample_count_stats.get('minimum', float('nan')):.0f}"
    )
    print(
        "samples_per_run_median="
        f"{run_sample_count_stats.get('median', float('nan')):.1f}"
    )
    print(
        "samples_per_run_max="
        f"{run_sample_count_stats.get('maximum', float('nan')):.0f}"
    )
    print(f"windows_per_run_approximately_equal={approximately_equal}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
