#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = [
    "run_id",
    "split",
    "end_epoch",
    "y_graph",
    "y_node",
]


OPTIONAL_KEYS = [
    "attackers",
    "profile",
    "active_cores",
    "strength",
    "seed",
]


def load_array(path: Path) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (ValueError, TypeError):
        return np.load(path, allow_pickle=True)


def value_to_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value_to_text(value.item())

        return json.dumps(
            value.tolist(),
            default=str,
            separators=(",", ":"),
        )

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (list, tuple, dict)):
        return json.dumps(
            value,
            default=str,
            separators=(",", ":"),
        )

    return str(value)


def unique_text_values(
    array: np.ndarray,
    indices: np.ndarray,
) -> list[str]:
    return sorted(
        {
            value_to_text(array[index])
            for index in indices
        }
    )


def summarize_constant_field(
    array: np.ndarray,
    indices: np.ndarray,
) -> tuple[str, bool]:
    values = unique_text_values(array, indices)

    if len(values) == 1:
        return values[0], True

    return "MIXED:" + json.dumps(values), False


def typical_step(values: np.ndarray) -> str:
    if len(values) <= 1:
        return ""

    differences = np.diff(values)

    if len(differences) == 0:
        return ""

    count = Counter(
        difference.item()
        if isinstance(difference, np.generic)
        else difference
        for difference in differences
    )

    return str(count.most_common(1)[0][0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build V3 sample and run metadata tables."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--sample-out",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--run-out",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()

    arrays: dict[str, np.ndarray] = {}

    for key in REQUIRED_KEYS:
        path = data_dir / f"{key}.npy"

        if not path.is_file():
            raise SystemExit(
                f"Required metadata file missing: {path}"
            )

        arrays[key] = load_array(path)

    for key in OPTIONAL_KEYS:
        path = data_dir / f"{key}.npy"

        if path.is_file():
            arrays[key] = load_array(path)

    sample_count = len(arrays["run_id"])

    for key, array in arrays.items():
        if len(array) != sample_count:
            raise SystemExit(
                f"Length mismatch: {key} has {len(array)}, "
                f"expected {sample_count}"
            )

    run_id = np.asarray(arrays["run_id"]).astype(str).reshape(-1)
    split = np.asarray(arrays["split"]).astype(str).reshape(-1)
    end_epoch = np.asarray(arrays["end_epoch"]).reshape(-1)
    y_graph = np.asarray(arrays["y_graph"]).astype(np.int64).reshape(-1)
    y_node = np.asarray(arrays["y_node"]).astype(np.int64)

    attacker_count = y_node.sum(axis=1).astype(np.int64)

    candidate_ids = np.asarray(
        [
            f"{run}:{epoch}"
            for run, epoch in zip(run_id, end_epoch)
        ],
        dtype=object,
    )

    candidate_unique = len(np.unique(candidate_ids)) == sample_count

    print("sample_count:", sample_count)
    print("candidate_sample_id_format: run_id:end_epoch")
    print("candidate_sample_id_unique:", candidate_unique)

    if candidate_unique:
        sample_ids = candidate_ids
    else:
        print(
            "WARNING: run_id:end_epoch is not unique; "
            "sample_index is appended."
        )

        sample_ids = np.asarray(
            [
                f"{run}:{epoch}:{index}"
                for index, (run, epoch) in enumerate(
                    zip(run_id, end_epoch)
                )
            ],
            dtype=object,
        )

    args.sample_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.run_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_fields = [
        "sample_index",
        "sample_id",
        "run_id",
        "split",
        "end_epoch",
        "true_graph",
        "true_attacker_count",
        "attackers",
        "profile",
        "active_cores",
        "strength",
        "seed",
    ]

    with args.sample_out.open(
        "w",
        newline="",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=sample_fields,
        )
        writer.writeheader()

        for index in range(sample_count):
            row = {
                "sample_index": index,
                "sample_id": sample_ids[index],
                "run_id": run_id[index],
                "split": split[index],
                "end_epoch": value_to_text(end_epoch[index]),
                "true_graph": int(y_graph[index]),
                "true_attacker_count": int(attacker_count[index]),
                "attackers": (
                    value_to_text(arrays["attackers"][index])
                    if "attackers" in arrays
                    else ""
                ),
                "profile": (
                    value_to_text(arrays["profile"][index])
                    if "profile" in arrays
                    else ""
                ),
                "active_cores": (
                    value_to_text(arrays["active_cores"][index])
                    if "active_cores" in arrays
                    else ""
                ),
                "strength": (
                    value_to_text(arrays["strength"][index])
                    if "strength" in arrays
                    else ""
                ),
                "seed": (
                    value_to_text(arrays["seed"][index])
                    if "seed" in arrays
                    else ""
                ),
            }

            writer.writerow(row)

    run_to_indices: dict[str, list[int]] = defaultdict(list)

    for index, current_run in enumerate(run_id):
        run_to_indices[current_run].append(index)

    run_fields = [
        "run_id",
        "split",
        "number_of_samples",
        "graph_label",
        "attacker_count",
        "attackers",
        "profile",
        "active_cores",
        "strength",
        "seed",
        "first_end_epoch",
        "last_end_epoch",
        "epoch_step",
    ]

    critical_constancy_violations: list[str] = []
    informational_mixed_fields: list[str] = []

    with args.run_out.open(
        "w",
        newline="",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=run_fields,
        )
        writer.writeheader()

        for current_run in sorted(run_to_indices):
            indices = np.asarray(
                run_to_indices[current_run],
                dtype=np.int64,
            )

            split_value, split_constant = summarize_constant_field(
                arrays["split"],
                indices,
            )

            if not split_constant:
                critical_constancy_violations.append(
                    f"{current_run}:split"
                )

            graph_value, graph_constant = summarize_constant_field(
                arrays["y_graph"],
                indices,
            )

            attacker_count_value, attacker_count_constant = (
                summarize_constant_field(
                    attacker_count,
                    indices,
                )
            )

            if not graph_constant:
                informational_mixed_fields.append(
                    f"{current_run}:graph_label"
                )

            if not attacker_count_constant:
                informational_mixed_fields.append(
                    f"{current_run}:attacker_count"
                )

            optional_values: dict[str, str] = {}

            for key in OPTIONAL_KEYS:
                if key not in arrays:
                    optional_values[key] = ""
                    continue

                value, constant = summarize_constant_field(
                    arrays[key],
                    indices,
                )
                optional_values[key] = value

                if not constant:
                    informational_mixed_fields.append(
                        f"{current_run}:{key}"
                    )

            epochs = end_epoch[indices]

            row = {
                "run_id": current_run,
                "split": split_value,
                "number_of_samples": len(indices),
                "graph_label": graph_value,
                "attacker_count": attacker_count_value,
                "attackers": optional_values.get("attackers", ""),
                "profile": optional_values.get("profile", ""),
                "active_cores": optional_values.get(
                    "active_cores",
                    "",
                ),
                "strength": optional_values.get("strength", ""),
                "seed": optional_values.get("seed", ""),
                "first_end_epoch": value_to_text(epochs[0]),
                "last_end_epoch": value_to_text(epochs[-1]),
                "epoch_step": typical_step(epochs),
            }

            writer.writerow(row)

    print("sample_table:", args.sample_out.resolve())
    print("run_table:", args.run_out.resolve())
    print("unique_run_count:", len(run_to_indices))
    print(
        "critical_constancy_violation_count:",
        len(critical_constancy_violations),
    )
    print(
        "critical_constancy_violations:",
        critical_constancy_violations,
    )
    print(
        "informational_mixed_field_count:",
        len(informational_mixed_fields),
    )
    print(
        "informational_mixed_fields_preview:",
        informational_mixed_fields[:100],
    )


if __name__ == "__main__":
    main()
