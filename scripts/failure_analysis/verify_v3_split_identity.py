#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_MODEL_DIRS = [
    Path("models/v3/temporal_gcn_ports_v3_builtin_split"),
    Path("models/v3/temporal_tcn_gcn_ports_v3_builtin_split"),
    Path("models/v3/temporal_tcn_meanpool_gcn_ports_v3_builtin_split"),
    Path("models/v3/temporal_tcn_maxpool_gcn_ports_v3_builtin_split"),
]


def load_array(path: Path) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (ValueError, TypeError):
        return np.load(path, allow_pickle=True)


def canonical_indices(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.int64).reshape(-1)


def sha256_indices(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(
        canonical_indices(array),
        dtype=np.int64,
    )
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def inspect_index_set(
    name: str,
    indices: np.ndarray,
    sample_count: int,
) -> None:
    indices = canonical_indices(indices)

    print(f"{name}_count:", len(indices))
    print(f"{name}_minimum:", int(indices.min()) if len(indices) else None)
    print(f"{name}_maximum:", int(indices.max()) if len(indices) else None)
    print(f"{name}_unique_count:", len(np.unique(indices)))
    print(f"{name}_duplicate_count:", len(indices) - len(np.unique(indices)))
    print(f"{name}_sha256:", sha256_indices(indices))

    out_of_range = indices[
        (indices < 0) | (indices >= sample_count)
    ]

    print(
        f"{name}_out_of_range_count:",
        len(out_of_range),
    )

    if len(out_of_range):
        print(
            f"{name}_out_of_range_preview:",
            out_of_range[:20].tolist(),
        )


def compare_arrays(
    left_name: str,
    left: np.ndarray,
    right_name: str,
    right: np.ndarray,
) -> None:
    left = canonical_indices(left)
    right = canonical_indices(right)

    equal = np.array_equal(left, right)

    print(
        f"compare {left_name} vs {right_name}:",
        "IDENTICAL" if equal else "DIFFERENT",
    )

    if equal:
        return

    left_set = set(left.tolist())
    right_set = set(right.tolist())

    print(
        "  only_left_count:",
        len(left_set - right_set),
    )
    print(
        "  only_right_count:",
        len(right_set - left_set),
    )
    print(
        "  only_left_preview:",
        sorted(left_set - right_set)[:20],
    )
    print(
        "  only_right_preview:",
        sorted(right_set - left_set)[:20],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify V3 built-in split integrity and model consistency."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    model_dirs = args.model_dir or DEFAULT_MODEL_DIRS

    split_path = data_dir / "split.npy"
    run_path = data_dir / "run_id.npy"
    graph_path = data_dir / "y_graph.npy"

    for required_path in [split_path, run_path, graph_path]:
        if not required_path.is_file():
            raise SystemExit(
                f"Required file missing: {required_path}"
            )

    split = np.asarray(load_array(split_path)).astype(str).reshape(-1)
    run_id = np.asarray(load_array(run_path)).astype(str).reshape(-1)
    y_graph = np.asarray(load_array(graph_path)).astype(np.int64).reshape(-1)

    sample_count = len(split)

    if len(run_id) != sample_count or len(y_graph) != sample_count:
        raise SystemExit(
            "split, run_id and y_graph do not have equal lengths"
        )

    print("data_dir:", data_dir)
    print("sample_count:", sample_count)
    print("split_value_counts:", dict(Counter(split.tolist())))

    accepted_splits = {"train", "val", "test"}
    observed_splits = set(split.tolist())
    unknown_splits = observed_splits - accepted_splits

    print("observed_splits:", sorted(observed_splits))
    print("unknown_splits:", sorted(unknown_splits))

    expected = {
        "train_idx": np.where(split == "train")[0].astype(np.int64),
        "val_idx": np.where(split == "val")[0].astype(np.int64),
        "test_idx": np.where(split == "test")[0].astype(np.int64),
    }

    print("\n" + "=" * 80)
    print("EXPECTED DATASET-DERIVED INDICES")
    print("=" * 80)

    for key, values in expected.items():
        inspect_index_set(key, values, sample_count)

    print("\n" + "=" * 80)
    print("EXPECTED SPLIT DISJOINTNESS")
    print("=" * 80)

    train_set = set(expected["train_idx"].tolist())
    val_set = set(expected["val_idx"].tolist())
    test_set = set(expected["test_idx"].tolist())

    print("train_val_overlap:", len(train_set & val_set))
    print("train_test_overlap:", len(train_set & test_set))
    print("val_test_overlap:", len(val_set & test_set))

    covered = train_set | val_set | test_set

    print("covered_sample_count:", len(covered))
    print("uncovered_sample_count:", sample_count - len(covered))
    print("covers_every_sample:", len(covered) == sample_count)

    print("\n" + "=" * 80)
    print("RUN-LEVEL SPLIT MEMBERSHIP")
    print("=" * 80)

    cross_split_runs: list[tuple[str, list[str]]] = []

    for current_run in sorted(np.unique(run_id).tolist()):
        memberships = sorted(
            np.unique(split[run_id == current_run]).tolist()
        )

        if len(memberships) != 1:
            cross_split_runs.append((current_run, memberships))

        print(
            f"{current_run}\t"
            f"samples={int((run_id == current_run).sum())}\t"
            f"splits={memberships}"
        )

    print("cross_split_run_count:", len(cross_split_runs))
    print("cross_split_runs:", cross_split_runs)

    print("\n" + "=" * 80)
    print("SPLIT CLASS AND RUN COUNTS")
    print("=" * 80)

    for split_name in ["train", "val", "test"]:
        mask = split == split_name

        print(
            split_name,
            {
                "samples": int(mask.sum()),
                "runs": int(len(np.unique(run_id[mask]))),
                "normal_samples": int((y_graph[mask] == 0).sum()),
                "attack_samples": int((y_graph[mask] == 1).sum()),
            },
        )

    loaded_model_splits: dict[str, dict[str, np.ndarray]] = {}

    for model_dir in model_dirs:
        model_dir = model_dir.resolve()
        split_file = model_dir / "splits.npz"

        print("\n" + "=" * 80)
        print("MODEL:", model_dir)
        print("=" * 80)

        if not split_file.is_file():
            print("CRITICAL: splits.npz missing:", split_file)
            continue

        stored = np.load(split_file, allow_pickle=False)
        print("split_file:", split_file)
        print("keys:", stored.files)

        current: dict[str, np.ndarray] = {}

        for key in ["train_idx", "val_idx", "test_idx"]:
            if key not in stored.files:
                print("CRITICAL: missing key:", key)
                continue

            values = canonical_indices(stored[key])
            current[key] = values

            inspect_index_set(
                f"{model_dir.name}.{key}",
                values,
                sample_count,
            )

            compare_arrays(
                f"dataset.{key}",
                expected[key],
                f"{model_dir.name}.{key}",
                values,
            )

        loaded_model_splits[model_dir.name] = current

    print("\n" + "=" * 80)
    print("CROSS-MODEL COMPARISON")
    print("=" * 80)

    model_names = list(loaded_model_splits)

    if model_names:
        reference_name = model_names[0]
        reference = loaded_model_splits[reference_name]

        for other_name in model_names[1:]:
            other = loaded_model_splits[other_name]

            for key in ["train_idx", "val_idx", "test_idx"]:
                if key not in reference or key not in other:
                    print(
                        f"cannot_compare {reference_name} vs "
                        f"{other_name} for {key}"
                    )
                    continue

                compare_arrays(
                    f"{reference_name}.{key}",
                    reference[key],
                    f"{other_name}.{key}",
                    other[key],
                )


if __name__ == "__main__":
    main()
