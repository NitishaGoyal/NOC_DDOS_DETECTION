#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_FILES = [
    "x.npy",
    "y_graph.npy",
    "y_node.npy",
    "edge_index.npy",
    "run_id.npy",
    "end_epoch.npy",
    "attackers.npy",
    "split.npy",
    "feature_cols.npy",
    "profile.npy",
    "active_cores.npy",
    "strength.npy",
    "seed.npy",
]


def load_array(path: Path) -> np.ndarray:
    """
    Load numeric arrays through mmap where possible.

    Object arrays cannot be memory-mapped, so those are loaded normally.
    """
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (ValueError, TypeError):
        return np.load(path, allow_pickle=True)


def format_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return np.array2string(
            value,
            threshold=20,
            edgeitems=5,
            separator=", ",
        )

    if isinstance(value, np.generic):
        value = value.item()

    return repr(value)


def preview_array(array: np.ndarray, limit: int = 8) -> list[str]:
    if array.size == 0:
        return []

    flat = array.reshape(-1)
    count = min(limit, flat.size)

    return [format_value(flat[i]) for i in range(count)]


def exact_unique_count(
    array: np.ndarray,
    maximum_elements: int = 5_000_000,
) -> str:
    if array.size > maximum_elements:
        return f"skipped_exact_count(size={array.size})"

    try:
        return str(len(np.unique(np.asarray(array))))
    except Exception as exc:
        return f"ERROR({exc!r})"


def print_basic_array_info(name: str, array: np.ndarray) -> None:
    print(f"\nARRAY: {name}")
    print("-" * 80)
    print("shape:", array.shape)
    print("dtype:", array.dtype)
    print("ndim:", array.ndim)
    print("elements:", array.size)
    print("memory_bytes:", array.nbytes)
    print("memory_mib:", f"{array.nbytes / (1024 ** 2):.3f}")
    print("preview:", preview_array(array))

    if array.size == 0:
        return

    if np.issubdtype(array.dtype, np.number):
        numeric = np.asarray(array)

        if np.issubdtype(array.dtype, np.floating):
            finite_mask = np.isfinite(numeric)
            finite_count = int(finite_mask.sum())
            nan_count = int(np.isnan(numeric).sum())
            inf_count = int(np.isinf(numeric).sum())

            print("finite_count:", finite_count)
            print("nan_count:", nan_count)
            print("inf_count:", inf_count)

            if finite_count:
                finite_values = numeric[finite_mask]
                print("minimum:", float(finite_values.min()))
                print("maximum:", float(finite_values.max()))
                print("mean:", float(finite_values.mean()))
                print("std:", float(finite_values.std()))
        else:
            print("minimum:", int(numeric.min()))
            print("maximum:", int(numeric.max()))
            print("mean:", float(numeric.mean()))
            print("std:", float(numeric.std()))

        print("unique_values:", exact_unique_count(array))

    else:
        print("unique_values:", exact_unique_count(array))


def inspect_x(
    x: np.ndarray,
    chunk_size: int,
) -> None:
    print("\n" + "=" * 80)
    print("DETAILED X INSPECTION")
    print("=" * 80)

    print("x_shape:", x.shape)
    print("x_dtype:", x.dtype)
    print("x_memory_bytes:", x.nbytes)
    print("x_memory_gib:", f"{x.nbytes / (1024 ** 3):.4f}")

    if x.ndim < 2:
        print("ERROR: x has fewer than two dimensions")
        return

    feature_count = int(x.shape[-1])
    sample_count = int(x.shape[0])

    print("sample_count:", sample_count)

    if x.ndim == 4:
        print("interpreted_layout: [samples, routers, time, features]")
        print("router_count:", x.shape[1])
        print("temporal_length:", x.shape[2])
        print("feature_count:", x.shape[3])
    else:
        print("interpreted_layout: unknown")
        print("feature_axis: last")
        print("feature_count:", feature_count)

    minimum = np.full(feature_count, np.inf, dtype=np.float64)
    maximum = np.full(feature_count, -np.inf, dtype=np.float64)
    value_sum = np.zeros(feature_count, dtype=np.float64)
    value_sumsq = np.zeros(feature_count, dtype=np.float64)
    finite_count = np.zeros(feature_count, dtype=np.int64)
    nan_count = np.zeros(feature_count, dtype=np.int64)
    inf_count = np.zeros(feature_count, dtype=np.int64)
    zero_count = np.zeros(feature_count, dtype=np.int64)
    one_count = np.zeros(feature_count, dtype=np.int64)

    total_chunks = math.ceil(sample_count / chunk_size)
    progress_interval = max(1, total_chunks // 10)

    for chunk_number, start in enumerate(
        range(0, sample_count, chunk_size),
        start=1,
    ):
        end = min(start + chunk_size, sample_count)

        chunk = np.asarray(x[start:end], dtype=np.float64)
        flat = chunk.reshape(-1, feature_count)

        finite = np.isfinite(flat)
        nan_mask = np.isnan(flat)
        inf_mask = np.isinf(flat)

        nan_count += nan_mask.sum(axis=0)
        inf_count += inf_mask.sum(axis=0)
        zero_count += (flat == 0.0).sum(axis=0)
        one_count += (flat == 1.0).sum(axis=0)

        for feature_index in range(feature_count):
            valid = flat[finite[:, feature_index], feature_index]

            if valid.size == 0:
                continue

            minimum[feature_index] = min(
                minimum[feature_index],
                float(valid.min()),
            )
            maximum[feature_index] = max(
                maximum[feature_index],
                float(valid.max()),
            )
            value_sum[feature_index] += float(valid.sum())
            value_sumsq[feature_index] += float(np.square(valid).sum())
            finite_count[feature_index] += int(valid.size)

        if (
            chunk_number == 1
            or chunk_number == total_chunks
            or chunk_number % progress_interval == 0
        ):
            print(
                f"x_progress: chunk {chunk_number}/{total_chunks}, "
                f"samples {start}:{end}"
            )

    mean = np.divide(
        value_sum,
        finite_count,
        out=np.full(feature_count, np.nan),
        where=finite_count > 0,
    )

    variance = np.divide(
        value_sumsq,
        finite_count,
        out=np.full(feature_count, np.nan),
        where=finite_count > 0,
    ) - np.square(mean)

    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)

    positions_per_feature = int(np.prod(x.shape[:-1]))

    print("\nPER-FEATURE X STATISTICS")
    print(
        "feature\tminimum\tmaximum\tmean\tstd\tfinite\t"
        "nan_pct\tinf_pct\tzero_pct\tone_pct"
    )

    for feature_index in range(feature_count):
        print(
            f"{feature_index}\t"
            f"{minimum[feature_index]:.10g}\t"
            f"{maximum[feature_index]:.10g}\t"
            f"{mean[feature_index]:.10g}\t"
            f"{std[feature_index]:.10g}\t"
            f"{finite_count[feature_index]}\t"
            f"{100.0 * nan_count[feature_index] / positions_per_feature:.6f}\t"
            f"{100.0 * inf_count[feature_index] / positions_per_feature:.6f}\t"
            f"{100.0 * zero_count[feature_index] / positions_per_feature:.6f}\t"
            f"{100.0 * one_count[feature_index] / positions_per_feature:.6f}"
        )


def inspect_labels(arrays: dict[str, np.ndarray]) -> None:
    if "y_graph" not in arrays:
        return

    y_graph = np.asarray(arrays["y_graph"]).astype(np.int64).reshape(-1)

    print("\n" + "=" * 80)
    print("GRAPH LABEL ANALYSIS")
    print("=" * 80)

    graph_counts = Counter(y_graph.tolist())

    print("total_graphs:", len(y_graph))
    print("graph_label_counts:", dict(sorted(graph_counts.items())))
    print("normal_samples:", int((y_graph == 0).sum()))
    print("attack_samples:", int((y_graph == 1).sum()))

    if "split" in arrays:
        split = np.asarray(arrays["split"]).astype(str).reshape(-1)

        if len(split) != len(y_graph):
            print(
                "ERROR: split length differs from y_graph length:",
                len(split),
                len(y_graph),
            )
        else:
            print("\nGRAPH COUNTS BY SPLIT")

            for split_name in sorted(np.unique(split).tolist()):
                mask = split == split_name
                labels = y_graph[mask]

                print(
                    split_name,
                    {
                        "samples": int(mask.sum()),
                        "normal": int((labels == 0).sum()),
                        "attack": int((labels == 1).sum()),
                    },
                )

    if "y_node" not in arrays:
        return

    y_node = np.asarray(arrays["y_node"]).astype(np.int64)

    print("\n" + "=" * 80)
    print("NODE LABEL ANALYSIS")
    print("=" * 80)

    print("y_node_shape:", y_node.shape)
    print("total_node_labels:", y_node.size)
    print("positive_node_labels:", int(y_node.sum()))
    print("negative_node_labels:", int(y_node.size - y_node.sum()))

    attacker_count = y_node.sum(axis=1).astype(np.int64)
    attacker_distribution = Counter(attacker_count.tolist())

    print(
        "attacker_count_distribution:",
        dict(sorted(attacker_distribution.items())),
    )

    normal_positive_graphs = int(
        ((y_graph == 0) & (attacker_count > 0)).sum()
    )
    attack_zero_attacker_graphs = int(
        ((y_graph == 1) & (attacker_count == 0)).sum()
    )

    print(
        "normal_graphs_with_positive_node_labels:",
        normal_positive_graphs,
    )
    print(
        "attack_graphs_with_zero_attackers:",
        attack_zero_attacker_graphs,
    )

    if "split" in arrays:
        split = np.asarray(arrays["split"]).astype(str).reshape(-1)

        if len(split) == len(attacker_count):
            print("\nATTACKER COUNTS BY SPLIT")

            for split_name in sorted(np.unique(split).tolist()):
                counts = attacker_count[split == split_name]

                print(
                    split_name,
                    dict(sorted(Counter(counts.tolist()).items())),
                )


def inspect_runs(arrays: dict[str, np.ndarray]) -> None:
    if "run_id" not in arrays:
        return

    run_id = np.asarray(arrays["run_id"]).astype(str).reshape(-1)
    unique_runs, counts = np.unique(run_id, return_counts=True)

    print("\n" + "=" * 80)
    print("RUN ANALYSIS")
    print("=" * 80)

    print("sample_count:", len(run_id))
    print("unique_run_count:", len(unique_runs))
    print("minimum_samples_per_run:", int(counts.min()))
    print("maximum_samples_per_run:", int(counts.max()))
    print("median_samples_per_run:", float(np.median(counts)))

    print("\nRUN INDEX RANGES")

    for current_run, count in zip(unique_runs, counts):
        indices = np.flatnonzero(run_id == current_run)

        print(
            f"{current_run}\t"
            f"samples={int(count)}\t"
            f"first_index={int(indices[0])}\t"
            f"last_index={int(indices[-1])}"
        )

    if "split" in arrays:
        split = np.asarray(arrays["split"]).astype(str).reshape(-1)

        print("\nRUN SPLIT MEMBERSHIP")

        for current_run in unique_runs:
            memberships = sorted(
                np.unique(split[run_id == current_run]).tolist()
            )

            print(
                f"{current_run}\t"
                f"splits={memberships}"
            )


def inspect_end_epoch(arrays: dict[str, np.ndarray]) -> None:
    if "end_epoch" not in arrays:
        return

    end_epoch = np.asarray(arrays["end_epoch"]).reshape(-1)

    print("\n" + "=" * 80)
    print("END-EPOCH ANALYSIS")
    print("=" * 80)

    print("minimum_end_epoch:", end_epoch.min())
    print("maximum_end_epoch:", end_epoch.max())
    print("unique_end_epochs:", len(np.unique(end_epoch)))

    if "run_id" not in arrays:
        return

    run_id = np.asarray(arrays["run_id"]).astype(str).reshape(-1)
    unique_runs = np.unique(run_id)

    non_monotonic_runs: list[str] = []
    duplicate_epoch_runs: list[str] = []
    common_steps: Counter[int | float] = Counter()

    for current_run in unique_runs:
        epochs = end_epoch[run_id == current_run]

        if len(epochs) <= 1:
            continue

        differences = np.diff(epochs)

        if np.any(differences < 0):
            non_monotonic_runs.append(current_run)

        if len(np.unique(epochs)) != len(epochs):
            duplicate_epoch_runs.append(current_run)

        for difference in differences:
            try:
                common_steps[difference.item()] += 1
            except AttributeError:
                common_steps[difference] += 1

    print("non_monotonic_run_count:", len(non_monotonic_runs))
    print("non_monotonic_runs:", non_monotonic_runs)
    print(
        "runs_with_duplicate_end_epochs_count:",
        len(duplicate_epoch_runs),
    )
    print(
        "runs_with_duplicate_end_epochs:",
        duplicate_epoch_runs,
    )
    print("most_common_epoch_steps:", common_steps.most_common(20))

    candidate_sample_ids = np.asarray(
        [
            f"{run}:{epoch}"
            for run, epoch in zip(run_id, end_epoch)
        ],
        dtype=object,
    )

    unique_sample_ids = len(np.unique(candidate_sample_ids))

    print(
        "candidate_sample_id_format:",
        "run_id:end_epoch",
    )
    print(
        "candidate_sample_id_unique:",
        unique_sample_ids == len(candidate_sample_ids),
    )
    print(
        "candidate_sample_id_unique_count:",
        unique_sample_ids,
    )
    print(
        "candidate_sample_id_total_count:",
        len(candidate_sample_ids),
    )

    if unique_sample_ids != len(candidate_sample_ids):
        values, counts = np.unique(
            candidate_sample_ids,
            return_counts=True,
        )

        duplicates = values[counts > 1]

        print(
            "duplicate_candidate_sample_ids_preview:",
            duplicates[:20].tolist(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only inspection of the directory-backed V3 dataset."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()

    if not data_dir.is_dir():
        raise SystemExit(
            f"Dataset directory does not exist: {data_dir}"
        )

    print("data_dir:", data_dir)
    print("expected_files:")

    for filename in EXPECTED_FILES:
        print(
            f"  {filename}:",
            "FOUND" if (data_dir / filename).is_file() else "MISSING",
        )

    npy_files = sorted(data_dir.glob("*.npy"))

    print("\nall_npy_files:")

    for path in npy_files:
        print(" ", path.name)

    arrays: dict[str, np.ndarray] = {}

    for path in npy_files:
        key = path.stem

        try:
            arrays[key] = load_array(path)
        except Exception as exc:
            print(
                f"\nARRAY: {key}\n"
                f"ERROR loading {path}: {exc!r}"
            )

    for key in sorted(arrays):
        if key == "x":
            continue

        print_basic_array_info(key, arrays[key])

    if "feature_cols" in arrays:
        feature_cols = np.asarray(arrays["feature_cols"]).reshape(-1)

        print("\n" + "=" * 80)
        print("FEATURE COLUMN ORDER")
        print("=" * 80)

        for index, feature_name in enumerate(feature_cols):
            print(index, format_value(feature_name))

    if "x" in arrays:
        inspect_x(
            arrays["x"],
            chunk_size=args.chunk_size,
        )

    inspect_labels(arrays)
    inspect_runs(arrays)
    inspect_end_epoch(arrays)


if __name__ == "__main__":
    main()
