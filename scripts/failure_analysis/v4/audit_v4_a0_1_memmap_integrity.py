#!/usr/bin/env python3
"""
V4-A0.1: full memmap integrity and structural-array audit.

Read-only scientific audit for the completed V4 temporal-graph memmap.

This stage:
- opens all NumPy arrays with mmap_mode="r";
- validates shapes, dtypes, alignment, run boundaries, labels, splits, and topology;
- scans the complete x.npy tensor once in bounded chunks;
- never loads the complete x.npy tensor into RAM;
- performs no model construction, inference, thresholding, or training;
- writes only to the requested report directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_VERSION = "1.0.0"

REQUIRED_FILES = [
    "x.npy",
    "y_graph.npy",
    "y_node.npy",
    "edge_index.npy",
    "end_epoch.npy",
    "run_index.npy",
    "split_id.npy",
    "profile_id.npy",
    "attack_kind_id.npy",
    "attacker_count.npy",
    "strength.npy",
    "metadata.json",
    "completed_runs.txt",
]

EXPECTED_DTYPES = {
    "x.npy": "float32",
    "y_graph.npy": "int64",
    "y_node.npy": "float32",
    "edge_index.npy": "int64",
    "end_epoch.npy": "int32",
    "run_index.npy": "int32",
    "split_id.npy": "int16",
    "profile_id.npy": "int16",
    "attack_kind_id.npy": "int16",
    "attacker_count.npy": "int16",
    "strength.npy": "int16",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def value_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(v.item() if isinstance(v, np.generic) else v): int(c)
            for v, c in zip(unique, counts)}


def load_memmap(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


def expected_mesh_edges(rows: int = 4, cols: int = 4) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    node_count = rows * cols

    for node in range(node_count):
        edges.add((node, node))
        r, c = divmod(node, cols)

        if r > 0:
            edges.add((node, node - cols))
        if r + 1 < rows:
            edges.add((node, node + cols))
        if c > 0:
            edges.add((node, node - 1))
        if c + 1 < cols:
            edges.add((node, node + 1))

    return edges



def decode_code_map(metadata: Any, field: str) -> dict[int, str]:
    """Return an integer-code -> human-readable-name map when metadata provides one."""
    if not isinstance(metadata, dict):
        return {}

    code_maps = metadata.get("code_maps")
    if not isinstance(code_maps, dict):
        return {}

    raw = code_maps.get(field)
    if not isinstance(raw, dict):
        return {}

    decoded: dict[int, str] = {}
    for key, value in raw.items():
        try:
            decoded[int(value)] = str(key)
            continue
        except (TypeError, ValueError):
            pass

        try:
            decoded[int(key)] = str(value)
        except (TypeError, ValueError):
            continue

    return decoded



def metadata_summary(metadata: Any, expected_runs: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "top_level_type": type(metadata).__name__,
        "top_level_keys": sorted(metadata.keys()) if isinstance(metadata, dict) else [],
        "selected_fields": {},
        "candidate_run_sequences": [],
    }

    if not isinstance(metadata, dict):
        return summary

    interesting = {
        "total_samples",
        "samples_per_run",
        "window_epochs",
        "window_length",
        "stride_epochs",
        "stride",
        "num_routers",
        "max_epochs_per_run",
        "selected_runs",
        "feature_cols",
        "code_maps",
        "split_counts",
        "run_count",
        "num_runs",
    }

    for key, value in metadata.items():
        if key in interesting:
            if isinstance(value, list) and len(value) > 100:
                summary["selected_fields"][key] = {
                    "type": "list",
                    "length": len(value),
                    "preview": value[:5],
                }
            else:
                summary["selected_fields"][key] = value

        if isinstance(value, list) and len(value) == expected_runs:
            summary["candidate_run_sequences"].append({
                "key": key,
                "length": len(value),
                "preview": value[:5],
            })

    return summary


def scan_x(
    x: np.ndarray,
    chunk_size: int,
    feature_count: int,
) -> dict[str, Any]:
    total_samples = int(x.shape[0])
    feature_min = np.full(feature_count, np.inf, dtype=np.float64)
    feature_max = np.full(feature_count, -np.inf, dtype=np.float64)
    feature_sum = np.zeros(feature_count, dtype=np.float64)
    feature_sumsq = np.zeros(feature_count, dtype=np.float64)
    finite_count = np.zeros(feature_count, dtype=np.int64)
    nonfinite_count = np.zeros(feature_count, dtype=np.int64)
    below_zero_count = np.zeros(feature_count, dtype=np.int64)
    above_one_count = np.zeros(feature_count, dtype=np.int64)
    exact_zero_count = np.zeros(feature_count, dtype=np.int64)
    exact_one_count = np.zeros(feature_count, dtype=np.int64)

    total_chunks = math.ceil(total_samples / chunk_size)
    print(
        f"x_scan_start samples={total_samples} chunk_size={chunk_size} "
        f"chunks={total_chunks}",
        flush=True,
    )

    for chunk_index, start in enumerate(range(0, total_samples, chunk_size), start=1):
        end = min(total_samples, start + chunk_size)
        chunk = np.asarray(x[start:end])

        finite = np.isfinite(chunk)
        reduce_axes = (0, 1, 2)

        per_feature_nonfinite = np.count_nonzero(~finite, axis=reduce_axes)
        nonfinite_count += per_feature_nonfinite.astype(np.int64)
        finite_count += np.count_nonzero(finite, axis=reduce_axes).astype(np.int64)

        below_zero_count += np.count_nonzero(
            finite & (chunk < 0.0), axis=reduce_axes
        ).astype(np.int64)
        above_one_count += np.count_nonzero(
            finite & (chunk > 1.0), axis=reduce_axes
        ).astype(np.int64)
        exact_zero_count += np.count_nonzero(
            finite & (chunk == 0.0), axis=reduce_axes
        ).astype(np.int64)
        exact_one_count += np.count_nonzero(
            finite & (chunk == 1.0), axis=reduce_axes
        ).astype(np.int64)

        if int(per_feature_nonfinite.sum()) == 0:
            feature_min = np.minimum(
                feature_min,
                np.min(chunk, axis=reduce_axes).astype(np.float64),
            )
            feature_max = np.maximum(
                feature_max,
                np.max(chunk, axis=reduce_axes).astype(np.float64),
            )
            feature_sum += np.sum(chunk, axis=reduce_axes, dtype=np.float64)
            feature_sumsq += np.sum(
                np.square(chunk, dtype=np.float64),
                axis=reduce_axes,
                dtype=np.float64,
            )
        else:
            for feature_index in range(feature_count):
                view = chunk[..., feature_index]
                valid = np.isfinite(view)
                if not np.any(valid):
                    continue
                values = view[valid].astype(np.float64, copy=False)
                feature_min[feature_index] = min(
                    feature_min[feature_index],
                    float(values.min()),
                )
                feature_max[feature_index] = max(
                    feature_max[feature_index],
                    float(values.max()),
                )
                feature_sum[feature_index] += float(values.sum(dtype=np.float64))
                feature_sumsq[feature_index] += float(
                    np.square(values, dtype=np.float64).sum(dtype=np.float64)
                )

        if (
            chunk_index == 1
            or chunk_index == total_chunks
            or chunk_index % max(1, total_chunks // 20) == 0
        ):
            print(
                f"x_scan_progress chunk={chunk_index}/{total_chunks} "
                f"samples={end}/{total_samples}",
                flush=True,
            )

    means = np.divide(
        feature_sum,
        finite_count,
        out=np.full(feature_count, np.nan, dtype=np.float64),
        where=finite_count > 0,
    )
    second_moment = np.divide(
        feature_sumsq,
        finite_count,
        out=np.full(feature_count, np.nan, dtype=np.float64),
        where=finite_count > 0,
    )
    variance = np.maximum(0.0, second_moment - np.square(means))
    std = np.sqrt(variance)

    total_positions_per_feature = int(np.prod(x.shape[:-1]))

    rows = []
    for index in range(feature_count):
        rows.append({
            "feature_index": index,
            "minimum": float(feature_min[index]),
            "maximum": float(feature_max[index]),
            "mean": float(means[index]),
            "std": float(std[index]),
            "finite_count": int(finite_count[index]),
            "nonfinite_count": int(nonfinite_count[index]),
            "below_zero_count": int(below_zero_count[index]),
            "above_one_count": int(above_one_count[index]),
            "exact_zero_count": int(exact_zero_count[index]),
            "exact_one_count": int(exact_one_count[index]),
            "fraction_equal_zero": (
                float(exact_zero_count[index] / total_positions_per_feature)
                if total_positions_per_feature else 0.0
            ),
            "fraction_equal_one": (
                float(exact_one_count[index] / total_positions_per_feature)
                if total_positions_per_feature else 0.0
            ),
        })

    print("x_scan_complete", flush=True)

    return {
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "total_samples_scanned": total_samples,
        "total_positions_per_feature": total_positions_per_feature,
        "all_values_finite": int(nonfinite_count.sum()) == 0,
        "all_values_in_documented_range_0_1": (
            int(below_zero_count.sum()) == 0
            and int(above_one_count.sum()) == 0
        ),
        "nonfinite_total": int(nonfinite_count.sum()),
        "below_zero_total": int(below_zero_count.sum()),
        "above_one_total": int(above_one_count.sum()),
        "features": rows,
    }


def run_structure_audit(
    arrays: dict[str, np.ndarray],
    expected_runs: int,
    expected_samples_per_run: int,
    expected_first_end_epoch: int,
    expected_last_end_epoch: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_index = np.asarray(arrays["run_index.npy"])
    total_samples = len(run_index)

    change_points = np.flatnonzero(run_index[1:] != run_index[:-1]) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [total_samples]))
    segment_values = run_index[starts]
    segment_counts = ends - starts

    expected_epoch_sequence = np.arange(
        expected_first_end_epoch,
        expected_last_end_epoch + 1,
        dtype=arrays["end_epoch.npy"].dtype,
    )

    per_run_rows: list[dict[str, Any]] = []
    repeated_segment_values = [
        int(value)
        for value, count in Counter(segment_values.tolist()).items()
        if count > 1
    ]

    run_constant_failures = defaultdict(list)
    end_epoch_failures: list[int] = []
    label_failures: list[int] = []
    node_label_failures: list[int] = []

    for segment_number, (start, end, run_value) in enumerate(
        zip(starts, ends, segment_values)
    ):
        count = int(end - start)

        end_epoch_view = np.asarray(arrays["end_epoch.npy"][start:end])
        end_epoch_ok = (
            count == len(expected_epoch_sequence)
            and np.array_equal(end_epoch_view, expected_epoch_sequence)
        )
        if not end_epoch_ok:
            end_epoch_failures.append(int(run_value))

        constants: dict[str, Any] = {}
        for name in [
            "y_graph.npy",
            "split_id.npy",
            "profile_id.npy",
            "attack_kind_id.npy",
            "attacker_count.npy",
            "strength.npy",
        ]:
            view = np.asarray(arrays[name][start:end])
            unique = np.unique(view)
            constants[name] = {
                "constant": len(unique) == 1,
                "unique_values": unique.tolist(),
            }
            if len(unique) != 1:
                run_constant_failures[name].append(int(run_value))

        y_node_view = np.asarray(arrays["y_node.npy"][start:end])
        node_constant = np.all(y_node_view == y_node_view[0], axis=None)
        if not node_constant:
            run_constant_failures["y_node.npy"].append(int(run_value))

        graph_value = int(arrays["y_graph.npy"][start])
        attacker_count_value = int(arrays["attacker_count.npy"][start])
        node_bitmap = np.asarray(arrays["y_node.npy"][start])
        node_positive_count = int(np.count_nonzero(node_bitmap == 1.0))

        graph_count_ok = (
            (graph_value == 0 and attacker_count_value == 0)
            or (graph_value == 1 and attacker_count_value >= 1)
        )
        if not graph_count_ok:
            label_failures.append(int(run_value))

        node_count_ok = node_positive_count == attacker_count_value
        if not node_count_ok:
            node_label_failures.append(int(run_value))

        per_run_rows.append({
            "segment_number": segment_number,
            "run_index": int(run_value),
            "start_sample": int(start),
            "end_sample_exclusive": int(end),
            "sample_count": count,
            "samples_per_run_ok": count == expected_samples_per_run,
            "end_epoch_sequence_ok": end_epoch_ok,
            "y_graph": graph_value,
            "split_id": int(arrays["split_id.npy"][start]),
            "profile_id": int(arrays["profile_id.npy"][start]),
            "attack_kind_id": int(arrays["attack_kind_id.npy"][start]),
            "attacker_count": attacker_count_value,
            "strength": int(arrays["strength.npy"][start]),
            "y_node_constant": bool(node_constant),
            "node_positive_count": node_positive_count,
            "graph_attacker_count_consistent": graph_count_ok,
            "node_attacker_count_consistent": node_count_ok,
        })

    expected_run_values = np.arange(expected_runs, dtype=segment_values.dtype)

    audit = {
        "total_samples": total_samples,
        "segment_count": int(len(starts)),
        "unique_run_index_count": int(len(np.unique(run_index))),
        "segment_values": segment_values.tolist(),
        "run_indices_are_0_through_expected_minus_1": (
            len(segment_values) == expected_runs
            and np.array_equal(segment_values, expected_run_values)
        ),
        "no_repeated_run_segments": len(repeated_segment_values) == 0,
        "repeated_run_segment_values": repeated_segment_values,
        "all_segments_have_expected_sample_count": bool(
            np.all(segment_counts == expected_samples_per_run)
        ),
        "segment_sample_count_min": int(segment_counts.min()) if len(segment_counts) else 0,
        "segment_sample_count_max": int(segment_counts.max()) if len(segment_counts) else 0,
        "end_epoch_sequence_failures": end_epoch_failures,
        "run_constant_failures": dict(run_constant_failures),
        "graph_attacker_count_failures": label_failures,
        "node_attacker_count_failures": node_label_failures,
        "expected_end_epoch_start": expected_first_end_epoch,
        "expected_end_epoch_end": expected_last_end_epoch,
    }
    return audit, per_run_rows


def edge_audit(edge_index: np.ndarray, expected_nodes: int) -> dict[str, Any]:
    edge = np.asarray(edge_index)
    edges = [(int(src), int(dst)) for src, dst in edge.T.tolist()]
    unique_edges = set(edges)

    self_loops = {(src, dst) for src, dst in unique_edges if src == dst}
    nonself = {(src, dst) for src, dst in unique_edges if src != dst}
    reciprocal = {(src, dst) for src, dst in nonself if (dst, src) in nonself}

    adjacency: dict[int, set[int]] = {node: set() for node in range(expected_nodes)}
    for src, dst in nonself:
        if 0 <= src < expected_nodes and 0 <= dst < expected_nodes:
            adjacency[src].add(dst)
            adjacency[dst].add(src)

    visited: set[int] = set()
    if expected_nodes:
        stack = [0]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adjacency[node] - visited)

    observed_degrees = {str(node): len(adjacency[node]) for node in range(expected_nodes)}
    expected_degrees = {}
    for node in range(expected_nodes):
        r, c = divmod(node, 4)
        degree = 4
        if r in (0, 3):
            degree -= 1
        if c in (0, 3):
            degree -= 1
        expected_degrees[str(node)] = degree

    exact_expected = expected_mesh_edges(4, 4)

    return {
        "shape": list(edge.shape),
        "dtype": str(edge.dtype),
        "edge_count": len(edges),
        "unique_edge_count": len(unique_edges),
        "duplicate_edge_count": len(edges) - len(unique_edges),
        "minimum_router_id": int(edge.min()) if edge.size else None,
        "maximum_router_id": int(edge.max()) if edge.size else None,
        "self_loop_count": len(self_loops),
        "nonself_directed_edge_count": len(nonself),
        "reciprocal_nonself_edge_count": len(reciprocal),
        "all_nonself_edges_reciprocal": len(reciprocal) == len(nonself),
        "all_nodes_connected": len(visited) == expected_nodes,
        "visited_nodes": sorted(visited),
        "observed_undirected_degrees": observed_degrees,
        "expected_undirected_degrees": expected_degrees,
        "degrees_match_4x4_mesh": observed_degrees == expected_degrees,
        "exact_row_major_4x4_edge_set_match": unique_edges == exact_expected,
        "missing_expected_edges": sorted(exact_expected - unique_edges),
        "unexpected_edges": sorted(unique_edges - exact_expected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-samples", type=int, default=2_990_044)
    parser.add_argument("--expected-runs", type=int, default=908)
    parser.add_argument("--expected-samples-per-run", type=int, default=3293)
    parser.add_argument("--expected-nodes", type=int, default=16)
    parser.add_argument("--expected-window", type=int, default=8)
    parser.add_argument("--expected-features", type=int, default=24)
    parser.add_argument("--expected-normal-runs", type=int, default=300)
    parser.add_argument("--expected-attack-runs", type=int, default=608)
    parser.add_argument("--expected-train-runs", type=int, default=649)
    parser.add_argument("--expected-val-runs", type=int, default=139)
    parser.add_argument("--expected-test-runs", type=int, default=120)
    parser.add_argument("--expected-single-attacker-runs", type=int, default=288)
    parser.add_argument("--expected-two-attacker-runs", type=int, default=120)
    parser.add_argument("--expected-three-attacker-runs", type=int, default=120)
    parser.add_argument("--expected-four-attacker-runs", type=int, default=80)
    parser.add_argument("--expected-first-end-epoch", type=int, default=7)
    parser.add_argument("--expected-last-end-epoch", type=int, default=3299)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not data_dir.is_dir():
        raise SystemExit(f"STOP: data directory does not exist: {data_dir}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"STOP: output directory already non-empty: {output_dir}")

    missing = [name for name in REQUIRED_FILES if not (data_dir / name).is_file()]
    if missing:
        raise SystemExit(f"STOP: required files missing: {missing}")

    source_before = {
        "size": data_dir.stat().st_size,
        "mtime_ns": data_dir.stat().st_mtime_ns,
        "mode": data_dir.stat().st_mode,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    started = utc_now()
    print("V4-A0.1 MEMMAP INTEGRITY AUDIT START", flush=True)
    print(f"data_dir={data_dir}", flush=True)
    print(f"output_dir={output_dir}", flush=True)

    arrays: dict[str, np.ndarray] = {}
    for name in EXPECTED_DTYPES:
        arrays[name] = load_memmap(data_dir / name)

    with (data_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    completed_runs = [
        line.strip()
        for line in (data_dir / "completed_runs.txt").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line.strip()
    ]

    expected_shapes = {
        "x.npy": [
            args.expected_samples,
            args.expected_nodes,
            args.expected_window,
            args.expected_features,
        ],
        "y_graph.npy": [args.expected_samples],
        "y_node.npy": [args.expected_samples, args.expected_nodes],
        "edge_index.npy": [2, 64],
        "end_epoch.npy": [args.expected_samples],
        "run_index.npy": [args.expected_samples],
        "split_id.npy": [args.expected_samples],
        "profile_id.npy": [args.expected_samples],
        "attack_kind_id.npy": [args.expected_samples],
        "attacker_count.npy": [args.expected_samples],
        "strength.npy": [args.expected_samples],
    }

    array_rows: list[dict[str, Any]] = []
    shape_dtype_checks: dict[str, bool] = {}

    for name, array in arrays.items():
        expected_shape = expected_shapes[name]
        expected_dtype = EXPECTED_DTYPES[name]
        shape_ok = list(array.shape) == expected_shape
        dtype_ok = str(array.dtype) == expected_dtype
        mmap_ok = isinstance(array, np.memmap)

        shape_dtype_checks[f"{name}:shape"] = shape_ok
        shape_dtype_checks[f"{name}:dtype"] = dtype_ok
        shape_dtype_checks[f"{name}:memmap"] = mmap_ok

        array_rows.append({
            "name": name,
            "shape": list(array.shape),
            "expected_shape": expected_shape,
            "shape_ok": shape_ok,
            "dtype": str(array.dtype),
            "expected_dtype": expected_dtype,
            "dtype_ok": dtype_ok,
            "memory_mapped": mmap_ok,
            "file_size_bytes": int((data_dir / name).stat().st_size),
        })

    array_audit = {
        "arrays": array_rows,
        "checks": shape_dtype_checks,
        "all_shape_dtype_memmap_checks_pass": all(shape_dtype_checks.values()),
    }
    write_json(output_dir / "v4_array_shape_dtype_audit.json", array_audit)

    x_stats = scan_x(
        arrays["x.npy"],
        chunk_size=args.chunk_size,
        feature_count=args.expected_features,
    )
    write_json(output_dir / "v4_chunk_feature_stats.json", x_stats)

    # Small/medium label arrays are safe to inspect in full.
    y_graph = np.asarray(arrays["y_graph.npy"])
    y_node = np.asarray(arrays["y_node.npy"])
    attacker_count = np.asarray(arrays["attacker_count.npy"])

    y_graph_binary = bool(np.all((y_graph == 0) | (y_graph == 1)))
    y_node_binary = bool(np.all((y_node == 0.0) | (y_node == 1.0)))

    graph_counts = value_counts(y_graph)
    node_positive_total = int(np.count_nonzero(y_node == 1.0))
    node_positive_by_router = (
        np.count_nonzero(y_node == 1.0, axis=0).astype(np.int64).tolist()
    )
    attacker_count_counts = value_counts(attacker_count)

    label_audit = {
        "y_graph_binary": y_graph_binary,
        "y_node_binary": y_node_binary,
        "y_graph_value_counts": graph_counts,
        "attacker_count_value_counts": attacker_count_counts,
        "node_positive_total": node_positive_total,
        "node_positive_by_router": node_positive_by_router,
        "all_routers_have_positive_attacker_labels": all(
            count > 0 for count in node_positive_by_router
        ),
        "normal_sample_count": int(np.count_nonzero(y_graph == 0)),
        "attack_sample_count": int(np.count_nonzero(y_graph == 1)),
    }
    write_json(output_dir / "v4_label_count_audit.json", label_audit)

    run_audit, per_run_rows = run_structure_audit(
        arrays=arrays,
        expected_runs=args.expected_runs,
        expected_samples_per_run=args.expected_samples_per_run,
        expected_first_end_epoch=args.expected_first_end_epoch,
        expected_last_end_epoch=args.expected_last_end_epoch,
    )
    write_json(output_dir / "v4_run_boundary_audit.json", run_audit)

    per_run_csv = output_dir / "v4_run_level_summary.csv"
    with per_run_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_run_rows)

    split_values = np.asarray(arrays["split_id.npy"])
    split_sample_counts = value_counts(split_values)
    split_run_counts = dict(Counter(str(row["split_id"]) for row in per_run_rows))
    split_code_map = decode_code_map(metadata, "split")

    named_split_sample_counts = {
        split_code_map.get(int(code), f"code_{code}"): int(count)
        for code, count in split_sample_counts.items()
    }
    named_split_run_counts = {
        split_code_map.get(int(code), f"code_{code}"): int(count)
        for code, count in split_run_counts.items()
    }

    expected_split_run_counts = {
        "train": args.expected_train_runs,
        "val": args.expected_val_runs,
        "test": args.expected_test_runs,
    }
    expected_split_sample_counts = {
        name: count * args.expected_samples_per_run
        for name, count in expected_split_run_counts.items()
    }

    split_names_available = {"train", "val", "test"}.issubset(
        set(named_split_run_counts)
    )

    split_audit = {
        "split_code_map": split_code_map,
        "sample_counts_by_split_id": split_sample_counts,
        "run_counts_by_split_id": split_run_counts,
        "sample_counts_by_split_name": named_split_sample_counts,
        "run_counts_by_split_name": named_split_run_counts,
        "expected_run_counts_by_split_name": expected_split_run_counts,
        "expected_sample_counts_by_split_name": expected_split_sample_counts,
        "split_names_train_val_test_available": split_names_available,
        "named_run_counts_match_expected": (
            split_names_available
            and all(
                named_split_run_counts.get(name) == expected
                for name, expected in expected_split_run_counts.items()
            )
        ),
        "named_sample_counts_match_expected": (
            split_names_available
            and all(
                named_split_sample_counts.get(name) == expected
                for name, expected in expected_split_sample_counts.items()
            )
        ),
        "run_count_multiset_matches_expected": (
            sorted(split_run_counts.values())
            == sorted(expected_split_run_counts.values())
        ),
        "sample_count_multiset_matches_expected": (
            sorted(split_sample_counts.values())
            == sorted(expected_split_sample_counts.values())
        ),
        "profile_sample_counts": value_counts(np.asarray(arrays["profile_id.npy"])),
        "attack_kind_sample_counts": value_counts(
            np.asarray(arrays["attack_kind_id.npy"])
        ),
        "strength_sample_counts": value_counts(np.asarray(arrays["strength.npy"])),
    }
    write_json(output_dir / "v4_split_count_audit.json", split_audit)

    graph_run_counts = Counter(int(row["y_graph"]) for row in per_run_rows)
    attacker_run_counts = Counter(int(row["attacker_count"]) for row in per_run_rows)
    expected_attacker_run_counts = {
        0: args.expected_normal_runs,
        1: args.expected_single_attacker_runs,
        2: args.expected_two_attacker_runs,
        3: args.expected_three_attacker_runs,
        4: args.expected_four_attacker_runs,
    }

    distribution_audit = {
        "graph_run_counts": {str(k): int(v) for k, v in sorted(graph_run_counts.items())},
        "expected_graph_run_counts": {
            "0": args.expected_normal_runs,
            "1": args.expected_attack_runs,
        },
        "attacker_count_run_counts": {
            str(k): int(v) for k, v in sorted(attacker_run_counts.items())
        },
        "expected_attacker_count_run_counts": {
            str(k): int(v) for k, v in expected_attacker_run_counts.items()
        },
        "graph_run_counts_match_expected": (
            graph_run_counts.get(0, 0) == args.expected_normal_runs
            and graph_run_counts.get(1, 0) == args.expected_attack_runs
            and set(graph_run_counts).issubset({0, 1})
        ),
        "attacker_count_run_counts_match_expected": all(
            attacker_run_counts.get(count, 0) == expected
            for count, expected in expected_attacker_run_counts.items()
        ) and set(attacker_run_counts).issubset(set(expected_attacker_run_counts)),
        "graph_sample_counts_match_expected": (
            label_audit["normal_sample_count"]
            == args.expected_normal_runs * args.expected_samples_per_run
            and label_audit["attack_sample_count"]
            == args.expected_attack_runs * args.expected_samples_per_run
        ),
    }
    write_json(output_dir / "v4_scenario_distribution_audit.json", distribution_audit)

    topology_audit = edge_audit(
        arrays["edge_index.npy"],
        expected_nodes=args.expected_nodes,
    )
    write_json(output_dir / "v4_edge_index_audit.json", topology_audit)

    metadata_audit = {
        "metadata_sha256": sha256_file(data_dir / "metadata.json"),
        "completed_runs_sha256": sha256_file(data_dir / "completed_runs.txt"),
        "completed_run_line_count": len(completed_runs),
        "completed_run_unique_count": len(set(completed_runs)),
        "completed_runs_count_matches_expected": len(completed_runs) == args.expected_runs,
        "completed_runs_unique": len(set(completed_runs)) == len(completed_runs),
        "completed_runs_preview": completed_runs[:10],
        "metadata_summary": metadata_summary(metadata, args.expected_runs),
    }
    write_json(output_dir / "v4_metadata_summary.json", metadata_audit)

    source_after = {
        "size": data_dir.stat().st_size,
        "mtime_ns": data_dir.stat().st_mtime_ns,
        "mode": data_dir.stat().st_mode,
    }

    hard_checks = {
        "all_required_files_present": len(missing) == 0,
        "all_arrays_shape_dtype_memmap_valid": array_audit[
            "all_shape_dtype_memmap_checks_pass"
        ],
        "x_all_values_finite": x_stats["all_values_finite"],
        "x_all_values_in_documented_range_0_1": x_stats[
            "all_values_in_documented_range_0_1"
        ],
        "y_graph_binary": label_audit["y_graph_binary"],
        "y_node_binary": label_audit["y_node_binary"],
        "all_routers_have_attacker_labels": label_audit[
            "all_routers_have_positive_attacker_labels"
        ],
        "graph_sample_counts_match_expected": distribution_audit[
            "graph_sample_counts_match_expected"
        ],
        "graph_run_counts_match_expected": distribution_audit[
            "graph_run_counts_match_expected"
        ],
        "attacker_count_run_distribution_matches_expected": distribution_audit[
            "attacker_count_run_counts_match_expected"
        ],
        "split_code_map_contains_train_val_test": split_audit[
            "split_names_train_val_test_available"
        ],
        "split_run_counts_match_expected": split_audit[
            "named_run_counts_match_expected"
        ],
        "split_sample_counts_match_expected": split_audit[
            "named_sample_counts_match_expected"
        ],
        "run_segment_count_matches_expected": (
            run_audit["segment_count"] == args.expected_runs
        ),
        "unique_run_count_matches_expected": (
            run_audit["unique_run_index_count"] == args.expected_runs
        ),
        "run_indices_are_expected_sequence": run_audit[
            "run_indices_are_0_through_expected_minus_1"
        ],
        "no_repeated_run_segments": run_audit["no_repeated_run_segments"],
        "all_runs_have_expected_samples": run_audit[
            "all_segments_have_expected_sample_count"
        ],
        "all_end_epoch_sequences_valid": (
            len(run_audit["end_epoch_sequence_failures"]) == 0
        ),
        "run_level_metadata_constant": all(
            len(values) == 0
            for values in run_audit["run_constant_failures"].values()
        ),
        "graph_labels_match_attacker_counts": (
            len(run_audit["graph_attacker_count_failures"]) == 0
        ),
        "node_labels_match_attacker_counts": (
            len(run_audit["node_attacker_count_failures"]) == 0
        ),
        "completed_runs_count_matches_expected": metadata_audit[
            "completed_runs_count_matches_expected"
        ],
        "completed_runs_unique": metadata_audit["completed_runs_unique"],
        "edge_shape_valid": topology_audit["shape"] == [2, 64],
        "edge_ids_valid": (
            topology_audit["minimum_router_id"] == 0
            and topology_audit["maximum_router_id"] == args.expected_nodes - 1
        ),
        "edge_count_and_uniqueness_valid": (
            topology_audit["edge_count"] == 64
            and topology_audit["unique_edge_count"] == 64
        ),
        "edge_self_loop_count_valid": topology_audit["self_loop_count"] == 16,
        "edge_nonself_count_valid": (
            topology_audit["nonself_directed_edge_count"] == 48
        ),
        "edge_reciprocal": topology_audit["all_nonself_edges_reciprocal"],
        "edge_connected": topology_audit["all_nodes_connected"],
        "edge_degrees_match_mesh": topology_audit["degrees_match_4x4_mesh"],
        "edge_exact_row_major_mesh_match": topology_audit[
            "exact_row_major_4x4_edge_set_match"
        ],
        "source_root_metadata_unchanged": source_before == source_after,
    }

    warnings = {
        "feature_semantics_not_yet_audited": True,
        "background_family_leakage_not_yet_audited": True,
        "active_core_shortcut_not_yet_audited": True,
        "victim_metadata_not_yet_audited": True,
        "attack_kind_generator_semantics_not_yet_audited": True,
        "full_x_sha256_not_computed_in_this_stage": True,
    }

    hard_pass = all(hard_checks.values())
    verdict = (
        "V4_A0_1_MEMMAP_INTEGRITY_PASS"
        if hard_pass
        else "V4_A0_1_MEMMAP_INTEGRITY_FAIL"
    )

    report_md = output_dir / "v4_memmap_integrity_report.md"
    report_lines = [
        "# V4-A0.1 — Memmap Integrity and Structural Array Audit",
        "",
        f"- Generated: `{utc_now()}`",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Dataset: `{data_dir}`",
        f"- Verdict: **{verdict}**",
        "",
        "## Safety boundary",
        "",
        "- Full model inference performed: **False**",
        "- Model training performed: **False**",
        "- Threshold selection performed: **False**",
        "- Dataset tree modified: **False**",
        "- Complete `x.npy` loaded into RAM: **False**",
        "- Validation/test samples structurally inspected: **True**",
        "- Validation/test predictions accessed: **False**",
        "",
        "## Hard checks",
        "",
    ]
    for name, passed in hard_checks.items():
        report_lines.append(f"- [{'PASS' if passed else 'FAIL'}] `{name}`")

    report_lines += [
        "",
        "## Documented remaining gates",
        "",
        "A pass here authorizes only V4-A0.2 manifest-to-array alignment.",
        "It does not authorize model training.",
        "",
    ]
    report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "stage": "V4-A0.1",
        "script_version": SCRIPT_VERSION,
        "started_utc": started,
        "completed_utc": utc_now(),
        "dataset": str(data_dir),
        "output_dir": str(output_dir),
        "expected": {
            "samples": args.expected_samples,
            "runs": args.expected_runs,
            "samples_per_run": args.expected_samples_per_run,
            "nodes": args.expected_nodes,
            "window": args.expected_window,
            "features": args.expected_features,
            "normal_runs": args.expected_normal_runs,
            "attack_runs": args.expected_attack_runs,
            "train_runs": args.expected_train_runs,
            "val_runs": args.expected_val_runs,
            "test_runs": args.expected_test_runs,
            "attacker_count_run_distribution": {
                "1": args.expected_single_attacker_runs,
                "2": args.expected_two_attacker_runs,
                "3": args.expected_three_attacker_runs,
                "4": args.expected_four_attacker_runs,
            },
            "first_end_epoch": args.expected_first_end_epoch,
            "last_end_epoch": args.expected_last_end_epoch,
        },
        "hard_checks": hard_checks,
        "warnings": warnings,
        "hard_pass": hard_pass,
        "verdict": verdict,
        "model_inference_performed": False,
        "training_performed": False,
        "threshold_selection_performed": False,
        "validation_predictions_accessed": False,
        "test_predictions_accessed": False,
        "validation_samples_structurally_inspected": True,
        "test_samples_structurally_inspected": True,
        "dataset_tree_modified": False,
        "complete_x_loaded_into_ram": False,
        "next_authorized_stage": (
            "V4-A0.2 manifest-to-array alignment"
            if hard_pass
            else "Repair or explain V4-A0.1 hard failures"
        ),
        "training_authorized": False,
        "artifacts": [
            "v4_array_shape_dtype_audit.json",
            "v4_chunk_feature_stats.json",
            "v4_label_count_audit.json",
            "v4_split_count_audit.json",
            "v4_scenario_distribution_audit.json",
            "v4_run_boundary_audit.json",
            "v4_run_level_summary.csv",
            "v4_edge_index_audit.json",
            "v4_metadata_summary.json",
            "v4_memmap_integrity_report.md",
        ],
    }
    manifest_path = output_dir / "v4_a0_1_manifest.json"
    write_json(manifest_path, manifest)

    artifact_rows = []
    for artifact in sorted(output_dir.iterdir()):
        if artifact.is_file() and artifact.name != "v4_a0_1_artifact_hashes.csv":
            artifact_rows.append({
                "artifact": artifact.name,
                "size_bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            })

    artifact_hashes = output_dir / "v4_a0_1_artifact_hashes.csv"
    with artifact_hashes.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["artifact", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(artifact_rows)

    print("=" * 88)
    print(verdict)
    print(f"hard_checks_passed={sum(hard_checks.values())}/{len(hard_checks)}")
    print(f"x_samples_scanned={x_stats['total_samples_scanned']}")
    print(f"x_nonfinite_total={x_stats['nonfinite_total']}")
    print(f"x_below_zero_total={x_stats['below_zero_total']}")
    print(f"x_above_one_total={x_stats['above_one_total']}")
    print(f"run_segments={run_audit['segment_count']}")
    print(f"completed_runs={len(completed_runs)}")
    print(f"model_inference_performed=False")
    print(f"training_performed=False")
    print(f"dataset_tree_modified=False")
    print(f"training_authorized=False")
    print(f"manifest={manifest_path}")
    print(f"next_authorized_stage={manifest['next_authorized_stage']}")
    print("=" * 88)

    if args.strict and not hard_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
