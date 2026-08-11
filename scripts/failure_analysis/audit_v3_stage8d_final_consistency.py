#!/usr/bin/env python3
"""Final Stage 8D consistency audit for the V3 NoC temporal graph dataset.

The audit is read-only with respect to the dataset. The large x.npy tensor is
opened with NumPy memory mapping and processed in chunks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EXPECTED_FEATURES = [
    "ifd_in_norm",
    "ifd_out_norm",
    "input_flit_count_norm",
    "output_flit_count_norm",
    "in_count_norm_local",
    "in_count_norm_north",
    "in_count_norm_east",
    "in_count_norm_south",
    "in_count_norm_west",
    "out_count_norm_local",
    "out_count_norm_north",
    "out_count_norm_east",
    "out_count_norm_south",
    "out_count_norm_west",
    "ifd_in_norm_local",
    "ifd_in_norm_north",
    "ifd_in_norm_east",
    "ifd_in_norm_south",
    "ifd_in_norm_west",
    "ifd_out_norm_local",
    "ifd_out_norm_north",
    "ifd_out_norm_east",
    "ifd_out_norm_south",
    "ifd_out_norm_west",
]

REQUIRED_FILES = [
    "x.npy",
    "y_graph.npy",
    "y_node.npy",
    "edge_index.npy",
    "run_id.npy",
    "end_epoch.npy",
    "split.npy",
    "active_cores.npy",
    "profile.npy",
    "seed.npy",
    "strength.npy",
    "attackers.npy",
    "feature_cols.npy",
]

EXPECTED_OUTPUTS = [
    "STAGE8D_FINAL_CONSISTENCY_REPORT.md",
    "logs/46_stage8d_validation.txt",
    "tables/stage8d_run_structure.csv",
    "tables/stage8d_run_metadata_consistency.csv",
    "tables/stage8d_feature_range_summary.csv",
    "tables/stage8d_count_consistency.csv",
    "tables/stage8d_ifd_one_rate_by_feature.csv",
    "tables/stage8d_ifd_one_rate_by_router.csv",
    "tables/stage8d_ifd_one_rate_by_split_class.csv",
    "tables/stage8d_boundary_port_audit.csv",
    "tables/stage8d_hard_run_audit.csv",
    "tables/stage8d_summary.json",
]

HARD_RUN_EXPECTATIONS = {
    "N-3-7-8-12-Pmixed-R18-V3": {
        "split": "test",
        "y_graph": 0,
        "attackers": [],
    },
    "N-5-10-Pbursty-R51-A-12-S20-V3": {
        "split": "test",
        "y_graph": 1,
        "attackers": [12],
    },
}

PORTS = ["local", "north", "east", "south", "west"]


@dataclass
class Validation:
    name: str
    passed: bool
    severity: str
    details: str


@dataclass
class AuditState:
    validations: list[Validation] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def add_check(self, name: str, passed: bool, details: str, severity: str = "hard") -> None:
        self.validations.append(
            Validation(name=name, passed=bool(passed), severity=severity, details=details)
        )

    @property
    def hard_failures(self) -> list[Validation]:
        return [v for v in self.validations if v.severity == "hard" and not v.passed]

    @property
    def warnings(self) -> list[Validation]:
        return [v for v in self.validations if v.severity != "hard" and not v.passed]


@dataclass(frozen=True)
class AuditConfig:
    expected_samples: int
    expected_runs: int
    expected_samples_per_run: int
    expected_end_start: int
    expected_end_stop: int
    expected_nodes: int = 16
    expected_window_epochs: int = 8
    expected_features: int = 24
    expected_edges: int = 64
    tolerance: float = 1e-6


@dataclass
class RunSegment:
    run_id: str
    start: int
    stop: int

    @property
    def count(self) -> int:
        return self.stop - self.start


class AuditError(RuntimeError):
    """Raised for unrecoverable audit configuration or input errors."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_output_layout(output_root: Path, overwrite: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    (output_root / "tables").mkdir(parents=True, exist_ok=True)

    existing = [output_root / rel for rel in EXPECTED_OUTPUTS if (output_root / rel).exists()]
    if existing and not overwrite:
        joined = "\n".join(f"  {p}" for p in existing)
        raise AuditError(
            "Refusing to overwrite existing Stage 8D outputs. "
            "Use a new output directory or pass --overwrite:\n" + joined
        )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def load_arrays(dataset_root: Path) -> dict[str, np.ndarray]:
    missing = [name for name in REQUIRED_FILES if not (dataset_root / name).is_file()]
    if missing:
        raise AuditError("Missing required arrays: " + ", ".join(missing))

    arrays: dict[str, np.ndarray] = {}
    for name in REQUIRED_FILES:
        arrays[name[:-4]] = np.load(dataset_root / name, mmap_mode="r", allow_pickle=False)
    return arrays


def parse_attackers(value: Any) -> list[int]:
    text = str(value).strip()
    if text in {"", "none", "None", "idle", "-"}:
        return []
    text = text.replace(",", "-")
    values = [int(token) for token in text.split("-") if token != ""]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate attacker IDs in {value!r}")
    return sorted(values)


def derive_run_segments(run_ids: np.ndarray) -> list[RunSegment]:
    if run_ids.ndim != 1 or run_ids.size == 0:
        return []
    change_points = np.flatnonzero(run_ids[1:] != run_ids[:-1]) + 1
    starts = np.concatenate(([0], change_points))
    stops = np.concatenate((change_points, [run_ids.shape[0]]))
    return [
        RunSegment(run_id=str(run_ids[start]), start=int(start), stop=int(stop))
        for start, stop in zip(starts, stops, strict=True)
    ]


def constant_value(array: np.ndarray, start: int, stop: int) -> tuple[bool, Any]:
    first = array[start]
    return bool(np.all(array[start:stop] == first)), first


def bitmap_routers(bitmap: np.ndarray, tolerance: float) -> list[int]:
    return [int(i) for i in np.flatnonzero(np.asarray(bitmap) > 0.5)]


def graph_connected(edge_pairs: set[tuple[int, int]], num_nodes: int) -> bool:
    adjacency: dict[int, set[int]] = {i: set() for i in range(num_nodes)}
    for src, dst in edge_pairs:
        if src == dst:
            continue
        adjacency[src].add(dst)
        adjacency[dst].add(src)
    seen: set[int] = set()
    queue: deque[int] = deque([0])
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        queue.extend(adjacency[node] - seen)
    return len(seen) == num_nodes


def hash_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def expected_invalid_direction(router: int, direction: str) -> bool:
    row, col = divmod(router, 4)
    return {
        "local": False,
        "north": row == 0,
        "east": col == 3,
        "south": row == 3,
        "west": col == 0,
    }[direction]


def scan_features(
    x: np.ndarray,
    split: np.ndarray,
    y_graph: np.ndarray,
    feature_names: Sequence[str],
    chunk_size: int,
    tolerance: float,
) -> dict[str, Any]:
    num_samples, num_nodes, num_epochs, num_features = x.shape

    feature_min = np.full(num_features, np.inf, dtype=np.float64)
    feature_max = np.full(num_features, -np.inf, dtype=np.float64)
    finite_count = np.zeros(num_features, dtype=np.int64)
    nonfinite_count = np.zeros(num_features, dtype=np.int64)
    zero_count = np.zeros(num_features, dtype=np.int64)
    one_count = np.zeros(num_features, dtype=np.int64)

    count_stats = {
        "input": defaultdict(float),
        "output": defaultdict(float),
    }
    for side in count_stats:
        count_stats[side].update({
            "elements": 0,
            "unsaturated_elements": 0,
            "saturated_elements": 0,
            "unsaturated_violations": 0,
            "saturated_violations": 0,
            "max_unsaturated_abs_error": 0.0,
            "minimum_saturated_margin": float("inf"),
        })

    ifd_features = list(range(14, 24))
    ifd_feature_acc: dict[int, Counter[str]] = {idx: Counter() for idx in ifd_features}
    ifd_router_acc: dict[tuple[int, int], Counter[str]] = {
        (router, idx): Counter() for router in range(num_nodes) for idx in ifd_features
    }
    ifd_group_acc: dict[tuple[str, int, int], Counter[str]] = defaultdict(Counter)
    boundary_acc: dict[tuple[int, str, str], Counter[str]] = defaultdict(Counter)

    for start in range(0, num_samples, chunk_size):
        stop = min(start + chunk_size, num_samples)
        chunk = np.asarray(x[start:stop])
        finite = np.isfinite(chunk)

        nonfinite_count += np.sum(~finite, axis=(0, 1, 2), dtype=np.int64)
        finite_count += np.sum(finite, axis=(0, 1, 2), dtype=np.int64)
        zero_count += np.sum(chunk == 0.0, axis=(0, 1, 2), dtype=np.int64)
        one_count += np.sum(chunk == 1.0, axis=(0, 1, 2), dtype=np.int64)

        for feature_idx in range(num_features):
            values = chunk[..., feature_idx]
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                feature_min[feature_idx] = min(feature_min[feature_idx], float(finite_values.min()))
                feature_max[feature_idx] = max(feature_max[feature_idx], float(finite_values.max()))

        for side, aggregate_idx, directional_slice in [
            ("input", 2, slice(4, 9)),
            ("output", 3, slice(9, 14)),
        ]:
            aggregate = chunk[..., aggregate_idx].astype(np.float64, copy=False)
            directions = chunk[..., directional_slice].astype(np.float64, copy=False)
            directional_mean = directions.mean(axis=-1)
            saturated = np.any(directions >= 1.0 - tolerance, axis=-1)
            unsaturated = ~saturated
            difference = aggregate - directional_mean
            stats = count_stats[side]
            stats["elements"] += int(difference.size)
            stats["unsaturated_elements"] += int(unsaturated.sum())
            stats["saturated_elements"] += int(saturated.sum())
            if np.any(unsaturated):
                unsat_error = np.abs(difference[unsaturated])
                stats["unsaturated_violations"] += int(np.sum(unsat_error > tolerance))
                stats["max_unsaturated_abs_error"] = max(
                    float(stats["max_unsaturated_abs_error"]), float(unsat_error.max())
                )
            if np.any(saturated):
                sat_margin = difference[saturated]
                stats["saturated_violations"] += int(np.sum(sat_margin < -tolerance))
                stats["minimum_saturated_margin"] = min(
                    float(stats["minimum_saturated_margin"]), float(sat_margin.min())
                )

        split_chunk = np.asarray(split[start:stop]).astype(str)
        label_chunk = np.asarray(y_graph[start:stop]).astype(np.int64)

        for ifd_idx in ifd_features:
            count_idx = ifd_idx - 10
            values = chunk[..., ifd_idx]
            counts = chunk[..., count_idx]
            is_one = values == 1.0
            paired_zero = is_one & (counts == 0.0)
            paired_positive = is_one & (counts > 0.0)

            acc = ifd_feature_acc[ifd_idx]
            acc["total"] += int(values.size)
            acc["one"] += int(is_one.sum())
            acc["one_count_zero"] += int(paired_zero.sum())
            acc["one_count_positive"] += int(paired_positive.sum())

            for router in range(num_nodes):
                router_values = values[:, router, :]
                router_counts = counts[:, router, :]
                router_one = router_values == 1.0
                router_acc = ifd_router_acc[(router, ifd_idx)]
                router_acc["total"] += int(router_values.size)
                router_acc["one"] += int(router_one.sum())
                router_acc["one_count_zero"] += int(
                    np.sum(router_one & (router_counts == 0.0))
                )
                router_acc["one_count_positive"] += int(
                    np.sum(router_one & (router_counts > 0.0))
                )

            for split_name in np.unique(split_chunk):
                for label in (0, 1):
                    sample_mask = (split_chunk == split_name) & (label_chunk == label)
                    if not np.any(sample_mask):
                        continue
                    group_values = values[sample_mask]
                    group_counts = counts[sample_mask]
                    group_one = group_values == 1.0
                    group_acc = ifd_group_acc[(str(split_name), int(label), ifd_idx)]
                    group_acc["total"] += int(group_values.size)
                    group_acc["one"] += int(group_one.sum())
                    group_acc["one_count_zero"] += int(
                        np.sum(group_one & (group_counts == 0.0))
                    )
                    group_acc["one_count_positive"] += int(
                        np.sum(group_one & (group_counts > 0.0))
                    )

        for side, count_base, ifd_base in [("input", 4, 14), ("output", 9, 19)]:
            for port_idx, direction in enumerate(PORTS):
                count_idx = count_base + port_idx
                ifd_idx = ifd_base + port_idx
                count_values = chunk[..., count_idx]
                ifd_values = chunk[..., ifd_idx]
                for router in range(num_nodes):
                    counts_r = count_values[:, router, :]
                    ifd_r = ifd_values[:, router, :]
                    acc = boundary_acc[(router, direction, side)]
                    acc["total"] += int(counts_r.size)
                    acc["count_zero"] += int(np.sum(counts_r == 0.0))
                    acc["count_positive"] += int(np.sum(counts_r > 0.0))
                    acc["ifd_one"] += int(np.sum(ifd_r == 1.0))
                    acc["zero_and_one"] += int(np.sum((counts_r == 0.0) & (ifd_r == 1.0)))
                    acc["positive_and_one"] += int(np.sum((counts_r > 0.0) & (ifd_r == 1.0)))

    total_values_per_feature = num_samples * num_nodes * num_epochs
    feature_rows: list[dict[str, Any]] = []
    for idx, name in enumerate(feature_names):
        feature_rows.append({
            "feature_index": idx,
            "feature_name": name,
            "minimum": None if np.isinf(feature_min[idx]) else float(feature_min[idx]),
            "maximum": None if np.isinf(feature_max[idx]) else float(feature_max[idx]),
            "finite_count": int(finite_count[idx]),
            "nonfinite_count": int(nonfinite_count[idx]),
            "zero_count": int(zero_count[idx]),
            "zero_rate": float(zero_count[idx] / total_values_per_feature),
            "one_count": int(one_count[idx]),
            "one_rate": float(one_count[idx] / total_values_per_feature),
        })

    count_rows: list[dict[str, Any]] = []
    for side, stats in count_stats.items():
        minimum_margin = stats["minimum_saturated_margin"]
        if minimum_margin == float("inf"):
            minimum_margin = None
        count_rows.append({
            "side": side,
            "elements": int(stats["elements"]),
            "unsaturated_elements": int(stats["unsaturated_elements"]),
            "saturated_elements": int(stats["saturated_elements"]),
            "unsaturated_violations": int(stats["unsaturated_violations"]),
            "saturated_violations": int(stats["saturated_violations"]),
            "max_unsaturated_abs_error": float(stats["max_unsaturated_abs_error"]),
            "minimum_saturated_margin": minimum_margin,
        })

    ifd_feature_rows: list[dict[str, Any]] = []
    for idx in ifd_features:
        acc = ifd_feature_acc[idx]
        offset = idx - 14 if idx < 19 else idx - 19
        side = "input" if idx < 19 else "output"
        ifd_feature_rows.append({
            "feature_index": idx,
            "feature_name": feature_names[idx],
            "side": side,
            "direction": PORTS[offset],
            "total_values": int(acc["total"]),
            "one_count": int(acc["one"]),
            "one_rate": float(acc["one"] / acc["total"]) if acc["total"] else 0.0,
            "one_with_zero_count": int(acc["one_count_zero"]),
            "one_with_positive_count": int(acc["one_count_positive"]),
        })

    ifd_router_rows: list[dict[str, Any]] = []
    for (router, idx), acc in sorted(ifd_router_acc.items()):
        offset = idx - 14 if idx < 19 else idx - 19
        side = "input" if idx < 19 else "output"
        ifd_router_rows.append({
            "router": router,
            "feature_index": idx,
            "feature_name": feature_names[idx],
            "side": side,
            "direction": PORTS[offset],
            "total_values": int(acc["total"]),
            "one_count": int(acc["one"]),
            "one_rate": float(acc["one"] / acc["total"]) if acc["total"] else 0.0,
            "one_with_zero_count": int(acc["one_count_zero"]),
            "one_with_positive_count": int(acc["one_count_positive"]),
        })

    ifd_group_rows: list[dict[str, Any]] = []
    for (split_name, label, idx), acc in sorted(ifd_group_acc.items()):
        ifd_group_rows.append({
            "split": split_name,
            "graph_label": label,
            "feature_index": idx,
            "feature_name": feature_names[idx],
            "total_values": int(acc["total"]),
            "one_count": int(acc["one"]),
            "one_rate": float(acc["one"] / acc["total"]) if acc["total"] else 0.0,
            "one_with_zero_count": int(acc["one_count_zero"]),
            "one_with_positive_count": int(acc["one_count_positive"]),
        })

    boundary_rows: list[dict[str, Any]] = []
    for (router, direction, side), acc in sorted(boundary_acc.items()):
        total = int(acc["total"])
        boundary_rows.append({
            "router": router,
            "row_major_row": router // 4,
            "row_major_col": router % 4,
            "side": side,
            "direction": direction,
            "expected_invalid_under_row_major_assumption": expected_invalid_direction(router, direction),
            "total_values": total,
            "count_zero_rate": float(acc["count_zero"] / total) if total else 0.0,
            "count_positive_rate": float(acc["count_positive"] / total) if total else 0.0,
            "ifd_one_rate": float(acc["ifd_one"] / total) if total else 0.0,
            "zero_count_and_ifd_one_rate": float(acc["zero_and_one"] / total) if total else 0.0,
            "positive_count_and_ifd_one_rate": float(acc["positive_and_one"] / total) if total else 0.0,
        })

    return {
        "feature_rows": feature_rows,
        "count_rows": count_rows,
        "ifd_feature_rows": ifd_feature_rows,
        "ifd_router_rows": ifd_router_rows,
        "ifd_group_rows": ifd_group_rows,
        "boundary_rows": boundary_rows,
    }


def audit_runs(
    arrays: Mapping[str, np.ndarray],
    config: AuditConfig,
    state: AuditState,
) -> tuple[list[RunSegment], list[dict[str, Any]], list[dict[str, Any]]]:
    run_ids = arrays["run_id"]
    segments = derive_run_segments(run_ids)
    unique_run_ids = np.unique(run_ids)

    state.add_check(
        "unique_run_count",
        len(unique_run_ids) == config.expected_runs,
        f"observed={len(unique_run_ids)} expected={config.expected_runs}",
    )
    state.add_check(
        "run_contiguity",
        len(segments) == len(unique_run_ids),
        f"segments={len(segments)} unique_runs={len(unique_run_ids)}",
    )

    duplicate_segments = [run for run, count in Counter(s.run_id for s in segments).items() if count > 1]
    state.add_check(
        "no_repeated_run_segments",
        not duplicate_segments,
        "repeated=" + repr(duplicate_segments),
    )

    expected_end_epochs = np.arange(
        config.expected_end_start,
        config.expected_end_stop + 1,
        dtype=np.int64,
    )

    run_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    all_sample_counts_ok = True
    all_epoch_sequences_ok = True
    all_split_values_ok = True
    all_graph_labels_ok = True
    all_node_labels_ok = True
    all_metadata_ok = True
    all_attacker_labels_ok = True

    split_run_counts: Counter[str] = Counter()

    for segment in segments:
        start, stop = segment.start, segment.stop
        run_id = segment.run_id
        end_values = np.asarray(arrays["end_epoch"][start:stop], dtype=np.int64)
        end_epoch_ok = bool(np.array_equal(end_values, expected_end_epochs))
        sample_count_ok = segment.count == config.expected_samples_per_run
        all_sample_counts_ok &= sample_count_ok
        all_epoch_sequences_ok &= end_epoch_ok

        split_ok, split_value = constant_value(arrays["split"], start, stop)
        split_value = str(split_value)
        split_domain_ok = split_value in {"train", "val", "test"}
        all_split_values_ok &= split_ok and split_domain_ok
        split_run_counts[split_value] += 1

        graph_ok, graph_value = constant_value(arrays["y_graph"], start, stop)
        graph_value_float = float(graph_value)
        graph_binary = graph_value_float in {0.0, 1.0}
        all_graph_labels_ok &= graph_ok and graph_binary

        node_first = np.asarray(arrays["y_node"][start])
        node_constant = bool(np.all(arrays["y_node"][start:stop] == node_first))
        node_binary = bool(np.all((node_first == 0.0) | (node_first == 1.0)))
        all_node_labels_ok &= node_constant and node_binary

        metadata_values: dict[str, Any] = {}
        metadata_flags: dict[str, bool] = {}
        for key in ["active_cores", "profile", "seed", "strength", "attackers"]:
            is_constant, value = constant_value(arrays[key], start, stop)
            metadata_values[key] = str(value)
            metadata_flags[key] = is_constant
        metadata_constant = all(metadata_flags.values())
        all_metadata_ok &= metadata_constant

        try:
            attackers = parse_attackers(metadata_values["attackers"])
            parse_ok = True
        except Exception:
            attackers = []
            parse_ok = False

        expected_bitmap = np.zeros(config.expected_nodes, dtype=np.float32)
        attackers_in_range = all(0 <= attacker < config.expected_nodes for attacker in attackers)
        if attackers_in_range:
            for attacker in attackers:
                expected_bitmap[attacker] = 1.0

        node_matches_attackers = bool(
            parse_ok
            and attackers_in_range
            and np.array_equal(node_first.astype(np.float32), expected_bitmap)
        )
        normal_attack_consistent = (
            (graph_value_float == 0.0 and not attackers and np.all(node_first == 0.0))
            or (graph_value_float == 1.0 and bool(attackers) and node_matches_attackers)
        )
        all_attacker_labels_ok &= node_matches_attackers and normal_attack_consistent

        run_rows.append({
            "run_id": run_id,
            "first_sample_index": start,
            "last_sample_index": stop - 1,
            "sample_count": segment.count,
            "sample_count_ok": sample_count_ok,
            "first_end_epoch": int(end_values[0]) if end_values.size else None,
            "last_end_epoch": int(end_values[-1]) if end_values.size else None,
            "end_epoch_sequence_ok": end_epoch_ok,
            "split": split_value,
            "graph_label": int(graph_value_float) if graph_binary else graph_value_float,
            "node_positive_routers": "-".join(map(str, bitmap_routers(node_first, config.tolerance))),
        })

        metadata_rows.append({
            "run_id": run_id,
            "split": split_value,
            "split_constant": split_ok,
            "split_domain_ok": split_domain_ok,
            "graph_label": graph_value_float,
            "graph_label_constant": graph_ok,
            "graph_label_binary": graph_binary,
            "node_label_constant": node_constant,
            "node_label_binary": node_binary,
            "active_cores": metadata_values["active_cores"],
            "active_cores_constant": metadata_flags["active_cores"],
            "profile": metadata_values["profile"],
            "profile_constant": metadata_flags["profile"],
            "seed": metadata_values["seed"],
            "seed_constant": metadata_flags["seed"],
            "strength": metadata_values["strength"],
            "strength_constant": metadata_flags["strength"],
            "attackers": metadata_values["attackers"],
            "attackers_constant": metadata_flags["attackers"],
            "attackers_parse_ok": parse_ok,
            "attackers_in_range": attackers_in_range,
            "node_matches_attackers": node_matches_attackers,
            "normal_attack_consistent": normal_attack_consistent,
            "all_metadata_constant": metadata_constant,
        })

    state.add_check(
        "samples_per_run",
        all_sample_counts_ok,
        f"expected={config.expected_samples_per_run}; failures="
        f"{sum(not bool(row['sample_count_ok']) for row in run_rows)}",
    )
    state.add_check(
        "end_epoch_sequence",
        all_epoch_sequences_ok,
        f"expected={config.expected_end_start}..{config.expected_end_stop}; failures="
        f"{sum(not bool(row['end_epoch_sequence_ok']) for row in run_rows)}",
    )
    state.add_check(
        "run_level_split_isolation",
        all_split_values_ok,
        f"run_counts_by_split={dict(split_run_counts)}",
    )
    state.add_check(
        "graph_labels_constant_and_binary",
        all_graph_labels_ok,
        "Every run must have one graph label in {0,1}.",
    )
    state.add_check(
        "node_labels_constant_and_binary",
        all_node_labels_ok,
        "Every run must have one binary 16-router bitmap.",
    )
    state.add_check(
        "scenario_metadata_constant",
        all_metadata_ok,
        "split/active_cores/profile/seed/strength/attackers must be constant within runs.",
    )
    state.add_check(
        "attacker_metadata_matches_node_labels",
        all_attacker_labels_ok,
        "Normal runs must have zero bitmap; attacks must match attackers metadata.",
    )

    if config.expected_runs == 71:
        expected_split_counts = {"train": 45, "val": 13, "test": 13}
        state.add_check(
            "expected_run_counts_by_split",
            dict(split_run_counts) == expected_split_counts,
            f"observed={dict(split_run_counts)} expected={expected_split_counts}",
        )

    return segments, run_rows, metadata_rows


def audit_graph(edge_index: np.ndarray, config: AuditConfig, state: AuditState) -> dict[str, Any]:
    shape_ok = edge_index.shape == (2, config.expected_edges)
    integer_dtype = np.issubdtype(edge_index.dtype, np.integer)
    values = np.asarray(edge_index, dtype=np.int64)
    ids_in_range = bool(values.size and values.min() >= 0 and values.max() < config.expected_nodes)
    pairs = [tuple(map(int, values[:, idx])) for idx in range(values.shape[1])] if values.ndim == 2 and values.shape[0] == 2 else []
    unique_pairs = set(pairs)
    no_duplicates = len(unique_pairs) == len(pairs)
    self_loops = sum(src == dst for src, dst in pairs)
    nonself = len(pairs) - self_loops
    all_nodes = sorted(set(values.ravel().tolist())) if values.size else []
    connected = graph_connected(unique_pairs, config.expected_nodes) if ids_in_range else False
    reciprocal_nonself = sum((dst, src) in unique_pairs for src, dst in unique_pairs if src != dst)

    state.add_check("edge_index_shape", shape_ok, f"observed={edge_index.shape} expected={(2, config.expected_edges)}")
    state.add_check("edge_index_integer_dtype", integer_dtype, f"dtype={edge_index.dtype}")
    state.add_check("edge_index_router_ids", ids_in_range, f"min={values.min() if values.size else None} max={values.max() if values.size else None}")
    state.add_check("edge_index_no_duplicates", no_duplicates, f"edges={len(pairs)} unique={len(unique_pairs)}")
    state.add_check("edge_index_self_loops", self_loops == config.expected_nodes, f"observed={self_loops} expected={config.expected_nodes}")
    expected_nonself = config.expected_edges - config.expected_nodes
    state.add_check("edge_index_nonself_edges", nonself == expected_nonself, f"observed={nonself} expected={expected_nonself}")
    state.add_check("edge_index_all_nodes_present", all_nodes == list(range(config.expected_nodes)), f"nodes={all_nodes}")
    state.add_check("edge_index_connected", connected, "Connectivity checked after ignoring edge direction.")
    state.add_check(
        "edge_index_reciprocal_nonself",
        reciprocal_nonself == nonself,
        f"reciprocal_nonself={reciprocal_nonself} nonself={nonself}",
        severity="warning",
    )

    return {
        "shape": list(edge_index.shape),
        "dtype": str(edge_index.dtype),
        "minimum_router_id": int(values.min()) if values.size else None,
        "maximum_router_id": int(values.max()) if values.size else None,
        "edge_count": len(pairs),
        "unique_edge_count": len(unique_pairs),
        "self_loops": self_loops,
        "nonself_edges": nonself,
        "connected": connected,
        "all_nodes": all_nodes,
        "reciprocal_nonself_edges": reciprocal_nonself,
    }


def audit_hard_runs(
    arrays: Mapping[str, np.ndarray],
    segments: Sequence[RunSegment],
    x: np.ndarray,
    chunk_size: int,
    tolerance: float,
    state: AuditState,
) -> list[dict[str, Any]]:
    segment_by_id = {segment.run_id: segment for segment in segments}
    rows: list[dict[str, Any]] = []
    all_ok = True

    for run_id, expected in HARD_RUN_EXPECTATIONS.items():
        segment = segment_by_id.get(run_id)
        if segment is None:
            rows.append({
                "run_id": run_id,
                "present": False,
                "expected_split": expected["split"],
                "expected_graph_label": expected["y_graph"],
                "expected_attackers": "-".join(map(str, expected["attackers"])),
            })
            all_ok = False
            continue

        start, stop = segment.start, segment.stop
        split_value = str(arrays["split"][start])
        graph_value = int(float(arrays["y_graph"][start]))
        attackers = parse_attackers(arrays["attackers"][start])
        node_routers = bitmap_routers(np.asarray(arrays["y_node"][start]), tolerance)
        end_values = np.asarray(arrays["end_epoch"][start:stop])

        minimum = float("inf")
        maximum = float("-inf")
        ifd_one = 0
        ifd_total = 0
        count_zero = 0
        count_total = 0
        for chunk_start in range(start, stop, chunk_size):
            chunk_stop = min(chunk_start + chunk_size, stop)
            chunk = np.asarray(x[chunk_start:chunk_stop])
            minimum = min(minimum, float(np.nanmin(chunk)))
            maximum = max(maximum, float(np.nanmax(chunk)))
            ifd_values = chunk[..., 14:24]
            count_values = chunk[..., 4:14]
            ifd_one += int(np.sum(ifd_values == 1.0))
            ifd_total += int(ifd_values.size)
            count_zero += int(np.sum(count_values == 0.0))
            count_total += int(count_values.size)

        row_ok = (
            split_value == expected["split"]
            and graph_value == expected["y_graph"]
            and attackers == expected["attackers"]
            and node_routers == expected["attackers"]
        )
        all_ok &= row_ok
        rows.append({
            "run_id": run_id,
            "present": True,
            "sample_count": segment.count,
            "split": split_value,
            "expected_split": expected["split"],
            "graph_label": graph_value,
            "expected_graph_label": expected["y_graph"],
            "attackers_metadata": "-".join(map(str, attackers)),
            "node_positive_routers": "-".join(map(str, node_routers)),
            "expected_attackers": "-".join(map(str, expected["attackers"])),
            "end_epoch_first": int(end_values[0]),
            "end_epoch_last": int(end_values[-1]),
            "feature_minimum": minimum,
            "feature_maximum": maximum,
            "directional_ifd_one_rate": ifd_one / ifd_total if ifd_total else 0.0,
            "directional_count_zero_rate": count_zero / count_total if count_total else 0.0,
            "metadata_and_labels_ok": row_ok,
        })

    state.add_check(
        "hard_runs_present_and_correct",
        all_ok,
        "Both dominant hard runs must exist with expected split and labels.",
    )
    return rows


def choose_verdict(state: AuditState) -> str:
    if state.hard_failures:
        return "FAIL — REPROCESS REQUIRED"
    if state.limitations or state.warnings:
        return "PASS WITH DOCUMENTED LIMITATIONS"
    return "PASS"


def validation_text(state: AuditState, verdict: str) -> str:
    lines = [f"verdict: {verdict}", ""]
    for validation in state.validations:
        status = "PASS" if validation.passed else "FAIL"
        lines.append(f"[{status}] [{validation.severity}] {validation.name}: {validation.details}")
    if state.limitations:
        lines.append("")
        lines.append("DOCUMENTED LIMITATIONS")
        for limitation in state.limitations:
            lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def markdown_report(
    dataset_root: Path,
    config: AuditConfig,
    state: AuditState,
    verdict: str,
    summary: Mapping[str, Any],
) -> str:
    passed = sum(v.passed for v in state.validations)
    failed = len(state.validations) - passed
    hard_failures = len(state.hard_failures)
    warning_failures = len(state.warnings)

    lines = [
        "# Stage 8D Final V3 Consistency Report",
        "",
        f"- Generated: `{utc_now_iso()}`",
        f"- Dataset: `{dataset_root}`",
        f"- Verdict: **{verdict}**",
        f"- Checks passed: **{passed}/{len(state.validations)}**",
        f"- Hard failures: **{hard_failures}**",
        f"- Warning-level findings: **{warning_failures}**",
        "",
        "## Scope",
        "",
        "This audit checks whether the unpacked V3 arrays obey the recovered builder logic. "
        "The main tensor was opened read-only with NumPy memory mapping and scanned in chunks.",
        "",
        "## Expected structure",
        "",
        f"- Samples: `{config.expected_samples}`",
        f"- Runs: `{config.expected_runs}`",
        f"- Samples per run: `{config.expected_samples_per_run}`",
        f"- End epochs: `{config.expected_end_start}..{config.expected_end_stop}`",
        f"- Tensor nodes/time/features: `{config.expected_nodes} × {config.expected_window_epochs} × {config.expected_features}`",
        "",
        "## Validation results",
        "",
        "| Status | Severity | Check | Details |",
        "|---|---|---|---|",
    ]
    for validation in state.validations:
        status = "PASS" if validation.passed else "FAIL"
        details = validation.details.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {status} | {validation.severity} | `{validation.name}` | {details} |")

    lines.extend(["", "## Documented limitations", ""])
    if state.limitations:
        lines.extend(f"- {item}" for item in state.limitations)
    else:
        lines.append("- None recorded by this audit.")

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    if verdict == "PASS":
        lines.append("The final arrays match the recovered builder rules without detected structural inconsistencies.")
    elif verdict == "PASS WITH DOCUMENTED LIMITATIONS":
        lines.append(
            "The final arrays match the recovered builder rules. The remaining findings are representation, "
            "coverage, or provenance limitations rather than evidence of array corruption."
        )
    else:
        lines.append(
            "One or more hard consistency checks failed. Model retraining should pause until the dataset is "
            "reprocessed or the provenance mismatch is resolved."
        )

    lines.extend([
        "",
        "## Output tables",
        "",
        "- `tables/stage8d_run_structure.csv`",
        "- `tables/stage8d_run_metadata_consistency.csv`",
        "- `tables/stage8d_feature_range_summary.csv`",
        "- `tables/stage8d_count_consistency.csv`",
        "- `tables/stage8d_ifd_one_rate_by_feature.csv`",
        "- `tables/stage8d_ifd_one_rate_by_router.csv`",
        "- `tables/stage8d_ifd_one_rate_by_split_class.csv`",
        "- `tables/stage8d_boundary_port_audit.csv`",
        "- `tables/stage8d_hard_run_audit.csv`",
        "- `tables/stage8d_summary.json`",
        "",
        "## What this audit cannot prove",
        "",
        "- The physical meaning of headerless raw-trace columns without the gem5 instrumentation source.",
        "- Exact attack-worker startup timing without `traffic_worker_v2`.",
        "- Whether each IFD value of 1.0 means invalid, idle, no-gap, or clipped-long-gap using only the final tensor.",
        "- Physical compass-direction mapping without the graph and trace mapping sources.",
        "",
        "## Summary snapshot",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
    ])
    return "\n".join(lines)


def run_audit(
    dataset_root: Path,
    output_root: Path,
    config: AuditConfig,
    chunk_size: int,
    overwrite: bool,
) -> str:
    ensure_output_layout(output_root, overwrite)
    arrays = load_arrays(dataset_root)
    state = AuditState()

    x = arrays["x"]
    expected_shape = (
        config.expected_samples,
        config.expected_nodes,
        config.expected_window_epochs,
        config.expected_features,
    )
    state.add_check("x_shape", x.shape == expected_shape, f"observed={x.shape} expected={expected_shape}")
    state.add_check("x_dtype", x.dtype == np.float32, f"observed={x.dtype} expected=float32", severity="warning")
    state.add_check("x_memory_mapped", isinstance(x, np.memmap), f"type={type(x).__name__}")

    sample_arrays = [
        "y_graph", "y_node", "run_id", "end_epoch", "split", "active_cores",
        "profile", "seed", "strength", "attackers",
    ]
    sample_alignment_ok = all(arrays[name].shape[0] == config.expected_samples for name in sample_arrays)
    state.add_check(
        "sample_array_alignment",
        sample_alignment_ok,
        "; ".join(f"{name}={arrays[name].shape}" for name in sample_arrays),
    )
    state.add_check(
        "y_node_shape",
        arrays["y_node"].shape == (config.expected_samples, config.expected_nodes),
        f"observed={arrays['y_node'].shape}",
    )

    feature_names = [str(value) for value in np.asarray(arrays["feature_cols"]).tolist()]
    state.add_check(
        "feature_columns",
        feature_names == EXPECTED_FEATURES,
        f"observed_count={len(feature_names)} expected_count={len(EXPECTED_FEATURES)}",
    )

    segments, run_rows, metadata_rows = audit_runs(arrays, config, state)
    graph_summary = audit_graph(arrays["edge_index"], config, state)

    feature_scan = scan_features(
        x=x,
        split=arrays["split"],
        y_graph=arrays["y_graph"],
        feature_names=feature_names,
        chunk_size=chunk_size,
        tolerance=config.tolerance,
    )

    nonfinite_total = sum(int(row["nonfinite_count"]) for row in feature_scan["feature_rows"])
    out_of_range = [
        row for row in feature_scan["feature_rows"]
        if row["minimum"] is None or row["maximum"] is None
        or float(row["minimum"]) < -config.tolerance
        or float(row["maximum"]) > 1.0 + config.tolerance
    ]
    state.add_check("feature_values_finite", nonfinite_total == 0, f"nonfinite_total={nonfinite_total}")
    state.add_check("feature_values_in_range", not out_of_range, f"out_of_range_features={[row['feature_name'] for row in out_of_range]}")

    count_violations = sum(
        int(row["unsaturated_violations"]) + int(row["saturated_violations"])
        for row in feature_scan["count_rows"]
    )
    state.add_check(
        "aggregate_directional_count_consistency",
        count_violations == 0,
        f"total_violations={count_violations}; tolerance={config.tolerance}",
    )

    if config.expected_runs == 71 and config.expected_samples == 233803:
        hard_rows = audit_hard_runs(
            arrays=arrays,
            segments=segments,
            x=x,
            chunk_size=chunk_size,
            tolerance=config.tolerance,
            state=state,
        )
    else:
        hard_rows = []
        state.add_check(
            "hard_runs_not_applicable_to_smoke_test",
            True,
            "Production hard-run checks are skipped for synthetic smoke-test dimensions.",
            severity="warning",
        )

    # Representation and provenance limitations are expected, not hard failures.
    total_ifd_values = sum(int(row["total_values"]) for row in feature_scan["ifd_feature_rows"])
    total_ifd_ones = sum(int(row["one_count"]) for row in feature_scan["ifd_feature_rows"])
    if total_ifd_ones:
        state.limitations.append(
            f"Directional IFD contains exact 1.0 values: {total_ifd_ones}/{total_ifd_values} "
            f"({total_ifd_ones / total_ifd_values:.6%}). Their physical causes are overloaded."
        )
    state.limitations.extend([
        "Boundary-port diagnostics use a provisional row-major 4x4 compass mapping and are not treated as hard failures.",
        "Final arrays alone cannot distinguish invalid, idle, no-gap, and clipped-long-gap IFD states.",
        "The dataset has many overlapping windows but only 71 independent simulation runs.",
        "The current V3 test split has been used for diagnosis and is no longer an untouched final publication holdout.",
        "Archive-derived 1980 modification times cannot establish the original generation date; hashes should be used for identity.",
    ])

    verdict = choose_verdict(state)

    tables_root = output_root / "tables"
    logs_root = output_root / "logs"
    atomic_write_csv(tables_root / "stage8d_run_structure.csv", run_rows, list(run_rows[0].keys()))
    atomic_write_csv(
        tables_root / "stage8d_run_metadata_consistency.csv",
        metadata_rows,
        list(metadata_rows[0].keys()),
    )
    atomic_write_csv(
        tables_root / "stage8d_feature_range_summary.csv",
        feature_scan["feature_rows"],
        list(feature_scan["feature_rows"][0].keys()),
    )
    atomic_write_csv(
        tables_root / "stage8d_count_consistency.csv",
        feature_scan["count_rows"],
        list(feature_scan["count_rows"][0].keys()),
    )
    atomic_write_csv(
        tables_root / "stage8d_ifd_one_rate_by_feature.csv",
        feature_scan["ifd_feature_rows"],
        list(feature_scan["ifd_feature_rows"][0].keys()),
    )
    atomic_write_csv(
        tables_root / "stage8d_ifd_one_rate_by_router.csv",
        feature_scan["ifd_router_rows"],
        list(feature_scan["ifd_router_rows"][0].keys()),
    )
    atomic_write_csv(
        tables_root / "stage8d_ifd_one_rate_by_split_class.csv",
        feature_scan["ifd_group_rows"],
        list(feature_scan["ifd_group_rows"][0].keys()),
    )
    atomic_write_csv(
        tables_root / "stage8d_boundary_port_audit.csv",
        feature_scan["boundary_rows"],
        list(feature_scan["boundary_rows"][0].keys()),
    )
    hard_run_fields = sorted({key for row in hard_rows for key in row}) or [
        "run_id", "present", "note"
    ]
    if not hard_rows:
        hard_rows = [{
            "run_id": "",
            "present": "",
            "note": "Production hard-run checks are not applicable to this synthetic smoke test.",
        }]
    atomic_write_csv(
        tables_root / "stage8d_hard_run_audit.csv",
        hard_rows,
        hard_run_fields,
    )

    summary = {
        "generated_at": utc_now_iso(),
        "verdict": verdict,
        "dataset_root": str(dataset_root.resolve()),
        "output_root": str(output_root.resolve()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "chunk_size": chunk_size,
        "x_shape": list(x.shape),
        "x_dtype": str(x.dtype),
        "x_memory_mapped": isinstance(x, np.memmap),
        "unique_runs": len(np.unique(arrays["run_id"])),
        "run_segments": len(segments),
        "graph": graph_summary,
        "checks_total": len(state.validations),
        "checks_passed": sum(v.passed for v in state.validations),
        "hard_failures": [v.name for v in state.hard_failures],
        "warning_findings": [v.name for v in state.warnings],
        "limitations": state.limitations,
        "input_files": {
            name: {
                "path": str((dataset_root / f"{name}.npy").resolve()),
                "size_bytes": (dataset_root / f"{name}.npy").stat().st_size,
                "shape": list(arrays[name].shape),
                "dtype": str(arrays[name].dtype),
            }
            for name in arrays
        },
    }

    atomic_write_json(tables_root / "stage8d_summary.json", summary)
    atomic_write_text(logs_root / "46_stage8d_validation.txt", validation_text(state, verdict))
    atomic_write_text(
        output_root / "STAGE8D_FINAL_CONSISTENCY_REPORT.md",
        markdown_report(dataset_root, config, state, verdict, summary),
    )

    print(f"Stage 8D verdict: {verdict}")
    print(f"Checks passed: {summary['checks_passed']}/{summary['checks_total']}")
    if summary["hard_failures"]:
        print("Hard failures:")
        for name in summary["hard_failures"]:
            print(f"  - {name}")
    print(f"Report: {output_root / 'STAGE8D_FINAL_CONSISTENCY_REPORT.md'}")
    print(f"Validation: {logs_root / '46_stage8d_validation.txt'}")
    print(f"Summary: {tables_root / 'stage8d_summary.json'}")
    return verdict


def mesh4x4_edge_index() -> np.ndarray:
    edges: list[tuple[int, int]] = []
    for router in range(16):
        edges.append((router, router))
    for row in range(4):
        for col in range(4):
            src = row * 4 + col
            if col + 1 < 4:
                dst = row * 4 + col + 1
                edges.extend([(src, dst), (dst, src)])
            if row + 1 < 4:
                dst = (row + 1) * 4 + col
                edges.extend([(src, dst), (dst, src)])
    return np.array(edges, dtype=np.int64).T


def create_synthetic_dataset(root: Path) -> AuditConfig:
    root.mkdir(parents=True, exist_ok=True)
    runs = [
        ("N-idle-Pidle-R0-V3", "train", 0, "idle", "idle", "0", "", ""),
        ("N-0-Pstream-R1-A-12-S20-V3", "val", 1, "0", "stream", "1", "20", "12"),
        ("N-5-Pbursty-R2-A-1-7-S63-V3", "test", 1, "5", "bursty", "2", "63", "1-7"),
    ]
    samples_per_run = 5
    window_epochs = 3
    total_samples = len(runs) * samples_per_run
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 0.8, size=(total_samples, 16, window_epochs, 24)).astype(np.float32)

    # Make aggregate counts exactly equal to directional means.
    x[..., 2] = x[..., 4:9].mean(axis=-1)
    x[..., 3] = x[..., 9:14].mean(axis=-1)
    # Add expected sentinel-like values without breaking counts.
    x[:, 0, :, 15] = 1.0
    x[:, 0, :, 5] = 0.0
    x[..., 2] = x[..., 4:9].mean(axis=-1)
    x[..., 3] = x[..., 9:14].mean(axis=-1)

    y_graph: list[float] = []
    y_node: list[np.ndarray] = []
    run_id: list[str] = []
    end_epoch: list[int] = []
    split: list[str] = []
    active_cores: list[str] = []
    profile: list[str] = []
    seed: list[str] = []
    strength: list[str] = []
    attackers: list[str] = []

    for run, split_name, label, active, prof, seed_value, strength_value, attacker_text in runs:
        bitmap = np.zeros(16, dtype=np.float32)
        for attacker in parse_attackers(attacker_text):
            bitmap[attacker] = 1.0
        for end in range(2, 7):
            y_graph.append(float(label))
            y_node.append(bitmap.copy())
            run_id.append(run)
            end_epoch.append(end)
            split.append(split_name)
            active_cores.append(active)
            profile.append(prof)
            seed.append(seed_value)
            strength.append(strength_value)
            attackers.append(attacker_text)

    arrays = {
        "x": x,
        "y_graph": np.array(y_graph, dtype=np.float32),
        "y_node": np.stack(y_node).astype(np.float32),
        "edge_index": mesh4x4_edge_index(),
        "run_id": np.array(run_id),
        "end_epoch": np.array(end_epoch, dtype=np.int32),
        "split": np.array(split),
        "active_cores": np.array(active_cores),
        "profile": np.array(profile),
        "seed": np.array(seed),
        "strength": np.array(strength),
        "attackers": np.array(attackers),
        "feature_cols": np.array(EXPECTED_FEATURES),
    }
    for name, array in arrays.items():
        np.save(root / f"{name}.npy", array)

    return AuditConfig(
        expected_samples=total_samples,
        expected_runs=len(runs),
        expected_samples_per_run=samples_per_run,
        expected_end_start=2,
        expected_end_stop=6,
        expected_window_epochs=window_epochs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only, chunked Stage 8D consistency audit for the V3 NoC graph dataset."
    )
    parser.add_argument("--dataset-root", type=Path, help="Directory containing unpacked .npy arrays.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for Stage 8D reports.")
    parser.add_argument("--chunk-size", type=int, default=512, help="Samples per x.npy chunk (default: 512).")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing Stage 8D outputs.")
    parser.add_argument("--smoke-test", action="store_true", help="Generate and audit a small synthetic dataset.")
    parser.add_argument("--expected-samples", type=int, default=233803)
    parser.add_argument("--expected-runs", type=int, default=71)
    parser.add_argument("--expected-samples-per-run", type=int, default=3293)
    parser.add_argument("--expected-end-start", type=int, default=7)
    parser.add_argument("--expected-end-stop", type=int, default=3299)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        raise AuditError("--chunk-size must be positive")

    if args.smoke_test:
        synthetic_root = args.output_root / "synthetic_dataset"
        if synthetic_root.exists() and not args.overwrite:
            raise AuditError(
                f"Synthetic dataset already exists: {synthetic_root}. "
                "Use a new smoke-test output directory or --overwrite."
            )
        if synthetic_root.exists() and args.overwrite:
            shutil.rmtree(synthetic_root)
        config = create_synthetic_dataset(synthetic_root)
        verdict = run_audit(
            dataset_root=synthetic_root,
            output_root=args.output_root,
            config=config,
            chunk_size=min(args.chunk_size, 4),
            overwrite=args.overwrite,
        )
    else:
        if args.dataset_root is None:
            raise AuditError("--dataset-root is required unless --smoke-test is used")
        config = AuditConfig(
            expected_samples=args.expected_samples,
            expected_runs=args.expected_runs,
            expected_samples_per_run=args.expected_samples_per_run,
            expected_end_start=args.expected_end_start,
            expected_end_stop=args.expected_end_stop,
            tolerance=args.tolerance,
        )
        verdict = run_audit(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            config=config,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
        )

    return 0 if verdict in {"PASS", "PASS WITH DOCUMENTED LIMITATIONS"} else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
