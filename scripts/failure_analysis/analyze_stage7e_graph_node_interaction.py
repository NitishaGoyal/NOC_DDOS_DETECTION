#!/usr/bin/env python3
"""
Stage 7E-lite: graph-head versus node-head interaction analysis.

This script reads saved prediction exports only. It does not load models,
run inference, train, modify the dataset, or invoke gem5.

Supported environment:
  Python 3.12+
  NumPy 2.x
  pandas 3.x
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"

REQUIRED_KEYS = (
    "model_name",
    "sample_order_within_export",
    "sample_index",
    "sample_id",
    "run_id",
    "split",
    "end_epoch",
    "true_graph",
    "true_attacker_mask",
    "true_attacker_count",
    "graph_logit",
    "graph_probability",
    "pred_graph_at_0_50",
    "graph_threshold",
    "node_logits",
    "node_probabilities",
    "pred_attacker_mask_at_0_50",
    "node_threshold_0_50",
    "pred_attacker_count_at_0_50",
    "exact_localization_at_0_50",
)

MODEL_FILES = (
    "conv1d_gcn_predictions.npz",
    "tcn_attention_gcn_predictions.npz",
    "tcn_maxpool_gcn_predictions.npz",
    "tcn_meanpool_gcn_predictions.npz",
)

HARD_ATTACK = "N-5-10-Pbursty-R51-A-12-S20-V3"
HARD_NORMAL = "N-3-7-8-12-Pmixed-R18-V3"

DECLARED_CONTROLS = (
    ("hard_attack", HARD_ATTACK),
    ("hard_normal", HARD_NORMAL),
    ("exact_benign_background", "N-5-10-Pbursty-R16-V3"),
    ("same_attacker12_s20_bursty_train_candidate", "N-0-6-Pbursty-R24-A-12-S20-V3"),
    ("same_attacker12_s20_stream_train_candidate", "N-0-15-Pstream-R20-A-12-S20-V3"),
    ("same_attacker12_s20_stream_test", "N-2-13-Pstream-R50-A-12-S20-V3"),
    ("same_attacker12_s20_mixed_train_candidate", "N-0-6-9-15-Pmixed-R26-A-12-S20-V3"),
    ("same_attacker12_s20_mixed_test", "N-1-6-9-14-Pmixed-R52-A-12-S20-V3"),
)

OUTPUT_FILES = (
    "STAGE7E_GRAPH_NODE_INTERACTION_REPORT.md",
    "logs/48_stage7e_validation.txt",
    "tables/stage7e_input_inventory.csv",
    "tables/stage7e_prediction_array_manifest.csv",
    "tables/stage7e_alignment_checks.csv",
    "tables/stage7e_model_thresholds.csv",
    "tables/stage7e_run_model_summary.csv",
    "tables/stage7e_graph_node_categories.csv",
    "tables/stage7e_hard_attack_window_details.csv",
    "tables/stage7e_hard_attack_temporal_blocks.csv",
    "tables/stage7e_control_run_comparison.csv",
    "tables/stage7e_attacker_rank_summary.csv",
    "tables/stage7e_graph_node_probability_summary.csv",
    "tables/stage7e_neighbour_spread_summary.csv",
    "tables/stage7e_cross_model_diagnosis.csv",
    "tables/stage7e_summary.json",
)


class Stage7EError(RuntimeError):
    """Raised when Stage 7E cannot safely proceed."""


@dataclass(frozen=True)
class Paths:
    predictions_dir: Path
    metadata_csv: Path
    dataset_root: Path
    selected_thresholds_csv: Path
    output_root: Path


@dataclass
class ModelExport:
    name: str
    path: Path
    arrays: dict[str, np.ndarray]
    reference_graph_threshold: float
    node_threshold: float
    selected_graph_threshold: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_value(array: np.ndarray, name: str) -> Any:
    value = np.asarray(array)
    if value.size != 1:
        raise Stage7EError(f"{name} must contain exactly one value; got shape {value.shape}")
    return value.reshape(-1)[0].item()


def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def ensure_output_is_safe(output_root: Path, overwrite: bool) -> None:
    existing = [output_root / rel for rel in OUTPUT_FILES if (output_root / rel).exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise Stage7EError(
            "Refusing to overwrite existing Stage 7E outputs. "
            "Use --overwrite only after reviewing them:\n" + formatted
        )


def parse_attackers(text: Any, num_nodes: int = 16) -> tuple[int, ...]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ()
    cleaned = str(text).strip()
    if cleaned in {"", "nan", "None"}:
        return ()
    cleaned = cleaned.replace(",", "-")
    parsed: list[int] = []
    for token in cleaned.split("-"):
        if token == "":
            continue
        number = float(token)
        if not number.is_integer():
            raise Stage7EError(f"Non-integer attacker ID in {text!r}")
        parsed.append(int(number))
    values = tuple(parsed)
    if len(set(values)) != len(values):
        raise Stage7EError(f"Duplicate attacker IDs in {text!r}")
    if any(value < 0 or value >= num_nodes for value in values):
        raise Stage7EError(f"Attacker ID outside 0..{num_nodes - 1}: {text!r}")
    return values


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 2:
        return float("nan")
    left = left[mask]
    right = right[mask]
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_minimum": float("nan"),
            f"{prefix}_maximum": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p25": float("nan"),
            f"{prefix}_p75": float("nan"),
            f"{prefix}_p95": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_minimum": float(np.min(values)),
        f"{prefix}_maximum": float(np.max(values)),
        f"{prefix}_p05": float(np.quantile(values, 0.05)),
        f"{prefix}_p25": float(np.quantile(values, 0.25)),
        f"{prefix}_p75": float(np.quantile(values, 0.75)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
    }


def rank_descending(probabilities: np.ndarray, true_mask: np.ndarray) -> dict[str, np.ndarray]:
    """
    Rank 1 is best. Ties use stable router-ID order via mergesort.
    Returns per-window best/worst true rank and true probability summaries.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    true_mask = np.asarray(true_mask, dtype=bool)
    order = np.argsort(-probabilities, axis=1, kind="mergesort")
    inverse = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    inverse[rows, order] = np.arange(order.shape[1])[None, :]
    ranks = inverse + 1

    true_count = true_mask.sum(axis=1)
    attack_rows = true_count > 0

    best_rank = np.full(probabilities.shape[0], np.nan, dtype=np.float64)
    worst_rank = np.full(probabilities.shape[0], np.nan, dtype=np.float64)
    mean_true_prob = np.full(probabilities.shape[0], np.nan, dtype=np.float64)
    min_true_prob = np.full(probabilities.shape[0], np.nan, dtype=np.float64)
    max_true_prob = np.full(probabilities.shape[0], np.nan, dtype=np.float64)
    margin = np.full(probabilities.shape[0], np.nan, dtype=np.float64)

    if attack_rows.any():
        attack_mask = true_mask[attack_rows]
        attack_prob = probabilities[attack_rows]
        attack_ranks = ranks[attack_rows]

        best_rank[attack_rows] = np.where(
            attack_mask, attack_ranks, probabilities.shape[1] + 1
        ).min(axis=1)
        worst_rank[attack_rows] = np.where(attack_mask, attack_ranks, 0).max(axis=1)

        true_sum = np.where(attack_mask, attack_prob, 0.0).sum(axis=1)
        mean_true_prob[attack_rows] = true_sum / true_count[attack_rows]
        min_true_prob[attack_rows] = np.where(
            attack_mask, attack_prob, np.inf
        ).min(axis=1)
        max_true_prob[attack_rows] = np.where(
            attack_mask, attack_prob, -np.inf
        ).max(axis=1)
        max_nontrue = np.where(~attack_mask, attack_prob, -np.inf).max(axis=1)
        margin[attack_rows] = min_true_prob[attack_rows] - max_nontrue

    return {
        "ranks": ranks,
        "best_true_rank": best_rank,
        "worst_true_rank": worst_rank,
        "mean_true_probability": mean_true_prob,
        "minimum_true_probability": min_true_prob,
        "maximum_true_probability": max_true_prob,
        "true_vs_best_nontrue_margin": margin,
        "maximum_node_probability": probabilities.max(axis=1),
        "mean_node_probability": probabilities.mean(axis=1),
    }


def node_categories(
    true_mask: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    true_mask = np.asarray(true_mask, dtype=bool)
    predicted = np.asarray(probabilities >= threshold, dtype=bool)
    categories: list[str] = []

    for truth, pred in zip(true_mask, predicted, strict=True):
        true_count = int(truth.sum())
        pred_count = int(pred.sum())
        overlap = int(np.logical_and(truth, pred).sum())

        if true_count == 0:
            categories.append("node_clean" if pred_count == 0 else "node_false_localization")
        elif pred_count == 0:
            categories.append("empty")
        elif np.array_equal(truth, pred):
            categories.append("exact")
        elif overlap == true_count and pred_count > true_count:
            categories.append("overpredicting")
        elif 0 < overlap < true_count:
            categories.append("partial")
        elif overlap == 0:
            categories.append("wrong_source")
        else:
            categories.append("other")

    return np.asarray(categories, dtype="<U24")


def build_neighbour_sets(edge_index: np.ndarray, node: int, num_nodes: int = 16) -> tuple[set[int], set[int], set[int]]:
    adjacency = {router: set() for router in range(num_nodes)}
    for source, target in np.asarray(edge_index, dtype=np.int64).T:
        if source == target:
            continue
        adjacency[int(source)].add(int(target))
        adjacency[int(target)].add(int(source))

    one_hop = set(adjacency[node])
    two_hop: set[int] = set()
    for neighbor in one_hop:
        two_hop.update(adjacency[neighbor])
    two_hop.discard(node)
    two_hop.difference_update(one_hop)
    remaining = set(range(num_nodes)) - {node} - one_hop - two_hop
    return one_hop, two_hop, remaining


def spread_category(
    probabilities: np.ndarray,
    attacker: int,
    one_hop: set[int],
    node_threshold: float,
) -> np.ndarray:
    result: list[str] = []
    for row in np.asarray(probabilities, dtype=np.float64):
        attacker_probability = float(row[attacker])
        other = row.copy()
        other[attacker] = -np.inf
        best_other = int(np.argmax(other))
        margin = attacker_probability - float(np.max(other))
        predicted_count = int(np.sum(row >= node_threshold))

        if float(np.max(row)) < node_threshold or (
            attacker_probability < 0.25 and float(np.max(row)) < 0.5
        ):
            result.append("absent")
        elif int(np.argmax(row)) == attacker and margin >= 0.10:
            result.append("localized")
        elif attacker_probability >= 0.25 and best_other in one_hop:
            result.append("one_hop_spread")
        elif predicted_count >= 4 or margin < 0.0:
            result.append("diffuse")
        else:
            result.append("diffuse")
    return np.asarray(result, dtype="<U24")


def read_selected_thresholds(path: Path) -> dict[str, float]:
    if not path.exists():
        raise Stage7EError(f"Selected graph-threshold table not found: {path}")
    frame = pd.read_csv(path)
    required = {"model_name", "selected_threshold"}
    missing = required - set(frame.columns)
    if missing:
        raise Stage7EError(f"{path} missing columns: {sorted(missing)}")
    result: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        name = str(getattr(row, "model_name"))
        threshold = float(getattr(row, "selected_threshold"))
        if not (0.0 <= threshold <= 1.0):
            raise Stage7EError(f"Selected threshold outside [0,1] for {name}: {threshold}")
        result[name] = threshold
    return result


def load_dataset_arrays(dataset_root: Path) -> dict[str, np.ndarray]:
    required = ("run_id.npy", "end_epoch.npy", "split.npy", "y_graph.npy", "y_node.npy", "edge_index.npy")
    missing = [name for name in required if not (dataset_root / name).exists()]
    if missing:
        raise Stage7EError(f"Dataset root is missing arrays: {missing}")

    arrays = {
        "run_id": np.load(dataset_root / "run_id.npy", mmap_mode="r", allow_pickle=False),
        "end_epoch": np.load(dataset_root / "end_epoch.npy", mmap_mode="r", allow_pickle=False),
        "split": np.load(dataset_root / "split.npy", mmap_mode="r", allow_pickle=False),
        "y_graph": np.load(dataset_root / "y_graph.npy", mmap_mode="r", allow_pickle=False),
        "y_node": np.load(dataset_root / "y_node.npy", mmap_mode="r", allow_pickle=False),
        "edge_index": np.load(dataset_root / "edge_index.npy", mmap_mode="r", allow_pickle=False),
    }
    return arrays


def load_model_export(path: Path, selected_thresholds: Mapping[str, float]) -> ModelExport:
    if not path.exists():
        raise Stage7EError(f"Prediction export not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_KEYS if key not in archive.files]
        if missing:
            raise Stage7EError(f"{path.name} missing prediction arrays: {missing}")
        arrays = {key: np.asarray(archive[key]) for key in archive.files}

    name = str(scalar_value(arrays["model_name"], f"{path.name}:model_name"))
    reference_graph_threshold = float(
        scalar_value(arrays["graph_threshold"], f"{path.name}:graph_threshold")
    )
    node_threshold = float(
        scalar_value(arrays["node_threshold_0_50"], f"{path.name}:node_threshold_0_50")
    )
    if name not in selected_thresholds:
        raise Stage7EError(
            f"No validation-selected graph threshold for model {name!r} in selected threshold table"
        )
    return ModelExport(
        name=name,
        path=path,
        arrays=arrays,
        reference_graph_threshold=reference_graph_threshold,
        node_threshold=node_threshold,
        selected_graph_threshold=float(selected_thresholds[name]),
    )


def file_inventory(paths: Paths, model_exports: Sequence[ModelExport]) -> pd.DataFrame:
    entries: list[dict[str, Any]] = []
    input_paths = [
        ("prediction_metadata", paths.metadata_csv),
        ("selected_graph_thresholds", paths.selected_thresholds_csv),
        ("dataset_run_id", paths.dataset_root / "run_id.npy"),
        ("dataset_end_epoch", paths.dataset_root / "end_epoch.npy"),
        ("dataset_split", paths.dataset_root / "split.npy"),
        ("dataset_y_graph", paths.dataset_root / "y_graph.npy"),
        ("dataset_y_node", paths.dataset_root / "y_node.npy"),
        ("dataset_edge_index", paths.dataset_root / "edge_index.npy"),
    ]
    input_paths += [(f"prediction_{model.name}", model.path) for model in model_exports]

    for role, path in input_paths:
        stat = path.stat()
        entries.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "size_bytes": int(stat.st_size),
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(entries)


def prediction_array_manifest(model_exports: Sequence[ModelExport]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in model_exports:
        for key in sorted(model.arrays):
            array = np.asarray(model.arrays[key])
            numeric = np.issubdtype(array.dtype, np.number)
            finite = bool(np.isfinite(array).all()) if numeric else True
            minimum = float(np.min(array)) if numeric and array.size else float("nan")
            maximum = float(np.max(array)) if numeric and array.size else float("nan")
            rows.append(
                {
                    "model_name": model.name,
                    "prediction_file": str(model.path.resolve()),
                    "array_key": key,
                    "shape": "x".join(str(dim) for dim in array.shape),
                    "ndim": int(array.ndim),
                    "dtype": str(array.dtype),
                    "size": int(array.size),
                    "finite": finite,
                    "minimum": minimum,
                    "maximum": maximum,
                }
            )
    return pd.DataFrame(rows)


def add_check(
    rows: list[dict[str, Any]],
    model_name: str,
    check: str,
    passed: bool,
    detail: str,
    severity: str = "hard",
) -> None:
    rows.append(
        {
            "model_name": model_name,
            "check": check,
            "passed": bool(passed),
            "severity": severity,
            "detail": detail,
        }
    )


def validate_alignment(
    model_exports: Sequence[ModelExport],
    metadata: pd.DataFrame,
    dataset: Mapping[str, np.ndarray],
    smoke_test: bool,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    metadata_required = {
        "sample_order_within_export",
        "sample_index",
        "sample_id",
        "run_id",
        "split",
        "end_epoch",
        "true_graph",
        "true_attacker_count",
        "profile",
        "active_cores",
        "attackers",
        "strength",
        "seed",
    }
    missing_metadata = metadata_required - set(metadata.columns)
    add_check(
        checks,
        "__global__",
        "prediction metadata contains required columns",
        not missing_metadata,
        f"missing={sorted(missing_metadata)}",
    )
    if missing_metadata:
        return pd.DataFrame(checks)

    expected_count = len(metadata)
    add_check(
        checks,
        "__global__",
        "real export sample count equals 85618",
        smoke_test or expected_count == 85618,
        f"observed={expected_count}; smoke_test={smoke_test}",
    )
    add_check(
        checks,
        "__global__",
        "metadata sample_index unique",
        bool(metadata["sample_index"].is_unique),
        f"duplicates={int(metadata['sample_index'].duplicated().sum())}",
    )
    add_check(
        checks,
        "__global__",
        "metadata sample_id unique",
        bool(metadata["sample_id"].is_unique),
        f"duplicates={int(metadata['sample_id'].duplicated().sum())}",
    )
    pair_duplicates = int(metadata.duplicated(["run_id", "end_epoch"]).sum())
    add_check(
        checks,
        "__global__",
        "metadata run_id/end_epoch keys unique",
        pair_duplicates == 0,
        f"duplicates={pair_duplicates}",
    )

    dataset_length = len(dataset["run_id"])
    sample_indices = metadata["sample_index"].to_numpy(dtype=np.int64)
    indices_in_range = bool(
        sample_indices.size == 0
        or (sample_indices.min() >= 0 and sample_indices.max() < dataset_length)
    )
    add_check(
        checks,
        "__global__",
        "sample indices lie inside dataset arrays",
        indices_in_range,
        (
            f"minimum={sample_indices.min() if sample_indices.size else 'n/a'}; "
            f"maximum={sample_indices.max() if sample_indices.size else 'n/a'}; "
            f"dataset_length={dataset_length}"
        ),
    )

    if indices_in_range:
        dataset_run = np.asarray(dataset["run_id"][sample_indices]).astype(str)
        dataset_end = np.asarray(dataset["end_epoch"][sample_indices], dtype=np.int64)
        dataset_split = np.asarray(dataset["split"][sample_indices]).astype(str)
        dataset_graph = np.asarray(dataset["y_graph"][sample_indices], dtype=np.float64)
        dataset_node = np.asarray(dataset["y_node"][sample_indices], dtype=np.float64)

        add_check(
            checks,
            "__global__",
            "metadata run_id matches dataset",
            bool(np.array_equal(dataset_run, metadata["run_id"].astype(str).to_numpy())),
            "comparison by sample_index",
        )
        add_check(
            checks,
            "__global__",
            "metadata end_epoch matches dataset",
            bool(np.array_equal(dataset_end, metadata["end_epoch"].to_numpy(dtype=np.int64))),
            "comparison by sample_index",
        )
        add_check(
            checks,
            "__global__",
            "metadata split matches dataset",
            bool(np.array_equal(dataset_split, metadata["split"].astype(str).to_numpy())),
            "comparison by sample_index",
        )
        add_check(
            checks,
            "__global__",
            "metadata true_graph matches dataset",
            bool(np.array_equal(dataset_graph, metadata["true_graph"].to_numpy(dtype=np.float64))),
            "comparison by sample_index",
        )
    else:
        dataset_node = np.empty((0, 16), dtype=np.float64)

    common_keys: tuple[np.ndarray, np.ndarray] | None = None

    for model in model_exports:
        arrays = model.arrays
        lengths = {}
        for key in REQUIRED_KEYS:
            array = np.asarray(arrays[key])
            if key in {"model_name", "graph_threshold", "node_threshold_0_50"}:
                continue
            lengths[key] = array.shape[0]

        add_check(
            checks,
            model.name,
            "all sample arrays have metadata length",
            all(length == expected_count for length in lengths.values()),
            f"expected={expected_count}; lengths={lengths}",
        )

        node_shapes_ok = (
            np.asarray(arrays["true_attacker_mask"]).shape == (expected_count, 16)
            and np.asarray(arrays["node_logits"]).shape == (expected_count, 16)
            and np.asarray(arrays["node_probabilities"]).shape == (expected_count, 16)
            and np.asarray(arrays["pred_attacker_mask_at_0_50"]).shape == (expected_count, 16)
        )
        add_check(
            checks,
            model.name,
            "node arrays have shape [samples,16]",
            node_shapes_ok,
            (
                f"true={np.asarray(arrays['true_attacker_mask']).shape}; "
                f"logits={np.asarray(arrays['node_logits']).shape}; "
                f"probabilities={np.asarray(arrays['node_probabilities']).shape}; "
                f"predictions={np.asarray(arrays['pred_attacker_mask_at_0_50']).shape}"
            ),
        )

        sample_index = np.asarray(arrays["sample_index"], dtype=np.int64)
        sample_id = np.asarray(arrays["sample_id"]).astype(str)
        run_id = np.asarray(arrays["run_id"]).astype(str)
        split = np.asarray(arrays["split"]).astype(str)
        end_epoch = np.asarray(arrays["end_epoch"], dtype=np.int64)
        true_graph = np.asarray(arrays["true_graph"], dtype=np.float64)
        true_node = np.asarray(arrays["true_attacker_mask"], dtype=np.float64)
        graph_probability = np.asarray(arrays["graph_probability"], dtype=np.float64)
        graph_logit = np.asarray(arrays["graph_logit"], dtype=np.float64)
        node_probability = np.asarray(arrays["node_probabilities"], dtype=np.float64)
        node_logit = np.asarray(arrays["node_logits"], dtype=np.float64)

        add_check(
            checks,
            model.name,
            "sample order matches prediction_metadata.csv",
            bool(
                np.array_equal(sample_index, metadata["sample_index"].to_numpy(dtype=np.int64))
                and np.array_equal(sample_id, metadata["sample_id"].astype(str).to_numpy())
                and np.array_equal(run_id, metadata["run_id"].astype(str).to_numpy())
                and np.array_equal(split, metadata["split"].astype(str).to_numpy())
                and np.array_equal(end_epoch, metadata["end_epoch"].to_numpy(dtype=np.int64))
            ),
            "sample_index, sample_id, run_id, split, end_epoch compared in order",
        )
        add_check(
            checks,
            model.name,
            "true graph labels match metadata",
            bool(np.array_equal(true_graph, metadata["true_graph"].to_numpy(dtype=np.float64))),
            "exact array equality",
        )
        if indices_in_range:
            add_check(
                checks,
                model.name,
                "true attacker masks match dataset y_node",
                bool(np.array_equal(true_node, dataset_node)),
                "comparison by sample_index",
            )

        probabilities_ok = bool(
            np.isfinite(graph_probability).all()
            and np.isfinite(node_probability).all()
            and np.all((graph_probability >= 0.0) & (graph_probability <= 1.0))
            and np.all((node_probability >= 0.0) & (node_probability <= 1.0))
        )
        add_check(
            checks,
            model.name,
            "graph and node probabilities finite and inside [0,1]",
            probabilities_ok,
            (
                f"graph_range=({graph_probability.min():.8g},{graph_probability.max():.8g}); "
                f"node_range=({node_probability.min():.8g},{node_probability.max():.8g})"
            ),
        )
        logits_ok = bool(np.isfinite(graph_logit).all() and np.isfinite(node_logit).all())
        add_check(
            checks,
            model.name,
            "graph and node logits finite",
            logits_ok,
            f"graph_finite={np.isfinite(graph_logit).all()}; node_finite={np.isfinite(node_logit).all()}",
        )

        saved_graph = np.asarray(arrays["pred_graph_at_0_50"], dtype=bool)
        recomputed_graph = graph_probability >= model.reference_graph_threshold
        saved_node = np.asarray(arrays["pred_attacker_mask_at_0_50"], dtype=bool)
        recomputed_node = node_probability >= model.node_threshold
        add_check(
            checks,
            model.name,
            "saved 0.50 graph predictions reproduce from probabilities",
            bool(np.array_equal(saved_graph, recomputed_graph)),
            f"threshold={model.reference_graph_threshold}",
        )
        add_check(
            checks,
            model.name,
            "saved 0.50 node predictions reproduce from probabilities",
            bool(np.array_equal(saved_node, recomputed_node)),
            f"threshold={model.node_threshold}",
        )

        if common_keys is None:
            common_keys = (sample_id.copy(), sample_index.copy())
        else:
            add_check(
                checks,
                model.name,
                "sample keys identical across all model exports",
                bool(
                    np.array_equal(common_keys[0], sample_id)
                    and np.array_equal(common_keys[1], sample_index)
                ),
                "compared with first model export",
            )

    return pd.DataFrame(checks)


def model_threshold_table(model_exports: Sequence[ModelExport]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in model_exports:
        rows.append(
            {
                "model_name": model.name,
                "reference_graph_threshold": model.reference_graph_threshold,
                "validation_selected_graph_threshold": model.selected_graph_threshold,
                "node_threshold": model.node_threshold,
                "node_threshold_source": "saved_reference_0.50; no test tuning",
            }
        )
    return pd.DataFrame(rows)


def model_frame(model: ModelExport, metadata: pd.DataFrame) -> pd.DataFrame:
    arrays = model.arrays
    frame = metadata.copy()
    frame["model_name"] = model.name
    frame["graph_logit"] = np.asarray(arrays["graph_logit"], dtype=np.float64)
    frame["graph_probability"] = np.asarray(arrays["graph_probability"], dtype=np.float64)
    frame["pred_graph_reference"] = (
        frame["graph_probability"].to_numpy() >= model.reference_graph_threshold
    ).astype(np.int8)
    frame["pred_graph_selected"] = (
        frame["graph_probability"].to_numpy() >= model.selected_graph_threshold
    ).astype(np.int8)
    return frame


def summarize_run_operating_point(
    frame: pd.DataFrame,
    node_probability: np.ndarray,
    true_node: np.ndarray,
    node_category: np.ndarray,
    rank_info: Mapping[str, np.ndarray],
    threshold_name: str,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prediction_column = "pred_graph_reference" if threshold_name == "reference_0_50" else "pred_graph_selected"

    for run_id, group in frame.groupby("run_id", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        true_graph_values = group["true_graph"].unique()
        if len(true_graph_values) != 1:
            raise Stage7EError(f"Run {run_id} has changing graph labels")
        true_graph = int(true_graph_values[0])
        pred = group[prediction_column].to_numpy(dtype=np.int8)
        correct = pred == true_graph

        row: dict[str, Any] = {
            "model_name": str(group["model_name"].iloc[0]),
            "operating_point": threshold_name,
            "graph_threshold": threshold,
            "run_id": run_id,
            "split": str(group["split"].iloc[0]),
            "true_graph": true_graph,
            "profile": str(group["profile"].iloc[0]),
            "active_cores": str(group["active_cores"].iloc[0]),
            "attackers": "" if pd.isna(group["attackers"].iloc[0]) else str(group["attackers"].iloc[0]),
            "strength": "" if pd.isna(group["strength"].iloc[0]) else str(group["strength"].iloc[0]),
            "number_of_windows": len(group),
            "graph_correct_count": int(correct.sum()),
            "graph_wrong_count": int((~correct).sum()),
            "graph_recall": float(pred.mean()) if true_graph == 1 else float("nan"),
            "normal_false_positive_rate": float(pred.mean()) if true_graph == 0 else float("nan"),
        }
        row.update(quantiles(group["graph_probability"].to_numpy(dtype=np.float64), "graph_probability"))

        categories = node_category[indices]
        counts = Counter(categories.tolist())
        for category in (
            "node_clean",
            "node_false_localization",
            "exact",
            "overpredicting",
            "partial",
            "wrong_source",
            "empty",
            "other",
        ):
            row[f"node_{category}_count"] = int(counts.get(category, 0))
            row[f"node_{category}_rate"] = float(counts.get(category, 0) / len(group))

        attack_rows = true_node[indices].sum(axis=1) > 0
        if attack_rows.any():
            best_rank = np.asarray(rank_info["best_true_rank"])[indices]
            mean_true_probability = np.asarray(rank_info["mean_true_probability"])[indices]
            margin = np.asarray(rank_info["true_vs_best_nontrue_margin"])[indices]
            row["true_attacker_top1_rate"] = float(np.mean(best_rank == 1))
            row["true_attacker_top3_rate"] = float(np.mean(best_rank <= 3))
            row["true_attacker_mean_rank"] = float(np.nanmean(best_rank))
            row["true_attacker_mean_probability"] = float(np.nanmean(mean_true_probability))
            row["true_attacker_positive_margin_rate"] = float(np.nanmean(margin > 0))
            row["true_attacker_mean_margin"] = float(np.nanmean(margin))
        else:
            row["true_attacker_top1_rate"] = float("nan")
            row["true_attacker_top3_rate"] = float("nan")
            row["true_attacker_mean_rank"] = float("nan")
            row["true_attacker_mean_probability"] = float("nan")
            row["true_attacker_positive_margin_rate"] = float("nan")
            row["true_attacker_mean_margin"] = float("nan")
        rows.append(row)
    return rows


def graph_node_category_table(
    frame: pd.DataFrame,
    node_category: np.ndarray,
    rank_info: Mapping[str, np.ndarray],
    model: ModelExport,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for operating_point, pred_col, threshold in (
        ("reference_0_50", "pred_graph_reference", model.reference_graph_threshold),
        ("validation_selected", "pred_graph_selected", model.selected_graph_threshold),
    ):
        graph_correct = frame[pred_col].to_numpy(dtype=np.int8) == frame["true_graph"].to_numpy(dtype=np.int8)
        best_rank = np.asarray(rank_info["best_true_rank"])
        mean_true_prob = np.asarray(rank_info["mean_true_probability"])
        informative = (best_rank <= 3) | (mean_true_prob >= model.node_threshold)

        for split in ("val", "test"):
            split_mask = frame["split"].astype(str).to_numpy() == split
            for graph_status, graph_mask in (
                ("graph_correct", graph_correct),
                ("graph_wrong", ~graph_correct),
            ):
                mask = split_mask & graph_mask
                if not mask.any():
                    continue
                counts = Counter(node_category[mask].tolist())
                for category, count in sorted(counts.items()):
                    rows.append(
                        {
                            "model_name": model.name,
                            "operating_point": operating_point,
                            "graph_threshold": threshold,
                            "split": split,
                            "graph_status": graph_status,
                            "node_category": category,
                            "count": int(count),
                            "rate_within_graph_status": float(count / mask.sum()),
                        }
                    )
                attack_mask = mask & (frame["true_graph"].to_numpy(dtype=np.int8) == 1)
                rows.append(
                    {
                        "model_name": model.name,
                        "operating_point": operating_point,
                        "graph_threshold": threshold,
                        "split": split,
                        "graph_status": graph_status,
                        "node_category": "threshold_independent_informative",
                        "count": int(np.sum(attack_mask & informative)),
                        "rate_within_graph_status": (
                            float(np.sum(attack_mask & informative) / attack_mask.sum())
                            if attack_mask.sum()
                            else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def temporal_blocks(end_epochs: np.ndarray) -> np.ndarray:
    order = np.argsort(end_epochs, kind="mergesort")
    labels = np.empty(len(end_epochs), dtype="<U8")
    blocks = np.array_split(order, 3)
    for name, indices in zip(("early", "middle", "late"), blocks, strict=True):
        labels[indices] = name
    return labels


def hard_attack_analysis(
    model: ModelExport,
    frame: pd.DataFrame,
    true_node: np.ndarray,
    node_probability: np.ndarray,
    node_category: np.ndarray,
    rank_info: Mapping[str, np.ndarray],
    edge_index: np.ndarray,
    hard_attack: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mask = frame["run_id"].astype(str).to_numpy() == hard_attack
    if not mask.any():
        raise Stage7EError(f"Hard attack missing from prediction export: {hard_attack}")

    indices = np.flatnonzero(mask)
    sub = frame.loc[indices].copy().reset_index(drop=True)
    if len(sub) != 3293 and len(frame) == 85618:
        raise Stage7EError(f"Hard attack should have 3293 windows; got {len(sub)}")
    attackers = parse_attackers(sub["attackers"].iloc[0])
    if attackers != (12,):
        raise Stage7EError(f"Hard attack attacker metadata should be (12,), got {attackers}")

    attacker = 12
    one_hop, two_hop, remaining = build_neighbour_sets(edge_index, attacker)
    probabilities = node_probability[indices]
    true_prob = probabilities[:, attacker]
    max_nonattacker = np.max(np.delete(probabilities, attacker, axis=1), axis=1)
    rank = np.asarray(rank_info["best_true_rank"])[indices]
    margin = np.asarray(rank_info["true_vs_best_nontrue_margin"])[indices]
    category = node_category[indices]
    spread = spread_category(probabilities, attacker, one_hop, model.node_threshold)
    blocks = temporal_blocks(sub["end_epoch"].to_numpy(dtype=np.int64))

    one_hop_ids = sorted(one_hop)
    two_hop_ids = sorted(two_hop)
    remaining_ids = sorted(remaining)

    details = pd.DataFrame(
        {
            "model_name": model.name,
            "sample_index": sub["sample_index"].to_numpy(dtype=np.int64),
            "sample_id": sub["sample_id"].astype(str).to_numpy(),
            "run_id": hard_attack,
            "split": sub["split"].astype(str).to_numpy(),
            "end_epoch": sub["end_epoch"].to_numpy(dtype=np.int64),
            "temporal_block": blocks,
            "graph_probability": sub["graph_probability"].to_numpy(dtype=np.float64),
            "pred_graph_reference": sub["pred_graph_reference"].to_numpy(dtype=np.int8),
            "pred_graph_selected": sub["pred_graph_selected"].to_numpy(dtype=np.int8),
            "router12_probability": true_prob,
            "router12_rank": rank,
            "router12_margin_over_best_nonattacker": margin,
            "best_nonattacker_probability": max_nonattacker,
            "maximum_node_probability": probabilities.max(axis=1),
            "mean_node_probability": probabilities.mean(axis=1),
            "node_category_at_0_50": category,
            "node_informative_top3_or_p50": (rank <= 3) | (true_prob >= model.node_threshold),
            "one_hop_max_probability": probabilities[:, one_hop_ids].max(axis=1),
            "one_hop_mean_probability": probabilities[:, one_hop_ids].mean(axis=1),
            "two_hop_max_probability": probabilities[:, two_hop_ids].max(axis=1),
            "two_hop_mean_probability": probabilities[:, two_hop_ids].mean(axis=1),
            "remaining_mean_probability": (
                probabilities[:, remaining_ids].mean(axis=1)
                if remaining_ids
                else np.full(len(probabilities), np.nan)
            ),
            "spread_category": spread,
        }
    )

    block_rows: list[dict[str, Any]] = []
    for operating_point, pred_col, threshold in (
        ("reference_0_50", "pred_graph_reference", model.reference_graph_threshold),
        ("validation_selected", "pred_graph_selected", model.selected_graph_threshold),
    ):
        for block in ("early", "middle", "late"):
            block_frame = details[details["temporal_block"] == block]
            graph_prediction = block_frame[pred_col].to_numpy(dtype=np.int8)
            wrong = graph_prediction == 0
            informative = block_frame["node_informative_top3_or_p50"].to_numpy(dtype=bool)
            block_rows.append(
                {
                    "model_name": model.name,
                    "operating_point": operating_point,
                    "graph_threshold": threshold,
                    "temporal_block": block,
                    "window_count": len(block_frame),
                    "end_epoch_minimum": int(block_frame["end_epoch"].min()),
                    "end_epoch_maximum": int(block_frame["end_epoch"].max()),
                    "graph_recall": float(graph_prediction.mean()),
                    "mean_graph_probability": float(block_frame["graph_probability"].mean()),
                    "mean_router12_probability": float(block_frame["router12_probability"].mean()),
                    "router12_top1_rate": float((block_frame["router12_rank"] == 1).mean()),
                    "router12_top3_rate": float((block_frame["router12_rank"] <= 3).mean()),
                    "graph_wrong_node_informative_rate": (
                        float(informative[wrong].mean()) if wrong.any() else float("nan")
                    ),
                }
            )

    rank_row: dict[str, Any] = {
        "model_name": model.name,
        "run_id": hard_attack,
        "window_count": len(details),
        "router12_mean_rank": float(details["router12_rank"].mean()),
        "router12_median_rank": float(details["router12_rank"].median()),
        "router12_top1_rate": float((details["router12_rank"] == 1).mean()),
        "router12_top3_rate": float((details["router12_rank"] <= 3).mean()),
        "router12_positive_margin_rate": float(
            (details["router12_margin_over_best_nonattacker"] > 0).mean()
        ),
    }
    rank_row.update(quantiles(details["router12_probability"].to_numpy(), "router12_probability"))
    rank_row["router12_mean_margin"] = float(
        details["router12_margin_over_best_nonattacker"].mean()
    )

    spread_counts = Counter(details["spread_category"].tolist())
    spread_row = {
        "model_name": model.name,
        "run_id": hard_attack,
        "attacker_router": attacker,
        "one_hop_routers": "-".join(map(str, one_hop_ids)),
        "two_hop_routers": "-".join(map(str, two_hop_ids)),
        "remaining_routers": "-".join(map(str, remaining_ids)),
        "mean_attacker_probability": float(details["router12_probability"].mean()),
        "mean_one_hop_max_probability": float(details["one_hop_max_probability"].mean()),
        "mean_one_hop_mean_probability": float(details["one_hop_mean_probability"].mean()),
        "mean_two_hop_max_probability": float(details["two_hop_max_probability"].mean()),
        "mean_two_hop_mean_probability": float(details["two_hop_mean_probability"].mean()),
        "mean_remaining_probability": float(details["remaining_mean_probability"].mean()),
    }
    for name in ("localized", "one_hop_spread", "diffuse", "absent"):
        spread_row[f"{name}_count"] = int(spread_counts.get(name, 0))
        spread_row[f"{name}_rate"] = float(spread_counts.get(name, 0) / len(details))

    return (
        details,
        pd.DataFrame(block_rows),
        pd.DataFrame([rank_row]),
        pd.DataFrame([spread_row]),
    )


def control_run_ids(metadata: pd.DataFrame) -> list[tuple[str, str]]:
    controls = list(DECLARED_CONTROLS)
    seen = {run_id for _, run_id in controls}

    # Add every exported single-attacker S20 attack without cherry-picking success.
    unique_runs = metadata.drop_duplicates("run_id")
    for row in unique_runs.itertuples(index=False):
        attackers = parse_attackers(getattr(row, "attackers"))
        strength = "" if pd.isna(getattr(row, "strength")) else str(getattr(row, "strength"))
        if len(attackers) == 1 and strength == "20":
            run_id = str(getattr(row, "run_id"))
            if run_id not in seen:
                controls.append(("all_exported_single_attacker_s20", run_id))
                seen.add(run_id)
    return controls


def control_comparison(
    run_summary: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    available_runs = set(metadata["run_id"].astype(str))
    for control_role, run_id in control_run_ids(metadata):
        if run_id not in available_runs:
            rows.append(
                {
                    "control_role": control_role,
                    "run_id": run_id,
                    "present_in_export": False,
                    "model_name": "",
                    "operating_point": "",
                    "missing_reason": "run is absent from exported val/test cohort",
                }
            )
            continue
        matched = run_summary[run_summary["run_id"] == run_id]
        for row in matched.to_dict(orient="records"):
            row = dict(row)
            row["control_role"] = control_role
            row["present_in_export"] = True
            row["missing_reason"] = ""
            rows.append(row)
    return pd.DataFrame(rows)


def probability_summary(
    model: ModelExport,
    frame: pd.DataFrame,
    true_node: np.ndarray,
    node_probability: np.ndarray,
    hard_attack: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rank_info = rank_descending(node_probability, true_node)

    cohorts = {
        "all_attack_val_test": frame["true_graph"].to_numpy(dtype=np.int8) == 1,
        "hard_attack": frame["run_id"].astype(str).to_numpy() == hard_attack,
        "all_test_attacks": (
            (frame["true_graph"].to_numpy(dtype=np.int8) == 1)
            & (frame["split"].astype(str).to_numpy() == "test")
        ),
    }

    for cohort, mask in cohorts.items():
        if not mask.any():
            continue
        graph_probability = frame.loc[mask, "graph_probability"].to_numpy(dtype=np.float64)
        maximum_node = node_probability[mask].max(axis=1)
        mean_true = np.asarray(rank_info["mean_true_probability"])[mask]
        rows.append(
            {
                "model_name": model.name,
                "cohort": cohort,
                "window_count": int(mask.sum()),
                "graph_vs_max_node_pearson": safe_pearson(graph_probability, maximum_node),
                "graph_vs_mean_true_attacker_pearson": safe_pearson(graph_probability, mean_true),
                "mean_graph_probability": float(graph_probability.mean()),
                "mean_maximum_node_probability": float(maximum_node.mean()),
                "mean_true_attacker_probability": float(np.nanmean(mean_true)),
            }
        )

    hard_mask = cohorts["hard_attack"]
    hard_graph = frame.loc[hard_mask, "graph_probability"].to_numpy(dtype=np.float64)
    hard_true = np.asarray(rank_info["mean_true_probability"])[hard_mask]
    for operating_point, threshold in (
        ("reference_0_50", model.reference_graph_threshold),
        ("validation_selected", model.selected_graph_threshold),
    ):
        low_graph = hard_graph < threshold
        high_attacker = hard_true >= model.node_threshold
        quadrant_counts = {
            "low_graph_high_attacker": int(np.sum(low_graph & high_attacker)),
            "high_graph_high_attacker": int(np.sum(~low_graph & high_attacker)),
            "low_graph_low_attacker": int(np.sum(low_graph & ~high_attacker)),
            "high_graph_low_attacker": int(np.sum(~low_graph & ~high_attacker)),
        }
        for quadrant, count in quadrant_counts.items():
            rows.append(
                {
                    "model_name": model.name,
                    "cohort": f"hard_attack_quadrant:{operating_point}:{quadrant}",
                    "window_count": count,
                    "graph_vs_max_node_pearson": float("nan"),
                    "graph_vs_mean_true_attacker_pearson": float("nan"),
                    "mean_graph_probability": float("nan"),
                    "mean_maximum_node_probability": float("nan"),
                    "mean_true_attacker_probability": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def diagnosis_table(
    model_exports: Sequence[ModelExport],
    hard_details: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    graph_recalls: list[float] = []
    node_top1_rates: list[float] = []

    for model in model_exports:
        sub = hard_details[hard_details["model_name"] == model.name]
        graph_pred = sub["pred_graph_selected"].to_numpy(dtype=np.int8)
        false_negative = graph_pred == 0
        if false_negative.sum() == 0:
            rank1_given_fn = float("nan")
            top3_given_fn = float("nan")
            informative_given_fn = float("nan")
            exact_given_fn = float("nan")
            empty_given_fn = float("nan")
            wrong_source_given_fn = float("nan")
            overpredict_given_fn = float("nan")
            onehop_given_fn = float("nan")
            median_true_probability_fn = float("nan")
        else:
            fn = sub.loc[false_negative]
            rank1_given_fn = float((fn["router12_rank"] == 1).mean())
            top3_given_fn = float((fn["router12_rank"] <= 3).mean())
            informative_given_fn = float(fn["node_informative_top3_or_p50"].mean())
            exact_given_fn = float((fn["node_category_at_0_50"] == "exact").mean())
            empty_given_fn = float((fn["node_category_at_0_50"] == "empty").mean())
            wrong_source_given_fn = float(
                (fn["node_category_at_0_50"] == "wrong_source").mean()
            )
            overpredict_given_fn = float(
                (fn["node_category_at_0_50"] == "overpredicting").mean()
            )
            onehop_given_fn = float((fn["spread_category"] == "one_hop_spread").mean())
            median_true_probability_fn = float(fn["router12_probability"].median())

        flags: list[str] = []
        if false_negative.sum() > 0 and rank1_given_fn >= 0.70 and informative_given_fn >= 0.70:
            flags.append("A_graph_readout_failure")
        if (
            false_negative.sum() > 0
            and rank1_given_fn >= 0.70
            and exact_given_fn < 0.30
            and 0.30 <= median_true_probability_fn < model.node_threshold
        ):
            flags.append("B_node_threshold_calibration")
        if false_negative.sum() > 0 and (
            onehop_given_fn >= 0.40 or overpredict_given_fn >= 0.40
        ):
            flags.append("C_spatial_smearing")
        if false_negative.sum() > 0 and (
            informative_given_fn <= 0.30
            and (empty_given_fn + wrong_source_given_fn) >= 0.50
        ):
            flags.append("D_shared_encoder_training")
        if not flags:
            flags.append("F_mixed_evidence")

        primary = flags[0]
        graph_recall = float(graph_pred.mean())
        node_top1 = float((sub["router12_rank"] == 1).mean())
        graph_recalls.append(graph_recall)
        node_top1_rates.append(node_top1)

        rows.append(
            {
                "model_name": model.name,
                "selected_graph_threshold": model.selected_graph_threshold,
                "hard_attack_graph_recall": graph_recall,
                "hard_attack_false_negative_count": int(false_negative.sum()),
                "router12_top1_rate_all_windows": node_top1,
                "router12_rank1_given_graph_false_negative": rank1_given_fn,
                "router12_top3_given_graph_false_negative": top3_given_fn,
                "node_informative_given_graph_false_negative": informative_given_fn,
                "node_exact_given_graph_false_negative": exact_given_fn,
                "node_empty_given_graph_false_negative": empty_given_fn,
                "node_wrong_source_given_graph_false_negative": wrong_source_given_fn,
                "node_overpredicting_given_graph_false_negative": overpredict_given_fn,
                "one_hop_spread_given_graph_false_negative": onehop_given_fn,
                "median_router12_probability_given_graph_false_negative": median_true_probability_fn,
                "diagnostic_flags": ";".join(flags),
                "primary_diagnosis": primary,
            }
        )

    cross_model_correlation = safe_pearson(np.asarray(graph_recalls), np.asarray(node_top1_rates))
    if np.isfinite(cross_model_correlation) and cross_model_correlation <= -0.50:
        for row in rows:
            row["diagnostic_flags"] += ";E_multitask_inconsistency_cross_model"
            row["cross_model_graph_recall_vs_node_top1_correlation"] = cross_model_correlation
    else:
        for row in rows:
            row["cross_model_graph_recall_vs_node_top1_correlation"] = cross_model_correlation

    return pd.DataFrame(rows)




def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without requiring the optional tabulate package."""
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    display = display.replace({np.nan: ""})
    columns = [str(column) for column in display.columns]

    def escape(value: Any) -> str:
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(escape(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(escape(value) for value in row) + " |")
    return "\n".join(lines)


def markdown_report(
    verdict: str,
    hard_failures: Sequence[str],
    model_thresholds: pd.DataFrame,
    diagnoses: pd.DataFrame,
    hard_rank: pd.DataFrame,
    hard_blocks: pd.DataFrame,
    controls: pd.DataFrame,
) -> str:
    lines = [
        "# Stage 7E — Graph-Head versus Node-Head Interaction",
        "",
        f"Generated: `{utc_now()}`",
        f"Script version: `{SCRIPT_VERSION}`",
        "",
        f"## Verdict: **{verdict}**",
        "",
    ]
    if hard_failures:
        lines += ["### Hard failures", ""]
        lines += [f"- {item}" for item in hard_failures]
        lines.append("")
        return "\n".join(lines)

    lines += [
        "The analysis used saved validation/test prediction exports only. "
        "No model inference, training, dataset modification, or gem5 execution was performed.",
        "",
        "## Operating points",
        "",
        dataframe_to_markdown(model_thresholds),
        "",
        "## Hard-attack attacker-rank summary",
        "",
        dataframe_to_markdown(hard_rank),
        "",
        "## Cross-model diagnosis",
        "",
        dataframe_to_markdown(diagnoses),
        "",
        "## Temporal blocks",
        "",
        dataframe_to_markdown(hard_blocks),
        "",
        "## Control availability",
        "",
    ]
    control_view = controls[
        ["control_role", "run_id", "present_in_export", "missing_reason"]
    ].drop_duplicates()
    lines += [
        dataframe_to_markdown(control_view),
        "",
        "## Interpretation rules",
        "",
        "- **A — Graph-readout failure:** graph false negatives retain strong rank-1/top-3 attacker evidence.",
        "- **B — Node calibration:** the attacker ranks highly but remains just below the fixed node threshold.",
        "- **C — Spatial smearing:** attacker evidence spreads strongly into one-hop neighbours or extra routers.",
        "- **D — Shared encoder/training:** graph false negatives also have weak or absent true-attacker evidence.",
        "- **E — Multitask inconsistency:** graph and node quality move in opposing directions across models.",
        "- **F — Mixed evidence:** no single mechanism meets the declared diagnostic rule.",
        "",
        "These are diagnostic classifications, not causal proofs. Stage 9 interventions should be selected "
        "from the observed pattern and then tested in controlled retraining experiments.",
        "",
    ]
    return "\n".join(lines)


def make_smoke_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    smoke = root / "_smoke_inputs"
    if smoke.exists():
        shutil.rmtree(smoke)
    predictions = smoke / "predictions"
    dataset = smoke / "dataset"
    tables = smoke / "tables"
    predictions.mkdir(parents=True)
    dataset.mkdir()
    tables.mkdir()

    runs = [
        ("N-idle-Pidle-R1-V3", "val", 0, (), "idle", "idle", "", "1"),
        (HARD_NORMAL, "test", 0, (), "mixed", "3-7-8-12", "", "18"),
        ("N-5-10-Pbursty-R16-V3", "test", 0, (), "bursty", "5-10", "", "16"),
        (HARD_ATTACK, "test", 1, (12,), "bursty", "5-10", "20", "51"),
        ("N-2-13-Pstream-R50-A-12-S20-V3", "test", 1, (12,), "stream", "2-13", "20", "50"),
    ]
    windows_per_run = 9
    run_ids: list[str] = []
    splits: list[str] = []
    end_epochs: list[int] = []
    y_graph: list[float] = []
    y_node: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []

    for run_id, split, graph, attackers, profile, active, strength, seed in runs:
        for offset in range(windows_per_run):
            dataset_index = len(run_ids)
            end_epoch = 7 + offset
            mask = np.zeros(16, dtype=np.float32)
            for attacker in attackers:
                mask[attacker] = 1.0
            run_ids.append(run_id)
            splits.append(split)
            end_epochs.append(end_epoch)
            y_graph.append(float(graph))
            y_node.append(mask)
            metadata_rows.append(
                {
                    "sample_order_within_export": dataset_index,
                    "sample_index": dataset_index,
                    "sample_id": f"{run_id}:{end_epoch}",
                    "run_id": run_id,
                    "split": split,
                    "end_epoch": end_epoch,
                    "true_graph": graph,
                    "true_attacker_count": len(attackers),
                    "profile": profile,
                    "active_cores": active,
                    "attackers": "-".join(map(str, attackers)),
                    "strength": strength,
                    "seed": seed,
                }
            )

    run_array = np.asarray(run_ids)
    split_array = np.asarray(splits)
    end_array = np.asarray(end_epochs, dtype=np.int32)
    graph_array = np.asarray(y_graph, dtype=np.float32)
    node_array = np.stack(y_node).astype(np.float32)

    # A small connected 4x4 graph with self loops and grid links.
    edges: list[tuple[int, int]] = []
    for node in range(16):
        edges.append((node, node))
        row, col = divmod(node, 4)
        for other in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            rr, cc = other
            if 0 <= rr < 4 and 0 <= cc < 4:
                edges.append((node, rr * 4 + cc))
    edge_index = np.asarray(edges, dtype=np.int64).T

    np.save(dataset / "run_id.npy", run_array)
    np.save(dataset / "split.npy", split_array)
    np.save(dataset / "end_epoch.npy", end_array)
    np.save(dataset / "y_graph.npy", graph_array)
    np.save(dataset / "y_node.npy", node_array)
    np.save(dataset / "edge_index.npy", edge_index)
    pd.DataFrame(metadata_rows).to_csv(predictions / "prediction_metadata.csv", index=False)

    rng = np.random.default_rng(7)
    model_names = (
        "conv1d_gcn",
        "tcn_attention_gcn",
        "tcn_maxpool_gcn",
        "tcn_meanpool_gcn",
    )
    for model_index, model_name in enumerate(model_names):
        n = len(metadata_rows)
        graph_probability = np.where(
            graph_array == 1,
            0.35 + 0.08 * model_index + rng.normal(0, 0.06, n),
            0.20 + rng.normal(0, 0.05, n),
        )
        graph_probability = np.clip(graph_probability, 0.001, 0.999).astype(np.float32)
        graph_logit = np.log(graph_probability / (1.0 - graph_probability)).astype(np.float32)

        node_probability = np.clip(rng.normal(0.08, 0.03, (n, 16)), 0.001, 0.999)
        attack_rows = np.flatnonzero(graph_array == 1)
        node_probability[attack_rows, 12] = np.clip(
            0.70 - 0.08 * model_index + rng.normal(0, 0.05, len(attack_rows)),
            0.001,
            0.999,
        )
        node_probability = node_probability.astype(np.float32)
        node_logits = np.log(node_probability / (1.0 - node_probability)).astype(np.float32)
        pred_node = node_probability >= 0.5

        np.savez_compressed(
            predictions / f"{model_name}_predictions.npz",
            model_name=np.asarray([model_name]),
            sample_order_within_export=np.arange(n, dtype=np.int64),
            sample_index=np.arange(n, dtype=np.int64),
            sample_id=np.asarray([row["sample_id"] for row in metadata_rows]),
            run_id=run_array,
            split=split_array,
            end_epoch=end_array,
            true_graph=graph_array,
            true_attacker_mask=node_array,
            true_attacker_count=node_array.sum(axis=1).astype(np.int32),
            graph_logit=graph_logit,
            graph_probability=graph_probability,
            pred_graph_at_0_50=(graph_probability >= 0.5).astype(np.int8),
            graph_threshold=np.asarray([0.5], dtype=np.float32),
            node_logits=node_logits,
            node_probabilities=node_probability,
            pred_attacker_mask_at_0_50=pred_node.astype(np.int8),
            node_threshold_0_50=np.asarray([0.5], dtype=np.float32),
            pred_attacker_count_at_0_50=pred_node.sum(axis=1).astype(np.int32),
            exact_localization_at_0_50=np.all(pred_node == node_array.astype(bool), axis=1).astype(np.int8),
        )

    pd.DataFrame(
        {
            "model_name": list(model_names),
            "selected_threshold": [0.47, 0.48, 0.46, 0.49],
        }
    ).to_csv(tables / "selected_graph_thresholds.csv", index=False)

    return predictions, predictions / "prediction_metadata.csv", dataset, tables / "selected_graph_thresholds.csv"


def run_analysis(paths: Paths, overwrite: bool, smoke_test: bool) -> int:
    ensure_output_is_safe(paths.output_root, overwrite)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    (paths.output_root / "logs").mkdir(exist_ok=True)
    (paths.output_root / "tables").mkdir(exist_ok=True)

    selected_thresholds = read_selected_thresholds(paths.selected_thresholds_csv)
    dataset = load_dataset_arrays(paths.dataset_root)
    metadata = pd.read_csv(
        paths.metadata_csv,
        keep_default_na=True,
        dtype={
            "sample_id": str,
            "run_id": str,
            "split": str,
            "profile": str,
            "active_cores": str,
            "seed": str,
        },
    )

    model_paths = [paths.predictions_dir / name for name in MODEL_FILES]
    model_exports = [load_model_export(path, selected_thresholds) for path in model_paths]

    inventory = file_inventory(paths, model_exports)
    array_manifest = prediction_array_manifest(model_exports)
    alignment = validate_alignment(model_exports, metadata, dataset, smoke_test=smoke_test)
    hard_alignment_failures = alignment[
        (alignment["severity"] == "hard") & (~alignment["passed"])
    ]
    hard_failures = [
        f"{row.model_name}: {row.check} ({row.detail})"
        for row in hard_alignment_failures.itertuples(index=False)
    ]

    thresholds = model_threshold_table(model_exports)

    if hard_failures:
        verdict = "STOP — ALIGNMENT FAILURE"
        empty = pd.DataFrame()
        outputs = {
            "inventory": inventory,
            "array_manifest": array_manifest,
            "alignment": alignment,
            "thresholds": thresholds,
        }
        for name, frame in outputs.items():
            target = {
                "inventory": "stage7e_input_inventory.csv",
                "array_manifest": "stage7e_prediction_array_manifest.csv",
                "alignment": "stage7e_alignment_checks.csv",
                "thresholds": "stage7e_model_thresholds.csv",
            }[name]
            atomic_write_csv(paths.output_root / "tables" / target, frame)
        validation_text = "\n".join(
            f"{row.model_name} | {row.check}: {row.passed} | {row.detail}"
            for row in alignment.itertuples(index=False)
        ) + "\n"
        atomic_write_text(paths.output_root / "logs/48_stage7e_validation.txt", validation_text)
        summary = {
            "generated_at": utc_now(),
            "script_version": SCRIPT_VERSION,
            "verdict": verdict,
            "hard_failures": hard_failures,
            "checks_passed": int(alignment["passed"].sum()),
            "checks_total": int(len(alignment)),
        }
        atomic_write_json(paths.output_root / "tables/stage7e_summary.json", summary)
        atomic_write_text(
            paths.output_root / "STAGE7E_GRAPH_NODE_INTERACTION_REPORT.md",
            markdown_report(verdict, hard_failures, thresholds, empty, empty, empty, empty),
        )
        print(verdict)
        return 2

    run_summary_rows: list[dict[str, Any]] = []
    category_frames: list[pd.DataFrame] = []
    hard_detail_frames: list[pd.DataFrame] = []
    hard_block_frames: list[pd.DataFrame] = []
    rank_frames: list[pd.DataFrame] = []
    probability_frames: list[pd.DataFrame] = []
    spread_frames: list[pd.DataFrame] = []

    edge_index = np.asarray(dataset["edge_index"])
    for model in model_exports:
        frame = model_frame(model, metadata)
        true_node = np.asarray(model.arrays["true_attacker_mask"], dtype=np.float64)
        node_probability = np.asarray(model.arrays["node_probabilities"], dtype=np.float64)
        categories = node_categories(true_node, node_probability, model.node_threshold)
        rank_info = rank_descending(node_probability, true_node)

        run_summary_rows.extend(
            summarize_run_operating_point(
                frame,
                node_probability,
                true_node,
                categories,
                rank_info,
                "reference_0_50",
                model.reference_graph_threshold,
            )
        )
        run_summary_rows.extend(
            summarize_run_operating_point(
                frame,
                node_probability,
                true_node,
                categories,
                rank_info,
                "validation_selected",
                model.selected_graph_threshold,
            )
        )
        category_frames.append(
            graph_node_category_table(frame, categories, rank_info, model)
        )
        details, blocks, ranks, spread = hard_attack_analysis(
            model,
            frame,
            true_node,
            node_probability,
            categories,
            rank_info,
            edge_index,
            HARD_ATTACK,
        )
        hard_detail_frames.append(details)
        hard_block_frames.append(blocks)
        rank_frames.append(ranks)
        spread_frames.append(spread)
        probability_frames.append(
            probability_summary(model, frame, true_node, node_probability, HARD_ATTACK)
        )

    run_summary = pd.DataFrame(run_summary_rows)
    graph_node_categories = pd.concat(category_frames, ignore_index=True)
    hard_details = pd.concat(hard_detail_frames, ignore_index=True)
    hard_blocks = pd.concat(hard_block_frames, ignore_index=True)
    attacker_rank = pd.concat(rank_frames, ignore_index=True)
    probability_summary_frame = pd.concat(probability_frames, ignore_index=True)
    neighbour_spread = pd.concat(spread_frames, ignore_index=True)
    controls = control_comparison(run_summary, metadata)
    diagnoses = diagnosis_table(model_exports, hard_details)

    primary_counts = Counter(diagnoses["primary_diagnosis"].tolist())
    dominant_primary, dominant_count = primary_counts.most_common(1)[0]
    verdict = (
        "PASS — DOMINANT FAILURE MECHANISM IDENTIFIED"
        if dominant_primary != "F_mixed_evidence" and dominant_count >= 3
        else "PASS WITH MIXED EVIDENCE"
    )

    atomic_write_csv(paths.output_root / "tables/stage7e_input_inventory.csv", inventory)
    atomic_write_csv(
        paths.output_root / "tables/stage7e_prediction_array_manifest.csv",
        array_manifest,
    )
    atomic_write_csv(paths.output_root / "tables/stage7e_alignment_checks.csv", alignment)
    atomic_write_csv(paths.output_root / "tables/stage7e_model_thresholds.csv", thresholds)
    atomic_write_csv(paths.output_root / "tables/stage7e_run_model_summary.csv", run_summary)
    atomic_write_csv(
        paths.output_root / "tables/stage7e_graph_node_categories.csv",
        graph_node_categories,
    )
    atomic_write_csv(
        paths.output_root / "tables/stage7e_hard_attack_window_details.csv",
        hard_details,
    )
    atomic_write_csv(
        paths.output_root / "tables/stage7e_hard_attack_temporal_blocks.csv",
        hard_blocks,
    )
    atomic_write_csv(
        paths.output_root / "tables/stage7e_control_run_comparison.csv",
        controls,
    )
    atomic_write_csv(
        paths.output_root / "tables/stage7e_attacker_rank_summary.csv",
        attacker_rank,
    )
    atomic_write_csv(
        paths.output_root / "tables/stage7e_graph_node_probability_summary.csv",
        probability_summary_frame,
    )
    atomic_write_csv(
        paths.output_root / "tables/stage7e_neighbour_spread_summary.csv",
        neighbour_spread,
    )
    atomic_write_csv(
        paths.output_root / "tables/stage7e_cross_model_diagnosis.csv",
        diagnoses,
    )

    validation_text = "\n".join(
        f"{row.model_name} | {row.check}: {row.passed} | {row.detail}"
        for row in alignment.itertuples(index=False)
    ) + "\n"
    atomic_write_text(paths.output_root / "logs/48_stage7e_validation.txt", validation_text)

    summary = {
        "generated_at": utc_now(),
        "script_version": SCRIPT_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "verdict": verdict,
        "hard_failures": [],
        "checks_passed": int(alignment["passed"].sum()),
        "checks_total": int(len(alignment)),
        "model_count": len(model_exports),
        "models": [model.name for model in model_exports],
        "exported_sample_count": int(len(metadata)),
        "hard_attack": HARD_ATTACK,
        "hard_normal": HARD_NORMAL,
        "dominant_primary_diagnosis": dominant_primary,
        "dominant_primary_diagnosis_model_count": int(dominant_count),
        "primary_diagnosis_counts": dict(primary_counts),
        "no_model_loading": True,
        "no_inference": True,
        "no_training": True,
        "no_gem5": True,
        "selected_graph_thresholds_from_validation_table": True,
        "node_threshold_test_tuning_performed": False,
    }
    atomic_write_json(paths.output_root / "tables/stage7e_summary.json", summary)
    atomic_write_text(
        paths.output_root / "STAGE7E_GRAPH_NODE_INTERACTION_REPORT.md",
        markdown_report(
            verdict,
            [],
            thresholds,
            diagnoses,
            attacker_rank,
            hard_blocks,
            controls,
        ),
    )

    print(f"Stage 7E verdict: {verdict}")
    print(f"Checks passed: {int(alignment['passed'].sum())}/{len(alignment)}")
    print(f"Dominant primary diagnosis: {dominant_primary} ({dominant_count}/{len(model_exports)} models)")
    print(f"Output root: {paths.output_root}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse saved graph and node prediction exports for Stage 7E-lite. "
            "No model inference or training is performed."
        )
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("reports/v3_failure_analysis/predictions"),
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("reports/v3_failure_analysis/predictions/prediction_metadata.csv"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.home()
        / "tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3",
    )
    parser.add_argument(
        "--selected-thresholds",
        type=Path,
        default=Path("reports/v3_failure_analysis/tables/selected_graph_thresholds.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "reports/v3_failure_analysis/stage7e_graph_node_interaction"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing Stage 7E report/table outputs.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Generate small synthetic inputs and run the full analysis pipeline.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()

    if args.smoke_test:
        predictions, metadata, dataset, thresholds = make_smoke_inputs(output_root)
        paths = Paths(
            predictions_dir=predictions,
            metadata_csv=metadata,
            dataset_root=dataset,
            selected_thresholds_csv=thresholds,
            output_root=output_root,
        )
    else:
        paths = Paths(
            predictions_dir=args.predictions_dir.expanduser().resolve(),
            metadata_csv=args.metadata_csv.expanduser().resolve(),
            dataset_root=args.dataset_root.expanduser().resolve(),
            selected_thresholds_csv=args.selected_thresholds.expanduser().resolve(),
            output_root=output_root,
        )

    try:
        return run_analysis(paths, overwrite=args.overwrite, smoke_test=args.smoke_test)
    except Stage7EError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # defensive top-level boundary
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
