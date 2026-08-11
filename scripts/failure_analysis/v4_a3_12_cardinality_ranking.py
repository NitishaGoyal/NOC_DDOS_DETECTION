#!/usr/bin/env python3
"""
V4-A3.12
Cardinality and Router-Ranking Decomposition

Descriptive analysis of the frozen A3 model on:
  1. single-window A3 outputs
  2. the frozen H32 graph-gated candidate-localization policy

No model training occurs.
No threshold, horizon, gate, decoder, or persistence parameter is selected on
the development-test split.

Primary questions
-----------------
* How much error comes from predicting the wrong attacker count?
* How much error remains even when the true attacker count is supplied?
* Are true attackers already ranked near the top?
* Which true-count, profile, attack-kind, and topology groups remain weak?
* Which windows are rescued or regressed by temporal aggregation?

Outputs
-------
summary.json
split_overview.csv
count_confusion.csv
count_group_metrics.csv
profile_group_metrics.csv
attack_kind_group_metrics.csv
topology_group_metrics.csv
rank_distribution.csv
error_category_counts.csv
paired_exact_overlap.csv
per_sample_decomposition.npz
A3_12_LOCK.json
V4_A3_12_CARDINALITY_RANKING_PASS
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("a3_11_validation_search", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def invert_code_map(mapping: Any) -> dict[int, str]:
    if not isinstance(mapping, dict):
        return {}
    out: dict[int, str] = {}
    for key, value in mapping.items():
        try:
            out[int(value)] = str(key)
        except (TypeError, ValueError):
            continue
    return out


def build_node_classes(edge_index: np.ndarray, num_nodes: int) -> dict[int, str]:
    edge = np.asarray(edge_index)
    if edge.shape[0] != 2:
        raise ValueError(f"edge_index expected [2,E], got {edge.shape}")
    neighbors = {i: set() for i in range(num_nodes)}
    for src, dst in zip(edge[0], edge[1]):
        s = int(src)
        d = int(dst)
        if 0 <= s < num_nodes and 0 <= d < num_nodes and s != d:
            neighbors[s].add(d)
            neighbors[d].add(s)

    degrees = {node: len(values) for node, values in neighbors.items()}
    unique = sorted(set(degrees.values()))
    if len(unique) < 2:
        return {node: f"degree_{degree}" for node, degree in degrees.items()}

    minimum = min(unique)
    maximum = max(unique)
    classes: dict[int, str] = {}
    for node, degree in degrees.items():
        if degree == minimum:
            classes[node] = "corner"
        elif degree == maximum:
            classes[node] = "interior"
        else:
            classes[node] = "edge"
    return classes


def sample_topology_label(
    y_node_row: np.ndarray,
    node_classes: dict[int, str],
) -> str:
    attackers = np.flatnonzero(np.asarray(y_node_row) > 0.5)
    if attackers.size == 0:
        return "normal"
    labels = sorted({node_classes[int(node)] for node in attackers})
    if len(labels) == 1:
        return f"all_{labels[0]}"
    return "mixed_" + "_".join(labels)


def topk_predictions(scores: np.ndarray, counts: np.ndarray) -> np.ndarray:
    score = np.asarray(scores)
    k_values = np.asarray(counts, dtype=np.int64)
    pred = np.zeros_like(score, dtype=np.int8)
    order = np.argsort(-score, axis=1, kind="stable")
    for i, k in enumerate(k_values):
        k = int(np.clip(k, 0, score.shape[1]))
        if k:
            pred[i, order[i, :k]] = 1
    return pred


def classify_set_error(true_set: np.ndarray, pred_set: np.ndarray) -> str:
    true_indices = set(np.flatnonzero(true_set > 0.5).tolist())
    pred_indices = set(np.flatnonzero(pred_set > 0.5).tolist())

    if pred_indices == true_indices:
        return "exact"
    if not pred_indices:
        return "empty"
    missed = true_indices - pred_indices
    extra = pred_indices - true_indices
    if missed and not extra:
        return "misses_only"
    if extra and not missed:
        return "extras_only"
    if missed and extra and len(true_indices) == len(pred_indices):
        return "substitution_equal_count"
    if missed and extra:
        return "mixed_miss_extra"
    return "other"


def binary_set_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.int8)
    pred = np.asarray(y_pred, dtype=np.int8)

    tp = int(np.sum((true == 1) & (pred == 1)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = np.all(true == pred, axis=1)
    empty = np.sum(pred, axis=1) == 0
    return {
        "sample_count": int(true.shape[0]),
        "node_tp": tp,
        "node_fp": fp,
        "node_fn": fn,
        "node_precision": precision,
        "node_recall": recall,
        "node_f1": f1,
        "exact_localization": float(np.mean(exact)) if exact.size else math.nan,
        "empty_prediction_rate": float(np.mean(empty)) if empty.size else math.nan,
        "mean_predicted_count": float(np.mean(np.sum(pred, axis=1))),
        "mean_true_count": float(np.mean(np.sum(true, axis=1))),
    }


def true_attacker_rank_arrays(
    scores: np.ndarray,
    y_node: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = np.asarray(scores)
    true = np.asarray(y_node) > 0.5
    ranks = np.argsort(np.argsort(-score, axis=1, kind="stable"), axis=1) + 1

    flat_true_ranks = ranks[true].astype(np.int16)
    worst_rank = np.zeros(score.shape[0], dtype=np.int16)
    best_rank = np.zeros(score.shape[0], dtype=np.int16)

    for i in range(score.shape[0]):
        values = ranks[i, true[i]]
        if values.size:
            worst_rank[i] = int(np.max(values))
            best_rank[i] = int(np.min(values))

    return flat_true_ranks, best_rank, worst_rank


def rank_summary(
    split: str,
    representation: str,
    scores: np.ndarray,
    y_node: np.ndarray,
    true_count: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    score = np.asarray(scores)
    true = np.asarray(y_node)
    count = np.asarray(true_count, dtype=np.int64)
    all_counts = sorted(int(x) for x in np.unique(count) if int(x) > 0)

    for group_count in [None, *all_counts]:
        if group_count is None:
            mask = count > 0
            label = "all_attacks"
        else:
            mask = count == group_count
            label = str(group_count)

        if not np.any(mask):
            continue

        flat, best, worst = true_attacker_rank_arrays(score[mask], true[mask])
        row = {
            "split": split,
            "representation": representation,
            "true_count_group": label,
            "sample_count": int(np.sum(mask)),
            "true_attacker_instance_count": int(flat.size),
            "rank_mean": float(np.mean(flat)),
            "rank_median": float(np.median(flat)),
            "rank_p90": float(np.quantile(flat, 0.90)),
            "rank_p95": float(np.quantile(flat, 0.95)),
            "rank_top1_rate": float(np.mean(flat <= 1)),
            "rank_top2_rate": float(np.mean(flat <= 2)),
            "rank_top4_rate": float(np.mean(flat <= 4)),
            "worst_true_rank_mean": float(np.mean(worst)),
            "worst_true_rank_median": float(np.median(worst)),
            "all_true_within_top_true_count_rate": float(
                np.mean(worst <= count[mask])
            ),
            "at_least_one_true_rank1_rate": float(np.mean(best == 1)),
        }
        rows.append(row)

    return rows


def aggregate_temporal_candidate(
    module,
    prepared_data,
    count_prob: np.ndarray,
    policy: dict[str, Any],
    deoverlap_stride: int,
) -> dict[str, np.ndarray]:
    dense_segments = module.make_segments(prepared_data)
    segments = module.mode_segments(
        dense_segments,
        "deoverlap",
        deoverlap_stride,
    )
    horizon = int(policy["horizon"])

    graph_logit = module.logit(prepared_data.graph_prob)
    node_logit = module.logit(prepared_data.node_prob)
    count_logprob = np.log(np.clip(count_prob, 1e-8, 1.0))

    endpoints, graph_agg = module.aggregate_array(
        graph_logit,
        segments,
        horizon,
        module.uniform_weights(horizon),
    )
    node_endpoints, node_agg = module.aggregate_array(
        node_logit,
        segments,
        horizon,
        module.uniform_weights(horizon),
    )
    count_endpoints, count_agg = module.aggregate_array(
        count_logprob,
        segments,
        horizon,
        module.uniform_weights(horizon),
    )
    if not (
        np.array_equal(endpoints, node_endpoints)
        and np.array_equal(endpoints, count_endpoints)
    ):
        raise RuntimeError("temporal aggregation endpoint mismatch")

    graph_score = module.sigmoid(graph_agg)
    node_score = module.sigmoid(node_agg)
    count_score = np.exp(count_agg - np.max(count_agg, axis=1, keepdims=True))
    count_score /= np.sum(count_score, axis=1, keepdims=True)

    graph_gate = graph_score >= float(policy["graph_threshold"])
    threshold_pred = (
        node_score >= float(policy["node_threshold"])
    ).astype(np.int8)
    threshold_pred[~graph_gate] = 0

    predicted_count = np.argmax(count_score, axis=1).astype(np.int64)
    count_topk_pred = topk_predictions(node_score, predicted_count)
    count_topk_pred[~graph_gate] = 0

    true_count = prepared_data.attacker_count[endpoints].astype(np.int64)
    oracle_pred = topk_predictions(node_score, true_count)

    return {
        "endpoints": endpoints,
        "graph_score": graph_score,
        "node_score": node_score,
        "count_score": count_score,
        "graph_gate": graph_gate,
        "threshold_pred": threshold_pred,
        "predicted_count": predicted_count,
        "count_topk_pred": count_topk_pred,
        "oracle_pred": oracle_pred,
    }


def decoder_bundle(
    node_score: np.ndarray,
    count_score: np.ndarray,
    y_node: np.ndarray,
    true_count: np.ndarray,
    node_threshold: float,
    graph_gate: np.ndarray | None = None,
) -> dict[str, Any]:
    score = np.asarray(node_score)
    count_prob = np.asarray(count_score)
    true = np.asarray(y_node)
    true_k = np.asarray(true_count, dtype=np.int64)

    predicted_count = np.argmax(count_prob, axis=1).astype(np.int64)
    count_topk = topk_predictions(score, predicted_count)
    threshold = (score >= float(node_threshold)).astype(np.int8)
    oracle = topk_predictions(score, true_k)

    if graph_gate is not None:
        gate = np.asarray(graph_gate, dtype=bool)
        count_topk[~gate] = 0
        threshold[~gate] = 0

    return {
        "predicted_count": predicted_count,
        "count_topk": count_topk,
        "threshold": threshold,
        "oracle": oracle,
    }


def group_metrics_rows(
    split: str,
    representation: str,
    decoder_name: str,
    group_name: str,
    group_values: np.ndarray,
    group_labels: dict[Any, str],
    y_node: np.ndarray,
    y_pred: np.ndarray,
    true_count: np.ndarray,
    predicted_count: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = np.asarray(group_values)

    for value in sorted(np.unique(groups).tolist(), key=lambda x: str(x)):
        mask = groups == value
        if not np.any(mask):
            continue
        metrics = binary_set_metrics(y_node[mask], y_pred[mask])
        errors = Counter(
            classify_set_error(t, p)
            for t, p in zip(y_node[mask], y_pred[mask])
        )
        row = {
            "split": split,
            "representation": representation,
            "decoder": decoder_name,
            group_name: value,
            f"{group_name}_label": group_labels.get(value, str(value)),
            **metrics,
            "count_accuracy": float(
                np.mean(predicted_count[mask] == true_count[mask])
            ),
            "count_mae": float(
                np.mean(np.abs(predicted_count[mask] - true_count[mask]))
            ),
        }
        for category in [
            "exact",
            "empty",
            "misses_only",
            "extras_only",
            "substitution_equal_count",
            "mixed_miss_extra",
            "other",
        ]:
            row[f"error_{category}_count"] = int(errors.get(category, 0))
            row[f"error_{category}_rate"] = (
                errors.get(category, 0) / int(np.sum(mask))
            )
        rows.append(row)

    return rows


def evaluate_split(
    split: str,
    archive_path: Path,
    data_dir: Path,
    module,
    metadata: dict[str, Any],
    node_classes: dict[int, str],
    candidate_policy: dict[str, Any],
    base_node_threshold: float,
    deoverlap_stride: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    with np.load(archive_path, allow_pickle=True) as archive:
        graph_prob = np.asarray(archive["graph_prob"])
        node_prob = np.asarray(archive["node_prob"])
        count_prob = np.asarray(archive["count_prob"])
        y_graph = np.asarray(archive["y_graph"])
        y_node = np.asarray(archive["y_node"])
        true_count = np.asarray(archive["attacker_count"], dtype=np.int64)
        global_index = np.asarray(archive["global_index"], dtype=np.int64)
        run_index = np.asarray(archive["run_index"], dtype=np.int64)
        profile_id = np.asarray(archive["profile_id"], dtype=np.int64)
        attack_kind_id = np.asarray(archive["attack_kind_id"], dtype=np.int64)

    prepared = module.prepare_data(archive_path, data_dir)

    base = decoder_bundle(
        node_prob,
        count_prob,
        y_node,
        true_count,
        base_node_threshold,
        graph_gate=None,
    )
    temporal = aggregate_temporal_candidate(
        module,
        prepared,
        count_prob,
        candidate_policy,
        deoverlap_stride,
    )

    ep = temporal["endpoints"]
    temporal_y_node = y_node[ep]
    temporal_true_count = true_count[ep]
    temporal_profile = profile_id[ep]
    temporal_attack_kind = attack_kind_id[ep]
    temporal_global_index = global_index[ep]
    temporal_run_index = run_index[ep]

    topology_all = np.asarray(
        [sample_topology_label(row, node_classes) for row in y_node],
        dtype=object,
    )
    temporal_topology = topology_all[ep]

    base_topology = topology_all

    profile_map = invert_code_map(metadata.get("code_maps", {}).get("profile"))
    attack_map = invert_code_map(metadata.get("code_maps", {}).get("attack_kind"))

    base_bundle = {
        "CountTopK": base["count_topk"],
        "NodeThreshold": base["threshold"],
        "OracleTrueCountTopK": base["oracle"],
    }
    temporal_bundle = {
        "CountTopK": temporal["count_topk_pred"],
        "NodeThreshold": temporal["threshold_pred"],
        "OracleTrueCountTopK": temporal["oracle_pred"],
    }

    overview_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    attack_kind_rows: list[dict[str, Any]] = []
    topology_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    representation_payloads = {
        "single_window": {
            "scores": node_prob,
            "y_node": y_node,
            "true_count": true_count,
            "predicted_count": base["predicted_count"],
            "bundle": base_bundle,
            "profile": profile_id,
            "attack_kind": attack_kind_id,
            "topology": base_topology,
        },
        "temporal_h32_candidate": {
            "scores": temporal["node_score"],
            "y_node": temporal_y_node,
            "true_count": temporal_true_count,
            "predicted_count": temporal["predicted_count"],
            "bundle": temporal_bundle,
            "profile": temporal_profile,
            "attack_kind": temporal_attack_kind,
            "topology": temporal_topology,
        },
    }

    for representation, payload in representation_payloads.items():
        scores = payload["scores"]
        current_y = payload["y_node"]
        current_true_count = payload["true_count"]
        current_pred_count = payload["predicted_count"]

        rank_rows.extend(
            rank_summary(
                split,
                representation,
                scores,
                current_y,
                current_true_count,
            )
        )

        for true_k in sorted(np.unique(current_true_count).tolist()):
            for pred_k in sorted(np.unique(current_pred_count).tolist()):
                mask = (
                    (current_true_count == true_k)
                    & (current_pred_count == pred_k)
                )
                if np.any(mask):
                    confusion_rows.append(
                        {
                            "split": split,
                            "representation": representation,
                            "true_count": int(true_k),
                            "predicted_count": int(pred_k),
                            "sample_count": int(np.sum(mask)),
                        }
                    )

        for decoder_name, pred in payload["bundle"].items():
            metrics = binary_set_metrics(current_y, pred)
            attack_mask = current_true_count > 0
            attack_metrics = binary_set_metrics(
                current_y[attack_mask],
                pred[attack_mask],
            )
            overview_rows.append(
                {
                    "split": split,
                    "representation": representation,
                    "decoder": decoder_name,
                    **metrics,
                    "attack_sample_count": int(np.sum(attack_mask)),
                    "attack_node_precision": attack_metrics["node_precision"],
                    "attack_node_recall": attack_metrics["node_recall"],
                    "attack_node_f1": attack_metrics["node_f1"],
                    "attack_exact_localization": attack_metrics[
                        "exact_localization"
                    ],
                    "attack_empty_prediction_rate": attack_metrics[
                        "empty_prediction_rate"
                    ],
                    "count_accuracy": float(
                        np.mean(current_pred_count == current_true_count)
                    ),
                    "count_mae": float(
                        np.mean(
                            np.abs(current_pred_count - current_true_count)
                        )
                    ),
                }
            )

            categories = Counter(
                classify_set_error(t, p)
                for t, p in zip(current_y, pred)
            )
            for category, count in sorted(categories.items()):
                error_rows.append(
                    {
                        "split": split,
                        "representation": representation,
                        "decoder": decoder_name,
                        "error_category": category,
                        "sample_count": int(count),
                        "rate": count / len(pred),
                    }
                )

            count_rows.extend(
                group_metrics_rows(
                    split,
                    representation,
                    decoder_name,
                    "true_count",
                    current_true_count,
                    {value: str(value) for value in np.unique(current_true_count)},
                    current_y,
                    pred,
                    current_true_count,
                    current_pred_count,
                )
            )
            profile_rows.extend(
                group_metrics_rows(
                    split,
                    representation,
                    decoder_name,
                    "profile_id",
                    payload["profile"],
                    profile_map,
                    current_y,
                    pred,
                    current_true_count,
                    current_pred_count,
                )
            )
            attack_kind_rows.extend(
                group_metrics_rows(
                    split,
                    representation,
                    decoder_name,
                    "attack_kind_id",
                    payload["attack_kind"],
                    attack_map,
                    current_y,
                    pred,
                    current_true_count,
                    current_pred_count,
                )
            )
            topology_values = payload["topology"]
            topology_labels = {
                value: str(value) for value in np.unique(topology_values)
            }
            topology_rows.extend(
                group_metrics_rows(
                    split,
                    representation,
                    decoder_name,
                    "topology_group",
                    topology_values,
                    topology_labels,
                    current_y,
                    pred,
                    current_true_count,
                    current_pred_count,
                )
            )

    base_ep_counttopk = base["count_topk"][ep]
    base_ep_threshold = base["threshold"][ep]
    base_ep_oracle = base["oracle"][ep]

    paired_rows: list[dict[str, Any]] = []
    paired_code: dict[str, np.ndarray] = {}

    pairs = {
        "CountTopK": (
            base_ep_counttopk,
            temporal["count_topk_pred"],
        ),
        "NodeThreshold": (
            base_ep_threshold,
            temporal["threshold_pred"],
        ),
        "OracleTrueCountTopK": (
            base_ep_oracle,
            temporal["oracle_pred"],
        ),
    }

    for decoder_name, (base_pred, temporal_pred) in pairs.items():
        base_exact = np.all(base_pred == temporal_y_node, axis=1)
        temporal_exact = np.all(temporal_pred == temporal_y_node, axis=1)

        labels = np.empty(base_exact.shape[0], dtype=np.int8)
        labels[(base_exact == 1) & (temporal_exact == 1)] = 0
        labels[(base_exact == 1) & (temporal_exact == 0)] = 1
        labels[(base_exact == 0) & (temporal_exact == 1)] = 2
        labels[(base_exact == 0) & (temporal_exact == 0)] = 3
        paired_code[decoder_name] = labels

        for code, label in [
            (0, "both_exact"),
            (1, "single_only_exact"),
            (2, "temporal_only_exact"),
            (3, "both_wrong"),
        ]:
            mask = labels == code
            paired_rows.append(
                {
                    "split": split,
                    "decoder": decoder_name,
                    "overlap_category": label,
                    "sample_count": int(np.sum(mask)),
                    "rate": float(np.mean(mask)),
                }
            )

    overview_lookup = {
        (row["representation"], row["decoder"]): row
        for row in overview_rows
    }
    temporal_primary = overview_lookup[
        ("temporal_h32_candidate", "NodeThreshold")
    ]
    temporal_oracle = overview_lookup[
        ("temporal_h32_candidate", "OracleTrueCountTopK")
    ]
    oracle_gap = (
        temporal_oracle["attack_exact_localization"]
        - temporal_primary["attack_exact_localization"]
    )

    split_summary = {
        "split": split,
        "archive": str(archive_path),
        "single_window_sample_count": int(y_node.shape[0]),
        "temporal_sample_count": int(ep.size),
        "base_node_threshold": float(base_node_threshold),
        "temporal_policy": {
            "horizon": int(candidate_policy["horizon"]),
            "node_threshold": float(candidate_policy["node_threshold"]),
            "graph_threshold": float(candidate_policy["graph_threshold"]),
            "mode": str(candidate_policy["mode"]),
        },
        "temporal_primary_attack_exact": temporal_primary[
            "attack_exact_localization"
        ],
        "temporal_oracle_attack_exact": temporal_oracle[
            "attack_exact_localization"
        ],
        "temporal_oracle_minus_primary_exact_gap": oracle_gap,
    }

    tables = {
        "overview": overview_rows,
        "count": count_rows,
        "profile": profile_rows,
        "attack_kind": attack_kind_rows,
        "topology": topology_rows,
        "rank": rank_rows,
        "errors": error_rows,
        "confusion": confusion_rows,
        "paired": paired_rows,
    }

    sample_payload = {
        f"{split}__temporal_endpoints": ep.astype(np.int64),
        f"{split}__temporal_global_index": temporal_global_index.astype(np.int64),
        f"{split}__temporal_run_index": temporal_run_index.astype(np.int64),
        f"{split}__temporal_true_count": temporal_true_count.astype(np.int16),
        f"{split}__temporal_predicted_count": temporal["predicted_count"].astype(np.int16),
        f"{split}__base_counttopk_exact": np.all(
            base_ep_counttopk == temporal_y_node,
            axis=1,
        ).astype(np.int8),
        f"{split}__temporal_counttopk_exact": np.all(
            temporal["count_topk_pred"] == temporal_y_node,
            axis=1,
        ).astype(np.int8),
        f"{split}__base_threshold_exact": np.all(
            base_ep_threshold == temporal_y_node,
            axis=1,
        ).astype(np.int8),
        f"{split}__temporal_threshold_exact": np.all(
            temporal["threshold_pred"] == temporal_y_node,
            axis=1,
        ).astype(np.int8),
        f"{split}__base_oracle_exact": np.all(
            base_ep_oracle == temporal_y_node,
            axis=1,
        ).astype(np.int8),
        f"{split}__temporal_oracle_exact": np.all(
            temporal["oracle_pred"] == temporal_y_node,
            axis=1,
        ).astype(np.int8),
    }
    for decoder_name, labels in paired_code.items():
        sample_payload[
            f"{split}__paired_{decoder_name.lower()}_code"
        ] = labels.astype(np.int8)

    return split_summary, tables, sample_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-search-script", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--validation-policy-lock", type=Path, required=True)
    parser.add_argument("--test-transfer-lock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--operational-policy", type=Path, required=True)
    parser.add_argument("--operational-policy-lock", type=Path, required=True)
    parser.add_argument("--a3-11-closure-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-node-threshold", type=float, default=0.78)
    parser.add_argument("--deoverlap-stride", type=int, default=8)
    args = parser.parse_args()

    metadata_path = args.data_dir / "metadata.json"
    edge_path = args.data_dir / "edge_index.npy"

    required = [
        args.validation_search_script,
        args.data_dir,
        args.validation_predictions,
        args.test_predictions,
        args.validation_policy_lock,
        args.test_transfer_lock,
        args.checkpoint,
        args.operational_policy,
        args.operational_policy_lock,
        args.a3_11_closure_lock,
        metadata_path,
        edge_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("A3.12 FAIL: missing required paths", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(f"A3.12 FAIL: output exists: {args.output_dir}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True)

    validation_lock = load_json(args.validation_policy_lock)
    test_lock = load_json(args.test_transfer_lock)
    operational_lock = load_json(args.operational_policy_lock)
    closure_lock = load_json(args.a3_11_closure_lock)
    operational = load_json(args.operational_policy)

    checkpoint_sha = sha256_file(args.checkpoint)
    validation_sha = sha256_file(args.validation_predictions)
    test_sha = sha256_file(args.test_predictions)
    operational_sha = sha256_file(args.operational_policy)

    failures: list[str] = []
    if validation_lock.get("validation_predictions_sha256") != validation_sha:
        failures.append("validation prediction hash mismatch")
    if test_lock.get("test_predictions_sha256") != test_sha:
        failures.append("test prediction hash mismatch")
    if operational_lock.get("operational_policy_family_sha256") != operational_sha:
        failures.append("operational policy hash mismatch")
    for label, value in [
        ("validation checkpoint", validation_lock.get("checkpoint_sha256")),
        ("test checkpoint", test_lock.get("checkpoint_sha256")),
        ("operational checkpoint", operational_lock.get("checkpoint_sha256")),
        ("closure checkpoint", closure_lock.get("checkpoint_sha256")),
    ]:
        if value != checkpoint_sha:
            failures.append(f"{label} mismatch")
    if closure_lock.get("policy_selection_performed_on_test") is not False:
        failures.append("closure lock reports test policy selection")

    if failures:
        payload = {"status": "FAIL", "failures": failures}
        (args.output_dir / "a3_12_failure.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1

    metadata = load_json(metadata_path)
    edge_index = np.load(edge_path)
    node_classes = build_node_classes(edge_index, 16)
    module = load_module(args.validation_search_script)

    candidate_policy = operational["policy_roles"][
        "candidate_localization"
    ]["policy"]

    all_tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summaries: dict[str, Any] = {}
    sample_payload: dict[str, np.ndarray] = {}

    for split, archive in [
        ("validation", args.validation_predictions),
        ("development_test", args.test_predictions),
    ]:
        print(f"processing split={split}", flush=True)
        summary, tables, payload = evaluate_split(
            split,
            archive,
            args.data_dir,
            module,
            metadata,
            node_classes,
            candidate_policy,
            args.base_node_threshold,
            args.deoverlap_stride,
        )
        summaries[split] = summary
        for name, rows in tables.items():
            all_tables[name].extend(rows)
        sample_payload.update(payload)

    write_csv(args.output_dir / "split_overview.csv", all_tables["overview"])
    write_csv(args.output_dir / "count_confusion.csv", all_tables["confusion"])
    write_csv(args.output_dir / "count_group_metrics.csv", all_tables["count"])
    write_csv(args.output_dir / "profile_group_metrics.csv", all_tables["profile"])
    write_csv(
        args.output_dir / "attack_kind_group_metrics.csv",
        all_tables["attack_kind"],
    )
    write_csv(
        args.output_dir / "topology_group_metrics.csv",
        all_tables["topology"],
    )
    write_csv(args.output_dir / "rank_distribution.csv", all_tables["rank"])
    write_csv(
        args.output_dir / "error_category_counts.csv",
        all_tables["errors"],
    )
    write_csv(
        args.output_dir / "paired_exact_overlap.csv",
        all_tables["paired"],
    )
    np.savez_compressed(
        args.output_dir / "per_sample_decomposition.npz",
        **sample_payload,
    )

    validation_gap = summaries["validation"][
        "temporal_oracle_minus_primary_exact_gap"
    ]
    validation_oracle = summaries["validation"][
        "temporal_oracle_attack_exact"
    ]

    if validation_gap >= 0.10 and validation_oracle >= 0.75:
        recommendation = (
            "A4a structured null-aware set head is justified after A3.13 "
            "confirms the encoder representation is not the dominant limiter."
        )
        decision_class = "decoder_cardinality_gap_is_large"
    elif validation_oracle < 0.70:
        recommendation = (
            "Router ranking remains too weak for a decoder-only fix. "
            "Prioritize A3.13 and A3.15, then A4b or V5."
        )
        decision_class = "representation_ranking_is_primary_limiter"
    else:
        recommendation = (
            "Both decoder/cardinality and representation errors matter. "
            "Proceed to A3.13 and A3.15 before choosing A4a versus A4b."
        )
        decision_class = "mixed_decoder_and_representation_limit"

    summary = {
        "status": "PASS",
        "designation": "V4-A3.12 Cardinality and Router-Ranking Decomposition",
        "analysis_role": "descriptive; no test policy selection",
        "base_node_threshold": args.base_node_threshold,
        "candidate_policy": {
            "horizon": candidate_policy["horizon"],
            "mode": candidate_policy["mode"],
            "node_threshold": candidate_policy["node_threshold"],
            "graph_threshold": candidate_policy["graph_threshold"],
        },
        "split_summaries": summaries,
        "validation_based_decision_class": decision_class,
        "validation_based_recommendation": recommendation,
        "development_test_used_for_selection": False,
        "node_topology_classes": node_classes,
        "provenance": {
            "checkpoint_sha256": checkpoint_sha,
            "validation_predictions_sha256": validation_sha,
            "test_predictions_sha256": test_sha,
            "operational_policy_sha256": operational_sha,
            "a3_11_closure_summary_sha256": closure_lock[
                "closure_summary_sha256"
            ],
            "validation_search_script_sha256": sha256_file(
                args.validation_search_script
            ),
            "metadata_sha256": sha256_file(metadata_path),
        },
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lock = {
        "status": "A3_12_CARDINALITY_RANKING_COMPLETE",
        "development_test_used_for_selection": False,
        "checkpoint_sha256": checkpoint_sha,
        "validation_predictions_sha256": validation_sha,
        "test_predictions_sha256": test_sha,
        "summary_sha256": sha256_file(summary_path),
        "per_sample_decomposition_sha256": sha256_file(
            args.output_dir / "per_sample_decomposition.npz"
        ),
    }
    (args.output_dir / "A3_12_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "V4_A3_12_CARDINALITY_RANKING_PASS").write_text(
        "V4_A3_12_CARDINALITY_RANKING_PASS\n",
        encoding="utf-8",
    )

    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    print("V4_A3_12_CARDINALITY_RANKING_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
