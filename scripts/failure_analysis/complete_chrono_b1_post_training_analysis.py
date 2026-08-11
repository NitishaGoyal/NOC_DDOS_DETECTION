#!/usr/bin/env python3
"""
Chrono-B1 combined post-training analysis: stages B1.14 through B1.17.

This script completes all remaining scientific analysis without retraining:
- detailed attack-only localization analysis on validation and blind test;
- grouping by attacker count, profile, strength, attacker placement,
  active-core configuration, seed, and run ID;
- matched A1/B1 disagreements for graph correctness, exact localization,
  Top-1 localization, and Top-3 localization;
- paired sample-level bootstrap confidence intervals;
- paired run-level bootstrap intervals for macro FPR and macro recall;
- frozen safeguard evaluation;
- formal retain / promising but inconclusive / reject verdict;
- one evidence-based next-experiment recommendation;
- JSON, CSV, and Markdown reports.

Thresholds are read from the validation-only B1.11 report and are never
reselected on test.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HARD_NORMAL_RUN = "N-3-7-8-12-Pmixed-R18-V3"

REQUIRED_PREDICTION_KEYS = (
    "split",
    "real_index",
    "graph_prob",
    "node_prob",
    "y_graph",
    "y_node",
    "run_id",
    "profile",
    "strength",
    "active_cores",
    "attackers",
    "end_epoch",
    "seed",
    "dataset_split",
)

ALIGNMENT_KEYS = (
    "real_index",
    "y_graph",
    "y_node",
    "run_id",
    "profile",
    "strength",
    "active_cores",
    "attackers",
    "end_epoch",
    "seed",
    "dataset_split",
)

GROUP_DIMENSIONS = (
    "attacker_count",
    "profile",
    "strength",
    "attacker_placement",
    "active_cores",
    "seed",
    "run_id",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def relative_delta(delta: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return float(delta / abs(baseline))


def to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def text(value: Any) -> str:
    value = to_python(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if value is None:
        return "NA"
    result = str(value).strip()
    return result if result else "NA"


def canonical_strength(value: Any) -> str:
    raw = text(value).strip('"').strip("'")
    if raw.lower() in {"na", "nan", "none", "null", ""}:
        return "NA"
    try:
        number = float(raw)
        if math.isfinite(number):
            return str(int(number)) if number.is_integer() else f"{number:g}"
    except ValueError:
        pass
    return raw


def parse_sequence(value: Any) -> list[Any]:
    value = to_python(value)
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

    raw = str(value).strip()
    if not raw or raw.lower() in {"na", "nan", "none", "null", "[]", "()"}:
        return []

    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple, set, np.ndarray)):
            return list(parsed)
        if isinstance(parsed, (int, float)):
            return [parsed]
    except (ValueError, SyntaxError):
        pass

    for separator in ("-", ",", "_", " "):
        if separator in raw:
            return [part.strip() for part in raw.split(separator) if part.strip()]

    return [raw]


def canonical_sequence(value: Any) -> str:
    normalized: list[str] = []
    for item in parse_sequence(value):
        raw = text(item)
        if raw.lower() in {"na", "nan", "none", "null", "-1"}:
            continue
        try:
            normalized.append(str(int(float(raw))))
        except ValueError:
            normalized.append(raw)

    if not normalized:
        return "NA"

    def key_fn(token: str) -> tuple[int, Any]:
        try:
            return (0, int(token))
        except ValueError:
            return (1, token)

    return "-".join(sorted(dict.fromkeys(normalized), key=key_fn))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_prediction_file(
    label: str,
    data: dict[str, np.ndarray],
    expected_split: str,
) -> dict[str, bool]:
    missing = sorted(set(REQUIRED_PREDICTION_KEYS) - set(data))
    if missing:
        raise ValueError(f"{label}: missing arrays: {missing}")

    graph_prob = np.asarray(data["graph_prob"])
    node_prob = np.asarray(data["node_prob"])
    y_graph = np.asarray(data["y_graph"])
    y_node = np.asarray(data["y_node"])

    checks = {
        "split_marker_correct": str(np.asarray(data["split"]).item()) == expected_split,
        "all_dataset_split_rows_correct": bool(
            np.all(np.asarray(data["dataset_split"]).astype(str) == expected_split)
        ),
        "graph_probability_shape_valid": graph_prob.shape == y_graph.shape,
        "node_probability_shape_valid": node_prob.shape == y_node.shape,
        "row_counts_match": (
            len(graph_prob)
            == len(node_prob)
            == len(data["real_index"])
            == len(data["run_id"])
        ),
        "graph_probabilities_finite": bool(np.isfinite(graph_prob).all()),
        "node_probabilities_finite": bool(np.isfinite(node_prob).all()),
        "graph_probabilities_in_unit_interval": bool(
            np.all((graph_prob >= 0.0) & (graph_prob <= 1.0))
        ),
        "node_probabilities_in_unit_interval": bool(
            np.all((node_prob >= 0.0) & (node_prob <= 1.0))
        ),
        "graph_labels_binary": bool(
            np.all(np.isin(np.unique(y_graph), [0, 1, 0.0, 1.0]))
        ),
        "node_labels_binary": bool(
            np.all(np.isin(np.unique(y_node), [0, 1, 0.0, 1.0]))
        ),
        "real_indices_unique": len(np.unique(data["real_index"])) == len(data["real_index"]),
    }
    return checks


def validate_alignment(
    a1: dict[str, np.ndarray],
    b1: dict[str, np.ndarray],
    split_name: str,
) -> dict[str, bool]:
    checks = {
        key: bool(np.array_equal(a1[key], b1[key]))
        for key in ALIGNMENT_KEYS
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{split_name}: A1/B1 alignment failed: {failed}")
    return checks


def graph_sample_arrays(
    y_graph: np.ndarray,
    graph_prob: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    truth = np.asarray(y_graph, dtype=np.int64).reshape(-1)
    probability = np.asarray(graph_prob, dtype=np.float64).reshape(-1)
    prediction = (probability >= threshold).astype(np.int64)
    correct = prediction == truth
    return {
        "truth": truth,
        "probability": probability,
        "prediction": prediction,
        "correct": correct,
    }


def graph_metrics_from_arrays(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    truth = arrays["truth"]
    prediction = arrays["prediction"]

    tp = int(np.sum((truth == 1) & (prediction == 1)))
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)

    return {
        "sample_count": int(len(truth)),
        "positive_count": int(np.sum(truth == 1)),
        "negative_count": int(np.sum(truth == 0)),
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": safe_div(fp, fp + tn),
        "tnr": safe_div(tn, tn + fp),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def localization_sample_arrays(
    y_node: np.ndarray,
    node_prob: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    truth = np.asarray(y_node, dtype=np.int64)
    probability = np.asarray(node_prob, dtype=np.float64)

    if truth.ndim != 2 or probability.shape != truth.shape:
        raise ValueError(
            f"Localization shape mismatch: truth={truth.shape}, probability={probability.shape}"
        )

    attacker_count = truth.sum(axis=1).astype(np.int64)
    if np.any(attacker_count <= 0):
        bad = int(np.sum(attacker_count <= 0))
        raise ValueError(f"{bad} attack-positive samples have no positive node labels.")

    prediction = (probability >= threshold).astype(np.int64)
    tp = np.sum((truth == 1) & (prediction == 1), axis=1).astype(np.int64)
    fp = np.sum((truth == 0) & (prediction == 1), axis=1).astype(np.int64)
    fn = np.sum((truth == 1) & (prediction == 0), axis=1).astype(np.int64)
    exact = np.all(prediction == truth, axis=1)

    order = np.argsort(-probability, axis=1, kind="stable")
    inverse_rank = np.empty_like(order)
    row_ids = np.arange(len(order))[:, None]
    inverse_rank[row_ids, order] = np.arange(1, order.shape[1] + 1)[None, :]

    top_hits: dict[int, np.ndarray] = {}
    top_recalls: dict[int, np.ndarray] = {}

    best_rank = np.empty(len(truth), dtype=np.int64)
    mean_all_rank = np.empty(len(truth), dtype=np.float64)
    reciprocal_best_rank = np.empty(len(truth), dtype=np.float64)
    exact_top_m = np.empty(len(truth), dtype=bool)
    ranking_margin = np.empty(len(truth), dtype=np.float64)

    for i in range(len(truth)):
        true_nodes = np.flatnonzero(truth[i] == 1)
        false_nodes = np.flatnonzero(truth[i] == 0)
        ranks = inverse_rank[i, true_nodes].astype(np.int64)

        best_rank[i] = int(ranks.min())
        mean_all_rank[i] = float(np.mean(ranks))
        reciprocal_best_rank[i] = 1.0 / float(best_rank[i])

        for k in (1, 2, 3):
            selected = order[i, :k]
            true_in_top = int(np.sum(np.isin(selected, true_nodes)))
            if k not in top_hits:
                top_hits[k] = np.empty(len(truth), dtype=bool)
                top_recalls[k] = np.empty(len(truth), dtype=np.float64)
            top_hits[k][i] = true_in_top > 0
            top_recalls[k][i] = true_in_top / len(true_nodes)

        m = len(true_nodes)
        exact_top_m[i] = set(order[i, :m].tolist()) == set(true_nodes.tolist())

        minimum_true = float(np.min(probability[i, true_nodes]))
        maximum_false = (
            float(np.max(probability[i, false_nodes]))
            if len(false_nodes)
            else 0.0
        )
        ranking_margin[i] = minimum_true - maximum_false

    return {
        "truth": truth,
        "probability": probability,
        "prediction": prediction,
        "attacker_count": attacker_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "exact": exact,
        "top1_hit": top_hits[1],
        "top2_hit": top_hits[2],
        "top3_hit": top_hits[3],
        "top1_recall": top_recalls[1],
        "top2_recall": top_recalls[2],
        "top3_recall": top_recalls[3],
        "best_rank": best_rank,
        "mean_all_rank": mean_all_rank,
        "reciprocal_best_rank": reciprocal_best_rank,
        "exact_top_m": exact_top_m,
        "ranking_margin": ranking_margin,
    }


def aggregate_localization(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray | None = None,
) -> dict[str, Any]:
    if indices is None:
        indices = np.arange(len(arrays["tp"]), dtype=np.int64)
    else:
        indices = np.asarray(indices, dtype=np.int64)

    tp = int(np.sum(arrays["tp"][indices]))
    fp = int(np.sum(arrays["fp"][indices]))
    fn = int(np.sum(arrays["fn"][indices]))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)

    return {
        "sample_count": int(len(indices)),
        "attacker_node_count": int(
            np.sum(arrays["attacker_count"][indices])
        ),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "exact_localization": float(np.mean(arrays["exact"][indices])),
        "top1_hit_rate": float(np.mean(arrays["top1_hit"][indices])),
        "top2_hit_rate": float(np.mean(arrays["top2_hit"][indices])),
        "top3_hit_rate": float(np.mean(arrays["top3_hit"][indices])),
        "top1_attacker_recall": float(np.mean(arrays["top1_recall"][indices])),
        "top2_attacker_recall": float(np.mean(arrays["top2_recall"][indices])),
        "top3_attacker_recall": float(np.mean(arrays["top3_recall"][indices])),
        "mean_reciprocal_rank": float(
            np.mean(arrays["reciprocal_best_rank"][indices])
        ),
        "mean_best_attacker_rank": float(
            np.mean(arrays["best_rank"][indices])
        ),
        "mean_all_attacker_rank": float(
            np.mean(arrays["mean_all_rank"][indices])
        ),
        "exact_top_m_set_accuracy": float(
            np.mean(arrays["exact_top_m"][indices])
        ),
        "mean_ranking_margin": float(
            np.mean(arrays["ranking_margin"][indices])
        ),
    }


def build_attack_metadata(
    data: dict[str, np.ndarray],
    attack_indices: np.ndarray,
    attacker_count_values: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "attacker_count": np.asarray(
            [str(int(value)) for value in attacker_count_values]
        ),
        "profile": np.asarray(
            [text(value).lower() for value in data["profile"][attack_indices]]
        ),
        "strength": np.asarray(
            [canonical_strength(value) for value in data["strength"][attack_indices]]
        ),
        "attacker_placement": np.asarray(
            [canonical_sequence(value) for value in data["attackers"][attack_indices]]
        ),
        "active_cores": np.asarray(
            [canonical_sequence(value) for value in data["active_cores"][attack_indices]]
        ),
        "seed": np.asarray(
            [text(value) for value in data["seed"][attack_indices]]
        ),
        "run_id": np.asarray(
            [text(value) for value in data["run_id"][attack_indices]]
        ),
    }


def localization_group_rows(
    split_name: str,
    model_name: str,
    threshold: float,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in GROUP_DIMENSIONS:
        values = metadata[dimension]
        for group in sorted(set(values.tolist())):
            indices = np.flatnonzero(values == group)
            metrics = aggregate_localization(arrays, indices)
            rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "node_threshold": threshold,
                    "denominator": "attack-positive samples only",
                    "dimension": dimension,
                    "group": group,
                    **metrics,
                }
            )
    return rows


def metric_delta_rows(
    split_name: str,
    a1_graph: dict[str, Any],
    b1_graph: dict[str, Any],
    a1_loc: dict[str, Any],
    b1_loc: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics = {
        "graph_accuracy": (a1_graph["accuracy"], b1_graph["accuracy"]),
        "graph_precision": (a1_graph["precision"], b1_graph["precision"]),
        "graph_recall": (a1_graph["recall"], b1_graph["recall"]),
        "graph_f1": (a1_graph["f1"], b1_graph["f1"]),
        "graph_fpr": (a1_graph["fpr"], b1_graph["fpr"]),
        "attack_only_node_precision": (
            a1_loc["micro_precision"],
            b1_loc["micro_precision"],
        ),
        "attack_only_node_recall": (
            a1_loc["micro_recall"],
            b1_loc["micro_recall"],
        ),
        "attack_only_node_f1": (a1_loc["micro_f1"], b1_loc["micro_f1"]),
        "attack_only_exact_localization": (
            a1_loc["exact_localization"],
            b1_loc["exact_localization"],
        ),
        "top1_hit_rate": (a1_loc["top1_hit_rate"], b1_loc["top1_hit_rate"]),
        "top2_hit_rate": (a1_loc["top2_hit_rate"], b1_loc["top2_hit_rate"]),
        "top3_hit_rate": (a1_loc["top3_hit_rate"], b1_loc["top3_hit_rate"]),
        "top1_attacker_recall": (
            a1_loc["top1_attacker_recall"],
            b1_loc["top1_attacker_recall"],
        ),
        "top2_attacker_recall": (
            a1_loc["top2_attacker_recall"],
            b1_loc["top2_attacker_recall"],
        ),
        "top3_attacker_recall": (
            a1_loc["top3_attacker_recall"],
            b1_loc["top3_attacker_recall"],
        ),
        "mean_reciprocal_rank": (
            a1_loc["mean_reciprocal_rank"],
            b1_loc["mean_reciprocal_rank"],
        ),
        "mean_best_attacker_rank": (
            a1_loc["mean_best_attacker_rank"],
            b1_loc["mean_best_attacker_rank"],
        ),
        "mean_all_attacker_rank": (
            a1_loc["mean_all_attacker_rank"],
            b1_loc["mean_all_attacker_rank"],
        ),
        "exact_top_m_set_accuracy": (
            a1_loc["exact_top_m_set_accuracy"],
            b1_loc["exact_top_m_set_accuracy"],
        ),
    }

    rows: list[dict[str, Any]] = []
    for name, (a1_value, b1_value) in metrics.items():
        delta = float(b1_value - a1_value)
        rows.append(
            {
                "split": split_name,
                "metric": name,
                "a1": float(a1_value),
                "b1": float(b1_value),
                "absolute_delta_b1_minus_a1": delta,
                "relative_delta": relative_delta(delta, float(a1_value)),
            }
        )
    return rows


def disagreement_status(a_correct: np.ndarray, b_correct: np.ndarray) -> np.ndarray:
    status = np.empty(len(a_correct), dtype=object)
    status[a_correct & b_correct] = "both_correct"
    status[(~a_correct) & (~b_correct)] = "both_wrong"
    status[(~a_correct) & b_correct] = "B1_fixes_A1"
    status[a_correct & (~b_correct)] = "B1_breaks_A1"
    return status


def disagreement_summary_row(
    split_name: str,
    metric_name: str,
    status: np.ndarray,
    denominator: str,
) -> dict[str, Any]:
    counts = Counter(status.tolist())
    total = len(status)
    return {
        "split": split_name,
        "metric": metric_name,
        "denominator": denominator,
        "sample_count": total,
        "both_correct": counts.get("both_correct", 0),
        "both_wrong": counts.get("both_wrong", 0),
        "B1_fixes_A1": counts.get("B1_fixes_A1", 0),
        "B1_breaks_A1": counts.get("B1_breaks_A1", 0),
        "net_fixed_minus_broken": (
            counts.get("B1_fixes_A1", 0) - counts.get("B1_breaks_A1", 0)
        ),
        "fix_rate": safe_div(counts.get("B1_fixes_A1", 0), total),
        "break_rate": safe_div(counts.get("B1_breaks_A1", 0), total),
    }


def base_sample_metadata(
    data: dict[str, np.ndarray],
    index: int,
) -> dict[str, Any]:
    y_graph = int(data["y_graph"][index])
    placement = canonical_sequence(data["attackers"][index])
    active_cores = canonical_sequence(data["active_cores"][index])
    return {
        "real_index": int(data["real_index"][index]),
        "run_id": text(data["run_id"][index]),
        "profile": text(data["profile"][index]).lower(),
        "strength": canonical_strength(data["strength"][index]),
        "attacker_count": (
            int(np.sum(data["y_node"][index]))
            if y_graph == 1
            else 0
        ),
        "attacker_placement": placement,
        "active_cores": active_cores,
        "seed": text(data["seed"][index]),
        "end_epoch": int(data["end_epoch"][index]),
        "y_graph": y_graph,
    }


def select_important_disagreements(
    *,
    split_name: str,
    data: dict[str, np.ndarray],
    attack_indices: np.ndarray,
    a1_graph: dict[str, np.ndarray],
    b1_graph: dict[str, np.ndarray],
    a1_loc: dict[str, np.ndarray],
    b1_loc: dict[str, np.ndarray],
    graph_threshold_a1: float,
    graph_threshold_b1: float,
    top_n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    graph_status = disagreement_status(a1_graph["correct"], b1_graph["correct"])
    summary_rows.append(
        disagreement_summary_row(
            split_name,
            "graph_correctness",
            graph_status,
            "all samples",
        )
    )

    metric_specs = [
        (
            "exact_localization",
            a1_loc["exact"],
            b1_loc["exact"],
        ),
        (
            "top1_localization",
            a1_loc["top1_hit"],
            b1_loc["top1_hit"],
        ),
        (
            "top3_localization",
            a1_loc["top3_hit"],
            b1_loc["top3_hit"],
        ),
    ]

    location_statuses: dict[str, np.ndarray] = {}
    for metric_name, a_correct, b_correct in metric_specs:
        status = disagreement_status(a_correct, b_correct)
        location_statuses[metric_name] = status
        summary_rows.append(
            disagreement_summary_row(
                split_name,
                metric_name,
                status,
                "attack-positive samples only",
            )
        )

    # Graph fixed/broken examples.
    graph_importance = (
        np.abs(a1_graph["probability"] - graph_threshold_a1)
        + np.abs(b1_graph["probability"] - graph_threshold_b1)
    )
    for transition in ("B1_fixes_A1", "B1_breaks_A1"):
        candidates = np.flatnonzero(graph_status == transition)
        order = candidates[
            np.argsort(-graph_importance[candidates], kind="stable")
        ][:top_n]

        for index in order:
            row = {
                "split": split_name,
                "metric": "graph_correctness",
                "transition": transition,
                "importance_score": float(graph_importance[index]),
                **base_sample_metadata(data, int(index)),
                "a1_graph_probability": float(a1_graph["probability"][index]),
                "b1_graph_probability": float(b1_graph["probability"][index]),
                "a1_graph_prediction": int(a1_graph["prediction"][index]),
                "b1_graph_prediction": int(b1_graph["prediction"][index]),
                "a1_graph_correct": bool(a1_graph["correct"][index]),
                "b1_graph_correct": bool(b1_graph["correct"][index]),
                "a1_exact_localization": "",
                "b1_exact_localization": "",
                "a1_top1_hit": "",
                "b1_top1_hit": "",
                "a1_top3_hit": "",
                "b1_top3_hit": "",
                "a1_best_attacker_rank": "",
                "b1_best_attacker_rank": "",
                "a1_ranking_margin": "",
                "b1_ranking_margin": "",
            }
            sample_rows.append(row)

    # Localization fixed/broken examples.
    local_importance = np.abs(
        b1_loc["ranking_margin"] - a1_loc["ranking_margin"]
    )
    for metric_name, status in location_statuses.items():
        for transition in ("B1_fixes_A1", "B1_breaks_A1"):
            candidates = np.flatnonzero(status == transition)
            order = candidates[
                np.argsort(-local_importance[candidates], kind="stable")
            ][:top_n]

            for local_index in order:
                real_position = int(attack_indices[local_index])
                row = {
                    "split": split_name,
                    "metric": metric_name,
                    "transition": transition,
                    "importance_score": float(local_importance[local_index]),
                    **base_sample_metadata(data, real_position),
                    "a1_graph_probability": float(
                        a1_graph["probability"][real_position]
                    ),
                    "b1_graph_probability": float(
                        b1_graph["probability"][real_position]
                    ),
                    "a1_graph_prediction": int(
                        a1_graph["prediction"][real_position]
                    ),
                    "b1_graph_prediction": int(
                        b1_graph["prediction"][real_position]
                    ),
                    "a1_graph_correct": bool(
                        a1_graph["correct"][real_position]
                    ),
                    "b1_graph_correct": bool(
                        b1_graph["correct"][real_position]
                    ),
                    "a1_exact_localization": bool(a1_loc["exact"][local_index]),
                    "b1_exact_localization": bool(b1_loc["exact"][local_index]),
                    "a1_top1_hit": bool(a1_loc["top1_hit"][local_index]),
                    "b1_top1_hit": bool(b1_loc["top1_hit"][local_index]),
                    "a1_top3_hit": bool(a1_loc["top3_hit"][local_index]),
                    "b1_top3_hit": bool(b1_loc["top3_hit"][local_index]),
                    "a1_best_attacker_rank": int(
                        a1_loc["best_rank"][local_index]
                    ),
                    "b1_best_attacker_rank": int(
                        b1_loc["best_rank"][local_index]
                    ),
                    "a1_ranking_margin": float(
                        a1_loc["ranking_margin"][local_index]
                    ),
                    "b1_ranking_margin": float(
                        b1_loc["ranking_margin"][local_index]
                    ),
                }
                sample_rows.append(row)

    return summary_rows, sample_rows


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def bootstrap_graph_deltas(
    y_true: np.ndarray,
    a_pred: np.ndarray,
    b_pred: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    n = len(y_true)
    results = {
        "graph_f1": np.empty(repetitions, dtype=np.float64),
        "graph_recall": np.empty(repetitions, dtype=np.float64),
        "graph_fpr": np.empty(repetitions, dtype=np.float64),
    }

    def metrics(truth: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
        tp = int(np.sum((truth == 1) & (pred == 1)))
        tn = int(np.sum((truth == 0) & (pred == 0)))
        fp = int(np.sum((truth == 0) & (pred == 1)))
        fn = int(np.sum((truth == 1) & (pred == 0)))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        fpr = safe_div(fp, fp + tn)
        return f1, recall, fpr

    for repetition in range(repetitions):
        indices = rng.integers(0, n, size=n)
        truth = y_true[indices]
        a_values = metrics(truth, a_pred[indices])
        b_values = metrics(truth, b_pred[indices])
        results["graph_f1"][repetition] = b_values[0] - a_values[0]
        results["graph_recall"][repetition] = b_values[1] - a_values[1]
        results["graph_fpr"][repetition] = b_values[2] - a_values[2]

    return results


def bootstrap_localization_deltas(
    a_arrays: dict[str, np.ndarray],
    b_arrays: dict[str, np.ndarray],
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    n = len(a_arrays["tp"])
    results = {
        "attack_only_node_f1": np.empty(repetitions, dtype=np.float64),
        "attack_only_exact_localization": np.empty(repetitions, dtype=np.float64),
        "top1_hit_rate": np.empty(repetitions, dtype=np.float64),
        "top3_hit_rate": np.empty(repetitions, dtype=np.float64),
        "mean_reciprocal_rank": np.empty(repetitions, dtype=np.float64),
    }

    def node_f1(arrays: dict[str, np.ndarray], indices: np.ndarray) -> float:
        tp = int(np.sum(arrays["tp"][indices]))
        fp = int(np.sum(arrays["fp"][indices]))
        fn = int(np.sum(arrays["fn"][indices]))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        return safe_div(2.0 * precision * recall, precision + recall)

    for repetition in range(repetitions):
        indices = rng.integers(0, n, size=n)
        results["attack_only_node_f1"][repetition] = (
            node_f1(b_arrays, indices) - node_f1(a_arrays, indices)
        )
        results["attack_only_exact_localization"][repetition] = (
            float(np.mean(b_arrays["exact"][indices]))
            - float(np.mean(a_arrays["exact"][indices]))
        )
        results["top1_hit_rate"][repetition] = (
            float(np.mean(b_arrays["top1_hit"][indices]))
            - float(np.mean(a_arrays["top1_hit"][indices]))
        )
        results["top3_hit_rate"][repetition] = (
            float(np.mean(b_arrays["top3_hit"][indices]))
            - float(np.mean(a_arrays["top3_hit"][indices]))
        )
        results["mean_reciprocal_rank"][repetition] = (
            float(np.mean(b_arrays["reciprocal_best_rank"][indices]))
            - float(np.mean(a_arrays["reciprocal_best_rank"][indices]))
        )

    return results


def per_run_graph_values(
    data: dict[str, np.ndarray],
    graph_arrays: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, float]]:
    run_ids = np.asarray([text(value) for value in data["run_id"]])
    truth = graph_arrays["truth"]
    prediction = graph_arrays["prediction"]

    normal_fpr: dict[str, float] = {}
    attack_recall: dict[str, float] = {}

    for run_id in sorted(set(run_ids.tolist())):
        mask = run_ids == run_id
        run_truth = truth[mask]
        run_pred = prediction[mask]

        positives = int(np.sum(run_truth == 1))
        negatives = int(np.sum(run_truth == 0))

        if positives > 0 and negatives == 0:
            tp = int(np.sum((run_truth == 1) & (run_pred == 1)))
            fn = int(np.sum((run_truth == 1) & (run_pred == 0)))
            attack_recall[run_id] = safe_div(tp, tp + fn)
        elif negatives > 0 and positives == 0:
            fp = int(np.sum((run_truth == 0) & (run_pred == 1)))
            tn = int(np.sum((run_truth == 0) & (run_pred == 0)))
            normal_fpr[run_id] = safe_div(fp, fp + tn)

    return normal_fpr, attack_recall


def bootstrap_run_macro_deltas(
    a_values: dict[str, float],
    b_values: dict[str, float],
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if set(a_values) != set(b_values):
        raise ValueError("A1/B1 run sets differ during run-level bootstrap.")

    run_ids = sorted(a_values)
    a = np.asarray([a_values[run_id] for run_id in run_ids], dtype=np.float64)
    b = np.asarray([b_values[run_id] for run_id in run_ids], dtype=np.float64)
    count = len(run_ids)
    result = np.empty(repetitions, dtype=np.float64)

    for repetition in range(repetitions):
        indices = rng.integers(0, count, size=count)
        result[repetition] = float(np.mean(b[indices]) - np.mean(a[indices]))

    return result


def bootstrap_rows_for_split(
    *,
    split_name: str,
    graph_truth: np.ndarray,
    a_graph: dict[str, np.ndarray],
    b_graph: dict[str, np.ndarray],
    a_loc: dict[str, np.ndarray],
    b_loc: dict[str, np.ndarray],
    a_normal_runs: dict[str, float],
    b_normal_runs: dict[str, float],
    a_attack_runs: dict[str, float],
    b_attack_runs: dict[str, float],
    point_deltas: dict[str, float],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)

    graph_results = bootstrap_graph_deltas(
        graph_truth,
        a_graph["prediction"],
        b_graph["prediction"],
        repetitions,
        rng,
    )
    localization_results = bootstrap_localization_deltas(
        a_loc,
        b_loc,
        repetitions,
        rng,
    )
    normal_macro = bootstrap_run_macro_deltas(
        a_normal_runs,
        b_normal_runs,
        repetitions,
        rng,
    )
    attack_macro = bootstrap_run_macro_deltas(
        a_attack_runs,
        b_attack_runs,
        repetitions,
        rng,
    )

    result_sets = {
        **graph_results,
        **localization_results,
        "normal_run_macro_fpr": normal_macro,
        "attack_run_macro_recall": attack_macro,
    }

    rows: list[dict[str, Any]] = []
    for metric_name, values in result_sets.items():
        lower, upper = percentile_interval(values)
        point = float(point_deltas[metric_name])
        rows.append(
            {
                "split": split_name,
                "metric": metric_name,
                "resampling_unit": (
                    "run"
                    if metric_name in {
                        "normal_run_macro_fpr",
                        "attack_run_macro_recall",
                    }
                    else (
                        "attack-positive sample"
                        if metric_name in {
                            "attack_only_node_f1",
                            "attack_only_exact_localization",
                            "top1_hit_rate",
                            "top3_hit_rate",
                            "mean_reciprocal_rank",
                        }
                        else "sample"
                    )
                ),
                "repetitions": repetitions,
                "seed": seed,
                "point_delta_b1_minus_a1": point,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
                "probability_delta_gt_zero": float(np.mean(values > 0.0)),
                "probability_delta_lt_zero": float(np.mean(values < 0.0)),
            }
        )
    return rows


def comparison_lookup(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    return {
        row["metric"]: float(row["absolute_delta_b1_minus_a1"])
        for row in rows
        if row["split"] == split
    }


def safeguard_rows(
    graph_report: dict[str, Any],
    validation_deltas: dict[str, float],
    test_deltas: dict[str, float],
) -> list[dict[str, Any]]:
    val_run = graph_report["validation"]["deltas_b1_minus_a1"]
    test_run = graph_report["test"]["deltas_b1_minus_a1"]

    criteria = [
        {
            "criterion": "normal_run_macro_fpr_improvement_at_least_0.02",
            "direction": "delta <= -0.02",
            "validation_delta": float(val_run["normal_run_macro_fpr"]),
            "test_delta": float(test_run["normal_run_macro_fpr"]),
            "limit": -0.02,
            "validation_pass": float(val_run["normal_run_macro_fpr"]) <= -0.02,
            "test_supports": float(test_run["normal_run_macro_fpr"]) <= 0.0,
        },
        {
            "criterion": "worst_normal_run_fpr_improvement_at_least_0.05",
            "direction": "delta <= -0.05",
            "validation_delta": float(val_run["worst_normal_run_fpr"]),
            "test_delta": float(test_run["worst_normal_run_fpr"]),
            "limit": -0.05,
            "validation_pass": float(val_run["worst_normal_run_fpr"]) <= -0.05,
            "test_supports": float(test_run["worst_normal_run_fpr"]) <= 0.0,
        },
        {
            "criterion": "overall_graph_recall_drop_no_worse_than_0.02",
            "direction": "delta >= -0.02",
            "validation_delta": validation_deltas["graph_recall"],
            "test_delta": test_deltas["graph_recall"],
            "limit": -0.02,
            "validation_pass": validation_deltas["graph_recall"] >= -0.02,
            "test_supports": test_deltas["graph_recall"] >= -0.02,
        },
        {
            "criterion": "attack_run_macro_recall_drop_no_worse_than_0.02",
            "direction": "delta >= -0.02",
            "validation_delta": float(val_run["attack_run_macro_recall"]),
            "test_delta": float(test_run["attack_run_macro_recall"]),
            "limit": -0.02,
            "validation_pass": float(val_run["attack_run_macro_recall"]) >= -0.02,
            "test_supports": float(test_run["attack_run_macro_recall"]) >= -0.02,
        },
        {
            "criterion": "strength20_recall_drop_no_worse_than_0.03",
            "direction": "delta >= -0.03",
            "validation_delta": float(val_run["strength20_macro_recall"]),
            "test_delta": float(test_run["strength20_macro_recall"]),
            "limit": -0.03,
            "validation_pass": float(val_run["strength20_macro_recall"]) >= -0.03,
            "test_supports": float(test_run["strength20_macro_recall"]) >= -0.03,
        },
        {
            "criterion": "worst_attack_run_recall_drop_no_worse_than_0.05",
            "direction": "delta >= -0.05",
            "validation_delta": float(val_run["worst_attack_run_recall"]),
            "test_delta": float(test_run["worst_attack_run_recall"]),
            "limit": -0.05,
            "validation_pass": float(val_run["worst_attack_run_recall"]) >= -0.05,
            "test_supports": float(test_run["worst_attack_run_recall"]) >= -0.05,
        },
        {
            "criterion": "attack_only_node_f1_drop_no_worse_than_0.02",
            "direction": "delta >= -0.02",
            "validation_delta": validation_deltas["attack_only_node_f1"],
            "test_delta": test_deltas["attack_only_node_f1"],
            "limit": -0.02,
            "validation_pass": validation_deltas["attack_only_node_f1"] >= -0.02,
            "test_supports": test_deltas["attack_only_node_f1"] >= -0.02,
        },
        {
            "criterion": "attack_only_exact_localization_drop_no_worse_than_0.03",
            "direction": "delta >= -0.03",
            "validation_delta": validation_deltas[
                "attack_only_exact_localization"
            ],
            "test_delta": test_deltas["attack_only_exact_localization"],
            "limit": -0.03,
            "validation_pass": validation_deltas[
                "attack_only_exact_localization"
            ]
            >= -0.03,
            "test_supports": test_deltas[
                "attack_only_exact_localization"
            ]
            >= -0.03,
        },
        {
            "criterion": "top1_drop_no_worse_than_0.02",
            "direction": "delta >= -0.02",
            "validation_delta": validation_deltas["top1_hit_rate"],
            "test_delta": test_deltas["top1_hit_rate"],
            "limit": -0.02,
            "validation_pass": validation_deltas["top1_hit_rate"] >= -0.02,
            "test_supports": test_deltas["top1_hit_rate"] >= -0.02,
        },
        {
            "criterion": "top3_drop_no_worse_than_0.01",
            "direction": "delta >= -0.01",
            "validation_delta": validation_deltas["top3_hit_rate"],
            "test_delta": test_deltas["top3_hit_rate"],
            "limit": -0.01,
            "validation_pass": validation_deltas["top3_hit_rate"] >= -0.01,
            "test_supports": test_deltas["top3_hit_rate"] >= -0.01,
        },
    ]

    for row in criteria:
        row["formal_result"] = "PASS" if row["validation_pass"] else "FAIL"
        row["test_corroboration"] = (
            "supports" if row["test_supports"] else "contradicts"
        )

    return criteria


def decide_verdict(
    safeguards: list[dict[str, Any]],
    validation_deltas: dict[str, float],
    test_deltas: dict[str, float],
    graph_report: dict[str, Any],
) -> tuple[str, str, str]:
    pass_count = sum(bool(row["validation_pass"]) for row in safeguards)
    all_pass = pass_count == len(safeguards)
    primary_fpr_pass = all(
        row["validation_pass"]
        for row in safeguards[:2]
    )

    test_normal_macro_delta = float(
        graph_report["test"]["deltas_b1_minus_a1"]["normal_run_macro_fpr"]
    )
    test_severe_contradiction = (
        test_normal_macro_delta > 0.02
        or test_deltas["graph_fpr"] > 0.02
        or test_deltas["top1_hit_rate"] < -0.02
        or test_deltas["top3_hit_rate"] < -0.01
    )

    if all_pass and not test_severe_contradiction:
        verdict = "retain"
        rationale = (
            "All frozen validation safeguards passed and the blind test did "
            "not materially contradict the expected benefit."
        )
        recommendation = (
            "Retain Chrono-B1 sampling and freeze it before testing a separate "
            "victim-identification head."
        )
    elif primary_fpr_pass and pass_count >= 7 and not test_severe_contradiction:
        verdict = "promising but inconclusive"
        rationale = (
            "The primary normal-run FPR objectives passed, most safeguards "
            "held, and blind-test evidence was not strongly contradictory, "
            "but the result is not sufficiently consistent for retention."
        )
        recommendation = (
            "Keep Chrono-A1 as the production baseline and repeat only the "
            "sampling ablation across additional fixed seeds before changing "
            "the architecture."
        )
    else:
        verdict = "reject"
        rationale = (
            f"Only {pass_count}/{len(safeguards)} frozen validation safeguards "
            "passed. The primary normal-run FPR objectives failed or the blind "
            "test materially contradicted the intended benefit."
        )

        if (
            validation_deltas["graph_fpr"] > 0
            or test_deltas["graph_fpr"] > 0
            or test_normal_macro_delta > 0
        ):
            recommendation = (
                "Return to Chrono-A1 unchanged. The next single-change "
                "experiment should test an alternative graph readout while "
                "freezing the Conv1D encoder, GCN layers, node head, losses, "
                "splits, and evaluation protocol."
            )
        elif (
            validation_deltas["attack_only_node_f1"] < -0.02
            or validation_deltas["attack_only_exact_localization"] < -0.03
        ):
            recommendation = (
                "Return to Chrono-A1 unchanged. The next single-change "
                "experiment should adjust the localization loss only."
            )
        else:
            recommendation = (
                "Return to Chrono-A1 unchanged and perform no further "
                "architecture change until the remaining error clusters are "
                "audited."
            )

    return verdict, rationale, recommendation


def markdown_table(headers: list[str], rows: Iterable[list[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def build_markdown(
    *,
    verdict: str,
    rationale: str,
    recommendation: str,
    comparison_rows: list[dict[str, Any]],
    safeguards: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    disagreement_rows: list[dict[str, Any]],
    graph_report: dict[str, Any],
    thresholds: dict[str, float],
    prerequisite_pass: bool,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> str:
    lines: list[str] = []
    lines.append("# Chrono-B1 Final Post-Training Report")
    lines.append("")
    lines.append(f"## Formal verdict: **{verdict.upper()}**")
    lines.append("")
    lines.append(rationale)
    lines.append("")
    lines.append("## Final retained architecture")
    lines.append("")
    if verdict == "retain":
        lines.append(
            "Chrono-B1 becomes the retained Conv1D-TemporalGCN training configuration."
        )
    else:
        lines.append(
            "Chrono-A1 remains the retained Conv1D-TemporalGCN baseline. "
            "The scenario-balanced sampler is excluded."
        )
    lines.append("")
    lines.append("## Next single-change experiment")
    lines.append("")
    lines.append(recommendation)
    lines.append("")
    lines.append("## Protocol integrity")
    lines.append("")
    lines.append(f"- Prerequisite and alignment checks passed: **{prerequisite_pass}**")
    lines.append("- Thresholds were selected on validation only and transferred unchanged to test.")
    lines.append("- Localization denominators contain attack-positive samples only.")
    lines.append(
        f"- Paired bootstrap: {bootstrap_repetitions} repetitions, seed {bootstrap_seed}."
    )
    lines.append("")
    lines.append("### Frozen thresholds")
    lines.append("")
    lines.append(
        markdown_table(
            ["Model", "Graph threshold", "Node threshold"],
            [
                ["A1", fmt(thresholds["a1_graph"], 3), fmt(thresholds["a1_node"], 3)],
                ["B1", fmt(thresholds["b1_graph"], 3), fmt(thresholds["b1_node"], 3)],
            ],
        )
    )
    lines.append("")
    lines.append("## Overall A1 versus B1")
    lines.append("")
    selected_metrics = {
        "graph_f1",
        "graph_recall",
        "graph_fpr",
        "attack_only_node_f1",
        "attack_only_exact_localization",
        "top1_hit_rate",
        "top3_hit_rate",
        "mean_reciprocal_rank",
    }
    comparison_selected = [
        row for row in comparison_rows if row["metric"] in selected_metrics
    ]
    lines.append(
        markdown_table(
            ["Split", "Metric", "A1", "B1", "B1−A1"],
            [
                [
                    row["split"],
                    row["metric"],
                    fmt(row["a1"]),
                    fmt(row["b1"]),
                    f"{row['absolute_delta_b1_minus_a1']:+.6f}",
                ]
                for row in comparison_selected
            ],
        )
    )
    lines.append("")
    lines.append("## Frozen safeguard table")
    lines.append("")
    lines.append(
        markdown_table(
            [
                "Criterion",
                "Validation delta",
                "Limit",
                "Formal result",
                "Test delta",
                "Test corroboration",
            ],
            [
                [
                    row["criterion"],
                    f"{row['validation_delta']:+.6f}",
                    row["direction"],
                    row["formal_result"],
                    f"{row['test_delta']:+.6f}",
                    row["test_corroboration"],
                ]
                for row in safeguards
            ],
        )
    )
    lines.append("")
    lines.append("## Run-level graph summary")
    lines.append("")
    run_rows = []
    for split_key, split_name in (("validation", "val"), ("test", "test")):
        result = graph_report[split_key]
        for model_key in ("a1", "b1"):
            summary = result[model_key]
            run_rows.append(
                [
                    split_name,
                    model_key.upper(),
                    fmt(summary["normal_run_macro_fpr"]),
                    fmt(summary["worst_normal_run_fpr"]),
                    fmt(summary["attack_run_macro_recall"]),
                    fmt(summary["worst_attack_run_recall"]),
                    fmt(summary["strength20_macro_recall"]),
                    fmt(summary["hard_normal_run_fpr"]),
                ]
            )
    lines.append(
        markdown_table(
            [
                "Split",
                "Model",
                "Normal macro FPR",
                "Worst normal FPR",
                "Attack macro recall",
                "Worst attack recall",
                "S20 recall",
                "Hard-normal FPR",
            ],
            run_rows,
        )
    )
    lines.append("")
    lines.append("## Paired bootstrap confidence intervals")
    lines.append("")
    lines.append(
        markdown_table(
            [
                "Split",
                "Metric",
                "Point delta",
                "95% CI",
                "Unit",
                "CI excludes zero",
            ],
            [
                [
                    row["split"],
                    row["metric"],
                    f"{row['point_delta_b1_minus_a1']:+.6f}",
                    (
                        f"[{row['ci95_lower']:+.6f}, "
                        f"{row['ci95_upper']:+.6f}]"
                    ),
                    row["resampling_unit"],
                    row["ci_excludes_zero"],
                ]
                for row in bootstrap_rows
            ],
        )
    )
    lines.append("")
    lines.append("## Matched disagreement summary")
    lines.append("")
    lines.append(
        markdown_table(
            [
                "Split",
                "Metric",
                "Both correct",
                "Both wrong",
                "B1 fixes",
                "B1 breaks",
                "Net fixes",
            ],
            [
                [
                    row["split"],
                    row["metric"],
                    row["both_correct"],
                    row["both_wrong"],
                    row["B1_fixes_A1"],
                    row["B1_breaks_A1"],
                    row["net_fixed_minus_broken"],
                ]
                for row in disagreement_rows
            ],
        )
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(recommendation)
    lines.append("")
    lines.append(
        "The next experiment must preserve the corrected chronological split and "
        "change only the named intervention."
    )
    lines.append("")
    return "\n".join(lines)


def analyze_split(
    *,
    split_name: str,
    a1: dict[str, np.ndarray],
    b1: dict[str, np.ndarray],
    a1_graph_threshold: float,
    b1_graph_threshold: float,
    a1_node_threshold: float,
    b1_node_threshold: float,
    top_disagreements: int,
) -> dict[str, Any]:
    validation_a1 = validate_prediction_file(f"A1 {split_name}", a1, split_name)
    validation_b1 = validate_prediction_file(f"B1 {split_name}", b1, split_name)
    alignment = validate_alignment(a1, b1, split_name)

    if not all(validation_a1.values()) or not all(validation_b1.values()):
        raise ValueError(
            f"{split_name}: invalid prediction file. "
            f"A1={validation_a1}, B1={validation_b1}"
        )

    a1_graph = graph_sample_arrays(
        a1["y_graph"],
        a1["graph_prob"],
        a1_graph_threshold,
    )
    b1_graph = graph_sample_arrays(
        b1["y_graph"],
        b1["graph_prob"],
        b1_graph_threshold,
    )
    a1_graph_metrics = graph_metrics_from_arrays(a1_graph)
    b1_graph_metrics = graph_metrics_from_arrays(b1_graph)

    attack_indices = np.flatnonzero(np.asarray(a1["y_graph"]).reshape(-1) == 1)
    a1_loc = localization_sample_arrays(
        a1["y_node"][attack_indices],
        a1["node_prob"][attack_indices],
        a1_node_threshold,
    )
    b1_loc = localization_sample_arrays(
        b1["y_node"][attack_indices],
        b1["node_prob"][attack_indices],
        b1_node_threshold,
    )
    a1_loc_overall = aggregate_localization(a1_loc)
    b1_loc_overall = aggregate_localization(b1_loc)

    attack_metadata = build_attack_metadata(
        a1,
        attack_indices,
        a1_loc["attacker_count"],
    )
    group_rows = (
        localization_group_rows(
            split_name,
            "A1",
            a1_node_threshold,
            a1_loc,
            attack_metadata,
        )
        + localization_group_rows(
            split_name,
            "B1",
            b1_node_threshold,
            b1_loc,
            attack_metadata,
        )
    )

    comparison_rows = metric_delta_rows(
        split_name,
        a1_graph_metrics,
        b1_graph_metrics,
        a1_loc_overall,
        b1_loc_overall,
    )

    disagreement_summary, disagreement_samples = select_important_disagreements(
        split_name=split_name,
        data=a1,
        attack_indices=attack_indices,
        a1_graph=a1_graph,
        b1_graph=b1_graph,
        a1_loc=a1_loc,
        b1_loc=b1_loc,
        graph_threshold_a1=a1_graph_threshold,
        graph_threshold_b1=b1_graph_threshold,
        top_n=top_disagreements,
    )

    a1_normal_runs, a1_attack_runs = per_run_graph_values(a1, a1_graph)
    b1_normal_runs, b1_attack_runs = per_run_graph_values(b1, b1_graph)

    point_delta_lookup = {
        row["metric"]: float(row["absolute_delta_b1_minus_a1"])
        for row in comparison_rows
    }
    point_delta_lookup["normal_run_macro_fpr"] = float(
        np.mean(list(b1_normal_runs.values()))
        - np.mean(list(a1_normal_runs.values()))
    )
    point_delta_lookup["attack_run_macro_recall"] = float(
        np.mean(list(b1_attack_runs.values()))
        - np.mean(list(a1_attack_runs.values()))
    )

    return {
        "input_checks": {
            "a1": validation_a1,
            "b1": validation_b1,
            "alignment": alignment,
        },
        "a1_graph_arrays": a1_graph,
        "b1_graph_arrays": b1_graph,
        "a1_localization_arrays": a1_loc,
        "b1_localization_arrays": b1_loc,
        "a1_graph_metrics": a1_graph_metrics,
        "b1_graph_metrics": b1_graph_metrics,
        "a1_localization_overall": a1_loc_overall,
        "b1_localization_overall": b1_loc_overall,
        "localization_group_rows": group_rows,
        "comparison_rows": comparison_rows,
        "disagreement_summary": disagreement_summary,
        "disagreement_samples": disagreement_samples,
        "a1_normal_runs": a1_normal_runs,
        "b1_normal_runs": b1_normal_runs,
        "a1_attack_runs": a1_attack_runs,
        "b1_attack_runs": b1_attack_runs,
        "point_delta_lookup": point_delta_lookup,
        "attack_indices": attack_indices,
    }


def strip_numpy_arrays(result: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "a1_graph_arrays",
        "b1_graph_arrays",
        "a1_localization_arrays",
        "b1_localization_arrays",
        "attack_indices",
    }
    return {key: value for key, value in result.items() if key not in excluded}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1-validation", required=True, type=Path)
    parser.add_argument("--b1-validation", required=True, type=Path)
    parser.add_argument("--a1-test", required=True, type=Path)
    parser.add_argument("--b1-test", required=True, type=Path)
    parser.add_argument("--threshold-report", required=True, type=Path)
    parser.add_argument("--test-transfer-report", required=True, type=Path)
    parser.add_argument("--graph-run-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--top-disagreements", type=int, default=200)
    args = parser.parse_args()

    input_paths = {
        "a1_validation": args.a1_validation.resolve(),
        "b1_validation": args.b1_validation.resolve(),
        "a1_test": args.a1_test.resolve(),
        "b1_test": args.b1_test.resolve(),
        "threshold_report": args.threshold_report.resolve(),
        "test_transfer_report": args.test_transfer_report.resolve(),
        "graph_run_report": args.graph_run_report.resolve(),
    }
    for label, path in input_paths.items():
        if not path.is_file():
            raise SystemExit(f"STOP: missing prerequisite {label}: {path}")

    if args.bootstrap_repetitions <= 0:
        raise SystemExit("STOP: bootstrap repetitions must be positive.")
    if args.top_disagreements <= 0:
        raise SystemExit("STOP: top-disagreements must be positive.")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: output directory exists and is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_report = load_json(input_paths["threshold_report"])
    transfer_report = load_json(input_paths["test_transfer_report"])
    graph_report = load_json(input_paths["graph_run_report"])

    if threshold_report.get("stage") != "B1.11":
        raise SystemExit("STOP: threshold report is not the B1.11 artifact.")
    if transfer_report.get("stage") != "B1.12":
        raise SystemExit("STOP: test-transfer report is not the B1.12 artifact.")
    if graph_report.get("stage") != "B1.13":
        raise SystemExit("STOP: graph-run report is not the B1.13 artifact.")
    if threshold_report.get("data_scope") != "validation only":
        raise SystemExit("STOP: threshold report is not validation-only.")
    if threshold_report.get("test_data_read") is not False:
        raise SystemExit("STOP: threshold selection protocol indicates test access.")
    if transfer_report["protocol"].get("thresholds_reselected_on_test") is not False:
        raise SystemExit("STOP: test thresholds were reportedly reselected.")

    frozen = threshold_report["frozen_thresholds_for_test_transfer"]
    thresholds = {
        "a1_graph": float(frozen["a1_graph"]),
        "a1_node": float(frozen["a1_node"]),
        "b1_graph": float(frozen["b1_graph"]),
        "b1_node": float(frozen["b1_node"]),
    }

    transfer_frozen = transfer_report["frozen_thresholds"]
    for key, value in thresholds.items():
        if not math.isclose(
            value,
            float(transfer_frozen[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SystemExit(
                f"STOP: frozen threshold mismatch for {key}: "
                f"B1.11={value}, B1.12={transfer_frozen[key]}"
            )

    val_a1 = load_npz(input_paths["a1_validation"])
    val_b1 = load_npz(input_paths["b1_validation"])
    test_a1 = load_npz(input_paths["a1_test"])
    test_b1 = load_npz(input_paths["b1_test"])

    validation = analyze_split(
        split_name="val",
        a1=val_a1,
        b1=val_b1,
        a1_graph_threshold=thresholds["a1_graph"],
        b1_graph_threshold=thresholds["b1_graph"],
        a1_node_threshold=thresholds["a1_node"],
        b1_node_threshold=thresholds["b1_node"],
        top_disagreements=args.top_disagreements,
    )
    test = analyze_split(
        split_name="test",
        a1=test_a1,
        b1=test_b1,
        a1_graph_threshold=thresholds["a1_graph"],
        b1_graph_threshold=thresholds["b1_graph"],
        a1_node_threshold=thresholds["a1_node"],
        b1_node_threshold=thresholds["b1_node"],
        top_disagreements=args.top_disagreements,
    )

    prerequisite_pass = all(
        all(section.values())
        for result in (validation, test)
        for section in result["input_checks"].values()
    )

    comparison_rows = validation["comparison_rows"] + test["comparison_rows"]
    validation_deltas = comparison_lookup(comparison_rows, "val")
    test_deltas = comparison_lookup(comparison_rows, "test")

    bootstrap_rows = []
    bootstrap_rows.extend(
        bootstrap_rows_for_split(
            split_name="val",
            graph_truth=validation["a1_graph_arrays"]["truth"],
            a_graph=validation["a1_graph_arrays"],
            b_graph=validation["b1_graph_arrays"],
            a_loc=validation["a1_localization_arrays"],
            b_loc=validation["b1_localization_arrays"],
            a_normal_runs=validation["a1_normal_runs"],
            b_normal_runs=validation["b1_normal_runs"],
            a_attack_runs=validation["a1_attack_runs"],
            b_attack_runs=validation["b1_attack_runs"],
            point_deltas=validation["point_delta_lookup"],
            repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed,
        )
    )
    bootstrap_rows.extend(
        bootstrap_rows_for_split(
            split_name="test",
            graph_truth=test["a1_graph_arrays"]["truth"],
            a_graph=test["a1_graph_arrays"],
            b_graph=test["b1_graph_arrays"],
            a_loc=test["a1_localization_arrays"],
            b_loc=test["b1_localization_arrays"],
            a_normal_runs=test["a1_normal_runs"],
            b_normal_runs=test["b1_normal_runs"],
            a_attack_runs=test["a1_attack_runs"],
            b_attack_runs=test["b1_attack_runs"],
            point_deltas=test["point_delta_lookup"],
            repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed + 1,
        )
    )

    safeguards = safeguard_rows(
        graph_report,
        validation_deltas,
        test_deltas,
    )
    verdict, rationale, recommendation = decide_verdict(
        safeguards,
        validation_deltas,
        test_deltas,
        graph_report,
    )

    localization_group_rows_all = (
        validation["localization_group_rows"]
        + test["localization_group_rows"]
    )
    disagreement_summary_all = (
        validation["disagreement_summary"]
        + test["disagreement_summary"]
    )
    disagreement_samples_all = (
        validation["disagreement_samples"]
        + test["disagreement_samples"]
    )

    write_csv(output_dir / "overall_model_comparison.csv", comparison_rows)
    write_csv(
        output_dir / "localization_grouped_metrics.csv",
        localization_group_rows_all,
    )
    write_csv(
        output_dir / "matched_disagreement_summary.csv",
        disagreement_summary_all,
    )
    write_csv(
        output_dir / "matched_disagreement_samples.csv",
        disagreement_samples_all,
    )
    write_csv(
        output_dir / "bootstrap_confidence_intervals.csv",
        bootstrap_rows,
    )
    write_csv(output_dir / "safeguard_table.csv", safeguards)

    final_report = {
        "stage": "B1.14-B1.17 combined",
        "formal_verdict": verdict,
        "verdict_rationale": rationale,
        "final_retained_architecture": (
            "Chrono-B1 Conv1D-TemporalGCN"
            if verdict == "retain"
            else "Chrono-A1 Conv1D-TemporalGCN"
        ),
        "scenario_balanced_sampler_retained": verdict == "retain",
        "next_experiment_recommendation": recommendation,
        "protocol": {
            "retraining_performed": False,
            "thresholds_selected_on": "validation only",
            "test_threshold_reselection": False,
            "localization_denominator": "attack-positive samples only",
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed_validation": args.bootstrap_seed,
            "bootstrap_seed_test": args.bootstrap_seed + 1,
            "bootstrap_interval": "paired percentile 95%",
            "sample_level_resampling": [
                "graph F1",
                "graph recall",
                "graph FPR",
                "attack-only node F1",
                "exact localization",
                "Top-1 hit rate",
                "Top-3 hit rate",
                "mean reciprocal rank",
            ],
            "run_level_resampling": [
                "normal-run macro FPR",
                "attack-run macro recall",
            ],
        },
        "thresholds": thresholds,
        "prerequisite_checks_pass": prerequisite_pass,
        "input_files": {
            label: {
                "path": str(path),
                "sha256": sha256(path),
            }
            for label, path in input_paths.items()
        },
        "validation": strip_numpy_arrays(validation),
        "test": strip_numpy_arrays(test),
        "overall_model_comparison": comparison_rows,
        "bootstrap_confidence_intervals": bootstrap_rows,
        "safeguards": {
            "formal_basis": "frozen validation criteria",
            "pass_count": sum(
                bool(row["validation_pass"]) for row in safeguards
            ),
            "total_count": len(safeguards),
            "all_pass": all(
                bool(row["validation_pass"]) for row in safeguards
            ),
            "rows": safeguards,
        },
        "matched_disagreement_summary": disagreement_summary_all,
        "hard_normal_run": HARD_NORMAL_RUN,
        "graph_run_scenario_report": graph_report,
    }

    json_path = output_dir / "final_b1_report.json"
    json_path.write_text(
        json.dumps(final_report, indent=2) + "\n",
        encoding="utf-8",
    )

    markdown = build_markdown(
        verdict=verdict,
        rationale=rationale,
        recommendation=recommendation,
        comparison_rows=comparison_rows,
        safeguards=safeguards,
        bootstrap_rows=bootstrap_rows,
        disagreement_rows=disagreement_summary_all,
        graph_report=graph_report,
        thresholds=thresholds,
        prerequisite_pass=prerequisite_pass,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    markdown_path = output_dir / "final_b1_report.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    print("B1.14-B1.17 COMBINED POST-TRAINING ANALYSIS: PASS")
    print(f"prerequisite_checks_pass={prerequisite_pass}")
    print(f"bootstrap_repetitions={args.bootstrap_repetitions}")
    print(f"bootstrap_seed_validation={args.bootstrap_seed}")
    print(f"bootstrap_seed_test={args.bootstrap_seed + 1}")

    print("\nVALIDATION KEY DELTAS (B1 - A1)")
    for key in (
        "graph_f1",
        "graph_recall",
        "graph_fpr",
        "attack_only_node_f1",
        "attack_only_exact_localization",
        "top1_hit_rate",
        "top3_hit_rate",
        "mean_reciprocal_rank",
    ):
        print(f"{key}={validation_deltas[key]:+.6f}")

    print("\nTEST KEY DELTAS (B1 - A1)")
    for key in (
        "graph_f1",
        "graph_recall",
        "graph_fpr",
        "attack_only_node_f1",
        "attack_only_exact_localization",
        "top1_hit_rate",
        "top3_hit_rate",
        "mean_reciprocal_rank",
    ):
        print(f"{key}={test_deltas[key]:+.6f}")

    print("\nFROZEN SAFEGUARDS")
    for row in safeguards:
        print(
            f"{row['formal_result']} {row['criterion']} "
            f"validation_delta={row['validation_delta']:+.6f} "
            f"test_delta={row['test_delta']:+.6f}"
        )

    print("\nMATCHED DISAGREEMENTS")
    for row in disagreement_summary_all:
        print(
            f"{row['split']} {row['metric']}: "
            f"fixes={row['B1_fixes_A1']} "
            f"breaks={row['B1_breaks_A1']} "
            f"net={row['net_fixed_minus_broken']:+d}"
        )

    print("\nFORMAL B1 VERDICT")
    print(f"verdict={verdict}")
    print(
        "retained_architecture="
        + (
            "Chrono-B1 Conv1D-TemporalGCN"
            if verdict == "retain"
            else "Chrono-A1 Conv1D-TemporalGCN"
        )
    )
    print(f"scenario_balanced_sampler_retained={verdict == 'retain'}")
    print(f"next_experiment={recommendation}")
    print(f"json_report={json_path}")
    print(f"markdown_report={markdown_path}")


if __name__ == "__main__":
    main()
