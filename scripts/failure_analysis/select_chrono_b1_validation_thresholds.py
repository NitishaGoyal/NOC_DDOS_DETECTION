#!/usr/bin/env python3
"""
Chrono-B1 Stage B1.11: validation-only threshold selection and localization ranking.

Primary graph threshold rule, applied independently to A1 and B1:
1. maximize graph F1;
2. among ties, minimize FPR;
3. then maximize recall;
4. then choose the threshold closest to 0.5;
5. then choose the lower threshold.

Primary node threshold rule, applied independently to A1 and B1 on
attack-positive validation samples only:
1. maximize micro node F1;
2. among ties, maximize exact localization;
3. then maximize precision;
4. then maximize recall;
5. then choose the threshold closest to 0.5;
6. then choose the lower threshold.

The threshold grid is fixed at 0.005, 0.010, ..., 0.995.
Top-k and rank metrics are threshold-independent and use attack-positive
validation samples only.

No test data is read.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


THRESHOLDS = np.round(np.arange(0.005, 1.0, 0.005), 3)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = (np.asarray(y_prob).reshape(-1) >= threshold).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    fpr = safe_div(fp, fp + tn)
    tnr = safe_div(tn, tn + fp)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tnr": tnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def node_metrics_attack_only(
    y_graph: np.ndarray,
    y_node: np.ndarray,
    node_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    attack_mask = np.asarray(y_graph).reshape(-1) == 1
    truth = np.asarray(y_node[attack_mask], dtype=np.int64)
    prob = np.asarray(node_prob[attack_mask])
    pred = (prob >= threshold).astype(np.int64)

    flat_truth = truth.reshape(-1)
    flat_pred = pred.reshape(-1)

    tp = int(np.sum((flat_truth == 1) & (flat_pred == 1)))
    tn = int(np.sum((flat_truth == 0) & (flat_pred == 0)))
    fp = int(np.sum((flat_truth == 0) & (flat_pred == 1)))
    fn = int(np.sum((flat_truth == 1) & (flat_pred == 0)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    exact = float(np.mean(np.all(pred == truth, axis=1))) if len(truth) else 0.0

    predicted_count = pred.sum(axis=1)
    true_count = truth.sum(axis=1)

    return {
        "threshold": float(threshold),
        "attack_sample_count": int(len(truth)),
        "positive_node_count": int(flat_truth.sum()),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "exact_localization": exact,
        "mean_predicted_nodes": float(np.mean(predicted_count)) if len(pred) else 0.0,
        "mean_true_nodes": float(np.mean(true_count)) if len(truth) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def choose_graph(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row["f1"],
            -row["fpr"],
            row["recall"],
            -abs(row["threshold"] - 0.5),
            -row["threshold"],
        ),
    )


def choose_node(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row["micro_f1"],
            row["exact_localization"],
            row["micro_precision"],
            row["micro_recall"],
            -abs(row["threshold"] - 0.5),
            -row["threshold"],
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ranking_metrics(y_graph: np.ndarray, y_node: np.ndarray, node_prob: np.ndarray) -> dict[str, Any]:
    attack_mask = np.asarray(y_graph).reshape(-1) == 1
    truth = np.asarray(y_node[attack_mask], dtype=np.int64)
    prob = np.asarray(node_prob[attack_mask], dtype=np.float64)

    if len(truth) == 0:
        raise ValueError("No attack-positive validation samples available.")

    attacker_counts = truth.sum(axis=1).astype(np.int64)
    if np.any(attacker_counts <= 0):
        bad = int(np.sum(attacker_counts <= 0))
        raise ValueError(f"{bad} attack-positive samples have no positive node label.")

    # Stable descending ordering makes ties deterministic by node index.
    order = np.argsort(-prob, axis=1, kind="stable")
    inverse_rank = np.empty_like(order)
    row_ids = np.arange(len(order))[:, None]
    inverse_rank[row_ids, order] = np.arange(1, order.shape[1] + 1)[None, :]

    true_ranks_all: list[int] = []
    best_ranks: list[int] = []
    top_hits = {1: [], 2: [], 3: []}
    top_recalls = {1: [], 2: [], 3: []}
    exact_top_m: list[bool] = []

    per_sample: list[dict[str, Any]] = []

    for i in range(len(truth)):
        true_nodes = np.flatnonzero(truth[i] == 1)
        ranks = inverse_rank[i, true_nodes].astype(np.int64)
        best_rank = int(ranks.min())
        true_ranks_all.extend(ranks.tolist())
        best_ranks.append(best_rank)

        sample = {
            "sample_offset": i,
            "attacker_count": int(len(true_nodes)),
            "best_attacker_rank": best_rank,
            "mean_attacker_rank": float(np.mean(ranks)),
            "reciprocal_best_rank": 1.0 / best_rank,
        }

        for k in (1, 2, 3):
            selected = set(order[i, :k].tolist())
            hit = bool(any(int(node) in selected for node in true_nodes))
            recall = float(sum(int(node) in selected for node in true_nodes) / len(true_nodes))
            top_hits[k].append(hit)
            top_recalls[k].append(recall)
            sample[f"top{k}_hit"] = hit
            sample[f"top{k}_attacker_recall"] = recall

        m = len(true_nodes)
        exact = set(order[i, :m].tolist()) == set(true_nodes.tolist())
        exact_top_m.append(exact)
        sample["exact_top_m_set"] = bool(exact)
        per_sample.append(sample)

    overall = {
        "attack_sample_count": int(len(truth)),
        "attacker_node_count": int(attacker_counts.sum()),
        "top1_hit_rate": float(np.mean(top_hits[1])),
        "top2_hit_rate": float(np.mean(top_hits[2])),
        "top3_hit_rate": float(np.mean(top_hits[3])),
        "top1_attacker_recall": float(np.mean(top_recalls[1])),
        "top2_attacker_recall": float(np.mean(top_recalls[2])),
        "top3_attacker_recall": float(np.mean(top_recalls[3])),
        "mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in best_ranks])),
        "mean_best_attacker_rank": float(np.mean(best_ranks)),
        "median_best_attacker_rank": float(np.median(best_ranks)),
        "mean_all_attacker_rank": float(np.mean(true_ranks_all)),
        "median_all_attacker_rank": float(np.median(true_ranks_all)),
        "exact_top_m_set_accuracy": float(np.mean(exact_top_m)),
    }

    by_attacker_count: dict[str, dict[str, Any]] = {}
    for count in sorted(np.unique(attacker_counts).tolist()):
        mask = attacker_counts == count
        indices = np.flatnonzero(mask)
        rows = [per_sample[int(i)] for i in indices]
        by_attacker_count[str(int(count))] = {
            "sample_count": int(len(rows)),
            "top1_hit_rate": float(np.mean([row["top1_hit"] for row in rows])),
            "top2_hit_rate": float(np.mean([row["top2_hit"] for row in rows])),
            "top3_hit_rate": float(np.mean([row["top3_hit"] for row in rows])),
            "top1_attacker_recall": float(
                np.mean([row["top1_attacker_recall"] for row in rows])
            ),
            "top2_attacker_recall": float(
                np.mean([row["top2_attacker_recall"] for row in rows])
            ),
            "top3_attacker_recall": float(
                np.mean([row["top3_attacker_recall"] for row in rows])
            ),
            "mean_reciprocal_rank": float(
                np.mean([row["reciprocal_best_rank"] for row in rows])
            ),
            "mean_best_attacker_rank": float(
                np.mean([row["best_attacker_rank"] for row in rows])
            ),
            "mean_all_attacker_rank": float(
                np.mean([row["mean_attacker_rank"] for row in rows])
            ),
            "exact_top_m_set_accuracy": float(
                np.mean([row["exact_top_m_set"] for row in rows])
            ),
        }

    return {
        "denominator": "attack-positive validation samples only",
        "overall": overall,
        "by_attacker_count": by_attacker_count,
    }


def thresholded_node_by_attacker_count(
    y_graph: np.ndarray,
    y_node: np.ndarray,
    node_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    attack_mask = np.asarray(y_graph).reshape(-1) == 1
    truth = np.asarray(y_node[attack_mask], dtype=np.int64)
    prob = np.asarray(node_prob[attack_mask])
    counts = truth.sum(axis=1).astype(np.int64)

    result: dict[str, Any] = {}
    for count in sorted(np.unique(counts).tolist()):
        group_mask = counts == count
        group_truth = truth[group_mask]
        group_prob = prob[group_mask]
        group_graph = np.ones(len(group_truth), dtype=np.int64)
        result[str(int(count))] = node_metrics_attack_only(
            group_graph,
            group_truth,
            group_prob,
            threshold,
        )
    return result


def validate_input(label: str, data: dict[str, np.ndarray]) -> None:
    split = str(np.asarray(data["split"]).item())
    if split != "val":
        raise ValueError(f"{label}: expected split='val', got {split!r}")
    if not np.all(np.asarray(data["dataset_split"]).astype(str) == "val"):
        raise ValueError(f"{label}: prediction file contains non-validation rows.")
    for key in ("graph_prob", "node_prob"):
        values = np.asarray(data[key])
        if not np.isfinite(values).all():
            raise ValueError(f"{label}: {key} contains non-finite values.")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"{label}: {key} lies outside [0, 1].")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as d:
        return {key: np.asarray(d[key]) for key in d.files}


def evaluate_model(label: str, data: dict[str, np.ndarray], output_dir: Path) -> dict[str, Any]:
    graph_rows = [
        binary_metrics(data["y_graph"], data["graph_prob"], float(threshold))
        for threshold in THRESHOLDS
    ]
    node_rows = [
        node_metrics_attack_only(
            data["y_graph"],
            data["y_node"],
            data["node_prob"],
            float(threshold),
        )
        for threshold in THRESHOLDS
    ]

    graph_selected = choose_graph(graph_rows)
    node_selected = choose_node(node_rows)

    graph_default = binary_metrics(
        data["y_graph"], data["graph_prob"], 0.5
    )
    node_default = node_metrics_attack_only(
        data["y_graph"], data["y_node"], data["node_prob"], 0.5
    )

    ranking = ranking_metrics(
        data["y_graph"], data["y_node"], data["node_prob"]
    )
    node_by_count = thresholded_node_by_attacker_count(
        data["y_graph"],
        data["y_node"],
        data["node_prob"],
        node_selected["threshold"],
    )

    lower = label.lower()
    write_csv(output_dir / f"{lower}_graph_threshold_sweep.csv", graph_rows)
    write_csv(output_dir / f"{lower}_node_threshold_sweep.csv", node_rows)

    return {
        "graph_threshold": {
            "selected": graph_selected["threshold"],
            "selected_metrics": graph_selected,
            "default_0_5_metrics": graph_default,
        },
        "node_threshold": {
            "selected": node_selected["threshold"],
            "selected_metrics_attack_only": node_selected,
            "default_0_5_metrics_attack_only": node_default,
            "selected_metrics_by_attacker_count": node_by_count,
        },
        "threshold_independent_localization_ranking": ranking,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1-predictions", required=True, type=Path)
    parser.add_argument("--b1-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    a1_path = args.a1_predictions.resolve()
    b1_path = args.b1_predictions.resolve()
    output_dir = args.output_dir.resolve()

    if not a1_path.is_file() or not b1_path.is_file():
        raise SystemExit("STOP: A1 or B1 validation prediction file is missing.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: output directory exists and is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    a1 = load_npz(a1_path)
    b1 = load_npz(b1_path)
    validate_input("A1", a1)
    validate_input("B1", b1)

    for key in ("real_index", "y_graph", "y_node"):
        if not np.array_equal(a1[key], b1[key]):
            raise SystemExit(f"STOP: A1/B1 alignment mismatch in {key}")

    a1_result = evaluate_model("A1", a1, output_dir)
    b1_result = evaluate_model("B1", b1, output_dir)

    report = {
        "stage": "B1.11",
        "data_scope": "validation only",
        "test_data_read": False,
        "threshold_grid": {
            "minimum": 0.005,
            "maximum": 0.995,
            "step": 0.005,
            "count": int(len(THRESHOLDS)),
        },
        "selection_protocol": {
            "graph": [
                "maximize graph F1",
                "minimize FPR",
                "maximize recall",
                "threshold closest to 0.5",
                "lower threshold",
            ],
            "node": [
                "attack-positive validation samples only",
                "maximize micro node F1",
                "maximize exact localization",
                "maximize precision",
                "maximize recall",
                "threshold closest to 0.5",
                "lower threshold",
            ],
            "selection_applied_independently_to_each_model": True,
        },
        "localization_denominator": (
            "All node-threshold and ranking metrics use graph-label-positive "
            "validation samples only. Normal samples are excluded."
        ),
        "a1": a1_result,
        "b1": b1_result,
        "validation_deltas_b1_minus_a1": {
            "selected_graph_f1": (
                b1_result["graph_threshold"]["selected_metrics"]["f1"]
                - a1_result["graph_threshold"]["selected_metrics"]["f1"]
            ),
            "selected_graph_recall": (
                b1_result["graph_threshold"]["selected_metrics"]["recall"]
                - a1_result["graph_threshold"]["selected_metrics"]["recall"]
            ),
            "selected_graph_fpr": (
                b1_result["graph_threshold"]["selected_metrics"]["fpr"]
                - a1_result["graph_threshold"]["selected_metrics"]["fpr"]
            ),
            "selected_attack_only_node_f1": (
                b1_result["node_threshold"]["selected_metrics_attack_only"]["micro_f1"]
                - a1_result["node_threshold"]["selected_metrics_attack_only"]["micro_f1"]
            ),
            "selected_attack_only_exact_localization": (
                b1_result["node_threshold"]["selected_metrics_attack_only"]["exact_localization"]
                - a1_result["node_threshold"]["selected_metrics_attack_only"]["exact_localization"]
            ),
            "top1_hit_rate": (
                b1_result["threshold_independent_localization_ranking"]["overall"]["top1_hit_rate"]
                - a1_result["threshold_independent_localization_ranking"]["overall"]["top1_hit_rate"]
            ),
            "top3_hit_rate": (
                b1_result["threshold_independent_localization_ranking"]["overall"]["top3_hit_rate"]
                - a1_result["threshold_independent_localization_ranking"]["overall"]["top3_hit_rate"]
            ),
            "mean_reciprocal_rank": (
                b1_result["threshold_independent_localization_ranking"]["overall"]["mean_reciprocal_rank"]
                - a1_result["threshold_independent_localization_ranking"]["overall"]["mean_reciprocal_rank"]
            ),
        },
        "frozen_thresholds_for_test_transfer": {
            "a1_graph": a1_result["graph_threshold"]["selected"],
            "a1_node": a1_result["node_threshold"]["selected"],
            "b1_graph": b1_result["graph_threshold"]["selected"],
            "b1_node": b1_result["node_threshold"]["selected"],
        },
    }

    report_path = output_dir / "validation_threshold_selection.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("B1.11 VALIDATION-ONLY THRESHOLD SELECTION: PASS")
    for label, result in (("A1", a1_result), ("B1", b1_result)):
        graph = result["graph_threshold"]["selected_metrics"]
        node = result["node_threshold"]["selected_metrics_attack_only"]
        rank = result["threshold_independent_localization_ranking"]["overall"]
        print(f"\n{label}")
        print(f"graph_threshold={graph['threshold']:.3f}")
        print(f"graph_f1={graph['f1']:.6f}")
        print(f"graph_recall={graph['recall']:.6f}")
        print(f"graph_fpr={graph['fpr']:.6f}")
        print(f"node_threshold={node['threshold']:.3f}")
        print(f"attack_only_node_f1={node['micro_f1']:.6f}")
        print(f"attack_only_exact_localization={node['exact_localization']:.6f}")
        print(f"top1_hit_rate={rank['top1_hit_rate']:.6f}")
        print(f"top2_hit_rate={rank['top2_hit_rate']:.6f}")
        print(f"top3_hit_rate={rank['top3_hit_rate']:.6f}")
        print(f"top1_attacker_recall={rank['top1_attacker_recall']:.6f}")
        print(f"top2_attacker_recall={rank['top2_attacker_recall']:.6f}")
        print(f"top3_attacker_recall={rank['top3_attacker_recall']:.6f}")
        print(f"mean_reciprocal_rank={rank['mean_reciprocal_rank']:.6f}")
        print(f"mean_best_attacker_rank={rank['mean_best_attacker_rank']:.6f}")
        print(f"mean_all_attacker_rank={rank['mean_all_attacker_rank']:.6f}")
        print(f"exact_top_m_set_accuracy={rank['exact_top_m_set_accuracy']:.6f}")

    print("\nVALIDATION DELTAS (B1 - A1)")
    for key, value in report["validation_deltas_b1_minus_a1"].items():
        print(f"{key}={value:+.6f}")

    print("\nFROZEN THRESHOLDS FOR TEST TRANSFER")
    for key, value in report["frozen_thresholds_for_test_transfer"].items():
        print(f"{key}={value:.3f}")
    print(f"output={report_path}")


if __name__ == "__main__":
    main()
