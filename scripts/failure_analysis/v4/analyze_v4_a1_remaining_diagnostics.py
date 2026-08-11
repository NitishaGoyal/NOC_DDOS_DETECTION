#!/usr/bin/env python3
"""
Final post-hoc V4-A1 diagnostics for set decoding and architecture decisions.

This script reuses frozen validation/test prediction archives. It does not train
or run the neural model, retune anything on test, or modify the dataset.

It depends on the previously installed sibling module:
    analyze_v4_a1_failures.py

Validation-only fitting:
- multinomial logistic attacker-count decoder (0..4)
- per-topology Platt calibrators for node scores
- per-predicted-count node thresholds

Transferred unchanged to the development-comparison test split.
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCRIPT_VERSION = "1.0.0"
NUM_ROUTERS = 16
CORNERS = np.asarray([0, 3, 12, 15], dtype=np.int64)
EDGES = np.asarray([1, 2, 4, 7, 8, 11, 13, 14], dtype=np.int64)
INTERIORS = np.asarray([5, 6, 9, 10], dtype=np.int64)
TOPOLOGY_GROUPS = {"corner": CORNERS, "edge": EDGES, "interior": INTERIORS}


def require_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required. Activate the project venv and verify "
            "`python -c \"import sklearn\"`."
        ) from exc
    return LogisticRegression, StandardScaler


def load_base_module(path: Path):
    spec = importlib.util.spec_from_file_location("v4_a1_base_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base analysis module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        seen: set[str] = set()
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fields = ordered or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def clipped_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def entropy_binary(p: np.ndarray, axis: int = 1) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return -np.sum(q * np.log(q) + (1.0 - q) * np.log(1.0 - q), axis=axis)


def topology_of_router(router: int) -> str:
    if router in {0, 3, 12, 15}:
        return "corner"
    if router in {5, 6, 9, 10}:
        return "interior"
    return "edge"


def manhattan(a: int, b: int) -> int:
    return abs(a // 4 - b // 4) + abs(a % 4 - b % 4)


def count_feature_matrix(pred: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    graph = np.asarray(pred["graph_prob"], dtype=np.float64)
    node = np.asarray(pred["node_prob"], dtype=np.float64)
    sorted_scores = np.sort(node, axis=1)[:, ::-1]
    features: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        features.append(values)
        names.append(name)

    add("graph_prob", graph)
    add("graph_logit", clipped_logit(graph))
    add("node_sum", node.sum(axis=1))
    add("node_mean", node.mean(axis=1))
    add("node_max", node.max(axis=1))
    add("node_min", node.min(axis=1))
    add("node_std", node.std(axis=1))
    add("node_binary_entropy", entropy_binary(node))

    for rank in range(8):
        add(f"score_rank_{rank + 1}", sorted_scores[:, rank])
    for rank in range(7):
        add(f"gap_rank_{rank + 1}_{rank + 2}", sorted_scores[:, rank] - sorted_scores[:, rank + 1])
    for threshold in np.arange(0.20, 0.81, 0.10):
        add(f"count_ge_{threshold:.2f}", (node >= threshold).sum(axis=1))

    for topo_name, routers in TOPOLOGY_GROUPS.items():
        values = node[:, routers]
        add(f"{topo_name}_sum", values.sum(axis=1))
        add(f"{topo_name}_mean", values.mean(axis=1))
        add(f"{topo_name}_max", values.max(axis=1))
        add(f"{topo_name}_std", values.std(axis=1))

    return np.column_stack(features), names


def stratified_cap_indices(labels: np.ndarray, cap: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.size
    if cap <= 0 or n <= cap:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    allocations: dict[int, int] = {}
    remaining = cap
    for cls, count in zip(classes.tolist(), counts.tolist()):
        alloc = max(1, int(round(cap * count / n)))
        alloc = min(alloc, count)
        allocations[int(cls)] = alloc
        remaining -= alloc
    # Reconcile rounding while preserving at least one per class.
    while remaining > 0:
        candidates = [int(c) for c, count in zip(classes, counts) if allocations[int(c)] < int(count)]
        if not candidates:
            break
        for cls in candidates:
            if remaining <= 0:
                break
            allocations[cls] += 1
            remaining -= 1
    while remaining < 0:
        candidates = [int(c) for c in classes if allocations[int(c)] > 1]
        if not candidates:
            break
        for cls in candidates:
            if remaining >= 0:
                break
            allocations[cls] -= 1
            remaining += 1
    selected: list[np.ndarray] = []
    for cls in classes.tolist():
        idx = np.flatnonzero(labels == cls)
        take = allocations[int(cls)]
        selected.append(rng.choice(idx, size=take, replace=False))
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out.astype(np.int64)


def fit_count_decoder(
    validation_pred: Mapping[str, np.ndarray], train_cap: int, seed: int
) -> tuple[Any, Any, list[str], dict[str, Any]]:
    LogisticRegression, StandardScaler = require_sklearn()
    x, names = count_feature_matrix(validation_pred)
    y = np.asarray(validation_pred["y_node"] >= 0.5).sum(axis=1).astype(np.int64)
    y = np.clip(y, 0, 4)
    idx = stratified_cap_indices(y, train_cap, seed)
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x[idx])
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(x_fit, y[idx])
    info = {
        "classes": model.classes_.astype(int).tolist(),
        "feature_names": names,
        "validation_samples_total": int(y.size),
        "validation_samples_used": int(idx.size),
        "class_counts_used": {str(int(k)): int(v) for k, v in Counter(y[idx].tolist()).items()},
        "iterations": np.asarray(model.n_iter_).astype(int).tolist(),
        "solver": "lbfgs",
        "class_weight": "balanced",
        "test_used_for_fit": False,
    }
    return scaler, model, names, info


def predict_counts(pred: Mapping[str, np.ndarray], scaler: Any, model: Any) -> tuple[np.ndarray, np.ndarray]:
    x, _ = count_feature_matrix(pred)
    probs = model.predict_proba(scaler.transform(x))
    labels = np.asarray(model.classes_, dtype=np.int64)
    counts = labels[np.argmax(probs, axis=1)]
    return counts.astype(np.int64), probs


def largest_gap_counts(pred: Mapping[str, np.ndarray], graph_threshold: float) -> np.ndarray:
    node = np.asarray(pred["node_prob"], dtype=np.float64)
    graph = np.asarray(pred["graph_prob"], dtype=np.float64)
    sorted_scores = np.sort(node, axis=1)[:, ::-1]
    gaps = sorted_scores[:, :4] - sorted_scores[:, 1:5]
    counts = np.argmax(gaps, axis=1).astype(np.int64) + 1
    counts[graph < graph_threshold] = 0
    return counts


def fit_topology_calibrators(
    validation_pred: Mapping[str, np.ndarray], node_cap: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    LogisticRegression, _ = require_sklearn()
    node_prob = np.asarray(validation_pred["node_prob"], dtype=np.float64)
    truth = np.asarray(validation_pred["y_node"] >= 0.5, dtype=np.int64)
    rng = np.random.default_rng(seed)
    calibrators: dict[str, Any] = {}
    info: dict[str, Any] = {}
    for topo_name, routers in TOPOLOGY_GROUPS.items():
        x = clipped_logit(node_prob[:, routers]).reshape(-1, 1)
        y = truth[:, routers].reshape(-1)
        if node_cap > 0 and y.size > node_cap:
            pos = np.flatnonzero(y == 1)
            neg = np.flatnonzero(y == 0)
            target_pos = min(len(pos), max(1, node_cap // 2))
            target_neg = min(len(neg), node_cap - target_pos)
            idx = np.concatenate([
                rng.choice(pos, size=target_pos, replace=False),
                rng.choice(neg, size=target_neg, replace=False),
            ])
            rng.shuffle(idx)
        else:
            idx = np.arange(y.size)
        unique_classes = np.unique(y[idx])
        if unique_classes.size < 2:
            calibrators[topo_name] = None
            info[topo_name] = {
                "routers": routers.astype(int).tolist(),
                "samples_total": int(y.size),
                "samples_used": int(idx.size),
                "positive_fraction_used": float(np.mean(y[idx])),
                "status": "identity_fallback_single_class",
                "test_used_for_fit": False,
            }
            continue
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(x[idx], y[idx])
        calibrators[topo_name] = model
        info[topo_name] = {
            "routers": routers.astype(int).tolist(),
            "samples_total": int(y.size),
            "samples_used": int(idx.size),
            "positive_fraction_used": float(np.mean(y[idx])),
            "coefficient": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]),
            "status": "fitted",
            "test_used_for_fit": False,
        }
    return calibrators, info


def apply_topology_calibration(pred: Mapping[str, np.ndarray], calibrators: Mapping[str, Any]) -> np.ndarray:
    node_prob = np.asarray(pred["node_prob"], dtype=np.float64)
    out = np.zeros_like(node_prob, dtype=np.float64)
    for topo_name, routers in TOPOLOGY_GROUPS.items():
        model = calibrators[topo_name]
        if model is None:
            out[:, routers] = node_prob[:, routers]
            continue
        x = clipped_logit(node_prob[:, routers]).reshape(-1, 1)
        calibrated = model.predict_proba(x)[:, 1].reshape(node_prob.shape[0], len(routers))
        out[:, routers] = calibrated
    return out


def topk_prediction(scores: np.ndarray, counts: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    out = np.zeros_like(scores, dtype=bool)
    order = np.argsort(-scores, axis=1, kind="stable")
    for k in range(1, NUM_ROUTERS + 1):
        mask = counts == k
        if np.any(mask):
            rows = np.flatnonzero(mask)
            out[rows[:, None], order[mask, :k]] = True
    return out


def attack_only_metrics(y_node: np.ndarray, pred_node: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y_node >= 0.5, dtype=bool)
    pred = np.asarray(pred_node, dtype=bool)
    mask = truth.any(axis=1)
    if not np.any(mask):
        raise RuntimeError("No attack samples available")
    t = truth[mask]
    p = pred[mask]
    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    true_count = t.sum(axis=1)
    pred_count = p.sum(axis=1)
    return {
        "attack_sample_count": int(mask.sum()),
        "node_precision": precision,
        "node_recall": recall,
        "node_f1": f1,
        "exact_localization": float(np.mean(np.all(t == p, axis=1))),
        "count_accuracy": float(np.mean(true_count == pred_count)),
        "count_mae": float(np.mean(np.abs(true_count - pred_count))),
        "empty_prediction_fraction": float(np.mean(pred_count == 0)),
        "mean_true_count": float(np.mean(true_count)),
        "mean_predicted_count": float(np.mean(pred_count)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def overall_exact(y_node: np.ndarray, pred_node: np.ndarray) -> float:
    truth = np.asarray(y_node >= 0.5, dtype=bool)
    return float(np.mean(np.all(truth == np.asarray(pred_node, dtype=bool), axis=1)))


def method_row(name: str, pred: Mapping[str, np.ndarray], node_prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(pred["y_node"] >= 0.5, dtype=bool)
    row = {
        "method": name,
        "sample_count": int(truth.shape[0]),
        "overall_exact_localization": overall_exact(pred["y_node"], node_prediction),
    }
    row.update({f"attack_{k}": v for k, v in attack_only_metrics(pred["y_node"], node_prediction).items()})
    return row


def exact_error_type(truth: np.ndarray, pred: np.ndarray) -> str:
    t = set(np.flatnonzero(truth).tolist())
    p = set(np.flatnonzero(pred).tolist())
    if p == t:
        return "exact"
    if not p:
        return "empty_prediction"
    if not (p & t):
        return "no_true_attacker_found"
    if t < p:
        return "all_true_plus_extras"
    if p < t:
        return "strict_subset_only"
    return "mixed_misses_and_extras"


def exact_decomposition(y_node: np.ndarray, pred_node: np.ndarray) -> list[dict[str, Any]]:
    truth = np.asarray(y_node >= 0.5, dtype=bool)
    pred = np.asarray(pred_node, dtype=bool)
    mask = truth.any(axis=1)
    labels = [exact_error_type(t, p) for t, p in zip(truth[mask], pred[mask])]
    counts = Counter(labels)
    total = len(labels)
    return [
        {"error_type": key, "sample_count": int(count), "fraction": safe_div(count, total)}
        for key, count in sorted(counts.items())
    ]


def count_confusion(y_node: np.ndarray, pred_counts: np.ndarray) -> list[dict[str, Any]]:
    truth_counts = np.asarray(y_node >= 0.5).sum(axis=1).astype(np.int64)
    rows: list[dict[str, Any]] = []
    for t in range(5):
        mask = truth_counts == t
        denom = int(mask.sum())
        for p in range(5):
            count = int(np.sum(mask & (pred_counts == p)))
            rows.append({
                "true_count": t,
                "predicted_count": p,
                "sample_count": count,
                "fraction_within_true_count": safe_div(count, denom),
            })
    return rows


def fit_predicted_count_thresholds(
    validation_pred: Mapping[str, np.ndarray], predicted_counts: np.ndarray,
    threshold_start: float, threshold_end: float, threshold_step: float,
) -> dict[int, float]:
    truth = np.asarray(validation_pred["y_node"] >= 0.5, dtype=bool)
    scores = np.asarray(validation_pred["node_prob"], dtype=np.float64)
    thresholds = np.arange(threshold_start, threshold_end + threshold_step / 2, threshold_step)
    selected: dict[int, float] = {}
    for count in range(5):
        mask = predicted_counts == count
        if not np.any(mask):
            selected[count] = 1.0 if count == 0 else 0.5
            continue
        best_key: tuple[float, float, float] | None = None
        best_thr = 0.5
        for thr in thresholds:
            p = scores[mask] >= thr
            exact = float(np.mean(np.all(truth[mask] == p, axis=1)))
            pred_count = p.sum(axis=1)
            count_acc = float(np.mean(pred_count == truth[mask].sum(axis=1)))
            key = (exact, count_acc, -abs(float(thr) - 0.5))
            if best_key is None or key > best_key:
                best_key = key
                best_thr = float(thr)
        selected[count] = best_thr
    return selected


def apply_predicted_count_thresholds(
    scores: np.ndarray, predicted_counts: np.ndarray, thresholds: Mapping[int, float]
) -> np.ndarray:
    out = np.zeros_like(scores, dtype=bool)
    for count, threshold in thresholds.items():
        mask = predicted_counts == int(count)
        if np.any(mask):
            out[mask] = scores[mask] >= float(threshold)
    return out


def graph_node_consistency(
    pred: Mapping[str, np.ndarray], node_prediction: np.ndarray, graph_threshold: float
) -> list[dict[str, Any]]:
    graph_truth = np.asarray(pred["y_graph"] >= 0.5, dtype=bool)
    graph_pred = np.asarray(pred["graph_prob"] >= graph_threshold, dtype=bool)
    node_nonempty = np.asarray(node_prediction, dtype=bool).any(axis=1)
    rows: list[dict[str, Any]] = []
    total = graph_truth.size
    for truth_label in [0, 1]:
        truth_mask = graph_truth == bool(truth_label)
        denom = int(truth_mask.sum())
        for graph_label in [0, 1]:
            for node_label in [0, 1]:
                mask = truth_mask & (graph_pred == bool(graph_label)) & (node_nonempty == bool(node_label))
                count = int(mask.sum())
                rows.append({
                    "true_graph": truth_label,
                    "predicted_graph": graph_label,
                    "node_set_nonempty": node_label,
                    "sample_count": count,
                    "fraction_of_truth_class": safe_div(count, denom),
                    "fraction_overall": safe_div(count, total),
                })
    return rows


def attacker_rank_rows(pred: Mapping[str, np.ndarray], records: Sequence[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scores = np.asarray(pred["node_prob"], dtype=np.float64)
    truth = np.asarray(pred["y_node"] >= 0.5, dtype=bool)
    order = np.argsort(-scores, axis=1, kind="stable")
    inverse = np.empty_like(order)
    rows_index = np.arange(scores.shape[0])[:, None]
    inverse[rows_index, order] = np.arange(NUM_ROUTERS)[None, :]
    true_counts = truth.sum(axis=1).astype(np.int64)
    run_index = np.asarray(pred["run_index"], dtype=np.int64)

    detailed: list[dict[str, Any]] = []
    summary_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for pos in np.flatnonzero(true_counts > 0):
        record = records[int(run_index[pos])]
        ranks = (inverse[pos, np.flatnonzero(truth[pos])] + 1).astype(int)
        for router, rank in zip(np.flatnonzero(truth[pos]).tolist(), ranks.tolist()):
            detailed.append({
                "sample_position": int(pos),
                "global_index": int(pred["global_index"][pos]),
                "run_id": record.run_id,
                "true_count": int(true_counts[pos]),
                "router": int(router),
                "topology": topology_of_router(int(router)),
                "rank": int(rank),
                "score": float(scores[pos, router]),
                "profile": record.profile,
                "attack_kind": record.attack_kind,
                "strength": record.strength,
            })
            summary_groups[(int(true_counts[pos]), topology_of_router(int(router)))].append(int(rank))

    summary: list[dict[str, Any]] = []
    for (count, topology), ranks in sorted(summary_groups.items()):
        arr = np.asarray(ranks, dtype=np.float64)
        summary.append({
            "true_count": count,
            "topology": topology,
            "attacker_instances": int(arr.size),
            "mean_rank": float(arr.mean()),
            "median_rank": float(np.median(arr)),
            "p90_rank": float(np.quantile(arr, 0.90)),
            "top1_fraction": float(np.mean(arr <= 1)),
            "top2_fraction": float(np.mean(arr <= 2)),
            "top4_fraction": float(np.mean(arr <= 4)),
        })
    return detailed, summary


def substitution_distance_rows(y_node: np.ndarray, pred_node: np.ndarray) -> list[dict[str, Any]]:
    truth = np.asarray(y_node >= 0.5, dtype=bool)
    pred = np.asarray(pred_node, dtype=bool)
    counts: Counter[int] = Counter()
    sample_counts: Counter[str] = Counter()
    attack_samples = 0
    for t, p in zip(truth, pred):
        true_nodes = np.flatnonzero(t).tolist()
        if not true_nodes:
            continue
        attack_samples += 1
        fp_nodes = np.flatnonzero(p & ~t).tolist()
        missed = np.flatnonzero(t & ~p).tolist()
        for fp in fp_nodes:
            counts[min(manhattan(fp, q) for q in true_nodes)] += 1
        if missed and fp_nodes:
            min_distance = min(manhattan(m, fp) for m in missed for fp in fp_nodes)
            if min_distance <= 1:
                sample_counts["mixed_with_substitution_within_1"] += 1
            elif min_distance <= 2:
                sample_counts["mixed_with_substitution_within_2"] += 1
            else:
                sample_counts["mixed_with_substitution_beyond_2"] += 1
        elif missed:
            sample_counts["misses_without_extras"] += 1
        elif fp_nodes:
            sample_counts["extras_without_misses"] += 1
        else:
            sample_counts["exact"] += 1
    rows = [
        {"kind": "false_positive_distance", "category": str(distance), "count": int(count),
         "fraction": safe_div(count, sum(counts.values()))}
        for distance, count in sorted(counts.items())
    ]
    rows.extend(
        {"kind": "sample_substitution", "category": category, "count": int(count),
         "fraction": safe_div(count, attack_samples)}
        for category, count in sorted(sample_counts.items())
    )
    return rows


def scenario_method_rows(
    pred: Mapping[str, np.ndarray], records: Sequence[Any], methods: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    run_indices = np.asarray(pred["run_index"], dtype=np.int64)
    fields = {
        "profile": np.asarray([records[int(i)].profile or "<empty>" for i in run_indices], dtype=object),
        "attack_kind": np.asarray([records[int(i)].attack_kind or "normal" for i in run_indices], dtype=object),
        "strength": np.asarray([records[int(i)].strength if records[int(i)].strength is not None else -1 for i in run_indices]),
        "attacker_count": np.asarray(pred["y_node"] >= 0.5).sum(axis=1).astype(int),
    }
    truth = np.asarray(pred["y_node"] >= 0.5, dtype=bool)
    rows: list[dict[str, Any]] = []
    for field, values in fields.items():
        for value in sorted(np.unique(values).tolist(), key=lambda x: str(x)):
            mask = values == value
            attack_mask = mask & truth.any(axis=1)
            if not np.any(mask):
                continue
            for method_name, node_pred in methods.items():
                p = np.asarray(node_pred, dtype=bool)
                overall_exact_value = float(np.mean(np.all(truth[mask] == p[mask], axis=1)))
                if np.any(attack_mask):
                    attack_exact = float(np.mean(np.all(truth[attack_mask] == p[attack_mask], axis=1)))
                    attack_count_acc = float(np.mean(truth[attack_mask].sum(axis=1) == p[attack_mask].sum(axis=1)))
                else:
                    attack_exact = math.nan
                    attack_count_acc = math.nan
                rows.append({
                    "group_field": field,
                    "group_value": value,
                    "method": method_name,
                    "sample_count": int(mask.sum()),
                    "attack_sample_count": int(attack_mask.sum()),
                    "overall_exact": overall_exact_value,
                    "attack_exact": attack_exact,
                    "attack_count_accuracy": attack_count_acc,
                })
    return rows


def score_distribution_by_group(pred: Mapping[str, np.ndarray], records: Sequence[Any]) -> list[dict[str, Any]]:
    graph = np.asarray(pred["graph_prob"], dtype=np.float64)
    node = np.asarray(pred["node_prob"], dtype=np.float64)
    truth = np.asarray(pred["y_node"] >= 0.5, dtype=bool)
    run_indices = np.asarray(pred["run_index"], dtype=np.int64)
    fields = {
        "attacker_count": truth.sum(axis=1).astype(int),
        "profile": np.asarray([records[int(i)].profile or "<empty>" for i in run_indices], dtype=object),
        "attack_kind": np.asarray([records[int(i)].attack_kind or "normal" for i in run_indices], dtype=object),
        "strength": np.asarray([records[int(i)].strength if records[int(i)].strength is not None else -1 for i in run_indices]),
    }
    rows: list[dict[str, Any]] = []
    for field, values in fields.items():
        for value in sorted(np.unique(values).tolist(), key=lambda x: str(x)):
            mask = values == value
            if not np.any(mask):
                continue
            graph_values = graph[mask]
            attacker_values = node[mask][truth[mask]]
            nonattacker_values = node[mask][~truth[mask]]
            for score_kind, arr in [
                ("graph", graph_values),
                ("true_attacker_node", attacker_values),
                ("nonattacker_node", nonattacker_values),
            ]:
                if arr.size == 0:
                    continue
                rows.append({
                    "group_field": field,
                    "group_value": value,
                    "score_kind": score_kind,
                    "count": int(arr.size),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "p10": float(np.quantile(arr, 0.10)),
                    "median": float(np.median(arr)),
                    "p90": float(np.quantile(arr, 0.90)),
                })
    return rows


def select_validation_method(
    validation_rows: Sequence[Mapping[str, Any]], oracle_attack_exact: float, frozen_attack_exact: float
) -> dict[str, Any]:
    candidates = [row for row in validation_rows if not str(row["method"]).startswith("oracle")]
    best = max(
        candidates,
        key=lambda row: (
            float(row["attack_exact_localization"]),
            float(row["attack_node_f1"]),
            float(row["overall_exact_localization"]),
        ),
    )
    oracle_gap = max(oracle_attack_exact - frozen_attack_exact, 0.0)
    recovered = float(best["attack_exact_localization"]) - frozen_attack_exact
    fraction = safe_div(max(recovered, 0.0), oracle_gap)
    return {
        "selected_validation_method": best["method"],
        "selected_validation_attack_exact": float(best["attack_exact_localization"]),
        "frozen_validation_attack_exact": frozen_attack_exact,
        "oracle_validation_attack_exact": oracle_attack_exact,
        "oracle_gap": oracle_gap,
        "gap_recovered": recovered,
        "fraction_of_oracle_gap_recovered": fraction,
        "selection_rule": "max validation attack-only exact; tie attack node F1; tie overall exact",
        "test_used_for_selection": False,
    }


def build_report(verdict: Mapping[str, Any]) -> str:
    d = verdict["decision"]
    lines = [
        "# V4-A1 Remaining Diagnostics Report",
        "",
        f"- Analysis pass: **{verdict['analysis_pass']}**",
        "- Model inference performed: **False**",
        "- Test used for fitting/selection: **False**",
        "",
        "## Decoder decision",
        "",
        f"- Selected validation method: **{d['decoder_selection']['selected_validation_method']}**",
        f"- Fraction of validation oracle gap recovered: **{d['decoder_selection']['fraction_of_oracle_gap_recovered']:.3f}**",
        f"- Recommended next model action: **{d['recommended_next_model_action']}**",
        f"- Plain A2 decision: **{d['plain_a2_decision']}**",
        f"- More GCN layers: **{d['more_gcn_layers_decision']}**",
        "",
        "## Interpretation gates",
        "",
    ]
    for gate, value in d["interpretation_gates"].items():
        lines.append(f"- {gate}: `{value}`")
    lines.extend([
        "",
        "## Hard limitations",
        "",
        "- The dataset retains the confirmed matched-control active-core shortcut.",
        "- Saved tensors do not contain explicit valid-port/clipping masks.",
        "- Temporal max-pooling causality cannot be proven without an encoder ablation.",
        "- Mean-readout causality cannot be proven without saved embeddings or a readout ablation.",
        "- The test split is a development-comparison set, not an independent publication holdout.",
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--validation-predictions", required=True)
    parser.add_argument("--test-predictions", required=True)
    parser.add_argument("--selected-thresholds", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-analysis-script")
    parser.add_argument("--count-train-cap", type=int, default=250000)
    parser.add_argument("--calibration-node-cap-per-topology", type=int, default=500000)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir).resolve()
    validation_path = Path(args.validation_predictions).resolve()
    test_path = Path(args.test_predictions).resolve()
    thresholds_path = Path(args.selected_thresholds).resolve()
    out = Path(args.out_dir).resolve()
    base_path = (
        Path(args.base_analysis_script).resolve()
        if args.base_analysis_script
        else Path(__file__).resolve().with_name("analyze_v4_a1_failures.py")
    )
    required = [
        data_dir / "metadata.json", validation_path, test_path, thresholds_path, base_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    ensure_empty_output(out)
    base = load_base_module(base_path)
    metadata = base.load_json(data_dir / "metadata.json")
    records = base.build_run_records(metadata)
    val = base.load_predictions(validation_path)
    test = base.load_predictions(test_path)
    thresholds = base.load_json(thresholds_path)
    graph_threshold = float(thresholds["graph_threshold"])
    node_threshold = float(thresholds["node_threshold"])

    # Fit validation-only count decoder and topology calibration.
    scaler, count_model, count_feature_names, count_info = fit_count_decoder(
        val, args.count_train_cap, args.seed
    )
    val_learned_count, val_count_prob = predict_counts(val, scaler, count_model)
    test_learned_count, test_count_prob = predict_counts(test, scaler, count_model)
    calibrators, calibration_info = fit_topology_calibrators(
        val, args.calibration_node_cap_per_topology, args.seed
    )
    val_cal_scores = apply_topology_calibration(val, calibrators)
    test_cal_scores = apply_topology_calibration(test, calibrators)

    # Predeclared decoders.
    methods_by_split: dict[str, dict[str, np.ndarray]] = {}
    count_predictions: dict[str, dict[str, np.ndarray]] = {}
    for split_name, pred, learned_count, cal_scores in [
        ("validation", val, val_learned_count, val_cal_scores),
        ("test", test, test_learned_count, test_cal_scores),
    ]:
        truth = np.asarray(pred["y_node"] >= 0.5, dtype=bool)
        true_count = truth.sum(axis=1).astype(np.int64)
        graph_gate = np.asarray(pred["graph_prob"] >= graph_threshold, dtype=bool)
        frozen = np.asarray(pred["node_prob"] >= node_threshold, dtype=bool)
        graph_gated_frozen = frozen & graph_gate[:, None]
        oracle = topk_prediction(pred["node_prob"], true_count)
        learned = topk_prediction(pred["node_prob"], learned_count)
        gated_learned_count = learned_count.copy()
        gated_learned_count[~graph_gate] = 0
        gated_learned = topk_prediction(pred["node_prob"], gated_learned_count)
        gap_count = largest_gap_counts(pred, graph_threshold)
        gap = topk_prediction(pred["node_prob"], gap_count)
        calibrated_learned = topk_prediction(cal_scores, learned_count)
        calibrated_gated = topk_prediction(cal_scores, gated_learned_count)
        methods_by_split[split_name] = {
            "frozen_threshold": frozen,
            "graph_gated_frozen": graph_gated_frozen,
            "oracle_true_count_topk": oracle,
            "learned_count_topk": learned,
            "graph_gated_learned_count_topk": gated_learned,
            "largest_gap_graph_gated_topk": gap,
            "topology_calibrated_learned_count_topk": calibrated_learned,
            "topology_calibrated_graph_gated_topk": calibrated_gated,
        }
        count_predictions[split_name] = {
            "learned_count": learned_count,
            "graph_gated_learned_count": gated_learned_count,
            "largest_gap_count": gap_count,
        }

    # Fit predicted-count-specific thresholds on validation only and transfer.
    per_count_thresholds = fit_predicted_count_thresholds(
        val, val_learned_count,
        args.threshold_start, args.threshold_end, args.threshold_step,
    )
    methods_by_split["validation"]["predicted_count_specific_thresholds"] = apply_predicted_count_thresholds(
        np.asarray(val["node_prob"]), val_learned_count, per_count_thresholds
    )
    methods_by_split["test"]["predicted_count_specific_thresholds"] = apply_predicted_count_thresholds(
        np.asarray(test["node_prob"]), test_learned_count, per_count_thresholds
    )

    # Write model fitting provenance.
    provenance = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "base_analysis_script": str(base_path),
        "base_analysis_script_sha256": sha256_file(base_path),
        "metadata": str(data_dir / "metadata.json"),
        "metadata_sha256": sha256_file(data_dir / "metadata.json"),
        "validation_predictions": str(validation_path),
        "validation_predictions_sha256": sha256_file(validation_path),
        "test_predictions": str(test_path),
        "test_predictions_sha256": sha256_file(test_path),
        "selected_thresholds": str(thresholds_path),
        "selected_thresholds_sha256": sha256_file(thresholds_path),
        "graph_threshold": graph_threshold,
        "node_threshold": node_threshold,
        "model_inference_performed": False,
        "neural_model_training_performed": False,
        "test_used_for_fit_or_selection": False,
        "validation_only_fitted_components": [
            "multinomial count decoder", "topology Platt calibrators",
            "predicted-count-specific thresholds",
        ],
        "count_decoder": count_info,
        "topology_calibrators": calibration_info,
        "predicted_count_specific_thresholds": {str(k): v for k, v in per_count_thresholds.items()},
    }
    json_dump(out / "provenance.json", provenance)

    split_method_rows: dict[str, list[dict[str, Any]]] = {}
    for split_name, pred in [("validation", val), ("test", test)]:
        split_dir = out / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        rows = [method_row(name, pred, node_pred) for name, node_pred in methods_by_split[split_name].items()]
        split_method_rows[split_name] = rows
        write_csv(split_dir / "decoder_method_comparison.csv", rows)
        for name, node_pred in methods_by_split[split_name].items():
            write_csv(split_dir / f"exact_set_decomposition__{name}.csv", exact_decomposition(pred["y_node"], node_pred))
            write_csv(split_dir / f"graph_node_consistency__{name}.csv", graph_node_consistency(pred, node_pred, graph_threshold))
            write_csv(split_dir / f"substitution_distance__{name}.csv", substitution_distance_rows(pred["y_node"], node_pred))
        for count_name, counts in count_predictions[split_name].items():
            write_csv(split_dir / f"count_confusion__{count_name}.csv", count_confusion(pred["y_node"], counts))
        rank_detail, rank_summary = attacker_rank_rows(pred, records)
        write_csv(split_dir / "true_attacker_rank_detail.csv", rank_detail)
        write_csv(split_dir / "true_attacker_rank_summary.csv", rank_summary)
        write_csv(split_dir / "scenario_method_comparison.csv", scenario_method_rows(pred, records, methods_by_split[split_name]))
        write_csv(split_dir / "score_distributions_by_scenario.csv", score_distribution_by_group(pred, records))

    # Validation-only model selection and transferred test report.
    val_map = {row["method"]: row for row in split_method_rows["validation"]}
    frozen_val = val_map["frozen_threshold"]
    oracle_val = val_map["oracle_true_count_topk"]
    decoder_selection = select_validation_method(
        split_method_rows["validation"],
        float(oracle_val["attack_exact_localization"]),
        float(frozen_val["attack_exact_localization"]),
    )
    selected_method = decoder_selection["selected_validation_method"]
    test_map = {row["method"]: row for row in split_method_rows["test"]}
    selected_test = test_map[selected_method]
    frozen_test = test_map["frozen_threshold"]
    oracle_test = test_map["oracle_true_count_topk"]

    topology_method = "topology_calibrated_learned_count_topk"
    topology_gain_val = (
        float(val_map[topology_method]["attack_exact_localization"])
        - float(val_map["learned_count_topk"]["attack_exact_localization"])
    )
    learned_recovery = decoder_selection["fraction_of_oracle_gap_recovered"]

    if learned_recovery >= 0.50:
        next_action = "prioritize_explicit_count_head_and_count_conditioned_topk"
        a2_decision = "skip_plain_a2_and_move_to_structural_count_aware_model"
    elif learned_recovery >= 0.20:
        next_action = "run_plain_a2_only_as_one_seed_capacity_probe_then_structural_model"
        a2_decision = "optional_secondary_capacity_probe"
    else:
        next_action = "representation_ranking_dominates_build_source_preserving_model"
        a2_decision = "plain_a2_low_priority"

    if topology_gain_val >= 0.03:
        topology_gate = "large_gain_fix_topology_semantics_before_capacity"
    elif topology_gain_val >= 0.01:
        topology_gate = "meaningful_gain_include_topology_masks_in_next_model"
    else:
        topology_gate = "small_posthoc_gain_but_masks_still_required_for_dataset_validity"

    verdict = {
        "analysis_pass": True,
        "hard_checks": {
            "required_inputs_present": True,
            "validation_only_fitting": True,
            "test_used_for_fit_or_selection": False,
            "model_inference_performed": False,
            "neural_model_training_performed": False,
            "dataset_modified": False,
            "all_predeclared_decoders_evaluated": True,
        },
        "thresholds": {"graph": graph_threshold, "node": node_threshold},
        "decision": {
            "decoder_selection": decoder_selection,
            "selected_method_test_transfer": selected_test,
            "frozen_test": frozen_test,
            "oracle_test": oracle_test,
            "topology_calibration_validation_attack_exact_gain": topology_gain_val,
            "recommended_next_model_action": next_action,
            "plain_a2_decision": a2_decision,
            "more_gcn_layers_decision": "do_not_add_more_gcn_layers; neighbour spreading is already a dominant failure",
            "wider_model_decision": "width_only_is_secondary; use width inside a source_preserving architecture",
            "count_head_decision": "yes_in_next_structural_model",
            "message_passing_decision": "use_one_graph_layer_plus_local_skip_or_gated_propagation",
            "graph_readout_decision": "test_mean_plus_max_ablation_for_single_attacker_detection",
            "temporal_encoder_decision": "test_mean_plus_max_temporal pooling or a shallow dilated TCN; do not jump to RNN",
            "topology_gate": topology_gate,
            "interpretation_gates": {
                "decoder_gap_recovery_ge_0_50": learned_recovery >= 0.50,
                "decoder_gap_recovery_ge_0_20": learned_recovery >= 0.20,
                "topology_calibration_gain_ge_0_03": topology_gain_val >= 0.03,
                "oracle_test_attack_exact_below_0_50": float(oracle_test["attack_exact_localization"]) < 0.50,
                "ranking_representation_problem_remains": float(oracle_test["attack_exact_localization"]) < 0.50,
            },
        },
        "limitations": [
            "Current V4 active-core matched-control shortcut remains a publication blocker",
            "No explicit valid/idle/clipped port masks",
            "Mean-readout causality requires an embedding/readout ablation",
            "Temporal-max causality requires an encoder ablation",
            "Test is a development-comparison set, not an independent publication holdout",
        ],
    }
    json_dump(out / "final_decision.json", verdict)
    (out / "remaining_diagnostics_report.md").write_text(build_report(verdict), encoding="utf-8")

    manifest_rows = []
    for path in sorted(out.rglob("*")):
        if path.is_file():
            manifest_rows.append({
                "relative_path": str(path.relative_to(out)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    write_csv(out / "artifact_manifest.csv", manifest_rows)

    print("V4_A1_REMAINING_DIAGNOSTICS_PASS")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
