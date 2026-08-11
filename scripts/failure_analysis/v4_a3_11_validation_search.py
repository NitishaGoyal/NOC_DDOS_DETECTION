#!/usr/bin/env python3
"""
V4-A3.11 validation-only temporal aggregation search.

This script:
  * reads ONLY the frozen A3 validation prediction archive
  * verifies the A3.11 preflight pass marker
  * never reads the development-test prediction archive
  * searches causal temporal aggregation policies on validation only
  * selects separate alert and confirmed-isolation policies
  * writes a frozen validation policy package for later one-time transfer

Dense mode:
  consecutive stride-1 prediction windows are aggregated.
  H predictions span H+7 original epochs because each base sample has T=8
  and neighboring samples have stride 1.

De-overlapped mode:
  every eighth base prediction is used before aggregation.
  H predictions span approximately 8H original epochs.

No model training occurs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EPS = 1e-6
HORIZONS = (1, 2, 4, 8, 16, 32, 64)
EMA_ALPHAS = (0.2, 0.5, 0.8)
NODE_THRESHOLDS = tuple(
    sorted(
        set(
            [round(x, 4) for x in np.arange(0.50, 0.951, 0.025)]
            + [0.78]
        )
    )
)
GRAPH_THRESHOLDS = tuple(
    sorted(
        set(
            [round(x, 4) for x in np.arange(0.05, 0.951, 0.01)]
            + [0.46]
        )
    )
)
PERSISTENCE_FRACTIONS = (0.50, 0.75, 1.00)
STABLE_STREAK = 4


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out.astype(np.float32)


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q)).astype(np.float32)


def normalize_prob_rows(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    a = np.maximum(a, EPS)
    sums = np.sum(a, axis=1, keepdims=True)
    return (a / sums).astype(np.float32)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    yt = np.asarray(y_true, dtype=np.int8).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.int8).reshape(-1)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {
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


def exact_rate(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    exact = np.all(np.asarray(y_true) == np.asarray(y_pred), axis=1)
    if mask is not None:
        exact = exact[np.asarray(mask, dtype=bool)]
    return float(np.mean(exact)) if exact.size else 0.0


def decode_count_topk(node_score: np.ndarray, pred_count: np.ndarray) -> np.ndarray:
    score = np.asarray(node_score)
    count = np.asarray(pred_count, dtype=np.int64)
    n = score.shape[0]
    pred = np.zeros((n, 16), dtype=np.int8)
    order = np.argsort(-score, axis=1, kind="stable")
    for k in range(1, 5):
        rows = np.flatnonzero(count == k)
        if rows.size:
            pred[rows[:, None], order[rows, :k]] = 1
    return pred


def node_metrics(
    y_graph: np.ndarray,
    y_node: np.ndarray,
    pred_node: np.ndarray,
    true_count: np.ndarray,
    pred_count: np.ndarray,
) -> dict[str, float | int]:
    yg = np.asarray(y_graph, dtype=np.int8)
    yt = np.asarray(y_node, dtype=np.int8)
    yp = np.asarray(pred_node, dtype=np.int8)
    tc = np.asarray(true_count, dtype=np.int64)
    pc = np.asarray(pred_count, dtype=np.int64)
    attack = yg == 1
    normal = ~attack

    overall = binary_metrics(yt, yp)
    attack_b = binary_metrics(yt[attack], yp[attack])

    pred_sizes = np.sum(yp, axis=1)
    attack_count = int(np.sum(attack))
    normal_count = int(np.sum(normal))

    return {
        "node_precision": overall["precision"],
        "node_recall": overall["recall"],
        "node_f1": overall["f1"],
        "node_fp": overall["fp"],
        "node_fn": overall["fn"],
        "node_tp": overall["tp"],
        "attack_node_precision": attack_b["precision"],
        "attack_node_recall": attack_b["recall"],
        "attack_node_f1": attack_b["f1"],
        "attack_node_fp": attack_b["fp"],
        "attack_node_fn": attack_b["fn"],
        "attack_node_tp": attack_b["tp"],
        "exact_localization": exact_rate(yt, yp),
        "attack_exact_localization": exact_rate(yt, yp, attack),
        "count_accuracy": float(np.mean(tc == pc)),
        "count_mae": float(np.mean(np.abs(tc - pc))),
        "attack_count_accuracy": float(np.mean(tc[attack] == pc[attack])),
        "attack_count_mae": float(np.mean(np.abs(tc[attack] - pc[attack]))),
        "attack_empty_prediction_rate": float(np.mean(pred_sizes[attack] == 0)),
        "attack_isolation_coverage": float(np.mean(pred_sizes[attack] > 0)),
        "normal_false_isolation_rate": float(np.mean(pred_sizes[normal] > 0)),
        "false_routers_per_attack_window": safe_div(
            int(np.sum(yp[attack] & (1 - yt[attack]))), attack_count
        ),
        "missed_attackers_per_attack_window": safe_div(
            int(np.sum((1 - yp[attack]) & yt[attack])), attack_count
        ),
        "false_routers_per_normal_window": safe_div(
            int(np.sum(yp[normal])), normal_count
        ),
        "sample_count": int(yg.size),
        "attack_sample_count": attack_count,
        "normal_sample_count": normal_count,
    }


def run_boundaries(run_index: np.ndarray) -> list[tuple[int, int]]:
    r = np.asarray(run_index)
    if r.size == 0:
        return []
    starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    ends = np.r_[starts[1:], r.size]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def longest_true_streak(values: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in np.asarray(values, dtype=bool):
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def rolling_true_after_streak(values: np.ndarray, streak: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.int8)
    out = np.zeros_like(v, dtype=bool)
    if v.size < streak:
        return out
    c = np.cumsum(np.r_[0, v])
    sums = c[streak:] - c[:-streak]
    out[streak - 1 :] = sums == streak
    return out


def operational_metrics(
    run_idx: np.ndarray,
    y_graph: np.ndarray,
    y_node: np.ndarray,
    pred_node: np.ndarray,
) -> dict[str, float | int]:
    runs = np.asarray(run_idx)
    yg = np.asarray(y_graph, dtype=np.int8)
    yt = np.asarray(y_node, dtype=np.int8)
    yp = np.asarray(pred_node, dtype=np.int8)

    pred_nonempty = np.sum(yp, axis=1) > 0
    exact = np.all(yt == yp, axis=1)
    attack = yg == 1
    normal = ~attack

    normal_runs = 0
    normal_ever_isolated = 0
    normal_persistent = 0
    max_false_streak = 0
    false_streak_lengths: list[int] = []

    attack_runs = 0
    first_exact_delays: list[int] = []
    first_stable_delays: list[int] = []
    stable_exact_points = 0
    attack_points = 0
    churn_values: list[float] = []

    for s, e in run_boundaries(runs):
        run_attack = bool(np.any(attack[s:e]))
        if not run_attack:
            normal_runs += 1
            f = pred_nonempty[s:e]
            if np.any(f):
                normal_ever_isolated += 1
            max_run = longest_true_streak(f)
            max_false_streak = max(max_false_streak, max_run)
            if max_run >= STABLE_STREAK:
                normal_persistent += 1

            cur = 0
            for val in f:
                if val:
                    cur += 1
                elif cur:
                    false_streak_lengths.append(cur)
                    cur = 0
            if cur:
                false_streak_lengths.append(cur)
        else:
            attack_runs += 1
            ex = exact[s:e] & attack[s:e]
            attack_positions = np.flatnonzero(attack[s:e])
            if attack_positions.size:
                first_attack = int(attack_positions[0])
                exact_after = np.flatnonzero(ex[first_attack:])
                if exact_after.size:
                    first_exact_delays.append(int(exact_after[0]))
                stable = rolling_true_after_streak(ex[first_attack:], STABLE_STREAK)
                stable_pos = np.flatnonzero(stable)
                if stable_pos.size:
                    first_stable_delays.append(int(stable_pos[0]))
                stable_exact_points += int(np.sum(stable))
                attack_points += int(np.sum(attack[s:e]))

            for i in range(s + 1, e):
                if not (attack[i] and attack[i - 1]):
                    continue
                a = yp[i - 1].astype(bool)
                b = yp[i].astype(bool)
                union = int(np.sum(a | b))
                inter = int(np.sum(a & b))
                churn_values.append(0.0 if union == 0 else 1.0 - inter / union)

    return {
        "normal_run_count": normal_runs,
        "normal_run_ever_isolated_fraction": safe_div(
            normal_ever_isolated, normal_runs
        ),
        "normal_run_persistent_false_isolation_fraction": safe_div(
            normal_persistent, normal_runs
        ),
        "maximum_false_isolation_streak": max_false_streak,
        "mean_false_isolation_streak_length": float(
            np.mean(false_streak_lengths)
        ) if false_streak_lengths else 0.0,
        "attack_run_count": attack_runs,
        "attack_run_first_exact_coverage": safe_div(
            len(first_exact_delays), attack_runs
        ),
        "attack_run_first_stable_exact_coverage": safe_div(
            len(first_stable_delays), attack_runs
        ),
        "median_time_to_first_exact_decisions": float(
            np.median(first_exact_delays)
        ) if first_exact_delays else math.nan,
        "median_time_to_stable_exact_decisions": float(
            np.median(first_stable_delays)
        ) if first_stable_delays else math.nan,
        "stable_attack_exact_point_rate": safe_div(
            stable_exact_points, attack_points
        ),
        "mean_attack_set_churn": float(np.mean(churn_values))
        if churn_values else 0.0,
    }


@dataclass
class PreparedData:
    graph_prob: np.ndarray
    node_prob: np.ndarray
    count_prob: np.ndarray
    y_graph: np.ndarray
    y_node: np.ndarray
    attacker_count: np.ndarray
    global_index: np.ndarray
    run_index: np.ndarray
    end_epoch: np.ndarray
    split_id: np.ndarray
    attack_kind_id: np.ndarray
    profile_id: np.ndarray
    strength: np.ndarray


def prepare_data(pred_path: Path, data_dir: Path) -> PreparedData:
    with np.load(pred_path, allow_pickle=False) as z:
        raw = {key: np.asarray(z[key]) for key in z.files}

    gi = raw["global_index"].astype(np.int64)
    end_epoch_all = np.load(
        data_dir / "end_epoch.npy", mmap_mode="r", allow_pickle=False
    )
    strength_all = np.load(
        data_dir / "strength.npy", mmap_mode="r", allow_pickle=False
    )

    end_epoch = np.asarray(end_epoch_all[gi], dtype=np.int64)
    strength = np.asarray(strength_all[gi], dtype=np.int64)

    # Sort once by run and time. Labels/metadata follow the same order.
    order = np.lexsort((end_epoch, raw["run_index"].astype(np.int64)))

    def take(name: str) -> np.ndarray:
        return np.asarray(raw[name][order])

    return PreparedData(
        graph_prob=take("graph_prob").astype(np.float32),
        node_prob=take("node_prob").astype(np.float32),
        count_prob=take("count_prob").astype(np.float32),
        y_graph=take("y_graph").astype(np.int8),
        y_node=take("y_node").astype(np.int8),
        attacker_count=take("attacker_count").astype(np.int64),
        global_index=take("global_index").astype(np.int64),
        run_index=take("run_index").astype(np.int64),
        end_epoch=end_epoch[order],
        split_id=take("split_id").astype(np.int64),
        attack_kind_id=take("attack_kind_id").astype(np.int64),
        profile_id=take("profile_id").astype(np.int64),
        strength=strength[order],
    )


def make_segments(data: PreparedData) -> list[np.ndarray]:
    """
    Split by run and by any label change. This prevents an aggregation window
    from crossing run or target-label boundaries.
    """
    n = data.run_index.size
    if n == 0:
        return []

    change = np.ones(n, dtype=bool)
    change[1:] = (
        (data.run_index[1:] != data.run_index[:-1])
        | (data.y_graph[1:] != data.y_graph[:-1])
        | (data.attacker_count[1:] != data.attacker_count[:-1])
        | np.any(data.y_node[1:] != data.y_node[:-1], axis=1)
    )
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], n]
    return [np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends)]


def mode_segments(
    segments: list[np.ndarray], mode: str, base_stride: int
) -> list[np.ndarray]:
    if mode == "dense":
        return segments
    if mode == "deoverlap":
        return [seg[::base_stride] for seg in segments if seg.size]
    raise ValueError(f"unsupported mode: {mode}")


def aggregate_array(
    arr: np.ndarray,
    segments: list[np.ndarray],
    horizon: int,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate along each segment. weights are ordered oldest -> newest.
    Returns endpoint indices and aggregated values.
    """
    values = np.asarray(arr)
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False

    endpoint_parts: list[np.ndarray] = []
    output_parts: list[np.ndarray] = []

    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)

    for seg in segments:
        if seg.size < horizon:
            continue
        v = values[seg]
        windows = np.lib.stride_tricks.sliding_window_view(
            v, window_shape=horizon, axis=0
        )
        # shape: [L-H+1, D, H]
        aggregated = np.tensordot(windows, w, axes=([2], [0]))
        endpoint_parts.append(seg[horizon - 1 :])
        output_parts.append(aggregated.astype(np.float32))

    if not endpoint_parts:
        empty_shape = (0,) if squeeze else (0, values.shape[1])
        return np.empty((0,), dtype=np.int64), np.empty(
            empty_shape, dtype=np.float32
        )

    endpoints = np.concatenate(endpoint_parts)
    out = np.concatenate(output_parts, axis=0)
    if squeeze:
        out = out[:, 0]
    return endpoints, out


def aggregate_minmax(
    arr: np.ndarray,
    segments: list[np.ndarray],
    horizon: int,
    reducer: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(arr)
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False

    endpoint_parts: list[np.ndarray] = []
    output_parts: list[np.ndarray] = []
    for seg in segments:
        if seg.size < horizon:
            continue
        v = values[seg]
        windows = np.lib.stride_tricks.sliding_window_view(
            v, window_shape=horizon, axis=0
        )
        if reducer == "max":
            agg = np.max(windows, axis=2)
        else:
            raise ValueError(reducer)
        endpoint_parts.append(seg[horizon - 1 :])
        output_parts.append(agg.astype(np.float32))

    if not endpoint_parts:
        empty_shape = (0,) if squeeze else (0, values.shape[1])
        return np.empty((0,), dtype=np.int64), np.empty(
            empty_shape, dtype=np.float32
        )
    endpoints = np.concatenate(endpoint_parts)
    out = np.concatenate(output_parts, axis=0)
    if squeeze:
        out = out[:, 0]
    return endpoints, out


def rank_scores(node_prob: np.ndarray) -> np.ndarray:
    order = np.argsort(-node_prob, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(node_prob.shape[0])[:, None]
    ranks[rows, order] = np.arange(16)[None, :]
    return (1.0 - ranks.astype(np.float32) / 15.0).astype(np.float32)


def ema_weights(horizon: int, alpha: float) -> np.ndarray:
    # oldest -> newest
    lags = np.arange(horizon - 1, -1, -1, dtype=np.float64)
    return alpha * np.power(1.0 - alpha, lags)


def uniform_weights(horizon: int) -> np.ndarray:
    return np.ones(horizon, dtype=np.float64)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def row_identity(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "mode",
        "horizon",
        "evidence_span_epochs",
        "graph_method",
        "graph_threshold",
        "node_method",
        "count_method",
        "decoder",
        "node_threshold",
        "persistence_fraction",
        "graph_gating",
    )
    return {key: row[key] for key in keys if key in row}


def choose_best_alert(
    rows: list[dict[str, Any]], fpr_cap: float
) -> dict[str, Any] | None:
    feasible = [r for r in rows if float(r["fpr"]) <= fpr_cap]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda r: (
            float(r["f1"]),
            float(r["recall"]),
            float(r["precision"]),
            -float(r["fpr"]),
        ),
    )


def safety_distance(row: dict[str, Any]) -> float:
    precision_deficit = max(0.0, 0.95 - float(row["attack_node_precision"]))
    false_iso_excess = max(
        0.0, float(row["normal_false_isolation_rate"]) - 0.05
    )
    coverage_deficit = max(
        0.0, 0.50 - float(row["attack_isolation_coverage"])
    )
    return (
        5.0 * precision_deficit
        + 5.0 * false_iso_excess
        + 2.0 * coverage_deficit
    )


def select_isolation_shortlist(
    rows: list[dict[str, Any]], limit: int = 25
) -> tuple[list[dict[str, Any]], bool]:
    feasible = [
        r
        for r in rows
        if float(r["attack_node_precision"]) >= 0.95
        and float(r["normal_false_isolation_rate"]) <= 0.05
        and float(r["attack_isolation_coverage"]) >= 0.50
    ]
    if feasible:
        ranked = sorted(
            feasible,
            key=lambda r: (
                -float(r["attack_exact_localization"]),
                -float(r["attack_node_f1"]),
                float(r["normal_false_isolation_rate"]),
                -float(r["attack_isolation_coverage"]),
            ),
        )
        return ranked[:limit], True

    ranked = sorted(
        rows,
        key=lambda r: (
            safety_distance(r),
            -float(r["attack_exact_localization"]),
            -float(r["attack_node_f1"]),
        ),
    )
    return ranked[:limit], False


def reconstruct_policy_predictions(
    row: dict[str, Any],
    data: PreparedData,
    segments: list[np.ndarray],
    horizon: int,
    mode: str,
    base_stride: int,
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct node predictions for a selected row.
    """
    node_method = row["node_method"]
    decoder = row["decoder"]
    count_method = row.get("count_method", "")
    node_threshold = row.get("node_threshold", "")
    persistence_fraction = row.get("persistence_fraction", "")

    endpoints, node_score = cache[f"node::{node_method}"]

    if decoder == "NodeThreshold":
        threshold = float(node_threshold)
        pred = (node_score >= threshold).astype(np.int8)
    elif decoder == "NodePersistence":
        fraction = float(persistence_fraction)
        _, vote = cache["node::base_threshold_vote"]
        pred = (vote >= fraction - 1e-12).astype(np.int8)
    elif decoder == "CountTopK":
        pred_count = cache[f"count::{count_method}"][1].astype(np.int64)
        pred = decode_count_topk(node_score, pred_count)
    else:
        raise ValueError(f"unsupported decoder {decoder}")

    if str(row.get("graph_gating", "False")).lower() == "true":
        graph_method = row["graph_method"]
        graph_threshold = float(row["graph_threshold"])
        g_end, g_score = cache[f"graph::{graph_method}"]
        if not np.array_equal(g_end, endpoints):
            raise RuntimeError("graph/node endpoint mismatch")
        gate = g_score >= graph_threshold
        pred = pred.copy()
        pred[~gate] = 0

    return endpoints, pred


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-window-length", type=int, default=8)
    parser.add_argument("--deoverlap-stride", type=int, default=8)
    args = parser.parse_args()

    required = [
        args.data_dir,
        args.validation_predictions,
        args.thresholds,
        args.preflight_dir / "V4_A3_11_PREFLIGHT_PASS",
        args.preflight_dir / "preflight_report.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("VALIDATION SEARCH FAIL: missing required paths", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(
            f"VALIDATION SEARCH FAIL: output already exists: {args.output_dir}",
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True)

    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    frozen_graph_threshold = float(thresholds["graph_threshold"])
    frozen_node_threshold = float(thresholds["node_threshold"])

    data = prepare_data(args.validation_predictions, args.data_dir)
    segments_dense = make_segments(data)

    # Global temporal sanity.
    bad_gaps = 0
    for seg in segments_dense:
        if seg.size > 1:
            gaps = np.diff(data.end_epoch[seg])
            bad_gaps += int(np.sum(gaps != 1))
    if bad_gaps:
        print(
            f"VALIDATION SEARCH FAIL: {bad_gaps} non-unit epoch gaps in segments",
            file=sys.stderr,
        )
        return 1

    graph_logit = logit(data.graph_prob)
    node_logit = logit(data.node_prob)
    node_rank = rank_scores(data.node_prob)
    count_logprob = np.log(np.clip(data.count_prob, EPS, 1.0)).astype(np.float32)
    count_argmax = np.argmax(data.count_prob, axis=1).astype(np.int64)
    count_onehot = np.eye(5, dtype=np.float32)[count_argmax]
    base_node_binary = (data.node_prob >= frozen_node_threshold).astype(np.float32)
    base_graph_binary = (data.graph_prob >= frozen_graph_threshold).astype(np.float32)

    alert_rows: list[dict[str, Any]] = []
    loc_rows: list[dict[str, Any]] = []
    shortlisted_reconstructions: dict[tuple[str, int], dict[str, Any]] = {}
    per_configuration_cache: dict[tuple[str, int], dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    for mode in ("dense", "deoverlap"):
        segments = mode_segments(
            segments_dense, mode, args.deoverlap_stride
        )
        for horizon in HORIZONS:
            if mode == "dense":
                evidence_span = args.base_window_length + horizon - 1
            else:
                evidence_span = args.base_window_length * horizon

            cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

            # Graph methods.
            end, mean_graph_logit = aggregate_array(
                graph_logit, segments, horizon, uniform_weights(horizon)
            )
            cache["graph::mean_logit"] = (end, sigmoid(mean_graph_logit))

            for alpha in EMA_ALPHAS:
                name = f"ema_logit_a{alpha:g}"
                e, agg = aggregate_array(
                    graph_logit, segments, horizon, ema_weights(horizon, alpha)
                )
                cache[f"graph::{name}"] = (e, sigmoid(agg))

            e, max_graph = aggregate_minmax(
                data.graph_prob, segments, horizon, "max"
            )
            cache["graph::max_prob"] = (e, max_graph)

            e, graph_vote = aggregate_array(
                base_graph_binary,
                segments,
                horizon,
                uniform_weights(horizon),
            )
            cache["graph::base_threshold_vote"] = (e, graph_vote)

            # Node methods.
            e, mean_node_logit = aggregate_array(
                node_logit, segments, horizon, uniform_weights(horizon)
            )
            cache["node::mean_logit"] = (e, sigmoid(mean_node_logit))

            for alpha in EMA_ALPHAS:
                name = f"ema_logit_a{alpha:g}"
                e, agg = aggregate_array(
                    node_logit, segments, horizon, ema_weights(horizon, alpha)
                )
                cache[f"node::{name}"] = (e, sigmoid(agg))

            e, mean_rank = aggregate_array(
                node_rank, segments, horizon, uniform_weights(horizon)
            )
            cache["node::mean_rank"] = (e, mean_rank)

            e, max_node = aggregate_minmax(
                data.node_prob, segments, horizon, "max"
            )
            cache["node::max_prob"] = (e, max_node)

            e, node_vote = aggregate_array(
                base_node_binary,
                segments,
                horizon,
                uniform_weights(horizon),
            )
            cache["node::base_threshold_vote"] = (e, node_vote)

            # Count methods.
            e, mean_count_prob = aggregate_array(
                data.count_prob, segments, horizon, uniform_weights(horizon)
            )
            mean_count_prob = normalize_prob_rows(mean_count_prob)
            cache["count::mean_prob"] = (
                e,
                np.argmax(mean_count_prob, axis=1).astype(np.int64),
            )

            e, mean_count_logprob = aggregate_array(
                count_logprob, segments, horizon, uniform_weights(horizon)
            )
            mean_count_logprob = mean_count_logprob - np.max(
                mean_count_logprob, axis=1, keepdims=True
            )
            geo_prob = normalize_prob_rows(np.exp(mean_count_logprob))
            cache["count::mean_logprob"] = (
                e,
                np.argmax(geo_prob, axis=1).astype(np.int64),
            )

            e, mode_vote = aggregate_array(
                count_onehot, segments, horizon, uniform_weights(horizon)
            )
            cache["count::mode"] = (
                e,
                np.argmax(mode_vote, axis=1).astype(np.int64),
            )

            # Median count.
            median_endpoints: list[np.ndarray] = []
            median_values: list[np.ndarray] = []
            for seg in segments:
                if seg.size < horizon:
                    continue
                vals = count_argmax[seg]
                windows = np.lib.stride_tricks.sliding_window_view(
                    vals, window_shape=horizon
                )
                med = np.floor(np.median(windows, axis=1) + 0.5).astype(np.int64)
                median_endpoints.append(seg[horizon - 1 :])
                median_values.append(med)
            med_end = np.concatenate(median_endpoints)
            med_val = np.concatenate(median_values)
            cache["count::median"] = (med_end, med_val)

            per_configuration_cache[(mode, horizon)] = cache

            # Endpoint labels/metadata.
            endpoints = end
            yg = data.y_graph[endpoints]
            yn = data.y_node[endpoints]
            tc = data.attacker_count[endpoints]
            ri = data.run_index[endpoints]

            # Alert candidates: score methods + threshold sweep.
            for key, (g_end, score) in cache.items():
                if not key.startswith("graph::"):
                    continue
                method = key.split("::", 1)[1]
                if not np.array_equal(g_end, endpoints):
                    raise RuntimeError("graph endpoint mismatch")

                if method == "base_threshold_vote":
                    for fraction in PERSISTENCE_FRACTIONS:
                        pred = score >= fraction - 1e-12
                        m = binary_metrics(yg, pred)
                        alert_rows.append(
                            {
                                "mode": mode,
                                "horizon": horizon,
                                "evidence_span_epochs": evidence_span,
                                "graph_method": method,
                                "graph_threshold": fraction,
                                **m,
                            }
                        )
                else:
                    for threshold in GRAPH_THRESHOLDS:
                        pred = score >= threshold
                        m = binary_metrics(yg, pred)
                        alert_rows.append(
                            {
                                "mode": mode,
                                "horizon": horizon,
                                "evidence_span_epochs": evidence_span,
                                "graph_method": method,
                                "graph_threshold": threshold,
                                **m,
                            }
                        )

            # Per-configuration alert gate for optional confirmed isolation.
            same_alert_rows = [
                r
                for r in alert_rows
                if r["mode"] == mode and r["horizon"] == horizon
            ]
            local_alert = choose_best_alert(same_alert_rows, 0.10)
            if local_alert is None:
                local_alert = min(
                    same_alert_rows,
                    key=lambda r: (
                        float(r["fpr"]),
                        -float(r["f1"]),
                    ),
                )
            gate_method = str(local_alert["graph_method"])
            gate_threshold = float(local_alert["graph_threshold"])
            g_end, g_score = cache[f"graph::{gate_method}"]
            graph_gate = g_score >= gate_threshold

            node_methods = [
                "mean_logit",
                "ema_logit_a0.2",
                "ema_logit_a0.5",
                "ema_logit_a0.8",
                "mean_rank",
                "max_prob",
            ]
            count_methods = ["mean_prob", "mean_logprob", "mode", "median"]

            def add_localization_row(
                pred: np.ndarray,
                decoder: str,
                node_method: str,
                count_method: str = "",
                node_threshold: float | str = "",
                persistence_fraction: float | str = "",
                graph_gating: bool = False,
            ) -> None:
                pred_count = np.sum(pred, axis=1).astype(np.int64)
                metrics = node_metrics(yg, yn, pred, tc, pred_count)
                loc_rows.append(
                    {
                        "mode": mode,
                        "horizon": horizon,
                        "evidence_span_epochs": evidence_span,
                        "decoder": decoder,
                        "node_method": node_method,
                        "count_method": count_method,
                        "node_threshold": node_threshold,
                        "persistence_fraction": persistence_fraction,
                        "graph_gating": graph_gating,
                        "graph_method": gate_method if graph_gating else "",
                        "graph_threshold": gate_threshold if graph_gating else "",
                        **metrics,
                    }
                )

            # CountTopK.
            for node_method in node_methods:
                n_end, node_score = cache[f"node::{node_method}"]
                if not np.array_equal(n_end, endpoints):
                    raise RuntimeError("node endpoint mismatch")
                for count_method in count_methods:
                    c_end, pred_count = cache[f"count::{count_method}"]
                    if not np.array_equal(c_end, endpoints):
                        raise RuntimeError("count endpoint mismatch")
                    pred = decode_count_topk(node_score, pred_count)
                    add_localization_row(
                        pred,
                        decoder="CountTopK",
                        node_method=node_method,
                        count_method=count_method,
                    )
                    gated = pred.copy()
                    gated[~graph_gate] = 0
                    add_localization_row(
                        gated,
                        decoder="CountTopK",
                        node_method=node_method,
                        count_method=count_method,
                        graph_gating=True,
                    )

            # NodeThreshold.
            for node_method in node_methods:
                _, node_score = cache[f"node::{node_method}"]
                for threshold in NODE_THRESHOLDS:
                    pred = (node_score >= threshold).astype(np.int8)
                    add_localization_row(
                        pred,
                        decoder="NodeThreshold",
                        node_method=node_method,
                        node_threshold=threshold,
                    )
                    gated = pred.copy()
                    gated[~graph_gate] = 0
                    add_localization_row(
                        gated,
                        decoder="NodeThreshold",
                        node_method=node_method,
                        node_threshold=threshold,
                        graph_gating=True,
                    )

            # Persistence of the original frozen node threshold.
            _, vote = cache["node::base_threshold_vote"]
            for fraction in PERSISTENCE_FRACTIONS:
                pred = (vote >= fraction - 1e-12).astype(np.int8)
                add_localization_row(
                    pred,
                    decoder="NodePersistence",
                    node_method="base_threshold_vote",
                    persistence_fraction=fraction,
                )
                gated = pred.copy()
                gated[~graph_gate] = 0
                add_localization_row(
                    gated,
                    decoder="NodePersistence",
                    node_method="base_threshold_vote",
                    persistence_fraction=fraction,
                    graph_gating=True,
                )

            print(
                f"completed mode={mode} H={horizon} "
                f"samples={endpoints.size}",
                flush=True,
            )

    # Select alert policies globally.
    selected_alert_010 = choose_best_alert(alert_rows, 0.10)
    selected_alert_008 = choose_best_alert(alert_rows, 0.08)

    # Shortlist isolation policies by safety constraints, then compute
    # operational persistence/streak metrics for the shortlist.
    shortlist, had_feasible = select_isolation_shortlist(loc_rows, limit=25)
    enriched_shortlist: list[dict[str, Any]] = []

    for row in shortlist:
        mode = str(row["mode"])
        horizon = int(row["horizon"])
        segments = mode_segments(
            segments_dense, mode, args.deoverlap_stride
        )
        cache = per_configuration_cache[(mode, horizon)]
        endpoints, pred = reconstruct_policy_predictions(
            row,
            data,
            segments,
            horizon,
            mode,
            args.deoverlap_stride,
            cache,
        )
        op = operational_metrics(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            data.y_node[endpoints],
            pred,
        )
        enriched = dict(row)
        enriched.update(op)
        enriched["isolation_constraints_satisfied"] = bool(
            float(row["attack_node_precision"]) >= 0.95
            and float(row["normal_false_isolation_rate"]) <= 0.05
            and float(row["attack_isolation_coverage"]) >= 0.50
            and float(
                op["normal_run_persistent_false_isolation_fraction"]
            )
            <= 0.01
        )
        enriched_shortlist.append(enriched)

    feasible_enriched = [
        r for r in enriched_shortlist if r["isolation_constraints_satisfied"]
    ]
    if feasible_enriched:
        selected_isolation = max(
            feasible_enriched,
            key=lambda r: (
                float(r["attack_exact_localization"]),
                float(r["stable_attack_exact_point_rate"]),
                float(r["attack_node_f1"]),
                -float(r["normal_false_isolation_rate"]),
            ),
        )
        isolation_status = "FEASIBLE_POLICY_SELECTED"
    else:
        selected_isolation = min(
            enriched_shortlist,
            key=lambda r: (
                safety_distance(r),
                -float(r["attack_exact_localization"]),
            ),
        )
        isolation_status = "NO_POLICY_MET_ALL_SAFETY_CONSTRAINTS"

    write_csv(args.output_dir / "alert_candidates.csv", alert_rows)
    write_csv(args.output_dir / "localization_candidates.csv", loc_rows)
    write_csv(
        args.output_dir / "isolation_shortlist_operational.csv",
        enriched_shortlist,
    )

    preflight_report = json.loads(
        (args.preflight_dir / "preflight_report.json").read_text(
            encoding="utf-8"
        )
    )

    provenance = {
        "validation_predictions": str(args.validation_predictions),
        "validation_predictions_sha256": sha256_file(
            args.validation_predictions
        ),
        "thresholds": str(args.thresholds),
        "thresholds_sha256": sha256_file(args.thresholds),
        "preflight_report_sha256": sha256_file(
            args.preflight_dir / "preflight_report.json"
        ),
        "checkpoint_sha256": preflight_report["provenance"][
            "checkpoint_sha256"
        ],
        "selection_split": "validation",
        "development_test_accessed": False,
    }

    selected_policy = {
        "designation": (
            "V4-A3.11 Causal Temporal Aggregation and "
            "Precision-First Operational Analysis"
        ),
        "selection_split": "validation",
        "development_test_accessed": False,
        "stable_streak_decisions": STABLE_STREAK,
        "alert_policy_fpr_cap_0_10": selected_alert_010,
        "alert_policy_sensitivity_fpr_cap_0_08": selected_alert_008,
        "confirmed_isolation_status": isolation_status,
        "confirmed_isolation_policy": selected_isolation,
        "confirmed_isolation_constraints": {
            "attack_node_precision_min": 0.95,
            "normal_window_false_isolation_max": 0.05,
            "normal_run_persistent_false_isolation_max": 0.01,
            "attack_isolation_coverage_min": 0.50,
        },
        "provenance": provenance,
        "notes": [
            "Alert and confirmed-isolation policies are distinct.",
            "No development-test prediction file was accepted by this script.",
            "Count softmax logits are unavailable; mean probability and "
            "mean log-probability aggregation were evaluated instead.",
            "Max-probability methods are diagnostic candidates only.",
        ],
    }

    (args.output_dir / "selected_validation_policy.json").write_text(
        json.dumps(jsonable(selected_policy), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "status": "PASS",
        "alert_candidate_count": len(alert_rows),
        "localization_candidate_count": len(loc_rows),
        "isolation_shortlist_count": len(enriched_shortlist),
        "had_window_level_feasible_isolation_candidates": had_feasible,
        "confirmed_isolation_status": isolation_status,
        "selected_alert_fpr_cap_0_10": selected_alert_010,
        "selected_alert_fpr_cap_0_08": selected_alert_008,
        "selected_confirmed_isolation": selected_isolation,
        "provenance": provenance,
    }
    (args.output_dir / "validation_search_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    policy_sha = sha256_file(
        args.output_dir / "selected_validation_policy.json"
    )
    lock = {
        "status": "VALIDATION_POLICY_FROZEN",
        "selection_split": "validation",
        "development_test_accessed": False,
        "selected_validation_policy_sha256": policy_sha,
        "validation_predictions_sha256": provenance[
            "validation_predictions_sha256"
        ],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
    }
    (args.output_dir / "VALIDATION_POLICY_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "V4_A3_11_VALIDATION_SEARCH_PASS").write_text(
        "V4_A3_11_VALIDATION_SEARCH_PASS\n",
        encoding="utf-8",
    )

    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    print("V4_A3_11_VALIDATION_SEARCH_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
