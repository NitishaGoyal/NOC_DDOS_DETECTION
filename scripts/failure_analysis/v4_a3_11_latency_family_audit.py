#!/usr/bin/env python3
"""
Freeze and operationally audit the predeclared V4-A3.11 latency policy family.

Validation only. This script does not accept a development-test archive.

Policies audited:
  P0_FAST_ALERT:
      deoverlap H=4, graph mean-logit, threshold 0.33
  P1_LOW_LATENCY_CANDIDATE:
      deoverlap H=16, node mean-logit, threshold 0.675, no graph gate
  P2_BALANCED_CONFIRMED:
      deoverlap H=32, node mean-logit, threshold 0.675, no graph gate
  P3_MAX_SAFETY:
      deoverlap H=64, node mean-logit threshold 0.625,
      graph mean-logit gate threshold 0.32

The script imports tested utilities from the installed
v4_a3_11_validation_search.py implementation, recomputes each policy from the
frozen validation predictions, calculates run-level persistence metrics, and
writes a locked latency-policy family.

No policy is changed based on development-test behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("a3_11_validation_search", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_boundaries(run_index: np.ndarray) -> list[tuple[int, int]]:
    r = np.asarray(run_index)
    if r.size == 0:
        return []
    starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    ends = np.r_[starts[1:], r.size]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def longest_true_streak(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def stable_mask(values: np.ndarray, streak: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.int8)
    out = np.zeros_like(v, dtype=bool)
    if v.size < streak:
        return out
    c = np.cumsum(np.r_[0, v])
    sums = c[streak:] - c[:-streak]
    out[streak - 1:] = sums == streak
    return out


def graph_operational_metrics(
    run_index: np.ndarray,
    y_graph: np.ndarray,
    pred_graph: np.ndarray,
) -> dict[str, Any]:
    runs = np.asarray(run_index)
    y = np.asarray(y_graph, dtype=np.int8)
    p = np.asarray(pred_graph, dtype=np.int8)

    normal_runs = 0
    normal_ever_alarm = 0
    normal_persistent_alarm = 0
    max_false_alarm_streak = 0
    false_streaks: list[int] = []

    attack_runs = 0
    attack_first_alert_delays: list[int] = []
    attack_first_stable_delays: list[int] = []
    attack_stable_coverage = 0

    for start, end in run_boundaries(runs):
        yr = y[start:end]
        pr = p[start:end]
        is_attack_run = bool(np.any(yr == 1))

        if not is_attack_run:
            normal_runs += 1
            flags = pr == 1
            if np.any(flags):
                normal_ever_alarm += 1
            longest = longest_true_streak(flags)
            max_false_alarm_streak = max(max_false_alarm_streak, longest)
            if longest >= STABLE_STREAK:
                normal_persistent_alarm += 1

            current = 0
            for flag in flags:
                if flag:
                    current += 1
                elif current:
                    false_streaks.append(current)
                    current = 0
            if current:
                false_streaks.append(current)
        else:
            attack_runs += 1
            attack_positions = np.flatnonzero(yr == 1)
            if attack_positions.size == 0:
                continue
            onset = int(attack_positions[0])
            alert_positions = np.flatnonzero((pr[onset:] == 1) & (yr[onset:] == 1))
            if alert_positions.size:
                attack_first_alert_delays.append(int(alert_positions[0]))

            correct_alert = (pr[onset:] == 1) & (yr[onset:] == 1)
            stable = stable_mask(correct_alert, STABLE_STREAK)
            stable_positions = np.flatnonzero(stable)
            if stable_positions.size:
                attack_first_stable_delays.append(int(stable_positions[0]))
                attack_stable_coverage += 1

    return {
        "normal_run_count": normal_runs,
        "normal_run_ever_alarmed_fraction": (
            normal_ever_alarm / normal_runs if normal_runs else 0.0
        ),
        "normal_run_persistent_false_alarm_fraction": (
            normal_persistent_alarm / normal_runs if normal_runs else 0.0
        ),
        "maximum_false_alarm_streak": max_false_alarm_streak,
        "mean_false_alarm_streak_length": (
            float(np.mean(false_streaks)) if false_streaks else 0.0
        ),
        "attack_run_count": attack_runs,
        "attack_run_first_alert_coverage": (
            len(attack_first_alert_delays) / attack_runs if attack_runs else 0.0
        ),
        "attack_run_stable_alert_coverage": (
            attack_stable_coverage / attack_runs if attack_runs else 0.0
        ),
        "median_time_to_first_alert_decisions": (
            float(np.median(attack_first_alert_delays))
            if attack_first_alert_delays else math.nan
        ),
        "median_time_to_stable_alert_decisions": (
            float(np.median(attack_first_stable_delays))
            if attack_first_stable_delays else math.nan
        ),
    }


def build_cache(module, data, segments, horizon: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    graph_logit = module.logit(data.graph_prob)
    node_logit = module.logit(data.node_prob)

    endpoints, graph_agg = module.aggregate_array(
        graph_logit,
        segments,
        horizon,
        module.uniform_weights(horizon),
    )
    _, node_agg = module.aggregate_array(
        node_logit,
        segments,
        horizon,
        module.uniform_weights(horizon),
    )

    return {
        "graph_mean_logit": (endpoints, module.sigmoid(graph_agg)),
        "node_mean_logit": (endpoints, module.sigmoid(node_agg)),
    }


def localization_policy(
    module,
    data,
    segments,
    horizon: int,
    node_threshold: float,
    graph_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache = build_cache(module, data, segments, horizon)
    endpoints, node_score = cache["node_mean_logit"]
    pred = (node_score >= node_threshold).astype(np.int8)

    if graph_threshold is not None:
        graph_endpoints, graph_score = cache["graph_mean_logit"]
        if not np.array_equal(graph_endpoints, endpoints):
            raise RuntimeError("graph/node endpoint mismatch")
        pred[graph_score < graph_threshold] = 0

    metrics = module.node_metrics(
        data.y_graph[endpoints],
        data.y_node[endpoints],
        pred,
        data.attacker_count[endpoints],
        np.sum(pred, axis=1).astype(np.int64),
    )
    metrics.update(
        module.operational_metrics(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            data.y_node[endpoints],
            pred,
        )
    )
    return endpoints, pred, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-search-script", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--validation-search-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deoverlap-stride", type=int, default=8)
    parser.add_argument("--cycles-per-epoch", type=int, default=1000)
    args = parser.parse_args()

    required = [
        args.validation_search_script,
        args.data_dir,
        args.validation_predictions,
        args.validation_search_dir / "V4_A3_11_VALIDATION_SEARCH_PASS",
        args.validation_search_dir / "VALIDATION_POLICY_LOCK.json",
        args.validation_search_dir / "selected_validation_policy.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("LATENCY FAMILY AUDIT FAIL: missing required paths", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(f"LATENCY FAMILY AUDIT FAIL: output exists: {args.output_dir}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True)

    module = load_module(args.validation_search_script)
    data = module.prepare_data(args.validation_predictions, args.data_dir)
    dense_segments = module.make_segments(data)
    segments = module.mode_segments(
        dense_segments, "deoverlap", args.deoverlap_stride
    )

    policies = {
        "P0_FAST_ALERT": {
            "role": "alert",
            "mode": "deoverlap",
            "horizon": 4,
            "evidence_span_epochs": 32,
            "graph_method": "mean_logit",
            "graph_threshold": 0.33,
        },
        "P1_LOW_LATENCY_CANDIDATE": {
            "role": "candidate_localization",
            "mode": "deoverlap",
            "horizon": 16,
            "evidence_span_epochs": 128,
            "decoder": "NodeThreshold",
            "node_method": "mean_logit",
            "node_threshold": 0.675,
            "graph_gating": False,
        },
        "P2_BALANCED_CONFIRMED": {
            "role": "confirmed_isolation_candidate",
            "mode": "deoverlap",
            "horizon": 32,
            "evidence_span_epochs": 256,
            "decoder": "NodeThreshold",
            "node_method": "mean_logit",
            "node_threshold": 0.675,
            "graph_gating": False,
        },
        "P3_MAX_SAFETY": {
            "role": "maximum_safety_fallback",
            "mode": "deoverlap",
            "horizon": 64,
            "evidence_span_epochs": 512,
            "decoder": "NodeThreshold",
            "node_method": "mean_logit",
            "node_threshold": 0.625,
            "graph_gating": True,
            "graph_method": "mean_logit",
            "graph_threshold": 0.32,
        },
    }

    results: dict[str, Any] = {}

    # Alert policy.
    alert = policies["P0_FAST_ALERT"]
    cache = build_cache(module, data, segments, alert["horizon"])
    endpoints, graph_score = cache["graph_mean_logit"]
    graph_pred = (graph_score >= alert["graph_threshold"]).astype(np.int8)
    alert_metrics = module.binary_metrics(
        data.y_graph[endpoints],
        graph_pred,
    )
    alert_metrics.update(
        graph_operational_metrics(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            graph_pred,
        )
    )
    results["P0_FAST_ALERT"] = {
        "policy": alert,
        "metrics": alert_metrics,
    }

    # Localization policies.
    for name in (
        "P1_LOW_LATENCY_CANDIDATE",
        "P2_BALANCED_CONFIRMED",
        "P3_MAX_SAFETY",
    ):
        policy = policies[name]
        graph_threshold = (
            float(policy["graph_threshold"])
            if policy.get("graph_gating", False)
            else None
        )
        _, _, metrics = localization_policy(
            module,
            data,
            segments,
            int(policy["horizon"]),
            float(policy["node_threshold"]),
            graph_threshold,
        )
        results[name] = {
            "policy": policy,
            "metrics": metrics,
        }

    # Predeclared operational gates.
    p2 = results["P2_BALANCED_CONFIRMED"]["metrics"]
    p3 = results["P3_MAX_SAFETY"]["metrics"]

    p2_gate = {
        "attack_node_precision_at_least_0_95": (
            p2["attack_node_precision"] >= 0.95
        ),
        "normal_window_false_isolation_at_most_0_05": (
            p2["normal_false_isolation_rate"] <= 0.05
        ),
        "normal_run_persistent_false_isolation_at_most_0_01": (
            p2["normal_run_persistent_false_isolation_fraction"] <= 0.01
        ),
        "attack_isolation_coverage_at_least_0_50": (
            p2["attack_isolation_coverage"] >= 0.50
        ),
        "stable_attack_exact_at_least_0_75": (
            p2["stable_attack_exact_point_rate"] >= 0.75
        ),
    }
    p2_pass = all(p2_gate.values())

    p3_gate = {
        "attack_node_precision_at_least_0_95": (
            p3["attack_node_precision"] >= 0.95
        ),
        "normal_window_false_isolation_at_most_0_05": (
            p3["normal_false_isolation_rate"] <= 0.05
        ),
        "normal_run_persistent_false_isolation_at_most_0_01": (
            p3["normal_run_persistent_false_isolation_fraction"] <= 0.01
        ),
        "attack_isolation_coverage_at_least_0_50": (
            p3["attack_isolation_coverage"] >= 0.50
        ),
        "stable_attack_exact_at_least_0_80": (
            p3["stable_attack_exact_point_rate"] >= 0.80
        ),
    }
    p3_pass = all(p3_gate.values())

    for name, item in results.items():
        horizon = int(item["policy"]["horizon"])
        evidence_epochs = int(item["policy"]["evidence_span_epochs"])
        item["latency"] = {
            "cycles_per_epoch": args.cycles_per_epoch,
            "evidence_cycles": evidence_epochs * args.cycles_per_epoch,
            "decision_interval_cycles": (
                args.deoverlap_stride * args.cycles_per_epoch
            ),
            "four_decision_stability_extra_cycles": (
                (STABLE_STREAK - 1)
                * args.deoverlap_stride
                * args.cycles_per_epoch
            ),
            "minimum_stable_confirmation_cycles": (
                evidence_epochs * args.cycles_per_epoch
                + (STABLE_STREAK - 1)
                * args.deoverlap_stride
                * args.cycles_per_epoch
            ),
        }

    source_lock = json.loads(
        (args.validation_search_dir / "VALIDATION_POLICY_LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    provenance = {
        "validation_predictions": str(args.validation_predictions),
        "validation_predictions_sha256": sha256_file(
            args.validation_predictions
        ),
        "validation_search_script": str(args.validation_search_script),
        "validation_search_script_sha256": sha256_file(
            args.validation_search_script
        ),
        "source_validation_policy_lock": source_lock,
        "selection_split": "validation",
        "development_test_accessed": False,
    }

    report = {
        "status": "PASS",
        "designation": "V4-A3.11 latency-tier operational audit",
        "stable_streak_decisions": STABLE_STREAK,
        "policies": results,
        "balanced_confirmed_gate": p2_gate,
        "balanced_confirmed_gate_pass": p2_pass,
        "max_safety_gate": p3_gate,
        "max_safety_gate_pass": p3_pass,
        "recommended_transfer_status": (
            "READY_TO_FREEZE_AND_TRANSFER"
            if p2_pass and p3_pass
            else "HOLD_BEFORE_TEST_TRANSFER"
        ),
        "provenance": provenance,
    }

    report_path = args.output_dir / "latency_policy_family_validation.json"
    report_path.write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    policy_package = {
        "designation": "V4-A3.11 frozen latency policy family",
        "selection_split": "validation",
        "development_test_accessed": False,
        "stable_streak_decisions": STABLE_STREAK,
        "policies": policies,
        "balanced_confirmed_gate_pass": p2_pass,
        "max_safety_gate_pass": p3_pass,
        "recommended_transfer_status": report[
            "recommended_transfer_status"
        ],
        "provenance": provenance,
    }
    policy_path = args.output_dir / "FROZEN_LATENCY_POLICY_FAMILY.json"
    policy_path.write_text(
        json.dumps(policy_package, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    policy_sha = sha256_file(policy_path)
    lock = {
        "status": "LATENCY_POLICY_FAMILY_FROZEN",
        "selection_split": "validation",
        "development_test_accessed": False,
        "frozen_latency_policy_family_sha256": policy_sha,
        "validation_predictions_sha256": provenance[
            "validation_predictions_sha256"
        ],
        "checkpoint_sha256": source_lock["checkpoint_sha256"],
        "ready_to_transfer": report[
            "recommended_transfer_status"
        ] == "READY_TO_FREEZE_AND_TRANSFER",
    }
    (args.output_dir / "LATENCY_POLICY_FAMILY_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    marker_name = (
        "V4_A3_11_LATENCY_FAMILY_PASS"
        if lock["ready_to_transfer"]
        else "V4_A3_11_LATENCY_FAMILY_HOLD"
    )
    (args.output_dir / marker_name).write_text(
        marker_name + "\n",
        encoding="utf-8",
    )

    print(json.dumps(jsonable(report), indent=2, sort_keys=True))
    print(marker_name)
    return 0 if lock["ready_to_transfer"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
