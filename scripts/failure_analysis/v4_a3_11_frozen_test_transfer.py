#!/usr/bin/env python3
"""
V4-A3.11-T
One-Time Frozen Operational Policy Development-Test Transfer

This script performs no threshold, horizon, decoder, gate, or persistence
selection. It applies the already frozen validation policies exactly once to
the development-comparison test prediction archive.

Transferred policies:
  P0_FAST_ALERT
      From the frozen latency-family package.
  P1_L2_CANDIDATE
      From the frozen operational-policy package.
  P2_L2_CONFIRMED
      From the frozen operational-policy package.
  P3_MAX_SAFETY
      From the frozen latency-family package.

The script verifies:
  * frozen checkpoint provenance
  * policy package hashes and locks
  * development-test prediction lock/hash
  * no output overwrite
  * alignment through global_index
  * causal run-bounded aggregation

No policy is changed based on development-test results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


AUDIT_STABLE_STREAK = 4


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


def run_boundaries(run_index: np.ndarray) -> list[tuple[int, int]]:
    runs = np.asarray(run_index)
    if runs.size == 0:
        return []
    starts = np.flatnonzero(np.r_[True, runs[1:] != runs[:-1]])
    ends = np.r_[starts[1:], runs.size]
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
    out[streak - 1 :] = sums == streak
    return out


def same_set_persistence(
    pred: np.ndarray,
    run_index: np.ndarray,
    required: int,
) -> np.ndarray:
    p = np.asarray(pred, dtype=np.int8)
    out = np.zeros_like(p, dtype=np.int8)

    for start, end in run_boundaries(run_index):
        if end - start < required:
            continue
        for i in range(start + required - 1, end):
            window = p[i - required + 1 : i + 1]
            current = window[-1]
            if np.sum(current) == 0:
                continue
            if np.all(window == current):
                out[i] = current
    return out


def router_three_of_four(
    pred: np.ndarray,
    run_index: np.ndarray,
) -> np.ndarray:
    p = np.asarray(pred, dtype=np.int8)
    out = np.zeros_like(p, dtype=np.int8)
    for start, end in run_boundaries(run_index):
        if end - start < 4:
            continue
        csum = np.cumsum(
            np.vstack([np.zeros((1, p.shape[1]), dtype=np.int64), p[start:end]]),
            axis=0,
        )
        sums = csum[4:] - csum[:-4]
        out[start + 3 : end] = (sums >= 3).astype(np.int8)
    return out


def apply_persistence(
    pred: np.ndarray,
    run_index: np.ndarray,
    rule: str,
) -> np.ndarray:
    if rule in ("", "none", None):
        return np.asarray(pred, dtype=np.int8)
    if rule.startswith("same_set_"):
        required = int(rule.rsplit("_", 1)[1])
        return same_set_persistence(pred, run_index, required)
    if rule == "router_3_of_4":
        return router_three_of_four(pred, run_index)
    raise ValueError(f"unsupported persistence rule: {rule}")


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
    first_alert_delays: list[int] = []
    first_stable_alert_delays: list[int] = []
    stable_alert_runs = 0

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
            if longest >= AUDIT_STABLE_STREAK:
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
            if not attack_positions.size:
                continue
            onset = int(attack_positions[0])
            correct = (pr[onset:] == 1) & (yr[onset:] == 1)
            positions = np.flatnonzero(correct)
            if positions.size:
                first_alert_delays.append(int(positions[0]))
            stable = stable_mask(correct, AUDIT_STABLE_STREAK)
            stable_positions = np.flatnonzero(stable)
            if stable_positions.size:
                first_stable_alert_delays.append(int(stable_positions[0]))
                stable_alert_runs += 1

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
            len(first_alert_delays) / attack_runs if attack_runs else 0.0
        ),
        "attack_run_stable_alert_coverage": (
            stable_alert_runs / attack_runs if attack_runs else 0.0
        ),
        "median_time_to_first_alert_decisions": (
            float(np.median(first_alert_delays))
            if first_alert_delays else math.nan
        ),
        "median_time_to_stable_alert_decisions": (
            float(np.median(first_stable_alert_delays))
            if first_stable_alert_delays else math.nan
        ),
    }


def hard_normal_run_rows(
    run_index: np.ndarray,
    y_graph: np.ndarray,
    pred_node: np.ndarray,
) -> list[dict[str, Any]]:
    runs = np.asarray(run_index)
    yg = np.asarray(y_graph, dtype=np.int8)
    pred = np.asarray(pred_node, dtype=np.int8)
    rows: list[dict[str, Any]] = []

    for start, end in run_boundaries(runs):
        if np.any(yg[start:end] == 1):
            continue
        run_pred = pred[start:end]
        flags = np.sum(run_pred, axis=1) > 0
        if not np.any(flags):
            continue

        longest = longest_true_streak(flags)
        router_counts = np.sum(run_pred, axis=0).astype(int)
        top_router = int(np.argmax(router_counts))
        rows.append(
            {
                "run_index": int(runs[start]),
                "decision_count": int(end - start),
                "false_isolation_decisions": int(np.sum(flags)),
                "false_isolation_fraction": float(np.mean(flags)),
                "maximum_false_isolation_streak": int(longest),
                "top_false_router": top_router,
                "top_false_router_decisions": int(router_counts[top_router]),
                "false_router_assignments": int(np.sum(run_pred)),
            }
        )

    rows.sort(
        key=lambda row: (
            -row["maximum_false_isolation_streak"],
            -row["false_isolation_decisions"],
        )
    )
    return rows


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
            writer.writerow({key: row.get(key, "") for key in fields})


def aggregate_scores(
    module,
    data,
    dense_segments,
    horizon: int,
    deoverlap_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segments = module.mode_segments(
        dense_segments, "deoverlap", deoverlap_stride
    )
    graph_logit = module.logit(data.graph_prob)
    node_logit = module.logit(data.node_prob)

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
    if not np.array_equal(endpoints, node_endpoints):
        raise RuntimeError("graph/node endpoint mismatch")

    return endpoints, module.sigmoid(graph_agg), module.sigmoid(node_agg)


def evaluate_localization_policy(
    module,
    data,
    dense_segments,
    policy: dict[str, Any],
    deoverlap_stride: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    horizon = int(policy["horizon"])
    endpoints, graph_score, node_score = aggregate_scores(
        module,
        data,
        dense_segments,
        horizon,
        deoverlap_stride,
    )

    pred = (
        node_score >= float(policy["node_threshold"])
    ).astype(np.int8)

    if bool(policy.get("graph_gating", True)):
        graph_threshold = float(policy["graph_threshold"])
        pred[graph_score < graph_threshold] = 0

    pred = apply_persistence(
        pred,
        data.run_index[endpoints],
        str(policy.get("persistence_rule", "none")),
    )

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
    metrics["graph_gate_attack_open_rate"] = float(
        np.mean(graph_score[data.y_graph[endpoints] == 1]
                >= float(policy["graph_threshold"]))
    )
    metrics["graph_gate_normal_open_rate"] = float(
        np.mean(graph_score[data.y_graph[endpoints] == 0]
                >= float(policy["graph_threshold"]))
    )

    return endpoints, pred, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-search-script", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--test-metrics", type=Path, required=True)
    parser.add_argument("--test-lock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--operational-policy", type=Path, required=True)
    parser.add_argument("--operational-policy-lock", type=Path, required=True)
    parser.add_argument("--latency-policy", type=Path, required=True)
    parser.add_argument("--latency-policy-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deoverlap-stride", type=int, default=8)
    parser.add_argument("--cycles-per-epoch", type=int, default=1000)
    args = parser.parse_args()

    required = [
        args.validation_search_script,
        args.data_dir,
        args.test_predictions,
        args.test_metrics,
        args.test_lock,
        args.checkpoint,
        args.operational_policy,
        args.operational_policy_lock,
        args.latency_policy,
        args.latency_policy_lock,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("A3.11-T FAIL: missing required paths", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(f"A3.11-T FAIL: output exists: {args.output_dir}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True)

    operational = load_json(args.operational_policy)
    operational_lock = load_json(args.operational_policy_lock)
    latency = load_json(args.latency_policy)
    latency_lock = load_json(args.latency_policy_lock)
    test_lock = load_json(args.test_lock)

    checkpoint_sha = sha256_file(args.checkpoint)
    test_pred_sha = sha256_file(args.test_predictions)
    test_metrics_sha = sha256_file(args.test_metrics)
    operational_sha = sha256_file(args.operational_policy)
    latency_sha = sha256_file(args.latency_policy)

    failures: list[str] = []

    if operational_lock.get("status") != "OPERATIONAL_POLICY_FAMILY_FROZEN":
        failures.append("operational policy lock status is not frozen")
    if operational_lock.get("development_test_accessed") is not False:
        failures.append("operational policy says development test was accessed")
    if operational_lock.get("operational_policy_family_sha256") != operational_sha:
        failures.append("operational policy package hash mismatch")
    if operational_lock.get("checkpoint_sha256") != checkpoint_sha:
        failures.append("operational policy checkpoint hash mismatch")

    if latency_lock.get("status") != "LATENCY_POLICY_FAMILY_FROZEN":
        failures.append("latency policy lock status is not frozen")
    if latency_lock.get("frozen_latency_policy_family_sha256") != latency_sha:
        failures.append("latency policy package hash mismatch")
    if latency_lock.get("checkpoint_sha256") != checkpoint_sha:
        failures.append("latency policy checkpoint hash mismatch")

    if test_lock.get("checkpoint_sha256") != checkpoint_sha:
        failures.append("development-test lock checkpoint hash mismatch")
    if test_lock.get("test_predictions_sha256") != test_pred_sha:
        failures.append("development-test predictions hash mismatch")
    if test_lock.get("test_metrics_sha256") != test_metrics_sha:
        failures.append("development-test metrics hash mismatch")

    if failures:
        report = {"status": "FAIL", "failures": failures}
        (args.output_dir / "transfer_failure.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    module = load_module(args.validation_search_script)
    data = module.prepare_data(args.test_predictions, args.data_dir)
    dense_segments = module.make_segments(data)

    latency_policies = latency["policies"]
    operational_roles = operational["policy_roles"]

    p0 = latency_policies["P0_FAST_ALERT"]
    p3 = latency_policies["P3_MAX_SAFETY"]
    candidate = operational_roles["candidate_localization"]["policy"]
    confirmed = operational_roles["confirmed_isolation"]["policy"]

    results: dict[str, Any] = {}
    hard_normal_rows: list[dict[str, Any]] = []

    # P0 graph alert.
    p0_endpoints, p0_graph_score, _ = aggregate_scores(
        module,
        data,
        dense_segments,
        int(p0["horizon"]),
        args.deoverlap_stride,
    )
    p0_pred = (
        p0_graph_score >= float(p0["graph_threshold"])
    ).astype(np.int8)
    p0_metrics = module.binary_metrics(
        data.y_graph[p0_endpoints],
        p0_pred,
    )
    p0_metrics.update(
        graph_operational_metrics(
            data.run_index[p0_endpoints],
            data.y_graph[p0_endpoints],
            p0_pred,
        )
    )
    results["P0_FAST_ALERT"] = {
        "policy": p0,
        "metrics": p0_metrics,
    }

    # Candidate, confirmed, and max-safety localization.
    named_policies = {
        "P1_L2_CANDIDATE": candidate,
        "P2_L2_CONFIRMED": confirmed,
        "P3_MAX_SAFETY": p3,
    }

    for name, policy in named_policies.items():
        endpoints, pred, metrics = evaluate_localization_policy(
            module,
            data,
            dense_segments,
            policy,
            args.deoverlap_stride,
        )
        results[name] = {
            "policy": policy,
            "metrics": metrics,
        }

        rows = hard_normal_run_rows(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            pred,
        )
        for row in rows:
            row["policy_name"] = name
        hard_normal_rows.extend(rows)

    # Frozen gates are evaluated, not used for reselection.
    gate_definitions = {
        "P1_L2_CANDIDATE": {
            "attack_node_precision_min": 0.95,
            "attack_exact_localization_min": 0.75,
            "attack_isolation_coverage_min": 0.50,
        },
        "P2_L2_CONFIRMED": {
            "attack_node_precision_min": 0.95,
            "attack_exact_localization_min": 0.75,
            "stable_attack_exact_point_rate_min": 0.75,
            "attack_isolation_coverage_min": 0.50,
            "normal_false_isolation_rate_max": 0.05,
            "normal_run_persistent_false_isolation_fraction_max": 0.01,
        },
        "P3_MAX_SAFETY": {
            "attack_node_precision_min": 0.95,
            "stable_attack_exact_point_rate_min": 0.80,
            "attack_isolation_coverage_min": 0.50,
            "normal_false_isolation_rate_max": 0.05,
            "normal_run_persistent_false_isolation_fraction_max": 0.01,
        },
    }

    gate_results: dict[str, Any] = {}
    for name, definitions in gate_definitions.items():
        metrics = results[name]["metrics"]
        checks: dict[str, bool] = {}
        for key, value in definitions.items():
            if key.endswith("_min"):
                metric = key[:-4]
                checks[key] = float(metrics[metric]) >= float(value)
            elif key.endswith("_max"):
                metric = key[:-4]
                checks[key] = float(metrics[metric]) <= float(value)
            else:
                raise RuntimeError(f"bad gate key: {key}")
        gate_results[name] = {
            "checks": checks,
            "all_pass": all(checks.values()),
        }

    for item in results.values():
        policy = item["policy"]
        horizon = int(policy["horizon"])
        evidence_epochs = int(policy["evidence_span_epochs"])
        persistence_rule = str(policy.get("persistence_rule", "none"))
        if persistence_rule == "none":
            extra_decisions = 0
        elif persistence_rule.startswith("same_set_"):
            extra_decisions = int(persistence_rule.rsplit("_", 1)[1]) - 1
        elif persistence_rule == "router_3_of_4":
            extra_decisions = 3
        else:
            extra_decisions = 0

        decision_interval_cycles = args.deoverlap_stride * args.cycles_per_epoch
        item["latency"] = {
            "cycles_per_epoch": args.cycles_per_epoch,
            "decision_interval_cycles": decision_interval_cycles,
            "evidence_cycles": evidence_epochs * args.cycles_per_epoch,
            "persistence_extra_decisions": extra_decisions,
            "persistence_extra_cycles": extra_decisions * decision_interval_cycles,
            "minimum_policy_output_cycles": (
                evidence_epochs * args.cycles_per_epoch
                + extra_decisions * decision_interval_cycles
            ),
        }

    provenance = {
        "selection_split": "validation",
        "development_test_role": "development comparison; not publication holdout",
        "development_test_accessed_for_selection": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "test_predictions": str(args.test_predictions),
        "test_predictions_sha256": test_pred_sha,
        "test_metrics": str(args.test_metrics),
        "test_metrics_sha256": test_metrics_sha,
        "operational_policy": str(args.operational_policy),
        "operational_policy_sha256": operational_sha,
        "latency_policy": str(args.latency_policy),
        "latency_policy_sha256": latency_sha,
        "validation_search_script_sha256": sha256_file(
            args.validation_search_script
        ),
    }

    report = {
        "status": "PASS",
        "designation": (
            "V4-A3.11-T One-Time Frozen Operational Policy "
            "Development-Test Transfer"
        ),
        "policy_selection_performed_on_test": False,
        "results": results,
        "frozen_gate_results": gate_results,
        "provenance": provenance,
        "interpretation_rule": {
            "all_primary_policies_transfer": (
                "P1 candidate and P2 confirmed both pass their frozen gates."
            ),
            "only_max_safety_transfers": (
                "Shorter operational policy does not generalize; retain H64 "
                "as an upper-bound mitigation policy and treat latency as unresolved."
            ),
            "confirmed_policy_safety_failure": (
                "Do not authorize isolation with the H32 confirmed policy. "
                "Continue to A3.12-A3.15 diagnostics."
            ),
        },
    }

    report_path = args.output_dir / "frozen_policy_test_transfer_report.json"
    report_path.write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    write_csv(
        args.output_dir / "hard_normal_runs_by_policy.csv",
        hard_normal_rows,
    )

    summary_rows: list[dict[str, Any]] = []
    for name, item in results.items():
        row = {
            "policy_name": name,
            **item["policy"],
            **item["metrics"],
            **item["latency"],
        }
        if name in gate_results:
            row["frozen_gates_pass"] = gate_results[name]["all_pass"]
        summary_rows.append(row)
    write_csv(args.output_dir / "policy_test_summary.csv", summary_rows)

    report_sha = sha256_file(report_path)
    transfer_lock = {
        "status": "FROZEN_POLICY_TEST_TRANSFER_COMPLETE",
        "policy_selection_performed_on_test": False,
        "development_test_role": "development comparison; not publication holdout",
        "frozen_policy_test_transfer_report_sha256": report_sha,
        "test_predictions_sha256": test_pred_sha,
        "operational_policy_sha256": operational_sha,
        "latency_policy_sha256": latency_sha,
        "checkpoint_sha256": checkpoint_sha,
    }
    (args.output_dir / "FROZEN_POLICY_TEST_TRANSFER_LOCK.json").write_text(
        json.dumps(transfer_lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "V4_A3_11_FROZEN_TEST_TRANSFER_PASS").write_text(
        "V4_A3_11_FROZEN_TEST_TRANSFER_PASS\n",
        encoding="utf-8",
    )

    print(json.dumps(jsonable(report), indent=2, sort_keys=True))
    print("V4_A3_11_FROZEN_TEST_TRANSFER_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
