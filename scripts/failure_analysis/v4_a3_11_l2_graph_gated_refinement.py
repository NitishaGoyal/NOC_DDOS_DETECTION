#!/usr/bin/env python3
"""
V4-A3.11-L2
Run-Safe Graph-Gated Isolation Refinement

Validation-only search over shorter graph-gated isolation policies.

This script:
  * accepts only the frozen A3 validation prediction archive
  * requires the completed A3.11 validation search and latency audit
  * never reads the development-test prediction archive
  * evaluates H={16,32} de-overlapped mean-logit aggregation
  * searches bounded node and graph thresholds
  * evaluates graph gating plus confirmation/persistence rules
  * selects the shortest run-safe policy satisfying predeclared gates
  * writes a frozen policy package for later one-time test transfer

No model training occurs.
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


HORIZONS = (16, 32)
NODE_THRESHOLDS = (0.625, 0.650, 0.675, 0.700, 0.725, 0.750)
GRAPH_THRESHOLDS = (0.250, 0.275, 0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450)
PERSISTENCE_RULES = (
    "none",
    "same_set_2",
    "same_set_3",
    "same_set_4",
    "router_3_of_4",
)
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


def same_set_persistence(
    pred: np.ndarray,
    run_index: np.ndarray,
    required: int,
) -> np.ndarray:
    """
    Emit a non-empty set only when the identical non-empty set has appeared
    for `required` consecutive decisions within the same run.
    """
    p = np.asarray(pred, dtype=np.int8)
    out = np.zeros_like(p, dtype=np.int8)

    for start, end in run_boundaries(run_index):
        length = end - start
        if length < required:
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
    """
    Emit each router when it was selected in at least 3 of the latest 4
    decisions within the same run.
    """
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
    if rule == "none":
        return np.asarray(pred, dtype=np.int8)
    if rule.startswith("same_set_"):
        required = int(rule.rsplit("_", 1)[1])
        return same_set_persistence(pred, run_index, required)
    if rule == "router_3_of_4":
        return router_three_of_four(pred, run_index)
    raise ValueError(f"unknown persistence rule: {rule}")


def persistence_extra_decisions(rule: str) -> int:
    if rule == "none":
        return 0
    if rule.startswith("same_set_"):
        return int(rule.rsplit("_", 1)[1]) - 1
    if rule == "router_3_of_4":
        return 3
    raise ValueError(rule)


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

        longest = 0
        current = 0
        for flag in flags:
            if flag:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

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


def candidate_passes(row: dict[str, Any]) -> bool:
    return bool(
        row["attack_node_precision"] >= 0.95
        and row["attack_exact_localization"] >= 0.75
        and row["stable_attack_exact_point_rate"] >= 0.75
        and row["attack_isolation_coverage"] >= 0.50
        and row["normal_false_isolation_rate"] <= 0.05
        and row["normal_run_persistent_false_isolation_fraction"] <= 0.01
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-search-script", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--validation-search-dir", type=Path, required=True)
    parser.add_argument("--latency-audit-dir", type=Path, required=True)
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
        args.latency_audit_dir / "V4_A3_11_LATENCY_FAMILY_HOLD",
        args.latency_audit_dir / "LATENCY_POLICY_FAMILY_LOCK.json",
        args.latency_audit_dir / "latency_policy_family_validation.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("A3.11-L2 FAIL: missing required paths", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(f"A3.11-L2 FAIL: output exists: {args.output_dir}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True)

    module = load_module(args.validation_search_script)
    data = module.prepare_data(args.validation_predictions, args.data_dir)
    dense_segments = module.make_segments(data)
    segments = module.mode_segments(
        dense_segments, "deoverlap", args.deoverlap_stride
    )

    graph_logit = module.logit(data.graph_prob)
    node_logit = module.logit(data.node_prob)

    candidates: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}

    for horizon in HORIZONS:
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

        graph_score = module.sigmoid(graph_agg)
        node_score = module.sigmoid(node_agg)

        y_graph = data.y_graph[endpoints]
        y_node = data.y_node[endpoints]
        true_count = data.attacker_count[endpoints]
        run_index = data.run_index[endpoints]

        evidence_span_epochs = horizon * args.deoverlap_stride

        for node_threshold in NODE_THRESHOLDS:
            raw_node_pred = (node_score >= node_threshold).astype(np.int8)

            for graph_threshold in GRAPH_THRESHOLDS:
                graph_gate = graph_score >= graph_threshold
                gated_pred = raw_node_pred.copy()
                gated_pred[~graph_gate] = 0

                gate_attack_open_rate = float(
                    np.mean(graph_gate[y_graph == 1])
                )
                gate_normal_open_rate = float(
                    np.mean(graph_gate[y_graph == 0])
                )

                for rule in PERSISTENCE_RULES:
                    final_pred = apply_persistence(
                        gated_pred,
                        run_index,
                        rule,
                    )

                    metrics = module.node_metrics(
                        y_graph,
                        y_node,
                        final_pred,
                        true_count,
                        np.sum(final_pred, axis=1).astype(np.int64),
                    )
                    operational = module.operational_metrics(
                        run_index,
                        y_graph,
                        y_node,
                        final_pred,
                    )

                    extra_decisions = persistence_extra_decisions(rule)
                    decision_interval_cycles = (
                        args.deoverlap_stride * args.cycles_per_epoch
                    )
                    evidence_cycles = (
                        evidence_span_epochs * args.cycles_per_epoch
                    )
                    confirmation_extra_cycles = (
                        extra_decisions * decision_interval_cycles
                    )
                    minimum_confirmation_cycles = (
                        evidence_cycles + confirmation_extra_cycles
                    )

                    row: dict[str, Any] = {
                        "mode": "deoverlap",
                        "horizon": horizon,
                        "evidence_span_epochs": evidence_span_epochs,
                        "node_method": "mean_logit",
                        "node_threshold": node_threshold,
                        "graph_method": "mean_logit",
                        "graph_threshold": graph_threshold,
                        "graph_gating": True,
                        "persistence_rule": rule,
                        "graph_gate_attack_open_rate": gate_attack_open_rate,
                        "graph_gate_normal_open_rate": gate_normal_open_rate,
                        "cycles_per_epoch": args.cycles_per_epoch,
                        "decision_interval_cycles": decision_interval_cycles,
                        "evidence_cycles": evidence_cycles,
                        "confirmation_extra_decisions": extra_decisions,
                        "confirmation_extra_cycles": confirmation_extra_cycles,
                        "minimum_confirmation_cycles": minimum_confirmation_cycles,
                        **metrics,
                        **operational,
                    }
                    row["all_gates_pass"] = candidate_passes(row)
                    candidates.append(row)

                    key = (
                        horizon,
                        node_threshold,
                        graph_threshold,
                        rule,
                    )
                    prediction_cache[key] = (endpoints, final_pred)

        print(
            f"completed H={horizon}; cumulative_candidates={len(candidates)}",
            flush=True,
        )

    feasible = [row for row in candidates if row["all_gates_pass"]]

    selected: dict[str, Any] | None
    if feasible:
        # Shortest valid operational latency first, then strongest safety and
        # localization performance.
        selected = min(
            feasible,
            key=lambda row: (
                int(row["minimum_confirmation_cycles"]),
                -float(row["stable_attack_exact_point_rate"]),
                -float(row["attack_exact_localization"]),
                -float(row["attack_node_precision"]),
                -float(row["attack_isolation_coverage"]),
                float(row["normal_false_isolation_rate"]),
            ),
        )
        selection_status = "RUN_SAFE_SHORTER_POLICY_SELECTED"
    else:
        selected = None
        selection_status = "NO_H16_OR_H32_POLICY_MET_ALL_GATES"

    write_csv(args.output_dir / "l2_all_candidates.csv", candidates)

    # Also write the best candidates at each horizon for easier inspection.
    frontier: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        rows = [row for row in candidates if row["horizon"] == horizon]
        rows.sort(
            key=lambda row: (
                not bool(row["all_gates_pass"]),
                int(row["minimum_confirmation_cycles"]),
                -float(row["stable_attack_exact_point_rate"]),
                -float(row["attack_exact_localization"]),
                -float(row["attack_node_precision"]),
            )
        )
        frontier.extend(rows[:20])
    write_csv(args.output_dir / "l2_shortlist.csv", frontier)

    hard_rows: list[dict[str, Any]] = []
    if selected is not None:
        key = (
            int(selected["horizon"]),
            float(selected["node_threshold"]),
            float(selected["graph_threshold"]),
            str(selected["persistence_rule"]),
        )
        endpoints, pred = prediction_cache[key]
        hard_rows = hard_normal_run_rows(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            pred,
        )
    write_csv(args.output_dir / "selected_policy_hard_normal_runs.csv", hard_rows)

    validation_lock = json.loads(
        (args.validation_search_dir / "VALIDATION_POLICY_LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    latency_lock = json.loads(
        (args.latency_audit_dir / "LATENCY_POLICY_FAMILY_LOCK.json").read_text(
            encoding="utf-8"
        )
    )

    provenance = {
        "selection_split": "validation",
        "development_test_accessed": False,
        "validation_predictions": str(args.validation_predictions),
        "validation_predictions_sha256": sha256_file(
            args.validation_predictions
        ),
        "validation_search_script": str(args.validation_search_script),
        "validation_search_script_sha256": sha256_file(
            args.validation_search_script
        ),
        "source_validation_policy_lock": validation_lock,
        "source_latency_policy_lock": latency_lock,
    }

    gates = {
        "attack_node_precision_min": 0.95,
        "attack_exact_localization_min": 0.75,
        "stable_attack_exact_point_rate_min": 0.75,
        "attack_isolation_coverage_min": 0.50,
        "normal_window_false_isolation_max": 0.05,
        "normal_run_persistent_false_isolation_max": 0.01,
    }

    report = {
        "status": "PASS",
        "designation": (
            "V4-A3.11-L2 Run-Safe Graph-Gated Isolation Refinement"
        ),
        "selection_status": selection_status,
        "candidate_count": len(candidates),
        "feasible_candidate_count": len(feasible),
        "search_space": {
            "horizons": HORIZONS,
            "node_thresholds": NODE_THRESHOLDS,
            "graph_thresholds": GRAPH_THRESHOLDS,
            "persistence_rules": PERSISTENCE_RULES,
            "mode": "deoverlap",
            "node_method": "mean_logit",
            "graph_method": "mean_logit",
        },
        "selection_gates": gates,
        "selected_policy": selected,
        "selected_policy_hard_normal_run_count": len(hard_rows),
        "recommended_next_status": (
            "READY_TO_FREEZE_SHORTER_POLICY_AND_TRANSFER"
            if selected is not None
            else "ONLY_H64_REMAINS_VALIDATED_FOR_TRANSFER"
        ),
        "provenance": provenance,
    }

    report_path = args.output_dir / "l2_validation_report.json"
    report_path.write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    policy_package = {
        "designation": (
            "V4-A3.11-L2 frozen graph-gated isolation policy"
        ),
        "selection_split": "validation",
        "development_test_accessed": False,
        "selection_status": selection_status,
        "selected_policy": selected,
        "selection_gates": gates,
        "provenance": provenance,
    }
    policy_path = args.output_dir / "FROZEN_L2_POLICY.json"
    policy_path.write_text(
        json.dumps(jsonable(policy_package), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lock = {
        "status": (
            "L2_SHORTER_POLICY_FROZEN"
            if selected is not None
            else "L2_NO_SHORTER_POLICY"
        ),
        "selection_split": "validation",
        "development_test_accessed": False,
        "frozen_l2_policy_sha256": sha256_file(policy_path),
        "validation_predictions_sha256": provenance[
            "validation_predictions_sha256"
        ],
        "checkpoint_sha256": validation_lock["checkpoint_sha256"],
        "ready_to_transfer": selected is not None,
    }
    (args.output_dir / "L2_POLICY_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    marker = (
        "V4_A3_11_L2_PASS"
        if selected is not None
        else "V4_A3_11_L2_HOLD"
    )
    (args.output_dir / marker).write_text(marker + "\n", encoding="utf-8")

    print(json.dumps(jsonable(report), indent=2, sort_keys=True))
    print(marker)
    return 0 if selected is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
