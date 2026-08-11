#!/usr/bin/env python3
"""Aggregate the complete 3-operator × 3-seed V5 P2-G3 matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any


OPERATORS = ("conv1d", "gcnconv", "graphconv")
SEEDS = (107, 117, 127)
ROLES = ("source", "transit", "victim", "path")

STAGE = "V5_P2_G3_ROLE_AWARE_MULTILABEL_MATRIX_AGGREGATION"
COMPLETE = f"{STAGE}_COMPLETE"
RUN_STAGE = "V5_P2_G3_ROLE_AWARE_MULTILABEL_SINGLE_RUN"
RUN_COMPLETE = f"{RUN_STAGE}_COMPLETE"

EXPECTED_G1_REPORT_SHA = "299f33cfbc55d82075a6992793540bdc3d237846b1315caab8361efeee8c2e7a"
EXPECTED_G2_REPORT_SHA = "66748c90e128e2e869e30b55b461a49383a5253ccc4b49f3fb72111221b95a74"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "standard_deviation": stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def extract_row(report: dict[str, Any]) -> dict[str, Any]:
    best = report["best_checkpoint"]
    selected_metrics = report["post_checkpoint_validation"]["selected_metrics"]
    threshold_results = report["post_checkpoint_validation"]["threshold_selection"]
    latency = report["cost"]["inference_latency_proxy"]
    operation = report["cost"]["operation_count"]
    row: dict[str, Any] = {
        "operator": report["operator"],
        "seed": report["seed"],
        "best_epoch": best["epoch"],
        "selection_score": best["selection_score"],
        "parameter_count": report["architecture"]["parameter_count"],
        "total_linear_macs": operation["total_linear_macs"],
        "graph_message_scalar_ops": operation["graph_message_scalar_ops"],
        "pooling_scalar_ops": operation["pooling_scalar_ops"],
        "peak_cuda_memory_allocated_bytes": report["cost"][
            "peak_cuda_memory_allocated_bytes"
        ],
        "inference_microseconds_per_item": latency["microseconds_per_item"],
        "total_training_seconds": report["training"]["total_elapsed_seconds"],
        "joint_exact_all_roles_overall": report[
            "post_checkpoint_validation"
        ]["joint_exact_all_four_roles"]["overall"],
        "joint_exact_all_roles_attack": report[
            "post_checkpoint_validation"
        ]["joint_exact_all_four_roles"]["attack"],
        "joint_exact_all_roles_control": report[
            "post_checkpoint_validation"
        ]["joint_exact_all_four_roles"]["control"],
    }
    for role in ROLES:
        checkpoint_role = best["validation_metrics_at_selection"]["roles"][role]
        selected = threshold_results[role]["selected"]
        metrics = selected_metrics[role]
        thresholded = metrics["thresholded"]
        exact = metrics["exact_set"]
        row.update(
            {
                f"{role}_checkpoint_ap": checkpoint_role["average_precision"],
                f"{role}_checkpoint_auroc": checkpoint_role["auroc"],
                f"{role}_selected_threshold": selected["threshold"],
                f"{role}_threshold_objective": selected["objective"],
                f"{role}_node_accuracy": thresholded["accuracy"],
                f"{role}_node_balanced_accuracy": thresholded[
                    "balanced_accuracy"
                ],
                f"{role}_node_f1": thresholded["f1"],
                f"{role}_node_precision": thresholded["precision"],
                f"{role}_node_recall": thresholded["recall"],
                f"{role}_node_fpr": thresholded["fpr"],
                f"{role}_node_tnr": thresholded["tnr"],
                f"{role}_exact_set_overall": exact["overall"],
                f"{role}_exact_set_attack": exact["attack"],
                f"{role}_exact_set_control": exact["control"],
                f"{role}_exact_set_role_active": exact["role_active"],
            }
        )
    row["macro_role_node_f1"] = mean(
        [float(row[f"{role}_node_f1"]) for role in ROLES]
    )
    row["macro_role_exact_set_attack"] = mean(
        [float(row[f"{role}_exact_set_attack"]) for role in ROLES]
    )
    row["macro_role_checkpoint_ap"] = mean(
        [float(row[f"{role}_checkpoint_ap"]) for role in ROLES]
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--g1-aggregation-dir", type=Path, required=True)
    parser.add_argument("--g2-aggregation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reports_root = args.reports_root.expanduser().resolve()
    g1_dir = args.g1_aggregation_dir.expanduser().resolve()
    g2_dir = args.g2_aggregation_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"STOP: output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    g1_report_path = (
        g1_dir
        / "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION.json"
    )
    g1_lock_path = (
        g1_dir
        / "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION_LOCK.json"
    )
    g1_marker = (
        g1_dir
        / "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION_COMPLETE"
    )
    g2_report_path = (
        g2_dir
        / "V5_P2_G2_DIRECT_GRAPH_MATRIX_AGGREGATION.json"
    )
    g2_lock_path = (
        g2_dir
        / "V5_P2_G2_DIRECT_GRAPH_MATRIX_AGGREGATION_LOCK.json"
    )
    g2_marker = (
        g2_dir
        / "V5_P2_G2_DIRECT_GRAPH_MATRIX_AGGREGATION_COMPLETE"
    )
    for path in (
        g1_report_path,
        g1_lock_path,
        g1_marker,
        g2_report_path,
        g2_lock_path,
        g2_marker,
    ):
        if not path.is_file():
            raise SystemExit(f"STOP: missing frozen prerequisite: {path}")

    if sha256_file(g1_report_path) != EXPECTED_G1_REPORT_SHA:
        raise SystemExit("STOP: frozen G1 aggregation report SHA changed")
    if sha256_file(g2_report_path) != EXPECTED_G2_REPORT_SHA:
        raise SystemExit("STOP: frozen G2 aggregation report SHA changed")
    g1_lock = load_json(g1_lock_path)
    g2_lock = load_json(g2_lock_path)
    if g1_lock.get("report_sha256") != EXPECTED_G1_REPORT_SHA:
        raise SystemExit("STOP: frozen G1 lock/report mismatch")
    if g2_lock.get("report_sha256") != EXPECTED_G2_REPORT_SHA:
        raise SystemExit("STOP: frozen G2 lock/report mismatch")

    per_seed: list[dict[str, Any]] = []
    report_paths: dict[str, str] = {}
    for operator in OPERATORS:
        for seed in SEEDS:
            run_dir = reports_root / operator / f"seed_{seed}"
            report_path = run_dir / f"{RUN_STAGE}.json"
            lock_path = run_dir / f"{RUN_STAGE}_LOCK.json"
            marker = run_dir / RUN_COMPLETE
            for path in (report_path, lock_path, marker):
                if not path.is_file():
                    raise SystemExit(
                        f"STOP: incomplete G3 run operator={operator} "
                        f"seed={seed}: missing {path}"
                    )
            report = load_json(report_path)
            lock = load_json(lock_path)
            if report.get("status") != "COMPLETE":
                raise SystemExit(
                    f"STOP: G3 report not complete: {report_path}"
                )
            if report.get("operator") != operator or report.get("seed") != seed:
                raise SystemExit(
                    f"STOP: G3 run identity mismatch: {report_path}"
                )
            if lock.get("report_sha256") != sha256_file(report_path):
                raise SystemExit(
                    f"STOP: G3 run lock/report mismatch: {report_path}"
                )
            boundary = report.get("security_boundary", {})
            if boundary.get("p2_test_directory_enumerated") is not False:
                raise SystemExit("STOP: G3 test directory boundary violated")
            if boundary.get("p2_test_tensors_deserialized") is not False:
                raise SystemExit("STOP: G3 test tensor boundary violated")
            if boundary.get("architecture_selected") is not False:
                raise SystemExit(
                    "STOP: a G3 run unexpectedly selected architecture"
                )
            row = extract_row(report)
            per_seed.append(row)
            report_paths[f"{operator}/seed_{seed}"] = str(report_path)

    if len(per_seed) != 9:
        raise SystemExit(
            f"STOP: expected 9 complete G3 runs, found {len(per_seed)}"
        )

    summary_rows: list[dict[str, Any]] = []
    statistics: dict[str, Any] = {}
    summary_metrics = [
        "selection_score",
        "macro_role_checkpoint_ap",
        "macro_role_node_f1",
        "macro_role_exact_set_attack",
        "joint_exact_all_roles_attack",
        "joint_exact_all_roles_overall",
        "inference_microseconds_per_item",
        "total_training_seconds",
    ]
    for role in ROLES:
        summary_metrics.extend(
            [
                f"{role}_checkpoint_ap",
                f"{role}_checkpoint_auroc",
                f"{role}_selected_threshold",
                f"{role}_node_balanced_accuracy",
                f"{role}_node_f1",
                f"{role}_node_precision",
                f"{role}_node_recall",
                f"{role}_node_fpr",
                f"{role}_exact_set_attack",
                f"{role}_exact_set_control",
            ]
        )

    for operator in OPERATORS:
        rows = [row for row in per_seed if row["operator"] == operator]
        operator_stats = {
            metric: summarize([float(row[metric]) for row in rows])
            for metric in summary_metrics
        }
        operator_stats["parameter_count"] = int(rows[0]["parameter_count"])
        operator_stats["total_linear_macs"] = int(rows[0]["total_linear_macs"])
        operator_stats["graph_message_scalar_ops"] = int(
            rows[0]["graph_message_scalar_ops"]
        )
        operator_stats["pooling_scalar_ops"] = int(
            rows[0]["pooling_scalar_ops"]
        )
        statistics[operator] = operator_stats

        summary_row: dict[str, Any] = {
            "operator": operator,
            "macro_role_node_f1_mean": operator_stats[
                "macro_role_node_f1"
            ]["mean"],
            "macro_role_node_f1_std": operator_stats[
                "macro_role_node_f1"
            ]["standard_deviation"],
            "macro_role_exact_set_attack_mean": operator_stats[
                "macro_role_exact_set_attack"
            ]["mean"],
            "macro_role_exact_set_attack_std": operator_stats[
                "macro_role_exact_set_attack"
            ]["standard_deviation"],
            "joint_exact_all_roles_attack_mean": operator_stats[
                "joint_exact_all_roles_attack"
            ]["mean"],
            "joint_exact_all_roles_attack_std": operator_stats[
                "joint_exact_all_roles_attack"
            ]["standard_deviation"],
            "parameter_count": operator_stats["parameter_count"],
            "total_linear_macs": operator_stats["total_linear_macs"],
            "graph_message_scalar_ops": operator_stats[
                "graph_message_scalar_ops"
            ],
            "inference_microseconds_per_item_mean": operator_stats[
                "inference_microseconds_per_item"
            ]["mean"],
        }
        for role in ROLES:
            summary_row[f"{role}_f1_mean"] = operator_stats[
                f"{role}_node_f1"
            ]["mean"]
            summary_row[f"{role}_f1_std"] = operator_stats[
                f"{role}_node_f1"
            ]["standard_deviation"]
            summary_row[f"{role}_exact_attack_mean"] = operator_stats[
                f"{role}_exact_set_attack"
            ]["mean"]
            summary_row[f"{role}_exact_attack_std"] = operator_stats[
                f"{role}_exact_set_attack"
            ]["standard_deviation"]
            summary_row[f"{role}_ap_mean"] = operator_stats[
                f"{role}_checkpoint_ap"
            ]["mean"]
            summary_row[f"{role}_ap_std"] = operator_stats[
                f"{role}_checkpoint_ap"
            ]["standard_deviation"]
        summary_rows.append(summary_row)

    metric_leaders: dict[str, Any] = {}
    for metric in (
        "macro_role_node_f1",
        "macro_role_exact_set_attack",
        "joint_exact_all_roles_attack",
        "source_node_f1",
        "transit_node_f1",
        "victim_node_f1",
        "path_node_f1",
    ):
        leader = max(
            OPERATORS,
            key=lambda operator: statistics[operator][metric]["mean"],
        )
        metric_leaders[metric] = {
            "operator": leader,
            "mean": statistics[leader][metric]["mean"],
            "architecture_selected": False,
        }

    per_seed_csv = output_dir / "V5_P2_G3_PER_SEED_RESULTS.csv"
    summary_csv = output_dir / "V5_P2_G3_THREE_SEED_SUMMARY.csv"
    write_csv(per_seed_csv, per_seed)
    write_csv(summary_csv, summary_rows)

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "matrix": {
            "task": "C_ROLE_AWARE_MULTILABEL_LOCALIZATION",
            "operators": list(OPERATORS),
            "seeds": list(SEEDS),
            "completed_runs": 9,
            "execution_policy": "serial one-model-at-a-time",
            "loss": (
                "sum of independently class-weighted source, transit, "
                "victim, and path BCEWithLogits losses"
            ),
            "checkpoint_score": (
                "0.30 source AP + 0.25 transit AP + "
                "0.25 victim AP + 0.20 path AP"
            ),
            "threshold_policy": (
                "four separate validation-only thresholds; each maximizes "
                "0.60 node F1 + 0.40 exact role-set accuracy on attack items"
            ),
            "roles_mutually_exclusive": False,
            "gat_included": False,
            "gat_exclusion_reason": (
                "DO_NOT_PROMOTE_GAT_FROM_G1_AND_G2"
            ),
        },
        "prior_stage_provenance": {
            "g1_aggregation_report": str(g1_report_path),
            "g1_aggregation_report_sha256": EXPECTED_G1_REPORT_SHA,
            "g2_aggregation_report": str(g2_report_path),
            "g2_aggregation_report_sha256": EXPECTED_G2_REPORT_SHA,
        },
        "per_seed": per_seed,
        "mean_and_standard_deviation": statistics,
        "validation_only_metric_leaders": metric_leaders,
        "decision_boundary": {
            "architecture_selected": False,
            "interpretation": (
                "G3 metric leaders are validation evidence only. Review "
                "G1, G2, and G3 jointly before executing frozen Task D."
            ),
        },
        "security_boundary": {
            "p2_test_directory_enumerated": False,
            "p2_test_tensors_deserialized": False,
            "test_evaluation_performed": False,
            "architecture_selected": False,
            "quantization_performed": False,
            "rtl_generated": False,
        },
        "next_action": (
            "JOINT_G1_G2_G3_REVIEW_THEN_FROZEN_TASK_D_FULL_MULTITASK_SYSTEM"
        ),
        "report_paths": report_paths,
    }
    report_path = (
        output_dir
        / "V5_P2_G3_ROLE_AWARE_MULTILABEL_MATRIX_AGGREGATION.json"
    )
    write_json(report_path, report)
    lock = {
        "status": COMPLETE,
        "completed_runs": 9,
        "report_sha256": sha256_file(report_path),
        "per_seed_csv_sha256": sha256_file(per_seed_csv),
        "summary_csv_sha256": sha256_file(summary_csv),
        "g1_aggregation_report_sha256": EXPECTED_G1_REPORT_SHA,
        "g2_aggregation_report_sha256": EXPECTED_G2_REPORT_SHA,
        "p2_test_directory_enumerated": False,
        "p2_test_tensors_deserialized": False,
        "architecture_selected": False,
        "next_action": (
            "JOINT_G1_G2_G3_REVIEW_THEN_FROZEN_TASK_D_FULL_MULTITASK_SYSTEM"
        ),
    }
    write_json(
        output_dir
        / "V5_P2_G3_ROLE_AWARE_MULTILABEL_MATRIX_AGGREGATION_LOCK.json",
        lock,
    )
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-G3 ROLE-AWARE MULTILABEL MATRIX AGGREGATION =====")
    print("status: COMPLETE")
    print("completed_runs: 9")
    for operator in OPERATORS:
        stats = statistics[operator]
        print(
            f"{operator}: "
            f"macro_role_f1={stats['macro_role_node_f1']['mean']:.6f}"
            f"±{stats['macro_role_node_f1']['standard_deviation']:.6f} "
            f"macro_exact_attack="
            f"{stats['macro_role_exact_set_attack']['mean']:.6f}"
            f"±{stats['macro_role_exact_set_attack']['standard_deviation']:.6f} "
            f"joint_exact_attack="
            f"{stats['joint_exact_all_roles_attack']['mean']:.6f}"
            f"±{stats['joint_exact_all_roles_attack']['standard_deviation']:.6f}"
        )
    print("architecture_selected: false")
    print("test_directory_enumerated: false")
    print("test_tensors_deserialized: false")
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
