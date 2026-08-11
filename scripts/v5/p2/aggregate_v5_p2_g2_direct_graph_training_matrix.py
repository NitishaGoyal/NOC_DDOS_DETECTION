#!/usr/bin/env python3
"""Aggregate the complete 4-operator × 3-seed V5 P2-G2 matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any

OPERATORS = ("conv1d", "gcnconv", "graphconv", "gatconv")
SEEDS = (107, 117, 127)
STAGE = "V5_P2_G2_DIRECT_GRAPH_MATRIX_AGGREGATION"
COMPLETE = f"{STAGE}_COMPLETE"
RUN_STAGE = "V5_P2_G2_DIRECT_GRAPH_SINGLE_RUN"
RUN_COMPLETE = f"{RUN_STAGE}_COMPLETE"


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


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "standard_deviation": stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def extract_row(report: dict[str, Any]) -> dict[str, Any]:
    selected = report["post_checkpoint_validation"]["threshold_selection"]["selected"]
    graph = selected["metrics"]
    thresholded = graph["thresholded"]
    latency = report["cost"]["inference_latency_proxy"]
    operation = report["cost"]["operation_count"]
    best = report["best_checkpoint"]["validation_metrics_at_selection"]
    return {
        "operator": report["operator"],
        "seed": report["seed"],
        "best_epoch": report["best_checkpoint"]["epoch"],
        "selection_score": best["selection_score"],
        "checkpoint_graph_average_precision": best["graph"]["average_precision"],
        "checkpoint_graph_auroc": best["graph"]["auroc"],
        "selected_threshold": selected["threshold"],
        "threshold_objective_balanced_accuracy": selected["objective"],
        "graph_average_precision": graph["average_precision"],
        "graph_auroc": graph["auroc"],
        "graph_accuracy": thresholded["accuracy"],
        "graph_balanced_accuracy": thresholded["balanced_accuracy"],
        "graph_f1": thresholded["f1"],
        "graph_precision": thresholded["precision"],
        "graph_recall": thresholded["recall"],
        "graph_fpr": thresholded["fpr"],
        "graph_tnr": thresholded["tnr"],
        "tp": thresholded["tp"],
        "tn": thresholded["tn"],
        "fp": thresholded["fp"],
        "fn": thresholded["fn"],
        "parameter_count": report["architecture"]["parameter_count"],
        "total_linear_macs": operation["total_linear_macs"],
        "graph_message_scalar_ops": operation["graph_message_scalar_ops"],
        "pooling_scalar_ops": operation["pooling_scalar_ops"],
        "peak_cuda_memory_allocated_bytes": report["cost"]["peak_cuda_memory_allocated_bytes"],
        "inference_microseconds_per_item": latency["microseconds_per_item"],
        "total_training_seconds": report["training"]["total_elapsed_seconds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--g1-aggregation-dir", type=Path, required=True)
    parser.add_argument("--g1-freeze-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reports_root = args.reports_root.expanduser().resolve()
    g1_aggregation_dir = args.g1_aggregation_dir.expanduser().resolve()
    g1_freeze_dir = args.g1_freeze_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"STOP: output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    g1_marker = g1_aggregation_dir / "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION_COMPLETE"
    g1_report_path = g1_aggregation_dir / "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION.json"
    g1_lock_path = g1_aggregation_dir / "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION_LOCK.json"
    g1_decision_path = g1_freeze_dir / "g1_decision.txt"
    g1_freeze_marker = g1_freeze_dir / "V5_P2_G1C_VALIDATION_RESULT_FROZEN"
    for path in (g1_marker, g1_report_path, g1_lock_path, g1_decision_path, g1_freeze_marker):
        if not path.is_file():
            raise SystemExit(f"STOP: missing frozen G1 prerequisite: {path}")
    g1_report = json.loads(g1_report_path.read_text(encoding="utf-8"))
    g1_lock = json.loads(g1_lock_path.read_text(encoding="utf-8"))
    if g1_report.get("status") != "COMPLETE" or g1_lock.get("status") != "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION_COMPLETE":
        raise SystemExit("STOP: frozen G1 aggregation is not complete")
    if g1_lock.get("report_sha256") != sha256_file(g1_report_path):
        raise SystemExit("STOP: frozen G1 report SHA mismatch")

    rows: list[dict[str, Any]] = []
    for operator in OPERATORS:
        for seed in SEEDS:
            run_dir = reports_root / operator / f"seed_{seed}"
            marker = run_dir / RUN_COMPLETE
            report_path = run_dir / f"{RUN_STAGE}.json"
            if not marker.is_file() or not report_path.is_file():
                raise SystemExit(f"STOP: incomplete G2 run {operator} seed {seed}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("status") != "COMPLETE":
                raise SystemExit(f"STOP: non-COMPLETE report {report_path}")
            if report.get("operator") != operator or report.get("seed") != seed:
                raise SystemExit(f"STOP: identity mismatch in {report_path}")
            security = report.get("security_boundary", {})
            if security.get("p2_test_directory_enumerated") is not False or security.get("p2_test_tensors_deserialized") is not False:
                raise SystemExit(f"STOP: test security flag changed in {report_path}")
            if security.get("architecture_selected") is not False:
                raise SystemExit(f"STOP: architecture-selected flag changed in {report_path}")
            rows.append(extract_row(report))

    metrics_to_summarize = [
        "selection_score",
        "checkpoint_graph_average_precision",
        "checkpoint_graph_auroc",
        "selected_threshold",
        "threshold_objective_balanced_accuracy",
        "graph_average_precision",
        "graph_auroc",
        "graph_accuracy",
        "graph_balanced_accuracy",
        "graph_f1",
        "graph_precision",
        "graph_recall",
        "graph_fpr",
        "graph_tnr",
        "inference_microseconds_per_item",
        "total_training_seconds",
    ]
    architecture_summary: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for operator in OPERATORS:
        operator_rows = [row for row in rows if row["operator"] == operator]
        summary = {metric: summarize([float(row[metric]) for row in operator_rows]) for metric in metrics_to_summarize}
        summary["parameter_count"] = int(operator_rows[0]["parameter_count"])
        summary["total_linear_macs"] = int(operator_rows[0]["total_linear_macs"])
        summary["graph_message_scalar_ops"] = int(operator_rows[0]["graph_message_scalar_ops"])
        summary["pooling_scalar_ops"] = int(operator_rows[0]["pooling_scalar_ops"])
        memory_values = [row["peak_cuda_memory_allocated_bytes"] for row in operator_rows if row["peak_cuda_memory_allocated_bytes"] is not None]
        summary["peak_cuda_memory_allocated_bytes"] = summarize([float(value) for value in memory_values]) if len(memory_values) == 3 else None
        architecture_summary[operator] = summary
        summary_rows.append({
            "operator": operator,
            "graph_balanced_accuracy_mean": summary["graph_balanced_accuracy"]["mean"],
            "graph_balanced_accuracy_std": summary["graph_balanced_accuracy"]["standard_deviation"],
            "graph_f1_mean": summary["graph_f1"]["mean"],
            "graph_f1_std": summary["graph_f1"]["standard_deviation"],
            "graph_ap_mean": summary["graph_average_precision"]["mean"],
            "graph_ap_std": summary["graph_average_precision"]["standard_deviation"],
            "graph_auroc_mean": summary["graph_auroc"]["mean"],
            "graph_auroc_std": summary["graph_auroc"]["standard_deviation"],
            "graph_fpr_mean": summary["graph_fpr"]["mean"],
            "graph_fpr_std": summary["graph_fpr"]["standard_deviation"],
            "parameter_count": summary["parameter_count"],
            "total_linear_macs": summary["total_linear_macs"],
            "inference_microseconds_per_item_mean": summary["inference_microseconds_per_item"]["mean"],
        })

    graphconv = architecture_summary["graphconv"]
    gat = architecture_summary["gatconv"]
    f1_delta = gat["graph_f1"]["mean"] - graphconv["graph_f1"]["mean"]
    balanced_delta = gat["graph_balanced_accuracy"]["mean"] - graphconv["graph_balanced_accuracy"]["mean"]
    explicit_gate = f1_delta >= 0.02 or balanced_delta >= 0.02
    lower_f1_variance = gat["graph_f1"]["standard_deviation"] < graphconv["graph_f1"]["standard_deviation"]
    lower_balanced_variance = gat["graph_balanced_accuracy"]["standard_deviation"] < graphconv["graph_balanced_accuracy"]["standard_deviation"]
    if explicit_gate:
        gat_decision = "PROMOTE_GAT_BY_EXPLICIT_0P02_G2_PERFORMANCE_GATE"
    elif lower_f1_variance or lower_balanced_variance:
        gat_decision = "MANUAL_VARIANCE_REVIEW_REQUIRED_NO_AUTOMATIC_G2_PROMOTION"
    else:
        gat_decision = "DO_NOT_PROMOTE_GAT_FROM_G2"

    g1_gat_decision = g1_report.get("gat_screening", {}).get("decision")
    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "matrix": {
            "task": "B_DIRECT_GRAPH_DETECTION",
            "operators": list(OPERATORS),
            "seeds": list(SEEDS),
            "completed_runs": 12,
            "execution_policy": "serial one-model-at-a-time",
            "loss": "unweighted graph BCEWithLogits",
            "checkpoint_score": "0.5 graph AP + 0.5 graph AUROC",
            "threshold_objective": "maximize validation balanced accuracy",
        },
        "per_seed": rows,
        "mean_and_standard_deviation": architecture_summary,
        "gat_screening": {
            "rule": "promote only if GAT beats GraphConv by at least 0.02 absolute graph F1 or graph balanced accuracy, or materially reduces multi-seed variance",
            "graph_f1_mean_delta_vs_graphconv": f1_delta,
            "graph_balanced_accuracy_mean_delta_vs_graphconv": balanced_delta,
            "explicit_performance_gate_pass": explicit_gate,
            "gat_graph_f1_std_lower_than_graphconv": lower_f1_variance,
            "gat_graph_balanced_accuracy_std_lower_than_graphconv": lower_balanced_variance,
            "g1_screening_decision": g1_gat_decision,
            "g2_screening_decision": gat_decision,
            "variance_materiality_not_numerically_defined_by_g0": True,
            "architecture_selected": False,
        },
        "g1_provenance": {
            "aggregation_report": str(g1_report_path),
            "aggregation_report_sha256": sha256_file(g1_report_path),
            "decision_file": str(g1_decision_path),
            "decision_file_sha256": sha256_file(g1_decision_path),
        },
        "security_boundary": {
            "p2_test_directory_enumerated": False,
            "p2_test_tensors_deserialized": False,
            "test_evaluation_performed": False,
            "architecture_selected": False,
            "quantization_performed": False,
            "rtl_generated": False,
        },
        "next_stage": "V5_P2_G3_ROLE_AWARE_MULTILABEL_LOCALIZATION",
    }

    report_path = output_dir / f"{STAGE}.json"
    per_seed_path = output_dir / "V5_P2_G2_PER_SEED_RESULTS.csv"
    summary_path = output_dir / "V5_P2_G2_THREE_SEED_SUMMARY.csv"
    write_json(report_path, report)
    write_csv(per_seed_path, rows)
    write_csv(summary_path, summary_rows)
    lock = {
        "status": COMPLETE,
        "report_sha256": sha256_file(report_path),
        "per_seed_csv_sha256": sha256_file(per_seed_path),
        "summary_csv_sha256": sha256_file(summary_path),
        "completed_runs": 12,
        "g1_aggregation_report_sha256": sha256_file(g1_report_path),
        "p2_test_directory_enumerated": False,
        "p2_test_tensors_deserialized": False,
        "architecture_selected": False,
        "next_stage": report["next_stage"],
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-G2 DIRECT GRAPH MATRIX AGGREGATION =====")
    print("status: COMPLETE")
    print("completed_runs: 12")
    for row in summary_rows:
        print(
            f"{row['operator']}: graph_bal_acc={row['graph_balanced_accuracy_mean']:.6f}±{row['graph_balanced_accuracy_std']:.6f} "
            f"graph_f1={row['graph_f1_mean']:.6f}±{row['graph_f1_std']:.6f} "
            f"graph_ap={row['graph_ap_mean']:.6f}±{row['graph_ap_std']:.6f}"
        )
    print("g1_gat_screening_decision:", g1_gat_decision)
    print("g2_gat_screening_decision:", gat_decision)
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
