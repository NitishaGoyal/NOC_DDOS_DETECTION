#!/usr/bin/env python3
"""Aggregate the frozen 2-candidate x 5-seed Task-D validation matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

STAGE = "V5_P2_TASK_D_FULL_MULTITASK_MATRIX_AGGREGATION"
COMPLETE = f"{STAGE}_COMPLETE"
CANDIDATES = ["conv1d", "graphconv"]
SEEDS = [107, 117, 127, 137, 147]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--promotion-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    training_root = args.training_root.expanduser().resolve()
    promotion_dir = args.promotion_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        print(f"STOP: output exists: {out}")
        return 2
    out.mkdir(parents=True)

    promotion_report = promotion_dir / "V5_P2_G123_GRAPH_OPERATOR_PROMOTION.json"
    promotion_lock = promotion_dir / "V5_P2_G123_GRAPH_OPERATOR_PROMOTION_LOCK.json"
    if not promotion_report.is_file() or not promotion_lock.is_file():
        raise RuntimeError("promotion freeze missing")
    plock = json.loads(promotion_lock.read_text())
    if plock["report_sha256"] != sha256_file(promotion_report):
        raise RuntimeError("promotion report SHA mismatch")

    per_seed: list[dict[str, Any]] = []
    report_hashes: dict[str, str] = {}
    common_hashes: dict[int, dict[str, str]] = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for candidate in CANDIDATES:
            run_dir = training_root / candidate / f"seed_{seed}"
            marker = run_dir / "V5_P2_TASK_D_FULL_MULTITASK_SINGLE_RUN_COMPLETE"
            report_path = run_dir / f"V5_P2_TASK_D_{candidate.upper()}_SEED_{seed}_REPORT.json"
            lock_path = run_dir / f"V5_P2_TASK_D_{candidate.upper()}_SEED_{seed}_LOCK.json"
            for path in (marker, report_path, lock_path):
                if not path.is_file():
                    raise RuntimeError(f"missing completed-run artifact: {path}")
            report = json.loads(report_path.read_text())
            lock = json.loads(lock_path.read_text())
            if lock["report_sha256"] != sha256_file(report_path):
                raise RuntimeError(f"report SHA mismatch: {candidate} seed {seed}")
            checkpoint_path = Path(report["best"]["checkpoint_path"])
            if not checkpoint_path.is_file():
                raise RuntimeError(f"checkpoint missing: {checkpoint_path}")
            if report["best"]["checkpoint_sha256"] != sha256_file(checkpoint_path):
                raise RuntimeError(f"checkpoint SHA mismatch: {candidate} seed {seed}")
            if report["security_boundary"]["test_tensors_deserialized"] is not False:
                raise RuntimeError("test tensor boundary violated")
            if report["security_boundary"]["threshold_tuning_performed"] is not False:
                raise RuntimeError("thresholds were tuned during Task-D training")

            m = report["best"]["validation_metrics"]
            row = {
                "candidate": candidate,
                "seed": seed,
                "best_epoch": report["best"]["epoch"],
                "selection_score": m["selection_score"],
                "validation_loss": m["loss"],
                "graph_auroc": m["graph"]["auroc"],
                "graph_average_precision": m["graph"]["average_precision"],
                "graph_fixed_0_5_f1": m["graph"]["fixed_0_5"]["f1"],
                "count_active_macro_f1": m["count_active"]["macro_f1"],
                "count_active_accuracy": m["count_active"]["accuracy"],
                "source_average_precision": m["roles"]["source"]["average_precision"],
                "transit_average_precision": m["roles"]["transit"]["average_precision"],
                "victim_average_precision": m["roles"]["victim"]["average_precision"],
                "path_average_precision": m["roles"]["path"]["average_precision"],
                "source_fixed_0_5_f1": m["roles"]["source"]["fixed_0_5"]["f1"],
                "transit_fixed_0_5_f1": m["roles"]["transit"]["fixed_0_5"]["f1"],
                "victim_fixed_0_5_f1": m["roles"]["victim"]["fixed_0_5"]["f1"],
                "path_fixed_0_5_f1": m["roles"]["path"]["fixed_0_5"]["f1"],
                "parameter_count": report["architecture"]["parameter_count"],
                "total_linear_macs": report["deployment_proxies"]["total_linear_macs"],
                "graph_message_scalar_ops": report["deployment_proxies"]["graph_message_scalar_ops"],
                "inference_microseconds_per_item": report["deployment_proxies"]["inference_microseconds_per_item"],
                "peak_cuda_memory_bytes": report["deployment_proxies"]["peak_cuda_memory_bytes"],
                "common_initial_state_sha256": report["architecture"]["common_initial_state_sha256"],
                "checkpoint_sha256": report["best"]["checkpoint_sha256"],
            }
            per_seed.append(row)
            common_hashes[seed][candidate] = row["common_initial_state_sha256"]
            report_hashes[f"{candidate}_seed_{seed}"] = sha256_file(report_path)

    for seed in SEEDS:
        if common_hashes[seed]["conv1d"] != common_hashes[seed]["graphconv"]:
            raise RuntimeError(f"common matched-seed initialization differs for seed {seed}")

    metric_names = [
        "selection_score", "validation_loss", "graph_auroc", "graph_average_precision",
        "graph_fixed_0_5_f1", "count_active_macro_f1", "count_active_accuracy",
        "source_average_precision", "transit_average_precision", "victim_average_precision",
        "path_average_precision", "source_fixed_0_5_f1", "transit_fixed_0_5_f1",
        "victim_fixed_0_5_f1", "path_fixed_0_5_f1", "inference_microseconds_per_item",
        "peak_cuda_memory_bytes",
    ]
    summary: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        rows = [row for row in per_seed if row["candidate"] == candidate]
        result: dict[str, Any] = {
            "candidate": candidate,
            "parameter_count": rows[0]["parameter_count"],
            "total_linear_macs": rows[0]["total_linear_macs"],
            "graph_message_scalar_ops": rows[0]["graph_message_scalar_ops"],
        }
        for metric in metric_names:
            mean, std = mean_std([float(row[metric]) for row in rows])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
        summary.append(result)

    paired: list[dict[str, Any]] = []
    paired_metrics = [
        "selection_score", "graph_auroc", "graph_average_precision", "count_active_macro_f1",
        "source_average_precision", "transit_average_precision", "victim_average_precision",
        "path_average_precision", "inference_microseconds_per_item", "peak_cuda_memory_bytes",
    ]
    for seed in SEEDS:
        conv = next(row for row in per_seed if row["candidate"] == "conv1d" and row["seed"] == seed)
        graph = next(row for row in per_seed if row["candidate"] == "graphconv" and row["seed"] == seed)
        row: dict[str, Any] = {"seed": seed}
        for metric in paired_metrics:
            row[f"graphconv_minus_conv1d_{metric}"] = float(graph[metric]) - float(conv[metric])
        paired.append(row)

    leader = max(summary, key=lambda row: float(row["selection_score_mean"]))["candidate"]
    per_seed_path = out / "V5_P2_TASK_D_PER_SEED_RESULTS.csv"
    summary_path = out / "V5_P2_TASK_D_FIVE_SEED_SUMMARY.csv"
    paired_path = out / "V5_P2_TASK_D_PAIRED_SEED_DELTAS.csv"
    write_csv(per_seed_path, per_seed)
    write_csv(summary_path, summary)
    write_csv(paired_path, paired)

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "completed_runs": 10,
        "candidates": CANDIDATES,
        "seeds": SEEDS,
        "summary": summary,
        "paired_seed_deltas": paired,
        "provisional_validation_selection_score_leader": leader,
        "architecture_selected": False,
        "selection_deferred_to": "V5_P2_G4_GRAPH_OPERATOR_STABILITY_AND_SELECTION",
        "threshold_tuning_performed": False,
        "security_boundary": {
            "test_directory_enumerated": False,
            "test_tensors_deserialized": False,
            "test_evaluation_performed": False,
        },
        "provenance": {
            "promotion_report_sha256": sha256_file(promotion_report),
            "run_report_sha256": report_hashes,
        },
        "next_stage": "V5_P2_G4_GRAPH_OPERATOR_STABILITY_AND_SELECTION",
    }
    report_path = out / f"{STAGE}.json"
    write_json(report_path, report)
    lock = {
        "status": COMPLETE,
        "completed_runs": 10,
        "report_sha256": sha256_file(report_path),
        "per_seed_csv_sha256": sha256_file(per_seed_path),
        "summary_csv_sha256": sha256_file(summary_path),
        "paired_csv_sha256": sha256_file(paired_path),
        "provisional_leader": leader,
        "architecture_selected": False,
        "threshold_tuning_performed": False,
        "p2_test_directory_enumerated": False,
        "p2_test_tensors_deserialized": False,
        "script_sha256": sha256_file(Path(__file__)),
        "next_action": "V5_P2_G4_GRAPH_OPERATOR_STABILITY_AND_SELECTION",
    }
    write_json(out / f"{STAGE}_LOCK.json", lock)
    (out / COMPLETE).write_text(COMPLETE + "\n", encoding="utf-8")

    print("===== V5 P2 TASK-D MATRIX AGGREGATION =====")
    print("status: COMPLETE")
    print("completed_runs: 10")
    for row in summary:
        print(
            f"{row['candidate']}: selection_score={row['selection_score_mean']:.6f}±{row['selection_score_std']:.6f} "
            f"graph_auroc={row['graph_auroc_mean']:.6f}±{row['graph_auroc_std']:.6f} "
            f"count_f1={row['count_active_macro_f1_mean']:.6f}±{row['count_active_macro_f1_std']:.6f}"
        )
    print("provisional_leader:", leader)
    print("architecture_selected: false")
    print("threshold_tuning_performed: false")
    print("test_directory_enumerated: false")
    print("test_tensors_deserialized: false")
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
