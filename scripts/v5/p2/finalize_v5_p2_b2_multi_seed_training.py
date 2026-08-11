#!/usr/bin/env python3
"""
V5 P2-B2 Multi-Seed Finalization

Validates the completed seed-107, seed-117, and seed-127 training runs,
freezes their checkpoint hashes and validation metrics, and authorizes B3.

This stage does not select the winning seed. Seed/checkpoint selection remains
the separate B3 stage. It does not construct datasets or access test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


STAGE = "V5_P2_B2_MULTI_SEED_TRAINING"
COMPLETE = f"{STAGE}_COMPLETE"
SEEDS = [107, 117, 127]
EXPECTED_PROTOCOL_SHA = (
    "0817ae7812f91e3c75260589acaf52bd8a74b07b2114a451b4773578efde4b60"
)


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
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
    atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1-dir", type=Path, required=True)
    parser.add_argument(
        "--seed-report-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--seed-model-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    b1_dir = args.b1_dir.expanduser().resolve()
    seed_report_root = (
        args.seed_report_root.expanduser().resolve()
    )
    seed_model_root = (
        args.seed_model_root.expanduser().resolve()
    )
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(
            f"STOP: output already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    b1_report_path = (
        b1_dir
        / "V5_P2_B1_TRAINING_PROTOCOL_LOCK.json"
    )
    b1_lock_path = (
        b1_dir
        / "V5_P2_B1_TRAINING_PROTOCOL_LOCK_LOCK.json"
    )
    protocol_path = (
        b1_dir
        / "V5_P2_B1_TRAINING_PROTOCOL.json"
    )

    for path in (
        b1_report_path,
        b1_lock_path,
        protocol_path,
    ):
        if not path.is_file():
            failures.append(f"missing prerequisite: {path}")

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        return 1

    b1_report = load_json(b1_report_path)
    b1_lock = load_json(b1_lock_path)
    protocol = load_json(protocol_path)

    if b1_report.get("status") != "COMPLETE":
        failures.append("B1 status is not COMPLETE")
    if (
        b1_lock.get("report_sha256")
        != sha256_file(b1_report_path)
    ):
        failures.append("B1 report SHA mismatch")
    if (
        b1_lock.get("protocol_file_sha256")
        != sha256_file(protocol_path)
    ):
        failures.append("B1 protocol-file SHA mismatch")
    if protocol.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA:
        failures.append("B1 protocol SHA changed")
    if b1_lock.get("seeds") != SEEDS:
        failures.append("B1 seed set changed")

    rows: list[dict[str, Any]] = []
    seed_records: dict[str, Any] = {}

    for seed in SEEDS:
        report_dir = (
            seed_report_root / f"seed_{seed}"
        )
        model_dir = (
            seed_model_root / f"seed_{seed}"
        )

        report_path = (
            report_dir
            / f"V5_P2_B2_SEED_{seed}_TRAINING_REPORT.json"
        )
        lock_path = (
            report_dir
            / f"V5_P2_B2_SEED_{seed}_LOCK.json"
        )
        complete_path = (
            report_dir
            / "V5_P2_B2_SINGLE_SEED_TRAINING_COMPLETE"
        )
        checkpoint_path = (
            model_dir
            / f"v5_p2_b2_seed_{seed}_best.pt"
        )

        for path in (
            report_path,
            lock_path,
            complete_path,
            checkpoint_path,
        ):
            if not path.is_file():
                failures.append(
                    f"seed {seed} missing artifact: {path}"
                )

        if failures and (
            not report_path.is_file()
            or not lock_path.is_file()
            or not checkpoint_path.is_file()
        ):
            continue

        report = load_json(report_path)
        lock = load_json(lock_path)

        if report.get("status") != "COMPLETE":
            failures.append(
                f"seed {seed} report is not COMPLETE"
            )
        if lock.get("status") != (
            "V5_P2_B2_SINGLE_SEED_TRAINING_COMPLETE"
        ):
            failures.append(
                f"seed {seed} lock status changed"
            )
        if lock.get("seed") != seed:
            failures.append(
                f"seed {seed} lock stores a different seed"
            )
        if (
            lock.get("report_sha256")
            != sha256_file(report_path)
        ):
            failures.append(
                f"seed {seed} report SHA mismatch"
            )
        if (
            lock.get("checkpoint_sha256")
            != sha256_file(checkpoint_path)
        ):
            failures.append(
                f"seed {seed} checkpoint SHA mismatch"
            )
        if (
            lock.get("protocol_sha256")
            != EXPECTED_PROTOCOL_SHA
        ):
            failures.append(
                f"seed {seed} protocol SHA mismatch"
            )
        if lock.get("threshold_tuning_performed") is not False:
            failures.append(
                f"seed {seed} performed threshold tuning"
            )
        if lock.get("test_tensor_contents_accessed") is not False:
            failures.append(
                f"seed {seed} accessed test tensors"
            )
        if lock.get("test_evaluation_performed") is not False:
            failures.append(
                f"seed {seed} performed test evaluation"
            )

        best = report["best"]
        metrics = best["validation_metrics"]

        row = {
            "seed": seed,
            "best_epoch": best["epoch"],
            "completed_epoch": report["training"][
                "completed_epoch"
            ],
            "stopped_early": report["training"][
                "stopped_early"
            ],
            "selection_score": metrics[
                "selection_score"
            ],
            "graph_auroc": metrics["graph"]["auroc"],
            "graph_average_precision": metrics["graph"][
                "average_precision"
            ],
            "graph_fixed_0_5_f1": metrics["graph"][
                "fixed_0_5"
            ]["f1"],
            "count_active_macro_f1": metrics[
                "count_active"
            ]["macro_f1"],
            "count_active_accuracy": metrics[
                "count_active"
            ]["accuracy"],
            "source_average_precision": metrics["roles"][
                "source"
            ]["average_precision"],
            "transit_average_precision": metrics["roles"][
                "transit"
            ]["average_precision"],
            "victim_average_precision": metrics["roles"][
                "victim"
            ]["average_precision"],
            "path_average_precision": metrics["roles"][
                "path"
            ]["average_precision"],
            "validation_loss": metrics["loss"],
            "checkpoint_sha256": (
                sha256_file(checkpoint_path)
            ),
            "initial_state_sha256": lock[
                "initial_state_sha256"
            ],
        }
        rows.append(row)

        seed_records[str(seed)] = {
            "report_path": str(report_path),
            "report_sha256": sha256_file(report_path),
            "lock_path": str(lock_path),
            "lock_sha256": sha256_file(lock_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": (
                sha256_file(checkpoint_path)
            ),
            "best_epoch": best["epoch"],
            "validation_metrics": metrics,
            "initial_state_sha256": lock[
                "initial_state_sha256"
            ],
        }

    if len(rows) != 3:
        failures.append(
            f"complete seed row count={len(rows)}, expected 3"
        )

    initial_hashes = {
        row["initial_state_sha256"]
        for row in rows
    }
    if len(initial_hashes) != 3:
        failures.append(
            "initial-state hashes are not unique across seeds"
        )

    checkpoint_hashes = {
        row["checkpoint_sha256"]
        for row in rows
    }
    if len(checkpoint_hashes) != 3:
        failures.append(
            "best-checkpoint hashes are not unique across seeds"
        )

    comparison_path = (
        output_dir
        / "V5_P2_B2_MULTI_SEED_COMPARISON.csv"
    )
    write_csv(comparison_path, rows)

    report = {
        "stage": STAGE,
        "status": (
            "COMPLETE"
            if not failures
            else "HOLD"
        ),
        "decision": (
            "FREEZE_ALL_P2_B2_SEED_RUNS_AND_AUTHORIZE_B3"
            if not failures
            else "BLOCK_P2_B3"
        ),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "seeds": SEEDS,
        "seed_records": seed_records,
        "comparison_rows": rows,
        "ranking_policy_for_next_stage": {
            "selection_is_not_performed_in_B2": True,
            "B3_primary": (
                "highest validation selection score"
            ),
            "B3_tie_breakers": [
                "higher graph AUROC",
                "higher graph average precision",
                "lower validation total loss",
                "earlier epoch",
            ],
        },
        "artifacts": {
            "comparison_csv": [
                str(comparison_path),
                sha256_file(comparison_path),
            ],
        },
        "provenance": {
            "b1_report_sha256": (
                sha256_file(b1_report_path)
            ),
            "b1_lock_sha256": (
                sha256_file(b1_lock_path)
            ),
            "protocol_file_sha256": (
                sha256_file(protocol_path)
            ),
        },
        "security_boundary": {
            "dataset_constructed": False,
            "run_tensors_deserialized": False,
            "threshold_tuning_performed": False,
            "test_directory_existence_checked": False,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
            "test_evaluation_performed": False,
        },
        "failures": failures,
        "warnings": warnings,
        "next_stage": (
            "V5_P2_B3_SEED_AND_CHECKPOINT_SELECTION"
            if not failures
            else None
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    if failures:
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    lock = {
        "status": COMPLETE,
        "decision": (
            "FREEZE_ALL_P2_B2_SEED_RUNS_AND_AUTHORIZE_B3"
        ),
        "report_sha256": sha256_file(report_path),
        "comparison_sha256": (
            sha256_file(comparison_path)
        ),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "seeds": SEEDS,
        "seed_checkpoint_sha256": {
            str(seed): seed_records[str(seed)][
                "checkpoint_sha256"
            ]
            for seed in SEEDS
        },
        "threshold_tuning_performed": False,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "test_evaluation_performed": False,
        "next_stage": (
            "V5_P2_B3_SEED_AND_CHECKPOINT_SELECTION"
        ),
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(
        output_dir / f"{STAGE}_LOCK.json",
        lock,
    )
    atomic_write(
        output_dir / COMPLETE,
        COMPLETE + "\n",
    )

    print("===== V5 P2-B2 MULTI-SEED FINALIZATION =====")
    print("status: COMPLETE")
    print(
        "decision: "
        "FREEZE_ALL_P2_B2_SEED_RUNS_AND_AUTHORIZE_B3"
    )
    print("protocol_sha256:", EXPECTED_PROTOCOL_SHA)
    for row in rows:
        print(
            f"seed={row['seed']} "
            f"best_epoch={row['best_epoch']} "
            f"score={row['selection_score']:.10g} "
            f"graph_auroc={row['graph_auroc']:.10g} "
            f"graph_ap={row['graph_average_precision']:.10g} "
            f"count_f1={row['count_active_macro_f1']:.10g} "
            f"source_ap={row['source_average_precision']:.10g} "
            f"transit_ap={row['transit_average_precision']:.10g} "
            f"victim_ap={row['victim_average_precision']:.10g} "
            f"path_ap={row['path_average_precision']:.10g}"
        )
    print("seed_selection_performed: false")
    print("threshold_tuning_performed: false")
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")
    print("test_evaluation_performed: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print(
        "next_stage: "
        "V5_P2_B3_SEED_AND_CHECKPOINT_SELECTION"
    )
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
