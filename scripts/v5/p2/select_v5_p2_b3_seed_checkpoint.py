#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

STAGE = "V5_P2_B3_SEED_AND_CHECKPOINT_SELECTION"
COMPLETE = f"{STAGE}_COMPLETE"
SEEDS = [107, 117, 127]
EXPECTED_PROTOCOL_SHA = "0817ae7812f91e3c75260589acaf52bd8a74b07b2114a451b4773578efde4b60"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rank(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    return (
        float(row["selection_score"]),
        float(row["graph_auroc"]),
        float(row["graph_average_precision"]),
        -float(row["validation_loss"]),
        -int(row["best_epoch"]),
    )


def hold(output_dir: Path, failures: list[str], warnings: list[str]) -> int:
    report = {
        "stage": STAGE,
        "status": "HOLD",
        "failures": failures,
        "warnings": warnings,
        "dataset_constructed": False,
        "checkpoint_deserialized": False,
        "threshold_tuning_performed": False,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
    }
    write_json(output_dir / f"{STAGE}.json", report)
    atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
    print(f"{STAGE}_HOLD")
    for failure in failures:
        print("FAIL:", failure)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1-dir", type=Path, required=True)
    parser.add_argument("--b2-finalization-dir", type=Path, required=True)
    parser.add_argument("--seed-report-root", type=Path, required=True)
    parser.add_argument("--seed-model-root", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--model-source-path", type=Path, required=True)
    parser.add_argument("--selected-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    b1_dir = args.b1_dir.expanduser().resolve()
    b2_dir = args.b2_finalization_dir.expanduser().resolve()
    seed_report_root = args.seed_report_root.expanduser().resolve()
    seed_model_root = args.seed_model_root.expanduser().resolve()
    loader_path = args.loader_path.expanduser().resolve()
    model_source_path = args.model_source_path.expanduser().resolve()
    selected_model_dir = args.selected_model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
        return 2
    if selected_model_dir.exists():
        print(f"STOP: selected model directory already exists: {selected_model_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    b1_report_path = b1_dir / "V5_P2_B1_TRAINING_PROTOCOL_LOCK.json"
    b1_lock_path = b1_dir / "V5_P2_B1_TRAINING_PROTOCOL_LOCK_LOCK.json"
    protocol_path = b1_dir / "V5_P2_B1_TRAINING_PROTOCOL.json"
    b2_report_path = b2_dir / "V5_P2_B2_MULTI_SEED_TRAINING.json"
    b2_lock_path = b2_dir / "V5_P2_B2_MULTI_SEED_TRAINING_LOCK.json"

    for path in (b1_report_path, b1_lock_path, protocol_path, b2_report_path, b2_lock_path, loader_path, model_source_path):
        if not path.is_file():
            failures.append(f"missing prerequisite: {path}")
    if failures:
        return hold(output_dir, failures, warnings)

    b1_report = load_json(b1_report_path)
    b1_lock = load_json(b1_lock_path)
    protocol = load_json(protocol_path)
    b2_report = load_json(b2_report_path)
    b2_lock = load_json(b2_lock_path)

    if b1_report.get("status") != "COMPLETE":
        failures.append("B1 status is not COMPLETE")
    if b1_lock.get("report_sha256") != sha256_file(b1_report_path):
        failures.append("B1 report SHA mismatch")
    if b1_lock.get("protocol_file_sha256") != sha256_file(protocol_path):
        failures.append("B1 protocol-file SHA mismatch")
    if protocol.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA:
        failures.append("B1 protocol SHA changed")
    if b1_lock.get("loader_sha256") != sha256_file(loader_path):
        failures.append("B1 loader SHA mismatch")
    if b1_lock.get("model_sha256") != sha256_file(model_source_path):
        failures.append("B1 model-source SHA mismatch")
    if b1_lock.get("test_evaluation_authorized") is not False:
        failures.append("B1 unexpectedly authorizes test evaluation")

    if b2_report.get("status") != "COMPLETE":
        failures.append("B2 status is not COMPLETE")
    if b2_report.get("decision") != "FREEZE_ALL_P2_B2_SEED_RUNS_AND_AUTHORIZE_B3":
        failures.append("B2 decision did not authorize B3")
    if b2_lock.get("report_sha256") != sha256_file(b2_report_path):
        failures.append("B2 report SHA mismatch")
    if b2_lock.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA:
        failures.append("B2 protocol SHA changed")
    if b2_lock.get("seeds") != SEEDS:
        failures.append("B2 seed set changed")
    if b2_lock.get("threshold_tuning_performed") is not False:
        failures.append("B2 performed threshold tuning")
    if b2_lock.get("test_tensor_contents_accessed") is not False:
        failures.append("B2 accessed test tensors")
    if b2_lock.get("test_evaluation_performed") is not False:
        failures.append("B2 performed test evaluation")

    rows = b2_report.get("comparison_rows", [])
    if len(rows) != 3:
        failures.append(f"B2 comparison row count={len(rows)}, expected 3")
    if sorted(int(row["seed"]) for row in rows) != SEEDS:
        failures.append("B2 comparison seeds changed")

    verified: list[dict[str, Any]] = []
    for row in rows:
        seed = int(row["seed"])
        report_path = seed_report_root / f"seed_{seed}" / f"V5_P2_B2_SEED_{seed}_TRAINING_REPORT.json"
        lock_path = seed_report_root / f"seed_{seed}" / f"V5_P2_B2_SEED_{seed}_LOCK.json"
        checkpoint_path = seed_model_root / f"seed_{seed}" / f"v5_p2_b2_seed_{seed}_best.pt"

        for path in (report_path, lock_path, checkpoint_path):
            if not path.is_file():
                failures.append(f"seed {seed} missing artifact: {path}")
        if not (report_path.is_file() and lock_path.is_file() and checkpoint_path.is_file()):
            continue

        seed_report = load_json(report_path)
        seed_lock = load_json(lock_path)
        checkpoint_sha = sha256_file(checkpoint_path)

        if seed_report.get("status") != "COMPLETE":
            failures.append(f"seed {seed} report is not COMPLETE")
        if seed_lock.get("report_sha256") != sha256_file(report_path):
            failures.append(f"seed {seed} report SHA mismatch")
        if seed_lock.get("checkpoint_sha256") != checkpoint_sha:
            failures.append(f"seed {seed} checkpoint SHA mismatch")
        if b2_lock.get("seed_checkpoint_sha256", {}).get(str(seed)) != checkpoint_sha:
            failures.append(f"seed {seed} B2-finalization checkpoint SHA mismatch")
        if seed_lock.get("threshold_tuning_performed") is not False:
            failures.append(f"seed {seed} performed threshold tuning")
        if seed_lock.get("test_tensor_contents_accessed") is not False:
            failures.append(f"seed {seed} accessed test tensors")
        if seed_lock.get("test_evaluation_performed") is not False:
            failures.append(f"seed {seed} performed test evaluation")

        best = seed_report["best"]
        metrics = best["validation_metrics"]
        expected = {
            "best_epoch": int(best["epoch"]),
            "selection_score": float(metrics["selection_score"]),
            "graph_auroc": float(metrics["graph"]["auroc"]),
            "graph_average_precision": float(metrics["graph"]["average_precision"]),
            "validation_loss": float(metrics["loss"]),
            "checkpoint_sha256": checkpoint_sha,
        }
        for key, value in expected.items():
            observed = row[key]
            if isinstance(value, float):
                if abs(float(observed) - value) > 1e-12:
                    failures.append(f"seed {seed} comparison mismatch for {key}")
            elif observed != value:
                failures.append(f"seed {seed} comparison mismatch for {key}")

        enriched = dict(row)
        enriched["checkpoint_path"] = str(checkpoint_path.resolve())
        enriched["seed_report_path"] = str(report_path.resolve())
        enriched["seed_lock_path"] = str(lock_path.resolve())
        enriched["ranking_tuple"] = list(rank(enriched))
        verified.append(enriched)

    if failures:
        return hold(output_dir, failures, warnings)

    ranked = sorted(verified, key=rank, reverse=True)
    selected = ranked[0]
    runner_up = ranked[1]

    selected_seed = int(selected["seed"])
    selected_epoch = int(selected["best_epoch"])
    source_checkpoint = Path(selected["checkpoint_path"])
    source_sha = sha256_file(source_checkpoint)

    selected_model_dir.mkdir(parents=True)
    selected_checkpoint = selected_model_dir / f"v5_p2_b3_selected_seed_{selected_seed}_epoch_{selected_epoch}.pt"
    shutil.copy2(source_checkpoint, selected_checkpoint)
    selected_sha = sha256_file(selected_checkpoint)
    if selected_sha != source_sha:
        failures.append("selected checkpoint copy SHA mismatch")
        return hold(output_dir, failures, warnings)

    margin = float(selected["selection_score"]) - float(runner_up["selection_score"])

    manifest = {
        "contract_name": "V5_P2_B3_VALIDATION_ONLY_SEED_AND_CHECKPOINT_SELECTION",
        "contract_version": 1,
        "ranking_rule": {
            "primary": "higher validation selection score",
            "tie_breakers": [
                "higher graph AUROC",
                "higher graph average precision",
                "lower validation total loss",
                "earlier best epoch",
            ],
        },
        "ranked_candidates": ranked,
        "selected": {
            "seed": selected_seed,
            "best_epoch": selected_epoch,
            "selection_score": float(selected["selection_score"]),
            "graph_auroc": float(selected["graph_auroc"]),
            "graph_average_precision": float(selected["graph_average_precision"]),
            "count_active_macro_f1": float(selected["count_active_macro_f1"]),
            "source_average_precision": float(selected["source_average_precision"]),
            "transit_average_precision": float(selected["transit_average_precision"]),
            "victim_average_precision": float(selected["victim_average_precision"]),
            "path_average_precision": float(selected["path_average_precision"]),
            "validation_loss": float(selected["validation_loss"]),
            "source_checkpoint_path": str(source_checkpoint),
            "source_checkpoint_sha256": source_sha,
            "frozen_checkpoint_path": str(selected_checkpoint),
            "frozen_checkpoint_sha256": selected_sha,
        },
        "runner_up": {
            "seed": int(runner_up["seed"]),
            "selection_score": float(runner_up["selection_score"]),
            "score_margin": margin,
        },
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "loader_sha256": sha256_file(loader_path),
        "model_source_sha256": sha256_file(model_source_path),
        "selection_uses_validation_only": True,
        "threshold_tuning_performed": False,
        "test_evaluation_authorized": False,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
    }

    manifest_path = output_dir / "V5_P2_B3_SELECTED_CHECKPOINT_MANIFEST.json"
    write_json(manifest_path, manifest)

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "decision": "FREEZE_SELECTED_P2_CHECKPOINT_AND_AUTHORIZE_B4",
        "selection": manifest,
        "artifacts": {
            "selection_manifest": [str(manifest_path), sha256_file(manifest_path)],
            "selected_checkpoint": [str(selected_checkpoint), selected_sha],
        },
        "provenance": {
            "b1_report_sha256": sha256_file(b1_report_path),
            "b1_lock_sha256": sha256_file(b1_lock_path),
            "b1_protocol_file_sha256": sha256_file(protocol_path),
            "b2_report_sha256": sha256_file(b2_report_path),
            "b2_lock_sha256": sha256_file(b2_lock_path),
            "loader_sha256": sha256_file(loader_path),
            "model_source_sha256": sha256_file(model_source_path),
        },
        "security_boundary": {
            "dataset_constructed": False,
            "run_tensors_deserialized": False,
            "checkpoint_deserialized": False,
            "selection_uses_validation_only": True,
            "threshold_tuning_performed": False,
            "test_directory_existence_checked": False,
            "test_directory_enumerated": False,
            "test_tensor_files_opened": False,
            "test_tensor_contents_accessed": False,
            "test_evaluation_performed": False,
            "test_evaluation_authorized": False,
        },
        "failures": [],
        "warnings": warnings,
        "next_stage": "V5_P2_B4_VALIDATION_THRESHOLD_TUNING",
    }
    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    lock = {
        "status": COMPLETE,
        "decision": "FREEZE_SELECTED_P2_CHECKPOINT_AND_AUTHORIZE_B4",
        "report_sha256": sha256_file(report_path),
        "selection_manifest_sha256": sha256_file(manifest_path),
        "selected_seed": selected_seed,
        "selected_best_epoch": selected_epoch,
        "selected_selection_score": float(selected["selection_score"]),
        "selected_checkpoint_path": str(selected_checkpoint),
        "selected_checkpoint_sha256": selected_sha,
        "runner_up_seed": int(runner_up["seed"]),
        "selection_score_margin": margin,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "loader_sha256": sha256_file(loader_path),
        "model_source_sha256": sha256_file(model_source_path),
        "selection_uses_validation_only": True,
        "threshold_tuning_performed": False,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "test_evaluation_performed": False,
        "test_evaluation_authorized": False,
        "next_stage": "V5_P2_B4_VALIDATION_THRESHOLD_TUNING",
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-B3 SEED + CHECKPOINT SELECTION =====")
    print("status: COMPLETE")
    print("decision: FREEZE_SELECTED_P2_CHECKPOINT_AND_AUTHORIZE_B4")
    for index, row in enumerate(ranked, start=1):
        print(
            f"rank={index} seed={row['seed']} best_epoch={row['best_epoch']} "
            f"score={float(row['selection_score']):.10g} "
            f"graph_auroc={float(row['graph_auroc']):.10g} "
            f"graph_ap={float(row['graph_average_precision']):.10g} "
            f"val_loss={float(row['validation_loss']):.10g}"
        )
    print("selected_seed:", selected_seed)
    print("selected_best_epoch:", selected_epoch)
    print("selected_selection_score:", selected["selection_score"])
    print("runner_up_seed:", runner_up["seed"])
    print("selection_score_margin:", margin)
    print("selected_checkpoint_sha256:", selected_sha)
    print("selection_uses_validation_only: true")
    print("checkpoint_deserialized: false")
    print("threshold_tuning_performed: false")
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")
    print("test_evaluation_performed: false")
    print("test_evaluation_authorized: false")
    print("failure_count: 0")
    print("warning_count:", len(warnings))
    print("next_stage: V5_P2_B4_VALIDATION_THRESHOLD_TUNING")
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
