#!/usr/bin/env python3
"""
V5 P0 Final Results Archive and P2 Handoff

This stage performs no training and no dataset construction.

It verifies and archives the completed P0 evidence chain:
- B8 frozen architecture;
- C0 final-training protocol;
- C1 selected checkpoint;
- C2 frozen thresholds;
- C3 official one-shot test result.

It records that the project is proceeding directly to P2 and that P1 was not
executed. It does not modify any historical report or checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


STAGE = "V5_P0_FINAL_RESULTS_ARCHIVE_AND_P2_HANDOFF"
COMPLETE = f"{STAGE}_COMPLETE"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_and_record(
    source: Path,
    archive_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    destination = archive_dir / source.name
    if destination.exists():
        raise FileExistsError(f"archive destination exists: {destination}")
    shutil.copy2(source, destination)
    records.append(
        {
            "source": str(source),
            "archived_copy": str(destination),
            "source_sha256": sha256_file(source),
            "archived_copy_sha256": sha256_file(destination),
            "size_bytes": source.stat().st_size,
        }
    )


def metric_delta(test: float, validation: float) -> float:
    return float(test - validation)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b8-dir", type=Path, required=True)
    parser.add_argument("--c0-dir", type=Path, required=True)
    parser.add_argument("--c1-report-dir", type=Path, required=True)
    parser.add_argument("--c1-model-dir", type=Path, required=True)
    parser.add_argument("--c2-report-dir", type=Path, required=True)
    parser.add_argument("--c2-model-dir", type=Path, required=True)
    parser.add_argument("--c3-report-dir", type=Path, required=True)
    parser.add_argument("--c3-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    b8_dir = args.b8_dir.expanduser().resolve()
    c0_dir = args.c0_dir.expanduser().resolve()
    c1_report_dir = args.c1_report_dir.expanduser().resolve()
    c1_model_dir = args.c1_model_dir.expanduser().resolve()
    c2_report_dir = args.c2_report_dir.expanduser().resolve()
    c2_model_dir = args.c2_model_dir.expanduser().resolve()
    c3_report_dir = args.c3_report_dir.expanduser().resolve()
    c3_model_dir = args.c3_model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output directory exists: {output_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True)
    archive_dir = output_dir / "archive"
    archive_dir.mkdir()

    failures: list[str] = []
    warnings: list[str] = []

    paths = {
        "b8_report": b8_dir / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE.json",
        "b8_lock": b8_dir / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE_LOCK.json",
        "b8_contract": b8_dir / "V5_P0_B8_B3_CONV1D_ONLY_ARCHITECTURE_CONTRACT.json",
        "b8_spec": b8_dir / "V5_P0_B8_ARCHITECTURE_SPEC.md",
        "b8_marker": b8_dir / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE_COMPLETE",
        "c0_report": c0_dir / "V5_P0_C0_FINAL_B3_TRAINING_PROTOCOL_LOCK.json",
        "c0_lock": c0_dir / "V5_P0_C0_FINAL_B3_TRAINING_PROTOCOL_LOCK_LOCK.json",
        "c0_protocol": c0_dir / "V5_P0_C0_FINAL_B3_PROTOCOL.json",
        "c0_protocol_md": c0_dir / "V5_P0_C0_FINAL_B3_PROTOCOL.md",
        "c0_marker": c0_dir / "V5_P0_C0_FINAL_B3_TRAINING_PROTOCOL_LOCK_COMPLETE",
        "c1_report": c1_report_dir / "V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING.json",
        "c1_lock": c1_report_dir / "V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING_LOCK.json",
        "c1_summary": c1_report_dir / "V5_P0_C1_SEED_SUMMARY.csv",
        "c1_marker": c1_report_dir / "V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING_COMPLETE",
        "c1_selected_checkpoint": c1_model_dir / "selected" / "selected_best_checkpoint.pt",
        "c2_report": c2_report_dir / "V5_P0_C2_FINAL_CHECKPOINT_THRESHOLD_FREEZE.json",
        "c2_lock": c2_report_dir / "V5_P0_C2_FINAL_CHECKPOINT_THRESHOLD_FREEZE_LOCK.json",
        "c2_manifest": c2_report_dir / "V5_P0_C2_FROZEN_CHECKPOINT_THRESHOLDS.json",
        "c2_attack_sweep": c2_report_dir / "V5_P0_C2_ATTACK_THRESHOLD_SWEEP.csv",
        "c2_role_sweep": c2_report_dir / "V5_P0_C2_ROLE_THRESHOLD_SWEEP.csv",
        "c2_marker": c2_report_dir / "V5_P0_C2_FINAL_CHECKPOINT_THRESHOLD_FREEZE_COMPLETE",
        "c2_checkpoint": c2_model_dir / "final_locked_b3_checkpoint.pt",
        "c3_report": c3_report_dir / "V5_P0_C3_ONE_SHOT_LOCKED_TEST_EVALUATION.json",
        "c3_lock": c3_report_dir / "V5_P0_C3_ONE_SHOT_LOCKED_TEST_EVALUATION_LOCK.json",
        "c3_access_marker": c3_report_dir / "V5_P0_C3_ONE_SHOT_LOCKED_TEST_EVALUATION_TEST_ACCESS_STARTED",
        "c3_marker": c3_report_dir / "V5_P0_C3_ONE_SHOT_LOCKED_TEST_EVALUATION_COMPLETE",
        "c3_predictions": c3_model_dir / "locked_test_predictions_and_targets.pt",
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing required artifact {name}: {path}")

    documents: dict[str, Any] = {}
    if not failures:
        for name in (
            "b8_report", "b8_lock", "b8_contract",
            "c0_report", "c0_lock", "c0_protocol",
            "c1_report", "c1_lock",
            "c2_report", "c2_lock", "c2_manifest",
            "c3_report", "c3_lock",
        ):
            documents[name] = load_json(paths[name])

        for label, report_name in (
            ("B8", "b8_report"),
            ("C0", "c0_report"),
            ("C1", "c1_report"),
            ("C2", "c2_report"),
            ("C3", "c3_report"),
        ):
            report = documents[report_name]
            if report.get("status") != "COMPLETE":
                failures.append(
                    f"{label} status={report.get('status')!r}, expected COMPLETE"
                )

        for label, report_name, lock_name in (
            ("B8", "b8_report", "b8_lock"),
            ("C0", "c0_report", "c0_lock"),
            ("C1", "c1_report", "c1_lock"),
            ("C2", "c2_report", "c2_lock"),
            ("C3", "c3_report", "c3_lock"),
        ):
            if (
                documents[lock_name].get("report_sha256")
                != sha256_file(paths[report_name])
            ):
                failures.append(f"{label} report SHA mismatch")

        b8_contract_hash = documents["b8_contract"].get(
            "architecture_contract_sha256"
        )
        c0_protocol_hash = documents["c0_protocol"].get("protocol_sha256")

        if (
            documents["c1_lock"].get("architecture_contract_sha256")
            != b8_contract_hash
        ):
            failures.append("C1 architecture-contract hash mismatch")
        if (
            documents["c2_lock"].get("architecture_contract_sha256")
            != b8_contract_hash
        ):
            failures.append("C2 architecture-contract hash mismatch")
        if (
            documents["c1_lock"].get("protocol_sha256")
            != c0_protocol_hash
        ):
            failures.append("C1 protocol hash mismatch")
        if (
            documents["c2_lock"].get("protocol_sha256")
            != c0_protocol_hash
        ):
            failures.append("C2 protocol hash mismatch")

        checkpoint_hash = sha256_file(paths["c2_checkpoint"])
        if (
            documents["c2_lock"].get("final_checkpoint_sha256")
            != checkpoint_hash
        ):
            failures.append("C2 final checkpoint hash mismatch")
        if (
            documents["c3_lock"].get("checkpoint_sha256")
            != checkpoint_hash
        ):
            failures.append("C3 checkpoint hash mismatch")
        if (
            documents["c3_report"].get("checkpoint", {}).get("sha256")
            != checkpoint_hash
        ):
            failures.append("C3 report checkpoint hash mismatch")

        threshold_manifest_file_hash = sha256_file(paths["c2_manifest"])
        if (
            documents["c2_lock"].get("threshold_manifest_file_sha256")
            != threshold_manifest_file_hash
        ):
            failures.append("C2 threshold-manifest file hash mismatch")
        if (
            documents["c3_lock"].get("threshold_manifest_file_sha256")
            != threshold_manifest_file_hash
        ):
            failures.append("C3 threshold-manifest hash mismatch")

        frozen_thresholds = documents["c2_lock"].get("frozen_thresholds")
        if documents["c3_lock"].get("frozen_thresholds") != frozen_thresholds:
            failures.append("C3 thresholds differ from C2 frozen thresholds")

        c3 = documents["c3_report"]
        if c3.get("decision") != "FINAL_P0_TEST_RESULT_RECORDED":
            failures.append("C3 does not record the final P0 result")
        if c3.get("test_split_accessed") is not True:
            failures.append("C3 does not record test access")
        if c3.get("test_evaluation_count") != 1:
            failures.append("C3 test evaluation count is not exactly one")
        if c3.get("training_performed") is not False:
            failures.append("C3 unexpectedly records training")
        if c3.get("checkpoint_selection_performed") is not False:
            failures.append("C3 unexpectedly records checkpoint selection")
        if c3.get("threshold_calibration_performed") is not False:
            failures.append("C3 unexpectedly records threshold calibration")

    if failures:
        hold = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "training_performed": False,
            "dataset_constructed": False,
        }
        write_json(output_dir / f"{STAGE}.json", hold)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    # Copy small, human-readable and machine-readable evidence.
    copied_records: list[dict[str, Any]] = []
    for key in (
        "b8_report", "b8_lock", "b8_contract", "b8_spec",
        "c0_report", "c0_lock", "c0_protocol", "c0_protocol_md",
        "c1_report", "c1_lock", "c1_summary",
        "c2_report", "c2_lock", "c2_manifest",
        "c2_attack_sweep", "c2_role_sweep",
        "c3_report", "c3_lock",
    ):
        copy_and_record(paths[key], archive_dir, copied_records)

    # Large binary files are referenced and hash-locked rather than duplicated.
    binary_references = {
        "c1_selected_checkpoint": {
            "path": str(paths["c1_selected_checkpoint"]),
            "sha256": sha256_file(paths["c1_selected_checkpoint"]),
            "size_bytes": paths["c1_selected_checkpoint"].stat().st_size,
        },
        "c2_final_checkpoint": {
            "path": str(paths["c2_checkpoint"]),
            "sha256": sha256_file(paths["c2_checkpoint"]),
            "size_bytes": paths["c2_checkpoint"].stat().st_size,
        },
        "c3_test_predictions": {
            "path": str(paths["c3_predictions"]),
            "sha256": sha256_file(paths["c3_predictions"]),
            "size_bytes": paths["c3_predictions"].stat().st_size,
        },
    }

    c2_report = documents["c2_report"]
    c3_report = documents["c3_report"]
    validation = c2_report["validation_metrics_at_frozen_thresholds"]
    test = c3_report["test_metrics"]

    summary = {
        "architecture": {
            "name": "B3_CAUSAL_DEPTHWISE_SEPARABLE_CONV1D_ONLY",
            "parameter_count": 43208,
            "architecture_contract_sha256": documents["b8_contract"][
                "architecture_contract_sha256"
            ],
        },
        "final_training": {
            "seeds": documents["c1_report"]["seeds"],
            "selected_seed": documents["c1_report"]["selected_checkpoint"]["seed"],
            "selected_epoch": documents["c1_report"]["selected_checkpoint"]["best_epoch"],
            "protocol_sha256": documents["c0_protocol"]["protocol_sha256"],
        },
        "checkpoint": {
            "sha256": sha256_file(paths["c2_checkpoint"]),
        },
        "frozen_thresholds": documents["c2_lock"]["frozen_thresholds"],
        "validation_at_frozen_thresholds": {
            "graph_balanced_accuracy": validation["graph"]["balanced_accuracy"],
            "graph_accuracy": validation["graph"]["accuracy"],
            "graph_precision": validation["graph"]["precision"],
            "graph_recall": validation["graph"]["recall"],
            "graph_f1": validation["graph"]["f1"],
            "graph_fpr": validation["graph"]["fpr"],
            "count_macro_f1": validation["count"]["macro_f1"],
            "source_f1_attack": validation["roles"]["source"]["attack_windows"]["node_f1"],
            "transit_f1_attack": validation["roles"]["transit"]["attack_windows"]["node_f1"],
            "victim_f1_attack": validation["roles"]["victim"]["attack_windows"]["node_f1"],
            "path_f1_attack": validation["roles"]["path"]["attack_windows"]["node_f1"],
            "all_tasks_exact": validation["all_tasks_exact"],
        },
        "official_locked_test": {
            "graph_balanced_accuracy": test["graph"]["balanced_accuracy"],
            "graph_accuracy": test["graph"]["accuracy"],
            "graph_precision": test["graph"]["precision"],
            "graph_recall": test["graph"]["recall"],
            "graph_f1": test["graph"]["f1"],
            "graph_fpr": test["graph"]["fpr"],
            "graph_confusion": {
                "tn": test["graph"]["tn"],
                "fp": test["graph"]["fp"],
                "fn": test["graph"]["fn"],
                "tp": test["graph"]["tp"],
            },
            "count_macro_f1": test["count"]["macro_f1"],
            "source_f1_attack": test["roles"]["source"]["attack_windows"]["node_f1"],
            "transit_f1_attack": test["roles"]["transit"]["attack_windows"]["node_f1"],
            "victim_f1_attack": test["roles"]["victim"]["attack_windows"]["node_f1"],
            "path_f1_attack": test["roles"]["path"]["attack_windows"]["node_f1"],
            "source_exact_attack": test["roles"]["source"]["attack_windows"]["exact_set"],
            "victim_exact_attack": test["roles"]["victim"]["attack_windows"]["exact_set"],
            "all_tasks_exact": test["all_tasks_exact"],
            "window_count": test["window_count"],
            "test_evaluation_count": c3_report["test_evaluation_count"],
        },
    }

    summary["validation_to_test_delta"] = {
        key: metric_delta(
            summary["official_locked_test"][key],
            summary["validation_at_frozen_thresholds"][key],
        )
        for key in (
            "graph_balanced_accuracy",
            "graph_accuracy",
            "graph_precision",
            "graph_recall",
            "graph_f1",
            "graph_fpr",
            "count_macro_f1",
            "source_f1_attack",
            "transit_f1_attack",
            "victim_f1_attack",
            "path_f1_attack",
            "all_tasks_exact",
        )
    }

    summary_path = output_dir / "V5_P0_FINAL_OFFICIAL_RESULT_SUMMARY.json"
    write_json(summary_path, summary)

    p2_handoff = {
        "handoff_name": "V5_P0_TO_P2_HANDOFF",
        "handoff_version": 1,
        "project_route": {
            "P0": "complete",
            "P1": "skipped_not_executed_by_project_decision",
            "P2": "next",
        },
        "next_stage": "V5_P2_INDEPENDENT_DATASET_AUDIT",
        "P0_test_status": {
            "consumed": True,
            "evaluation_count": 1,
            "must_not_be_reused_for_model_or_threshold_selection": True,
        },
        "P2_entry_requirements": [
            "independent P2 file and run inventory",
            "P2 split and leakage audit",
            "tensor shape and label-contract audit",
            "PRIMARY58 compatibility audit",
            "normalization provenance audit",
            "physical-mask recoverability audit",
            "window-32 and stride-8 compatibility audit",
            "metadata/provenance quarantine",
            "locked P2 test manifest before training",
        ],
        "frozen_B3_status_for_P2": {
            "role": "mandatory transferred baseline if P2 interface is compatible",
            "architecture_may_not_be_silently_changed": True,
            "architecture_contract_sha256": documents["b8_contract"][
                "architecture_contract_sha256"
            ],
            "P0_checkpoint_zero_shot_use": (
                "optional only after P2 audit and only on P2 validation or a "
                "separately locked transfer split"
            ),
            "P2_final_training": (
                "train the frozen B3 architecture from scratch on P2 train "
                "unless the explicitly named P2 study defines a different "
                "pre-registered objective"
            ),
        },
        "known_P0_limitation": {
            "primary_issue": "false positives on control traffic",
            "test_graph_fpr": test["graph"]["fpr"],
            "test_false_positives": test["graph"]["fp"],
            "test_false_negatives": test["graph"]["fn"],
            "test_graph_accuracy": test["graph"]["accuracy"],
            "test_graph_recall": test["graph"]["recall"],
        },
        "P2_test_boundary": {
            "do_not_access_before_checkpoint_and_threshold_freeze": True,
            "final_locked_evaluation_count": 1,
        },
    }
    p2_handoff["handoff_sha256"] = canonical_sha256(p2_handoff)

    handoff_path = output_dir / "V5_P0_TO_P2_HANDOFF.json"
    write_json(handoff_path, p2_handoff)

    markdown = f"""# V5 P0 Final Result and P2 Handoff

## Official P0 model

- B3 causal depthwise-separable Conv1D-only
- 43,208 parameters
- selected seed 117, epoch 101
- checkpoint SHA-256:
  `{summary["checkpoint"]["sha256"]}`

## Frozen thresholds

```text
attack  = {summary["frozen_thresholds"]["attack"]}
source  = {summary["frozen_thresholds"]["source"]}
transit = {summary["frozen_thresholds"]["transit"]}
victim  = {summary["frozen_thresholds"]["victim"]}
path    = {summary["frozen_thresholds"]["path"]}
```

## Official locked P0 test

```text
graph accuracy          = {summary["official_locked_test"]["graph_accuracy"]:.4f}
graph balanced accuracy = {summary["official_locked_test"]["graph_balanced_accuracy"]:.4f}
graph precision         = {summary["official_locked_test"]["graph_precision"]:.4f}
graph recall            = {summary["official_locked_test"]["graph_recall"]:.4f}
graph F1                = {summary["official_locked_test"]["graph_f1"]:.4f}
graph FPR               = {summary["official_locked_test"]["graph_fpr"]:.4f}

count macro F1          = {summary["official_locked_test"]["count_macro_f1"]:.4f}
source F1               = {summary["official_locked_test"]["source_f1_attack"]:.4f}
transit F1              = {summary["official_locked_test"]["transit_f1_attack"]:.4f}
victim F1               = {summary["official_locked_test"]["victim_f1_attack"]:.4f}
path F1                 = {summary["official_locked_test"]["path_f1_attack"]:.4f}
all-task exact          = {summary["official_locked_test"]["all_tasks_exact"]:.4f}
```

The P0 test was accessed exactly once. It must not be used again for model,
checkpoint, or threshold selection.

## Route correction

P1 is skipped and was not executed. The project proceeds directly to:

`V5_P2_INDEPENDENT_DATASET_AUDIT`

## Main P0 limitation

The model retained high attack recall but generated 56 false positives and 12
false negatives on the locked test. The primary P2 question is whether the
larger/more representative P2 dataset reduces control-traffic false positives
without sacrificing localization.

## P2 handoff hash

`{p2_handoff["handoff_sha256"]}`
"""
    markdown_path = output_dir / "V5_P0_FINAL_RESULT_AND_P2_HANDOFF.md"
    atomic_write(markdown_path, markdown)

    archive_manifest = {
        "copied_artifacts": copied_records,
        "binary_references": binary_references,
    }
    archive_manifest["archive_manifest_sha256"] = canonical_sha256(
        archive_manifest
    )
    archive_manifest_path = output_dir / "V5_P0_FINAL_ARCHIVE_MANIFEST.json"
    write_json(archive_manifest_path, archive_manifest)

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "decision": "ARCHIVE_P0_AND_PROCEED_DIRECTLY_TO_P2",
        "official_result_summary": summary,
        "P2_handoff": p2_handoff,
        "archive_manifest": archive_manifest,
        "artifacts": {
            "official_summary": [
                str(summary_path),
                sha256_file(summary_path),
            ],
            "P2_handoff": [
                str(handoff_path),
                sha256_file(handoff_path),
            ],
            "handoff_markdown": [
                str(markdown_path),
                sha256_file(markdown_path),
            ],
            "archive_manifest": [
                str(archive_manifest_path),
                sha256_file(archive_manifest_path),
            ],
        },
        "training_performed": False,
        "dataset_constructed": False,
        "test_evaluation_performed": False,
        "P0_test_already_consumed_in_C3": True,
        "P1_executed": False,
        "P1_status": "skipped_not_executed",
        "failures": failures,
        "warnings": warnings,
        "next_stage": "V5_P2_INDEPENDENT_DATASET_AUDIT",
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    lock = {
        "status": COMPLETE,
        "decision": "ARCHIVE_P0_AND_PROCEED_DIRECTLY_TO_P2",
        "report_sha256": sha256_file(report_path),
        "official_summary_sha256": sha256_file(summary_path),
        "P2_handoff_file_sha256": sha256_file(handoff_path),
        "P2_handoff_sha256": p2_handoff["handoff_sha256"],
        "handoff_markdown_sha256": sha256_file(markdown_path),
        "archive_manifest_file_sha256": sha256_file(
            archive_manifest_path
        ),
        "archive_manifest_sha256": archive_manifest[
            "archive_manifest_sha256"
        ],
        "P0_checkpoint_sha256": summary["checkpoint"]["sha256"],
        "P0_test_evaluation_count": 1,
        "P1_executed": False,
        "P2_next": True,
        "training_performed": False,
        "dataset_constructed": False,
        "test_evaluation_performed": False,
        "next_stage": "V5_P2_INDEPENDENT_DATASET_AUDIT",
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    official = summary["official_locked_test"]
    print("===== V5 P0 FINAL ARCHIVE + P2 HANDOFF =====")
    print("status: COMPLETE")
    print("decision: ARCHIVE_P0_AND_PROCEED_DIRECTLY_TO_P2")
    print("P1_executed: false")
    print("P1_status: skipped_not_executed")
    print("P2_next: true")
    print("checkpoint_sha256:", summary["checkpoint"]["sha256"])
    print("graph_accuracy:", f"{official['graph_accuracy']:.4f}")
    print(
        "graph_balanced_accuracy:",
        f"{official['graph_balanced_accuracy']:.4f}",
    )
    print("graph_recall:", f"{official['graph_recall']:.4f}")
    print("graph_f1:", f"{official['graph_f1']:.4f}")
    print("graph_fpr:", f"{official['graph_fpr']:.4f}")
    print("count_macro_f1:", f"{official['count_macro_f1']:.4f}")
    print("source_f1_attack:", f"{official['source_f1_attack']:.4f}")
    print("victim_f1_attack:", f"{official['victim_f1_attack']:.4f}")
    print("all_tasks_exact:", f"{official['all_tasks_exact']:.4f}")
    print("P0_test_evaluation_count: 1")
    print("training_performed: false")
    print("dataset_constructed: false")
    print("test_evaluation_performed: false")
    print("handoff_sha256:", p2_handoff["handoff_sha256"])
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("next_stage: V5_P2_INDEPENDENT_DATASET_AUDIT")
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
