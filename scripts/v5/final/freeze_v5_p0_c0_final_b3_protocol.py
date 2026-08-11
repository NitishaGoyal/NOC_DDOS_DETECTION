#!/usr/bin/env python3
"""
V5 P0-C0 Final B3 Training Protocol Lock

This stage performs no model training and never constructs the P0 test split.

It verifies the completed P0-B8 architecture freeze and writes the immutable
protocol for:
- C1 fresh-seed validation-monitored B3 training;
- C2 validation-only checkpoint and threshold freeze;
- C3 one-shot locked P0 test evaluation.

The selected architecture remains the B3 causal depthwise-separable
Conv1D-only model. Existing P0 checkpoints are not selected here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


STAGE = "V5_P0_C0_FINAL_B3_TRAINING_PROTOCOL_LOCK"
COMPLETE = f"{STAGE}_COMPLETE"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b8-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--b3-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    b8_dir = args.b8_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    wrapper_path = args.wrapper.expanduser().resolve()
    b3_script_path = args.b3_script.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(
            f"STOP: output directory already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    b8_report_path = (
        b8_dir
        / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE.json"
    )
    b8_lock_path = (
        b8_dir
        / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE_LOCK.json"
    )
    b8_contract_path = (
        b8_dir
        / "V5_P0_B8_B3_CONV1D_ONLY_ARCHITECTURE_CONTRACT.json"
    )
    b8_marker_path = (
        b8_dir
        / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE_COMPLETE"
    )

    required = (
        b8_report_path,
        b8_lock_path,
        b8_contract_path,
        b8_marker_path,
        wrapper_path,
        b3_script_path,
    )
    for path in required:
        if not path.is_file():
            failures.append(f"missing prerequisite file: {path}")
    if not data_root.is_dir():
        failures.append(f"P0 data root does not exist: {data_root}")

    b8_report: dict[str, Any] = {}
    b8_lock: dict[str, Any] = {}
    b8_contract: dict[str, Any] = {}

    if not failures:
        b8_report = load_json(b8_report_path)
        b8_lock = load_json(b8_lock_path)
        b8_contract = load_json(b8_contract_path)

        if b8_report.get("status") != "COMPLETE":
            failures.append("B8 report status is not COMPLETE")
        if b8_report.get("decision") != "FREEZE_B3_CONV1D_ONLY":
            failures.append("B8 did not freeze B3")
        if b8_report.get("training_performed") is not False:
            failures.append("B8 unexpectedly records training")
        if b8_report.get("test_split_accessed") is not False:
            failures.append("B8 does not certify untouched P0 test")

        if b8_lock.get("report_sha256") != sha256_file(b8_report_path):
            failures.append("B8 report SHA does not match B8 lock")
        if (
            b8_lock.get("contract_file_sha256")
            != sha256_file(b8_contract_path)
        ):
            failures.append("B8 contract-file SHA does not match B8 lock")
        if (
            b8_lock.get("architecture_contract_sha256")
            != b8_contract.get("architecture_contract_sha256")
        ):
            failures.append("B8 embedded architecture-contract hash mismatch")

        selected = b8_contract.get("selected_architecture", {})
        if (
            selected.get("name")
            != "B3_CAUSAL_DEPTHWISE_SEPARABLE_CONV1D_ONLY"
        ):
            failures.append("B8 contract selected architecture is not B3")
        if selected.get("parameter_count") != 43208:
            failures.append("B8 B3 parameter count is not 43,208")
        if selected.get("graph_message_passing") is not False:
            failures.append("B8 selected architecture unexpectedly uses graph MP")

        input_contract = b8_contract.get("input", {})
        if input_contract.get("feature_variant") != "PRIMARY58":
            failures.append("B8 feature variant is not PRIMARY58")
        if input_contract.get("window") != 32:
            failures.append("B8 window is not 32")
        if input_contract.get("stride") != 8:
            failures.append("B8 stride is not 8")

    if failures:
        hold = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "training_performed": False,
            "validation_split_constructed": False,
            "test_split_constructed": False,
            "test_split_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", hold)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    protocol = {
        "protocol_name": "V5_P0_FINAL_B3_TRAINING_SELECTION_AND_TEST",
        "protocol_version": 1,
        "architecture_contract": {
            "name": "V5_P0_FROZEN_B3_CONV1D_ONLY",
            "architecture_contract_sha256": (
                b8_contract["architecture_contract_sha256"]
            ),
            "parameter_count": 43208,
            "weights_frozen": False,
            "train_from_scratch": True,
        },
        "data_contract": {
            "root": str(data_root),
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "feature_variant": "PRIMARY58",
            "x_shape": ["B", 16, 58, 32],
            "window": 32,
            "stride": 8,
            "active_only": False,
            "stored_x_already_standardized": True,
            "second_normalization_forbidden": True,
            "physical_port_mask": {
                "shape": ["B", 16, 10],
                "dtype": "bool",
                "source": "topology-derived recovered binary mask",
            },
            "metadata_as_model_input": False,
            "test_split_boundary": (
                "C0, C1, and C2 must not construct, enumerate, load, "
                "or evaluate the P0 test split"
            ),
        },
        "fresh_training_seeds": [107, 117, 127],
        "seed_rationale": (
            "Fresh seeds are used because seeds 7, 17, 27 and 101 were "
            "already observed during architecture-selection experiments."
        ),
        "training": {
            "model_initialization": "from scratch for every seed",
            "maximum_epochs": 150,
            "minimum_epochs_before_early_stop": 25,
            "batch_size": 128,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 5.0,
            "losses_and_weights": {
                "attack": ["BCEWithLogitsLoss", 1.0],
                "attacker_count": ["CrossEntropyLoss", 0.5],
                "source": ["BCEWithLogitsLoss", 1.0],
                "transit": ["BCEWithLogitsLoss", 0.5],
                "victim": ["BCEWithLogitsLoss", 0.75],
                "attack_path": ["BCEWithLogitsLoss", 0.5],
            },
            "positive_weight_rule": "min(20, negatives/positives)",
            "scheduler": {
                "type": "ReduceLROnPlateau",
                "mode": "max",
                "factor": 0.5,
                "patience_epochs": 6,
                "threshold": 0.0001,
                "minimum_learning_rate": 0.00001,
            },
            "early_stopping": {
                "enabled": True,
                "patience_epochs": 20,
                "minimum_delta": 0.0001,
                "minimum_training_epochs": 25,
                "restore_best_checkpoint": True,
            },
            "validation_frequency_epochs": 1,
            "checkpoint_storage": {
                "save_best_only": True,
                "save_final_epoch_for_diagnostics": True,
                "optimizer_state_in_best_checkpoint": True,
            },
        },
        "checkpoint_selection": {
            "thresholds_during_selection": {
                "attack": 0.5,
                "source": 0.5,
                "transit": 0.5,
                "victim": 0.5,
                "attack_path": 0.5,
            },
            "graph_gating_of_role_predictions": False,
            "selection_score": {
                "formula": (
                    "0.25*graph_balanced_accuracy"
                    " + 0.20*graph_f1"
                    " + 0.10*count_macro_f1"
                    " + 0.10*source_f1_attack"
                    " + 0.10*victim_f1_attack"
                    " + 0.05*transit_f1_attack"
                    " + 0.05*path_f1_attack"
                    " + 0.15*all_tasks_exact"
                    " - 0.20*max(0,0.90-graph_recall)"
                    " - 0.10*max(0,graph_fpr-0.20)"
                ),
                "higher_is_better": True,
                "all_component_metrics_range": [0.0, 1.0],
            },
            "epoch_tie_break_order": [
                "lower graph false negatives",
                "lower graph false-positive rate",
                "higher victim F1 on attack windows",
                "earlier epoch",
            ],
            "seed_selection": (
                "Select the seed's best checkpoint with the highest fixed "
                "validation selection score; use the same tie breaks."
            ),
            "validation_only": True,
            "test_information_forbidden": True,
        },
        "threshold_calibration": {
            "stage": "C2 only",
            "checkpoint": "single validation-selected checkpoint from C1",
            "candidate_thresholds": [
                round(0.10 + 0.05 * index, 2)
                for index in range(17)
            ],
            "attack": {
                "eligibility": "validation FPR <= 0.15",
                "maximize_in_order": [
                    "graph recall",
                    "graph F1",
                    "graph balanced accuracy",
                    "closest threshold to 0.5",
                ],
                "fallback_if_no_eligible_threshold": [
                    "minimum graph FPR",
                    "maximum graph recall",
                    "maximum graph F1",
                    "closest threshold to 0.5",
                ],
            },
            "source_and_victim": {
                "calibration_scope": "ground-truth validation attack windows",
                "maximize_in_order": [
                    "node F1",
                    "exact set accuracy",
                    "node precision",
                    "closest threshold to 0.5",
                ],
            },
            "transit_and_path": {
                "calibration_scope": "ground-truth validation attack windows",
                "maximize_in_order": [
                    "node F1",
                    "exact set accuracy",
                    "node recall",
                    "closest threshold to 0.5",
                ],
            },
            "attacker_count": "argmax; no threshold",
            "graph_gating_of_role_predictions": False,
            "validation_only": True,
            "test_information_forbidden": True,
        },
        "c3_one_shot_test": {
            "required_inputs": [
                "C1 selected checkpoint hash",
                "C2 threshold-freeze hash",
                "C0 protocol hash",
                "B8 architecture-contract hash",
            ],
            "test_evaluation_count": 1,
            "checkpoint_selection_on_test": False,
            "threshold_calibration_on_test": False,
            "retraining_after_test": False,
            "report_all_metrics": True,
        },
        "stages": {
            "C1": "V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING",
            "C2": "V5_P0_C2_FINAL_CHECKPOINT_THRESHOLD_FREEZE",
            "C3": "V5_P0_C3_ONE_SHOT_LOCKED_TEST_EVALUATION",
            "after_C3": "V5_P1_INDEPENDENT_DATASET_AUDIT",
        },
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)

    protocol_path = output_dir / "V5_P0_C0_FINAL_B3_PROTOCOL.json"
    protocol_md_path = output_dir / "V5_P0_C0_FINAL_B3_PROTOCOL.md"
    write_json(protocol_path, protocol)

    markdown = f"""# V5 P0-C0 Final B3 Training Protocol

## Frozen architecture

- B3 causal depthwise-separable Conv1D-only
- 43,208 trainable parameters
- PRIMARY58, window 32, stride 8
- recovered topology-derived Boolean mask
- no graph message passing
- no metadata or provenance inputs

## Fresh seeds

`107, 117, 127`

These are fresh relative to the architecture-search seeds.

## Training

- train from scratch per seed
- maximum 150 epochs
- minimum 25 epochs
- AdamW, learning rate `1e-3`, weight decay `1e-4`
- batch size 128
- gradient clipping at 5.0
- ReduceLROnPlateau, factor 0.5, patience 6
- early-stopping patience 20
- best-checkpoint restoration

## Model selection

Validation metrics are computed every epoch at fixed threshold 0.5.

```text
score =
  0.25 * graph balanced accuracy
+ 0.20 * graph F1
+ 0.10 * count macro F1
+ 0.10 * source F1
+ 0.10 * victim F1
+ 0.05 * transit F1
+ 0.05 * path F1
+ 0.15 * all-task exact
- 0.20 * max(0, 0.90 - graph recall)
- 0.10 * max(0, graph FPR - 0.20)
```

The best checkpoint from each seed is retained. One final checkpoint is selected
using validation only.

## Threshold calibration

C2 calibrates only the selected checkpoint on validation using the fixed grid:

`0.10, 0.15, ..., 0.90`

The resulting checkpoint and thresholds are hashed and frozen before test access.

## Test rule

C0, C1, and C2 must not construct or inspect the test split. C3 evaluates the
locked test exactly once.

## Protocol hash

`{protocol["protocol_sha256"]}`
"""
    atomic_write(protocol_md_path, markdown)

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "decision": "LOCK_FINAL_P0_B3_TRAINING_PROTOCOL",
        "architecture_contract_sha256": (
            b8_contract["architecture_contract_sha256"]
        ),
        "protocol": protocol,
        "artifacts": {
            "protocol_json": [
                str(protocol_path),
                sha256_file(protocol_path),
            ],
            "protocol_markdown": [
                str(protocol_md_path),
                sha256_file(protocol_md_path),
            ],
        },
        "provenance": {
            "b8_report_sha256": sha256_file(b8_report_path),
            "b8_lock_sha256": sha256_file(b8_lock_path),
            "b8_contract_file_sha256": sha256_file(b8_contract_path),
            "wrapper_sha256": sha256_file(wrapper_path),
            "b3_reference_script_sha256": sha256_file(b3_script_path),
            "c0_script_sha256": sha256_file(Path(__file__)),
        },
        "training_performed": False,
        "train_split_constructed": False,
        "validation_split_constructed": False,
        "test_split_constructed": False,
        "test_split_accessed": False,
        "failures": failures,
        "warnings": warnings,
        "next_stage": "V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING",
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    lock = {
        "status": COMPLETE,
        "decision": "LOCK_FINAL_P0_B3_TRAINING_PROTOCOL",
        "protocol_sha256": protocol["protocol_sha256"],
        "report_sha256": sha256_file(report_path),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_markdown_sha256": sha256_file(protocol_md_path),
        "architecture_contract_sha256": (
            b8_contract["architecture_contract_sha256"]
        ),
        "c0_script_sha256": sha256_file(Path(__file__)),
        "training_performed": False,
        "test_split_constructed": False,
        "test_split_accessed": False,
        "next_stage": "V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING",
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P0-C0 FINAL B3 PROTOCOL LOCK =====")
    print("status: COMPLETE")
    print("architecture: B3_CAUSAL_DEPTHWISE_SEPARABLE_CONV1D_ONLY")
    print("parameter_count: 43208")
    print("seeds: [107, 117, 127]")
    print("maximum_epochs: 150")
    print("minimum_epochs: 25")
    print("early_stopping_patience: 20")
    print("scheduler: ReduceLROnPlateau")
    print("threshold_selection_grid: 0.10..0.90 step 0.05")
    print("protocol_sha256:", protocol["protocol_sha256"])
    print("training_performed: false")
    print("test_split_constructed: false")
    print("test_split_accessed: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("next_stage: V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING")
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
