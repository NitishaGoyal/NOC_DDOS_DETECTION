#!/usr/bin/env python3
"""
V5 P2-B5 One-Shot Test Authorization

This stage validates the fully frozen P2 chain:
- B1 training protocol;
- B3 selected checkpoint;
- B4 validation-only thresholds.

It then creates one immutable authorization token for exactly one P2 test
evaluation.

This stage does NOT:
- enumerate the test directory;
- open any test tensor;
- deserialize the selected checkpoint;
- construct any dataset;
- perform inference.

The next stage must consume the authorization token exactly once before
opening test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


STAGE = "V5_P2_B5_ONE_SHOT_TEST_AUTHORIZATION"
COMPLETE = f"{STAGE}_COMPLETE"

EXPECTED_PROTOCOL_SHA = (
    "0817ae7812f91e3c75260589acaf52bd8a74b07b2114a451b4773578efde4b60"
)
EXPECTED_SELECTED_SEED = 127
EXPECTED_SELECTED_EPOCH = 59
EXPECTED_CHECKPOINT_SHA = (
    "7d4afae2214f67ecd9c65c6ff4614ba8238614234d7b07cdf1408b6d3efb07ef"
)
EXPECTED_THRESHOLD_MANIFEST_SHA = (
    "64bb23f1d0f36ec5f09236c4939317cddc96260f0ed4695036c4e802cd3056a7"
)
EXPECTED_THRESHOLDS = {
    "attack": 0.2,
    "source": 0.778,
    "transit": 0.824,
    "victim": 0.661,
    "path": 0.545,
}


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
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
    atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def explicit_false(
    document: dict[str, Any],
    key: str,
) -> bool:
    boundary = document.get("security_boundary")
    if isinstance(boundary, dict) and key in boundary:
        return boundary[key] is False
    return document.get(key) is False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1-dir", type=Path, required=True)
    parser.add_argument("--b3-dir", type=Path, required=True)
    parser.add_argument("--b4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    b1_dir = args.b1_dir.expanduser().resolve()
    b3_dir = args.b3_dir.expanduser().resolve()
    b4_dir = args.b4_dir.expanduser().resolve()
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

    paths = {
        "b1_report": (
            b1_dir
            / "V5_P2_B1_TRAINING_PROTOCOL_LOCK.json"
        ),
        "b1_lock": (
            b1_dir
            / "V5_P2_B1_TRAINING_PROTOCOL_LOCK_LOCK.json"
        ),
        "b1_protocol": (
            b1_dir
            / "V5_P2_B1_TRAINING_PROTOCOL.json"
        ),
        "b3_report": (
            b3_dir
            / "V5_P2_B3_SEED_AND_CHECKPOINT_SELECTION.json"
        ),
        "b3_lock": (
            b3_dir
            / "V5_P2_B3_SEED_AND_CHECKPOINT_SELECTION_LOCK.json"
        ),
        "b3_manifest": (
            b3_dir
            / "V5_P2_B3_SELECTED_CHECKPOINT_MANIFEST.json"
        ),
        "b4_report": (
            b4_dir
            / "V5_P2_B4_VALIDATION_THRESHOLD_TUNING.json"
        ),
        "b4_lock": (
            b4_dir
            / "V5_P2_B4_VALIDATION_THRESHOLD_TUNING_LOCK.json"
        ),
        "b4_threshold_manifest": (
            b4_dir
            / "V5_P2_B4_FROZEN_THRESHOLD_MANIFEST.json"
        ),
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(
                f"missing prerequisite {name}: {path}"
            )

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
            "test_evaluation_performed": False,
            "test_evaluation_authorized": False,
        }
        write_json(
            output_dir / f"{STAGE}.json",
            report,
        )
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        return 1

    b1_report = load_json(paths["b1_report"])
    b1_lock = load_json(paths["b1_lock"])
    b1_protocol = load_json(paths["b1_protocol"])
    b3_report = load_json(paths["b3_report"])
    b3_lock = load_json(paths["b3_lock"])
    b3_manifest = load_json(paths["b3_manifest"])
    b4_report = load_json(paths["b4_report"])
    b4_lock = load_json(paths["b4_lock"])
    b4_manifest = load_json(
        paths["b4_threshold_manifest"]
    )

    # B1
    if b1_report.get("status") != "COMPLETE":
        failures.append("B1 status is not COMPLETE")
    if (
        b1_lock.get("report_sha256")
        != sha256_file(paths["b1_report"])
    ):
        failures.append("B1 report SHA mismatch")
    if (
        b1_lock.get("protocol_file_sha256")
        != sha256_file(paths["b1_protocol"])
    ):
        failures.append("B1 protocol-file SHA mismatch")
    if (
        b1_protocol.get("protocol_sha256")
        != EXPECTED_PROTOCOL_SHA
    ):
        failures.append("B1 protocol SHA changed")
    if b1_lock.get("test_evaluation_authorized") is not False:
        failures.append("B1 unexpectedly authorized test")

    # B3
    if b3_report.get("status") != "COMPLETE":
        failures.append("B3 status is not COMPLETE")
    if (
        b3_lock.get("report_sha256")
        != sha256_file(paths["b3_report"])
    ):
        failures.append("B3 report SHA mismatch")
    if (
        b3_lock.get("selection_manifest_sha256")
        != sha256_file(paths["b3_manifest"])
    ):
        failures.append("B3 selection-manifest SHA mismatch")
    if b3_lock.get("selected_seed") != EXPECTED_SELECTED_SEED:
        failures.append("B3 selected seed changed")
    if (
        b3_lock.get("selected_best_epoch")
        != EXPECTED_SELECTED_EPOCH
    ):
        failures.append("B3 selected epoch changed")
    if (
        b3_lock.get("selected_checkpoint_sha256")
        != EXPECTED_CHECKPOINT_SHA
    ):
        failures.append("B3 selected checkpoint SHA changed")
    if b3_lock.get("selection_uses_validation_only") is not True:
        failures.append("B3 was not validation-only")
    if b3_lock.get("threshold_tuning_performed") is not False:
        failures.append("B3 already performed threshold tuning")
    if b3_lock.get("test_evaluation_authorized") is not False:
        failures.append("B3 unexpectedly authorized test")

    selected_checkpoint = Path(
        b3_lock["selected_checkpoint_path"]
    ).resolve()
    if not selected_checkpoint.is_file():
        failures.append(
            f"selected checkpoint missing: {selected_checkpoint}"
        )
    elif (
        sha256_file(selected_checkpoint)
        != EXPECTED_CHECKPOINT_SHA
    ):
        failures.append(
            "selected checkpoint file SHA mismatch"
        )

    # B4
    if b4_report.get("status") != "COMPLETE":
        failures.append("B4 status is not COMPLETE")
    if (
        b4_lock.get("report_sha256")
        != sha256_file(paths["b4_report"])
    ):
        failures.append("B4 report SHA mismatch")
    if (
        b4_lock.get("threshold_manifest_file_sha256")
        != sha256_file(paths["b4_threshold_manifest"])
    ):
        failures.append(
            "B4 threshold-manifest file SHA mismatch"
        )
    if (
        b4_lock.get("threshold_manifest_sha256")
        != EXPECTED_THRESHOLD_MANIFEST_SHA
    ):
        failures.append(
            "B4 canonical threshold-manifest SHA changed"
        )
    if (
        b4_manifest.get("threshold_manifest_sha256")
        != EXPECTED_THRESHOLD_MANIFEST_SHA
    ):
        failures.append(
            "B4 threshold manifest stores a different canonical SHA"
        )
    if (
        b4_lock.get("selected_checkpoint_sha256")
        != EXPECTED_CHECKPOINT_SHA
    ):
        failures.append(
            "B4 selected-checkpoint SHA changed"
        )
    if b4_lock.get("selected_seed") != EXPECTED_SELECTED_SEED:
        failures.append("B4 selected seed changed")
    if b4_lock.get("selected_epoch") != EXPECTED_SELECTED_EPOCH:
        failures.append("B4 selected epoch changed")
    if b4_lock.get("training_performed") is not False:
        failures.append("B4 performed training")
    if b4_lock.get("model_weights_changed") is not False:
        failures.append("B4 changed model weights")
    if b4_lock.get("test_evaluation_authorized") is not False:
        failures.append("B4 unexpectedly authorized test")

    observed_thresholds = b4_lock.get("thresholds")
    if observed_thresholds != EXPECTED_THRESHOLDS:
        failures.append(
            f"B4 thresholds={observed_thresholds}, "
            f"expected={EXPECTED_THRESHOLDS}"
        )
    if b4_manifest.get("thresholds") != EXPECTED_THRESHOLDS:
        failures.append(
            "B4 threshold manifest values changed"
        )
    if b4_lock.get("count_decision") != "argmax":
        failures.append("B4 count decision changed")

    # Every previous stage must certify no test access.
    for label, document in (
        ("B1", b1_report),
        ("B3", b3_report),
        ("B4", b4_report),
    ):
        if not explicit_false(
            document,
            "test_tensor_contents_accessed",
        ):
            failures.append(
                f"{label} does not certify untouched test tensors"
            )

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "selected_checkpoint_deserialized": False,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
            "test_evaluation_performed": False,
            "test_evaluation_authorized": False,
        }
        write_json(
            output_dir / f"{STAGE}.json",
            report,
        )
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    authorization_core = {
        "authorization_name": (
            "V5_P2_SINGLE_USE_BLIND_TEST_EVALUATION"
        ),
        "authorization_version": 1,
        "authorized_evaluation_count": 1,
        "authorized_stage": (
            "V5_P2_B6_ONE_SHOT_TEST_EVALUATION"
        ),
        "selected_seed": EXPECTED_SELECTED_SEED,
        "selected_epoch": EXPECTED_SELECTED_EPOCH,
        "selected_checkpoint_path": str(
            selected_checkpoint
        ),
        "selected_checkpoint_sha256": (
            EXPECTED_CHECKPOINT_SHA
        ),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "threshold_manifest_path": str(
            paths["b4_threshold_manifest"]
        ),
        "threshold_manifest_file_sha256": (
            sha256_file(paths["b4_threshold_manifest"])
        ),
        "threshold_manifest_sha256": (
            EXPECTED_THRESHOLD_MANIFEST_SHA
        ),
        "thresholds": EXPECTED_THRESHOLDS,
        "count_decision": {
            "method": "argmax",
            "class_index_to_raw_count": {
                "0": 1,
                "1": 2,
                "2": 3,
                "3": 4,
            },
        },
        "evaluation_contract": {
            "test_split": "P2 test only",
            "model_weights_must_not_change": True,
            "thresholds_must_not_change": True,
            "no_post_hoc_threshold_adjustment": True,
            "no_checkpoint_reselection": True,
            "no_seed_reselection": True,
            "no_test_driven_analysis_before_primary_report": True,
            "one_complete_test_pass_only": True,
            "rerun_after_success_forbidden": True,
            "test_prediction_cache_may_be_written_once": True,
            "primary_metrics_must_be_written_before_error_analysis": True,
        },
        "authorization_consumed": False,
        "test_evaluation_performed": False,
        "test_directory_enumerated_at_authorization": False,
        "test_tensor_contents_accessed_at_authorization": False,
    }
    authorization_id = canonical_sha256(
        authorization_core
    )
    authorization = {
        **authorization_core,
        "authorization_id": authorization_id,
    }

    token_path = (
        output_dir
        / "V5_P2_B5_ONE_SHOT_TEST_AUTHORIZATION_TOKEN.json"
    )
    write_json(token_path, authorization)

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "decision": (
            "AUTHORIZE_EXACTLY_ONE_P2_TEST_EVALUATION"
        ),
        "authorization": authorization,
        "artifacts": {
            "authorization_token": [
                str(token_path),
                sha256_file(token_path),
            ],
        },
        "provenance": {
            name: sha256_file(path)
            for name, path in paths.items()
        },
        "security_boundary": {
            "selected_checkpoint_deserialized": False,
            "dataset_constructed": False,
            "run_tensors_deserialized": False,
            "test_directory_existence_checked": False,
            "test_directory_enumerated": False,
            "test_tensor_files_opened": False,
            "test_tensor_bytes_read": False,
            "test_tensor_contents_accessed": False,
            "test_dataset_constructed": False,
            "test_windows_constructed": False,
            "test_evaluation_performed": False,
            "test_evaluation_authorized": True,
            "authorized_evaluation_count": 1,
        },
        "failures": failures,
        "warnings": warnings,
        "next_stage": (
            "V5_P2_B6_ONE_SHOT_TEST_EVALUATION"
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    lock = {
        "status": COMPLETE,
        "decision": (
            "AUTHORIZE_EXACTLY_ONE_P2_TEST_EVALUATION"
        ),
        "report_sha256": sha256_file(report_path),
        "authorization_token_sha256": (
            sha256_file(token_path)
        ),
        "authorization_id": authorization_id,
        "authorized_evaluation_count": 1,
        "authorization_consumed": False,
        "selected_seed": EXPECTED_SELECTED_SEED,
        "selected_epoch": EXPECTED_SELECTED_EPOCH,
        "selected_checkpoint_sha256": (
            EXPECTED_CHECKPOINT_SHA
        ),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "threshold_manifest_sha256": (
            EXPECTED_THRESHOLD_MANIFEST_SHA
        ),
        "thresholds": EXPECTED_THRESHOLDS,
        "count_decision": "argmax",
        "selected_checkpoint_deserialized": False,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "test_evaluation_performed": False,
        "test_evaluation_authorized": True,
        "next_stage": (
            "V5_P2_B6_ONE_SHOT_TEST_EVALUATION"
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

    print("===== V5 P2-B5 ONE-SHOT TEST AUTHORIZATION =====")
    print("status: COMPLETE")
    print(
        "decision: AUTHORIZE_EXACTLY_ONE_P2_TEST_EVALUATION"
    )
    print("selected_seed:", EXPECTED_SELECTED_SEED)
    print("selected_epoch:", EXPECTED_SELECTED_EPOCH)
    print(
        "selected_checkpoint_sha256:",
        EXPECTED_CHECKPOINT_SHA,
    )
    print(
        "threshold_manifest_sha256:",
        EXPECTED_THRESHOLD_MANIFEST_SHA,
    )
    for name, value in EXPECTED_THRESHOLDS.items():
        print(f"{name}_threshold:", value)
    print("count_decision: argmax")
    print("authorized_evaluation_count: 1")
    print("authorization_consumed: false")
    print("selected_checkpoint_deserialized: false")
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")
    print("test_evaluation_performed: false")
    print("test_evaluation_authorized: true")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("authorization_id:", authorization_id)
    print(
        "next_stage: V5_P2_B6_ONE_SHOT_TEST_EVALUATION"
    )
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
