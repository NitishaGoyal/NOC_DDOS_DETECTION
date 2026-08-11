from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = (
    "V5_P3_D13_EPLR_V2_FROZEN_IMPLEMENTATION_"
    "HANDOVER_AND_INDEPENDENT_VALIDATION_WAIT_STATE"
)
CAMPAIGN_LABEL = (
    "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic "
    "— EPLR-V2 Frozen Software Handover"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def deterministic_zip(
    archive_path: Path,
    files: list[tuple[Path, str]],
) -> None:
    temporary = archive_path.with_suffix(archive_path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for source, arcname in sorted(files, key=lambda item: item[1]):
            info = zipfile.ZipInfo(arcname)
            info.date_time = (2026, 8, 5, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())
    os.replace(temporary, archive_path)


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    wait_state_path = output_dir / (
        "V5_P3_EPLR_V2_INDEPENDENT_VALIDATION_WAIT_STATE.json"
    )
    handover_note_path = output_dir / (
        "V5_P3_EPLR_V2_HANDOVER_NOTE.md"
    )
    integration_contract_path = output_dir / (
        "V5_P3_EPLR_V2_INTEGRATION_CONTRACT.json"
    )
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    artifact_dir = (
        repo / "artifacts/decoders/v5_p3_eplr_v2_frozen_handover"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    handover_archive = artifact_dir / (
        "V5_P3_EPLR_V2_FROZEN_SOFTWARE_HANDOVER.zip"
    )

    d12_dir = (
        repo
        / "reports/v5/p3_d12_eplr_v2_implementation_and_synthetic_tests"
    )
    d12_report_path = d12_dir / (
        "V5_P3_D12_EPLR_V2_DETERMINISTIC_IMPLEMENTATION_"
        "AND_SYNTHETIC_TESTS_REPORT.json"
    )
    d12_lock_path = d12_dir / (
        "V5_P3_D12_EPLR_V2_DETERMINISTIC_IMPLEMENTATION_"
        "AND_SYNTHETIC_TESTS_LOCK.json"
    )
    d12_results_path = d12_dir / (
        "V5_P3_D12_EPLR_V2_SYNTHETIC_TEST_RESULTS.json"
    )

    d11_dir = repo / "reports/v5/p3_d11_eplr_v2_semantic_freeze"
    d11_report_path = d11_dir / (
        "V5_P3_D11_EPLR_V2_PRESERVE_OR_PASSTHROUGH_"
        "SEMANTIC_FREEZE_REPORT.json"
    )
    d11_lock_path = d11_dir / (
        "V5_P3_D11_EPLR_V2_PRESERVE_OR_PASSTHROUGH_"
        "SEMANTIC_FREEZE_LOCK.json"
    )
    d11_semantic_path = d11_dir / (
        "V5_P3_EPLR_V2_PRESERVE_OR_PASSTHROUGH_SEMANTIC_CONTRACT.json"
    )
    d11_output_path = d11_dir / (
        "V5_P3_EPLR_V2_SOFTWARE_OUTPUT_CONTRACT.json"
    )
    d11_future_protocol_path = d11_dir / (
        "V5_P3_EPLR_V2_FUTURE_INDEPENDENT_VALIDATION_PROTOCOL.json"
    )
    d11_governance_path = d11_dir / (
        "V5_P3_EPLR_V2_GOVERNANCE_NOTE.md"
    )

    d10_dir = (
        repo
        / "reports/v5/p3_d10_eplr_v1_comparison_and_carry_forward"
    )
    d10_decision_path = d10_dir / (
        "V5_P3_D10_EPLR_V1_CARRY_FORWARD_DECISION.md"
    )
    d10_root_cause_path = d10_dir / (
        "V5_P3_D10_EPLR_V1_STATUS_AND_REPAIR_ROOT_CAUSE_AUDIT.json"
    )

    decoder_path = repo / "src/decoders/v5_p3_eplr_v2.py"
    calibration_path = (
        repo / "src/decoders/v5_p3_eplr_v2_calibration.json"
    )
    route_library_path = repo / "src/decoders/v5_xy_route_library.py"
    v1_dependency_path = repo / "src/decoders/v5_p3_eplr_v1.py"

    required = [
        d12_report_path,
        d12_lock_path,
        d12_results_path,
        d11_report_path,
        d11_lock_path,
        d11_semantic_path,
        d11_output_path,
        d11_future_protocol_path,
        d11_governance_path,
        d10_decision_path,
        d10_root_cause_path,
        decoder_path,
        calibration_path,
        route_library_path,
        v1_dependency_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")

    d12 = json.loads(d12_report_path.read_text(encoding="utf-8"))
    d12_lock = json.loads(d12_lock_path.read_text(encoding="utf-8"))
    d11 = json.loads(d11_report_path.read_text(encoding="utf-8"))
    d11_lock = json.loads(d11_lock_path.read_text(encoding="utf-8"))

    if d12.get("status") != "PASS":
        raise RuntimeError("D12 is not PASS")
    if d12_lock.get("report_sha256") != sha256_file(d12_report_path):
        raise RuntimeError("D12 report/lock mismatch")
    if d12_lock.get("test_results_sha256") != sha256_file(
        d12_results_path
    ):
        raise RuntimeError("D12 test-results/lock mismatch")
    if d12_lock.get("EPLR_V2_source_sha256") != sha256_file(
        decoder_path
    ):
        raise RuntimeError("installed EPLR-V2 differs from D12 lock")
    if d12_lock.get("EPLR_V2_calibration_sha256") != sha256_file(
        calibration_path
    ):
        raise RuntimeError("installed calibration differs from D12 lock")
    if int(d12_lock.get("synthetic_tests", -1)) != 9:
        raise RuntimeError("unexpected D12 synthetic test count")
    if int(d12_lock.get("synthetic_tests_passed", -1)) != 9:
        raise RuntimeError("D12 synthetic tests are not all PASS")
    if d12_lock.get("A_validation_replay_authorized") is not False:
        raise RuntimeError("D12 unexpectedly authorizes A-validation")
    if d12_lock.get("A_test_data_loaded") is not False:
        raise RuntimeError("D12 reports A-test access")

    if d11.get("status") != "PASS":
        raise RuntimeError("D11 is not PASS")
    if d11_lock.get("report_sha256") != sha256_file(d11_report_path):
        raise RuntimeError("D11 report/lock mismatch")
    if d11_lock.get("semantic_contract_sha256") != sha256_file(
        d11_semantic_path
    ):
        raise RuntimeError("D11 semantic contract mismatch")
    if d11_lock.get("output_contract_sha256") != sha256_file(
        d11_output_path
    ):
        raise RuntimeError("D11 output contract mismatch")
    if d11_lock.get("future_validation_protocol_sha256") != sha256_file(
        d11_future_protocol_path
    ):
        raise RuntimeError("D11 future validation protocol mismatch")
    if d11_lock.get("A_validation_replay_authorized") is not False:
        raise RuntimeError("D11 unexpectedly authorizes A-validation")

    integration_contract = {
        "stage": STAGE,
        "status": "FROZEN",
        "decoder_identity": "V5-P3 EPLR-V2 Preserve-or-Passthrough",
        "software_only": True,
        "input_contract": {
            "raw_logit_count": 69,
            "layout": {
                "graph": [0, 1],
                "count": [1, 5],
                "source": [5, 21],
                "transit": [21, 37],
                "victim": [37, 53],
                "path": [53, 69],
            },
        },
        "immutable_predictions": [
            "graph equals Raw",
            "count candidate equals Raw",
            "source equals Raw on every active row",
            "victim equals Raw on every active row",
        ],
        "legal_endpoint_behavior": {
            "transit": "route-derived union",
            "path": "route-derived union",
            "route_certificate": True,
        },
        "unresolved_behavior": {
            "all_role_masks": "Raw passthrough",
            "route_ids": "empty",
            "route_certificate": False,
            "status": "RAW_ENDPOINTS_UNRESOLVED_PASSTHROUGH",
        },
        "calibration": {
            "source": "D8 A-train-only frozen calibration",
            "sha256": sha256_file(calibration_path),
            "recalibration_permitted_before_independent_validation": False,
        },
        "hardware_boundary": {
            "decoder_inside_RTL": False,
            "neural_accelerator_output_unchanged": True,
            "RTL_team_implements_EPLR": False,
        },
        "validation_boundary": {
            "Tranche_A_validation_replay_permitted": False,
            "A_test_permitted": False,
            "eligible_future_boundaries": [
                "frozen Tranche-B validation",
                "future fresh A+B validation",
            ],
        },
    }
    atomic_json(integration_contract_path, integration_contract)

    wait_state = {
        "stage": STAGE,
        "status": "WAITING_FOR_INDEPENDENT_VALIDATION_BOUNDARY",
        "implementation_complete": True,
        "synthetic_verification_complete": True,
        "current_validation_claim": False,
        "Tranche_A_validation_replay_authorized": False,
        "A_test_evaluation_authorized": False,
        "eligible_next_evaluation": [
            "frozen Tranche-B validation",
            "future fresh A+B validation under final protocol",
        ],
        "blocked_actions": [
            "retune EPLR-V2 using D9/D10 Tranche-A validation",
            "evaluate EPLR-V2 on Tranche-A validation",
            "evaluate EPLR-V2 on A-test",
            "claim D10 post-hoc counterfactual as frozen validation result",
        ],
        "parallel_paths": {
            "hardware": (
                "V5_P3_H1_DYNAMIC70_QUANTIZATION_AND_"
                "ACCUMULATOR_WIDTH_FREEZE"
            ),
            "dataset_governance": (
                "resume existing A6 Tranche-B pre-frozen specification review; "
                "generation remains separately governed"
            ),
            "regional_wrapper": "shared-weight 8x8 wrapper engineering",
        },
    }
    atomic_json(wait_state_path, wait_state)

    handover_note = f"""# V5-P3 EPLR-V2 frozen software handover

## Classification

**{CAMPAIGN_LABEL}**

This archive contains the frozen EPLR-V2 software decoder and its contracts.
It does not contain a successful EPLR-V2 validation result.

## Included implementation

- `v5_p3_eplr_v2.py`
- `v5_p3_eplr_v2_calibration.json`
- `v5_p3_eplr_v1.py` as the endpoint-preserving route-selection dependency
- `v5_xy_route_library.py`

## Frozen semantics

- graph, count, source and victim remain Raw;
- legal endpoint sets may receive route-derived transit/path unions;
- infeasible endpoint sets pass through all Raw role masks;
- unresolved rows carry no route certificate;
- endpoint repair is absent.

## Validation state

EPLR-V2 is waiting for an independent boundary.

Permitted:
- frozen Tranche-B validation;
- future fresh A+B validation.

Not permitted:
- another Tranche-A validation replay;
- A-test access;
- using the D10 post-hoc counterfactual as a frozen candidate claim.

## Hardware boundary

EPLR-V2 remains software-side. The RTL accelerator interface remains the
existing 69 raw logits.
"""
    atomic_text(handover_note_path, handover_note)

    files = [
        (decoder_path, "implementation/v5_p3_eplr_v2.py"),
        (
            calibration_path,
            "implementation/v5_p3_eplr_v2_calibration.json",
        ),
        (v1_dependency_path, "dependencies/v5_p3_eplr_v1.py"),
        (
            route_library_path,
            "dependencies/v5_xy_route_library.py",
        ),
        (d11_semantic_path, "contracts/eplr_v2_semantic_contract.json"),
        (d11_output_path, "contracts/eplr_v2_output_contract.json"),
        (
            d11_future_protocol_path,
            "contracts/future_independent_validation_protocol.json",
        ),
        (
            integration_contract_path,
            "contracts/integration_contract.json",
        ),
        (d12_report_path, "verification/d12_report.json"),
        (d12_lock_path, "verification/d12_lock.json"),
        (
            d12_results_path,
            "verification/d12_synthetic_test_results.json",
        ),
        (
            d10_root_cause_path,
            "provenance/d10_root_cause_audit.json",
        ),
        (
            d10_decision_path,
            "provenance/d10_carry_forward_decision.md",
        ),
        (wait_state_path, "WAIT_STATE.json"),
        (handover_note_path, "README.md"),
    ]
    deterministic_zip(handover_archive, files)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "scope": (
            "Create a frozen EPLR-V2 software handover archive and place the "
            "decoder in an explicit independent-validation wait state."
        ),
        "implementation": {
            "decoder_path": str(decoder_path),
            "decoder_sha256": sha256_file(decoder_path),
            "calibration_path": str(calibration_path),
            "calibration_sha256": sha256_file(calibration_path),
            "synthetic_tests": 9,
            "synthetic_tests_passed": 9,
        },
        "handover": {
            "archive_path": str(handover_archive),
            "archive_sha256": sha256_file(handover_archive),
            "archive_items": len(files),
            "integration_contract_path": str(integration_contract_path),
            "integration_contract_sha256": sha256_file(
                integration_contract_path
            ),
            "handover_note_path": str(handover_note_path),
            "handover_note_sha256": sha256_file(handover_note_path),
        },
        "wait_state": {
            "path": str(wait_state_path),
            "sha256": sha256_file(wait_state_path),
            "status": (
                "WAITING_FOR_INDEPENDENT_VALIDATION_BOUNDARY"
            ),
            "Tranche_A_validation_replay_authorized": False,
            "A_test_evaluation_authorized": False,
        },
        "decision": {
            "D13_complete": True,
            "decoder_branch_closed_until_independent_boundary": True,
            "H1_may_continue": True,
            "Tranche_B_governance_may_continue": True,
            "shared_8x8_wrapper_engineering_may_continue": True,
            "next_decoder_stage": (
                "none until frozen Tranche-B or future A+B validation exists"
            ),
            "next_hardware_stage": (
                "V5_P3_H1_DYNAMIC70_QUANTIZATION_AND_"
                "ACCUMULATOR_WIDTH_FREEZE"
            ),
            "next_dataset_governance_stage": (
                "resume existing A6 Tranche-B pre-frozen specification review"
            ),
        },
        "data_access": {
            "model_loaded": False,
            "training_data_loaded": False,
            "Tranche_A_validation_loaded": False,
            "A_test_loaded": False,
        },
        "provenance": {
            "D12_report_sha256": sha256_file(d12_report_path),
            "D12_lock_sha256": sha256_file(d12_lock_path),
            "D12_test_results_sha256": sha256_file(d12_results_path),
            "D11_report_sha256": sha256_file(d11_report_path),
            "D11_lock_sha256": sha256_file(d11_lock_path),
            "D10_root_cause_sha256": sha256_file(d10_root_cause_path),
            "D10_decision_sha256": sha256_file(d10_decision_path),
            "installed_script_sha256": sha256_file(installed_script),
        },
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(report_path),
        "handover_archive_sha256": sha256_file(handover_archive),
        "integration_contract_sha256": sha256_file(
            integration_contract_path
        ),
        "wait_state_sha256": sha256_file(wait_state_path),
        "decoder_sha256": sha256_file(decoder_path),
        "calibration_sha256": sha256_file(calibration_path),
        "Tranche_A_validation_replay_authorized": False,
        "A_test_loaded": False,
    }
    atomic_json(lock_path, lock)
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign_label={CAMPAIGN_LABEL}")
    print("implementation_complete=true")
    print("synthetic_tests_passed=9")
    print("decoder_branch_wait_state=true")
    print(
        "wait_state=WAITING_FOR_INDEPENDENT_VALIDATION_BOUNDARY"
    )
    print("Tranche_A_validation_replay_authorized=false")
    print("A_test_loaded=false")
    print("H1_may_continue=true")
    print("Tranche_B_governance_may_continue=true")
    print("shared_8x8_wrapper_engineering_may_continue=true")
    print(f"handover_archive={handover_archive}")
    print(
        "handover_archive_sha256="
        f"{sha256_file(handover_archive)}"
    )
    print(
        "next_hardware_stage="
        "V5_P3_H1_DYNAMIC70_QUANTIZATION_AND_"
        "ACCUMULATOR_WIDTH_FREEZE"
    )
    print(
        "next_dataset_governance_stage="
        "resume_existing_A6_Tranche_B_prefrozen_spec_review"
    )
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
