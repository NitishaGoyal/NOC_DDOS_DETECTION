#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGE = "V5_P2_D6_POST_HOC_NUMERICAL_RECOVERY_READINESS_AND_DISCLOSURE_FREEZE"
POLICY_ID = "V5P2-D1-6ce8cc7c76b8d2cff367631077d65f20"
FINAL_ARCHITECTURE = (
    "Causal Depthwise-Separable Conv1D Temporal Encoder + Two-Layer GraphConv"
)
CHECKPOINT_SHA = "82314baedb4bf3969842abf9076679682369daae2e57db07d4b4aeaa8623f7cc"
MODEL_SOURCE_SHA = "ccdfcb74c1ddab7f20b98c922444873ad76c204ec4b5fd6827d52fb716f11fd9"

EXPECTED = {
    "f1": "0265de528df00fc84186688ad64621a1ac688a7aae65b2381f93fb8757c372e8",
    "d0": "671c6524810a765a0fc54116afd1fe3fd2e1599780310f3b379b6920c8708d81",
    "d1": "b63efbfe1272fb7f6ab922645182fba86e28d7c8f42a70e0f3ac453b88bec71f",
    "d2": "fcb78ae4059f49e4c78d4e3cf603160f7d8bad72857d6ef6b777cc4de0ab7055",
    "d3": "11339c6afcec5275a310219e05a360638121cdfabd2f02589273e2c84af56def",
    "d4": "0be3b06a31b30fabda6e7f6f5582be9cf4bfde105a4d01bf893dd4bcc0545a3b",
    "d5": "6fe42ee47b6b126b76203e94bb7c3cf3a655541506c9e1b32b0c4fa5207f774b",
    "legacy_decoder": "8da32b3ca3915365b8b01049a4ddc6a043583aa7193589ccf4af8116a4b0783c",
    "certified_decoder": "30309376cb0990a0f482045e2dac413d4370729cf94f9e556f895af2c5c8c71d",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: observed={observed}, expected={expected}"
        )
    return observed


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def assert_false(boundary: dict[str, Any], keys: tuple[str, ...], stage: str) -> None:
    for key in keys:
        if boundary.get(key) is True:
            raise RuntimeError(f"{stage} security boundary violated: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.expanduser().resolve()
    a = repo / "artifacts/v5"

    paths = {
        "f1": a / "p2_final_f1_post_inference_numerical_failure_disposition"
                 / "V5_P2_FINAL_F1_POST_INFERENCE_NUMERICAL_FAILURE_DISPOSITION.json",
        "d0": a / "p2_decoder_d0_numerical_root_cause_audit"
                 / "V5_P2_DECODER_D0_NUMERICAL_ROOT_CAUSE_AUDIT.json",
        "d1": a / "p2_decoder_d1_prospective_numerical_policy_freeze"
                 / "V5_P2_D1_PROSPECTIVE_NUMERICAL_CERTIFICATION_POLICY.json",
        "d2": a / "p2_decoder_d2_certified_implementation"
                 / "V5_P2_DECODER_D2_CERTIFIED_IMPLEMENTATION.json",
        "d3": a / "p2_decoder_d3_full_validation_certification_proof"
                 / "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF.json",
        "d4": a / "p2_evaluator_d4_staged_raw_a0_a1_validation"
                 / "V5_P2_EVALUATOR_D4_STAGED_RAW_A0_A1_VALIDATION.json",
        "d5": a / "p2_evaluator_d5_finalized_architecture_integration"
                 / "V5_P2_EVALUATOR_D5_FINALIZED_ARCHITECTURE_E1_INTEGRATION.json",
        "legacy_decoder": repo / "src/decoders/v5_legal_xy_exact_decoder.py",
        "certified_decoder": repo / "src/decoders/v5_legal_xy_exact_decoder_certified.py",
    }
    observed = {
        key: require_hash(path, EXPECTED[key], key)
        for key, path in paths.items()
    }

    f1, d0, d1, d2, d3, d4, d5 = (
        read_json(paths[key]) for key in ("f1", "d0", "d1", "d2", "d3", "d4", "d5")
    )

    if f1.get("classification") != "POST_INFERENCE_MID_A1_NUMERICAL_CERTIFICATION_ABORT":
        raise RuntimeError("F1 classification changed")

    # F1 stores authorization/result disposition under scientific_disposition.
    # The original D6 freezer incorrectly looked for these fields at top level.
    f1_scientific = f1.get("scientific_disposition")
    if not isinstance(f1_scientific, dict):
        raise RuntimeError("F1 scientific_disposition is missing")
    if f1_scientific.get("recovery_authorization_consumed") is not True:
        raise RuntimeError("F1 recovery authorization state changed")
    if f1_scientific.get("recovery_rerun_authorized") is not False:
        raise RuntimeError("F1 unexpectedly authorizes a rerun")
    if f1_scientific.get("official_complete_test_result_obtained") is not False:
        raise RuntimeError("F1 unexpectedly records a complete official result")

    if d0.get("root_cause", {}).get("classification") != (
        "OVERLY_BRITTLE_MIP_GAP_REPORTING_GATE_AT_MACHINE_EPSILON_SCALE"
    ):
        raise RuntimeError("D0 root-cause classification changed")

    if d1.get("status") != "FROZEN" or d1.get("policy_id") != POLICY_ID:
        raise RuntimeError("D1 policy changed")
    if d2.get("status") != "COMPLETE" or d2.get("policy_id") != POLICY_ID:
        raise RuntimeError("D2 binding changed")
    if d3.get("status") != "COMPLETE" or d3.get("policy_id") != POLICY_ID:
        raise RuntimeError("D3 binding changed")
    if d3.get("coverage", {}).get("all_outputs_certified") is not True:
        raise RuntimeError("D3 full validation certificate changed")
    if int(d3.get("coverage", {}).get("validation_items_certified", -1)) != 12528:
        raise RuntimeError("D3 validation coverage changed")

    if d4.get("status") != "COMPLETE":
        raise RuntimeError("D4 is incomplete")
    if d4.get("failure_isolation_proof", {}).get("status") != "PASS":
        raise RuntimeError("D4 failure-isolation proof changed")
    sci4 = d4.get("scientific_status", {})
    if sci4.get("Raw_A0_persist_before_A1") is not True:
        raise RuntimeError("D4 staged persistence changed")
    if sci4.get("A1_failure_cannot_invalidate_Raw_A0") is not True:
        raise RuntimeError("D4 failure isolation changed")

    if d5.get("status") != "COMPLETE":
        raise RuntimeError("D5 is incomplete")
    proof = d5.get("integration_proof", {})
    required = (
        "E1_all_arrays_bitwise_equal",
        "E1_six_logit_arrays_bitwise_equal",
        "E1_labels_and_order_bitwise_equal",
        "D4_Raw_outputs_bitwise_equal",
        "D4_A0_outputs_bitwise_equal",
        "D4_A1_outputs_bitwise_equal",
        "D4_Raw_metrics_exact",
        "D4_A0_metrics_exact",
        "D4_A1_metrics_exact",
        "first_batch_exact_repeat",
        "D3_transitive_binding_to_regenerated_E1",
    )
    for key in required:
        if proof.get(key) is not True:
            raise RuntimeError(f"D5 proof missing or false: {key}")

    system = d5.get("finalized_system", {})
    if system.get("architecture") != FINAL_ARCHITECTURE:
        raise RuntimeError("final architecture changed")
    if int(system.get("selected_seed", -1)) != 107:
        raise RuntimeError("selected seed changed")
    if int(system.get("selected_epoch", -1)) != 25:
        raise RuntimeError("selected epoch changed")
    if int(system.get("parameter_count", -1)) != 59785:
        raise RuntimeError("parameter count changed")
    if d5.get("source_hashes", {}).get("checkpoint") != CHECKPOINT_SHA:
        raise RuntimeError("checkpoint hash changed")
    if d5.get("source_hashes", {}).get("task_d_model") != MODEL_SOURCE_SHA:
        raise RuntimeError("model source hash changed")

    for stage_name, artifact in (
        ("D0", d0), ("D1", d1), ("D2", d2),
        ("D3", d3), ("D4", d4), ("D5", d5),
    ):
        assert_false(
            artifact.get("security_boundary", {}),
            (
                "test_path_formed",
                "test_directory_checked",
                "test_directory_enumerated",
                "test_tensors_deserialized",
                "evaluation_authorization_created",
                "authorization_created",
            ),
            stage_name,
        )

    disclosure = {
        "canonical_result_label": (
            "P2 Post-Hoc Test-Informed Numerical-Recovery Evaluation"
        ),
        "required_statements": [
            (
                "The original P2 blind one-shot evaluation did not produce a "
                "complete official Raw/A0/A1 test result."
            ),
            (
                "A single authorized recovery execution completed neural "
                "inference for 11,666/11,666 items and aborted during A1 exact "
                "decoding after at least 6,200 items because a machine-scale "
                "reported MIP-gap residue exceeded the original 64-epsilon gate."
            ),
            (
                "The numerical policy, certified decoder, and staged evaluator "
                "were repaired and validated using validation and synthetic data only."
            ),
            (
                "Any later P2 execution is test-informed and post hoc, not an "
                "untouched blind test or the original one-shot evaluation."
            ),
            (
                "Raw, A0, and A1 results must be reported separately, with "
                "immutable Raw/A0 persistence before A1 begins."
            ),
        ],
        "forbidden_claims": [
            "blind test result",
            "untouched test result",
            "original one-shot result",
            "prospective P2 test",
            "unseen test evaluation",
            "recovered blind result",
        ],
    }

    report = {
        "stage": STAGE,
        "status": "FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "R0_F1_schema_patch": {
            "enabled": True,
            "correct_container": "scientific_disposition",
            "scientific_values_changed": False,
        },
        "scientific_disposition": {
            "original_complete_blind_result_exists": False,
            "original_recovery_authorization_consumed": True,
            "P2_test_boundary_crossed": True,
            "future_P2_status": "post_hoc_test_informed_only",
            "official_complete_P2_test_result_exists": False,
        },
        "technical_readiness": {
            "finalized_architecture_bound": True,
            "finalized_checkpoint_bound": True,
            "validation_E1_bitwise_reproduced": True,
            "full_validation_decoder_certificate": True,
            "staged_Raw_A0_A1_failure_isolation": True,
            "D4_outputs_and_metrics_exactly_reproduced": True,
            "ready_for_separately_authorized_post_hoc_run": True,
        },
        "authorization": {
            "created_by_D6": False,
            "P2_post_hoc_execution_authorized_by_D6": False,
        },
        "mandatory_disclosure": disclosure,
        "recommendation": {
            "recommended_next_stage": (
                "V6_PROSPECTIVE_DATASET_AND_GENERALIZATION_PROTOCOL"
            ),
            "default_action": "CLOSE_P2_AND_PROCEED_TO_V6",
            "optional_exception": (
                "A single P2 post-hoc recovery run may be considered only after "
                "a separate explicit authorization freeze and only when a complete "
                "P2 Raw/A0/A1 diagnostic table is scientifically necessary."
            ),
        },
        "frozen_system": {
            "architecture": FINAL_ARCHITECTURE,
            "checkpoint_seed": 107,
            "checkpoint_epoch": 25,
            "checkpoint_sha256": CHECKPOINT_SHA,
            "model_source_sha256": MODEL_SOURCE_SHA,
            "parameter_count": 59785,
            "policy_id": POLICY_ID,
            "legacy_decoder_sha256": observed["legacy_decoder"],
            "certified_decoder_sha256": observed["certified_decoder"],
        },
        "evidence_chain": {
            key: {"path": str(paths[key]), "sha256": observed[key]}
            for key in ("f1", "d0", "d1", "d2", "d3", "d4", "d5")
        },
        "security_boundary": {
            "test_path_formed": False,
            "test_directory_checked": False,
            "test_directory_enumerated": False,
            "test_tensor_files_opened": False,
            "test_tensors_deserialized": False,
            "model_checkpoint_loaded": False,
            "inference_performed": False,
            "decoder_executed": False,
            "threshold_selection_performed": False,
            "authorization_created": False,
        },
    }

    artifact_dir = a / "p2_d6_post_hoc_recovery_readiness_disclosure_freeze"
    report_dir = repo / "reports/v5/p2_d6_post_hoc_recovery_readiness_disclosure_freeze"
    if artifact_dir.exists() or report_dir.exists():
        raise RuntimeError("D6 output already exists; do not overwrite it")
    artifact_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    artifact_json = artifact_dir / "V5_P2_D6_POST_HOC_RECOVERY_READINESS_DISCLOSURE.json"
    atomic_json(artifact_json, report)
    artifact_sha = sha256_file(artifact_json)

    disclosure_md = artifact_dir / "V5_P2_D6_MANDATORY_POST_HOC_DISCLOSURE.md"
    atomic_text(
        disclosure_md,
        """# Mandatory P2 Post-Hoc Disclosure

## Required result label

**P2 Post-Hoc Test-Informed Numerical-Recovery Evaluation**

## Required disclosure

The original P2 blind one-shot evaluation did not produce a complete official
Raw/A0/A1 test result. A single authorized recovery execution completed neural
inference for 11,666/11,666 items and then aborted during A1 exact decoding
after at least 6,200 items because a machine-scale reported MIP-gap residue
exceeded the original 64-epsilon reporting gate.

The numerical-certification policy, certified decoder, and staged evaluator
were subsequently repaired and validated using validation and synthetic data
only. Any later P2 execution is therefore test-informed and post hoc. It is not
an untouched blind test and is not the original one-shot evaluation.

Raw, A0, and A1 results must be reported separately. Raw and A0 artifacts must
be committed immutably before A1 begins.
""",
    )

    report_json = report_dir / "V5_P2_D6_POST_HOC_RECOVERY_READINESS_DISCLOSURE.json"
    report_copy = dict(report)
    report_copy["artifact"] = {
        "path": str(artifact_json),
        "sha256": artifact_sha,
    }
    atomic_json(report_json, report_copy)

    report_md = report_dir / "V5_P2_D6_POST_HOC_RECOVERY_READINESS_DISCLOSURE.md"
    atomic_text(
        report_md,
        f"""# V5 P2 D6 Post-Hoc Recovery Readiness and Disclosure Freeze

## Status

- Status: **FROZEN**
- Technical readiness for a separately authorized post-hoc run: **true**
- New authorization created: **false**
- P2 execution authorized by D6: **false**

## Scientific disposition

The original P2 blind one-shot evaluation did not yield a complete official
Raw/A0/A1 test result. The recovery authorization was consumed and the test
boundary was crossed. Any future P2 execution must be labeled:

> **P2 Post-Hoc Test-Informed Numerical-Recovery Evaluation**

It cannot be presented as an untouched blind result.

## Evidence closed before D6

- D0 identified the brittle machine-epsilon-scale reporting gate.
- D1 froze the replacement numerical-certification policy.
- D2 implemented the certified versioned decoder.
- D3 certified all 12,528 validation items and both MILPs per item.
- D4 proved Raw/A0 persistence before A1 and A1-failure isolation.
- D5 reproduced E1 bitwise from the finalized architecture and checkpoint and
  reproduced the D4 Raw/A0/A1 pipeline exactly.

## Recommendation

**Close P2 and proceed to the prospective V6 dataset/generalization protocol.**

A single P2 post-hoc run remains an optional exception requiring a separate,
explicit authorization freeze.

## Safety boundary

- Test path formed: **false**
- Test directory checked/enumerated: **false**
- Test tensors opened/deserialized: **false**
- Model/checkpoint loaded: **false**
- Inference performed: **false**
- Decoder executed: **false**
- Authorization created: **false**

Artifact SHA-256: `{artifact_sha}`
""",
    )

    lock = {
        "stage": STAGE + "_LOCK",
        "status": "LOCKED",
        "artifact_path": str(artifact_json),
        "artifact_sha256": artifact_sha,
        "disclosure_sha256": sha256_file(disclosure_md),
        "report_json_sha256": sha256_file(report_json),
        "report_markdown_sha256": sha256_file(report_md),
        "technical_readiness": True,
        "P2_execution_authorized": False,
        "authorization_created": False,
        "recommended_next_stage": (
            "V6_PROSPECTIVE_DATASET_AND_GENERALIZATION_PROTOCOL"
        ),
    }
    lock_path = report_dir / "V5_P2_D6_POST_HOC_RECOVERY_READINESS_DISCLOSURE_LOCK.json"
    atomic_json(lock_path, lock)

    marker = report_dir / "V5_P2_D6_POST_HOC_RECOVERY_READINESS_DISCLOSURE_FREEZE_COMPLETE"
    atomic_text(marker, marker.name + "\n")

    checksum_paths = [
        artifact_json, disclosure_md, report_json, report_md, lock_path, marker
    ]
    checksum_path = report_dir / "V5_P2_D6_POST_HOC_RECOVERY_READINESS_DISCLOSURE_SHA256SUMS.txt"
    atomic_text(
        checksum_path,
        "".join(f"{sha256_file(path)}  {path}\n" for path in checksum_paths),
    )

    print("V5_P2_D6_POST_HOC_RECOVERY_READINESS_DISCLOSURE_FREEZE_COMPLETE")
    print("status=FROZEN")
    print("D6_R0_F1_SCIENTIFIC_DISPOSITION_SCHEMA_PATCH=true")
    print("technical_readiness_for_post_hoc_recovery=true")
    print("post_hoc_P2_execution_authorized_by_D6=false")
    print("new_authorization_created=false")
    print("original_complete_blind_result_exists=false")
    print("P2_test_boundary_crossed=true")
    print("future_P2_status=post_hoc_test_informed_only")
    print("mandatory_result_label=P2_POST_HOC_TEST_INFORMED_NUMERICAL_RECOVERY_EVALUATION")
    print("finalized_architecture_bound=true")
    print("finalized_checkpoint_bound=true")
    print("validation_E1_bitwise_reproduced=true")
    print("full_validation_decoder_certificate=true")
    print("staged_Raw_A0_A1_failure_isolation=true")
    print("recommended_next_stage=V6_PROSPECTIVE_DATASET_AND_GENERALIZATION_PROTOCOL")
    print("test_path_formed=false")
    print("test_tensors_deserialized=false")
    print("model_checkpoint_loaded=false")
    print("inference_performed=false")
    print("decoder_executed=false")
    print("authorization_created=false")
    print(f"artifact={artifact_json}")
    print(f"artifact_sha256={artifact_sha}")
    print(f"mandatory_disclosure={disclosure_md}")
    print(f"report_markdown={report_md}")
    print(f"report_json={report_json}")
    print(f"sha256s={checksum_path}")


if __name__ == "__main__":
    main()
