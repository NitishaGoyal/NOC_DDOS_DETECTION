#!/usr/bin/env python3
"""Create an immutable, self-contained preservation bundle for canonical V5-P2.

This bundle is created before retrospective R0/R1 experiments so the canonical
neural model, frozen checkpoint, legal-route decoder, thresholds, topology, and
final P2 evidence cannot be mixed with later experimental variants.

The script:
- verifies every critical canonical hash;
- copies only frozen sources/artifacts into a dedicated preservation tree;
- computes a complete manifest and tree digest;
- creates a self-contained ZIP;
- marks copied files read-only;
- never opens train/validation/test tensors;
- never runs inference, training, threshold tuning, or decoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGE = "V5_P2_CANONICAL_NEURAL_LEGAL_ROUTER_PRESERVATION"

EXPECTED = {
    "model_source": "ccdfcb74c1ddab7f20b98c922444873ad76c204ec4b5fd6827d52fb716f11fd9",
    "b3_source": "56ee3207d039b60e8e3a898a689cd8e361247c86b7450a3a5e95f423ebe30def",
    "checkpoint": "82314baedb4bf3969842abf9076679682369daae2e57db07d4b4aeaa8623f7cc",
    "edge_index": "f6b8050bc158de509b0ff1c5d1d7cb1ffe32c08b0f2287398270b1b891b57aff",
    "physical_mask": "a93f81a9ce4315d0eef911f9b9c45529ceb9b2da726dd578a247b46d260e6a3a",
    "route_library": "3ae860817ddef7cdbc0360057c51dc86908337b40b1df4c7df8fc2c8b82ea0d7",
    "certified_decoder": "30309376cb0990a0f482045e2dac413d4370729cf94f9e556f895af2c5c8c71d",
    "threshold_contract": "b9acb9dac24cc1d0fe2cdae32525a4c0bdddc40e1607824599ad5bc3a446d957",
    "d1": "b63efbfe1272fb7f6ab922645182fba86e28d7c8f42a70e0f3ac453b88bec71f",
    "d2": "fcb78ae4059f49e4c78d4e3cf603160f7d8bad72857d6ef6b777cc4de0ab7055",
    "d3": "11339c6afcec5275a310219e05a360638121cdfabd2f02589273e2c84af56def",
    "d4": "0be3b06a31b30fabda6e7f6f5582be9cf4bfde105a4d01bf893dd4bcc0545a3b",
    "d5": "6fe42ee47b6b126b76203e94bb7c3cf3a655541506c9e1b32b0c4fa5207f774b",
    "d6": "b99b524c8bf172949f4f411281eed74e5aeb24493ba1a8b27e9de4909e945fd3",
    "d6_disclosure": "6033f28f6d55e93d1df0bcce200bdfc6a49a6313156a25d41d568a23284982ba",
    "d8": "6f07f6e81505fb7c15738e34026a20f1e4df3cfff25723c75fbf269a57167ddc",
}

ARCHITECTURE = (
    "Causal Depthwise-Separable Conv1D Temporal Encoder + Two-Layer GraphConv"
)
CHECKPOINT_SEED = 107
CHECKPOINT_EPOCH = 25
PARAMETERS = 59785
POLICY_ID = "V5P2-D1-6ce8cc7c76b8d2cff367631077d65f20"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def safe_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o444)


def git_capture(root: Path, args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return proc.stdout
    except Exception as exc:
        return f"git capture failed: {type(exc).__name__}: {exc}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.expanduser().resolve()
    reports = repo / "reports/v5"
    artifacts = repo / "artifacts/v5"

    sources = {
        "model_source": repo / "src/models/v5_p2_task_d_full_multitask_count4.py",
        "b3_source": repo / "src/models/v5_p2_b3_conv1d_only_count4.py",
        "checkpoint": reports / "p2_task_d_checkpoint_selection_freeze"
            / "selected_graphconv_checkpoint.pt",
        "checkpoint_freeze_json": reports / "p2_task_d_checkpoint_selection_freeze"
            / "V5_P2_TASK_D_CHECKPOINT_SELECTION_FREEZE.json",
        "checkpoint_freeze_md": reports / "p2_task_d_checkpoint_selection_freeze"
            / "V5_P2_TASK_D_CHECKPOINT_SELECTION_FREEZE.md",
        "edge_index": reports / "p2_g1a_r2a_canonical_static_topology_contract"
            / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
        "physical_mask": reports / "p2_a2_r2_feature_normalization_mask_contract"
            / "V5_P2_A2_R2_TOPOLOGY_DERIVED_BOOLEAN_PORT_MASK.pt",
        "route_library": repo / "src/decoders/v5_xy_route_library.py",
        "certified_decoder": repo / "src/decoders/v5_legal_xy_exact_decoder_certified.py",
        "threshold_contract": artifacts / "p2_e2_raw_threshold_freeze"
            / "V5_P2_E2_FROZEN_RAW_THRESHOLDS.json",
        "d1": artifacts / "p2_decoder_d1_prospective_numerical_policy_freeze"
            / "V5_P2_D1_PROSPECTIVE_NUMERICAL_CERTIFICATION_POLICY.json",
        "d2": artifacts / "p2_decoder_d2_certified_implementation"
            / "V5_P2_DECODER_D2_CERTIFIED_IMPLEMENTATION.json",
        "d3": artifacts / "p2_decoder_d3_full_validation_certification_proof"
            / "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF.json",
        "d4": artifacts / "p2_evaluator_d4_staged_raw_a0_a1_validation"
            / "V5_P2_EVALUATOR_D4_STAGED_RAW_A0_A1_VALIDATION.json",
        "d5": artifacts / "p2_evaluator_d5_finalized_architecture_integration"
            / "V5_P2_EVALUATOR_D5_FINALIZED_ARCHITECTURE_E1_INTEGRATION.json",
        "d6": artifacts / "p2_d6_post_hoc_recovery_readiness_disclosure_freeze"
            / "V5_P2_D6_POST_HOC_RECOVERY_READINESS_DISCLOSURE.json",
        "d6_disclosure": artifacts / "p2_d6_post_hoc_recovery_readiness_disclosure_freeze"
            / "V5_P2_D6_MANDATORY_POST_HOC_DISCLOSURE.md",
        "d8": artifacts / "p2_post_hoc_test_informed_numerical_recovery_evaluation"
            / "V5_P2_D8_POST_HOC_TEST_INFORMED_NUMERICAL_RECOVERY_EVALUATION.json",
        "d8_report": reports / "p2_post_hoc_test_informed_numerical_recovery_evaluation"
            / "V5_P2_D8_POST_HOC_TEST_INFORMED_NUMERICAL_RECOVERY_EVALUATION.md",
        "d8_lock": reports / "p2_post_hoc_test_informed_numerical_recovery_evaluation"
            / "V5_P2_D8_POST_HOC_TEST_INFORMED_NUMERICAL_RECOVERY_EVALUATION_LOCK.json",
        "d7_contract": reports / "p2_d7_one_time_post_hoc_execution_authorization"
            / "V5_P2_D7_POST_HOC_EXECUTION_AUTHORIZATION.json",
        "d7_consumed_token": reports / "p2_d7_one_time_post_hoc_execution_authorization"
            / "V5_P2_D7_SINGLE_USE_AUTHORIZATION_TOKEN.json.consumed",
    }

    critical = {
        key: require_hash(sources[key], EXPECTED[key], key)
        for key in EXPECTED
    }

    for key in (
        "checkpoint_freeze_json",
        "checkpoint_freeze_md",
        "d8_report",
        "d8_lock",
        "d7_contract",
        "d7_consumed_token",
    ):
        if not sources[key].is_file():
            raise FileNotFoundError(f"{key} missing: {sources[key]}")

    d1 = read_json(sources["d1"])
    d3 = read_json(sources["d3"])
    d5 = read_json(sources["d5"])
    d6 = read_json(sources["d6"])
    d8 = read_json(sources["d8"])

    if d1.get("status") != "FROZEN" or d1.get("policy_id") != POLICY_ID:
        raise RuntimeError("D1 policy binding changed")
    if d3.get("status") != "COMPLETE":
        raise RuntimeError("D3 is incomplete")
    if d5.get("status") != "COMPLETE":
        raise RuntimeError("D5 is incomplete")
    if d6.get("status") != "FROZEN":
        raise RuntimeError("D6 is not frozen")
    if d8.get("status") != "COMPLETE":
        raise RuntimeError("D8 is incomplete")
    if d8.get("result_label") != (
        "P2 Post-Hoc Test-Informed Numerical-Recovery Evaluation"
    ):
        raise RuntimeError("D8 scientific label changed")

    preservation = reports / "p2_canonical_preservation"
    zip_path = reports / "V5_P2_CANONICAL_NEURAL_LEGAL_ROUTER_PRESERVATION.zip"
    if preservation.exists() or zip_path.exists():
        raise RuntimeError(
            "canonical preservation output already exists; do not overwrite it"
        )
    preservation.mkdir(parents=True)

    destinations = {
        "model_source": "model/v5_p2_task_d_full_multitask_count4.py",
        "b3_source": "model/v5_p2_b3_conv1d_only_count4.py",
        "checkpoint": "model/selected_graphconv_checkpoint.pt",
        "checkpoint_freeze_json": "model/V5_P2_TASK_D_CHECKPOINT_SELECTION_FREEZE.json",
        "checkpoint_freeze_md": "model/V5_P2_TASK_D_CHECKPOINT_SELECTION_FREEZE.md",
        "edge_index": "topology/V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
        "physical_mask": "topology/V5_P2_A2_R2_TOPOLOGY_DERIVED_BOOLEAN_PORT_MASK.pt",
        "route_library": "decoder/v5_xy_route_library.py",
        "certified_decoder": "decoder/v5_legal_xy_exact_decoder_certified.py",
        "threshold_contract": "thresholds/V5_P2_E2_FROZEN_RAW_THRESHOLDS.json",
        "d1": "decoder/V5_P2_D1_PROSPECTIVE_NUMERICAL_CERTIFICATION_POLICY.json",
        "d2": "decoder/V5_P2_DECODER_D2_CERTIFIED_IMPLEMENTATION.json",
        "d3": "decoder/V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF.json",
        "d4": "evaluation/V5_P2_EVALUATOR_D4_STAGED_RAW_A0_A1_VALIDATION.json",
        "d5": "evaluation/V5_P2_EVALUATOR_D5_FINALIZED_ARCHITECTURE_E1_INTEGRATION.json",
        "d6": "evaluation/V5_P2_D6_POST_HOC_RECOVERY_READINESS_DISCLOSURE.json",
        "d6_disclosure": "evaluation/V5_P2_D6_MANDATORY_POST_HOC_DISCLOSURE.md",
        "d7_contract": "evaluation/V5_P2_D7_POST_HOC_EXECUTION_AUTHORIZATION.json",
        "d7_consumed_token": "evaluation/V5_P2_D7_SINGLE_USE_AUTHORIZATION_TOKEN.json.consumed",
        "d8": "evaluation/V5_P2_D8_POST_HOC_TEST_INFORMED_NUMERICAL_RECOVERY_EVALUATION.json",
        "d8_report": "evaluation/V5_P2_D8_POST_HOC_TEST_INFORMED_NUMERICAL_RECOVERY_EVALUATION.md",
        "d8_lock": "evaluation/V5_P2_D8_POST_HOC_TEST_INFORMED_NUMERICAL_RECOVERY_EVALUATION_LOCK.json",
    }

    for key, relative in destinations.items():
        safe_copy(sources[key], preservation / relative)

    provenance = preservation / "provenance"
    provenance.mkdir()
    atomic_text(provenance / "git_status.txt", git_capture(repo, ["status", "--short", "--branch"]))
    atomic_text(provenance / "git_head.txt", git_capture(repo, ["rev-parse", "HEAD"]))
    atomic_text(provenance / "git_log_10.txt", git_capture(repo, ["log", "--oneline", "--decorate", "-10"]))
    atomic_json(
        provenance / "environment.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "repo_root": str(repo),
            "test_tensor_access": False,
            "training_performed": False,
            "inference_performed": False,
            "decoder_execution_performed": False,
        },
    )

    system_manifest = {
        "stage": STAGE,
        "status": "PRESERVED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Immutable canonical reference before V5-P2 retrospective R0/R1 "
            "model-improvement experiments."
        ),
        "canonical_system": {
            "architecture": ARCHITECTURE,
            "checkpoint_seed": CHECKPOINT_SEED,
            "checkpoint_epoch": CHECKPOINT_EPOCH,
            "parameter_count": PARAMETERS,
            "policy_id": POLICY_ID,
            "model_source_sha256": EXPECTED["model_source"],
            "checkpoint_sha256": EXPECTED["checkpoint"],
            "route_library_sha256": EXPECTED["route_library"],
            "certified_decoder_sha256": EXPECTED["certified_decoder"],
            "D8_result_sha256": EXPECTED["d8"],
        },
        "scientific_boundary": {
            "P2_test_result_status": "post_hoc_test_informed_complete",
            "P2_test_must_not_be_reused_for_model_selection": True,
            "later_R0_R1_outputs_are_separate_experiments": True,
            "canonical_files_must_not_be_edited": True,
        },
        "critical_source_hashes": critical,
        "copied_files": destinations,
    }
    atomic_json(preservation / "PRESERVATION_MANIFEST.json", system_manifest)

    readme = f"""# V5-P2 Canonical Neural + Legal-Router Preservation

This directory is the immutable reference system preserved before retrospective
R0/R1 model-improvement work.

## Canonical neural model

- Architecture: **{ARCHITECTURE}**
- Parameters: **{PARAMETERS:,}**
- Checkpoint: **seed {CHECKPOINT_SEED}, epoch {CHECKPOINT_EPOCH}**
- Checkpoint SHA-256: `{EXPECTED['checkpoint']}`

## Canonical legal-route system

- Certified decoder policy: `{POLICY_ID}`
- Route library SHA-256: `{EXPECTED['route_library']}`
- Certified decoder SHA-256: `{EXPECTED['certified_decoder']}`

## Final P2 evidence

The preserved D8 result is the completed:

> **P2 Post-Hoc Test-Informed Numerical-Recovery Evaluation**

D8 artifact SHA-256:

`{EXPECTED['d8']}`

The P2 test result is preserved for diagnosis and reporting only. It must not be
used to select R0/R1 or later model variants.

## Retrospective boundary

All new models, checkpoints, logs, reports, and metrics must be written outside
this directory under dedicated retrospective paths. No file in this
preservation directory may be edited or replaced.
"""
    atomic_text(preservation / "README.md", readme)

    file_hashes = {
        str(path.relative_to(preservation)): sha256_file(path)
        for path in sorted(preservation.rglob("*"))
        if path.is_file()
    }
    tree_sha = sha256_json(file_hashes)
    atomic_json(
        preservation / "PRESERVATION_TREE.json",
        {
            "stage": STAGE,
            "file_count": len(file_hashes),
            "tree_sha256": tree_sha,
            "files": file_hashes,
        },
    )

    final_hashes = {
        str(path.relative_to(preservation)): sha256_file(path)
        for path in sorted(preservation.rglob("*"))
        if path.is_file()
    }
    atomic_text(
        preservation / "MANIFEST_SHA256SUMS.txt",
        "".join(
            f"{digest}  {relative}\n"
            for relative, digest in sorted(final_hashes.items())
        ),
    )
    atomic_text(
        preservation / "PRESERVATION_COMPLETE",
        f"{STAGE}_COMPLETE\n"
        f"tree_sha256={tree_sha}\n"
    )

    for path in preservation.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(
        [p for p in preservation.rglob("*") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        path.chmod(0o555)
    preservation.chmod(0o555)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(preservation.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    arcname=f"p2_canonical_preservation/{path.relative_to(preservation)}",
                )
    zip_path.chmod(0o444)

    print("V5_P2_CANONICAL_NEURAL_LEGAL_ROUTER_PRESERVATION_COMPLETE")
    print("status=PRESERVED")
    print(f"architecture={ARCHITECTURE}")
    print(f"parameter_count={PARAMETERS}")
    print(f"checkpoint_seed={CHECKPOINT_SEED}")
    print(f"checkpoint_epoch={CHECKPOINT_EPOCH}")
    print(f"checkpoint_sha256={EXPECTED['checkpoint']}")
    print(f"certified_decoder_sha256={EXPECTED['certified_decoder']}")
    print(f"route_library_sha256={EXPECTED['route_library']}")
    print(f"D8_result_sha256={EXPECTED['d8']}")
    print(f"preservation_tree_sha256={tree_sha}")
    print("test_tensor_access=false")
    print("training_performed=false")
    print("inference_performed=false")
    print("decoder_execution_performed=false")
    print(f"preservation_directory={preservation}")
    print(f"preservation_zip={zip_path}")
    print(f"preservation_zip_sha256={sha256_file(zip_path)}")


if __name__ == "__main__":
    main()
