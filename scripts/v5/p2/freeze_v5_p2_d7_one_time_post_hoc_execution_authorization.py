#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGE = "V5_P2_D7_ONE_TIME_POST_HOC_EXECUTION_AUTHORIZATION"
EXPECTED_EXECUTOR_SHA = "76c6d4869fc2d12ab38e77a8798d1d7194baf4c27658acacb4fc350aa36c69fb"
EXPECTED_D6_SHA = "b99b524c8bf172949f4f411281eed74e5aeb24493ba1a8b27e9de4909e945fd3"
EXPECTED_D5_SHA = "6fe42ee47b6b126b76203e94bb7c3cf3a655541506c9e1b32b0c4fa5207f774b"
EXPECTED_D4_SHA = "0be3b06a31b30fabda6e7f6f5582be9cf4bfde105a4d01bf893dd4bcc0545a3b"
EXPECTED_DECODER_SHA = "30309376cb0990a0f482045e2dac413d4370729cf94f9e556f895af2c5c8c71d"
MANDATORY_LABEL = "P2 Post-Hoc Test-Informed Numerical-Recovery Evaluation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: observed={observed}, expected={expected}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--executor-source", type=Path, required=True)
    parser.add_argument("--self-test-report", type=Path, required=True)
    parser.add_argument("--data-root-text", required=True)
    parser.add_argument("--output-dir-text", required=True)
    parser.add_argument("--authorization-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.expanduser().resolve()
    executor = args.executor_source.expanduser().resolve()
    self_test = args.self_test_report.expanduser().resolve()
    auth_dir = args.authorization_dir.expanduser().resolve()

    if auth_dir.exists():
        raise RuntimeError("D7 authorization directory already exists")
    if Path(os.path.abspath(os.path.expanduser(args.output_dir_text))).exists():
        raise RuntimeError("bound D8 output directory already exists")

    require_hash(executor, EXPECTED_EXECUTOR_SHA, "D8 executor")
    self_test_report = read_json(self_test)
    if self_test_report.get("status") != "PASS":
        raise RuntimeError("D8 source-only self-test did not pass")
    if self_test_report.get("evaluator_source_sha256") != EXPECTED_EXECUTOR_SHA:
        raise RuntimeError("self-test evaluator hash mismatch")
    for key in (
        "test_path_formed",
        "test_directory_checked",
        "test_tensors_deserialized",
        "authorization_created",
    ):
        if self_test_report.get(key) is not False:
            raise RuntimeError(f"self-test safety boundary changed: {key}")

    artifacts = root / "artifacts/v5"
    d6_path = artifacts / "p2_d6_post_hoc_recovery_readiness_disclosure_freeze" \
        / "V5_P2_D6_POST_HOC_RECOVERY_READINESS_DISCLOSURE.json"
    d5_path = artifacts / "p2_evaluator_d5_finalized_architecture_integration" \
        / "V5_P2_EVALUATOR_D5_FINALIZED_ARCHITECTURE_E1_INTEGRATION.json"
    d4_path = artifacts / "p2_evaluator_d4_staged_raw_a0_a1_validation" \
        / "V5_P2_EVALUATOR_D4_STAGED_RAW_A0_A1_VALIDATION.json"
    decoder_path = root / "src/decoders/v5_legal_xy_exact_decoder_certified.py"
    require_hash(d6_path, EXPECTED_D6_SHA, "D6")
    require_hash(d5_path, EXPECTED_D5_SHA, "D5")
    require_hash(d4_path, EXPECTED_D4_SHA, "D4")
    require_hash(decoder_path, EXPECTED_DECODER_SHA, "certified decoder")

    d6 = read_json(d6_path)
    if d6.get("status") != "FROZEN":
        raise RuntimeError("D6 is not frozen")
    if d6.get("technical_readiness", {}).get(
        "ready_for_separately_authorized_post_hoc_run"
    ) is not True:
        raise RuntimeError("D6 technical readiness changed")
    if d6.get("authorization", {}).get(
        "P2_post_hoc_execution_authorized_by_D6"
    ) is not False:
        raise RuntimeError("D6 authorization disposition changed")
    if d6.get("scientific_disposition", {}).get(
        "future_P2_status"
    ) != "post_hoc_test_informed_only":
        raise RuntimeError("D6 reporting disposition changed")

    data_root_text = os.path.abspath(os.path.expanduser(args.data_root_text))
    output_dir_text = os.path.abspath(os.path.expanduser(args.output_dir_text))
    created = datetime.now(timezone.utc).isoformat()
    nonce = secrets.token_hex(32)

    base = {
        "stage": STAGE,
        "status": "AUTHORIZED",
        "created_utc": created,
        "authorization_basis": (
            "User explicitly requested the one-time P2 post-hoc evaluation "
            "in this conversation on 2026-08-02T02:24:00+05:30."
        ),
        "mandatory_result_label": MANDATORY_LABEL,
        "scientific_classification": "POST_HOC_TEST_INFORMED_ONLY",
        "single_use": True,
        "evaluation_count": 1,
        "second_test_inference_authorized": False,
        "A1_only_continuation_after_committed_E1_allowed": True,
        "data_root_text_sha256": sha256_text(data_root_text),
        "output_dir_text_sha256": sha256_text(output_dir_text),
        "evaluator_source_sha256": EXPECTED_EXECUTOR_SHA,
        "self_test_report_sha256": sha256_file(self_test),
        "frozen_bindings": {
            "D6_sha256": EXPECTED_D6_SHA,
            "D5_sha256": EXPECTED_D5_SHA,
            "D4_sha256": EXPECTED_D4_SHA,
            "certified_decoder_sha256": EXPECTED_DECODER_SHA,
            "checkpoint_seed": 107,
            "checkpoint_epoch": 25,
            "A0_thresholds_frozen": True,
            "A1_margin_threshold": 8.7205320882398425,
        },
        "execution_contract": {
            "consume_token_before_first_test_path_operation": True,
            "single_neural_inference_pass": True,
            "commit_E1_before_Raw_A0_A1": True,
            "commit_Raw_and_A0_before_A1": True,
            "A1_uses_certified_decoder": True,
            "A1_atomic_chunk_resume_from_E1_only": True,
            "no_training_or_fine_tuning": True,
            "no_threshold_or_checkpoint_changes": True,
            "no_test_driven_changes": True,
        },
    }
    authorization_id = "V5P2-POSTHOC-" + hashlib.sha256(
        (sha256_json(base) + nonce).encode()
    ).hexdigest()[:32]
    base["authorization_id"] = authorization_id

    token = {
        "stage": STAGE + "_TOKEN",
        "authorization_id": authorization_id,
        "created_utc": created,
        "single_use": True,
        "evaluation_count": 1,
        "evaluator_source_sha256": EXPECTED_EXECUTOR_SHA,
        "data_root_text_sha256": sha256_text(data_root_text),
        "output_dir_text_sha256": sha256_text(output_dir_text),
        "nonce": nonce,
    }

    auth_dir.mkdir(parents=True)
    token_path = auth_dir / "V5_P2_D7_SINGLE_USE_AUTHORIZATION_TOKEN.json"
    atomic_json(token_path, token)
    base["authorization_token_sha256"] = sha256_file(token_path)
    contract_path = auth_dir / "V5_P2_D7_POST_HOC_EXECUTION_AUTHORIZATION.json"
    atomic_json(contract_path, base)

    lock = {
        "stage": STAGE + "_LOCK",
        "status": "LOCKED",
        "authorization_id": authorization_id,
        "contract_sha256": sha256_file(contract_path),
        "token_sha256": sha256_file(token_path),
        "evaluator_source_sha256": EXPECTED_EXECUTOR_SHA,
        "single_use": True,
        "evaluation_count": 1,
        "test_path_formed": False,
        "test_directory_checked": False,
        "test_tensors_deserialized": False,
        "authorization_created": True,
    }
    lock_path = auth_dir / "V5_P2_D7_POST_HOC_EXECUTION_AUTHORIZATION_LOCK.json"
    atomic_json(lock_path, lock)
    marker = auth_dir / "V5_P2_D7_POST_HOC_EXECUTION_AUTHORIZATION_COMPLETE"
    atomic_text(marker, marker.name + "\n")
    sums = auth_dir / "V5_P2_D7_POST_HOC_EXECUTION_AUTHORIZATION_SHA256SUMS.txt"
    atomic_text(
        sums,
        "".join(
            f"{sha256_file(path)}  {path}\n"
            for path in (contract_path, token_path, lock_path, marker, self_test)
        ),
    )

    print("V5_P2_D7_ONE_TIME_POST_HOC_EXECUTION_AUTHORIZATION_COMPLETE")
    print("status=AUTHORIZED")
    print(f"authorization_id={authorization_id}")
    print("single_use=true")
    print("evaluation_count=1")
    print("second_test_inference_authorized=false")
    print("A1_only_continuation_allowed=true")
    print("mandatory_result_label=P2_POST_HOC_TEST_INFORMED_NUMERICAL_RECOVERY_EVALUATION")
    print("test_path_formed=false")
    print("test_directory_checked=false")
    print("test_tensors_deserialized=false")
    print("authorization_created=true")
    print(f"contract={contract_path}")
    print(f"contract_sha256={sha256_file(contract_path)}")
    print(f"token={token_path}")
    print(f"token_sha256={sha256_file(token_path)}")


if __name__ == "__main__":
    main()
