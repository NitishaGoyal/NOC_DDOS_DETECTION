#!/usr/bin/env python3
"""D8: one-time P2 post-hoc test-informed numerical-recovery evaluation.

Initial mode:
  - verifies the D7 contract/token and all frozen non-test dependencies;
  - consumes the single-use token before the first test-path operation;
  - performs exactly one neural inference pass;
  - commits E1 neural logits/labels;
  - commits E2 Raw and E3 A0 before A1 begins;
  - decodes A1 in resumable, atomic chunks using the D2 certified decoder;
  - commits E4 A1 and the final disclosed report.

Resume mode:
  - never forms the test path;
  - never loads the model/checkpoint;
  - reads only the already committed E1 archive;
  - resumes A1 chunks within the same consumed authorization.

This is not a blind evaluation. The mandatory label is:
P2 Post-Hoc Test-Informed Numerical-Recovery Evaluation.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import secrets
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

STAGE = "V5_P2_D8_POST_HOC_TEST_INFORMED_NUMERICAL_RECOVERY_EVALUATION"
COMPLETE = STAGE + "_COMPLETE"
ACCESS_STARTED = STAGE + "_TEST_ACCESS_STARTED"
FAILURE = STAGE + "_FAILURE"
A1_CHUNK_SIZE = 100
EXPECTED_TEST_ITEMS = 11666
EXPECTED_TEST_PAIRS = 69
EXPECTED_TEST_BATCHES = 92
POLICY_ID = "V5P2-D1-6ce8cc7c76b8d2cff367631077d65f20"
MANDATORY_LABEL = "P2 Post-Hoc Test-Informed Numerical-Recovery Evaluation"

EXPECTED_HASHES = {
    "helper_evaluator": "770ad8937427c32680c1cced7b3040bf6055662615c88edbf544e8805d9fe7c9",
    "l5_contract": "a9cd7b7d437e7e362b1bb942796df9773a603c90ab95e4b6647b9049114fc078",
    "checkpoint": "82314baedb4bf3969842abf9076679682369daae2e57db07d4b4aeaa8623f7cc",
    "task_d_model": "ccdfcb74c1ddab7f20b98c922444873ad76c204ec4b5fd6827d52fb716f11fd9",
    "b3_model": "56ee3207d039b60e8e3a898a689cd8e361247c86b7450a3a5e95f423ebe30def",
    "edge_index": "f6b8050bc158de509b0ff1c5d1d7cb1ffe32c08b0f2287398270b1b891b57aff",
    "physical_mask": "a93f81a9ce4315d0eef911f9b9c45529ceb9b2da726dd578a247b46d260e6a3a",
    "manifest": "f42f40446d03161a6932ee060f6d8c859f894cb5075d1fc926ea161461413ab5",
    "route_library": "3ae860817ddef7cdbc0360057c51dc86908337b40b1df4c7df8fc2c8b82ea0d7",
    "certified_decoder": "30309376cb0990a0f482045e2dac413d4370729cf94f9e556f895af2c5c8c71d",
    "threshold_contract": "b9acb9dac24cc1d0fe2cdae32525a4c0bdddc40e1607824599ad5bc3a446d957",
    "f1": "0265de528df00fc84186688ad64621a1ac688a7aae65b2381f93fb8757c372e8",
    "d0": "671c6524810a765a0fc54116afd1fe3fd2e1599780310f3b379b6920c8708d81",
    "d1": "b63efbfe1272fb7f6ab922645182fba86e28d7c8f42a70e0f3ac453b88bec71f",
    "d2": "fcb78ae4059f49e4c78d4e3cf603160f7d8bad72857d6ef6b777cc4de0ab7055",
    "d3": "11339c6afcec5275a310219e05a360638121cdfabd2f02589273e2c84af56def",
    "d4": "0be3b06a31b30fabda6e7f6f5582be9cf4bfde105a4d01bf893dd4bcc0545a3b",
    "d5": "6fe42ee47b6b126b76203e94bb7c3cf3a655541506c9e1b32b0c4fa5207f774b",
    "d6": "b99b524c8bf172949f4f411281eed74e5aeb24493ba1a8b27e9de4909e945fd3",
    "d6_disclosure": "6033f28f6d55e93d1df0bcce200bdfc6a49a6313156a25d41d568a23284982ba",
}

A0_THRESHOLDS = {
    "graph": 0.47174675035328983,
    "source": 0.94960549299285935,
    "transit": 0.85703332488359107,
    "victim": 0.82601148026998117,
    "path": 0.83393474660904676,
}
A1_MARGIN_THRESHOLD = 8.7205320882398425
ROLE_NAMES = ("source", "transit", "victim", "path")


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
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n")


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
    os.replace(tmp, path)


def import_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def repo_paths(root: Path) -> dict[str, Path]:
    reports = root / "reports/v5"
    artifacts = root / "artifacts/v5"
    return {
        "helper_evaluator": root / "scripts/v5/p2/evaluate_v5_p2_recovery_raw_a0_a1.py",
        "l5_contract": reports / "p2_l5_final_pretest_freeze"
            / "V5_P2_ONE_SHOT_BLIND_EVALUATION_CONTRACT.json",
        "checkpoint": reports / "p2_task_d_checkpoint_selection_freeze"
            / "selected_graphconv_checkpoint.pt",
        "task_d_model": root / "src/models/v5_p2_task_d_full_multitask_count4.py",
        "b3_model": root / "src/models/v5_p2_b3_conv1d_only_count4.py",
        "edge_index": reports / "p2_g1a_r2a_canonical_static_topology_contract"
            / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
        "physical_mask": reports / "p2_a2_r2_feature_normalization_mask_contract"
            / "V5_P2_A2_R2_TOPOLOGY_DERIVED_BOOLEAN_PORT_MASK.pt",
        "manifest": reports / "p2_a1_r2_pair_aligned_window_contract"
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv",
        "route_library": root / "src/decoders/v5_xy_route_library.py",
        "certified_decoder": root / "src/decoders/v5_legal_xy_exact_decoder_certified.py",
        "threshold_contract": artifacts / "p2_e2_raw_threshold_freeze"
            / "V5_P2_E2_FROZEN_RAW_THRESHOLDS.json",
        "f1": artifacts / "p2_final_f1_post_inference_numerical_failure_disposition"
            / "V5_P2_FINAL_F1_POST_INFERENCE_NUMERICAL_FAILURE_DISPOSITION.json",
        "d0": artifacts / "p2_decoder_d0_numerical_root_cause_audit"
            / "V5_P2_DECODER_D0_NUMERICAL_ROOT_CAUSE_AUDIT.json",
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
    }


def verify_non_test_chain(root: Path) -> tuple[dict[str, Path], dict[str, str]]:
    paths = repo_paths(root)
    observed = {
        key: require_hash(paths[key], expected, key)
        for key, expected in EXPECTED_HASHES.items()
    }
    d1 = read_json(paths["d1"])
    d2 = read_json(paths["d2"])
    d3 = read_json(paths["d3"])
    d4 = read_json(paths["d4"])
    d5 = read_json(paths["d5"])
    d6 = read_json(paths["d6"])
    if d1.get("status") != "FROZEN" or d1.get("policy_id") != POLICY_ID:
        raise RuntimeError("D1 policy changed")
    if d2.get("status") != "COMPLETE" or d2.get("policy_id") != POLICY_ID:
        raise RuntimeError("D2 binding changed")
    if d3.get("status") != "COMPLETE" or d3.get("policy_id") != POLICY_ID:
        raise RuntimeError("D3 binding changed")
    if d4.get("status") != "COMPLETE":
        raise RuntimeError("D4 incomplete")
    if d5.get("status") != "COMPLETE":
        raise RuntimeError("D5 incomplete")
    if d6.get("status") != "FROZEN":
        raise RuntimeError("D6 not frozen")
    if d6.get("authorization", {}).get(
        "P2_post_hoc_execution_authorized_by_D6"
    ) is not False:
        raise RuntimeError("D6 authorization disposition changed")
    if d6.get("scientific_disposition", {}).get(
        "future_P2_status"
    ) != "post_hoc_test_informed_only":
        raise RuntimeError("D6 future-P2 disposition changed")
    if d6.get("technical_readiness", {}).get(
        "ready_for_separately_authorized_post_hoc_run"
    ) is not True:
        raise RuntimeError("D6 technical readiness changed")
    return paths, observed


def self_test(root: Path, output: Path) -> None:
    paths, observed = verify_non_test_chain(root)
    helper = import_source(paths["helper_evaluator"], "v5_p2_d8_helper_selftest")
    route = import_source(paths["route_library"], "src.decoders.v5_xy_route_library")
    certified = import_source(
        paths["certified_decoder"],
        "src.decoders.v5_legal_xy_exact_decoder_certified",
    )

    torch.manual_seed(0)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    model_paths = {
        "b3_model": paths["b3_model"],
        "task_d_model": paths["task_d_model"],
        "edge_index": paths["edge_index"],
        "checkpoint": paths["checkpoint"],
    }
    model, summary = helper.build_frozen_model(paths=model_paths, device=device)
    mask = helper.load_physical_mask(paths["physical_mask"])
    x = torch.zeros((2, 16, 58, 32), dtype=torch.float32)
    m = mask.unsqueeze(0).expand(2, -1, -1).clone()
    with torch.inference_mode():
        first = model(x, m)
        second = model(x, m)
    for key in helper.REQUIRED_OUTPUT_KEYS:
        if not torch.equal(first[key], second[key]):
            raise RuntimeError(f"self-test nondeterminism: {key}")

    h = certified.decode_best_attack_hypothesis(
        graph_logit=float(first["attack_logits"][0]),
        count_logits=first["count_logits"][0].cpu().numpy().astype(np.float64),
        source_logits=first["source_logits"][0].cpu().numpy().astype(np.float64),
        transit_logits=first["transit_logits"][0].cpu().numpy().astype(np.float64),
        victim_logits=first["victim_logits"][0].cpu().numpy().astype(np.float64),
        path_logits=first["path_logits"][0].cpu().numpy().astype(np.float64),
    )
    if h.certificate_status != "CERTIFIED":
        raise RuntimeError("certified decoder self-test failed")
    decoded = certified.apply_margin_threshold(h, A1_MARGIN_THRESHOLD)
    if decoded.attack:
        combined = route.combine_route_ids(
            decoded.route_ids, expected_k=decoded.attacker_count
        )
        if combined.source_mask != decoded.source_mask:
            raise RuntimeError("route/source mask self-test failed")

    result = {
        "stage": STAGE + "_SOURCE_ONLY_SELF_TEST",
        "status": "PASS",
        "evaluator_source_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_source_hashes": observed,
        "model_summary": summary,
        "certified_decoder_policy_id": h.certificate_policy_id,
        "certified_decoder_status": h.certificate_status,
        "test_path_formed": False,
        "test_directory_checked": False,
        "test_tensors_deserialized": False,
        "authorization_created": False,
    }
    atomic_json(output, result)
    print("V5_P2_D8_SOURCE_ONLY_SELF_TEST_PASS")
    print(f"self_test_report={output}")
    print(f"self_test_report_sha256={sha256_file(output)}")


def validate_authorization(
    *,
    auth_dir: Path,
    evaluator_sha: str,
    data_root_text: str,
    output_dir_text: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    contract_path = auth_dir / "V5_P2_D7_POST_HOC_EXECUTION_AUTHORIZATION.json"
    token_path = auth_dir / "V5_P2_D7_SINGLE_USE_AUTHORIZATION_TOKEN.json"
    lock_path = auth_dir / "V5_P2_D7_POST_HOC_EXECUTION_AUTHORIZATION_LOCK.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"authorization lock missing: {lock_path}")
    contract = read_json(contract_path)
    token = read_json(token_path)
    lock = read_json(lock_path)
    if lock.get("status") != "LOCKED":
        raise RuntimeError("D7 authorization lock is not LOCKED")
    if lock.get("contract_sha256") != sha256_file(contract_path):
        raise RuntimeError("D7 contract/lock SHA mismatch")
    if lock.get("token_sha256") != sha256_file(token_path):
        raise RuntimeError("D7 token/lock SHA mismatch")
    if lock.get("evaluator_source_sha256") != evaluator_sha:
        raise RuntimeError("D7 lock evaluator binding mismatch")
    if contract.get("status") != "AUTHORIZED":
        raise RuntimeError("D7 contract is not AUTHORIZED")
    if contract.get("single_use") is not True or contract.get("evaluation_count") != 1:
        raise RuntimeError("D7 is not single-use")
    if contract.get("mandatory_result_label") != MANDATORY_LABEL:
        raise RuntimeError("D7 disclosure label changed")
    if contract.get("evaluator_source_sha256") != evaluator_sha:
        raise RuntimeError("D7 evaluator binding mismatch")
    if contract.get("data_root_text_sha256") != sha256_text(data_root_text):
        raise RuntimeError("D7 data-root binding mismatch")
    if contract.get("output_dir_text_sha256") != sha256_text(output_dir_text):
        raise RuntimeError("D7 output-dir binding mismatch")
    if contract.get("authorization_token_sha256") != sha256_file(token_path):
        raise RuntimeError("D7 token hash mismatch")
    if token.get("authorization_id") != contract.get("authorization_id"):
        raise RuntimeError("authorization ID mismatch")
    if token.get("evaluator_source_sha256") != evaluator_sha:
        raise RuntimeError("token evaluator binding mismatch")
    if token.get("data_root_text_sha256") != sha256_text(data_root_text):
        raise RuntimeError("token data-root binding mismatch")
    if token.get("output_dir_text_sha256") != sha256_text(output_dir_text):
        raise RuntimeError("token output-dir binding mismatch")
    return contract, token, contract_path, token_path


def consume_token(token_path: Path) -> Path:
    consumed = token_path.with_name(token_path.name + ".consumed")
    if consumed.exists():
        raise RuntimeError("authorization token already consumed")
    os.replace(token_path, consumed)
    consumed.chmod(0o444)
    return consumed


def commit_stage(
    parent: Path,
    name: str,
    *,
    arrays: dict[str, np.ndarray] | None,
    metrics: dict[str, Any],
    bindings: dict[str, Any],
    metadata: list[dict[str, Any]] | None = None,
    pair_rows: list[dict[str, Any]] | None = None,
) -> tuple[Path, str]:
    final = parent / name
    if final.exists():
        raise RuntimeError(f"stage already exists: {final}")
    tmp = Path(tempfile.mkdtemp(prefix=f".{name}_tmp_", dir=parent))
    try:
        if arrays is not None:
            atomic_npz(tmp / "outputs.npz", **arrays)
        atomic_json(tmp / "metrics.json", metrics)
        atomic_json(tmp / "bindings.json", bindings)
        if metadata is not None:
            with (tmp / "stable_items.jsonl").open("w", encoding="utf-8") as handle:
                for record in metadata:
                    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        if pair_rows is not None:
            with (tmp / "test_pair_window_manifest.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
                writer.writeheader()
                writer.writerows(pair_rows)
        files = {
            p.name: sha256_file(p)
            for p in sorted(tmp.iterdir())
            if p.is_file()
        }
        atomic_json(tmp / "LOCK.json", {
            "stage": name,
            "status": "LOCKED",
            "files": files,
        })
        atomic_text(tmp / "COMPLETE", name + "_COMPLETE\n")
        os.replace(tmp, final)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    digest = sha256_json({
        str(p.relative_to(final)): sha256_file(p)
        for p in sorted(final.rglob("*")) if p.is_file()
    })
    return final, digest


def load_stage_outputs(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / "outputs.npz", allow_pickle=False) as value:
        return {key: np.asarray(value[key]).copy() for key in value.files}


def decode_a1_chunk(
    arrays: dict[str, np.ndarray],
    start: int,
    stop: int,
    certified: Any,
    route: Any,
    helper: Any,
) -> dict[str, np.ndarray]:
    n = stop - start
    out: dict[str, np.ndarray] = {
        "validation_item_index": np.arange(start, stop, dtype=np.int64),
        "margin": np.empty(n, dtype=np.float64),
        "attack_prediction": np.empty(n, dtype=np.uint8),
        "count_prediction": np.empty(n, dtype=np.int64),
        "route_ids": np.full((n, 4), -1, dtype=np.int16),
        "source_mask": np.zeros(n, dtype=np.uint16),
        "transit_mask": np.zeros(n, dtype=np.uint16),
        "victim_mask": np.zeros(n, dtype=np.uint16),
        "path_mask": np.zeros(n, dtype=np.uint16),
        "primary_gap": np.empty(n, dtype=np.float64),
        "lexicographic_gap": np.empty(n, dtype=np.float64),
        "primary_integrality_error": np.empty(n, dtype=np.float64),
        "lexicographic_integrality_error": np.empty(n, dtype=np.float64),
        "primary_bound_violation": np.empty(n, dtype=np.float64),
        "lexicographic_bound_violation": np.empty(n, dtype=np.float64),
        "primary_linear_violation": np.empty(n, dtype=np.float64),
        "lexicographic_linear_violation": np.empty(n, dtype=np.float64),
        "primary_objective_difference": np.empty(n, dtype=np.float64),
        "lexicographic_objective_difference": np.empty(n, dtype=np.float64),
    }

    for local, index in enumerate(range(start, stop)):
        h = certified.decode_best_attack_hypothesis(
            graph_logit=float(arrays["graph_logits"][index]),
            count_logits=arrays["count_logits"][index].astype(np.float64),
            source_logits=arrays["source_logits"][index].astype(np.float64),
            transit_logits=arrays["transit_logits"][index].astype(np.float64),
            victim_logits=arrays["victim_logits"][index].astype(np.float64),
            path_logits=arrays["path_logits"][index].astype(np.float64),
        )
        if (
            h.certificate_status != "CERTIFIED"
            or h.certificate_policy_id != POLICY_ID
            or h.primary_certificate.certificate_status != "CERTIFIED"
            or h.lexicographic_certificate.certificate_status != "CERTIFIED"
        ):
            raise RuntimeError(f"A1 item {index} is not certified")

        decoded = certified.apply_margin_threshold(h, A1_MARGIN_THRESHOLD)
        out["margin"][local] = h.margin
        out["attack_prediction"][local] = decoded.attack
        out["count_prediction"][local] = decoded.attacker_count

        if decoded.attack == 1:
            combined = route.combine_route_ids(
                decoded.route_ids, expected_k=decoded.attacker_count
            )
            if (
                combined.source_mask != decoded.source_mask
                or combined.transit_mask != decoded.transit_mask
                or combined.victim_mask != decoded.victim_mask
                or combined.path_mask != decoded.path_mask
            ):
                raise RuntimeError(f"A1 legality mismatch at item {index}")
            out["route_ids"][local, :len(decoded.route_ids)] = np.asarray(
                decoded.route_ids, dtype=np.int16
            )
            for role in ROLE_NAMES:
                out[f"{role}_mask"][local] = getattr(decoded, f"{role}_mask")
        elif (
            decoded.attacker_count != 0
            or decoded.route_ids
            or decoded.source_mask
            or decoded.transit_mask
            or decoded.victim_mask
            or decoded.path_mask
        ):
            raise RuntimeError(f"A1 normal output nonempty at item {index}")

        pc = h.primary_certificate
        lc = h.lexicographic_certificate
        out["primary_gap"][local] = pc.raw_reported_mip_gap
        out["lexicographic_gap"][local] = lc.raw_reported_mip_gap
        out["primary_integrality_error"][local] = pc.maximum_integrality_error
        out["lexicographic_integrality_error"][local] = lc.maximum_integrality_error
        out["primary_bound_violation"][local] = pc.maximum_bound_violation
        out["lexicographic_bound_violation"][local] = lc.maximum_bound_violation
        out["primary_linear_violation"][local] = pc.maximum_linear_constraint_violation
        out["lexicographic_linear_violation"][local] = lc.maximum_linear_constraint_violation
        out["primary_objective_difference"][local] = pc.objective_absolute_difference
        out["lexicographic_objective_difference"][local] = lc.objective_absolute_difference

    for role in ROLE_NAMES:
        out[f"{role}_prediction"] = helper.masks_to_binary(out[f"{role}_mask"])
    return out


def a1_metrics(all_out: dict[str, np.ndarray], labels: dict[str, np.ndarray], helper: Any) -> dict[str, Any]:
    roles = {role: all_out[f"{role}_prediction"] for role in ROLE_NAMES}
    metrics = helper.structured_group_metrics(
        graph_prediction=all_out["attack_prediction"],
        count_prediction=all_out["count_prediction"],
        role_predictions=roles,
        graph_score=all_out["margin"],
        labels=labels,
        count_classes=(0, 1, 2, 3, 4),
        raw_count_attack_only=False,
    )
    metrics["margin_auroc"] = helper.binary_auroc(all_out["margin"], labels["graph"])
    metrics["margin_average_precision"] = helper.average_precision(
        all_out["margin"], labels["graph"]
    )
    attack_count = int(np.sum(all_out["attack_prediction"]))
    legal_count = int(np.sum(all_out["attack_prediction"]))
    metrics["route_legality_rate"] = 1.0 if attack_count == 0 else legal_count / attack_count
    metrics["certificate_summary"] = {
        "policy_id": POLICY_ID,
        "item_count": int(all_out["margin"].shape[0]),
        "primary_MILPs_certified": int(all_out["margin"].shape[0]),
        "lexicographic_MILPs_certified": int(all_out["margin"].shape[0]),
        "maximum_primary_reported_mip_gap": float(np.max(all_out["primary_gap"])),
        "maximum_lexicographic_reported_mip_gap": float(np.max(all_out["lexicographic_gap"])),
        "maximum_integrality_error": float(max(
            np.max(all_out["primary_integrality_error"]),
            np.max(all_out["lexicographic_integrality_error"]),
        )),
        "maximum_bound_violation": float(max(
            np.max(all_out["primary_bound_violation"]),
            np.max(all_out["lexicographic_bound_violation"]),
        )),
        "maximum_linear_constraint_violation": float(max(
            np.max(all_out["primary_linear_violation"]),
            np.max(all_out["lexicographic_linear_violation"]),
        )),
        "maximum_objective_difference": float(max(
            np.max(all_out["primary_objective_difference"]),
            np.max(all_out["lexicographic_objective_difference"]),
        )),
    }
    return metrics


def run_a1(
    *,
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    paths: dict[str, Path],
    helper: Any,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    existing_e4 = output_dir / "E4_A1_CERTIFIED_STRUCTURED"
    if existing_e4.exists():
        if not (existing_e4 / "COMPLETE").is_file():
            raise RuntimeError("partial E4 directory exists")
        existing_outputs = load_stage_outputs(existing_e4)
        existing_metrics = read_json(existing_e4 / "metrics.json")
        return existing_e4, existing_metrics, existing_outputs

    route = import_source(paths["route_library"], "src.decoders.v5_xy_route_library")
    certified = import_source(
        paths["certified_decoder"],
        "src.decoders.v5_legal_xy_exact_decoder_certified",
    )
    working = output_dir / "A1_WORKING"
    chunks = working / "chunks"
    state_path = working / "STATE.json"
    working.mkdir(exist_ok=True)
    chunks.mkdir(exist_ok=True)
    n = int(arrays["graph_logits"].shape[0])
    total = (n + A1_CHUNK_SIZE - 1) // A1_CHUNK_SIZE
    if state_path.exists():
        state = read_json(state_path)
        if state.get("item_count") != n or state.get("chunk_size") != A1_CHUNK_SIZE:
            raise RuntimeError("A1 working contract changed")
    else:
        state = {
            "status": "IN_PROGRESS",
            "item_count": n,
            "chunk_size": A1_CHUNK_SIZE,
            "completed": [],
        }
        atomic_json(state_path, state)

    completed = {entry["filename"]: entry for entry in state["completed"]}
    for chunk_index, start in enumerate(range(0, n, A1_CHUNK_SIZE)):
        stop = min(start + A1_CHUNK_SIZE, n)
        filename = f"chunk_{chunk_index:05d}_{start:05d}_{stop:05d}.npz"
        path = chunks / filename
        if filename in completed:
            if sha256_file(path) != completed[filename]["sha256"]:
                raise RuntimeError(f"A1 chunk hash changed: {path}")
            print(
                f"D8_A1_chunk={chunk_index + 1}/{total} "
                f"coverage={stop}/{n} status=RESUMED",
                flush=True,
            )
            continue
        if path.exists():
            record = {
                "filename": filename,
                "start": start,
                "stop": stop,
                "sha256": sha256_file(path),
                "reconciled": True,
            }
            state["completed"].append(record)
            atomic_json(state_path, state)
            completed[filename] = record
            print(
                f"D8_A1_chunk={chunk_index + 1}/{total} "
                f"coverage={stop}/{n} status=RECONCILED",
                flush=True,
            )
            continue

        result = decode_a1_chunk(arrays, start, stop, certified, route, helper)
        atomic_npz(path, **result)
        record = {
            "filename": filename,
            "start": start,
            "stop": stop,
            "sha256": sha256_file(path),
            "reconciled": False,
        }
        state["completed"].append(record)
        atomic_json(state_path, state)
        completed[filename] = record
        print(
            f"D8_A1_chunk={chunk_index + 1}/{total} "
            f"coverage={stop}/{n} status=COMMITTED",
            flush=True,
        )

    state = read_json(state_path)
    records = sorted(state["completed"], key=lambda x: x["start"])
    if len(records) != total:
        raise RuntimeError("A1 chunk count incomplete")
    parts: dict[str, list[np.ndarray]] = {}
    expected_start = 0
    for record in records:
        if record["start"] != expected_start:
            raise RuntimeError("A1 chunk coverage is not contiguous")
        path = chunks / record["filename"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError("A1 chunk hash changed")
        with np.load(path, allow_pickle=False) as value:
            for key in value.files:
                parts.setdefault(key, []).append(np.asarray(value[key]).copy())
        expected_start = record["stop"]
    if expected_start != n:
        raise RuntimeError("A1 final coverage mismatch")
    all_out = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
    metrics = a1_metrics(all_out, labels, helper)
    stage_dir, digest = commit_stage(
        output_dir,
        "E4_A1_CERTIFIED_STRUCTURED",
        arrays=all_out,
        metrics=metrics,
        bindings={
            "D1_policy_id": POLICY_ID,
            "certified_decoder_sha256": EXPECTED_HASHES["certified_decoder"],
            "A1_margin_threshold": A1_MARGIN_THRESHOLD,
            "A1_chunk_count": total,
        },
    )
    state["status"] = "COMPLETE"
    state["E4_tree_sha256"] = digest
    atomic_json(state_path, state)
    return stage_dir, metrics, all_out


def finalize(
    *,
    root: Path,
    output_dir: Path,
    report_dir: Path,
    contract: dict[str, Any],
    token: dict[str, Any],
    contract_path: Path,
    consumed_token: Path,
    source_hashes: dict[str, str],
    inference_summary: dict[str, Any],
    stage_hashes: dict[str, str],
    raw_metrics: dict[str, Any],
    a0_metrics: dict[str, Any],
    a1_metrics_value: dict[str, Any],
) -> None:
    if report_dir.exists():
        raise RuntimeError("final report directory already exists")
    report_dir.mkdir(parents=True)
    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "result_label": MANDATORY_LABEL,
        "scientific_classification": "POST_HOC_TEST_INFORMED_NUMERICAL_RECOVERY",
        "authorization": {
            "authorization_id": token["authorization_id"],
            "contract_path": str(contract_path),
            "contract_sha256": sha256_file(contract_path),
            "consumed_token_path": str(consumed_token),
            "consumed_token_sha256": sha256_file(consumed_token),
            "single_use_consumed": True,
            "test_evaluation_count": 1,
            "successful_second_test_inference_authorized": False,
            "A1_only_continuation_allowed": True,
        },
        "frozen_system": {
            "architecture": (
                "Causal Depthwise-Separable Conv1D Temporal Encoder "
                "+ Two-Layer GraphConv"
            ),
            "checkpoint_seed": 107,
            "checkpoint_epoch": 25,
            "checkpoint_sha256": EXPECTED_HASHES["checkpoint"],
            "A0_thresholds": A0_THRESHOLDS,
            "A1_margin_threshold": A1_MARGIN_THRESHOLD,
            "policy_id": POLICY_ID,
        },
        "dataset": inference_summary,
        "pipeline": {
            "single_neural_inference_pass": True,
            "E1_committed_before_Raw_A0_A1": True,
            "E2_Raw_committed_before_A1": True,
            "E3_A0_committed_before_A1": True,
            "E4_A1_resumable_from_E1_without_test_reaccess": True,
            "stage_tree_sha256": stage_hashes,
        },
        "official_metrics": {
            "raw_neural": raw_metrics,
            "A0_lightweight": a0_metrics,
            "A1_certified_structured": a1_metrics_value,
        },
        "mandatory_disclosure": {
            "original_complete_blind_result_exists": False,
            "original_recovery_authorization_consumed": True,
            "P2_test_boundary_previously_crossed": True,
            "this_result_is_test_informed_and_post_hoc": True,
            "this_result_is_not_blind": True,
            "this_result_is_not_the_original_one_shot_result": True,
        },
        "source_hashes": source_hashes,
        "security_boundary": {
            "authorization_consumed": True,
            "test_path_formed": True,
            "test_directory_checked": True,
            "test_directory_enumerated": True,
            "test_tensor_files_opened": True,
            "test_tensors_deserialized": True,
            "model_checkpoint_loaded": True,
            "neural_inference_performed": True,
            "neural_inference_pass_count": 1,
            "threshold_selection_performed": False,
            "model_or_checkpoint_selection_performed": False,
            "decoder_selection_performed": False,
            "test_driven_changes_performed": False,
            "second_test_inference_authorized": False,
        },
        "next_stage": "CLOSE_V5_P2_AND_BEGIN_V6_PROSPECTIVE_PROTOCOL",
    }
    artifact_path = output_dir / f"{STAGE}.json"
    atomic_json(artifact_path, report)
    artifact_sha = sha256_file(artifact_path)
    report_copy = dict(report)
    report_copy["artifact"] = {"path": str(artifact_path), "sha256": artifact_sha}
    report_json = report_dir / f"{STAGE}.json"
    atomic_json(report_json, report_copy)
    report_md = report_dir / f"{STAGE}.md"
    atomic_text(
        report_md,
        f"""# {MANDATORY_LABEL}

## Status

- Status: **COMPLETE**
- Classification: **post-hoc, test-informed numerical recovery**
- Blind/untouched result: **no**
- Test items: **{inference_summary['item_count']}**
- Test pairs: **{inference_summary['pair_count']}**
- Neural inference passes: **1**

## Frozen groups

| Group | Graph accuracy | Graph F1 | Graph FPR | Strict all-task exactness |
|---|---:|---:|---:|---:|
| Raw | `{raw_metrics['graph']['accuracy']:.6f}` | `{raw_metrics['graph']['f1']:.6f}` | `{raw_metrics['graph']['fpr']:.6f}` | `{raw_metrics['strict_all_task_exactness']:.6f}` |
| A0 | `{a0_metrics['graph']['accuracy']:.6f}` | `{a0_metrics['graph']['f1']:.6f}` | `{a0_metrics['graph']['fpr']:.6f}` | `{a0_metrics['strict_all_task_exactness']:.6f}` |
| Certified A1 | `{a1_metrics_value['graph']['accuracy']:.6f}` | `{a1_metrics_value['graph']['f1']:.6f}` | `{a1_metrics_value['graph']['fpr']:.6f}` | `{a1_metrics_value['strict_all_task_exactness']:.6f}` |

## Mandatory disclosure

The original P2 blind one-shot evaluation did not produce a complete official
Raw/A0/A1 test result. The numerical policy, certified decoder, and staged
evaluator were repaired using validation and synthetic data. This completed
execution is therefore test-informed and post hoc. It is not an untouched
blind test and is not the original one-shot evaluation.

Raw and A0 were committed immutably before A1 began.

Artifact SHA-256: `{artifact_sha}`
""",
    )
    lock = {
        "stage": STAGE + "_LOCK",
        "status": "LOCKED",
        "artifact_sha256": artifact_sha,
        "report_json_sha256": sha256_file(report_json),
        "report_markdown_sha256": sha256_file(report_md),
        "authorization_consumed": True,
        "test_evaluation_count": 1,
        "second_test_inference_authorized": False,
    }
    lock_path = report_dir / f"{STAGE}_LOCK.json"
    atomic_json(lock_path, lock)
    marker = report_dir / COMPLETE
    atomic_text(marker, COMPLETE + "\n")
    sums = report_dir / f"{STAGE}_SHA256SUMS.txt"
    atomic_text(
        sums,
        "".join(
            f"{sha256_file(path)}  {path}\n"
            for path in (
                artifact_path, report_json, report_md, lock_path, marker,
                contract_path, consumed_token,
            )
        ),
    )

    print(COMPLETE)
    print("status=COMPLETE")
    print("result_label=P2_POST_HOC_TEST_INFORMED_NUMERICAL_RECOVERY_EVALUATION")
    print(f"test_items={inference_summary['item_count']}")
    print(f"test_pairs={inference_summary['pair_count']}")
    print(f"test_batches={inference_summary['batch_count']}")
    print(f"Raw_graph_accuracy={raw_metrics['graph']['accuracy']:.17g}")
    print(f"A0_graph_accuracy={a0_metrics['graph']['accuracy']:.17g}")
    print(f"A1_graph_accuracy={a1_metrics_value['graph']['accuracy']:.17g}")
    print(
        f"Raw_strict_all_task_exactness="
        f"{raw_metrics['strict_all_task_exactness']:.17g}"
    )
    print(
        f"A0_strict_all_task_exactness="
        f"{a0_metrics['strict_all_task_exactness']:.17g}"
    )
    print(
        f"A1_strict_all_task_exactness="
        f"{a1_metrics_value['strict_all_task_exactness']:.17g}"
    )
    print("blind_result=false")
    print("post_hoc_test_informed=true")
    print("authorization_consumed=true")
    print("test_evaluation_count=1")
    print("second_test_inference_authorized=false")
    print(f"artifact={artifact_path}")
    print(f"artifact_sha256={artifact_sha}")
    print(f"report_json={report_json}")
    print(f"report_markdown={report_md}")
    print(f"sha256s={sums}")


def initial_run(args: argparse.Namespace) -> int:
    root = args.repo_root.expanduser().resolve()
    evaluator_path = Path(__file__).resolve()
    evaluator_sha = sha256_file(evaluator_path)
    paths, source_hashes = verify_non_test_chain(root)

    data_root_text = os.path.abspath(os.path.expanduser(args.data_root))
    output_dir_text = os.path.abspath(os.path.expanduser(args.output_dir))
    auth_dir = args.authorization_dir.expanduser().resolve()
    contract, token, contract_path, token_path = validate_authorization(
        auth_dir=auth_dir,
        evaluator_sha=evaluator_sha,
        data_root_text=data_root_text,
        output_dir_text=output_dir_text,
    )

    output_dir = Path(output_dir_text)
    report_dir = root / "reports/v5/p2_post_hoc_test_informed_numerical_recovery_evaluation"
    if output_dir.exists() or report_dir.exists():
        raise RuntimeError("D8 output/report already exists; do not overwrite")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()

    try:
        consumed = consume_token(token_path)
    except Exception:
        output_dir.rmdir()
        raise

    try:
        atomic_json(output_dir / ACCESS_STARTED, {
            "stage": STAGE,
            "event": "TEST_ACCESS_STARTED",
            "authorization_id": token["authorization_id"],
            "contract_sha256": sha256_file(contract_path),
            "consumed_token_sha256": sha256_file(consumed),
            "evaluator_source_sha256": evaluator_sha,
            "mandatory_result_label": MANDATORY_LABEL,
            "test_access_started": True,
            "test_evaluation_count": 1,
        })
        run_state = {
            "authorization_id": token["authorization_id"],
            "contract_path": str(contract_path),
            "contract_sha256": sha256_file(contract_path),
            "consumed_token_path": str(consumed),
            "consumed_token_sha256": sha256_file(consumed),
            "data_root_text_sha256": sha256_text(data_root_text),
            "output_dir_text_sha256": sha256_text(output_dir_text),
            "inference_summary": None,
            "stage_hashes": {},
            "source_hashes": source_hashes,
        }
        atomic_json(output_dir / "D8_RUN_STATE.json", run_state)

        # First test-path operation occurs only here, after token consumption.
        data_root = Path(data_root_text)
        helper = import_source(paths["helper_evaluator"], "v5_p2_d8_helper")
        mask = helper.load_physical_mask(paths["physical_mask"])
        dataset = helper.FrozenP2RecoveryTestDataset(
            data_root=data_root,
            physical_port_mask=mask,
        )

        torch.manual_seed(0)
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.use_deterministic_algorithms(True)
        device = torch.device("cpu")
        model, model_summary = helper.build_frozen_model(
            paths={
                "b3_model": paths["b3_model"],
                "task_d_model": paths["task_d_model"],
                "edge_index": paths["edge_index"],
                "checkpoint": paths["checkpoint"],
            },
            device=device,
        )
        arrays, metadata, inference_summary = helper.perform_single_inference_pass(
            dataset=dataset, model=model, device=device
        )
        if inference_summary["item_count"] != EXPECTED_TEST_ITEMS:
            raise RuntimeError("test item count changed")
        if inference_summary["pair_count"] != EXPECTED_TEST_PAIRS:
            raise RuntimeError("test pair count changed")
        if inference_summary["batch_count"] != EXPECTED_TEST_BATCHES:
            raise RuntimeError("test batch count changed")
        labels = helper.validate_labels(arrays)

        e1_dir, e1_hash = commit_stage(
            output_dir,
            "E1_NEURAL_ARCHIVE",
            arrays=arrays,
            metrics={
                "item_count": inference_summary["item_count"],
                "batch_count": inference_summary["batch_count"],
                "pair_count": inference_summary["pair_count"],
                "single_neural_inference_pass": True,
                "model_summary": model_summary,
            },
            bindings={
                "checkpoint_sha256": EXPECTED_HASHES["checkpoint"],
                "evaluator_source_sha256": evaluator_sha,
                "authorization_id": token["authorization_id"],
            },
            metadata=metadata,
            pair_rows=dataset.pair_rows,
        )
        run_state["inference_summary"] = inference_summary
        run_state["stage_hashes"]["E1_NEURAL_ARCHIVE"] = e1_hash
        atomic_json(output_dir / "D8_RUN_STATE.json", run_state)
        print("D8_stage=E1_NEURAL_ARCHIVE status=COMMITTED", flush=True)

        raw, a0, metrics = helper.produce_raw_and_a0(arrays, labels)
        e2_dir, e2_hash = commit_stage(
            output_dir,
            "E2_RAW_NEURAL",
            arrays=raw,
            metrics=metrics["raw_neural"],
            bindings={"E1_tree_sha256": e1_hash},
        )
        run_state["stage_hashes"]["E2_RAW_NEURAL"] = e2_hash
        atomic_json(output_dir / "D8_RUN_STATE.json", run_state)
        print("D8_stage=E2_RAW_NEURAL status=COMMITTED", flush=True)
        e3_dir, e3_hash = commit_stage(
            output_dir,
            "E3_A0_LIGHTWEIGHT",
            arrays=a0,
            metrics=metrics["A0_lightweight"],
            bindings={"E1_tree_sha256": e1_hash, "E2_tree_sha256": e2_hash},
        )
        run_state["stage_hashes"]["E3_A0_LIGHTWEIGHT"] = e3_hash
        atomic_json(output_dir / "D8_RUN_STATE.json", run_state)
        print("D8_stage=E3_A0_LIGHTWEIGHT status=COMMITTED", flush=True)

        e4_dir, a1_metric, a1_out = run_a1(
            output_dir=output_dir,
            arrays=arrays,
            labels=labels,
            paths=paths,
            helper=helper,
        )
        e4_hash = sha256_json({
            str(p.relative_to(e4_dir)): sha256_file(p)
            for p in sorted(e4_dir.rglob("*")) if p.is_file()
        })
        run_state["stage_hashes"]["E4_A1_CERTIFIED_STRUCTURED"] = e4_hash
        atomic_json(output_dir / "D8_RUN_STATE.json", run_state)

        finalize(
            root=root,
            output_dir=output_dir,
            report_dir=report_dir,
            contract=contract,
            token=token,
            contract_path=contract_path,
            consumed_token=consumed,
            source_hashes=source_hashes,
            inference_summary=inference_summary,
            stage_hashes=run_state["stage_hashes"],
            raw_metrics=metrics["raw_neural"],
            a0_metrics=metrics["A0_lightweight"],
            a1_metrics_value=a1_metric,
        )
        return 0
    except Exception as exc:
        e1_complete = (output_dir / "E1_NEURAL_ARCHIVE/COMPLETE").is_file()
        e2_complete = (output_dir / "E2_RAW_NEURAL/COMPLETE").is_file()
        e3_complete = (output_dir / "E3_A0_LIGHTWEIGHT/COMPLETE").is_file()
        continuation = bool(e1_complete)
        failure = {
            "stage": STAGE,
            "status": (
                "A1_CONTINUATION_AVAILABLE"
                if continuation
                else "IRREVERSIBLE_FAILURE_AFTER_AUTHORIZATION_CONSUMPTION"
            ),
            "authorization_consumed": True,
            "test_access_started": True,
            "E1_committed": e1_complete,
            "E2_Raw_committed": e2_complete,
            "E3_A0_committed": e3_complete,
            "A1_only_resume_allowed": continuation,
            "second_test_inference_authorized": False,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(output_dir / f"{FAILURE}.json", failure)
        print(failure["status"])
        print(f"FAIL: {type(exc).__name__}: {exc}")
        if continuation:
            print("Run the A1-only resume launcher. It will not access test tensors.")
        else:
            print("Authorization was consumed before E1 completed. Do not rerun.")
        return 1


def resume_a1(args: argparse.Namespace) -> int:
    root = args.repo_root.expanduser().resolve()
    output_dir = Path(os.path.abspath(os.path.expanduser(args.output_dir)))
    report_dir = root / "reports/v5/p2_post_hoc_test_informed_numerical_recovery_evaluation"
    if report_dir.exists() or (output_dir / f"{STAGE}.json").exists():
        raise RuntimeError("D8 is already complete")
    paths, source_hashes = verify_non_test_chain(root)
    access = read_json(output_dir / ACCESS_STARTED)
    run_state = read_json(output_dir / "D8_RUN_STATE.json")
    if access.get("test_evaluation_count") != 1:
        raise RuntimeError("access record changed")
    if not (output_dir / "E1_NEURAL_ARCHIVE/COMPLETE").is_file():
        raise RuntimeError("resume prerequisite missing: E1_NEURAL_ARCHIVE")
    consumed = Path(run_state["consumed_token_path"])
    if not consumed.is_file():
        raise RuntimeError("consumed token missing")
    contract_path = Path(run_state["contract_path"])
    contract = read_json(contract_path)
    token = read_json(consumed)

    helper = import_source(paths["helper_evaluator"], "v5_p2_d8_helper_resume")
    arrays = load_stage_outputs(output_dir / "E1_NEURAL_ARCHIVE")
    labels = helper.validate_labels(arrays)
    raw, a0, recomputed_metrics = helper.produce_raw_and_a0(arrays, labels)

    e2_path = output_dir / "E2_RAW_NEURAL"
    if e2_path.exists():
        if not (e2_path / "COMPLETE").is_file():
            raise RuntimeError("partial E2 directory exists")
        raw_metrics = read_json(e2_path / "metrics.json")
    else:
        e2_path, e2_hash = commit_stage(
            output_dir,
            "E2_RAW_NEURAL",
            arrays=raw,
            metrics=recomputed_metrics["raw_neural"],
            bindings={"E1_tree_sha256": run_state["stage_hashes"]["E1_NEURAL_ARCHIVE"]},
        )
        run_state["stage_hashes"]["E2_RAW_NEURAL"] = e2_hash
        atomic_json(output_dir / "D8_RUN_STATE.json", run_state)
        raw_metrics = recomputed_metrics["raw_neural"]
        print("D8_stage=E2_RAW_NEURAL status=CONTINUED", flush=True)

    e3_path = output_dir / "E3_A0_LIGHTWEIGHT"
    if e3_path.exists():
        if not (e3_path / "COMPLETE").is_file():
            raise RuntimeError("partial E3 directory exists")
        a0_metrics = read_json(e3_path / "metrics.json")
    else:
        e3_path, e3_hash = commit_stage(
            output_dir,
            "E3_A0_LIGHTWEIGHT",
            arrays=a0,
            metrics=recomputed_metrics["A0_lightweight"],
            bindings={
                "E1_tree_sha256": run_state["stage_hashes"]["E1_NEURAL_ARCHIVE"],
                "E2_tree_sha256": run_state["stage_hashes"]["E2_RAW_NEURAL"],
            },
        )
        run_state["stage_hashes"]["E3_A0_LIGHTWEIGHT"] = e3_hash
        atomic_json(output_dir / "D8_RUN_STATE.json", run_state)
        a0_metrics = recomputed_metrics["A0_lightweight"]
        print("D8_stage=E3_A0_LIGHTWEIGHT status=CONTINUED", flush=True)

    e4_dir, a1_metric, a1_out = run_a1(
        output_dir=output_dir,
        arrays=arrays,
        labels=labels,
        paths=paths,
        helper=helper,
    )
    e4_hash = sha256_json({
        str(p.relative_to(e4_dir)): sha256_file(p)
        for p in sorted(e4_dir.rglob("*")) if p.is_file()
    })
    run_state["stage_hashes"]["E4_A1_CERTIFIED_STRUCTURED"] = e4_hash
    atomic_json(output_dir / "D8_RUN_STATE.json", run_state)
    finalize(
        root=root,
        output_dir=output_dir,
        report_dir=report_dir,
        contract=contract,
        token=token,
        contract_path=contract_path,
        consumed_token=consumed,
        source_hashes=source_hashes,
        inference_summary=run_state["inference_summary"],
        stage_hashes=run_state["stage_hashes"],
        raw_metrics=raw_metrics,
        a0_metrics=a0_metrics,
        a1_metrics_value=a1_metric,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-only-self-test", action="store_true")
    parser.add_argument("--self-test-output", type=Path)
    parser.add_argument("--resume-a1-only", action="store_true")
    parser.add_argument("--authorization-dir", type=Path)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    if args.source_only_self_test:
        if args.self_test_output is None:
            parser.error("--self-test-output is required")
        self_test(
            args.repo_root.expanduser().resolve(),
            args.self_test_output.expanduser().resolve(),
        )
        return 0

    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.resume_a1_only:
        return resume_a1(args)

    if args.authorization_dir is None or args.data_root is None:
        parser.error("--authorization-dir and --data-root are required")
    return initial_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
