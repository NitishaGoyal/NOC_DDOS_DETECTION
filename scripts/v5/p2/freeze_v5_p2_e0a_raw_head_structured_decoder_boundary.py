#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TECHNICAL_NAME = (
    "Causal Depthwise-Separable Conv1D Temporal Encoder "
    "+ Two-Layer GraphConv"
)
PAPER_SHORT_NAME = "Temporal Conv1D-GraphConv"
TEMPORAL_FRONTEND_REFERENCE = "B3"
SELECTED_SEED = 107
SELECTED_EPOCH = 25
EXPECTED_CHECKPOINT_SHA256 = (
    "82314baedb4bf3969842abf9076679682369daae2e57db07d4b4aeaa8623f7cc"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--e0-dir", type=Path, required=True)
    parser.add_argument("--architecture-freeze-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-freeze-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a top-level JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def require_false(mapping: dict[str, Any], field: str, source: str) -> None:
    if mapping.get(field) is not False:
        raise RuntimeError(
            f"{source}: expected {field}=false, observed {mapping.get(field)!r}"
        )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    e0_dir = args.e0_dir.resolve()
    architecture_dir = args.architecture_freeze_dir.resolve()
    checkpoint_dir = args.checkpoint_freeze_dir.resolve()
    output_dir = args.output_dir.resolve()

    e0_json = e0_dir / "V5_P2_E0_VALIDATION_THRESHOLD_PROTOCOL_LOCK.json"
    e0_md = e0_dir / "V5_P2_E0_VALIDATION_THRESHOLD_PROTOCOL_LOCK.md"
    e0_marker = e0_dir / "V5_P2_E0_VALIDATION_THRESHOLD_PROTOCOL_LOCK_COMPLETE"
    arch_json = architecture_dir / "V5_P2_TASK_D_ARCHITECTURE_SELECTION_FREEZE.json"
    ckpt_json = checkpoint_dir / "V5_P2_TASK_D_CHECKPOINT_SELECTION_FREEZE.json"
    ckpt_file = checkpoint_dir / "selected_graphconv_checkpoint.pt"

    for path in (e0_json, e0_md, e0_marker, arch_json, ckpt_json, ckpt_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = output_dir / "V5_P2_E0A_RAW_HEAD_STRUCTURED_DECODER_BOUNDARY_COMPLETE"
    if complete_marker.is_file():
        print("V5_P2_E0A_RAW_HEAD_STRUCTURED_DECODER_BOUNDARY_ALREADY_COMPLETE")
        print(complete_marker.read_text(encoding="utf-8"), end="")
        return
    if any(output_dir.iterdir()):
        raise RuntimeError("E0A output directory is not empty")

    e0 = load_json(e0_json)
    arch = load_json(arch_json)
    ckpt = load_json(ckpt_json)

    if e0.get("status") != "COMPLETE":
        raise RuntimeError("E0 is not complete")
    state = e0.get("state_after_e0")
    if not isinstance(state, dict):
        raise RuntimeError("E0 state_after_e0 is missing")
    expected_state = {
        "architecture_selected": True,
        "checkpoint_selected": True,
        "threshold_protocol_locked": True,
        "validation_predictions_exported": False,
        "validation_threshold_selection_performed": False,
        "thresholds_frozen": False,
        "decoder_frozen": False,
        "ready_for_one_shot_test": False,
    }
    for field, expected in expected_state.items():
        if state.get(field) is not expected:
            raise RuntimeError(
                f"E0 state mismatch for {field}: expected {expected!r}, "
                f"observed {state.get(field)!r}"
            )

    e0_arch = e0.get("architecture")
    if not isinstance(e0_arch, dict):
        raise RuntimeError("E0 architecture record is missing")
    if e0_arch.get("technical_name") != TECHNICAL_NAME:
        raise RuntimeError("E0 technical name does not match selected architecture")
    if e0_arch.get("internal_protocol_reference") != TEMPORAL_FRONTEND_REFERENCE:
        raise RuntimeError("Unexpected temporal frontend reference")

    if arch.get("status") != "COMPLETE" or arch.get("architecture_selected") is not True:
        raise RuntimeError("Architecture-selection freeze is incomplete")
    if arch.get("selected_architecture") != "graphconv":
        raise RuntimeError("Selected architecture is not graphconv")

    if ckpt.get("status") != "COMPLETE" or ckpt.get("checkpoint_selected") is not True:
        raise RuntimeError("Checkpoint-selection freeze is incomplete")
    if ckpt.get("selected_architecture") != "graphconv":
        raise RuntimeError("Selected checkpoint is not graphconv")
    if int(ckpt.get("selected_seed")) != SELECTED_SEED:
        raise RuntimeError("Unexpected selected seed")
    if int(ckpt.get("selected_epoch")) != SELECTED_EPOCH:
        raise RuntimeError("Unexpected selected epoch")

    checkpoint_sha = sha256(ckpt_file)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Checkpoint hash mismatch:\n"
            f"expected={EXPECTED_CHECKPOINT_SHA256}\nactual={checkpoint_sha}"
        )
    if ckpt.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("Checkpoint JSON hash does not match checkpoint file")

    e0_sec = e0.get("security_boundary")
    if not isinstance(e0_sec, dict):
        raise RuntimeError("E0 security boundary is missing")
    for field in (
        "test_directory_existence_checked",
        "test_directory_enumerated",
        "test_tensor_files_opened",
        "test_tensors_deserialized",
        "test_evaluation_performed",
    ):
        require_false(e0_sec, field, "E0")
    for name, source in (("architecture freeze", arch), ("checkpoint freeze", ckpt)):
        require_false(source, "test_directory_enumerated", name)
        require_false(source, "test_tensors_deserialized", name)

    stage_order = [
        {"order": 0, "stage": "E0", "status": "COMPLETE", "name": "raw_threshold_protocol_lock"},
        {"order": 1, "stage": "E0A", "status": "COMPLETE", "name": "raw_head_structured_decoder_boundary"},
        {"order": 2, "stage": "L0", "status": "NEXT", "name": "structured_decoder_protocol_lock"},
        {"order": 3, "stage": "L1", "status": "PENDING", "name": "route_library_and_unit_tests"},
        {"order": 4, "stage": "L2", "status": "PENDING", "name": "exact_decoder_implementation"},
        {"order": 5, "stage": "L3", "status": "PENDING", "name": "hardware_beam_decoder_implementation"},
        {"order": 6, "stage": "E1", "status": "PENDING", "name": "immutable_validation_logit_export"},
        {"order": 7, "stage": "E2", "status": "PENDING", "name": "raw_threshold_selection_and_freeze"},
        {"order": 8, "stage": "L4", "status": "PENDING", "name": "decoder_validation_selection_and_freeze"},
        {"order": 9, "stage": "L5", "status": "PENDING", "name": "final_pretest_freeze_bundle"},
        {"order": 10, "stage": "PREFLIGHT", "status": "PENDING", "name": "non_test_execution_preflight"},
        {"order": 11, "stage": "P2_ONE_SHOT", "status": "PENDING", "name": "one_shot_blind_evaluation"},
    ]

    boundary = {
        "stage": "V5_P2_E0A_RAW_HEAD_STRUCTURED_DECODER_BOUNDARY",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "append_only_policy": {
            "e0_preserved_unchanged": True,
            "architecture_freeze_preserved_unchanged": True,
            "checkpoint_freeze_preserved_unchanged": True,
            "e0a_is_clarification_not_replacement": True,
        },
        "naming": {
            "temporal_frontend_reference": TEMPORAL_FRONTEND_REFERENCE,
            "selected_full_architecture": TECHNICAL_NAME,
            "paper_short_name": PAPER_SHORT_NAME,
            "prohibited_ambiguous_usage": (
                "Do not refer to the complete selected model only as B3; "
                "B3 identifies the temporal frontend reference."
            ),
        },
        "frozen_neural_candidate": {
            "selected_architecture": "graphconv",
            "technical_name": TECHNICAL_NAME,
            "temporal_frontend_reference": TEMPORAL_FRONTEND_REFERENCE,
            "selected_seed": SELECTED_SEED,
            "selected_epoch": SELECTED_EPOCH,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_path": str(ckpt_file),
            "architecture_change_allowed": False,
            "checkpoint_change_allowed": False,
            "retraining_allowed": False,
        },
        "systems": {
            "A0_raw_neural_baseline": {
                "name": "Independently Thresholded Temporal Conv1D-GraphConv",
                "graph_decision": "Use the E2 graph threshold selected under E0",
                "count_decision": "Argmax over K1-K4; graph-negative maps to count zero only in explicit A0 output policy",
                "role_decisions": "Apply independently frozen E2 source, transit, victim, and path thresholds",
                "structured_route_cleanup_allowed": False,
                "metrics_reported_separately": True,
            },
            "A1_exact_structured_decoder": {
                "name": "Exact Deterministic Legal-Route Decoder",
                "graph_head_is_hard_gate": False,
                "full_count_logits_used": True,
                "legal_route_semantics_frozen_by": "L0",
                "implementation_frozen_by": "L2",
                "configuration_selected_by": "L4",
                "continuous_margin_required": True,
                "metrics_reported_separately": True,
            },
            "A2_hardware_beam_decoder": {
                "name": "Hardware-Oriented Deterministic Beam Decoder",
                "legal_route_semantics_frozen_by": "L0",
                "implementation_frozen_by": "L3",
                "configuration_selected_by": "L4",
                "exact_decoder_equivalence_report_required": True,
                "metrics_reported_separately": True,
            },
        },
        "reporting_boundary": {
            "raw_logits_shared_across_A0_A1_A2": True,
            "A0_metrics_may_be_overwritten_by_decoder": False,
            "A1_metrics_may_be_reported_as_raw_metrics": False,
            "A2_metrics_may_be_reported_as_exact_metrics": False,
            "all_three_systems_reported_on_test": True,
            "post_test_system_selection_allowed": False,
            "post_test_threshold_change_allowed": False,
            "post_test_decoder_change_allowed": False,
        },
        "validation_archive_policy": {
            "fresh_E1_archive_required": True,
            "archive_source": "selected seed-107 epoch-25 Temporal Conv1D-GraphConv checkpoint",
            "raw_logits_required": True,
            "validation_only": True,
            "legacy_B4_cache_reuse_allowed": False,
            "legacy_B6_cache_reuse_allowed": False,
        },
        "authorization_boundary": {
            "E0A_authorizes_validation_export": False,
            "E0A_authorizes_decoder_implementation": False,
            "L0_required_before_L1": True,
            "L1_required_before_L2_and_L3": True,
            "L2_and_L3_required_before_E1": True,
            "E2_required_before_L5": True,
            "L4_required_before_L5": True,
            "L5_required_before_preflight": True,
            "preflight_required_before_test": True,
        },
        "official_P2_one_shot_policy": {
            "test_currently_untouched": True,
            "single_authorized_test_execution": True,
            "predeclared_systems": ["A0", "A1", "A2"],
            "all_predeclared_systems_must_be_reported": True,
            "no_variant_may_be_selected_after_test": True,
            "raw_logits_and_all_decoded_outputs_saved_in_one_cache": True,
            "test_threshold_reselection_allowed": False,
            "test_decoder_reselection_allowed": False,
            "test_checkpoint_reselection_allowed": False,
        },
        "stage_order": stage_order,
        "state_after_E0A": {
            "architecture_selected": True,
            "checkpoint_selected": True,
            "threshold_protocol_locked": True,
            "raw_decoder_boundary_locked": True,
            "structured_decoder_protocol_locked": False,
            "route_library_frozen": False,
            "exact_decoder_implemented": False,
            "beam_decoder_implemented": False,
            "validation_predictions_exported": False,
            "raw_thresholds_frozen": False,
            "decoder_parameters_frozen": False,
            "decoder_frozen": False,
            "ready_for_one_shot_test": False,
        },
        "security_boundary": {
            "selection_basis": "validation_only",
            "test_directory_existence_checked": False,
            "test_directory_enumerated": False,
            "test_tensor_files_opened": False,
            "test_tensors_deserialized": False,
            "test_evaluation_performed": False,
        },
        "evidence_hashes": {
            "E0_json_sha256": sha256(e0_json),
            "E0_markdown_sha256": sha256(e0_md),
            "E0_marker_sha256": sha256(e0_marker),
            "architecture_selection_json_sha256": sha256(arch_json),
            "checkpoint_selection_json_sha256": sha256(ckpt_json),
            "selected_checkpoint_sha256": checkpoint_sha,
        },
    }

    boundary_json = output_dir / "V5_P2_E0A_RAW_HEAD_STRUCTURED_DECODER_BOUNDARY.json"
    stage_json = output_dir / "V5_P2_E0A_PRETEST_SYSTEM_AND_STAGE_ORDER.json"
    lock_json = output_dir / "V5_P2_E0A_RAW_HEAD_STRUCTURED_DECODER_BOUNDARY_LOCK.json"
    md_path = output_dir / "V5_P2_E0A_RAW_HEAD_STRUCTURED_DECODER_BOUNDARY.md"

    write_json(boundary_json, boundary)
    write_json(stage_json, {"stage_order": stage_order})
    write_json(
        lock_json,
        {
            "stage": "V5_P2_E0A_RAW_HEAD_STRUCTURED_DECODER_BOUNDARY_LOCK",
            "status": "LOCKED",
            "selected_full_architecture": TECHNICAL_NAME,
            "temporal_frontend_reference": TEMPORAL_FRONTEND_REFERENCE,
            "selected_seed": SELECTED_SEED,
            "selected_epoch": SELECTED_EPOCH,
            "selected_checkpoint_sha256": checkpoint_sha,
            "A0_raw_baseline_frozen_as_required_report": True,
            "A1_exact_decoder_frozen_as_required_report": True,
            "A2_beam_decoder_frozen_as_required_report": True,
            "legacy_validation_cache_reuse_allowed": False,
            "legacy_test_cache_reuse_allowed": False,
            "next_stage": "V5_P2_L0_STRUCTURED_DECODER_PROTOCOL_LOCK",
            "validation_predictions_exported": False,
            "raw_thresholds_frozen": False,
            "decoder_frozen": False,
            "test_directory_enumerated": False,
            "test_tensors_deserialized": False,
            "ready_for_one_shot_test": False,
            "boundary_json_sha256": sha256(boundary_json),
            "stage_order_json_sha256": sha256(stage_json),
        },
    )

    md_path.write_text(
        f"""# V5 P2 E0A Raw-Head / Structured-Decoder Boundary\n\n"
        f"- Selected full architecture: **{TECHNICAL_NAME}**\n"
        f"- Temporal frontend reference: **{TEMPORAL_FRONTEND_REFERENCE}**\n"
        f"- Paper short name: **{PAPER_SHORT_NAME}**\n"
        f"- Selected seed: **{SELECTED_SEED}**\n"
        f"- Selected epoch: **{SELECTED_EPOCH}**\n"
        f"- Checkpoint SHA-256: `{checkpoint_sha}`\n\n"
        "The complete selected model must not be referred to only as B3. "
        "B3 is the temporal frontend reference.\n\n"
        "## Predeclared systems\n\n"
        "- **A0:** independently thresholded raw neural baseline\n"
        "- **A1:** exact deterministic legal-route decoder\n"
        "- **A2:** hardware-oriented deterministic beam decoder\n\n"
        "## Cache and reporting boundary\n\n"
        "- A fresh E1 validation-logit archive is required.\n"
        "- Legacy B4 and B6 caches may not be reused.\n"
        "- A0, A1, and A2 metrics remain separate.\n"
        "- No system, threshold, decoder, checkpoint, or variant may be "
        "selected after test access.\n\n"
        "## Frozen order\n\n"
        "```text\n"
        "E0   threshold protocol lock — complete\n"
        "E0A  raw-head / structured-decoder boundary — complete\n"
        "L0   structured decoder protocol lock — next\n"
        "L1   route-library generation and tests\n"
        "L2   exact decoder implementation\n"
        "L3   hardware beam decoder implementation\n"
        "E1   immutable validation-logit export\n"
        "E2   raw threshold selection and freeze\n"
        "L4   decoder validation selection and freeze\n"
        "L5   final pretest freeze bundle\n"
        "PREFLIGHT\n"
        "P2   one-shot blind evaluation of A0, A1, and A2\n"
        "```\n\n"
        "## Current state\n\n"
        "- Raw/decoder boundary locked: true\n"
        "- Structured decoder protocol locked: false\n"
        "- Validation predictions exported: false\n"
        "- Raw thresholds frozen: false\n"
        "- Decoder frozen: false\n"
        "- Test directory enumerated: false\n"
        "- Test tensors deserialized: false\n"
        "- Ready for one-shot test: false\n""",
        encoding="utf-8",
    )

    complete_marker.write_text(
        "stage=V5_P2_E0A_RAW_HEAD_STRUCTURED_DECODER_BOUNDARY\n"
        "status=COMPLETE\n"
        f"selected_full_architecture={TECHNICAL_NAME}\n"
        f"temporal_frontend_reference={TEMPORAL_FRONTEND_REFERENCE}\n"
        f"selected_seed={SELECTED_SEED}\n"
        f"selected_epoch={SELECTED_EPOCH}\n"
        f"checkpoint_sha256={checkpoint_sha}\n"
        "A0_raw_neural_baseline=predeclared\n"
        "A1_exact_structured_decoder=predeclared\n"
        "A2_hardware_beam_decoder=predeclared\n"
        "legacy_B4_cache_reuse_allowed=false\n"
        "legacy_B6_cache_reuse_allowed=false\n"
        "raw_decoder_boundary_locked=true\n"
        "structured_decoder_protocol_locked=false\n"
        "validation_predictions_exported=false\n"
        "raw_thresholds_frozen=false\n"
        "decoder_frozen=false\n"
        "test_directory_enumerated=false\n"
        "test_tensors_deserialized=false\n"
        "ready_for_one_shot_test=false\n"
        "next_stage=V5_P2_L0_STRUCTURED_DECODER_PROTOCOL_LOCK\n",
        encoding="utf-8",
    )

    digest_file = output_dir / "V5_P2_E0A_RAW_HEAD_STRUCTURED_DECODER_BOUNDARY_SHA256SUMS.txt"
    with digest_file.open("w", encoding="utf-8") as handle:
        for path in (
            e0_json, e0_md, e0_marker, arch_json, ckpt_json, ckpt_file,
            boundary_json, stage_json, lock_json, md_path, complete_marker,
        ):
            handle.write(f"{sha256(path)}  {path}\n")

    print("V5_P2_E0A_RAW_HEAD_STRUCTURED_DECODER_BOUNDARY_COMPLETE")
    print(f"selected_full_architecture={TECHNICAL_NAME}")
    print(f"temporal_frontend_reference={TEMPORAL_FRONTEND_REFERENCE}")
    print(f"selected_seed={SELECTED_SEED}")
    print(f"selected_epoch={SELECTED_EPOCH}")
    print(f"checkpoint_sha256={checkpoint_sha}")
    print("A0_raw_neural_baseline=predeclared")
    print("A1_exact_structured_decoder=predeclared")
    print("A2_hardware_beam_decoder=predeclared")
    print("legacy_B4_cache_reuse_allowed=false")
    print("legacy_B6_cache_reuse_allowed=false")
    print("raw_decoder_boundary_locked=true")
    print("structured_decoder_protocol_locked=false")
    print("validation_predictions_exported=false")
    print("raw_thresholds_frozen=false")
    print("decoder_frozen=false")
    print("test_directory_enumerated=false")
    print("test_tensors_deserialized=false")
    print("ready_for_one_shot_test=false")
    print("next_stage=V5_P2_L0_STRUCTURED_DECODER_PROTOCOL_LOCK")
    print(f"boundary_json={boundary_json}")
    print(f"boundary_markdown={md_path}")
    print(f"lock_json={lock_json}")
    print(f"sha256s={digest_file}")


if __name__ == "__main__":
    main()
