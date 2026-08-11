#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

STAGE = "V5_P2_C2A_LEGAL_NOC_DECODER_AMENDMENT"
COMPLETE = STAGE + "_COMPLETE"
EXPECTED_C2_COMPLETE = "V5_P2_C2_NEXT_DATASET_AND_MODEL_REVISION_CONTRACT_COMPLETE"
EXPECTED_C2_DECISION = (
    "FREEZE_FRESH_HARD_LEGITIMATE_MATCHED_DATASET_AND_"
    "ZERO_PARAMETER_PAIR_MARGIN_STUDY"
)
EXPECTED_C2_CONTRACT_SHA = (
    "d863a0b14d5113f283208bafa2e5804da52336f11f8ddf001e6c332c942f34b0"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    c2_dir = args.c2_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)

    c2_report_path = c2_dir / "V5_P2_C2_NEXT_DATASET_AND_MODEL_REVISION_CONTRACT.json"
    c2_lock_path = c2_dir / "V5_P2_C2_NEXT_DATASET_AND_MODEL_REVISION_CONTRACT_LOCK.json"
    c2_complete_path = c2_dir / "V5_P2_C2_NEXT_DATASET_AND_MODEL_REVISION_CONTRACT_COMPLETE"

    failures: list[str] = []
    warnings: list[str] = []
    for path in (c2_report_path, c2_lock_path, c2_complete_path):
        if not path.is_file():
            failures.append(f"missing C2 prerequisite: {path}")

    if failures:
        write_json(output_dir / f"{STAGE}.json", {
            "stage": STAGE, "status": "HOLD", "failures": failures,
            "warnings": warnings, "c2_modified": False,
            "p2_prediction_cache_accessed": False,
            "p2_test_tensor_contents_accessed": False,
        })
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        return 1

    c2_report = json.loads(c2_report_path.read_text(encoding="utf-8"))
    c2_lock = json.loads(c2_lock_path.read_text(encoding="utf-8"))

    if c2_complete_path.read_text(encoding="utf-8").strip() != EXPECTED_C2_COMPLETE:
        failures.append("C2 COMPLETE marker changed")
    if c2_report.get("status") != "COMPLETE":
        failures.append("C2 report is not COMPLETE")
    if c2_report.get("decision") != EXPECTED_C2_DECISION:
        failures.append("C2 decision changed")
    if c2_lock.get("report_sha256") != sha256_file(c2_report_path):
        failures.append("C2 report SHA mismatch")
    if c2_lock.get("contract_file_sha256") != sha256_file(c2_report_path):
        failures.append("C2 contract-file SHA mismatch")
    if c2_lock.get("contract_sha256") != EXPECTED_C2_CONTRACT_SHA:
        failures.append("C2 canonical contract SHA changed")
    if c2_lock.get("p2_prediction_cache_accessed") is not False:
        failures.append("C2 unexpectedly accessed the P2 prediction cache")
    if c2_lock.get("p2_test_tensor_contents_accessed") is not False:
        failures.append("C2 unexpectedly accessed P2 test tensors")

    if failures:
        write_json(output_dir / f"{STAGE}.json", {
            "stage": STAGE, "status": "HOLD", "failures": failures,
            "warnings": warnings, "c2_modified": False,
            "p2_prediction_cache_accessed": False,
            "p2_test_tensor_contents_accessed": False,
        })
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    policy_core = {
        "policy_name": "V5_P2_C2A_P2_VS_V6_LEGAL_NOC_DECODER_EVALUATION_POLICY",
        "policy_version": 1,
        "p2": {
            "official_b6_result_remains_immutable": True,
            "official_graph_accuracy": 0.7992456711812104,
            "official_graph_fpr": 0.27943313953488375,
            "official_graph_recall": 0.9897360703812317,
            "official_count_accuracy": 1.0,
            "decoder_development_source": "B4 validation prediction cache only",
            "decoder_selection_split": "validation",
            "b6_cache_use": "post-hoc exploratory structured decoding only",
            "b6_decoder_result_replaces_official_result": False,
            "test_model_rerun_allowed": False,
            "checkpoint_or_threshold_retuning_allowed": False,
            "required_posthoc_label": (
                "POST-HOC EXPLORATORY STRUCTURED DECODING ON FROZEN P2 OUTPUTS"
            ),
        },
        "v6": {
            "first_clean_blind_decoder_claim": True,
            "fresh_hard_legitimate_dataset_required": True,
            "checkpoint_selection": "validation only",
            "decoder_selection": "validation only",
            "threshold_selection": "validation only",
            "blind_test": "exactly one authorized pass",
        },
    }
    policy = {**policy_core, "policy_sha256": canonical_sha256(policy_core)}

    plan_core = {
        "plan_name": "V5_P2_C2A_LEGAL_NOC_DECODER_STAGE_PLAN",
        "plan_version": 1,
        "stages": [
            {
                "stage": "V5_P2_L0_LEGAL_XY_DECODER_CONTRACT",
                "purpose": "freeze semantics, constraints, scoring, tie-breaking, and validation grid",
                "data_access": "none",
            },
            {
                "stage": "V5_P2_L1_XY_ROUTE_LIBRARY_AND_UNIT_TESTS",
                "purpose": "generate and verify all 240 directed 4x4 XY routes",
                "data_access": "none",
            },
            {
                "stage": "V5_P2_L2_EXACT_LEGAL_XY_DECODER",
                "purpose": "implement reduced exhaustive oracle and full exact 4x4 decoder",
                "data_access": "hand-constructed tests only",
            },
            {
                "stage": "V5_P2_L3_HARDWARE_LEGAL_XY_DECODER",
                "purpose": "implement deterministic route-ROM candidate/beam decoder and equivalence tests",
                "data_access": "synthetic tests first",
            },
            {
                "stage": "V5_P2_L4_VALIDATION_DECODER_SELECTION",
                "purpose": "select and freeze decoder using B4 validation outputs only",
                "data_access": "P2 B4 validation cache only",
            },
            {
                "stage": "V5_P2_L5_POSTHOC_B6_DECODER_ANALYSIS",
                "purpose": "apply the frozen decoder to the frozen B6 cache",
                "data_access": "B6 cache only",
                "claim_status": "post-hoc exploratory",
            },
            {
                "stage": "V6_D0_FRESH_HARD_LEGITIMATE_DATASET_GENERATION_PLAN",
                "purpose": "integrate route-pair labels and decoder requirements into V6",
                "data_access": "no V6 blind-test contents",
            },
        ],
        "ordering_constraints": [
            "L0 must complete before decoder implementation.",
            "L1 must complete before L2 or L3.",
            "L2 must exist before L3 equivalence testing.",
            "L4 uses P2 validation only.",
            "L5 may run only after L4 decoder freeze.",
            "V6 checkpoint, decoder, and thresholds freeze before blind-test authorization.",
        ],
    }
    plan = {**plan_core, "plan_sha256": canonical_sha256(plan_core)}

    amendment_core = {
        "stage": STAGE,
        "status": "COMPLETE",
        "decision": (
            "PRESERVE_C2_AND_ADD_VALIDATION_FROZEN_LEGAL_NOC_STRUCTURED_DECODER_STUDY"
        ),
        "relationship_to_c2": {
            "c2_remains_authoritative": True,
            "c2_modified": False,
            "c2_contract_sha256": EXPECTED_C2_CONTRACT_SHA,
            "hard_legitimate_dataset_plan_preserved": True,
            "matched_pair_margin_ablation_preserved": True,
            "decoder_study_is_orthogonal": True,
        },
        "decoder_scope": {
            "position": "after B3 causal Conv1D-count4 heads",
            "topology": "4x4 2D mesh",
            "routing": "deterministic XY",
            "attacker_count": [1, 2, 3, 4],
            "one_victim_assignment_per_source": True,
            "shared_victims_allowed": True,
            "overlapping_routes_allowed": True,
            "overlapping_source_transit_victim_roles_allowed": True,
            "one_source_targeting_multiple_victims": False,
            "normal_hypothesis_emits_empty_structure": True,
        },
        "frozen_decisions": {
            "count_head_change_allowed": False,
            "graph_decision": "joint graph-head plus best legal-structure evidence",
            "graph_head_as_only_gate": False,
            "structure_only_graph_decision": False,
            "learned_graph_message_passing_in_main_decoder": False,
            "exact_reference_decoder_required": True,
            "hardware_decoder_required": True,
            "deterministic_tie_breaking_required": True,
            "route_legality_by_construction": True,
        },
        "implementation_gate": {
            "decoder_implementation_authorized": False,
            "next_required_stage": "V5_P2_L0_LEGAL_XY_DECODER_CONTRACT",
            "l0_must_freeze": [
                "input logit semantics",
                "source/transit/victim/path union semantics",
                "route-pair hypothesis semantics",
                "XY route legality",
                "multi-attacker overlap semantics",
                "count handling",
                "structured scoring equation",
                "normal-versus-attack decision rule",
                "validation parameter grid",
                "deterministic tie-breaking",
                "exact-versus-hardware equivalence targets",
                "P2 versus V6 reporting boundaries",
            ],
        },
        "evaluation_policy_sha256": policy["policy_sha256"],
        "stage_plan_sha256": plan["plan_sha256"],
        "provenance": {
            "c2_report_sha256": sha256_file(c2_report_path),
            "c2_lock_sha256": sha256_file(c2_lock_path),
            "c2_complete_sha256": sha256_file(c2_complete_path),
        },
        "security_boundary": {
            "c2_modified": False,
            "p2_prediction_cache_accessed": False,
            "p2_checkpoint_loaded": False,
            "p2_model_loaded": False,
            "p2_dataset_root_accessed": False,
            "p2_test_directory_enumerated": False,
            "p2_test_tensor_contents_accessed": False,
            "p2_test_inference_rerun": False,
            "decoder_implementation_performed": False,
        },
        "failures": failures,
        "warnings": warnings,
        "next_stage": "V5_P2_L0_LEGAL_XY_DECODER_CONTRACT",
    }
    amendment = {
        **amendment_core,
        "amendment_sha256": canonical_sha256(amendment_core),
    }

    amendment_path = output_dir / "V5_P2_C2A_LEGAL_NOC_DECODER_AMENDMENT.json"
    markdown_path = output_dir / "V5_P2_C2A_LEGAL_NOC_DECODER_AMENDMENT.md"
    policy_path = output_dir / "V5_P2_C2A_P2_VS_V6_EVALUATION_POLICY.json"
    plan_path = output_dir / "V5_P2_C2A_DECODER_STAGE_PLAN.json"

    write_json(amendment_path, amendment)
    write_json(policy_path, policy)
    write_json(plan_path, plan)

    markdown = """# V5 P2-C2A Legal NoC Decoder Amendment

## Status

C2 remains authoritative and unchanged. This append-only amendment adds an orthogonal structured-decoder study.

## Frozen pipeline

`B3 causal Conv1D-count4 -> raw graph/count/source/transit/victim/path scores -> Legal NoC structured decoder -> coherent attack hypothesis`

## Initial scope

- 4x4 2D mesh with deterministic XY routing.
- K1-K4 sources.
- One victim assignment per source.
- Shared victims and overlapping legal routes are allowed.
- Source/transit/victim roles may overlap.
- One source targeting multiple victims is deferred.
- The four-class count head remains unchanged.

## Graph decision

The graph head is not discarded and is not the only hard gate. The final decision must compare the normal hypothesis against the best legal structured attack hypothesis using graph and node/route evidence jointly.

## Evaluation boundary

- P2 B6 remains the official blind result.
- P2 decoder selection uses B4 validation outputs only.
- Frozen-decoder analysis on B6 is post-hoc exploratory only.
- V6 provides the first clean blind-test result for model plus decoder.

## Implementation gate

Decoder implementation is not yet authorized. L0 must first freeze label semantics, hypothesis construction, legality constraints, scoring, count handling, tie-breaking, validation grid, and exact-versus-hardware equivalence targets.

## Next stage

`V5_P2_L0_LEGAL_XY_DECODER_CONTRACT`
"""
    atomic_write(markdown_path, markdown)

    lock = {
        "status": COMPLETE,
        "decision": amendment["decision"],
        "amendment_file_sha256": sha256_file(amendment_path),
        "amendment_sha256": amendment["amendment_sha256"],
        "markdown_sha256": sha256_file(markdown_path),
        "evaluation_policy_file_sha256": sha256_file(policy_path),
        "evaluation_policy_sha256": policy["policy_sha256"],
        "stage_plan_file_sha256": sha256_file(plan_path),
        "stage_plan_sha256": plan["plan_sha256"],
        "c2_report_sha256": sha256_file(c2_report_path),
        "c2_lock_sha256": sha256_file(c2_lock_path),
        "c2_contract_sha256": EXPECTED_C2_CONTRACT_SHA,
        "c2_modified": False,
        "decoder_implementation_authorized": False,
        "p2_prediction_cache_accessed": False,
        "p2_checkpoint_loaded": False,
        "p2_test_directory_enumerated": False,
        "p2_test_tensor_contents_accessed": False,
        "p2_test_inference_rerun": False,
        "next_stage": "V5_P2_L0_LEGAL_XY_DECODER_CONTRACT",
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(
        output_dir / "V5_P2_C2A_LEGAL_NOC_DECODER_AMENDMENT_LOCK.json",
        lock,
    )
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-C2A LEGAL NOC DECODER AMENDMENT =====")
    print("status: COMPLETE")
    print("decision:", amendment["decision"])
    print("c2_preserved_unchanged: true")
    print("c2_contract_sha256:", EXPECTED_C2_CONTRACT_SHA)
    print("decoder_position: AFTER_B3_CAUSAL_CONV1D_COUNT4_HEADS")
    print("topology: 4x4_2D_MESH")
    print("routing: DETERMINISTIC_XY")
    print("one_victim_assignment_per_source: true")
    print("shared_victims_allowed: true")
    print("overlapping_routes_allowed: true")
    print("count_head_change_allowed: false")
    print("graph_decision: JOINT_GRAPH_PLUS_LEGAL_STRUCTURE")
    print("exact_reference_decoder_required: true")
    print("hardware_decoder_required: true")
    print("p2_official_result_changed: false")
    print("p2_decoder_tuning_source: B4_VALIDATION_ONLY")
    print("p2_b6_decoder_claim: POSTHOC_EXPLORATORY_ONLY")
    print("v6_first_clean_blind_decoder_claim: true")
    print("decoder_implementation_authorized: false")
    print("amendment_sha256:", amendment["amendment_sha256"])
    print("evaluation_policy_sha256:", policy["policy_sha256"])
    print("stage_plan_sha256:", plan["plan_sha256"])
    print("p2_prediction_cache_accessed: false")
    print("p2_checkpoint_loaded: false")
    print("p2_test_directory_enumerated: false")
    print("p2_test_tensor_contents_accessed: false")
    print("p2_test_inference_rerun: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("next_stage: V5_P2_L0_LEGAL_XY_DECODER_CONTRACT")
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
