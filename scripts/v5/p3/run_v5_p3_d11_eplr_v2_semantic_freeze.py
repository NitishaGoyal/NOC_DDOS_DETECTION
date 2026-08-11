from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "V5_P3_D11_EPLR_V2_PRESERVE_OR_PASSTHROUGH_SEMANTIC_FREEZE"
CAMPAIGN_LABEL = (
    "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic "
    "— EPLR-V2 Preserve-or-Passthrough Design Freeze"
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


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    semantic_path = output_dir / (
        "V5_P3_EPLR_V2_PRESERVE_OR_PASSTHROUGH_SEMANTIC_CONTRACT.json"
    )
    output_path = output_dir / (
        "V5_P3_EPLR_V2_SOFTWARE_OUTPUT_CONTRACT.json"
    )
    d12_matrix_path = output_dir / (
        "V5_P3_EPLR_V2_D12_SYNTHETIC_TEST_MATRIX.json"
    )
    future_validation_path = output_dir / (
        "V5_P3_EPLR_V2_FUTURE_INDEPENDENT_VALIDATION_PROTOCOL.json"
    )
    governance_path = output_dir / (
        "V5_P3_EPLR_V2_GOVERNANCE_NOTE.md"
    )
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    d10_dir = (
        repo
        / "reports/v5/p3_d10_eplr_v1_comparison_and_carry_forward"
    )
    d10_report_path = d10_dir / (
        "V5_P3_D10_RAW_A0_A1_EPLR_V1_COMPARISON_"
        "AND_CARRY_FORWARD_DECISION_REPORT.json"
    )
    d10_lock_path = d10_dir / (
        "V5_P3_D10_RAW_A0_A1_EPLR_V1_COMPARISON_"
        "AND_CARRY_FORWARD_DECISION_LOCK.json"
    )
    d10_v2_input_path = d10_dir / (
        "V5_P3_D10_EPLR_V2_PRESERVE_OR_PASSTHROUGH_DESIGN_INPUT.md"
    )
    d10_root_cause_path = d10_dir / (
        "V5_P3_D10_EPLR_V1_STATUS_AND_REPAIR_ROOT_CAUSE_AUDIT.json"
    )

    d8_dir = (
        repo
        / "reports/v5/p3_d8_eplr_v1_implementation_and_train_calibration"
    )
    d8_report_path = d8_dir / (
        "V5_P3_D8_EPLR_V1_DETERMINISTIC_IMPLEMENTATION_"
        "AND_A_TRAIN_ONLY_CALIBRATION_REPORT.json"
    )
    d8_lock_path = d8_dir / (
        "V5_P3_D8_EPLR_V1_DETERMINISTIC_IMPLEMENTATION_"
        "AND_A_TRAIN_ONLY_CALIBRATION_LOCK.json"
    )

    route_library_path = repo / "src/decoders/v5_xy_route_library.py"
    eplr_v1_path = repo / "src/decoders/v5_p3_eplr_v1.py"
    eplr_v1_config_path = (
        repo / "src/decoders/v5_p3_eplr_v1_calibration.json"
    )

    required = [
        d10_report_path,
        d10_lock_path,
        d10_v2_input_path,
        d10_root_cause_path,
        d8_report_path,
        d8_lock_path,
        route_library_path,
        eplr_v1_path,
        eplr_v1_config_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")

    d10 = json.loads(d10_report_path.read_text(encoding="utf-8"))
    d10_lock = json.loads(d10_lock_path.read_text(encoding="utf-8"))
    d10_root = json.loads(
        d10_root_cause_path.read_text(encoding="utf-8")
    )
    d8 = json.loads(d8_report_path.read_text(encoding="utf-8"))
    d8_lock = json.loads(d8_lock_path.read_text(encoding="utf-8"))

    if d10.get("status") != "PASS":
        raise RuntimeError("D10 is not PASS")
    if d10_lock.get("report_sha256") != sha256_file(d10_report_path):
        raise RuntimeError("D10 report/lock mismatch")
    if not d10["scientific_decision"].get(
        "EPLR_V2_preserve_or_passthrough_design_authorized"
    ):
        raise RuntimeError("D10 did not authorize EPLR-V2 design")
    if d10["scientific_decision"].get(
        "EPLR_V2_A_validation_replay_authorized"
    ):
        raise RuntimeError("D10 unexpectedly authorized A-validation replay")
    if d10_lock.get("V2_design_input_sha256") != sha256_file(
        d10_v2_input_path
    ):
        raise RuntimeError("D10 V2 design input/lock mismatch")
    if d10_lock.get("root_cause_audit_sha256") != sha256_file(
        d10_root_cause_path
    ):
        raise RuntimeError("D10 root-cause audit/lock mismatch")

    if d8.get("status") != "PASS":
        raise RuntimeError("D8 is not PASS")
    if d8_lock.get("report_sha256") != sha256_file(d8_report_path):
        raise RuntimeError("D8 report/lock mismatch")
    if d8_lock.get("decoder_sha256") != sha256_file(eplr_v1_path):
        raise RuntimeError("EPLR-V1 source differs from D8 lock")
    if d8_lock.get("config_sha256") != sha256_file(
        eplr_v1_config_path
    ):
        raise RuntimeError("EPLR-V1 calibration differs from D8 lock")

    if not d10_root.get("preserved_source_outputs_equal_Raw"):
        raise RuntimeError("D10 did not verify preserved source identity")
    if not d10_root.get("preserved_victim_outputs_equal_Raw"):
        raise RuntimeError("D10 did not verify preserved victim identity")

    semantic_contract = {
        "stage": STAGE,
        "status": "FROZEN",
        "decoder": {
            "name": "V5-P3 EPLR-V2",
            "expansion": (
                "Endpoint-Preserving Legal-XY Repair, "
                "Preserve-or-Passthrough Revision"
            ),
            "classification": (
                "new validation-informed design; not an evaluated "
                "Tranche-A validation candidate"
            ),
        },
        "hardware_boundary": {
            "inside_RTL": False,
            "input": "existing 69 raw neural logits",
            "accelerator_interface_changed": False,
        },
        "immutable_global_outputs": {
            "graph_prediction": (
                "exactly Raw sigmoid(graph_logit) >= 0.5"
            ),
            "active_count_prediction": (
                "exactly Raw 1 + argmax(count_logits)"
            ),
            "source_bitmap": (
                "exactly Raw sigmoid(source_logits) >= 0.5"
            ),
            "victim_bitmap": (
                "exactly Raw sigmoid(victim_logits) >= 0.5"
            ),
            "decoder_may_change_graph": False,
            "decoder_may_change_count": False,
            "decoder_may_change_source": False,
            "decoder_may_change_victim": False,
        },
        "inactive_policy": {
            "effective_count": 0,
            "selected_output_source": "all zero",
            "selected_output_transit": "all zero",
            "selected_output_victim": "all zero",
            "selected_output_path": "all zero",
            "status": "INACTIVE_RAW",
        },
        "active_endpoint_feasibility": {
            "raw_source_mask": "threshold 0.5",
            "raw_victim_mask": "threshold 0.5",
            "exactly_K_legal_routes_required": True,
            "attacker_sources_unique": True,
            "shared_victims_allowed": True,
            "global_role_disjointness": False,
            "endpoint_on_one_route_may_be_transit_on_another": True,
            "path_semantics": "union router bitmap",
        },
        "feasible_endpoint_policy": {
            "source_output": "Raw source bitmap",
            "victim_output": "Raw victim bitmap",
            "transit_output": (
                "route-derived transit union from best exact-preserving "
                "legal route tuple"
            ),
            "path_output": (
                "route-derived path union from best exact-preserving "
                "legal route tuple"
            ),
            "route_selection_weights": (
                "reuse the D8 A-train-only frozen transit=1.0, path=1.0 "
                "calibration without modification"
            ),
            "low_support_thresholds": (
                "reuse the D8 A-train-only frozen per-K thresholds "
                "without modification"
            ),
            "status": [
                "RAW_ENDPOINTS_LEGAL",
                "RAW_ENDPOINTS_LEGAL_LOW_ROUTE_SUPPORT",
            ],
            "route_certified": True,
        },
        "infeasible_endpoint_policy": {
            "endpoint_repair_solver_used": False,
            "source_output": "Raw source bitmap passthrough",
            "transit_output": "Raw transit bitmap passthrough",
            "victim_output": "Raw victim bitmap passthrough",
            "path_output": "Raw path bitmap passthrough",
            "selected_route_ids": "empty",
            "route_certified": False,
            "status": "RAW_ENDPOINTS_UNRESOLVED_PASSTHROUGH",
            "interpretation": (
                "Raw prediction is returned unchanged, but no legal-route "
                "certificate is claimed."
            ),
        },
        "removed_V1_behaviors": {
            "endpoint_repair_MILP": False,
            "endpoint_reassignment": False,
            "zero_masks_on_no_solution": False,
            "old_A1_fallback": False,
        },
        "failure_policy": {
            "software_exception": "fail closed with explicit error",
            "infeasible_raw_endpoints": (
                "passthrough with unresolved non-certified status"
            ),
            "invent_legal_explanation": False,
        },
    }
    atomic_json(semantic_path, semantic_contract)

    output_contract = {
        "stage": STAGE,
        "status": "FROZEN",
        "per_sample_outputs": {
            "raw_graph_probability": "float64",
            "raw_graph_prediction": "uint8",
            "raw_attacker_count_candidate": "int 1..4",
            "effective_attacker_count": "int 0..4",
            "raw_source_bitmap": "uint16",
            "raw_transit_bitmap": "uint16",
            "raw_victim_bitmap": "uint16",
            "raw_path_bitmap": "uint16",
            "selected_source_bitmap": "uint16",
            "selected_transit_bitmap": "uint16",
            "selected_victim_bitmap": "uint16",
            "selected_path_bitmap": "uint16",
            "selected_route_ids": "ordered tuple; empty when unresolved",
            "route_consistency_score": (
                "float64 for certified legal route; null/NaN when unresolved"
            ),
            "route_certified": "bool",
            "endpoint_preserved": "bool",
            "decoder_status": "enum",
        },
        "invariants": {
            "selected_source_equals_Raw_for_all_active_rows": True,
            "selected_victim_equals_Raw_for_all_active_rows": True,
            "selected_graph_equals_Raw_for_all_rows": True,
            "selected_count_candidate_equals_Raw_for_all_rows": True,
            "unresolved_passthrough_all_role_masks_equal_Raw": True,
        },
        "statuses": [
            "INACTIVE_RAW",
            "RAW_ENDPOINTS_LEGAL",
            "RAW_ENDPOINTS_LEGAL_LOW_ROUTE_SUPPORT",
            "RAW_ENDPOINTS_UNRESOLVED_PASSTHROUGH",
        ],
    }
    atomic_json(output_path, output_contract)

    d12_matrix = {
        "stage": (
            "V5_P3_D12_EPLR_V2_DETERMINISTIC_IMPLEMENTATION_"
            "AND_SYNTHETIC_TESTS"
        ),
        "status": "PREFROZEN_TEST_MATRIX",
        "tests": [
            {
                "id": "D12-T01",
                "name": "inactive_raw",
                "requirements": [
                    "effective count zero",
                    "all selected role masks zero",
                    "status INACTIVE_RAW",
                ],
            },
            {
                "id": "D12-T02",
                "name": "K1_exact_endpoint_preservation",
                "requirements": [
                    "source and victim exactly Raw",
                    "route certified",
                    "route-derived transit/path",
                ],
            },
            {
                "id": "D12-T03",
                "name": "K2_K4_shared_victim_and_union_semantics",
                "requirements": [
                    "exactly K routes",
                    "shared victims legal",
                    "union masks correct",
                ],
            },
            {
                "id": "D12-T04",
                "name": "cross_route_role_overlap",
                "requirements": [
                    "no global endpoint/transit disjointness",
                    "route union remains valid",
                ],
            },
            {
                "id": "D12-T05",
                "name": "infeasible_endpoint_passthrough",
                "requirements": [
                    "no repair solver invocation",
                    "all four selected role masks equal Raw",
                    "route IDs empty",
                    "route_certified false",
                    "status unresolved passthrough",
                ],
            },
            {
                "id": "D12-T06",
                "name": "global_graph_count_source_victim_immutability",
                "requirements": [
                    "randomized property tests",
                    "bit-identical Raw invariants",
                ],
            },
            {
                "id": "D12-T07",
                "name": "D8_calibration_provenance",
                "requirements": [
                    "transit weight 1.0",
                    "path weight 1.0",
                    "per-K thresholds SHA-locked",
                    "no recalibration",
                ],
            },
            {
                "id": "D12-T08",
                "name": "deterministic_repeat",
                "requirements": [
                    "same logits yield bit-identical outputs",
                    "same route-ID tuple",
                ],
            },
            {
                "id": "D12-T09",
                "name": "no_validation_or_test_access",
                "requirements": [
                    "implementation uses synthetic vectors only",
                    "no D1/D9 output loading",
                    "no A-test loading",
                ],
            },
        ],
        "A_validation_replay": False,
        "A_test_access": False,
    }
    atomic_json(d12_matrix_path, d12_matrix)

    future_validation_protocol = {
        "stage": STAGE,
        "status": "PREFROZEN_FOR_FUTURE_INDEPENDENT_BOUNDARY",
        "eligible_confirmation_boundaries": [
            "frozen Tranche-B validation",
            "future fresh A+B validation under the final protocol",
        ],
        "ineligible_confirmation_boundaries": [
            "Tranche-A validation reused after D9/D10",
            "A-test",
            "closed V5-P2 test",
        ],
        "comparison": "Raw versus EPLR-V2 on identical neural logits",
        "hard_invariants": [
            "graph predictions bit-identical to Raw",
            "count candidates bit-identical to Raw",
            "source masks bit-identical to Raw",
            "victim masks bit-identical to Raw",
        ],
        "effect_criteria": {
            "transit_exact_gain": (
                "pair-clustered 95% confidence interval lower bound > 0"
            ),
            "strict_exact_gain": (
                "pair-clustered 95% confidence interval lower bound > 0"
            ),
            "path_exact_noninferiority": (
                "pair-clustered 95% confidence interval lower bound >= -0.01"
            ),
        },
        "operational_criteria": {
            "all_rows_explicitly_classified": True,
            "unresolved_rows_never_claim_route_certificate": True,
            "deterministic_outputs": True,
        },
        "selection_rule": (
            "Carry forward only if every hard invariant and every effect/"
            "operational criterion passes on the independent boundary."
        ),
        "no_post_result_retuning": True,
    }
    atomic_json(future_validation_path, future_validation_protocol)

    governance_note = f"""# V5-P3 EPLR-V2 governance

## Classification

**{CAMPAIGN_LABEL}**

EPLR-V2 is a new design informed by the completed D9/D10 Tranche-A validation
analysis. It is not an evaluated Tranche-A candidate.

## Frozen decision

EPLR-V2 removes endpoint repair entirely.

When Raw source/victim masks admit exactly K legal routes, EPLR-V2 may replace
only transit/path with route-derived union masks. Source and victim remain Raw.

When Raw endpoints are not legally explainable, EPLR-V2 passes through every
Raw role mask unchanged and reports:

`RAW_ENDPOINTS_UNRESOLVED_PASSTHROUGH`

It does not claim a route certificate for that row.

## Calibration

The D8 A-train-only calibration is reused exactly:

- transit weight: 1.0
- path weight: 1.0
- same per-K low-support thresholds
- no recalibration

## Validation boundary

No second EPLR replay on Tranche-A validation is authorized.

Future confirmation must use an independent boundary:
- frozen Tranche-B validation; or
- future fresh A+B validation under the final protocol.

A-test remains sealed.

## Parallel work

H1 quantization, 8x8 wrapper engineering and Tranche-B work remain unaffected.
"""
    atomic_text(governance_path, governance_note)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "scope": (
            "Freeze EPLR-V2 preserve-or-passthrough semantics, remove endpoint "
            "repair, preserve the frozen hardware boundary, and define the "
            "independent future-validation protocol."
        ),
        "frozen_artifacts": {
            "semantic_contract": {
                "path": str(semantic_path),
                "sha256": sha256_file(semantic_path),
            },
            "output_contract": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
            },
            "D12_test_matrix": {
                "path": str(d12_matrix_path),
                "sha256": sha256_file(d12_matrix_path),
            },
            "future_validation_protocol": {
                "path": str(future_validation_path),
                "sha256": sha256_file(future_validation_path),
            },
            "governance_note": {
                "path": str(governance_path),
                "sha256": sha256_file(governance_path),
            },
        },
        "design_decision": {
            "graph_immutable": True,
            "count_immutable": True,
            "source_immutable": True,
            "victim_immutable": True,
            "endpoint_repair_removed": True,
            "infeasible_endpoint_action": "Raw role-mask passthrough",
            "route_certificate_claimed_when_unresolved": False,
            "D8_train_calibration_reused_without_change": True,
            "inside_RTL": False,
        },
        "decision": {
            "D12_authorized": True,
            "A_validation_replay_authorized": False,
            "A_test_evaluation_authorized": False,
            "independent_future_validation_required": True,
            "H1_parallel_work_unaffected": True,
            "Tranche_B_parallel_work_unaffected": True,
            "next_stage": (
                "V5_P3_D12_EPLR_V2_DETERMINISTIC_IMPLEMENTATION_"
                "AND_SYNTHETIC_TESTS"
            ),
        },
        "provenance": {
            "D10_report_sha256": sha256_file(d10_report_path),
            "D10_lock_sha256": sha256_file(d10_lock_path),
            "D10_root_cause_sha256": sha256_file(d10_root_cause_path),
            "D10_V2_design_input_sha256": sha256_file(
                d10_v2_input_path
            ),
            "D8_report_sha256": sha256_file(d8_report_path),
            "D8_lock_sha256": sha256_file(d8_lock_path),
            "route_library_sha256": sha256_file(route_library_path),
            "EPLR_V1_source_sha256": sha256_file(eplr_v1_path),
            "D8_calibration_sha256": sha256_file(
                eplr_v1_config_path
            ),
            "installed_script_sha256": sha256_file(installed_script),
        },
        "data_access": {
            "model_loaded": False,
            "training_data_loaded": False,
            "A_validation_data_loaded": False,
            "A_test_data_loaded": False,
        },
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(report_path),
        "semantic_contract_sha256": sha256_file(semantic_path),
        "output_contract_sha256": sha256_file(output_path),
        "D12_test_matrix_sha256": sha256_file(d12_matrix_path),
        "future_validation_protocol_sha256": sha256_file(
            future_validation_path
        ),
        "governance_note_sha256": sha256_file(governance_path),
        "D8_calibration_sha256": sha256_file(eplr_v1_config_path),
        "A_validation_replay_authorized": False,
        "A_test_data_loaded": False,
    }
    atomic_json(lock_path, lock)
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign_label={CAMPAIGN_LABEL}")
    print("graph_immutable=true")
    print("count_immutable=true")
    print("source_immutable=true")
    print("victim_immutable=true")
    print("endpoint_repair_removed=true")
    print("unresolved_action=Raw_role_mask_passthrough")
    print("unresolved_route_certificate=false")
    print("D8_train_calibration_reused_without_change=true")
    print("decoder_inside_RTL=false")
    print("A_validation_replay_authorized=false")
    print("A_test_data_loaded=false")
    print("independent_future_validation_required=true")
    print("D12_authorized=true")
    print(
        "next_stage="
        "V5_P3_D12_EPLR_V2_DETERMINISTIC_IMPLEMENTATION_"
        "AND_SYNTHETIC_TESTS"
    )
    print(f"semantic_contract={semantic_path}")
    print(f"output_contract={output_path}")
    print(f"D12_test_matrix={d12_matrix_path}")
    print(f"future_validation_protocol={future_validation_path}")
    print(f"governance_note={governance_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
