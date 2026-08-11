from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "V5_P3_D5_TRANCHE_A_DECODER_REVIEW_AND_4X4_EXPERT_INTERFACE_FREEZE"
CAMPAIGN_LABEL = "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic"

EXPECTED = {
    "routers": 16,
    "features": 70,
    "window": 32,
    "mask_features": 10,
    "directed_edges": 48,
    "parameters": 60_553,
    "outputs": 69,
    "checkpoint_epoch": 14,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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


def require_single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one match for {pattern} under {directory}, "
            f"found={matches}"
        )
    return matches[0]


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    interface_path = output_dir / (
        "V5_P3_4X4_DYNAMIC70_EXPERT_INTERFACE_CONTRACT.json"
    )
    decoder_path = output_dir / (
        "V5_P3_TRANCHE_A_DECODER_POLICY_DECISION.json"
    )
    rtl_note_path = output_dir / (
        "V5_P3_D5_RTL_TEAM_DECISION_NOTE.md"
    )
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    r3_dir = repo / "reports/v5/p3_d4_r3_resume_a1_exact_validation_hybrid"
    r3_report_path = r3_dir / (
        "V5_P3_D4_R3_RESUME_A1_EXACT_VALIDATION_WITH_"
        "FROZEN_CERTIFIED_HYBRID_DECODER_REPORT.json"
    )
    r3_lock_path = r3_dir / (
        "V5_P3_D4_R3_RESUME_A1_EXACT_VALIDATION_WITH_"
        "FROZEN_CERTIFIED_HYBRID_DECODER_LOCK.json"
    )
    merged_path = r3_dir / (
        "V5_P3_D4_R3_COMPLETE_A1_HYBRID_VALIDATION_OUTPUTS.npz"
    )

    r2r2_dir = repo / "reports/v5/p3_d4_r2_r2_hybrid_decoder_freeze"
    r2r2_report_path = r2r2_dir / (
        "V5_P3_D4_R2_R2_CANONICAL_FIRST_CERTIFIED_"
        "PRESOLVE_FALSE_FALLBACK_FREEZE_REPORT.json"
    )
    r2r2_lock_path = r2r2_dir / (
        "V5_P3_D4_R2_R2_CANONICAL_FIRST_CERTIFIED_"
        "PRESOLVE_FALSE_FALLBACK_FREEZE_LOCK.json"
    )

    h0_dir = repo / "artifacts/rtl/v5_p3_h0_current_4x4_dynamic70_rtl_handoff"
    h0_report_path = h0_dir / (
        "V5_P3_H0_CURRENT_4X4_DYNAMIC70_RTL_HANDOFF_"
        "PREQUANTIZATION_REFERENCE_REPORT.json"
    )
    h0_lock_path = h0_dir / (
        "V5_P3_H0_CURRENT_4X4_DYNAMIC70_RTL_HANDOFF_"
        "PREQUANTIZATION_REFERENCE_LOCK.json"
    )
    h0_archive_path = h0_dir / (
        "V5_P3_H0_CURRENT_4X4_DYNAMIC70_RTL_HANDOFF_"
        "PREQUANTIZATION_REFERENCE.zip"
    )

    model_path = repo / (
        "src/models/v6_p0_dynamic70_task_d_full_multitask_count4.py"
    )
    canonical_decoder_path = repo / (
        "src/decoders/v5_legal_xy_exact_decoder_certified.py"
    )
    fallback_decoder_path = repo / (
        "src/decoders/v5_p3_a1_exact_decoder_presolve_false.py"
    )
    hybrid_decoder_path = repo / (
        "src/decoders/v5_p3_a1_exact_decoder_certified_hybrid.py"
    )

    required = [
        r3_report_path,
        r3_lock_path,
        merged_path,
        r2r2_report_path,
        r2r2_lock_path,
        h0_report_path,
        h0_lock_path,
        h0_archive_path,
        model_path,
        canonical_decoder_path,
        fallback_decoder_path,
        hybrid_decoder_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")

    r3 = json.loads(r3_report_path.read_text(encoding="utf-8"))
    r3_lock = json.loads(r3_lock_path.read_text(encoding="utf-8"))
    r2r2 = json.loads(r2r2_report_path.read_text(encoding="utf-8"))
    r2r2_lock = json.loads(r2r2_lock_path.read_text(encoding="utf-8"))
    h0 = json.loads(h0_report_path.read_text(encoding="utf-8"))
    h0_lock = json.loads(h0_lock_path.read_text(encoding="utf-8"))

    if r3.get("status") != "PASS":
        raise RuntimeError("R3 report is not PASS")
    if r3_lock.get("report_sha256") != sha256_file(r3_report_path):
        raise RuntimeError("R3 report/lock mismatch")
    if r3_lock.get("merged_outputs_sha256") != sha256_file(merged_path):
        raise RuntimeError("R3 merged output/lock mismatch")
    if r2r2.get("status") != "PASS":
        raise RuntimeError("R2-R2 report is not PASS")
    if r2r2_lock.get("report_sha256") != sha256_file(r2r2_report_path):
        raise RuntimeError("R2-R2 report/lock mismatch")
    if h0.get("status") != "PASS":
        raise RuntimeError("H0 report is not PASS")
    if h0_lock.get("report_sha256") != sha256_file(h0_report_path):
        raise RuntimeError("H0 report/lock mismatch")
    if h0_lock.get("handoff_archive_sha256") != sha256_file(h0_archive_path):
        raise RuntimeError("H0 archive/lock mismatch")

    metrics = r3["metrics"]
    raw = metrics["Raw"]
    a0 = metrics["A0"]
    a1 = metrics["A1"]
    comparison = metrics["comparison"]

    # Contract checks.
    if int(h0["model"]["parameter_count"]) != EXPECTED["parameters"]:
        raise RuntimeError("H0 parameter count changed")
    if list(h0["model"]["input_shape"]) != [
        EXPECTED["routers"],
        EXPECTED["features"],
        EXPECTED["window"],
    ]:
        raise RuntimeError("H0 input shape changed")
    if list(h0["model"]["mask_shape"]) != [
        EXPECTED["routers"],
        EXPECTED["mask_features"],
    ]:
        raise RuntimeError("H0 mask shape changed")
    if int(h0["model"]["output_scalars"]) != EXPECTED["outputs"]:
        raise RuntimeError("H0 output width changed")
    if int(h0["topology"]["directed_edges"]) != EXPECTED["directed_edges"]:
        raise RuntimeError("H0 topology changed")
    if int(h0["model"]["checkpoint_epoch"]) != EXPECTED["checkpoint_epoch"]:
        raise RuntimeError("H0 checkpoint epoch changed")

    raw_strict = float(raw["strict_exact"])
    a0_strict = float(a0["strict_exact"])
    a1_strict = float(a1["strict_exact"])
    raw_graph_accuracy = float(raw["graph"]["accuracy"])
    a1_graph_accuracy = float(a1["graph"]["accuracy"])
    raw_graph_fpr = float(raw["graph"]["fpr"])
    a1_graph_fpr = float(a1["graph"]["fpr"])

    a1_strict_gain = a1_strict - raw_strict
    a1_graph_accuracy_delta = a1_graph_accuracy - raw_graph_accuracy
    a1_graph_fpr_delta = a1_graph_fpr - raw_graph_fpr

    if a1_strict_gain <= 0.10:
        raise RuntimeError(
            "A1 strict-exact gain is smaller than the review contract"
        )
    if a0_strict >= raw_strict:
        raise RuntimeError(
            "A0 no longer underperforms Raw strict exact; review changed"
        )
    if a1_graph_accuracy_delta >= 0:
        raise RuntimeError(
            "A1 unexpectedly no longer trades graph accuracy for structure"
        )
    if a1_graph_fpr_delta <= 0:
        raise RuntimeError(
            "A1 unexpectedly no longer increases graph FPR"
        )

    interface_contract = {
        "stage": STAGE,
        "status": "FROZEN",
        "classification": (
            "architecture-and-interface freeze for the reusable 4x4 regional "
            "expert; current weights remain Tranche-A preliminary"
        ),
        "model_identity": (
            "Causal Depthwise-Separable Conv1D Temporal Encoder "
            "+ Two-Layer GraphConv, Dynamic70 adaptation"
        ),
        "temporal_frontend_reference": "B3",
        "full_neural_architecture": (
            "B3 temporal frontend + two-layer GraphConv"
        ),
        "input_contract": {
            "x_shape": ["batch", 16, 70, 32],
            "x_layout": "x[batch][router][feature][time]",
            "physical_port_mask_shape": ["batch", 16, 10],
            "feature_representation": "Dynamic70",
            "window_length": 32,
        },
        "graph_contract": {
            "mesh": "4x4",
            "routers": 16,
            "router_numbering": "row-major",
            "directed_edges": 48,
            "topology": "fixed",
        },
        "output_contract": {
            "raw_logit_count": 69,
            "layout": {
                "graph": {"offset": 0, "length": 1},
                "count": {
                    "offset": 1,
                    "length": 4,
                    "classes": [1, 2, 3, 4],
                },
                "source": {"offset": 5, "length": 16},
                "transit": {"offset": 21, "length": 16},
                "victim": {"offset": 37, "length": 16},
                "path": {"offset": 53, "length": 16},
            },
            "hardware_boundary": (
                "raw neural logits before sigmoid, thresholding, or "
                "structured decoding"
            ),
        },
        "parameter_contract": {
            "current_checkpoint_epoch": 14,
            "current_parameter_count": 60_553,
            "current_weights_classification": (
                "Tranche-A preliminary engineering reference"
            ),
            "future_A_plus_B_retraining_must_preserve_interface": True,
            "current_weights_are_final": False,
        },
        "decoder_boundary": {
            "inside_RTL_accelerator": False,
            "raw_graph_logit_is_primary_hardware_detection_signal": True,
            "structured_exact_decoder_is_software_side": True,
        },
        "shared_regional_reuse": {
            "8x8_wrapper_supported": True,
            "regions": 4,
            "same_model_instance_and_shared_weights": True,
            "cross_region_reasoning": False,
        },
        "forbidden_prebaseline_changes": [
            "attention",
            "directional GraphConv replacement",
            "new graph pooling",
            "multiscale graph hierarchy",
            "new localization loss",
            "new output labels",
        ],
        "source_handoff_archive": {
            "path": str(h0_archive_path),
            "sha256": sha256_file(h0_archive_path),
        },
    }
    atomic_json(interface_path, interface_contract)

    decoder_decision = {
        "stage": STAGE,
        "status": "FROZEN_PRELIMINARY_DECISION",
        "classification": CAMPAIGN_LABEL,
        "validation_only": True,
        "test_accessed": False,
        "policies": {
            "Raw_0.5": {
                "role": (
                    "primary current P3 neural baseline and primary hardware "
                    "graph-detection output policy"
                ),
                "graph_accuracy": raw_graph_accuracy,
                "graph_f1": float(raw["graph"]["f1"]),
                "graph_fpr": raw_graph_fpr,
                "strict_exact": raw_strict,
                "retained": True,
            },
            "Frozen_P2_A0_transfer": {
                "role": "diagnostic transfer only",
                "graph_accuracy": float(a0["graph"]["accuracy"]),
                "graph_f1": float(a0["graph"]["f1"]),
                "graph_fpr": float(a0["graph"]["fpr"]),
                "strict_exact": a0_strict,
                "retained_as_primary": False,
                "reason": (
                    "slightly changes graph metrics but reduces strict exact "
                    "relative to Raw"
                ),
            },
            "Frozen_P2_A1_exact_transfer_with_P3_hybrid_numerics": {
                "role": (
                    "optional software-side structured localization and "
                    "consistency decoder"
                ),
                "graph_accuracy": a1_graph_accuracy,
                "graph_f1": float(a1["graph"]["f1"]),
                "graph_fpr": a1_graph_fpr,
                "strict_exact": a1_strict,
                "count_active_macro_f1": float(
                    a1["count_active_macro_f1"]
                ),
                "source_exact_active": float(
                    a1["roles"]["source"]["exact_active"]
                ),
                "transit_exact_active": float(
                    a1["roles"]["transit"]["exact_active"]
                ),
                "victim_exact_active": float(
                    a1["roles"]["victim"]["exact_active"]
                ),
                "path_exact_active": float(
                    a1["roles"]["path"]["exact_active"]
                ),
                "retained": True,
                "replaces_primary_graph_detector": False,
                "reason": (
                    "large strict-structure gain, but lower graph accuracy and "
                    "higher graph FPR than Raw"
                ),
            },
        },
        "deltas": {
            "A1_minus_Raw_strict_exact": a1_strict_gain,
            "A1_minus_Raw_graph_accuracy": a1_graph_accuracy_delta,
            "A1_minus_Raw_graph_fpr": a1_graph_fpr_delta,
            "A1_minus_A0_strict_exact": a1_strict - a0_strict,
        },
        "frozen_numerical_policy": {
            "canonical": "presolve=True certified exact decoder",
            "fallback": (
                "presolve=False certified exact decoder only after canonical "
                "ExactDecoderError"
            ),
            "fail_closed_if_both_fail": True,
            "certificate_tolerance_relaxed": False,
            "threshold_or_margin_retuned": False,
            "beam_selected": False,
        },
        "decision": {
            "hardware_exports_raw_69_logits": True,
            "hardware_implements_exact_MILP_decoder": False,
            "Raw_0.5_retained_as_current_P3_baseline": True,
            "A0_transfer_rejected_as_primary": True,
            "A1_retained_as_optional_software_structured_decoder": True,
            "A1_rejected_as_primary_graph_detector": True,
            "architecture_change_authorized": False,
            "additional_Tranche_A_seed_authorized": False,
            "sealed_A_test_evaluation_authorized": False,
        },
    }
    atomic_json(decoder_path, decoder_decision)

    rtl_note = f"""# V5-P3 D5 decision for the RTL team

## Frozen neural expert interface

- Input: `x[16][70][32]`
- Physical-port mask: `[16][10]`
- Fixed topology: 4x4 mesh, 48 directed edges
- Current model parameters: 60,553
- Output: 69 raw neural logits
- Current checkpoint: Tranche-A epoch 14
- H0 handoff SHA-256: `{sha256_file(h0_archive_path)}`

## Hardware/software boundary

The accelerator must produce the 69 raw logits. The exact structured MILP
decoder remains software-side and is not part of the synthesizable
Conv1D-GraphConv engine.

The current primary hardware graph-detection reference remains the raw graph
logit with the preliminary 0.5 validation policy. The final A+B model may
require a newly frozen threshold, but the 69-logit interface will not change.

## Decoder review

- Raw strict exact: `{raw_strict:.8f}`
- A1 strict exact: `{a1_strict:.8f}`
- A1 strict gain: `{a1_strict_gain:.8f}`
- Raw graph accuracy: `{raw_graph_accuracy:.8f}`
- A1 graph accuracy: `{a1_graph_accuracy:.8f}`
- Raw graph FPR: `{raw_graph_fpr:.8f}`
- A1 graph FPR: `{a1_graph_fpr:.8f}`

A1 is useful for software-side structured localization because it substantially
improves strict consistency. It does not replace the raw neural graph detector,
because graph accuracy decreases and false-positive rate increases.

## Current implementation boundary

RTL may proceed with memory planning, scheduling, testbench construction,
quantization integration, Conv1D, GraphConv, pooling and output-head datapaths.

The current epoch-14 weights are a preliminary engineering reference. A future
fresh A+B model will preserve the same tensor and output interface.
"""
    atomic_text(rtl_note_path, rtl_note)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "scope": (
            "Review the completed Raw/A0/A1 Tranche-A validation diagnostic, "
            "freeze the decoder role, and ratify the reusable 4x4 Dynamic70 "
            "neural expert interface for RTL and regional-wrapper engineering."
        ),
        "metric_summary": {
            "Raw": {
                "graph_accuracy": raw_graph_accuracy,
                "graph_f1": float(raw["graph"]["f1"]),
                "graph_fpr": raw_graph_fpr,
                "strict_exact": raw_strict,
            },
            "A0": {
                "graph_accuracy": float(a0["graph"]["accuracy"]),
                "graph_f1": float(a0["graph"]["f1"]),
                "graph_fpr": float(a0["graph"]["fpr"]),
                "strict_exact": a0_strict,
            },
            "A1": {
                "graph_accuracy": a1_graph_accuracy,
                "graph_auroc": float(a1["graph_auroc"]),
                "graph_average_precision": float(
                    a1["graph_average_precision"]
                ),
                "graph_f1": float(a1["graph"]["f1"]),
                "graph_fpr": a1_graph_fpr,
                "strict_exact": a1_strict,
                "count_active_macro_f1": float(
                    a1["count_active_macro_f1"]
                ),
                "source_exact_active": float(
                    a1["roles"]["source"]["exact_active"]
                ),
                "transit_exact_active": float(
                    a1["roles"]["transit"]["exact_active"]
                ),
                "victim_exact_active": float(
                    a1["roles"]["victim"]["exact_active"]
                ),
                "path_exact_active": float(
                    a1["roles"]["path"]["exact_active"]
                ),
            },
            "deltas": {
                "A1_minus_Raw_strict_exact": a1_strict_gain,
                "A1_minus_Raw_graph_accuracy": a1_graph_accuracy_delta,
                "A1_minus_Raw_graph_fpr": a1_graph_fpr_delta,
                "A1_minus_A0_strict_exact": a1_strict - a0_strict,
            },
        },
        "review_conclusion": {
            "graph_bottleneck_remains": True,
            "A0_transfer_selected": False,
            "A1_structured_decoder_selected_for_software_optional_use": True,
            "A1_selected_as_primary_graph_detector": False,
            "Raw_graph_logit_retained_for_hardware_detection": True,
            "architecture_change_required_before_B": False,
            "victim_and_path_remain_weaker_than_source_and_transit": True,
        },
        "interface_freeze": {
            "contract_path": str(interface_path),
            "contract_sha256": sha256_file(interface_path),
            "status": "FROZEN",
            "current_weights_final": False,
            "A_plus_B_model_must_preserve_interface": True,
        },
        "decoder_freeze": {
            "decision_path": str(decoder_path),
            "decision_sha256": sha256_file(decoder_path),
            "hybrid_policy_frozen": True,
            "canonical_decoder_modified": False,
            "certificate_tolerance_relaxed": False,
            "threshold_or_margin_retuned": False,
        },
        "RTL": {
            "decision_note_path": str(rtl_note_path),
            "decision_note_sha256": sha256_file(rtl_note_path),
            "H0_archive_path": str(h0_archive_path),
            "H0_archive_sha256": sha256_file(h0_archive_path),
            "quantization_frozen": False,
            "compute_RTL_implemented": False,
        },
        "decision": {
            "D5_complete": True,
            "shared_8x8_wrapper_engineering_authorized": True,
            "H1_quantization_work_authorized": True,
            "Tranche_B_spec_remains_frozen": True,
            "Tranche_B_generation_authorized_by_D5": False,
            "A_test_evaluation_authorized": False,
            "final_model_claim_authorized": False,
            "next_hardware_stage": (
                "V5_P3_H1_DYNAMIC70_QUANTIZATION_AND_ACCUMULATOR_WIDTH_FREEZE"
            ),
            "next_dataset_governance_stage": (
                "resume_existing_A6_Tranche_B_prefrozen_spec review; "
                "generation remains separately governed"
            ),
        },
        "sealed_test": {
            "test_dataset_instantiated": False,
            "test_length_computed": False,
            "test_tensor_loaded": False,
            "evaluation_authorized": False,
        },
        "provenance": {
            "R3_report_sha256": sha256_file(r3_report_path),
            "R3_lock_sha256": sha256_file(r3_lock_path),
            "R3_merged_outputs_sha256": sha256_file(merged_path),
            "R2_R2_report_sha256": sha256_file(r2r2_report_path),
            "R2_R2_lock_sha256": sha256_file(r2r2_lock_path),
            "H0_report_sha256": sha256_file(h0_report_path),
            "H0_lock_sha256": sha256_file(h0_lock_path),
            "model_source_sha256": sha256_file(model_path),
            "canonical_decoder_sha256": sha256_file(
                canonical_decoder_path
            ),
            "fallback_decoder_sha256": sha256_file(
                fallback_decoder_path
            ),
            "hybrid_decoder_sha256": sha256_file(
                hybrid_decoder_path
            ),
            "installed_script_sha256": sha256_file(installed_script),
        },
        "model_retrained": False,
        "threshold_tuning_performed": False,
        "decoder_reexecuted": False,
        "test_tensor_loaded": False,
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(report_path),
        "interface_contract_sha256": sha256_file(interface_path),
        "decoder_decision_sha256": sha256_file(decoder_path),
        "RTL_note_sha256": sha256_file(rtl_note_path),
        "H0_archive_sha256": sha256_file(h0_archive_path),
        "R3_merged_outputs_sha256": sha256_file(merged_path),
        "Raw_strict_exact": raw_strict,
        "A1_strict_exact": a1_strict,
        "A1_minus_Raw_strict_exact": a1_strict_gain,
        "Raw_graph_accuracy": raw_graph_accuracy,
        "A1_graph_accuracy": a1_graph_accuracy,
        "Raw_graph_fpr": raw_graph_fpr,
        "A1_graph_fpr": a1_graph_fpr,
        "interface_frozen": True,
        "current_weights_final": False,
        "test_tensor_loaded": False,
    }
    atomic_json(lock_path, lock)
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign_label={CAMPAIGN_LABEL}")
    print(f"Raw_graph_accuracy={raw_graph_accuracy:.8f}")
    print(f"Raw_graph_fpr={raw_graph_fpr:.8f}")
    print(f"Raw_strict_exact={raw_strict:.8f}")
    print(f"A0_strict_exact={a0_strict:.8f}")
    print(f"A1_graph_accuracy={a1_graph_accuracy:.8f}")
    print(f"A1_graph_fpr={a1_graph_fpr:.8f}")
    print(f"A1_strict_exact={a1_strict:.8f}")
    print(f"A1_minus_Raw_strict_exact={a1_strict_gain:.8f}")
    print(
        "primary_hardware_graph_detection="
        "raw_neural_graph_logit"
    )
    print(
        "software_structured_decoder="
        "frozen_certified_hybrid_A1"
    )
    print("A0_transfer_selected=false")
    print("A1_selected_as_primary_graph_detector=false")
    print("A1_retained_for_optional_structured_localization=true")
    print("interface_input=16x70x32")
    print("interface_mask=16x10")
    print("interface_directed_edges=48")
    print("interface_output_logits=69")
    print("interface_parameter_count=60553")
    print("interface_frozen=true")
    print("current_Tranche_A_weights_final=false")
    print("quantization_frozen=false")
    print("shared_8x8_wrapper_engineering_authorized=true")
    print("H1_quantization_work_authorized=true")
    print("Tranche_B_generation_authorized_by_D5=false")
    print("test_tensor_loaded=false")
    print(
        "next_hardware_stage="
        "V5_P3_H1_DYNAMIC70_QUANTIZATION_AND_ACCUMULATOR_WIDTH_FREEZE"
    )
    print(f"interface_contract={interface_path}")
    print(f"decoder_decision={decoder_path}")
    print(f"RTL_team_note={rtl_note_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
