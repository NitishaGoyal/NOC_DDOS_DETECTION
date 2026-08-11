#!/usr/bin/env python3
"""
V4-A4a-1C
Full-Training Runtime and Selection Contract Freeze

This stage consumes the completed A4a-0S, A4a-1A, A4a-1B and A4a-1B2
artifacts and freezes the decoder-only full-training protocol.

No model training, checkpoint selection, threshold search, validation
optimization or development-test access occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


EXPECTED_TRAINER_SHA = (
    "e5f9a48e9f7ca873c351743df92e3bb0"
    "0d00a893333434c7fd0e04734e010525"
)
EXPECTED_MODEL_SOURCE_SHA = (
    "5ebcf80385b95f329faf935cad8039b64"
    "afd79f464903308eb07e674d78a1704"
)
EXPECTED_BASE_CHECKPOINT_SHA = (
    "f61c1add6c057f7f53dd34fb1f9f4e95"
    "b01cefd5053e0e42bc100c76dac7f923"
)
EXPECTED_SLOT_MODULE_SHA = (
    "97b56aabb3239dfc7543c241665cacf99"
    "8eb6b1a1f5f67f931cdeb70e75c111c"
)
EXPECTED_SMOKE_SCRIPT_SHA = (
    "a1c93ed64198c0bd946e64da116b3438"
    "012d742c0c5fafc092e8517b4761a35d"
)
EXPECTED_LABEL_AUDIT_SCRIPT_SHA = (
    "4dd81d0aea6ab529765cdcb11056ef33"
    "2e522fe2f9c5ddbd7cc2e025977a469c"
)
EXPECTED_TOTAL_PARAMETERS = 3189
EXPECTED_TRAINABLE_PARAMETERS = 1076


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def markdown(contract: dict[str, Any]) -> str:
    runtime = contract["runtime_contract"]
    losses = contract["loss_contract"]
    selection = contract["checkpoint_and_validation_contract"]

    lines = [
        "# V4-A4a-1C Full-Training Contract",
        "",
        "## Model",
        "",
        "- Primary model: A4a-Slot-Full",
        "- Frozen A3 temporal/local/graph encoder and graph head",
        "- Trainable slot decoder parameters: 1,076",
        "- Total deployed parameters: 3,189",
        "",
        "## Runtime",
        "",
    ]

    for key, value in runtime.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Losses", ""])
    for key, value in losses.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Checkpoint and validation", ""])
    for key, value in selection.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Ablation order", ""])
    for item in contract["ablation_sequence"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Prohibitions", ""])
    for item in contract["prohibitions"]:
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--slot-module", type=Path, required=True)
    parser.add_argument("--smoke-script", type=Path, required=True)
    parser.add_argument("--label-audit-script", type=Path, required=True)
    parser.add_argument("--a4a-0s-dir", type=Path, required=True)
    parser.add_argument("--a4a-1a-dir", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--label-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    required = [
        args.trainer,
        args.model_source,
        args.base_checkpoint,
        args.slot_module,
        args.smoke_script,
        args.label_audit_script,
        args.a4a_0s_dir / "V4_A4A_0S_SLOT_CONTRACT_AUDIT_PASS",
        args.a4a_0s_dir / "A4A_SLOT_D0_FROZEN_CONTRACT.json",
        args.a4a_0s_dir / "A4A_0S_LOCK.json",
        args.a4a_1a_dir / "V4_A4A_1A_SLOT_IMPLEMENTATION_PASS",
        args.a4a_1a_dir / "implementation_audit.json",
        args.a4a_1a_dir / "A4A_1A_LOCK.json",
        args.smoke_dir / "V4_A4A_1B_TWO_EPOCH_SMOKE_PASS",
        args.smoke_dir / "smoke_report.json",
        args.smoke_dir / "A4A_1B_LOCK.json",
        args.label_audit_dir
        / "V4_A4A_1B2_LABEL_COVERAGE_AUDIT_PASS",
        args.label_audit_dir / "label_coverage_audit.json",
        args.label_audit_dir / "A4A_1B2_LOCK.json",
    ]

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("A4a-1C FAIL: missing required paths", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(
            f"A4a-1C FAIL: output exists: {args.output_dir}",
            file=sys.stderr,
        )
        return 2

    args.output_dir.mkdir(parents=True)

    hashes = {
        "trainer_sha256": sha256_file(args.trainer),
        "model_source_sha256": sha256_file(args.model_source),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "slot_module_sha256": sha256_file(args.slot_module),
        "smoke_script_sha256": sha256_file(args.smoke_script),
        "label_audit_script_sha256": sha256_file(
            args.label_audit_script
        ),
    }
    expected = {
        "trainer_sha256": EXPECTED_TRAINER_SHA,
        "model_source_sha256": EXPECTED_MODEL_SOURCE_SHA,
        "base_checkpoint_sha256": EXPECTED_BASE_CHECKPOINT_SHA,
        "slot_module_sha256": EXPECTED_SLOT_MODULE_SHA,
        "smoke_script_sha256": EXPECTED_SMOKE_SCRIPT_SHA,
        "label_audit_script_sha256": EXPECTED_LABEL_AUDIT_SCRIPT_SHA,
    }

    failures = [
        f"{key} mismatch: {hashes[key]} != {expected_value}"
        for key, expected_value in expected.items()
        if hashes[key] != expected_value
    ]

    a4a_0s = load_json(
        args.a4a_0s_dir / "A4A_SLOT_D0_FROZEN_CONTRACT.json"
    )
    a4a_0s_lock = load_json(args.a4a_0s_dir / "A4A_0S_LOCK.json")
    a4a_1a = load_json(args.a4a_1a_dir / "implementation_audit.json")
    a4a_1a_lock = load_json(args.a4a_1a_dir / "A4A_1A_LOCK.json")
    smoke = load_json(args.smoke_dir / "smoke_report.json")
    smoke_lock = load_json(args.smoke_dir / "A4A_1B_LOCK.json")
    coverage = load_json(
        args.label_audit_dir / "label_coverage_audit.json"
    )
    coverage_lock = load_json(
        args.label_audit_dir / "A4A_1B2_LOCK.json"
    )

    if a4a_0s.get("status") != "PASS":
        failures.append("A4a-0S contract is not PASS")
    if a4a_0s_lock.get("status") != (
        "A4A_0S_SLOT_CONTRACT_AUDIT_COMPLETE"
    ):
        failures.append("A4a-0S lock is incomplete")

    if a4a_1a.get("status") != "PASS":
        failures.append("A4a-1A report is not PASS")
    if a4a_1a_lock.get("status") != (
        "A4A_1A_SLOT_IMPLEMENTATION_COMPLETE"
    ):
        failures.append("A4a-1A lock is incomplete")

    if smoke.get("status") != "PASS":
        failures.append("A4a-1B smoke is not PASS")
    if smoke_lock.get("status") != (
        "A4A_1B_TWO_EPOCH_SMOKE_COMPLETE"
    ):
        failures.append("A4a-1B lock is incomplete")

    if coverage.get("status") != "PASS":
        failures.append("A4a-1B2 coverage audit is not PASS")
    if coverage_lock.get("status") != (
        "A4A_1B2_LABEL_COVERAGE_AUDIT_COMPLETE"
    ):
        failures.append("A4a-1B2 lock is incomplete")
    if coverage.get("coverage_integrity_pass") is not True:
        failures.append("A4a-1B2 coverage integrity did not pass")

    for stage_name, payload in (
        ("A4a-0S", a4a_0s),
        ("A4a-1A", a4a_1a),
        ("A4a-1B", smoke),
        ("A4a-1B2", coverage),
    ):
        if payload.get("development_test_accessed") is not False:
            failures.append(
                f"{stage_name} does not affirm no development-test access"
            )

    integrity = smoke.get("integrity_checks", {})
    required_smoke_checks = (
        "vectorized_matching_crosscheck_pass",
        "checkpoint_reload_pass",
        "all_losses_finite",
        "all_gradients_finite",
        "gradient_clipping_exercised",
        "exact_unique_decode_exercised",
        "greedy_decode_compared_on_validation",
        "parameter_count_pass",
    )
    for key in required_smoke_checks:
        if integrity.get(key) is not True:
            failures.append(f"smoke integrity check failed: {key}")

    frozen_encoder = integrity.get("frozen_encoder_unchanged", {})
    if frozen_encoder.get("unchanged") is not True:
        failures.append("frozen encoder changed during smoke")

    parameters = smoke.get("parameter_count", {})
    if parameters.get("total") != EXPECTED_TOTAL_PARAMETERS:
        failures.append("unexpected deployed parameter count")
    if parameters.get("trainable") != EXPECTED_TRAINABLE_PARAMETERS:
        failures.append("unexpected trainable parameter count")

    null_weight = smoke.get("null_weight", {})
    applied_null_weight = null_weight.get("applied_null_weight")
    if not finite_number(applied_null_weight):
        failures.append("invalid frozen NULL weight")
    elif not 0.05 <= float(applied_null_weight) <= 1.0:
        failures.append("frozen NULL weight is outside [0.05,1.0]")

    count_diagnostics = coverage.get(
        "per_true_count_diagnostics",
        {},
    )
    for count in range(5):
        row = count_diagnostics.get(str(count))
        if not isinstance(row, dict):
            failures.append(
                f"missing label-coverage diagnostics for count {count}"
            )
        elif int(row.get("samples", 0)) <= 0:
            failures.append(
                f"zero label-coverage samples for count {count}"
            )

    matching_by_count = coverage.get(
        "matching_loss_by_true_count",
        {},
    )
    for count in range(5):
        row = matching_by_count.get(str(count), {})
        if not finite_number(row.get("mean")):
            failures.append(
                f"non-finite matching loss for count {count}"
            )

    if failures:
        payload = {
            "status": "FAIL",
            "failures": failures,
            "training_performed": False,
            "development_test_accessed": False,
        }
        (args.output_dir / "contract_freeze_failure.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (
            args.output_dir / "V4_A4A_1C_FULL_TRAINING_CONTRACT_FAIL"
        ).write_text(
            "V4_A4A_1C_FULL_TRAINING_CONTRACT_FAIL\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1

    contract = {
        "status": "PASS",
        "designation": (
            "V4-A4a-1C Decoder-Only Full-Training Runtime Contract"
        ),
        "primary_variant": "A4a-Slot-Full",
        "scientific_role": (
            "Recover part of the A3.12 decoder/cardinality gap while "
            "preserving the frozen A3 representation and graph gate."
        ),
        "model_contract": {
            "frozen_modules": [
                "temporal_conv",
                "local_projection",
                "gcn",
                "graph_head",
            ],
            "trainable_modules": [
                "router_key_projection",
                "global_context_projection",
                "slot_identity_embeddings",
                "slot_null_head",
            ],
            "total_parameters": EXPECTED_TOTAL_PARAMETERS,
            "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
            "slot_count": 4,
            "slot_classes": 17,
            "slot_class_semantics": "routers 0-15 plus NULL",
            "reference_decode": "exact unique constrained assignment",
            "hardware_candidate_decode": "greedy unique assignment",
        },
        "runtime_contract": {
            "seed": 7,
            "maximum_epochs": 75,
            "early_stopping_patience_epochs": 10,
            "minimum_delta": 0.0001,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 5.0,
            "batch_size_primary": 512,
            "batch_size_fallback": 256,
            "fallback_requires_authorized_resource_probe_failure": True,
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "automatic_mixed_precision": False,
            "full_validation_every_epochs": 3,
            "training_split_only_for_optimizer_steps": True,
            "validation_split_only_for_checkpointing_and_selection": True,
            "development_test_loader_constructed": False,
        },
        "loss_contract": {
            "permutation_invariant_matching_weight": 1.0,
            "derived_union_membership_bce_weight": 0.5,
            "derived_cardinality_cross_entropy_weight": 0.5,
            "duplicate_slot_penalty_weight": 0.1,
            "null_target_weight": float(applied_null_weight),
            "null_target_weight_source": (
                "A4a-1B training-split prevalence only"
            ),
            "null_target_weight_report_sha256": (
                smoke_lock.get("smoke_report_sha256")
            ),
            "matching_implementation": (
                "vectorized exact enumeration, cross-checked against "
                "the audited reference implementation"
            ),
        },
        "checkpoint_and_validation_contract": {
            "training_early_stop_monitor": (
                "minimum full-validation total slot loss"
            ),
            "save_every_full_validation_checkpoint": True,
            "primary_model_selection_occurs_after_training": True,
            "primary_model_selection_source": "validation only",
            "selection_hard_gates": {
                "attacker_set_precision_min": 0.98,
                "candidate_coverage_min": 0.90,
                "persistent_normal_run_false_isolation_fraction": 0.0,
                "parameter_budget_pass": True,
                "latency_budget_pass": True,
            },
            "selection_after_hard_gates": (
                "maximize stable exact localization, then attack exact "
                "localization, then minimize normal window false isolation"
            ),
            "minimum_meaningful_validation_exact_gain_percentage_points": 2.0,
            "frozen_development_test_transfers_allowed": 1,
            "post_transfer_retuning_allowed": False,
            "run_level_paired_bootstrap_required": True,
        },
        "resource_probe_contract": {
            "required_before_full_training": True,
            "primary_batch_size_to_probe": 512,
            "probe_batches": 2,
            "required_checks": [
                "finite forward loss",
                "finite backward gradients",
                "gradient clipping",
                "frozen encoder unchanged",
                "GPU memory recorded",
                "checkpoint serialization",
            ],
            "fallback_batch_size": 256,
            "fallback_trigger": (
                "documented CUDA OOM or host-memory/loader failure only"
            ),
        },
        "ablation_sequence": [
            "A4a-Slot: matching loss only",
            (
                "A4a-Slot-Full: matching + derived membership + "
                "derived count + duplicate penalty"
            ),
            (
                "A4a-TopK control: retrained membership/count decoder "
                "with count-controlled top-k"
            ),
            (
                "A4a-Slot-Full+DirectionalProxy only if validation shows "
                "the non-proxy slot model is useful"
            ),
        ],
        "kill_criteria": a4a_0s["kill_criteria"],
        "promotion_criteria": a4a_0s[
            "frozen_transfer_promotion_criteria"
        ],
        "paper_claim_boundary": a4a_0s["paper_claim_boundary"],
        "hard_normal_claim_boundary": a4a_0s[
            "hard_normal_claim_boundary"
        ],
        "smoke_evidence": {
            "train_loss_epoch1": smoke["smoke_observations"][
                "train_loss_epoch1"
            ],
            "train_loss_epoch2": smoke["smoke_observations"][
                "train_loss_epoch2"
            ],
            "train_loss_decreased": smoke["smoke_observations"][
                "train_loss_decreased"
            ],
            "gpu_peak_allocated_mib": max(
                float(row["gpu_peak_allocated_mib"])
                for row in smoke["history"]
            ),
            "frozen_encoder_unchanged": True,
            "checkpoint_reload_pass": True,
        },
        "label_coverage_evidence": {
            "counts_exercised": [0, 1, 2, 3, 4],
            "selected_samples": coverage["selected_sample_count"],
            "coverage_integrity_pass": True,
            "metrics_are_diagnostic_only": True,
        },
        "prohibitions": [
            "No encoder or graph-head fine-tuning in A4a decoder-only training.",
            "No development-test loader construction during training.",
            "No threshold search during optimizer training.",
            "No test-specific checkpoint, loss-weight or persistence selection.",
            "No smoke checkpoint may be promoted as the final model.",
            "No naive duplicate removal as the reference decoder.",
            "No directional proxy in the primary model.",
            "No autonomous-isolation claim unless persistent-normal safety gates pass.",
        ],
        "provenance": {
            **hashes,
            "a4a_0s_contract_sha256": sha256_file(
                args.a4a_0s_dir / "A4A_SLOT_D0_FROZEN_CONTRACT.json"
            ),
            "a4a_1a_audit_sha256": sha256_file(
                args.a4a_1a_dir / "implementation_audit.json"
            ),
            "smoke_report_sha256": sha256_file(
                args.smoke_dir / "smoke_report.json"
            ),
            "label_coverage_audit_sha256": sha256_file(
                args.label_audit_dir / "label_coverage_audit.json"
            ),
        },
        "training_performed": False,
        "validation_selection_performed": False,
        "threshold_search_performed": False,
        "development_test_accessed": False,
        "ready_for_a4a_1d_batch512_resource_probe": True,
    }

    contract_path = (
        args.output_dir / "A4A_1C_FULL_TRAINING_CONTRACT.json"
    )
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    markdown_path = (
        args.output_dir / "A4A_1C_FULL_TRAINING_CONTRACT.md"
    )
    markdown_path.write_text(
        markdown(contract),
        encoding="utf-8",
    )

    audit = {
        "status": "PASS",
        "designation": contract["designation"],
        "frozen_null_target_weight": float(applied_null_weight),
        "total_parameters": EXPECTED_TOTAL_PARAMETERS,
        "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        "primary_batch_size": 512,
        "fallback_batch_size": 256,
        "gradient_clip_norm": 5.0,
        "maximum_epochs": 75,
        "patience_epochs": 10,
        "full_validation_every_epochs": 3,
        "contract_sha256": sha256_file(contract_path),
        "contract_markdown_sha256": sha256_file(markdown_path),
        "training_performed": False,
        "development_test_accessed": False,
        "ready_for_a4a_1d_batch512_resource_probe": True,
    }

    audit_path = args.output_dir / "contract_freeze_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lock = {
        "status": "A4A_1C_FULL_TRAINING_CONTRACT_COMPLETE",
        "contract_sha256": sha256_file(contract_path),
        "contract_markdown_sha256": sha256_file(markdown_path),
        "contract_freeze_audit_sha256": sha256_file(audit_path),
        "base_checkpoint_sha256": hashes[
            "base_checkpoint_sha256"
        ],
        "slot_module_sha256": hashes["slot_module_sha256"],
        "smoke_report_sha256": contract["provenance"][
            "smoke_report_sha256"
        ],
        "label_coverage_audit_sha256": contract["provenance"][
            "label_coverage_audit_sha256"
        ],
        "training_performed": False,
        "validation_selection_performed": False,
        "threshold_search_performed": False,
        "development_test_accessed": False,
    }

    (args.output_dir / "A4A_1C_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (
        args.output_dir / "V4_A4A_1C_FULL_TRAINING_CONTRACT_PASS"
    ).write_text(
        "V4_A4A_1C_FULL_TRAINING_CONTRACT_PASS\n",
        encoding="utf-8",
    )

    print(json.dumps(audit, indent=2, sort_keys=True))
    print("V4_A4A_1C_FULL_TRAINING_CONTRACT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
