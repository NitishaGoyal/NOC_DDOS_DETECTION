#!/usr/bin/env python3
"""
V4-A4a Final Closure

Closes A4a as a validation-only negative result after the frozen A4a-2B HOLD.

This stage:
- reads only small reports, locks, contracts, and one retained checkpoint;
- performs no dataset loading;
- performs no model inference;
- performs no threshold search;
- performs no development-test access;
- does not select or promote a checkpoint.

Outputs:
- V4_A4A_FINAL_CLOSURE.json
- V4_A4A_FINAL_CLOSURE.md
- V4_A4A_FINAL_CLOSURE_LOCK.json
- V4_A4A_CLOSED_NEGATIVE_RESULT_PASS
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


EXPECTED_SELECTOR_V2_SHA = (
    "73c607fc721083ebb5549cf844edb6dc"
    "3e7fb53525f7e1d4eac3ed53a7f57da4"
)
EXPECTED_PRIMARY_TRAINER_SHA = (
    "c424a4877c39cdbf1df54dba89d15291"
    "3c7aef3d7a265930f21c9af4ec52f42d"
)
EXPECTED_BASE_CHECKPOINT_SHA = (
    "f61c1add6c057f7f53dd34fb1f9f4e95"
    "b01cefd5053e0e42bc100c76dac7f923"
)


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


def atomic_write(path: Path, text: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def recursive_find_metric_block(value: Any) -> dict[str, Any] | None:
    required = {
        "temporal_primary_attack_exact",
        "temporal_oracle_attack_exact",
        "temporal_oracle_minus_primary_exact_gap",
    }
    if isinstance(value, dict):
        if required.issubset(value.keys()):
            return value
        for child in value.values():
            found = recursive_find_metric_block(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = recursive_find_metric_block(child)
            if found is not None:
                return found
    return None


def locate_a3_12_summary(root: Path) -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith("_LOCK.json"):
            continue
        try:
            value = load_json(path)
        except Exception:
            continue
        block = recursive_find_metric_block(value)
        if block is not None:
            candidates.append((path, block))

    unique = []
    seen = set()
    for path, block in candidates:
        signature = (
            float(block["temporal_primary_attack_exact"]),
            float(block["temporal_oracle_attack_exact"]),
            float(block["temporal_oracle_minus_primary_exact_gap"]),
        )
        if signature not in seen:
            seen.add(signature)
            unique.append((path, block))

    if len(unique) != 1:
        raise RuntimeError(
            "expected one unique A3.12 validation metric block, found "
            f"{[(str(path), block) for path, block in unique]}"
        )
    return unique[0]


def format_percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-checkpoint", type=Path, required=True)
    parser.add_argument("--a3-12-dir", type=Path, required=True)
    parser.add_argument("--a3-12-lock", type=Path, required=True)
    parser.add_argument("--selector-v2", type=Path, required=True)
    parser.add_argument("--primary-trainer", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--full-training-contract", type=Path, required=True)
    parser.add_argument("--slot-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    required = [
        args.repo,
        args.selection_dir / "selection_report.json",
        args.selection_dir / "A4A_2B_LOCK.json",
        args.selection_dir / "V4_A4A_2B_VALIDATION_SELECTION_HOLD",
        args.diagnostic_checkpoint,
        args.a3_12_dir,
        args.a3_12_lock,
        args.selector_v2,
        args.primary_trainer,
        args.base_checkpoint,
        args.full_training_contract,
        args.slot_contract,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("V4 A4a closure FAIL: missing required paths", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(
            f"V4 A4a closure FAIL: output exists: {args.output_dir}",
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True)

    hashes = {
        "selector_v2_sha256": sha256_file(args.selector_v2),
        "primary_trainer_sha256": sha256_file(args.primary_trainer),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "selection_report_sha256": sha256_file(
            args.selection_dir / "selection_report.json"
        ),
        "selection_lock_sha256": sha256_file(
            args.selection_dir / "A4A_2B_LOCK.json"
        ),
        "diagnostic_checkpoint_sha256": sha256_file(
            args.diagnostic_checkpoint
        ),
        "a3_12_lock_sha256": sha256_file(args.a3_12_lock),
        "full_training_contract_sha256": sha256_file(
            args.full_training_contract
        ),
        "slot_contract_sha256": sha256_file(args.slot_contract),
        "closure_script_sha256": sha256_file(Path(__file__)),
    }

    failures = []
    if hashes["selector_v2_sha256"] != EXPECTED_SELECTOR_V2_SHA:
        failures.append("selector v2 hash mismatch")
    if hashes["primary_trainer_sha256"] != EXPECTED_PRIMARY_TRAINER_SHA:
        failures.append("primary trainer hash mismatch")
    if hashes["base_checkpoint_sha256"] != EXPECTED_BASE_CHECKPOINT_SHA:
        failures.append("base checkpoint hash mismatch")

    selection = load_json(args.selection_dir / "selection_report.json")
    selection_lock = load_json(args.selection_dir / "A4A_2B_LOCK.json")
    a3_12_lock = load_json(args.a3_12_lock)
    full_contract = load_json(args.full_training_contract)
    slot_contract = load_json(args.slot_contract)

    if selection.get("status") != "HOLD":
        failures.append(
            f"selection status is not HOLD: {selection.get('status')}"
        )
    if selection.get("selection_status") != (
        "HOLD_NO_CHECKPOINT_PASSES_ALL_FROZEN_GATES"
    ):
        failures.append(
            "selection_status is not the frozen all-gates HOLD"
        )
    if selection.get("eligible_checkpoint_epochs") != []:
        failures.append("eligible checkpoint list is not empty")
    if selection.get("selected_checkpoint") is not None:
        failures.append("a checkpoint was unexpectedly selected")
    if selection.get("development_test_transfer_authorized") is not False:
        failures.append("development-test transfer was unexpectedly authorized")
    for key in (
        "threshold_search_performed",
        "test_loader_constructed",
        "test_evaluated",
        "development_test_accessed",
    ):
        if selection.get(key) is not False:
            failures.append(f"selection report does not affirm {key}=false")

    if selection_lock.get("status") != "A4A_2B_VALIDATION_SELECTION_HOLD":
        failures.append("selection lock is not HOLD")
    if selection_lock.get("selection_report_sha256") != hashes[
        "selection_report_sha256"
    ]:
        failures.append("selection report hash differs from selection lock")
    if selection_lock.get("development_test_accessed") is not False:
        failures.append("selection lock reports development-test access")

    diagnostic = selection.get("diagnostic_best_checkpoint")
    if not isinstance(diagnostic, dict):
        failures.append("diagnostic best checkpoint is absent")
        diagnostic = {}

    checkpoint = torch.load(
        args.diagnostic_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    raw_metrics = checkpoint.get("validation_metrics")
    raw_losses = checkpoint.get("validation_losses")
    if not isinstance(raw_metrics, dict):
        failures.append("diagnostic checkpoint lacks validation_metrics")
        raw_metrics = {}
    if not isinstance(raw_losses, dict):
        failures.append("diagnostic checkpoint lacks validation_losses")
        raw_losses = {}

    diagnostic_epoch = int(diagnostic.get("checkpoint_epoch", -1))
    checkpoint_epoch = int(checkpoint.get("epoch", -2))
    if diagnostic_epoch != checkpoint_epoch:
        failures.append(
            f"diagnostic epoch mismatch: report={diagnostic_epoch}, "
            f"checkpoint={checkpoint_epoch}"
        )
    if diagnostic.get("checkpoint_sha256") != hashes[
        "diagnostic_checkpoint_sha256"
    ]:
        failures.append("diagnostic checkpoint hash differs from report")

    a3_12_summary_path, a3_12_metrics = locate_a3_12_summary(
        args.a3_12_dir
    )
    hashes["a3_12_summary_sha256"] = sha256_file(a3_12_summary_path)

    baseline_exact = float(
        a3_12_metrics["temporal_primary_attack_exact"]
    )
    oracle_exact = float(
        a3_12_metrics["temporal_oracle_attack_exact"]
    )
    oracle_gap = float(
        a3_12_metrics["temporal_oracle_minus_primary_exact_gap"]
    )

    frozen_gates = selection["frozen_gates"]
    minimum_gain_pp = float(
        frozen_gates["minimum_exact_improvement_percentage_points"]
    )
    required_exact = baseline_exact + minimum_gain_pp / 100.0

    temporal_exact = float(
        diagnostic.get("attack_exact_localization", float("nan"))
    )
    temporal_precision = float(
        diagnostic.get("attack_set_precision", float("nan"))
    )
    temporal_recall = float(
        diagnostic.get("attack_set_recall", float("nan"))
    )
    temporal_coverage = float(
        diagnostic.get("candidate_coverage", float("nan"))
    )
    temporal_persistent_false = float(
        diagnostic.get(
            "normal_run_persistent_false_isolation_fraction",
            float("nan"),
        )
    )
    temporal_stable_exact = float(
        diagnostic.get("stable_attack_exact_point_rate", float("nan"))
    )

    raw_membership_precision = float(
        raw_metrics.get("membership_precision", float("nan"))
    )
    raw_membership_recall = float(
        raw_metrics.get("membership_recall", float("nan"))
    )
    raw_membership_f1 = float(
        raw_metrics.get("membership_f1", float("nan"))
    )
    raw_membership_exact = float(
        raw_metrics.get("membership_exact_fraction", float("nan"))
    )
    raw_cardinality_accuracy = float(
        raw_metrics.get("cardinality_accuracy", float("nan"))
    )
    raw_all_null = float(
        raw_metrics.get("raw_all_null_proposal_fraction", float("nan"))
    )
    raw_duplicate = float(
        raw_metrics.get("raw_duplicate_proposal_fraction", float("nan"))
    )

    required_raw_keys = (
        "membership_precision",
        "membership_recall",
        "membership_f1",
        "membership_exact_fraction",
        "cardinality_accuracy",
        "raw_all_null_proposal_fraction",
        "raw_duplicate_proposal_fraction",
    )
    for key in required_raw_keys:
        if key not in raw_metrics:
            failures.append(f"diagnostic checkpoint lacks raw metric: {key}")

    hypothesis_outcome = "FAILED_TO_RECOVER_CARDINALITY_GAP"
    primary_failure = (
        "The structured slot decoder remained precision-oriented but "
        "under-predicted attacker membership and cardinality. The diagnostic "
        "checkpoint overused NULL, achieved insufficient membership recall, "
        "failed the candidate-coverage gate, and remained far below the "
        "frozen A3-H32 exact-localization reference."
    )

    closure = {
        "status": "FAIL" if failures else "PASS",
        "designation": "V4 A4a Final Scientific Closure",
        "hypothesis": (
            "A four-slot NULL-aware permutation-invariant decoder can recover "
            "a meaningful fraction of the A3.12 cardinality/set gap without "
            "changing the frozen A3 representation."
        ),
        "hypothesis_outcome": hypothesis_outcome,
        "failures": failures,
        "a4a_2b_outcome": {
            "selection_status": selection.get("selection_status"),
            "checkpoint_count_evaluated": selection.get(
                "checkpoint_count_evaluated"
            ),
            "eligible_checkpoint_epochs": selection.get(
                "eligible_checkpoint_epochs"
            ),
            "selected_checkpoint": selection.get("selected_checkpoint"),
            "development_test_transfer_authorized": selection.get(
                "development_test_transfer_authorized"
            ),
        },
        "a3_reference": {
            "a3_12_summary_path": str(a3_12_summary_path),
            "a3_h32_validation_attack_exact": baseline_exact,
            "a3_h32_validation_oracle_count_exact": oracle_exact,
            "a3_h32_validation_oracle_gap": oracle_gap,
            "required_a4a_validation_attack_exact": required_exact,
            "minimum_required_gain_percentage_points": minimum_gain_pp,
        },
        "diagnostic_best": {
            "epoch": checkpoint_epoch,
            "checkpoint_path": str(args.diagnostic_checkpoint.resolve()),
            "checkpoint_sha256": hashes[
                "diagnostic_checkpoint_sha256"
            ],
            "training_validation_total_loss": raw_losses.get("total"),
            "temporal_attack_exact_localization": temporal_exact,
            "temporal_stable_attack_exact_point_rate": temporal_stable_exact,
            "temporal_attack_set_precision": temporal_precision,
            "temporal_attack_set_recall": temporal_recall,
            "temporal_candidate_coverage": temporal_coverage,
            "temporal_persistent_normal_run_false_isolation_fraction": (
                temporal_persistent_false
            ),
            "exact_gain_over_a3_h32_percentage_points": (
                100.0 * (temporal_exact - baseline_exact)
            ),
            "raw_validation_membership_precision": (
                raw_membership_precision
            ),
            "raw_validation_membership_recall": raw_membership_recall,
            "raw_validation_membership_f1": raw_membership_f1,
            "raw_validation_membership_exact_fraction": (
                raw_membership_exact
            ),
            "raw_validation_cardinality_accuracy": (
                raw_cardinality_accuracy
            ),
            "raw_validation_all_null_proposal_fraction": raw_all_null,
            "raw_validation_duplicate_proposal_fraction": raw_duplicate,
        },
        "frozen_gate_results": diagnostic.get("gate_evaluation"),
        "interpretation": {
            "primary_failure": primary_failure,
            "decoder_result": (
                "High precision and benign safety were retained, but recall, "
                "coverage, cardinality accuracy, and exact set localization "
                "were insufficient."
            ),
            "dataset_implication": (
                "V4 does not provide enough explicit source/transit/victim "
                "and route-transition evidence for reliable attacker-set "
                "cardinality and role separation."
            ),
            "further_v4_decoder_tuning_authorized": False,
            "development_test_transfer_authorized": False,
            "next_stage": (
                "Freeze and validate the V5 role-counterfactual dataset "
                "contract, then run the excluded instrumentation pilot."
            ),
        },
        "claim_boundary": {
            "supported": [
                (
                    "A4a preserved high attacker-set precision and zero "
                    "persistent normal-run false isolation for its diagnostic "
                    "best checkpoint."
                ),
                (
                    "A4a did not recover the frozen A3.12 cardinality gap on "
                    "validation."
                ),
                (
                    "No A4a checkpoint passed every frozen promotion gate."
                ),
            ],
            "not_supported": [
                "Autonomous attacker isolation using A4a on V4.",
                "Development-test performance for A4a.",
                "A claim that the structured slot decoder improves over A3-H32.",
            ],
        },
        "closed_actions": {
            "additional_v4_loss_weight_search": False,
            "null_threshold_search": False,
            "cardinality_threshold_search": False,
            "additional_v4_decoder_architecture_search": False,
            "a4a_development_test_transfer": False,
        },
        "audit_boundary": {
            "dataset_samples_loaded": False,
            "model_inference_performed": False,
            "threshold_search_performed": False,
            "test_loader_constructed": False,
            "test_evaluated": False,
            "development_test_accessed": False,
        },
        "provenance": hashes,
    }

    if failures:
        atomic_write(
            args.output_dir / "V4_A4A_FINAL_CLOSURE_FAILED.json",
            json.dumps(closure, indent=2, sort_keys=True) + "\n",
        )
        print(json.dumps(closure, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    json_path = args.output_dir / "V4_A4A_FINAL_CLOSURE.json"
    atomic_write(
        json_path,
        json.dumps(closure, indent=2, sort_keys=True) + "\n",
    )

    markdown = f"""# V4 A4a Final Scientific Closure

## Final status

**Hypothesis outcome:** `{hypothesis_outcome}`

A4a-2B evaluated all {selection.get("checkpoint_count_evaluated")} retained
validation checkpoints. No checkpoint passed every frozen gate, no checkpoint
was selected, and development-test transfer was not authorized.

## Frozen A3 reference

- A3-H32 validation attack exact localization: {format_percent(baseline_exact)}
- A3-H32 oracle-count exact localization: {format_percent(oracle_exact)}
- A3-H32 oracle gap: {100.0 * oracle_gap:.2f} percentage points
- Required A4a validation exact localization: {format_percent(required_exact)}

## Diagnostic best A4a checkpoint

- Epoch: {checkpoint_epoch}
- Temporal attack exact localization: {format_percent(temporal_exact)}
- Stable attack exact point rate: {format_percent(temporal_stable_exact)}
- Attacker-set precision: {format_percent(temporal_precision)}
- Attacker-set recall: {format_percent(temporal_recall)}
- Candidate coverage: {format_percent(temporal_coverage)}
- Persistent normal-run false-isolation fraction: {format_percent(temporal_persistent_false)}
- Gain over A3-H32: {100.0 * (temporal_exact - baseline_exact):.2f} percentage points

## Raw checkpoint evidence

- Membership precision: {format_percent(raw_membership_precision)}
- Membership recall: {format_percent(raw_membership_recall)}
- Membership F1: {format_percent(raw_membership_f1)}
- Membership exact fraction: {format_percent(raw_membership_exact)}
- Cardinality accuracy: {format_percent(raw_cardinality_accuracy)}
- All-NULL proposal fraction: {format_percent(raw_all_null)}
- Duplicate proposal fraction: {format_percent(raw_duplicate)}

## Interpretation

{primary_failure}

A4a retained a useful precision/safety operating point, but it did not recover
the A3.12 cardinality gap. The remaining failure is consistent with missing
source/transit/victim and route-transition semantics in V4 rather than a small
checkpoint-selection error.

## Final boundary

No A4a development-test transfer will be run. Further V4 NULL, cardinality,
loss-weight, or decoder searches are closed. The next stage is V5 dataset
contract freeze and instrumentation-pilot validation.

## Audit boundary

- Dataset samples loaded: no
- Model inference performed: no
- Threshold search performed: no
- Test loader constructed: no
- Test evaluated: no
- Development test accessed: no
"""
    md_path = args.output_dir / "V4_A4A_FINAL_CLOSURE.md"
    atomic_write(md_path, markdown)

    lock = {
        "status": "V4_A4A_CLOSED_NEGATIVE_RESULT",
        "hypothesis_outcome": hypothesis_outcome,
        "closure_json_sha256": sha256_file(json_path),
        "closure_markdown_sha256": sha256_file(md_path),
        "selection_report_sha256": hashes["selection_report_sha256"],
        "diagnostic_checkpoint_sha256": hashes[
            "diagnostic_checkpoint_sha256"
        ],
        "development_test_transfer_authorized": False,
        "further_v4_decoder_tuning_authorized": False,
        "development_test_accessed": False,
        "next_stage": "V5_DATASET_CONTRACT_FREEZE",
    }
    lock_path = args.output_dir / "V4_A4A_FINAL_CLOSURE_LOCK.json"
    atomic_write(
        lock_path,
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
    )

    marker = args.output_dir / "V4_A4A_CLOSED_NEGATIVE_RESULT_PASS"
    atomic_write(marker, "V4_A4A_CLOSED_NEGATIVE_RESULT_PASS\n")

    print(json.dumps(closure, indent=2, sort_keys=True))
    print("V4_A4A_CLOSED_NEGATIVE_RESULT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
