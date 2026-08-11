#!/usr/bin/env python3
"""
V5 P0-B0-R1 Identifier Quarantine and B0 Release Freeze

This stage does not erase or reinterpret the valid B0 HOLD. It classifies the
HOLD as provenance-only leakage and freezes a strict metadata quarantine after
verifying:

1. B0 found only the identifier/serialization HOLD.
2. The direct mode baseline and learned identifier probe have the same graph
   confusion matrix.
3. run_id and pair_id directly encode source, victim, and attacker count.
4. Every matched attack/control pair has equal run length.
5. Train and validation runs/pairs are disjoint.
6. A2/A3 permit only PRIMARY58, edge_index, and the audited physical mask as
   model inputs.
7. Test was not accessed.

No model training or evaluation is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def graph_confusion(graph_metrics: dict[str, Any]) -> dict[str, int]:
    return {
        key: int(graph_metrics[key])
        for key in ("tn", "fp", "fn", "tp")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--b0-dir", type=Path, required=True)
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    a2_dir = args.a2_dir.expanduser().resolve()
    a3_dir = args.a3_dir.expanduser().resolve()
    b0_dir = args.b0_dir.expanduser().resolve()
    inventory_dir = args.inventory_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(
            f"STOP: output directory already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    a2_report_path = (
        a2_dir
        / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT.json"
    )
    a2_marker = (
        a2_dir
        / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_PASS"
    )

    a3_report_path = (
        a3_dir
        / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE.json"
    )
    a3_marker = (
        a3_dir
        / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_PASS"
    )

    b0_report_path = (
        b0_dir
        / "V5_P0_B0_SHORTCUT_AUDIT_SUITE.json"
    )
    b0_decision_path = (
        b0_dir
        / "V5_P0_B0_DECISION_REPORT.json"
    )
    b0_lock_path = (
        b0_dir
        / "V5_P0_B0_SHORTCUT_AUDIT_SUITE_LOCK.json"
    )
    b0_marker = (
        b0_dir
        / "V5_P0_B0_SHORTCUT_AUDIT_SUITE_HOLD"
    )

    inventory_path = (
        inventory_dir
        / "V5_P0_B0_R1A_IDENTIFIER_INVENTORY.json"
    )

    required = [
        a2_report_path,
        a2_marker,
        a3_report_path,
        a3_marker,
        b0_report_path,
        b0_decision_path,
        b0_lock_path,
        b0_marker,
        inventory_path,
    ]
    for path in required:
        if not path.is_file():
            failures.append(f"required artifact missing: {path}")

    if failures:
        report = {
            "stage": "V5_P0_B0_R1_IDENTIFIER_QUARANTINE_FREEZE",
            "status": "FAIL",
            "failures": failures,
        }
        report_path = (
            output_dir
            / "V5_P0_B0_R1_IDENTIFIER_QUARANTINE.json"
        )
        write_json(report_path, report)
        atomic_write(
            output_dir
            / "V5_P0_B0_R1_IDENTIFIER_QUARANTINE_HOLD",
            "V5_P0_B0_R1_IDENTIFIER_QUARANTINE_HOLD\n",
        )
        print("V5_P0_B0_R1_IDENTIFIER_QUARANTINE_HOLD")
        return 1

    a2 = load_json(a2_report_path)
    a3 = load_json(a3_report_path)
    b0 = load_json(b0_report_path)
    decision = load_json(b0_decision_path)
    b0_lock = load_json(b0_lock_path)
    inventory = load_json(inventory_path)

    # Provenance integrity.
    if a2.get("status") != "PASS":
        failures.append("A2 contract is not PASS")
    if a3.get("status") != "PASS":
        failures.append("A3 loader audit is not PASS")
    if b0.get("status") != "HOLD":
        failures.append(f"B0 status is not HOLD: {b0.get('status')!r}")
    if b0_lock.get("status") != "V5_P0_B0_SHORTCUT_AUDIT_SUITE_HOLD":
        failures.append(
            f"B0 lock status unexpected: {b0_lock.get('status')!r}"
        )
    if b0_lock.get("report_sha256") != sha256_file(b0_report_path):
        failures.append("B0 report SHA does not match B0 lock")

    hold_reasons = decision.get(
        "critical_hold_reasons",
        decision.get("hold_reasons", []),
    )
    expected_reason = (
        "identifier/serialization probe predicts validation labels"
    )
    if hold_reasons != [expected_reason]:
        failures.append(
            "B0 has unresolved HOLD reasons beyond the expected identifier "
            f"finding: {hold_reasons}"
        )

    if b0.get("test_split_accessed") is not False:
        failures.append("B0 does not explicitly record test_split_accessed=false")
    if inventory.get("test_split_accessed") is not False:
        failures.append(
            "identifier inventory does not explicitly record "
            "test_split_accessed=false"
        )

    # Frozen model-input contract.
    loader_contract = a2.get("loader_contract", {})
    if loader_contract.get("primary_feature_variant") != "PRIMARY58":
        failures.append("A2 primary feature variant is not PRIMARY58")
    if loader_contract.get("metadata_concatenated_to_x") is not False:
        failures.append("A2 does not explicitly forbid metadata in x")
    if loader_contract.get("raw_mask_supplied_separately") is not True:
        failures.append("A2 does not explicitly supply raw mask separately")

    if a3.get("feature_variant") != "PRIMARY58":
        failures.append("A3 did not audit PRIMARY58")
    if a3.get("normalization_applied_by_wrapper") is not False:
        failures.append("A3 wrapper unexpectedly applied normalization")

    # Exact graph-level explanation.
    identifier_graph = (
        b0.get("results", {})
        .get("identifier_serialization_probe", {})
        .get("graph", {})
    )
    mode_graph = (
        b0.get("details", {})
        .get("direct_metadata_audit", {})
        .get("mode_as_attack_baseline", {})
        .get("graph", {})
    )

    if not identifier_graph or not mode_graph:
        failures.append(
            "could not find identifier-probe and mode-baseline graph metrics"
        )
        identifier_confusion = None
        mode_confusion = None
    else:
        identifier_confusion = graph_confusion(identifier_graph)
        mode_confusion = graph_confusion(mode_graph)
        if identifier_confusion != mode_confusion:
            failures.append(
                "identifier probe graph confusion does not match direct mode "
                f"baseline: identifier={identifier_confusion}, "
                f"mode={mode_confusion}"
            )

    # Direct identifier content and matched-pair run-length checks.
    split_checks: dict[str, Any] = {}
    train_pairs: set[str] = set()
    validation_pairs: set[str] = set()
    train_runs: set[str] = set()
    validation_runs: set[str] = set()

    for split in ("train", "validation"):
        split_data = inventory.get("splits", {}).get(split)
        if not isinstance(split_data, dict):
            failures.append(f"identifier inventory missing split: {split}")
            continue

        run_count = int(split_data.get("run_count", -1))
        route_summary = split_data.get("field_parse_summary", {})

        for field in ("run_id", "pair_id"):
            summary = route_summary.get(field, {})
            if int(summary.get("parsed_run_count", -1)) != run_count:
                failures.append(
                    f"{split} {field} does not parse every run"
                )
            for metric in (
                "source_exact_active_mean",
                "victim_exact_active_mean",
                "parsed_count_matches_run_count",
            ):
                value = float(summary.get(metric, -1.0))
                if abs(value - 1.0) > 1e-12:
                    failures.append(
                        f"{split} {field} {metric}={value}, expected 1.0"
                    )

        records = split_data.get("records", [])
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record["pair_id"])].append(record)

        unequal_length_pairs = []
        malformed_pairs = []
        for pair_id, members in groups.items():
            modes = sorted(str(member["mode"]).lower() for member in members)
            if len(members) != 2 or modes != ["attack", "control"]:
                malformed_pairs.append(pair_id)
                continue
            lengths = {int(member["run_length"]) for member in members}
            if len(lengths) != 1:
                unequal_length_pairs.append(pair_id)

        if malformed_pairs:
            failures.append(
                f"{split} malformed matched pairs: {malformed_pairs}"
            )
        if unequal_length_pairs:
            failures.append(
                f"{split} attack/control run lengths differ in pairs: "
                f"{unequal_length_pairs}"
            )

        pairs = set(groups)
        runs = {str(record["run_id"]) for record in records}

        if split == "train":
            train_pairs = pairs
            train_runs = runs
        else:
            validation_pairs = pairs
            validation_runs = runs

        split_checks[split] = {
            "run_count": run_count,
            "pair_count": len(groups),
            "matched_pair_count": len(groups) - len(malformed_pairs),
            "equal_run_length_pair_count": (
                len(groups)
                - len(malformed_pairs)
                - len(unequal_length_pairs)
            ),
            "run_id_direct_source_victim_count_exact": True,
            "pair_id_direct_source_victim_count_exact": True,
        }

    pair_overlap = sorted(train_pairs & validation_pairs)
    run_overlap = sorted(train_runs & validation_runs)
    if pair_overlap:
        failures.append(f"train/validation pair overlap: {pair_overlap}")
    if run_overlap:
        failures.append(f"train/validation run overlap: {run_overlap}")

    classification = (
        "LABEL_BEARING_PROVENANCE_IDENTIFIERS_QUARANTINED"
    )

    quarantine = {
        "classification": classification,
        "finding": (
            "The B0 HOLD is explained by provenance fields that encode run "
            "mode, scenario identity, source routers, victim routers, seed, "
            "and attacker count. These fields are not traffic measurements."
        ),
        "forbidden_learned_inputs": [
            "mode",
            "run_id",
            "case_id",
            "pair_id",
            "file name",
            "file stem",
            "seed",
            "dataset index",
            "run/file serialization order",
            "run length",
            "run window count",
            "window_start",
            "window_target",
            "epoch_id",
            "any parsed or hashed form of these fields",
        ],
        "allowed_model_inputs": [
            "PRIMARY58 x",
            "edge_index",
            "topology-derived raw Boolean physical_port_mask",
        ],
        "metadata_permitted_only_for": [
            "split construction and integrity checks",
            "audit provenance",
            "grouped reporting after predictions are produced",
            "failure analysis after predictions are produced",
        ],
        "training_and_evaluation_requirements": [
            "never concatenate metadata to x",
            "never create embeddings from identifiers",
            "never parse identifiers into router/count labels",
            "never select checkpoints or thresholds using test",
            "keep test untouched until the final locked model",
        ],
        "original_b0_hold_preserved": True,
        "b0_release_authorized": not failures,
    }

    report = {
        "stage": "V5_P0_B0_R1_IDENTIFIER_QUARANTINE_FREEZE",
        "status": "PASS" if not failures else "HOLD",
        "classification": classification if not failures else None,
        "b0_original_status": "HOLD",
        "b0_original_hold_reasons": hold_reasons,
        "identifier_confusion": identifier_confusion,
        "direct_mode_confusion": mode_confusion,
        "confusion_matrices_identical": (
            identifier_confusion == mode_confusion
            if identifier_confusion is not None
            and mode_confusion is not None
            else False
        ),
        "split_checks": split_checks,
        "train_validation_pair_overlap": pair_overlap,
        "train_validation_run_overlap": run_overlap,
        "quarantine_contract": quarantine,
        "failures": failures,
        "warnings": warnings,
        "provenance": {
            "a2_contract_sha256": sha256_file(a2_report_path),
            "a3_report_sha256": sha256_file(a3_report_path),
            "b0_report_sha256": sha256_file(b0_report_path),
            "b0_decision_sha256": sha256_file(b0_decision_path),
            "b0_lock_sha256": sha256_file(b0_lock_path),
            "identifier_inventory_sha256": sha256_file(inventory_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "audit_boundary": {
            "model_training_performed": False,
            "model_inference_performed": False,
            "train_tensors_read": False,
            "validation_tensors_read": False,
            "test_directory_enumerated": False,
            "test_tensors_read": False,
            "test_performance_evaluated": False,
        },
        "next_stage": (
            "V5_P0_B1_STATIC_FINAL_EPOCH_MLP"
            if not failures
            else "HOLD_REMEDIATE_B0_R1"
        ),
    }

    report_path = (
        output_dir
        / "V5_P0_B0_R1_IDENTIFIER_QUARANTINE.json"
    )
    write_json(report_path, report)

    markdown = f"""# V5 P0-B0-R1 Identifier Quarantine

## Classification

`{classification}`

The original B0 HOLD remains preserved. It is classified as a provenance-only
shortcut because `run_id` and `pair_id` directly encode source, victim, and
attacker count, while `run_id`, file naming, and `mode` encode attack/control
run identity.

The learned identifier probe and the direct mode baseline have the same graph
confusion matrix:

`{identifier_confusion}`

Every matched attack/control pair has equal run length. Train and validation
contain no shared runs or pairs.

## Allowed learned inputs

- `PRIMARY58 x`
- `edge_index`
- topology-derived raw Boolean `physical_port_mask`

## Quarantined metadata

All identifiers, filenames, mode, seed, epoch/window indices, run length,
serialization position, and every parsed or hashed derivative are forbidden
as learned inputs.

## Next stage

`{report["next_stage"]}`
"""
    markdown_path = (
        output_dir
        / "V5_P0_B0_R1_IDENTIFIER_QUARANTINE.md"
    )
    atomic_write(markdown_path, markdown)

    lock = {
        "status": (
            "V5_P0_B0_R1_IDENTIFIER_QUARANTINE_PASS"
            if not failures
            else "V5_P0_B0_R1_IDENTIFIER_QUARANTINE_HOLD"
        ),
        "report_sha256": sha256_file(report_path),
        "markdown_sha256": sha256_file(markdown_path),
        "script_sha256": sha256_file(Path(__file__)),
        "original_b0_hold_preserved": True,
        "test_split_accessed": False,
        "next_stage": report["next_stage"],
    }
    write_json(
        output_dir
        / "V5_P0_B0_R1_IDENTIFIER_QUARANTINE_LOCK.json",
        lock,
    )

    print("===== V5 P0-B0-R1 IDENTIFIER QUARANTINE =====")
    print("original_b0_status: HOLD")
    print("original_b0_hold_preserved: true")
    print("identifier_confusion:", identifier_confusion)
    print("direct_mode_confusion:", mode_confusion)
    print(
        "confusion_matrices_identical:",
        report["confusion_matrices_identical"],
    )
    print("train_pair_count:", len(train_pairs))
    print("validation_pair_count:", len(validation_pairs))
    print("pair_overlap_count:", len(pair_overlap))
    print("run_overlap_count:", len(run_overlap))
    for split, check in split_checks.items():
        print(
            f"{split}_equal_run_length_pair_count:",
            check["equal_run_length_pair_count"],
        )
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("test_split_accessed: false")

    if failures:
        for failure in failures:
            print("HOLD:", failure)
        atomic_write(
            output_dir
            / "V5_P0_B0_R1_IDENTIFIER_QUARANTINE_HOLD",
            "V5_P0_B0_R1_IDENTIFIER_QUARANTINE_HOLD\n",
        )
        print("next_stage: HOLD_REMEDIATE_B0_R1")
        print("V5_P0_B0_R1_IDENTIFIER_QUARANTINE_HOLD")
        return 1

    atomic_write(
        output_dir
        / "V5_P0_B0_R1_IDENTIFIER_QUARANTINE_PASS",
        "V5_P0_B0_R1_IDENTIFIER_QUARANTINE_PASS\n",
    )
    print(
        "classification:",
        classification,
    )
    print("b0_release_authorized: true")
    print("next_stage: V5_P0_B1_STATIC_FINAL_EPOCH_MLP")
    print("V5_P0_B0_R1_IDENTIFIER_QUARANTINE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
