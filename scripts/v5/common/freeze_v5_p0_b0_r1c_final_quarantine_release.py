#!/usr/bin/env python3
"""
V5 P0-B0-R1C Final Quarantine Release

This stage preserves the original B0 and B0-R1 HOLD artifacts, verifies the
successful B0-R1B numeric-provenance audit, freezes the final metadata
quarantine contract, and authorizes B1.

No model is trained. TEST is not enumerated or read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--b0-dir", type=Path, required=True)
    parser.add_argument("--r1-dir", type=Path, required=True)
    parser.add_argument("--r1b-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    a2_dir = args.a2_dir.expanduser().resolve()
    a3_dir = args.a3_dir.expanduser().resolve()
    b0_dir = args.b0_dir.expanduser().resolve()
    r1_dir = args.r1_dir.expanduser().resolve()
    r1b_dir = args.r1b_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output directory already exists: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    paths = {
        "a2_report": a2_dir / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT.json",
        "a2_marker": a2_dir / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_PASS",
        "a3_report": a3_dir / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE.json",
        "a3_marker": a3_dir / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_PASS",
        "b0_report": b0_dir / "V5_P0_B0_SHORTCUT_AUDIT_SUITE.json",
        "b0_decision": b0_dir / "V5_P0_B0_DECISION_REPORT.json",
        "b0_lock": b0_dir / "V5_P0_B0_SHORTCUT_AUDIT_SUITE_LOCK.json",
        "b0_marker": b0_dir / "V5_P0_B0_SHORTCUT_AUDIT_SUITE_HOLD",
        "r1_report": r1_dir / "V5_P0_B0_R1_IDENTIFIER_QUARANTINE.json",
        "r1_marker": r1_dir / "V5_P0_B0_R1_IDENTIFIER_QUARANTINE_HOLD",
        "r1b_report": r1b_dir / "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT.json",
        "r1b_decision": r1b_dir / "V5_P0_B0_R1B_DECISION.json",
        "r1b_lock": r1b_dir / "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_LOCK.json",
        "r1b_marker": r1b_dir / "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_PASS",
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing required artifact {name}: {path}")

    if failures:
        report = {
            "stage": "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE",
            "status": "HOLD",
            "failures": failures,
            "test_split_accessed": False,
        }
        report_path = output_dir / "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE.json"
        write_json(report_path, report)
        atomic_write(
            output_dir / "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_HOLD",
            "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_HOLD\n",
        )
        print("V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_HOLD")
        return 1

    a2 = load_json(paths["a2_report"])
    a3 = load_json(paths["a3_report"])
    b0 = load_json(paths["b0_report"])
    b0_decision = load_json(paths["b0_decision"])
    b0_lock = load_json(paths["b0_lock"])
    r1 = load_json(paths["r1_report"])
    r1b = load_json(paths["r1b_report"])
    r1b_decision = load_json(paths["r1b_decision"])
    r1b_lock = load_json(paths["r1b_lock"])

    # Integrity checks.
    if a2.get("status") != "PASS":
        failures.append("A2 contract status is not PASS")
    if a3.get("status") != "PASS":
        failures.append("A3 loader contract status is not PASS")
    if b0.get("status") != "HOLD":
        failures.append(f"original B0 status is not HOLD: {b0.get('status')!r}")
    if r1.get("status") != "HOLD":
        failures.append(f"B0-R1 status is not HOLD: {r1.get('status')!r}")
    if r1b.get("status") != "PASS":
        failures.append(f"B0-R1B status is not PASS: {r1b.get('status')!r}")

    if b0_lock.get("report_sha256") != sha256_file(paths["b0_report"]):
        failures.append("B0 report SHA does not match B0 lock")
    if r1b_lock.get("report_sha256") != sha256_file(paths["r1b_report"]):
        failures.append("B0-R1B report SHA does not match B0-R1B lock")
    if r1b_lock.get("decision_sha256") != sha256_file(paths["r1b_decision"]):
        failures.append("B0-R1B decision SHA does not match B0-R1B lock")

    expected_b0_reason = "identifier/serialization probe predicts validation labels"
    b0_reasons = b0_decision.get(
        "critical_hold_reasons",
        b0_decision.get("hold_reasons", []),
    )
    if b0_reasons != [expected_b0_reason]:
        failures.append(f"unexpected B0 HOLD reasons: {b0_reasons}")

    r1_failures = r1.get("failures", [])
    expected_prefix = "train attack/control run lengths differ in pairs:"
    if len(r1_failures) != 1 or not str(r1_failures[0]).startswith(expected_prefix):
        failures.append(f"unexpected B0-R1 failure set: {r1_failures}")

    if r1b_decision.get("status") != "PASS":
        failures.append("B0-R1B decision is not PASS")
    if r1b_decision.get("hold_reasons") not in ([], None):
        failures.append(
            f"B0-R1B still has hold reasons: {r1b_decision.get('hold_reasons')}"
        )
    paired = r1b_decision.get("paired_validation_check", {})
    if paired.get("run_length_and_window_position_are_pair_matched") is not True:
        failures.append("validation attack/control pairs are not numerically matched")

    if r1b.get("audit_boundary", {}).get("traffic_x_values_read") is not False:
        failures.append("B0-R1B does not prove traffic x was untouched")
    if r1b.get("audit_boundary", {}).get("test_tensors_read") is not False:
        failures.append("B0-R1B does not prove test tensors were untouched")
    if r1b.get("audit_boundary", {}).get("test_performance_evaluated") is not False:
        failures.append("B0-R1B does not prove test performance was untouched")

    # Re-check every critical numeric-provenance threshold from the frozen report.
    thresholds = {
        "graph_balanced_accuracy": 0.70,
        "count_macro_f1": 0.55,
        "source_exact": 0.50,
        "victim_exact": 0.50,
    }
    critical_variants = (
        "run_length_only",
        "run_window_count_only",
        "within_run_position_only",
        "all_numeric_provenance",
    )

    numeric_metrics = {}
    for name in critical_variants:
        metric = r1b.get("results", {}).get(name)
        if not isinstance(metric, dict):
            failures.append(f"missing B0-R1B result: {name}")
            continue

        reduced = {
            "graph_balanced_accuracy": float(
                metric["graph"]["balanced_accuracy"]
            ),
            "count_macro_f1": float(metric["count"]["macro_f1"]),
            "source_exact": float(metric["roles"]["source"]["exact_set"]),
            "victim_exact": float(metric["roles"]["victim"]["exact_set"]),
        }
        numeric_metrics[name] = reduced

        if reduced["graph_balanced_accuracy"] >= thresholds["graph_balanced_accuracy"]:
            failures.append(
                f"{name} graph balanced accuracy violates release threshold"
            )
        if reduced["count_macro_f1"] >= thresholds["count_macro_f1"]:
            failures.append(f"{name} count macro F1 violates release threshold")
        if (
            reduced["source_exact"] >= thresholds["source_exact"]
            or reduced["victim_exact"] >= thresholds["victim_exact"]
        ):
            failures.append(
                f"{name} source/victim exact localization violates release threshold"
            )

    # Frozen model input boundary.
    loader_contract = a2.get("loader_contract", {})
    if loader_contract.get("primary_feature_variant") != "PRIMARY58":
        failures.append("A2 primary feature variant is not PRIMARY58")
    if loader_contract.get("metadata_concatenated_to_x") is not False:
        failures.append("A2 does not explicitly forbid metadata concatenation")
    if loader_contract.get("raw_mask_supplied_separately") is not True:
        failures.append("A2 does not explicitly require separate raw mask")
    if a3.get("feature_variant") != "PRIMARY58":
        failures.append("A3 did not validate PRIMARY58")
    if a3.get("normalization_applied_by_wrapper") is not False:
        failures.append("A3 wrapper applies unexpected normalization")

    quarantine_contract = {
        "classification": "PROVENANCE_ONLY_SHORTCUT_QUARANTINED",
        "original_b0_hold_preserved": True,
        "original_b0_r1_hold_preserved": True,
        "scientific_interpretation": (
            "Raw provenance identifiers encode attack/control mode, route endpoints, "
            "scenario identity, seed, and attacker count. Four TRAIN matched pairs "
            "also differ in run length, but TRAIN-only numeric-provenance probes fail "
            "to generalize strongly to numerically matched VALIDATION pairs."
        ),
        "allowed_learned_inputs": [
            "PRIMARY58 x",
            "edge_index",
            "topology-derived raw Boolean physical_port_mask",
        ],
        "forbidden_learned_inputs": [
            "mode",
            "run_id",
            "case_id",
            "pair_id",
            "file name",
            "file stem",
            "seed",
            "scenario identifier",
            "source/victim tokens parsed from identifiers",
            "dataset index",
            "run/file serialization order",
            "run length",
            "run window count",
            "window_start",
            "window_target",
            "epoch_id",
            "any hash, embedding, parser output, or derivative of forbidden metadata",
        ],
        "metadata_permitted_only_for": [
            "split construction and overlap auditing",
            "artifact provenance",
            "grouped reporting after predictions are frozen",
            "post-hoc failure analysis after predictions are frozen",
        ],
        "mandatory_training_rules": [
            "never concatenate metadata to model features",
            "never create identifier embeddings",
            "never parse filenames or IDs into model labels",
            "fit normalization only as frozen in A2",
            "use validation only for the declared baseline protocol",
            "keep test untouched until the final locked model",
        ],
        "b0_release_authorized": not failures,
    }

    report = {
        "stage": "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE",
        "status": "PASS" if not failures else "HOLD",
        "classification": (
            "PROVENANCE_ONLY_SHORTCUT_QUARANTINED"
            if not failures
            else "QUARANTINE_RELEASE_NOT_AUTHORIZED"
        ),
        "original_b0_status": "HOLD",
        "original_b0_r1_status": "HOLD",
        "b0_r1b_status": "PASS",
        "numeric_provenance_metrics": numeric_metrics,
        "release_thresholds": thresholds,
        "validation_pair_numeric_matching": paired,
        "quarantine_contract": quarantine_contract,
        "failures": failures,
        "warnings": warnings,
        "audit_boundary": {
            "model_training_performed": False,
            "model_inference_performed": False,
            "train_tensors_read": False,
            "validation_tensors_read": False,
            "test_directory_enumerated": False,
            "test_tensors_read": False,
            "test_performance_evaluated": False,
        },
        "provenance": {
            "a2_report_sha256": sha256_file(paths["a2_report"]),
            "a3_report_sha256": sha256_file(paths["a3_report"]),
            "b0_report_sha256": sha256_file(paths["b0_report"]),
            "b0_decision_sha256": sha256_file(paths["b0_decision"]),
            "b0_lock_sha256": sha256_file(paths["b0_lock"]),
            "r1_report_sha256": sha256_file(paths["r1_report"]),
            "r1b_report_sha256": sha256_file(paths["r1b_report"]),
            "r1b_decision_sha256": sha256_file(paths["r1b_decision"]),
            "r1b_lock_sha256": sha256_file(paths["r1b_lock"]),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "next_stage": (
            "V5_P0_B1_STATIC_FINAL_EPOCH_MLP"
            if not failures
            else "HOLD_REMEDIATE_B0_R1C"
        ),
    }

    report_path = output_dir / "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE.json"
    write_json(report_path, report)

    markdown = f"""# V5 P0-B0-R1C Final Quarantine Release

## Status

`{report["status"]}`

## Classification

`{report["classification"]}`

The original B0 and B0-R1 HOLD artifacts are preserved. B0-R1B demonstrated
that run length, run window count, within-run position, and all numeric
provenance combined do not cross the frozen validation shortcut thresholds.

## Allowed learned inputs

- `PRIMARY58 x`
- `edge_index`
- topology-derived raw Boolean `physical_port_mask`

## Forbidden learned inputs

All provenance identifiers, filenames, mode, seed, run/window timing metadata,
serialization position, and every parsed, hashed, embedded, or derived version
of those fields.

## Next stage

`{report["next_stage"]}`
"""
    markdown_path = output_dir / "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE.md"
    atomic_write(markdown_path, markdown)

    lock = {
        "status": (
            "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_PASS"
            if not failures
            else "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_HOLD"
        ),
        "report_sha256": sha256_file(report_path),
        "markdown_sha256": sha256_file(markdown_path),
        "script_sha256": sha256_file(Path(__file__)),
        "original_b0_hold_preserved": True,
        "original_b0_r1_hold_preserved": True,
        "test_split_accessed": False,
        "next_stage": report["next_stage"],
    }
    write_json(
        output_dir / "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_LOCK.json",
        lock,
    )

    print("===== V5 P0-B0-R1C FINAL QUARANTINE RELEASE =====")
    print("original_b0_status: HOLD")
    print("original_b0_hold_preserved: true")
    print("original_b0_r1_status: HOLD")
    print("original_b0_r1_hold_preserved: true")
    print("b0_r1b_status: PASS")
    print("validation_pair_matched_numeric_features: true")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("test_split_accessed: false")

    for failure in failures:
        print("HOLD:", failure)
    for warning in warnings:
        print("WARNING:", warning)

    if failures:
        atomic_write(
            output_dir / "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_HOLD",
            "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_HOLD\n",
        )
        print("next_stage: HOLD_REMEDIATE_B0_R1C")
        print("V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_HOLD")
        return 1

    atomic_write(
        output_dir / "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_PASS",
        "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_PASS\n",
    )
    print("classification: PROVENANCE_ONLY_SHORTCUT_QUARANTINED")
    print("b0_release_authorized: true")
    print("next_stage: V5_P0_B1_STATIC_FINAL_EPOCH_MLP")
    print("V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
