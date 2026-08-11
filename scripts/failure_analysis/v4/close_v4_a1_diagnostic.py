#!/usr/bin/env python3
"""Close the V4-A1 diagnostic baseline after validation thresholding and one-time test transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_empty(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise RuntimeError(f"Closure output directory must be empty: {path}")


def metric_line(prefix: str, graph: dict[str, Any], node: dict[str, Any]) -> str:
    return (
        f"{prefix}: graph_acc={graph['accuracy']:.6f}, "
        f"graph_precision={graph['precision']:.6f}, "
        f"graph_recall={graph['recall']:.6f}, "
        f"graph_f1={graph['f1']:.6f}, graph_fpr={graph['fpr']:.6f}, "
        f"node_precision={node['node_precision']:.6f}, "
        f"node_recall={node['node_recall']:.6f}, "
        f"node_f1={node['node_f1']:.6f}, "
        f"exact_localization={node['exact_localization']:.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-decision", required=True)
    parser.add_argument("--source-audit", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    ensure_empty(out)

    audit_path = Path(args.audit_decision)
    source_audit_path = Path(args.source_audit)
    model_dir = Path(args.model_dir)
    validation_dir = Path(args.validation_dir)
    test_dir = Path(args.test_dir)

    audit = load_json(audit_path)
    source_audit = load_json(source_audit_path)
    summary = load_json(model_dir / "summary.json")
    provenance = load_json(model_dir / "source_provenance.json")
    thresholds = load_json(validation_dir / "selected_thresholds.json")
    validation = load_json(validation_dir / "validation_metrics.json")
    test = load_json(test_dir / "test_metrics.json")
    test_lock = load_json(test_dir / "TEST_TRANSFER_LOCK.json")

    hard_checks = {
        "training_was_authorized": audit.get("training_authorized") is True,
        "authorization_was_diagnostic_only": audit.get("authorization_level") == "diagnostic_only",
        "source_audit_passed": source_audit.get("pass") is True,
        "parameter_count_882": summary.get("parameter_count") == 882,
        "test_was_not_used_for_thresholds": thresholds.get("test_accessed") is False,
        "test_transfer_completed_once": test_lock.get("status") == "COMPLETED",
        "test_designation_correct": test.get("designation") == "development blind test",
        "not_claimed_as_publication_holdout": test.get("independent_publication_holdout") is False,
    }
    closure_pass = all(hard_checks.values())

    active_core_cases = None
    limitations = audit.get("documented_limitations", [])
    # The diagnostic decision file may not expose the count, but its report and prior stage do.
    decision_dir = audit_path.parent
    shortcut_summary = decision_dir / "active_core_shortcut_summary.json"
    if shortcut_summary.is_file():
        shortcut = load_json(shortcut_summary)
        active_core_cases = shortcut.get("active_core_shortcut_case_count")

    technically_successful = bool(
        closure_pass
        and summary.get("best_epoch", -1) >= 1
        and validation.get("sample_count", 0) > 0
        and test.get("sample_count", 0) > 0
    )
    formal_claim_ready = False

    closure_record = {
        "stage": "V4-A1.7 diagnostic closure",
        "closure_pass": closure_pass,
        "hard_checks": hard_checks,
        "architecture": "Frozen Chrono-A1 Conv1D-TemporalGCN",
        "parameter_count": summary.get("parameter_count"),
        "source_sha256": provenance.get("trainer_sha256"),
        "frozen_v3_source_sha256": provenance.get("v3_source_sha256"),
        "dataset_audit_verdict": audit.get("verdict"),
        "training_authorization_level": audit.get("authorization_level"),
        "best_epoch": summary.get("best_epoch"),
        "best_validation_score": summary.get("best_validation_score"),
        "selected_thresholds": thresholds,
        "validation_metrics": validation,
        "test_metrics": test,
        "active_core_shortcut_cases": active_core_cases,
        "documented_limitations": limitations,
        "feature_semantic_limitation": "FEATURE_AMBIGUITY_PRESENT_NO_LABEL_SEPARATION",
        "technically_successful_diagnostic_baseline": technically_successful,
        "suitable_for_formal_scientific_claims": formal_claim_ready,
        "recommended_dataset_correction": (
            "Create matched benign controls in which every future attacker router already "
            "generates legitimate workload traffic, while preserving the same background "
            "family, active-core set and split."
        ),
        "recommended_next_model_experiment": (
            "First analyze V4-A1 failures. Consider a normalized separate-root graph operator "
            "only if localization smearing remains after controlling for the active-core shortcut."
        ),
        "test_designation": "development blind test",
        "publication_holdout_status": "not independent",
    }
    json_dump(out / "v4_a1_diagnostic_closure.json", closure_record)

    graph_val = validation["graph"]
    node_val = validation["node"]
    graph_test = test["graph"]
    node_test = test["node"]

    lines = [
        "# V4-A1 Diagnostic Closure",
        "",
        "## Experiment",
        "",
        "- Architecture: Frozen Chrono-A1 Conv1D-TemporalGCN",
        f"- Parameter count: {summary.get('parameter_count')}",
        f"- Frozen V3 source SHA-256: `{provenance.get('v3_source_sha256')}`",
        f"- V4 trainer SHA-256: `{provenance.get('trainer_sha256')}`",
        f"- Dataset audit verdict: `{audit.get('verdict')}`",
        f"- Authorization level: `{audit.get('authorization_level')}`",
        f"- Best epoch: {summary.get('best_epoch')}",
        f"- Best validation selection score: {summary.get('best_validation_score')}",
        "",
        "## Frozen validation thresholds",
        "",
        f"- Graph threshold: {thresholds.get('graph_threshold')}",
        f"- Node threshold: {thresholds.get('node_threshold')}",
        "",
        "## Metrics",
        "",
        metric_line("Validation", graph_val, node_val),
        "",
        metric_line("Development blind test", graph_test, node_test),
        "",
        "## Scientific limitations",
        "",
        f"- Active-core shortcut cases: {active_core_cases}",
        "- Every attack run may expose traffic at a router that was not a legitimate active workload router in its matched benign control.",
        "- Feature ambiguity is present, although the bounded audit did not demonstrate graph-label separation from that ambiguity.",
        "- The current test split is a development blind test, not an independent publication holdout.",
        "",
        "## Closure decision",
        "",
        f"- Technically successful diagnostic baseline: **{technically_successful}**",
        "- Suitable for formal scientific claims: **False**",
        f"- Closure gate passed: **{closure_pass}**",
        "",
        "## Recommended next actions",
        "",
        "1. Run per-run, per-router, per-attack-kind and per-attacker-count failure analysis.",
        "2. Correct the matched benign design so attacker routers already perform legitimate work.",
        "3. Retrain the same frozen baseline on the corrected dataset before formal architectural claims.",
        "4. Consider a new graph operator only if corrected failure analysis still shows localization smearing.",
        "",
    ]
    (out / "v4_a1_diagnostic_closure.md").write_text("\n".join(lines), encoding="utf-8")

    artifacts = []
    referenced = [
        audit_path,
        source_audit_path,
        model_dir / "best_model.pt",
        model_dir / "summary.json",
        model_dir / "source_provenance.json",
        validation_dir / "selected_thresholds.json",
        validation_dir / "validation_metrics.json",
        test_dir / "test_metrics.json",
        test_dir / "TEST_TRANSFER_LOCK.json",
        out / "v4_a1_diagnostic_closure.json",
        out / "v4_a1_diagnostic_closure.md",
    ]
    for path in referenced:
        artifacts.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    with (out / "v4_a1_closure_artifact_hashes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256", "bytes"])
        writer.writeheader()
        writer.writerows(artifacts)

    print("V4_A1_DIAGNOSTIC_CLOSURE_PASS" if closure_pass else "V4_A1_DIAGNOSTIC_CLOSURE_FAIL")
    print(json.dumps(closure_record, indent=2))
    return 0 if closure_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
