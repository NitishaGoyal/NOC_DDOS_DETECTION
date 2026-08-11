from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STAGE = "V5_P3_F4_P1R4_A4_A5_METRIC_PROVENANCE_RECOVERY"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"

TOLERANCE = 1e-6
CONTEXT_WINDOW = 220

EXPECTED_METRICS = {
    "selection_score": 0.72977369,
    "graph_auroc": 0.89083908,
    "graph_ap": 0.68912114,
    "graph_f1_at_0_5": 0.70116570,
    "graph_fpr_at_0_5": 0.21302659,
    "count_active_macro_f1": 0.99730396,
    "source_ap": 0.64469298,
    "source_exact_active": 0.92098378,
    "transit_ap": 0.63692713,
    "transit_exact_active": 0.68027211,
    "victim_ap": 0.62489619,
    "victim_exact_active": 0.94819466,
    "path_ap": 0.62463532,
    "path_exact_active": 0.81789639,
}

ALIASES = {
    "selection_score": (
        ("selection", "score"),
        ("stable", "selection", "score"),
    ),
    "graph_auroc": (
        ("graph", "auroc"),
        ("graph", "auc"),
        ("g", "auc"),
    ),
    "graph_ap": (
        ("graph", "ap"),
        ("graph", "average", "precision"),
        ("g", "ap"),
    ),
    "graph_f1_at_0_5": (
        ("graph", "f1"),
        ("g", "f1"),
    ),
    "graph_fpr_at_0_5": (
        ("graph", "fpr"),
        ("false", "positive", "rate"),
        ("g", "fpr"),
    ),
    "count_active_macro_f1": (
        ("count", "active", "macro", "f1"),
        ("active", "count", "macro", "f1"),
        ("count", "macro", "f1"),
    ),
    "source_ap": (
        ("source", "ap"),
        ("source", "average", "precision"),
        ("src", "ap"),
    ),
    "source_exact_active": (
        ("source", "exact", "active"),
        ("src", "exact", "active"),
        ("source", "exact"),
    ),
    "transit_ap": (
        ("transit", "ap"),
        ("transit", "average", "precision"),
    ),
    "transit_exact_active": (
        ("transit", "exact", "active"),
        ("transit", "exact"),
    ),
    "victim_ap": (
        ("victim", "ap"),
        ("victim", "average", "precision"),
    ),
    "victim_exact_active": (
        ("victim", "exact", "active"),
        ("victim", "exact"),
    ),
    "path_ap": (
        ("path", "ap"),
        ("path", "average", "precision"),
    ),
    "path_exact_active": (
        ("path", "exact", "active"),
        ("path", "exact"),
    ),
}

SOURCE_SUFFIXES = (
    ".json",
    ".csv",
    ".txt",
    ".log",
    ".md",
)

NEGATIVE_TERMS = (
    "test",
    "sealed",
    "decoder",
    "eplr",
    "milp",
    "posthoc",
    "post_hoc",
    "experiment",
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
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def flatten_json(value: Any, prefix: str = "$") -> Iterable[tuple[str, Any]]:
    yield prefix, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_json(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_json(child, f"{prefix}[{index}]")


def numeric_matches(value: Any, expected: float) -> bool:
    if not (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and abs(numeric - expected) <= TOLERANCE


def alias_score(metric: str, context: str) -> tuple[int, list[str]]:
    normalized = normalize(context)
    tokens = set(normalized.split("_"))
    best = 0
    reasons = []

    for alias_group in ALIASES[metric]:
        hits = sum(normalize(term) in normalized for term in alias_group)
        if hits == len(alias_group):
            score = 18 + 4 * hits
            if score > best:
                best = score
                reasons = [f"matched alias group {alias_group}"]
        elif hits:
            score = 3 * hits
            if score > best:
                best = score
                reasons = [f"partial alias group {alias_group}: {hits}"]

    if "validation" in tokens or "val" in tokens:
        best += 12
        reasons.append("validation context")
    if "a4" in tokens:
        best += 8
        reasons.append("A4 context")
    if "a5" in tokens:
        best += 8
        reasons.append("A5 context")
    if "selected" in tokens or "best" in tokens:
        best += 4
        reasons.append("selected/best context")
    if "preliminary" in tokens:
        best += 2
        reasons.append("preliminary context")

    for term in NEGATIVE_TERMS:
        if term in tokens:
            best -= 40
            reasons.append(f"negative term: {term}")

    return best, reasons


def scan_json(path: Path) -> list[dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    rows = []
    for json_path, value in flatten_json(document):
        for metric, expected in EXPECTED_METRICS.items():
            if not numeric_matches(value, expected):
                continue
            context = f"{path} {json_path}"
            score, reasons = alias_score(metric, context)
            rows.append({
                "metric": metric,
                "expected_value": expected,
                "observed_value": float(value),
                "absolute_difference": abs(float(value) - expected),
                "file": str(path),
                "file_sha256": sha256_file(path),
                "source_type": "json",
                "location": json_path,
                "context": json_path,
                "score": score,
                "reasons": reasons,
            })
    return rows


def scan_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return rows

    reader = csv.reader(text.splitlines())
    parsed = list(reader)
    if not parsed:
        return rows

    header = parsed[0]
    for row_index, row in enumerate(parsed[1:], start=2):
        row_context = " ".join(
            f"{header[index] if index < len(header) else index}={value}"
            for index, value in enumerate(row)
        )
        for column_index, value in enumerate(row):
            try:
                numeric = float(value)
            except Exception:
                continue
            for metric, expected in EXPECTED_METRICS.items():
                if abs(numeric - expected) > TOLERANCE:
                    continue
                column = (
                    header[column_index]
                    if column_index < len(header)
                    else str(column_index)
                )
                context = f"{path} row={row_index} column={column} {row_context}"
                score, reasons = alias_score(metric, context)
                rows.append({
                    "metric": metric,
                    "expected_value": expected,
                    "observed_value": numeric,
                    "absolute_difference": abs(numeric - expected),
                    "file": str(path),
                    "file_sha256": sha256_file(path),
                    "source_type": "csv",
                    "location": f"row={row_index},column={column}",
                    "context": row_context[:1000],
                    "score": score,
                    "reasons": reasons,
                })
    return rows


def scan_text(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    rows = []
    number_pattern = re.compile(
        r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?"
    )

    for match in number_pattern.finditer(text):
        try:
            numeric = float(match.group(0))
        except Exception:
            continue

        matched_metrics = [
            (metric, expected)
            for metric, expected in EXPECTED_METRICS.items()
            if abs(numeric - expected) <= TOLERANCE
        ]
        if not matched_metrics:
            continue

        start = max(0, match.start() - CONTEXT_WINDOW)
        end = min(len(text), match.end() + CONTEXT_WINDOW)
        context_text = text[start:end].replace("\x00", " ")
        line_number = text.count("\n", 0, match.start()) + 1

        for metric, expected in matched_metrics:
            context = f"{path} line={line_number} {context_text}"
            score, reasons = alias_score(metric, context)
            rows.append({
                "metric": metric,
                "expected_value": expected,
                "observed_value": numeric,
                "absolute_difference": abs(numeric - expected),
                "file": str(path),
                "file_sha256": sha256_file(path),
                "source_type": "text",
                "location": f"line={line_number}",
                "context": context_text[:1000],
                "score": score,
                "reasons": reasons,
            })
    return rows


def scan_source(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return scan_json(path)
    if suffix == ".csv":
        return scan_csv(path)
    return scan_text(path)


def choose_metric_candidates(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    by_metric = {metric: [] for metric in EXPECTED_METRICS}
    for row in candidates:
        by_metric[row["metric"]].append(row)

    resolved = {}
    missing = []
    low_confidence = []

    for metric, rows in by_metric.items():
        rows.sort(
            key=lambda row: (
                row["score"],
                -row["absolute_difference"],
                row["file"],
                row["location"],
            ),
            reverse=True,
        )
        if not rows:
            missing.append(metric)
            continue

        top_score = rows[0]["score"]
        top_rows = [row for row in rows if row["score"] == top_score]

        # Every candidate was already constrained to the frozen value.
        # Multiple equivalent occurrences are provenance duplication, not
        # conflicting scientific values.
        selected = dict(top_rows[0])
        selected["equivalent_top_occurrences"] = top_rows
        selected["all_candidate_count"] = len(rows)
        selected["confidence"] = (
            "HIGH" if top_score >= 20
            else "MEDIUM" if top_score >= 8
            else "LOW"
        )
        resolved[metric] = selected
        if selected["confidence"] == "LOW":
            low_confidence.append(metric)

    return {
        "resolved": resolved,
        "missing": missing,
        "low_confidence": low_confidence,
        "coverage": len(resolved) / len(EXPECTED_METRICS),
        "all_candidates_by_metric": by_metric,
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")

    p1r3_report_path = output_dir / (
        "V5_P3_F4_P1R3_COMPOSITE_LOGIT_"
        "AND_A5_METRIC_CONTRACT_FREEZE_REPORT.json"
    )
    p1r3_lock_path = output_dir / (
        "V5_P3_F4_P1R3_COMPOSITE_LOGIT_"
        "AND_A5_METRIC_CONTRACT_FREEZE_LOCK.json"
    )
    p1r2_report_path = output_dir / (
        "V5_P3_F4_P1R2_IMMUTABLE_LOGIT_"
        "AND_METRIC_LINEAGE_REVIEW_REPORT.json"
    )
    p1r2_lock_path = output_dir / (
        "V5_P3_F4_P1R2_IMMUTABLE_LOGIT_"
        "AND_METRIC_LINEAGE_REVIEW_LOCK.json"
    )

    required = [
        p1r3_report_path,
        p1r3_lock_path,
        p1r2_report_path,
        p1r2_lock_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"required prior artifacts missing: {missing}")

    p1r3_report = json.loads(
        p1r3_report_path.read_text(encoding="utf-8")
    )
    p1r3_lock = json.loads(
        p1r3_lock_path.read_text(encoding="utf-8")
    )
    p1r2_report = json.loads(
        p1r2_report_path.read_text(encoding="utf-8")
    )
    p1r2_lock = json.loads(
        p1r2_lock_path.read_text(encoding="utf-8")
    )

    require(p1r3_report.get("status") == "PASS", "F4-P1R3 is not PASS")
    require(
        p1r3_lock.get("report_sha256") == sha256_file(p1r3_report_path),
        "F4-P1R3 report/lock mismatch",
    )
    require(
        p1r3_lock.get("actual_F4_validation_reproduction_authorized")
        is False,
        "F4-P1R3 unexpectedly authorized actual F4",
    )
    require(
        p1r3_report["findings"].get("reference_is_six_head_composite")
        is True,
        "six-head composite contract is not certified",
    )
    require(
        int(p1r3_report["findings"].get("composite_logit_width", -1))
        == 69,
        "composite logit width is not 69",
    )
    require(p1r2_report.get("status") == "PASS", "F4-P1R2 is not PASS")
    require(
        p1r2_lock.get("report_sha256") == sha256_file(p1r2_report_path),
        "F4-P1R2 report/lock mismatch",
    )

    a4_dir = (
        repo
        / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107"
    )
    a5_dir = (
        repo
        / "reports/v5/p3_a5_tranche_a_review_and_b_handover_readiness"
    )
    require(a4_dir.is_dir(), "A4 report directory missing")
    require(a5_dir.is_dir(), "A5 report directory missing")

    source_paths = sorted(
        path
        for root in (a4_dir, a5_dir)
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SOURCE_SUFFIXES
        and path.stat().st_size <= 128 * 1024 * 1024
    )
    require(source_paths, "no A4/A5 metric source files found")

    all_candidates = []
    source_inventory = []
    for path in source_paths:
        rows = scan_source(path)
        source_inventory.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
            "candidate_count": len(rows),
        })
        all_candidates.extend(rows)

    resolution = choose_metric_candidates(all_candidates)

    # F4 baseline reproduction is a logit/label replay. A unique metric
    # evaluator is not required to authorize F4 itself when the immutable
    # six-head reference is reproduced. It remains mandatory before F5
    # permutation, where new logits must be scored.
    actual_f4_authorized = (
        resolution["coverage"] == 1.0
        and not resolution["missing"]
        and not resolution["low_confidence"]
        and p1r3_report["findings"]["reference_is_six_head_composite"]
        is True
        and p1r3_report["findings"]["D1_manifest_lineage_sufficient"]
        is True
        and p1r2_report["resolved"]["exporter_unique"] is True
        and p1r2_report["resolved"]["reference_artifact_unique"] is True
    )

    vector = {
        metric: row["observed_value"]
        for metric, row in resolution["resolved"].items()
    }

    provenance_path = output_dir / "F4_P1R4_A4_A5_14_METRIC_PROVENANCE.json"
    source_inventory_path = output_dir / "F4_P1R4_A4_A5_SOURCE_INVENTORY.json"
    boundary_path = output_dir / "F4_P1R4_F4_VS_F5_AUTHORIZATION_BOUNDARY.json"
    contract_path = output_dir / "F4_P1R4_FINAL_F4_REPLAY_CONTRACT.json"

    atomic_json(
        provenance_path,
        {
            "expected_metrics": EXPECTED_METRICS,
            "resolution": resolution,
            "resolved_vector": vector,
            "tolerance": TOLERANCE,
        },
    )
    atomic_json(
        source_inventory_path,
        {
            "A4_directory": str(a4_dir),
            "A5_directory": str(a5_dir),
            "sources": source_inventory,
        },
    )

    boundary = {
        "status": "FROZEN",
        "F4_baseline_reproduction": {
            "purpose": (
                "Replay the frozen checkpoint and prove that fresh validation "
                "labels/logits reproduce the immutable D1 six-head reference."
            ),
            "unique_metric_evaluator_required": False,
            "reason": (
                "Exact reference-logit and label reproduction is sufficient "
                "to inherit the already frozen A4/A5 metric vector."
            ),
            "authorized": actual_f4_authorized,
        },
        "F5_permutation": {
            "purpose": (
                "Generate new perturbed logits and compute new metrics."
            ),
            "unique_metric_adapter_required": True,
            "authorized": False,
            "next_prerequisite": (
                "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION"
            ),
        },
        "sealed_test_access": False,
    }
    atomic_json(boundary_path, boundary)

    final_contract = {
        "stage": STAGE,
        "status": "FROZEN" if actual_f4_authorized else "REVIEW_REQUIRED",
        "exporter": {
            "path": p1r2_report["resolved"]["exporter_path"],
            "sha256": p1r2_report["resolved"]["exporter_sha256"],
        },
        "immutable_reference": {
            "path": p1r2_report["resolved"]["reference_artifact_path"],
            "sha256": p1r2_report["resolved"][
                "reference_artifact_sha256"
            ],
            "format": "six-head NPZ",
            "composite_logit_width": 69,
        },
        "comparison_gates": {
            "validation_items": 13863,
            "labels_exact": True,
            "array_names_shapes_dtypes_exact": True,
            "per_head_max_absolute_logit_difference_max": 1e-6,
            "per_head_mean_absolute_logit_difference_max": 1e-8,
        },
        "frozen_metric_vector": vector,
        "metric_provenance": {
            metric: {
                "file": row["file"],
                "file_sha256": row["file_sha256"],
                "location": row["location"],
                "confidence": row["confidence"],
            }
            for metric, row in resolution["resolved"].items()
        },
        "metric_tolerance": 1e-6,
        "F4_authorized": actual_f4_authorized,
        "F5_authorized": False,
        "validation_only": True,
        "sealed_test_access": False,
        "A_test_access": False,
    }
    atomic_json(contract_path, final_contract)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "scope": (
            "Recover the 14-value frozen A4/A5 validation metric vector from "
            "all authoritative A4/A5 JSON, CSV and text artifacts, and "
            "separate F4 logit-replay authorization from the later F5 "
            "metric-adapter requirement."
        ),
        "metric_recovery": {
            "source_file_count": len(source_paths),
            "numeric_candidate_count": len(all_candidates),
            "resolved_metric_count": len(resolution["resolved"]),
            "coverage": resolution["coverage"],
            "missing": resolution["missing"],
            "low_confidence": resolution["low_confidence"],
            "resolved_vector": vector,
        },
        "decision": {
            "F4_P1R4_complete": True,
            "actual_F4_validation_reproduction_authorized": (
                actual_f4_authorized
            ),
            "F5_permutation_authorized": False,
            "validation_access_used_in_P1R4": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_"
                "BASELINE_REPRODUCTION"
                if actual_f4_authorized
                else "V5_P3_F4_P1R5_METRIC_SOURCE_MANUAL_REVIEW"
            ),
        },
        "artifacts": {
            "metric_provenance": str(provenance_path),
            "source_inventory": str(source_inventory_path),
            "F4_vs_F5_boundary": str(boundary_path),
            "final_F4_contract": str(contract_path),
        },
        "governance": {
            "model_instantiated": False,
            "checkpoint_deserialized": False,
            "array_payloads_loaded": False,
            "training_tensors_loaded": False,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
        },
        "provenance": {
            "F4_P1R2_report_sha256": sha256_file(p1r2_report_path),
            "F4_P1R2_lock_sha256": sha256_file(p1r2_lock_path),
            "F4_P1R3_report_sha256": sha256_file(p1r3_report_path),
            "F4_P1R3_lock_sha256": sha256_file(p1r3_lock_path),
            "installed_script_sha256": sha256_file(installed_script),
        },
    }

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    atomic_json(report_path, report)
    atomic_json(
        lock_path,
        {
            "stage": STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(report_path),
            "metric_provenance_sha256": sha256_file(provenance_path),
            "source_inventory_sha256": sha256_file(source_inventory_path),
            "F4_vs_F5_boundary_sha256": sha256_file(boundary_path),
            "final_F4_contract_sha256": sha256_file(contract_path),
            "resolved_metric_count": len(resolution["resolved"]),
            "metric_coverage": resolution["coverage"],
            "actual_F4_validation_reproduction_authorized": (
                actual_f4_authorized
            ),
            "F5_permutation_authorized": False,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"A4_A5_source_file_count={len(source_paths)}")
    print(f"numeric_candidate_count={len(all_candidates)}")
    print(f"resolved_metric_count={len(resolution['resolved'])}")
    print(f"metric_coverage={resolution['coverage']:.8f}")
    print(f"metric_missing={resolution['missing']}")
    print(f"metric_low_confidence={resolution['low_confidence']}")
    for metric in EXPECTED_METRICS:
        row = resolution["resolved"].get(metric)
        print(
            f"metric_{metric}="
            f"{row['observed_value'] if row else None}:"
            f"{row['confidence'] if row else None}:"
            f"{row['file'] if row else None}:"
            f"{row['location'] if row else None}"
        )
    print(
        "actual_F4_validation_reproduction_authorized="
        f"{str(actual_f4_authorized).lower()}"
    )
    print("F5_permutation_authorized=false")
    print(
        "F5_next_prerequisite="
        "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION"
    )
    print("model_instantiated=false")
    print("checkpoint_deserialized=false")
    print("array_payloads_loaded=false")
    print("validation_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={report['decision']['next_stage']}")
    print(f"metric_provenance={provenance_path}")
    print(f"source_inventory={source_inventory_path}")
    print(f"F4_vs_F5_boundary={boundary_path}")
    print(f"final_F4_contract={contract_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
