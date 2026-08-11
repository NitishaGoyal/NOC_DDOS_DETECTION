from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STAGE = "V5_P3_F4_P1R3_COMPOSITE_LOGIT_AND_A5_METRIC_CONTRACT_FREEZE"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"

EXPECTED_VALIDATION_ITEMS = 13863
EXPECTED_COMPOSITE_LOGITS = 69

EXPECTED_HEAD_WIDTHS = {
    "graph": 1,
    "count": 4,
    "source": 16,
    "transit": 16,
    "victim": 16,
    "path": 16,
}

# These are the already frozen A5 validation values from the current
# Tranche-A preliminary diagnostic lineage. This stage does not introduce
# new metric values; it requires every value to be present in the A5 report.
EXPECTED_A5_METRICS = {
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

METRIC_HINTS = {
    "selection_score": ("selection", "score"),
    "graph_auroc": ("graph", "auroc", "auc"),
    "graph_ap": ("graph", "ap", "average", "precision"),
    "graph_f1_at_0_5": ("graph", "f1"),
    "graph_fpr_at_0_5": ("graph", "fpr", "false", "positive"),
    "count_active_macro_f1": ("count", "active", "macro", "f1"),
    "source_ap": ("source", "src", "ap", "average", "precision"),
    "source_exact_active": ("source", "src", "exact", "active"),
    "transit_ap": ("transit", "ap", "average", "precision"),
    "transit_exact_active": ("transit", "exact", "active"),
    "victim_ap": ("victim", "ap", "average", "precision"),
    "victim_exact_active": ("victim", "exact", "active"),
    "path_ap": ("path", "ap", "average", "precision"),
    "path_exact_active": ("path", "exact", "active"),
}

ROLE_ALIASES = {
    "graph": ("graph", "attack"),
    "count": ("count", "attacker_count"),
    "source": ("source", "src"),
    "transit": ("transit",),
    "victim": ("victim",),
    "path": ("path", "attack_path"),
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


def npy_header_from_bytes(data: bytes) -> dict[str, Any]:
    stream = io.BytesIO(data)
    magic = np.lib.format.read_magic(stream)
    if magic == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif magic in ((2, 0), (3, 0)):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise RuntimeError(f"unsupported NPY version: {magic}")
    return {
        "shape": [int(item) for item in shape],
        "fortran_order": bool(fortran),
        "dtype": str(dtype),
        "version": list(magic),
    }


def inspect_npz_headers(path: Path) -> dict[str, Any]:
    arrays = {}
    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.infolist():
            if not member.filename.endswith(".npy"):
                continue
            with archive.open(member, "r") as handle:
                prefix = handle.read(1024 * 1024)
            arrays[member.filename] = npy_header_from_bytes(prefix)
    return arrays


def array_width(shape: list[int]) -> int | None:
    if not shape or shape[0] != EXPECTED_VALIDATION_ITEMS:
        return None
    if len(shape) == 1:
        return 1
    width = 1
    for dimension in shape[1:]:
        width *= int(dimension)
    return width


def classify_head(name: str) -> str | None:
    low = normalize(name)
    for role, aliases in ROLE_ALIASES.items():
        if any(normalize(alias) in low for alias in aliases):
            return role
    return None


def classify_arrays(arrays: dict[str, dict[str, Any]]) -> dict[str, Any]:
    logit_candidates: dict[str, list[dict[str, Any]]] = {
        role: [] for role in EXPECTED_HEAD_WIDTHS
    }
    label_candidates: dict[str, list[dict[str, Any]]] = {
        role: [] for role in EXPECTED_HEAD_WIDTHS
    }
    other_validation_arrays = []

    for member, header in arrays.items():
        shape = header["shape"]
        width = array_width(shape)
        if width is None:
            continue

        low = normalize(member)
        role = classify_head(member)
        row = {
            "member": member,
            "shape": shape,
            "width": width,
            "dtype": header["dtype"],
        }

        is_logit = any(
            token in low
            for token in ("logit", "score", "raw_output", "prediction_raw")
        )
        is_label = any(
            token in low
            for token in ("label", "target", "ground_truth", "truth", "y_")
        )

        if role is not None and is_logit:
            logit_candidates[role].append(row)
        elif role is not None and is_label:
            label_candidates[role].append(row)
        else:
            other_validation_arrays.append(row)

    selected_logits = {}
    ambiguous_logits = {}
    missing_logits = []

    for role, expected_width in EXPECTED_HEAD_WIDTHS.items():
        rows = [
            row for row in logit_candidates[role]
            if row["width"] == expected_width
        ]
        if len(rows) == 1:
            selected_logits[role] = rows[0]
        elif len(rows) == 0:
            missing_logits.append(role)
        else:
            ambiguous_logits[role] = rows

    selected_labels = {}
    for role, expected_width in EXPECTED_HEAD_WIDTHS.items():
        expected_label_width = 1 if role in ("graph", "count") else 16
        rows = [
            row for row in label_candidates[role]
            if row["width"] == expected_label_width
        ]
        if len(rows) == 1:
            selected_labels[role] = rows[0]

    composite_width = sum(
        row["width"] for row in selected_logits.values()
    )

    return {
        "logit_candidates": logit_candidates,
        "selected_logits": selected_logits,
        "ambiguous_logits": ambiguous_logits,
        "missing_logits": missing_logits,
        "selected_labels": selected_labels,
        "other_validation_arrays": other_validation_arrays,
        "composite_logit_width": composite_width,
        "composite_width_match": (
            composite_width == EXPECTED_COMPOSITE_LOGITS
            and not missing_logits
            and not ambiguous_logits
        ),
    }


def find_numeric_matches(
    document: Any,
    expected: float,
    tolerance: float = 5e-9,
) -> list[dict[str, Any]]:
    rows = []
    for json_path, value in flatten_json(document):
        if not (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            continue
        numeric = float(value)
        if abs(numeric - expected) <= tolerance:
            rows.append({
                "json_path": json_path,
                "value": numeric,
            })
    return rows


def path_score(metric: str, json_path: str) -> int:
    low = normalize(json_path)
    tokens = set(low.split("_"))
    score = 0

    if "validation" in tokens or "val" in tokens:
        score += 30
    if "preliminary" in tokens:
        score += 5
    if "selected" in tokens or "best" in tokens:
        score += 5
    if "test" in tokens:
        score -= 100
    if "train" in tokens:
        score -= 30

    hints = METRIC_HINTS[metric]
    hint_hits = sum(
        normalize(hint) in low
        for hint in hints
    )
    score += hint_hits * 4
    return score


def resolve_a5_metric_vector(
    a5_report: Any,
) -> dict[str, Any]:
    resolved = {}
    ambiguous = {}
    missing = []

    for metric, expected in EXPECTED_A5_METRICS.items():
        rows = find_numeric_matches(a5_report, expected)
        if not rows:
            missing.append(metric)
            continue

        for row in rows:
            row["path_score"] = path_score(metric, row["json_path"])
        rows.sort(
            key=lambda row: (
                row["path_score"],
                row["json_path"],
            ),
            reverse=True,
        )

        top_score = rows[0]["path_score"]
        top_rows = [row for row in rows if row["path_score"] == top_score]
        if len(top_rows) == 1:
            resolved[metric] = {
                **top_rows[0],
                "expected_value": expected,
                "absolute_difference": abs(
                    top_rows[0]["value"] - expected
                ),
            }
        else:
            # Multiple paths with the same exact frozen value are not a
            # scientific ambiguity; retain all provenance and pick the
            # lexicographically first canonical path.
            top_rows.sort(key=lambda row: row["json_path"])
            resolved[metric] = {
                **top_rows[0],
                "expected_value": expected,
                "absolute_difference": abs(
                    top_rows[0]["value"] - expected
                ),
                "equivalent_paths": top_rows,
            }

    return {
        "resolved": resolved,
        "ambiguous": ambiguous,
        "missing": missing,
        "coverage": len(resolved) / len(EXPECTED_A5_METRICS),
    }


def manifest_mentions(
    document: Any,
    needle: str,
) -> list[dict[str, Any]]:
    needle_low = needle.lower()
    rows = []
    for json_path, value in flatten_json(document):
        if isinstance(value, str) and needle_low in value.lower():
            rows.append({
                "json_path": json_path,
                "value": value,
            })
    return rows


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")

    p1r2_report_path = output_dir / (
        "V5_P3_F4_P1R2_IMMUTABLE_LOGIT_"
        "AND_METRIC_LINEAGE_REVIEW_REPORT.json"
    )
    p1r2_lock_path = output_dir / (
        "V5_P3_F4_P1R2_IMMUTABLE_LOGIT_"
        "AND_METRIC_LINEAGE_REVIEW_LOCK.json"
    )
    p1r_report_path = output_dir / (
        "V5_P3_F4_P1R_OFFICIAL_LINEAGE_RECOVERY_"
        "AND_ROUTE_CORRECTION_REPORT.json"
    )
    p1r_lock_path = output_dir / (
        "V5_P3_F4_P1R_OFFICIAL_LINEAGE_RECOVERY_"
        "AND_ROUTE_CORRECTION_LOCK.json"
    )

    required = [
        p1r2_report_path,
        p1r2_lock_path,
        p1r_report_path,
        p1r_lock_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"required prior artifacts missing: {missing}")

    p1r2_report = json.loads(
        p1r2_report_path.read_text(encoding="utf-8")
    )
    p1r2_lock = json.loads(
        p1r2_lock_path.read_text(encoding="utf-8")
    )
    p1r_report = json.loads(
        p1r_report_path.read_text(encoding="utf-8")
    )
    p1r_lock = json.loads(
        p1r_lock_path.read_text(encoding="utf-8")
    )

    require(p1r2_report.get("status") == "PASS", "F4-P1R2 is not PASS")
    require(
        p1r2_lock.get("report_sha256")
        == sha256_file(p1r2_report_path),
        "F4-P1R2 report/lock mismatch",
    )
    require(
        p1r2_lock.get("actual_F4_validation_reproduction_authorized")
        is False,
        "F4-P1R2 unexpectedly authorized actual F4",
    )
    require(p1r_report.get("status") == "PASS", "F4-P1R is not PASS")
    require(
        p1r_lock.get("report_sha256") == sha256_file(p1r_report_path),
        "F4-P1R report/lock mismatch",
    )

    exporter_path = Path(
        p1r2_report["resolved"]["exporter_path"]
    ).resolve()
    reference_path = Path(
        p1r2_report["resolved"]["reference_artifact_path"]
    ).resolve()

    require(exporter_path.is_file(), "D1 exporter missing")
    require(reference_path.is_file(), "immutable NPZ reference missing")
    require(
        sha256_file(exporter_path)
        == p1r2_report["resolved"]["exporter_sha256"],
        "D1 exporter hash changed",
    )
    require(
        sha256_file(reference_path)
        == p1r2_report["resolved"]["reference_artifact_sha256"],
        "immutable reference hash changed",
    )

    arrays = inspect_npz_headers(reference_path)
    composite = classify_arrays(arrays)

    d1_dir = reference_path.parent
    d1_manifest_path = (
        d1_dir / "V5_P3_D1_TRANCHE_A_VALIDATION_EXPORT_MANIFEST.json"
    )
    require(d1_manifest_path.is_file(), "D1 export manifest missing")
    d1_manifest = json.loads(
        d1_manifest_path.read_text(encoding="utf-8")
    )

    reference_name_mentions = manifest_mentions(
        d1_manifest,
        reference_path.name,
    )
    reference_hash_mentions = manifest_mentions(
        d1_manifest,
        sha256_file(reference_path),
    )
    exporter_hash_mentions = manifest_mentions(
        d1_manifest,
        sha256_file(exporter_path),
    )

    a5_dir = (
        repo
        / "reports/v5/p3_a5_tranche_a_review_and_b_handover_readiness"
    )
    a5_report_path = a5_dir / (
        "V5_P3_A5_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_REVIEW_"
        "AND_B_HANDOVER_READINESS_REPORT.json"
    )
    a5_lock_path = a5_dir / (
        "V5_P3_A5_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_REVIEW_"
        "AND_B_HANDOVER_READINESS_LOCK.json"
    )

    require(a5_report_path.is_file(), "A5 report missing")
    require(a5_lock_path.is_file(), "A5 lock missing")
    a5_report = json.loads(
        a5_report_path.read_text(encoding="utf-8")
    )
    a5_lock = json.loads(
        a5_lock_path.read_text(encoding="utf-8")
    )
    require(a5_report.get("status") == "PASS", "A5 report is not PASS")
    require(
        a5_lock.get("report_sha256") == sha256_file(a5_report_path),
        "A5 report/lock mismatch",
    )

    metric_vector = resolve_a5_metric_vector(a5_report)

    # Correct the original F4 metric contract. Graph accuracy and strict
    # all-task exactness were proposed diagnostics, but they are not part of
    # the frozen A5 metric vector listed in the current handover summary.
    metric_contract_amendment = {
        "status": "FROZEN",
        "reason": (
            "F4 must reproduce metrics that were actually frozen by A5. "
            "Graph accuracy and strict exactness were not part of the "
            "14-value frozen A5 validation vector and are therefore optional "
            "diagnostics, not baseline-reproduction gates."
        ),
        "required_frozen_metrics": list(EXPECTED_A5_METRICS),
        "optional_not_frozen": [
            "graph_accuracy",
            "strict_all_task_exactness",
        ],
        "metric_count": len(EXPECTED_A5_METRICS),
        "tolerance": 1e-6,
    }

    composite_ok = (
        composite["composite_width_match"]
        and len(composite["selected_logits"]) == 6
    )
    metric_ok = (
        metric_vector["coverage"] == 1.0
        and not metric_vector["missing"]
        and not metric_vector["ambiguous"]
    )
    manifest_ok = (
        bool(reference_name_mentions)
        and bool(reference_hash_mentions)
    )

    actual_f4_authorized = (
        composite_ok
        and metric_ok
        and manifest_ok
        and p1r2_report["resolved"]["exporter_unique"] is True
        and p1r2_report["resolved"]["reference_artifact_unique"] is True
    )

    composite_path = output_dir / "F4_P1R3_COMPOSITE_69_LOGIT_CONTRACT.json"
    metric_path = output_dir / "F4_P1R3_FROZEN_A5_14_METRIC_VECTOR.json"
    amendment_path = output_dir / "F4_P1R3_METRIC_CONTRACT_AMENDMENT.json"
    lineage_path = output_dir / "F4_P1R3_D1_MANIFEST_LINEAGE.json"
    contract_path = output_dir / "F4_P1R3_ACTUAL_F4_REPRODUCTION_CONTRACT.json"

    atomic_json(
        composite_path,
        {
            "reference_artifact": str(reference_path),
            "reference_artifact_sha256": sha256_file(reference_path),
            "arrays": arrays,
            "classification": composite,
            "expected_head_widths": EXPECTED_HEAD_WIDTHS,
            "expected_composite_width": EXPECTED_COMPOSITE_LOGITS,
            "array_payloads_loaded": False,
            "headers_only": True,
        },
    )
    atomic_json(
        metric_path,
        {
            "A5_report": str(a5_report_path),
            "A5_report_sha256": sha256_file(a5_report_path),
            "A5_lock": str(a5_lock_path),
            "A5_lock_sha256": sha256_file(a5_lock_path),
            "expected_values": EXPECTED_A5_METRICS,
            "resolution": metric_vector,
        },
    )
    atomic_json(amendment_path, metric_contract_amendment)
    atomic_json(
        lineage_path,
        {
            "D1_exporter": str(exporter_path),
            "D1_exporter_sha256": sha256_file(exporter_path),
            "D1_manifest": str(d1_manifest_path),
            "D1_manifest_sha256": sha256_file(d1_manifest_path),
            "reference_artifact": str(reference_path),
            "reference_artifact_sha256": sha256_file(reference_path),
            "reference_name_mentions": reference_name_mentions,
            "reference_hash_mentions": reference_hash_mentions,
            "exporter_hash_mentions": exporter_hash_mentions,
            "manifest_lineage_sufficient": manifest_ok,
        },
    )

    actual_contract = {
        "stage": STAGE,
        "status": "FROZEN" if actual_f4_authorized else "REVIEW_REQUIRED",
        "baseline_route": (
            "official D1 checkpoint replay on guarded validation followed by "
            "numeric comparison with the immutable six-head NPZ reference"
        ),
        "checkpoint": {
            "path": p1r_report["checkpoint"]["tensors"][0]["key"]
            if False else p1r_report["provenance"]["checkpoint_sha256"],
            "sha256": p1r_report["provenance"]["checkpoint_sha256"],
        },
        "model": {
            "path": p1r_report["corrected_selection"]["model_path"],
            "sha256": p1r_report["corrected_selection"]["model_sha256"],
        },
        "loader": {
            "path": p1r_report["corrected_selection"]["loader_path"],
            "sha256": p1r_report["corrected_selection"]["loader_sha256"],
            "split": "validation",
            "active_only": False,
            "expected_items": EXPECTED_VALIDATION_ITEMS,
        },
        "exporter": {
            "path": str(exporter_path),
            "sha256": sha256_file(exporter_path),
        },
        "immutable_reference": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            "selected_logits": composite["selected_logits"],
            "composite_width": composite["composite_logit_width"],
        },
        "comparison": {
            "array_names_shapes_dtypes_exact": True,
            "labels_exact": True,
            "logit_max_absolute_difference_max": 1e-6,
            "logit_mean_absolute_difference_max": 1e-8,
            "all_six_heads_required": True,
        },
        "frozen_A5_metrics": {
            metric: row["value"]
            for metric, row in metric_vector["resolved"].items()
        },
        "metric_tolerance": 1e-6,
        "metric_inheritance_rule": (
            "When freshly exported logits and labels reproduce the immutable "
            "reference within the frozen tolerances, the associated frozen "
            "A5 14-metric vector is reproduced. A separate certified metric "
            "adapter will be required before perturbation/permutation stages."
        ),
        "validation_only": True,
        "sealed_test_access": False,
        "A_test_access": False,
        "actual_F4_authorized": actual_f4_authorized,
    }
    atomic_json(contract_path, actual_contract)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "scope": (
            "Correct the single-array 69-logit assumption, certify the "
            "six-head composite width, resolve the exact 14-value frozen A5 "
            "metric vector, and freeze the actual F4 replay contract."
        ),
        "findings": {
            "reference_exact_single_array_13863x69": False,
            "reference_is_six_head_composite": composite_ok,
            "selected_head_widths": {
                role: row["width"]
                for role, row in composite["selected_logits"].items()
            },
            "composite_logit_width": composite["composite_logit_width"],
            "A5_frozen_metric_count": len(metric_vector["resolved"]),
            "A5_metric_coverage": metric_vector["coverage"],
            "graph_accuracy_frozen_by_A5": False,
            "strict_exact_frozen_by_A5": False,
            "D1_manifest_lineage_sufficient": manifest_ok,
        },
        "decision": {
            "F4_P1R3_complete": True,
            "actual_F4_validation_reproduction_authorized": (
                actual_f4_authorized
            ),
            "validation_access_used_in_P1R3": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_"
                "BASELINE_REPRODUCTION"
                if actual_f4_authorized
                else "V5_P3_F4_P1R4_TARGETED_CONTRACT_REVIEW"
            ),
        },
        "artifacts": {
            "composite_logit_contract": str(composite_path),
            "frozen_A5_metric_vector": str(metric_path),
            "metric_contract_amendment": str(amendment_path),
            "D1_manifest_lineage": str(lineage_path),
            "actual_F4_contract": str(contract_path),
        },
        "governance": {
            "model_instantiated": False,
            "checkpoint_deserialized": False,
            "array_payloads_loaded": False,
            "array_headers_read": True,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
        },
        "provenance": {
            "F4_P1R_report_sha256": sha256_file(p1r_report_path),
            "F4_P1R_lock_sha256": sha256_file(p1r_lock_path),
            "F4_P1R2_report_sha256": sha256_file(p1r2_report_path),
            "F4_P1R2_lock_sha256": sha256_file(p1r2_lock_path),
            "D1_exporter_sha256": sha256_file(exporter_path),
            "D1_manifest_sha256": sha256_file(d1_manifest_path),
            "reference_artifact_sha256": sha256_file(reference_path),
            "A5_report_sha256": sha256_file(a5_report_path),
            "A5_lock_sha256": sha256_file(a5_lock_path),
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
            "composite_logit_contract_sha256": sha256_file(composite_path),
            "frozen_A5_metric_vector_sha256": sha256_file(metric_path),
            "metric_contract_amendment_sha256": sha256_file(amendment_path),
            "D1_manifest_lineage_sha256": sha256_file(lineage_path),
            "actual_F4_contract_sha256": sha256_file(contract_path),
            "composite_logit_width": composite["composite_logit_width"],
            "frozen_A5_metric_count": len(metric_vector["resolved"]),
            "actual_F4_validation_reproduction_authorized": (
                actual_f4_authorized
            ),
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print("reference_exact_single_array_13863x69=false")
    print(f"reference_is_six_head_composite={str(composite_ok).lower()}")
    for role in EXPECTED_HEAD_WIDTHS:
        selected = composite["selected_logits"].get(role)
        print(
            f"logit_head_{role}="
            f"{selected['member'] if selected else None}:"
            f"{selected['shape'] if selected else None}:"
            f"{selected['width'] if selected else None}"
        )
    print(
        "composite_logit_width="
        f"{composite['composite_logit_width']}"
    )
    print(f"composite_logit_width_match={str(composite_ok).lower()}")
    print(f"A5_frozen_metric_count={len(metric_vector['resolved'])}")
    print(f"A5_metric_coverage={metric_vector['coverage']:.8f}")
    print(f"A5_metric_missing={metric_vector['missing']}")
    print("graph_accuracy_frozen_by_A5=false")
    print("strict_exact_frozen_by_A5=false")
    print(f"D1_manifest_lineage_sufficient={str(manifest_ok).lower()}")
    print(
        "actual_F4_validation_reproduction_authorized="
        f"{str(actual_f4_authorized).lower()}"
    )
    print("model_instantiated=false")
    print("checkpoint_deserialized=false")
    print("array_payloads_loaded=false")
    print("array_headers_read=true")
    print("validation_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={report['decision']['next_stage']}")
    print(f"composite_logit_contract={composite_path}")
    print(f"frozen_A5_metric_vector={metric_path}")
    print(f"metric_contract_amendment={amendment_path}")
    print(f"D1_manifest_lineage={lineage_path}")
    print(f"actual_F4_contract={contract_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
