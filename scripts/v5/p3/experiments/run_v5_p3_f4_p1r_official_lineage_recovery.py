from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STAGE = "V5_P3_F4_P1R_OFFICIAL_LINEAGE_RECOVERY_AND_ROUTE_CORRECTION"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
EXPECTED_TRAINABLE_PARAMETERS = 60553
EXPECTED_EDGE_INDEX_ELEMENTS = 96
EXPECTED_VALIDATION_ITEMS = 13863
MODEL_CLASS_NAME = "V6P0Dynamic70GraphConvCount4"

METRIC_ALIASES = {
    "selection_score": (
        "selection_score",
        "selected_score",
        "best_score",
        "score",
    ),
    "graph_accuracy": (
        "graph_accuracy",
        "graph_acc",
        "g_accuracy",
        "g_acc",
    ),
    "graph_auroc": (
        "graph_auroc",
        "graph_auc",
        "g_auroc",
        "g_auc",
        "auroc",
    ),
    "graph_ap": (
        "graph_average_precision",
        "graph_ap",
        "g_average_precision",
        "g_ap",
    ),
    "graph_f1": (
        "graph_f1",
        "g_f1",
    ),
    "graph_fpr": (
        "graph_fpr",
        "g_fpr",
        "false_positive_rate",
    ),
    "count_active_macro_f1": (
        "count_active_macro_f1",
        "active_count_macro_f1",
        "count_macro_f1",
        "count_f1_macro",
    ),
    "source_ap": (
        "source_average_precision",
        "source_ap",
        "src_average_precision",
        "src_ap",
    ),
    "source_exact_active": (
        "source_exact_active",
        "source_exact",
        "src_exact_active",
        "src_exact",
    ),
    "transit_ap": (
        "transit_average_precision",
        "transit_ap",
    ),
    "transit_exact_active": (
        "transit_exact_active",
        "transit_exact",
    ),
    "victim_ap": (
        "victim_average_precision",
        "victim_ap",
    ),
    "victim_exact_active": (
        "victim_exact_active",
        "victim_exact",
    ),
    "path_ap": (
        "path_average_precision",
        "path_ap",
    ),
    "path_exact_active": (
        "path_exact_active",
        "path_exact",
    ),
    "strict_exact": (
        "strict_all_task_exactness",
        "strict_all_task_exact",
        "strict_exact",
        "strict_exactness",
    ),
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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def safe_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def flatten_json(value: Any, prefix: str = "$") -> Iterable[tuple[str, Any]]:
    yield prefix, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_json(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_json(child, f"{prefix}[{index}]")


def find_exact_class_sources(repo: Path) -> list[dict[str, Any]]:
    rows = []
    for root in (repo / "src", repo / "scripts/v5/p3"):
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            text = read_text(path)
            if MODEL_CLASS_NAME not in text:
                continue
            definitions = []
            try:
                tree = ast.parse(text)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        definitions.append({
                            "name": node.name,
                            "line": int(node.lineno),
                        })
            except SyntaxError:
                pass
            rows.append({
                "path": str(path),
                "sha256": sha256_file(path),
                "under_experiments": "/experiments/" in str(path),
                "class_definitions": definitions,
                "exact_class_definition": any(
                    row["name"] == MODEL_CLASS_NAME
                    for row in definitions
                ),
            })
    rows.sort(
        key=lambda row: (
            row["exact_class_definition"],
            not row["under_experiments"],
            row["path"],
        ),
        reverse=True,
    )
    return rows


def find_official_evaluator_candidates(
    repo: Path,
    checkpoint_name: str,
) -> list[dict[str, Any]]:
    rows = []
    roots = [
        repo / "scripts/v5/p3",
        repo / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            path_text = str(path)
            low_path = normalize(path_text)
            if "/experiments/" in path_text:
                continue
            text = read_text(path)
            low_text = normalize(text)

            score = 0
            reasons = []

            if "p3_a4" in low_path or "tranche_a_preliminary_diagnostic" in low_path:
                score += 30
                reasons.append("A4 lineage path")
            if normalize(checkpoint_name) in low_text:
                score += 40
                reasons.append("exact checkpoint filename reference")
            if MODEL_CLASS_NAME.lower() in text.lower():
                score += 25
                reasons.append("exact model class reference")
            if "guardedv5p3trancheapreliminarydataset" in low_text:
                score += 20
                reasons.append("exact guarded loader class reference")
            if "validation" in low_text:
                score += 8
                reasons.append("validation logic")
            if "evaluate" in low_text or "evaluation" in low_text:
                score += 8
                reasons.append("evaluation logic")
            if "selection_score" in low_text:
                score += 12
                reasons.append("selection score")
            if "average_precision" in low_text:
                score += 4
                reasons.append("AP metric")
            if "roc_auc" in low_text or "auroc" in low_text:
                score += 4
                reasons.append("AUROC metric")
            if "false_positive" in low_text or "fpr" in low_text:
                score += 4
                reasons.append("FPR metric")
            if "d8_eplr" in low_path or "decoder" in low_path:
                score -= 100
                reasons.append("decoder-lineage penalty")
            if "f4_p0" in low_path or "f4_p1" in low_path:
                score -= 100
                reasons.append("feature-study self-selection penalty")

            if score > 0:
                rows.append({
                    "path": path_text,
                    "sha256": sha256_file(path),
                    "score": score,
                    "reasons": reasons,
                })

    rows.sort(key=lambda row: (row["score"], row["path"]), reverse=True)
    return rows


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    state_dict = None
    state_dict_key = None
    scalar_metadata = {}

    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                scalar_metadata[str(key)] = value

        for candidate_key in (
            "model_state_dict",
            "state_dict",
            "model",
            "network",
        ):
            candidate = payload.get(candidate_key)
            if isinstance(candidate, dict):
                state_dict = candidate
                state_dict_key = candidate_key
                break

        if state_dict is None and payload and all(
            hasattr(value, "shape") for value in payload.values()
        ):
            state_dict = payload
            state_dict_key = "__top_level__"

    require(isinstance(state_dict, dict), "checkpoint state_dict not found")

    tensor_rows = []
    total_elements = 0
    edge_index_elements = 0
    buffer_like_elements = 0
    parameter_like_elements = 0

    for key, value in state_dict.items():
        if not hasattr(value, "shape"):
            continue
        shape = [int(item) for item in value.shape]
        numel = int(np.prod(shape, dtype=np.int64)) if shape else 1
        total_elements += numel

        normalized_key = normalize(str(key))
        is_edge_index = normalized_key.endswith("edge_index")
        is_buffer_like = (
            is_edge_index
            or normalized_key.endswith("running_mean")
            or normalized_key.endswith("running_var")
            or normalized_key.endswith("num_batches_tracked")
            or normalized_key.endswith("physical_port_mask")
            or normalized_key.endswith("adjacency")
        )

        if is_edge_index:
            edge_index_elements += numel
        if is_buffer_like:
            buffer_like_elements += numel
        else:
            parameter_like_elements += numel

        tensor_rows.append({
            "key": str(key),
            "shape": shape,
            "numel": numel,
            "dtype": str(getattr(value, "dtype", "")),
            "is_edge_index": is_edge_index,
            "is_buffer_like": is_buffer_like,
        })

    return {
        "payload_type": type(payload).__name__,
        "state_dict_key": state_dict_key,
        "scalar_metadata": scalar_metadata,
        "state_dict_tensor_count": len(tensor_rows),
        "state_dict_total_elements": total_elements,
        "edge_index_elements": edge_index_elements,
        "buffer_like_elements": buffer_like_elements,
        "parameter_like_elements": parameter_like_elements,
        "expected_parameter_like_elements": EXPECTED_TRAINABLE_PARAMETERS,
        "expected_edge_index_elements": EXPECTED_EDGE_INDEX_ELEMENTS,
        "parameter_count_match": (
            parameter_like_elements == EXPECTED_TRAINABLE_PARAMETERS
        ),
        "edge_index_count_match": (
            edge_index_elements == EXPECTED_EDGE_INDEX_ELEMENTS
        ),
        "tensors": tensor_rows,
    }


def metric_matches(path: str, aliases: tuple[str, ...]) -> bool:
    normalized_path = normalize(path)
    path_tokens = normalized_path.split("_")
    for alias in aliases:
        normalized_alias = normalize(alias)
        if normalized_path.endswith(normalized_alias):
            return True
        if normalized_alias in normalized_path:
            return True
        alias_tokens = normalized_alias.split("_")
        if all(token in path_tokens for token in alias_tokens):
            return True
    return False


def extract_metric_candidates(
    report_paths: list[Path],
) -> dict[str, list[dict[str, Any]]]:
    by_metric = {key: [] for key in METRIC_ALIASES}
    for path in report_paths:
        document = safe_json(path)
        if document is None:
            continue
        for json_path, value in flatten_json(document):
            if not (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                continue
            for metric, aliases in METRIC_ALIASES.items():
                if metric_matches(json_path, aliases):
                    normalized_path = normalize(json_path)
                    score = 0
                    reasons = []
                    if "validation" in normalized_path or "_val_" in normalized_path:
                        score += 40
                        reasons.append("validation path")
                    if "selected" in normalized_path:
                        score += 20
                        reasons.append("selected path")
                    if "best" in normalized_path:
                        score += 10
                        reasons.append("best path")
                    if "a4" in normalize(str(path)):
                        score += 10
                        reasons.append("A4 report")
                    if "a5" in normalize(str(path)):
                        score += 5
                        reasons.append("A5 report")
                    if "test" in normalized_path:
                        score -= 100
                        reasons.append("test path penalty")
                    if "train" in normalized_path:
                        score -= 20
                        reasons.append("train path penalty")
                    by_metric[metric].append({
                        "file": str(path),
                        "file_sha256": sha256_file(path),
                        "json_path": json_path,
                        "value": float(value),
                        "score": score,
                        "reasons": reasons,
                    })

    for metric in by_metric:
        by_metric[metric].sort(
            key=lambda row: (
                row["score"],
                row["file"],
                row["json_path"],
            ),
            reverse=True,
        )
    return by_metric


def resolve_metric_vector(
    by_metric: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    resolved = {}
    ambiguous = {}
    missing = []

    for metric, rows in by_metric.items():
        if not rows:
            missing.append(metric)
            continue

        top_score = rows[0]["score"]
        top_rows = [row for row in rows if row["score"] == top_score]
        unique_values = sorted({
            round(float(row["value"]), 15)
            for row in top_rows
        })

        if len(unique_values) == 1:
            resolved[metric] = rows[0]
        else:
            ambiguous[metric] = top_rows

    return {
        "resolved": resolved,
        "ambiguous": ambiguous,
        "missing": missing,
        "coverage": len(resolved) / len(METRIC_ALIASES),
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")

    p1_report_path = output_dir / (
        "V5_P3_F4_P1_OFFICIAL_EVALUATOR_ROUTE_"
        "AND_ADAPTER_FREEZE_REPORT.json"
    )
    p1_lock_path = output_dir / (
        "V5_P3_F4_P1_OFFICIAL_EVALUATOR_ROUTE_"
        "AND_ADAPTER_FREEZE_LOCK.json"
    )
    p0_report_path = output_dir / (
        "V5_P3_F4_P0_FROZEN_BASELINE_REPRODUCTION_PREFLIGHT_REPORT.json"
    )
    p0_lock_path = output_dir / (
        "V5_P3_F4_P0_FROZEN_BASELINE_REPRODUCTION_PREFLIGHT_LOCK.json"
    )

    required = [p1_report_path, p1_lock_path, p0_report_path, p0_lock_path]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"required F4-P0/P1 artifacts missing: {missing}")

    p1_report = json.loads(p1_report_path.read_text(encoding="utf-8"))
    p1_lock = json.loads(p1_lock_path.read_text(encoding="utf-8"))
    p0_report = json.loads(p0_report_path.read_text(encoding="utf-8"))
    p0_lock = json.loads(p0_lock_path.read_text(encoding="utf-8"))

    require(p1_report.get("status") == "PASS", "F4-P1 is not PASS")
    require(
        p1_lock.get("report_sha256") == sha256_file(p1_report_path),
        "F4-P1 report/lock mismatch",
    )
    require(
        p1_lock.get("actual_F4_validation_reproduction_authorized") is False,
        "F4-P1 unexpectedly authorized validation reproduction",
    )
    require(p0_report.get("status") == "PASS", "F4-P0 is not PASS")
    require(
        p0_lock.get("report_sha256") == sha256_file(p0_report_path),
        "F4-P0 report/lock mismatch",
    )

    checkpoint_path = Path(
        p0_report["frozen_checkpoint"]["path"]
    ).expanduser().resolve()
    require(checkpoint_path.is_file(), "frozen checkpoint missing")
    require(
        sha256_file(checkpoint_path)
        == p0_report["frozen_checkpoint"]["sha256"],
        "checkpoint hash changed",
    )

    checkpoint_review = inspect_checkpoint(checkpoint_path)

    canonical_loader = (
        repo / "src/data/v5_p3_tranche_a_guarded_loader.py"
    )
    require(canonical_loader.is_file(), "canonical guarded loader missing")
    loader_text = read_text(canonical_loader)
    require(
        "GuardedV5P3TrancheAPreliminaryDataset" in loader_text,
        "canonical loader class missing",
    )

    model_candidates = find_exact_class_sources(repo)
    require(model_candidates, f"{MODEL_CLASS_NAME} source not found")
    exact_model_candidates = [
        row for row in model_candidates
        if row["exact_class_definition"] and not row["under_experiments"]
    ]
    require(
        len(exact_model_candidates) == 1,
        "expected exactly one non-experimental exact model-class definition; "
        f"found={len(exact_model_candidates)}",
    )
    selected_model = exact_model_candidates[0]

    evaluator_candidates = find_official_evaluator_candidates(
        repo,
        checkpoint_path.name,
    )
    require(evaluator_candidates, "no non-experimental official evaluator candidates")
    evaluator_unique = (
        len(evaluator_candidates) == 1
        or evaluator_candidates[0]["score"]
        >= evaluator_candidates[1]["score"] + 10
    )
    selected_evaluator = evaluator_candidates[0]

    a4_dir = (
        repo
        / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107"
    )
    a5_dir = (
        repo
        / "reports/v5/p3_a5_tranche_a_review_and_b_handover_readiness"
    )
    report_paths = sorted(
        [
            path
            for root in (a4_dir, a5_dir)
            if root.is_dir()
            for path in root.rglob("*.json")
            if path.is_file()
            and (
                "report" in path.name.lower()
                or "metric" in path.name.lower()
                or "summary" in path.name.lower()
                or "result" in path.name.lower()
            )
        ]
    )
    require(report_paths, "no A4/A5 JSON reports found")

    metric_candidates = extract_metric_candidates(report_paths)
    metric_resolution = resolve_metric_vector(metric_candidates)

    checkpoint_corrected = (
        checkpoint_review["parameter_count_match"]
        and checkpoint_review["edge_index_count_match"]
        and checkpoint_review["state_dict_total_elements"]
        == EXPECTED_TRAINABLE_PARAMETERS + EXPECTED_EDGE_INDEX_ELEMENTS
    )
    metric_coverage = float(metric_resolution["coverage"])

    actual_f4_authorized = (
        checkpoint_corrected
        and evaluator_unique
        and len(exact_model_candidates) == 1
        and metric_coverage >= 0.75
        and len(metric_resolution["ambiguous"]) == 0
    )

    false_selection_review = {
        "F4_P1_selected_evaluator": p1_report["selected"]["evaluator_path"],
        "F4_P1_selected_model": p1_report["selected"]["model_path"],
        "F4_P1_selected_loader": p1_report["selected"]["loader_path"],
        "classification": {
            "evaluator": (
                "INVALID_SELF_SELECTION_FEATURE_STUDY_PREFLIGHT"
            ),
            "model": (
                "INVALID_SELF_SELECTION_FEATURE_STUDY_ROUTE_FREEZER"
            ),
            "loader": (
                "INVALID_DECODER_CALIBRATION_SCRIPT_NOT_CANONICAL_LOADER"
            ),
        },
        "reason": (
            "F4-P1 ranked token-rich recently created scripts instead of "
            "following the A4 model/evaluator lineage."
        ),
    }

    corrected_contract = {
        "stage": STAGE,
        "status": "FROZEN" if actual_f4_authorized else "REVIEW_REQUIRED",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "state_dict_total_elements": checkpoint_review[
                "state_dict_total_elements"
            ],
            "trainable_parameter_like_elements": checkpoint_review[
                "parameter_like_elements"
            ],
            "edge_index_buffer_elements": checkpoint_review[
                "edge_index_elements"
            ],
            "interpretation": (
                "60,649 total state_dict elements = 60,553 trainable "
                "parameter-like elements + 96 edge_index buffer elements."
            ),
        },
        "model": selected_model,
        "loader": {
            "path": str(canonical_loader),
            "sha256": sha256_file(canonical_loader),
            "class": "GuardedV5P3TrancheAPreliminaryDataset",
        },
        "evaluator": {
            "selected": selected_evaluator,
            "unique_route": evaluator_unique,
            "candidates": evaluator_candidates[:20],
        },
        "frozen_metric_vector": metric_resolution,
        "validation": {
            "expected_items": EXPECTED_VALIDATION_ITEMS,
            "tolerance": 1e-6,
            "split": "validation",
            "active_only": False,
            "sealed_test_access": False,
            "A_test_access": False,
        },
        "actual_F4_authorized": actual_f4_authorized,
    }

    false_selection_path = output_dir / "F4_P1R_FALSE_SELECTION_REVIEW.json"
    checkpoint_path_out = output_dir / "F4_P1R_CHECKPOINT_PARAMETER_BUFFER_DECOMPOSITION.json"
    lineage_path = output_dir / "F4_P1R_CORRECTED_MODEL_LOADER_EVALUATOR_LINEAGE.json"
    metric_path = output_dir / "F4_P1R_FROZEN_VALIDATION_METRIC_VECTOR.json"
    contract_path = output_dir / "F4_P1R_CORRECTED_F4_ADAPTER_CONTRACT.json"

    atomic_json(false_selection_path, false_selection_review)
    atomic_json(checkpoint_path_out, checkpoint_review)
    atomic_json(
        lineage_path,
        {
            "model_candidates": model_candidates,
            "selected_model": selected_model,
            "canonical_loader": {
                "path": str(canonical_loader),
                "sha256": sha256_file(canonical_loader),
            },
            "evaluator_candidates": evaluator_candidates,
            "selected_evaluator": selected_evaluator,
            "evaluator_unique": evaluator_unique,
        },
    )
    atomic_json(
        metric_path,
        {
            "report_paths": [str(path) for path in report_paths],
            "candidate_rows": metric_candidates,
            "resolution": metric_resolution,
        },
    )
    atomic_json(contract_path, corrected_contract)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "scope": (
            "Reject F4-P1 token-based self-selections, decompose checkpoint "
            "parameters versus registered buffers, recover the exact A4 "
            "model/loader/evaluator lineage, and freeze a corrected F4 route."
        ),
        "F4_P1_findings": {
            "selected_routes_valid": False,
            "false_selection_review": false_selection_review,
            "checkpoint_total_was_not_trainable_parameter_count": True,
        },
        "checkpoint": checkpoint_review,
        "corrected_selection": {
            "model_path": selected_model["path"],
            "model_sha256": selected_model["sha256"],
            "loader_path": str(canonical_loader),
            "loader_sha256": sha256_file(canonical_loader),
            "evaluator_path": selected_evaluator["path"],
            "evaluator_sha256": selected_evaluator["sha256"],
            "evaluator_score": selected_evaluator["score"],
            "evaluator_unique": evaluator_unique,
            "metric_coverage": metric_coverage,
            "metric_missing": metric_resolution["missing"],
            "metric_ambiguous": sorted(metric_resolution["ambiguous"]),
        },
        "decision": {
            "F4_P1R_complete": True,
            "actual_F4_validation_reproduction_authorized": (
                actual_f4_authorized
            ),
            "validation_access_used_in_P1R": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_"
                "BASELINE_REPRODUCTION"
                if actual_f4_authorized
                else "V5_P3_F4_P1R2_METRIC_OR_EVALUATOR_MANUAL_REVIEW"
            ),
        },
        "artifacts": {
            "false_selection_review": str(false_selection_path),
            "checkpoint_decomposition": str(checkpoint_path_out),
            "corrected_lineage": str(lineage_path),
            "metric_vector": str(metric_path),
            "corrected_adapter_contract": str(contract_path),
        },
        "governance": {
            "checkpoint_deserialized_for_structure_only": True,
            "model_instantiated": False,
            "training_tensors_loaded": False,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
        },
        "provenance": {
            "F4_P0_report_sha256": sha256_file(p0_report_path),
            "F4_P0_lock_sha256": sha256_file(p0_lock_path),
            "F4_P1_report_sha256": sha256_file(p1_report_path),
            "F4_P1_lock_sha256": sha256_file(p1_lock_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
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
            "false_selection_review_sha256": sha256_file(false_selection_path),
            "checkpoint_decomposition_sha256": sha256_file(checkpoint_path_out),
            "corrected_lineage_sha256": sha256_file(lineage_path),
            "metric_vector_sha256": sha256_file(metric_path),
            "corrected_adapter_contract_sha256": sha256_file(contract_path),
            "checkpoint_parameter_like_elements": checkpoint_review[
                "parameter_like_elements"
            ],
            "checkpoint_edge_index_elements": checkpoint_review[
                "edge_index_elements"
            ],
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
    print("F4_P1_selected_routes_valid=false")
    print(
        "checkpoint_state_dict_total_elements="
        f"{checkpoint_review['state_dict_total_elements']}"
    )
    print(
        "checkpoint_parameter_like_elements="
        f"{checkpoint_review['parameter_like_elements']}"
    )
    print(
        "checkpoint_edge_index_buffer_elements="
        f"{checkpoint_review['edge_index_elements']}"
    )
    print(
        "checkpoint_parameter_count_match="
        f"{str(checkpoint_review['parameter_count_match']).lower()}"
    )
    print(
        "checkpoint_edge_index_count_match="
        f"{str(checkpoint_review['edge_index_count_match']).lower()}"
    )
    print(f"selected_model={selected_model['path']}")
    print(f"selected_model_sha256={selected_model['sha256']}")
    print(f"selected_loader={canonical_loader}")
    print(f"selected_loader_sha256={sha256_file(canonical_loader)}")
    print(f"selected_evaluator={selected_evaluator['path']}")
    print(f"selected_evaluator_sha256={selected_evaluator['sha256']}")
    print(f"selected_evaluator_score={selected_evaluator['score']}")
    print(f"selected_evaluator_unique={str(evaluator_unique).lower()}")
    print(f"frozen_metric_coverage={metric_coverage:.8f}")
    print(f"frozen_metric_missing={metric_resolution['missing']}")
    print(
        "frozen_metric_ambiguous="
        f"{sorted(metric_resolution['ambiguous'])}"
    )
    print(
        "actual_F4_validation_reproduction_authorized="
        f"{str(actual_f4_authorized).lower()}"
    )
    print("model_instantiated=false")
    print("validation_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={report['decision']['next_stage']}")
    print(f"false_selection_review={false_selection_path}")
    print(f"checkpoint_decomposition={checkpoint_path_out}")
    print(f"corrected_lineage={lineage_path}")
    print(f"metric_vector={metric_path}")
    print(f"corrected_adapter_contract={contract_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
