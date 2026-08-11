from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STAGE = "V5_P3_F4_P1R2_IMMUTABLE_LOGIT_AND_METRIC_LINEAGE_REVIEW"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"

EXPECTED_VALIDATION_ITEMS = 13863
EXPECTED_LOGITS = 69
EXPECTED_MODEL_PARAMETERS = 60553
EXPECTED_EDGE_INDEX_ELEMENTS = 96

METRICS = {
    "selection_score": (
        "selection_score",
        "selected_score",
        "best_score",
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

DECODER_PENALTY_TERMS = (
    "a0",
    "a1",
    "eplr",
    "decoder",
    "post_hoc",
    "posthoc",
    "milp",
    "repair",
)

RAW_NEURAL_POSITIVE_TERMS = (
    "raw",
    "neural",
    "immutable",
    "validation",
    "logit",
    "baseline",
)

LOGIT_ARTIFACT_SUFFIXES = (
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".json",
    ".csv",
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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def safe_json(path: Path) -> Any | None:
    try:
        if path.stat().st_size > 64 * 1024 * 1024:
            return None
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


def extract_string_literals(path: Path) -> list[str]:
    try:
        tree = ast.parse(read_text(path))
    except SyntaxError:
        return []
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.append(node.value)
    return values


def extract_imports_and_calls(path: Path) -> dict[str, Any]:
    try:
        tree = ast.parse(read_text(path))
    except SyntaxError:
        return {
            "imports": [],
            "from_imports": [],
            "functions": [],
            "classes": [],
            "calls": [],
        }

    imports = []
    from_imports = []
    functions = []
    classes = []
    calls = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            from_imports.append({
                "module": node.module,
                "names": [alias.name for alias in node.names],
            })
        elif isinstance(node, ast.FunctionDef):
            functions.append({
                "name": node.name,
                "line": int(node.lineno),
                "arguments": [arg.arg for arg in node.args.args],
            })
        elif isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "line": int(node.lineno),
            })
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)

    return {
        "imports": sorted(set(imports)),
        "from_imports": from_imports,
        "functions": functions,
        "classes": classes,
        "calls": sorted(set(calls)),
    }


def npy_header_from_bytes(data: bytes) -> dict[str, Any]:
    stream = io.BytesIO(data)
    magic = np.lib.format.read_magic(stream)
    if magic == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif magic in ((2, 0), (3, 0)):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise RuntimeError(f"unsupported NPY version {magic}")
    return {
        "shape": [int(item) for item in shape],
        "fortran_order": bool(fortran),
        "dtype": str(dtype),
        "version": list(magic),
    }


def inspect_array_artifact(path: Path) -> dict[str, Any]:
    record = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "suffix": path.suffix.lower(),
    }

    if path.suffix.lower() == ".npy":
        with path.open("rb") as handle:
            header_prefix = handle.read(1024 * 1024)
        try:
            record["arrays"] = {
                "__array__": npy_header_from_bytes(header_prefix)
            }
        except Exception as exc:
            record["header_error"] = repr(exc)
        return record

    if path.suffix.lower() == ".npz":
        arrays = {}
        try:
            with zipfile.ZipFile(path, "r") as archive:
                for member in archive.infolist():
                    if not member.filename.endswith(".npy"):
                        continue
                    with archive.open(member, "r") as handle:
                        header_prefix = handle.read(1024 * 1024)
                    arrays[member.filename] = npy_header_from_bytes(
                        header_prefix
                    )
            record["arrays"] = arrays
        except Exception as exc:
            record["header_error"] = repr(exc)
        return record

    if path.suffix.lower() in (".pt", ".pth"):
        record["tensor_payload_not_loaded"] = True
        return record

    if path.suffix.lower() == ".json":
        document = safe_json(path)
        if document is not None:
            scalar_rows = []
            for json_path, value in flatten_json(document):
                if isinstance(value, (str, int, float, bool)) or value is None:
                    scalar_rows.append({
                        "json_path": json_path,
                        "value": value,
                    })
                    if len(scalar_rows) >= 300:
                        break
            record["json_scalar_preview"] = scalar_rows
        return record

    return record


def artifact_shape_score(record: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    arrays = record.get("arrays", {})
    for name, info in arrays.items():
        shape = info.get("shape", [])
        if EXPECTED_VALIDATION_ITEMS in shape:
            score += 30
            reasons.append(
                f"{name} includes validation dimension {EXPECTED_VALIDATION_ITEMS}"
            )
        if EXPECTED_LOGITS in shape:
            score += 30
            reasons.append(
                f"{name} includes logit dimension {EXPECTED_LOGITS}"
            )
        if shape == [EXPECTED_VALIDATION_ITEMS, EXPECTED_LOGITS]:
            score += 50
            reasons.append("exact [13863,69] logit matrix")
        if (
            len(shape) >= 2
            and shape[0] == EXPECTED_VALIDATION_ITEMS
            and shape[-1] in (1, 4, 16, 69)
        ):
            score += 10
            reasons.append(f"plausible validation output shape {shape}")
    return score, reasons


def find_d1_exporter(repo: Path) -> list[dict[str, Any]]:
    candidates = []
    for path in (repo / "scripts/v5/p3").rglob("*.py"):
        if "/experiments/" in str(path):
            continue
        low_path = normalize(str(path))
        text = read_text(path)
        low_text = normalize(text)

        score = 0
        reasons = []

        if "run_v5_p3_d1_immutable_validation_logit_export" in low_path:
            score += 100
            reasons.append("exact D1 exporter filename")
        if "immutable_validation_logit_export" in low_path:
            score += 50
            reasons.append("immutable validation logit export path")
        if "v6p0dynamic70graphconvcount4" in low_text:
            score += 20
            reasons.append("exact model class")
        if "guardedv5p3trancheapreliminarydataset" in low_text:
            score += 20
            reasons.append("exact guarded loader")
        if "validation" in low_text and "logit" in low_text:
            score += 15
            reasons.append("validation logit logic")
        if "numpy_savez" in low_text or "savez" in low_text:
            score += 5
            reasons.append("NPZ export")
        if any(term in low_path for term in DECODER_PENALTY_TERMS):
            score -= 100

        if score > 0:
            literals = extract_string_literals(path)
            candidates.append({
                "path": str(path),
                "sha256": sha256_file(path),
                "score": score,
                "reasons": reasons,
                "analysis": extract_imports_and_calls(path),
                "string_literals": literals[:500],
            })

    candidates.sort(key=lambda row: (row["score"], row["path"]), reverse=True)
    return candidates


def find_d1_artifacts(
    repo: Path,
    exporter: dict[str, Any],
) -> list[dict[str, Any]]:
    literal_basenames = {
        Path(value).name
        for value in exporter.get("string_literals", [])
        if any(
            value.lower().endswith(suffix)
            for suffix in LOGIT_ARTIFACT_SUFFIXES
        )
    }

    candidates = []
    for root in (
        repo / "reports/v5",
        repo / "artifacts",
    ):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in LOGIT_ARTIFACT_SUFFIXES:
                continue

            low = normalize(str(path))
            score = 0
            reasons = []

            if "p3_d1" in low:
                score += 40
                reasons.append("D1 path")
            if "immutable" in low:
                score += 15
                reasons.append("immutable path")
            if "validation" in low:
                score += 15
                reasons.append("validation path")
            if "logit" in low or "prediction" in low:
                score += 20
                reasons.append("logit/prediction path")
            if path.name in literal_basenames:
                score += 30
                reasons.append("filename referenced by D1 exporter")
            if any(term in low for term in DECODER_PENALTY_TERMS):
                score -= 100

            if score <= 0:
                continue

            record = inspect_array_artifact(path)
            shape_score, shape_reasons = artifact_shape_score(record)
            score += shape_score
            reasons.extend(shape_reasons)
            record["score"] = score
            record["reasons"] = reasons
            candidates.append(record)

    candidates.sort(key=lambda row: (row["score"], row["path"]), reverse=True)
    return candidates


def metric_match(json_path: str, aliases: tuple[str, ...]) -> bool:
    normalized_path = normalize(json_path)
    return any(normalize(alias) in normalized_path for alias in aliases)


def score_metric_source(path: Path, document: Any) -> dict[str, Any]:
    low = normalize(str(path))
    score = 0
    reasons = []

    if "validation" in low:
        score += 20
        reasons.append("validation path")
    if "raw" in low:
        score += 20
        reasons.append("raw path")
    if "immutable" in low:
        score += 10
        reasons.append("immutable path")
    if "d2" in low or "d3" in low or "d4" in low or "d5" in low:
        score += 5
        reasons.append("post-export evaluation stage")
    if "a4" in low or "a5" in low:
        score += 10
        reasons.append("A4/A5 lineage")
    if any(term in low for term in DECODER_PENALTY_TERMS):
        score -= 100
        reasons.append("decoder/posthoc penalty")

    matches = {metric: [] for metric in METRICS}
    for json_path, value in flatten_json(document):
        if not (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            continue
        normalized_json_path = normalize(json_path)
        path_penalty = 0
        if any(term in normalized_json_path for term in DECODER_PENALTY_TERMS):
            path_penalty -= 100
        if "test" in normalized_json_path:
            path_penalty -= 100
        if "train" in normalized_json_path:
            path_penalty -= 20
        if "validation" in normalized_json_path or "_val_" in normalized_json_path:
            path_penalty += 25
        if "raw" in normalized_json_path:
            path_penalty += 20
        if "selected" in normalized_json_path or "best" in normalized_json_path:
            path_penalty += 5

        for metric, aliases in METRICS.items():
            if metric_match(json_path, aliases):
                matches[metric].append({
                    "json_path": json_path,
                    "value": float(value),
                    "path_score": path_penalty,
                })

    resolved = {}
    ambiguous = {}
    for metric, rows in matches.items():
        if not rows:
            continue
        rows.sort(
            key=lambda row: (
                row["path_score"],
                row["json_path"],
            ),
            reverse=True,
        )
        top_score = rows[0]["path_score"]
        top_rows = [row for row in rows if row["path_score"] == top_score]
        values = {
            round(float(row["value"]), 15)
            for row in top_rows
        }
        if len(values) == 1:
            resolved[metric] = rows[0]
        else:
            ambiguous[metric] = top_rows

    coverage = len(resolved) / len(METRICS)
    score += int(coverage * 100)
    score -= 10 * len(ambiguous)

    if coverage >= 0.75:
        reasons.append(f"metric coverage {coverage:.3f}")
    if not ambiguous:
        reasons.append("no internal top-rank ambiguity")

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "score": score,
        "reasons": reasons,
        "resolved": resolved,
        "ambiguous": ambiguous,
        "missing": sorted(set(METRICS) - set(resolved) - set(ambiguous)),
        "coverage": coverage,
    }


def find_metric_sources(repo: Path) -> list[dict[str, Any]]:
    candidates = []
    for root in (
        repo / "reports/v5",
        repo / "artifacts",
    ):
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            if "/experiments/" in str(path):
                continue
            document = safe_json(path)
            if document is None:
                continue
            result = score_metric_source(path, document)
            if result["coverage"] > 0:
                candidates.append(result)

    candidates.sort(key=lambda row: (row["score"], row["path"]), reverse=True)
    return candidates


def find_metric_evaluators(
    repo: Path,
    metric_source: dict[str, Any] | None,
    d1_exporter: dict[str, Any],
) -> list[dict[str, Any]]:
    source_name = (
        Path(metric_source["path"]).name
        if metric_source is not None
        else ""
    )
    d1_name = Path(d1_exporter["path"]).name

    candidates = []
    for path in (repo / "scripts/v5/p3").rglob("*.py"):
        if "/experiments/" in str(path):
            continue
        low_path = normalize(str(path))
        text = read_text(path)
        low_text = normalize(text)
        literals = extract_string_literals(path)

        score = 0
        reasons = []

        if source_name and source_name in literals:
            score += 50
            reasons.append("references canonical metric report filename")
        if d1_name in literals or normalize(d1_name) in low_text:
            score += 40
            reasons.append("references D1 exporter")
        if "logit" in low_text and "metric" in low_text:
            score += 15
            reasons.append("logit metric logic")
        if "average_precision" in low_text:
            score += 5
            reasons.append("AP metric")
        if "roc_auc" in low_text or "auroc" in low_text:
            score += 5
            reasons.append("AUROC metric")
        if "strict_exact" in low_text:
            score += 5
            reasons.append("strict exact metric")
        if "validation" in low_path or "validation" in low_text:
            score += 5
            reasons.append("validation logic")
        if any(term in low_path for term in DECODER_PENALTY_TERMS):
            score -= 100
            reasons.append("decoder penalty")
        if "d1_immutable_validation_logit_export" in low_path:
            score -= 10
            reasons.append("exporter, not downstream metric evaluator")

        if score > 0:
            candidates.append({
                "path": str(path),
                "sha256": sha256_file(path),
                "score": score,
                "reasons": reasons,
                "analysis": extract_imports_and_calls(path),
                "string_literals": literals[:500],
            })

    candidates.sort(key=lambda row: (row["score"], row["path"]), reverse=True)
    return candidates


def select_unique(
    rows: list[dict[str, Any]],
    margin: int,
) -> tuple[dict[str, Any] | None, bool]:
    if not rows:
        return None, False
    if len(rows) == 1:
        return rows[0], True
    unique = rows[0]["score"] >= rows[1]["score"] + margin
    return rows[0], unique


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")

    p1r_report_path = output_dir / (
        "V5_P3_F4_P1R_OFFICIAL_LINEAGE_RECOVERY_"
        "AND_ROUTE_CORRECTION_REPORT.json"
    )
    p1r_lock_path = output_dir / (
        "V5_P3_F4_P1R_OFFICIAL_LINEAGE_RECOVERY_"
        "AND_ROUTE_CORRECTION_LOCK.json"
    )
    p1_report_path = output_dir / (
        "V5_P3_F4_P1_OFFICIAL_EVALUATOR_ROUTE_"
        "AND_ADAPTER_FREEZE_REPORT.json"
    )
    p1_lock_path = output_dir / (
        "V5_P3_F4_P1_OFFICIAL_EVALUATOR_ROUTE_"
        "AND_ADAPTER_FREEZE_LOCK.json"
    )

    required = [
        p1r_report_path,
        p1r_lock_path,
        p1_report_path,
        p1_lock_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"required F4-P1/P1R artifacts missing: {missing}")

    p1r_report = json.loads(
        p1r_report_path.read_text(encoding="utf-8")
    )
    p1r_lock = json.loads(
        p1r_lock_path.read_text(encoding="utf-8")
    )
    p1_report = json.loads(p1_report_path.read_text(encoding="utf-8"))
    p1_lock = json.loads(p1_lock_path.read_text(encoding="utf-8"))

    require(p1r_report.get("status") == "PASS", "F4-P1R is not PASS")
    require(
        p1r_lock.get("report_sha256") == sha256_file(p1r_report_path),
        "F4-P1R report/lock mismatch",
    )
    require(
        p1r_lock.get("actual_F4_validation_reproduction_authorized")
        is False,
        "F4-P1R unexpectedly authorized actual F4",
    )
    require(p1_report.get("status") == "PASS", "F4-P1 is not PASS")
    require(
        p1_lock.get("report_sha256") == sha256_file(p1_report_path),
        "F4-P1 report/lock mismatch",
    )

    corrected_model = Path(
        p1r_report["corrected_selection"]["model_path"]
    ).resolve()
    corrected_loader = Path(
        p1r_report["corrected_selection"]["loader_path"]
    ).resolve()
    require(corrected_model.is_file(), "corrected model source missing")
    require(corrected_loader.is_file(), "corrected guarded loader missing")
    require(
        sha256_file(corrected_model)
        == p1r_report["corrected_selection"]["model_sha256"],
        "corrected model hash changed",
    )
    require(
        sha256_file(corrected_loader)
        == p1r_report["corrected_selection"]["loader_sha256"],
        "corrected loader hash changed",
    )

    d1_exporters = find_d1_exporter(repo)
    selected_exporter, exporter_unique = select_unique(
        d1_exporters,
        margin=25,
    )
    require(selected_exporter is not None, "D1 exporter not found")

    d1_artifacts = find_d1_artifacts(repo, selected_exporter)
    selected_artifact, artifact_unique = select_unique(
        d1_artifacts,
        margin=20,
    )

    metric_sources = find_metric_sources(repo)
    selected_metric_source, metric_source_unique = select_unique(
        metric_sources,
        margin=15,
    )

    metric_evaluators = find_metric_evaluators(
        repo,
        selected_metric_source,
        selected_exporter,
    )
    selected_metric_evaluator, metric_evaluator_unique = select_unique(
        metric_evaluators,
        margin=15,
    )

    artifact_exact_shape = False
    artifact_validation_dimension = False
    artifact_logit_dimension = False
    if selected_artifact is not None:
        for info in selected_artifact.get("arrays", {}).values():
            shape = info.get("shape", [])
            if shape == [EXPECTED_VALIDATION_ITEMS, EXPECTED_LOGITS]:
                artifact_exact_shape = True
            if EXPECTED_VALIDATION_ITEMS in shape:
                artifact_validation_dimension = True
            if EXPECTED_LOGITS in shape:
                artifact_logit_dimension = True

    metric_coverage = (
        float(selected_metric_source["coverage"])
        if selected_metric_source is not None
        else 0.0
    )
    metric_ambiguity_count = (
        len(selected_metric_source["ambiguous"])
        if selected_metric_source is not None
        else len(METRICS)
    )
    strict_exact_resolved = (
        selected_metric_source is not None
        and "strict_exact" in selected_metric_source["resolved"]
    )

    actual_f4_authorized = (
        exporter_unique
        and selected_artifact is not None
        and artifact_unique
        and artifact_validation_dimension
        and artifact_logit_dimension
        and selected_metric_source is not None
        and metric_source_unique
        and metric_coverage >= 0.875
        and metric_ambiguity_count == 0
        and strict_exact_resolved
        and selected_metric_evaluator is not None
        and metric_evaluator_unique
    )

    preferred_route = (
        "D1 immutable validation-logit export + canonical downstream "
        "raw-neural metric evaluator"
    )

    exporter_path = output_dir / "F4_P1R2_D1_EXPORTER_LINEAGE.json"
    artifact_path = output_dir / "F4_P1R2_IMMUTABLE_VALIDATION_ARTIFACT_LINEAGE.json"
    metric_source_path = output_dir / "F4_P1R2_CANONICAL_RAW_METRIC_SOURCE.json"
    metric_eval_path = output_dir / "F4_P1R2_DOWNSTREAM_METRIC_EVALUATOR_LINEAGE.json"
    contract_path = output_dir / "F4_P1R2_IMMUTABLE_LOGIT_REPRODUCTION_CONTRACT.json"

    atomic_json(
        exporter_path,
        {
            "selected": selected_exporter,
            "unique": exporter_unique,
            "candidates": d1_exporters[:20],
        },
    )
    atomic_json(
        artifact_path,
        {
            "selected": selected_artifact,
            "unique": artifact_unique,
            "candidates": d1_artifacts[:50],
            "selected_exact_13863x69_shape": artifact_exact_shape,
            "selected_has_validation_dimension": artifact_validation_dimension,
            "selected_has_69_logit_dimension": artifact_logit_dimension,
            "array_payloads_loaded": False,
            "headers_only": True,
        },
    )
    atomic_json(
        metric_source_path,
        {
            "selected": selected_metric_source,
            "unique": metric_source_unique,
            "candidates": metric_sources[:30],
        },
    )
    atomic_json(
        metric_eval_path,
        {
            "selected": selected_metric_evaluator,
            "unique": metric_evaluator_unique,
            "candidates": metric_evaluators[:30],
        },
    )

    reproduction_contract = {
        "stage": STAGE,
        "status": "FROZEN" if actual_f4_authorized else "REVIEW_REQUIRED",
        "route": preferred_route,
        "model": {
            "path": str(corrected_model),
            "sha256": sha256_file(corrected_model),
        },
        "loader": {
            "path": str(corrected_loader),
            "sha256": sha256_file(corrected_loader),
            "split": "validation",
            "active_only": False,
            "expected_items": EXPECTED_VALIDATION_ITEMS,
        },
        "exporter": {
            "selected": selected_exporter,
            "unique": exporter_unique,
        },
        "immutable_reference_artifact": {
            "selected": selected_artifact,
            "unique": artifact_unique,
            "exact_13863x69_shape": artifact_exact_shape,
        },
        "metric_source": {
            "selected": selected_metric_source,
            "unique": metric_source_unique,
            "required_coverage": 0.875,
            "strict_exact_required": True,
        },
        "metric_evaluator": {
            "selected": selected_metric_evaluator,
            "unique": metric_evaluator_unique,
        },
        "actual_F4": {
            "replay_checkpoint_on_guarded_validation": True,
            "export_69_raw_logits": True,
            "compare_export_hash_or_numeric_content_to_immutable_reference": True,
            "recompute_metrics_with_canonical_evaluator": True,
            "metric_tolerance": 1e-6,
            "sealed_test_access": False,
            "A_test_access": False,
        },
        "actual_F4_authorized": actual_f4_authorized,
    }
    atomic_json(contract_path, reproduction_contract)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "scope": (
            "Resolve the official F4 route as a two-stage lineage: immutable "
            "D1 validation-logit export followed by the canonical downstream "
            "raw-neural metric evaluator and metric source."
        ),
        "resolved": {
            "exporter_path": selected_exporter["path"],
            "exporter_sha256": selected_exporter["sha256"],
            "exporter_unique": exporter_unique,
            "reference_artifact_path": (
                selected_artifact["path"]
                if selected_artifact is not None
                else None
            ),
            "reference_artifact_sha256": (
                selected_artifact["sha256"]
                if selected_artifact is not None
                else None
            ),
            "reference_artifact_unique": artifact_unique,
            "reference_exact_13863x69": artifact_exact_shape,
            "metric_source_path": (
                selected_metric_source["path"]
                if selected_metric_source is not None
                else None
            ),
            "metric_source_sha256": (
                selected_metric_source["sha256"]
                if selected_metric_source is not None
                else None
            ),
            "metric_source_unique": metric_source_unique,
            "metric_coverage": metric_coverage,
            "metric_ambiguity_count": metric_ambiguity_count,
            "strict_exact_resolved": strict_exact_resolved,
            "metric_evaluator_path": (
                selected_metric_evaluator["path"]
                if selected_metric_evaluator is not None
                else None
            ),
            "metric_evaluator_sha256": (
                selected_metric_evaluator["sha256"]
                if selected_metric_evaluator is not None
                else None
            ),
            "metric_evaluator_unique": metric_evaluator_unique,
        },
        "decision": {
            "F4_P1R2_complete": True,
            "actual_F4_validation_reproduction_authorized": (
                actual_f4_authorized
            ),
            "validation_access_used_in_P1R2": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_"
                "BASELINE_REPRODUCTION"
                if actual_f4_authorized
                else "V5_P3_F4_P1R3_TARGETED_LINEAGE_REVIEW"
            ),
        },
        "artifacts": {
            "exporter_lineage": str(exporter_path),
            "immutable_artifact_lineage": str(artifact_path),
            "canonical_metric_source": str(metric_source_path),
            "downstream_metric_evaluator": str(metric_eval_path),
            "reproduction_contract": str(contract_path),
        },
        "governance": {
            "model_instantiated": False,
            "checkpoint_deserialized": False,
            "array_payloads_loaded": False,
            "array_headers_read": True,
            "training_tensors_loaded": False,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
        },
        "provenance": {
            "F4_P1_report_sha256": sha256_file(p1_report_path),
            "F4_P1_lock_sha256": sha256_file(p1_lock_path),
            "F4_P1R_report_sha256": sha256_file(p1r_report_path),
            "F4_P1R_lock_sha256": sha256_file(p1r_lock_path),
            "corrected_model_sha256": sha256_file(corrected_model),
            "corrected_loader_sha256": sha256_file(corrected_loader),
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
            "exporter_lineage_sha256": sha256_file(exporter_path),
            "immutable_artifact_lineage_sha256": sha256_file(artifact_path),
            "canonical_metric_source_sha256": sha256_file(
                metric_source_path
            ),
            "downstream_metric_evaluator_sha256": sha256_file(
                metric_eval_path
            ),
            "reproduction_contract_sha256": sha256_file(contract_path),
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
    print(f"selected_exporter={selected_exporter['path']}")
    print(f"selected_exporter_sha256={selected_exporter['sha256']}")
    print(f"selected_exporter_unique={str(exporter_unique).lower()}")
    print(
        "selected_reference_artifact="
        f"{selected_artifact['path'] if selected_artifact else None}"
    )
    print(
        "selected_reference_artifact_sha256="
        f"{selected_artifact['sha256'] if selected_artifact else None}"
    )
    print(
        "selected_reference_artifact_unique="
        f"{str(artifact_unique).lower()}"
    )
    print(
        "selected_reference_exact_13863x69="
        f"{str(artifact_exact_shape).lower()}"
    )
    print(
        "selected_reference_has_validation_dimension="
        f"{str(artifact_validation_dimension).lower()}"
    )
    print(
        "selected_reference_has_69_logit_dimension="
        f"{str(artifact_logit_dimension).lower()}"
    )
    print(
        "selected_metric_source="
        f"{selected_metric_source['path'] if selected_metric_source else None}"
    )
    print(
        "selected_metric_source_sha256="
        f"{selected_metric_source['sha256'] if selected_metric_source else None}"
    )
    print(
        "selected_metric_source_unique="
        f"{str(metric_source_unique).lower()}"
    )
    print(f"metric_coverage={metric_coverage:.8f}")
    print(f"metric_ambiguity_count={metric_ambiguity_count}")
    print(f"strict_exact_resolved={str(strict_exact_resolved).lower()}")
    print(
        "selected_metric_evaluator="
        f"{selected_metric_evaluator['path'] if selected_metric_evaluator else None}"
    )
    print(
        "selected_metric_evaluator_sha256="
        f"{selected_metric_evaluator['sha256'] if selected_metric_evaluator else None}"
    )
    print(
        "selected_metric_evaluator_unique="
        f"{str(metric_evaluator_unique).lower()}"
    )
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
    print(f"exporter_lineage={exporter_path}")
    print(f"immutable_artifact_lineage={artifact_path}")
    print(f"canonical_metric_source={metric_source_path}")
    print(f"downstream_metric_evaluator={metric_eval_path}")
    print(f"reproduction_contract={contract_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
