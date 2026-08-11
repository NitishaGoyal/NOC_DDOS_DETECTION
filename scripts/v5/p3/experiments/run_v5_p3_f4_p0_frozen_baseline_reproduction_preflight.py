from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STAGE = "V5_P3_F4_P0_FROZEN_BASELINE_REPRODUCTION_PREFLIGHT"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"

METRIC_TERMS = (
    "selection_score",
    "score",
    "graph_accuracy",
    "graph_acc",
    "graph_auroc",
    "graph_auc",
    "graph_ap",
    "graph_average_precision",
    "graph_f1",
    "graph_fpr",
    "count_macro_f1",
    "count_active_macro_f1",
    "source_ap",
    "source_exact",
    "transit_ap",
    "transit_exact",
    "victim_ap",
    "victim_exact",
    "path_ap",
    "path_exact",
    "strict_exact",
    "strict_all",
)

CODE_TERMS = (
    "evaluate",
    "evaluation",
    "metric",
    "validation",
    "checkpoint",
    "dynamic70",
    "graphconv",
    "v6p0",
    "multitask",
    "selection_score",
    "average_precision",
    "auroc",
    "fpr",
)

ARTIFACT_SUFFIXES = (
    ".json",
    ".csv",
    ".txt",
    ".log",
    ".pt",
    ".pth",
    ".ckpt",
    ".npz",
    ".npy",
    ".py",
    ".sh",
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


def flatten_json(value: Any, prefix: str = "$") -> Iterable[tuple[str, Any]]:
    yield prefix, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_json(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_json(child, f"{prefix}[{index}]")


def safe_read_json(path: Path) -> Any | None:
    try:
        if path.stat().st_size > 64 * 1024 * 1024:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def file_record(path: Path, repo: Path) -> dict[str, Any]:
    stat = path.stat()
    record = {
        "path": str(path),
        "relative_path": str(path.relative_to(repo)),
        "size_bytes": int(stat.st_size),
        "suffix": path.suffix.lower(),
    }
    if stat.st_size <= 2 * 1024 * 1024 * 1024:
        record["sha256"] = sha256_file(path)
    return record


def find_paths(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    results = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        low = str(path).lower()
        if any(pattern in low for pattern in patterns):
            results.append(path)
    return sorted(set(results))


def scan_code(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {
            "path": str(path),
            "readable": False,
        }

    matches = {}
    lower = text.lower()
    for term in CODE_TERMS:
        if term in lower:
            lines = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                if term in line.lower():
                    lines.append({
                        "line": line_number,
                        "text": line.strip()[:500],
                    })
                    if len(lines) >= 8:
                        break
            matches[term] = lines

    imports = []
    class_defs = []
    function_defs = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            if any(
                token in stripped.lower()
                for token in (
                    "model",
                    "metric",
                    "loader",
                    "torch",
                    "sklearn",
                )
            ):
                imports.append({
                    "line": line_number,
                    "text": stripped[:500],
                })
        if stripped.startswith("class "):
            class_defs.append({
                "line": line_number,
                "text": stripped[:500],
            })
        if stripped.startswith("def "):
            if any(
                token in stripped.lower()
                for token in (
                    "eval",
                    "metric",
                    "score",
                    "forward",
                    "load",
                )
            ):
                function_defs.append({
                    "line": line_number,
                    "text": stripped[:500],
                })

    return {
        "path": str(path),
        "readable": True,
        "sha256": sha256_file(path),
        "matches": matches,
        "imports": imports[:50],
        "class_definitions": class_defs[:50],
        "candidate_functions": function_defs[:100],
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")

    study_dir = repo / "reports/v5/p3_experiments/f0_d70_feature_study"
    paired_dir = study_dir / "paired_analysis"
    f3_report_path = paired_dir / (
        "V5_P3_F3_DYNAMIC70_TRAINING_ONLY_MATCHED_"
        "CONTROL_ATTACK_RESPONSE_ANALYSIS_REPORT.json"
    )
    f3_lock_path = paired_dir / (
        "V5_P3_F3_DYNAMIC70_TRAINING_ONLY_MATCHED_"
        "CONTROL_ATTACK_RESPONSE_ANALYSIS_LOCK.json"
    )
    recovery_report_path = paired_dir / (
        "V5_P3_F3_R1_JSON_SERIALIZATION_RECOVERY_"
        "AND_FINALIZATION_REPORT.json"
    )
    recovery_lock_path = paired_dir / (
        "V5_P3_F3_R1_JSON_SERIALIZATION_RECOVERY_"
        "AND_FINALIZATION_LOCK.json"
    )
    f0_lock_path = study_dir / "V5_P3_F0_D70_FEATURE_STUDY_LOCK.json"
    metric_contract_path = study_dir / "METRIC_CONTRACT.json"
    data_access_path = study_dir / "DATA_ACCESS_CONTRACT.json"

    required = [
        f3_report_path,
        f3_lock_path,
        recovery_report_path,
        recovery_lock_path,
        f0_lock_path,
        metric_contract_path,
        data_access_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"required artifacts missing: {missing}")

    f3_report = json.loads(f3_report_path.read_text(encoding="utf-8"))
    f3_lock = json.loads(f3_lock_path.read_text(encoding="utf-8"))
    recovery_report = json.loads(
        recovery_report_path.read_text(encoding="utf-8")
    )
    recovery_lock = json.loads(
        recovery_lock_path.read_text(encoding="utf-8")
    )
    f0_lock = json.loads(f0_lock_path.read_text(encoding="utf-8"))
    data_access = json.loads(data_access_path.read_text(encoding="utf-8"))

    require(f3_report.get("status") == "PASS", "F3 is not PASS")
    require(
        f3_lock.get("report_sha256") == sha256_file(f3_report_path),
        "F3 report/lock mismatch",
    )
    require(
        f3_lock.get("F4_authorized") is True,
        "F3 did not authorize F4",
    )
    require(recovery_report.get("status") == "PASS", "F3-R1 is not PASS")
    require(
        recovery_lock.get("report_sha256")
        == sha256_file(recovery_report_path),
        "F3-R1 report/lock mismatch",
    )
    require(f0_lock.get("status") == "FROZEN", "F0 is not frozen")
    require(
        f0_lock["authorizations"].get("Tranche_A_sealed_test_access")
        is False,
        "sealed-test access unexpectedly authorized",
    )
    require(
        data_access["F4_F6"].get("Tranche_A_validation") is True,
        "F4 validation access not authorized by frozen contract",
    )

    a4_dirs = [
        path
        for path in (repo / "reports/v5").glob("p3_a4*")
        if path.is_dir()
    ]
    a5_dirs = [
        path
        for path in (repo / "reports/v5").glob("p3_a5*")
        if path.is_dir()
    ]
    require(a4_dirs, "no p3_a4 report directory found")
    require(a5_dirs, "no p3_a5 report directory found")

    expected_checkpoint = (
        repo
        / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107/"
        "V5_P3_A4_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_SEED107_BEST.pt"
    )
    checkpoint_candidates = find_paths(
        repo / "reports/v5",
        (
            "p3_a4",
            "dynamic70",
            "seed107",
        ),
    )
    checkpoint_candidates = [
        path
        for path in checkpoint_candidates
        if path.suffix.lower() in (".pt", ".pth", ".ckpt")
    ]
    if expected_checkpoint.is_file():
        checkpoint_path = expected_checkpoint
    else:
        require(
            len(checkpoint_candidates) == 1,
            "could not uniquely identify the frozen A4 checkpoint",
        )
        checkpoint_path = checkpoint_candidates[0]

    a4_a5_files = []
    for root in a4_dirs + a5_dirs:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in ARTIFACT_SUFFIXES:
                a4_a5_files.append(path)

    json_documents = []
    metric_candidates = []
    path_candidates = []
    for path in sorted(
        path for path in a4_a5_files if path.suffix.lower() == ".json"
    ):
        document = safe_read_json(path)
        if document is None:
            continue
        json_documents.append({
            "path": str(path),
            "sha256": sha256_file(path),
        })
        for json_path, value in flatten_json(document):
            low_path = json_path.lower()
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and any(term in low_path for term in METRIC_TERMS)
            ):
                metric_candidates.append({
                    "file": str(path),
                    "json_path": json_path,
                    "value": value,
                })
            if isinstance(value, str):
                low_value = value.lower()
                if (
                    value.endswith((".py", ".pt", ".pth", ".npz", ".npy"))
                    or any(
                        token in low_path
                        for token in (
                            "model",
                            "checkpoint",
                            "loader",
                            "script",
                            "evaluator",
                            "metric",
                            "prediction",
                            "logit",
                        )
                    )
                ):
                    path_candidates.append({
                        "file": str(path),
                        "json_path": json_path,
                        "value": value,
                        "exists_as_written": Path(value).expanduser().exists(),
                    })

    code_roots = [
        repo / "scripts/v5/p3",
        repo / "src/models",
        repo / "src/data",
    ]
    code_candidates = []
    for root in code_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in (".py", ".sh"):
                continue
            low = str(path).lower()
            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).lower()
            except Exception:
                text = ""
            score = 0
            for term in CODE_TERMS:
                if term in low:
                    score += 4
                if term in text:
                    score += 1
            if score > 0:
                code_candidates.append({
                    "path": path,
                    "score": score,
                })

    code_candidates.sort(
        key=lambda row: (row["score"], str(row["path"])),
        reverse=True,
    )
    code_scan = [
        {
            "score": row["score"],
            **scan_code(row["path"]),
        }
        for row in code_candidates[:30]
    ]

    prediction_artifacts = []
    search_roots = [
        repo / "reports/v5",
        repo / "artifacts",
    ]
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            low = str(path).lower()
            if not any(
                token in low
                for token in (
                    "logit",
                    "prediction",
                    "validation_output",
                    "raw_output",
                    "inference",
                )
            ):
                continue
            if path.suffix.lower() not in (
                ".npz",
                ".npy",
                ".pt",
                ".pth",
                ".json",
                ".csv",
            ):
                continue
            prediction_artifacts.append(
                file_record(path, repo)
            )

    model_sources = []
    for root in (repo / "src/models", repo / "scripts/v5/p3"):
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            low = path.name.lower()
            if any(
                token in low
                for token in (
                    "dynamic70",
                    "graphconv",
                    "v6_p0",
                    "v6p0",
                    "p3_a4",
                )
            ):
                model_sources.append(file_record(path, repo))

    loader_candidates = []
    for path in (repo / "src/data").rglob("*.py"):
        low = path.name.lower()
        if "v5_p3" in low or "tranche_a" in low or "dynamic70" in low:
            loader_candidates.append(file_record(path, repo))

    official_evaluator_candidates = [
        row
        for row in code_scan
        if any(
            term in row.get("matches", {})
            for term in (
                "evaluate",
                "evaluation",
                "selection_score",
                "average_precision",
                "auroc",
                "fpr",
            )
        )
    ]

    if official_evaluator_candidates:
        recommended_strategy = (
            "reuse_official_evaluator_with_frozen_checkpoint_and_"
            "guarded_validation_loader"
        )
        actual_f4_authorized = True
    elif prediction_artifacts:
        recommended_strategy = (
            "reproduce_metrics_from_frozen_validation_logits_after_"
            "provenance_verification"
        )
        actual_f4_authorized = True
    else:
        recommended_strategy = (
            "build_explicit_model_forward_and_metric_adapter_after_"
            "manual_preflight_review"
        )
        actual_f4_authorized = False

    inventory_path = output_dir / "F4_P0_A4_A5_ARTIFACT_INVENTORY.json"
    metric_extract_path = output_dir / "F4_P0_FROZEN_METRIC_EXTRACT.json"
    code_scan_path = output_dir / "F4_P0_EVALUATOR_AND_MODEL_CODE_SCAN.json"
    strategy_path = output_dir / "F4_P0_REPRODUCTION_STRATEGY.json"

    atomic_json(
        inventory_path,
        {
            "A4_directories": [str(path) for path in a4_dirs],
            "A5_directories": [str(path) for path in a5_dirs],
            "frozen_checkpoint": file_record(checkpoint_path, repo),
            "A4_A5_files": [
                file_record(path, repo)
                for path in sorted(set(a4_a5_files))
            ],
            "model_sources": model_sources,
            "loader_candidates": loader_candidates,
            "prediction_or_logit_artifacts": prediction_artifacts,
            "JSON_documents": json_documents,
            "JSON_path_candidates": path_candidates,
        },
    )
    atomic_json(
        metric_extract_path,
        {
            "metric_candidates": metric_candidates,
            "count": len(metric_candidates),
        },
    )
    atomic_json(
        code_scan_path,
        {
            "code_candidates": code_scan,
            "official_evaluator_candidates": official_evaluator_candidates,
        },
    )
    atomic_json(
        strategy_path,
        {
            "recommended_strategy": recommended_strategy,
            "actual_F4_evaluation_authorized": actual_f4_authorized,
            "frozen_checkpoint_path": str(checkpoint_path),
            "frozen_checkpoint_sha256": sha256_file(checkpoint_path),
            "validation_item_expectation": 13863,
            "baseline_reproduction_tolerance": float(
                f0_lock["frozen_thresholds"][
                    "baseline_reproduction_tolerance"
                ]
            ),
            "requirements_for_actual_F4": [
                "use the frozen A4 seed-107 checkpoint",
                "use the guarded Tranche-A validation split",
                "use the official A4/A5 metric implementation",
                "load no sealed-test or A-test tensor",
                "compare every frozen metric within 1e-6",
                "write raw metric vector before any perturbation",
            ],
        },
    )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "scope": (
            "Discover and hash the frozen A4/A5 checkpoint, metric records, "
            "official evaluator/model/loader code, and reusable validation "
            "prediction artifacts before executing F4."
        ),
        "frozen_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "discovery": {
            "A4_directories": len(a4_dirs),
            "A5_directories": len(a5_dirs),
            "A4_A5_files": len(a4_a5_files),
            "metric_candidates": len(metric_candidates),
            "model_sources": len(model_sources),
            "loader_candidates": len(loader_candidates),
            "evaluator_candidates": len(official_evaluator_candidates),
            "prediction_artifacts": len(prediction_artifacts),
        },
        "decision": {
            "F4_P0_complete": True,
            "actual_F4_evaluation_authorized": actual_f4_authorized,
            "recommended_strategy": recommended_strategy,
            "validation_access_used_in_preflight": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_"
                "BASELINE_REPRODUCTION"
                if actual_f4_authorized
                else "V5_P3_F4_P1_REPRODUCTION_ADAPTER_REVIEW"
            ),
        },
        "artifacts": {
            "inventory": str(inventory_path),
            "inventory_sha256": sha256_file(inventory_path),
            "metric_extract": str(metric_extract_path),
            "metric_extract_sha256": sha256_file(metric_extract_path),
            "code_scan": str(code_scan_path),
            "code_scan_sha256": sha256_file(code_scan_path),
            "strategy": str(strategy_path),
            "strategy_sha256": sha256_file(strategy_path),
        },
        "governance": {
            "model_loaded": False,
            "checkpoint_deserialized": False,
            "training_tensors_loaded": False,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
        },
        "provenance": {
            "F3_report_sha256": sha256_file(f3_report_path),
            "F3_lock_sha256": sha256_file(f3_lock_path),
            "F3_R1_report_sha256": sha256_file(recovery_report_path),
            "F3_R1_lock_sha256": sha256_file(recovery_lock_path),
            "F0_lock_sha256": sha256_file(f0_lock_path),
            "metric_contract_sha256": sha256_file(metric_contract_path),
            "data_access_contract_sha256": sha256_file(data_access_path),
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
            "inventory_sha256": sha256_file(inventory_path),
            "metric_extract_sha256": sha256_file(metric_extract_path),
            "code_scan_sha256": sha256_file(code_scan_path),
            "strategy_sha256": sha256_file(strategy_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "actual_F4_evaluation_authorized": actual_f4_authorized,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign={CAMPAIGN}")
    print(f"frozen_checkpoint={checkpoint_path}")
    print(f"frozen_checkpoint_sha256={sha256_file(checkpoint_path)}")
    print(f"A4_A5_file_count={len(a4_a5_files)}")
    print(f"metric_candidate_count={len(metric_candidates)}")
    print(f"model_source_count={len(model_sources)}")
    print(f"loader_candidate_count={len(loader_candidates)}")
    print(
        "official_evaluator_candidate_count="
        f"{len(official_evaluator_candidates)}"
    )
    print(
        "prediction_artifact_count="
        f"{len(prediction_artifacts)}"
    )
    print(f"recommended_strategy={recommended_strategy}")
    print(
        "actual_F4_evaluation_authorized="
        f"{str(actual_f4_authorized).lower()}"
    )
    print("model_loaded=false")
    print("checkpoint_deserialized=false")
    print("validation_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={report['decision']['next_stage']}")
    print(f"inventory={inventory_path}")
    print(f"metric_extract={metric_extract_path}")
    print(f"code_scan={code_scan_path}")
    print(f"strategy={strategy_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
