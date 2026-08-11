from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "V5_P3_F4_P1_OFFICIAL_EVALUATOR_ROUTE_AND_ADAPTER_FREEZE"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"

REQUIRED_METRIC_FAMILIES = {
    "selection_score": ("selection_score", "selection score"),
    "graph_accuracy": ("graph_accuracy", "graph_acc", "g_acc"),
    "graph_auroc": ("graph_auroc", "graph_auc", "g_auc"),
    "graph_ap": ("graph_ap", "graph_average_precision", "g_ap"),
    "graph_f1": ("graph_f1", "g_f1"),
    "graph_fpr": ("graph_fpr", "g_fpr"),
    "count_macro_f1": (
        "count_active_macro_f1",
        "count_macro_f1",
        "count_f1_macro",
    ),
    "source_ap": ("source_ap", "src_ap"),
    "source_exact": ("source_exact", "src_exact"),
    "transit_ap": ("transit_ap",),
    "transit_exact": ("transit_exact",),
    "victim_ap": ("victim_ap",),
    "victim_exact": ("victim_exact",),
    "path_ap": ("path_ap",),
    "path_exact": ("path_exact",),
    "strict_exact": ("strict_exact", "strict_all_task_exact"),
}

EVALUATOR_TOKENS = (
    "evaluate",
    "evaluation",
    "validation",
    "metric",
    "selection_score",
    "average_precision",
    "roc_auc",
    "false_positive",
    "fpr",
    "exact",
)

MODEL_TOKENS = (
    "dynamic70",
    "graphconv",
    "v6p0",
    "count4",
    "multitask",
)

LOADER_TOKENS = (
    "guarded",
    "tranche_a",
    "validation",
    "dataset",
    "dynamic70",
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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def safe_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def extract_argparse_options(path: Path) -> list[dict[str, Any]]:
    text = read_text(path)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    options: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "add_argument"
        ):
            continue

        names = []
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                names.append(arg.value)

        kwargs = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            if isinstance(keyword.value, ast.Constant):
                kwargs[keyword.arg] = keyword.value.value
            elif isinstance(keyword.value, (ast.List, ast.Tuple)):
                values = []
                ok = True
                for element in keyword.value.elts:
                    if isinstance(element, ast.Constant):
                        values.append(element.value)
                    else:
                        ok = False
                        break
                if ok:
                    kwargs[keyword.arg] = values
            else:
                kwargs[keyword.arg] = ast.unparse(keyword.value)

        options.append({
            "names": names,
            "kwargs": kwargs,
            "line": int(node.lineno),
        })
    return options


def extract_imports(path: Path) -> list[str]:
    text = read_text(path)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ",".join(alias.name for alias in node.names)
            imports.append(f"{module}:{names}")
    return imports


def extract_definitions(path: Path) -> dict[str, list[dict[str, Any]]]:
    text = read_text(path)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {"classes": [], "functions": []}

    classes = []
    functions = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "line": int(node.lineno),
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({
                "name": node.name,
                "line": int(node.lineno),
                "arguments": [arg.arg for arg in node.args.args],
            })
    return {
        "classes": classes,
        "functions": functions,
    }


def metric_presence(text: str) -> dict[str, bool]:
    low = normalize(text)
    return {
        family: any(normalize(alias) in low for alias in aliases)
        for family, aliases in REQUIRED_METRIC_FAMILIES.items()
    }


def score_evaluator(path: Path, checkpoint_name: str) -> dict[str, Any]:
    text = read_text(path)
    low_path = normalize(str(path))
    low_text = normalize(text)

    score = 0
    reasons = []

    if "scripts_v5_p3" in low_path:
        score += 12
        reasons.append("under scripts/v5/p3")

    if any(token in low_path for token in ("a4", "tranche_a")):
        score += 10
        reasons.append("A4/Tranche-A path")

    if "validation" in low_path:
        score += 8
        reasons.append("validation in path")

    if "evaluate" in low_path or "evaluation" in low_path:
        score += 8
        reasons.append("evaluator naming")

    metric_map = metric_presence(text)
    metric_count = sum(metric_map.values())
    score += metric_count * 3
    if metric_count:
        reasons.append(f"{metric_count} metric families present")

    token_hits = sum(token in low_text for token in EVALUATOR_TOKENS)
    score += token_hits

    model_hits = sum(token in low_text for token in MODEL_TOKENS)
    score += model_hits * 2
    if model_hits:
        reasons.append(f"{model_hits} model-token hits")

    loader_hits = sum(token in low_text for token in LOADER_TOKENS)
    score += loader_hits
    if loader_hits:
        reasons.append(f"{loader_hits} loader-token hits")

    if normalize(checkpoint_name) in low_text:
        score += 20
        reasons.append("exact checkpoint filename referenced")

    if "torch_load" in low_text:
        score += 4
        reasons.append("loads torch checkpoint")

    if any(
        token in low_text
        for token in (
            "split_validation",
            "split_val",
            "validation_loader",
            "val_loader",
        )
    ):
        score += 6
        reasons.append("explicit validation loader")

    options = extract_argparse_options(path)
    option_names = {
        name
        for option in options
        for name in option["names"]
    }
    if any(
        name in option_names
        for name in (
            "--checkpoint",
            "--checkpoint-path",
            "--ckpt",
            "--resume",
        )
    ):
        score += 6
        reasons.append("checkpoint CLI option")

    if any(
        name in option_names
        for name in (
            "--split",
            "--validation-only",
            "--eval-only",
            "--evaluate",
            "--mode",
        )
    ):
        score += 6
        reasons.append("evaluation/split CLI option")

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "score": score,
        "reasons": reasons,
        "metric_presence": metric_map,
        "argparse_options": options,
        "imports": extract_imports(path),
        "definitions": extract_definitions(path),
    }


def score_model(path: Path) -> dict[str, Any]:
    text = read_text(path)
    low = normalize(text + " " + str(path))
    score = 0
    reasons = []

    for token in MODEL_TOKENS:
        if token in low:
            score += 5
            reasons.append(token)

    if "60553" in low:
        score += 15
        reasons.append("parameter count 60553")

    if "69" in low and "logit" in low:
        score += 8
        reasons.append("69-logit output")

    if "graphconv" in low:
        score += 8

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "score": score,
        "reasons": sorted(set(reasons)),
        "definitions": extract_definitions(path),
        "imports": extract_imports(path),
    }


def score_loader(path: Path) -> dict[str, Any]:
    text = read_text(path)
    low = normalize(text + " " + str(path))
    score = 0
    reasons = []

    for token in LOADER_TOKENS:
        if token in low:
            score += 4
            reasons.append(token)

    if "guardedv5p3trancheapreliminarydataset" in low:
        score += 20
        reasons.append("exact guarded dataset class")

    if "split" in low and "validation" in low:
        score += 8
        reasons.append("validation split support")

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "score": score,
        "reasons": sorted(set(reasons)),
        "definitions": extract_definitions(path),
        "imports": extract_imports(path),
    }


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "loaded": False,
            "reason": f"torch import failed: {exc!r}",
        }

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        return {
            "loaded": False,
            "reason": repr(exc),
        }

    record: dict[str, Any] = {
        "loaded": True,
        "payload_type": type(payload).__name__,
    }

    state_dict = None
    metadata = {}

    if isinstance(payload, dict):
        record["top_level_keys"] = sorted(str(key) for key in payload.keys())
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[str(key)] = value

        for key in (
            "model_state_dict",
            "state_dict",
            "model",
            "network",
        ):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                state_dict = candidate
                record["state_dict_key"] = key
                break

        if state_dict is None and payload:
            if all(hasattr(value, "shape") for value in payload.values()):
                state_dict = payload
                record["state_dict_key"] = "__top_level__"

    record["scalar_metadata"] = metadata

    if isinstance(state_dict, dict):
        tensor_rows = []
        parameter_count = 0
        for key, value in state_dict.items():
            if not hasattr(value, "shape"):
                continue
            shape = [int(item) for item in value.shape]
            count = 1
            for dimension in shape:
                count *= dimension
            parameter_count += count
            tensor_rows.append({
                "key": str(key),
                "shape": shape,
                "numel": count,
                "dtype": str(getattr(value, "dtype", "")),
            })
        record["state_dict_tensor_count"] = len(tensor_rows)
        record["parameter_count"] = parameter_count
        record["state_dict_tensors"] = tensor_rows
    return record


def extract_frozen_metrics(metric_extract: dict[str, Any]) -> dict[str, Any]:
    rows = metric_extract.get("metric_candidates", [])
    by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in REQUIRED_METRIC_FAMILIES
    }

    for row in rows:
        search = normalize(
            f"{row.get('file', '')} {row.get('json_path', '')}"
        )
        for family, aliases in REQUIRED_METRIC_FAMILIES.items():
            if any(normalize(alias) in search for alias in aliases):
                by_family[family].append(row)

    resolved = {}
    ambiguous = {}
    missing = []

    for family, family_rows in by_family.items():
        if not family_rows:
            missing.append(family)
            continue

        preferred = sorted(
            family_rows,
            key=lambda row: (
                int("validation" in normalize(row.get("json_path", ""))),
                int("selected" in normalize(row.get("json_path", ""))),
                int("best" in normalize(row.get("json_path", ""))),
                int("a5" in normalize(row.get("file", ""))),
                -len(str(row.get("json_path", ""))),
            ),
            reverse=True,
        )

        top = preferred[0]
        top_key = (
            int("validation" in normalize(top.get("json_path", ""))),
            int("selected" in normalize(top.get("json_path", ""))),
            int("best" in normalize(top.get("json_path", ""))),
            int("a5" in normalize(top.get("file", ""))),
        )
        equally_ranked = []
        for row in preferred:
            key = (
                int("validation" in normalize(row.get("json_path", ""))),
                int("selected" in normalize(row.get("json_path", ""))),
                int("best" in normalize(row.get("json_path", ""))),
                int("a5" in normalize(row.get("file", ""))),
            )
            if key == top_key:
                equally_ranked.append(row)

        unique_values = {
            json.dumps(row.get("value"), sort_keys=True)
            for row in equally_ranked
        }
        if len(unique_values) == 1:
            resolved[family] = top
        else:
            ambiguous[family] = equally_ranked

    return {
        "resolved": resolved,
        "ambiguous": ambiguous,
        "missing": missing,
        "coverage": len(resolved) / len(REQUIRED_METRIC_FAMILIES),
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")

    p0_report_path = output_dir / (
        "V5_P3_F4_P0_FROZEN_BASELINE_REPRODUCTION_PREFLIGHT_REPORT.json"
    )
    p0_lock_path = output_dir / (
        "V5_P3_F4_P0_FROZEN_BASELINE_REPRODUCTION_PREFLIGHT_LOCK.json"
    )
    inventory_path = output_dir / "F4_P0_A4_A5_ARTIFACT_INVENTORY.json"
    metric_extract_path = output_dir / "F4_P0_FROZEN_METRIC_EXTRACT.json"
    code_scan_path = output_dir / "F4_P0_EVALUATOR_AND_MODEL_CODE_SCAN.json"
    strategy_path = output_dir / "F4_P0_REPRODUCTION_STRATEGY.json"

    required = [
        p0_report_path,
        p0_lock_path,
        inventory_path,
        metric_extract_path,
        code_scan_path,
        strategy_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"required P0 artifacts missing: {missing}")

    p0_report = safe_json(p0_report_path)
    p0_lock = safe_json(p0_lock_path)
    inventory = safe_json(inventory_path)
    metric_extract = safe_json(metric_extract_path)
    strategy = safe_json(strategy_path)

    require(p0_report.get("status") == "PASS", "F4-P0 is not PASS")
    require(
        p0_lock.get("report_sha256") == sha256_file(p0_report_path),
        "F4-P0 report/lock mismatch",
    )
    require(
        p0_lock.get("actual_F4_evaluation_authorized") is True,
        "F4-P0 did not authorize route resolution",
    )

    checkpoint_path = Path(
        p0_report["frozen_checkpoint"]["path"]
    ).expanduser().resolve()
    require(checkpoint_path.is_file(), "frozen checkpoint missing")
    require(
        sha256_file(checkpoint_path)
        == p0_report["frozen_checkpoint"]["sha256"],
        "checkpoint hash changed after F4-P0",
    )

    evaluator_paths = []
    for root in (
        repo / "scripts/v5/p3",
        repo / "reports/v5",
    ):
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            low = normalize(str(path))
            text = read_text(path)
            text_low = normalize(text)
            if any(token in low or token in text_low for token in EVALUATOR_TOKENS):
                evaluator_paths.append(path)

    evaluator_rows = [
        score_evaluator(path, checkpoint_path.name)
        for path in sorted(set(evaluator_paths))
    ]
    evaluator_rows.sort(
        key=lambda row: (row["score"], row["path"]),
        reverse=True,
    )

    model_paths = []
    for root in (
        repo / "src/models",
        repo / "scripts/v5/p3",
    ):
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            low = normalize(str(path) + " " + read_text(path))
            if any(token in low for token in MODEL_TOKENS):
                model_paths.append(path)

    model_rows = [
        score_model(path) for path in sorted(set(model_paths))
    ]
    model_rows.sort(
        key=lambda row: (row["score"], row["path"]),
        reverse=True,
    )

    loader_paths = []
    for root in (repo / "src/data", repo / "scripts/v5/p3"):
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            low = normalize(str(path) + " " + read_text(path))
            if any(token in low for token in LOADER_TOKENS):
                loader_paths.append(path)

    loader_rows = [
        score_loader(path) for path in sorted(set(loader_paths))
    ]
    loader_rows.sort(
        key=lambda row: (row["score"], row["path"]),
        reverse=True,
    )

    require(evaluator_rows, "no evaluator candidates found")
    require(model_rows, "no model candidates found")
    require(loader_rows, "no loader candidates found")

    checkpoint_inspection = inspect_checkpoint(checkpoint_path)
    frozen_metrics = extract_frozen_metrics(metric_extract)

    evaluator_unique = (
        len(evaluator_rows) == 1
        or (
            evaluator_rows[0]["score"] >= evaluator_rows[1]["score"] + 8
        )
    )
    evaluator_metric_coverage = sum(
        evaluator_rows[0]["metric_presence"].values()
    ) / len(REQUIRED_METRIC_FAMILIES)
    model_unique = (
        len(model_rows) == 1
        or model_rows[0]["score"] >= model_rows[1]["score"] + 5
    )
    loader_unique = (
        len(loader_rows) == 1
        or loader_rows[0]["score"] >= loader_rows[1]["score"] + 5
    )
    checkpoint_parameter_match = (
        checkpoint_inspection.get("parameter_count") == 60553
    )
    frozen_metric_coverage = float(frozen_metrics["coverage"])

    actual_f4_authorized = (
        evaluator_unique
        and evaluator_metric_coverage >= 0.60
        and model_unique
        and loader_unique
        and checkpoint_inspection.get("loaded") is True
        and checkpoint_parameter_match
        and frozen_metric_coverage >= 0.60
    )

    selected_evaluator = evaluator_rows[0]
    selected_model = model_rows[0]
    selected_loader = loader_rows[0]

    adapter_contract = {
        "stage": STAGE,
        "status": "FROZEN" if actual_f4_authorized else "REVIEW_REQUIRED",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "inspection": checkpoint_inspection,
        },
        "evaluator": {
            "selected": selected_evaluator,
            "runner_mode": (
                "import_and_invoke_official_functions_or_reuse_official_main"
            ),
            "source_modification_authorized": False,
        },
        "model": {
            "selected": selected_model,
            "source_modification_authorized": False,
        },
        "loader": {
            "selected": selected_loader,
            "validation_split": "validation",
            "active_only": False,
            "source_modification_authorized": False,
        },
        "frozen_metrics": frozen_metrics,
        "validation_boundary": {
            "expected_items": 13863,
            "tolerance": float(
                strategy["baseline_reproduction_tolerance"]
            ),
            "sealed_test_access": False,
            "A_test_access": False,
        },
        "actual_F4_authorized": actual_f4_authorized,
    }

    ranking_path = output_dir / "F4_P1_EVALUATOR_MODEL_LOADER_RANKING.json"
    checkpoint_path_out = output_dir / "F4_P1_CHECKPOINT_INSPECTION.json"
    metric_resolution_path = output_dir / "F4_P1_FROZEN_METRIC_RESOLUTION.json"
    adapter_path = output_dir / "F4_P1_OFFICIAL_EVALUATOR_ADAPTER_CONTRACT.json"

    atomic_json(
        ranking_path,
        {
            "evaluator_candidates": evaluator_rows[:25],
            "model_candidates": model_rows[:20],
            "loader_candidates": loader_rows[:20],
            "selection_checks": {
                "evaluator_unique": evaluator_unique,
                "evaluator_metric_coverage": evaluator_metric_coverage,
                "model_unique": model_unique,
                "loader_unique": loader_unique,
            },
        },
    )
    atomic_json(checkpoint_path_out, checkpoint_inspection)
    atomic_json(metric_resolution_path, frozen_metrics)
    atomic_json(adapter_path, adapter_contract)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "scope": (
            "Resolve and freeze the exact official evaluator, model source, "
            "guarded loader, checkpoint structure, and frozen metric vector "
            "before validation tensors are loaded."
        ),
        "selected": {
            "evaluator_path": selected_evaluator["path"],
            "evaluator_sha256": selected_evaluator["sha256"],
            "evaluator_score": selected_evaluator["score"],
            "evaluator_metric_coverage": evaluator_metric_coverage,
            "model_path": selected_model["path"],
            "model_sha256": selected_model["sha256"],
            "model_score": selected_model["score"],
            "loader_path": selected_loader["path"],
            "loader_sha256": selected_loader["sha256"],
            "loader_score": selected_loader["score"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_parameter_count": checkpoint_inspection.get(
                "parameter_count"
            ),
            "frozen_metric_coverage": frozen_metric_coverage,
        },
        "decision": {
            "F4_P1_complete": True,
            "actual_F4_validation_reproduction_authorized": (
                actual_f4_authorized
            ),
            "validation_access_used_in_P1": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_"
                "BASELINE_REPRODUCTION"
                if actual_f4_authorized
                else "V5_P3_F4_P1R_OFFICIAL_EVALUATOR_ROUTE_REVIEW"
            ),
        },
        "artifacts": {
            "ranking": str(ranking_path),
            "ranking_sha256": sha256_file(ranking_path),
            "checkpoint_inspection": str(checkpoint_path_out),
            "checkpoint_inspection_sha256": sha256_file(
                checkpoint_path_out
            ),
            "metric_resolution": str(metric_resolution_path),
            "metric_resolution_sha256": sha256_file(
                metric_resolution_path
            ),
            "adapter_contract": str(adapter_path),
            "adapter_contract_sha256": sha256_file(adapter_path),
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
            "F4_P0_inventory_sha256": sha256_file(inventory_path),
            "F4_P0_metric_extract_sha256": sha256_file(
                metric_extract_path
            ),
            "F4_P0_strategy_sha256": sha256_file(strategy_path),
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
            "ranking_sha256": sha256_file(ranking_path),
            "checkpoint_inspection_sha256": sha256_file(
                checkpoint_path_out
            ),
            "metric_resolution_sha256": sha256_file(
                metric_resolution_path
            ),
            "adapter_contract_sha256": sha256_file(adapter_path),
            "selected_evaluator_sha256": selected_evaluator["sha256"],
            "selected_model_sha256": selected_model["sha256"],
            "selected_loader_sha256": selected_loader["sha256"],
            "checkpoint_sha256": sha256_file(checkpoint_path),
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
    print(f"campaign={CAMPAIGN}")
    print(f"selected_evaluator={selected_evaluator['path']}")
    print(f"selected_evaluator_sha256={selected_evaluator['sha256']}")
    print(f"selected_evaluator_score={selected_evaluator['score']}")
    print(
        "selected_evaluator_metric_coverage="
        f"{evaluator_metric_coverage:.8f}"
    )
    print(f"selected_model={selected_model['path']}")
    print(f"selected_model_sha256={selected_model['sha256']}")
    print(f"selected_model_score={selected_model['score']}")
    print(f"selected_loader={selected_loader['path']}")
    print(f"selected_loader_sha256={selected_loader['sha256']}")
    print(f"selected_loader_score={selected_loader['score']}")
    print(f"checkpoint_parameter_count={checkpoint_inspection.get('parameter_count')}")
    print(f"frozen_metric_coverage={frozen_metric_coverage:.8f}")
    print(
        "actual_F4_validation_reproduction_authorized="
        f"{str(actual_f4_authorized).lower()}"
    )
    print("checkpoint_deserialized_for_structure_only=true")
    print("model_instantiated=false")
    print("validation_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={report['decision']['next_stage']}")
    print(f"ranking={ranking_path}")
    print(f"checkpoint_inspection={checkpoint_path_out}")
    print(f"metric_resolution={metric_resolution_path}")
    print(f"adapter_contract={adapter_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
