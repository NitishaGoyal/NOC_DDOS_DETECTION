from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


STAGE = "V5_P3_F4M_R2_SELECTION_SCORE_FORMULA_RECOVERY_AND_ADAPTER_FREEZE"
CANONICAL_STAGE = "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"

SELECTION_TOLERANCE = 1e-8
METRIC_TOLERANCE = 1e-6

EXCLUDED_PATH_TERMS = (
    "/experiments/",
    "decoder",
    "eplr",
    "milp",
    "posthoc",
    "post_hoc",
)

METRIC_ALIASES = {
    "selection_score": "selection_score",
    "graph_auroc": "graph_auroc",
    "graph_auc": "graph_auroc",
    "g_auc": "graph_auroc",
    "g_auroc": "graph_auroc",
    "graph_ap": "graph_ap",
    "g_ap": "graph_ap",
    "graph_f1_at_0_5": "graph_f1_at_0_5",
    "graph_f1": "graph_f1_at_0_5",
    "g_f1": "graph_f1_at_0_5",
    "graph_fpr_at_0_5": "graph_fpr_at_0_5",
    "graph_fpr": "graph_fpr_at_0_5",
    "g_fpr": "graph_fpr_at_0_5",
    "count_active_macro_f1": "count_active_macro_f1",
    "count_macro_f1": "count_active_macro_f1",
    "active_count_macro_f1": "count_active_macro_f1",
    "source_ap": "source_ap",
    "src_ap": "source_ap",
    "source_exact_active": "source_exact_active",
    "source_exact": "source_exact_active",
    "src_exact_active": "source_exact_active",
    "transit_ap": "transit_ap",
    "transit_exact_active": "transit_exact_active",
    "transit_exact": "transit_exact_active",
    "victim_ap": "victim_ap",
    "victim_exact_active": "victim_exact_active",
    "victim_exact": "victim_exact_active",
    "path_ap": "path_ap",
    "path_exact_active": "path_exact_active",
    "path_exact": "path_exact_active",
    "graph_accuracy": "graph_accuracy",
    "graph_acc": "graph_accuracy",
    "g_acc": "graph_accuracy",
    "graph_precision_at_0_5": "graph_precision_at_0_5",
    "graph_precision": "graph_precision_at_0_5",
    "g_precision": "graph_precision_at_0_5",
    "graph_recall_at_0_5": "graph_recall_at_0_5",
    "graph_recall": "graph_recall_at_0_5",
    "g_recall": "graph_recall_at_0_5",
    "count_active_accuracy": "count_active_accuracy",
    "count_accuracy": "count_active_accuracy",
    "source_precision": "source_precision",
    "source_recall": "source_recall",
    "source_f1": "source_f1",
    "src_precision": "source_precision",
    "src_recall": "source_recall",
    "src_f1": "source_f1",
    "transit_precision": "transit_precision",
    "transit_recall": "transit_recall",
    "transit_f1": "transit_f1",
    "victim_precision": "victim_precision",
    "victim_recall": "victim_recall",
    "victim_f1": "victim_f1",
    "path_precision": "path_precision",
    "path_recall": "path_recall",
    "path_f1": "path_f1",
}

METRIC_DICT_NAMES = (
    "metrics",
    "metric",
    "values",
    "results",
    "result",
    "validation_metrics",
    "val_metrics",
    "stable_metrics",
    "current_metrics",
    "m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-source", required=True)
    parser.add_argument("--installed-adapter", required=True)
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


def import_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "module import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_source_namespace(
    path: Path,
    required_function: str | None = None,
) -> SimpleNamespace:
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text, filename=str(path))

    allowed_import_roots = {
        "__future__",
        "math",
        "statistics",
        "operator",
        "functools",
        "itertools",
        "collections",
        "typing",
        "types",
        "dataclasses",
        "pathlib",
        "numpy",
        "scipy",
        "sklearn",
        "torch",
    }

    safe_nodes: list[ast.stmt] = [
        ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        )
    ]

    for node in tree.body:
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            if roots <= allowed_import_roots:
                safe_nodes.append(copy.deepcopy(node))
            continue

        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in allowed_import_roots:
                safe_nodes.append(copy.deepcopy(node))
            continue

        if isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except Exception:
                continue
            safe_nodes.append(copy.deepcopy(node))
            continue

        if isinstance(node, ast.AnnAssign) and node.value is not None:
            try:
                ast.literal_eval(node.value)
            except Exception:
                continue
            safe_nodes.append(copy.deepcopy(node))
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            copied = copy.deepcopy(node)
            copied.decorator_list = []
            safe_nodes.append(copied)

        if isinstance(node, ast.ClassDef):
            copied = copy.deepcopy(node)
            copied.decorator_list = []
            safe_nodes.append(copied)

    safe_module = ast.Module(body=safe_nodes, type_ignores=[])
    ast.fix_missing_locations(safe_module)

    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "pow": pow,
        "range": range,
        "round": round,
        "set": set,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "isinstance": isinstance,
        "getattr": getattr,
        "hasattr": hasattr,
        "ValueError": ValueError,
        "RuntimeError": RuntimeError,
        "TypeError": TypeError,
    }

    namespace: dict[str, Any] = {
        "__name__": (
            "_v5_p3_f4m_r2_safe_"
            + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
        ),
        "__file__": str(path),
        "__builtins__": safe_builtins,
    }

    code = compile(safe_module, str(path), "exec")
    exec(code, namespace, namespace)

    if required_function is not None:
        require(
            required_function in namespace
            and callable(namespace[required_function]),
            f"safe AST subset did not expose {required_function}",
        )

    return SimpleNamespace(**namespace)


def expand_aliases(metrics: dict[str, float]) -> dict[str, float]:
    expanded = dict(metrics)
    for alias, canonical in METRIC_ALIASES.items():
        if canonical in expanded:
            expanded[alias] = expanded[canonical]
    return expanded


def metric_name_for_parameter(
    parameter_name: str,
    metrics: dict[str, float],
) -> str | None:
    normalized = normalize(parameter_name)
    canonical = METRIC_ALIASES.get(normalized, normalized)
    return canonical if canonical in metrics else None


def expression_environment(
    metrics: dict[str, float],
    module: SimpleNamespace,
) -> dict[str, Any]:
    expanded = expand_aliases(metrics)
    environment = dict(vars(module))
    environment.update(expanded)

    for name in METRIC_DICT_NAMES:
        environment[name] = expanded

    for name in ("ns", "metric_namespace", "metrics_namespace"):
        environment[name] = SimpleNamespace(**expanded)

    environment.update({
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "float": float,
        "int": int,
        "round": round,
        "pow": pow,
        "len": len,
    })
    return environment


def call_function_candidate(
    module: SimpleNamespace,
    function_name: str,
    metrics: dict[str, float],
    call_spec: dict[str, Any],
) -> float:
    function = getattr(module, function_name)
    expanded = expand_aliases(metrics)
    style = call_spec["call_style"]

    if style == "no_arguments":
        value = function()
    elif style == "metrics_dict_positional":
        value = function(expanded)
    elif style == "namespace_positional":
        value = function(SimpleNamespace(**expanded))
    elif style == "expanded_kwargs":
        value = function(**expanded)
    elif style == "signature_mapped_kwargs":
        kwargs = {
            parameter: expanded[metric]
            for parameter, metric in call_spec["parameter_mapping"].items()
        }
        value = function(**kwargs)
    elif style == "signature_mapped_positional":
        values = [
            expanded[metric]
            for metric in call_spec["positional_metric_order"]
        ]
        value = function(*values)
    else:
        raise RuntimeError(f"unsupported call style: {style}")

    numeric = float(value)
    require(np.isfinite(numeric), "candidate returned a nonfinite value")
    return numeric


def call_expression_candidate(
    module: SimpleNamespace,
    expression: str,
    metrics: dict[str, float],
) -> float:
    environment = expression_environment(metrics, module)
    parsed = ast.parse(expression, mode="eval")
    code = compile(parsed, "<selection-expression>", "eval")
    value = eval(code, {"__builtins__": {}}, environment)
    numeric = float(value)
    require(np.isfinite(numeric), "expression returned a nonfinite value")
    return numeric


def metric_dependency_names(text: str) -> list[str]:
    low = normalize(text)
    found = {
        canonical
        for alias, canonical in METRIC_ALIASES.items()
        if canonical != "selection_score" and normalize(alias) in low
    }
    return sorted(found)


def source_roots(repo: Path) -> list[Path]:
    return [
        repo / "scripts/v5/p3",
        repo / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107",
        repo / "reports/v5/p3_a5_tranche_a_review_and_b_handover_readiness",
    ]


def candidate_source_files(repo: Path) -> list[Path]:
    paths = []
    for root in source_roots(repo):
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            low = str(path).lower()
            if any(term in low for term in EXCLUDED_PATH_TERMS):
                continue
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            if "score" not in text and "selection" not in text:
                continue
            paths.append(path)
    return sorted(set(paths))


def function_score(
    path: Path,
    node: ast.FunctionDef,
    source: str,
    expected_literal: str,
) -> tuple[int, list[str]]:
    low_path = str(path).lower()
    low_name = node.name.lower()
    low_source = source.lower()
    score = 0
    reasons = []

    if low_name in ("selection_score", "stable_selection_score"):
        score += 90
        reasons.append("exact selection-score function name")
    elif "selection" in low_name and "score" in low_name:
        score += 70
        reasons.append("selection/score function name")
    elif "score" in low_name:
        score += 20
        reasons.append("score function name")

    if "p3_a4" in low_path or "tranche_a_preliminary_diagnostic" in low_path:
        score += 30
        reasons.append("A4 lineage")
    if "p3_a5" in low_path:
        score += 20
        reasons.append("A5 lineage")
    if expected_literal in low_source:
        score += 15
        reasons.append("contains frozen selection literal")
    dependencies = metric_dependency_names(source)
    if len(dependencies) >= 2:
        score += 20
        reasons.append(f"uses {len(dependencies)} metric names")
    if "return" in low_source:
        score += 3

    return score, reasons


def assignment_target_text(target: ast.AST) -> str:
    try:
        return ast.unparse(target)
    except Exception:
        return ""


def assignment_score(
    path: Path,
    target: str,
    expression: str,
    expected_literal: str,
) -> tuple[int, list[str]]:
    low_path = str(path).lower()
    low_target = normalize(target)
    low_expression = expression.lower()
    score = 0
    reasons = []

    if "selection_score" in low_target:
        score += 90
        reasons.append("selection_score assignment target")
    elif "selection" in low_target and "score" in low_target:
        score += 70
        reasons.append("selection/score assignment target")
    elif low_target.endswith("score") or "stable_score" in low_target:
        score += 25
        reasons.append("score assignment target")

    if "p3_a4" in low_path or "tranche_a_preliminary_diagnostic" in low_path:
        score += 30
        reasons.append("A4 lineage")
    if "p3_a5" in low_path:
        score += 20
        reasons.append("A5 lineage")
    if expected_literal in low_expression:
        score += 10
        reasons.append("contains frozen selection literal")
    dependencies = metric_dependency_names(expression)
    if len(dependencies) >= 2:
        score += 20
        reasons.append(f"uses {len(dependencies)} metric names")

    return score, reasons


def enumerate_candidates(
    repo: Path,
    expected: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    function_records = []
    expression_records = []
    expected_literal = f"{expected:.8f}"

    for path in candidate_source_files(repo):
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                source = ast.get_source_segment(text, node) or ast.unparse(node)
                score, reasons = function_score(
                    path,
                    node,
                    source,
                    expected_literal,
                )
                if score > 0:
                    function_records.append({
                        "kind": "function",
                        "source_path": str(path),
                        "source_sha256": sha256_file(path),
                        "function_name": node.name,
                        "line": int(node.lineno),
                        "source": source[:20000],
                        "score": score,
                        "reasons": reasons,
                        "metric_dependencies": metric_dependency_names(source),
                    })

        for node in ast.walk(tree):
            target = None
            value = None
            if isinstance(node, ast.Assign):
                if len(node.targets) == 1:
                    target = node.targets[0]
                    value = node.value
            elif isinstance(node, ast.AnnAssign):
                target = node.target
                value = node.value
            elif isinstance(node, ast.NamedExpr):
                target = node.target
                value = node.value

            if target is None or value is None:
                continue

            target_text = assignment_target_text(target)
            try:
                expression = ast.unparse(value)
            except Exception:
                continue

            score, reasons = assignment_score(
                path,
                target_text,
                expression,
                expected_literal,
            )
            if score <= 0:
                continue

            expression_records.append({
                "kind": "expression",
                "source_path": str(path),
                "source_sha256": sha256_file(path),
                "line": int(getattr(node, "lineno", 0)),
                "target": target_text,
                "expression": expression,
                "score": score,
                "reasons": reasons,
                "metric_dependencies": metric_dependency_names(expression),
            })

    function_records.sort(
        key=lambda row: (
            row["score"],
            row["source_path"],
            row["function_name"],
        ),
        reverse=True,
    )
    expression_records.sort(
        key=lambda row: (
            row["score"],
            row["source_path"],
            row["line"],
        ),
        reverse=True,
    )
    return function_records, expression_records


def function_call_attempts(
    record: dict[str, Any],
    metrics: dict[str, float],
) -> list[dict[str, Any]]:
    path = Path(record["source_path"]).resolve()
    try:
        module = safe_source_namespace(path, record["function_name"])
    except BaseException as exc:
        return [{
            "status": "SAFE_SUBSET_LOAD_FAILED",
            "error": repr(exc),
        }]

    function = getattr(module, record["function_name"])
    expanded = expand_aliases(metrics)
    attempts = []

    def attempt(call_spec: dict[str, Any]):
        try:
            value = call_function_candidate(
                module,
                record["function_name"],
                metrics,
                call_spec,
            )
            attempts.append({
                "status": "SUCCESS",
                "value": value,
                "call_spec": call_spec,
            })
        except BaseException as exc:
            attempts.append({
                "status": "CALL_FAILED",
                "error": repr(exc),
                "call_spec": call_spec,
            })

    attempt({"call_style": "no_arguments"})
    attempt({"call_style": "metrics_dict_positional"})
    attempt({"call_style": "namespace_positional"})
    attempt({"call_style": "expanded_kwargs"})

    try:
        signature = inspect.signature(function)
        kwargs_mapping = {}
        positional_order = []
        supported = True

        for parameter in signature.parameters.values():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if parameter.default is not inspect.Parameter.empty:
                continue

            normalized = normalize(parameter.name)
            if normalized in METRIC_DICT_NAMES:
                supported = False
                break

            metric_name = metric_name_for_parameter(parameter.name, expanded)
            if metric_name is None:
                supported = False
                break

            kwargs_mapping[parameter.name] = metric_name
            positional_order.append(metric_name)

        if supported:
            attempt({
                "call_style": "signature_mapped_kwargs",
                "parameter_mapping": kwargs_mapping,
            })
            attempt({
                "call_style": "signature_mapped_positional",
                "positional_metric_order": positional_order,
            })
    except BaseException as exc:
        attempts.append({
            "status": "SIGNATURE_PROBE_FAILED",
            "error": repr(exc),
        })

    return attempts


def expression_attempt(
    record: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    path = Path(record["source_path"]).resolve()
    try:
        module = safe_source_namespace(path, None)
        value = call_expression_candidate(
            module,
            record["expression"],
            metrics,
        )
        return {
            "status": "SUCCESS",
            "value": value,
        }
    except BaseException as exc:
        return {
            "status": "EVALUATION_FAILED",
            "error": repr(exc),
        }


def perturbations(metrics: dict[str, float]) -> list[dict[str, float]]:
    variants = []

    for metric, delta in (
        ("graph_ap", 1e-3),
        ("graph_auroc", -1e-3),
        ("graph_fpr_at_0_5", 1e-3),
        ("count_active_macro_f1", -1e-3),
        ("source_ap", 1e-3),
        ("path_ap", -1e-3),
    ):
        if metric not in metrics:
            continue
        changed = dict(metrics)
        changed[metric] = changed[metric] + delta
        variants.append(changed)

    changed_all = dict(metrics)
    for metric in (
        "graph_ap",
        "source_ap",
        "transit_ap",
        "victim_ap",
        "path_ap",
    ):
        if metric in changed_all:
            changed_all[metric] += 5e-4
    variants.append(changed_all)
    return variants


def sensitivity_for_function(
    record: dict[str, Any],
    call_spec: dict[str, Any],
    metrics: dict[str, float],
    baseline: float,
) -> dict[str, Any]:
    path = Path(record["source_path"]).resolve()
    module = safe_source_namespace(path, record["function_name"])
    rows = []

    for index, variant in enumerate(perturbations(metrics)):
        try:
            value = call_function_candidate(
                module,
                record["function_name"],
                variant,
                call_spec,
            )
            rows.append({
                "perturbation_index": index,
                "value": value,
                "absolute_change": abs(value - baseline),
            })
        except BaseException as exc:
            rows.append({
                "perturbation_index": index,
                "error": repr(exc),
            })

    sensitive = any(
        row.get("absolute_change", 0.0) > 1e-12
        for row in rows
    )
    return {
        "sensitive_to_metric_perturbation": sensitive,
        "rows": rows,
    }


def sensitivity_for_expression(
    record: dict[str, Any],
    metrics: dict[str, float],
    baseline: float,
) -> dict[str, Any]:
    path = Path(record["source_path"]).resolve()
    module = safe_source_namespace(path, None)
    rows = []

    for index, variant in enumerate(perturbations(metrics)):
        try:
            value = call_expression_candidate(
                module,
                record["expression"],
                variant,
            )
            rows.append({
                "perturbation_index": index,
                "value": value,
                "absolute_change": abs(value - baseline),
            })
        except BaseException as exc:
            rows.append({
                "perturbation_index": index,
                "error": repr(exc),
            })

    sensitive = any(
        row.get("absolute_change", 0.0) > 1e-12
        for row in rows
    )
    return {
        "sensitive_to_metric_perturbation": sensitive,
        "rows": rows,
    }


def evaluate_candidates(
    functions: list[dict[str, Any]],
    expressions: list[dict[str, Any]],
    metrics: dict[str, float],
    expected: float,
) -> dict[str, Any]:
    evaluated_functions = []
    evaluated_expressions = []
    matches = []

    for record in functions:
        attempts = function_call_attempts(record, metrics)
        result = {
            "record": record,
            "attempts": attempts,
        }
        evaluated_functions.append(result)

        for attempt in attempts:
            if attempt.get("status") != "SUCCESS":
                continue
            difference = abs(float(attempt["value"]) - expected)
            if difference > SELECTION_TOLERANCE:
                continue

            sensitivity = sensitivity_for_function(
                record,
                attempt["call_spec"],
                metrics,
                float(attempt["value"]),
            )
            if not sensitivity["sensitive_to_metric_perturbation"]:
                continue

            matches.append({
                "kind": "function",
                "record": record,
                "value": float(attempt["value"]),
                "absolute_difference": difference,
                "call_spec": attempt["call_spec"],
                "sensitivity": sensitivity,
                "effective_score": record["score"] + 20,
            })

    for record in expressions:
        attempt = expression_attempt(record, metrics)
        result = {
            "record": record,
            "attempt": attempt,
        }
        evaluated_expressions.append(result)

        if attempt.get("status") != "SUCCESS":
            continue
        difference = abs(float(attempt["value"]) - expected)
        if difference > SELECTION_TOLERANCE:
            continue

        sensitivity = sensitivity_for_expression(
            record,
            metrics,
            float(attempt["value"]),
        )
        if not sensitivity["sensitive_to_metric_perturbation"]:
            continue

        matches.append({
            "kind": "expression",
            "record": record,
            "value": float(attempt["value"]),
            "absolute_difference": difference,
            "sensitivity": sensitivity,
            "effective_score": record["score"] + 20,
        })

    # Collapse duplicate call styles for the same function locator.
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in matches:
        if row["kind"] == "function":
            locator = (
                "function",
                row["record"]["source_path"],
                row["record"]["function_name"],
            )
        else:
            locator = (
                "expression",
                row["record"]["source_path"],
                row["record"]["line"],
                row["record"]["expression"],
            )
        grouped.setdefault(locator, []).append(row)

    collapsed = []
    call_style_priority = {
        "signature_mapped_kwargs": 6,
        "signature_mapped_positional": 5,
        "metrics_dict_positional": 4,
        "namespace_positional": 3,
        "no_arguments": 2,
        "expanded_kwargs": 1,
    }

    for locator, rows in grouped.items():
        rows.sort(
            key=lambda row: (
                row["effective_score"],
                call_style_priority.get(
                    row.get("call_spec", {}).get("call_style", ""),
                    0,
                ),
                -row["absolute_difference"],
            ),
            reverse=True,
        )
        selected = dict(rows[0])
        selected["equivalent_matching_attempts"] = rows
        selected["locator"] = locator
        collapsed.append(selected)

    collapsed.sort(
        key=lambda row: (
            row["effective_score"],
            -row["absolute_difference"],
            str(row["locator"]),
        ),
        reverse=True,
    )

    if not collapsed:
        resolution = {
            "status": "UNRESOLVED",
            "reason": (
                "No nonconstant official A4/A5 function or inline expression "
                "reproduced the frozen selection score within 1e-8."
            ),
        }
    else:
        top = collapsed[0]
        if len(collapsed) == 1:
            unique = True
        else:
            unique = (
                top["effective_score"]
                >= collapsed[1]["effective_score"] + 10
            )

        if unique:
            resolution = {
                "status": "RESOLVED",
                "selected": top,
                "runner_up": collapsed[1] if len(collapsed) > 1 else None,
            }
        else:
            resolution = {
                "status": "AMBIGUOUS",
                "top_candidates": collapsed[:10],
                "reason": (
                    "Multiple distinct formula locators reproduce the frozen "
                    "score without a sufficient provenance-score margin."
                ),
            }

    return {
        "function_candidate_count": len(functions),
        "expression_candidate_count": len(expressions),
        "evaluated_functions": evaluated_functions,
        "evaluated_expressions": evaluated_expressions,
        "collapsed_matches": collapsed,
        "resolution": resolution,
    }


def selection_spec_from_resolution(
    resolution: dict[str, Any],
) -> dict[str, Any]:
    require(resolution["status"] == "RESOLVED", "selection formula unresolved")
    selected = resolution["selected"]
    record = selected["record"]

    if selected["kind"] == "function":
        spec = {
            "status": "RESOLVED",
            "kind": "function",
            "source_path": record["source_path"],
            "source_sha256": record["source_sha256"],
            "function_name": record["function_name"],
            "call_style": selected["call_spec"]["call_style"],
            "frozen_probe_value": selected["value"],
            "frozen_probe_absolute_difference": selected[
                "absolute_difference"
            ],
            "metric_dependencies": record["metric_dependencies"],
            "source_line": record["line"],
            "provenance_score": selected["effective_score"],
        }
        for key in ("parameter_mapping", "positional_metric_order"):
            if key in selected["call_spec"]:
                spec[key] = selected["call_spec"][key]
        return spec

    return {
        "status": "RESOLVED",
        "kind": "expression",
        "source_path": record["source_path"],
        "source_sha256": record["source_sha256"],
        "expression": record["expression"],
        "assignment_target": record["target"],
        "source_line": record["line"],
        "frozen_probe_value": selected["value"],
        "frozen_probe_absolute_difference": selected[
            "absolute_difference"
        ],
        "metric_dependencies": record["metric_dependencies"],
        "provenance_score": selected["effective_score"],
    }


def selected_logit_keys(contract: dict[str, Any]) -> dict[str, str]:
    selected = contract["immutable_reference"]["selected_logits"]
    result = {}
    for role, row in selected.items():
        member = row["member"]
        result[role] = member[:-4] if member.endswith(".npy") else member
    return result


def latest_replay_run(workspace: Path) -> Path:
    candidates = [
        path
        for path in workspace.glob("run_*")
        if path.is_dir()
        and (path / "F4_FRESH_VS_IMMUTABLE_NPZ_COMPARISON.json").is_file()
    ]
    require(candidates, f"no F4 replay run found under {workspace}")
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return candidates[0]


def npz_headers(path: Path) -> dict[str, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: {
                "shape": list(archive[key].shape),
                "dtype": str(archive[key].dtype),
            }
            for key in archive.files
        }


def select_fresh_npz(run_dir: Path, reference_path: Path) -> Path:
    reference_headers = npz_headers(reference_path)
    reference_keys = set(reference_headers)
    rows = []

    for path in run_dir.rglob("*.npz"):
        if path.resolve() == reference_path.resolve():
            continue
        try:
            current = npz_headers(path)
        except Exception:
            continue

        shared = set(current) & reference_keys
        rows.append({
            "path": path,
            "exact_keys": set(current) == reference_keys,
            "key_overlap": len(shared),
            "shape_matches": sum(
                current[key]["shape"] == reference_headers[key]["shape"]
                for key in shared
            ),
            "dtype_matches": sum(
                current[key]["dtype"] == reference_headers[key]["dtype"]
                for key in shared
            ),
        })

    require(rows, f"no fresh NPZ under {run_dir}")
    rows.sort(
        key=lambda row: (
            row["exact_keys"],
            row["key_overlap"],
            row["shape_matches"],
            row["dtype_matches"],
            row["path"].stat().st_mtime_ns,
        ),
        reverse=True,
    )
    return rows[0]["path"]


def compare_metrics(
    metrics: dict[str, float],
    frozen: dict[str, float],
) -> dict[str, Any]:
    rows = {}
    all_pass = True

    for metric, frozen_value in frozen.items():
        require(metric in metrics, f"adapter missing metric {metric}")
        value = float(metrics[metric])
        difference = abs(value - float(frozen_value))
        passed = difference <= METRIC_TOLERANCE
        all_pass &= passed
        rows[metric] = {
            "frozen_value": float(frozen_value),
            "adapter_value": value,
            "absolute_difference": difference,
            "tolerance": METRIC_TOLERANCE,
            "pass": passed,
        }

    return {
        "metrics": rows,
        "all_pass": bool(all_pass),
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    adapter_source = Path(args.adapter_source).resolve()
    installed_adapter = Path(args.installed_adapter).resolve()
    installed_script = Path(args.installed_script).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")
    require(adapter_source.is_file(), "package adapter source missing")
    require(installed_adapter.is_file(), "installed adapter missing")
    require(
        sha256_file(adapter_source) == sha256_file(installed_adapter),
        "installed adapter differs from package adapter",
    )

    baseline_dir = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/"
        "baseline_reproduction"
    )
    metric_dir = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/"
        "metric_adapter"
    )

    prior_report_path = metric_dir / (
        "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION_REPORT.json"
    )
    prior_lock_path = metric_dir / (
        "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION_LOCK.json"
    )
    canonical_f4_report_path = baseline_dir / (
        "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_"
        "BASELINE_REPRODUCTION_REPORT.json"
    )
    canonical_f4_lock_path = baseline_dir / (
        "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_"
        "BASELINE_REPRODUCTION_LOCK.json"
    )
    final_contract_path = baseline_dir / (
        "F4_P1R4_FINAL_F4_REPLAY_CONTRACT.json"
    )
    p1r3_contract_path = baseline_dir / (
        "F4_P1R3_ACTUAL_F4_REPRODUCTION_CONTRACT.json"
    )

    required = [
        prior_report_path,
        prior_lock_path,
        canonical_f4_report_path,
        canonical_f4_lock_path,
        final_contract_path,
        p1r3_contract_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"required prior artifacts missing: {missing}")

    prior_report = json.loads(prior_report_path.read_text(encoding="utf-8"))
    prior_lock = json.loads(prior_lock_path.read_text(encoding="utf-8"))
    canonical_f4_report = json.loads(
        canonical_f4_report_path.read_text(encoding="utf-8")
    )
    canonical_f4_lock = json.loads(
        canonical_f4_lock_path.read_text(encoding="utf-8")
    )
    final_contract = json.loads(
        final_contract_path.read_text(encoding="utf-8")
    )
    p1r3_contract = json.loads(
        p1r3_contract_path.read_text(encoding="utf-8")
    )

    require(prior_report.get("status") == "PASS", "prior F4M not PASS")
    require(
        prior_lock.get("report_sha256") == sha256_file(prior_report_path),
        "prior F4M report/lock mismatch",
    )
    require(
        prior_lock.get("selection_score_resolved") is False,
        "F4M-R2 expected unresolved selection score",
    )
    require(
        prior_lock.get("all_14_metrics_certified") is False,
        "prior F4M unexpectedly certified all metrics",
    )
    require(canonical_f4_report.get("status") == "PASS", "canonical F4 not PASS")
    require(
        canonical_f4_lock.get("report_sha256")
        == sha256_file(canonical_f4_report_path),
        "canonical F4 report/lock mismatch",
    )
    require(
        canonical_f4_lock.get("F4M_authorized") is True,
        "canonical F4 did not authorize F4M",
    )
    require(final_contract.get("F5_authorized") is False, "F5 must remain held")

    reference_path = Path(
        final_contract["immutable_reference"]["path"]
    ).resolve()
    require(reference_path.is_file(), "immutable reference NPZ missing")
    require(
        sha256_file(reference_path)
        == final_contract["immutable_reference"]["sha256"],
        "immutable reference hash changed",
    )

    replay_run = latest_replay_run(baseline_dir / "f4_replay_runs")
    fresh_path = select_fresh_npz(replay_run, reference_path)
    logit_keys = selected_logit_keys(p1r3_contract)

    adapter_module = import_module_from_path(
        installed_adapter,
        "_v5_p3_f4m_r2_adapter",
    )

    reference_base = adapter_module.compute_metrics_from_npz(
        reference_path,
        logit_keys,
        selection_spec=None,
    )
    fresh_base = adapter_module.compute_metrics_from_npz(
        fresh_path,
        logit_keys,
        selection_spec=None,
    )

    frozen_metrics = {
        key: float(value)
        for key, value in final_contract["frozen_metric_vector"].items()
    }
    expected_selection = frozen_metrics["selection_score"]

    functions, expressions = enumerate_candidates(
        repo,
        expected_selection,
    )
    evaluation = evaluate_candidates(
        functions,
        expressions,
        reference_base,
        expected_selection,
    )

    candidate_inventory_path = output_dir / (
        "F4M_R2_SELECTION_FORMULA_CANDIDATE_INVENTORY.json"
    )
    evaluation_path = output_dir / (
        "F4M_R2_SELECTION_FORMULA_STATIC_AND_DYNAMIC_EVALUATION.json"
    )
    atomic_json(
        candidate_inventory_path,
        {
            "function_candidates": functions,
            "expression_candidates": expressions,
        },
    )
    atomic_json(evaluation_path, evaluation)

    resolved = evaluation["resolution"]["status"] == "RESOLVED"
    if resolved:
        selection_spec = selection_spec_from_resolution(
            evaluation["resolution"]
        )
    else:
        selection_spec = {
            "status": evaluation["resolution"]["status"],
            "resolution": evaluation["resolution"],
        }

    selection_spec_path = output_dir / (
        "F4M_R2_FROZEN_SELECTION_SCORE_SPEC.json"
    )
    atomic_json(selection_spec_path, selection_spec)

    if resolved:
        reference_metrics = adapter_module.compute_metrics_from_npz(
            reference_path,
            logit_keys,
            selection_spec=selection_spec,
        )
        fresh_metrics = adapter_module.compute_metrics_from_npz(
            fresh_path,
            logit_keys,
            selection_spec=selection_spec,
        )
        reference_comparison = compare_metrics(
            reference_metrics,
            frozen_metrics,
        )
        fresh_comparison = compare_metrics(
            fresh_metrics,
            frozen_metrics,
        )
    else:
        reference_metrics = reference_base
        fresh_metrics = fresh_base
        reference_comparison = compare_metrics(
            reference_metrics,
            {
                key: value
                for key, value in frozen_metrics.items()
                if key != "selection_score"
            },
        )
        fresh_comparison = compare_metrics(
            fresh_metrics,
            {
                key: value
                for key, value in frozen_metrics.items()
                if key != "selection_score"
            },
        )

    reference_cert_path = output_dir / (
        "F4M_R2_IMMUTABLE_REFERENCE_14_METRIC_CERTIFICATION.json"
    )
    fresh_cert_path = output_dir / (
        "F4M_R2_FRESH_REPLAY_14_METRIC_CERTIFICATION.json"
    )
    atomic_json(
        reference_cert_path,
        {
            "npz": str(reference_path),
            "npz_sha256": sha256_file(reference_path),
            "metrics": reference_metrics,
            "comparison": reference_comparison,
        },
    )
    atomic_json(
        fresh_cert_path,
        {
            "npz": str(fresh_path),
            "npz_sha256": sha256_file(fresh_path),
            "metrics": fresh_metrics,
            "comparison": fresh_comparison,
        },
    )

    all_14_certified = bool(
        resolved
        and reference_comparison["all_pass"]
        and fresh_comparison["all_pass"]
        and len(reference_comparison["metrics"]) == 14
        and len(fresh_comparison["metrics"]) == 14
    )

    adapter_contract_path = output_dir / (
        "F4M_R2_CANONICAL_14_METRIC_ADAPTER_CONTRACT.json"
    )
    adapter_contract = {
        "stage": STAGE,
        "status": "FROZEN" if all_14_certified else "REVIEW_REQUIRED",
        "adapter": {
            "path": str(installed_adapter),
            "sha256": sha256_file(installed_adapter),
        },
        "selection_score": selection_spec,
        "selection_score_resolution": evaluation["resolution"],
        "certification": {
            "immutable_reference_pass": reference_comparison["all_pass"],
            "fresh_replay_pass": fresh_comparison["all_pass"],
            "all_14_metrics_certified": all_14_certified,
            "metric_tolerance": METRIC_TOLERANCE,
        },
        "F5_group_block_permutation_authorized": all_14_certified,
        "F6_integrated_gradients_authorized": False,
        "sealed_test_access": False,
        "A_test_access": False,
    }
    atomic_json(adapter_contract_path, adapter_contract)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "scope": (
            "Recover the official selection-score formula from nonexperimental "
            "A4/A5 functions or inline expressions, reject constant-value "
            "false positives by metric-perturbation testing, and freeze the "
            "executable 14-metric adapter."
        ),
        "selection_recovery": {
            "resolved": resolved,
            "resolution": evaluation["resolution"],
            "selection_spec": selection_spec,
        },
        "certification": {
            "immutable_reference": reference_comparison,
            "fresh_replay": fresh_comparison,
            "all_14_metrics_certified": all_14_certified,
        },
        "decision": {
            "F4M_R2_complete": all_14_certified,
            "canonical_F4M_complete": all_14_certified,
            "F5_group_block_permutation_authorized": all_14_certified,
            "F6_integrated_gradients_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F5_DYNAMIC70_GROUP_BLOCK_PERMUTATION"
                if all_14_certified
                else "V5_P3_F4M_R3_MANUAL_SELECTION_FORMULA_PIN"
            ),
        },
        "governance": {
            "candidate_full_module_imported": False,
            "candidate_safe_AST_subset_used": True,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "training_tensors_loaded": False,
            "validation_dataset_tensors_loaded": False,
            "validation_output_artifact_payloads_loaded": True,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "metric_formulas_changed": False,
            "thresholds_changed": False,
        },
        "artifacts": {
            "candidate_inventory": str(candidate_inventory_path),
            "candidate_evaluation": str(evaluation_path),
            "selection_spec": str(selection_spec_path),
            "immutable_reference_certification": str(reference_cert_path),
            "fresh_replay_certification": str(fresh_cert_path),
            "adapter_contract": str(adapter_contract_path),
        },
        "provenance": {
            "prior_F4M_report_sha256": sha256_file(prior_report_path),
            "prior_F4M_lock_sha256": sha256_file(prior_lock_path),
            "canonical_F4_report_sha256": sha256_file(
                canonical_f4_report_path
            ),
            "canonical_F4_lock_sha256": sha256_file(canonical_f4_lock_path),
            "final_F4_contract_sha256": sha256_file(final_contract_path),
            "F4_P1R3_contract_sha256": sha256_file(p1r3_contract_path),
            "adapter_sha256": sha256_file(installed_adapter),
            "installed_script_sha256": sha256_file(installed_script),
            "candidate_inventory_sha256": sha256_file(
                candidate_inventory_path
            ),
            "candidate_evaluation_sha256": sha256_file(evaluation_path),
            "selection_spec_sha256": sha256_file(selection_spec_path),
            "immutable_reference_certification_sha256": sha256_file(
                reference_cert_path
            ),
            "fresh_replay_certification_sha256": sha256_file(
                fresh_cert_path
            ),
            "adapter_contract_sha256": sha256_file(adapter_contract_path),
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
            "adapter_sha256": sha256_file(installed_adapter),
            "candidate_inventory_sha256": sha256_file(
                candidate_inventory_path
            ),
            "candidate_evaluation_sha256": sha256_file(evaluation_path),
            "selection_spec_sha256": sha256_file(selection_spec_path),
            "immutable_reference_certification_sha256": sha256_file(
                reference_cert_path
            ),
            "fresh_replay_certification_sha256": sha256_file(
                fresh_cert_path
            ),
            "adapter_contract_sha256": sha256_file(adapter_contract_path),
            "selection_score_resolved": resolved,
            "all_14_metrics_certified": all_14_certified,
            "F5_authorized": all_14_certified,
            "F6_authorized": False,
            "validation_dataset_tensors_loaded": False,
            "validation_output_artifact_payloads_loaded": True,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    if all_14_certified:
        canonical_report_path = output_dir / (
            f"{CANONICAL_STAGE}_REPORT.json"
        )
        canonical_lock_path = output_dir / (
            f"{CANONICAL_STAGE}_LOCK.json"
        )
        canonical_complete_path = output_dir / (
            f"{CANONICAL_STAGE}_COMPLETE"
        )

        canonical_report = {
            "stage": CANONICAL_STAGE,
            "status": "PASS",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "campaign": CAMPAIGN,
            "classification": (
                "VALIDATION-EXPLORATORY canonical metric-adapter certification"
            ),
            "adapter": {
                "path": str(installed_adapter),
                "sha256": sha256_file(installed_adapter),
            },
            "selection_score": selection_spec,
            "certification": {
                "immutable_reference_pass": True,
                "fresh_replay_pass": True,
                "all_14_metrics_certified": True,
            },
            "decision": {
                "F4M_complete": True,
                "F5_group_block_permutation_authorized": True,
                "F6_integrated_gradients_authorized": False,
                "sealed_test_access_authorized": False,
                "next_stage": (
                    "V5_P3_F5_DYNAMIC70_GROUP_BLOCK_PERMUTATION"
                ),
            },
            "provenance": {
                "F4M_R2_report_sha256": sha256_file(report_path),
                "F4M_R2_lock_sha256": sha256_file(lock_path),
                "adapter_contract_sha256": sha256_file(
                    adapter_contract_path
                ),
                "selection_spec_sha256": sha256_file(selection_spec_path),
            },
        }
        atomic_json(canonical_report_path, canonical_report)
        atomic_json(
            canonical_lock_path,
            {
                "stage": CANONICAL_STAGE,
                "status": "PASS",
                "report_sha256": sha256_file(canonical_report_path),
                "F4M_R2_report_sha256": sha256_file(report_path),
                "F4M_R2_lock_sha256": sha256_file(lock_path),
                "adapter_sha256": sha256_file(installed_adapter),
                "selection_spec_sha256": sha256_file(selection_spec_path),
                "adapter_contract_sha256": sha256_file(
                    adapter_contract_path
                ),
                "all_14_metrics_certified": True,
                "F5_authorized": True,
                "F6_authorized": False,
                "sealed_test_tensors_loaded": False,
            },
        )
        atomic_text(
            canonical_complete_path,
            f"{CANONICAL_STAGE}_COMPLETE\n",
        )
    else:
        canonical_report_path = None
        canonical_lock_path = None

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(
        "function_candidate_count="
        f"{evaluation['function_candidate_count']}"
    )
    print(
        "expression_candidate_count="
        f"{evaluation['expression_candidate_count']}"
    )
    print(f"selection_score_resolved={str(resolved).lower()}")
    if resolved:
        print(f"selection_score_kind={selection_spec['kind']}")
        print(
            "selection_score_source="
            f"{selection_spec['source_path']}"
        )
        print(
            "selection_score_source_sha256="
            f"{selection_spec['source_sha256']}"
        )
        print(
            "selection_score_function="
            f"{selection_spec.get('function_name')}"
        )
        print(
            "selection_score_expression="
            f"{selection_spec.get('expression')}"
        )
        print(
            "selection_score_call_style="
            f"{selection_spec.get('call_style')}"
        )
        print(
            "selection_score_metric_dependencies="
            f"{selection_spec.get('metric_dependencies')}"
        )
    else:
        print(
            "selection_resolution_status="
            f"{evaluation['resolution']['status']}"
        )
    print(
        "immutable_reference_adapter_pass="
        f"{str(reference_comparison['all_pass']).lower()}"
    )
    print(
        "fresh_replay_adapter_pass="
        f"{str(fresh_comparison['all_pass']).lower()}"
    )
    print(
        "all_14_metrics_certified="
        f"{str(all_14_certified).lower()}"
    )
    print(f"F4M_complete={str(all_14_certified).lower()}")
    print(
        "F5_group_block_permutation_authorized="
        f"{str(all_14_certified).lower()}"
    )
    print("F6_integrated_gradients_authorized=false")
    print("candidate_full_module_imported=false")
    print("candidate_safe_AST_subset_used=true")
    print("model_loaded=false")
    print("checkpoint_loaded=false")
    print("validation_dataset_tensors_loaded=false")
    print("validation_output_artifact_payloads_loaded=true")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={report['decision']['next_stage']}")
    print(f"candidate_inventory={candidate_inventory_path}")
    print(f"candidate_evaluation={evaluation_path}")
    print(f"selection_spec={selection_spec_path}")
    print(f"immutable_reference_certification={reference_cert_path}")
    print(f"fresh_replay_certification={fresh_cert_path}")
    print(f"adapter_contract={adapter_contract_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    if canonical_report_path is not None:
        print(f"{CANONICAL_STAGE}_COMPLETE")
        print("canonical_F4M_status=PASS")
        print("all_14_metrics_certified=true")
        print("F5_group_block_permutation_authorized=true")
        print(f"canonical_F4M_report={canonical_report_path}")
        print(f"canonical_F4M_lock={canonical_lock_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
