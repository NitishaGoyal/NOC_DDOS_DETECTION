from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


STAGE = "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
METRIC_TOLERANCE = 1e-6

BASE_METRICS = (
    "graph_auroc",
    "graph_ap",
    "graph_f1_at_0_5",
    "graph_fpr_at_0_5",
    "count_active_macro_f1",
    "source_ap",
    "source_exact_active",
    "transit_ap",
    "transit_exact_active",
    "victim_ap",
    "victim_exact_active",
    "path_ap",
    "path_exact_active",
)

SELECTION_NAME_TERMS = (
    "selection_score",
    "selection",
    "score",
    "stable_score",
    "checkpoint_score",
)

EXCLUDE_PATH_TERMS = (
    "/experiments/",
    "decoder",
    "eplr",
    "milp",
    "posthoc",
    "post_hoc",
)

METRIC_ALIASES = {
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
    "graph_recall_at_0_5": "graph_recall_at_0_5",
    "graph_recall": "graph_recall_at_0_5",
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
    import re
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def import_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "module import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_logit_keys(contract: dict[str, Any]) -> dict[str, str]:
    selected = contract["immutable_reference"]["selected_logits"]
    result = {}
    for role, row in selected.items():
        member = row["member"]
        result[role] = member[:-4] if member.endswith(".npy") else member
    return result


def latest_failed_run(workspace: Path) -> Path:
    candidates = []
    for path in workspace.glob("run_*"):
        if not path.is_dir():
            continue
        if (path / "F4_FRESH_VS_IMMUTABLE_NPZ_COMPARISON.json").is_file():
            candidates.append(path)
    require(candidates, f"no F4 replay run found under {workspace}")
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return candidates[0]


def npz_headers(path: Path) -> dict[str, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: {
                "shape": [int(item) for item in archive[key].shape],
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

    require(rows, f"no fresh NPZ found under {run_dir}")
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


def expand_aliases(metrics: dict[str, float]) -> dict[str, float]:
    expanded = dict(metrics)
    for alias, canonical in METRIC_ALIASES.items():
        if canonical in expanded:
            expanded[alias] = expanded[canonical]
    return expanded


def candidate_function_records(repo: Path) -> list[dict[str, Any]]:
    records = []

    roots = [
        repo / "scripts/v5/p3",
        repo / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107",
        repo / "reports/v5/p3_a5_tranche_a_review_and_b_handover_readiness",
    ]

    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            path_text = str(path)
            low_path = path_text.lower()
            if any(term in low_path for term in EXCLUDE_PATH_TERMS):
                continue

            text = path.read_text(encoding="utf-8", errors="replace")
            low_text = text.lower()
            if "selection" not in low_text or "score" not in low_text:
                continue

            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue

            for node in tree.body:
                if not isinstance(node, ast.FunctionDef):
                    continue

                source = ast.get_source_segment(text, node) or ""
                low_name = node.name.lower()
                low_source = source.lower()

                score = 0
                reasons = []

                if "selection" in low_name and "score" in low_name:
                    score += 50
                    reasons.append("selection/score function name")
                if "selection_score" in low_source:
                    score += 20
                    reasons.append("selection_score in function source")
                if "p3_a4" in low_path or "tranche_a_preliminary_diagnostic" in low_path:
                    score += 20
                    reasons.append("A4 lineage")
                if "p3_a5" in low_path:
                    score += 10
                    reasons.append("A5 lineage")
                if "stable" in low_name or "stable" in low_source:
                    score += 5
                    reasons.append("stable naming")
                if "return" in low_source:
                    score += 2

                if score > 0:
                    records.append({
                        "path": path_text,
                        "source_sha256": sha256_file(path),
                        "function_name": node.name,
                        "line": int(node.lineno),
                        "score": score,
                        "reasons": reasons,
                        "signature": {
                            "args": [arg.arg for arg in node.args.args],
                            "kwonlyargs": [arg.arg for arg in node.args.kwonlyargs],
                            "vararg": (
                                node.args.vararg.arg
                                if node.args.vararg is not None
                                else None
                            ),
                            "kwarg": (
                                node.args.kwarg.arg
                                if node.args.kwarg is not None
                                else None
                            ),
                        },
                        "source": source[:12000],
                    })

    records.sort(
        key=lambda row: (row["score"], row["path"], row["function_name"]),
        reverse=True,
    )
    return records


def metric_name_for_parameter(name: str, metrics: dict[str, float]) -> str | None:
    normalized = normalize(name)
    if normalized in METRIC_ALIASES:
        canonical = METRIC_ALIASES[normalized]
        if canonical in metrics:
            return canonical
    if normalized in metrics:
        return normalized
    return None


def probe_function(
    record: dict[str, Any],
    metrics: dict[str, float],
    expected: float,
) -> list[dict[str, Any]]:
    path = Path(record["path"]).resolve()
    module_name = "_f4m_probe_" + record["source_sha256"][:12]
    try:
        module = import_module_from_path(path, module_name)
    except Exception as exc:
        return [{
            "status": "IMPORT_FAILED",
            "error": repr(exc),
        }]

    if not hasattr(module, record["function_name"]):
        return [{
            "status": "FUNCTION_MISSING_AFTER_IMPORT",
        }]

    function = getattr(module, record["function_name"])
    expanded = expand_aliases(metrics)
    attempts = []

    def attempt(call_style: str, callback, extra: dict[str, Any] | None = None):
        try:
            value = float(callback())
            attempts.append({
                "status": "SUCCESS",
                "call_style": call_style,
                "value": value,
                "absolute_difference": abs(value - expected),
                "matches_expected": abs(value - expected) <= 1e-8,
                **(extra or {}),
            })
        except Exception as exc:
            attempts.append({
                "status": "CALL_FAILED",
                "call_style": call_style,
                "error": repr(exc),
                **(extra or {}),
            })

    attempt(
        "metrics_dict_positional",
        lambda: function(expanded),
    )
    attempt(
        "namespace_positional",
        lambda: function(SimpleNamespace(**expanded)),
    )
    attempt(
        "expanded_kwargs",
        lambda: function(**expanded),
    )

    try:
        signature = inspect.signature(function)
        kwargs = {}
        positional = []
        positional_metric_order = []
        kwargs_mapping = {}
        supported_kwargs = True
        supported_positional = True

        for parameter in signature.parameters.values():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if parameter.default is not inspect.Parameter.empty:
                continue

            normalized = normalize(parameter.name)
            if normalized in ("metrics", "metric", "results", "result", "values"):
                kwargs[parameter.name] = expanded
                positional.append(expanded)
                positional_metric_order.append("__metrics_dict__")
                kwargs_mapping[parameter.name] = "__metrics_dict__"
                continue

            metric_name = metric_name_for_parameter(parameter.name, expanded)
            if metric_name is None:
                supported_kwargs = False
                supported_positional = False
                break

            kwargs[parameter.name] = expanded[metric_name]
            positional.append(expanded[metric_name])
            positional_metric_order.append(metric_name)
            kwargs_mapping[parameter.name] = metric_name

        if supported_kwargs:
            attempt(
                "signature_mapped_kwargs",
                lambda: function(**kwargs),
                {"parameter_mapping": kwargs_mapping},
            )
        if supported_positional:
            attempt(
                "signature_mapped_positional",
                lambda: function(*positional),
                {"positional_metric_order": positional_metric_order},
            )
    except Exception as exc:
        attempts.append({
            "status": "SIGNATURE_PROBE_FAILED",
            "error": repr(exc),
        })

    return attempts


def resolve_selection_spec(
    repo: Path,
    metrics: dict[str, float],
    expected: float,
) -> dict[str, Any]:
    candidates = candidate_function_records(repo)
    evaluated = []

    for record in candidates:
        attempts = probe_function(record, metrics, expected)
        evaluated.append({
            "record": record,
            "attempts": attempts,
        })

    matches = []
    for row in evaluated:
        for attempt in row["attempts"]:
            if attempt.get("matches_expected") is True:
                matches.append({
                    "record": row["record"],
                    "attempt": attempt,
                })

    if not matches:
        return {
            "status": "UNRESOLVED",
            "candidate_count": len(candidates),
            "evaluated": evaluated,
        }

    matches.sort(
        key=lambda row: (
            row["record"]["score"],
            -row["attempt"]["absolute_difference"],
            row["record"]["path"],
            row["record"]["function_name"],
        ),
        reverse=True,
    )

    selected = matches[0]
    top_score = selected["record"]["score"]
    top_matches = [
        row for row in matches
        if row["record"]["score"] == top_score
    ]

    spec = {
        "status": "RESOLVED",
        "source_path": selected["record"]["path"],
        "source_sha256": selected["record"]["source_sha256"],
        "function_name": selected["record"]["function_name"],
        "call_style": selected["attempt"]["call_style"],
        "frozen_probe_value": selected["attempt"]["value"],
        "frozen_probe_absolute_difference": selected["attempt"][
            "absolute_difference"
        ],
        "record_score": selected["record"]["score"],
        "record_reasons": selected["record"]["reasons"],
        "equivalent_matching_candidates": top_matches,
        "all_evaluated_candidates": evaluated,
    }

    if "parameter_mapping" in selected["attempt"]:
        mapping = selected["attempt"]["parameter_mapping"]
        if "__metrics_dict__" in mapping.values():
            # Adapter call style for this case is a plain metrics-dict positional
            # call, which was already probed separately. Do not freeze an
            # ambiguous mixed mapping.
            return {
                "status": "UNRESOLVED_MIXED_SIGNATURE_MAPPING",
                "selected_probe": spec,
            }
        spec["parameter_mapping"] = mapping

    if "positional_metric_order" in selected["attempt"]:
        order = selected["attempt"]["positional_metric_order"]
        if "__metrics_dict__" in order:
            return {
                "status": "UNRESOLVED_MIXED_SIGNATURE_MAPPING",
                "selected_probe": spec,
            }
        spec["positional_metric_order"] = order

    return spec


def compare_metrics(
    metrics: dict[str, float],
    frozen: dict[str, float],
) -> dict[str, Any]:
    rows = {}
    all_pass = True

    for metric, frozen_value in frozen.items():
        require(metric in metrics, f"adapter missing metric: {metric}")
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
        "installed adapter hash differs from package adapter",
    )

    baseline_dir = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/"
        "baseline_reproduction"
    )

    canonical_f4_report_path = baseline_dir / (
        "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_"
        "BASELINE_REPRODUCTION_REPORT.json"
    )
    canonical_f4_lock_path = baseline_dir / (
        "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_"
        "BASELINE_REPRODUCTION_LOCK.json"
    )
    r5_report_path = baseline_dir / (
        "V5_P3_F4_R5_TARGETED_METRIC_DELTA_RECONSTRUCTION_REPORT.json"
    )
    r5_lock_path = baseline_dir / (
        "V5_P3_F4_R5_TARGETED_METRIC_DELTA_RECONSTRUCTION_LOCK.json"
    )
    final_contract_path = baseline_dir / "F4_P1R4_FINAL_F4_REPLAY_CONTRACT.json"
    p1r3_contract_path = baseline_dir / (
        "F4_P1R3_ACTUAL_F4_REPRODUCTION_CONTRACT.json"
    )

    required = [
        canonical_f4_report_path,
        canonical_f4_lock_path,
        r5_report_path,
        r5_lock_path,
        final_contract_path,
        p1r3_contract_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"required F4 artifacts missing: {missing}")

    canonical_f4_report = json.loads(
        canonical_f4_report_path.read_text(encoding="utf-8")
    )
    canonical_f4_lock = json.loads(
        canonical_f4_lock_path.read_text(encoding="utf-8")
    )
    r5_report = json.loads(r5_report_path.read_text(encoding="utf-8"))
    r5_lock = json.loads(r5_lock_path.read_text(encoding="utf-8"))
    final_contract = json.loads(
        final_contract_path.read_text(encoding="utf-8")
    )
    p1r3_contract = json.loads(
        p1r3_contract_path.read_text(encoding="utf-8")
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
    require(r5_report.get("status") == "PASS", "F4-R5 not PASS")
    require(
        r5_lock.get("report_sha256") == sha256_file(r5_report_path),
        "F4-R5 report/lock mismatch",
    )
    require(
        r5_lock.get("all_14_frozen_metrics_reproduced") is True,
        "F4-R5 did not reproduce all frozen metrics",
    )
    require(final_contract.get("F5_authorized") is False, "F5 must still be held")

    reference_path = Path(
        final_contract["immutable_reference"]["path"]
    ).resolve()
    require(reference_path.is_file(), "immutable reference missing")
    require(
        sha256_file(reference_path)
        == final_contract["immutable_reference"]["sha256"],
        "immutable reference hash changed",
    )

    run_dir = latest_failed_run(baseline_dir / "f4_replay_runs")
    fresh_path = select_fresh_npz(run_dir, reference_path)
    logit_keys = selected_logit_keys(p1r3_contract)

    adapter_module = import_module_from_path(
        installed_adapter,
        "_v5_p3_f4m_canonical_metric_adapter",
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
    frozen_selection_score = frozen_metrics["selection_score"]

    selection_spec = resolve_selection_spec(
        repo,
        reference_base,
        frozen_selection_score,
    )

    selection_spec_path = output_dir / (
        "F4M_FROZEN_SELECTION_SCORE_FUNCTION_SPEC.json"
    )
    atomic_json(selection_spec_path, selection_spec)

    selection_resolved = selection_spec.get("status") == "RESOLVED"

    if selection_resolved:
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

    reference_path_out = output_dir / (
        "F4M_IMMUTABLE_REFERENCE_ADAPTER_CERTIFICATION.json"
    )
    fresh_path_out = output_dir / (
        "F4M_FRESH_REPLAY_ADAPTER_CERTIFICATION.json"
    )
    adapter_contract_path = output_dir / (
        "F4M_CANONICAL_METRIC_ADAPTER_CONTRACT.json"
    )

    atomic_json(
        reference_path_out,
        {
            "npz": str(reference_path),
            "npz_sha256": sha256_file(reference_path),
            "metrics": reference_metrics,
            "comparison_to_frozen": reference_comparison,
        },
    )
    atomic_json(
        fresh_path_out,
        {
            "npz": str(fresh_path),
            "npz_sha256": sha256_file(fresh_path),
            "metrics": fresh_metrics,
            "comparison_to_frozen": fresh_comparison,
        },
    )

    all_14_certified = bool(
        selection_resolved
        and reference_comparison["all_pass"]
        and fresh_comparison["all_pass"]
        and len(reference_comparison["metrics"]) == 14
        and len(fresh_comparison["metrics"]) == 14
    )

    contract = {
        "stage": STAGE,
        "status": "FROZEN" if all_14_certified else "REVIEW_REQUIRED",
        "adapter": {
            "path": str(installed_adapter),
            "sha256": sha256_file(installed_adapter),
        },
        "logit_keys": logit_keys,
        "base_metric_formulas": {
            "graph_auroc": "roc_auc_score(y_attack, raw attack logits)",
            "graph_ap": "average_precision_score(y_attack, raw attack logits)",
            "graph_f1_at_0_5": "binary F1 using attack_logits >= 0",
            "graph_fpr_at_0_5": "false-positive rate using attack_logits >= 0",
            "count_active_macro_f1": (
                "active samples; y_attacker_count-1; argmax count logits; "
                "macro F1 over labels [0,1,2,3]"
            ),
            "role_ap": (
                "flatten all validation samples and all 16 nodes; "
                "average_precision_score(target, raw role logits)"
            ),
            "role_exact_active": (
                "active samples only; all 16 nodes; role_logits >= 0; "
                "empty/no-role active samples count as non-exact"
            ),
        },
        "selection_score": selection_spec,
        "certification": {
            "immutable_reference_pass": reference_comparison["all_pass"],
            "fresh_replay_pass": fresh_comparison["all_pass"],
            "all_14_metrics_certified": all_14_certified,
            "metric_tolerance": METRIC_TOLERANCE,
        },
        "F5_permutation_authorized": all_14_certified,
        "sealed_test_access": False,
        "A_test_access": False,
    }
    atomic_json(adapter_contract_path, contract)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": (
            "VALIDATION-EXPLORATORY metric-adapter certification"
        ),
        "scope": (
            "Freeze an executable metric adapter for all 14 F4/F5 metrics, "
            "including the exact official A4/A5 selection-score function, "
            "and certify it on both the immutable and fresh validation NPZs."
        ),
        "adapter": {
            "path": str(installed_adapter),
            "sha256": sha256_file(installed_adapter),
        },
        "selection_score": {
            "resolved": selection_resolved,
            "specification": selection_spec,
        },
        "certification": {
            "immutable_reference": reference_comparison,
            "fresh_replay": fresh_comparison,
            "all_14_metrics_certified": all_14_certified,
        },
        "decision": {
            "F4M_complete": all_14_certified,
            "F5_group_block_permutation_authorized": all_14_certified,
            "F6_integrated_gradients_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F5_DYNAMIC70_GROUP_BLOCK_PERMUTATION"
                if all_14_certified
                else "V5_P3_F4M_R1_SELECTION_SCORE_FORMULA_REVIEW"
            ),
        },
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "training_tensors_loaded": False,
            "validation_dataset_tensors_loaded": False,
            "validation_output_artifact_payloads_loaded": True,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "thresholds_changed": False,
            "metric_formulas_changed": False,
        },
        "artifacts": {
            "selection_score_spec": str(selection_spec_path),
            "immutable_reference_certification": str(reference_path_out),
            "fresh_replay_certification": str(fresh_path_out),
            "adapter_contract": str(adapter_contract_path),
        },
        "provenance": {
            "canonical_F4_report_sha256": sha256_file(
                canonical_f4_report_path
            ),
            "canonical_F4_lock_sha256": sha256_file(canonical_f4_lock_path),
            "F4_R5_report_sha256": sha256_file(r5_report_path),
            "F4_R5_lock_sha256": sha256_file(r5_lock_path),
            "final_F4_contract_sha256": sha256_file(final_contract_path),
            "F4_P1R3_contract_sha256": sha256_file(p1r3_contract_path),
            "adapter_sha256": sha256_file(installed_adapter),
            "installed_script_sha256": sha256_file(installed_script),
            "selection_score_spec_sha256": sha256_file(selection_spec_path),
            "immutable_reference_certification_sha256": sha256_file(
                reference_path_out
            ),
            "fresh_replay_certification_sha256": sha256_file(
                fresh_path_out
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
            "selection_score_spec_sha256": sha256_file(selection_spec_path),
            "immutable_reference_certification_sha256": sha256_file(
                reference_path_out
            ),
            "fresh_replay_certification_sha256": sha256_file(
                fresh_path_out
            ),
            "adapter_contract_sha256": sha256_file(adapter_contract_path),
            "selection_score_resolved": selection_resolved,
            "all_14_metrics_certified": all_14_certified,
            "F5_authorized": all_14_certified,
            "F6_authorized": False,
            "validation_dataset_tensors_loaded": False,
            "validation_output_artifact_payloads_loaded": True,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"adapter={installed_adapter}")
    print(f"adapter_sha256={sha256_file(installed_adapter)}")
    print(f"selection_score_resolved={str(selection_resolved).lower()}")
    print(
        "selection_score_source="
        f"{selection_spec.get('source_path')}"
    )
    print(
        "selection_score_function="
        f"{selection_spec.get('function_name')}"
    )
    print(
        "selection_score_call_style="
        f"{selection_spec.get('call_style')}"
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
    print(
        "F4M_complete="
        f"{str(all_14_certified).lower()}"
    )
    print(
        "F5_group_block_permutation_authorized="
        f"{str(all_14_certified).lower()}"
    )
    print("F6_integrated_gradients_authorized=false")
    print("model_loaded=false")
    print("checkpoint_loaded=false")
    print("validation_dataset_tensors_loaded=false")
    print("validation_output_artifact_payloads_loaded=true")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={report['decision']['next_stage']}")
    print(f"selection_score_spec={selection_spec_path}")
    print(f"immutable_reference_certification={reference_path_out}")
    print(f"fresh_replay_certification={fresh_path_out}")
    print(f"adapter_contract={adapter_contract_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
