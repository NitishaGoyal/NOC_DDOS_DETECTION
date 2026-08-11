from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STAGE = "V5_P3_F4M_R3_MANUAL_SELECTION_FORMULA_PROVENANCE_DOSSIER"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"

TARGET_SYMBOLS = (
    "selection_score",
    "select_graph_threshold_fpr_cap",
)

KNOWN_CALLSITE_RELATIVE = Path(
    "scripts/v4/train/train_v4_a3_sourcepreserve.py"
)

TEXT_SUFFIXES = (
    ".json",
    ".log",
    ".txt",
    ".md",
    ".yaml",
    ".yml",
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


def json_default(value: Any) -> Any:
    try:
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass

    try:
        import torch
        if isinstance(value, torch.Tensor):
            if value.numel() <= 256:
                return value.detach().cpu().tolist()
            return {
                "tensor_shape": list(value.shape),
                "tensor_dtype": str(value.dtype),
                "tensor_numel": int(value.numel()),
            }
    except Exception:
        pass

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=True,
            default=json_default,
        )
        + "\n",
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


def source_segment(
    path: Path,
    node: ast.AST,
    context_lines: int = 0,
) -> dict[str, Any]:
    text = read_text(path)
    lines = text.splitlines()

    start = max(1, int(getattr(node, "lineno", 1)) - context_lines)
    end_node = int(getattr(node, "end_lineno", getattr(node, "lineno", 1)))
    end = min(len(lines), end_node + context_lines)

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "line_start": start,
        "line_end": end,
        "source": "\n".join(lines[start - 1:end]),
    }


def parse_tree(path: Path) -> ast.Module:
    return ast.parse(read_text(path), filename=str(path))


def function_definitions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = parse_tree(path)
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


def find_call_nodes(
    path: Path,
    function_name: str,
) -> list[ast.Call]:
    tree = parse_tree(path)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == function_name:
            calls.append(node)
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == function_name
        ):
            calls.append(node)
    return calls


def import_records(path: Path) -> list[dict[str, Any]]:
    tree = parse_tree(path)
    records = []

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            names = [
                {
                    "name": alias.name,
                    "asname": alias.asname,
                }
                for alias in node.names
            ]
            records.append({
                "kind": "from",
                "module": node.module,
                "level": int(node.level),
                "names": names,
                "line": int(node.lineno),
                "source": ast.get_source_segment(read_text(path), node),
            })
        elif isinstance(node, ast.Import):
            records.append({
                "kind": "import",
                "names": [
                    {
                        "name": alias.name,
                        "asname": alias.asname,
                    }
                    for alias in node.names
                ],
                "line": int(node.lineno),
                "source": ast.get_source_segment(read_text(path), node),
            })

    return records


def target_imports(path: Path) -> list[dict[str, Any]]:
    rows = []
    for record in import_records(path):
        imported = {
            item["asname"] or item["name"]: item["name"]
            for item in record["names"]
        }
        hits = {
            local_name: original_name
            for local_name, original_name in imported.items()
            if local_name in TARGET_SYMBOLS
            or original_name in TARGET_SYMBOLS
        }
        if hits:
            rows.append({
                **record,
                "target_hits": hits,
            })
    return rows


def module_path_candidates(
    repo: Path,
    callsite: Path,
    record: dict[str, Any],
) -> list[Path]:
    module = record.get("module")
    if not module:
        return []

    module_parts = module.split(".")
    candidates = []

    # Repository-root absolute module.
    candidates.append(repo.joinpath(*module_parts).with_suffix(".py"))
    candidates.append(repo.joinpath(*module_parts, "__init__.py"))

    # Common script roots.
    for prefix in (
        repo / "scripts",
        repo / "scripts/v4",
        repo / "scripts/v4/train",
        repo / "scripts/v5",
        repo / "src",
    ):
        candidates.append(prefix.joinpath(*module_parts).with_suffix(".py"))
        candidates.append(prefix.joinpath(*module_parts, "__init__.py"))

    # Relative import resolution.
    level = int(record.get("level", 0))
    if level > 0:
        base = callsite.parent
        for _ in range(level - 1):
            base = base.parent
        candidates.append(base.joinpath(*module_parts).with_suffix(".py"))
        candidates.append(base.joinpath(*module_parts, "__init__.py"))

    return [path.resolve() for path in candidates if path.is_file()]


def search_symbol_definitions(
    repo: Path,
    symbols: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []

    for path in repo.rglob("*.py"):
        low_path = str(path).lower()
        if "/.venv/" in low_path or "/site-packages/" in low_path:
            continue
        if "/reports/v5/p3_experiments/" in low_path:
            continue
        try:
            definitions = function_definitions(path)
        except Exception:
            continue

        hits = sorted(set(symbols) & set(definitions))
        if not hits:
            continue

        rows.append({
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "defined_symbols": hits,
            "defines_all_targets": set(symbols) <= set(definitions),
        })

    rows.sort(
        key=lambda row: (
            row["defines_all_targets"],
            len(row["defined_symbols"]),
            row["path"],
        ),
        reverse=True,
    )
    return rows


def resolve_symbol_source(
    repo: Path,
    callsite: Path,
    imports: list[dict[str, Any]],
    symbol: str,
    definition_inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}

    for record in imports:
        local_original = record["target_hits"]
        if symbol not in local_original and symbol not in local_original.values():
            continue

        for path in module_path_candidates(repo, callsite, record):
            try:
                definitions = function_definitions(path)
            except Exception:
                continue
            if symbol not in definitions:
                continue
            candidates[str(path)] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "reason": "resolved from call-site import",
                "import_record": record,
            }

    for row in definition_inventory:
        if symbol not in row["defined_symbols"]:
            continue
        path = Path(row["path"])
        score = 0
        reasons = []

        if str(path) in candidates:
            score += 100
            reasons.append("call-site import resolution")
        if "/scripts/v4/" in str(path):
            score += 20
            reasons.append("V4 training lineage")
        if "/train/" in str(path):
            score += 10
            reasons.append("training helper lineage")
        if row["defines_all_targets"]:
            score += 20
            reasons.append("defines both target helpers")

        existing = candidates.setdefault(
            str(path),
            {
                "path": str(path),
                "sha256": row["sha256"],
                "reason": "repository symbol search",
            },
        )
        existing["score"] = max(existing.get("score", 0), score)
        existing.setdefault("reasons", []).extend(reasons)

    ranked = sorted(
        candidates.values(),
        key=lambda row: (
            row.get("score", 0),
            "/scripts/v4/" in row["path"],
            row["path"],
        ),
        reverse=True,
    )

    require(ranked, f"no source candidate defines {symbol}")

    unique = (
        len(ranked) == 1
        or ranked[0].get("score", 0) >= ranked[1].get("score", 0) + 25
    )

    selected = ranked[0]
    selected_path = Path(selected["path"])
    node = function_definitions(selected_path)[symbol]

    return {
        "symbol": symbol,
        "selected": {
            **selected,
            "function": source_segment(selected_path, node, context_lines=2),
            "signature": {
                "positional": [arg.arg for arg in node.args.args],
                "kwonly": [arg.arg for arg in node.args.kwonlyargs],
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
        },
        "candidate_count": len(ranked),
        "unique": unique,
        "candidates": ranked[:30],
    }


def callsite_dossier(callsite: Path) -> dict[str, Any]:
    calls = {
        symbol: find_call_nodes(callsite, symbol)
        for symbol in TARGET_SYMBOLS
    }
    return {
        "path": str(callsite),
        "sha256": sha256_file(callsite),
        "imports": target_imports(callsite),
        "calls": {
            symbol: [
                {
                    "line": int(node.lineno),
                    "source": ast.get_source_segment(read_text(callsite), node),
                    "context": source_segment(
                        callsite,
                        node,
                        context_lines=8,
                    ),
                }
                for node in nodes
            ]
            for symbol, nodes in calls.items()
        },
    }


def flatten_json(value: Any, prefix: str = "$") -> Iterable[tuple[str, Any]]:
    yield prefix, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_json(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_json(child, f"{prefix}[{index}]")


def graph_fpr_cap_evidence(repo: Path, callsite: Path) -> list[dict[str, Any]]:
    rows = []

    # Python AST/default/call-site evidence.
    for path in (
        callsite,
        repo / "scripts/v5/p3",
        repo / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107",
        repo / "reports/v5/p3_a5_tranche_a_review_and_b_handover_readiness",
    ):
        paths = [path] if path.is_file() else list(path.rglob("*.py")) if path.is_dir() else []
        for candidate in paths:
            if "/experiments/" in str(candidate):
                continue
            try:
                text = read_text(candidate)
                tree = ast.parse(text)
            except Exception:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function_name = ""
                if isinstance(node.func, ast.Attribute):
                    function_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    function_name = node.func.id

                if function_name == "add_argument":
                    option_names = []
                    for arg in node.args:
                        try:
                            value = ast.literal_eval(arg)
                        except Exception:
                            continue
                        if isinstance(value, str):
                            option_names.append(value)

                    joined = " ".join(option_names)
                    if "graph-fpr-cap" not in joined and "graph_fpr_cap" not in joined:
                        continue

                    defaults = []
                    for keyword in node.keywords:
                        if keyword.arg == "default":
                            try:
                                defaults.append(float(ast.literal_eval(keyword.value)))
                            except Exception:
                                pass
                    rows.append({
                        "source": str(candidate),
                        "sha256": sha256_file(candidate),
                        "kind": "argparse_default",
                        "line": int(node.lineno),
                        "value_candidates": defaults,
                        "context": source_segment(candidate, node, context_lines=3),
                    })

                if function_name == "select_graph_threshold_fpr_cap":
                    for keyword in node.keywords:
                        if keyword.arg != "fpr_cap":
                            continue
                        value_text = ast.unparse(keyword.value)
                        rows.append({
                            "source": str(candidate),
                            "sha256": sha256_file(candidate),
                            "kind": "threshold_helper_call",
                            "line": int(node.lineno),
                            "value_expression": value_text,
                            "context": source_segment(candidate, node, context_lines=5),
                        })

    # Report/config evidence.
    for root in (
        repo / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107",
        repo / "reports/v5/p3_a5_tranche_a_review_and_b_handover_readiness",
    ):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.stat().st_size > 128 * 1024 * 1024:
                continue

            if path.suffix.lower() == ".json":
                try:
                    document = json.loads(read_text(path))
                except Exception:
                    continue
                for json_path, value in flatten_json(document):
                    low = normalize(json_path)
                    if "graph_fpr_cap" not in low and not (
                        "threshold_policy" in low and "fpr" in low
                    ):
                        continue
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        rows.append({
                            "source": str(path),
                            "sha256": sha256_file(path),
                            "kind": "json_numeric",
                            "location": json_path,
                            "value": float(value),
                        })
            else:
                text = read_text(path)
                for line_number, line in enumerate(text.splitlines(), start=1):
                    low = normalize(line)
                    if "graph_fpr_cap" not in low and not (
                        "threshold_policy" in low and "fpr" in low
                    ):
                        continue
                    rows.append({
                        "source": str(path),
                        "sha256": sha256_file(path),
                        "kind": "text_context",
                        "line": line_number,
                        "text": line[:2000],
                    })

    return rows


def checkpoint_path_from_p0(baseline_dir: Path) -> Path:
    p0_path = baseline_dir / (
        "V5_P3_F4_P0_FROZEN_BASELINE_REPRODUCTION_PREFLIGHT_REPORT.json"
    )
    require(p0_path.is_file(), f"F4-P0 report missing: {p0_path}")
    document = json.loads(read_text(p0_path))
    checkpoint = Path(document["frozen_checkpoint"]["path"]).resolve()
    require(checkpoint.is_file(), f"frozen checkpoint missing: {checkpoint}")
    require(
        sha256_file(checkpoint) == document["frozen_checkpoint"]["sha256"],
        "frozen checkpoint hash changed",
    )
    return checkpoint


def sanitize_checkpoint_value(value: Any, depth: int = 0) -> Any:
    if depth > 12:
        return "<max-depth>"

    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            # Exclude weight payloads from the dossier.
            if str(key) in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict"):
                result[str(key)] = "<excluded-weight-or-optimizer-payload>"
            else:
                result[str(key)] = sanitize_checkpoint_value(child, depth + 1)
        return result

    if isinstance(value, (list, tuple)):
        if len(value) > 512:
            return {
                "sequence_length": len(value),
                "preview": [
                    sanitize_checkpoint_value(item, depth + 1)
                    for item in value[:20]
                ],
            }
        return [sanitize_checkpoint_value(item, depth + 1) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    try:
        import torch
        if isinstance(value, torch.Tensor):
            if value.numel() <= 256:
                return value.detach().cpu().tolist()
            return {
                "tensor_shape": list(value.shape),
                "tensor_dtype": str(value.dtype),
                "tensor_numel": int(value.numel()),
            }
    except Exception:
        pass

    return repr(value)


def checkpoint_metric_dossier(checkpoint: Path) -> dict[str, Any]:
    import torch

    loaded = None
    mode = None
    error = None

    try:
        loaded = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        mode = "weights_only_true"
    except Exception as exc:
        error = repr(exc)

    require(
        isinstance(loaded, dict),
        "checkpoint metadata could not be deserialized safely with "
        f"weights_only=True; error={error}",
    )

    validation_metrics = loaded.get("validation_metrics")
    require(
        isinstance(validation_metrics, dict),
        "checkpoint does not expose validation_metrics dictionary",
    )

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "deserialization_mode": mode,
        "top_level_keys": sorted(str(key) for key in loaded),
        "validation_metric_keys": sorted(
            str(key) for key in validation_metrics
        ),
        "validation_metrics": sanitize_checkpoint_value(validation_metrics),
        "selection_components": sanitize_checkpoint_value(
            validation_metrics.get("selection_components")
        ),
        "selection_score": sanitize_checkpoint_value(
            validation_metrics.get("selection_score")
        ),
        "threshold_policy": sanitize_checkpoint_value(
            validation_metrics.get("threshold_policy")
        ),
    }


def print_source_block(label: str, segment: dict[str, Any]) -> None:
    print(f"----- BEGIN {label} -----")
    print(
        f"path={segment['path']}:"
        f"lines={segment['line_start']}-{segment['line_end']}"
    )
    print(segment["source"])
    print(f"----- END {label} -----")


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")

    metric_dir = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/"
        "metric_adapter"
    )
    baseline_dir = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/"
        "baseline_reproduction"
    )

    r2_report_path = metric_dir / (
        "V5_P3_F4M_R2_SELECTION_SCORE_FORMULA_RECOVERY_"
        "AND_ADAPTER_FREEZE_REPORT.json"
    )
    r2_lock_path = metric_dir / (
        "V5_P3_F4M_R2_SELECTION_SCORE_FORMULA_RECOVERY_"
        "AND_ADAPTER_FREEZE_LOCK.json"
    )
    require(r2_report_path.is_file(), "F4M-R2 report missing")
    require(r2_lock_path.is_file(), "F4M-R2 lock missing")

    r2_report = json.loads(read_text(r2_report_path))
    r2_lock = json.loads(read_text(r2_lock_path))

    require(r2_report.get("status") == "PASS", "F4M-R2 is not PASS")
    require(
        r2_lock.get("report_sha256") == sha256_file(r2_report_path),
        "F4M-R2 report/lock mismatch",
    )
    require(
        r2_lock.get("selection_score_resolved") is False,
        "F4M-R3 expected unresolved selection score",
    )
    require(
        r2_lock.get("F5_authorized") is False,
        "F5 must remain unauthorized before manual pin",
    )

    callsite = repo / KNOWN_CALLSITE_RELATIVE
    require(
        callsite.is_file(),
        f"known selection-score call site missing: {callsite}",
    )

    callsite_info = callsite_dossier(callsite)
    imports = callsite_info["imports"]
    definition_inventory = search_symbol_definitions(repo, TARGET_SYMBOLS)

    selection_resolution = resolve_symbol_source(
        repo,
        callsite,
        imports,
        "selection_score",
        definition_inventory,
    )
    threshold_resolution = resolve_symbol_source(
        repo,
        callsite,
        imports,
        "select_graph_threshold_fpr_cap",
        definition_inventory,
    )

    cap_evidence = graph_fpr_cap_evidence(repo, callsite)
    checkpoint = checkpoint_path_from_p0(baseline_dir)
    checkpoint_info = checkpoint_metric_dossier(checkpoint)

    same_source = (
        selection_resolution["selected"]["path"]
        == threshold_resolution["selected"]["path"]
    )

    manual_pin_ready = bool(
        selection_resolution["unique"]
        and threshold_resolution["unique"]
        and checkpoint_info["selection_score"] is not None
        and checkpoint_info["selection_components"] is not None
        and checkpoint_info["threshold_policy"] is not None
    )

    callsite_path = output_dir / "F4M_R3_SELECTION_CALLSITE_AND_IMPORT_LINEAGE.json"
    source_path = output_dir / "F4M_R3_SELECTION_AND_THRESHOLD_HELPER_SOURCE.json"
    checkpoint_path = output_dir / "F4M_R3_CHECKPOINT_SELECTION_METADATA.json"
    cap_path = output_dir / "F4M_R3_GRAPH_FPR_CAP_EVIDENCE.json"
    dossier_path = output_dir / "F4M_R3_MANUAL_SELECTION_FORMULA_PIN_DOSSIER.json"

    atomic_json(callsite_path, callsite_info)
    atomic_json(
        source_path,
        {
            "definition_inventory": definition_inventory,
            "selection_score": selection_resolution,
            "threshold_helper": threshold_resolution,
            "same_source_file": same_source,
        },
    )
    atomic_json(checkpoint_path, checkpoint_info)
    atomic_json(cap_path, {"evidence": cap_evidence})

    dossier = {
        "stage": STAGE,
        "status": "READY_FOR_MANUAL_PIN" if manual_pin_ready else "REVIEW_REQUIRED",
        "callsite": callsite_info,
        "selection_score": selection_resolution,
        "threshold_helper": threshold_resolution,
        "graph_fpr_cap_evidence": cap_evidence,
        "checkpoint_metadata": checkpoint_info,
        "manual_pin_ready": manual_pin_ready,
        "manual_pin_requirements": {
            "freeze_selection_source_path_and_hash": True,
            "freeze_selection_function_body": True,
            "freeze_threshold_helper_path_hash_and_body": True,
            "freeze_graph_fpr_cap": True,
            "freeze_selection_components_structure": True,
            "certify_on_immutable_and_fresh_NPZ": True,
        },
        "F5_authorized": False,
    }
    atomic_json(dossier_path, dossier)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "scope": (
            "Resolve the V4 helper import lineage used by the A4/A5 "
            "selection-score call site and expose the exact function bodies, "
            "threshold policy, and frozen checkpoint selection components for "
            "a manual formula pin."
        ),
        "findings": {
            "callsite": str(callsite),
            "callsite_sha256": sha256_file(callsite),
            "selection_source": selection_resolution["selected"]["path"],
            "selection_source_sha256": selection_resolution["selected"]["sha256"],
            "selection_source_unique": selection_resolution["unique"],
            "threshold_source": threshold_resolution["selected"]["path"],
            "threshold_source_sha256": threshold_resolution["selected"]["sha256"],
            "threshold_source_unique": threshold_resolution["unique"],
            "helpers_share_source": same_source,
            "checkpoint_selection_score": checkpoint_info["selection_score"],
            "selection_components_available": (
                checkpoint_info["selection_components"] is not None
            ),
            "threshold_policy_available": (
                checkpoint_info["threshold_policy"] is not None
            ),
            "graph_fpr_cap_evidence_count": len(cap_evidence),
            "manual_pin_ready": manual_pin_ready,
        },
        "decision": {
            "F4M_R3_dossier_complete": True,
            "manual_selection_formula_pin_authorized": manual_pin_ready,
            "F4M_complete": False,
            "F5_group_block_permutation_authorized": False,
            "F6_integrated_gradients_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F4M_R3A_MANUAL_SELECTION_FORMULA_PIN"
                if manual_pin_ready
                else "V5_P3_F4M_R3R_SELECTION_LINEAGE_REVIEW"
            ),
        },
        "governance": {
            "model_instantiated": False,
            "checkpoint_deserialized_for_metadata_only": True,
            "model_state_dict_used": False,
            "optimizer_state_dict_used": False,
            "training_tensors_loaded": False,
            "validation_dataset_tensors_loaded": False,
            "validation_output_artifact_payloads_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "candidate_modules_imported": False,
        },
        "artifacts": {
            "callsite_and_import_lineage": str(callsite_path),
            "helper_source": str(source_path),
            "checkpoint_selection_metadata": str(checkpoint_path),
            "graph_fpr_cap_evidence": str(cap_path),
            "manual_pin_dossier": str(dossier_path),
        },
        "provenance": {
            "F4M_R2_report_sha256": sha256_file(r2_report_path),
            "F4M_R2_lock_sha256": sha256_file(r2_lock_path),
            "callsite_sha256": sha256_file(callsite),
            "checkpoint_sha256": sha256_file(checkpoint),
            "installed_script_sha256": sha256_file(installed_script),
            "callsite_lineage_sha256": sha256_file(callsite_path),
            "helper_source_sha256": sha256_file(source_path),
            "checkpoint_metadata_sha256": sha256_file(checkpoint_path),
            "cap_evidence_sha256": sha256_file(cap_path),
            "dossier_sha256": sha256_file(dossier_path),
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
            "callsite_lineage_sha256": sha256_file(callsite_path),
            "helper_source_sha256": sha256_file(source_path),
            "checkpoint_metadata_sha256": sha256_file(checkpoint_path),
            "cap_evidence_sha256": sha256_file(cap_path),
            "dossier_sha256": sha256_file(dossier_path),
            "selection_source_unique": selection_resolution["unique"],
            "threshold_source_unique": threshold_resolution["unique"],
            "manual_pin_ready": manual_pin_ready,
            "F4M_complete": False,
            "F5_authorized": False,
            "F6_authorized": False,
            "checkpoint_deserialized_for_metadata_only": True,
            "validation_dataset_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"callsite={callsite}")
    print(f"callsite_sha256={sha256_file(callsite)}")
    print(
        "selection_source="
        f"{selection_resolution['selected']['path']}"
    )
    print(
        "selection_source_sha256="
        f"{selection_resolution['selected']['sha256']}"
    )
    print(
        "selection_source_unique="
        f"{str(selection_resolution['unique']).lower()}"
    )
    print(
        "selection_signature="
        f"{selection_resolution['selected']['signature']}"
    )
    print(
        "threshold_source="
        f"{threshold_resolution['selected']['path']}"
    )
    print(
        "threshold_source_sha256="
        f"{threshold_resolution['selected']['sha256']}"
    )
    print(
        "threshold_source_unique="
        f"{str(threshold_resolution['unique']).lower()}"
    )
    print(
        "threshold_signature="
        f"{threshold_resolution['selected']['signature']}"
    )
    print(f"helpers_share_source={str(same_source).lower()}")
    print(
        "checkpoint_validation_metric_keys="
        f"{checkpoint_info['validation_metric_keys']}"
    )
    print(
        "checkpoint_selection_score="
        f"{checkpoint_info['selection_score']}"
    )
    print(
        "checkpoint_selection_components="
        f"{json.dumps(checkpoint_info['selection_components'], sort_keys=True)}"
    )
    print(
        "checkpoint_threshold_policy="
        f"{json.dumps(checkpoint_info['threshold_policy'], sort_keys=True)}"
    )
    print(f"graph_fpr_cap_evidence_count={len(cap_evidence)}")
    for index, row in enumerate(cap_evidence[:30]):
        print(
            f"graph_fpr_cap_evidence_{index}="
            f"{json.dumps(row, sort_keys=True, default=json_default)}"
        )

    print_source_block(
        "SELECTION_SCORE_FUNCTION",
        selection_resolution["selected"]["function"],
    )
    print_source_block(
        "GRAPH_THRESHOLD_HELPER",
        threshold_resolution["selected"]["function"],
    )

    for symbol, calls in callsite_info["calls"].items():
        for index, row in enumerate(calls):
            print_source_block(
                f"CALLSITE_{symbol}_{index}",
                row["context"],
            )

    print(f"manual_pin_ready={str(manual_pin_ready).lower()}")
    print("F4M_complete=false")
    print("F5_group_block_permutation_authorized=false")
    print("F6_integrated_gradients_authorized=false")
    print("model_instantiated=false")
    print("checkpoint_deserialized_for_metadata_only=true")
    print("validation_dataset_tensors_loaded=false")
    print("validation_output_artifact_payloads_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={report['decision']['next_stage']}")
    print(f"callsite_and_import_lineage={callsite_path}")
    print(f"helper_source={source_path}")
    print(f"checkpoint_selection_metadata={checkpoint_path}")
    print(f"graph_fpr_cap_evidence={cap_path}")
    print(f"manual_pin_dossier={dossier_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
