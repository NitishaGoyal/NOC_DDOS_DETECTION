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


STAGE = "V5_P3_F5_P0_GROUP_BLOCK_PERMUTATION_PREFLIGHT"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"

EXPECTED_MODEL_PARAMETER_COUNT = 60553
EXPECTED_VALIDATION_ITEMS = 13863
EXPECTED_INPUT_SHAPE = [16, 70, 32]
EXPECTED_MASK_SHAPE = [16, 10]
EXPECTED_EDGE_COUNT = 48
EXPECTED_LOGIT_WIDTH = 69

PERMUTATION_REPEATS = 10
PERMUTATION_SEEDS = [
    5101,
    5102,
    5103,
    5104,
    5105,
    5106,
    5107,
    5108,
    5109,
    5110,
]

MACRO_GROUPS = {
    "directional_traffic_volume": {
        "start": 0,
        "end_exclusive": 10,
        "count": 10,
    },
    "inter_flit_timing": {
        "start": 10,
        "end_exclusive": 30,
        "count": 20,
    },
    "queue_activity": {
        "start": 30,
        "end_exclusive": 40,
        "count": 10,
    },
    "buffer_pressure": {
        "start": 40,
        "end_exclusive": 55,
        "count": 15,
    },
    "flow_control_stalls": {
        "start": 55,
        "end_exclusive": 70,
        "count": 15,
    },
}

OUTPUT_HEADS = {
    "graph": 1,
    "count": 4,
    "source": 16,
    "transit": 16,
    "victim": 16,
    "path": 16,
}

SOURCE_TERMS = (
    "validation",
    "checkpoint",
    "state_dict",
    "model",
    "loader",
    "dataloader",
    "dataset",
    "np.savez",
    "savez",
    "attack_logits",
    "count_logits",
    "source_logits",
    "transit_logits",
    "victim_logits",
    "path_logits",
    "edge_index",
    "physical_port_mask",
)

FUNCTION_NAME_TERMS = (
    "load",
    "loader",
    "validation",
    "dataset",
    "model",
    "export",
    "evaluate",
    "forward",
    "collate",
    "build",
    "create",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
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


def source_segment(
    path: Path,
    node: ast.AST,
    context_lines: int = 4,
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
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


def signature_from_function(node: ast.FunctionDef) -> dict[str, Any]:
    defaults = [None] * (
        len(node.args.args) - len(node.args.defaults)
    ) + [
        ast.unparse(default)
        for default in node.args.defaults
    ]

    positional = []
    for argument, default in zip(node.args.args, defaults):
        positional.append({
            "name": argument.arg,
            "default": default,
        })

    kwonly = []
    for argument, default in zip(
        node.args.kwonlyargs,
        node.args.kw_defaults,
    ):
        kwonly.append({
            "name": argument.arg,
            "default": (
                ast.unparse(default)
                if default is not None
                else None
            ),
        })

    return {
        "positional": positional,
        "kwonly": kwonly,
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
    }


def source_inventory(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text, filename=str(path))

    classes = []
    functions = []
    imports = []
    assignments = []
    relevant_calls = []
    relevant_functions = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = []
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    methods.append({
                        "name": child.name,
                        "line": int(child.lineno),
                        "signature": signature_from_function(child),
                    })
            classes.append({
                "name": node.name,
                "line": int(node.lineno),
                "methods": methods,
            })

        elif isinstance(node, ast.FunctionDef):
            record = {
                "name": node.name,
                "line": int(node.lineno),
                "signature": signature_from_function(node),
            }
            functions.append(record)
            low_name = normalize(node.name)
            body_text = ast.get_source_segment(text, node) or ""
            low_body = body_text.lower()

            if (
                any(term in low_name for term in FUNCTION_NAME_TERMS)
                or any(term in low_body for term in SOURCE_TERMS)
            ):
                relevant_functions.append({
                    **record,
                    "source": source_segment(path, node, context_lines=1),
                })

        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append({
                "line": int(node.lineno),
                "source": ast.get_source_segment(text, node),
            })

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = (
                node.value
                if isinstance(node, ast.Assign)
                else node.value
            )
            if value is None:
                continue
            try:
                expression = ast.unparse(value)
            except Exception:
                continue

            target_text = ""
            if isinstance(node, ast.Assign):
                try:
                    target_text = ", ".join(
                        ast.unparse(target)
                        for target in node.targets
                    )
                except Exception:
                    target_text = ""
            else:
                try:
                    target_text = ast.unparse(node.target)
                except Exception:
                    target_text = ""

            joined = f"{target_text} {expression}".lower()
            if any(term in joined for term in SOURCE_TERMS):
                assignments.append({
                    "line": int(getattr(node, "lineno", 0)),
                    "target": target_text,
                    "expression": expression[:4000],
                    "context": source_segment(path, node),
                })

        if isinstance(node, ast.Call):
            try:
                call_text = ast.unparse(node)
            except Exception:
                continue
            low = call_text.lower()
            if any(term in low for term in SOURCE_TERMS):
                relevant_calls.append({
                    "line": int(getattr(node, "lineno", 0)),
                    "call": call_text[:4000],
                    "context": source_segment(path, node),
                })

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "classes": classes,
        "functions": functions,
        "imports": imports,
        "relevant_functions": relevant_functions,
        "relevant_assignments": assignments,
        "relevant_calls": relevant_calls,
    }


def argparse_inventory(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text, filename=str(path))
    rows = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue

        names = []
        for argument in node.args:
            try:
                value = ast.literal_eval(argument)
            except Exception:
                continue
            if isinstance(value, str):
                names.append(value)

        kwargs = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            try:
                kwargs[keyword.arg] = ast.literal_eval(keyword.value)
            except Exception:
                try:
                    kwargs[keyword.arg] = ast.unparse(keyword.value)
                except Exception:
                    kwargs[keyword.arg] = None

        rows.append({
            "line": int(node.lineno),
            "names": names,
            "kwargs": kwargs,
        })

    return rows


def find_values(
    value: Any,
    key_terms: tuple[str, ...],
    prefix: str = "$",
) -> list[dict[str, Any]]:
    rows = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            low = normalize(str(key))
            if any(term in low for term in key_terms):
                rows.append({
                    "json_path": child_path,
                    "value": child,
                })
            rows.extend(find_values(child, key_terms, child_path))

    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(
                find_values(
                    child,
                    key_terms,
                    f"{prefix}[{index}]",
                )
            )

    return rows


def verify_report_lock(
    report_path: Path,
    lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(report_path.is_file(), f"report missing: {report_path}")
    require(lock_path.is_file(), f"lock missing: {lock_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    require(report.get("status") == "PASS", f"report not PASS: {report_path}")
    require(
        lock.get("report_sha256") == sha256_file(report_path),
        f"report/lock mismatch: {report_path}",
    )
    return report, lock


def checkpoint_structure(
    checkpoint: Path,
) -> dict[str, Any]:
    import torch

    loaded = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    require(
        isinstance(loaded, dict),
        "checkpoint is not a dictionary",
    )

    state_dict = None
    state_dict_location = None

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
    ):
        candidate = loaded.get(key)
        if isinstance(candidate, dict):
            state_dict = candidate
            state_dict_location = key
            break

    if state_dict is None:
        tensor_like = {
            key: value
            for key, value in loaded.items()
            if hasattr(value, "shape")
        }
        if tensor_like:
            state_dict = tensor_like
            state_dict_location = "$"

    require(
        isinstance(state_dict, dict),
        "checkpoint state_dict could not be identified",
    )

    rows = []
    parameter_like_elements = 0
    buffer_like_elements = 0

    for key, value in state_dict.items():
        if not hasattr(value, "shape"):
            continue
        shape = [int(item) for item in value.shape]
        numel = 1
        for item in shape:
            numel *= item

        is_edge_buffer = (
            normalize(str(key)).endswith("edge_index")
            and shape == [2, EXPECTED_EDGE_COUNT]
        )

        if is_edge_buffer:
            buffer_like_elements += numel
        else:
            parameter_like_elements += numel

        rows.append({
            "key": str(key),
            "shape": shape,
            "dtype": str(value.dtype),
            "numel": int(numel),
            "edge_index_buffer": is_edge_buffer,
        })

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "state_dict_location": state_dict_location,
        "state_dict_tensor_count": len(rows),
        "parameter_like_elements": parameter_like_elements,
        "buffer_like_elements": buffer_like_elements,
        "total_tensor_elements": (
            parameter_like_elements + buffer_like_elements
        ),
        "state_dict": rows,
        "top_level_keys": sorted(str(key) for key in loaded),
    }


def macro_group_contract() -> dict[str, Any]:
    covered = []
    for group, row in MACRO_GROUPS.items():
        require(
            row["end_exclusive"] - row["start"] == row["count"],
            f"macro-group count mismatch: {group}",
        )
        covered.extend(range(row["start"], row["end_exclusive"]))

    require(
        covered == list(range(70)),
        f"macro groups do not exactly cover channels 0..69: {covered}",
    )

    return {
        "input_shape": EXPECTED_INPUT_SHAPE,
        "mask_shape": EXPECTED_MASK_SHAPE,
        "groups": MACRO_GROUPS,
        "coverage": covered,
        "coverage_exact": True,
        "permutation_unit": (
            "entire [16, C_group, 32] block from one validation item"
        ),
        "permutation_axis": "validation-sample axis only",
        "within_block_router_order_preserved": True,
        "within_block_channel_order_preserved": True,
        "within_block_temporal_order_preserved": True,
        "labels_and_metadata_permuted": False,
        "physical_port_mask_permuted": False,
        "edge_index_permuted": False,
        "other_feature_groups_unchanged": True,
        "repeats": PERMUTATION_REPEATS,
        "seeds": PERMUTATION_SEEDS,
        "self_mapping_policy": (
            "unrestricted random permutation; fixed points are allowed and "
            "reported per repeat"
        ),
        "primary_effect": (
            "permuted metric minus same-runtime unpermuted baseline metric"
        ),
        "importance_direction": {
            "higher_is_better_metrics": "baseline - permuted",
            "graph_fpr": "permuted - baseline",
        },
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")
    require(data_link.is_symlink(), f"dataset symlink missing: {data_link}")
    data_root = data_link.resolve()
    require(data_root.is_dir(), f"dataset target missing: {data_root}")

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

    f4m_report_path = metric_dir / (
        "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_"
        "CERTIFICATION_FINAL_REPORT.json"
    )
    f4m_lock_path = metric_dir / (
        "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_"
        "CERTIFICATION_FINAL_LOCK.json"
    )
    r3a_report_path = metric_dir / (
        "V5_P3_F4M_R3A_MANUAL_SELECTION_FORMULA_PIN_REPORT.json"
    )
    r3a_lock_path = metric_dir / (
        "V5_P3_F4M_R3A_MANUAL_SELECTION_FORMULA_PIN_LOCK.json"
    )
    f4_report_path = baseline_dir / (
        "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_"
        "BASELINE_REPRODUCTION_REPORT.json"
    )
    f4_lock_path = baseline_dir / (
        "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_"
        "BASELINE_REPRODUCTION_LOCK.json"
    )
    p1r_report_path = baseline_dir / (
        "V5_P3_F4_P1R_OFFICIAL_LINEAGE_RECOVERY_"
        "AND_ROUTE_CORRECTION_REPORT.json"
    )
    p0_report_path = baseline_dir / (
        "V5_P3_F4_P0_FROZEN_BASELINE_REPRODUCTION_PREFLIGHT_REPORT.json"
    )
    p1r3_contract_path = baseline_dir / (
        "F4_P1R3_ACTUAL_F4_REPRODUCTION_CONTRACT.json"
    )

    f4m_report, f4m_lock = verify_report_lock(
        f4m_report_path,
        f4m_lock_path,
    )
    r3a_report, r3a_lock = verify_report_lock(
        r3a_report_path,
        r3a_lock_path,
    )
    f4_report, f4_lock = verify_report_lock(
        f4_report_path,
        f4_lock_path,
    )

    require(
        f4m_lock.get("all_14_metrics_certified") is True,
        "F4M final lock does not certify all 14 metrics",
    )
    require(
        f4m_lock.get("F5_authorized") is True,
        "F4M final lock does not authorize F5",
    )
    require(
        f4m_lock.get("F6_authorized") is False,
        "F6 must remain unauthorized",
    )
    require(
        r3a_lock.get("all_14_metrics_certified") is True,
        "F4M-R3A does not certify all 14 metrics",
    )
    require(
        r3a_lock.get("F5_authorized") is True,
        "F4M-R3A does not authorize F5",
    )
    require(
        f4_lock.get("F4M_authorized") is True,
        "canonical F4 did not authorize F4M",
    )

    for path in (
        p1r_report_path,
        p0_report_path,
        p1r3_contract_path,
    ):
        require(path.is_file(), f"required lineage artifact missing: {path}")

    p1r_report = json.loads(p1r_report_path.read_text(encoding="utf-8"))
    p0_report = json.loads(p0_report_path.read_text(encoding="utf-8"))
    p1r3_contract = json.loads(
        p1r3_contract_path.read_text(encoding="utf-8")
    )

    model_path = Path(
        p1r_report["corrected_selection"]["model_path"]
    ).resolve()
    loader_path = Path(
        p1r_report["corrected_selection"]["loader_path"]
    ).resolve()
    exporter_path = Path(
        p1r3_contract["exporter"]["path"]
    ).resolve()
    checkpoint_path = Path(
        p0_report["frozen_checkpoint"]["path"]
    ).resolve()
    adapter_path = Path(
        f4m_report["adapter"]["path"]
    ).resolve()

    certified_files = {
        "model": {
            "path": model_path,
            "expected_sha256": p1r_report[
                "corrected_selection"
            ]["model_sha256"],
        },
        "loader": {
            "path": loader_path,
            "expected_sha256": p1r_report[
                "corrected_selection"
            ]["loader_sha256"],
        },
        "exporter": {
            "path": exporter_path,
            "expected_sha256": p1r3_contract["exporter"]["sha256"],
        },
        "checkpoint": {
            "path": checkpoint_path,
            "expected_sha256": p0_report[
                "frozen_checkpoint"
            ]["sha256"],
        },
        "adapter": {
            "path": adapter_path,
            "expected_sha256": f4m_report["adapter"]["sha256"],
        },
    }

    for label, row in certified_files.items():
        path = row["path"]
        require(path.is_file(), f"{label} missing: {path}")
        actual = sha256_file(path)
        require(
            actual == row["expected_sha256"],
            f"{label} hash changed: {path}",
        )
        row["actual_sha256"] = actual
        row["path"] = str(path)

    checkpoint = checkpoint_structure(checkpoint_path)
    require(
        checkpoint["parameter_like_elements"]
        == EXPECTED_MODEL_PARAMETER_COUNT,
        "checkpoint parameter-like element count changed",
    )
    require(
        checkpoint["buffer_like_elements"] == 96,
        "checkpoint edge-index buffer element count changed",
    )

    model_inventory = source_inventory(model_path)
    loader_inventory = source_inventory(loader_path)
    exporter_inventory = source_inventory(exporter_path)
    adapter_inventory = source_inventory(adapter_path)
    exporter_cli = argparse_inventory(exporter_path)

    model_classes = model_inventory["classes"]
    require(model_classes, "model source contains no class definitions")

    model_forward_candidates = []
    for class_record in model_classes:
        for method in class_record["methods"]:
            if method["name"] == "forward":
                model_forward_candidates.append({
                    "class_name": class_record["name"],
                    "method": method,
                })

    require(
        len(model_forward_candidates) >= 1,
        "model source contains no forward method",
    )

    output_width = sum(OUTPUT_HEADS.values())
    require(
        output_width == EXPECTED_LOGIT_WIDTH,
        "output-head width does not equal 69",
    )

    group_contract = macro_group_contract()

    route_inventory_path = output_dir / (
        "F5_P0_EXECUTION_ROUTE_SOURCE_INVENTORY.json"
    )
    protocol_path = output_dir / (
        "F5_P0_GROUP_BLOCK_PERMUTATION_PROTOCOL.json"
    )
    authorization_path = output_dir / (
        "F5_P0_F5_VS_F6_AUTHORIZATION_BOUNDARY.json"
    )
    checkpoint_path_out = output_dir / (
        "F5_P0_CHECKPOINT_STRUCTURE.json"
    )

    atomic_json(
        route_inventory_path,
        {
            "certified_files": certified_files,
            "model": model_inventory,
            "loader": loader_inventory,
            "exporter": exporter_inventory,
            "adapter": adapter_inventory,
            "exporter_cli": exporter_cli,
            "model_forward_candidates": model_forward_candidates,
        },
    )
    atomic_json(protocol_path, group_contract)
    atomic_json(checkpoint_path_out, checkpoint)

    route_evidence = {
        "model_forward_candidates": len(model_forward_candidates),
        "loader_relevant_function_count": len(
            loader_inventory["relevant_functions"]
        ),
        "exporter_relevant_function_count": len(
            exporter_inventory["relevant_functions"]
        ),
        "exporter_relevant_call_count": len(
            exporter_inventory["relevant_calls"]
        ),
        "exporter_cli_argument_count": len(exporter_cli),
        "adapter_function_count": len(adapter_inventory["functions"]),
        "output_head_width": output_width,
    }

    actual_f5_execution_authorized = bool(
        len(model_forward_candidates) >= 1
        and len(loader_inventory["relevant_functions"]) >= 1
        and len(exporter_inventory["relevant_functions"]) >= 1
        and len(adapter_inventory["functions"]) >= 1
        and checkpoint["parameter_like_elements"]
        == EXPECTED_MODEL_PARAMETER_COUNT
        and group_contract["coverage_exact"]
    )

    atomic_json(
        authorization_path,
        {
            "F4_complete": True,
            "F4M_complete": True,
            "F5_group_block_permutation_authorized_by_F4M": True,
            "F5_execution_route_evidence_sufficient": (
                actual_f5_execution_authorized
            ),
            "actual_F5_execution_authorized": (
                actual_f5_execution_authorized
            ),
            "F6_integrated_gradients_authorized": False,
            "sealed_test_access": False,
            "A_test_access": False,
            "next_stage": (
                "V5_P3_F5_DYNAMIC70_GROUP_BLOCK_PERMUTATION"
                if actual_f5_execution_authorized
                else "V5_P3_F5_P1_EXECUTION_ROUTE_MANUAL_REVIEW"
            ),
        },
    )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": "VALIDATION-EXPLORATORY preflight",
        "scope": (
            "Freeze the five-group Dynamic70 permutation protocol, verify "
            "F4/F4M authorization and source hashes, inspect the official "
            "model/loader/exporter/adapter route, and decide whether actual "
            "validation permutation execution may begin."
        ),
        "authorization": {
            "canonical_F4_complete": True,
            "canonical_F4M_complete": True,
            "F5_authorized_by_F4M": True,
            "actual_F5_execution_authorized": (
                actual_f5_execution_authorized
            ),
            "F6_authorized": False,
        },
        "protocol": group_contract,
        "route_evidence": route_evidence,
        "checkpoint": {
            "parameter_like_elements": checkpoint[
                "parameter_like_elements"
            ],
            "buffer_like_elements": checkpoint[
                "buffer_like_elements"
            ],
            "state_dict_tensor_count": checkpoint[
                "state_dict_tensor_count"
            ],
        },
        "governance": {
            "model_instantiated": False,
            "checkpoint_deserialized_for_structure_only": True,
            "training_tensors_loaded": False,
            "validation_dataset_tensors_loaded": False,
            "validation_output_artifact_payloads_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "thresholds_changed": False,
            "metric_formulas_changed": False,
            "feature_groups_changed": False,
        },
        "decision": {
            "F5_P0_complete": True,
            "actual_F5_execution_authorized": (
                actual_f5_execution_authorized
            ),
            "F6_integrated_gradients_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F5_DYNAMIC70_GROUP_BLOCK_PERMUTATION"
                if actual_f5_execution_authorized
                else "V5_P3_F5_P1_EXECUTION_ROUTE_MANUAL_REVIEW"
            ),
        },
        "artifacts": {
            "execution_route_inventory": str(route_inventory_path),
            "permutation_protocol": str(protocol_path),
            "checkpoint_structure": str(checkpoint_path_out),
            "authorization_boundary": str(authorization_path),
        },
        "provenance": {
            "canonical_F4_report_sha256": sha256_file(f4_report_path),
            "canonical_F4_lock_sha256": sha256_file(f4_lock_path),
            "canonical_F4M_report_sha256": sha256_file(f4m_report_path),
            "canonical_F4M_lock_sha256": sha256_file(f4m_lock_path),
            "F4M_R3A_report_sha256": sha256_file(r3a_report_path),
            "F4M_R3A_lock_sha256": sha256_file(r3a_lock_path),
            "model_sha256": sha256_file(model_path),
            "loader_sha256": sha256_file(loader_path),
            "exporter_sha256": sha256_file(exporter_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "adapter_sha256": sha256_file(adapter_path),
            "installed_script_sha256": sha256_file(installed_script),
            "route_inventory_sha256": sha256_file(route_inventory_path),
            "protocol_sha256": sha256_file(protocol_path),
            "checkpoint_structure_sha256": sha256_file(
                checkpoint_path_out
            ),
            "authorization_boundary_sha256": sha256_file(
                authorization_path
            ),
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
            "route_inventory_sha256": sha256_file(route_inventory_path),
            "protocol_sha256": sha256_file(protocol_path),
            "checkpoint_structure_sha256": sha256_file(
                checkpoint_path_out
            ),
            "authorization_boundary_sha256": sha256_file(
                authorization_path
            ),
            "actual_F5_execution_authorized": (
                actual_f5_execution_authorized
            ),
            "F6_authorized": False,
            "model_instantiated": False,
            "checkpoint_deserialized_for_structure_only": True,
            "validation_dataset_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print("canonical_F4_complete=true")
    print("canonical_F4M_complete=true")
    print("F5_authorized_by_F4M=true")
    print(f"validation_items_expected={EXPECTED_VALIDATION_ITEMS}")
    print(f"input_shape={EXPECTED_INPUT_SHAPE}")
    print(f"physical_mask_shape={EXPECTED_MASK_SHAPE}")
    print(f"directed_edge_count={EXPECTED_EDGE_COUNT}")
    print(f"output_head_width={output_width}")
    print(f"permutation_repeat_count={PERMUTATION_REPEATS}")
    print(f"permutation_seeds={PERMUTATION_SEEDS}")
    for group, row in MACRO_GROUPS.items():
        print(
            f"macro_group_{group}="
            f"{row['start']}:{row['end_exclusive']}:"
            f"channels={row['count']}"
        )
    print("macro_group_coverage_exact=true")
    print(
        "permutation_unit="
        "entire_[16,C_group,32]_validation_item_block"
    )
    print("permutation_axis=validation_sample_only")
    print("labels_permuted=false")
    print("physical_port_mask_permuted=false")
    print("edge_index_permuted=false")
    print(
        "checkpoint_parameter_like_elements="
        f"{checkpoint['parameter_like_elements']}"
    )
    print(
        "checkpoint_buffer_like_elements="
        f"{checkpoint['buffer_like_elements']}"
    )
    print(
        "model_forward_candidate_count="
        f"{len(model_forward_candidates)}"
    )
    for row in model_forward_candidates:
        print(
            "model_forward_candidate="
            f"{row['class_name']}:{row['method']['signature']}"
        )
    print(
        "loader_relevant_function_count="
        f"{len(loader_inventory['relevant_functions'])}"
    )
    for row in loader_inventory["relevant_functions"][:30]:
        print(
            "loader_relevant_function="
            f"{row['name']}:{row['signature']}:line={row['line']}"
        )
    print(
        "exporter_relevant_function_count="
        f"{len(exporter_inventory['relevant_functions'])}"
    )
    for row in exporter_inventory["relevant_functions"][:30]:
        print(
            "exporter_relevant_function="
            f"{row['name']}:{row['signature']}:line={row['line']}"
        )
    print(
        "exporter_cli_argument_count="
        f"{len(exporter_cli)}"
    )
    for row in exporter_cli:
        print(
            "exporter_cli_argument="
            f"{row['names']}:{row['kwargs']}"
        )
    print(
        "canonical_adapter_function_count="
        f"{len(adapter_inventory['functions'])}"
    )
    for row in adapter_inventory["functions"][:30]:
        print(
            "canonical_adapter_function="
            f"{row['name']}:{row['signature']}:line={row['line']}"
        )
    print(
        "actual_F5_execution_authorized="
        f"{str(actual_f5_execution_authorized).lower()}"
    )
    print("F6_integrated_gradients_authorized=false")
    print("model_instantiated=false")
    print("checkpoint_deserialized_for_structure_only=true")
    print("validation_dataset_tensors_loaded=false")
    print("validation_output_artifact_payloads_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={report['decision']['next_stage']}")
    print(f"execution_route_inventory={route_inventory_path}")
    print(f"permutation_protocol={protocol_path}")
    print(f"checkpoint_structure={checkpoint_path_out}")
    print(f"authorization_boundary={authorization_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
