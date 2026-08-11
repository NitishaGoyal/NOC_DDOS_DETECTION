from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "V5_P3_F7_P0_SAME_WIDTH_RETRAINED_GROUP_ABLATION_PREFLIGHT"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "TRAIN-DERIVED"

EXPECTED_MODEL_CLASS = "V6P0Dynamic70GraphConvCount4"
EXPECTED_PARAMETER_COUNT = 60553
EXPECTED_TRAIN_ITEMS = 110855
EXPECTED_VALIDATION_ITEMS = 13863
EXPECTED_MAPPING_FINGERPRINT = (
    "8b6657a8287c899d97437ed3531f6d43b3cce327f0c45f743a9ae324365d5ae8"
)

GROUPS = {
    "directional_traffic_volume": [0, 10],
    "inter_flit_timing": [10, 30],
    "queue_activity": [30, 40],
    "buffer_pressure": [40, 55],
    "flow_control_stalls": [55, 70],
}

PRIMARY_RUNS = (
    "control_dynamic70",
    "ablate_directional_traffic_volume",
    "ablate_inter_flit_timing",
    "ablate_queue_activity",
    "ablate_buffer_pressure",
    "ablate_flow_control_stalls",
)

TRAINER_REQUIRED_TERMS = (
    "V6P0Dynamic70GraphConvCount4",
    "optimizer",
    "backward",
    "DataLoader",
)
TRAINER_SUPPORT_TERMS = (
    "validation",
    "checkpoint",
    "early",
    "source",
    "transit",
    "victim",
    "path",
    "count",
    "selection",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(rows, f"cannot write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_report_lock(
    report_path: Path,
    lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = load_json(report_path)
    lock = load_json(lock_path)
    require(report.get("status") == "PASS", f"report not PASS: {report_path}")
    require(
        lock.get("report_sha256") == sha256_file(report_path),
        f"report/lock mismatch: {report_path}",
    )
    return report, lock


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def recursive_find(
    value: Any,
    key_terms: tuple[str, ...],
    path: str = "root",
) -> list[dict[str, Any]]:
    rows = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            normalized_key = normalize(str(key))
            if any(term in normalized_key for term in key_terms):
                rows.append({
                    "path": child_path,
                    "value": child,
                })
            rows.extend(recursive_find(child, key_terms, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(
                recursive_find(child, key_terms, f"{path}[{index}]")
            )
    return rows


def safe_python_parse(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
        syntax_ok = True
        syntax_error = None
    except SyntaxError as exc:
        tree = None
        syntax_ok = False
        syntax_error = repr(exc)

    function_names = []
    class_names = []
    imports = []
    string_literals = []

    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                class_names.append(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.unparse(node))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if len(node.value) <= 500:
                    string_literals.append(node.value)

    return {
        "text": text,
        "syntax_ok": syntax_ok,
        "syntax_error": syntax_error,
        "function_names": sorted(set(function_names)),
        "class_names": sorted(set(class_names)),
        "imports": sorted(set(imports)),
        "string_literals": string_literals[:500],
    }


def trainer_candidates(repo: Path) -> list[dict[str, Any]]:
    roots = [
        repo / "scripts/v5/p3",
        repo / "scripts/v6",
        repo / "scripts",
    ]
    seen = set()
    candidates = []

    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            try:
                parsed = safe_python_parse(path)
            except Exception:
                continue
            text = parsed["text"]

            required_hits = {
                term: term in text
                for term in TRAINER_REQUIRED_TERMS
            }
            support_hits = {
                term: term.lower() in text.lower()
                for term in TRAINER_SUPPORT_TERMS
            }

            score = 0
            score += 10 * sum(required_hits.values())
            score += 1 * sum(support_hits.values())
            if "train" in normalize(path.name):
                score += 4
            if "eval" in normalize(path.name):
                score -= 2
            if "export" in normalize(path.name):
                score -= 4
            if "permutation" in str(path).lower():
                score -= 5
            if "integrated_gradient" in str(path).lower():
                score -= 5

            if required_hits[EXPECTED_MODEL_CLASS] and score >= 20:
                candidates.append({
                    "path": str(resolved),
                    "sha256": sha256_file(path),
                    "score": score,
                    "required_hits": required_hits,
                    "support_hits": support_hits,
                    "syntax_ok": parsed["syntax_ok"],
                    "function_names": parsed["function_names"],
                    "class_names": parsed["class_names"],
                })

    candidates.sort(key=lambda row: (-row["score"], row["path"]))
    return candidates


def select_trainer(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    strict = [
        row
        for row in candidates
        if (
            row["syntax_ok"]
            and all(row["required_hits"].values())
            and row["support_hits"]["validation"]
            and row["support_hits"]["checkpoint"]
            and row["support_hits"]["count"]
            and row["support_hits"]["source"]
            and row["support_hits"]["victim"]
            and row["support_hits"]["path"]
        )
    ]
    if not strict:
        return None, "no_strict_training_entrypoint_candidate"

    highest = strict[0]["score"]
    highest_rows = [row for row in strict if row["score"] == highest]
    if len(highest_rows) != 1:
        return None, "multiple_equal_highest_training_entrypoints"

    if len(strict) > 1 and strict[0]["score"] - strict[1]["score"] < 2:
        return None, "training_entrypoint_score_margin_too_small"

    return strict[0], "unique_strict_training_entrypoint"


def collect_a4_protocol_evidence(repo: Path) -> dict[str, Any]:
    a4_root = (
        repo
        / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107"
    )
    require(a4_root.is_dir(), f"A4 directory missing: {a4_root}")

    evidence = []
    for path in sorted(a4_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".json", ".txt", ".yaml", ".yml", ".csv"):
            continue
        if path.stat().st_size > 64 * 1024 * 1024:
            continue

        record: dict[str, Any] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        try:
            if path.suffix.lower() == ".json":
                value = load_json(path)
                record["seed_fields"] = recursive_find(value, ("seed",))
                record["epoch_fields"] = recursive_find(
                    value,
                    ("epoch", "patience", "early_stop"),
                )
                record["optimizer_fields"] = recursive_find(
                    value,
                    ("optimizer", "learning_rate", "weight_decay", "lr"),
                )
                record["batch_fields"] = recursive_find(
                    value,
                    ("batch",),
                )
                record["selection_fields"] = recursive_find(
                    value,
                    ("selection", "score", "metric"),
                )
                record["source_fields"] = recursive_find(
                    value,
                    ("script", "source", "trainer", "command"),
                )
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
                record["relevant_lines"] = [
                    {
                        "line": index,
                        "text": line[:1200],
                    }
                    for index, line in enumerate(text.splitlines(), start=1)
                    if any(
                        token in line.lower()
                        for token in (
                            "seed",
                            "epoch",
                            "patience",
                            "optimizer",
                            "learning rate",
                            "weight decay",
                            "batch",
                            "selection",
                            "command",
                            "train",
                        )
                    )
                ][:500]
        except Exception as exc:
            record["parse_error"] = repr(exc)

        evidence.append(record)

    return {
        "A4_root": str(a4_root.resolve()),
        "artifact_count": len(evidence),
        "artifacts": evidence,
    }


def check_training_counts(
    schema_root: Path,
) -> dict[str, Any]:
    f2_control_path = schema_root / "F2_CONTROL_ATTACK_STATISTICS.csv"
    require(f2_control_path.is_file(), f"F2 training statistics missing: {f2_control_path}")

    with f2_control_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    total_candidates = []
    for row in rows:
        for key, value in row.items():
            if value is None:
                continue
            normalized_key = normalize(str(key))
            if any(term in normalized_key for term in ("count", "items", "samples")):
                try:
                    integer = int(float(str(value)))
                except Exception:
                    continue
                total_candidates.append({
                    "field": key,
                    "value": integer,
                    "row": row,
                })

    return {
        "source": str(f2_control_path.resolve()),
        "source_sha256": sha256_file(f2_control_path),
        "numeric_count_candidates": total_candidates,
        "expected_training_items": EXPECTED_TRAIN_ITEMS,
        "expected_training_items_observed": any(
            row["value"] == EXPECTED_TRAIN_ITEMS
            for row in total_candidates
        ),
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

    feature_root = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study"
    )
    f6r_root = feature_root / "integrated_gradients_review"
    f5r_root = feature_root / "permutation_review"
    schema_root = feature_root / "schema_audit"
    metric_root = feature_root / "metric_adapter"
    baseline_root = feature_root / "baseline_reproduction"

    f6r_report_path = f6r_root / (
        "V5_P3_F6R_TASK_SPECIFIC_INTEGRATED_GRADIENTS_RESULT_REVIEW_REPORT.json"
    )
    f6r_lock_path = f6r_root / (
        "V5_P3_F6R_TASK_SPECIFIC_INTEGRATED_GRADIENTS_RESULT_REVIEW_LOCK.json"
    )
    f5r_report_path = f5r_root / (
        "V5_P3_F5R_GROUP_BLOCK_PERMUTATION_RESULT_REVIEW_REPORT.json"
    )
    f5r_lock_path = f5r_root / (
        "V5_P3_F5R_GROUP_BLOCK_PERMUTATION_RESULT_REVIEW_LOCK.json"
    )
    f4m_report_path = metric_root / (
        "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION_FINAL_REPORT.json"
    )
    f4m_lock_path = metric_root / (
        "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION_FINAL_LOCK.json"
    )
    f4_report_path = baseline_root / (
        "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_BASELINE_REPRODUCTION_REPORT.json"
    )
    f4_lock_path = baseline_root / (
        "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_BASELINE_REPRODUCTION_LOCK.json"
    )

    f6r_report, f6r_lock = verify_report_lock(
        f6r_report_path,
        f6r_lock_path,
    )
    f5r_report, f5r_lock = verify_report_lock(
        f5r_report_path,
        f5r_lock_path,
    )
    f4m_report, f4m_lock = verify_report_lock(
        f4m_report_path,
        f4m_lock_path,
    )
    f4_report, f4_lock = verify_report_lock(
        f4_report_path,
        f4_lock_path,
    )

    require(f6r_lock.get("F6R_complete") is True, "F6R incomplete")
    require(
        f6r_lock.get("F7_protocol_preflight_authorized") is True,
        "F6R did not authorize F7 protocol preflight",
    )
    require(
        f6r_lock.get("actual_F7_retraining_authorized") is False,
        "actual F7 was already authorized before preflight",
    )
    require(
        f6r_lock.get("feature_removal_authorized") is False,
        "feature removal already authorized",
    )
    require(
        f6r_lock.get("sealed_test_tensors_loaded") is False,
        "sealed-test access detected in F6R",
    )
    require(f5r_lock.get("F5R_complete") is True, "F5R incomplete")

    interpretation_path = f6r_root / (
        "F6R_FEATURE_INTERPRETATION_AND_F7_DECISION.json"
    )
    interpretation = load_json(interpretation_path)
    require(
        interpretation["decision"]["F7_protocol_preflight_authorized"] is True,
        "F6R interpretation does not authorize F7-P0",
    )
    require(
        interpretation["decision"]["actual_F7_retraining_authorized"] is False,
        "F6R interpretation prematurely authorized F7 training",
    )
    require(
        interpretation["flow_control_reduction_hypothesis_supported"] is True,
        "flow-control reduction hypothesis was not triangulated",
    )

    cross_method_path = f6r_root / (
        "F6R_CROSS_METHOD_GROUP_TRIANGULATION.csv"
    )
    with cross_method_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:
        cross_rows = list(csv.DictReader(handle))
    require(len(cross_rows) == 5, "cross-method group row count changed")

    group_names = {row["group"] for row in cross_rows}
    require(group_names == set(GROUPS), f"cross-method groups changed: {group_names}")

    flow_row = next(
        row for row in cross_rows
        if row["group"] == "flow_control_stalls"
    )
    require(int(flow_row["F5_overall_selection_rank"]) == 5, "flow stalls not F5 rank 5")
    require(int(flow_row["F6_task_last_count"]) == 6, "flow stalls not F6-last for all tasks")
    require(int(flow_row["F2R_constant_channel_count"]) == 6, "flow stalls constant count changed")

    route_inventory_path = (
        feature_root
        / "permutation/F5_P0_EXECUTION_ROUTE_SOURCE_INVENTORY.json"
    )
    route_inventory = load_json(route_inventory_path)
    certified = route_inventory["certified_files"]

    for key in ("model", "loader", "checkpoint", "adapter"):
        row = certified[key]
        path = Path(row["path"]).resolve()
        require(path.is_file(), f"certified {key} missing: {path}")
        require(
            sha256_file(path) == row["actual_sha256"],
            f"certified {key} hash changed",
        )

    require(
        certified["model"].get("model_class", EXPECTED_MODEL_CLASS)
        in (EXPECTED_MODEL_CLASS, None),
        "certified model class changed",
    )

    candidates = trainer_candidates(repo)
    selected_trainer, trainer_decision = select_trainer(candidates)

    a4_evidence = collect_a4_protocol_evidence(repo)
    training_counts = check_training_counts(schema_root)

    masking_protocol = {
        "tensor_shape_preserved": [16, 70, 32],
        "model_parameter_count_preserved": EXPECTED_PARAMETER_COUNT,
        "mask_insertion_point": (
            "after the official guarded loader returns normalized model-input "
            "x and before the first model forward pass"
        ),
        "operation": (
            "clone x and set x[:, :, start:end, :] = 0.0 for exactly one "
            "frozen macro-group"
        ),
        "physical_port_mask_operation": "unchanged",
        "edge_index_operation": "unchanged",
        "labels_operation": "unchanged",
        "zero_semantics": (
            "zero in the official post-loader model-input space; this is an "
            "information-suppression intervention, not a claim that zero is "
            "the physical mean of the raw counter"
        ),
        "control_run": "identical code path with no channel group masked",
        "no_compact_model_in_F7": True,
        "no_parameter_pruning_in_F7": True,
    }

    run_matrix = []
    for run_name in PRIMARY_RUNS:
        group = None
        channel_range = None
        if run_name.startswith("ablate_"):
            group = run_name[len("ablate_"):]
            channel_range = GROUPS[group]

        run_matrix.append({
            "run_id": run_name,
            "seed": 107,
            "group_masked": group,
            "channel_start": (
                channel_range[0] if channel_range is not None else None
            ),
            "channel_end_exclusive": (
                channel_range[1] if channel_range is not None else None
            ),
            "input_width": 70,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "training_split": "Tranche-A train only",
            "checkpoint_selection_split": "Tranche-A validation only",
            "sealed_test_access": False,
        })

    metric_protocol = {
        "primary_selection_formula": (
            "(graph_auroc + graph_ap + source_ap + transit_ap + victim_ap "
            "+ path_ap + count_active_macro_f1) / 7"
        ),
        "threshold_policy": {
            "graph_reporting_threshold": 0.5,
            "threshold_tuning_per_run": False,
            "selection_metrics_threshold_free_except_count_macro_f1": True,
        },
        "primary_comparison": (
            "each masked seed-107 run versus the same-code-path seed-107 "
            "Dynamic70 control retraining"
        ),
        "secondary_reference": (
            "official frozen A4/F4 baseline; not substituted for the same-run "
            "control"
        ),
        "candidate_reduction_gates": {
            "selection_score_drop_max": 0.02,
            "graph_average_precision_drop_max": 0.03,
            "source_average_precision_drop_max": 0.03,
            "transit_average_precision_drop_max": 0.03,
            "victim_average_precision_drop_max": 0.03,
            "path_average_precision_drop_max": 0.03,
            "graph_FPR_increase_max": 0.03,
            "count_active_macro_F1_min": 0.98,
            "single_task_AP_collapse_max": 0.05,
        },
        "gate_interpretation": (
            "passing marks a group as a candidate for F8 multi-seed "
            "confirmation; it does not authorize feature deletion"
        ),
    }

    protocol = {
        "stage": STAGE,
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_question": (
            "How much does each Dynamic70 macro-group contribute when the "
            "model is retrained from scratch without that group's information, "
            "while architecture width, parameter count, data split, training "
            "budget and checkpoint-selection rule remain fixed?"
        ),
        "primary_design": (
            "same-width post-loader zero masking with one independently "
            "retrained control and five independently retrained group "
            "ablations at seed 107"
        ),
        "groups": {
            group: {
                "start": bounds[0],
                "end_exclusive": bounds[1],
                "channel_count": bounds[1] - bounds[0],
            }
            for group, bounds in GROUPS.items()
        },
        "masking_protocol": masking_protocol,
        "run_matrix": run_matrix,
        "training_recipe_policy": {
            "inherit_without_change": (
                "official A4 seed-107 training loop, optimizer, loss weights, "
                "batching, epoch budget, early stopping, checkpoint selection "
                "and official D1 runtime"
            ),
            "only_allowed_code-path_difference": (
                "the frozen post-loader channel-zeroing operation controlled "
                "by group_masked"
            ),
            "fresh_initialization_per_run": True,
            "checkpoint_reuse_between_runs": False,
            "seed_107_primary_matrix": True,
            "F8_extra_seeds": [117, 127],
            "F8_scope": (
                "only the two most decision-relevant groups selected after "
                "the complete seed-107 matrix is reviewed"
            ),
        },
        "metric_protocol": metric_protocol,
        "governance": {
            "training_data_access_in_actual_F7": True,
            "validation_data_access_in_actual_F7": True,
            "sealed_test_access": False,
            "A_test_access": False,
            "F7_results_label": "VALIDATION-EXPLORATORY",
            "feature_removal_authorized": False,
            "compact_interface_equivalence_authorized": False,
            "hardware_reduction_claim_authorized": False,
        },
    }

    trainer_inventory_path = output_dir / "F7_P0_TRAINER_ROUTE_INVENTORY.json"
    a4_evidence_path = output_dir / "F7_P0_A4_TRAINING_PROTOCOL_EVIDENCE.json"
    training_count_path = output_dir / "F7_P0_TRAINING_COUNT_EVIDENCE.json"
    protocol_path = output_dir / "F7_P0_FROZEN_SAME_WIDTH_ABLATION_PROTOCOL.json"
    run_matrix_path = output_dir / "F7_P0_PRIMARY_RUN_MATRIX.csv"
    authorization_path = output_dir / "F7_P0_EXECUTION_AUTHORIZATION.json"

    atomic_json(
        trainer_inventory_path,
        {
            "candidate_count": len(candidates),
            "selection_decision": trainer_decision,
            "selected_trainer": selected_trainer,
            "candidates": candidates,
        },
    )
    atomic_json(a4_evidence_path, a4_evidence)
    atomic_json(training_count_path, training_counts)
    atomic_json(protocol_path, protocol)
    write_csv(run_matrix_path, run_matrix)

    actual_F7_retraining_authorized = bool(
        selected_trainer is not None
        and training_counts["expected_training_items_observed"]
        and f6r_lock.get("F7_protocol_preflight_authorized") is True
    )

    next_stage = (
        "V5_P3_F7_SAME_WIDTH_RETRAINED_GROUP_ABLATION_EXECUTION"
        if actual_F7_retraining_authorized
        else "V5_P3_F7_P1_OFFICIAL_TRAINING_ROUTE_RECOVERY"
    )

    atomic_json(
        authorization_path,
        {
            "F6R_complete": True,
            "F7_protocol_frozen": True,
            "trainer_route_resolved": selected_trainer is not None,
            "trainer_route_decision": trainer_decision,
            "training_item_count_certified": training_counts[
                "expected_training_items_observed"
            ],
            "primary_run_count": len(run_matrix),
            "control_run_count": 1,
            "masked_run_count": 5,
            "actual_F7_retraining_authorized": actual_F7_retraining_authorized,
            "F8_multi_seed_authorized": False,
            "feature_removal_authorized": False,
            "compact_interface_equivalence_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "sealed_test_access": False,
            "next_stage": next_stage,
        },
    )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "scope": (
            "Verify F4/F4M/F5R/F6R lineage, freeze the five Dynamic70 "
            "macro-group ranges, define same-width post-loader zero masking, "
            "freeze a six-run seed-107 control/ablation matrix, preserve the "
            "official A4 optimization and checkpoint-selection recipe, "
            "recover and hash a unique official training entrypoint, freeze "
            "comparison metrics and reduction-candidate gates, and decide "
            "whether actual F7 execution may begin."
        ),
        "finding": {
            "F6R_complete": True,
            "flow_control_reduction_hypothesis_supported": True,
            "flow_control_evidence": {
                "F5_overall_rank": int(flow_row["F5_overall_selection_rank"]),
                "F6_task_last_count": int(flow_row["F6_task_last_count"]),
                "F2R_constant_channel_count": int(
                    flow_row["F2R_constant_channel_count"]
                ),
            },
            "trainer_candidate_count": len(candidates),
            "trainer_selection_decision": trainer_decision,
            "selected_trainer": selected_trainer,
            "training_item_count_certified": training_counts[
                "expected_training_items_observed"
            ],
            "primary_run_count": len(run_matrix),
        },
        "decision": {
            "F7_P0_complete": True,
            "F7_protocol_frozen": True,
            "actual_F7_retraining_authorized": actual_F7_retraining_authorized,
            "F8_multi_seed_authorized": False,
            "feature_removal_authorized": False,
            "compact_interface_equivalence_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": next_stage,
        },
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "training_feature_tensors_loaded": False,
            "validation_feature_tensors_loaded": False,
            "saved_F6_outputs_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "training_source_scanned": True,
            "scientific_metrics_changed": False,
            "feature_removal_authorized": False,
        },
        "artifacts": {
            "trainer_route_inventory": str(trainer_inventory_path),
            "A4_training_protocol_evidence": str(a4_evidence_path),
            "training_count_evidence": str(training_count_path),
            "frozen_protocol": str(protocol_path),
            "primary_run_matrix": str(run_matrix_path),
            "execution_authorization": str(authorization_path),
        },
        "provenance": {
            "F6R_report_sha256": sha256_file(f6r_report_path),
            "F6R_lock_sha256": sha256_file(f6r_lock_path),
            "F6R_interpretation_sha256": sha256_file(interpretation_path),
            "F6R_cross_method_csv_sha256": sha256_file(cross_method_path),
            "F5R_report_sha256": sha256_file(f5r_report_path),
            "F5R_lock_sha256": sha256_file(f5r_lock_path),
            "F4M_report_sha256": sha256_file(f4m_report_path),
            "F4M_lock_sha256": sha256_file(f4m_lock_path),
            "F4_report_sha256": sha256_file(f4_report_path),
            "F4_lock_sha256": sha256_file(f4_lock_path),
            "route_inventory_sha256": sha256_file(route_inventory_path),
            "selected_trainer_sha256": (
                selected_trainer["sha256"]
                if selected_trainer is not None
                else None
            ),
            "installed_script_sha256": sha256_file(installed_script),
            "trainer_inventory_sha256": sha256_file(trainer_inventory_path),
            "A4_evidence_sha256": sha256_file(a4_evidence_path),
            "training_count_evidence_sha256": sha256_file(training_count_path),
            "frozen_protocol_sha256": sha256_file(protocol_path),
            "run_matrix_sha256": sha256_file(run_matrix_path),
            "authorization_sha256": sha256_file(authorization_path),
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
            "trainer_inventory_sha256": sha256_file(trainer_inventory_path),
            "A4_evidence_sha256": sha256_file(a4_evidence_path),
            "training_count_evidence_sha256": sha256_file(training_count_path),
            "frozen_protocol_sha256": sha256_file(protocol_path),
            "run_matrix_sha256": sha256_file(run_matrix_path),
            "authorization_sha256": sha256_file(authorization_path),
            "F7_P0_complete": True,
            "F7_protocol_frozen": True,
            "actual_F7_retraining_authorized": actual_F7_retraining_authorized,
            "F8_multi_seed_authorized": False,
            "feature_removal_authorized": False,
            "compact_interface_equivalence_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "training_feature_tensors_loaded": False,
            "validation_feature_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print("F6R_complete=true")
    print("flow_control_reduction_hypothesis_supported=true")
    print("F7_protocol_frozen=true")
    print("same_width_input_channels=70")
    print(f"parameter_count_preserved={EXPECTED_PARAMETER_COUNT}")
    print("mask_operation=post_loader_group_channels_set_to_zero")
    print("physical_port_mask_changed=false")
    print("edge_index_changed=false")
    print("control_retraining_included=true")
    print("primary_seed=107")
    print(f"primary_run_count={len(run_matrix)}")
    print(f"trainer_candidate_count={len(candidates)}")
    print(f"trainer_route_decision={trainer_decision}")
    if selected_trainer is not None:
        print(f"selected_trainer={selected_trainer['path']}")
        print(f"selected_trainer_sha256={selected_trainer['sha256']}")
    else:
        print("selected_trainer=None")
    print(
        "training_item_count_certified="
        f"{str(training_counts['expected_training_items_observed']).lower()}"
    )
    print(
        "actual_F7_retraining_authorized="
        f"{str(actual_F7_retraining_authorized).lower()}"
    )
    print("F8_multi_seed_authorized=false")
    print("feature_removal_authorized=false")
    print("compact_interface_equivalence_authorized=false")
    print("hardware_reduction_claim_authorized=false")
    print("model_loaded=false")
    print("checkpoint_loaded=false")
    print("training_feature_tensors_loaded=false")
    print("validation_feature_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={next_stage}")
    print(f"trainer_route_inventory={trainer_inventory_path}")
    print(f"A4_training_protocol_evidence={a4_evidence_path}")
    print(f"training_count_evidence={training_count_path}")
    print(f"frozen_protocol={protocol_path}")
    print(f"primary_run_matrix={run_matrix_path}")
    print(f"execution_authorization={authorization_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
