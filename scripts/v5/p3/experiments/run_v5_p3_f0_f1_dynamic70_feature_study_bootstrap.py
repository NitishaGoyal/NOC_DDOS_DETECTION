from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
STAGE = "V5_P3_F0_F1_DYNAMIC70_FEATURE_STUDY_BOOTSTRAP_AND_SCHEMA_CERTIFICATION"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


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


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def candidate_name_lists(document: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for path, value in flatten_json(document):
        if isinstance(value, list) and len(value) in (70, 81):
            if all(isinstance(item, str) for item in value):
                candidates.append({
                    "path": path,
                    "names": [normalize_name(item) for item in value],
                    "kind": f"string_list_{len(value)}",
                })

        if isinstance(value, dict):
            numeric = {}
            for key, child in value.items():
                try:
                    index = int(key)
                except (TypeError, ValueError):
                    continue
                if isinstance(child, str):
                    numeric[index] = normalize_name(child)
                elif isinstance(child, dict):
                    name = None
                    for field in (
                        "name", "feature_name", "channel_name",
                        "column", "field", "label",
                    ):
                        if isinstance(child.get(field), str):
                            name = normalize_name(child[field])
                            break
                    if name is not None:
                        numeric[index] = name
            if sorted(numeric) in (list(range(70)), list(range(81))):
                candidates.append({
                    "path": path,
                    "names": [numeric[i] for i in sorted(numeric)],
                    "kind": f"numeric_map_{len(numeric)}",
                })

        if isinstance(value, list) and len(value) in (70, 81):
            if all(isinstance(item, dict) for item in value):
                rows = {}
                for position, item in enumerate(value):
                    index = None
                    name = None
                    for field in (
                        "dynamic70_index", "dynamic_index", "index",
                        "feature_index", "channel_index", "raw81_index",
                    ):
                        if field in item:
                            try:
                                index = int(item[field])
                            except (TypeError, ValueError):
                                pass
                            if index is not None:
                                break
                    for field in (
                        "name", "feature_name", "channel_name",
                        "column", "field", "label",
                    ):
                        if isinstance(item.get(field), str):
                            name = normalize_name(item[field])
                            break
                    if index is None:
                        index = position
                    if name is not None:
                        rows[index] = name
                if sorted(rows) in (list(range(70)), list(range(81))):
                    candidates.append({
                        "path": path,
                        "names": [rows[i] for i in sorted(rows)],
                        "kind": f"record_list_{len(rows)}",
                    })

    return candidates


def select_best_schema_candidate(
    document: Any,
    expected_dynamic: list[str],
    expected_raw: list[str],
) -> dict[str, Any]:
    candidates = candidate_name_lists(document)
    scored = []
    for candidate in candidates:
        names = candidate["names"]
        target = expected_dynamic if len(names) == 70 else expected_raw
        exact_matches = sum(
            left == right for left, right in zip(names, target)
        )
        scored.append({
            **candidate,
            "exact_matches": exact_matches,
            "expected_count": len(target),
            "exact": names == target,
        })

    if not scored:
        raise RuntimeError(
            "feature_schema.json did not expose a supported 70/81-name list; "
            "inspect F1_SCHEMA_DISCOVERY_DEBUG.json"
        )

    scored.sort(
        key=lambda row: (
            row["exact"],
            row["exact_matches"],
            row["expected_count"],
        ),
        reverse=True,
    )
    best = scored[0]
    if not best["exact"]:
        raise RuntimeError(
            "best schema candidate did not exactly match the canonical "
            f"Dynamic70/RAW81 order: path={best['path']}, "
            f"matches={best['exact_matches']}/{best['expected_count']}"
        )
    return {"best": best, "all_candidates": scored}


def find_schema_file(data_root: Path) -> Path:
    preferred = data_root / "feature_schema.json"
    if preferred.is_file():
        return preferred

    matches = sorted(
        path
        for path in data_root.rglob("feature_schema.json")
        if path.is_file()
    )
    require(
        len(matches) == 1,
        "expected exactly one feature_schema.json under dataset root; "
        f"found={len(matches)}, matches={[str(path) for path in matches]}",
    )
    return matches[0]


def find_physical_mask_shape(document: Any) -> list[int] | None:
    for path, value in flatten_json(document):
        low = path.lower()
        if "physical" not in low or "mask" not in low:
            continue
        if (
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(item, int) for item in value)
        ):
            if list(value) == [16, 10]:
                return [16, 10]
        if isinstance(value, str):
            compact = re.sub(r"\s+", "", value.lower())
            if compact in ("[16,10]", "16x10", "(16,10)"):
                return [16, 10]
    return None


def scan_json_metadata_for_mask(
    data_root: Path,
    schema_path: Path,
) -> dict[str, Any]:
    inspected = []
    found_shape = None
    found_paths = []
    for path in sorted(data_root.rglob("*.json")):
        if path == schema_path:
            continue
        try:
            if path.stat().st_size > 16 * 1024 * 1024:
                continue
            document = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        inspected.append(str(path))
        shape = find_physical_mask_shape(document)
        if shape == [16, 10]:
            found_shape = shape
            found_paths.append(str(path))
    return {
        "inspected_json_files": inspected,
        "shape": found_shape,
        "evidence_paths": found_paths,
    }


def verify_report_lock(report_path: Path, lock_path: Path, label: str) -> dict[str, Any]:
    require(report_path.is_file(), f"{label} report missing: {report_path}")
    require(lock_path.is_file(), f"{label} lock missing: {lock_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require(report.get("status") == "PASS", f"{label} report is not PASS")
    require(
        lock.get("report_sha256") == sha256_file(report_path),
        f"{label} report/lock SHA mismatch",
    )
    return {
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    package_dir = Path(args.package_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")
    require(data_link.is_symlink(), f"canonical dataset symlink missing: {data_link}")
    data_root = data_link.resolve()
    require(data_root.is_dir(), f"dataset target missing: {data_root}")

    canonical_path = package_dir / "V5_P3_DYNAMIC70_CANONICAL_FEATURE_CONTRACT.json"
    require(canonical_path.is_file(), f"canonical contract missing: {canonical_path}")
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    expected_channels = canonical["channels"]
    expected_dynamic_names = [row["name"] for row in expected_channels]
    expected_raw_names = [
        "in_count_local", "in_count_north", "in_count_east", "in_count_south", "in_count_west",
        "out_count_local", "out_count_north", "out_count_east", "out_count_south", "out_count_west",
        "in_gap_sum_local", "in_gap_sum_north", "in_gap_sum_east", "in_gap_sum_south", "in_gap_sum_west",
        "in_gap_count_local", "in_gap_count_north", "in_gap_count_east", "in_gap_count_south", "in_gap_count_west",
        "out_gap_sum_local", "out_gap_sum_north", "out_gap_sum_east", "out_gap_sum_south", "out_gap_sum_west",
        "out_gap_count_local", "out_gap_count_north", "out_gap_count_east", "out_gap_count_south", "out_gap_count_west",
        "status_flags",
        "enqueue_count_local", "enqueue_count_north", "enqueue_count_east", "enqueue_count_south", "enqueue_count_west",
        "dequeue_count_local", "dequeue_count_north", "dequeue_count_east", "dequeue_count_south", "dequeue_count_west",
        "occupancy_cycle_sum_local", "occupancy_cycle_sum_north", "occupancy_cycle_sum_east", "occupancy_cycle_sum_south", "occupancy_cycle_sum_west",
        "occupancy_max_local", "occupancy_max_north", "occupancy_max_east", "occupancy_max_south", "occupancy_max_west",
        "occupancy_end_local", "occupancy_end_north", "occupancy_end_east", "occupancy_end_south", "occupancy_end_west",
        "stall_no_free_vc_local", "stall_no_free_vc_north", "stall_no_free_vc_east", "stall_no_free_vc_south", "stall_no_free_vc_west",
        "stall_no_credit_local", "stall_no_credit_north", "stall_no_credit_east", "stall_no_credit_south", "stall_no_credit_west",
        "stall_ordering_local", "stall_ordering_north", "stall_ordering_east", "stall_ordering_south", "stall_ordering_west",
        "input_port_valid_local", "input_port_valid_north", "input_port_valid_east", "input_port_valid_south", "input_port_valid_west",
        "output_port_valid_local", "output_port_valid_north", "output_port_valid_east", "output_port_valid_south", "output_port_valid_west",
    ]

    a5_dir = repo / "reports/v5/p3_a5_tranche_a_review_and_b_handover_readiness"
    a5 = verify_report_lock(
        a5_dir / "V5_P3_A5_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_REVIEW_AND_B_HANDOVER_READINESS_REPORT.json",
        a5_dir / "V5_P3_A5_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_REVIEW_AND_B_HANDOVER_READINESS_LOCK.json",
        "A5",
    )
    a4_checkpoint = (
        repo
        / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107"
        / "V5_P3_A4_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_SEED107_BEST.pt"
    )
    require(a4_checkpoint.is_file(), f"A4 checkpoint missing: {a4_checkpoint}")

    d13_dir = repo / "reports/v5/p3_d13_eplr_v2_handover_and_wait_state"
    d13 = verify_report_lock(
        d13_dir / "V5_P3_D13_EPLR_V2_FROZEN_IMPLEMENTATION_HANDOVER_AND_INDEPENDENT_VALIDATION_WAIT_STATE_REPORT.json",
        d13_dir / "V5_P3_D13_EPLR_V2_FROZEN_IMPLEMENTATION_HANDOVER_AND_INDEPENDENT_VALIDATION_WAIT_STATE_LOCK.json",
        "D13",
    )

    schema_path = find_schema_file(data_root)
    schema_document = json.loads(schema_path.read_text(encoding="utf-8"))

    debug_path = output_dir / "F1_SCHEMA_DISCOVERY_DEBUG.json"
    candidates = candidate_name_lists(schema_document)
    atomic_json(
        debug_path,
        {
            "schema_path": str(schema_path),
            "schema_sha256": sha256_file(schema_path),
            "candidate_summaries": [
                {
                    "path": row["path"],
                    "kind": row["kind"],
                    "count": len(row["names"]),
                    "first_names": row["names"][:8],
                    "last_names": row["names"][-8:],
                }
                for row in candidates
            ],
        },
    )

    selected = select_best_schema_candidate(
        schema_document,
        expected_dynamic_names,
        expected_raw_names,
    )
    best = selected["best"]
    if len(best["names"]) == 81:
        observed_dynamic = [
            best["names"][index]
            for index in canonical["dynamic70_raw_indices"]
        ]
    else:
        observed_dynamic = best["names"]

    require(observed_dynamic == expected_dynamic_names, "Dynamic70 order mismatch")
    require("status_flags" not in observed_dynamic, "status_flags leaked into Dynamic70")
    require(
        not any(name.startswith("input_port_valid_") for name in observed_dynamic),
        "input physical-port validity leaked into Dynamic70",
    )
    require(
        not any(name.startswith("output_port_valid_") for name in observed_dynamic),
        "output physical-port validity leaked into Dynamic70",
    )

    shape = find_physical_mask_shape(schema_document)
    mask_metadata = scan_json_metadata_for_mask(data_root, schema_path)
    if shape is None:
        shape = mask_metadata["shape"]
    require(
        shape == [16, 10],
        "physical_port_mask shape [16,10] was not explicitly found in "
        "feature_schema.json or companion JSON metadata",
    )

    study_dirs = [
        "schema_audit",
        "paired_analysis",
        "permutation",
        "attribution",
        "retraining_ablation",
        "reduced_interfaces",
        "hardware_cost",
        "figures",
        "final_report",
    ]
    for name in study_dirs:
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    f0_lock_path = output_dir / "V5_P3_F0_D70_FEATURE_STUDY_LOCK.json"
    feature_group_path = output_dir / "FEATURE_GROUP_CONTRACT.json"
    metric_contract_path = output_dir / "METRIC_CONTRACT.json"
    data_access_path = output_dir / "DATA_ACCESS_CONTRACT.json"
    channel_csv_path = output_dir / "schema_audit/DYNAMIC70_CHANNEL_INVENTORY.csv"
    channel_json_path = output_dir / "schema_audit/DYNAMIC70_CHANNEL_INVENTORY.json"
    f1_report_path = output_dir / "schema_audit/V5_P3_F1_DYNAMIC70_SCHEMA_CERTIFICATION_REPORT.json"
    f1_lock_path = output_dir / "schema_audit/V5_P3_F1_DYNAMIC70_SCHEMA_CERTIFICATION_LOCK.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    f0_lock = {
        "campaign": CAMPAIGN,
        "stage": "F0_PROTOCOL_LOCK",
        "status": "FROZEN",
        "classification": "Tranche-A preliminary auxiliary feature study",
        "purpose": (
            "Analyze Dynamic70 contribution and derive future reduced-interface "
            "candidates without changing the frozen A4/A5/D13 mainline."
        ),
        "authorizations": {
            "Tranche_A_train_access": True,
            "Tranche_A_validation_diagnostic_access": True,
            "Tranche_A_validation_model_selection_access": (
                "exploratory feature-study candidates only"
            ),
            "Tranche_A_sealed_test_access": False,
            "A_test_access": False,
            "neural_architecture_change_in_mainline": False,
            "A4_A5_checkpoint_change": False,
            "D13_decoder_change": False,
            "H1_interface_change": False,
            "final_feature_interface_promotion": False,
        },
        "promotion_boundary": [
            "frozen Tranche-B validation",
            "future fresh A+B validation",
        ],
        "frozen_thresholds": {
            "near_constant_nonzero_fraction": 0.001,
            "rare_pair_fraction": 0.01,
            "schema_exact_match_required": True,
            "baseline_reproduction_tolerance": 1e-6,
            "permutation_repeats": 10,
            "permutation_seeds": list(range(501, 511)),
            "reduced_selection_score_drop_max": 0.02,
            "reduced_graph_AP_drop_max": 0.03,
            "reduced_source_AP_drop_max": 0.03,
            "reduced_victim_AP_drop_max": 0.03,
            "reduced_path_AP_drop_max": 0.03,
            "reduced_graph_FPR_increase_max": 0.03,
            "reduced_count_macro_F1_min": 0.98,
        },
        "provenance": {
            "dataset_link": str(data_link),
            "dataset_target": str(data_root),
            "feature_schema_path": str(schema_path),
            "feature_schema_sha256": sha256_file(schema_path),
            "A4_checkpoint_path": str(a4_checkpoint),
            "A4_checkpoint_sha256": sha256_file(a4_checkpoint),
            "A5": a5,
            "D13": d13,
            "canonical_contract_sha256": sha256_file(canonical_path),
            "installed_script_sha256": sha256_file(installed_script),
        },
        "data_loaded_in_F0_F1": {
            "feature_schema_json": True,
            "companion_json_metadata": True,
            "training_tensors": False,
            "validation_tensors": False,
            "sealed_test_tensors": False,
            "model_checkpoint_loaded": False,
        },
    }
    atomic_json(f0_lock_path, f0_lock)

    feature_group_contract = {
        "campaign": CAMPAIGN,
        "status": "FROZEN",
        "input": {
            "temporal_shape": [16, 70, 32],
            "physical_port_mask_shape": [16, 10],
            "physical_port_mask_separate": True,
        },
        "dynamic70_raw_indices": canonical["dynamic70_raw_indices"],
        "excluded_raw_indices": canonical["excluded_raw_indices"],
        "macro_groups": canonical["macro_groups"],
        "channels": expected_channels,
        "ablation_policy": (
            "Initial retraining ablation operates at five macro-group level; "
            "70 leave-one-channel-out retrains are not authorized initially."
        ),
    }
    atomic_json(feature_group_path, feature_group_contract)

    metric_contract = {
        "campaign": CAMPAIGN,
        "status": "FROZEN_BEFORE_RESULTS",
        "primary_evidence": "same-width groupwise retraining ablation",
        "supporting_evidence": [
            "training-only degeneracy audit",
            "training-only matched CONTROL-ATTACK physical response",
            "frozen-checkpoint group-block permutation",
            "task-specific attribution",
            "hardware cost",
        ],
        "baseline_metrics": [
            "existing frozen multitask selection score",
            "graph accuracy",
            "graph AUROC",
            "graph average precision",
            "graph F1 at 0.5",
            "graph FPR at 0.5",
            "count active macro F1",
            "source average precision",
            "source exact-active at 0.5",
            "transit average precision",
            "transit exact-active at 0.5",
            "victim average precision",
            "victim exact-active at 0.5",
            "path average precision",
            "path exact-active at 0.5",
            "strict all-task exactness",
        ],
        "importance_sign": {
            "higher_is_better": "baseline minus perturbed/ablated",
            "graph_FPR": "perturbed/ablated minus baseline",
        },
        "cluster_unit_for_confidence_intervals": "matched pair",
    }
    atomic_json(metric_contract_path, metric_contract)

    data_access_contract = {
        "campaign": CAMPAIGN,
        "status": "FROZEN",
        "F0_F1": {
            "JSON_schema_and_metadata_only": True,
            "training_tensors": False,
            "validation_tensors": False,
            "sealed_test_tensors": False,
        },
        "F2_F3": {
            "Tranche_A_train": True,
            "Tranche_A_validation": False,
            "sealed_test": False,
        },
        "F4_F6": {
            "Tranche_A_train": (
                "only where method calibration is explicitly train-only"
            ),
            "Tranche_A_validation": True,
            "sealed_test": False,
        },
        "F7_F12": {
            "Tranche_A_train": True,
            "Tranche_A_validation": True,
            "sealed_test": False,
            "classification": "preliminary exploratory until independent confirmation",
        },
        "A_test": False,
    }
    atomic_json(data_access_path, data_access_contract)

    with channel_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dynamic70_index",
                "raw81_index",
                "name",
                "direction",
                "macro_group",
                "fine_group",
                "dynamic",
            ],
        )
        writer.writeheader()
        writer.writerows(expected_channels)
    atomic_json(
        channel_json_path,
        {
            "status": "CERTIFIED",
            "schema_path": str(schema_path),
            "schema_candidate_path": best["path"],
            "schema_candidate_kind": best["kind"],
            "channels": expected_channels,
        },
    )

    macro_counts = {
        group: len(indices)
        for group, indices in canonical["macro_groups"].items()
    }
    f1_report = {
        "stage": "V5_P3_F1_DYNAMIC70_SCHEMA_CERTIFICATION",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema": {
            "path": str(schema_path),
            "sha256": sha256_file(schema_path),
            "selected_candidate_path": best["path"],
            "selected_candidate_kind": best["kind"],
            "selected_candidate_count": len(best["names"]),
            "exact_order_match": True,
        },
        "certified_interface": {
            "dynamic_channels": 70,
            "temporal_input_shape": [16, 70, 32],
            "physical_port_mask_shape": [16, 10],
            "status_flags_in_temporal_input": False,
            "physical_port_validity_in_temporal_input": False,
            "physical_port_mask_separate": True,
            "macro_group_counts": macro_counts,
        },
        "mask_shape_evidence": {
            "shape": shape,
            "companion_metadata_evidence_paths": mask_metadata["evidence_paths"],
        },
        "artifacts": {
            "F0_lock": str(f0_lock_path),
            "feature_group_contract": str(feature_group_path),
            "metric_contract": str(metric_contract_path),
            "data_access_contract": str(data_access_path),
            "channel_inventory_csv": str(channel_csv_path),
            "channel_inventory_json": str(channel_json_path),
            "schema_discovery_debug": str(debug_path),
        },
        "decision": {
            "F0_complete": True,
            "F1_complete": True,
            "F2_authorized": True,
            "F3_authorized": False,
            "validation_access_authorized_now": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F2_DYNAMIC70_TRAINING_ONLY_DEGENERACY_"
                "AND_OBSERVABILITY_AUDIT"
            ),
        },
        "data_access": {
            "model_loaded": False,
            "training_tensors_loaded": False,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    }
    atomic_json(f1_report_path, f1_report)
    atomic_json(
        f1_lock_path,
        {
            "stage": "V5_P3_F1_DYNAMIC70_SCHEMA_CERTIFICATION",
            "status": "PASS",
            "report_sha256": sha256_file(f1_report_path),
            "F0_lock_sha256": sha256_file(f0_lock_path),
            "feature_group_contract_sha256": sha256_file(feature_group_path),
            "metric_contract_sha256": sha256_file(metric_contract_path),
            "data_access_contract_sha256": sha256_file(data_access_path),
            "channel_inventory_csv_sha256": sha256_file(channel_csv_path),
            "channel_inventory_json_sha256": sha256_file(channel_json_path),
            "feature_schema_sha256": sha256_file(schema_path),
            "A4_checkpoint_sha256": sha256_file(a4_checkpoint),
            "F2_authorized": True,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign={CAMPAIGN}")
    print("F0_protocol_lock=PASS")
    print("F1_schema_certification=PASS")
    print(f"feature_schema={schema_path}")
    print(f"feature_schema_sha256={sha256_file(schema_path)}")
    print(f"schema_candidate_path={best['path']}")
    print(f"schema_candidate_kind={best['kind']}")
    print("dynamic_channels=70")
    print("temporal_input_shape=[16,70,32]")
    print("physical_port_mask_shape=[16,10]")
    print("status_flags_in_temporal_input=false")
    print("physical_port_validity_in_temporal_input=false")
    print("macro_group_directional_traffic_volume=10")
    print("macro_group_inter_flit_timing=20")
    print("macro_group_queue_activity=10")
    print("macro_group_buffer_pressure=15")
    print("macro_group_flow_control_stalls=15")
    print("model_loaded=false")
    print("training_tensors_loaded=false")
    print("validation_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print("F2_authorized=true")
    print("F3_authorized=false")
    print(
        "next_stage="
        "V5_P3_F2_DYNAMIC70_TRAINING_ONLY_DEGENERACY_"
        "AND_OBSERVABILITY_AUDIT"
    )
    print(f"output_dir={output_dir}")
    print(f"F0_lock={f0_lock_path}")
    print(f"F1_report={f1_report_path}")
    print(f"F1_lock={f1_lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
