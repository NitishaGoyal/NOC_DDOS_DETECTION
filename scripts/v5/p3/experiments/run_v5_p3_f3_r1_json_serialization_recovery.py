from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


RECOVERY_STAGE = "V5_P3_F3_R1_JSON_SERIALIZATION_RECOVERY_AND_FINALIZATION"
F3_STAGE = (
    "V5_P3_F3_DYNAMIC70_TRAINING_ONLY_MATCHED_"
    "CONTROL_ATTACK_RESPONSE_ANALYSIS"
)
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
ALIGNMENT_COVERAGE_GATE = 0.99
ZERO_TOLERANCE = 1e-12


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


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        return result
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
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


def import_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def scalarize(value: Any) -> Any | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, bool, int, float, np.integer, np.floating)):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value)
    except Exception:
        return None
    if array.size != 1:
        return None
    result = array.reshape(-1)[0]
    if isinstance(result, bytes):
        return result.decode("utf-8", errors="replace")
    if isinstance(result, np.generic):
        result = result.item()
    if isinstance(result, float) and not math.isfinite(result):
        return str(result)
    if isinstance(result, (str, bool, int, float)):
        return result
    return str(result)


def write_recovery_figure(
    signed_mean: np.ndarray,
    role_names: tuple[str, ...],
    output_png: Path,
    output_pdf: Path,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt

        matrix = np.full(
            (len(role_names), signed_mean.shape[3]),
            np.nan,
            dtype=np.float64,
        )
        for role_index in range(len(role_names)):
            for channel_index in range(signed_mean.shape[3]):
                matrix[role_index, channel_index] = np.nanmedian(
                    signed_mean[:, 0, role_index, channel_index]
                )

        fig, ax = plt.subplots(figsize=(18, 5))
        image = ax.imshow(matrix, aspect="auto")
        ax.set_xlabel("Dynamic70 channel index")
        ax.set_ylabel("Router role")
        ax.set_xticks(range(0, signed_mean.shape[3], 2))
        ax.set_xticklabels(
            [str(index) for index in range(0, signed_mean.shape[3], 2)],
            fontsize=7,
        )
        ax.set_yticks(range(len(role_names)))
        ax.set_yticklabels(role_names)
        fig.colorbar(
            image,
            ax=ax,
            label="Median matched attack-control response",
        )
        fig.tight_layout()
        fig.savefig(output_png, dpi=180)
        fig.savefig(output_pdf)
        plt.close(fig)
        return {
            "generated": True,
            "png": str(output_png),
            "pdf": str(output_pdf),
            "exception": None,
        }
    except Exception as exc:
        return {
            "generated": False,
            "png": None,
            "pdf": None,
            "exception": repr(exc),
        }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()

    require(repo.is_dir(), f"repository missing: {repo}")
    require(data_link.is_symlink(), f"dataset symlink missing: {data_link}")
    data_root = data_link.resolve()
    require(data_root.is_dir(), f"dataset target missing: {data_root}")

    study_dir = repo / "reports/v5/p3_experiments/f0_d70_feature_study"
    audit_dir = study_dir / "schema_audit"
    paired_dir = study_dir / "paired_analysis"
    paired_dir.mkdir(parents=True, exist_ok=True)

    original_script_path = (
        repo
        / "scripts/v5/p3/experiments/"
        "run_v5_p3_f3_dynamic70_matched_control_attack_response.py"
    )
    loader_path = repo / "src/data/v5_p3_tranche_a_guarded_loader.py"

    f0_lock_path = study_dir / "V5_P3_F0_D70_FEATURE_STUDY_LOCK.json"
    group_contract_path = study_dir / "FEATURE_GROUP_CONTRACT.json"
    metric_contract_path = study_dir / "METRIC_CONTRACT.json"
    data_access_path = study_dir / "DATA_ACCESS_CONTRACT.json"
    f2_report_path = audit_dir / (
        "V5_P3_F2_DYNAMIC70_TRAINING_ONLY_DEGENERACY_"
        "AND_OBSERVABILITY_AUDIT_REPORT.json"
    )
    f2_lock_path = audit_dir / (
        "V5_P3_F2_DYNAMIC70_TRAINING_ONLY_DEGENERACY_"
        "AND_OBSERVABILITY_AUDIT_LOCK.json"
    )
    f2r_report_path = audit_dir / (
        "V5_P3_F2R_DYNAMIC70_DEGENERACY_FINDING_REVIEW_REPORT.json"
    )
    f2r_lock_path = audit_dir / (
        "V5_P3_F2R_DYNAMIC70_DEGENERACY_FINDING_REVIEW_LOCK.json"
    )
    f2r_amendment_path = audit_dir / (
        "F2R_PROTOCOL_AMENDMENT_CONSTANT_CHANNEL_HANDLING.json"
    )

    pair_effects_path = paired_dir / "F3_PAIR_LEVEL_EFFECTS.npz"
    channel_csv = paired_dir / "F3_PER_CHANNEL_ROLE_EFFECTS.csv"
    group_csv = paired_dir / "F3_PER_GROUP_ROLE_EFFECTS.csv"
    k_csv = paired_dir / "F3_ATTACKER_COUNT_GROUP_EFFECTS.csv"
    shared_csv = paired_dir / "F3_SHARED_VICTIM_GROUP_EFFECTS.csv"
    phase_csv = paired_dir / "F3_TEMPORAL_PHASE_GROUP_EFFECTS.csv"

    required = [
        original_script_path,
        loader_path,
        f0_lock_path,
        group_contract_path,
        metric_contract_path,
        data_access_path,
        f2_report_path,
        f2_lock_path,
        f2r_report_path,
        f2r_lock_path,
        f2r_amendment_path,
        pair_effects_path,
        channel_csv,
        group_csv,
        k_csv,
        shared_csv,
        phase_csv,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(
        not missing,
        "F3 recovery requires the committed pre-crash artifacts; "
        f"missing={missing}",
    )

    f0_lock = json.loads(f0_lock_path.read_text(encoding="utf-8"))
    group_contract = json.loads(
        group_contract_path.read_text(encoding="utf-8")
    )
    data_access = json.loads(data_access_path.read_text(encoding="utf-8"))
    f2_report = json.loads(f2_report_path.read_text(encoding="utf-8"))
    f2_lock = json.loads(f2_lock_path.read_text(encoding="utf-8"))
    f2r_report = json.loads(f2r_report_path.read_text(encoding="utf-8"))
    f2r_lock = json.loads(f2r_lock_path.read_text(encoding="utf-8"))

    require(f0_lock.get("status") == "FROZEN", "F0 is not frozen")
    require(f2_report.get("status") == "PASS", "F2 is not PASS")
    require(
        f2_lock.get("report_sha256") == sha256_file(f2_report_path),
        "F2 report/lock mismatch",
    )
    require(f2r_report.get("status") == "PASS", "F2R is not PASS")
    require(
        f2r_lock.get("report_sha256") == sha256_file(f2r_report_path),
        "F2R report/lock mismatch",
    )
    require(
        f2r_lock.get("F3_authorized") is True,
        "F2R did not authorize F3",
    )
    require(
        data_access["F2_F3"].get("Tranche_A_validation") is False,
        "F3 recovery must not access validation",
    )

    original = import_source(
        original_script_path,
        "_v5_p3_f3_original_failed_module",
    )
    loader_module = import_source(
        loader_path,
        "_v5_p3_f3_r1_guarded_loader",
    )

    with np.load(pair_effects_path, allow_pickle=False) as loaded:
        pair_arrays = {
            key: loaded[key].copy()
            for key in loaded.files
        }

    expected_arrays = {
        "signed_sum",
        "absolute_sum",
        "value_count",
        "signed_mean",
        "absolute_mean",
        "pair_k",
        "pair_shared_victim",
        "pair_active_windows",
        "pair_ids",
    }
    require(
        expected_arrays.issubset(pair_arrays),
        f"pair-effect archive missing arrays: "
        f"{sorted(expected_arrays - set(pair_arrays))}",
    )
    require(
        tuple(pair_arrays["signed_mean"].shape) == (600, 3, 7, 70),
        f"unexpected signed_mean shape: "
        f"{pair_arrays['signed_mean'].shape}",
    )

    pair_ids_from_npz = [
        str(value) for value in pair_arrays["pair_ids"].tolist()
    ]
    require(
        len(pair_ids_from_npz) == 600,
        "pair-effect archive does not contain 600 pairs",
    )
    matched_active_from_npz = int(
        np.sum(pair_arrays["pair_active_windows"])
    )
    require(
        matched_active_from_npz == int(f2_report["dataset"]["attack_items"]),
        "committed F3 pair effects do not cover every active training window",
    )

    Dataset = loader_module.GuardedV5P3TrancheAPreliminaryDataset
    try:
        dataset = Dataset(data_root, "train", active_only=False)
    except TypeError:
        dataset = Dataset(
            data_root=data_root,
            split="train",
            active_only=False,
        )

    item_count = len(dataset)
    require(
        item_count == int(f2_report["dataset"]["items"]),
        "training item count differs from F2",
    )

    first = dataset[0]
    pair_key, _ = original.extract_pair_id(first)
    attack_key, _ = original.extract_attack_label(first)
    count_key, _ = original.extract_count_label(first)

    excluded_large = {
        "x",
        "features",
        "input",
        "physical_port_mask",
        "port_mask",
        "physical_mask",
        "edge_index",
        "y_source",
        "y_transit",
        "y_victim",
        "y_attack_path",
        "y_path",
    }
    scalar_keys = [
        key
        for key, value in first.items()
        if key not in excluded_large and scalarize(value) is not None
    ]
    metadata: dict[str, list[Any]] = {
        key: [None] * item_count for key in scalar_keys
    }
    pair_ids = [""] * item_count
    attack_labels = np.zeros(item_count, dtype=np.uint8)

    for index in range(item_count):
        sample = first if index == 0 else dataset[index]
        observed_pair_key, pair_id = original.extract_pair_id(sample)
        observed_attack_key, attack = original.extract_attack_label(sample)
        require(observed_pair_key == pair_key, "pair key changed")
        require(observed_attack_key == attack_key, "attack key changed")
        pair_ids[index] = pair_id
        attack_labels[index] = attack
        for key in scalar_keys:
            if key in sample:
                metadata[key][index] = scalarize(sample[key])

        if (
            (index + 1) % 10000 == 0
            or index + 1 == item_count
        ):
            print(
                f"F3_R1_metadata_items={index + 1}/{item_count}",
                flush=True,
            )

    pair_to_list: dict[str, list[int]] = defaultdict(list)
    for index, pair_id in enumerate(pair_ids):
        pair_to_list[pair_id].append(index)
    pair_to_indices = {
        pair_id: np.asarray(indices, dtype=np.int64)
        for pair_id, indices in pair_to_list.items()
    }
    require(len(pair_to_indices) == 600, "metadata pass found != 600 pairs")
    require(
        sorted(pair_to_indices) == sorted(pair_ids_from_npz),
        "metadata pair IDs differ from committed F3 pair effects",
    )

    excluded_run_keys = {
        pair_key,
        attack_key,
        count_key,
        "y_attack",
        "y_graph",
        "attack_label",
        "graph_label",
        "y_attacker_count",
        "attacker_count",
        "y_count",
        "count_label",
    }
    run_candidates = original.discover_run_candidates(
        pair_to_indices,
        attack_labels,
        metadata,
        excluded_run_keys,
    )
    require(run_candidates, "no run/member metadata candidates found")
    best_run = run_candidates[0]
    require(
        best_run["valid_pair_fraction"] >= 0.99,
        "run partition coverage below 99%",
    )
    run_key = best_run["key"]
    run_values = metadata[run_key]

    time_candidate_rows = []
    excluded_time_keys = excluded_run_keys | {run_key}
    for key, values in metadata.items():
        if key in excluded_time_keys:
            continue
        unique_global = len({
            original.canonical_scalar_key(value)
            for value in values
            if value is not None
        })
        if unique_global <= 8:
            continue
        time_candidate_rows.append(
            original.evaluate_time_candidate(
                key,
                values,
                pair_to_indices,
                run_values,
                attack_labels,
            )
        )

    time_candidate_rows.sort(
        key=lambda row: (
            row["matched_active_windows"],
            row["pair_coverage"],
            -row["duplicate_time_pairs"],
            int(
                any(
                    token in row["time_key"].lower()
                    for token in (
                        "window",
                        "frame",
                        "epoch",
                        "time",
                        "start",
                        "end",
                        "index",
                    )
                )
            ),
        ),
        reverse=True,
    )

    active_total = int(np.sum(attack_labels == 1))
    if (
        time_candidate_rows
        and time_candidate_rows[0]["active_alignment_fraction"]
        >= ALIGNMENT_COVERAGE_GATE
        and time_candidate_rows[0]["duplicate_time_pairs"] == 0
    ):
        alignment_mode = "metadata_key"
        alignment_key = time_candidate_rows[0]["time_key"]
        alignment_summary = time_candidate_rows[0]
    else:
        ordinal = original.build_ordinal_alignment(
            pair_to_indices,
            run_values,
            attack_labels,
        )
        alignment_mode = ordinal["summary"]["mode"]
        alignment_key = ordinal["summary"]["time_key"]
        alignment_summary = ordinal["summary"]

    require(
        float(alignment_summary["active_alignment_fraction"])
        >= ALIGNMENT_COVERAGE_GATE,
        "recovered alignment coverage below 99%",
    )
    require(
        int(alignment_summary["matched_active_windows"])
        == matched_active_from_npz,
        "recovered alignment count differs from committed pair effects",
    )
    require(
        active_total == matched_active_from_npz,
        "not all active windows are represented in committed pair effects",
    )

    constant_indices = [
        int(row["dynamic70_index"])
        for row in f2r_report["constant_channels"]
    ]
    constant_max_abs = float(
        np.nanmax(
            np.abs(
                pair_arrays["signed_mean"][:, 0, :, constant_indices]
            )
        )
    )
    require(
        constant_max_abs <= ZERO_TOLERANCE,
        "F2R zero-signal channels have nonzero committed response",
    )

    role_names = (
        "whole_graph",
        "source",
        "transit",
        "victim",
        "path_any",
        "path_non_endpoint",
        "non_path",
    )
    heatmap_png = paired_dir / "F3_ROLE_FEATURE_RESPONSE_HEATMAP.png"
    heatmap_pdf = paired_dir / "F3_ROLE_FEATURE_RESPONSE_HEATMAP.pdf"
    figure_status = write_recovery_figure(
        pair_arrays["signed_mean"],
        role_names,
        heatmap_png,
        heatmap_pdf,
    )

    alignment_path = paired_dir / "F3_PAIR_ALIGNMENT_AUDIT.json"
    atomic_json(
        alignment_path,
        {
            "stage": F3_STAGE,
            "recovery_stage": RECOVERY_STAGE,
            "pair_key": pair_key,
            "attack_label_key": attack_key,
            "count_label_key": count_key,
            "scalar_metadata_keys": scalar_keys,
            "run_key": run_key,
            "run_candidate_rows": run_candidates,
            "time_candidate_rows": time_candidate_rows,
            "selected_alignment_mode": alignment_mode,
            "selected_alignment_key": alignment_key,
            "selected_alignment_summary": alignment_summary,
            "active_windows_total": active_total,
            "active_windows_aligned": matched_active_from_npz,
            "active_alignment_fraction": (
                matched_active_from_npz / active_total
                if active_total
                else 0.0
            ),
            "pairs": 600,
            "pairs_with_active_matches": int(
                np.sum(pair_arrays["pair_active_windows"] > 0)
            ),
            "json_serialization_recovery": {
                "original_failure": (
                    "numpy.int64 in alignment metadata was not serializable "
                    "by the original atomic_json implementation"
                ),
                "scientific_arrays_recomputed": False,
                "committed_pair_effects_reused": True,
            },
        },
    )

    active_coverage = (
        matched_active_from_npz / active_total
        if active_total
        else 0.0
    )
    f4_authorized = (
        active_coverage >= ALIGNMENT_COVERAGE_GATE
        and constant_max_abs <= ZERO_TOLERANCE
    )

    report = {
        "stage": F3_STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": (
            "Tranche-A training-only matched-pair response study"
        ),
        "recovery": {
            "recovery_stage": RECOVERY_STAGE,
            "reason": (
                "The original run completed all 29,902 aligned-window "
                "calculations and committed pair-level/CSV outputs, then "
                "failed while serializing numpy.int64 alignment metadata."
            ),
            "pair_effect_arrays_recomputed": False,
            "pair_effect_arrays_reused": True,
            "metadata_alignment_rediscovered": True,
            "scientific_result_changed": False,
        },
        "method": {
            "comparison": (
                "attack-bearing run minus matched control run at the same "
                "metadata time key or certified within-run ordinal"
            ),
            "active_windows_only": True,
            "physical_invalid_ports_excluded": True,
            "pair_cluster_unit": True,
            "early_fraction": 0.25,
            "bootstrap_repeats": 1000,
            "bootstrap_seed": 607,
        },
        "alignment": {
            "pair_key": pair_key,
            "run_key": run_key,
            "mode": alignment_mode,
            "time_key": alignment_key,
            "pairs": 600,
            "active_windows_total": active_total,
            "active_windows_aligned": matched_active_from_npz,
            "active_alignment_fraction": active_coverage,
            "gate": ALIGNMENT_COVERAGE_GATE,
        },
        "integrity": {
            "nonfinite_delta_count": 0,
            "physical_mask_mismatch_count": 0,
            "label_inconsistency_count": 0,
            "F2R_constant_channel_max_abs_response": constant_max_abs,
            "basis": (
                "The original run would have failed before committing "
                "pair effects if any of these integrity checks had failed."
            ),
        },
        "artifacts": {
            "pair_alignment_audit": str(alignment_path),
            "pair_level_effects": str(pair_effects_path),
            "per_channel_role_effects": str(channel_csv),
            "per_group_role_effects": str(group_csv),
            "attacker_count_group_effects": str(k_csv),
            "shared_victim_group_effects": str(shared_csv),
            "temporal_phase_group_effects": str(phase_csv),
            "figure": figure_status,
        },
        "decision": {
            "F3_complete": True,
            "F4_authorized": f4_authorized,
            "validation_access_authorized_now": False,
            "sealed_test_access_authorized": False,
            "constant_channel_removal_authorized": False,
            "next_stage": (
                "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_"
                "BASELINE_REPRODUCTION"
                if f4_authorized
                else "V5_P3_F3R_MATCHED_RESPONSE_ALIGNMENT_REVIEW"
            ),
        },
        "governance": {
            "model_loaded": False,
            "training_samples_loaded_for_recovery_metadata": True,
            "training_feature_arrays_consumed_in_recovery": False,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
        },
        "provenance": {
            "F0_lock_sha256": sha256_file(f0_lock_path),
            "feature_group_contract_sha256": sha256_file(
                group_contract_path
            ),
            "metric_contract_sha256": sha256_file(metric_contract_path),
            "data_access_contract_sha256": sha256_file(data_access_path),
            "F2_report_sha256": sha256_file(f2_report_path),
            "F2_lock_sha256": sha256_file(f2_lock_path),
            "F2R_report_sha256": sha256_file(f2r_report_path),
            "F2R_lock_sha256": sha256_file(f2r_lock_path),
            "F2R_amendment_sha256": sha256_file(f2r_amendment_path),
            "original_failed_script_sha256": sha256_file(
                original_script_path
            ),
            "recovery_script_sha256": sha256_file(installed_script),
            "pair_level_effects_sha256": sha256_file(pair_effects_path),
        },
    }

    report_path = paired_dir / f"{F3_STAGE}_REPORT.json"
    lock_path = paired_dir / f"{F3_STAGE}_LOCK.json"
    complete_path = paired_dir / f"{F3_STAGE}_COMPLETE"
    recovery_report_path = paired_dir / f"{RECOVERY_STAGE}_REPORT.json"
    recovery_lock_path = paired_dir / f"{RECOVERY_STAGE}_LOCK.json"

    atomic_json(report_path, report)
    atomic_json(
        lock_path,
        {
            "stage": F3_STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(report_path),
            "pair_alignment_audit_sha256": sha256_file(alignment_path),
            "pair_level_effects_sha256": sha256_file(pair_effects_path),
            "per_channel_role_effects_sha256": sha256_file(channel_csv),
            "per_group_role_effects_sha256": sha256_file(group_csv),
            "attacker_count_group_effects_sha256": sha256_file(k_csv),
            "shared_victim_group_effects_sha256": sha256_file(shared_csv),
            "temporal_phase_group_effects_sha256": sha256_file(phase_csv),
            "active_alignment_fraction": active_coverage,
            "constant_channel_max_abs_response": constant_max_abs,
            "F4_authorized": f4_authorized,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
            "recovered_after_json_serialization_failure": True,
        },
    )
    atomic_text(complete_path, f"{F3_STAGE}_COMPLETE\n")

    recovery_report = {
        "stage": RECOVERY_STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "failure_classification": "output-serialization-only",
        "original_exception": (
            "TypeError: Object of type int64 is not JSON serializable"
        ),
        "committed_scientific_outputs_reused": True,
        "aligned_tensor_pass_repeated": False,
        "metadata_pass_repeated": True,
        "F3_report_path": str(report_path),
        "F3_report_sha256": sha256_file(report_path),
        "F3_lock_path": str(lock_path),
        "F3_lock_sha256": sha256_file(lock_path),
        "validation_tensors_loaded": False,
        "sealed_test_tensors_loaded": False,
    }
    atomic_json(recovery_report_path, recovery_report)
    atomic_json(
        recovery_lock_path,
        {
            "stage": RECOVERY_STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(recovery_report_path),
            "F3_report_sha256": sha256_file(report_path),
            "F3_lock_sha256": sha256_file(lock_path),
            "pair_level_effects_sha256": sha256_file(pair_effects_path),
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )

    print(f"{RECOVERY_STAGE}_COMPLETE")
    print("status=PASS")
    print("failure_classification=output_serialization_only")
    print("aligned_tensor_pass_repeated=false")
    print("metadata_pass_repeated=true")
    print(f"pair_key={pair_key}")
    print(f"run_key={run_key}")
    print(f"alignment_mode={alignment_mode}")
    print(f"alignment_key={alignment_key}")
    print("pairs=600")
    print(f"active_windows_total={active_total}")
    print(f"active_windows_aligned={matched_active_from_npz}")
    print(f"active_alignment_fraction={active_coverage:.8f}")
    print(
        "F2R_constant_channel_max_abs_response="
        f"{constant_max_abs:.12g}"
    )
    print(f"figure_generated={str(figure_status['generated']).lower()}")
    print("model_loaded=false")
    print("validation_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"F4_authorized={str(f4_authorized).lower()}")
    print(f"next_stage={report['decision']['next_stage']}")
    print(f"F3_report={report_path}")
    print(f"F3_lock={lock_path}")
    print(f"recovery_report={recovery_report_path}")
    print(f"recovery_lock={recovery_lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
