from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = "V5_P3_F4M_R3A_MANUAL_SELECTION_FORMULA_PIN"
CANONICAL_STAGE = "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"

METRIC_TOLERANCE = 1e-6
CHECKPOINT_IDENTITY_TOLERANCE = 1e-15

CHECKPOINT_TO_ADAPTER = {
    "graph_auroc": "graph_auroc",
    "graph_average_precision": "graph_ap",
    "source_average_precision": "source_ap",
    "transit_average_precision": "transit_ap",
    "victim_average_precision": "victim_ap",
    "path_average_precision": "path_ap",
    "count_active_macro_f1": "count_active_macro_f1",
}

EXPECTED_COMPONENT_ORDER = (
    "graph_auroc",
    "graph_average_precision",
    "source_average_precision",
    "transit_average_precision",
    "victim_average_precision",
    "path_average_precision",
    "count_active_macro_f1",
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


def import_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
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
        "installed adapter differs from package adapter",
    )

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

    r3_report_path = metric_dir / (
        "V5_P3_F4M_R3_MANUAL_SELECTION_FORMULA_"
        "PROVENANCE_DOSSIER_REPORT.json"
    )
    r3_lock_path = metric_dir / (
        "V5_P3_F4M_R3_MANUAL_SELECTION_FORMULA_"
        "PROVENANCE_DOSSIER_LOCK.json"
    )
    checkpoint_metadata_path = metric_dir / (
        "F4M_R3_CHECKPOINT_SELECTION_METADATA.json"
    )
    r2_report_path = metric_dir / (
        "V5_P3_F4M_R2_SELECTION_SCORE_FORMULA_RECOVERY_"
        "AND_ADAPTER_FREEZE_REPORT.json"
    )
    r2_lock_path = metric_dir / (
        "V5_P3_F4M_R2_SELECTION_SCORE_FORMULA_RECOVERY_"
        "AND_ADAPTER_FREEZE_LOCK.json"
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
        r3_report_path,
        r3_lock_path,
        checkpoint_metadata_path,
        r2_report_path,
        r2_lock_path,
        canonical_f4_report_path,
        canonical_f4_lock_path,
        final_contract_path,
        p1r3_contract_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"required prior artifacts missing: {missing}")

    r3_report = json.loads(r3_report_path.read_text(encoding="utf-8"))
    r3_lock = json.loads(r3_lock_path.read_text(encoding="utf-8"))
    checkpoint_metadata = json.loads(
        checkpoint_metadata_path.read_text(encoding="utf-8")
    )
    r2_report = json.loads(r2_report_path.read_text(encoding="utf-8"))
    r2_lock = json.loads(r2_lock_path.read_text(encoding="utf-8"))
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

    require(r3_report.get("status") == "PASS", "F4M-R3 is not PASS")
    require(
        r3_lock.get("report_sha256") == sha256_file(r3_report_path),
        "F4M-R3 report/lock mismatch",
    )
    require(
        r3_lock.get("checkpoint_metadata_sha256")
        == sha256_file(checkpoint_metadata_path),
        "F4M-R3 checkpoint metadata hash mismatch",
    )
    require(
        r3_lock.get("manual_pin_ready") is True,
        "F4M-R3 did not authorize manual pin",
    )
    require(r2_report.get("status") == "PASS", "F4M-R2 is not PASS")
    require(
        r2_lock.get("report_sha256") == sha256_file(r2_report_path),
        "F4M-R2 report/lock mismatch",
    )
    require(
        r2_lock.get("selection_score_resolved") is False,
        "F4M-R3A expected unresolved automated selection score",
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
    require(final_contract.get("F5_authorized") is False, "F5 must be held")

    checkpoint_components = checkpoint_metadata["selection_components"]
    checkpoint_score = float(checkpoint_metadata["selection_score"])
    threshold_policy = checkpoint_metadata["threshold_policy"]

    require(
        isinstance(checkpoint_components, dict),
        "checkpoint selection_components is not a dictionary",
    )
    require(
        set(checkpoint_components) == set(EXPECTED_COMPONENT_ORDER),
        "unexpected checkpoint selection-component keys: "
        f"{sorted(checkpoint_components)}",
    )

    ordered_checkpoint_values = np.asarray(
        [
            float(checkpoint_components[key])
            for key in EXPECTED_COMPONENT_ORDER
        ],
        dtype=np.float64,
    )
    checkpoint_mean = float(
        np.mean(ordered_checkpoint_values, dtype=np.float64)
    )
    checkpoint_identity_difference = abs(
        checkpoint_mean - checkpoint_score
    )
    require(
        checkpoint_identity_difference <= CHECKPOINT_IDENTITY_TOLERANCE,
        "checkpoint selection score is not the exact arithmetic mean of "
        "the seven recorded selection components",
    )

    require(
        float(threshold_policy["reporting_threshold"]) == 0.5,
        "unexpected checkpoint reporting threshold",
    )
    require(
        threshold_policy[
            "selection_metrics_are_threshold_free_except_count_macro_f1"
        ] is True,
        "checkpoint does not identify threshold-free selection metrics",
    )
    require(
        threshold_policy["threshold_tuning_performed"] is False,
        "checkpoint indicates threshold tuning was performed",
    )

    reference_path = Path(
        final_contract["immutable_reference"]["path"]
    ).resolve()
    require(reference_path.is_file(), "immutable validation NPZ missing")
    require(
        sha256_file(reference_path)
        == final_contract["immutable_reference"]["sha256"],
        "immutable validation NPZ hash changed",
    )

    replay_run = latest_replay_run(baseline_dir / "f4_replay_runs")
    fresh_path = select_fresh_npz(replay_run, reference_path)
    logit_keys = selected_logit_keys(p1r3_contract)

    adapter_module = import_module_from_path(
        installed_adapter,
        "_v5_p3_f4m_r3a_adapter",
    )
    reference_metrics = adapter_module.compute_metrics_from_npz(
        reference_path,
        logit_keys,
    )
    fresh_metrics = adapter_module.compute_metrics_from_npz(
        fresh_path,
        logit_keys,
    )

    frozen_metrics = {
        key: float(value)
        for key, value in final_contract["frozen_metric_vector"].items()
    }
    require(len(frozen_metrics) == 14, "expected 14 frozen metrics")

    reference_comparison = compare_metrics(
        reference_metrics,
        frozen_metrics,
    )
    fresh_comparison = compare_metrics(
        fresh_metrics,
        frozen_metrics,
    )

    component_alignment = {}
    component_alignment_pass = True
    for checkpoint_key, adapter_key in CHECKPOINT_TO_ADAPTER.items():
        checkpoint_value = float(checkpoint_components[checkpoint_key])
        reference_value = float(reference_metrics[adapter_key])
        fresh_value = float(fresh_metrics[adapter_key])

        reference_difference = abs(reference_value - checkpoint_value)
        fresh_difference = abs(fresh_value - checkpoint_value)
        reference_pass = reference_difference <= METRIC_TOLERANCE
        fresh_pass = fresh_difference <= METRIC_TOLERANCE
        component_alignment_pass &= reference_pass and fresh_pass

        component_alignment[checkpoint_key] = {
            "adapter_metric": adapter_key,
            "checkpoint_value": checkpoint_value,
            "immutable_adapter_value": reference_value,
            "fresh_adapter_value": fresh_value,
            "immutable_absolute_difference": reference_difference,
            "fresh_absolute_difference": fresh_difference,
            "tolerance": METRIC_TOLERANCE,
            "immutable_pass": reference_pass,
            "fresh_pass": fresh_pass,
        }

    reference_selection_checkpoint_difference = abs(
        float(reference_metrics["selection_score"]) - checkpoint_score
    )
    fresh_selection_checkpoint_difference = abs(
        float(fresh_metrics["selection_score"]) - checkpoint_score
    )

    selection_checkpoint_alignment_pass = (
        reference_selection_checkpoint_difference <= METRIC_TOLERANCE
        and fresh_selection_checkpoint_difference <= METRIC_TOLERANCE
    )

    all_14_certified = bool(
        reference_comparison["all_pass"]
        and fresh_comparison["all_pass"]
        and component_alignment_pass
        and selection_checkpoint_alignment_pass
        and len(reference_comparison["metrics"]) == 14
        and len(fresh_comparison["metrics"]) == 14
    )

    formula_spec = {
        "status": "FROZEN",
        "authority": (
            "frozen seed-107 checkpoint validation_metrics."
            "selection_components and selection_score"
        ),
        "formula": "arithmetic_mean_of_seven_components",
        "checkpoint_component_order": list(EXPECTED_COMPONENT_ORDER),
        "adapter_component_order": [
            CHECKPOINT_TO_ADAPTER[key]
            for key in EXPECTED_COMPONENT_ORDER
        ],
        "equation": (
            "(graph_auroc + graph_ap + source_ap + transit_ap + "
            "victim_ap + path_ap + count_active_macro_f1) / 7"
        ),
        "checkpoint_component_values": {
            key: float(checkpoint_components[key])
            for key in EXPECTED_COMPONENT_ORDER
        },
        "checkpoint_recorded_selection_score": checkpoint_score,
        "checkpoint_reconstructed_selection_score": checkpoint_mean,
        "checkpoint_identity_absolute_difference": (
            checkpoint_identity_difference
        ),
        "checkpoint_identity_tolerance": CHECKPOINT_IDENTITY_TOLERANCE,
        "published_rounded_selection_score": float(
            frozen_metrics["selection_score"]
        ),
        "threshold_policy": threshold_policy,
        "V4_constrained_threshold_formula_selected": False,
        "V4_helper_role": (
            "ancestor/call-site lineage evidence only; not authoritative for "
            "the V5-P3 seed-107 checkpoint selection score"
        ),
    }

    formula_spec_path = output_dir / (
        "F4M_R3A_FROZEN_CHECKPOINT_NATIVE_SELECTION_FORMULA.json"
    )
    component_path = output_dir / (
        "F4M_R3A_CHECKPOINT_TO_ADAPTER_COMPONENT_ALIGNMENT.json"
    )
    reference_cert_path = output_dir / (
        "F4M_R3A_IMMUTABLE_REFERENCE_14_METRIC_CERTIFICATION.json"
    )
    fresh_cert_path = output_dir / (
        "F4M_R3A_FRESH_REPLAY_14_METRIC_CERTIFICATION.json"
    )
    adapter_contract_path = output_dir / (
        "F4M_R3A_CANONICAL_14_METRIC_ADAPTER_CONTRACT.json"
    )

    atomic_json(formula_spec_path, formula_spec)
    atomic_json(
        component_path,
        {
            "component_alignment": component_alignment,
            "all_components_aligned": component_alignment_pass,
            "immutable_selection_score": reference_metrics["selection_score"],
            "fresh_selection_score": fresh_metrics["selection_score"],
            "checkpoint_selection_score": checkpoint_score,
            "immutable_selection_absolute_difference": (
                reference_selection_checkpoint_difference
            ),
            "fresh_selection_absolute_difference": (
                fresh_selection_checkpoint_difference
            ),
            "selection_alignment_pass": (
                selection_checkpoint_alignment_pass
            ),
        },
    )
    atomic_json(
        reference_cert_path,
        {
            "npz": str(reference_path),
            "npz_sha256": sha256_file(reference_path),
            "metrics": reference_metrics,
            "comparison_to_frozen_vector": reference_comparison,
        },
    )
    atomic_json(
        fresh_cert_path,
        {
            "npz": str(fresh_path),
            "npz_sha256": sha256_file(fresh_path),
            "metrics": fresh_metrics,
            "comparison_to_frozen_vector": fresh_comparison,
        },
    )

    adapter_contract = {
        "stage": STAGE,
        "status": "FROZEN" if all_14_certified else "REVIEW_REQUIRED",
        "adapter": {
            "path": str(installed_adapter),
            "sha256": sha256_file(installed_adapter),
            "contract": adapter_module.adapter_contract(),
        },
        "selection_formula_spec": formula_spec,
        "certification": {
            "checkpoint_mean_identity_pass": True,
            "checkpoint_component_alignment_pass": component_alignment_pass,
            "immutable_reference_14_metric_pass": (
                reference_comparison["all_pass"]
            ),
            "fresh_replay_14_metric_pass": (
                fresh_comparison["all_pass"]
            ),
            "selection_checkpoint_alignment_pass": (
                selection_checkpoint_alignment_pass
            ),
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
        "classification": (
            "VALIDATION-EXPLORATORY checkpoint-native selection-formula pin"
        ),
        "scope": (
            "Pin the exact V5-P3 seed-107 checkpoint selection formula from "
            "its recorded seven-component vector, certify the final executable "
            "14-metric adapter on immutable and fresh validation outputs, and "
            "separate the checkpoint-native formula from the ancestral V4 "
            "constrained-threshold helper."
        ),
        "finding": {
            "checkpoint_selection_score": checkpoint_score,
            "checkpoint_component_mean": checkpoint_mean,
            "checkpoint_identity_absolute_difference": (
                checkpoint_identity_difference
            ),
            "selection_formula": formula_spec["equation"],
            "threshold_policy": threshold_policy,
            "V4_constrained_threshold_formula_selected": False,
        },
        "certification": {
            "checkpoint_component_alignment": component_alignment,
            "checkpoint_component_alignment_pass": component_alignment_pass,
            "immutable_reference": reference_comparison,
            "fresh_replay": fresh_comparison,
            "selection_checkpoint_alignment_pass": (
                selection_checkpoint_alignment_pass
            ),
            "all_14_metrics_certified": all_14_certified,
        },
        "decision": {
            "F4M_R3A_complete": all_14_certified,
            "canonical_F4M_complete": all_14_certified,
            "F5_group_block_permutation_authorized": all_14_certified,
            "F6_integrated_gradients_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F5_DYNAMIC70_GROUP_BLOCK_PERMUTATION"
                if all_14_certified
                else "V5_P3_F4M_R3B_COMPONENT_ALIGNMENT_REVIEW"
            ),
        },
        "governance": {
            "model_loaded": False,
            "checkpoint_deserialized": False,
            "checkpoint_metadata_artifact_loaded": True,
            "training_tensors_loaded": False,
            "validation_dataset_tensors_loaded": False,
            "validation_output_artifact_payloads_loaded": True,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "threshold_tuning_performed": False,
            "metric_formulas_changed": False,
        },
        "artifacts": {
            "selection_formula_spec": str(formula_spec_path),
            "component_alignment": str(component_path),
            "immutable_reference_certification": str(reference_cert_path),
            "fresh_replay_certification": str(fresh_cert_path),
            "adapter_contract": str(adapter_contract_path),
        },
        "provenance": {
            "F4M_R3_report_sha256": sha256_file(r3_report_path),
            "F4M_R3_lock_sha256": sha256_file(r3_lock_path),
            "F4M_R3_checkpoint_metadata_sha256": sha256_file(
                checkpoint_metadata_path
            ),
            "F4M_R2_report_sha256": sha256_file(r2_report_path),
            "F4M_R2_lock_sha256": sha256_file(r2_lock_path),
            "canonical_F4_report_sha256": sha256_file(
                canonical_f4_report_path
            ),
            "canonical_F4_lock_sha256": sha256_file(
                canonical_f4_lock_path
            ),
            "final_F4_contract_sha256": sha256_file(final_contract_path),
            "F4_P1R3_contract_sha256": sha256_file(p1r3_contract_path),
            "adapter_sha256": sha256_file(installed_adapter),
            "installed_script_sha256": sha256_file(installed_script),
            "formula_spec_sha256": sha256_file(formula_spec_path),
            "component_alignment_sha256": sha256_file(component_path),
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
            "formula_spec_sha256": sha256_file(formula_spec_path),
            "component_alignment_sha256": sha256_file(component_path),
            "immutable_reference_certification_sha256": sha256_file(
                reference_cert_path
            ),
            "fresh_replay_certification_sha256": sha256_file(
                fresh_cert_path
            ),
            "adapter_contract_sha256": sha256_file(adapter_contract_path),
            "checkpoint_mean_identity_pass": True,
            "all_14_metrics_certified": all_14_certified,
            "F5_authorized": all_14_certified,
            "F6_authorized": False,
            "validation_dataset_tensors_loaded": False,
            "validation_output_artifact_payloads_loaded": True,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    final_report_path = output_dir / (
        "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION_FINAL_REPORT.json"
    )
    final_lock_path = output_dir / (
        "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION_FINAL_LOCK.json"
    )
    final_complete_path = output_dir / (
        "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION_FINAL_COMPLETE"
    )

    if all_14_certified:
        final_report = {
            "stage": CANONICAL_STAGE,
            "status": "PASS",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "campaign": CAMPAIGN,
            "classification": (
                "VALIDATION-EXPLORATORY canonical 14-metric adapter"
            ),
            "adapter": {
                "path": str(installed_adapter),
                "sha256": sha256_file(installed_adapter),
            },
            "selection_formula": formula_spec,
            "certification": {
                "checkpoint_mean_identity_pass": True,
                "immutable_reference_14_metric_pass": True,
                "fresh_replay_14_metric_pass": True,
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
                "F4M_R3A_report_sha256": sha256_file(report_path),
                "F4M_R3A_lock_sha256": sha256_file(lock_path),
                "adapter_contract_sha256": sha256_file(
                    adapter_contract_path
                ),
                "selection_formula_spec_sha256": sha256_file(
                    formula_spec_path
                ),
            },
        }
        atomic_json(final_report_path, final_report)
        atomic_json(
            final_lock_path,
            {
                "stage": CANONICAL_STAGE,
                "status": "PASS",
                "report_sha256": sha256_file(final_report_path),
                "F4M_R3A_report_sha256": sha256_file(report_path),
                "F4M_R3A_lock_sha256": sha256_file(lock_path),
                "adapter_sha256": sha256_file(installed_adapter),
                "adapter_contract_sha256": sha256_file(
                    adapter_contract_path
                ),
                "selection_formula_spec_sha256": sha256_file(
                    formula_spec_path
                ),
                "all_14_metrics_certified": True,
                "F5_authorized": True,
                "F6_authorized": False,
                "sealed_test_tensors_loaded": False,
            },
        )
        atomic_text(
            final_complete_path,
            f"{CANONICAL_STAGE}_FINAL_COMPLETE\n",
        )

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(
        "selection_formula_authority="
        "frozen_checkpoint_selection_components"
    )
    print(
        "checkpoint_selection_component_count="
        f"{len(checkpoint_components)}"
    )
    print(
        "checkpoint_selection_score="
        f"{checkpoint_score}"
    )
    print(
        "checkpoint_component_mean="
        f"{checkpoint_mean}"
    )
    print(
        "checkpoint_mean_identity_absolute_difference="
        f"{checkpoint_identity_difference}"
    )
    print(
        "selection_formula="
        "(graph_auroc+graph_ap+source_ap+transit_ap+victim_ap+"
        "path_ap+count_active_macro_f1)/7"
    )
    print("selection_metrics_threshold_free_except_count_macro_f1=true")
    print("threshold_tuning_performed=false")
    print("V4_constrained_threshold_formula_selected=false")
    print(
        "checkpoint_component_alignment_pass="
        f"{str(component_alignment_pass).lower()}"
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
        "immutable_selection_score="
        f"{reference_metrics['selection_score']}"
    )
    print(
        "fresh_selection_score="
        f"{fresh_metrics['selection_score']}"
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
    print("model_loaded=false")
    print("checkpoint_deserialized=false")
    print("checkpoint_metadata_artifact_loaded=true")
    print("validation_dataset_tensors_loaded=false")
    print("validation_output_artifact_payloads_loaded=true")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={report['decision']['next_stage']}")
    print(f"selection_formula_spec={formula_spec_path}")
    print(f"component_alignment={component_path}")
    print(f"immutable_reference_certification={reference_cert_path}")
    print(f"fresh_replay_certification={fresh_cert_path}")
    print(f"adapter_contract={adapter_contract_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")

    if all_14_certified:
        print(f"{CANONICAL_STAGE}_FINAL_COMPLETE")
        print("canonical_F4M_status=PASS")
        print("all_14_metrics_certified=true")
        print("F5_group_block_permutation_authorized=true")
        print(f"canonical_F4M_final_report={final_report_path}")
        print(f"canonical_F4M_final_lock={final_lock_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
