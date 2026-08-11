#!/usr/bin/env python3
"""
V5 P0-A2-1 Structural-Mask Transform Discrepancy and Recoverability Audit

Purpose
-------
The A2-0 evidence showed:
- feature names 71..80 are physical input/output validity masks;
- normalization.pt says mask_transform='identity';
- normalization.pt marks zero features as mask_features;
- stored tensor values at 71..80 are consistent with
  log1p_then_standardize rather than identity.

This read-only audit:
1. verifies the A1 and A2-0 PASS chains;
2. proves whether stored mask channels are exactly recoverable from topology;
3. searches all eight axis/orientation conventions and requires a unique match;
4. verifies the match across all 48 run tensors without evaluating labels or
   model performance;
5. writes a topology-derived raw Boolean [16,10] mask artifact;
6. records a corrective feature-contract recommendation.

No model training, inference, threshold selection, checkpoint selection,
feature selection by validation, or test-performance evaluation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch


EXPECTED_MASK_NAMES = [
    "input_port_valid_local",
    "input_port_valid_north",
    "input_port_valid_east",
    "input_port_valid_south",
    "input_port_valid_west",
    "output_port_valid_local",
    "output_port_valid_north",
    "output_port_valid_east",
    "output_port_valid_south",
    "output_port_valid_west",
]

ZERO_VARIANCE_TRAIN_INDICES = [
    56, 57, 58, 59, 60, 61, 63, 66, 67, 68, 69, 70,
]

DYNAMIC70_INDICES = list(range(0, 30)) + list(range(31, 71))
PRIMARY58_INDICES = (
    list(range(0, 30))
    + list(range(31, 56))
    + [62, 64, 65]
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_pass_chain(
    directory: Path,
    marker_name: str,
    lock_name: str,
    report_name: str,
    expected_lock_status: str,
    failures: list[str],
) -> dict[str, Any]:
    marker = directory / marker_name
    lock_path = directory / lock_name
    report_path = directory / report_name

    for path in (marker, lock_path, report_path):
        if not path.is_file():
            failures.append(f"required provenance artifact missing: {path}")

    result = {
        "directory": str(directory),
        "marker_present": marker.is_file(),
        "lock_path": str(lock_path),
        "report_path": str(report_path),
    }

    if not lock_path.is_file() or not report_path.is_file():
        return result

    try:
        lock = load_json(lock_path)
        report = load_json(report_path)
    except Exception as exc:
        failures.append(f"failed to parse provenance chain in {directory}: {exc!r}")
        return result

    result.update({
        "lock_status": lock.get("status"),
        "lock_sha256": sha256_file(lock_path),
        "report_sha256": sha256_file(report_path),
    })

    if lock.get("status") != expected_lock_status:
        failures.append(
            f"unexpected lock status in {directory}: {lock.get('status')!r}"
        )

    expected_report_sha = lock.get("report_sha256")
    actual_report_sha = sha256_file(report_path)
    if expected_report_sha != actual_report_sha:
        failures.append(
            f"report SHA mismatch in {directory}: "
            f"expected={expected_report_sha} actual={actual_report_sha}"
        )

    if report.get("status") != "PASS":
        failures.append(f"report status is not PASS in {directory}")

    return result


def candidate_direction_mappings() -> list[dict[str, tuple[int, int]]]:
    """
    Generate all 8 orthogonal axis/orientation mappings.

    Coordinates are treated as two integer axes. One axis is assigned to
    north/south and the other to east/west. Each positive direction may map
    to either named direction.
    """
    mappings = []
    for ns_axis in (0, 1):
        ew_axis = 1 - ns_axis
        for north_sign in (-1, 1):
            for east_sign in (-1, 1):
                mappings.append({
                    "north": (ns_axis, north_sign),
                    "south": (ns_axis, -north_sign),
                    "east": (ew_axis, east_sign),
                    "west": (ew_axis, -east_sign),
                })
    return mappings


def mapping_name(mapping: dict[str, tuple[int, int]]) -> str:
    return "__".join(
        f"{name}=axis{axis}{'+' if sign > 0 else '-'}"
        for name, (axis, sign) in (
            ("north", mapping["north"]),
            ("east", mapping["east"]),
            ("south", mapping["south"]),
            ("west", mapping["west"]),
        )
    )


def raw_mask_from_coordinates(
    coordinates: torch.Tensor,
    mapping: dict[str, tuple[int, int]],
) -> torch.Tensor:
    """
    Return [16,10] bool masks in feature order:
    input L,N,E,S,W then output L,N,E,S,W.
    """
    coordinates = coordinates.detach().cpu().long()
    coordinate_set = {
        tuple(int(v) for v in row)
        for row in coordinates.tolist()
    }

    cardinal = {}
    for direction, (axis, sign) in mapping.items():
        values = []
        for row in coordinates.tolist():
            neighbor = [int(row[0]), int(row[1])]
            neighbor[axis] += sign
            values.append(tuple(neighbor) in coordinate_set)
        cardinal[direction] = torch.tensor(values, dtype=torch.bool)

    local = torch.ones(16, dtype=torch.bool)
    five = torch.stack([
        local,
        cardinal["north"],
        cardinal["east"],
        cardinal["south"],
        cardinal["west"],
    ], dim=1)
    return torch.cat([five, five], dim=1)


def transformed_mask(
    raw_mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    raw = raw_mask.to(dtype=torch.float32)
    transformed = torch.log1p(raw)
    return (transformed - mean[71:81]) / std[71:81]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--a2-0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    a1_dir = args.a1_dir.expanduser().resolve()
    a2_dir = args.a2_0_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not root.is_dir():
        print(f"STOP: dataset root missing: {root}", file=sys.stderr)
        return 2
    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    a1_provenance = verify_pass_chain(
        a1_dir,
        "V5_P0_A1_TENSOR_HANDOVER_AUDIT_PASS",
        "V5_P0_A1_TENSOR_HANDOVER_LOCK.json",
        "tensor_handover_audit.json",
        "V5_P0_A1_TENSOR_HANDOVER_AUDIT_PASS",
        failures,
    )
    a2_provenance = verify_pass_chain(
        a2_dir,
        "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT_PASS",
        "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT_LOCK.json",
        "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT.json",
        "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT_PASS",
        failures,
    )

    normalization_path = root / "normalization.pt"
    topology_path = root / "topology.pt"
    for path in (normalization_path, topology_path):
        if not path.is_file():
            failures.append(f"required dataset artifact missing: {path}")

    if failures:
        report = {
            "stage": "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT",
            "status": "FAIL",
            "failures": failures,
            "warnings": warnings,
        }
        report_path = output_dir / "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT.json"
        write_json(report_path, report)
        atomic_write(
            output_dir / "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_HOLD",
            "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_HOLD\n",
        )
        print("V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_HOLD")
        return 1

    norm = torch.load(
        normalization_path,
        map_location="cpu",
        weights_only=False,
    )
    topology = torch.load(
        topology_path,
        map_location="cpu",
        weights_only=False,
    )

    names = [str(name) for name in norm["feature_names"]]
    mean = norm["mean"].detach().cpu().float()
    std = norm["std"].detach().cpu().float()
    mask_features = norm["mask_features"].detach().cpu().bool()
    coordinates = topology["router_coordinates"].detach().cpu().long()

    if len(names) != 81:
        failures.append(f"normalization feature count={len(names)}, expected 81")
    if names[71:81] != EXPECTED_MASK_NAMES:
        failures.append(
            "mask feature names at 71:81 differ from expected names: "
            f"{names[71:81]}"
        )
    if tuple(mean.shape) != (81,) or tuple(std.shape) != (81,):
        failures.append(
            f"normalization shapes mean={tuple(mean.shape)} std={tuple(std.shape)}"
        )
    if tuple(mask_features.shape) != (81,):
        failures.append(
            f"mask_features shape={tuple(mask_features.shape)}, expected (81,)"
        )
    if int(mask_features.sum().item()) != 0:
        failures.append(
            "expected observed conversion defect mask_feature_count=0, "
            f"found {int(mask_features.sum().item())}"
        )
    if str(norm.get("dynamic_transform")) != "log1p_then_standardize":
        failures.append(
            "unexpected dynamic_transform: "
            f"{norm.get('dynamic_transform')!r}"
        )
    if str(norm.get("mask_transform")) != "identity":
        failures.append(
            f"unexpected mask_transform: {norm.get('mask_transform')!r}"
        )
    if str(norm.get("fit_split")) != "train":
        failures.append(f"normalization fit_split is not train: {norm.get('fit_split')!r}")
    if int(norm.get("fit_router_epoch_rows")) != 360640:
        failures.append(
            "normalization fit_router_epoch_rows is not 360640: "
            f"{norm.get('fit_router_epoch_rows')!r}"
        )
    if tuple(coordinates.shape) != (16, 2):
        failures.append(
            f"router_coordinates shape={tuple(coordinates.shape)}, expected (16,2)"
        )

    # Use the first available tensor to resolve the unique orientation.
    run_paths = []
    for split in ("train", "validation", "test"):
        run_paths.extend(sorted((root / "runs" / split).glob("*.pt")))
    if len(run_paths) != 48:
        failures.append(f"run count={len(run_paths)}, expected 48")

    observed_first = None
    if run_paths:
        payload = torch.load(
            run_paths[0],
            map_location="cpu",
            weights_only=False,
        )
        x = payload["x"].detach().cpu().float()
        observed_first = x[0, :, 71:81]

    candidate_rows = []
    matching_candidates = []

    if observed_first is not None and not failures:
        for mapping in candidate_direction_mappings():
            raw = raw_mask_from_coordinates(coordinates, mapping)
            expected = transformed_mask(raw, mean, std)
            max_abs = float((expected - observed_first).abs().max().item())
            exact = max_abs <= args.atol
            row = {
                "mapping": mapping_name(mapping),
                "max_abs_error_first_run_first_epoch": max_abs,
                "matches": exact,
            }
            candidate_rows.append(row)
            if exact:
                matching_candidates.append((mapping, raw, expected))

    if len(matching_candidates) != 1:
        failures.append(
            "expected exactly one coordinate orientation matching stored masks, "
            f"found {len(matching_candidates)}"
        )

    full_run_rows = []
    global_max_abs_error = 0.0
    time_variation_max = 0.0

    selected_mapping = None
    raw_mask = None
    expected_stored_mask = None

    if len(matching_candidates) == 1:
        selected_mapping, raw_mask, expected_stored_mask = matching_candidates[0]

        for path in run_paths:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            x = payload["x"].detach().cpu().float()
            stored = x[:, :, 71:81]
            expected = expected_stored_mask.unsqueeze(0).expand_as(stored)
            max_abs = float((stored - expected).abs().max().item())
            variation = float(
                (stored - stored[0:1]).abs().max().item()
            )
            global_max_abs_error = max(global_max_abs_error, max_abs)
            time_variation_max = max(time_variation_max, variation)
            full_run_rows.append({
                "path": str(path),
                "split": str(payload.get("split")),
                "mode": str(payload.get("mode")),
                "T": int(x.shape[0]),
                "max_abs_error_against_reconstruction": max_abs,
                "max_temporal_variation_in_mask_channels": variation,
                "matches": max_abs <= args.atol,
            })
            if max_abs > args.atol:
                failures.append(
                    f"stored masks do not match topology reconstruction in {path.name}: "
                    f"max_abs_error={max_abs}"
                )
            if variation > args.atol:
                failures.append(
                    f"stored mask channels vary over time in {path.name}: "
                    f"max_variation={variation}"
                )

    # Confirm status is structurally zero over all runs.
    status_global_max_abs = 0.0
    for path in run_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        x = payload["x"].detach().cpu().float()
        status_global_max_abs = max(
            status_global_max_abs,
            float(x[:, :, 30].abs().max().item()),
        )
    if status_global_max_abs > args.atol:
        failures.append(
            f"status_flags is not globally zero: max_abs={status_global_max_abs}"
        )

    # Save reconstructed raw mask only after exact verification.
    raw_mask_path = output_dir / "topology_derived_raw_physical_port_mask.pt"
    raw_mask_sha = None
    if raw_mask is not None and not failures:
        torch.save(
            {
                "physical_port_mask": raw_mask,
                "feature_indices": list(range(71, 81)),
                "feature_names": EXPECTED_MASK_NAMES,
                "shape": [16, 10],
                "dtype": "torch.bool",
                "coordinate_mapping": mapping_name(selected_mapping),
                "source_topology_sha256": sha256_file(topology_path),
                "derivation": (
                    "local ports valid for every router; cardinal validity "
                    "derived from router_coordinates under the unique "
                    "orientation matching stored standardized channels"
                ),
            },
            raw_mask_path,
        )
        raw_mask_sha = sha256_file(raw_mask_path)

    recommendation = {
        "defect_classification": (
            "RECOVERABLE_TENSOR_CONVERSION_MASK_METADATA_DEFECT"
        ),
        "observed_inconsistency": {
            "mask_transform_metadata": norm.get("mask_transform"),
            "mask_feature_count": int(mask_features.sum().item()),
            "stored_mask_behavior": "log1p_then_standardize",
            "interpretation": (
                "The ten physical-port mask features were not marked in "
                "mask_features and therefore followed the dynamic transform "
                "despite mask_transform='identity'."
            ),
        },
        "normalization_policy": {
            "stored_x_already_transformed": True,
            "apply_normalization_pt_during_training": False,
            "apply_log1p_again": False,
            "refit_normalization": False,
        },
        "feature_variants": {
            "ALL81_COMPATIBILITY": {
                "indices": list(range(81)),
                "purpose": (
                    "package-as-delivered compatibility baseline; stored "
                    "mask channels encode topology after standardization"
                ),
                "primary_scientific_model": False,
            },
            "DYNAMIC70": {
                "indices": DYNAMIC70_INDICES,
                "excluded_indices": [30] + list(range(71, 81)),
                "purpose": (
                    "all dynamic candidates, retaining zero-variance P0 "
                    "stall channels for forward schema compatibility"
                ),
                "primary_scientific_model": False,
            },
            "PRIMARY58": {
                "indices": PRIMARY58_INDICES,
                "excluded_indices": (
                    [30]
                    + ZERO_VARIANCE_TRAIN_INDICES
                    + list(range(71, 81))
                ),
                "separate_structure": (
                    "edge_index plus topology-derived raw Boolean [16,10] "
                    "physical-port mask"
                ),
                "purpose": (
                    "primary P0 model with only varying learned channels and "
                    "deterministic structural validity"
                ),
                "primary_scientific_model": True,
            },
        },
        "status_feature_policy": {
            "index": 30,
            "name": "status_flags",
            "learned_input": False,
            "reason": "globally constant zero audit field",
        },
        "stored_mask_channel_policy": {
            "indices": list(range(71, 81)),
            "learned_input_in_primary": False,
            "boolean_cast_allowed": False,
            "reason": (
                "stored channels are standardized real values, not raw "
                "Boolean masks"
            ),
        },
        "zero_variance_training_dynamic_policy": {
            "indices": ZERO_VARIANCE_TRAIN_INDICES,
            "retained_in_dynamic70_ablation": True,
            "excluded_from_primary58": True,
        },
        "reconstructed_mask_artifact": (
            str(raw_mask_path) if raw_mask_path.is_file() else None
        ),
        "reconstructed_mask_sha256": raw_mask_sha,
        "contract_freeze_authorized": not failures,
    }

    report = {
        "stage": "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT",
        "status": "FAIL" if failures else "PASS",
        "dataset_root": str(root),
        "failures": failures,
        "warnings": warnings,
        "a1_provenance": a1_provenance,
        "a2_0_provenance": a2_provenance,
        "normalization_sha256": sha256_file(normalization_path),
        "topology_sha256": sha256_file(topology_path),
        "normalization_metadata": {
            "dynamic_transform": norm.get("dynamic_transform"),
            "mask_transform": norm.get("mask_transform"),
            "fit_split": norm.get("fit_split"),
            "fit_router_epoch_rows": norm.get("fit_router_epoch_rows"),
            "mask_feature_count": int(mask_features.sum().item()),
        },
        "selected_coordinate_mapping": (
            mapping_name(selected_mapping)
            if selected_mapping is not None
            else None
        ),
        "matching_coordinate_mapping_count": len(matching_candidates),
        "global_max_abs_error": global_max_abs_error,
        "global_max_temporal_mask_variation": time_variation_max,
        "status_global_max_abs": status_global_max_abs,
        "run_count_checked": len(run_paths),
        "recommendation": recommendation,
        "audit_boundary": {
            "all_splits_read_for_structural_mask_equality_only": True,
            "labels_evaluated": False,
            "model_training_performed": False,
            "model_inference_performed": False,
            "feature_selection_using_validation_or_test": False,
            "threshold_selection_performed": False,
            "checkpoint_selection_performed": False,
            "test_performance_evaluated": False,
        },
        "next_stage": (
            "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_FREEZE"
            if not failures
            else "HOLD_FIX_MASK_RECOVERABILITY"
        ),
    }

    write_json(
        output_dir / "feature_contract_recommendation.json",
        recommendation,
    )
    write_json(
        output_dir / "coordinate_mapping_candidates.json",
        candidate_rows,
    )
    write_json(
        output_dir / "per_run_mask_reconstruction.json",
        full_run_rows,
    )

    report_path = output_dir / "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT.json"
    write_json(report_path, report)

    lock = {
        "status": (
            "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_PASS"
            if not failures
            else "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_HOLD"
        ),
        "report_sha256": sha256_file(report_path),
        "script_sha256": sha256_file(Path(__file__)),
        "normalization_sha256": sha256_file(normalization_path),
        "topology_sha256": sha256_file(topology_path),
        "reconstructed_mask_sha256": raw_mask_sha,
        "run_count_checked": len(run_paths),
        "contract_freeze_authorized": not failures,
        "test_performance_evaluated": False,
        "next_stage": report["next_stage"],
    }
    write_json(
        output_dir / "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_LOCK.json",
        lock,
    )

    print("===== V5 P0-A2-1 MASK RECOVERABILITY AUDIT =====")
    print(f"mask_feature_count_metadata: {int(mask_features.sum().item())}")
    print(f"mask_transform_metadata: {norm.get('mask_transform')}")
    print("stored_mask_behavior: log1p_then_standardize")
    print(f"matching_coordinate_mapping_count: {len(matching_candidates)}")
    print(
        "selected_coordinate_mapping: "
        f"{report['selected_coordinate_mapping']}"
    )
    print(f"run_count_checked: {len(run_paths)}")
    print(f"global_max_abs_error: {global_max_abs_error:.9g}")
    print(
        "global_max_temporal_mask_variation: "
        f"{time_variation_max:.9g}"
    )
    print(f"status_global_max_abs: {status_global_max_abs:.9g}")
    print(f"failure_count: {len(failures)}")
    print(f"warning_count: {len(warnings)}")

    if warnings:
        print("\n===== WARNINGS =====")
        for warning in warnings:
            print(f"WARNING: {warning}")

    if failures:
        print("\n===== FAILURES =====", file=sys.stderr)
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        atomic_write(
            output_dir / "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_HOLD",
            "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_HOLD\n",
        )
        print("contract_freeze_authorized: false")
        print("next_stage: HOLD_FIX_MASK_RECOVERABILITY")
        print("V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_HOLD")
        return 1

    atomic_write(
        output_dir / "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_PASS",
        "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_PASS\n",
    )
    print(
        "defect_classification: "
        "RECOVERABLE_TENSOR_CONVERSION_MASK_METADATA_DEFECT"
    )
    print("contract_freeze_authorized: true")
    print(
        "next_stage: "
        "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_FREEZE"
    )
    print("V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
