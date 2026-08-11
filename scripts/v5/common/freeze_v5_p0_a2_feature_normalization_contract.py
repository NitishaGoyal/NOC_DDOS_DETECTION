#!/usr/bin/env python3
"""
V5 P0-A2 Feature and Normalization Contract Freeze

This stage freezes the P0 tensor-input contract after:
- P0-A1 tensor handover audit PASS;
- P0-A2-0 feature/normalization preflight PASS;
- P0-A2-1 structural-mask recoverability audit PASS.

It performs no model training, model inference, threshold selection,
checkpoint selection, validation-based feature selection, or test evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


ALL81 = list(range(81))
STATUS_INDEX = 30
MASK_INDICES = list(range(71, 81))
ZERO_VARIANCE_TRAIN_INDICES = [
    56, 57, 58, 59, 60, 61, 63, 66, 67, 68, 69, 70,
]
DYNAMIC70 = list(range(0, 30)) + list(range(31, 71))
PRIMARY58 = (
    list(range(0, 30))
    + list(range(31, 56))
    + [62, 64, 65]
)

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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_write(path, "")
        return

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_chain(
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

    result: dict[str, Any] = {
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

    actual_report_sha = sha256_file(report_path)
    result.update({
        "lock_status": lock.get("status"),
        "lock_sha256": sha256_file(lock_path),
        "report_sha256": actual_report_sha,
    })

    if lock.get("status") != expected_lock_status:
        failures.append(
            f"unexpected lock status in {directory}: {lock.get('status')!r}"
        )
    if lock.get("report_sha256") != actual_report_sha:
        failures.append(
            f"report SHA mismatch in {directory}: "
            f"expected={lock.get('report_sha256')} actual={actual_report_sha}"
        )
    if report.get("status") != "PASS":
        failures.append(f"report status is not PASS in {directory}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--a2-0-dir", type=Path, required=True)
    parser.add_argument("--a2-1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    a1_dir = args.a1_dir.expanduser().resolve()
    a2_0_dir = args.a2_0_dir.expanduser().resolve()
    a2_1_dir = args.a2_1_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not root.is_dir():
        print(f"STOP: dataset root missing: {root}", file=sys.stderr)
        return 2
    if output_dir.exists():
        print(f"STOP: output directory already exists: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    provenance = {
        "a1": verify_chain(
            a1_dir,
            "V5_P0_A1_TENSOR_HANDOVER_AUDIT_PASS",
            "V5_P0_A1_TENSOR_HANDOVER_LOCK.json",
            "tensor_handover_audit.json",
            "V5_P0_A1_TENSOR_HANDOVER_AUDIT_PASS",
            failures,
        ),
        "a2_0": verify_chain(
            a2_0_dir,
            "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT_PASS",
            "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT_LOCK.json",
            "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT.json",
            "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT_PASS",
            failures,
        ),
        "a2_1": verify_chain(
            a2_1_dir,
            "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_PASS",
            "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_LOCK.json",
            "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT.json",
            "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_PASS",
            failures,
        ),
    }

    normalization_path = root / "normalization.pt"
    topology_path = root / "topology.pt"
    schema_path = root / "feature_schema.json"
    recommendation_path = a2_1_dir / "feature_contract_recommendation.json"
    raw_mask_path = a2_1_dir / "topology_derived_raw_physical_port_mask.pt"

    for path in (
        normalization_path,
        topology_path,
        schema_path,
        recommendation_path,
        raw_mask_path,
    ):
        if not path.is_file():
            failures.append(f"required contract artifact missing: {path}")

    if failures:
        report = {
            "stage": "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_FREEZE",
            "status": "FAIL",
            "failures": failures,
            "warnings": warnings,
        }
        report_path = output_dir / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_FAILED.json"
        write_json(report_path, report)
        atomic_write(
            output_dir / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_HOLD",
            "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_HOLD\n",
        )
        print("V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_HOLD")
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
    mask_artifact = torch.load(
        raw_mask_path,
        map_location="cpu",
        weights_only=False,
    )
    recommendation = load_json(recommendation_path)

    feature_names = [str(name) for name in norm["feature_names"]]
    mean = norm["mean"].detach().cpu()
    std = norm["std"].detach().cpu()
    mask_features = norm["mask_features"].detach().cpu().bool()

    if len(feature_names) != 81:
        failures.append(f"feature_names count={len(feature_names)}, expected 81")
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
            f"expected documented mask metadata defect count=0, "
            f"found {int(mask_features.sum().item())}"
        )
    if str(norm.get("dynamic_transform")) != "log1p_then_standardize":
        failures.append(
            f"unexpected dynamic_transform={norm.get('dynamic_transform')!r}"
        )
    if str(norm.get("mask_transform")) != "identity":
        failures.append(
            f"unexpected mask_transform={norm.get('mask_transform')!r}"
        )
    if str(norm.get("fit_split")) != "train":
        failures.append(
            f"normalization fit_split={norm.get('fit_split')!r}, expected train"
        )
    if int(norm.get("fit_router_epoch_rows")) != 360640:
        failures.append(
            "normalization fit_router_epoch_rows="
            f"{norm.get('fit_router_epoch_rows')!r}, expected 360640"
        )

    if feature_names[30] != "status_flags":
        failures.append(
            f"feature 30={feature_names[30]!r}, expected status_flags"
        )
    if feature_names[71:81] != EXPECTED_MASK_NAMES:
        failures.append(
            f"mask feature names mismatch: {feature_names[71:81]}"
        )

    if len(set(ALL81)) != 81:
        failures.append("ALL81 indices are not unique")
    if len(DYNAMIC70) != 70 or len(set(DYNAMIC70)) != 70:
        failures.append("DYNAMIC70 contract does not contain 70 unique indices")
    if len(PRIMARY58) != 58 or len(set(PRIMARY58)) != 58:
        failures.append("PRIMARY58 contract does not contain 58 unique indices")

    expected_dynamic70 = set(range(81)) - {30} - set(MASK_INDICES)
    if set(DYNAMIC70) != expected_dynamic70:
        failures.append("DYNAMIC70 indices do not match expected schema partition")

    expected_primary58 = (
        expected_dynamic70 - set(ZERO_VARIANCE_TRAIN_INDICES)
    )
    if set(PRIMARY58) != expected_primary58:
        failures.append("PRIMARY58 indices do not match expected informative set")

    raw_mask = mask_artifact.get("physical_port_mask")
    if not isinstance(raw_mask, torch.Tensor):
        failures.append("mask artifact lacks tensor physical_port_mask")
    else:
        raw_mask = raw_mask.detach().cpu()
        if tuple(raw_mask.shape) != (16, 10):
            failures.append(
                f"raw physical mask shape={tuple(raw_mask.shape)}, expected (16,10)"
            )
        if raw_mask.dtype != torch.bool:
            failures.append(
                f"raw physical mask dtype={raw_mask.dtype}, expected torch.bool"
            )
        if raw_mask[:, 0].numel() and not bool(raw_mask[:, 0].all()):
            failures.append("input local port mask is not valid for all routers")
        if raw_mask[:, 5].numel() and not bool(raw_mask[:, 5].all()):
            failures.append("output local port mask is not valid for all routers")
        if not torch.equal(raw_mask[:, :5], raw_mask[:, 5:10]):
            failures.append("input and output physical masks are not identical")
        cardinal_counts = raw_mask[:, [1, 2, 3, 4]].sum(dim=0).tolist()
        if cardinal_counts != [12, 12, 12, 12]:
            failures.append(
                f"cardinal valid-router counts={cardinal_counts}, expected [12,12,12,12]"
            )

    edge_index = topology.get("edge_index")
    if not isinstance(edge_index, torch.Tensor):
        failures.append("topology missing edge_index tensor")
    elif tuple(edge_index.shape) != (2, 48):
        failures.append(
            f"edge_index shape={tuple(edge_index.shape)}, expected (2,48)"
        )

    if recommendation.get("defect_classification") != (
        "RECOVERABLE_TENSOR_CONVERSION_MASK_METADATA_DEFECT"
    ):
        failures.append(
            "A2-1 recommendation does not classify the known recoverable defect"
        )
    if not bool(recommendation.get("contract_freeze_authorized")):
        failures.append("A2-1 did not authorize contract freeze")

    # Build frozen feature rows.
    rows: list[dict[str, Any]] = []
    zero_variance_set = set(ZERO_VARIANCE_TRAIN_INDICES)
    primary_set = set(PRIMARY58)
    dynamic_set = set(DYNAMIC70)
    mask_set = set(MASK_INDICES)

    for index, name in enumerate(feature_names):
        if index == STATUS_INDEX:
            category = "status_or_audit"
        elif index in mask_set:
            category = "stored_standardized_structural_mask"
        else:
            category = "dynamic"

        rows.append({
            "index": index,
            "name": name,
            "category": category,
            "in_all81": True,
            "in_dynamic70": index in dynamic_set,
            "in_primary58": index in primary_set,
            "zero_variance_in_training": index in zero_variance_set,
            "stored_mean": float(mean[index].item()),
            "stored_std": float(std[index].item()),
        })

    contract = {
        "status": "FAIL" if failures else "PASS",
        "stage": "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_FREEZE",
        "contract_version": "V5_P0_A2_D0",
        "dataset_root": str(root),
        "feature_schema": {
            "total_feature_count": 81,
            "feature_names": feature_names,
            "status_index": STATUS_INDEX,
            "status_name": feature_names[STATUS_INDEX],
            "stored_mask_indices": MASK_INDICES,
            "stored_mask_names": feature_names[71:81],
            "zero_variance_training_dynamic_indices": (
                ZERO_VARIANCE_TRAIN_INDICES
            ),
        },
        "normalization": {
            "stored_x_is_already_transformed": True,
            "stored_dynamic_transform": norm.get("dynamic_transform"),
            "stored_mask_transform_metadata": norm.get("mask_transform"),
            "stored_mask_feature_count": int(mask_features.sum().item()),
            "fit_split": norm.get("fit_split"),
            "fit_router_epoch_rows": norm.get("fit_router_epoch_rows"),
            "apply_log1p_in_loader": False,
            "apply_normalization_pt_in_loader": False,
            "refit_normalization": False,
            "normalization_pt_role": [
                "provenance",
                "schema alignment",
                "consistency checking",
                "optional inverse transform for diagnostics",
            ],
        },
        "known_tensor_conversion_issue": {
            "classification": (
                "RECOVERABLE_TENSOR_CONVERSION_MASK_METADATA_DEFECT"
            ),
            "description": (
                "Physical-port validity features 71..80 were not marked by "
                "mask_features and therefore followed log1p_then_standardize "
                "despite mask_transform='identity'."
            ),
            "original_tensors_modified": False,
            "repair_strategy": (
                "Exclude stored standardized mask channels from the primary "
                "learned input and supply topology-derived raw Boolean masks "
                "separately."
            ),
            "stored_mask_boolean_cast_forbidden": True,
        },
        "feature_variants": {
            "ALL81_COMPATIBILITY": {
                "indices": ALL81,
                "feature_count": 81,
                "normalization": "use stored x directly",
                "structural_mask": (
                    "stored standardized mask channels remain inside x"
                ),
                "use": (
                    "package-as-delivered compatibility baseline only"
                ),
                "primary_model": False,
            },
            "DYNAMIC70": {
                "indices": DYNAMIC70,
                "feature_count": 70,
                "excluded_indices": [STATUS_INDEX] + MASK_INDICES,
                "normalization": "use stored x directly",
                "structural_mask": (
                    "topology-derived raw Boolean mask supplied separately"
                ),
                "use": (
                    "schema-complete dynamic ablation retaining P0 "
                    "zero-variance stall channels"
                ),
                "primary_model": False,
            },
            "PRIMARY58": {
                "indices": PRIMARY58,
                "feature_count": 58,
                "excluded_indices": (
                    [STATUS_INDEX]
                    + ZERO_VARIANCE_TRAIN_INDICES
                    + MASK_INDICES
                ),
                "normalization": "use stored x directly",
                "structural_mask": (
                    "topology-derived raw Boolean [16,10] mask supplied "
                    "separately"
                ),
                "use": (
                    "primary P0 learned feature contract"
                ),
                "primary_model": True,
            },
        },
        "structural_inputs": {
            "edge_index": {
                "source": str(topology_path),
                "shape": [2, 48],
                "learned_router_id_embedding": False,
            },
            "physical_port_mask": {
                "source": str(raw_mask_path),
                "shape": [16, 10],
                "dtype": "torch.bool",
                "feature_names": EXPECTED_MASK_NAMES,
                "learned_as_ordinary_feature_in_primary": False,
                "allowed_uses": [
                    "fixed port validity gating",
                    "structural consistency checks",
                    "mask-only shortcut baseline",
                ],
            },
            "router_coordinates": {
                "learned_input_in_primary": False,
                "reason": (
                    "avoid router-position memorization; coordinates remain "
                    "structural provenance only"
                ),
            },
        },
        "loader_contract": {
            "input_tensor_source_shape": "[T,16,81]",
            "window_output_layout": "[16,F,window]",
            "window_target": "final epoch",
            "default_window": 32,
            "default_stride": 8,
            "primary_feature_variant": "PRIMARY58",
            "normalization_applied_by_loader": False,
            "metadata_concatenated_to_x": False,
            "stored_mask_channels_boolean_cast": False,
            "raw_mask_supplied_separately": True,
        },
        "evaluation_boundary": {
            "validation_tensors_used_for_contract_selection": False,
            "test_tensors_used_for_contract_selection": False,
            "test_performance_evaluated": False,
            "feature_selection_basis": (
                "training-only statistics plus structural semantics and "
                "frozen topology"
            ),
        },
        "provenance": {
            **provenance,
            "feature_schema_sha256": sha256_file(schema_path),
            "normalization_sha256": sha256_file(normalization_path),
            "topology_sha256": sha256_file(topology_path),
            "raw_mask_sha256": sha256_file(raw_mask_path),
            "a2_1_recommendation_sha256": sha256_file(recommendation_path),
            "contract_script_sha256": sha256_file(Path(__file__)),
        },
        "failures": failures,
        "warnings": warnings,
        "next_stage": (
            "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE"
            if not failures
            else "HOLD_FIX_A2_CONTRACT"
        ),
    }

    report_path = (
        output_dir
        / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT.json"
    )
    write_json(report_path, contract)
    write_csv(
        output_dir / "V5_P0_A2_FEATURE_INDEX_CONTRACT.csv",
        rows,
    )

    markdown = f"""# V5 P0-A2 Feature and Normalization Contract

## Status

**Contract version:** `V5_P0_A2_D0`

**Primary feature variant:** `PRIMARY58`

## Frozen input variants

### ALL81_COMPATIBILITY

- Feature count: 81
- Uses the package exactly as delivered
- Includes the constant `status_flags` channel and stored standardized mask
  channels
- Compatibility baseline only

### DYNAMIC70

- Feature count: 70
- Indices: 0-29 and 31-70
- Excludes `status_flags` and stored mask channels
- Retains zero-variance P0 stall channels for schema-complete ablation

### PRIMARY58

- Feature count: 58
- Indices: 0-29, 31-55, 62, 64, 65
- Excludes status, twelve zero-variance training channels, and stored mask
  channels
- Receives topology-derived raw Boolean physical-port masks separately
- This is the primary P0 model input

## Normalization

The stored `x` tensor is already transformed. The loader must not apply
`log1p`, `normalization.pt`, or any newly fitted normalization.

## Recoverable mask metadata defect

Features 71-80 were named as physical-port validity masks but were not marked
inside `mask_features`. They therefore followed `log1p_then_standardize`
despite the metadata field `mask_transform='identity'`.

The original tensors remain immutable. The primary loader excludes those
stored mask channels and uses the audited topology-derived Boolean mask:

`{raw_mask_path}`

## Test boundary

No validation or test performance was used to freeze this contract.

## Next stage

`V5_P0_A3_LOADER_CONTRACT_AND_SMOKE`
"""
    markdown_path = (
        output_dir
        / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT.md"
    )
    atomic_write(markdown_path, markdown)

    lock = {
        "status": (
            "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_PASS"
            if not failures
            else "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_HOLD"
        ),
        "contract_version": "V5_P0_A2_D0",
        "contract_json_sha256": sha256_file(report_path),
        "contract_markdown_sha256": sha256_file(markdown_path),
        "feature_index_csv_sha256": sha256_file(
            output_dir / "V5_P0_A2_FEATURE_INDEX_CONTRACT.csv"
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "raw_mask_sha256": sha256_file(raw_mask_path),
        "primary_feature_variant": "PRIMARY58",
        "primary_feature_count": 58,
        "normalization_applied_by_loader": False,
        "validation_performance_used": False,
        "test_performance_evaluated": False,
        "next_stage": contract["next_stage"],
    }
    lock_path = (
        output_dir
        / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_LOCK.json"
    )
    write_json(lock_path, lock)

    print(
        "===== V5 P0-A2 FEATURE/NORMALIZATION CONTRACT FREEZE ====="
    )
    print(f"contract_version: V5_P0_A2_D0")
    print(f"all81_feature_count: {len(ALL81)}")
    print(f"dynamic70_feature_count: {len(DYNAMIC70)}")
    print(f"primary58_feature_count: {len(PRIMARY58)}")
    print(f"status_index: {STATUS_INDEX}")
    print(
        "zero_variance_training_dynamic_count: "
        f"{len(ZERO_VARIANCE_TRAIN_INDICES)}"
    )
    print(f"stored_mask_feature_count: {len(MASK_INDICES)}")
    print(
        "normalization_applied_by_loader: false"
    )
    print(
        "stored_mask_boolean_cast_allowed: false"
    )
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
            output_dir
            / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_HOLD",
            "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_HOLD\n",
        )
        print("next_stage: HOLD_FIX_A2_CONTRACT")
        print(
            "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_HOLD"
        )
        return 1

    atomic_write(
        output_dir
        / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_PASS",
        "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_PASS\n",
    )
    print("primary_feature_variant: PRIMARY58")
    print(
        "next_stage: V5_P0_A3_LOADER_CONTRACT_AND_SMOKE"
    )
    print(
        "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
