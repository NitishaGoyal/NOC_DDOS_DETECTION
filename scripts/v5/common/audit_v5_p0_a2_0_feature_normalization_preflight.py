#!/usr/bin/env python3
"""
V5 P0-A2-0 Feature and Normalization Contract Preflight

Read-only preflight that:
- verifies the frozen P0-A1 PASS/lock/report chain;
- resolves all 81 feature definitions from feature_schema.json;
- inventories normalization.pt recursively;
- streams TRAIN tensors only to compute per-feature statistics;
- proposes dynamic / structural-mask / status feature groups;
- determines whether stored x appears already standardized;
- writes evidence for the later frozen A2 feature contract.

It performs no feature selection, model training, model inference,
threshold selection, checkpoint selection, or test-performance evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "")
        return

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_name(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in (
            "name",
            "feature_name",
            "column",
            "original_name",
        ):
            if key in item:
                return str(item[key])
    return ""


def find_feature_list(value: Any) -> list[Any] | None:
    candidates: list[list[Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, list):
            if len(node) == 81:
                names = [feature_name(item) for item in node]
                if all(names):
                    candidates.append(node)
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            for child in node.values():
                visit(child)

    visit(value)

    if not candidates:
        return None

    unique: dict[tuple[str, ...], list[Any]] = {}
    for candidate in candidates:
        signature = tuple(feature_name(item) for item in candidate)
        unique.setdefault(signature, candidate)

    if len(unique) != 1:
        return None
    return next(iter(unique.values()))


def flatten_object(
    value: Any,
    prefix: str = "",
) -> Iterable[dict[str, Any]]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        row = {
            "key": prefix,
            "object_type": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
            "finite": (
                bool(torch.isfinite(tensor).all())
                if tensor.is_floating_point()
                else True
            ),
        }
        if tensor.numel() and tensor.is_floating_point():
            row["min"] = float(tensor.min().item())
            row["max"] = float(tensor.max().item())
            row["mean"] = float(tensor.double().mean().item())
        yield row
        return

    if isinstance(value, dict):
        if not value:
            yield {
                "key": prefix,
                "object_type": "empty_dict",
            }
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_object(child, child_prefix)
        return

    if isinstance(value, (list, tuple)):
        if not value:
            yield {
                "key": prefix,
                "object_type": type(value).__name__,
                "length": 0,
            }
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            yield from flatten_object(child, child_prefix)
        return

    yield {
        "key": prefix,
        "object_type": type(value).__name__,
        "value_repr": repr(value)[:500],
    }


def infer_group(name: str, definition: Any) -> tuple[str, str]:
    text = name.lower()
    if isinstance(definition, dict):
        text += " " + json.dumps(definition, sort_keys=True).lower()

    status_tokens = (
        "status",
        "instrumentation",
        "overflow",
        "invalid",
        "validity",
        "dropped_sample",
        "counter_error",
    )
    mask_tokens = (
        "physical",
        "port_mask",
        "valid_port",
        "input_mask",
        "output_mask",
        "in_mask",
        "out_mask",
    )

    if any(token in text for token in status_tokens):
        return "status_or_audit", "name_or_definition_status_token"
    if (
        "mask" in text
        and any(token in text for token in ("port", "physical", "valid"))
    ) or any(token in text for token in mask_tokens):
        return "structural_mask", "name_or_definition_mask_token"
    return "dynamic_candidate", "default_nonmask_nonstatus"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    a1_dir = args.a1_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not root.is_dir():
        print(f"STOP: dataset root missing: {root}", file=sys.stderr)
        return 2
    if not a1_dir.is_dir():
        print(f"STOP: A1 directory missing: {a1_dir}", file=sys.stderr)
        return 2
    if output_dir.exists():
        print(
            f"STOP: output directory already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    # Verify A1 PASS chain.
    a1_pass = a1_dir / "V5_P0_A1_TENSOR_HANDOVER_AUDIT_PASS"
    a1_lock_path = a1_dir / "V5_P0_A1_TENSOR_HANDOVER_LOCK.json"
    a1_report_path = a1_dir / "tensor_handover_audit.json"

    for path in (a1_pass, a1_lock_path, a1_report_path):
        if not path.is_file():
            failures.append(f"required A1 artifact missing: {path}")

    a1_lock: dict[str, Any] = {}
    if a1_lock_path.is_file() and a1_report_path.is_file():
        try:
            a1_lock = load_json(a1_lock_path)
            a1_report = load_json(a1_report_path)
        except Exception as exc:
            failures.append(f"failed to parse A1 artifacts: {exc!r}")
        else:
            if a1_lock.get("status") != (
                "V5_P0_A1_TENSOR_HANDOVER_AUDIT_PASS"
            ):
                failures.append(
                    "A1 lock does not authorize downstream work"
                )
            actual_report_sha = sha256_file(a1_report_path)
            expected_report_sha = a1_lock.get("report_sha256")
            if actual_report_sha != expected_report_sha:
                failures.append(
                    "A1 report SHA mismatch: "
                    f"expected={expected_report_sha} "
                    f"actual={actual_report_sha}"
                )
            if a1_report.get("status") != "PASS":
                failures.append("A1 report status is not PASS")
            if not bool(a1_report.get("training_authorized")):
                failures.append(
                    "A1 report did not authorize the next stage"
                )

    # Resolve feature schema.
    schema_path = root / "feature_schema.json"
    if not schema_path.is_file():
        failures.append(f"feature schema missing: {schema_path}")
        feature_list = None
    else:
        try:
            schema = load_json(schema_path)
            feature_list = find_feature_list(schema)
        except Exception as exc:
            failures.append(
                f"failed to parse feature schema: {exc!r}"
            )
            feature_list = None

    if feature_list is None:
        failures.append(
            "could not resolve a unique 81-entry feature list"
        )
        feature_names: list[str] = []
    else:
        feature_names = [
            feature_name(item)
            for item in feature_list
        ]

    if len(feature_names) != 81:
        failures.append(
            f"feature count={len(feature_names)}, expected 81"
        )

    normalized_names = [
        name.strip().lower()
        for name in feature_names
    ]
    duplicates = sorted(
        name
        for name, count in Counter(normalized_names).items()
        if count > 1
    )
    if duplicates:
        failures.append(f"duplicate feature names: {duplicates}")

    # Inventory normalization.pt.
    normalization_path = root / "normalization.pt"
    normalization_rows: list[dict[str, Any]] = []
    if not normalization_path.is_file():
        failures.append(
            f"normalization object missing: {normalization_path}"
        )
    else:
        try:
            normalization = torch.load(
                normalization_path,
                map_location="cpu",
                weights_only=False,
            )
            normalization_rows = list(
                flatten_object(normalization)
            )
        except Exception as exc:
            failures.append(
                f"failed to load normalization.pt: {exc!r}"
            )

    for row in normalization_rows:
        if row.get("object_type") == "tensor":
            if row.get("finite") is False:
                failures.append(
                    "non-finite normalization tensor: "
                    f"{row.get('key')}"
                )

    # Stream training tensors only.
    train_dir = root / "runs" / "train"
    train_paths = sorted(train_dir.glob("*.pt"))
    if len(train_paths) != 32:
        failures.append(
            f"training tensor count={len(train_paths)}, expected 32"
        )

    feature_count = 81
    sum_x = torch.zeros(
        feature_count,
        dtype=torch.float64,
    )
    sumsq_x = torch.zeros(
        feature_count,
        dtype=torch.float64,
    )
    min_x = torch.full(
        (feature_count,),
        float("inf"),
        dtype=torch.float64,
    )
    max_x = torch.full(
        (feature_count,),
        float("-inf"),
        dtype=torch.float64,
    )
    zero_count = torch.zeros(
        feature_count,
        dtype=torch.int64,
    )
    one_count = torch.zeros(
        feature_count,
        dtype=torch.int64,
    )
    nonfinite_count = torch.zeros(
        feature_count,
        dtype=torch.int64,
    )
    binary_possible = torch.ones(
        feature_count,
        dtype=torch.bool,
    )
    sample_count = 0
    train_run_rows: list[dict[str, Any]] = []

    for path in train_paths:
        try:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            x = payload["x"].detach().cpu()
        except Exception as exc:
            failures.append(
                f"failed to load training tensor {path.name}: {exc!r}"
            )
            continue

        if tuple(x.shape[1:]) != (16, 81):
            failures.append(
                f"{path.name} x shape={tuple(x.shape)}, "
                "expected [T,16,81]"
            )
            continue

        flat = x.reshape(-1, feature_count).to(
            dtype=torch.float64
        )
        finite_mask = torch.isfinite(flat)
        nonfinite_per_feature = (
            (~finite_mask).sum(dim=0).to(torch.int64)
        )
        nonfinite_count += nonfinite_per_feature

        safe = torch.where(
            finite_mask,
            flat,
            torch.zeros_like(flat),
        )
        sum_x += safe.sum(dim=0)
        sumsq_x += (safe * safe).sum(dim=0)

        feature_min = torch.where(
            finite_mask,
            flat,
            torch.full_like(flat, float("inf")),
        ).min(dim=0).values
        feature_max = torch.where(
            finite_mask,
            flat,
            torch.full_like(flat, float("-inf")),
        ).max(dim=0).values
        min_x = torch.minimum(min_x, feature_min)
        max_x = torch.maximum(max_x, feature_max)

        zero_count += (flat == 0).sum(dim=0).to(torch.int64)
        one_count += (flat == 1).sum(dim=0).to(torch.int64)
        binary_possible &= (
            ((flat == 0) | (flat == 1)).all(dim=0)
        )

        current_samples = int(flat.shape[0])
        sample_count += current_samples
        train_run_rows.append({
            "filename": path.name,
            "T": int(x.shape[0]),
            "router_rows": current_samples,
            "x_dtype": str(x.dtype),
            "x_min": float(x.min().item()),
            "x_max": float(x.max().item()),
            "x_mean": float(x.double().mean().item()),
            "x_std": float(
                x.double().std(unbiased=False).item()
            ),
        })

    if sample_count == 0:
        failures.append(
            "no valid training feature rows were accumulated"
        )

    feature_rows: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    dynamic_indices: list[int] = []
    mask_indices: list[int] = []
    status_indices: list[int] = []

    standardized_feature_count = 0
    dynamic_feature_count_for_test = 0

    if sample_count > 0 and len(feature_names) == 81:
        denominator = float(sample_count)
        means = sum_x / denominator
        variances = torch.clamp(
            sumsq_x / denominator - means * means,
            min=0.0,
        )
        stds = torch.sqrt(variances)

        for index, name in enumerate(feature_names):
            definition = feature_list[index]
            proposed_group, group_reason = infer_group(
                name,
                definition,
            )
            group_counts[proposed_group] += 1

            if proposed_group == "dynamic_candidate":
                dynamic_indices.append(index)
                dynamic_feature_count_for_test += 1
                if (
                    abs(float(means[index].item())) <= 0.25
                    and 0.50
                    <= float(stds[index].item())
                    <= 1.50
                ):
                    standardized_feature_count += 1
            elif proposed_group == "structural_mask":
                mask_indices.append(index)
            else:
                status_indices.append(index)

            feature_rows.append({
                "index": index,
                "name": name,
                "proposed_group": proposed_group,
                "group_reason": group_reason,
                "mean_train": float(means[index].item()),
                "std_train": float(stds[index].item()),
                "min_train": float(min_x[index].item()),
                "max_train": float(max_x[index].item()),
                "zero_fraction_train": (
                    int(zero_count[index].item()) / denominator
                ),
                "one_fraction_train": (
                    int(one_count[index].item()) / denominator
                ),
                "binary_train": bool(
                    binary_possible[index].item()
                ),
                "nonfinite_count_train": int(
                    nonfinite_count[index].item()
                ),
                "definition_json": (
                    json.dumps(definition, sort_keys=True)
                    if not isinstance(definition, str)
                    else json.dumps({"name": definition})
                ),
            })

    if bool((nonfinite_count != 0).any()):
        failures.append(
            "training x contains non-finite feature values"
        )

    if dynamic_feature_count_for_test:
        standardized_fraction = (
            standardized_feature_count
            / dynamic_feature_count_for_test
        )
    else:
        standardized_fraction = 0.0

    appears_already_standardized = (
        standardized_fraction >= 0.75
    )

    binary_indices = [
        index
        for index in range(feature_count)
        if bool(binary_possible[index].item())
    ]

    contract_evidence = {
        "feature_count": len(feature_names),
        "proposed_group_counts": dict(group_counts),
        "proposed_dynamic_indices": dynamic_indices,
        "proposed_structural_mask_indices": mask_indices,
        "proposed_status_or_audit_indices": status_indices,
        "binary_training_feature_indices": binary_indices,
        "training_router_epoch_rows": sample_count,
        "dynamic_features_near_standardized_count": (
            standardized_feature_count
        ),
        "dynamic_feature_count_evaluated": (
            dynamic_feature_count_for_test
        ),
        "dynamic_features_near_standardized_fraction": (
            standardized_fraction
        ),
        "stored_x_appears_already_standardized": (
            appears_already_standardized
        ),
        "normalization_application_recommendation": (
            "DO_NOT_NORMALIZE_AGAIN_PENDING_A2_CONTRACT"
            if appears_already_standardized
            else "REVIEW_NORMALIZATION_OBJECT_AND_SCHEMA_BEFORE_TRAINING"
        ),
        "group_classification_is_provisional": True,
        "feature_selection_performed": False,
    }

    report = {
        "stage": (
            "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT"
        ),
        "status": "FAIL" if failures else "PASS",
        "dataset_root": str(root),
        "a1_dir": str(a1_dir),
        "failures": failures,
        "warnings": warnings,
        "a1_provenance": {
            "a1_report_sha256": (
                sha256_file(a1_report_path)
                if a1_report_path.is_file()
                else None
            ),
            "a1_lock_sha256": (
                sha256_file(a1_lock_path)
                if a1_lock_path.is_file()
                else None
            ),
            "a1_pass_marker_present": a1_pass.is_file(),
        },
        "schema_sha256": (
            sha256_file(schema_path)
            if schema_path.is_file()
            else None
        ),
        "normalization_sha256": (
            sha256_file(normalization_path)
            if normalization_path.is_file()
            else None
        ),
        "normalization_inventory_entry_count": len(
            normalization_rows
        ),
        "training_run_count": len(train_paths),
        "contract_evidence": contract_evidence,
        "audit_boundary": {
            "train_tensors_read_for_statistics": True,
            "validation_tensors_read": False,
            "test_tensors_read": False,
            "feature_selection_performed": False,
            "model_training_performed": False,
            "model_inference_performed": False,
            "threshold_selection_performed": False,
            "checkpoint_selection_performed": False,
            "test_performance_evaluated": False,
        },
        "next_stage": (
            "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_FREEZE"
            if not failures
            else "HOLD_FIX_A2_PREFLIGHT"
        ),
    }

    write_csv(
        output_dir / "feature_statistics_train.csv",
        feature_rows,
    )
    write_csv(
        output_dir / "normalization_inventory.csv",
        normalization_rows,
    )
    write_csv(
        output_dir / "training_run_inventory.csv",
        train_run_rows,
    )
    write_json(
        output_dir / "feature_normalization_evidence.json",
        contract_evidence,
    )

    report_path = (
        output_dir
        / "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT.json"
    )
    write_json(report_path, report)

    lock = {
        "status": (
            "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT_PASS"
            if not failures
            else "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT_HOLD"
        ),
        "report_sha256": sha256_file(report_path),
        "script_sha256": sha256_file(Path(__file__)),
        "dataset_root": str(root),
        "feature_schema_sha256": report["schema_sha256"],
        "normalization_sha256": report[
            "normalization_sha256"
        ],
        "feature_count": len(feature_names),
        "training_run_count": len(train_paths),
        "feature_selection_performed": False,
        "validation_tensors_read": False,
        "test_tensors_read": False,
        "next_stage": report["next_stage"],
    }
    write_json(
        output_dir
        / "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT_LOCK.json",
        lock,
    )

    print(
        "===== V5 P0-A2-0 FEATURE/NORMALIZATION PREFLIGHT ====="
    )
    print(f"dataset_root: {root}")
    print(f"feature_count: {len(feature_names)}")
    print(f"training_run_count: {len(train_paths)}")
    print(
        "training_router_epoch_rows: "
        f"{sample_count}"
    )
    print(
        "proposed_group_counts: "
        f"{dict(group_counts)}"
    )
    print(
        "binary_training_feature_count: "
        f"{len(binary_indices)}"
    )
    print(
        "dynamic_features_near_standardized_fraction: "
        f"{standardized_fraction:.6f}"
    )
    print(
        "stored_x_appears_already_standardized: "
        f"{appears_already_standardized}"
    )
    print(
        "normalization_inventory_entry_count: "
        f"{len(normalization_rows)}"
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
        atomic_write_text(
            output_dir
            / "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT_HOLD",
            "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT_HOLD\n",
        )
        print(
            "next_stage: HOLD_FIX_A2_PREFLIGHT"
        )
        print(
            "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT_HOLD"
        )
        return 1

    atomic_write_text(
        output_dir
        / "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT_PASS",
        "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT_PASS\n",
    )
    print(
        "next_stage: "
        "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_FREEZE"
    )
    print(
        "V5_P0_A2_0_FEATURE_NORMALIZATION_PREFLIGHT_PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
