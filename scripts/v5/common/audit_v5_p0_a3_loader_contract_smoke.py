#!/usr/bin/env python3
"""
V5 P0-A3 Loader Contract and Smoke Audit.

Checks:
- A2 and A3-0 provenance chains;
- supplied loader signature and current SHA;
- exact source-window equality;
- final-epoch target equality;
- no run-boundary or split-boundary crossing;
- frozen PRIMARY58 selection without another normalization;
- topology-derived raw Boolean mask delivery;
- deterministic sample and deterministic batch behavior;
- train/validation/test structural behavior without performance evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


REQUIRED_BASE_KEYS = {
    "x",
    "edge_index",
    "y_attack",
    "y_attacker_count",
    "y_source",
    "y_transit",
    "y_victim",
    "y_attack_path",
    "role_mask",
    "epoch_id",
    "case_id",
    "run_id",
    "mode",
    "window_start",
    "window_target",
}

LABEL_KEYS = [
    "y_attack",
    "y_attacker_count",
    "y_source",
    "y_transit",
    "y_victim",
    "y_attack_path",
    "role_mask",
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"expected scalar tensor, got {tuple(value.shape)}"
            )
        return value.detach().cpu().item()
    return value


def metadata_text(value: Any) -> str:
    result = scalar(value)
    if isinstance(result, bytes):
        return result.decode("utf-8")
    return str(result)


def exact_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(
        right,
        torch.Tensor,
    ):
        return torch.equal(
            left.detach().cpu(),
            right.detach().cpu(),
        )

    if isinstance(left, torch.Tensor):
        if left.numel() != 1:
            return False
        return left.detach().cpu().item() == right

    if isinstance(right, torch.Tensor):
        if right.numel() != 1:
            return False
        return left == right.detach().cpu().item()

    if isinstance(left, dict) and isinstance(right, dict):
        return (
            left.keys() == right.keys()
            and all(
                exact_equal(left[key], right[key])
                for key in left
            )
        )

    if isinstance(left, (list, tuple)) and isinstance(
        right,
        type(left),
    ):
        return (
            len(left) == len(right)
            and all(
                exact_equal(a, b)
                for a, b in zip(left, right)
            )
        )

    return left == right


def verify_chain(
    directory: Path,
    marker_name: str,
    lock_name: str | None,
    report_name: str,
    expected_lock_status: str | None,
    failures: list[str],
) -> dict[str, Any]:
    marker = directory / marker_name
    report_path = directory / report_name

    if not marker.is_file():
        failures.append(f"provenance marker missing: {marker}")
    if not report_path.is_file():
        failures.append(f"provenance report missing: {report_path}")

    result: dict[str, Any] = {
        "directory": str(directory),
        "marker_present": marker.is_file(),
        "report_path": str(report_path),
    }

    if not report_path.is_file():
        return result

    report = load_json(report_path)
    result["report_sha256"] = sha256_file(report_path)
    result["report_status"] = report.get("status")

    if report.get("status") != "PASS":
        failures.append(
            f"provenance report is not PASS: {report_path}"
        )

    if lock_name is not None:
        lock_path = directory / lock_name
        if not lock_path.is_file():
            failures.append(f"provenance lock missing: {lock_path}")
            return result

        lock = load_json(lock_path)
        result["lock_path"] = str(lock_path)
        result["lock_sha256"] = sha256_file(lock_path)
        result["lock_status"] = lock.get("status")

        if lock.get("status") != expected_lock_status:
            failures.append(
                f"unexpected lock status: {lock.get('status')!r}"
            )

        expected_report_sha = (
            lock.get("contract_json_sha256")
            or lock.get("report_sha256")
        )
        actual_report_sha = sha256_file(report_path)
        if expected_report_sha != actual_report_sha:
            failures.append(
                f"provenance report SHA mismatch: "
                f"expected={expected_report_sha} "
                f"actual={actual_report_sha}"
            )

    return result


def import_class(path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location(
        f"imported_{class_name}",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(
            f"{path} does not define {class_name}"
        )
    return cls


def load_split_sources(
    root: Path,
    split: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Path],
]:
    sources: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}

    for path in sorted((root / "runs" / split).glob("*.pt")):
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        run_id = metadata_text(payload["run_id"])
        if run_id in sources:
            raise RuntimeError(
                f"duplicate run_id in {split}: {run_id}"
            )
        sources[run_id] = payload
        paths[run_id] = path

    return sources, paths


def expected_window_count(
    source: dict[str, Any],
    window: int,
    stride: int,
) -> int:
    total_epochs = int(source["x"].shape[0])
    if total_epochs < window:
        return 0
    return 1 + (total_epochs - window) // stride


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--a2-1-dir", type=Path, required=True)
    parser.add_argument("--a3-0-dir", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    a2_dir = args.a2_dir.expanduser().resolve()
    a2_1_dir = args.a2_1_dir.expanduser().resolve()
    a3_0_dir = args.a3_0_dir.expanduser().resolve()
    wrapper_path = args.wrapper.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    window = int(args.window)
    stride = int(args.stride)

    if output_dir.exists():
        print(
            f"STOP: output directory already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    provenance = {
        "a2": verify_chain(
            a2_dir,
            "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_PASS",
            "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_LOCK.json",
            "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT.json",
            "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_PASS",
            failures,
        ),
        "a3_0": verify_chain(
            a3_0_dir,
            "V5_P0_A3_0_LOADER_INTERFACE_INSPECTION_COMPLETE",
            None,
            "V5_P0_A3_0_LOADER_INTERFACE_INSPECTION.json",
            None,
            failures,
        ),
    }

    loader_path = root / "dataset_loader.py"
    topology_path = root / "topology.pt"
    mask_path = (
        a2_1_dir
        / "topology_derived_raw_physical_port_mask.pt"
    )

    for path in (
        loader_path,
        topology_path,
        mask_path,
        wrapper_path,
    ):
        if not path.is_file():
            failures.append(f"required file missing: {path}")

    if failures:
        report = {
            "stage": "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE",
            "status": "FAIL",
            "failures": failures,
            "warnings": warnings,
        }
        report_path = (
            output_dir
            / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE.json"
        )
        write_json(report_path, report)
        atomic_write(
            output_dir
            / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_HOLD",
            "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_HOLD\n",
        )
        print("V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_HOLD")
        return 1

    a3_0_report = load_json(
        a3_0_dir
        / "V5_P0_A3_0_LOADER_INTERFACE_INSPECTION.json"
    )
    current_loader_sha = sha256_file(loader_path)
    if a3_0_report.get("loader_sha256") != current_loader_sha:
        failures.append(
            "supplied loader changed after A3-0 inspection"
        )
    construction = a3_0_report.get("construction", {})
    if not bool(construction.get("success")):
        failures.append(
            "A3-0 did not successfully construct supplied loader"
        )
    if construction.get("dataset_class") != "V5P0WindowDataset":
        failures.append(
            "A3-0 resolved unexpected dataset class: "
            f"{construction.get('dataset_class')!r}"
        )

    ContractDataset = import_class(
        wrapper_path,
        "V5P0ContractWindowDataset",
    )

    topology = torch.load(
        topology_path,
        map_location="cpu",
        weights_only=False,
    )
    topology_edge_index = (
        topology["edge_index"].detach().cpu()
    )

    mask_payload = torch.load(
        mask_path,
        map_location="cpu",
        weights_only=False,
    )
    physical_mask = (
        mask_payload["physical_port_mask"]
        .detach()
        .cpu()
    )

    split_reports: dict[str, Any] = {}
    total_samples_checked = 0
    target_representations: Counter[str] = Counter()

    first_train_dataset = None

    for split in ("train", "validation", "test"):
        dataset = ContractDataset(
            root=root,
            split=split,
            contract_dir=a2_dir,
            mask_audit_dir=a2_1_dir,
            window=window,
            stride=stride,
            active_only=False,
            feature_variant="PRIMARY58",
        )

        if split == "train":
            first_train_dataset = dataset

        sources, source_paths = load_split_sources(
            root,
            split,
        )
        expected_total = sum(
            expected_window_count(
                source,
                window,
                stride,
            )
            for source in sources.values()
        )

        if len(dataset) != expected_total:
            failures.append(
                f"{split}: loader length={len(dataset)}, "
                f"expected={expected_total}"
            )

        seen_starts: dict[str, list[int]] = defaultdict(list)
        split_failures_before = len(failures)

        for index in range(len(dataset)):
            item = dataset[index]
            total_samples_checked += 1

            if index % 500 == 0:
                print(
                    f"{split}: checking sample "
                    f"{index}/{len(dataset)}"
                )

            missing = REQUIRED_BASE_KEYS - set(item.keys())
            if missing:
                failures.append(
                    f"{split}[{index}] missing keys: "
                    f"{sorted(missing)}"
                )
                continue

            if "physical_port_mask" not in item:
                failures.append(
                    f"{split}[{index}] lacks physical_port_mask"
                )
                continue

            x = item["x"]
            if not isinstance(x, torch.Tensor):
                failures.append(
                    f"{split}[{index}] x is not a tensor"
                )
                continue
            if tuple(x.shape) != (16, 58, window):
                failures.append(
                    f"{split}[{index}] x shape="
                    f"{tuple(x.shape)}, expected=(16,58,{window})"
                )
            if x.is_floating_point() and not bool(
                torch.isfinite(x).all()
            ):
                failures.append(
                    f"{split}[{index}] x contains non-finite values"
                )

            item_mask = item["physical_port_mask"]
            if not isinstance(item_mask, torch.Tensor):
                failures.append(
                    f"{split}[{index}] mask is not tensor"
                )
            else:
                if tuple(item_mask.shape) != (16, 10):
                    failures.append(
                        f"{split}[{index}] mask shape="
                        f"{tuple(item_mask.shape)}"
                    )
                if item_mask.dtype != torch.bool:
                    failures.append(
                        f"{split}[{index}] mask dtype="
                        f"{item_mask.dtype}"
                    )
                if not torch.equal(
                    item_mask.detach().cpu(),
                    physical_mask,
                ):
                    failures.append(
                        f"{split}[{index}] mask differs "
                        "from audited artifact"
                    )

            edge_index = item["edge_index"]
            if not isinstance(edge_index, torch.Tensor):
                failures.append(
                    f"{split}[{index}] edge_index is not tensor"
                )
            elif not torch.equal(
                edge_index.detach().cpu(),
                topology_edge_index,
            ):
                failures.append(
                    f"{split}[{index}] edge_index mismatch"
                )

            run_id = metadata_text(item["run_id"])
            if run_id not in sources:
                failures.append(
                    f"{split}[{index}] run_id is outside split: "
                    f"{run_id}"
                )
                continue

            source = sources[run_id]
            start = int(scalar(item["window_start"]))
            target_index = start + window - 1
            seen_starts[run_id].append(start)

            if start < 0:
                failures.append(
                    f"{split}[{index}] negative window_start"
                )
                continue
            if target_index >= int(source["x"].shape[0]):
                failures.append(
                    f"{split}[{index}] window crosses run boundary"
                )
                continue

            expected_all81 = (
                source["x"][start:start + window]
                .permute(1, 2, 0)
                .contiguous()
            )
            expected_primary58 = (
                expected_all81[
                    :,
                    dataset.feature_indices,
                    :,
                ]
                .contiguous()
            )

            if not torch.equal(
                x.detach().cpu(),
                expected_primary58,
            ):
                failures.append(
                    f"{split}[{index}] x is not an exact "
                    "PRIMARY58 source slice"
                )

            if metadata_text(item["case_id"]) != metadata_text(
                source["case_id"]
            ):
                failures.append(
                    f"{split}[{index}] case_id mismatch"
                )
            if metadata_text(item["mode"]) != metadata_text(
                source["mode"]
            ):
                failures.append(
                    f"{split}[{index}] mode mismatch"
                )

            expected_epoch = source["epoch_id"][target_index]
            if not exact_equal(
                item["epoch_id"],
                expected_epoch,
            ):
                failures.append(
                    f"{split}[{index}] epoch_id is not "
                    "the final-window epoch"
                )

            for key in LABEL_KEYS:
                expected_label = source[key][target_index]
                if not exact_equal(
                    item[key],
                    expected_label,
                ):
                    failures.append(
                        f"{split}[{index}] {key} is not "
                        "the final-window target"
                    )

            raw_window_target = scalar(
                item["window_target"]
            )
            expected_epoch_scalar = scalar(expected_epoch)
            if raw_window_target == target_index:
                target_representations[
                    "source_index"
                ] += 1
            elif raw_window_target == expected_epoch_scalar:
                target_representations[
                    "epoch_id"
                ] += 1
            else:
                target_representations[
                    "other"
                ] += 1
                failures.append(
                    f"{split}[{index}] window_target="
                    f"{raw_window_target!r} is neither final "
                    f"source index {target_index} nor epoch "
                    f"{expected_epoch_scalar!r}"
                )

        for run_id, source in sources.items():
            expected_starts = list(
                range(
                    0,
                    int(source["x"].shape[0]) - window + 1,
                    stride,
                )
            )
            actual_starts = seen_starts.get(run_id, [])
            if actual_starts != expected_starts:
                failures.append(
                    f"{split}: start sequence mismatch for "
                    f"{run_id}; expected_count="
                    f"{len(expected_starts)} actual_count="
                    f"{len(actual_starts)}"
                )

        split_reports[split] = {
            "dataset_length": len(dataset),
            "expected_window_count": expected_total,
            "source_run_count": len(sources),
            "samples_checked": len(dataset),
            "failure_count_added": (
                len(failures) - split_failures_before
            ),
            "run_files": {
                run_id: str(path)
                for run_id, path in source_paths.items()
            },
        }

    if target_representations.get("other", 0) != 0:
        failures.append(
            "window_target has unsupported representation"
        )
    active_target_forms = [
        key
        for key in ("source_index", "epoch_id")
        if target_representations.get(key, 0) > 0
    ]
    if len(active_target_forms) != 1:
        failures.append(
            "window_target representation is not unique: "
            f"{dict(target_representations)}"
        )

    sample_repeat_exact = False
    batch_repeat_exact = False
    batch_summary: dict[str, Any] = {}

    if first_train_dataset is not None:
        first = first_train_dataset[0]
        repeated = first_train_dataset[0]
        sample_repeat_exact = exact_equal(
            first,
            repeated,
        )
        if not sample_repeat_exact:
            failures.append(
                "contract loader sample 0 is not deterministic"
            )

        loader_a = DataLoader(
            first_train_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
        loader_b = DataLoader(
            first_train_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
        batch_a = next(iter(loader_a))
        batch_b = next(iter(loader_b))
        batch_repeat_exact = exact_equal(
            batch_a,
            batch_b,
        )
        if not batch_repeat_exact:
            failures.append(
                "first contract-loader batch is not deterministic"
            )

        expected_batch_shapes = {
            "x": (4, 16, 58, window),
            "physical_port_mask": (4, 16, 10),
            "edge_index": (4, 2, 48),
            "y_source": (4, 16),
            "y_transit": (4, 16),
            "y_victim": (4, 16),
            "y_attack_path": (4, 16),
            "role_mask": (4, 16),
        }

        for key, expected_shape in expected_batch_shapes.items():
            tensor = batch_a.get(key)
            if not isinstance(tensor, torch.Tensor):
                failures.append(
                    f"batch key {key} is not a tensor"
                )
                continue
            actual_shape = tuple(tensor.shape)
            if actual_shape != expected_shape:
                failures.append(
                    f"batch {key} shape={actual_shape}, "
                    f"expected={expected_shape}"
                )
            batch_summary[key] = {
                "shape": list(actual_shape),
                "dtype": str(tensor.dtype),
            }

    report = {
        "stage": "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE",
        "status": "FAIL" if failures else "PASS",
        "dataset_root": str(root),
        "window": window,
        "stride": stride,
        "feature_variant": "PRIMARY58",
        "feature_count": 58,
        "source_layout": "[T,16,81]",
        "wrapped_layout": f"[16,58,{window}]",
        "normalization_applied_by_wrapper": False,
        "raw_physical_mask_shape": [16, 10],
        "edge_index_shape": [2, 48],
        "split_reports": split_reports,
        "total_samples_checked": total_samples_checked,
        "target_representations": dict(
            target_representations
        ),
        "sample_0_repeat_exact": sample_repeat_exact,
        "first_batch_repeat_exact": batch_repeat_exact,
        "batch_summary": batch_summary,
        "failures": failures,
        "warnings": warnings,
        "provenance": {
            **provenance,
            "supplied_loader_sha256": current_loader_sha,
            "wrapper_sha256": sha256_file(wrapper_path),
            "topology_sha256": sha256_file(topology_path),
            "raw_mask_sha256": sha256_file(mask_path),
        },
        "audit_boundary": {
            "all_splits_read_for_loader_equality": True,
            "labels_read_only_for_final_target_equality": True,
            "label_distributions_computed": False,
            "model_training_performed": False,
            "model_inference_performed": False,
            "threshold_selection_performed": False,
            "checkpoint_selection_performed": False,
            "validation_performance_evaluated": False,
            "test_performance_evaluated": False,
        },
        "next_stage": (
            "V5_P0_A4_TINY_SUBSET_OVERFIT"
            if not failures
            else "HOLD_FIX_A3_LOADER"
        ),
    }

    report_path = (
        output_dir
        / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE.json"
    )
    write_json(report_path, report)

    lock = {
        "status": (
            "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_PASS"
            if not failures
            else "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_HOLD"
        ),
        "report_sha256": sha256_file(report_path),
        "script_sha256": sha256_file(Path(__file__)),
        "wrapper_sha256": sha256_file(wrapper_path),
        "supplied_loader_sha256": current_loader_sha,
        "feature_variant": "PRIMARY58",
        "feature_count": 58,
        "window": window,
        "stride": stride,
        "normalization_applied_by_wrapper": False,
        "test_performance_evaluated": False,
        "next_stage": report["next_stage"],
    }
    write_json(
        output_dir
        / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_LOCK.json",
        lock,
    )

    print("===== V5 P0-A3 LOADER CONTRACT AND SMOKE =====")
    print(f"feature_variant: PRIMARY58")
    print(f"feature_count: 58")
    print(f"window: {window}")
    print(f"stride: {stride}")
    for split in ("train", "validation", "test"):
        summary = split_reports.get(split, {})
        print(
            f"{split}_windows: "
            f"{summary.get('dataset_length')}"
        )
        print(
            f"{split}_runs: "
            f"{summary.get('source_run_count')}"
        )
    print(
        "total_samples_checked: "
        f"{total_samples_checked}"
    )
    print(
        "target_representations: "
        f"{dict(target_representations)}"
    )
    print(
        "sample_0_repeat_exact: "
        f"{sample_repeat_exact}"
    )
    print(
        "first_batch_repeat_exact: "
        f"{batch_repeat_exact}"
    )
    print("normalization_applied_by_wrapper: false")
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
            / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_HOLD",
            "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_HOLD\n",
        )
        print("next_stage: HOLD_FIX_A3_LOADER")
        print(
            "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_HOLD"
        )
        return 1

    atomic_write(
        output_dir
        / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_PASS",
        "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_PASS\n",
    )
    print("next_stage: V5_P0_A4_TINY_SUBSET_OVERFIT")
    print(
        "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
