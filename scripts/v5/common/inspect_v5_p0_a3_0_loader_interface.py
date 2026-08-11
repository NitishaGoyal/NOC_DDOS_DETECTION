#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def summarize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": (
                bool(torch.isfinite(value).all())
                if value.is_floating_point()
                else None
            ),
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": list(value.keys()),
            "items": {str(k): summarize(v) for k, v in value.items()},
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "items": [summarize(v) for v in value],
        }
    return {
        "type": type(value).__name__,
        "repr": repr(value),
    }


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            values_equal(a[k], b[k]) for k in a
        )
    if isinstance(a, (list, tuple)) and isinstance(b, type(a)):
        return len(a) == len(b) and all(
            values_equal(x, y) for x, y in zip(a, b)
        )
    return a == b


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    loader_path = root / "dataset_loader.py"

    if not root.is_dir():
        print(f"STOP: dataset root missing: {root}", file=sys.stderr)
        return 2
    if not loader_path.is_file():
        print(f"STOP: loader missing: {loader_path}", file=sys.stderr)
        return 2
    if out.exists():
        print(f"STOP: output directory already exists: {out}", file=sys.stderr)
        return 2

    out.mkdir(parents=True)

    report: dict[str, Any] = {
        "stage": "V5_P0_A3_0_LOADER_INTERFACE_INSPECTION",
        "status": "PASS",
        "dataset_root": str(root),
        "loader_path": str(loader_path),
        "loader_sha256": sha256_file(loader_path),
        "model_training_performed": False,
        "model_inference_performed": False,
        "validation_tensors_read": False,
        "test_tensors_read": False,
        "test_performance_evaluated": False,
    }

    spec = importlib.util.spec_from_file_location(
        "v5_p0_supplied_loader",
        loader_path,
    )
    if spec is None or spec.loader is None:
        raise SystemExit("STOP: could not create import specification")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    functions = []
    for name, value in inspect.getmembers(module, inspect.isfunction):
        if value.__module__ != module.__name__:
            continue
        functions.append({
            "name": name,
            "signature": str(inspect.signature(value)),
        })

    classes = []
    dataset_candidates = []
    for name, value in inspect.getmembers(module, inspect.isclass):
        if value.__module__ != module.__name__:
            continue
        entry = {
            "name": name,
            "signature": str(inspect.signature(value)),
            "bases": [
                f"{base.__module__}.{base.__name__}"
                for base in value.__bases__
            ],
        }
        try:
            is_dataset = issubclass(value, torch.utils.data.Dataset)
        except TypeError:
            is_dataset = False
        entry["dataset_subclass"] = is_dataset
        classes.append(entry)
        if is_dataset:
            dataset_candidates.append((name, value))

    report["module_functions"] = functions
    report["module_classes"] = classes
    report["dataset_candidates"] = [name for name, _ in dataset_candidates]

    construction = {
        "attempted": False,
        "success": False,
    }

    if len(dataset_candidates) == 1:
        dataset_name, dataset_class = dataset_candidates[0]
        signature = inspect.signature(dataset_class)
        aliases = {
            "root": root,
            "dataset_root": root,
            "data_root": root,
            "data_dir": root,
            "dataset_dir": root,
            "split": "train",
            "window": 32,
            "window_size": 32,
            "window_length": 32,
            "sequence_length": 32,
            "stride": 8,
            "window_stride": 8,
            "target_mode": "final",
            "target": "final",
        }

        kwargs = {}
        unresolved = []
        for parameter_name, parameter in signature.parameters.items():
            if parameter_name in aliases:
                kwargs[parameter_name] = aliases[parameter_name]
            elif (
                parameter.default is inspect.Parameter.empty
                and parameter.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ):
                unresolved.append(parameter_name)

        construction.update({
            "attempted": True,
            "dataset_class": dataset_name,
            "dataset_signature": str(signature),
            "kwargs": {k: str(v) for k, v in kwargs.items()},
            "unresolved_required_parameters": unresolved,
        })

        if not unresolved:
            try:
                dataset = dataset_class(**kwargs)
            except Exception as exc:
                construction["error"] = repr(exc)
            else:
                construction["success"] = True
                construction["dataset_length"] = len(dataset)
                construction["instance_attribute_names"] = sorted(
                    vars(dataset).keys()
                )
                if len(dataset) > 0:
                    sample0 = dataset[0]
                    sample0_repeat = dataset[0]
                    construction["sample0"] = summarize(sample0)
                    construction["sample0_repeat_exact"] = values_equal(
                        sample0,
                        sample0_repeat,
                    )

    report["construction"] = construction

    source_path = out / "dataset_loader_source.py"
    source_path.write_text(
        loader_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    report["copied_source_sha256"] = sha256_file(source_path)

    report_path = out / "V5_P0_A3_0_LOADER_INTERFACE_INSPECTION.json"
    atomic_write(
        report_path,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )

    marker = out / "V5_P0_A3_0_LOADER_INTERFACE_INSPECTION_COMPLETE"
    atomic_write(
        marker,
        "V5_P0_A3_0_LOADER_INTERFACE_INSPECTION_COMPLETE\n",
    )

    print("===== V5 P0-A3-0 LOADER INTERFACE INSPECTION =====")
    print("loader_sha256:", report["loader_sha256"])
    print("dataset_candidates:", report["dataset_candidates"])
    print("construction_attempted:", construction["attempted"])
    print("construction_success:", construction["success"])
    print(
        "dataset_class:",
        construction.get("dataset_class"),
    )
    print(
        "dataset_signature:",
        construction.get("dataset_signature"),
    )
    print(
        "unresolved_required_parameters:",
        construction.get("unresolved_required_parameters"),
    )
    print(
        "dataset_length:",
        construction.get("dataset_length"),
    )

    sample = construction.get("sample0")
    if isinstance(sample, dict):
        print("sample0_type:", sample.get("type"))
        print("sample0_keys:", sample.get("keys"))

    print(
        "sample0_repeat_exact:",
        construction.get("sample0_repeat_exact"),
    )
    print("model_training_performed: false")
    print("validation_tensors_read: false")
    print("test_tensors_read: false")
    print("test_performance_evaluated: false")
    print("V5_P0_A3_0_LOADER_INTERFACE_INSPECTION_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
