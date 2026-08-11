#!/usr/bin/env python3
"""
V5 P2-A1 Tensor Schema and Non-Test Sample Audit

Loads a deterministic sample of TRAIN and VALIDATION run tensors only.

It verifies:
- A0 completion and report hash;
- support tensor readability (normalization.pt and topology.pt);
- sampled run payload type, keys, tensor shapes, and dtypes;
- feature_count=81 and num_nodes=16 compatibility;
- temporal length >= recommended window 32;
- matched ATTACK/CONTROL pair shape and temporal-length agreement;
- graph/count/source/transit/victim/path target discoverability;
- basic filename-to-label semantics;
- finite feature values.

It never enumerates or opens runs/test and never constructs test windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch


STAGE = "V5_P2_A1_TENSOR_SCHEMA_AND_NONTEST_SAMPLE_AUDIT"
COMPLETE = f"{STAGE}_COMPLETE"

ALIASES = {
    "x": ("x", "features", "node_features", "feature_tensor"),
    "graph": ("y_graph", "y_attack", "graph_label", "attack_label"),
    "count": (
        "y_attacker_count",
        "attacker_count",
        "y_count",
        "count_label",
    ),
    "source": ("y_source", "source", "source_label"),
    "transit": ("y_transit", "transit", "transit_label"),
    "victim": ("y_victim", "victim", "victim_label"),
    "path": (
        "y_attack_path",
        "y_path",
        "attack_path",
        "path_label",
    ),
    "edge_index": ("edge_index", "edges"),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pair_key(path: Path) -> str | None:
    stem = path.stem
    if stem.endswith("_ATTACK"):
        return stem[:-len("_ATTACK")]
    if stem.endswith("_CONTROL"):
        return stem[:-len("_CONTROL")]
    return None


def class_name(path: Path) -> str | None:
    if path.stem.endswith("_ATTACK"):
        return "attack"
    if path.stem.endswith("_CONTROL"):
        return "control"
    return None


def find_key(payload: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in payload:
            return alias
    return None


def tensor_record(value: torch.Tensor) -> dict[str, Any]:
    record: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "requires_grad": bool(value.requires_grad),
    }
    if value.numel() == 0:
        record["empty"] = True
        return record

    if value.is_floating_point():
        record["finite"] = bool(torch.isfinite(value).all().item())
        finite = value[torch.isfinite(value)]
        if finite.numel():
            record["minimum"] = float(finite.min().item())
            record["maximum"] = float(finite.max().item())
            record["mean"] = float(finite.float().mean().item())
    elif value.dtype == torch.bool:
        record["true_count"] = int(value.sum().item())
    else:
        record["minimum"] = int(value.min().item())
        record["maximum"] = int(value.max().item())
        unique = torch.unique(value)
        record["unique_count"] = int(unique.numel())
        if unique.numel() <= 32:
            record["unique_values"] = [
                int(item) for item in unique.tolist()
            ]
    return record


def summarize_object(value: Any, depth: int = 0) -> Any:
    if isinstance(value, torch.Tensor):
        return {"type": "Tensor", **tensor_record(value)}
    if isinstance(value, dict):
        if depth >= 3:
            return {
                "type": "dict",
                "keys": sorted(str(key) for key in value),
            }
        return {
            "type": "dict",
            "items": {
                str(key): summarize_object(nested, depth + 1)
                for key, nested in value.items()
            },
        }
    if isinstance(value, (list, tuple)):
        if depth >= 3 or len(value) > 32:
            return {
                "type": type(value).__name__,
                "length": len(value),
            }
        return {
            "type": type(value).__name__,
            "items": [
                summarize_object(nested, depth + 1)
                for nested in value
            ],
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {
            "type": type(value).__name__,
            "value": value,
        }
    return {
        "type": type(value).__name__,
        "repr": repr(value)[:500],
    }


def infer_x_key(
    payload: dict[str, Any],
    expected_nodes: int,
    expected_features: int,
) -> tuple[str | None, list[str]]:
    direct = find_key(payload, ALIASES["x"])
    if direct is not None:
        return direct, []

    candidates: list[str] = []
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim != 3 or not value.is_floating_point():
            continue
        shape = tuple(value.shape)
        if expected_nodes in shape and expected_features in shape:
            candidates.append(str(key))

    if len(candidates) == 1:
        return candidates[0], candidates
    return None, candidates


def infer_axes(
    shape: tuple[int, ...],
    expected_nodes: int,
    expected_features: int,
) -> dict[str, Any] | None:
    if len(shape) != 3:
        return None

    node_axes = [
        index for index, size in enumerate(shape)
        if size == expected_nodes
    ]
    feature_axes = [
        index for index, size in enumerate(shape)
        if size == expected_features
    ]

    possibilities = []
    for node_axis in node_axes:
        for feature_axis in feature_axes:
            if node_axis == feature_axis:
                continue
            time_axis = ({0, 1, 2} - {node_axis, feature_axis}).pop()
            possibilities.append(
                {
                    "time_axis": time_axis,
                    "node_axis": node_axis,
                    "feature_axis": feature_axis,
                    "time_length": shape[time_axis],
                }
            )

    if len(possibilities) == 1:
        return possibilities[0]
    return None


def target_time_compatible(
    target: torch.Tensor,
    time_length: int,
    node_count: int | None = None,
) -> bool:
    shape = tuple(target.shape)
    if target.ndim == 0:
        return False
    if node_count is None:
        return time_length in shape
    return time_length in shape and node_count in shape


def select_pair_keys(files: list[Path]) -> list[str]:
    keys = sorted(
        {
            key
            for path in files
            if (key := pair_key(path)) is not None
        }
    )
    if not keys:
        return []
    indices = sorted({0, len(keys) // 2, len(keys) - 1})
    return [keys[index] for index in indices]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root_input = args.root.expanduser()
    root = root_input.resolve()
    a0_dir = args.a0_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    paths = {
        "a0_report": (
            a0_dir
            / "V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT.json"
        ),
        "a0_lock": (
            a0_dir
            / "V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT_LOCK.json"
        ),
        "a0_marker": (
            a0_dir
            / "V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT_COMPLETE"
        ),
        "dataset_summary": root / "dataset_summary.json",
        "feature_schema": root / "feature_schema.json",
        "normalization": root / "normalization.pt",
        "topology": root / "topology.pt",
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    if not (root / "runs" / "train").is_dir():
        failures.append("missing runs/train")
    if not (root / "runs" / "validation").is_dir():
        failures.append("missing runs/validation")
    if not (root / "runs" / "test").is_dir():
        failures.append("missing runs/test")

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "train_tensor_contents_accessed": False,
            "validation_tensor_contents_accessed": False,
            "test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        return 1

    a0_report = load_json(paths["a0_report"])
    a0_lock = load_json(paths["a0_lock"])

    if a0_report.get("status") != "COMPLETE":
        failures.append("A0 status is not COMPLETE")
    if (
        a0_lock.get("report_sha256")
        != sha256_file(paths["a0_report"])
    ):
        failures.append("A0 report SHA mismatch")
    if a0_report.get("security_boundary", {}).get(
        "test_tensor_contents_accessed"
    ) is not False:
        failures.append("A0 does not certify untouched test tensors")

    dataset_summary = load_json(paths["dataset_summary"])
    feature_schema = load_json(paths["feature_schema"])

    expected_nodes = dataset_summary.get("num_nodes")
    expected_features = dataset_summary.get("feature_count")
    recommended_window = dataset_summary.get("recommended_window")
    recommended_stride = dataset_summary.get("recommended_stride")

    if expected_nodes != 16:
        failures.append(
            f"num_nodes={expected_nodes!r}, expected 16"
        )
    if expected_features != 81:
        failures.append(
            f"feature_count={expected_features!r}, expected 81"
        )
    if recommended_window != 32:
        failures.append(
            f"recommended_window={recommended_window!r}, expected 32"
        )
    if recommended_stride != 8:
        failures.append(
            f"recommended_stride={recommended_stride!r}, expected 8"
        )

    features = feature_schema.get("features")
    if not isinstance(features, list) or len(features) != 81:
        failures.append(
            "feature_schema does not contain exactly 81 features"
        )

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "train_tensor_contents_accessed": False,
            "validation_tensor_contents_accessed": False,
            "test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    # Support tensors are not run tensors and are safe to read in this stage.
    normalization = torch.load(
        paths["normalization"],
        map_location="cpu",
        weights_only=False,
    )
    topology = torch.load(
        paths["topology"],
        map_location="cpu",
        weights_only=False,
    )

    support_tensor_summary = {
        "normalization": summarize_object(normalization),
        "topology": summarize_object(topology),
    }

    split_samples: dict[str, Any] = {}
    discovered_key_sets: dict[str, list[list[str]]] = {
        "train": [],
        "validation": [],
    }
    discovered_x_orders: list[dict[str, Any]] = []
    matched_pair_checks: list[dict[str, Any]] = []

    train_accessed = False
    validation_accessed = False

    for split in ("train", "validation"):
        split_dir = root / "runs" / split
        files = sorted(split_dir.glob("*.pt"))
        keys = select_pair_keys(files)

        by_name = {path.name: path for path in files}
        sample_paths: list[Path] = []
        for key in keys:
            for suffix in ("ATTACK", "CONTROL"):
                name = f"{key}_{suffix}.pt"
                path = by_name.get(name)
                if path is None:
                    failures.append(
                        f"{split} sample pair member missing: {name}"
                    )
                else:
                    sample_paths.append(path)

        split_records: list[dict[str, Any]] = []
        pair_record_map: dict[str, dict[str, Any]] = {}

        for path in sample_paths:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            if split == "train":
                train_accessed = True
            else:
                validation_accessed = True

            record: dict[str, Any] = {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "class": class_name(path),
                "pair_key": pair_key(path),
                "payload_type": type(payload).__name__,
            }

            if not isinstance(payload, dict):
                failures.append(
                    f"{split}/{path.name} payload is "
                    f"{type(payload).__name__}, expected dict"
                )
                record["payload_summary"] = summarize_object(payload)
                split_records.append(record)
                continue

            key_list = sorted(str(key) for key in payload)
            discovered_key_sets[split].append(key_list)
            record["keys"] = key_list
            record["payload_summary"] = summarize_object(payload)

            resolved_keys: dict[str, str | None] = {}
            x_key, x_candidates = infer_x_key(
                payload,
                expected_nodes=16,
                expected_features=81,
            )
            resolved_keys["x"] = x_key
            record["x_fallback_candidates"] = x_candidates

            for logical in (
                "graph",
                "count",
                "source",
                "transit",
                "victim",
                "path",
                "edge_index",
            ):
                resolved_keys[logical] = find_key(
                    payload,
                    ALIASES[logical],
                )
            record["resolved_keys"] = resolved_keys

            for logical in (
                "x",
                "graph",
                "count",
                "source",
                "transit",
                "victim",
                "path",
            ):
                if resolved_keys[logical] is None:
                    failures.append(
                        f"{split}/{path.name} cannot resolve "
                        f"required logical field {logical}"
                    )

            if x_key is None:
                split_records.append(record)
                continue

            x = payload[x_key]
            if not isinstance(x, torch.Tensor):
                failures.append(
                    f"{split}/{path.name} x is not Tensor"
                )
                split_records.append(record)
                continue
            if x.ndim != 3:
                failures.append(
                    f"{split}/{path.name} x.ndim={x.ndim}, expected 3"
                )
                split_records.append(record)
                continue
            if not x.is_floating_point():
                failures.append(
                    f"{split}/{path.name} x dtype={x.dtype}, "
                    "expected floating point"
                )
            if not torch.isfinite(x).all():
                failures.append(
                    f"{split}/{path.name} x contains NaN or Inf"
                )

            axes = infer_axes(tuple(x.shape), 16, 81)
            record["x_axes"] = axes
            if axes is None:
                failures.append(
                    f"{split}/{path.name} cannot uniquely infer "
                    f"time/node/feature axes from x shape {tuple(x.shape)}"
                )
                split_records.append(record)
                continue

            discovered_x_orders.append(
                {
                    "file": f"{split}/{path.name}",
                    **axes,
                    "shape": list(x.shape),
                }
            )
            time_length = int(axes["time_length"])
            record["time_length"] = time_length
            if time_length < 32:
                failures.append(
                    f"{split}/{path.name} time_length={time_length} < 32"
                )

            for logical in (
                "graph",
                "count",
                "source",
                "transit",
                "victim",
                "path",
            ):
                key = resolved_keys[logical]
                if key is None:
                    continue
                target = payload[key]
                if not isinstance(target, torch.Tensor):
                    failures.append(
                        f"{split}/{path.name} {logical} target "
                        "is not Tensor"
                    )
                    continue
                compatible = target_time_compatible(
                    target,
                    time_length,
                    node_count=(
                        None
                        if logical in ("graph", "count")
                        else 16
                    ),
                )
                if not compatible:
                    failures.append(
                        f"{split}/{path.name} {logical} shape "
                        f"{tuple(target.shape)} is not compatible "
                        f"with time={time_length}"
                    )

            graph_key = resolved_keys["graph"]
            if graph_key is not None and isinstance(
                payload[graph_key],
                torch.Tensor,
            ):
                graph = payload[graph_key]
                positives = int((graph > 0).sum().item())
                record["graph_positive_count"] = positives
                if class_name(path) == "control" and positives != 0:
                    failures.append(
                        f"{split}/{path.name} CONTROL has "
                        f"{positives} positive graph labels"
                    )
                if class_name(path) == "attack" and positives == 0:
                    failures.append(
                        f"{split}/{path.name} ATTACK has no "
                        "positive graph labels"
                    )

            for logical in ("source", "transit", "victim", "path"):
                key = resolved_keys[logical]
                if key is None or not isinstance(
                    payload[key],
                    torch.Tensor,
                ):
                    continue
                positives = int((payload[key] > 0).sum().item())
                record[f"{logical}_positive_count"] = positives
                if class_name(path) == "control" and positives != 0:
                    failures.append(
                        f"{split}/{path.name} CONTROL has "
                        f"{positives} positive {logical} labels"
                    )
                if (
                    class_name(path) == "attack"
                    and logical in ("source", "victim", "path")
                    and positives == 0
                ):
                    failures.append(
                        f"{split}/{path.name} ATTACK has no "
                        f"positive {logical} labels"
                    )

            pair = record["pair_key"]
            if pair is not None:
                pair_record_map.setdefault(pair, {})[
                    record["class"]
                ] = {
                    "x_shape": list(x.shape),
                    "time_length": time_length,
                    "resolved_keys": resolved_keys,
                }

            split_records.append(record)

        for pair, members in sorted(pair_record_map.items()):
            attack = members.get("attack")
            control = members.get("control")
            check = {
                "split": split,
                "pair_key": pair,
                "attack_present": attack is not None,
                "control_present": control is not None,
            }
            if attack is None or control is None:
                failures.append(
                    f"{split}/{pair} sampled pair incomplete"
                )
            else:
                check["x_shape_match"] = (
                    attack["x_shape"] == control["x_shape"]
                )
                check["time_length_match"] = (
                    attack["time_length"]
                    == control["time_length"]
                )
                check["resolved_key_map_match"] = (
                    attack["resolved_keys"]
                    == control["resolved_keys"]
                )
                if not check["x_shape_match"]:
                    failures.append(
                        f"{split}/{pair} ATTACK/CONTROL x shapes differ"
                    )
                if not check["time_length_match"]:
                    failures.append(
                        f"{split}/{pair} ATTACK/CONTROL lengths differ"
                    )
                if not check["resolved_key_map_match"]:
                    failures.append(
                        f"{split}/{pair} ATTACK/CONTROL key maps differ"
                    )
            matched_pair_checks.append(check)

        split_samples[split] = {
            "selected_pair_keys": keys,
            "sample_file_count": len(sample_paths),
            "records": split_records,
        }

    # Schema consistency across all sampled run payloads.
    all_key_sets = (
        discovered_key_sets["train"]
        + discovered_key_sets["validation"]
    )
    unique_key_sets = {
        tuple(key_set) for key_set in all_key_sets
    }
    if len(unique_key_sets) != 1:
        failures.append(
            f"sampled run tensors expose {len(unique_key_sets)} "
            "different top-level key sets"
        )

    axis_signatures = {
        (
            item["time_axis"],
            item["node_axis"],
            item["feature_axis"],
        )
        for item in discovered_x_orders
    }
    if len(axis_signatures) != 1:
        failures.append(
            f"sampled x tensors expose {len(axis_signatures)} "
            "different axis orders"
        )

    report = {
        "stage": STAGE,
        "status": "COMPLETE" if not failures else "HOLD",
        "decision": (
            "AUTHORIZE_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT"
            if not failures
            else "BLOCK_P2_A2"
        ),
        "dataset": {
            "link_path": str(root_input.absolute()),
            "resolved_root": str(root),
            "feature_count": expected_features,
            "num_nodes": expected_nodes,
            "recommended_window": recommended_window,
            "recommended_stride": recommended_stride,
        },
        "a0_provenance": {
            "report_sha256": sha256_file(paths["a0_report"]),
            "lock_sha256": sha256_file(paths["a0_lock"]),
        },
        "support_tensor_summary": support_tensor_summary,
        "sample_policy": {
            "splits": ["train", "validation"],
            "pair_positions_per_split": [
                "lexicographically_first",
                "lexicographically_middle",
                "lexicographically_last",
            ],
            "members_per_pair": ["ATTACK", "CONTROL"],
        },
        "split_samples": split_samples,
        "matched_pair_checks": matched_pair_checks,
        "schema_consistency": {
            "unique_top_level_key_set_count": len(unique_key_sets),
            "unique_top_level_key_sets": [
                list(item) for item in sorted(unique_key_sets)
            ],
            "unique_x_axis_signature_count": len(axis_signatures),
            "x_axis_signatures": [
                {
                    "time_axis": item[0],
                    "node_axis": item[1],
                    "feature_axis": item[2],
                }
                for item in sorted(axis_signatures)
            ],
        },
        "security_boundary": {
            "train_tensor_contents_accessed": train_accessed,
            "validation_tensor_contents_accessed": validation_accessed,
            "test_directory_existence_checked": True,
            "test_directory_enumerated": False,
            "test_tensor_files_opened": False,
            "test_tensor_bytes_read": False,
            "test_tensor_contents_accessed": False,
            "test_dataset_constructed": False,
            "test_windows_constructed": False,
            "test_labels_read": False,
        },
        "failures": failures,
        "warnings": warnings,
        "next_stage": (
            "V5_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT"
            if not failures
            else None
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    if failures:
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures[:50]:
            print("FAIL:", failure)
        print("failure_count:", len(failures))
        print("warning_count:", len(warnings))
        return 1

    lock = {
        "status": COMPLETE,
        "decision": "AUTHORIZE_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT",
        "report_sha256": sha256_file(report_path),
        "resolved_dataset_root": str(root),
        "sampled_train_file_count": (
            split_samples["train"]["sample_file_count"]
        ),
        "sampled_validation_file_count": (
            split_samples["validation"]["sample_file_count"]
        ),
        "feature_count": expected_features,
        "num_nodes": expected_nodes,
        "recommended_window": recommended_window,
        "recommended_stride": recommended_stride,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "next_stage": "V5_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT",
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    signature = next(iter(axis_signatures))
    print("===== V5 P2-A1 TENSOR SCHEMA + NONTEST SAMPLE AUDIT =====")
    print("status: COMPLETE")
    print(
        "sampled_train_files:",
        split_samples["train"]["sample_file_count"],
    )
    print(
        "sampled_validation_files:",
        split_samples["validation"]["sample_file_count"],
    )
    print("feature_count:", expected_features)
    print("num_nodes:", expected_nodes)
    print("recommended_window:", recommended_window)
    print("recommended_stride:", recommended_stride)
    print(
        "x_axis_order:",
        {
            "time_axis": signature[0],
            "node_axis": signature[1],
            "feature_axis": signature[2],
        },
    )
    print("unique_top_level_key_set_count:", len(unique_key_sets))
    print("matched_pair_check_count:", len(matched_pair_checks))
    print("train_tensor_contents_accessed:", train_accessed)
    print(
        "validation_tensor_contents_accessed:",
        validation_accessed,
    )
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")
    print("test_dataset_constructed: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print(
        "next_stage: "
        "V5_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT"
    )
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
