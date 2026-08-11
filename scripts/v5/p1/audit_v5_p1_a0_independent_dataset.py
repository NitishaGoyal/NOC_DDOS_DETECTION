#!/usr/bin/env python3
"""
V5 P1-A0 Independent Dataset Audit

This is the first P1 stage. It performs no training.

Security boundary
-----------------
- TRAIN and VALIDATION .pt payloads are loaded and audited.
- TEST file metadata is enumerated to create a lock manifest.
- TEST .pt payloads are never opened with torch.load.
- No test labels, tensors, distributions, or performance are inspected.

The audit:
- verifies the frozen P0-B8 architecture contract;
- discovers explicit train/validation/test directories;
- inventories files and schemas;
- audits train/validation tensor shapes, dtypes, labels, topology, non-finite
  values, and PRIMARY58 compatibility;
- checks train/validation content duplicates and identifier overlap;
- creates a metadata-only test lock manifest;
- records suspicious provenance/metadata keys for quarantine;
- emits PASS or HOLD without touching test tensor payloads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


STAGE = "V5_P1_A0_INDEPENDENT_DATASET_AUDIT"
PRIMARY58 = tuple(list(range(0, 30)) + list(range(31, 56)) + [62, 64, 65])

LABEL_ALIASES = {
    "attack": ("y_attack", "y_graph", "attack_label", "graph_label"),
    "attacker_count": (
        "y_attacker_count",
        "attacker_count",
        "y_count",
    ),
    "source": ("y_source", "source_label", "source_labels"),
    "transit": ("y_transit", "transit_label", "transit_labels"),
    "victim": ("y_victim", "victim_label", "victim_labels"),
    "path": (
        "y_attack_path",
        "y_path",
        "attack_path_label",
        "path_labels",
    ),
}

SUSPICIOUS_KEY_PATTERN = re.compile(
    r"(run|pair|file|name|mode|seed|source_id|victim_id|scenario|"
    r"position|index|epoch|split|profile|attack_kind|strength|"
    r"serialization|hash|coordinate|router_id)",
    re.IGNORECASE,
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_string(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return str(value.item())
    return None


def find_alias(mapping: dict[str, Any], names: tuple[str, ...]):
    for name in names:
        if name in mapping:
            return name, mapping[name]
    return None, None


def tensor_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {
            "type": type(value).__name__,
            "shape": None,
            "dtype": None,
        }
    return {
        "type": "Tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def expected_mesh_edges(rows: int = 4, cols: int = 4) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            if row > 0:
                edges.add((node, node - cols))
            if col < cols - 1:
                edges.add((node, node + 1))
            if row < rows - 1:
                edges.add((node, node + cols))
            if col > 0:
                edges.add((node, node - 1))
    return edges


def edge_index_is_mesh(edge_index: Any) -> bool:
    if not isinstance(edge_index, torch.Tensor):
        return False
    if tuple(edge_index.shape) != (2, 48):
        return False
    edge_index = edge_index.detach().cpu().long()
    actual = {
        (int(source), int(destination))
        for source, destination in zip(
            edge_index[0].tolist(),
            edge_index[1].tolist(),
        )
    }
    return actual == expected_mesh_edges()


def discover_split_dir(
    root: Path,
    explicit: Path | None,
    candidates: tuple[str, ...],
) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()

    for candidate in candidates:
        direct = root / candidate
        if direct.is_dir():
            return direct.resolve()

    lowered = {candidate.lower() for candidate in candidates}
    matches = [
        child.resolve()
        for child in root.iterdir()
        if child.is_dir() and child.name.lower() in lowered
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def pt_files(directory: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in directory.rglob("*.pt")
        if path.is_file()
    )


def metadata_manifest_row(
    split: str,
    root: Path,
    path: Path,
    payload_sha256: str | None,
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "split": split,
        "relative_path": str(path.relative_to(root)),
        "basename": path.name,
        "stem": path.stem,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "payload_sha256": payload_sha256 or "",
        "tensor_payload_loaded": split != "test",
    }


class StreamingMoments:
    def __init__(self, feature_count: int):
        self.feature_count = feature_count
        self.count = 0
        self.sum = torch.zeros(feature_count, dtype=torch.float64)
        self.sumsq = torch.zeros(feature_count, dtype=torch.float64)
        self.minimum = torch.full(
            (feature_count,),
            float("inf"),
            dtype=torch.float64,
        )
        self.maximum = torch.full(
            (feature_count,),
            float("-inf"),
            dtype=torch.float64,
        )
        self.nonfinite = torch.zeros(feature_count, dtype=torch.long)

    def update(self, x: torch.Tensor) -> None:
        flat = x.detach().cpu().to(torch.float64).reshape(-1, x.shape[-1])
        finite = torch.isfinite(flat)
        self.nonfinite += (~finite).sum(dim=0)

        safe = torch.where(finite, flat, torch.zeros_like(flat))
        self.sum += safe.sum(dim=0)
        self.sumsq += (safe * safe).sum(dim=0)

        finite_counts = finite.sum(dim=0)
        self.count += int(flat.shape[0])

        positive_inf = torch.full_like(flat, float("inf"))
        negative_inf = torch.full_like(flat, float("-inf"))
        self.minimum = torch.minimum(
            self.minimum,
            torch.where(finite, flat, positive_inf).min(dim=0).values,
        )
        self.maximum = torch.maximum(
            self.maximum,
            torch.where(finite, flat, negative_inf).max(dim=0).values,
        )

    def report(self) -> dict[str, Any]:
        valid_count = (
            torch.full(
                (self.feature_count,),
                self.count,
                dtype=torch.long,
            )
            - self.nonfinite
        ).clamp_min(1)

        valid_count_f = valid_count.to(torch.float64)
        mean = self.sum / valid_count_f
        variance = (
            self.sumsq / valid_count_f - mean * mean
        ).clamp_min(0)
        std = torch.sqrt(variance)

        return {
            "scalar_observations_per_feature": self.count,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "nonfinite_count": self.nonfinite.tolist(),
            "total_nonfinite": int(self.nonfinite.sum().item()),
            "median_abs_mean": float(mean.abs().median().item()),
            "median_std": float(std.median().item()),
        }


def audit_payload_file(
    split: str,
    path: Path,
    expected_feature_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(
            f"{path}: expected dict payload, got {type(obj).__name__}"
        )

    keys = sorted(str(key) for key in obj)
    x = obj.get("x")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{path}: missing Tensor key 'x'")
    if x.ndim != 3:
        raise ValueError(
            f"{path}: x must be [T,16,F], got {tuple(x.shape)}"
        )

    time_length, router_count, feature_count = map(int, x.shape)

    labels: dict[str, dict[str, Any]] = {}
    label_values: dict[str, torch.Tensor] = {}
    for logical_name, aliases in LABEL_ALIASES.items():
        source_key, value = find_alias(obj, aliases)
        labels[logical_name] = {
            "source_key": source_key,
            **tensor_summary(value),
        }
        if isinstance(value, torch.Tensor):
            label_values[logical_name] = value

    edge_index = obj.get("edge_index")
    edge_summary = tensor_summary(edge_index)
    edge_summary["is_bidirectional_4x4_mesh"] = edge_index_is_mesh(edge_index)

    run_id = scalar_string(obj.get("run_id"))
    pair_id = scalar_string(obj.get("pair_id"))

    attack_tensor = label_values.get("attack")
    source_tensor = label_values.get("source")
    if attack_tensor is not None:
        run_is_attack = bool(attack_tensor.bool().any().item())
    elif source_tensor is not None:
        run_is_attack = bool(source_tensor.bool().any().item())
    else:
        run_is_attack = None

    suspicious_keys = [
        key for key in keys
        if SUSPICIOUS_KEY_PATTERN.search(key)
    ]

    schema = {
        "split": split,
        "path": str(path),
        "keys": keys,
        "x": tensor_summary(x),
        "time_length": time_length,
        "router_count": router_count,
        "feature_count": feature_count,
        "labels": labels,
        "edge_index": edge_summary,
        "run_id": run_id,
        "pair_id": pair_id,
        "run_is_attack": run_is_attack,
        "suspicious_metadata_keys": suspicious_keys,
    }

    runtime = {
        "payload": obj,
        "x": x,
        "time_length": time_length,
        "router_count": router_count,
        "feature_count": feature_count,
        "labels": label_values,
        "edge_index": edge_index,
        "run_id": run_id,
        "pair_id": pair_id,
        "run_is_attack": run_is_attack,
        "keys": keys,
        "suspicious_keys": suspicious_keys,
        "expected_feature_count_match": (
            feature_count == expected_feature_count
        ),
    }
    return schema, runtime


def validate_runtime(
    path: Path,
    runtime: dict[str, Any],
    failures: list[str],
    warnings: list[str],
) -> None:
    t = runtime["time_length"]
    n = runtime["router_count"]
    f = runtime["feature_count"]

    if t < 32:
        failures.append(f"{path}: time length {t} is smaller than window 32")
    if n != 16:
        failures.append(f"{path}: router count {n}, expected 16")
    if f != 81:
        failures.append(f"{path}: feature count {f}, expected stored ALL81")
    if max(PRIMARY58) >= f:
        failures.append(
            f"{path}: PRIMARY58 indices are incompatible with F={f}"
        )

    x = runtime["x"]
    if not x.is_floating_point():
        failures.append(f"{path}: x dtype {x.dtype} is not floating point")
    if not torch.isfinite(x).all():
        failures.append(f"{path}: x contains NaN or Inf")

    edge_index = runtime["edge_index"]
    if not edge_index_is_mesh(edge_index):
        failures.append(
            f"{path}: edge_index is not the audited bidirectional 4x4 mesh"
        )

    labels = runtime["labels"]
    for logical_name in (
        "attack",
        "attacker_count",
        "source",
        "transit",
        "victim",
        "path",
    ):
        if logical_name not in labels:
            failures.append(f"{path}: missing label {logical_name}")

    for logical_name in ("attack", "attacker_count"):
        value = labels.get(logical_name)
        if value is None:
            continue
        if value.ndim != 1 or int(value.shape[0]) != t:
            failures.append(
                f"{path}: {logical_name} shape {tuple(value.shape)}, "
                f"expected [{t}]"
            )

    for logical_name in ("source", "transit", "victim", "path"):
        value = labels.get(logical_name)
        if value is None:
            continue
        if value.ndim != 2 or tuple(value.shape) != (t, 16):
            failures.append(
                f"{path}: {logical_name} shape {tuple(value.shape)}, "
                f"expected [{t},16]"
            )

    attack = labels.get("attack")
    source = labels.get("source")
    count = labels.get("attacker_count")

    if attack is not None:
        unique = set(torch.unique(attack.detach().cpu()).tolist())
        if not unique.issubset({0, 1, False, True}):
            failures.append(
                f"{path}: attack labels are not binary: {sorted(unique)}"
            )

    for role in ("source", "transit", "victim", "path"):
        value = labels.get(role)
        if value is None:
            continue
        unique = set(torch.unique(value.detach().cpu()).tolist())
        if not unique.issubset({0, 1, False, True}):
            failures.append(
                f"{path}: {role} labels are not binary: {sorted(unique)}"
            )

    if count is not None:
        unique = set(torch.unique(count.detach().cpu()).tolist())
        if not unique.issubset({0, 1, 2}):
            warnings.append(
                f"{path}: attacker_count classes are {sorted(unique)}, "
                "not limited to {0,1,2}"
            )

    if attack is not None and source is not None:
        derived_attack = source.bool().any(dim=1)
        mismatch = int(
            (derived_attack != attack.bool()).sum().item()
        )
        if mismatch:
            warnings.append(
                f"{path}: {mismatch} windows where y_attack differs from "
                "any(y_source)"
            )

    if count is not None and source is not None:
        derived_count = source.bool().sum(dim=1).long()
        mismatch = int(
            (derived_count != count.long()).sum().item()
        )
        if mismatch:
            warnings.append(
                f"{path}: {mismatch} windows where attacker_count differs "
                "from source-mask cardinality"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--b8-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    b8_dir = args.b8_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(
            f"STOP: output directory already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    if not root.is_dir():
        failures.append(f"P1 root does not exist: {root}")

    b8_report_path = (
        b8_dir
        / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE.json"
    )
    b8_lock_path = (
        b8_dir
        / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE_LOCK.json"
    )
    b8_contract_path = (
        b8_dir
        / "V5_P0_B8_B3_CONV1D_ONLY_ARCHITECTURE_CONTRACT.json"
    )
    b8_marker_path = (
        b8_dir
        / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE_COMPLETE"
    )

    for path in (
        b8_report_path,
        b8_lock_path,
        b8_contract_path,
        b8_marker_path,
    ):
        if not path.is_file():
            failures.append(f"missing B8 prerequisite: {path}")

    b8_report = {}
    b8_lock = {}
    b8_contract = {}
    if not failures:
        b8_report = load_json(b8_report_path)
        b8_lock = load_json(b8_lock_path)
        b8_contract = load_json(b8_contract_path)

        if b8_report.get("status") != "COMPLETE":
            failures.append("B8 report status is not COMPLETE")
        if b8_report.get("decision") != "FREEZE_B3_CONV1D_ONLY":
            failures.append("B8 decision does not freeze B3")
        if b8_lock.get("report_sha256") != sha256_file(b8_report_path):
            failures.append("B8 report SHA does not match B8 lock")
        if (
            b8_lock.get("contract_file_sha256")
            != sha256_file(b8_contract_path)
        ):
            failures.append("B8 contract file SHA does not match B8 lock")
        if (
            b8_lock.get("architecture_contract_sha256")
            != b8_contract.get("architecture_contract_sha256")
        ):
            failures.append("B8 embedded architecture contract hash mismatch")
        if b8_report.get("test_split_accessed") is not False:
            failures.append("B8 does not certify untouched P0 test")

    train_dir = None
    validation_dir = None
    test_dir = None
    if root.is_dir():
        train_dir = discover_split_dir(
            root,
            args.train_dir,
            ("train", "training"),
        )
        validation_dir = discover_split_dir(
            root,
            args.validation_dir,
            ("validation", "val", "valid"),
        )
        test_dir = discover_split_dir(
            root,
            args.test_dir,
            ("test", "testing"),
        )

    for name, directory in (
        ("train", train_dir),
        ("validation", validation_dir),
        ("test", test_dir),
    ):
        if directory is None:
            failures.append(
                f"could not discover explicit {name} directory under {root}; "
                f"pass --{name if name != 'validation' else 'validation'}-dir"
            )
        elif not directory.is_dir():
            failures.append(f"{name} directory does not exist: {directory}")

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "root": str(root),
            "failures": failures,
            "warnings": warnings,
            "training_performed": False,
            "test_file_metadata_enumerated": False,
            "test_tensor_payloads_loaded": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    assert train_dir is not None
    assert validation_dir is not None
    assert test_dir is not None

    train_files = pt_files(train_dir)
    validation_files = pt_files(validation_dir)
    test_files = pt_files(test_dir)

    for split, files in (
        ("train", train_files),
        ("validation", validation_files),
        ("test", test_files),
    ):
        if not files:
            failures.append(f"no .pt files found in {split} directory")

    manifest_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    schema_reports: list[dict[str, Any]] = []

    hashes_by_split: dict[str, dict[str, list[str]]] = {
        "train": defaultdict(list),
        "validation": defaultdict(list),
    }
    identifiers: dict[str, dict[str, set[str]]] = {
        "train": {"run_id": set(), "pair_id": set()},
        "validation": {"run_id": set(), "pair_id": set()},
    }

    attack_run_counts = {
        "train": Counter(),
        "validation": Counter(),
    }
    key_counts = {
        "train": Counter(),
        "validation": Counter(),
    }
    suspicious_key_counts = {
        "train": Counter(),
        "validation": Counter(),
    }

    moments = {
        "train": StreamingMoments(81),
        "validation": StreamingMoments(81),
    }

    for split, files in (
        ("train", train_files),
        ("validation", validation_files),
    ):
        for index, path in enumerate(files, start=1):
            payload_hash = sha256_file(path)
            hashes_by_split[split][payload_hash].append(str(path))
            manifest_rows.append(
                metadata_manifest_row(
                    split,
                    root,
                    path,
                    payload_hash,
                )
            )

            try:
                schema, runtime = audit_payload_file(
                    split,
                    path,
                    expected_feature_count=81,
                )
                validate_runtime(
                    path,
                    runtime,
                    failures,
                    warnings,
                )
            except Exception as exc:
                failures.append(f"{path}: payload audit failed: {exc}")
                continue

            schema_reports.append(schema)
            schema_rows.append(
                {
                    "split": split,
                    "relative_path": str(path.relative_to(root)),
                    "time_length": runtime["time_length"],
                    "router_count": runtime["router_count"],
                    "feature_count": runtime["feature_count"],
                    "x_dtype": str(runtime["x"].dtype),
                    "edge_index_is_mesh": edge_index_is_mesh(
                        runtime["edge_index"]
                    ),
                    "run_id_present": runtime["run_id"] is not None,
                    "pair_id_present": runtime["pair_id"] is not None,
                    "run_is_attack": runtime["run_is_attack"],
                    "key_count": len(runtime["keys"]),
                    "suspicious_key_count": len(
                        runtime["suspicious_keys"]
                    ),
                }
            )

            moments[split].update(runtime["x"])

            for key in runtime["keys"]:
                key_counts[split][key] += 1
            for key in runtime["suspicious_keys"]:
                suspicious_key_counts[split][key] += 1

            for identifier_name in ("run_id", "pair_id"):
                identifier = runtime[identifier_name]
                if identifier is not None:
                    identifiers[split][identifier_name].add(identifier)

            run_is_attack = runtime["run_is_attack"]
            attack_run_counts[split][
                "unknown" if run_is_attack is None
                else ("attack" if run_is_attack else "control")
            ] += 1

            if index == 1 or index % 25 == 0 or index == len(files):
                print(
                    f"audited {split}: {index}/{len(files)} files"
                )

    # TEST: metadata only. No hashing and no torch.load.
    for path in test_files:
        manifest_rows.append(
            metadata_manifest_row(
                "test",
                root,
                path,
                payload_sha256=None,
            )
        )

    for split in ("train", "validation"):
        duplicate_groups = [
            paths
            for paths in hashes_by_split[split].values()
            if len(paths) > 1
        ]
        if duplicate_groups:
            failures.append(
                f"{split} contains {len(duplicate_groups)} exact "
                "content-duplicate groups"
            )

    cross_hashes = (
        set(hashes_by_split["train"])
        & set(hashes_by_split["validation"])
    )
    if cross_hashes:
        failures.append(
            f"train/validation contain {len(cross_hashes)} exact "
            "content-overlap groups"
        )

    overlap_report = {}
    for identifier_name in ("run_id", "pair_id"):
        overlap = sorted(
            identifiers["train"][identifier_name]
            & identifiers["validation"][identifier_name]
        )
        overlap_report[identifier_name] = overlap
        if overlap:
            failures.append(
                f"train/validation share {len(overlap)} "
                f"{identifier_name} values"
            )

    train_stems = {path.stem for path in train_files}
    validation_stems = {path.stem for path in validation_files}
    test_stems = {path.stem for path in test_files}

    stem_overlap = {
        "train_validation": sorted(train_stems & validation_stems),
        "train_test": sorted(train_stems & test_stems),
        "validation_test": sorted(validation_stems & test_stems),
    }
    if stem_overlap["train_validation"]:
        warnings.append(
            f"train/validation share "
            f"{len(stem_overlap['train_validation'])} filename stems"
        )
    if stem_overlap["train_test"] or stem_overlap["validation_test"]:
        warnings.append(
            "test shares filename stems with train/validation; test payloads "
            "were not opened, so content overlap was not checked"
        )

    normalization = {
        split: moments[split].report()
        for split in ("train", "validation")
    }

    for split in ("train", "validation"):
        if normalization[split]["total_nonfinite"] != 0:
            failures.append(
                f"{split} contains non-finite feature values"
            )

        median_abs_mean = normalization[split]["median_abs_mean"]
        median_std = normalization[split]["median_std"]
        if median_abs_mean > 0.5:
            warnings.append(
                f"{split} median absolute feature mean is "
                f"{median_abs_mean:.4f}; verify normalization provenance"
            )
        if not (0.25 <= median_std <= 2.5):
            warnings.append(
                f"{split} median feature std is {median_std:.4f}; "
                "verify normalization provenance"
            )

    key_inventory = {
        split: {
            "key_counts": dict(sorted(key_counts[split].items())),
            "suspicious_metadata_key_counts": dict(
                sorted(suspicious_key_counts[split].items())
            ),
        }
        for split in ("train", "validation")
    }

    split_summary = {
        "train": {
            "directory": str(train_dir),
            "file_count": len(train_files),
            "attack_run_counts": dict(attack_run_counts["train"]),
        },
        "validation": {
            "directory": str(validation_dir),
            "file_count": len(validation_files),
            "attack_run_counts": dict(
                attack_run_counts["validation"]
            ),
        },
        "test": {
            "directory": str(test_dir),
            "file_count": len(test_files),
            "payloads_loaded": False,
            "payload_hashes_computed": False,
        },
    }

    manifest_path = (
        output_dir / "V5_P1_A0_FILE_MANIFEST.csv"
    )
    schema_csv_path = (
        output_dir / "V5_P1_A0_TRAIN_VALIDATION_SCHEMA.csv"
    )
    schema_json_path = (
        output_dir / "V5_P1_A0_TRAIN_VALIDATION_SCHEMA.json"
    )
    normalization_path = (
        output_dir / "V5_P1_A0_NORMALIZATION_STATISTICS.json"
    )
    quarantine_path = (
        output_dir / "V5_P1_A0_METADATA_QUARANTINE_INVENTORY.json"
    )
    test_lock_path = (
        output_dir / "V5_P1_A0_TEST_METADATA_LOCK.json"
    )

    write_csv(manifest_path, manifest_rows)
    write_csv(schema_csv_path, schema_rows)
    write_json(schema_json_path, schema_reports)
    write_json(normalization_path, normalization)
    write_json(quarantine_path, key_inventory)

    test_metadata = [
        {
            "relative_path": row["relative_path"],
            "size_bytes": row["size_bytes"],
            "mtime_ns": row["mtime_ns"],
        }
        for row in manifest_rows
        if row["split"] == "test"
    ]
    test_lock = {
        "stage": STAGE,
        "root": str(root),
        "test_directory": str(test_dir),
        "test_file_count": len(test_files),
        "test_files": test_metadata,
        "test_metadata_manifest_sha256": canonical_sha256(
            test_metadata
        ),
        "test_file_metadata_enumerated": True,
        "test_payload_hashes_computed": False,
        "test_tensor_payloads_loaded": False,
        "test_labels_inspected": False,
        "test_distributions_inspected": False,
        "test_performance_evaluated": False,
    }
    write_json(test_lock_path, test_lock)

    status = "PASS" if not failures else "HOLD"
    next_stage = (
        "V5_P1_A1_FEATURE_MASK_AND_LOADER_CONTRACT"
        if status == "PASS"
        else STAGE
    )

    report = {
        "stage": STAGE,
        "status": status,
        "root": str(root),
        "frozen_architecture_contract": {
            "decision": "FREEZE_B3_CONV1D_ONLY",
            "architecture_contract_sha256": (
                b8_contract.get("architecture_contract_sha256")
            ),
            "b8_report_sha256": sha256_file(b8_report_path),
            "b8_lock_sha256": sha256_file(b8_lock_path),
            "b8_contract_file_sha256": sha256_file(
                b8_contract_path
            ),
        },
        "split_summary": split_summary,
        "stored_tensor_contract_observed": {
            "expected_x_layout": "[T,16,81]",
            "primary58_indices": list(PRIMARY58),
            "window_compatibility_required": 32,
            "stride_compatibility_required": 8,
            "topology_required": "bidirectional cardinal 4x4 mesh",
        },
        "train_validation_overlap": {
            "content_hash_overlap_count": len(cross_hashes),
            "identifier_overlap": overlap_report,
            "filename_stem_overlap": stem_overlap,
        },
        "normalization_statistics": normalization,
        "metadata_quarantine_inventory": key_inventory,
        "test_boundary": test_lock,
        "artifacts": {
            "file_manifest_csv": [
                str(manifest_path),
                sha256_file(manifest_path),
            ],
            "schema_csv": [
                str(schema_csv_path),
                sha256_file(schema_csv_path),
            ],
            "schema_json": [
                str(schema_json_path),
                sha256_file(schema_json_path),
            ],
            "normalization_json": [
                str(normalization_path),
                sha256_file(normalization_path),
            ],
            "metadata_quarantine_json": [
                str(quarantine_path),
                sha256_file(quarantine_path),
            ],
            "test_metadata_lock_json": [
                str(test_lock_path),
                sha256_file(test_lock_path),
            ],
        },
        "training_performed": False,
        "test_file_metadata_enumerated": True,
        "test_payload_hashes_computed": False,
        "test_tensor_payloads_loaded": False,
        "test_labels_inspected": False,
        "test_performance_evaluated": False,
        "failures": failures,
        "warnings": warnings,
        "next_stage": next_stage,
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    lock = {
        "status": f"{STAGE}_{status}",
        "report_sha256": sha256_file(report_path),
        "test_metadata_lock_sha256": sha256_file(test_lock_path),
        "test_metadata_manifest_sha256": (
            test_lock["test_metadata_manifest_sha256"]
        ),
        "frozen_architecture_contract_sha256": (
            b8_contract.get("architecture_contract_sha256")
        ),
        "training_performed": False,
        "test_tensor_payloads_loaded": False,
        "next_stage": next_stage,
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)

    marker = f"{STAGE}_{status}"
    atomic_write(output_dir / marker, marker + "\n")

    print("\n===== V5 P1-A0 FINAL =====")
    print("status:", status)
    print("train_files:", len(train_files))
    print("validation_files:", len(validation_files))
    print("test_files:", len(test_files))
    print(
        "train_attack_runs:",
        attack_run_counts["train"].get("attack", 0),
    )
    print(
        "train_control_runs:",
        attack_run_counts["train"].get("control", 0),
    )
    print(
        "validation_attack_runs:",
        attack_run_counts["validation"].get("attack", 0),
    )
    print(
        "validation_control_runs:",
        attack_run_counts["validation"].get("control", 0),
    )
    print("train_validation_content_overlap:", len(cross_hashes))
    print(
        "train_validation_run_id_overlap:",
        len(overlap_report["run_id"]),
    )
    print(
        "train_validation_pair_id_overlap:",
        len(overlap_report["pair_id"]),
    )
    print(
        "train_median_abs_feature_mean:",
        f"{normalization['train']['median_abs_mean']:.6f}",
    )
    print(
        "train_median_feature_std:",
        f"{normalization['train']['median_std']:.6f}",
    )
    print(
        "validation_median_abs_feature_mean:",
        f"{normalization['validation']['median_abs_mean']:.6f}",
    )
    print(
        "validation_median_feature_std:",
        f"{normalization['validation']['median_std']:.6f}",
    )
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("training_performed: false")
    print("test_file_metadata_enumerated: true")
    print("test_payload_hashes_computed: false")
    print("test_tensor_payloads_loaded: false")
    print("test_labels_inspected: false")
    print("test_performance_evaluated: false")
    print("next_stage:", next_stage)
    print(marker)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
