#!/usr/bin/env python3
"""
V5 P2-A0 Independent Dataset Metadata Audit

Metadata-only audit for the linked P2 dataset.

This stage:
- reads JSON/Markdown/text metadata;
- enumerates filenames and file sizes;
- verifies matched ATTACK/CONTROL pairs and split disjointness;
- checks manifest/summary consistency where the schemas expose the data;
- does not torch.load any run tensor;
- does not read any .pt tensor bytes;
- does not construct train/validation/test window datasets;
- does not access test tensor contents.

Next stage:
    V5_P2_A1_TENSOR_SCHEMA_AND_NONTEST_SAMPLE_AUDIT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


STAGE = "V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT"
COMPLETE = f"{STAGE}_COMPLETE"

EXPECTED_COUNTS = {
    "train": {"total": 830, "attack": 415, "control": 415, "pairs": 415},
    "validation": {"total": 148, "attack": 74, "control": 74, "pairs": 74},
    "test": {"total": 138, "attack": 69, "control": 69, "pairs": 69},
}
EXPECTED_TOTAL_RUNS = 1116
EXPECTED_TOTAL_PAIRS = 558


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


def normalize_split_name(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"val", "valid", "validation"}:
        return "validation"
    return lowered


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


def recursively_collect_strings(value: Any) -> list[str]:
    strings: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, dict):
            for key, nested in item.items():
                if isinstance(key, str):
                    strings.append(key)
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return strings


def recursively_find_numeric(
    value: Any,
    wanted_keys: set[str],
) -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []

    def visit(item: Any, prefix: str = "") -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if (
                    str(key).lower() in wanted_keys
                    and isinstance(nested, (int, float))
                    and not isinstance(nested, bool)
                ):
                    found.append((path, float(nested)))
                visit(nested, path)
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                visit(nested, f"{prefix}[{index}]")

    visit(value)
    return found


def support_file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_sha256sums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    pattern = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+)$")
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if not match:
            raise ValueError(
                f"invalid SHA256SUMS line {line_number}: {raw_line!r}"
            )
        digest = match.group(1).lower()
        relative = match.group(2).strip()
        entries[relative] = digest
    return entries


def checksum_lookup(
    entries: dict[str, str],
    root: Path,
    path: Path,
) -> str | None:
    relative = path.relative_to(root).as_posix()
    candidates = {
        relative,
        f"./{relative}",
        path.name,
        f"p2_complete558_dynamic_graph/{relative}",
    }
    for candidate in candidates:
        if candidate in entries:
            return entries[candidate]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root_input = args.root.expanduser()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    if not root_input.exists():
        failures.append(f"dataset root does not exist: {root_input}")
        root = root_input.absolute()
    else:
        root = root_input.resolve()

    required_files = (
        "README.md",
        "SHA256SUMS.txt",
        "campaign_manifest.json",
        "dataset_loader.py",
        "dataset_summary.json",
        "feature_schema.json",
        "normalization.pt",
        "split_manifest.json",
        "tensor_manifest.json",
        "topology.pt",
    )
    required_dirs = (
        "runs/train",
        "runs/validation",
        "runs/test",
    )

    for name in required_files:
        if not (root / name).is_file():
            failures.append(f"missing required file: {name}")
    for name in required_dirs:
        if not (root / name).is_dir():
            failures.append(f"missing required directory: {name}")

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "run_tensor_files_opened": False,
            "run_tensor_bytes_read": False,
            "test_tensor_files_opened": False,
            "test_tensor_bytes_read": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    documents = {
        "campaign_manifest": load_json(root / "campaign_manifest.json"),
        "dataset_summary": load_json(root / "dataset_summary.json"),
        "feature_schema": load_json(root / "feature_schema.json"),
        "split_manifest": load_json(root / "split_manifest.json"),
        "tensor_manifest": load_json(root / "tensor_manifest.json"),
    }

    split_inventory: dict[str, Any] = {}
    pair_sets: dict[str, set[str]] = {}
    all_run_paths: list[Path] = []

    for split in ("train", "validation", "test"):
        split_dir = root / "runs" / split
        files = sorted(split_dir.glob("*.pt"))
        all_run_paths.extend(files)

        classes = Counter(class_name(path) for path in files)
        unrecognized = [
            path.name for path in files if class_name(path) is None
        ]

        attack_by_pair: dict[str, list[str]] = {}
        control_by_pair: dict[str, list[str]] = {}
        for path in files:
            key = pair_key(path)
            kind = class_name(path)
            if key is None or kind is None:
                continue
            target = (
                attack_by_pair if kind == "attack" else control_by_pair
            )
            target.setdefault(key, []).append(path.name)

        attack_keys = set(attack_by_pair)
        control_keys = set(control_by_pair)
        matched = attack_keys & control_keys
        missing_attack = sorted(control_keys - attack_keys)
        missing_control = sorted(attack_keys - control_keys)

        duplicate_attack = {
            key: names
            for key, names in attack_by_pair.items()
            if len(names) != 1
        }
        duplicate_control = {
            key: names
            for key, names in control_by_pair.items()
            if len(names) != 1
        }

        expected = EXPECTED_COUNTS[split]
        actual = {
            "total": len(files),
            "attack": int(classes.get("attack", 0)),
            "control": int(classes.get("control", 0)),
            "pairs": len(matched),
        }

        for key, expected_value in expected.items():
            if actual[key] != expected_value:
                failures.append(
                    f"{split} {key}={actual[key]}, "
                    f"expected {expected_value}"
                )

        if unrecognized:
            failures.append(
                f"{split} has unrecognized .pt filenames: "
                f"{unrecognized[:10]}"
            )
        if missing_attack:
            failures.append(
                f"{split} missing ATTACK mate for "
                f"{len(missing_attack)} pair keys"
            )
        if missing_control:
            failures.append(
                f"{split} missing CONTROL mate for "
                f"{len(missing_control)} pair keys"
            )
        if duplicate_attack:
            failures.append(
                f"{split} duplicate ATTACK members for "
                f"{len(duplicate_attack)} pairs"
            )
        if duplicate_control:
            failures.append(
                f"{split} duplicate CONTROL members for "
                f"{len(duplicate_control)} pairs"
            )

        pair_sets[split] = matched
        split_inventory[split] = {
            "directory": str(split_dir),
            "run_count": len(files),
            "attack_count": actual["attack"],
            "control_count": actual["control"],
            "matched_pair_count": len(matched),
            "total_size_bytes": sum(
                path.stat().st_size for path in files
            ),
            "minimum_run_size_bytes": min(
                (path.stat().st_size for path in files),
                default=0,
            ),
            "maximum_run_size_bytes": max(
                (path.stat().st_size for path in files),
                default=0,
            ),
            "unrecognized_filenames": unrecognized,
            "missing_attack_pair_keys": missing_attack,
            "missing_control_pair_keys": missing_control,
            "duplicate_attack_pairs": duplicate_attack,
            "duplicate_control_pairs": duplicate_control,
            "pair_keys_sha256": canonical_sha256(sorted(matched)),
        }

    total_runs = sum(
        item["run_count"] for item in split_inventory.values()
    )
    total_pairs = sum(
        item["matched_pair_count"] for item in split_inventory.values()
    )
    if total_runs != EXPECTED_TOTAL_RUNS:
        failures.append(
            f"total run count={total_runs}, expected {EXPECTED_TOTAL_RUNS}"
        )
    if total_pairs != EXPECTED_TOTAL_PAIRS:
        failures.append(
            f"total matched pairs={total_pairs}, "
            f"expected {EXPECTED_TOTAL_PAIRS}"
        )

    split_names = ("train", "validation", "test")
    split_overlaps: dict[str, list[str]] = {}
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1:]:
            overlap = sorted(pair_sets[left] & pair_sets[right])
            split_overlaps[f"{left}__{right}"] = overlap
            if overlap:
                failures.append(
                    f"{left}/{right} pair-key overlap: {len(overlap)}"
                )

    summary = documents["dataset_summary"]
    if not isinstance(summary, dict):
        failures.append("dataset_summary.json is not a JSON object")
    else:
        direct_expectations = {
            "total_runs": EXPECTED_TOTAL_RUNS,
            "matched_pairs": EXPECTED_TOTAL_PAIRS,
            "attack_runs": EXPECTED_TOTAL_PAIRS,
            "control_runs": EXPECTED_TOTAL_PAIRS,
        }
        for key, expected in direct_expectations.items():
            if key in summary and summary[key] != expected:
                failures.append(
                    f"dataset_summary {key}={summary[key]!r}, "
                    f"expected {expected}"
                )

        split_run_counts = summary.get("split_run_counts")
        if isinstance(split_run_counts, dict):
            normalized = {
                normalize_split_name(str(key)): value
                for key, value in split_run_counts.items()
            }
            for split in split_names:
                if (
                    split in normalized
                    and normalized[split] != EXPECTED_COUNTS[split]["total"]
                ):
                    failures.append(
                        f"dataset_summary split_run_counts[{split}]="
                        f"{normalized[split]!r}, expected "
                        f"{EXPECTED_COUNTS[split]['total']}"
                    )
        else:
            warnings.append(
                "dataset_summary split_run_counts is not a dictionary"
            )

    feature_schema = documents["feature_schema"]
    feature_count = None
    feature_list_length = None
    if not isinstance(feature_schema, dict):
        failures.append("feature_schema.json is not a JSON object")
    else:
        feature_count = feature_schema.get("feature_count")
        features = feature_schema.get("features")
        if not isinstance(feature_count, int) or feature_count <= 0:
            failures.append(
                f"invalid feature_schema feature_count={feature_count!r}"
            )
        if isinstance(features, list):
            feature_list_length = len(features)
            if (
                isinstance(feature_count, int)
                and feature_list_length != feature_count
            ):
                failures.append(
                    f"feature list length={feature_list_length}, "
                    f"feature_count={feature_count}"
                )
        else:
            failures.append("feature_schema features is not a list")

        if "input_tensor_order" not in feature_schema:
            failures.append("feature_schema lacks input_tensor_order")
        if "forbidden_model_inputs" not in feature_schema:
            failures.append("feature_schema lacks forbidden_model_inputs")
        if "excluded_observation_columns" not in feature_schema:
            failures.append(
                "feature_schema lacks excluded_observation_columns"
            )

    split_manifest_strings = recursively_collect_strings(
        documents["split_manifest"]
    )
    tensor_manifest_strings = recursively_collect_strings(
        documents["tensor_manifest"]
    )
    all_basenames = {path.name for path in all_run_paths}

    split_manifest_pt_basenames = {
        Path(value).name
        for value in split_manifest_strings
        if value.endswith(".pt")
    }
    tensor_manifest_pt_basenames = {
        Path(value).name
        for value in tensor_manifest_strings
        if value.endswith(".pt")
    }

    if split_manifest_pt_basenames:
        missing = sorted(all_basenames - split_manifest_pt_basenames)
        extras = sorted(split_manifest_pt_basenames - all_basenames)
        if missing:
            failures.append(
                f"split manifest omits {len(missing)} run tensor basenames"
            )
        if extras:
            failures.append(
                f"split manifest references {len(extras)} unknown .pt files"
            )
    else:
        warnings.append(
            "split manifest contains no explicit .pt filenames; "
            "structural consistency recorded only"
        )

    if tensor_manifest_pt_basenames:
        missing = sorted(all_basenames - tensor_manifest_pt_basenames)
        extras = sorted(tensor_manifest_pt_basenames - all_basenames)
        if missing:
            failures.append(
                f"tensor manifest omits {len(missing)} run tensor basenames"
            )
        if extras:
            failures.append(
                f"tensor manifest references {len(extras)} unknown .pt files"
            )
    else:
        warnings.append(
            "tensor manifest contains no explicit .pt filenames; "
            "structural consistency recorded only"
        )

    tensor_manifest_runs = documents["tensor_manifest"].get("runs")
    tensor_manifest_run_count = None
    if isinstance(tensor_manifest_runs, list):
        tensor_manifest_run_count = len(tensor_manifest_runs)
    elif isinstance(tensor_manifest_runs, dict):
        tensor_manifest_run_count = len(tensor_manifest_runs)
    else:
        failures.append("tensor_manifest runs is neither list nor dict")
    if (
        tensor_manifest_run_count is not None
        and tensor_manifest_run_count != EXPECTED_TOTAL_RUNS
    ):
        failures.append(
            f"tensor_manifest run count={tensor_manifest_run_count}, "
            f"expected {EXPECTED_TOTAL_RUNS}"
        )

    checksum_entries: dict[str, str] = {}
    try:
        checksum_entries = parse_sha256sums(root / "SHA256SUMS.txt")
    except Exception as exc:
        failures.append(f"SHA256SUMS parse failure: {exc}")

    support_names_to_verify = (
        "README.md",
        "campaign_manifest.json",
        "dataset_loader.py",
        "dataset_summary.json",
        "feature_schema.json",
        "normalization.pt",
        "split_manifest.json",
        "tensor_manifest.json",
        "topology.pt",
    )
    support_records: dict[str, Any] = {}
    for name in support_names_to_verify:
        path = root / name
        record = support_file_record(path)
        expected_digest = checksum_lookup(checksum_entries, root, path)
        record["listed_sha256"] = expected_digest
        record["checksum_match"] = (
            expected_digest == record["sha256"]
            if expected_digest is not None
            else None
        )
        if expected_digest is None:
            warnings.append(f"SHA256SUMS has no recognizable entry for {name}")
        elif expected_digest != record["sha256"]:
            failures.append(f"checksum mismatch for {name}")
        support_records[name] = record

    # Explicitly do not hash or open run tensor files.
    report = {
        "stage": STAGE,
        "status": "COMPLETE" if not failures else "HOLD",
        "decision": (
            "AUTHORIZE_P2_A1_NONTEST_TENSOR_SCHEMA_AUDIT"
            if not failures
            else "BLOCK_P2_A1"
        ),
        "dataset": {
            "link_path": str(root_input.absolute()),
            "link_is_symlink": root_input.is_symlink(),
            "resolved_root": str(root),
            "dataset_name": summary.get("dataset_name"),
            "campaign_id": summary.get("campaign_id"),
            "schema_version": summary.get("schema_version"),
            "feature_count": feature_count,
            "feature_list_length": feature_list_length,
            "num_nodes": summary.get("num_nodes"),
            "recommended_window": summary.get("recommended_window"),
            "recommended_stride": summary.get("recommended_stride"),
            "total_runs": total_runs,
            "matched_pairs": total_pairs,
        },
        "split_inventory": split_inventory,
        "split_pair_key_overlaps": split_overlaps,
        "manifest_checks": {
            "tensor_manifest_run_count": tensor_manifest_run_count,
            "split_manifest_explicit_pt_basename_count": len(
                split_manifest_pt_basenames
            ),
            "tensor_manifest_explicit_pt_basename_count": len(
                tensor_manifest_pt_basenames
            ),
            "sha256sums_entry_count": len(checksum_entries),
        },
        "support_file_integrity": support_records,
        "metadata_top_level_keys": {
            name: sorted(document.keys())
            if isinstance(document, dict)
            else None
            for name, document in documents.items()
        },
        "security_boundary": {
            "run_tensor_files_enumerated": True,
            "run_tensor_file_sizes_read": True,
            "run_tensor_files_opened": False,
            "run_tensor_bytes_read": False,
            "run_tensors_deserialized": False,
            "train_tensor_contents_accessed": False,
            "validation_tensor_contents_accessed": False,
            "test_tensor_contents_accessed": False,
            "test_dataset_constructed": False,
            "test_windows_constructed": False,
            "test_labels_read": False,
            "test_predictions_computed": False,
        },
        "failures": failures,
        "warnings": warnings,
        "next_stage": (
            "V5_P2_A1_TENSOR_SCHEMA_AND_NONTEST_SAMPLE_AUDIT"
            if not failures
            else None
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    if failures:
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        print("warning_count:", len(warnings))
        return 1

    lock = {
        "status": COMPLETE,
        "decision": "AUTHORIZE_P2_A1_NONTEST_TENSOR_SCHEMA_AUDIT",
        "report_sha256": sha256_file(report_path),
        "resolved_dataset_root": str(root),
        "total_runs": total_runs,
        "matched_pairs": total_pairs,
        "split_counts": {
            split: {
                "runs": split_inventory[split]["run_count"],
                "pairs": split_inventory[split]["matched_pair_count"],
            }
            for split in split_names
        },
        "run_tensor_files_opened": False,
        "run_tensor_bytes_read": False,
        "test_tensor_contents_accessed": False,
        "test_dataset_constructed": False,
        "next_stage": "V5_P2_A1_TENSOR_SCHEMA_AND_NONTEST_SAMPLE_AUDIT",
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-A0 INDEPENDENT DATASET METADATA AUDIT =====")
    print("status: COMPLETE")
    print("resolved_root:", root)
    print("total_runs:", total_runs)
    print("matched_pairs:", total_pairs)
    print(
        "train:",
        split_inventory["train"]["run_count"],
        "runs /",
        split_inventory["train"]["matched_pair_count"],
        "pairs",
    )
    print(
        "validation:",
        split_inventory["validation"]["run_count"],
        "runs /",
        split_inventory["validation"]["matched_pair_count"],
        "pairs",
    )
    print(
        "test:",
        split_inventory["test"]["run_count"],
        "runs /",
        split_inventory["test"]["matched_pair_count"],
        "pairs",
    )
    print("feature_count:", feature_count)
    print("num_nodes:", summary.get("num_nodes"))
    print("recommended_window:", summary.get("recommended_window"))
    print("recommended_stride:", summary.get("recommended_stride"))
    print("split_overlap_count:", sum(len(v) for v in split_overlaps.values()))
    print("run_tensor_files_opened: false")
    print("run_tensor_bytes_read: false")
    print("test_tensor_contents_accessed: false")
    print("test_dataset_constructed: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("next_stage: V5_P2_A1_TENSOR_SCHEMA_AND_NONTEST_SAMPLE_AUDIT")
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
