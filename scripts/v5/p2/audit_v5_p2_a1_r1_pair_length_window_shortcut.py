#!/usr/bin/env python3
"""
V5 P2-A1-R1 Pair-Length and Window-Shortcut Audit

Remediates the overly strict A1 assumption that matched ATTACK and CONTROL
runs must have identical temporal lengths.

This stage sequentially loads every TRAIN and VALIDATION run tensor, but never
enumerates or opens runs/test. It verifies internal tensor alignment and then
measures whether pair-length differences create unequal window counts under
the locked/recommended window=32 and stride=8.

No tensor is modified or truncated here.

Possible decisions:
- AUTHORIZE_A2_WITH_NATIVE_PAIR_LENGTHS:
    all matched pairs produce equal window counts despite any epoch delta.
- REQUIRE_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT:
    one or more pairs produce unequal window counts and therefore require a
    pair-aligned window policy before training to remove a mode/count shortcut.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


STAGE = "V5_P2_A1_R1_PAIR_LENGTH_AND_WINDOW_SHORTCUT_AUDIT"
COMPLETE = f"{STAGE}_COMPLETE"

WINDOW = 32
STRIDE = 8

EXPECTED_KEYS = {
    "case_id",
    "category",
    "epoch_id",
    "feature_count",
    "mode",
    "num_nodes",
    "observation_schema",
    "pair_id",
    "role_mask",
    "run_id",
    "split",
    "truth_schema_version",
    "x",
    "y_attack",
    "y_attack_path",
    "y_attacker_count",
    "y_source",
    "y_transit",
    "y_victim",
}

NODE_TARGETS = (
    "y_source",
    "y_transit",
    "y_victim",
    "y_attack_path",
    "role_mask",
)


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


def pair_key(path: Path) -> str | None:
    stem = path.stem
    if stem.endswith("_ATTACK"):
        return stem[:-len("_ATTACK")]
    if stem.endswith("_CONTROL"):
        return stem[:-len("_CONTROL")]
    return None


def mode_from_filename(path: Path) -> str | None:
    if path.stem.endswith("_ATTACK"):
        return "attack"
    if path.stem.endswith("_CONTROL"):
        return "control"
    return None


def window_count(length: int) -> int:
    if length < WINDOW:
        return 0
    return 1 + (length - WINDOW) // STRIDE


def validate_run(
    path: Path,
    expected_split: str,
    failures: list[str],
) -> dict[str, Any]:
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    record: dict[str, Any] = {
        "path": str(path),
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "filename_mode": mode_from_filename(path),
        "filename_pair_key": pair_key(path),
    }

    if not isinstance(payload, dict):
        failures.append(
            f"{expected_split}/{path.name}: payload type "
            f"{type(payload).__name__}, expected dict"
        )
        record["valid"] = False
        return record

    keys = set(payload)
    record["keys"] = sorted(str(key) for key in keys)
    missing = sorted(EXPECTED_KEYS - keys)
    extras = sorted(keys - EXPECTED_KEYS)
    record["missing_keys"] = missing
    record["extra_keys"] = extras

    if missing:
        failures.append(
            f"{expected_split}/{path.name}: missing keys {missing}"
        )
    if extras:
        failures.append(
            f"{expected_split}/{path.name}: unexpected keys {extras}"
        )

    if missing:
        record["valid"] = False
        return record

    expected_mode = mode_from_filename(path)
    expected_case = pair_key(path)

    metadata = {
        "case_id": payload["case_id"],
        "pair_id": payload["pair_id"],
        "split": payload["split"],
        "category": payload["category"],
        "mode": payload["mode"],
        "run_id": payload["run_id"],
        "feature_count": payload["feature_count"],
        "num_nodes": payload["num_nodes"],
        "truth_schema_version": payload["truth_schema_version"],
        "observation_schema": payload["observation_schema"],
    }
    record["metadata"] = metadata

    if payload["split"] != expected_split:
        failures.append(
            f"{expected_split}/{path.name}: payload split="
            f"{payload['split']!r}"
        )
    if payload["mode"] != expected_mode:
        failures.append(
            f"{expected_split}/{path.name}: payload mode="
            f"{payload['mode']!r}, expected {expected_mode!r}"
        )
    if payload["case_id"] != expected_case:
        failures.append(
            f"{expected_split}/{path.name}: case_id="
            f"{payload['case_id']!r}, expected {expected_case!r}"
        )
    if payload["feature_count"] != 81:
        failures.append(
            f"{expected_split}/{path.name}: feature_count="
            f"{payload['feature_count']!r}"
        )
    if payload["num_nodes"] != 16:
        failures.append(
            f"{expected_split}/{path.name}: num_nodes="
            f"{payload['num_nodes']!r}"
        )
    if payload["truth_schema_version"] != 3:
        failures.append(
            f"{expected_split}/{path.name}: truth_schema_version="
            f"{payload['truth_schema_version']!r}"
        )
    if payload["observation_schema"] != "V5RouterEpoch":
        failures.append(
            f"{expected_split}/{path.name}: observation_schema="
            f"{payload['observation_schema']!r}"
        )
    if expected_mode is not None and not str(
        payload["run_id"]
    ).endswith(expected_mode.upper()):
        failures.append(
            f"{expected_split}/{path.name}: run_id suffix does not "
            f"match {expected_mode}"
        )

    x = payload["x"]
    epoch_id = payload["epoch_id"]
    y_attack = payload["y_attack"]
    y_count = payload["y_attacker_count"]

    tensors = {
        "x": x,
        "epoch_id": epoch_id,
        "y_attack": y_attack,
        "y_attacker_count": y_count,
        **{key: payload[key] for key in NODE_TARGETS},
    }
    for key, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            failures.append(
                f"{expected_split}/{path.name}: {key} is not Tensor"
            )

    if not all(
        isinstance(value, torch.Tensor)
        for value in tensors.values()
    ):
        record["valid"] = False
        return record

    if x.ndim != 3 or tuple(x.shape[1:]) != (16, 81):
        failures.append(
            f"{expected_split}/{path.name}: x shape={tuple(x.shape)}, "
            "expected [T,16,81]"
        )
        record["valid"] = False
        return record

    length = int(x.shape[0])
    record["length"] = length
    record["native_window_count"] = window_count(length)

    expected_shapes = {
        "epoch_id": (length,),
        "y_attack": (length,),
        "y_attacker_count": (length,),
        "y_source": (length, 16),
        "y_transit": (length, 16),
        "y_victim": (length, 16),
        "y_attack_path": (length, 16),
        "role_mask": (length, 16),
    }

    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(payload[key].shape)
        if actual_shape != expected_shape:
            failures.append(
                f"{expected_split}/{path.name}: {key} shape="
                f"{actual_shape}, expected {expected_shape}"
            )

    if length < WINDOW:
        failures.append(
            f"{expected_split}/{path.name}: length={length} < {WINDOW}"
        )
    if not x.is_floating_point():
        failures.append(
            f"{expected_split}/{path.name}: x dtype={x.dtype}"
        )
    elif not torch.isfinite(x).all():
        failures.append(
            f"{expected_split}/{path.name}: x has NaN/Inf"
        )

    if epoch_id.dtype != torch.int64:
        failures.append(
            f"{expected_split}/{path.name}: epoch_id dtype="
            f"{epoch_id.dtype}"
        )
    if y_count.dtype != torch.int64:
        failures.append(
            f"{expected_split}/{path.name}: y_attacker_count dtype="
            f"{y_count.dtype}"
        )
    if payload["role_mask"].dtype not in (
        torch.uint8,
        torch.bool,
    ):
        failures.append(
            f"{expected_split}/{path.name}: role_mask dtype="
            f"{payload['role_mask'].dtype}"
        )

    if length > 1:
        differences = epoch_id[1:] - epoch_id[:-1]
        if not bool((differences > 0).all().item()):
            failures.append(
                f"{expected_split}/{path.name}: epoch_id is not "
                "strictly increasing"
            )

    graph_positive = int((y_attack > 0).sum().item())
    count_positive = int((y_count > 0).sum().item())
    role_positive = {
        key: int((payload[key] > 0).sum().item())
        for key in NODE_TARGETS
    }

    record["label_counts"] = {
        "graph_positive_epochs": graph_positive,
        "attacker_count_positive_epochs": count_positive,
        **{
            f"{key}_positive_entries": value
            for key, value in role_positive.items()
        },
    }

    if expected_mode == "control":
        if graph_positive != 0:
            failures.append(
                f"{expected_split}/{path.name}: CONTROL has "
                f"{graph_positive} positive graph epochs"
            )
        if count_positive != 0:
            failures.append(
                f"{expected_split}/{path.name}: CONTROL has "
                f"{count_positive} positive attacker-count epochs"
            )
        for key in NODE_TARGETS:
            if role_positive[key] != 0:
                failures.append(
                    f"{expected_split}/{path.name}: CONTROL has "
                    f"{role_positive[key]} positive {key} entries"
                )
    elif expected_mode == "attack":
        if graph_positive == 0:
            failures.append(
                f"{expected_split}/{path.name}: ATTACK has no "
                "positive graph epochs"
            )
        if count_positive == 0:
            failures.append(
                f"{expected_split}/{path.name}: ATTACK has no "
                "positive attacker-count epochs"
            )
        for key in (
            "y_source",
            "y_victim",
            "y_attack_path",
            "role_mask",
        ):
            if role_positive[key] == 0:
                failures.append(
                    f"{expected_split}/{path.name}: ATTACK has no "
                    f"positive {key} entries"
                )

    record["valid"] = True
    return record


def pair_record(
    split: str,
    key: str,
    attack: dict[str, Any],
    control: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    attack_meta = attack["metadata"]
    control_meta = control["metadata"]

    same_fields = (
        "case_id",
        "pair_id",
        "split",
        "category",
        "feature_count",
        "num_nodes",
        "truth_schema_version",
        "observation_schema",
    )
    metadata_matches = {
        field: attack_meta[field] == control_meta[field]
        for field in same_fields
    }
    for field, matches in metadata_matches.items():
        if not matches:
            failures.append(
                f"{split}/{key}: pair metadata differs for {field}"
            )

    attack_length = int(attack["length"])
    control_length = int(control["length"])
    attack_windows = int(attack["native_window_count"])
    control_windows = int(control["native_window_count"])
    common_length = min(attack_length, control_length)
    common_windows = window_count(common_length)

    return {
        "split": split,
        "pair_key": key,
        "category": attack_meta["category"],
        "pair_id": attack_meta["pair_id"],
        "attack_length": attack_length,
        "control_length": control_length,
        "signed_length_delta_attack_minus_control": (
            attack_length - control_length
        ),
        "absolute_length_delta": abs(
            attack_length - control_length
        ),
        "attack_native_window_count": attack_windows,
        "control_native_window_count": control_windows,
        "signed_native_window_delta_attack_minus_control": (
            attack_windows - control_windows
        ),
        "absolute_native_window_delta": abs(
            attack_windows - control_windows
        ),
        "common_aligned_length": common_length,
        "common_aligned_window_count_per_member": common_windows,
        "attack_windows_removed_by_pair_alignment": (
            attack_windows - common_windows
        ),
        "control_windows_removed_by_pair_alignment": (
            control_windows - common_windows
        ),
        "metadata_matches": all(metadata_matches.values()),
    }


def distribution(values: list[int]) -> dict[str, Any]:
    counts = Counter(values)
    return {
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "mean": (
            sum(values) / len(values)
            if values
            else None
        ),
        "value_counts": {
            str(key): counts[key]
            for key in sorted(counts)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a0-dir", type=Path, required=True)
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root_input = args.root.expanduser()
    root = root_input.resolve()
    a0_dir = args.a0_dir.expanduser().resolve()
    a1_dir = args.a1_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(
            f"STOP: output already exists: {output_dir}",
            file=sys.stderr,
        )
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
        "a1_report": (
            a1_dir
            / "V5_P2_A1_TENSOR_SCHEMA_AND_NONTEST_SAMPLE_AUDIT.json"
        ),
        "a1_hold": (
            a1_dir
            / "V5_P2_A1_TENSOR_SCHEMA_AND_NONTEST_SAMPLE_AUDIT_HOLD"
        ),
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    train_dir = root / "runs" / "train"
    validation_dir = root / "runs" / "validation"
    test_dir = root / "runs" / "test"

    if not train_dir.is_dir():
        failures.append("missing runs/train")
    if not validation_dir.is_dir():
        failures.append("missing runs/validation")
    if not test_dir.is_dir():
        failures.append("missing runs/test")

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        return 1

    a0_report = load_json(paths["a0_report"])
    a0_lock = load_json(paths["a0_lock"])
    a1_report = load_json(paths["a1_report"])

    if a0_report.get("status") != "COMPLETE":
        failures.append("A0 status is not COMPLETE")
    if (
        a0_lock.get("report_sha256")
        != sha256_file(paths["a0_report"])
    ):
        failures.append("A0 report SHA mismatch")
    if a1_report.get("status") != "HOLD":
        failures.append("A1 report status is not HOLD")
    if a1_report.get("security_boundary", {}).get(
        "test_tensor_contents_accessed"
    ) is not False:
        failures.append("A1 does not certify untouched test tensors")

    expected_a1_failure_fragments = {
        "ATTACK/CONTROL x shapes differ",
        "ATTACK/CONTROL lengths differ",
    }
    a1_failures = a1_report.get("failures", [])
    for fragment in expected_a1_failure_fragments:
        if not any(fragment in item for item in a1_failures):
            failures.append(
                f"A1 HOLD lacks expected failure fragment: {fragment}"
            )

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
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

    print(
        "===== V5 P2-A1-R1 PAIR-LENGTH + WINDOW-SHORTCUT AUDIT ====="
    )
    print("window:", WINDOW)
    print("stride:", STRIDE)
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")

    all_pair_rows: list[dict[str, Any]] = []
    split_run_counts: dict[str, int] = {}
    split_pair_counts: dict[str, int] = {}
    loaded_run_count = 0

    for split, split_dir in (
        ("train", train_dir),
        ("validation", validation_dir),
    ):
        files = sorted(split_dir.glob("*.pt"))
        split_run_counts[split] = len(files)

        records_by_pair: dict[str, dict[str, dict[str, Any]]] = {}
        for index, path in enumerate(files, start=1):
            record = validate_run(path, split, failures)
            loaded_run_count += 1

            key = record.get("filename_pair_key")
            mode = record.get("filename_mode")
            if key is None or mode is None:
                failures.append(
                    f"{split}/{path.name}: cannot infer pair/mode"
                )
                continue

            members = records_by_pair.setdefault(key, {})
            if mode in members:
                failures.append(
                    f"{split}/{key}: duplicate {mode} member"
                )
            members[mode] = record

            if index % 100 == 0 or index == len(files):
                print(
                    f"{split}: loaded {index}/{len(files)} run tensors"
                )

        split_pair_counts[split] = len(records_by_pair)
        for key, members in sorted(records_by_pair.items()):
            attack = members.get("attack")
            control = members.get("control")
            if attack is None or control is None:
                failures.append(
                    f"{split}/{key}: incomplete matched pair"
                )
                continue
            if not attack.get("valid") or not control.get("valid"):
                continue
            all_pair_rows.append(
                pair_record(
                    split,
                    key,
                    attack,
                    control,
                    failures,
                )
            )

    unequal_length_rows = [
        row for row in all_pair_rows
        if row["absolute_length_delta"] > 0
    ]
    unequal_window_rows = [
        row for row in all_pair_rows
        if row["absolute_native_window_delta"] > 0
    ]

    total_native_windows = sum(
        row["attack_native_window_count"]
        + row["control_native_window_count"]
        for row in all_pair_rows
    )
    total_aligned_windows = sum(
        2 * row["common_aligned_window_count_per_member"]
        for row in all_pair_rows
    )
    windows_removed = total_native_windows - total_aligned_windows

    length_deltas = [
        int(row["signed_length_delta_attack_minus_control"])
        for row in all_pair_rows
    ]
    absolute_length_deltas = [
        int(row["absolute_length_delta"])
        for row in all_pair_rows
    ]
    window_deltas = [
        int(row["signed_native_window_delta_attack_minus_control"])
        for row in all_pair_rows
    ]
    absolute_window_deltas = [
        int(row["absolute_native_window_delta"])
        for row in all_pair_rows
    ]

    pair_csv = output_dir / "V5_P2_A1_R1_PAIR_LENGTH_WINDOW_AUDIT.csv"
    write_csv(pair_csv, all_pair_rows)

    if failures:
        status = "HOLD"
        decision = "BLOCK_REMEDIATION_CONTRACT"
        next_stage = None
    elif unequal_window_rows:
        status = "COMPLETE"
        decision = "REQUIRE_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT"
        next_stage = "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT"
    else:
        status = "COMPLETE"
        decision = "AUTHORIZE_A2_WITH_NATIVE_PAIR_LENGTHS"
        next_stage = "V5_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT"

    report = {
        "stage": STAGE,
        "status": status,
        "decision": decision,
        "dataset": {
            "link_path": str(root_input.absolute()),
            "resolved_root": str(root),
            "window": WINDOW,
            "stride": STRIDE,
        },
        "provenance": {
            "a0_report_sha256": sha256_file(paths["a0_report"]),
            "a0_lock_sha256": sha256_file(paths["a0_lock"]),
            "a1_hold_report_sha256": sha256_file(paths["a1_report"]),
            "a1_hold_marker_sha256": sha256_file(paths["a1_hold"]),
        },
        "coverage": {
            "train_run_tensors_loaded": split_run_counts.get("train", 0),
            "validation_run_tensors_loaded": split_run_counts.get(
                "validation",
                0,
            ),
            "total_nontest_run_tensors_loaded": loaded_run_count,
            "train_pairs_audited": split_pair_counts.get("train", 0),
            "validation_pairs_audited": split_pair_counts.get(
                "validation",
                0,
            ),
            "total_nontest_pairs_audited": len(all_pair_rows),
        },
        "results": {
            "unequal_length_pair_count": len(unequal_length_rows),
            "unequal_native_window_count_pair_count": len(
                unequal_window_rows
            ),
            "equal_length_pair_count": (
                len(all_pair_rows) - len(unequal_length_rows)
            ),
            "equal_native_window_count_pair_count": (
                len(all_pair_rows) - len(unequal_window_rows)
            ),
            "signed_length_delta_distribution": distribution(
                length_deltas
            ),
            "absolute_length_delta_distribution": distribution(
                absolute_length_deltas
            ),
            "signed_native_window_delta_distribution": distribution(
                window_deltas
            ),
            "absolute_native_window_delta_distribution": distribution(
                absolute_window_deltas
            ),
            "total_native_windows_both_members": total_native_windows,
            "total_pair_aligned_windows_both_members": (
                total_aligned_windows
            ),
            "total_windows_removed_by_pair_alignment": windows_removed,
            "pair_alignment_removed_fraction": (
                windows_removed / total_native_windows
                if total_native_windows
                else 0.0
            ),
            "first_20_unequal_length_pairs": unequal_length_rows[:20],
            "first_20_unequal_window_pairs": unequal_window_rows[:20],
        },
        "recommended_contract": (
            {
                "policy": "PAIR_ALIGNED_COMMON_PREFIX",
                "per_pair_common_length": (
                    "min(T_attack, T_control)"
                ),
                "window": WINDOW,
                "stride": STRIDE,
                "identical_window_endpoints_for_pair_members": True,
                "independent_native_window_counts_forbidden": True,
                "reason": (
                    "prevent attack/control mode leakage through run "
                    "length, sample count, and terminal-window position"
                ),
            }
            if unequal_window_rows
            else None
        ),
        "artifacts": {
            "pair_csv": [
                str(pair_csv),
                sha256_file(pair_csv),
            ],
        },
        "security_boundary": {
            "train_tensor_contents_accessed": True,
            "validation_tensor_contents_accessed": True,
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
        "next_stage": next_stage,
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    if failures:
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures[:100]:
            print("FAIL:", failure)
        print("failure_count:", len(failures))
        print("warning_count:", len(warnings))
        return 1

    lock = {
        "status": COMPLETE,
        "decision": decision,
        "report_sha256": sha256_file(report_path),
        "pair_csv_sha256": sha256_file(pair_csv),
        "window": WINDOW,
        "stride": STRIDE,
        "total_nontest_run_tensors_loaded": loaded_run_count,
        "total_nontest_pairs_audited": len(all_pair_rows),
        "unequal_length_pair_count": len(unequal_length_rows),
        "unequal_native_window_count_pair_count": len(
            unequal_window_rows
        ),
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "next_stage": next_stage,
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("\n===== V5 P2-A1-R1 FINAL =====")
    print("status: COMPLETE")
    print("decision:", decision)
    print("nontest_run_tensors_loaded:", loaded_run_count)
    print("nontest_pairs_audited:", len(all_pair_rows))
    print("unequal_length_pair_count:", len(unequal_length_rows))
    print(
        "unequal_native_window_count_pair_count:",
        len(unequal_window_rows),
    )
    print("total_native_windows:", total_native_windows)
    print("total_pair_aligned_windows:", total_aligned_windows)
    print("windows_removed_by_pair_alignment:", windows_removed)
    print(
        "pair_alignment_removed_fraction:",
        f"{report['results']['pair_alignment_removed_fraction']:.8f}",
    )
    print(
        "maximum_absolute_length_delta:",
        max(absolute_length_deltas, default=0),
    )
    print(
        "maximum_absolute_native_window_delta:",
        max(absolute_window_deltas, default=0),
    )
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("next_stage:", next_stage)
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
