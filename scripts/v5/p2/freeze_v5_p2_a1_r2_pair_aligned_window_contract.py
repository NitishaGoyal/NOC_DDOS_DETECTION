#!/usr/bin/env python3
"""
V5 P2-A1-R2 Pair-Aligned Window Contract

Freezes the remediation required by A1-R1:

For every TRAIN or VALIDATION ATTACK/CONTROL pair:
    common_length = min(T_attack, T_control)

Both members use the identical ordered window starts:
    0, stride, 2*stride, ...
up to the last full window inside common_length.

This stage reads only the A1-R1 report and pair CSV. It does not deserialize
any run tensor and does not enumerate or open runs/test.
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


STAGE = "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT"
COMPLETE = f"{STAGE}_COMPLETE"
WINDOW = 32
STRIDE = 8

EXPECTED = {
    "pair_count": 489,
    "train_pairs": 415,
    "validation_pairs": 74,
    "native_windows": 82708,
    "aligned_windows": 82694,
    "removed_windows": 14,
    "unequal_length_pairs": 77,
    "unequal_native_window_pairs": 14,
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


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid integer field {key!r} in pair row "
            f"{row.get('split')}/{row.get('pair_key')}"
        ) from exc


def window_count(length: int) -> int:
    if length < WINDOW:
        return 0
    return 1 + (length - WINDOW) // STRIDE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a0-dir", type=Path, required=True)
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--a1-r1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root_input = args.root.expanduser()
    root = root_input.resolve()
    a0_dir = args.a0_dir.expanduser().resolve()
    a1_dir = args.a1_dir.expanduser().resolve()
    a1_r1_dir = args.a1_r1_dir.expanduser().resolve()
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
        "a1_report": (
            a1_dir
            / "V5_P2_A1_TENSOR_SCHEMA_AND_NONTEST_SAMPLE_AUDIT.json"
        ),
        "a1_hold": (
            a1_dir
            / "V5_P2_A1_TENSOR_SCHEMA_AND_NONTEST_SAMPLE_AUDIT_HOLD"
        ),
        "r1_report": (
            a1_r1_dir
            / "V5_P2_A1_R1_PAIR_LENGTH_AND_WINDOW_SHORTCUT_AUDIT.json"
        ),
        "r1_lock": (
            a1_r1_dir
            / "V5_P2_A1_R1_PAIR_LENGTH_AND_WINDOW_SHORTCUT_AUDIT_LOCK.json"
        ),
        "r1_csv": (
            a1_r1_dir
            / "V5_P2_A1_R1_PAIR_LENGTH_WINDOW_AUDIT.csv"
        ),
        "r1_marker": (
            a1_r1_dir
            / "V5_P2_A1_R1_PAIR_LENGTH_AND_WINDOW_SHORTCUT_AUDIT_COMPLETE"
        ),
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    if not root.is_dir():
        failures.append(f"dataset root is missing: {root}")

    documents: dict[str, Any] = {}
    rows: list[dict[str, str]] = []

    if not failures:
        documents["a0_report"] = load_json(paths["a0_report"])
        documents["a0_lock"] = load_json(paths["a0_lock"])
        documents["a1_report"] = load_json(paths["a1_report"])
        documents["r1_report"] = load_json(paths["r1_report"])
        documents["r1_lock"] = load_json(paths["r1_lock"])
        rows = load_csv(paths["r1_csv"])

        if documents["a0_report"].get("status") != "COMPLETE":
            failures.append("A0 status is not COMPLETE")
        if documents["a1_report"].get("status") != "HOLD":
            failures.append("A1 historical report is not HOLD")
        if documents["r1_report"].get("status") != "COMPLETE":
            failures.append("A1-R1 status is not COMPLETE")
        if (
            documents["r1_report"].get("decision")
            != "REQUIRE_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT"
        ):
            failures.append(
                "A1-R1 did not require pair-aligned window remediation"
            )

        if (
            documents["a0_lock"].get("report_sha256")
            != sha256_file(paths["a0_report"])
        ):
            failures.append("A0 report SHA mismatch")
        if (
            documents["r1_lock"].get("report_sha256")
            != sha256_file(paths["r1_report"])
        ):
            failures.append("A1-R1 report SHA mismatch")
        if (
            documents["r1_lock"].get("pair_csv_sha256")
            != sha256_file(paths["r1_csv"])
        ):
            failures.append("A1-R1 pair CSV SHA mismatch")

        for label, document in (
            ("A0", documents["a0_report"]),
            ("A1", documents["a1_report"]),
            ("A1-R1", documents["r1_report"]),
        ):
            security = document.get("security_boundary", {})
            if security.get("test_tensor_contents_accessed") is not False:
                failures.append(
                    f"{label} does not certify untouched test tensors"
                )

        if documents["r1_lock"].get("window") != WINDOW:
            failures.append("A1-R1 window is not 32")
        if documents["r1_lock"].get("stride") != STRIDE:
            failures.append("A1-R1 stride is not 8")

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "run_tensors_deserialized": False,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    required_columns = {
        "split",
        "pair_key",
        "category",
        "pair_id",
        "attack_length",
        "control_length",
        "attack_native_window_count",
        "control_native_window_count",
        "common_aligned_length",
        "common_aligned_window_count_per_member",
        "attack_windows_removed_by_pair_alignment",
        "control_windows_removed_by_pair_alignment",
    }
    missing_columns = sorted(required_columns - set(rows[0] if rows else []))
    if missing_columns:
        failures.append(
            f"A1-R1 pair CSV missing columns: {missing_columns}"
        )

    seen: set[tuple[str, str]] = set()
    manifest_rows: list[dict[str, Any]] = []
    split_counts = Counter()
    unequal_length_count = 0
    unequal_window_count = 0
    native_total = 0
    aligned_total = 0
    removed_total = 0

    for row in rows:
        split = row.get("split", "")
        pair_key = row.get("pair_key", "")
        identity = (split, pair_key)

        if split not in {"train", "validation"}:
            failures.append(
                f"unexpected split {split!r} for pair {pair_key!r}"
            )
            continue
        if identity in seen:
            failures.append(
                f"duplicate pair row: {split}/{pair_key}"
            )
            continue
        seen.add(identity)
        split_counts[split] += 1

        attack_length = as_int(row, "attack_length")
        control_length = as_int(row, "control_length")
        attack_native = as_int(
            row,
            "attack_native_window_count",
        )
        control_native = as_int(
            row,
            "control_native_window_count",
        )
        common_length = as_int(row, "common_aligned_length")
        common_windows = as_int(
            row,
            "common_aligned_window_count_per_member",
        )
        attack_removed = as_int(
            row,
            "attack_windows_removed_by_pair_alignment",
        )
        control_removed = as_int(
            row,
            "control_windows_removed_by_pair_alignment",
        )

        expected_common_length = min(
            attack_length,
            control_length,
        )
        expected_attack_native = window_count(attack_length)
        expected_control_native = window_count(control_length)
        expected_common_windows = window_count(
            expected_common_length
        )

        if common_length != expected_common_length:
            failures.append(
                f"{split}/{pair_key}: common_length={common_length}, "
                f"expected {expected_common_length}"
            )
        if attack_native != expected_attack_native:
            failures.append(
                f"{split}/{pair_key}: attack native windows mismatch"
            )
        if control_native != expected_control_native:
            failures.append(
                f"{split}/{pair_key}: control native windows mismatch"
            )
        if common_windows != expected_common_windows:
            failures.append(
                f"{split}/{pair_key}: aligned window count mismatch"
            )
        if attack_removed != attack_native - common_windows:
            failures.append(
                f"{split}/{pair_key}: attack removed-window mismatch"
            )
        if control_removed != control_native - common_windows:
            failures.append(
                f"{split}/{pair_key}: control removed-window mismatch"
            )
        if attack_removed < 0 or control_removed < 0:
            failures.append(
                f"{split}/{pair_key}: negative removed-window count"
            )

        if attack_length != control_length:
            unequal_length_count += 1
        if attack_native != control_native:
            unequal_window_count += 1

        native_total += attack_native + control_native
        aligned_total += 2 * common_windows
        removed_total += attack_removed + control_removed

        last_start = (
            (common_windows - 1) * STRIDE
            if common_windows > 0
            else None
        )
        last_end_exclusive = (
            last_start + WINDOW
            if last_start is not None
            else None
        )

        manifest_rows.append(
            {
                "split": split,
                "pair_key": pair_key,
                "category": row.get("category"),
                "pair_id": row.get("pair_id"),
                "attack_length": attack_length,
                "control_length": control_length,
                "common_length": common_length,
                "window": WINDOW,
                "stride": STRIDE,
                "window_count_per_member": common_windows,
                "first_window_start": 0,
                "last_window_start": last_start,
                "last_window_end_exclusive": last_end_exclusive,
                "attack_native_window_count": attack_native,
                "control_native_window_count": control_native,
                "attack_windows_removed": attack_removed,
                "control_windows_removed": control_removed,
                "identical_ordered_window_starts": True,
            }
        )

    actuals = {
        "pair_count": len(manifest_rows),
        "train_pairs": split_counts["train"],
        "validation_pairs": split_counts["validation"],
        "native_windows": native_total,
        "aligned_windows": aligned_total,
        "removed_windows": removed_total,
        "unequal_length_pairs": unequal_length_count,
        "unequal_native_window_pairs": unequal_window_count,
    }
    for key, expected in EXPECTED.items():
        if actuals[key] != expected:
            failures.append(
                f"{key}={actuals[key]}, expected {expected}"
            )

    if native_total - aligned_total != removed_total:
        failures.append(
            "native-aligned total does not equal removed total"
        )

    manifest_rows.sort(
        key=lambda item: (item["split"], item["pair_key"])
    )
    manifest_path = (
        output_dir
        / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
    )
    write_csv(manifest_path, manifest_rows)

    contract = {
        "contract_name": "V5_P2_PAIR_ALIGNED_COMMON_PREFIX_WINDOWS",
        "contract_version": 1,
        "scope": {
            "splits": ["train", "validation"],
            "test_excluded": True,
            "matched_pair_unit": "ATTACK_CONTROL_PAIR",
        },
        "windowing": {
            "window": WINDOW,
            "stride": STRIDE,
            "common_length_rule": "min(T_attack,T_control)",
            "window_start_rule": (
                "start_k = k*stride for "
                "k=0..floor((common_length-window)/stride)"
            ),
            "window_interval": "[start_k,start_k+window)",
            "identical_ordered_window_starts_for_pair_members": True,
            "native_unaligned_windowing_forbidden": True,
            "source_tensors_modified": False,
            "source_tensors_truncated_on_disk": False,
            "loader_slices_common_prefix_only": True,
        },
        "sampling": {
            "all_aligned_windows_included": True,
            "equal_windows_per_pair_member": True,
            "attack_control_sample_balance_within_each_pair": True,
            "pair_members_may_not_be_sampled_asymmetrically": True,
        },
        "learned_input_prohibitions": [
            "case_id",
            "pair_id",
            "run_id",
            "filename",
            "split",
            "mode",
            "category",
            "native_run_length",
            "common_length",
            "native_window_count",
            "aligned_window_count",
            "window_start",
            "window_end",
            "distance_to_terminal_epoch",
            "serialization_size",
            "manifest_position",
            "any direct or derived provenance identifier",
        ],
        "bookkeeping_only_fields": [
            "split",
            "pair_key",
            "pair_id",
            "category",
            "attack_length",
            "control_length",
            "common_length",
            "window_count_per_member",
        ],
        "label_semantics": {
            "status": "DEFERRED_TO_P2_LOADER_CONTRACT",
            "statement": (
                "This remediation changes only eligible window indices. "
                "It does not redefine target extraction or aggregation."
            ),
        },
        "audit_totals": {
            **actuals,
            "removed_fraction": (
                removed_total / native_total
                if native_total
                else 0.0
            ),
        },
        "reason": (
            "Remove ATTACK/CONTROL shortcut leakage through unequal "
            "run lengths, unequal window counts, and terminal-window "
            "position while discarding only 14 of 82,708 native "
            "non-test windows."
        ),
        "pair_manifest_file": str(manifest_path),
        "pair_manifest_file_sha256": sha256_file(manifest_path),
        "test_tensor_contents_accessed": False,
    }
    contract["contract_sha256"] = canonical_sha256(contract)

    contract_path = (
        output_dir
        / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT.json"
    )
    write_json(contract_path, contract)

    markdown = f"""# V5 P2 Pair-Aligned Window Contract

## Decision

Use a common-prefix window index for every TRAIN and VALIDATION matched pair.

```text
common_length = min(T_attack, T_control)
window = {WINDOW}
stride = {STRIDE}
starts = 0, 8, 16, ... while start + 32 <= common_length
```

Both ATTACK and CONTROL members receive exactly the same ordered starts.

## Audit result

```text
non-test pairs                         = {actuals["pair_count"]}
pairs with unequal epoch lengths       = {actuals["unequal_length_pairs"]}
pairs with unequal native window count = {actuals["unequal_native_window_pairs"]}
native windows                         = {actuals["native_windows"]}
aligned windows                        = {actuals["aligned_windows"]}
windows removed                        = {actuals["removed_windows"]}
removed fraction                       = {contract["audit_totals"]["removed_fraction"]:.8f}
```

Only 14 windows are removed, so the leakage control costs approximately
0.0169% of the native non-test windows.

## Security rule

Run length, pair identity, mode, category, window position, terminal distance,
filenames, IDs, and all derived provenance values are bookkeeping only and
must never be supplied to the model.

## Source immutability

Original `.pt` tensors remain unchanged. Alignment is applied by the loader
when constructing eligible windows.

## Test boundary

No P2 test directory enumeration or test tensor access occurred in this stage.

## Contract hash

`{contract["contract_sha256"]}`
"""
    markdown_path = (
        output_dir
        / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT.md"
    )
    atomic_write(markdown_path, markdown)

    status = "COMPLETE" if not failures else "HOLD"
    report = {
        "stage": STAGE,
        "status": status,
        "decision": (
            "FREEZE_PAIR_ALIGNED_WINDOW_CONTRACT_AND_AUTHORIZE_A2"
            if not failures
            else "BLOCK_A2"
        ),
        "dataset": {
            "link_path": str(root_input.absolute()),
            "resolved_root": str(root),
        },
        "contract": contract,
        "provenance": {
            "a0_report_sha256": sha256_file(paths["a0_report"]),
            "a0_lock_sha256": sha256_file(paths["a0_lock"]),
            "a1_hold_report_sha256": sha256_file(paths["a1_report"]),
            "a1_hold_marker_sha256": sha256_file(paths["a1_hold"]),
            "a1_r1_report_sha256": sha256_file(paths["r1_report"]),
            "a1_r1_lock_sha256": sha256_file(paths["r1_lock"]),
            "a1_r1_pair_csv_sha256": sha256_file(paths["r1_csv"]),
        },
        "artifacts": {
            "contract_json": [
                str(contract_path),
                sha256_file(contract_path),
            ],
            "contract_markdown": [
                str(markdown_path),
                sha256_file(markdown_path),
            ],
            "pair_aligned_manifest": [
                str(manifest_path),
                sha256_file(manifest_path),
            ],
        },
        "security_boundary": {
            "run_tensors_deserialized": False,
            "train_tensor_contents_accessed": False,
            "validation_tensor_contents_accessed": False,
            "test_directory_existence_checked": False,
            "test_directory_enumerated": False,
            "test_tensor_files_opened": False,
            "test_tensor_bytes_read": False,
            "test_tensor_contents_accessed": False,
            "test_dataset_constructed": False,
            "test_windows_constructed": False,
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
        for failure in failures:
            print("FAIL:", failure)
        return 1

    lock = {
        "status": COMPLETE,
        "decision": (
            "FREEZE_PAIR_ALIGNED_WINDOW_CONTRACT_AND_AUTHORIZE_A2"
        ),
        "report_sha256": sha256_file(report_path),
        "contract_file_sha256": sha256_file(contract_path),
        "contract_sha256": contract["contract_sha256"],
        "pair_manifest_sha256": sha256_file(manifest_path),
        "window": WINDOW,
        "stride": STRIDE,
        **actuals,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "next_stage": (
            "V5_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT"
        ),
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-A1-R2 PAIR-ALIGNED WINDOW CONTRACT =====")
    print("status: COMPLETE")
    print(
        "decision: "
        "FREEZE_PAIR_ALIGNED_WINDOW_CONTRACT_AND_AUTHORIZE_A2"
    )
    print("window:", WINDOW)
    print("stride:", STRIDE)
    print("pair_count:", actuals["pair_count"])
    print("train_pairs:", actuals["train_pairs"])
    print("validation_pairs:", actuals["validation_pairs"])
    print(
        "unequal_length_pairs:",
        actuals["unequal_length_pairs"],
    )
    print(
        "unequal_native_window_pairs:",
        actuals["unequal_native_window_pairs"],
    )
    print("native_windows:", actuals["native_windows"])
    print("aligned_windows:", actuals["aligned_windows"])
    print("removed_windows:", actuals["removed_windows"])
    print(
        "removed_fraction:",
        f"{contract['audit_totals']['removed_fraction']:.8f}",
    )
    print("contract_sha256:", contract["contract_sha256"])
    print("run_tensors_deserialized: false")
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")
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
