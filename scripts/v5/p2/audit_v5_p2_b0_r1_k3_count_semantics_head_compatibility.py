#!/usr/bin/env python3
"""
V5 P2-B0-R1 K3 Count-Semantics and Head-Compatibility Audit

Why this stage exists
---------------------
The historical B0 audit assumed only K1/K2/K4 categories. The complete P2
dataset also contains K3 pairs. This stage determines the exact non-test count
semantics and checks whether the frozen P0/B3 three-logit attacker-count head
can represent them.

This is a diagnosis/remediation gate only. It does not alter the model,
dataset, loader, or prior reports.

Coverage
--------
- all TRAIN and VALIDATION pair-manifest rows;
- aligned target epochs only;
- ATTACK and CONTROL labels;
- no test-directory enumeration and no test-tensor access.

Expected clean diagnosis
------------------------
If positive aligned windows use counts {1,2,3,4}, then:
- K1 -> 1
- K2 -> 2
- K3 -> 3
- K4 -> 4
- current three-logit head is incompatible;
- P2 requires four active-count logits;
- frozen temporal encoder and all other heads can remain unchanged;
- A4 must be repeated with all four count classes before B0-R2.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


STAGE = "V5_P2_B0_R1_K3_COUNT_SEMANTICS_AND_HEAD_COMPATIBILITY_AUDIT"
COMPLETE = f"{STAGE}_COMPLETE"

WINDOW = 32
STRIDE = 8
EXPECTED_CATEGORIES = (1, 2, 3, 4)
EXPECTED_ACTIVE_COUNTS = (1, 2, 3, 4)
EXPECTED_CURRENT_COUNT_LOGITS = 3
EXPECTED_CURRENT_PARAMETER_COUNT = 43_208
EXPECTED_EXPANDED_PARAMETER_COUNT = 43_273


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
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


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid integer field {key!r} for "
            f"{row.get('split')}/{row.get('pair_key')}"
        ) from exc


def infer_k(pair_key: str) -> int | None:
    match = re.search(r"-K([1-4])-", pair_key)
    return int(match.group(1)) if match else None


def import_model(path: Path):
    spec = importlib.util.spec_from_file_location(
        "v5_frozen_b3_count_contract_b0_r1",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import model from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a1-r2-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--a4-dir", type=Path, required=True)
    parser.add_argument("--b0-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root_input = args.root.expanduser()
    root = root_input.resolve()
    a1_r2_dir = args.a1_r2_dir.expanduser().resolve()
    a3_dir = args.a3_dir.expanduser().resolve()
    a4_dir = args.a4_dir.expanduser().resolve()
    b0_dir = args.b0_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    paths = {
        "pair_manifest": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
        ),
        "a1_r2_report": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT.json"
        ),
        "a1_r2_lock": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT_LOCK.json"
        ),
        "a3_report": (
            a3_dir
            / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT.json"
        ),
        "a3_lock": (
            a3_dir
            / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT_LOCK.json"
        ),
        "a4_report": (
            a4_dir
            / "V5_P2_A4_LOADER_INTEGRATION_AND_TINY_OVERFIT.json"
        ),
        "a4_lock": (
            a4_dir
            / "V5_P2_A4_LOADER_INTEGRATION_AND_TINY_OVERFIT_LOCK.json"
        ),
        "b0_report": (
            b0_dir
            / "V5_P2_B0_NONTEST_LABEL_AND_SHORTCUT_AUDIT.json"
        ),
        "b0_hold": (
            b0_dir
            / "V5_P2_B0_NONTEST_LABEL_AND_SHORTCUT_AUDIT_HOLD"
        ),
        "model": model_path,
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    for split in ("train", "validation"):
        if not (root / "runs" / split).is_dir():
            failures.append(f"missing runs/{split}")

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
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        return 1

    a1_report = load_json(paths["a1_r2_report"])
    a1_lock = load_json(paths["a1_r2_lock"])
    a3_report = load_json(paths["a3_report"])
    a3_lock = load_json(paths["a3_lock"])
    a4_report = load_json(paths["a4_report"])
    a4_lock = load_json(paths["a4_lock"])
    b0_report = load_json(paths["b0_report"])

    for label, report, lock, report_path in (
        ("A1-R2", a1_report, a1_lock, paths["a1_r2_report"]),
        ("A3", a3_report, a3_lock, paths["a3_report"]),
        ("A4", a4_report, a4_lock, paths["a4_report"]),
    ):
        if report.get("status") != "COMPLETE":
            failures.append(f"{label} status is not COMPLETE")
        if lock.get("report_sha256") != sha256_file(report_path):
            failures.append(f"{label} report SHA mismatch")
        if report.get("security_boundary", {}).get(
            "test_tensor_contents_accessed"
        ) is not False:
            failures.append(
                f"{label} does not certify untouched test tensors"
            )

    if b0_report.get("status") != "HOLD":
        failures.append("historical B0 status is not HOLD")

    b0_failures = b0_report.get("failures", [])
    if len(b0_failures) != 122:
        failures.append(
            f"historical B0 failure count={len(b0_failures)}, expected 122"
        )
    unexpected_b0 = [
        item for item in b0_failures
        if "cannot infer K1/K2/K4" not in item
    ]
    if unexpected_b0:
        failures.append(
            "historical B0 HOLD contains failures unrelated to K3"
        )
    if b0_report.get("security_boundary", {}).get(
        "test_tensor_contents_accessed"
    ) is not False:
        failures.append("historical B0 does not certify untouched test")

    if a1_lock.get("pair_manifest_sha256") != sha256_file(
        paths["pair_manifest"]
    ):
        failures.append("A1-R2 pair-manifest SHA mismatch")
    if a1_lock.get("pair_count") != 489:
        failures.append("A1-R2 pair count changed")
    if a1_lock.get("aligned_windows") != 82694:
        failures.append("A1-R2 aligned-window count changed")
    if a3_lock.get("total_aligned_items") != 82694:
        failures.append("A3 aligned-item count changed")
    if a4_lock.get("positive_raw_attacker_counts") != [1, 2, 4]:
        failures.append(
            "A4 historical count coverage is not exactly [1,2,4]"
        )
    if a4_lock.get("all_tasks_all_exact") is not True:
        failures.append("A4 did not complete exact tiny overfit")

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
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    model_module = import_model(model_path)
    model = model_module.FrozenB3Conv1DOnly()
    current_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    current_count_logits = int(model.count_head.out_features)

    if current_parameter_count != EXPECTED_CURRENT_PARAMETER_COUNT:
        failures.append(
            f"current B3 parameter count={current_parameter_count}, "
            f"expected {EXPECTED_CURRENT_PARAMETER_COUNT}"
        )
    if current_count_logits != EXPECTED_CURRENT_COUNT_LOGITS:
        failures.append(
            f"current count logits={current_count_logits}, expected 3"
        )

    rows = load_manifest_rows(paths["pair_manifest"])
    category_pair_counts = Counter()
    category_split_pair_counts: dict[int, Counter[str]] = defaultdict(Counter)
    active_count_distribution = Counter()
    category_active_count_distribution: dict[int, Counter[int]] = defaultdict(
        Counter
    )
    split_active_count_distribution: dict[str, Counter[int]] = defaultdict(
        Counter
    )
    category_active_windows = Counter()
    category_inactive_attack_windows = Counter()
    category_control_windows = Counter()

    violation_counts = Counter()
    inspected_runs = Counter()
    inspected_pairs = Counter()
    inspected_aligned_items = Counter()

    first_examples: dict[int, dict[str, Any]] = {}

    print("===== V5 P2-B0-R1 K3 COUNT SEMANTICS AUDIT =====")
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")

    for row_index, row in enumerate(rows, start=1):
        split = row.get("split", "")
        pair_key = row.get("pair_key", "")
        category = infer_k(pair_key)

        if split not in {"train", "validation"}:
            failures.append(
                f"unexpected split {split!r} for {pair_key}"
            )
            continue
        if category not in EXPECTED_CATEGORIES:
            failures.append(
                f"{split}/{pair_key}: cannot infer K1/K2/K3/K4"
            )
            continue

        common_length = as_int(row, "common_length")
        window_count = as_int(row, "window_count_per_member")
        target_indices = torch.arange(
            WINDOW - 1,
            common_length,
            STRIDE,
            dtype=torch.int64,
        )
        if int(target_indices.numel()) != window_count:
            failures.append(
                f"{split}/{pair_key}: aligned target count mismatch"
            )
            continue

        category_pair_counts[category] += 1
        category_split_pair_counts[category][split] += 1
        inspected_pairs[split] += 1

        pair_example = {
            "split": split,
            "pair_key": pair_key,
            "category": category,
            "common_length": common_length,
            "window_count_per_member": window_count,
        }

        for mode in ("attack", "control"):
            path = (
                root
                / "runs"
                / split
                / f"{pair_key}_{mode.upper()}.pt"
            )
            if not path.is_file():
                failures.append(f"missing run tensor: {path}")
                continue

            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            inspected_runs[split] += 1
            inspected_aligned_items[split] += window_count

            if not isinstance(payload, dict):
                failures.append(f"{split}/{path.name}: payload not dict")
                continue
            if payload.get("mode") != mode:
                violation_counts["payload_mode_mismatch"] += 1
            if payload.get("case_id") != pair_key:
                violation_counts["payload_case_id_mismatch"] += 1

            count = payload["y_attacker_count"][target_indices].to(
                torch.int64
            )
            graph = payload["y_attack"][target_indices].to(torch.int64)
            source_count = (
                payload["y_source"][target_indices]
                .to(torch.int64)
                .sum(dim=1)
            )

            if mode == "control":
                violation_counts["control_nonzero_count"] += int(
                    (count != 0).sum().item()
                )
                violation_counts["control_positive_graph"] += int(
                    (graph != 0).sum().item()
                )
                violation_counts["control_positive_source_count"] += int(
                    (source_count != 0).sum().item()
                )
                category_control_windows[category] += window_count
            else:
                active = graph == 1
                inactive = ~active

                category_active_windows[category] += int(
                    active.sum().item()
                )
                category_inactive_attack_windows[category] += int(
                    inactive.sum().item()
                )

                violation_counts[
                    "attack_graph_count_activity_mismatch"
                ] += int(
                    (
                        active != (count > 0)
                    ).sum().item()
                )
                violation_counts[
                    "attack_count_source_count_mismatch"
                ] += int(
                    (
                        count != source_count
                    ).sum().item()
                )
                violation_counts[
                    "attack_active_count_not_equal_category"
                ] += int(
                    (
                        count[active] != category
                    ).sum().item()
                )
                violation_counts[
                    "attack_inactive_count_nonzero"
                ] += int(
                    (
                        count[inactive] != 0
                    ).sum().item()
                )

                for value, frequency in zip(
                    *torch.unique(
                        count[active],
                        return_counts=True,
                    )
                ):
                    integer_value = int(value.item())
                    integer_frequency = int(frequency.item())
                    active_count_distribution[
                        integer_value
                    ] += integer_frequency
                    category_active_count_distribution[
                        category
                    ][integer_value] += integer_frequency
                    split_active_count_distribution[
                        split
                    ][integer_value] += integer_frequency

                pair_example.update(
                    {
                        "active_window_count": int(
                            active.sum().item()
                        ),
                        "inactive_attack_window_count": int(
                            inactive.sum().item()
                        ),
                        "unique_active_counts": sorted(
                            int(value)
                            for value in torch.unique(
                                count[active]
                            ).tolist()
                        ),
                    }
                )

        first_examples.setdefault(category, pair_example)

        if row_index % 100 == 0 or row_index == len(rows):
            print(
                f"audited {row_index}/{len(rows)} non-test pairs"
            )

    expected_split_pairs = {"train": 415, "validation": 74}
    expected_split_runs = {"train": 830, "validation": 148}
    expected_split_items = {"train": 70166, "validation": 12528}

    for split, expected in expected_split_pairs.items():
        if inspected_pairs[split] != expected:
            failures.append(
                f"{split} inspected pairs={inspected_pairs[split]}, "
                f"expected {expected}"
            )
    for split, expected in expected_split_runs.items():
        if inspected_runs[split] != expected:
            failures.append(
                f"{split} inspected runs={inspected_runs[split]}, "
                f"expected {expected}"
            )
    for split, expected in expected_split_items.items():
        if inspected_aligned_items[split] != expected:
            failures.append(
                f"{split} inspected aligned items="
                f"{inspected_aligned_items[split]}, expected {expected}"
            )

    for key, count in violation_counts.items():
        if count != 0:
            failures.append(f"semantic violation {key}={count}")

    observed_categories = tuple(sorted(category_pair_counts))
    observed_active_counts = tuple(sorted(active_count_distribution))

    if observed_categories != EXPECTED_CATEGORIES:
        failures.append(
            f"observed categories={observed_categories}, "
            f"expected {EXPECTED_CATEGORIES}"
        )
    if observed_active_counts != EXPECTED_ACTIVE_COUNTS:
        failures.append(
            f"observed active counts={observed_active_counts}, "
            f"expected {EXPECTED_ACTIVE_COUNTS}"
        )

    for category in EXPECTED_CATEGORIES:
        observed = tuple(
            sorted(category_active_count_distribution[category])
        )
        if observed != (category,):
            failures.append(
                f"K{category} active counts={observed}, "
                f"expected ({category},)"
            )
        if category_active_windows[category] <= 0:
            failures.append(f"K{category} has no active aligned windows")

    required_count_logits = len(observed_active_counts)
    count_head_compatible = current_count_logits == required_count_logits
    parameter_delta = (
        (required_count_logits - current_count_logits)
        * (64 + 1)
    )
    projected_parameter_count = current_parameter_count + parameter_delta

    if (
        observed_active_counts == EXPECTED_ACTIVE_COUNTS
        and current_count_logits == 3
    ):
        decision = (
            "REQUIRE_P2_COUNT_HEAD_EXPANSION_3_TO_4_"
            "THEN_REPEAT_A4_R1_AND_B0_R2"
        )
        next_stage = (
            "V5_P2_B0_R2_COUNT_HEAD_EXPANSION_AND_A4_REPEAT"
        )
        status = "COMPLETE"
    elif count_head_compatible and not failures:
        decision = "CURRENT_COUNT_HEAD_COMPATIBLE"
        next_stage = "V5_P2_B0_R2_CORRECTED_SHORTCUT_AUDIT"
        status = "COMPLETE"
    else:
        decision = "REQUIRE_MANUAL_COUNT_CONTRACT_REVIEW"
        next_stage = None
        status = "HOLD"

    if projected_parameter_count != EXPECTED_EXPANDED_PARAMETER_COUNT:
        failures.append(
            f"projected four-logit parameter count="
            f"{projected_parameter_count}, expected "
            f"{EXPECTED_EXPANDED_PARAMETER_COUNT}"
        )

    diagnosis = {
        "observed_pair_categories": list(observed_categories),
        "observed_active_attacker_counts": list(
            observed_active_counts
        ),
        "category_pair_counts": {
            str(key): int(value)
            for key, value in sorted(category_pair_counts.items())
        },
        "category_split_pair_counts": {
            str(category): {
                split: int(count)
                for split, count in sorted(counts.items())
            }
            for category, counts in sorted(
                category_split_pair_counts.items()
            )
        },
        "category_active_count_distribution": {
            str(category): {
                str(count): int(frequency)
                for count, frequency in sorted(distribution.items())
            }
            for category, distribution in sorted(
                category_active_count_distribution.items()
            )
        },
        "split_active_count_distribution": {
            split: {
                str(count): int(frequency)
                for count, frequency in sorted(distribution.items())
            }
            for split, distribution in sorted(
                split_active_count_distribution.items()
            )
        },
        "active_count_distribution": {
            str(key): int(value)
            for key, value in sorted(active_count_distribution.items())
        },
        "category_active_windows": {
            str(key): int(value)
            for key, value in sorted(category_active_windows.items())
        },
        "category_inactive_attack_windows": {
            str(key): int(value)
            for key, value in sorted(
                category_inactive_attack_windows.items()
            )
        },
        "category_control_windows": {
            str(key): int(value)
            for key, value in sorted(category_control_windows.items())
        },
        "first_example_by_category": {
            str(key): value
            for key, value in sorted(first_examples.items())
        },
        "semantic_violation_counts": {
            key: int(value)
            for key, value in sorted(violation_counts.items())
        },
        "current_model": {
            "architecture": model.architecture_name,
            "parameter_count": current_parameter_count,
            "count_head_in_features": int(
                model.count_head.in_features
            ),
            "count_head_out_features": current_count_logits,
            "historical_a4_positive_count_coverage": (
                a4_lock.get("positive_raw_attacker_counts")
            ),
        },
        "required_p2_count_contract": {
            "graph_head_represents_zero_count": True,
            "count_head_active_only": True,
            "active_count_values": list(observed_active_counts),
            "required_count_logits": required_count_logits,
            "class_mapping": {
                str(raw_count): class_index
                for class_index, raw_count in enumerate(
                    observed_active_counts
                )
            },
            "count_head_compatible": count_head_compatible,
            "required_output_change": (
                "Linear(64,3) -> Linear(64,4)"
                if required_count_logits == 4
                else None
            ),
            "parameter_delta": parameter_delta,
            "projected_parameter_count": (
                projected_parameter_count
            ),
            "temporal_encoder_change_required": False,
            "role_head_change_required": False,
            "graph_head_change_required": False,
        },
        "historical_a4_interpretation": {
            "integration_pass_still_valid_for": [
                "loader collation",
                "PRIMARY58 tensor path",
                "physical mask path",
                "graph head",
                "source head",
                "transit head",
                "victim head",
                "path head",
                "gradient and optimizer plumbing",
            ],
            "integration_pass_incomplete_for": [
                "full attacker-count class coverage",
                "K3 count-head behavior",
            ],
            "audit_weights_saved": False,
        },
    }

    diagnosis_path = (
        output_dir
        / "V5_P2_B0_R1_K3_COUNT_SEMANTICS_DIAGNOSIS.json"
    )
    write_json(diagnosis_path, diagnosis)

    report = {
        "stage": STAGE,
        "status": "HOLD" if failures else status,
        "decision": (
            "REQUIRE_MANUAL_COUNT_CONTRACT_REVIEW"
            if failures else decision
        ),
        "dataset": {
            "link_path": str(root_input.absolute()),
            "resolved_root": str(root),
        },
        "coverage": {
            "train_pairs": int(inspected_pairs["train"]),
            "validation_pairs": int(
                inspected_pairs["validation"]
            ),
            "total_pairs": int(sum(inspected_pairs.values())),
            "train_runs": int(inspected_runs["train"]),
            "validation_runs": int(inspected_runs["validation"]),
            "total_runs": int(sum(inspected_runs.values())),
            "train_aligned_items": int(
                inspected_aligned_items["train"]
            ),
            "validation_aligned_items": int(
                inspected_aligned_items["validation"]
            ),
            "total_aligned_items": int(
                sum(inspected_aligned_items.values())
            ),
        },
        "diagnosis": diagnosis,
        "historical_b0_hold": {
            "failure_count": len(b0_failures),
            "failure_class": (
                "audit omitted valid K3 category"
            ),
            "report_sha256": sha256_file(paths["b0_report"]),
            "hold_marker_sha256": sha256_file(paths["b0_hold"]),
        },
        "artifacts": {
            "diagnosis_json": [
                str(diagnosis_path),
                sha256_file(diagnosis_path),
            ]
        },
        "provenance": {
            "a1_r2_report_sha256": sha256_file(
                paths["a1_r2_report"]
            ),
            "a1_r2_lock_sha256": sha256_file(
                paths["a1_r2_lock"]
            ),
            "pair_manifest_sha256": sha256_file(
                paths["pair_manifest"]
            ),
            "a3_report_sha256": sha256_file(
                paths["a3_report"]
            ),
            "a3_lock_sha256": sha256_file(paths["a3_lock"]),
            "a4_report_sha256": sha256_file(
                paths["a4_report"]
            ),
            "a4_lock_sha256": sha256_file(paths["a4_lock"]),
            "historical_b0_report_sha256": sha256_file(
                paths["b0_report"]
            ),
            "model_sha256": sha256_file(paths["model"]),
        },
        "security_boundary": {
            "train_tensor_contents_accessed": True,
            "validation_tensor_contents_accessed": True,
            "test_directory_existence_checked": False,
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
        "next_stage": None if failures else next_stage,
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    if failures or status == "HOLD":
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print("===== V5 P2-B0-R1 FINAL =====")
        print("status: HOLD")
        print("decision:", report["decision"])
        print("failure_count:", len(failures))
        for failure in failures:
            print("FAIL:", failure)
        print("warning_count:", len(warnings))
        print(f"{STAGE}_HOLD")
        return 1

    lock = {
        "status": COMPLETE,
        "decision": decision,
        "report_sha256": sha256_file(report_path),
        "diagnosis_sha256": sha256_file(diagnosis_path),
        "total_pairs": int(sum(inspected_pairs.values())),
        "total_runs": int(sum(inspected_runs.values())),
        "total_aligned_items": int(
            sum(inspected_aligned_items.values())
        ),
        "observed_categories": list(observed_categories),
        "observed_active_counts": list(observed_active_counts),
        "current_count_logits": current_count_logits,
        "required_count_logits": required_count_logits,
        "current_parameter_count": current_parameter_count,
        "projected_parameter_count": projected_parameter_count,
        "parameter_delta": parameter_delta,
        "temporal_encoder_change_required": False,
        "other_head_change_required": False,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "next_stage": next_stage,
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-B0-R1 FINAL =====")
    print("status: COMPLETE")
    print("decision:", decision)
    print("train_pairs:", inspected_pairs["train"])
    print("validation_pairs:", inspected_pairs["validation"])
    print("total_pairs:", sum(inspected_pairs.values()))
    print("total_runs:", sum(inspected_runs.values()))
    print("total_aligned_items:", sum(inspected_aligned_items.values()))
    print("observed_categories:", list(observed_categories))
    print("observed_active_counts:", list(observed_active_counts))
    for category in EXPECTED_CATEGORIES:
        print(
            f"K{category}_pair_count:",
            category_pair_counts[category],
        )
        print(
            f"K{category}_active_count_distribution:",
            dict(category_active_count_distribution[category]),
        )
    print("semantic_violation_count:", sum(violation_counts.values()))
    print("current_count_logits:", current_count_logits)
    print("required_count_logits:", required_count_logits)
    print("current_parameter_count:", current_parameter_count)
    print("parameter_delta:", parameter_delta)
    print("projected_parameter_count:", projected_parameter_count)
    print("temporal_encoder_change_required: false")
    print("other_head_change_required: false")
    print("historical_a4_full_count_coverage: false")
    print("historical_a4_other_integration_checks_valid: true")
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("next_stage:", next_stage)
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
