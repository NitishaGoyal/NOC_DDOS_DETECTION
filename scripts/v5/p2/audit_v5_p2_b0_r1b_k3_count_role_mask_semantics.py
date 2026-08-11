#!/usr/bin/env python3
"""
V5 P2-B0-R1B K3 Count and Role-Mask Semantics Audit

Historical B0 exposed two audit assumptions:
1. it recognized K1/K2/K4 but omitted valid K3 pairs;
2. it assumed role_mask was binary, although role_mask may be a categorical
   or bit-field encoding.

This stage audits every aligned TRAIN/VALIDATION target and determines:
- complete K-category and attacker-count semantics;
- current three-logit count-head compatibility;
- exact role_mask value distribution;
- whether role_mask equals a binary union, role count, or an exact bit-field
  encoding of source/transit/victim[/path].

It preserves all prior HOLD reports, modifies no tensors or models, and never
enumerates or opens runs/test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import torch


STAGE = "V5_P2_B0_R1B_K3_COUNT_AND_ROLE_MASK_SEMANTICS_AUDIT"
COMPLETE = f"{STAGE}_COMPLETE"

WINDOW = 32
STRIDE = 8

EXPECTED_CATEGORIES = (1, 2, 3, 4)
EXPECTED_ACTIVE_COUNTS = (1, 2, 3, 4)

CURRENT_COUNT_LOGITS = 3
CURRENT_PARAMETER_COUNT = 43_208
EXPANDED_PARAMETER_COUNT = 43_273

EXPECTED_DIRECT_K3_FAILURES = 115
EXPECTED_HISTORICAL_FAILURES = 122

EXPECTED_COVERAGE_FAILURES = {
    "train processed pair count=318",
    "train processed run count=636",
    "train processed aligned items=53878, expected 70166",
    "validation processed pair count=56",
    "validation processed run count=112",
    "validation processed aligned items=9496, expected 12528",
}
EXPECTED_ROLE_MASK_FAILURE = (
    "label integrity violation nonbinary_role_mask=374"
)


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(chunk_size),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
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
        "v5_frozen_b3_role_mask_audit",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import model from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def certifies_test_untouched(report: dict[str, Any]) -> bool:
    """
    Historical early-stop audit reports used two schemas:
    - full reports: security_boundary.test_tensor_contents_accessed
    - prerequisite HOLD reports: top-level test_tensor_contents_accessed

    Accept either only when the value is explicitly False.
    """
    security_boundary = report.get("security_boundary")
    if (
        isinstance(security_boundary, dict)
        and "test_tensor_contents_accessed" in security_boundary
    ):
        return (
            security_boundary["test_tensor_contents_accessed"]
            is False
        )

    return report.get("test_tensor_contents_accessed") is False


def binary_tensor(value: torch.Tensor) -> bool:
    if value.numel() == 0:
        return True
    return bool(
        ((value == 0) | (value == 1)).all().item()
    )


def build_role_candidates() -> dict[
    str,
    Callable[
        [
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        torch.Tensor,
    ],
]:
    candidates: dict[
        str,
        Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
            torch.Tensor,
        ],
    ] = {}

    candidates[
        "binary_union_source_transit_victim"
    ] = (
        lambda source, transit, victim, path:
        (
            (source > 0)
            | (transit > 0)
            | (victim > 0)
        ).to(torch.int64)
    )

    candidates[
        "binary_path"
    ] = (
        lambda source, transit, victim, path:
        (path > 0).to(torch.int64)
    )

    candidates[
        "binary_union_source_transit_victim_path"
    ] = (
        lambda source, transit, victim, path:
        (
            (source > 0)
            | (transit > 0)
            | (victim > 0)
            | (path > 0)
        ).to(torch.int64)
    )

    candidates[
        "role_count_source_transit_victim"
    ] = (
        lambda source, transit, victim, path:
        (
            (source > 0).to(torch.int64)
            + (transit > 0).to(torch.int64)
            + (victim > 0).to(torch.int64)
        )
    )

    candidates[
        "role_count_source_transit_victim_path"
    ] = (
        lambda source, transit, victim, path:
        (
            (source > 0).to(torch.int64)
            + (transit > 0).to(torch.int64)
            + (victim > 0).to(torch.int64)
            + (path > 0).to(torch.int64)
        )
    )

    three_roles = ("source", "transit", "victim")
    for permutation in itertools.permutations(
        (1, 2, 4),
    ):
        mapping = dict(zip(three_roles, permutation))
        name = (
            "bitfield3_"
            f"source{mapping['source']}_"
            f"transit{mapping['transit']}_"
            f"victim{mapping['victim']}"
        )

        def make_three(
            local_mapping: dict[str, int],
        ):
            return (
                lambda source, transit, victim, path:
                (
                    (source > 0).to(torch.int64)
                    * local_mapping["source"]
                    + (transit > 0).to(torch.int64)
                    * local_mapping["transit"]
                    + (victim > 0).to(torch.int64)
                    * local_mapping["victim"]
                )
            )

        candidates[name] = make_three(mapping)

    four_roles = (
        "source",
        "transit",
        "victim",
        "path",
    )
    for permutation in itertools.permutations(
        (1, 2, 4, 8),
    ):
        mapping = dict(zip(four_roles, permutation))
        name = (
            "bitfield4_"
            f"source{mapping['source']}_"
            f"transit{mapping['transit']}_"
            f"victim{mapping['victim']}_"
            f"path{mapping['path']}"
        )

        def make_four(
            local_mapping: dict[str, int],
        ):
            return (
                lambda source, transit, victim, path:
                (
                    (source > 0).to(torch.int64)
                    * local_mapping["source"]
                    + (transit > 0).to(torch.int64)
                    * local_mapping["transit"]
                    + (victim > 0).to(torch.int64)
                    * local_mapping["victim"]
                    + (path > 0).to(torch.int64)
                    * local_mapping["path"]
                )
            )

        candidates[name] = make_four(mapping)

    return candidates


def candidate_family(name: str) -> str:
    if name.startswith("bitfield3_"):
        return "bitfield3_source_transit_victim"
    if name.startswith("bitfield4_"):
        return "bitfield4_source_transit_victim_path"
    if name.startswith("binary_"):
        return "binary"
    if name.startswith("role_count_"):
        return "role_count"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--a1-r2-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--a3-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--a4-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--b0-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--b0-r1-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--b0-r1a-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    root_input = args.root.expanduser()
    root = root_input.resolve()
    a1_r2_dir = args.a1_r2_dir.expanduser().resolve()
    a3_dir = args.a3_dir.expanduser().resolve()
    a4_dir = args.a4_dir.expanduser().resolve()
    b0_dir = args.b0_dir.expanduser().resolve()
    b0_r1_dir = args.b0_r1_dir.expanduser().resolve()
    b0_r1a_dir = args.b0_r1a_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
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
        "b0_r1_report": (
            b0_r1_dir
            / "V5_P2_B0_R1_K3_COUNT_SEMANTICS_AND_HEAD_COMPATIBILITY_AUDIT.json"
        ),
        "b0_r1_hold": (
            b0_r1_dir
            / "V5_P2_B0_R1_K3_COUNT_SEMANTICS_AND_HEAD_COMPATIBILITY_AUDIT_HOLD"
        ),
        "b0_r1a_report": (
            b0_r1a_dir
            / "V5_P2_B0_R1A_K3_COUNT_AND_ROLE_MASK_SEMANTICS_AUDIT.json"
        ),
        "b0_r1a_hold": (
            b0_r1a_dir
            / "V5_P2_B0_R1A_K3_COUNT_AND_ROLE_MASK_SEMANTICS_AUDIT_HOLD"
        ),
        "model": model_path,
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(
                f"missing prerequisite {name}: {path}"
            )

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
        write_json(
            output_dir / f"{STAGE}.json",
            report,
        )
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        return 1

    a1_report = load_json(paths["a1_r2_report"])
    a1_lock = load_json(paths["a1_r2_lock"])
    a3_report = load_json(paths["a3_report"])
    a3_lock = load_json(paths["a3_lock"])
    a4_report = load_json(paths["a4_report"])
    a4_lock = load_json(paths["a4_lock"])
    b0_report = load_json(paths["b0_report"])
    b0_r1_report = load_json(paths["b0_r1_report"])
    b0_r1a_report = load_json(paths["b0_r1a_report"])

    for label, report, lock, report_path in (
        (
            "A1-R2",
            a1_report,
            a1_lock,
            paths["a1_r2_report"],
        ),
        (
            "A3",
            a3_report,
            a3_lock,
            paths["a3_report"],
        ),
        (
            "A4",
            a4_report,
            a4_lock,
            paths["a4_report"],
        ),
    ):
        if report.get("status") != "COMPLETE":
            failures.append(
                f"{label} status is not COMPLETE"
            )
        if (
            lock.get("report_sha256")
            != sha256_file(report_path)
        ):
            failures.append(
                f"{label} report SHA mismatch"
            )
        if (
            report.get(
                "security_boundary",
                {},
            ).get(
                "test_tensor_contents_accessed"
            )
            is not False
        ):
            failures.append(
                f"{label} does not certify untouched test"
            )

    if b0_report.get("status") != "HOLD":
        failures.append(
            "historical B0 status is not HOLD"
        )
    if b0_r1_report.get("status") != "HOLD":
        failures.append(
            "historical B0-R1 status is not HOLD"
        )
    if b0_r1a_report.get("status") != "HOLD":
        failures.append(
            "historical B0-R1A status is not HOLD"
        )

    historical_failures = b0_report.get(
        "failures",
        [],
    )
    if (
        len(historical_failures)
        != EXPECTED_HISTORICAL_FAILURES
    ):
        failures.append(
            "historical B0 failure count="
            f"{len(historical_failures)}, expected "
            f"{EXPECTED_HISTORICAL_FAILURES}"
        )

    direct_k3 = [
        item
        for item in historical_failures
        if "cannot infer K1/K2/K4" in item
    ]
    coverage = {
        item
        for item in historical_failures
        if item in EXPECTED_COVERAGE_FAILURES
    }
    role_mask_failures = [
        item
        for item in historical_failures
        if item == EXPECTED_ROLE_MASK_FAILURE
    ]
    classified = (
        set(direct_k3)
        | coverage
        | set(role_mask_failures)
    )
    unclassified = [
        item
        for item in historical_failures
        if item not in classified
    ]

    if len(direct_k3) != EXPECTED_DIRECT_K3_FAILURES:
        failures.append(
            f"historical direct K3 failures="
            f"{len(direct_k3)}, expected "
            f"{EXPECTED_DIRECT_K3_FAILURES}"
        )
    if coverage != EXPECTED_COVERAGE_FAILURES:
        failures.append(
            "historical derivative coverage failures differ "
            "from the observed six-item set"
        )
    if len(role_mask_failures) != 1:
        failures.append(
            "historical nonbinary role-mask failure missing "
            "or duplicated"
        )
    if unclassified:
        failures.append(
            "historical B0 contains unclassified failures: "
            f"{unclassified[:20]}"
        )

    b0_r1_failures = b0_r1_report.get(
        "failures",
        [],
    )
    if b0_r1_failures != [
        (
            "historical B0 HOLD contains failures "
            "unrelated to K3"
        )
    ]:
        failures.append(
            "historical B0-R1 failure is not the expected "
            "overly strict prerequisite classification"
        )

    b0_r1a_failures = b0_r1a_report.get(
        "failures",
        [],
    )
    if b0_r1a_failures != [
        (
            "historical B0-R1 does not certify "
            "untouched test"
        )
    ]:
        failures.append(
            "historical B0-R1A failure is not the expected "
            "report-schema mismatch"
        )

    if not certifies_test_untouched(b0_report):
        failures.append(
            "historical B0 does not certify untouched test"
        )
    if not certifies_test_untouched(b0_r1_report):
        failures.append(
            "historical B0-R1 does not certify untouched test"
        )
    if not certifies_test_untouched(b0_r1a_report):
        failures.append(
            "historical B0-R1A does not certify untouched test"
        )

    if (
        False
        and b0_report.get(
            "security_boundary",
            {},
        ).get(
            "test_tensor_contents_accessed"
        )
        is not False
    ):
        failures.append(
            "historical B0 does not certify untouched test"
        )
    if (
        False
        and b0_r1_report.get(
            "security_boundary",
            {},
        ).get(
            "test_tensor_contents_accessed"
        )
        is not False
    ):
        failures.append(
            "historical B0-R1 does not certify untouched test"
        )

    if (
        a1_lock.get("pair_manifest_sha256")
        != sha256_file(paths["pair_manifest"])
    ):
        failures.append(
            "A1-R2 pair-manifest SHA mismatch"
        )
    if a1_lock.get("pair_count") != 489:
        failures.append(
            "A1-R2 pair count changed"
        )
    if a1_lock.get("aligned_windows") != 82694:
        failures.append(
            "A1-R2 aligned-window count changed"
        )
    if a3_lock.get("total_aligned_items") != 82694:
        failures.append(
            "A3 aligned-item count changed"
        )
    if a4_lock.get(
        "positive_raw_attacker_counts"
    ) != [1, 2, 4]:
        failures.append(
            "historical A4 count coverage is not [1,2,4]"
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
        write_json(
            output_dir / f"{STAGE}.json",
            report,
        )
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    model_module = import_model(model_path)
    model = model_module.FrozenB3Conv1DOnly()

    current_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    current_count_logits = int(
        model.count_head.out_features
    )

    if (
        current_parameter_count
        != CURRENT_PARAMETER_COUNT
    ):
        failures.append(
            "current model parameter count="
            f"{current_parameter_count}, expected "
            f"{CURRENT_PARAMETER_COUNT}"
        )
    if current_count_logits != CURRENT_COUNT_LOGITS:
        failures.append(
            "current count logits="
            f"{current_count_logits}, expected "
            f"{CURRENT_COUNT_LOGITS}"
        )

    role_candidates = build_role_candidates()
    candidate_mismatches = Counter()
    candidate_compared_entries = 0

    role_mask_values = Counter()
    role_mask_values_by_mode = {
        "attack_active": Counter(),
        "attack_inactive": Counter(),
        "control": Counter(),
    }

    category_pair_counts = Counter()
    category_split_pair_counts = {
        category: Counter()
        for category in EXPECTED_CATEGORIES
    }
    active_count_distribution = Counter()
    active_count_by_category = {
        category: Counter()
        for category in EXPECTED_CATEGORIES
    }
    active_count_by_split = {
        "train": Counter(),
        "validation": Counter(),
    }

    semantic_violations = Counter()
    processed_pairs = Counter()
    processed_runs = Counter()
    processed_items = Counter()

    first_example_by_category: dict[
        int,
        dict[str, Any],
    ] = {}

    rows = load_manifest(paths["pair_manifest"])

    print(
        "===== V5 P2-B0-R1B K3 COUNT + ROLE-MASK AUDIT ====="
    )
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")

    for row_index, row in enumerate(
        rows,
        start=1,
    ):
        split = row.get("split", "")
        pair_key = row.get("pair_key", "")
        category = infer_k(pair_key)

        if split not in {
            "train",
            "validation",
        }:
            failures.append(
                f"unexpected split {split!r} for {pair_key}"
            )
            continue

        if category not in EXPECTED_CATEGORIES:
            failures.append(
                f"{split}/{pair_key}: "
                "cannot infer K1/K2/K3/K4"
            )
            continue

        common_length = as_int(
            row,
            "common_length",
        )
        window_count = as_int(
            row,
            "window_count_per_member",
        )
        target_indices = torch.arange(
            WINDOW - 1,
            common_length,
            STRIDE,
            dtype=torch.int64,
        )

        if int(target_indices.numel()) != window_count:
            failures.append(
                f"{split}/{pair_key}: target count mismatch"
            )
            continue

        processed_pairs[split] += 1
        category_pair_counts[category] += 1
        category_split_pair_counts[
            category
        ][split] += 1

        example = {
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
                failures.append(
                    f"missing run tensor: {path}"
                )
                continue

            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            processed_runs[split] += 1
            processed_items[split] += window_count

            if not isinstance(payload, dict):
                failures.append(
                    f"{split}/{path.name}: payload is not dict"
                )
                continue

            graph = payload[
                "y_attack"
            ][target_indices].to(torch.int64)
            count = payload[
                "y_attacker_count"
            ][target_indices].to(torch.int64)
            source = payload[
                "y_source"
            ][target_indices].to(torch.int64)
            transit = payload[
                "y_transit"
            ][target_indices].to(torch.int64)
            victim = payload[
                "y_victim"
            ][target_indices].to(torch.int64)
            path_label = payload[
                "y_attack_path"
            ][target_indices].to(torch.int64)
            role_mask = payload[
                "role_mask"
            ][target_indices].to(torch.int64)

            for logical_name, tensor in (
                ("graph", graph),
                ("source", source),
                ("transit", transit),
                ("victim", victim),
                ("path", path_label),
            ):
                if not binary_tensor(tensor):
                    semantic_violations[
                        f"nonbinary_{logical_name}"
                    ] += 1

            if bool((role_mask < 0).any().item()):
                semantic_violations[
                    "negative_role_mask_entries"
                ] += int(
                    (role_mask < 0).sum().item()
                )

            unique_values, frequencies = torch.unique(
                role_mask,
                return_counts=True,
            )
            for value, frequency in zip(
                unique_values.tolist(),
                frequencies.tolist(),
            ):
                role_mask_values[int(value)] += int(
                    frequency
                )

            active = graph == 1
            inactive = ~active

            if mode == "attack":
                for value, frequency in zip(
                    *torch.unique(
                        role_mask[active],
                        return_counts=True,
                    )
                ):
                    role_mask_values_by_mode[
                        "attack_active"
                    ][int(value.item())] += int(
                        frequency.item()
                    )

                for value, frequency in zip(
                    *torch.unique(
                        role_mask[inactive],
                        return_counts=True,
                    )
                ):
                    role_mask_values_by_mode[
                        "attack_inactive"
                    ][int(value.item())] += int(
                        frequency.item()
                    )
            else:
                for value, frequency in zip(
                    unique_values.tolist(),
                    frequencies.tolist(),
                ):
                    role_mask_values_by_mode[
                        "control"
                    ][int(value)] += int(
                        frequency
                    )

            source_count = source.sum(dim=1)

            if mode == "control":
                semantic_violations[
                    "control_positive_graph"
                ] += int(
                    (graph != 0).sum().item()
                )
                semantic_violations[
                    "control_nonzero_count"
                ] += int(
                    (count != 0).sum().item()
                )
                semantic_violations[
                    "control_positive_source"
                ] += int(
                    source.sum().item()
                )
                semantic_violations[
                    "control_positive_transit"
                ] += int(
                    transit.sum().item()
                )
                semantic_violations[
                    "control_positive_victim"
                ] += int(
                    victim.sum().item()
                )
                semantic_violations[
                    "control_positive_path"
                ] += int(
                    path_label.sum().item()
                )
            else:
                semantic_violations[
                    "attack_graph_count_activity_mismatch"
                ] += int(
                    (
                        active
                        != (count > 0)
                    ).sum().item()
                )
                semantic_violations[
                    "attack_count_source_count_mismatch"
                ] += int(
                    (
                        count != source_count
                    ).sum().item()
                )
                semantic_violations[
                    "attack_active_count_not_equal_category"
                ] += int(
                    (
                        count[active]
                        != category
                    ).sum().item()
                )
                semantic_violations[
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
                    raw_count = int(value.item())
                    raw_frequency = int(
                        frequency.item()
                    )
                    active_count_distribution[
                        raw_count
                    ] += raw_frequency
                    active_count_by_category[
                        category
                    ][raw_count] += raw_frequency
                    active_count_by_split[
                        split
                    ][raw_count] += raw_frequency

                example.update(
                    {
                        "active_window_count": int(
                            active.sum().item()
                        ),
                        "inactive_window_count": int(
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

            for candidate_name, function in (
                role_candidates.items()
            ):
                predicted = function(
                    source,
                    transit,
                    victim,
                    path_label,
                )
                candidate_mismatches[
                    candidate_name
                ] += int(
                    (
                        predicted != role_mask
                    ).sum().item()
                )

            candidate_compared_entries += int(
                role_mask.numel()
            )

        first_example_by_category.setdefault(
            category,
            example,
        )

        if (
            row_index % 100 == 0
            or row_index == len(rows)
        ):
            print(
                f"audited {row_index}/{len(rows)} "
                "non-test pairs"
            )

    expected_pairs = {
        "train": 415,
        "validation": 74,
    }
    expected_runs = {
        "train": 830,
        "validation": 148,
    }
    expected_items = {
        "train": 70166,
        "validation": 12528,
    }

    for split, expected in expected_pairs.items():
        if processed_pairs[split] != expected:
            failures.append(
                f"{split} processed pairs="
                f"{processed_pairs[split]}, expected "
                f"{expected}"
            )
    for split, expected in expected_runs.items():
        if processed_runs[split] != expected:
            failures.append(
                f"{split} processed runs="
                f"{processed_runs[split]}, expected "
                f"{expected}"
            )
    for split, expected in expected_items.items():
        if processed_items[split] != expected:
            failures.append(
                f"{split} processed aligned items="
                f"{processed_items[split]}, expected "
                f"{expected}"
            )

    for key, value in semantic_violations.items():
        if value != 0:
            failures.append(
                f"semantic violation {key}={value}"
            )

    observed_categories = tuple(
        sorted(category_pair_counts)
    )
    observed_counts = tuple(
        sorted(active_count_distribution)
    )

    if observed_categories != EXPECTED_CATEGORIES:
        failures.append(
            f"observed categories={observed_categories}, "
            f"expected {EXPECTED_CATEGORIES}"
        )
    if observed_counts != EXPECTED_ACTIVE_COUNTS:
        failures.append(
            f"observed active counts={observed_counts}, "
            f"expected {EXPECTED_ACTIVE_COUNTS}"
        )

    for category in EXPECTED_CATEGORIES:
        observed_for_category = tuple(
            sorted(active_count_by_category[category])
        )
        if observed_for_category != (category,):
            failures.append(
                f"K{category} active counts="
                f"{observed_for_category}, expected "
                f"({category},)"
            )

    candidate_rows = []
    for candidate_name in sorted(
        candidate_mismatches,
        key=lambda name: (
            candidate_mismatches[name],
            name,
        ),
    ):
        mismatches = int(
            candidate_mismatches[candidate_name]
        )
        candidate_rows.append(
            {
                "name": candidate_name,
                "family": candidate_family(
                    candidate_name
                ),
                "mismatch_count": mismatches,
                "compared_entries": (
                    candidate_compared_entries
                ),
                "mismatch_rate": (
                    mismatches
                    / max(
                        1,
                        candidate_compared_entries,
                    )
                ),
            }
        )

    zero_error_candidates = [
        row
        for row in candidate_rows
        if row["mismatch_count"] == 0
    ]

    role_mask_semantics_resolved = (
        len(zero_error_candidates) == 1
    )
    if not role_mask_semantics_resolved:
        failures.append(
            "role_mask zero-error candidate count="
            f"{len(zero_error_candidates)}, expected 1"
        )

    selected_role_semantics = (
        zero_error_candidates[0]
        if role_mask_semantics_resolved
        else (
            candidate_rows[0]
            if candidate_rows
            else None
        )
    )

    required_count_logits = len(observed_counts)
    count_head_compatible = (
        current_count_logits
        == required_count_logits
    )
    parameter_delta = (
        required_count_logits
        - current_count_logits
    ) * (64 + 1)
    projected_parameter_count = (
        current_parameter_count
        + parameter_delta
    )

    if (
        projected_parameter_count
        != EXPANDED_PARAMETER_COUNT
    ):
        failures.append(
            "projected parameter count="
            f"{projected_parameter_count}, expected "
            f"{EXPANDED_PARAMETER_COUNT}"
        )

    if not failures:
        decision = (
            "RESOLVE_B0_HOLD_AS_K3_OMISSION_AND_"
            "ROLE_MASK_SEMANTIC_MISCLASSIFICATION_"
            "REQUIRE_COUNT_HEAD_3_TO_4_AND_REPEAT_A4"
        )
        next_stage = (
            "V5_P2_B0_R2_COUNT_HEAD_EXPANSION_AND_A4_REPEAT"
        )
        status = "COMPLETE"
    else:
        decision = (
            "REQUIRE_MANUAL_K3_OR_ROLE_MASK_REVIEW"
        )
        next_stage = None
        status = "HOLD"

    diagnosis = {
        "historical_b0_resolution": {
            "direct_k3_omission_failures": (
                len(direct_k3)
            ),
            "derivative_coverage_failures": sorted(
                coverage
            ),
            "role_mask_failure": (
                EXPECTED_ROLE_MASK_FAILURE
            ),
            "interpretation": (
                "K3 omission caused the six coverage "
                "failures; role_mask binary assumption "
                "requires independent semantic resolution. "
                "B0-R1A stopped before data audit because it "
                "expected a nested security-boundary field in "
                "an early-stop report that stored the explicit "
                "false flag at top level."
            ),
        },
        "coverage": {
            "train_pairs": int(
                processed_pairs["train"]
            ),
            "validation_pairs": int(
                processed_pairs["validation"]
            ),
            "total_pairs": int(
                sum(processed_pairs.values())
            ),
            "train_runs": int(
                processed_runs["train"]
            ),
            "validation_runs": int(
                processed_runs["validation"]
            ),
            "total_runs": int(
                sum(processed_runs.values())
            ),
            "train_aligned_items": int(
                processed_items["train"]
            ),
            "validation_aligned_items": int(
                processed_items["validation"]
            ),
            "total_aligned_items": int(
                sum(processed_items.values())
            ),
        },
        "count_semantics": {
            "observed_categories": list(
                observed_categories
            ),
            "observed_active_counts": list(
                observed_counts
            ),
            "category_pair_counts": {
                str(key): int(value)
                for key, value in sorted(
                    category_pair_counts.items()
                )
            },
            "category_split_pair_counts": {
                str(category): {
                    split: int(value)
                    for split, value in sorted(
                        counter.items()
                    )
                }
                for category, counter in sorted(
                    category_split_pair_counts.items()
                )
            },
            "active_count_distribution": {
                str(key): int(value)
                for key, value in sorted(
                    active_count_distribution.items()
                )
            },
            "active_count_by_category": {
                str(category): {
                    str(count): int(value)
                    for count, value in sorted(
                        counter.items()
                    )
                }
                for category, counter in sorted(
                    active_count_by_category.items()
                )
            },
            "active_count_by_split": {
                split: {
                    str(count): int(value)
                    for count, value in sorted(
                        counter.items()
                    )
                }
                for split, counter in sorted(
                    active_count_by_split.items()
                )
            },
            "first_example_by_category": {
                str(key): value
                for key, value in sorted(
                    first_example_by_category.items()
                )
            },
        },
        "role_mask_semantics": {
            "unique_values": {
                str(key): int(value)
                for key, value in sorted(
                    role_mask_values.items()
                )
            },
            "unique_values_by_mode": {
                mode: {
                    str(key): int(value)
                    for key, value in sorted(
                        counter.items()
                    )
                }
                for mode, counter in (
                    role_mask_values_by_mode.items()
                )
            },
            "compared_entries": (
                candidate_compared_entries
            ),
            "candidate_scores": candidate_rows,
            "zero_error_candidate_count": (
                len(zero_error_candidates)
            ),
            "selected_semantics": (
                selected_role_semantics
            ),
            "binary_target_assumption_valid": (
                selected_role_semantics is not None
                and selected_role_semantics[
                    "family"
                ] == "binary"
            ),
            "role_mask_is_model_target": False,
            "role_mask_training_role": (
                "auxiliary target/loss bookkeeping only; "
                "not a learned model input"
            ),
        },
        "model_compatibility": {
            "architecture": model.architecture_name,
            "current_parameter_count": (
                current_parameter_count
            ),
            "current_count_logits": (
                current_count_logits
            ),
            "required_count_logits": (
                required_count_logits
            ),
            "count_head_compatible": (
                count_head_compatible
            ),
            "required_change": (
                "Linear(64,3) -> Linear(64,4)"
            ),
            "parameter_delta": parameter_delta,
            "projected_parameter_count": (
                projected_parameter_count
            ),
            "temporal_encoder_change_required": False,
            "graph_head_change_required": False,
            "role_head_change_required": False,
        },
        "historical_a4": {
            "still_valid_for": [
                "A3 loader integration",
                "PRIMARY58 data path",
                "physical-port mask path",
                "graph head",
                "source head",
                "transit head",
                "victim head",
                "path head",
                "gradient flow",
                "optimizer operation",
            ],
            "incomplete_for": [
                "K3 attacker-count class",
                "full four-class count-head coverage",
            ],
            "audit_weights_saved": False,
        },
        "semantic_violation_counts": {
            key: int(value)
            for key, value in sorted(
                semantic_violations.items()
            )
        },
    }

    diagnosis_path = (
        output_dir
        / "V5_P2_B0_R1B_K3_COUNT_AND_ROLE_MASK_DIAGNOSIS.json"
    )
    write_json(
        diagnosis_path,
        diagnosis,
    )

    report = {
        "stage": STAGE,
        "status": status,
        "decision": decision,
        "dataset": {
            "link_path": str(
                root_input.absolute()
            ),
            "resolved_root": str(root),
        },
        "diagnosis": diagnosis,
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
            "a3_lock_sha256": sha256_file(
                paths["a3_lock"]
            ),
            "a4_report_sha256": sha256_file(
                paths["a4_report"]
            ),
            "a4_lock_sha256": sha256_file(
                paths["a4_lock"]
            ),
            "historical_b0_report_sha256": (
                sha256_file(
                    paths["b0_report"]
                )
            ),
            "historical_b0_hold_sha256": (
                sha256_file(
                    paths["b0_hold"]
                )
            ),
            "historical_b0_r1_report_sha256": (
                sha256_file(
                    paths["b0_r1_report"]
                )
            ),
            "historical_b0_r1_hold_sha256": (
                sha256_file(
                    paths["b0_r1_hold"]
                )
            ),
            "historical_b0_r1a_report_sha256": (
                sha256_file(
                    paths["b0_r1a_report"]
                )
            ),
            "historical_b0_r1a_hold_sha256": (
                sha256_file(
                    paths["b0_r1a_hold"]
                )
            ),
            "model_sha256": sha256_file(
                paths["model"]
            ),
        },
        "artifacts": {
            "diagnosis_json": [
                str(diagnosis_path),
                sha256_file(diagnosis_path),
            ]
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
        "next_stage": next_stage,
    }

    report_path = (
        output_dir
        / f"{STAGE}.json"
    )
    write_json(
        report_path,
        report,
    )

    if status == "HOLD":
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(
            "===== V5 P2-B0-R1B FINAL ====="
        )
        print("status: HOLD")
        print("decision:", decision)
        print(
            "observed_categories:",
            list(observed_categories),
        )
        print(
            "observed_active_counts:",
            list(observed_counts),
        )
        print(
            "role_mask_unique_values:",
            dict(role_mask_values),
        )
        print(
            "role_mask_zero_error_candidate_count:",
            len(zero_error_candidates),
        )
        if selected_role_semantics:
            print(
                "best_role_mask_candidate:",
                selected_role_semantics["name"],
            )
            print(
                "best_role_mask_mismatches:",
                selected_role_semantics[
                    "mismatch_count"
                ],
            )
        print("failure_count:", len(failures))
        for failure in failures:
            print("FAIL:", failure)
        print("warning_count:", len(warnings))
        print(f"{STAGE}_HOLD")
        return 1

    lock = {
        "status": COMPLETE,
        "decision": decision,
        "report_sha256": sha256_file(
            report_path
        ),
        "diagnosis_sha256": sha256_file(
            diagnosis_path
        ),
        "total_pairs": int(
            sum(processed_pairs.values())
        ),
        "total_runs": int(
            sum(processed_runs.values())
        ),
        "total_aligned_items": int(
            sum(processed_items.values())
        ),
        "observed_categories": list(
            observed_categories
        ),
        "observed_active_counts": list(
            observed_counts
        ),
        "role_mask_unique_values": sorted(
            role_mask_values
        ),
        "role_mask_semantics": (
            selected_role_semantics["name"]
        ),
        "role_mask_zero_error_candidate_count": 1,
        "current_count_logits": (
            current_count_logits
        ),
        "required_count_logits": (
            required_count_logits
        ),
        "current_parameter_count": (
            current_parameter_count
        ),
        "parameter_delta": parameter_delta,
        "projected_parameter_count": (
            projected_parameter_count
        ),
        "temporal_encoder_change_required": False,
        "other_head_change_required": False,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "next_stage": next_stage,
        "script_sha256": sha256_file(
            Path(__file__)
        ),
    }
    write_json(
        output_dir / f"{STAGE}_LOCK.json",
        lock,
    )
    atomic_write(
        output_dir / COMPLETE,
        COMPLETE + "\n",
    )

    print(
        "===== V5 P2-B0-R1B FINAL ====="
    )
    print("status: COMPLETE")
    print("decision:", decision)
    print(
        "historical_b0_resolution: "
        "K3_omission_plus_role_mask_semantic_"
        "misclassification"
    )
    print(
        "train_pairs:",
        processed_pairs["train"],
    )
    print(
        "validation_pairs:",
        processed_pairs["validation"],
    )
    print(
        "total_pairs:",
        sum(processed_pairs.values()),
    )
    print(
        "total_runs:",
        sum(processed_runs.values()),
    )
    print(
        "total_aligned_items:",
        sum(processed_items.values()),
    )
    print(
        "observed_categories:",
        list(observed_categories),
    )
    print(
        "observed_active_counts:",
        list(observed_counts),
    )
    for category in EXPECTED_CATEGORIES:
        print(
            f"K{category}_pair_count:",
            category_pair_counts[category],
        )
        print(
            f"K{category}_active_count_distribution:",
            dict(
                active_count_by_category[
                    category
                ]
            ),
        )
    print(
        "semantic_violation_count:",
        sum(semantic_violations.values()),
    )
    print(
        "role_mask_unique_values:",
        sorted(role_mask_values),
    )
    print(
        "role_mask_zero_error_candidate_count:",
        len(zero_error_candidates),
    )
    print(
        "role_mask_semantics:",
        selected_role_semantics["name"],
    )
    print(
        "role_mask_binary_target_assumption_valid:",
        str(
            diagnosis[
                "role_mask_semantics"
            ][
                "binary_target_assumption_valid"
            ]
        ).lower(),
    )
    print(
        "current_count_logits:",
        current_count_logits,
    )
    print(
        "required_count_logits:",
        required_count_logits,
    )
    print(
        "current_parameter_count:",
        current_parameter_count,
    )
    print(
        "parameter_delta:",
        parameter_delta,
    )
    print(
        "projected_parameter_count:",
        projected_parameter_count,
    )
    print(
        "temporal_encoder_change_required: false"
    )
    print(
        "other_head_change_required: false"
    )
    print(
        "historical_a4_full_count_coverage: false"
    )
    print(
        "historical_a4_other_integration_checks_valid: true"
    )
    print(
        "test_directory_enumerated: false"
    )
    print(
        "test_tensor_contents_accessed: false"
    )
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("next_stage:", next_stage)
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
