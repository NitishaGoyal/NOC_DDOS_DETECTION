#!/usr/bin/env python3
"""
V5 P2-B0-R2 Count-Head Expansion and A4 Repeat

Purpose
-------
Freeze the minimum P2-specific architectural correction established by
B0-R1B and repeat the tiny-overfit integration audit with complete attacker
count coverage.

Architectural delta
-------------------
    Linear(64,3) -> Linear(64,4)
    parameter count 43,208 -> 43,273

Everything else remains unchanged:
- PRIMARY58 input;
- corrected topology-derived [16,10] mask;
- causal depthwise-separable Conv1D encoder;
- graph attack head;
- source/transit/victim/path heads;
- no message passing.

Tiny batch
----------
Two active matched pair-windows from each K1/K2/K3/K4 category plus each
matched CONTROL window at the identical start:

    8 active attacks + 8 controls = 16 items

The four count classes map:
    raw 1 -> class 0
    raw 2 -> class 1
    raw 3 -> class 2
    raw 4 -> class 3

role_mask is verified as the frozen bit-field:
    source + 2*transit + 4*victim

It is audit/loss bookkeeping only and is not a model input or model output.

No validation tensor is opened. No test directory is enumerated or opened.
Audit weights are not saved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


STAGE = "V5_P2_B0_R2_COUNT_HEAD_EXPANSION_AND_A4_REPEAT"
COMPLETE = f"{STAGE}_COMPLETE"

SEED = 2405
MAX_STEPS = 4000
LEARNING_RATE = 1e-2
REQUIRED_STABLE_STEPS = 20

EXPECTED_PARAMETER_COUNT = 43_273
EXPECTED_COUNT_CLASS_VALUES = [1, 2, 3, 4]

LOSS_WEIGHTS = {
    "attack": 1.0,
    "count": 0.5,
    "source": 1.0,
    "transit": 0.5,
    "victim": 0.75,
    "path": 0.5,
}


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
    atomic_write(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    import csv

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def import_module(
    path: Path,
    module_name: str,
):
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"cannot import {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def category_from_pair_key(
    pair_key: str,
) -> int | None:
    match = re.search(
        r"-K([1-4])-",
        pair_key,
    )
    return int(match.group(1)) if match else None


def positive_weight(
    target: torch.Tensor,
) -> torch.Tensor:
    target_float = target.float()
    positive = target_float.sum()
    negative = target_float.numel() - positive

    if positive.item() <= 0:
        return torch.tensor(
            1.0,
            device=target.device,
            dtype=torch.float32,
        )

    return (
        (negative / positive)
        .clamp(min=1.0, max=20.0)
        .detach()
        .to(torch.float32)
    )


def exact_binary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    prediction = torch.sigmoid(logits) >= 0.5
    truth = target >= 0.5
    equality = prediction == truth

    if equality.ndim == 1:
        exact_items = equality
    else:
        exact_items = equality.reshape(
            equality.shape[0],
            -1,
        ).all(dim=1)

    return {
        "element_accuracy": float(
            equality.float().mean().item()
        ),
        "exact_item_accuracy": float(
            exact_items.float().mean().item()
        ),
        "all_exact": bool(
            equality.all().item()
        ),
    }


def evaluate(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    count_targets: torch.Tensor,
    active_mask: torch.Tensor,
) -> dict[str, Any]:
    graph = exact_binary_metrics(
        outputs["attack_logits"],
        batch["y_attack"],
    )

    count_prediction = outputs[
        "count_logits"
    ][active_mask].argmax(dim=-1)
    count_truth = count_targets[active_mask]
    count_exact = count_prediction == count_truth

    roles = {}
    for logical, output_key, target_key in (
        (
            "source",
            "source_logits",
            "y_source",
        ),
        (
            "transit",
            "transit_logits",
            "y_transit",
        ),
        (
            "victim",
            "victim_logits",
            "y_victim",
        ),
        (
            "path",
            "path_logits",
            "y_attack_path",
        ),
    ):
        roles[logical] = exact_binary_metrics(
            outputs[output_key],
            batch[target_key],
        )

    all_exact = (
        graph["all_exact"]
        and bool(count_exact.all().item())
        and all(
            value["all_exact"]
            for value in roles.values()
        )
    )

    return {
        "graph": graph,
        "count_active_accuracy": float(
            count_exact.float().mean().item()
        ),
        "count_all_exact": bool(
            count_exact.all().item()
        ),
        "roles": roles,
        "all_tasks_all_exact": all_exact,
    }


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
        "--b0-r1b-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--loader-path",
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
    b0_r1b_dir = (
        args.b0_r1b_dir.expanduser().resolve()
    )
    loader_path = (
        args.loader_path.expanduser().resolve()
    )
    model_path = (
        args.model_path.expanduser().resolve()
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
    )

    if output_dir.exists():
        print(
            f"STOP: output already exists: "
            f"{output_dir}",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    paths = {
        "a1_r2_report": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT.json"
        ),
        "a1_r2_lock": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT_LOCK.json"
        ),
        "pair_manifest": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
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
        "b0_r1b_report": (
            b0_r1b_dir
            / "V5_P2_B0_R1B_K3_COUNT_AND_ROLE_MASK_SEMANTICS_AUDIT.json"
        ),
        "b0_r1b_lock": (
            b0_r1b_dir
            / "V5_P2_B0_R1B_K3_COUNT_AND_ROLE_MASK_SEMANTICS_AUDIT_LOCK.json"
        ),
        "loader": loader_path,
        "model": model_path,
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(
                f"missing prerequisite {name}: {path}"
            )

    if not (
        root
        / "runs"
        / "train"
    ).is_dir():
        failures.append(
            "missing runs/train"
        )

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "training_performed": False,
            "validation_tensor_contents_accessed": False,
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

    a1_report = load_json(
        paths["a1_r2_report"]
    )
    a1_lock = load_json(
        paths["a1_r2_lock"]
    )
    a3_report = load_json(
        paths["a3_report"]
    )
    a3_lock = load_json(
        paths["a3_lock"]
    )
    a4_report = load_json(
        paths["a4_report"]
    )
    a4_lock = load_json(
        paths["a4_lock"]
    )
    b0_r1b_report = load_json(
        paths["b0_r1b_report"]
    )
    b0_r1b_lock = load_json(
        paths["b0_r1b_lock"]
    )

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
        (
            "B0-R1B",
            b0_r1b_report,
            b0_r1b_lock,
            paths["b0_r1b_report"],
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

    expected_b0_r1b_decision = (
        "RESOLVE_B0_HOLD_AS_K3_OMISSION_AND_"
        "ROLE_MASK_SEMANTIC_MISCLASSIFICATION_"
        "REQUIRE_COUNT_HEAD_3_TO_4_AND_REPEAT_A4"
    )
    if (
        b0_r1b_lock.get("decision")
        != expected_b0_r1b_decision
    ):
        failures.append(
            "B0-R1B did not authorize count-head "
            "expansion and A4 repeat"
        )
    if b0_r1b_lock.get(
        "observed_categories"
    ) != [1, 2, 3, 4]:
        failures.append(
            "B0-R1B observed categories changed"
        )
    if b0_r1b_lock.get(
        "observed_active_counts"
    ) != [1, 2, 3, 4]:
        failures.append(
            "B0-R1B observed active counts changed"
        )
    if b0_r1b_lock.get(
        "role_mask_semantics"
    ) != "bitfield3_source1_transit2_victim4":
        failures.append(
            "B0-R1B role_mask semantics changed"
        )
    if b0_r1b_lock.get(
        "current_count_logits"
    ) != 3:
        failures.append(
            "B0-R1B current count logits changed"
        )
    if b0_r1b_lock.get(
        "required_count_logits"
    ) != 4:
        failures.append(
            "B0-R1B required count logits changed"
        )
    if b0_r1b_lock.get(
        "projected_parameter_count"
    ) != EXPECTED_PARAMETER_COUNT:
        failures.append(
            "B0-R1B projected parameter count changed"
        )

    if (
        a1_lock.get("pair_manifest_sha256")
        != sha256_file(paths["pair_manifest"])
    ):
        failures.append(
            "A1-R2 pair-manifest SHA mismatch"
        )
    if (
        a3_lock.get("loader_sha256")
        != sha256_file(paths["loader"])
    ):
        failures.append(
            "A3 loader SHA mismatch"
        )
    if a3_lock.get(
        "test_constructor_rejected"
    ) is not True:
        failures.append(
            "A3 test-constructor rejection changed"
        )
    if a4_lock.get(
        "audit_weights_saved"
    ) is not False:
        failures.append(
            "historical A4 unexpectedly saved weights"
        )

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "training_performed": False,
            "validation_tensor_contents_accessed": False,
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

    set_seed(SEED)

    loader_module = import_module(
        loader_path,
        "v5_p2_pair_aligned_loader_b0_r2",
    )
    model_module = import_module(
        model_path,
        "v5_p2_b3_count4_b0_r2",
    )

    DatasetClass = (
        loader_module
        .V5P2PairAlignedPrimary58Dataset
    )
    ModelClass = (
        model_module
        .P2B3Conv1DOnlyCount4
    )

    train_dataset = DatasetClass(
        root=root,
        split="train",
        pair_manifest=paths["pair_manifest"],
    )

    pair_bases: dict[
        str,
        list[int],
    ] = defaultdict(list)

    for base in range(
        0,
        len(train_dataset._index),
        2,
    ):
        attack_entry = train_dataset._index[base]
        control_entry = train_dataset._index[
            base + 1
        ]

        if (
            attack_entry.mode != "attack"
            or control_entry.mode != "control"
            or attack_entry.pair_key
            != control_entry.pair_key
            or attack_entry.start
            != control_entry.start
            or attack_entry.target
            != control_entry.target
        ):
            failures.append(
                f"train pair-index adjacency failed "
                f"at base={base}"
            )
            continue

        pair_bases[
            attack_entry.pair_key
        ].append(base)

    selected_by_category: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for pair_key in sorted(pair_bases):
        category = category_from_pair_key(
            pair_key
        )
        if category not in {
            1,
            2,
            3,
            4,
        }:
            continue
        if len(
            selected_by_category[category]
        ) >= 2:
            continue

        bases = pair_bases[pair_key]
        first_entry = train_dataset._index[
            bases[0]
        ]
        attack_run = train_dataset._load_run(
            first_entry.file_path
        )

        active_bases = [
            base
            for base in bases
            if int(
                attack_run["y_attack"][
                    train_dataset._index[
                        base
                    ].target
                ].item()
            )
            == 1
        ]
        if not active_bases:
            continue

        base = active_bases[-1]
        attack_entry = train_dataset._index[
            base
        ]
        control_entry = train_dataset._index[
            base + 1
        ]

        attack_item = train_dataset[base]
        control_item = train_dataset[
            base + 1
        ]

        raw_count = int(
            attack_item[
                "y_attacker_count"
            ].item()
        )
        source_count = int(
            attack_item[
                "y_source"
            ].sum().item()
        )

        if float(
            attack_item["y_attack"].item()
        ) != 1.0:
            failures.append(
                f"{pair_key}: selected attack is inactive"
            )
            continue
        if float(
            control_item["y_attack"].item()
        ) != 0.0:
            failures.append(
                f"{pair_key}: matched control is active"
            )
            continue
        if raw_count != category:
            failures.append(
                f"{pair_key}: attack count={raw_count}, "
                f"category={category}"
            )
            continue
        if source_count != category:
            failures.append(
                f"{pair_key}: source count="
                f"{source_count}, category={category}"
            )
            continue
        if int(
            control_item[
                "y_attacker_count"
            ].item()
        ) != 0:
            failures.append(
                f"{pair_key}: control count is nonzero"
            )
            continue

        selected_by_category[
            category
        ].append(
            {
                "category": category,
                "pair_key": pair_key,
                "attack_index": base,
                "control_index": base + 1,
                "window_start": (
                    attack_entry.start
                ),
                "window_target": (
                    attack_entry.target
                ),
                "common_length": (
                    attack_entry.common_length
                ),
                "raw_positive_attacker_count": (
                    raw_count
                ),
            }
        )

        if all(
            len(
                selected_by_category[
                    category_value
                ]
            )
            >= 2
            for category_value in (
                1,
                2,
                3,
                4,
            )
        ):
            break

    for category in (1, 2, 3, 4):
        if len(
            selected_by_category[category]
        ) != 2:
            failures.append(
                f"selected "
                f"{len(selected_by_category[category])} "
                f"active K{category} pairs, expected 2"
            )

    selected_pairs = [
        record
        for category in (1, 2, 3, 4)
        for record in selected_by_category[
            category
        ]
    ]

    selected_indices: list[int] = []
    for record in selected_pairs:
        selected_indices.extend(
            [
                record["attack_index"],
                record["control_index"],
            ]
        )

    if len(selected_pairs) != 8:
        failures.append(
            f"selected pair count="
            f"{len(selected_pairs)}, expected 8"
        )
    if len(selected_indices) != 16:
        failures.append(
            f"tiny item count="
            f"{len(selected_indices)}, expected 16"
        )

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "selection": selected_pairs,
            "training_performed": False,
            "validation_tensor_contents_accessed": False,
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

    subset = Subset(
        train_dataset,
        selected_indices,
    )
    batch_loader = DataLoader(
        subset,
        batch_size=len(subset),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    batch = next(iter(batch_loader))

    expected_batch_keys = {
        "x",
        "physical_port_mask",
        "y_attack",
        "y_attacker_count",
        "y_source",
        "y_transit",
        "y_victim",
        "y_attack_path",
        "role_mask",
    }
    if set(batch) != expected_batch_keys:
        failures.append(
            f"collated keys={sorted(batch)}, "
            f"expected={sorted(expected_batch_keys)}"
        )
    if tuple(
        batch["x"].shape
    ) != (16, 16, 58, 32):
        failures.append(
            f"collated x shape="
            f"{tuple(batch['x'].shape)}"
        )
    if tuple(
        batch[
            "physical_port_mask"
        ].shape
    ) != (16, 16, 10):
        failures.append(
            "collated physical-port-mask "
            "shape mismatch"
        )

    expected_role_mask = (
        batch["y_source"].to(torch.int64)
        + 2
        * batch["y_transit"].to(
            torch.int64
        )
        + 4
        * batch["y_victim"].to(
            torch.int64
        )
    )
    actual_role_mask = batch[
        "role_mask"
    ].to(torch.int64)

    role_mask_mismatch_count = int(
        (
            actual_role_mask
            != expected_role_mask
        ).sum().item()
    )
    if role_mask_mismatch_count != 0:
        failures.append(
            f"tiny batch role_mask bit-field "
            f"mismatches={role_mask_mismatch_count}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    active_mask = (
        batch["y_attack"] >= 0.5
    )
    if int(
        active_mask.sum().item()
    ) != 8:
        failures.append(
            f"active item count="
            f"{int(active_mask.sum().item())}, "
            "expected 8"
        )

    count_class_mapping = {
        raw_count: class_index
        for class_index, raw_count in enumerate(
            EXPECTED_COUNT_CLASS_VALUES
        )
    }
    count_targets = torch.full(
        batch[
            "y_attacker_count"
        ].shape,
        fill_value=-100,
        dtype=torch.int64,
        device=device,
    )

    for raw_count, class_index in (
        count_class_mapping.items()
    ):
        selector = (
            active_mask
            & (
                batch[
                    "y_attacker_count"
                ]
                == raw_count
            )
        )
        count_targets[selector] = class_index

    if bool(
        (
            count_targets[active_mask]
            < 0
        ).any().item()
    ):
        failures.append(
            "one or more active count targets "
            "were not mapped"
        )

    active_raw_counts = sorted(
        int(value)
        for value in torch.unique(
            batch[
                "y_attacker_count"
            ][active_mask]
        ).tolist()
    )
    if (
        active_raw_counts
        != EXPECTED_COUNT_CLASS_VALUES
    ):
        failures.append(
            f"tiny batch active raw counts="
            f"{active_raw_counts}, expected "
            f"{EXPECTED_COUNT_CLASS_VALUES}"
        )

    model = ModelClass().to(device)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if (
        parameter_count
        != EXPECTED_PARAMETER_COUNT
    ):
        failures.append(
            f"model parameter count="
            f"{parameter_count}, expected "
            f"{EXPECTED_PARAMETER_COUNT}"
        )
    if (
        trainable_parameter_count
        != EXPECTED_PARAMETER_COUNT
    ):
        failures.append(
            f"trainable parameter count="
            f"{trainable_parameter_count}, expected "
            f"{EXPECTED_PARAMETER_COUNT}"
        )
    if model.count_head.out_features != 4:
        failures.append(
            "P2 count head does not have 4 logits"
        )
    if list(
        model.count_class_values
    ) != EXPECTED_COUNT_CLASS_VALUES:
        failures.append(
            "model count-class values changed"
        )

    with torch.no_grad():
        shape_outputs = model(
            batch["x"],
            batch["physical_port_mask"],
        )

    expected_output_shapes = {
        "attack_logits": (16,),
        "count_logits": (16, 4),
        "source_logits": (16, 16),
        "transit_logits": (16, 16),
        "victim_logits": (16, 16),
        "path_logits": (16, 16),
    }
    for key, expected_shape in (
        expected_output_shapes.items()
    ):
        if tuple(
            shape_outputs[key].shape
        ) != expected_shape:
            failures.append(
                f"{key} shape="
                f"{tuple(shape_outputs[key].shape)}, "
                f"expected {expected_shape}"
            )

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "selection": selected_pairs,
            "training_performed": False,
            "validation_tensor_contents_accessed": False,
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

    graph_pos_weight = positive_weight(
        batch["y_attack"]
    )
    role_pos_weights = {
        "source": positive_weight(
            batch["y_source"]
        ),
        "transit": positive_weight(
            batch["y_transit"]
        ),
        "victim": positive_weight(
            batch["y_victim"]
        ),
        "path": positive_weight(
            batch["y_attack_path"]
        ),
    }

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history: list[dict[str, Any]] = []
    initial_loss = None
    final_loss = None
    final_metrics = None
    completed_step = 0
    stable_exact_steps = 0
    maximum_gradient_norm = 0.0
    maximum_nonzero_gradient_tensors = 0

    def compute_loss(
        outputs: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        components = {
            "attack": (
                F.binary_cross_entropy_with_logits(
                    outputs["attack_logits"],
                    batch["y_attack"].float(),
                    pos_weight=graph_pos_weight,
                )
            ),
            "count": (
                F.cross_entropy(
                    outputs[
                        "count_logits"
                    ][active_mask],
                    count_targets[active_mask],
                )
            ),
            "source": (
                F.binary_cross_entropy_with_logits(
                    outputs["source_logits"],
                    batch["y_source"].float(),
                    pos_weight=role_pos_weights[
                        "source"
                    ],
                )
            ),
            "transit": (
                F.binary_cross_entropy_with_logits(
                    outputs["transit_logits"],
                    batch["y_transit"].float(),
                    pos_weight=role_pos_weights[
                        "transit"
                    ],
                )
            ),
            "victim": (
                F.binary_cross_entropy_with_logits(
                    outputs["victim_logits"],
                    batch["y_victim"].float(),
                    pos_weight=role_pos_weights[
                        "victim"
                    ],
                )
            ),
            "path": (
                F.binary_cross_entropy_with_logits(
                    outputs["path_logits"],
                    batch[
                        "y_attack_path"
                    ].float(),
                    pos_weight=role_pos_weights[
                        "path"
                    ],
                )
            ),
        }

        total = sum(
            LOSS_WEIGHTS[key]
            * value
            for key, value in components.items()
        )
        return total, components

    for step in range(
        1,
        MAX_STEPS + 1,
    ):
        model.train()
        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            batch["x"],
            batch["physical_port_mask"],
        )
        loss, components = compute_loss(
            outputs
        )

        if not torch.isfinite(loss):
            failures.append(
                f"non-finite loss at step {step}"
            )
            break

        if initial_loss is None:
            initial_loss = float(
                loss.item()
            )

        loss.backward()

        squared_norm = 0.0
        nonzero_gradient_tensors = 0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue

            gradient = parameter.grad.detach()
            squared_norm += float(
                gradient.float()
                .pow(2)
                .sum()
                .item()
            )
            if bool(
                (gradient != 0).any().item()
            ):
                nonzero_gradient_tensors += 1

        gradient_norm = math.sqrt(
            squared_norm
        )
        maximum_gradient_norm = max(
            maximum_gradient_norm,
            gradient_norm,
        )
        maximum_nonzero_gradient_tensors = max(
            maximum_nonzero_gradient_tensors,
            nonzero_gradient_tensors,
        )

        optimizer.step()

        model.eval()
        with torch.no_grad():
            current_outputs = model(
                batch["x"],
                batch[
                    "physical_port_mask"
                ],
            )
            current_loss, current_components = (
                compute_loss(current_outputs)
            )
            metrics = evaluate(
                current_outputs,
                batch,
                count_targets,
                active_mask,
            )

        completed_step = step
        final_loss = float(
            current_loss.item()
        )
        final_metrics = metrics

        if (
            metrics[
                "all_tasks_all_exact"
            ]
            and final_loss <= 0.05
        ):
            stable_exact_steps += 1
        else:
            stable_exact_steps = 0

        if (
            step == 1
            or step % 25 == 0
            or stable_exact_steps > 0
        ):
            history.append(
                {
                    "step": step,
                    "loss": final_loss,
                    "attack_loss": float(
                        current_components[
                            "attack"
                        ].item()
                    ),
                    "count_loss": float(
                        current_components[
                            "count"
                        ].item()
                    ),
                    "source_loss": float(
                        current_components[
                            "source"
                        ].item()
                    ),
                    "transit_loss": float(
                        current_components[
                            "transit"
                        ].item()
                    ),
                    "victim_loss": float(
                        current_components[
                            "victim"
                        ].item()
                    ),
                    "path_loss": float(
                        current_components[
                            "path"
                        ].item()
                    ),
                    "gradient_norm_before_step": (
                        gradient_norm
                    ),
                    "graph_all_exact": (
                        metrics[
                            "graph"
                        ][
                            "all_exact"
                        ]
                    ),
                    "count_all_exact": (
                        metrics[
                            "count_all_exact"
                        ]
                    ),
                    "count_active_accuracy": (
                        metrics[
                            "count_active_accuracy"
                        ]
                    ),
                    "source_all_exact": (
                        metrics[
                            "roles"
                        ][
                            "source"
                        ][
                            "all_exact"
                        ]
                    ),
                    "transit_all_exact": (
                        metrics[
                            "roles"
                        ][
                            "transit"
                        ][
                            "all_exact"
                        ]
                    ),
                    "victim_all_exact": (
                        metrics[
                            "roles"
                        ][
                            "victim"
                        ][
                            "all_exact"
                        ]
                    ),
                    "path_all_exact": (
                        metrics[
                            "roles"
                        ][
                            "path"
                        ][
                            "all_exact"
                        ]
                    ),
                    "all_tasks_all_exact": (
                        metrics[
                            "all_tasks_all_exact"
                        ]
                    ),
                    "stable_exact_steps": (
                        stable_exact_steps
                    ),
                }
            )

        if (
            stable_exact_steps
            >= REQUIRED_STABLE_STEPS
        ):
            break

    if (
        initial_loss is None
        or final_loss is None
        or final_metrics is None
    ):
        failures.append(
            "tiny-overfit loop did not produce metrics"
        )
        loss_reduction_fraction = None
    else:
        loss_reduction_fraction = (
            (
                initial_loss
                - final_loss
            )
            / initial_loss
            if initial_loss > 0
            else 0.0
        )

        if not final_metrics[
            "all_tasks_all_exact"
        ]:
            failures.append(
                "16-item batch was not memorized "
                "across all tasks"
            )
        if final_loss > 0.05:
            failures.append(
                f"final loss={final_loss:.8g} > 0.05"
            )
        if loss_reduction_fraction < 0.95:
            failures.append(
                "loss reduction is below 95%"
            )
        if (
            stable_exact_steps
            < REQUIRED_STABLE_STEPS
        ):
            failures.append(
                f"stable exact steps="
                f"{stable_exact_steps}, expected "
                f"{REQUIRED_STABLE_STEPS}"
            )
        if maximum_gradient_norm <= 0:
            failures.append(
                "no nonzero aggregate gradient observed"
            )
        if (
            maximum_nonzero_gradient_tensors
            <= 0
        ):
            failures.append(
                "no parameter tensor received "
                "nonzero gradient"
            )

    history_path = (
        output_dir
        / "V5_P2_B0_R2_A4_REPEAT_HISTORY.csv"
    )
    selection_path = (
        output_dir
        / "V5_P2_B0_R2_A4_REPEAT_SELECTION.json"
    )
    architecture_contract_path = (
        output_dir
        / "V5_P2_B0_R2_P2_B3_COUNT4_ARCHITECTURE_CONTRACT.json"
    )

    write_csv(
        history_path,
        history,
    )
    write_json(
        selection_path,
        {
            "seed": SEED,
            "selection_policy": (
                "two active latest pair-aligned "
                "windows per K1/K2/K3/K4 plus "
                "matched controls"
            ),
            "selected_pairs": selected_pairs,
            "selected_indices": (
                selected_indices
            ),
            "active_raw_counts": (
                active_raw_counts
            ),
            "count_class_mapping": {
                str(key): value
                for key, value in (
                    count_class_mapping.items()
                )
            },
            "role_mask_semantics": (
                "source + 2*transit + 4*victim"
            ),
            "role_mask_mismatch_count": (
                role_mask_mismatch_count
            ),
            "role_mask_used_as_model_input": False,
            "role_mask_used_as_model_output": False,
        },
    )

    architecture_contract = {
        "contract_name": (
            "V5_P2_B3_COUNT4_MINIMAL_ARCHITECTURAL_DELTA"
        ),
        "contract_version": 1,
        "base_architecture": (
            "B3_CAUSAL_DEPTHWISE_SEPARABLE_CONV1D_ONLY"
        ),
        "p2_architecture": (
            model.architecture_name
        ),
        "unchanged_components": [
            "PRIMARY58 input contract",
            "topology-derived Boolean [16,10] mask",
            "Conv1d 58->64 input projection",
            "causal depthwise-separable residual blocks",
            "kernel size 3",
            "dilations [1,2,4,8]",
            "final causal timestep readout",
            "node projection Linear(74,64)",
            "source head",
            "transit head",
            "victim head",
            "path head",
            "mean+max graph pooling",
            "graph projection",
            "attack head",
            "no message passing",
        ],
        "changed_component": {
            "name": "count_head",
            "before": "Linear(64,3)",
            "after": "Linear(64,4)",
            "active_raw_count_values": [
                1,
                2,
                3,
                4,
            ],
            "class_mapping": {
                "1": 0,
                "2": 1,
                "3": 2,
                "4": 3,
            },
            "zero_count_representation": (
                "graph attack head predicts normal; "
                "count loss applies only to active attacks"
            ),
        },
        "parameter_count_before": 43_208,
        "parameter_count_after": 43_273,
        "parameter_delta": 65,
        "role_mask": {
            "semantics": (
                "source + 2*transit + 4*victim"
            ),
            "learned_input": False,
            "model_output": False,
            "training_loss_target": False,
            "bookkeeping_only": True,
        },
        "test_boundary": {
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        },
        "model_sha256": sha256_file(
            model_path
        ),
    }
    architecture_contract[
        "contract_sha256"
    ] = canonical_sha256(
        architecture_contract
    )
    write_json(
        architecture_contract_path,
        architecture_contract,
    )

    status = (
        "COMPLETE"
        if not failures
        else "HOLD"
    )
    decision = (
        "FREEZE_P2_B3_COUNT4_AND_AUTHORIZE_"
        "CORRECTED_B0_R3_SHORTCUT_AUDIT"
        if not failures
        else "BLOCK_P2_B0_R3"
    )
    next_stage = (
        "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT"
        if not failures
        else None
    )

    report = {
        "stage": STAGE,
        "status": status,
        "decision": decision,
        "purpose": (
            "minimal P2 count-head correction and "
            "train-only integration repeat; "
            "not final P2 training"
        ),
        "dataset": {
            "link_path": str(
                root_input.absolute()
            ),
            "resolved_root": str(root),
        },
        "architecture": {
            "name": (
                model.architecture_name
            ),
            "parameter_count": (
                parameter_count
            ),
            "trainable_parameter_count": (
                trainable_parameter_count
            ),
            "count_logits": 4,
            "count_class_values": (
                EXPECTED_COUNT_CLASS_VALUES
            ),
            "graph_message_passing": False,
            "edge_index_model_input": False,
            "only_change_from_p0_b3": (
                "count_head Linear(64,3) -> "
                "Linear(64,4)"
            ),
        },
        "tiny_subset": {
            "pair_count": len(
                selected_pairs
            ),
            "item_count": len(
                selected_indices
            ),
            "active_attack_count": int(
                active_mask.sum().item()
            ),
            "control_count": int(
                (~active_mask).sum().item()
            ),
            "selected_pairs": (
                selected_pairs
            ),
            "active_raw_counts": (
                active_raw_counts
            ),
            "count_class_mapping": {
                str(key): value
                for key, value in (
                    count_class_mapping.items()
                )
            },
            "role_mask_bitfield_mismatch_count": (
                role_mask_mismatch_count
            ),
        },
        "protocol": {
            "seed": SEED,
            "device": str(device),
            "optimizer": "Adam",
            "learning_rate": (
                LEARNING_RATE
            ),
            "maximum_steps": (
                MAX_STEPS
            ),
            "required_stable_exact_steps": (
                REQUIRED_STABLE_STEPS
            ),
            "loss_weights": (
                LOSS_WEIGHTS
            ),
            "count_loss_scope": (
                "active attack items only"
            ),
            "role_mask_loss_applied": False,
            "audit_weights_saved": False,
            "audit_weights_authorized_for_reuse": False,
        },
        "results": {
            "completed_step": (
                completed_step
            ),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction_fraction": (
                loss_reduction_fraction
            ),
            "maximum_gradient_norm": (
                maximum_gradient_norm
            ),
            "maximum_nonzero_gradient_parameter_tensors": (
                maximum_nonzero_gradient_tensors
            ),
            "stable_exact_steps": (
                stable_exact_steps
            ),
            "final_metrics": (
                final_metrics
            ),
        },
        "artifacts": {
            "history_csv": [
                str(history_path),
                sha256_file(
                    history_path
                ),
            ],
            "selection_json": [
                str(selection_path),
                sha256_file(
                    selection_path
                ),
            ],
            "architecture_contract_json": [
                str(
                    architecture_contract_path
                ),
                sha256_file(
                    architecture_contract_path
                ),
            ],
        },
        "provenance": {
            "a1_r2_report_sha256": (
                sha256_file(
                    paths["a1_r2_report"]
                )
            ),
            "a1_r2_lock_sha256": (
                sha256_file(
                    paths["a1_r2_lock"]
                )
            ),
            "pair_manifest_sha256": (
                sha256_file(
                    paths["pair_manifest"]
                )
            ),
            "a3_report_sha256": (
                sha256_file(
                    paths["a3_report"]
                )
            ),
            "a3_lock_sha256": (
                sha256_file(
                    paths["a3_lock"]
                )
            ),
            "historical_a4_report_sha256": (
                sha256_file(
                    paths["a4_report"]
                )
            ),
            "historical_a4_lock_sha256": (
                sha256_file(
                    paths["a4_lock"]
                )
            ),
            "b0_r1b_report_sha256": (
                sha256_file(
                    paths["b0_r1b_report"]
                )
            ),
            "b0_r1b_lock_sha256": (
                sha256_file(
                    paths["b0_r1b_lock"]
                )
            ),
            "loader_sha256": sha256_file(
                paths["loader"]
            ),
            "model_sha256": sha256_file(
                paths["model"]
            ),
        },
        "security_boundary": {
            "training_performed": True,
            "training_scope": (
                "fixed 16-item train-only "
                "audit batch"
            ),
            "final_experiment_training_performed": False,
            "audit_weights_saved": False,
            "train_tensor_contents_accessed": True,
            "validation_tensor_contents_accessed": False,
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
            "===== V5 P2-B0-R2 FINAL ====="
        )
        print("status: HOLD")
        print("decision:", decision)
        print(
            "completed_step:",
            completed_step,
        )
        print(
            "initial_loss:",
            initial_loss,
        )
        print(
            "final_loss:",
            final_loss,
        )
        print(
            "stable_exact_steps:",
            stable_exact_steps,
        )
        print(
            "failure_count:",
            len(failures),
        )
        for failure in failures:
            print("FAIL:", failure)
        print(
            "warning_count:",
            len(warnings),
        )
        print(f"{STAGE}_HOLD")
        return 1

    lock = {
        "status": COMPLETE,
        "decision": decision,
        "report_sha256": (
            sha256_file(report_path)
        ),
        "history_sha256": (
            sha256_file(history_path)
        ),
        "selection_sha256": (
            sha256_file(selection_path)
        ),
        "architecture_contract_sha256": (
            architecture_contract[
                "contract_sha256"
            ]
        ),
        "architecture_contract_file_sha256": (
            sha256_file(
                architecture_contract_path
            )
        ),
        "model_sha256": (
            sha256_file(model_path)
        ),
        "architecture_name": (
            model.architecture_name
        ),
        "parameter_count": (
            parameter_count
        ),
        "count_logits": 4,
        "count_class_values": (
            EXPECTED_COUNT_CLASS_VALUES
        ),
        "tiny_pair_count": len(
            selected_pairs
        ),
        "tiny_item_count": len(
            selected_indices
        ),
        "active_attack_count": int(
            active_mask.sum().item()
        ),
        "control_count": int(
            (~active_mask).sum().item()
        ),
        "role_mask_semantics": (
            "bitfield3_source1_transit2_victim4"
        ),
        "role_mask_mismatch_count": (
            role_mask_mismatch_count
        ),
        "completed_step": (
            completed_step
        ),
        "initial_loss": (
            initial_loss
        ),
        "final_loss": (
            final_loss
        ),
        "loss_reduction_fraction": (
            loss_reduction_fraction
        ),
        "all_tasks_all_exact": (
            final_metrics[
                "all_tasks_all_exact"
            ]
        ),
        "stable_exact_steps": (
            stable_exact_steps
        ),
        "audit_weights_saved": False,
        "audit_weights_authorized_for_reuse": False,
        "validation_tensor_contents_accessed": False,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "next_stage": next_stage,
        "script_sha256": sha256_file(
            Path(__file__)
        ),
    }
    write_json(
        output_dir
        / f"{STAGE}_LOCK.json",
        lock,
    )
    atomic_write(
        output_dir / COMPLETE,
        COMPLETE + "\n",
    )

    print(
        "===== V5 P2-B0-R2 FINAL ====="
    )
    print("status: COMPLETE")
    print("decision:", decision)
    print(
        "architecture:",
        model.architecture_name,
    )
    print(
        "parameter_count:",
        parameter_count,
    )
    print("count_logits: 4")
    print(
        "count_class_values:",
        EXPECTED_COUNT_CLASS_VALUES,
    )
    print("device:", device)
    print(
        "tiny_pair_count:",
        len(selected_pairs),
    )
    print(
        "tiny_item_count:",
        len(selected_indices),
    )
    print(
        "active_attack_count:",
        int(active_mask.sum().item()),
    )
    print(
        "control_count:",
        int((~active_mask).sum().item()),
    )
    print(
        "active_raw_counts:",
        active_raw_counts,
    )
    print(
        "count_class_mapping:",
        count_class_mapping,
    )
    print(
        "role_mask_semantics: "
        "bitfield3_source1_transit2_victim4"
    )
    print(
        "role_mask_mismatch_count:",
        role_mask_mismatch_count,
    )
    print(
        "completed_step:",
        completed_step,
    )
    print(
        "initial_loss:",
        f"{initial_loss:.10g}",
    )
    print(
        "final_loss:",
        f"{final_loss:.10g}",
    )
    print(
        "loss_reduction_fraction:",
        f"{loss_reduction_fraction:.10g}",
    )
    print(
        "maximum_gradient_norm:",
        f"{maximum_gradient_norm:.10g}",
    )
    print(
        "all_tasks_all_exact:",
        str(
            final_metrics[
                "all_tasks_all_exact"
            ]
        ).lower(),
    )
    print(
        "stable_exact_steps:",
        stable_exact_steps,
    )
    print(
        "audit_weights_saved: false"
    )
    print(
        "audit_weights_authorized_for_reuse: false"
    )
    print(
        "validation_tensor_contents_accessed: false"
    )
    print(
        "test_directory_enumerated: false"
    )
    print(
        "test_tensor_contents_accessed: false"
    )
    print(
        "failure_count:",
        len(failures),
    )
    print(
        "warning_count:",
        len(warnings),
    )
    print(
        "next_stage:",
        next_stage,
    )
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
