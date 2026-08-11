#!/usr/bin/env python3
"""
V5 P2-A4 Loader Integration and Tiny-Overfit Audit

This is an integration test, not final P2 training.

It:
- verifies A3 and prior locks;
- constructs the audited TRAIN loader only;
- selects two active matched pair-windows from each positive attacker-count
  category exposed by the train split;
- includes each matched CONTROL item at the identical window start;
- batches the 12 selected items through torch DataLoader;
- instantiates the frozen 43,208-parameter B3 Conv1D-only architecture;
- trains only on this tiny fixed batch until it memorizes all graph, count,
  source, transit, victim, and path targets;
- records gradients, loss reduction, and exact predictions.

The resulting weights are disposable audit weights and are not saved or
authorized for any experiment.

No validation tensor is opened and no test directory is enumerated or opened.
"""

from __future__ import annotations

import argparse
import csv
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


STAGE = "V5_P2_A4_LOADER_INTEGRATION_AND_TINY_OVERFIT"
COMPLETE = f"{STAGE}_COMPLETE"

SEED = 2404
MAX_STEPS = 3000
LEARNING_RATE = 1e-2
REQUIRED_STABLE_STEPS = 20

LOSS_WEIGHTS = {
    "attack": 1.0,
    "count": 0.5,
    "source": 1.0,
    "transit": 0.5,
    "victim": 0.75,
    "path": 0.5,
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


def import_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")

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
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def category_from_pair_key(pair_key: str) -> str | None:
    match = re.search(r"-K([124])-", pair_key)
    if match is None:
        return None
    return f"K{match.group(1)}"


def positive_weight(target: torch.Tensor) -> torch.Tensor:
    target_float = target.float()
    positive = target_float.sum()
    negative = target_float.numel() - positive

    if positive.item() <= 0:
        return torch.tensor(
            1.0,
            device=target.device,
            dtype=torch.float32,
        )
    value = (negative / positive).clamp(min=1.0, max=20.0)
    return value.detach().to(torch.float32)


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
        exact_items = equality.reshape(equality.shape[0], -1).all(dim=1)

    return {
        "element_accuracy": float(equality.float().mean().item()),
        "exact_item_accuracy": float(
            exact_items.float().mean().item()
        ),
        "all_exact": bool(equality.all().item()),
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

    active_count_prediction = outputs["count_logits"][
        active_mask
    ].argmax(dim=-1)
    active_count_truth = count_targets[active_mask]
    count_accuracy = float(
        (
            active_count_prediction == active_count_truth
        ).float().mean().item()
    )
    count_all_exact = bool(
        (
            active_count_prediction == active_count_truth
        ).all().item()
    )

    roles = {}
    for logical, output_key, target_key in (
        ("source", "source_logits", "y_source"),
        ("transit", "transit_logits", "y_transit"),
        ("victim", "victim_logits", "y_victim"),
        ("path", "path_logits", "y_attack_path"),
    ):
        roles[logical] = exact_binary_metrics(
            outputs[output_key],
            batch[target_key],
        )

    all_exact = (
        graph["all_exact"]
        and count_all_exact
        and all(item["all_exact"] for item in roles.values())
    )

    return {
        "graph": graph,
        "count_active_accuracy": count_accuracy,
        "count_all_exact": count_all_exact,
        "roles": roles,
        "all_tasks_all_exact": all_exact,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a1-r2-dir", type=Path, required=True)
    parser.add_argument("--a2-r2-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root_input = args.root.expanduser()
    root = root_input.resolve()
    a1_r2_dir = args.a1_r2_dir.expanduser().resolve()
    a2_r2_dir = args.a2_r2_dir.expanduser().resolve()
    a3_dir = args.a3_dir.expanduser().resolve()
    loader_path = args.loader_path.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
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
        "a2_r2_report": (
            a2_r2_dir
            / "V5_P2_A2_R2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT.json"
        ),
        "a2_r2_lock": (
            a2_r2_dir
            / "V5_P2_A2_R2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT_LOCK.json"
        ),
        "a3_report": (
            a3_dir
            / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT.json"
        ),
        "a3_lock": (
            a3_dir
            / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT_LOCK.json"
        ),
        "loader": loader_path,
        "model": model_path,
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    if not (root / "runs" / "train").is_dir():
        failures.append("missing runs/train")

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
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        return 1

    a1_report = load_json(paths["a1_r2_report"])
    a1_lock = load_json(paths["a1_r2_lock"])
    a2_report = load_json(paths["a2_r2_report"])
    a2_lock = load_json(paths["a2_r2_lock"])
    a3_report = load_json(paths["a3_report"])
    a3_lock = load_json(paths["a3_lock"])

    for label, report in (
        ("A1-R2", a1_report),
        ("A2-R2", a2_report),
        ("A3", a3_report),
    ):
        if report.get("status") != "COMPLETE":
            failures.append(f"{label} status is not COMPLETE")
        if report.get("security_boundary", {}).get(
            "test_tensor_contents_accessed"
        ) is not False:
            failures.append(
                f"{label} does not certify untouched test tensors"
            )

    if a1_lock.get("report_sha256") != sha256_file(paths["a1_r2_report"]):
        failures.append("A1-R2 report SHA mismatch")
    if a2_lock.get("report_sha256") != sha256_file(paths["a2_r2_report"]):
        failures.append("A2-R2 report SHA mismatch")
    if a3_lock.get("report_sha256") != sha256_file(paths["a3_report"]):
        failures.append("A3 report SHA mismatch")
    if a3_lock.get("loader_sha256") != sha256_file(paths["loader"]):
        failures.append("A3 loader SHA mismatch")
    if a3_lock.get("total_aligned_items") != 82694:
        failures.append("A3 total aligned item count changed")
    if a3_lock.get("x_shape") != [16, 58, 32]:
        failures.append("A3 x shape changed")
    if a3_lock.get("mask_shape") != [16, 10]:
        failures.append("A3 mask shape changed")
    if a3_lock.get("test_constructor_rejected") is not True:
        failures.append("A3 did not lock test-constructor rejection")
    if a2_lock.get("graph_message_passing") is not False:
        failures.append("A2-R2 graph-message-passing flag changed")
    if a2_lock.get("edge_index_model_input") is not False:
        failures.append("A2-R2 edge-index input flag changed")
    if a2_lock.get("second_normalization_forbidden") is not True:
        failures.append("A2-R2 second-normalization flag changed")

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
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    set_seed(SEED)

    loader_module = import_module(
        loader_path,
        "v5_p2_pair_aligned_primary58_dataset_a4",
    )
    model_module = import_module(
        model_path,
        "v5_frozen_b3_conv1d_only_a4",
    )

    DatasetClass = loader_module.V5P2PairAlignedPrimary58Dataset
    ModelClass = model_module.FrozenB3Conv1DOnly

    train_dataset = DatasetClass(
        root=root,
        split="train",
        pair_manifest=paths["pair_manifest"],
    )

    # Locate the final aligned ATTACK/CONTROL start for each pair.
    last_pair_base: dict[str, int] = {}
    for base in range(0, len(train_dataset._index), 2):
        attack_entry = train_dataset._index[base]
        control_entry = train_dataset._index[base + 1]
        if (
            attack_entry.mode != "attack"
            or control_entry.mode != "control"
            or attack_entry.pair_key != control_entry.pair_key
            or attack_entry.start != control_entry.start
        ):
            failures.append(
                f"train index pair adjacency failed at base={base}"
            )
            continue
        last_pair_base[attack_entry.pair_key] = base

    selected_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_count_by_category: dict[str, set[int]] = defaultdict(set)

    for pair_key in sorted(last_pair_base):
        category = category_from_pair_key(pair_key)
        if category not in {"K1", "K2", "K4"}:
            continue
        if len(selected_by_category[category]) >= 2:
            continue

        base = last_pair_base[pair_key]
        attack_item = train_dataset[base]
        control_item = train_dataset[base + 1]

        attack_label = float(attack_item["y_attack"].item())
        control_label = float(control_item["y_attack"].item())
        raw_count = int(attack_item["y_attacker_count"].item())

        if attack_label < 0.5:
            continue
        if control_label >= 0.5:
            failures.append(
                f"{pair_key}: matched control is graph-positive"
            )
            continue
        if int(control_item["y_attacker_count"].item()) != 0:
            failures.append(
                f"{pair_key}: matched control attacker count is nonzero"
            )
            continue
        if raw_count <= 0:
            failures.append(
                f"{pair_key}: active attack has non-positive count"
            )
            continue

        entry = train_dataset._index[base]
        selected_by_category[category].append(
            {
                "category": category,
                "pair_key": pair_key,
                "base_index": base,
                "attack_index": base,
                "control_index": base + 1,
                "window_start": entry.start,
                "window_target": entry.target,
                "common_length": entry.common_length,
                "raw_positive_attacker_count": raw_count,
            }
        )
        raw_count_by_category[category].add(raw_count)

        if all(
            len(selected_by_category[item]) >= 2
            for item in ("K1", "K2", "K4")
        ):
            break

    for category in ("K1", "K2", "K4"):
        if len(selected_by_category[category]) != 2:
            failures.append(
                f"selected {len(selected_by_category[category])} "
                f"active pairs for {category}, expected 2"
            )
        if len(raw_count_by_category[category]) != 1:
            failures.append(
                f"{category} maps to raw counts "
                f"{sorted(raw_count_by_category[category])}, "
                "expected one value"
            )

    selected_pairs = [
        record
        for category in ("K1", "K2", "K4")
        for record in selected_by_category[category]
    ]
    positive_raw_counts = sorted(
        {
            record["raw_positive_attacker_count"]
            for record in selected_pairs
        }
    )
    if len(positive_raw_counts) != 3:
        failures.append(
            f"positive raw attacker counts={positive_raw_counts}, "
            "expected exactly three categories"
        )

    count_class_mapping = {
        raw_count: class_index
        for class_index, raw_count in enumerate(positive_raw_counts)
    }

    selected_indices = []
    for record in selected_pairs:
        selected_indices.extend(
            [record["attack_index"], record["control_index"]]
        )

    if len(selected_indices) != 12:
        failures.append(
            f"tiny subset size={len(selected_indices)}, expected 12"
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
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    subset = Subset(train_dataset, selected_indices)
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
            f"collated batch keys={sorted(batch)}, "
            f"expected {sorted(expected_batch_keys)}"
        )
    if tuple(batch["x"].shape) != (12, 16, 58, 32):
        failures.append(
            f"collated x shape={tuple(batch['x'].shape)}"
        )
    if batch["x"].dtype != torch.float32:
        failures.append(
            f"collated x dtype={batch['x'].dtype}"
        )
    if tuple(batch["physical_port_mask"].shape) != (12, 16, 10):
        failures.append(
            "collated physical-port-mask shape mismatch"
        )
    if batch["physical_port_mask"].dtype != torch.bool:
        failures.append(
            "collated physical-port-mask dtype mismatch"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    active_mask = batch["y_attack"] >= 0.5
    if int(active_mask.sum().item()) != 6:
        failures.append(
            f"active attack item count={int(active_mask.sum())}, expected 6"
        )

    count_targets = torch.full(
        batch["y_attacker_count"].shape,
        fill_value=-100,
        dtype=torch.int64,
        device=device,
    )
    for raw_count, class_index in count_class_mapping.items():
        count_targets[
            active_mask
            & (batch["y_attacker_count"] == raw_count)
        ] = class_index
    if bool((count_targets[active_mask] < 0).any().item()):
        failures.append("one or more active count targets were not mapped")

    model = ModelClass().to(device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if parameter_count != 43208:
        failures.append(
            f"model parameter_count={parameter_count}, expected 43208"
        )
    if trainable_parameter_count != 43208:
        failures.append(
            f"trainable_parameter_count={trainable_parameter_count}, "
            "expected 43208"
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
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    graph_pos_weight = positive_weight(batch["y_attack"])
    role_pos_weights = {
        "source": positive_weight(batch["y_source"]),
        "transit": positive_weight(batch["y_transit"]),
        "victim": positive_weight(batch["y_victim"]),
        "path": positive_weight(batch["y_attack_path"]),
    }

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history: list[dict[str, Any]] = []
    initial_loss = None
    final_loss = None
    final_metrics = None
    maximum_gradient_norm = 0.0
    nonzero_gradient_parameter_count = 0
    stable_exact_steps = 0
    completed_step = 0

    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            batch["x"],
            batch["physical_port_mask"],
        )

        attack_loss = F.binary_cross_entropy_with_logits(
            outputs["attack_logits"],
            batch["y_attack"].float(),
            pos_weight=graph_pos_weight,
        )
        count_loss = F.cross_entropy(
            outputs["count_logits"][active_mask],
            count_targets[active_mask],
        )
        source_loss = F.binary_cross_entropy_with_logits(
            outputs["source_logits"],
            batch["y_source"].float(),
            pos_weight=role_pos_weights["source"],
        )
        transit_loss = F.binary_cross_entropy_with_logits(
            outputs["transit_logits"],
            batch["y_transit"].float(),
            pos_weight=role_pos_weights["transit"],
        )
        victim_loss = F.binary_cross_entropy_with_logits(
            outputs["victim_logits"],
            batch["y_victim"].float(),
            pos_weight=role_pos_weights["victim"],
        )
        path_loss = F.binary_cross_entropy_with_logits(
            outputs["path_logits"],
            batch["y_attack_path"].float(),
            pos_weight=role_pos_weights["path"],
        )

        loss = (
            LOSS_WEIGHTS["attack"] * attack_loss
            + LOSS_WEIGHTS["count"] * count_loss
            + LOSS_WEIGHTS["source"] * source_loss
            + LOSS_WEIGHTS["transit"] * transit_loss
            + LOSS_WEIGHTS["victim"] * victim_loss
            + LOSS_WEIGHTS["path"] * path_loss
        )

        if not torch.isfinite(loss):
            failures.append(
                f"non-finite tiny-overfit loss at step {step}"
            )
            break

        if initial_loss is None:
            initial_loss = float(loss.item())

        loss.backward()

        squared_norm = 0.0
        current_nonzero = 0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            squared_norm += float(
                gradient.float().pow(2).sum().item()
            )
            if bool((gradient != 0).any().item()):
                current_nonzero += 1

        gradient_norm = math.sqrt(squared_norm)
        maximum_gradient_norm = max(
            maximum_gradient_norm,
            gradient_norm,
        )
        nonzero_gradient_parameter_count = max(
            nonzero_gradient_parameter_count,
            current_nonzero,
        )

        optimizer.step()

        model.eval()
        with torch.no_grad():
            current_outputs = model(
                batch["x"],
                batch["physical_port_mask"],
            )
            metrics = evaluate(
                current_outputs,
                batch,
                count_targets,
                active_mask,
            )

            # Recompute the aggregate loss after the optimizer step.
            current_loss = (
                LOSS_WEIGHTS["attack"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["attack_logits"],
                    batch["y_attack"].float(),
                    pos_weight=graph_pos_weight,
                )
                + LOSS_WEIGHTS["count"]
                * F.cross_entropy(
                    current_outputs["count_logits"][active_mask],
                    count_targets[active_mask],
                )
                + LOSS_WEIGHTS["source"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["source_logits"],
                    batch["y_source"].float(),
                    pos_weight=role_pos_weights["source"],
                )
                + LOSS_WEIGHTS["transit"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["transit_logits"],
                    batch["y_transit"].float(),
                    pos_weight=role_pos_weights["transit"],
                )
                + LOSS_WEIGHTS["victim"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["victim_logits"],
                    batch["y_victim"].float(),
                    pos_weight=role_pos_weights["victim"],
                )
                + LOSS_WEIGHTS["path"]
                * F.binary_cross_entropy_with_logits(
                    current_outputs["path_logits"],
                    batch["y_attack_path"].float(),
                    pos_weight=role_pos_weights["path"],
                )
            )

        completed_step = step
        final_loss = float(current_loss.item())
        final_metrics = metrics

        if (
            metrics["all_tasks_all_exact"]
            and final_loss <= 0.05
        ):
            stable_exact_steps += 1
        else:
            stable_exact_steps = 0

        if step == 1 or step % 25 == 0 or stable_exact_steps:
            history.append(
                {
                    "step": step,
                    "loss": final_loss,
                    "gradient_norm_before_step": gradient_norm,
                    "graph_all_exact": (
                        metrics["graph"]["all_exact"]
                    ),
                    "graph_element_accuracy": (
                        metrics["graph"]["element_accuracy"]
                    ),
                    "count_active_accuracy": (
                        metrics["count_active_accuracy"]
                    ),
                    "source_element_accuracy": (
                        metrics["roles"]["source"][
                            "element_accuracy"
                        ]
                    ),
                    "source_all_exact": (
                        metrics["roles"]["source"]["all_exact"]
                    ),
                    "transit_element_accuracy": (
                        metrics["roles"]["transit"][
                            "element_accuracy"
                        ]
                    ),
                    "transit_all_exact": (
                        metrics["roles"]["transit"]["all_exact"]
                    ),
                    "victim_element_accuracy": (
                        metrics["roles"]["victim"][
                            "element_accuracy"
                        ]
                    ),
                    "victim_all_exact": (
                        metrics["roles"]["victim"]["all_exact"]
                    ),
                    "path_element_accuracy": (
                        metrics["roles"]["path"][
                            "element_accuracy"
                        ]
                    ),
                    "path_all_exact": (
                        metrics["roles"]["path"]["all_exact"]
                    ),
                    "all_tasks_all_exact": (
                        metrics["all_tasks_all_exact"]
                    ),
                    "stable_exact_steps": stable_exact_steps,
                }
            )

        if stable_exact_steps >= REQUIRED_STABLE_STEPS:
            break

    if initial_loss is None or final_loss is None or final_metrics is None:
        failures.append("tiny-overfit loop did not produce metrics")
        loss_reduction_fraction = None
    else:
        loss_reduction_fraction = (
            (initial_loss - final_loss) / initial_loss
            if initial_loss > 0
            else 0.0
        )

        if not final_metrics["all_tasks_all_exact"]:
            failures.append(
                "tiny batch was not memorized exactly across all tasks"
            )
        if final_loss > 0.05:
            failures.append(
                f"final tiny-overfit loss={final_loss:.8g} > 0.05"
            )
        if loss_reduction_fraction < 0.95:
            failures.append(
                "tiny-overfit loss reduction is below 95%"
            )
        if stable_exact_steps < REQUIRED_STABLE_STEPS:
            failures.append(
                f"stable exact steps={stable_exact_steps}, "
                f"expected {REQUIRED_STABLE_STEPS}"
            )
        if maximum_gradient_norm <= 0:
            failures.append("no nonzero aggregate gradient observed")
        if nonzero_gradient_parameter_count <= 0:
            failures.append("no parameter received a nonzero gradient")

    history_path = (
        output_dir
        / "V5_P2_A4_TINY_OVERFIT_HISTORY.csv"
    )
    write_csv(history_path, history)

    selection_path = (
        output_dir
        / "V5_P2_A4_TINY_OVERFIT_SELECTION.json"
    )
    write_json(
        selection_path,
        {
            "seed": SEED,
            "selection_policy": (
                "two active final pair-aligned windows per "
                "K1/K2/K4 category plus matched controls"
            ),
            "selected_pairs": selected_pairs,
            "selected_dataset_indices": selected_indices,
            "positive_raw_attacker_counts": positive_raw_counts,
            "count_class_mapping": {
                str(key): value
                for key, value in count_class_mapping.items()
            },
            "model_inputs": ["x", "physical_port_mask"],
            "targets": [
                "y_attack",
                "y_attacker_count",
                "y_source",
                "y_transit",
                "y_victim",
                "y_attack_path",
            ],
            "provenance_used_as_model_input": False,
        },
    )

    status = "COMPLETE" if not failures else "HOLD"
    report = {
        "stage": STAGE,
        "status": status,
        "decision": (
            "AUTHORIZE_P2_B0_NONTEST_LABEL_AND_SHORTCUT_AUDIT"
            if not failures
            else "BLOCK_P2_B0"
        ),
        "purpose": (
            "loader/model/loss integration test only; "
            "not final P2 training"
        ),
        "dataset": {
            "link_path": str(root_input.absolute()),
            "resolved_root": str(root),
        },
        "architecture": {
            "name": model.architecture_name,
            "parameter_count": parameter_count,
            "trainable_parameter_count": (
                trainable_parameter_count
            ),
            "graph_message_passing": False,
            "edge_index_model_input": False,
            "x_shape": [12, 16, 58, 32],
            "physical_port_mask_shape": [12, 16, 10],
        },
        "tiny_subset": {
            "item_count": len(selected_indices),
            "active_attack_count": int(active_mask.sum().item()),
            "control_count": int((~active_mask).sum().item()),
            "selected_pairs": selected_pairs,
            "positive_raw_attacker_counts": positive_raw_counts,
            "count_class_mapping": {
                str(key): value
                for key, value in count_class_mapping.items()
            },
        },
        "protocol": {
            "seed": SEED,
            "device": str(device),
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "maximum_steps": MAX_STEPS,
            "required_stable_exact_steps": REQUIRED_STABLE_STEPS,
            "loss_weights": LOSS_WEIGHTS,
            "audit_weights_saved": False,
            "audit_weights_authorized_for_reuse": False,
        },
        "results": {
            "completed_step": completed_step,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction_fraction": loss_reduction_fraction,
            "maximum_gradient_norm": maximum_gradient_norm,
            "maximum_nonzero_gradient_parameter_tensors": (
                nonzero_gradient_parameter_count
            ),
            "stable_exact_steps": stable_exact_steps,
            "final_metrics": final_metrics,
        },
        "artifacts": {
            "history_csv": [
                str(history_path),
                sha256_file(history_path),
            ],
            "selection_json": [
                str(selection_path),
                sha256_file(selection_path),
            ],
        },
        "provenance": {
            "a1_r2_report_sha256": sha256_file(
                paths["a1_r2_report"]
            ),
            "a1_r2_lock_sha256": sha256_file(
                paths["a1_r2_lock"]
            ),
            "a2_r2_report_sha256": sha256_file(
                paths["a2_r2_report"]
            ),
            "a2_r2_lock_sha256": sha256_file(
                paths["a2_r2_lock"]
            ),
            "a3_report_sha256": sha256_file(
                paths["a3_report"]
            ),
            "a3_lock_sha256": sha256_file(paths["a3_lock"]),
            "loader_sha256": sha256_file(paths["loader"]),
            "model_sha256": sha256_file(paths["model"]),
        },
        "security_boundary": {
            "training_performed": True,
            "training_scope": "fixed 12-item train-only audit batch",
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
        },
        "failures": failures,
        "warnings": warnings,
        "next_stage": (
            "V5_P2_B0_NONTEST_LABEL_AND_SHORTCUT_AUDIT"
            if not failures
            else None
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    if failures:
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print("===== V5 P2-A4 LOADER INTEGRATION + TINY OVERFIT =====")
        print("status: HOLD")
        print("completed_step:", completed_step)
        print("initial_loss:", initial_loss)
        print("final_loss:", final_loss)
        print("stable_exact_steps:", stable_exact_steps)
        print("failure_count:", len(failures))
        for failure in failures:
            print("FAIL:", failure)
        print("warning_count:", len(warnings))
        print(f"{STAGE}_HOLD")
        return 1

    lock = {
        "status": COMPLETE,
        "decision": (
            "AUTHORIZE_P2_B0_NONTEST_LABEL_AND_SHORTCUT_AUDIT"
        ),
        "report_sha256": sha256_file(report_path),
        "history_sha256": sha256_file(history_path),
        "selection_sha256": sha256_file(selection_path),
        "architecture_name": model.architecture_name,
        "parameter_count": parameter_count,
        "tiny_item_count": len(selected_indices),
        "active_attack_count": int(active_mask.sum().item()),
        "control_count": int((~active_mask).sum().item()),
        "positive_raw_attacker_counts": positive_raw_counts,
        "count_class_mapping": {
            str(key): value
            for key, value in count_class_mapping.items()
        },
        "completed_step": completed_step,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_fraction": loss_reduction_fraction,
        "all_tasks_all_exact": (
            final_metrics["all_tasks_all_exact"]
        ),
        "stable_exact_steps": stable_exact_steps,
        "audit_weights_saved": False,
        "audit_weights_authorized_for_reuse": False,
        "validation_tensor_contents_accessed": False,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "next_stage": (
            "V5_P2_B0_NONTEST_LABEL_AND_SHORTCUT_AUDIT"
        ),
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-A4 LOADER INTEGRATION + TINY OVERFIT =====")
    print("status: COMPLETE")
    print(
        "decision: "
        "AUTHORIZE_P2_B0_NONTEST_LABEL_AND_SHORTCUT_AUDIT"
    )
    print("architecture:", model.architecture_name)
    print("parameter_count:", parameter_count)
    print("device:", device)
    print("tiny_item_count:", len(selected_indices))
    print("active_attack_count:", int(active_mask.sum().item()))
    print("control_count:", int((~active_mask).sum().item()))
    print("positive_raw_attacker_counts:", positive_raw_counts)
    print("count_class_mapping:", count_class_mapping)
    print("completed_step:", completed_step)
    print("initial_loss:", f"{initial_loss:.10g}")
    print("final_loss:", f"{final_loss:.10g}")
    print(
        "loss_reduction_fraction:",
        f"{loss_reduction_fraction:.10g}",
    )
    print("maximum_gradient_norm:", f"{maximum_gradient_norm:.10g}")
    print(
        "all_tasks_all_exact:",
        str(final_metrics["all_tasks_all_exact"]).lower(),
    )
    print("stable_exact_steps:", stable_exact_steps)
    print("audit_weights_saved: false")
    print("audit_weights_authorized_for_reuse: false")
    print("validation_tensor_contents_accessed: false")
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print(
        "next_stage: "
        "V5_P2_B0_NONTEST_LABEL_AND_SHORTCUT_AUDIT"
    )
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
