from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


STAGE = "V5_P3_A3_PRELIMINARY_BASELINE_PREFLIGHT"
CAMPAIGN_LABEL = "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic"
EXPECTED_TRAIN_LENGTH = 110_855
EXPECTED_VALIDATION_LENGTH = 13_863
EXPECTED_PARAMETER_COUNT = 60_553
SEED = 107


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exact(sample: dict[str, Any], key: str) -> Any:
    if key not in sample:
        raise RuntimeError(
            f"certified sample key missing: {key}; "
            f"keys={sorted(sample)}"
        )
    return sample[key]


def pair_id(sample: dict[str, Any]) -> str:
    for key in ("pair_id", "pair_key", "pair"):
        if key in sample:
            return str(sample[key])
    raise RuntimeError(
        "certified sample does not expose pair identity; "
        f"keys={sorted(sample)}"
    )


def canonical(sample: dict[str, Any]) -> dict[str, Any]:
    x = torch.as_tensor(exact(sample, "x")).detach().cpu()
    mask = torch.as_tensor(
        exact(sample, "physical_port_mask")
    ).detach().cpu()
    edge = torch.as_tensor(
        exact(sample, "edge_index")
    ).detach().cpu()
    y_attack = torch.as_tensor(
        exact(sample, "y_attack")
    ).reshape(()).detach().cpu()
    y_count = torch.as_tensor(
        exact(sample, "y_attacker_count")
    ).reshape(()).detach().cpu()

    labels = {
        role: torch.as_tensor(
            exact(sample, key)
        ).detach().cpu()
        for role, key in {
            "source": "y_source",
            "transit": "y_transit",
            "victim": "y_victim",
            "path": "y_attack_path",
        }.items()
    }

    if tuple(x.shape) != (16, 70, 32):
        raise RuntimeError(f"x shape={tuple(x.shape)}")
    if tuple(mask.shape) != (16, 10):
        raise RuntimeError(f"mask shape={tuple(mask.shape)}")
    if tuple(edge.shape) != (2, 48):
        raise RuntimeError(f"edge shape={tuple(edge.shape)}")
    for role, value in labels.items():
        if tuple(value.shape) != (16,):
            raise RuntimeError(
                f"{role} shape={tuple(value.shape)}"
            )

    graph = int(y_attack.item())
    count = int(y_count.item())
    if graph not in (0, 1):
        raise RuntimeError(f"invalid graph label={graph}")
    if count not in (0, 1, 2, 3, 4):
        raise RuntimeError(f"invalid count label={count}")
    if graph == 1 and count not in (1, 2, 3, 4):
        raise RuntimeError(
            f"active sample has invalid count={count}"
        )

    return {
        "x": x.float(),
        "mask": mask.float(),
        "edge": edge.long(),
        "graph": graph,
        "count": count,
        "pair_id": pair_id(sample),
        **{
            role: value.float()
            for role, value in labels.items()
        },
    }


def scan_split(dataset, split: str) -> dict[str, Any]:
    graph_counts = Counter()
    active_count_distribution = Counter()
    role_positive_entries = Counter()
    role_exact_nonempty = Counter()
    pair_windows = Counter()
    unique_pairs = set()

    for index in range(len(dataset)):
        item = canonical(dataset[index])
        graph_counts[item["graph"]] += 1
        if item["graph"] == 1:
            active_count_distribution[item["count"]] += 1
        for role in ("source", "transit", "victim", "path"):
            positives = int(item[role].sum().item())
            role_positive_entries[role] += positives
            if positives > 0:
                role_exact_nonempty[role] += 1
        unique_pairs.add(item["pair_id"])
        pair_windows[item["pair_id"]] += 1

        if (index + 1) % 10_000 == 0 or index + 1 == len(dataset):
            print(
                f"{split}_samples_scanned={index + 1}/{len(dataset)}"
            )

    if sorted(active_count_distribution) != [1, 2, 3, 4]:
        raise RuntimeError(
            f"{split} active counts are not 1..4: "
            f"{dict(active_count_distribution)}"
        )

    role_stats = {}
    total_node_entries = len(dataset) * 16
    for role in ("source", "transit", "victim", "path"):
        positives = int(role_positive_entries[role])
        negatives = total_node_entries - positives
        if positives <= 0:
            raise RuntimeError(
                f"{split} has no positive {role} entries"
            )
        raw_weight = negatives / positives
        role_stats[role] = {
            "positive_entries": positives,
            "negative_entries": negatives,
            "windows_with_positive_label": int(
                role_exact_nonempty[role]
            ),
            "raw_positive_weight": raw_weight,
            "canonical_clamped_positive_weight": min(
                20.0,
                max(1.0, raw_weight),
            ),
        }

    graph_positive = int(graph_counts[1])
    graph_negative = int(graph_counts[0])
    if graph_positive <= 0 or graph_negative <= 0:
        raise RuntimeError(
            f"{split} graph distribution invalid: "
            f"{dict(graph_counts)}"
        )

    window_counts = list(pair_windows.values())
    return {
        "split": split,
        "items": len(dataset),
        "unique_pairs": len(unique_pairs),
        "pair_ids": sorted(unique_pairs),
        "graph_positive": graph_positive,
        "graph_negative": graph_negative,
        "graph_positive_weight": (
            graph_negative / graph_positive
        ),
        "active_count_distribution": {
            str(key): int(value)
            for key, value in sorted(
                active_count_distribution.items()
            )
        },
        "role_statistics": role_stats,
        "windows_per_pair": {
            "minimum": min(window_counts),
            "maximum": max(window_counts),
            "mean": sum(window_counts) / len(window_counts),
        },
    }


def make_batch(dataset, indices: list[int]) -> dict[str, torch.Tensor]:
    items = [canonical(dataset[index]) for index in indices]
    edge = items[0]["edge"]
    if any(
        not torch.equal(edge, item["edge"])
        for item in items[1:]
    ):
        raise RuntimeError("edge_index differs within dry-run batch")
    return {
        "x": torch.stack([item["x"] for item in items]),
        "mask": torch.stack([item["mask"] for item in items]),
        "edge": edge,
        "graph": torch.tensor(
            [item["graph"] for item in items],
            dtype=torch.float32,
        ),
        "count": torch.tensor(
            [item["count"] for item in items],
            dtype=torch.long,
        ),
        **{
            role: torch.stack(
                [item[role] for item in items]
            )
            for role in (
                "source",
                "transit",
                "victim",
                "path",
            )
        },
    }


def dry_run_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    protocol: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    active = batch["graph"] > 0.5
    if not bool(active.any()):
        raise RuntimeError(
            "dry-run batch contains no active samples"
        )

    graph_weight = torch.tensor(
        protocol["loss"]["graph_positive_weight"],
        device=device,
        dtype=torch.float32,
    )
    role_weights = {
        role: torch.tensor(
            protocol["loss"]["role_positive_weights"][role],
            device=device,
            dtype=torch.float32,
        )
        for role in ("source", "transit", "victim", "path")
    }

    losses = {
        "graph": F.binary_cross_entropy_with_logits(
            outputs["attack_logits"],
            batch["graph"],
            pos_weight=graph_weight,
        ),
        "count": F.cross_entropy(
            outputs["count_logits"][active],
            batch["count"][active] - 1,
        ),
        **{
            role: F.binary_cross_entropy_with_logits(
                outputs[f"{role}_logits"],
                batch[role],
                pos_weight=role_weights[role],
            )
            for role in (
                "source",
                "transit",
                "victim",
                "path",
            )
        },
    }
    losses["total"] = sum(losses.values())
    if not all(
        torch.isfinite(value)
        for value in losses.values()
    ):
        raise RuntimeError("non-finite dry-run loss")
    return {
        key: float(value.detach().item())
        for key, value in losses.items()
    }


def main() -> int:
    args = parse_args()
    set_seed(SEED)

    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    data_root = data_link.resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(
        args.installed_script
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    protocol_path = output_dir / (
        "V5_P3_TRANCHE_A_PRELIMINARY_BASELINE_PROTOCOL.json"
    )
    train_stats_path = output_dir / (
        "V5_P3_A3_TRAIN_LABEL_AND_PAIR_STATISTICS.json"
    )
    validation_stats_path = output_dir / (
        "V5_P3_A3_VALIDATION_LABEL_AND_PAIR_STATISTICS.json"
    )
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    a2_dir = (
        repo
        / "reports/v5/p3_a2_train_only_tiny_overfit_integration_test"
    )
    a2_report_path = a2_dir / (
        "V5_P3_A2_TRAIN_ONLY_TINY_OVERFIT_"
        "INTEGRATION_TEST_REPORT.json"
    )
    a2_lock_path = a2_dir / (
        "V5_P3_A2_TRAIN_ONLY_TINY_OVERFIT_"
        "INTEGRATION_TEST_LOCK.json"
    )
    a1_wrapper_path = (
        repo / "src/data/v5_p3_tranche_a_guarded_loader.py"
    )
    b3_path = (
        repo / "src/models/v5_p2_b3_conv1d_only_count4.py"
    )
    canonical_path = (
        repo / "src/models/v5_p2_task_d_full_multitask_count4.py"
    )
    dynamic_path = (
        repo
        / "src/models/v6_p0_dynamic70_task_d_full_multitask_count4.py"
    )
    normalization_path = (
        data_root / "normalization_tranche_a_provisional.pt"
    )
    sufficient_statistics_path = (
        data_root
        / "normalization_sufficient_statistics_tranche_a.pt"
    )

    required = [
        a2_report_path,
        a2_lock_path,
        a1_wrapper_path,
        b3_path,
        canonical_path,
        dynamic_path,
        normalization_path,
        sufficient_statistics_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required files missing: {missing}")
    if not data_link.is_symlink():
        raise RuntimeError(
            f"dataset path is not canonical symlink: {data_link}"
        )

    a2_report = json.loads(
        a2_report_path.read_text(encoding="utf-8")
    )
    a2_lock = json.loads(
        a2_lock_path.read_text(encoding="utf-8")
    )
    if a2_report.get("status") != "PASS":
        raise RuntimeError("A2 report is not PASS")
    if not a2_report.get("decision", {}).get(
        "preliminary_baseline_preflight_authorized"
    ):
        raise RuntimeError(
            "A2 did not authorize A3 preflight"
        )
    if a2_report.get("sealed_test", {}).get(
        "test_tensor_loaded"
    ):
        raise RuntimeError(
            "A2 reports sealed-test tensor access"
        )
    if a2_lock.get("report_sha256") != sha256_file(
        a2_report_path
    ):
        raise RuntimeError("A2 report/lock SHA mismatch")

    # Process-wide sealed-test deserialization guard.
    original_torch_load = torch.load
    loaded_paths: list[str] = []

    def guarded_torch_load(file, *load_args, **load_kwargs):
        try:
            candidate = Path(
                os.fspath(file)
            ).expanduser().resolve()
        except TypeError:
            candidate = None
        if candidate is not None:
            candidate_text = str(candidate)
            loaded_paths.append(candidate_text)
            if "/runs/test/" in candidate_text:
                raise PermissionError(
                    "A3 sealed-test guard blocked "
                    f"{candidate_text}"
                )
        return original_torch_load(
            file,
            *load_args,
            **load_kwargs,
        )

    torch.load = guarded_torch_load

    wrapper_mod = import_module(
        a1_wrapper_path,
        "_v5_p3_a3_guarded_loader",
    )
    b3_mod = import_module(b3_path, "_v5_p3_a3_b3")
    canonical_mod = import_module(
        canonical_path,
        "_v5_p3_a3_canonical",
    )
    dynamic_mod = import_module(
        dynamic_path,
        "_v5_p3_a3_dynamic70",
    )

    GuardedDataset = (
        wrapper_mod.GuardedV5P3TrancheAPreliminaryDataset
    )
    SealedTestAccessError = (
        wrapper_mod.SealedTestAccessError
    )

    try:
        GuardedDataset(data_root, "test")
    except SealedTestAccessError:
        test_negative_check = True
    else:
        test_negative_check = False
    if not test_negative_check:
        raise RuntimeError("guard failed to reject A_test")

    train_dataset = GuardedDataset(
        data_root,
        "train",
        active_only=False,
    )
    validation_dataset = GuardedDataset(
        data_root,
        "validation",
        active_only=False,
    )
    if len(train_dataset) != EXPECTED_TRAIN_LENGTH:
        raise RuntimeError(
            f"train length={len(train_dataset)}"
        )
    if len(validation_dataset) != EXPECTED_VALIDATION_LENGTH:
        raise RuntimeError(
            f"validation length={len(validation_dataset)}"
        )

    print("===== TRAIN LABEL/PAIR SCAN =====")
    train_stats = scan_split(train_dataset, "train")
    print("===== VALIDATION LABEL/PAIR SCAN =====")
    validation_stats = scan_split(
        validation_dataset,
        "validation",
    )

    train_pairs = set(train_stats.pop("pair_ids"))
    validation_pairs = set(
        validation_stats.pop("pair_ids")
    )
    overlap = sorted(train_pairs & validation_pairs)
    if overlap:
        raise RuntimeError(
            f"train/validation pair overlap: {overlap[:20]}"
        )

    atomic_json(train_stats_path, train_stats)
    atomic_json(
        validation_stats_path,
        validation_stats,
    )

    protocol = {
        "stage": (
            "V5_P3_TRANCHE_A_PRELIMINARY_BASELINE_PROTOCOL"
        ),
        "status": "FROZEN",
        "campaign_label": CAMPAIGN_LABEL,
        "scientific_scope": {
            "result_label": CAMPAIGN_LABEL,
            "preliminary_only": True,
            "final_1500_pair_claim": False,
            "generalization_claim_scope": (
                "A_train/A_validation only"
            ),
            "sealed_test_evaluation_authorized": False,
        },
        "data": {
            "representation": "Dynamic70",
            "input_shape": [16, 70, 32],
            "physical_mask_shape": [16, 10],
            "edge_index_shape": [2, 48],
            "train_items": EXPECTED_TRAIN_LENGTH,
            "validation_items": (
                EXPECTED_VALIDATION_LENGTH
            ),
            "train_pairs": train_stats["unique_pairs"],
            "validation_pairs": (
                validation_stats["unique_pairs"]
            ),
            "pair_split_overlap": 0,
            "normalization": (
                "Tranche-A provisional A_train-only"
            ),
            "normalization_path": str(
                normalization_path
            ),
        },
        "model": {
            "builder": (
                "build_v6_p0_dynamic70_from_"
                "canonical_structure"
            ),
            "seed": SEED,
            "fresh_initialization": True,
            "all_parameters_trainable": True,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "architecture": (
                "Causal Depthwise-Separable Conv1D Temporal "
                "Encoder + Two-Layer GraphConv"
            ),
            "graph_readout": "mean_plus_max",
            "outputs": [
                "attack_logits",
                "count_logits",
                "source_logits",
                "transit_logits",
                "victim_logits",
                "path_logits",
            ],
        },
        "loss": {
            "task_coefficients": {
                "graph": 1.0,
                "count": 1.0,
                "source": 1.0,
                "transit": 1.0,
                "victim": 1.0,
                "path": 1.0,
            },
            "graph_positive_weight": (
                train_stats["graph_positive_weight"]
            ),
            "role_positive_weights": {
                role: train_stats[
                    "role_statistics"
                ][role][
                    "canonical_clamped_positive_weight"
                ]
                for role in (
                    "source",
                    "transit",
                    "victim",
                    "path",
                )
            },
            "count_class_weights_applied": False,
            "count_semantics": {
                "inactive_target_zero": "masked",
                "active_mapping": "target_minus_one",
                "class_values": [1, 2, 3, 4],
            },
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
            "maximum_epochs": 100,
            "batch_size": 256,
            "num_workers": 0,
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "mode": "max",
                "factor": 0.5,
                "patience": 4,
                "threshold": 1e-4,
                "minimum_lr": 1e-5,
            },
            "early_stopping": {
                "minimum_epochs": 15,
                "patience": 12,
                "minimum_improvement": 1e-4,
            },
        },
        "selection": {
            "threshold_tuning_during_training": False,
            "checkpoint_selection_uses_validation": True,
            "test_used_for_selection": False,
            "primary_metric_family": (
                "frozen multitask validation composite"
            ),
            "tie_break_order": [
                "lower_validation_loss",
                "earlier_epoch",
            ],
        },
        "prohibited_before_baseline_completion": [
            "architecture changes",
            "new pooling",
            "attention",
            "directional graph layers",
            "multiscale graph hierarchy",
            "new labels",
            "new localization losses",
            "A_test access",
        ],
    }
    atomic_json(protocol_path, protocol)

    # Dry run: one mixed train batch and one validation batch.
    train_indices = list(range(16))
    # Add deterministic spaced candidates until an active item exists.
    if not any(
        canonical(train_dataset[index])["graph"] == 1
        for index in train_indices
    ):
        for index in np.linspace(
            0,
            len(train_dataset) - 1,
            256,
            dtype=np.int64,
        ):
            index = int(index)
            if canonical(train_dataset[index])["graph"] == 1:
                train_indices[-1] = index
                break

    validation_indices = list(range(16))
    if not any(
        canonical(validation_dataset[index])["graph"] == 1
        for index in validation_indices
    ):
        for index in np.linspace(
            0,
            len(validation_dataset) - 1,
            128,
            dtype=np.int64,
        ):
            index = int(index)
            if canonical(validation_dataset[index])["graph"] == 1:
                validation_indices[-1] = index
                break

    train_batch = make_batch(
        train_dataset,
        train_indices,
    )
    validation_batch = make_batch(
        validation_dataset,
        validation_indices,
    )
    edge = train_batch["edge"]

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    canonical_model = (
        canonical_mod.P2TaskDGraphConvCount4(
            reference_b3,
            edge,
        )
    )
    model = dynamic_mod.build_v6_p0_dynamic70_from_canonical_structure(
        canonical_model,
        seed=SEED,
    )
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"parameter count={parameter_count}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    def move(batch):
        return {
            key: (
                value.to(device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in batch.items()
        }

    train_device = move(train_batch)
    validation_device = move(validation_batch)

    model.train()
    train_outputs = model(
        train_device["x"],
        train_device["mask"],
    )
    train_losses = dry_run_loss(
        train_outputs,
        train_device,
        protocol,
        device,
    )
    model.zero_grad(set_to_none=True)
    total_train_loss = sum(
        F.binary_cross_entropy_with_logits(
            train_outputs[f"{role}_logits"],
            train_device[role],
        )
        for role in (
            "source",
            "transit",
            "victim",
            "path",
        )
    )
    total_train_loss = (
        total_train_loss
        + F.binary_cross_entropy_with_logits(
            train_outputs["attack_logits"],
            train_device["graph"],
        )
    )
    active = train_device["graph"] > 0.5
    total_train_loss = (
        total_train_loss
        + F.cross_entropy(
            train_outputs["count_logits"][active],
            train_device["count"][active] - 1,
        )
    )
    total_train_loss.backward()
    nonzero_gradient_parameters = sum(
        1
        for parameter in model.parameters()
        if parameter.grad is not None
        and float(
            parameter.grad.detach().abs().sum().item()
        ) > 0.0
    )
    if nonzero_gradient_parameters == 0:
        raise RuntimeError(
            "dry-run backward produced no gradients"
        )

    model.eval()
    with torch.no_grad():
        validation_outputs = model(
            validation_device["x"],
            validation_device["mask"],
        )
        validation_losses = dry_run_loss(
            validation_outputs,
            validation_device,
            protocol,
            device,
        )

    test_loaded_paths = [
        path
        for path in loaded_paths
        if "/runs/test/" in path
    ]
    if test_loaded_paths:
        raise RuntimeError(
            f"test paths loaded: {test_loaded_paths}"
        )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "data": {
            "canonical_link": str(
                data_link.absolute()
            ),
            "resolved_root": str(data_root),
            "train_items": len(train_dataset),
            "validation_items": len(
                validation_dataset
            ),
            "train_pairs": train_stats["unique_pairs"],
            "validation_pairs": (
                validation_stats["unique_pairs"]
            ),
            "train_validation_pair_overlap": 0,
            "test_dataset_instantiated": False,
            "test_tensor_loaded": False,
        },
        "model": {
            "class": model.__class__.__name__,
            "input_features": (
                model.input_projection.in_channels
            ),
            "parameter_count": parameter_count,
            "fresh_seed": SEED,
            "all_parameters_trainable": all(
                parameter.requires_grad
                for parameter in model.parameters()
            ),
        },
        "normalization": {
            "provisional": True,
            "scope": "A_train only",
            "normalization_sha256": sha256_file(
                normalization_path
            ),
            "sufficient_statistics_sha256": (
                sha256_file(
                    sufficient_statistics_path
                )
            ),
            "final_union_normalization": (
                "deferred until A_train+B_train"
            ),
        },
        "label_statistics": {
            "train": str(train_stats_path),
            "validation": str(
                validation_stats_path
            ),
        },
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "dry_run": {
            "device": str(device),
            "train_batch_size": len(train_indices),
            "validation_batch_size": len(
                validation_indices
            ),
            "train_losses": train_losses,
            "validation_losses": (
                validation_losses
            ),
            "nonzero_gradient_parameters": (
                nonzero_gradient_parameters
            ),
            "optimizer_step_performed": False,
        },
        "sealed_test": {
            "negative_guard_check": (
                test_negative_check
            ),
            "test_dataset_instantiated": False,
            "test_length_computed": False,
            "test_tensor_loaded": False,
            "evaluation_authorized": False,
        },
        "decision": {
            "preliminary_baseline_preflight_passed": True,
            "preliminary_baseline_training_authorized": True,
            "validation_monitoring_authorized": True,
            "sealed_test_evaluation_authorized": False,
            "next_stage": (
                "V5_P3_A4_TRANCHE_A_PRELIMINARY_"
                "DIAGNOSTIC_SEED107"
            ),
        },
        "provenance": {
            "a2_report_sha256": sha256_file(
                a2_report_path
            ),
            "a2_lock_sha256": sha256_file(
                a2_lock_path
            ),
            "guarded_loader_sha256": sha256_file(
                a1_wrapper_path
            ),
            "b3_source_sha256": sha256_file(b3_path),
            "canonical_model_sha256": sha256_file(
                canonical_path
            ),
            "dynamic70_model_sha256": sha256_file(
                dynamic_path
            ),
            "installed_script_sha256": sha256_file(
                installed_script
            ),
        },
        "source_modified": True,
        "certified_dataset_modified": False,
        "model_trained": False,
        "generalization_claim_authorized": False,
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "campaign_label": CAMPAIGN_LABEL,
        "report_sha256": sha256_file(report_path),
        "protocol_sha256": sha256_file(
            protocol_path
        ),
        "train_statistics_sha256": sha256_file(
            train_stats_path
        ),
        "validation_statistics_sha256": sha256_file(
            validation_stats_path
        ),
        "dynamic70_model_sha256": sha256_file(
            dynamic_path
        ),
        "guarded_loader_sha256": sha256_file(
            a1_wrapper_path
        ),
        "normalization_sha256": sha256_file(
            normalization_path
        ),
        "parameter_count": parameter_count,
        "input_features": 70,
        "train_items": len(train_dataset),
        "validation_items": len(
            validation_dataset
        ),
        "test_tensor_loaded": False,
        "certified_dataset_modified": False,
    }
    atomic_json(lock_path, lock)
    complete_path.write_text(
        f"{STAGE}_COMPLETE\n",
        encoding="utf-8",
    )

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign_label={CAMPAIGN_LABEL}")
    print(f"train_items={len(train_dataset)}")
    print(
        f"validation_items={len(validation_dataset)}"
    )
    print(f"train_pairs={train_stats['unique_pairs']}")
    print(
        "validation_pairs="
        f"{validation_stats['unique_pairs']}"
    )
    print("train_validation_pair_overlap=0")
    print("representation=Dynamic70")
    print("input_features=70")
    print(f"parameter_count={parameter_count}")
    print("fresh_initialization=true")
    print("all_parameters_trainable=true")
    print(
        "normalization_scope="
        "Tranche-A_provisional_A_train_only"
    )
    print(
        "dry_run_nonzero_gradient_parameters="
        f"{nonzero_gradient_parameters}"
    )
    print("optimizer_step_performed=false")
    print("test_dataset_instantiated=false")
    print("test_length_computed=false")
    print("test_tensor_loaded=false")
    print(
        "preliminary_baseline_training_authorized=true"
    )
    print(
        "sealed_test_evaluation_authorized=false"
    )
    print(
        "next_stage="
        "V5_P3_A4_TRANCHE_A_PRELIMINARY_"
        "DIAGNOSTIC_SEED107"
    )
    print(f"protocol={protocol_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
