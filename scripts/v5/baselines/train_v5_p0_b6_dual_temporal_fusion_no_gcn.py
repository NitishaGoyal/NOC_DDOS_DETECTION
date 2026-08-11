#!/usr/bin/env python3
"""
V5 P0-B6 Dual Temporal Fusion, No GCN

Scientific question
-------------------
Can a single-stage router-wise model combine:
- B2's stable order-free full-window evidence; and
- B3's learned causal temporal evidence;

without graph smoothing?

Frozen learned inputs
---------------------
- PRIMARY58 x over all 32 epochs;
- topology-derived raw Boolean physical_port_mask.

Architecture
------------
Branch A:
    temporal mean over 32 epochs
    -> Linear(58, 64)
    -> ReLU

Branch B:
    same causal depthwise-separable Conv1D encoder as B3
    with dilations 1, 2, 4, 8
    -> final causal encoded time step

Fusion:
    concatenate mean branch, Conv1D branch, and recovered Boolean mask
    -> shared router-wise Linear(138, 64)
    -> ReLU

Outputs:
    shared node heads for source, transit, victim, and path;
    mean+max router pooling;
    graph heads for attack and attacker count.

Explicitly unused
-----------------
- edge_index and graph message passing;
- stored standardized mask channels;
- identifiers, filenames, mode, seed, scenario metadata;
- run/window position, serialization fields, and their derivatives.

Protocol
--------
- Same seed, optimizer, epochs, batch size, learning rate, loss weights,
  threshold, and fixed-final-epoch evaluation protocol as B1-B5.
- TRAIN fitting only.
- VALIDATION evaluated once after the fixed final epoch.
- No early stopping, validation checkpoint selection, or threshold search.
- TEST is never constructed, enumerated, read, or evaluated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROLE_KEYS = {
    "source": "y_source",
    "transit": "y_transit",
    "victim": "y_victim",
    "path": "y_attack_path",
}
ROLES = tuple(ROLE_KEYS)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def import_contract_dataset(wrapper_path: Path):
    spec = importlib.util.spec_from_file_location(
        "v5_p0_contract_loader_b6",
        wrapper_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import wrapper: {wrapper_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset_class = getattr(module, "V5P0ContractWindowDataset", None)
    if dataset_class is None:
        raise AttributeError(
            "wrapper does not define V5P0ContractWindowDataset"
        )
    return dataset_class


def verify_prerequisites(
    a2_dir: Path,
    a3_dir: Path,
    r1c_dir: Path,
    baseline_dirs: dict[str, Path],
    wrapper_path: Path,
) -> tuple[
    list[str],
    list[str],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    failures: list[str] = []
    warnings: list[str] = []

    paths: dict[str, Path] = {
        "a2_report": (
            a2_dir / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT.json"
        ),
        "a2_marker": (
            a2_dir / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_PASS"
        ),
        "a3_report": (
            a3_dir / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE.json"
        ),
        "a3_lock": (
            a3_dir / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_LOCK.json"
        ),
        "a3_marker": (
            a3_dir / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_PASS"
        ),
        "r1c_report": (
            r1c_dir / "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE.json"
        ),
        "r1c_lock": (
            r1c_dir / "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_LOCK.json"
        ),
        "r1c_marker": (
            r1c_dir / "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_PASS"
        ),
        "wrapper": wrapper_path,
    }

    stage_files = {
        "b1": (
            "V5_P0_B1_STATIC_FINAL_EPOCH_MLP.json",
            "V5_P0_B1_STATIC_FINAL_EPOCH_MLP_LOCK.json",
            "V5_P0_B1_STATIC_FINAL_EPOCH_MLP_COMPLETE",
        ),
        "b2": (
            "V5_P0_B2_TEMPORAL_MEAN_POOLING.json",
            "V5_P0_B2_TEMPORAL_MEAN_POOLING_LOCK.json",
            "V5_P0_B2_TEMPORAL_MEAN_POOLING_COMPLETE",
        ),
        "b3": (
            "V5_P0_B3_CONV1D_ONLY.json",
            "V5_P0_B3_CONV1D_ONLY_LOCK.json",
            "V5_P0_B3_CONV1D_ONLY_COMPLETE",
        ),
        "b4": (
            "V5_P0_B4_GCN_ONLY.json",
            "V5_P0_B4_GCN_ONLY_LOCK.json",
            "V5_P0_B4_GCN_ONLY_COMPLETE",
        ),
        "b5": (
            "V5_P0_B5_CONV1D_GCN.json",
            "V5_P0_B5_CONV1D_GCN_LOCK.json",
            "V5_P0_B5_CONV1D_GCN_COMPLETE",
        ),
    }

    for stage, directory in baseline_dirs.items():
        report_name, lock_name, marker_name = stage_files[stage]
        paths[f"{stage}_report"] = directory / report_name
        paths[f"{stage}_lock"] = directory / lock_name
        paths[f"{stage}_marker"] = directory / marker_name

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    if failures:
        return failures, warnings, {}, {}

    a2 = load_json(paths["a2_report"])
    a3 = load_json(paths["a3_report"])
    a3_lock = load_json(paths["a3_lock"])
    r1c = load_json(paths["r1c_report"])
    r1c_lock = load_json(paths["r1c_lock"])

    reports = {
        stage: load_json(paths[f"{stage}_report"])
        for stage in baseline_dirs
    }
    locks = {
        stage: load_json(paths[f"{stage}_lock"])
        for stage in baseline_dirs
    }

    expected_status = (
        ("A2", a2, "PASS"),
        ("A3", a3, "PASS"),
        ("B0-R1C", r1c, "PASS"),
    )
    for name, report, expected in expected_status:
        if report.get("status") != expected:
            failures.append(
                f"{name} status is {report.get('status')!r}, "
                f"expected {expected!r}"
            )

    for stage, report in reports.items():
        if report.get("status") != "COMPLETE":
            failures.append(
                f"{stage.upper()} status is {report.get('status')!r}, "
                "expected 'COMPLETE'"
            )

    if a3_lock.get("report_sha256") != sha256_file(paths["a3_report"]):
        failures.append("A3 report SHA does not match A3 lock")
    if r1c_lock.get("report_sha256") != sha256_file(paths["r1c_report"]):
        failures.append("B0-R1C report SHA does not match B0-R1C lock")

    for stage in baseline_dirs:
        if (
            locks[stage].get("report_sha256")
            != sha256_file(paths[f"{stage}_report"])
        ):
            failures.append(
                f"{stage.upper()} report SHA does not match lock"
            )

    loader_contract = a2.get("loader_contract", {})
    if loader_contract.get("primary_feature_variant") != "PRIMARY58":
        failures.append("A2 primary feature variant is not PRIMARY58")
    if loader_contract.get("metadata_concatenated_to_x") is not False:
        failures.append("A2 does not forbid metadata in x")
    if loader_contract.get("raw_mask_supplied_separately") is not True:
        failures.append("A2 does not require separate physical mask")

    if a3.get("feature_variant") != "PRIMARY58":
        failures.append("A3 did not validate PRIMARY58")
    if int(a3.get("window", -1)) != 32:
        failures.append("A3 window is not 32")
    if int(a3.get("stride", -1)) != 8:
        failures.append("A3 stride is not 8")
    if a3.get("normalization_applied_by_wrapper") is not False:
        failures.append("A3 wrapper unexpectedly applies normalization")

    expected_protocol = {
        "seed": 101,
        "epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "binary_threshold": 0.5,
        "early_stopping": False,
        "validation_checkpoint_selection": False,
    }
    for stage, report in reports.items():
        protocol = report.get("protocol", {})
        for key, expected in expected_protocol.items():
            if protocol.get(key) != expected:
                failures.append(
                    f"{stage.upper()} protocol mismatch for {key}: "
                    f"{protocol.get(key)!r} != {expected!r}"
                )

    expected_parameters = {
        "b1": 25544,
        "b2": 25544,
        "b3": 43208,
        "b4": 25544,
        "b5": 47368,
    }
    for stage, expected in expected_parameters.items():
        actual = reports[stage].get("model", {}).get("parameter_count")
        if actual != expected:
            failures.append(
                f"{stage.upper()} parameter count is {actual!r}, "
                f"expected {expected}"
            )

    legacy_next = reports["b5"].get("next_stage")
    allowed_next = {
        "V5_P0_B6_ARCHITECTURE_DECISION",
        "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN",
    }
    if legacy_next not in allowed_next:
        warnings.append(
            f"B5 next_stage is {legacy_next!r}; proceeding because B6 "
            "is the preregistered architecture-decision experiment"
        )

    provenance = {
        "a2_report_sha256": sha256_file(paths["a2_report"]),
        "a3_report_sha256": sha256_file(paths["a3_report"]),
        "a3_lock_sha256": sha256_file(paths["a3_lock"]),
        "r1c_report_sha256": sha256_file(paths["r1c_report"]),
        "r1c_lock_sha256": sha256_file(paths["r1c_lock"]),
        "wrapper_sha256": sha256_file(wrapper_path),
    }
    for stage in baseline_dirs:
        provenance[f"{stage}_report_sha256"] = sha256_file(
            paths[f"{stage}_report"]
        )
        provenance[f"{stage}_lock_sha256"] = sha256_file(
            paths[f"{stage}_lock"]
        )

    return failures, warnings, provenance, reports


class CausalDepthwiseResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.left_padding,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        y = self.depthwise(x)
        y = y[..., :length]
        y = F.relu(y)
        y = self.pointwise(y)
        y = F.relu(y)
        return x + y


class DualTemporalFusionModel(nn.Module):
    def __init__(
        self,
        feature_count: int = 58,
        temporal_channels: int = 64,
        mean_channels: int = 64,
        mask_count: int = 10,
        fusion_channels: int = 64,
        graph_hidden: int = 64,
        role_hidden: int = 32,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Conv1d(
            feature_count,
            temporal_channels,
            kernel_size=1,
        )
        self.temporal_blocks = nn.ModuleList(
            [
                CausalDepthwiseResidualBlock(
                    channels=temporal_channels,
                    kernel_size=3,
                    dilation=dilation,
                )
                for dilation in (1, 2, 4, 8)
            ]
        )

        self.mean_projection = nn.Sequential(
            nn.Linear(feature_count, mean_channels),
            nn.ReLU(),
        )

        self.node_fusion = nn.Sequential(
            nn.Linear(
                temporal_channels + mean_channels + mask_count,
                fusion_channels,
            ),
            nn.ReLU(),
        )

        self.role_heads = nn.ModuleDict(
            {
                role: nn.Sequential(
                    nn.Linear(fusion_channels, role_hidden),
                    nn.ReLU(),
                    nn.Linear(role_hidden, 1),
                )
                for role in ROLES
            }
        )

        self.graph_encoder = nn.Sequential(
            nn.Linear(fusion_channels * 2, graph_hidden),
            nn.ReLU(),
        )
        self.attack_head = nn.Linear(graph_hidden, 1)
        self.count_head = nn.Linear(graph_hidden, 3)

    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1:] != (16, 58, 32):
            raise ValueError(
                f"expected x [B,16,58,32], got {tuple(x.shape)}"
            )
        if (
            physical_port_mask.ndim != 3
            or physical_port_mask.shape[1:] != (16, 10)
        ):
            raise ValueError(
                "expected physical_port_mask [B,16,10], got "
                f"{tuple(physical_port_mask.shape)}"
            )

        batch_size, router_count, feature_count, window = x.shape

        # Branch A: B2-style stable order-free evidence.
        temporal_mean = x.mean(dim=-1)
        h_mean = self.mean_projection(temporal_mean)

        # Branch B: B3-style learned causal temporal evidence.
        temporal = x.reshape(
            batch_size * router_count,
            feature_count,
            window,
        )
        temporal = F.relu(self.input_projection(temporal))
        for block in self.temporal_blocks:
            temporal = block(temporal)
        h_conv = temporal[..., -1].reshape(
            batch_size,
            router_count,
            -1,
        )

        mask = physical_port_mask.to(
            device=x.device,
            dtype=x.dtype,
        )

        node_repr = self.node_fusion(
            torch.cat([h_mean, h_conv, mask], dim=-1)
        )

        graph_repr = torch.cat(
            [
                node_repr.mean(dim=1),
                node_repr.max(dim=1).values,
            ],
            dim=-1,
        )
        graph_repr = self.graph_encoder(graph_repr)

        outputs = {
            "attack": self.attack_head(graph_repr).squeeze(-1),
            "count": self.count_head(graph_repr),
        }
        for role in ROLES:
            outputs[role] = (
                self.role_heads[role](node_repr).squeeze(-1)
            )
        return outputs


def target_statistics(loader: DataLoader) -> dict[str, Any]:
    total_windows = 0
    graph_positive = 0
    count_counts = torch.zeros(3, dtype=torch.long)
    role_positive = {role: 0 for role in ROLES}
    role_entries = {role: 0 for role in ROLES}

    for batch in loader:
        attack = batch["y_attack"].bool()
        count = batch["y_attacker_count"].long()
        total_windows += int(attack.numel())
        graph_positive += int(attack.sum().item())
        count_counts += torch.bincount(count, minlength=3)

        for role, key in ROLE_KEYS.items():
            target = batch[key].bool()
            role_positive[role] += int(target.sum().item())
            role_entries[role] += int(target.numel())

    return {
        "window_count": total_windows,
        "graph_positive_windows": graph_positive,
        "graph_negative_windows": total_windows - graph_positive,
        "count_class_counts": count_counts.tolist(),
        "role_positive_entries": role_positive,
        "role_total_entries": role_entries,
    }


def positive_weight(
    positives: int,
    total: int,
    cap: float = 20.0,
) -> float:
    negatives = total - positives
    if positives <= 0:
        return 1.0
    return min(cap, negatives / positives)


def build_positive_weights(
    stats: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    values = {
        "attack": positive_weight(
            stats["graph_positive_windows"],
            stats["window_count"],
        )
    }
    for role in ROLES:
        values[role] = positive_weight(
            stats["role_positive_entries"][role],
            stats["role_total_entries"][role],
        )

    tensors = {
        key: torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in values.items()
    }
    return tensors, values


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["attack"].device

    losses: dict[str, torch.Tensor] = {
        "attack": F.binary_cross_entropy_with_logits(
            outputs["attack"],
            batch["y_attack"].to(
                device=device,
                dtype=torch.float32,
            ),
            pos_weight=pos_weights["attack"],
        ),
        "count": F.cross_entropy(
            outputs["count"],
            batch["y_attacker_count"].to(
                device=device,
                dtype=torch.long,
            ),
        ),
    }

    for role, key in ROLE_KEYS.items():
        losses[role] = F.binary_cross_entropy_with_logits(
            outputs[role],
            batch[key].to(
                device=device,
                dtype=torch.float32,
            ),
            pos_weight=pos_weights[role],
        )

    total = (
        1.00 * losses["attack"]
        + 0.50 * losses["count"]
        + 1.00 * losses["source"]
        + 0.50 * losses["transit"]
        + 0.75 * losses["victim"]
        + 0.50 * losses["path"]
    )
    return total, {
        key: float(value.detach().item())
        for key, value in losses.items()
    }


def binary_metrics(
    truth: torch.Tensor,
    prediction: torch.Tensor,
) -> dict[str, Any]:
    truth = truth.bool()
    prediction = prediction.bool()

    tp = int((truth & prediction).sum().item())
    tn = int((~truth & ~prediction).sum().item())
    fp = int((~truth & prediction).sum().item())
    fn = int((truth & ~prediction).sum().item())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    f1 = (
        2 * precision * recall
        / max(1e-12, precision + recall)
    )

    return {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "balanced_accuracy": 0.5 * (recall + tnr),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / max(1, fp + tn),
        "tnr": tnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support_negative": tn + fp,
        "support_positive": tp + fn,
    }


def multiclass_metrics(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    class_count: int = 3,
) -> dict[str, Any]:
    truth = truth.long()
    prediction = prediction.long()
    confusion = torch.zeros(
        class_count,
        class_count,
        dtype=torch.long,
    )

    for actual, predicted in zip(
        truth.tolist(),
        prediction.tolist(),
    ):
        confusion[actual, predicted] += 1

    per_class = {}
    f1_values = []
    for label in range(class_count):
        tp = int(confusion[label, label].item())
        fp = int(confusion[:, label].sum().item()) - tp
        fn = int(confusion[label, :].sum().item()) - tp
        support = int(confusion[label, :].sum().item())

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (
            2 * precision * recall
            / max(1e-12, precision + recall)
        )
        f1_values.append(f1)
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "accuracy": float(
            (truth == prediction).float().mean().item()
        ),
        "macro_f1": sum(f1_values) / class_count,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def node_role_metrics(
    truth: torch.Tensor,
    prediction: torch.Tensor,
) -> dict[str, Any]:
    truth = truth.bool()
    prediction = prediction.bool()

    tp = int((truth & prediction).sum().item())
    tn = int((~truth & ~prediction).sum().item())
    fp = int((~truth & prediction).sum().item())
    fn = int((truth & ~prediction).sum().item())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        2 * precision * recall
        / max(1e-12, precision + recall)
    )
    exact = float(
        (truth == prediction)
        .all(dim=1)
        .float()
        .mean()
        .item()
    )

    return {
        "node_accuracy": (
            (tp + tn) / max(1, tp + tn + fp + fn)
        ),
        "node_precision": precision,
        "node_recall": recall,
        "node_f1": f1,
        "exact_set": exact,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_entries": int(truth.sum().item()),
        "window_count": int(truth.shape[0]),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()

    values: dict[str, list[torch.Tensor]] = {
        "attack_truth": [],
        "attack_prediction": [],
        "attack_probability": [],
        "count_truth": [],
        "count_prediction": [],
    }
    for role in ROLES:
        values[f"{role}_truth"] = []
        values[f"{role}_prediction"] = []
        values[f"{role}_probability"] = []

    for batch in loader:
        outputs = model(
            x=batch["x"].to(
                device=device,
                dtype=torch.float32,
            ),
            physical_port_mask=batch[
                "physical_port_mask"
            ].to(device=device),
        )

        attack_probability = torch.sigmoid(outputs["attack"])
        attack_prediction = attack_probability >= 0.5
        count_prediction = outputs["count"].argmax(dim=-1)

        values["attack_truth"].append(
            batch["y_attack"].detach().cpu().bool()
        )
        values["attack_probability"].append(
            attack_probability.detach().cpu()
        )
        values["attack_prediction"].append(
            attack_prediction.detach().cpu()
        )
        values["count_truth"].append(
            batch["y_attacker_count"].detach().cpu().long()
        )
        values["count_prediction"].append(
            count_prediction.detach().cpu()
        )

        for role, key in ROLE_KEYS.items():
            probability = torch.sigmoid(outputs[role])
            prediction = probability >= 0.5

            values[f"{role}_truth"].append(
                batch[key].detach().cpu().bool()
            )
            values[f"{role}_probability"].append(
                probability.detach().cpu()
            )
            values[f"{role}_prediction"].append(
                prediction.detach().cpu()
            )

    combined = {
        key: torch.cat(parts, dim=0)
        for key, parts in values.items()
    }

    attack_truth = combined["attack_truth"]
    metrics = {
        "graph": binary_metrics(
            attack_truth,
            combined["attack_prediction"],
        ),
        "count": multiclass_metrics(
            combined["count_truth"],
            combined["count_prediction"],
        ),
        "roles": {},
    }

    all_exact = (
        combined["attack_truth"]
        == combined["attack_prediction"]
    ) & (
        combined["count_truth"]
        == combined["count_prediction"]
    )

    for role in ROLES:
        truth = combined[f"{role}_truth"]
        prediction = combined[f"{role}_prediction"]
        active = attack_truth.bool()

        metrics["roles"][role] = {
            "all_windows": node_role_metrics(
                truth,
                prediction,
            ),
            "attack_windows": node_role_metrics(
                truth[active],
                prediction[active],
            ),
        }

        all_exact = all_exact & (
            truth == prediction
        ).all(dim=1)

    metrics["all_tasks_exact"] = float(
        all_exact.float().mean().item()
    )
    metrics["window_count"] = int(attack_truth.shape[0])

    return metrics, combined


def flatten_metric_rows(
    split: str,
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {
            "split": split,
            "task": "graph",
            **metrics["graph"],
        },
        {
            "split": split,
            "task": "attacker_count",
            "accuracy": metrics["count"]["accuracy"],
            "macro_f1": metrics["count"]["macro_f1"],
        },
        {
            "split": split,
            "task": "all_tasks",
            "exact_set": metrics["all_tasks_exact"],
        },
    ]

    for role in ROLES:
        for scope in ("all_windows", "attack_windows"):
            rows.append(
                {
                    "split": split,
                    "task": role,
                    "scope": scope,
                    **metrics["roles"][role][scope],
                }
            )
    return rows


def key_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "graph_accuracy": metrics["graph"]["accuracy"],
        "graph_balanced_accuracy": (
            metrics["graph"]["balanced_accuracy"]
        ),
        "graph_precision": metrics["graph"]["precision"],
        "graph_recall": metrics["graph"]["recall"],
        "graph_f1": metrics["graph"]["f1"],
        "graph_fpr": metrics["graph"]["fpr"],
        "graph_false_negatives": float(metrics["graph"]["fn"]),
        "count_macro_f1": metrics["count"]["macro_f1"],
        "source_node_f1_attack": (
            metrics["roles"]["source"]["attack_windows"]["node_f1"]
        ),
        "source_exact_attack": (
            metrics["roles"]["source"]["attack_windows"]["exact_set"]
        ),
        "transit_node_f1_attack": (
            metrics["roles"]["transit"]["attack_windows"]["node_f1"]
        ),
        "victim_node_f1_attack": (
            metrics["roles"]["victim"]["attack_windows"]["node_f1"]
        ),
        "victim_exact_attack": (
            metrics["roles"]["victim"]["attack_windows"]["exact_set"]
        ),
        "path_node_f1_attack": (
            metrics["roles"]["path"]["attack_windows"]["node_f1"]
        ),
        "all_tasks_exact": metrics["all_tasks_exact"],
    }


def metric_delta(
    current: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, float]:
    current_values = key_metrics(current)
    reference_values = key_metrics(reference)
    return {
        key: current_values[key] - reference_values[key]
        for key in current_values
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--a2-1-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--r1c-dir", type=Path, required=True)
    parser.add_argument("--b1-dir", type=Path, required=True)
    parser.add_argument("--b2-dir", type=Path, required=True)
    parser.add_argument("--b3-dir", type=Path, required=True)
    parser.add_argument("--b4-dir", type=Path, required=True)
    parser.add_argument("--b5-dir", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    a2_dir = args.a2_dir.expanduser().resolve()
    a2_1_dir = args.a2_1_dir.expanduser().resolve()
    a3_dir = args.a3_dir.expanduser().resolve()
    r1c_dir = args.r1c_dir.expanduser().resolve()
    wrapper_path = args.wrapper.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()

    baseline_dirs = {
        "b1": args.b1_dir.expanduser().resolve(),
        "b2": args.b2_dir.expanduser().resolve(),
        "b3": args.b3_dir.expanduser().resolve(),
        "b4": args.b4_dir.expanduser().resolve(),
        "b5": args.b5_dir.expanduser().resolve(),
    }

    if output_dir.exists():
        print(
            f"STOP: output directory already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    if model_dir.exists():
        print(
            f"STOP: model directory already exists: {model_dir}",
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    (
        failures,
        warnings,
        prerequisite_provenance,
        baseline_reports,
    ) = verify_prerequisites(
        a2_dir=a2_dir,
        a3_dir=a3_dir,
        r1c_dir=r1c_dir,
        baseline_dirs=baseline_dirs,
        wrapper_path=wrapper_path,
    )

    if failures:
        report = {
            "stage": "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN",
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "test_split_accessed": False,
        }
        write_json(
            output_dir / "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN.json",
            report,
        )
        atomic_write(
            output_dir / "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_HOLD",
            "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_HOLD\n",
        )
        print("V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_HOLD")
        return 1

    set_seed(args.seed)
    ContractDataset = import_contract_dataset(wrapper_path)

    train_dataset = ContractDataset(
        root=root,
        split="train",
        contract_dir=a2_dir,
        mask_audit_dir=a2_1_dir,
        window=32,
        stride=8,
        active_only=False,
        feature_variant="PRIMARY58",
    )
    validation_dataset = ContractDataset(
        root=root,
        split="validation",
        contract_dir=a2_dir,
        mask_audit_dir=a2_1_dir,
        window=32,
        stride=8,
        active_only=False,
        feature_variant="PRIMARY58",
    )

    sample = train_dataset[0]
    if tuple(sample["x"].shape) != (16, 58, 32):
        failures.append(
            f"train sample x shape unexpected: "
            f"{tuple(sample['x'].shape)}"
        )
    if tuple(sample["physical_port_mask"].shape) != (16, 10):
        failures.append(
            "physical_port_mask shape unexpected: "
            f"{tuple(sample['physical_port_mask'].shape)}"
        )
    if sample["physical_port_mask"].dtype != torch.bool:
        failures.append("physical_port_mask is not torch.bool")

    if failures:
        report = {
            "stage": "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN",
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "test_split_accessed": False,
        }
        write_json(
            output_dir / "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN.json",
            report,
        )
        atomic_write(
            output_dir / "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_HOLD",
            "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_HOLD\n",
        )
        print("V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_HOLD")
        return 1

    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    train_stats = target_statistics(train_eval_loader)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = DualTemporalFusionModel().to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    expected_parameter_count = 51080
    if parameter_count != expected_parameter_count:
        failures.append(
            f"B6 parameter_count={parameter_count}, "
            f"expected {expected_parameter_count}"
        )

    if failures:
        report = {
            "stage": "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN",
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "test_split_accessed": False,
        }
        write_json(
            output_dir / "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN.json",
            report,
        )
        atomic_write(
            output_dir / "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_HOLD",
            "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_HOLD\n",
        )
        print("V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_HOLD")
        return 1

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    pos_weights, pos_weight_values = build_positive_weights(
        train_stats,
        device=device,
    )

    print("===== V5 P0-B6 DUAL TEMPORAL FUSION, NO GCN =====")
    print("device:", device)
    print("parameter_count:", parameter_count)
    print("b2_parameter_count:", 25544)
    print("b3_parameter_count:", 43208)
    print("train_window_count:", len(train_dataset))
    print("validation_window_count:", len(validation_dataset))
    print("mean_branch: x.mean(dim=-1) -> Linear(58,64)")
    print("conv_branch: causal_depthwise_separable_conv1d")
    print("temporal_dilations: [1, 2, 4, 8]")
    print("temporal_receptive_field: 31")
    print("conv_readout: final_causal_encoded_timestep")
    print("fusion_input_dim: 138")
    print("fusion_output_dim: 64")
    print("physical_port_mask_used: true")
    print("physical_port_mask_source: topology_derived_boolean")
    print("temporal_order_used: true")
    print("edge_index_used: false")
    print("graph_message_passing_used: false")
    print("metadata_used_as_model_input: false")
    print("validation_during_training: false")
    print("test_split_accessed: false")
    print("positive_weights:", pos_weight_values)

    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_total = 0.0
        epoch_samples = 0
        component_totals = {
            "attack": 0.0,
            "count": 0.0,
            "source": 0.0,
            "transit": 0.0,
            "victim": 0.0,
            "path": 0.0,
        }

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                x=batch["x"].to(
                    device=device,
                    dtype=torch.float32,
                ),
                physical_port_mask=batch[
                    "physical_port_mask"
                ].to(device=device),
            )
            loss, components = compute_loss(
                outputs=outputs,
                batch=batch,
                pos_weights=pos_weights,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch {epoch}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            optimizer.step()

            batch_size = int(batch["x"].shape[0])
            epoch_total += (
                float(loss.detach().item()) * batch_size
            )
            epoch_samples += batch_size
            for key, value in components.items():
                component_totals[key] += value * batch_size

        mean_loss = epoch_total / max(1, epoch_samples)
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                **{
                    f"train_loss_{key}": (
                        value / max(1, epoch_samples)
                    )
                    for key, value in component_totals.items()
                },
            }
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == args.epochs
        ):
            print(
                f"epoch={epoch:03d}/{args.epochs} "
                f"train_loss={mean_loss:.6f}"
            )

    train_metrics, _ = evaluate(
        model=model,
        loader=train_eval_loader,
        device=device,
    )
    validation_metrics, validation_predictions = evaluate(
        model=model,
        loader=validation_loader,
        device=device,
    )

    checkpoint_path = model_dir / "final_model.pt"
    torch.save(
        {
            "stage": "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN",
            "model_class": "DualTemporalFusionModel",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_epoch": args.epochs,
            "seed": args.seed,
            "feature_variant": "PRIMARY58",
            "window": 32,
            "stride": 8,
            "mean_branch": "mean_over_32_epochs_then_linear_58_to_64",
            "temporal_encoder": "causal_depthwise_separable_conv1d",
            "temporal_dilations": [1, 2, 4, 8],
            "temporal_receptive_field": 31,
            "conv_readout": "final_causal_encoded_timestep",
            "physical_port_mask_source": "topology_derived_boolean",
            "temporal_order_used": True,
            "edge_index_used": False,
            "graph_message_passing_used": False,
            "metadata_used_as_model_input": False,
            "binary_threshold": 0.5,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "model_config": {
                "feature_count": 58,
                "temporal_channels": 64,
                "mean_channels": 64,
                "mask_count": 10,
                "fusion_input_dim": 138,
                "fusion_channels": 64,
                "graph_hidden": 64,
                "role_hidden": 32,
                "parameter_count": parameter_count,
            },
        },
        checkpoint_path,
    )

    predictions_path = (
        model_dir / "validation_predictions_and_targets.pt"
    )
    torch.save(
        {
            "stage": "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN",
            "split": "validation",
            "binary_threshold": 0.5,
            "predictions_and_targets": validation_predictions,
            "metadata_included": False,
        },
        predictions_path,
    )

    history_path = (
        output_dir
        / "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_HISTORY.csv"
    )
    metrics_path = (
        output_dir
        / "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_METRICS.csv"
    )
    write_csv(history_path, history)
    write_csv(
        metrics_path,
        flatten_metric_rows("train", train_metrics)
        + flatten_metric_rows(
            "validation",
            validation_metrics,
        ),
    )

    comparison: dict[str, Any] = {}
    for stage, report in baseline_reports.items():
        comparison[f"train_b6_minus_{stage}"] = metric_delta(
            train_metrics,
            report["train_metrics"],
        )
        comparison[f"validation_b6_minus_{stage}"] = metric_delta(
            validation_metrics,
            report["validation_metrics"],
        )

    comparison["key_metrics"] = {
        f"{stage}_validation": key_metrics(
            report["validation_metrics"]
        )
        for stage, report in baseline_reports.items()
    }
    comparison["key_metrics"]["b6_validation"] = key_metrics(
        validation_metrics
    )

    comparison["selection_rule"] = {
        "primary_gate": (
            "Reject candidates with materially collapsed attack recall."
        ),
        "primary_metrics": [
            "graph_recall",
            "graph_balanced_accuracy",
            "graph_fpr",
            "count_macro_f1",
        ],
        "secondary_metrics": [
            "source_node_f1_attack",
            "victim_node_f1_attack",
            "transit_node_f1_attack",
            "path_node_f1_attack",
        ],
        "supplementary_metric": "all_tasks_exact",
        "test_must_remain_untouched": True,
    }

    comparison_path = (
        output_dir
        / "V5_P0_B6_VS_B1_B2_B3_B4_B5_COMPARISON.json"
    )
    write_json(comparison_path, comparison)

    report = {
        "stage": "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN",
        "status": "COMPLETE",
        "scientific_question": (
            "Can stable order-free temporal evidence and learned causal "
            "temporal evidence be fused in one router-wise model without "
            "GCN smoothing?"
        ),
        "model": {
            "class": "DualTemporalFusionModel",
            "parameter_count": parameter_count,
            "single_stage": True,
            "node_input": (
                "PRIMARY58 full window + recovered Boolean physical mask"
            ),
            "mean_branch": "x.mean(dim=-1) -> Linear(58,64) -> ReLU",
            "conv_branch": (
                "B3 causal depthwise-separable Conv1D, dilations "
                "[1,2,4,8], final encoded timestep"
            ),
            "fusion": (
                "concat(mean64, conv64, mask10) -> Linear(138,64) -> ReLU"
            ),
            "temporal_order_used": True,
            "edge_index_used": False,
            "graph_message_passing_used": False,
            "physical_port_mask_source": "topology_derived_boolean",
            "stored_standardized_mask_channels_used": False,
            "metadata_used_as_model_input": False,
            "pooling": ["node_mean", "node_max"],
        },
        "protocol": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": 5.0,
            "binary_threshold": 0.5,
            "early_stopping": False,
            "validation_checkpoint_selection": False,
            "validation_evaluated_only_after_training": True,
            "test_split_accessed": False,
        },
        "fairness": {
            "same_seed_as_b1_to_b5": args.seed == 101,
            "same_epochs_as_b1_to_b5": args.epochs == 100,
            "same_batch_size_as_b1_to_b5": args.batch_size == 128,
            "same_learning_rate_as_b1_to_b5": (
                args.learning_rate == 0.001
            ),
            "same_weight_decay_as_b1_to_b5": (
                args.weight_decay == 0.0001
            ),
            "same_losses_and_weights": True,
            "same_binary_threshold": True,
            "same_conv_encoder_as_b3": True,
            "same_graph_and_role_head_dimensions_as_b3": True,
            "parameter_count_not_matched": True,
            "predeclared_architectural_change": (
                "add B2-style mean branch and router-wise fusion to B3; "
                "do not use edge_index or message passing"
            ),
        },
        "train_target_statistics": train_stats,
        "positive_weights": pos_weight_values,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "comparison_to_b1_b2_b3_b4_b5": comparison,
        "selection_rule": comparison["selection_rule"],
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "validation_predictions": str(predictions_path),
            "validation_predictions_sha256": sha256_file(
                predictions_path
            ),
            "history_csv": str(history_path),
            "history_csv_sha256": sha256_file(history_path),
            "metrics_csv": str(metrics_path),
            "metrics_csv_sha256": sha256_file(metrics_path),
            "comparison_json": str(comparison_path),
            "comparison_json_sha256": sha256_file(comparison_path),
        },
        "failures": failures,
        "warnings": warnings,
        "audit_boundary": {
            "train_used_for_fitting": True,
            "validation_used_for_final_evaluation": True,
            "validation_used_for_checkpoint_selection": False,
            "validation_used_for_threshold_selection": False,
            "test_directory_enumerated": False,
            "test_dataset_constructed": False,
            "test_tensors_read": False,
            "test_performance_evaluated": False,
        },
        "provenance": {
            **prerequisite_provenance,
            "script_sha256": sha256_file(Path(__file__)),
        },
        "next_stage": "V5_P0_B7_ARCHITECTURE_FREEZE",
    }

    report_path = (
        output_dir / "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN.json"
    )
    write_json(report_path, report)

    lock = {
        "status": "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_COMPLETE",
        "report_sha256": sha256_file(report_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_predictions_sha256": sha256_file(
            predictions_path
        ),
        "history_csv_sha256": sha256_file(history_path),
        "metrics_csv_sha256": sha256_file(metrics_path),
        "comparison_json_sha256": sha256_file(comparison_path),
        "script_sha256": sha256_file(Path(__file__)),
        "test_split_accessed": False,
        "next_stage": "V5_P0_B7_ARCHITECTURE_FREEZE",
    }
    write_json(
        output_dir
        / "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_LOCK.json",
        lock,
    )

    atomic_write(
        output_dir
        / "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_COMPLETE",
        "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_COMPLETE\n",
    )

    delta_b3 = comparison["validation_b6_minus_b3"]
    delta_b2 = comparison["validation_b6_minus_b2"]

    print("\n===== B6 FINAL RESULTS =====")
    print(
        "train:",
        f"g_bal_acc={train_metrics['graph']['balanced_accuracy']:.4f}",
        f"g_recall={train_metrics['graph']['recall']:.4f}",
        f"g_f1={train_metrics['graph']['f1']:.4f}",
        f"g_fpr={train_metrics['graph']['fpr']:.4f}",
        f"count_macro_f1={train_metrics['count']['macro_f1']:.4f}",
        (
            "source_f1_attack="
            f"{train_metrics['roles']['source']['attack_windows']['node_f1']:.4f}"
        ),
        (
            "victim_f1_attack="
            f"{train_metrics['roles']['victim']['attack_windows']['node_f1']:.4f}"
        ),
        f"all_exact={train_metrics['all_tasks_exact']:.4f}",
    )
    print(
        "validation:",
        f"g_bal_acc={validation_metrics['graph']['balanced_accuracy']:.4f}",
        f"g_recall={validation_metrics['graph']['recall']:.4f}",
        f"g_f1={validation_metrics['graph']['f1']:.4f}",
        f"g_fpr={validation_metrics['graph']['fpr']:.4f}",
        f"graph_fn={validation_metrics['graph']['fn']}",
        f"count_macro_f1={validation_metrics['count']['macro_f1']:.4f}",
        (
            "source_f1_attack="
            f"{validation_metrics['roles']['source']['attack_windows']['node_f1']:.4f}"
        ),
        (
            "victim_f1_attack="
            f"{validation_metrics['roles']['victim']['attack_windows']['node_f1']:.4f}"
        ),
        f"all_exact={validation_metrics['all_tasks_exact']:.4f}",
    )
    print(
        "validation_delta_B6_minus_B3:",
        f"g_bal_acc={delta_b3['graph_balanced_accuracy']:+.4f}",
        f"g_recall={delta_b3['graph_recall']:+.4f}",
        f"g_f1={delta_b3['graph_f1']:+.4f}",
        f"g_fpr={delta_b3['graph_fpr']:+.4f}",
        f"graph_fn={delta_b3['graph_false_negatives']:+.0f}",
        f"count_macro_f1={delta_b3['count_macro_f1']:+.4f}",
        f"source_f1={delta_b3['source_node_f1_attack']:+.4f}",
        f"transit_f1={delta_b3['transit_node_f1_attack']:+.4f}",
        f"victim_f1={delta_b3['victim_node_f1_attack']:+.4f}",
        f"path_f1={delta_b3['path_node_f1_attack']:+.4f}",
        f"all_exact={delta_b3['all_tasks_exact']:+.4f}",
    )
    print(
        "validation_delta_B6_minus_B2:",
        f"g_bal_acc={delta_b2['graph_balanced_accuracy']:+.4f}",
        f"g_recall={delta_b2['graph_recall']:+.4f}",
        f"g_f1={delta_b2['graph_f1']:+.4f}",
        f"g_fpr={delta_b2['graph_fpr']:+.4f}",
        f"graph_fn={delta_b2['graph_false_negatives']:+.0f}",
        f"count_macro_f1={delta_b2['count_macro_f1']:+.4f}",
        f"source_f1={delta_b2['source_node_f1_attack']:+.4f}",
        f"transit_f1={delta_b2['transit_node_f1_attack']:+.4f}",
        f"victim_f1={delta_b2['victim_node_f1_attack']:+.4f}",
        f"path_f1={delta_b2['path_node_f1_attack']:+.4f}",
        f"all_exact={delta_b2['all_tasks_exact']:+.4f}",
    )
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("test_split_accessed: false")
    print("next_stage: V5_P0_B7_ARCHITECTURE_FREEZE")
    print("V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
