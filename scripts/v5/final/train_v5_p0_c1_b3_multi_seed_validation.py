#!/usr/bin/env python3
"""
V5 P0-C1 B3 Multi-Seed Validation Training

Runs the frozen B3 architecture from scratch for fresh seeds 107, 117, and 127
under the immutable C0 protocol.

Security boundary:
- constructs only TRAIN and VALIDATION datasets;
- never constructs, enumerates, loads, or evaluates TEST;
- performs no threshold calibration beyond the fixed 0.5 selection threshold.

Outputs:
- best and final checkpoints per seed;
- per-epoch histories;
- validation predictions and metrics per seed;
- one validation-selected checkpoint copied into selected/;
- immutable C1 report, lock, and completion marker.
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
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


STAGE = "V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING"
COMPLETE = f"{STAGE}_COMPLETE"
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
        "v5_p0_contract_loader_c1",
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


class FrozenB3Model(nn.Module):
    def __init__(
        self,
        feature_count: int = 58,
        temporal_channels: int = 64,
        mask_count: int = 10,
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
                    temporal_channels,
                    kernel_size=3,
                    dilation=dilation,
                )
                for dilation in (1, 2, 4, 8)
            ]
        )
        self.node_post = nn.Sequential(
            nn.Linear(
                temporal_channels + mask_count,
                temporal_channels,
            ),
            nn.ReLU(),
        )
        self.role_heads = nn.ModuleDict(
            {
                role: nn.Sequential(
                    nn.Linear(temporal_channels, role_hidden),
                    nn.ReLU(),
                    nn.Linear(role_hidden, 1),
                )
                for role in ROLES
            }
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(temporal_channels * 2, graph_hidden),
            nn.ReLU(),
        )
        self.attack_head = nn.Linear(graph_hidden, 1)
        self.count_head = nn.Linear(graph_hidden, 3)

    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (16, 58, 32):
            raise ValueError(
                f"expected x [B,16,58,32], got {tuple(x.shape)}"
            )
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
        ):
            raise ValueError(
                "expected physical_port_mask [B,16,10], got "
                f"{tuple(physical_port_mask.shape)}"
            )

        batch_size, routers, features, window = x.shape
        temporal = x.reshape(
            batch_size * routers,
            features,
            window,
        )
        temporal = F.relu(self.input_projection(temporal))
        for block in self.temporal_blocks:
            temporal = block(temporal)

        node_temporal = temporal[..., -1].reshape(
            batch_size,
            routers,
            -1,
        )
        mask = physical_port_mask.to(
            device=node_temporal.device,
            dtype=node_temporal.dtype,
        )
        node_repr = self.node_post(
            torch.cat([node_temporal, mask], dim=-1)
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


def positive_weight(positives: int, total: int) -> float:
    if positives <= 0:
        return 1.0
    return min(20.0, (total - positives) / positives)


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
    positive_weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["attack"].device

    losses: dict[str, torch.Tensor] = {
        "attack": F.binary_cross_entropy_with_logits(
            outputs["attack"],
            batch["y_attack"].to(
                device=device,
                dtype=torch.float32,
            ),
            pos_weight=positive_weights["attack"],
        ),
        "count": F.cross_entropy(
            outputs["count"],
            batch["y_attacker_count"].to(
                device=device,
                dtype=torch.long,
            ),
        ),
    }

    for role, target_key in ROLE_KEYS.items():
        losses[role] = F.binary_cross_entropy_with_logits(
            outputs[role],
            batch[target_key].to(
                device=device,
                dtype=torch.float32,
            ),
            pos_weight=positive_weights[role],
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

    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
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
        "attack_probability": [],
        "attack_prediction": [],
        "count_truth": [],
        "count_prediction": [],
    }
    for role in ROLES:
        values[f"{role}_truth"] = []
        values[f"{role}_probability"] = []
        values[f"{role}_prediction"] = []

    for batch in loader:
        outputs = model(
            batch["x"].to(device=device, dtype=torch.float32),
            batch["physical_port_mask"].to(device=device),
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

        for role, target_key in ROLE_KEYS.items():
            probability = torch.sigmoid(outputs[role])
            prediction = probability >= 0.5
            values[f"{role}_truth"].append(
                batch[target_key].detach().cpu().bool()
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
    metrics: dict[str, Any] = {
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
        attack_windows = attack_truth.bool()

        metrics["roles"][role] = {
            "all_windows": node_role_metrics(truth, prediction),
            "attack_windows": node_role_metrics(
                truth[attack_windows],
                prediction[attack_windows],
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


def selection_score(metrics: dict[str, Any]) -> float:
    graph = metrics["graph"]
    roles = metrics["roles"]
    score = (
        0.25 * graph["balanced_accuracy"]
        + 0.20 * graph["f1"]
        + 0.10 * metrics["count"]["macro_f1"]
        + 0.10 * roles["source"]["attack_windows"]["node_f1"]
        + 0.10 * roles["victim"]["attack_windows"]["node_f1"]
        + 0.05 * roles["transit"]["attack_windows"]["node_f1"]
        + 0.05 * roles["path"]["attack_windows"]["node_f1"]
        + 0.15 * metrics["all_tasks_exact"]
        - 0.20 * max(0.0, 0.90 - graph["recall"])
        - 0.10 * max(0.0, graph["fpr"] - 0.20)
    )
    return float(score)


def tie_key(metrics: dict[str, Any], epoch: int) -> tuple[float, ...]:
    return (
        -float(metrics["graph"]["fn"]),
        -float(metrics["graph"]["fpr"]),
        float(
            metrics["roles"]["victim"]["attack_windows"]["node_f1"]
        ),
        -float(epoch),
    )


def better_candidate(
    score: float,
    metrics: dict[str, Any],
    epoch: int,
    best_score: float | None,
    best_metrics: dict[str, Any] | None,
    best_epoch: int | None,
) -> bool:
    if best_score is None or best_metrics is None or best_epoch is None:
        return True
    if score > best_score + 1e-12:
        return True
    if abs(score - best_score) <= 1e-12:
        return tie_key(metrics, epoch) > tie_key(
            best_metrics,
            best_epoch,
        )
    return False


def compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "graph_balanced_accuracy": (
            metrics["graph"]["balanced_accuracy"]
        ),
        "graph_recall": metrics["graph"]["recall"],
        "graph_f1": metrics["graph"]["f1"],
        "graph_fpr": metrics["graph"]["fpr"],
        "graph_fn": float(metrics["graph"]["fn"]),
        "count_macro_f1": metrics["count"]["macro_f1"],
        "source_f1_attack": (
            metrics["roles"]["source"]["attack_windows"]["node_f1"]
        ),
        "transit_f1_attack": (
            metrics["roles"]["transit"]["attack_windows"]["node_f1"]
        ),
        "victim_f1_attack": (
            metrics["roles"]["victim"]["attack_windows"]["node_f1"]
        ),
        "path_f1_attack": (
            metrics["roles"]["path"]["attack_windows"]["node_f1"]
        ),
        "all_tasks_exact": metrics["all_tasks_exact"],
    }


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--a2-1-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--r1c-dir", type=Path, required=True)
    parser.add_argument("--b8-dir", type=Path, required=True)
    parser.add_argument("--c0-dir", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--b3-reference-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    a2_dir = args.a2_dir.expanduser().resolve()
    a2_1_dir = args.a2_1_dir.expanduser().resolve()
    a3_dir = args.a3_dir.expanduser().resolve()
    r1c_dir = args.r1c_dir.expanduser().resolve()
    b8_dir = args.b8_dir.expanduser().resolve()
    c0_dir = args.c0_dir.expanduser().resolve()
    wrapper_path = args.wrapper.expanduser().resolve()
    b3_reference_script = (
        args.b3_reference_script.expanduser().resolve()
    )
    output_dir = args.output_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output exists: {output_dir}", file=sys.stderr)
        return 2
    if model_dir.exists():
        print(f"STOP: model directory exists: {model_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    prerequisite_files = {
        "a2_report": (
            a2_dir
            / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT.json"
        ),
        "a2_marker": (
            a2_dir
            / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_PASS"
        ),
        "a2_1_report": (
            a2_1_dir
            / "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT.json"
        ),
        "a2_1_marker": (
            a2_1_dir
            / "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_PASS"
        ),
        "a3_report": (
            a3_dir / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE.json"
        ),
        "a3_lock": (
            a3_dir
            / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_LOCK.json"
        ),
        "a3_marker": (
            a3_dir
            / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_PASS"
        ),
        "r1c_report": (
            r1c_dir
            / "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE.json"
        ),
        "r1c_lock": (
            r1c_dir
            / "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_LOCK.json"
        ),
        "r1c_marker": (
            r1c_dir
            / "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_PASS"
        ),
        "b8_report": (
            b8_dir
            / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE.json"
        ),
        "b8_lock": (
            b8_dir
            / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE_LOCK.json"
        ),
        "b8_contract": (
            b8_dir
            / "V5_P0_B8_B3_CONV1D_ONLY_ARCHITECTURE_CONTRACT.json"
        ),
        "b8_marker": (
            b8_dir
            / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE_COMPLETE"
        ),
        "c0_report": (
            c0_dir / "V5_P0_C0_FINAL_B3_TRAINING_PROTOCOL_LOCK.json"
        ),
        "c0_lock": (
            c0_dir
            / "V5_P0_C0_FINAL_B3_TRAINING_PROTOCOL_LOCK_LOCK.json"
        ),
        "c0_protocol": (
            c0_dir / "V5_P0_C0_FINAL_B3_PROTOCOL.json"
        ),
        "c0_marker": (
            c0_dir
            / "V5_P0_C0_FINAL_B3_TRAINING_PROTOCOL_LOCK_COMPLETE"
        ),
        "wrapper": wrapper_path,
        "b3_reference_script": b3_reference_script,
    }

    for name, path in prerequisite_files.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    if not root.is_dir():
        failures.append(f"P0 data root missing: {root}")

    prerequisite_reports: dict[str, Any] = {}
    protocol: dict[str, Any] = {}
    b8_contract: dict[str, Any] = {}

    if not failures:
        a2 = load_json(prerequisite_files["a2_report"])
        a2_1 = load_json(prerequisite_files["a2_1_report"])
        a3 = load_json(prerequisite_files["a3_report"])
        a3_lock = load_json(prerequisite_files["a3_lock"])
        r1c = load_json(prerequisite_files["r1c_report"])
        r1c_lock = load_json(prerequisite_files["r1c_lock"])
        b8 = load_json(prerequisite_files["b8_report"])
        b8_lock = load_json(prerequisite_files["b8_lock"])
        b8_contract = load_json(prerequisite_files["b8_contract"])
        c0 = load_json(prerequisite_files["c0_report"])
        c0_lock = load_json(prerequisite_files["c0_lock"])
        protocol = load_json(prerequisite_files["c0_protocol"])

        prerequisite_reports = {
            "a2": a2,
            "a2_1": a2_1,
            "a3": a3,
            "r1c": r1c,
            "b8": b8,
            "c0": c0,
        }

        for name, report, expected in (
            ("A2", a2, "PASS"),
            ("A2-1", a2_1, "PASS"),
            ("A3", a3, "PASS"),
            ("R1C", r1c, "PASS"),
            ("B8", b8, "COMPLETE"),
            ("C0", c0, "COMPLETE"),
        ):
            if report.get("status") != expected:
                failures.append(
                    f"{name} status={report.get('status')!r}, "
                    f"expected {expected!r}"
                )

        if (
            a3_lock.get("report_sha256")
            != sha256_file(prerequisite_files["a3_report"])
        ):
            failures.append("A3 report SHA mismatch")
        if (
            r1c_lock.get("report_sha256")
            != sha256_file(prerequisite_files["r1c_report"])
        ):
            failures.append("R1C report SHA mismatch")
        if (
            b8_lock.get("report_sha256")
            != sha256_file(prerequisite_files["b8_report"])
        ):
            failures.append("B8 report SHA mismatch")
        if (
            c0_lock.get("report_sha256")
            != sha256_file(prerequisite_files["c0_report"])
        ):
            failures.append("C0 report SHA mismatch")
        if (
            c0_lock.get("protocol_file_sha256")
            != sha256_file(prerequisite_files["c0_protocol"])
        ):
            failures.append("C0 protocol-file SHA mismatch")

        stored_protocol_hash = protocol.get("protocol_sha256")
        protocol_without_hash = dict(protocol)
        protocol_without_hash.pop("protocol_sha256", None)
        if stored_protocol_hash != canonical_sha256(
            protocol_without_hash
        ):
            failures.append("C0 canonical protocol hash mismatch")
        if c0_lock.get("protocol_sha256") != stored_protocol_hash:
            failures.append("C0 lock protocol hash mismatch")

        if c0.get("test_split_accessed") is not False:
            failures.append("C0 does not certify untouched test")
        if c0.get("test_split_constructed") is not False:
            failures.append("C0 unexpectedly constructed test")

        if protocol.get("fresh_training_seeds") != [107, 117, 127]:
            failures.append("C0 seeds are not [107,117,127]")

        training = protocol.get("training", {})
        if training.get("maximum_epochs") != 150:
            failures.append("C0 maximum epochs is not 150")
        if training.get("batch_size") != 128:
            failures.append("C0 batch size is not 128")

        if (
            c0.get("provenance", {}).get("wrapper_sha256")
            != sha256_file(wrapper_path)
        ):
            failures.append("wrapper changed after C0 lock")
        if (
            c0.get("provenance", {}).get(
                "b3_reference_script_sha256"
            )
            != sha256_file(b3_reference_script)
        ):
            failures.append("B3 reference script changed after C0 lock")

        selected = b8_contract.get("selected_architecture", {})
        if selected.get("parameter_count") != 43208:
            failures.append("B8 B3 parameter count is not 43,208")
        if selected.get("graph_message_passing") is not False:
            failures.append("B8 selected architecture uses graph MP")

    if failures:
        hold = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "training_performed": False,
            "test_split_constructed": False,
            "test_split_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", hold)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

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
            f"sample x shape={tuple(sample['x'].shape)}"
        )
    if tuple(sample["physical_port_mask"].shape) != (16, 10):
        failures.append(
            "sample physical_port_mask shape="
            f"{tuple(sample['physical_port_mask'].shape)}"
        )
    if sample["physical_port_mask"].dtype != torch.bool:
        failures.append("physical_port_mask is not Boolean")

    if failures:
        hold = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "training_performed": False,
            "test_split_constructed": False,
            "test_split_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", hold)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        return 1

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    train_stats = target_statistics(train_eval_loader)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("===== V5 P0-C1 B3 MULTI-SEED VALIDATION TRAINING =====")
    print("device:", device)
    print("seeds: [107, 117, 127]")
    print("train_window_count:", len(train_dataset))
    print("validation_window_count:", len(validation_dataset))
    print("maximum_epochs: 150")
    print("early_stopping_patience: 20")
    print("minimum_epochs: 25")
    print("scheduler: ReduceLROnPlateau")
    print("selection_threshold: 0.5")
    print("test_split_constructed: false")
    print("test_split_accessed: false")

    training = protocol["training"]
    scheduler_contract = training["scheduler"]
    early_contract = training["early_stopping"]

    seed_results: list[dict[str, Any]] = []

    for seed in protocol["fresh_training_seeds"]:
        print(f"\n===== C1 B3 SEED {seed} =====")
        set_seed(seed)

        train_generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=training["batch_size"],
            shuffle=True,
            num_workers=0,
            generator=train_generator,
        )

        model = FrozenB3Model().to(device)
        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )
        if parameter_count != 43208:
            raise RuntimeError(
                f"parameter_count={parameter_count}, expected 43208"
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_contract["mode"],
            factor=scheduler_contract["factor"],
            patience=scheduler_contract["patience_epochs"],
            threshold=scheduler_contract["threshold"],
            min_lr=scheduler_contract["minimum_learning_rate"],
        )
        positive_weights, positive_weight_values = (
            build_positive_weights(train_stats, device)
        )

        seed_report_dir = output_dir / f"seed_{seed:03d}"
        seed_model_dir = model_dir / f"seed_{seed:03d}"
        seed_report_dir.mkdir()
        seed_model_dir.mkdir()

        best_checkpoint_path = (
            seed_model_dir / "best_checkpoint.pt"
        )
        final_checkpoint_path = (
            seed_model_dir / "final_epoch_checkpoint.pt"
        )
        predictions_path = (
            seed_model_dir
            / "best_validation_predictions_and_targets.pt"
        )
        history_path = seed_report_dir / "history.csv"
        metrics_path = seed_report_dir / "best_metrics.json"

        history: list[dict[str, Any]] = []
        best_score: float | None = None
        best_metrics: dict[str, Any] | None = None
        best_epoch: int | None = None
        best_train_loss: float | None = None
        early_reference_score = float("-inf")
        no_improvement_epochs = 0
        stopped_early = False
        stop_epoch = training["maximum_epochs"]

        for epoch in range(1, training["maximum_epochs"] + 1):
            model.train()
            total_loss = 0.0
            sample_count = 0
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
                    batch["x"].to(
                        device=device,
                        dtype=torch.float32,
                    ),
                    batch["physical_port_mask"].to(device=device),
                )
                loss, components = compute_loss(
                    outputs,
                    batch,
                    positive_weights,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite loss seed={seed} epoch={epoch}"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    training["gradient_clip_norm"],
                )
                optimizer.step()

                batch_size = int(batch["x"].shape[0])
                total_loss += float(loss.item()) * batch_size
                sample_count += batch_size
                for key, value in components.items():
                    component_totals[key] += value * batch_size

            mean_train_loss = total_loss / max(1, sample_count)
            validation_metrics, validation_predictions = evaluate(
                model,
                validation_loader,
                device,
            )
            score = selection_score(validation_metrics)
            learning_rate = float(
                optimizer.param_groups[0]["lr"]
            )

            selected_this_epoch = better_candidate(
                score,
                validation_metrics,
                epoch,
                best_score,
                best_metrics,
                best_epoch,
            )

            if selected_this_epoch:
                best_score = score
                best_metrics = validation_metrics
                best_epoch = epoch
                best_train_loss = mean_train_loss
                torch.save(
                    {
                        "stage": STAGE,
                        "seed": seed,
                        "epoch": epoch,
                        "selection_score": score,
                        "model_class": "FrozenB3Model",
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "validation_metrics": validation_metrics,
                        "parameter_count": parameter_count,
                        "architecture_contract_sha256": (
                            b8_contract[
                                "architecture_contract_sha256"
                            ]
                        ),
                        "protocol_sha256": protocol["protocol_sha256"],
                        "thresholds": {
                            "attack": 0.5,
                            "source": 0.5,
                            "transit": 0.5,
                            "victim": 0.5,
                            "path": 0.5,
                        },
                        "test_split_accessed": False,
                    },
                    best_checkpoint_path,
                )
                torch.save(
                    {
                        "stage": STAGE,
                        "seed": seed,
                        "epoch": epoch,
                        "selection_score": score,
                        "predictions_and_targets": (
                            validation_predictions
                        ),
                        "test_split_accessed": False,
                    },
                    predictions_path,
                )

            if (
                score
                > early_reference_score
                + early_contract["minimum_delta"]
            ):
                early_reference_score = score
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            scheduler.step(score)

            compact = compact_metrics(validation_metrics)
            history.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": mean_train_loss,
                    "validation_selection_score": score,
                    "learning_rate_before_scheduler_step": learning_rate,
                    "selected_as_best": selected_this_epoch,
                    "epochs_without_material_improvement": (
                        no_improvement_epochs
                    ),
                    **{
                        f"train_loss_{key}": (
                            value / max(1, sample_count)
                        )
                        for key, value in component_totals.items()
                    },
                    **compact,
                }
            )

            if epoch == 1 or epoch % 5 == 0 or selected_this_epoch:
                print(
                    f"seed={seed} epoch={epoch:03d}/150 "
                    f"train_loss={mean_train_loss:.6f} "
                    f"score={score:.6f} "
                    f"g_f1={compact['graph_f1']:.4f} "
                    f"g_rec={compact['graph_recall']:.4f} "
                    f"g_fpr={compact['graph_fpr']:.4f} "
                    f"victim_f1={compact['victim_f1_attack']:.4f} "
                    f"all_exact={compact['all_tasks_exact']:.4f} "
                    f"lr={learning_rate:.6g}"
                )

            if (
                epoch >= early_contract["minimum_training_epochs"]
                and no_improvement_epochs
                >= early_contract["patience_epochs"]
            ):
                stopped_early = True
                stop_epoch = epoch
                print(
                    f"seed={seed} early_stop epoch={epoch} "
                    f"best_epoch={best_epoch} "
                    f"best_score={best_score:.6f}"
                )
                break

        torch.save(
            {
                "stage": STAGE,
                "seed": seed,
                "epoch": stop_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "parameter_count": parameter_count,
                "test_split_accessed": False,
            },
            final_checkpoint_path,
        )

        if best_epoch is None or best_metrics is None or best_score is None:
            raise RuntimeError(f"seed {seed}: no best checkpoint selected")

        checkpoint = torch.load(
            best_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        confirmed_metrics, confirmed_predictions = evaluate(
            model,
            validation_loader,
            device,
        )
        confirmed_score = selection_score(confirmed_metrics)

        if abs(confirmed_score - best_score) > 1e-10:
            raise RuntimeError(
                f"seed {seed}: restored score mismatch "
                f"{confirmed_score} != {best_score}"
            )
        torch.save(
            {
                "stage": STAGE,
                "seed": seed,
                "epoch": best_epoch,
                "selection_score": confirmed_score,
                "predictions_and_targets": confirmed_predictions,
                "test_split_accessed": False,
            },
            predictions_path,
        )

        write_csv(history_path, history)
        seed_metrics_report = {
            "stage": STAGE,
            "status": "COMPLETE",
            "seed": seed,
            "parameter_count": parameter_count,
            "best_epoch": best_epoch,
            "stop_epoch": stop_epoch,
            "stopped_early": stopped_early,
            "best_selection_score": best_score,
            "best_train_loss": best_train_loss,
            "best_validation_metrics": confirmed_metrics,
            "positive_weights": positive_weight_values,
            "artifacts": {
                "best_checkpoint": [
                    str(best_checkpoint_path),
                    sha256_file(best_checkpoint_path),
                ],
                "final_checkpoint": [
                    str(final_checkpoint_path),
                    sha256_file(final_checkpoint_path),
                ],
                "validation_predictions": [
                    str(predictions_path),
                    sha256_file(predictions_path),
                ],
                "history": [
                    str(history_path),
                    sha256_file(history_path),
                ],
            },
            "test_split_constructed": False,
            "test_split_accessed": False,
        }
        write_json(metrics_path, seed_metrics_report)

        seed_results.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "stop_epoch": stop_epoch,
                "stopped_early": stopped_early,
                "best_selection_score": best_score,
                "metrics": confirmed_metrics,
                "compact_metrics": compact_metrics(
                    confirmed_metrics
                ),
                "best_checkpoint_path": str(best_checkpoint_path),
                "best_checkpoint_sha256": sha256_file(
                    best_checkpoint_path
                ),
                "validation_predictions_path": str(
                    predictions_path
                ),
                "validation_predictions_sha256": sha256_file(
                    predictions_path
                ),
                "seed_metrics_report": str(metrics_path),
                "seed_metrics_report_sha256": sha256_file(
                    metrics_path
                ),
            }
        )

        compact = compact_metrics(confirmed_metrics)
        print(
            "C1_SEED_RESULT",
            f"seed={seed}",
            f"best_epoch={best_epoch}",
            f"stop_epoch={stop_epoch}",
            f"score={best_score:.6f}",
            f"g_bal_acc={compact['graph_balanced_accuracy']:.4f}",
            f"g_recall={compact['graph_recall']:.4f}",
            f"g_f1={compact['graph_f1']:.4f}",
            f"g_fpr={compact['graph_fpr']:.4f}",
            f"count_f1={compact['count_macro_f1']:.4f}",
            f"source_f1={compact['source_f1_attack']:.4f}",
            f"victim_f1={compact['victim_f1_attack']:.4f}",
            f"all_exact={compact['all_tasks_exact']:.4f}",
        )

    selected_result: dict[str, Any] | None = None
    for result in seed_results:
        if selected_result is None:
            selected_result = result
            continue
        current_score = result["best_selection_score"]
        selected_score = selected_result["best_selection_score"]
        current_metrics = result["metrics"]
        selected_metrics = selected_result["metrics"]
        current_epoch = result["best_epoch"]
        selected_epoch = selected_result["best_epoch"]
        if better_candidate(
            current_score,
            current_metrics,
            current_epoch,
            selected_score,
            selected_metrics,
            selected_epoch,
        ):
            selected_result = result

    assert selected_result is not None

    selected_dir = model_dir / "selected"
    selected_dir.mkdir()
    selected_checkpoint_path = (
        selected_dir / "selected_best_checkpoint.pt"
    )
    selected_predictions_path = (
        selected_dir
        / "selected_validation_predictions_and_targets.pt"
    )
    shutil.copy2(
        selected_result["best_checkpoint_path"],
        selected_checkpoint_path,
    )
    shutil.copy2(
        selected_result["validation_predictions_path"],
        selected_predictions_path,
    )

    summary_rows: list[dict[str, Any]] = []
    for result in seed_results:
        summary_rows.append(
            {
                "seed": result["seed"],
                "best_epoch": result["best_epoch"],
                "stop_epoch": result["stop_epoch"],
                "stopped_early": result["stopped_early"],
                "selection_score": result["best_selection_score"],
                "selected_final_checkpoint": (
                    result["seed"] == selected_result["seed"]
                ),
                **result["compact_metrics"],
            }
        )
    summary_csv_path = output_dir / "V5_P0_C1_SEED_SUMMARY.csv"
    write_csv(summary_csv_path, summary_rows)

    metric_names = list(seed_results[0]["compact_metrics"])
    aggregate = {
        metric_name: mean_std(
            [
                result["compact_metrics"][metric_name]
                for result in seed_results
            ]
        )
        for metric_name in metric_names
    }
    aggregate["selection_score"] = mean_std(
        [
            result["best_selection_score"]
            for result in seed_results
        ]
    )

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "decision": "SELECT_ONE_B3_CHECKPOINT_FOR_C2_CALIBRATION",
        "architecture": {
            "name": "B3_CAUSAL_DEPTHWISE_SEPARABLE_CONV1D_ONLY",
            "parameter_count": 43208,
            "architecture_contract_sha256": (
                b8_contract["architecture_contract_sha256"]
            ),
        },
        "protocol_sha256": protocol["protocol_sha256"],
        "seeds": protocol["fresh_training_seeds"],
        "train_window_count": len(train_dataset),
        "validation_window_count": len(validation_dataset),
        "train_target_statistics": train_stats,
        "seed_results": seed_results,
        "aggregate_best_validation_metrics": aggregate,
        "selected_checkpoint": {
            "seed": selected_result["seed"],
            "best_epoch": selected_result["best_epoch"],
            "selection_score": (
                selected_result["best_selection_score"]
            ),
            "validation_metrics": selected_result["metrics"],
            "source_checkpoint_path": (
                selected_result["best_checkpoint_path"]
            ),
            "source_checkpoint_sha256": (
                selected_result["best_checkpoint_sha256"]
            ),
            "copied_checkpoint_path": str(
                selected_checkpoint_path
            ),
            "copied_checkpoint_sha256": sha256_file(
                selected_checkpoint_path
            ),
            "validation_predictions_path": str(
                selected_predictions_path
            ),
            "validation_predictions_sha256": sha256_file(
                selected_predictions_path
            ),
        },
        "artifacts": {
            "seed_summary_csv": [
                str(summary_csv_path),
                sha256_file(summary_csv_path),
            ],
        },
        "provenance": {
            "c0_report_sha256": sha256_file(
                prerequisite_files["c0_report"]
            ),
            "c0_lock_sha256": sha256_file(
                prerequisite_files["c0_lock"]
            ),
            "c0_protocol_file_sha256": sha256_file(
                prerequisite_files["c0_protocol"]
            ),
            "b8_contract_file_sha256": sha256_file(
                prerequisite_files["b8_contract"]
            ),
            "wrapper_sha256": sha256_file(wrapper_path),
            "b3_reference_script_sha256": sha256_file(
                b3_reference_script
            ),
            "c1_script_sha256": sha256_file(Path(__file__)),
        },
        "training_performed": True,
        "validation_used_for_checkpoint_selection": True,
        "threshold_calibration_performed": False,
        "test_split_constructed": False,
        "test_split_accessed": False,
        "test_performance_evaluated": False,
        "failures": failures,
        "warnings": warnings,
        "next_stage": (
            "V5_P0_C2_FINAL_CHECKPOINT_THRESHOLD_FREEZE"
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    lock = {
        "status": COMPLETE,
        "decision": "SELECT_ONE_B3_CHECKPOINT_FOR_C2_CALIBRATION",
        "report_sha256": sha256_file(report_path),
        "selected_seed": selected_result["seed"],
        "selected_best_epoch": selected_result["best_epoch"],
        "selected_checkpoint_sha256": sha256_file(
            selected_checkpoint_path
        ),
        "selected_validation_predictions_sha256": sha256_file(
            selected_predictions_path
        ),
        "seed_summary_csv_sha256": sha256_file(summary_csv_path),
        "architecture_contract_sha256": (
            b8_contract["architecture_contract_sha256"]
        ),
        "protocol_sha256": protocol["protocol_sha256"],
        "c1_script_sha256": sha256_file(Path(__file__)),
        "threshold_calibration_performed": False,
        "test_split_constructed": False,
        "test_split_accessed": False,
        "next_stage": (
            "V5_P0_C2_FINAL_CHECKPOINT_THRESHOLD_FREEZE"
        ),
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    selected_compact = selected_result["compact_metrics"]
    print("\n===== V5 P0-C1 FINAL SELECTION =====")
    print("selected_seed:", selected_result["seed"])
    print("selected_best_epoch:", selected_result["best_epoch"])
    print(
        "selected_selection_score:",
        f"{selected_result['best_selection_score']:.6f}",
    )
    print(
        "selected_graph_balanced_accuracy:",
        f"{selected_compact['graph_balanced_accuracy']:.4f}",
    )
    print(
        "selected_graph_recall:",
        f"{selected_compact['graph_recall']:.4f}",
    )
    print(
        "selected_graph_f1:",
        f"{selected_compact['graph_f1']:.4f}",
    )
    print(
        "selected_graph_fpr:",
        f"{selected_compact['graph_fpr']:.4f}",
    )
    print(
        "selected_count_macro_f1:",
        f"{selected_compact['count_macro_f1']:.4f}",
    )
    print(
        "selected_source_f1_attack:",
        f"{selected_compact['source_f1_attack']:.4f}",
    )
    print(
        "selected_victim_f1_attack:",
        f"{selected_compact['victim_f1_attack']:.4f}",
    )
    print(
        "selected_all_tasks_exact:",
        f"{selected_compact['all_tasks_exact']:.4f}",
    )
    print(
        "selected_checkpoint_sha256:",
        sha256_file(selected_checkpoint_path),
    )
    print("threshold_calibration_performed: false")
    print("test_split_constructed: false")
    print("test_split_accessed: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print(
        "next_stage: V5_P0_C2_FINAL_CHECKPOINT_THRESHOLD_FREEZE"
    )
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
