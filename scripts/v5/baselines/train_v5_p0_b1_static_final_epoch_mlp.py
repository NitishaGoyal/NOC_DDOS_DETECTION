#!/usr/bin/env python3
"""
V5 P0-B1 Static Final-Epoch MLP Baseline

Scientific question
-------------------
How much of the multitask NoC attack problem can be solved using only the
final time step in each 32-epoch window, without temporal modeling and without
message passing?

Frozen learned inputs
---------------------
- PRIMARY58 x at x[..., -1] only;
- topology-derived raw Boolean physical_port_mask.

Explicitly unused
-----------------
- all other time steps;
- edge_index;
- every identifier, filename, mode, seed, scenario field, run/window position,
  serialization field, and every derivative of such metadata.

Protocol
--------
- Fit on TRAIN only.
- Evaluate TRAIN and VALIDATION only after the fixed final training epoch.
- No early stopping.
- No validation checkpoint selection.
- No threshold search.
- Binary threshold fixed at 0.5.
- Count prediction uses argmax.
- TEST is never constructed, enumerated, read, or evaluated.
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
        "v5_p0_contract_loader_b1",
        wrapper_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import wrapper: {wrapper_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset_class = getattr(
        module,
        "V5P0ContractWindowDataset",
        None,
    )
    if dataset_class is None:
        raise AttributeError(
            "wrapper does not define V5P0ContractWindowDataset"
        )
    return dataset_class


def verify_prerequisites(
    a2_dir: Path,
    a3_dir: Path,
    r1c_dir: Path,
    wrapper_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []

    paths = {
        "a2_report": (
            a2_dir
            / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT.json"
        ),
        "a2_marker": (
            a2_dir
            / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_PASS"
        ),
        "a3_report": (
            a3_dir
            / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE.json"
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
        "wrapper": wrapper_path,
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    if failures:
        return failures, {}

    a2 = load_json(paths["a2_report"])
    a3 = load_json(paths["a3_report"])
    a3_lock = load_json(paths["a3_lock"])
    r1c = load_json(paths["r1c_report"])
    r1c_lock = load_json(paths["r1c_lock"])

    if a2.get("status") != "PASS":
        failures.append("A2 status is not PASS")
    if a3.get("status") != "PASS":
        failures.append("A3 status is not PASS")
    if r1c.get("status") != "PASS":
        failures.append("B0-R1C status is not PASS")

    if a3_lock.get("report_sha256") != sha256_file(paths["a3_report"]):
        failures.append("A3 report SHA does not match A3 lock")
    if r1c_lock.get("report_sha256") != sha256_file(paths["r1c_report"]):
        failures.append("B0-R1C report SHA does not match B0-R1C lock")

    loader_contract = a2.get("loader_contract", {})
    if loader_contract.get("primary_feature_variant") != "PRIMARY58":
        failures.append("A2 primary feature variant is not PRIMARY58")
    if loader_contract.get("metadata_concatenated_to_x") is not False:
        failures.append("A2 does not explicitly forbid metadata in x")
    if loader_contract.get("raw_mask_supplied_separately") is not True:
        failures.append("A2 does not require a separate raw physical mask")

    if a3.get("feature_variant") != "PRIMARY58":
        failures.append("A3 did not validate PRIMARY58")
    if int(a3.get("window", -1)) != 32:
        failures.append("A3 window is not 32")
    if int(a3.get("stride", -1)) != 8:
        failures.append("A3 stride is not 8")
    if a3.get("normalization_applied_by_wrapper") is not False:
        failures.append("A3 wrapper unexpectedly applies normalization")

    quarantine = r1c.get("quarantine_contract", {})
    if quarantine.get("b0_release_authorized") is not True:
        failures.append("B0-R1C did not authorize B0 release")
    if r1c.get("next_stage") != "V5_P0_B1_STATIC_FINAL_EPOCH_MLP":
        failures.append(
            f"B0-R1C next stage is unexpected: {r1c.get('next_stage')!r}"
        )

    return failures, {
        "a2_report_sha256": sha256_file(paths["a2_report"]),
        "a3_report_sha256": sha256_file(paths["a3_report"]),
        "a3_lock_sha256": sha256_file(paths["a3_lock"]),
        "r1c_report_sha256": sha256_file(paths["r1c_report"]),
        "r1c_lock_sha256": sha256_file(paths["r1c_lock"]),
        "wrapper_sha256": sha256_file(wrapper_path),
    }


class StaticFinalEpochMLP(nn.Module):
    def __init__(
        self,
        feature_count: int = 58,
        mask_count: int = 10,
        node_hidden: int = 64,
        graph_hidden: int = 64,
        role_hidden: int = 32,
    ) -> None:
        super().__init__()

        node_input = feature_count + mask_count
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input, node_hidden),
            nn.ReLU(),
            nn.Linear(node_hidden, node_hidden),
            nn.ReLU(),
        )

        self.role_heads = nn.ModuleDict(
            {
                role: nn.Sequential(
                    nn.Linear(node_hidden, role_hidden),
                    nn.ReLU(),
                    nn.Linear(role_hidden, 1),
                )
                for role in ROLES
            }
        )

        self.graph_encoder = nn.Sequential(
            nn.Linear(node_hidden * 2, graph_hidden),
            nn.ReLU(),
        )
        self.attack_head = nn.Linear(graph_hidden, 1)
        self.count_head = nn.Linear(graph_hidden, 3)

    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B,N,F,W], got {tuple(x.shape)}")
        if x.shape[1:] != (16, 58, 32):
            raise ValueError(
                "expected x trailing shape [16,58,32], got "
                f"{tuple(x.shape[1:])}"
            )
        if physical_port_mask.ndim != 3:
            raise ValueError(
                "physical_port_mask must have shape [B,16,10], got "
                f"{tuple(physical_port_mask.shape)}"
            )
        if physical_port_mask.shape[1:] != (16, 10):
            raise ValueError(
                "expected mask trailing shape [16,10], got "
                f"{tuple(physical_port_mask.shape[1:])}"
            )

        # The only temporal operation in B1: select the final epoch.
        final_epoch = x[..., -1]
        mask = physical_port_mask.to(
            device=final_epoch.device,
            dtype=final_epoch.dtype,
        )

        node_repr = self.node_encoder(
            torch.cat([final_epoch, mask], dim=-1)
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

    attack_target = batch["y_attack"].to(
        device=device,
        dtype=torch.float32,
    )
    count_target = batch["y_attacker_count"].to(
        device=device,
        dtype=torch.long,
    )

    losses: dict[str, torch.Tensor] = {
        "attack": F.binary_cross_entropy_with_logits(
            outputs["attack"],
            attack_target,
            pos_weight=pos_weights["attack"],
        ),
        "count": F.cross_entropy(
            outputs["count"],
            count_target,
        ),
    }

    for role, key in ROLE_KEYS.items():
        role_target = batch[key].to(
            device=device,
            dtype=torch.float32,
        )
        losses[role] = F.binary_cross_entropy_with_logits(
            outputs[role],
            role_target,
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
        x = batch["x"].to(
            device=device,
            dtype=torch.float32,
        )
        mask = batch["physical_port_mask"].to(device=device)

        outputs = model(
            x=x,
            physical_port_mask=mask,
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
        key: torch.cat(items, dim=0)
        for key, items in values.items()
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

        all_windows = node_role_metrics(
            truth,
            prediction,
        )

        active = attack_truth.bool()
        attack_windows = node_role_metrics(
            truth[active],
            prediction[active],
        )

        metrics["roles"][role] = {
            "all_windows": all_windows,
            "attack_windows": attack_windows,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--a2-1-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--r1c-dir", type=Path, required=True)
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

    failures, prerequisite_provenance = verify_prerequisites(
        a2_dir=a2_dir,
        a3_dir=a3_dir,
        r1c_dir=r1c_dir,
        wrapper_path=wrapper_path,
    )
    warnings: list[str] = []

    if failures:
        report = {
            "stage": "V5_P0_B1_STATIC_FINAL_EPOCH_MLP",
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "test_split_accessed": False,
        }
        report_path = (
            output_dir
            / "V5_P0_B1_STATIC_FINAL_EPOCH_MLP.json"
        )
        write_json(report_path, report)
        atomic_write(
            output_dir
            / "V5_P0_B1_STATIC_FINAL_EPOCH_MLP_HOLD",
            "V5_P0_B1_STATIC_FINAL_EPOCH_MLP_HOLD\n",
        )
        print("V5_P0_B1_STATIC_FINAL_EPOCH_MLP_HOLD")
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
            f"train sample x shape unexpected: {tuple(sample['x'].shape)}"
        )
    if tuple(sample["physical_port_mask"].shape) != (16, 10):
        failures.append(
            "train sample physical_port_mask shape unexpected: "
            f"{tuple(sample['physical_port_mask'].shape)}"
        )
    if sample["physical_port_mask"].dtype != torch.bool:
        failures.append(
            "train sample physical_port_mask is not torch.bool"
        )

    if failures:
        report = {
            "stage": "V5_P0_B1_STATIC_FINAL_EPOCH_MLP",
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "test_split_accessed": False,
        }
        report_path = (
            output_dir
            / "V5_P0_B1_STATIC_FINAL_EPOCH_MLP.json"
        )
        write_json(report_path, report)
        atomic_write(
            output_dir
            / "V5_P0_B1_STATIC_FINAL_EPOCH_MLP_HOLD",
            "V5_P0_B1_STATIC_FINAL_EPOCH_MLP_HOLD\n",
        )
        print("V5_P0_B1_STATIC_FINAL_EPOCH_MLP_HOLD")
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

    # A separate deterministic pass computes TRAIN-only class weights.
    train_stats = target_statistics(train_eval_loader)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = StaticFinalEpochMLP().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    pos_weights, pos_weight_values = build_positive_weights(
        train_stats,
        device=device,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("===== V5 P0-B1 STATIC FINAL-EPOCH MLP =====")
    print("device:", device)
    print("parameter_count:", parameter_count)
    print("train_window_count:", len(train_dataset))
    print("validation_window_count:", len(validation_dataset))
    print("input_temporal_reduction: x[..., -1]")
    print("edge_index_used: false")
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
            x = batch["x"].to(
                device=device,
                dtype=torch.float32,
            )
            mask = batch["physical_port_mask"].to(device=device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                x=x,
                physical_port_mask=mask,
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

            batch_size = int(x.shape[0])
            epoch_total += float(loss.detach().item()) * batch_size
            epoch_samples += batch_size
            for key, value in components.items():
                component_totals[key] += value * batch_size

        mean_loss = epoch_total / max(1, epoch_samples)
        row = {
            "epoch": epoch,
            "train_loss": mean_loss,
            **{
                f"train_loss_{key}": (
                    value / max(1, epoch_samples)
                )
                for key, value in component_totals.items()
            },
        }
        history.append(row)

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == args.epochs
        ):
            print(
                f"epoch={epoch:03d}/{args.epochs} "
                f"train_loss={mean_loss:.6f}"
            )

    # Strict final-epoch protocol: evaluate only after all training epochs.
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
            "stage": "V5_P0_B1_STATIC_FINAL_EPOCH_MLP",
            "model_class": "StaticFinalEpochMLP",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_epoch": args.epochs,
            "seed": args.seed,
            "feature_variant": "PRIMARY58",
            "window": 32,
            "stride": 8,
            "temporal_reduction": "final_epoch_x[..., -1]",
            "edge_index_used": False,
            "physical_port_mask_used": True,
            "metadata_used_as_model_input": False,
            "binary_threshold": 0.5,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "model_config": {
                "feature_count": 58,
                "mask_count": 10,
                "node_hidden": 64,
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
            "stage": "V5_P0_B1_STATIC_FINAL_EPOCH_MLP",
            "split": "validation",
            "binary_threshold": 0.5,
            "predictions_and_targets": validation_predictions,
            "metadata_included": False,
        },
        predictions_path,
    )

    history_path = (
        output_dir
        / "V5_P0_B1_STATIC_FINAL_EPOCH_MLP_HISTORY.csv"
    )
    metrics_path = (
        output_dir
        / "V5_P0_B1_STATIC_FINAL_EPOCH_MLP_METRICS.csv"
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

    report = {
        "stage": "V5_P0_B1_STATIC_FINAL_EPOCH_MLP",
        "status": "COMPLETE",
        "scientific_question": (
            "How much can a shared static node MLP solve from only the "
            "final epoch of each window, without temporal modeling or "
            "graph message passing?"
        ),
        "model": {
            "class": "StaticFinalEpochMLP",
            "parameter_count": parameter_count,
            "node_input": "PRIMARY58 final epoch + raw physical mask",
            "temporal_reduction": "x[..., -1]",
            "edge_index_used": False,
            "message_passing_used": False,
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
        "train_target_statistics": train_stats,
        "positive_weights": pos_weight_values,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
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
        "next_stage": "V5_P0_B2_TEMPORAL_MEAN_POOLING",
    }

    report_path = (
        output_dir
        / "V5_P0_B1_STATIC_FINAL_EPOCH_MLP.json"
    )
    write_json(report_path, report)

    lock = {
        "status": "V5_P0_B1_STATIC_FINAL_EPOCH_MLP_COMPLETE",
        "report_sha256": sha256_file(report_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_predictions_sha256": sha256_file(
            predictions_path
        ),
        "history_csv_sha256": sha256_file(history_path),
        "metrics_csv_sha256": sha256_file(metrics_path),
        "script_sha256": sha256_file(Path(__file__)),
        "test_split_accessed": False,
        "next_stage": "V5_P0_B2_TEMPORAL_MEAN_POOLING",
    }
    lock_path = (
        output_dir
        / "V5_P0_B1_STATIC_FINAL_EPOCH_MLP_LOCK.json"
    )
    write_json(lock_path, lock)

    marker_path = (
        output_dir
        / "V5_P0_B1_STATIC_FINAL_EPOCH_MLP_COMPLETE"
    )
    atomic_write(
        marker_path,
        "V5_P0_B1_STATIC_FINAL_EPOCH_MLP_COMPLETE\n",
    )

    print("\n===== B1 FINAL RESULTS =====")
    print(
        "train:",
        f"g_bal_acc={train_metrics['graph']['balanced_accuracy']:.4f}",
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
        f"g_f1={validation_metrics['graph']['f1']:.4f}",
        f"g_fpr={validation_metrics['graph']['fpr']:.4f}",
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
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("test_split_accessed: false")
    print("next_stage: V5_P0_B2_TEMPORAL_MEAN_POOLING")
    print("V5_P0_B1_STATIC_FINAL_EPOCH_MLP_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
