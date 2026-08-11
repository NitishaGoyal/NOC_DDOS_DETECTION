#!/usr/bin/env python3
"""
V5 P0-B0 Shortcut and Trivial Baseline Audit Suite.

Fits on TRAIN, evaluates on VALIDATION, and never constructs or reads TEST.
Binary thresholds are fixed at 0.5. Probe epochs are fixed; validation is not
used for checkpoint or threshold selection.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

ROLES = ("source", "transit", "victim", "path")
ROLE_KEYS = {
    "source": "y_source",
    "transit": "y_transit",
    "victim": "y_victim",
    "path": "y_attack_path",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"expected scalar, got {tuple(value.shape)}")
        return value.detach().cpu().item()
    return value


def text_scalar(value: Any) -> str:
    value = scalar(value)
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def import_contract_dataset(path: Path):
    spec = importlib.util.spec_from_file_location("v5_contract_loader_b0", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, "V5P0ContractWindowDataset", None)
    if cls is None:
        raise AttributeError("V5P0ContractWindowDataset missing")
    return cls


def verify_stage(
    directory: Path,
    marker: str,
    report: str,
    lock: str,
    expected_lock_status: str,
    failures: list[str],
) -> dict[str, Any]:
    marker_path = directory / marker
    report_path = directory / report
    lock_path = directory / lock
    for path in (marker_path, report_path, lock_path):
        if not path.is_file():
            failures.append(f"missing provenance artifact: {path}")
    if not report_path.is_file() or not lock_path.is_file():
        return {}
    report_data = load_json(report_path)
    lock_data = load_json(lock_path)
    if report_data.get("status") != "PASS":
        failures.append(f"provenance report not PASS: {report_path}")
    if lock_data.get("status") != expected_lock_status:
        failures.append(f"unexpected lock status: {lock_data.get('status')!r}")
    if lock_data.get("report_sha256") != sha256_file(report_path):
        failures.append(f"provenance SHA mismatch: {report_path}")
    return {
        "report_sha256": sha256_file(report_path),
        "lock_sha256": sha256_file(lock_path),
    }


def mode_lowest(values: torch.Tensor) -> int:
    counts = Counter(int(v) for v in values.reshape(-1).tolist())
    best = max(counts.values())
    return min(key for key, count in counts.items() if count == best)


def load_run_metadata(root: Path, split: str) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted((root / "runs" / split).glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        run_id = text_scalar(payload["run_id"])
        total_epochs = int(payload["x"].shape[0])
        result[run_id] = {
            "pair_id": text_scalar(payload["pair_id"]),
            "case_id": text_scalar(payload["case_id"]),
            "mode": text_scalar(payload["mode"]),
            "run_length": total_epochs,
            "window_count": 0 if total_epochs < 32 else 1 + (total_epochs - 32) // 8,
            "file_stem": path.stem,
        }
    return result


def collect_split(
    dataset: Dataset,
    run_meta: dict[str, dict[str, Any]],
    time_permutation: torch.Tensor | None = None,
) -> dict[str, Any]:
    attack, count = [], []
    roles = {role: [] for role in ROLES}
    epoch_id, window_start, window_target = [], [], []
    run_id, case_id, mode, pair_id, file_stem = [], [], [], [], []
    run_length, run_window_count, dataset_index = [], [], []
    node_summary = []

    for index in range(len(dataset)):
        item = dataset[index]
        rid = text_scalar(item["run_id"])
        meta = run_meta[rid]
        x = item["x"].detach().cpu().float()
        if time_permutation is not None:
            x = x[..., time_permutation]
        # Order-aware summary: mean/std are order invariant; final step is not.
        summary = torch.cat(
            [x.mean(dim=-1), x.std(dim=-1, unbiased=False), x[..., -1]],
            dim=-1,
        )
        node_summary.append(summary)

        attack.append(int(scalar(item["y_attack"])))
        count.append(int(scalar(item["y_attacker_count"])))
        for role in ROLES:
            roles[role].append(item[ROLE_KEYS[role]].detach().cpu().bool())
        epoch_id.append(float(scalar(item["epoch_id"])))
        window_start.append(float(scalar(item["window_start"])))
        window_target.append(float(scalar(item["window_target"])))
        run_id.append(rid)
        case_id.append(text_scalar(item["case_id"]))
        mode.append(text_scalar(item["mode"]))
        pair_id.append(meta["pair_id"])
        file_stem.append(meta["file_stem"])
        run_length.append(meta["run_length"])
        run_window_count.append(meta["window_count"])
        dataset_index.append(index)

    return {
        "attack": torch.tensor(attack, dtype=torch.long),
        "count": torch.tensor(count, dtype=torch.long),
        "roles": {role: torch.stack(values) for role, values in roles.items()},
        "epoch_id": torch.tensor(epoch_id, dtype=torch.float32),
        "window_start": torch.tensor(window_start, dtype=torch.float32),
        "window_target": torch.tensor(window_target, dtype=torch.float32),
        "run_id": run_id,
        "case_id": case_id,
        "mode": mode,
        "pair_id": pair_id,
        "file_stem": file_stem,
        "run_length": torch.tensor(run_length, dtype=torch.float32),
        "run_window_count": torch.tensor(run_window_count, dtype=torch.float32),
        "dataset_index": torch.tensor(dataset_index, dtype=torch.float32),
        "node_summary": torch.stack(node_summary),  # [samples,16,174]
    }


def binary_metrics(truth: torch.Tensor, pred: torch.Tensor) -> dict[str, Any]:
    truth, pred = truth.bool().reshape(-1), pred.bool().reshape(-1)
    tp = int((truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    return {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "balanced_accuracy": 0.5 * (recall + tnr),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "fpr": fp / max(1, fp + tn),
        "tnr": tnr,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "support_negative": tn + fp,
        "support_positive": tp + fn,
    }


def count_metrics(truth: torch.Tensor, pred: torch.Tensor) -> dict[str, Any]:
    truth, pred = truth.long().reshape(-1), pred.long().reshape(-1)
    cm = torch.zeros(3, 3, dtype=torch.long)
    for a, p in zip(truth.tolist(), pred.tolist()):
        if 0 <= a <= 2 and 0 <= p <= 2:
            cm[a, p] += 1
    f1s, per_class = [], {}
    for cls in range(3):
        tp = int(cm[cls, cls])
        fp = int(cm[:, cls].sum()) - tp
        fn = int(cm[cls, :].sum()) - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        f1s.append(f1)
        per_class[str(cls)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[cls, :].sum()),
        }
    return {
        "accuracy": float((truth == pred).float().mean()),
        "macro_f1": sum(f1s) / 3.0,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def node_metrics(
    truth: torch.Tensor,
    pred: torch.Tensor,
    attack_mask: torch.Tensor,
) -> dict[str, Any]:
    truth, pred = truth.bool(), pred.bool()

    def one(mask: torch.Tensor | None) -> dict[str, Any]:
        t, p = (truth, pred) if mask is None else (truth[mask], pred[mask])
        if t.numel() == 0:
            return {
                "window_count": 0, "node_precision": 0.0,
                "node_recall": 0.0, "node_f1": 0.0, "exact_set": 0.0,
            }
        tf, pf = t.reshape(-1), p.reshape(-1)
        tp = int((tf & pf).sum())
        fp = int((~tf & pf).sum())
        fn = int((tf & ~pf).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        return {
            "window_count": int(t.shape[0]),
            "node_precision": precision,
            "node_recall": recall,
            "node_f1": 2 * precision * recall / max(1e-12, precision + recall),
            "exact_set": float((t == p).all(dim=1).float().mean()),
            "positive_entries": int(tf.sum()),
            "tp": tp, "fp": fp, "fn": fn,
        }

    return {"all_windows": one(None), "attack_windows": one(attack_mask.bool())}


def evaluate(truth: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    result = {
        "graph": binary_metrics(truth["attack"], pred["attack"]),
        "count": count_metrics(truth["count"], pred["count"]),
        "roles": {},
    }
    all_exact = (
        truth["attack"].bool() == pred["attack"].bool()
    ) & (truth["count"].long() == pred["count"].long())
    for role in ROLES:
        result["roles"][role] = node_metrics(
            truth["roles"][role], pred["roles"][role], truth["attack"]
        )
        all_exact &= (truth["roles"][role].bool() == pred["roles"][role].bool()).all(dim=1)
    result["all_tasks_exact"] = float(all_exact.float().mean())
    return result


def logits_to_prediction(outputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "attack": torch.sigmoid(outputs["attack"]) >= 0.5,
        "count": outputs["count"].argmax(dim=-1),
        "roles": {
            role: torch.sigmoid(outputs[role]) >= 0.5 for role in ROLES
        },
    }


def positive_weight(target: torch.Tensor, cap: float = 20.0) -> float:
    target = target.float()
    pos = float(target.sum())
    neg = float(target.numel() - pos)
    return 1.0 if pos <= 0 else min(cap, neg / pos)


def loss_weights(truth: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    values = {"attack": positive_weight(truth["attack"])}
    values.update({role: positive_weight(truth["roles"][role]) for role in ROLES})
    return {
        key: torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in values.items()
    }


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    truth: dict[str, Any],
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        outputs["attack"], truth["attack"].float(), pos_weight=weights["attack"]
    )
    loss = loss + 0.5 * F.cross_entropy(outputs["count"], truth["count"].long())
    coefficients = {"source": 1.0, "transit": 0.5, "victim": 0.75, "path": 0.5}
    for role in ROLES:
        loss = loss + coefficients[role] * F.binary_cross_entropy_with_logits(
            outputs[role], truth["roles"][role].float(), pos_weight=weights[role]
        )
    return loss


class VectorProbe(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.attack = nn.Linear(hidden, 1)
        self.count = nn.Linear(hidden, 3)
        self.role_heads = nn.ModuleDict({
            role: nn.Linear(hidden, 16) for role in ROLES
        })

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(features)
        return {
            "attack": self.attack(hidden).squeeze(-1),
            "count": self.count(hidden),
            **{role: head(hidden) for role, head in self.role_heads.items()},
        }


class SummaryProbe(nn.Module):
    """Traffic probe using per-node mean, std, and final time step."""

    def __init__(self, node_input_dim: int = 174 + 10, hidden: int = 128) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.role_heads = nn.ModuleDict({
            role: nn.Linear(hidden, 1) for role in ROLES
        })
        self.attack = nn.Linear(hidden * 2, 1)
        self.count = nn.Linear(hidden * 2, 3)

    def forward(
        self,
        node_summary: torch.Tensor,
        physical_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = int(node_summary.shape[0])
        if physical_mask.ndim == 2:
            physical_mask = physical_mask.unsqueeze(0).repeat(batch, 1, 1)
        node_hidden = self.node_encoder(
            torch.cat(
                [node_summary, physical_mask.to(node_summary.dtype)],
                dim=-1,
            )
        )
        graph_hidden = torch.cat(
            [node_hidden.mean(dim=1), node_hidden.max(dim=1).values],
            dim=-1,
        )
        return {
            "attack": self.attack(graph_hidden).squeeze(-1),
            "count": self.count(graph_hidden),
            **{
                role: head(node_hidden).squeeze(-1)
                for role, head in self.role_heads.items()
            },
        }


class MaskOnlyProbe(nn.Module):
    """Shared node function over the fixed physical-port mask only."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(10, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.role_heads = nn.ModuleDict({
            role: nn.Linear(hidden, 1) for role in ROLES
        })
        self.attack = nn.Linear(hidden * 2, 1)
        self.count = nn.Linear(hidden * 2, 3)

    def forward(
        self,
        mask: torch.Tensor,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        node_hidden = self.node_encoder(mask.float())
        graph_hidden = torch.cat(
            [node_hidden.mean(dim=0), node_hidden.max(dim=0).values],
            dim=-1,
        ).unsqueeze(0).repeat(batch_size, 1)
        return {
            "attack": self.attack(graph_hidden).squeeze(-1),
            "count": self.count(graph_hidden),
            **{
                role: head(node_hidden).squeeze(-1).unsqueeze(0).repeat(batch_size, 1)
                for role, head in self.role_heads.items()
            },
        }


def truth_batch(
    truth: dict[str, Any],
    indices: torch.Tensor,
    device: torch.device,
    permutation: torch.Tensor | None = None,
) -> dict[str, Any]:
    label_indices = indices if permutation is None else permutation[indices]
    return {
        "attack": truth["attack"][label_indices].to(device),
        "count": truth["count"][label_indices].to(device),
        "roles": {
            role: truth["roles"][role][label_indices].to(device)
            for role in ROLES
        },
    }


@torch.no_grad()
def predict_vector(
    model: VectorProbe,
    features: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    output_parts = {"attack": [], "count": [], **{role: [] for role in ROLES}}
    for start in range(0, int(features.shape[0]), 512):
        outputs = model(features[start:start + 512].to(device))
        for key, value in outputs.items():
            output_parts[key].append(value.detach().cpu())
    return logits_to_prediction({
        key: torch.cat(values, dim=0) for key, values in output_parts.items()
    })


def train_vector_probe(
    name: str,
    train_features: torch.Tensor,
    validation_features: torch.Tensor,
    truth: dict[str, Any],
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    set_seed(seed)
    model = VectorProbe(int(train_features.shape[1])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights = loss_weights(truth, device)
    generator = torch.Generator().manual_seed(seed)
    final_loss = math.nan

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train_features.shape[0], generator=generator)
        loss_sum, sample_sum = 0.0, 0
        for start in range(0, len(order), 256):
            indices = order[start:start + 256]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(train_features[indices].to(device))
            loss = multitask_loss(
                outputs, truth_batch(truth, indices, device), weights
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: non-finite loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            sample_sum += len(indices)
        final_loss = loss_sum / max(1, sample_sum)
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"{name}: epoch={epoch:03d}/{epochs} loss={final_loss:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(train_features.shape[1]),
            "epochs": epochs,
            "seed": seed,
        },
        checkpoint,
    )
    return predict_vector(model, validation_features, device), {
        "final_train_loss": final_loss,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "epochs": epochs,
    }


def train_mask_probe(
    mask: torch.Tensor,
    train_truth: dict[str, Any],
    validation_count: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    set_seed(seed)
    model = MaskOnlyProbe().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights = loss_weights(train_truth, device)
    full_truth = {
        "attack": train_truth["attack"].to(device),
        "count": train_truth["count"].to(device),
        "roles": {
            role: train_truth["roles"][role].to(device) for role in ROLES
        },
    }
    final_loss = math.nan
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(mask.to(device), int(train_truth["attack"].shape[0]))
        loss = multitask_loss(outputs, full_truth, weights)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"mask_only: epoch={epoch:03d}/{epochs} loss={final_loss:.6f}")

    torch.save(
        {"model_state_dict": model.state_dict(), "epochs": epochs, "seed": seed},
        checkpoint,
    )
    model.eval()
    with torch.no_grad():
        outputs = {
            key: value.detach().cpu()
            for key, value in model(mask.to(device), validation_count).items()
        }
    return logits_to_prediction(outputs), {
        "final_train_loss": final_loss,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "epochs": epochs,
    }


@torch.no_grad()
def predict_summary(
    model: SummaryProbe,
    summaries: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    parts = {"attack": [], "count": [], **{role: [] for role in ROLES}}
    for start in range(0, int(summaries.shape[0]), 256):
        outputs = model(
            summaries[start:start + 256].to(device),
            mask.to(device),
        )
        for key, value in outputs.items():
            parts[key].append(value.detach().cpu())
    return logits_to_prediction({
        key: torch.cat(values, dim=0) for key, values in parts.items()
    })


def train_summary_probe(
    name: str,
    train_summaries: torch.Tensor,
    validation_summaries: torch.Tensor,
    mask: torch.Tensor,
    train_truth: dict[str, Any],
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    checkpoint: Path,
    label_permutation: torch.Tensor | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    set_seed(seed)
    model = SummaryProbe().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    effective_truth = train_truth
    weights = loss_weights(
        train_truth if label_permutation is None else {
            "attack": train_truth["attack"][label_permutation],
            "count": train_truth["count"][label_permutation],
            "roles": {
                role: train_truth["roles"][role][label_permutation]
                for role in ROLES
            },
        },
        device,
    )
    generator = torch.Generator().manual_seed(seed)
    final_loss = math.nan

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train_summaries.shape[0], generator=generator)
        loss_sum, sample_sum = 0.0, 0
        for start in range(0, len(order), 128):
            indices = order[start:start + 128]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                train_summaries[indices].to(device),
                mask.to(device),
            )
            loss = multitask_loss(
                outputs,
                truth_batch(
                    effective_truth,
                    indices,
                    device,
                    permutation=label_permutation,
                ),
                weights,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            sample_sum += len(indices)
        final_loss = loss_sum / max(1, sample_sum)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"{name}: epoch={epoch:03d}/{epochs} loss={final_loss:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": epochs,
            "seed": seed,
            "label_permutation": (
                label_permutation.tolist() if label_permutation is not None else None
            ),
        },
        checkpoint,
    )
    return predict_summary(model, validation_summaries, mask, device), {
        "final_train_loss": final_loss,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "epochs": epochs,
    }


def majority_baseline(
    train: dict[str, Any],
    validation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    attack_mode = mode_lowest(train["attack"])
    count_mode = mode_lowest(train["count"])
    n = int(validation["attack"].shape[0])
    return {
        "attack": torch.full((n,), attack_mode, dtype=torch.bool),
        "count": torch.full((n,), count_mode, dtype=torch.long),
        "roles": {
            role: torch.zeros(n, 16, dtype=torch.bool) for role in ROLES
        },
    }, {
        "attack_majority": attack_mode,
        "count_majority": count_mode,
        "node_policy": "all_zero",
    }


def router_prior_baseline(
    train: dict[str, Any],
    validation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    attack_truth = train["attack"].bool()
    prevalence = float(attack_truth.float().mean())
    always_attack = prevalence >= 0.5
    positive_count_mode = (
        mode_lowest(train["count"][attack_truth]) if bool(attack_truth.any()) else 0
    )
    n = int(validation["attack"].shape[0])
    details = {
        "train_attack_prevalence": prevalence,
        "always_attack": always_attack,
        "positive_count_mode": positive_count_mode,
        "roles": {},
    }
    role_predictions = {}
    for role in ROLES:
        labels = train["roles"][role][attack_truth]
        frequencies = labels.float().mean(dim=0) if labels.numel() else torch.zeros(16)
        cardinality = (
            int(round(float(labels.sum(dim=1).float().mean())))
            if labels.numel() else 0
        )
        cardinality = max(0, min(16, cardinality))
        static = torch.zeros(16, dtype=torch.bool)
        if always_attack and cardinality > 0:
            indices = torch.argsort(frequencies, descending=True, stable=True)[:cardinality]
            static[indices] = True
        role_predictions[role] = static.unsqueeze(0).repeat(n, 1)
        details["roles"][role] = {
            "frequencies": frequencies.tolist(),
            "cardinality": cardinality,
            "predicted_indices": torch.where(static)[0].tolist(),
        }
    return {
        "attack": torch.full((n,), always_attack, dtype=torch.bool),
        "count": torch.full(
            (n,), positive_count_mode if always_attack else 0, dtype=torch.long
        ),
        "roles": role_predictions,
    }, details


def polynomial_time_features(
    values: torch.Tensor,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    t = ((values - minimum) / max(1e-6, maximum - minimum)).clamp(0.0, 1.0)
    return torch.stack(
        [
            t, t.square(), t.pow(3),
            torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t),
            torch.sin(4 * math.pi * t), torch.cos(4 * math.pi * t),
            torch.sin(8 * math.pi * t), torch.cos(8 * math.pi * t),
        ],
        dim=1,
    )


def build_epoch_features(
    train: dict[str, Any],
    validation: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    minimum = float(train["window_target"].min())
    maximum = float(train["window_target"].max())
    return (
        polynomial_time_features(train["window_target"], minimum, maximum),
        polynomial_time_features(validation["window_target"], minimum, maximum),
        {
            "source": "window_target_only",
            "train_min": minimum,
            "train_max": maximum,
            "router_specific_node_output_weights": True,
        },
    )


def stable_hash_vector(text: str, buckets: int = 128) -> torch.Tensor:
    vector = torch.zeros(buckets)
    normalized = "^" + text.lower() + "$"
    for width in (2, 3, 4):
        for start in range(max(0, len(normalized) - width + 1)):
            gram = normalized[start:start + width].encode()
            digest = hashlib.sha256(gram).digest()
            bucket = int.from_bytes(digest[:4], "little") % buckets
            vector[bucket] += 1.0 if digest[4] % 2 == 0 else -1.0
    norm = float(vector.norm())
    return vector / norm if norm > 0 else vector


def build_identifier_features(
    train: dict[str, Any],
    validation: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    def numeric(summary: dict[str, Any]) -> torch.Tensor:
        length = summary["run_length"].clamp_min(1)
        index_denominator = max(1, int(summary["dataset_index"].numel()) - 1)
        return torch.stack(
            [
                summary["window_start"],
                summary["window_target"],
                summary["epoch_id"],
                summary["run_length"],
                summary["run_window_count"],
                summary["window_target"] / length,
                summary["window_start"] / length,
                summary["dataset_index"] / index_denominator,
            ],
            dim=1,
        )

    train_num = numeric(train)
    val_num = numeric(validation)
    mean = train_num.mean(dim=0)
    std = train_num.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    train_num, val_num = (train_num - mean) / std, (val_num - mean) / std

    def hashed(summary: dict[str, Any]) -> torch.Tensor:
        return torch.stack([
            stable_hash_vector(f"{rid}|{cid}|{stem}")
            for rid, cid, stem in zip(
                summary["run_id"], summary["case_id"], summary["file_stem"]
            )
        ])

    return (
        torch.cat([train_num, hashed(train)], dim=1),
        torch.cat([val_num, hashed(validation)], dim=1),
        {
            "mode_included": False,
            "explicit_source_target_parser_included": False,
            "numeric_fields": [
                "window_start", "window_target", "epoch_id", "run_length",
                "run_window_count", "target_fraction", "start_fraction",
                "serialization_fraction",
            ],
            "identifier_fields": ["run_id", "case_id", "file_stem"],
            "hash": "signed_sha256_character_ngrams_2_3_4_128_buckets",
        },
    )


PAIR_PATTERN = re.compile(r"S(\d+)T(\d+)", re.IGNORECASE)


def direct_metadata_audit(validation: dict[str, Any]) -> dict[str, Any]:
    n = int(validation["attack"].shape[0])
    mode_attack = torch.tensor(
        [value.lower() == "attack" for value in validation["mode"]],
        dtype=torch.bool,
    )
    mode_prediction = {
        "attack": mode_attack,
        "count": torch.zeros(n, dtype=torch.long),
        "roles": {role: torch.zeros(n, 16, dtype=torch.bool) for role in ROLES},
    }

    source = torch.zeros(n, 16, dtype=torch.bool)
    victim = torch.zeros(n, 16, dtype=torch.bool)
    count = torch.zeros(n, dtype=torch.long)
    parsed = torch.zeros(n, dtype=torch.bool)

    for i, case_id in enumerate(validation["case_id"]):
        pairs = [
            (int(s), int(t)) for s, t in PAIR_PATTERN.findall(case_id)
            if 0 <= int(s) < 16 and 0 <= int(t) < 16
        ]
        if pairs:
            parsed[i] = True
            count[i] = len(pairs)
            for s, t in pairs:
                source[i, s] = True
                victim[i, t] = True

    active = validation["attack"].bool()
    mask = active & parsed

    def exact(truth: torch.Tensor, pred: torch.Tensor) -> float:
        return (
            float((truth[mask] == pred[mask]).all(dim=1).float().mean())
            if bool(mask.any()) else 0.0
        )

    return {
        "mode_as_attack_baseline": evaluate(validation, mode_prediction),
        "case_id_parser": {
            "pattern": PAIR_PATTERN.pattern,
            "active_parsed_window_count": int(mask.sum()),
            "source_exact_active_parsed": exact(validation["roles"]["source"], source),
            "victim_exact_active_parsed": exact(validation["roles"]["victim"], victim),
            "count_accuracy_active_parsed": (
                float((validation["count"][mask] == count[mask]).float().mean())
                if bool(mask.any()) else 0.0
            ),
        },
        "policy": "all provenance strings and mode are forbidden model inputs",
    }


def flatten_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": name,
        "graph_accuracy": metrics["graph"]["accuracy"],
        "graph_balanced_accuracy": metrics["graph"]["balanced_accuracy"],
        "graph_f1": metrics["graph"]["f1"],
        "count_accuracy": metrics["count"]["accuracy"],
        "count_macro_f1": metrics["count"]["macro_f1"],
        "source_f1_attack": metrics["roles"]["source"]["attack_windows"]["node_f1"],
        "source_exact_attack": metrics["roles"]["source"]["attack_windows"]["exact_set"],
        "transit_f1_attack": metrics["roles"]["transit"]["attack_windows"]["node_f1"],
        "transit_exact_attack": metrics["roles"]["transit"]["attack_windows"]["exact_set"],
        "victim_f1_attack": metrics["roles"]["victim"]["attack_windows"]["node_f1"],
        "victim_exact_attack": metrics["roles"]["victim"]["attack_windows"]["exact_set"],
        "path_f1_attack": metrics["roles"]["path"]["attack_windows"]["node_f1"],
        "path_exact_attack": metrics["roles"]["path"]["attack_windows"]["exact_set"],
        "all_tasks_exact": metrics["all_tasks_exact"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--a2-1-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--a4-dir", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--traffic-epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    a2_dir = args.a2_dir.expanduser().resolve()
    a2_1_dir = args.a2_1_dir.expanduser().resolve()
    a3_dir = args.a3_dir.expanduser().resolve()
    a4_dir = args.a4_dir.expanduser().resolve()
    wrapper = args.wrapper.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
        return 2
    if model_dir.exists():
        print(f"STOP: model directory already exists: {model_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []
    holds: list[str] = []

    provenance = {
        "a3": verify_stage(
            a3_dir,
            "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_PASS",
            "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE.json",
            "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_LOCK.json",
            "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_PASS",
            failures,
        ),
        "a4": verify_stage(
            a4_dir,
            "V5_P0_A4_TINY_SUBSET_OVERFIT_PASS",
            "V5_P0_A4_TINY_SUBSET_OVERFIT.json",
            "V5_P0_A4_TINY_SUBSET_OVERFIT_LOCK.json",
            "V5_P0_A4_TINY_SUBSET_OVERFIT_PASS",
            failures,
        ),
    }

    mask_path = a2_1_dir / "topology_derived_raw_physical_port_mask.pt"
    if not wrapper.is_file():
        failures.append(f"missing contract loader: {wrapper}")
    if not mask_path.is_file():
        failures.append(f"missing audited physical mask: {mask_path}")

    if failures:
        report = {
            "stage": "V5_P0_B0_SHORTCUT_AUDIT_SUITE",
            "status": "FAIL",
            "failures": failures,
            "warnings": warnings,
        }
        report_path = output_dir / "V5_P0_B0_SHORTCUT_AUDIT_SUITE.json"
        write_json(report_path, report)
        atomic_write(
            output_dir / "V5_P0_B0_SHORTCUT_AUDIT_SUITE_HOLD",
            "V5_P0_B0_SHORTCUT_AUDIT_SUITE_HOLD\n",
        )
        print("V5_P0_B0_SHORTCUT_AUDIT_SUITE_HOLD")
        return 1

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== V5 P0-B0 SHORTCUT AUDIT SUITE =====")
    print("device:", device)
    print("test_split_accessed: false")

    ContractDataset = import_contract_dataset(wrapper)
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

    train_meta = load_run_metadata(root, "train")
    validation_meta = load_run_metadata(root, "validation")
    train_pairs = {value["pair_id"] for value in train_meta.values()}
    validation_pairs = {value["pair_id"] for value in validation_meta.values()}
    pair_overlap = sorted(train_pairs & validation_pairs)
    run_overlap = sorted(set(train_meta) & set(validation_meta))
    if pair_overlap:
        holds.append(f"train/validation pair overlap: {pair_overlap}")
    if run_overlap:
        holds.append(f"train/validation run overlap: {run_overlap}")

    print("collecting ordered TRAIN summaries...")
    train = collect_split(train_dataset, train_meta)
    print("collecting ordered VALIDATION summaries...")
    validation = collect_split(validation_dataset, validation_meta)

    time_generator = torch.Generator().manual_seed(args.seed + 200)
    time_permutation = torch.randperm(32, generator=time_generator)
    if torch.equal(time_permutation, torch.arange(32)):
        time_permutation = torch.roll(time_permutation, 1)

    print("collecting shuffled-time TRAIN summaries...")
    train_time_shuffled = collect_split(
        train_dataset,
        train_meta,
        time_permutation=time_permutation,
    )
    print("collecting shuffled-time VALIDATION summaries...")
    validation_time_shuffled = collect_split(
        validation_dataset,
        validation_meta,
        time_permutation=time_permutation,
    )

    if len(torch.unique(validation["attack"])) < 2:
        holds.append("validation graph labels do not contain both classes")

    metric_contract = {
        "version": "V5_P0_B0_METRICS_D0",
        "fit_split": "train",
        "evaluation_split": "validation",
        "test_split_accessed": False,
        "binary_threshold": 0.5,
        "checkpoint_selection": "none_fixed_epochs",
        "graph_metrics": [
            "accuracy", "balanced_accuracy", "precision", "recall",
            "f1", "fpr", "tnr", "confusion_counts",
        ],
        "count_metrics": ["accuracy", "macro_f1", "3x3_confusion_matrix"],
        "node_metrics": {
            "all_windows": ["precision", "recall", "f1", "exact_set"],
            "attack_windows": ["precision", "recall", "f1", "exact_set"],
        },
        "reason_for_attack_window_metrics": (
            "prevents all-zero control windows from inflating localization"
        ),
    }
    write_json(output_dir / "V5_P0_B0_METRIC_CONTRACT.json", metric_contract)

    mask_payload = torch.load(mask_path, map_location="cpu", weights_only=False)
    physical_mask = mask_payload["physical_port_mask"].detach().cpu().bool()

    results: dict[str, Any] = {}
    details: dict[str, Any] = {}

    print("\n===== B0-1 MAJORITY / ALL-ZERO =====")
    pred, info = majority_baseline(train, validation)
    results["majority_all_zero"] = evaluate(validation, pred)
    details["majority_all_zero"] = info

    print("\n===== B0-2 ROUTER-FREQUENCY PRIOR =====")
    pred, info = router_prior_baseline(train, validation)
    results["router_frequency_prior"] = evaluate(validation, pred)
    details["router_frequency_prior"] = info

    print("\n===== B0-3 EPOCH-ROUTER PROBE =====")
    epoch_train, epoch_validation, info = build_epoch_features(train, validation)
    pred, train_info = train_vector_probe(
        "epoch_router_probe",
        epoch_train,
        epoch_validation,
        train,
        args.probe_epochs,
        args.learning_rate,
        args.seed + 1,
        device,
        model_dir / "epoch_router_probe.pt",
    )
    results["epoch_router_probe"] = evaluate(validation, pred)
    details["epoch_router_probe"] = {**info, **train_info}

    print("\n===== B0-4 MASK/TOPOLOGY-ONLY PROBE =====")
    pred, train_info = train_mask_probe(
        physical_mask,
        train,
        int(validation["attack"].shape[0]),
        args.probe_epochs,
        args.learning_rate,
        args.seed + 2,
        device,
        model_dir / "mask_only_probe.pt",
    )
    results["mask_only_probe"] = evaluate(validation, pred)
    details["mask_only_probe"] = train_info

    print("\n===== B0-5 IDENTIFIER / SERIALIZATION PROBE =====")
    id_train, id_validation, info = build_identifier_features(train, validation)
    pred, train_info = train_vector_probe(
        "identifier_serialization_probe",
        id_train,
        id_validation,
        train,
        args.probe_epochs,
        args.learning_rate,
        args.seed + 3,
        device,
        model_dir / "identifier_serialization_probe.pt",
    )
    results["identifier_serialization_probe"] = evaluate(validation, pred)
    details["identifier_serialization_probe"] = {**info, **train_info}
    direct_metadata = direct_metadata_audit(validation)
    details["direct_metadata_audit"] = direct_metadata

    print("\n===== B0-6 LABEL-SHUFFLE COLLAPSE =====")
    permutation_generator = torch.Generator().manual_seed(args.seed + 100)
    label_permutation = torch.randperm(len(train_dataset), generator=permutation_generator)
    if torch.equal(label_permutation, torch.arange(len(train_dataset))):
        label_permutation = torch.roll(label_permutation, 1)
    pred, train_info = train_summary_probe(
        "label_shuffle_probe",
        train["node_summary"],
        validation["node_summary"],
        physical_mask,
        train,
        args.traffic_epochs,
        args.learning_rate,
        args.seed + 4,
        device,
        model_dir / "label_shuffle_probe.pt",
        label_permutation=label_permutation,
    )
    results["label_shuffle_probe"] = evaluate(validation, pred)
    details["label_shuffle_probe"] = {
        **train_info,
        "whole_multitask_bundle_shuffled": True,
        "permutation_sha256": hashlib.sha256(
            label_permutation.numpy().tobytes()
        ).hexdigest(),
    }

    print("\n===== B0-7 ORDERED-TIME SUMMARY PROBE =====")
    pred, train_info = train_summary_probe(
        "ordered_time_probe",
        train["node_summary"],
        validation["node_summary"],
        physical_mask,
        train,
        args.traffic_epochs,
        args.learning_rate,
        args.seed + 5,
        device,
        model_dir / "ordered_time_probe.pt",
    )
    results["ordered_time_probe"] = evaluate(validation, pred)
    details["ordered_time_probe"] = train_info

    print("\n===== B0-7 SHUFFLED-TIME SUMMARY PROBE =====")
    pred, train_info = train_summary_probe(
        "shuffled_time_probe",
        train_time_shuffled["node_summary"],
        validation_time_shuffled["node_summary"],
        physical_mask,
        train,
        args.traffic_epochs,
        args.learning_rate,
        args.seed + 6,
        device,
        model_dir / "shuffled_time_probe.pt",
    )
    results["shuffled_time_probe"] = evaluate(validation, pred)
    details["shuffled_time_probe"] = {
        **train_info,
        "time_permutation": time_permutation.tolist(),
    }

    # B0-8 decision thresholds.
    router = results["router_frequency_prior"]
    epoch = results["epoch_router_probe"]
    mask = results["mask_only_probe"]
    identifier = results["identifier_serialization_probe"]
    shuffled_label = results["label_shuffle_probe"]
    ordered = results["ordered_time_probe"]
    shuffled_time = results["shuffled_time_probe"]

    if (
        router["roles"]["source"]["attack_windows"]["node_f1"] >= 0.60
        or router["roles"]["source"]["attack_windows"]["exact_set"] >= 0.50
        or router["roles"]["victim"]["attack_windows"]["node_f1"] >= 0.60
        or router["roles"]["victim"]["attack_windows"]["exact_set"] >= 0.50
    ):
        holds.append("static router prior is too predictive of source/victim")

    if (
        epoch["graph"]["balanced_accuracy"] >= 0.85
        or epoch["count"]["macro_f1"] >= 0.75
        or epoch["roles"]["source"]["attack_windows"]["exact_set"] >= 0.70
        or epoch["roles"]["victim"]["attack_windows"]["exact_set"] >= 0.70
    ):
        holds.append("epoch-plus-router shortcut probe is highly predictive")

    if (
        mask["graph"]["balanced_accuracy"] >= 0.70
        or mask["roles"]["source"]["attack_windows"]["node_f1"] >= 0.60
        or mask["roles"]["source"]["attack_windows"]["exact_set"] >= 0.50
        or mask["roles"]["victim"]["attack_windows"]["node_f1"] >= 0.60
        or mask["roles"]["victim"]["attack_windows"]["exact_set"] >= 0.50
    ):
        holds.append("mask/topology-only probe is too predictive")

    if (
        identifier["graph"]["balanced_accuracy"] >= 0.85
        or identifier["count"]["macro_f1"] >= 0.75
        or identifier["roles"]["source"]["attack_windows"]["exact_set"] >= 0.70
        or identifier["roles"]["victim"]["attack_windows"]["exact_set"] >= 0.70
    ):
        holds.append("identifier/serialization probe predicts validation labels")

    if (
        shuffled_label["graph"]["balanced_accuracy"] >= 0.70
        or shuffled_label["count"]["macro_f1"] >= 0.55
        or shuffled_label["roles"]["source"]["attack_windows"]["node_f1"] >= 0.50
        or shuffled_label["roles"]["victim"]["attack_windows"]["node_f1"] >= 0.50
        or shuffled_label["all_tasks_exact"] >= 0.20
    ):
        holds.append("label-shuffled traffic probe retained excessive validation signal")

    parser_info = direct_metadata["case_id_parser"]
    if (
        parser_info["source_exact_active_parsed"] >= 0.90
        or parser_info["victim_exact_active_parsed"] >= 0.90
    ):
        warnings.append(
            "case_id directly encodes source/target placement; this remains "
            "acceptable only because A2/A3 exclude provenance metadata"
        )

    ordered_graph = ordered["graph"]["balanced_accuracy"]
    shuffled_graph = shuffled_time["graph"]["balanced_accuracy"]
    time_gap = ordered_graph - shuffled_graph
    if abs(time_gap) <= 0.02:
        warnings.append(
            "ordered and retrained shuffled-time probes differ by at most 0.02; "
            "temporal order may not be necessary for this summary probe"
        )

    summary_rows = [flatten_row(name, metrics) for name, metrics in results.items()]
    write_csv(output_dir / "V5_P0_B0_BASELINE_SUMMARY.csv", summary_rows)

    passed = not failures and not holds
    decision = {
        "status": "PASS" if passed else "HOLD",
        "critical_hold_reasons": holds,
        "warnings": warnings,
        "pair_integrity": {
            "train_pair_count": len(train_pairs),
            "validation_pair_count": len(validation_pairs),
            "pair_overlap": pair_overlap,
            "run_overlap": run_overlap,
        },
        "fixed_thresholds": {
            "router_prior_role_f1": 0.60,
            "router_prior_role_exact": 0.50,
            "epoch_graph_balanced_accuracy": 0.85,
            "epoch_count_macro_f1": 0.75,
            "epoch_role_exact": 0.70,
            "mask_graph_balanced_accuracy": 0.70,
            "mask_role_f1": 0.60,
            "mask_role_exact": 0.50,
            "identifier_graph_balanced_accuracy": 0.85,
            "identifier_count_macro_f1": 0.75,
            "identifier_role_exact": 0.70,
            "label_shuffle_graph_balanced_accuracy": 0.70,
            "label_shuffle_count_macro_f1": 0.55,
            "label_shuffle_role_f1": 0.50,
            "label_shuffle_all_tasks_exact": 0.20,
        },
        "time_diagnostic": {
            "ordered_graph_balanced_accuracy": ordered_graph,
            "shuffled_time_graph_balanced_accuracy": shuffled_graph,
            "ordered_minus_shuffled": time_gap,
            "time_permutation": time_permutation.tolist(),
        },
        "direct_metadata_policy": (
            "mode and identifier leakage are documented, but become a dataset "
            "failure only if they enter learned inputs; A2/A3 forbid them"
        ),
        "next_stage": (
            "V5_P0_B1_STATIC_FINAL_EPOCH_MLP"
            if passed
            else "HOLD_REMEDIATE_B0_SHORTCUTS"
        ),
    }

    report = {
        "stage": "V5_P0_B0_SHORTCUT_AUDIT_SUITE",
        "status": "PASS" if passed else "HOLD",
        "device": str(device),
        "train_window_count": len(train_dataset),
        "validation_window_count": len(validation_dataset),
        "test_split_accessed": False,
        "feature_variant": "PRIMARY58",
        "window": 32,
        "stride": 8,
        "probe_epochs": args.probe_epochs,
        "traffic_probe_epochs": args.traffic_epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "metric_contract": metric_contract,
        "results": results,
        "details": details,
        "decision": decision,
        "failures": failures,
        "warnings": warnings,
        "provenance": {
            **provenance,
            "wrapper_sha256": sha256_file(wrapper),
            "mask_sha256": sha256_file(mask_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "audit_boundary": {
            "train_used_for_fitting": True,
            "validation_used_for_evaluation": True,
            "validation_used_for_checkpoint_selection": False,
            "test_directory_enumerated": False,
            "test_dataset_constructed": False,
            "test_tensors_read": False,
            "test_performance_evaluated": False,
            "threshold_search_performed": False,
        },
        "next_stage": decision["next_stage"],
    }

    report_path = output_dir / "V5_P0_B0_SHORTCUT_AUDIT_SUITE.json"
    decision_path = output_dir / "V5_P0_B0_DECISION_REPORT.json"
    write_json(report_path, report)
    write_json(decision_path, decision)

    lock = {
        "status": (
            "V5_P0_B0_SHORTCUT_AUDIT_SUITE_PASS"
            if passed
            else "V5_P0_B0_SHORTCUT_AUDIT_SUITE_HOLD"
        ),
        "report_sha256": sha256_file(report_path),
        "decision_sha256": sha256_file(decision_path),
        "metric_contract_sha256": sha256_file(
            output_dir / "V5_P0_B0_METRIC_CONTRACT.json"
        ),
        "summary_csv_sha256": sha256_file(
            output_dir / "V5_P0_B0_BASELINE_SUMMARY.csv"
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "wrapper_sha256": sha256_file(wrapper),
        "test_split_accessed": False,
        "next_stage": decision["next_stage"],
    }
    write_json(
        output_dir / "V5_P0_B0_SHORTCUT_AUDIT_SUITE_LOCK.json",
        lock,
    )

    print("\n===== B0 SUMMARY =====")
    for row in summary_rows:
        print(
            f"{row['baseline']}: "
            f"g_bal_acc={row['graph_balanced_accuracy']:.4f} "
            f"count_macro_f1={row['count_macro_f1']:.4f} "
            f"src_f1_attack={row['source_f1_attack']:.4f} "
            f"victim_f1_attack={row['victim_f1_attack']:.4f} "
            f"all_exact={row['all_tasks_exact']:.4f}"
        )

    print("\n===== B0 DECISION =====")
    print("pair_overlap_count:", len(pair_overlap))
    print("run_overlap_count:", len(run_overlap))
    print("hold_reason_count:", len(holds))
    print("warning_count:", len(warnings))
    print("test_split_accessed: false")

    for reason in holds:
        print("HOLD:", reason)
    for warning in warnings:
        print("WARNING:", warning)

    if not passed:
        atomic_write(
            output_dir / "V5_P0_B0_SHORTCUT_AUDIT_SUITE_HOLD",
            "V5_P0_B0_SHORTCUT_AUDIT_SUITE_HOLD\n",
        )
        print("next_stage: HOLD_REMEDIATE_B0_SHORTCUTS")
        print("V5_P0_B0_SHORTCUT_AUDIT_SUITE_HOLD")
        return 1

    atomic_write(
        output_dir / "V5_P0_B0_SHORTCUT_AUDIT_SUITE_PASS",
        "V5_P0_B0_SHORTCUT_AUDIT_SUITE_PASS\n",
    )
    print("next_stage: V5_P0_B1_STATIC_FINAL_EPOCH_MLP")
    print("V5_P0_B0_SHORTCUT_AUDIT_SUITE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
