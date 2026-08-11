#!/usr/bin/env python3
"""
V5 P2-G3 role-aware multilabel localization training for one operator/seed.

Authorized operators:
    conv1d, gcnconv, graphconv
Authorized seeds:
    107, 117, 127

Authorized data:
    P2 train and validation only.

Outputs:
    source_logits  [B,16]
    transit_logits [B,16]
    victim_logits  [B,16]
    path_logits    [B,16]

The four roles are independent binary multilabel targets and may overlap.
No graph, count, test, quantization, or RTL work is performed in this stage.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from statistics import mean
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

try:
    from torch_geometric.nn import GCNConv, GraphConv
except Exception as exc:  # pragma: no cover
    GCNConv = None
    GraphConv = None
    PYG_IMPORT_ERROR: str | None = f"{type(exc).__name__}: {exc}"
else:
    PYG_IMPORT_ERROR = None


STAGE = "V5_P2_G3_ROLE_AWARE_MULTILABEL_SINGLE_RUN"
COMPLETE = f"{STAGE}_COMPLETE"
HOLD = f"{STAGE}_HOLD"

OPERATORS = ("conv1d", "gcnconv", "graphconv")
SEEDS = (107, 117, 127)
ROLES = ("source", "transit", "victim", "path")
TARGET_KEYS = {
    "source": "y_source",
    "transit": "y_transit",
    "victim": "y_victim",
    "path": "y_attack_path",
}

EXPECTED_TRAIN_ITEMS = 70_166
EXPECTED_VALIDATION_ITEMS = 12_528
EXPECTED_LOADER_SHA = "2725ff993f4f03ebee3d5ffb775b1fedd6b131a9c4c89ed8049f45b249c24ac2"
EXPECTED_B3_MODEL_SHA = "56ee3207d039b60e8e3a898a689cd8e361247c86b7450a3a5e95f423ebe30def"
EXPECTED_EDGE_SHA = "f6b8050bc158de509b0ff1c5d1d7cb1ffe32c08b0f2287398270b1b891b57aff"
EXPECTED_G1_REPORT_SHA = "299f33cfbc55d82075a6992793540bdc3d237846b1315caab8361efeee8c2e7a"
EXPECTED_G2_REPORT_SHA = "66748c90e128e2e869e30b55b461a49383a5253ccc4b49f3fb72111221b95a74"
EXPECTED_FULL_B3_PARAMETERS = 43_273
EXPECTED_CLEAN_PARAMETERS = {
    "conv1d": 34_692,
    "gcnconv": 43_012,
    "graphconv": 51_204,
}

MAX_EPOCHS = 100
MIN_EPOCHS = 15
EARLY_STOP_PATIENCE = 12
EARLY_STOP_MIN_DELTA = 1e-4
ITEM_BATCH_SIZE = 256
PAIR_BLOCK_BATCH_SIZE = ITEM_BATCH_SIZE // 2
GRADIENT_CLIP = 1.0
THRESHOLD_GRID = np.linspace(0.0, 1.0, 1001, dtype=np.float64)

CHECKPOINT_WEIGHTS = {
    "source": 0.30,
    "transit": 0.25,
    "victim": 0.25,
    "path": 0.20,
}
THRESHOLD_WEIGHTS = {
    "node_f1": 0.60,
    "exact_set_attack": 0.40,
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state_dict(state_dict: OrderedDict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in state_dict.items():
        digest.update(key.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def import_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
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


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_auroc(truth: np.ndarray, score: np.ndarray) -> float:
    y = truth.astype(np.int64).reshape(-1)
    s = score.astype(np.float64).reshape(-1)
    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC undefined because one class is absent")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        stop = start + 1
        while stop < len(s) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    positive_rank_sum = ranks[y == 1].sum()
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def average_precision(truth: np.ndarray, score: np.ndarray) -> float:
    y = truth.astype(np.int64).reshape(-1)
    s = score.astype(np.float64).reshape(-1)
    positive_count = int((y == 1).sum())
    if positive_count == 0:
        raise ValueError("average precision undefined because positives are absent")
    order = np.argsort(-s, kind="mergesort")
    sorted_truth = y[order]
    cumulative_positive = np.cumsum(sorted_truth)
    precision_at_rank = cumulative_positive / np.arange(1, len(y) + 1)
    return float((precision_at_rank * sorted_truth).sum() / positive_count)


def binary_metrics(truth: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    y = truth.astype(np.int64).reshape(-1)
    prediction = (score.reshape(-1) >= threshold).astype(np.int64)
    tp = int(((y == 1) & (prediction == 1)).sum())
    tn = int(((y == 0) & (prediction == 0)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    tnr = safe_divide(tn, tn + fp)
    return {
        "threshold": float(threshold),
        "accuracy": safe_divide(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": safe_divide(fp, fp + tn),
        "tnr": tnr,
        "balanced_accuracy": 0.5 * (recall + tnr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def exact_set_metrics(
    truth_2d: np.ndarray,
    score_2d: np.ndarray,
    threshold: float,
    attack_truth: np.ndarray,
) -> dict[str, float]:
    truth = truth_2d.astype(np.int64)
    prediction = (score_2d >= threshold).astype(np.int64)
    exact = np.all(prediction == truth, axis=1)
    attack = attack_truth.astype(bool).reshape(-1)
    control = ~attack
    role_active = np.any(truth == 1, axis=1)
    return {
        "overall": float(exact.mean()),
        "attack": float(exact[attack].mean()) if bool(attack.any()) else 0.0,
        "control": float(exact[control].mean()) if bool(control.any()) else 0.0,
        "role_active": (
            float(exact[role_active].mean()) if bool(role_active.any()) else 0.0
        ),
    }


def role_metrics(
    truth_2d: np.ndarray,
    score_2d: np.ndarray,
    attack_truth: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    if truth_2d.shape != score_2d.shape or truth_2d.ndim != 2:
        raise ValueError("role truth/score must be matching [items,16] arrays")
    flat_truth = truth_2d.reshape(-1)
    flat_score = score_2d.reshape(-1)
    return {
        "auroc": binary_auroc(flat_truth, flat_score),
        "average_precision": average_precision(flat_truth, flat_score),
        "thresholded": binary_metrics(flat_truth, flat_score, threshold),
        "exact_set": exact_set_metrics(
            truth_2d, score_2d, threshold, attack_truth
        ),
        "positive_entries": int((flat_truth == 1).sum()),
        "negative_entries": int((flat_truth == 0).sum()),
    }


def all_role_metrics(
    truths: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    attack_truth: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or {role: 0.5 for role in ROLES}
    return {
        role: role_metrics(
            truths[role], scores[role], attack_truth, thresholds[role]
        )
        for role in ROLES
    }


def checkpoint_selection_score(metrics: dict[str, Any]) -> float:
    return float(
        sum(
            CHECKPOINT_WEIGHTS[role] * metrics[role]["average_precision"]
            for role in ROLES
        )
    )


def tune_role_threshold(
    truth_2d: np.ndarray,
    score_2d: np.ndarray,
    attack_truth: np.ndarray,
) -> dict[str, Any]:
    if truth_2d.shape != score_2d.shape or truth_2d.ndim != 2:
        raise ValueError("threshold inputs must be matching [items,16] arrays")
    truth_bool = truth_2d.astype(bool, copy=False)
    score = score_2d.astype(np.float64, copy=False)
    attack = attack_truth.astype(bool).reshape(-1)
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_rank: tuple[float, ...] | None = None

    chunk_size = 32
    for start in range(0, len(THRESHOLD_GRID), chunk_size):
        thresholds = THRESHOLD_GRID[start:start + chunk_size]
        prediction = score[None, :, :] >= thresholds[:, None, None]

        tp = np.logical_and(prediction, truth_bool[None, :, :]).sum(axis=(1, 2))
        tn = np.logical_and(~prediction, ~truth_bool[None, :, :]).sum(axis=(1, 2))
        fp = np.logical_and(prediction, ~truth_bool[None, :, :]).sum(axis=(1, 2))
        fn = np.logical_and(~prediction, truth_bool[None, :, :]).sum(axis=(1, 2))
        precision = np.divide(
            tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) != 0
        )
        recall = np.divide(
            tp, tp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fn) != 0
        )
        tnr = np.divide(
            tn, tn + fp, out=np.zeros_like(tn, dtype=np.float64), where=(tn + fp) != 0
        )
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
        )
        exact = np.all(prediction == truth_bool[None, :, :], axis=2)
        exact_attack = exact[:, attack].mean(axis=1)
        balanced_accuracy = 0.5 * (recall + tnr)
        objective = (
            THRESHOLD_WEIGHTS["node_f1"] * f1
            + THRESHOLD_WEIGHTS["exact_set_attack"] * exact_attack
        )

        for index, threshold in enumerate(thresholds):
            row = {
                "threshold": float(threshold),
                "objective": float(objective[index]),
                "node_f1": float(f1[index]),
                "exact_set_attack": float(exact_attack[index]),
                "node_balanced_accuracy": float(balanced_accuracy[index]),
            }
            rows.append(row)
            rank = (
                row["objective"],
                row["node_f1"],
                row["exact_set_attack"],
                row["node_balanced_accuracy"],
                -abs(row["threshold"] - 0.5),
                -row["threshold"],
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_row = row

    if best_row is None or best_rank is None:
        raise RuntimeError("threshold selection produced no candidate")
    selected_metrics = role_metrics(
        truth_2d,
        score_2d,
        attack_truth,
        best_row["threshold"],
    )
    return {
        "grid": {
            "kind": "uniform_closed_interval",
            "start": 0.0,
            "stop": 1.0,
            "step": 0.001,
            "candidate_count": len(THRESHOLD_GRID),
        },
        "objective_weights": THRESHOLD_WEIGHTS,
        "selected": {
            **best_row,
            "metrics": selected_metrics,
            "ranking_tuple": list(best_rank),
        },
        "rows": rows,
    }


def joint_exact_metrics(
    truths: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    attack_truth: np.ndarray,
) -> dict[str, float]:
    exact_per_role = []
    for role in ROLES:
        prediction = (scores[role] >= thresholds[role]).astype(np.int64)
        exact_per_role.append(np.all(prediction == truths[role], axis=1))
    exact = np.logical_and.reduce(exact_per_role)
    attack = attack_truth.astype(bool).reshape(-1)
    control = ~attack
    return {
        "overall": float(exact.mean()),
        "attack": float(exact[attack].mean()) if bool(attack.any()) else 0.0,
        "control": float(exact[control].mean()) if bool(control.any()) else 0.0,
    }


class PairBlockBatchSampler(Sampler[list[int]]):
    """Deterministic pair-aligned ATTACK/CONTROL block sampler."""

    def __init__(
        self,
        dataset,
        *,
        block_batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.block_batch_size = int(block_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if self.block_batch_size <= 0:
            raise ValueError("block_batch_size must be positive")
        if len(dataset) % 2 != 0:
            raise ValueError("dataset item count must be even")

        groups: OrderedDict[str, list[int]] = OrderedDict()
        for base in range(0, len(dataset._index), 2):
            attack = dataset._index[base]
            control = dataset._index[base + 1]
            if (
                attack.mode != "attack"
                or control.mode != "control"
                or attack.pair_key != control.pair_key
                or attack.start != control.start
                or attack.target != control.target
            ):
                raise ValueError(f"pair-block contract failed at base {base}")
            groups.setdefault(attack.pair_key, []).append(base)
        self.groups = groups
        self.block_count = sum(len(group) for group in groups.values())
        if self.block_count * 2 != len(dataset):
            raise RuntimeError("pair-block count does not cover dataset")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(self.block_count / self.block_batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        pair_keys = list(self.groups)
        if self.shuffle:
            rng.shuffle(pair_keys)
        ordered_blocks: list[int] = []
        for pair_key in pair_keys:
            blocks = list(self.groups[pair_key])
            if self.shuffle:
                rng.shuffle(blocks)
            ordered_blocks.extend(blocks)
        for start in range(0, len(ordered_blocks), self.block_batch_size):
            item_indices: list[int] = []
            for base in ordered_blocks[start:start + self.block_batch_size]:
                item_indices.extend((base, base + 1))
            yield item_indices


def batched_edge_index(
    base_edge_index: torch.Tensor,
    batch_size: int,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    base = base_edge_index.to(device=device, dtype=torch.long)
    result = torch.cat(
        [base + graph_index * num_nodes for graph_index in range(batch_size)],
        dim=1,
    )
    source_graph = torch.div(result[0], num_nodes, rounding_mode="floor")
    target_graph = torch.div(result[1], num_nodes, rounding_mode="floor")
    if not torch.equal(source_graph, target_graph):
        raise RuntimeError("batched edge_index contains cross-graph edges")
    return result


def validate_edge_index(edge_index: torch.Tensor) -> dict[str, Any]:
    edge = edge_index.detach().cpu().long().contiguous()
    if tuple(edge.shape) != (2, 48):
        raise ValueError(f"edge_index shape={tuple(edge.shape)}, expected (2,48)")
    if int(edge.min()) != 0 or int(edge.max()) != 15:
        raise ValueError("edge_index router range is not exactly 0..15")
    if int((edge[0] == edge[1]).sum()) != 0:
        raise ValueError("stored physical edge_index contains self-loops")
    edge_set = {tuple(map(int, pair)) for pair in edge.t().tolist()}
    if len(edge_set) != 48:
        raise ValueError("edge_index does not contain 48 unique directed edges")
    for source, destination in edge_set:
        sr, sc = divmod(source, 4)
        dr, dc = divmod(destination, 4)
        if abs(sr - dr) + abs(sc - dc) != 1:
            raise ValueError(f"non-cardinal physical edge {source}->{destination}")
        if (destination, source) not in edge_set:
            raise ValueError(f"missing reverse edge for {source}->{destination}")
    return {
        "shape": [2, 48],
        "directed_edges": 48,
        "undirected_links": 24,
        "physical_self_loops": 0,
    }


class CleanRoleAwareBaseline(nn.Module):
    """Clean four-head role model extracted from a seeded frozen B3 reference."""

    def __init__(
        self,
        reference_b3: nn.Module,
        operator: str,
        base_edge_index: torch.Tensor,
    ) -> None:
        super().__init__()
        if operator not in OPERATORS:
            raise ValueError(f"unsupported operator: {operator}")
        self.operator = operator
        self.input_projection = copy.deepcopy(reference_b3.input_projection)
        self.temporal_blocks = copy.deepcopy(reference_b3.temporal_blocks)
        self.node_projection = copy.deepcopy(reference_b3.node_projection)
        self.source_head = copy.deepcopy(reference_b3.source_head)
        self.transit_head = copy.deepcopy(reference_b3.transit_head)
        self.victim_head = copy.deepcopy(reference_b3.victim_head)
        self.path_head = copy.deepcopy(reference_b3.path_head)
        self.register_buffer(
            "base_edge_index",
            base_edge_index.detach().cpu().long().contiguous(),
        )

        if operator == "conv1d":
            self.graph1 = None
            self.graph2 = None
        elif operator == "gcnconv":
            self.graph1 = GCNConv(64, 64, add_self_loops=True, normalize=True)
            self.graph2 = GCNConv(64, 64, add_self_loops=True, normalize=True)
        elif operator == "graphconv":
            self.graph1 = GraphConv(64, 64, aggr="add")
            self.graph2 = GraphConv(64, 64, aggr="add")
        self.activation = nn.ReLU()

    def encode_nodes(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (16, 58, 32):
            raise ValueError(f"x shape={tuple(x.shape)}, expected [B,16,58,32]")
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError("physical_port_mask must have shape [B,16,10]")
        batch_size, num_nodes, num_features, time_steps = x.shape
        y = x.reshape(batch_size * num_nodes, num_features, time_steps)
        y = F.relu(self.input_projection(y))
        for block in self.temporal_blocks:
            y = block(y)
        temporal_embedding = y[:, :, -1].reshape(batch_size, num_nodes, 64)
        node_input = torch.cat(
            (
                temporal_embedding,
                physical_port_mask.to(dtype=temporal_embedding.dtype),
            ),
            dim=-1,
        )
        return F.relu(self.node_projection(node_input))

    def logits_without_graph(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_embedding = self.encode_nodes(x, physical_port_mask)
        return {
            "source": self.source_head(node_embedding).squeeze(-1),
            "transit": self.transit_head(node_embedding).squeeze(-1),
            "victim": self.victim_head(node_embedding).squeeze(-1),
            "path": self.path_head(node_embedding).squeeze(-1),
        }

    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_embedding = self.encode_nodes(x, physical_port_mask)
        batch_size, num_nodes, embedding_dim = node_embedding.shape
        if self.operator != "conv1d":
            flat = node_embedding.reshape(batch_size * num_nodes, embedding_dim)
            edges = batched_edge_index(
                self.base_edge_index,
                batch_size,
                num_nodes,
                flat.device,
            )
            flat = self.activation(self.graph1(flat, edges))
            flat = self.activation(self.graph2(flat, edges))
            node_embedding = flat.reshape(batch_size, num_nodes, embedding_dim)
        logits = {
            "source": self.source_head(node_embedding).squeeze(-1),
            "transit": self.transit_head(node_embedding).squeeze(-1),
            "victim": self.victim_head(node_embedding).squeeze(-1),
            "path": self.path_head(node_embedding).squeeze(-1),
        }
        for role, value in logits.items():
            if tuple(value.shape) != (batch_size, 16):
                raise RuntimeError(
                    f"{role} head returned unexpected shape {tuple(value.shape)}"
                )
        return logits


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=(device.type == "cuda"))
        for key, value in batch.items()
    }


def weighted_role_losses(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    pos_weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses: dict[str, torch.Tensor] = {}
    for role in ROLES:
        target = batch[TARGET_KEYS[role]].float()
        if logits[role].shape != target.shape:
            raise RuntimeError(f"{role} logits/target shape mismatch")
        losses[role] = F.binary_cross_entropy_with_logits(
            logits[role],
            target,
            pos_weight=pos_weights[role],
        )
    total = sum(losses.values())
    return total, losses


def train_one_epoch(
    *,
    model: CleanRoleAwareBaseline,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weights: dict[str, torch.Tensor],
) -> dict[str, float]:
    model.train()
    item_count = 0
    weighted_total_loss = 0.0
    weighted_role_loss = {role: 0.0 for role in ROLES}
    maximum_preclip_gradient_norm = 0.0

    for batch in loader:
        batch = move_batch(batch, device)
        batch_items = int(batch["x"].shape[0])
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["x"], batch["physical_port_mask"])
        total_loss, role_losses = weighted_role_losses(logits, batch, pos_weights)
        if not torch.isfinite(total_loss):
            raise RuntimeError("non-finite training loss")
        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=GRADIENT_CLIP
        )
        maximum_preclip_gradient_norm = max(
            maximum_preclip_gradient_norm,
            float(gradient_norm.item()),
        )
        optimizer.step()
        item_count += batch_items
        weighted_total_loss += float(total_loss.item()) * batch_items
        for role in ROLES:
            weighted_role_loss[role] += float(role_losses[role].item()) * batch_items

    if item_count != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(
            f"training epoch consumed {item_count} items; expected {EXPECTED_TRAIN_ITEMS}"
        )
    result = {
        "loss": weighted_total_loss / item_count,
        "maximum_preclip_gradient_norm": maximum_preclip_gradient_norm,
    }
    result.update(
        {f"{role}_loss": weighted_role_loss[role] / item_count for role in ROLES}
    )
    return result


def validate(
    *,
    model: CleanRoleAwareBaseline,
    loader: DataLoader,
    device: torch.device,
    pos_weights: dict[str, torch.Tensor],
    return_predictions: bool,
) -> dict[str, Any]:
    model.eval()
    item_count = 0
    weighted_total_loss = 0.0
    weighted_role_loss = {role: 0.0 for role in ROLES}
    truth_parts = {role: [] for role in ROLES}
    score_parts = {role: [] for role in ROLES}
    attack_parts: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            batch_items = int(batch["x"].shape[0])
            logits = model(batch["x"], batch["physical_port_mask"])
            total_loss, role_losses = weighted_role_losses(logits, batch, pos_weights)
            if not torch.isfinite(total_loss):
                raise RuntimeError("non-finite validation loss")
            item_count += batch_items
            weighted_total_loss += float(total_loss.item()) * batch_items
            for role in ROLES:
                weighted_role_loss[role] += float(role_losses[role].item()) * batch_items
                truth_parts[role].append(
                    batch[TARGET_KEYS[role]].detach().cpu().numpy().astype(np.int64)
                )
                score_parts[role].append(
                    torch.sigmoid(logits[role]).detach().cpu().numpy()
                )
            attack_parts.append(
                batch["y_attack"].reshape(-1).detach().cpu().numpy().astype(np.int64)
            )

    if item_count != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(
            f"validation consumed {item_count} items; expected {EXPECTED_VALIDATION_ITEMS}"
        )
    truths = {role: np.concatenate(truth_parts[role], axis=0) for role in ROLES}
    scores = {role: np.concatenate(score_parts[role], axis=0) for role in ROLES}
    attack_truth = np.concatenate(attack_parts, axis=0)
    metrics = all_role_metrics(truths, scores, attack_truth)
    selection_score = checkpoint_selection_score(metrics)
    result: dict[str, Any] = {
        "loss": weighted_total_loss / item_count,
        "selection_score": selection_score,
        "roles": metrics,
    }
    result.update(
        {f"{role}_loss": weighted_role_loss[role] / item_count for role in ROLES}
    )
    if return_predictions:
        result["truths"] = truths
        result["scores"] = scores
        result["attack_truth"] = attack_truth
    return result


def checkpoint_rank(
    validation_metrics: dict[str, Any],
    epoch: int,
) -> tuple[float, ...]:
    roles = validation_metrics["roles"]
    return (
        float(validation_metrics["selection_score"]),
        float(roles["source"]["average_precision"]),
        float(roles["transit"]["average_precision"]),
        float(roles["victim"]["average_precision"]),
        float(roles["path"]["average_precision"]),
        -float(validation_metrics["loss"]),
        -int(epoch),
    )


def operator_configuration(operator: str) -> dict[str, Any]:
    if operator == "conv1d":
        return {"message_passing": False, "graph_layers": 0}
    if operator == "gcnconv":
        return {
            "message_passing": True,
            "graph_layers": 2,
            "operator": "GCNConv",
            "self_loops": True,
            "normalization": "symmetric",
            "edge_weights": "unit",
            "bias": True,
        }
    if operator == "graphconv":
        return {
            "message_passing": True,
            "graph_layers": 2,
            "operator": "GraphConv",
            "aggregation": "add",
            "stored_physical_self_loops": False,
            "explicit_root_transform": True,
            "bias": True,
        }
    raise ValueError(f"unknown operator: {operator}")


def analytic_operation_count(operator: str) -> dict[str, int | str]:
    input_projection = 16 * 32 * 58 * 64
    temporal_block = 16 * 32 * 64 * 3 + 16 * 32 * 64 * 64
    temporal_blocks = 4 * temporal_block
    node_projection = 16 * 74 * 64
    four_role_heads = 4 * 16 * (64 * 32 + 32)
    base_macs = (
        input_projection + temporal_blocks + node_projection + four_role_heads
    )
    graph_linear_macs = 0
    graph_message_scalar_ops = 0
    if operator == "gcnconv":
        graph_linear_macs = 2 * 16 * 64 * 64
        graph_message_scalar_ops = 2 * (48 + 16) * 64
    elif operator == "graphconv":
        graph_linear_macs = 2 * 2 * 16 * 64 * 64
        graph_message_scalar_ops = 2 * 48 * 64
    return {
        "method": "analytic_per_sample_proxy",
        "base_temporal_node_role_head_macs": base_macs,
        "graph_linear_macs": graph_linear_macs,
        "graph_message_scalar_ops": graph_message_scalar_ops,
        "pooling_scalar_ops": 0,
        "total_linear_macs": base_macs + graph_linear_macs,
    }


def measure_inference_latency(
    model: CleanRoleAwareBaseline,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    warmup: int = 10,
    repetitions: int = 50,
) -> dict[str, float | int]:
    model.eval()
    x = batch["x"].to(device)
    mask = batch["physical_port_mask"].to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x, mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repetitions):
            model(x, mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    batch_items = int(x.shape[0])
    milliseconds_per_batch = 1000.0 * elapsed / repetitions
    return {
        "batch_items": batch_items,
        "warmup_iterations": warmup,
        "timed_iterations": repetitions,
        "milliseconds_per_batch": milliseconds_per_batch,
        "microseconds_per_item": 1000.0 * milliseconds_per_batch / batch_items,
    }


def parse_distribution(value: dict[str, Any]) -> dict[int, int]:
    return {int(key): int(frequency) for key, frequency in value.items()}


def role_positive_count(distribution: dict[int, int]) -> int:
    return sum(count * frequency for count, frequency in distribution.items())


def role_positive_weights(b0_report: dict[str, Any]) -> dict[str, float]:
    train_summary = b0_report["label_summaries"]["train"]
    distribution_keys = {
        "source": "active_source_count_distribution",
        "transit": "active_transit_count_distribution",
        "victim": "active_victim_count_distribution",
        "path": "active_path_count_distribution",
    }
    total_entries = EXPECTED_TRAIN_ITEMS * 16
    result: dict[str, float] = {}
    for role in ROLES:
        positives = role_positive_count(
            parse_distribution(train_summary[distribution_keys[role]])
        )
        negatives = total_entries - positives
        raw = negatives / positives
        result[role] = min(20.0, max(1.0, raw))
    return result


def verify_contracts(args: argparse.Namespace) -> dict[str, Any]:
    g0 = args.g0_dir.resolve()
    topology = args.topology_dir.resolve()
    b0_r3 = args.b0_r3_dir.resolve()
    g1 = args.g1_aggregation_dir.resolve()
    g2 = args.g2_aggregation_dir.resolve()

    paths = {
        "operator_contract": g0 / "V5_P2_G0_GRAPH_OPERATOR_CONTRACTS.json",
        "training_policy": g0 / "V5_P2_G0_TRAINING_SELECTION_AND_SEED_POLICY.json",
        "task_matrix": g0 / "V5_P2_G0_TASK_AND_MODEL_MATRIX.json",
        "test_policy": g0 / "V5_P2_G0_P2_TEST_ACCESS_POLICY.json",
        "topology_report": topology / "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT.json",
        "topology_lock": topology / "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT_LOCK.json",
        "edge_index": topology / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
        "b0_r3_report": b0_r3 / "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT.json",
        "b0_r3_lock": b0_r3 / "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT_LOCK.json",
        "g1_report": g1 / "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION.json",
        "g1_lock": g1 / "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION_LOCK.json",
        "g1_complete": g1 / "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION_COMPLETE",
        "g2_report": g2 / "V5_P2_G2_DIRECT_GRAPH_MATRIX_AGGREGATION.json",
        "g2_lock": g2 / "V5_P2_G2_DIRECT_GRAPH_MATRIX_AGGREGATION_LOCK.json",
        "g2_complete": g2 / "V5_P2_G2_DIRECT_GRAPH_MATRIX_AGGREGATION_COMPLETE",
        "loader": args.loader_path.resolve(),
        "b3_model": args.b3_model_path.resolve(),
        "pair_manifest": args.pair_manifest.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing prerequisite {name}: {path}")

    if sha256_file(paths["loader"]) != EXPECTED_LOADER_SHA:
        raise RuntimeError("frozen P2 loader SHA changed")
    if sha256_file(paths["b3_model"]) != EXPECTED_B3_MODEL_SHA:
        raise RuntimeError("frozen B3 model SHA changed")
    if sha256_file(paths["edge_index"]) != EXPECTED_EDGE_SHA:
        raise RuntimeError("canonical edge_index SHA changed")
    if sha256_file(paths["g1_report"]) != EXPECTED_G1_REPORT_SHA:
        raise RuntimeError("frozen G1 aggregation report SHA changed")
    if sha256_file(paths["g2_report"]) != EXPECTED_G2_REPORT_SHA:
        raise RuntimeError("frozen G2 aggregation report SHA changed")

    operator_contract = load_json(paths["operator_contract"])
    training_policy = load_json(paths["training_policy"])
    task_matrix = load_json(paths["task_matrix"])
    test_policy = load_json(paths["test_policy"])
    topology_report = load_json(paths["topology_report"])
    topology_lock = load_json(paths["topology_lock"])
    b0_report = load_json(paths["b0_r3_report"])
    b0_lock = load_json(paths["b0_r3_lock"])
    g1_report = load_json(paths["g1_report"])
    g1_lock = load_json(paths["g1_lock"])
    g2_report = load_json(paths["g2_report"])
    g2_lock = load_json(paths["g2_lock"])

    if operator_contract.get("common_architecture", {}).get("graph_layer_count") != 2:
        raise RuntimeError("G0 graph layer count changed")
    mandatory = task_matrix.get("operator_suite", {}).get("mandatory")
    if mandatory != ["CONV1D_ONLY", "GCNCONV", "GRAPHCONV"]:
        raise RuntimeError("G0 mandatory operator suite changed")
    task_c = next(
        task for task in task_matrix.get("tasks", [])
        if task.get("task_id") == "C_ROLE_AWARE_MULTILABEL_LOCALIZATION"
    )
    if task_c.get("label_type") != "four_independent_binary_multilabel_heads":
        raise RuntimeError("G3 label type changed")
    if task_c.get("mutually_exclusive_roles") is not False:
        raise RuntimeError("G3 role overlap contract changed")
    if task_c.get("participating_losses") != [
        "source_weighted_bce",
        "transit_weighted_bce",
        "victim_weighted_bce",
        "path_weighted_bce",
    ]:
        raise RuntimeError("G3 participating losses changed")
    if task_c.get("targets") != [
        "source[16]",
        "transit[16]",
        "victim[16]",
        "path[16]",
    ]:
        raise RuntimeError("G3 target contract changed")

    base = training_policy.get("training_base", {})
    expected_base = {
        "amp": False,
        "checkpoint_selection_split": "validation_only",
        "early_stopping_patience": 12,
        "learning_rate": 0.001,
        "max_epochs": 100,
        "minimum_epochs": 15,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "threshold_tuning_during_training": False,
        "weight_decay": 0.0001,
    }
    for key, expected in expected_base.items():
        if base.get(key) != expected:
            raise RuntimeError(f"G0 training field {key} changed")
    if base.get("initial_screening_seeds") != list(SEEDS):
        raise RuntimeError("G0 initial seeds changed")
    task_policy = training_policy.get("task_selection", {}).get(
        "C_ROLE_AWARE_MULTILABEL_LOCALIZATION", {}
    )
    if task_policy.get("checkpoint_score") != {
        "path_average_precision": 0.2,
        "source_average_precision": 0.3,
        "transit_average_precision": 0.25,
        "victim_average_precision": 0.25,
    }:
        raise RuntimeError("G3 checkpoint score changed")
    if task_policy.get("loss") != "weighted source + transit + victim + path BCE losses":
        raise RuntimeError("G3 loss contract changed")
    if task_policy.get("post_checkpoint_threshold_selection") != (
        "separate validation-only role thresholds using the same "
        "node-F1/exact-set policy for every operator"
    ):
        raise RuntimeError("G3 threshold policy changed")

    if test_policy.get("g1_to_g6", {}).get("p2_test_tensor_access_allowed") is not False:
        raise RuntimeError("test-access policy unexpectedly changed")
    if b0_report.get("status") != "COMPLETE":
        raise RuntimeError("B0-R3 report is not complete")
    if b0_lock.get("report_sha256") != sha256_file(paths["b0_r3_report"]):
        raise RuntimeError("B0-R3 report SHA mismatch")
    if b0_lock.get("shortcut_block_count") != 0:
        raise RuntimeError("B0-R3 shortcut block count is not zero")
    if b0_lock.get("label_integrity_pass") is not True:
        raise RuntimeError("B0-R3 label integrity did not pass")

    if g1_report.get("status") != "COMPLETE" or g1_lock.get("completed_runs") != 12:
        raise RuntimeError("G1 aggregation is not complete")
    if g2_report.get("status") != "COMPLETE" or g2_lock.get("completed_runs") != 12:
        raise RuntimeError("G2 aggregation is not complete")
    if g1_report.get("gat_screening", {}).get("decision") != "DO_NOT_PROMOTE_GAT_FROM_G1":
        raise RuntimeError("G1 GAT decision changed")
    if g2_report.get("gat_screening", {}).get("g2_screening_decision") != "DO_NOT_PROMOTE_GAT_FROM_G2":
        raise RuntimeError("G2 GAT decision changed")
    for report in (g1_report, g2_report):
        boundary = report.get("security_boundary", {})
        if boundary.get("p2_test_directory_enumerated") is not False:
            raise RuntimeError("prior-stage test directory boundary changed")
        if boundary.get("p2_test_tensors_deserialized") is not False:
            raise RuntimeError("prior-stage test tensor boundary changed")
        if boundary.get("architecture_selected") is not False:
            raise RuntimeError("prior stage unexpectedly selected architecture")

    edge_index = torch.from_numpy(
        np.load(paths["edge_index"], allow_pickle=False)
    ).long()
    edge_runtime = validate_edge_index(edge_index)
    return {
        "paths": {key: str(path) for key, path in paths.items()},
        "edge_index": edge_index,
        "edge_runtime": edge_runtime,
        "training_policy": training_policy,
        "operator_contract": operator_contract,
        "task_matrix": task_matrix,
        "test_policy": test_policy,
        "b0_report": b0_report,
        "g1_report": g1_report,
        "g2_report": g2_report,
    }


def write_hold(
    report_dir: Path,
    model_dir: Path,
    operator: str,
    seed: int,
    failure: str,
) -> None:
    report = {
        "stage": STAGE,
        "status": "HOLD",
        "operator": operator,
        "seed": seed,
        "failure": failure,
        "training_started": (
            report_dir / f"{STAGE}_TRAINING_STARTED"
        ).is_file(),
        "scientific_checkpoint_created": any(model_dir.glob("*.pt"))
        if model_dir.is_dir()
        else False,
        "p2_test_directory_enumerated": False,
        "p2_test_tensors_deserialized": False,
        "architecture_selected": False,
    }
    write_json(report_dir / f"{STAGE}.json", report)
    atomic_write(report_dir / HOLD, HOLD + "\n")


def run(args: argparse.Namespace) -> int:
    operator = args.operator
    seed = int(args.seed)
    model_dir = args.model_dir.resolve()
    report_dir = args.report_dir.resolve()

    if operator not in OPERATORS:
        raise ValueError(f"operator must be one of {OPERATORS}")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if PYG_IMPORT_ERROR is not None and operator != "conv1d":
        raise RuntimeError(f"PyTorch Geometric unavailable: {PYG_IMPORT_ERROR}")
    if model_dir.exists() or report_dir.exists():
        raise FileExistsError("model/report destination already exists")
    model_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    contracts = verify_contracts(args)
    preflight_dir = args.preflight_dir.resolve()
    preflight_lock_path = (
        preflight_dir
        / "V5_P2_G3_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT_LOCK.json"
    )
    preflight_complete = (
        preflight_dir
        / "V5_P2_G3_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT_COMPLETE"
    )
    if not preflight_lock_path.is_file() or not preflight_complete.is_file():
        raise FileNotFoundError("clean G3 implementation preflight is not complete")
    preflight_lock = load_json(preflight_lock_path)
    if (
        preflight_lock.get("status")
        != "V5_P2_G3_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT_COMPLETE"
    ):
        raise RuntimeError("clean G3 implementation preflight status changed")
    if preflight_lock.get("scientific_training_authorized") is not True:
        raise RuntimeError("clean G3 implementation did not authorize training")
    if preflight_lock.get("trainer_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("trainer source changed after clean preflight")
    if preflight_lock.get("edge_index_sha256") != EXPECTED_EDGE_SHA:
        raise RuntimeError("preflight edge_index SHA changed")

    set_seed(seed)
    loader_module = import_module(
        args.loader_path.resolve(), f"g3_loader_{operator}_{seed}"
    )
    b3_module = import_module(
        args.b3_model_path.resolve(), f"g3_b3_{operator}_{seed}"
    )
    DatasetClass = loader_module.V5P2PairAlignedPrimary58Dataset
    B3Class = b3_module.P2B3Conv1DOnlyCount4

    train_dataset = DatasetClass(
        root=args.root.resolve(),
        split="train",
        pair_manifest=args.pair_manifest.resolve(),
        window=32,
        stride=8,
    )
    validation_dataset = DatasetClass(
        root=args.root.resolve(),
        split="validation",
        pair_manifest=args.pair_manifest.resolve(),
        window=32,
        stride=8,
    )
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(
            f"train length={len(train_dataset)}, expected {EXPECTED_TRAIN_ITEMS}"
        )
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(
            f"validation length={len(validation_dataset)}, "
            f"expected {EXPECTED_VALIDATION_ITEMS}"
        )

    raw_pos_weights = role_positive_weights(contracts["b0_report"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = PairBlockBatchSampler(
        train_dataset,
        block_batch_size=PAIR_BLOCK_BATCH_SIZE,
        shuffle=True,
        seed=seed,
    )
    validation_sampler = PairBlockBatchSampler(
        validation_dataset,
        block_batch_size=PAIR_BLOCK_BATCH_SIZE,
        shuffle=False,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )

    reference_b3 = B3Class()
    if (
        sum(parameter.numel() for parameter in reference_b3.parameters())
        != EXPECTED_FULL_B3_PARAMETERS
    ):
        raise RuntimeError("frozen B3 parameter count changed")
    reference_b3_initial_sha = sha256_state_dict(reference_b3.state_dict())

    model = CleanRoleAwareBaseline(
        reference_b3=reference_b3,
        operator=operator,
        base_edge_index=contracts["edge_index"],
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_CLEAN_PARAMETERS[operator]:
        raise RuntimeError(
            f"clean {operator} parameter count={parameter_count}, "
            f"expected {EXPECTED_CLEAN_PARAMETERS[operator]}"
        )

    equivalence_items = [train_dataset[index] for index in (0, 1)]
    equivalence_batch = {
        key: torch.stack([item[key] for item in equivalence_items], dim=0)
        for key in equivalence_items[0]
    }
    reference_b3.eval()
    model.eval()
    with torch.no_grad():
        reference_output = reference_b3(
            equivalence_batch["x"], equivalence_batch["physical_port_mask"]
        )
        clean_output = model.logits_without_graph(
            equivalence_batch["x"], equivalence_batch["physical_port_mask"]
        )
    equivalence_errors = {
        "source": float(
            (reference_output["source_logits"] - clean_output["source"])
            .abs().max()
        ),
        "transit": float(
            (reference_output["transit_logits"] - clean_output["transit"])
            .abs().max()
        ),
        "victim": float(
            (reference_output["victim_logits"] - clean_output["victim"])
            .abs().max()
        ),
        "path": float(
            (reference_output["path_logits"] - clean_output["path"])
            .abs().max()
        ),
    }
    if any(error != 0.0 for error in equivalence_errors.values()):
        raise RuntimeError(
            f"clean encoder/role-head equivalence failed: {equivalence_errors}"
        )
    del reference_b3

    model = model.to(device)
    initial_state_sha = sha256_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        threshold=1e-4,
        threshold_mode="abs",
        cooldown=0,
        min_lr=1e-5,
    )
    pos_weights = {
        role: torch.tensor(
            raw_pos_weights[role], dtype=torch.float32, device=device
        )
        for role in ROLES
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    checkpoint_path = (
        model_dir / f"v5_p2_g3_{operator}_seed_{seed}_best.pt"
    )
    history_path = (
        report_dir
        / f"V5_P2_G3_{operator.upper()}_SEED_{seed}_HISTORY.csv"
    )
    progress_path = (
        report_dir
        / f"V5_P2_G3_{operator.upper()}_SEED_{seed}_PROGRESS.json"
    )
    predictions_path = (
        report_dir
        / f"V5_P2_G3_{operator.upper()}_SEED_{seed}_VALIDATION_PREDICTIONS.npz"
    )
    threshold_csv_path = (
        report_dir
        / f"V5_P2_G3_{operator.upper()}_SEED_{seed}_THRESHOLD_SWEEP.csv"
    )

    best_rank: tuple[float, ...] | None = None
    best_epoch: int | None = None
    best_validation_metrics: dict[str, Any] | None = None
    best_checkpoint_sha: str | None = None
    best_early_stop_score = -float("inf")
    early_stop_counter = 0
    stopped_early = False
    history_rows: list[dict[str, Any]] = []
    start_time = time.time()

    print("===== V5 P2-G3 ROLE-AWARE MULTILABEL SINGLE RUN =====")
    print(f"operator: {operator}")
    print(f"seed: {seed}")
    print(f"device: {device}")
    print(f"train_items: {len(train_dataset)}")
    print(f"validation_items: {len(validation_dataset)}")
    print(f"train_batches: {len(train_loader)}")
    print(f"validation_batches: {len(validation_loader)}")
    print(f"parameter_count: {parameter_count}")
    for role in ROLES:
        print(f"{role}_pos_weight: {raw_pos_weights[role]}")
    print("roles_mutually_exclusive: false")
    print("test_directory_enumerated: false")
    print("test_tensors_deserialized: false")
    atomic_write(
        report_dir / f"{STAGE}_TRAINING_STARTED",
        f"{STAGE}_TRAINING_STARTED\n",
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            pos_weights=pos_weights,
        )
        validation_metrics = validate(
            model=model,
            loader=validation_loader,
            device=device,
            pos_weights=pos_weights,
            return_predictions=False,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        candidate_rank = checkpoint_rank(validation_metrics, epoch)
        is_best = best_rank is None or candidate_rank > best_rank
        if is_best:
            best_rank = candidate_rank
            best_epoch = epoch
            best_validation_metrics = validation_metrics
            checkpoint_payload = {
                "stage": STAGE,
                "operator": operator,
                "seed": seed,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "initial_state_sha256": initial_state_sha,
                "reference_b3_initial_state_sha256": reference_b3_initial_sha,
                "role_positive_weights": raw_pos_weights,
                "validation_metrics": validation_metrics,
                "edge_index_sha256": EXPECTED_EDGE_SHA,
                "test_tensors_deserialized": False,
            }
            atomic_torch_save(checkpoint_payload, checkpoint_path)
            best_checkpoint_sha = sha256_file(checkpoint_path)

        selection_score = float(validation_metrics["selection_score"])
        if selection_score > best_early_stop_score + EARLY_STOP_MIN_DELTA:
            best_early_stop_score = selection_score
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        scheduler.step(selection_score)
        epoch_seconds = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "selection_score": selection_score,
            "source_ap": validation_metrics["roles"]["source"]["average_precision"],
            "transit_ap": validation_metrics["roles"]["transit"]["average_precision"],
            "victim_ap": validation_metrics["roles"]["victim"]["average_precision"],
            "path_ap": validation_metrics["roles"]["path"]["average_precision"],
            "learning_rate": learning_rate,
            "best_epoch": best_epoch,
            "early_stop_counter": early_stop_counter,
            "epoch_seconds": epoch_seconds,
            "maximum_preclip_gradient_norm": train_metrics[
                "maximum_preclip_gradient_norm"
            ],
        }
        for role in ROLES:
            row[f"train_{role}_loss"] = train_metrics[f"{role}_loss"]
            row[f"validation_{role}_loss"] = validation_metrics[f"{role}_loss"]
        history_rows.append(row)
        write_csv(history_path, history_rows)
        write_json(
            progress_path,
            {
                "stage": STAGE,
                "operator": operator,
                "seed": seed,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_checkpoint_sha256": best_checkpoint_sha,
                "early_stop_counter": early_stop_counter,
                "current_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "p2_test_directory_enumerated": False,
                "p2_test_tensors_deserialized": False,
            },
        )
        print(
            f"operator={operator} seed={seed} epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"score={selection_score:.6f} "
            f"source_ap={validation_metrics['roles']['source']['average_precision']:.6f} "
            f"transit_ap={validation_metrics['roles']['transit']['average_precision']:.6f} "
            f"victim_ap={validation_metrics['roles']['victim']['average_precision']:.6f} "
            f"path_ap={validation_metrics['roles']['path']['average_precision']:.6f} "
            f"lr={learning_rate:.6g} best_epoch={best_epoch} "
            f"patience={early_stop_counter}",
            flush=True,
        )

        if epoch >= MIN_EPOCHS and early_stop_counter >= EARLY_STOP_PATIENCE:
            stopped_early = True
            print(
                f"early stopping at epoch {epoch}; "
                f"patience reached {EARLY_STOP_PATIENCE}",
                flush=True,
            )
            break

    if best_epoch is None or best_validation_metrics is None:
        raise RuntimeError("training produced no best checkpoint")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    final_validation = validate(
        model=model,
        loader=validation_loader,
        device=device,
        pos_weights=pos_weights,
        return_predictions=True,
    )
    truths = final_validation.pop("truths")
    scores = final_validation.pop("scores")
    attack_truth = final_validation.pop("attack_truth")

    threshold_results: dict[str, Any] = {}
    threshold_rows: list[dict[str, Any]] = []
    selected_thresholds: dict[str, float] = {}
    for role in ROLES:
        result = tune_role_threshold(
            truths[role], scores[role], attack_truth
        )
        threshold_results[role] = {
            key: value for key, value in result.items() if key != "rows"
        }
        selected_thresholds[role] = float(
            result["selected"]["threshold"]
        )
        for row in result["rows"]:
            threshold_rows.append({"role": role, **row})
    write_csv(threshold_csv_path, threshold_rows)

    selected_metrics = all_role_metrics(
        truths,
        scores,
        attack_truth,
        selected_thresholds,
    )
    joint_exact = joint_exact_metrics(
        truths,
        scores,
        selected_thresholds,
        attack_truth,
    )
    np.savez_compressed(
        predictions_path,
        attack_truth=attack_truth,
        **{f"{role}_truth": truths[role] for role in ROLES},
        **{f"{role}_score": scores[role] for role in ROLES},
    )

    latency_batch = {
        key: value
        for key, value in next(iter(validation_loader)).items()
    }
    latency = measure_inference_latency(model, latency_batch, device)
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    elapsed = time.time() - start_time
    operation_count = analytic_operation_count(operator)

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "operator": operator,
        "seed": seed,
        "task": "C_ROLE_AWARE_MULTILABEL_LOCALIZATION",
        "architecture": {
            "operator_configuration": operator_configuration(operator),
            "parameter_count": parameter_count,
            "input_shape": ["items", 16, 58, 32],
            "graph_width": 64,
            "graph_layers": 0 if operator == "conv1d" else 2,
            "role_heads": {
                role: "Linear(64,32)+ReLU+Linear(32,1)"
                for role in ROLES
            },
            "roles_mutually_exclusive": False,
            "clean_conv1d_equivalence_max_abs_error": equivalence_errors,
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "scheduler": "ReduceLROnPlateau(mode=max,factor=0.5,patience=4,min_lr=1e-5)",
            "maximum_epochs": MAX_EPOCHS,
            "minimum_epochs": MIN_EPOCHS,
            "early_stopping_patience": EARLY_STOP_PATIENCE,
            "early_stopping_min_delta": EARLY_STOP_MIN_DELTA,
            "gradient_clip": GRADIENT_CLIP,
            "item_batch_size": ITEM_BATCH_SIZE,
            "amp": False,
            "loss": "sum of four independently class-weighted BCEWithLogits losses",
            "role_positive_weights": raw_pos_weights,
            "total_elapsed_seconds": elapsed,
            "stopped_early": stopped_early,
            "epochs_completed": len(history_rows),
        },
        "best_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": best_epoch,
            "selection_weights": CHECKPOINT_WEIGHTS,
            "selection_score": best_validation_metrics["selection_score"],
            "validation_metrics_at_selection": best_validation_metrics,
        },
        "post_checkpoint_validation": {
            "threshold_policy": (
                "separate role thresholds; each maximizes "
                "0.60 node F1 + 0.40 exact role-set accuracy on attack items"
            ),
            "threshold_selection": threshold_results,
            "selected_thresholds": selected_thresholds,
            "selected_metrics": selected_metrics,
            "joint_exact_all_four_roles": joint_exact,
            "predictions_npz": str(predictions_path),
            "threshold_sweep_csv": str(threshold_csv_path),
        },
        "cost": {
            "operation_count": operation_count,
            "peak_cuda_memory_allocated_bytes": peak_memory,
            "inference_latency_proxy": latency,
        },
        "provenance": {
            "trainer_sha256": sha256_file(Path(__file__)),
            "loader_sha256": EXPECTED_LOADER_SHA,
            "b3_model_sha256": EXPECTED_B3_MODEL_SHA,
            "edge_index_sha256": EXPECTED_EDGE_SHA,
            "g1_aggregation_report_sha256": EXPECTED_G1_REPORT_SHA,
            "g2_aggregation_report_sha256": EXPECTED_G2_REPORT_SHA,
            "initial_state_sha256": initial_state_sha,
            "reference_b3_initial_state_sha256": reference_b3_initial_sha,
        },
        "security_boundary": {
            "p2_test_directory_enumerated": False,
            "p2_test_tensors_deserialized": False,
            "test_evaluation_performed": False,
            "architecture_selected": False,
            "quantization_performed": False,
            "rtl_generated": False,
        },
    }
    report_path = report_dir / f"{STAGE}.json"
    write_json(report_path, report)
    lock = {
        "status": COMPLETE,
        "operator": operator,
        "seed": seed,
        "report_sha256": sha256_file(report_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history_sha256": sha256_file(history_path),
        "predictions_sha256": sha256_file(predictions_path),
        "threshold_sweep_sha256": sha256_file(threshold_csv_path),
        "trainer_sha256": sha256_file(Path(__file__)),
        "p2_test_directory_enumerated": False,
        "p2_test_tensors_deserialized": False,
        "architecture_selected": False,
    }
    write_json(report_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(report_dir / COMPLETE, COMPLETE + "\n")
    print(COMPLETE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--g0-dir", type=Path, required=True)
    parser.add_argument("--topology-dir", type=Path, required=True)
    parser.add_argument("--b0-r3-dir", type=Path, required=True)
    parser.add_argument("--g1-aggregation-dir", type=Path, required=True)
    parser.add_argument("--g2-aggregation-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--b3-model-path", type=Path, required=True)
    parser.add_argument("--operator", choices=OPERATORS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    report_dir = args.report_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    try:
        return run(args)
    except Exception as exc:
        report_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        failure = f"{type(exc).__name__}: {exc}"
        write_hold(
            report_dir,
            model_dir,
            str(args.operator),
            int(args.seed),
            failure,
        )
        print(f"HOLD: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
