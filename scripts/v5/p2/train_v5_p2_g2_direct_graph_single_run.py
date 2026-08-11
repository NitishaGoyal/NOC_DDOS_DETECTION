#!/usr/bin/env python3
"""V5 P2-G2 direct graph-detection training for one operator and seed.

This stage trains one clean direct graph classifier from:
    conv1d, gcnconv, graphconv, gatconv
with one frozen screening seed from:
    107, 117, 127

Only P2 train and validation are authorized. The script never constructs,
lists, opens, or evaluates the P2 test split. The best checkpoint is selected
with 0.5 graph AUROC + 0.5 graph average precision. A graph threshold is tuned
only after checkpoint selection to maximize validation balanced accuracy.
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
    from torch_geometric.nn import GATConv, GCNConv, GraphConv
except Exception as exc:  # pragma: no cover
    GATConv = None
    GCNConv = None
    GraphConv = None
    PYG_IMPORT_ERROR: str | None = f"{type(exc).__name__}: {exc}"
else:
    PYG_IMPORT_ERROR = None


STAGE = "V5_P2_G2_DIRECT_GRAPH_SINGLE_RUN"
COMPLETE = f"{STAGE}_COMPLETE"
HOLD = f"{STAGE}_HOLD"
OPERATORS = ("conv1d", "gcnconv", "graphconv", "gatconv")
SEEDS = (107, 117, 127)
EXPECTED_TRAIN_ITEMS = 70_166
EXPECTED_VALIDATION_ITEMS = 12_528
EXPECTED_LOADER_SHA = "2725ff993f4f03ebee3d5ffb775b1fedd6b131a9c4c89ed8049f45b249c24ac2"
EXPECTED_B3_MODEL_SHA = "56ee3207d039b60e8e3a898a689cd8e361247c86b7450a3a5e95f423ebe30def"
EXPECTED_EDGE_SHA = "f6b8050bc158de509b0ff1c5d1d7cb1ffe32c08b0f2287398270b1b891b57aff"
EXPECTED_TOPOLOGY_CONTRACT_SHA = "74112ce145cd3347148d4c784be125add7945943e80e98948368f4c076d047af"
EXPECTED_G1_AGGREGATION_REPORT_SHA = "299f33cfbc55d82075a6992793540bdc3d237846b1315caab8361efeee8c2e7a"
EXPECTED_G1_DECISION_SHA = "aede27f98706c0252536f7f79d1e397df2d8a65e3dc7d36e2e6dd9dcd6e7f50d"
EXPECTED_FULL_B3_PARAMETERS = 43_273
EXPECTED_CLEAN_PARAMETERS = {
    "conv1d": 34_561,
    "gcnconv": 42_881,
    "graphconv": 51_073,
    "gatconv": 43_137,
}
MAX_EPOCHS = 100
MIN_EPOCHS = 15
EARLY_STOP_PATIENCE = 12
EARLY_STOP_MIN_DELTA = 1e-4
ITEM_BATCH_SIZE = 256
PAIR_BLOCK_BATCH_SIZE = ITEM_BATCH_SIZE // 2
GRADIENT_CLIP = 1.0
THRESHOLD_GRID = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
CHECKPOINT_WEIGHTS = {"graph_average_precision": 0.5, "graph_auroc": 0.5}


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


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def graph_metrics(truth: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    y = truth.astype(np.int64).reshape(-1)
    s = score.astype(np.float64).reshape(-1)
    if y.shape != s.shape:
        raise ValueError("graph truth and score shapes differ")
    return {
        "auroc": binary_auroc(y, s),
        "average_precision": average_precision(y, s),
        "thresholded": binary_metrics(y, s, threshold),
        "positive_items": int((y == 1).sum()),
        "negative_items": int((y == 0).sum()),
    }


def checkpoint_selection_score(metrics: dict[str, Any]) -> float:
    return float(
        CHECKPOINT_WEIGHTS["graph_average_precision"] * metrics["average_precision"]
        + CHECKPOINT_WEIGHTS["graph_auroc"] * metrics["auroc"]
    )


def tune_graph_threshold(truth: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    y = truth.astype(np.int64).reshape(-1)
    s = score.astype(np.float64).reshape(-1)
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_rank: tuple[float, ...] | None = None
    for threshold in THRESHOLD_GRID:
        metrics = binary_metrics(y, s, float(threshold))
        row = {
            "threshold": float(threshold),
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "fpr": metrics["fpr"],
            "tnr": metrics["tnr"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        }
        rows.append(row)
        rank = (
            float(metrics["balanced_accuracy"]),
            float(metrics["f1"]),
            float(metrics["accuracy"]),
            -float(metrics["fpr"]),
            -abs(float(threshold) - 0.5),
            -float(threshold),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_row = row
    if best_row is None:
        raise RuntimeError("graph threshold tuning produced no candidate")
    selected_metrics = graph_metrics(y, s, float(best_row["threshold"]))
    return {
        "grid": {
            "kind": "uniform_closed_interval",
            "start": 0.0,
            "stop": 1.0,
            "step": 0.001,
            "candidate_count": len(THRESHOLD_GRID),
        },
        "primary_objective": "maximize_validation_balanced_accuracy",
        "tie_break": ["graph_f1", "accuracy", "lower_fpr", "closer_to_0.5", "lower_threshold"],
        "selected": {
            "threshold": float(best_row["threshold"]),
            "objective": float(best_row["balanced_accuracy"]),
            "metrics": selected_metrics,
            "ranking_tuple": list(best_rank) if best_rank is not None else None,
        },
        "rows": rows,
    }


class PairBlockBatchSampler(Sampler[list[int]]):
    """Deterministic pair-aligned ATTACK/CONTROL block sampler."""

    def __init__(self, dataset, *, block_batch_size: int, shuffle: bool, seed: int) -> None:
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


def batched_edge_index(base_edge_index: torch.Tensor, batch_size: int, num_nodes: int, device: torch.device) -> torch.Tensor:
    base = base_edge_index.to(device=device, dtype=torch.long)
    result = torch.cat([base + graph_index * num_nodes for graph_index in range(batch_size)], dim=1)
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
    return {"shape": [2, 48], "directed_edges": 48, "undirected_links": 24, "physical_self_loops": 0}


class CleanDirectGraphBaseline(nn.Module):
    """Clean direct-graph model extracted from a seeded frozen B3 reference."""

    def __init__(self, reference_b3: nn.Module, operator: str, base_edge_index: torch.Tensor) -> None:
        super().__init__()
        if operator not in OPERATORS:
            raise ValueError(f"unsupported operator: {operator}")
        self.operator = operator
        self.input_projection = copy.deepcopy(reference_b3.input_projection)
        self.temporal_blocks = copy.deepcopy(reference_b3.temporal_blocks)
        self.node_projection = copy.deepcopy(reference_b3.node_projection)
        self.graph_projection = copy.deepcopy(reference_b3.graph_projection)
        self.attack_head = copy.deepcopy(reference_b3.attack_head)
        self.register_buffer("base_edge_index", base_edge_index.detach().cpu().long().contiguous())
        if operator == "conv1d":
            self.graph1 = None
            self.graph2 = None
        elif operator == "gcnconv":
            self.graph1 = GCNConv(64, 64, add_self_loops=True, normalize=True)
            self.graph2 = GCNConv(64, 64, add_self_loops=True, normalize=True)
        elif operator == "graphconv":
            self.graph1 = GraphConv(64, 64, aggr="add")
            self.graph2 = GraphConv(64, 64, aggr="add")
        elif operator == "gatconv":
            self.graph1 = GATConv(64, 64, heads=1, concat=False, negative_slope=0.2, dropout=0.0, add_self_loops=True, bias=True)
            self.graph2 = GATConv(64, 64, heads=1, concat=False, negative_slope=0.2, dropout=0.0, add_self_loops=True, bias=True)
        self.activation = nn.ReLU()

    def encode_nodes(self, x: torch.Tensor, physical_port_mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (16, 58, 32):
            raise ValueError(f"x shape={tuple(x.shape)}, expected [B,16,58,32]")
        if physical_port_mask.ndim != 3 or tuple(physical_port_mask.shape[1:]) != (16, 10) or physical_port_mask.shape[0] != x.shape[0]:
            raise ValueError("physical_port_mask must have shape [B,16,10]")
        batch_size, num_nodes, num_features, time_steps = x.shape
        y = x.reshape(batch_size * num_nodes, num_features, time_steps)
        y = F.relu(self.input_projection(y))
        for block in self.temporal_blocks:
            y = block(y)
        temporal_embedding = y[:, :, -1].reshape(batch_size, num_nodes, 64)
        node_input = torch.cat((temporal_embedding, physical_port_mask.to(dtype=temporal_embedding.dtype)), dim=-1)
        return F.relu(self.node_projection(node_input))

    def graph_logits_without_message_passing(self, x: torch.Tensor, physical_port_mask: torch.Tensor) -> torch.Tensor:
        node_embedding = self.encode_nodes(x, physical_port_mask)
        pooled = torch.cat((node_embedding.mean(dim=1), node_embedding.amax(dim=1)), dim=-1)
        graph_embedding = self.graph_projection(pooled)
        return self.attack_head(graph_embedding).squeeze(-1)

    def forward(self, x: torch.Tensor, physical_port_mask: torch.Tensor) -> torch.Tensor:
        node_embedding = self.encode_nodes(x, physical_port_mask)
        batch_size, num_nodes, embedding_dim = node_embedding.shape
        if self.operator != "conv1d":
            flat = node_embedding.reshape(batch_size * num_nodes, embedding_dim)
            edges = batched_edge_index(self.base_edge_index, batch_size, num_nodes, flat.device)
            flat = self.activation(self.graph1(flat, edges))
            flat = self.activation(self.graph2(flat, edges))
            node_embedding = flat.reshape(batch_size, num_nodes, embedding_dim)
        pooled = torch.cat((node_embedding.mean(dim=1), node_embedding.amax(dim=1)), dim=-1)
        graph_embedding = self.graph_projection(pooled)
        logits = self.attack_head(graph_embedding).squeeze(-1)
        if logits.ndim != 1 or logits.shape[0] != x.shape[0]:
            raise RuntimeError(f"attack head returned unexpected shape {tuple(logits.shape)}")
        return logits


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=(device.type == "cuda")) for key, value in batch.items()}


def train_one_epoch(*, model: CleanDirectGraphBaseline, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    model.train()
    item_count = 0
    weighted_loss = 0.0
    maximum_preclip_gradient_norm = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        batch_items = int(batch["x"].shape[0])
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["x"], batch["physical_port_mask"])
        target = batch["y_attack"].float().reshape(-1)
        if logits.shape != target.shape:
            raise RuntimeError("graph logits/target shape mismatch")
        loss = F.binary_cross_entropy_with_logits(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
        maximum_preclip_gradient_norm = max(maximum_preclip_gradient_norm, float(gradient_norm.item()))
        optimizer.step()
        item_count += batch_items
        weighted_loss += float(loss.item()) * batch_items
    if item_count != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"training epoch consumed {item_count} items; expected {EXPECTED_TRAIN_ITEMS}")
    return {"loss": weighted_loss / item_count, "maximum_preclip_gradient_norm": maximum_preclip_gradient_norm}


def validate(*, model: CleanDirectGraphBaseline, loader: DataLoader, device: torch.device, return_predictions: bool) -> dict[str, Any]:
    model.eval()
    item_count = 0
    weighted_loss = 0.0
    truth_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            batch_items = int(batch["x"].shape[0])
            logits = model(batch["x"], batch["physical_port_mask"])
            target = batch["y_attack"].float().reshape(-1)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite validation loss")
            item_count += batch_items
            weighted_loss += float(loss.item()) * batch_items
            truth_parts.append(target.detach().cpu().numpy().astype(np.int64))
            score_parts.append(torch.sigmoid(logits).detach().cpu().numpy())
    if item_count != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation consumed {item_count} items; expected {EXPECTED_VALIDATION_ITEMS}")
    truth = np.concatenate(truth_parts, axis=0)
    score = np.concatenate(score_parts, axis=0)
    metrics = graph_metrics(truth, score, threshold=0.5)
    result: dict[str, Any] = {
        "loss": weighted_loss / item_count,
        "selection_score": checkpoint_selection_score(metrics),
        "graph": metrics,
    }
    if return_predictions:
        result["truth"] = truth
        result["score"] = score
    return result


def checkpoint_rank(validation_metrics: dict[str, Any], epoch: int) -> tuple[float, float, float, float, int]:
    return (
        float(validation_metrics["selection_score"]),
        float(validation_metrics["graph"]["average_precision"]),
        float(validation_metrics["graph"]["auroc"]),
        -float(validation_metrics["loss"]),
        -int(epoch),
    )


def operator_configuration(operator: str) -> dict[str, Any]:
    if operator == "conv1d":
        return {"message_passing": False, "graph_layers": 0}
    if operator == "gcnconv":
        return {"message_passing": True, "graph_layers": 2, "operator": "GCNConv", "self_loops": True, "normalization": "symmetric", "edge_weights": "unit", "bias": True}
    if operator == "graphconv":
        return {"message_passing": True, "graph_layers": 2, "operator": "GraphConv", "aggregation": "add", "stored_physical_self_loops": False, "explicit_root_transform": True, "bias": True}
    if operator == "gatconv":
        return {"message_passing": True, "graph_layers": 2, "operator": "GATConv", "heads": 1, "concat": False, "self_loops": True, "negative_slope": 0.2, "attention_dropout": 0.0, "bias": True, "screening_only": True}
    raise ValueError(f"unknown operator: {operator}")


def analytic_operation_count(operator: str) -> dict[str, int | str]:
    input_projection = 16 * 32 * 58 * 64
    temporal_block = 16 * 32 * 64 * 3 + 16 * 32 * 64 * 64
    temporal_blocks = 4 * temporal_block
    node_projection = 16 * 74 * 64
    graph_projection = 128 * 64
    attack_head = 64
    base_macs = input_projection + temporal_blocks + node_projection + graph_projection + attack_head
    graph_linear_macs = 0
    graph_message_scalar_ops = 0
    if operator == "gcnconv":
        graph_linear_macs = 2 * 16 * 64 * 64
        graph_message_scalar_ops = 2 * (48 + 16) * 64
    elif operator == "graphconv":
        graph_linear_macs = 2 * 2 * 16 * 64 * 64
        graph_message_scalar_ops = 2 * 48 * 64
    elif operator == "gatconv":
        graph_linear_macs = 2 * 16 * 64 * 64
        graph_message_scalar_ops = 2 * (2 * 16 * 64 + (48 + 16) * (2 + 2 * 64))
    pooling_scalar_ops = 16 * 64 + 15 * 64
    return {
        "method": "analytic_per_sample_proxy",
        "base_temporal_node_graph_head_macs": base_macs,
        "graph_linear_macs": graph_linear_macs,
        "graph_message_scalar_ops": graph_message_scalar_ops,
        "pooling_scalar_ops": pooling_scalar_ops,
        "total_linear_macs": base_macs + graph_linear_macs,
    }


def measure_inference_latency(model: CleanDirectGraphBaseline, batch: dict[str, torch.Tensor], device: torch.device, warmup: int = 10, repetitions: int = 50) -> dict[str, float | int]:
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


def verify_contracts(args: argparse.Namespace) -> dict[str, Any]:
    g0 = args.g0_dir.expanduser().resolve()
    topology = args.topology_dir.expanduser().resolve()
    b0 = args.b0_r3_dir.expanduser().resolve()
    g1_freeze = args.g1_freeze_dir.expanduser().resolve()
    g1_aggregation = args.g1_aggregation_dir.expanduser().resolve()
    paths = {
        "operator_contract": g0 / "V5_P2_G0_GRAPH_OPERATOR_CONTRACTS.json",
        "training_policy": g0 / "V5_P2_G0_TRAINING_SELECTION_AND_SEED_POLICY.json",
        "task_matrix": g0 / "V5_P2_G0_TASK_AND_MODEL_MATRIX.json",
        "test_policy": g0 / "V5_P2_G0_P2_TEST_ACCESS_POLICY.json",
        "topology_report": topology / "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT.json",
        "topology_lock": topology / "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT_LOCK.json",
        "edge_index": topology / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
        "b0_report": b0 / "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT.json",
        "b0_lock": b0 / "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT_LOCK.json",
        "g1_decision": g1_freeze / "g1_decision.txt",
        "g1_freeze_marker": g1_freeze / "V5_P2_G1C_VALIDATION_RESULT_FROZEN",
        "g1_aggregation_report": g1_aggregation / "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION.json",
        "g1_aggregation_lock": g1_aggregation / "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION_LOCK.json",
        "g1_aggregation_marker": g1_aggregation / "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION_COMPLETE",
        "loader": args.loader_path.expanduser().resolve(),
        "b3_model": args.b3_model_path.expanduser().resolve(),
        "pair_manifest": args.pair_manifest.expanduser().resolve(),
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
    if sha256_file(paths["g1_aggregation_report"]) != EXPECTED_G1_AGGREGATION_REPORT_SHA:
        raise RuntimeError("frozen G1 aggregation report SHA changed")
    if sha256_file(paths["g1_decision"]) != EXPECTED_G1_DECISION_SHA:
        raise RuntimeError("frozen G1 decision SHA changed")

    operator_contract = load_json(paths["operator_contract"])
    training_policy = load_json(paths["training_policy"])
    task_matrix = load_json(paths["task_matrix"])
    test_policy = load_json(paths["test_policy"])
    topology_report = load_json(paths["topology_report"])
    topology_lock = load_json(paths["topology_lock"])
    b0_report = load_json(paths["b0_report"])
    b0_lock = load_json(paths["b0_lock"])
    g1_aggregation_report = load_json(paths["g1_aggregation_report"])
    g1_aggregation_lock = load_json(paths["g1_aggregation_lock"])

    common = operator_contract.get("common_architecture", {})
    if common.get("graph_layer_count") != 2:
        raise RuntimeError("G0 graph layer count changed")
    if common.get("graph_readout", "identical pooling and graph head across operators") not in ("identical pooling and graph head across operators", None):
        raise RuntimeError("unexpected graph readout contract")
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
    graph_task = training_policy.get("task_selection", {}).get("B_DIRECT_GRAPH_DETECTION", {})
    if graph_task.get("checkpoint_score") != {"graph_auroc": 0.5, "graph_average_precision": 0.5}:
        raise RuntimeError("direct graph checkpoint score changed")
    if graph_task.get("loss") != "graph_bce_with_logits":
        raise RuntimeError("direct graph loss changed")
    if graph_task.get("post_checkpoint_threshold_objective") != "maximize validation balanced accuracy":
        raise RuntimeError("direct graph threshold objective changed")
    task_b = next((task for task in task_matrix.get("tasks", []) if task.get("task_id") == "B_DIRECT_GRAPH_DETECTION"), None)
    if not isinstance(task_b, dict):
        raise RuntimeError("task B missing from matrix")
    if task_b.get("participating_losses") != ["graph_bce"]:
        raise RuntimeError("G2 participating loss changed")
    if task_b.get("targets") != ["graph_attack"]:
        raise RuntimeError("G2 graph target changed")
    if task_b.get("graph_score") != "direct learned graph logit":
        raise RuntimeError("G2 graph score changed")
    if task_b.get("graph_readout") != "identical pooling and graph head across operators":
        raise RuntimeError("G2 graph readout changed")
    access = test_policy.get("g1_to_g6", {})
    if access.get("p2_test_tensor_access_allowed") is not False or access.get("p2_test_directory_enumeration_allowed") is not False:
        raise RuntimeError("test-access policy unexpectedly changed")
    canonical = topology_report.get("canonical_contract", {})
    if canonical.get("contract_sha256") != EXPECTED_TOPOLOGY_CONTRACT_SHA:
        raise RuntimeError("topology contract SHA changed")
    if topology_lock.get("edge_index_sha256") != EXPECTED_EDGE_SHA:
        raise RuntimeError("topology lock edge SHA changed")
    if b0_report.get("status") != "COMPLETE":
        raise RuntimeError("B0-R3 report is not complete")
    if b0_lock.get("report_sha256") != sha256_file(paths["b0_report"]):
        raise RuntimeError("B0-R3 report SHA mismatch")
    if b0_lock.get("shortcut_block_count") != 0 or b0_lock.get("label_integrity_pass") is not True:
        raise RuntimeError("B0-R3 shortcut/label integrity changed")
    if g1_aggregation_lock.get("status") != "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION_COMPLETE":
        raise RuntimeError("G1 aggregation is not complete")
    if g1_aggregation_lock.get("report_sha256") != EXPECTED_G1_AGGREGATION_REPORT_SHA:
        raise RuntimeError("G1 aggregation lock report SHA changed")
    if g1_aggregation_lock.get("p2_test_directory_enumerated") is not False or g1_aggregation_lock.get("p2_test_tensors_deserialized") is not False:
        raise RuntimeError("G1 security boundary changed")
    if g1_aggregation_report.get("status") != "COMPLETE":
        raise RuntimeError("G1 aggregation report status changed")
    decision_text = paths["g1_decision"].read_text(encoding="utf-8")
    required_lines = ["status=COMPLETE", "test_directory_enumerated=false", "test_tensors_deserialized=false", "architecture_selected=false", "g2_status=NOT_STARTED"]
    for line in required_lines:
        if line not in decision_text:
            raise RuntimeError(f"frozen G1 decision missing line: {line}")

    edge_index = torch.from_numpy(np.load(paths["edge_index"], allow_pickle=False)).long()
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
        "g1_aggregation_report": g1_aggregation_report,
    }


def write_hold(report_dir: Path, model_dir: Path, operator: str, seed: int, failure: str) -> None:
    report = {
        "stage": STAGE,
        "status": "HOLD",
        "operator": operator,
        "seed": seed,
        "failure": failure,
        "training_started": (report_dir / f"{STAGE}_TRAINING_STARTED").is_file(),
        "scientific_checkpoint_created": any(model_dir.glob("*.pt")) if model_dir.is_dir() else False,
        "p2_test_directory_enumerated": False,
        "p2_test_tensors_deserialized": False,
        "architecture_selected": False,
    }
    write_json(report_dir / f"{STAGE}.json", report)
    atomic_write(report_dir / HOLD, HOLD + "\n")


def run(args: argparse.Namespace) -> int:
    operator = str(args.operator)
    seed = int(args.seed)
    model_dir = args.model_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
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
    preflight_dir = args.preflight_dir.expanduser().resolve()
    preflight_lock_path = preflight_dir / "V5_P2_G2_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT_LOCK.json"
    preflight_complete = preflight_dir / "V5_P2_G2_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT_COMPLETE"
    if not preflight_lock_path.is_file() or not preflight_complete.is_file():
        raise FileNotFoundError("clean G2 implementation preflight is not complete")
    preflight_lock = load_json(preflight_lock_path)
    if preflight_lock.get("status") != "V5_P2_G2_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT_COMPLETE":
        raise RuntimeError("clean G2 preflight status changed")
    if preflight_lock.get("scientific_training_authorized") is not True:
        raise RuntimeError("clean G2 implementation did not authorize training")
    if preflight_lock.get("trainer_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("trainer source changed after G2 preflight")
    if preflight_lock.get("edge_index_sha256") != EXPECTED_EDGE_SHA:
        raise RuntimeError("preflight edge SHA changed")

    set_seed(seed)
    loader_module = import_module(args.loader_path.expanduser().resolve(), f"g2_loader_{operator}_{seed}")
    b3_module = import_module(args.b3_model_path.expanduser().resolve(), f"g2_b3_{operator}_{seed}")
    DatasetClass = loader_module.V5P2PairAlignedPrimary58Dataset
    B3Class = b3_module.P2B3Conv1DOnlyCount4
    train_dataset = DatasetClass(root=args.root.expanduser().resolve(), split="train", pair_manifest=args.pair_manifest.expanduser().resolve(), window=32, stride=8)
    validation_dataset = DatasetClass(root=args.root.expanduser().resolve(), split="validation", pair_manifest=args.pair_manifest.expanduser().resolve(), window=32, stride=8)
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}, expected {EXPECTED_TRAIN_ITEMS}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation length={len(validation_dataset)}, expected {EXPECTED_VALIDATION_ITEMS}")

    b0_train = contracts["b0_report"]["label_summaries"]["train"]
    b0_validation = contracts["b0_report"]["label_summaries"]["validation"]
    if int(b0_train["items"]) != EXPECTED_TRAIN_ITEMS or int(b0_validation["items"]) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError("B0 item counts changed")
    train_graph_positive = int(b0_train["graph_positive"])
    train_graph_negative = int(b0_train["graph_negative"])
    validation_graph_positive = int(b0_validation["graph_positive"])
    validation_graph_negative = int(b0_validation["graph_negative"])
    if train_graph_positive + train_graph_negative != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError("B0 training graph counts do not sum to train items")
    if validation_graph_positive + validation_graph_negative != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError("B0 validation graph counts do not sum to validation items")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = PairBlockBatchSampler(train_dataset, block_batch_size=PAIR_BLOCK_BATCH_SIZE, shuffle=True, seed=seed)
    validation_sampler = PairBlockBatchSampler(validation_dataset, block_batch_size=PAIR_BLOCK_BATCH_SIZE, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

    reference_b3 = B3Class()
    reference_parameters = sum(parameter.numel() for parameter in reference_b3.parameters())
    if reference_parameters != EXPECTED_FULL_B3_PARAMETERS:
        raise RuntimeError(f"reference B3 parameters={reference_parameters}, expected {EXPECTED_FULL_B3_PARAMETERS}")
    reference_b3_initial_sha = sha256_state_dict(reference_b3.state_dict())
    reference_graph_projection_sha = sha256_state_dict(reference_b3.graph_projection.state_dict())
    reference_attack_head_sha = sha256_state_dict(reference_b3.attack_head.state_dict())
    model = CleanDirectGraphBaseline(reference_b3, operator, contracts["edge_index"])
    clean_graph_projection_sha = sha256_state_dict(model.graph_projection.state_dict())
    clean_attack_head_sha = sha256_state_dict(model.attack_head.state_dict())
    if clean_graph_projection_sha != reference_graph_projection_sha or clean_attack_head_sha != reference_attack_head_sha:
        raise RuntimeError("clean graph head extraction changed weights")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_CLEAN_PARAMETERS[operator]:
        raise RuntimeError(f"parameter_count={parameter_count}, expected {EXPECTED_CLEAN_PARAMETERS[operator]}")

    equivalence_batch = next(iter(validation_loader))
    reference_b3.eval()
    model.eval()
    with torch.no_grad():
        reference_logits = reference_b3(equivalence_batch["x"], equivalence_batch["physical_port_mask"])["attack_logits"]
        clean_logits = model.graph_logits_without_message_passing(equivalence_batch["x"], equivalence_batch["physical_port_mask"])
    equivalence_max_abs_error = float((reference_logits - clean_logits).abs().max())
    if equivalence_max_abs_error != 0.0:
        raise RuntimeError(f"clean direct-graph extraction equivalence error={equivalence_max_abs_error}")
    del reference_b3

    initial_state_sha = sha256_state_dict(model.state_dict())
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, threshold=1e-4, threshold_mode="abs", min_lr=1e-5)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    checkpoint_path = model_dir / "best_validation_checkpoint.pt"
    history_path = report_dir / "training_history.csv"
    progress_path = report_dir / "progress.json"
    predictions_path = report_dir / "validation_predictions.npz"
    threshold_csv_path = report_dir / "validation_graph_threshold_sweep.csv"
    atomic_write(report_dir / f"{STAGE}_TRAINING_STARTED", f"operator={operator}\nseed={seed}\n")

    print("===== V5 P2-G2 DIRECT GRAPH SINGLE RUN =====")
    print("operator:", operator)
    print("seed:", seed)
    print("device:", device)
    print("train_items:", len(train_dataset))
    print("validation_items:", len(validation_dataset))
    print("train_batches:", len(train_loader))
    print("validation_batches:", len(validation_loader))
    print("parameter_count:", parameter_count)
    print("graph_loss: unweighted_bce_with_logits")
    print("train_graph_positive:", train_graph_positive)
    print("train_graph_negative:", train_graph_negative)
    print("test_directory_enumerated: false")
    print("test_tensors_deserialized: false")

    history_rows: list[dict[str, Any]] = []
    best_epoch: int | None = None
    best_rank: tuple[float, ...] | None = None
    best_checkpoint_sha: str | None = None
    best_validation_metrics: dict[str, Any] | None = None
    early_stop_counter = 0
    stopped_early = False
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        train_metrics = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device)
        validation_metrics = validate(model=model, loader=validation_loader, device=device, return_predictions=False)
        current_rank = checkpoint_rank(validation_metrics, epoch)
        current_score = float(validation_metrics["selection_score"])
        improved = best_rank is None or current_score > float(best_rank[0]) + EARLY_STOP_MIN_DELTA
        if improved:
            best_epoch = epoch
            best_rank = current_rank
            best_validation_metrics = copy.deepcopy(validation_metrics)
            checkpoint = {
                "stage": STAGE,
                "operator": operator,
                "seed": seed,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_metrics": validation_metrics,
                "checkpoint_rank": current_rank,
                "p2_test_directory_enumerated": False,
                "p2_test_tensors_deserialized": False,
            }
            atomic_torch_save(checkpoint, checkpoint_path)
            best_checkpoint_sha = sha256_file(checkpoint_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        scheduler.step(current_score)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        elapsed = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "maximum_preclip_gradient_norm": train_metrics["maximum_preclip_gradient_norm"],
            "validation_loss": validation_metrics["loss"],
            "selection_score": current_score,
            "graph_average_precision": validation_metrics["graph"]["average_precision"],
            "graph_auroc": validation_metrics["graph"]["auroc"],
            "graph_balanced_accuracy_at_0p5": validation_metrics["graph"]["thresholded"]["balanced_accuracy"],
            "graph_f1_at_0p5": validation_metrics["graph"]["thresholded"]["f1"],
            "learning_rate_after_scheduler": learning_rate,
            "is_best": improved,
            "best_epoch": best_epoch,
            "early_stop_patience_counter": early_stop_counter,
            "elapsed_seconds": elapsed,
        }
        history_rows.append(row)
        write_csv(history_path, history_rows)
        write_json(progress_path, {
            "stage": STAGE,
            "status": "RUNNING",
            "operator": operator,
            "seed": seed,
            "completed_epoch": epoch,
            "best_epoch": best_epoch,
            "best_checkpoint_sha256": best_checkpoint_sha,
            "current_validation_metrics": validation_metrics,
            "early_stop_patience_counter": early_stop_counter,
            "current_learning_rate_after_scheduler": learning_rate,
            "p2_test_directory_enumerated": False,
            "p2_test_tensors_deserialized": False,
        })
        print(
            f"operator={operator} seed={seed} epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} val_loss={validation_metrics['loss']:.6f} "
            f"score={current_score:.6f} graph_ap={validation_metrics['graph']['average_precision']:.6f} "
            f"graph_auc={validation_metrics['graph']['auroc']:.6f} "
            f"bal_acc_0p5={validation_metrics['graph']['thresholded']['balanced_accuracy']:.6f} "
            f"lr={learning_rate:.8g} best_epoch={best_epoch} patience={early_stop_counter}"
        )
        if epoch >= MIN_EPOCHS and early_stop_counter >= EARLY_STOP_PATIENCE:
            stopped_early = True
            print(f"early stopping at epoch {epoch}; patience reached {EARLY_STOP_PATIENCE}")
            break

    if best_epoch is None or not checkpoint_path.is_file() or best_checkpoint_sha is None:
        raise RuntimeError("no best checkpoint was created")
    if best_checkpoint_sha != sha256_file(checkpoint_path):
        raise RuntimeError("best checkpoint SHA changed")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    final_validation = validate(model=model, loader=validation_loader, device=device, return_predictions=True)
    truth = final_validation.pop("truth")
    score = final_validation.pop("score")
    threshold_result = tune_graph_threshold(truth, score)
    write_csv(threshold_csv_path, threshold_result.pop("rows"))
    np.savez_compressed(predictions_path, y_attack=truth.astype(np.uint8), graph_score=score.astype(np.float32))

    representative_batch = next(iter(validation_loader))
    latency = measure_inference_latency(model, representative_batch, device)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    total_elapsed = time.time() - start_time
    selected_threshold = threshold_result["selected"]

    run_core = {
        "stage": STAGE,
        "status": "COMPLETE",
        "operator": operator,
        "seed": seed,
        "architecture": {
            "name": f"CLEAN_B3_DIRECT_GRAPH_{operator.upper()}",
            "parameter_count": parameter_count,
            "expected_parameter_count": EXPECTED_CLEAN_PARAMETERS[operator],
            "graph_layers": 0 if operator == "conv1d" else 2,
            "graph_width": 64,
            "operator_configuration": operator_configuration(operator),
            "activation": "ReLU",
            "dropout": 0.0,
            "residual": False,
            "readout": "concatenate node mean and node max, Linear(128,64), ReLU, Linear(64,1)",
            "readout_identical_across_operators": True,
            "initial_state_sha256": initial_state_sha,
            "reference_b3_initial_state_sha256": reference_b3_initial_sha,
            "reference_graph_projection_sha256": reference_graph_projection_sha,
            "clean_graph_projection_sha256": clean_graph_projection_sha,
            "reference_attack_head_sha256": reference_attack_head_sha,
            "clean_attack_head_sha256": clean_attack_head_sha,
            "clean_equivalence_max_abs_error": equivalence_max_abs_error,
            "unused_multitask_heads_present": False,
            "hook_based_extraction_used": False,
        },
        "data": {
            "train_items": len(train_dataset),
            "validation_items": len(validation_dataset),
            "item_batch_size": ITEM_BATCH_SIZE,
            "pair_block_batch_size": PAIR_BLOCK_BATCH_SIZE,
            "train_batches": len(train_loader),
            "validation_batches": len(validation_loader),
            "num_workers": 0,
            "window": 32,
            "stride": 8,
            "input_shape": ["items", 16, 58, 32],
            "pair_manifest": str(args.pair_manifest.expanduser().resolve()),
            "train_graph_positive": train_graph_positive,
            "train_graph_negative": train_graph_negative,
            "validation_graph_positive": validation_graph_positive,
            "validation_graph_negative": validation_graph_negative,
        },
        "loss": {"name": "graph_bce_with_logits", "class_weighted": False, "positive_weight": None},
        "training": {
            "device": str(device),
            "maximum_epochs": MAX_EPOCHS,
            "minimum_epochs": MIN_EPOCHS,
            "completed_epoch": len(history_rows),
            "stopped_early": stopped_early,
            "early_stopping_patience": EARLY_STOP_PATIENCE,
            "early_stopping_minimum_delta": EARLY_STOP_MIN_DELTA,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": {"name": "ReduceLROnPlateau", "mode": "max", "factor": 0.5, "patience": 4, "threshold": 1e-4, "threshold_mode": "abs", "minimum_learning_rate": 1e-5},
            "gradient_clip_global_norm": GRADIENT_CLIP,
            "automatic_mixed_precision": False,
            "deterministic_algorithms": "enabled_warn_only",
            "total_elapsed_seconds": total_elapsed,
        },
        "best_checkpoint": {
            "epoch": best_epoch,
            "checkpoint_selection_weights": CHECKPOINT_WEIGHTS,
            "validation_metrics_at_selection": best_validation_metrics,
            "path": str(checkpoint_path),
            "sha256": best_checkpoint_sha,
            "ranking_tuple": list(best_rank) if best_rank is not None else None,
        },
        "post_checkpoint_validation": {"untuned_metrics": final_validation, "threshold_selection": threshold_result},
        "cost": {"operation_count": analytic_operation_count(operator), "peak_cuda_memory_allocated_bytes": peak_memory, "inference_latency_proxy": latency},
        "artifacts": {
            "history_csv": [str(history_path), sha256_file(history_path)],
            "progress_json": [str(progress_path), sha256_file(progress_path)],
            "checkpoint": [str(checkpoint_path), sha256_file(checkpoint_path)],
            "validation_predictions": [str(predictions_path), sha256_file(predictions_path)],
            "threshold_sweep_csv": [str(threshold_csv_path), sha256_file(threshold_csv_path)],
        },
        "contracts": {
            "topology_contract_sha256": EXPECTED_TOPOLOGY_CONTRACT_SHA,
            "edge_index_sha256": EXPECTED_EDGE_SHA,
            "loader_sha256": EXPECTED_LOADER_SHA,
            "b3_model_sha256": EXPECTED_B3_MODEL_SHA,
            "g1_aggregation_report_sha256": EXPECTED_G1_AGGREGATION_REPORT_SHA,
            "g1_decision_sha256": EXPECTED_G1_DECISION_SHA,
        },
        "security_boundary": {
            "p2_train_used": True,
            "p2_validation_used": True,
            "p2_test_directory_existence_checked": False,
            "p2_test_directory_enumerated": False,
            "p2_test_tensor_files_opened": False,
            "p2_test_tensors_deserialized": False,
            "p2_test_evaluation_performed": False,
            "threshold_tuning_during_training": False,
            "post_checkpoint_validation_threshold_tuning": True,
            "architecture_selected": False,
            "quantization_performed": False,
            "rtl_generated": False,
        },
        "next_stage": "V5_P2_G2_DIRECT_GRAPH_MATRIX_AGGREGATION",
    }
    run_report = {**run_core, "run_contract_sha256": canonical_sha256(run_core)}
    report_path = report_dir / f"{STAGE}.json"
    write_json(report_path, run_report)
    atomic_write(report_dir / COMPLETE, COMPLETE + "\n")
    print(COMPLETE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--g0-dir", type=Path, required=True)
    parser.add_argument("--topology-dir", type=Path, required=True)
    parser.add_argument("--b0-r3-dir", type=Path, required=True)
    parser.add_argument("--g1-freeze-dir", type=Path, required=True)
    parser.add_argument("--g1-aggregation-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--b3-model-path", type=Path, required=True)
    parser.add_argument("--operator", choices=OPERATORS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    report_dir = args.report_dir.expanduser().resolve()
    operator = str(args.operator)
    seed = int(args.seed)
    try:
        return run(args)
    except Exception as exc:
        report_dir.mkdir(parents=True, exist_ok=True)
        failure = f"{type(exc).__name__}: {exc}"
        write_hold(report_dir, args.model_dir.expanduser().resolve(), operator, seed, failure)
        print(HOLD)
        print("FAIL:", failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
