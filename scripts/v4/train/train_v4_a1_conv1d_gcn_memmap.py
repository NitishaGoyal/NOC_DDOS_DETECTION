#!/usr/bin/env python3
"""
V4-A1 frozen Chrono-A1 Conv1D-TemporalGCN memmap trainer.

Modes
-----
source-audit       Verify the frozen V3 source hash and V4-A1 architecture.
loader-preflight   Validate read-only, worker-local memmap loading.
one-batch          Run one forward/backward batch without an optimizer step.
smoke              Run a bounded training-integrity smoke test.
train              Run the full seed-7 diagnostic baseline. The test split is never opened.

The model architecture intentionally preserves the accepted V3 Chrono-A1 model:
Conv1D(24->8, k=3, p=1) -> temporal max -> normalized GCN 8->16
-> normalized GCN 16->8 -> attacker-node head and mean-readout graph head.
"""

from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import inspect
import json
import math
import os
import random
import resource
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


EXPECTED_PARAMETER_COUNT = 882
MODEL_NAME = "Conv1D-TemporalGCN"
ALLOWED_AUTHORIZATION_LEVELS = {"diagnostic_only", "formal", "formal_a1"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_empty_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = list(path.iterdir())
    if existing:
        raise RuntimeError(
            f"Output directory must be empty for a new stage: {path}\n"
            f"Existing entries: {[p.name for p in existing[:20]]}"
        )


def rss_mib() -> float:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    # ru_maxrss is KiB on Linux.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def load_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("metadata.json must contain a JSON object.")
    return data


def split_code_map(metadata: dict[str, Any]) -> dict[str, int]:
    maps = metadata.get("code_maps", {})
    mapping = maps.get("split", {}) if isinstance(maps, dict) else {}
    required = {"train", "val", "test"}
    if not isinstance(mapping, dict) or not required.issubset(mapping):
        raise RuntimeError(
            "metadata.json must contain code_maps.split with train, val and test."
        )
    return {name: int(mapping[name]) for name in required}


def load_split_indices(
    data_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, np.ndarray]:
    split_path = data_dir / "split_id.npy"
    split_ids = np.load(split_path, mmap_mode="r")
    mapping = split_code_map(metadata)
    indices = {
        name: np.flatnonzero(split_ids == code).astype(np.int64)
        for name, code in mapping.items()
    }
    total = sum(len(v) for v in indices.values())
    if total != int(split_ids.shape[0]):
        raise RuntimeError(
            f"Split codes do not cover all samples: covered={total}, total={split_ids.shape[0]}"
        )
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        # Arrays are sorted because flatnonzero returns sorted positions.
        overlap = np.intersect1d(indices[left], indices[right], assume_unique=True)
        if overlap.size:
            raise RuntimeError(f"Split overlap between {left} and {right}: {overlap[:10]}")
    return indices


def validate_training_authorization(decision_path: Path) -> dict[str, Any]:
    if not decision_path.is_file():
        raise FileNotFoundError(decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("training_authorized") is not True:
        raise RuntimeError(f"Training is not authorized by {decision_path}")
    level = str(decision.get("authorization_level", ""))
    if level not in ALLOWED_AUTHORIZATION_LEVELS:
        raise RuntimeError(f"Unsupported authorization level: {level!r}")
    return decision


class V4MemmapDataset(Dataset):
    """Index-only dataset with worker-local, lazily opened read-only memmaps."""

    def __init__(self, data_dir: str | Path, indices: np.ndarray):
        self.data_dir = str(Path(data_dir).resolve())
        self.indices = np.asarray(indices, dtype=np.int64)
        self._x = None
        self._y_graph = None
        self._y_node = None
        self._run_index = None
        self._split_id = None

    def __len__(self) -> int:
        return int(self.indices.size)

    def _open(self) -> None:
        if self._x is None:
            root = Path(self.data_dir)
            self._x = np.load(root / "x.npy", mmap_mode="r")
            self._y_graph = np.load(root / "y_graph.npy", mmap_mode="r")
            self._y_node = np.load(root / "y_node.npy", mmap_mode="r")
            self._run_index = np.load(root / "run_index.npy", mmap_mode="r")
            self._split_id = np.load(root / "split_id.npy", mmap_mode="r")

    def __getitem__(self, local_index: int) -> dict[str, torch.Tensor]:
        self._open()
        global_index = int(self.indices[local_index])
        # copy=True prevents non-writable NumPy warnings and isolates only one sample.
        x = np.array(self._x[global_index], dtype=np.float32, copy=True)
        y_graph = np.float32(self._y_graph[global_index])
        y_node = np.array(self._y_node[global_index], dtype=np.float32, copy=True)
        return {
            "x": torch.from_numpy(x),
            "y_graph": torch.tensor(y_graph, dtype=torch.float32),
            "y_node": torch.from_numpy(y_node),
            "global_index": torch.tensor(global_index, dtype=torch.int64),
            "run_index": torch.tensor(
                int(self._run_index[global_index]), dtype=torch.int64
            ),
            "split_id": torch.tensor(
                int(self._split_id[global_index]), dtype=torch.int64
            ),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        for key in ("_x", "_y_graph", "_y_node", "_run_index", "_split_id"):
            state[key] = None
        return state


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def build_normalized_adjacency(
    edge_index: np.ndarray,
    num_nodes: int,
) -> torch.Tensor:
    """Build A_hat = D^{-1/2} A D^{-1/2}; edge list already includes self-loops."""
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    src = edge_index[0]
    dst = edge_index[1]
    for source, destination in zip(src, dst):
        adjacency[int(destination), int(source)] = 1.0
    degree = adjacency.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    d_inv_sqrt = torch.diag(degree_inv_sqrt)
    return d_inv_sqrt @ adjacency @ d_inv_sqrt


# ---------------------------------------------------------------------------
# Frozen Chrono-A1 model classes. Do not change for V4-A1.
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        agg = torch.einsum("ij,bjf->bif", a_hat, h)
        out = self.linear(agg)
        return out


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        in_features: int = 2,
        embedding_dim: int = 8,
        kernel_size: int = 3,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=embedding_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, time_steps, feature_dim = x.shape
        x = x.reshape(batch_size * num_nodes, time_steps, feature_dim)
        x = x.permute(0, 2, 1)
        h = F.relu(self.conv(x))
        h = torch.max(h, dim=2).values
        return h.reshape(batch_size, num_nodes, -1)


class TemporalGCN(nn.Module):
    def __init__(
        self,
        input_features: int,
        temporal_dim: int = 8,
        gcn_hidden: int = 16,
        gcn_out: int = 8,
    ):
        super().__init__()
        self.temporal = TemporalEncoder(input_features, temporal_dim, kernel_size=3)
        self.gcn1 = GCNLayer(temporal_dim, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_out)
        self.node_head = nn.Linear(gcn_out, 1)
        self.graph_head = nn.Linear(gcn_out, 1)

    def forward(
        self,
        x: torch.Tensor,
        a_hat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.temporal(x)
        h = F.relu(self.gcn1(h, a_hat))
        h = F.relu(self.gcn2(h, a_hat))
        node_logits = self.node_head(h).squeeze(-1)
        graph_embedding = h.mean(dim=1)
        graph_logits = self.graph_head(graph_embedding).squeeze(-1)
        return graph_logits, node_logits


@dataclass
class BinaryCounts:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, truth: torch.Tensor, pred: torch.Tensor) -> None:
        truth_b = truth.bool()
        pred_b = pred.bool()
        self.tp += int((truth_b & pred_b).sum().item())
        self.tn += int((~truth_b & ~pred_b).sum().item())
        self.fp += int((~truth_b & pred_b).sum().item())
        self.fn += int((truth_b & ~pred_b).sum().item())

    def metrics(self) -> dict[str, float | int]:
        total = self.tp + self.tn + self.fp + self.fn
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        return {
            "accuracy": (self.tp + self.tn) / max(total, 1),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": self.fp / max(self.fp + self.tn, 1),
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }


class StreamingTaskMetrics:
    def __init__(self, graph_threshold: float, node_threshold: float):
        self.graph_threshold = graph_threshold
        self.node_threshold = node_threshold
        self.graph = BinaryCounts()
        self.node = BinaryCounts()
        self.exact_correct = 0
        self.samples = 0

    @torch.no_grad()
    def update(
        self,
        graph_logits: torch.Tensor,
        node_logits: torch.Tensor,
        y_graph: torch.Tensor,
        y_node: torch.Tensor,
    ) -> None:
        graph_pred = torch.sigmoid(graph_logits) >= self.graph_threshold
        node_pred = torch.sigmoid(node_logits) >= self.node_threshold
        graph_truth = y_graph >= 0.5
        node_truth = y_node >= 0.5
        self.graph.update(graph_truth, graph_pred)
        self.node.update(node_truth.reshape(-1), node_pred.reshape(-1))
        self.exact_correct += int(
            torch.all(node_truth == node_pred, dim=1).sum().item()
        )
        self.samples += int(y_graph.shape[0])

    def metrics(self) -> dict[str, Any]:
        graph = self.graph.metrics()
        node = self.node.metrics()
        return {
            "acc": graph["accuracy"],
            "precision": graph["precision"],
            "recall": graph["recall"],
            "f1": graph["f1"],
            "fpr": graph["fpr"],
            "tn": graph["tn"],
            "fp": graph["fp"],
            "fn": graph["fn"],
            "tp": graph["tp"],
            "node_accuracy": node["accuracy"],
            "node_precision": node["precision"],
            "node_recall": node["recall"],
            "node_f1": node["f1"],
            "node_tn": node["tn"],
            "node_fp": node["fp"],
            "node_fn": node["fn"],
            "node_tp": node["tp"],
            "exact_localization": self.exact_correct / max(self.samples, 1),
            "samples": self.samples,
        }


def model_signature(model: TemporalGCN) -> dict[str, Any]:
    parameter_count = sum(p.numel() for p in model.parameters())
    return {
        "model_name": MODEL_NAME,
        "parameter_count": parameter_count,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "parameter_count_ok": parameter_count == EXPECTED_PARAMETER_COUNT,
        "temporal_conv": {
            "in_channels": model.temporal.conv.in_channels,
            "out_channels": model.temporal.conv.out_channels,
            "kernel_size": list(model.temporal.conv.kernel_size),
            "stride": list(model.temporal.conv.stride),
            "padding": list(model.temporal.conv.padding),
        },
        "gcn1": {
            "in_features": model.gcn1.linear.in_features,
            "out_features": model.gcn1.linear.out_features,
        },
        "gcn2": {
            "in_features": model.gcn2.linear.in_features,
            "out_features": model.gcn2.linear.out_features,
        },
        "node_head": {
            "in_features": model.node_head.in_features,
            "out_features": model.node_head.out_features,
        },
        "graph_head": {
            "in_features": model.graph_head.in_features,
            "out_features": model.graph_head.out_features,
        },
        "graph_readout": "mean(dim=1)",
    }


def validate_dataset_headers(
    data_dir: Path,
    expected_nodes: int = 16,
    expected_window: int = 8,
    expected_features: int = 24,
) -> dict[str, Any]:
    x = np.load(data_dir / "x.npy", mmap_mode="r")
    y_graph = np.load(data_dir / "y_graph.npy", mmap_mode="r")
    y_node = np.load(data_dir / "y_node.npy", mmap_mode="r")
    edge = np.load(data_dir / "edge_index.npy")
    if x.ndim != 4:
        raise ValueError(f"x must be rank 4, got {x.shape}")
    expected_tail = (expected_nodes, expected_window, expected_features)
    if tuple(x.shape[1:]) != expected_tail:
        raise ValueError(f"x tail must be {expected_tail}, got {x.shape[1:]}")
    if y_graph.shape != (x.shape[0],):
        raise ValueError(f"y_graph shape mismatch: {y_graph.shape}")
    if y_node.shape != (x.shape[0], expected_nodes):
        raise ValueError(f"y_node shape mismatch: {y_node.shape}")
    if edge.shape != (2, 64):
        raise ValueError(f"edge_index must be (2,64), got {edge.shape}")
    self_loops = int(np.sum(edge[0] == edge[1]))
    if self_loops != expected_nodes:
        raise ValueError(f"Expected {expected_nodes} self-loops, got {self_loops}")
    return {
        "x_shape": list(x.shape),
        "x_dtype": str(x.dtype),
        "y_graph_shape": list(y_graph.shape),
        "y_graph_dtype": str(y_graph.dtype),
        "y_node_shape": list(y_node.shape),
        "y_node_dtype": str(y_node.dtype),
        "edge_index_shape": list(edge.shape),
        "edge_index_dtype": str(edge.dtype),
        "self_loops": self_loops,
        "directed_non_self_edges": int(edge.shape[1] - self_loops),
    }


def compute_class_weights(
    data_dir: Path,
    train_indices: np.ndarray,
    chunk_size: int = 100_000,
) -> dict[str, float]:
    y_graph = np.load(data_dir / "y_graph.npy", mmap_mode="r")
    y_node = np.load(data_dir / "y_node.npy", mmap_mode="r")
    graph_pos = 0.0
    node_pos = 0.0
    samples = 0
    node_values = 0
    for start in range(0, len(train_indices), chunk_size):
        idx = train_indices[start : start + chunk_size]
        graph_chunk = np.asarray(y_graph[idx], dtype=np.float64)
        node_chunk = np.asarray(y_node[idx], dtype=np.float64)
        graph_pos += float(graph_chunk.sum())
        node_pos += float(node_chunk.sum())
        samples += int(graph_chunk.size)
        node_values += int(node_chunk.size)
    graph_neg = samples - graph_pos
    node_neg = node_values - node_pos
    return {
        "graph_positive": graph_pos,
        "graph_negative": graph_neg,
        "graph_pos_weight": graph_neg / max(graph_pos, 1.0),
        "node_positive": node_pos,
        "node_negative": node_neg,
        "node_pos_weight": node_neg / max(node_pos, 1.0),
        "training_samples": samples,
        "training_node_labels": node_values,
    }


def make_loss_functions(
    weights: dict[str, float],
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    graph_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            weights["graph_pos_weight"], dtype=torch.float32, device=device
        )
    )
    node_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            weights["node_pos_weight"], dtype=torch.float32, device=device
        )
    )
    return graph_loss, node_loss


def run_epoch(
    model: TemporalGCN,
    loader: DataLoader,
    a_hat: torch.Tensor,
    device: torch.device,
    graph_loss_fn: nn.Module,
    node_loss_fn: nn.Module,
    node_loss_weight: float,
    graph_threshold: float,
    node_threshold: float,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None = None,
) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    metrics = StreamingTaskMetrics(graph_threshold, node_threshold)
    loss_sum = 0.0
    sample_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_number, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_number > max_batches:
                break
            x = batch["x"].to(device, non_blocking=True)
            y_graph = batch["y_graph"].to(device, non_blocking=True)
            y_node = batch["y_node"].to(device, non_blocking=True)
            graph_logits, node_logits = model(x, a_hat)
            graph_loss = graph_loss_fn(graph_logits, y_graph)
            node_loss = node_loss_fn(node_logits, y_node)
            loss = graph_loss + node_loss_weight * node_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at batch {batch_number}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"Non-finite gradient in {name}")
                optimizer.step()
            batch_size = int(x.shape[0])
            loss_sum += float(loss.detach().item()) * batch_size
            sample_count += batch_size
            metrics.update(
                graph_logits.detach(), node_logits.detach(), y_graph, y_node
            )
    return loss_sum / max(sample_count, 1), metrics.metrics()


def print_epoch_metrics(prefix: str, loss: float, metrics: dict[str, Any]) -> None:
    print(
        f"{prefix} loss={loss:.4f} "
        f"g_acc={metrics['acc']:.4f} "
        f"g_prec={metrics['precision']:.4f} "
        f"g_rec={metrics['recall']:.4f} "
        f"g_f1={metrics['f1']:.4f} "
        f"g_fpr={metrics['fpr']:.4f} "
        f"node_prec={metrics['node_precision']:.4f} "
        f"node_rec={metrics['node_recall']:.4f} "
        f"node_f1={metrics['node_f1']:.4f} "
        f"exact_loc={metrics['exact_localization']:.4f}"
    )


def architecture_text(signature: dict[str, Any]) -> str:
    return f"""V4-A1 frozen Chrono-A1 architecture
model_name: {signature['model_name']}
parameter_count: {signature['parameter_count']}
input: [B,16,8,24]
temporal: Conv1d(24,8,kernel_size=3,stride=1,padding=1) -> ReLU -> max over time
gcn1: normalized adjacency, Linear(8,16) -> ReLU
gcn2: normalized adjacency, Linear(16,8) -> ReLU
node_head: Linear(8,1) per node -> [B,16]
graph_readout: mean over 16 routers
graph_head: Linear(8,1) -> [B]
"""


def extract_class_source(path: Path, class_names: Iterable[str]) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    output: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in class_names:
            start = node.lineno - 1
            end = node.end_lineno
            output[node.name] = "".join(lines[start:end])
    return output


def source_audit(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    ensure_empty_output(out)
    data_dir = Path(args.data_dir).resolve()
    v3_source = Path(args.v3_source).resolve()
    current_source = Path(__file__).resolve()
    v3_hash = sha256_file(v3_source)
    hash_ok = v3_hash == args.expected_v3_sha256
    headers = validate_dataset_headers(data_dir)
    model = TemporalGCN(
        input_features=24,
        temporal_dim=args.temporal_dim,
        gcn_hidden=args.gcn_hidden,
        gcn_out=args.gcn_out,
    )
    signature = model_signature(model)
    architecture_ok = (
        signature["parameter_count_ok"]
        and signature["temporal_conv"]["in_channels"] == 24
        and signature["temporal_conv"]["out_channels"] == 8
        and signature["temporal_conv"]["kernel_size"] == [3]
        and signature["temporal_conv"]["padding"] == [1]
        and signature["gcn1"] == {"in_features": 8, "out_features": 16}
        and signature["gcn2"] == {"in_features": 16, "out_features": 8}
        and signature["graph_readout"] == "mean(dim=1)"
    )
    names = ["GCNLayer", "TemporalEncoder", "TemporalGCN"]
    v3_classes = extract_class_source(v3_source, names)
    v4_classes = extract_class_source(current_source, names)
    missing = [
        f"{label}:{name}"
        for label, values in (("v3", v3_classes), ("v4", v4_classes))
        for name in names
        if name not in values
    ]
    diff_lines: list[str] = []
    for name in names:
        diff_lines.extend(
            difflib.unified_diff(
                v3_classes.get(name, "").splitlines(),
                v4_classes.get(name, "").splitlines(),
                fromfile=f"V3/{name}",
                tofile=f"V4/{name}",
                lineterm="",
            )
        )
    (out / "architecture_class_diff.txt").write_text(
        "\n".join(diff_lines) + "\n", encoding="utf-8"
    )
    current_model_text = "\n".join(v4_classes.values())
    prohibited_tokens = [
        "GraphConv",
        "GraphSAGE",
        "GATConv",
        "LSTM(",
        "GRU(",
        "node_victim_head",
        "mean_max",
    ]
    prohibited_found = [token for token in prohibited_tokens if token in current_model_text]
    report = {
        "stage": "V4-A1.0 source and architecture audit",
        "v3_source": str(v3_source),
        "v3_expected_sha256": args.expected_v3_sha256,
        "v3_actual_sha256": v3_hash,
        "v3_hash_ok": hash_ok,
        "v4_source": str(current_source),
        "v4_source_sha256": sha256_file(current_source),
        "dataset_headers": headers,
        "model_signature": signature,
        "architecture_ok": architecture_ok,
        "missing_class_definitions": missing,
        "prohibited_tokens_found": prohibited_found,
        "full_source_diff_written": "architecture_class_diff.txt",
        "pass": bool(hash_ok and architecture_ok and not missing and not prohibited_found),
        "note": (
            "The full V4 trainer differs in loader, split, provenance and stage-control "
            "logic. The frozen model is gated by its semantic module signature and 882 "
            "parameter count; the class-level textual diff is retained for inspection."
        ),
    }
    json_dump(out / "v4_a1_source_audit.json", report)
    (out / "model_architecture.txt").write_text(
        architecture_text(signature), encoding="utf-8"
    )
    (out / "source_sha256.txt").write_text(
        f"{v3_hash}  {v3_source}\n"
        f"{sha256_file(current_source)}  {current_source}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print("V4_A1_SOURCE_AUDIT_PASS" if report["pass"] else "V4_A1_SOURCE_AUDIT_FAIL")
    return 0 if report["pass"] else 2


def check_sample(
    dataset: V4MemmapDataset,
    local_index: int,
    expected_split_code: int,
) -> dict[str, Any]:
    sample = dataset[local_index]
    x = sample["x"]
    y_graph = sample["y_graph"]
    y_node = sample["y_node"]
    if tuple(x.shape) != (16, 8, 24):
        raise ValueError(f"Bad x shape: {tuple(x.shape)}")
    if tuple(y_node.shape) != (16,):
        raise ValueError(f"Bad y_node shape: {tuple(y_node.shape)}")
    if not torch.isfinite(x).all():
        raise FloatingPointError("Non-finite x value")
    if not torch.isfinite(y_graph):
        raise FloatingPointError("Non-finite graph label")
    if not torch.isfinite(y_node).all():
        raise FloatingPointError("Non-finite node label")
    if int(sample["split_id"]) != expected_split_code:
        raise RuntimeError("Split membership mismatch")
    return {
        "global_index": int(sample["global_index"]),
        "run_index": int(sample["run_index"]),
        "split_id": int(sample["split_id"]),
        "x_shape": list(x.shape),
        "x_dtype": str(x.dtype),
        "y_graph_shape": list(y_graph.shape),
        "y_node_shape": list(y_node.shape),
        "finite": True,
    }


def loader_preflight(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    ensure_empty_output(out)
    data_dir = Path(args.data_dir).resolve()
    headers = validate_dataset_headers(data_dir)
    metadata = load_metadata(data_dir)
    mapping = split_code_map(metadata)
    splits = load_split_indices(data_dir, metadata)
    run_index = np.load(data_dir / "run_index.npy", mmap_mode="r")
    boundary_positions = np.flatnonzero(np.diff(run_index) != 0)
    selected_boundaries: list[int] = []
    if boundary_positions.size:
        picks = np.linspace(
            0, boundary_positions.size - 1, num=min(10, boundary_positions.size), dtype=int
        )
        for pick in picks:
            boundary = int(boundary_positions[pick])
            selected_boundaries.extend([boundary, boundary + 1])
    checks: dict[str, Any] = {
        "headers": headers,
        "split_counts": {k: int(len(v)) for k, v in splits.items()},
        "rss_mib_before": rss_mib(),
        "samples": {},
        "loader_batches": {},
    }
    datasets = {
        name: V4MemmapDataset(data_dir, idx)
        for name, idx in splits.items()
    }
    checks["samples"]["train_first"] = check_sample(
        datasets["train"], 0, mapping["train"]
    )
    checks["samples"]["train_last"] = check_sample(
        datasets["train"], len(datasets["train"]) - 1, mapping["train"]
    )
    checks["samples"]["val_first"] = check_sample(
        datasets["val"], 0, mapping["val"]
    )
    checks["samples"]["test_first"] = check_sample(
        datasets["test"], 0, mapping["test"]
    )
    # Boundary samples are read through a small diagnostic dataset, not by split.
    boundary_dataset = V4MemmapDataset(
        data_dir, np.asarray(selected_boundaries, dtype=np.int64)
    )
    boundary_records = []
    for i in range(len(boundary_dataset)):
        sample = boundary_dataset[i]
        boundary_records.append(
            {
                "global_index": int(sample["global_index"]),
                "run_index": int(sample["run_index"]),
                "split_id": int(sample["split_id"]),
                "shape": list(sample["x"].shape),
                "finite": bool(torch.isfinite(sample["x"]).all()),
            }
        )
    checks["run_boundary_samples"] = boundary_records
    for workers in (0, args.num_workers):
        key = f"num_workers_{workers}"
        loader = build_loader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=args.pin_memory,
            persistent_workers=False,
            prefetch_factor=args.prefetch_factor,
            seed=args.seed,
        )
        batch_records = []
        for batch_number, batch in enumerate(loader, start=1):
            batch_records.append(
                {
                    "batch": batch_number,
                    "x_shape": list(batch["x"].shape),
                    "y_graph_shape": list(batch["y_graph"].shape),
                    "y_node_shape": list(batch["y_node"].shape),
                    "finite": bool(
                        torch.isfinite(batch["x"]).all()
                        and torch.isfinite(batch["y_graph"]).all()
                        and torch.isfinite(batch["y_node"]).all()
                    ),
                }
            )
            if batch_number >= args.loader_test_batches:
                break
        checks["loader_batches"][key] = batch_records
    checks["rss_mib_after"] = rss_mib()
    checks["rss_growth_mib"] = (
        checks["rss_mib_after"] - checks["rss_mib_before"]
    )
    checks["split_overlap_count"] = 0
    checks["x_full_tensor_materialized"] = False
    checks["worker_local_lazy_memmap"] = True
    checks["pass"] = all(
        record["finite"]
        for records in checks["loader_batches"].values()
        for record in records
    )
    json_dump(out / "loader_preflight.json", checks)
    print(json.dumps(checks, indent=2))
    print("V4_A1_LOADER_PREFLIGHT_PASS" if checks["pass"] else "V4_A1_LOADER_PREFLIGHT_FAIL")
    return 0 if checks["pass"] else 2


def prepare_training_objects(
    args: argparse.Namespace,
    include_val: bool,
) -> dict[str, Any]:
    data_dir = Path(args.data_dir).resolve()
    metadata = load_metadata(data_dir)
    splits = load_split_indices(data_dir, metadata)
    headers = validate_dataset_headers(data_dir)
    edge_index = np.load(data_dir / "edge_index.npy").astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_hat = build_normalized_adjacency(edge_index, 16).to(device)
    model = TemporalGCN(
        input_features=24,
        temporal_dim=args.temporal_dim,
        gcn_hidden=args.gcn_hidden,
        gcn_out=args.gcn_out,
    ).to(device)
    signature = model_signature(model)
    if signature["parameter_count"] != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Parameter count drift: {signature}")
    weights = compute_class_weights(data_dir, splits["train"])
    graph_loss_fn, node_loss_fn = make_loss_functions(weights, device)
    train_dataset = V4MemmapDataset(data_dir, splits["train"])
    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
    )
    result = {
        "data_dir": data_dir,
        "metadata": metadata,
        "splits": splits,
        "headers": headers,
        "device": device,
        "a_hat": a_hat,
        "model": model,
        "signature": signature,
        "weights": weights,
        "graph_loss_fn": graph_loss_fn,
        "node_loss_fn": node_loss_fn,
        "train_loader": train_loader,
    }
    if include_val:
        val_dataset = V4MemmapDataset(data_dir, splits["val"])
        result["val_loader"] = build_loader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            seed=args.seed,
        )
    return result


def one_batch_preflight(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    ensure_empty_output(out)
    validate_training_authorization(Path(args.diagnostic_decision))
    set_seed(args.seed)
    objects = prepare_training_objects(args, include_val=False)
    device = objects["device"]
    model: TemporalGCN = objects["model"]
    batch = next(iter(objects["train_loader"]))
    x = batch["x"].to(device, non_blocking=True)
    y_graph = batch["y_graph"].to(device, non_blocking=True)
    y_node = batch["y_node"].to(device, non_blocking=True)
    a_hat = objects["a_hat"]

    # Explicit intermediate path for shape validation.
    temporal = model.temporal(x)
    gcn1 = F.relu(model.gcn1(temporal, a_hat))
    gcn2 = F.relu(model.gcn2(gcn1, a_hat))
    node_logits = model.node_head(gcn2).squeeze(-1)
    graph_embedding = gcn2.mean(dim=1)
    graph_logits = model.graph_head(graph_embedding).squeeze(-1)

    graph_loss = objects["graph_loss_fn"](graph_logits, y_graph)
    node_loss = objects["node_loss_fn"](node_logits, y_node)
    combined = graph_loss + args.node_loss_weight * node_loss
    model.zero_grad(set_to_none=True)
    combined.backward()

    gradients = {}
    all_finite = True
    any_nonzero = False
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            gradients[name] = {"present": False}
            continue
        finite = bool(torch.isfinite(parameter.grad).all().item())
        norm = float(parameter.grad.norm().item())
        gradients[name] = {"present": True, "finite": finite, "norm": norm}
        all_finite = all_finite and finite
        any_nonzero = any_nonzero or norm > 0.0

    expected_shapes = {
        "input": [args.batch_size, 16, 8, 24],
        "temporal_embedding": [args.batch_size, 16, 8],
        "gcn1_output": [args.batch_size, 16, 16],
        "gcn2_output": [args.batch_size, 16, 8],
        "node_logits": [args.batch_size, 16],
        "graph_embedding": [args.batch_size, 8],
        "graph_logits": [args.batch_size],
    }
    actual_shapes = {
        "input": list(x.shape),
        "temporal_embedding": list(temporal.shape),
        "gcn1_output": list(gcn1.shape),
        "gcn2_output": list(gcn2.shape),
        "node_logits": list(node_logits.shape),
        "graph_embedding": list(graph_embedding.shape),
        "graph_logits": list(graph_logits.shape),
    }
    # Last batch is not involved because we explicitly read the first full batch.
    shapes_ok = actual_shapes == expected_shapes
    report = {
        "stage": "V4-A1.2 one-batch forward/backward preflight",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "expected_shapes": expected_shapes,
        "actual_shapes": actual_shapes,
        "shapes_ok": shapes_ok,
        "graph_loss": float(graph_loss.item()),
        "node_loss": float(node_loss.item()),
        "combined_loss": float(combined.item()),
        "loss_finite": bool(torch.isfinite(combined).item()),
        "parameter_count": objects["signature"]["parameter_count"],
        "gradients": gradients,
        "all_gradients_finite": all_finite,
        "any_nonzero_gradient": any_nonzero,
        "cuda_memory_allocated_mib": (
            torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        ),
        "cuda_memory_reserved_mib": (
            torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0.0
        ),
        "optimizer_step_performed": False,
        "checkpoint_written": False,
        "validation_opened": False,
        "test_opened": False,
    }
    report["pass"] = bool(
        shapes_ok
        and report["loss_finite"]
        and report["parameter_count"] == EXPECTED_PARAMETER_COUNT
        and all_finite
        and any_nonzero
    )
    json_dump(out / "one_batch_preflight.json", report)
    print(json.dumps(report, indent=2))
    print("V4_A1_ONE_BATCH_PREFLIGHT_PASS" if report["pass"] else "V4_A1_ONE_BATCH_PREFLIGHT_FAIL")
    return 0 if report["pass"] else 2


def smoke_test(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    ensure_empty_output(out)
    decision = validate_training_authorization(Path(args.diagnostic_decision))
    set_seed(args.seed)
    objects = prepare_training_objects(args, include_val=True)
    model: TemporalGCN = objects["model"]
    device = objects["device"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    losses: list[float] = []
    train_iter = iter(objects["train_loader"])
    for step in range(1, args.smoke_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(objects["train_loader"])
            batch = next(train_iter)
        x = batch["x"].to(device, non_blocking=True)
        y_graph = batch["y_graph"].to(device, non_blocking=True)
        y_node = batch["y_node"].to(device, non_blocking=True)
        graph_logits, node_logits = model(x, objects["a_hat"])
        graph_loss = objects["graph_loss_fn"](graph_logits, y_graph)
        node_loss = objects["node_loss_fn"](node_logits, y_node)
        loss = graph_loss + args.node_loss_weight * node_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite smoke loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if step == 1 or step % max(1, args.smoke_steps // 10) == 0:
            print(f"smoke_step={step}/{args.smoke_steps} loss={loss.item():.6f}")

    parameter_delta_sq = 0.0
    for name, parameter in model.named_parameters():
        delta = parameter.detach().cpu() - before[name]
        parameter_delta_sq += float(torch.sum(delta * delta).item())
    parameter_delta_l2 = math.sqrt(parameter_delta_sq)

    val_loss, val_metrics = run_epoch(
        model=model,
        loader=objects["val_loader"],
        a_hat=objects["a_hat"],
        device=device,
        graph_loss_fn=objects["graph_loss_fn"],
        node_loss_fn=objects["node_loss_fn"],
        node_loss_weight=args.node_loss_weight,
        graph_threshold=args.graph_threshold,
        node_threshold=args.node_threshold,
        optimizer=None,
        max_batches=args.smoke_val_batches,
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "mode": "smoke",
        "steps": args.smoke_steps,
        "args": vars(args),
        "model_signature": objects["signature"],
        "authorization": decision,
    }
    checkpoint_path = out / "smoke_last_model.pt"
    torch.save(checkpoint, checkpoint_path)

    # Resume-format test.
    reloaded = TemporalGCN(24, args.temporal_dim, args.gcn_hidden, args.gcn_out)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(loaded["model_state_dict"], strict=True)

    first_window = losses[: min(10, len(losses))]
    last_window = losses[-min(10, len(losses)) :]
    report = {
        "stage": "V4-A1.3 short training-integrity smoke test",
        "authorization_level": decision.get("authorization_level"),
        "steps": args.smoke_steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_first_window_mean": float(np.mean(first_window)),
        "loss_last_window_mean": float(np.mean(last_window)),
        "all_losses_finite": bool(np.isfinite(losses).all()),
        "parameter_delta_l2": parameter_delta_l2,
        "parameters_changed": parameter_delta_l2 > 0.0,
        "validation_batches": args.smoke_val_batches,
        "validation_loss": val_loss,
        "validation_metrics": val_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_reload_pass": True,
        "best_checkpoint_written": False,
        "test_split_opened": False,
        "rss_mib": rss_mib(),
        "cuda_peak_memory_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else 0.0
        ),
    }
    report["pass"] = bool(
        report["all_losses_finite"]
        and report["parameters_changed"]
        and report["checkpoint_reload_pass"]
        and np.isfinite(val_loss)
    )
    json_dump(out / "smoke_test_summary.json", report)
    with (out / "smoke_losses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss"])
        writer.writerows(enumerate(losses, start=1))
    print(json.dumps(report, indent=2))
    print("V4_A1_SMOKE_TEST_PASS" if report["pass"] else "V4_A1_SMOKE_TEST_FAIL")
    return 0 if report["pass"] else 2


def save_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    fields = [
        "epoch",
        "epoch_duration_seconds",
        "train_loss",
        "val_loss",
        "train_acc",
        "train_precision",
        "train_recall",
        "train_f1",
        "train_fpr",
        "train_node_precision",
        "train_node_recall",
        "train_node_f1",
        "train_exact_localization",
        "val_acc",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_fpr",
        "val_node_precision",
        "val_node_recall",
        "val_node_f1",
        "val_exact_localization",
        "selection_score",
        "best_score_after_epoch",
        "best_epoch_after_epoch",
        "patience_counter",
        "gpu_peak_allocated_mib",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in history:
            writer.writerow(record)


def full_train(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    ensure_empty_output(out)
    decision = validate_training_authorization(Path(args.diagnostic_decision))
    set_seed(args.seed)
    if args.seed != 7:
        raise RuntimeError("V4-A1 frozen first baseline requires seed 7.")
    objects = prepare_training_objects(args, include_val=True)
    model: TemporalGCN = objects["model"]
    device = objects["device"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    config = {
        **vars(args),
        "data_dir_resolved": str(objects["data_dir"]),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_name": MODEL_NAME,
        "authorization": decision,
        "test_evaluation_during_training": False,
        "checkpoint_score": "validation graph F1 + validation node F1 at fixed 0.50 thresholds",
    }
    json_dump(out / "config.json", config)
    json_dump(out / "class_weights.json", objects["weights"])
    json_dump(
        out / "split_counts.json",
        {name: int(len(idx)) for name, idx in objects["splits"].items()},
    )
    (out / "model_architecture.txt").write_text(
        architecture_text(objects["signature"]), encoding="utf-8"
    )
    provenance = {
        "trainer": str(Path(__file__).resolve()),
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "v3_source": str(Path(args.v3_source).resolve()),
        "v3_source_sha256": sha256_file(Path(args.v3_source).resolve()),
        "expected_v3_source_sha256": args.expected_v3_sha256,
        "v3_source_hash_ok": (
            sha256_file(Path(args.v3_source).resolve()) == args.expected_v3_sha256
        ),
        "diagnostic_decision": str(Path(args.diagnostic_decision).resolve()),
        "diagnostic_decision_sha256": sha256_file(Path(args.diagnostic_decision)),
        "dataset_metadata_sha256": sha256_file(objects["data_dir"] / "metadata.json"),
        "model_signature": objects["signature"],
    }
    if not provenance["v3_source_hash_ok"]:
        raise RuntimeError("Frozen V3 source hash changed; refusing full training.")
    json_dump(out / "source_provenance.json", provenance)
    np.savez_compressed(
        out / "splits.npz",
        train_idx=objects["splits"]["train"],
        val_idx=objects["splits"]["val"],
        test_idx=objects["splits"]["test"],
    )

    best_score = -float("inf")
    best_epoch = -1
    patience_counter = 0
    history: list[dict[str, Any]] = []
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_loss, train_metrics = run_epoch(
            model=model,
            loader=objects["train_loader"],
            a_hat=objects["a_hat"],
            device=device,
            graph_loss_fn=objects["graph_loss_fn"],
            node_loss_fn=objects["node_loss_fn"],
            node_loss_weight=args.node_loss_weight,
            graph_threshold=args.graph_threshold,
            node_threshold=args.node_threshold,
            optimizer=optimizer,
        )
        val_loss, val_metrics = run_epoch(
            model=model,
            loader=objects["val_loader"],
            a_hat=objects["a_hat"],
            device=device,
            graph_loss_fn=objects["graph_loss_fn"],
            node_loss_fn=objects["node_loss_fn"],
            node_loss_weight=args.node_loss_weight,
            graph_threshold=args.graph_threshold,
            node_threshold=args.node_threshold,
            optimizer=None,
        )
        selection_score = float(val_metrics["f1"] + val_metrics["node_f1"])
        improvement = selection_score - best_score
        if improvement > args.min_delta:
            best_score = selection_score
            best_epoch = epoch
            patience_counter = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "A_hat": objects["a_hat"].detach().cpu(),
                "best_epoch": best_epoch,
                "best_val_score": best_score,
                "val_metrics": val_metrics,
                "model_name": MODEL_NAME,
                "model_signature": objects["signature"],
                "authorization_level": decision.get("authorization_level"),
            }
            torch.save(checkpoint, out / "best_model.pt")
            improved = True
        else:
            patience_counter += 1
            improved = False

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "A_hat": objects["a_hat"].detach().cpu(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_score": best_score,
                "patience_counter": patience_counter,
                "model_name": MODEL_NAME,
                "model_signature": objects["signature"],
            },
            out / "last_model.pt",
        )
        duration = time.perf_counter() - epoch_start
        gpu_peak = (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else 0.0
        )
        record = {
            "epoch": epoch,
            "epoch_duration_seconds": duration,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_metrics["acc"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_fpr": train_metrics["fpr"],
            "train_node_precision": train_metrics["node_precision"],
            "train_node_recall": train_metrics["node_recall"],
            "train_node_f1": train_metrics["node_f1"],
            "train_exact_localization": train_metrics["exact_localization"],
            "val_acc": val_metrics["acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_fpr": val_metrics["fpr"],
            "val_node_precision": val_metrics["node_precision"],
            "val_node_recall": val_metrics["node_recall"],
            "val_node_f1": val_metrics["node_f1"],
            "val_exact_localization": val_metrics["exact_localization"],
            "selection_score": selection_score,
            "best_score_after_epoch": best_score,
            "best_epoch_after_epoch": best_epoch,
            "patience_counter": patience_counter,
            "gpu_peak_allocated_mib": gpu_peak,
        }
        history.append(record)
        save_history_csv(out / "training_history.csv", history)
        print(f"\nepoch {epoch:03d} duration={duration:.1f}s gpu_peak={gpu_peak:.1f}MiB")
        print_epoch_metrics("  train", train_loss, train_metrics)
        print_epoch_metrics("  val  ", val_loss, val_metrics)
        print(
            f"  selection_score={selection_score:.6f} "
            f"best_epoch={best_epoch} best_score={best_score:.6f} "
            f"patience={patience_counter}/{args.patience} improved={improved}"
        )
        if patience_counter >= args.patience:
            stopped_early = True
            print(f"EARLY_STOP epoch={epoch}")
            break

    if not (out / "best_model.pt").is_file():
        raise RuntimeError("Training ended without a best_model.pt checkpoint.")
    best_checkpoint = torch.load(
        out / "best_model.pt", map_location="cpu", weights_only=False
    )
    summary = {
        "model": MODEL_NAME,
        "parameter_count": objects["signature"]["parameter_count"],
        "authorization_level": decision.get("authorization_level"),
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_epoch": int(best_checkpoint["best_epoch"]),
        "best_validation_score": float(best_checkpoint["best_val_score"]),
        "best_validation_metrics_at_fixed_threshold_0_5": best_checkpoint["val_metrics"],
        "graph_threshold_during_training": args.graph_threshold,
        "node_threshold_during_training": args.node_threshold,
        "test_evaluated": False,
        "test_threshold_selected": False,
        "next_stage": "V4-A1.5 validation-only threshold selection",
    }
    json_dump(out / "summary.json", summary)
    artifact_hashes = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.csv":
            artifact_hashes.append(
                {"filename": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            )
    with (out / "artifact_hashes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "sha256", "bytes"])
        writer.writeheader()
        writer.writerows(artifact_hashes)
    print("\nV4_A1_FULL_TRAINING_COMPLETE")
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen V4-A1 Chrono-A1 memmap training pipeline."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["source-audit", "loader-preflight", "one-batch", "smoke", "train"],
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--v3-source",
        default="scripts/train_temporal_gcn_v3.py",
    )
    parser.add_argument(
        "--expected-v3-sha256",
        default="2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b",
    )
    parser.add_argument("--diagnostic-decision")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--loader-test-batches", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temporal-dim", type=int, default=8)
    parser.add_argument("--gcn-hidden", type=int, default=16)
    parser.add_argument("--gcn-out", type=int, default=8)
    parser.add_argument("--node-loss-weight", type=float, default=1.0)
    parser.add_argument("--graph-threshold", type=float, default=0.5)
    parser.add_argument("--node-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke-steps", type=int, default=200)
    parser.add_argument("--smoke-val-batches", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid loader settings.")
    if args.patience <= 0 or args.epochs <= 0 or args.min_delta < 0:
        raise ValueError("Invalid training settings.")
    if args.mode in {"one-batch", "smoke", "train"} and not args.diagnostic_decision:
        raise ValueError(f"--diagnostic-decision is required for mode {args.mode}")
    if args.mode == "source-audit":
        return source_audit(args)
    if args.mode == "loader-preflight":
        return loader_preflight(args)
    if args.mode == "one-batch":
        return one_batch_preflight(args)
    if args.mode == "smoke":
        return smoke_test(args)
    if args.mode == "train":
        return full_train(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
