#!/usr/bin/env python3
"""Shared V4-A3 model, dataset, metrics, and safe-artifact utilities.

The primary V4-A3 model is a reusable 4x4 regional expert:
  Conv1D(24->16,k=3,p=1) -> temporal mean+max -> append five physical
  valid-port bits -> local projection 37->16 -> one normalized GCN 16->16
  -> concatenate local/context -> attacker MLP 32->16->1 -> mean+max regional
  embedding [64] -> graph head and five-class attacker-count head.

The physical valid-port mask is distinct from local-region adjacency. For the
standalone 4x4 V4 data they coincide at the mesh boundary. In a hierarchical
8x8/16x16 integration, a cross-region port may be physically valid even though
its neighbour is intentionally absent from the regional GCN adjacency.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MODEL_NAME = "V4-A3 SourcePreserve CountAware TemporalGCN"
EXPERIMENT_DESIGNATION = "a3_sourcepreserve_countaware_rank_mask"
STAGE_LABEL = "V4-A3"
EXPECTED_PARAMETER_COUNT = 2983
NUM_ROUTERS = 16
MESH_ROWS = 4
MESH_COLS = 4
WINDOW_EPOCHS = 8
INPUT_FEATURES = 24
TEMPORAL_DIM = 16
LOCAL_DIM = 16
GRAPH_DIM = 16
NODE_FUSED_DIM = 32
REGIONAL_EMBEDDING_DIM = 64
COUNT_CLASSES = 5
PORT_ORDER = ("local", "north", "east", "south", "west")
PRIMARY_SEED = 7


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Full deterministic algorithms can make scatter/einsum paths slower or fail
    # on some CUDA versions. Seeds and loader generators are fixed; the setting
    # is recorded rather than silently changing operators.
    torch.use_deterministic_algorithms(False)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv_dump(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    ordered.append(key)
                    seen.add(key)
        fields = ordered or ["empty"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def require_files(paths: Iterable[Path], label: str = "required inputs") -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label}:\n  " + "\n  ".join(missing))


def ensure_new_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    entries = list(path.iterdir())
    if entries:
        raise RuntimeError(
            f"Output directory must be empty for a new stage: {path}\n"
            f"Existing entries: {[entry.name for entry in entries[:20]]}"
        )


def load_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("metadata.json must contain a JSON object")
    return value


def split_code_map(metadata: Mapping[str, Any]) -> dict[str, int]:
    code_maps = metadata.get("code_maps", {})
    split_map = code_maps.get("split", {}) if isinstance(code_maps, Mapping) else {}
    required = {"train", "val", "test"}
    if not isinstance(split_map, Mapping) or not required.issubset(split_map):
        raise RuntimeError("metadata.json code_maps.split must contain train/val/test")
    return {name: int(split_map[name]) for name in required}


def load_split_indices(data_dir: Path, metadata: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
    metadata = load_metadata(data_dir) if metadata is None else metadata
    split_ids = np.load(data_dir / "split_id.npy", mmap_mode="r")
    mapping = split_code_map(metadata)
    indices = {
        name: np.flatnonzero(split_ids == code).astype(np.int64)
        for name, code in mapping.items()
    }
    covered = sum(len(index) for index in indices.values())
    if covered != int(split_ids.shape[0]):
        raise RuntimeError(f"Split codes cover {covered}/{split_ids.shape[0]} samples")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = np.intersect1d(indices[left], indices[right], assume_unique=True)
        if overlap.size:
            raise RuntimeError(f"Split overlap {left}/{right}: {overlap[:10].tolist()}")
    return indices


def validate_dataset_headers(data_dir: Path) -> dict[str, Any]:
    required = [
        data_dir / "metadata.json",
        data_dir / "x.npy",
        data_dir / "y_graph.npy",
        data_dir / "y_node.npy",
        data_dir / "edge_index.npy",
        data_dir / "run_index.npy",
        data_dir / "split_id.npy",
    ]
    require_files(required, "V4 dataset inputs")
    x = np.load(data_dir / "x.npy", mmap_mode="r")
    y_graph = np.load(data_dir / "y_graph.npy", mmap_mode="r")
    y_node = np.load(data_dir / "y_node.npy", mmap_mode="r")
    edge_index = np.load(data_dir / "edge_index.npy")
    run_index = np.load(data_dir / "run_index.npy", mmap_mode="r")
    split_id = np.load(data_dir / "split_id.npy", mmap_mode="r")
    expected_tail = (NUM_ROUTERS, WINDOW_EPOCHS, INPUT_FEATURES)
    checks = {
        "x_rank": x.ndim == 4,
        "x_tail": tuple(x.shape[1:]) == expected_tail,
        "y_graph": tuple(y_graph.shape) == (x.shape[0],),
        "y_node": tuple(y_node.shape) == (x.shape[0], NUM_ROUTERS),
        "edge_index": tuple(edge_index.shape) == (2, 64),
        "run_index": tuple(run_index.shape) == (x.shape[0],),
        "split_id": tuple(split_id.shape) == (x.shape[0],),
        "finite_label_sample": bool(
            np.isfinite(np.asarray(y_graph[: min(10000, len(y_graph))])).all()
            and np.isfinite(np.asarray(y_node[: min(10000, len(y_node))])).all()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Dataset header audit failed: {checks}")
    self_loops = int(np.sum(edge_index[0] == edge_index[1]))
    if self_loops != NUM_ROUTERS:
        raise RuntimeError(f"Expected {NUM_ROUTERS} self loops, found {self_loops}")
    metadata = load_metadata(data_dir)
    splits = load_split_indices(data_dir, metadata)
    y_node_sample = np.asarray(y_node[: min(200000, y_node.shape[0])], dtype=np.float32)
    count_sample = y_node_sample.sum(axis=1)
    if np.max(count_sample, initial=0) > 4.0 + 1e-6:
        raise RuntimeError("A3 count head supports 0..4 attackers, but labels exceed 4")
    return {
        "checks": checks,
        "x_shape": list(x.shape),
        "x_dtype": str(x.dtype),
        "y_graph_shape": list(y_graph.shape),
        "y_node_shape": list(y_node.shape),
        "edge_index_shape": list(edge_index.shape),
        "self_loops": self_loops,
        "directed_non_self_edges": int(edge_index.shape[1] - self_loops),
        "split_counts": {name: int(len(index)) for name, index in splits.items()},
        "metadata_sha256": sha256_file(data_dir / "metadata.json"),
    }


class V4A3MemmapDataset(Dataset):
    """Index-only dataset with worker-local read-only memmaps."""

    def __init__(self, data_dir: str | Path, indices: np.ndarray, include_metadata: bool = False):
        self.data_dir = str(Path(data_dir).resolve())
        self.indices = np.asarray(indices, dtype=np.int64)
        self.include_metadata = bool(include_metadata)
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return int(self.indices.size)

    def _open(self) -> None:
        if self._arrays:
            return
        root = Path(self.data_dir)
        required = ("x", "y_graph", "y_node", "run_index", "split_id")
        for name in required:
            self._arrays[name] = np.load(root / f"{name}.npy", mmap_mode="r")
        if self.include_metadata:
            for name in (
                "attack_kind_id",
                "attacker_count",
                "profile_id",
                "strength_id",
                "active_core_group_id",
                "seed_id",
            ):
                path = root / f"{name}.npy"
                if path.is_file():
                    self._arrays[name] = np.load(path, mmap_mode="r")

    def __getitem__(self, local_index: int) -> dict[str, torch.Tensor]:
        self._open()
        global_index = int(self.indices[local_index])
        y_node_np = np.array(self._arrays["y_node"][global_index], dtype=np.float32, copy=True)
        attacker_count = int(np.rint(y_node_np.sum()))
        if attacker_count < 0 or attacker_count >= COUNT_CLASSES:
            raise RuntimeError(f"Invalid attacker count {attacker_count} at index {global_index}")
        item: dict[str, torch.Tensor] = {
            "x": torch.from_numpy(
                np.array(self._arrays["x"][global_index], dtype=np.float32, copy=True)
            ),
            "y_graph": torch.tensor(
                np.float32(self._arrays["y_graph"][global_index]), dtype=torch.float32
            ),
            "y_node": torch.from_numpy(y_node_np),
            "attacker_count": torch.tensor(attacker_count, dtype=torch.int64),
            "global_index": torch.tensor(global_index, dtype=torch.int64),
            "run_index": torch.tensor(
                int(self._arrays["run_index"][global_index]), dtype=torch.int64
            ),
            "split_id": torch.tensor(
                int(self._arrays["split_id"][global_index]), dtype=torch.int64
            ),
        }
        for name, array in self._arrays.items():
            if name in {"x", "y_graph", "y_node", "run_index", "split_id"}:
                continue
            item[name] = torch.tensor(int(array[global_index]), dtype=torch.int64)
        return item

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = {}
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


def build_physical_valid_port_mask(
    mesh_rows: int = MESH_ROWS,
    mesh_cols: int = MESH_COLS,
) -> torch.Tensor:
    """Return [routers,5] in local,north,east,south,west order."""
    if mesh_rows * mesh_cols != NUM_ROUTERS:
        raise ValueError("This V4-A3 model expects exactly 16 routers")
    mask = torch.zeros((NUM_ROUTERS, 5), dtype=torch.float32)
    for router in range(NUM_ROUTERS):
        row, col = divmod(router, mesh_cols)
        mask[router, 0] = 1.0
        mask[router, 1] = float(row > 0)
        mask[router, 2] = float(col < mesh_cols - 1)
        mask[router, 3] = float(row < mesh_rows - 1)
        mask[router, 4] = float(col > 0)
    return mask


def validate_port_mask(mask: torch.Tensor) -> dict[str, Any]:
    expected = build_physical_valid_port_mask()
    checks = {
        "shape": tuple(mask.shape) == (NUM_ROUTERS, 5),
        "binary": bool(torch.all((mask == 0) | (mask == 1)).item()),
        "local_always_valid": bool(torch.all(mask[:, 0] == 1).item()),
        "matches_router_order": bool(torch.equal(mask.cpu(), expected)),
        "corner_valid_counts": [int(mask[i].sum().item()) for i in (0, 3, 12, 15)],
        "interior_valid_counts": [int(mask[i].sum().item()) for i in (5, 6, 9, 10)],
    }
    checks["pass"] = bool(
        checks["shape"]
        and checks["binary"]
        and checks["local_always_valid"]
        and checks["matches_router_order"]
        and checks["corner_valid_counts"] == [3, 3, 3, 3]
        and checks["interior_valid_counts"] == [5, 5, 5, 5]
    )
    return checks


def build_normalized_adjacency(edge_index: np.ndarray, num_nodes: int = NUM_ROUTERS) -> torch.Tensor:
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for source, destination in zip(edge_index[0], edge_index[1]):
        adjacency[int(destination), int(source)] = 1.0
    degree = adjacency.sum(dim=1)
    inv = torch.pow(degree, -0.5)
    inv[torch.isinf(inv)] = 0.0
    diagonal = torch.diag(inv)
    return diagonal @ adjacency @ diagonal


def manhattan_distance_matrix(rows: int = MESH_ROWS, cols: int = MESH_COLS) -> torch.Tensor:
    coords = [(router // cols, router % cols) for router in range(rows * cols)]
    matrix = torch.zeros((rows * cols, rows * cols), dtype=torch.int64)
    for i, (ri, ci) in enumerate(coords):
        for j, (rj, cj) in enumerate(coords):
            matrix[i, j] = abs(ri - rj) + abs(ci - cj)
    return matrix


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        aggregate = torch.einsum("ij,bjf->bif", a_hat, h)
        return self.linear(aggregate)


class A3SourcePreserveModel(nn.Module):
    def __init__(self, input_features: int = INPUT_FEATURES):
        super().__init__()
        self.temporal_conv = nn.Conv1d(
            input_features,
            TEMPORAL_DIM,
            kernel_size=3,
            padding=1,
        )
        self.local_projection = nn.Linear(TEMPORAL_DIM * 2 + 5, LOCAL_DIM)
        self.gcn = GCNLayer(LOCAL_DIM, GRAPH_DIM)
        self.attacker_hidden = nn.Linear(LOCAL_DIM + GRAPH_DIM, 16)
        self.attacker_out = nn.Linear(16, 1)
        self.graph_head = nn.Linear(REGIONAL_EMBEDDING_DIM, 1)
        self.count_head = nn.Linear(REGIONAL_EMBEDDING_DIM, COUNT_CLASSES)

    def forward(
        self,
        x: torch.Tensor,
        a_hat: torch.Tensor,
        physical_port_mask: torch.Tensor,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (NUM_ROUTERS, WINDOW_EPOCHS, INPUT_FEATURES):
            raise ValueError(f"Expected [B,16,8,24], got {tuple(x.shape)}")
        batch_size = int(x.shape[0])
        temporal_input = x.reshape(
            batch_size * NUM_ROUTERS, WINDOW_EPOCHS, INPUT_FEATURES
        ).permute(0, 2, 1)
        temporal_sequence = F.relu(self.temporal_conv(temporal_input))
        temporal_mean = temporal_sequence.mean(dim=2)
        temporal_max = temporal_sequence.max(dim=2).values
        temporal_pooled = torch.cat([temporal_mean, temporal_max], dim=1).reshape(
            batch_size, NUM_ROUTERS, TEMPORAL_DIM * 2
        )
        if tuple(physical_port_mask.shape) == (NUM_ROUTERS, 5):
            mask = physical_port_mask.unsqueeze(0).expand(batch_size, -1, -1)
        elif tuple(physical_port_mask.shape) == (batch_size, NUM_ROUTERS, 5):
            mask = physical_port_mask
        else:
            raise ValueError(
                f"physical_port_mask must be [16,5] or [B,16,5], got {tuple(physical_port_mask.shape)}"
            )
        local_input = torch.cat([temporal_pooled, mask.to(temporal_pooled.dtype)], dim=2)
        h_local = F.relu(self.local_projection(local_input))
        h_graph = F.relu(self.gcn(h_local, a_hat))
        h_node = torch.cat([h_local, h_graph], dim=2)
        node_hidden = F.relu(self.attacker_hidden(h_node))
        node_logits = self.attacker_out(node_hidden).squeeze(-1)
        regional_mean = h_node.mean(dim=1)
        regional_max = h_node.max(dim=1).values
        regional_embedding = torch.cat([regional_mean, regional_max], dim=1)
        graph_logits = self.graph_head(regional_embedding).squeeze(-1)
        count_logits = self.count_head(regional_embedding)
        output = {
            "graph_logits": graph_logits,
            "node_logits": node_logits,
            "count_logits": count_logits,
            "regional_embedding": regional_embedding,
        }
        if return_intermediates:
            output.update(
                {
                    "temporal_sequence": temporal_sequence.reshape(
                        batch_size, NUM_ROUTERS, TEMPORAL_DIM, WINDOW_EPOCHS
                    ).permute(0, 1, 3, 2),
                    "temporal_pooled": temporal_pooled,
                    "mask": mask,
                    "local_input": local_input,
                    "h_local": h_local,
                    "h_graph": h_graph,
                    "h_node": h_node,
                }
            )
        return output


def model_signature(model: A3SourcePreserveModel) -> dict[str, Any]:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    gcn_instances = sum(1 for module in model.modules() if isinstance(module, GCNLayer))
    signature = {
        "model_name": MODEL_NAME,
        "experiment_designation": EXPERIMENT_DESIGNATION,
        "stage_label": STAGE_LABEL,
        "parameter_count": parameter_count,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "parameter_count_ok": parameter_count == EXPECTED_PARAMETER_COUNT,
        "input": ["B", NUM_ROUTERS, WINDOW_EPOCHS, INPUT_FEATURES],
        "temporal_conv": {
            "in_channels": model.temporal_conv.in_channels,
            "out_channels": model.temporal_conv.out_channels,
            "kernel_size": list(model.temporal_conv.kernel_size),
            "padding": list(model.temporal_conv.padding),
        },
        "temporal_readout": "concatenate mean and max over 8 epochs",
        "physical_valid_port_mask": list(PORT_ORDER),
        "local_projection": [model.local_projection.in_features, model.local_projection.out_features],
        "gcn_count": gcn_instances,
        "gcn": [model.gcn.linear.in_features, model.gcn.linear.out_features],
        "source_preserving_fusion": "concat(h_local,h_graph)",
        "attacker_head": [32, 16, 1],
        "regional_embedding_dim": REGIONAL_EMBEDDING_DIM,
        "graph_head": [64, 1],
        "count_head": [64, COUNT_CLASSES],
        "graph_gates_attacker_decoder": False,
    }
    return signature


def architecture_contract() -> dict[str, Any]:
    model = A3SourcePreserveModel()
    signature = model_signature(model)
    return {
        "contract_version": "1.1.0",
        "model": signature,
        "training": {
            "seed": PRIMARY_SEED,
            "batch_size": 256,
            "batch_size_fallback": "128 only after recorded CUDA OOM in one-batch preflight",
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "maximum_epochs": 100,
            "patience": 15,
            "minimum_delta": 1e-4,
            "graph_bce_weight": 1.0,
            "node_bce_weight": 1.0,
            "count_ce_weight": 0.5,
            "rank_loss_weight_final": 0.2,
            "rank_margin": 0.2,
            "rank_schedule": {
                "epochs_1_3": 0.0,
                "epochs_4_8": "linear 0.04,0.08,0.12,0.16,0.20",
                "epoch_9_onward": 0.2,
            },
            "test_access_during_training": False,
            "amp_primary_run": False,
            "runtime_schedule": {
                "training_topk_metrics": False,
                "ranking_skipped_when_weight_zero": True,
                "full_validation": "epoch 1, every 3 epochs, and the patience boundary",
                "checkpoint_only_on_full_validation": True,
                "failure_diagnostics": "post-training only",
            },
        },
        "checkpoint_selection": {
            "graph_threshold_selection": "validation-only max graph F1 under FPR<=0.10",
            "score": "graph_f1_fpr10 + attack_exact_topk + attack_node_f1_topk + 0.5*attack_count_accuracy",
            "tie_breaks": [
                "higher graph recall at constrained threshold",
                "lower graph FPR",
                "lower validation total loss",
            ],
            "test_not_used": True,
        },
        "decoder": {
            "count": "argmax five-class count logits",
            "attacker_set": "empty for count=0, otherwise top-k raw node logits",
            "graph_gating": False,
        },
        "rtl_interface": {
            "regional_input": "[16,T,F]",
            "graph_output": "scalar",
            "attacker_output": "[16]",
            "count_output": "[5]",
            "regional_embedding": "[64]",
            "weights_shared_across_regions": True,
            "embedding_dimension_configurable_in_wrapper": True,
            "physical_validity_distinct_from_regional_adjacency": True,
        },
    }


def compute_training_class_weights(
    data_dir: Path,
    train_indices: np.ndarray,
    chunk_size: int = 100_000,
) -> dict[str, float]:
    y_graph = np.load(data_dir / "y_graph.npy", mmap_mode="r")
    y_node = np.load(data_dir / "y_node.npy", mmap_mode="r")
    graph_positive = 0.0
    node_positive = 0.0
    samples = 0
    node_values = 0
    count_histogram = np.zeros(COUNT_CLASSES, dtype=np.int64)
    for start in range(0, len(train_indices), chunk_size):
        index = train_indices[start : start + chunk_size]
        graph = np.asarray(y_graph[index], dtype=np.float64)
        node = np.asarray(y_node[index], dtype=np.float64)
        graph_positive += float(graph.sum())
        node_positive += float(node.sum())
        samples += int(graph.size)
        node_values += int(node.size)
        counts = np.rint(node.sum(axis=1)).astype(np.int64)
        if np.any((counts < 0) | (counts >= COUNT_CLASSES)):
            raise RuntimeError("Training labels include attacker count outside 0..4")
        count_histogram += np.bincount(counts, minlength=COUNT_CLASSES)
    graph_negative = samples - graph_positive
    node_negative = node_values - node_positive
    return {
        "graph_positive": graph_positive,
        "graph_negative": graph_negative,
        "graph_pos_weight": graph_negative / max(graph_positive, 1.0),
        "node_positive": node_positive,
        "node_negative": node_negative,
        "node_pos_weight": node_negative / max(node_positive, 1.0),
        "training_samples": samples,
        "training_node_labels": node_values,
        "count_histogram": count_histogram.tolist(),
        "count_loss_weighting": "unweighted cross entropy; histogram recorded from train only",
    }


def rank_weight_for_epoch(epoch: int, final_weight: float = 0.2) -> float:
    if epoch <= 3:
        return 0.0
    if epoch <= 8:
        return final_weight * float(epoch - 3) / 5.0
    return final_weight


def hard_negative_ranking_loss_reference(
    node_logits: torch.Tensor,
    y_node: torch.Tensor,
    distance_matrix: torch.Tensor,
    margin: float = 0.2,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Per-sample, per-attacker hard-negative hinge loss.

    For each true attacker, choose at most one highest-scoring non-attacker from
    each category: one hop, two hops, and all remaining distances. Every true
    attacker is excluded from every negative pool. Pair losses are averaged per
    attacker, then per sample, so multi-attacker windows do not dominate.
    """
    if node_logits.shape != y_node.shape:
        raise ValueError("node_logits and y_node shapes must match")
    device = node_logits.device
    distance_matrix = distance_matrix.to(device=device)
    sample_losses: list[torch.Tensor] = []
    pair_count = 0
    attack_samples = 0
    for batch_index in range(node_logits.shape[0]):
        true_mask = y_node[batch_index] >= 0.5
        positives = torch.nonzero(true_mask, as_tuple=False).flatten()
        negatives_mask = ~true_mask
        if positives.numel() == 0 or not bool(negatives_mask.any().item()):
            continue
        attack_samples += 1
        attacker_losses: list[torch.Tensor] = []
        for attacker in positives:
            distances = distance_matrix[int(attacker.item())]
            categories = (
                negatives_mask & (distances == 1),
                negatives_mask & (distances == 2),
                negatives_mask & (distances >= 3),
            )
            pair_losses: list[torch.Tensor] = []
            positive_score = node_logits[batch_index, attacker]
            for category_mask in categories:
                candidate_indices = torch.nonzero(category_mask, as_tuple=False).flatten()
                if candidate_indices.numel() == 0:
                    continue
                candidate_scores = node_logits[batch_index, candidate_indices]
                hardest_position = torch.argmax(candidate_scores)
                negative_score = candidate_scores[hardest_position]
                pair_losses.append(F.relu(margin - positive_score + negative_score))
                pair_count += 1
            if pair_losses:
                attacker_losses.append(torch.stack(pair_losses).mean())
        if attacker_losses:
            sample_losses.append(torch.stack(attacker_losses).mean())
    if not sample_losses:
        return node_logits.sum() * 0.0, {"attack_samples": 0, "pairs": 0}
    return torch.stack(sample_losses).mean(), {
        "attack_samples": attack_samples,
        "pairs": pair_count,
    }


def hard_negative_ranking_loss(
    node_logits: torch.Tensor,
    y_node: torch.Tensor,
    distance_matrix: torch.Tensor,
    margin: float = 0.2,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Vectorized equivalent of the reference hard-negative ranking loss.

    For each true attacker, select the highest-scoring non-attacker in each
    available distance category (1 hop, 2 hops, >=3 hops). Average category
    losses per attacker, attackers per sample, then attack samples per batch.
    """
    if node_logits.shape != y_node.shape:
        raise ValueError("node_logits and y_node shapes must match")
    if node_logits.ndim != 2 or node_logits.shape[1] != NUM_ROUTERS:
        raise ValueError("node_logits and y_node must be [B,16]")

    device = node_logits.device
    distances = distance_matrix.to(device=device)
    if distances.shape != (NUM_ROUTERS, NUM_ROUTERS):
        raise ValueError("distance_matrix must be [16,16]")

    positives = y_node >= 0.5                         # [B,P]
    negatives = ~positives                           # [B,N]
    base_candidate = positives[:, :, None] & negatives[:, None, :]  # [B,P,N]

    category_masks = torch.stack(
        (distances == 1, distances == 2, distances >= 3), dim=0
    )                                                # [C,P,N]
    candidate_mask = base_candidate[:, None, :, :] & category_masks[None, :, :, :]
                                                        # [B,C,P,N]
    valid_category = candidate_mask.any(dim=-1)      # [B,C,P]

    negative_scores = node_logits[:, None, None, :].expand(-1, 3, NUM_ROUTERS, -1)
    hardest_negative = negative_scores.masked_fill(~candidate_mask, -torch.inf).max(dim=-1).values
                                                        # [B,C,P]
    positive_scores = node_logits[:, None, :]        # [B,1,P]
    category_loss = F.relu(float(margin) - positive_scores + hardest_negative)
    category_loss = torch.where(valid_category, category_loss, torch.zeros_like(category_loss))

    category_count = valid_category.sum(dim=1)       # [B,P]
    valid_attacker = positives & (category_count > 0)
    attacker_loss = category_loss.sum(dim=1) / category_count.clamp_min(1)
    attacker_loss = torch.where(valid_attacker, attacker_loss, torch.zeros_like(attacker_loss))

    attacker_count = valid_attacker.sum(dim=1)       # [B]
    valid_sample = attacker_count > 0
    sample_loss = attacker_loss.sum(dim=1) / attacker_count.clamp_min(1)

    if not bool(valid_sample.any().item()):
        return node_logits.sum() * 0.0, {"attack_samples": 0, "pairs": 0}

    return sample_loss[valid_sample].mean(), {
        "attack_samples": int(valid_sample.sum().item()),
        "pairs": int(valid_category.sum().item()),
    }


def topk_decode_from_count(
    node_scores: torch.Tensor,
    count_pred: torch.Tensor,
) -> torch.Tensor:
    """Vectorized count-conditioned top-k decoding for [B,16] node scores."""
    if node_scores.ndim != 2 or node_scores.shape[1] != NUM_ROUTERS:
        raise ValueError("node_scores must be [B,16]")
    if count_pred.shape != (node_scores.shape[0],):
        raise ValueError("count_pred must be [B]")
    counts = count_pred.to(torch.long).clamp(min=0, max=NUM_ROUTERS)
    order = torch.argsort(node_scores, dim=1, descending=True)
    selected_sorted = (
        torch.arange(NUM_ROUTERS, device=node_scores.device)[None, :]
        < counts[:, None]
    )
    output = torch.zeros_like(node_scores, dtype=torch.bool)
    output.scatter_(1, order, selected_sorted)
    return output


def numpy_topk_decode(node_scores: np.ndarray, count_pred: np.ndarray) -> np.ndarray:
    scores = np.asarray(node_scores)
    counts = np.asarray(count_pred, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[1] != NUM_ROUTERS:
        raise ValueError("node_scores must be [N,16]")
    output = np.zeros(scores.shape, dtype=bool)
    for k in range(1, COUNT_CLASSES):
        rows = np.flatnonzero(counts == k)
        if rows.size == 0:
            continue
        selected = np.argpartition(scores[rows], -k, axis=1)[:, -k:]
        output[rows[:, None], selected] = True
    return output


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


class EpochMetrics:
    def __init__(
        self,
        graph_threshold: float = 0.5,
        node_threshold: float = 0.5,
        compute_topk: bool = True,
    ):
        self.graph_threshold = float(graph_threshold)
        self.node_threshold = float(node_threshold)
        self.compute_topk = bool(compute_topk)
        self.graph = BinaryCounts()
        self.node_threshold_counts = BinaryCounts()
        self.node_topk_counts = BinaryCounts()
        self.attack_node_topk_counts = BinaryCounts()
        self.samples = 0
        self.threshold_exact = 0
        self.topk_exact = 0
        self.attack_samples = 0
        self.attack_topk_exact = 0
        self.empty_attack_predictions = 0
        self.attack_count_correct = 0
        self.attack_count_abs_error = 0.0
        self.graph_prob_parts: list[np.ndarray] = []
        self.graph_truth_parts: list[np.ndarray] = []

    @torch.no_grad()
    def update(
        self,
        graph_logits: torch.Tensor,
        node_logits: torch.Tensor,
        count_logits: torch.Tensor,
        y_graph: torch.Tensor,
        y_node: torch.Tensor,
        collect_graph_scores: bool = False,
    ) -> None:
        graph_prob = torch.sigmoid(graph_logits)
        node_prob = torch.sigmoid(node_logits)
        graph_pred = graph_prob >= self.graph_threshold
        node_threshold_pred = node_prob >= self.node_threshold
        graph_truth = y_graph >= 0.5
        node_truth = y_node >= 0.5
        count_pred = torch.argmax(count_logits, dim=1)
        true_count = node_truth.sum(dim=1).to(torch.int64)

        self.graph.update(graph_truth, graph_pred)
        self.node_threshold_counts.update(node_truth.reshape(-1), node_threshold_pred.reshape(-1))
        self.threshold_exact += int(torch.all(node_truth == node_threshold_pred, dim=1).sum().item())
        self.samples += int(y_graph.shape[0])

        attack_mask = graph_truth
        attack_count = int(attack_mask.sum().item())
        if attack_count:
            self.attack_samples += attack_count
            self.attack_count_correct += int(
                (count_pred[attack_mask] == true_count[attack_mask]).sum().item()
            )
            self.attack_count_abs_error += float(
                torch.abs(count_pred[attack_mask] - true_count[attack_mask]).sum().item()
            )

        if self.compute_topk:
            topk_pred = topk_decode_from_count(node_logits, count_pred)
            self.node_topk_counts.update(node_truth.reshape(-1), topk_pred.reshape(-1))
            self.topk_exact += int(torch.all(node_truth == topk_pred, dim=1).sum().item())
            if attack_count:
                self.attack_node_topk_counts.update(
                    node_truth[attack_mask].reshape(-1), topk_pred[attack_mask].reshape(-1)
                )
                self.attack_topk_exact += int(
                    torch.all(node_truth[attack_mask] == topk_pred[attack_mask], dim=1).sum().item()
                )
                self.empty_attack_predictions += int((count_pred[attack_mask] == 0).sum().item())

        if collect_graph_scores:
            self.graph_prob_parts.append(graph_prob.detach().cpu().numpy().astype(np.float32))
            self.graph_truth_parts.append(graph_truth.detach().cpu().numpy().astype(np.uint8))

    def metrics(self) -> dict[str, Any]:
        graph = self.graph.metrics()
        node_threshold = self.node_threshold_counts.metrics()
        result: dict[str, Any] = {
            "graph": graph,
            "node_threshold": node_threshold,
            "threshold_exact_localization": self.threshold_exact / max(self.samples, 1),
            "attack_count_accuracy": self.attack_count_correct / max(self.attack_samples, 1),
            "attack_count_mae": self.attack_count_abs_error / max(self.attack_samples, 1),
            "samples": self.samples,
            "attack_samples": self.attack_samples,
            "topk_metrics_computed": self.compute_topk,
        }
        if self.compute_topk:
            result.update(
                {
                    "node_topk": self.node_topk_counts.metrics(),
                    "attack_node_topk": self.attack_node_topk_counts.metrics(),
                    "topk_exact_localization": self.topk_exact / max(self.samples, 1),
                    "attack_topk_exact_localization": self.attack_topk_exact / max(self.attack_samples, 1),
                    "attack_empty_prediction_fraction": self.empty_attack_predictions / max(self.attack_samples, 1),
                }
            )
        else:
            result.update(
                {
                    "node_topk": None,
                    "attack_node_topk": None,
                    "topk_exact_localization": None,
                    "attack_topk_exact_localization": None,
                    "attack_empty_prediction_fraction": None,
                }
            )
        return result

    def graph_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.graph_prob_parts:
            raise RuntimeError("Graph probabilities were not collected")
        return (
            np.concatenate(self.graph_truth_parts, axis=0),
            np.concatenate(self.graph_prob_parts, axis=0),
        )


def binary_metrics_numpy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    truth = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_prob) >= threshold
    tp = int(np.sum(truth & pred))
    tn = int(np.sum(~truth & ~pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / max(fp + tn, 1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def node_metrics_from_prediction(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y_true).astype(bool)
    pred = np.asarray(prediction).astype(bool)
    tp = int(np.sum(truth & pred))
    tn = int(np.sum(~truth & ~pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "node_accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "node_precision": precision,
        "node_recall": recall,
        "node_f1": f1,
        "node_tn": tn,
        "node_fp": fp,
        "node_fn": fn,
        "node_tp": tp,
        "exact_localization": float(np.mean(np.all(truth == pred, axis=1))),
    }


def threshold_values(start: float = 0.05, end: float = 0.95, step: float = 0.01) -> np.ndarray:
    count = int(round((end - start) / step)) + 1
    return np.linspace(start, end, count, dtype=np.float64)


def select_graph_threshold_fpr_cap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fpr_cap: float = 0.10,
    start: float = 0.05,
    end: float = 0.95,
    step: float = 0.01,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        binary_metrics_numpy(y_true, y_prob, float(threshold))
        for threshold in threshold_values(start, end, step)
    ]
    feasible = [row for row in rows if float(row["fpr"]) <= fpr_cap + 1e-12]
    if not feasible:
        raise RuntimeError(f"No validation graph threshold satisfies FPR<={fpr_cap}")
    selected = max(
        feasible,
        key=lambda row: (
            row["f1"],
            row["recall"],
            -row["fpr"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    return selected, rows


def selection_score(validation_metrics: Mapping[str, Any], constrained_graph: Mapping[str, Any]) -> float:
    return float(
        constrained_graph["f1"]
        + validation_metrics["attack_topk_exact_localization"]
        + validation_metrics["attack_node_topk"]["f1"]
        + 0.5 * validation_metrics["attack_count_accuracy"]
    )


def estimate_macs_per_sample() -> dict[str, int]:
    conv = NUM_ROUTERS * WINDOW_EPOCHS * TEMPORAL_DIM * INPUT_FEATURES * 3
    local_projection = NUM_ROUTERS * (TEMPORAL_DIM * 2 + 5) * LOCAL_DIM
    adjacency_aggregate = 64 * GRAPH_DIM
    gcn_linear = NUM_ROUTERS * LOCAL_DIM * GRAPH_DIM
    attacker_hidden = NUM_ROUTERS * NODE_FUSED_DIM * 16
    attacker_out = NUM_ROUTERS * 16
    graph_head = REGIONAL_EMBEDDING_DIM
    count_head = REGIONAL_EMBEDDING_DIM * COUNT_CLASSES
    total = sum(
        (
            conv,
            local_projection,
            adjacency_aggregate,
            gcn_linear,
            attacker_hidden,
            attacker_out,
            graph_head,
            count_head,
        )
    )
    return {
        "temporal_conv": conv,
        "local_projection": local_projection,
        "adjacency_aggregate_approx": adjacency_aggregate,
        "gcn_linear": gcn_linear,
        "attacker_hidden": attacker_hidden,
        "attacker_out": attacker_out,
        "graph_head": graph_head,
        "count_head": count_head,
        "total_approx": total,
    }


def feature_order(data_dir: Path) -> list[str]:
    path = data_dir / "feature_cols.npy"
    if path.is_file():
        values = np.load(path, allow_pickle=True)
        return [str(value) for value in values.tolist()]
    metadata = load_metadata(data_dir)
    values = metadata.get("feature_cols")
    if isinstance(values, list):
        return [str(value) for value in values]
    return [f"feature_{index}" for index in range(INPUT_FEATURES)]
