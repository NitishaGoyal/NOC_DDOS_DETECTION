#!/usr/bin/env python3
# STAGE 9 B1 GENERATED FILE
# Derived from frozen scripts/train_temporal_gcn_v3.py
# Frozen A1 source SHA-256: 2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b
# Do not edit A1 in place. Review the source diff before dry run.


import argparse
import csv
import hashlib
import math
import json
import random
from pathlib import Path

import numpy as np

def load_graph_arrays(data_path):
    data_path = Path(data_path)

    if data_path.is_dir():
        print("loading dataset folder:", data_path)
        out = {
            "x": np.load(data_path / "x.npy", mmap_mode="r"),
            "y_graph": np.load(data_path / "y_graph.npy", mmap_mode="r"),
            "y_node": np.load(data_path / "y_node.npy", mmap_mode="r"),
            "edge_index": np.load(data_path / "edge_index.npy"),
            "run_id": np.load(data_path / "run_id.npy", allow_pickle=True),
            "end_epoch": np.load(data_path / "end_epoch.npy", mmap_mode="r"),
        }

        for name in ["attackers", "split", "feature_cols", "profile", "active_cores", "strength", "seed"]:
            f = data_path / f"{name}.npy"
            out[name] = np.load(f, allow_pickle=True) if f.exists() else None

        return out

    print("loading npz dataset:", data_path)
    d = np.load(data_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def make_v3_splits(data_path):
    d = load_graph_arrays(data_path)

    if d.get("split") is None:
        raise RuntimeError("V3 split requested, but dataset has no split.npy / split field.")

    split = d["split"].astype(str)

    train_idx = np.where(split == "train")[0].astype(np.int64)
    val_idx = np.where(split == "val")[0].astype(np.int64)
    test_idx = np.where(split == "test")[0].astype(np.int64)

    print()
    print("using internal V3 split")
    print("train:", len(train_idx))
    print("val:  ", len(val_idx))
    print("test: ", len(test_idx))

    return train_idx, val_idx, test_idx


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NoCTemporalGraphDataset(Dataset):
    def __init__(self, npz_path: str, indices: np.ndarray):
        d = load_graph_arrays(npz_path)
        self.x = d["x"].astype(np.float32)
        self.y_graph = d["y_graph"].astype(np.float32)
        self.y_node = d["y_node"].astype(np.float32)
        self.run_id = d["run_id"].astype(str)
        self.end_epoch = d["end_epoch"].astype(np.int32)
        self.attackers = d["attackers"].astype(str)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        return {
            "x": torch.from_numpy(self.x[real_idx]),
            "y_graph": torch.tensor(self.y_graph[real_idx], dtype=torch.float32),
            "y_node": torch.from_numpy(self.y_node[real_idx]),
            "run_id": self.run_id[real_idx],
            "end_epoch": int(self.end_epoch[real_idx]),
            "attackers": self.attackers[real_idx],
        }


def make_splits(npz_path: str, split_mode: str):
    d = load_graph_arrays(npz_path)
    run_ids = d["run_id"].astype(str)
    y_graph = d["y_graph"].astype(np.int64)
    unique_runs = sorted(set(run_ids.tolist()))

    print("Runs in dataset:")
    for run_id in unique_runs:
        count = int((run_ids == run_id).sum())
        first_index = np.where(run_ids == run_id)[0][0]
        label = int(y_graph[first_index])
        print(f"  {run_id:38s} samples={count} label={label}")

    if split_mode == "prototype":
        train_idx, val_idx, test_idx = [], [], []
        for run_id in unique_runs:
            idx = np.sort(np.where(run_ids == run_id)[0])
            n = len(idx)
            n_train = int(0.70 * n)
            n_val = int(0.15 * n)
            train_idx.extend(idx[:n_train])
            val_idx.extend(idx[n_train:n_train + n_val])
            test_idx.extend(idx[n_train + n_val:])

        return (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(val_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
        )

    if split_mode == "placement":
        train_attack_runs = {
            "N-0-15-A-1-S20-V2",
            "N-0-15-A-7-S20-V2",
            "N-0-15-A-11-S20-V2",
            "N-0-15-A-1-7-S63-V2",
            "N-0-15-A-1-11-S63-V2",
            "N-0-15-A-7-12-S63-V2",
            "N-0-15-A-1-7-11-S77-V2",
            "N-0-15-A-1-7-12-S77-V2",
        }
        val_attack_runs = {
            "N-0-15-A-12-S20-V2",
            "N-0-15-A-11-12-S63-V2",
        }
        test_attack_runs = {
            "N-0-15-A-1-11-12-S77-V2",
            "N-0-15-A-7-11-12-S77-V2",
        }

        train_idx, val_idx, test_idx = [], [], []
        for run_id in unique_runs:
            idx = np.sort(np.where(run_ids == run_id)[0])
            if "-A-" not in run_id:
                n = len(idx)
                n_train = int(0.70 * n)
                n_val = int(0.15 * n)
                train_idx.extend(idx[:n_train])
                val_idx.extend(idx[n_train:n_train + n_val])
                test_idx.extend(idx[n_train + n_val:])
            elif run_id in train_attack_runs:
                train_idx.extend(idx)
            elif run_id in val_attack_runs:
                val_idx.extend(idx)
            elif run_id in test_attack_runs:
                test_idx.extend(idx)
            else:
                raise RuntimeError(f"Run {run_id!r} was not assigned to any split.")

        return (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(val_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
        )

    raise ValueError(f"Unknown split mode: {split_mode}")


def build_normalized_adjacency(edge_index: np.ndarray, num_nodes: int) -> torch.Tensor:
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


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        aggregated = torch.einsum("ij,bjf->bif", a_hat, h)
        return self.linear(aggregated)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)
    y_true = y_true.astype(np.int64)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "acc": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tnr": float(tnr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def node_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)
    y_true = y_true.astype(np.int64)
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    accuracy = accuracy_score(flat_true, flat_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_true, flat_pred, average="binary", zero_division=0
    )
    exact_localization = np.all(y_pred == y_true, axis=1).mean()
    return {
        "node_acc": float(accuracy),
        "node_precision": float(precision),
        "node_recall": float(recall),
        "node_f1": float(f1),
        "exact_localization": float(exact_localization),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    a_hat: torch.Tensor,
    device: torch.device,
    graph_loss_fn: nn.Module,
    node_loss_fn: nn.Module,
    node_loss_weight: float,
    graph_threshold: float = 0.5,
    node_threshold: float = 0.5,
):
    model.eval()
    graph_probs, graph_true = [], []
    node_probs, node_true = [], []
    total_loss = 0.0
    batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_graph = batch["y_graph"].to(device)
        y_node = batch["y_node"].to(device)
        graph_logits, node_logits = model(x, a_hat)

        graph_loss = graph_loss_fn(graph_logits, y_graph)
        node_loss = node_loss_fn(node_logits, y_node)
        loss = graph_loss + node_loss_weight * node_loss

        total_loss += float(loss.item())
        batches += 1
        graph_probs.append(torch.sigmoid(graph_logits).cpu().numpy())
        graph_true.append(y_graph.cpu().numpy())
        node_probs.append(torch.sigmoid(node_logits).cpu().numpy())
        node_true.append(y_node.cpu().numpy())

    graph_probs = np.concatenate(graph_probs)
    graph_true = np.concatenate(graph_true)
    node_probs = np.concatenate(node_probs)
    node_true = np.concatenate(node_true)

    graph_results = binary_metrics(graph_true, graph_probs, graph_threshold)
    node_results = node_metrics(node_true, node_probs, node_threshold)
    return {
        "loss": float(total_loss / max(1, batches)),
        **graph_results,
        **node_results,
    }


def print_metrics(prefix: str, metrics: dict) -> None:
    print(
        f"{prefix} "
        f"loss={metrics['loss']:.4f} "
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


class TemporalEncoder(nn.Module):
    def __init__(self, in_features: int = 2, embedding_dim: int = 8, kernel_size: int = 3):
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
    def __init__(self, input_features: int, temporal_dim: int = 8, gcn_hidden: int = 16, gcn_out: int = 8):
        super().__init__()
        self.temporal = TemporalEncoder(input_features, temporal_dim, kernel_size=3)
        self.gcn1 = GCNLayer(temporal_dim, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_out)
        self.node_head = nn.Linear(gcn_out, 1)
        self.graph_head = nn.Linear(gcn_out, 1)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor):
        h = self.temporal(x)
        h = F.relu(self.gcn1(h, a_hat))
        h = F.relu(self.gcn2(h, a_hat))
        node_logits = self.node_head(h).squeeze(-1)
        graph_embedding = h.mean(dim=1)
        graph_logits = self.graph_head(graph_embedding).squeeze(-1)
        return graph_logits, node_logits



# ---------------------------------------------------------------------------
# Stage 9 B1: graph-class-balanced, class-conditional run-uniform sampling
# ---------------------------------------------------------------------------

A1_SOURCE_SHA256 = "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b"
A1_CHECKPOINT_SHA256 = "f82ab1d03050820215f407b3ca5b53ccb9117b1f01d8bb253d1f9254cf0c5cde"
A1_SPLITS_SHA256 = "4a58d9fa6667de6d15c5d6563db0e7a0ec306382cd9816900f7098c80bbee8a2"
EXPECTED_TRAIN_SAMPLES = 148185
EXPECTED_VAL_SAMPLES = 42809
EXPECTED_TEST_SAMPLES = 42809
EXPECTED_NORMAL_RUNS = 14
EXPECTED_ATTACK_RUNS = 31
EXPECTED_WINDOWS_PER_RUN = 3293
EXPECTED_PARAMETER_COUNT = 882
EXPECTED_GRAPH_POS_WEIGHT = 0.45161290322580644
EXPECTED_NODE_POS_WEIGHT = 13.4


def b1_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def b1_sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def b1_atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def b1_atomic_json(path: Path, value: dict) -> None:
    b1_atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def b1_write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def b1_count_tokens(value: object) -> int:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "idle"}:
        return 0
    return len([token for token in text.split("-") if token])


class B1AuditedWeightedRandomSampler(Sampler[int]):
    """Audited equivalent of WeightedRandomSampler using torch.multinomial."""

    def __init__(
        self,
        *,
        weights: torch.Tensor,
        num_samples: int,
        replacement: bool,
        generator: torch.Generator,
        train_idx: np.ndarray,
        y_graph: np.ndarray,
        run_id: np.ndarray,
        end_epoch: np.ndarray,
        persist_dir: Path | None,
    ) -> None:
        if weights.ndim != 1:
            raise ValueError("Sampler weights must be one-dimensional.")
        if len(weights) != len(train_idx):
            raise ValueError("Weights and train_idx lengths differ.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if not replacement:
            raise ValueError("B1 requires replacement=True.")

        self.weights = weights.to(dtype=torch.double, device="cpu")
        self.num_samples = int(num_samples)
        self.replacement = bool(replacement)
        self.generator = generator
        self.train_idx = np.asarray(train_idx, dtype=np.int64)
        self.y_graph = np.asarray(y_graph)
        self.run_id = np.asarray(run_id).astype(str)
        self.end_epoch = np.asarray(end_epoch)
        self.persist_dir = persist_dir
        self.epoch_number = 0
        self.last_local_indices = None
        self.last_global_indices = None
        self.last_audit = None

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        self.epoch_number += 1
        state_before = self.generator.get_state().cpu().numpy().tobytes()

        local_tensor = torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement,
            generator=self.generator,
        )

        state_after = self.generator.get_state().cpu().numpy().tobytes()
        local_indices = local_tensor.cpu().numpy().astype(np.int64, copy=False)
        global_indices = self.train_idx[local_indices]

        self.last_local_indices = local_indices
        self.last_global_indices = global_indices
        self.last_audit = {
            "epoch": int(self.epoch_number),
            "num_samples": int(self.num_samples),
            "replacement": bool(self.replacement),
            "sampler_state_before_sha256": b1_sha256_bytes(state_before),
            "sampler_state_after_sha256": b1_sha256_bytes(state_after),
            "local_indices_sha256": b1_sha256_bytes(local_indices.tobytes()),
            "global_indices_sha256": b1_sha256_bytes(global_indices.tobytes()),
        }

        if self.persist_dir is not None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            run_lookup, run_codes = np.unique(
                self.run_id[global_indices],
                return_inverse=True,
            )
            epoch_path = (
                self.persist_dir
                / f"epoch_{self.epoch_number:03d}_sampled_indices.npz"
            )
            np.savez_compressed(
                epoch_path,
                dataset_local_positions=local_indices,
                global_v3_indices=global_indices,
                graph_labels=self.y_graph[global_indices].astype(np.int8),
                run_codes=run_codes.astype(np.int16),
                run_lookup=run_lookup.astype(str),
                end_epochs=self.end_epoch[global_indices].astype(np.int32),
            )
            self.last_audit["npz_path"] = str(epoch_path)
            self.last_audit["npz_sha256"] = b1_sha256_file(epoch_path)

            ledger = self.persist_dir / "sampler_epoch_audit.jsonl"
            with ledger.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(self.last_audit, sort_keys=True) + "\n")

        return iter(local_indices.tolist())


def b1_build_sample_weights(
    *,
    train_idx: np.ndarray,
    y_graph: np.ndarray,
    run_id: np.ndarray,
):
    train_idx = np.asarray(train_idx, dtype=np.int64)
    labels = np.asarray(y_graph[train_idx], dtype=np.int64)
    runs = np.asarray(run_id[train_idx]).astype(str)

    unique_runs = np.unique(runs)
    run_sizes = {run: int(np.sum(runs == run)) for run in unique_runs}
    run_labels = {}

    for run in unique_runs:
        values = np.unique(labels[runs == run])
        if len(values) != 1:
            raise RuntimeError(f"Run {run!r} has nonconstant graph labels.")
        run_labels[run] = int(values[0])

    normal_runs = sorted(run for run, label in run_labels.items() if label == 0)
    attack_runs = sorted(run for run, label in run_labels.items() if label == 1)

    if len(normal_runs) != EXPECTED_NORMAL_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_NORMAL_RUNS} normal runs, found {len(normal_runs)}."
        )
    if len(attack_runs) != EXPECTED_ATTACK_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_ATTACK_RUNS} attack runs, found {len(attack_runs)}."
        )

    unexpected_sizes = {
        run: size
        for run, size in run_sizes.items()
        if size != EXPECTED_WINDOWS_PER_RUN
    }
    if unexpected_sizes:
        raise RuntimeError(
            "B1 requires 3293 windows per training run; mismatches: "
            + repr(unexpected_sizes)
        )

    class_run_counts = {0: len(normal_runs), 1: len(attack_runs)}
    weights = np.empty(len(train_idx), dtype=np.float64)

    for local_position, (label, run) in enumerate(zip(labels, runs)):
        weights[local_position] = (
            0.5 / class_run_counts[int(label)] / run_sizes[str(run)]
        )

    normal_weight = float(weights[labels == 0][0])
    attack_weight = float(weights[labels == 1][0])
    relative_ratio = normal_weight / attack_weight
    expected_ratio = EXPECTED_ATTACK_RUNS / EXPECTED_NORMAL_RUNS

    if not np.isclose(relative_ratio, expected_ratio, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"Unexpected normal/attack sample-weight ratio: {relative_ratio}"
        )

    info = {
        "normal_runs": normal_runs,
        "attack_runs": attack_runs,
        "run_sizes": run_sizes,
        "normal_per_window_probability": normal_weight,
        "attack_per_window_probability": attack_weight,
        "normal_to_attack_weight_ratio": relative_ratio,
        "equivalent_integer_weights": {
            "normal": EXPECTED_ATTACK_RUNS,
            "attack": EXPECTED_NORMAL_RUNS,
        },
    }
    return torch.from_numpy(weights), info


def b1_build_loaders(
    *,
    args,
    train_dataset,
    val_dataset,
    test_dataset,
    train_idx: np.ndarray,
    y_graph: np.ndarray,
    run_id: np.ndarray,
    end_epoch: np.ndarray,
):
    if args.sampler_mode != "class_run_balanced":
        raise RuntimeError("B1 supports only class_run_balanced sampling.")
    if len(train_dataset) != len(train_idx):
        raise RuntimeError("train_dataset and train_idx lengths differ.")
    if len(train_idx) != EXPECTED_TRAIN_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_SAMPLES} training samples, "
            f"found {len(train_idx)}."
        )
    if args.samples_per_epoch != EXPECTED_TRAIN_SAMPLES:
        raise RuntimeError(
            "For isolated B1, --samples-per-epoch must be 148185."
        )

    weights, sampler_info = b1_build_sample_weights(
        train_idx=train_idx,
        y_graph=y_graph,
        run_id=run_id,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.sampler_seed))

    persist_dir = None
    if not args.dry_run_sampler:
        persist_dir = Path(args.sampler_log_dir).expanduser().resolve()

    sampler = B1AuditedWeightedRandomSampler(
        weights=weights,
        num_samples=int(args.samples_per_epoch),
        replacement=True,
        generator=generator,
        train_idx=train_idx,
        y_graph=y_graph,
        run_id=run_id,
        end_epoch=end_epoch,
        persist_dir=persist_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    sampler_info.update(
        {
            "sampler_mode": args.sampler_mode,
            "sampler_seed": int(args.sampler_seed),
            "samples_per_epoch": int(args.samples_per_epoch),
            "replacement": True,
            "batch_size": int(args.batch_size),
            "batch_count": int(math.ceil(args.samples_per_epoch / args.batch_size)),
            "drop_last": False,
            "train_metrics_distribution": "full_original_training_cohort",
            "optimization_distribution": "class_balanced_run_uniform_replacement",
        }
    )
    return (
        train_loader,
        train_eval_loader,
        val_loader,
        test_loader,
        sampler,
        sampler_info,
    )


def b1_group_rows(values: np.ndarray, column_name: str):
    values = np.asarray(values).astype(str)
    unique, counts = np.unique(values, return_counts=True)
    total = int(counts.sum())
    return [
        {
            column_name: str(value),
            "draw_count": int(count),
            "draw_fraction": float(count / total),
        }
        for value, count in zip(unique, counts)
    ]


def b1_validate_model_structure(args, repo_root: Path):
    model = TemporalGCN(
        input_features=int(args._b1_input_feature_dim),
        temporal_dim=args.temporal_dim,
        gcn_hidden=args.gcn_hidden,
        gcn_out=args.gcn_out,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    checkpoint_path = (
        repo_root
        / "models/v3/stage9_a1_conv1d_exact_reproduction_seed7/best_model.pt"
    )
    if not checkpoint_path.is_file():
        raise RuntimeError(f"A1 checkpoint not found: {checkpoint_path}")
    if b1_sha256_file(checkpoint_path) != A1_CHECKPOINT_SHA256:
        raise RuntimeError("A1 checkpoint SHA-256 mismatch.")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    reference_state = checkpoint["model_state_dict"]
    current_state = model.state_dict()

    same_keys = list(reference_state.keys()) == list(current_state.keys())
    shape_mismatches = {
        key: {
            "a1": list(reference_state[key].shape),
            "b1": list(current_state[key].shape),
        }
        for key in reference_state.keys() & current_state.keys()
        if tuple(reference_state[key].shape) != tuple(current_state[key].shape)
    }

    return {
        "parameter_count": int(parameter_count),
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "parameter_count_match": parameter_count == EXPECTED_PARAMETER_COUNT,
        "state_dict_keys_match": bool(same_keys),
        "state_dict_shape_mismatches": shape_mismatches,
        "a1_checkpoint_sha256": A1_CHECKPOINT_SHA256,
    }


def b1_run_dry_run(
    *,
    args,
    data: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    graph_pos_weight: float,
    node_pos_weight: float,
    train_loader,
    train_eval_loader,
    val_loader,
    test_loader,
    sampler: B1AuditedWeightedRandomSampler,
    sampler_info: dict,
) -> None:
    report_dir = Path(args.sampler_log_dir).expanduser().resolve()
    if report_dir.exists():
        raise RuntimeError(
            "Dry-run directory already exists; archive it instead of overwriting: "
            f"{report_dir}"
        )

    model_output = Path(args.out_dir).expanduser().resolve()
    if model_output.exists():
        raise RuntimeError(
            "B1 model output path must be unused before dry run: "
            f"{model_output}"
        )

    repo_root = Path(__file__).resolve().parents[1]
    a1_source = repo_root / "scripts/train_temporal_gcn_v3.py"
    a1_splits = (
        repo_root
        / "models/v3/stage9_a1_conv1d_exact_reproduction_seed7/splits.npz"
    )

    source_hash = b1_sha256_file(a1_source)
    if source_hash != A1_SOURCE_SHA256:
        raise RuntimeError(
            f"A1 source hash mismatch: expected {A1_SOURCE_SHA256}, got {source_hash}"
        )
    if b1_sha256_file(a1_splits) != A1_SPLITS_SHA256:
        raise RuntimeError("A1 split artifact SHA-256 mismatch.")

    with np.load(a1_splits, allow_pickle=False) as split_data:
        split_identity = {
            "train_idx": np.array_equal(train_idx, split_data["train_idx"]),
            "val_idx": np.array_equal(val_idx, split_data["val_idx"]),
            "test_idx": np.array_equal(test_idx, split_data["test_idx"]),
        }

    local_positions = np.asarray(list(iter(sampler)), dtype=np.int64)
    global_indices = train_idx[local_positions]
    labels = np.asarray(data["y_graph"][global_indices], dtype=np.int64)
    runs = np.asarray(data["run_id"][global_indices]).astype(str)
    end_epochs = np.asarray(data["end_epoch"][global_indices], dtype=np.int64)

    train_set = set(np.asarray(train_idx, dtype=np.int64).tolist())
    val_set = set(np.asarray(val_idx, dtype=np.int64).tolist())
    test_set = set(np.asarray(test_idx, dtype=np.int64).tolist())
    sampled_set = set(global_indices.tolist())

    class_values, class_counts = np.unique(labels, return_counts=True)
    class_count_map = {
        int(label): int(count)
        for label, count in zip(class_values, class_counts)
    }
    normal_fraction = class_count_map.get(0, 0) / len(global_indices)
    attack_fraction = class_count_map.get(1, 0) / len(global_indices)

    run_values, run_counts = np.unique(runs, return_counts=True)
    run_rows = []
    train_runs = np.asarray(data["run_id"][train_idx]).astype(str)
    train_labels = np.asarray(data["y_graph"][train_idx], dtype=np.int64)

    for run, count in zip(run_values, run_counts):
        run_label_values = np.unique(train_labels[train_runs == run])
        if len(run_label_values) != 1:
            raise RuntimeError(f"Run {run!r} has inconsistent labels.")
        run_label = int(run_label_values[0])
        class_runs = EXPECTED_NORMAL_RUNS if run_label == 0 else EXPECTED_ATTACK_RUNS
        expected = len(global_indices) * 0.5 / class_runs
        probability = 0.5 / class_runs
        variance = len(global_indices) * probability * (1.0 - probability)
        z_score = (float(count) - expected) / max(math.sqrt(variance), 1.0)
        run_rows.append(
            {
                "run_id": str(run),
                "graph_class": "normal" if run_label == 0 else "attack",
                "draw_count": int(count),
                "expected_draw_count": float(expected),
                "deviation": float(count - expected),
                "standardized_deviation": float(z_score),
            }
        )

    uniformity = {}
    for class_name in ("normal", "attack"):
        counts = np.asarray(
            [row["draw_count"] for row in run_rows if row["graph_class"] == class_name],
            dtype=np.float64,
        )
        expected = float(np.mean(counts))
        uniformity[class_name] = {
            "coefficient_of_variation": float(np.std(counts, ddof=1) / expected),
            "chi_square_statistic": float(np.sum((counts - expected) ** 2 / expected)),
            "maximum_absolute_standardized_deviation": float(
                max(
                    abs(row["standardized_deviation"])
                    for row in run_rows
                    if row["graph_class"] == class_name
                )
            ),
        }

    unique_positions, multiplicities = np.unique(
        local_positions,
        return_counts=True,
    )
    duplicate_statistics = {
        "total_draws": int(len(local_positions)),
        "unique_training_positions_drawn": int(len(unique_positions)),
        "duplicate_draws_beyond_first": int(len(local_positions) - len(unique_positions)),
        "unique_fraction": float(len(unique_positions) / len(local_positions)),
        "maximum_multiplicity": int(multiplicities.max()),
        "mean_multiplicity_among_drawn_positions": float(multiplicities.mean()),
    }

    report_dir.mkdir(parents=True, exist_ok=False)
    run_lookup, run_codes = np.unique(runs, return_inverse=True)
    sampled_npz = report_dir / "b1_sampled_epoch_indices.npz"
    np.savez_compressed(
        sampled_npz,
        dataset_local_positions=local_positions,
        global_v3_indices=global_indices,
        graph_labels=labels.astype(np.int8),
        run_codes=run_codes.astype(np.int16),
        run_lookup=run_lookup.astype(str),
        end_epochs=end_epochs.astype(np.int32),
    )
    sampled_npz_hash = b1_sha256_file(sampled_npz)
    b1_atomic_text(
        report_dir / "b1_sampled_epoch_indices_sha256.txt",
        f"{sampled_npz_hash}  {sampled_npz.name}\n",
    )

    b1_write_csv(
        report_dir / "b1_per_run_draws.csv",
        run_rows,
        [
            "run_id",
            "graph_class",
            "draw_count",
            "expected_draw_count",
            "deviation",
            "standardized_deviation",
        ],
    )
    b1_write_csv(
        report_dir / "b1_class_draws.csv",
        [
            {
                "graph_class": "normal",
                "draw_count": class_count_map.get(0, 0),
                "draw_fraction": normal_fraction,
            },
            {
                "graph_class": "attack",
                "draw_count": class_count_map.get(1, 0),
                "draw_fraction": attack_fraction,
            },
        ],
        ["graph_class", "draw_count", "draw_fraction"],
    )

    profile = np.asarray(data["profile"][global_indices]).astype(str)
    strength = np.asarray(data["strength"][global_indices]).astype(str)
    attackers = np.asarray(data["attackers"][global_indices]).astype(str)
    active_cores = np.asarray(data["active_cores"][global_indices]).astype(str)
    attacker_count = np.asarray([b1_count_tokens(value) for value in attackers])
    active_core_count = np.asarray([b1_count_tokens(value) for value in active_cores])

    for filename, values, column in [
        ("b1_profile_draws.csv", profile, "profile"),
        ("b1_strength_draws.csv", strength, "strength"),
        ("b1_attacker_count_draws.csv", attacker_count, "attacker_count"),
        ("b1_active_core_group_draws.csv", active_core_count, "active_core_count"),
    ]:
        rows = b1_group_rows(values, column)
        b1_write_csv(
            report_dir / filename,
            rows,
            [column, "draw_count", "draw_fraction"],
        )

    temporal_edges = np.linspace(
        int(end_epochs.min()),
        int(end_epochs.max()) + 1,
        11,
    )
    temporal_bin = np.clip(
        np.digitize(end_epochs, temporal_edges[1:-1], right=False),
        0,
        9,
    )
    temporal_rows = []
    for bin_index in range(10):
        mask = temporal_bin == bin_index
        temporal_rows.append(
            {
                "temporal_decile": int(bin_index),
                "minimum_end_epoch": int(end_epochs[mask].min()) if np.any(mask) else "",
                "maximum_end_epoch": int(end_epochs[mask].max()) if np.any(mask) else "",
                "draw_count": int(mask.sum()),
                "draw_fraction": float(mask.mean()),
            }
        )
    b1_write_csv(
        report_dir / "b1_temporal_draws.csv",
        temporal_rows,
        [
            "temporal_decile",
            "minimum_end_epoch",
            "maximum_end_epoch",
            "draw_count",
            "draw_fraction",
        ],
    )
    b1_atomic_json(
        report_dir / "b1_duplicate_statistics.json",
        duplicate_statistics,
    )

    model_validation = b1_validate_model_structure(args, repo_root)

    loader_validation = {
        "optimization_loader_uses_replacement_sampler": isinstance(
            train_loader.sampler,
            B1AuditedWeightedRandomSampler,
        ),
        "optimization_loader_drop_last": bool(train_loader.drop_last),
        "optimization_loader_batches": int(len(train_loader)),
        "expected_batches": int(math.ceil(EXPECTED_TRAIN_SAMPLES / args.batch_size)),
        "train_eval_loader_samples": int(len(train_eval_loader.dataset)),
        "train_eval_loader_drop_last": bool(train_eval_loader.drop_last),
        "validation_loader_samples": int(len(val_loader.dataset)),
        "validation_loader_batch_size": int(val_loader.batch_size),
        "validation_loader_drop_last": bool(val_loader.drop_last),
        "validation_loader_num_workers": int(val_loader.num_workers),
        "test_loader_samples": int(len(test_loader.dataset)),
        "test_loader_batch_size": int(test_loader.batch_size),
        "test_loader_drop_last": bool(test_loader.drop_last),
        "test_loader_num_workers": int(test_loader.num_workers),
    }

    class_weight_validation = {
        "graph_pos_weight": float(graph_pos_weight),
        "expected_graph_pos_weight": EXPECTED_GRAPH_POS_WEIGHT,
        "graph_pos_weight_match": bool(
            np.isclose(
                graph_pos_weight,
                EXPECTED_GRAPH_POS_WEIGHT,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "node_pos_weight": float(node_pos_weight),
        "expected_node_pos_weight": EXPECTED_NODE_POS_WEIGHT,
        "node_pos_weight_match": bool(
            np.isclose(
                node_pos_weight,
                EXPECTED_NODE_POS_WEIGHT,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "weights_calculated_from_full_train_idx": True,
        "weights_recomputed_from_sampled_epoch": False,
    }

    checks = {
        "a1_source_hash_match": source_hash == A1_SOURCE_SHA256,
        "split_train_matches_a1": split_identity["train_idx"],
        "split_val_matches_a1": split_identity["val_idx"],
        "split_test_matches_a1": split_identity["test_idx"],
        "train_samples_exact": len(train_idx) == EXPECTED_TRAIN_SAMPLES,
        "val_samples_exact": len(val_idx) == EXPECTED_VAL_SAMPLES,
        "test_samples_exact": len(test_idx) == EXPECTED_TEST_SAMPLES,
        "total_draws_exact": len(local_positions) == EXPECTED_TRAIN_SAMPLES,
        "batch_count_exact": len(train_loader) == 579,
        "replacement_true": sampler.replacement is True,
        "all_local_positions_valid": bool(
            np.all((local_positions >= 0) & (local_positions < len(train_idx)))
        ),
        "all_draws_are_training_samples": sampled_set.issubset(train_set),
        "no_validation_sample_drawn": sampled_set.isdisjoint(val_set),
        "no_test_sample_drawn": sampled_set.isdisjoint(test_set),
        "all_14_normal_runs_represented": sum(
            row["graph_class"] == "normal" for row in run_rows
        ) == EXPECTED_NORMAL_RUNS,
        "all_31_attack_runs_represented": sum(
            row["graph_class"] == "attack" for row in run_rows
        ) == EXPECTED_ATTACK_RUNS,
        "normal_fraction_in_tolerance": 0.49 <= normal_fraction <= 0.51,
        "attack_fraction_in_tolerance": 0.49 <= attack_fraction <= 0.51,
        "parameter_count_match": model_validation["parameter_count_match"],
        "model_state_keys_match": model_validation["state_dict_keys_match"],
        "model_state_shapes_match": not model_validation[
            "state_dict_shape_mismatches"
        ],
        "graph_class_weight_match": class_weight_validation[
            "graph_pos_weight_match"
        ],
        "node_class_weight_match": class_weight_validation[
            "node_pos_weight_match"
        ],
        "validation_loader_unchanged": (
            len(val_loader.dataset) == EXPECTED_VAL_SAMPLES
            and val_loader.batch_size == args.batch_size
            and val_loader.drop_last is False
            and val_loader.num_workers == 0
        ),
        "test_loader_unchanged": (
            len(test_loader.dataset) == EXPECTED_TEST_SAMPLES
            and test_loader.batch_size == args.batch_size
            and test_loader.drop_last is False
            and test_loader.num_workers == 0
        ),
        "b1_model_output_path_unused": not model_output.exists(),
    }
    overall_success = all(checks.values())

    b1_atomic_json(
        report_dir / "b1_dry_run_summary.json",
        {
            "stage": "Stage 9 B1 sampler dry run",
            "verdict": "PASS" if overall_success else "FAIL",
            "sampler": sampler_info,
            "sampler_last_audit": sampler.last_audit,
            "class_counts": {
                "normal": class_count_map.get(0, 0),
                "attack": class_count_map.get(1, 0),
                "normal_fraction": normal_fraction,
                "attack_fraction": attack_fraction,
            },
            "run_uniformity_diagnostics": uniformity,
            "duplicates": duplicate_statistics,
            "model_structure": model_validation,
            "class_weights": class_weight_validation,
            "loaders": loader_validation,
            "split_identity": split_identity,
            "checks": checks,
            "overall_success": overall_success,
            "training_started": False,
            "gpu_transfer_performed": False,
            "forward_pass_performed": False,
            "backpropagation_performed": False,
            "optimizer_update_performed": False,
            "checkpoint_created": False,
        },
    )

    b1_atomic_text(
        report_dir / "b1_split_validation.txt",
        "\n".join(
            [
                f"A1 source SHA-256 match: {checks['a1_source_hash_match']}",
                f"train_idx exact A1 match: {split_identity['train_idx']}",
                f"val_idx exact A1 match: {split_identity['val_idx']}",
                f"test_idx exact A1 match: {split_identity['test_idx']}",
                f"all draws are training samples: {checks['all_draws_are_training_samples']}",
                f"no validation sample drawn: {checks['no_validation_sample_drawn']}",
                f"no test sample drawn: {checks['no_test_sample_drawn']}",
            ]
        )
        + "\n",
    )
    b1_atomic_text(
        report_dir / "b1_class_weight_validation.txt",
        "\n".join(
            f"{key}: {value}"
            for key, value in class_weight_validation.items()
        )
        + "\n",
    )
    b1_atomic_text(
        report_dir / "b1_model_structure_validation.txt",
        "\n".join(
            f"{key}: {value}"
            for key, value in model_validation.items()
        )
        + "\n",
    )
    b1_atomic_text(
        report_dir / "b1_dataloader_validation.txt",
        "\n".join(
            f"{key}: {value}"
            for key, value in loader_validation.items()
        )
        + "\n",
    )
    b1_atomic_text(
        report_dir / "b1_dry_run_validation.txt",
        "\n".join(f"{key}: {value}" for key, value in checks.items())
        + f"\noverall success: {overall_success}\n"
        + "training started: False\n"
        + "checkpoint created: False\n",
    )

    report = f"""# Stage 9 B1 Sampler Dry Run

## Verdict

**{"PASS" if overall_success else "FAIL"}**

## Intervention

The sole optimization change is graph-class-balanced,
class-conditional run-uniform sampling with replacement.

- Model seed: `{args.seed}`
- Sampler seed: `{args.sampler_seed}`
- Samples per epoch: `{args.samples_per_epoch}`
- Batch size: `{args.batch_size}`
- Batches per epoch: `{len(train_loader)}`
- Replacement: `True`

## Distribution

- Normal draws: `{class_count_map.get(0, 0)}`
- Attack draws: `{class_count_map.get(1, 0)}`
- Normal fraction: `{normal_fraction:.6f}`
- Attack fraction: `{attack_fraction:.6f}`
- Expected draws per normal run: `{len(global_indices) * 0.5 / EXPECTED_NORMAL_RUNS:.3f}`
- Expected draws per attack run: `{len(global_indices) * 0.5 / EXPECTED_ATTACK_RUNS:.3f}`

## Frozen historical objective interaction

B1 retains graph positive weight `{graph_pos_weight:.12f}` while sampling
approximately 50/50 graph classes. Ignoring prediction difficulty:

- normal contribution: `0.5 × 1.0 = 0.5`
- attack contribution: `0.5 × {graph_pos_weight:.12f} = {0.5 * graph_pos_weight:.12f}`

B1 is balanced sampling under the frozen historical loss, not a fully
balanced objective.

## Train metrics

`train_loader` is optimization-only. `train_eval_loader` traverses the
complete original training cohort without replacement for diagnostic train
metrics. Validation and test remain the comparison/model-selection sources.

## Safety

- GPU transfer performed: `False`
- Forward pass performed: `False`
- Backpropagation performed: `False`
- Optimizer update performed: `False`
- Checkpoint created: `False`
"""
    b1_atomic_text(report_dir / "B1_DRY_RUN_REPORT.md", report)

    print("===== STAGE 9 B1 SAMPLER DRY RUN =====")
    print("verdict:", "PASS" if overall_success else "FAIL")
    print("normal draws:", class_count_map.get(0, 0))
    print("attack draws:", class_count_map.get(1, 0))
    print("batches:", len(train_loader))
    print("parameter count:", model_validation["parameter_count"])
    print("output:", report_dir)
    print("training started: False")
    print("checkpoint created: False")

    if not overall_success:
        raise SystemExit("B1 sampler dry run failed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="graph_dataset/paper1_temporal_graphs_full.npz")
    parser.add_argument("--out-dir", default="models/temporal_gcn_conv1d_early_stop")
    parser.add_argument("--split-mode", choices=["prototype", "placement", "v3"], default="prototype")
    parser.add_argument("--splits-file", default=None, help="Optional .npz file containing train_idx, val_idx, test_idx.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temporal-dim", type=int, default=8)
    parser.add_argument("--gcn-hidden", type=int, default=16)
    parser.add_argument("--gcn-out", type=int, default=8)
    parser.add_argument("--node-loss-weight", type=float, default=1.0)
    parser.add_argument("--graph-threshold", type=float, default=0.5)
    parser.add_argument("--node-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    
    parser.add_argument(
        "--sampler-mode",
        required=True,
        choices=["class_run_balanced"],
        help="Stage 9 B1 intervention; only class_run_balanced is allowed.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        required=True,
        help="Dedicated sampler RNG seed, separate from the model seed.",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=148185,
        help="Replacement draws per optimization epoch; frozen to 148185.",
    )
    parser.add_argument(
        "--sampler-log-dir",
        required=True,
        help="Unused output directory for dry-run or per-epoch sampler audit.",
    )
    parser.add_argument(
        "--dry-run-sampler",
        action="store_true",
        help="Validate one full sampled epoch on CPU and exit before training.",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")
    if args.min_delta < 0:
        raise ValueError("--min-delta cannot be negative.")

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    if args.dry_run_sampler:
        if out_dir.exists():
            raise FileExistsError(
                f"B1 model output path must be unused: {out_dir}"
            )
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data = load_graph_arrays(args.data)
    x_shape = data["x"].shape
    if len(x_shape) != 4:
        raise ValueError(f"Expected x shape [samples, nodes, time, features], received {x_shape}.")

    num_nodes = int(x_shape[1])
    input_feature_dim = int(x_shape[-1])
    edge_index = data["edge_index"].astype(np.int64)
    a_hat = build_normalized_adjacency(edge_index, num_nodes).to(device)

    if args.split_mode == "v3":
        train_idx, val_idx, test_idx = make_v3_splits(args.data)
    elif args.splits_file is not None:
        split_data = np.load(args.splits_file)
        train_idx = split_data["train_idx"].astype(np.int64)
        val_idx = split_data["val_idx"].astype(np.int64)
        test_idx = split_data["test_idx"].astype(np.int64)
        print("\nloaded external splits:", args.splits_file)
    else:
        train_idx, val_idx, test_idx = make_splits(args.data, args.split_mode)

    print("\nsplit sizes:")
    print("  train:", len(train_idx))
    print("  val:  ", len(val_idx))
    print("  test: ", len(test_idx))

    y_graph_train = data["y_graph"][train_idx].astype(np.float32)
    y_node_train = data["y_node"][train_idx].astype(np.float32)
    graph_pos = float(y_graph_train.sum())
    graph_neg = float(len(y_graph_train) - graph_pos)
    graph_pos_weight = graph_neg / max(graph_pos, 1.0)
    node_pos = float(y_node_train.sum())
    node_neg = float(y_node_train.size - node_pos)
    node_pos_weight = node_neg / max(node_pos, 1.0)

    print("\nclass weights:")
    print("  graph pos weight:", graph_pos_weight)
    print("  node pos weight: ", node_pos_weight)

    train_dataset = NoCTemporalGraphDataset(args.data, train_idx)
    val_dataset = NoCTemporalGraphDataset(args.data, val_idx)
    test_dataset = NoCTemporalGraphDataset(args.data, test_idx)

    (
        train_loader,
        train_eval_loader,
        val_loader,
        test_loader,
        b1_sampler,
        b1_sampler_info,
    ) = b1_build_loaders(
        args=args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_idx=train_idx,
        y_graph=data["y_graph"],
        run_id=data["run_id"],
        end_epoch=data["end_epoch"],
    )

    args._b1_input_feature_dim = int(data["x"].shape[-1])

    if args.dry_run_sampler:
        b1_run_dry_run(
            args=args,
            data=data,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            graph_pos_weight=float(graph_pos_weight),
            node_pos_weight=float(node_pos_weight),
            train_loader=train_loader,
            train_eval_loader=train_eval_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            sampler=b1_sampler,
            sampler_info=b1_sampler_info,
        )
        return

    print("input feature dim:", input_feature_dim)
    print("number of nodes:", num_nodes)

    model = TemporalGCN(input_features=input_feature_dim, temporal_dim=args.temporal_dim, gcn_hidden=args.gcn_hidden, gcn_out=args.gcn_out).to(device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print("\nmodel:")
    print(model)
    print("parameters:", total_params)

    graph_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(graph_pos_weight, dtype=torch.float32, device=device)
    )
    node_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(node_pos_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_score = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_early = False
    stopped_epoch = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False)

        for batch in progress:
            x = batch["x"].to(device)
            y_graph = batch["y_graph"].to(device)
            y_node = batch["y_node"].to(device)
            graph_logits, node_logits = model(x, a_hat)
            graph_loss = graph_loss_fn(graph_logits, y_graph)
            node_loss = node_loss_fn(node_logits, y_node)
            loss = graph_loss + args.node_loss_weight * node_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_step_loss = total_loss / max(1, batches)
        eval_kwargs = {
            "a_hat": a_hat,
            "device": device,
            "graph_loss_fn": graph_loss_fn,
            "node_loss_fn": node_loss_fn,
            "node_loss_weight": args.node_loss_weight,
            "graph_threshold": args.graph_threshold,
            "node_threshold": args.node_threshold,
        }
        train_metrics = evaluate(model, train_eval_loader, **eval_kwargs)
        val_metrics = evaluate(model, val_loader, **eval_kwargs)

        print(f"\nepoch {epoch:03d} train_step_loss={train_step_loss:.4f}")
        print_metrics("  train", train_metrics)
        print_metrics("  val  ", val_metrics)

        val_score = val_metrics["f1"] + val_metrics["node_f1"]
        previous_best = None if best_val_score == -float("inf") else float(best_val_score)
        improvement = val_score - best_val_score

        if improvement > args.min_delta:
            best_val_score = float(val_score)
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "A_hat": a_hat.detach().cpu(),
                "best_epoch": best_epoch,
                "best_val_score": best_val_score,
                "val_metrics": val_metrics,
                "model_name": "Conv1D-TemporalGCN",
            }
            torch.save(checkpoint, out_dir / "best_model.pt")
            print(f"  validation score improved to {best_val_score:.6f}; saved best_model.pt")
        else:
            epochs_without_improvement += 1
            print(f"  no meaningful validation improvement ({epochs_without_improvement}/{args.patience})")

        history.append({
            "epoch": int(epoch),
            "train_loss_step": float(train_step_loss),
            "train": train_metrics,
            "val": val_metrics,
            "val_score": float(val_score),
            "best_val_score_before_epoch": previous_best,
            "best_val_score_after_epoch": float(best_val_score),
            "epochs_without_improvement": int(epochs_without_improvement),
        })

        with (out_dir / "history.json").open("w") as file:
            json.dump(history, file, indent=2)

        if epochs_without_improvement >= args.patience:
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"\nearly stopping triggered at epoch {stopped_epoch}. "
                f"Best epoch was {best_epoch} with validation score {best_val_score:.6f}."
            )
            break

    print("\nloading best checkpoint...")
    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    eval_kwargs = {
        "a_hat": a_hat,
        "device": device,
        "graph_loss_fn": graph_loss_fn,
        "node_loss_fn": node_loss_fn,
        "node_loss_weight": args.node_loss_weight,
        "graph_threshold": args.graph_threshold,
        "node_threshold": args.node_threshold,
    }
    train_metrics = evaluate(model, train_eval_loader, **eval_kwargs)
    val_metrics = evaluate(model, val_loader, **eval_kwargs)
    test_metrics = evaluate(model, test_loader, **eval_kwargs)

    print("\nFINAL")
    print("model:", "Conv1D-TemporalGCN")
    print("best epoch:", checkpoint["best_epoch"])
    print("best validation score:", checkpoint["best_val_score"])
    print_metrics("train", train_metrics)
    print_metrics("val  ", val_metrics)
    print_metrics("test ", test_metrics)

    summary = {
        "model": "Conv1D-TemporalGCN",
        "data": args.data,
        "split_mode": args.split_mode,
        "splits_file": args.splits_file,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_score": float(checkpoint["best_val_score"]),
        "parameters": int(total_params),
        "input_feature_dim": int(input_feature_dim),
        "num_nodes": int(num_nodes),
        "training": {
            "maximum_epochs": int(args.epochs),
            "epochs_completed": int(len(history)),
            "patience": int(args.patience),
            "min_delta": float(args.min_delta),
            "stopped_early": bool(stopped_early),
            "stopped_epoch": stopped_epoch,
            "seed": int(args.seed),
        },
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "class_weights": {
            "graph_pos_weight": float(graph_pos_weight),
            "node_pos_weight": float(node_pos_weight),
        },
        "thresholds": {
            "graph": float(args.graph_threshold),
            "node": float(args.node_threshold),
        },
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }

    summary["stage9_b1"] = {
        "intervention": "graph_class_balanced_class_conditional_run_uniform_sampling",
        "sampler_mode": args.sampler_mode,
        "sampler_seed": int(args.sampler_seed),
        "samples_per_epoch": int(args.samples_per_epoch),
        "replacement": True,
        "sampler_log_dir": str(Path(args.sampler_log_dir).expanduser().resolve()),
        "train_metrics_distribution": "complete_original_training_cohort",
        "optimization_distribution": "class_balanced_run_uniform_replacement",
        "class_weights_source": "complete_unmodified_train_idx",
        "graph_readout_changed": False,
        "model_architecture_changed": False,
        "validation_loader_changed": False,
        "test_loader_changed": False,
    }
    with (out_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)

    np.savez_compressed(
        out_dir / "splits.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    print("\nwrote:")
    print(" ", out_dir / "best_model.pt")
    print(" ", out_dir / "history.json")
    print(" ", out_dir / "summary.json")
    print(" ", out_dir / "splits.npz")


if __name__ == "__main__":
    main()
