#!/usr/bin/env python3

import argparse
import json
import math
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset
# -----------------------------

class NoCTemporalGraphDataset(Dataset):
    def __init__(self, npz_path, indices):
        d = np.load(npz_path, allow_pickle=True)

        self.x = d["x"].astype(np.float32)
        self.y_graph = d["y_graph"].astype(np.float32)
        self.y_node = d["y_node"].astype(np.float32)
        self.run_id = d["run_id"].astype(str)
        self.end_epoch = d["end_epoch"].astype(np.int32)
        self.attackers = d["attackers"].astype(str)

        self.indices = np.array(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        return {
            "x": torch.from_numpy(self.x[real_idx]),              # [16, 8, 2]
            "y_graph": torch.tensor(self.y_graph[real_idx]),      # scalar
            "y_node": torch.from_numpy(self.y_node[real_idx]),    # [16]
            "run_id": self.run_id[real_idx],
            "end_epoch": int(self.end_epoch[real_idx]),
            "attackers": self.attackers[real_idx],
        }


# -----------------------------
# Splits
# -----------------------------

def make_splits(npz_path, split_mode):
    d = np.load(npz_path, allow_pickle=True)
    run_ids = d["run_id"].astype(str)
    y_graph = d["y_graph"].astype(np.int64)

    unique_runs = sorted(set(run_ids.tolist()))

    print("Runs in dataset:")
    for r in unique_runs:
        count = int((run_ids == r).sum())
        label = int(y_graph[np.where(run_ids == r)[0][0]])
        print(f"  {r:38s} samples={count} label={label}")

    if split_mode == "prototype":
        # We have only one normal run, so for the first prototype we split
        # temporally inside every run:
        # first 70% train, next 15% validation, last 15% test.
        #
        # This is NOT the final paper-quality split.
        # It is the first sanity-check split to verify the model can learn.
        train_idx = []
        val_idx = []
        test_idx = []

        for r in unique_runs:
            idx = np.where(run_ids == r)[0]
            idx = np.sort(idx)

            n = len(idx)
            n_train = int(0.70 * n)
            n_val = int(0.15 * n)

            train_idx.extend(idx[:n_train])
            val_idx.extend(idx[n_train:n_train + n_val])
            test_idx.extend(idx[n_train + n_val:])

        return np.array(train_idx), np.array(val_idx), np.array(test_idx)

    elif split_mode == "placement":
        # A more meaningful attack-placement split.
        #
        # Problem: current dataset has only one normal run, so normal samples
        # still need temporal splitting.
        #
        # Attack runs are split by run_id.
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

        train_idx = []
        val_idx = []
        test_idx = []

        for r in unique_runs:
            idx = np.where(run_ids == r)[0]
            idx = np.sort(idx)

            if "-A-" not in r:
                n = len(idx)
                n_train = int(0.70 * n)
                n_val = int(0.15 * n)

                train_idx.extend(idx[:n_train])
                val_idx.extend(idx[n_train:n_train + n_val])
                test_idx.extend(idx[n_train + n_val:])
            elif r in train_attack_runs:
                train_idx.extend(idx)
            elif r in val_attack_runs:
                val_idx.extend(idx)
            elif r in test_attack_runs:
                test_idx.extend(idx)
            else:
                raise RuntimeError(f"run {r} not assigned to any split")

        return np.array(train_idx), np.array(val_idx), np.array(test_idx)

    else:
        raise ValueError(f"unknown split mode: {split_mode}")


# -----------------------------
# Graph utilities
# -----------------------------

def build_normalized_adjacency(edge_index, num_nodes=16):
    """
    Build A_hat = D^{-1/2} A D^{-1/2}
    edge_index shape: [2, E]
    """
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    src = edge_index[0]
    dst = edge_index[1]

    for s, d in zip(src, dst):
        A[int(d), int(s)] = 1.0
        # We use message from src to dst.
        # Since edge list includes both directions and self-loops,
        # this becomes the usual undirected mesh adjacency.

    degree = A.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0

    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    return A_hat


# -----------------------------
# Model
# -----------------------------

class TemporalEncoder(nn.Module):
    """
    Shared Conv1D encoder applied to every router independently.

    Input per sample:  [B, 16, 8, 2]
    Reshape to:        [B*16, 2, 8]
    Conv1D output:     [B*16, D, 8]
    Max pool over T:   [B*16, D]
    Reshape to:        [B, 16, D]
    """
    def __init__(self, in_features=2, embedding_dim=8, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=embedding_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x):
        B, N, T, Fdim = x.shape

        x = x.reshape(B * N, T, Fdim)      # [B*N, T, F]
        x = x.permute(0, 2, 1)             # [B*N, F, T]

        h = self.conv(x)                   # [B*N, D, T]
        h = F.relu(h)
        h = torch.max(h, dim=2).values     # [B*N, D]

        h = h.reshape(B, N, -1)            # [B, N, D]
        return h


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h, A_hat):
        # h: [B, N, Fin]
        # A_hat: [N, N]
        agg = torch.einsum("ij,bjf->bif", A_hat, h)
        out = self.linear(agg)
        return out


class TemporalGCN(nn.Module):
    def __init__(
        self,
        num_nodes=16,
        input_features=2,
        temporal_dim=8,
        gcn_hidden=16,
        gcn_out=8,
    ):
        super().__init__()

        self.temporal = TemporalEncoder(
            in_features=input_features,
            embedding_dim=temporal_dim,
            kernel_size=3,
        )

        self.gcn1 = GCNLayer(temporal_dim, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_out)

        self.node_head = nn.Linear(gcn_out, 1)
        self.graph_head = nn.Linear(gcn_out, 1)

    def forward(self, x, A_hat):
        # x: [B, 16, 8, 2]
        h = self.temporal(x)               # [B, 16, 8]

        h = self.gcn1(h, A_hat)
        h = F.relu(h)

        h = self.gcn2(h, A_hat)
        h = F.relu(h)

        node_logits = self.node_head(h).squeeze(-1)  # [B, 16]

        graph_embedding = h.mean(dim=1)              # [B, 8]
        graph_logits = self.graph_head(graph_embedding).squeeze(-1)  # [B]

        return graph_logits, node_logits


# -----------------------------
# Metrics
# -----------------------------

def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)
    y_true = y_true.astype(np.int64)

    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tnr": tnr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def node_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)
    y_true = y_true.astype(np.int64)

    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)

    acc = accuracy_score(flat_true, flat_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_true,
        flat_pred,
        average="binary",
        zero_division=0,
    )

    # Exact localization:
    # for every sample, did the predicted malicious bitmap match exactly?
    exact = np.all(y_pred == y_true, axis=1).mean()

    return {
        "node_acc": acc,
        "node_precision": precision,
        "node_recall": recall,
        "node_f1": f1,
        "exact_localization": exact,
    }


@torch.no_grad()
def evaluate(model, loader, A_hat, device, graph_threshold=0.5, node_threshold=0.5):
    model.eval()

    graph_probs = []
    graph_true = []

    node_probs = []
    node_true = []

    total_loss = 0.0
    batches = 0

    bce_graph = nn.BCEWithLogitsLoss()
    bce_node = nn.BCEWithLogitsLoss()

    for batch in loader:
        x = batch["x"].to(device)
        yg = batch["y_graph"].to(device)
        yn = batch["y_node"].to(device)

        graph_logits, node_logits = model(x, A_hat)

        loss_graph = bce_graph(graph_logits, yg)
        loss_node = bce_node(node_logits, yn)
        loss = loss_graph + loss_node

        total_loss += float(loss.item())
        batches += 1

        graph_probs.append(torch.sigmoid(graph_logits).cpu().numpy())
        graph_true.append(yg.cpu().numpy())

        node_probs.append(torch.sigmoid(node_logits).cpu().numpy())
        node_true.append(yn.cpu().numpy())

    graph_probs = np.concatenate(graph_probs)
    graph_true = np.concatenate(graph_true)

    node_probs = np.concatenate(node_probs)
    node_true = np.concatenate(node_true)

    gm = binary_metrics(graph_true, graph_probs, threshold=graph_threshold)
    nm = node_metrics(node_true, node_probs, threshold=node_threshold)

    return {
        "loss": total_loss / max(1, batches),
        **gm,
        **nm,
    }


def print_metrics(prefix, metrics):
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


# -----------------------------
# Main training
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="graph_dataset/paper1_temporal_graphs_full.npz")
    parser.add_argument("--out-dir", default="models/temporal_gcn_float")
    parser.add_argument("--split-mode", choices=["prototype", "placement"], default="prototype")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temporal-dim", type=int, default=8)
    parser.add_argument("--gcn-hidden", type=int, default=16)
    parser.add_argument("--gcn-out", type=int, default=8)
    parser.add_argument("--node-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    d = np.load(args.data, allow_pickle=True)
    edge_index = d["edge_index"].astype(np.int64)
    A_hat = build_normalized_adjacency(edge_index, num_nodes=16).to(device)

    train_idx, val_idx, test_idx = make_splits(args.data, args.split_mode)

    print()
    print("split sizes:")
    print("  train:", len(train_idx))
    print("  val:  ", len(val_idx))
    print("  test: ", len(test_idx))

    # Compute class imbalance weights from train split.
    y_graph_train = d["y_graph"][train_idx].astype(np.float32)
    y_node_train = d["y_node"][train_idx].astype(np.float32)

    graph_pos = y_graph_train.sum()
    graph_neg = len(y_graph_train) - graph_pos
    graph_pos_weight = graph_neg / max(graph_pos, 1.0)

    node_pos = y_node_train.sum()
    node_neg = y_node_train.size - node_pos
    node_pos_weight = node_neg / max(node_pos, 1.0)

    print()
    print("class weights:")
    print("  graph pos weight:", float(graph_pos_weight))
    print("  node pos weight: ", float(node_pos_weight))

    train_ds = NoCTemporalGraphDataset(args.data, train_idx)
    val_ds = NoCTemporalGraphDataset(args.data, val_idx)
    test_ds = NoCTemporalGraphDataset(args.data, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    input_feature_dim = int(d["x"].shape[-1])
    print("input feature dim:", input_feature_dim)

    model = TemporalGCN(
        num_nodes=16,
        input_features=input_feature_dim,
        temporal_dim=args.temporal_dim,
        gcn_hidden=args.gcn_hidden,
        gcn_out=args.gcn_out,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print()
    print("model:", model)
    print("parameters:", total_params)

    bce_graph = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(graph_pos_weight), device=device)
    )

    bce_node = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(node_pos_weight), device=device)
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_f1 = -1.0
    best_epoch = -1

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        batches = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False)

        for batch in pbar:
            x = batch["x"].to(device)
            yg = batch["y_graph"].to(device)
            yn = batch["y_node"].to(device)

            graph_logits, node_logits = model(x, A_hat)

            loss_graph = bce_graph(graph_logits, yg)
            loss_node = bce_node(node_logits, yn)
            loss = loss_graph + args.node_loss_weight * loss_node

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(1, batches)

        train_metrics = evaluate(model, train_loader, A_hat, device)
        val_metrics = evaluate(model, val_loader, A_hat, device)

        print()
        print(f"epoch {epoch:03d} train_loss={train_loss:.4f}")
        print_metrics("  train", train_metrics)
        print_metrics("  val  ", val_metrics)

        item = {
            "epoch": epoch,
            "train_loss_step": train_loss,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(item)

        # Select by node localization F1 first, then graph F1.
        val_score = val_metrics["node_f1"] + val_metrics["f1"]

        if val_score > best_val_f1:
            best_val_f1 = val_score
            best_epoch = epoch

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "A_hat": A_hat.detach().cpu(),
                "best_epoch": best_epoch,
                "best_val_score": best_val_f1,
                "val_metrics": val_metrics,
            }

            torch.save(checkpoint, out_dir / "best_model.pt")
            print("  saved best_model.pt")

        with (out_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

    print()
    print("loading best checkpoint...")
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    train_metrics = evaluate(model, train_loader, A_hat, device)
    val_metrics = evaluate(model, val_loader, A_hat, device)
    test_metrics = evaluate(model, test_loader, A_hat, device)

    print()
    print("FINAL")
    print("best epoch:", ckpt["best_epoch"])
    print_metrics("train", train_metrics)
    print_metrics("val  ", val_metrics)
    print_metrics("test ", test_metrics)

    summary = {
        "data": args.data,
        "split_mode": args.split_mode,
        "best_epoch": int(ckpt["best_epoch"]),
        "parameters": int(total_params),
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "class_weights": {
            "graph_pos_weight": float(graph_pos_weight),
            "node_pos_weight": float(node_pos_weight),
        },
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Save split indices for repeatability.
    np.savez_compressed(
        out_dir / "splits.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    print()
    print("wrote:")
    print(" ", out_dir / "best_model.pt")
    print(" ", out_dir / "history.json")
    print(" ", out_dir / "summary.json")
    print(" ", out_dir / "splits.npz")


if __name__ == "__main__":
    main()
