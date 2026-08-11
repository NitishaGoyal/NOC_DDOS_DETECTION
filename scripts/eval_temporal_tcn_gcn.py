#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_temporal_tcn_gcn_early_stopping import (
    NoCTemporalGraphDataset,
    TemporalTCNGCN,
    build_normalized_adjacency,
    make_splits,
)


def get_ckpt_arg(ckpt_args, name, default):
    if isinstance(ckpt_args, dict):
        return ckpt_args.get(name, default)
    return getattr(ckpt_args, name, default)


def binary_metrics(y_true, y_prob, threshold):
    y_true = y_true.astype(np.int64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = float((y_true == y_pred).mean())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "acc": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tnr": float(tnr),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def node_metrics(y_true, y_prob, threshold):
    y_true = y_true.astype(np.int64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    flat = binary_metrics(y_true.reshape(-1), y_prob.reshape(-1), threshold)
    exact = float(np.all(y_true == y_pred, axis=1).mean())

    return {
        "node_acc": flat["acc"],
        "node_precision": flat["precision"],
        "node_recall": flat["recall"],
        "node_f1": flat["f1"],
        "exact_localization": exact,
    }


def load_indices(data_path, model_dir, split_mode):
    split_file = model_dir / "splits.npz"

    if split_file.exists():
        splits = np.load(split_file)
        return splits["train_idx"], splits["val_idx"], splits["test_idx"]

    train_idx, val_idx, test_idx = make_splits(str(data_path), split_mode)
    return train_idx, val_idx, test_idx


def load_model_and_data(args):
    data_path = Path(args.data)
    model_dir = Path(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("device:", device)

    data = np.load(data_path, allow_pickle=True)
    x_shape = data["x"].shape
    input_feature_dim = int(x_shape[-1])
    num_nodes = int(x_shape[1])

    print("input feature dim:", input_feature_dim)
    print("number of nodes:", num_nodes)

    ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    split_mode = args.split_mode or get_ckpt_arg(ckpt_args, "split_mode", "placement")

    model = TemporalTCNGCN(
        input_features=input_feature_dim,
        temporal_dim=int(get_ckpt_arg(ckpt_args, "temporal_dim", 8)),
        gcn_hidden=int(get_ckpt_arg(ckpt_args, "gcn_hidden", 16)),
        gcn_out=int(get_ckpt_arg(ckpt_args, "gcn_out", 8)),
        tcn_dropout=float(get_ckpt_arg(ckpt_args, "tcn_dropout", 0.1)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if "A_hat" in ckpt:
        a_hat = ckpt["A_hat"].to(device)
    else:
        edge_index = data["edge_index"].astype(np.int64)
        a_hat = build_normalized_adjacency(edge_index, num_nodes).to(device)

    train_idx, val_idx, test_idx = load_indices(data_path, model_dir, split_mode)

    print("split mode:", split_mode)
    print("split sizes:")
    print("  train:", len(train_idx))
    print("  val:  ", len(val_idx))
    print("  test: ", len(test_idx))

    return data_path, model_dir, device, model, a_hat, train_idx, val_idx, test_idx


@torch.no_grad()
def infer_split(data_path, indices, model, a_hat, device, batch_size):
    dataset = NoCTemporalGraphDataset(str(data_path), indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    graph_probs = []
    node_probs = []
    y_graph = []
    y_node = []
    run_ids = []
    attackers = []
    end_epochs = []

    for batch in loader:
        x = batch["x"].to(device)
        graph_logits, node_logits = model(x, a_hat)

        graph_probs.append(torch.sigmoid(graph_logits).cpu().numpy())
        node_probs.append(torch.sigmoid(node_logits).cpu().numpy())
        y_graph.append(batch["y_graph"].numpy())
        y_node.append(batch["y_node"].numpy())

        run_ids.extend([str(x) for x in batch["run_id"]])
        attackers.extend([str(x) for x in batch["attackers"]])
        end_epochs.extend([int(x) for x in batch["end_epoch"]])

    return {
        "graph_prob": np.concatenate(graph_probs),
        "node_prob": np.concatenate(node_probs),
        "y_graph": np.concatenate(y_graph),
        "y_node": np.concatenate(y_node),
        "run_id": np.asarray(run_ids, dtype=str),
        "attackers": np.asarray(attackers, dtype=str),
        "end_epoch": np.asarray(end_epochs, dtype=np.int64),
    }


def threshold_sweep(val, test, graph_threshold):
    thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)

    print("\nNode threshold sweep on validation set")
    print(f"{'thr':>6s} {'g_f1':>8s} {'node_prec':>10s} {'node_rec':>10s} {'node_f1':>10s} {'exact':>10s}")
    print("-" * 62)

    best_thr = None
    best_node_f1 = -1.0

    for thr in thresholds:
        g = binary_metrics(val["y_graph"], val["graph_prob"], graph_threshold)
        n = node_metrics(val["y_node"], val["node_prob"], thr)

        print(
            f"{thr:6.2f} "
            f"{g['f1']:8.4f} "
            f"{n['node_precision']:10.4f} "
            f"{n['node_recall']:10.4f} "
            f"{n['node_f1']:10.4f} "
            f"{n['exact_localization']:10.4f}"
        )

        if n["node_f1"] > best_node_f1:
            best_node_f1 = n["node_f1"]
            best_thr = float(thr)

    print(f"\nbest validation node threshold by node F1: {best_thr:.2f}")

    g = binary_metrics(test["y_graph"], test["graph_prob"], graph_threshold)
    n = node_metrics(test["y_node"], test["node_prob"], best_thr)

    print("\nTest set using best validation node threshold")
    print("graph acc:", g["acc"])
    print("graph precision:", g["precision"])
    print("graph recall:", g["recall"])
    print("graph f1:", g["f1"])
    print("graph fpr:", g["fpr"])
    print("node precision:", n["node_precision"])
    print("node recall:", n["node_recall"])
    print("node f1:", n["node_f1"])
    print("exact localization:", n["exact_localization"])

    print("\nTest set using fixed node threshold 0.50")
    n050 = node_metrics(test["y_node"], test["node_prob"], 0.50)
    print("node precision:", n050["node_precision"])
    print("node recall:", n050["node_recall"])
    print("node f1:", n050["node_f1"])
    print("exact localization:", n050["exact_localization"])


def topk_for_mask(y_node, node_prob, mask):
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return {
            "n": 0,
            "top1_any": 0.0,
            "topK_exact": 0.0,
            "top2_all": 0.0,
            "top3_all": 0.0,
            "top4_all": 0.0,
            "avg_worst_rank": 0.0,
        }

    top1_any = 0
    topK_exact = 0
    top2_all = 0
    top3_all = 0
    top4_all = 0
    worst_ranks = []

    for i in indices:
        true_nodes = set(np.where(y_node[i] >= 0.5)[0].tolist())
        k = len(true_nodes)
        order = np.argsort(-node_prob[i]).tolist()
        rank = {node: r + 1 for r, node in enumerate(order)}

        top1_any += int(order[0] in true_nodes)
        topK_exact += int(set(order[:k]) == true_nodes)
        top2_all += int(true_nodes.issubset(set(order[:2])))
        top3_all += int(true_nodes.issubset(set(order[:3])))
        top4_all += int(true_nodes.issubset(set(order[:4])))

        worst_ranks.append(max(rank[node] for node in true_nodes))

    n = len(indices)
    return {
        "n": int(n),
        "top1_any": top1_any / n,
        "topK_exact": topK_exact / n,
        "top2_all": top2_all / n,
        "top3_all": top3_all / n,
        "top4_all": top4_all / n,
        "avg_worst_rank": float(np.mean(worst_ranks)),
    }


def print_topk_row(name, metrics):
    print(
        f"{name:38s} "
        f"n={metrics['n']:5d} "
        f"top1_any={metrics['top1_any']:.4f} "
        f"topK_exact={metrics['topK_exact']:.4f} "
        f"top2_all={metrics['top2_all']:.4f} "
        f"top3_all={metrics['top3_all']:.4f} "
        f"top4_all={metrics['top4_all']:.4f} "
        f"avg_worst_rank={metrics['avg_worst_rank']:.2f}"
    )


def topk_report(split_name, split):
    y_node = split["y_node"].astype(np.int64)
    node_prob = split["node_prob"]
    run_ids = split["run_id"]

    attacker_count = y_node.sum(axis=1).astype(np.int64)
    attack_mask = attacker_count > 0

    print("\n" + "=" * 80)
    print(split_name)
    print("=" * 80)
    print(
        f"{'group':38s} {'n':>7s} "
        f"{'top1_any':>10s} {'topK_exact':>11s} "
        f"{'top2_all':>10s} {'top3_all':>10s} "
        f"{'top4_all':>10s} {'avg_worst_rank':>15s}"
    )
    print("-" * 125)

    print_topk_row("all", topk_for_mask(y_node, node_prob, attack_mask))

    for k in sorted(set(attacker_count[attack_mask].tolist())):
        print_topk_row(f"{k}_attackers", topk_for_mask(y_node, node_prob, attacker_count == k))

    print("\nPer-run breakdown:")
    print("-" * 125)

    for run_id in sorted(set(run_ids[attack_mask].tolist())):
        mask = (run_ids == run_id) & attack_mask
        print_topk_row(run_id, topk_for_mask(y_node, node_prob, mask))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--graph-threshold", type=float, default=0.50)
    parser.add_argument("--split-mode", choices=["prototype", "placement"], default=None)
    parser.add_argument("--mode", choices=["thresholds", "topk", "both"], default="both")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    data_path, model_dir, device, model, a_hat, train_idx, val_idx, test_idx = load_model_and_data(args)

    val = infer_split(data_path, val_idx, model, a_hat, device, args.batch_size)
    test = infer_split(data_path, test_idx, model, a_hat, device, args.batch_size)

    if args.mode in ("thresholds", "both"):
        threshold_sweep(val, test, args.graph_threshold)

    if args.mode in ("topk", "both"):
        topk_report("val", val)
        topk_report("test", test)


if __name__ == "__main__":
    main()
