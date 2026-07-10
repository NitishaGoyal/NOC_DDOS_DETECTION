#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))

from train_temporal_gcn import (
    NoCTemporalGraphDataset,
    TemporalGCN,
    build_normalized_adjacency,
)


@torch.no_grad()
def collect_predictions(model, loader, A_hat, device):
    model.eval()

    graph_probs = []
    node_probs = []
    y_graph = []
    y_node = []
    run_ids = []

    for batch in loader:
        x = batch["x"].to(device)

        graph_logits, node_logits = model(x, A_hat)

        graph_probs.append(torch.sigmoid(graph_logits).cpu().numpy())
        node_probs.append(torch.sigmoid(node_logits).cpu().numpy())
        y_graph.append(batch["y_graph"].numpy())
        y_node.append(batch["y_node"].numpy())
        run_ids.extend(batch["run_id"])

    return {
        "graph_probs": np.concatenate(graph_probs),
        "node_probs": np.concatenate(node_probs),
        "y_graph": np.concatenate(y_graph),
        "y_node": np.concatenate(y_node),
        "run_ids": np.array(run_ids, dtype=str),
    }


def summarize_topk(name, pred):
    node_probs = pred["node_probs"]
    y_node = pred["y_node"].astype(np.int64)
    y_graph = pred["y_graph"].astype(np.int64)
    run_ids = pred["run_ids"]

    attack_mask = y_graph == 1

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    if attack_mask.sum() == 0:
        print("No attack samples.")
        return

    grouped = defaultdict(list)

    for i in np.where(attack_mask)[0]:
        true_nodes = set(np.where(y_node[i] == 1)[0].tolist())
        k_true = len(true_nodes)

        ranked = np.argsort(-node_probs[i])

        top1 = set(ranked[:1].tolist())
        top2 = set(ranked[:2].tolist())
        top3 = set(ranked[:3].tolist())
        top4 = set(ranked[:4].tolist())
        top_true_k = set(ranked[:k_true].tolist())

        row = {
            "run_id": run_ids[i],
            "k_true": k_true,
            "top1_any_hit": int(len(top1 & true_nodes) > 0),
            "top2_all_hit": int(true_nodes.issubset(top2)),
            "top3_all_hit": int(true_nodes.issubset(top3)),
            "top4_all_hit": int(true_nodes.issubset(top4)),
            "top_true_k_exact": int(top_true_k == true_nodes),
            "best_true_rank": min(np.where(np.isin(ranked, list(true_nodes)))[0]) + 1,
            "worst_true_rank": max(np.where(np.isin(ranked, list(true_nodes)))[0]) + 1,
        }

        grouped["all"].append(row)
        grouped[f"{k_true}_attackers"].append(row)
        grouped[run_ids[i]].append(row)

    def avg(rows, key):
        return sum(r[key] for r in rows) / max(1, len(rows))

    def print_group(label, rows):
        print(
            f"{label:38s} "
            f"n={len(rows):5d} "
            f"top1_any={avg(rows, 'top1_any_hit'):.4f} "
            f"topK_exact={avg(rows, 'top_true_k_exact'):.4f} "
            f"top2_all={avg(rows, 'top2_all_hit'):.4f} "
            f"top3_all={avg(rows, 'top3_all_hit'):.4f} "
            f"top4_all={avg(rows, 'top4_all_hit'):.4f} "
            f"avg_worst_rank={avg(rows, 'worst_true_rank'):.2f}"
        )

    print(
        f"{'group':38s} "
        f"{'n':>7s} "
        f"{'top1_any':>12s} "
        f"{'topK_exact':>12s} "
        f"{'top2_all':>10s} "
        f"{'top3_all':>10s} "
        f"{'top4_all':>10s} "
        f"{'avg_worst_rank':>15s}"
    )
    print("-" * 125)

    print_group("all", grouped["all"])

    for label in ["1_attackers", "2_attackers", "3_attackers"]:
        if label in grouped:
            print_group(label, grouped[label])

    print()
    print("Per-run breakdown:")
    print("-" * 125)

    for run_id in sorted(k for k in grouped.keys() if k not in {"all", "1_attackers", "2_attackers", "3_attackers"}):
        print_group(run_id, grouped[run_id])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    d = np.load(args.data, allow_pickle=True)
    splits = np.load(model_dir / "splits.npz")

    edge_index = d["edge_index"].astype(np.int64)
    A_hat = build_normalized_adjacency(edge_index, num_nodes=16).to(device)

    input_feature_dim = int(d["x"].shape[-1])
    print("input feature dim:", input_feature_dim)

    ckpt = torch.load(
        model_dir / "best_model.pt",
        map_location=device,
        weights_only=False,
    )

    model_args = ckpt["args"]

    model = TemporalGCN(
        num_nodes=16,
        input_features=input_feature_dim,
        temporal_dim=model_args["temporal_dim"],
        gcn_hidden=model_args["gcn_hidden"],
        gcn_out=model_args["gcn_out"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    for split_name in ["val", "test"]:
        ds = NoCTemporalGraphDataset(args.data, splits[f"{split_name}_idx"])
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        pred = collect_predictions(model, loader, A_hat, device)
        summarize_topk(split_name, pred)


if __name__ == "__main__":
    main()
