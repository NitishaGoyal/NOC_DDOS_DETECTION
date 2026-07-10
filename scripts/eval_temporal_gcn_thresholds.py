#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))

from train_temporal_gcn import (
    NoCTemporalGraphDataset,
    TemporalGCN,
    build_normalized_adjacency,
    evaluate,
)


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

    ckpt = torch.load(
        model_dir / "best_model.pt",
        map_location=device,
        weights_only=False,
    )

    model_args = ckpt["args"]

    input_feature_dim = int(d["x"].shape[-1])
    print("input feature dim:", input_feature_dim)

    model = TemporalGCN(
        num_nodes=16,
        input_features=input_feature_dim,
        temporal_dim=model_args["temporal_dim"],
        gcn_hidden=model_args["gcn_hidden"],
        gcn_out=model_args["gcn_out"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_ds = NoCTemporalGraphDataset(args.data, splits["val_idx"])
    test_ds = NoCTemporalGraphDataset(args.data, splits["test_idx"])

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print()
    print("Node threshold sweep on validation set")
    print(
        f"{'thr':>6s} "
        f"{'g_f1':>8s} "
        f"{'node_prec':>10s} "
        f"{'node_rec':>10s} "
        f"{'node_f1':>10s} "
        f"{'exact':>10s}"
    )
    print("-" * 62)

    best_thr = None
    best_score = -1.0

    for thr in np.arange(0.05, 0.96, 0.05):
        m = evaluate(
            model,
            val_loader,
            A_hat,
            device,
            graph_threshold=0.5,
            node_threshold=float(thr),
        )

        score = m["node_f1"]

        if score > best_score:
            best_score = score
            best_thr = float(thr)

        print(
            f"{thr:6.2f} "
            f"{m['f1']:8.4f} "
            f"{m['node_precision']:10.4f} "
            f"{m['node_recall']:10.4f} "
            f"{m['node_f1']:10.4f} "
            f"{m['exact_localization']:10.4f}"
        )

    print()
    print("best validation node threshold by node F1:", f"{best_thr:.2f}")

    print()
    print("Test set using best validation node threshold")
    m = evaluate(
        model,
        test_loader,
        A_hat,
        device,
        graph_threshold=0.5,
        node_threshold=best_thr,
    )

    print("graph acc:", m["acc"])
    print("graph precision:", m["precision"])
    print("graph recall:", m["recall"])
    print("graph f1:", m["f1"])
    print("graph fpr:", m["fpr"])
    print("node precision:", m["node_precision"])
    print("node recall:", m["node_recall"])
    print("node f1:", m["node_f1"])
    print("exact localization:", m["exact_localization"])


if __name__ == "__main__":
    main()
