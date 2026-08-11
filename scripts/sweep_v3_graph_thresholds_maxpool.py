#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from compare_v3_conv_vs_tcn_errors import (
    load_graph_arrays,
    infer,
    binary_metrics,
)
import train_temporal_tcn_maxpool_gcn_v3 as mpmod


def load_meanpool_model(model_dir, data, device):
    model_dir = Path(model_dir)
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    def get_arg(name, default):
        if isinstance(ckpt_args, dict):
            return ckpt_args.get(name, default)
        return getattr(ckpt_args, name, default)

    input_feature_dim = int(data["x"].shape[-1])
    num_nodes = int(data["x"].shape[1])
    edge_index = data["edge_index"].astype(np.int64)

    model = mpmod.TemporalTCNGCN(
        input_features=input_feature_dim,
        temporal_dim=int(get_arg("temporal_dim", 8)),
        gcn_hidden=int(get_arg("gcn_hidden", 16)),
        gcn_out=int(get_arg("gcn_out", 8)),
        tcn_dropout=float(get_arg("tcn_dropout", 0.1)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if "A_hat" in ckpt:
        a_hat = ckpt["A_hat"].to(device)
    else:
        a_hat = mpmod.build_normalized_adjacency(edge_index, num_nodes).to(device)

    return model, a_hat


def sweep_one(name, pred):
    lines = []
    lines.append("")
    lines.append("=" * 100)
    lines.append(name)
    lines.append("=" * 100)
    lines.append(f"{'thr':>6s} {'acc':>8s} {'prec':>8s} {'rec':>8s} {'f1':>8s} {'fpr':>8s} {'fp':>8s} {'fn':>8s}")
    lines.append("-" * 100)

    best_f1 = (-1, None, None)
    best_low_fpr_10 = (-1, None, None)
    best_low_fpr_05 = (-1, None, None)

    for thr in np.round(np.arange(0.05, 1.00, 0.05), 2):
        m = binary_metrics(pred["y_graph"], pred["graph_prob"], float(thr))
        lines.append(
            f"{thr:6.2f} {m['acc']:8.4f} {m['precision']:8.4f} {m['recall']:8.4f} "
            f"{m['f1']:8.4f} {m['fpr']:8.4f} {m['fp']:8d} {m['fn']:8d}"
        )

        if m["f1"] > best_f1[0]:
            best_f1 = (m["f1"], thr, m)

        if m["fpr"] <= 0.10 and m["f1"] > best_low_fpr_10[0]:
            best_low_fpr_10 = (m["f1"], thr, m)

        if m["fpr"] <= 0.05 and m["f1"] > best_low_fpr_05[0]:
            best_low_fpr_05 = (m["f1"], thr, m)

    lines.append("")
    lines.append(f"Best F1 threshold: {best_f1[1]:.2f} f1={best_f1[2]['f1']:.4f} fpr={best_f1[2]['fpr']:.4f} recall={best_f1[2]['recall']:.4f}")

    if best_low_fpr_10[1] is not None:
        lines.append(f"Best threshold with FPR <= 0.10: {best_low_fpr_10[1]:.2f} f1={best_low_fpr_10[2]['f1']:.4f} fpr={best_low_fpr_10[2]['fpr']:.4f} recall={best_low_fpr_10[2]['recall']:.4f}")
    else:
        lines.append("No threshold found with FPR <= 0.10")

    if best_low_fpr_05[1] is not None:
        lines.append(f"Best threshold with FPR <= 0.05: {best_low_fpr_05[1]:.2f} f1={best_low_fpr_05[2]['f1']:.4f} fpr={best_low_fpr_05[2]['fpr']:.4f} recall={best_low_fpr_05[2]['recall']:.4f}")
    else:
        lines.append("No threshold found with FPR <= 0.05")

    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    data = load_graph_arrays(args.data)
    split = data["split"].astype(str)
    val_idx = np.where(split == "val")[0].astype(np.int64)
    test_idx = np.where(split == "test")[0].astype(np.int64)

    model, a_hat = load_meanpool_model(args.model_dir, data, device)

    lines = []
    lines.append("V3 MaxPool graph-threshold sweep")
    lines.append("=" * 100)
    lines.append(f"data: {args.data}")
    lines.append(f"model_dir: {args.model_dir}")

    for split_name, idx in [("VAL", val_idx), ("TEST", test_idx)]:
        pred = infer(data, idx, model, a_hat, device, args.batch_size)

        lines.extend(sweep_one(f"{split_name} TCN-MaxPool-GCN", pred))

    text = "\n".join(lines)
    print(text)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print()
    print("wrote:", out)

if __name__ == "__main__":
    main()
