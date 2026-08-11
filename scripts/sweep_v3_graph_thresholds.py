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
    load_model,
    infer,
    binary_metrics,
)

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
    ap.add_argument("--conv-dir", required=True)
    ap.add_argument("--tcn-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    data = load_graph_arrays(args.data)
    split = data["split"].astype(str)
    val_idx = np.where(split == "val")[0].astype(np.int64)
    test_idx = np.where(split == "test")[0].astype(np.int64)

    conv_model, conv_a = load_model("conv", args.conv_dir, data, device)
    tcn_model, tcn_a = load_model("tcn", args.tcn_dir, data, device)

    lines = []
    lines.append("V3 graph-threshold sweep")
    lines.append("=" * 100)
    lines.append(f"data: {args.data}")
    lines.append(f"conv_dir: {args.conv_dir}")
    lines.append(f"tcn_dir: {args.tcn_dir}")

    for split_name, idx in [("VAL", val_idx), ("TEST", test_idx)]:
        conv_pred = infer(data, idx, conv_model, conv_a, device, args.batch_size)
        tcn_pred = infer(data, idx, tcn_model, tcn_a, device, args.batch_size)

        lines.extend(sweep_one(f"{split_name} Conv1D-GCN", conv_pred))
        lines.extend(sweep_one(f"{split_name} TCN-GCN", tcn_pred))

    text = "\n".join(lines)
    print(text)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print()
    print("wrote:", out)

if __name__ == "__main__":
    main()
