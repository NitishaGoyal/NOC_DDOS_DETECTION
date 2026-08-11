#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import train_temporal_gcn_v3 as convmod
import train_temporal_tcn_gcn_v3 as tcnmod


def load_graph_arrays(data_path):
    data_path = Path(data_path)

    if data_path.is_dir():
        return {
            "x": np.load(data_path / "x.npy", mmap_mode="r"),
            "y_graph": np.load(data_path / "y_graph.npy", mmap_mode="r"),
            "y_node": np.load(data_path / "y_node.npy", mmap_mode="r"),
            "edge_index": np.load(data_path / "edge_index.npy"),
            "run_id": np.load(data_path / "run_id.npy", allow_pickle=True).astype(str),
            "split": np.load(data_path / "split.npy", allow_pickle=True).astype(str),
            "profile": np.load(data_path / "profile.npy", allow_pickle=True).astype(str),
            "active_cores": np.load(data_path / "active_cores.npy", allow_pickle=True).astype(str),
            "attackers": np.load(data_path / "attackers.npy", allow_pickle=True).astype(str),
            "strength": np.load(data_path / "strength.npy", allow_pickle=True).astype(str),
            "seed": np.load(data_path / "seed.npy", allow_pickle=True).astype(str),
        }

    d = np.load(data_path, allow_pickle=True)
    out = {k: d[k] for k in d.files}

    for key in ["run_id", "split", "profile", "active_cores", "attackers", "strength", "seed"]:
        if key in out:
            out[key] = out[key].astype(str)

    return out


class ArrayDataset(Dataset):
    def __init__(self, data, indices):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        return {
            "x": torch.tensor(self.data["x"][j], dtype=torch.float32),
            "y_graph": torch.tensor(self.data["y_graph"][j], dtype=torch.float32),
            "y_node": torch.tensor(self.data["y_node"][j], dtype=torch.float32),
            "idx": j,
        }


def get_arg(args_obj, name, default):
    if isinstance(args_obj, dict):
        return args_obj.get(name, default)
    return getattr(args_obj, name, default)


def load_model(kind, model_dir, data, device):
    model_dir = Path(model_dir)
    ckpt_path = model_dir / "best_model.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    input_feature_dim = int(data["x"].shape[-1])
    num_nodes = int(data["x"].shape[1])
    edge_index = data["edge_index"].astype(np.int64)

    if kind == "conv":
        model = convmod.TemporalGCN(
            input_features=input_feature_dim,
            temporal_dim=int(get_arg(ckpt_args, "temporal_dim", 8)),
            gcn_hidden=int(get_arg(ckpt_args, "gcn_hidden", 16)),
            gcn_out=int(get_arg(ckpt_args, "gcn_out", 8)),
        ).to(device)
        build_adj = convmod.build_normalized_adjacency

    elif kind == "tcn":
        model = tcnmod.TemporalTCNGCN(
            input_features=input_feature_dim,
            temporal_dim=int(get_arg(ckpt_args, "temporal_dim", 8)),
            gcn_hidden=int(get_arg(ckpt_args, "gcn_hidden", 16)),
            gcn_out=int(get_arg(ckpt_args, "gcn_out", 8)),
            tcn_dropout=float(get_arg(ckpt_args, "tcn_dropout", 0.1)),
        ).to(device)
        build_adj = tcnmod.build_normalized_adjacency

    else:
        raise ValueError(f"Unknown model kind: {kind}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if "A_hat" in ckpt:
        a_hat = ckpt["A_hat"].to(device)
    else:
        a_hat = build_adj(edge_index, num_nodes).to(device)

    return model, a_hat


@torch.no_grad()
def infer(data, indices, model, a_hat, device, batch_size):
    dataset = ArrayDataset(data, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    graph_probs = []
    node_probs = []
    y_graph = []
    y_node = []
    global_idx = []

    for batch in loader:
        x = batch["x"].to(device)
        graph_logits, node_logits = model(x, a_hat)

        graph_probs.append(torch.sigmoid(graph_logits).cpu().numpy())
        node_probs.append(torch.sigmoid(node_logits).cpu().numpy())
        y_graph.append(batch["y_graph"].numpy())
        y_node.append(batch["y_node"].numpy())
        global_idx.append(batch["idx"].numpy())

    return {
        "graph_prob": np.concatenate(graph_probs),
        "node_prob": np.concatenate(node_probs),
        "y_graph": np.concatenate(y_graph).astype(np.int64),
        "y_node": np.concatenate(y_node).astype(np.int64),
        "idx": np.concatenate(global_idx).astype(np.int64),
    }


def binary_metrics(y_true, y_prob, threshold):
    y_true = y_true.astype(np.int64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tnr": tnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def node_metrics(y_true, y_prob, threshold):
    y_true = y_true.astype(np.int64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    flat = binary_metrics(y_true.reshape(-1), y_prob.reshape(-1), threshold)
    exact = float(np.all(y_pred == y_true, axis=1).mean()) if len(y_true) else 0.0

    return {
        "node_precision": flat["precision"],
        "node_recall": flat["recall"],
        "node_f1": flat["f1"],
        "exact_loc": exact,
    }


def topk_metrics(y_true, y_prob):
    y_true = y_true.astype(np.int64)

    attack_mask = y_true.sum(axis=1) > 0
    y_true = y_true[attack_mask]
    y_prob = y_prob[attack_mask]

    if len(y_true) == 0:
        return {
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

    for yt, yp in zip(y_true, y_prob):
        true_nodes = set(np.where(yt > 0)[0].tolist())
        k = len(true_nodes)
        order = np.argsort(-yp).tolist()
        rank = {node: r + 1 for r, node in enumerate(order)}

        top1_any += int(order[0] in true_nodes)
        topK_exact += int(set(order[:k]) == true_nodes)
        top2_all += int(true_nodes.issubset(set(order[:2])))
        top3_all += int(true_nodes.issubset(set(order[:3])))
        top4_all += int(true_nodes.issubset(set(order[:4])))
        worst_ranks.append(max(rank[node] for node in true_nodes))

    n = len(y_true)

    return {
        "top1_any": top1_any / n,
        "topK_exact": topK_exact / n,
        "top2_all": top2_all / n,
        "top3_all": top3_all / n,
        "top4_all": top4_all / n,
        "avg_worst_rank": float(np.mean(worst_ranks)),
    }


def find_best_node_threshold(pred):
    best_thr = 0.50
    best_f1 = -1.0

    for thr in np.round(np.arange(0.05, 1.00, 0.05), 2):
        m = node_metrics(pred["y_node"], pred["node_prob"], float(thr))
        if m["node_f1"] > best_f1:
            best_f1 = m["node_f1"]
            best_thr = float(thr)

    return best_thr, best_f1


def summarize_model(pred, node_thr):
    graph = binary_metrics(pred["y_graph"], pred["graph_prob"], 0.50)
    node = node_metrics(pred["y_node"], pred["node_prob"], node_thr)
    topk = topk_metrics(pred["y_node"], pred["node_prob"])

    return {
        "graph_acc": graph["acc"],
        "graph_precision": graph["precision"],
        "graph_recall": graph["recall"],
        "graph_f1": graph["f1"],
        "graph_fpr": graph["fpr"],
        "node_precision": node["node_precision"],
        "node_recall": node["node_recall"],
        "node_f1": node["node_f1"],
        "exact_loc": node["exact_loc"],
        "top1_any": topk["top1_any"],
        "topK_exact": topk["topK_exact"],
        "top2_all": topk["top2_all"],
        "top3_all": topk["top3_all"],
        "top4_all": topk["top4_all"],
        "avg_worst_rank": topk["avg_worst_rank"],
    }


def format_summary(title, summary):
    return (
        f"{title:12s} "
        f"g_f1={summary['graph_f1']:.4f} "
        f"g_fpr={summary['graph_fpr']:.4f} "
        f"node_p={summary['node_precision']:.4f} "
        f"node_r={summary['node_recall']:.4f} "
        f"node_f1={summary['node_f1']:.4f} "
        f"exact={summary['exact_loc']:.4f} "
        f"topK={summary['topK_exact']:.4f} "
        f"top4={summary['top4_all']:.4f}"
    )


def group_values(data, pred, group_name):
    global_idx = pred["idx"]

    if group_name == "attacker_count":
        return pred["y_node"].sum(axis=1).astype(int).astype(str)

    if group_name not in data:
        raise KeyError(f"Dataset does not contain metadata field: {group_name}")

    return data[group_name][global_idx].astype(str)


def group_metrics(data, pred, group_name, node_thr):
    values = group_values(data, pred, group_name)

    rows = []

    for value in sorted(set(values.tolist())):
        mask = values == value
        sub = {
            "graph_prob": pred["graph_prob"][mask],
            "node_prob": pred["node_prob"][mask],
            "y_graph": pred["y_graph"][mask],
            "y_node": pred["y_node"][mask],
            "idx": pred["idx"][mask],
        }

        summary = summarize_model(sub, node_thr)
        summary["name"] = value
        summary["n"] = int(mask.sum())
        rows.append(summary)

    return rows


def add_group_table(lines, data, conv_pred, tcn_pred, group_name, conv_thr, tcn_thr):
    lines.append("")
    lines.append("=" * 120)
    lines.append(f"GROUP: {group_name} using val-tuned node thresholds")
    lines.append("=" * 120)

    conv_rows = {r["name"]: r for r in group_metrics(data, conv_pred, group_name, conv_thr)}
    tcn_rows = {r["name"]: r for r in group_metrics(data, tcn_pred, group_name, tcn_thr)}

    names = sorted(set(conv_rows) | set(tcn_rows))

    lines.append(
        f"{'group_value':40s} {'n':>7s} "
        f"{'conv_gf1':>9s} {'tcn_gf1':>9s} "
        f"{'conv_fpr':>9s} {'tcn_fpr':>9s} "
        f"{'conv_nf1':>9s} {'tcn_nf1':>9s} "
        f"{'conv_exact':>11s} {'tcn_exact':>10s} "
        f"{'conv_topK':>10s} {'tcn_topK':>9s} "
        f"{'conv_top4':>10s} {'tcn_top4':>9s}"
    )
    lines.append("-" * 150)

    for name in names:
        c = conv_rows.get(name)
        t = tcn_rows.get(name)
        n = c["n"] if c else t["n"]

        lines.append(
            f"{name[:40]:40s} {n:7d} "
            f"{(c['graph_f1'] if c else 0):9.4f} {(t['graph_f1'] if t else 0):9.4f} "
            f"{(c['graph_fpr'] if c else 0):9.4f} {(t['graph_fpr'] if t else 0):9.4f} "
            f"{(c['node_f1'] if c else 0):9.4f} {(t['node_f1'] if t else 0):9.4f} "
            f"{(c['exact_loc'] if c else 0):11.4f} {(t['exact_loc'] if t else 0):10.4f} "
            f"{(c['topK_exact'] if c else 0):10.4f} {(t['topK_exact'] if t else 0):9.4f} "
            f"{(c['top4_all'] if c else 0):10.4f} {(t['top4_all'] if t else 0):9.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--conv-dir", required=True)
    parser.add_argument("--tcn-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("device:", device)

    data = load_graph_arrays(args.data)

    if "split" not in data:
        raise RuntimeError("Dataset must contain split metadata for V3 comparison.")

    split = data["split"].astype(str)
    val_idx = np.where(split == "val")[0].astype(np.int64)
    test_idx = np.where(split == "test")[0].astype(np.int64)

    print("val samples:", len(val_idx))
    print("test samples:", len(test_idx))

    print("loading Conv1D-GCN:", args.conv_dir)
    conv_model, conv_a = load_model("conv", args.conv_dir, data, device)

    print("loading TCN-GCN:", args.tcn_dir)
    tcn_model, tcn_a = load_model("tcn", args.tcn_dir, data, device)

    print("running Conv1D-GCN validation inference...")
    conv_val = infer(data, val_idx, conv_model, conv_a, device, args.batch_size)

    print("running TCN-GCN validation inference...")
    tcn_val = infer(data, val_idx, tcn_model, tcn_a, device, args.batch_size)

    print("running Conv1D-GCN test inference...")
    conv_test = infer(data, test_idx, conv_model, conv_a, device, args.batch_size)

    print("running TCN-GCN test inference...")
    tcn_test = infer(data, test_idx, tcn_model, tcn_a, device, args.batch_size)

    conv_thr, conv_val_f1 = find_best_node_threshold(conv_val)
    tcn_thr, tcn_val_f1 = find_best_node_threshold(tcn_val)

    lines = []
    lines.append("V3 Conv1D-GCN vs TCN-GCN Error Analysis")
    lines.append("=" * 120)
    lines.append(f"data:     {args.data}")
    lines.append(f"conv_dir: {args.conv_dir}")
    lines.append(f"tcn_dir:  {args.tcn_dir}")
    lines.append("")
    lines.append(f"Conv1D best validation node threshold: {conv_thr:.2f} val_node_f1={conv_val_f1:.4f}")
    lines.append(f"TCN    best validation node threshold: {tcn_thr:.2f} val_node_f1={tcn_val_f1:.4f}")
    lines.append("")

    lines.append("OVERALL TEST, FIXED NODE THRESHOLD 0.50")
    lines.append("-" * 120)
    lines.append(format_summary("Conv1D", summarize_model(conv_test, 0.50)))
    lines.append(format_summary("TCN", summarize_model(tcn_test, 0.50)))
    lines.append("")

    lines.append("OVERALL TEST, VAL-TUNED NODE THRESHOLD")
    lines.append("-" * 120)
    lines.append(format_summary("Conv1D", summarize_model(conv_test, conv_thr)))
    lines.append(format_summary("TCN", summarize_model(tcn_test, tcn_thr)))
    lines.append("")

    for group_name in [
        "attacker_count",
        "profile",
        "strength",
        "seed",
        "active_cores",
        "attackers",
        "run_id",
    ]:
        if group_name == "attacker_count" or group_name in data:
            add_group_table(lines, data, conv_test, tcn_test, group_name, conv_thr, tcn_thr)

    report = "\n".join(lines)
    print()
    print(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print()
    print("wrote:", out)


if __name__ == "__main__":
    main()
