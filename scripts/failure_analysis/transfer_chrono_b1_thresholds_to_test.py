#!/usr/bin/env python3
"""
Chrono-B1 Stage B1.12: transfer frozen validation thresholds to the blind test set.

This script:
- reads the four thresholds selected in B1.11;
- exports aligned A1/B1 test predictions;
- applies the frozen thresholds unchanged;
- computes overall graph metrics;
- computes attack-only node metrics;
- computes threshold-independent Top-1/2/3 and rank metrics;
- writes aligned test prediction NPZ files and a transfer report.

No threshold is re-selected on test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("chrono_a1_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class IndexedTemporalDataset(Dataset):
    def __init__(self, data_dir: Path, indices: np.ndarray):
        self.x = np.load(data_dir / "x.npy", mmap_mode="r")
        self.y_graph = np.load(data_dir / "y_graph.npy", mmap_mode="r")
        self.y_node = np.load(data_dir / "y_node.npy", mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        real_index = int(self.indices[position])
        return {
            "x": torch.from_numpy(
                np.asarray(self.x[real_index], dtype=np.float32)
            ),
            "y_graph": torch.tensor(
                float(self.y_graph[real_index]), dtype=torch.float32
            ),
            "y_node": torch.from_numpy(
                np.asarray(self.y_node[real_index], dtype=np.float32)
            ),
            "real_index": torch.tensor(real_index, dtype=torch.int64),
        }


def read_metadata(data_dir: Path, indices: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in (
        "run_id",
        "profile",
        "strength",
        "active_cores",
        "attackers",
        "end_epoch",
        "seed",
        "split",
    ):
        path = data_dir / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing metadata file: {path}")
        array = np.load(
            path,
            mmap_mode="r" if name == "end_epoch" else None,
            allow_pickle=True,
        )
        result[name] = np.asarray(array[indices])
    return result


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = (np.asarray(y_prob).reshape(-1) >= threshold).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    fpr = safe_div(fp, fp + tn)
    tnr = safe_div(tn, tn + fp)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tnr": tnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def node_metrics_attack_only(
    y_graph: np.ndarray,
    y_node: np.ndarray,
    node_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    attack_mask = np.asarray(y_graph).reshape(-1) == 1
    truth = np.asarray(y_node[attack_mask], dtype=np.int64)
    prob = np.asarray(node_prob[attack_mask])
    pred = (prob >= threshold).astype(np.int64)

    flat_truth = truth.reshape(-1)
    flat_pred = pred.reshape(-1)

    tp = int(np.sum((flat_truth == 1) & (flat_pred == 1)))
    tn = int(np.sum((flat_truth == 0) & (flat_pred == 0)))
    fp = int(np.sum((flat_truth == 0) & (flat_pred == 1)))
    fn = int(np.sum((flat_truth == 1) & (flat_pred == 0)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    exact = float(np.mean(np.all(pred == truth, axis=1))) if len(truth) else 0.0

    return {
        "threshold": float(threshold),
        "denominator": "attack-positive test samples only",
        "attack_sample_count": int(len(truth)),
        "positive_node_count": int(flat_truth.sum()),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "exact_localization": exact,
        "mean_predicted_nodes": float(np.mean(pred.sum(axis=1))) if len(pred) else 0.0,
        "mean_true_nodes": float(np.mean(truth.sum(axis=1))) if len(truth) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def ranking_metrics(y_graph: np.ndarray, y_node: np.ndarray, node_prob: np.ndarray) -> dict[str, Any]:
    attack_mask = np.asarray(y_graph).reshape(-1) == 1
    truth = np.asarray(y_node[attack_mask], dtype=np.int64)
    prob = np.asarray(node_prob[attack_mask], dtype=np.float64)

    if len(truth) == 0:
        raise ValueError("No attack-positive test samples available.")

    attacker_counts = truth.sum(axis=1).astype(np.int64)
    if np.any(attacker_counts <= 0):
        raise ValueError("At least one attack-positive sample has no attacker label.")

    order = np.argsort(-prob, axis=1, kind="stable")
    inverse_rank = np.empty_like(order)
    row_ids = np.arange(len(order))[:, None]
    inverse_rank[row_ids, order] = np.arange(1, order.shape[1] + 1)[None, :]

    best_ranks: list[int] = []
    all_true_ranks: list[int] = []
    top_hits = {1: [], 2: [], 3: []}
    top_recalls = {1: [], 2: [], 3: []}
    exact_top_m: list[bool] = []
    sample_rows: list[dict[str, Any]] = []

    for i in range(len(truth)):
        true_nodes = np.flatnonzero(truth[i] == 1)
        ranks = inverse_rank[i, true_nodes].astype(np.int64)
        best_rank = int(ranks.min())
        best_ranks.append(best_rank)
        all_true_ranks.extend(ranks.tolist())

        row = {
            "attacker_count": int(len(true_nodes)),
            "best_rank": best_rank,
            "mean_rank": float(np.mean(ranks)),
            "reciprocal_best_rank": 1.0 / best_rank,
        }

        for k in (1, 2, 3):
            selected = set(order[i, :k].tolist())
            hit = bool(any(int(node) in selected for node in true_nodes))
            recall = float(sum(int(node) in selected for node in true_nodes) / len(true_nodes))
            top_hits[k].append(hit)
            top_recalls[k].append(recall)
            row[f"top{k}_hit"] = hit
            row[f"top{k}_recall"] = recall

        m = len(true_nodes)
        exact = set(order[i, :m].tolist()) == set(true_nodes.tolist())
        exact_top_m.append(exact)
        row["exact_top_m"] = bool(exact)
        sample_rows.append(row)

    overall = {
        "denominator": "attack-positive test samples only",
        "attack_sample_count": int(len(truth)),
        "attacker_node_count": int(attacker_counts.sum()),
        "top1_hit_rate": float(np.mean(top_hits[1])),
        "top2_hit_rate": float(np.mean(top_hits[2])),
        "top3_hit_rate": float(np.mean(top_hits[3])),
        "top1_attacker_recall": float(np.mean(top_recalls[1])),
        "top2_attacker_recall": float(np.mean(top_recalls[2])),
        "top3_attacker_recall": float(np.mean(top_recalls[3])),
        "mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in best_ranks])),
        "mean_best_attacker_rank": float(np.mean(best_ranks)),
        "median_best_attacker_rank": float(np.median(best_ranks)),
        "mean_all_attacker_rank": float(np.mean(all_true_ranks)),
        "median_all_attacker_rank": float(np.median(all_true_ranks)),
        "exact_top_m_set_accuracy": float(np.mean(exact_top_m)),
    }

    by_count: dict[str, dict[str, Any]] = {}
    for count in sorted(np.unique(attacker_counts).tolist()):
        rows = [
            row
            for row in sample_rows
            if int(row["attacker_count"]) == int(count)
        ]
        by_count[str(int(count))] = {
            "sample_count": int(len(rows)),
            "top1_hit_rate": float(np.mean([row["top1_hit"] for row in rows])),
            "top2_hit_rate": float(np.mean([row["top2_hit"] for row in rows])),
            "top3_hit_rate": float(np.mean([row["top3_hit"] for row in rows])),
            "top1_attacker_recall": float(np.mean([row["top1_recall"] for row in rows])),
            "top2_attacker_recall": float(np.mean([row["top2_recall"] for row in rows])),
            "top3_attacker_recall": float(np.mean([row["top3_recall"] for row in rows])),
            "mean_reciprocal_rank": float(
                np.mean([row["reciprocal_best_rank"] for row in rows])
            ),
            "mean_best_attacker_rank": float(np.mean([row["best_rank"] for row in rows])),
            "mean_all_attacker_rank": float(np.mean([row["mean_rank"] for row in rows])),
            "exact_top_m_set_accuracy": float(np.mean([row["exact_top_m"] for row in rows])),
        }

    return {"overall": overall, "by_attacker_count": by_count}


def export_model(
    *,
    label: str,
    model_dir: Path,
    source_module,
    data_dir: Path,
    output_path: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = model_dir / "best_model.pt"
    splits_path = model_dir / "splits.npz"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    splits = np.load(splits_path)
    test_idx = splits["test_idx"].astype(np.int64)

    saved_args = checkpoint["args"]
    x = np.load(data_dir / "x.npy", mmap_mode="r")
    edge_index = np.load(data_dir / "edge_index.npy").astype(np.int64)

    num_nodes = int(x.shape[1])
    input_features = int(x.shape[-1])

    model = source_module.TemporalGCN(
        input_features=input_features,
        temporal_dim=int(saved_args["temporal_dim"]),
        gcn_hidden=int(saved_args["gcn_hidden"]),
        gcn_out=int(saved_args["gcn_out"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    a_hat = source_module.build_normalized_adjacency(
        edge_index,
        num_nodes,
    ).to(device)

    dataset = IndexedTemporalDataset(data_dir, test_idx)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    graph_parts: list[np.ndarray] = []
    node_parts: list[np.ndarray] = []
    y_graph_parts: list[np.ndarray] = []
    y_node_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"export {label} test", leave=False):
            batch_x = batch["x"].to(device, non_blocking=True)
            graph_logits, node_logits = model(batch_x, a_hat)

            graph_parts.append(
                torch.sigmoid(graph_logits).cpu().numpy().astype(np.float32)
            )
            node_parts.append(
                torch.sigmoid(node_logits).cpu().numpy().astype(np.float32)
            )
            y_graph_parts.append(batch["y_graph"].numpy().astype(np.float32))
            y_node_parts.append(batch["y_node"].numpy().astype(np.float32))
            index_parts.append(batch["real_index"].numpy().astype(np.int64))

    graph_prob = np.concatenate(graph_parts, axis=0)
    node_prob = np.concatenate(node_parts, axis=0)
    y_graph = np.concatenate(y_graph_parts, axis=0)
    y_node = np.concatenate(y_node_parts, axis=0)
    real_index = np.concatenate(index_parts, axis=0)

    if not np.array_equal(real_index, test_idx):
        raise RuntimeError(f"{label}: row order does not match saved test_idx")

    metadata = read_metadata(data_dir, real_index)

    np.savez_compressed(
        output_path,
        model_label=np.asarray(label),
        split=np.asarray("test"),
        real_index=real_index,
        graph_prob=graph_prob,
        node_prob=node_prob,
        y_graph=y_graph,
        y_node=y_node,
        run_id=metadata["run_id"],
        profile=metadata["profile"],
        strength=metadata["strength"],
        active_cores=metadata["active_cores"],
        attackers=metadata["attackers"],
        end_epoch=metadata["end_epoch"],
        seed=metadata["seed"],
        dataset_split=metadata["split"],
    )

    return {
        "label": label,
        "rows": int(len(real_index)),
        "nodes": int(node_prob.shape[1]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as d:
        return {key: np.asarray(d[key]) for key in d.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model-source", required=True, type=Path)
    parser.add_argument("--a1-dir", required=True, type=Path)
    parser.add_argument("--b1-dir", required=True, type=Path)
    parser.add_argument("--threshold-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    source_path = args.model_source.resolve()
    a1_dir = args.a1_dir.resolve()
    b1_dir = args.b1_dir.resolve()
    threshold_path = args.threshold_report.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: output directory exists and is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
    frozen = thresholds["frozen_thresholds_for_test_transfer"]

    source_module = load_module(source_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    a1_path = output_dir / "a1_test_predictions.npz"
    b1_path = output_dir / "b1_test_predictions.npz"

    a1_manifest = export_model(
        label="A1",
        model_dir=a1_dir,
        source_module=source_module,
        data_dir=data_dir,
        output_path=a1_path,
        batch_size=args.batch_size,
        device=device,
    )
    b1_manifest = export_model(
        label="B1",
        model_dir=b1_dir,
        source_module=source_module,
        data_dir=data_dir,
        output_path=b1_path,
        batch_size=args.batch_size,
        device=device,
    )

    a1 = load_npz(a1_path)
    b1 = load_npz(b1_path)

    alignment_checks = {
        key: bool(np.array_equal(a1[key], b1[key]))
        for key in (
            "real_index",
            "y_graph",
            "y_node",
            "run_id",
            "profile",
            "strength",
            "active_cores",
            "attackers",
            "end_epoch",
            "seed",
            "dataset_split",
        )
    }

    if not all(alignment_checks.values()):
        failed = [k for k, v in alignment_checks.items() if not v]
        raise SystemExit(f"STOP: A1/B1 test alignment failed: {failed}")

    if not np.all(a1["dataset_split"].astype(str) == "test"):
        raise SystemExit("STOP: A1 export contains non-test rows")
    if not np.all(b1["dataset_split"].astype(str) == "test"):
        raise SystemExit("STOP: B1 export contains non-test rows")

    results: dict[str, Any] = {}
    for label, data, graph_thr, node_thr in (
        ("a1", a1, float(frozen["a1_graph"]), float(frozen["a1_node"])),
        ("b1", b1, float(frozen["b1_graph"]), float(frozen["b1_node"])),
    ):
        results[label] = {
            "graph": binary_metrics(
                data["y_graph"],
                data["graph_prob"],
                graph_thr,
            ),
            "node_attack_only": node_metrics_attack_only(
                data["y_graph"],
                data["y_node"],
                data["node_prob"],
                node_thr,
            ),
            "localization_ranking": ranking_metrics(
                data["y_graph"],
                data["y_node"],
                data["node_prob"],
            ),
        }

    deltas = {
        "graph_f1": results["b1"]["graph"]["f1"] - results["a1"]["graph"]["f1"],
        "graph_recall": results["b1"]["graph"]["recall"] - results["a1"]["graph"]["recall"],
        "graph_fpr": results["b1"]["graph"]["fpr"] - results["a1"]["graph"]["fpr"],
        "attack_only_node_f1": (
            results["b1"]["node_attack_only"]["micro_f1"]
            - results["a1"]["node_attack_only"]["micro_f1"]
        ),
        "attack_only_exact_localization": (
            results["b1"]["node_attack_only"]["exact_localization"]
            - results["a1"]["node_attack_only"]["exact_localization"]
        ),
        "top1_hit_rate": (
            results["b1"]["localization_ranking"]["overall"]["top1_hit_rate"]
            - results["a1"]["localization_ranking"]["overall"]["top1_hit_rate"]
        ),
        "top3_hit_rate": (
            results["b1"]["localization_ranking"]["overall"]["top3_hit_rate"]
            - results["a1"]["localization_ranking"]["overall"]["top3_hit_rate"]
        ),
        "mean_reciprocal_rank": (
            results["b1"]["localization_ranking"]["overall"]["mean_reciprocal_rank"]
            - results["a1"]["localization_ranking"]["overall"]["mean_reciprocal_rank"]
        ),
    }

    report = {
        "stage": "B1.12",
        "protocol": {
            "thresholds_selected_on": "validation only",
            "thresholds_reselected_on_test": False,
            "blind_test_transfer": True,
        },
        "frozen_thresholds": frozen,
        "alignment_checks": alignment_checks,
        "alignment_pass": all(alignment_checks.values()),
        "a1_manifest": a1_manifest,
        "b1_manifest": b1_manifest,
        "a1": results["a1"],
        "b1": results["b1"],
        "test_deltas_b1_minus_a1": deltas,
    }

    report_path = output_dir / "test_threshold_transfer.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("B1.12 BLIND-TEST THRESHOLD TRANSFER: PASS")
    for label in ("a1", "b1"):
        graph = results[label]["graph"]
        node = results[label]["node_attack_only"]
        rank = results[label]["localization_ranking"]["overall"]
        print(f"\n{label.upper()}")
        print(f"graph_threshold={graph['threshold']:.3f}")
        print(f"graph_accuracy={graph['accuracy']:.6f}")
        print(f"graph_precision={graph['precision']:.6f}")
        print(f"graph_recall={graph['recall']:.6f}")
        print(f"graph_f1={graph['f1']:.6f}")
        print(f"graph_fpr={graph['fpr']:.6f}")
        print(f"graph_tnr={graph['tnr']:.6f}")
        print(f"node_threshold={node['threshold']:.3f}")
        print(f"attack_only_node_precision={node['micro_precision']:.6f}")
        print(f"attack_only_node_recall={node['micro_recall']:.6f}")
        print(f"attack_only_node_f1={node['micro_f1']:.6f}")
        print(f"attack_only_exact_localization={node['exact_localization']:.6f}")
        print(f"top1_hit_rate={rank['top1_hit_rate']:.6f}")
        print(f"top2_hit_rate={rank['top2_hit_rate']:.6f}")
        print(f"top3_hit_rate={rank['top3_hit_rate']:.6f}")
        print(f"top1_attacker_recall={rank['top1_attacker_recall']:.6f}")
        print(f"top2_attacker_recall={rank['top2_attacker_recall']:.6f}")
        print(f"top3_attacker_recall={rank['top3_attacker_recall']:.6f}")
        print(f"mean_reciprocal_rank={rank['mean_reciprocal_rank']:.6f}")
        print(f"mean_best_attacker_rank={rank['mean_best_attacker_rank']:.6f}")
        print(f"mean_all_attacker_rank={rank['mean_all_attacker_rank']:.6f}")
        print(f"exact_top_m_set_accuracy={rank['exact_top_m_set_accuracy']:.6f}")

    print("\nTEST DELTAS (B1 - A1)")
    for key, value in deltas.items():
        print(f"{key}={value:+.6f}")

    print(f"\na1_rows={a1_manifest['rows']}")
    print(f"b1_rows={b1_manifest['rows']}")
    print(f"alignment_pass={all(alignment_checks.values())}")
    print(f"output={report_path}")


if __name__ == "__main__":
    main()
