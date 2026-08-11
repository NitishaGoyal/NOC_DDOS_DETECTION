#!/usr/bin/env python3
"""
V4-A1 validation threshold selection and one-time development-test transfer.

Modes
-----
validation  Infer on validation only, sweep thresholds, freeze thresholds.
test        Infer on test exactly once using frozen validation thresholds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


MODEL_NAME = "Conv1D-TemporalGCN"
EXPECTED_PARAMETER_COUNT = 882


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_empty_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = list(path.iterdir())
    if existing:
        raise RuntimeError(f"Output directory must be empty: {path}; entries={[p.name for p in existing[:20]]}")


def load_metadata(data_dir: Path) -> dict[str, Any]:
    return json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))


def split_indices(data_dir: Path, split_name: str) -> np.ndarray:
    metadata = load_metadata(data_dir)
    mapping = metadata["code_maps"]["split"]
    split_id = np.load(data_dir / "split_id.npy", mmap_mode="r")
    return np.flatnonzero(split_id == int(mapping[split_name])).astype(np.int64)


class V4MemmapDataset(Dataset):
    def __init__(self, data_dir: str | Path, indices: np.ndarray):
        self.data_dir = str(Path(data_dir).resolve())
        self.indices = np.asarray(indices, dtype=np.int64)
        self._x = None
        self._y_graph = None
        self._y_node = None
        self._run_index = None
        self._attack_kind_id = None
        self._attacker_count = None

    def __len__(self) -> int:
        return int(self.indices.size)

    def _open(self) -> None:
        if self._x is None:
            root = Path(self.data_dir)
            self._x = np.load(root / "x.npy", mmap_mode="r")
            self._y_graph = np.load(root / "y_graph.npy", mmap_mode="r")
            self._y_node = np.load(root / "y_node.npy", mmap_mode="r")
            self._run_index = np.load(root / "run_index.npy", mmap_mode="r")
            self._attack_kind_id = np.load(root / "attack_kind_id.npy", mmap_mode="r")
            self._attacker_count = np.load(root / "attacker_count.npy", mmap_mode="r")

    def __getitem__(self, local_index: int) -> dict[str, torch.Tensor]:
        self._open()
        global_index = int(self.indices[local_index])
        return {
            "x": torch.from_numpy(np.array(self._x[global_index], dtype=np.float32, copy=True)),
            "y_graph": torch.tensor(np.float32(self._y_graph[global_index])),
            "y_node": torch.from_numpy(np.array(self._y_node[global_index], dtype=np.float32, copy=True)),
            "global_index": torch.tensor(global_index, dtype=torch.int64),
            "run_index": torch.tensor(int(self._run_index[global_index]), dtype=torch.int64),
            "attack_kind_id": torch.tensor(int(self._attack_kind_id[global_index]), dtype=torch.int64),
            "attacker_count": torch.tensor(int(self._attacker_count[global_index]), dtype=torch.int64),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("_x", "_y_graph", "_y_node", "_run_index", "_attack_kind_id", "_attacker_count"):
            state[key] = None
        return state


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        agg = torch.einsum("ij,bjf->bif", a_hat, h)
        return self.linear(agg)


class TemporalEncoder(nn.Module):
    def __init__(self, in_features: int = 2, embedding_dim: int = 8, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_features, embedding_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, time_steps, feature_dim = x.shape
        x = x.reshape(batch_size * num_nodes, time_steps, feature_dim).permute(0, 2, 1)
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
        graph_logits = self.graph_head(h.mean(dim=1)).squeeze(-1)
        return graph_logits, node_logits


def build_normalized_adjacency(edge_index: np.ndarray, num_nodes: int = 16) -> torch.Tensor:
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for source, destination in zip(edge_index[0], edge_index[1]):
        adjacency[int(destination), int(source)] = 1.0
    degree = adjacency.sum(dim=1)
    inverse = torch.pow(degree, -0.5)
    inverse[torch.isinf(inverse)] = 0.0
    diagonal = torch.diag(inverse)
    return diagonal @ adjacency @ diagonal


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    truth = y_true.astype(bool)
    pred = y_prob >= threshold
    tp = int(np.sum(truth & pred))
    tn = int(np.sum(~truth & ~pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
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


def node_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    flat = binary_metrics(y_true.reshape(-1), y_prob.reshape(-1), threshold)
    truth = y_true >= 0.5
    pred = y_prob >= threshold
    flat.update(
        {
            "node_accuracy": flat.pop("accuracy"),
            "node_precision": flat.pop("precision"),
            "node_recall": flat.pop("recall"),
            "node_f1": flat.pop("f1"),
            "exact_localization": float(np.mean(np.all(truth == pred, axis=1))),
        }
    )
    return flat


def infer(
    data_dir: Path,
    indices: np.ndarray,
    checkpoint_path: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("args", {})
    model = TemporalGCN(
        input_features=24,
        temporal_dim=int(model_args.get("temporal_dim", 8)),
        gcn_hidden=int(model_args.get("gcn_hidden", 16)),
        gcn_out=int(model_args.get("gcn_out", 8)),
    ).to(device)
    if sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("Model parameter count drift.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    edge_index = np.load(data_dir / "edge_index.npy").astype(np.int64)
    a_hat = checkpoint.get("A_hat")
    if a_hat is None:
        a_hat = build_normalized_adjacency(edge_index)
    a_hat = a_hat.to(device)

    dataset = V4MemmapDataset(data_dir, indices)
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(**kwargs)

    output: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for batch_number, batch in enumerate(loader, start=1):
            x = batch["x"].to(device, non_blocking=True)
            graph_logits, node_logits = model(x, a_hat)
            output["graph_prob"].append(torch.sigmoid(graph_logits).cpu().numpy().astype(np.float32))
            output["node_prob"].append(torch.sigmoid(node_logits).cpu().numpy().astype(np.float32))
            for key in ("y_graph", "y_node", "global_index", "run_index", "attack_kind_id", "attacker_count"):
                output[key].append(batch[key].cpu().numpy())
            if batch_number == 1 or batch_number % 500 == 0:
                print(f"inference_batches={batch_number}")
    return {key: np.concatenate(values, axis=0) for key, values in output.items()}


def threshold_values(start: float, end: float, step: float) -> np.ndarray:
    count = int(round((end - start) / step)) + 1
    return np.linspace(start, end, count, dtype=np.float64)


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validation_mode(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    ensure_empty_output(out)
    data_dir = Path(args.data_dir).resolve()
    checkpoint = Path(args.model_dir) / "best_model.pt"
    indices = split_indices(data_dir, "val")
    pred = infer(
        data_dir, indices, checkpoint, args.batch_size, args.num_workers,
        args.pin_memory, args.persistent_workers, args.prefetch_factor
    )
    np.savez_compressed(out / "validation_predictions.npz", **pred)
    graph_rows = [
        binary_metrics(pred["y_graph"], pred["graph_prob"], float(t))
        for t in threshold_values(args.threshold_start, args.threshold_end, args.threshold_step)
    ]
    node_rows = [
        node_metrics(pred["y_node"], pred["node_prob"], float(t))
        for t in threshold_values(args.threshold_start, args.threshold_end, args.threshold_step)
    ]
    for row in graph_rows:
        row["task"] = "graph"
    for row in node_rows:
        row["task"] = "node"
    all_rows = graph_rows + node_rows
    all_fields = sorted({key for row in all_rows for key in row})
    write_rows(out / "threshold_sweep.csv", all_rows, all_fields)

    # Graph: maximize F1, then minimize FPR, then maximize recall, then closest to 0.5.
    best_graph = max(
        graph_rows,
        key=lambda r: (r["f1"], -r["fpr"], r["recall"], -abs(r["threshold"] - 0.5)),
    )
    # Node: maximize node F1, then exact localization, precision, then closest to 0.5.
    best_node = max(
        node_rows,
        key=lambda r: (
            r["node_f1"],
            r["exact_localization"],
            r["node_precision"],
            -abs(r["threshold"] - 0.5),
        ),
    )
    selected = {
        "graph_threshold": best_graph["threshold"],
        "node_threshold": best_node["threshold"],
        "graph_selection_rule": "max graph F1; tie min FPR; tie max recall; tie closest to 0.5",
        "node_selection_rule": "max node F1; tie max exact localization; tie max precision; tie closest to 0.5",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split": "validation",
        "test_accessed": False,
    }
    json_dump(out / "selected_thresholds.json", selected)
    validation_metrics = {
        "graph": binary_metrics(pred["y_graph"], pred["graph_prob"], best_graph["threshold"]),
        "node": node_metrics(pred["y_node"], pred["node_prob"], best_node["threshold"]),
        "sample_count": int(len(indices)),
        "selection": selected,
    }
    json_dump(out / "validation_metrics.json", validation_metrics)
    print("V4_A1_VALIDATION_THRESHOLD_SELECTION_PASS")
    print(json.dumps(validation_metrics, indent=2))
    return 0


def decode_map(metadata: dict[str, Any], name: str) -> dict[int, str]:
    mapping = metadata.get("code_maps", {}).get(name, {})
    return {int(code): str(label) for label, code in mapping.items()}


def group_metric_rows(
    group_name: str,
    group_values: np.ndarray,
    pred: dict[str, np.ndarray],
    graph_threshold: float,
    node_threshold: float,
    label_map: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for value in sorted(np.unique(group_values).tolist()):
        mask = group_values == value
        graph = binary_metrics(pred["y_graph"][mask], pred["graph_prob"][mask], graph_threshold)
        node = node_metrics(pred["y_node"][mask], pred["node_prob"][mask], node_threshold)
        row = {
            group_name: label_map.get(int(value), str(value)) if label_map else value,
            "sample_count": int(mask.sum()),
            **{f"graph_{k}": v for k, v in graph.items() if k != "threshold"},
            **{k: v for k, v in node.items() if k != "threshold"},
        }
        rows.append(row)
    return rows


def test_mode(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    ensure_empty_output(out)
    lock = out / "TEST_TRANSFER_LOCK.json"
    thresholds_path = Path(args.selected_thresholds)
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    graph_threshold = float(thresholds["graph_threshold"])
    node_threshold = float(thresholds["node_threshold"])
    checkpoint = Path(args.model_dir) / "best_model.pt"
    expected_checkpoint_hash = thresholds.get("checkpoint_sha256")
    actual_checkpoint_hash = sha256_file(checkpoint)
    if expected_checkpoint_hash and expected_checkpoint_hash != actual_checkpoint_hash:
        raise RuntimeError("Checkpoint differs from validation threshold-selection checkpoint.")
    lock_data = {
        "status": "STARTED",
        "mode": "one-time development-test transfer",
        "checkpoint_sha256": actual_checkpoint_hash,
        "selected_thresholds_sha256": sha256_file(thresholds_path),
    }
    json_dump(lock, lock_data)

    data_dir = Path(args.data_dir).resolve()
    indices = split_indices(data_dir, "test")
    pred = infer(
        data_dir, indices, checkpoint, args.batch_size, args.num_workers,
        args.pin_memory, args.persistent_workers, args.prefetch_factor
    )
    np.savez_compressed(out / "test_predictions.npz", **pred)
    graph = binary_metrics(pred["y_graph"], pred["graph_prob"], graph_threshold)
    node = node_metrics(pred["y_node"], pred["node_prob"], node_threshold)
    metrics = {
        "designation": "development blind test",
        "independent_publication_holdout": False,
        "sample_count": int(len(indices)),
        "graph": graph,
        "node": node,
        "checkpoint_sha256": actual_checkpoint_hash,
        "selected_thresholds": thresholds,
        "thresholds_selected_on_test": False,
    }
    json_dump(out / "test_metrics.json", metrics)
    write_rows(
        out / "test_confusion_matrix.csv",
        [
            {"actual": "normal", "predicted_normal": graph["tn"], "predicted_attack": graph["fp"]},
            {"actual": "attack", "predicted_normal": graph["fn"], "predicted_attack": graph["tp"]},
        ],
    )

    metadata = load_metadata(data_dir)
    run_rows = group_metric_rows(
        "run_index", pred["run_index"], pred, graph_threshold, node_threshold
    )
    run_records = metadata.get("runs", [])
    by_run = {i: record for i, record in enumerate(run_records)}
    for row in run_rows:
        record = by_run.get(int(row["run_index"]), {})
        row["run_id"] = record.get("run_id", "")
        row["split"] = record.get("split", "")
        row["profile"] = record.get("profile", "")
        row["attack_kind"] = record.get("attack_kind", "")
        row["attacker_count_metadata"] = record.get("attacker_count", "")
    write_rows(out / "test_per_run_metrics.csv", run_rows)

    kind_map = decode_map(metadata, "attack_kind")
    write_rows(
        out / "test_per_attack_kind.csv",
        group_metric_rows(
            "attack_kind", pred["attack_kind_id"], pred,
            graph_threshold, node_threshold, kind_map
        ),
    )
    write_rows(
        out / "test_per_attacker_count.csv",
        group_metric_rows(
            "attacker_count", pred["attacker_count"], pred,
            graph_threshold, node_threshold
        ),
    )
    router_rows = []
    for router in range(pred["y_node"].shape[1]):
        row = binary_metrics(
            pred["y_node"][:, router], pred["node_prob"][:, router], node_threshold
        )
        router_rows.append({"router": router, **row})
    write_rows(out / "test_per_router.csv", router_rows)

    graph_pred = pred["graph_prob"] >= graph_threshold
    node_pred = pred["node_prob"] >= node_threshold
    graph_truth = pred["y_graph"] >= 0.5
    node_truth = pred["y_node"] >= 0.5
    failure_mask = (graph_pred != graph_truth) | (~np.all(node_pred == node_truth, axis=1))
    failure_rows = []
    positions = np.flatnonzero(failure_mask)
    for position in positions:
        true_nodes = np.flatnonzero(node_truth[position]).tolist()
        predicted_nodes = np.flatnonzero(node_pred[position]).tolist()
        failure_rows.append(
            {
                "global_index": int(pred["global_index"][position]),
                "run_index": int(pred["run_index"][position]),
                "true_graph": int(graph_truth[position]),
                "predicted_graph": int(graph_pred[position]),
                "graph_probability": float(pred["graph_prob"][position]),
                "true_attackers": "-".join(map(str, true_nodes)),
                "predicted_attackers": "-".join(map(str, predicted_nodes)),
            }
        )
    write_rows(
        out / "test_failure_cases.csv",
        failure_rows,
        [
            "global_index", "run_index", "true_graph", "predicted_graph",
            "graph_probability", "true_attackers", "predicted_attackers"
        ],
    )
    lock_data.update(
        {
            "status": "COMPLETED",
            "test_sample_count": int(len(indices)),
            "test_predictions_sha256": sha256_file(out / "test_predictions.npz"),
            "test_metrics_sha256": sha256_file(out / "test_metrics.json"),
        }
    )
    json_dump(lock, lock_data)
    print("V4_A1_ONE_TIME_DEVELOPMENT_TEST_TRANSFER_PASS")
    print(json.dumps(metrics, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["validation", "test"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--selected-thresholds")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "validation":
        return validation_mode(args)
    if not args.selected_thresholds:
        raise ValueError("--selected-thresholds is required for test mode")
    return test_mode(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
