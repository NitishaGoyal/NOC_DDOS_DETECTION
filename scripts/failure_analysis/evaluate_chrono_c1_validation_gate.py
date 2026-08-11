#!/usr/bin/env python3
"""
C1.4: validation-only A1 versus C1 evaluation and advancement gate.

This script performs no training and no test inference.

It:
- verifies C1.0-C1.3 provenance and source/checkpoint integrity;
- reconstructs A1 and C1 from their own frozen sources;
- exports aligned validation predictions only;
- selects graph and node thresholds independently on validation using the
  frozen common protocol;
- computes graph, run-level, strength-20, node, exact-localization, Top-k,
  and ranking metrics;
- applies the frozen C1 validation advancement gate;
- writes JSON, Markdown, CSV, and aligned NPZ artifacts.

Formal C1 advancement requires:
- at least one primary improvement;
- every mandatory safeguard;
- validation data only.

The test split is not inferred, calibrated, inspected, or used here.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


EXPECTED_A1_SOURCE_SHA256 = (
    "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b"
)
EXPECTED_A1_CHECKPOINT_SHA256 = (
    "c29cdd2303669a61454ed118ef0be4b8809a635b12c0c42806f8b6992340e0ef"
)
EXPECTED_C1_CHECKPOINT_SHA256 = (
    "9c7890a7a2fd1c389564621f4036f8ef1ecdc5cd40aeb82443410e5e32ee0865"
)

THRESHOLDS = np.round(np.arange(0.005, 1.0, 0.005), 3)

METADATA_NAMES = (
    "run_id",
    "profile",
    "strength",
    "active_cores",
    "attackers",
    "end_epoch",
    "seed",
    "split",
)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    raise ValueError("Checkpoint contains no model_state_dict/state_dict.")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def text(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if value is None:
        return "NA"
    result = str(value).strip()
    return result if result else "NA"


def parse_sequence(value: Any) -> list[Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, (int, np.integer)):
        number = int(value)
        return [] if number < 0 else [number]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number) or number < 0:
            return []
        return [int(number)]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    raw = str(value).strip()
    if not raw or raw.lower() in {"na", "nan", "none", "null", "[]", "()"}:
        return []

    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple, set, np.ndarray)):
            return list(parsed)
        if isinstance(parsed, (int, float)):
            return [parsed]
    except (ValueError, SyntaxError):
        pass

    for separator in ("-", ",", "_", " "):
        if separator in raw:
            return [part.strip() for part in raw.split(separator) if part.strip()]
    return [raw]


def canonical_sequence(value: Any) -> str:
    normalized: list[str] = []
    for item in parse_sequence(value):
        raw = text(item)
        if raw.lower() in {"na", "nan", "none", "null", "-1"}:
            continue
        try:
            normalized.append(str(int(float(raw))))
        except ValueError:
            normalized.append(raw)

    if not normalized:
        return "NA"

    def sort_key(token: str) -> tuple[int, Any]:
        try:
            return (0, int(token))
        except ValueError:
            return (1, token)

    return "-".join(sorted(dict.fromkeys(normalized), key=sort_key))


def canonical_strength(value: Any) -> str:
    raw = text(value).strip("'").strip('"')
    if raw.lower() in {"na", "nan", "none", "null", ""}:
        return "NA"
    try:
        number = float(raw)
        if math.isfinite(number):
            return str(int(number)) if number.is_integer() else f"{number:g}"
    except ValueError:
        pass
    return raw


class IndexedValidationDataset(Dataset):
    def __init__(self, data_dir: Path, indices: np.ndarray):
        self.x = np.load(data_dir / "x.npy", mmap_mode="r")
        self.y_graph = np.load(data_dir / "y_graph.npy", mmap_mode="r")
        self.y_node = np.load(data_dir / "y_node.npy", mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        real_index = int(self.indices[position])
        return {
            "x": torch.from_numpy(
                np.array(self.x[real_index], dtype=np.float32, copy=True)
            ),
            "y_graph": torch.tensor(
                float(self.y_graph[real_index]),
                dtype=torch.float32,
            ),
            "y_node": torch.from_numpy(
                np.array(self.y_node[real_index], dtype=np.float32, copy=True)
            ),
            "real_index": torch.tensor(real_index, dtype=torch.int64),
        }


def read_metadata(data_dir: Path, indices: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in METADATA_NAMES:
        path = data_dir / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing metadata file: {path}")
        array = np.load(path, allow_pickle=True)
        result[name] = np.asarray(array[indices])
    return result


def export_validation_predictions(
    *,
    label: str,
    source_path: Path,
    model_dir: Path,
    data_dir: Path,
    expected_val_idx: np.ndarray,
    output_path: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = model_dir / "best_model.pt"
    splits_path = model_dir / "splits.npz"

    module = load_module(source_path, f"chrono_c1_val_{label.lower()}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, dict):
        raise ValueError(f"{label}: checkpoint has no args dictionary.")

    with np.load(splits_path) as splits:
        val_idx = np.asarray(splits["val_idx"], dtype=np.int64)

    if not np.array_equal(val_idx, expected_val_idx):
        raise ValueError(f"{label}: validation indices differ from A1.")

    x = np.load(data_dir / "x.npy", mmap_mode="r")
    edge_index = np.load(data_dir / "edge_index.npy").astype(np.int64)
    node_count = int(x.shape[1])
    input_features = int(x.shape[-1])

    model = module.TemporalGCN(
        input_features=input_features,
        temporal_dim=int(saved_args["temporal_dim"]),
        gcn_hidden=int(saved_args["gcn_hidden"]),
        gcn_out=int(saved_args["gcn_out"]),
    ).to(device)
    model.load_state_dict(load_state_dict(checkpoint), strict=True)
    model.eval()

    a_hat = module.build_normalized_adjacency(
        edge_index,
        node_count,
    ).to(device)

    dataset = IndexedValidationDataset(data_dir, val_idx)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    graph_parts: list[np.ndarray] = []
    node_parts: list[np.ndarray] = []
    y_graph_parts: list[np.ndarray] = []
    y_node_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"export {label} val", leave=False):
            batch_x = batch["x"].to(device, non_blocking=True)
            graph_logits, node_logits = model(batch_x, a_hat)

            graph_parts.append(
                torch.sigmoid(graph_logits).cpu().numpy().astype(np.float32)
            )
            node_parts.append(
                torch.sigmoid(node_logits).cpu().numpy().astype(np.float32)
            )
            y_graph_parts.append(
                batch["y_graph"].numpy().astype(np.float32)
            )
            y_node_parts.append(
                batch["y_node"].numpy().astype(np.float32)
            )
            index_parts.append(
                batch["real_index"].numpy().astype(np.int64)
            )

    graph_prob = np.concatenate(graph_parts)
    node_prob = np.concatenate(node_parts)
    y_graph = np.concatenate(y_graph_parts)
    y_node = np.concatenate(y_node_parts)
    real_index = np.concatenate(index_parts)

    if not np.array_equal(real_index, val_idx):
        raise ValueError(f"{label}: exported row order differs from val_idx.")
    if not np.isfinite(graph_prob).all() or not np.isfinite(node_prob).all():
        raise ValueError(f"{label}: non-finite probabilities exported.")
    if np.any((graph_prob < 0) | (graph_prob > 1)):
        raise ValueError(f"{label}: graph probabilities outside [0, 1].")
    if np.any((node_prob < 0) | (node_prob > 1)):
        raise ValueError(f"{label}: node probabilities outside [0, 1].")

    metadata = read_metadata(data_dir, real_index)
    if not np.all(metadata["split"].astype(str) == "val"):
        raise ValueError(f"{label}: export contains non-validation rows.")

    np.savez_compressed(
        output_path,
        model_label=np.asarray(label),
        split=np.asarray("val"),
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
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "splits": str(splits_path),
        "splits_sha256": sha256(splits_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "saved_args": saved_args,
    }


def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = (
        np.asarray(y_prob, dtype=np.float64).reshape(-1) >= threshold
    ).astype(np.int64)

    tp = int(np.sum((truth == 1) & (pred == 1)))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "threshold": float(threshold),
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": safe_div(fp, fp + tn),
        "tnr": safe_div(tn, tn + fp),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def node_metrics(
    y_graph: np.ndarray,
    y_node: np.ndarray,
    node_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    attack_mask = np.asarray(y_graph).reshape(-1) == 1
    truth = np.asarray(y_node[attack_mask], dtype=np.int64)
    probability = np.asarray(node_prob[attack_mask], dtype=np.float64)
    pred = (probability >= threshold).astype(np.int64)

    flat_truth = truth.reshape(-1)
    flat_pred = pred.reshape(-1)

    tp = int(np.sum((flat_truth == 1) & (flat_pred == 1)))
    tn = int(np.sum((flat_truth == 0) & (flat_pred == 0)))
    fp = int(np.sum((flat_truth == 0) & (flat_pred == 1)))
    fn = int(np.sum((flat_truth == 1) & (flat_pred == 0)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    exact = float(np.mean(np.all(pred == truth, axis=1)))

    return {
        "threshold": float(threshold),
        "denominator": "attack-positive validation samples only",
        "attack_sample_count": int(len(truth)),
        "positive_node_count": int(flat_truth.sum()),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "exact_localization": exact,
        "mean_predicted_nodes": float(np.mean(pred.sum(axis=1))),
        "mean_true_nodes": float(np.mean(truth.sum(axis=1))),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def choose_graph(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row["f1"],
            -row["fpr"],
            row["recall"],
            -abs(row["threshold"] - 0.5),
            -row["threshold"],
        ),
    )


def choose_node(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row["micro_f1"],
            row["exact_localization"],
            row["micro_precision"],
            row["micro_recall"],
            -abs(row["threshold"] - 0.5),
            -row["threshold"],
        ),
    )


def ranking_metrics(
    y_graph: np.ndarray,
    y_node: np.ndarray,
    node_prob: np.ndarray,
) -> dict[str, Any]:
    attack_mask = np.asarray(y_graph).reshape(-1) == 1
    truth = np.asarray(y_node[attack_mask], dtype=np.int64)
    probability = np.asarray(node_prob[attack_mask], dtype=np.float64)

    attacker_counts = truth.sum(axis=1).astype(np.int64)
    if np.any(attacker_counts <= 0):
        raise ValueError("Attack-positive row without positive node label.")

    order = np.argsort(-probability, axis=1, kind="stable")
    inverse_rank = np.empty_like(order)
    row_ids = np.arange(len(order))[:, None]
    inverse_rank[row_ids, order] = np.arange(
        1,
        order.shape[1] + 1,
    )[None, :]

    top_hits = {1: [], 2: [], 3: []}
    top_recalls = {1: [], 2: [], 3: []}
    best_ranks: list[int] = []
    all_ranks: list[int] = []
    exact_top_m: list[bool] = []

    for i in range(len(truth)):
        true_nodes = np.flatnonzero(truth[i] == 1)
        ranks = inverse_rank[i, true_nodes].astype(np.int64)
        best_ranks.append(int(ranks.min()))
        all_ranks.extend(ranks.tolist())

        for k in (1, 2, 3):
            selected = set(order[i, :k].tolist())
            recovered = sum(int(node) in selected for node in true_nodes)
            top_hits[k].append(recovered > 0)
            top_recalls[k].append(recovered / len(true_nodes))

        m = len(true_nodes)
        exact_top_m.append(
            set(order[i, :m].tolist()) == set(true_nodes.tolist())
        )

    return {
        "denominator": "attack-positive validation samples only",
        "attack_sample_count": int(len(truth)),
        "top1_hit_rate": float(np.mean(top_hits[1])),
        "top2_hit_rate": float(np.mean(top_hits[2])),
        "top3_hit_rate": float(np.mean(top_hits[3])),
        "top1_attacker_recall": float(np.mean(top_recalls[1])),
        "top2_attacker_recall": float(np.mean(top_recalls[2])),
        "top3_attacker_recall": float(np.mean(top_recalls[3])),
        "mean_reciprocal_rank": float(
            np.mean([1.0 / rank for rank in best_ranks])
        ),
        "mean_best_attacker_rank": float(np.mean(best_ranks)),
        "mean_all_attacker_rank": float(np.mean(all_ranks)),
        "exact_top_m_set_accuracy": float(np.mean(exact_top_m)),
    }


def run_rows(
    label: str,
    data: dict[str, np.ndarray],
    threshold: float,
) -> list[dict[str, Any]]:
    run_ids = np.asarray([text(value) for value in data["run_id"]])
    profile = np.asarray(
        [text(value).lower() for value in data["profile"]]
    )
    strength = np.asarray(
        [canonical_strength(value) for value in data["strength"]]
    )
    active_cores = np.asarray(
        [canonical_sequence(value) for value in data["active_cores"]]
    )
    attackers = np.asarray(
        [canonical_sequence(value) for value in data["attackers"]]
    )
    seed = np.asarray([text(value) for value in data["seed"]])

    rows: list[dict[str, Any]] = []
    for run_id in sorted(set(run_ids.tolist())):
        mask = run_ids == run_id
        metrics = binary_metrics(
            data["y_graph"][mask],
            data["graph_prob"][mask],
            threshold,
        )
        positive = int(np.sum(data["y_graph"][mask] == 1))
        negative = int(np.sum(data["y_graph"][mask] == 0))
        if positive > 0 and negative == 0:
            run_class = "attack"
        elif negative > 0 and positive == 0:
            run_class = "normal"
        else:
            run_class = "mixed"

        first = int(np.flatnonzero(mask)[0])
        rows.append(
            {
                "model": label,
                "run_id": run_id,
                "run_class": run_class,
                "sample_count": int(np.sum(mask)),
                "profile": profile[first],
                "strength": strength[first],
                "active_cores": active_cores[first],
                "attackers": attackers[first],
                "seed": seed[first],
                **metrics,
            }
        )
    return rows


def summarize_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normal = [row for row in rows if row["run_class"] == "normal"]
    attack = [row for row in rows if row["run_class"] == "attack"]
    strength20 = [
        row for row in attack if canonical_strength(row["strength"]) == "20"
    ]

    worst_normal = max(
        normal,
        key=lambda row: (row["fpr"], row["run_id"]),
    )
    worst_attack = min(
        attack,
        key=lambda row: (row["recall"], row["run_id"]),
    )

    return {
        "normal_run_count": len(normal),
        "attack_run_count": len(attack),
        "normal_run_macro_fpr": float(
            np.mean([row["fpr"] for row in normal])
        ),
        "worst_normal_run_fpr": float(worst_normal["fpr"]),
        "worst_normal_run_id": worst_normal["run_id"],
        "attack_run_macro_recall": float(
            np.mean([row["recall"] for row in attack])
        ),
        "worst_attack_run_recall": float(worst_attack["recall"]),
        "worst_attack_run_id": worst_attack["run_id"],
        "strength20_run_count": len(strength20),
        "strength20_macro_recall": float(
            np.mean([row["recall"] for row in strength20])
        ),
    }


def grouped_graph_rows(
    label: str,
    data: dict[str, np.ndarray],
    threshold: float,
) -> list[dict[str, Any]]:
    y_graph = np.asarray(data["y_graph"])
    dimensions = {
        "profile": np.asarray(
            [text(value).lower() for value in data["profile"]]
        ),
        "strength": np.asarray(
            [canonical_strength(value) for value in data["strength"]]
        ),
        "active_cores": np.asarray(
            [canonical_sequence(value) for value in data["active_cores"]]
        ),
        "attacker_placement": np.asarray(
            [canonical_sequence(value) for value in data["attackers"]]
        ),
        "seed": np.asarray([text(value) for value in data["seed"]]),
    }

    rows: list[dict[str, Any]] = []
    for dimension, values in dimensions.items():
        for group in sorted(set(values.tolist())):
            mask = values == group
            rows.append(
                {
                    "model": label,
                    "dimension": dimension,
                    "group": group,
                    "threshold": threshold,
                    "sample_count": int(np.sum(mask)),
                    **binary_metrics(
                        y_graph[mask],
                        data["graph_prob"][mask],
                        threshold,
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def npz_to_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def evaluate_model(
    label: str,
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    graph_sweep = [
        binary_metrics(
            data["y_graph"],
            data["graph_prob"],
            float(threshold),
        )
        for threshold in THRESHOLDS
    ]
    node_sweep = [
        node_metrics(
            data["y_graph"],
            data["y_node"],
            data["node_prob"],
            float(threshold),
        )
        for threshold in THRESHOLDS
    ]

    selected_graph = choose_graph(graph_sweep)
    selected_node = choose_node(node_sweep)
    ranking = ranking_metrics(
        data["y_graph"],
        data["y_node"],
        data["node_prob"],
    )
    runs = run_rows(label, data, selected_graph["threshold"])
    run_summary = summarize_runs(runs)
    grouped = grouped_graph_rows(
        label,
        data,
        selected_graph["threshold"],
    )

    lower = label.lower()
    write_csv(
        output_dir / f"{lower}_graph_threshold_sweep.csv",
        graph_sweep,
    )
    write_csv(
        output_dir / f"{lower}_node_threshold_sweep.csv",
        node_sweep,
    )
    write_csv(
        output_dir / f"{lower}_validation_per_run.csv",
        runs,
    )
    write_csv(
        output_dir / f"{lower}_validation_grouped_graph.csv",
        grouped,
    )

    return {
        "selected_graph_threshold": selected_graph["threshold"],
        "graph": selected_graph,
        "selected_node_threshold": selected_node["threshold"],
        "node_attack_only": selected_node,
        "ranking_attack_only": ranking,
        "run_summary": run_summary,
    }


def delta(c1: float, a1: float) -> float:
    return float(c1 - a1)


def build_gate(
    a1: dict[str, Any],
    c1: dict[str, Any],
) -> dict[str, Any]:
    deltas = {
        "graph_f1": delta(c1["graph"]["f1"], a1["graph"]["f1"]),
        "graph_recall": delta(
            c1["graph"]["recall"],
            a1["graph"]["recall"],
        ),
        "graph_fpr": delta(c1["graph"]["fpr"], a1["graph"]["fpr"]),
        "normal_run_macro_fpr": delta(
            c1["run_summary"]["normal_run_macro_fpr"],
            a1["run_summary"]["normal_run_macro_fpr"],
        ),
        "worst_normal_run_fpr": delta(
            c1["run_summary"]["worst_normal_run_fpr"],
            a1["run_summary"]["worst_normal_run_fpr"],
        ),
        "attack_run_macro_recall": delta(
            c1["run_summary"]["attack_run_macro_recall"],
            a1["run_summary"]["attack_run_macro_recall"],
        ),
        "strength20_macro_recall": delta(
            c1["run_summary"]["strength20_macro_recall"],
            a1["run_summary"]["strength20_macro_recall"],
        ),
        "worst_attack_run_recall": delta(
            c1["run_summary"]["worst_attack_run_recall"],
            a1["run_summary"]["worst_attack_run_recall"],
        ),
        "attack_only_node_f1": delta(
            c1["node_attack_only"]["micro_f1"],
            a1["node_attack_only"]["micro_f1"],
        ),
        "attack_only_exact_localization": delta(
            c1["node_attack_only"]["exact_localization"],
            a1["node_attack_only"]["exact_localization"],
        ),
        "top1_hit_rate": delta(
            c1["ranking_attack_only"]["top1_hit_rate"],
            a1["ranking_attack_only"]["top1_hit_rate"],
        ),
        "top3_hit_rate": delta(
            c1["ranking_attack_only"]["top3_hit_rate"],
            a1["ranking_attack_only"]["top3_hit_rate"],
        ),
        "mean_reciprocal_rank": delta(
            c1["ranking_attack_only"]["mean_reciprocal_rank"],
            a1["ranking_attack_only"]["mean_reciprocal_rank"],
        ),
    }

    primary = {
        "graph_f1_improvement_at_least_0.010": (
            deltas["graph_f1"] >= 0.010
        ),
        "strength20_macro_recall_improvement_at_least_0.030": (
            deltas["strength20_macro_recall"] >= 0.030
        ),
        "worst_attack_run_recall_improvement_at_least_0.050": (
            deltas["worst_attack_run_recall"] >= 0.050
        ),
    }
    safeguards = {
        "normal_run_macro_fpr_not_worse_than_plus_0.020": (
            deltas["normal_run_macro_fpr"] <= 0.020
        ),
        "worst_normal_run_fpr_not_worse_than_plus_0.050": (
            deltas["worst_normal_run_fpr"] <= 0.050
        ),
        "overall_graph_recall_drop_no_worse_than_0.020": (
            deltas["graph_recall"] >= -0.020
        ),
        "attack_run_macro_recall_drop_no_worse_than_0.020": (
            deltas["attack_run_macro_recall"] >= -0.020
        ),
        "attack_only_node_f1_drop_no_worse_than_0.020": (
            deltas["attack_only_node_f1"] >= -0.020
        ),
        "exact_localization_drop_no_worse_than_0.030": (
            deltas["attack_only_exact_localization"] >= -0.030
        ),
        "top1_drop_no_worse_than_0.020": (
            deltas["top1_hit_rate"] >= -0.020
        ),
        "top3_drop_no_worse_than_0.010": (
            deltas["top3_hit_rate"] >= -0.010
        ),
    }

    primary_pass = any(primary.values())
    safeguards_pass = all(safeguards.values())
    advance = primary_pass and safeguards_pass

    return {
        "delta_definition": "C1 minus Chrono-A1",
        "deltas": deltas,
        "primary_improvements": primary,
        "primary_any_pass": primary_pass,
        "mandatory_safeguards": safeguards,
        "mandatory_all_pass": safeguards_pass,
        "advance_to_blind_test": advance,
        "formal_validation_verdict": (
            "PROMISING_ADVANCE_TO_BLIND_TEST"
            if advance
            else "REJECT_WITHOUT_FORMAL_TEST_TRANSFER"
        ),
    }


def markdown_report(
    a1: dict[str, Any],
    c1: dict[str, Any],
    gate: dict[str, Any],
) -> str:
    lines = [
        "# C1 Validation-Only Advancement Gate",
        "",
        f"**Verdict:** `{gate['formal_validation_verdict']}`",
        "",
        "No test inference or test threshold selection was performed.",
        "",
        "## Frozen validation thresholds",
        "",
        "| Model | Graph threshold | Node threshold |",
        "|---|---:|---:|",
        (
            f"| A1 | {a1['selected_graph_threshold']:.3f} | "
            f"{a1['selected_node_threshold']:.3f} |"
        ),
        (
            f"| C1 | {c1['selected_graph_threshold']:.3f} | "
            f"{c1['selected_node_threshold']:.3f} |"
        ),
        "",
        "## Key metrics",
        "",
        "| Metric | A1 | C1 | C1−A1 |",
        "|---|---:|---:|---:|",
    ]

    metric_rows = [
        ("graph_f1", a1["graph"]["f1"], c1["graph"]["f1"]),
        ("graph_recall", a1["graph"]["recall"], c1["graph"]["recall"]),
        ("graph_fpr", a1["graph"]["fpr"], c1["graph"]["fpr"]),
        (
            "normal_run_macro_fpr",
            a1["run_summary"]["normal_run_macro_fpr"],
            c1["run_summary"]["normal_run_macro_fpr"],
        ),
        (
            "worst_normal_run_fpr",
            a1["run_summary"]["worst_normal_run_fpr"],
            c1["run_summary"]["worst_normal_run_fpr"],
        ),
        (
            "attack_run_macro_recall",
            a1["run_summary"]["attack_run_macro_recall"],
            c1["run_summary"]["attack_run_macro_recall"],
        ),
        (
            "strength20_macro_recall",
            a1["run_summary"]["strength20_macro_recall"],
            c1["run_summary"]["strength20_macro_recall"],
        ),
        (
            "worst_attack_run_recall",
            a1["run_summary"]["worst_attack_run_recall"],
            c1["run_summary"]["worst_attack_run_recall"],
        ),
        (
            "attack_only_node_f1",
            a1["node_attack_only"]["micro_f1"],
            c1["node_attack_only"]["micro_f1"],
        ),
        (
            "exact_localization",
            a1["node_attack_only"]["exact_localization"],
            c1["node_attack_only"]["exact_localization"],
        ),
        (
            "top1_hit_rate",
            a1["ranking_attack_only"]["top1_hit_rate"],
            c1["ranking_attack_only"]["top1_hit_rate"],
        ),
        (
            "top3_hit_rate",
            a1["ranking_attack_only"]["top3_hit_rate"],
            c1["ranking_attack_only"]["top3_hit_rate"],
        ),
    ]
    for name, a1_value, c1_value in metric_rows:
        lines.append(
            f"| {name} | {a1_value:.6f} | {c1_value:.6f} | "
            f"{c1_value - a1_value:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## Primary improvements",
            "",
        ]
    )
    for name, passed in gate["primary_improvements"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")

    lines.extend(
        [
            "",
            "## Mandatory safeguards",
            "",
        ]
    )
    for name, passed in gate["mandatory_safeguards"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")

    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "Proceed to frozen-threshold blind-test transfer."
                if gate["advance_to_blind_test"]
                else (
                    "Reject C1 without formal blind-test transfer. "
                    "Chrono-A1 remains the final V3 architecture."
                )
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--a1-source", required=True, type=Path)
    parser.add_argument("--c1-source", required=True, type=Path)
    parser.add_argument("--a1-model-dir", required=True, type=Path)
    parser.add_argument("--c1-model-dir", required=True, type=Path)
    parser.add_argument("--c1-spec", required=True, type=Path)
    parser.add_argument("--c1-integrity-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    a1_source = args.a1_source.resolve()
    c1_source = args.c1_source.resolve()
    a1_model_dir = args.a1_model_dir.resolve()
    c1_model_dir = args.c1_model_dir.resolve()
    spec_path = args.c1_spec.resolve()
    integrity_path = args.c1_integrity_report.resolve()
    output_dir = args.output_dir.resolve()

    required_paths = {
        "data_dir": data_dir,
        "a1_source": a1_source,
        "c1_source": c1_source,
        "a1_checkpoint": a1_model_dir / "best_model.pt",
        "c1_checkpoint": c1_model_dir / "best_model.pt",
        "a1_splits": a1_model_dir / "splits.npz",
        "c1_splits": c1_model_dir / "splits.npz",
        "c1_spec": spec_path,
        "c1_integrity_report": integrity_path,
    }
    for label, path in required_paths.items():
        if label == "data_dir":
            if not path.is_dir():
                raise SystemExit(f"STOP: missing {label}: {path}")
        elif not path.is_file():
            raise SystemExit(f"STOP: missing {label}: {path}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: C1.4 output directory is already non-empty: {output_dir}"
        )
    if args.batch_size <= 0:
        raise SystemExit("STOP: batch size must be positive.")

    specification = load_json(spec_path)
    integrity = load_json(integrity_path)

    prerequisite_checks = {
        "spec_stage_is_c1_0": specification.get("stage") == "C1.0",
        "spec_status_frozen": (
            specification.get("status") == "SPECIFICATION_FROZEN"
        ),
        "integrity_stage_is_c1_3": integrity.get("stage") == "C1.3",
        "integrity_training_completed": (
            integrity.get("status") == "TRAINING_COMPLETED_AND_VERIFIED"
        ),
        "integrity_return_code_zero": integrity.get("return_code") == 0,
        "integrity_splits_identical": all(
            integrity.get("split_checks", {}).values()
        ),
        "integrity_arguments_match": all(
            integrity.get("checkpoint_argument_checks", {}).values()
        ),
        "integrity_architecture_match": all(
            integrity.get("architecture_checks", {}).values()
        ),
        "formal_gate_not_previously_evaluated": (
            integrity.get("formal_validation_gate_evaluated") is False
        ),
        "formal_test_transfer_not_performed": (
            integrity.get("formal_test_transfer_performed") is False
        ),
        "a1_source_hash_frozen": (
            sha256(a1_source) == EXPECTED_A1_SOURCE_SHA256
        ),
        "a1_checkpoint_hash_frozen": (
            sha256(a1_model_dir / "best_model.pt")
            == EXPECTED_A1_CHECKPOINT_SHA256
        ),
        "c1_checkpoint_hash_frozen": (
            sha256(c1_model_dir / "best_model.pt")
            == EXPECTED_C1_CHECKPOINT_SHA256
        ),
        "c1_source_hash_matches_integrity": (
            sha256(c1_source)
            == integrity["source"]["sha256"]
        ),
    }
    if not all(prerequisite_checks.values()):
        failed = [
            name for name, passed in prerequisite_checks.items()
            if not passed
        ]
        raise SystemExit(f"STOP: C1.4 prerequisite checks failed: {failed}")

    with np.load(a1_model_dir / "splits.npz") as a1_splits:
        a1_val_idx = np.asarray(a1_splits["val_idx"], dtype=np.int64)
    with np.load(c1_model_dir / "splits.npz") as c1_splits:
        c1_val_idx = np.asarray(c1_splits["val_idx"], dtype=np.int64)

    if not np.array_equal(a1_val_idx, c1_val_idx):
        raise SystemExit("STOP: A1/C1 validation indices differ.")

    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    a1_npz = output_dir / "a1_validation_predictions.npz"
    c1_npz = output_dir / "c1_validation_predictions.npz"

    a1_manifest = export_validation_predictions(
        label="A1",
        source_path=a1_source,
        model_dir=a1_model_dir,
        data_dir=data_dir,
        expected_val_idx=a1_val_idx,
        output_path=a1_npz,
        batch_size=args.batch_size,
        device=device,
    )
    c1_manifest = export_validation_predictions(
        label="C1",
        source_path=c1_source,
        model_dir=c1_model_dir,
        data_dir=data_dir,
        expected_val_idx=a1_val_idx,
        output_path=c1_npz,
        batch_size=args.batch_size,
        device=device,
    )

    a1_data = npz_to_dict(a1_npz)
    c1_data = npz_to_dict(c1_npz)

    alignment_checks = {
        key: bool(np.array_equal(a1_data[key], c1_data[key]))
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
        failed = [
            key for key, passed in alignment_checks.items()
            if not passed
        ]
        raise SystemExit(f"STOP: A1/C1 validation alignment failed: {failed}")

    if not np.all(a1_data["dataset_split"].astype(str) == "val"):
        raise SystemExit("STOP: A1 export contains non-validation rows.")
    if not np.all(c1_data["dataset_split"].astype(str) == "val"):
        raise SystemExit("STOP: C1 export contains non-validation rows.")

    a1_result = evaluate_model("A1", a1_data, output_dir)
    c1_result = evaluate_model("C1", c1_data, output_dir)
    gate = build_gate(a1_result, c1_result)

    report = {
        "stage": "C1.4",
        "status": "VALIDATION_GATE_COMPLETED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "data_scope": "validation only",
            "test_inference_performed": False,
            "test_threshold_selection_performed": False,
            "threshold_grid": {
                "minimum": 0.005,
                "maximum": 0.995,
                "step": 0.005,
                "count": int(len(THRESHOLDS)),
            },
            "graph_threshold_rule": [
                "maximize graph F1",
                "minimize FPR",
                "maximize recall",
                "threshold closest to 0.5",
                "lower threshold",
            ],
            "node_threshold_rule": [
                "attack-positive validation samples only",
                "maximize micro node F1",
                "maximize exact localization",
                "maximize precision",
                "maximize recall",
                "threshold closest to 0.5",
                "lower threshold",
            ],
        },
        "prerequisite_checks": prerequisite_checks,
        "alignment_checks": alignment_checks,
        "a1_manifest": a1_manifest,
        "c1_manifest": c1_manifest,
        "a1": a1_result,
        "c1": c1_result,
        "gate": gate,
        "frozen_thresholds_if_advanced": {
            "a1_graph": a1_result["selected_graph_threshold"],
            "a1_node": a1_result["selected_node_threshold"],
            "c1_graph": c1_result["selected_graph_threshold"],
            "c1_node": c1_result["selected_node_threshold"],
        },
    }

    report_path = output_dir / "c1_validation_gate.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    markdown_path = output_dir / "c1_validation_gate.md"
    markdown_path.write_text(
        markdown_report(a1_result, c1_result, gate),
        encoding="utf-8",
    )

    gate_rows = []
    for name, passed in gate["primary_improvements"].items():
        gate_rows.append(
            {
                "category": "primary",
                "criterion": name,
                "passed": passed,
            }
        )
    for name, passed in gate["mandatory_safeguards"].items():
        gate_rows.append(
            {
                "category": "safeguard",
                "criterion": name,
                "passed": passed,
            }
        )
    write_csv(output_dir / "c1_validation_gate_table.csv", gate_rows)

    print("C1.4 VALIDATION-ONLY ADVANCEMENT GATE: PASS")
    print(f"validation_rows={a1_manifest['rows']}")
    print(f"node_count={a1_manifest['nodes']}")
    print(f"alignment_pass={all(alignment_checks.values())}")
    print("test_inference_performed=False")
    print("test_threshold_selection_performed=False")

    for label, result in (("A1", a1_result), ("C1", c1_result)):
        print(f"\n{label}")
        print(
            f"graph_threshold={result['selected_graph_threshold']:.3f}"
        )
        print(f"graph_f1={result['graph']['f1']:.6f}")
        print(f"graph_recall={result['graph']['recall']:.6f}")
        print(f"graph_fpr={result['graph']['fpr']:.6f}")
        print(
            "normal_run_macro_fpr="
            f"{result['run_summary']['normal_run_macro_fpr']:.6f}"
        )
        print(
            "worst_normal_run_fpr="
            f"{result['run_summary']['worst_normal_run_fpr']:.6f}"
        )
        print(
            "worst_normal_run_id="
            f"{result['run_summary']['worst_normal_run_id']}"
        )
        print(
            "attack_run_macro_recall="
            f"{result['run_summary']['attack_run_macro_recall']:.6f}"
        )
        print(
            "strength20_macro_recall="
            f"{result['run_summary']['strength20_macro_recall']:.6f}"
        )
        print(
            "worst_attack_run_recall="
            f"{result['run_summary']['worst_attack_run_recall']:.6f}"
        )
        print(
            "worst_attack_run_id="
            f"{result['run_summary']['worst_attack_run_id']}"
        )
        print(
            f"node_threshold={result['selected_node_threshold']:.3f}"
        )
        print(
            "attack_only_node_f1="
            f"{result['node_attack_only']['micro_f1']:.6f}"
        )
        print(
            "attack_only_exact_localization="
            f"{result['node_attack_only']['exact_localization']:.6f}"
        )
        print(
            "top1_hit_rate="
            f"{result['ranking_attack_only']['top1_hit_rate']:.6f}"
        )
        print(
            "top3_hit_rate="
            f"{result['ranking_attack_only']['top3_hit_rate']:.6f}"
        )
        print(
            "mean_reciprocal_rank="
            f"{result['ranking_attack_only']['mean_reciprocal_rank']:.6f}"
        )

    print("\nVALIDATION DELTAS (C1 - A1)")
    for key, value in gate["deltas"].items():
        print(f"{key}={value:+.6f}")

    print("\nPRIMARY IMPROVEMENTS")
    for key, passed in gate["primary_improvements"].items():
        print(f"{'PASS' if passed else 'FAIL'} {key}")

    print("\nMANDATORY SAFEGUARDS")
    for key, passed in gate["mandatory_safeguards"].items():
        print(f"{'PASS' if passed else 'FAIL'} {key}")

    print("\nFORMAL VALIDATION DECISION")
    print(f"primary_any_pass={gate['primary_any_pass']}")
    print(f"mandatory_all_pass={gate['mandatory_all_pass']}")
    print(f"advance_to_blind_test={gate['advance_to_blind_test']}")
    print(f"verdict={gate['formal_validation_verdict']}")
    print(f"json_report={report_path}")
    print(f"markdown_report={markdown_path}")


if __name__ == "__main__":
    main()
