#!/usr/bin/env python3
"""
G1.3 validation-only threshold selection and advancement gate.

This stage performs inference only on the frozen chronological validation split.
It independently selects A1 and G1 graph/node thresholds on validation, compares
their run-level and localization metrics, and applies the predeclared G1 gate.

It performs:
- no training;
- no optimizer updates;
- no test inference;
- no test threshold sweep;
- no test threshold selection.
"""

from __future__ import annotations

import argparse
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


EXPECTED_A1_SOURCE_SHA256 = (
    "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b"
)
EXPECTED_A1_CHECKPOINT_SHA256 = (
    "c29cdd2303669a61454ed118ef0be4b8809a635b12c0c42806f8b6992340e0ef"
)
EXPECTED_G1_SOURCE_SHA256 = (
    "3ebe7b8d37c5104c606daf0e65b488a3f553b815b01689ce172b12b8bbbffb5e"
)
EXPECTED_G1_CHECKPOINT_SHA256 = (
    "9597892ad3e517fd87fbfaeab1d178cf7608a82cd68c4eee9d525ba1de024ae8"
)

# These values freeze the already-established A1 validation protocol.
EXPECTED_A1_SELECTION = {
    "graph_threshold": 0.410,
    "graph_f1": 0.958716,
    "graph_recall": 0.977494,
    "graph_fpr": 0.138779,
    "normal_run_macro_fpr": 0.138779,
    "worst_normal_run_fpr": 0.432736,
    "attack_run_macro_recall": 0.977494,
    "strength20_macro_recall": 0.959611,
    "worst_attack_run_recall": 0.942606,
    "node_threshold": 0.765,
    "attack_only_node_f1": 0.905798,
    "attack_only_exact_localization": 0.704963,
    "top1_hit_rate": 0.952256,
    "top3_hit_rate": 0.995107,
    "mean_reciprocal_rank": 0.973435,
}

THRESHOLDS = np.round(np.arange(0.005, 1.000, 0.005), 3)
METRIC_TOLERANCE = 5e-4


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
        raise RuntimeError(f"Could not import source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], str]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "model_state"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value:
                if all(isinstance(v, torch.Tensor) for v in value.values()):
                    return value, key

        if checkpoint and all(
            isinstance(value, torch.Tensor) for value in checkpoint.values()
        ):
            return checkpoint, "checkpoint_root"

    raise ValueError("Unable to locate a tensor state_dict in checkpoint.")


def decode_strings(values: np.ndarray) -> np.ndarray:
    decoded = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return np.asarray(decoded, dtype=str)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    fpr = safe_divide(fp, fp + tn)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def attack_only_localization_metrics(
    y_graph: np.ndarray,
    y_node: np.ndarray,
    node_prob: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    attack_mask = np.asarray(y_graph, dtype=np.int64) == 1
    true_nodes = np.asarray(y_node[attack_mask], dtype=np.int64)
    probabilities = np.asarray(node_prob[attack_mask], dtype=np.float64)
    predicted_nodes = (probabilities >= threshold).astype(np.int64)

    flat = binary_metrics(true_nodes.reshape(-1), predicted_nodes.reshape(-1))
    exact = float(np.mean(np.all(predicted_nodes == true_nodes, axis=1)))

    ranked = np.argsort(-probabilities, axis=1)
    top1_hits = []
    top3_hits = []
    reciprocal_ranks = []

    for row_index in range(len(true_nodes)):
        true_attackers = set(np.flatnonzero(true_nodes[row_index] == 1).tolist())
        order = ranked[row_index].tolist()

        top1_hits.append(int(bool(true_attackers.intersection(order[:1]))))
        top3_hits.append(int(bool(true_attackers.intersection(order[:3]))))

        best_rank = min(
            (rank + 1 for rank, node in enumerate(order) if node in true_attackers),
            default=0,
        )
        reciprocal_ranks.append(1.0 / best_rank if best_rank else 0.0)

    predicted_count = predicted_nodes.sum(axis=1)
    true_count = true_nodes.sum(axis=1)

    return {
        "threshold": float(threshold),
        "attack_samples": int(len(true_nodes)),
        "precision": float(flat["precision"]),
        "recall": float(flat["recall"]),
        "f1": float(flat["f1"]),
        "exact_localization": exact,
        "top1_hit_rate": float(np.mean(top1_hits)),
        "top3_hit_rate": float(np.mean(top3_hits)),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "empty_prediction_rate": float(np.mean(predicted_count == 0)),
        "overprediction_rate": float(np.mean(predicted_count > true_count)),
        "underprediction_rate": float(np.mean(predicted_count < true_count)),
    }


def run_metrics(
    y_graph: np.ndarray,
    graph_pred: np.ndarray,
    run_id: np.ndarray,
    strength: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for current_run in np.unique(run_id):
        mask = run_id == current_run
        labels = np.asarray(y_graph[mask], dtype=np.int64)
        predictions = np.asarray(graph_pred[mask], dtype=np.int64)

        if not np.all(labels == labels[0]):
            raise ValueError(f"Run contains mixed graph labels: {current_run}")

        label = int(labels[0])
        run_strength_values = np.asarray(strength[mask])
        run_strength = None
        nonnegative_strengths = run_strength_values[
            np.asarray(run_strength_values, dtype=np.float64) >= 0
        ]
        if len(nonnegative_strengths):
            run_strength = float(nonnegative_strengths[0])

        if label == 0:
            rate = float(np.mean(predictions == 1))
            row = {
                "run_id": str(current_run),
                "label": 0,
                "strength": run_strength,
                "fpr": rate,
                "recall": None,
            }
        else:
            rate = float(np.mean(predictions == 1))
            row = {
                "run_id": str(current_run),
                "label": 1,
                "strength": run_strength,
                "fpr": None,
                "recall": rate,
            }
        rows.append(row)

    normal_rows = [row for row in rows if row["label"] == 0]
    attack_rows = [row for row in rows if row["label"] == 1]
    strength20_rows = [
        row
        for row in attack_rows
        if row["strength"] is not None
        and math.isclose(float(row["strength"]), 20.0, abs_tol=1e-9)
    ]

    worst_normal = max(
        normal_rows,
        key=lambda row: float(row["fpr"]),
    )
    worst_attack = min(
        attack_rows,
        key=lambda row: float(row["recall"]),
    )
    worst_strength20 = min(
        strength20_rows,
        key=lambda row: float(row["recall"]),
    )

    summary = {
        "normal_run_count": len(normal_rows),
        "attack_run_count": len(attack_rows),
        "strength20_run_count": len(strength20_rows),
        "normal_run_macro_fpr": float(
            np.mean([row["fpr"] for row in normal_rows])
        ),
        "worst_normal_run_fpr": float(worst_normal["fpr"]),
        "worst_normal_run_id": worst_normal["run_id"],
        "attack_run_macro_recall": float(
            np.mean([row["recall"] for row in attack_rows])
        ),
        "worst_attack_run_recall": float(worst_attack["recall"]),
        "worst_attack_run_id": worst_attack["run_id"],
        "strength20_macro_recall": float(
            np.mean([row["recall"] for row in strength20_rows])
        ),
        "worst_strength20_run_recall": float(worst_strength20["recall"]),
        "worst_strength20_run_id": worst_strength20["run_id"],
    }
    return rows, summary


def select_graph_threshold(
    y_graph: np.ndarray,
    graph_prob: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    sweep = []
    for threshold in THRESHOLDS:
        metrics = binary_metrics(
            y_graph,
            (graph_prob >= threshold).astype(np.int64),
        )
        sweep.append({"threshold": float(threshold), **metrics})

    selected = max(
        sweep,
        key=lambda row: (
            float(row["f1"]),
            -float(row["fpr"]),
            float(row["recall"]),
            float(row["threshold"]),
        ),
    )
    return float(selected["threshold"]), sweep


def select_node_threshold(
    y_graph: np.ndarray,
    y_node: np.ndarray,
    node_prob: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    sweep = []
    for threshold in THRESHOLDS:
        metrics = attack_only_localization_metrics(
            y_graph,
            y_node,
            node_prob,
            float(threshold),
        )
        sweep.append(metrics)

    selected = max(
        sweep,
        key=lambda row: (
            float(row["f1"]),
            float(row["exact_localization"]),
            float(row["top1_hit_rate"]),
            float(row["threshold"]),
        ),
    )
    return float(selected["threshold"]), sweep


def infer_validation(
    *,
    source: Path,
    checkpoint_path: Path,
    data_dir: Path,
    validation_idx: np.ndarray,
    batch_size: int,
    device: torch.device,
    module_name: str,
) -> dict[str, Any]:
    module = load_module(source, module_name)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict, state_location = extract_state_dict(checkpoint)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

    x = np.load(data_dir / "x.npy", mmap_mode="r")
    edge_index = np.load(data_dir / "edge_index.npy").astype(np.int64)

    temporal_dim = int(saved_args.get("temporal_dim", 8))
    gcn_hidden = int(saved_args.get("gcn_hidden", 16))
    gcn_out = int(saved_args.get("gcn_out", 8))

    model = module.TemporalGCN(
        input_features=int(x.shape[-1]),
        temporal_dim=temporal_dim,
        gcn_hidden=gcn_hidden,
        gcn_out=gcn_out,
    )
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(
            f"Strict load failed for {checkpoint_path}: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )

    adjacency = module.build_normalized_adjacency(
        edge_index,
        int(x.shape[1]),
    ).to(device)

    model.to(device)
    model.eval()

    graph_prob_parts = []
    node_prob_parts = []

    with torch.no_grad():
        for start in range(0, len(validation_idx), batch_size):
            indices = validation_idx[start:start + batch_size]
            batch_x = torch.from_numpy(
                np.array(x[indices], dtype=np.float32, copy=True)
            ).to(device)
            graph_logits, node_logits = model(batch_x, adjacency)
            graph_prob_parts.append(
                torch.sigmoid(graph_logits).detach().cpu().numpy()
            )
            node_prob_parts.append(
                torch.sigmoid(node_logits).detach().cpu().numpy()
            )

    graph_prob = np.concatenate(graph_prob_parts, axis=0)
    node_prob = np.concatenate(node_prob_parts, axis=0)

    return {
        "graph_prob": graph_prob,
        "node_prob": node_prob,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "state_dict_location": state_location,
        "saved_args": saved_args,
        "adjacency_nonzero_entries": int(adjacency.sum().item()),
        "adjacency_zero_diagonal": bool(
            torch.all(torch.diag(adjacency) == 0).item()
        ),
    }


def evaluate_model(
    *,
    name: str,
    graph_prob: np.ndarray,
    node_prob: np.ndarray,
    y_graph: np.ndarray,
    y_node: np.ndarray,
    run_id: np.ndarray,
    strength: np.ndarray,
) -> dict[str, Any]:
    graph_threshold, graph_sweep = select_graph_threshold(
        y_graph,
        graph_prob,
    )
    graph_pred = (graph_prob >= graph_threshold).astype(np.int64)
    graph = binary_metrics(y_graph, graph_pred)
    run_rows, run_summary = run_metrics(
        y_graph,
        graph_pred,
        run_id,
        strength,
    )

    node_threshold, node_sweep = select_node_threshold(
        y_graph,
        y_node,
        node_prob,
    )
    node = attack_only_localization_metrics(
        y_graph,
        y_node,
        node_prob,
        node_threshold,
    )

    return {
        "name": name,
        "graph_threshold": graph_threshold,
        "node_threshold": node_threshold,
        "graph": graph,
        "node": node,
        "run_summary": run_summary,
        "run_rows": run_rows,
        "graph_sweep": graph_sweep,
        "node_sweep": node_sweep,
    }


def flatten_metrics(result: dict[str, Any]) -> dict[str, float]:
    return {
        "graph_f1": float(result["graph"]["f1"]),
        "graph_recall": float(result["graph"]["recall"]),
        "graph_fpr": float(result["graph"]["fpr"]),
        "normal_run_macro_fpr": float(
            result["run_summary"]["normal_run_macro_fpr"]
        ),
        "worst_normal_run_fpr": float(
            result["run_summary"]["worst_normal_run_fpr"]
        ),
        "attack_run_macro_recall": float(
            result["run_summary"]["attack_run_macro_recall"]
        ),
        "strength20_macro_recall": float(
            result["run_summary"]["strength20_macro_recall"]
        ),
        "worst_attack_run_recall": float(
            result["run_summary"]["worst_attack_run_recall"]
        ),
        "worst_strength20_run_recall": float(
            result["run_summary"]["worst_strength20_run_recall"]
        ),
        "attack_only_node_f1": float(result["node"]["f1"]),
        "attack_only_exact_localization": float(
            result["node"]["exact_localization"]
        ),
        "top1_hit_rate": float(result["node"]["top1_hit_rate"]),
        "top3_hit_rate": float(result["node"]["top3_hit_rate"]),
        "mean_reciprocal_rank": float(
            result["node"]["mean_reciprocal_rank"]
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_model_result(label: str, result: dict[str, Any]) -> None:
    graph = result["graph"]
    node = result["node"]
    run = result["run_summary"]

    print(label)
    print(f"graph_threshold={result['graph_threshold']:.3f}")
    print(f"graph_accuracy={graph['accuracy']:.6f}")
    print(f"graph_precision={graph['precision']:.6f}")
    print(f"graph_recall={graph['recall']:.6f}")
    print(f"graph_f1={graph['f1']:.6f}")
    print(f"graph_fpr={graph['fpr']:.6f}")
    print(f"normal_run_macro_fpr={run['normal_run_macro_fpr']:.6f}")
    print(f"worst_normal_run_fpr={run['worst_normal_run_fpr']:.6f}")
    print(f"worst_normal_run_id={run['worst_normal_run_id']}")
    print(f"attack_run_macro_recall={run['attack_run_macro_recall']:.6f}")
    print(f"strength20_macro_recall={run['strength20_macro_recall']:.6f}")
    print(
        "worst_strength20_run_recall="
        f"{run['worst_strength20_run_recall']:.6f}"
    )
    print(
        "worst_strength20_run_id="
        f"{run['worst_strength20_run_id']}"
    )
    print(f"worst_attack_run_recall={run['worst_attack_run_recall']:.6f}")
    print(f"worst_attack_run_id={run['worst_attack_run_id']}")
    print(f"node_threshold={result['node_threshold']:.3f}")
    print(f"attack_only_node_f1={node['f1']:.6f}")
    print(
        "attack_only_exact_localization="
        f"{node['exact_localization']:.6f}"
    )
    print(f"top1_hit_rate={node['top1_hit_rate']:.6f}")
    print(f"top3_hit_rate={node['top3_hit_rate']:.6f}")
    print(f"mean_reciprocal_rank={node['mean_reciprocal_rank']:.6f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--a1-source", required=True, type=Path)
    parser.add_argument("--g1-source", required=True, type=Path)
    parser.add_argument("--a1-model-dir", required=True, type=Path)
    parser.add_argument("--g1-model-dir", required=True, type=Path)
    parser.add_argument("--g1-training-integrity", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    data_dir = args.data_dir.resolve()
    a1_source = args.a1_source.resolve()
    g1_source = args.g1_source.resolve()
    a1_model_dir = args.a1_model_dir.resolve()
    g1_model_dir = args.g1_model_dir.resolve()
    training_integrity_path = args.g1_training_integrity.resolve()
    output_dir = args.output_dir.resolve()

    a1_checkpoint = a1_model_dir / "best_model.pt"
    g1_checkpoint = g1_model_dir / "best_model.pt"
    a1_splits = a1_model_dir / "splits.npz"
    g1_splits = g1_model_dir / "splits.npz"

    required = {
        "repo_root": repo_root,
        "data_dir": data_dir,
        "a1_source": a1_source,
        "g1_source": g1_source,
        "a1_checkpoint": a1_checkpoint,
        "g1_checkpoint": g1_checkpoint,
        "a1_splits": a1_splits,
        "g1_splits": g1_splits,
        "g1_training_integrity": training_integrity_path,
        "y_graph": data_dir / "y_graph.npy",
        "y_node": data_dir / "y_node.npy",
        "run_id": data_dir / "run_id.npy",
        "strength": data_dir / "strength.npy",
    }
    for label, path in required.items():
        if label in ("repo_root", "data_dir"):
            if not path.is_dir():
                raise SystemExit(f"STOP: missing {label}: {path}")
        elif not path.is_file():
            raise SystemExit(f"STOP: missing {label}: {path}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: G1.3 output directory already non-empty: {output_dir}"
        )

    integrity = load_json(training_integrity_path)
    prerequisite_checks = {
        "g1_2_stage": integrity.get("stage") == "G1.2",
        "g1_2_passed": (
            integrity.get("status") == "TRAINING_COMPLETED_AND_VERIFIED"
        ),
        "g1_2_return_code_zero": integrity.get("return_code") == 0,
        "g1_2_test_threshold_selection_false": (
            integrity.get("protocol", {})
            .get("test_threshold_selection_performed") is False
        ),
        "a1_source_hash": sha256(a1_source) == EXPECTED_A1_SOURCE_SHA256,
        "a1_checkpoint_hash": (
            sha256(a1_checkpoint) == EXPECTED_A1_CHECKPOINT_SHA256
        ),
        "g1_source_hash": sha256(g1_source) == EXPECTED_G1_SOURCE_SHA256,
        "g1_checkpoint_hash": (
            sha256(g1_checkpoint) == EXPECTED_G1_CHECKPOINT_SHA256
        ),
        "g1_checkpoint_matches_integrity": (
            sha256(g1_checkpoint) == integrity["hashes"]["checkpoint"]
        ),
        "g1_source_matches_integrity": (
            sha256(g1_source) == integrity["hashes"]["g1_source"]
        ),
    }
    if not all(prerequisite_checks.values()):
        failed = [
            name for name, passed in prerequisite_checks.items()
            if not passed
        ]
        raise SystemExit(f"STOP: G1.3 prerequisite checks failed: {failed}")

    with np.load(a1_splits) as a1_split_data, np.load(g1_splits) as g1_split_data:
        split_checks = {
            "split_keys_equal": set(a1_split_data.files) == set(g1_split_data.files)
        }
        for key in sorted(set(a1_split_data.files) | set(g1_split_data.files)):
            split_checks[f"{key}_present"] = (
                key in a1_split_data.files and key in g1_split_data.files
            )
            if key in a1_split_data.files and key in g1_split_data.files:
                split_checks[f"{key}_equal"] = np.array_equal(
                    np.asarray(a1_split_data[key]),
                    np.asarray(g1_split_data[key]),
                )

        validation_idx = np.asarray(
            a1_split_data["val_idx"],
            dtype=np.int64,
        )

    if not all(split_checks.values()):
        failed = [name for name, passed in split_checks.items() if not passed]
        raise SystemExit(f"STOP: A1/G1 split mismatch: {failed}")

    y_graph_all = np.load(data_dir / "y_graph.npy", mmap_mode="r")
    y_node_all = np.load(data_dir / "y_node.npy", mmap_mode="r")
    run_id_all = np.load(data_dir / "run_id.npy", allow_pickle=True)
    strength_all = np.load(data_dir / "strength.npy", allow_pickle=True)

    y_graph = np.asarray(y_graph_all[validation_idx], dtype=np.int64)
    y_node = np.asarray(y_node_all[validation_idx], dtype=np.int64)
    run_id = decode_strings(np.asarray(run_id_all[validation_idx]))
    strength = np.asarray(strength_all[validation_idx], dtype=np.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a1_inference = infer_validation(
        source=a1_source,
        checkpoint_path=a1_checkpoint,
        data_dir=data_dir,
        validation_idx=validation_idx,
        batch_size=args.batch_size,
        device=device,
        module_name="g1_gate_a1_model",
    )
    g1_inference = infer_validation(
        source=g1_source,
        checkpoint_path=g1_checkpoint,
        data_dir=data_dir,
        validation_idx=validation_idx,
        batch_size=args.batch_size,
        device=device,
        module_name="g1_gate_g1_model",
    )

    alignment_checks = {
        "validation_row_count_is_42809": len(validation_idx) == 42809,
        "a1_graph_rows_align": (
            len(a1_inference["graph_prob"]) == len(validation_idx)
        ),
        "g1_graph_rows_align": (
            len(g1_inference["graph_prob"]) == len(validation_idx)
        ),
        "a1_node_shape_aligns": (
            tuple(a1_inference["node_prob"].shape)
            == (len(validation_idx), 16)
        ),
        "g1_node_shape_aligns": (
            tuple(g1_inference["node_prob"].shape)
            == (len(validation_idx), 16)
        ),
        "a1_parameter_count_is_882": (
            a1_inference["parameter_count"] == 882
        ),
        "g1_parameter_count_is_1138": (
            g1_inference["parameter_count"] == 1138
        ),
        "a1_adjacency_has_64_weighted_entries": (
            a1_inference["adjacency_nonzero_entries"] == 64
        ),
        "g1_adjacency_has_48_entries": (
            g1_inference["adjacency_nonzero_entries"] == 48
        ),
        "g1_adjacency_zero_diagonal": (
            g1_inference["adjacency_zero_diagonal"] is True
        ),
    }
    if not all(alignment_checks.values()):
        failed = [
            name for name, passed in alignment_checks.items()
            if not passed
        ]
        raise SystemExit(f"STOP: G1.3 alignment checks failed: {failed}")

    a1 = evaluate_model(
        name="A1",
        graph_prob=a1_inference["graph_prob"],
        node_prob=a1_inference["node_prob"],
        y_graph=y_graph,
        y_node=y_node,
        run_id=run_id,
        strength=strength,
    )
    g1 = evaluate_model(
        name="G1",
        graph_prob=g1_inference["graph_prob"],
        node_prob=g1_inference["node_prob"],
        y_graph=y_graph,
        y_node=y_node,
        run_id=run_id,
        strength=strength,
    )

    a1_flat = flatten_metrics(a1)
    g1_flat = flatten_metrics(g1)
    deltas = {
        key: g1_flat[key] - a1_flat[key]
        for key in a1_flat
    }

    a1_protocol_checks = {
        "graph_threshold": math.isclose(
            a1["graph_threshold"],
            EXPECTED_A1_SELECTION["graph_threshold"],
            abs_tol=1e-12,
        ),
        "node_threshold": math.isclose(
            a1["node_threshold"],
            EXPECTED_A1_SELECTION["node_threshold"],
            abs_tol=1e-12,
        ),
    }
    for key, expected in EXPECTED_A1_SELECTION.items():
        if key in ("graph_threshold", "node_threshold"):
            continue
        observed = a1_flat[key]
        a1_protocol_checks[key] = math.isclose(
            observed,
            expected,
            abs_tol=METRIC_TOLERANCE,
        )

    if not all(a1_protocol_checks.values()):
        failed = [
            {
                "metric": key,
                "observed": (
                    a1.get(key)
                    if key in ("graph_threshold", "node_threshold")
                    else a1_flat.get(key)
                ),
                "expected": EXPECTED_A1_SELECTION[key],
            }
            for key, passed in a1_protocol_checks.items()
            if not passed
        ]
        raise SystemExit(
            "STOP: A1 validation-selection protocol mismatch: "
            + json.dumps(failed)
        )

    primary = {
        "exact_localization_improvement_at_least_0.030": (
            deltas["attack_only_exact_localization"] >= 0.030
        ),
        "attack_only_node_f1_improvement_at_least_0.020": (
            deltas["attack_only_node_f1"] >= 0.020
        ),
        "top1_improvement_at_least_0.020": (
            deltas["top1_hit_rate"] >= 0.020
        ),
        "graph_f1_improvement_at_least_0.010": (
            deltas["graph_f1"] >= 0.010
        ),
        "worst_strength20_recall_improvement_at_least_0.050": (
            deltas["worst_strength20_run_recall"] >= 0.050
        ),
    }

    safeguards = {
        "normal_run_macro_fpr_not_worse_than_plus_0.020": (
            deltas["normal_run_macro_fpr"] <= 0.020
        ),
        "worst_normal_run_fpr_not_worse_than_plus_0.050": (
            deltas["worst_normal_run_fpr"] <= 0.050
        ),
        "graph_recall_drop_no_worse_than_0.020": (
            deltas["graph_recall"] >= -0.020
        ),
        "attack_run_macro_recall_drop_no_worse_than_0.020": (
            deltas["attack_run_macro_recall"] >= -0.020
        ),
        "strength20_recall_drop_no_worse_than_0.030": (
            deltas["strength20_macro_recall"] >= -0.030
        ),
        "worst_attack_run_recall_drop_no_worse_than_0.050": (
            deltas["worst_attack_run_recall"] >= -0.050
        ),
        "node_f1_drop_no_worse_than_0.020": (
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
        "parameter_count_at_most_1200": (
            g1_inference["parameter_count"] <= 1200
        ),
    }

    primary_any_pass = any(primary.values())
    mandatory_all_pass = all(safeguards.values())
    advance = primary_any_pass and mandatory_all_pass

    verdict = (
        "PROMISING_ADVANCE_TO_FROZEN_TEST_TRANSFER"
        if advance
        else "VALIDATION_DOES_NOT_JUSTIFY_TEST_TRANSFER"
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    np.savez_compressed(
        output_dir / "a1_validation_predictions.npz",
        sample_idx=validation_idx,
        y_graph=y_graph,
        y_node=y_node,
        run_id=run_id,
        strength=strength,
        graph_prob=a1_inference["graph_prob"],
        node_prob=a1_inference["node_prob"],
    )
    np.savez_compressed(
        output_dir / "g1_validation_predictions.npz",
        sample_idx=validation_idx,
        y_graph=y_graph,
        y_node=y_node,
        run_id=run_id,
        strength=strength,
        graph_prob=g1_inference["graph_prob"],
        node_prob=g1_inference["node_prob"],
    )

    write_csv(output_dir / "a1_graph_threshold_sweep.csv", a1["graph_sweep"])
    write_csv(output_dir / "g1_graph_threshold_sweep.csv", g1["graph_sweep"])
    write_csv(output_dir / "a1_node_threshold_sweep.csv", a1["node_sweep"])
    write_csv(output_dir / "g1_node_threshold_sweep.csv", g1["node_sweep"])
    write_csv(output_dir / "a1_validation_per_run.csv", a1["run_rows"])
    write_csv(output_dir / "g1_validation_per_run.csv", g1["run_rows"])

    gate_rows = [
        {
            "group": "primary",
            "criterion": key,
            "passed": value,
        }
        for key, value in primary.items()
    ] + [
        {
            "group": "mandatory_safeguard",
            "criterion": key,
            "passed": value,
        }
        for key, value in safeguards.items()
    ]
    write_csv(output_dir / "g1_validation_gate.csv", gate_rows)

    report = {
        "stage": "G1.3",
        "status": "VALIDATION_GATE_COMPLETED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "validation_only": True,
            "training_performed": False,
            "test_inference_performed": False,
            "test_threshold_sweep_performed": False,
            "test_threshold_selection_performed": False,
            "threshold_grid_start": float(THRESHOLDS[0]),
            "threshold_grid_end": float(THRESHOLDS[-1]),
            "threshold_grid_step": 0.005,
            "graph_selection": (
                "maximize graph F1; tie-break lower FPR, higher recall, "
                "then higher threshold"
            ),
            "node_selection": (
                "maximize attack-only node F1; tie-break exact localization, "
                "Top-1 hit rate, then higher threshold"
            ),
        },
        "paths": {
            "data_dir": str(data_dir),
            "a1_source": str(a1_source),
            "g1_source": str(g1_source),
            "a1_checkpoint": str(a1_checkpoint),
            "g1_checkpoint": str(g1_checkpoint),
            "a1_splits": str(a1_splits),
            "g1_splits": str(g1_splits),
            "g1_training_integrity": str(training_integrity_path),
        },
        "hashes": {
            "a1_source": sha256(a1_source),
            "g1_source": sha256(g1_source),
            "a1_checkpoint": sha256(a1_checkpoint),
            "g1_checkpoint": sha256(g1_checkpoint),
            "a1_splits": sha256(a1_splits),
            "g1_splits": sha256(g1_splits),
        },
        "validation_rows": int(len(validation_idx)),
        "node_count": int(y_node.shape[1]),
        "checks": {
            **prerequisite_checks,
            **split_checks,
            **alignment_checks,
            **{
                f"a1_protocol_{key}": value
                for key, value in a1_protocol_checks.items()
            },
        },
        "a1": {
            "graph_threshold": a1["graph_threshold"],
            "node_threshold": a1["node_threshold"],
            "graph": a1["graph"],
            "node": a1["node"],
            "run_summary": a1["run_summary"],
            "parameter_count": a1_inference["parameter_count"],
        },
        "g1": {
            "graph_threshold": g1["graph_threshold"],
            "node_threshold": g1["node_threshold"],
            "graph": g1["graph"],
            "node": g1["node"],
            "run_summary": g1["run_summary"],
            "parameter_count": g1_inference["parameter_count"],
        },
        "gate": {
            "deltas": deltas,
            "primary": primary,
            "mandatory_safeguards": safeguards,
            "primary_any_pass": primary_any_pass,
            "mandatory_all_pass": mandatory_all_pass,
            "advance_to_frozen_test_transfer": advance,
            "verdict": verdict,
        },
    }

    report_path = output_dir / "g1_validation_gate.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# G1.3 Validation-Only Gate",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        f"- Primary condition passed: `{primary_any_pass}`",
        f"- All safeguards passed: `{mandatory_all_pass}`",
        f"- Advance to frozen test transfer: `{advance}`",
        "",
        "No test inference or test threshold selection was performed.",
    ]
    summary_path = output_dir / "g1_validation_gate.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("G1.3 VALIDATION-ONLY ADVANCEMENT GATE: PASS")
    print(f"validation_rows={len(validation_idx)}")
    print(f"node_count={y_node.shape[1]}")
    print("alignment_pass=True")
    print("test_inference_performed=False")
    print("test_threshold_sweep_performed=False")
    print("test_threshold_selection_performed=False")
    print()

    print_model_result("A1", a1)
    print_model_result("G1", g1)

    print("VALIDATION DELTAS (G1 - A1)")
    for key, value in deltas.items():
        print(f"{key}={value:+.6f}")
    print()

    print("PRIMARY ADVANCEMENT")
    for key, passed in primary.items():
        print(f"{'PASS' if passed else 'FAIL'} {key}")
    print()

    print("MANDATORY SAFEGUARDS")
    for key, passed in safeguards.items():
        print(f"{'PASS' if passed else 'FAIL'} {key}")
    print()

    print("FORMAL VALIDATION DECISION")
    print(f"primary_any_pass={primary_any_pass}")
    print(f"mandatory_all_pass={mandatory_all_pass}")
    print(f"advance_to_frozen_test_transfer={advance}")
    print(f"verdict={verdict}")
    print(f"json_report={report_path}")
    print(f"markdown_report={summary_path}")


if __name__ == "__main__":
    main()
