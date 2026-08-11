#!/usr/bin/env python3
"""No-training preflight for the Stage 9 Chrono-A1 Conv1D-GCN control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


EXPECTED_SOURCE_SHA256 = "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b"
EXPECTED_A1_SPLITS_SHA256 = "4a58d9fa6667de6d15c5d6563db0e7a0ec306382cd9816900f7098c80bbee8a2"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_array(root: Path, name: str):
    path = root / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError:
        return np.load(path, allow_pickle=True)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Could not identify a model state_dict in the checkpoint.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--a1-checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    data = args.data.expanduser().resolve()
    source = args.source.expanduser().resolve()
    checkpoint_path = args.a1_checkpoint.expanduser().resolve()
    report_path = args.report.expanduser().resolve()

    checks = {}
    details = {}

    checks["data_is_directory"] = data.is_dir()
    checks["source_exists"] = source.is_file()
    checks["a1_checkpoint_exists"] = checkpoint_path.is_file()

    occupied = []
    for path in (args.out_dir, args.log, args.status, report_path):
        resolved = path.expanduser().resolve()
        if resolved.exists():
            occupied.append(str(resolved))
    checks["planned_output_paths_unused"] = not occupied
    details["occupied_output_paths"] = occupied

    if checks["source_exists"]:
        source_hash = sha256_file(source)
        details["source_sha256"] = source_hash
        checks["source_matches_frozen_a1"] = source_hash == EXPECTED_SOURCE_SHA256
    else:
        checks["source_matches_frozen_a1"] = False

    required_arrays = (
        "x",
        "y_graph",
        "y_node",
        "edge_index",
        "run_id",
        "end_epoch",
        "feature_cols",
        "split",
    )

    arrays = {}
    if checks["data_is_directory"]:
        try:
            arrays = {name: load_array(data, name) for name in required_arrays}
            checks["required_arrays_present"] = True
        except Exception as exc:
            details["array_load_error"] = str(exc)
            checks["required_arrays_present"] = False
    else:
        checks["required_arrays_present"] = False

    if checks["required_arrays_present"]:
        x = arrays["x"]
        y_graph = np.asarray(arrays["y_graph"])
        y_node = np.asarray(arrays["y_node"])
        edge_index = np.asarray(arrays["edge_index"])
        split = np.asarray(arrays["split"]).astype(str)
        feature_cols = np.asarray(arrays["feature_cols"]).astype(str)

        details["x_shape"] = list(x.shape)
        details["y_graph_shape"] = list(y_graph.shape)
        details["y_node_shape"] = list(y_node.shape)
        details["edge_index_shape"] = list(edge_index.shape)
        details["feature_count"] = int(len(feature_cols))
        details["split_labels"] = sorted(np.unique(split).tolist())

        checks["sample_count_233803"] = x.shape[0] == 233803
        checks["x_shape_expected"] = tuple(x.shape) == (233803, 16, 8, 24)
        checks["y_graph_shape_expected"] = tuple(y_graph.shape) == (233803,)
        checks["y_node_shape_expected"] = tuple(y_node.shape) == (233803, 16)
        checks["edge_index_shape_expected"] = tuple(edge_index.shape) == (2, 64)
        checks["feature_count_24"] = len(feature_cols) == 24
        checks["split_labels_exact"] = set(np.unique(split)) == {"train", "val", "test"}

        split_counts = {
            name: int(np.count_nonzero(split == name))
            for name in ("train", "val", "test")
        }
        details["split_counts"] = split_counts
        checks["train_count_148185"] = split_counts["train"] == 148185
        checks["val_count_42809"] = split_counts["val"] == 42809
        checks["test_count_42809"] = split_counts["test"] == 42809
        checks["split_is_exclusive_and_complete"] = (
            sum(split_counts.values()) == len(split)
        )

        train_mask = split == "train"
        train_graph = y_graph[train_mask].astype(np.int64)
        train_node = y_node[train_mask].astype(np.int64)

        graph_pos = int(train_graph.sum())
        graph_neg = int(train_graph.size - graph_pos)
        node_pos = int(train_node.sum())
        node_neg = int(train_node.size - node_pos)

        graph_pos_weight = graph_neg / graph_pos if graph_pos else float("inf")
        node_pos_weight = node_neg / node_pos if node_pos else float("inf")

        details["class_weights_from_chrono_train"] = {
            "graph_positive_count": graph_pos,
            "graph_negative_count": graph_neg,
            "graph_pos_weight": graph_pos_weight,
            "node_positive_count": node_pos,
            "node_negative_count": node_neg,
            "node_pos_weight": node_pos_weight,
        }

    if checks["a1_checkpoint_exists"]:
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            state_dict = extract_state_dict(checkpoint)
            parameter_count = int(
                sum(tensor.numel() for tensor in state_dict.values())
            )
            details["a1_checkpoint_parameter_count"] = parameter_count
            details["a1_checkpoint_state_keys"] = sorted(state_dict)
            checks["a1_checkpoint_parameter_count_882"] = parameter_count == 882
        except Exception as exc:
            details["checkpoint_error"] = str(exc)
            checks["a1_checkpoint_parameter_count_882"] = False

    overall = bool(checks) and all(checks.values())
    payload = {
        "stage": "Stage 9 Chrono-A1 no-training preflight",
        "training_started": False,
        "gpu_transfer_performed": False,
        "forward_pass_performed": False,
        "backpropagation_performed": False,
        "optimizer_update_performed": False,
        "checkpoint_written": False,
        "checks": checks,
        "details": details,
        "overall_success": overall,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for key, value in checks.items():
        print(f"{key}: {value}")
    print(f"overall_success: {overall}")
    print(f"report: {report_path}")
    print(f"CHRONO-A1 PREFLIGHT: {'PASS' if overall else 'FAIL'}")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
