#!/usr/bin/env python3
"""Validation-only threshold selection and frozen A3 development-test transfer.

Validation mode is the only mode allowed to select graph/node thresholds. The
primary attacker-set decoder is frozen by design: argmax the five-class count
head and select top-k node scores, with no graph gating. Test mode verifies the
checkpoint and threshold-package hashes before inference and labels the split as
a reused development-comparison set, never an independent publication holdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from v4_a3_sourcepreserve import (  # noqa: E402
    A3SourcePreserveModel,
    COUNT_CLASSES,
    EXPECTED_PARAMETER_COUNT,
    EXPERIMENT_DESIGNATION,
    MODEL_NAME,
    NUM_ROUTERS,
    V4A3MemmapDataset,
    atomic_csv_dump,
    atomic_json_dump,
    binary_metrics_numpy,
    build_loader,
    build_normalized_adjacency,
    build_physical_valid_port_mask,
    load_metadata,
    load_split_indices,
    model_signature,
    node_metrics_from_prediction,
    numpy_topk_decode,
    select_graph_threshold_fpr_cap,
    sha256_file,
    threshold_values,
)

VALIDATION_PASS = "V4_A3_VALIDATION_THRESHOLD_AND_DECODER_SELECTION_PASS"
TEST_PASS = "V4_A3_FROZEN_DEVELOPMENT_TEST_TRANSFER_PASS"


def ensure_empty(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    entries = list(path.iterdir())
    if entries:
        raise RuntimeError(f"Output directory must be empty: {path}; entries={[e.name for e in entries[:20]]}")


def load_checkpoint(model_dir: Path, device: torch.device) -> tuple[A3SourcePreserveModel, dict[str, Any]]:
    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    signature = checkpoint.get("model_signature", {})
    checks = {
        "designation": signature.get("experiment_designation") == EXPERIMENT_DESIGNATION,
        "parameter_count": int(signature.get("parameter_count", -1)) == EXPECTED_PARAMETER_COUNT,
        "one_gcn": int(signature.get("gcn_count", -1)) == 1,
        "embedding_64": int(signature.get("regional_embedding_dim", -1)) == 64,
        "test_not_evaluated": checkpoint.get("test_evaluated") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Checkpoint audit failed: {checks}")
    model = A3SourcePreserveModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    actual = model_signature(model)
    if actual["parameter_count"] != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Loaded model parameter mismatch: {actual}")
    model.eval()
    return model, checkpoint


def infer(
    data_dir: Path,
    split_name: str,
    model_dir: Path,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    metadata = load_metadata(data_dir)
    indices = load_split_indices(data_dir, metadata)[split_name]
    dataset = V4A3MemmapDataset(data_dir, indices, include_metadata=True)
    loader = build_loader(
        dataset,
        args.batch_size,
        False,
        args.num_workers,
        args.pin_memory,
        args.persistent_workers,
        args.prefetch_factor,
        args.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(model_dir, device)
    edge_index = np.load(data_dir / "edge_index.npy")
    adjacency = build_normalized_adjacency(edge_index).to(device)
    mask = build_physical_valid_port_mask().to(device)
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for batch_number, batch in enumerate(loader, start=1):
            x = batch["x"].to(device, non_blocking=True)
            output = model(x, adjacency, mask)
            parts["graph_prob"].append(torch.sigmoid(output["graph_logits"]).cpu().numpy().astype(np.float32))
            parts["node_prob"].append(torch.sigmoid(output["node_logits"]).cpu().numpy().astype(np.float32))
            parts["count_prob"].append(torch.softmax(output["count_logits"], dim=1).cpu().numpy().astype(np.float32))
            for key in ("y_graph", "y_node", "attacker_count", "global_index", "run_index", "split_id"):
                parts[key].append(batch[key].cpu().numpy())
            for optional in (
                "attack_kind_id",
                "profile_id",
                "strength_id",
                "active_core_group_id",
                "seed_id",
            ):
                if optional in batch:
                    parts[optional].append(batch[optional].cpu().numpy())
            if batch_number == 1 or batch_number % 500 == 0:
                print(f"inference_batches={batch_number}")
    result = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    expected = int(len(indices))
    if any(array.shape[0] != expected for array in result.values()):
        raise RuntimeError("Inference output alignment mismatch")
    if not np.array_equal(result["global_index"], indices):
        raise RuntimeError("Inference global_index order differs from split indices")
    return result


def node_metrics_at_threshold(y_node: np.ndarray, node_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = node_prob >= threshold
    overall = node_metrics_from_prediction(y_node, pred)
    attack_mask = np.sum(y_node, axis=1) > 0
    attack = node_metrics_from_prediction(y_node[attack_mask], pred[attack_mask])
    return {
        "threshold": float(threshold),
        **overall,
        "attack_node_precision": attack["node_precision"],
        "attack_node_recall": attack["node_recall"],
        "attack_node_f1": attack["node_f1"],
        "attack_exact_localization": attack["exact_localization"],
    }


def count_topk_metrics(prediction: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], np.ndarray]:
    count_pred = np.argmax(prediction["count_prob"], axis=1).astype(np.int64)
    node_pred = numpy_topk_decode(prediction["node_prob"], count_pred)
    y_node = prediction["y_node"] >= 0.5
    y_graph = prediction["y_graph"] >= 0.5
    true_count = y_node.sum(axis=1).astype(np.int64)
    overall = node_metrics_from_prediction(y_node, node_pred)
    attack = node_metrics_from_prediction(y_node[y_graph], node_pred[y_graph])
    normal = node_metrics_from_prediction(y_node[~y_graph], node_pred[~y_graph])
    metrics = {
        "decoder": "argmax count head; count=0 empty, otherwise top-k node probability; no graph gate",
        "overall": overall,
        "attack": attack,
        "normal": normal,
        "overall_count_accuracy": float(np.mean(count_pred == true_count)),
        "attack_count_accuracy": float(np.mean(count_pred[y_graph] == true_count[y_graph])),
        "attack_count_mae": float(np.mean(np.abs(count_pred[y_graph] - true_count[y_graph]))),
        "attack_empty_prediction_fraction": float(np.mean(count_pred[y_graph] == 0)),
        "mean_true_attack_count": float(np.mean(true_count[y_graph])),
        "mean_predicted_attack_count": float(np.mean(count_pred[y_graph])),
        "predicted_count_histogram": np.bincount(count_pred, minlength=COUNT_CLASSES).tolist(),
        "true_count_histogram": np.bincount(true_count, minlength=COUNT_CLASSES).tolist(),
    }
    return metrics, node_pred


def validation_mode(args: argparse.Namespace) -> int:
    out = Path(args.out_dir).resolve()
    ensure_empty(out)
    data_dir = Path(args.data_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    prediction = infer(data_dir, "val", model_dir, args)
    np.savez_compressed(out / "validation_predictions.npz", **prediction)

    graph_rows = [
        binary_metrics_numpy(prediction["y_graph"], prediction["graph_prob"], float(threshold))
        for threshold in threshold_values(args.threshold_start, args.threshold_end, args.threshold_step)
    ]
    unconstrained = max(
        graph_rows,
        key=lambda row: (row["f1"], -row["fpr"], row["recall"], -abs(row["threshold"] - 0.5)),
    )
    constrained, _ = select_graph_threshold_fpr_cap(
        prediction["y_graph"],
        prediction["graph_prob"],
        fpr_cap=args.graph_fpr_cap,
        start=args.threshold_start,
        end=args.threshold_end,
        step=args.threshold_step,
    )
    node_rows = [
        node_metrics_at_threshold(prediction["y_node"], prediction["node_prob"], float(threshold))
        for threshold in threshold_values(args.threshold_start, args.threshold_end, args.threshold_step)
    ]
    selected_node = max(
        node_rows,
        key=lambda row: (
            row["attack_node_f1"],
            row["attack_exact_localization"],
            row["node_precision"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    for row in graph_rows:
        row["task"] = "graph"
    for row in node_rows:
        row["task"] = "node"
    atomic_csv_dump(out / "graph_threshold_sweep.csv", graph_rows)
    atomic_csv_dump(out / "node_threshold_sweep.csv", node_rows)
    count_metrics, _ = count_topk_metrics(prediction)
    checkpoint = model_dir / "best_model.pt"
    selected = {
        "experiment_designation": EXPERIMENT_DESIGNATION,
        "model_name": MODEL_NAME,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "selection_split": "validation",
        "test_accessed": False,
        "graph_threshold": float(constrained["threshold"]),
        "graph_fpr_cap": float(args.graph_fpr_cap),
        "graph_selection_rule": "max graph F1 among validation thresholds with FPR<=0.10; tie recall, lower FPR, closest 0.5",
        "unconstrained_graph_threshold": float(unconstrained["threshold"]),
        "node_threshold": float(selected_node["threshold"]),
        "node_selection_rule": "max validation attack-only node F1; tie attack exact, precision, closest 0.5",
        "primary_count_decoder": "argmax count logits and top-k node scores",
        "graph_gating": False,
        "count_decoder_selected_on_test": False,
    }
    atomic_json_dump(out / "selected_thresholds.json", selected)
    metrics = {
        "fixed_0_5": {
            "graph": binary_metrics_numpy(prediction["y_graph"], prediction["graph_prob"], 0.5),
            "node": node_metrics_at_threshold(prediction["y_node"], prediction["node_prob"], 0.5),
        },
        "unconstrained_graph": unconstrained,
        "fpr_constrained_graph": constrained,
        "selected_node_threshold": selected_node,
        "count_conditioned_topk": count_metrics,
        "sample_count": int(prediction["y_graph"].shape[0]),
        "selection": selected,
    }
    atomic_json_dump(out / "validation_metrics.json", metrics)
    provenance = {
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "common_sha256": sha256_file(COMMON_DIR / "v4_a3_sourcepreserve.py"),
        "checkpoint_sha256": sha256_file(checkpoint),
        "prediction_sha256": sha256_file(out / "validation_predictions.npz"),
        "thresholds_selected_on_validation_only": True,
        "test_accessed": False,
    }
    atomic_json_dump(out / "provenance.json", provenance)
    print(VALIDATION_PASS)
    print(json.dumps(metrics, indent=2))
    return 0


def decode_map(metadata: Mapping[str, Any], name: str) -> dict[int, str]:
    mapping = metadata.get("code_maps", {}).get(name, {})
    return {int(code): str(label) for label, code in mapping.items()}


def test_mode(args: argparse.Namespace) -> int:
    out = Path(args.out_dir).resolve()
    ensure_empty(out)
    if not args.selected_thresholds:
        raise ValueError("--selected-thresholds is required for test mode")
    thresholds_path = Path(args.selected_thresholds).resolve()
    selected = json.loads(thresholds_path.read_text(encoding="utf-8"))
    if selected.get("experiment_designation") != EXPERIMENT_DESIGNATION:
        raise RuntimeError("Threshold package designation mismatch")
    if selected.get("selection_split") != "validation" or selected.get("test_accessed") is not False:
        raise RuntimeError("Threshold package is not validation-only")
    model_dir = Path(args.model_dir).resolve()
    checkpoint = model_dir / "best_model.pt"
    actual_hash = sha256_file(checkpoint)
    if selected.get("checkpoint_sha256") != actual_hash:
        raise RuntimeError("Checkpoint changed after validation selection")
    lock_path = out / "DEVELOPMENT_TEST_TRANSFER_LOCK.json"
    atomic_json_dump(
        lock_path,
        {
            "status": "STARTED",
            "checkpoint_sha256": actual_hash,
            "threshold_package_sha256": sha256_file(thresholds_path),
            "test_role": "reused development-comparison test; not publication holdout",
        },
    )
    data_dir = Path(args.data_dir).resolve()
    prediction = infer(data_dir, "test", model_dir, args)
    np.savez_compressed(out / "test_predictions.npz", **prediction)
    graph = binary_metrics_numpy(
        prediction["y_graph"], prediction["graph_prob"], float(selected["graph_threshold"])
    )
    node_threshold = node_metrics_at_threshold(
        prediction["y_node"], prediction["node_prob"], float(selected["node_threshold"])
    )
    count_metrics, node_topk_pred = count_topk_metrics(prediction)
    metrics = {
        "designation": "development comparison test; reused after A1/A2 analysis",
        "independent_publication_holdout": False,
        "experiment_designation": EXPERIMENT_DESIGNATION,
        "model_name": MODEL_NAME,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "sample_count": int(prediction["y_graph"].shape[0]),
        "graph": graph,
        "node_threshold": node_threshold,
        "count_conditioned_topk": count_metrics,
        "checkpoint_sha256": actual_hash,
        "thresholds": selected,
        "thresholds_selected_on_test": False,
        "graph_gating": False,
    }
    atomic_json_dump(out / "test_metrics.json", metrics)

    metadata = load_metadata(data_dir)
    run_rows: list[dict[str, Any]] = []
    for run in sorted(np.unique(prediction["run_index"]).tolist()):
        sample_mask = prediction["run_index"] == run
        graph_row = binary_metrics_numpy(
            prediction["y_graph"][sample_mask],
            prediction["graph_prob"][sample_mask],
            float(selected["graph_threshold"]),
        )
        node_row = node_metrics_from_prediction(
            prediction["y_node"][sample_mask] >= 0.5,
            node_topk_pred[sample_mask],
        )
        record = metadata.get("runs", [])[int(run)] if int(run) < len(metadata.get("runs", [])) else {}
        run_rows.append(
            {
                "run_index": int(run),
                "run_id": record.get("run_id", ""),
                "profile": record.get("profile", ""),
                "strength": record.get("strength", ""),
                "attacker_count": record.get("attacker_count", ""),
                "sample_count": int(sample_mask.sum()),
                **{f"graph_{key}": value for key, value in graph_row.items() if key != "threshold"},
                **node_row,
            }
        )
    atomic_csv_dump(out / "test_per_run_metrics.csv", run_rows)

    router_rows: list[dict[str, Any]] = []
    for router in range(NUM_ROUTERS):
        router_metric = node_metrics_from_prediction(
            (prediction["y_node"][:, [router]] >= 0.5),
            node_topk_pred[:, [router]],
        )
        router_rows.append({"router": router, **router_metric})
    atomic_csv_dump(out / "test_per_router_topk.csv", router_rows)

    atomic_json_dump(
        lock_path,
        {
            "status": "COMPLETED",
            "checkpoint_sha256": actual_hash,
            "threshold_package_sha256": sha256_file(thresholds_path),
            "test_predictions_sha256": sha256_file(out / "test_predictions.npz"),
            "test_metrics_sha256": sha256_file(out / "test_metrics.json"),
            "test_role": "reused development-comparison test; not publication holdout",
        },
    )
    provenance = {
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "common_sha256": sha256_file(COMMON_DIR / "v4_a3_sourcepreserve.py"),
        "checkpoint_sha256": actual_hash,
        "selected_thresholds_sha256": sha256_file(thresholds_path),
        "test_predictions_sha256": sha256_file(out / "test_predictions.npz"),
        "test_used_for_selection": False,
    }
    atomic_json_dump(out / "provenance.json", provenance)
    print(TEST_PASS)
    print(json.dumps(metrics, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--graph-fpr-cap", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid loader settings")
    if args.mode == "validation":
        return validation_mode(args)
    return test_mode(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
