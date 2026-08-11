#!/usr/bin/env python3
"""
H1A-D0: diagnose disagreement between saved G1.3 A1 validation probabilities
and fresh Chrono-A1 inference.

Runs validation-only inference on CPU and, when available, CUDA using the exact
frozen source/checkpoint and batch size. It performs no training and no test
access. It does not create golden vectors.

The purpose is to decide which inference path should be the canonical RTL FP32
reference and whether the saved G1.3 predictions are aligned with the current
checkpoint/source.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPECTED_SOURCE_SHA256 = (
    "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "c29cdd2303669a61454ed118ef0be4b8809a635b12c0c42806f8b6992340e0ef"
)
EXPECTED_GRAPH_THRESHOLD = 0.410
EXPECTED_NODE_THRESHOLD = 0.765
EXPECTED_VAL_ROWS = 42809


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


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "model_state"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value and all(
                isinstance(v, torch.Tensor) for v in value.values()
            ):
                return value
        if checkpoint and all(
            isinstance(v, torch.Tensor) for v in checkpoint.values()
        ):
            return checkpoint
    raise RuntimeError("No tensor state_dict found in checkpoint.")


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

    def div(a: float, b: float) -> float:
        return float(a / b) if b else 0.0

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    f1 = div(2.0 * precision * recall, precision + recall)
    return {
        "accuracy": div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": div(fp, fp + tn),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def error_summary(
    fresh: np.ndarray,
    saved: np.ndarray,
) -> dict[str, float]:
    error = np.abs(
        np.asarray(fresh, dtype=np.float64)
        - np.asarray(saved, dtype=np.float64)
    ).reshape(-1)
    return {
        "mean_abs_error": float(np.mean(error)),
        "median_abs_error": float(np.median(error)),
        "p90_abs_error": float(np.quantile(error, 0.90)),
        "p99_abs_error": float(np.quantile(error, 0.99)),
        "p999_abs_error": float(np.quantile(error, 0.999)),
        "max_abs_error": float(np.max(error)),
    }


def infer(
    *,
    module,
    state_dict: dict[str, torch.Tensor],
    saved_args: dict[str, Any],
    x: np.ndarray,
    edge_index: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = module.TemporalGCN(
        input_features=int(x.shape[-1]),
        temporal_dim=int(saved_args.get("temporal_dim", 8)),
        gcn_hidden=int(saved_args.get("gcn_hidden", 16)),
        gcn_out=int(saved_args.get("gcn_out", 8)),
    )
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Strict load failed on {device}: "
            f"missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )

    model.to(device)
    model.eval()
    adjacency = module.build_normalized_adjacency(
        edge_index,
        int(x.shape[1]),
    ).to(device)

    graph_parts = []
    node_parts = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_x = torch.from_numpy(
                np.array(x[batch_idx], dtype=np.float32, copy=True)
            ).to(device)
            graph_logits, node_logits = model(batch_x, adjacency)
            graph_parts.append(
                torch.sigmoid(graph_logits).detach().cpu().numpy()
            )
            node_parts.append(
                torch.sigmoid(node_logits).detach().cpu().numpy()
            )

    return (
        np.concatenate(graph_parts, axis=0).astype(np.float32),
        np.concatenate(node_parts, axis=0).astype(np.float32),
    )


def compare_path(
    *,
    name: str,
    graph_prob: np.ndarray,
    node_prob: np.ndarray,
    saved_graph: np.ndarray,
    saved_node: np.ndarray,
    y_graph: np.ndarray,
    graph_threshold: float,
    node_threshold: float,
) -> dict[str, Any]:
    graph_decision = (graph_prob >= graph_threshold).astype(np.uint8)
    saved_graph_decision = (
        saved_graph >= graph_threshold
    ).astype(np.uint8)

    node_decision = (node_prob >= node_threshold).astype(np.uint8)
    saved_node_decision = (
        saved_node >= node_threshold
    ).astype(np.uint8)

    top1 = np.argmax(node_prob, axis=1)
    saved_top1 = np.argmax(saved_node, axis=1)

    return {
        "name": name,
        "graph_error": error_summary(graph_prob, saved_graph),
        "node_error": error_summary(node_prob, saved_node),
        "graph_decision_mismatch_count": int(
            np.sum(graph_decision != saved_graph_decision)
        ),
        "node_bit_mismatch_count": int(
            np.sum(node_decision != saved_node_decision)
        ),
        "sample_with_any_node_decision_mismatch": int(
            np.sum(np.any(node_decision != saved_node_decision, axis=1))
        ),
        "top1_mismatch_count": int(np.sum(top1 != saved_top1)),
        "fresh_graph_metrics": binary_metrics(
            y_graph,
            graph_decision,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--threshold-report", required=True, type=Path)
    parser.add_argument(
        "--validation-predictions",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    source = args.source.resolve()
    model_dir = args.model_dir.resolve()
    data_dir = args.data_dir.resolve()
    threshold_path = args.threshold_report.resolve()
    prediction_path = args.validation_predictions.resolve()
    output_dir = args.output_dir.resolve()

    checkpoint_path = model_dir / "best_model.pt"
    splits_path = model_dir / "splits.npz"
    x_path = data_dir / "x.npy"
    edge_path = data_dir / "edge_index.npy"

    for path in (
        source,
        checkpoint_path,
        splits_path,
        x_path,
        edge_path,
        threshold_path,
        prediction_path,
    ):
        if not path.is_file():
            raise SystemExit(f"STOP: missing prerequisite: {path}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: diagnostic output directory is non-empty: {output_dir}"
        )

    if sha256(source) != EXPECTED_SOURCE_SHA256:
        raise SystemExit("STOP: frozen source hash mismatch")
    if sha256(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("STOP: frozen checkpoint hash mismatch")

    threshold_report = load_json(threshold_path)
    graph_threshold = float(
        threshold_report["a1"]["graph_threshold"]
    )
    node_threshold = float(
        threshold_report["a1"]["node_threshold"]
    )
    if graph_threshold != EXPECTED_GRAPH_THRESHOLD:
        raise SystemExit(
            f"STOP: unexpected graph threshold {graph_threshold}"
        )
    if node_threshold != EXPECTED_NODE_THRESHOLD:
        raise SystemExit(
            f"STOP: unexpected node threshold {node_threshold}"
        )

    with np.load(splits_path) as split_npz:
        val_idx = np.asarray(split_npz["val_idx"], dtype=np.int64)

    with np.load(prediction_path, allow_pickle=True) as pred_npz:
        sample_idx = np.asarray(
            pred_npz["sample_idx"],
            dtype=np.int64,
        )
        y_graph = np.asarray(
            pred_npz["y_graph"],
            dtype=np.int64,
        )
        y_node = np.asarray(
            pred_npz["y_node"],
            dtype=np.int64,
        )
        saved_graph = np.asarray(
            pred_npz["graph_prob"],
            dtype=np.float32,
        )
        saved_node = np.asarray(
            pred_npz["node_prob"],
            dtype=np.float32,
        )

    alignment_checks = {
        "validation_rows_42809": len(val_idx) == EXPECTED_VAL_ROWS,
        "saved_rows_42809": len(sample_idx) == EXPECTED_VAL_ROWS,
        "sample_idx_matches_val_idx": np.array_equal(
            sample_idx,
            val_idx,
        ),
        "y_graph_shape": tuple(y_graph.shape) == (EXPECTED_VAL_ROWS,),
        "y_node_shape": tuple(y_node.shape) == (EXPECTED_VAL_ROWS, 16),
        "saved_graph_shape": (
            tuple(saved_graph.shape) == (EXPECTED_VAL_ROWS,)
        ),
        "saved_node_shape": (
            tuple(saved_node.shape) == (EXPECTED_VAL_ROWS, 16)
        ),
        "all_saved_probabilities_finite": bool(
            np.isfinite(saved_graph).all()
            and np.isfinite(saved_node).all()
        ),
    }
    if not all(alignment_checks.values()):
        failed = [
            key for key, value in alignment_checks.items()
            if not value
        ]
        raise SystemExit(f"STOP: alignment failed: {failed}")

    module = load_module(source, "h1a_d0_model")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = extract_state_dict(checkpoint)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

    x = np.load(x_path, mmap_mode="r")
    edge_index = np.load(edge_path).astype(np.int64)

    print("H1A-D0 VALIDATION INFERENCE DIAGNOSTIC")
    print(f"validation_rows={len(val_idx)}")
    print(f"batch_size={args.batch_size}")
    print("test_accessed=False")
    print("running_cpu_validation_inference=True")

    cpu_graph, cpu_node = infer(
        module=module,
        state_dict=state_dict,
        saved_args=saved_args,
        x=x,
        edge_index=edge_index,
        indices=val_idx,
        device=torch.device("cpu"),
        batch_size=args.batch_size,
    )
    cpu_compare = compare_path(
        name="CPU_FP32",
        graph_prob=cpu_graph,
        node_prob=cpu_node,
        saved_graph=saved_graph,
        saved_node=saved_node,
        y_graph=y_graph,
        graph_threshold=graph_threshold,
        node_threshold=node_threshold,
    )

    cuda_compare = None
    if torch.cuda.is_available():
        print("running_cuda_validation_inference=True")
        cuda_graph, cuda_node = infer(
            module=module,
            state_dict=state_dict,
            saved_args=saved_args,
            x=x,
            edge_index=edge_index,
            indices=val_idx,
            device=torch.device("cuda"),
            batch_size=args.batch_size,
        )
        cuda_compare = compare_path(
            name="CUDA_FP32",
            graph_prob=cuda_graph,
            node_prob=cuda_node,
            saved_graph=saved_graph,
            saved_node=saved_node,
            y_graph=y_graph,
            graph_threshold=graph_threshold,
            node_threshold=node_threshold,
        )
        cpu_cuda = {
            "graph_error": error_summary(cpu_graph, cuda_graph),
            "node_error": error_summary(cpu_node, cuda_node),
            "graph_decision_mismatch_count": int(
                np.sum(
                    (cpu_graph >= graph_threshold)
                    != (cuda_graph >= graph_threshold)
                )
            ),
            "node_bit_mismatch_count": int(
                np.sum(
                    (cpu_node >= node_threshold)
                    != (cuda_node >= node_threshold)
                )
            ),
            "top1_mismatch_count": int(
                np.sum(
                    np.argmax(cpu_node, axis=1)
                    != np.argmax(cuda_node, axis=1)
                )
            ),
        }
    else:
        print("running_cuda_validation_inference=False")
        cpu_cuda = None

    saved_metrics = binary_metrics(
        y_graph,
        (saved_graph >= graph_threshold).astype(np.uint8),
    )
    expected_metrics = threshold_report["a1"]["graph"]

    expected_metric_checks = {
        metric: abs(
            float(saved_metrics[metric])
            - float(expected_metrics[metric])
        ) <= 5e-7
        for metric in ("accuracy", "precision", "recall", "f1", "fpr")
    }

    cpu_close = (
        cpu_compare["graph_error"]["max_abs_error"] <= 1e-5
        and cpu_compare["node_error"]["max_abs_error"] <= 1e-5
    )
    cuda_close = (
        cuda_compare is not None
        and cuda_compare["graph_error"]["max_abs_error"] <= 1e-5
        and cuda_compare["node_error"]["max_abs_error"] <= 1e-5
    )

    if cuda_close:
        diagnosis = "SAVED_PREDICTIONS_MATCH_FRESH_CUDA"
        recommended_canonical = "CUDA_FP32"
    elif cpu_close:
        diagnosis = "SAVED_PREDICTIONS_MATCH_FRESH_CPU"
        recommended_canonical = "CPU_FP32"
    elif all(expected_metric_checks.values()):
        diagnosis = (
            "SAVED_PREDICTIONS_ARE_INTERNALLY_VALID_BUT_DO_NOT_MATCH_"
            "FRESH_SOURCE_CHECKPOINT_INFERENCE"
        )
        recommended_canonical = "UNRESOLVED_DO_NOT_EXPORT_GOLDENS"
    else:
        diagnosis = "ALIGNMENT_OR_PREDICTION_ARTIFACT_MISMATCH"
        recommended_canonical = "UNRESOLVED_DO_NOT_EXPORT_GOLDENS"

    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "stage": "H1A-D0",
        "status": "DIAGNOSTIC_COMPLETED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(source),
        "checkpoint_sha256": sha256(checkpoint_path),
        "validation_only": True,
        "test_accessed": False,
        "batch_size": args.batch_size,
        "alignment_checks": alignment_checks,
        "thresholds": {
            "graph": graph_threshold,
            "node": node_threshold,
        },
        "saved_metrics": saved_metrics,
        "expected_report_metrics": expected_metrics,
        "saved_metric_checks": expected_metric_checks,
        "cpu_vs_saved": cpu_compare,
        "cuda_vs_saved": cuda_compare,
        "cpu_vs_cuda": cpu_cuda,
        "diagnosis": diagnosis,
        "recommended_canonical_reference": recommended_canonical,
    }
    report_path = output_dir / "h1a_prediction_mismatch_diagnostic.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("CPU VS SAVED")
    print(
        "graph_max_abs_error="
        f"{cpu_compare['graph_error']['max_abs_error']:.9e}"
    )
    print(
        "node_max_abs_error="
        f"{cpu_compare['node_error']['max_abs_error']:.9e}"
    )
    print(
        "graph_decision_mismatch_count="
        f"{cpu_compare['graph_decision_mismatch_count']}"
    )
    print(
        "node_bit_mismatch_count="
        f"{cpu_compare['node_bit_mismatch_count']}"
    )
    print(
        "top1_mismatch_count="
        f"{cpu_compare['top1_mismatch_count']}"
    )

    if cuda_compare is not None:
        print()
        print("CUDA VS SAVED")
        print(
            "graph_max_abs_error="
            f"{cuda_compare['graph_error']['max_abs_error']:.9e}"
        )
        print(
            "node_max_abs_error="
            f"{cuda_compare['node_error']['max_abs_error']:.9e}"
        )
        print(
            "graph_decision_mismatch_count="
            f"{cuda_compare['graph_decision_mismatch_count']}"
        )
        print(
            "node_bit_mismatch_count="
            f"{cuda_compare['node_bit_mismatch_count']}"
        )
        print(
            "top1_mismatch_count="
            f"{cuda_compare['top1_mismatch_count']}"
        )

    if cpu_cuda is not None:
        print()
        print("CPU VS CUDA")
        print(
            "graph_max_abs_error="
            f"{cpu_cuda['graph_error']['max_abs_error']:.9e}"
        )
        print(
            "node_max_abs_error="
            f"{cpu_cuda['node_error']['max_abs_error']:.9e}"
        )
        print(
            "graph_decision_mismatch_count="
            f"{cpu_cuda['graph_decision_mismatch_count']}"
        )
        print(
            "node_bit_mismatch_count="
            f"{cpu_cuda['node_bit_mismatch_count']}"
        )
        print(
            "top1_mismatch_count="
            f"{cpu_cuda['top1_mismatch_count']}"
        )

    print()
    print(f"saved_metrics_match_g1_3={all(expected_metric_checks.values())}")
    print(f"diagnosis={diagnosis}")
    print(
        "recommended_canonical_reference="
        f"{recommended_canonical}"
    )
    print(f"diagnostic_report={report_path}")


if __name__ == "__main__":
    main()
