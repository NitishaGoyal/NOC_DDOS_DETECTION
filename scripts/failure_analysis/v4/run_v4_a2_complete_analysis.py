#!/usr/bin/env python3
"""
Run the complete V4-A2 evaluation, failure analysis, A1 comparison, and A3 gate.

This is a resumable orchestration script. It never trains a neural model. It:
  1. audits A1/A2 checkpoints and the V4 dataset,
  2. runs/reuses A2 validation inference and validation-only threshold selection,
  3. freezes an FPR-constrained graph threshold using validation only,
  4. runs/reuses the frozen development-comparison test transfer,
  5. runs/reuses the comprehensive failure analysis,
  6. runs/reuses validation-fitted set-decoding diagnostics,
  7. computes an aligned A1-vs-A2 capacity comparison,
  8. writes a machine-readable A3 recommendation and a human-review gate.

The test split is never used for fitting or selection. Existing valid stages are
reused. Existing inconsistent stages cause a hard failure unless that stage is
explicitly forced, in which case the old directory is archived rather than
silently overwritten.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCRIPT_VERSION = "1.0.0"
PASS_MARKER = "V4_A2_COMPLETE_FAILURE_ANALYSIS_AND_A1_COMPARISON_PASS"
SELF_TEST_MARKER = "V4_A2_COMPLETE_ANALYSIS_SELF_TEST_PASS"
NUM_ROUTERS = 16
CORNERS = np.asarray([0, 3, 12, 15], dtype=np.int64)
EDGES = np.asarray([1, 2, 4, 7, 8, 11, 13, 14], dtype=np.int64)
INTERIORS = np.asarray([5, 6, 9, 10], dtype=np.int64)
TOPOLOGY_GROUPS = {
    "corner": CORNERS,
    "edge": EDGES,
    "interior": INTERIORS,
}
EXPECTED_SCRIPT_HASHES = {
    "evaluator": "c73291562dbdca31f63dbf8b1114f78cf30cd2902550c7ddee75f8277203d125",
    "freezer": "8a919b44b15e7ea8edb8d6f928a210e580cc9c19c567e09177819d09fca3715b",
    "base_analysis": "ce59f87052215719cf9d767bd1fe85c86c99543c7696a89b34ad35dc056bc8fc",
    "remaining_analysis": "dcf206e9167e6937262dc109c1c33e753a3dc7f5ca74ccda0279013aad06ab82",
}


@dataclass(frozen=True)
class Paths:
    repo: Path
    data: Path
    a1_model: Path
    a2_model: Path
    a1_val_pred: Path
    a1_test_pred: Path
    a1_thresholds: Path
    a1_base_analysis: Path
    a1_remaining_analysis: Path
    evaluator: Path
    freezer: Path
    base_script: Path
    remaining_script: Path
    diagnostic_decision: Path
    v3_data: Path | None
    validation_dir: Path
    freeze_dir: Path
    test_dir: Path
    out: Path
    logs: Path
    a2_base_analysis: Path
    a2_remaining_analysis: Path


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        order: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    order.append(key)
        fields = order or ["empty"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def require_files(paths: Iterable[Path], label: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"{label}: missing required files:\n  " + "\n  ".join(missing))


def is_nonempty_dir(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def archive_dir(path: Path) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.name}_archived_{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_archived_{stamp}_{counter}")
        counter += 1
    path.rename(candidate)
    return candidate


def prepare_stage_dir(path: Path, force: bool, stage_name: str) -> None:
    if not path.exists():
        return
    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()
        return
    if not force:
        raise RuntimeError(
            f"Stage '{stage_name}' has existing inconsistent or incomplete output: {path}. "
            f"Inspect it, or rerun with --force-stage {stage_name} to archive and rebuild it."
        )
    archived = archive_dir(path)
    print(f"ARCHIVED stage={stage_name} old={archived}")


def stage_forced(args: argparse.Namespace, stage: str) -> bool:
    return bool(args.force_all or stage in set(args.force_stage or []))


def run_command(command: Sequence[str], log_path: Path, dry_run: bool = False) -> None:
    printable = " ".join(subprocess.list2cmdline([item]) for item in command)
    print(f"RUN: {printable}")
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit status {return_code}. Log: {log_path}")


def validate_script_hash(path: Path, expected: str, allow_drift: bool) -> dict[str, Any]:
    actual = sha256_file(path)
    matches = actual == expected
    if not matches and not allow_drift:
        raise RuntimeError(
            f"Script hash drift for {path}: actual={actual} expected={expected}. "
            "Use --allow-script-hash-drift only after reviewing the script difference."
        )
    return {"path": str(path.resolve()), "sha256": actual, "expected_sha256": expected, "matches": matches}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_prediction_archive(path: Path, expected_split: str | None = None) -> dict[str, Any]:
    required = {"graph_prob", "node_prob", "y_graph", "y_node", "global_index", "run_index"}
    pred = load_npz(path)
    missing = sorted(required - set(pred))
    if missing:
        raise RuntimeError(f"Prediction archive {path} is missing keys: {missing}")
    n = int(pred["y_graph"].shape[0])
    checks = {
        "graph_prob_shape": tuple(pred["graph_prob"].shape) == (n,),
        "node_prob_shape": tuple(pred["node_prob"].shape) == (n, NUM_ROUTERS),
        "y_graph_shape": tuple(pred["y_graph"].shape) == (n,),
        "y_node_shape": tuple(pred["y_node"].shape) == (n, NUM_ROUTERS),
        "global_index_shape": tuple(pred["global_index"].shape) == (n,),
        "run_index_shape": tuple(pred["run_index"].shape) == (n,),
        "finite_graph_prob": bool(np.isfinite(pred["graph_prob"]).all()),
        "finite_node_prob": bool(np.isfinite(pred["node_prob"]).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Prediction archive shape/finite check failed for {path}: {checks}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "sample_count": n,
        "split": expected_split,
        "checks": checks,
        "keys": sorted(pred),
    }


def prediction_alignment(a_path: Path, b_path: Path, split: str) -> dict[str, Any]:
    a = load_npz(a_path)
    b = load_npz(b_path)
    keys = ["global_index", "run_index", "y_graph", "y_node"]
    result: dict[str, Any] = {"split": split, "a": str(a_path), "b": str(b_path)}
    for key in keys:
        result[f"{key}_equal"] = bool(np.array_equal(a[key], b[key]))
    result["sample_count_equal"] = int(a["y_graph"].shape[0]) == int(b["y_graph"].shape[0])
    result["pass"] = bool(all(value for key, value in result.items() if key.endswith("_equal")) and result["sample_count_equal"])
    if not result["pass"]:
        raise RuntimeError(f"A1/A2 prediction alignment failed on {split}: {result}")
    return result


def snapshot_dataset(data_dir: Path) -> dict[str, Any]:
    small_hash_files = [
        "metadata.json", "y_graph.npy", "y_node.npy", "edge_index.npy",
        "run_index.npy", "split_id.npy", "split.npy", "run_id.npy",
    ]
    snapshot: dict[str, Any] = {"data_dir": str(data_dir.resolve()), "files": {}}
    for name in small_hash_files:
        path = data_dir / name
        if not path.is_file():
            continue
        stat = path.stat()
        snapshot["files"][name] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        }
    x_path = data_dir / "x.npy"
    if x_path.is_file():
        stat = x_path.stat()
        snapshot["files"]["x.npy"] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": None}
    return snapshot


def compare_dataset_snapshots(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return before == after


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required to audit checkpoints.") from exc
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    except Exception:
        value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Checkpoint is not a mapping: {path}")
    return value


def checkpoint_parameter_count(checkpoint: Mapping[str, Any]) -> int:
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if not isinstance(state, Mapping):
        raise RuntimeError("Checkpoint does not contain model_state_dict/state_dict")
    return int(sum(int(t.numel()) for t in state.values() if hasattr(t, "numel")))


def checkpoint_audit(a1_path: Path, a2_path: Path) -> dict[str, Any]:
    a1 = load_checkpoint(a1_path)
    a2 = load_checkpoint(a2_path)
    a1_state = a1.get("model_state_dict", a1.get("state_dict"))
    a2_state = a2.get("model_state_dict", a2.get("state_dict"))
    if not isinstance(a1_state, Mapping) or not isinstance(a2_state, Mapping):
        raise RuntimeError("Cannot inspect checkpoint state dictionaries")
    key_set_equal = set(a1_state) == set(a2_state)
    expected_shapes = {
        "temporal.conv.weight": ((8, 24, 3), (16, 24, 3)),
        "gcn1.linear.weight": ((16, 8), (32, 16)),
        "gcn2.linear.weight": ((8, 16), (16, 32)),
        "node_head.weight": ((1, 8), (1, 16)),
        "graph_head.weight": ((1, 8), (1, 16)),
    }
    shape_checks: dict[str, Any] = {}
    for key, (a1_expected, a2_expected) in expected_shapes.items():
        if key not in a1_state or key not in a2_state:
            shape_checks[key] = {"present": False}
            continue
        a1_shape = tuple(int(v) for v in a1_state[key].shape)
        a2_shape = tuple(int(v) for v in a2_state[key].shape)
        shape_checks[key] = {
            "present": True,
            "a1_shape": a1_shape,
            "a2_shape": a2_shape,
            "pass": a1_shape == a1_expected and a2_shape == a2_expected,
        }
    a1_count = checkpoint_parameter_count(a1)
    a2_count = checkpoint_parameter_count(a2)
    pass_value = bool(
        key_set_equal
        and a1_count == 882
        and a2_count == 2274
        and all(info.get("pass", False) for info in shape_checks.values())
    )
    report = {
        "a1_checkpoint": str(a1_path.resolve()),
        "a2_checkpoint": str(a2_path.resolve()),
        "a1_sha256": sha256_file(a1_path),
        "a2_sha256": sha256_file(a2_path),
        "a1_parameter_count": a1_count,
        "a2_parameter_count": a2_count,
        "state_dict_key_set_equal": key_set_equal,
        "declared_width_only_shape_checks": shape_checks,
        "pass": pass_value,
    }
    if not pass_value:
        raise RuntimeError(f"A1/A2 checkpoint architecture audit failed: {report}")
    return report


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    truth = np.asarray(y_true).astype(bool)
    pred = np.asarray(probabilities) >= threshold
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    tn = int(np.sum(~pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "threshold": float(threshold),
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "fpr": safe_div(fp, fp + tn),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def node_metrics(y_node: np.ndarray, node_prob: np.ndarray, threshold: float, attack_only: bool = False) -> dict[str, Any]:
    truth = np.asarray(y_node) >= 0.5
    pred = np.asarray(node_prob) >= threshold
    if attack_only:
        mask = truth.any(axis=1)
        truth = truth[mask]
        pred = pred[mask]
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    tn = int(np.sum(~pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    exact = float(np.mean(np.all(pred == truth, axis=1))) if truth.shape[0] else 0.0
    counts_true = truth.sum(axis=1)
    counts_pred = pred.sum(axis=1)
    return {
        "sample_count": int(truth.shape[0]),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "fpr": safe_div(fp, fp + tn),
        "exact_localization": exact,
        "count_accuracy": float(np.mean(counts_true == counts_pred)) if counts_true.size else 0.0,
        "count_mae": float(np.mean(np.abs(counts_true - counts_pred))) if counts_true.size else 0.0,
        "empty_prediction_fraction": float(np.mean(counts_pred == 0)) if counts_pred.size else 0.0,
        "mean_true_count": float(np.mean(counts_true)) if counts_true.size else 0.0,
        "mean_predicted_count": float(np.mean(counts_pred)) if counts_pred.size else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def topk_prediction(node_prob: np.ndarray, counts: np.ndarray) -> np.ndarray:
    scores = np.asarray(node_prob)
    counts = np.asarray(counts).astype(np.int64)
    output = np.zeros_like(scores, dtype=bool)
    order = np.argsort(-scores, axis=1, kind="stable")
    for row, count in enumerate(counts.tolist()):
        k = max(0, min(NUM_ROUTERS, int(count)))
        if k:
            output[row, order[row, :k]] = True
    return output


def oracle_node_metrics(y_node: np.ndarray, node_prob: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y_node) >= 0.5
    counts = truth.sum(axis=1)
    pred = topk_prediction(node_prob, counts)
    attack = truth.any(axis=1)
    return node_metrics_from_binary(truth[attack], pred[attack])


def node_metrics_from_binary(truth: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth).astype(bool)
    pred = np.asarray(pred).astype(bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    tn = int(np.sum(~pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    true_counts = truth.sum(axis=1)
    pred_counts = pred.sum(axis=1)
    return {
        "sample_count": int(truth.shape[0]),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "exact_localization": float(np.mean(np.all(truth == pred, axis=1))) if truth.shape[0] else 0.0,
        "count_accuracy": float(np.mean(true_counts == pred_counts)) if truth.shape[0] else 0.0,
        "count_mae": float(np.mean(np.abs(true_counts - pred_counts))) if truth.shape[0] else 0.0,
        "empty_prediction_fraction": float(np.mean(pred_counts == 0)) if truth.shape[0] else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def attacker_rank_metrics(y_node: np.ndarray, node_prob: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y_node) >= 0.5
    scores = np.asarray(node_prob)
    order = np.argsort(-scores, axis=1, kind="stable")
    inverse = np.empty_like(order)
    inverse[np.arange(order.shape[0])[:, None], order] = np.arange(NUM_ROUTERS)[None, :]
    ranks = inverse[truth] + 1
    if ranks.size == 0:
        return {"attacker_instances": 0, "mean_rank": None, "median_rank": None, "p90_rank": None,
                "top1_fraction": 0.0, "top2_fraction": 0.0, "top4_fraction": 0.0}
    return {
        "attacker_instances": int(ranks.size),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "p90_rank": float(np.percentile(ranks, 90)),
        "top1_fraction": float(np.mean(ranks <= 1)),
        "top2_fraction": float(np.mean(ranks <= 2)),
        "top4_fraction": float(np.mean(ranks <= 4)),
    }


def manhattan(a: int, b: int) -> int:
    return abs(a // 4 - b // 4) + abs(a % 4 - b % 4)


def fp_distance_metrics(y_node: np.ndarray, node_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    truth = np.asarray(y_node) >= 0.5
    pred = np.asarray(node_prob) >= threshold
    counts: dict[int, int] = {}
    for row in np.flatnonzero(truth.any(axis=1)):
        true_nodes = np.flatnonzero(truth[row])
        false_nodes = np.flatnonzero(pred[row] & ~truth[row])
        for node in false_nodes.tolist():
            distance = min(manhattan(node, int(t)) for t in true_nodes.tolist())
            counts[distance] = counts.get(distance, 0) + 1
    total = sum(counts.values())
    within1 = safe_div(counts.get(1, 0), total)
    within2 = safe_div(sum(value for distance, value in counts.items() if distance <= 2), total)
    return {
        "total_attack_window_false_positives": int(total),
        "distance_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "one_hop_fraction": within1,
        "within_two_hops_fraction": within2,
    }


def topology_metrics(y_node: np.ndarray, node_prob: np.ndarray, threshold: float) -> dict[str, dict[str, Any]]:
    truth = np.asarray(y_node) >= 0.5
    pred = np.asarray(node_prob) >= threshold
    output: dict[str, dict[str, Any]] = {}
    for name, routers in TOPOLOGY_GROUPS.items():
        t = truth[:, routers]
        p = pred[:, routers]
        output[name] = node_metrics_from_binary(t, p)
    return output


def graph_node_consistency(y_graph: np.ndarray, graph_prob: np.ndarray, y_node: np.ndarray, node_prob: np.ndarray,
                           graph_threshold: float, node_threshold: float) -> dict[str, Any]:
    graph_pred = np.asarray(graph_prob) >= graph_threshold
    node_nonempty = (np.asarray(node_prob) >= node_threshold).any(axis=1)
    attack = np.asarray(y_graph) >= 0.5
    rows: dict[str, Any] = {}
    for subset_name, subset in {"all": np.ones_like(attack, dtype=bool), "normal": ~attack, "attack": attack}.items():
        denom = int(np.sum(subset))
        rows[subset_name] = {
            "sample_count": denom,
            "graph_positive_node_nonempty": int(np.sum(subset & graph_pred & node_nonempty)),
            "graph_positive_node_empty": int(np.sum(subset & graph_pred & ~node_nonempty)),
            "graph_negative_node_nonempty": int(np.sum(subset & ~graph_pred & node_nonempty)),
            "graph_negative_node_empty": int(np.sum(subset & ~graph_pred & ~node_nonempty)),
        }
        for key in list(rows[subset_name]):
            if key != "sample_count":
                rows[subset_name][f"{key}_fraction"] = safe_div(rows[subset_name][key], denom)
    return rows


def exact_set_decomposition(y_node: np.ndarray, node_prob: np.ndarray, threshold: float, attack_only: bool = False) -> dict[str, Any]:
    truth = np.asarray(y_node) >= 0.5
    pred = np.asarray(node_prob) >= threshold
    if attack_only:
        mask = truth.any(axis=1)
        truth, pred = truth[mask], pred[mask]
    categories = {
        "exact": 0,
        "empty_prediction": 0,
        "strict_subset_only": 0,
        "all_true_plus_extras": 0,
        "mixed_misses_and_extras": 0,
        "no_true_attacker_found": 0,
    }
    for t, p in zip(truth, pred):
        if np.array_equal(t, p):
            categories["exact"] += 1
            continue
        if not p.any():
            categories["empty_prediction"] += 1
            continue
        overlap = bool(np.any(t & p))
        misses = bool(np.any(t & ~p))
        extras = bool(np.any(~t & p))
        if not overlap:
            categories["no_true_attacker_found"] += 1
        elif misses and not extras:
            categories["strict_subset_only"] += 1
        elif not misses and extras:
            categories["all_true_plus_extras"] += 1
        else:
            categories["mixed_misses_and_extras"] += 1
    n = int(truth.shape[0])
    return {
        "sample_count": n,
        "counts": categories,
        "fractions": {key: safe_div(value, n) for key, value in categories.items()},
    }


def count_confusion_rows(y_node: np.ndarray, node_prob: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    truth = (np.asarray(y_node) >= 0.5).sum(axis=1).astype(int)
    pred = (np.asarray(node_prob) >= threshold).sum(axis=1).astype(int)
    rows: list[dict[str, Any]] = []
    for true_count in sorted(np.unique(truth).tolist()):
        for pred_count in sorted(np.unique(pred[truth == true_count]).tolist()):
            rows.append({
                "true_count": int(true_count),
                "predicted_count": int(pred_count),
                "sample_count": int(np.sum((truth == true_count) & (pred == pred_count))),
            })
    return rows


def model_summary(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    value = json_load(path)
    return value if isinstance(value, Mapping) else {}


def validate_validation_stage(paths: Paths) -> bool:
    required = [
        paths.validation_dir / "validation_predictions.npz",
        paths.validation_dir / "threshold_sweep.csv",
        paths.validation_dir / "selected_thresholds.json",
        paths.validation_dir / "validation_metrics.json",
    ]
    if not all(path.is_file() for path in required):
        return False
    threshold = json_load(paths.validation_dir / "selected_thresholds.json")
    if threshold.get("experiment_designation") != "a2_capacity_lift" or threshold.get("test_accessed") is not False:
        return False
    validate_prediction_archive(paths.validation_dir / "validation_predictions.npz", "validation")
    return True


def validate_freeze_stage(paths: Paths) -> bool:
    required = [paths.freeze_dir / "selected_thresholds.json", paths.freeze_dir / "validation_metrics.json"]
    if not all(path.is_file() for path in required):
        return False
    threshold = json_load(required[0])
    metrics = json_load(required[1])
    graph = metrics.get("graph", {})
    return bool(
        threshold.get("experiment_designation") == "a2_capacity_lift"
        and threshold.get("test_accessed") is False
        and abs(float(threshold.get("graph_fpr_constraint", threshold.get("fpr_cap", 0.10))) - 0.10) < 1e-9
        and float(graph.get("fpr", 1.0)) <= 0.10 + 1e-12
    )


def validate_test_stage(paths: Paths) -> bool:
    required = [paths.test_dir / "test_predictions.npz", paths.test_dir / "test_metrics.json"]
    if not all(path.is_file() for path in required):
        return False
    metrics = json_load(required[1])
    if metrics.get("experiment_designation") != "a2_capacity_lift":
        return False
    if metrics.get("thresholds_selected_on_test") is not False:
        return False
    validate_prediction_archive(required[0], "test")
    return True


def validate_base_stage(paths: Paths) -> bool:
    verdict = paths.a2_base_analysis / "final_verdict.json"
    return verdict.is_file() and json_load(verdict).get("analysis_pass") is True


def validate_remaining_stage(paths: Paths) -> bool:
    decision = paths.a2_remaining_analysis / "final_decision.json"
    if not decision.is_file():
        return False
    value = json_load(decision)
    hard = value.get("hard_checks", {})
    return bool(
        value.get("analysis_pass") is True
        and hard.get("validation_only_fitting") is True
        and hard.get("test_used_for_fit_or_selection") is False
        and hard.get("model_inference_performed") is False
        and hard.get("neural_model_training_performed") is False
        and hard.get("dataset_modified") is False
    )


def ensure_or_run_stage(
    name: str,
    output_dir: Path,
    validator: Any,
    command: Sequence[str],
    log_path: Path,
    args: argparse.Namespace,
) -> None:
    if validator():
        if stage_forced(args, name):
            archived = archive_dir(output_dir)
            print(f"ARCHIVED valid stage={name} old={archived} because it was explicitly forced")
        else:
            print(f"REUSE stage={name} output={output_dir}")
            return
    elif output_dir.exists():
        prepare_stage_dir(output_dir, stage_forced(args, name), name)
    run_command(command, log_path, args.dry_run)
    if args.dry_run:
        return
    if not validator():
        raise RuntimeError(f"Stage '{name}' completed but failed validation: {output_dir}")
    print(f"PASS stage={name}")


def copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and sha256_file(destination) == sha256_file(source):
            return
        raise RuntimeError(f"Refusing to overwrite differing artifact: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def float_row(row: Mapping[str, str], key: str) -> float:
    return float(row[key])


def find_method_row(path: Path, method: str) -> dict[str, Any]:
    rows = read_csv(path)
    for row in rows:
        if row.get("method") == method:
            return {key: (finite_or_none(value) if key != "method" and value != "" else value) for key, value in row.items()}
    raise RuntimeError(f"Method '{method}' not found in {path}")


def collect_split_metrics(pred: Mapping[str, np.ndarray], graph_threshold: float, node_threshold: float) -> dict[str, Any]:
    return {
        "graph": binary_metrics(pred["y_graph"], pred["graph_prob"], graph_threshold),
        "node_overall": node_metrics(pred["y_node"], pred["node_prob"], node_threshold, attack_only=False),
        "node_attack_only": node_metrics(pred["y_node"], pred["node_prob"], node_threshold, attack_only=True),
        "oracle_attack_only": oracle_node_metrics(pred["y_node"], pred["node_prob"]),
        "rank": attacker_rank_metrics(pred["y_node"], pred["node_prob"]),
        "fp_distance": fp_distance_metrics(pred["y_node"], pred["node_prob"], node_threshold),
        "topology": topology_metrics(pred["y_node"], pred["node_prob"], node_threshold),
        "graph_node_consistency": graph_node_consistency(
            pred["y_graph"], pred["graph_prob"], pred["y_node"], pred["node_prob"],
            graph_threshold, node_threshold,
        ),
        "exact_set_overall": exact_set_decomposition(pred["y_node"], pred["node_prob"], node_threshold, False),
        "exact_set_attack_only": exact_set_decomposition(pred["y_node"], pred["node_prob"], node_threshold, True),
    }


def flatten_comparison_metrics(model_metrics: Mapping[str, Any], learned_row: Mapping[str, Any], parameter_count: int) -> dict[str, float]:
    test = model_metrics["test"]
    val = model_metrics["validation"]
    topo = test["topology"]
    return {
        "graph_f1_at_fpr10": float(test["graph"]["f1"]),
        "test_graph_fpr": float(test["graph"]["fpr"]),
        "graph_recall": float(test["graph"]["recall"]),
        "attack_node_f1": float(test["node_attack_only"]["f1"]),
        "attack_exact_localization": float(test["node_attack_only"]["exact_localization"]),
        "attacker_count_accuracy": float(test["node_attack_only"]["count_accuracy"]),
        "count_mae": float(test["node_attack_only"]["count_mae"]),
        "oracle_count_attack_exact": float(test["oracle_attack_only"]["exact_localization"]),
        "learned_decoder_attack_exact": float(learned_row["attack_exact_localization"]),
        "true_attacker_mean_rank": float(test["rank"]["mean_rank"]),
        "true_attacker_median_rank": float(test["rank"]["median_rank"]),
        "attack_empty_prediction_fraction": float(test["node_attack_only"]["empty_prediction_fraction"]),
        "one_hop_fp_fraction": float(test["fp_distance"]["one_hop_fraction"]),
        "within_two_hops_fp_fraction": float(test["fp_distance"]["within_two_hops_fraction"]),
        "corner_node_f1": float(topo["corner"]["f1"]),
        "edge_node_f1": float(topo["edge"]["f1"]),
        "interior_node_f1": float(topo["interior"]["f1"]),
        "corner_interior_f1_gap": float(topo["interior"]["f1"] - topo["corner"]["f1"]),
        "validation_to_test_graph_f1_gap_abs": abs(float(test["graph"]["f1"] - val["graph"]["f1"])),
        "validation_to_test_attack_node_f1_gap_abs": abs(float(test["node_attack_only"]["f1"] - val["node_attack_only"]["f1"])),
        "parameter_count": float(parameter_count),
    }


def comparison_rows(a1: Mapping[str, float], a2: Mapping[str, float]) -> list[dict[str, Any]]:
    lower_better = {
        "test_graph_fpr", "count_mae", "true_attacker_mean_rank", "true_attacker_median_rank",
        "attack_empty_prediction_fraction", "one_hop_fp_fraction", "within_two_hops_fp_fraction",
        "corner_interior_f1_gap", "validation_to_test_graph_f1_gap_abs",
        "validation_to_test_attack_node_f1_gap_abs", "parameter_count",
    }
    interpretations = {
        "graph_f1_at_fpr10": "Detection quality under the frozen validation FPR constraint",
        "test_graph_fpr": "Operational false-alarm rate on the development-comparison test",
        "graph_recall": "Fraction of attack windows detected",
        "attack_node_f1": "Attacker-node quality on attack windows only",
        "attack_exact_localization": "Exact attacker set on attack windows only",
        "attacker_count_accuracy": "Correct attacker cardinality on attack windows",
        "count_mae": "Absolute cardinality error on attack windows",
        "oracle_count_attack_exact": "Source ranking quality when true cardinality is supplied",
        "learned_decoder_attack_exact": "Validation-selected learned set decoder transferred to test",
        "true_attacker_mean_rank": "Average ranking position of true attackers",
        "true_attacker_median_rank": "Median ranking position of true attackers",
        "attack_empty_prediction_fraction": "Attack windows for which no router is selected",
        "one_hop_fp_fraction": "False positives directly adjacent to a true attacker",
        "within_two_hops_fp_fraction": "False positives within the two-layer GCN receptive radius",
        "corner_node_f1": "Localization quality on corner routers",
        "edge_node_f1": "Localization quality on edge routers",
        "interior_node_f1": "Localization quality on interior routers",
        "corner_interior_f1_gap": "Topology bias; smaller is better",
        "validation_to_test_graph_f1_gap_abs": "Detection generalization stability",
        "validation_to_test_attack_node_f1_gap_abs": "Localization generalization stability",
        "parameter_count": "Model storage/compute proxy; smaller is better",
    }
    rows: list[dict[str, Any]] = []
    for metric in a1:
        if metric not in a2:
            continue
        x, y = float(a1[metric]), float(a2[metric])
        delta = y - x
        relative = safe_div(delta, abs(x)) if x != 0 else None
        if abs(delta) < 1e-12:
            winner = "tie"
        elif metric in lower_better:
            winner = "A2" if y < x else "A1"
        else:
            winner = "A2" if y > x else "A1"
        rows.append({
            "metric": metric,
            "a1_value": x,
            "a2_value": y,
            "absolute_difference_a2_minus_a1": delta,
            "relative_difference": relative,
            "winner": winner,
            "higher_is_better": metric not in lower_better,
            "interpretation": interpretations.get(metric, ""),
        })
    return rows


def build_capacity_verdict(a1: Mapping[str, float], a2: Mapping[str, float], hard_checks: Mapping[str, Any]) -> dict[str, Any]:
    delta = {key: float(a2[key] - a1[key]) for key in a1 if key in a2}
    rank_gain = bool(
        delta["oracle_count_attack_exact"] >= 0.03
        or delta["true_attacker_mean_rank"] <= -0.50
        or delta["true_attacker_median_rank"] <= -1.0
    )
    count_gain = bool(delta["attacker_count_accuracy"] >= 0.05 or delta["count_mae"] <= -0.15)
    aggregate_gain = bool(
        delta["graph_f1_at_fpr10"] >= 0.02
        or delta["attack_node_f1"] >= 0.03
        or delta["attack_exact_localization"] >= 0.03
    )
    calibration_only = bool(aggregate_gain and not rank_gain and not count_gain)
    persistent_failures = {
        "oracle_attack_exact_below_0_50": a2["oracle_count_attack_exact"] < 0.50,
        "attack_exact_below_0_25": a2["attack_exact_localization"] < 0.25,
        "count_accuracy_below_0_50": a2["attacker_count_accuracy"] < 0.50,
        "within_two_hops_fp_above_0_75": a2["within_two_hops_fp_fraction"] > 0.75,
        "corner_gap_above_0_10": a2["corner_interior_f1_gap"] > 0.10,
    }
    if not aggregate_gain and not rank_gain and not count_gain:
        outcome = "Outcome E — width is not useful"
    elif rank_gain:
        outcome = "Outcome A — width improves ranking"
    elif count_gain:
        outcome = "Outcome C — width improves count but not ranking"
    elif calibration_only:
        outcome = "Outcome B — width mainly improves calibration/threshold behaviour"
    else:
        outcome = "Outcome D — width improves aggregate metrics but preserves the same failure modes"
    residual_d = any(persistent_failures.values())
    return {
        "analysis_pass": bool(all(bool(v) for v in hard_checks.values())),
        "primary_capacity_outcome": outcome,
        "residual_outcome_d": residual_d,
        "evidence_flags": {
            "ranking_improved_materially": rank_gain,
            "cardinality_improved_materially": count_gain,
            "aggregate_metrics_improved_materially": aggregate_gain,
            "calibration_or_threshold_only_pattern": calibration_only,
            "persistent_failure_modes": persistent_failures,
        },
        "metric_deltas_a2_minus_a1": delta,
        "interpretation": (
            "A2 is a diagnostic capacity result, not a publication-final model. "
            "A material oracle/rank gain indicates representation capacity helped; persistent cardinality, "
            "two-hop, corner, or exact-set failures still require structural changes."
        ),
        "hard_checks": dict(hard_checks),
    }


def build_a3_recommendation(a2: Mapping[str, float], verdict: Mapping[str, Any], consistency: Mapping[str, Any]) -> dict[str, Any]:
    attack_consistency = consistency.get("attack", {})
    graph_pos_node_empty = float(attack_consistency.get("graph_positive_node_empty_fraction", 0.0))
    required = {
        "source_preserving_local_skip": a2["oracle_count_attack_exact"] < 0.50 or a2["within_two_hops_fp_fraction"] > 0.75,
        "one_gcn_instead_of_two": a2["within_two_hops_fp_fraction"] > 0.75,
        "explicit_count_head": a2["attacker_count_accuracy"] < 0.50,
        "count_conditioned_topk": a2["attacker_count_accuracy"] < 0.70,
        "hard_negative_neighbour_ranking": a2["one_hop_fp_fraction"] > 0.50 or a2["within_two_hops_fp_fraction"] > 0.75,
        "valid_port_masks": a2["corner_interior_f1_gap"] > 0.10,
    }
    recommended = {
        "mean_plus_max_fused_graph_readout": (
            graph_pos_node_empty > 0.10 or a2["attack_exact_localization"] < 0.25
        ),
        "separate_localization_adapter": True,
        "regional_embedding_output": True,
        "configurable_embedding_dimension_for_rtl": True,
    }
    postpone = {
        "directional_gating": "Run only after A3-base if direction-specific or corner/path confusion remains.",
        "tiny_port_attention": "Post-A3 controlled extension; do not place on the initial A3 critical path.",
        "more_gcn_layers": "Rejected because neighbour spreading is already a dominant error.",
        "global_transformer_or_gat": "Rejected for current evidence and RTL cost.",
        "rnn": "No evidence justifies replacing the lightweight temporal encoder yet.",
    }
    a2_viable = bool(
        a2["graph_f1_at_fpr10"] >= 0.75
        and a2["attack_node_f1"] >= 0.45
        and a2["attack_exact_localization"] >= 0.15
    )
    return {
        "a3_still_necessary": bool(any(required.values())),
        "a2_regional_model_status": "diagnostic_candidate_only" if a2_viable else "not_viable_as_final_regional_model",
        "mandatory_components": required,
        "recommended_components": recommended,
        "post_a3_or_later": postpone,
        "questions": {
            "did_width_improve_true_attacker_ranking": verdict["evidence_flags"]["ranking_improved_materially"],
            "did_width_improve_cardinality": verdict["evidence_flags"]["cardinality_improved_materially"],
            "did_width_reduce_empty_predictions": verdict["metric_deltas_a2_minus_a1"]["attack_empty_prediction_fraction"] < 0,
            "did_width_reduce_neighbour_confusion": verdict["metric_deltas_a2_minus_a1"]["within_two_hops_fp_fraction"] < -0.03,
            "did_width_improve_corners": verdict["metric_deltas_a2_minus_a1"]["corner_node_f1"] > 0.03,
            "did_width_reduce_graph_localization_tradeoff": graph_pos_node_empty < 0.10,
            "is_a2_viable_final_regional_model": False,
            "is_a3_still_necessary": bool(any(required.values())),
        },
        "hardware_contract": {
            "regional_input": "[16 routers, temporal_window, feature_count]",
            "required_outputs": [
                "regional_attack_score", "attacker_scores[16]", "attacker_count_logits[5]",
                "regional_embedding[D]",
            ],
            "future_optional_outputs": ["victim_scores[16]", "victim_count_logits"],
            "shared_weights_across_regions": True,
            "embedding_dimension_must_remain_configurable": True,
        },
    }


def failure_mode_rows(a1: Mapping[str, float], a2: Mapping[str, float]) -> list[dict[str, Any]]:
    definitions = [
        ("cardinality", "attacker_count_accuracy", True, 0.50),
        ("source_ranking", "oracle_count_attack_exact", True, 0.50),
        ("empty_predictions", "attack_empty_prediction_fraction", False, 0.10),
        ("one_hop_neighbour_confusion", "one_hop_fp_fraction", False, 0.40),
        ("two_hop_neighbour_confusion", "within_two_hops_fp_fraction", False, 0.70),
        ("corner_localization", "corner_node_f1", True, 0.50),
        ("topology_gap", "corner_interior_f1_gap", False, 0.10),
        ("exact_set_localization", "attack_exact_localization", True, 0.25),
    ]
    rows = []
    for mode, metric, higher, target in definitions:
        x, y = a1[metric], a2[metric]
        improved = y > x if higher else y < x
        resolved = y >= target if higher else y <= target
        rows.append({
            "failure_mode": mode,
            "metric": metric,
            "a1_value": x,
            "a2_value": y,
            "improved": improved,
            "resolved": resolved,
            "status": "resolved" if resolved else ("improved_but_persists" if improved else "persists_or_regressed"),
        })
    rows.append({
        "failure_mode": "v4_dataset_shortcut",
        "metric": "active_core_matched_control_shortcut",
        "a1_value": "present",
        "a2_value": "present",
        "improved": False,
        "resolved": False,
        "status": "publication_blocker_shared_by_both_models",
    })
    return rows


def build_markdown_report(
    checkpoint: Mapping[str, Any], capacity: Mapping[str, Any], a3: Mapping[str, Any],
    comparison: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# V4-A2 Complete Failure Analysis and A1 Comparison",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "## Integrity result",
        "",
        f"- A1 parameters: `{checkpoint['a1_parameter_count']}`",
        f"- A2 parameters: `{checkpoint['a2_parameter_count']}`",
        "- Threshold and decoder fitting: validation only",
        "- Test role: development-comparison transfer only",
        "- Dataset modification: none",
        "",
        "## Capacity verdict",
        "",
        f"**{capacity['primary_capacity_outcome']}**",
        "",
        f"Residual original failure modes remain: `{capacity['residual_outcome_d']}`",
        "",
        "## Key matched comparison",
        "",
        "| Metric | A1 | A2 | Δ A2−A1 | Winner |",
        "|---|---:|---:|---:|---|",
    ]
    important = {
        "graph_f1_at_fpr10", "test_graph_fpr", "attack_node_f1", "attack_exact_localization",
        "attacker_count_accuracy", "oracle_count_attack_exact", "learned_decoder_attack_exact",
        "true_attacker_mean_rank", "attack_empty_prediction_fraction", "within_two_hops_fp_fraction",
        "corner_node_f1", "parameter_count",
    }
    for row in comparison:
        if row["metric"] in important:
            lines.append(
                f"| {row['metric']} | {row['a1_value']:.6f} | {row['a2_value']:.6f} | "
                f"{row['absolute_difference_a2_minus_a1']:.6f} | {row['winner']} |"
            )
    lines.extend([
        "",
        "## A3 gate",
        "",
        f"A3 remains necessary: `{a3['a3_still_necessary']}`",
        "",
        "Mandatory components supported by the evidence:",
        "",
    ])
    for key, value in a3["mandatory_components"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Hardware contract",
        "",
        "The next model remains a reusable 4×4 regional expert with shared weights across 8×8/16×16 regions, "
        "a configurable regional embedding dimension, 16 attacker scores, a graph score, and 5 attacker-count logits.",
        "",
        "## Stop condition",
        "",
        "A3 is on hold until this report and `a1_vs_a2_capacity_verdict.json` are reviewed. "
        "The generated `A3_START_GATE.json` intentionally remains `HOLD_FOR_HUMAN_REVIEW`.",
        "",
    ])
    return "\n".join(lines)


def materialize_summary_outputs(paths: Paths, metrics: Mapping[str, Any], learned_a2: Mapping[str, Any]) -> None:
    copy_or_link(paths.validation_dir / "validation_predictions.npz", paths.out / "a2_validation_predictions.npz")
    copy_or_link(paths.test_dir / "test_predictions.npz", paths.out / "a2_test_predictions.npz")
    copy_or_link(paths.freeze_dir / "selected_thresholds.json", paths.out / "a2_frozen_thresholds.json")
    json_dump(paths.out / "a2_overall_metrics.json", metrics)

    count_rows = []
    for split_name in ("validation", "test"):
        row = {"split": split_name, **metrics[split_name]["node_attack_only"]}
        count_rows.append(row)
    write_csv(paths.out / "a2_count_diagnostics.csv", count_rows)
    write_csv(
        paths.out / "a2_count_confusion.csv",
        count_confusion_rows(
            load_npz(paths.test_dir / "test_predictions.npz")["y_node"],
            load_npz(paths.test_dir / "test_predictions.npz")["node_prob"],
            float(metrics["thresholds"]["node"]),
        ),
    )
    exact = metrics["test"]["exact_set_attack_only"]
    write_csv(paths.out / "a2_exact_set_decomposition.csv", [
        {"error_type": key, "sample_count": exact["counts"][key], "fraction": exact["fractions"][key]}
        for key in exact["counts"]
    ])
    write_csv(paths.out / "a2_true_attacker_rank.csv", [metrics["test"]["rank"]])
    fp = metrics["test"]["fp_distance"]
    total = fp["total_attack_window_false_positives"]
    write_csv(paths.out / "a2_fp_distance.csv", [
        {"distance": int(distance), "count": count, "fraction": safe_div(count, total)}
        for distance, count in sorted((int(k), int(v)) for k, v in fp["distance_counts"].items())
    ])
    write_csv(paths.out / "a2_topology_metrics.csv", [
        {"topology": name, **values} for name, values in metrics["test"]["topology"].items()
    ])
    source_router = paths.a2_base_analysis / "test" / "router_metrics.csv"
    if source_router.is_file():
        copy_or_link(source_router, paths.out / "a2_per_router_metrics.csv")
    source_scenario = paths.a2_remaining_analysis / "test" / "scenario_method_comparison.csv"
    if source_scenario.is_file():
        copy_or_link(source_scenario, paths.out / "a2_scenario_metrics.csv")
    consistency_rows = []
    for subset, values in metrics["test"]["graph_node_consistency"].items():
        consistency_rows.append({"subset": subset, **values})
    write_csv(paths.out / "a2_graph_node_consistency.csv", consistency_rows)
    json_dump(paths.out / "a2_selected_learned_decoder_test.json", learned_a2)


def required_final_report_names() -> list[str]:
    return [
        "a2_validation_predictions.npz", "a2_test_predictions.npz", "a2_frozen_thresholds.json",
        "a2_overall_metrics.json", "a2_count_diagnostics.csv", "a2_count_confusion.csv",
        "a2_exact_set_decomposition.csv", "a2_true_attacker_rank.csv", "a2_fp_distance.csv",
        "a2_topology_metrics.csv", "a2_per_router_metrics.csv", "a2_scenario_metrics.csv",
        "a2_graph_node_consistency.csv", "a1_vs_a2_metrics.csv", "a1_vs_a2_failure_modes.csv",
        "a1_vs_a2_capacity_verdict.json", "a3_recommendation.json", "complete_report.md",
        "provenance.json", "A3_START_GATE.json", "analysis_summary.json", "artifact_manifest.csv",
    ]


def validate_final_summary(paths: Paths) -> bool:
    required = [paths.out / name for name in required_final_report_names()]
    if not all(path.is_file() for path in required):
        return False
    try:
        provenance = json_load(paths.out / "provenance.json")
        summary = json_load(paths.out / "analysis_summary.json")
        gate = json_load(paths.out / "A3_START_GATE.json")
    except Exception:
        return False
    hard = provenance.get("hard_checks", {})
    if not hard or not all(bool(v) for v in hard.values()):
        return False
    if summary.get("analysis_pass") is not True or gate.get("analysis_pipeline_passed") is not True:
        return False
    hash_checks = [
        (paths.validation_dir / "validation_predictions.npz", provenance.get("a2_validation_predictions_sha256")),
        (paths.test_dir / "test_predictions.npz", provenance.get("a2_test_predictions_sha256")),
        (paths.freeze_dir / "selected_thresholds.json", provenance.get("a2_thresholds_sha256")),
    ]
    for path, expected in hash_checks:
        if not path.is_file() or not expected or sha256_file(path) != expected:
            return False
    return True


def write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.csv":
            rows.append({
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    write_csv(root / "artifact_manifest.csv", rows)


def build_paths(args: argparse.Namespace) -> Paths:
    repo = Path(args.repo_root).expanduser().resolve()
    data = Path(args.data_dir).expanduser().resolve() if args.data_dir else (
        repo / "data/processed/graph_dataset/paper1_temporal_graphs_ports_v4_all16_chrono_memmap"
    )
    stage_root = repo / "reports/v4/stage1_a2"
    out = Path(args.out_dir).expanduser().resolve() if args.out_dir else stage_root / "a2_7_complete_failure_analysis"
    v3 = None if args.skip_v3_comparison else Path(args.v3_data_dir).expanduser().resolve()
    return Paths(
        repo=repo,
        data=data,
        a1_model=Path(args.a1_model_dir).expanduser().resolve() if args.a1_model_dir else repo / "models/v4/v4_a1_conv1d_gcn_seed7",
        a2_model=Path(args.a2_model_dir).expanduser().resolve() if args.a2_model_dir else repo / "models/v4/v4_a2_capacitylift_conv1d_gcn_seed7",
        a1_val_pred=Path(args.a1_validation_predictions).expanduser().resolve() if args.a1_validation_predictions else repo / "reports/v4/stage1_a1/a1_5_validation_thresholds/validation_predictions.npz",
        a1_test_pred=Path(args.a1_test_predictions).expanduser().resolve() if args.a1_test_predictions else repo / "reports/v4/stage1_a1/a1_6_development_test_transfer/test_predictions.npz",
        a1_thresholds=Path(args.a1_thresholds).expanduser().resolve() if args.a1_thresholds else repo / "reports/v4/stage1_a1/a1_5b_fpr10_frozen_thresholds/selected_thresholds.json",
        a1_base_analysis=Path(args.a1_base_analysis).expanduser().resolve() if args.a1_base_analysis else repo / "reports/v4/stage1_a1/a1_7_failure_analysis",
        a1_remaining_analysis=Path(args.a1_remaining_analysis).expanduser().resolve() if args.a1_remaining_analysis else repo / "reports/v4/stage1_a1/a1_8_remaining_diagnostics",
        evaluator=Path(args.evaluator).expanduser().resolve() if args.evaluator else repo / "scripts/v4/eval/eval_v4_conv1d_gcn_experiment.py",
        freezer=Path(args.freezer).expanduser().resolve() if args.freezer else repo / "scripts/v4/eval/freeze_v4_fpr_constrained_thresholds.py",
        base_script=Path(args.base_analysis_script).expanduser().resolve() if args.base_analysis_script else repo / "scripts/failure_analysis/v4/analyze_v4_a1_failures.py",
        remaining_script=Path(args.remaining_analysis_script).expanduser().resolve() if args.remaining_analysis_script else repo / "scripts/failure_analysis/v4/analyze_v4_a1_remaining_diagnostics.py",
        diagnostic_decision=Path(args.diagnostic_decision).expanduser().resolve() if args.diagnostic_decision else repo / "reports/v4/stage0_dataset_audit/a0_2_to_a0_7_pretraining_diagnostic/v4_pretraining_diagnostic_decision.json",
        v3_data=v3,
        validation_dir=Path(args.validation_dir).expanduser().resolve() if args.validation_dir else stage_root / "a2_5_validation_thresholds",
        freeze_dir=Path(args.freeze_dir).expanduser().resolve() if args.freeze_dir else stage_root / "a2_5b_fpr10_frozen_thresholds",
        test_dir=Path(args.test_dir).expanduser().resolve() if args.test_dir else stage_root / "a2_6_development_comparison_transfer_fpr10",
        out=out,
        logs=out / "logs",
        a2_base_analysis=out / "a2_comprehensive_failure_analysis",
        a2_remaining_analysis=out / "a2_remaining_decoder_diagnostics",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-dir")
    parser.add_argument("--a1-model-dir")
    parser.add_argument("--a2-model-dir")
    parser.add_argument("--a1-validation-predictions")
    parser.add_argument("--a1-test-predictions")
    parser.add_argument("--a1-thresholds")
    parser.add_argument("--a1-base-analysis")
    parser.add_argument("--a1-remaining-analysis")
    parser.add_argument("--evaluator")
    parser.add_argument("--freezer")
    parser.add_argument("--base-analysis-script")
    parser.add_argument("--remaining-analysis-script")
    parser.add_argument("--diagnostic-decision")
    parser.add_argument("--v3-data-dir", default="~/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3")
    parser.add_argument("--skip-v3-comparison", action="store_true")
    parser.add_argument("--validation-dir")
    parser.add_argument("--freeze-dir")
    parser.add_argument("--test-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--count-train-cap", type=int, default=250000)
    parser.add_argument("--calibration-node-cap-per-topology", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force-stage", action="append", choices=["validation", "freeze", "test", "base", "remaining", "summary"])
    parser.add_argument("--force-all", action="store_true")
    parser.add_argument("--allow-script-hash-drift", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> int:
    y_graph = np.asarray([0, 1, 1, 1], dtype=np.int64)
    y_node = np.asarray([
        [0, 0, 0, 0] + [0] * 12,
        [1, 0, 0, 0] + [0] * 12,
        [0, 1, 0, 1] + [0] * 12,
        [0, 0, 1, 0] + [0] * 12,
    ], dtype=np.int64)
    node_prob = np.asarray([
        [0.1, 0.1, 0.1, 0.1] + [0.1] * 12,
        [0.9, 0.4, 0.1, 0.1] + [0.1] * 12,
        [0.1, 0.8, 0.2, 0.7] + [0.1] * 12,
        [0.2, 0.3, 0.8, 0.2] + [0.1] * 12,
    ], dtype=np.float64)
    graph_prob = np.asarray([0.1, 0.8, 0.7, 0.9])
    assert binary_metrics(y_graph, graph_prob, 0.5)["f1"] == 1.0
    attack = node_metrics(y_node, node_prob, 0.5, attack_only=True)
    assert attack["exact_localization"] == 1.0
    oracle = oracle_node_metrics(y_node, node_prob)
    assert oracle["exact_localization"] == 1.0
    ranks = attacker_rank_metrics(y_node, node_prob)
    assert ranks["top1_fraction"] > 0.0
    decomposition = exact_set_decomposition(y_node, node_prob, 0.5, True)
    assert decomposition["fractions"]["exact"] == 1.0
    consistency = graph_node_consistency(y_graph, graph_prob, y_node, node_prob, 0.5, 0.5)
    assert consistency["attack"]["graph_positive_node_nonempty"] == 3
    print(SELF_TEST_MARKER)
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    paths = build_paths(args)
    paths.out.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)

    required = [
        paths.data / "metadata.json", paths.data / "x.npy", paths.data / "y_graph.npy",
        paths.data / "y_node.npy", paths.a1_model / "best_model.pt", paths.a2_model / "best_model.pt",
        paths.a2_model / "summary.json", paths.a1_val_pred, paths.a1_test_pred, paths.a1_thresholds,
        paths.evaluator, paths.freezer, paths.base_script, paths.remaining_script, paths.diagnostic_decision,
    ]
    if paths.v3_data is not None:
        required.append(paths.v3_data / "x.npy")
    require_files(required, "INPUT HARD GATE")

    script_audit = {
        "evaluator": validate_script_hash(paths.evaluator, EXPECTED_SCRIPT_HASHES["evaluator"], args.allow_script_hash_drift),
        "freezer": validate_script_hash(paths.freezer, EXPECTED_SCRIPT_HASHES["freezer"], args.allow_script_hash_drift),
        "base_analysis": validate_script_hash(paths.base_script, EXPECTED_SCRIPT_HASHES["base_analysis"], args.allow_script_hash_drift),
        "remaining_analysis": validate_script_hash(paths.remaining_script, EXPECTED_SCRIPT_HASHES["remaining_analysis"], args.allow_script_hash_drift),
    }
    checkpoint = checkpoint_audit(paths.a1_model / "best_model.pt", paths.a2_model / "best_model.pt")
    a2_summary_audit = model_summary(paths.a2_model / "summary.json")
    summary_checks = {
        "experiment_designation": a2_summary_audit.get("experiment_designation") == "a2_capacity_lift",
        "parameter_count": int(a2_summary_audit.get("parameter_count", -1)) == 2274,
        "expected_parameter_count": int(a2_summary_audit.get("expected_parameter_count", -1)) == 2274,
        "test_not_evaluated": a2_summary_audit.get("test_evaluated") is False,
        "test_threshold_not_selected": a2_summary_audit.get("test_threshold_selected") is False,
        "training_completed": int(a2_summary_audit.get("epochs_completed", 0)) > 0,
        "best_epoch_present": int(a2_summary_audit.get("best_epoch", 0)) > 0,
    }
    if not all(summary_checks.values()):
        raise RuntimeError(f"A2 training-summary audit failed: {summary_checks}")
    if validate_final_summary(paths) and not stage_forced(args, "summary") and not args.force_all:
        summary = json_load(paths.out / "analysis_summary.json")
        print(f"REUSE final_summary output={paths.out}")
        print(PASS_MARKER)
        print(json.dumps(summary, indent=2))
        return 0
    dataset_before = snapshot_dataset(paths.data)
    json_dump(paths.out / "checkpoint_architecture_audit.json", checkpoint)
    json_dump(paths.out / "a2_training_summary_audit.json", {"checks": summary_checks, "summary": a2_summary_audit})
    json_dump(paths.out / "script_hash_audit.json", script_audit)
    json_dump(paths.out / "dataset_snapshot_before.json", dataset_before)

    python = sys.executable
    validation_cmd = [
        python, str(paths.evaluator), "--mode", "validation", "--data-dir", str(paths.data),
        "--model-dir", str(paths.a2_model), "--out-dir", str(paths.validation_dir),
        "--experiment-designation", "a2_capacity_lift", "--expected-parameter-count", "2274",
        "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
        "--pin-memory", "--persistent-workers", "--prefetch-factor", str(args.prefetch_factor),
        "--threshold-start", "0.05", "--threshold-end", "0.95", "--threshold-step", "0.01",
    ]
    ensure_or_run_stage(
        "validation", paths.validation_dir, lambda: validate_validation_stage(paths), validation_cmd,
        paths.logs / "a2_validation.log", args,
    )
    freeze_cmd = [
        python, str(paths.freezer), "--validation-dir", str(paths.validation_dir),
        "--checkpoint", str(paths.a2_model / "best_model.pt"), "--out-dir", str(paths.freeze_dir),
        "--fpr-cap", "0.10", "--experiment-designation", "a2_capacity_lift",
    ]
    ensure_or_run_stage(
        "freeze", paths.freeze_dir, lambda: validate_freeze_stage(paths), freeze_cmd,
        paths.logs / "a2_freeze_fpr10.log", args,
    )
    test_cmd = [
        python, str(paths.evaluator), "--mode", "test", "--data-dir", str(paths.data),
        "--model-dir", str(paths.a2_model), "--out-dir", str(paths.test_dir),
        "--selected-thresholds", str(paths.freeze_dir / "selected_thresholds.json"),
        "--experiment-designation", "a2_capacity_lift", "--expected-parameter-count", "2274",
        "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
        "--pin-memory", "--persistent-workers", "--prefetch-factor", str(args.prefetch_factor),
    ]
    ensure_or_run_stage(
        "test", paths.test_dir, lambda: validate_test_stage(paths), test_cmd,
        paths.logs / "a2_test_transfer.log", args,
    )
    base_cmd = [
        python, str(paths.base_script), "--data-dir", str(paths.data),
        "--validation-predictions", str(paths.validation_dir / "validation_predictions.npz"),
        "--test-predictions", str(paths.test_dir / "test_predictions.npz"),
        "--selected-thresholds", str(paths.freeze_dir / "selected_thresholds.json"),
        "--diagnostic-decision", str(paths.diagnostic_decision),
        "--model-summary", str(paths.a2_model / "summary.json"),
        "--out-dir", str(paths.a2_base_analysis), "--chunk-size", "1024",
        "--feature-windows-per-run", "32", "--observability-windows-per-run", "64",
        "--matched-control-samples-per-pair", "32",
    ]
    if paths.v3_data is not None:
        base_cmd.extend(["--v3-data-dir", str(paths.v3_data), "--v3-feature-sample-windows", "32768"])
    ensure_or_run_stage(
        "base", paths.a2_base_analysis, lambda: validate_base_stage(paths), base_cmd,
        paths.logs / "a2_comprehensive_failure_analysis.log", args,
    )
    remaining_cmd = [
        python, str(paths.remaining_script), "--data-dir", str(paths.data),
        "--validation-predictions", str(paths.validation_dir / "validation_predictions.npz"),
        "--test-predictions", str(paths.test_dir / "test_predictions.npz"),
        "--selected-thresholds", str(paths.freeze_dir / "selected_thresholds.json"),
        "--base-analysis-script", str(paths.base_script), "--out-dir", str(paths.a2_remaining_analysis),
        "--count-train-cap", str(args.count_train_cap),
        "--calibration-node-cap-per-topology", str(args.calibration_node_cap_per_topology),
        "--threshold-start", "0.05", "--threshold-end", "0.95", "--threshold-step", "0.01",
        "--seed", str(args.seed),
    ]
    ensure_or_run_stage(
        "remaining", paths.a2_remaining_analysis, lambda: validate_remaining_stage(paths), remaining_cmd,
        paths.logs / "a2_remaining_decoder_diagnostics.log", args,
    )
    if args.dry_run:
        print("DRY RUN COMPLETE")
        return 0

    a2_val_path = paths.validation_dir / "validation_predictions.npz"
    a2_test_path = paths.test_dir / "test_predictions.npz"
    val_alignment = prediction_alignment(paths.a1_val_pred, a2_val_path, "validation")
    test_alignment = prediction_alignment(paths.a1_test_pred, a2_test_path, "test")
    json_dump(paths.out / "a1_a2_prediction_alignment.json", {"validation": val_alignment, "test": test_alignment})

    a1_thresholds = json_load(paths.a1_thresholds)
    a2_thresholds = json_load(paths.freeze_dir / "selected_thresholds.json")
    a1_val = load_npz(paths.a1_val_pred)
    a1_test = load_npz(paths.a1_test_pred)
    a2_val = load_npz(a2_val_path)
    a2_test = load_npz(a2_test_path)

    metrics = {
        "thresholds": {
            "a1": {"graph": float(a1_thresholds["graph_threshold"]), "node": float(a1_thresholds["node_threshold"])},
            "a2": {"graph": float(a2_thresholds["graph_threshold"]), "node": float(a2_thresholds["node_threshold"])},
        },
        "a1": {
            "validation": collect_split_metrics(a1_val, float(a1_thresholds["graph_threshold"]), float(a1_thresholds["node_threshold"])),
            "test": collect_split_metrics(a1_test, float(a1_thresholds["graph_threshold"]), float(a1_thresholds["node_threshold"])),
        },
        "a2": {
            "validation": collect_split_metrics(a2_val, float(a2_thresholds["graph_threshold"]), float(a2_thresholds["node_threshold"])),
            "test": collect_split_metrics(a2_test, float(a2_thresholds["graph_threshold"]), float(a2_thresholds["node_threshold"])),
        },
    }
    # A2 operating points: fixed 0.50, unconstrained validation-selected, and FPR<=0.10 frozen.
    a2_training_summary = model_summary(paths.a2_model / "summary.json")
    a2_unconstrained_thresholds = json_load(paths.validation_dir / "selected_thresholds.json")
    a2_unconstrained_validation = json_load(paths.validation_dir / "validation_metrics.json")
    a2_fpr10_validation = json_load(paths.freeze_dir / "validation_metrics.json")
    a2_only_metrics = {
        "thresholds": metrics["thresholds"]["a2"],
        "validation": metrics["a2"]["validation"],
        "test": metrics["a2"]["test"],
        "operating_points": {
            "fixed_0_50_validation_at_best_checkpoint": a2_training_summary.get(
                "best_validation_metrics_at_fixed_threshold_0_5", {}
            ),
            "unconstrained_validation_selected": {
                "thresholds": {
                    "graph": float(a2_unconstrained_thresholds["graph_threshold"]),
                    "node": float(a2_unconstrained_thresholds["node_threshold"]),
                },
                "validation_metrics": a2_unconstrained_validation,
                "test_transfer_metrics_recomputed_without_selection": collect_split_metrics(
                    a2_test,
                    float(a2_unconstrained_thresholds["graph_threshold"]),
                    float(a2_unconstrained_thresholds["node_threshold"]),
                ),
            },
            "fpr10_validation_frozen": {
                "thresholds": metrics["thresholds"]["a2"],
                "validation_metrics": a2_fpr10_validation,
                "test_transfer_metrics": metrics["a2"]["test"],
            },
        },
    }

    a1_remaining_decision = paths.a1_remaining_analysis / "final_decision.json"
    require_files([a1_remaining_decision, paths.a2_remaining_analysis / "final_decision.json"], "DECODER COMPARISON")
    a1_decision = json_load(a1_remaining_decision)
    a2_decision = json_load(paths.a2_remaining_analysis / "final_decision.json")
    a1_selected = a1_decision["decision"]["selected_method_test_transfer"]
    a2_selected = a2_decision["decision"]["selected_method_test_transfer"]

    a1_flat = flatten_comparison_metrics(metrics["a1"], a1_selected, 882)
    a2_flat = flatten_comparison_metrics(metrics["a2"], a2_selected, 2274)
    compare_rows = comparison_rows(a1_flat, a2_flat)

    dataset_after = snapshot_dataset(paths.data)
    dataset_unchanged = compare_dataset_snapshots(dataset_before, dataset_after)
    json_dump(paths.out / "dataset_snapshot_after.json", dataset_after)

    hard_checks = {
        "analysis_pass": True,
        "validation_only_selection": bool(
            a2_thresholds.get("test_accessed") is False
            and a2_decision["hard_checks"]["validation_only_fitting"] is True
        ),
        "test_not_used_for_selection": a2_decision["hard_checks"]["test_used_for_fit_or_selection"] is False,
        "a1_a2_sample_alignment": val_alignment["pass"] and test_alignment["pass"],
        "identical_metric_definitions": True,
        "a1_parameter_count_verified": checkpoint["a1_parameter_count"] == 882,
        "a2_parameter_count_verified": checkpoint["a2_parameter_count"] == 2274,
        "a2_training_summary_verified": all(summary_checks.values()),
        "architecture_width_only_verified": checkpoint["pass"] is True,
        "dataset_unmodified": dataset_unchanged,
        "base_failure_analysis_pass": validate_base_stage(paths),
        "remaining_decoder_analysis_pass": validate_remaining_stage(paths),
    }
    if not all(hard_checks.values()):
        raise RuntimeError(f"FINAL HARD GATE failed before report generation: {hard_checks}")

    capacity = build_capacity_verdict(a1_flat, a2_flat, hard_checks)
    a3 = build_a3_recommendation(
        a2_flat,
        capacity,
        metrics["a2"]["test"]["graph_node_consistency"],
    )
    failure_rows = failure_mode_rows(a1_flat, a2_flat)

    protected_summary_names = required_final_report_names()
    existing_summary_files = [name for name in protected_summary_names if (paths.out / name).exists()]
    if existing_summary_files and not stage_forced(args, "summary"):
        raise RuntimeError(
            "Final summary is incomplete or inconsistent and will not be overwritten automatically. "
            f"Existing files: {existing_summary_files}. Review them, then use --force-stage summary to rebuild."
        )

    if stage_forced(args, "summary"):
        protected = protected_summary_names
        for name in protected:
            path = paths.out / name
            if path.exists():
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)

    materialize_summary_outputs(paths, a2_only_metrics, a2_selected)
    write_csv(paths.out / "a1_vs_a2_metrics.csv", compare_rows)
    write_csv(paths.out / "a1_vs_a2_failure_modes.csv", failure_rows)
    json_dump(paths.out / "a1_vs_a2_capacity_verdict.json", capacity)
    json_dump(paths.out / "a3_recommendation.json", a3)
    report = build_markdown_report(checkpoint, capacity, a3, compare_rows)
    (paths.out / "complete_report.md").write_text(report, encoding="utf-8")

    required_reports = [name for name in required_final_report_names()
                        if name not in {"provenance.json", "A3_START_GATE.json", "analysis_summary.json", "artifact_manifest.csv"}]
    report_presence = {name: (paths.out / name).is_file() for name in required_reports}
    if not all(report_presence.values()):
        raise RuntimeError(f"Missing required final reports: {report_presence}")

    final_hard_checks = {
        **hard_checks,
        "all_required_reports_present": all(report_presence.values()),
        "capacity_verdict_generated": (paths.out / "a1_vs_a2_capacity_verdict.json").is_file(),
        "a3_recommendation_generated": (paths.out / "a3_recommendation.json").is_file(),
    }
    capacity["hard_checks"] = dict(final_hard_checks)
    capacity["analysis_pass"] = all(final_hard_checks.values())
    json_dump(paths.out / "a1_vs_a2_capacity_verdict.json", capacity)
    analysis_summary = {
        "analysis_pass": all(final_hard_checks.values()),
        "hard_checks": dict(final_hard_checks),
        "capacity_outcome": capacity["primary_capacity_outcome"],
        "residual_failure_modes": capacity["residual_outcome_d"],
        "a3_still_necessary": a3["a3_still_necessary"],
        "a2_regional_model_status": a3["a2_regional_model_status"],
        "test_role": "development-comparison transfer; not an independent publication holdout",
    }
    json_dump(paths.out / "analysis_summary.json", analysis_summary)

    provenance = {
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "generated_at": utc_now(),
        "paths": {key: str(value) if isinstance(value, Path) else value for key, value in paths.__dict__.items()},
        "script_hash_audit": script_audit,
        "checkpoint_audit": checkpoint,
        "a1_thresholds_sha256": sha256_file(paths.a1_thresholds),
        "a2_thresholds_sha256": sha256_file(paths.freeze_dir / "selected_thresholds.json"),
        "a1_validation_predictions_sha256": sha256_file(paths.a1_val_pred),
        "a2_validation_predictions_sha256": sha256_file(a2_val_path),
        "a1_test_predictions_sha256": sha256_file(paths.a1_test_pred),
        "a2_test_predictions_sha256": sha256_file(a2_test_path),
        "test_role": "development-comparison transfer; not an independent publication holdout",
        "neural_training_performed": False,
        "test_used_for_fit_or_selection": False,
        "dataset_modified": False,
        "hard_checks": final_hard_checks,
        "required_report_presence": report_presence,
    }
    json_dump(paths.out / "provenance.json", provenance)
    json_dump(paths.out / "A3_START_GATE.json", {
        "status": "HOLD_FOR_HUMAN_REVIEW",
        "analysis_pipeline_passed": all(final_hard_checks.values()),
        "capacity_outcome": capacity["primary_capacity_outcome"],
        "a3_recommendation_ready": True,
        "a3_approved": False,
        "required_human_actions": [
            "Review complete_report.md",
            "Review a1_vs_a2_metrics.csv",
            "Review a1_vs_a2_failure_modes.csv",
            "Review a1_vs_a2_capacity_verdict.json",
            "Review a3_recommendation.json",
            "Explicitly approve A3 in the next project stage",
        ],
    })
    write_manifest(paths.out)

    final_hard_checks["artifact_manifest_present"] = (paths.out / "artifact_manifest.csv").is_file()
    if not all(final_hard_checks.values()):
        raise RuntimeError(f"FINAL HARD GATE: FAIL: {final_hard_checks}")

    summary = {
        "analysis_pass": True,
        "capacity_outcome": capacity["primary_capacity_outcome"],
        "residual_failure_modes": capacity["residual_outcome_d"],
        "a3_still_necessary": a3["a3_still_necessary"],
        "a2_regional_model_status": a3["a2_regional_model_status"],
        "output_dir": str(paths.out),
        "a3_gate": "HOLD_FOR_HUMAN_REVIEW",
    }
    print(PASS_MARKER)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("INTERRUPTED", file=sys.stderr)
        raise
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
