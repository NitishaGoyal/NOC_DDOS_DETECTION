#!/usr/bin/env python3
"""
V4-A3.13
Frozen Local-versus-Graph Branch Attribution

This script instruments the exact frozen A3SourcePreserveModel and compares:

  full fusion:  concat(h_local, h_graph)
  local only:   concat(h_local, 0)
  graph only:   concat(0, h_graph)

The original attacker_hidden and attacker_out layers are reused unchanged.
The graph gate remains the frozen full-model graph gate, so node-branch
attribution is not confounded by changing the graph detector.

Primary evidence comes from validation. The development-comparison test split
is descriptive only and is never used to select a branch, threshold, horizon,
or architecture.

No training occurs.
No weights are modified.
No threshold search occurs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


BRANCHES = ("full", "local_only", "graph_only")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def load_file_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_a3_module(repo: Path, source_path: Path):
    scripts_root = repo / "scripts"
    for candidate in (str(repo), str(scripts_root), str(source_path.parent)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    try:
        module = importlib.import_module("v4.common.v4_a3_sourcepreserve")
        imported_path = Path(module.__file__).resolve()
        if imported_path != source_path.resolve():
            raise RuntimeError(
                f"imported unexpected model source: {imported_path}"
            )
        return module
    except Exception:
        return load_file_module(source_path, "resolved_v4_a3_sourcepreserve")


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value and all(
                isinstance(v, torch.Tensor) for v in value.values()
            ):
                return value
        if checkpoint and all(
            isinstance(v, torch.Tensor) for v in checkpoint.values()
        ):
            return checkpoint
    raise RuntimeError("model state_dict not found in checkpoint")


def topk_predictions(scores: np.ndarray, counts: np.ndarray) -> np.ndarray:
    score = np.asarray(scores)
    k_values = np.asarray(counts, dtype=np.int64)
    pred = np.zeros_like(score, dtype=np.int8)
    order = np.argsort(-score, axis=1, kind="stable")
    for i, k in enumerate(k_values):
        k = int(np.clip(k, 0, score.shape[1]))
        if k:
            pred[i, order[i, :k]] = 1
    return pred


def binary_set_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.int8)
    pred = np.asarray(y_pred, dtype=np.int8)

    tp = int(np.sum((true == 1) & (pred == 1)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    exact = np.all(true == pred, axis=1)
    empty = np.sum(pred, axis=1) == 0
    return {
        "sample_count": int(true.shape[0]),
        "node_tp": tp,
        "node_fp": fp,
        "node_fn": fn,
        "node_precision": precision,
        "node_recall": recall,
        "node_f1": f1,
        "exact_localization": float(np.mean(exact)) if exact.size else math.nan,
        "empty_prediction_rate": (
            float(np.mean(empty)) if empty.size else math.nan
        ),
        "mean_predicted_count": float(np.mean(np.sum(pred, axis=1))),
        "mean_true_count": float(np.mean(np.sum(true, axis=1))),
    }


def true_rank_summary(
    scores: np.ndarray,
    y_node: np.ndarray,
) -> dict[str, Any]:
    score = np.asarray(scores)
    truth = np.asarray(y_node) > 0.5
    ranks = (
        np.argsort(
            np.argsort(-score, axis=1, kind="stable"),
            axis=1,
        )
        + 1
    )
    flat = ranks[truth]
    true_count = np.sum(truth, axis=1)
    worst = np.zeros(score.shape[0], dtype=np.int16)
    best = np.zeros(score.shape[0], dtype=np.int16)
    for i in range(score.shape[0]):
        values = ranks[i, truth[i]]
        if values.size:
            worst[i] = int(np.max(values))
            best[i] = int(np.min(values))

    return {
        "true_attacker_instance_count": int(flat.size),
        "rank_mean": float(np.mean(flat)),
        "rank_median": float(np.median(flat)),
        "rank_p90": float(np.quantile(flat, 0.90)),
        "rank_p95": float(np.quantile(flat, 0.95)),
        "rank_top1_rate": float(np.mean(flat <= 1)),
        "rank_top2_rate": float(np.mean(flat <= 2)),
        "rank_top4_rate": float(np.mean(flat <= 4)),
        "worst_true_rank_mean": float(np.mean(worst)),
        "worst_true_rank_median": float(np.median(worst)),
        "all_true_within_top_true_count_rate": float(
            np.mean(worst <= true_count)
        ),
        "at_least_one_true_rank1_rate": float(np.mean(best == 1)),
    }


def run_boundaries(run_index: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(run_index)
    if values.size == 0:
        return []
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    ends = np.r_[starts[1:], values.size]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def same_set_persistence(
    pred: np.ndarray,
    run_index: np.ndarray,
    required: int,
) -> np.ndarray:
    values = np.asarray(pred, dtype=np.int8)
    out = np.zeros_like(values)
    for start, end in run_boundaries(run_index):
        for i in range(start + required - 1, end):
            window = values[i - required + 1 : i + 1]
            current = window[-1]
            if np.sum(current) == 0:
                continue
            if np.all(window == current):
                out[i] = current
    return out


def router_three_of_four(
    pred: np.ndarray,
    run_index: np.ndarray,
) -> np.ndarray:
    values = np.asarray(pred, dtype=np.int8)
    out = np.zeros_like(values)
    for start, end in run_boundaries(run_index):
        if end - start < 4:
            continue
        csum = np.cumsum(
            np.vstack(
                [
                    np.zeros((1, values.shape[1]), dtype=np.int64),
                    values[start:end],
                ]
            ),
            axis=0,
        )
        sums = csum[4:] - csum[:-4]
        out[start + 3 : end] = (sums >= 3).astype(np.int8)
    return out


def apply_persistence(
    pred: np.ndarray,
    run_index: np.ndarray,
    rule: str,
) -> np.ndarray:
    if rule in ("", "none", None):
        return np.asarray(pred, dtype=np.int8)
    if rule.startswith("same_set_"):
        return same_set_persistence(
            pred,
            run_index,
            int(rule.rsplit("_", 1)[1]),
        )
    if rule == "router_3_of_4":
        return router_three_of_four(pred, run_index)
    raise ValueError(f"unsupported persistence rule: {rule}")


def maximum_streak(flags: np.ndarray) -> int:
    best = 0
    current = 0
    for value in np.asarray(flags, dtype=bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def head_logits(model, h_node: torch.Tensor) -> torch.Tensor:
    hidden = F.relu(model.attacker_hidden(h_node))
    return model.attacker_out(hidden).squeeze(-1)


def generate_branch_probabilities(
    split: str,
    archive_path: Path,
    x_memmap: np.ndarray,
    output_dir: Path,
    model,
    a_hat: torch.Tensor,
    physical_mask: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Path], dict[str, Any]]:
    with np.load(archive_path, allow_pickle=True) as archive:
        global_index = np.asarray(archive["global_index"], dtype=np.int64)
        archived_node = np.asarray(archive["node_prob"], dtype=np.float32)
        archived_graph = np.asarray(archive["graph_prob"], dtype=np.float32)
        archived_count = np.asarray(archive["count_prob"], dtype=np.float32)

    branch_dir = output_dir / "branch_probabilities"
    branch_dir.mkdir(exist_ok=True)

    paths = {
        branch: branch_dir / f"{split}__{branch}__node_prob.npy"
        for branch in BRANCHES
    }
    arrays = {
        branch: np.lib.format.open_memmap(
            paths[branch],
            mode="w+",
            dtype=np.float32,
            shape=archived_node.shape,
        )
        for branch in BRANCHES
    }

    max_node_diff = 0.0
    sum_node_diff = 0.0
    max_graph_diff = 0.0
    sum_graph_diff = 0.0
    max_count_diff = 0.0
    sum_count_diff = 0.0
    element_count_node = 0
    element_count_graph = 0
    element_count_count = 0

    model.eval()
    with torch.inference_mode():
        for start in range(0, global_index.size, batch_size):
            end = min(start + batch_size, global_index.size)
            gidx = global_index[start:end]
            x_batch = torch.from_numpy(
                np.asarray(x_memmap[gidx], dtype=np.float32)
            ).to(device, non_blocking=True)

            output = model(
                x_batch,
                a_hat,
                physical_mask,
                return_intermediates=True,
            )
            h_local = output["h_local"]
            h_graph = output["h_graph"]
            zeros_local = torch.zeros_like(h_local)
            zeros_graph = torch.zeros_like(h_graph)

            logits = {
                "full": output["node_logits"],
                "local_only": head_logits(
                    model,
                    torch.cat([h_local, zeros_graph], dim=2),
                ),
                "graph_only": head_logits(
                    model,
                    torch.cat([zeros_local, h_graph], dim=2),
                ),
            }

            for branch, tensor in logits.items():
                arrays[branch][start:end] = (
                    torch.sigmoid(tensor).cpu().numpy().astype(np.float32)
                )

            full_node = arrays["full"][start:end]
            full_graph = (
                torch.sigmoid(output["graph_logits"])
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            full_count = (
                torch.softmax(output["count_logits"], dim=1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            node_diff = np.abs(full_node - archived_node[start:end])
            graph_diff = np.abs(full_graph - archived_graph[start:end])
            count_diff = np.abs(full_count - archived_count[start:end])

            max_node_diff = max(max_node_diff, float(np.max(node_diff)))
            sum_node_diff += float(np.sum(node_diff))
            element_count_node += int(node_diff.size)

            max_graph_diff = max(max_graph_diff, float(np.max(graph_diff)))
            sum_graph_diff += float(np.sum(graph_diff))
            element_count_graph += int(graph_diff.size)

            max_count_diff = max(max_count_diff, float(np.max(count_diff)))
            sum_count_diff += float(np.sum(count_diff))
            element_count_count += int(count_diff.size)

            if start == 0 or end == global_index.size or start % (batch_size * 100) == 0:
                print(
                    f"{split}: inference {end}/{global_index.size}",
                    flush=True,
                )

    for array in arrays.values():
        array.flush()

    reproduction = {
        "split": split,
        "sample_count": int(global_index.size),
        "node_probability_max_abs_diff": max_node_diff,
        "node_probability_mean_abs_diff": (
            sum_node_diff / element_count_node
        ),
        "graph_probability_max_abs_diff": max_graph_diff,
        "graph_probability_mean_abs_diff": (
            sum_graph_diff / element_count_graph
        ),
        "count_probability_max_abs_diff": max_count_diff,
        "count_probability_mean_abs_diff": (
            sum_count_diff / element_count_count
        ),
        "reproduction_tolerance": 1e-4,
        "reproduction_pass": (
            max_node_diff <= 1e-4
            and max_graph_diff <= 1e-4
            and max_count_diff <= 1e-4
        ),
    }
    return paths, reproduction


def aggregate_branch(
    helper,
    prepared,
    node_prob: np.ndarray,
    graph_prob: np.ndarray,
    horizon: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense_segments = helper.make_segments(prepared)
    segments = helper.mode_segments(
        dense_segments,
        "deoverlap",
        stride,
    )

    endpoints, node_agg = helper.aggregate_array(
        helper.logit(node_prob),
        segments,
        horizon,
        helper.uniform_weights(horizon),
    )
    graph_endpoints, graph_agg = helper.aggregate_array(
        helper.logit(graph_prob),
        segments,
        horizon,
        helper.uniform_weights(horizon),
    )
    if not np.array_equal(endpoints, graph_endpoints):
        raise RuntimeError("node/graph aggregation endpoints differ")

    return (
        endpoints,
        helper.sigmoid(node_agg),
        helper.sigmoid(graph_agg),
    )


def evaluate_branch(
    split: str,
    branch: str,
    scores: np.ndarray,
    graph_scores: np.ndarray,
    y_node: np.ndarray,
    true_count: np.ndarray,
    node_threshold: float,
    graph_threshold: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    gate = graph_scores >= graph_threshold
    threshold_pred = (scores >= node_threshold).astype(np.int8)
    threshold_pred[~gate] = 0
    oracle_pred = topk_predictions(scores, true_count)

    attack = true_count > 0
    threshold_metrics = binary_set_metrics(
        y_node[attack],
        threshold_pred[attack],
    )
    oracle_metrics = binary_set_metrics(
        y_node[attack],
        oracle_pred[attack],
    )
    rank = true_rank_summary(scores[attack], y_node[attack])

    row = {
        "split": split,
        "branch": branch,
        "attack_sample_count": int(np.sum(attack)),
        "frozen_threshold_node_precision": threshold_metrics["node_precision"],
        "frozen_threshold_node_recall": threshold_metrics["node_recall"],
        "frozen_threshold_node_f1": threshold_metrics["node_f1"],
        "frozen_threshold_exact": threshold_metrics["exact_localization"],
        "frozen_threshold_empty_rate": threshold_metrics[
            "empty_prediction_rate"
        ],
        "oracle_exact": oracle_metrics["exact_localization"],
        "oracle_node_f1": oracle_metrics["node_f1"],
        **rank,
    }
    return row, threshold_pred, oracle_pred


def count_group_rows(
    split: str,
    branch: str,
    scores: np.ndarray,
    y_node: np.ndarray,
    true_count: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for count in sorted(
        int(value)
        for value in np.unique(true_count)
        if int(value) > 0
    ):
        mask = true_count == count
        oracle = topk_predictions(scores[mask], true_count[mask])
        metrics = binary_set_metrics(y_node[mask], oracle)
        rank = true_rank_summary(scores[mask], y_node[mask])
        rows.append(
            {
                "split": split,
                "branch": branch,
                "true_count": count,
                "sample_count": int(np.sum(mask)),
                "oracle_exact": metrics["exact_localization"],
                "oracle_node_f1": metrics["node_f1"],
                **rank,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--source-resolution", type=Path, required=True)
    parser.add_argument("--source-resolution-marker", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--a3-12-lock", type=Path, required=True)
    parser.add_argument("--operational-policy", type=Path, required=True)
    parser.add_argument("--operational-policy-lock", type=Path, required=True)
    parser.add_argument("--latency-policy", type=Path, required=True)
    parser.add_argument("--latency-policy-lock", type=Path, required=True)
    parser.add_argument("--test-transfer-lock", type=Path, required=True)
    parser.add_argument("--hard-normal-csv", type=Path, required=True)
    parser.add_argument("--validation-search-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--deoverlap-stride", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    required = [
        args.repo,
        args.model_source,
        args.source_resolution,
        args.source_resolution_marker,
        args.checkpoint,
        args.data_dir,
        args.validation_predictions,
        args.test_predictions,
        args.a3_12_lock,
        args.operational_policy,
        args.operational_policy_lock,
        args.latency_policy,
        args.latency_policy_lock,
        args.test_transfer_lock,
        args.hard_normal_csv,
        args.validation_search_script,
        args.data_dir / "x.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("A3.13 FAIL: missing required paths", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(
            f"A3.13 FAIL: output exists: {args.output_dir}",
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True)

    source_resolution = load_json(args.source_resolution)
    a3_12_lock = load_json(args.a3_12_lock)
    operational = load_json(args.operational_policy)
    operational_lock = load_json(args.operational_policy_lock)
    latency = load_json(args.latency_policy)
    latency_lock = load_json(args.latency_policy_lock)
    test_lock = load_json(args.test_transfer_lock)

    checkpoint_sha = sha256_file(args.checkpoint)
    source_sha = sha256_file(args.model_source)
    val_sha = sha256_file(args.validation_predictions)
    test_sha = sha256_file(args.test_predictions)
    operational_sha = sha256_file(args.operational_policy)
    latency_sha = sha256_file(args.latency_policy)

    failures: list[str] = []
    if not source_resolution.get("source_hash_matches_checkpoint"):
        failures.append("source resolution did not pass checkpoint hash")
    if source_resolution.get("source_sha256") != source_sha:
        failures.append("resolved source hash mismatch")
    if a3_12_lock.get("checkpoint_sha256") != checkpoint_sha:
        failures.append("A3.12 checkpoint hash mismatch")
    if a3_12_lock.get("validation_predictions_sha256") != val_sha:
        failures.append("validation prediction hash mismatch")
    if a3_12_lock.get("test_predictions_sha256") != test_sha:
        failures.append("test prediction hash mismatch")
    if operational_lock.get("checkpoint_sha256") != checkpoint_sha:
        failures.append("operational checkpoint hash mismatch")
    if operational_lock.get("operational_policy_family_sha256") != operational_sha:
        failures.append("operational policy hash mismatch")
    if latency_lock.get("checkpoint_sha256") != checkpoint_sha:
        failures.append("latency-policy checkpoint hash mismatch")
    if latency_lock.get("frozen_latency_policy_family_sha256") != latency_sha:
        failures.append("latency-policy hash mismatch")
    if test_lock.get("checkpoint_sha256") != checkpoint_sha:
        failures.append("test-transfer checkpoint hash mismatch")
    if test_lock.get("test_predictions_sha256") != test_sha:
        failures.append("test-transfer prediction hash mismatch")
    if test_lock.get("policy_selection_performed_on_test") is not False:
        failures.append("test-transfer lock reports test policy selection")

    if failures:
        payload = {"status": "FAIL", "failures": failures}
        (args.output_dir / "a3_13_failure.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    recorded_source_sha = (
        checkpoint.get("source_hashes", {}).get("common")
        if isinstance(checkpoint, dict)
        else None
    )
    if recorded_source_sha != source_sha:
        raise RuntimeError(
            "checkpoint-recorded common source hash does not match model source"
        )

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print(f"device={device}", flush=True)

    model_module = load_a3_module(args.repo, args.model_source)
    model = model_module.A3SourcePreserveModel()
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.to(device)
    model.eval()

    a_hat = checkpoint["A_hat"].to(device=device, dtype=torch.float32)
    physical_mask = checkpoint["physical_valid_port_mask"].to(
        device=device,
        dtype=torch.float32,
    )

    helper = load_file_module(
        args.validation_search_script,
        "resolved_v4_a3_11_validation_search",
    )
    x_memmap = np.load(args.data_dir / "x.npy", mmap_mode="r")

    archives = {
        "validation": args.validation_predictions,
        "development_test": args.test_predictions,
    }

    branch_paths: dict[str, dict[str, Path]] = {}
    reproduction_rows: list[dict[str, Any]] = []

    for split, archive_path in archives.items():
        paths, reproduction = generate_branch_probabilities(
            split,
            archive_path,
            x_memmap,
            args.output_dir,
            model,
            a_hat,
            physical_mask,
            device,
            args.batch_size,
        )
        branch_paths[split] = paths
        reproduction_rows.append(reproduction)
        print(json.dumps(reproduction, indent=2), flush=True)

    write_csv(
        args.output_dir / "frozen_reproduction_check.csv",
        reproduction_rows,
    )
    if not all(row["reproduction_pass"] for row in reproduction_rows):
        payload = {
            "status": "HOLD",
            "reason": "frozen prediction reproduction exceeded tolerance",
            "reproduction": reproduction_rows,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (args.output_dir / "V4_A3_13_REPRODUCTION_HOLD").write_text(
            "V4_A3_13_REPRODUCTION_HOLD\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2))
        print("V4_A3_13_REPRODUCTION_HOLD")
        return 0

    candidate_policy = operational["policy_roles"][
        "candidate_localization"
    ]["policy"]

    overview_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    aggregate_cache: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}

    for split, archive_path in archives.items():
        with np.load(archive_path, allow_pickle=True) as archive:
            archived_graph = np.asarray(archive["graph_prob"])
            y_node = np.asarray(archive["y_node"])
            true_count = np.asarray(
                archive["attacker_count"],
                dtype=np.int64,
            )

        prepared = helper.prepare_data(archive_path, args.data_dir)
        horizon = int(candidate_policy["horizon"])

        for branch in BRANCHES:
            node_prob = np.load(
                branch_paths[split][branch],
                mmap_mode="r",
            )
            endpoints, scores, graph_scores = aggregate_branch(
                helper,
                prepared,
                node_prob,
                archived_graph,
                horizon,
                args.deoverlap_stride,
            )
            aggregate_cache[(split, branch, horizon)] = {
                "endpoints": endpoints,
                "scores": scores,
                "graph_scores": graph_scores,
            }

            row, _, _ = evaluate_branch(
                split,
                branch,
                scores,
                graph_scores,
                y_node[endpoints],
                true_count[endpoints],
                float(candidate_policy["node_threshold"]),
                float(candidate_policy["graph_threshold"]),
            )
            overview_rows.append(row)
            count_rows.extend(
                count_group_rows(
                    split,
                    branch,
                    scores,
                    y_node[endpoints],
                    true_count[endpoints],
                )
            )

    write_csv(
        args.output_dir / "branch_attribution_overview.csv",
        overview_rows,
    )
    write_csv(
        args.output_dir / "branch_attribution_by_true_count.csv",
        count_rows,
    )

    policies = {
        "P1_L2_CANDIDATE": candidate_policy,
        "P2_L2_CONFIRMED": operational["policy_roles"][
            "confirmed_isolation"
        ]["policy"],
        "P3_MAX_SAFETY": latency["policies"]["P3_MAX_SAFETY"],
    }

    with args.hard_normal_csv.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        hard_rows = list(csv.DictReader(handle))

    with np.load(args.test_predictions, allow_pickle=True) as archive:
        test_graph = np.asarray(archive["graph_prob"])
        test_run_index = np.asarray(archive["run_index"], dtype=np.int64)

    test_prepared = helper.prepare_data(
        args.test_predictions,
        args.data_dir,
    )
    hard_output_rows: list[dict[str, Any]] = []

    for hard in hard_rows:
        policy_name = hard["policy_name"]
        if policy_name not in policies:
            continue
        policy = policies[policy_name]
        horizon = int(policy["horizon"])
        target_run = int(hard["run_index"])
        target_router = int(hard["top_false_router"])

        branch_results: dict[str, dict[str, Any]] = {}
        for branch in BRANCHES:
            key = ("development_test", branch, horizon)
            if key not in aggregate_cache:
                node_prob = np.load(
                    branch_paths["development_test"][branch],
                    mmap_mode="r",
                )
                endpoints, scores, graph_scores = aggregate_branch(
                    helper,
                    test_prepared,
                    node_prob,
                    test_graph,
                    horizon,
                    args.deoverlap_stride,
                )
                aggregate_cache[key] = {
                    "endpoints": endpoints,
                    "scores": scores,
                    "graph_scores": graph_scores,
                }

            cached = aggregate_cache[key]
            endpoints = cached["endpoints"]
            scores = cached["scores"]
            graph_scores = cached["graph_scores"]
            run_at_endpoint = test_run_index[endpoints]
            run_mask = run_at_endpoint == target_run

            gated = (
                scores >= float(policy["node_threshold"])
            ).astype(np.int8)
            gated[
                graph_scores < float(policy["graph_threshold"])
            ] = 0
            final = apply_persistence(
                gated,
                run_at_endpoint,
                str(policy.get("persistence_rule", "none")),
            )

            target_flags = final[run_mask, target_router] == 1
            target_scores = scores[run_mask, target_router]
            ranks = (
                np.argsort(
                    np.argsort(
                        -scores[run_mask],
                        axis=1,
                        kind="stable",
                    ),
                    axis=1,
                )[:, target_router]
                + 1
            )

            branch_results[branch] = {
                "selected_count": int(np.sum(target_flags)),
                "maximum_streak": maximum_streak(target_flags),
                "score_mean_all_decisions": float(np.mean(target_scores)),
                "score_p95_all_decisions": float(
                    np.quantile(target_scores, 0.95)
                ),
                "rank_median_all_decisions": float(np.median(ranks)),
                "rank_top1_rate_all_decisions": float(np.mean(ranks == 1)),
                "selected_flags": target_flags,
                "scores": target_scores,
            }

        full_flags = branch_results["full"]["selected_flags"]
        for branch in BRANCHES:
            result = branch_results[branch]
            score_on_full_false = result["scores"][full_flags]
            hard_output_rows.append(
                {
                    "policy_name": policy_name,
                    "run_index": target_run,
                    "target_router": target_router,
                    "branch": branch,
                    "selected_count": result["selected_count"],
                    "maximum_streak": result["maximum_streak"],
                    "score_mean_all_decisions": result[
                        "score_mean_all_decisions"
                    ],
                    "score_p95_all_decisions": result[
                        "score_p95_all_decisions"
                    ],
                    "rank_median_all_decisions": result[
                        "rank_median_all_decisions"
                    ],
                    "rank_top1_rate_all_decisions": result[
                        "rank_top1_rate_all_decisions"
                    ],
                    "score_mean_on_full_false_decisions": (
                        float(np.mean(score_on_full_false))
                        if score_on_full_false.size
                        else math.nan
                    ),
                    "reported_full_selected_count": int(
                        hard["false_isolation_decisions"]
                    ),
                }
            )

        if (
            branch_results["full"]["selected_count"]
            != int(hard["false_isolation_decisions"])
        ):
            raise RuntimeError(
                f"hard-normal full reconstruction mismatch for "
                f"{policy_name} run={target_run}"
            )

    write_csv(
        args.output_dir / "hard_normal_branch_attribution.csv",
        hard_output_rows,
    )

    lookup = {
        (row["split"], row["branch"]): row
        for row in overview_rows
    }
    val_full = lookup[("validation", "full")]
    val_local = lookup[("validation", "local_only")]
    val_graph = lookup[("validation", "graph_only")]

    full_oracle = float(val_full["oracle_exact"])
    local_oracle = float(val_local["oracle_exact"])
    graph_oracle = float(val_graph["oracle_exact"])

    local_gap = full_oracle - local_oracle
    graph_gap = full_oracle - graph_oracle

    if local_gap <= 0.02 and graph_gap >= 0.05:
        decision_class = "local_branch_dominant"
        recommendation = (
            "The local branch preserves nearly all full-model ranking while "
            "the graph branch does not. Proceed to A3.15 source/transit "
            "analysis and consider adaptive suppression of graph context "
            "before A4a."
        )
    elif graph_gap <= 0.02 and local_gap >= 0.05:
        decision_class = "graph_branch_dominant"
        recommendation = (
            "The graph branch preserves nearly all full-model ranking. "
            "Inspect graph propagation errors and hard-normal router bias "
            "before changing the decoder."
        )
    elif local_gap >= 0.03 and graph_gap >= 0.03:
        decision_class = "local_graph_fusion_is_complementary"
        recommendation = (
            "Neither branch alone preserves full ranking. Keep source-"
            "preserving fusion and proceed to A4a structured null-aware set "
            "decoding, while A3.15 separately addresses hard-normal safety."
        )
    else:
        decision_class = "branches_are_partly_redundant"
        recommendation = (
            "Both branches retain substantial ranking signal. Preserve the "
            "full encoder for A4a and use A3.15 to determine whether a small "
            "source/transit gate is required."
        )

    summary = {
        "status": "PASS",
        "designation": "V4-A3.13 Frozen Local-versus-Graph Attribution",
        "device": str(device),
        "branch_definitions": {
            "full": "concat(h_local,h_graph)",
            "local_only": "concat(h_local,zeros_like(h_graph))",
            "graph_only": "concat(zeros_like(h_local),h_graph)",
        },
        "graph_gate_source": "frozen full-model graph probability",
        "validation_primary_oracle_exact": {
            "full": full_oracle,
            "local_only": local_oracle,
            "graph_only": graph_oracle,
        },
        "validation_full_minus_branch_oracle_gap": {
            "local_only": local_gap,
            "graph_only": graph_gap,
        },
        "validation_based_decision_class": decision_class,
        "validation_based_recommendation": recommendation,
        "development_test_used_for_selection": False,
        "development_test_role": (
            "descriptive frozen confirmation and hard-normal attribution only"
        ),
        "training_performed": False,
        "weights_modified": False,
        "threshold_search_performed": False,
        "provenance": {
            "checkpoint_sha256": checkpoint_sha,
            "model_source_sha256": source_sha,
            "validation_predictions_sha256": val_sha,
            "test_predictions_sha256": test_sha,
            "operational_policy_sha256": operational_sha,
            "latency_policy_sha256": latency_sha,
            "validation_search_script_sha256": sha256_file(
                args.validation_search_script
            ),
            "source_resolution_sha256": sha256_file(
                args.source_resolution
            ),
        },
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lock = {
        "status": "A3_13_LOCAL_GRAPH_ATTRIBUTION_COMPLETE",
        "checkpoint_sha256": checkpoint_sha,
        "model_source_sha256": source_sha,
        "validation_predictions_sha256": val_sha,
        "test_predictions_sha256": test_sha,
        "summary_sha256": sha256_file(summary_path),
        "development_test_used_for_selection": False,
        "training_performed": False,
        "weights_modified": False,
    }
    (args.output_dir / "A3_13_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "V4_A3_13_LOCAL_GRAPH_ATTRIBUTION_PASS").write_text(
        "V4_A3_13_LOCAL_GRAPH_ATTRIBUTION_PASS\n",
        encoding="utf-8",
    )

    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    print("V4_A3_13_LOCAL_GRAPH_ATTRIBUTION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
