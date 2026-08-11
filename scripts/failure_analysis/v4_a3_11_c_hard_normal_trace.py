#!/usr/bin/env python3
"""
V4-A3.11-C
Hard-Normal Feature and Score Trace

Descriptive closure analysis only. This script does not search, tune, promote,
or replace any policy. It reconstructs the already frozen P1/P2/P3 policies on
the development-comparison test archive and traces the exact hard-normal runs
already reported by the frozen transfer.

Main outputs:
  * target_run_inventory.csv
  * reference_run_inventory.csv
  * false_isolation_intervals.csv
  * target_decision_trace.csv
  * feature_contrast_long.csv
  * cohort_score_summary.csv
  * closure_summary.json
  * A3_11_CLOSURE_LOCK.json
  * V4_A3_11_CLOSURE_HARD_NORMAL_TRACE_PASS

The script uses:
  - frozen policy packages and locks
  - frozen test-transfer report and hard-normal CSV
  - the original test prediction archive
  - x.npy via memory mapping
  - metadata.json and edge_index.npy

It never changes thresholds or selects a development-test policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FALLBACK_FEATURE_NAMES = [
    "ifd_in_norm",
    "ifd_out_norm",
    "input_flit_count_norm",
    "output_flit_count_norm",
    "in_count_norm_local",
    "in_count_norm_north",
    "in_count_norm_east",
    "in_count_norm_south",
    "in_count_norm_west",
    "out_count_norm_local",
    "out_count_norm_north",
    "out_count_norm_east",
    "out_count_norm_south",
    "out_count_norm_west",
    "ifd_in_norm_local",
    "ifd_in_norm_north",
    "ifd_in_norm_east",
    "ifd_in_norm_south",
    "ifd_in_norm_west",
    "ifd_out_norm_local",
    "ifd_out_norm_north",
    "ifd_out_norm_east",
    "ifd_out_norm_south",
    "ifd_out_norm_west",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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
    return value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("a3_11_validation_search", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_boundaries(run_index: np.ndarray) -> list[tuple[int, int]]:
    runs = np.asarray(run_index)
    if runs.size == 0:
        return []
    starts = np.flatnonzero(np.r_[True, runs[1:] != runs[:-1]])
    ends = np.r_[starts[1:], runs.size]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def same_set_persistence(
    pred: np.ndarray,
    run_index: np.ndarray,
    required: int,
) -> np.ndarray:
    p = np.asarray(pred, dtype=np.int8)
    out = np.zeros_like(p, dtype=np.int8)
    for start, end in run_boundaries(run_index):
        if end - start < required:
            continue
        for i in range(start + required - 1, end):
            window = p[i - required + 1 : i + 1]
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
    p = np.asarray(pred, dtype=np.int8)
    out = np.zeros_like(p, dtype=np.int8)
    for start, end in run_boundaries(run_index):
        if end - start < 4:
            continue
        csum = np.cumsum(
            np.vstack(
                [np.zeros((1, p.shape[1]), dtype=np.int64), p[start:end]]
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
        required = int(rule.rsplit("_", 1)[1])
        return same_set_persistence(pred, run_index, required)
    if rule == "router_3_of_4":
        return router_three_of_four(pred, run_index)
    raise ValueError(f"unsupported persistence rule: {rule}")


def parse_nodes(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, np.ndarray)):
        out: set[int] = set()
        for item in value:
            out |= parse_nodes(item)
        return out
    if isinstance(value, (int, np.integer)):
        return {int(value)}
    text = str(value).strip()
    if not text:
        return set()
    for separator in [",", ";", " ", "|"]:
        text = text.replace(separator, "-")
    out = set()
    for token in text.split("-"):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(token))
        except ValueError:
            continue
    return out


def feature_names_from_metadata(meta: dict[str, Any], feature_count: int) -> list[str]:
    candidates = [
        meta.get("feature_cols"),
        meta.get("feature_names"),
        meta.get("features"),
    ]
    dataset = meta.get("dataset")
    if isinstance(dataset, dict):
        candidates.extend(
            [
                dataset.get("feature_cols"),
                dataset.get("feature_names"),
                dataset.get("features"),
            ]
        )
    for candidate in candidates:
        if isinstance(candidate, list) and len(candidate) == feature_count:
            return [str(x) for x in candidate]
    if feature_count == len(FALLBACK_FEATURE_NAMES):
        return FALLBACK_FEATURE_NAMES.copy()
    return [f"feature_{i}" for i in range(feature_count)]


def build_neighbors(edge_index: np.ndarray, num_nodes: int) -> dict[int, list[int]]:
    edge = np.asarray(edge_index)
    if edge.shape[0] != 2:
        raise ValueError(f"edge_index expected shape [2,E], got {edge.shape}")
    neighbors = {i: set() for i in range(num_nodes)}
    for src, dst in zip(edge[0], edge[1]):
        s = int(src)
        d = int(dst)
        if 0 <= s < num_nodes and 0 <= d < num_nodes and s != d:
            neighbors[s].add(d)
            neighbors[d].add(s)
    return {k: sorted(v) for k, v in neighbors.items()}


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


def contiguous_intervals(flags: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(flags, dtype=bool)
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(values):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            intervals.append((start, i - 1))
            start = None
    if start is not None:
        intervals.append((start, len(values) - 1))
    return intervals


def aggregate_policy(
    module,
    data,
    dense_segments,
    count_prob: np.ndarray,
    policy: dict[str, Any],
    deoverlap_stride: int,
) -> dict[str, Any]:
    horizon = int(policy["horizon"])
    segments = module.mode_segments(
        dense_segments,
        "deoverlap",
        deoverlap_stride,
    )

    graph_logit = module.logit(data.graph_prob)
    node_logit = module.logit(data.node_prob)
    count_logprob = np.log(np.clip(count_prob, 1e-8, 1.0))

    endpoints, graph_agg = module.aggregate_array(
        graph_logit,
        segments,
        horizon,
        module.uniform_weights(horizon),
    )
    node_endpoints, node_agg = module.aggregate_array(
        node_logit,
        segments,
        horizon,
        module.uniform_weights(horizon),
    )
    count_endpoints, count_agg = module.aggregate_array(
        count_logprob,
        segments,
        horizon,
        module.uniform_weights(horizon),
    )
    if not (
        np.array_equal(endpoints, node_endpoints)
        and np.array_equal(endpoints, count_endpoints)
    ):
        raise RuntimeError("aggregation endpoint mismatch")

    graph_score = module.sigmoid(graph_agg)
    node_score = module.sigmoid(node_agg)
    count_score = np.exp(count_agg - np.max(count_agg, axis=1, keepdims=True))
    count_score /= np.sum(count_score, axis=1, keepdims=True)

    raw_pred = (
        node_score >= float(policy["node_threshold"])
    ).astype(np.int8)

    graph_threshold = float(policy["graph_threshold"])
    graph_gate = graph_score >= graph_threshold
    gated_pred = raw_pred.copy()
    gated_pred[~graph_gate] = 0

    final_pred = apply_persistence(
        gated_pred,
        data.run_index[endpoints],
        str(policy.get("persistence_rule", "none")),
    )

    return {
        "endpoints": endpoints,
        "graph_score": graph_score,
        "node_score": node_score,
        "count_score": count_score,
        "graph_gate": graph_gate,
        "raw_pred": raw_pred,
        "gated_pred": gated_pred,
        "final_pred": final_pred,
    }


def rolling_feature_sequence(
    x_memmap: np.ndarray,
    dataset_run_index: np.ndarray,
    decision_global_indices: np.ndarray,
    run_index: int,
    router: int,
    horizon: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    decision_gidx = np.asarray(decision_global_indices, dtype=np.int64)
    if decision_gidx.size == 0:
        return (
            np.empty((0, x_memmap.shape[-1]), dtype=np.float32),
            np.empty((0, x_memmap.shape[-1]), dtype=np.float32),
        )

    first_base = int(decision_gidx[0]) - (horizon - 1) * stride
    last_base = int(decision_gidx[-1])
    base_gidx = np.arange(first_base, last_base + 1, stride, dtype=np.int64)

    expected = base_gidx[horizon - 1 :]
    if not np.array_equal(expected, decision_gidx):
        raise RuntimeError(
            f"feature endpoint alignment failed for run={run_index}, "
            f"router={router}, H={horizon}"
        )

    if np.any(np.asarray(dataset_run_index[base_gidx]) != run_index):
        raise RuntimeError(
            f"feature evidence crossed run boundary for run={run_index}"
        )

    block = np.asarray(
        x_memmap[base_gidx, router, :, :],
        dtype=np.float32,
    )
    sample_mean = np.mean(block, axis=1, dtype=np.float64)
    sample_peak = np.max(block, axis=1)

    csum = np.cumsum(
        np.vstack(
            [
                np.zeros((1, sample_mean.shape[1]), dtype=np.float64),
                sample_mean,
            ]
        ),
        axis=0,
    )
    rolling_mean = (csum[horizon:] - csum[:-horizon]) / float(horizon)

    rolling_peak = np.empty_like(rolling_mean, dtype=np.float32)
    for i in range(horizon - 1, sample_peak.shape[0]):
        rolling_peak[i - horizon + 1] = np.max(
            sample_peak[i - horizon + 1 : i + 1],
            axis=0,
        )

    return rolling_mean.astype(np.float32), rolling_peak.astype(np.float32)


def summarize_vector_cohort(
    case_id: str,
    cohort: str,
    mean_vectors: list[np.ndarray],
    peak_vectors: list[np.ndarray],
    feature_names: list[str],
) -> list[dict[str, Any]]:
    if not mean_vectors:
        return []
    mean_matrix = np.concatenate(mean_vectors, axis=0)
    peak_matrix = np.concatenate(peak_vectors, axis=0)
    rows: list[dict[str, Any]] = []

    for aggregation, matrix in (
        ("evidence_mean", mean_matrix),
        ("evidence_peak", peak_matrix),
    ):
        for index, name in enumerate(feature_names):
            values = matrix[:, index].astype(np.float64)
            rows.append(
                {
                    "case_id": case_id,
                    "cohort": cohort,
                    "aggregation": aggregation,
                    "feature_index": index,
                    "feature_name": name,
                    "decision_count": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "p90": float(np.quantile(values, 0.90)),
                    "p95": float(np.quantile(values, 0.95)),
                    "maximum": float(np.max(values)),
                }
            )
    return rows


def summarize_scores(
    case_id: str,
    cohort: str,
    graph_scores: list[np.ndarray],
    node_scores: list[np.ndarray],
    ranks: list[np.ndarray],
) -> dict[str, Any] | None:
    if not graph_scores:
        return None
    g = np.concatenate(graph_scores).astype(np.float64)
    n = np.concatenate(node_scores).astype(np.float64)
    r = np.concatenate(ranks).astype(np.float64)
    return {
        "case_id": case_id,
        "cohort": cohort,
        "decision_count": int(g.size),
        "graph_score_mean": float(np.mean(g)),
        "graph_score_median": float(np.median(g)),
        "graph_score_p95": float(np.quantile(g, 0.95)),
        "node_score_mean": float(np.mean(n)),
        "node_score_median": float(np.median(n)),
        "node_score_p95": float(np.quantile(n, 0.95)),
        "router_rank_mean": float(np.mean(r)),
        "router_rank_median": float(np.median(r)),
        "router_rank_best": int(np.min(r)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-search-script", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--test-transfer-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--operational-policy", type=Path, required=True)
    parser.add_argument("--operational-policy-lock", type=Path, required=True)
    parser.add_argument("--latency-policy", type=Path, required=True)
    parser.add_argument("--latency-policy-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deoverlap-stride", type=int, default=8)
    parser.add_argument("--cycles-per-epoch", type=int, default=1000)
    parser.add_argument("--reference-runs-per-target", type=int, default=5)
    args = parser.parse_args()

    hard_csv = args.test_transfer_dir / "hard_normal_runs_by_policy.csv"
    transfer_report = (
        args.test_transfer_dir / "frozen_policy_test_transfer_report.json"
    )
    transfer_lock_path = (
        args.test_transfer_dir / "FROZEN_POLICY_TEST_TRANSFER_LOCK.json"
    )
    transfer_marker = (
        args.test_transfer_dir / "V4_A3_11_FROZEN_TEST_TRANSFER_PASS"
    )
    metadata_path = args.data_dir / "metadata.json"
    x_path = args.data_dir / "x.npy"
    edge_path = args.data_dir / "edge_index.npy"
    dataset_run_index_path = args.data_dir / "run_index.npy"
    end_epoch_path = args.data_dir / "end_epoch.npy"

    required = [
        args.validation_search_script,
        args.data_dir,
        args.test_predictions,
        hard_csv,
        transfer_report,
        transfer_lock_path,
        transfer_marker,
        args.checkpoint,
        args.operational_policy,
        args.operational_policy_lock,
        args.latency_policy,
        args.latency_policy_lock,
        metadata_path,
        x_path,
        edge_path,
        dataset_run_index_path,
        end_epoch_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("A3.11-C FAIL: missing required paths", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(
            f"A3.11-C FAIL: output exists: {args.output_dir}",
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True)

    operational = load_json(args.operational_policy)
    operational_lock = load_json(args.operational_policy_lock)
    latency = load_json(args.latency_policy)
    latency_lock = load_json(args.latency_policy_lock)
    transfer_lock = load_json(transfer_lock_path)

    checkpoint_sha = sha256_file(args.checkpoint)
    test_pred_sha = sha256_file(args.test_predictions)
    operational_sha = sha256_file(args.operational_policy)
    latency_sha = sha256_file(args.latency_policy)

    failures: list[str] = []
    if operational_lock.get("operational_policy_family_sha256") != operational_sha:
        failures.append("operational policy hash mismatch")
    if latency_lock.get("frozen_latency_policy_family_sha256") != latency_sha:
        failures.append("latency policy hash mismatch")
    if transfer_lock.get("test_predictions_sha256") != test_pred_sha:
        failures.append("test prediction hash mismatch")
    for name, value in (
        ("operational lock checkpoint", operational_lock.get("checkpoint_sha256")),
        ("latency lock checkpoint", latency_lock.get("checkpoint_sha256")),
        ("transfer lock checkpoint", transfer_lock.get("checkpoint_sha256")),
    ):
        if value != checkpoint_sha:
            failures.append(f"{name} mismatch")
    if transfer_lock.get("policy_selection_performed_on_test") is not False:
        failures.append("transfer lock reports test policy selection")

    if failures:
        payload = {"status": "FAIL", "failures": failures}
        (args.output_dir / "closure_failure.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1

    with hard_csv.open(newline="", encoding="utf-8") as handle:
        hard_rows = list(csv.DictReader(handle))
    if not hard_rows:
        raise RuntimeError("hard-normal CSV is empty")

    meta = load_json(metadata_path)
    runs = meta.get("runs", meta.get("run_metadata", meta.get("completed_runs")))
    if not isinstance(runs, list):
        raise RuntimeError("metadata run list was not found")

    with np.load(args.test_predictions, allow_pickle=True) as pred:
        graph_prob = np.asarray(pred["graph_prob"])
        node_prob = np.asarray(pred["node_prob"])
        count_prob = np.asarray(pred["count_prob"])
        y_graph = np.asarray(pred["y_graph"])
        y_node = np.asarray(pred["y_node"])
        true_count = np.asarray(pred["attacker_count"])
        global_index = np.asarray(pred["global_index"])
        run_index = np.asarray(pred["run_index"])

    module = load_module(args.validation_search_script)
    data = module.prepare_data(args.test_predictions, args.data_dir)
    dense_segments = module.make_segments(data)

    policies = {
        "P1_L2_CANDIDATE": operational["policy_roles"][
            "candidate_localization"
        ]["policy"],
        "P2_L2_CONFIRMED": operational["policy_roles"][
            "confirmed_isolation"
        ]["policy"],
        "P3_MAX_SAFETY": latency["policies"]["P3_MAX_SAFETY"],
    }

    policy_cache: dict[str, dict[str, Any]] = {}
    for name, policy in policies.items():
        policy_cache[name] = aggregate_policy(
            module,
            data,
            dense_segments,
            count_prob,
            policy,
            args.deoverlap_stride,
        )

    x_memmap = np.load(x_path, mmap_mode="r")
    dataset_run_index = np.load(dataset_run_index_path, mmap_mode="r")
    dataset_end_epoch = np.load(end_epoch_path, mmap_mode="r")
    edge_index = np.load(edge_path)
    feature_names = feature_names_from_metadata(meta, x_memmap.shape[-1])
    neighbors = build_neighbors(edge_index, node_prob.shape[1])

    feature_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}

    def get_run_indices(policy_name: str, target_run: int) -> np.ndarray:
        endpoints = policy_cache[policy_name]["endpoints"]
        return np.flatnonzero(run_index[endpoints] == target_run)

    def get_feature_sequence(
        policy_name: str,
        target_run: int,
        router: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        horizon = int(policies[policy_name]["horizon"])
        key = (target_run, router, horizon)
        if key in feature_cache:
            return feature_cache[key]
        positions = get_run_indices(policy_name, target_run)
        endpoints = policy_cache[policy_name]["endpoints"][positions]
        gidx = global_index[endpoints]
        value = rolling_feature_sequence(
            x_memmap,
            dataset_run_index,
            gidx,
            target_run,
            router,
            horizon,
            args.deoverlap_stride,
        )
        feature_cache[key] = value
        return value

    target_inventory: list[dict[str, Any]] = []
    reference_inventory: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    feature_summary_rows: list[dict[str, Any]] = []
    score_summary_rows: list[dict[str, Any]] = []
    closure_cases: list[dict[str, Any]] = []

    unique_test_runs = sorted(int(x) for x in np.unique(run_index))

    for hard in hard_rows:
        policy_name = hard["policy_name"]
        target_run = int(hard["run_index"])
        router = int(hard["top_false_router"])
        if policy_name not in policies:
            continue
        if target_run >= len(runs) or not isinstance(runs[target_run], dict):
            raise RuntimeError(f"metadata missing for run {target_run}")

        run_meta = runs[target_run]
        active_cores = parse_nodes(run_meta.get("active_cores"))
        target_active = router in active_cores
        profile = str(run_meta.get("profile", ""))
        active_count = int(run_meta.get("active_count", len(active_cores)))

        case_id = f"{policy_name}__run{target_run}__router{router}"
        target_inventory.append(
            {
                "case_id": case_id,
                "policy_name": policy_name,
                "run_index": target_run,
                "run_id": run_meta.get("run_id", ""),
                "background_id": run_meta.get("background_id", ""),
                "profile": profile,
                "placement_family": run_meta.get("placement_family", ""),
                "active_cores": run_meta.get("active_cores", ""),
                "active_count": active_count,
                "seed": run_meta.get("seed", ""),
                "target_router": router,
                "target_router_active": target_active,
                "router_degree": len(neighbors[router]),
                "router_neighbors": "-".join(map(str, neighbors[router])),
                "reported_false_decisions": int(hard["false_isolation_decisions"]),
                "reported_maximum_streak": int(
                    hard["maximum_false_isolation_streak"]
                ),
            }
        )

        cache = policy_cache[policy_name]
        positions = get_run_indices(policy_name, target_run)
        endpoints = cache["endpoints"][positions]
        final_pred = cache["final_pred"][positions]
        gated_pred = cache["gated_pred"][positions]
        raw_pred = cache["raw_pred"][positions]
        graph_score = cache["graph_score"][positions]
        node_score = cache["node_score"][positions]
        count_score = cache["count_score"][positions]
        graph_gate = cache["graph_gate"][positions]

        target_selected = final_pred[:, router] == 1
        if int(np.sum(target_selected)) != int(hard["false_isolation_decisions"]):
            raise RuntimeError(
                f"reconstructed false-decision count mismatch for {case_id}: "
                f"{int(np.sum(target_selected))} vs "
                f"{hard['false_isolation_decisions']}"
            )

        decision_gidx = global_index[endpoints]
        decision_epochs = np.asarray(dataset_end_epoch[decision_gidx])
        feature_mean, feature_peak = get_feature_sequence(
            policy_name,
            target_run,
            router,
        )
        if feature_mean.shape[0] != positions.size:
            raise RuntimeError(f"feature length mismatch for {case_id}")

        target_ranks = (
            np.argsort(
                np.argsort(-node_score, axis=1),
                axis=1,
            )[:, router]
            + 1
        )

        neighbor_list = neighbors[router]
        if neighbor_list:
            neighbor_scores = node_score[:, neighbor_list]
            neighbor_max = np.max(neighbor_scores, axis=1)
            neighbor_mean = np.mean(neighbor_scores, axis=1)
        else:
            neighbor_max = np.full(positions.size, np.nan)
            neighbor_mean = np.full(positions.size, np.nan)

        interval_id_by_decision = np.full(positions.size, -1, dtype=int)
        for interval_id, (start, end) in enumerate(
            contiguous_intervals(target_selected),
            start=1,
        ):
            interval_id_by_decision[start : end + 1] = interval_id
            interval_rows.append(
                {
                    "case_id": case_id,
                    "policy_name": policy_name,
                    "run_index": target_run,
                    "run_id": run_meta.get("run_id", ""),
                    "target_router": router,
                    "interval_id": interval_id,
                    "start_decision_ordinal": start,
                    "end_decision_ordinal": end,
                    "decision_count": end - start + 1,
                    "start_end_epoch": int(decision_epochs[start]),
                    "end_end_epoch": int(decision_epochs[end]),
                    "start_global_index": int(decision_gidx[start]),
                    "end_global_index": int(decision_gidx[end]),
                    "between_first_last_cycles": (
                        (end - start)
                        * args.deoverlap_stride
                        * args.cycles_per_epoch
                    ),
                    "covered_decision_cycles": (
                        (end - start + 1)
                        * args.deoverlap_stride
                        * args.cycles_per_epoch
                    ),
                }
            )

        for i in range(positions.size):
            selected_set = "-".join(
                str(x) for x in np.flatnonzero(final_pred[i]).tolist()
            )
            row: dict[str, Any] = {
                "case_id": case_id,
                "policy_name": policy_name,
                "run_index": target_run,
                "run_id": run_meta.get("run_id", ""),
                "profile": profile,
                "placement_family": run_meta.get("placement_family", ""),
                "target_router": router,
                "target_router_active": target_active,
                "decision_ordinal": i,
                "interval_id": (
                    int(interval_id_by_decision[i])
                    if interval_id_by_decision[i] >= 0
                    else ""
                ),
                "endpoint_archive_row": int(endpoints[i]),
                "global_index": int(decision_gidx[i]),
                "end_epoch": int(decision_epochs[i]),
                "raw_graph_probability": float(graph_prob[endpoints[i]]),
                "aggregated_graph_probability": float(graph_score[i]),
                "graph_gate_open": bool(graph_gate[i]),
                "raw_router_probability": float(node_prob[endpoints[i], router]),
                "aggregated_router_probability": float(node_score[i, router]),
                "router_rank": int(target_ranks[i]),
                "node_threshold_pass": bool(raw_pred[i, router]),
                "graph_gated_router_selected": bool(gated_pred[i, router]),
                "policy_output_router_selected": bool(target_selected[i]),
                "policy_selected_set": selected_set,
                "raw_count_prediction": int(np.argmax(count_prob[endpoints[i]])),
                "aggregated_count_prediction": int(np.argmax(count_score[i])),
                "true_attacker_count": int(true_count[endpoints[i]]),
                "neighbor_score_mean": float(neighbor_mean[i]),
                "neighbor_score_max": float(neighbor_max[i]),
                "target_minus_neighbor_max": float(
                    node_score[i, router] - neighbor_max[i]
                ),
            }
            for feature_index, feature_name in enumerate(feature_names):
                row[f"mean__{feature_name}"] = float(
                    feature_mean[i, feature_index]
                )
                row[f"peak__{feature_name}"] = float(
                    feature_peak[i, feature_index]
                )
            trace_rows.append(row)

        # Deterministic matched clean normal references.
        candidate_refs: list[int] = []
        attack_refs_same_profile: list[int] = []
        attack_refs_any_profile: list[int] = []

        full_endpoints = cache["endpoints"]
        full_final_pred = cache["final_pred"]
        for candidate_run in unique_test_runs:
            if candidate_run == target_run or candidate_run >= len(runs):
                continue
            candidate_meta = runs[candidate_run]
            if not isinstance(candidate_meta, dict):
                continue
            c_active = parse_nodes(candidate_meta.get("active_cores"))
            c_profile = str(candidate_meta.get("profile", ""))
            c_label = int(candidate_meta.get("label", 0) or 0)
            c_attackers = parse_nodes(candidate_meta.get("attackers"))
            c_positions = np.flatnonzero(
                run_index[full_endpoints] == candidate_run
            )
            if c_positions.size == 0:
                continue

            if c_label == 0:
                same_profile = c_profile == profile
                same_active_count = int(
                    candidate_meta.get("active_count", len(c_active))
                ) == active_count
                same_router_activity = (router in c_active) == target_active
                target_never_selected = not np.any(
                    full_final_pred[c_positions, router] == 1
                )
                if (
                    same_profile
                    and same_active_count
                    and same_router_activity
                    and target_never_selected
                ):
                    candidate_refs.append(candidate_run)
            elif router in c_attackers:
                if c_profile == profile:
                    attack_refs_same_profile.append(candidate_run)
                else:
                    attack_refs_any_profile.append(candidate_run)

        candidate_refs.sort(key=lambda value: (abs(value - target_run), value))
        attack_refs_same_profile.sort(
            key=lambda value: (abs(value - target_run), value)
        )
        attack_refs_any_profile.sort(
            key=lambda value: (abs(value - target_run), value)
        )

        clean_refs = candidate_refs[: args.reference_runs_per_target]
        attack_refs = (
            attack_refs_same_profile + attack_refs_any_profile
        )[: args.reference_runs_per_target]

        for ref_type, ref_runs in (
            ("matched_clean_normal", clean_refs),
            ("true_attacker", attack_refs),
        ):
            for reference_run in ref_runs:
                ref_meta = runs[reference_run]
                reference_inventory.append(
                    {
                        "case_id": case_id,
                        "reference_type": ref_type,
                        "reference_run_index": reference_run,
                        "reference_run_id": ref_meta.get("run_id", ""),
                        "profile": ref_meta.get("profile", ""),
                        "placement_family": ref_meta.get(
                            "placement_family", ""
                        ),
                        "active_cores": ref_meta.get("active_cores", ""),
                        "active_count": ref_meta.get("active_count", ""),
                        "attackers": ref_meta.get("attackers", ""),
                        "target_router": router,
                        "target_router_active": (
                            router
                            in parse_nodes(ref_meta.get("active_cores"))
                        ),
                    }
                )

        cohort_mean_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
        cohort_peak_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
        cohort_graph_scores: dict[str, list[np.ndarray]] = defaultdict(list)
        cohort_node_scores: dict[str, list[np.ndarray]] = defaultdict(list)
        cohort_ranks: dict[str, list[np.ndarray]] = defaultdict(list)

        for cohort, mask in (
            ("hard_false_decisions", target_selected),
            ("hard_nonselected_decisions", ~target_selected),
        ):
            if np.any(mask):
                cohort_mean_vectors[cohort].append(feature_mean[mask])
                cohort_peak_vectors[cohort].append(feature_peak[mask])
                cohort_graph_scores[cohort].append(graph_score[mask])
                cohort_node_scores[cohort].append(node_score[mask, router])
                cohort_ranks[cohort].append(target_ranks[mask])

        for cohort, ref_runs in (
            ("matched_clean_normal_decisions", clean_refs),
            ("true_attacker_decisions", attack_refs),
        ):
            for reference_run in ref_runs:
                ref_positions = get_run_indices(policy_name, reference_run)
                if ref_positions.size == 0:
                    continue
                ref_endpoints = cache["endpoints"][ref_positions]
                ref_mean, ref_peak = get_feature_sequence(
                    policy_name,
                    reference_run,
                    router,
                )
                ref_graph = cache["graph_score"][ref_positions]
                ref_node = cache["node_score"][ref_positions, router]
                ref_rank = (
                    np.argsort(
                        np.argsort(
                            -cache["node_score"][ref_positions],
                            axis=1,
                        ),
                        axis=1,
                    )[:, router]
                    + 1
                )
                if cohort == "true_attacker_decisions":
                    keep = y_node[ref_endpoints, router] == 1
                else:
                    keep = np.ones(ref_positions.size, dtype=bool)
                if not np.any(keep):
                    continue
                cohort_mean_vectors[cohort].append(ref_mean[keep])
                cohort_peak_vectors[cohort].append(ref_peak[keep])
                cohort_graph_scores[cohort].append(ref_graph[keep])
                cohort_node_scores[cohort].append(ref_node[keep])
                cohort_ranks[cohort].append(ref_rank[keep])

        for cohort in sorted(cohort_mean_vectors):
            feature_summary_rows.extend(
                summarize_vector_cohort(
                    case_id,
                    cohort,
                    cohort_mean_vectors[cohort],
                    cohort_peak_vectors[cohort],
                    feature_names,
                )
            )
            score_row = summarize_scores(
                case_id,
                cohort,
                cohort_graph_scores[cohort],
                cohort_node_scores[cohort],
                cohort_ranks[cohort],
            )
            if score_row is not None:
                score_summary_rows.append(score_row)

        closure_cases.append(
            {
                "case_id": case_id,
                "policy_name": policy_name,
                "run_index": target_run,
                "run_id": run_meta.get("run_id", ""),
                "profile": profile,
                "placement_family": run_meta.get("placement_family", ""),
                "target_router": router,
                "target_router_active": target_active,
                "false_decision_count": int(np.sum(target_selected)),
                "false_interval_count": len(
                    contiguous_intervals(target_selected)
                ),
                "maximum_false_streak": max(
                    (
                        end - start + 1
                        for start, end in contiguous_intervals(target_selected)
                    ),
                    default=0,
                ),
                "matched_clean_reference_runs": clean_refs,
                "true_attacker_reference_runs": attack_refs,
            }
        )

    write_csv(args.output_dir / "target_run_inventory.csv", target_inventory)
    write_csv(
        args.output_dir / "reference_run_inventory.csv",
        reference_inventory,
    )
    write_csv(
        args.output_dir / "false_isolation_intervals.csv",
        interval_rows,
    )
    write_csv(
        args.output_dir / "target_decision_trace.csv",
        trace_rows,
    )
    write_csv(
        args.output_dir / "feature_contrast_long.csv",
        feature_summary_rows,
    )
    write_csv(
        args.output_dir / "cohort_score_summary.csv",
        score_summary_rows,
    )

    persistent_cases = [
        case for case in closure_cases if case["maximum_false_streak"] >= 4
    ]
    persistent_routers = sorted(
        {case["target_router"] for case in persistent_cases}
    )
    active_states = sorted(
        {bool(case["target_router_active"]) for case in persistent_cases}
    )

    if len(persistent_routers) == 1 and active_states == [False, True]:
        structural_classification = (
            "mixed legitimate-source and transit-hotspot ambiguity "
            "concentrated on one router"
        )
    elif len(persistent_routers) == 1:
        structural_classification = (
            "persistent ambiguity concentrated on one router"
        )
    else:
        structural_classification = (
            "persistent ambiguity distributed across routers"
        )

    provenance = {
        "policy_selection_performed_on_test": False,
        "analysis_role": "descriptive closure only",
        "development_test_role": "development comparison; not publication holdout",
        "checkpoint_sha256": checkpoint_sha,
        "test_predictions_sha256": test_pred_sha,
        "hard_normal_csv_sha256": sha256_file(hard_csv),
        "frozen_transfer_report_sha256": sha256_file(transfer_report),
        "operational_policy_sha256": operational_sha,
        "latency_policy_sha256": latency_sha,
        "x_npy_path": str(x_path),
        "metadata_sha256": sha256_file(metadata_path),
        "validation_search_script_sha256": sha256_file(
            args.validation_search_script
        ),
    }

    summary = {
        "status": "PASS",
        "designation": "V4-A3.11-C Hard-Normal Feature and Score Trace",
        "case_count": len(closure_cases),
        "cases": closure_cases,
        "persistent_case_count": len(persistent_cases),
        "persistent_target_routers": persistent_routers,
        "structural_classification": structural_classification,
        "local_vs_graph_attribution_available": False,
        "local_vs_graph_note": (
            "The prediction archive contains final node scores only. "
            "Attributing an error to the local or graph branch requires "
            "checkpoint instrumentation in A3.13."
        ),
        "recommended_next_stage": (
            "A3.12 cardinality/ranking decomposition, followed by A3.13 "
            "local-versus-graph instrumentation and A3.15 directional "
            "source/transit analysis."
        ),
        "provenance": provenance,
    }

    summary_path = args.output_dir / "closure_summary.json"
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lock = {
        "status": "A3_11_CLOSURE_HARD_NORMAL_TRACE_COMPLETE",
        "policy_selection_performed_on_test": False,
        "checkpoint_sha256": checkpoint_sha,
        "test_predictions_sha256": test_pred_sha,
        "closure_summary_sha256": sha256_file(summary_path),
        "target_decision_trace_sha256": sha256_file(
            args.output_dir / "target_decision_trace.csv"
        ),
        "feature_contrast_sha256": sha256_file(
            args.output_dir / "feature_contrast_long.csv"
        ),
    }
    (args.output_dir / "A3_11_CLOSURE_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    marker = args.output_dir / "V4_A3_11_CLOSURE_HARD_NORMAL_TRACE_PASS"
    marker.write_text(
        "V4_A3_11_CLOSURE_HARD_NORMAL_TRACE_PASS\n",
        encoding="utf-8",
    )

    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    print("V4_A3_11_CLOSURE_HARD_NORMAL_TRACE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
