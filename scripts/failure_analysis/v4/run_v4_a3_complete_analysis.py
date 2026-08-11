#!/usr/bin/env python3
"""Complete V4-A3 failure analysis, A2 comparison, acceptance gate, and RTL export.

Analyze mode consumes validation/test predictions created by the A3 evaluator.
All selection artifacts must be validation-only. The test split is treated as a
reused development comparison. The script also invokes the existing generic V4
comprehensive and remaining-diagnostics scripts so A3 receives the same depth
of run/scenario/shortcut analysis as A1/A2, then adds explicit count-head,
count-conditioned top-k, A2 comparison, and hardware-aware acceptance reports.

Export mode is blocked unless analyze mode classified A3 as the primary model.
It exports floating-point artifacts and golden vectors only; it does not perform
quantization or RTL conversion.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

REPO_FROM_SCRIPT = Path(__file__).resolve().parents[3]
COMMON_DIR = REPO_FROM_SCRIPT / "scripts" / "v4" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from v4_a3_sourcepreserve import (  # noqa: E402
    A3SourcePreserveModel,
    COUNT_CLASSES,
    EXPECTED_PARAMETER_COUNT,
    EXPERIMENT_DESIGNATION,
    MODEL_NAME,
    NUM_ROUTERS,
    PORT_ORDER,
    REGIONAL_EMBEDDING_DIM,
    atomic_csv_dump,
    atomic_json_dump,
    binary_metrics_numpy,
    build_normalized_adjacency,
    build_physical_valid_port_mask,
    estimate_macs_per_sample,
    feature_order,
    load_metadata,
    model_signature,
    node_metrics_from_prediction,
    numpy_topk_decode,
    sha256_file,
)

ANALYSIS_PASS = "V4_A3_COMPLETE_FAILURE_ANALYSIS_AND_A2_COMPARISON_PASS"
EXPORT_PASS = "V4_A3_RTL_FLOAT_EXPORT_PASS"
SELF_TEST_PASS = "V4_A3_COMPLETE_ANALYSIS_SELF_TEST_PASS"
CORNERS = np.asarray([0, 3, 12, 15], dtype=np.int64)
EDGES = np.asarray([1, 2, 4, 7, 8, 11, 13, 14], dtype=np.int64)
INTERIORS = np.asarray([5, 6, 9, 10], dtype=np.int64)
TOPOLOGY = {"corner": CORNERS, "edge": EDGES, "interior": INTERIORS}


def require_files(paths: Iterable[Path], label: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label}:\n  " + "\n  ".join(missing))


def ensure_new(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    entries = list(path.iterdir())
    if entries:
        raise RuntimeError(f"Output directory must be empty: {path}; entries={[e.name for e in entries[:20]]}")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_prediction_archive(path: Path, split: str) -> dict[str, np.ndarray]:
    prediction = load_npz(path)
    required = {"graph_prob", "node_prob", "count_prob", "y_graph", "y_node", "attacker_count", "global_index", "run_index", "split_id"}
    missing = sorted(required - set(prediction))
    if missing:
        raise RuntimeError(f"{path} missing keys {missing}")
    n = prediction["y_graph"].shape[0]
    checks = {
        "graph": prediction["graph_prob"].shape == (n,),
        "node": prediction["node_prob"].shape == (n, NUM_ROUTERS),
        "count": prediction["count_prob"].shape == (n, COUNT_CLASSES),
        "y_node": prediction["y_node"].shape == (n, NUM_ROUTERS),
        "global_index": prediction["global_index"].shape == (n,),
        "run_index": prediction["run_index"].shape == (n,),
        "finite_graph": bool(np.isfinite(prediction["graph_prob"]).all()),
        "finite_node": bool(np.isfinite(prediction["node_prob"]).all()),
        "finite_count": bool(np.isfinite(prediction["count_prob"]).all()),
        "count_prob_sum": bool(np.allclose(prediction["count_prob"].sum(axis=1), 1.0, atol=2e-4)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"{split} prediction audit failed: {checks}")
    order = np.argsort(prediction["global_index"], kind="stable")
    return {key: value[order] for key, value in prediction.items()}


def align_predictions(validation: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> dict[str, Any]:
    overlap = np.intersect1d(validation["global_index"], test["global_index"], assume_unique=True)
    return {
        "validation_unique": len(np.unique(validation["global_index"])) == len(validation["global_index"]),
        "test_unique": len(np.unique(test["global_index"])) == len(test["global_index"]),
        "validation_test_overlap": int(overlap.size),
        "pass": bool(overlap.size == 0),
    }


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def sklearn_probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    except Exception as error:
        return {"auroc": float("nan"), "average_precision": float("nan"), "brier": float("nan"), "sklearn_error": str(error)}
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    prob = np.asarray(y_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(truth)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (prob >= lower) & (prob < upper if upper < 1.0 else prob <= upper)
        if not mask.any():
            continue
        accuracy = float(truth[mask].mean())
        confidence = float(prob[mask].mean())
        ece += float(mask.sum()) / max(total, 1) * abs(accuracy - confidence)
    return ece


def manhattan(a: int, b: int) -> int:
    ar, ac = divmod(a, 4)
    br, bc = divmod(b, 4)
    return abs(ar - br) + abs(ac - bc)


def decode_sets(prediction: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    y_node = prediction["y_node"] >= 0.5
    count_pred = np.argmax(prediction["count_prob"], axis=1).astype(np.int64)
    learned = numpy_topk_decode(prediction["node_prob"], count_pred)
    true_count = y_node.sum(axis=1).astype(np.int64)
    oracle = numpy_topk_decode(prediction["node_prob"], true_count)
    return {
        "truth": y_node,
        "count_pred": count_pred,
        "true_count": true_count,
        "learned": learned,
        "oracle": oracle,
    }


def rank_statistics(y_node: np.ndarray, node_prob: np.ndarray) -> dict[str, Any]:
    truth = y_node >= 0.5
    ranks: list[int] = []
    all_true_top1 = 0
    all_true_top2 = 0
    all_true_top4 = 0
    attack_samples = 0
    rows: list[dict[str, Any]] = []
    for sample in range(len(truth)):
        attackers = np.flatnonzero(truth[sample])
        if attackers.size == 0:
            continue
        attack_samples += 1
        order = np.argsort(-node_prob[sample], kind="stable")
        inverse = np.empty(NUM_ROUTERS, dtype=np.int64)
        inverse[order] = np.arange(1, NUM_ROUTERS + 1)
        sample_ranks = inverse[attackers]
        ranks.extend(sample_ranks.tolist())
        all_true_top1 += int(np.all(sample_ranks <= 1))
        all_true_top2 += int(np.all(sample_ranks <= 2))
        all_true_top4 += int(np.all(sample_ranks <= 4))
        rows.append(
            {
                "sample_position": sample,
                "attacker_count": int(attackers.size),
                "attacker_ranks": "-".join(map(str, sample_ranks.tolist())),
                "best_rank": int(sample_ranks.min()),
                "worst_rank": int(sample_ranks.max()),
                "mean_rank": float(sample_ranks.mean()),
            }
        )
    array = np.asarray(ranks, dtype=np.float64)
    return {
        "summary": {
            "attacker_instances": int(len(ranks)),
            "attack_samples": attack_samples,
            "mean_rank": float(array.mean()) if array.size else float("nan"),
            "median_rank": float(np.median(array)) if array.size else float("nan"),
            "p90_rank": float(np.quantile(array, 0.90)) if array.size else float("nan"),
            "all_attackers_top1_fraction": safe_div(all_true_top1, attack_samples),
            "all_attackers_top2_fraction": safe_div(all_true_top2, attack_samples),
            "all_attackers_top4_fraction": safe_div(all_true_top4, attack_samples),
        },
        "rows": rows,
    }


def fp_distance_statistics(y_node: np.ndarray, node_pred: np.ndarray) -> dict[str, Any]:
    truth = y_node >= 0.5
    pred = node_pred.astype(bool)
    distances: list[int] = []
    rows: list[dict[str, Any]] = []
    for sample in range(len(truth)):
        attackers = np.flatnonzero(truth[sample])
        false_positives = np.flatnonzero(pred[sample] & ~truth[sample])
        if attackers.size == 0:
            continue
        for router in false_positives:
            distance = min(manhattan(int(router), int(attacker)) for attacker in attackers)
            distances.append(distance)
            rows.append(
                {
                    "sample_position": sample,
                    "false_positive_router": int(router),
                    "distance_to_nearest_attacker": int(distance),
                }
            )
    count = len(distances)
    return {
        "summary": {
            "false_positive_instances": count,
            "one_hop_fraction": safe_div(sum(distance == 1 for distance in distances), count),
            "within_two_hops_fraction": safe_div(sum(distance <= 2 for distance in distances), count),
            "distance_histogram": dict(sorted(Counter(distances).items())),
        },
        "rows": rows,
    }


def exact_set_decomposition(y_node: np.ndarray, node_pred: np.ndarray) -> list[dict[str, Any]]:
    truth = y_node >= 0.5
    pred = node_pred.astype(bool)
    categories = Counter()
    for true_set, predicted_set in zip(truth, pred):
        true_count = int(true_set.sum())
        pred_count = int(predicted_set.sum())
        true_found = int(np.sum(true_set & predicted_set))
        extras = int(np.sum(~true_set & predicted_set))
        misses = int(np.sum(true_set & ~predicted_set))
        if misses == 0 and extras == 0:
            category = "exact"
        elif pred_count == 0 and true_count > 0:
            category = "empty_prediction"
        elif true_found == 0 and true_count > 0:
            category = "no_true_attacker_found"
        elif extras == 0 and misses > 0:
            category = "strict_subset_only"
        elif misses == 0 and extras > 0:
            category = "all_true_plus_extras"
        else:
            category = "mixed_misses_and_extras"
        categories[category] += 1
    total = len(truth)
    return [
        {"category": category, "count": int(count), "fraction": safe_div(count, total)}
        for category, count in sorted(categories.items())
    ]


def topology_metrics(y_node: np.ndarray, node_pred: np.ndarray) -> list[dict[str, Any]]:
    truth = y_node >= 0.5
    pred = node_pred.astype(bool)
    rows = []
    for name, routers in TOPOLOGY.items():
        metric = node_metrics_from_prediction(truth[:, routers], pred[:, routers])
        rows.append({"topology": name, "router_count": int(len(routers)), **metric})
    return rows


def per_router_metrics(y_node: np.ndarray, node_pred: np.ndarray) -> list[dict[str, Any]]:
    truth = y_node >= 0.5
    pred = node_pred.astype(bool)
    rows = []
    for router in range(NUM_ROUTERS):
        metric = node_metrics_from_prediction(truth[:, [router]], pred[:, [router]])
        topology = "corner" if router in CORNERS else "edge" if router in EDGES else "interior"
        rows.append({"router": router, "topology": topology, **metric})
    return rows


def count_diagnostics(prediction: Mapping[str, np.ndarray], decoded: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_graph = prediction["y_graph"] >= 0.5
    true_count = decoded["true_count"]
    count_pred = decoded["count_pred"]
    attack = y_graph
    confusion = np.zeros((COUNT_CLASSES, COUNT_CLASSES), dtype=np.int64)
    for true_value, pred_value in zip(true_count, count_pred):
        confusion[int(true_value), int(pred_value)] += 1
    rows = [
        {"true_count": true_value, "predicted_count": pred_value, "samples": int(confusion[true_value, pred_value])}
        for true_value in range(COUNT_CLASSES)
        for pred_value in range(COUNT_CLASSES)
    ]
    return {
        "overall_accuracy": float(np.mean(true_count == count_pred)),
        "attack_accuracy": float(np.mean(true_count[attack] == count_pred[attack])),
        "attack_mae": float(np.mean(np.abs(true_count[attack] - count_pred[attack]))),
        "attack_empty_prediction_fraction": float(np.mean(count_pred[attack] == 0)),
        "mean_true_count_attack": float(np.mean(true_count[attack])),
        "mean_predicted_count_attack": float(np.mean(count_pred[attack])),
        "true_histogram": np.bincount(true_count, minlength=COUNT_CLASSES).tolist(),
        "predicted_histogram": np.bincount(count_pred, minlength=COUNT_CLASSES).tolist(),
    }, rows


def graph_node_consistency(prediction: Mapping[str, np.ndarray], decoded: Mapping[str, np.ndarray], graph_threshold: float) -> list[dict[str, Any]]:
    graph_pred = prediction["graph_prob"] >= graph_threshold
    node_nonempty = decoded["count_pred"] > 0
    graph_truth = prediction["y_graph"] >= 0.5
    rows = []
    for truth_name, truth_mask in (("normal", ~graph_truth), ("attack", graph_truth)):
        total = int(truth_mask.sum())
        for graph_value, graph_name in ((False, "graph_negative"), (True, "graph_positive")):
            for node_value, node_name in ((False, "node_empty"), (True, "node_nonempty")):
                count = int(np.sum(truth_mask & (graph_pred == graph_value) & (node_nonempty == node_value)))
                rows.append(
                    {
                        "truth": truth_name,
                        "graph_state": graph_name,
                        "node_state": node_name,
                        "count": count,
                        "fraction_within_truth": safe_div(count, total),
                    }
                )
    return rows


def per_attacker_count(prediction: Mapping[str, np.ndarray], decoded: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for count in range(COUNT_CLASSES):
        mask = decoded["true_count"] == count
        if not mask.any():
            continue
        learned = node_metrics_from_prediction(decoded["truth"][mask], decoded["learned"][mask])
        oracle = node_metrics_from_prediction(decoded["truth"][mask], decoded["oracle"][mask])
        rows.append(
            {
                "attacker_count": count,
                "samples": int(mask.sum()),
                "learned_node_f1": learned["node_f1"],
                "learned_exact": learned["exact_localization"],
                "oracle_node_f1": oracle["node_f1"],
                "oracle_exact": oracle["exact_localization"],
                "count_accuracy": float(np.mean(decoded["count_pred"][mask] == count)),
                "empty_prediction_fraction": float(np.mean(decoded["count_pred"][mask] == 0)),
            }
        )
    return rows


def run_subprocess(command: Sequence[str], log_path: Path) -> None:
    print("RUN:", " ".join(subprocess.list2cmdline([item]) for item in command))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        status = process.wait()
    if status != 0:
        raise RuntimeError(f"Subprocess failed status={status}; log={log_path}")


def run_generic_analyses(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    base_script = Path(args.base_analysis_script).resolve() if args.base_analysis_script else repo / "scripts/failure_analysis/v4/analyze_v4_a1_failures.py"
    remaining_script = Path(args.remaining_analysis_script).resolve() if args.remaining_analysis_script else repo / "scripts/failure_analysis/v4/analyze_v4_a1_remaining_diagnostics.py"
    require_files([base_script, remaining_script], "generic analysis scripts")
    base_out = out / "generic_comprehensive"
    remaining_out = out / "generic_remaining"
    logs = out / "logs"
    if not base_out.exists():
        command = [
            sys.executable,
            str(base_script),
            "--data-dir",
            str(Path(args.data_dir).resolve()),
            "--validation-predictions",
            str(Path(args.validation_dir).resolve() / "validation_predictions.npz"),
            "--test-predictions",
            str(Path(args.test_dir).resolve() / "test_predictions.npz"),
            "--selected-thresholds",
            str(Path(args.validation_dir).resolve() / "selected_thresholds.json"),
            "--out-dir",
            str(base_out),
            "--model-summary",
            str(Path(args.model_dir).resolve() / "summary.json"),
        ]
        if args.v3_data_dir:
            command.extend(["--v3-data-dir", str(Path(args.v3_data_dir).resolve())])
        run_subprocess(command, logs / "generic_comprehensive.log")
    if not remaining_out.exists():
        command = [
            sys.executable,
            str(remaining_script),
            "--data-dir",
            str(Path(args.data_dir).resolve()),
            "--validation-predictions",
            str(Path(args.validation_dir).resolve() / "validation_predictions.npz"),
            "--test-predictions",
            str(Path(args.test_dir).resolve() / "test_predictions.npz"),
            "--selected-thresholds",
            str(Path(args.validation_dir).resolve() / "selected_thresholds.json"),
            "--out-dir",
            str(remaining_out),
            "--base-analysis-script",
            str(base_script),
            "--seed",
            str(args.seed),
        ]
        run_subprocess(command, logs / "generic_remaining.log")
    return {
        "base_script": str(base_script),
        "base_script_sha256": sha256_file(base_script),
        "remaining_script": str(remaining_script),
        "remaining_script_sha256": sha256_file(remaining_script),
        "base_output": str(base_out),
        "remaining_output": str(remaining_out),
        "base_output_exists": base_out.is_dir(),
        "remaining_output_exists": remaining_out.is_dir(),
    }


def load_a2_values(a2_analysis_dir: Path) -> dict[str, float]:
    path = a2_analysis_dir / "a1_vs_a2_metrics.csv"
    require_files([path], "A2 comparison")
    values: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[row["metric"]] = float(row["a2_value"])
    return values


def build_metric_summary(
    prediction: Mapping[str, np.ndarray],
    decoded: Mapping[str, np.ndarray],
    graph_threshold: float,
) -> dict[str, Any]:
    y_graph = prediction["y_graph"] >= 0.5
    learned_overall = node_metrics_from_prediction(decoded["truth"], decoded["learned"])
    learned_attack = node_metrics_from_prediction(decoded["truth"][y_graph], decoded["learned"][y_graph])
    oracle_attack = node_metrics_from_prediction(decoded["truth"][y_graph], decoded["oracle"][y_graph])
    graph = binary_metrics_numpy(prediction["y_graph"], prediction["graph_prob"], graph_threshold)
    count, _ = count_diagnostics(prediction, decoded)
    ranks = rank_statistics(prediction["y_node"], prediction["node_prob"])["summary"]
    fp = fp_distance_statistics(prediction["y_node"], decoded["learned"])["summary"]
    topology = {row["topology"]: row for row in topology_metrics(prediction["y_node"], decoded["learned"])}
    probability = sklearn_probability_metrics(prediction["y_graph"], prediction["graph_prob"])
    probability["ece"] = expected_calibration_error(prediction["y_graph"], prediction["graph_prob"])
    return {
        "graph": graph,
        "graph_probability": probability,
        "learned_overall": learned_overall,
        "learned_attack": learned_attack,
        "oracle_attack": oracle_attack,
        "count": count,
        "ranks": ranks,
        "fp_distance": fp,
        "topology": topology,
        "corner_interior_gap": topology["interior"]["node_f1"] - topology["corner"]["node_f1"],
    }


def comparison_rows(a2: Mapping[str, float], a3: Mapping[str, float]) -> list[dict[str, Any]]:
    lower_is_better = {
        "test_graph_fpr",
        "count_mae",
        "true_attacker_mean_rank",
        "true_attacker_median_rank",
        "attack_empty_prediction_fraction",
        "one_hop_fp_fraction",
        "within_two_hops_fp_fraction",
        "corner_interior_f1_gap",
        "validation_to_test_graph_f1_gap_abs",
        "validation_to_test_attack_node_f1_gap_abs",
        "parameter_count",
    }
    interpretations = {
        "graph_f1_at_fpr10": "Detection quality under validation-frozen FPR constraint",
        "test_graph_fpr": "Development-comparison false-alarm rate",
        "graph_recall": "Attack-window detection recall",
        "attack_node_f1": "Count-conditioned attacker-node quality on attack windows",
        "attack_exact_localization": "Exact attacker set using explicit count head",
        "attacker_count_accuracy": "Explicit five-class count-head accuracy",
        "count_mae": "Absolute attacker-cardinality error",
        "oracle_count_attack_exact": "Source ranking with true cardinality",
        "learned_decoder_attack_exact": "Explicit count-head top-k exact localization",
        "true_attacker_mean_rank": "Average rank of true attackers",
        "true_attacker_median_rank": "Median rank of true attackers",
        "attack_empty_prediction_fraction": "Attack windows decoded as empty",
        "one_hop_fp_fraction": "False positives adjacent to an attacker",
        "within_two_hops_fp_fraction": "False positives within two hops",
        "corner_node_f1": "Corner-router localization F1",
        "edge_node_f1": "Edge-router localization F1",
        "interior_node_f1": "Interior-router localization F1",
        "corner_interior_f1_gap": "Topology bias; smaller is better",
        "validation_to_test_graph_f1_gap_abs": "Graph generalization gap",
        "validation_to_test_attack_node_f1_gap_abs": "Localization generalization gap",
        "parameter_count": "Weight-storage/compute proxy",
    }
    rows = []
    for metric, a2_value in a2.items():
        if metric not in a3:
            continue
        a3_value = float(a3[metric])
        difference = a3_value - a2_value
        lower = metric in lower_is_better
        if math.isclose(a3_value, a2_value, rel_tol=0.0, abs_tol=1e-12):
            winner = "tie"
        elif (a3_value < a2_value) if lower else (a3_value > a2_value):
            winner = "A3"
        else:
            winner = "A2"
        rows.append(
            {
                "metric": metric,
                "a2_value": a2_value,
                "a3_value": a3_value,
                "absolute_difference_a3_minus_a2": difference,
                "relative_difference": safe_div(difference, abs(a2_value)),
                "winner": winner,
                "higher_is_better": not lower,
                "interpretation": interpretations.get(metric, ""),
            }
        )
    return rows


def acceptance_decision(a2: Mapping[str, float], a3: Mapping[str, float]) -> dict[str, Any]:
    checks = {
        "graph_fpr_within_0_10": a3["test_graph_fpr"] <= 0.10 + 1e-12,
        "graph_f1_no_material_regression": a3["graph_f1_at_fpr10"] >= a2["graph_f1_at_fpr10"] - 0.015,
        "attack_node_f1_no_material_regression": a3["attack_node_f1"] >= a2["attack_node_f1"] - 0.01,
        "attack_exact_improves_2pp": a3["attack_exact_localization"] >= a2["attack_exact_localization"] + 0.02,
        "oracle_exact_improves_2pp": a3["oracle_count_attack_exact"] >= a2["oracle_count_attack_exact"] + 0.02,
        "count_improves": (
            a3["attacker_count_accuracy"] >= a2["attacker_count_accuracy"] + 0.02
            or a3["count_mae"] <= a2["count_mae"] - 0.05
        ),
        "nearby_fp_improves": (
            a3["one_hop_fp_fraction"] <= a2["one_hop_fp_fraction"] - 0.03
            or a3["within_two_hops_fp_fraction"] <= a2["within_two_hops_fp_fraction"] - 0.03
        ),
        "topology_improves": (
            a3["corner_node_f1"] >= a2["corner_node_f1"] + 0.02
            or a3["corner_interior_f1_gap"] <= a2["corner_interior_f1_gap"] - 0.03
        ),
        "hardware_budget": a3["parameter_count"] <= 3500,
    }
    detection = checks["graph_fpr_within_0_10"] and checks["graph_f1_no_material_regression"]
    structural_core = checks["attack_exact_improves_2pp"] and checks["oracle_exact_improves_2pp"]
    secondary_count = sum(
        bool(checks[name])
        for name in (
            "attack_node_f1_no_material_regression",
            "count_improves",
            "nearby_fp_improves",
            "topology_improves",
            "hardware_budget",
        )
    )
    if detection and structural_core and secondary_count >= 4:
        classification = "accepted as primary model"
    elif structural_core and checks["hardware_budget"]:
        classification = "accepted as complementary localization model"
    elif any(
        checks[name]
        for name in (
            "attack_exact_improves_2pp",
            "oracle_exact_improves_2pp",
            "count_improves",
            "nearby_fp_improves",
            "topology_improves",
        )
    ):
        classification = "diagnostic improvement only"
    else:
        classification = "rejected in favour of A2"
    return {
        "classification": classification,
        "checks": checks,
        "detection_gate": detection,
        "structural_localization_gate": structural_core,
        "secondary_checks_passed": secondary_count,
        "decision_rule": {
            "primary": "detection gate + >=2pp attack exact + >=2pp oracle exact + at least four of five secondary checks",
            "complementary": "structural localization gate and hardware budget, but primary detection/secondary gate incomplete",
            "diagnostic": "at least one targeted failure improves without an overall deployable trade-off",
            "rejected": "no targeted failure improves materially",
        },
        "test_role": "development comparison, not independent publication holdout",
        "v4_shortcut_remains_publication_blocker": True,
    }


def analyze(args: argparse.Namespace) -> int:
    out = Path(args.out_dir).resolve()
    ensure_new(out)
    data_dir = Path(args.data_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    val_dir = Path(args.validation_dir).resolve()
    test_dir = Path(args.test_dir).resolve()
    a2_dir = Path(args.a2_analysis_dir).resolve()
    required = [
        data_dir / "metadata.json",
        data_dir / "x.npy",
        data_dir / "edge_index.npy",
        model_dir / "best_model.pt",
        model_dir / "summary.json",
        val_dir / "validation_predictions.npz",
        val_dir / "selected_thresholds.json",
        val_dir / "validation_metrics.json",
        test_dir / "test_predictions.npz",
        test_dir / "test_metrics.json",
        test_dir / "DEVELOPMENT_TEST_TRANSFER_LOCK.json",
        a2_dir / "a1_vs_a2_metrics.csv",
        a2_dir / "analysis_summary.json",
    ]
    require_files(required, "A3 analysis inputs")
    thresholds = json.loads((val_dir / "selected_thresholds.json").read_text(encoding="utf-8"))
    lock = json.loads((test_dir / "DEVELOPMENT_TEST_TRANSFER_LOCK.json").read_text(encoding="utf-8"))
    summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint = model_dir / "best_model.pt"
    hard_checks = {
        "validation_only_selection": thresholds.get("selection_split") == "validation" and thresholds.get("test_accessed") is False,
        "test_not_used_for_selection": lock.get("status") == "COMPLETED",
        "checkpoint_hash_matches_thresholds": thresholds.get("checkpoint_sha256") == sha256_file(checkpoint),
        "parameter_count_verified": int(summary.get("parameter_count", -1)) == EXPECTED_PARAMETER_COUNT,
        "designation_verified": summary.get("experiment_designation") == EXPERIMENT_DESIGNATION,
        "test_not_independent_holdout": True,
    }
    if not all(hard_checks.values()):
        raise RuntimeError(f"A3 analysis hard gate failed: {hard_checks}")

    validation = validate_prediction_archive(val_dir / "validation_predictions.npz", "validation")
    test = validate_prediction_archive(test_dir / "test_predictions.npz", "test")
    alignment = align_predictions(validation, test)
    if not alignment["pass"]:
        raise RuntimeError(f"Prediction alignment failed: {alignment}")
    generic = run_generic_analyses(args, out)

    val_decoded = decode_sets(validation)
    test_decoded = decode_sets(test)
    graph_threshold = float(thresholds["graph_threshold"])
    val_summary = build_metric_summary(validation, val_decoded, graph_threshold)
    test_summary = build_metric_summary(test, test_decoded, graph_threshold)

    rank = rank_statistics(test["y_node"], test["node_prob"])
    fp = fp_distance_statistics(test["y_node"], test_decoded["learned"])
    count, count_rows = count_diagnostics(test, test_decoded)
    exact_rows = exact_set_decomposition(test["y_node"], test_decoded["learned"])
    topology_rows = topology_metrics(test["y_node"], test_decoded["learned"])
    router_rows = per_router_metrics(test["y_node"], test_decoded["learned"])
    consistency_rows = graph_node_consistency(test, test_decoded, graph_threshold)
    per_count_rows = per_attacker_count(test, test_decoded)

    atomic_json_dump(out / "a3_overall_metrics.json", {"validation": val_summary, "test": test_summary})
    atomic_json_dump(out / "a3_count_diagnostics.json", count)
    atomic_csv_dump(out / "a3_count_confusion.csv", count_rows)
    atomic_csv_dump(out / "a3_exact_set_decomposition.csv", exact_rows)
    atomic_csv_dump(out / "a3_true_attacker_rank.csv", rank["rows"])
    atomic_json_dump(out / "a3_true_attacker_rank_summary.json", rank["summary"])
    atomic_csv_dump(out / "a3_fp_distance.csv", fp["rows"])
    atomic_json_dump(out / "a3_fp_distance_summary.json", fp["summary"])
    atomic_csv_dump(out / "a3_topology_metrics.csv", topology_rows)
    atomic_csv_dump(out / "a3_per_router_metrics.csv", router_rows)
    atomic_csv_dump(out / "a3_graph_node_consistency.csv", consistency_rows)
    atomic_csv_dump(out / "a3_per_attacker_count.csv", per_count_rows)

    a2 = load_a2_values(a2_dir)
    topology_map = {row["topology"]: row for row in topology_rows}
    a3_metrics = {
        "graph_f1_at_fpr10": test_summary["graph"]["f1"],
        "test_graph_fpr": test_summary["graph"]["fpr"],
        "graph_recall": test_summary["graph"]["recall"],
        "attack_node_f1": test_summary["learned_attack"]["node_f1"],
        "attack_exact_localization": test_summary["learned_attack"]["exact_localization"],
        "attacker_count_accuracy": count["attack_accuracy"],
        "count_mae": count["attack_mae"],
        "oracle_count_attack_exact": test_summary["oracle_attack"]["exact_localization"],
        "learned_decoder_attack_exact": test_summary["learned_attack"]["exact_localization"],
        "true_attacker_mean_rank": rank["summary"]["mean_rank"],
        "true_attacker_median_rank": rank["summary"]["median_rank"],
        "attack_empty_prediction_fraction": count["attack_empty_prediction_fraction"],
        "one_hop_fp_fraction": fp["summary"]["one_hop_fraction"],
        "within_two_hops_fp_fraction": fp["summary"]["within_two_hops_fraction"],
        "corner_node_f1": topology_map["corner"]["node_f1"],
        "edge_node_f1": topology_map["edge"]["node_f1"],
        "interior_node_f1": topology_map["interior"]["node_f1"],
        "corner_interior_f1_gap": topology_map["interior"]["node_f1"] - topology_map["corner"]["node_f1"],
        "validation_to_test_graph_f1_gap_abs": abs(val_summary["graph"]["f1"] - test_summary["graph"]["f1"]),
        "validation_to_test_attack_node_f1_gap_abs": abs(val_summary["learned_attack"]["node_f1"] - test_summary["learned_attack"]["node_f1"]),
        "parameter_count": float(EXPECTED_PARAMETER_COUNT),
    }
    rows = comparison_rows(a2, a3_metrics)
    atomic_csv_dump(out / "a2_vs_a3_metrics.csv", rows)
    decision = acceptance_decision(a2, a3_metrics)
    decision["a2_metrics"] = a2
    decision["a3_metrics"] = a3_metrics
    decision["hardware_context"] = {
        "shared_4x4_regional_expert": True,
        "regional_embedding_dim": REGIONAL_EMBEDDING_DIM,
        "outputs": ["graph scalar", "attacker scores[16]", "count logits[5]", "embedding[64]"],
        "future_optional_victim_outputs": True,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "mac_estimate": estimate_macs_per_sample(),
    }
    atomic_json_dump(out / "a3_acceptance_decision.json", decision)

    targeted = {
        "ranking_loss_reduced_nearby_fp": (
            a3_metrics["one_hop_fp_fraction"] < a2["one_hop_fp_fraction"]
            or a3_metrics["within_two_hops_fp_fraction"] < a2["within_two_hops_fp_fraction"]
        ),
        "valid_masks_reduced_boundary_bias": (
            a3_metrics["corner_node_f1"] > a2["corner_node_f1"]
            or a3_metrics["corner_interior_f1_gap"] < a2["corner_interior_f1_gap"]
        ),
        "one_gcn_local_skip_improved_oracle_ranking": a3_metrics["oracle_count_attack_exact"] > a2["oracle_count_attack_exact"],
        "explicit_count_head_reduced_cardinality_error": (
            a3_metrics["attacker_count_accuracy"] > a2["attacker_count_accuracy"]
            and a3_metrics["count_mae"] < a2["count_mae"]
        ),
        "graph_localization_tradeoff_resolved": (
            a3_metrics["graph_f1_at_fpr10"] >= a2["graph_f1_at_fpr10"]
            and a3_metrics["attack_exact_localization"] >= a2["attack_exact_localization"]
        ),
    }
    atomic_json_dump(out / "a3_component_evidence.json", targeted)

    hard_checks.update(
        {
            "prediction_alignment": alignment["pass"],
            "generic_comprehensive_present": generic["base_output_exists"],
            "generic_remaining_present": generic["remaining_output_exists"],
            "a2_comparison_generated": (out / "a2_vs_a3_metrics.csv").is_file(),
            "acceptance_decision_generated": (out / "a3_acceptance_decision.json").is_file(),
        }
    )
    analysis_summary = {
        "analysis_pass": all(hard_checks.values()),
        "hard_checks": hard_checks,
        "alignment": alignment,
        "generic_analysis": generic,
        "acceptance_classification": decision["classification"],
        "test_role": "development comparison; not independent publication holdout",
        "v4_shortcut_publication_blocker": True,
    }
    atomic_json_dump(out / "analysis_summary.json", analysis_summary)
    report_lines = [
        "# V4-A3 Complete Failure Analysis and A2 Comparison",
        "",
        f"**Decision:** {decision['classification']}",
        "",
        "## Key A2 versus A3 metrics",
        "",
        "| Metric | A2 | A3 | Delta | Winner |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['metric']} | {row['a2_value']:.6f} | {row['a3_value']:.6f} | "
            f"{row['absolute_difference_a3_minus_a2']:.6f} | {row['winner']} |"
        )
    report_lines.extend(
        [
            "",
            "## Component evidence",
            "",
            *[f"- `{key}`: `{value}`" for key, value in targeted.items()],
            "",
            "## Integrity",
            "",
            *[f"- `{key}`: `{value}`" for key, value in hard_checks.items()],
            "",
            "The V4 matched-control shortcut remains a publication blocker shared by A1/A2/A3.",
        ]
    )
    (out / "complete_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    provenance = {
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "common_script_sha256": sha256_file(COMMON_DIR / "v4_a3_sourcepreserve.py"),
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation_predictions_sha256": sha256_file(val_dir / "validation_predictions.npz"),
        "test_predictions_sha256": sha256_file(test_dir / "test_predictions.npz"),
        "selected_thresholds_sha256": sha256_file(val_dir / "selected_thresholds.json"),
        "dataset_metadata_sha256": sha256_file(data_dir / "metadata.json"),
        "test_used_for_selection": False,
        "dataset_modified": False,
    }
    atomic_json_dump(out / "provenance.json", provenance)
    artifacts = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.csv":
            artifacts.append(
                {
                    "relative_path": str(path.relative_to(out)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    atomic_csv_dump(out / "artifact_manifest.csv", artifacts, ["relative_path", "bytes", "sha256"])
    if not analysis_summary["analysis_pass"]:
        raise RuntimeError(f"A3 complete analysis failed: {hard_checks}")
    print(ANALYSIS_PASS)
    print(json.dumps({"classification": decision["classification"], "checks": decision["checks"]}, indent=2))
    return 0


def normalization_subset(metadata: Mapping[str, Any]) -> dict[str, Any]:
    selected = {}
    for key, value in metadata.items():
        lower = str(key).lower()
        if any(token in lower for token in ("norm", "clip", "feature", "window", "stride")):
            selected[key] = value
    return selected


def export_rtl(args: argparse.Namespace) -> int:
    analysis_dir = Path(args.analysis_dir).resolve()
    decision_path = analysis_dir / "a3_acceptance_decision.json"
    summary_path = analysis_dir / "analysis_summary.json"
    require_files([decision_path, summary_path], "A3 export gate")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("analysis_pass") is not True:
        raise RuntimeError("A3 analysis did not pass")
    if decision.get("classification") != "accepted as primary model" and not args.allow_nonprimary_export:
        raise RuntimeError(
            f"RTL export blocked: classification={decision.get('classification')!r}. "
            "Use --allow-nonprimary-export only for an explicitly labelled diagnostic package."
        )
    out = Path(args.out_dir).resolve()
    ensure_new(out)
    data_dir = Path(args.data_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    val_dir = Path(args.validation_dir).resolve()
    checkpoint_path = model_dir / "best_model.pt"
    thresholds_path = val_dir / "selected_thresholds.json"
    require_files([checkpoint_path, model_dir / "config.json", model_dir / "summary.json", thresholds_path], "RTL export inputs")

    shutil.copy2(checkpoint_path, out / "best_model_float.pt")
    shutil.copy2(model_dir / "config.json", out / "training_config.json")
    shutil.copy2(model_dir / "summary.json", out / "training_summary.json")
    shutil.copy2(thresholds_path, out / "thresholds_and_decoder.json")
    shutil.copy2(analysis_dir / "a3_acceptance_decision.json", out / "acceptance_decision.json")

    model = A3SourcePreserveModel()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    signature = model_signature(model)
    state_manifest = [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "elements": int(tensor.numel()),
        }
        for name, tensor in model.state_dict().items()
    ]
    atomic_json_dump(out / "model_configuration.json", signature)
    atomic_csv_dump(out / "parameter_manifest.csv", state_manifest, ["name", "shape", "dtype", "elements"])
    atomic_json_dump(out / "feature_order.json", {"features": feature_order(data_dir)})
    metadata = load_metadata(data_dir)
    atomic_json_dump(
        out / "normalization_metadata.json",
        {
            "selected_metadata": normalization_subset(metadata),
            "full_metadata_sha256": sha256_file(data_dir / "metadata.json"),
            "note": "Verify normalization constants against the final frontend before fixed-point conversion.",
        },
    )
    physical_mask = build_physical_valid_port_mask()
    atomic_json_dump(
        out / "valid_port_mask_specification.json",
        {
            "standalone_4x4_mask": physical_mask.tolist(),
            "port_order": list(PORT_ORDER),
            "router_order": "router_id = row * 4 + column",
            "hierarchical_rule": "physical validity is computed from global mesh coordinates and is distinct from intra-region GCN adjacency",
        },
    )
    atomic_json_dump(
        out / "regional_output_interface.json",
        {
            "regional_input": [16, "T", "F"],
            "current_T": 8,
            "current_F": 24,
            "graph_output": [1],
            "attacker_output": [16],
            "count_output": [5],
            "regional_embedding": [64],
            "future_optional_outputs": ["victim_scores[16]", "victim_count_logits"],
            "shared_weights_across_regions": True,
            "embedding_dimension_configurable_in_hierarchical_wrapper": True,
        },
    )
    atomic_json_dump(
        out / "runtime_and_mac_estimate.json",
        {
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "float32_weight_bytes": EXPECTED_PARAMETER_COUNT * 4,
            "int8_weight_bytes_projected_not_measured": EXPECTED_PARAMETER_COUNT,
            "macs_per_regional_sample_approx": estimate_macs_per_sample(),
            "attention_or_softmax": False,
            "quantization_performed": False,
        },
    )
    atomic_json_dump(
        out / "future_quantization_notes.json",
        {
            "status": "not performed",
            "required_next_steps": [
                "calibrate activations on train/validation only",
                "compare float versus fixed-point golden vectors",
                "synthesize accepted arithmetic only",
                "do not change decoder thresholds after test inspection",
            ],
        },
    )

    validation = validate_prediction_archive(val_dir / "validation_predictions.npz", "validation")
    global_indices = validation["global_index"][:2].astype(np.int64)
    x_memmap = np.load(data_dir / "x.npy", mmap_mode="r")
    golden_x = np.asarray(x_memmap[global_indices], dtype=np.float32)
    edge_index = np.load(data_dir / "edge_index.npy")
    adjacency = build_normalized_adjacency(edge_index)
    with torch.no_grad():
        output = model(torch.from_numpy(golden_x), adjacency, physical_mask, return_intermediates=True)
    np.savez_compressed(
        out / "golden_float_vectors.npz",
        global_index=global_indices,
        input_x=golden_x,
        adjacency=adjacency.numpy(),
        physical_valid_port_mask=physical_mask.numpy(),
        graph_logits=output["graph_logits"].numpy(),
        attacker_logits=output["node_logits"].numpy(),
        count_logits=output["count_logits"].numpy(),
        regional_embedding=output["regional_embedding"].numpy(),
        h_local=output["h_local"].numpy(),
        h_graph=output["h_graph"].numpy(),
    )
    atomic_json_dump(
        out / "golden_vector_specification.json",
        {
            "samples": global_indices.tolist(),
            "dtype": "float32",
            "comparison_requirement": "bitwise where implementation matches PyTorch operation order; otherwise tolerance must be predeclared before RTL comparison",
        },
    )
    hashes = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name not in {"artifact_hashes.csv", "rtl_float_export.zip"}:
            hashes.append({"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    atomic_csv_dump(out / "artifact_hashes.csv", hashes, ["filename", "bytes", "sha256"])
    zip_path = out / "rtl_float_export.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.is_file() and path != zip_path:
                archive.write(path, arcname=path.name)
    atomic_json_dump(
        out / "export_summary.json",
        {
            "status": "FLOAT_EXPORT_COMPLETE",
            "classification": decision["classification"],
            "diagnostic_override": bool(args.allow_nonprimary_export),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "zip_sha256": sha256_file(zip_path),
            "quantization_performed": False,
            "rtl_conversion_performed": False,
        },
    )
    print(EXPORT_PASS)
    print(out / "export_summary.json")
    return 0


def self_test() -> int:
    rng = np.random.default_rng(7)
    n = 64
    y_graph = np.zeros(n, dtype=np.float32)
    y_graph[n // 2 :] = 1
    y_node = np.zeros((n, 16), dtype=np.float32)
    y_node[n // 2 :, 5] = 1
    graph_prob = np.clip(y_graph * 0.75 + (1 - y_graph) * 0.15 + rng.normal(0, 0.03, n), 0, 1).astype(np.float32)
    node_prob = rng.uniform(0.01, 0.2, size=(n, 16)).astype(np.float32)
    node_prob[n // 2 :, 5] = 0.85
    count_prob = np.full((n, 5), 0.02, dtype=np.float32)
    count_prob[: n // 2, 0] = 0.92
    count_prob[n // 2 :, 1] = 0.92
    count_prob /= count_prob.sum(axis=1, keepdims=True)
    prediction = {
        "graph_prob": graph_prob,
        "node_prob": node_prob,
        "count_prob": count_prob,
        "y_graph": y_graph,
        "y_node": y_node,
        "attacker_count": y_node.sum(axis=1).astype(np.int64),
        "global_index": np.arange(n),
        "run_index": np.repeat(np.arange(4), n // 4),
        "split_id": np.zeros(n, dtype=np.int64),
    }
    decoded = decode_sets(prediction)
    summary = build_metric_summary(prediction, decoded, 0.5)
    checks = {
        "graph_f1": summary["graph"]["f1"] > 0.95,
        "exact": summary["learned_attack"]["exact_localization"] > 0.95,
        "count": summary["count"]["attack_accuracy"] > 0.95,
        "oracle": summary["oracle_attack"]["exact_localization"] > 0.95,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Self-test failed: {checks}")
    print(SELF_TEST_PASS)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["analyze", "export-rtl", "self-test"])
    parser.add_argument("--repo-root", default=str(REPO_FROM_SCRIPT))
    parser.add_argument("--data-dir")
    parser.add_argument("--model-dir")
    parser.add_argument("--validation-dir")
    parser.add_argument("--test-dir")
    parser.add_argument("--a2-analysis-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--analysis-dir")
    parser.add_argument("--v3-data-dir")
    parser.add_argument("--base-analysis-script")
    parser.add_argument("--remaining-analysis-script")
    parser.add_argument("--allow-nonprimary-export", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "self-test":
        return self_test()
    if args.mode == "analyze":
        required = ("data_dir", "model_dir", "validation_dir", "test_dir", "a2_analysis_dir", "out_dir")
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            raise ValueError(f"Analyze mode missing arguments: {missing}")
        return analyze(args)
    required = ("data_dir", "model_dir", "validation_dir", "analysis_dir", "out_dir")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise ValueError(f"Export mode missing arguments: {missing}")
    return export_rtl(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
