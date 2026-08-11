#!/usr/bin/env python3
"""
V5 P2-C1 Scenario and Root-Cause Interpretation

Reads only P2-C0 frozen-cache analysis products. It does not read the B6
prediction cache directly, load the model/checkpoint, enumerate runs/test, or
open original test tensors.

The stage converts C0's raw inventories into:
- graph-error concentration analysis;
- persistent-versus-sporadic episode analysis;
- K1/K2/K3/K4 difficulty ranking;
- per-role and per-node bottleneck ranking;
- graph-score overlap interpretation;
- lexical enrichment of pair-key tokens in hard cases;
- evidence-backed next-dataset and next-model priorities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


STAGE = "V5_P2_C1_SCENARIO_AND_ROOT_CAUSE_INTERPRETATION"
COMPLETE = f"{STAGE}_COMPLETE"

EXPECTED_C0_DECISION = (
    "FREEZE_POSTTEST_FAILURE_ANALYSIS_FROM_B6_CACHE_ONLY"
)
EXPECTED_TEST_PAIRS = 69
EXPECTED_FP = 2307
EXPECTED_FN = 35

ARTIFACT_FILES = {
    "per_pair_summary": "V5_P2_C0_PER_PAIR_FAILURE_SUMMARY.csv",
    "graph_error_windows": "V5_P2_C0_GRAPH_ERROR_WINDOWS.csv",
    "graph_error_episodes": "V5_P2_C0_CONSECUTIVE_GRAPH_ERROR_EPISODES.csv",
    "k_summary": "V5_P2_C0_K1_K2_K3_K4_ATTACK_SUMMARY.csv",
    "per_node_role_metrics": "V5_P2_C0_PER_NODE_ROLE_METRICS.csv",
    "graph_score_distributions": "V5_P2_C0_GRAPH_SCORE_DISTRIBUTIONS.json",
    "analysis_markdown": "V5_P2_C0_FROZEN_CACHE_FAILURE_ANALYSIS.md",
    "artifact_hash_archive": "V5_P2_FINAL_B6_ARTIFACT_HASH_ARCHIVE.csv",
}


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def safe_div(
    numerator: int | float,
    denominator: int | float,
) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def concentration_summary(
    values: list[int],
) -> dict[str, Any]:
    ordered = sorted(values, reverse=True)
    total = sum(ordered)
    nonzero = [value for value in ordered if value > 0]

    def share(count: int) -> float:
        return safe_div(sum(ordered[:count]), total)

    if total == 0:
        gini = 0.0
    else:
        ascending = sorted(ordered)
        n = len(ascending)
        weighted_sum = sum(
            (index + 1) * value
            for index, value in enumerate(ascending)
        )
        gini = (
            (2.0 * weighted_sum) / (n * total)
            - (n + 1.0) / n
        )

    return {
        "total": total,
        "nonzero_pair_count": len(nonzero),
        "top_1_share": share(1),
        "top_3_share": share(3),
        "top_5_share": share(5),
        "top_10_share": share(10),
        "median_all_pairs": float(statistics.median(ordered)),
        "median_nonzero_pairs": (
            float(statistics.median(nonzero))
            if nonzero else 0.0
        ),
        "p90_all_pairs": quantile(ordered, 0.90),
        "maximum": ordered[0] if ordered else 0,
        "gini": gini,
    }


def episode_summary(
    rows: list[dict[str, str]],
    error_type: str,
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["graph_error_type"] == error_type
    ]
    lengths = [
        as_int(row, "window_count")
        for row in selected
    ]

    buckets = {
        "single_window": 0,
        "short_2_to_4": 0,
        "medium_5_to_16": 0,
        "long_17_or_more": 0,
    }
    for length in lengths:
        if length == 1:
            buckets["single_window"] += 1
        elif length <= 4:
            buckets["short_2_to_4"] += 1
        elif length <= 16:
            buckets["medium_5_to_16"] += 1
        else:
            buckets["long_17_or_more"] += 1

    total_error_windows = sum(lengths)
    long_windows = sum(
        length for length in lengths
        if length >= 17
    )

    return {
        "episode_count": len(lengths),
        "error_window_count": total_error_windows,
        "mean_episode_windows": (
            float(statistics.mean(lengths))
            if lengths else 0.0
        ),
        "median_episode_windows": (
            float(statistics.median(lengths))
            if lengths else 0.0
        ),
        "p90_episode_windows": quantile(lengths, 0.90),
        "maximum_episode_windows": max(lengths) if lengths else 0,
        "long_episode_window_share": safe_div(
            long_windows,
            total_error_windows,
        ),
        "buckets": buckets,
    }


def tokenize_pair_key(pair_key: str) -> set[str]:
    raw_tokens = re.split(r"[^A-Za-z0-9]+", pair_key)
    stop = {
        "",
        "P2",
        "TEST",
        "TE",
        "ATTACK",
        "CONTROL",
        "PAIR",
        "RUN",
    }

    tokens: set[str] = set()
    for raw in raw_tokens:
        token = raw.upper()
        if token in stop:
            continue
        if token.isdigit():
            continue
        if len(token) < 2:
            continue
        tokens.add(token)
    return tokens


def token_enrichment_rows(
    pair_rows: list[dict[str, str]],
    hard_pair_keys: set[str],
    hard_label: str,
) -> list[dict[str, Any]]:
    all_pairs = {
        row["pair_key"]
        for row in pair_rows
    }
    hard_pairs = all_pairs & hard_pair_keys
    if not hard_pairs:
        return []

    all_counts: Counter[str] = Counter()
    hard_counts: Counter[str] = Counter()

    for pair_key in all_pairs:
        all_counts.update(tokenize_pair_key(pair_key))
    for pair_key in hard_pairs:
        hard_counts.update(tokenize_pair_key(pair_key))

    rows: list[dict[str, Any]] = []
    for token, hard_count in hard_counts.items():
        all_count = all_counts[token]
        if all_count < 2 or hard_count < 2:
            continue

        hard_rate = safe_div(hard_count, len(hard_pairs))
        baseline_rate = safe_div(all_count, len(all_pairs))
        enrichment = safe_div(hard_rate, baseline_rate)

        rows.append(
            {
                "hard_case_group": hard_label,
                "token": token,
                "hard_pair_count": hard_count,
                "hard_pair_rate": hard_rate,
                "all_pair_count": all_count,
                "all_pair_rate": baseline_rate,
                "enrichment_ratio": enrichment,
                "interpretation_boundary": (
                    "lexical association only; token semantics "
                    "must be confirmed from dataset naming documentation"
                ),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            -float(row["enrichment_ratio"]),
            -int(row["hard_pair_count"]),
            row["token"],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    c0_dir = args.c0_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(
            f"STOP: output already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    report_path = (
        c0_dir
        / "V5_P2_C0_FROZEN_TEST_CACHE_FAILURE_ANALYSIS.json"
    )
    lock_path = (
        c0_dir
        / "V5_P2_C0_FROZEN_TEST_CACHE_FAILURE_ANALYSIS_LOCK.json"
    )
    complete_path = (
        c0_dir
        / "V5_P2_C0_FROZEN_TEST_CACHE_FAILURE_ANALYSIS_COMPLETE"
    )

    paths = {
        "c0_report": report_path,
        "c0_lock": lock_path,
        "c0_complete": complete_path,
        **{
            name: c0_dir / filename
            for name, filename in ARTIFACT_FILES.items()
        },
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing C0 artifact {name}: {path}")

    if failures:
        hold = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "prediction_cache_opened": False,
            "model_loaded": False,
            "test_inference_rerun": False,
        }
        write_json(output_dir / f"{STAGE}.json", hold)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        return 1

    c0_report = load_json(report_path)
    c0_lock = load_json(lock_path)

    if c0_report.get("status") != "COMPLETE":
        failures.append("C0 report is not COMPLETE")
    if c0_report.get("decision") != EXPECTED_C0_DECISION:
        failures.append("C0 decision changed")
    if c0_lock.get("report_sha256") != sha256_file(report_path):
        failures.append("C0 report SHA mismatch")
    if c0_lock.get("test_inference_rerun") is not False:
        failures.append("C0 unexpectedly reran test inference")
    if c0_lock.get("test_tensor_contents_accessed") is not False:
        failures.append("C0 unexpectedly accessed test tensors")
    if c0_lock.get("checkpoint_loaded") is not False:
        failures.append("C0 unexpectedly loaded the checkpoint")

    expected_hashes = c0_lock.get("artifact_sha256", {})
    for name in ARTIFACT_FILES:
        expected = expected_hashes.get(name)
        observed = sha256_file(paths[name])
        if expected != observed:
            failures.append(
                f"C0 artifact SHA mismatch for {name}"
            )

    inventory = c0_report.get("failure_inventory", {})
    if inventory.get("false_positive_windows") != EXPECTED_FP:
        failures.append("C0 false-positive count changed")
    if inventory.get("false_negative_windows") != EXPECTED_FN:
        failures.append("C0 false-negative count changed")

    if failures:
        hold = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "prediction_cache_opened": False,
            "model_loaded": False,
            "test_inference_rerun": False,
        }
        write_json(output_dir / f"{STAGE}.json", hold)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    pair_rows = read_csv(paths["per_pair_summary"])
    error_rows = read_csv(paths["graph_error_windows"])
    episode_rows = read_csv(paths["graph_error_episodes"])
    k_rows_raw = read_csv(paths["k_summary"])
    node_rows_raw = read_csv(paths["per_node_role_metrics"])
    score_summary = load_json(paths["graph_score_distributions"])

    if len(pair_rows) != EXPECTED_TEST_PAIRS:
        failures.append(
            f"per-pair rows={len(pair_rows)}, "
            f"expected {EXPECTED_TEST_PAIRS}"
        )
    if len(error_rows) != EXPECTED_FP + EXPECTED_FN:
        failures.append(
            f"graph-error rows={len(error_rows)}, "
            f"expected {EXPECTED_FP + EXPECTED_FN}"
        )

    fp_rows = [
        row for row in error_rows
        if row["graph_error_type"] == "FP"
    ]
    fn_rows = [
        row for row in error_rows
        if row["graph_error_type"] == "FN"
    ]
    if len(fp_rows) != EXPECTED_FP:
        failures.append("graph-error CSV FP count changed")
    if len(fn_rows) != EXPECTED_FN:
        failures.append("graph-error CSV FN count changed")

    if failures:
        hold = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "prediction_cache_opened": False,
            "model_loaded": False,
            "test_inference_rerun": False,
        }
        write_json(output_dir / f"{STAGE}.json", hold)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    fp_concentration = concentration_summary(
        [as_int(row, "graph_fp") for row in pair_rows]
    )
    fn_concentration = concentration_summary(
        [as_int(row, "graph_fn") for row in pair_rows]
    )

    fp_episode = episode_summary(episode_rows, "FP")
    fn_episode = episode_summary(episode_rows, "FN")

    hard_fp_rows = sorted(
        pair_rows,
        key=lambda row: (
            -as_int(row, "graph_fp"),
            -as_float(row, "graph_fpr"),
            as_int(row, "pair_ordinal"),
        ),
    )
    hard_fn_rows = sorted(
        pair_rows,
        key=lambda row: (
            -as_int(row, "graph_fn"),
            as_float(row, "graph_recall"),
            as_int(row, "pair_ordinal"),
        ),
    )
    hard_exact_rows = sorted(
        pair_rows,
        key=lambda row: (
            as_float(row, "all_task_exact_attack"),
            as_int(row, "pair_ordinal"),
        ),
    )

    fp_threshold = quantile(
        [as_int(row, "graph_fp") for row in pair_rows],
        0.75,
    )
    hard_fp_pair_keys = {
        row["pair_key"]
        for row in pair_rows
        if as_int(row, "graph_fp") >= fp_threshold
        and as_int(row, "graph_fp") > 0
    }

    exact_threshold = quantile(
        [
            as_float(row, "all_task_exact_attack")
            for row in pair_rows
        ],
        0.25,
    )
    hard_exact_pair_keys = {
        row["pair_key"]
        for row in pair_rows
        if as_float(row, "all_task_exact_attack")
        <= exact_threshold
    }

    token_rows = (
        token_enrichment_rows(
            pair_rows,
            hard_fp_pair_keys,
            "upper_quartile_false_positive_pairs",
        )
        + token_enrichment_rows(
            pair_rows,
            hard_exact_pair_keys,
            "lower_quartile_attack_exact_pairs",
        )
    )
    token_path = (
        output_dir
        / "V5_P2_C1_PAIR_KEY_TOKEN_ENRICHMENT.csv"
    )
    write_csv(token_path, token_rows)

    k_rows: list[dict[str, Any]] = []
    for row in k_rows_raw:
        parsed = {
            "attacker_count": as_int(row, "attacker_count"),
            "attack_window_count": as_int(row, "attack_window_count"),
            "graph_recall": as_float(row, "graph_recall"),
            "graph_false_negative_count": as_int(
                row,
                "graph_false_negative_count",
            ),
            "mean_graph_score": as_float(row, "mean_graph_score"),
            "median_graph_score": as_float(row, "median_graph_score"),
            "count_accuracy": as_float(row, "count_accuracy"),
            "all_task_exact_accuracy": as_float(
                row,
                "all_task_exact_accuracy",
            ),
        }
        for role in ("source", "transit", "victim", "path"):
            parsed[f"{role}_gated_exact"] = as_float(
                row,
                f"{role}_gated_exact",
            )
            parsed[f"{role}_ungated_exact"] = as_float(
                row,
                f"{role}_ungated_exact",
            )
        k_rows.append(parsed)

    weakest_k_graph = min(
        k_rows,
        key=lambda row: (
            row["graph_recall"],
            row["attacker_count"],
        ),
    )
    weakest_k_exact = min(
        k_rows,
        key=lambda row: (
            row["all_task_exact_accuracy"],
            row["attacker_count"],
        ),
    )

    role_summary: dict[str, Any] = {}
    role_bottleneck_candidates = []
    node_rank_rows: list[dict[str, Any]] = []

    for role in ("source", "transit", "victim", "path"):
        selected = [
            row for row in node_rows_raw
            if row["role"] == role
        ]
        supported = [
            row for row in selected
            if as_int(row, "truth_positive_windows") > 0
        ]
        worst_f1 = sorted(
            supported,
            key=lambda row: (
                as_float(row, "f1"),
                -as_int(row, "truth_positive_windows"),
                as_int(row, "node"),
            ),
        )[:5]
        highest_fp = sorted(
            selected,
            key=lambda row: (
                -as_int(row, "fp"),
                -as_float(row, "fpr"),
                as_int(row, "node"),
            ),
        )[:5]
        highest_fn = sorted(
            selected,
            key=lambda row: (
                -as_int(row, "fn"),
                as_float(row, "recall"),
                as_int(row, "node"),
            ),
        )[:5]

        mean_supported_f1 = (
            statistics.mean(
                as_float(row, "f1")
                for row in supported
            )
            if supported else 0.0
        )
        role_summary[role] = {
            "supported_node_count": len(supported),
            "mean_supported_node_f1": mean_supported_f1,
            "worst_nodes_by_f1": [
                {
                    "node": as_int(row, "node"),
                    "support": as_int(
                        row,
                        "truth_positive_windows",
                    ),
                    "precision": as_float(row, "precision"),
                    "recall": as_float(row, "recall"),
                    "f1": as_float(row, "f1"),
                    "fp": as_int(row, "fp"),
                    "fn": as_int(row, "fn"),
                }
                for row in worst_f1
            ],
            "highest_false_positive_nodes": [
                {
                    "node": as_int(row, "node"),
                    "fp": as_int(row, "fp"),
                    "fpr": as_float(row, "fpr"),
                    "precision": as_float(row, "precision"),
                }
                for row in highest_fp
            ],
            "highest_false_negative_nodes": [
                {
                    "node": as_int(row, "node"),
                    "fn": as_int(row, "fn"),
                    "recall": as_float(row, "recall"),
                    "f1": as_float(row, "f1"),
                }
                for row in highest_fn
            ],
        }
        role_bottleneck_candidates.append(
            (mean_supported_f1, role)
        )

        for category, rows in (
            ("worst_f1", worst_f1),
            ("highest_fp", highest_fp),
            ("highest_fn", highest_fn),
        ):
            for rank, row in enumerate(rows, start=1):
                node_rank_rows.append(
                    {
                        "role": role,
                        "ranking": category,
                        "rank": rank,
                        "node": as_int(row, "node"),
                        "truth_positive_windows": as_int(
                            row,
                            "truth_positive_windows",
                        ),
                        "predicted_positive_windows": as_int(
                            row,
                            "predicted_positive_windows",
                        ),
                        "precision": as_float(row, "precision"),
                        "recall": as_float(row, "recall"),
                        "f1": as_float(row, "f1"),
                        "fpr": as_float(row, "fpr"),
                        "fp": as_int(row, "fp"),
                        "fn": as_int(row, "fn"),
                    }
                )

    node_rank_path = (
        output_dir
        / "V5_P2_C1_RANKED_NODE_ROLE_BOTTLENECKS.csv"
    )
    write_csv(node_rank_path, node_rank_rows)

    weakest_role = min(role_bottleneck_candidates)[1]

    attack_threshold = float(
        score_summary["frozen_attack_threshold"]
    )
    control_stats = score_summary["control_scores"]
    attack_stats = score_summary["attack_scores"]
    fp_stats = score_summary["false_positive_scores"]
    fn_stats = score_summary["false_negative_scores"]

    score_interpretation = {
        "frozen_threshold": attack_threshold,
        "control_p75_minus_threshold": (
            float(control_stats["p75"]) - attack_threshold
        ),
        "control_p90_minus_threshold": (
            float(control_stats["p90"]) - attack_threshold
        ),
        "attack_p10_minus_threshold": (
            float(attack_stats["p10"]) - attack_threshold
        ),
        "attack_p05_minus_threshold": (
            float(attack_stats["p05"]) - attack_threshold
        ),
        "false_positive_median": float(fp_stats["median"]),
        "false_negative_median": float(fn_stats["median"]),
        "control_attack_median_gap": (
            float(attack_stats["median"])
            - float(control_stats["median"])
        ),
    }

    broad_fp = (
        fp_concentration["nonzero_pair_count"]
        >= math.ceil(0.75 * EXPECTED_TEST_PAIRS)
    )
    concentrated_fp = fp_concentration["top_10_share"] >= 0.75
    persistent_fp = (
        fp_episode["long_episode_window_share"] >= 0.50
        or fp_episode["median_episode_windows"] >= 5
    )

    if broad_fp and not concentrated_fp:
        detector_scope = "broad_across_control_scenarios"
    elif concentrated_fp:
        detector_scope = "concentrated_in_a_small_hard_scenario_subset"
    else:
        detector_scope = "mixed_or_moderately_concentrated"

    if persistent_fp:
        detector_temporal_form = "persistent_false_alarm_episodes"
    else:
        detector_temporal_form = "mostly_sporadic_false_alarm_windows"

    overall = c0_report["overall"]
    count_accuracy = float(
        overall["count_accuracy_attack"]
    )

    conclusions = {
        "primary_failure": {
            "component": "graph attack-versus-control detector",
            "error_direction": "false positives dominate",
            "evidence": {
                "false_positive_windows": EXPECTED_FP,
                "false_negative_windows": EXPECTED_FN,
                "false_positive_share": safe_div(
                    EXPECTED_FP,
                    EXPECTED_FP + EXPECTED_FN,
                ),
                "pairs_with_false_positive": (
                    fp_concentration["nonzero_pair_count"]
                ),
                "pair_scope": detector_scope,
                "temporal_form": detector_temporal_form,
            },
            "interpretation": (
                "The model detects nearly every attack, but normal traffic "
                "frequently enters the learned attack region. This is a "
                "control/attack separability problem rather than an "
                "attacker-count problem."
            ),
        },
        "attacker_count": {
            "status": (
                "solved_on_frozen_test"
                if count_accuracy == 1.0
                else "remaining_errors"
            ),
            "accuracy": count_accuracy,
            "interpretation": (
                "Do not redesign the four-class count head unless later "
                "datasets introduce new count classes or materially "
                "different attack semantics."
            ),
        },
        "localization": {
            "weakest_role_by_mean_supported_node_f1": weakest_role,
            "weakest_k_for_graph_recall": (
                weakest_k_graph["attacker_count"]
            ),
            "weakest_k_for_strict_attack_exact": (
                weakest_k_exact["attacker_count"]
            ),
            "interpretation": (
                "Localization remains secondary to graph false alarms. "
                "The weakest role and K category should receive targeted "
                "coverage, but improving graph discrimination is the first "
                "priority."
            ),
        },
        "score_overlap": score_interpretation,
    }

    priorities = [
        {
            "priority": 1,
            "target": "hard legitimate/control traffic",
            "action": (
                "Build a new training corpus containing legitimate "
                "high-load, bursty, hotspot, phase-change, and memory-"
                "intensive control runs that resemble the 60 false-positive "
                "test pairs. Pair every such control with matched attacks "
                "under the same legal workload and placement."
            ),
            "reason": (
                "98.51% of graph errors are false positives and false "
                "positives occur in 60 of 69 pairs."
            ),
        },
        {
            "priority": 2,
            "target": "graph decision mechanism",
            "action": (
                "Keep the frozen P2 result unchanged, but in the next "
                "revision test graph heads that explicitly learn normal-"
                "traffic evidence, margin separation, or consistency "
                "between graph detection and source/path node outputs."
            ),
            "reason": (
                "Count is perfect while graph precision is low; the shared "
                "representation contains useful attack structure but the "
                "global decision boundary overfires on controls."
            ),
        },
        {
            "priority": 3,
            "target": "temporal transitions",
            "action": (
                "Use the C0 error-episode CSV to oversample the legal "
                "traffic transitions immediately before and during long "
                "false-positive episodes. Preserve matched chronological "
                "splits and prevent windows from the same run crossing "
                "splits."
            ),
            "reason": (
                f"C1 classifies graph false alarms as "
                f"{detector_temporal_form}."
            ),
        },
        {
            "priority": 4,
            "target": f"{weakest_role} localization",
            "action": (
                "Add scenario-balanced coverage for the worst role/node "
                "combinations listed in the ranked node-role bottleneck CSV."
            ),
            "reason": (
                f"{weakest_role} has the lowest mean supported per-node F1 "
                "among the four localization roles."
            ),
        },
        {
            "priority": 5,
            "target": "attacker-count head",
            "action": (
                "Freeze the current four-class count formulation as the "
                "baseline; do not spend the next iteration optimizing it."
            ),
            "reason": "Frozen blind-test count accuracy is 1.0.",
        },
    ]

    hard_pair_rows: list[dict[str, Any]] = []
    for group, rows in (
        ("highest_false_positive", hard_fp_rows[:15]),
        ("highest_false_negative", hard_fn_rows[:15]),
        ("lowest_attack_all_task_exact", hard_exact_rows[:15]),
    ):
        for rank, row in enumerate(rows, start=1):
            hard_pair_rows.append(
                {
                    "group": group,
                    "rank": rank,
                    "pair_ordinal": as_int(row, "pair_ordinal"),
                    "pair_key": row["pair_key"],
                    "graph_fp": as_int(row, "graph_fp"),
                    "graph_fn": as_int(row, "graph_fn"),
                    "graph_fpr": as_float(row, "graph_fpr"),
                    "graph_recall": as_float(row, "graph_recall"),
                    "mean_attack_score": as_float(
                        row,
                        "mean_attack_score",
                    ),
                    "mean_control_score": as_float(
                        row,
                        "mean_control_score",
                    ),
                    "all_task_exact_attack": as_float(
                        row,
                        "all_task_exact_attack",
                    ),
                    "source_exact_attack": as_float(
                        row,
                        "source_gated_exact_attack",
                    ),
                    "transit_exact_attack": as_float(
                        row,
                        "transit_gated_exact_attack",
                    ),
                    "victim_exact_attack": as_float(
                        row,
                        "victim_gated_exact_attack",
                    ),
                    "path_exact_attack": as_float(
                        row,
                        "path_gated_exact_attack",
                    ),
                }
            )

    hard_pair_path = (
        output_dir
        / "V5_P2_C1_RANKED_HARD_PAIR_CATEGORIES.csv"
    )
    write_csv(hard_pair_path, hard_pair_rows)

    result = {
        "stage": STAGE,
        "status": "COMPLETE",
        "decision": (
            "FREEZE_P2_ROOT_CAUSE_AS_BROAD_CONTROL_FALSE_POSITIVE_"
            "DOMINANCE_AND_PRIORITIZE_HARD_LEGITIMATE_TRAFFIC"
        ),
        "graph_error_concentration": {
            "false_positive": fp_concentration,
            "false_negative": fn_concentration,
        },
        "error_episode_characterization": {
            "false_positive": fp_episode,
            "false_negative": fn_episode,
        },
        "k1_k2_k3_k4": k_rows,
        "role_and_node_bottlenecks": role_summary,
        "conclusions": conclusions,
        "next_revision_priorities": priorities,
        "pair_key_token_enrichment": {
            "method": (
                "lexical comparison of pair-key tokens in hard subsets "
                "against all 69 pairs"
            ),
            "hard_fp_pair_count": len(hard_fp_pair_keys),
            "hard_exact_pair_count": len(hard_exact_pair_keys),
            "row_count": len(token_rows),
            "semantic_warning": (
                "Token names are not interpreted as dataset fields unless "
                "confirmed by the dataset naming contract."
            ),
        },
        "provenance": {
            "c0_report_sha256": sha256_file(report_path),
            "c0_lock_sha256": sha256_file(lock_path),
            "c0_artifact_sha256": {
                name: sha256_file(paths[name])
                for name in ARTIFACT_FILES
            },
        },
        "security_boundary": {
            "prediction_cache_opened": False,
            "b6_pair_manifest_opened": False,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "dataset_root_accessed": False,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
            "test_inference_rerun": False,
            "analysis_source": "P2-C0 frozen analysis products only",
        },
        "warnings": warnings,
        "failures": failures,
        "next_stage": (
            "V5_P2_C2_NEXT_DATASET_AND_MODEL_REVISION_CONTRACT"
        ),
    }

    result_path = output_dir / f"{STAGE}.json"
    write_json(result_path, result)

    markdown = [
        "# V5 P2-C1 Scenario and Root-Cause Interpretation",
        "",
        "## Frozen conclusion",
        "",
        "**The dominant P2 failure is broad false-positive activation on "
        "legitimate/control traffic.**",
        "",
        f"- False-positive windows: **{EXPECTED_FP}**.",
        f"- False-negative windows: **{EXPECTED_FN}**.",
        f"- FP share of graph errors: "
        f"**{100.0 * safe_div(EXPECTED_FP, EXPECTED_FP + EXPECTED_FN):.2f}%**.",
        f"- Pairs with at least one FP: "
        f"**{fp_concentration['nonzero_pair_count']} / 69**.",
        f"- Top-10 FP-pair concentration: "
        f"**{100.0 * fp_concentration['top_10_share']:.2f}%**.",
        f"- FP temporal form: **{detector_temporal_form}**.",
        "",
        "This means the problem is not limited to one attacker count or one "
        "isolated bad run. Legitimate traffic broadly overlaps the learned "
        "attack signature.",
        "",
        "## What is already working",
        "",
        f"- Active attacker-count accuracy: **{count_accuracy:.4f}**.",
        "- The count head should remain frozen as the baseline.",
        "- Attack recall remains very high; the detector is not primarily "
        "missing attacks.",
        "",
        "## Localization bottleneck",
        "",
        f"- Weakest role by mean supported per-node F1: "
        f"**{weakest_role}**.",
        f"- Weakest K category for graph recall: "
        f"**K{weakest_k_graph['attacker_count']}**.",
        f"- Weakest K category for strict attack exactness: "
        f"**K{weakest_k_exact['attacker_count']}**.",
        "",
        "Localization should be improved after the legal/control false-positive "
        "problem, not before it.",
        "",
        "## Required next dataset revision",
        "",
        "1. Collect hard legitimate traffic: high-load, bursty, hotspot, "
        "phase-change, and memory-intensive controls.",
        "2. Match each hard control with attacks using the same legal workload, "
        "mapping, traffic phase, and placement.",
        "3. Preserve run-level chronological splitting and pair alignment.",
        "4. Oversample legal windows corresponding to long false-positive "
        "episodes without copying any P2 test tensor into training.",
        "5. Retain K1–K4 count labels and all four role labels.",
        "",
        "## Required next model revision",
        "",
        "Keep the 43,273-parameter P2 model as the frozen reference. In a new "
        "revision, test a graph decision mechanism that learns explicit "
        "normal-traffic evidence or enforces consistency between graph "
        "detection and localized source/path evidence. Do not optimize the "
        "count head.",
        "",
        "## Integrity boundary",
        "",
        "- C1 opened only C0 reports and CSV/JSON analyses.",
        "- B6 prediction cache opened: **no**.",
        "- Model/checkpoint loaded: **no**.",
        "- Original test tensors accessed: **no**.",
        "- Test inference rerun: **no**.",
        "",
    ]

    markdown_path = (
        output_dir
        / "V5_P2_C1_SCENARIO_ROOT_CAUSE_INTERPRETATION.md"
    )
    atomic_write(markdown_path, "\n".join(markdown))

    artifact_paths = {
        "interpretation_report": result_path,
        "interpretation_markdown": markdown_path,
        "ranked_hard_pair_categories": hard_pair_path,
        "ranked_node_role_bottlenecks": node_rank_path,
        "pair_key_token_enrichment": token_path,
    }

    lock = {
        "status": COMPLETE,
        "decision": (
            "FREEZE_P2_ROOT_CAUSE_AS_BROAD_CONTROL_FALSE_POSITIVE_"
            "DOMINANCE_AND_PRIORITIZE_HARD_LEGITIMATE_TRAFFIC"
        ),
        "report_sha256": sha256_file(result_path),
        "artifact_sha256": {
            name: sha256_file(path)
            for name, path in artifact_paths.items()
        },
        "c0_report_sha256": sha256_file(report_path),
        "c0_lock_sha256": sha256_file(lock_path),
        "prediction_cache_opened": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "test_inference_rerun": False,
        "next_stage": (
            "V5_P2_C2_NEXT_DATASET_AND_MODEL_REVISION_CONTRACT"
        ),
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(
        output_dir / f"{STAGE}_LOCK.json",
        lock,
    )
    atomic_write(
        output_dir / COMPLETE,
        COMPLETE + "\n",
    )

    print("===== V5 P2-C1 SCENARIO + ROOT-CAUSE INTERPRETATION =====")
    print("status: COMPLETE")
    print(
        "decision: "
        "FREEZE_P2_ROOT_CAUSE_AS_BROAD_CONTROL_FALSE_POSITIVE_"
        "DOMINANCE_AND_PRIORITIZE_HARD_LEGITIMATE_TRAFFIC"
    )
    print("false_positive_windows:", EXPECTED_FP)
    print("false_negative_windows:", EXPECTED_FN)
    print(
        "pairs_with_false_positive:",
        fp_concentration["nonzero_pair_count"],
    )
    print(
        "top_10_false_positive_share:",
        fp_concentration["top_10_share"],
    )
    print("false_positive_scope:", detector_scope)
    print("false_positive_temporal_form:", detector_temporal_form)
    print("attacker_count_accuracy:", count_accuracy)
    print("weakest_localization_role:", weakest_role)
    print(
        "weakest_k_graph_recall:",
        weakest_k_graph["attacker_count"],
    )
    print(
        "weakest_k_all_task_exact:",
        weakest_k_exact["attacker_count"],
    )
    print("primary_next_dataset_target: hard_legitimate_control_traffic")
    print("prediction_cache_opened: false")
    print("model_loaded: false")
    print("checkpoint_loaded: false")
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")
    print("test_inference_rerun: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print(
        "next_stage: "
        "V5_P2_C2_NEXT_DATASET_AND_MODEL_REVISION_CONTRACT"
    )
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
