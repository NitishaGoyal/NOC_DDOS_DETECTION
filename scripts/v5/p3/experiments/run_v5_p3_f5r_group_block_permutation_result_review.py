from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = "V5_P3_F5R_GROUP_BLOCK_PERMUTATION_RESULT_REVIEW"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "VALIDATION-EXPLORATORY"

GROUPS = (
    "directional_traffic_volume",
    "inter_flit_timing",
    "queue_activity",
    "buffer_pressure",
    "flow_control_stalls",
)

SEEDS = (
    5101,
    5102,
    5103,
    5104,
    5105,
    5106,
    5107,
    5108,
    5109,
    5110,
)

METRICS = (
    "selection_score",
    "graph_auroc",
    "graph_ap",
    "graph_f1_at_0_5",
    "graph_fpr_at_0_5",
    "count_active_macro_f1",
    "source_ap",
    "source_exact_active",
    "transit_ap",
    "transit_exact_active",
    "victim_ap",
    "victim_exact_active",
    "path_ap",
    "path_exact_active",
)

PRIMARY_TASK_METRICS = (
    "graph_ap",
    "source_ap",
    "transit_ap",
    "victim_ap",
    "path_ap",
    "count_active_macro_f1",
)

EXPECTED_VALIDATION_ITEMS = 13863
EXPECTED_PERMUTATION_UNITS = 50
EXPECTED_PARAMETER_COUNT = 60553


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(rows, f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_report_lock(
    report_path: Path,
    lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = load_json(report_path)
    lock = load_json(lock_path)
    require(report.get("status") == "PASS", f"report not PASS: {report_path}")
    require(
        lock.get("report_sha256") == sha256_file(report_path),
        f"report/lock mismatch: {report_path}",
    )
    return report, lock


def active_run(permutation_root: Path, explicit: str) -> Path:
    if explicit:
        run_dir = Path(explicit).expanduser().resolve()
    else:
        pointer = permutation_root / "F5_ACTIVE_RUN_PATH.txt"
        require(pointer.is_file(), f"active run pointer missing: {pointer}")
        run_dir = Path(
            pointer.read_text(encoding="utf-8").strip()
        ).expanduser().resolve()
    require(run_dir.is_dir(), f"run directory missing: {run_dir}")
    return run_dir


def t_critical_95(df: int) -> float:
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:
        return 2.2621571627409915 if df == 9 else 1.96


def stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size == 10, "expected ten repeats")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1))
    sem = std / math.sqrt(array.size)
    half = t_critical_95(array.size - 1) * sem
    return {
        "repeat_count": int(array.size),
        "values": [float(value) for value in array],
        "mean": mean,
        "standard_deviation": std,
        "standard_error": sem,
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean_ci95_lower": mean - half,
        "mean_ci95_upper": mean + half,
        "positive_repeat_count": int(np.sum(array > 0)),
        "negative_repeat_count": int(np.sum(array < 0)),
        "zero_repeat_count": int(np.sum(array == 0)),
    }


def paired_difference_stats(
    left: list[float],
    right: list[float],
) -> dict[str, Any]:
    require(len(left) == len(right) == 10, "paired comparison requires 10 values")
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    result = stats([float(value) for value in difference])
    result["left_greater_repeat_count"] = int(np.sum(difference > 0))
    result["right_greater_repeat_count"] = int(np.sum(difference < 0))
    result["tie_count"] = int(np.sum(difference == 0))
    result["ci_excludes_zero"] = bool(
        result["mean_ci95_lower"] > 0
        or result["mean_ci95_upper"] < 0
    )
    return result


def metric_importance(
    baseline: float,
    permuted: float,
    metric: str,
) -> float:
    if metric == "graph_fpr_at_0_5":
        return permuted - baseline
    return baseline - permuted


def make_figures(
    output_dir: Path,
    selection_rows: list[dict[str, Any]],
    task_matrix: dict[str, dict[str, float]],
) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    figure_paths = []

    ranking_path = output_dir / "F5R_SELECTION_SCORE_REVIEW.png"
    figure = plt.figure(figsize=(11, 6))
    axis = figure.add_subplot(111)
    labels = [row["group"] for row in selection_rows]
    means = [row["mean_importance"] for row in selection_rows]
    errors = [
        row["ci95_upper"] - row["mean_importance"]
        for row in selection_rows
    ]
    axis.bar(range(len(labels)), means, yerr=errors, capsize=4)
    axis.axhline(0.0, linewidth=1)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(
        [label.replace("_", "\n") for label in labels],
    )
    axis.set_ylabel("Baseline − permuted selection score")
    axis.set_title("F5 reviewed macro-group contribution")
    figure.tight_layout()
    figure.savefig(ranking_path, dpi=180)
    plt.close(figure)
    figure_paths.append(ranking_path)

    metrics = list(PRIMARY_TASK_METRICS)
    groups = list(GROUPS)
    matrix = np.asarray(
        [
            [task_matrix[group][metric] for group in groups]
            for metric in metrics
        ],
        dtype=np.float64,
    )

    task_path = output_dir / "F5R_PRIMARY_TASK_IMPORTANCE_HEATMAP.png"
    figure = plt.figure(figsize=(12, 6))
    axis = figure.add_subplot(111)
    image = axis.imshow(matrix, aspect="auto")
    axis.set_xticks(range(len(groups)))
    axis.set_xticklabels(
        [group.replace("_", "\n") for group in groups]
    )
    axis.set_yticks(range(len(metrics)))
    axis.set_yticklabels(metrics)
    axis.set_title("Mean group-block permutation effect by primary task")
    figure.colorbar(image, ax=axis, label="Performance degradation")
    figure.tight_layout()
    figure.savefig(task_path, dpi=180)
    plt.close(figure)
    figure_paths.append(task_path)

    return figure_paths


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")

    permutation_root = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/permutation"
    )
    run_dir = active_run(permutation_root, args.run_dir)

    canonical_report_path = permutation_root / (
        "V5_P3_F5_DYNAMIC70_GROUP_BLOCK_PERMUTATION_REPORT.json"
    )
    canonical_lock_path = permutation_root / (
        "V5_P3_F5_DYNAMIC70_GROUP_BLOCK_PERMUTATION_LOCK.json"
    )
    run_report_path = run_dir / (
        "V5_P3_F5_DYNAMIC70_GROUP_BLOCK_PERMUTATION_REPORT.json"
    )
    run_lock_path = run_dir / (
        "V5_P3_F5_DYNAMIC70_GROUP_BLOCK_PERMUTATION_LOCK.json"
    )
    r3b_report_path = permutation_root / (
        "V5_P3_F5_R3B_OFFICIAL_RUNTIME_BASELINE_RECOVERY_REPORT.json"
    )
    r3b_lock_path = permutation_root / (
        "V5_P3_F5_R3B_OFFICIAL_RUNTIME_BASELINE_RECOVERY_LOCK.json"
    )

    canonical_report, canonical_lock = verify_report_lock(
        canonical_report_path,
        canonical_lock_path,
    )
    run_report, run_lock = verify_report_lock(
        run_report_path,
        run_lock_path,
    )
    r3b_report, r3b_lock = verify_report_lock(
        r3b_report_path,
        r3b_lock_path,
    )

    require(canonical_lock.get("F5_complete") is True, "canonical F5 incomplete")
    require(canonical_lock.get("F5R_authorized") is True, "F5R not authorized")
    require(canonical_lock.get("F6_authorized") is False, "F6 already authorized")
    require(canonical_lock.get("F7_authorized") is False, "F7 already authorized")
    require(run_lock.get("F5_complete") is True, "run-level F5 incomplete")
    require(run_lock.get("F5R_authorized") is True, "run-level F5R held")
    require(
        r3b_lock.get("F5_baseline_gate_complete") is True,
        "official-runtime baseline recovery incomplete",
    )
    require(
        r3b_lock.get("F5_permutation_units_authorized") is True,
        "R3B did not authorize permutation units",
    )

    execution = canonical_report["execution"]
    require(
        execution["validation_items"] == EXPECTED_VALIDATION_ITEMS,
        "validation item count changed",
    )
    require(
        execution["parameter_count"] == EXPECTED_PARAMETER_COUNT,
        "model parameter count changed",
    )
    require(execution["group_count"] == 5, "group count changed")
    require(execution["repeat_count_per_group"] == 10, "repeat count changed")
    require(
        execution["permuted_inference_units"] == EXPECTED_PERMUTATION_UNITS,
        "permutation unit count changed",
    )
    require(
        execution["baseline_reproduction_pass"] is True,
        "baseline reproduction gate did not pass",
    )

    artifacts = canonical_report["artifacts"]
    results_path = Path(artifacts["results_json"]).resolve()
    repeat_csv_path = Path(artifacts["repeat_csv"]).resolve()
    summary_csv_path = Path(artifacts["summary_csv"]).resolve()
    ranking_csv_path = Path(artifacts["ranking_csv"]).resolve()

    for path in (
        results_path,
        repeat_csv_path,
        summary_csv_path,
        ranking_csv_path,
    ):
        require(path.is_file(), f"F5 result artifact missing: {path}")

    results = load_json(results_path)
    baseline = {
        metric: float(results["baseline_metrics"][metric])
        for metric in METRICS
    }

    repeat_metrics = results["repeat_metrics"]
    require(set(repeat_metrics) == set(GROUPS), "unexpected result groups")

    verified_units = []
    raw_importance: dict[str, dict[str, list[float]]] = {
        group: {metric: [] for metric in METRICS}
        for group in GROUPS
    }

    for group in GROUPS:
        group_results = repeat_metrics[group]
        require(
            set(group_results) == {str(seed) for seed in SEEDS},
            f"seed set mismatch for {group}",
        )
        for seed in SEEDS:
            metrics = group_results[str(seed)]
            require(
                set(METRICS) <= set(metrics),
                f"missing metrics for {group} seed {seed}",
            )
            unit_dir = run_dir / "units" / group / f"seed_{seed}"
            complete_path = unit_dir / "UNIT_COMPLETE"
            metrics_path = unit_dir / "METRICS.json"
            manifest_path = unit_dir / "UNIT_MANIFEST.json"

            require(complete_path.is_file(), f"unit incomplete: {unit_dir}")
            require(metrics_path.is_file(), f"unit metrics missing: {unit_dir}")
            require(manifest_path.is_file(), f"unit manifest missing: {unit_dir}")

            manifest = load_json(manifest_path)
            require(manifest.get("status") == "PASS", f"unit not PASS: {unit_dir}")
            require(
                manifest.get("metrics_sha256") == sha256_file(metrics_path),
                f"unit metrics hash mismatch: {unit_dir}",
            )

            unit_metrics_document = load_json(metrics_path)
            require(
                unit_metrics_document.get("group") == group,
                f"unit group mismatch: {unit_dir}",
            )
            require(
                int(unit_metrics_document.get("seed")) == seed,
                f"unit seed mismatch: {unit_dir}",
            )

            for metric in METRICS:
                expected = float(metrics[metric])
                observed = float(
                    unit_metrics_document["metrics"][metric]
                )
                require(
                    abs(expected - observed) <= 1e-15,
                    f"results/unit metric mismatch: {group} {seed} {metric}",
                )
                raw_importance[group][metric].append(
                    metric_importance(
                        baseline[metric],
                        observed,
                        metric,
                    )
                )

            verified_units.append({
                "group": group,
                "seed": seed,
                "unit_dir": str(unit_dir),
                "metrics_sha256": sha256_file(metrics_path),
                "fixed_points": unit_metrics_document.get("fixed_points"),
            })

    require(
        len(verified_units) == EXPECTED_PERMUTATION_UNITS,
        "verified unit count mismatch",
    )

    summaries = {
        group: {
            metric: stats(raw_importance[group][metric])
            for metric in METRICS
        }
        for group in GROUPS
    }

    selection_rows = []
    for group in GROUPS:
        row = summaries[group]["selection_score"]
        selection_rows.append({
            "group": group,
            "mean_importance": row["mean"],
            "standard_deviation": row["standard_deviation"],
            "ci95_lower": row["mean_ci95_lower"],
            "ci95_upper": row["mean_ci95_upper"],
            "positive_repeat_count": row["positive_repeat_count"],
            "fraction_of_baseline_percent": (
                100.0 * row["mean"] / baseline["selection_score"]
            ),
            "ci_excludes_zero": bool(row["mean_ci95_lower"] > 0),
        })

    selection_rows.sort(
        key=lambda row: row["mean_importance"],
        reverse=True,
    )
    for rank, row in enumerate(selection_rows, start=1):
        row["rank"] = rank

    require(
        [row["group"] for row in selection_rows]
        == [
            "inter_flit_timing",
            "buffer_pressure",
            "queue_activity",
            "directional_traffic_volume",
            "flow_control_stalls",
        ],
        "selection-score group ranking differs from terminal result",
    )

    pairwise_rows = []
    for left_index, left in enumerate(GROUPS):
        for right in GROUPS[left_index + 1:]:
            comparison = paired_difference_stats(
                raw_importance[left]["selection_score"],
                raw_importance[right]["selection_score"],
            )
            pairwise_rows.append({
                "left_group": left,
                "right_group": right,
                "mean_left_minus_right": comparison["mean"],
                "standard_deviation": comparison["standard_deviation"],
                "ci95_lower": comparison["mean_ci95_lower"],
                "ci95_upper": comparison["mean_ci95_upper"],
                "left_greater_repeat_count": comparison[
                    "left_greater_repeat_count"
                ],
                "right_greater_repeat_count": comparison[
                    "right_greater_repeat_count"
                ],
                "ci_excludes_zero": comparison["ci_excludes_zero"],
            })

    task_rankings = {}
    task_matrix = {
        group: {}
        for group in GROUPS
    }
    for metric in PRIMARY_TASK_METRICS:
        rows = []
        for group in GROUPS:
            mean = summaries[group][metric]["mean"]
            task_matrix[group][metric] = mean
            rows.append({
                "group": group,
                "mean_importance": mean,
                "ci95_lower": summaries[group][metric]["mean_ci95_lower"],
                "ci95_upper": summaries[group][metric]["mean_ci95_upper"],
                "positive_repeat_count": summaries[group][metric][
                    "positive_repeat_count"
                ],
            })
        rows.sort(key=lambda row: row["mean_importance"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        task_rankings[metric] = rows

    flow_control_selection = summaries[
        "flow_control_stalls"
    ]["selection_score"]
    flow_control_low_effect = bool(
        flow_control_selection["mean"] < 0.01
        and flow_control_selection["mean_ci95_upper"] < 0.01
    )

    all_groups_stable_positive = all(
        summaries[group]["selection_score"]["positive_repeat_count"] == 10
        and summaries[group]["selection_score"]["mean_ci95_lower"] > 0
        for group in GROUPS
    )

    baseline_implied = []
    for group in GROUPS:
        permuted_scores = [
            float(repeat_metrics[group][str(seed)]["selection_score"])
            for seed in SEEDS
        ]
        implied = [
            permuted + importance
            for permuted, importance in zip(
                permuted_scores,
                raw_importance[group]["selection_score"],
            )
        ]
        baseline_implied.extend(implied)

    baseline_consistency_max_abs = float(
        np.max(
            np.abs(
                np.asarray(baseline_implied, dtype=np.float64)
                - baseline["selection_score"]
            )
        )
    )
    require(
        baseline_consistency_max_abs <= 1e-15,
        "importance values do not reconstruct the common baseline",
    )

    protocol_complete = bool(
        len(verified_units) == EXPECTED_PERMUTATION_UNITS
        and all_groups_stable_positive
        and baseline_consistency_max_abs <= 1e-15
        and all(np.isfinite(row["mean_importance"]) for row in selection_rows)
    )

    F6_authorized = protocol_complete
    F7_authorized = False
    removal_authorized = False

    selection_csv_path = output_dir / "F5R_SELECTION_SCORE_RANKING.csv"
    pairwise_csv_path = output_dir / "F5R_SELECTION_SCORE_PAIRWISE_REVIEW.csv"
    task_json_path = output_dir / "F5R_TASK_SPECIFIC_GROUP_RANKINGS.json"
    review_json_path = output_dir / "F5R_REVIEWED_RESULTS.json"
    decision_path = output_dir / "F5R_FEATURE_INTERPRETATION_AND_STAGE_DECISION.json"

    write_csv(selection_csv_path, selection_rows)
    write_csv(pairwise_csv_path, pairwise_rows)
    atomic_json(task_json_path, task_rankings)

    figures = make_figures(output_dir, selection_rows, task_matrix)

    reviewed_results = {
        "baseline_metrics": baseline,
        "selection_score_ranking": selection_rows,
        "selection_score_pairwise_review": pairwise_rows,
        "task_specific_rankings": task_rankings,
        "full_metric_summaries": summaries,
        "verified_permutation_units": verified_units,
        "baseline_consistency_max_abs_difference": (
            baseline_consistency_max_abs
        ),
    }
    atomic_json(review_json_path, reviewed_results)

    interpretation = {
        "selection_score_order": [
            row["group"]
            for row in selection_rows
        ],
        "strongest_group": selection_rows[0]["group"],
        "weakest_group": selection_rows[-1]["group"],
        "flow_control_stalls_low_effect": flow_control_low_effect,
        "flow_control_stalls_note": (
            "The full 15-channel stall group has a very small permutation "
            "effect. Combined with the prior F2R observation that six stall "
            "channels are constant zero, this makes the stall group the "
            "leading candidate for F6/F7 scrutiny. It does not authorize "
            "removing all stall channels because nine channels remain "
            "training-observed and permutation importance can be suppressed "
            "by correlated or redundant inputs."
        ),
        "permutation_scope_note": (
            "This is a grouped validation permutation-contribution result, "
            "not a causal proof and not a compact-interface result."
        ),
        "hardware_reduction_claim_authorized": False,
        "feature_removal_authorized": False,
        "next_evidence_required": [
            "F6 task-specific integrated gradients",
            "F7 same-width retrained group ablation",
            "later compact reduced-input retraining",
        ],
        "F6_integrated_gradients_authorized": F6_authorized,
        "F7_retraining_ablation_authorized": F7_authorized,
    }
    atomic_json(decision_path, interpretation)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "scope": (
            "Verify all 50 completed permutation units and hashes, "
            "independently reconstruct all metric effects and confidence "
            "intervals, review task-specific rankings, and decide whether "
            "task-specific integrated gradients may begin."
        ),
        "verification": {
            "validation_items": execution["validation_items"],
            "parameter_count": execution["parameter_count"],
            "verified_permutation_units": len(verified_units),
            "expected_permutation_units": EXPECTED_PERMUTATION_UNITS,
            "baseline_reproduction_pass": True,
            "runtime_recovery_pass": True,
            "baseline_consistency_max_abs_difference": (
                baseline_consistency_max_abs
            ),
            "all_groups_selection_effect_positive_10_of_10": (
                all_groups_stable_positive
            ),
            "protocol_complete": protocol_complete,
        },
        "results": {
            "baseline_selection_score": baseline["selection_score"],
            "selection_score_ranking": selection_rows,
            "task_specific_rankings": task_rankings,
            "flow_control_stalls_low_effect": flow_control_low_effect,
        },
        "interpretation": interpretation,
        "decision": {
            "F5R_complete": True,
            "F6_integrated_gradients_authorized": F6_authorized,
            "F7_retraining_ablation_authorized": False,
            "feature_removal_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F6_TASK_SPECIFIC_INTEGRATED_GRADIENTS"
                if F6_authorized
                else "V5_P3_F5R_RESULT_REVIEW_RECOVERY"
            ),
        },
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "validation_feature_tensors_loaded": False,
            "validation_output_artifact_payloads_loaded": False,
            "permutation_result_artifacts_loaded": True,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "metric_formulas_changed": False,
            "model_weights_changed": False,
        },
        "artifacts": {
            "reviewed_results": str(review_json_path),
            "selection_score_ranking_csv": str(selection_csv_path),
            "selection_score_pairwise_csv": str(pairwise_csv_path),
            "task_specific_rankings": str(task_json_path),
            "interpretation_and_decision": str(decision_path),
            "figures": [str(path) for path in figures],
        },
        "provenance": {
            "canonical_F5_report_sha256": sha256_file(canonical_report_path),
            "canonical_F5_lock_sha256": sha256_file(canonical_lock_path),
            "run_F5_report_sha256": sha256_file(run_report_path),
            "run_F5_lock_sha256": sha256_file(run_lock_path),
            "F5_R3B_report_sha256": sha256_file(r3b_report_path),
            "F5_R3B_lock_sha256": sha256_file(r3b_lock_path),
            "F5_results_json_sha256": sha256_file(results_path),
            "F5_repeat_csv_sha256": sha256_file(repeat_csv_path),
            "F5_summary_csv_sha256": sha256_file(summary_csv_path),
            "F5_ranking_csv_sha256": sha256_file(ranking_csv_path),
            "installed_script_sha256": sha256_file(installed_script),
            "reviewed_results_sha256": sha256_file(review_json_path),
            "selection_score_ranking_csv_sha256": sha256_file(
                selection_csv_path
            ),
            "selection_score_pairwise_csv_sha256": sha256_file(
                pairwise_csv_path
            ),
            "task_specific_rankings_sha256": sha256_file(task_json_path),
            "interpretation_and_decision_sha256": sha256_file(decision_path),
        },
    }

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    atomic_json(report_path, report)
    atomic_json(
        lock_path,
        {
            "stage": STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(report_path),
            "reviewed_results_sha256": sha256_file(review_json_path),
            "selection_score_ranking_csv_sha256": sha256_file(
                selection_csv_path
            ),
            "selection_score_pairwise_csv_sha256": sha256_file(
                pairwise_csv_path
            ),
            "task_specific_rankings_sha256": sha256_file(task_json_path),
            "interpretation_and_decision_sha256": sha256_file(decision_path),
            "F5R_complete": True,
            "F6_authorized": F6_authorized,
            "F7_authorized": False,
            "feature_removal_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print(f"run_directory={run_dir}")
    print(f"baseline_selection_score={baseline['selection_score']}")
    print(f"verified_permutation_units={len(verified_units)}")
    print(
        "baseline_consistency_max_abs_difference="
        f"{baseline_consistency_max_abs}"
    )
    print(
        "all_groups_selection_effect_positive_10_of_10="
        f"{str(all_groups_stable_positive).lower()}"
    )
    for row in selection_rows:
        print(
            "reviewed_selection_score_ranking="
            f"{row['rank']}:{row['group']}:"
            f"mean={row['mean_importance']}:"
            f"std={row['standard_deviation']}:"
            f"ci95=[{row['ci95_lower']},{row['ci95_upper']}]:"
            f"percent_of_baseline={row['fraction_of_baseline_percent']}:"
            f"positive_repeats={row['positive_repeat_count']}"
        )
    for metric in PRIMARY_TASK_METRICS:
        leader = task_rankings[metric][0]
        print(
            f"task_leader_{metric}="
            f"{leader['group']}:mean={leader['mean_importance']}:"
            f"ci95=[{leader['ci95_lower']},{leader['ci95_upper']}]"
        )
    print(
        "flow_control_stalls_low_effect="
        f"{str(flow_control_low_effect).lower()}"
    )
    print("feature_removal_authorized=false")
    print("hardware_reduction_claim_authorized=false")
    print(f"F6_integrated_gradients_authorized={str(F6_authorized).lower()}")
    print("F7_retraining_ablation_authorized=false")
    print("sealed_test_tensors_loaded=false")
    print(
        "next_stage="
        + (
            "V5_P3_F6_TASK_SPECIFIC_INTEGRATED_GRADIENTS"
            if F6_authorized
            else "V5_P3_F5R_RESULT_REVIEW_RECOVERY"
        )
    )
    print(f"reviewed_results={review_json_path}")
    print(f"selection_score_ranking_csv={selection_csv_path}")
    print(f"selection_score_pairwise_csv={pairwise_csv_path}")
    print(f"task_specific_rankings={task_json_path}")
    print(f"interpretation_and_decision={decision_path}")
    for path in figures:
        print(f"figure={path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
