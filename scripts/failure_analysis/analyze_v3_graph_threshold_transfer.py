#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REFERENCE_THRESHOLD = 0.50

MODEL_FILES = {
    "conv1d_gcn": "conv1d_gcn_predictions.npz",
    "tcn_attention_gcn": "tcn_attention_gcn_predictions.npz",
    "tcn_meanpool_gcn": "tcn_meanpool_gcn_predictions.npz",
    "tcn_maxpool_gcn": "tcn_maxpool_gcn_predictions.npz",
}
MODEL_ORDER = list(MODEL_FILES)

MODEL_LABELS = {
    "conv1d_gcn": "Conv1D-GCN",
    "tcn_attention_gcn": "TCN-Attention-GCN",
    "tcn_meanpool_gcn": "TCN-MeanPool-GCN",
    "tcn_maxpool_gcn": "TCN-MaxPool-GCN",
}

HARD_NORMAL_RUN = "N-3-7-8-12-Pmixed-R18-V3"
HARD_ATTACK_RUN = "N-5-10-Pbursty-R51-A-12-S20-V3"


def atomic_csv(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        dataframe.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        figure.savefig(temporary, dpi=180, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true, dtype=np.int8).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int8).reshape(-1)
    return {
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    tp = counts["TP"]

    accuracy = safe_divide(tn + tp, tn + fp + fn + tp)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    if (
        np.isfinite(precision)
        and np.isfinite(recall)
        and precision + recall > 0
    ):
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    tnr = safe_divide(tn, tn + fp)
    balanced_accuracy = (
        float((recall + tnr) / 2.0)
        if np.isfinite(recall) and np.isfinite(tnr)
        else float("nan")
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": float(f1),
        "FPR": safe_divide(fp, fp + tn),
        "FNR": safe_divide(fn, fn + tp),
        "balanced_accuracy": balanced_accuracy,
    }


def parse_active_core_count(value: Any) -> int:
    text = str(value).strip()
    if not text or text.lower() == "idle":
        return 0
    return len([part for part in text.split("-") if part.strip()])


def load_inputs(
    prediction_directory: Path,
    metadata_path: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    metadata = pd.read_csv(
        metadata_path,
        keep_default_na=False,
        low_memory=False,
        dtype={
            "sample_id": str,
            "run_id": str,
            "split": str,
            "profile": str,
            "active_cores": str,
            "attackers": str,
            "strength": str,
            "seed": str,
        },
    )

    required_metadata = {
        "sample_order_within_export",
        "sample_index",
        "sample_id",
        "run_id",
        "split",
        "end_epoch",
        "true_graph",
        "profile",
        "active_cores",
        "attackers",
        "true_attacker_count",
        "strength",
        "seed",
    }
    missing_metadata = required_metadata - set(metadata.columns)
    if missing_metadata:
        raise RuntimeError(
            f"prediction_metadata.csv is missing columns: {sorted(missing_metadata)}"
        )

    loaded: dict[str, dict[str, np.ndarray]] = {}

    for model_name, filename in MODEL_FILES.items():
        path = prediction_directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction file: {path}")

        with np.load(path, allow_pickle=False) as source:
            required_arrays = {
                "sample_index",
                "sample_id",
                "split",
                "true_graph",
                "graph_logit",
                "graph_probability",
            }
            missing_arrays = required_arrays - set(source.files)
            if missing_arrays:
                raise RuntimeError(
                    f"{model_name} prediction file is missing: {sorted(missing_arrays)}"
                )
            loaded[model_name] = {name: source[name] for name in source.files}

        print(f"loaded {model_name}: {path}", flush=True)

    return metadata, loaded


def verify_alignment(
    metadata: pd.DataFrame,
    loaded: dict[str, dict[str, np.ndarray]],
) -> None:
    expected = {
        "sample_index": metadata["sample_index"].to_numpy(dtype=np.int64),
        "sample_id": metadata["sample_id"].astype(str).to_numpy(),
        "split": metadata["split"].astype(str).to_numpy(),
        "true_graph": metadata["true_graph"].to_numpy(dtype=np.int64),
    }

    reference: dict[str, np.ndarray] | None = None

    for model_name, arrays in loaded.items():
        if len(arrays["sample_index"]) != len(metadata):
            raise RuntimeError(
                f"{model_name} has {len(arrays['sample_index'])} samples; "
                f"metadata has {len(metadata)}."
            )

        for key, expected_values in expected.items():
            if key in {"sample_id", "split"}:
                actual_values = np.asarray(arrays[key]).astype(str)
            else:
                actual_values = np.asarray(arrays[key], dtype=np.int64)

            if not np.array_equal(actual_values, expected_values):
                raise RuntimeError(
                    f"{model_name} is not aligned with metadata for {key}."
                )

        if reference is None:
            reference = arrays
            continue

        for key in ["sample_index", "sample_id", "split", "true_graph"]:
            if key in {"sample_id", "split"}:
                left = np.asarray(reference[key]).astype(str)
                right = np.asarray(arrays[key]).astype(str)
            else:
                left = np.asarray(reference[key], dtype=np.int64)
                right = np.asarray(arrays[key], dtype=np.int64)

            if not np.array_equal(left, right):
                raise RuntimeError(
                    f"Cross-model alignment failure for {model_name}: {key}"
                )


def build_model_frame(
    metadata: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    model_name: str,
) -> pd.DataFrame:
    frame = metadata.rename(
        columns={"sample_order_within_export": "sample_order"}
    ).copy()

    frame["model_name"] = model_name
    frame["sample_index"] = frame["sample_index"].astype(np.int64)
    frame["end_epoch"] = frame["end_epoch"].astype(np.int64)
    frame["true_graph"] = frame["true_graph"].astype(np.int8)
    frame["attacker_count"] = frame["true_attacker_count"].astype(np.int64)
    frame["active_core_count"] = (
        frame["active_cores"].map(parse_active_core_count).astype(np.int64)
    )
    frame["graph_logit"] = np.asarray(arrays["graph_logit"], dtype=np.float64)
    frame["graph_probability"] = np.asarray(
        arrays["graph_probability"], dtype=np.float64
    )

    if not np.isfinite(
        frame[["graph_logit", "graph_probability"]].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(f"Non-finite graph output detected for {model_name}.")

    return frame


def per_run_metrics(
    frame: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for run_id, run in frame.groupby("run_id", sort=True):
        run = run.sort_values("end_epoch")
        unique_labels = run["true_graph"].unique()

        if len(unique_labels) != 1:
            raise RuntimeError(f"Run {run_id} mixes graph labels: {unique_labels}")

        true_graph = int(unique_labels[0])
        probabilities = run["graph_probability"].to_numpy(dtype=float)
        predictions = (probabilities >= threshold).astype(np.int8)
        truth = run["true_graph"].to_numpy(dtype=np.int8)
        counts = confusion_counts(truth, predictions)
        metrics = metrics_from_counts(counts)

        error_rate = float(np.mean(predictions != truth))
        normal_run_fpr = error_rate if true_graph == 0 else float("nan")
        attack_run_fnr = error_rate if true_graph == 1 else float("nan")

        rows.append(
            {
                "model_name": str(run["model_name"].iloc[0]),
                "split": str(run["split"].iloc[0]),
                "run_id": str(run_id),
                "threshold": float(threshold),
                "profile": str(run["profile"].iloc[0]),
                "active_cores": str(run["active_cores"].iloc[0]),
                "active_core_count": int(run["active_core_count"].iloc[0]),
                "attackers": str(run["attackers"].iloc[0]),
                "attacker_count": int(run["attacker_count"].iloc[0]),
                "strength": str(run["strength"].iloc[0]),
                "seed": str(run["seed"].iloc[0]),
                "true_graph": true_graph,
                "number_of_windows": int(len(run)),
                **counts,
                **metrics,
                "error_rate": error_rate,
                "normal_run_FPR": normal_run_fpr,
                "attack_run_FNR": attack_run_fnr,
                "mean_graph_probability": float(probabilities.mean()),
                "median_graph_probability": float(np.median(probabilities)),
                "minimum_graph_probability": float(probabilities.min()),
                "maximum_graph_probability": float(probabilities.max()),
            }
        )

    return pd.DataFrame(rows)


def evaluate_threshold(
    frame: pd.DataFrame,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    probabilities = frame["graph_probability"].to_numpy(dtype=float)
    truth = frame["true_graph"].to_numpy(dtype=np.int8)
    predictions = (probabilities >= threshold).astype(np.int8)

    counts = confusion_counts(truth, predictions)
    metrics = metrics_from_counts(counts)
    runs = per_run_metrics(frame, threshold)

    normal_runs = runs[runs["true_graph"].eq(0)]
    attack_runs = runs[runs["true_graph"].eq(1)]

    mean_normal_fpr = (
        float(normal_runs["error_rate"].mean())
        if not normal_runs.empty
        else float("nan")
    )
    mean_attack_fnr = (
        float(attack_runs["error_rate"].mean())
        if not attack_runs.empty
        else float("nan")
    )
    worst_normal_fpr = (
        float(normal_runs["error_rate"].max())
        if not normal_runs.empty
        else float("nan")
    )
    worst_attack_fnr = (
        float(attack_runs["error_rate"].max())
        if not attack_runs.empty
        else float("nan")
    )
    run_macro_error_rate = float(runs["error_rate"].mean())

    catastrophic_normal_runs = int(
        (normal_runs["error_rate"] > 0.50).sum()
    )
    catastrophic_attack_runs = int(
        (attack_runs["error_rate"] > 0.50).sum()
    )

    validation_score = (
        0.50 * metrics["F1"]
        + 0.20 * (1.0 - run_macro_error_rate)
        + 0.15 * (1.0 - worst_normal_fpr)
        + 0.15 * (1.0 - worst_attack_fnr)
    )

    summary = {
        "threshold": float(threshold),
        **counts,
        **metrics,
        "number_of_runs": int(runs["run_id"].nunique()),
        "number_of_normal_runs": int(len(normal_runs)),
        "number_of_attack_runs": int(len(attack_runs)),
        "run_macro_error_rate": run_macro_error_rate,
        "mean_normal_run_FPR": mean_normal_fpr,
        "mean_attack_run_FNR": mean_attack_fnr,
        "worst_normal_run_FPR": worst_normal_fpr,
        "worst_attack_run_FNR": worst_attack_fnr,
        "maximum_worst_run_error": float(
            max(worst_normal_fpr, worst_attack_fnr)
        ),
        "catastrophic_normal_runs": catastrophic_normal_runs,
        "catastrophic_attack_runs": catastrophic_attack_runs,
        "validation_selection_score": float(validation_score),
        "passes_worst_run_guardrails": bool(
            worst_normal_fpr <= 0.50 and worst_attack_fnr <= 0.50
        ),
    }

    return summary, runs


def create_threshold_grid(
    minimum: float,
    maximum: float,
    step: float,
) -> np.ndarray:
    if not (0.0 < minimum < maximum < 1.0):
        raise ValueError("Threshold range must satisfy 0 < minimum < maximum < 1.")
    if step <= 0:
        raise ValueError("Threshold step must be positive.")

    values = np.arange(minimum, maximum + step / 2.0, step)
    values = np.clip(values, minimum, maximum)
    values = np.unique(np.round(values, 10))

    if not np.any(np.isclose(values, REFERENCE_THRESHOLD, atol=1e-12)):
        values = np.sort(np.append(values, REFERENCE_THRESHOLD))

    return values


def select_validation_threshold(
    validation_sweep: pd.DataFrame,
) -> tuple[pd.Series, str, int]:
    eligible = validation_sweep[
        validation_sweep["passes_worst_run_guardrails"]
    ].copy()

    if not eligible.empty:
        ranked = eligible.sort_values(
            [
                "validation_selection_score",
                "maximum_worst_run_error",
                "run_macro_error_rate",
                "F1",
                "threshold_distance_from_0_50",
            ],
            ascending=[False, True, True, False, True],
        )
        return (
            ranked.iloc[0],
            "guardrails_then_max_validation_score",
            int(len(eligible)),
        )

    ranked = validation_sweep.sort_values(
        [
            "maximum_worst_run_error",
            "validation_selection_score",
            "run_macro_error_rate",
            "threshold_distance_from_0_50",
        ],
        ascending=[True, False, True, True],
    )

    return (
        ranked.iloc[0],
        "fallback_minimax_no_threshold_passed_guardrails",
        0,
    )


def add_operating_point(
    summary: dict[str, Any],
    *,
    model_name: str,
    split_name: str,
    operating_point: str,
    threshold_source: str,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "split": split_name,
        "operating_point": operating_point,
        "threshold_source": threshold_source,
        **summary,
    }


def compare_operating_points(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "F1",
        "FPR",
        "FNR",
        "balanced_accuracy",
        "run_macro_error_rate",
        "mean_normal_run_FPR",
        "mean_attack_run_FNR",
        "worst_normal_run_FPR",
        "worst_attack_run_FNR",
        "catastrophic_normal_runs",
        "catastrophic_attack_runs",
    ]

    for (model_name, split_name), group in results.groupby(
        ["model_name", "split"], sort=True
    ):
        reference = group[group["operating_point"].eq("reference_0_50")]
        selected = group[group["operating_point"].eq("validation_selected")]

        if len(reference) != 1 or len(selected) != 1:
            raise RuntimeError(
                f"Missing operating point for {model_name} {split_name}."
            )

        reference_row = reference.iloc[0]
        selected_row = selected.iloc[0]

        for metric_name in metric_names:
            reference_value = float(reference_row[metric_name])
            selected_value = float(selected_row[metric_name])
            absolute_change = selected_value - reference_value
            relative_change = (
                absolute_change / abs(reference_value)
                if reference_value != 0
                else float("nan")
            )

            rows.append(
                {
                    "model_name": model_name,
                    "split": split_name,
                    "metric": metric_name,
                    "reference_threshold": float(reference_row["threshold"]),
                    "selected_threshold": float(selected_row["threshold"]),
                    "reference_value": reference_value,
                    "selected_value": selected_value,
                    "absolute_change": absolute_change,
                    "relative_change": relative_change,
                }
            )

    return pd.DataFrame(rows)


def build_dominant_run_table(
    by_run: pd.DataFrame,
) -> pd.DataFrame:
    target_runs = [HARD_NORMAL_RUN, HARD_ATTACK_RUN]
    result = by_run[by_run["run_id"].isin(target_runs)].copy()

    if set(result["run_id"].unique()) != set(target_runs):
        missing = set(target_runs) - set(result["run_id"].unique())
        raise RuntimeError(f"Dominant failure runs missing from outputs: {sorted(missing)}")

    return result.sort_values(
        ["model_name", "run_id", "split", "operating_point"]
    )


def validate_reference_metrics(
    results: pd.DataFrame,
    stage3_metrics: pd.DataFrame,
) -> tuple[bool, list[str]]:
    messages: list[str] = []
    matched = True

    reference_results = results[
        results["operating_point"].eq("reference_0_50")
    ]

    for _, result in reference_results.iterrows():
        expected = stage3_metrics[
            stage3_metrics["model"].eq(result["model_name"])
            & stage3_metrics["split"].eq(result["split"])
        ]

        if len(expected) != 1:
            matched = False
            messages.append(
                f"Missing Stage 3 row for {result['model_name']} {result['split']}"
            )
            continue

        expected_row = expected.iloc[0]

        for result_name, expected_name in {
            "TN": "tn",
            "FP": "fp",
            "FN": "fn",
            "TP": "tp",
        }.items():
            if int(result[result_name]) != int(expected_row[expected_name]):
                matched = False
                messages.append(
                    f"{result['model_name']} {result['split']} {result_name}: "
                    f"{int(result[result_name])} != {int(expected_row[expected_name])}"
                )

        for result_name, expected_name in {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "F1": "f1",
            "FPR": "fpr",
        }.items():
            if not np.isclose(
                float(result[result_name]),
                float(expected_row[expected_name]),
                atol=1e-12,
                rtol=0.0,
            ):
                matched = False
                messages.append(
                    f"{result['model_name']} {result['split']} {result_name}: "
                    f"{float(result[result_name])} != {float(expected_row[expected_name])}"
                )

    return matched, messages


def build_report(
    selected_thresholds: pd.DataFrame,
    transfer_results: pd.DataFrame,
    dominant_runs: pd.DataFrame,
) -> str:
    lines = [
        "# V3 Validation-Selected Graph-Threshold Transfer",
        "",
        "Thresholds were selected using validation data only and then frozen before "
        "test evaluation. No model inference was rerun.",
        "",
        "## Selection rule",
        "",
        "A threshold must first satisfy both validation guardrails when possible:",
        "",
        "- worst normal-run FPR ≤ 0.50;",
        "- worst attack-run FNR ≤ 0.50.",
        "",
        "Among thresholds satisfying both guardrails, selection maximizes:",
        "",
        "```text",
        "0.50 × graph F1",
        "+ 0.20 × (1 − run-macro error rate)",
        "+ 0.15 × (1 − worst normal-run FPR)",
        "+ 0.15 × (1 − worst attack-run FNR)",
        "```",
        "",
        "If no threshold satisfies both guardrails, the script uses a validation-only "
        "minimax fallback. Test metrics never participate in threshold selection.",
        "",
        "## Selected thresholds",
        "",
        "| Model | Threshold | Selection mode | Eligible validation thresholds |",
        "|---|---:|---|---:|",
    ]

    for model_name in MODEL_ORDER:
        row = selected_thresholds[
            selected_thresholds["model_name"].eq(model_name)
        ].iloc[0]
        lines.append(
            f"| {MODEL_LABELS[model_name]} | {float(row['selected_threshold']):.2f} | "
            f"{row['selection_mode']} | {int(row['eligible_threshold_count'])} |"
        )

    lines.extend(
        [
            "",
            "## Test transfer",
            "",
            "| Model | Threshold | F1 change | FPR change | FNR change | "
            "Worst normal FPR change | Worst attack FNR change |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for model_name in MODEL_ORDER:
        model_test = transfer_results[
            transfer_results["model_name"].eq(model_name)
            & transfer_results["split"].eq("test")
        ]
        reference = model_test[
            model_test["operating_point"].eq("reference_0_50")
        ].iloc[0]
        selected = model_test[
            model_test["operating_point"].eq("validation_selected")
        ].iloc[0]

        lines.append(
            f"| {MODEL_LABELS[model_name]} | {float(selected['threshold']):.2f} | "
            f"{float(selected['F1'] - reference['F1']):+.4f} | "
            f"{float(selected['FPR'] - reference['FPR']):+.4f} | "
            f"{float(selected['FNR'] - reference['FNR']):+.4f} | "
            f"{float(selected['worst_normal_run_FPR'] - reference['worst_normal_run_FPR']):+.4f} | "
            f"{float(selected['worst_attack_run_FNR'] - reference['worst_attack_run_FNR']):+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Dominant failure runs",
            "",
            "| Model | Run | True class | Operating point | Threshold | Error rate | "
            "Mean probability | Median probability |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )

    for _, row in dominant_runs.iterrows():
        lines.append(
            f"| {MODEL_LABELS[row['model_name']]} | `{row['run_id']}` | "
            f"{int(row['true_graph'])} | {row['operating_point']} | "
            f"{float(row['threshold']):.2f} | {float(row['error_rate']):.4f} | "
            f"{float(row['mean_graph_probability']):.4f} | "
            f"{float(row['median_graph_probability']):.4f} |"
        )

    lines.extend(["", "## Interpretation", ""])

    meaningful_improvements = 0
    catastrophic_remain = 0

    for model_name in MODEL_ORDER:
        selected_test = transfer_results[
            transfer_results["model_name"].eq(model_name)
            & transfer_results["split"].eq("test")
            & transfer_results["operating_point"].eq("validation_selected")
        ].iloc[0]
        reference_test = transfer_results[
            transfer_results["model_name"].eq(model_name)
            & transfer_results["split"].eq("test")
            & transfer_results["operating_point"].eq("reference_0_50")
        ].iloc[0]

        if (
            selected_test["FPR"] < reference_test["FPR"]
            and selected_test["F1"] >= reference_test["F1"] - 0.01
        ):
            meaningful_improvements += 1

        if (
            selected_test["worst_normal_run_FPR"] > 0.50
            or selected_test["worst_attack_run_FNR"] > 0.50
        ):
            catastrophic_remain += 1

    if meaningful_improvements:
        lines.append(
            f"- Validation-selected calibration improves the FPR/F1 operating trade-off "
            f"for {meaningful_improvements} of four models without using test labels for selection."
        )
    else:
        lines.append(
            "- Validation-selected calibration does not materially improve the FPR/F1 "
            "trade-off for any model."
        )

    if catastrophic_remain:
        lines.append(
            f"- At least one catastrophic run-level failure remains for {catastrophic_remain} "
            "of four models after threshold transfer."
        )
    else:
        lines.append(
            "- No selected operating point retains a run with more than 50% errors."
        )

    lines.extend(
        [
            "- Threshold transfer measures the calibration limit; it does not prove the "
            "cause of the remaining errors.",
            "- If the hard normal run remains above threshold while the weak attack remains "
            "below it, the score ordering—not merely the threshold—is the central problem.",
            "",
        ]
    )

    return "\n".join(lines)


def make_plots(
    validation_sweep: pd.DataFrame,
    transfer_results: pd.DataFrame,
    plot_directory: Path,
) -> None:
    plot_directory.mkdir(parents=True, exist_ok=True)

    for model_name in MODEL_ORDER:
        data = validation_sweep[
            validation_sweep["model_name"].eq(model_name)
        ].sort_values("threshold")

        selected_threshold = float(
            data.loc[data["is_selected_threshold"], "threshold"].iloc[0]
        )

        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(data["threshold"], data["F1"], label="Validation F1")
        axis.plot(data["threshold"], data["FPR"], label="Validation FPR")
        axis.plot(data["threshold"], data["FNR"], label="Validation FNR")
        axis.plot(
            data["threshold"],
            data["worst_normal_run_FPR"],
            label="Worst normal-run FPR",
            linestyle="--",
        )
        axis.plot(
            data["threshold"],
            data["worst_attack_run_FNR"],
            label="Worst attack-run FNR",
            linestyle="--",
        )
        axis.axvline(REFERENCE_THRESHOLD, linestyle=":", label="Reference 0.50")
        axis.axvline(selected_threshold, linestyle="-.", label="Selected threshold")
        axis.set_xlabel("Graph threshold")
        axis.set_ylabel("Metric")
        axis.set_ylim(0.0, 1.02)
        axis.set_title(
            f"Validation threshold sweep: {MODEL_LABELS[model_name]}"
        )
        axis.legend(fontsize=8)
        save_figure(
            figure,
            plot_directory / f"validation_threshold_sweep_{model_name}.png",
        )

    test = transfer_results[transfer_results["split"].eq("test")].copy()
    metrics = ["F1", "FPR", "FNR", "worst_normal_run_FPR", "worst_attack_run_FNR"]

    for metric in metrics:
        pivot = test.pivot(
            index="model_name",
            columns="operating_point",
            values=metric,
        ).reindex(MODEL_ORDER)

        figure, axis = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=axis)
        axis.set_ylabel(metric)
        axis.set_xlabel("Model")
        axis.set_title(f"Test {metric}: reference versus validation-selected threshold")
        axis.set_xticklabels(
            [MODEL_LABELS[name] for name in pivot.index],
            rotation=25,
            ha="right",
        )
        save_figure(
            figure,
            plot_directory / f"test_{metric.lower()}_threshold_transfer.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select graph thresholds on V3 validation predictions only and apply "
            "them unchanged to the test split."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("reports/v3_failure_analysis"),
    )
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    prediction_directory = root / "predictions"
    table_directory = root / "tables"
    log_directory = root / "logs"
    plot_directory = root / "plots" / "threshold_transfer"

    metadata_path = prediction_directory / "prediction_metadata.csv"
    stage3_metrics_path = table_directory / "exported_metric_reproduction.csv"

    output_paths = {
        "validation_sweep": table_directory / "graph_threshold_sweep_validation.csv",
        "selected_thresholds": table_directory / "selected_graph_thresholds.csv",
        "transfer_results": table_directory / "graph_threshold_transfer_results.csv",
        "transfer_changes": table_directory / "graph_threshold_transfer_changes.csv",
        "by_run": table_directory / "graph_threshold_transfer_by_run.csv",
        "dominant_runs": table_directory / "dominant_run_threshold_transfer.csv",
        "report": root / "GRAPH_THRESHOLD_TRANSFER.md",
        "validation": log_directory / "30_graph_threshold_transfer_validation.txt",
    }

    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Stage 5 outputs already exist. Use --overwrite to replace them:\n"
            + "\n".join(str(path) for path in existing)
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not stage3_metrics_path.is_file():
        raise FileNotFoundError(stage3_metrics_path)

    print(f"root: {root}", flush=True)
    print(f"metadata: {metadata_path}", flush=True)
    print(f"stage3 metrics: {stage3_metrics_path}", flush=True)
    print(
        f"threshold grid: {args.threshold_min:.2f} to {args.threshold_max:.2f} "
        f"step {args.threshold_step:.2f}",
        flush=True,
    )

    metadata, loaded = load_inputs(prediction_directory, metadata_path)
    verify_alignment(metadata, loaded)
    print("input alignment verified", flush=True)

    threshold_grid = create_threshold_grid(
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    )
    print(f"threshold count: {len(threshold_grid)}", flush=True)

    sweep_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    by_run_frames: list[pd.DataFrame] = []

    for model_index, model_name in enumerate(MODEL_ORDER, start=1):
        print()
        print("=" * 100)
        print(f"MODEL {model_index}/{len(MODEL_ORDER)}: {model_name}")
        print("=" * 100)

        model_frame = build_model_frame(metadata, loaded[model_name], model_name)
        validation_frame = model_frame[model_frame["split"].eq("val")].copy()
        test_frame = model_frame[model_frame["split"].eq("test")].copy()

        model_sweep_rows: list[dict[str, Any]] = []

        for threshold in threshold_grid:
            summary, _ = evaluate_threshold(validation_frame, float(threshold))
            summary.update(
                {
                    "model_name": model_name,
                    "split": "val",
                    "threshold_distance_from_0_50": abs(
                        float(threshold) - REFERENCE_THRESHOLD
                    ),
                }
            )
            model_sweep_rows.append(summary)

        model_sweep = pd.DataFrame(model_sweep_rows)
        selected_row, selection_mode, eligible_count = select_validation_threshold(
            model_sweep
        )
        selected_threshold = float(selected_row["threshold"])

        model_sweep["is_selected_threshold"] = np.isclose(
            model_sweep["threshold"],
            selected_threshold,
            atol=1e-12,
        )
        sweep_rows.extend(model_sweep.to_dict(orient="records"))

        selected_rows.append(
            {
                "model_name": model_name,
                "selected_threshold": selected_threshold,
                "selection_mode": selection_mode,
                "eligible_threshold_count": eligible_count,
                "threshold_grid_min": float(threshold_grid.min()),
                "threshold_grid_max": float(threshold_grid.max()),
                "threshold_grid_step": float(args.threshold_step),
                "selection_used_test_data": False,
                "validation_selection_score": float(
                    selected_row["validation_selection_score"]
                ),
                "validation_F1": float(selected_row["F1"]),
                "validation_FPR": float(selected_row["FPR"]),
                "validation_FNR": float(selected_row["FNR"]),
                "validation_run_macro_error_rate": float(
                    selected_row["run_macro_error_rate"]
                ),
                "validation_worst_normal_run_FPR": float(
                    selected_row["worst_normal_run_FPR"]
                ),
                "validation_worst_attack_run_FNR": float(
                    selected_row["worst_attack_run_FNR"]
                ),
                "validation_catastrophic_normal_runs": int(
                    selected_row["catastrophic_normal_runs"]
                ),
                "validation_catastrophic_attack_runs": int(
                    selected_row["catastrophic_attack_runs"]
                ),
            }
        )

        print(
            f"selected threshold: {selected_threshold:.2f} "
            f"({selection_mode}; eligible={eligible_count})",
            flush=True,
        )

        for split_name, split_frame in [
            ("val", validation_frame),
            ("test", test_frame),
        ]:
            for operating_point, threshold, source in [
                (
                    "reference_0_50",
                    REFERENCE_THRESHOLD,
                    "fixed_reference",
                ),
                (
                    "validation_selected",
                    selected_threshold,
                    "validation_only",
                ),
            ]:
                summary, runs = evaluate_threshold(split_frame, threshold)

                result_rows.append(
                    add_operating_point(
                        summary,
                        model_name=model_name,
                        split_name=split_name,
                        operating_point=operating_point,
                        threshold_source=source,
                    )
                )

                runs["operating_point"] = operating_point
                runs["threshold_source"] = source
                by_run_frames.append(runs)

                print(
                    f"{split_name:4s} {operating_point:19s} "
                    f"thr={threshold:.2f} "
                    f"F1={summary['F1']:.6f} "
                    f"FPR={summary['FPR']:.6f} "
                    f"FNR={summary['FNR']:.6f} "
                    f"worst_normal={summary['worst_normal_run_FPR']:.6f} "
                    f"worst_attack={summary['worst_attack_run_FNR']:.6f}",
                    flush=True,
                )

    validation_sweep = pd.DataFrame(sweep_rows)
    selected_thresholds = pd.DataFrame(selected_rows)
    transfer_results = pd.DataFrame(result_rows)
    by_run = pd.concat(by_run_frames, ignore_index=True)
    transfer_changes = compare_operating_points(transfer_results)
    dominant_runs = build_dominant_run_table(by_run)

    stage3_metrics = pd.read_csv(stage3_metrics_path, low_memory=False)
    reference_matched, validation_messages = validate_reference_metrics(
        transfer_results,
        stage3_metrics,
    )

    input_alignment_verified = True
    threshold_selection_test_free = bool(
        (~selected_thresholds["selection_used_test_data"]).all()
    )
    all_selected_thresholds_on_grid = bool(
        selected_thresholds["selected_threshold"].apply(
            lambda value: np.any(
                np.isclose(threshold_grid, float(value), atol=1e-12)
            )
        ).all()
    )
    all_operating_points_present = bool(
        (
            transfer_results.groupby(["model_name", "split"])["operating_point"]
            .nunique()
            .eq(2)
        ).all()
    )
    all_samples_accounted_for = True

    expected_split_counts = (
        metadata.groupby("split")["sample_id"].count().to_dict()
    )
    for (model_name, split_name, operating_point), group in by_run.groupby(
        ["model_name", "split", "operating_point"]
    ):
        expected_count = int(expected_split_counts[split_name])
        actual_count = int(group["number_of_windows"].sum())
        if actual_count != expected_count:
            all_samples_accounted_for = False
            validation_messages.append(
                f"{model_name} {split_name} {operating_point}: "
                f"{actual_count} windows != {expected_count}"
            )

    numeric_columns = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "F1",
        "FPR",
        "FNR",
        "balanced_accuracy",
        "run_macro_error_rate",
        "mean_normal_run_FPR",
        "mean_attack_run_FNR",
        "worst_normal_run_FPR",
        "worst_attack_run_FNR",
    ]
    all_statistics_finite = bool(
        np.isfinite(
            transfer_results[numeric_columns].to_numpy(dtype=float)
        ).all()
    )
    dominant_runs_present = bool(
        set(dominant_runs["run_id"].unique())
        == {HARD_NORMAL_RUN, HARD_ATTACK_RUN}
    )

    overall_success = all(
        [
            input_alignment_verified,
            reference_matched,
            threshold_selection_test_free,
            all_selected_thresholds_on_grid,
            all_operating_points_present,
            all_samples_accounted_for,
            all_statistics_finite,
            dominant_runs_present,
        ]
    )

    validation_lines = [
        f"input alignment verified: {input_alignment_verified}",
        f"reference 0.50 metrics matched Stage 3: {reference_matched}",
        f"threshold selection used validation only: {threshold_selection_test_free}",
        f"all selected thresholds belong to validation grid: {all_selected_thresholds_on_grid}",
        f"all operating points present: {all_operating_points_present}",
        f"all samples accounted for: {all_samples_accounted_for}",
        f"all statistics finite: {all_statistics_finite}",
        f"dominant failure runs present: {dominant_runs_present}",
        f"overall success: {overall_success}",
    ]
    if validation_messages:
        validation_lines.extend(["", "DETAILS:", *validation_messages])
    validation_text = "\n".join(validation_lines) + "\n"

    atomic_text(validation_text, output_paths["validation"])
    print()
    print(validation_text, end="", flush=True)

    if not overall_success:
        raise SystemExit(
            f"Stage 5 validation failed. Inspect {output_paths['validation']}."
        )

    atomic_csv(validation_sweep, output_paths["validation_sweep"])
    atomic_csv(selected_thresholds, output_paths["selected_thresholds"])
    atomic_csv(transfer_results, output_paths["transfer_results"])
    atomic_csv(transfer_changes, output_paths["transfer_changes"])
    atomic_csv(by_run, output_paths["by_run"])
    atomic_csv(dominant_runs, output_paths["dominant_runs"])

    report = build_report(
        selected_thresholds,
        transfer_results,
        dominant_runs,
    )
    atomic_text(report, output_paths["report"])

    make_plots(validation_sweep, transfer_results, plot_directory)

    print("Stage 5 output files:", flush=True)
    for name, path in output_paths.items():
        print(f"  {name}: {path}", flush=True)
    print(f"  plots: {plot_directory}", flush=True)
    print("overall success: True", flush=True)


if __name__ == "__main__":
    main()
