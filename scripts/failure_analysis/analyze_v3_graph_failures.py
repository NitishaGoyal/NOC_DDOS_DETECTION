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

THRESHOLD = 0.50

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


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        fig.savefig(tmp, dpi=180, bbox_inches="tight")
        os.replace(tmp, path)
    finally:
        plt.close(fig)
        tmp.unlink(missing_ok=True)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true, dtype=np.int8).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int8).reshape(-1)
    return {
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


def metrics_from_counts(c: dict[str, int]) -> dict[str, float]:
    tn, fp, fn, tp = c["TN"], c["FP"], c["FN"], c["TP"]
    accuracy = safe_divide(tn + tp, tn + fp + fn + tp)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else float("nan")
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": float(f1),
        "FPR": safe_divide(fp, fp + tn),
        "FNR": safe_divide(fn, fn + tp),
    }


def parse_active_core_count(value: Any) -> int:
    text = str(value).strip()
    if not text or text.lower() == "idle":
        return 0
    return len([token for token in text.split("-") if token.strip()])


def classify_confidence(true_graph: int, probability: float, is_error: bool) -> str:
    if not is_error:
        return "correct"
    if 0.40 <= probability <= 0.60:
        return "borderline"
    if true_graph == 0:
        return "high_confidence_error" if probability > 0.80 else "moderate_confidence_error"
    return "high_confidence_error" if probability < 0.20 else "moderate_confidence_error"


def percentile(series: pd.Series, p: float) -> float:
    return float(np.percentile(series.to_numpy(dtype=float), p))


def load_predictions(prediction_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for model_name, filename in MODEL_FILES.items():
        path = prediction_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction file: {path}")
        with np.load(path, allow_pickle=False) as src:
            loaded[model_name] = {key: src[key] for key in src.files}
        print(f"loaded {model_name}: {path}", flush=True)
    return loaded


def verify_alignment(metadata: pd.DataFrame, loaded: dict[str, dict[str, np.ndarray]]) -> None:
    required_metadata = {
        "sample_order_within_export", "sample_index", "sample_id", "run_id",
        "split", "end_epoch", "true_graph", "true_attacker_count", "profile",
        "active_cores", "attackers", "strength", "seed",
    }
    missing = required_metadata - set(metadata.columns)
    if missing:
        raise RuntimeError(f"Metadata missing columns: {sorted(missing)}")

    reference = None
    for model_name, arrays in loaded.items():
        required_arrays = {
            "sample_index", "sample_id", "split", "true_graph",
            "graph_logit", "graph_probability",
        }
        missing_arrays = required_arrays - set(arrays)
        if missing_arrays:
            raise RuntimeError(f"{model_name} missing arrays: {sorted(missing_arrays)}")
        if len(arrays["sample_index"]) != len(metadata):
            raise RuntimeError(f"{model_name} sample count differs from metadata")

        expected = {
            "sample_index": metadata["sample_index"].to_numpy(dtype=np.int64),
            "sample_id": metadata["sample_id"].astype(str).to_numpy(),
            "split": metadata["split"].astype(str).to_numpy(),
            "true_graph": metadata["true_graph"].to_numpy(dtype=np.int64),
        }
        for key, exp in expected.items():
            act = np.asarray(arrays[key]).astype(exp.dtype)
            if not np.array_equal(act, exp):
                raise RuntimeError(f"{model_name} metadata alignment failed for {key}")

        if reference is None:
            reference = arrays
        else:
            for key in ["sample_index", "sample_id", "split", "true_graph"]:
                if key in {"sample_id", "split"}:
                    left = np.asarray(reference[key]).astype(str)
                    right = np.asarray(arrays[key]).astype(str)
                else:
                    left = np.asarray(reference[key], dtype=np.int64)
                    right = np.asarray(arrays[key], dtype=np.int64)
                if not np.array_equal(left, right):
                    raise RuntimeError(f"Cross-model alignment failed for {model_name}: {key}")


def build_window_table(
    metadata: pd.DataFrame,
    loaded: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    base = metadata.rename(columns={"sample_order_within_export": "sample_order"}).copy()
    base["sample_index"] = base["sample_index"].astype(int)
    base["end_epoch"] = base["end_epoch"].astype(int)
    base["true_graph"] = base["true_graph"].astype(int)
    base["attacker_count"] = base["true_attacker_count"].astype(int)
    base["active_core_count"] = base["active_cores"].map(parse_active_core_count).astype(int)

    frames = []
    for model_name in MODEL_ORDER:
        arrays = loaded[model_name]
        frame = base.copy()
        frame["model_name"] = model_name
        frame["graph_logit"] = np.asarray(arrays["graph_logit"], dtype=np.float64)
        frame["graph_probability"] = np.asarray(arrays["graph_probability"], dtype=np.float64)

        numeric = frame[["graph_logit", "graph_probability"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"Non-finite outputs for {model_name}")

        frame["pred_graph"] = (frame["graph_probability"] >= THRESHOLD).astype(int)
        truth = frame["true_graph"].to_numpy(dtype=int)
        pred = frame["pred_graph"].to_numpy(dtype=int)
        frame["error_type"] = np.select(
            [
                (truth == 0) & (pred == 0),
                (truth == 0) & (pred == 1),
                (truth == 1) & (pred == 0),
                (truth == 1) & (pred == 1),
            ],
            ["TN", "FP", "FN", "TP"],
            default="UNKNOWN",
        )
        frame["is_correct"] = truth == pred
        frame["is_error"] = ~frame["is_correct"]
        frame["is_false_positive"] = frame["error_type"].eq("FP")
        frame["is_false_negative"] = frame["error_type"].eq("FN")
        frame["probability_margin"] = frame["graph_probability"] - THRESHOLD
        frame["absolute_threshold_distance"] = frame["probability_margin"].abs()
        frame["confidence_category"] = [
            classify_confidence(int(t), float(p), bool(e))
            for t, p, e in zip(
                frame["true_graph"],
                frame["graph_probability"],
                frame["is_error"],
            )
        ]
        frame["high_confidence_error"] = frame["confidence_category"].eq("high_confidence_error")
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(subset=["model_name", "sample_id"]).any():
        raise RuntimeError("Duplicate model/sample rows created")

    return result[
        [
            "sample_order", "sample_index", "sample_id", "run_id", "split",
            "end_epoch", "model_name", "true_graph", "graph_logit",
            "graph_probability", "pred_graph", "error_type", "is_correct",
            "is_error", "is_false_positive", "is_false_negative",
            "probability_margin", "absolute_threshold_distance",
            "confidence_category", "high_confidence_error", "profile",
            "active_cores", "active_core_count", "attackers", "attacker_count",
            "strength", "seed",
        ]
    ]


def build_error_streaks(window_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    priority = {"high_confidence_error": 3, "moderate_confidence_error": 2, "borderline": 1}

    for (model_name, split_name, run_id), run in window_table.groupby(
        ["model_name", "split", "run_id"], sort=True
    ):
        errors = run.sort_values("end_epoch")
        errors = errors[errors["is_error"]]
        if errors.empty:
            continue

        current: list[pd.Series] = []
        previous_epoch = None
        previous_error_type = None
        streak_id = 0

        def flush() -> None:
            nonlocal current, streak_id
            if not current:
                return
            streak_id += 1
            streak = pd.DataFrame(current)
            counts = streak["confidence_category"].value_counts()
            dominant = sorted(
                counts.index,
                key=lambda category: (counts[category], priority.get(category, 0)),
                reverse=True,
            )[0]
            rows.append(
                {
                    "model_name": model_name,
                    "split": split_name,
                    "run_id": run_id,
                    "true_graph": int(streak["true_graph"].iloc[0]),
                    "error_type": str(streak["error_type"].iloc[0]),
                    "streak_id": streak_id,
                    "start_end_epoch": int(streak["end_epoch"].iloc[0]),
                    "finish_end_epoch": int(streak["end_epoch"].iloc[-1]),
                    "streak_length": int(len(streak)),
                    "mean_probability": float(streak["graph_probability"].mean()),
                    "minimum_probability": float(streak["graph_probability"].min()),
                    "maximum_probability": float(streak["graph_probability"].max()),
                    "confidence_category": dominant,
                }
            )
            current = []

        for _, row in errors.iterrows():
            epoch = int(row["end_epoch"])
            error_type = str(row["error_type"])
            consecutive = (
                previous_epoch is not None
                and epoch == previous_epoch + 1
                and error_type == previous_error_type
            )
            if current and not consecutive:
                flush()
            current.append(row)
            previous_epoch = epoch
            previous_error_type = error_type
        flush()

    columns = [
        "model_name", "split", "run_id", "true_graph", "error_type",
        "streak_id", "start_end_epoch", "finish_end_epoch", "streak_length",
        "mean_probability", "minimum_probability", "maximum_probability",
        "confidence_category",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_run_metrics(window_table: pd.DataFrame, streaks: pd.DataFrame) -> pd.DataFrame:
    streak_summary: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not streaks.empty:
        for key, group in streaks.groupby(["model_name", "split", "run_id"]):
            streak_summary[key] = {
                "number_of_error_streaks": int(len(group)),
                "longest_error_streak": int(group["streak_length"].max()),
                "mean_error_streak_length": float(group["streak_length"].mean()),
                "median_error_streak_length": float(group["streak_length"].median()),
            }

    rows = []
    for key, run in window_table.groupby(["model_name", "split", "run_id"], sort=True):
        model_name, split_name, run_id = key
        ordered = run.sort_values("end_epoch")
        counts = binary_counts(ordered["true_graph"], ordered["pred_graph"])
        metrics = metrics_from_counts(counts)
        probabilities = ordered["graph_probability"]
        errors = ordered[ordered["is_error"]]
        streak_info = streak_summary.get(
            key,
            {
                "number_of_error_streaks": 0,
                "longest_error_streak": 0,
                "mean_error_streak_length": 0.0,
                "median_error_streak_length": 0.0,
            },
        )

        rows.append(
            {
                "model_name": model_name,
                "split": split_name,
                "run_id": run_id,
                "profile": str(ordered["profile"].iloc[0]),
                "active_cores": str(ordered["active_cores"].iloc[0]),
                "active_core_count": int(ordered["active_core_count"].iloc[0]),
                "attackers": str(ordered["attackers"].iloc[0]),
                "attacker_count": int(ordered["attacker_count"].iloc[0]),
                "strength": str(ordered["strength"].iloc[0]),
                "seed": str(ordered["seed"].iloc[0]),
                "true_graph": int(ordered["true_graph"].iloc[0]),
                "number_of_windows": int(len(ordered)),
                **counts,
                "accuracy": metrics["accuracy"],
                "error_rate": float(ordered["is_error"].mean()),
                "FPR": metrics["FPR"],
                "FNR": metrics["FNR"],
                "mean_graph_probability": float(probabilities.mean()),
                "std_graph_probability": float(probabilities.std(ddof=0)),
                "minimum_graph_probability": float(probabilities.min()),
                "maximum_graph_probability": float(probabilities.max()),
                "p01": percentile(probabilities, 1),
                "p05": percentile(probabilities, 5),
                "p25": percentile(probabilities, 25),
                "p50": percentile(probabilities, 50),
                "p75": percentile(probabilities, 75),
                "p95": percentile(probabilities, 95),
                "p99": percentile(probabilities, 99),
                "number_of_error_windows": int(len(errors)),
                "number_of_borderline_errors": int(errors["confidence_category"].eq("borderline").sum()),
                "number_of_moderate_confidence_errors": int(
                    errors["confidence_category"].eq("moderate_confidence_error").sum()
                ),
                "number_of_high_confidence_errors": int(
                    errors["confidence_category"].eq("high_confidence_error").sum()
                ),
                "first_error_epoch": int(errors["end_epoch"].min()) if not errors.empty else np.nan,
                "last_error_epoch": int(errors["end_epoch"].max()) if not errors.empty else np.nan,
                "fraction_of_run_in_error": float(ordered["is_error"].mean()),
                **streak_info,
            }
        )
    return pd.DataFrame(rows)


def build_fp_fn_tables(run_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fp = run_metrics[run_metrics["true_graph"].eq(0)].copy()
    fp["number_of_FP_windows"] = fp["FP"]
    fp["no_false_positives"] = fp["FP"].eq(0)
    fp["more_than_50_percent_FP"] = fp["fraction_of_run_in_error"] > 0.50
    fp["more_than_80_percent_FP"] = fp["fraction_of_run_in_error"] > 0.80
    fp = fp.sort_values(
        ["model_name", "split", "fraction_of_run_in_error", "number_of_FP_windows"],
        ascending=[True, True, False, False],
    )

    fn = run_metrics[run_metrics["true_graph"].eq(1)].copy()
    fn["number_of_FN_windows"] = fn["FN"]
    fn["no_false_negatives"] = fn["FN"].eq(0)
    fn["more_than_50_percent_FN"] = fn["fraction_of_run_in_error"] > 0.50
    fn = fn.sort_values(
        ["model_name", "split", "fraction_of_run_in_error", "number_of_FN_windows"],
        ascending=[True, True, False, False],
    )
    return fp, fn


def calculate_group_metrics(group: pd.DataFrame) -> dict[str, Any]:
    counts = binary_counts(group["true_graph"], group["pred_graph"])
    metrics = metrics_from_counts(counts)
    return {
        "number_of_runs": int(group["run_id"].nunique()),
        "number_of_windows": int(len(group)),
        **counts,
        **metrics,
        "mean_graph_probability": float(group["graph_probability"].mean()),
        "std_graph_probability": float(group["graph_probability"].std(ddof=0)),
    }


def build_group_metrics(window_table: pd.DataFrame) -> pd.DataFrame:
    working = window_table.copy()
    working["normal_vs_attack"] = np.where(working["true_graph"].eq(0), "normal", "attack")
    variables = [
        "profile", "active_core_count", "attacker_count", "strength",
        "attackers", "active_cores", "seed", "normal_vs_attack",
    ]
    rows = []
    for variable in variables:
        temporary = working.assign(_group_value=working[variable].astype(str).replace("", "<none>"))
        for (model_name, split_name, value), group in temporary.groupby(
            ["model_name", "split", "_group_value"], sort=True
        ):
            rows.append(
                {
                    "model_name": model_name,
                    "split": split_name,
                    "grouping_variable": variable,
                    "group_value": value,
                    **calculate_group_metrics(group),
                }
            )
    return pd.DataFrame(rows)


def build_error_concentration(run_metrics: pd.DataFrame) -> pd.DataFrame:
    selectors = [
        ("worst_1_run", lambda n: 1),
        ("worst_2_runs", lambda n: 2),
        ("worst_3_runs", lambda n: 3),
        ("worst_5_percent", lambda n: math.ceil(n * 0.05)),
        ("worst_10_percent", lambda n: math.ceil(n * 0.10)),
        ("worst_20_percent", lambda n: math.ceil(n * 0.20)),
        ("all_eligible_runs", lambda n: n),
    ]
    rows = []
    for model_name in MODEL_ORDER:
        for split_name in ["val", "test"]:
            for error_type, true_graph, column in [("FP", 0, "FP"), ("FN", 1, "FN")]:
                eligible = run_metrics[
                    run_metrics["model_name"].eq(model_name)
                    & run_metrics["split"].eq(split_name)
                    & run_metrics["true_graph"].eq(true_graph)
                ].copy()
                eligible = eligible.sort_values(
                    ["fraction_of_run_in_error", column], ascending=[False, False]
                )
                n = len(eligible)
                if n == 0:
                    continue
                total_errors = int(eligible[column].sum())
                for name, function in selectors:
                    selected_count = min(n, max(1, function(n)))
                    selected = eligible.head(selected_count)
                    selected_errors = int(selected[column].sum())
                    share = selected_errors / total_errors if total_errors else 0.0
                    rows.append(
                        {
                            "model_name": model_name,
                            "split": split_name,
                            "error_type": error_type,
                            "selection": name,
                            "eligible_run_count": n,
                            "selected_run_count": selected_count,
                            "selected_error_windows": selected_errors,
                            "total_error_windows": total_errors,
                            "window_weighted_error_share": share,
                            "run_weighted_mean_error_rate": float(
                                selected["fraction_of_run_in_error"].mean()
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def build_val_test_gap(group_metrics: pd.DataFrame) -> pd.DataFrame:
    requested = {"profile", "attacker_count", "strength", "active_core_count", "normal_vs_attack"}
    metric_names = ["accuracy", "precision", "recall", "F1", "FPR", "FNR", "mean_graph_probability"]
    selected = group_metrics[group_metrics["grouping_variable"].isin(requested)]
    rows = []
    for (model_name, variable, value), group in selected.groupby(
        ["model_name", "grouping_variable", "group_value"], sort=True
    ):
        split_rows = {row["split"]: row for _, row in group.iterrows()}
        if "val" not in split_rows or "test" not in split_rows:
            continue
        val_row, test_row = split_rows["val"], split_rows["test"]
        for metric in metric_names:
            val_metric = float(val_row[metric])
            test_metric = float(test_row[metric])
            if not (np.isfinite(val_metric) and np.isfinite(test_metric)):
                continue
            absolute_gap = test_metric - val_metric
            relative_gap = absolute_gap / abs(val_metric) if val_metric != 0 else float("nan")
            rows.append(
                {
                    "model_name": model_name,
                    "grouping_variable": variable,
                    "group_value": value,
                    "metric": metric,
                    "validation_metric": val_metric,
                    "test_metric": test_metric,
                    "absolute_gap": absolute_gap,
                    "relative_gap": relative_gap,
                    "validation_run_count": int(val_row["number_of_runs"]),
                    "test_run_count": int(test_row["number_of_runs"]),
                    "validation_window_count": int(val_row["number_of_windows"]),
                    "test_window_count": int(test_row["number_of_windows"]),
                }
            )
    return pd.DataFrame(rows)


def build_model_agreement(window_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_columns = [
        "sample_order", "sample_index", "sample_id", "run_id", "split",
        "end_epoch", "true_graph", "profile", "active_cores",
        "active_core_count", "attackers", "attacker_count", "strength", "seed",
    ]
    agreement = (
        window_table[metadata_columns]
        .drop_duplicates(subset=["sample_id"])
        .sort_values("sample_order")
        .reset_index(drop=True)
    )

    pred_pivot = window_table.pivot(index="sample_id", columns="model_name", values="pred_graph")
    correct_pivot = window_table.pivot(index="sample_id", columns="model_name", values="is_correct")
    prob_pivot = window_table.pivot(index="sample_id", columns="model_name", values="graph_probability")

    for model_name in MODEL_ORDER:
        agreement[f"pred_{model_name}"] = agreement["sample_id"].map(pred_pivot[model_name]).astype(int)
        agreement[f"correct_{model_name}"] = agreement["sample_id"].map(correct_pivot[model_name]).astype(bool)
        agreement[f"probability_{model_name}"] = agreement["sample_id"].map(prob_pivot[model_name]).astype(float)

    correctness_columns = [f"correct_{name}" for name in MODEL_ORDER]
    agreement["correct_model_count"] = agreement[correctness_columns].sum(axis=1).astype(int)
    agreement["wrong_model_count"] = len(MODEL_ORDER) - agreement["correct_model_count"]
    agreement["all_four_correct"] = agreement["correct_model_count"].eq(4)
    agreement["all_four_wrong"] = agreement["correct_model_count"].eq(0)

    for model_name in MODEL_ORDER:
        others = [f"correct_{other}" for other in MODEL_ORDER if other != model_name]
        agreement[f"only_{model_name}_correct"] = (
            agreement[f"correct_{model_name}"] & ~agreement[others].any(axis=1)
        )

    agreement["conv1d_correct_all_tcns_wrong"] = agreement["only_conv1d_gcn_correct"]
    agreement["all_tcns_correct_conv1d_wrong"] = (
        ~agreement["correct_conv1d_gcn"]
        & agreement[
            [
                "correct_tcn_attention_gcn",
                "correct_tcn_meanpool_gcn",
                "correct_tcn_maxpool_gcn",
            ]
        ].all(axis=1)
    )
    agreement["exactly_one_model_wrong"] = agreement["wrong_model_count"].eq(1)
    agreement["exactly_two_models_wrong"] = agreement["wrong_model_count"].eq(2)
    agreement["exactly_three_models_wrong"] = agreement["wrong_model_count"].eq(3)
    agreement["attention_only_failure"] = (
        ~agreement["correct_tcn_attention_gcn"]
        & agreement[
            ["correct_conv1d_gcn", "correct_tcn_meanpool_gcn", "correct_tcn_maxpool_gcn"]
        ].all(axis=1)
    )
    agreement["maxpool_only_success"] = (
        agreement["correct_tcn_maxpool_gcn"]
        & ~agreement[
            ["correct_conv1d_gcn", "correct_tcn_attention_gcn", "correct_tcn_meanpool_gcn"]
        ].any(axis=1)
    )

    def primary(row: pd.Series) -> str:
        if row["all_four_correct"]:
            return "all_four_correct"
        if row["all_four_wrong"]:
            return "all_four_wrong"
        for model_name in MODEL_ORDER:
            if row[f"only_{model_name}_correct"]:
                return f"only_{model_name}_correct"
        if row["all_tcns_correct_conv1d_wrong"]:
            return "all_tcn_models_correct_conv1d_wrong"
        if row["exactly_one_model_wrong"]:
            return "exactly_one_model_wrong"
        if row["exactly_two_models_wrong"]:
            return "exactly_two_models_wrong"
        if row["exactly_three_models_wrong"]:
            return "exactly_three_models_wrong"
        return "other"

    agreement["primary_agreement_category"] = agreement.apply(primary, axis=1)

    run_rows = []
    for (split_name, run_id), run in agreement.groupby(["split", "run_id"], sort=True):
        same_fp = run["all_four_wrong"] & run["true_graph"].eq(0)
        same_fn = run["all_four_wrong"] & run["true_graph"].eq(1)
        tcn_fail_conv_ok = run["conv1d_correct_all_tcns_wrong"]
        conv_fail_tcn_ok = run["all_tcns_correct_conv1d_wrong"]
        attention_only = run["attention_only_failure"]
        maxpool_only = run["maxpool_only_success"]
        run_rows.append(
            {
                "split": split_name,
                "run_id": run_id,
                "true_graph": int(run["true_graph"].iloc[0]),
                "profile": str(run["profile"].iloc[0]),
                "active_cores": str(run["active_cores"].iloc[0]),
                "active_core_count": int(run["active_core_count"].iloc[0]),
                "attackers": str(run["attackers"].iloc[0]),
                "attacker_count": int(run["attacker_count"].iloc[0]),
                "strength": str(run["strength"].iloc[0]),
                "number_of_windows": int(len(run)),
                "all_models_same_FP_count": int(same_fp.sum()),
                "all_models_same_FN_count": int(same_fn.sum()),
                "tcn_models_fail_conv1d_succeeds_count": int(tcn_fail_conv_ok.sum()),
                "conv1d_fails_all_tcn_models_succeed_count": int(conv_fail_tcn_ok.sum()),
                "attention_uniquely_fails_count": int(attention_only.sum()),
                "maxpool_uniquely_succeeds_count": int(maxpool_only.sum()),
                "all_models_same_error_fraction": float((same_fp | same_fn).mean()),
                "tcn_fail_conv1d_succeeds_fraction": float(tcn_fail_conv_ok.mean()),
                "conv1d_fails_all_tcn_succeed_fraction": float(conv_fail_tcn_ok.mean()),
            }
        )
    return agreement, pd.DataFrame(run_rows)


def make_plots(
    window_table: pd.DataFrame,
    run_metrics: pd.DataFrame,
    streaks: pd.DataFrame,
    agreement: pd.DataFrame,
    plot_dir: Path,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    test_runs = run_metrics[run_metrics["split"].eq("test")]

    normal = test_runs[test_runs["true_graph"].eq(0)]
    if not normal.empty:
        pivot = normal.pivot(index="run_id", columns="model_name", values="FPR").reindex(columns=MODEL_ORDER)
        fig, ax = plt.subplots(figsize=(max(11, len(pivot) * 1.3), 6))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("False-positive rate")
        ax.set_xlabel("Normal test run")
        ax.set_title("Test false-positive rate by normal run and model")
        ax.tick_params(axis="x", labelrotation=45)
        ax.legend([MODEL_LABELS[name] for name in pivot.columns])
        save_figure(fig, plot_dir / "test_false_positive_rate_by_run.png")

    attack = test_runs[test_runs["true_graph"].eq(1)]
    if not attack.empty:
        pivot = attack.pivot(index="run_id", columns="model_name", values="FNR").reindex(columns=MODEL_ORDER)
        fig, ax = plt.subplots(figsize=(max(12, len(pivot) * 1.3), 6))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("False-negative rate")
        ax.set_xlabel("Attack test run")
        ax.set_title("Test false-negative rate by attack run and model")
        ax.tick_params(axis="x", labelrotation=45)
        ax.legend([MODEL_LABELS[name] for name in pivot.columns])
        save_figure(fig, plot_dir / "test_false_negative_rate_by_run.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    for model_name in MODEL_ORDER:
        for error_type, true_graph, marker in [("FP", 0, "o"), ("FN", 1, "x")]:
            eligible = test_runs[
                test_runs["model_name"].eq(model_name)
                & test_runs["true_graph"].eq(true_graph)
            ].sort_values("fraction_of_run_in_error", ascending=False)
            total = float(eligible[error_type].sum())
            if eligible.empty or total <= 0:
                continue
            cumulative = eligible[error_type].cumsum().to_numpy(dtype=float) / total
            ax.plot(
                np.arange(1, len(cumulative) + 1),
                cumulative,
                marker=marker,
                label=f"{MODEL_LABELS[model_name]} {error_type}",
            )
    ax.set_xlabel("Number of worst runs included")
    ax.set_ylabel("Cumulative share of error windows")
    ax.set_ylim(0, 1.05)
    ax.set_title("Test error-window concentration")
    ax.legend(fontsize=8)
    save_figure(fig, plot_dir / "test_error_concentration_curve.png")

    aggregate_rows = []
    for model_name in MODEL_ORDER:
        for split_name in ["val", "test"]:
            subset = window_table[
                window_table["model_name"].eq(model_name)
                & window_table["split"].eq(split_name)
            ]
            metrics = metrics_from_counts(binary_counts(subset["true_graph"], subset["pred_graph"]))
            aggregate_rows.append(
                {
                    "model_name": model_name,
                    "split": split_name,
                    "FPR": metrics["FPR"],
                    "FNR": metrics["FNR"],
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    for metric in ["FPR", "FNR"]:
        pivot = aggregate.pivot(index="model_name", columns="split", values=metric).reindex(MODEL_ORDER)
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(metric)
        ax.set_xlabel("Model")
        ax.set_title(f"Validation versus test {metric}")
        ax.set_xticklabels([MODEL_LABELS[name] for name in pivot.index], rotation=25, ha="right")
        save_figure(fig, plot_dir / f"val_test_{metric.lower()}_by_model.png")

    test_windows = window_table[window_table["split"].eq("test")]
    for model_name in MODEL_ORDER:
        data = test_windows[test_windows["model_name"].eq(model_name)]
        run_ids = sorted(data["run_id"].unique())
        values = [
            data.loc[data["run_id"].eq(run_id), "graph_probability"].to_numpy()
            for run_id in run_ids
        ]
        fig, ax = plt.subplots(figsize=(max(14, len(run_ids) * 0.9), 6))
        ax.boxplot(values, tick_labels=run_ids, showfliers=False)
        ax.axhline(THRESHOLD, linestyle="--")
        ax.set_ylabel("Graph probability")
        ax.set_xlabel("Test run")
        ax.set_title(f"Graph probability by test run: {MODEL_LABELS[model_name]}")
        ax.tick_params(axis="x", labelrotation=55)
        save_figure(fig, plot_dir / f"graph_probability_boxplot_test_{model_name}.png")

    if not streaks.empty:
        maximum = max(2, int(streaks["streak_length"].max()))
        bins = np.unique(np.clip(np.logspace(0, math.log10(maximum), 30).astype(int), 1, None))
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_name in MODEL_ORDER:
            values = streaks.loc[
                streaks["model_name"].eq(model_name) & streaks["split"].eq("test"),
                "streak_length",
            ].to_numpy(dtype=int)
            if values.size:
                ax.hist(
                    values,
                    bins=bins,
                    histtype="step",
                    linewidth=1.5,
                    label=MODEL_LABELS[model_name],
                )
        ax.set_xscale("log")
        ax.set_xlabel("Error streak length (log scale)")
        ax.set_ylabel("Number of streaks")
        ax.set_title("Test error-streak length distribution")
        ax.legend()
        save_figure(fig, plot_dir / "error_streak_length_distribution.png")

    test_agreement = agreement[agreement["split"].eq("test")]
    counts = test_agreement["primary_agreement_category"].value_counts().sort_values(ascending=False)
    if not counts.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        counts.plot(kind="bar", ax=ax)
        ax.set_ylabel("Number of test windows")
        ax.set_xlabel("Agreement category")
        ax.set_title("Cross-model graph-error agreement")
        ax.tick_params(axis="x", labelrotation=35)
        save_figure(fig, plot_dir / "test_cross_model_error_agreement_counts.png")


def top_run_lines(table: pd.DataFrame, model_name: str, error_column: str) -> list[str]:
    subset = table[
        table["model_name"].eq(model_name)
        & table["split"].eq("test")
    ].sort_values(
        ["fraction_of_run_in_error", error_column],
        ascending=[False, False],
    )
    lines = []
    for _, row in subset.head(3).iterrows():
        lines.append(
            f"- `{row['run_id']}`: {int(row[error_column])}/"
            f"{int(row['number_of_windows'])} error windows "
            f"({row['fraction_of_run_in_error']:.2%}); "
            f"profile={row['profile']}; active_cores={row['active_cores']}; "
            f"attackers={row['attackers'] or '<none>'}; "
            f"strength={row['strength'] or '<none>'}; "
            f"longest streak={int(row['longest_error_streak'])}."
        )
    return lines


def build_report(
    window_table: pd.DataFrame,
    run_metrics: pd.DataFrame,
    fp_runs: pd.DataFrame,
    fn_runs: pd.DataFrame,
    concentration: pd.DataFrame,
    agreement: pd.DataFrame,
) -> str:
    lines = [
        "# V3 Graph Failure Analysis",
        "",
        "This report analyses graph-level detection at the fixed threshold of 0.50.",
        "",
        "## Overall test errors",
        "",
        "| Model | FP | FN | Total errors | FPR | FNR |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    aggregate: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_ORDER:
        subset = window_table[
            window_table["model_name"].eq(model_name)
            & window_table["split"].eq("test")
        ]
        counts = binary_counts(subset["true_graph"], subset["pred_graph"])
        metrics = metrics_from_counts(counts)
        aggregate[model_name] = {**counts, **metrics}
        lines.append(
            f"| {MODEL_LABELS[model_name]} | {counts['FP']} | {counts['FN']} | "
            f"{counts['FP'] + counts['FN']} | {metrics['FPR']:.4f} | {metrics['FNR']:.4f} |"
        )

    total_fp = sum(aggregate[name]["FP"] for name in MODEL_ORDER)
    total_fn = sum(aggregate[name]["FN"] for name in MODEL_ORDER)
    largest_fp = max(MODEL_ORDER, key=lambda name: aggregate[name]["FP"])
    largest_fn = max(MODEL_ORDER, key=lambda name: aggregate[name]["FN"])

    lines += [
        "",
        "## Direct findings",
        "",
        f"- Across all four models: **{total_fp} false-positive windows** and "
        f"**{total_fn} false-negative windows**.",
        f"- Dominant error type: **{'false positives' if total_fp > total_fn else 'false negatives'}**.",
        f"- Largest FP problem: **{MODEL_LABELS[largest_fp]}** "
        f"({aggregate[largest_fp]['FP']} test FPs).",
        f"- Largest FN problem: **{MODEL_LABELS[largest_fn]}** "
        f"({aggregate[largest_fn]['FN']} test FNs).",
        "",
        "## Worst normal test runs",
        "",
    ]

    for model_name in MODEL_ORDER:
        lines.append(f"### {MODEL_LABELS[model_name]}")
        lines.extend(top_run_lines(fp_runs, model_name, "number_of_FP_windows"))
        subset = fp_runs[fp_runs["model_name"].eq(model_name) & fp_runs["split"].eq("test")]
        lines.append(
            f"- Normal test runs with zero FPs: {int(subset['no_false_positives'].sum())}; "
            f"above 50% FP: {int(subset['more_than_50_percent_FP'].sum())}; "
            f"above 80% FP: {int(subset['more_than_80_percent_FP'].sum())}."
        )
        lines.append("")

    lines += ["## Worst attack test runs", ""]
    for model_name in MODEL_ORDER:
        lines.append(f"### {MODEL_LABELS[model_name]}")
        lines.extend(top_run_lines(fn_runs, model_name, "number_of_FN_windows"))
        subset = fn_runs[fn_runs["model_name"].eq(model_name) & fn_runs["split"].eq("test")]
        lines.append(
            f"- Attack test runs with zero FNs: {int(subset['no_false_negatives'].sum())}; "
            f"above 50% FN: {int(subset['more_than_50_percent_FN'].sum())}."
        )
        lines.append("")

    lines += ["## Error concentration", ""]
    for model_name in MODEL_ORDER:
        parts = []
        for error_type in ["FP", "FN"]:
            row = concentration[
                concentration["model_name"].eq(model_name)
                & concentration["split"].eq("test")
                & concentration["error_type"].eq(error_type)
                & concentration["selection"].eq("worst_2_runs")
            ]
            if not row.empty:
                parts.append(
                    f"{error_type}: worst two runs contribute "
                    f"{float(row.iloc[0]['window_weighted_error_share']):.1%}"
                )
        lines.append(f"- **{MODEL_LABELS[model_name]}:** " + "; ".join(parts) + ".")

    test_errors = window_table[
        window_table["split"].eq("test") & window_table["is_error"]
    ]
    confidence_counts = test_errors["confidence_category"].value_counts()
    total_test_errors = max(1, len(test_errors))
    borderline_share = int(confidence_counts.get("borderline", 0)) / total_test_errors
    high_share = int(confidence_counts.get("high_confidence_error", 0)) / total_test_errors
    test_run_metrics = run_metrics[run_metrics["split"].eq("test")]
    longest = test_run_metrics.sort_values("longest_error_streak", ascending=False).iloc[0]

    lines += [
        "",
        "## Confidence and persistence",
        "",
        f"- Borderline errors: {borderline_share:.1%} of test graph-error windows.",
        f"- High-confidence errors: {high_share:.1%} of test graph-error windows.",
        f"- Longest test error streak: {int(longest['longest_error_streak'])} consecutive windows "
        f"in `{longest['run_id']}` for {MODEL_LABELS[longest['model_name']]}.",
    ]
    if high_share >= 0.50:
        lines.append(
            "- The large number of high-confidence errors suggests that threshold adjustment alone "
            "is unlikely to solve the problem."
        )
    elif borderline_share >= 0.60:
        lines.append(
            "- The probability behaviour is consistent with a possible threshold-transfer problem, "
            "but a formal threshold sweep is still required."
        )
    else:
        lines.append(
            "- Error confidence is mixed. Threshold mismatch may contribute, but it is not proven."
        )

    test_agreement = agreement[agreement["split"].eq("test")]
    lines += [
        "",
        "## Cross-model agreement",
        "",
        f"- All four models correct: {int(test_agreement['all_four_correct'].sum())} test windows.",
        f"- All four models wrong: {int(test_agreement['all_four_wrong'].sum())} test windows.",
        f"- Conv1D correct while all TCN models are wrong: "
        f"{int(test_agreement['conv1d_correct_all_tcns_wrong'].sum())} windows.",
        f"- All TCN models correct while Conv1D is wrong: "
        f"{int(test_agreement['all_tcns_correct_conv1d_wrong'].sum())} windows.",
        f"- Attention uniquely fails: {int(test_agreement['attention_only_failure'].sum())} windows.",
        f"- MaxPool uniquely succeeds: {int(test_agreement['maxpool_only_success'].sum())} windows.",
    ]

    gaps = []
    for model_name in MODEL_ORDER:
        split_metrics = {}
        for split_name in ["val", "test"]:
            subset = window_table[
                window_table["model_name"].eq(model_name)
                & window_table["split"].eq(split_name)
            ]
            split_metrics[split_name] = metrics_from_counts(
                binary_counts(subset["true_graph"], subset["pred_graph"])
            )
        gaps.append(
            {
                "model_name": model_name,
                "FPR_gap": split_metrics["test"]["FPR"] - split_metrics["val"]["FPR"],
                "FNR_gap": split_metrics["test"]["FNR"] - split_metrics["val"]["FNR"],
            }
        )

    gap_table = pd.DataFrame(gaps)
    mean_fpr_gap = float(gap_table["FPR_gap"].mean())
    mean_fnr_gap = float(gap_table["FNR_gap"].mean())

    if mean_fpr_gap > 1.25 * mean_fnr_gap:
        diagnosis = (
            "primarily a benign-distribution false-positive shift, with a secondary "
            "attack-generalization failure"
        )
    elif mean_fnr_gap > 1.25 * mean_fpr_gap:
        diagnosis = (
            "primarily an attack-generalization false-negative shift, with a secondary "
            "benign false-positive shift"
        )
    else:
        diagnosis = "a combination of benign false-positive shift and attack false-negative shift"

    lines += [
        "",
        "## Current graph-level diagnosis",
        "",
        f"The mean validation-to-test FPR increase is {mean_fpr_gap:.4f}. "
        f"The mean FNR increase is {mean_fnr_gap:.4f}. "
        f"The current collapse is best described as **{diagnosis}**.",
        "",
        "This does not prove whether calibration, feature ambiguity, temporal behaviour, "
        "or representation shift is causal. The next controlled step is a threshold-transfer analysis.",
        "",
    ]
    return "\n".join(lines)


def validate_analysis(
    window_table: pd.DataFrame,
    run_metrics: pd.DataFrame,
    streaks: pd.DataFrame,
    stage3_metrics: pd.DataFrame,
) -> tuple[bool, str]:
    messages = []
    aggregate_metrics_matched = True

    for model_name in MODEL_ORDER:
        for split_name in ["val", "test"]:
            subset = window_table[
                window_table["model_name"].eq(model_name)
                & window_table["split"].eq(split_name)
            ]
            counts = binary_counts(subset["true_graph"], subset["pred_graph"])
            metrics = metrics_from_counts(counts)
            expected = stage3_metrics[
                stage3_metrics["model"].eq(model_name)
                & stage3_metrics["split"].eq(split_name)
            ]
            if len(expected) != 1:
                aggregate_metrics_matched = False
                messages.append(f"Missing/duplicate Stage 3 row: {model_name} {split_name}")
                continue
            row = expected.iloc[0]
            for calc_name, saved_name in {"TN": "tn", "FP": "fp", "FN": "fn", "TP": "tp"}.items():
                if int(counts[calc_name]) != int(row[saved_name]):
                    aggregate_metrics_matched = False
                    messages.append(
                        f"{model_name} {split_name} {calc_name}: "
                        f"{counts[calc_name]} != {int(row[saved_name])}"
                    )
            for calc_name, saved_name in {
                "accuracy": "accuracy", "precision": "precision",
                "recall": "recall", "F1": "f1", "FPR": "fpr",
            }.items():
                if not np.isclose(float(metrics[calc_name]), float(row[saved_name]), atol=1e-12, rtol=0):
                    aggregate_metrics_matched = False
                    messages.append(
                        f"{model_name} {split_name} {calc_name}: "
                        f"{metrics[calc_name]} != {float(row[saved_name])}"
                    )

    all_samples_accounted_for = True
    for (model_name, split_name), group in window_table.groupby(["model_name", "split"]):
        run_total = int(
            run_metrics[
                run_metrics["model_name"].eq(model_name)
                & run_metrics["split"].eq(split_name)
            ]["number_of_windows"].sum()
        )
        if run_total != len(group):
            all_samples_accounted_for = False
            messages.append(
                f"Sample mismatch {model_name} {split_name}: {run_total} != {len(group)}"
            )

    all_errors_accounted_for = (
        int(run_metrics["number_of_error_windows"].sum())
        == int(window_table["is_error"].sum())
    )
    all_streaks_accounted_for = (
        int(streaks["streak_length"].sum()) if not streaks.empty else 0
    ) == int(window_table["is_error"].sum())
    all_run_labels_constant = bool(
        (window_table.groupby(["split", "run_id"])["true_graph"].nunique() == 1).all()
    )
    all_model_sample_keys_unique = not bool(
        window_table.duplicated(subset=["model_name", "sample_id"]).any()
    )
    probability_columns = [
        "mean_graph_probability", "std_graph_probability",
        "minimum_graph_probability", "maximum_graph_probability",
        "p01", "p05", "p25", "p50", "p75", "p95", "p99",
    ]
    all_statistics_finite = bool(
        np.isfinite(run_metrics[probability_columns].to_numpy(dtype=float)).all()
    )

    overall = all(
        [
            aggregate_metrics_matched,
            all_samples_accounted_for,
            all_errors_accounted_for,
            all_streaks_accounted_for,
            all_run_labels_constant,
            all_model_sample_keys_unique,
            all_statistics_finite,
        ]
    )

    lines = [
        f"aggregate metrics matched: {aggregate_metrics_matched}",
        f"all samples accounted for: {all_samples_accounted_for}",
        f"all errors accounted for: {all_errors_accounted_for}",
        f"all streaks accounted for: {all_streaks_accounted_for}",
        f"all run labels constant: {all_run_labels_constant}",
        f"all model/sample keys unique: {all_model_sample_keys_unique}",
        f"all statistics finite: {all_statistics_finite}",
        f"overall success: {overall}",
    ]
    if messages:
        lines += ["", "DETAILS:", *messages]
    return overall, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse V3 graph false positives, false negatives, concentration and agreement."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("reports/v3_failure_analysis"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    prediction_dir = root / "predictions"
    table_dir = root / "tables"
    plot_dir = root / "plots" / "graph_failures"
    log_dir = root / "logs"

    metadata_path = prediction_dir / "prediction_metadata.csv"
    stage3_metrics_path = table_dir / "exported_metric_reproduction.csv"
    report_path = root / "GRAPH_FAILURE_ANALYSIS.md"
    validation_path = log_dir / "28_graph_failure_validation.txt"

    outputs = {
        "window_predictions": table_dir / "graph_window_predictions.csv",
        "run_metrics": table_dir / "graph_metrics_by_run.csv",
        "error_streaks": table_dir / "graph_error_streaks.csv",
        "false_positive_runs": table_dir / "false_positive_runs.csv",
        "false_negative_runs": table_dir / "false_negative_runs.csv",
        "group_metrics": table_dir / "graph_metrics_by_group.csv",
        "error_concentration": table_dir / "graph_error_concentration.csv",
        "val_test_gap": table_dir / "graph_val_test_gap.csv",
        "model_agreement": table_dir / "graph_model_agreement.csv",
        "common_error_runs": table_dir / "common_graph_error_runs.csv",
        "report": report_path,
        "validation": validation_path,
    }

    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Stage 4 outputs already exist. Use --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not stage3_metrics_path.is_file():
        raise FileNotFoundError(stage3_metrics_path)

    print(f"root: {root}", flush=True)
    print(f"metadata: {metadata_path}", flush=True)
    print(f"stage3 metrics: {stage3_metrics_path}", flush=True)

    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    stage3_metrics = pd.read_csv(stage3_metrics_path, keep_default_na=False)
    loaded = load_predictions(prediction_dir)
    verify_alignment(metadata, loaded)
    print("input alignment verified", flush=True)

    window_table = build_window_table(metadata, loaded)
    print(f"model/sample rows: {len(window_table)}", flush=True)

    streaks = build_error_streaks(window_table)
    print(f"error streaks: {len(streaks)}", flush=True)

    run_metrics = build_run_metrics(window_table, streaks)
    fp_runs, fn_runs = build_fp_fn_tables(run_metrics)
    group_metrics = build_group_metrics(window_table)
    concentration = build_error_concentration(run_metrics)
    val_test_gap = build_val_test_gap(group_metrics)
    agreement, common_error_runs = build_model_agreement(window_table)

    success, validation_text = validate_analysis(
        window_table, run_metrics, streaks, stage3_metrics
    )
    atomic_text(validation_text, validation_path)
    print(validation_text, end="", flush=True)
    if not success:
        raise SystemExit(
            f"Stage 4 validation failed. Inspect {validation_path}. "
            "No analysis tables were written."
        )

    atomic_csv(window_table, outputs["window_predictions"])
    atomic_csv(run_metrics, outputs["run_metrics"])
    atomic_csv(streaks, outputs["error_streaks"])
    atomic_csv(fp_runs, outputs["false_positive_runs"])
    atomic_csv(fn_runs, outputs["false_negative_runs"])
    atomic_csv(group_metrics, outputs["group_metrics"])
    atomic_csv(concentration, outputs["error_concentration"])
    atomic_csv(val_test_gap, outputs["val_test_gap"])
    atomic_csv(agreement, outputs["model_agreement"])
    atomic_csv(common_error_runs, outputs["common_error_runs"])

    make_plots(window_table, run_metrics, streaks, agreement, plot_dir)
    report = build_report(
        window_table, run_metrics, fp_runs, fn_runs, concentration, agreement
    )
    atomic_text(report, report_path)

    print(f"plots written to: {plot_dir}", flush=True)
    print("Stage 4 output files:", flush=True)
    for name, path in outputs.items():
        print(f"  {name}: {path}", flush=True)
    print("overall success: True", flush=True)


if __name__ == "__main__":
    main()
