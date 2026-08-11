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


# ---------------------------------------------------------------------------
# File-writing helpers
# ---------------------------------------------------------------------------

def atomic_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    try:
        dataframe.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(
    text: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_figure(
    figure: plt.Figure,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        path.stem + ".tmp" + path.suffix
    )

    try:
        figure.savefig(
            temporary,
            dpi=180,
            bbox_inches="tight",
        )
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# General metric helpers
# ---------------------------------------------------------------------------

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return float("nan")

    return float(numerator / denominator)


def binary_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, int]:
    y_true = np.asarray(
        y_true,
        dtype=np.int8,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.int8,
    ).reshape(-1)

    return {
        "TN": int(
            ((y_true == 0) & (y_pred == 0)).sum()
        ),
        "FP": int(
            ((y_true == 0) & (y_pred == 1)).sum()
        ),
        "FN": int(
            ((y_true == 1) & (y_pred == 0)).sum()
        ),
        "TP": int(
            ((y_true == 1) & (y_pred == 1)).sum()
        ),
    }


def metrics_from_counts(
    counts: dict[str, int],
) -> dict[str, float]:
    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    tp = counts["TP"]

    accuracy = safe_divide(
        tn + tp,
        tn + fp + fn + tp,
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    if (
        np.isfinite(precision)
        and np.isfinite(recall)
        and precision + recall > 0
    ):
        f1 = (
            2.0 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": float(f1),
        "FPR": safe_divide(fp, fp + tn),
        "FNR": safe_divide(fn, fn + tp),
    }


def parse_active_core_count(
    value: Any,
) -> int:
    text = str(value).strip()

    if not text or text.lower() == "idle":
        return 0

    return len(
        [
            token
            for token in text.split("-")
            if token.strip()
        ]
    )


def classify_confidence(
    true_graph: int,
    probability: float,
    is_error: bool,
) -> str:
    if not is_error:
        return "correct"

    if 0.40 <= probability <= 0.60:
        return "borderline"

    if true_graph == 0:
        if probability > 0.80:
            return "high_confidence_error"

        return "moderate_confidence_error"

    if probability < 0.20:
        return "high_confidence_error"

    return "moderate_confidence_error"


def percentile(
    series: pd.Series,
    value: float,
) -> float:
    return float(
        np.percentile(
            series.to_numpy(dtype=float),
            value,
        )
    )


# ---------------------------------------------------------------------------
# Input loading and alignment
# ---------------------------------------------------------------------------

def load_prediction_files(
    prediction_dir: Path,
) -> dict[str, dict[str, np.ndarray]]:
    loaded: dict[str, dict[str, np.ndarray]] = {}

    for model_name, filename in MODEL_FILES.items():
        path = prediction_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Prediction file is missing: {path}"
            )

        with np.load(
            path,
            allow_pickle=False,
        ) as prediction_file:
            loaded[model_name] = {
                key: prediction_file[key]
                for key in prediction_file.files
            }

        print(
            f"loaded {model_name}: {path}",
            flush=True,
        )

    return loaded


def verify_alignment(
    metadata: pd.DataFrame,
    loaded: dict[str, dict[str, np.ndarray]],
) -> None:
    required_metadata = {
        "sample_order_within_export",
        "sample_index",
        "sample_id",
        "run_id",
        "split",
        "end_epoch",
        "true_graph",
        "true_attacker_count",
        "profile",
        "active_cores",
        "attackers",
        "strength",
        "seed",
    }

    missing_metadata = (
        required_metadata - set(metadata.columns)
    )

    if missing_metadata:
        raise RuntimeError(
            "Prediction metadata is missing columns: "
            f"{sorted(missing_metadata)}"
        )

    reference: dict[str, np.ndarray] | None = None

    for model_name, arrays in loaded.items():
        required_arrays = {
            "sample_index",
            "sample_id",
            "split",
            "true_graph",
            "graph_logit",
            "graph_probability",
        }

        missing_arrays = (
            required_arrays - set(arrays)
        )

        if missing_arrays:
            raise RuntimeError(
                f"{model_name} is missing prediction arrays: "
                f"{sorted(missing_arrays)}"
            )

        if len(arrays["sample_index"]) != len(metadata):
            raise RuntimeError(
                f"{model_name} contains "
                f"{len(arrays['sample_index'])} samples, "
                f"but metadata contains {len(metadata)}."
            )

        expected_values = {
            "sample_index": metadata[
                "sample_index"
            ].to_numpy(dtype=np.int64),
            "sample_id": metadata[
                "sample_id"
            ].astype(str).to_numpy(),
            "split": metadata[
                "split"
            ].astype(str).to_numpy(),
            "true_graph": metadata[
                "true_graph"
            ].to_numpy(dtype=np.int64),
        }

        for key, expected in expected_values.items():
            actual = np.asarray(
                arrays[key]
            ).astype(expected.dtype)

            if not np.array_equal(
                actual,
                expected,
            ):
                raise RuntimeError(
                    f"{model_name} does not align with "
                    f"prediction_metadata.csv for {key}."
                )

        if reference is None:
            reference = arrays
            continue

        for key in [
            "sample_index",
            "sample_id",
            "split",
            "true_graph",
        ]:
            if key in {"sample_id", "split"}:
                left = np.asarray(
                    reference[key]
                ).astype(str)

                right = np.asarray(
                    arrays[key]
                ).astype(str)
            else:
                left = np.asarray(
                    reference[key],
                    dtype=np.int64,
                )

                right = np.asarray(
                    arrays[key],
                    dtype=np.int64,
                )

            if not np.array_equal(left, right):
                raise RuntimeError(
                    f"Cross-model alignment failure for "
                    f"{model_name}: {key}"
                )


# ---------------------------------------------------------------------------
# Build the per-window graph table
# ---------------------------------------------------------------------------

def build_window_table(
    metadata: pd.DataFrame,
    loaded: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    base = metadata.rename(
        columns={
            "sample_order_within_export": "sample_order"
        }
    ).copy()

    base["sample_index"] = base[
        "sample_index"
    ].astype(int)

    base["end_epoch"] = base[
        "end_epoch"
    ].astype(int)

    base["true_graph"] = base[
        "true_graph"
    ].astype(int)

    base["attacker_count"] = base[
        "true_attacker_count"
    ].astype(int)

    base["active_core_count"] = base[
        "active_cores"
    ].map(parse_active_core_count).astype(int)

    model_frames: list[pd.DataFrame] = []

    for model_name in MODEL_ORDER:
        arrays = loaded[model_name]
        frame = base.copy()

        frame["model_name"] = model_name

        frame["graph_logit"] = np.asarray(
            arrays["graph_logit"],
            dtype=np.float64,
        )

        frame["graph_probability"] = np.asarray(
            arrays["graph_probability"],
            dtype=np.float64,
        )

        if not np.isfinite(
            frame[
                [
                    "graph_logit",
                    "graph_probability",
                ]
            ].to_numpy(dtype=float)
        ).all():
            raise RuntimeError(
                f"Non-finite graph output detected for "
                f"{model_name}."
            )

        frame["pred_graph"] = (
            frame["graph_probability"]
            >= THRESHOLD
        ).astype(int)

        true_graph = frame[
            "true_graph"
        ].to_numpy(dtype=int)

        pred_graph = frame[
            "pred_graph"
        ].to_numpy(dtype=int)

        frame["error_type"] = np.select(
            [
                (true_graph == 0)
                & (pred_graph == 0),

                (true_graph == 0)
                & (pred_graph == 1),

                (true_graph == 1)
                & (pred_graph == 0),

                (true_graph == 1)
                & (pred_graph == 1),
            ],
            [
                "TN",
                "FP",
                "FN",
                "TP",
            ],
            default="UNKNOWN",
        )

        frame["is_correct"] = (
            true_graph == pred_graph
        )

        frame["is_error"] = (
            ~frame["is_correct"]
        )

        frame["is_false_positive"] = (
            frame["error_type"].eq("FP")
        )

        frame["is_false_negative"] = (
            frame["error_type"].eq("FN")
        )

        frame["probability_margin"] = (
            frame["graph_probability"]
            - THRESHOLD
        )

        frame[
            "absolute_threshold_distance"
        ] = frame[
            "probability_margin"
        ].abs()

        frame["confidence_category"] = [
            classify_confidence(
                int(true_label),
                float(probability),
                bool(is_error),
            )
            for (
                true_label,
                probability,
                is_error,
            ) in zip(
                frame["true_graph"],
                frame["graph_probability"],
                frame["is_error"],
            )
        ]

        frame["high_confidence_error"] = (
            frame["confidence_category"]
            .eq("high_confidence_error")
        )

        model_frames.append(frame)

    result = pd.concat(
        model_frames,
        ignore_index=True,
    )

    if result.duplicated(
        subset=[
            "model_name",
            "sample_id",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate model/sample rows were created."
        )

    return result[
        [
            "sample_order",
            "sample_index",
            "sample_id",
            "run_id",
            "split",
            "end_epoch",
            "model_name",
            "true_graph",
            "graph_logit",
            "graph_probability",
            "pred_graph",
            "error_type",
            "is_correct",
            "is_error",
            "is_false_positive",
            "is_false_negative",
            "probability_margin",
            "absolute_threshold_distance",
            "confidence_category",
            "high_confidence_error",
            "profile",
            "active_cores",
            "active_core_count",
            "attackers",
            "attacker_count",
            "strength",
            "seed",
        ]
    ]


# ---------------------------------------------------------------------------
# Consecutive error streaks
# ---------------------------------------------------------------------------

def build_error_streaks(
    window_table: pd.DataFrame,
) -> pd.DataFrame:
    streak_rows: list[dict[str, Any]] = []

    confidence_priority = {
        "high_confidence_error": 3,
        "moderate_confidence_error": 2,
        "borderline": 1,
    }

    grouped = window_table.groupby(
        [
            "model_name",
            "split",
            "run_id",
        ],
        sort=True,
    )

    for (
        model_name,
        split_name,
        run_id,
    ), run_data in grouped:
        ordered = run_data.sort_values(
            "end_epoch"
        )

        errors = ordered[
            ordered["is_error"]
        ]

        if errors.empty:
            continue

        current_rows: list[pd.Series] = []
        previous_epoch: int | None = None
        previous_error_type: str | None = None
        streak_id = 0

        def flush_streak() -> None:
            nonlocal current_rows
            nonlocal streak_id

            if not current_rows:
                return

            streak_id += 1
            streak = pd.DataFrame(current_rows)

            confidence_counts = streak[
                "confidence_category"
            ].value_counts()

            dominant_confidence = sorted(
                confidence_counts.index,
                key=lambda category: (
                    confidence_counts[category],
                    confidence_priority.get(
                        category,
                        0,
                    ),
                ),
                reverse=True,
            )[0]

            streak_rows.append(
                {
                    "model_name": model_name,
                    "split": split_name,
                    "run_id": run_id,
                    "true_graph": int(
                        streak[
                            "true_graph"
                        ].iloc[0]
                    ),
                    "error_type": str(
                        streak[
                            "error_type"
                        ].iloc[0]
                    ),
                    "streak_id": streak_id,
                    "start_end_epoch": int(
                        streak[
                            "end_epoch"
                        ].iloc[0]
                    ),
                    "finish_end_epoch": int(
                        streak[
                            "end_epoch"
                        ].iloc[-1]
                    ),
                    "streak_length": int(
                        len(streak)
                    ),
                    "mean_probability": float(
                        streak[
                            "graph_probability"
                        ].mean()
                    ),
                    "minimum_probability": float(
                        streak[
                            "graph_probability"
                        ].min()
                    ),
                    "maximum_probability": float(
                        streak[
                            "graph_probability"
                        ].max()
                    ),
                    "confidence_category": (
                        dominant_confidence
                    ),
                }
            )

            current_rows = []

        for _, row in errors.iterrows():
            current_epoch = int(
                row["end_epoch"]
            )

            current_error_type = str(
                row["error_type"]
            )

            consecutive = (
                previous_epoch is not None
                and current_epoch
                == previous_epoch + 1
                and current_error_type
                == previous_error_type
            )

            if current_rows and not consecutive:
                flush_streak()

            current_rows.append(row)
            previous_epoch = current_epoch
            previous_error_type = (
                current_error_type
            )

        flush_streak()

    columns = [
        "model_name",
        "split",
        "run_id",
        "true_graph",
        "error_type",
        "streak_id",
        "start_end_epoch",
        "finish_end_epoch",
        "streak_length",
        "mean_probability",
        "minimum_probability",
        "maximum_probability",
        "confidence_category",
    ]

    return pd.DataFrame(
        streak_rows,
        columns=columns,
    )


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------

def build_run_metrics(
    window_table: pd.DataFrame,
    streaks: pd.DataFrame,
) -> pd.DataFrame:
    streak_summary: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    if not streaks.empty:
        grouped_streaks = streaks.groupby(
            [
                "model_name",
                "split",
                "run_id",
            ]
        )

        for key, group in grouped_streaks:
            streak_summary[key] = {
                "number_of_error_streaks": (
                    int(len(group))
                ),
                "longest_error_streak": int(
                    group[
                        "streak_length"
                    ].max()
                ),
                "mean_error_streak_length": (
                    float(
                        group[
                            "streak_length"
                        ].mean()
                    )
                ),
                "median_error_streak_length": (
                    float(
                        group[
                            "streak_length"
                        ].median()
                    )
                ),
            }

    rows: list[dict[str, Any]] = []

    grouped_runs = window_table.groupby(
        [
            "model_name",
            "split",
            "run_id",
        ],
        sort=True,
    )

    for key, run_data in grouped_runs:
        (
            model_name,
            split_name,
            run_id,
        ) = key

        ordered = run_data.sort_values(
            "end_epoch"
        )

        true_labels = ordered[
            "true_graph"
        ].to_numpy(dtype=int)

        predicted_labels = ordered[
            "pred_graph"
        ].to_numpy(dtype=int)

        counts = binary_counts(
            true_labels,
            predicted_labels,
        )

        metrics = metrics_from_counts(
            counts
        )

        probabilities = ordered[
            "graph_probability"
        ]

        errors = ordered[
            ordered["is_error"]
        ]

        streak_information = (
            streak_summary.get(
                key,
                {
                    "number_of_error_streaks": 0,
                    "longest_error_streak": 0,
                    "mean_error_streak_length": 0.0,
                    "median_error_streak_length": 0.0,
                },
            )
        )

        rows.append(
            {
                "model_name": model_name,
                "split": split_name,
                "run_id": run_id,
                "profile": str(
                    ordered[
                        "profile"
                    ].iloc[0]
                ),
                "active_cores": str(
                    ordered[
                        "active_cores"
                    ].iloc[0]
                ),
                "active_core_count": int(
                    ordered[
                        "active_core_count"
                    ].iloc[0]
                ),
                "attackers": str(
                    ordered[
                        "attackers"
                    ].iloc[0]
                ),
                "attacker_count": int(
                    ordered[
                        "attacker_count"
                    ].iloc[0]
                ),
                "strength": str(
                    ordered[
                        "strength"
                    ].iloc[0]
                ),
                "seed": str(
                    ordered[
                        "seed"
                    ].iloc[0]
                ),
                "true_graph": int(
                    ordered[
                        "true_graph"
                    ].iloc[0]
                ),
                "number_of_windows": int(
                    len(ordered)
                ),
                **counts,
                "accuracy": metrics["accuracy"],
                "error_rate": float(
                    ordered[
                        "is_error"
                    ].mean()
                ),
                "FPR": metrics["FPR"],
                "FNR": metrics["FNR"],
                "mean_graph_probability": float(
                    probabilities.mean()
                ),
                "std_graph_probability": float(
                    probabilities.std(ddof=0)
                ),
                "minimum_graph_probability": float(
                    probabilities.min()
                ),
                "maximum_graph_probability": float(
                    probabilities.max()
                ),
                "p01": percentile(
                    probabilities,
                    1,
                ),
                "p05": percentile(
                    probabilities,
                    5,
                ),
                "p25": percentile(
                    probabilities,
                    25,
                ),
                "p50": percentile(
                    probabilities,
                    50,
                ),
                "p75": percentile(
                    probabilities,
                    75,
                ),
                "p95": percentile(
                    probabilities,
                    95,
                ),
                "p99": percentile(
                    probabilities,
                    99,
                ),
                "number_of_error_windows": int(
                    len(errors)
                ),
                "number_of_borderline_errors": int(
                    errors[
                        "confidence_category"
                    ].eq("borderline").sum()
                ),
                "number_of_moderate_confidence_errors": int(
                    errors[
                        "confidence_category"
                    ].eq(
                        "moderate_confidence_error"
                    ).sum()
                ),
                "number_of_high_confidence_errors": int(
                    errors[
                        "confidence_category"
                    ].eq(
                        "high_confidence_error"
                    ).sum()
                ),
                "first_error_epoch": (
                    int(
                        errors[
                            "end_epoch"
                        ].min()
                    )
                    if not errors.empty
                    else np.nan
                ),
                "last_error_epoch": (
                    int(
                        errors[
                            "end_epoch"
                        ].max()
                    )
                    if not errors.empty
                    else np.nan
                ),
                "fraction_of_run_in_error": (
                    float(
                        ordered[
                            "is_error"
                        ].mean()
                    )
                ),
                **streak_information,
            }
        )

    return pd.DataFrame(rows)


def build_false_positive_negative_tables(
    run_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    false_positive_runs = run_metrics[
        run_metrics["true_graph"].eq(0)
    ].copy()

    false_positive_runs[
        "number_of_FP_windows"
    ] = false_positive_runs["FP"]

    false_positive_runs[
        "no_false_positives"
    ] = false_positive_runs["FP"].eq(0)

    false_positive_runs[
        "more_than_50_percent_FP"
    ] = (
        false_positive_runs[
            "fraction_of_run_in_error"
        ]
        > 0.50
    )

    false_positive_runs[
        "more_than_80_percent_FP"
    ] = (
        false_positive_runs[
            "fraction_of_run_in_error"
        ]
        > 0.80
    )

    false_positive_runs = (
        false_positive_runs.sort_values(
            [
                "model_name",
                "split",
                "fraction_of_run_in_error",
                "number_of_FP_windows",
            ],
            ascending=[
                True,
                True,
                False,
                False,
            ],
        )
    )

    false_negative_runs = run_metrics[
        run_metrics["true_graph"].eq(1)
    ].copy()

    false_negative_runs[
        "number_of_FN_windows"
    ] = false_negative_runs["FN"]

    false_negative_runs[
        "no_false_negatives"
    ] = false_negative_runs["FN"].eq(0)

    false_negative_runs[
        "more_than_50_percent_FN"
    ] = (
        false_negative_runs[
            "fraction_of_run_in_error"
        ]
        > 0.50
    )

    false_negative_runs = (
        false_negative_runs.sort_values(
            [
                "model_name",
                "split",
                "fraction_of_run_in_error",
                "number_of_FN_windows",
            ],
            ascending=[
                True,
                True,
                False,
                False,
            ],
        )
    )

    return (
        false_positive_runs,
        false_negative_runs,
    )


# ---------------------------------------------------------------------------
# Group-level metrics
# ---------------------------------------------------------------------------

def calculate_group_metrics(
    group: pd.DataFrame,
) -> dict[str, Any]:
    counts = binary_counts(
        group[
            "true_graph"
        ].to_numpy(dtype=int),
        group[
            "pred_graph"
        ].to_numpy(dtype=int),
    )

    metrics = metrics_from_counts(
        counts
    )

    return {
        "number_of_runs": int(
            group["run_id"].nunique()
        ),
        "number_of_windows": int(
            len(group)
        ),
        **counts,
        **metrics,
        "mean_graph_probability": float(
            group[
                "graph_probability"
            ].mean()
        ),
        "std_graph_probability": float(
            group[
                "graph_probability"
            ].std(ddof=0)
        ),
    }


def build_group_metrics(
    window_table: pd.DataFrame,
) -> pd.DataFrame:
    working = window_table.copy()

    working[
        "normal_vs_attack"
    ] = np.where(
        working["true_graph"].eq(0),
        "normal",
        "attack",
    )

    grouping_variables = [
        "profile",
        "active_core_count",
        "attacker_count",
        "strength",
        "attackers",
        "active_cores",
        "seed",
        "normal_vs_attack",
    ]

    rows: list[dict[str, Any]] = []

    for grouping_variable in grouping_variables:
        group_values = (
            working[
                grouping_variable
            ]
            .astype(str)
            .replace("", "<none>")
        )

        temporary = working.assign(
            _group_value=group_values
        )

        grouped = temporary.groupby(
            [
                "model_name",
                "split",
                "_group_value",
            ],
            sort=True,
        )

        for (
            model_name,
            split_name,
            group_value,
        ), group in grouped:
            rows.append(
                {
                    "model_name": model_name,
                    "split": split_name,
                    "grouping_variable": (
                        grouping_variable
                    ),
                    "group_value": group_value,
                    **calculate_group_metrics(
                        group
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Error concentration
# ---------------------------------------------------------------------------

def build_error_concentration(
    run_metrics: pd.DataFrame,
) -> pd.DataFrame:
    selections = [
        (
            "worst_1_run",
            lambda run_count: 1,
        ),
        (
            "worst_2_runs",
            lambda run_count: 2,
        ),
        (
            "worst_3_runs",
            lambda run_count: 3,
        ),
        (
            "worst_5_percent",
            lambda run_count: math.ceil(
                run_count * 0.05
            ),
        ),
        (
            "worst_10_percent",
            lambda run_count: math.ceil(
                run_count * 0.10
            ),
        ),
        (
            "worst_20_percent",
            lambda run_count: math.ceil(
                run_count * 0.20
            ),
        ),
        (
            "all_eligible_runs",
            lambda run_count: run_count,
        ),
    ]

    rows: list[dict[str, Any]] = []

    for model_name in MODEL_ORDER:
        for split_name in [
            "val",
            "test",
        ]:
            for (
                error_type,
                true_graph,
                error_column,
            ) in [
                ("FP", 0, "FP"),
                ("FN", 1, "FN"),
            ]:
                eligible_runs = run_metrics[
                    run_metrics[
                        "model_name"
                    ].eq(model_name)
                    & run_metrics[
                        "split"
                    ].eq(split_name)
                    & run_metrics[
                        "true_graph"
                    ].eq(true_graph)
                ].copy()

                eligible_runs = (
                    eligible_runs.sort_values(
                        [
                            "fraction_of_run_in_error",
                            error_column,
                        ],
                        ascending=[
                            False,
                            False,
                        ],
                    )
                )

                eligible_run_count = len(
                    eligible_runs
                )

                if eligible_run_count == 0:
                    continue

                total_error_windows = int(
                    eligible_runs[
                        error_column
                    ].sum()
                )

                for (
                    selection_name,
                    selection_function,
                ) in selections:
                    selected_run_count = min(
                        eligible_run_count,
                        max(
                            1,
                            selection_function(
                                eligible_run_count
                            ),
                        ),
                    )

                    selected_runs = (
                        eligible_runs.head(
                            selected_run_count
                        )
                    )

                    selected_error_windows = int(
                        selected_runs[
                            error_column
                        ].sum()
                    )

                    if total_error_windows > 0:
                        error_share = (
                            selected_error_windows
                            / total_error_windows
                        )
                    else:
                        error_share = 0.0

                    rows.append(
                        {
                            "model_name": model_name,
                            "split": split_name,
                            "error_type": (
                                error_type
                            ),
                            "selection": (
                                selection_name
                            ),
                            "eligible_run_count": (
                                eligible_run_count
                            ),
                            "selected_run_count": (
                                selected_run_count
                            ),
                            "selected_error_windows": (
                                selected_error_windows
                            ),
                            "total_error_windows": (
                                total_error_windows
                            ),
                            "window_weighted_error_share": (
                                error_share
                            ),
                            "run_weighted_mean_error_rate": (
                                float(
                                    selected_runs[
                                        "fraction_of_run_in_error"
                                    ].mean()
                                )
                            ),
                        }
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation-to-test gaps
# ---------------------------------------------------------------------------

def build_val_test_gap(
    group_metrics: pd.DataFrame,
) -> pd.DataFrame:
    requested_groups = {
        "profile",
        "attacker_count",
        "strength",
        "active_core_count",
        "normal_vs_attack",
    }

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "F1",
        "FPR",
        "FNR",
        "mean_graph_probability",
    ]

    selected = group_metrics[
        group_metrics[
            "grouping_variable"
        ].isin(requested_groups)
    ]

    rows: list[dict[str, Any]] = []

    grouped = selected.groupby(
        [
            "model_name",
            "grouping_variable",
            "group_value",
        ],
        sort=True,
    )

    for (
        model_name,
        grouping_variable,
        group_value,
    ), group in grouped:
        split_rows = {
            row["split"]: row
            for _, row in group.iterrows()
        }

        if (
            "val" not in split_rows
            or "test" not in split_rows
        ):
            continue

        validation_row = split_rows["val"]
        test_row = split_rows["test"]

        for metric_name in metric_names:
            validation_metric = float(
                validation_row[
                    metric_name
                ]
            )

            test_metric = float(
                test_row[
                    metric_name
                ]
            )

            if not (
                np.isfinite(
                    validation_metric
                )
                and np.isfinite(
                    test_metric
                )
            ):
                continue

            absolute_gap = (
                test_metric
                - validation_metric
            )

            relative_gap = (
                absolute_gap
                / abs(validation_metric)
                if validation_metric != 0
                else float("nan")
            )

            rows.append(
                {
                    "model_name": model_name,
                    "grouping_variable": (
                        grouping_variable
                    ),
                    "group_value": group_value,
                    "metric": metric_name,
                    "validation_metric": (
                        validation_metric
                    ),
                    "test_metric": test_metric,
                    "absolute_gap": (
                        absolute_gap
                    ),
                    "relative_gap": (
                        relative_gap
                    ),
                    "validation_run_count": int(
                        validation_row[
                            "number_of_runs"
                        ]
                    ),
                    "test_run_count": int(
                        test_row[
                            "number_of_runs"
                        ]
                    ),
                    "validation_window_count": int(
                        validation_row[
                            "number_of_windows"
                        ]
                    ),
                    "test_window_count": int(
                        test_row[
                            "number_of_windows"
                        ]
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-model agreement
# ---------------------------------------------------------------------------

def build_model_agreement(
    window_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_columns = [
        "sample_order",
        "sample_index",
        "sample_id",
        "run_id",
        "split",
        "end_epoch",
        "true_graph",
        "profile",
        "active_cores",
        "active_core_count",
        "attackers",
        "attacker_count",
        "strength",
        "seed",
    ]

    agreement = (
        window_table[
            metadata_columns
        ]
        .drop_duplicates(
            subset=["sample_id"]
        )
        .sort_values(
            "sample_order"
        )
        .reset_index(drop=True)
    )

    prediction_pivot = (
        window_table.pivot(
            index="sample_id",
            columns="model_name",
            values="pred_graph",
        )
    )

    correctness_pivot = (
        window_table.pivot(
            index="sample_id",
            columns="model_name",
            values="is_correct",
        )
    )

    probability_pivot = (
        window_table.pivot(
            index="sample_id",
            columns="model_name",
            values="graph_probability",
        )
    )

    for model_name in MODEL_ORDER:
        agreement[
            f"pred_{model_name}"
        ] = (
            agreement["sample_id"]
            .map(
                prediction_pivot[
                    model_name
                ]
            )
            .astype(int)
        )

        agreement[
            f"correct_{model_name}"
        ] = (
            agreement["sample_id"]
            .map(
                correctness_pivot[
                    model_name
                ]
            )
            .astype(bool)
        )

        agreement[
            f"probability_{model_name}"
        ] = (
            agreement["sample_id"]
            .map(
                probability_pivot[
                    model_name
                ]
            )
            .astype(float)
        )

    correctness_columns = [
        f"correct_{model_name}"
        for model_name in MODEL_ORDER
    ]

    agreement[
        "correct_model_count"
    ] = (
        agreement[
            correctness_columns
        ]
        .sum(axis=1)
        .astype(int)
    )

    agreement[
        "wrong_model_count"
    ] = (
        len(MODEL_ORDER)
        - agreement[
            "correct_model_count"
        ]
    )

    agreement["all_four_correct"] = (
        agreement[
            "correct_model_count"
        ].eq(4)
    )

    agreement["all_four_wrong"] = (
        agreement[
            "correct_model_count"
        ].eq(0)
    )

    for model_name in MODEL_ORDER:
        other_correctness_columns = [
            f"correct_{other_model}"
            for other_model in MODEL_ORDER
            if other_model != model_name
        ]

        agreement[
            f"only_{model_name}_correct"
        ] = (
            agreement[
                f"correct_{model_name}"
            ]
            & ~agreement[
                other_correctness_columns
            ].any(axis=1)
        )

    agreement[
        "conv1d_correct_all_tcns_wrong"
    ] = agreement[
        "only_conv1d_gcn_correct"
    ]

    agreement[
        "all_tcns_correct_conv1d_wrong"
    ] = (
        ~agreement[
            "correct_conv1d_gcn"
        ]
        & agreement[
            [
                "correct_tcn_attention_gcn",
                "correct_tcn_meanpool_gcn",
                "correct_tcn_maxpool_gcn",
            ]
        ].all(axis=1)
    )

    agreement[
        "exactly_one_model_wrong"
    ] = agreement[
        "wrong_model_count"
    ].eq(1)

    agreement[
        "exactly_two_models_wrong"
    ] = agreement[
        "wrong_model_count"
    ].eq(2)

    agreement[
        "exactly_three_models_wrong"
    ] = agreement[
        "wrong_model_count"
    ].eq(3)

    agreement[
        "attention_only_failure"
    ] = (
        ~agreement[
            "correct_tcn_attention_gcn"
        ]
        & agreement[
            [
                "correct_conv1d_gcn",
                "correct_tcn_meanpool_gcn",
                "correct_tcn_maxpool_gcn",
            ]
        ].all(axis=1)
    )

    agreement[
        "maxpool_only_success"
    ] = (
        agreement[
            "correct_tcn_maxpool_gcn"
        ]
        & ~agreement[
            [
                "correct_conv1d_gcn",
                "correct_tcn_attention_gcn",
                "correct_tcn_meanpool_gcn",
            ]
        ].any(axis=1)
    )

    def primary_category(
        row: pd.Series,
    ) -> str:
        if row["all_four_correct"]:
            return "all_four_correct"

        if row["all_four_wrong"]:
            return "all_four_wrong"

        for model_name in MODEL_ORDER:
            if row[
                f"only_{model_name}_correct"
            ]:
                return (
                    f"only_{model_name}_correct"
                )

        if row[
            "all_tcns_correct_conv1d_wrong"
        ]:
            return (
                "all_tcn_models_correct_"
                "conv1d_wrong"
            )

        if row[
            "exactly_one_model_wrong"
        ]:
            return "exactly_one_model_wrong"

        if row[
            "exactly_two_models_wrong"
        ]:
            return "exactly_two_models_wrong"

        if row[
            "exactly_three_models_wrong"
        ]:
            return (
                "exactly_three_models_wrong"
            )

        return "other"

    agreement[
        "primary_agreement_category"
    ] = agreement.apply(
        primary_category,
        axis=1,
    )

    run_rows: list[dict[str, Any]] = []

    grouped_runs = agreement.groupby(
        [
            "split",
            "run_id",
        ],
        sort=True,
    )

    for (
        split_name,
        run_id,
    ), run_data in grouped_runs:
        same_false_positive = (
            run_data["all_four_wrong"]
            & run_data["true_graph"].eq(0)
        )

        same_false_negative = (
            run_data["all_four_wrong"]
            & run_data["true_graph"].eq(1)
        )

        tcn_fail_conv_succeeds = (
            run_data[
                "conv1d_correct_all_tcns_wrong"
            ]
        )

        conv_fails_tcns_succeed = (
            run_data[
                "all_tcns_correct_conv1d_wrong"
            ]
        )

        attention_only_failure = (
            run_data[
                "attention_only_failure"
            ]
        )

        maxpool_only_success = (
            run_data[
                "maxpool_only_success"
            ]
        )

        run_rows.append(
            {
                "split": split_name,
                "run_id": run_id,
                "true_graph": int(
                    run_data[
                        "true_graph"
                    ].iloc[0]
                ),
                "profile": str(
                    run_data[
                        "profile"
                    ].iloc[0]
                ),
                "active_cores": str(
                    run_data[
                        "active_cores"
                    ].iloc[0]
                ),
                "active_core_count": int(
                    run_data[
                        "active_core_count"
                    ].iloc[0]
                ),
                "attackers": str(
                    run_data[
                        "attackers"
                    ].iloc[0]
                ),
                "attacker_count": int(
                    run_data[
                        "attacker_count"
                    ].iloc[0]
                ),
                "strength": str(
                    run_data[
                        "strength"
                    ].iloc[0]
                ),
                "number_of_windows": int(
                    len(run_data)
                ),
                "all_models_same_FP_count": int(
                    same_false_positive.sum()
                ),
                "all_models_same_FN_count": int(
                    same_false_negative.sum()
                ),
                "tcn_models_fail_conv1d_succeeds_count": int(
                    tcn_fail_conv_succeeds.sum()
                ),
                "conv1d_fails_all_tcn_models_succeed_count": int(
                    conv_fails_tcns_succeed.sum()
                ),
                "attention_uniquely_fails_count": int(
                    attention_only_failure.sum()
                ),
                "maxpool_uniquely_succeeds_count": int(
                    maxpool_only_success.sum()
                ),
                "all_models_same_error_fraction": (
                    float(
                        (
                            same_false_positive
                            | same_false_negative
                        ).mean()
                    )
                ),
                "tcn_fail_conv1d_succeeds_fraction": (
                    float(
                        tcn_fail_conv_succeeds.mean()
                    )
                ),
                "conv1d_fails_all_tcn_succeed_fraction": (
                    float(
                        conv_fails_tcns_succeed.mean()
                    )
                ),
            }
        )

    return (
        agreement,
        pd.DataFrame(run_rows),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(
    window_table: pd.DataFrame,
    run_metrics: pd.DataFrame,
    streaks: pd.DataFrame,
    agreement: pd.DataFrame,
    plot_directory: Path,
) -> None:
    plot_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_run_metrics = run_metrics[
        run_metrics["split"].eq("test")
    ]

    normal_test_runs = test_run_metrics[
        test_run_metrics[
            "true_graph"
        ].eq(0)
    ]

    if not normal_test_runs.empty:
        pivot = normal_test_runs.pivot(
            index="run_id",
            columns="model_name",
            values="FPR",
        ).reindex(columns=MODEL_ORDER)

        figure, axis = plt.subplots(
            figsize=(
                max(11, len(pivot) * 1.3),
                6,
            )
        )

        pivot.plot(
            kind="bar",
            ax=axis,
        )

        axis.set_ylabel(
            "False-positive rate"
        )

        axis.set_xlabel(
            "Normal test run"
        )

        axis.set_title(
            "Test false-positive rate "
            "by normal run and model"
        )

        axis.tick_params(
            axis="x",
            labelrotation=45,
        )

        axis.legend(
            [
                MODEL_LABELS[
                    model_name
                ]
                for model_name in pivot.columns
            ]
        )

        save_figure(
            figure,
            plot_directory
            / "test_false_positive_rate_by_run.png",
        )

    attack_test_runs = test_run_metrics[
        test_run_metrics[
            "true_graph"
        ].eq(1)
    ]

    if not attack_test_runs.empty:
        pivot = attack_test_runs.pivot(
            index="run_id",
            columns="model_name",
            values="FNR",
        ).reindex(columns=MODEL_ORDER)

        figure, axis = plt.subplots(
            figsize=(
                max(12, len(pivot) * 1.3),
                6,
            )
        )

        pivot.plot(
            kind="bar",
            ax=axis,
        )

        axis.set_ylabel(
            "False-negative rate"
        )

        axis.set_xlabel(
            "Attack test run"
        )

        axis.set_title(
            "Test false-negative rate "
            "by attack run and model"
        )

        axis.tick_params(
            axis="x",
            labelrotation=45,
        )

        axis.legend(
            [
                MODEL_LABELS[
                    model_name
                ]
                for model_name in pivot.columns
            ]
        )

        save_figure(
            figure,
            plot_directory
            / "test_false_negative_rate_by_run.png",
        )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    for model_name in MODEL_ORDER:
        for (
            error_type,
            true_graph,
            marker,
        ) in [
            ("FP", 0, "o"),
            ("FN", 1, "x"),
        ]:
            eligible = test_run_metrics[
                test_run_metrics[
                    "model_name"
                ].eq(model_name)
                & test_run_metrics[
                    "true_graph"
                ].eq(true_graph)
            ].sort_values(
                "fraction_of_run_in_error",
                ascending=False,
            )

            total_errors = float(
                eligible[
                    error_type
                ].sum()
            )

            if (
                eligible.empty
                or total_errors <= 0
            ):
                continue

            cumulative_share = (
                eligible[
                    error_type
                ].cumsum().to_numpy(
                    dtype=float
                )
                / total_errors
            )

            run_rank = np.arange(
                1,
                len(cumulative_share) + 1,
            )

            axis.plot(
                run_rank,
                cumulative_share,
                marker=marker,
                label=(
                    f"{MODEL_LABELS[model_name]} "
                    f"{error_type}"
                ),
            )

    axis.set_xlabel(
        "Number of worst runs included"
    )

    axis.set_ylabel(
        "Cumulative share of error windows"
    )

    axis.set_ylim(
        0,
        1.05,
    )

    axis.set_title(
        "Test error-window concentration"
    )

    axis.legend(fontsize=8)

    save_figure(
        figure,
        plot_directory
        / "test_error_concentration_curve.png",
    )

    aggregate_rows: list[
        dict[str, Any]
    ] = []

    for model_name in MODEL_ORDER:
        for split_name in [
            "val",
            "test",
        ]:
            subset = window_table[
                window_table[
                    "model_name"
                ].eq(model_name)
                & window_table[
                    "split"
                ].eq(split_name)
            ]

            metrics = metrics_from_counts(
                binary_counts(
                    subset["true_graph"],
                    subset["pred_graph"],
                )
            )

            aggregate_rows.append(
                {
                    "model_name": (
                        model_name
                    ),
                    "split": split_name,
                    "FPR": metrics["FPR"],
                    "FNR": metrics["FNR"],
                }
            )

    aggregate = pd.DataFrame(
        aggregate_rows
    )

    for metric_name in [
        "FPR",
        "FNR",
    ]:
        pivot = aggregate.pivot(
            index="model_name",
            columns="split",
            values=metric_name,
        ).reindex(MODEL_ORDER)

        figure, axis = plt.subplots(
            figsize=(9, 5)
        )

        pivot.plot(
            kind="bar",
            ax=axis,
        )

        axis.set_ylabel(metric_name)
        axis.set_xlabel("Model")

        axis.set_title(
            f"Validation versus test "
            f"{metric_name}"
        )

        axis.set_xticklabels(
            [
                MODEL_LABELS[
                    model_name
                ]
                for model_name in pivot.index
            ],
            rotation=25,
            ha="right",
        )

        save_figure(
            figure,
            plot_directory
            / (
                f"val_test_"
                f"{metric_name.lower()}_"
                "by_model.png"
            ),
        )

    test_windows = window_table[
        window_table["split"].eq("test")
    ]

    for model_name in MODEL_ORDER:
        model_data = test_windows[
            test_windows[
                "model_name"
            ].eq(model_name)
        ]

        run_ids = sorted(
            model_data[
                "run_id"
            ].unique()
        )

        values = [
            model_data.loc[
                model_data[
                    "run_id"
                ].eq(run_id),
                "graph_probability",
            ].to_numpy()
            for run_id in run_ids
        ]

        figure, axis = plt.subplots(
            figsize=(
                max(14, len(run_ids) * 0.9),
                6,
            )
        )

        axis.boxplot(
            values,
            tick_labels=run_ids,
            showfliers=False,
        )

        axis.axhline(
            THRESHOLD,
            linestyle="--",
        )

        axis.set_ylabel(
            "Graph probability"
        )

        axis.set_xlabel(
            "Test run"
        )

        axis.set_title(
            "Graph probability by test run: "
            f"{MODEL_LABELS[model_name]}"
        )

        axis.tick_params(
            axis="x",
            labelrotation=55,
        )

        save_figure(
            figure,
            plot_directory
            / (
                "graph_probability_boxplot_"
                f"test_{model_name}.png"
            ),
        )

    if not streaks.empty:
        maximum_streak = max(
            2,
            int(
                streaks[
                    "streak_length"
                ].max()
            ),
        )

        bins = np.unique(
            np.clip(
                np.logspace(
                    0,
                    math.log10(
                        maximum_streak
                    ),
                    30,
                ).astype(int),
                1,
                None,
            )
        )

        figure, axis = plt.subplots(
            figsize=(10, 6)
        )

        for model_name in MODEL_ORDER:
            values = streaks.loc[
                streaks[
                    "model_name"
                ].eq(model_name)
                & streaks[
                    "split"
                ].eq("test"),
                "streak_length",
            ].to_numpy(dtype=int)

            if values.size == 0:
                continue

            axis.hist(
                values,
                bins=bins,
                histtype="step",
                linewidth=1.5,
                label=(
                    MODEL_LABELS[
                        model_name
                    ]
                ),
            )

        axis.set_xscale("log")

        axis.set_xlabel(
            "Error streak length "
            "(log scale)"
        )

        axis.set_ylabel(
            "Number of streaks"
        )

        axis.set_title(
            "Test error-streak "
            "length distribution"
        )

        axis.legend()

        save_figure(
            figure,
            plot_directory
            / "error_streak_length_distribution.png",
        )

    test_agreement = agreement[
        agreement["split"].eq("test")
    ]

    agreement_counts = (
        test_agreement[
            "primary_agreement_category"
        ]
        .value_counts()
        .sort_values(
            ascending=False
        )
    )

    if not agreement_counts.empty:
        figure, axis = plt.subplots(
            figsize=(11, 6)
        )

        agreement_counts.plot(
            kind="bar",
            ax=axis,
        )

        axis.set_ylabel(
            "Number of test windows"
        )

        axis.set_xlabel(
            "Agreement category"
        )

        axis.set_title(
            "Cross-model graph-error "
            "agreement"
        )

        axis.tick_params(
            axis="x",
            labelrotation=35,
        )

        save_figure(
            figure,
            plot_directory
            / "test_cross_model_error_agreement_counts.png",
        )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def top_run_lines(
    table: pd.DataFrame,
    model_name: str,
    error_column: str,
) -> list[str]:
    subset = table[
        table[
            "model_name"
        ].eq(model_name)
        & table[
            "split"
        ].eq("test")
    ].sort_values(
        [
            "fraction_of_run_in_error",
            error_column,
        ],
        ascending=[
            False,
            False,
        ],
    )

    lines: list[str] = []

    for _, row in subset.head(3).iterrows():
        lines.append(
            f"- `{row['run_id']}`: "
            f"{int(row[error_column])}/"
            f"{int(row['number_of_windows'])} "
            f"error windows "
            f"({row['fraction_of_run_in_error']:.2%}); "
            f"profile={row['profile']}; "
            f"active_cores={row['active_cores']}; "
            f"attackers={row['attackers'] or '<none>'}; "
            f"strength={row['strength'] or '<none>'}; "
            f"longest streak="
            f"{int(row['longest_error_streak'])}."
        )

    return lines


def build_report(
    window_table: pd.DataFrame,
    run_metrics: pd.DataFrame,
    false_positive_runs: pd.DataFrame,
    false_negative_runs: pd.DataFrame,
    concentration: pd.DataFrame,
    agreement: pd.DataFrame,
) -> str:
    lines = [
        "# V3 Graph Failure Analysis",
        "",
        "This report analyses graph-level detection "
        "at the fixed reference threshold of 0.50. "
        "No model inference was rerun.",
        "",
        "## Overall test errors",
        "",
        "| Model | FP | FN | Total errors | FPR | FNR |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    aggregate: dict[
        str,
        dict[str, Any],
    ] = {}

    for model_name in MODEL_ORDER:
        subset = window_table[
            window_table[
                "model_name"
            ].eq(model_name)
            & window_table[
                "split"
            ].eq("test")
        ]

        counts = binary_counts(
            subset["true_graph"],
            subset["pred_graph"],
        )

        metrics = metrics_from_counts(
            counts
        )

        aggregate[model_name] = {
            **counts,
            **metrics,
        }

        lines.append(
            f"| {MODEL_LABELS[model_name]} | "
            f"{counts['FP']} | "
            f"{counts['FN']} | "
            f"{counts['FP'] + counts['FN']} | "
            f"{metrics['FPR']:.4f} | "
            f"{metrics['FNR']:.4f} |"
        )

    total_fp = sum(
        aggregate[
            model_name
        ]["FP"]
        for model_name in MODEL_ORDER
    )

    total_fn = sum(
        aggregate[
            model_name
        ]["FN"]
        for model_name in MODEL_ORDER
    )

    largest_fp_model = max(
        MODEL_ORDER,
        key=lambda model_name: (
            aggregate[
                model_name
            ]["FP"]
        ),
    )

    largest_fn_model = max(
        MODEL_ORDER,
        key=lambda model_name: (
            aggregate[
                model_name
            ]["FN"]
        ),
    )

    lines.extend(
        [
            "",
            "## Direct findings",
            "",
            f"- Across all four models, there are "
            f"**{total_fp} false-positive windows** and "
            f"**{total_fn} false-negative windows**.",
            f"- The dominant graph-error count is "
            f"**{'false positives' if total_fp > total_fn else 'false negatives'}**.",
            f"- Largest FP problem: "
            f"**{MODEL_LABELS[largest_fp_model]}**, "
            f"with {aggregate[largest_fp_model]['FP']} "
            f"test false positives.",
            f"- Largest FN problem: "
            f"**{MODEL_LABELS[largest_fn_model]}**, "
            f"with {aggregate[largest_fn_model]['FN']} "
            f"test false negatives.",
            "",
            "## Worst normal test runs",
            "",
        ]
    )

    for model_name in MODEL_ORDER:
        lines.append(
            f"### {MODEL_LABELS[model_name]}"
        )

        lines.extend(
            top_run_lines(
                false_positive_runs,
                model_name,
                "number_of_FP_windows",
            )
        )

        no_fp_count = int(
            false_positive_runs[
                false_positive_runs[
                    "model_name"
                ].eq(model_name)
                & false_positive_runs[
                    "split"
                ].eq("test")
                & false_positive_runs[
                    "no_false_positives"
                ]
            ].shape[0]
        )

        over_half_count = int(
            false_positive_runs[
                false_positive_runs[
                    "model_name"
                ].eq(model_name)
                & false_positive_runs[
                    "split"
                ].eq("test")
                & false_positive_runs[
                    "more_than_50_percent_FP"
                ]
            ].shape[0]
        )

        over_eighty_count = int(
            false_positive_runs[
                false_positive_runs[
                    "model_name"
                ].eq(model_name)
                & false_positive_runs[
                    "split"
                ].eq("test")
                & false_positive_runs[
                    "more_than_80_percent_FP"
                ]
            ].shape[0]
        )

        lines.append(
            f"- Normal test runs with zero FPs: "
            f"{no_fp_count}; above 50% FP: "
            f"{over_half_count}; above 80% FP: "
            f"{over_eighty_count}."
        )

        lines.append("")

    lines.extend(
        [
            "## Worst attack test runs",
            "",
        ]
    )

    for model_name in MODEL_ORDER:
        lines.append(
            f"### {MODEL_LABELS[model_name]}"
        )

        lines.extend(
            top_run_lines(
                false_negative_runs,
                model_name,
                "number_of_FN_windows",
            )
        )

        no_fn_count = int(
            false_negative_runs[
                false_negative_runs[
                    "model_name"
                ].eq(model_name)
                & false_negative_runs[
                    "split"
                ].eq("test")
                & false_negative_runs[
                    "no_false_negatives"
                ]
            ].shape[0]
        )

        over_half_count = int(
            false_negative_runs[
                false_negative_runs[
                    "model_name"
                ].eq(model_name)
                & false_negative_runs[
                    "split"
                ].eq("test")
                & false_negative_runs[
                    "more_than_50_percent_FN"
                ]
            ].shape[0]
        )

        lines.append(
            f"- Attack test runs with zero FNs: "
            f"{no_fn_count}; above 50% FN: "
            f"{over_half_count}."
        )

        lines.append("")

    lines.extend(
        [
            "## Error concentration",
            "",
        ]
    )

    for model_name in MODEL_ORDER:
        entries = []

        for error_type in [
            "FP",
            "FN",
        ]:
            row = concentration[
                concentration[
                    "model_name"
                ].eq(model_name)
                & concentration[
                    "split"
                ].eq("test")
                & concentration[
                    "error_type"
                ].eq(error_type)
                & concentration[
                    "selection"
                ].eq("worst_2_runs")
            ]

            if row.empty:
                continue

            entries.append(
                f"{error_type}: worst two runs "
                f"contribute "
                f"{float(row.iloc[0]['window_weighted_error_share']):.1%}"
            )

        lines.append(
            f"- **{MODEL_LABELS[model_name]}:** "
            + "; ".join(entries)
            + "."
        )

    test_errors = window_table[
        window_table[
            "split"
        ].eq("test")
        & window_table[
            "is_error"
        ]
    ]

    confidence_counts = (
        test_errors[
            "confidence_category"
        ].value_counts()
    )

    total_test_errors = max(
        1,
        len(test_errors),
    )

    borderline_share = (
        int(
            confidence_counts.get(
                "borderline",
                0,
            )
        )
        / total_test_errors
    )

    high_confidence_share = (
        int(
            confidence_counts.get(
                "high_confidence_error",
                0,
            )
        )
        / total_test_errors
    )

    test_run_metrics = run_metrics[
        run_metrics["split"].eq("test")
    ]

    longest_streak_row = (
        test_run_metrics.sort_values(
            "longest_error_streak",
            ascending=False,
        ).iloc[0]
    )

    lines.extend(
        [
            "",
            "## Confidence and persistence",
            "",
            f"- Borderline errors: "
            f"{borderline_share:.1%} of test "
            f"graph-error windows.",
            f"- High-confidence errors: "
            f"{high_confidence_share:.1%} of test "
            f"graph-error windows.",
            f"- Longest test error streak: "
            f"{int(longest_streak_row['longest_error_streak'])} "
            f"consecutive windows in "
            f"`{longest_streak_row['run_id']}` "
            f"for "
            f"{MODEL_LABELS[longest_streak_row['model_name']]}.",
        ]
    )

    if high_confidence_share >= 0.50:
        lines.append(
            "- The large number of high-confidence "
            "errors suggests that threshold adjustment "
            "alone is unlikely to solve the problem."
        )
    elif borderline_share >= 0.60:
        lines.append(
            "- The probability behaviour is consistent "
            "with a possible threshold-transfer problem, "
            "but a formal threshold sweep is still required."
        )
    else:
        lines.append(
            "- Error confidence is mixed. Threshold "
            "mismatch may contribute, but it is not proven."
        )

    test_agreement = agreement[
        agreement["split"].eq("test")
    ]

    lines.extend(
        [
            "",
            "## Cross-model agreement",
            "",
            f"- All four models correct: "
            f"{int(test_agreement['all_four_correct'].sum())} "
            f"test windows.",
            f"- All four models wrong: "
            f"{int(test_agreement['all_four_wrong'].sum())} "
            f"test windows.",
            f"- Conv1D correct while all TCN models "
            f"are wrong: "
            f"{int(test_agreement['conv1d_correct_all_tcns_wrong'].sum())} "
            f"windows.",
            f"- All TCN models correct while Conv1D "
            f"is wrong: "
            f"{int(test_agreement['all_tcns_correct_conv1d_wrong'].sum())} "
            f"windows.",
            f"- Attention uniquely fails: "
            f"{int(test_agreement['attention_only_failure'].sum())} "
            f"windows.",
            f"- MaxPool uniquely succeeds: "
            f"{int(test_agreement['maxpool_only_success'].sum())} "
            f"windows.",
        ]
    )

    gap_rows = []

    for model_name in MODEL_ORDER:
        split_metrics = {}

        for split_name in [
            "val",
            "test",
        ]:
            subset = window_table[
                window_table[
                    "model_name"
                ].eq(model_name)
                & window_table[
                    "split"
                ].eq(split_name)
            ]

            split_metrics[split_name] = (
                metrics_from_counts(
                    binary_counts(
                        subset[
                            "true_graph"
                        ],
                        subset[
                            "pred_graph"
                        ],
                    )
                )
            )

        gap_rows.append(
            {
                "model_name": model_name,
                "FPR_gap": (
                    split_metrics[
                        "test"
                    ]["FPR"]
                    - split_metrics[
                        "val"
                    ]["FPR"]
                ),
                "FNR_gap": (
                    split_metrics[
                        "test"
                    ]["FNR"]
                    - split_metrics[
                        "val"
                    ]["FNR"]
                ),
            }
        )

    gap_table = pd.DataFrame(
        gap_rows
    )

    mean_fpr_gap = float(
        gap_table[
            "FPR_gap"
        ].mean()
    )

    mean_fnr_gap = float(
        gap_table[
            "FNR_gap"
        ].mean()
    )

    if mean_fpr_gap > 1.25 * mean_fnr_gap:
        diagnosis = (
            "primarily a benign-distribution "
            "false-positive shift, with a secondary "
            "attack-generalization failure"
        )
    elif mean_fnr_gap > 1.25 * mean_fpr_gap:
        diagnosis = (
            "primarily an attack-generalization "
            "false-negative shift, with a secondary "
            "benign false-positive shift"
        )
    else:
        diagnosis = (
            "a combination of benign false-positive "
            "shift and attack false-negative shift"
        )

    lines.extend(
        [
            "",
            "## Current graph-level diagnosis",
            "",
            f"The mean validation-to-test FPR increase "
            f"is {mean_fpr_gap:.4f}. The mean FNR increase "
            f"is {mean_fnr_gap:.4f}. The current collapse "
            f"is best described as **{diagnosis}**.",
            "",
            "This does not yet prove whether calibration, "
            "feature ambiguity, temporal behaviour, or "
            "representation shift is causal. The next "
            "controlled step is a validation-to-test "
            "threshold-transfer analysis.",
            "",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_analysis(
    window_table: pd.DataFrame,
    run_metrics: pd.DataFrame,
    streaks: pd.DataFrame,
    stage3_metrics: pd.DataFrame,
) -> tuple[bool, str]:
    messages: list[str] = []
    aggregate_metrics_matched = True

    for model_name in MODEL_ORDER:
        for split_name in [
            "val",
            "test",
        ]:
            subset = window_table[
                window_table[
                    "model_name"
                ].eq(model_name)
                & window_table[
                    "split"
                ].eq(split_name)
            ]

            counts = binary_counts(
                subset["true_graph"],
                subset["pred_graph"],
            )

            metrics = metrics_from_counts(
                counts
            )

            expected = stage3_metrics[
                stage3_metrics[
                    "model"
                ].eq(model_name)
                & stage3_metrics[
                    "split"
                ].eq(split_name)
            ]

            if len(expected) != 1:
                aggregate_metrics_matched = False

                messages.append(
                    f"Missing or duplicate Stage 3 "
                    f"metric row: {model_name} "
                    f"{split_name}"
                )

                continue

            expected_row = expected.iloc[0]

            count_mapping = {
                "TN": "tn",
                "FP": "fp",
                "FN": "fn",
                "TP": "tp",
            }

            for calculated_name, stored_name in (
                count_mapping.items()
            ):
                calculated_value = int(
                    counts[
                        calculated_name
                    ]
                )

                stored_value = int(
                    expected_row[
                        stored_name
                    ]
                )

                if (
                    calculated_value
                    != stored_value
                ):
                    aggregate_metrics_matched = False

                    messages.append(
                        f"{model_name} "
                        f"{split_name} "
                        f"{calculated_name}: "
                        f"{calculated_value} "
                        f"!= {stored_value}"
                    )

            metric_mapping = {
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "F1": "f1",
                "FPR": "fpr",
            }

            for calculated_name, stored_name in (
                metric_mapping.items()
            ):
                calculated_value = float(
                    metrics[
                        calculated_name
                    ]
                )

                stored_value = float(
                    expected_row[
                        stored_name
                    ]
                )

                if not np.isclose(
                    calculated_value,
                    stored_value,
                    atol=1e-12,
                    rtol=0,
                ):
                    aggregate_metrics_matched = False

                    messages.append(
                        f"{model_name} "
                        f"{split_name} "
                        f"{calculated_name}: "
                        f"{calculated_value} "
                        f"!= {stored_value}"
                    )

    all_samples_accounted_for = True

    for (
        model_name,
        split_name,
    ), group in window_table.groupby(
        [
            "model_name",
            "split",
        ]
    ):
        run_total = int(
            run_metrics[
                run_metrics[
                    "model_name"
                ].eq(model_name)
                & run_metrics[
                    "split"
                ].eq(split_name)
            ][
                "number_of_windows"
            ].sum()
        )

        if run_total != len(group):
            all_samples_accounted_for = False

            messages.append(
                f"Sample mismatch: {model_name} "
                f"{split_name}: "
                f"{run_total} != {len(group)}"
            )

    all_errors_accounted_for = (
        int(
            run_metrics[
                "number_of_error_windows"
            ].sum()
        )
        == int(
            window_table[
                "is_error"
            ].sum()
        )
    )

    all_streaks_accounted_for = (
        int(
            streaks[
                "streak_length"
            ].sum()
            if not streaks.empty
            else 0
        )
        == int(
            window_table[
                "is_error"
            ].sum()
        )
    )

    all_run_labels_constant = bool(
        (
            window_table.groupby(
                [
                    "split",
                    "run_id",
                ]
            )[
                "true_graph"
            ].nunique()
            == 1
        ).all()
    )

    all_model_sample_keys_unique = not bool(
        window_table.duplicated(
            subset=[
                "model_name",
                "sample_id",
            ]
        ).any()
    )

    probability_statistic_columns = [
        "mean_graph_probability",
        "std_graph_probability",
        "minimum_graph_probability",
        "maximum_graph_probability",
        "p01",
        "p05",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
    ]

    all_statistics_finite = bool(
        np.isfinite(
            run_metrics[
                probability_statistic_columns
            ].to_numpy(dtype=float)
        ).all()
    )

    overall_success = all(
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

    validation_lines = [
        "aggregate metrics matched: "
        f"{aggregate_metrics_matched}",
        "all samples accounted for: "
        f"{all_samples_accounted_for}",
        "all errors accounted for: "
        f"{all_errors_accounted_for}",
        "all streaks accounted for: "
        f"{all_streaks_accounted_for}",
        "all run labels constant: "
        f"{all_run_labels_constant}",
        "all model/sample keys unique: "
        f"{all_model_sample_keys_unique}",
        "all statistics finite: "
        f"{all_statistics_finite}",
        "overall success: "
        f"{overall_success}",
    ]

    if messages:
        validation_lines.extend(
            [
                "",
                "DETAILS:",
                *messages,
            ]
        )

    return (
        overall_success,
        "\n".join(
            validation_lines
        )
        + "\n",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse V3 graph-level false positives, "
            "false negatives, run concentration and "
            "cross-model agreement."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "reports/v3_failure_analysis"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    root = args.root.resolve()
    prediction_directory = (
        root / "predictions"
    )
    table_directory = root / "tables"
    plot_directory = (
        root
        / "plots"
        / "graph_failures"
    )
    log_directory = root / "logs"

    metadata_path = (
        prediction_directory
        / "prediction_metadata.csv"
    )

    stage3_metrics_path = (
        table_directory
        / "exported_metric_reproduction.csv"
    )

    report_path = (
        root
        / "GRAPH_FAILURE_ANALYSIS.md"
    )

    validation_path = (
        log_directory
        / "28_graph_failure_validation.txt"
    )

    output_paths = {
        "window_predictions": (
            table_directory
            / "graph_window_predictions.csv"
        ),
        "run_metrics": (
            table_directory
            / "graph_metrics_by_run.csv"
        ),
        "error_streaks": (
            table_directory
            / "graph_error_streaks.csv"
        ),
        "false_positive_runs": (
            table_directory
            / "false_positive_runs.csv"
        ),
        "false_negative_runs": (
            table_directory
            / "false_negative_runs.csv"
        ),
        "group_metrics": (
            table_directory
            / "graph_metrics_by_group.csv"
        ),
        "error_concentration": (
            table_directory
            / "graph_error_concentration.csv"
        ),
        "val_test_gap": (
            table_directory
            / "graph_val_test_gap.csv"
        ),
        "model_agreement": (
            table_directory
            / "graph_model_agreement.csv"
        ),
        "common_error_runs": (
            table_directory
            / "common_graph_error_runs.csv"
        ),
        "report": report_path,
        "validation": validation_path,
    }

    existing_outputs = [
        path
        for path in output_paths.values()
        if path.exists()
    ]

    if (
        existing_outputs
        and not args.overwrite
    ):
        raise FileExistsError(
            "Stage 4 outputs already exist. "
            "Use --overwrite to replace them:\n"
            + "\n".join(
                str(path)
                for path in existing_outputs
            )
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            metadata_path
        )

    if not stage3_metrics_path.is_file():
        raise FileNotFoundError(
            stage3_metrics_path
        )

    print(
        f"root: {root}",
        flush=True,
    )

    print(
        f"metadata: {metadata_path}",
        flush=True,
    )

    print(
        f"stage3 metrics: "
        f"{stage3_metrics_path}",
        flush=True,
    )

    metadata = pd.read_csv(
        metadata_path,
        keep_default_na=False,
    )

    stage3_metrics = pd.read_csv(
        stage3_metrics_path,
        keep_default_na=False,
    )

    loaded_predictions = (
        load_prediction_files(
            prediction_directory
        )
    )

    verify_alignment(
        metadata,
        loaded_predictions,
    )

    print(
        "input alignment verified",
        flush=True,
    )

    window_table = build_window_table(
        metadata,
        loaded_predictions,
    )

    print(
        f"model/sample rows: "
        f"{len(window_table)}",
        flush=True,
    )

    streaks = build_error_streaks(
        window_table
    )

    print(
        f"error streaks: {len(streaks)}",
        flush=True,
    )

    run_metrics = build_run_metrics(
        window_table,
        streaks,
    )

    (
        false_positive_runs,
        false_negative_runs,
    ) = (
        build_false_positive_negative_tables(
            run_metrics
        )
    )

    group_metrics = build_group_metrics(
        window_table
    )

    error_concentration = (
        build_error_concentration(
            run_metrics
        )
    )

    val_test_gap = build_val_test_gap(
        group_metrics
    )

    (
        model_agreement,
        common_error_runs,
    ) = build_model_agreement(
        window_table
    )

    (
        validation_success,
        validation_text,
    ) = validate_analysis(
        window_table,
        run_metrics,
        streaks,
        stage3_metrics,
    )

    atomic_text(
        validation_text,
        validation_path,
    )

    print(
        validation_text,
        end="",
        flush=True,
    )

    if not validation_success:
        raise SystemExit(
            "Stage 4 validation failed. "
            "No analysis tables were written. "
            f"Inspect {validation_path}."
        )

    atomic_csv(
        window_table,
        output_paths[
            "window_predictions"
        ],
    )

    atomic_csv(
        run_metrics,
        output_paths[
            "run_metrics"
        ],
    )

    atomic_csv(
        streaks,
        output_paths[
            "error_streaks"
        ],
    )

    atomic_csv(
        false_positive_runs,
        output_paths[
            "false_positive_runs"
        ],
    )

    atomic_csv(
        false_negative_runs,
        output_paths[
            "false_negative_runs"
        ],
    )

    atomic_csv(
        group_metrics,
        output_paths[
            "group_metrics"
        ],
    )

    atomic_csv(
        error_concentration,
        output_paths[
            "error_concentration"
        ],
    )

    atomic_csv(
        val_test_gap,
        output_paths[
            "val_test_gap"
        ],
    )

    atomic_csv(
        model_agreement,
        output_paths[
            "model_agreement"
        ],
    )

    atomic_csv(
        common_error_runs,
        output_paths[
            "common_error_runs"
        ],
    )

    make_plots(
        window_table,
        run_metrics,
        streaks,
        model_agreement,
        plot_directory,
    )

    report = build_report(
        window_table,
        run_metrics,
        false_positive_runs,
        false_negative_runs,
        error_concentration,
        model_agreement,
    )

    atomic_text(
        report,
        report_path,
    )

    print(
        f"plots written to: "
        f"{plot_directory}",
        flush=True,
    )

    print(
        "Stage 4 output files:",
        flush=True,
    )

    for name, path in output_paths.items():
        print(
            f"  {name}: {path}",
            flush=True,
        )

    print(
        "overall success: True",
        flush=True,
    )


if __name__ == "__main__":
    main()
