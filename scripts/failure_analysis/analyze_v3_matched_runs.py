#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path
from typing import Any, Iterable

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
PRIMARY_NORMAL_CONTROL = "N-1-6-9-14-Pmixed-R17-V3"

BROADER_NORMAL_CONTROLS = [
    "N-2-13-Pstream-R15-V3",
    "N-5-10-Pbursty-R16-V3",
    "N-idle-Pidle-R2-V3",
    "N-5-Pcompute-R14-V3",
]

HARD_ATTACK_RUN = "N-5-10-Pbursty-R51-A-12-S20-V3"
PRIMARY_ATTACK_CONTROLS = [
    "N-2-13-Pstream-R50-A-12-S20-V3",
    "N-1-6-9-14-Pmixed-R52-A-12-S20-V3",
]

QUANTILES = {
    "p01": 0.01,
    "p05": 0.05,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p95": 0.95,
    "p99": 0.99,
}


# ---------------------------------------------------------------------------
# Atomic output helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Basic parsing and numerical helpers
# ---------------------------------------------------------------------------

def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def parse_token_set(value: Any) -> frozenset[str]:
    text = str(value).strip()
    if not text or text.lower() in {"idle", "none", "nan"}:
        return frozenset()
    return frozenset(part.strip() for part in text.split("-") if part.strip())


def parse_count(value: Any) -> int:
    return len(parse_token_set(value))


def quantile_values(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.quantile(values, fraction))
        for name, fraction in QUANTILES.items()
    }


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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": float(f1),
        "FPR": safe_divide(fp, fp + tn),
        "FNR": safe_divide(fn, fn + tp),
    }


def run_lengths(states: np.ndarray) -> list[tuple[bool, int, int, int]]:
    states = np.asarray(states, dtype=bool).reshape(-1)
    if states.size == 0:
        return []

    results: list[tuple[bool, int, int, int]] = []
    start = 0
    current = bool(states[0])

    for index in range(1, len(states)):
        state = bool(states[index])
        if state != current:
            results.append((current, start, index - 1, index - start))
            start = index
            current = state

    results.append((current, start, len(states) - 1, len(states) - start))
    return results


def count_threshold_crossings(states: np.ndarray) -> int:
    states = np.asarray(states, dtype=np.int8).reshape(-1)
    if states.size < 2:
        return 0
    return int(np.sum(states[1:] != states[:-1]))


def mean_duration_for_state(
    segments: list[tuple[bool, int, int, int]],
    target_state: bool,
) -> float:
    durations = [length for state, _, _, length in segments if state == target_state]
    if not durations:
        return 0.0
    return float(np.mean(durations))


def median_duration_for_state(
    segments: list[tuple[bool, int, int, int]],
    target_state: bool,
) -> float:
    durations = [length for state, _, _, length in segments if state == target_state]
    if not durations:
        return 0.0
    return float(np.median(durations))


def histogram_overlap(
    left: np.ndarray,
    right: np.ndarray,
    bins: int = 100,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    left_hist, _ = np.histogram(left, bins=edges, density=False)
    right_hist, _ = np.histogram(right, bins=edges, density=False)

    left_mass = left_hist / max(1, left_hist.sum())
    right_mass = right_hist / max(1, right_hist.sum())
    return float(np.minimum(left_mass, right_mass).sum())


def ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(np.asarray(left, dtype=float))
    right = np.sort(np.asarray(right, dtype=float))

    combined = np.sort(np.unique(np.concatenate([left, right])))
    left_cdf = np.searchsorted(left, combined, side="right") / len(left)
    right_cdf = np.searchsorted(right, combined, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)

    if len(left) != len(right):
        raise ValueError("Correlation arrays have different lengths.")

    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0

    return float(np.corrcoef(left, right)[0, 1])


def shared_epoch_arrays(
    left: pd.DataFrame,
    right: pd.DataFrame,
    value_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    merged = left[["end_epoch", value_column]].merge(
        right[["end_epoch", value_column]],
        on="end_epoch",
        suffixes=("_left", "_right"),
        how="inner",
    )

    return (
        merged[f"{value_column}_left"].to_numpy(dtype=float),
        merged[f"{value_column}_right"].to_numpy(dtype=float),
    )


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------

def load_inputs(
    prediction_directory: Path,
    metadata_path: Path,
    selected_thresholds_path: Path,
    stage5_results_path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, np.ndarray]],
    pd.DataFrame,
    pd.DataFrame,
]:
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
            f"Metadata missing columns: {sorted(missing_metadata)}"
        )

    selected_thresholds = pd.read_csv(
        selected_thresholds_path,
        keep_default_na=False,
        low_memory=False,
    )
    required_threshold_columns = {
        "model_name",
        "selected_threshold",
        "selection_used_test_data",
    }
    missing_threshold_columns = (
        required_threshold_columns - set(selected_thresholds.columns)
    )
    if missing_threshold_columns:
        raise RuntimeError(
            "Selected-threshold table missing columns: "
            f"{sorted(missing_threshold_columns)}"
        )

    stage5_results = pd.read_csv(
        stage5_results_path,
        keep_default_na=False,
        low_memory=False,
    )
    required_stage5_columns = {
        "model_name",
        "split",
        "operating_point",
        "threshold",
        "TN",
        "FP",
        "FN",
        "TP",
        "F1",
        "FPR",
        "FNR",
    }
    missing_stage5_columns = required_stage5_columns - set(stage5_results.columns)
    if missing_stage5_columns:
        raise RuntimeError(
            f"Stage 5 results missing columns: {sorted(missing_stage5_columns)}"
        )

    loaded: dict[str, dict[str, np.ndarray]] = {}

    for model_name, filename in MODEL_FILES.items():
        path = prediction_directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)

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
                    f"{model_name} prediction file missing: {sorted(missing_arrays)}"
                )
            loaded[model_name] = {
                name: source[name]
                for name in source.files
            }

        print(f"loaded {model_name}: {path}", flush=True)

    return metadata, loaded, selected_thresholds, stage5_results


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
                f"{model_name} sample count differs from metadata."
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
                    f"Cross-model alignment failed for {model_name}: {key}"
                )


def selected_threshold_map(
    selected_thresholds: pd.DataFrame,
) -> dict[str, float]:
    mapping: dict[str, float] = {}

    for model_name in MODEL_ORDER:
        rows = selected_thresholds[
            selected_thresholds["model_name"].eq(model_name)
        ]
        if len(rows) != 1:
            raise RuntimeError(
                f"Expected exactly one selected threshold for {model_name}."
            )

        row = rows.iloc[0]
        used_test = str(row["selection_used_test_data"]).strip().lower()
        if used_test not in {"false", "0"}:
            raise RuntimeError(
                f"Selected threshold for {model_name} used test data."
            )

        threshold = float(row["selected_threshold"])
        if not 0.0 < threshold < 1.0:
            raise RuntimeError(
                f"Invalid selected threshold for {model_name}: {threshold}"
            )
        mapping[model_name] = threshold

    return mapping


# ---------------------------------------------------------------------------
# Working tables and run inventory
# ---------------------------------------------------------------------------

def build_window_table(
    metadata: pd.DataFrame,
    loaded: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    base = metadata.rename(
        columns={"sample_order_within_export": "sample_order"}
    ).copy()

    base["sample_index"] = base["sample_index"].astype(np.int64)
    base["end_epoch"] = base["end_epoch"].astype(np.int64)
    base["true_graph"] = base["true_graph"].astype(np.int8)
    base["attacker_count"] = base["true_attacker_count"].astype(np.int64)
    base["active_core_count"] = (
        base["active_cores"].map(parse_count).astype(np.int64)
    )
    base["active_core_set"] = base["active_cores"].map(
        lambda value: "-".join(sorted(parse_token_set(value), key=int))
        if parse_token_set(value)
        else ""
    )
    base["attacker_set"] = base["attackers"].map(
        lambda value: "-".join(sorted(parse_token_set(value), key=int))
        if parse_token_set(value)
        else ""
    )

    frames = []

    for model_name in MODEL_ORDER:
        arrays = loaded[model_name]
        frame = base.copy()
        frame["model_name"] = model_name
        frame["graph_logit"] = np.asarray(
            arrays["graph_logit"], dtype=np.float64
        )
        frame["graph_probability"] = np.asarray(
            arrays["graph_probability"], dtype=np.float64
        )

        if not np.isfinite(
            frame[["graph_logit", "graph_probability"]].to_numpy(dtype=float)
        ).all():
            raise RuntimeError(
                f"Non-finite output detected for {model_name}."
            )

        frames.append(frame)

    table = pd.concat(frames, ignore_index=True)

    if table.duplicated(["model_name", "sample_id"]).any():
        raise RuntimeError("Duplicate model/sample rows in window table.")

    return table


def build_run_metadata(window_table: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        "split",
        "run_id",
        "true_graph",
        "profile",
        "active_cores",
        "active_core_count",
        "active_core_set",
        "attackers",
        "attacker_count",
        "attacker_set",
        "strength",
        "seed",
    ]

    run_metadata = (
        window_table[metadata_columns]
        .drop_duplicates()
        .sort_values(["split", "run_id"])
        .reset_index(drop=True)
    )

    if run_metadata.duplicated(["split", "run_id"]).any():
        raise RuntimeError("A run has inconsistent metadata.")

    return run_metadata


def role_labels_for_run(row: pd.Series) -> list[str]:
    run_id = str(row["run_id"])
    roles: list[str] = []

    if run_id == HARD_NORMAL_RUN:
        roles.append("hard_normal_target")
    if run_id == PRIMARY_NORMAL_CONTROL:
        roles.append("normal_primary_test_control")
    if run_id in BROADER_NORMAL_CONTROLS:
        roles.append("normal_broader_test_control")

    if run_id == HARD_ATTACK_RUN:
        roles.append("hard_attack_target")
    if run_id in PRIMARY_ATTACK_CONTROLS:
        roles.append("attack_primary_test_control")

    if int(row["true_graph"]) == 0:
        if str(row["profile"]) == "mixed":
            roles.append("normal_mixed")
        if int(row["active_core_count"]) == 4:
            roles.append("normal_four_core")
        if (
            str(row["profile"]) == "mixed"
            and int(row["active_core_count"]) == 4
        ):
            roles.append("normal_mixed_four_core")
            if str(row["split"]) == "val":
                roles.append("normal_validation_analogue")
    else:
        if int(row["attacker_count"]) == 1:
            roles.append("attack_one_attacker")
        if str(row["strength"]) == "20":
            roles.append("attack_strength_20")
        if str(row["profile"]) == "bursty":
            roles.append("attack_bursty")
        if str(row["active_core_set"]) == "5-10":
            roles.append("attack_active_cores_5_10")
        if "12" in parse_token_set(row["attackers"]):
            roles.append("attack_contains_attacker_12")
        if int(row["attacker_count"]) == 2 and str(row["strength"]) == "63":
            roles.append("attack_two_attacker_strength_63")
        if int(row["attacker_count"]) == 3 and str(row["strength"]) == "77":
            roles.append("attack_three_attacker_strength_77")
        if (
            int(row["attacker_count"]) == 1
            and str(row["strength"]) == "20"
            and "12" in parse_token_set(row["attackers"])
        ):
            roles.append("attack_exact_S20_A12_family")
            if str(row["split"]) == "val":
                roles.append("attack_validation_analogue")

    if not roles:
        roles.append("context_run")

    return roles


def factor_match_flags(
    target: pd.Series,
    candidate: pd.Series,
) -> dict[str, Any]:
    same_class = int(target["true_graph"]) == int(candidate["true_graph"])
    flags = {
        "same_true_graph": same_class,
        "same_split": str(target["split"]) == str(candidate["split"]),
        "same_profile": str(target["profile"]) == str(candidate["profile"]),
        "same_active_core_count": (
            int(target["active_core_count"])
            == int(candidate["active_core_count"])
        ),
        "same_active_cores": (
            str(target["active_core_set"])
            == str(candidate["active_core_set"])
        ),
        "same_attacker_count": (
            int(target["attacker_count"])
            == int(candidate["attacker_count"])
        ),
        "same_attackers": (
            str(target["attacker_set"])
            == str(candidate["attacker_set"])
        ),
        "same_strength": (
            str(target["strength"]) == str(candidate["strength"])
        ),
        "same_seed": str(target["seed"]) == str(candidate["seed"]),
    }

    normal_weights = {
        "same_true_graph": 4,
        "same_split": 1,
        "same_profile": 3,
        "same_active_core_count": 2,
        "same_active_cores": 4,
        "same_seed": 1,
    }
    attack_weights = {
        "same_true_graph": 4,
        "same_split": 1,
        "same_profile": 2,
        "same_active_core_count": 1,
        "same_active_cores": 3,
        "same_attacker_count": 3,
        "same_attackers": 4,
        "same_strength": 3,
        "same_seed": 1,
    }

    weights = normal_weights if int(target["true_graph"]) == 0 else attack_weights
    match_score = sum(
        weight for name, weight in weights.items() if flags[name]
    )
    maximum_score = sum(weights.values())

    flags["factor_match_score"] = int(match_score)
    flags["factor_match_fraction"] = float(match_score / maximum_score)
    return flags


def build_inventory(
    run_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_runs = {
        HARD_NORMAL_RUN,
        PRIMARY_NORMAL_CONTROL,
        *BROADER_NORMAL_CONTROLS,
        HARD_ATTACK_RUN,
        *PRIMARY_ATTACK_CONTROLS,
    }
    available_runs = set(run_metadata["run_id"].astype(str))
    missing = required_runs - available_runs
    if missing:
        raise RuntimeError(
            f"Required matched runs are missing: {sorted(missing)}"
        )

    target_normal = run_metadata[
        run_metadata["run_id"].eq(HARD_NORMAL_RUN)
    ].iloc[0]
    target_attack = run_metadata[
        run_metadata["run_id"].eq(HARD_ATTACK_RUN)
    ].iloc[0]

    inventory_rows = []
    factor_rows = []

    for _, row in run_metadata.iterrows():
        roles = role_labels_for_run(row)
        target = target_normal if int(row["true_graph"]) == 0 else target_attack
        flags = factor_match_flags(target, row)

        exact_match_type = "partial_context"
        if str(row["run_id"]) in {
            HARD_NORMAL_RUN,
            HARD_ATTACK_RUN,
        }:
            exact_match_type = "target"
        elif str(row["run_id"]) == PRIMARY_NORMAL_CONTROL:
            exact_match_type = "exact_normal_profile_corecount_test_control"
        elif str(row["run_id"]) in PRIMARY_ATTACK_CONTROLS:
            exact_match_type = "exact_attack_A12_S20_single_attacker_test_control"
        elif "normal_validation_analogue" in roles:
            exact_match_type = "validation_normal_profile_corecount_analogue"
        elif "attack_validation_analogue" in roles:
            exact_match_type = "validation_attack_A12_S20_single_attacker_analogue"

        inventory_rows.append(
            {
                **row.to_dict(),
                "roles": "|".join(roles),
                "comparison_target": (
                    HARD_NORMAL_RUN
                    if int(row["true_graph"]) == 0
                    else HARD_ATTACK_RUN
                ),
                "match_type": exact_match_type,
                "factor_match_score": flags["factor_match_score"],
                "factor_match_fraction": flags["factor_match_fraction"],
                "probability_source_available": True,
                "feature_tensor_loaded": False,
                "training_prediction_available": False,
            }
        )

        factor_rows.append(
            {
                "target_run_id": str(target["run_id"]),
                "candidate_run_id": str(row["run_id"]),
                "candidate_split": str(row["split"]),
                "candidate_true_graph": int(row["true_graph"]),
                "candidate_roles": "|".join(roles),
                **flags,
                "target_profile": str(target["profile"]),
                "candidate_profile": str(row["profile"]),
                "target_active_cores": str(target["active_cores"]),
                "candidate_active_cores": str(row["active_cores"]),
                "target_attackers": str(target["attackers"]),
                "candidate_attackers": str(row["attackers"]),
                "target_strength": str(target["strength"]),
                "candidate_strength": str(row["strength"]),
                "target_seed": str(target["seed"]),
                "candidate_seed": str(row["seed"]),
            }
        )

    inventory = pd.DataFrame(inventory_rows).sort_values(
        ["true_graph", "factor_match_score", "split", "run_id"],
        ascending=[True, False, True, True],
    )
    factor_comparison = pd.DataFrame(factor_rows).sort_values(
        ["target_run_id", "factor_match_score", "candidate_split"],
        ascending=[True, False, True],
    )

    return inventory, factor_comparison


# ---------------------------------------------------------------------------
# Probability summaries, operating points and temporal behaviour
# ---------------------------------------------------------------------------

def operating_points_for_model(
    model_name: str,
    threshold_map: dict[str, float],
) -> list[tuple[str, float]]:
    return [
        ("reference_0_50", REFERENCE_THRESHOLD),
        ("validation_selected", float(threshold_map[model_name])),
    ]


def summarize_probabilities(window_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    quantile_rows = []

    for (model_name, split_name, run_id), run in window_table.groupby(
        ["model_name", "split", "run_id"],
        sort=True,
    ):
        run = run.sort_values("end_epoch")
        values = run["graph_probability"].to_numpy(dtype=float)
        quantiles = quantile_values(values)

        metadata = {
            "model_name": model_name,
            "split": split_name,
            "run_id": run_id,
            "true_graph": int(run["true_graph"].iloc[0]),
            "profile": str(run["profile"].iloc[0]),
            "active_cores": str(run["active_cores"].iloc[0]),
            "active_core_count": int(run["active_core_count"].iloc[0]),
            "attackers": str(run["attackers"].iloc[0]),
            "attacker_count": int(run["attacker_count"].iloc[0]),
            "strength": str(run["strength"].iloc[0]),
            "seed": str(run["seed"].iloc[0]),
            "number_of_windows": int(len(run)),
        }

        summary_rows.append(
            {
                **metadata,
                "mean_graph_probability": float(np.mean(values)),
                "median_graph_probability": float(np.median(values)),
                "std_graph_probability": float(np.std(values, ddof=0)),
                "minimum_graph_probability": float(np.min(values)),
                "maximum_graph_probability": float(np.max(values)),
                "fraction_below_0_10": float(np.mean(values < 0.10)),
                "fraction_below_0_25": float(np.mean(values < 0.25)),
                "fraction_between_0_40_and_0_60": float(
                    np.mean((values >= 0.40) & (values <= 0.60))
                ),
                "fraction_above_0_75": float(np.mean(values > 0.75)),
                "fraction_above_0_90": float(np.mean(values > 0.90)),
                **quantiles,
            }
        )

        quantile_rows.append(
            {
                **metadata,
                **quantiles,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(quantile_rows)


def build_error_streak_rows(
    error_states: np.ndarray,
    epochs: np.ndarray,
) -> list[dict[str, Any]]:
    segments = run_lengths(error_states)
    rows = []

    streak_id = 0
    for state, start_index, finish_index, length in segments:
        if not state:
            continue
        streak_id += 1
        rows.append(
            {
                "streak_id": streak_id,
                "start_end_epoch": int(epochs[start_index]),
                "finish_end_epoch": int(epochs[finish_index]),
                "streak_length": int(length),
            }
        )

    return rows


def build_operating_point_tables(
    window_table: pd.DataFrame,
    threshold_map: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    streak_rows = []
    crossing_rows = []

    for (model_name, split_name, run_id), run in window_table.groupby(
        ["model_name", "split", "run_id"],
        sort=True,
    ):
        run = run.sort_values("end_epoch")
        values = run["graph_probability"].to_numpy(dtype=float)
        truth = run["true_graph"].to_numpy(dtype=np.int8)
        epochs = run["end_epoch"].to_numpy(dtype=np.int64)
        true_graph = int(truth[0])

        metadata = {
            "model_name": model_name,
            "split": split_name,
            "run_id": run_id,
            "true_graph": true_graph,
            "profile": str(run["profile"].iloc[0]),
            "active_cores": str(run["active_cores"].iloc[0]),
            "active_core_count": int(run["active_core_count"].iloc[0]),
            "attackers": str(run["attackers"].iloc[0]),
            "attacker_count": int(run["attacker_count"].iloc[0]),
            "strength": str(run["strength"].iloc[0]),
            "seed": str(run["seed"].iloc[0]),
            "number_of_windows": int(len(run)),
        }

        for operating_point, threshold in operating_points_for_model(
            model_name, threshold_map
        ):
            predicted = (values >= threshold).astype(np.int8)
            errors = predicted != truth
            counts = confusion_counts(truth, predicted)
            metrics = metrics_from_counts(counts)
            states_above = predicted.astype(bool)
            state_segments = run_lengths(states_above)
            error_streaks = build_error_streak_rows(errors, epochs)

            if error_streaks:
                lengths = np.asarray(
                    [row["streak_length"] for row in error_streaks],
                    dtype=float,
                )
                longest_error_streak = int(np.max(lengths))
                median_error_streak = float(np.median(lengths))
                mean_error_streak = float(np.mean(lengths))
                first_error_epoch = int(
                    min(row["start_end_epoch"] for row in error_streaks)
                )
                last_error_epoch = int(
                    max(row["finish_end_epoch"] for row in error_streaks)
                )
            else:
                longest_error_streak = 0
                median_error_streak = 0.0
                mean_error_streak = 0.0
                first_error_epoch = np.nan
                last_error_epoch = np.nan

            n = len(run)
            first_cut = max(1, int(math.ceil(n * 0.10)))
            final_cut = min(n - 1, int(math.floor(n * 0.90)))

            early_error_rate = float(np.mean(errors[:first_cut]))
            middle_error_rate = float(np.mean(errors[first_cut:final_cut]))
            late_error_rate = float(np.mean(errors[final_cut:]))

            metric_rows.append(
                {
                    **metadata,
                    "operating_point": operating_point,
                    "threshold": float(threshold),
                    **counts,
                    **metrics,
                    "error_count": int(errors.sum()),
                    "error_rate": float(np.mean(errors)),
                    "fraction_of_run_in_error": float(np.mean(errors)),
                    "number_of_error_streaks": int(len(error_streaks)),
                    "longest_error_streak": longest_error_streak,
                    "median_error_streak_length": median_error_streak,
                    "mean_error_streak_length": mean_error_streak,
                    "first_error_epoch": first_error_epoch,
                    "last_error_epoch": last_error_epoch,
                    "early_10pct_error_rate": early_error_rate,
                    "middle_80pct_error_rate": middle_error_rate,
                    "late_10pct_error_rate": late_error_rate,
                }
            )

            for row in error_streaks:
                streak_rows.append(
                    {
                        **metadata,
                        "operating_point": operating_point,
                        "threshold": float(threshold),
                        **row,
                    }
                )

            crossing_rows.append(
                {
                    **metadata,
                    "operating_point": operating_point,
                    "threshold": float(threshold),
                    "threshold_crossing_count": count_threshold_crossings(
                        states_above
                    ),
                    "number_of_above_threshold_segments": int(
                        sum(state for state, _, _, _ in state_segments)
                    ),
                    "number_of_below_threshold_segments": int(
                        sum(not state for state, _, _, _ in state_segments)
                    ),
                    "mean_duration_above_threshold": mean_duration_for_state(
                        state_segments, True
                    ),
                    "median_duration_above_threshold": median_duration_for_state(
                        state_segments, True
                    ),
                    "maximum_duration_above_threshold": int(
                        max(
                            [
                                length
                                for state, _, _, length in state_segments
                                if state
                            ],
                            default=0,
                        )
                    ),
                    "mean_duration_below_threshold": mean_duration_for_state(
                        state_segments, False
                    ),
                    "median_duration_below_threshold": median_duration_for_state(
                        state_segments, False
                    ),
                    "maximum_duration_below_threshold": int(
                        max(
                            [
                                length
                                for state, _, _, length in state_segments
                                if not state
                            ],
                            default=0,
                        )
                    ),
                    "fraction_above_threshold": float(np.mean(states_above)),
                    "fraction_below_threshold": float(np.mean(~states_above)),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    streaks = pd.DataFrame(streak_rows)
    crossings = pd.DataFrame(crossing_rows)

    return metrics, streaks, crossings


def build_streak_summary(streaks: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    keys = ["model_name", "split", "run_id", "operating_point", "threshold"]

    grouped_streaks = {
        key: group
        for key, group in streaks.groupby(keys, sort=True)
    } if not streaks.empty else {}

    for _, metric_row in metrics.iterrows():
        key = tuple(metric_row[column] for column in keys)
        group = grouped_streaks.get(key)

        if group is None or group.empty:
            number_of_streaks = 0
            longest = 0
            median = 0.0
            mean = 0.0
            first = np.nan
            last = np.nan
        else:
            number_of_streaks = int(len(group))
            longest = int(group["streak_length"].max())
            median = float(group["streak_length"].median())
            mean = float(group["streak_length"].mean())
            first = int(group["start_end_epoch"].min())
            last = int(group["finish_end_epoch"].max())

        summary_rows.append(
            {
                "model_name": metric_row["model_name"],
                "split": metric_row["split"],
                "run_id": metric_row["run_id"],
                "true_graph": int(metric_row["true_graph"]),
                "operating_point": metric_row["operating_point"],
                "threshold": float(metric_row["threshold"]),
                "error_count": int(metric_row["error_count"]),
                "fraction_of_run_in_error": float(
                    metric_row["fraction_of_run_in_error"]
                ),
                "number_of_error_streaks": number_of_streaks,
                "longest_error_streak": longest,
                "median_error_streak_length": median,
                "mean_error_streak_length": mean,
                "first_error_epoch": first,
                "last_error_epoch": last,
                "early_10pct_error_rate": float(
                    metric_row["early_10pct_error_rate"]
                ),
                "middle_80pct_error_rate": float(
                    metric_row["middle_80pct_error_rate"]
                ),
                "late_10pct_error_rate": float(
                    metric_row["late_10pct_error_rate"]
                ),
            }
        )

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Pairwise target/control comparisons
# ---------------------------------------------------------------------------

def controls_for_target(
    run_metadata: pd.DataFrame,
    target_run_id: str,
) -> list[str]:
    target = run_metadata[
        run_metadata["run_id"].eq(target_run_id)
    ].iloc[0]
    return sorted(
        run_metadata.loc[
            run_metadata["true_graph"].eq(int(target["true_graph"]))
            & ~run_metadata["run_id"].eq(target_run_id),
            "run_id",
        ].astype(str)
    )


def build_pairwise_differences(
    window_table: pd.DataFrame,
    probability_summary: pd.DataFrame,
    operating_metrics: pd.DataFrame,
    factor_comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for target_run_id in [HARD_NORMAL_RUN, HARD_ATTACK_RUN]:
        target_metadata_rows = factor_comparison[
            factor_comparison["target_run_id"].eq(target_run_id)
        ]
        control_ids = target_metadata_rows[
            ~target_metadata_rows["candidate_run_id"].eq(target_run_id)
        ]["candidate_run_id"].tolist()

        for model_name in MODEL_ORDER:
            target_windows = window_table[
                window_table["model_name"].eq(model_name)
                & window_table["run_id"].eq(target_run_id)
            ].sort_values("end_epoch")
            target_probabilities = target_windows[
                "graph_probability"
            ].to_numpy(dtype=float)

            target_summary = probability_summary[
                probability_summary["model_name"].eq(model_name)
                & probability_summary["run_id"].eq(target_run_id)
            ].iloc[0]

            for control_run_id in control_ids:
                control_windows = window_table[
                    window_table["model_name"].eq(model_name)
                    & window_table["run_id"].eq(control_run_id)
                ].sort_values("end_epoch")
                control_probabilities = control_windows[
                    "graph_probability"
                ].to_numpy(dtype=float)

                control_summary = probability_summary[
                    probability_summary["model_name"].eq(model_name)
                    & probability_summary["run_id"].eq(control_run_id)
                ].iloc[0]

                factor_row = target_metadata_rows[
                    target_metadata_rows["candidate_run_id"].eq(control_run_id)
                ].iloc[0]

                temporal_left, temporal_right = shared_epoch_arrays(
                    target_windows,
                    control_windows,
                    "graph_probability",
                )
                temporal_corr = pearson_correlation(
                    temporal_left,
                    temporal_right,
                )

                base = {
                    "model_name": model_name,
                    "target_run_id": target_run_id,
                    "control_run_id": control_run_id,
                    "control_split": str(control_summary["split"]),
                    "target_true_graph": int(target_summary["true_graph"]),
                    "control_true_graph": int(control_summary["true_graph"]),
                    "factor_match_score": int(
                        factor_row["factor_match_score"]
                    ),
                    "factor_match_fraction": float(
                        factor_row["factor_match_fraction"]
                    ),
                    "target_mean_probability": float(
                        target_summary["mean_graph_probability"]
                    ),
                    "control_mean_probability": float(
                        control_summary["mean_graph_probability"]
                    ),
                    "mean_probability_difference_target_minus_control": float(
                        target_summary["mean_graph_probability"]
                        - control_summary["mean_graph_probability"]
                    ),
                    "target_median_probability": float(
                        target_summary["median_graph_probability"]
                    ),
                    "control_median_probability": float(
                        control_summary["median_graph_probability"]
                    ),
                    "median_probability_difference_target_minus_control": float(
                        target_summary["median_graph_probability"]
                        - control_summary["median_graph_probability"]
                    ),
                    "histogram_probability_overlap": histogram_overlap(
                        target_probabilities,
                        control_probabilities,
                    ),
                    "ks_statistic": ks_statistic(
                        target_probabilities,
                        control_probabilities,
                    ),
                    "epoch_aligned_probability_correlation": temporal_corr,
                    "same_profile": bool(factor_row["same_profile"]),
                    "same_active_core_count": bool(
                        factor_row["same_active_core_count"]
                    ),
                    "same_active_cores": bool(
                        factor_row["same_active_cores"]
                    ),
                    "same_attacker_count": bool(
                        factor_row["same_attacker_count"]
                    ),
                    "same_attackers": bool(
                        factor_row["same_attackers"]
                    ),
                    "same_strength": bool(
                        factor_row["same_strength"]
                    ),
                }

                for operating_point in [
                    "reference_0_50",
                    "validation_selected",
                ]:
                    target_metric = operating_metrics[
                        operating_metrics["model_name"].eq(model_name)
                        & operating_metrics["run_id"].eq(target_run_id)
                        & operating_metrics["operating_point"].eq(
                            operating_point
                        )
                    ].iloc[0]
                    control_metric = operating_metrics[
                        operating_metrics["model_name"].eq(model_name)
                        & operating_metrics["run_id"].eq(control_run_id)
                        & operating_metrics["operating_point"].eq(
                            operating_point
                        )
                    ].iloc[0]

                    rows.append(
                        {
                            **base,
                            "operating_point": operating_point,
                            "threshold": float(target_metric["threshold"]),
                            "target_error_rate": float(
                                target_metric["error_rate"]
                            ),
                            "control_error_rate": float(
                                control_metric["error_rate"]
                            ),
                            "error_rate_difference_target_minus_control": float(
                                target_metric["error_rate"]
                                - control_metric["error_rate"]
                            ),
                            "target_longest_error_streak": int(
                                target_metric["longest_error_streak"]
                            ),
                            "control_longest_error_streak": int(
                                control_metric["longest_error_streak"]
                            ),
                            "longest_streak_difference_target_minus_control": int(
                                target_metric["longest_error_streak"]
                                - control_metric["longest_error_streak"]
                            ),
                            "target_middle_80pct_error_rate": float(
                                target_metric["middle_80pct_error_rate"]
                            ),
                            "control_middle_80pct_error_rate": float(
                                control_metric["middle_80pct_error_rate"]
                            ),
                        }
                    )

    return pd.DataFrame(rows).sort_values(
        [
            "target_run_id",
            "model_name",
            "operating_point",
            "factor_match_score",
        ],
        ascending=[True, True, True, False],
    )


# ---------------------------------------------------------------------------
# Cross-model agreement and temporal correlations
# ---------------------------------------------------------------------------

def build_dominant_agreement(
    window_table: pd.DataFrame,
    threshold_map: dict[str, float],
) -> pd.DataFrame:
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
        "attackers",
        "strength",
        "seed",
    ]

    dominant = window_table[
        window_table["run_id"].isin(
            [HARD_NORMAL_RUN, HARD_ATTACK_RUN]
        )
    ].copy()

    metadata = (
        dominant[metadata_columns]
        .drop_duplicates(["sample_id"])
        .sort_values(["run_id", "end_epoch"])
        .reset_index(drop=True)
    )

    rows = []

    for operating_point in ["reference_0_50", "validation_selected"]:
        result = metadata.copy()
        result["operating_point"] = operating_point

        for model_name in MODEL_ORDER:
            model_rows = dominant[
                dominant["model_name"].eq(model_name)
            ].set_index("sample_id")
            probabilities = result["sample_id"].map(
                model_rows["graph_probability"]
            ).to_numpy(dtype=float)

            threshold = (
                REFERENCE_THRESHOLD
                if operating_point == "reference_0_50"
                else threshold_map[model_name]
            )
            predictions = (probabilities >= threshold).astype(np.int8)
            truth = result["true_graph"].to_numpy(dtype=np.int8)

            result[f"threshold_{model_name}"] = threshold
            result[f"probability_{model_name}"] = probabilities
            result[f"prediction_{model_name}"] = predictions
            result[f"correct_{model_name}"] = predictions == truth

        correct_columns = [f"correct_{name}" for name in MODEL_ORDER]
        prediction_columns = [f"prediction_{name}" for name in MODEL_ORDER]

        result["correct_model_count"] = (
            result[correct_columns].sum(axis=1).astype(int)
        )
        result["wrong_model_count"] = (
            len(MODEL_ORDER) - result["correct_model_count"]
        )
        result["all_models_correct"] = result["correct_model_count"].eq(4)
        result["all_models_wrong"] = result["wrong_model_count"].eq(4)
        result["all_predictions_equal"] = (
            result[prediction_columns].nunique(axis=1).eq(1)
        )
        result["prediction_disagreement"] = ~result["all_predictions_equal"]
        result["mean_model_probability"] = result[
            [f"probability_{name}" for name in MODEL_ORDER]
        ].mean(axis=1)
        result["std_model_probability"] = result[
            [f"probability_{name}" for name in MODEL_ORDER]
        ].std(axis=1, ddof=0)

        rows.append(result)

    return pd.concat(rows, ignore_index=True)


def build_temporal_correlations(
    window_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (split_name, run_id), run in window_table.groupby(
        ["split", "run_id"],
        sort=True,
    ):
        pivot = run.pivot(
            index="end_epoch",
            columns="model_name",
            values="graph_probability",
        ).sort_index()

        if not set(MODEL_ORDER).issubset(pivot.columns):
            raise RuntimeError(
                f"Missing model probabilities for run {run_id}."
            )

        for left_model, right_model in itertools.combinations(
            MODEL_ORDER, 2
        ):
            rows.append(
                {
                    "split": split_name,
                    "run_id": run_id,
                    "model_left": left_model,
                    "model_right": right_model,
                    "number_of_shared_epochs": int(len(pivot)),
                    "pearson_probability_correlation": pearson_correlation(
                        pivot[left_model].to_numpy(dtype=float),
                        pivot[right_model].to_numpy(dtype=float),
                    ),
                    "mean_absolute_probability_difference": float(
                        np.mean(
                            np.abs(
                                pivot[left_model].to_numpy(dtype=float)
                                - pivot[right_model].to_numpy(dtype=float)
                            )
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_probability_trajectories(
    window_table: pd.DataFrame,
    inventory: pd.DataFrame,
    threshold_map: dict[str, float],
    plot_directory: Path,
) -> None:
    trajectory_directory = plot_directory / "probability_trajectories"

    ordered_runs = inventory.sort_values(
        ["true_graph", "factor_match_score", "split", "run_id"],
        ascending=[True, False, True, True],
    )["run_id"].tolist()

    for run_id in ordered_runs:
        run = window_table[window_table["run_id"].eq(run_id)]
        if run.empty:
            continue

        figure, axis = plt.subplots(figsize=(12, 5))

        for model_name in MODEL_ORDER:
            model_run = run[
                run["model_name"].eq(model_name)
            ].sort_values("end_epoch")
            axis.plot(
                model_run["end_epoch"],
                model_run["graph_probability"],
                label=MODEL_LABELS[model_name],
                linewidth=1.0,
            )

        axis.axhline(
            REFERENCE_THRESHOLD,
            linestyle="--",
            label="Reference threshold 0.50",
        )
        axis.set_xlabel("End epoch")
        axis.set_ylabel("Graph attack probability")
        axis.set_ylim(-0.02, 1.02)
        true_graph = int(run["true_graph"].iloc[0])
        split_name = str(run["split"].iloc[0])
        axis.set_title(
            f"Probability trajectory: {run_id} "
            f"(split={split_name}, true_graph={true_graph})"
        )
        axis.legend(fontsize=8, ncol=2)

        save_figure(
            figure,
            trajectory_directory / f"{run_id}.png",
        )


def plot_target_control_distributions(
    window_table: pd.DataFrame,
    target_run_id: str,
    control_run_ids: list[str],
    prefix: str,
    plot_directory: Path,
) -> None:
    selected_runs = [target_run_id, *control_run_ids]

    for model_name in MODEL_ORDER:
        figure, axis = plt.subplots(figsize=(10, 6))

        for run_id in selected_runs:
            values = window_table.loc[
                window_table["model_name"].eq(model_name)
                & window_table["run_id"].eq(run_id),
                "graph_probability",
            ].to_numpy(dtype=float)

            axis.hist(
                values,
                bins=np.linspace(0.0, 1.0, 51),
                density=True,
                histtype="step",
                linewidth=1.4,
                label=run_id,
            )

        axis.set_xlabel("Graph attack probability")
        axis.set_ylabel("Density")
        axis.set_title(
            f"{prefix} probability distributions: "
            f"{MODEL_LABELS[model_name]}"
        )
        axis.legend(fontsize=7)

        save_figure(
            figure,
            plot_directory
            / f"{prefix}_probability_distribution_{model_name}.png",
        )


def plot_error_timeline(
    agreement: pd.DataFrame,
    run_id: str,
    operating_point: str,
    plot_directory: Path,
) -> None:
    data = agreement[
        agreement["run_id"].eq(run_id)
        & agreement["operating_point"].eq(operating_point)
    ].sort_values("end_epoch")

    matrix = np.vstack(
        [
            (~data[f"correct_{model_name}"]).astype(int).to_numpy()
            for model_name in MODEL_ORDER
        ]
    )

    figure, axis = plt.subplots(figsize=(14, 3.5))
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        vmin=0,
        vmax=1,
    )
    axis.set_yticks(np.arange(len(MODEL_ORDER)))
    axis.set_yticklabels([MODEL_LABELS[name] for name in MODEL_ORDER])
    axis.set_xlabel("Window order within run")
    axis.set_title(
        f"Error-state timeline: {run_id} ({operating_point})"
    )
    figure.colorbar(image, ax=axis, label="Error state (1 = wrong)")

    save_figure(
        figure,
        plot_directory
        / f"error_timeline_{operating_point}_{run_id}.png",
    )


def plot_disagreement_timeline(
    agreement: pd.DataFrame,
    run_id: str,
    plot_directory: Path,
) -> None:
    data = agreement[
        agreement["run_id"].eq(run_id)
        & agreement["operating_point"].eq("reference_0_50")
    ].sort_values("end_epoch")

    figure, axis = plt.subplots(figsize=(13, 4))
    axis.plot(
        data["end_epoch"],
        data["wrong_model_count"],
        linewidth=1.0,
    )
    axis.set_xlabel("End epoch")
    axis.set_ylabel("Number of wrong models")
    axis.set_ylim(-0.2, 4.2)
    axis.set_title(
        f"Cross-model disagreement timeline at threshold 0.50: {run_id}"
    )

    save_figure(
        figure,
        plot_directory / f"model_disagreement_timeline_{run_id}.png",
    )


def plot_correlation_matrix(
    window_table: pd.DataFrame,
    run_id: str,
    plot_directory: Path,
) -> None:
    data = window_table[
        window_table["run_id"].eq(run_id)
    ].pivot(
        index="end_epoch",
        columns="model_name",
        values="graph_probability",
    ).reindex(columns=MODEL_ORDER)

    matrix = data.corr().to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(
        matrix,
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(MODEL_ORDER)))
    axis.set_yticks(np.arange(len(MODEL_ORDER)))
    axis.set_xticklabels(
        [MODEL_LABELS[name] for name in MODEL_ORDER],
        rotation=30,
        ha="right",
    )
    axis.set_yticklabels([MODEL_LABELS[name] for name in MODEL_ORDER])

    for row in range(len(MODEL_ORDER)):
        for column in range(len(MODEL_ORDER)):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
            )

    axis.set_title(f"Cross-model probability correlation: {run_id}")
    figure.colorbar(image, ax=axis, label="Pearson correlation")

    save_figure(
        figure,
        plot_directory / f"probability_correlation_{run_id}.png",
    )


def plot_run_heatmap(
    operating_metrics: pd.DataFrame,
    operating_point: str,
    plot_directory: Path,
) -> None:
    data = operating_metrics[
        operating_metrics["operating_point"].eq(operating_point)
    ].copy()

    pivot = data.pivot(
        index="run_id",
        columns="model_name",
        values="error_rate",
    ).reindex(columns=MODEL_ORDER)

    ordering = (
        data[["run_id", "true_graph", "split"]]
        .drop_duplicates()
        .sort_values(["true_graph", "split", "run_id"])
    )
    pivot = pivot.reindex(ordering["run_id"].tolist())

    figure, axis = plt.subplots(
        figsize=(8, max(7, len(pivot) * 0.32))
    )
    image = axis.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    axis.set_xticks(np.arange(len(MODEL_ORDER)))
    axis.set_xticklabels(
        [MODEL_LABELS[name] for name in MODEL_ORDER],
        rotation=25,
        ha="right",
    )
    axis.set_yticks(np.arange(len(pivot)))
    axis.set_yticklabels(pivot.index, fontsize=7)
    axis.set_title(f"Run-level error-rate heatmap: {operating_point}")
    figure.colorbar(image, ax=axis, label="Run error rate")

    save_figure(
        figure,
        plot_directory / f"run_error_heatmap_{operating_point}.png",
    )


def plot_quantile_comparison(
    probability_quantiles: pd.DataFrame,
    target_run_id: str,
    control_run_ids: list[str],
    prefix: str,
    plot_directory: Path,
) -> None:
    runs = [target_run_id, *control_run_ids]
    quantile_columns = list(QUANTILES)

    for model_name in MODEL_ORDER:
        data = probability_quantiles[
            probability_quantiles["model_name"].eq(model_name)
            & probability_quantiles["run_id"].isin(runs)
        ].set_index("run_id").reindex(runs)

        figure, axis = plt.subplots(figsize=(10, 6))
        for run_id in runs:
            axis.plot(
                list(QUANTILES.values()),
                data.loc[run_id, quantile_columns].to_numpy(dtype=float),
                marker="o",
                label=run_id,
            )

        axis.set_xlabel("Quantile")
        axis.set_ylabel("Graph attack probability")
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(
            f"{prefix} probability quantiles: {MODEL_LABELS[model_name]}"
        )
        axis.legend(fontsize=7)

        save_figure(
            figure,
            plot_directory
            / f"{prefix}_quantiles_{model_name}.png",
        )


def plot_threshold_crossings(
    crossings: pd.DataFrame,
    inventory: pd.DataFrame,
    plot_directory: Path,
) -> None:
    selected_runs = inventory[
        inventory["roles"].str.contains(
            "target|primary_test_control|broader_test_control",
            regex=True,
        )
    ]["run_id"].tolist()

    data = crossings[
        crossings["run_id"].isin(selected_runs)
        & crossings["operating_point"].eq("reference_0_50")
    ].copy()

    pivot = data.pivot(
        index="run_id",
        columns="model_name",
        values="threshold_crossing_count",
    ).reindex(columns=MODEL_ORDER)

    figure, axis = plt.subplots(
        figsize=(11, max(6, len(pivot) * 0.55))
    )
    pivot.plot(kind="barh", ax=axis)
    axis.set_xlabel("Number of threshold crossings")
    axis.set_ylabel("Run")
    axis.set_title("Threshold-crossing comparison at threshold 0.50")
    axis.legend([MODEL_LABELS[name] for name in pivot.columns], fontsize=8)

    save_figure(
        figure,
        plot_directory / "threshold_crossing_comparison.png",
    )


def make_plots(
    window_table: pd.DataFrame,
    inventory: pd.DataFrame,
    probability_quantiles: pd.DataFrame,
    operating_metrics: pd.DataFrame,
    crossings: pd.DataFrame,
    agreement: pd.DataFrame,
    threshold_map: dict[str, float],
    plot_directory: Path,
) -> None:
    plot_directory.mkdir(parents=True, exist_ok=True)

    plot_probability_trajectories(
        window_table,
        inventory,
        threshold_map,
        plot_directory,
    )

    plot_target_control_distributions(
        window_table,
        HARD_NORMAL_RUN,
        [PRIMARY_NORMAL_CONTROL],
        "hard_normal_vs_primary_control",
        plot_directory,
    )
    plot_target_control_distributions(
        window_table,
        HARD_ATTACK_RUN,
        PRIMARY_ATTACK_CONTROLS,
        "hard_attack_vs_primary_controls",
        plot_directory,
    )

    plot_quantile_comparison(
        probability_quantiles,
        HARD_NORMAL_RUN,
        [PRIMARY_NORMAL_CONTROL],
        "hard_normal_vs_primary_control",
        plot_directory,
    )
    plot_quantile_comparison(
        probability_quantiles,
        HARD_ATTACK_RUN,
        PRIMARY_ATTACK_CONTROLS,
        "hard_attack_vs_primary_controls",
        plot_directory,
    )

    for run_id in [HARD_NORMAL_RUN, HARD_ATTACK_RUN]:
        for operating_point in [
            "reference_0_50",
            "validation_selected",
        ]:
            plot_error_timeline(
                agreement,
                run_id,
                operating_point,
                plot_directory,
            )

        plot_disagreement_timeline(
            agreement,
            run_id,
            plot_directory,
        )
        plot_correlation_matrix(
            window_table,
            run_id,
            plot_directory,
        )

    for operating_point in [
        "reference_0_50",
        "validation_selected",
    ]:
        plot_run_heatmap(
            operating_metrics,
            operating_point,
            plot_directory,
        )

    plot_threshold_crossings(
        crossings,
        inventory,
        plot_directory,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def primary_pair_lines(
    probability_summary: pd.DataFrame,
    operating_metrics: pd.DataFrame,
    target_run_id: str,
    control_run_ids: list[str],
) -> list[str]:
    lines = []

    for model_name in MODEL_ORDER:
        target_probability = probability_summary[
            probability_summary["model_name"].eq(model_name)
            & probability_summary["run_id"].eq(target_run_id)
        ].iloc[0]
        target_metric = operating_metrics[
            operating_metrics["model_name"].eq(model_name)
            & operating_metrics["run_id"].eq(target_run_id)
            & operating_metrics["operating_point"].eq("reference_0_50")
        ].iloc[0]

        lines.append(f"### {MODEL_LABELS[model_name]}")
        lines.append(
            f"- Target `{target_run_id}`: median probability "
            f"{float(target_probability['median_graph_probability']):.4f}, "
            f"error rate {float(target_metric['error_rate']):.2%}, "
            f"longest error streak {int(target_metric['longest_error_streak'])}."
        )

        for control_run_id in control_run_ids:
            control_probability = probability_summary[
                probability_summary["model_name"].eq(model_name)
                & probability_summary["run_id"].eq(control_run_id)
            ].iloc[0]
            control_metric = operating_metrics[
                operating_metrics["model_name"].eq(model_name)
                & operating_metrics["run_id"].eq(control_run_id)
                & operating_metrics["operating_point"].eq("reference_0_50")
            ].iloc[0]

            lines.append(
                f"- Control `{control_run_id}`: median probability "
                f"{float(control_probability['median_graph_probability']):.4f}, "
                f"error rate {float(control_metric['error_rate']):.2%}, "
                f"longest error streak "
                f"{int(control_metric['longest_error_streak'])}."
            )
        lines.append("")

    return lines


def best_stage7_controls(
    factor_comparison: pd.DataFrame,
    pairwise: pd.DataFrame,
    target_run_id: str,
    limit: int = 3,
) -> list[str]:
    factor = factor_comparison[
        factor_comparison["target_run_id"].eq(target_run_id)
        & ~factor_comparison["candidate_run_id"].eq(target_run_id)
    ].copy()

    pair = pairwise[
        pairwise["target_run_id"].eq(target_run_id)
        & pairwise["operating_point"].eq("reference_0_50")
    ].groupby("control_run_id").agg(
        mean_absolute_error_gap=(
            "error_rate_difference_target_minus_control",
            lambda values: float(np.mean(np.abs(values))),
        ),
        mean_distribution_overlap=(
            "histogram_probability_overlap",
            "mean",
        ),
    ).reset_index()

    merged = factor.merge(
        pair,
        left_on="candidate_run_id",
        right_on="control_run_id",
        how="left",
    )

    merged = merged.sort_values(
        [
            "factor_match_score",
            "mean_distribution_overlap",
            "mean_absolute_error_gap",
        ],
        ascending=[False, False, False],
    )

    return merged["candidate_run_id"].head(limit).tolist()


def build_report(
    inventory: pd.DataFrame,
    probability_summary: pd.DataFrame,
    operating_metrics: pd.DataFrame,
    streak_summary: pd.DataFrame,
    crossings: pd.DataFrame,
    pairwise: pd.DataFrame,
    agreement: pd.DataFrame,
    correlations: pd.DataFrame,
    factor_comparison: pd.DataFrame,
) -> str:
    lines = [
        "# V3 Matched Dominant-Run Analysis",
        "",
        "This stage compares run metadata, model probabilities, temporal behaviour, "
        "threshold crossings, error persistence and cross-model agreement. It does "
        "not load the original 24-feature tensor and does not rerun inference.",
        "",
        "Only validation and test probabilities are available because Stage 3 did "
        "not export training predictions. Training-run feature analogues can be "
        "examined in Stage 7 using the original dataset.",
        "",
        "## Exact and partial controls",
        "",
        f"- Hard normal target: `{HARD_NORMAL_RUN}`.",
        f"- Exact normal test control by class/profile/core count: "
        f"`{PRIMARY_NORMAL_CONTROL}`.",
        f"- Hard attack target: `{HARD_ATTACK_RUN}`.",
        "- Exact attack test controls by attacker count, attacker router and strength: "
        + ", ".join(f"`{run_id}`" for run_id in PRIMARY_ATTACK_CONTROLS)
        + ".",
        "- All remaining same-class validation and test runs are partial controls "
        "ranked by factor-match score.",
        "",
        "## Hard normal versus primary control",
        "",
    ]

    lines.extend(
        primary_pair_lines(
            probability_summary,
            operating_metrics,
            HARD_NORMAL_RUN,
            [PRIMARY_NORMAL_CONTROL],
        )
    )

    lines.extend(
        [
            "## Hard attack versus primary controls",
            "",
        ]
    )
    lines.extend(
        primary_pair_lines(
            probability_summary,
            operating_metrics,
            HARD_ATTACK_RUN,
            PRIMARY_ATTACK_CONTROLS,
        )
    )

    hard_normal_metrics = operating_metrics[
        operating_metrics["run_id"].eq(HARD_NORMAL_RUN)
        & operating_metrics["operating_point"].eq("reference_0_50")
    ]
    primary_normal_metrics = operating_metrics[
        operating_metrics["run_id"].eq(PRIMARY_NORMAL_CONTROL)
        & operating_metrics["operating_point"].eq("reference_0_50")
    ]

    hard_attack_metrics = operating_metrics[
        operating_metrics["run_id"].eq(HARD_ATTACK_RUN)
        & operating_metrics["operating_point"].eq("reference_0_50")
    ]

    normal_target_mean_error = float(hard_normal_metrics["error_rate"].mean())
    normal_control_mean_error = float(
        primary_normal_metrics["error_rate"].mean()
    )
    attack_target_mean_error = float(hard_attack_metrics["error_rate"].mean())

    normal_target_middle = float(
        hard_normal_metrics["middle_80pct_error_rate"].mean()
    )
    attack_target_middle = float(
        hard_attack_metrics["middle_80pct_error_rate"].mean()
    )

    dominant_reference = agreement[
        agreement["operating_point"].eq("reference_0_50")
    ]
    normal_agreement = dominant_reference[
        dominant_reference["run_id"].eq(HARD_NORMAL_RUN)
    ]
    attack_agreement = dominant_reference[
        dominant_reference["run_id"].eq(HARD_ATTACK_RUN)
    ]

    normal_all_wrong = float(normal_agreement["all_models_wrong"].mean())
    attack_all_wrong = float(attack_agreement["all_models_wrong"].mean())
    normal_disagreement = float(
        normal_agreement["prediction_disagreement"].mean()
    )
    attack_disagreement = float(
        attack_agreement["prediction_disagreement"].mean()
    )

    normal_controls = best_stage7_controls(
        factor_comparison,
        pairwise,
        HARD_NORMAL_RUN,
    )
    attack_controls = best_stage7_controls(
        factor_comparison,
        pairwise,
        HARD_ATTACK_RUN,
    )

    lines.extend(
        [
            "## Direct Stage 6 findings",
            "",
            f"- The hard normal run has a mean four-model error rate of "
            f"**{normal_target_mean_error:.1%}** at threshold 0.50, versus "
            f"**{normal_control_mean_error:.1%}** for the exact mixed/four-core "
            "test control.",
            f"- Its middle-80% mean error rate is **{normal_target_middle:.1%}**, "
            "so the failure is not confined to run startup or shutdown.",
            f"- The hard attack has a mean four-model error rate of "
            f"**{attack_target_mean_error:.1%}** and a middle-80% mean error rate "
            f"of **{attack_target_middle:.1%}**.",
            f"- All four models are simultaneously wrong on "
            f"**{normal_all_wrong:.1%}** of hard-normal windows and "
            f"**{attack_all_wrong:.1%}** of hard-attack windows at threshold 0.50.",
            f"- Model predictions disagree on **{normal_disagreement:.1%}** of "
            f"hard-normal windows and **{attack_disagreement:.1%}** of hard-attack "
            "windows.",
            "",
            "## What Stage 6 can establish",
            "",
            "- Whether the target run is uniquely difficult relative to same-class controls.",
            "- Whether the failure persists through steady-state epochs.",
            "- Whether models fail at the same temporal locations.",
            "- Whether profile, core count, exact placement, attacker count, attacker "
            "location or strength are sufficient metadata explanations.",
            "- Which matched controls should be carried into Stage 7.",
            "",
            "## What remains unproven until Stage 7",
            "",
            "- Whether directional IFD saturation causes the overlap.",
            "- Whether benign congestion and weak attacks are inseparable in the "
            "24-feature space.",
            "- Whether normalization or clipping destroys magnitude information.",
            "- Which routers and directions produce the conflicting signals.",
            "- Whether pooling suppresses a weak temporal feature that is present in "
            "the raw input.",
            "",
            "## Recommended Stage 7 comparisons",
            "",
            "### Hard normal controls",
        ]
    )

    for run_id in normal_controls:
        lines.append(f"- `{run_id}`")

    lines.extend(["", "### Hard attack controls"])
    for run_id in attack_controls:
        lines.append(f"- `{run_id}`")

    lines.extend(
        [
            "",
            "These controls are ranked using metadata match quality plus probability-"
            "distribution behaviour. Stage 7 should still inspect all exact controls, "
            "not only the top-ranked entries.",
            "",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 5 reproduction check and final validation
# ---------------------------------------------------------------------------

def validate_against_stage5(
    operating_metrics: pd.DataFrame,
    stage5_results: pd.DataFrame,
) -> tuple[bool, list[str]]:
    messages = []
    matched = True

    for model_name in MODEL_ORDER:
        for split_name in ["val", "test"]:
            for operating_point in [
                "reference_0_50",
                "validation_selected",
            ]:
                run_rows = operating_metrics[
                    operating_metrics["model_name"].eq(model_name)
                    & operating_metrics["split"].eq(split_name)
                    & operating_metrics["operating_point"].eq(
                        operating_point
                    )
                ]

                counts = {
                    name: int(run_rows[name].sum())
                    for name in ["TN", "FP", "FN", "TP"]
                }
                metrics = metrics_from_counts(counts)

                expected = stage5_results[
                    stage5_results["model_name"].eq(model_name)
                    & stage5_results["split"].eq(split_name)
                    & stage5_results["operating_point"].eq(
                        operating_point
                    )
                ]

                if len(expected) != 1:
                    matched = False
                    messages.append(
                        f"Missing Stage 5 row: {model_name} {split_name} "
                        f"{operating_point}"
                    )
                    continue

                expected_row = expected.iloc[0]

                for name in ["TN", "FP", "FN", "TP"]:
                    if counts[name] != int(expected_row[name]):
                        matched = False
                        messages.append(
                            f"{model_name} {split_name} {operating_point} "
                            f"{name}: {counts[name]} != {int(expected_row[name])}"
                        )

                for calculated_name, expected_name in {
                    "F1": "F1",
                    "FPR": "FPR",
                    "FNR": "FNR",
                }.items():
                    if not np.isclose(
                        metrics[calculated_name],
                        float(expected_row[expected_name]),
                        atol=1e-12,
                        rtol=0.0,
                    ):
                        matched = False
                        messages.append(
                            f"{model_name} {split_name} {operating_point} "
                            f"{calculated_name}: {metrics[calculated_name]} != "
                            f"{float(expected_row[expected_name])}"
                        )

    return matched, messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare dominant V3 graph-failure runs against matched validation "
            "and test controls using exported probabilities only."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("reports/v3_failure_analysis"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    prediction_directory = root / "predictions"
    table_directory = root / "tables"
    log_directory = root / "logs"
    plot_directory = root / "plots" / "matched_runs"

    metadata_path = prediction_directory / "prediction_metadata.csv"
    selected_thresholds_path = (
        table_directory / "selected_graph_thresholds.csv"
    )
    stage5_results_path = (
        table_directory / "graph_threshold_transfer_results.csv"
    )

    output_paths = {
        "inventory": table_directory / "matched_run_inventory.csv",
        "probability_summary": table_directory / "matched_run_probability_summary.csv",
        "operating_metrics": table_directory / "matched_run_operating_point_metrics.csv",
        "probability_quantiles": table_directory / "matched_run_probability_quantiles.csv",
        "error_streak_summary": table_directory / "matched_run_error_streak_summary.csv",
        "threshold_crossings": table_directory / "matched_run_threshold_crossings.csv",
        "pairwise_differences": table_directory / "matched_run_pairwise_differences.csv",
        "model_agreement": table_directory / "matched_run_model_agreement.csv",
        "temporal_correlations": table_directory / "matched_run_temporal_correlations.csv",
        "factor_comparison": table_directory / "matched_run_factor_comparison.csv",
        "report": root / "MATCHED_RUN_ANALYSIS.md",
        "validation": log_directory / "32_matched_run_validation.txt",
    }

    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Stage 6 outputs already exist. Use --overwrite to replace them:\n"
            + "\n".join(str(path) for path in existing)
        )

    for required_path in [
        metadata_path,
        selected_thresholds_path,
        stage5_results_path,
    ]:
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    print(f"root: {root}", flush=True)
    print(f"metadata: {metadata_path}", flush=True)
    print(f"selected thresholds: {selected_thresholds_path}", flush=True)
    print(f"Stage 5 results: {stage5_results_path}", flush=True)
    print("feature tensor loaded: False", flush=True)
    print("inference rerun: False", flush=True)
    print("training run: False", flush=True)

    (
        metadata,
        loaded,
        selected_thresholds,
        stage5_results,
    ) = load_inputs(
        prediction_directory,
        metadata_path,
        selected_thresholds_path,
        stage5_results_path,
    )

    verify_alignment(metadata, loaded)
    threshold_map = selected_threshold_map(selected_thresholds)
    print("input alignment verified", flush=True)
    print(f"selected thresholds: {threshold_map}", flush=True)

    window_table = build_window_table(metadata, loaded)
    run_metadata = build_run_metadata(window_table)
    inventory, factor_comparison = build_inventory(run_metadata)

    print(
        f"available independent runs: {run_metadata['run_id'].nunique()} "
        f"(validation/test only)",
        flush=True,
    )
    print(
        "training prediction comparisons available: False",
        flush=True,
    )

    probability_summary, probability_quantiles = summarize_probabilities(
        window_table
    )
    (
        operating_metrics,
        raw_streaks,
        threshold_crossings,
    ) = build_operating_point_tables(
        window_table,
        threshold_map,
    )
    error_streak_summary = build_streak_summary(
        raw_streaks,
        operating_metrics,
    )
    pairwise_differences = build_pairwise_differences(
        window_table,
        probability_summary,
        operating_metrics,
        factor_comparison,
    )
    model_agreement = build_dominant_agreement(
        window_table,
        threshold_map,
    )
    temporal_correlations = build_temporal_correlations(window_table)

    stage5_matched, validation_messages = validate_against_stage5(
        operating_metrics,
        stage5_results,
    )

    all_runs_inventory = (
        inventory["run_id"].nunique()
        == run_metadata["run_id"].nunique()
    )
    required_runs_present = {
        HARD_NORMAL_RUN,
        PRIMARY_NORMAL_CONTROL,
        *BROADER_NORMAL_CONTROLS,
        HARD_ATTACK_RUN,
        *PRIMARY_ATTACK_CONTROLS,
    }.issubset(set(inventory["run_id"]))

    exact_controls_present = bool(
        inventory["run_id"].isin(
            [PRIMARY_NORMAL_CONTROL, *PRIMARY_ATTACK_CONTROLS]
        ).sum()
        == 3
    )

    expected_probability_rows = (
        len(MODEL_ORDER) * run_metadata["run_id"].nunique()
    )
    probability_rows_complete = (
        len(probability_summary) == expected_probability_rows
        and len(probability_quantiles) == expected_probability_rows
    )

    expected_operating_rows = expected_probability_rows * 2
    operating_rows_complete = (
        len(operating_metrics) == expected_operating_rows
        and len(error_streak_summary) == expected_operating_rows
        and len(threshold_crossings) == expected_operating_rows
    )

    dominant_agreement_complete = (
        len(model_agreement)
        == 2
        * metadata["run_id"].isin(
            [HARD_NORMAL_RUN, HARD_ATTACK_RUN]
        ).sum()
    )

    unique_keys = all(
        [
            not probability_summary.duplicated(
                ["model_name", "split", "run_id"]
            ).any(),
            not operating_metrics.duplicated(
                ["model_name", "split", "run_id", "operating_point"]
            ).any(),
            not threshold_crossings.duplicated(
                ["model_name", "split", "run_id", "operating_point"]
            ).any(),
            not model_agreement.duplicated(
                ["sample_id", "operating_point"]
            ).any(),
        ]
    )

    finite_columns = [
        "mean_graph_probability",
        "median_graph_probability",
        "std_graph_probability",
        "minimum_graph_probability",
        "maximum_graph_probability",
        *QUANTILES.keys(),
    ]
    all_statistics_finite = bool(
        np.isfinite(
            probability_summary[finite_columns].to_numpy(dtype=float)
        ).all()
        and np.isfinite(
            operating_metrics[
                [
                    "threshold",
                    "error_rate",
                    "fraction_of_run_in_error",
                    "early_10pct_error_rate",
                    "middle_80pct_error_rate",
                    "late_10pct_error_rate",
                ]
            ].to_numpy(dtype=float)
        ).all()
        and np.isfinite(
            threshold_crossings[
                [
                    "threshold",
                    "threshold_crossing_count",
                    "mean_duration_above_threshold",
                    "mean_duration_below_threshold",
                    "fraction_above_threshold",
                    "fraction_below_threshold",
                ]
            ].to_numpy(dtype=float)
        ).all()
    )

    overall_success = all(
        [
            stage5_matched,
            all_runs_inventory,
            required_runs_present,
            exact_controls_present,
            probability_rows_complete,
            operating_rows_complete,
            dominant_agreement_complete,
            unique_keys,
            all_statistics_finite,
        ]
    )

    validation_lines = [
        "input alignment verified: True",
        f"Stage 5 operating points reproduced: {stage5_matched}",
        f"all available validation/test runs inventoried: {all_runs_inventory}",
        f"all required target/control runs present: {required_runs_present}",
        f"exact primary controls present: {exact_controls_present}",
        f"probability summaries complete: {probability_rows_complete}",
        f"operating-point summaries complete: {operating_rows_complete}",
        f"dominant-run model agreement complete: {dominant_agreement_complete}",
        f"all table keys unique: {unique_keys}",
        f"all statistics finite: {all_statistics_finite}",
        "feature tensor loaded: False",
        "inference rerun: False",
        "training performed: False",
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
            f"Stage 6 validation failed. Inspect {output_paths['validation']}."
        )

    atomic_csv(inventory, output_paths["inventory"])
    atomic_csv(
        probability_summary,
        output_paths["probability_summary"],
    )
    atomic_csv(
        operating_metrics,
        output_paths["operating_metrics"],
    )
    atomic_csv(
        probability_quantiles,
        output_paths["probability_quantiles"],
    )
    atomic_csv(
        error_streak_summary,
        output_paths["error_streak_summary"],
    )
    atomic_csv(
        threshold_crossings,
        output_paths["threshold_crossings"],
    )
    atomic_csv(
        pairwise_differences,
        output_paths["pairwise_differences"],
    )
    atomic_csv(
        model_agreement,
        output_paths["model_agreement"],
    )
    atomic_csv(
        temporal_correlations,
        output_paths["temporal_correlations"],
    )
    atomic_csv(
        factor_comparison,
        output_paths["factor_comparison"],
    )

    report = build_report(
        inventory,
        probability_summary,
        operating_metrics,
        error_streak_summary,
        threshold_crossings,
        pairwise_differences,
        model_agreement,
        temporal_correlations,
        factor_comparison,
    )
    atomic_text(report, output_paths["report"])

    make_plots(
        window_table,
        inventory,
        probability_quantiles,
        operating_metrics,
        threshold_crossings,
        model_agreement,
        threshold_map,
        plot_directory,
    )

    print("Stage 6 output files:", flush=True)
    for name, path in output_paths.items():
        print(f"  {name}: {path}", flush=True)
    print(f"  plots: {plot_directory}", flush=True)
    print("overall success: True", flush=True)


if __name__ == "__main__":
    main()
