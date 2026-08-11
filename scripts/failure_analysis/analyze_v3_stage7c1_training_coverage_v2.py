#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-12
SCRIPT_VERSION = "stage7c1_v2"
HIST_BINS = 1001
EXPECTED_FEATURES = 24
EXPECTED_ROUTERS = 16
EXPECTED_TEMPORAL = 8
EXPECTED_TOTAL_SAMPLES = 233803
EXPECTED_TRAIN_SAMPLES = 148185
EXPECTED_TRAIN_RUNS = 45
EXPECTED_WINDOWS_PER_RUN = 3293

HARD_NORMAL = "N-3-7-8-12-Pmixed-R18-V3"
HARD_ATTACK = "N-5-10-Pbursty-R51-A-12-S20-V3"
SUSPICIOUS_ROUTER = 12
SUSPICIOUS_FEATURES = {15, 18, 20, 23}

R1_METRICS = [
    "mean",
    "std",
    "median",
    "p05",
    "p95",
    "fraction_equal_0",
    "fraction_equal_1",
    "fraction_above_0_95",
    "mean_temporal_variance",
    "mean_absolute_adjacent_temporal_change",
    "mean_spatial_variance",
]

R3_METRICS = [
    "mean",
    "fraction_equal_1",
    "fraction_above_0_95",
    "mean_temporal_variance",
    "mean_temporal_excursion",
]

R4_METRICS = [
    "mean",
    "fraction_equal_1",
    "fraction_above_0_95",
    "mean_spatial_variance",
    "mean_absolute_change_from_previous_position",
]

R5_METRICS = [
    "mean_router_std",
    "mean_maximum_minus_median",
    "mean_top_router_share",
    "mean_top_two_router_share",
]

PRIMARY_COHORTS: dict[str, int | None] = {
    "all_training": None,
    "normal_training": 0,
    "attack_training": 1,
}


# -----------------------------------------------------------------------------
# Safe output helpers
# -----------------------------------------------------------------------------

def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(text)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(temp, index=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_json(data: dict[str, Any], path: Path) -> None:
    atomic_text(json.dumps(data, indent=2, sort_keys=True) + "\n", path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez_compressed(temp, **arrays)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        fig.savefig(temp, dpi=180, bbox_inches="tight")
        os.replace(temp, path)
    finally:
        plt.close(fig)
        temp.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# Metadata and dimension construction
# -----------------------------------------------------------------------------

def strength_group(true_graph: int, strength: int) -> str:
    if true_graph == 0 or strength <= 0:
        return "normal"
    if strength <= 20:
        return "weak_1_20"
    if strength <= 63:
        return "medium_21_63"
    return "strong_64_plus"


def build_metadata_lookup(run_inventory: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in run_inventory.iterrows():
        run_id = str(row["run_id"])
        true_graph = int(row["true_graph"])
        strength = int(row["strength_from_name"])
        lookup[run_id] = {
            "run_id": run_id,
            "split": str(row["split"]),
            "true_graph": true_graph,
            "profile": str(row["profile_from_name"]),
            "active_cores": str(row["active_cores_from_name"]),
            "active_core_count": int(row["active_core_count_from_name"]),
            "attackers": str(row["attackers_from_name"]),
            "attacker_count": int(row["attacker_count_from_name"]),
            "strength": strength,
            "strength_group": strength_group(true_graph, strength),
            "seed": int(row["seed_from_name"]),
            "number_of_windows": int(row["number_of_windows"]),
        }
    return lookup


def build_feature_lookup(taxonomy: pd.DataFrame) -> dict[int, dict[str, Any]]:
    return {
        int(row["feature_index"]): {
            "feature_index": int(row["feature_index"]),
            "feature_name": str(row["feature_name"]),
            "feature_family": str(row["feature_family"]),
            "measurement_type": str(row["measurement_type"]),
            "flow_side": str(row["flow_side"]),
            "direction": str(row["direction"]),
        }
        for _, row in taxonomy.iterrows()
    }


def dimension_id(
    representation: str,
    metric: str,
    feature_index: int = -1,
    feature_family: str = "",
    router_id: int = -1,
    temporal_position: int = -1,
) -> str:
    parts = [representation]
    if representation == "R2_feature_family":
        parts.append(f"family={feature_family}")
    elif feature_index >= 0:
        parts.append(f"feature={feature_index}")
    if router_id >= 0:
        parts.append(f"router={router_id}")
    if temporal_position >= 0:
        parts.append(f"position={temporal_position}")
    parts.append(f"metric={metric}")
    return "|".join(parts)


def long_row(
    metadata: dict[str, Any],
    representation: str,
    metric: str,
    value: float,
    provenance_table: str,
    feature: dict[str, Any] | None = None,
    feature_family: str = "",
    router_id: int = -1,
    temporal_position: int = -1,
    metadata_dependent: bool = False,
) -> dict[str, Any]:
    if feature is None:
        feature = {
            "feature_index": -1,
            "feature_name": "",
            "feature_family": feature_family,
            "measurement_type": "",
            "flow_side": "",
            "direction": "",
        }

    effective_family = feature_family or str(feature["feature_family"])
    feature_index = int(feature["feature_index"])
    return {
        **metadata,
        "representation": representation,
        "dimension_id": dimension_id(
            representation=representation,
            metric=metric,
            feature_index=feature_index,
            feature_family=effective_family,
            router_id=router_id,
            temporal_position=temporal_position,
        ),
        "metric": metric,
        "value": float(value),
        "feature_index": feature_index,
        "feature_name": str(feature["feature_name"]),
        "feature_family": effective_family,
        "measurement_type": str(feature["measurement_type"]),
        "flow_side": str(feature["flow_side"]),
        "direction": str(feature["direction"]),
        "router_id": int(router_id),
        "temporal_position": int(temporal_position),
        "metadata_dependent": bool(metadata_dependent),
        "provenance_table": provenance_table,
    }


def build_r1_r2_dimensions(
    run_summary: pd.DataFrame,
    metadata_lookup: dict[str, dict[str, Any]],
    feature_lookup: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id, run_frame in run_summary.groupby("run_id", sort=True):
        metadata = metadata_lookup[str(run_id)]
        for _, row in run_frame.iterrows():
            feature = feature_lookup[int(row["feature_index"])]
            for metric in R1_METRICS:
                rows.append(
                    long_row(
                        metadata=metadata,
                        representation="R1_global_run_feature",
                        metric=metric,
                        value=float(row[metric]),
                        provenance_table="stage7b_run_feature_summary.csv",
                        feature=feature,
                    )
                )

        for family, family_frame in run_frame.groupby("feature_family", sort=True):
            for metric in R1_METRICS:
                rows.append(
                    long_row(
                        metadata=metadata,
                        representation="R2_feature_family",
                        metric=metric,
                        value=float(family_frame[metric].mean()),
                        provenance_table="stage7b_run_feature_summary.csv",
                        feature_family=str(family),
                    )
                )
    return pd.DataFrame(rows)


def build_selected_r3_dimensions(
    router_summary: pd.DataFrame,
    metadata_lookup: dict[str, dict[str, Any]],
    feature_lookup: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in router_summary.iterrows():
        metadata = metadata_lookup[str(row["run_id"])]
        feature = feature_lookup[int(row["feature_index"])]
        router_id = int(row["router_id"])
        for metric in R3_METRICS:
            rows.append(
                long_row(
                    metadata=metadata,
                    representation="R3_router_feature",
                    metric=metric,
                    value=float(row[metric]),
                    provenance_table="stage7b_selected_router_feature_summary.csv",
                    feature=feature,
                    router_id=router_id,
                )
            )
    return pd.DataFrame(rows)


def build_selected_r4_dimensions(
    temporal_summary: pd.DataFrame,
    metadata_lookup: dict[str, dict[str, Any]],
    feature_lookup: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in temporal_summary.iterrows():
        metadata = metadata_lookup[str(row["run_id"])]
        feature = feature_lookup[int(row["feature_index"])]
        temporal_position = int(row["temporal_position"])
        for metric in R4_METRICS:
            rows.append(
                long_row(
                    metadata=metadata,
                    representation="R4_temporal_position_feature",
                    metric=metric,
                    value=float(row[metric]),
                    provenance_table="stage7b_selected_temporal_feature_summary.csv",
                    feature=feature,
                    temporal_position=temporal_position,
                )
            )
    return pd.DataFrame(rows)


def build_selected_r5_dimensions(
    spatial_summary: pd.DataFrame,
    metadata_lookup: dict[str, dict[str, Any]],
    feature_lookup: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in spatial_summary.iterrows():
        metadata = metadata_lookup[str(row["run_id"])]
        feature = feature_lookup[int(row["feature_index"])]
        for metric in R5_METRICS:
            rows.append(
                long_row(
                    metadata=metadata,
                    representation="R5_spatial_concentration",
                    metric=metric,
                    value=float(row[metric]),
                    provenance_table="stage7b_selected_spatial_concentration.csv",
                    feature=feature,
                    metadata_dependent=False,
                )
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Training raw scan: R3, R4, R5 and histograms
# -----------------------------------------------------------------------------

def top_shares(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.minimum(values.min(axis=1, keepdims=True), 0.0)
    totals = shifted.sum(axis=1)
    ordered = np.sort(shifted, axis=1)
    top_one = ordered[:, -1]
    top_two = ordered[:, -2:].sum(axis=1)
    one_share = np.divide(
        top_one,
        totals,
        out=np.zeros_like(top_one),
        where=totals > EPS,
    )
    two_share = np.divide(
        top_two,
        totals,
        out=np.zeros_like(top_two),
        where=totals > EPS,
    )
    return one_share, two_share


def summarize_training_run(
    run_data: np.ndarray,
    metadata: dict[str, Any],
    feature_lookup: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    windows, routers, temporal_positions, features = run_data.shape

    temporal_variance = np.var(run_data, axis=2)
    temporal_excursion = run_data.max(axis=2) - run_data.min(axis=2)

    for router_id in range(routers):
        for feature_index in range(features):
            values = run_data[:, router_id, :, feature_index]
            feature = feature_lookup[feature_index]
            metrics = {
                "mean": float(values.mean()),
                "fraction_equal_1": float(np.mean(values == 1.0)),
                "fraction_above_0_95": float(np.mean(values > 0.95)),
                "mean_temporal_variance": float(
                    temporal_variance[:, router_id, feature_index].mean()
                ),
                "mean_temporal_excursion": float(
                    temporal_excursion[:, router_id, feature_index].mean()
                ),
            }
            for metric, value in metrics.items():
                rows.append(
                    long_row(
                        metadata=metadata,
                        representation="R3_router_feature",
                        metric=metric,
                        value=value,
                        provenance_table="training_raw_tensor_scan",
                        feature=feature,
                        router_id=router_id,
                    )
                )

    for temporal_position in range(temporal_positions):
        for feature_index in range(features):
            values = run_data[:, :, temporal_position, feature_index]
            feature = feature_lookup[feature_index]
            if temporal_position == 0:
                previous_change = 0.0
            else:
                previous_change = float(
                    np.abs(
                        values
                        - run_data[:, :, temporal_position - 1, feature_index]
                    ).mean()
                )
            metrics = {
                "mean": float(values.mean()),
                "fraction_equal_1": float(np.mean(values == 1.0)),
                "fraction_above_0_95": float(np.mean(values > 0.95)),
                "mean_spatial_variance": float(np.var(values, axis=1).mean()),
                "mean_absolute_change_from_previous_position": previous_change,
            }
            for metric, value in metrics.items():
                rows.append(
                    long_row(
                        metadata=metadata,
                        representation="R4_temporal_position_feature",
                        metric=metric,
                        value=value,
                        provenance_table="training_raw_tensor_scan",
                        feature=feature,
                        temporal_position=temporal_position,
                    )
                )

    for feature_index in range(features):
        feature = feature_lookup[feature_index]
        values = (
            run_data[..., feature_index]
            .transpose(0, 2, 1)
            .reshape(windows * temporal_positions, routers)
        )
        router_std = values.std(axis=1)
        maximum_minus_median = values.max(axis=1) - np.median(values, axis=1)
        top_one_share, top_two_share = top_shares(values)
        metrics = {
            "mean_router_std": float(router_std.mean()),
            "mean_maximum_minus_median": float(maximum_minus_median.mean()),
            "mean_top_router_share": float(top_one_share.mean()),
            "mean_top_two_router_share": float(top_two_share.mean()),
        }
        for metric, value in metrics.items():
            rows.append(
                long_row(
                    metadata=metadata,
                    representation="R5_spatial_concentration",
                    metric=metric,
                    value=value,
                    provenance_table="training_raw_tensor_scan",
                    feature=feature,
                )
            )

    return rows


COHORT_INDEX = {
    "all_training": 0,
    "normal_training": 1,
    "attack_training": 2,
}


def cohort_indices_for_class(true_graph: int) -> list[int]:
    return [
        COHORT_INDEX["all_training"],
        COHORT_INDEX[
            "attack_training" if true_graph == 1 else "normal_training"
        ],
    ]


def initialize_histograms() -> dict[str, np.ndarray]:
    cohort_count = len(COHORT_INDEX)

    def stats_arrays(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
        return {
            "count": np.zeros(shape, dtype=np.int64),
            "sum": np.zeros(shape, dtype=np.float64),
            "sumsq": np.zeros(shape, dtype=np.float64),
            "minimum": np.full(shape, np.inf, dtype=np.float64),
            "maximum": np.full(shape, -np.inf, dtype=np.float64),
            "exact_zero": np.zeros(shape, dtype=np.int64),
            "exact_one": np.zeros(shape, dtype=np.int64),
        }

    arrays: dict[str, np.ndarray] = {
        "bin_edges": np.linspace(0.0, 1.0, HIST_BINS + 1, dtype=np.float64),
        "cohort_names": np.asarray(
            ["all_training", "normal_training", "attack_training"]
        ),
        "global_feature_counts": np.zeros(
            (cohort_count, EXPECTED_FEATURES, HIST_BINS), dtype=np.int64
        ),
        "global_feature_underflow": np.zeros(
            (cohort_count, EXPECTED_FEATURES), dtype=np.int64
        ),
        "global_feature_overflow": np.zeros(
            (cohort_count, EXPECTED_FEATURES), dtype=np.int64
        ),
        "router_feature_counts": np.zeros(
            (
                cohort_count,
                EXPECTED_ROUTERS,
                EXPECTED_FEATURES,
                HIST_BINS,
            ),
            dtype=np.int64,
        ),
        "router_feature_underflow": np.zeros(
            (cohort_count, EXPECTED_ROUTERS, EXPECTED_FEATURES), dtype=np.int64
        ),
        "router_feature_overflow": np.zeros(
            (cohort_count, EXPECTED_ROUTERS, EXPECTED_FEATURES), dtype=np.int64
        ),
        "temporal_feature_counts": np.zeros(
            (
                cohort_count,
                EXPECTED_TEMPORAL,
                EXPECTED_FEATURES,
                HIST_BINS,
            ),
            dtype=np.int64,
        ),
        "temporal_feature_underflow": np.zeros(
            (cohort_count, EXPECTED_TEMPORAL, EXPECTED_FEATURES), dtype=np.int64
        ),
        "temporal_feature_overflow": np.zeros(
            (cohort_count, EXPECTED_TEMPORAL, EXPECTED_FEATURES), dtype=np.int64
        ),
    }

    for prefix, shape in [
        ("global_feature", (cohort_count, EXPECTED_FEATURES)),
        (
            "router_feature",
            (cohort_count, EXPECTED_ROUTERS, EXPECTED_FEATURES),
        ),
        (
            "temporal_feature",
            (cohort_count, EXPECTED_TEMPORAL, EXPECTED_FEATURES),
        ),
    ]:
        for suffix, array in stats_arrays(shape).items():
            arrays[f"{prefix}_{suffix}"] = array

    return arrays


def histogram_indices(values: np.ndarray) -> np.ndarray:
    indices = np.floor(np.asarray(values, dtype=np.float64) * HIST_BINS).astype(
        np.int64
    )
    return np.clip(indices, 0, HIST_BINS - 1)


def update_histogram(
    counts: np.ndarray,
    underflow: np.ndarray,
    overflow: np.ndarray,
    index: tuple[int, ...],
    values: np.ndarray,
) -> None:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    under = flat < 0.0
    over = flat > 1.0
    valid = ~(under | over)
    underflow[index] += int(under.sum())
    overflow[index] += int(over.sum())
    if np.any(valid):
        bins = histogram_indices(flat[valid])
        counts[index] += np.bincount(bins, minlength=HIST_BINS)


def update_moments(
    histograms: dict[str, np.ndarray],
    prefix: str,
    index: tuple[int, ...],
    values: np.ndarray,
) -> None:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    histograms[f"{prefix}_count"][index] += len(flat)
    histograms[f"{prefix}_sum"][index] += float(flat.sum())
    histograms[f"{prefix}_sumsq"][index] += float(np.square(flat).sum())
    histograms[f"{prefix}_minimum"][index] = min(
        float(histograms[f"{prefix}_minimum"][index]),
        float(flat.min()),
    )
    histograms[f"{prefix}_maximum"][index] = max(
        float(histograms[f"{prefix}_maximum"][index]),
        float(flat.max()),
    )
    histograms[f"{prefix}_exact_zero"][index] += int(np.sum(flat == 0.0))
    histograms[f"{prefix}_exact_one"][index] += int(np.sum(flat == 1.0))


def update_all_histograms(
    histograms: dict[str, np.ndarray],
    run_data: np.ndarray,
    true_graph: int,
) -> None:
    cohort_indices = cohort_indices_for_class(true_graph)

    for cohort_index in cohort_indices:
        for feature_index in range(EXPECTED_FEATURES):
            values = run_data[..., feature_index]
            index = (cohort_index, feature_index)
            update_histogram(
                histograms["global_feature_counts"],
                histograms["global_feature_underflow"],
                histograms["global_feature_overflow"],
                index,
                values,
            )
            update_moments(histograms, "global_feature", index, values)

        for router_id in range(EXPECTED_ROUTERS):
            for feature_index in range(EXPECTED_FEATURES):
                values = run_data[:, router_id, :, feature_index]
                index = (cohort_index, router_id, feature_index)
                update_histogram(
                    histograms["router_feature_counts"],
                    histograms["router_feature_underflow"],
                    histograms["router_feature_overflow"],
                    index,
                    values,
                )
                update_moments(histograms, "router_feature", index, values)

        for temporal_position in range(EXPECTED_TEMPORAL):
            for feature_index in range(EXPECTED_FEATURES):
                values = run_data[:, :, temporal_position, feature_index]
                index = (cohort_index, temporal_position, feature_index)
                update_histogram(
                    histograms["temporal_feature_counts"],
                    histograms["temporal_feature_underflow"],
                    histograms["temporal_feature_overflow"],
                    index,
                    values,
                )
                update_moments(histograms, "temporal_feature", index, values)


# -----------------------------------------------------------------------------
# Reference construction and coverage application
# -----------------------------------------------------------------------------

def reference_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise RuntimeError("A training reference dimension has no finite values.")
    q = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    std = float(np.std(values, ddof=0))
    iqr = float(q[4] - q[2])
    return {
        "training_run_count": int(len(values)),
        "training_mean": float(np.mean(values)),
        "training_std": std,
        "training_minimum": float(np.min(values)),
        "training_maximum": float(np.max(values)),
        "training_p01": float(q[0]),
        "training_p05": float(q[1]),
        "training_q1": float(q[2]),
        "training_median": float(q[3]),
        "training_q3": float(q[4]),
        "training_p95": float(q[5]),
        "training_p99": float(q[6]),
        "training_iqr": iqr,
        "zero_variance_dimension": bool(std <= EPS),
        "zero_iqr_dimension": bool(iqr <= EPS),
    }


def build_references(training_dimensions: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        "representation",
        "dimension_id",
        "metric",
        "feature_index",
        "feature_name",
        "feature_family",
        "measurement_type",
        "flow_side",
        "direction",
        "router_id",
        "temporal_position",
        "metadata_dependent",
        "provenance_table",
    ]
    rows: list[dict[str, Any]] = []
    for cohort_name, class_value in PRIMARY_COHORTS.items():
        if class_value is None:
            cohort = training_dimensions
        else:
            cohort = training_dimensions[
                training_dimensions["true_graph"].eq(class_value)
            ]
        for _, group in cohort.groupby("dimension_id", sort=True):
            first = group.iloc[0]
            rows.append(
                {
                    "reference_cohort": cohort_name,
                    **{column: first[column] for column in metadata_columns},
                    **reference_stats(group["value"].to_numpy(dtype=np.float64)),
                }
            )
    return pd.DataFrame(rows)


def build_descriptive_cohort_reference(
    training_dimensions: pd.DataFrame,
) -> pd.DataFrame:
    source = training_dimensions[
        training_dimensions["representation"].isin(
            ["R1_global_run_feature", "R2_feature_family"]
        )
    ].copy()
    cohort_definitions = [
        ("graph_class", "true_graph"),
        ("profile", "profile"),
        ("attacker_count", "attacker_count"),
        ("strength_group", "strength_group"),
        ("active_core_count", "active_core_count"),
    ]
    metadata_columns = [
        "representation",
        "dimension_id",
        "metric",
        "feature_index",
        "feature_name",
        "feature_family",
        "measurement_type",
        "flow_side",
        "direction",
        "router_id",
        "temporal_position",
        "metadata_dependent",
        "provenance_table",
    ]
    rows: list[dict[str, Any]] = []
    for cohort_type, cohort_column in cohort_definitions:
        for cohort_value, cohort in source.groupby(cohort_column, sort=True):
            for _, group in cohort.groupby("dimension_id", sort=True):
                first = group.iloc[0]
                rows.append(
                    {
                        "cohort_type": cohort_type,
                        "cohort_value": str(cohort_value),
                        **{column: first[column] for column in metadata_columns},
                        **reference_stats(group["value"].to_numpy(dtype=np.float64)),
                        "primary_coverage_reference": False,
                    }
                )
    return pd.DataFrame(rows)


def reference_role(target_class: int, cohort_name: str) -> str:
    if cohort_name == "all_training":
        return "all_training"
    if target_class == 0 and cohort_name == "normal_training":
        return "same_class"
    if target_class == 1 and cohort_name == "attack_training":
        return "same_class"
    return "opposite_class"


def validate_reference_cohort_attachment(
    coverage: pd.DataFrame,
    target_id_columns: list[str],
    context: str,
) -> None:
    required = set(PRIMARY_COHORTS)
    if coverage.empty:
        raise RuntimeError(f"{context} is empty.")
    missing_columns = set(target_id_columns + ["reference_cohort"]) - set(
        coverage.columns
    )
    if missing_columns:
        raise RuntimeError(
            f"{context} missing columns: {sorted(missing_columns)}"
        )

    bad_groups: list[tuple[Any, ...]] = []
    for keys, group in coverage.groupby(target_id_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        observed = set(group["reference_cohort"].astype(str))
        if observed != required or len(group) != len(required):
            bad_groups.append(keys)
            if len(bad_groups) >= 10:
                break

    if bad_groups:
        raise RuntimeError(
            f"{context} does not attach exactly {sorted(required)} to every "
            f"target dimension. Example keys: {bad_groups}"
        )


def run_reference_application_smoke_test() -> None:
    """Exercise indexed reference lookup before any full tensor scan."""
    base = {
        "profile": "smoke",
        "active_cores": "0",
        "active_core_count": 1,
        "attackers": "",
        "attacker_count": 0,
        "strength": 0,
        "strength_group": "normal",
        "seed": 0,
        "number_of_windows": 1,
        "representation": "R1_global_run_feature",
        "metric": "mean",
        "feature_index": 0,
        "feature_name": "ifd_in_norm",
        "feature_family": "aggregate_ifd",
        "measurement_type": "ifd",
        "flow_side": "input",
        "direction": "",
        "router_id": -1,
        "temporal_position": -1,
        "metadata_dependent": False,
        "provenance_table": "smoke_test",
    }
    dimension = dimension_id(
        representation="R1_global_run_feature",
        metric="mean",
        feature_index=0,
    )

    training_rows = []
    for run_id, true_graph, value in [
        ("smoke_train_normal", 0, 0.10),
        ("smoke_train_attack", 1, 0.80),
    ]:
        training_rows.append(
            {
                **base,
                "run_id": run_id,
                "split": "train",
                "true_graph": true_graph,
                "dimension_id": dimension,
                "value": value,
            }
        )

    target = pd.DataFrame(
        [
            {
                **base,
                "run_id": "smoke_target",
                "split": "test",
                "true_graph": 0,
                "dimension_id": dimension,
                "value": 0.20,
            }
        ]
    )
    references = build_references(pd.DataFrame(training_rows))
    summary_lookup = references.set_index(
        ["reference_cohort", "dimension_id"],
        drop=False,
        verify_integrity=True,
    )
    if not {"reference_cohort", "dimension_id"}.issubset(summary_lookup.columns):
        raise RuntimeError(
            "Summary smoke-test lookup dropped indexed provenance fields."
        )

    duplicate_rejected = False
    duplicate_references = pd.concat(
        [references, references.iloc[[0]]],
        ignore_index=True,
    )
    try:
        apply_references(target, duplicate_references)
    except RuntimeError as exc:
        duplicate_rejected = "Duplicate reference keys" in str(exc)
    if not duplicate_rejected:
        raise RuntimeError(
            "Summary smoke test did not reject duplicate reference keys."
        )

    coverage = apply_references(target, references)
    observed = set(coverage["reference_cohort"].astype(str))
    if observed != set(PRIMARY_COHORTS) or len(coverage) != len(PRIMARY_COHORTS):
        raise RuntimeError(
            "Reference-application smoke test did not attach all cohorts."
        )
    if not coverage["dimension_id"].eq(dimension).all():
        raise RuntimeError(
            "Reference-application smoke test lost dimension_id."
        )

    raw_dimension = dimension_id(
        representation="R6_raw_global_feature",
        metric="raw_distribution",
        feature_index=0,
    )
    raw_reference_rows = []
    for cohort_name, mean in [
        ("all_training", 0.45),
        ("normal_training", 0.10),
        ("attack_training", 0.80),
    ]:
        raw_reference_rows.append(
            {
                "reference_cohort": cohort_name,
                "representation": "R6_raw_global_feature",
                "dimension_id": raw_dimension,
                "metric": "raw_distribution",
                "feature_index": 0,
                "feature_name": "ifd_in_norm",
                "feature_family": "aggregate_ifd",
                "measurement_type": "ifd",
                "flow_side": "input",
                "direction": "",
                "router_id": -1,
                "temporal_position": -1,
                "training_run_count": 2,
                "training_value_count": 4,
                "training_mean": mean,
                "training_std": 0.10,
                "training_minimum": 0.0,
                "training_maximum": 1.0,
                "training_p01": 0.01,
                "training_p05": 0.05,
                "training_q1": 0.20,
                "training_median": mean,
                "training_q3": 0.70,
                "training_p95": 0.95,
                "training_p99": 0.99,
                "training_iqr": 0.50,
                "training_fraction_equal_0": 0.0,
                "training_fraction_equal_1": 0.0,
                "zero_variance_dimension": False,
                "zero_iqr_dimension": False,
                "quantiles_histogram_approximated": True,
                "histogram_bin_count": HIST_BINS,
            }
        )
    raw_references = pd.DataFrame(raw_reference_rows)
    raw_lookup = raw_references.set_index(
        ["reference_cohort", "dimension_id"],
        drop=False,
        verify_integrity=True,
    )
    if not {"reference_cohort", "dimension_id"}.issubset(raw_lookup.columns):
        raise RuntimeError(
            "Raw smoke-test lookup dropped indexed provenance fields."
        )

    duplicate_raw_rejected = False
    duplicate_raw = pd.concat(
        [raw_references, raw_references.iloc[[0]]],
        ignore_index=True,
    )
    try:
        duplicate_raw.set_index(
            ["reference_cohort", "dimension_id"],
            drop=False,
            verify_integrity=True,
        )
    except ValueError:
        duplicate_raw_rejected = True
    if not duplicate_raw_rejected:
        raise RuntimeError(
            "Raw smoke test did not reject duplicate reference keys."
        )

    raw_rows = []
    raw_metadata = {
        "run_id": "smoke_target",
        "split": "test",
        "true_graph": 0,
        "profile": "smoke",
        "active_cores": "0",
        "active_core_count": 1,
        "attackers": "",
        "attacker_count": 0,
        "strength": 0,
        "strength_group": "normal",
        "seed": 0,
        "number_of_windows": 1,
    }
    for cohort_name in PRIMARY_COHORTS:
        raw_rows.append(
            compare_raw_values_to_reference(
                np.asarray([0.10, 0.20], dtype=np.float32),
                raw_metadata,
                raw_lookup.loc[(cohort_name, raw_dimension)],
                reference_cohort=cohort_name,
            )
        )
    raw_coverage = pd.DataFrame(raw_rows)
    validate_reference_cohort_attachment(
        raw_coverage,
        target_id_columns=["target_run_id", "dimension_id"],
        context="raw smoke-test coverage",
    )

    print("summary reference smoke test: True", flush=True)
    print("raw reference smoke test: True", flush=True)
    print("all reference cohorts attached: True", flush=True)
    print("indexed provenance fields retained: True", flush=True)
    print("duplicate reference keys rejected: True", flush=True)
    print(
        "reference application smoke test: True "
        f"({','.join(sorted(observed))}; summary+raw)",
        flush=True,
    )


def compare_to_reference(
    target: pd.Series,
    reference: pd.Series,
    reference_cohort: str,
) -> dict[str, Any]:
    value = float(target["value"])
    mean = float(reference["training_mean"])
    std = float(reference["training_std"])
    median = float(reference["training_median"])
    iqr = float(reference["training_iqr"])

    below_min = value < float(reference["training_minimum"])
    above_max = value > float(reference["training_maximum"])
    outside_minmax = below_min or above_max
    outside_p01_p99 = (
        value < float(reference["training_p01"])
        or value > float(reference["training_p99"])
    )
    outside_p05_p95 = (
        value < float(reference["training_p05"])
        or value > float(reference["training_p95"])
    )

    if std <= EPS:
        z_score = np.nan
        constant_violation = not math.isclose(value, mean, abs_tol=EPS, rel_tol=0.0)
    else:
        z_score = (value - mean) / std
        constant_violation = False

    if iqr <= EPS:
        robust = np.nan
        zero_iqr_violation = not math.isclose(
            value, median, abs_tol=EPS, rel_tol=0.0
        )
    else:
        robust = (value - median) / iqr
        zero_iqr_violation = False

    abs_z = abs(float(z_score)) if np.isfinite(z_score) else np.nan
    abs_robust = abs(float(robust)) if np.isfinite(robust) else np.nan

    return {
        "target_run_id": str(target["run_id"]),
        "target_split": str(target["split"]),
        "target_true_graph": int(target["true_graph"]),
        "target_profile": str(target["profile"]),
        "target_active_cores": str(target["active_cores"]),
        "target_attacker_count": int(target["attacker_count"]),
        "target_attackers": str(target["attackers"]),
        "target_strength": int(target["strength"]),
        "target_strength_group": str(target["strength_group"]),
        "reference_cohort": str(reference_cohort),
        "reference_role": reference_role(
            int(target["true_graph"]), str(reference_cohort)
        ),
        "representation": str(target["representation"]),
        "dimension_id": str(target["dimension_id"]),
        "metric": str(target["metric"]),
        "target_value": value,
        "feature_index": int(target["feature_index"]),
        "feature_name": str(target["feature_name"]),
        "feature_family": str(target["feature_family"]),
        "measurement_type": str(target["measurement_type"]),
        "flow_side": str(target["flow_side"]),
        "direction": str(target["direction"]),
        "router_id": int(target["router_id"]),
        "temporal_position": int(target["temporal_position"]),
        "metadata_dependent": bool(target["metadata_dependent"]),
        "target_provenance_table": str(target["provenance_table"]),
        **{
            column: reference[column]
            for column in [
                "training_run_count",
                "training_mean",
                "training_std",
                "training_minimum",
                "training_maximum",
                "training_p01",
                "training_p05",
                "training_q1",
                "training_median",
                "training_q3",
                "training_p95",
                "training_p99",
                "training_iqr",
                "zero_variance_dimension",
                "zero_iqr_dimension",
            ]
        },
        "below_training_minimum": below_min,
        "above_training_maximum": above_max,
        "outside_training_minimum_maximum": outside_minmax,
        "outside_training_p01_p99": outside_p01_p99,
        "outside_training_p05_p95": outside_p05_p95,
        "z_score": z_score,
        "absolute_z_score": abs_z,
        "robust_deviation": robust,
        "absolute_robust_deviation": abs_robust,
        "absolute_z_gt_2": bool(np.isfinite(abs_z) and abs_z > 2.0),
        "absolute_z_gt_3": bool(np.isfinite(abs_z) and abs_z > 3.0),
        "absolute_z_gt_5": bool(np.isfinite(abs_z) and abs_z > 5.0),
        "constant_dimension_violation": constant_violation,
        "zero_iqr_dimension_violation": zero_iqr_violation,
    }


def apply_references(
    targets: pd.DataFrame,
    references: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {"reference_cohort", "dimension_id"}
    missing = required_columns - set(references.columns)
    if missing:
        raise RuntimeError(
            f"Reference table missing indexed fields: {sorted(missing)}"
        )

    duplicate_keys = references.duplicated(
        ["reference_cohort", "dimension_id"],
        keep=False,
    )
    if duplicate_keys.any():
        examples = references.loc[
            duplicate_keys,
            ["reference_cohort", "dimension_id"],
        ].head(10)
        raise RuntimeError(
            "Duplicate reference keys detected:\n"
            + examples.to_string(index=False)
        )

    # Keep indexed fields as ordinary columns as a defensive invariant, while
    # also passing the cohort explicitly so coverage rows never depend on
    # Pandas preserving index-level fields in the returned Series.
    lookup = references.set_index(
        ["reference_cohort", "dimension_id"],
        drop=False,
        verify_integrity=True,
    )

    rows: list[dict[str, Any]] = []
    for _, target in targets.iterrows():
        dimension = str(target["dimension_id"])
        for cohort_name in PRIMARY_COHORTS:
            key = (cohort_name, dimension)
            if key not in lookup.index:
                raise RuntimeError(f"Missing reference row for {key}.")
            reference = lookup.loc[key]
            if isinstance(reference, pd.DataFrame):
                raise RuntimeError(
                    f"Reference key {key} resolved to multiple rows."
                )
            rows.append(
                compare_to_reference(
                    target,
                    reference,
                    reference_cohort=cohort_name,
                )
            )

    coverage = pd.DataFrame(rows)
    validate_reference_cohort_attachment(
        coverage,
        target_id_columns=["target_run_id", "dimension_id"],
        context="summary target coverage",
    )
    return coverage


def summarize_coverage(coverage: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in coverage.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_record = dict(zip(group_columns, keys))
        abs_z = pd.to_numeric(group["absolute_z_score"], errors="coerce")
        abs_r = pd.to_numeric(
            group["absolute_robust_deviation"], errors="coerce"
        )
        finite_z = abs_z[np.isfinite(abs_z)]
        finite_r = abs_r[np.isfinite(abs_r)]
        rows.append(
            {
                **key_record,
                "dimension_count": int(len(group)),
                "outside_minimum_maximum_count": int(
                    group["outside_training_minimum_maximum"].sum()
                ),
                "outside_minimum_maximum_fraction": float(
                    group["outside_training_minimum_maximum"].mean()
                ),
                "outside_p01_p99_count": int(group["outside_training_p01_p99"].sum()),
                "outside_p01_p99_fraction": float(
                    group["outside_training_p01_p99"].mean()
                ),
                "outside_p05_p95_count": int(group["outside_training_p05_p95"].sum()),
                "outside_p05_p95_fraction": float(
                    group["outside_training_p05_p95"].mean()
                ),
                "absolute_z_gt_2_count": int(group["absolute_z_gt_2"].sum()),
                "absolute_z_gt_2_fraction": float(group["absolute_z_gt_2"].mean()),
                "absolute_z_gt_3_count": int(group["absolute_z_gt_3"].sum()),
                "absolute_z_gt_3_fraction": float(group["absolute_z_gt_3"].mean()),
                "absolute_z_gt_5_count": int(group["absolute_z_gt_5"].sum()),
                "absolute_z_gt_5_fraction": float(group["absolute_z_gt_5"].mean()),
                "constant_dimension_violation_count": int(
                    group["constant_dimension_violation"].sum()
                ),
                "zero_iqr_dimension_violation_count": int(
                    group["zero_iqr_dimension_violation"].sum()
                ),
                "finite_z_score_count": int(len(finite_z)),
                "median_absolute_z_score": (
                    float(np.median(finite_z)) if len(finite_z) else np.nan
                ),
                "maximum_absolute_z_score": (
                    float(np.max(finite_z)) if len(finite_z) else np.nan
                ),
                "finite_robust_deviation_count": int(len(finite_r)),
                "median_absolute_robust_deviation": (
                    float(np.median(finite_r)) if len(finite_r) else np.nan
                ),
                "maximum_absolute_robust_deviation": (
                    float(np.max(finite_r)) if len(finite_r) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def build_saturation_pattern_coverage(
    target_r3: pd.DataFrame,
    training_r3: pd.DataFrame,
    router_coverage: pd.DataFrame,
) -> pd.DataFrame:
    exact_target = target_r3[target_r3["metric"].eq("fraction_equal_1")]
    exact_training = training_r3[training_r3["metric"].eq("fraction_equal_1")]
    exact_coverage = router_coverage[
        router_coverage["metric"].eq("fraction_equal_1")
    ]

    rows: list[dict[str, Any]] = []
    for _, row in exact_coverage.iterrows():
        cohort_name = str(row["reference_cohort"])
        cohort = exact_training
        if cohort_name == "normal_training":
            cohort = cohort[cohort["true_graph"].eq(0)]
        elif cohort_name == "attack_training":
            cohort = cohort[cohort["true_graph"].eq(1)]

        training_values = cohort[
            cohort["dimension_id"].eq(str(row["dimension_id"]))
        ]["value"].to_numpy(dtype=np.float64)
        target_value = float(row["target_value"])

        rows.append(
            {
                **row.to_dict(),
                "training_runs_with_same_exact_rate": int(
                    np.sum(np.isclose(training_values, target_value, atol=EPS, rtol=0.0))
                ),
                "training_runs_with_constant_exact_one_pattern": int(
                    np.sum(np.isclose(training_values, 1.0, atol=EPS, rtol=0.0))
                ),
                "training_runs_with_constant_exact_zero_pattern": int(
                    np.sum(np.isclose(training_values, 0.0, atol=EPS, rtol=0.0))
                ),
                "target_is_constant_exact_one_pattern": bool(
                    math.isclose(target_value, 1.0, abs_tol=EPS, rel_tol=0.0)
                ),
                "target_is_constant_exact_zero_pattern": bool(
                    math.isclose(target_value, 0.0, abs_tol=EPS, rel_tol=0.0)
                ),
                "is_suspicious_attacker_router_pattern": bool(
                    str(row["target_run_id"]) == HARD_ATTACK
                    and int(row["router_id"]) == SUSPICIOUS_ROUTER
                    and int(row["feature_index"]) in SUSPICIOUS_FEATURES
                ),
            }
        )

    return pd.DataFrame(rows)


def histogram_quantile(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    quantile: float,
) -> float:
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total <= 0:
        return np.nan
    threshold = quantile * (total - 1)
    cumulative = np.cumsum(counts)
    index = int(np.searchsorted(cumulative, threshold, side="right"))
    index = min(max(index, 0), len(counts) - 1)
    return float((bin_edges[index] + bin_edges[index + 1]) / 2.0)


def raw_reference_record(
    histograms: dict[str, np.ndarray],
    cohort_name: str,
    prefix: str,
    index_without_cohort: tuple[int, ...],
    representation: str,
    feature: dict[str, Any],
    training_run_count: int,
    router_id: int = -1,
    temporal_position: int = -1,
) -> dict[str, Any]:
    cohort_index = COHORT_INDEX[cohort_name]
    index = (cohort_index, *index_without_cohort)
    count = int(histograms[f"{prefix}_count"][index])
    total_sum = float(histograms[f"{prefix}_sum"][index])
    total_sumsq = float(histograms[f"{prefix}_sumsq"][index])
    mean = total_sum / count
    variance = max(0.0, total_sumsq / count - mean * mean)
    std = math.sqrt(variance)
    minimum = float(histograms[f"{prefix}_minimum"][index])
    maximum = float(histograms[f"{prefix}_maximum"][index])
    exact_zero = int(histograms[f"{prefix}_exact_zero"][index])
    exact_one = int(histograms[f"{prefix}_exact_one"][index])
    counts = histograms[f"{prefix}_counts"][index]
    bin_edges = histograms["bin_edges"]
    p01 = histogram_quantile(counts, bin_edges, 0.01)
    p05 = histogram_quantile(counts, bin_edges, 0.05)
    q1 = histogram_quantile(counts, bin_edges, 0.25)
    median = histogram_quantile(counts, bin_edges, 0.50)
    q3 = histogram_quantile(counts, bin_edges, 0.75)
    p95 = histogram_quantile(counts, bin_edges, 0.95)
    p99 = histogram_quantile(counts, bin_edges, 0.99)
    iqr = q3 - q1

    return {
        "reference_cohort": cohort_name,
        "representation": representation,
        "dimension_id": dimension_id(
            representation=representation,
            metric="raw_distribution",
            feature_index=int(feature["feature_index"]),
            router_id=router_id,
            temporal_position=temporal_position,
        ),
        "metric": "raw_distribution",
        "feature_index": int(feature["feature_index"]),
        "feature_name": str(feature["feature_name"]),
        "feature_family": str(feature["feature_family"]),
        "measurement_type": str(feature["measurement_type"]),
        "flow_side": str(feature["flow_side"]),
        "direction": str(feature["direction"]),
        "router_id": int(router_id),
        "temporal_position": int(temporal_position),
        "metadata_dependent": False,
        "provenance_table": "training_raw_tensor_histogram_scan",
        "training_run_count": int(training_run_count),
        "training_value_count": count,
        "training_mean": mean,
        "training_std": std,
        "training_minimum": minimum,
        "training_maximum": maximum,
        "training_p01": p01,
        "training_p05": p05,
        "training_q1": q1,
        "training_median": median,
        "training_q3": q3,
        "training_p95": p95,
        "training_p99": p99,
        "training_iqr": iqr,
        "training_fraction_equal_0": exact_zero / count,
        "training_fraction_equal_1": exact_one / count,
        "zero_variance_dimension": bool(std <= EPS),
        "zero_iqr_dimension": bool(iqr <= EPS),
        "quantiles_histogram_approximated": True,
        "histogram_bin_count": HIST_BINS,
    }


def build_raw_references(
    histograms: dict[str, np.ndarray],
    taxonomy: pd.DataFrame,
    training_dimensions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_lookup = build_feature_lookup(taxonomy)
    cohort_run_counts = {
        "all_training": int(training_dimensions["run_id"].nunique()),
        "normal_training": int(
            training_dimensions.loc[
                training_dimensions["true_graph"].eq(0), "run_id"
            ].nunique()
        ),
        "attack_training": int(
            training_dimensions.loc[
                training_dimensions["true_graph"].eq(1), "run_id"
            ].nunique()
        ),
    }

    for cohort_name in PRIMARY_COHORTS:
        run_count = cohort_run_counts[cohort_name]
        for feature_index in range(EXPECTED_FEATURES):
            feature = feature_lookup[feature_index]
            rows.append(
                raw_reference_record(
                    histograms,
                    cohort_name,
                    "global_feature",
                    (feature_index,),
                    "R6_raw_global_feature",
                    feature,
                    run_count,
                )
            )

        for router_id in range(EXPECTED_ROUTERS):
            for feature_index in range(EXPECTED_FEATURES):
                feature = feature_lookup[feature_index]
                rows.append(
                    raw_reference_record(
                        histograms,
                        cohort_name,
                        "router_feature",
                        (router_id, feature_index),
                        "R6_raw_router_feature",
                        feature,
                        run_count,
                        router_id=router_id,
                    )
                )

        for temporal_position in range(EXPECTED_TEMPORAL):
            for feature_index in range(EXPECTED_FEATURES):
                feature = feature_lookup[feature_index]
                rows.append(
                    raw_reference_record(
                        histograms,
                        cohort_name,
                        "temporal_feature",
                        (temporal_position, feature_index),
                        "R6_raw_temporal_feature",
                        feature,
                        run_count,
                        temporal_position=temporal_position,
                    )
                )

    return pd.DataFrame(rows)


def compare_raw_values_to_reference(
    values: np.ndarray,
    metadata: dict[str, Any],
    reference: pd.Series,
    reference_cohort: str,
) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(flat).all():
        raise RuntimeError(
            f"Non-finite selected-target raw values in {metadata['run_id']}."
        )

    mean = float(reference["training_mean"])
    std = float(reference["training_std"])
    median = float(reference["training_median"])
    iqr = float(reference["training_iqr"])

    below_min = flat < float(reference["training_minimum"])
    above_max = flat > float(reference["training_maximum"])
    outside_p01_p99 = (
        (flat < float(reference["training_p01"]))
        | (flat > float(reference["training_p99"]))
    )
    outside_p05_p95 = (
        (flat < float(reference["training_p05"]))
        | (flat > float(reference["training_p95"]))
    )

    if std <= EPS:
        abs_z = np.full(len(flat), np.nan)
        constant_violation_fraction = float(
            np.mean(~np.isclose(flat, mean, atol=EPS, rtol=0.0))
        )
    else:
        abs_z = np.abs((flat - mean) / std)
        constant_violation_fraction = 0.0

    if iqr <= EPS:
        abs_robust = np.full(len(flat), np.nan)
        zero_iqr_violation_fraction = float(
            np.mean(~np.isclose(flat, median, atol=EPS, rtol=0.0))
        )
    else:
        abs_robust = np.abs((flat - median) / iqr)
        zero_iqr_violation_fraction = 0.0

    finite_z = abs_z[np.isfinite(abs_z)]
    finite_robust = abs_robust[np.isfinite(abs_robust)]

    return {
        **metadata,
        "target_run_id": metadata["run_id"],
        "target_split": metadata["split"],
        "target_true_graph": metadata["true_graph"],
        "reference_cohort": str(reference_cohort),
        "reference_role": reference_role(
            int(metadata["true_graph"]), str(reference_cohort)
        ),
        "representation": str(reference["representation"]),
        "dimension_id": str(reference["dimension_id"]),
        "metric": "raw_distribution",
        "feature_index": int(reference["feature_index"]),
        "feature_name": str(reference["feature_name"]),
        "feature_family": str(reference["feature_family"]),
        "measurement_type": str(reference["measurement_type"]),
        "flow_side": str(reference["flow_side"]),
        "direction": str(reference["direction"]),
        "router_id": int(reference["router_id"]),
        "temporal_position": int(reference["temporal_position"]),
        "target_value_count": int(len(flat)),
        "target_mean": float(flat.mean()),
        "target_std": float(flat.std(ddof=0)),
        "target_minimum": float(flat.min()),
        "target_maximum": float(flat.max()),
        "target_fraction_equal_0": float(np.mean(flat == 0.0)),
        "target_fraction_equal_1": float(np.mean(flat == 1.0)),
        **{
            column: reference[column]
            for column in [
                "training_run_count",
                "training_value_count",
                "training_mean",
                "training_std",
                "training_minimum",
                "training_maximum",
                "training_p01",
                "training_p05",
                "training_q1",
                "training_median",
                "training_q3",
                "training_p95",
                "training_p99",
                "training_iqr",
                "training_fraction_equal_0",
                "training_fraction_equal_1",
                "zero_variance_dimension",
                "zero_iqr_dimension",
                "quantiles_histogram_approximated",
                "histogram_bin_count",
            ]
        },
        "below_training_minimum_count": int(below_min.sum()),
        "below_training_minimum_fraction": float(below_min.mean()),
        "above_training_maximum_count": int(above_max.sum()),
        "above_training_maximum_fraction": float(above_max.mean()),
        "outside_training_minimum_maximum_count": int(
            np.sum(below_min | above_max)
        ),
        "outside_training_minimum_maximum_fraction": float(
            np.mean(below_min | above_max)
        ),
        "outside_training_p01_p99_count": int(outside_p01_p99.sum()),
        "outside_training_p01_p99_fraction": float(outside_p01_p99.mean()),
        "outside_training_p05_p95_count": int(outside_p05_p95.sum()),
        "outside_training_p05_p95_fraction": float(outside_p05_p95.mean()),
        "absolute_z_gt_2_count": int(np.sum(finite_z > 2.0)),
        "absolute_z_gt_2_fraction": float(
            np.mean(finite_z > 2.0) if len(finite_z) else 0.0
        ),
        "absolute_z_gt_3_count": int(np.sum(finite_z > 3.0)),
        "absolute_z_gt_3_fraction": float(
            np.mean(finite_z > 3.0) if len(finite_z) else 0.0
        ),
        "absolute_z_gt_5_count": int(np.sum(finite_z > 5.0)),
        "absolute_z_gt_5_fraction": float(
            np.mean(finite_z > 5.0) if len(finite_z) else 0.0
        ),
        "median_absolute_z_score": float(
            np.median(finite_z) if len(finite_z) else np.nan
        ),
        "maximum_absolute_z_score": float(
            np.max(finite_z) if len(finite_z) else np.nan
        ),
        "median_absolute_robust_deviation": float(
            np.median(finite_robust) if len(finite_robust) else np.nan
        ),
        "maximum_absolute_robust_deviation": float(
            np.max(finite_robust) if len(finite_robust) else np.nan
        ),
        "constant_dimension_violation_fraction": constant_violation_fraction,
        "zero_iqr_dimension_violation_fraction": zero_iqr_violation_fraction,
    }


def build_target_raw_coverage(
    x: np.ndarray,
    selected_run_ids: list[str],
    alignment: pd.DataFrame,
    metadata_lookup: dict[str, dict[str, Any]],
    raw_references: pd.DataFrame,
) -> pd.DataFrame:
    duplicate_keys = raw_references.duplicated(
        ["reference_cohort", "dimension_id"],
        keep=False,
    )
    if duplicate_keys.any():
        raise RuntimeError(
            "Duplicate raw-reference keys detected before target comparison."
        )
    reference_lookup = raw_references.set_index(
        ["reference_cohort", "dimension_id"],
        drop=False,
        verify_integrity=True,
    )
    indices_by_run = {
        run_id: group.sort_values("run_window_index")["dataset_index"].to_numpy(
            dtype=np.int64
        )
        for run_id, group in alignment[
            alignment["run_id"].isin(selected_run_ids)
        ].groupby("run_id", sort=True)
    }
    rows: list[dict[str, Any]] = []

    for number, run_id in enumerate(selected_run_ids, start=1):
        run_data = np.asarray(x[indices_by_run[run_id]], dtype=np.float32)
        metadata = metadata_lookup[run_id]
        print(
            f"[raw target {number:02d}/{len(selected_run_ids)}] {run_id}",
            flush=True,
        )

        dimensions: list[tuple[str, int, int, int, np.ndarray]] = []
        for feature_index in range(EXPECTED_FEATURES):
            dimensions.append(
                (
                    "R6_raw_global_feature",
                    feature_index,
                    -1,
                    -1,
                    run_data[..., feature_index],
                )
            )
        for router_id in range(EXPECTED_ROUTERS):
            for feature_index in range(EXPECTED_FEATURES):
                dimensions.append(
                    (
                        "R6_raw_router_feature",
                        feature_index,
                        router_id,
                        -1,
                        run_data[:, router_id, :, feature_index],
                    )
                )
        for temporal_position in range(EXPECTED_TEMPORAL):
            for feature_index in range(EXPECTED_FEATURES):
                dimensions.append(
                    (
                        "R6_raw_temporal_feature",
                        feature_index,
                        -1,
                        temporal_position,
                        run_data[:, :, temporal_position, feature_index],
                    )
                )

        for representation, feature_index, router_id, temporal_position, values in dimensions:
            dim_id = dimension_id(
                representation=representation,
                metric="raw_distribution",
                feature_index=feature_index,
                router_id=router_id,
                temporal_position=temporal_position,
            )
            for cohort_name in PRIMARY_COHORTS:
                reference = reference_lookup.loc[(cohort_name, dim_id)]
                if isinstance(reference, pd.DataFrame):
                    raise RuntimeError(
                        "Raw reference key resolved to multiple rows: "
                        f"{(cohort_name, dim_id)}"
                    )
                rows.append(
                    compare_raw_values_to_reference(
                        values,
                        metadata,
                        reference,
                        reference_cohort=cohort_name,
                    )
                )

        del run_data

    coverage = pd.DataFrame(rows)
    validate_reference_cohort_attachment(
        coverage,
        target_id_columns=["target_run_id", "dimension_id"],
        context="raw target coverage",
    )
    return coverage


def summarize_raw_coverage_by_representation(
    raw_coverage: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        "target_run_id",
        "target_split",
        "target_true_graph",
        "reference_cohort",
        "reference_role",
        "representation",
    ]
    for keys, group in raw_coverage.groupby(group_columns, sort=True):
        key_record = dict(zip(group_columns, keys))
        weights = group["target_value_count"].to_numpy(dtype=np.float64)
        weight_total = weights.sum()

        def weighted(column: str) -> float:
            values = group[column].to_numpy(dtype=np.float64)
            return float(np.sum(values * weights) / weight_total)

        rows.append(
            {
                **key_record,
                "dimension_count": int(len(group)),
                "outside_minimum_maximum_count": int(
                    group["outside_training_minimum_maximum_count"].sum()
                ),
                "outside_minimum_maximum_fraction": weighted(
                    "outside_training_minimum_maximum_fraction"
                ),
                "outside_p01_p99_count": int(
                    group["outside_training_p01_p99_count"].sum()
                ),
                "outside_p01_p99_fraction": weighted(
                    "outside_training_p01_p99_fraction"
                ),
                "outside_p05_p95_count": int(
                    group["outside_training_p05_p95_count"].sum()
                ),
                "outside_p05_p95_fraction": weighted(
                    "outside_training_p05_p95_fraction"
                ),
                "absolute_z_gt_2_count": int(group["absolute_z_gt_2_count"].sum()),
                "absolute_z_gt_2_fraction": weighted("absolute_z_gt_2_fraction"),
                "absolute_z_gt_3_count": int(group["absolute_z_gt_3_count"].sum()),
                "absolute_z_gt_3_fraction": weighted("absolute_z_gt_3_fraction"),
                "absolute_z_gt_5_count": int(group["absolute_z_gt_5_count"].sum()),
                "absolute_z_gt_5_fraction": weighted("absolute_z_gt_5_fraction"),
                "constant_dimension_violation_count": int(
                    np.sum(group["constant_dimension_violation_fraction"] > 0)
                ),
                "zero_iqr_dimension_violation_count": int(
                    np.sum(group["zero_iqr_dimension_violation_fraction"] > 0)
                ),
                "finite_z_score_count": int(
                    group["median_absolute_z_score"].notna().sum()
                ),
                "median_absolute_z_score": float(
                    group["median_absolute_z_score"].median(skipna=True)
                ),
                "maximum_absolute_z_score": float(
                    group["maximum_absolute_z_score"].max(skipna=True)
                ),
                "finite_robust_deviation_count": int(
                    group["median_absolute_robust_deviation"].notna().sum()
                ),
                "median_absolute_robust_deviation": float(
                    group["median_absolute_robust_deviation"].median(skipna=True)
                ),
                "maximum_absolute_robust_deviation": float(
                    group["maximum_absolute_robust_deviation"].max(skipna=True)
                ),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Hypothesis evidence
# -----------------------------------------------------------------------------

def one_summary_value(
    frame: pd.DataFrame,
    run_id: str,
    representation: str,
    reference_role_value: str,
    column: str,
) -> float:
    subset = frame[
        frame["target_run_id"].eq(run_id)
        & frame["representation"].eq(representation)
        & frame["reference_role"].eq(reference_role_value)
    ]
    return float(subset.iloc[0][column]) if len(subset) == 1 else np.nan


def build_hypothesis_evidence(
    coverage_by_representation: pd.DataFrame,
    saturation_pattern_coverage: pd.DataFrame,
) -> pd.DataFrame:
    hn_r1 = one_summary_value(
        coverage_by_representation,
        HARD_NORMAL,
        "R1_global_run_feature",
        "same_class",
        "outside_p01_p99_fraction",
    )
    hn_r3 = one_summary_value(
        coverage_by_representation,
        HARD_NORMAL,
        "R3_router_feature",
        "same_class",
        "outside_p01_p99_fraction",
    )
    ha_r1 = one_summary_value(
        coverage_by_representation,
        HARD_ATTACK,
        "R1_global_run_feature",
        "same_class",
        "outside_p01_p99_fraction",
    )
    ha_r3 = one_summary_value(
        coverage_by_representation,
        HARD_ATTACK,
        "R3_router_feature",
        "same_class",
        "outside_p01_p99_fraction",
    )

    suspicious = saturation_pattern_coverage[
        saturation_pattern_coverage["is_suspicious_attacker_router_pattern"]
        & saturation_pattern_coverage["reference_role"].eq("all_training")
    ]
    shared_constant = int(
        suspicious["training_runs_with_constant_exact_one_pattern"].sum()
    )
    outside_p01 = int(suspicious["outside_training_p01_p99"].sum())

    structured_normal = bool(
        np.isfinite(hn_r3)
        and np.isfinite(hn_r1)
        and hn_r3 > hn_r1 + 0.02
    )
    structured_attack = bool(
        np.isfinite(ha_r3)
        and np.isfinite(ha_r1)
        and ha_r3 > ha_r1 + 0.02
    )

    rows = [
        {
            "hypothesis_id": "H1",
            "hypothesis_name": "structured_router_direction_saturation",
            "stage7c1_status": "supported" if structured_normal else "inconclusive",
            "metric_1_name": "hard_normal_R3_same_class_p01_p99_violation_fraction",
            "metric_1_value": hn_r3,
            "metric_2_name": "hard_normal_R1_same_class_p01_p99_violation_fraction",
            "metric_2_value": hn_r1,
            "builder_recovery_required": True,
            "interpretation_boundary": (
                "Structured novelty can be established, but exact-1.0 meaning cannot."
            ),
        },
        {
            "hypothesis_id": "H2R",
            "hypothesis_name": "feature_relation_or_spatial_context",
            "stage7c1_status": "inconclusive",
            "metric_1_name": "hard_attack_R3_same_class_p01_p99_violation_fraction",
            "metric_1_value": ha_r3,
            "metric_2_name": "hard_attack_R1_same_class_p01_p99_violation_fraction",
            "metric_2_value": ha_r1,
            "builder_recovery_required": False,
            "interpretation_boundary": (
                "Stage 7C1 is univariate and does not test count–IFD joint geometry."
            ),
        },
        {
            "hypothesis_id": "H3",
            "hypothesis_name": "pooling_loss",
            "stage7c1_status": "not_tested",
            "metric_1_name": "",
            "metric_1_value": np.nan,
            "metric_2_name": "",
            "metric_2_value": np.nan,
            "builder_recovery_required": False,
            "interpretation_boundary": (
                "Coverage analysis cannot isolate the causal effect of pooling."
            ),
        },
        {
            "hypothesis_id": "H4",
            "hypothesis_name": "spatial_placement_pattern",
            "stage7c1_status": "supported" if structured_normal else "inconclusive",
            "metric_1_name": "hard_normal_R3_tail_fraction",
            "metric_1_value": hn_r3,
            "metric_2_name": "hard_normal_R1_tail_fraction",
            "metric_2_value": hn_r1,
            "builder_recovery_required": True,
            "interpretation_boundary": (
                "Route-level and physical-direction causation remain unproven."
            ),
        },
        {
            "hypothesis_id": "H5",
            "hypothesis_name": "normalization_or_upper_bound_encoding",
            "stage7c1_status": "inconclusive",
            "metric_1_name": "shared_training_constant_one_pattern_count",
            "metric_1_value": shared_constant,
            "metric_2_name": "suspicious_patterns_outside_p01_p99_count",
            "metric_2_value": outside_p01,
            "builder_recovery_required": True,
            "interpretation_boundary": (
                "Coverage shows recurrence or novelty, not the encoding mechanism."
            ),
        },
        {
            "hypothesis_id": "H6",
            "hypothesis_name": "invalid_or_no_traffic_port_encoding",
            "stage7c1_status": (
                "supported_as_encoding_suspicion" if shared_constant > 0 else "inconclusive"
            ),
            "metric_1_name": "training_constant_one_pattern_count",
            "metric_1_value": shared_constant,
            "metric_2_name": "suspicious_pattern_count",
            "metric_2_value": int(len(suspicious)),
            "builder_recovery_required": True,
            "interpretation_boundary": (
                "A repeated constant pattern supports an encoding suspicion only."
            ),
        },
        {
            "hypothesis_id": "H7",
            "hypothesis_name": "count_ifd_relationship",
            "stage7c1_status": "not_tested_univariate",
            "metric_1_name": "",
            "metric_1_value": np.nan,
            "metric_2_name": "",
            "metric_2_value": np.nan,
            "builder_recovery_required": False,
            "interpretation_boundary": (
                "Joint count–IFD relationships require Stage 7C2 or Stage 7D."
            ),
        },
        {
            "hypothesis_id": "H8",
            "hypothesis_name": "router_direction_training_coverage",
            "stage7c1_status": (
                "supported" if structured_normal or structured_attack else "inconclusive"
            ),
            "metric_1_name": "hard_normal_R3_same_class_tail_fraction",
            "metric_1_value": hn_r3,
            "metric_2_name": "hard_attack_R3_same_class_tail_fraction",
            "metric_2_value": ha_r3,
            "builder_recovery_required": False,
            "interpretation_boundary": (
                "Nearest-run class margins remain Stage 7C2."
            ),
        },
    ]
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting and report
# -----------------------------------------------------------------------------

def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    path: Path,
    xlabel: str,
    ylabel: str,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    save_figure(fig, path)


def make_plots(
    coverage_by_representation: pd.DataFrame,
    global_coverage: pd.DataFrame,
    router_coverage: pd.DataFrame,
    temporal_coverage: pd.DataFrame,
    saturation_coverage: pd.DataFrame,
    constant_violations: pd.DataFrame,
    plot_dir: Path,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    full = coverage_by_representation[
        coverage_by_representation["reference_role"].eq("all_training")
    ]
    matrix = full.pivot(
        index="target_run_id",
        columns="representation",
        values="outside_p01_p99_fraction",
    )
    plot_heatmap(
        matrix,
        "Outside full-training p01–p99 rate",
        plot_dir / "01_coverage_violation_rate_by_representation.png",
        "Representation",
        "Target run",
        vmin=0,
        vmax=1,
        figsize=(11, 8),
    )

    same = coverage_by_representation[
        coverage_by_representation["reference_role"].eq("same_class")
    ]
    matrix = same.pivot(
        index="target_run_id",
        columns="representation",
        values="outside_p01_p99_fraction",
    )
    plot_heatmap(
        matrix,
        "Same-class p01–p99 violation rate",
        plot_dir / "02_same_class_p01_p99_violation_heatmap.png",
        "Representation",
        "Target run",
        vmin=0,
        vmax=1,
        figsize=(11, 8),
    )

    matrix = same.pivot(
        index="target_run_id",
        columns="representation",
        values="median_absolute_robust_deviation",
    )
    plot_heatmap(
        matrix,
        "Median absolute robust deviation",
        plot_dir / "03_robust_deviation_heatmap.png",
        "Representation",
        "Target run",
        figsize=(11, 8),
    )

    for run_id, filename, title in [
        (HARD_NORMAL, "04_hard_normal_router_feature_novelty.png", "Hard normal router × feature novelty"),
        (HARD_ATTACK, "05_hard_attack_router_feature_novelty.png", "Hard attack router × feature novelty"),
    ]:
        subset = router_coverage[
            router_coverage["target_run_id"].eq(run_id)
            & router_coverage["reference_role"].eq("same_class")
        ].copy()
        subset["novelty"] = subset["absolute_robust_deviation"].fillna(
            subset["outside_training_minimum_maximum"].astype(float) * 5.0
        )
        matrix = subset.pivot_table(
            index="router_id",
            columns="feature_name",
            values="novelty",
            aggfunc="max",
        )
        plot_heatmap(matrix, title, plot_dir / filename, "Feature", "Router", figsize=(14, 7))

    subset = temporal_coverage[
        temporal_coverage["target_run_id"].eq(HARD_ATTACK)
        & temporal_coverage["reference_role"].eq("same_class")
    ].copy()
    subset["novelty"] = subset["absolute_robust_deviation"].fillna(
        subset["outside_training_minimum_maximum"].astype(float) * 5.0
    )
    matrix = subset.pivot_table(
        index="temporal_position",
        columns="feature_name",
        values="novelty",
        aggfunc="max",
    )
    plot_heatmap(
        matrix,
        "Hard attack temporal-position novelty",
        plot_dir / "06_hard_attack_temporal_position_novelty.png",
        "Feature",
        "Temporal position",
        figsize=(14, 6),
    )

    suspicious = saturation_coverage[
        saturation_coverage["is_suspicious_attacker_router_pattern"]
        & saturation_coverage["reference_role"].eq("all_training")
    ].sort_values("feature_index")
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(suspicious))
    width = 0.25
    ax.bar(positions - width, suspicious["target_value"], width, label="Target")
    ax.bar(positions, suspicious["training_median"], width, label="Training median")
    ax.bar(positions + width, suspicious["training_p95"], width, label="Training p95")
    ax.set_xticks(positions)
    ax.set_xticklabels(suspicious["feature_name"], rotation=45, ha="right")
    ax.set_ylabel("Exact-1.0 rate")
    ax.set_title("Router 12 exact-1.0 pattern novelty")
    ax.legend()
    save_figure(fig, plot_dir / "07_router12_exact_one_pattern_novelty.png")

    for run_id, filename, title in [
        (HARD_NORMAL, "08_hard_normal_same_vs_opposite_class.png", "Hard normal: same vs opposite class"),
        (HARD_ATTACK, "09_hard_attack_same_vs_opposite_class.png", "Hard attack: same vs opposite class"),
    ]:
        subset = coverage_by_representation[
            coverage_by_representation["target_run_id"].eq(run_id)
            & coverage_by_representation["reference_role"].isin(["same_class", "opposite_class"])
        ]
        pivot = subset.pivot(
            index="representation",
            columns="reference_role",
            values="outside_p01_p99_fraction",
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Outside p01–p99 fraction")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        save_figure(fig, plot_dir / filename)

    if constant_violations.empty:
        constant_summary = pd.DataFrame(columns=["target_run_id", "violation_count"])
    else:
        constant_summary = (
            constant_violations.groupby("target_run_id", as_index=False)
            .size()
            .rename(columns={"size": "violation_count"})
        )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(constant_summary["target_run_id"], constant_summary["violation_count"])
    ax.set_ylabel("Constant-dimension violations")
    ax.set_title("Constant training-dimension violations")
    ax.tick_params(axis="x", rotation=75)
    save_figure(fig, plot_dir / "10_constant_dimension_violations.png")

    subset = global_coverage[
        global_coverage["representation"].eq("R2_feature_family")
        & global_coverage["reference_role"].eq("same_class")
        & global_coverage["target_run_id"].isin([HARD_NORMAL, HARD_ATTACK])
    ]
    family_summary = (
        subset.groupby(["target_run_id", "feature_family"], as_index=False)[
            "outside_training_p01_p99"
        ].mean()
    )
    matrix = family_summary.pivot(
        index="target_run_id",
        columns="feature_family",
        values="outside_training_p01_p99",
    )
    plot_heatmap(
        matrix,
        "Feature-family same-class coverage",
        plot_dir / "11_feature_family_coverage_comparison.png",
        "Feature family",
        "Target run",
        vmin=0,
        vmax=1,
        figsize=(12, 5),
    )

    subset = saturation_coverage[
        saturation_coverage["reference_role"].eq("same_class")
    ]
    saturation_summary = (
        subset.groupby("target_run_id", as_index=False)
        .agg(
            outside_p01_p99_fraction=("outside_training_p01_p99", "mean"),
            maximum_absolute_robust_deviation=("absolute_robust_deviation", "max"),
        )
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        saturation_summary["target_run_id"],
        saturation_summary["outside_p01_p99_fraction"],
    )
    ax.set_ylabel("Exact-1.0 pattern outside p01–p99 fraction")
    ax.set_title("Saturation-pattern coverage")
    ax.tick_params(axis="x", rotation=75)
    save_figure(fig, plot_dir / "12_saturation_pattern_coverage.png")


def top_novel(
    coverage: pd.DataFrame,
    run_id: str,
    representation: str,
    count: int = 8,
) -> pd.DataFrame:
    subset = coverage[
        coverage["target_run_id"].eq(run_id)
        & coverage["representation"].eq(representation)
        & coverage["reference_role"].eq("same_class")
    ].copy()
    subset["ranking_value"] = subset["absolute_robust_deviation"].fillna(
        subset["outside_training_minimum_maximum"].astype(float) * 10.0
    )
    return subset.sort_values("ranking_value", ascending=False).head(count)


def build_report(
    coverage_by_representation: pd.DataFrame,
    global_coverage: pd.DataFrame,
    router_coverage: pd.DataFrame,
    temporal_coverage: pd.DataFrame,
    saturation_coverage: pd.DataFrame,
    hypothesis_evidence: pd.DataFrame,
    training_samples_scanned: int,
    selected_runs: list[str],
    plot_count: int,
) -> str:
    lines = [
        "# Stage 7C1 V2 — Training Distribution Coverage",
        "",
        "All fitted references were constructed using the training split only. ",
        "Validation and test targets were then compared against frozen full-training, same-class and opposite-class references.",
        "",
        "Nearest-run ranking, classifier fitting, inference and model training were not performed.",
        "",
        "## Training reference",
        "",
        f"- Training samples represented: **{training_samples_scanned}**.",
        "- Training runs represented: **45**.",
        f"- Selected validation/test targets evaluated: **{len(selected_runs)}**.",
        f"- Plots created: **{plot_count}**.",
        "",
        "## Interpretation rules",
        "",
        "- Min–max inclusion is not treated as sufficient evidence of coverage.",
        "- p01–p99, p05–p95, z-score, robust-IQR deviation and constant-dimension violations are reported separately.",
        "- Direction names are retained only as fixed feature labels until builder recovery confirms physical mapping.",
        "- Exact-1.0 pattern coverage can establish recurrence or novelty, not the encoding mechanism.",
        "",
        "## Dominant-run coverage summary",
        "",
        "| Target | Representation | Reference | Outside p01–p99 | Outside min–max | Median |z| | Median robust deviation |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    dominant = coverage_by_representation[
        coverage_by_representation["target_run_id"].isin([HARD_NORMAL, HARD_ATTACK])
    ].sort_values(["target_run_id", "representation", "reference_role"])
    for _, row in dominant.iterrows():
        lines.append(
            f"| `{row['target_run_id']}` | {row['representation']} | {row['reference_role']} | "
            f"{float(row['outside_p01_p99_fraction']):.4f} | "
            f"{float(row['outside_minimum_maximum_fraction']):.4f} | "
            f"{float(row['median_absolute_z_score']):.3f} | "
            f"{float(row['median_absolute_robust_deviation']):.3f} |"
        )

    lines.extend(["", "## Most novel same-class dimensions", ""])
    for run_id in [HARD_NORMAL, HARD_ATTACK]:
        lines.append(f"### `{run_id}`")
        for representation, coverage in [
            ("R1_global_run_feature", global_coverage),
            ("R3_router_feature", router_coverage),
            ("R4_temporal_position_feature", temporal_coverage),
        ]:
            lines.append(f"#### {representation}")
            for _, row in top_novel(coverage, run_id, representation).iterrows():
                location = []
                if int(row["router_id"]) >= 0:
                    location.append(f"router {int(row['router_id'])}")
                if int(row["temporal_position"]) >= 0:
                    location.append(f"position {int(row['temporal_position'])}")
                where = ", ".join(location) if location else "global"
                robust_value = row["absolute_robust_deviation"]
                robust_text = (
                    f"{float(robust_value):.3f}"
                    if np.isfinite(robust_value)
                    else "undefined (zero-IQR reference)"
                )
                label = str(row["feature_name"] or row["feature_family"])
                lines.append(
                    f"- `{label}` / `{row['metric']}` / {where}: target={float(row['target_value']):.5f}, "
                    f"training median={float(row['training_median']):.5f}, "
                    f"outside p01–p99={bool(row['outside_training_p01_p99'])}, "
                    f"|robust deviation|={robust_text}."
                )
        lines.append("")

    lines.extend(["## Suspicious router-12 exact-1.0 pattern", ""])
    suspicious = saturation_coverage[
        saturation_coverage["is_suspicious_attacker_router_pattern"]
        & saturation_coverage["reference_role"].eq("all_training")
    ].sort_values("feature_index")
    for _, row in suspicious.iterrows():
        lines.append(
            f"- `{row['feature_name']}`: target exact-1.0 rate={float(row['target_value']):.3f}; "
            f"training median={float(row['training_median']):.3f}; training p95={float(row['training_p95']):.3f}; "
            f"training runs with the same constant-one pattern="
            f"{int(row['training_runs_with_constant_exact_one_pattern'])}."
        )

    lines.extend([
        "",
        "## Hypothesis evidence",
        "",
        "| Hypothesis | Stage 7C1 status | Builder recovery required |",
        "|---|---|---:|",
    ])
    for _, row in hypothesis_evidence.iterrows():
        lines.append(
            f"| {row['hypothesis_id']} — {row['hypothesis_name']} | "
            f"{row['stage7c1_status']} | {row['builder_recovery_required']} |"
        )

    lines.extend([
        "",
        "## Boundary",
        "",
        "- Stage 7C1 establishes normalized-dimension coverage or novelty.",
        "- It does not establish nearest training runs or same/opposite-class distance margins.",
        "- It does not test joint count–IFD geometry.",
        "- It does not identify the meaning of exact 1.0 or zero-count values.",
        "- Builder and normalization recovery should proceed in parallel.",
        "",
        "## Next stage",
        "",
        "Stage 7C2 should use these frozen references to compute standardized nearest-training-run distances and same-class versus opposite-class margins.",
        "",
    ])
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def fractions_valid(frame: pd.DataFrame) -> bool:
    columns = [
        column
        for column in frame.columns
        if column.endswith("_fraction") or column.startswith("fraction_")
    ]
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) and ((finite < -1e-12).any() or (finite > 1.0 + 1e-12).any()):
            return False
    return True


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 7C1 V2: build training-only coverage references for V3 NoC features."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "~/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3"
        ),
    )
    parser.add_argument(
        "--failure-root",
        type=Path,
        default=Path("reports/v3_failure_analysis"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke-test-only",
        action="store_true",
        help=(
            "Run the synthetic indexed-reference application test and exit "
            "without reading the dataset."
        ),
    )
    args = parser.parse_args()

    run_reference_application_smoke_test()
    if args.smoke_test_only:
        print("smoke-test-only success: True", flush=True)
        return

    dataset_root = args.dataset_root.expanduser().resolve()
    failure_root = args.failure_root.resolve()
    table_dir = failure_root / "tables"
    log_dir = failure_root / "logs"
    plot_dir = failure_root / "plots" / "stage7c1_v2_training_coverage"
    staging_plot_dir = (
        failure_root
        / "plots"
        / f".stage7c1_v2_training_coverage_staging_{os.getpid()}"
    )
    if staging_plot_dir.exists():
        for staged_file in staging_plot_dir.glob("*"):
            staged_file.unlink(missing_ok=True)
        staging_plot_dir.rmdir()

    staging_output_root = (
        failure_root
        / f".stage7c1_v2_outputs_staging_{os.getpid()}"
    )
    if staging_output_root.exists():
        for staged_path in sorted(
            staging_output_root.rglob("*"),
            reverse=True,
        ):
            if staged_path.is_file():
                staged_path.unlink(missing_ok=True)
            elif staged_path.is_dir():
                staged_path.rmdir()
        staging_output_root.rmdir()

    x_path = dataset_root / "x.npy"
    alignment_path = table_dir / "stage7_sample_alignment.csv"
    run_inventory_path = table_dir / "stage7_run_inventory.csv"
    selected_inventory_path = table_dir / "stage7_selected_run_inventory.csv"
    taxonomy_path = table_dir / "stage7b_corrected_feature_taxonomy.csv"
    run_summary_path = table_dir / "stage7b_run_feature_summary.csv"
    router_summary_path = table_dir / "stage7b_selected_router_feature_summary.csv"
    temporal_summary_path = table_dir / "stage7b_selected_temporal_feature_summary.csv"
    spatial_summary_path = table_dir / "stage7b_selected_spatial_concentration.csv"
    stage7b_summary_path = table_dir / "stage7b_summary.json"

    outputs = {
        "training_run_reference": table_dir / "stage7c1_v2_training_run_reference.csv",
        "training_dimension_reference": table_dir / "stage7c1_v2_training_dimension_reference.csv",
        "training_histograms": table_dir / "stage7c1_v2_training_raw_feature_histograms.npz",
        "training_router_reference": table_dir / "stage7c1_v2_training_router_feature_reference.csv",
        "training_temporal_reference": table_dir / "stage7c1_v2_training_temporal_feature_reference.csv",
        "target_global_coverage": table_dir / "stage7c1_v2_target_global_coverage.csv",
        "target_router_coverage": table_dir / "stage7c1_v2_target_router_feature_coverage.csv",
        "target_temporal_coverage": table_dir / "stage7c1_v2_target_temporal_feature_coverage.csv",
        "target_spatial_coverage": table_dir / "stage7c1_v2_target_spatial_coverage.csv",
        "target_saturation_coverage": table_dir / "stage7c1_v2_target_saturation_pattern_coverage.csv",
        "target_raw_distribution_coverage": table_dir / "stage7c1_v2_target_raw_distribution_coverage.csv",
        "constant_violations": table_dir / "stage7c1_v2_constant_dimension_violations.csv",
        "coverage_by_run": table_dir / "stage7c1_v2_coverage_summary_by_run.csv",
        "coverage_by_representation": table_dir / "stage7c1_v2_coverage_summary_by_representation.csv",
        "training_cohort_reference": table_dir / "stage7c1_v2_training_cohort_reference.csv",
        "hypothesis_evidence": table_dir / "stage7c1_v2_hypothesis_evidence.csv",
        "scan_manifest": table_dir / "stage7c1_v2_scan_manifest.csv",
        "summary_json": table_dir / "stage7c1_v2_summary.json",
        "report": failure_root / "STAGE7C1_V2_TRAINING_COVERAGE_REPORT.md",
        "validation": log_dir / "40_stage7c1_v2_validation.txt",
    }

    staged_outputs: dict[str, Path] = {}
    for output_name, final_path in outputs.items():
        if final_path.parent == table_dir:
            relative_path = Path("tables") / final_path.name
        elif final_path.parent == log_dir:
            relative_path = Path("logs") / final_path.name
        elif final_path.parent == failure_root:
            relative_path = Path(final_path.name)
        else:
            raise RuntimeError(
                f"Unexpected final output parent for {final_path}."
            )
        staged_outputs[output_name] = (
            staging_output_root / relative_path
        )

    existing = [path for path in outputs.values() if path.exists()]
    if plot_dir.exists():
        existing.append(plot_dir)
    if existing and not args.overwrite:
        raise FileExistsError(
            "Stage 7C1 V2 outputs already exist. Use --overwrite to replace them:\n"
            + "\n".join(str(path) for path in existing)
        )

    required = [
        x_path,
        alignment_path,
        run_inventory_path,
        selected_inventory_path,
        taxonomy_path,
        run_summary_path,
        router_summary_path,
        temporal_summary_path,
        spatial_summary_path,
        stage7b_summary_path,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"script version: {SCRIPT_VERSION}", flush=True)
    print(f"dataset root: {dataset_root}", flush=True)
    print(f"failure root: {failure_root}", flush=True)
    print("training-only reference fitting: True", flush=True)
    print("nearest-run ranking: False", flush=True)
    print("classifier fitting: False", flush=True)
    print("node correctness grouping: False", flush=True)
    print("inference rerun: False", flush=True)
    print("training performed: False", flush=True)

    x = np.load(x_path, mmap_mode="r", allow_pickle=False)
    x_is_memmap = isinstance(x, np.memmap)
    x_shape_valid = tuple(x.shape) == (
        EXPECTED_TOTAL_SAMPLES,
        EXPECTED_ROUTERS,
        EXPECTED_TEMPORAL,
        EXPECTED_FEATURES,
    )
    if not x_is_memmap:
        raise RuntimeError("x.npy is not memory mapped.")
    if not x_shape_valid:
        raise RuntimeError(f"Unexpected x.npy shape: {x.shape}")

    alignment = pd.read_csv(
        alignment_path,
        keep_default_na=False,
        low_memory=False,
        dtype={"run_id": str, "split": str, "sample_id": str},
    )
    run_inventory = pd.read_csv(
        run_inventory_path,
        keep_default_na=False,
        low_memory=False,
        dtype={"run_id": str, "split": str},
    )
    selected_inventory = pd.read_csv(
        selected_inventory_path,
        keep_default_na=False,
        low_memory=False,
        dtype={"run_id": str, "split": str},
    )
    taxonomy = pd.read_csv(taxonomy_path, keep_default_na=False)
    run_summary = pd.read_csv(
        run_summary_path,
        keep_default_na=False,
        low_memory=False,
        dtype={"run_id": str, "split": str},
    )
    selected_router_summary = pd.read_csv(
        router_summary_path,
        keep_default_na=False,
        low_memory=False,
        dtype={"run_id": str, "split": str},
    )
    selected_temporal_summary = pd.read_csv(
        temporal_summary_path,
        keep_default_na=False,
        low_memory=False,
        dtype={"run_id": str, "split": str},
    )
    selected_spatial_summary = pd.read_csv(
        spatial_summary_path,
        keep_default_na=False,
        low_memory=False,
        dtype={"run_id": str, "split": str},
    )
    stage7b_summary = json.loads(stage7b_summary_path.read_text())
    if not stage7b_summary.get("overall_success", False):
        raise RuntimeError("Stage 7B did not report overall success.")

    taxonomy_valid = (
        len(taxonomy) == EXPECTED_FEATURES
        and taxonomy["feature_index"].tolist() == list(range(EXPECTED_FEATURES))
        and taxonomy["taxonomy_source"].eq(
            "fixed_stage7b_corrected_taxonomy"
        ).all()
    )
    if not taxonomy_valid:
        raise RuntimeError("Stage 7B taxonomy mismatch.")

    metadata_lookup = build_metadata_lookup(run_inventory)
    feature_lookup = build_feature_lookup(taxonomy)

    training_alignment = alignment[alignment["split"].eq("train")].copy()
    nontraining_alignment = alignment[~alignment["split"].eq("train")].copy()
    training_run_ids = sorted(training_alignment["run_id"].unique())
    selected_run_ids = sorted(selected_inventory["run_id"].unique())

    train_sample_count_valid = len(training_alignment) == EXPECTED_TRAIN_SAMPLES
    train_run_count_valid = len(training_run_ids) == EXPECTED_TRAIN_RUNS
    selected_runs_valid = len(selected_run_ids) == 10
    no_test_val_in_fit = (
        set(training_alignment["split"]) == {"train"}
        and not set(training_alignment["dataset_index"]).intersection(
            set(nontraining_alignment["dataset_index"])
        )
    )
    if not all(
        [
            train_sample_count_valid,
            train_run_count_valid,
            selected_runs_valid,
            no_test_val_in_fit,
        ]
    ):
        raise RuntimeError("Training/target split validation failed.")

    all_r1_r2 = build_r1_r2_dimensions(
        run_summary, metadata_lookup, feature_lookup
    )
    training_r1_r2 = all_r1_r2[all_r1_r2["split"].eq("train")].copy()
    target_r1_r2 = all_r1_r2[
        all_r1_r2["run_id"].isin(selected_run_ids)
    ].copy()

    target_r3 = build_selected_r3_dimensions(
        selected_router_summary[
            selected_router_summary["run_id"].isin(selected_run_ids)
        ],
        metadata_lookup,
        feature_lookup,
    )
    target_r4 = build_selected_r4_dimensions(
        selected_temporal_summary[
            selected_temporal_summary["run_id"].isin(selected_run_ids)
        ],
        metadata_lookup,
        feature_lookup,
    )
    target_r5 = build_selected_r5_dimensions(
        selected_spatial_summary[
            selected_spatial_summary["run_id"].isin(selected_run_ids)
        ],
        metadata_lookup,
        feature_lookup,
    )

    training_indices = {
        run_id: group.sort_values("run_window_index")["dataset_index"].to_numpy(
            dtype=np.int64
        )
        for run_id, group in training_alignment.groupby("run_id", sort=True)
    }

    histograms = initialize_histograms()
    training_detailed_rows: list[dict[str, Any]] = []
    scan_rows: list[dict[str, Any]] = []
    training_samples_scanned = 0
    all_training_finite = True
    values_outside_unit_interval = False

    for number, run_id in enumerate(training_run_ids, start=1):
        indices = training_indices[run_id]
        if len(indices) != EXPECTED_WINDOWS_PER_RUN:
            raise RuntimeError(
                f"Training run {run_id} has {len(indices)} windows."
            )
        run_data = np.asarray(x[indices], dtype=np.float32)
        expected_shape = (
            EXPECTED_WINDOWS_PER_RUN,
            EXPECTED_ROUTERS,
            EXPECTED_TEMPORAL,
            EXPECTED_FEATURES,
        )
        if tuple(run_data.shape) != expected_shape:
            raise RuntimeError(
                f"Training run {run_id} has unexpected shape {run_data.shape}."
            )

        finite = bool(np.isfinite(run_data).all())
        outside_unit = bool(np.any(run_data < 0.0) or np.any(run_data > 1.0))
        all_training_finite = all_training_finite and finite
        values_outside_unit_interval = values_outside_unit_interval or outside_unit
        if not finite:
            raise RuntimeError(f"Non-finite feature in {run_id}.")

        metadata = metadata_lookup[run_id]
        training_detailed_rows.extend(
            summarize_training_run(run_data, metadata, feature_lookup)
        )
        update_all_histograms(histograms, run_data, metadata["true_graph"])
        training_samples_scanned += len(indices)

        scan_rows.append(
            {
                "training_scan_order": number,
                "run_id": run_id,
                "split": "train",
                "true_graph": metadata["true_graph"],
                "profile": metadata["profile"],
                "number_of_windows": len(indices),
                "dataset_index_minimum": int(indices.min()),
                "dataset_index_maximum": int(indices.max()),
                "run_array_shape": "x".join(map(str, run_data.shape)),
                "run_array_bytes": int(run_data.nbytes),
                "all_values_finite": finite,
                "contains_value_outside_unit_interval": outside_unit,
                "used_for_reference_fitting": True,
                "full_tensor_in_memory": False,
            }
        )
        print(
            f"[{number:02d}/{len(training_run_ids)}] {run_id} | "
            f"class={metadata['true_graph']} | outside_unit_interval={outside_unit}",
            flush=True,
        )
        del run_data

    training_detailed = pd.DataFrame(training_detailed_rows)
    scan_manifest = pd.DataFrame(scan_rows)
    training_dimensions = pd.concat(
        [training_r1_r2, training_detailed], ignore_index=True
    )
    target_dimensions = pd.concat(
        [target_r1_r2, target_r3, target_r4, target_r5], ignore_index=True
    )

    print("constructing training-only references", flush=True)
    summary_references = build_references(training_dimensions)
    raw_references = build_raw_references(
        histograms, taxonomy, training_dimensions
    )
    references = pd.concat(
        [summary_references, raw_references],
        ignore_index=True,
        sort=False,
    )
    descriptive_cohorts = build_descriptive_cohort_reference(training_dimensions)

    print("applying frozen references to selected targets", flush=True)
    target_coverage = apply_references(
        target_dimensions, summary_references
    )
    target_raw_coverage = build_target_raw_coverage(
        x,
        selected_run_ids,
        alignment,
        metadata_lookup,
        raw_references,
    )

    global_coverage = target_coverage[
        target_coverage["representation"].isin(
            ["R1_global_run_feature", "R2_feature_family"]
        )
    ].copy()
    router_coverage = target_coverage[
        target_coverage["representation"].eq("R3_router_feature")
    ].copy()
    temporal_coverage = target_coverage[
        target_coverage["representation"].eq(
            "R4_temporal_position_feature"
        )
    ].copy()
    spatial_coverage = target_coverage[
        target_coverage["representation"].eq("R5_spatial_concentration")
    ].copy()

    training_router_reference = references[
        references["representation"].eq("R3_router_feature")
    ].copy()
    training_temporal_reference = references[
        references["representation"].eq("R4_temporal_position_feature")
    ].copy()

    saturation_coverage = build_saturation_pattern_coverage(
        target_r3,
        training_detailed[
            training_detailed["representation"].eq("R3_router_feature")
        ],
        router_coverage,
    )

    constant_violations = target_coverage[
        target_coverage["constant_dimension_violation"]
        | target_coverage["zero_iqr_dimension_violation"]
    ].copy()

    coverage_by_run = summarize_coverage(
        target_coverage,
        [
            "target_run_id",
            "target_split",
            "target_true_graph",
            "reference_cohort",
            "reference_role",
        ],
    )
    coverage_by_run["coverage_scope"] = "summary_dimensions"
    coverage_by_representation = summarize_coverage(
        target_coverage,
        [
            "target_run_id",
            "target_split",
            "target_true_graph",
            "reference_cohort",
            "reference_role",
            "representation",
        ],
    )
    raw_coverage_by_representation = (
        summarize_raw_coverage_by_representation(
            target_raw_coverage
        )
    )
    coverage_by_representation = pd.concat(
        [
            coverage_by_representation,
            raw_coverage_by_representation,
        ],
        ignore_index=True,
        sort=False,
    )
    raw_coverage_by_run = (
        raw_coverage_by_representation.groupby(
            [
                "target_run_id",
                "target_split",
                "target_true_graph",
                "reference_cohort",
                "reference_role",
            ],
            as_index=False,
        )
        .agg(
            dimension_count=("dimension_count", "sum"),
            outside_minimum_maximum_count=(
                "outside_minimum_maximum_count",
                "sum",
            ),
            outside_minimum_maximum_fraction=(
                "outside_minimum_maximum_fraction",
                "mean",
            ),
            outside_p01_p99_count=(
                "outside_p01_p99_count",
                "sum",
            ),
            outside_p01_p99_fraction=(
                "outside_p01_p99_fraction",
                "mean",
            ),
            outside_p05_p95_count=(
                "outside_p05_p95_count",
                "sum",
            ),
            outside_p05_p95_fraction=(
                "outside_p05_p95_fraction",
                "mean",
            ),
            absolute_z_gt_2_count=("absolute_z_gt_2_count", "sum"),
            absolute_z_gt_2_fraction=("absolute_z_gt_2_fraction", "mean"),
            absolute_z_gt_3_count=("absolute_z_gt_3_count", "sum"),
            absolute_z_gt_3_fraction=("absolute_z_gt_3_fraction", "mean"),
            absolute_z_gt_5_count=("absolute_z_gt_5_count", "sum"),
            absolute_z_gt_5_fraction=("absolute_z_gt_5_fraction", "mean"),
            constant_dimension_violation_count=(
                "constant_dimension_violation_count",
                "sum",
            ),
            zero_iqr_dimension_violation_count=(
                "zero_iqr_dimension_violation_count",
                "sum",
            ),
            finite_z_score_count=("finite_z_score_count", "sum"),
            median_absolute_z_score=("median_absolute_z_score", "median"),
            maximum_absolute_z_score=("maximum_absolute_z_score", "max"),
            finite_robust_deviation_count=(
                "finite_robust_deviation_count",
                "sum",
            ),
            median_absolute_robust_deviation=(
                "median_absolute_robust_deviation",
                "median",
            ),
            maximum_absolute_robust_deviation=(
                "maximum_absolute_robust_deviation",
                "max",
            ),
        )
    )
    raw_coverage_by_run["coverage_scope"] = (
        "raw_distribution_dimensions"
    )
    coverage_by_run = pd.concat(
        [coverage_by_run, raw_coverage_by_run],
        ignore_index=True,
        sort=False,
    )
    hypothesis_evidence = build_hypothesis_evidence(
        coverage_by_representation, saturation_coverage
    )

    global_expected = EXPECTED_TRAIN_SAMPLES * EXPECTED_ROUTERS * EXPECTED_TEMPORAL
    router_expected = EXPECTED_TRAIN_SAMPLES * EXPECTED_TEMPORAL
    temporal_expected = EXPECTED_TRAIN_SAMPLES * EXPECTED_ROUTERS

    all_index = COHORT_INDEX["all_training"]
    global_totals = (
        histograms["global_feature_counts"][all_index].sum(axis=1)
        + histograms["global_feature_underflow"][all_index]
        + histograms["global_feature_overflow"][all_index]
    )
    router_totals = (
        histograms["router_feature_counts"][all_index].sum(axis=2)
        + histograms["router_feature_underflow"][all_index]
        + histograms["router_feature_overflow"][all_index]
    )
    temporal_totals = (
        histograms["temporal_feature_counts"][all_index].sum(axis=2)
        + histograms["temporal_feature_underflow"][all_index]
        + histograms["temporal_feature_overflow"][all_index]
    )

    histogram_global_valid = bool(np.all(global_totals == global_expected))
    histogram_router_valid = bool(np.all(router_totals == router_expected))
    histogram_temporal_valid = bool(np.all(temporal_totals == temporal_expected))

    make_plots(
        coverage_by_representation,
        global_coverage,
        router_coverage,
        temporal_coverage,
        saturation_coverage,
        constant_violations,
        staging_plot_dir,
    )
    plot_count = len(list(staging_plot_dir.glob("*.png")))

    family_count = taxonomy["feature_family"].nunique()
    dims_r1 = EXPECTED_FEATURES * len(R1_METRICS)
    dims_r2 = family_count * len(R1_METRICS)
    dims_r3 = EXPECTED_ROUTERS * EXPECTED_FEATURES * len(R3_METRICS)
    dims_r4 = EXPECTED_TEMPORAL * EXPECTED_FEATURES * len(R4_METRICS)
    dims_r5 = EXPECTED_FEATURES * len(R5_METRICS)
    dims_per_run = dims_r1 + dims_r2 + dims_r3 + dims_r4 + dims_r5

    training_dimensions_complete = (
        len(training_dimensions) == EXPECTED_TRAIN_RUNS * dims_per_run
        and training_dimensions.groupby("run_id")["dimension_id"]
        .nunique()
        .eq(dims_per_run)
        .all()
    )
    target_dimensions_complete = (
        len(target_dimensions) == len(selected_run_ids) * dims_per_run
        and target_dimensions.groupby("run_id")["dimension_id"]
        .nunique()
        .eq(dims_per_run)
        .all()
    )
    references_complete = (
        summary_references["dimension_id"].nunique() == dims_per_run
        and summary_references.groupby("reference_cohort")["dimension_id"]
        .nunique()
        .eq(dims_per_run)
        .all()
    )
    raw_dimensions_per_cohort = (
        EXPECTED_FEATURES
        + EXPECTED_ROUTERS * EXPECTED_FEATURES
        + EXPECTED_TEMPORAL * EXPECTED_FEATURES
    )
    raw_references_complete = (
        raw_references.groupby("reference_cohort")["dimension_id"]
        .nunique()
        .eq(raw_dimensions_per_cohort)
        .all()
    )
    target_raw_coverage_complete = (
        len(target_raw_coverage)
        == len(selected_run_ids)
        * raw_dimensions_per_cohort
        * len(PRIMARY_COHORTS)
    )
    target_coverage_complete = (
        len(target_coverage)
        == len(selected_run_ids) * dims_per_run * len(PRIMARY_COHORTS)
    )
    required_cohort_set = set(PRIMARY_COHORTS)
    summary_reference_cohorts_attached = bool(
        target_coverage.groupby(
            ["target_run_id", "dimension_id"]
        )["reference_cohort"]
        .agg(lambda values: set(values.astype(str)) == required_cohort_set)
        .all()
        and target_coverage.groupby(
            ["target_run_id", "dimension_id"]
        ).size().eq(len(PRIMARY_COHORTS)).all()
    )
    raw_reference_cohorts_attached = bool(
        target_raw_coverage.groupby(
            ["target_run_id", "dimension_id"]
        )["reference_cohort"]
        .agg(lambda values: set(values.astype(str)) == required_cohort_set)
        .all()
        and target_raw_coverage.groupby(
            ["target_run_id", "dimension_id"]
        ).size().eq(len(PRIMARY_COHORTS)).all()
    )
    all_reference_cohorts_attached = bool(
        summary_reference_cohorts_attached
        and raw_reference_cohorts_attached
    )
    provenance_complete = references["provenance_table"].astype(str).str.len().gt(0).all()
    zero_variance_safe = target_coverage.loc[
        target_coverage["zero_variance_dimension"], "z_score"
    ].isna().all()
    coverage_fractions_valid = fractions_valid(coverage_by_run) and fractions_valid(
        coverage_by_representation
    )
    finite_deviations = (
        np.isfinite(
            target_coverage.loc[
                target_coverage["z_score"].notna(), "z_score"
            ].to_numpy(dtype=float)
        ).all()
        and np.isfinite(
            target_coverage.loc[
                target_coverage["robust_deviation"].notna(),
                "robust_deviation",
            ].to_numpy(dtype=float)
        ).all()
    )
    selected_targets_present = set(target_dimensions["run_id"]) == set(selected_run_ids)
    training_only_reference = set(training_dimensions["split"]) == {"train"}
    training_samples_scanned_valid = training_samples_scanned == EXPECTED_TRAIN_SAMPLES
    histogram_counts_valid = (
        histogram_global_valid and histogram_router_valid and histogram_temporal_valid
    )

    overall_success = all(
        [
            x_is_memmap,
            x_shape_valid,
            taxonomy_valid,
            train_sample_count_valid,
            train_run_count_valid,
            selected_runs_valid,
            no_test_val_in_fit,
            training_only_reference,
            training_samples_scanned_valid,
            all_training_finite,
            not values_outside_unit_interval,
            training_dimensions_complete,
            target_dimensions_complete,
            references_complete,
            raw_references_complete,
            target_coverage_complete,
            target_raw_coverage_complete,
            all_reference_cohorts_attached,
            provenance_complete,
            zero_variance_safe,
            coverage_fractions_valid,
            finite_deviations,
            selected_targets_present,
            histogram_counts_valid,
        ]
    )

    validation_lines = [
        "reference application smoke test passed: True",
        f"raw feature tensor memory mapped: {x_is_memmap}",
        f"raw feature tensor shape verified: {x_shape_valid}",
        f"feature taxonomy matches Stage 7B: {taxonomy_valid}",
        f"training reference uses only split=train: {training_only_reference}",
        f"all 45 training runs included: {train_run_count_valid}",
        f"all 148185 training samples represented: {training_samples_scanned_valid}",
        f"no validation/test samples included in fitting: {no_test_val_in_fit}",
        f"all selected target runs present: {selected_targets_present}",
        f"training run dimensions complete: {training_dimensions_complete}",
        f"selected target dimensions complete: {target_dimensions_complete}",
        f"all training-reference dimensions present: {references_complete}",
        f"all raw histogram-reference dimensions present: {raw_references_complete}",
        f"all selected-target raw coverage dimensions present: {target_raw_coverage_complete}",
        f"all three reference cohorts attached to every target dimension: {all_reference_cohorts_attached}",
        f"all training-reference dimensions have provenance: {provenance_complete}",
        f"zero-variance dimensions handled safely: {zero_variance_safe}",
        f"all coverage fractions lie in [0,1]: {coverage_fractions_valid}",
        f"all finite deviations are finite: {finite_deviations}",
        f"global histogram counts equal expected training values: {histogram_global_valid}",
        f"router-feature histogram counts equal expected training values: {histogram_router_valid}",
        f"temporal-feature histogram counts equal expected training values: {histogram_temporal_valid}",
        f"training values outside [0,1] observed: {values_outside_unit_interval}",
        f"fixed [0,1] histogram grid valid: {not values_outside_unit_interval}",
        "classifier fitted: False",
        "nearest-run ranking performed: False",
        "node correctness grouping performed: False",
        "inference rerun: False",
        "training performed: False",
        f"overall success: {overall_success}",
    ]
    validation_text = "\n".join(validation_lines) + "\n"
    print()
    print(validation_text, end="", flush=True)

    if not overall_success:
        for staged_file in staging_plot_dir.glob("*"):
            staged_file.unlink(missing_ok=True)
        staging_plot_dir.rmdir()
        if staging_output_root.exists():
            for staged_path in sorted(
                staging_output_root.rglob("*"),
                reverse=True,
            ):
                if staged_path.is_file():
                    staged_path.unlink(missing_ok=True)
                elif staged_path.is_dir():
                    staged_path.rmdir()
            staging_output_root.rmdir()
        raise SystemExit(
            "Stage 7C1 V2 validation failed; no final V2 outputs were "
            "promoted. Inspect the tee log for validation details."
        )

    # Build every non-plot artifact in a hidden staging tree first.
    # Final V2 paths are promoted only after all staged writes succeed.
    atomic_text(validation_text, staged_outputs["validation"])
    atomic_csv(
        training_dimensions,
        staged_outputs["training_run_reference"],
    )
    atomic_csv(
        references,
        staged_outputs["training_dimension_reference"],
    )
    atomic_csv(
        training_router_reference,
        staged_outputs["training_router_reference"],
    )
    atomic_csv(
        training_temporal_reference,
        staged_outputs["training_temporal_reference"],
    )
    atomic_csv(
        global_coverage,
        staged_outputs["target_global_coverage"],
    )
    atomic_csv(
        router_coverage,
        staged_outputs["target_router_coverage"],
    )
    atomic_csv(
        temporal_coverage,
        staged_outputs["target_temporal_coverage"],
    )
    atomic_csv(
        spatial_coverage,
        staged_outputs["target_spatial_coverage"],
    )
    atomic_csv(
        saturation_coverage,
        staged_outputs["target_saturation_coverage"],
    )
    atomic_csv(
        target_raw_coverage,
        staged_outputs["target_raw_distribution_coverage"],
    )
    atomic_csv(
        constant_violations,
        staged_outputs["constant_violations"],
    )
    atomic_csv(
        coverage_by_run,
        staged_outputs["coverage_by_run"],
    )
    atomic_csv(
        coverage_by_representation,
        staged_outputs["coverage_by_representation"],
    )
    atomic_csv(
        descriptive_cohorts,
        staged_outputs["training_cohort_reference"],
    )
    atomic_csv(
        hypothesis_evidence,
        staged_outputs["hypothesis_evidence"],
    )
    atomic_csv(
        scan_manifest,
        staged_outputs["scan_manifest"],
    )

    atomic_npz(
        staged_outputs["training_histograms"],
        **histograms,
        histogram_bin_count=np.asarray([HIST_BINS], dtype=np.int64),
        training_sample_count=np.asarray([EXPECTED_TRAIN_SAMPLES], dtype=np.int64),
        training_run_count=np.asarray([EXPECTED_TRAIN_RUNS], dtype=np.int64),
        feature_names=np.asarray(
            taxonomy.sort_values("feature_index")["feature_name"].tolist()
        ),
        histogram_scope=np.asarray(
            ["training_only_global_feature_router_feature_temporal_feature"]
        ),
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "dataset_root": str(dataset_root),
        "failure_root": str(failure_root),
        "training_sample_count": EXPECTED_TRAIN_SAMPLES,
        "training_run_count": EXPECTED_TRAIN_RUNS,
        "selected_target_runs": selected_run_ids,
        "selected_target_run_count": len(selected_run_ids),
        "dimensions_per_run": dims_per_run,
        "training_dimension_rows": len(training_dimensions),
        "training_reference_rows": len(references),
        "target_coverage_rows": len(target_coverage),
        "target_raw_distribution_coverage_rows": len(target_raw_coverage),
        "raw_reference_rows": len(raw_references),
        "all_reference_cohorts_attached": all_reference_cohorts_attached,
        "reference_application_smoke_test_passed": True,
        "coverage_by_run_rows": len(coverage_by_run),
        "coverage_by_representation_rows": len(coverage_by_representation),
        "constant_dimension_violation_rows": len(constant_violations),
        "plot_count": plot_count,
        "histogram_bin_count": HIST_BINS,
        "training_values_outside_unit_interval_observed": values_outside_unit_interval,
        "builder_recovery_required": True,
        "builder_recovery_should_proceed_in_parallel": True,
        "classifier_fitted": False,
        "nearest_run_ranking_performed": False,
        "node_correctness_grouping_performed": False,
        "inference_rerun": False,
        "training_performed": False,
        "overall_success": overall_success,
    }
    atomic_json(summary, staged_outputs["summary_json"])

    report = build_report(
        coverage_by_representation,
        global_coverage,
        router_coverage,
        temporal_coverage,
        saturation_coverage,
        hypothesis_evidence,
        training_samples_scanned,
        selected_run_ids,
        plot_count,
    )
    atomic_text(report, staged_outputs["report"])

    # All staged writes succeeded. Promote files and plots to final paths.
    for output_name, staged_path in staged_outputs.items():
        final_path = outputs[output_name]
        final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_path, final_path)

    if plot_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Final plot directory already exists: {plot_dir}. "
                "Use --overwrite for an intentional replacement."
            )
        for old_plot in plot_dir.glob("*"):
            old_plot.unlink(missing_ok=True)
        plot_dir.rmdir()
    os.replace(staging_plot_dir, plot_dir)

    # Remove now-empty staging directories.
    for staged_dir in sorted(
        [path for path in staging_output_root.rglob("*") if path.is_dir()],
        reverse=True,
    ):
        staged_dir.rmdir()
    staging_output_root.rmdir()

    print("Stage 7C1 V2 output files:", flush=True)
    for name, path in outputs.items():
        print(f"  {name}: {path}", flush=True)
    print(f"  plots: {plot_dir}", flush=True)
    print("overall success: True", flush=True)


if __name__ == "__main__":
    main()
