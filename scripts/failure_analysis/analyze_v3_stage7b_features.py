#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_FEATURE_NAMES = [
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

DIRECTIONS = ["local", "north", "east", "south", "west"]

HARD_NORMAL_RUN = "N-3-7-8-12-Pmixed-R18-V3"
NORMAL_TEST_CONTROL = "N-1-6-9-14-Pmixed-R17-V3"
NORMAL_VAL_CONTROL = "N-2-6-10-14-Pmixed-R13-V3"

HARD_ATTACK_RUN = "N-5-10-Pbursty-R51-A-12-S20-V3"
ATTACK_STREAM_CONTROL = "N-2-13-Pstream-R50-A-12-S20-V3"
ATTACK_MIXED_CONTROL = "N-1-6-9-14-Pmixed-R52-A-12-S20-V3"
ATTACK_BURSTY_S20_CONTROL = "N-4-15-Pbursty-R41-A-11-S20-V3"
ATTACK_S63_CONTROL = "N-5-10-Pbursty-R51-A-1-7-S63-V3"
ATTACK_S77_CONTROL = "N-5-10-Pbursty-R51-A-1-11-12-S77-V3"
NORMAL_BURSTY_BACKGROUND = "N-5-10-Pbursty-R16-V3"

PAIR_DEFINITIONS = [
    (
        "A_hard_normal_exact_test",
        "hard_normal_minus_exact_normal",
        HARD_NORMAL_RUN,
        NORMAL_TEST_CONTROL,
    ),
    (
        "B_hard_normal_validation_analogue",
        "hard_normal_minus_validation_normal",
        HARD_NORMAL_RUN,
        NORMAL_VAL_CONTROL,
    ),
    (
        "C_hard_attack_exact_A12_S20",
        "hard_attack_minus_stream_A12_S20",
        HARD_ATTACK_RUN,
        ATTACK_STREAM_CONTROL,
    ),
    (
        "C_hard_attack_exact_A12_S20",
        "hard_attack_minus_mixed_A12_S20",
        HARD_ATTACK_RUN,
        ATTACK_MIXED_CONTROL,
    ),
    (
        "D_hard_attack_bursty_S20",
        "hard_attack_minus_bursty_S20_control",
        HARD_ATTACK_RUN,
        ATTACK_BURSTY_S20_CONTROL,
    ),
    (
        "E_same_bursty_background_severity",
        "S20_minus_S63",
        HARD_ATTACK_RUN,
        ATTACK_S63_CONTROL,
    ),
    (
        "E_same_bursty_background_severity",
        "S20_minus_S77",
        HARD_ATTACK_RUN,
        ATTACK_S77_CONTROL,
    ),
    (
        "F_broad_bursty_background",
        "S20_attack_minus_normal_bursty_background",
        HARD_ATTACK_RUN,
        NORMAL_BURSTY_BACKGROUND,
    ),
]

QUANTILES = {
    "p01": 0.01,
    "p05": 0.05,
    "p25": 0.25,
    "median": 0.50,
    "p75": 0.75,
    "p95": 0.95,
    "p99": 0.99,
}

DISTRIBUTION_SAMPLE_LIMIT = 100_000
UNIQUE_SAMPLE_LIMIT = 100_000
HISTOGRAM_BINS = 100
TEMPORAL_DEVIATION_THRESHOLD = 0.10
EPSILON = 1e-12


# ---------------------------------------------------------------------------
# Atomic output helpers
# ---------------------------------------------------------------------------

def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
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


def atomic_json(data: dict[str, Any], path: Path) -> None:
    atomic_text(json.dumps(data, indent=2, sort_keys=True) + "\n", path)


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
# Feature taxonomy
# ---------------------------------------------------------------------------

def corrected_feature_taxonomy() -> pd.DataFrame:
    rows = []

    for index, name in enumerate(EXPECTED_FEATURE_NAMES):
        if index in {0, 1}:
            family = "aggregate_ifd"
            measurement = "ifd"
            flow_side = "input" if index == 0 else "output"
            direction = ""
        elif index in {2, 3}:
            family = "aggregate_flit_count"
            measurement = "flit_count"
            flow_side = "input" if index == 2 else "output"
            direction = ""
        elif 4 <= index <= 8:
            family = "directional_input_flit_count"
            measurement = "flit_count"
            flow_side = "input"
            direction = DIRECTIONS[index - 4]
        elif 9 <= index <= 13:
            family = "directional_output_flit_count"
            measurement = "flit_count"
            flow_side = "output"
            direction = DIRECTIONS[index - 9]
        elif 14 <= index <= 18:
            family = "directional_input_ifd"
            measurement = "ifd"
            flow_side = "input"
            direction = DIRECTIONS[index - 14]
        else:
            family = "directional_output_ifd"
            measurement = "ifd"
            flow_side = "output"
            direction = DIRECTIONS[index - 19]

        rows.append(
            {
                "feature_index": index,
                "feature_name": name,
                "feature_family": family,
                "measurement_type": measurement,
                "flow_side": flow_side,
                "direction": direction,
                "is_directional": bool(direction),
                "taxonomy_source": "fixed_stage7b_corrected_taxonomy",
            }
        )

    return pd.DataFrame(rows)


def verify_feature_order(feature_cols: np.ndarray) -> list[str]:
    actual = [str(value) for value in np.asarray(feature_cols).reshape(-1)]
    if actual != EXPECTED_FEATURE_NAMES:
        raise RuntimeError(
            "feature_cols.npy does not match the fixed Stage 7B taxonomy.\n"
            f"Expected: {EXPECTED_FEATURE_NAMES}\n"
            f"Actual:   {actual}"
        )
    return actual


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def deterministic_subsample(
    values: np.ndarray,
    limit: int,
) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if len(flat) <= limit:
        return flat
    positions = np.linspace(0, len(flat) - 1, limit, dtype=np.int64)
    return flat[positions]


def safe_fraction(mask: np.ndarray) -> float:
    return float(np.mean(np.asarray(mask, dtype=bool)))


def basic_summary(
    values: np.ndarray,
    include_unique: bool = True,
) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(flat)
    finite = flat[finite_mask]

    if len(finite) == 0:
        return {
            "number_of_values": int(len(flat)),
            "finite_value_count": 0,
            "mean": np.nan,
            "std": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            **{name: np.nan for name in QUANTILES},
            "fraction_equal_0": np.nan,
            "fraction_equal_1": np.nan,
            "fraction_below_0_05": np.nan,
            "fraction_above_0_95": np.nan,
            "fraction_above_0_99": np.nan,
            "fraction_outside_unit_interval": np.nan,
            "unique_value_count": 0,
            "unique_count_is_sampled": False,
            "unique_count_sample_size": 0,
        }

    quantile_values = np.quantile(
        finite,
        list(QUANTILES.values()),
    )

    if include_unique:
        unique_source = deterministic_subsample(
            finite,
            UNIQUE_SAMPLE_LIMIT,
        )
        unique_count = int(np.unique(unique_source).size)
        unique_sampled = len(finite) > UNIQUE_SAMPLE_LIMIT
        unique_sample_size = int(len(unique_source))
    else:
        unique_count = -1
        unique_sampled = False
        unique_sample_size = 0

    return {
        "number_of_values": int(len(flat)),
        "finite_value_count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        **{
            name: float(value)
            for name, value in zip(QUANTILES, quantile_values)
        },
        "fraction_equal_0": safe_fraction(finite == 0.0),
        "fraction_equal_1": safe_fraction(finite == 1.0),
        "fraction_below_0_05": safe_fraction(finite < 0.05),
        "fraction_above_0_95": safe_fraction(finite > 0.95),
        "fraction_above_0_99": safe_fraction(finite > 0.99),
        "fraction_outside_unit_interval": safe_fraction(
            (finite < 0.0) | (finite > 1.0)
        ),
        "unique_value_count": unique_count,
        "unique_count_is_sampled": unique_sampled,
        "unique_count_sample_size": unique_sample_size,
    }


def pooled_standardized_mean_difference(
    left_mean: float,
    right_mean: float,
    left_std: float,
    right_std: float,
) -> float:
    denominator = math.sqrt(
        max(0.0, (left_std ** 2 + right_std ** 2) / 2.0)
    )
    if denominator <= EPSILON:
        return 0.0
    return float((left_mean - right_mean) / denominator)


def histogram_metrics(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any]:
    left_sample = deterministic_subsample(
        np.asarray(left, dtype=np.float64),
        DISTRIBUTION_SAMPLE_LIMIT,
    )
    right_sample = deterministic_subsample(
        np.asarray(right, dtype=np.float64),
        DISTRIBUTION_SAMPLE_LIMIT,
    )

    combined_min = float(min(np.min(left_sample), np.min(right_sample)))
    combined_max = float(max(np.max(left_sample), np.max(right_sample)))

    if math.isclose(combined_min, combined_max):
        return {
            "distribution_sample_count_left": int(len(left_sample)),
            "distribution_sample_count_right": int(len(right_sample)),
            "distribution_sampled": bool(
                len(np.asarray(left).reshape(-1)) > len(left_sample)
                or len(np.asarray(right).reshape(-1)) > len(right_sample)
            ),
            "histogram_overlap": 1.0,
            "ks_statistic": 0.0,
            "jensen_shannon_divergence": 0.0,
            "histogram_minimum": combined_min,
            "histogram_maximum": combined_max,
        }

    edges = np.linspace(
        combined_min,
        combined_max,
        HISTOGRAM_BINS + 1,
    )
    left_hist, _ = np.histogram(left_sample, bins=edges)
    right_hist, _ = np.histogram(right_sample, bins=edges)

    left_mass = left_hist.astype(np.float64)
    right_mass = right_hist.astype(np.float64)
    left_mass /= max(1.0, left_mass.sum())
    right_mass /= max(1.0, right_mass.sum())

    overlap = float(np.minimum(left_mass, right_mass).sum())

    left_cdf = np.cumsum(left_mass)
    right_cdf = np.cumsum(right_mass)
    ks = float(np.max(np.abs(left_cdf - right_cdf)))

    midpoint = 0.5 * (left_mass + right_mass)

    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(
            np.sum(p[mask] * np.log2(p[mask] / np.maximum(q[mask], EPSILON)))
        )

    js = 0.5 * kl_divergence(left_mass, midpoint)
    js += 0.5 * kl_divergence(right_mass, midpoint)

    return {
        "distribution_sample_count_left": int(len(left_sample)),
        "distribution_sample_count_right": int(len(right_sample)),
        "distribution_sampled": bool(
            len(np.asarray(left).reshape(-1)) > len(left_sample)
            or len(np.asarray(right).reshape(-1)) > len(right_sample)
        ),
        "histogram_overlap": overlap,
        "ks_statistic": ks,
        "jensen_shannon_divergence": float(js),
        "histogram_minimum": combined_min,
        "histogram_maximum": combined_max,
    }


def normalized_shares(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.minimum(values.min(axis=1, keepdims=True), 0.0)
    sums = shifted.sum(axis=1, keepdims=True)

    sorted_values = np.sort(shifted, axis=1)
    maximum = sorted_values[:, -1]
    top_two = sorted_values[:, -2:].sum(axis=1)

    maximum_share = np.divide(
        maximum,
        sums[:, 0],
        out=np.zeros_like(maximum),
        where=sums[:, 0] > EPSILON,
    )
    top_two_share = np.divide(
        top_two,
        sums[:, 0],
        out=np.zeros_like(top_two),
        where=sums[:, 0] > EPSILON,
    )
    return maximum_share, top_two_share


def entropy_from_fractions(fractions: Iterable[float]) -> float:
    probabilities = np.asarray(list(fractions), dtype=np.float64)
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) == 0:
        return 0.0
    return float(-np.sum(probabilities * np.log2(probabilities)))


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def parse_integer_set(text: Any) -> list[int]:
    value = str(text).strip()
    if not value:
        return []
    return [
        int(token)
        for token in value.split("-")
        if token.strip().isdigit()
    ]


def run_metadata_record(
    run_inventory: pd.DataFrame,
    run_id: str,
) -> dict[str, Any]:
    rows = run_inventory[run_inventory["run_id"].eq(run_id)]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one run inventory row for {run_id}; found {len(rows)}."
        )
    row = rows.iloc[0]
    return {
        "run_id": run_id,
        "split": str(row["split"]),
        "true_graph": int(row["true_graph"]),
        "profile": str(row["profile_from_name"]),
        "active_cores": str(row["active_cores_from_name"]),
        "active_core_count": int(row["active_core_count_from_name"]),
        "attackers": str(row["attackers_from_name"]),
        "attacker_count": int(row["attacker_count_from_name"]),
        "strength": int(row["strength_from_name"]),
        "seed": int(row["seed_from_name"]),
        "number_of_windows": int(row["number_of_windows"]),
    }


def topology_neighbors(
    topology_manifest: pd.DataFrame,
) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    for _, row in topology_manifest.iterrows():
        mapping[int(row["router_id"])] = parse_integer_set(row["neighbors"])
    return mapping


def router_groups_for_run(
    metadata: dict[str, Any],
    neighbor_map: dict[int, list[int]],
    router_count: int,
) -> dict[str, list[int]]:
    all_routers = set(range(router_count))
    active = set(parse_integer_set(metadata["active_cores"]))
    attackers = set(parse_integer_set(metadata["attackers"]))
    attacker_neighbors = set()

    for attacker in attackers:
        attacker_neighbors.update(neighbor_map.get(attacker, []))

    groups: dict[str, list[int]] = {
        "network_all": sorted(all_routers),
    }

    if active:
        groups["active_routers"] = sorted(active)
        groups["inactive_routers"] = sorted(all_routers - active)

    if attackers:
        groups["attacker_routers"] = sorted(attackers)
        groups["attacker_neighbors"] = sorted(attacker_neighbors)
        groups["other_routers"] = sorted(
            all_routers - attackers - attacker_neighbors - active
        )

    return groups


def run_indices_map(
    alignment: pd.DataFrame,
) -> dict[str, np.ndarray]:
    return {
        run_id: group.sort_values("run_window_index")[
            "dataset_index"
        ].to_numpy(dtype=np.int64)
        for run_id, group in alignment.groupby("run_id", sort=True)
    }


# ---------------------------------------------------------------------------
# All-run compact summaries
# ---------------------------------------------------------------------------

def summarize_run_features(
    run_data: np.ndarray,
    metadata: dict[str, Any],
    taxonomy: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []

    temporal_variance = np.var(run_data, axis=2).mean(axis=(0, 1))
    adjacent_change = np.abs(
        np.diff(run_data, axis=2)
    ).mean(axis=(0, 1, 2))
    spatial_variance = np.var(run_data, axis=1).mean(axis=(0, 1))

    for feature_index in range(run_data.shape[-1]):
        values = run_data[..., feature_index]
        summary = basic_summary(values)
        tax = taxonomy.iloc[feature_index].to_dict()

        rows.append(
            {
                **metadata,
                **tax,
                **summary,
                "mean_temporal_variance": float(
                    temporal_variance[feature_index]
                ),
                "mean_absolute_adjacent_temporal_change": float(
                    adjacent_change[feature_index]
                ),
                "mean_spatial_variance": float(
                    spatial_variance[feature_index]
                ),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Selected-run router summaries
# ---------------------------------------------------------------------------

def summarize_router_features(
    run_data: np.ndarray,
    metadata: dict[str, Any],
    taxonomy: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    windows, routers, temporal_positions, features = run_data.shape

    router_time_mean = run_data.mean(axis=2)
    router_ranks = np.argsort(
        np.argsort(router_time_mean, axis=1),
        axis=1,
    ) + 1
    maximum_router = np.argmax(router_time_mean, axis=1)

    temporal_range = (
        run_data.max(axis=2) - run_data.min(axis=2)
    )

    for router_id in range(routers):
        for feature_index in range(features):
            values = run_data[:, router_id, :, feature_index]
            summary = basic_summary(values, include_unique=False)
            tax = taxonomy.iloc[feature_index].to_dict()

            rows.append(
                {
                    **metadata,
                    "router_id": router_id,
                    **tax,
                    **summary,
                    "mean_temporal_variance": float(
                        np.var(values, axis=1).mean()
                    ),
                    "mean_temporal_excursion": float(
                        temporal_range[:, router_id, feature_index].mean()
                    ),
                    "maximum_temporal_excursion": float(
                        temporal_range[:, router_id, feature_index].max()
                    ),
                    "mean_spatial_rank": float(
                        router_ranks[:, router_id, feature_index].mean()
                    ),
                    "fraction_router_is_spatial_maximum": float(
                        np.mean(
                            maximum_router[:, feature_index] == router_id
                        )
                    ),
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Temporal summaries and excursions
# ---------------------------------------------------------------------------

def summarize_temporal_positions(
    run_data: np.ndarray,
    metadata: dict[str, Any],
    taxonomy: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    temporal_rows = []
    excursion_rows = []

    network_temporal = run_data.mean(axis=1)
    windows, positions, features = network_temporal.shape

    for temporal_position in range(positions):
        for feature_index in range(features):
            values = run_data[:, :, temporal_position, feature_index]
            summary = basic_summary(values, include_unique=False)
            tax = taxonomy.iloc[feature_index].to_dict()

            if temporal_position == 0:
                previous_change = 0.0
            else:
                previous_change = float(
                    np.abs(
                        run_data[:, :, temporal_position, feature_index]
                        - run_data[:, :, temporal_position - 1, feature_index]
                    ).mean()
                )

            temporal_rows.append(
                {
                    **metadata,
                    "temporal_position": temporal_position,
                    **tax,
                    **summary,
                    "mean_spatial_variance": float(
                        np.var(values, axis=1).mean()
                    ),
                    "mean_absolute_change_from_previous_position": previous_change,
                }
            )

    for feature_index in range(features):
        values = network_temporal[:, :, feature_index]
        temporal_range = values.max(axis=1) - values.min(axis=1)
        maximum_minus_mean = values.max(axis=1) - values.mean(axis=1)
        maximum_positions = np.argmax(values, axis=1)
        above_count = np.sum(values > 0.95, axis=1)
        window_median = np.median(values, axis=1, keepdims=True)
        deviating_count = np.sum(
            np.abs(values - window_median) > TEMPORAL_DEVIATION_THRESHOLD,
            axis=1,
        )

        position_fractions = [
            float(np.mean(maximum_positions == position))
            for position in range(positions)
        ]
        mode_position = int(np.argmax(position_fractions))

        tax = taxonomy.iloc[feature_index].to_dict()
        excursion_rows.append(
            {
                **metadata,
                **tax,
                "number_of_windows": windows,
                "mean_temporal_range": float(np.mean(temporal_range)),
                "median_temporal_range": float(np.median(temporal_range)),
                "p95_temporal_range": float(
                    np.quantile(temporal_range, 0.95)
                ),
                "maximum_temporal_range": float(np.max(temporal_range)),
                "mean_temporal_maximum_minus_mean": float(
                    np.mean(maximum_minus_mean)
                ),
                "mean_positions_above_0_95": float(np.mean(above_count)),
                "fraction_windows_with_any_position_above_0_95": float(
                    np.mean(above_count > 0)
                ),
                "mean_positions_deviating_gt_0_10_from_window_median": float(
                    np.mean(deviating_count)
                ),
                "fraction_windows_with_sparse_single_position_excursion": float(
                    np.mean(deviating_count == 1)
                ),
                "maximum_position_mode": mode_position,
                "maximum_position_mode_fraction": position_fractions[
                    mode_position
                ],
                "maximum_position_entropy_bits": entropy_from_fractions(
                    position_fractions
                ),
                **{
                    f"fraction_maximum_at_position_{position}": fraction
                    for position, fraction in enumerate(position_fractions)
                },
            }
        )

    return temporal_rows, excursion_rows


# ---------------------------------------------------------------------------
# Directional summaries
# ---------------------------------------------------------------------------

def summarize_directions(
    run_data: np.ndarray,
    metadata: dict[str, Any],
    taxonomy: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    directional = taxonomy[taxonomy["is_directional"]].copy()

    grouped = directional.groupby(
        ["measurement_type", "flow_side"],
        sort=True,
    )

    for (measurement_type, flow_side), group in grouped:
        group = group.sort_values(
            "direction",
            key=lambda series: series.map(
                {direction: index for index, direction in enumerate(DIRECTIONS)}
            ),
        )
        feature_indices = group["feature_index"].to_numpy(dtype=np.int64)
        directional_values = run_data[..., feature_indices]

        direction_means = directional_values.mean(axis=(0, 2))
        imbalance = direction_means.max(axis=1) - direction_means.min(axis=1)
        maximum_share = direction_means.max(axis=1) / np.maximum(
            np.sum(np.abs(direction_means), axis=1),
            EPSILON,
        )

        for local_index, (_, feature_row) in enumerate(group.iterrows()):
            feature_index = int(feature_row["feature_index"])
            direction = str(feature_row["direction"])

            for router_id in range(run_data.shape[1]):
                values = run_data[:, router_id, :, feature_index]
                summary = basic_summary(values, include_unique=False)

                rows.append(
                    {
                        **metadata,
                        "router_id": router_id,
                        "measurement_type": measurement_type,
                        "flow_side": flow_side,
                        "direction": direction,
                        "feature_index": feature_index,
                        "feature_name": feature_row["feature_name"],
                        "mean": summary["mean"],
                        "std": summary["std"],
                        "median": summary["median"],
                        "fraction_equal_0": summary["fraction_equal_0"],
                        "fraction_equal_1": summary["fraction_equal_1"],
                        "fraction_above_0_95": summary[
                            "fraction_above_0_95"
                        ],
                        "nonzero_rate": 1.0 - summary["fraction_equal_0"],
                        "directional_imbalance_for_router": float(
                            imbalance[router_id]
                        ),
                        "maximum_direction_share_for_router": float(
                            maximum_share[router_id]
                        ),
                    }
                )

    return rows


# ---------------------------------------------------------------------------
# Spatial concentration summaries
# ---------------------------------------------------------------------------

def summarize_spatial_concentration(
    run_data: np.ndarray,
    metadata: dict[str, Any],
    taxonomy: pd.DataFrame,
    neighbor_map: dict[int, list[int]],
) -> list[dict[str, Any]]:
    rows = []
    windows, routers, positions, features = run_data.shape
    active = parse_integer_set(metadata["active_cores"])
    attackers = parse_integer_set(metadata["attackers"])
    inactive = sorted(set(range(routers)) - set(active))

    attacker_neighbors = sorted(
        {
            neighbor
            for attacker in attackers
            for neighbor in neighbor_map.get(attacker, [])
        }
    )

    for feature_index in range(features):
        values = run_data[..., feature_index].transpose(0, 2, 1).reshape(
            windows * positions,
            routers,
        )
        router_maximum = values.max(axis=1)
        router_mean = values.mean(axis=1)
        router_median = np.median(values, axis=1)
        router_std = values.std(axis=1)
        maximum_minus_median = router_maximum - router_median
        top_one_share, top_two_share = normalized_shares(values)

        if active:
            active_mean = values[:, active].mean(axis=1)
        else:
            active_mean = np.full(len(values), np.nan)

        if inactive:
            inactive_mean = values[:, inactive].mean(axis=1)
        else:
            inactive_mean = np.full(len(values), np.nan)

        if attackers:
            attacker_value = values[:, attackers].mean(axis=1)
        else:
            attacker_value = np.full(len(values), np.nan)

        if attacker_neighbors:
            neighbor_mean = values[:, attacker_neighbors].mean(axis=1)
        else:
            neighbor_mean = np.full(len(values), np.nan)

        tax = taxonomy.iloc[feature_index].to_dict()
        rows.append(
            {
                **metadata,
                **tax,
                "number_of_window_positions": int(len(values)),
                "mean_router_maximum": float(np.mean(router_maximum)),
                "p95_router_maximum": float(
                    np.quantile(router_maximum, 0.95)
                ),
                "mean_router_mean": float(np.mean(router_mean)),
                "mean_router_median": float(np.mean(router_median)),
                "mean_router_std": float(np.mean(router_std)),
                "p95_router_std": float(np.quantile(router_std, 0.95)),
                "mean_maximum_minus_median": float(
                    np.mean(maximum_minus_median)
                ),
                "mean_top_router_share": float(np.mean(top_one_share)),
                "p95_top_router_share": float(
                    np.quantile(top_one_share, 0.95)
                ),
                "mean_top_two_router_share": float(
                    np.mean(top_two_share)
                ),
                "mean_active_router_value": float(
                    np.nanmean(active_mean)
                ) if active else np.nan,
                "mean_inactive_router_value": float(
                    np.nanmean(inactive_mean)
                ) if inactive else np.nan,
                "active_minus_inactive_mean": float(
                    np.nanmean(active_mean) - np.nanmean(inactive_mean)
                ) if active and inactive else np.nan,
                "mean_attacker_router_value": float(
                    np.nanmean(attacker_value)
                ) if attackers else np.nan,
                "mean_attacker_neighbor_value": float(
                    np.nanmean(neighbor_mean)
                ) if attacker_neighbors else np.nan,
                "attacker_minus_neighbor_mean": float(
                    np.nanmean(attacker_value) - np.nanmean(neighbor_mean)
                ) if attackers and attacker_neighbors else np.nan,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Router group summaries
# ---------------------------------------------------------------------------

def summarize_router_groups(
    run_data: np.ndarray,
    metadata: dict[str, Any],
    taxonomy: pd.DataFrame,
    neighbor_map: dict[int, list[int]],
) -> list[dict[str, Any]]:
    rows = []
    groups = router_groups_for_run(
        metadata,
        neighbor_map,
        run_data.shape[1],
    )

    for group_name, routers in groups.items():
        if not routers:
            continue

        group_data = run_data[:, routers, :, :]
        temporal_range = (
            group_data.max(axis=2) - group_data.min(axis=2)
        )

        for feature_index in range(run_data.shape[-1]):
            values = group_data[..., feature_index]
            summary = basic_summary(values, include_unique=False)
            tax = taxonomy.iloc[feature_index].to_dict()

            rows.append(
                {
                    **metadata,
                    "router_group": group_name,
                    "router_ids": "-".join(map(str, routers)),
                    "router_count": len(routers),
                    **tax,
                    **summary,
                    "mean_temporal_variance": float(
                        np.var(values, axis=2).mean()
                    ),
                    "mean_temporal_excursion": float(
                        temporal_range[..., feature_index].mean()
                    ),
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Matched-pair comparisons
# ---------------------------------------------------------------------------

def merge_pair_summary(
    left: pd.Series,
    right: pd.Series,
    group_name: str,
    pair_name: str,
) -> dict[str, Any]:
    return {
        "group_name": group_name,
        "pair_name": pair_name,
        "target_run_id": str(left["run_id"]),
        "control_run_id": str(right["run_id"]),
        "target_split": str(left["split"]),
        "control_split": str(right["split"]),
        "target_true_graph": int(left["true_graph"]),
        "control_true_graph": int(right["true_graph"]),
        "feature_index": int(left["feature_index"]),
        "feature_name": str(left["feature_name"]),
        "feature_family": str(left["feature_family"]),
        "measurement_type": str(left["measurement_type"]),
        "flow_side": str(left["flow_side"]),
        "direction": str(left["direction"]),
        "target_mean": float(left["mean"]),
        "control_mean": float(right["mean"]),
        "mean_difference_target_minus_control": float(
            left["mean"] - right["mean"]
        ),
        "target_median": float(left["median"]),
        "control_median": float(right["median"]),
        "median_difference_target_minus_control": float(
            left["median"] - right["median"]
        ),
        "target_std": float(left["std"]),
        "control_std": float(right["std"]),
        "std_difference_target_minus_control": float(
            left["std"] - right["std"]
        ),
        "standardized_mean_difference": pooled_standardized_mean_difference(
            float(left["mean"]),
            float(right["mean"]),
            float(left["std"]),
            float(right["std"]),
        ),
        "target_fraction_equal_0": float(left["fraction_equal_0"]),
        "control_fraction_equal_0": float(right["fraction_equal_0"]),
        "fraction_equal_0_difference": float(
            left["fraction_equal_0"] - right["fraction_equal_0"]
        ),
        "target_fraction_equal_1": float(left["fraction_equal_1"]),
        "control_fraction_equal_1": float(right["fraction_equal_1"]),
        "fraction_equal_1_difference": float(
            left["fraction_equal_1"] - right["fraction_equal_1"]
        ),
        "target_fraction_above_0_95": float(
            left["fraction_above_0_95"]
        ),
        "control_fraction_above_0_95": float(
            right["fraction_above_0_95"]
        ),
        "fraction_above_0_95_difference": float(
            left["fraction_above_0_95"]
            - right["fraction_above_0_95"]
        ),
        "target_mean_temporal_variance": float(
            left["mean_temporal_variance"]
        ),
        "control_mean_temporal_variance": float(
            right["mean_temporal_variance"]
        ),
        "temporal_variance_difference": float(
            left["mean_temporal_variance"]
            - right["mean_temporal_variance"]
        ),
        "target_mean_absolute_adjacent_temporal_change": float(
            left["mean_absolute_adjacent_temporal_change"]
        ),
        "control_mean_absolute_adjacent_temporal_change": float(
            right["mean_absolute_adjacent_temporal_change"]
        ),
        "adjacent_temporal_change_difference": float(
            left["mean_absolute_adjacent_temporal_change"]
            - right["mean_absolute_adjacent_temporal_change"]
        ),
        "target_mean_spatial_variance": float(
            left["mean_spatial_variance"]
        ),
        "control_mean_spatial_variance": float(
            right["mean_spatial_variance"]
        ),
        "spatial_variance_difference": float(
            left["mean_spatial_variance"]
            - right["mean_spatial_variance"]
        ),
    }


def build_pair_feature_differences(
    run_summary: pd.DataFrame,
    temporal_excursion: pd.DataFrame,
    spatial_concentration: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    excursion_lookup = temporal_excursion.set_index(
        ["run_id", "feature_index"]
    )
    spatial_lookup = spatial_concentration.set_index(
        ["run_id", "feature_index"]
    )

    for group_name, pair_name, target_run, control_run in PAIR_DEFINITIONS:
        target_rows = run_summary[
            run_summary["run_id"].eq(target_run)
        ].sort_values("feature_index")
        control_rows = run_summary[
            run_summary["run_id"].eq(control_run)
        ].sort_values("feature_index")

        if len(target_rows) != 24 or len(control_rows) != 24:
            raise RuntimeError(
                f"Pair {pair_name} does not have 24 feature summaries."
            )

        for (_, left), (_, right) in zip(
            target_rows.iterrows(),
            control_rows.iterrows(),
        ):
            if int(left["feature_index"]) != int(right["feature_index"]):
                raise RuntimeError(f"Feature-order mismatch for pair {pair_name}.")

            row = merge_pair_summary(
                left,
                right,
                group_name,
                pair_name,
            )
            feature_index = int(left["feature_index"])

            left_excursion = excursion_lookup.loc[
                (target_run, feature_index)
            ]
            right_excursion = excursion_lookup.loc[
                (control_run, feature_index)
            ]
            left_spatial = spatial_lookup.loc[
                (target_run, feature_index)
            ]
            right_spatial = spatial_lookup.loc[
                (control_run, feature_index)
            ]

            row.update(
                {
                    "target_mean_temporal_range": float(
                        left_excursion["mean_temporal_range"]
                    ),
                    "control_mean_temporal_range": float(
                        right_excursion["mean_temporal_range"]
                    ),
                    "temporal_range_difference": float(
                        left_excursion["mean_temporal_range"]
                        - right_excursion["mean_temporal_range"]
                    ),
                    "target_sparse_single_position_fraction": float(
                        left_excursion[
                            "fraction_windows_with_sparse_single_position_excursion"
                        ]
                    ),
                    "control_sparse_single_position_fraction": float(
                        right_excursion[
                            "fraction_windows_with_sparse_single_position_excursion"
                        ]
                    ),
                    "sparse_single_position_fraction_difference": float(
                        left_excursion[
                            "fraction_windows_with_sparse_single_position_excursion"
                        ]
                        - right_excursion[
                            "fraction_windows_with_sparse_single_position_excursion"
                        ]
                    ),
                    "target_mean_top_router_share": float(
                        left_spatial["mean_top_router_share"]
                    ),
                    "control_mean_top_router_share": float(
                        right_spatial["mean_top_router_share"]
                    ),
                    "top_router_share_difference": float(
                        left_spatial["mean_top_router_share"]
                        - right_spatial["mean_top_router_share"]
                    ),
                    "target_mean_router_std": float(
                        left_spatial["mean_router_std"]
                    ),
                    "control_mean_router_std": float(
                        right_spatial["mean_router_std"]
                    ),
                    "router_std_difference": float(
                        left_spatial["mean_router_std"]
                        - right_spatial["mean_router_std"]
                    ),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def build_pair_distribution_distances(
    x: np.ndarray,
    run_indices: dict[str, np.ndarray],
    taxonomy: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for group_name, pair_name, target_run, control_run in PAIR_DEFINITIONS:
        target_data = np.asarray(
            x[run_indices[target_run]],
            dtype=np.float32,
        )
        control_data = np.asarray(
            x[run_indices[control_run]],
            dtype=np.float32,
        )

        for feature_index in range(target_data.shape[-1]):
            target_values = target_data[..., feature_index]
            control_values = control_data[..., feature_index]
            distances = histogram_metrics(
                target_values,
                control_values,
            )
            tax = taxonomy.iloc[feature_index].to_dict()

            rows.append(
                {
                    "group_name": group_name,
                    "pair_name": pair_name,
                    "target_run_id": target_run,
                    "control_run_id": control_run,
                    **tax,
                    **distances,
                }
            )

        del target_data
        del control_data

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Saturation tables
# ---------------------------------------------------------------------------

def build_saturation_tables(
    run_summary: pd.DataFrame,
    router_summary: pd.DataFrame,
    temporal_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_columns = [
        "run_id",
        "split",
        "true_graph",
        "profile",
        "active_cores",
        "attackers",
        "strength",
        "feature_index",
        "feature_name",
        "feature_family",
        "measurement_type",
        "flow_side",
        "direction",
        "fraction_equal_0",
        "fraction_below_0_05",
        "fraction_above_0_95",
        "fraction_above_0_99",
        "fraction_equal_1",
        "fraction_outside_unit_interval",
    ]
    router_columns = [
        "run_id",
        "split",
        "true_graph",
        "router_id",
        "feature_index",
        "feature_name",
        "feature_family",
        "measurement_type",
        "flow_side",
        "direction",
        "fraction_equal_0",
        "fraction_below_0_05",
        "fraction_above_0_95",
        "fraction_above_0_99",
        "fraction_equal_1",
    ]
    temporal_columns = [
        "run_id",
        "split",
        "true_graph",
        "temporal_position",
        "feature_index",
        "feature_name",
        "feature_family",
        "measurement_type",
        "flow_side",
        "direction",
        "fraction_equal_0",
        "fraction_below_0_05",
        "fraction_above_0_95",
        "fraction_above_0_99",
        "fraction_equal_1",
    ]

    return (
        run_summary[run_columns].copy(),
        router_summary[router_columns].copy(),
        temporal_summary[temporal_columns].copy(),
    )


# ---------------------------------------------------------------------------
# Preliminary hypothesis status
# ---------------------------------------------------------------------------

def mean_metric(
    frame: pd.DataFrame,
    run_id: str,
    metric: str,
    query: str | None = None,
) -> float:
    subset = frame[frame["run_id"].eq(run_id)]
    if query:
        subset = subset.query(query)
    if subset.empty:
        return float("nan")
    return float(subset[metric].mean())


def build_hypothesis_status(
    run_summary: pd.DataFrame,
    temporal_excursion: pd.DataFrame,
    spatial_concentration: pd.DataFrame,
    pair_differences: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    directional_ifd_query = (
        "measurement_type == 'ifd' and direction != ''"
    )
    hard_normal_sat = mean_metric(
        run_summary,
        HARD_NORMAL_RUN,
        "fraction_equal_1",
        directional_ifd_query,
    )
    exact_normal_sat = mean_metric(
        run_summary,
        NORMAL_TEST_CONTROL,
        "fraction_equal_1",
        directional_ifd_query,
    )
    val_normal_sat = mean_metric(
        run_summary,
        NORMAL_VAL_CONTROL,
        "fraction_equal_1",
        directional_ifd_query,
    )
    saturation_difference = hard_normal_sat - max(
        exact_normal_sat,
        val_normal_sat,
    )

    if saturation_difference >= 0.05:
        h1_status = "supported"
    elif saturation_difference <= -0.02:
        h1_status = "contradicted"
    else:
        h1_status = "inconclusive"

    rows.append(
        {
            "hypothesis_id": "H1",
            "hypothesis_name": "directional_ifd_saturation",
            "preliminary_status": h1_status,
            "decision_rule": (
                "Supported when mean exact-1 directional IFD rate in the hard "
                "normal exceeds both matched normal controls by at least 0.05."
            ),
            "primary_metric": "mean_directional_ifd_fraction_equal_1",
            "target_value": hard_normal_sat,
            "comparison_value_1": exact_normal_sat,
            "comparison_value_2": val_normal_sat,
            "evidence_table": "stage7b_saturation_by_run_feature.csv",
            "causal_claim_allowed": False,
        }
    )

    hard_attack_range = mean_metric(
        temporal_excursion,
        HARD_ATTACK_RUN,
        "mean_temporal_range",
    )
    exact_attack_ranges = [
        mean_metric(
            temporal_excursion,
            run_id,
            "mean_temporal_range",
        )
        for run_id in [
            ATTACK_STREAM_CONTROL,
            ATTACK_MIXED_CONTROL,
            ATTACK_S63_CONTROL,
            ATTACK_S77_CONTROL,
        ]
    ]
    lower_than_count = sum(
        hard_attack_range < value * 0.90
        for value in exact_attack_ranges
        if np.isfinite(value)
    )

    if lower_than_count >= 3:
        h2_status = "supported"
    elif lower_than_count == 0:
        h2_status = "contradicted"
    else:
        h2_status = "inconclusive"

    rows.append(
        {
            "hypothesis_id": "H2",
            "hypothesis_name": "weak_temporal_signal",
            "preliminary_status": h2_status,
            "decision_rule": (
                "Supported when the hard S20 mean temporal range is at least "
                "10% lower than at least three successful weak/strong controls."
            ),
            "primary_metric": "mean_temporal_range_across_features",
            "target_value": hard_attack_range,
            "comparison_value_1": float(np.mean(exact_attack_ranges[:2])),
            "comparison_value_2": float(np.mean(exact_attack_ranges[2:])),
            "evidence_table": "stage7b_temporal_excursion_summary.csv",
            "causal_claim_allowed": False,
        }
    )

    hard_sparse = mean_metric(
        temporal_excursion,
        HARD_ATTACK_RUN,
        "fraction_windows_with_sparse_single_position_excursion",
    )
    control_sparse = float(
        np.mean(
            [
                mean_metric(
                    temporal_excursion,
                    run_id,
                    "fraction_windows_with_sparse_single_position_excursion",
                )
                for run_id in [
                    ATTACK_STREAM_CONTROL,
                    ATTACK_MIXED_CONTROL,
                    ATTACK_S63_CONTROL,
                    ATTACK_S77_CONTROL,
                ]
            ]
        )
    )

    if hard_sparse > control_sparse + 0.05:
        h3_status = "supported"
    elif hard_sparse < control_sparse - 0.05:
        h3_status = "contradicted"
    else:
        h3_status = "inconclusive"

    rows.append(
        {
            "hypothesis_id": "H3",
            "hypothesis_name": "pooling_loss_precursor",
            "preliminary_status": h3_status,
            "decision_rule": (
                "Only precursor evidence: compare sparse single-position "
                "excursion frequency. Pooling causation cannot be proven here."
            ),
            "primary_metric": "sparse_single_position_excursion_fraction",
            "target_value": hard_sparse,
            "comparison_value_1": control_sparse,
            "comparison_value_2": np.nan,
            "evidence_table": "stage7b_temporal_excursion_summary.csv",
            "causal_claim_allowed": False,
        }
    )

    hard_normal_spatial = mean_metric(
        spatial_concentration,
        HARD_NORMAL_RUN,
        "mean_router_std",
    )
    normal_spatial_controls = [
        mean_metric(
            spatial_concentration,
            run_id,
            "mean_router_std",
        )
        for run_id in [
            NORMAL_TEST_CONTROL,
            NORMAL_VAL_CONTROL,
        ]
    ]
    spatial_gap = hard_normal_spatial - max(normal_spatial_controls)

    pair_subset = pair_differences[
        pair_differences["pair_name"].isin(
            [
                "hard_normal_minus_exact_normal",
                "hard_normal_minus_validation_normal",
            ]
        )
    ]
    large_spatial_features = int(
        (
            pair_subset["router_std_difference"].abs()
            > 0.05
        ).sum()
    )

    if spatial_gap > 0.02 or large_spatial_features >= 8:
        h4_status = "supported"
    elif spatial_gap < -0.02 and large_spatial_features < 4:
        h4_status = "contradicted"
    else:
        h4_status = "inconclusive"

    rows.append(
        {
            "hypothesis_id": "H4",
            "hypothesis_name": "spatial_placement_pattern",
            "preliminary_status": h4_status,
            "decision_rule": (
                "Supported by higher spatial router variation or many large "
                "router-concentration differences versus both mixed four-core controls."
            ),
            "primary_metric": "mean_router_std_across_features",
            "target_value": hard_normal_spatial,
            "comparison_value_1": normal_spatial_controls[0],
            "comparison_value_2": normal_spatial_controls[1],
            "evidence_table": (
                "stage7b_selected_spatial_concentration.csv;"
                "stage7b_matched_pair_feature_differences.csv"
            ),
            "causal_claim_allowed": False,
        }
    )

    severity_runs = [
        HARD_ATTACK_RUN,
        ATTACK_S63_CONTROL,
        ATTACK_S77_CONTROL,
    ]
    severity_saturation = [
        mean_metric(
            run_summary,
            run_id,
            "fraction_equal_1",
            directional_ifd_query,
        )
        for run_id in severity_runs
    ]
    collapse_spread = float(np.max(severity_saturation) - np.min(severity_saturation))
    high_saturation_count = sum(value >= 0.20 for value in severity_saturation)

    if collapse_spread <= 0.05 and high_saturation_count >= 2:
        h5_status = "supported"
    elif collapse_spread >= 0.20:
        h5_status = "contradicted"
    else:
        h5_status = "inconclusive"

    rows.append(
        {
            "hypothesis_id": "H5",
            "hypothesis_name": "normalization_upper_bound_symptom",
            "preliminary_status": h5_status,
            "decision_rule": (
                "Supported when S20/S63/S77 directional IFD exact-1 rates are "
                "similarly high despite severity differences. This does not prove clipping."
            ),
            "primary_metric": "directional_ifd_fraction_equal_1_severity_spread",
            "target_value": severity_saturation[0],
            "comparison_value_1": severity_saturation[1],
            "comparison_value_2": severity_saturation[2],
            "evidence_table": "stage7b_saturation_by_run_feature.csv",
            "causal_claim_allowed": False,
        }
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

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
    figure, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_xticks(np.arange(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=7)
    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels(matrix.index, fontsize=7)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    figure.colorbar(image, ax=axis)
    save_figure(figure, path)


def make_plots(
    run_summary: pd.DataFrame,
    router_summary: pd.DataFrame,
    temporal_summary: pd.DataFrame,
    direction_summary: pd.DataFrame,
    spatial_summary: pd.DataFrame,
    router_group_summary: pd.DataFrame,
    pair_differences: pd.DataFrame,
    temporal_excursion: pd.DataFrame,
    plot_directory: Path,
) -> None:
    plot_directory.mkdir(parents=True, exist_ok=True)

    saturation_matrix = run_summary.pivot(
        index="run_id",
        columns="feature_name",
        values="fraction_equal_1",
    )
    split_order = (
        run_summary[
            ["run_id", "split", "true_graph"]
        ]
        .drop_duplicates()
        .sort_values(["split", "true_graph", "run_id"])
        ["run_id"]
        .tolist()
    )
    saturation_matrix = saturation_matrix.reindex(split_order)
    plot_heatmap(
        saturation_matrix,
        "All-run exact-1.0 saturation rate",
        plot_directory / "01_all_run_exact_one_saturation_heatmap.png",
        "Feature",
        "Run",
        vmin=0,
        vmax=1,
        figsize=(14, 18),
    )

    for run_id, filename in [
        (HARD_NORMAL_RUN, "03_hard_normal_router_feature_mean.png"),
        (NORMAL_TEST_CONTROL, "03_exact_normal_control_router_feature_mean.png"),
    ]:
        matrix = router_summary[
            router_summary["run_id"].eq(run_id)
        ].pivot(
            index="router_id",
            columns="feature_name",
            values="mean",
        )
        plot_heatmap(
            matrix,
            f"Router × feature mean: {run_id}",
            plot_directory / filename,
            "Feature",
            "Router",
            figsize=(14, 7),
        )

    normal_pair = pair_differences[
        pair_differences["pair_name"].eq(
            "hard_normal_minus_exact_normal"
        )
    ].sort_values(
        "standardized_mean_difference",
        key=lambda series: series.abs(),
        ascending=False,
    ).head(15)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(
        normal_pair["feature_name"][::-1],
        normal_pair["standardized_mean_difference"][::-1],
    )
    axis.set_xlabel("Standardized mean difference")
    axis.set_title("Hard normal minus exact normal control: top features")
    save_figure(
        figure,
        plot_directory / "02_hard_normal_top_standardized_differences.png",
    )

    attack_runs = [
        HARD_ATTACK_RUN,
        ATTACK_STREAM_CONTROL,
        ATTACK_MIXED_CONTROL,
    ]
    attack_matrix = run_summary[
        run_summary["run_id"].isin(attack_runs)
    ].pivot(
        index="run_id",
        columns="feature_name",
        values="mean",
    ).reindex(attack_runs)
    plot_heatmap(
        attack_matrix,
        "Hard S20 attack versus exact A12-S20 controls: feature means",
        plot_directory / "04_hard_attack_vs_A12_S20_feature_means.png",
        "Feature",
        "Run",
        figsize=(14, 5),
    )

    severity_runs = [
        HARD_ATTACK_RUN,
        ATTACK_S63_CONTROL,
        ATTACK_S77_CONTROL,
    ]
    severity_matrix = run_summary[
        run_summary["run_id"].isin(severity_runs)
    ].pivot(
        index="run_id",
        columns="feature_name",
        values="mean",
    ).reindex(severity_runs)
    plot_heatmap(
        severity_matrix,
        "Same bursty background: S20 versus S63 versus S77 feature means",
        plot_directory / "05_S20_S63_S77_feature_means.png",
        "Feature",
        "Run",
        figsize=(14, 5),
    )

    selected_runs = [
        HARD_NORMAL_RUN,
        NORMAL_TEST_CONTROL,
        HARD_ATTACK_RUN,
        ATTACK_STREAM_CONTROL,
        ATTACK_MIXED_CONTROL,
        ATTACK_BURSTY_S20_CONTROL,
        ATTACK_S63_CONTROL,
        ATTACK_S77_CONTROL,
    ]

    directional_ifd = direction_summary[
        direction_summary["measurement_type"].eq("ifd")
        & direction_summary["run_id"].isin(selected_runs)
    ]
    ifd_matrix = directional_ifd.pivot_table(
        index=["run_id", "router_id"],
        columns=["flow_side", "direction"],
        values="fraction_equal_1",
        aggfunc="mean",
    )
    ifd_matrix.columns = [
        f"{flow}_{direction}"
        for flow, direction in ifd_matrix.columns
    ]
    plot_heatmap(
        ifd_matrix,
        "Directional IFD exact-1.0 rate",
        plot_directory / "06_directional_ifd_saturation_heatmap.png",
        "Direction",
        "Run / router",
        vmin=0,
        vmax=1,
        figsize=(12, 18),
    )

    directional_counts = direction_summary[
        direction_summary["measurement_type"].eq("flit_count")
        & direction_summary["run_id"].isin(selected_runs)
    ]
    count_matrix = directional_counts.pivot_table(
        index=["run_id", "router_id"],
        columns=["flow_side", "direction"],
        values="mean",
        aggfunc="mean",
    )
    count_matrix.columns = [
        f"{flow}_{direction}"
        for flow, direction in count_matrix.columns
    ]
    plot_heatmap(
        count_matrix,
        "Directional flit-count mean",
        plot_directory / "07_directional_flit_count_heatmap.png",
        "Direction",
        "Run / router",
        figsize=(12, 18),
    )

    hard_attack_temporal = temporal_summary[
        temporal_summary["run_id"].eq(HARD_ATTACK_RUN)
    ].pivot(
        index="temporal_position",
        columns="feature_name",
        values="mean",
    )
    plot_heatmap(
        hard_attack_temporal,
        "Hard S20 attack: temporal-position feature means",
        plot_directory / "08_hard_attack_temporal_position_heatmap.png",
        "Feature",
        "Temporal position",
        figsize=(14, 6),
    )

    excursion_matrix = temporal_excursion[
        temporal_excursion["run_id"].isin(
            [
                HARD_ATTACK_RUN,
                ATTACK_STREAM_CONTROL,
                ATTACK_MIXED_CONTROL,
                ATTACK_BURSTY_S20_CONTROL,
                ATTACK_S63_CONTROL,
                ATTACK_S77_CONTROL,
            ]
        )
    ].pivot(
        index="run_id",
        columns="feature_name",
        values="mean_temporal_range",
    )
    plot_heatmap(
        excursion_matrix,
        "Attack-control temporal excursion comparison",
        plot_directory / "09_attack_temporal_excursion_comparison.png",
        "Feature",
        "Run",
        figsize=(14, 7),
    )

    active_inactive = router_group_summary[
        router_group_summary["router_group"].isin(
            ["active_routers", "inactive_routers"]
        )
        & router_group_summary["run_id"].isin(
            [HARD_NORMAL_RUN, NORMAL_TEST_CONTROL, NORMAL_VAL_CONTROL]
        )
    ]
    figure, axis = plt.subplots(figsize=(12, 6))
    pivot = active_inactive.pivot_table(
        index="feature_name",
        columns=["run_id", "router_group"],
        values="mean",
        aggfunc="mean",
    )
    pivot.plot(ax=axis, linewidth=1)
    axis.set_title("Active versus inactive router feature means")
    axis.set_xlabel("Feature")
    axis.set_ylabel("Mean normalized value")
    axis.tick_params(axis="x", rotation=60)
    save_figure(
        figure,
        plot_directory / "10_active_vs_inactive_router_comparison.png",
    )

    attack_groups = router_group_summary[
        router_group_summary["run_id"].eq(HARD_ATTACK_RUN)
        & router_group_summary["router_group"].isin(
            [
                "attacker_routers",
                "attacker_neighbors",
                "active_routers",
                "other_routers",
            ]
        )
    ]
    matrix = attack_groups.pivot(
        index="router_group",
        columns="feature_name",
        values="mean",
    )
    plot_heatmap(
        matrix,
        "Hard attack router-group feature means",
        plot_directory / "11_attacker_neighbor_other_comparison.png",
        "Feature",
        "Router group",
        figsize=(14, 5),
    )

    spatial_matrix = spatial_summary[
        spatial_summary["run_id"].isin(selected_runs)
    ].pivot(
        index="run_id",
        columns="feature_name",
        values="mean_top_router_share",
    )
    plot_heatmap(
        spatial_matrix,
        "Spatial top-router concentration",
        plot_directory / "12_spatial_concentration_comparison.png",
        "Feature",
        "Run",
        vmin=0,
        vmax=1,
        figsize=(14, 8),
    )

    strongest = pair_differences.assign(
        absolute_smd=lambda frame: frame["standardized_mean_difference"].abs()
    ).sort_values("absolute_smd", ascending=False).head(20)
    labels = (
        strongest["pair_name"]
        + " | "
        + strongest["feature_name"]
    )
    figure, axis = plt.subplots(figsize=(12, 8))
    axis.barh(
        labels[::-1],
        strongest["standardized_mean_difference"][::-1],
    )
    axis.set_xlabel("Standardized mean difference")
    axis.set_title("Top matched-pair feature differences")
    save_figure(
        figure,
        plot_directory / "13_top_matched_pair_feature_differences.png",
    )

    family_saturation = (
        run_summary.groupby(
            ["feature_family", "run_id"],
            as_index=False,
        )["fraction_equal_1"]
        .mean()
    )
    selected_family = family_saturation[
        family_saturation["run_id"].isin(selected_runs)
    ]
    pivot = selected_family.pivot(
        index="run_id",
        columns="feature_family",
        values="fraction_equal_1",
    )
    plot_heatmap(
        pivot,
        "Observed exact-1.0 rate by feature family",
        plot_directory / "14_exact_one_rate_by_feature_family.png",
        "Feature family",
        "Run",
        vmin=0,
        vmax=1,
        figsize=(11, 7),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def top_pair_features(
    pair_differences: pd.DataFrame,
    pair_name: str,
    count: int = 5,
) -> list[pd.Series]:
    subset = pair_differences[
        pair_differences["pair_name"].eq(pair_name)
    ].copy()
    subset["absolute_smd"] = subset[
        "standardized_mean_difference"
    ].abs()
    return [
        row
        for _, row in subset.sort_values(
            "absolute_smd",
            ascending=False,
        ).head(count).iterrows()
    ]


def build_report(
    run_summary: pd.DataFrame,
    pair_differences: pd.DataFrame,
    hypothesis_status: pd.DataFrame,
    selected_runs: list[str],
    all_samples_scanned: int,
    plot_count: int,
) -> str:
    lines = [
        "# Stage 7B — Feature and Saturation Analysis",
        "",
        "Stage 7B scanned the normalized V3 feature tensor one run at a time. "
        "It produced compact summaries for all 71 runs and detailed router, "
        "temporal, directional and spatial summaries for the selected matched runs.",
        "",
        "The analysis reports observed normalized behaviour only. It does not "
        "establish the provenance or cause of clipping until the dataset builder "
        "and normalization logic are recovered.",
        "",
        "## Scan",
        "",
        f"- Samples scanned: **{all_samples_scanned}**.",
        "- Runs summarized: **71**.",
        "- Features summarized per run: **24**.",
        f"- Detailed selected runs: **{len(selected_runs)}**.",
        f"- Plots created: **{plot_count}**.",
        "- Inference rerun: **False**.",
        "- Training performed: **False**.",
        "",
        "## Metadata corrections applied",
        "",
        "- Features 4–13 were explicitly classified as directional input/output "
        "flit-count features.",
        "- Stage 7A was not rewritten; Stage 7B records the corrected taxonomy as "
        "a new provenance table.",
        "- `true_attacker_mask` is available in prediction exports and will be "
        "used later in Stage 7E. No node grouping was performed here.",
        "",
        "## Preliminary hypothesis status",
        "",
        "| Hypothesis | Status | Primary metric | Causal claim allowed |",
        "|---|---|---|---:|",
    ]

    for _, row in hypothesis_status.iterrows():
        lines.append(
            f"| {row['hypothesis_id']} — {row['hypothesis_name']} | "
            f"{row['preliminary_status']} | {row['primary_metric']} | "
            f"{row['causal_claim_allowed']} |"
        )

    lines.extend(
        [
            "",
            "## Strongest matched-pair feature differences",
            "",
        ]
    )

    for pair_name in [
        "hard_normal_minus_exact_normal",
        "hard_normal_minus_validation_normal",
        "hard_attack_minus_stream_A12_S20",
        "hard_attack_minus_mixed_A12_S20",
        "S20_minus_S63",
        "S20_minus_S77",
    ]:
        lines.append(f"### {pair_name}")
        for row in top_pair_features(pair_differences, pair_name):
            lines.append(
                f"- `{row['feature_name']}`: standardized mean difference "
                f"{float(row['standardized_mean_difference']):+.3f}; "
                f"exact-1.0-rate difference "
                f"{float(row['fraction_equal_1_difference']):+.3f}; "
                f"temporal-range difference "
                f"{float(row['temporal_range_difference']):+.3f}."
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "- A high exact-1.0 rate is described as observed normalized saturation.",
            "- Stage 7B cannot determine whether 1.0 came from clipping, a true "
            "physical maximum, or a default encoding.",
            "- Sparse temporal excursions may support a weak-signal explanation, "
            "but they do not by themselves prove that mean or max pooling caused "
            "the model failure.",
            "- Spatial differences support a placement-pattern hypothesis, but "
            "route-level causation requires routing and builder recovery.",
            "",
            "## Next stage",
            "",
            "Stage 7C should use training runs only to define coverage ranges, "
            "standardization, nearest-run distances and out-of-training-range rates.",
            "",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def fractions_valid(frame: pd.DataFrame) -> bool:
    fraction_columns = [
        column
        for column in frame.columns
        if column.startswith("fraction_")
        or column.endswith("_share")
        or column.endswith("_rate")
    ]
    for column in fraction_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) and (
            (finite < -1e-12).any() or (finite > 1.0 + 1e-12).any()
        ):
            return False
    return True


def numeric_finite_except_allowed_nan(
    frame: pd.DataFrame,
    excluded_columns: set[str] | None = None,
) -> bool:
    excluded_columns = excluded_columns or set()
    numeric = frame.select_dtypes(include=[np.number]).copy()
    numeric = numeric.drop(
        columns=[
            column
            for column in numeric.columns
            if column in excluded_columns
        ],
        errors="ignore",
    )
    return bool(
        np.isfinite(numeric.to_numpy(dtype=float)).all()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 7B: memory-safe global, router, temporal, directional, "
            "spatial and saturation summaries for the normalized V3 feature tensor."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "~/tools/architecture/gem5/experiments/"
            "paper1_temporal_graphs_ports_v3"
        ),
    )
    parser.add_argument(
        "--failure-root",
        type=Path,
        default=Path("reports/v3_failure_analysis"),
    )
    parser.add_argument("--expected-samples", type=int, default=233803)
    parser.add_argument("--expected-runs", type=int, default=71)
    parser.add_argument("--expected-windows-per-run", type=int, default=3293)
    parser.add_argument("--expected-routers", type=int, default=16)
    parser.add_argument("--expected-temporal-positions", type=int, default=8)
    parser.add_argument("--expected-features", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    failure_root = args.failure_root.resolve()
    table_directory = failure_root / "tables"
    log_directory = failure_root / "logs"
    plot_directory = failure_root / "plots" / "stage7b_features"

    x_path = dataset_root / "x.npy"
    feature_cols_path = dataset_root / "feature_cols.npy"

    alignment_path = table_directory / "stage7_sample_alignment.csv"
    run_inventory_path = table_directory / "stage7_run_inventory.csv"
    selected_inventory_path = (
        table_directory / "stage7_selected_run_inventory.csv"
    )
    topology_path = table_directory / "stage7_topology_manifest.csv"
    stage7a_summary_path = table_directory / "stage7a_alignment_summary.json"

    output_paths = {
        "corrected_taxonomy": table_directory / "stage7b_corrected_feature_taxonomy.csv",
        "run_feature_summary": table_directory / "stage7b_run_feature_summary.csv",
        "selected_router_summary": table_directory / "stage7b_selected_router_feature_summary.csv",
        "selected_temporal_summary": table_directory / "stage7b_selected_temporal_feature_summary.csv",
        "selected_direction_summary": table_directory / "stage7b_selected_direction_summary.csv",
        "selected_spatial_summary": table_directory / "stage7b_selected_spatial_concentration.csv",
        "saturation_run_feature": table_directory / "stage7b_saturation_by_run_feature.csv",
        "saturation_router_feature": table_directory / "stage7b_saturation_by_router_feature.csv",
        "saturation_temporal": table_directory / "stage7b_saturation_by_temporal_position.csv",
        "router_group_summary": table_directory / "stage7b_router_group_summary.csv",
        "pair_differences": table_directory / "stage7b_matched_pair_feature_differences.csv",
        "pair_distances": table_directory / "stage7b_matched_pair_distribution_distances.csv",
        "temporal_excursion": table_directory / "stage7b_temporal_excursion_summary.csv",
        "hypothesis_status": table_directory / "stage7b_hypothesis_status.csv",
        "scan_manifest": table_directory / "stage7b_scan_manifest.csv",
        "summary_json": table_directory / "stage7b_summary.json",
        "report": failure_root / "STAGE7B_FEATURE_AND_SATURATION_REPORT.md",
        "validation": log_directory / "36_stage7b_validation.txt",
    }

    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Stage 7B outputs already exist. Use --overwrite to replace them:\n"
            + "\n".join(str(path) for path in existing)
        )

    required_paths = [
        x_path,
        feature_cols_path,
        alignment_path,
        run_inventory_path,
        selected_inventory_path,
        topology_path,
        stage7a_summary_path,
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"dataset root: {dataset_root}", flush=True)
    print(f"failure root: {failure_root}", flush=True)
    print(f"x: {x_path}", flush=True)
    print("inference rerun: False", flush=True)
    print("training performed: False", flush=True)
    print("classifier fitting: False", flush=True)
    print("nearest-run analysis: False", flush=True)
    print("node correctness grouping: False", flush=True)

    x = np.load(x_path, mmap_mode="r", allow_pickle=False)
    feature_cols = np.load(feature_cols_path, allow_pickle=True)

    expected_shape = (
        args.expected_samples,
        args.expected_routers,
        args.expected_temporal_positions,
        args.expected_features,
    )
    x_shape_valid = tuple(x.shape) == expected_shape
    x_is_memmap = isinstance(x, np.memmap)
    if not x_shape_valid:
        raise RuntimeError(
            f"x.npy shape {x.shape} differs from expected {expected_shape}."
        )
    if not x_is_memmap:
        raise RuntimeError("x.npy is not memory mapped.")

    verify_feature_order(feature_cols)
    taxonomy = corrected_feature_taxonomy()

    alignment = pd.read_csv(
        alignment_path,
        keep_default_na=False,
        low_memory=False,
        dtype={
            "sample_id": str,
            "run_id": str,
            "split": str,
        },
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
    topology = pd.read_csv(
        topology_path,
        keep_default_na=False,
        low_memory=False,
    )
    stage7a_summary = json.loads(stage7a_summary_path.read_text())

    if not stage7a_summary.get("overall_success", False):
        raise RuntimeError("Stage 7A summary does not report overall success.")

    required_alignment_columns = {
        "dataset_index",
        "run_id",
        "split",
        "true_graph",
        "run_window_index",
    }
    missing_alignment = required_alignment_columns - set(alignment.columns)
    if missing_alignment:
        raise RuntimeError(
            f"Stage 7 alignment missing columns: {sorted(missing_alignment)}"
        )

    indices_by_run = run_indices_map(alignment)
    if len(indices_by_run) != args.expected_runs:
        raise RuntimeError(
            f"Found {len(indices_by_run)} runs; expected {args.expected_runs}."
        )

    selected_runs = sorted(selected_inventory["run_id"].unique())
    if any(run_id not in indices_by_run for run_id in selected_runs):
        raise RuntimeError("A selected Stage 7 run is absent from alignment.")

    neighbors = topology_neighbors(topology)

    all_run_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    excursion_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    router_group_rows: list[dict[str, Any]] = []
    scan_rows: list[dict[str, Any]] = []

    all_samples_scanned = 0
    all_features_finite = True

    ordered_run_ids = (
        run_inventory.sort_values(["split", "run_id"])["run_id"].tolist()
    )

    for run_number, run_id in enumerate(ordered_run_ids, start=1):
        indices = indices_by_run[run_id]

        if len(indices) != args.expected_windows_per_run:
            raise RuntimeError(
                f"Run {run_id} has {len(indices)} windows; "
                f"expected {args.expected_windows_per_run}."
            )

        metadata = run_metadata_record(run_inventory, run_id)
        run_data = np.asarray(x[indices], dtype=np.float32)

        if tuple(run_data.shape) != (
            args.expected_windows_per_run,
            args.expected_routers,
            args.expected_temporal_positions,
            args.expected_features,
        ):
            raise RuntimeError(
                f"Run data shape mismatch for {run_id}: {run_data.shape}"
            )

        finite = bool(np.isfinite(run_data).all())
        all_features_finite = all_features_finite and finite
        if not finite:
            raise RuntimeError(f"Non-finite feature value in run {run_id}.")

        all_run_rows.extend(
            summarize_run_features(
                run_data,
                metadata,
                taxonomy,
            )
        )

        detailed = run_id in selected_runs
        if detailed:
            router_rows.extend(
                summarize_router_features(
                    run_data,
                    metadata,
                    taxonomy,
                )
            )
            temporal, excursions = summarize_temporal_positions(
                run_data,
                metadata,
                taxonomy,
            )
            temporal_rows.extend(temporal)
            excursion_rows.extend(excursions)
            direction_rows.extend(
                summarize_directions(
                    run_data,
                    metadata,
                    taxonomy,
                )
            )
            spatial_rows.extend(
                summarize_spatial_concentration(
                    run_data,
                    metadata,
                    taxonomy,
                    neighbors,
                )
            )
            router_group_rows.extend(
                summarize_router_groups(
                    run_data,
                    metadata,
                    taxonomy,
                    neighbors,
                )
            )

        scan_rows.append(
            {
                "run_scan_order": run_number,
                "run_id": run_id,
                "split": metadata["split"],
                "true_graph": metadata["true_graph"],
                "number_of_windows": len(indices),
                "dataset_index_minimum": int(indices.min()),
                "dataset_index_maximum": int(indices.max()),
                "indices_contiguous": bool(
                    np.all(np.diff(np.sort(indices)) == 1)
                ),
                "run_array_shape": "x".join(map(str, run_data.shape)),
                "run_array_bytes": int(run_data.nbytes),
                "all_values_finite": finite,
                "detailed_selected_run_analysis": detailed,
                "full_tensor_in_memory": False,
            }
        )

        all_samples_scanned += len(indices)
        print(
            f"[{run_number:02d}/{len(ordered_run_ids)}] "
            f"{run_id} | split={metadata['split']} | "
            f"detailed={detailed}",
            flush=True,
        )

        del run_data

    run_summary = pd.DataFrame(all_run_rows)
    router_summary = pd.DataFrame(router_rows)
    temporal_summary = pd.DataFrame(temporal_rows)
    temporal_excursion = pd.DataFrame(excursion_rows)
    direction_summary = pd.DataFrame(direction_rows)
    spatial_summary = pd.DataFrame(spatial_rows)
    router_group_summary = pd.DataFrame(router_group_rows)
    scan_manifest = pd.DataFrame(scan_rows)

    (
        saturation_run,
        saturation_router,
        saturation_temporal,
    ) = build_saturation_tables(
        run_summary,
        router_summary,
        temporal_summary,
    )

    pair_differences = build_pair_feature_differences(
        run_summary,
        temporal_excursion,
        spatial_summary,
    )

    print("computing matched-pair distribution distances", flush=True)
    pair_distances = build_pair_distribution_distances(
        x,
        indices_by_run,
        taxonomy,
    )

    hypothesis_status = build_hypothesis_status(
        run_summary,
        temporal_excursion,
        spatial_summary,
        pair_differences,
    )

    make_plots(
        run_summary,
        router_summary,
        temporal_summary,
        direction_summary,
        spatial_summary,
        router_group_summary,
        pair_differences,
        temporal_excursion,
        plot_directory,
    )
    plot_count = len(list(plot_directory.glob("*.png")))

    expected_run_summary_rows = args.expected_runs * args.expected_features
    expected_router_rows = (
        len(selected_runs)
        * args.expected_routers
        * args.expected_features
    )
    expected_temporal_rows = (
        len(selected_runs)
        * args.expected_temporal_positions
        * args.expected_features
    )
    expected_excursion_rows = len(selected_runs) * args.expected_features
    expected_spatial_rows = len(selected_runs) * args.expected_features
    expected_pair_rows = len(PAIR_DEFINITIONS) * args.expected_features
    expected_direction_rows = (
        len(selected_runs)
        * args.expected_routers
        * 20
    )

    all_samples_accounted = all_samples_scanned == args.expected_samples
    all_runs_summarized = (
        run_summary["run_id"].nunique() == args.expected_runs
    )
    all_features_per_run = (
        len(run_summary) == expected_run_summary_rows
        and run_summary.groupby("run_id")["feature_index"].nunique().eq(
            args.expected_features
        ).all()
    )
    selected_runs_match_stage7a = (
        set(selected_runs)
        == set(selected_inventory["run_id"].unique())
    )
    selected_windows_valid = all(
        len(indices_by_run[run_id]) == args.expected_windows_per_run
        for run_id in selected_runs
    )
    routers_complete = (
        len(router_summary) == expected_router_rows
        and set(router_summary["router_id"].unique())
        == set(range(args.expected_routers))
    )
    temporal_positions_complete = (
        len(temporal_summary) == expected_temporal_rows
        and set(temporal_summary["temporal_position"].unique())
        == set(range(args.expected_temporal_positions))
    )
    excursion_complete = len(temporal_excursion) == expected_excursion_rows
    spatial_complete = len(spatial_summary) == expected_spatial_rows
    direction_complete = len(direction_summary) == expected_direction_rows
    pair_summaries_complete = (
        len(pair_differences) == expected_pair_rows
        and len(pair_distances) == expected_pair_rows
    )

    fractions_in_range = all(
        [
            fractions_valid(run_summary),
            fractions_valid(router_summary),
            fractions_valid(temporal_summary),
            fractions_valid(temporal_excursion),
            fractions_valid(direction_summary),
            fractions_valid(spatial_summary),
            fractions_valid(router_group_summary),
            fractions_valid(pair_distances),
        ]
    )

    run_summary_finite = numeric_finite_except_allowed_nan(
        run_summary,
    )
    router_summary_finite = numeric_finite_except_allowed_nan(
        router_summary,
    )
    temporal_summary_finite = numeric_finite_except_allowed_nan(
        temporal_summary,
    )
    excursion_finite = numeric_finite_except_allowed_nan(
        temporal_excursion,
    )
    direction_finite = numeric_finite_except_allowed_nan(
        direction_summary,
    )
    pair_distance_finite = numeric_finite_except_allowed_nan(
        pair_distances,
    )

    spatial_numeric = spatial_summary.select_dtypes(include=[np.number])
    spatial_required_finite_columns = [
        column
        for column in spatial_numeric.columns
        if not column.startswith("mean_attacker")
        and not column.startswith("attacker_")
        and column not in {
            "mean_active_router_value",
            "mean_inactive_router_value",
            "active_minus_inactive_mean",
        }
    ]
    spatial_finite = bool(
        np.isfinite(
            spatial_summary[
                spatial_required_finite_columns
            ].to_numpy(dtype=float)
        ).all()
    )

    pair_numeric = pair_differences.select_dtypes(include=[np.number])
    pair_finite = bool(
        np.isfinite(pair_numeric.to_numpy(dtype=float)).all()
    )

    all_statistics_finite_where_defined = all(
        [
            run_summary_finite,
            router_summary_finite,
            temporal_summary_finite,
            excursion_finite,
            direction_finite,
            spatial_finite,
            pair_finite,
            pair_distance_finite,
        ]
    )

    taxonomy_correct = (
        taxonomy["feature_name"].tolist() == EXPECTED_FEATURE_NAMES
        and taxonomy.loc[
            taxonomy["feature_index"].between(4, 13),
            "measurement_type",
        ].eq("flit_count").all()
    )

    no_split_changes = bool(
        alignment.groupby("run_id")["split"].nunique().eq(1).all()
    )

    overall_success = all(
        [
            x_shape_valid,
            x_is_memmap,
            taxonomy_correct,
            all_samples_accounted,
            all_runs_summarized,
            all_features_per_run,
            selected_runs_match_stage7a,
            selected_windows_valid,
            routers_complete,
            temporal_positions_complete,
            excursion_complete,
            spatial_complete,
            direction_complete,
            pair_summaries_complete,
            all_features_finite,
            fractions_in_range,
            all_statistics_finite_where_defined,
            no_split_changes,
        ]
    )

    validation_lines = [
        f"feature tensor shape verified: {x_shape_valid}",
        f"feature tensor memory mapped: {x_is_memmap}",
        f"corrected feature taxonomy verified: {taxonomy_correct}",
        f"all 233803 samples accounted for: {all_samples_accounted}",
        f"all 71 runs summarized: {all_runs_summarized}",
        f"all 24 features summarized per run: {all_features_per_run}",
        f"selected runs match Stage 7A: {selected_runs_match_stage7a}",
        f"selected runs contain 3293 windows: {selected_windows_valid}",
        f"all 16 routers represented: {routers_complete}",
        f"all 8 temporal positions represented: {temporal_positions_complete}",
        f"temporal excursion summaries complete: {excursion_complete}",
        f"spatial concentration summaries complete: {spatial_complete}",
        f"directional summaries complete: {direction_complete}",
        f"matched-pair summaries complete: {pair_summaries_complete}",
        f"all scanned feature values finite: {all_features_finite}",
        f"all fractions lie in [0,1]: {fractions_in_range}",
        f"all statistics finite where mathematically defined: {all_statistics_finite_where_defined}",
        f"split membership unchanged: {no_split_changes}",
        f"raw tensor values written individually to CSV: False",
        f"data-derived transformation fitted: False",
        f"classifier fitted: False",
        f"nearest-run analysis performed: False",
        f"node correctness grouping performed: False",
        f"inference rerun: False",
        f"training performed: False",
        f"overall success: {overall_success}",
    ]
    validation_text = "\n".join(validation_lines) + "\n"
    atomic_text(validation_text, output_paths["validation"])

    print()
    print(validation_text, end="", flush=True)

    if not overall_success:
        raise SystemExit(
            f"Stage 7B validation failed. Inspect {output_paths['validation']}."
        )

    atomic_csv(taxonomy, output_paths["corrected_taxonomy"])
    atomic_csv(run_summary, output_paths["run_feature_summary"])
    atomic_csv(router_summary, output_paths["selected_router_summary"])
    atomic_csv(temporal_summary, output_paths["selected_temporal_summary"])
    atomic_csv(direction_summary, output_paths["selected_direction_summary"])
    atomic_csv(spatial_summary, output_paths["selected_spatial_summary"])
    atomic_csv(saturation_run, output_paths["saturation_run_feature"])
    atomic_csv(saturation_router, output_paths["saturation_router_feature"])
    atomic_csv(saturation_temporal, output_paths["saturation_temporal"])
    atomic_csv(router_group_summary, output_paths["router_group_summary"])
    atomic_csv(pair_differences, output_paths["pair_differences"])
    atomic_csv(pair_distances, output_paths["pair_distances"])
    atomic_csv(temporal_excursion, output_paths["temporal_excursion"])
    atomic_csv(hypothesis_status, output_paths["hypothesis_status"])
    atomic_csv(scan_manifest, output_paths["scan_manifest"])

    summary_json = {
        "dataset_root": str(dataset_root),
        "failure_root": str(failure_root),
        "x_shape": list(x.shape),
        "x_dtype": str(x.dtype),
        "x_is_memmap": x_is_memmap,
        "all_samples_scanned": all_samples_scanned,
        "runs_summarized": int(run_summary["run_id"].nunique()),
        "selected_runs": selected_runs,
        "selected_run_count": len(selected_runs),
        "run_feature_summary_rows": len(run_summary),
        "router_summary_rows": len(router_summary),
        "temporal_summary_rows": len(temporal_summary),
        "direction_summary_rows": len(direction_summary),
        "spatial_summary_rows": len(spatial_summary),
        "pair_difference_rows": len(pair_differences),
        "pair_distance_rows": len(pair_distances),
        "plot_count": plot_count,
        "corrected_feature_taxonomy_applied": True,
        "stage7a_tables_modified": False,
        "true_attacker_mask_available_for_stage7e": True,
        "observed_normalized_saturation_only": True,
        "clipping_provenance_confirmed": False,
        "classifier_fitted": False,
        "nearest_run_analysis_performed": False,
        "node_correctness_grouping_performed": False,
        "inference_rerun": False,
        "training_performed": False,
        "overall_success": overall_success,
    }
    atomic_json(summary_json, output_paths["summary_json"])

    report = build_report(
        run_summary,
        pair_differences,
        hypothesis_status,
        selected_runs,
        all_samples_scanned,
        plot_count,
    )
    atomic_text(report, output_paths["report"])

    print("Stage 7B output files:", flush=True)
    for name, path in output_paths.items():
        print(f"  {name}: {path}", flush=True)
    print(f"  plots: {plot_directory}", flush=True)
    print("overall success: True", flush=True)


if __name__ == "__main__":
    main()
