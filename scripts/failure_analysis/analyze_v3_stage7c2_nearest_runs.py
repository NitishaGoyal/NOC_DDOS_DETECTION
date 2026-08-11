#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_VERSION = "stage7c2"
EPS = 1e-12
CONSTANT_TOLERANCE = 1e-12
CONSTANT_VIOLATION_PENALTY = 5.0
BLOCK_COUNT = 8
TOP_OVERALL = 10
TOP_CLASS = 5
CONTRIBUTION_TOP_N = 25

REPRESENTATIONS = [
    "R1_global_run_feature",
    "R2_feature_family",
    "R3_router_feature",
    "R4_temporal_position_feature",
    "R5_spatial_concentration",
]
COMBINED_REPRESENTATION = "R1_R5_equal_weight_combined"
ALL_REPRESENTATIONS = REPRESENTATIONS + [COMBINED_REPRESENTATION]

DISTANCE_METRICS = [
    "D1_standardized_rmse",
    "D2_robust_iqr_rmse",
    "D3_standardized_manhattan",
]

EXPECTED_DIMENSIONS = {
    "R1_global_run_feature": 264,
    "R2_feature_family": 66,
    "R3_router_feature": 1920,
    "R4_temporal_position_feature": 960,
    "R5_spatial_concentration": 96,
}

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

HARD_NORMAL_RUN = "N-3-7-8-12-Pmixed-R18-V3"
HARD_ATTACK_RUN = "N-5-10-Pbursty-R51-A-12-S20-V3"
DOMINANT_RUNS = [HARD_NORMAL_RUN, HARD_ATTACK_RUN]


# -----------------------------------------------------------------------------
# Atomic and staged output helpers
# -----------------------------------------------------------------------------

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


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def promote_file(staged: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        final.unlink()
    os.replace(staged, final)


def promote_directory(staged: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        shutil.rmtree(final)
    os.replace(staged, final)


# -----------------------------------------------------------------------------
# Input and metadata helpers
# -----------------------------------------------------------------------------

def read_csv(path: Path, string_columns: Iterable[str] = ()) -> pd.DataFrame:
    dtype = {column: str for column in string_columns}
    return pd.read_csv(
        path,
        keep_default_na=False,
        low_memory=False,
        dtype=dtype if dtype else None,
    )


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{name} missing columns: {sorted(missing)}")


def run_metadata_lookup(run_inventory: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in run_inventory.iterrows():
        run_id = str(row["run_id"])
        lookup[run_id] = {
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
    return lookup


def feature_lookup(taxonomy: pd.DataFrame) -> dict[int, dict[str, Any]]:
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


def class_relation(target_class: int, training_class: int) -> str:
    return "same_class" if target_class == training_class else "opposite_class"


def majority_label(classes: list[int]) -> str:
    counts = Counter(classes)
    if not counts:
        return "none"
    normal = counts.get(0, 0)
    attack = counts.get(1, 0)
    if normal == attack:
        return "tie"
    return "normal" if normal > attack else "attack"


def parse_int_list(value: Any) -> set[int]:
    text = str(value).strip()
    if not text:
        return set()
    return {int(token) for token in text.split("-") if token.strip().isdigit()}


# -----------------------------------------------------------------------------
# Canonical vector construction from Stage 7C1 V2 outputs
# -----------------------------------------------------------------------------

def canonical_training_dimensions(training_reference: pd.DataFrame) -> pd.DataFrame:
    required = [
        "run_id",
        "split",
        "true_graph",
        "representation",
        "dimension_id",
        "metric",
        "value",
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
    require_columns(training_reference, required, "training run reference")
    frame = training_reference[
        training_reference["representation"].isin(REPRESENTATIONS)
    ].copy()
    frame["run_id"] = frame["run_id"].astype(str)
    frame["dimension_id"] = frame["dimension_id"].astype(str)
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    if set(frame["split"].astype(str)) != {"train"}:
        raise RuntimeError("Training vector source contains a non-training split.")
    duplicates = frame.duplicated(["run_id", "dimension_id"], keep=False)
    if duplicates.any():
        raise RuntimeError(
            "Duplicate training vector dimensions detected:\n"
            + frame.loc[duplicates, ["run_id", "dimension_id"]]
            .head(10)
            .to_string(index=False)
        )
    return frame


def target_frame_from_coverage(path: Path) -> pd.DataFrame:
    frame = read_csv(
        path,
        string_columns=[
            "target_run_id",
            "target_split",
            "target_profile",
            "target_active_cores",
            "target_attackers",
            "reference_cohort",
            "reference_role",
            "representation",
            "dimension_id",
            "metric",
            "feature_name",
            "feature_family",
            "measurement_type",
            "flow_side",
            "direction",
            "target_provenance_table",
        ],
    )
    required = [
        "target_run_id",
        "target_split",
        "target_true_graph",
        "reference_cohort",
        "representation",
        "dimension_id",
        "metric",
        "target_value",
        "feature_index",
        "feature_name",
        "feature_family",
        "measurement_type",
        "flow_side",
        "direction",
        "router_id",
        "temporal_position",
        "metadata_dependent",
        "target_provenance_table",
    ]
    require_columns(frame, required, path.name)
    frame = frame[frame["reference_cohort"].eq("all_training")].copy()
    frame = frame.rename(
        columns={
            "target_run_id": "run_id",
            "target_split": "split",
            "target_true_graph": "true_graph",
            "target_profile": "profile",
            "target_active_cores": "active_cores",
            "target_attackers": "attackers",
            "target_attacker_count": "attacker_count",
            "target_strength": "strength",
            "target_value": "value",
            "target_provenance_table": "provenance_table",
        }
    )
    frame["run_id"] = frame["run_id"].astype(str)
    frame["dimension_id"] = frame["dimension_id"].astype(str)
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    duplicates = frame.duplicated(["run_id", "dimension_id"], keep=False)
    if duplicates.any():
        raise RuntimeError(
            f"Duplicate target dimensions in {path.name}:\n"
            + frame.loc[duplicates, ["run_id", "dimension_id"]]
            .head(10)
            .to_string(index=False)
        )
    return frame


def canonical_target_dimensions(paths: list[Path]) -> pd.DataFrame:
    frames = [target_frame_from_coverage(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["representation"].isin(REPRESENTATIONS)].copy()
    duplicates = combined.duplicated(["run_id", "dimension_id"], keep=False)
    if duplicates.any():
        raise RuntimeError(
            "Target dimensions overlap across coverage files:\n"
            + combined.loc[duplicates, ["run_id", "dimension_id"]]
            .head(10)
            .to_string(index=False)
        )
    return combined


def canonical_scaling_references(reference_table: pd.DataFrame) -> pd.DataFrame:
    required = [
        "reference_cohort",
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
        "training_mean",
        "training_std",
        "training_median",
        "training_iqr",
        "zero_variance_dimension",
        "zero_iqr_dimension",
        "provenance_table",
    ]
    require_columns(reference_table, required, "training dimension reference")
    frame = reference_table[
        reference_table["reference_cohort"].eq("all_training")
        & reference_table["representation"].isin(REPRESENTATIONS)
    ].copy()
    duplicates = frame.duplicated("dimension_id", keep=False)
    if duplicates.any():
        raise RuntimeError(
            "Duplicate all-training scaling dimensions detected:\n"
            + frame.loc[duplicates, ["representation", "dimension_id"]]
            .head(10)
            .to_string(index=False)
        )
    for column in ["training_mean", "training_std", "training_median", "training_iqr"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def validate_vector_dimensions(
    training: pd.DataFrame,
    targets: pd.DataFrame,
    references: pd.DataFrame,
) -> tuple[bool, dict[str, list[str]]]:
    dimension_map: dict[str, list[str]] = {}
    valid = True
    for representation in REPRESENTATIONS:
        train_rep = training[training["representation"].eq(representation)]
        target_rep = targets[targets["representation"].eq(representation)]
        ref_rep = references[references["representation"].eq(representation)]
        expected = EXPECTED_DIMENSIONS[representation]
        ref_dims = sorted(ref_rep["dimension_id"].astype(str).unique())
        dimension_map[representation] = ref_dims
        if len(ref_dims) != expected:
            valid = False
        train_counts = train_rep.groupby("run_id")["dimension_id"].nunique()
        target_counts = target_rep.groupby("run_id")["dimension_id"].nunique()
        if not train_counts.eq(expected).all() or not target_counts.eq(expected).all():
            valid = False
        if set(train_rep["dimension_id"].astype(str)) != set(ref_dims):
            valid = False
        if set(target_rep["dimension_id"].astype(str)) != set(ref_dims):
            valid = False
    return valid, dimension_map


def vector_manifest(
    dimensions: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
    vector_role: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (run_id, representation), group in dimensions.groupby(
        ["run_id", "representation"], sort=True
    ):
        meta = metadata[str(run_id)]
        rows.append(
            {
                "vector_role": vector_role,
                **meta,
                "representation": representation,
                "dimension_count": int(group["dimension_id"].nunique()),
                "all_values_finite": bool(
                    np.isfinite(group["value"].to_numpy(dtype=float)).all()
                ),
                "dimension_id_minimum": str(group["dimension_id"].min()),
                "dimension_id_maximum": str(group["dimension_id"].max()),
                "source_provenance_tables": ";".join(
                    sorted(set(group["provenance_table"].astype(str)))
                ),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Distance calculation
# -----------------------------------------------------------------------------

def scaled_difference(
    target: np.ndarray,
    training: np.ndarray,
    scale: np.ndarray,
    constant_penalty: float = CONSTANT_VIOLATION_PENALTY,
) -> tuple[np.ndarray, np.ndarray]:
    raw_difference = target[None, :] - training
    variable = scale > CONSTANT_TOLERANCE
    result = np.zeros_like(raw_difference, dtype=np.float64)
    result[:, variable] = raw_difference[:, variable] / scale[variable]
    constant_mismatch = np.zeros_like(raw_difference, dtype=bool)
    if np.any(~variable):
        constant_mismatch[:, ~variable] = (
            np.abs(raw_difference[:, ~variable]) > CONSTANT_TOLERANCE
        )
        result[:, ~variable] = np.where(
            constant_mismatch[:, ~variable],
            np.sign(raw_difference[:, ~variable]) * constant_penalty,
            0.0,
        )
    return result, constant_mismatch


def metric_distance(metric: str, scaled: np.ndarray) -> np.ndarray:
    if metric in {"D1_standardized_rmse", "D2_robust_iqr_rmse"}:
        return np.sqrt(np.mean(np.square(scaled), axis=1))
    if metric == "D3_standardized_manhattan":
        return np.mean(np.abs(scaled), axis=1)
    raise KeyError(metric)


def matrices_for_representation(
    training_dimensions: pd.DataFrame,
    target_dimensions: pd.DataFrame,
    references: pd.DataFrame,
    representation: str,
) -> dict[str, Any]:
    ref = references[references["representation"].eq(representation)].copy()
    ref = ref.sort_values("dimension_id")
    dimensions = ref["dimension_id"].astype(str).tolist()

    train = training_dimensions[
        training_dimensions["representation"].eq(representation)
    ].pivot(index="run_id", columns="dimension_id", values="value")
    target = target_dimensions[
        target_dimensions["representation"].eq(representation)
    ].pivot(index="run_id", columns="dimension_id", values="value")

    train = train.reindex(columns=dimensions)
    target = target.reindex(columns=dimensions)
    if train.isna().any().any() or target.isna().any().any():
        raise RuntimeError(f"Missing aligned vector value in {representation}.")

    return {
        "representation": representation,
        "dimension_metadata": ref.reset_index(drop=True),
        "dimension_ids": dimensions,
        "training_run_ids": train.index.astype(str).tolist(),
        "target_run_ids": target.index.astype(str).tolist(),
        "training_matrix": train.to_numpy(dtype=np.float64),
        "target_matrix": target.to_numpy(dtype=np.float64),
        "std_scale": ref["training_std"].to_numpy(dtype=np.float64),
        "iqr_scale": ref["training_iqr"].to_numpy(dtype=np.float64),
    }


def calculate_representation_distances(
    matrices: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    representation = str(matrices["representation"])
    training_ids: list[str] = matrices["training_run_ids"]
    target_ids: list[str] = matrices["target_run_ids"]
    training_matrix: np.ndarray = matrices["training_matrix"]
    target_matrix: np.ndarray = matrices["target_matrix"]
    std_scale: np.ndarray = matrices["std_scale"]
    iqr_scale: np.ndarray = matrices["iqr_scale"]

    for target_index, target_run_id in enumerate(target_ids):
        target_vector = target_matrix[target_index]
        std_diff, std_constant = scaled_difference(
            target_vector, training_matrix, std_scale
        )
        robust_diff, iqr_constant = scaled_difference(
            target_vector, training_matrix, iqr_scale
        )
        metric_payload = {
            "D1_standardized_rmse": (std_diff, std_constant),
            "D2_robust_iqr_rmse": (robust_diff, iqr_constant),
            "D3_standardized_manhattan": (std_diff, std_constant),
        }
        target_meta = metadata[target_run_id]

        for metric, (differences, constant_mismatch) in metric_payload.items():
            distances = metric_distance(metric, differences)
            for train_index, training_run_id in enumerate(training_ids):
                training_meta = metadata[training_run_id]
                rows.append(
                    {
                        "target_run_id": target_run_id,
                        "target_split": target_meta["split"],
                        "target_true_graph": target_meta["true_graph"],
                        "target_profile": target_meta["profile"],
                        "training_run_id": training_run_id,
                        "training_split": training_meta["split"],
                        "training_true_graph": training_meta["true_graph"],
                        "training_profile": training_meta["profile"],
                        "training_active_cores": training_meta["active_cores"],
                        "training_attackers": training_meta["attackers"],
                        "training_strength": training_meta["strength"],
                        "training_attacker_count": training_meta["attacker_count"],
                        "training_seed": training_meta["seed"],
                        "class_relation": class_relation(
                            target_meta["true_graph"], training_meta["true_graph"]
                        ),
                        "representation": representation,
                        "distance_metric": metric,
                        "distance": float(distances[train_index]),
                        "dimension_count": int(differences.shape[1]),
                        "variable_dimension_count": int(
                            np.sum(
                                (std_scale if metric != "D2_robust_iqr_rmse" else iqr_scale)
                                > CONSTANT_TOLERANCE
                            )
                        ),
                        "constant_dimension_count": int(
                            np.sum(
                                (std_scale if metric != "D2_robust_iqr_rmse" else iqr_scale)
                                <= CONSTANT_TOLERANCE
                            )
                        ),
                        "constant_dimension_violation_count": int(
                            constant_mismatch[train_index].sum()
                        ),
                        "scaling_reference": "all_training",
                        "constant_violation_penalty": CONSTANT_VIOLATION_PENALTY,
                    }
                )
    return pd.DataFrame(rows)


def add_combined_distances(distance_results: pd.DataFrame) -> pd.DataFrame:
    source = distance_results[
        distance_results["representation"].isin(REPRESENTATIONS)
    ].copy()
    grouped_columns = [
        "target_run_id",
        "target_split",
        "target_true_graph",
        "target_profile",
        "training_run_id",
        "training_split",
        "training_true_graph",
        "training_profile",
        "training_active_cores",
        "training_attackers",
        "training_strength",
        "training_attacker_count",
        "training_seed",
        "class_relation",
        "distance_metric",
    ]
    combined = (
        source.groupby(grouped_columns, as_index=False)
        .agg(
            distance=("distance", "mean"),
            representation_count=("representation", "nunique"),
            dimension_count=("dimension_count", "sum"),
            variable_dimension_count=("variable_dimension_count", "sum"),
            constant_dimension_count=("constant_dimension_count", "sum"),
            constant_dimension_violation_count=(
                "constant_dimension_violation_count",
                "sum",
            ),
        )
    )
    if not combined["representation_count"].eq(len(REPRESENTATIONS)).all():
        raise RuntimeError("Combined distance does not contain all five representations.")
    combined["representation"] = COMBINED_REPRESENTATION
    combined["scaling_reference"] = "all_training"
    combined["constant_violation_penalty"] = CONSTANT_VIOLATION_PENALTY
    combined["combined_weight_rule"] = "equal_weight_mean_of_R1_R5_distances"
    source["combined_weight_rule"] = "not_applicable"
    return pd.concat([source, combined[source.columns]], ignore_index=True)


def rank_distances(distance_results: pd.DataFrame) -> pd.DataFrame:
    ranked_parts: list[pd.DataFrame] = []
    group_columns = ["target_run_id", "representation", "distance_metric"]
    for _, group in distance_results.groupby(group_columns, sort=True):
        ordered = group.sort_values(
            ["distance", "training_run_id"], ascending=[True, True]
        ).copy()
        ordered["rank_overall"] = np.arange(1, len(ordered) + 1)
        ordered["rank_within_class_relation"] = (
            ordered.groupby("class_relation").cumcount() + 1
        )
        ranked_parts.append(ordered)
    return pd.concat(ranked_parts, ignore_index=True)


def nearest_tables(ranked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = ranked[ranked["rank_overall"] <= TOP_OVERALL].copy()
    same = ranked[
        ranked["class_relation"].eq("same_class")
        & (ranked["rank_within_class_relation"] <= TOP_CLASS)
    ].copy()
    opposite = ranked[
        ranked["class_relation"].eq("opposite_class")
        & (ranked["rank_within_class_relation"] <= TOP_CLASS)
    ].copy()
    return overall, same, opposite


def class_margin_tables(
    ranked: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    margin_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    group_columns = ["target_run_id", "representation", "distance_metric"]

    for keys, group in ranked.groupby(group_columns, sort=True):
        target_run_id, representation, metric = keys
        group = group.sort_values("rank_overall")
        same = group[group["class_relation"].eq("same_class")]
        opposite = group[group["class_relation"].eq("opposite_class")]
        if same.empty or opposite.empty:
            raise RuntimeError(f"Empty same/opposite class set for {keys}.")

        nearest_same = same.iloc[0]
        nearest_opposite = opposite.iloc[0]
        same_distance = float(nearest_same["distance"])
        opposite_distance = float(nearest_opposite["distance"])
        margin = opposite_distance - same_distance
        ratio = opposite_distance / max(same_distance, EPS)

        row = {
            "target_run_id": target_run_id,
            "target_split": str(group.iloc[0]["target_split"]),
            "target_true_graph": int(group.iloc[0]["target_true_graph"]),
            "representation": representation,
            "distance_metric": metric,
            "nearest_same_class_run_id": str(nearest_same["training_run_id"]),
            "nearest_same_class_distance": same_distance,
            "nearest_opposite_class_run_id": str(nearest_opposite["training_run_id"]),
            "nearest_opposite_class_distance": opposite_distance,
            "class_margin_opposite_minus_same": margin,
            "distance_ratio_opposite_over_same": ratio,
            "margin_interpretation": (
                "same_class_closer" if margin > 0 else
                "opposite_class_closer" if margin < 0 else "equal_distance"
            ),
            "mean_top3_same_class_distance": float(same.head(3)["distance"].mean()),
            "mean_top3_opposite_class_distance": float(opposite.head(3)["distance"].mean()),
            "mean_top5_same_class_distance": float(same.head(5)["distance"].mean()),
            "mean_top5_opposite_class_distance": float(opposite.head(5)["distance"].mean()),
        }
        for k in [3, 5, 10]:
            top = group.head(k)
            classes = top["training_true_graph"].astype(int).tolist()
            row[f"majority_class_top{k}"] = majority_label(classes)
            row[f"normal_count_top{k}"] = int(sum(value == 0 for value in classes))
            row[f"attack_count_top{k}"] = int(sum(value == 1 for value in classes))
            topk_rows.append(
                {
                    "target_run_id": target_run_id,
                    "target_true_graph": int(group.iloc[0]["target_true_graph"]),
                    "representation": representation,
                    "distance_metric": metric,
                    "k": k,
                    "normal_neighbor_count": int(sum(value == 0 for value in classes)),
                    "attack_neighbor_count": int(sum(value == 1 for value in classes)),
                    "majority_class": majority_label(classes),
                    "majority_matches_target": bool(
                        majority_label(classes)
                        == ("normal" if int(group.iloc[0]["target_true_graph"]) == 0 else "attack")
                    ),
                    "neighbor_run_ids": ";".join(top["training_run_id"].astype(str)),
                }
            )
        margin_rows.append(row)

    return pd.DataFrame(margin_rows), pd.DataFrame(topk_rows)


# -----------------------------------------------------------------------------
# Dimension-level contributions
# -----------------------------------------------------------------------------

def contribution_vector(
    metric: str,
    target: np.ndarray,
    training: np.ndarray,
    std_scale: np.ndarray,
    iqr_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if metric == "D2_robust_iqr_rmse":
        scaled, mismatch = scaled_difference(target, training[None, :], iqr_scale)
        signed = scaled[0]
        contribution = np.square(signed) / len(signed)
        scale = iqr_scale
    elif metric == "D1_standardized_rmse":
        scaled, mismatch = scaled_difference(target, training[None, :], std_scale)
        signed = scaled[0]
        contribution = np.square(signed) / len(signed)
        scale = std_scale
    elif metric == "D3_standardized_manhattan":
        scaled, mismatch = scaled_difference(target, training[None, :], std_scale)
        signed = scaled[0]
        contribution = np.abs(signed) / len(signed)
        scale = std_scale
    else:
        raise KeyError(metric)
    return signed, contribution, mismatch[0]


def build_dimension_contributions(
    matrices_by_representation: dict[str, dict[str, Any]],
    ranked: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for target_run_id in DOMINANT_RUNS:
        for representation in REPRESENTATIONS:
            matrices = matrices_by_representation[representation]
            target_index = matrices["target_run_ids"].index(target_run_id)
            target_vector = matrices["target_matrix"][target_index]
            dimension_metadata = matrices["dimension_metadata"].reset_index(drop=True)

            for metric in DISTANCE_METRICS:
                subset = ranked[
                    ranked["target_run_id"].eq(target_run_id)
                    & ranked["representation"].eq(representation)
                    & ranked["distance_metric"].eq(metric)
                ]
                same_row = subset[subset["class_relation"].eq("same_class")].sort_values(
                    "rank_within_class_relation"
                ).iloc[0]
                opposite_row = subset[
                    subset["class_relation"].eq("opposite_class")
                ].sort_values("rank_within_class_relation").iloc[0]

                neighbor_payload: dict[str, dict[str, Any]] = {}
                for role, neighbor_row in [
                    ("nearest_same_class", same_row),
                    ("nearest_opposite_class", opposite_row),
                ]:
                    neighbor_run_id = str(neighbor_row["training_run_id"])
                    train_index = matrices["training_run_ids"].index(neighbor_run_id)
                    train_vector = matrices["training_matrix"][train_index]
                    signed, contribution, mismatch = contribution_vector(
                        metric,
                        target_vector,
                        train_vector,
                        matrices["std_scale"],
                        matrices["iqr_scale"],
                    )
                    neighbor_payload[role] = {
                        "run_id": neighbor_run_id,
                        "training_vector": train_vector,
                        "signed": signed,
                        "contribution": contribution,
                        "mismatch": mismatch,
                    }
                    order = np.argsort(-contribution)[:CONTRIBUTION_TOP_N]
                    for rank, index in enumerate(order, start=1):
                        meta = dimension_metadata.iloc[index]
                        rows.append(
                            {
                                "target_run_id": target_run_id,
                                "representation": representation,
                                "distance_metric": metric,
                                "contribution_role": role,
                                "neighbor_run_id": neighbor_run_id,
                                "contribution_rank": rank,
                                "dimension_id": str(meta["dimension_id"]),
                                "feature_index": int(meta["feature_index"]),
                                "feature_name": str(meta["feature_name"]),
                                "feature_family": str(meta["feature_family"]),
                                "router_id": int(meta["router_id"]),
                                "temporal_position": int(meta["temporal_position"]),
                                "metric": str(meta["metric"]),
                                "target_value": float(target_vector[index]),
                                "training_run_value": float(train_vector[index]),
                                "training_scaling_value": float(
                                    matrices["iqr_scale"][index]
                                    if metric == "D2_robust_iqr_rmse"
                                    else matrices["std_scale"][index]
                                ),
                                "signed_scaled_difference": float(signed[index]),
                                "absolute_scaled_difference": float(abs(signed[index])),
                                "distance_contribution": float(contribution[index]),
                                "constant_dimension_violation": bool(mismatch[index]),
                                "contribution_difference_opposite_minus_same": np.nan,
                                "contribution_interpretation": (
                                    "distance_to_selected_neighbor"
                                ),
                            }
                        )

                same_contrib = neighbor_payload["nearest_same_class"]["contribution"]
                opposite_contrib = neighbor_payload["nearest_opposite_class"]["contribution"]
                contrast = opposite_contrib - same_contrib
                order = np.argsort(-np.abs(contrast))[:CONTRIBUTION_TOP_N]
                for rank, index in enumerate(order, start=1):
                    meta = dimension_metadata.iloc[index]
                    rows.append(
                        {
                            "target_run_id": target_run_id,
                            "representation": representation,
                            "distance_metric": metric,
                            "contribution_role": "opposite_minus_same_contrast",
                            "neighbor_run_id": (
                                neighbor_payload["nearest_opposite_class"]["run_id"]
                                + " minus "
                                + neighbor_payload["nearest_same_class"]["run_id"]
                            ),
                            "contribution_rank": rank,
                            "dimension_id": str(meta["dimension_id"]),
                            "feature_index": int(meta["feature_index"]),
                            "feature_name": str(meta["feature_name"]),
                            "feature_family": str(meta["feature_family"]),
                            "router_id": int(meta["router_id"]),
                            "temporal_position": int(meta["temporal_position"]),
                            "metric": str(meta["metric"]),
                            "target_value": float(target_vector[index]),
                            "training_run_value": np.nan,
                            "training_scaling_value": float(
                                matrices["iqr_scale"][index]
                                if metric == "D2_robust_iqr_rmse"
                                else matrices["std_scale"][index]
                            ),
                            "signed_scaled_difference": np.nan,
                            "absolute_scaled_difference": np.nan,
                            "distance_contribution": float(abs(contrast[index])),
                            "constant_dimension_violation": bool(
                                neighbor_payload["nearest_same_class"]["mismatch"][index]
                                or neighbor_payload["nearest_opposite_class"]["mismatch"][index]
                            ),
                            "contribution_difference_opposite_minus_same": float(
                                contrast[index]
                            ),
                            "contribution_interpretation": (
                                "positive_supports_same_class_closer;"
                                "negative_supports_opposite_class_closer"
                            ),
                        }
                    )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Block vector reconstruction from the original tensor
# -----------------------------------------------------------------------------

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


def top_shares(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.minimum(values.min(axis=1, keepdims=True), 0.0)
    totals = shifted.sum(axis=1)
    ordered = np.sort(shifted, axis=1)
    top_one = ordered[:, -1]
    top_two = ordered[:, -2:].sum(axis=1)
    one_share = np.divide(top_one, totals, out=np.zeros_like(top_one), where=totals > EPS)
    two_share = np.divide(top_two, totals, out=np.zeros_like(top_two), where=totals > EPS)
    return one_share, two_share


def block_vector_rows(
    run_data: np.ndarray,
    run_id: str,
    block_index: int,
    feature_meta: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    windows, routers, positions, features = run_data.shape

    r1_feature_rows: list[dict[str, Any]] = []
    for feature_index in range(features):
        values = run_data[..., feature_index]
        q = np.quantile(values, [0.05, 0.50, 0.95])
        metrics = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "median": float(q[1]),
            "p05": float(q[0]),
            "p95": float(q[2]),
            "fraction_equal_0": float(np.mean(values == 0.0)),
            "fraction_equal_1": float(np.mean(values == 1.0)),
            "fraction_above_0_95": float(np.mean(values > 0.95)),
            "mean_temporal_variance": float(np.var(values, axis=2).mean()),
            "mean_absolute_adjacent_temporal_change": float(
                np.abs(np.diff(values, axis=2)).mean()
            ),
            "mean_spatial_variance": float(np.var(values, axis=1).mean()),
        }
        meta = feature_meta[feature_index]
        for metric, value in metrics.items():
            row = {
                "block_target_id": f"{run_id}::block_{block_index}",
                "run_id": run_id,
                "block_index": block_index,
                "representation": "R1_global_run_feature",
                "dimension_id": dimension_id(
                    "R1_global_run_feature", metric, feature_index=feature_index
                ),
                "value": value,
                **meta,
                "router_id": -1,
                "temporal_position": -1,
                "metric": metric,
            }
            rows.append(row)
            r1_feature_rows.append(row)

    r1_frame = pd.DataFrame(r1_feature_rows)
    for family, family_frame in r1_frame.groupby("feature_family", sort=True):
        for metric in R1_METRICS:
            values = family_frame[family_frame["metric"].eq(metric)]["value"]
            rows.append(
                {
                    "block_target_id": f"{run_id}::block_{block_index}",
                    "run_id": run_id,
                    "block_index": block_index,
                    "representation": "R2_feature_family",
                    "dimension_id": dimension_id(
                        "R2_feature_family", metric, feature_family=str(family)
                    ),
                    "value": float(values.mean()),
                    "feature_index": -1,
                    "feature_name": "",
                    "feature_family": str(family),
                    "measurement_type": "",
                    "flow_side": "",
                    "direction": "",
                    "router_id": -1,
                    "temporal_position": -1,
                    "metric": metric,
                }
            )

    temporal_variance = np.var(run_data, axis=2)
    temporal_excursion = run_data.max(axis=2) - run_data.min(axis=2)
    for router_id in range(routers):
        for feature_index in range(features):
            values = run_data[:, router_id, :, feature_index]
            meta = feature_meta[feature_index]
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
                    {
                        "block_target_id": f"{run_id}::block_{block_index}",
                        "run_id": run_id,
                        "block_index": block_index,
                        "representation": "R3_router_feature",
                        "dimension_id": dimension_id(
                            "R3_router_feature",
                            metric,
                            feature_index=feature_index,
                            router_id=router_id,
                        ),
                        "value": value,
                        **meta,
                        "router_id": router_id,
                        "temporal_position": -1,
                        "metric": metric,
                    }
                )

    for temporal_position in range(positions):
        for feature_index in range(features):
            values = run_data[:, :, temporal_position, feature_index]
            meta = feature_meta[feature_index]
            previous_change = (
                0.0
                if temporal_position == 0
                else float(
                    np.abs(
                        run_data[:, :, temporal_position, feature_index]
                        - run_data[:, :, temporal_position - 1, feature_index]
                    ).mean()
                )
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
                    {
                        "block_target_id": f"{run_id}::block_{block_index}",
                        "run_id": run_id,
                        "block_index": block_index,
                        "representation": "R4_temporal_position_feature",
                        "dimension_id": dimension_id(
                            "R4_temporal_position_feature",
                            metric,
                            feature_index=feature_index,
                            temporal_position=temporal_position,
                        ),
                        "value": value,
                        **meta,
                        "router_id": -1,
                        "temporal_position": temporal_position,
                        "metric": metric,
                    }
                )

    for feature_index in range(features):
        values = (
            run_data[..., feature_index]
            .transpose(0, 2, 1)
            .reshape(windows * positions, routers)
        )
        meta = feature_meta[feature_index]
        router_std = values.std(axis=1)
        maximum_minus_median = values.max(axis=1) - np.median(values, axis=1)
        top_one, top_two = top_shares(values)
        metrics = {
            "mean_router_std": float(router_std.mean()),
            "mean_maximum_minus_median": float(maximum_minus_median.mean()),
            "mean_top_router_share": float(top_one.mean()),
            "mean_top_two_router_share": float(top_two.mean()),
        }
        for metric, value in metrics.items():
            rows.append(
                {
                    "block_target_id": f"{run_id}::block_{block_index}",
                    "run_id": run_id,
                    "block_index": block_index,
                    "representation": "R5_spatial_concentration",
                    "dimension_id": dimension_id(
                        "R5_spatial_concentration",
                        metric,
                        feature_index=feature_index,
                    ),
                    "value": value,
                    **meta,
                    "router_id": -1,
                    "temporal_position": -1,
                    "metric": metric,
                }
            )

    return pd.DataFrame(rows)


def deterministic_block_bounds(number_of_windows: int, block_count: int) -> list[tuple[int, int]]:
    bounds = []
    for block_index in range(block_count):
        start = math.floor(block_index * number_of_windows / block_count)
        end = math.floor((block_index + 1) * number_of_windows / block_count)
        bounds.append((start, end))
    return bounds


def block_distances_and_margins(
    x: np.ndarray,
    alignment: pd.DataFrame,
    matrices_by_representation: dict[str, dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    feature_meta: dict[int, dict[str, Any]],
) -> tuple[pd.DataFrame, bool]:
    rows: list[dict[str, Any]] = []
    deterministic_valid = True

    for run_id in DOMINANT_RUNS:
        run_alignment = alignment[alignment["run_id"].eq(run_id)].sort_values(
            "run_window_index"
        )
        indices = run_alignment["dataset_index"].to_numpy(dtype=np.int64)
        if len(indices) != 3293:
            raise RuntimeError(f"Dominant run {run_id} has {len(indices)} windows.")
        bounds = deterministic_block_bounds(len(indices), BLOCK_COUNT)
        covered = []
        for start, end in bounds:
            covered.extend(range(start, end))
        deterministic_valid = deterministic_valid and covered == list(range(len(indices)))

        for block_index, (start, end) in enumerate(bounds):
            block_indices = indices[start:end]
            run_data = np.asarray(x[block_indices], dtype=np.float32)
            block_dimensions = block_vector_rows(
                run_data,
                run_id,
                block_index,
                feature_meta,
            )
            block_id = f"{run_id}::block_{block_index}"
            rep_distances: dict[str, dict[str, np.ndarray]] = {}

            for representation in REPRESENTATIONS:
                matrices = matrices_by_representation[representation]
                ref_dims = matrices["dimension_ids"]
                vector = (
                    block_dimensions[
                        block_dimensions["representation"].eq(representation)
                    ]
                    .set_index("dimension_id")["value"]
                    .reindex(ref_dims)
                )
                if vector.isna().any():
                    raise RuntimeError(
                        f"Block vector alignment failed for {block_id}, {representation}."
                    )
                target_vector = vector.to_numpy(dtype=np.float64)
                training_matrix = matrices["training_matrix"]
                std_diff, _ = scaled_difference(
                    target_vector, training_matrix, matrices["std_scale"]
                )
                robust_diff, _ = scaled_difference(
                    target_vector, training_matrix, matrices["iqr_scale"]
                )
                rep_distances[representation] = {
                    "D1_standardized_rmse": metric_distance(
                        "D1_standardized_rmse", std_diff
                    ),
                    "D2_robust_iqr_rmse": metric_distance(
                        "D2_robust_iqr_rmse", robust_diff
                    ),
                    "D3_standardized_manhattan": metric_distance(
                        "D3_standardized_manhattan", std_diff
                    ),
                }

            for metric in DISTANCE_METRICS:
                distance_by_representation = {
                    representation: rep_distances[representation][metric]
                    for representation in REPRESENTATIONS
                }
                distance_by_representation[COMBINED_REPRESENTATION] = np.mean(
                    np.vstack(
                        [distance_by_representation[rep] for rep in REPRESENTATIONS]
                    ),
                    axis=0,
                )

                for representation in ALL_REPRESENTATIONS:
                    matrices = matrices_by_representation[REPRESENTATIONS[0]]
                    training_ids = matrices["training_run_ids"]
                    distances = distance_by_representation[representation]
                    records = []
                    for train_index, training_run_id in enumerate(training_ids):
                        records.append(
                            {
                                "training_run_id": training_run_id,
                                "training_true_graph": metadata[training_run_id]["true_graph"],
                                "distance": float(distances[train_index]),
                            }
                        )
                    frame = pd.DataFrame(records).sort_values(
                        ["distance", "training_run_id"]
                    )
                    target_class = metadata[run_id]["true_graph"]
                    same = frame[frame["training_true_graph"].eq(target_class)]
                    opposite = frame[~frame["training_true_graph"].eq(target_class)]
                    nearest_same = same.iloc[0]
                    nearest_opposite = opposite.iloc[0]
                    margin = float(nearest_opposite["distance"] - nearest_same["distance"])
                    top5 = frame.head(5)
                    rows.append(
                        {
                            "target_run_id": run_id,
                            "target_true_graph": target_class,
                            "block_index": block_index,
                            "block_start_window_index": start,
                            "block_end_window_index_exclusive": end,
                            "block_window_count": end - start,
                            "representation": representation,
                            "distance_metric": metric,
                            "nearest_same_class_run_id": str(
                                nearest_same["training_run_id"]
                            ),
                            "nearest_same_class_distance": float(
                                nearest_same["distance"]
                            ),
                            "nearest_opposite_class_run_id": str(
                                nearest_opposite["training_run_id"]
                            ),
                            "nearest_opposite_class_distance": float(
                                nearest_opposite["distance"]
                            ),
                            "class_margin_opposite_minus_same": margin,
                            "margin_interpretation": (
                                "same_class_closer" if margin > 0 else
                                "opposite_class_closer" if margin < 0 else "equal_distance"
                            ),
                            "top5_majority_class": majority_label(
                                top5["training_true_graph"].astype(int).tolist()
                            ),
                            "top5_neighbor_run_ids": ";".join(
                                top5["training_run_id"].astype(str)
                            ),
                            "scaling_reference": "frozen_all_training_stage7c1_v2",
                            "block_reference_refitted": False,
                        }
                    )
            del run_data

    return pd.DataFrame(rows), deterministic_valid


# -----------------------------------------------------------------------------
# Missing-scenario evidence
# -----------------------------------------------------------------------------

def metadata_similarity(target: dict[str, Any], neighbor: dict[str, Any]) -> str:
    matches = []
    if target["profile"] == neighbor["profile"]:
        matches.append("same_profile")
    if target["active_core_count"] == neighbor["active_core_count"]:
        matches.append("same_active_core_count")
    if parse_int_list(target["active_cores"]) == parse_int_list(neighbor["active_cores"]):
        matches.append("same_active_core_placement")
    if target["attacker_count"] == neighbor["attacker_count"]:
        matches.append("same_attacker_count")
    if target["strength"] == neighbor["strength"]:
        matches.append("same_strength")
    if parse_int_list(target["attackers"]) == parse_int_list(neighbor["attackers"]):
        matches.append("same_attacker_placement")
    return ";".join(matches) if matches else "no_exact_metadata_match"


def build_missing_scenario_evidence(
    margins: pd.DataFrame,
    contributions: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dominant = margins[margins["target_run_id"].isin(DOMINANT_RUNS)]

    for _, row in dominant.iterrows():
        target_run_id = str(row["target_run_id"])
        target_meta = metadata[target_run_id]
        same_run = str(row["nearest_same_class_run_id"])
        opposite_run = str(row["nearest_opposite_class_run_id"])
        same_meta = metadata[same_run]
        opposite_meta = metadata[opposite_run]
        rep = str(row["representation"])
        metric = str(row["distance_metric"])

        contribution_source = contributions[
            contributions["target_run_id"].eq(target_run_id)
            & contributions["distance_metric"].eq(metric)
            & contributions["contribution_role"].eq(
                "opposite_minus_same_contrast"
            )
        ].copy()
        if rep != COMBINED_REPRESENTATION:
            contribution_source = contribution_source[
                contribution_source["representation"].eq(rep)
            ]
        contribution_source["absolute_contrast"] = contribution_source[
            "contribution_difference_opposite_minus_same"
        ].abs()
        top_contribution = contribution_source.sort_values(
            "absolute_contrast", ascending=False
        ).head(5)
        dimension_labels = []
        for _, contribution_row in top_contribution.iterrows():
            feature_label = str(contribution_row["feature_name"]).strip()
            if not feature_label:
                feature_label = (
                    "feature_family=" + str(contribution_row["feature_family"])
                )
            dimension_labels.append(
                feature_label
                + "|"
                + str(contribution_row["metric"])
                + "|r="
                + str(int(contribution_row["router_id"]))
                + "|t="
                + str(int(contribution_row["temporal_position"]))
            )
        dimensions = ";".join(dimension_labels)

        margin = float(row["class_margin_opposite_minus_same"])
        if target_meta["true_graph"] == 0:
            evidence = (
                "Add benign configurations around the target placement and its "
                "nearest attack-like neighbourhood; preserve mixed traffic and "
                "router-local count/IFD structure."
            )
        else:
            evidence = (
                "Add weak attacks matching the target profile, active-core placement, "
                "attacker placement and count–IFD structure, especially where normal "
                "training neighbours are closer."
            )

        rows.append(
            {
                "target_run_id": target_run_id,
                "target_true_graph": target_meta["true_graph"],
                "representation": rep,
                "distance_metric": metric,
                "class_margin_opposite_minus_same": margin,
                "geometry_result": (
                    "same_class_closer" if margin > 0 else
                    "opposite_class_closer" if margin < 0 else "equal_distance"
                ),
                "nearest_same_class_run_id": same_run,
                "nearest_same_class_metadata_matches": metadata_similarity(
                    target_meta, same_meta
                ),
                "nearest_opposite_class_run_id": opposite_run,
                "nearest_opposite_class_metadata_matches": metadata_similarity(
                    target_meta, opposite_meta
                ),
                "top_distance_contributing_dimensions": dimensions,
                "missing_scenario_evidence": evidence,
                "v4_design_finalized": False,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    path: Path,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (12, 8),
) -> None:
    figure, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(matrix.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
    axis.set_xticks(np.arange(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels(matrix.index, fontsize=8)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    figure.colorbar(image, ax=axis)
    save_figure(figure, path)


def make_plots(
    margins: pd.DataFrame,
    ranked: pd.DataFrame,
    topk: pd.DataFrame,
    contributions: pd.DataFrame,
    block_stability: pd.DataFrame,
    plot_dir: Path,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    primary = margins[
        margins["representation"].eq(COMBINED_REPRESENTATION)
        & margins["distance_metric"].eq("D1_standardized_rmse")
    ].sort_values("target_run_id")
    figure, axis = plt.subplots(figsize=(12, 6))
    x = np.arange(len(primary))
    width = 0.35
    axis.bar(x - width / 2, primary["nearest_same_class_distance"], width, label="same class")
    axis.bar(x + width / 2, primary["nearest_opposite_class_distance"], width, label="opposite class")
    axis.set_xticks(x)
    axis.set_xticklabels(primary["target_run_id"], rotation=70, ha="right")
    axis.set_ylabel("Distance")
    axis.set_title("Nearest same-class versus opposite-class distance")
    axis.legend()
    save_figure(figure, plot_dir / "01_nearest_same_vs_opposite_distance.png")

    d1 = margins[margins["distance_metric"].eq("D1_standardized_rmse")]
    matrix = d1.pivot(index="target_run_id", columns="representation", values="class_margin_opposite_minus_same")
    plot_heatmap(
        matrix,
        "Class margin by target and representation (positive = same class closer)",
        plot_dir / "02_class_margin_by_target_representation.png",
        "Representation",
        "Target run",
        figsize=(13, 9),
    )

    for run_id, number, label in [
        (HARD_NORMAL_RUN, "03", "hard_normal"),
        (HARD_ATTACK_RUN, "04", "hard_attack"),
    ]:
        subset = ranked[
            ranked["target_run_id"].eq(run_id)
            & ranked["representation"].eq(COMBINED_REPRESENTATION)
            & ranked["distance_metric"].eq("D1_standardized_rmse")
        ].sort_values("rank_overall").head(15)
        figure, axis = plt.subplots(figsize=(11, 7))
        axis.barh(subset["training_run_id"][::-1], subset["distance"][::-1])
        axis.set_xlabel("Equal-weight combined distance")
        axis.set_title(f"{run_id}: nearest training runs")
        save_figure(figure, plot_dir / f"{number}_{label}_nearest_training_runs.png")

    combined_topk = topk[
        topk["representation"].eq(COMBINED_REPRESENTATION)
        & topk["distance_metric"].eq("D1_standardized_rmse")
    ]
    matrix = combined_topk.pivot(index="target_run_id", columns="k", values="attack_neighbor_count")
    plot_heatmap(
        matrix,
        "Attack-neighbour count among top-k combined neighbours",
        plot_dir / "05_topk_neighbor_class_composition.png",
        "k",
        "Target run",
        figsize=(8, 9),
    )

    dominant_d1 = margins[
        margins["target_run_id"].isin(DOMINANT_RUNS)
        & margins["distance_metric"].eq("D1_standardized_rmse")
        & margins["representation"].isin(REPRESENTATIONS)
    ]
    matrix = dominant_d1.pivot(index="target_run_id", columns="representation", values="class_margin_opposite_minus_same")
    plot_heatmap(
        matrix,
        "Dominant-run R1–R5 class margins",
        plot_dir / "06_R1_R5_distance_comparison.png",
        "Representation",
        "Dominant run",
        figsize=(12, 5),
    )

    for run_id, number, label in [
        (HARD_NORMAL_RUN, "07", "hard_normal"),
        (HARD_ATTACK_RUN, "08", "hard_attack"),
    ]:
        subset = contributions[
            contributions["target_run_id"].eq(run_id)
            & contributions["representation"].eq("R3_router_feature")
            & contributions["distance_metric"].eq("D1_standardized_rmse")
            & contributions["contribution_role"].eq("opposite_minus_same_contrast")
        ].sort_values("contribution_rank").head(20)
        labels = (
            subset["feature_name"]
            + "|r="
            + subset["router_id"].astype(str)
            + "|"
            + subset["metric"]
        )
        figure, axis = plt.subplots(figsize=(12, 8))
        axis.barh(labels[::-1], subset["contribution_difference_opposite_minus_same"][::-1])
        axis.set_xlabel("Opposite contribution minus same contribution")
        axis.set_title(f"{run_id}: R3 dimensions driving class geometry")
        save_figure(figure, plot_dir / f"{number}_{label}_dimension_contributions.png")

    for run_id, number, label in [
        (HARD_NORMAL_RUN, "09", "hard_normal"),
        (HARD_ATTACK_RUN, "10", "hard_attack"),
    ]:
        subset = block_stability[
            block_stability["target_run_id"].eq(run_id)
            & block_stability["representation"].eq(COMBINED_REPRESENTATION)
            & block_stability["distance_metric"].eq("D1_standardized_rmse")
        ].sort_values("block_index")
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(subset["block_index"], subset["class_margin_opposite_minus_same"], marker="o")
        axis.axhline(0.0, linewidth=1)
        axis.set_xlabel("Block index")
        axis.set_ylabel("Class margin (opposite minus same)")
        axis.set_title(f"{run_id}: block-level class margin")
        save_figure(figure, plot_dir / f"{number}_{label}_block_class_margin.png")

    metadata_rows = ranked[
        ranked["target_run_id"].isin(DOMINANT_RUNS)
        & ranked["representation"].eq(COMBINED_REPRESENTATION)
        & ranked["distance_metric"].eq("D1_standardized_rmse")
        & (ranked["rank_overall"] <= 10)
    ].copy()
    metadata_rows["profile_match"] = (
        metadata_rows["target_profile"] == metadata_rows["training_profile"]
    ).astype(float)
    metadata_rows["class_match"] = (
        metadata_rows["class_relation"] == "same_class"
    ).astype(float)
    metadata_rows["strength_match"] = (
        metadata_rows["training_strength"] == 20
    ).astype(float)
    matrix = metadata_rows.pivot_table(
        index=["target_run_id", "training_run_id"],
        values=["profile_match", "class_match", "strength_match"],
        aggfunc="first",
    )
    plot_heatmap(
        matrix,
        "Nearest-run metadata match indicators",
        plot_dir / "11_nearest_run_metadata_comparison.png",
        "Metadata indicator",
        "Target / training run",
        figsize=(9, 13),
    )

    combined = margins[
        margins["representation"].eq(COMBINED_REPRESENTATION)
    ]
    matrix = combined.pivot(index="target_run_id", columns="distance_metric", values="class_margin_opposite_minus_same")
    plot_heatmap(
        matrix,
        "Combined representation class margin across distance metrics",
        plot_dir / "12_combined_representation_class_margin.png",
        "Distance metric",
        "Target run",
        figsize=(10, 9),
    )


# -----------------------------------------------------------------------------
# Report and validation helpers
# -----------------------------------------------------------------------------

def dominant_outcome(margins: pd.DataFrame, run_id: str) -> str:
    subset = margins[
        margins["target_run_id"].eq(run_id)
        & margins["representation"].eq(COMBINED_REPRESENTATION)
    ]
    signs = np.sign(subset["class_margin_opposite_minus_same"].to_numpy(dtype=float))
    positive = int(np.sum(signs > 0))
    negative = int(np.sum(signs < 0))
    if positive == len(subset):
        return "same class is closer under every predeclared combined distance"
    if negative == len(subset):
        return "opposite class is closer under every predeclared combined distance"
    return f"distance formulations disagree ({positive} same-class-closer, {negative} opposite-class-closer)"


def build_report(
    margins: pd.DataFrame,
    ranked: pd.DataFrame,
    block_stability: pd.DataFrame,
    missing_evidence: pd.DataFrame,
    plot_count: int,
) -> str:
    lines = [
        "# Stage 7C2 — Nearest Training-Run and Class-Margin Analysis",
        "",
        "Stage 7C2 used the frozen Stage 7C1 V2 all-training scaling reference. ",
        "No validation/test value influenced scaling, feature selection, metric choice, ",
        "representation weighting, neighbour count or class-margin definition.",
        "",
        "R6 raw-distribution coverage was excluded from the primary geometry because ",
        "its exact-boundary histogram interpretation remains unresolved.",
        "",
        "## Fixed geometry",
        "",
        "- D1: root mean squared difference divided by the all-training standard deviation.",
        "- D2: root mean squared difference divided by the all-training IQR.",
        "- D3: mean absolute difference divided by the all-training standard deviation.",
        "- Constant training dimensions contribute zero when matched and a fixed finite ",
        f"penalty of {CONSTANT_VIOLATION_PENALTY:g} when violated.",
        "- R1–R5 are measured separately.",
        "- The combined distance is the equal-weight mean of the five representation distances.",
        "- Class margin = nearest opposite-class distance minus nearest same-class distance.",
        "- Positive margin means same-class training is nearer.",
        "- Negative margin means opposite-class training is nearer.",
        "",
        "## Dominant-run verdict",
        "",
        f"- Hard normal: **{dominant_outcome(margins, HARD_NORMAL_RUN)}**.",
        f"- Hard attack: **{dominant_outcome(margins, HARD_ATTACK_RUN)}**.",
        "",
        "## Combined margins",
        "",
        "| Target | Metric | Nearest same | Same distance | Nearest opposite | Opposite distance | Margin |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    combined = margins[
        margins["target_run_id"].isin(DOMINANT_RUNS)
        & margins["representation"].eq(COMBINED_REPRESENTATION)
    ].sort_values(["target_run_id", "distance_metric"])
    for _, row in combined.iterrows():
        lines.append(
            f"| `{row['target_run_id']}` | {row['distance_metric']} | "
            f"`{row['nearest_same_class_run_id']}` | {row['nearest_same_class_distance']:.4f} | "
            f"`{row['nearest_opposite_class_run_id']}` | {row['nearest_opposite_class_distance']:.4f} | "
            f"{row['class_margin_opposite_minus_same']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Block stability",
            "",
            "Each dominant run was divided into eight deterministic contiguous blocks. ",
            "Block summaries used the frozen full-training scaling and full-run training vectors; ",
            "no block-specific reference was fitted.",
            "",
        ]
    )
    for run_id in DOMINANT_RUNS:
        subset = block_stability[
            block_stability["target_run_id"].eq(run_id)
            & block_stability["representation"].eq(COMBINED_REPRESENTATION)
            & block_stability["distance_metric"].eq("D1_standardized_rmse")
        ]
        positive = int((subset["class_margin_opposite_minus_same"] > 0).sum())
        negative = int((subset["class_margin_opposite_minus_same"] < 0).sum())
        lines.append(
            f"- `{run_id}`: {positive}/8 blocks same-class closer; "
            f"{negative}/8 blocks opposite-class closer."
        )

    lines.extend(
        [
            "",
            "## Missing-scenario evidence",
            "",
        ]
    )
    primary_evidence = missing_evidence[
        missing_evidence["representation"].eq(COMBINED_REPRESENTATION)
        & missing_evidence["distance_metric"].eq("D1_standardized_rmse")
    ]
    for _, row in primary_evidence.iterrows():
        lines.append(
            f"- `{row['target_run_id']}`: {row['geometry_result']}. "
            f"Nearest same-class `{row['nearest_same_class_run_id']}`; "
            f"nearest opposite-class `{row['nearest_opposite_class_run_id']}`. "
            f"{row['missing_scenario_evidence']}"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Nearest-run geometry describes the Stage 7C1 summary space; it is not a causal model.",
            "- Direction names are not interpreted physically before builder recovery.",
            "- R6 is excluded from the primary conclusion.",
            "- No classifier was fitted and no model inference was rerun.",
            "- V4 scenarios are suggested as evidence, not finalized here.",
            "",
            f"Plots created: **{plot_count}**.",
            "",
        ]
    )
    return "\n".join(lines)


def validation_summary_frame(validation: list[tuple[str, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"validation_check": name, "passed": bool(value)}
            for name, value in validation
        ]
    )


# -----------------------------------------------------------------------------
# Synthetic smoke test
# -----------------------------------------------------------------------------

def synthetic_smoke_test(verbose: bool = True) -> bool:
    dimensions = ["d1", "d2", "d3", "d4"]
    training_ids = ["normal_a", "normal_b", "attack_a", "attack_b"]
    training_classes = [0, 0, 1, 1]
    training_matrix = np.asarray(
        [
            [0.0, 0.1, 0.0, 1.0],
            [0.1, 0.0, 0.0, 1.0],
            [1.0, 0.9, 1.0, 1.0],
            [0.9, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    target_matrix = np.asarray([[0.05, 0.05, 0.0, 1.0]], dtype=np.float64)
    std_scale = np.std(training_matrix, axis=0)
    iqr_scale = np.quantile(training_matrix, 0.75, axis=0) - np.quantile(
        training_matrix, 0.25, axis=0
    )
    std_diff, constant_mismatch = scaled_difference(
        target_matrix[0], training_matrix, std_scale
    )
    robust_diff, _ = scaled_difference(target_matrix[0], training_matrix, iqr_scale)
    d1 = metric_distance("D1_standardized_rmse", std_diff)
    d2 = metric_distance("D2_robust_iqr_rmse", robust_diff)
    d3 = metric_distance("D3_standardized_manhattan", std_diff)
    distances_nonnegative = bool(
        np.all(d1 >= 0) and np.all(d2 >= 0) and np.all(d3 >= 0)
    )
    nearest_is_normal = training_classes[int(np.argmin(d1))] == 0
    constant_safe = bool(np.all(~constant_mismatch[:, 3]))

    rep_a = d1
    rep_b = d1 * 2.0
    combined = np.mean(np.vstack([rep_a, rep_b]), axis=0)
    equal_weight_valid = bool(np.allclose(combined, 1.5 * d1))

    frame = pd.DataFrame(
        {
            "target_run_id": "target",
            "target_split": "test",
            "target_true_graph": 0,
            "target_profile": "mixed",
            "training_run_id": training_ids,
            "training_split": "train",
            "training_true_graph": training_classes,
            "training_profile": ["mixed"] * 4,
            "training_active_cores": [""] * 4,
            "training_attackers": [""] * 4,
            "training_strength": [0, 0, 20, 20],
            "training_attacker_count": [0, 0, 1, 1],
            "training_seed": [1, 2, 3, 4],
            "class_relation": ["same_class", "same_class", "opposite_class", "opposite_class"],
            "representation": "R1_global_run_feature",
            "distance_metric": "D1_standardized_rmse",
            "distance": d1,
            "dimension_count": len(dimensions),
            "variable_dimension_count": 3,
            "constant_dimension_count": 1,
            "constant_dimension_violation_count": constant_mismatch.sum(axis=1),
            "scaling_reference": "all_training",
            "constant_violation_penalty": CONSTANT_VIOLATION_PENALTY,
            "combined_weight_rule": "not_applicable",
        }
    )
    ranked = rank_distances(frame)
    margins, _ = class_margin_tables(ranked)
    class_margin_valid = bool(
        margins.iloc[0]["class_margin_opposite_minus_same"] > 0
    )
    rank_valid = bool(
        sorted(ranked["rank_overall"].tolist()) == [1, 2, 3, 4]
    )

    smoke_feature_meta = {}
    smoke_families = (
        ["aggregate_ifd"] * 2
        + ["aggregate_flit_count"] * 2
        + ["directional_input_flit_count"] * 5
        + ["directional_output_flit_count"] * 5
        + ["directional_input_ifd"] * 5
        + ["directional_output_ifd"] * 5
    )
    for feature_index in range(24):
        smoke_feature_meta[feature_index] = {
            "feature_index": feature_index,
            "feature_name": f"feature_{feature_index}",
            "feature_family": smoke_families[feature_index],
            "measurement_type": "synthetic",
            "flow_side": "",
            "direction": "",
        }
    smoke_tensor = np.random.default_rng(7).random(
        (9, 16, 8, 24), dtype=np.float32
    )
    smoke_block = block_vector_rows(
        smoke_tensor, "synthetic_target", 0, smoke_feature_meta
    )
    block_dimensions_valid = bool(
        smoke_block["dimension_id"].nunique() == sum(EXPECTED_DIMENSIONS.values())
        and all(
            smoke_block[
                smoke_block["representation"].eq(representation)
            ]["dimension_id"].nunique() == expected
            for representation, expected in EXPECTED_DIMENSIONS.items()
        )
    )

    checks = {
        "vector alignment smoke test": training_matrix.shape == (4, 4),
        "distance computation smoke test": distances_nonnegative and constant_safe,
        "nearest-class smoke test": nearest_is_normal,
        "class-margin smoke test": class_margin_valid and rank_valid,
        "combined equal-weight smoke test": equal_weight_valid,
        "block-vector dimension smoke test": block_dimensions_valid,
    }
    success = all(checks.values())
    if verbose:
        for name, value in checks.items():
            print(f"{name}: {value}")
        print(f"smoke-test-only success: {success}")
    return success


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 7C2: nearest training-run, class-margin, dimension-contribution "
            "and dominant-run block-stability analysis using frozen Stage 7C1 V2 "
            "training references."
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test-only", action="store_true")
    args = parser.parse_args()

    if args.smoke_test_only:
        if not synthetic_smoke_test(verbose=True):
            raise SystemExit(1)
        return

    if not synthetic_smoke_test(verbose=True):
        raise SystemExit("Built-in Stage 7C2 smoke test failed.")

    dataset_root = args.dataset_root.expanduser().resolve()
    failure_root = args.failure_root.resolve()
    table_dir = failure_root / "tables"
    log_dir = failure_root / "logs"
    plot_dir = failure_root / "plots" / "stage7c2_nearest_runs"

    inputs = {
        "training_run_reference": table_dir / "stage7c1_v2_training_run_reference.csv",
        "training_dimension_reference": table_dir / "stage7c1_v2_training_dimension_reference.csv",
        "target_global_coverage": table_dir / "stage7c1_v2_target_global_coverage.csv",
        "target_router_coverage": table_dir / "stage7c1_v2_target_router_feature_coverage.csv",
        "target_temporal_coverage": table_dir / "stage7c1_v2_target_temporal_feature_coverage.csv",
        "target_spatial_coverage": table_dir / "stage7c1_v2_target_spatial_coverage.csv",
        "stage7c1_summary": table_dir / "stage7c1_v2_summary.json",
        "run_inventory": table_dir / "stage7_run_inventory.csv",
        "selected_inventory": table_dir / "stage7_selected_run_inventory.csv",
        "taxonomy": table_dir / "stage7b_corrected_feature_taxonomy.csv",
        "alignment": table_dir / "stage7_sample_alignment.csv",
        "x": dataset_root / "x.npy",
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    outputs = {
        "training_vector_manifest": table_dir / "stage7c2_training_vector_manifest.csv",
        "target_vector_manifest": table_dir / "stage7c2_target_vector_manifest.csv",
        "distance_results": table_dir / "stage7c2_distance_results.csv",
        "nearest_overall": table_dir / "stage7c2_nearest_runs_overall.csv",
        "nearest_same": table_dir / "stage7c2_nearest_same_class.csv",
        "nearest_opposite": table_dir / "stage7c2_nearest_opposite_class.csv",
        "class_margin": table_dir / "stage7c2_class_margin_summary.csv",
        "topk": table_dir / "stage7c2_topk_class_summary.csv",
        "contributions": table_dir / "stage7c2_dimension_contributions.csv",
        "block_stability": table_dir / "stage7c2_dominant_run_block_stability.csv",
        "missing_evidence": table_dir / "stage7c2_missing_scenario_evidence.csv",
        "validation_summary": table_dir / "stage7c2_validation_summary.csv",
        "summary_json": table_dir / "stage7c2_summary.json",
        "report": failure_root / "STAGE7C2_NEAREST_TRAINING_RUN_REPORT.md",
        "validation_log": log_dir / "42_stage7c2_validation.txt",
    }

    existing = [path for path in outputs.values() if path.exists()]
    if plot_dir.exists():
        existing.append(plot_dir)
    if existing and not args.overwrite:
        raise FileExistsError(
            "Stage 7C2 outputs already exist. Use --overwrite only for an intentional rerun:\n"
            + "\n".join(str(path) for path in existing)
        )

    staging_root = failure_root / f".stage7c2_staging_{os.getpid()}"
    staged_table_dir = staging_root / "tables"
    staged_log_dir = staging_root / "logs"
    staged_plot_dir = staging_root / "plots" / "stage7c2_nearest_runs"
    staged_report = staging_root / "STAGE7C2_NEAREST_TRAINING_RUN_REPORT.md"
    remove_tree(staging_root)
    staged_table_dir.mkdir(parents=True, exist_ok=True)
    staged_log_dir.mkdir(parents=True, exist_ok=True)
    staged_plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"script version: {SCRIPT_VERSION}", flush=True)
    print(f"dataset root: {dataset_root}", flush=True)
    print(f"failure root: {failure_root}", flush=True)
    print("scaling reference: frozen all_training Stage 7C1 V2", flush=True)
    print("R6 included in primary geometry: False", flush=True)
    print("target-derived feature selection: False", flush=True)
    print("classifier fitted: False", flush=True)
    print("inference rerun: False", flush=True)
    print("model training performed: False", flush=True)

    stage7c1_summary = json.loads(inputs["stage7c1_summary"].read_text())
    if not stage7c1_summary.get("overall_success", False):
        raise RuntimeError("Stage 7C1 V2 summary does not report success.")
    if stage7c1_summary.get("nearest_run_ranking_performed", True):
        raise RuntimeError("Stage 7C1 V2 unexpectedly reports prior nearest-run ranking.")

    run_inventory = read_csv(inputs["run_inventory"], ["run_id", "split"])
    selected_inventory = read_csv(inputs["selected_inventory"], ["run_id", "split"])
    taxonomy = read_csv(inputs["taxonomy"], ["feature_name", "feature_family", "measurement_type", "flow_side", "direction"])
    alignment = read_csv(inputs["alignment"], ["run_id", "split", "sample_id"])
    metadata = run_metadata_lookup(run_inventory)
    feature_meta = feature_lookup(taxonomy)

    training_reference_raw = read_csv(
        inputs["training_run_reference"],
        [
            "run_id", "split", "profile", "active_cores", "attackers",
            "strength_group", "representation", "dimension_id", "metric",
            "feature_name", "feature_family", "measurement_type", "flow_side",
            "direction", "provenance_table",
        ],
    )
    dimension_reference_raw = read_csv(
        inputs["training_dimension_reference"],
        [
            "reference_cohort", "representation", "dimension_id", "metric",
            "feature_name", "feature_family", "measurement_type", "flow_side",
            "direction", "provenance_table",
        ],
    )
    training_dimensions = canonical_training_dimensions(training_reference_raw)
    scaling_references = canonical_scaling_references(dimension_reference_raw)
    target_dimensions = canonical_target_dimensions(
        [
            inputs["target_global_coverage"],
            inputs["target_router_coverage"],
            inputs["target_temporal_coverage"],
            inputs["target_spatial_coverage"],
        ]
    )

    vector_dimensions_valid, dimension_map = validate_vector_dimensions(
        training_dimensions, target_dimensions, scaling_references
    )
    if not vector_dimensions_valid:
        raise RuntimeError("R1–R5 training/target/reference dimension alignment failed.")

    training_run_ids = sorted(training_dimensions["run_id"].unique())
    target_run_ids = sorted(target_dimensions["run_id"].unique())
    selected_run_ids = sorted(selected_inventory["run_id"].astype(str).unique())
    if training_run_ids != sorted(
        run_inventory[run_inventory["split"].eq("train")]["run_id"].astype(str).unique()
    ):
        raise RuntimeError("Training vector run IDs do not match the Stage 7 run inventory.")
    if set(target_run_ids) != set(selected_run_ids):
        raise RuntimeError("Target vector run IDs do not match Stage 7 selected runs.")

    training_manifest = vector_manifest(training_dimensions, metadata, "training_neighbor")
    target_manifest = vector_manifest(target_dimensions, metadata, "selected_target")

    matrices_by_representation: dict[str, dict[str, Any]] = {}
    distance_parts: list[pd.DataFrame] = []
    for number, representation in enumerate(REPRESENTATIONS, start=1):
        print(f"[{number}/5] calculating distances for {representation}", flush=True)
        matrices = matrices_for_representation(
            training_dimensions, target_dimensions, scaling_references, representation
        )
        matrices_by_representation[representation] = matrices
        distance_parts.append(calculate_representation_distances(matrices, metadata))

    distance_results = add_combined_distances(pd.concat(distance_parts, ignore_index=True))
    ranked = rank_distances(distance_results)
    nearest_overall, nearest_same, nearest_opposite = nearest_tables(ranked)
    class_margins, topk_summary = class_margin_tables(ranked)

    print("calculating dominant-run dimension contributions", flush=True)
    contributions = build_dimension_contributions(matrices_by_representation, ranked)

    print("calculating eight-block dominant-run stability", flush=True)
    x = np.load(inputs["x"], mmap_mode="r", allow_pickle=False)
    x_is_memmap = isinstance(x, np.memmap)
    if tuple(x.shape) != (233803, 16, 8, 24):
        raise RuntimeError(f"Unexpected x shape: {x.shape}")
    block_stability, block_boundaries_deterministic = block_distances_and_margins(
        x, alignment, matrices_by_representation, metadata, feature_meta
    )

    missing_evidence = build_missing_scenario_evidence(
        class_margins, contributions, metadata
    )

    make_plots(
        class_margins,
        ranked,
        topk_summary,
        contributions,
        block_stability,
        staged_plot_dir,
    )
    plot_count = len(list(staged_plot_dir.glob("*.png")))

    only_training_neighbors = set(ranked["training_split"].astype(str)) == {"train"}
    all_45_training_runs = ranked["training_run_id"].nunique() == 45
    all_10_targets = ranked["target_run_id"].nunique() == 10
    no_val_test_neighbor = not ranked["training_split"].isin(["val", "test"]).any()
    frozen_scaling_reused = set(ranked["scaling_reference"].astype(str)) == {"all_training"}
    no_target_feature_selection = True
    dimensions_aligned = vector_dimensions_valid
    constant_dimensions_safe = bool(
        np.isfinite(ranked["distance"].to_numpy(dtype=float)).all()
        and (ranked["constant_dimension_violation_count"] >= 0).all()
    )
    no_infinite_distances = bool(np.isfinite(ranked["distance"].to_numpy(dtype=float)).all())
    all_distances_nonnegative = bool((ranked["distance"] >= 0).all())
    self_neighbors_impossible = bool(
        not (ranked["target_run_id"] == ranked["training_run_id"]).any()
    )
    same_opposite_nonempty = bool(
        class_margins["nearest_same_class_run_id"].astype(str).str.len().gt(0).all()
        and class_margins["nearest_opposite_class_run_id"].astype(str).str.len().gt(0).all()
    )
    expected_group_size = 45
    ranks_unique_complete = bool(
        ranked.groupby(["target_run_id", "representation", "distance_metric"])["rank_overall"]
        .agg(lambda values: sorted(values.astype(int)) == list(range(1, expected_group_size + 1)))
        .all()
    )
    nearest_counts_valid = (
        len(nearest_overall) == 10 * 6 * 3 * TOP_OVERALL
        and len(nearest_same) == 10 * 6 * 3 * TOP_CLASS
        and len(nearest_opposite) == 10 * 6 * 3 * TOP_CLASS
    )
    combined_equal_weights = bool(
        distance_results[
            distance_results["representation"].eq(COMBINED_REPRESENTATION)
        ]["combined_weight_rule"]
        .eq("equal_weight_mean_of_R1_R5_distances")
        .all()
    )
    block_rows_complete = len(block_stability) == 2 * BLOCK_COUNT * 6 * 3
    block_finite = bool(
        np.isfinite(
            block_stability[
                [
                    "nearest_same_class_distance",
                    "nearest_opposite_class_distance",
                    "class_margin_opposite_minus_same",
                ]
            ].to_numpy(dtype=float)
        ).all()
    )
    r6_excluded = not ranked["representation"].astype(str).str.startswith("R6").any()
    x_memory_mapped = x_is_memmap

    validation = [
        ("synthetic smoke test passed", True),
        ("only training runs used as neighbours", only_training_neighbors),
        ("all 45 training runs represented", all_45_training_runs),
        ("all 10 selected targets evaluated", all_10_targets),
        ("no validation/test run appears as a neighbour", no_val_test_neighbor),
        ("training-derived all-training scaling reused unchanged", frozen_scaling_reused),
        ("no target-derived feature selection performed", no_target_feature_selection),
        ("all expected R1-R5 dimensions aligned", dimensions_aligned),
        ("constant dimensions handled safely", constant_dimensions_safe),
        ("no infinite distances", no_infinite_distances),
        ("all distances non-negative", all_distances_nonnegative),
        ("self-neighbour matches impossible", self_neighbors_impossible),
        ("same/opposite class sets non-empty", same_opposite_nonempty),
        ("top-k ranks unique and complete", ranks_unique_complete),
        ("nearest-run table counts complete", nearest_counts_valid),
        ("combined representation uses equal fixed weights", combined_equal_weights),
        ("block boundaries deterministic", block_boundaries_deterministic),
        ("block stability rows complete", block_rows_complete),
        ("block distances finite", block_finite),
        ("original feature tensor memory mapped for block analysis", x_memory_mapped),
        ("R6 excluded from primary geometry", r6_excluded),
        ("classifier fitted", False),
        ("inference rerun", False),
        ("model training performed", False),
    ]
    overall_success = all(
        value for name, value in validation
        if name not in {"classifier fitted", "inference rerun", "model training performed"}
    ) and all(
        not value for name, value in validation
        if name in {"classifier fitted", "inference rerun", "model training performed"}
    )
    validation.append(("overall success", overall_success))
    validation_frame = validation_summary_frame(validation)
    validation_text = "\n".join(
        f"{name}: {value}" for name, value in validation
    ) + "\n"
    print()
    print(validation_text, end="", flush=True)

    if not overall_success:
        remove_tree(staging_root)
        raise SystemExit("Stage 7C2 validation failed; no final outputs were promoted.")

    report = build_report(
        class_margins,
        ranked,
        block_stability,
        missing_evidence,
        plot_count,
    )
    summary = {
        "script_version": SCRIPT_VERSION,
        "dataset_root": str(dataset_root),
        "failure_root": str(failure_root),
        "training_run_count": len(training_run_ids),
        "selected_target_run_count": len(target_run_ids),
        "representations": REPRESENTATIONS,
        "combined_representation": COMBINED_REPRESENTATION,
        "distance_metrics": DISTANCE_METRICS,
        "class_margin_sign_convention": "nearest_opposite_minus_nearest_same",
        "positive_margin_interpretation": "same_class_closer",
        "negative_margin_interpretation": "opposite_class_closer",
        "constant_violation_penalty": CONSTANT_VIOLATION_PENALTY,
        "block_count": BLOCK_COUNT,
        "distance_result_rows": len(ranked),
        "nearest_overall_rows": len(nearest_overall),
        "nearest_same_class_rows": len(nearest_same),
        "nearest_opposite_class_rows": len(nearest_opposite),
        "class_margin_rows": len(class_margins),
        "topk_summary_rows": len(topk_summary),
        "dimension_contribution_rows": len(contributions),
        "block_stability_rows": len(block_stability),
        "missing_scenario_evidence_rows": len(missing_evidence),
        "plot_count": plot_count,
        "r6_included_in_primary_geometry": False,
        "scaling_reference": "Stage 7C1 V2 all_training only",
        "classifier_fitted": False,
        "inference_rerun": False,
        "model_training_performed": False,
        "hard_normal_combined_outcome": dominant_outcome(class_margins, HARD_NORMAL_RUN),
        "hard_attack_combined_outcome": dominant_outcome(class_margins, HARD_ATTACK_RUN),
        "overall_success": overall_success,
    }

    staged_paths = {
        "training_vector_manifest": staged_table_dir / outputs["training_vector_manifest"].name,
        "target_vector_manifest": staged_table_dir / outputs["target_vector_manifest"].name,
        "distance_results": staged_table_dir / outputs["distance_results"].name,
        "nearest_overall": staged_table_dir / outputs["nearest_overall"].name,
        "nearest_same": staged_table_dir / outputs["nearest_same"].name,
        "nearest_opposite": staged_table_dir / outputs["nearest_opposite"].name,
        "class_margin": staged_table_dir / outputs["class_margin"].name,
        "topk": staged_table_dir / outputs["topk"].name,
        "contributions": staged_table_dir / outputs["contributions"].name,
        "block_stability": staged_table_dir / outputs["block_stability"].name,
        "missing_evidence": staged_table_dir / outputs["missing_evidence"].name,
        "validation_summary": staged_table_dir / outputs["validation_summary"].name,
        "summary_json": staged_table_dir / outputs["summary_json"].name,
        "validation_log": staged_log_dir / outputs["validation_log"].name,
    }

    atomic_csv(training_manifest, staged_paths["training_vector_manifest"])
    atomic_csv(target_manifest, staged_paths["target_vector_manifest"])
    atomic_csv(ranked, staged_paths["distance_results"])
    atomic_csv(nearest_overall, staged_paths["nearest_overall"])
    atomic_csv(nearest_same, staged_paths["nearest_same"])
    atomic_csv(nearest_opposite, staged_paths["nearest_opposite"])
    atomic_csv(class_margins, staged_paths["class_margin"])
    atomic_csv(topk_summary, staged_paths["topk"])
    atomic_csv(contributions, staged_paths["contributions"])
    atomic_csv(block_stability, staged_paths["block_stability"])
    atomic_csv(missing_evidence, staged_paths["missing_evidence"])
    atomic_csv(validation_frame, staged_paths["validation_summary"])
    atomic_json(summary, staged_paths["summary_json"])
    atomic_text(validation_text, staged_paths["validation_log"])
    atomic_text(report, staged_report)

    for key, final_path in outputs.items():
        if key == "report":
            promote_file(staged_report, final_path)
        elif key == "validation_log":
            promote_file(staged_paths[key], final_path)
        else:
            promote_file(staged_paths[key], final_path)
    promote_directory(staged_plot_dir, plot_dir)
    remove_tree(staging_root)

    print("Stage 7C2 output files:", flush=True)
    for name, path in outputs.items():
        print(f"  {name}: {path}", flush=True)
    print(f"  plots: {plot_dir}", flush=True)
    print("overall success: True", flush=True)


if __name__ == "__main__":
    main()
