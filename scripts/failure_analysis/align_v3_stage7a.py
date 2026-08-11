#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_FILES = {
    "conv1d_gcn": "conv1d_gcn_predictions.npz",
    "tcn_attention_gcn": "tcn_attention_gcn_predictions.npz",
    "tcn_meanpool_gcn": "tcn_meanpool_gcn_predictions.npz",
    "tcn_maxpool_gcn": "tcn_maxpool_gcn_predictions.npz",
}

REQUIRED_DATASET_ARRAYS = {
    "x": "x.npy",
    "y_graph": "y_graph.npy",
    "y_node": "y_node.npy",
    "feature_cols": "feature_cols.npy",
    "run_id": "run_id.npy",
    "end_epoch": "end_epoch.npy",
}

OPTIONAL_DATASET_ARRAYS = {
    "edge_index": "edge_index.npy",
    "sample_id": "sample_id.npy",
    "split": "split.npy",
    "train_idx": "train_idx.npy",
    "val_idx": "val_idx.npy",
    "test_idx": "test_idx.npy",
    "train_indices": "train_indices.npy",
    "val_indices": "val_indices.npy",
    "test_indices": "test_indices.npy",
}

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

SELECTED_GROUPS = {
    "A_hard_normal_exact_test": [
        ("target", HARD_NORMAL_RUN),
        ("control", NORMAL_TEST_CONTROL),
    ],
    "B_hard_normal_validation_analogue": [
        ("target", HARD_NORMAL_RUN),
        ("control", NORMAL_VAL_CONTROL),
    ],
    "C_hard_attack_exact_A12_S20": [
        ("target", HARD_ATTACK_RUN),
        ("control_stream", ATTACK_STREAM_CONTROL),
        ("control_mixed", ATTACK_MIXED_CONTROL),
    ],
    "D_hard_attack_bursty_S20": [
        ("target", HARD_ATTACK_RUN),
        ("control", ATTACK_BURSTY_S20_CONTROL),
    ],
    "E_same_bursty_background_severity": [
        ("S20_target", HARD_ATTACK_RUN),
        ("S63_control", ATTACK_S63_CONTROL),
        ("S77_control", ATTACK_S77_CONTROL),
    ],
    "F_broad_bursty_background": [
        ("normal_control", NORMAL_BURSTY_BACKGROUND),
        ("S20_target", HARD_ATTACK_RUN),
    ],
}

TRUE_NODE_CANDIDATES = [
    "true_node",
    "true_nodes",
    "y_node",
    "node_true",
    "node_labels",
    "true_node_label",
]
NODE_PROBABILITY_CANDIDATES = [
    "node_probability",
    "node_probabilities",
    "node_prob",
    "node_probs",
]
NODE_LOGIT_CANDIDATES = [
    "node_logit",
    "node_logits",
]
NODE_PREDICTION_CANDIDATES = [
    "node_prediction",
    "node_predictions",
    "node_pred",
    "node_preds",
]

DIRECTION_WORDS = ["local", "north", "east", "south", "west"]


# ---------------------------------------------------------------------------
# Atomic writers
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


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Data decoding and metadata parsing
# ---------------------------------------------------------------------------

def decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def decode_string_array(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    return np.asarray([decode_scalar(value) for value in flat], dtype=object)


def canonical_sample_ids(
    run_ids: np.ndarray,
    end_epochs: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            f"{decode_scalar(run_id)}:{int(end_epoch)}"
            for run_id, end_epoch in zip(run_ids, end_epochs)
        ],
        dtype=object,
    )


def parse_run_metadata(run_id: str) -> dict[str, Any]:
    run_id = str(run_id)
    pieces = run_id.split("-")

    is_attack = "-A-" in run_id
    profile_match = re.search(r"-P([^-]+)-R", run_id)
    seed_match = re.search(r"-R(\d+)", run_id)
    strength_match = re.search(r"-S(\d+)", run_id)
    attacker_match = re.search(r"-A-([0-9-]+)-S", run_id)

    profile = profile_match.group(1) if profile_match else ""
    seed = int(seed_match.group(1)) if seed_match else -1
    strength = int(strength_match.group(1)) if strength_match else 0

    prefix = run_id.split("-P", 1)[0]
    active_text = prefix[2:] if prefix.startswith("N-") else ""
    if active_text == "idle":
        active_cores: list[int] = []
    else:
        active_cores = [
            int(token)
            for token in active_text.split("-")
            if token.isdigit()
        ]

    if attacker_match:
        attackers = [
            int(token)
            for token in attacker_match.group(1).split("-")
            if token.isdigit()
        ]
    else:
        attackers = []

    return {
        "run_id": run_id,
        "true_graph_from_name": int(is_attack),
        "profile_from_name": profile,
        "seed_from_name": seed,
        "strength_from_name": strength,
        "active_cores_from_name": "-".join(map(str, active_cores)),
        "active_core_count_from_name": len(active_cores),
        "attackers_from_name": "-".join(map(str, attackers)),
        "attacker_count_from_name": len(attackers),
    }


def normalize_split_value(value: Any) -> str:
    text = decode_scalar(value).strip().lower()
    mapping = {
        "train": "train",
        "training": "train",
        "0": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "1": "val",
        "test": "test",
        "testing": "test",
        "2": "test",
    }
    if text not in mapping:
        raise ValueError(f"Unrecognized split value: {value!r}")
    return mapping[text]


def first_present(names: list[str], available: set[str]) -> str:
    for name in names:
        if name in available:
            return name
    return ""


# ---------------------------------------------------------------------------
# File and array manifest
# ---------------------------------------------------------------------------

def file_record(path: Path, role: str, required: bool) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "role": role,
        "required": required,
        "filename": path.name,
        "absolute_path": str(path.resolve()),
        "exists": exists,
        "size_bytes": int(path.stat().st_size) if exists else 0,
    }


def npy_array_record(
    path: Path,
    role: str,
    required: bool,
    mmap: bool,
    allow_pickle: bool,
) -> tuple[dict[str, Any], np.ndarray | None]:
    record = file_record(path, role, required)
    if not record["exists"]:
        record.update(
            {
                "array_loaded": False,
                "mmap_requested": mmap,
                "is_memmap": False,
                "shape": "",
                "ndim": -1,
                "dtype": "",
            }
        )
        return record, None

    array = np.load(
        path,
        mmap_mode="r" if mmap else None,
        allow_pickle=allow_pickle,
    )
    record.update(
        {
            "array_loaded": True,
            "mmap_requested": mmap,
            "is_memmap": isinstance(array, np.memmap),
            "shape": "x".join(str(part) for part in array.shape),
            "ndim": int(array.ndim),
            "dtype": str(array.dtype),
        }
    )
    return record, array


def prediction_manifest(
    prediction_directory: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    loaded: dict[str, dict[str, np.ndarray]] = {}

    for model_name, filename in MODEL_FILES.items():
        path = prediction_directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)

        model_arrays: dict[str, np.ndarray] = {}
        with np.load(path, allow_pickle=False) as source:
            for key in source.files:
                array = source[key]
                model_arrays[key] = array
                rows.append(
                    {
                        "model_name": model_name,
                        "prediction_file": filename,
                        "absolute_path": str(path.resolve()),
                        "array_key": key,
                        "shape": "x".join(str(part) for part in array.shape),
                        "ndim": int(array.ndim),
                        "dtype": str(array.dtype),
                        "is_node_related": "node" in key.lower(),
                        "is_graph_related": "graph" in key.lower(),
                    }
                )

        loaded[model_name] = model_arrays

    return pd.DataFrame(rows), loaded


# ---------------------------------------------------------------------------
# Split reconstruction
# ---------------------------------------------------------------------------

def indices_from_optional_files(
    dataset_root: Path,
    n_samples: int,
) -> tuple[np.ndarray | None, str]:
    direct_split_path = dataset_root / OPTIONAL_DATASET_ARRAYS["split"]
    if direct_split_path.is_file():
        direct = np.load(direct_split_path, allow_pickle=True)
        if direct.shape != (n_samples,):
            raise RuntimeError(
                f"split.npy has shape {direct.shape}; expected {(n_samples,)}."
            )
        normalized = np.asarray(
            [normalize_split_value(value) for value in direct],
            dtype=object,
        )
        return normalized, "dataset_split_array"

    candidate_sets = [
        ("train_idx.npy", "val_idx.npy", "test_idx.npy"),
        ("train_indices.npy", "val_indices.npy", "test_indices.npy"),
    ]

    for train_name, val_name, test_name in candidate_sets:
        paths = [
            dataset_root / train_name,
            dataset_root / val_name,
            dataset_root / test_name,
        ]
        if all(path.is_file() for path in paths):
            split = np.full(n_samples, "", dtype=object)
            for name, path in [
                ("train", paths[0]),
                ("val", paths[1]),
                ("test", paths[2]),
            ]:
                indices = np.asarray(
                    np.load(path, allow_pickle=False),
                    dtype=np.int64,
                ).reshape(-1)
                if np.any(indices < 0) or np.any(indices >= n_samples):
                    raise RuntimeError(f"Out-of-range indices in {path}.")
                if np.any(split[indices] != ""):
                    raise RuntimeError(
                        f"Split index overlap while applying {path}."
                    )
                split[indices] = name

            if np.any(split == ""):
                raise RuntimeError(
                    "Dataset split index files do not cover every sample."
                )
            return split, f"dataset_index_files:{train_name},{val_name},{test_name}"

    return None, ""


def reconstruct_split(
    n_samples: int,
    stage3_metadata: pd.DataFrame,
    dataset_root: Path,
) -> tuple[np.ndarray, str, bool]:
    dataset_split, dataset_method = indices_from_optional_files(
        dataset_root,
        n_samples,
    )

    stage3_split = np.full(n_samples, "train", dtype=object)
    export_indices = stage3_metadata["sample_index"].to_numpy(dtype=np.int64)
    export_splits = stage3_metadata["split"].map(
        normalize_split_value
    ).to_numpy(dtype=object)

    if set(np.unique(export_splits)) != {"val", "test"}:
        raise RuntimeError(
            "Stage 3 metadata must contain exactly validation and test exports."
        )

    stage3_split[export_indices] = export_splits

    if dataset_split is None:
        return stage3_split, "stage3_val_test_plus_training_complement", True

    matches = bool(np.array_equal(dataset_split, stage3_split))
    return dataset_split, dataset_method, matches


# ---------------------------------------------------------------------------
# Feature map and topology
# ---------------------------------------------------------------------------

def infer_direction(feature_name: str) -> str:
    lower = feature_name.lower()
    for direction in DIRECTION_WORDS:
        if re.search(rf"(^|[_\-]){direction}($|[_\-])", lower):
            return direction
    return ""


def infer_flow_side(feature_name: str) -> str:
    lower = feature_name.lower()
    if re.search(r"(^|[_\-])(in|input|ingress)($|[_\-])", lower):
        return "input"
    if re.search(r"(^|[_\-])(out|output|egress)($|[_\-])", lower):
        return "output"
    return ""


def infer_feature_family(feature_name: str) -> str:
    lower = feature_name.lower()
    if "ifd" in lower or "inter_flit" in lower or "interflit" in lower:
        return "ifd"
    if "flit" in lower and ("count" in lower or "cnt" in lower):
        return "flit_count"
    if "flit" in lower:
        return "flit_other"
    return "other"


def build_feature_map(feature_cols: np.ndarray) -> pd.DataFrame:
    names = decode_string_array(feature_cols)
    rows = []

    for index, name in enumerate(names):
        rows.append(
            {
                "feature_index": index,
                "feature_name": name,
                "feature_family_inferred": infer_feature_family(name),
                "flow_side_inferred": infer_flow_side(name),
                "direction_inferred": infer_direction(name),
                "is_directional_inferred": bool(infer_direction(name)),
                "inference_is_provisional": True,
            }
        )

    return pd.DataFrame(rows)


def normalize_edge_index(edge_index: np.ndarray) -> np.ndarray:
    array = np.asarray(edge_index)
    if array.ndim != 2:
        raise RuntimeError(f"edge_index must be 2D; found {array.shape}.")

    if array.shape[0] == 2:
        edges = array.T
    elif array.shape[1] == 2:
        edges = array
    else:
        raise RuntimeError(
            f"edge_index must have one dimension of size 2; found {array.shape}."
        )

    return np.asarray(edges, dtype=np.int64)


def expected_row_major_mesh_edges(rows: int, columns: int) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for row in range(rows):
        for column in range(columns):
            node = row * columns + column
            if row + 1 < rows:
                neighbor = (row + 1) * columns + column
                edges.add(tuple(sorted((node, neighbor))))
            if column + 1 < columns:
                neighbor = row * columns + column + 1
                edges.add(tuple(sorted((node, neighbor))))
    return edges


def build_topology_manifest(
    edge_index: np.ndarray | None,
    expected_nodes: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if edge_index is None:
        status = {
            "edge_index_available": False,
            "router_count": expected_nodes,
            "self_loop_count": 0,
            "directed_edge_count": 0,
            "undirected_edge_count": 0,
            "matches_row_major_4x4_mesh": False,
            "row_major_numbering_confirmed": False,
            "router_12_neighbors": "",
        }
        return pd.DataFrame(), status

    edges = normalize_edge_index(edge_index)
    directed_pairs = [(int(source), int(target)) for source, target in edges]
    self_loops = [(source, target) for source, target in directed_pairs if source == target]
    undirected = {
        tuple(sorted((source, target)))
        for source, target in directed_pairs
        if source != target
    }

    neighbors: dict[int, set[int]] = {
        node: set() for node in range(expected_nodes)
    }
    for source, target in undirected:
        neighbors[source].add(target)
        neighbors[target].add(source)

    expected_edges = expected_row_major_mesh_edges(4, 4)
    matches_row_major = expected_nodes == 16 and undirected == expected_edges

    rows = []
    for node in range(expected_nodes):
        rows.append(
            {
                "router_id": node,
                "neighbor_count": len(neighbors[node]),
                "neighbors": "-".join(map(str, sorted(neighbors[node]))),
                "row_major_row_if_confirmed": node // 4 if matches_row_major else -1,
                "row_major_column_if_confirmed": node % 4 if matches_row_major else -1,
            }
        )

    status = {
        "edge_index_available": True,
        "router_count": expected_nodes,
        "self_loop_count": len(self_loops),
        "directed_edge_count": len(directed_pairs),
        "undirected_edge_count": len(undirected),
        "matches_row_major_4x4_mesh": matches_row_major,
        "row_major_numbering_confirmed": matches_row_major,
        "router_12_neighbors": "-".join(
            map(str, sorted(neighbors.get(12, set())))
        ),
    }
    return pd.DataFrame(rows), status


# ---------------------------------------------------------------------------
# Node export detection
# ---------------------------------------------------------------------------

def inspect_node_exports(
    loaded_predictions: dict[str, dict[str, np.ndarray]],
    n_exported: int,
    expected_nodes: int,
) -> pd.DataFrame:
    rows = []

    for model_name, arrays in loaded_predictions.items():
        keys = set(arrays)
        true_key = first_present(TRUE_NODE_CANDIDATES, keys)
        probability_key = first_present(NODE_PROBABILITY_CANDIDATES, keys)
        logit_key = first_present(NODE_LOGIT_CANDIDATES, keys)
        prediction_key = first_present(NODE_PREDICTION_CANDIDATES, keys)

        def shape_text(key: str) -> str:
            if not key:
                return ""
            return "x".join(str(part) for part in arrays[key].shape)

        aligned_shapes = True
        for key in [true_key, probability_key, logit_key, prediction_key]:
            if not key:
                continue
            shape = arrays[key].shape
            if len(shape) < 2 or shape[0] != n_exported or shape[1] != expected_nodes:
                aligned_shapes = False

        rows.append(
            {
                "model_name": model_name,
                "true_node_key": true_key,
                "true_node_shape": shape_text(true_key),
                "node_probability_key": probability_key,
                "node_probability_shape": shape_text(probability_key),
                "node_logit_key": logit_key,
                "node_logit_shape": shape_text(logit_key),
                "node_prediction_key": prediction_key,
                "node_prediction_shape": shape_text(prediction_key),
                "node_labels_exported": bool(true_key),
                "node_probabilities_exported": bool(probability_key),
                "node_logits_exported": bool(logit_key),
                "node_predictions_exported": bool(prediction_key),
                "all_present_node_shapes_aligned": aligned_shapes,
                "stage7e_limited_node_grouping_possible": bool(
                    aligned_shapes
                    and (probability_key or prediction_key or logit_key)
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run, sample and selected-group manifests
# ---------------------------------------------------------------------------

def chunked_node_counts(y_node: np.ndarray, chunk_size: int = 8192) -> np.ndarray:
    n_samples = y_node.shape[0]
    counts = np.empty(n_samples, dtype=np.int16)
    for start in range(0, n_samples, chunk_size):
        finish = min(n_samples, start + chunk_size)
        chunk = np.asarray(y_node[start:finish])
        counts[start:finish] = np.sum(chunk > 0, axis=1).astype(np.int16)
    return counts


def add_temporal_positions(sample_alignment: pd.DataFrame) -> pd.DataFrame:
    frame = sample_alignment.sort_values(
        ["run_id", "end_epoch", "dataset_index"]
    ).copy()

    grouped = frame.groupby("run_id", sort=False)
    frame["run_window_index"] = grouped.cumcount()
    frame["windows_in_run"] = grouped["dataset_index"].transform("count")
    denominator = np.maximum(frame["windows_in_run"].to_numpy() - 1, 1)
    frame["run_window_fraction"] = (
        frame["run_window_index"].to_numpy() / denominator
    )

    fractions = frame["run_window_fraction"].to_numpy()
    frame["run_phase"] = np.where(
        fractions < 0.10,
        "early_10pct",
        np.where(fractions >= 0.90, "late_10pct", "middle_80pct"),
    )
    frame["temporal_decile"] = np.minimum(
        (fractions * 10).astype(int),
        9,
    )

    return frame.sort_values("dataset_index").reset_index(drop=True)


def build_run_inventory(
    alignment: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for run_id, group in alignment.groupby("run_id", sort=True):
        parsed = parse_run_metadata(run_id)
        split_values = group["split"].unique()
        label_values = group["true_graph"].unique()
        node_count_values = group["true_attacker_count"].unique()

        rows.append(
            {
                **parsed,
                "split": split_values[0] if len(split_values) == 1 else "MIXED",
                "true_graph": int(label_values[0]) if len(label_values) == 1 else -1,
                "number_of_windows": int(len(group)),
                "minimum_dataset_index": int(group["dataset_index"].min()),
                "maximum_dataset_index": int(group["dataset_index"].max()),
                "minimum_end_epoch": int(group["end_epoch"].min()),
                "maximum_end_epoch": int(group["end_epoch"].max()),
                "unique_end_epochs": int(group["end_epoch"].nunique()),
                "minimum_true_attacker_count": int(node_count_values.min()),
                "maximum_true_attacker_count": int(node_count_values.max()),
                "split_constant_within_run": len(split_values) == 1,
                "graph_label_constant_within_run": len(label_values) == 1,
                "attacker_count_constant_within_run": len(node_count_values) == 1,
            }
        )

    return pd.DataFrame(rows).sort_values(["split", "run_id"])


def build_selected_run_inventory(
    run_inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    selected_indices: dict[str, np.ndarray] = {}

    available = set(run_inventory["run_id"])
    required = {
        run_id
        for members in SELECTED_GROUPS.values()
        for _, run_id in members
    }
    missing = required - available
    if missing:
        raise RuntimeError(
            f"Selected Stage 7 run IDs are missing: {sorted(missing)}"
        )

    for group_name, members in SELECTED_GROUPS.items():
        for role, run_id in members:
            record = run_inventory[
                run_inventory["run_id"].eq(run_id)
            ].iloc[0]
            rows.append(
                {
                    "group_name": group_name,
                    "group_role": role,
                    **record.to_dict(),
                }
            )

    return pd.DataFrame(rows), selected_indices


def selected_index_arrays(
    alignment: pd.DataFrame,
    selected_inventory: pd.DataFrame,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    for run_id in selected_inventory["run_id"].unique():
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", run_id).strip("_")
        arrays[f"run__{safe_name}"] = alignment.loc[
            alignment["run_id"].eq(run_id),
            "dataset_index",
        ].to_numpy(dtype=np.int64)

    for group_name, group in selected_inventory.groupby("group_name"):
        run_ids = set(group["run_id"])
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", group_name).strip("_")
        arrays[f"group__{safe_name}"] = alignment.loc[
            alignment["run_id"].isin(run_ids),
            "dataset_index",
        ].to_numpy(dtype=np.int64)

    return arrays


def build_split_summary(
    alignment: pd.DataFrame,
    run_inventory: pd.DataFrame,
) -> pd.DataFrame:
    sample_counts = alignment.groupby("split").size()
    run_counts = run_inventory.groupby("split").size()

    rows = []
    for split_name in ["train", "val", "test"]:
        rows.append(
            {
                "split": split_name,
                "sample_count": int(sample_counts.get(split_name, 0)),
                "run_count": int(run_counts.get(split_name, 0)),
                "minimum_windows_per_run": int(
                    run_inventory.loc[
                        run_inventory["split"].eq(split_name),
                        "number_of_windows",
                    ].min()
                ),
                "maximum_windows_per_run": int(
                    run_inventory.loc[
                        run_inventory["split"].eq(split_name),
                        "number_of_windows",
                    ].max()
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prediction and Stage 6 alignment
# ---------------------------------------------------------------------------

def verify_prediction_exports(
    stage3_metadata: pd.DataFrame,
    loaded_predictions: dict[str, dict[str, np.ndarray]],
    alignment: pd.DataFrame,
    y_node: np.ndarray,
) -> tuple[bool, bool, list[str]]:
    messages: list[str] = []
    all_models_aligned = True
    node_labels_matched_when_present = True

    expected_index = stage3_metadata["sample_index"].to_numpy(dtype=np.int64)
    expected_sample_id = stage3_metadata["sample_id"].astype(str).to_numpy()
    expected_split = stage3_metadata["split"].map(
        normalize_split_value
    ).to_numpy(dtype=object)
    expected_true_graph = stage3_metadata["true_graph"].to_numpy(dtype=np.int64)

    dataset_rows = alignment.set_index("dataset_index").loc[expected_index]

    if not np.array_equal(
        dataset_rows["sample_id"].astype(str).to_numpy(),
        expected_sample_id,
    ):
        all_models_aligned = False
        messages.append("Stage 3 metadata sample_id does not match dataset.")

    if not np.array_equal(
        dataset_rows["split"].astype(str).to_numpy(),
        expected_split.astype(str),
    ):
        all_models_aligned = False
        messages.append("Stage 3 metadata split does not match reconstructed split.")

    if not np.array_equal(
        dataset_rows["true_graph"].to_numpy(dtype=np.int64),
        expected_true_graph,
    ):
        all_models_aligned = False
        messages.append("Stage 3 graph labels do not match dataset y_graph.")

    for model_name, arrays in loaded_predictions.items():
        required = {
            "sample_index",
            "sample_id",
            "split",
            "true_graph",
            "graph_probability",
        }
        missing = required - set(arrays)
        if missing:
            all_models_aligned = False
            messages.append(
                f"{model_name} missing prediction arrays: {sorted(missing)}"
            )
            continue

        checks = {
            "sample_index": np.array_equal(
                np.asarray(arrays["sample_index"], dtype=np.int64),
                expected_index,
            ),
            "sample_id": np.array_equal(
                np.asarray(arrays["sample_id"]).astype(str),
                expected_sample_id,
            ),
            "split": np.array_equal(
                np.asarray(arrays["split"]).astype(str),
                stage3_metadata["split"].astype(str).to_numpy(),
            ),
            "true_graph": np.array_equal(
                np.asarray(arrays["true_graph"], dtype=np.int64),
                expected_true_graph,
            ),
        }
        for key, passed in checks.items():
            if not passed:
                all_models_aligned = False
                messages.append(f"{model_name} failed {key} alignment.")

        true_node_key = first_present(
            TRUE_NODE_CANDIDATES,
            set(arrays),
        )
        if true_node_key:
            exported_true_node = np.asarray(arrays[true_node_key])
            dataset_true_node = np.asarray(y_node[expected_index])
            if not np.array_equal(
                exported_true_node.astype(np.int64),
                dataset_true_node.astype(np.int64),
            ):
                node_labels_matched_when_present = False
                messages.append(
                    f"{model_name} exported node labels do not match y_node."
                )

    return all_models_aligned, node_labels_matched_when_present, messages


def verify_stage6_alignment(
    stage6_agreement_path: Path,
    stage3_metadata: pd.DataFrame,
) -> tuple[bool, int, str]:
    if not stage6_agreement_path.is_file():
        return False, 0, "Stage 6 agreement table not found."

    agreement = pd.read_csv(
        stage6_agreement_path,
        keep_default_na=False,
        low_memory=False,
        dtype={"sample_id": str, "run_id": str},
    )

    required = {"sample_id", "run_id", "operating_point"}
    missing = required - set(agreement.columns)
    if missing:
        return False, len(agreement), f"Missing columns: {sorted(missing)}"

    available_ids = set(stage3_metadata["sample_id"].astype(str))
    stage6_ids = set(agreement["sample_id"].astype(str))
    aligned = stage6_ids.issubset(available_ids)

    expected_points = {"reference_0_50", "validation_selected"}
    actual_points = set(agreement["operating_point"].astype(str))
    aligned = aligned and actual_points == expected_points

    return aligned, len(agreement), "" if aligned else "Stage 6 IDs or operating points differ."


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def build_alignment_report(
    dataset_root: Path,
    failure_root: Path,
    x: np.ndarray,
    feature_map: pd.DataFrame,
    split_summary: pd.DataFrame,
    run_inventory: pd.DataFrame,
    selected_inventory: pd.DataFrame,
    node_status: pd.DataFrame,
    topology_status: dict[str, Any],
    split_source: str,
    split_crosscheck: bool,
    stage6_aligned: bool,
) -> str:
    lines = [
        "# Stage 7A — Dataset and Prediction Alignment",
        "",
        "Stage 7A establishes a leakage-free index map between the original V3 "
        "feature tensor, graph/node labels, all 71 runs, the train/validation/test "
        "split, Stage 3 prediction exports and Stage 6 diagnostic groups.",
        "",
        "No feature statistics, classifier fitting, inference or retraining were performed.",
        "",
        "## Paths",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- Failure-analysis root: `{failure_root}`",
        "",
        "## Tensor",
        "",
        f"- `x.npy` shape: `{tuple(x.shape)}`",
        f"- `x.npy` dtype: `{x.dtype}`",
        f"- Memory mapped: `{isinstance(x, np.memmap)}`",
        f"- Feature count: `{len(feature_map)}`",
        "",
        "## Split reconstruction",
        "",
        f"- Split source: `{split_source}`",
        f"- Dataset split cross-check passed: `{split_crosscheck}`",
        "",
        "| Split | Samples | Runs | Minimum windows/run | Maximum windows/run |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in split_summary.iterrows():
        lines.append(
            f"| {row['split']} | {int(row['sample_count'])} | "
            f"{int(row['run_count'])} | {int(row['minimum_windows_per_run'])} | "
            f"{int(row['maximum_windows_per_run'])} |"
        )

    lines.extend(
        [
            "",
            "## Selected Stage 7 groups",
            "",
            "| Group | Role | Run | Split | Class | Profile | Active cores | "
            "Attackers | Strength |",
            "|---|---|---|---|---:|---|---|---|---:|",
        ]
    )

    for _, row in selected_inventory.iterrows():
        lines.append(
            f"| {row['group_name']} | {row['group_role']} | "
            f"`{row['run_id']}` | {row['split']} | {int(row['true_graph'])} | "
            f"{row['profile_from_name']} | {row['active_cores_from_name']} | "
            f"{row['attackers_from_name']} | {int(row['strength_from_name'])} |"
        )

    lines.extend(
        [
            "",
            "## Node export status",
            "",
            "| Model | Labels | Probabilities | Logits | Predictions | "
            "Limited Stage 7E grouping possible |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in node_status.iterrows():
        lines.append(
            f"| {row['model_name']} | {row['node_labels_exported']} | "
            f"{row['node_probabilities_exported']} | "
            f"{row['node_logits_exported']} | "
            f"{row['node_predictions_exported']} | "
            f"{row['stage7e_limited_node_grouping_possible']} |"
        )

    lines.extend(
        [
            "",
            "## Topology",
            "",
            f"- `edge_index.npy` available: `{topology_status['edge_index_available']}`",
            f"- Row-major 4×4 mesh confirmed: "
            f"`{topology_status['row_major_numbering_confirmed']}`",
            f"- Router 12 neighbours: `{topology_status['router_12_neighbors']}`",
            "",
            "## Stage 6 alignment",
            "",
            f"- Stage 6 dominant-run agreement aligned: `{stage6_aligned}`",
            "",
            "## Leakage boundary",
            "",
            "- Training samples are explicitly identified before any summary fitting.",
            "- Stage 7C training ranges and distance standardization must use only "
            "`split=train` rows.",
            "- Stage 7D classifier fitting must use training runs only; limited "
            "regularization choices may use validation runs only.",
            "- Test runs remain frozen diagnostic evaluation cases.",
            "- Stage 7A performs no data-derived scaling and selects no features.",
            "",
            "## Next step",
            "",
            "After this alignment passes, Stage 7B may compute global, router, "
            "temporal, directional and saturation summaries from the normalized tensor.",
            "",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 7A: align the V3 feature tensor, labels, splits, prediction "
            "exports and selected diagnostic run groups without loading the full "
            "feature tensor into RAM."
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
    parser.add_argument("--expected-routers", type=int, default=16)
    parser.add_argument("--expected-window-length", type=int, default=8)
    parser.add_argument("--expected-features", type=int, default=24)
    parser.add_argument("--expected-runs", type=int, default=71)
    parser.add_argument("--expected-windows-per-run", type=int, default=3293)
    parser.add_argument("--expected-train-samples", type=int, default=148185)
    parser.add_argument("--expected-val-samples", type=int, default=42809)
    parser.add_argument("--expected-test-samples", type=int, default=42809)
    parser.add_argument("--expected-train-runs", type=int, default=45)
    parser.add_argument("--expected-val-runs", type=int, default=13)
    parser.add_argument("--expected-test-runs", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    failure_root = args.failure_root.resolve()
    prediction_directory = failure_root / "predictions"
    table_directory = failure_root / "tables"
    log_directory = failure_root / "logs"

    metadata_path = prediction_directory / "prediction_metadata.csv"
    stage6_agreement_path = (
        table_directory / "matched_run_model_agreement.csv"
    )

    output_paths = {
        "dataset_manifest": table_directory / "stage7_dataset_manifest.csv",
        "prediction_manifest": table_directory / "stage7_prediction_array_manifest.csv",
        "sample_alignment": table_directory / "stage7_sample_alignment.csv",
        "run_inventory": table_directory / "stage7_run_inventory.csv",
        "selected_run_inventory": table_directory / "stage7_selected_run_inventory.csv",
        "split_summary": table_directory / "stage7_split_summary.csv",
        "feature_map": table_directory / "stage7_feature_map.csv",
        "topology_manifest": table_directory / "stage7_topology_manifest.csv",
        "node_export_status": table_directory / "stage7_node_export_status.csv",
        "selected_indices": table_directory / "stage7_selected_sample_indices.npz",
        "alignment_summary": table_directory / "stage7a_alignment_summary.json",
        "report": failure_root / "STAGE7A_ALIGNMENT_REPORT.md",
        "validation": log_directory / "34_stage7a_alignment_validation.txt",
    }

    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Stage 7A outputs already exist. Use --overwrite to replace them:\n"
            + "\n".join(str(path) for path in existing)
        )

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    print(f"dataset root: {dataset_root}", flush=True)
    print(f"failure root: {failure_root}", flush=True)
    print(f"prediction metadata: {metadata_path}", flush=True)
    print("feature tensor full-load requested: False", flush=True)
    print("inference rerun: False", flush=True)
    print("training performed: False", flush=True)

    manifest_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray | None] = {}

    for role, filename in REQUIRED_DATASET_ARRAYS.items():
        path = dataset_root / filename
        mmap = role in {"x", "y_graph", "y_node", "run_id", "end_epoch"}
        allow_pickle = role in {"run_id", "feature_cols"}
        record, array = npy_array_record(
            path,
            role,
            True,
            mmap,
            allow_pickle,
        )
        manifest_rows.append(record)
        arrays[role] = array

    for role, filename in OPTIONAL_DATASET_ARRAYS.items():
        path = dataset_root / filename
        if role in arrays:
            continue
        mmap = role not in {"split"}
        allow_pickle = role in {"sample_id", "split"}
        record, array = npy_array_record(
            path,
            role,
            False,
            mmap,
            allow_pickle,
        )
        manifest_rows.append(record)
        arrays[role] = array

    dataset_manifest = pd.DataFrame(manifest_rows)

    missing_required = dataset_manifest[
        dataset_manifest["required"] & ~dataset_manifest["exists"]
    ]
    if not missing_required.empty:
        raise FileNotFoundError(
            "Required dataset arrays missing:\n"
            + missing_required[["role", "absolute_path"]].to_string(index=False)
        )

    x = arrays["x"]
    y_graph = arrays["y_graph"]
    y_node = arrays["y_node"]
    feature_cols = arrays["feature_cols"]
    run_id_array = arrays["run_id"]
    end_epoch_array = arrays["end_epoch"]

    assert x is not None
    assert y_graph is not None
    assert y_node is not None
    assert feature_cols is not None
    assert run_id_array is not None
    assert end_epoch_array is not None

    expected_x_shape = (
        args.expected_samples,
        args.expected_routers,
        args.expected_window_length,
        args.expected_features,
    )
    x_shape_valid = tuple(x.shape) == expected_x_shape
    graph_shape_valid = tuple(y_graph.shape) == (args.expected_samples,)
    node_shape_valid = tuple(y_node.shape) == (
        args.expected_samples,
        args.expected_routers,
    )
    run_id_shape_valid = tuple(run_id_array.shape) == (args.expected_samples,)
    end_epoch_shape_valid = tuple(end_epoch_array.shape) == (
        args.expected_samples,
    )
    feature_shape_valid = np.asarray(feature_cols).reshape(-1).shape == (
        args.expected_features,
    )
    x_is_memmap = isinstance(x, np.memmap)

    if not all(
        [
            x_shape_valid,
            graph_shape_valid,
            node_shape_valid,
            run_id_shape_valid,
            end_epoch_shape_valid,
            feature_shape_valid,
        ]
    ):
        raise RuntimeError(
            "Dataset shape validation failed. Inspect stage7_dataset_manifest.csv."
        )

    print(f"x shape: {x.shape}", flush=True)
    print(f"x dtype: {x.dtype}", flush=True)
    print(f"x is memory mapped: {x_is_memmap}", flush=True)

    run_ids = decode_string_array(run_id_array)
    end_epochs = np.asarray(end_epoch_array, dtype=np.int64).reshape(-1)
    graph_labels = np.asarray(y_graph, dtype=np.int8).reshape(-1)
    sample_ids = canonical_sample_ids(run_ids, end_epochs)
    true_attacker_count = chunked_node_counts(y_node)

    sample_ids_unique = len(np.unique(sample_ids)) == args.expected_samples
    graph_labels_binary = set(np.unique(graph_labels)).issubset({0, 1})
    node_labels_binary = set(
        np.unique(np.asarray(y_node[: min(8192, len(y_node))]))
    ).issubset({0, 1})

    if not sample_ids_unique:
        raise RuntimeError("Canonical dataset sample IDs are not unique.")
    if not graph_labels_binary:
        raise RuntimeError("y_graph contains values outside {0,1}.")
    if not node_labels_binary:
        raise RuntimeError(
            "Sampled y_node values contain labels outside {0,1}."
        )

    stage3_metadata = pd.read_csv(
        metadata_path,
        keep_default_na=False,
        low_memory=False,
        dtype={
            "sample_id": str,
            "run_id": str,
            "split": str,
        },
    )
    required_metadata_columns = {
        "sample_index",
        "sample_id",
        "run_id",
        "split",
        "end_epoch",
        "true_graph",
    }
    missing_metadata_columns = required_metadata_columns - set(
        stage3_metadata.columns
    )
    if missing_metadata_columns:
        raise RuntimeError(
            f"Stage 3 metadata missing: {sorted(missing_metadata_columns)}"
        )

    stage3_indices = stage3_metadata["sample_index"].to_numpy(dtype=np.int64)
    stage3_indices_unique = len(np.unique(stage3_indices)) == len(stage3_indices)
    stage3_indices_in_range = bool(
        np.all((stage3_indices >= 0) & (stage3_indices < args.expected_samples))
    )
    if not stage3_indices_unique or not stage3_indices_in_range:
        raise RuntimeError("Invalid Stage 3 sample indices.")

    split, split_source, split_crosscheck = reconstruct_split(
        args.expected_samples,
        stage3_metadata,
        dataset_root,
    )

    sample_alignment = pd.DataFrame(
        {
            "dataset_index": np.arange(args.expected_samples, dtype=np.int64),
            "sample_id": sample_ids,
            "run_id": run_ids,
            "end_epoch": end_epochs,
            "split": split,
            "true_graph": graph_labels,
            "true_attacker_count": true_attacker_count,
            "is_stage3_exported": False,
            "stage3_export_order": -1,
        }
    )
    sample_alignment.loc[stage3_indices, "is_stage3_exported"] = True
    sample_alignment.loc[
        stage3_indices,
        "stage3_export_order",
    ] = np.arange(len(stage3_indices), dtype=np.int64)
    sample_alignment = add_temporal_positions(sample_alignment)

    prediction_array_manifest, loaded_predictions = prediction_manifest(
        prediction_directory
    )
    predictions_aligned, node_labels_matched, prediction_messages = (
        verify_prediction_exports(
            stage3_metadata,
            loaded_predictions,
            sample_alignment,
            y_node,
        )
    )

    run_inventory = build_run_inventory(sample_alignment)
    selected_run_inventory, _ = build_selected_run_inventory(run_inventory)
    selected_arrays = selected_index_arrays(
        sample_alignment,
        selected_run_inventory,
    )
    split_summary = build_split_summary(
        sample_alignment,
        run_inventory,
    )
    feature_map = build_feature_map(feature_cols)

    edge_index = arrays.get("edge_index")
    topology_manifest, topology_status = build_topology_manifest(
        edge_index,
        args.expected_routers,
    )

    node_export_status = inspect_node_exports(
        loaded_predictions,
        len(stage3_metadata),
        args.expected_routers,
    )

    stage6_aligned, stage6_rows, stage6_message = verify_stage6_alignment(
        stage6_agreement_path,
        stage3_metadata,
    )

    split_counts = dict(
        zip(split_summary["split"], split_summary["sample_count"])
    )
    run_counts = dict(
        zip(split_summary["split"], split_summary["run_count"])
    )

    split_counts_valid = (
        split_counts.get("train") == args.expected_train_samples
        and split_counts.get("val") == args.expected_val_samples
        and split_counts.get("test") == args.expected_test_samples
    )
    run_counts_valid = (
        run_counts.get("train") == args.expected_train_runs
        and run_counts.get("val") == args.expected_val_runs
        and run_counts.get("test") == args.expected_test_runs
        and len(run_inventory) == args.expected_runs
    )
    windows_per_run_valid = bool(
        run_inventory["number_of_windows"]
        .eq(args.expected_windows_per_run)
        .all()
    )
    run_metadata_constant = bool(
        run_inventory[
            [
                "split_constant_within_run",
                "graph_label_constant_within_run",
                "attacker_count_constant_within_run",
            ]
        ].all().all()
    )
    selected_runs_present = (
        selected_run_inventory["run_id"].nunique()
        == len(
            {
                run_id
                for members in SELECTED_GROUPS.values()
                for _, run_id in members
            }
        )
    )
    feature_names_unique = (
        feature_map["feature_name"].nunique() == args.expected_features
    )
    selected_indices_complete = all(
        len(array) == args.expected_windows_per_run
        if key.startswith("run__")
        else len(array) > 0
        for key, array in selected_arrays.items()
    )

    deterministic_probe_indices = np.unique(
        np.concatenate(
            [
                np.asarray([0, args.expected_samples // 2, args.expected_samples - 1]),
                *[
                    values[: min(16, len(values))]
                    for key, values in selected_arrays.items()
                    if key.startswith("run__")
                ],
            ]
        )
    )
    probe = np.asarray(x[deterministic_probe_indices])
    probed_features_finite = bool(np.isfinite(probe).all())

    node_shapes_aligned = bool(
        node_export_status["all_present_node_shapes_aligned"].all()
    )

    overall_success = all(
        [
            x_shape_valid,
            graph_shape_valid,
            node_shape_valid,
            run_id_shape_valid,
            end_epoch_shape_valid,
            feature_shape_valid,
            x_is_memmap,
            sample_ids_unique,
            graph_labels_binary,
            node_labels_binary,
            stage3_indices_unique,
            stage3_indices_in_range,
            split_crosscheck,
            predictions_aligned,
            node_labels_matched,
            split_counts_valid,
            run_counts_valid,
            windows_per_run_valid,
            run_metadata_constant,
            selected_runs_present,
            selected_indices_complete,
            feature_names_unique,
            node_shapes_aligned,
            stage6_aligned,
            probed_features_finite,
        ]
    )

    validation_lines = [
        f"required dataset arrays present: {missing_required.empty}",
        f"feature tensor shape verified: {x_shape_valid}",
        f"feature tensor memory mapped: {x_is_memmap}",
        f"graph-label shape verified: {graph_shape_valid}",
        f"node-label shape verified: {node_shape_valid}",
        f"run-id shape verified: {run_id_shape_valid}",
        f"end-epoch shape verified: {end_epoch_shape_valid}",
        f"feature-name count verified: {feature_shape_valid}",
        f"canonical sample IDs unique: {sample_ids_unique}",
        f"graph labels binary: {graph_labels_binary}",
        f"sampled node labels binary: {node_labels_binary}",
        f"Stage 3 sample indices unique: {stage3_indices_unique}",
        f"Stage 3 sample indices in range: {stage3_indices_in_range}",
        f"dataset split cross-check passed: {split_crosscheck}",
        f"all model prediction exports aligned: {predictions_aligned}",
        f"exported node labels matched when present: {node_labels_matched}",
        f"node export shapes aligned when present: {node_shapes_aligned}",
        f"split sample counts verified: {split_counts_valid}",
        f"split run counts verified: {run_counts_valid}",
        f"windows per run verified: {windows_per_run_valid}",
        f"run metadata constant: {run_metadata_constant}",
        f"selected Stage 7 runs present: {selected_runs_present}",
        f"selected sample-index arrays complete: {selected_indices_complete}",
        f"feature names unique: {feature_names_unique}",
        f"Stage 6 agreement aligned: {stage6_aligned}",
        f"deterministic feature probes finite: {probed_features_finite}",
        f"full feature tensor scanned: False",
        f"inference rerun: False",
        f"training performed: False",
        f"overall success: {overall_success}",
    ]

    detail_messages = [
        *prediction_messages,
    ]
    if stage6_message:
        detail_messages.append(stage6_message)

    validation_text = "\n".join(validation_lines)
    if detail_messages:
        validation_text += "\n\nDETAILS:\n" + "\n".join(detail_messages)
    validation_text += "\n"

    atomic_text(validation_text, output_paths["validation"])
    print()
    print(validation_text, end="", flush=True)

    if not overall_success:
        raise SystemExit(
            f"Stage 7A validation failed. Inspect {output_paths['validation']}."
        )

    atomic_csv(dataset_manifest, output_paths["dataset_manifest"])
    atomic_csv(
        prediction_array_manifest,
        output_paths["prediction_manifest"],
    )
    atomic_csv(sample_alignment, output_paths["sample_alignment"])
    atomic_csv(run_inventory, output_paths["run_inventory"])
    atomic_csv(
        selected_run_inventory,
        output_paths["selected_run_inventory"],
    )
    atomic_csv(split_summary, output_paths["split_summary"])
    atomic_csv(feature_map, output_paths["feature_map"])
    atomic_csv(topology_manifest, output_paths["topology_manifest"])
    atomic_csv(node_export_status, output_paths["node_export_status"])
    atomic_npz(output_paths["selected_indices"], **selected_arrays)

    alignment_summary = {
        "dataset_root": str(dataset_root),
        "failure_root": str(failure_root),
        "x_shape": list(x.shape),
        "x_dtype": str(x.dtype),
        "x_is_memmap": x_is_memmap,
        "split_source": split_source,
        "split_crosscheck": split_crosscheck,
        "sample_counts": split_counts,
        "run_counts": run_counts,
        "total_runs": int(len(run_inventory)),
        "windows_per_run": args.expected_windows_per_run,
        "feature_count": int(len(feature_map)),
        "stage3_exported_samples": int(len(stage3_metadata)),
        "stage6_agreement_rows": int(stage6_rows),
        "stage6_aligned": stage6_aligned,
        "node_grouping_possible_by_model": {
            row["model_name"]: bool(
                row["stage7e_limited_node_grouping_possible"]
            )
            for _, row in node_export_status.iterrows()
        },
        "topology": topology_status,
        "full_feature_tensor_scanned": False,
        "inference_rerun": False,
        "training_performed": False,
        "overall_success": overall_success,
    }
    atomic_json(alignment_summary, output_paths["alignment_summary"])

    report = build_alignment_report(
        dataset_root,
        failure_root,
        x,
        feature_map,
        split_summary,
        run_inventory,
        selected_run_inventory,
        node_export_status,
        topology_status,
        split_source,
        split_crosscheck,
        stage6_aligned,
    )
    atomic_text(report, output_paths["report"])

    print("Stage 7A output files:", flush=True)
    for name, path in output_paths.items():
        print(f"  {name}: {path}", flush=True)
    print("overall success: True", flush=True)


if __name__ == "__main__":
    main()
