#!/usr/bin/env python3
"""
V5 P2-A2 Feature, Normalization, and Physical-Port Mask Contract

Purpose
-------
Freeze the P2 learned-input contract before implementing the P2 loader.

This stage verifies:
- the exact 81-feature schema and [time, router, feature] tensor order;
- compatibility with the frozen P0 PRIMARY58 indices;
- train-fitted log1p + standardization provenance;
- prohibition of a second normalization pass;
- topology integrity for the 4x4, 16-router directed mesh;
- recoverability of a raw Boolean [16,10] physical-port mask;
- empirical agreement, on deterministic TRAIN/VALIDATION samples only,
  between the topology-derived mask and inverse-transformed stored
  port-valid channels 71..80.

It never enumerates or opens runs/test.

The current dataset_loader.py is treated as a reference loader only. It is not
authorized for P2 training because it:
- performs native per-run windowing instead of the frozen pair-aligned policy;
- returns all 81 features instead of PRIMARY58;
- returns provenance fields that must be quarantined;
- always returns edge_index although frozen B3 has no graph message passing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


STAGE = "V5_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT"
COMPLETE = f"{STAGE}_COMPLETE"

WINDOW = 32
STRIDE = 8
NUM_NODES = 16
FEATURE_COUNT = 81

DIRECTIONS = ("local", "north", "east", "south", "west")

PRIMARY58_INDICES = (
    list(range(0, 30))
    + list(range(31, 56))
    + [62, 64, 65]
)
STATUS_FLAG_INDICES = [30]
NONPRIMARY_DYNAMIC_INDICES = list(range(56, 62)) + [63] + list(range(66, 71))
STORED_PORT_VALID_INDICES = list(range(71, 81))

EXPECTED_FEATURE_NAMES = (
    [f"in_count_{direction}" for direction in DIRECTIONS]
    + [f"out_count_{direction}" for direction in DIRECTIONS]
    + [f"in_gap_sum_{direction}" for direction in DIRECTIONS]
    + [f"in_gap_count_{direction}" for direction in DIRECTIONS]
    + [f"out_gap_sum_{direction}" for direction in DIRECTIONS]
    + [f"out_gap_count_{direction}" for direction in DIRECTIONS]
    + ["status_flags"]
    + [f"enqueue_count_{direction}" for direction in DIRECTIONS]
    + [f"dequeue_count_{direction}" for direction in DIRECTIONS]
    + [f"occupancy_cycle_sum_{direction}" for direction in DIRECTIONS]
    + [f"occupancy_max_{direction}" for direction in DIRECTIONS]
    + [f"occupancy_end_{direction}" for direction in DIRECTIONS]
    + [f"stall_no_free_vc_{direction}" for direction in DIRECTIONS]
    + [f"stall_no_credit_{direction}" for direction in DIRECTIONS]
    + [f"stall_ordering_{direction}" for direction in DIRECTIONS]
    + [f"input_port_valid_{direction}" for direction in DIRECTIONS]
    + [f"output_port_valid_{direction}" for direction in DIRECTIONS]
)

EXPECTED_FORBIDDEN_SCHEMA_TERMS = {
    "epoch_id",
    "router_id",
    "tick fields",
    "status fields",
    "valid_workload_epoch",
    "attack_active",
    "active_attack_count",
    "configured source/victim IDs",
    "packet truth",
    "path truth",
    "run/pair IDs",
    "seed",
    "mode",
    "split",
}

MODEL_PROVENANCE_FORBIDDEN = [
    "epoch_id",
    "router_id",
    "tick",
    "status_flags",
    "status fields",
    "valid_workload_epoch",
    "attack_active",
    "active_attack_count",
    "configured source IDs",
    "configured victim IDs",
    "packet truth",
    "path truth",
    "case_id",
    "pair_id",
    "run_id",
    "filename",
    "seed",
    "mode",
    "split",
    "category",
    "native_run_length",
    "common_pair_length",
    "native_window_count",
    "aligned_window_count",
    "window_start",
    "window_target",
    "window_end",
    "distance_to_terminal_epoch",
    "serialization size",
    "manifest position",
    "router coordinates as learned features",
    "learned router embeddings",
    "any direct or derived provenance identifier",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pair_key(path: Path) -> str | None:
    stem = path.stem
    if stem.endswith("_ATTACK"):
        return stem[:-len("_ATTACK")]
    if stem.endswith("_CONTROL"):
        return stem[:-len("_CONTROL")]
    return None


def select_sample_files(split_dir: Path) -> list[Path]:
    files = sorted(split_dir.glob("*.pt"))
    by_name = {path.name: path for path in files}
    keys = sorted(
        {
            key
            for path in files
            if (key := pair_key(path)) is not None
        }
    )
    if not keys:
        return []

    indices = sorted({0, len(keys) // 2, len(keys) - 1})
    selected: list[Path] = []
    for index in indices:
        key = keys[index]
        for suffix in ("ATTACK", "CONTROL"):
            name = f"{key}_{suffix}.pt"
            path = by_name.get(name)
            if path is not None:
                selected.append(path)
    return selected


def topology_port_mask(
    router_coordinates: torch.Tensor,
    mesh_rows: int,
    mesh_cols: int,
) -> torch.Tensor:
    """
    Return Boolean [num_nodes,10] in this exact order:
      input : local,north,east,south,west
      output: local,north,east,south,west

    Coordinates are interpreted as [row, col], with row increasing south and
    col increasing east.
    """
    coordinates = router_coordinates.to(torch.int64)
    row = coordinates[:, 0]
    col = coordinates[:, 1]

    local = torch.ones_like(row, dtype=torch.bool)
    north = row > 0
    east = col < mesh_cols - 1
    south = row < mesh_rows - 1
    west = col > 0

    one_side = torch.stack(
        (local, north, east, south, west),
        dim=1,
    )
    return torch.cat((one_side, one_side), dim=1)


def expected_directed_edges(
    router_coordinates: torch.Tensor,
) -> set[tuple[int, int]]:
    coordinates = [
        (int(item[0]), int(item[1]))
        for item in router_coordinates.tolist()
    ]
    coordinate_to_node = {
        coordinate: node
        for node, coordinate in enumerate(coordinates)
    }

    edges: set[tuple[int, int]] = set()
    for source, (row, col) in enumerate(coordinates):
        for delta_row, delta_col in (
            (-1, 0),
            (0, 1),
            (1, 0),
            (0, -1),
        ):
            target = coordinate_to_node.get(
                (row + delta_row, col + delta_col)
            )
            if target is not None:
                edges.add((source, target))
    return edges


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a0-dir", type=Path, required=True)
    parser.add_argument("--a1-r1-dir", type=Path, required=True)
    parser.add_argument("--a1-r2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root_input = args.root.expanduser()
    root = root_input.resolve()
    a0_dir = args.a0_dir.expanduser().resolve()
    a1_r1_dir = args.a1_r1_dir.expanduser().resolve()
    a1_r2_dir = args.a1_r2_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    paths = {
        "a0_report": (
            a0_dir
            / "V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT.json"
        ),
        "a0_lock": (
            a0_dir
            / "V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT_LOCK.json"
        ),
        "a1_r1_report": (
            a1_r1_dir
            / "V5_P2_A1_R1_PAIR_LENGTH_AND_WINDOW_SHORTCUT_AUDIT.json"
        ),
        "a1_r1_lock": (
            a1_r1_dir
            / "V5_P2_A1_R1_PAIR_LENGTH_AND_WINDOW_SHORTCUT_AUDIT_LOCK.json"
        ),
        "a1_r2_report": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT.json"
        ),
        "a1_r2_lock": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT_LOCK.json"
        ),
        "a1_r2_contract": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT.json"
        ),
        "feature_schema": root / "feature_schema.json",
        "normalization": root / "normalization.pt",
        "topology": root / "topology.pt",
        "dataset_loader": root / "dataset_loader.py",
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    for split in ("train", "validation"):
        if not (root / "runs" / split).is_dir():
            failures.append(f"missing runs/{split}")

    # Test existence is not checked here to keep the boundary stronger.
    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        return 1

    a0_report = load_json(paths["a0_report"])
    a0_lock = load_json(paths["a0_lock"])
    r1_report = load_json(paths["a1_r1_report"])
    r1_lock = load_json(paths["a1_r1_lock"])
    r2_report = load_json(paths["a1_r2_report"])
    r2_lock = load_json(paths["a1_r2_lock"])

    if a0_report.get("status") != "COMPLETE":
        failures.append("A0 status is not COMPLETE")
    if r1_report.get("status") != "COMPLETE":
        failures.append("A1-R1 status is not COMPLETE")
    if r2_report.get("status") != "COMPLETE":
        failures.append("A1-R2 status is not COMPLETE")

    if (
        a0_lock.get("report_sha256")
        != sha256_file(paths["a0_report"])
    ):
        failures.append("A0 report SHA mismatch")
    if (
        r1_lock.get("report_sha256")
        != sha256_file(paths["a1_r1_report"])
    ):
        failures.append("A1-R1 report SHA mismatch")
    if (
        r2_lock.get("report_sha256")
        != sha256_file(paths["a1_r2_report"])
    ):
        failures.append("A1-R2 report SHA mismatch")

    if (
        r2_lock.get("decision")
        != "FREEZE_PAIR_ALIGNED_WINDOW_CONTRACT_AND_AUTHORIZE_A2"
    ):
        failures.append("A1-R2 did not authorize A2")
    if r2_lock.get("window") != WINDOW:
        failures.append("A1-R2 window is not 32")
    if r2_lock.get("stride") != STRIDE:
        failures.append("A1-R2 stride is not 8")

    for label, document in (
        ("A0", a0_report),
        ("A1-R1", r1_report),
        ("A1-R2", r2_report),
    ):
        if document.get("security_boundary", {}).get(
            "test_tensor_contents_accessed"
        ) is not False:
            failures.append(
                f"{label} does not certify untouched test tensors"
            )

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    schema = load_json(paths["feature_schema"])
    normalization = torch.load(
        paths["normalization"],
        map_location="cpu",
        weights_only=False,
    )
    topology = torch.load(
        paths["topology"],
        map_location="cpu",
        weights_only=False,
    )
    loader_source = paths["dataset_loader"].read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Feature schema audit
    # ------------------------------------------------------------------
    if schema.get("schema_version") != 1:
        failures.append(
            f"feature schema version={schema.get('schema_version')!r}"
        )
    if schema.get("feature_count") != FEATURE_COUNT:
        failures.append(
            f"feature_count={schema.get('feature_count')!r}, expected 81"
        )
    if schema.get("input_tensor_order") != [
        "time",
        "router",
        "feature",
    ]:
        failures.append(
            "input_tensor_order is not ['time','router','feature']"
        )

    features = schema.get("features")
    if not isinstance(features, list) or len(features) != FEATURE_COUNT:
        failures.append("feature schema does not contain 81 records")
        feature_names: list[str] = []
    else:
        feature_names = [
            record.get("name") if isinstance(record, dict) else None
            for record in features
        ]

        if feature_names != EXPECTED_FEATURE_NAMES:
            mismatches = []
            for index, (actual, expected) in enumerate(
                zip(feature_names, EXPECTED_FEATURE_NAMES)
            ):
                if actual != expected:
                    mismatches.append(
                        {
                            "index": index,
                            "actual": actual,
                            "expected": expected,
                        }
                    )
            failures.append(
                "81-feature name/order mismatch; first mismatches="
                f"{mismatches[:10]}"
            )

        for index, record in enumerate(features):
            if not isinstance(record, dict):
                failures.append(
                    f"feature record {index} is not a dictionary"
                )
                continue
            if record.get("index") != index:
                failures.append(
                    f"feature record {index} stores index="
                    f"{record.get('index')!r}"
                )
            if record.get("transform") != (
                "log1p_then_train_standardize"
            ):
                failures.append(
                    f"feature {index} transform="
                    f"{record.get('transform')!r}"
                )
            if record.get("is_static_mask") is not False:
                warnings.append(
                    f"feature {index} unexpectedly marks "
                    "is_static_mask=true"
                )

    forbidden_schema = set(schema.get("forbidden_model_inputs", []))
    missing_forbidden = sorted(
        EXPECTED_FORBIDDEN_SCHEMA_TERMS - forbidden_schema
    )
    if missing_forbidden:
        failures.append(
            f"feature schema omits forbidden terms: {missing_forbidden}"
        )

    if schema.get("excluded_observation_columns") != [
        "tick",
        "epoch_id",
        "router_id",
    ]:
        failures.append(
            "excluded_observation_columns do not match contract"
        )

    # ------------------------------------------------------------------
    # Normalization provenance audit
    # ------------------------------------------------------------------
    if not isinstance(normalization, dict):
        failures.append("normalization.pt payload is not a dictionary")
    else:
        normalization_names = normalization.get("feature_names")
        mean = normalization.get("mean")
        std = normalization.get("std")
        mask_features = normalization.get("mask_features")

        if normalization_names != EXPECTED_FEATURE_NAMES:
            failures.append(
                "normalization feature_names do not match schema order"
            )
        if (
            not isinstance(mean, torch.Tensor)
            or tuple(mean.shape) != (FEATURE_COUNT,)
            or not mean.is_floating_point()
            or not bool(torch.isfinite(mean).all().item())
        ):
            failures.append("normalization mean is invalid")
        if (
            not isinstance(std, torch.Tensor)
            or tuple(std.shape) != (FEATURE_COUNT,)
            or not std.is_floating_point()
            or not bool(torch.isfinite(std).all().item())
            or not bool((std > 0).all().item())
        ):
            failures.append("normalization std is invalid")
        if (
            not isinstance(mask_features, torch.Tensor)
            or tuple(mask_features.shape) != (FEATURE_COUNT,)
            or mask_features.dtype != torch.bool
        ):
            failures.append("normalization mask_features is invalid")
        elif bool(mask_features.any().item()):
            failures.append(
                "normalization designates stored mask features, "
                "contrary to observed all-false P2 artifact"
            )
        else:
            warnings.append(
                "normalization mask_features is all false; stored "
                "port-valid channels are therefore not trusted as raw "
                "Boolean model inputs"
            )

        if normalization.get("dynamic_transform") != (
            "log1p_then_standardize"
        ):
            failures.append("unexpected dynamic_transform")
        if normalization.get("mask_transform") != "identity":
            failures.append("unexpected mask_transform")
        if normalization.get("fit_split") != "train":
            failures.append("normalization was not fit on train")
        if (
            not isinstance(
                normalization.get("fit_router_epoch_rows"),
                int,
            )
            or normalization["fit_router_epoch_rows"] <= 0
        ):
            failures.append("invalid fit_router_epoch_rows")

    # ------------------------------------------------------------------
    # Topology integrity and mask construction
    # ------------------------------------------------------------------
    required_topology_keys = {
        "edge_index",
        "num_nodes",
        "mesh_rows",
        "mesh_cols",
        "router_coordinates",
        "directed_edge_count",
        "undirected_edge_count",
    }
    if not isinstance(topology, dict):
        failures.append("topology.pt payload is not a dictionary")
        edge_index = None
        coordinates = None
        physical_mask = None
    else:
        missing = sorted(required_topology_keys - set(topology))
        if missing:
            failures.append(f"topology missing keys: {missing}")

        edge_index = topology.get("edge_index")
        coordinates = topology.get("router_coordinates")

        if topology.get("num_nodes") != NUM_NODES:
            failures.append("topology num_nodes is not 16")
        if topology.get("mesh_rows") != 4:
            failures.append("topology mesh_rows is not 4")
        if topology.get("mesh_cols") != 4:
            failures.append("topology mesh_cols is not 4")
        if topology.get("directed_edge_count") != 48:
            failures.append("topology directed_edge_count is not 48")
        if topology.get("undirected_edge_count") != 24:
            failures.append("topology undirected_edge_count is not 24")

        if (
            not isinstance(edge_index, torch.Tensor)
            or edge_index.dtype != torch.int64
            or tuple(edge_index.shape) != (2, 48)
        ):
            failures.append("edge_index is not int64 [2,48]")
        if (
            not isinstance(coordinates, torch.Tensor)
            or coordinates.dtype != torch.int64
            or tuple(coordinates.shape) != (16, 2)
        ):
            failures.append(
                "router_coordinates is not int64 [16,2]"
            )

        physical_mask = None
        if (
            isinstance(edge_index, torch.Tensor)
            and tuple(edge_index.shape) == (2, 48)
            and isinstance(coordinates, torch.Tensor)
            and tuple(coordinates.shape) == (16, 2)
        ):
            coordinate_tuples = {
                tuple(int(value) for value in item)
                for item in coordinates.tolist()
            }
            expected_coordinates = {
                (row, col)
                for row in range(4)
                for col in range(4)
            }
            if coordinate_tuples != expected_coordinates:
                failures.append(
                    "router coordinates do not cover the 4x4 grid"
                )

            actual_edges = {
                (int(source), int(target))
                for source, target in edge_index.t().tolist()
            }
            expected_edges = expected_directed_edges(coordinates)

            if len(actual_edges) != 48:
                failures.append(
                    f"edge_index has {len(actual_edges)} unique edges"
                )
            if any(source == target for source, target in actual_edges):
                failures.append("edge_index contains self loops")
            if actual_edges != expected_edges:
                failures.append(
                    "edge_index does not match coordinate-derived "
                    "4-neighbor directed mesh"
                )
            if any(
                (target, source) not in actual_edges
                for source, target in actual_edges
            ):
                failures.append(
                    "edge_index is not directionally symmetric"
                )

            physical_mask = topology_port_mask(
                coordinates,
                mesh_rows=4,
                mesh_cols=4,
            )
            if tuple(physical_mask.shape) != (16, 10):
                failures.append(
                    "derived physical mask is not [16,10]"
                )
            if physical_mask.dtype != torch.bool:
                failures.append(
                    "derived physical mask is not Boolean"
                )

    # ------------------------------------------------------------------
    # Empirical non-test recovery check for channels 71..80
    # ------------------------------------------------------------------
    sample_records: list[dict[str, Any]] = []
    maximum_mask_recovery_error = 0.0
    sampled_train_files = 0
    sampled_validation_files = 0

    if (
        isinstance(normalization, dict)
        and isinstance(normalization.get("mean"), torch.Tensor)
        and isinstance(normalization.get("std"), torch.Tensor)
        and isinstance(physical_mask, torch.Tensor)
    ):
        port_mean = normalization["mean"][
            STORED_PORT_VALID_INDICES
        ].view(1, 1, 10)
        port_std = normalization["std"][
            STORED_PORT_VALID_INDICES
        ].view(1, 1, 10)
        expected_mask = physical_mask.to(torch.float32).view(
            1,
            16,
            10,
        )

        for split in ("train", "validation"):
            sample_paths = select_sample_files(root / "runs" / split)
            if len(sample_paths) != 6:
                failures.append(
                    f"{split} deterministic sample count="
                    f"{len(sample_paths)}, expected 6"
                )

            for path in sample_paths:
                payload = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False,
                )
                if split == "train":
                    sampled_train_files += 1
                else:
                    sampled_validation_files += 1

                if not isinstance(payload, dict):
                    failures.append(
                        f"{split}/{path.name} is not a dict"
                    )
                    continue
                x = payload.get("x")
                if (
                    not isinstance(x, torch.Tensor)
                    or x.ndim != 3
                    or tuple(x.shape[1:]) != (16, 81)
                ):
                    failures.append(
                        f"{split}/{path.name} has invalid x shape"
                    )
                    continue
                if not bool(torch.isfinite(x).all().item()):
                    failures.append(
                        f"{split}/{path.name} x has NaN/Inf"
                    )
                    continue

                stored_standardized = x[
                    :,
                    :,
                    STORED_PORT_VALID_INDICES,
                ]
                pre_standardization = (
                    stored_standardized * port_std + port_mean
                )
                recovered_raw = torch.expm1(pre_standardization)
                expanded_expected = expected_mask.expand(
                    x.shape[0],
                    -1,
                    -1,
                )
                error = float(
                    (
                        recovered_raw - expanded_expected
                    ).abs().max().item()
                )
                maximum_mask_recovery_error = max(
                    maximum_mask_recovery_error,
                    error,
                )

                sample_records.append(
                    {
                        "split": split,
                        "file": path.name,
                        "time_length": int(x.shape[0]),
                        "maximum_port_mask_recovery_error": error,
                        "primary58_finite": bool(
                            torch.isfinite(
                                x[:, :, PRIMARY58_INDICES]
                            ).all().item()
                        ),
                    }
                )

                if error > 1e-4:
                    failures.append(
                        f"{split}/{path.name} stored port-valid "
                        f"channels do not recover topology mask; "
                        f"max error={error:.8g}"
                    )

    # ------------------------------------------------------------------
    # Reference-loader safety audit
    # ------------------------------------------------------------------
    loader_findings = {
        "native_per_run_windowing_detected": (
            "range(0, length - self.window + 1, self.stride)"
            in loader_source
        ),
        "returns_all_81_features_detected": (
            'run["x"][start:stop].permute(1, 2, 0)'
            in loader_source
        ),
        "returns_edge_index_detected": (
            '"edge_index"' in loader_source
        ),
        "returns_epoch_id_detected": (
            '"epoch_id"' in loader_source
        ),
        "returns_case_id_detected": (
            '"case_id"' in loader_source
        ),
        "returns_run_id_detected": (
            '"run_id"' in loader_source
        ),
        "returns_mode_detected": (
            '"mode"' in loader_source
        ),
        "returns_window_start_detected": (
            '"window_start"' in loader_source
        ),
        "returns_window_target_detected": (
            '"window_target"' in loader_source
        ),
    }

    if not all(loader_findings.values()):
        warnings.append(
            "one or more expected reference-loader findings were "
            "not detected textually; inspect dataset_loader.py manually"
        )

    # ------------------------------------------------------------------
    # Freeze the contract
    # ------------------------------------------------------------------
    selected_feature_names = [
        EXPECTED_FEATURE_NAMES[index]
        for index in PRIMARY58_INDICES
    ]
    excluded_feature_indices = sorted(
        set(range(FEATURE_COUNT)) - set(PRIMARY58_INDICES)
    )
    excluded_feature_names = [
        EXPECTED_FEATURE_NAMES[index]
        for index in excluded_feature_indices
    ]

    mask_channels = [
        f"{side}_{direction}"
        for side in ("input", "output")
        for direction in DIRECTIONS
    ]

    contract = {
        "contract_name": (
            "V5_P2_PRIMARY58_PLUS_TOPOLOGY_BOOLEAN_PORT_MASK"
        ),
        "contract_version": 1,
        "architecture_compatibility": {
            "frozen_architecture": (
                "B3_CAUSAL_DEPTHWISE_SEPARABLE_CONV1D_ONLY"
            ),
            "graph_message_passing": False,
            "learned_temporal_feature_count": 58,
            "topology_mask_feature_count": 10,
            "model_temporal_input_shape": "[B,16,58,32]",
            "model_topology_mask_shape": "[B,16,10]",
            "edge_index_model_input": False,
        },
        "stored_tensor_contract": {
            "stored_x_shape": "[T,16,81]",
            "stored_axis_order": ["time", "router", "feature"],
            "stored_feature_count": 81,
            "stored_values_already_transformed": True,
            "stored_transform": "log1p_then_train_standardize",
            "normalization_fit_split": "train",
            "second_normalization_forbidden": True,
            "inverse_transform_during_training_forbidden": True,
        },
        "primary58": {
            "selection_basis": (
                "exact frozen P0 index and feature-name contract"
            ),
            "indices": PRIMARY58_INDICES,
            "names": selected_feature_names,
            "count": len(PRIMARY58_INDICES),
        },
        "excluded_stored_features": {
            "indices": excluded_feature_indices,
            "names": excluded_feature_names,
            "status_flag_indices": STATUS_FLAG_INDICES,
            "nonprimary_dynamic_indices": (
                NONPRIMARY_DYNAMIC_INDICES
            ),
            "stored_port_valid_indices": (
                STORED_PORT_VALID_INDICES
            ),
            "stored_port_valid_channels_may_not_be_used_as_model_input": True,
        },
        "physical_port_mask": {
            "source": "topology.pt only",
            "shape": [16, 10],
            "dtype": "bool",
            "model_cast_dtype": "float32",
            "channel_order": mask_channels,
            "coordinate_interpretation": {
                "router_coordinates": "[row,col]",
                "row_increases": "south",
                "column_increases": "east",
            },
            "rules": {
                "local": "always true",
                "north": "row > 0",
                "east": "col < mesh_cols - 1",
                "south": "row < mesh_rows - 1",
                "west": "col > 0",
                "input_output_physical_validity_identical": True,
            },
            "router_coordinates_learned_input": False,
            "edge_index_learned_input_for_frozen_b3": False,
            "stored_standardized_port_valid_channels_used": False,
            "empirical_sample_recovery_max_abs_error": (
                maximum_mask_recovery_error
            ),
        },
        "windowing_dependency": {
            "required_contract": (
                "V5_P2_PAIR_ALIGNED_COMMON_PREFIX_WINDOWS"
            ),
            "window": WINDOW,
            "stride": STRIDE,
            "reference_dataset_loader_authorized": False,
            "reason": (
                "reference loader performs native per-run windowing "
                "and exposes prohibited provenance"
            ),
        },
        "model_input_prohibitions": MODEL_PROVENANCE_FORBIDDEN,
        "bookkeeping_only": [
            "edge_index",
            "router_coordinates",
            "case_id",
            "pair_id",
            "run_id",
            "mode",
            "category",
            "split",
            "epoch_id",
            "window_start",
            "window_target",
            "native lengths",
            "aligned lengths",
        ],
        "test_boundary": {
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
            "test_loader_constructed": False,
        },
    }
    contract["contract_sha256"] = canonical_sha256(contract)

    mask_path = (
        output_dir
        / "V5_P2_A2_TOPOLOGY_DERIVED_BOOLEAN_PORT_MASK.pt"
    )
    if isinstance(physical_mask, torch.Tensor):
        torch.save(
            {
                "mask": physical_mask.cpu(),
                "channel_order": mask_channels,
                "source_topology_sha256": sha256_file(
                    paths["topology"]
                ),
                "contract_sha256": contract["contract_sha256"],
            },
            mask_path,
        )

    contract_path = (
        output_dir
        / "V5_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT.json"
    )
    write_json(contract_path, contract)

    markdown = f"""# V5 P2 Feature, Normalization, and Mask Contract

## Decision

P2 is compatible with the frozen P0 `PRIMARY58` learned-input contract.

```text
stored x              = [T,16,81]
selected temporal x   = indices 0-29, 31-55, 62, 64, 65
selected feature count= 58
window                = 32
model temporal input  = [B,16,58,32]
physical mask         = topology-derived Boolean [B,16,10]
```

## Normalization

The stored tensors already use:

```text
log1p_then_train_standardize
fit split = train
```

A second normalization pass is forbidden.

## Port-valid channels

Stored indices 71-80 are not trusted as raw model-mask inputs. The physical
mask is reconstructed from `topology.pt` and passed separately.

Observed maximum inverse-transform recovery error on deterministic non-test
samples:

```text
{maximum_mask_recovery_error:.10g}
```

## Frozen B3 interface

The frozen B3 architecture has no graph message passing. `edge_index` and
router coordinates are audit/bookkeeping artifacts only and are not model
inputs.

## Loader status

The supplied `dataset_loader.py` is not authorized for training. A new loader
must enforce the A1-R2 pair-aligned window contract, select PRIMARY58, return
the topology-derived mask, and quarantine all provenance fields.

## Contract hash

`{contract["contract_sha256"]}`

## Test boundary

No P2 test-directory enumeration or test-tensor access occurred.
"""
    markdown_path = (
        output_dir
        / "V5_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT.md"
    )
    atomic_write(markdown_path, markdown)

    status = "COMPLETE" if not failures else "HOLD"
    report = {
        "stage": STAGE,
        "status": status,
        "decision": (
            "FREEZE_PRIMARY58_NORMALIZATION_AND_TOPOLOGY_MASK_CONTRACT"
            if not failures
            else "BLOCK_P2_A3"
        ),
        "dataset": {
            "link_path": str(root_input.absolute()),
            "resolved_root": str(root),
        },
        "feature_audit": {
            "feature_count": len(EXPECTED_FEATURE_NAMES),
            "primary58_count": len(PRIMARY58_INDICES),
            "primary58_indices": PRIMARY58_INDICES,
            "primary58_names": selected_feature_names,
            "excluded_indices": excluded_feature_indices,
            "excluded_names": excluded_feature_names,
        },
        "normalization_audit": {
            "fit_split": normalization.get("fit_split"),
            "dynamic_transform": normalization.get(
                "dynamic_transform"
            ),
            "mask_transform": normalization.get("mask_transform"),
            "fit_router_epoch_rows": normalization.get(
                "fit_router_epoch_rows"
            ),
            "mask_features_true_count": int(
                normalization["mask_features"].sum().item()
            ),
            "second_normalization_forbidden": True,
        },
        "topology_audit": {
            "num_nodes": topology.get("num_nodes"),
            "mesh_rows": topology.get("mesh_rows"),
            "mesh_cols": topology.get("mesh_cols"),
            "directed_edge_count": topology.get(
                "directed_edge_count"
            ),
            "undirected_edge_count": topology.get(
                "undirected_edge_count"
            ),
            "physical_mask_shape": (
                list(physical_mask.shape)
                if isinstance(physical_mask, torch.Tensor)
                else None
            ),
            "physical_mask_true_count": (
                int(physical_mask.sum().item())
                if isinstance(physical_mask, torch.Tensor)
                else None
            ),
        },
        "sample_recovery_audit": {
            "sampled_train_files": sampled_train_files,
            "sampled_validation_files": sampled_validation_files,
            "records": sample_records,
            "maximum_mask_recovery_error": (
                maximum_mask_recovery_error
            ),
            "tolerance": 1e-4,
        },
        "reference_loader_findings": loader_findings,
        "contract": contract,
        "provenance": {
            "a0_report_sha256": sha256_file(paths["a0_report"]),
            "a0_lock_sha256": sha256_file(paths["a0_lock"]),
            "a1_r1_report_sha256": sha256_file(
                paths["a1_r1_report"]
            ),
            "a1_r1_lock_sha256": sha256_file(paths["a1_r1_lock"]),
            "a1_r2_report_sha256": sha256_file(
                paths["a1_r2_report"]
            ),
            "a1_r2_lock_sha256": sha256_file(paths["a1_r2_lock"]),
            "feature_schema_sha256": sha256_file(
                paths["feature_schema"]
            ),
            "normalization_sha256": sha256_file(
                paths["normalization"]
            ),
            "topology_sha256": sha256_file(paths["topology"]),
            "dataset_loader_sha256": sha256_file(
                paths["dataset_loader"]
            ),
        },
        "artifacts": {
            "contract_json": [
                str(contract_path),
                sha256_file(contract_path),
            ],
            "contract_markdown": [
                str(markdown_path),
                sha256_file(markdown_path),
            ],
            "derived_mask": (
                [str(mask_path), sha256_file(mask_path)]
                if mask_path.is_file()
                else None
            ),
        },
        "security_boundary": {
            "train_tensor_contents_accessed": (
                sampled_train_files > 0
            ),
            "validation_tensor_contents_accessed": (
                sampled_validation_files > 0
            ),
            "test_directory_existence_checked": False,
            "test_directory_enumerated": False,
            "test_tensor_files_opened": False,
            "test_tensor_bytes_read": False,
            "test_tensor_contents_accessed": False,
            "test_dataset_constructed": False,
            "test_windows_constructed": False,
        },
        "failures": failures,
        "warnings": warnings,
        "next_stage": (
            "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT"
            if not failures
            else None
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    if failures:
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures[:100]:
            print("FAIL:", failure)
        print("failure_count:", len(failures))
        print("warning_count:", len(warnings))
        return 1

    lock = {
        "status": COMPLETE,
        "decision": (
            "FREEZE_PRIMARY58_NORMALIZATION_AND_TOPOLOGY_MASK_CONTRACT"
        ),
        "report_sha256": sha256_file(report_path),
        "contract_file_sha256": sha256_file(contract_path),
        "contract_sha256": contract["contract_sha256"],
        "derived_mask_sha256": sha256_file(mask_path),
        "feature_count": FEATURE_COUNT,
        "primary58_count": len(PRIMARY58_INDICES),
        "primary58_indices": PRIMARY58_INDICES,
        "topology_mask_shape": [16, 10],
        "window": WINDOW,
        "stride": STRIDE,
        "second_normalization_forbidden": True,
        "reference_dataset_loader_authorized": False,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "next_stage": (
            "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT"
        ),
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-A2 FEATURE/NORMALIZATION/MASK CONTRACT =====")
    print("status: COMPLETE")
    print(
        "decision: "
        "FREEZE_PRIMARY58_NORMALIZATION_AND_TOPOLOGY_MASK_CONTRACT"
    )
    print("stored_feature_count:", FEATURE_COUNT)
    print("primary58_count:", len(PRIMARY58_INDICES))
    print("primary58_indices:", PRIMARY58_INDICES)
    print("stored_values_already_standardized: true")
    print("normalization_fit_split: train")
    print("second_normalization_forbidden: true")
    print("topology_mask_shape: [16, 10]")
    print("topology_mask_dtype: bool")
    print(
        "sampled_train_files:",
        sampled_train_files,
    )
    print(
        "sampled_validation_files:",
        sampled_validation_files,
    )
    print(
        "maximum_port_mask_recovery_error:",
        f"{maximum_mask_recovery_error:.10g}",
    )
    print("reference_dataset_loader_authorized: false")
    print("graph_message_passing: false")
    print("edge_index_model_input: false")
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print(
        "next_stage: "
        "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT"
    )
    print("contract_sha256:", contract["contract_sha256"])
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
