#!/usr/bin/env python3
"""
V5 P2-A2-R2 Feature, Normalization, and Physical-Port Mask Contract

This stage resolves the historical A2 HOLD using the unique coordinate
convention proven by A2-R1:

    vertical axis   = router_coordinates[:, 0]
    horizontal axis = router_coordinates[:, 1]
    moving north    = increasing vertical coordinate
    moving east     = increasing horizontal coordinate

It freezes:
- the exact PRIMARY58 feature selection;
- no second normalization;
- the topology-derived Boolean [16,10] physical-port mask;
- the frozen B3 input interface;
- quarantine of all metadata/provenance;
- rejection of the supplied reference loader for P2 training.

It reads support artifacts and prior reports only. It does not deserialize any
run tensor and does not enumerate or open runs/test.
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


STAGE = "V5_P2_A2_R2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT"
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
NONPRIMARY_DYNAMIC_INDICES = (
    list(range(56, 62))
    + [63]
    + list(range(66, 71))
)
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

EXPECTED_SCHEMA_FORBIDDEN = {
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

MODEL_INPUT_PROHIBITIONS = [
    "epoch_id",
    "router_id",
    "tick fields",
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
    "edge_index for frozen B3",
    "stored standardized port-valid channels 71-80",
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


def corrected_port_mask(coordinates: torch.Tensor) -> torch.Tensor:
    """
    Return Boolean [16,10] in exact order:
      input : local,north,east,south,west
      output: local,north,east,south,west

    A2-R1 proved:
      vertical   = coord[0]
      horizontal = coord[1]
      north      = increasing coord[0]
      east       = increasing coord[1]
    """
    vertical = coordinates[:, 0].to(torch.int64)
    horizontal = coordinates[:, 1].to(torch.int64)

    vertical_min = int(vertical.min().item())
    vertical_max = int(vertical.max().item())
    horizontal_min = int(horizontal.min().item())
    horizontal_max = int(horizontal.max().item())

    local = torch.ones_like(vertical, dtype=torch.bool)
    north = vertical < vertical_max
    east = horizontal < horizontal_max
    south = vertical > vertical_min
    west = horizontal > horizontal_min

    one_side = torch.stack(
        (local, north, east, south, west),
        dim=1,
    )
    return torch.cat((one_side, one_side), dim=1)


def expected_directed_edges(
    coordinates: torch.Tensor,
) -> set[tuple[int, int]]:
    coordinate_rows = [
        (int(row[0]), int(row[1]))
        for row in coordinates.tolist()
    ]
    node_by_coordinate = {
        coordinate: node
        for node, coordinate in enumerate(coordinate_rows)
    }

    edges: set[tuple[int, int]] = set()
    for source, (vertical, horizontal) in enumerate(coordinate_rows):
        for delta_vertical, delta_horizontal in (
            (+1, 0),   # north
            (0, +1),   # east
            (-1, 0),   # south
            (0, -1),   # west
        ):
            target = node_by_coordinate.get(
                (
                    vertical + delta_vertical,
                    horizontal + delta_horizontal,
                )
            )
            if target is not None:
                edges.add((source, target))
    return edges


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a0-dir", type=Path, required=True)
    parser.add_argument("--a1-r2-dir", type=Path, required=True)
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--a2-r1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root_input = args.root.expanduser()
    root = root_input.resolve()
    a0_dir = args.a0_dir.expanduser().resolve()
    a1_r2_dir = args.a1_r2_dir.expanduser().resolve()
    a2_dir = args.a2_dir.expanduser().resolve()
    a2_r1_dir = args.a2_r1_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(
            f"STOP: output already exists: {output_dir}",
            file=sys.stderr,
        )
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
        "a1_r2_report": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT.json"
        ),
        "a1_r2_lock": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT_LOCK.json"
        ),
        "a2_report": (
            a2_dir
            / "V5_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT.json"
        ),
        "a2_hold": (
            a2_dir
            / "V5_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT_HOLD"
        ),
        "a2_r1_report": (
            a2_r1_dir
            / "V5_P2_A2_R1_PORT_VALID_COORDINATE_SEMANTICS_AUDIT.json"
        ),
        "a2_r1_lock": (
            a2_r1_dir
            / "V5_P2_A2_R1_PORT_VALID_COORDINATE_SEMANTICS_AUDIT_LOCK.json"
        ),
        "a2_r1_marker": (
            a2_r1_dir
            / "V5_P2_A2_R1_PORT_VALID_COORDINATE_SEMANTICS_AUDIT_COMPLETE"
        ),
        "feature_schema": root / "feature_schema.json",
        "normalization": root / "normalization.pt",
        "topology": root / "topology.pt",
        "dataset_loader": root / "dataset_loader.py",
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "run_tensors_deserialized": False,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        return 1

    a0_report = load_json(paths["a0_report"])
    a0_lock = load_json(paths["a0_lock"])
    a1_r2_report = load_json(paths["a1_r2_report"])
    a1_r2_lock = load_json(paths["a1_r2_lock"])
    a2_report = load_json(paths["a2_report"])
    a2_r1_report = load_json(paths["a2_r1_report"])
    a2_r1_lock = load_json(paths["a2_r1_lock"])

    if a0_report.get("status") != "COMPLETE":
        failures.append("A0 status is not COMPLETE")
    if a1_r2_report.get("status") != "COMPLETE":
        failures.append("A1-R2 status is not COMPLETE")
    if a2_report.get("status") != "HOLD":
        failures.append("historical A2 status is not HOLD")
    if a2_r1_report.get("status") != "COMPLETE":
        failures.append("A2-R1 status is not COMPLETE")

    if (
        a0_lock.get("report_sha256")
        != sha256_file(paths["a0_report"])
    ):
        failures.append("A0 report SHA mismatch")
    if (
        a1_r2_lock.get("report_sha256")
        != sha256_file(paths["a1_r2_report"])
    ):
        failures.append("A1-R2 report SHA mismatch")
    if (
        a2_r1_lock.get("report_sha256")
        != sha256_file(paths["a2_r1_report"])
    ):
        failures.append("A2-R1 report SHA mismatch")

    if (
        a1_r2_lock.get("decision")
        != "FREEZE_PAIR_ALIGNED_WINDOW_CONTRACT_AND_AUTHORIZE_A2"
    ):
        failures.append("A1-R2 did not authorize A2")
    if a1_r2_lock.get("window") != WINDOW:
        failures.append("A1-R2 window is not 32")
    if a1_r2_lock.get("stride") != STRIDE:
        failures.append("A1-R2 stride is not 8")

    expected_r1_decision = (
        "RESOLVE_A2_HOLD_AS_COORDINATE_CONVENTION_ERROR_"
        "AND_AUTHORIZE_A2_R2"
    )
    if a2_r1_lock.get("decision") != expected_r1_decision:
        failures.append("A2-R1 did not authorize A2-R2")

    selected = a2_r1_lock.get("selected_coordinate_convention", {})
    expected_selected = {
        "name": (
            "vertical=coord[0];horizontal=coord[1];"
            "north=high;east=high"
        ),
        "vertical_axis": 0,
        "horizontal_axis": 1,
        "north_is_low": False,
        "east_is_high": True,
    }
    if selected != expected_selected:
        failures.append(
            f"unexpected A2-R1 coordinate convention: {selected}"
        )

    if a2_r1_lock.get("maximum_distance_to_nearest_binary") != 0.0:
        failures.append("A2-R1 recovered values were not exactly binary")
    if a2_r1_lock.get("maximum_temporal_variation") != 0.0:
        failures.append("A2-R1 recovered mask was not temporally static")
    if a2_r1_lock.get("maximum_input_output_difference") != 0.0:
        failures.append("A2-R1 input/output validity halves differed")

    historical_failures = a2_report.get("failures", [])
    if len(historical_failures) != 12:
        failures.append(
            f"historical A2 failure count={len(historical_failures)}, "
            "expected 12"
        )
    unrelated_historical_failures = [
        item
        for item in historical_failures
        if "stored port-valid channels do not recover topology mask"
        not in item
    ]
    if unrelated_historical_failures:
        failures.append(
            "historical A2 HOLD contains unrelated failures"
        )

    for label, document in (
        ("A0", a0_report),
        ("A1-R2", a1_r2_report),
        ("A2", a2_report),
        ("A2-R1", a2_r1_report),
    ):
        security = document.get("security_boundary", {})
        if security.get("test_tensor_contents_accessed") is not False:
            failures.append(
                f"{label} does not certify untouched test tensors"
            )

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "run_tensors_deserialized": False,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
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
    loader_source = paths["dataset_loader"].read_text(
        encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Exact feature schema
    # ------------------------------------------------------------------
    if schema.get("schema_version") != 1:
        failures.append("feature schema version is not 1")
    if schema.get("feature_count") != FEATURE_COUNT:
        failures.append("feature count is not 81")
    if schema.get("input_tensor_order") != [
        "time",
        "router",
        "feature",
    ]:
        failures.append("stored tensor order is not [time,router,feature]")

    feature_records = schema.get("features")
    if (
        not isinstance(feature_records, list)
        or len(feature_records) != FEATURE_COUNT
    ):
        failures.append("feature schema does not contain 81 records")
        feature_names: list[str] = []
    else:
        feature_names = []
        for index, record in enumerate(feature_records):
            if not isinstance(record, dict):
                failures.append(
                    f"feature record {index} is not a dictionary"
                )
                feature_names.append(None)
                continue
            feature_names.append(record.get("name"))
            if record.get("index") != index:
                failures.append(
                    f"feature record {index} stores index "
                    f"{record.get('index')!r}"
                )
            if (
                record.get("transform")
                != "log1p_then_train_standardize"
            ):
                failures.append(
                    f"feature {index} transform mismatch"
                )

        if feature_names != EXPECTED_FEATURE_NAMES:
            failures.append(
                "feature names/order do not match frozen 81-feature schema"
            )

    forbidden_schema = set(
        schema.get("forbidden_model_inputs", [])
    )
    missing_forbidden = sorted(
        EXPECTED_SCHEMA_FORBIDDEN - forbidden_schema
    )
    if missing_forbidden:
        failures.append(
            f"feature schema omits forbidden inputs: {missing_forbidden}"
        )
    if schema.get("excluded_observation_columns") != [
        "tick",
        "epoch_id",
        "router_id",
    ]:
        failures.append("excluded observation columns mismatch")

    # ------------------------------------------------------------------
    # Exact normalization provenance
    # ------------------------------------------------------------------
    if not isinstance(normalization, dict):
        failures.append("normalization payload is not a dictionary")
    else:
        mean = normalization.get("mean")
        std = normalization.get("std")
        mask_features = normalization.get("mask_features")

        if normalization.get("feature_names") != EXPECTED_FEATURE_NAMES:
            failures.append(
                "normalization feature names/order mismatch"
            )
        if (
            not isinstance(mean, torch.Tensor)
            or tuple(mean.shape) != (81,)
            or not mean.is_floating_point()
            or not bool(torch.isfinite(mean).all().item())
        ):
            failures.append("normalization mean is invalid")
        if (
            not isinstance(std, torch.Tensor)
            or tuple(std.shape) != (81,)
            or not std.is_floating_point()
            or not bool(torch.isfinite(std).all().item())
            or not bool((std > 0).all().item())
        ):
            failures.append("normalization std is invalid")
        if (
            not isinstance(mask_features, torch.Tensor)
            or tuple(mask_features.shape) != (81,)
            or mask_features.dtype != torch.bool
        ):
            failures.append("normalization mask_features is invalid")
        elif bool(mask_features.any().item()):
            failures.append(
                "normalization unexpectedly marks stored mask features"
            )
        if (
            normalization.get("dynamic_transform")
            != "log1p_then_standardize"
        ):
            failures.append("dynamic transform mismatch")
        if normalization.get("mask_transform") != "identity":
            failures.append("mask transform mismatch")
        if normalization.get("fit_split") != "train":
            failures.append("normalization was not fit on train")
        if (
            not isinstance(
                normalization.get("fit_router_epoch_rows"),
                int,
            )
            or normalization["fit_router_epoch_rows"] <= 0
        ):
            failures.append("fit_router_epoch_rows is invalid")

    # ------------------------------------------------------------------
    # Corrected topology and mask
    # ------------------------------------------------------------------
    if not isinstance(topology, dict):
        failures.append("topology payload is not a dictionary")
        edge_index = None
        coordinates = None
        port_mask = None
    else:
        edge_index = topology.get("edge_index")
        coordinates = topology.get("router_coordinates")

        expected_scalars = {
            "num_nodes": 16,
            "mesh_rows": 4,
            "mesh_cols": 4,
            "directed_edge_count": 48,
            "undirected_edge_count": 24,
        }
        for key, expected_value in expected_scalars.items():
            if topology.get(key) != expected_value:
                failures.append(
                    f"topology {key}={topology.get(key)!r}, "
                    f"expected {expected_value}"
                )

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

        port_mask = None
        if (
            isinstance(edge_index, torch.Tensor)
            and tuple(edge_index.shape) == (2, 48)
            and isinstance(coordinates, torch.Tensor)
            and tuple(coordinates.shape) == (16, 2)
        ):
            coordinate_set = {
                (int(item[0]), int(item[1]))
                for item in coordinates.tolist()
            }
            expected_coordinate_set = {
                (vertical, horizontal)
                for vertical in range(4)
                for horizontal in range(4)
            }
            if coordinate_set != expected_coordinate_set:
                failures.append(
                    "router coordinates do not cover a 4x4 grid"
                )

            actual_edges = {
                (int(source), int(target))
                for source, target in edge_index.t().tolist()
            }
            expected_edges = expected_directed_edges(coordinates)

            if actual_edges != expected_edges:
                failures.append(
                    "edge_index does not match coordinate-derived "
                    "directed 4-neighbor mesh"
                )
            if any(source == target for source, target in actual_edges):
                failures.append("edge_index contains self loops")
            if any(
                (target, source) not in actual_edges
                for source, target in actual_edges
            ):
                failures.append("edge_index is not symmetric")

            port_mask = corrected_port_mask(coordinates)
            if tuple(port_mask.shape) != (16, 10):
                failures.append(
                    "corrected physical-port mask is not [16,10]"
                )
            if port_mask.dtype != torch.bool:
                failures.append(
                    "corrected physical-port mask is not Boolean"
                )
            if int(port_mask.sum().item()) != 128:
                failures.append(
                    f"corrected mask true count="
                    f"{int(port_mask.sum().item())}, expected 128"
                )

            selected_mask = torch.tensor(
                a2_r1_lock.get("selected_mask"),
                dtype=torch.uint8,
            ).to(torch.bool)
            if tuple(selected_mask.shape) != (16, 10):
                failures.append(
                    "A2-R1 selected mask is not [16,10]"
                )
            elif not torch.equal(port_mask, selected_mask):
                failures.append(
                    "reconstructed corrected mask differs from "
                    "A2-R1 zero-error mask"
                )

    # ------------------------------------------------------------------
    # Reference loader is not authorized
    # ------------------------------------------------------------------
    reference_loader_findings = {
        "native_per_run_windowing": (
            "range(0, length - self.window + 1, self.stride)"
            in loader_source
        ),
        "returns_unsliced_all81": (
            'run["x"][start:stop].permute(1, 2, 0)'
            in loader_source
        ),
        "returns_edge_index": '"edge_index"' in loader_source,
        "returns_epoch_id": '"epoch_id"' in loader_source,
        "returns_case_id": '"case_id"' in loader_source,
        "returns_run_id": '"run_id"' in loader_source,
        "returns_mode": '"mode"' in loader_source,
        "returns_window_start": '"window_start"' in loader_source,
        "returns_window_target": '"window_target"' in loader_source,
    }
    if not all(reference_loader_findings.values()):
        failures.append(
            "reference-loader safety findings changed unexpectedly"
        )

    selected_feature_names = [
        EXPECTED_FEATURE_NAMES[index]
        for index in PRIMARY58_INDICES
    ]
    excluded_indices = sorted(
        set(range(FEATURE_COUNT)) - set(PRIMARY58_INDICES)
    )
    excluded_names = [
        EXPECTED_FEATURE_NAMES[index]
        for index in excluded_indices
    ]

    mask_channel_order = [
        f"{side}_{direction}"
        for side in ("input", "output")
        for direction in DIRECTIONS
    ]

    contract = {
        "contract_name": (
            "V5_P2_PRIMARY58_PLUS_CORRECTED_TOPOLOGY_BOOLEAN_PORT_MASK"
        ),
        "contract_version": 2,
        "historical_resolution": {
            "historical_a2_status": "HOLD",
            "historical_failure_count": 12,
            "historical_failure_class": (
                "incorrect coordinate-direction assumption"
            ),
            "a2_r1_resolution": expected_r1_decision,
            "unique_zero_error_coordinate_convention": (
                expected_selected
            ),
            "maximum_distance_to_nearest_binary": 0.0,
            "maximum_temporal_variation": 0.0,
            "maximum_input_output_difference": 0.0,
        },
        "architecture_interface": {
            "frozen_architecture": (
                "B3_CAUSAL_DEPTHWISE_SEPARABLE_CONV1D_ONLY"
            ),
            "graph_message_passing": False,
            "temporal_input_shape": "[B,16,58,32]",
            "topology_mask_shape": "[B,16,10]",
            "edge_index_model_input": False,
            "router_coordinates_model_input": False,
        },
        "stored_tensor": {
            "shape": "[T,16,81]",
            "axis_order": ["time", "router", "feature"],
            "feature_count": 81,
            "already_transformed": True,
            "transform": "log1p_then_train_standardize",
            "normalization_fit_split": "train",
            "second_normalization_forbidden": True,
            "inverse_transform_of_primary58_for_training_forbidden": True,
        },
        "primary58": {
            "indices": PRIMARY58_INDICES,
            "names": selected_feature_names,
            "count": 58,
            "selection_basis": (
                "exact frozen P0 indices and feature-name order"
            ),
        },
        "excluded_stored_features": {
            "indices": excluded_indices,
            "names": excluded_names,
            "status_flag_indices": STATUS_FLAG_INDICES,
            "nonprimary_dynamic_indices": (
                NONPRIMARY_DYNAMIC_INDICES
            ),
            "stored_port_valid_indices": (
                STORED_PORT_VALID_INDICES
            ),
            "stored_standardized_port_valid_as_model_input": False,
        },
        "physical_port_mask": {
            "source": "topology.pt only",
            "shape": [16, 10],
            "stored_dtype": "bool",
            "model_cast_dtype": "float32",
            "channel_order": mask_channel_order,
            "coordinate_semantics": {
                "vertical_axis": "router_coordinates[:,0]",
                "horizontal_axis": "router_coordinates[:,1]",
                "moving_north": (
                    "increases vertical coordinate"
                ),
                "moving_east": (
                    "increases horizontal coordinate"
                ),
                "north_boundary": "maximum vertical coordinate",
                "east_boundary": "maximum horizontal coordinate",
            },
            "validity_rules": {
                "local": "always true",
                "north": "vertical < vertical_max",
                "east": "horizontal < horizontal_max",
                "south": "vertical > vertical_min",
                "west": "horizontal > horizontal_min",
                "input_output_halves_identical": True,
            },
            "true_entry_count": (
                int(port_mask.sum().item())
                if isinstance(port_mask, torch.Tensor)
                else None
            ),
            "stored_channels_71_80_role": (
                "audit-only semantic corroboration"
            ),
        },
        "windowing_dependency": {
            "required_contract": (
                "V5_P2_PAIR_ALIGNED_COMMON_PREFIX_WINDOWS"
            ),
            "window": WINDOW,
            "stride": STRIDE,
            "identical_window_starts_per_pair": True,
        },
        "reference_dataset_loader": {
            "authorized": False,
            "findings": reference_loader_findings,
            "replacement_required": True,
        },
        "model_input_prohibitions": MODEL_INPUT_PROHIBITIONS,
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

    contract_path = (
        output_dir
        / "V5_P2_A2_R2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT.json"
    )
    write_json(contract_path, contract)

    mask_path = (
        output_dir
        / "V5_P2_A2_R2_TOPOLOGY_DERIVED_BOOLEAN_PORT_MASK.pt"
    )
    if isinstance(port_mask, torch.Tensor):
        torch.save(
            {
                "mask": port_mask.cpu(),
                "channel_order": mask_channel_order,
                "coordinate_semantics": (
                    contract["physical_port_mask"][
                        "coordinate_semantics"
                    ]
                ),
                "source_topology_sha256": sha256_file(
                    paths["topology"]
                ),
                "a2_r1_lock_sha256": sha256_file(
                    paths["a2_r1_lock"]
                ),
                "contract_sha256": contract["contract_sha256"],
            },
            mask_path,
        )

    markdown = f"""# V5 P2 A2-R2 Feature, Normalization, and Mask Contract

## Historical resolution

The original A2 HOLD was caused by the wrong directional interpretation of
`router_coordinates`, not by corrupted port-valid channels.

A2-R1 proved a unique zero-error convention:

```text
vertical axis   = coord[0]
horizontal axis = coord[1]
north           = increasing coord[0]
east            = increasing coord[1]
```

Across 1,355,200 compared values:

```text
distance to nearest Boolean = 0
temporal variation          = 0
input/output difference     = 0
mask mismatch count         = 0
```

## Frozen learned-input contract

```text
stored x            = [T,16,81]
selected PRIMARY58  = 0-29, 31-55, 62, 64, 65
window              = 32
model temporal x    = [B,16,58,32]
topology mask       = Boolean [B,16,10]
```

The stored tensor is already `log1p` transformed and standardized using
train-only statistics. A second normalization pass is forbidden.

## Correct physical-port rules

```text
local = true
north = vertical < vertical_max
east  = horizontal < horizontal_max
south = vertical > vertical_min
west  = horizontal > horizontal_min
```

The same five channels are used for input and output physical validity.

## Frozen B3

B3 has no graph message passing. `edge_index` and router coordinates remain
audit/bookkeeping data and are not model inputs.

## Loader requirement

The supplied reference loader is not authorized. A replacement must enforce:

- A1-R2 pair-aligned common-prefix windows;
- PRIMARY58 selection;
- corrected topology-derived Boolean mask;
- no second normalization;
- no metadata or provenance in returned model inputs.

## Contract SHA-256

`{contract["contract_sha256"]}`

## Test boundary

No P2 test directory was enumerated and no test tensor was opened.
"""
    markdown_path = (
        output_dir
        / "V5_P2_A2_R2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT.md"
    )
    atomic_write(markdown_path, markdown)

    status = "COMPLETE" if not failures else "HOLD"
    report = {
        "stage": STAGE,
        "status": status,
        "decision": (
            "FREEZE_CORRECTED_PRIMARY58_NORMALIZATION_AND_MASK_"
            "CONTRACT_AND_AUTHORIZE_A3"
            if not failures
            else "BLOCK_P2_A3"
        ),
        "dataset": {
            "link_path": str(root_input.absolute()),
            "resolved_root": str(root),
        },
        "historical_resolution": (
            contract["historical_resolution"]
        ),
        "feature_audit": {
            "stored_feature_count": FEATURE_COUNT,
            "primary58_count": len(PRIMARY58_INDICES),
            "primary58_indices": PRIMARY58_INDICES,
            "primary58_names": selected_feature_names,
            "excluded_indices": excluded_indices,
            "excluded_names": excluded_names,
        },
        "normalization_audit": {
            "dynamic_transform": normalization.get(
                "dynamic_transform"
            ),
            "fit_split": normalization.get("fit_split"),
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
            "corrected_mask_shape": (
                list(port_mask.shape)
                if isinstance(port_mask, torch.Tensor)
                else None
            ),
            "corrected_mask_true_count": (
                int(port_mask.sum().item())
                if isinstance(port_mask, torch.Tensor)
                else None
            ),
            "coordinate_convention": expected_selected,
        },
        "reference_loader_findings": (
            reference_loader_findings
        ),
        "contract": contract,
        "provenance": {
            "a0_report_sha256": sha256_file(
                paths["a0_report"]
            ),
            "a0_lock_sha256": sha256_file(paths["a0_lock"]),
            "a1_r2_report_sha256": sha256_file(
                paths["a1_r2_report"]
            ),
            "a1_r2_lock_sha256": sha256_file(
                paths["a1_r2_lock"]
            ),
            "historical_a2_report_sha256": sha256_file(
                paths["a2_report"]
            ),
            "historical_a2_hold_sha256": sha256_file(
                paths["a2_hold"]
            ),
            "a2_r1_report_sha256": sha256_file(
                paths["a2_r1_report"]
            ),
            "a2_r1_lock_sha256": sha256_file(
                paths["a2_r1_lock"]
            ),
            "feature_schema_sha256": sha256_file(
                paths["feature_schema"]
            ),
            "normalization_sha256": sha256_file(
                paths["normalization"]
            ),
            "topology_sha256": sha256_file(
                paths["topology"]
            ),
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
            "corrected_port_mask": (
                [str(mask_path), sha256_file(mask_path)]
                if mask_path.is_file()
                else None
            ),
        },
        "security_boundary": {
            "run_tensors_deserialized": False,
            "train_tensor_contents_accessed": False,
            "validation_tensor_contents_accessed": False,
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
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures[:100]:
            print("FAIL:", failure)
        print("failure_count:", len(failures))
        print("warning_count:", len(warnings))
        return 1

    lock = {
        "status": COMPLETE,
        "decision": (
            "FREEZE_CORRECTED_PRIMARY58_NORMALIZATION_AND_MASK_"
            "CONTRACT_AND_AUTHORIZE_A3"
        ),
        "report_sha256": sha256_file(report_path),
        "contract_file_sha256": sha256_file(contract_path),
        "contract_sha256": contract["contract_sha256"],
        "corrected_port_mask_sha256": sha256_file(mask_path),
        "stored_feature_count": FEATURE_COUNT,
        "primary58_count": 58,
        "primary58_indices": PRIMARY58_INDICES,
        "window": WINDOW,
        "stride": STRIDE,
        "topology_mask_shape": [16, 10],
        "topology_mask_true_count": int(port_mask.sum().item()),
        "coordinate_convention": expected_selected,
        "stored_values_already_standardized": True,
        "second_normalization_forbidden": True,
        "reference_dataset_loader_authorized": False,
        "graph_message_passing": False,
        "edge_index_model_input": False,
        "run_tensors_deserialized": False,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "next_stage": (
            "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT"
        ),
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-A2-R2 FEATURE/NORMALIZATION/MASK CONTRACT =====")
    print("status: COMPLETE")
    print(
        "decision: "
        "FREEZE_CORRECTED_PRIMARY58_NORMALIZATION_AND_MASK_"
        "CONTRACT_AND_AUTHORIZE_A3"
    )
    print(
        "historical_a2_resolution: "
        "coordinate_convention_error"
    )
    print(
        "coordinate_convention: "
        "vertical=coord[0], horizontal=coord[1], "
        "north=high, east=high"
    )
    print("stored_feature_count:", FEATURE_COUNT)
    print("primary58_count:", len(PRIMARY58_INDICES))
    print("primary58_indices:", PRIMARY58_INDICES)
    print("stored_values_already_standardized: true")
    print("normalization_fit_split: train")
    print("second_normalization_forbidden: true")
    print("topology_mask_shape: [16, 10]")
    print("topology_mask_dtype: bool")
    print("topology_mask_true_count:", int(port_mask.sum().item()))
    print("reference_dataset_loader_authorized: false")
    print("graph_message_passing: false")
    print("edge_index_model_input: false")
    print("run_tensors_deserialized: false")
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
