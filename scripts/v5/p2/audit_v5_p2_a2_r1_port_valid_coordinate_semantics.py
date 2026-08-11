#!/usr/bin/env python3
"""
V5 P2-A2-R1 Port-Valid Coordinate-Semantics Audit

The original A2 HOLD assumed router_coordinates were [row, col], with the
first coordinate increasing south and the second increasing east. This audit
tests all eight axis/sign conventions against inverse-transformed stored
port-valid channels 71..80.

It loads deterministic TRAIN and VALIDATION samples only and never enumerates
or opens runs/test.

Possible completion decision:
    RESOLVE_A2_HOLD_AS_COORDINATE_CONVENTION_ERROR_AND_AUTHORIZE_A2_R2

That decision requires:
- stored channels recover values near {0,1};
- channels are temporally static within each sampled run;
- input/output halves agree;
- exactly one coordinate convention gives zero Boolean mismatches across all
  sampled train and validation epochs.
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


STAGE = "V5_P2_A2_R1_PORT_VALID_COORDINATE_SEMANTICS_AUDIT"
COMPLETE = f"{STAGE}_COMPLETE"

PORT_INDICES = list(range(71, 81))
DIRECTIONS = ("local", "north", "east", "south", "west")
TOLERANCE = 1e-4


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
            path = by_name.get(f"{key}_{suffix}.pt")
            if path is not None:
                selected.append(path)
    return selected


def build_candidate_mask(
    coordinates: torch.Tensor,
    vertical_axis: int,
    horizontal_axis: int,
    north_is_low: bool,
    east_is_high: bool,
) -> torch.Tensor:
    vertical = coordinates[:, vertical_axis]
    horizontal = coordinates[:, horizontal_axis]

    vertical_min = int(vertical.min().item())
    vertical_max = int(vertical.max().item())
    horizontal_min = int(horizontal.min().item())
    horizontal_max = int(horizontal.max().item())

    local = torch.ones_like(vertical, dtype=torch.bool)

    if north_is_low:
        north = vertical > vertical_min
        south = vertical < vertical_max
    else:
        north = vertical < vertical_max
        south = vertical > vertical_min

    if east_is_high:
        east = horizontal < horizontal_max
        west = horizontal > horizontal_min
    else:
        east = horizontal > horizontal_min
        west = horizontal < horizontal_max

    one_side = torch.stack(
        (local, north, east, south, west),
        dim=1,
    )
    return torch.cat((one_side, one_side), dim=1)


def candidate_name(
    vertical_axis: int,
    horizontal_axis: int,
    north_is_low: bool,
    east_is_high: bool,
) -> str:
    vertical_label = f"coord[{vertical_axis}]"
    horizontal_label = f"coord[{horizontal_axis}]"
    north_label = "low" if north_is_low else "high"
    east_label = "high" if east_is_high else "low"
    return (
        f"vertical={vertical_label};horizontal={horizontal_label};"
        f"north={north_label};east={east_label}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root_input = args.root.expanduser()
    root = root_input.resolve()
    a2_dir = args.a2_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    paths = {
        "a2_report": (
            a2_dir
            / "V5_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT.json"
        ),
        "a2_hold": (
            a2_dir
            / "V5_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT_HOLD"
        ),
        "feature_schema": root / "feature_schema.json",
        "normalization": root / "normalization.pt",
        "topology": root / "topology.pt",
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    for split in ("train", "validation"):
        if not (root / "runs" / split).is_dir():
            failures.append(f"missing runs/{split}")

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

    a2_report = load_json(paths["a2_report"])
    if a2_report.get("status") != "HOLD":
        failures.append("historical A2 report status is not HOLD")

    historical_failures = a2_report.get("failures", [])
    if len(historical_failures) != 12:
        failures.append(
            f"historical A2 failure_count={len(historical_failures)}, "
            "expected 12"
        )
    unexpected_historical = [
        item
        for item in historical_failures
        if "stored port-valid channels do not recover topology mask"
        not in item
    ]
    if unexpected_historical:
        failures.append(
            "historical A2 HOLD contains unrelated failures: "
            f"{unexpected_historical[:10]}"
        )

    security = a2_report.get("security_boundary", {})
    if security.get("test_tensor_contents_accessed") is not False:
        failures.append("historical A2 does not certify untouched test tensors")

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

    features = schema.get("features")
    names = [
        record.get("name")
        for record in features
        if isinstance(record, dict)
    ]
    expected_port_names = [
        f"{side}_port_valid_{direction}"
        for side in ("input", "output")
        for direction in DIRECTIONS
    ]
    actual_port_names = names[71:81] if len(names) == 81 else []
    if actual_port_names != expected_port_names:
        failures.append(
            f"port feature names/order mismatch: {actual_port_names}"
        )

    mean = normalization.get("mean")
    std = normalization.get("std")
    if (
        not isinstance(mean, torch.Tensor)
        or tuple(mean.shape) != (81,)
        or not isinstance(std, torch.Tensor)
        or tuple(std.shape) != (81,)
        or not bool((std > 0).all().item())
    ):
        failures.append("normalization mean/std are invalid")

    coordinates = topology.get("router_coordinates")
    if (
        not isinstance(coordinates, torch.Tensor)
        or coordinates.dtype != torch.int64
        or tuple(coordinates.shape) != (16, 2)
    ):
        failures.append("router_coordinates is not int64 [16,2]")

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

    candidates: dict[str, torch.Tensor] = {}
    candidate_metadata: dict[str, dict[str, Any]] = {}

    for vertical_axis, horizontal_axis in ((0, 1), (1, 0)):
        for north_is_low in (True, False):
            for east_is_high in (True, False):
                name = candidate_name(
                    vertical_axis,
                    horizontal_axis,
                    north_is_low,
                    east_is_high,
                )
                candidates[name] = build_candidate_mask(
                    coordinates,
                    vertical_axis,
                    horizontal_axis,
                    north_is_low,
                    east_is_high,
                )
                candidate_metadata[name] = {
                    "vertical_axis": vertical_axis,
                    "horizontal_axis": horizontal_axis,
                    "north_is_low": north_is_low,
                    "east_is_high": east_is_high,
                }

    port_mean = mean[PORT_INDICES].view(1, 1, 10)
    port_std = std[PORT_INDICES].view(1, 1, 10)

    mismatch_counts = {name: 0 for name in candidates}
    compared_elements = 0

    sampled_records: list[dict[str, Any]] = []
    maximum_distance_to_binary = 0.0
    maximum_temporal_variation = 0.0
    maximum_input_output_difference = 0.0

    sampled_train_files = 0
    sampled_validation_files = 0

    for split in ("train", "validation"):
        sample_paths = select_sample_files(root / "runs" / split)
        if len(sample_paths) != 6:
            failures.append(
                f"{split} deterministic sample count={len(sample_paths)}, "
                "expected 6"
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
                failures.append(f"{split}/{path.name} payload is not dict")
                continue

            x = payload.get("x")
            if (
                not isinstance(x, torch.Tensor)
                or x.ndim != 3
                or tuple(x.shape[1:]) != (16, 81)
            ):
                failures.append(f"{split}/{path.name} x shape is invalid")
                continue
            if not bool(torch.isfinite(x).all().item()):
                failures.append(f"{split}/{path.name} x contains NaN/Inf")
                continue

            standardized = x[:, :, PORT_INDICES]
            pre_standardization = standardized * port_std + port_mean
            recovered = torch.expm1(pre_standardization)

            rounded = (recovered >= 0.5)
            nearest_binary = torch.minimum(
                recovered.abs(),
                (recovered - 1.0).abs(),
            )
            distance_to_binary = float(nearest_binary.max().item())

            temporal_min = recovered.amin(dim=0)
            temporal_max = recovered.amax(dim=0)
            temporal_variation = float(
                (temporal_max - temporal_min).max().item()
            )

            input_output_difference = float(
                (
                    recovered[:, :, :5]
                    - recovered[:, :, 5:]
                ).abs().max().item()
            )

            maximum_distance_to_binary = max(
                maximum_distance_to_binary,
                distance_to_binary,
            )
            maximum_temporal_variation = max(
                maximum_temporal_variation,
                temporal_variation,
            )
            maximum_input_output_difference = max(
                maximum_input_output_difference,
                input_output_difference,
            )

            file_candidate_mismatches = {}
            for name, mask in candidates.items():
                expanded = mask.view(1, 16, 10).expand(
                    x.shape[0],
                    -1,
                    -1,
                )
                mismatches = int((rounded != expanded).sum().item())
                mismatch_counts[name] += mismatches
                file_candidate_mismatches[name] = mismatches

            compared_elements += int(rounded.numel())
            sampled_records.append(
                {
                    "split": split,
                    "file": path.name,
                    "time_length": int(x.shape[0]),
                    "recovered_minimum": float(recovered.min().item()),
                    "recovered_maximum": float(recovered.max().item()),
                    "maximum_distance_to_nearest_binary": distance_to_binary,
                    "maximum_temporal_variation": temporal_variation,
                    "maximum_input_output_difference": (
                        input_output_difference
                    ),
                    "candidate_mismatch_counts": (
                        file_candidate_mismatches
                    ),
                }
            )

    candidate_scores = []
    for name, mismatches in mismatch_counts.items():
        candidate_scores.append(
            {
                "name": name,
                **candidate_metadata[name],
                "mismatch_count": mismatches,
                "compared_elements": compared_elements,
                "mismatch_rate": (
                    mismatches / compared_elements
                    if compared_elements
                    else None
                ),
                "mask": candidates[name].to(torch.uint8).tolist(),
            }
        )
    candidate_scores.sort(
        key=lambda item: (item["mismatch_count"], item["name"])
    )

    zero_error_candidates = [
        item
        for item in candidate_scores
        if item["mismatch_count"] == 0
    ]
    best_candidate = candidate_scores[0] if candidate_scores else None

    if maximum_distance_to_binary > TOLERANCE:
        failures.append(
            "inverse-transformed port-valid values are not binary; "
            f"max distance={maximum_distance_to_binary:.8g}"
        )
    if maximum_temporal_variation > TOLERANCE:
        failures.append(
            "port-valid values are not temporally static; "
            f"max variation={maximum_temporal_variation:.8g}"
        )
    if maximum_input_output_difference > TOLERANCE:
        failures.append(
            "input/output port-valid halves differ; "
            f"max difference={maximum_input_output_difference:.8g}"
        )
    if len(zero_error_candidates) != 1:
        failures.append(
            f"zero-error coordinate candidate count="
            f"{len(zero_error_candidates)}, expected 1"
        )

    if failures:
        status = "HOLD"
        decision = "REQUIRE_MANUAL_PORT_VALID_SEMANTIC_REVIEW"
        next_stage = None
    else:
        status = "COMPLETE"
        decision = (
            "RESOLVE_A2_HOLD_AS_COORDINATE_CONVENTION_ERROR_"
            "AND_AUTHORIZE_A2_R2"
        )
        next_stage = (
            "V5_P2_A2_R2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT"
        )

    report = {
        "stage": STAGE,
        "status": status,
        "decision": decision,
        "dataset": {
            "link_path": str(root_input.absolute()),
            "resolved_root": str(root),
        },
        "historical_a2_hold": {
            "report_sha256": sha256_file(paths["a2_report"]),
            "hold_marker_sha256": sha256_file(paths["a2_hold"]),
            "failure_count": len(historical_failures),
            "failure_class": (
                "assumed_coordinate_convention_mask_mismatch"
            ),
        },
        "sample_policy": {
            "splits": ["train", "validation"],
            "pair_positions": ["first", "middle", "last"],
            "members": ["ATTACK", "CONTROL"],
            "sampled_train_files": sampled_train_files,
            "sampled_validation_files": sampled_validation_files,
        },
        "recovered_port_valid_semantics": {
            "feature_indices": PORT_INDICES,
            "feature_names": actual_port_names,
            "inverse_transform": (
                "expm1(standardized*std + mean)"
            ),
            "maximum_distance_to_nearest_binary": (
                maximum_distance_to_binary
            ),
            "maximum_temporal_variation": (
                maximum_temporal_variation
            ),
            "maximum_input_output_difference": (
                maximum_input_output_difference
            ),
            "tolerance": TOLERANCE,
        },
        "candidate_scores": candidate_scores,
        "zero_error_candidate_count": len(zero_error_candidates),
        "selected_candidate": (
            zero_error_candidates[0]
            if len(zero_error_candidates) == 1
            else best_candidate
        ),
        "sample_records": sampled_records,
        "security_boundary": {
            "train_tensor_contents_accessed": sampled_train_files > 0,
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
        "next_stage": next_stage,
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    if failures:
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print("===== V5 P2-A2-R1 PORT-VALID SEMANTICS AUDIT =====")
        print("status: HOLD")
        if best_candidate is not None:
            print("best_candidate:", best_candidate["name"])
            print(
                "best_candidate_mismatch_count:",
                best_candidate["mismatch_count"],
            )
            print(
                "best_candidate_mismatch_rate:",
                best_candidate["mismatch_rate"],
            )
        print(
            "maximum_distance_to_nearest_binary:",
            maximum_distance_to_binary,
        )
        print(
            "maximum_temporal_variation:",
            maximum_temporal_variation,
        )
        print(
            "maximum_input_output_difference:",
            maximum_input_output_difference,
        )
        print("failure_count:", len(failures))
        for failure in failures:
            print("FAIL:", failure)
        print("warning_count:", len(warnings))
        print(f"{STAGE}_HOLD")
        return 1

    selected = zero_error_candidates[0]
    lock = {
        "status": COMPLETE,
        "decision": decision,
        "report_sha256": sha256_file(report_path),
        "selected_coordinate_convention": {
            key: selected[key]
            for key in (
                "name",
                "vertical_axis",
                "horizontal_axis",
                "north_is_low",
                "east_is_high",
            )
        },
        "selected_mask": selected["mask"],
        "maximum_distance_to_nearest_binary": (
            maximum_distance_to_binary
        ),
        "maximum_temporal_variation": maximum_temporal_variation,
        "maximum_input_output_difference": (
            maximum_input_output_difference
        ),
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "next_stage": next_stage,
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-A2-R1 PORT-VALID SEMANTICS AUDIT =====")
    print("status: COMPLETE")
    print("decision:", decision)
    print("sampled_train_files:", sampled_train_files)
    print("sampled_validation_files:", sampled_validation_files)
    print("compared_elements:", compared_elements)
    print(
        "maximum_distance_to_nearest_binary:",
        f"{maximum_distance_to_binary:.10g}",
    )
    print(
        "maximum_temporal_variation:",
        f"{maximum_temporal_variation:.10g}",
    )
    print(
        "maximum_input_output_difference:",
        f"{maximum_input_output_difference:.10g}",
    )
    print("zero_error_candidate_count:", len(zero_error_candidates))
    print("selected_candidate:", selected["name"])
    print("vertical_axis:", selected["vertical_axis"])
    print("horizontal_axis:", selected["horizontal_axis"])
    print("north_is_low:", selected["north_is_low"])
    print("east_is_high:", selected["east_is_high"])
    print("selected_mismatch_count:", selected["mismatch_count"])
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("next_stage:", next_stage)
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
