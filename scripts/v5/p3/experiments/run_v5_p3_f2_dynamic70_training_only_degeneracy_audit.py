from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = (
    "V5_P3_F2_DYNAMIC70_TRAINING_ONLY_DEGENERACY_"
    "AND_OBSERVABILITY_AUDIT"
)
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
ZERO_TOLERANCE = 1e-12
CONSTANT_TOLERANCE = 1e-12
NEAR_CONSTANT_ACTIVE_FRACTION = 0.001
RARE_SAMPLE_FRACTION = 0.01
QUANTILE_SAMPLE_ITEMS = 128
PROGRESS_INTERVAL = 1000


@dataclass
class RunningStats:
    count: np.ndarray
    total: np.ndarray
    total_sq: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    zero_count: np.ndarray
    nonfinite_count: np.ndarray

    @classmethod
    def create(cls, channels: int) -> "RunningStats":
        return cls(
            count=np.zeros(channels, dtype=np.int64),
            total=np.zeros(channels, dtype=np.float64),
            total_sq=np.zeros(channels, dtype=np.float64),
            minimum=np.full(channels, np.inf, dtype=np.float64),
            maximum=np.full(channels, -np.inf, dtype=np.float64),
            zero_count=np.zeros(channels, dtype=np.int64),
            nonfinite_count=np.zeros(channels, dtype=np.int64),
        )

    def update(self, values: np.ndarray, valid: np.ndarray) -> None:
        # values/valid: [channels, routers, epochs]
        finite = np.isfinite(values)
        selected = valid & finite
        self.nonfinite_count += np.sum(valid & ~finite, axis=(1, 2))
        self.count += np.sum(selected, axis=(1, 2))
        safe = np.where(selected, values, 0.0)
        self.total += np.sum(safe, axis=(1, 2), dtype=np.float64)
        self.total_sq += np.sum(
            safe * safe,
            axis=(1, 2),
            dtype=np.float64,
        )
        self.zero_count += np.sum(
            selected & (np.abs(values) <= ZERO_TOLERANCE),
            axis=(1, 2),
        )
        local_min = np.min(
            np.where(selected, values, np.inf),
            axis=(1, 2),
        )
        local_max = np.max(
            np.where(selected, values, -np.inf),
            axis=(1, 2),
        )
        self.minimum = np.minimum(self.minimum, local_min)
        self.maximum = np.maximum(self.maximum, local_max)

    def finalize(self) -> dict[str, np.ndarray]:
        mean = np.divide(
            self.total,
            self.count,
            out=np.full_like(self.total, np.nan),
            where=self.count > 0,
        )
        second = np.divide(
            self.total_sq,
            self.count,
            out=np.full_like(self.total_sq, np.nan),
            where=self.count > 0,
        )
        variance = np.maximum(second - mean * mean, 0.0)
        std = np.sqrt(variance)
        zero_fraction = np.divide(
            self.zero_count,
            self.count,
            out=np.full(self.count.shape, np.nan, dtype=np.float64),
            where=self.count > 0,
        )
        minimum = self.minimum.copy()
        maximum = self.maximum.copy()
        minimum[~np.isfinite(minimum)] = np.nan
        maximum[~np.isfinite(maximum)] = np.nan
        return {
            "count": self.count,
            "mean": mean,
            "std": std,
            "minimum": minimum,
            "maximum": maximum,
            "range": maximum - minimum,
            "zero_fraction": zero_fraction,
            "nonfinite_count": self.nonfinite_count,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def import_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def scalar_value(value: Any) -> Any:
    array = tensor_to_numpy(value)
    if array.size != 1:
        return value
    result = array.reshape(-1)[0]
    if hasattr(result, "item"):
        result = result.item()
    return result


def extract_first(sample: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in sample:
            return sample[key]
    raise KeyError(f"none of the required keys exist: {keys}")


def extract_attack_label(sample: dict[str, Any]) -> int:
    return int(
        scalar_value(
            extract_first(
                sample,
                (
                    "y_attack",
                    "y_graph",
                    "attack_label",
                    "graph_label",
                ),
            )
        )
    )


def extract_count_label(sample: dict[str, Any]) -> int:
    for key in (
        "y_attacker_count",
        "attacker_count",
        "y_count",
        "count_label",
    ):
        if key in sample:
            return int(scalar_value(sample[key]))
    return 0


def extract_pair_id(sample: dict[str, Any]) -> str | None:
    for key in (
        "pair_id",
        "matched_pair_id",
        "pair_uid",
        "pair_index",
        "pair_key",
    ):
        if key in sample:
            value = scalar_value(sample[key])
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            return str(value)
    return None


def physical_mask_index(channel_name: str) -> int:
    direction_to_index = {
        "local": 0,
        "north": 1,
        "east": 2,
        "south": 3,
        "west": 4,
    }
    direction = channel_name.rsplit("_", 1)[-1]
    if direction not in direction_to_index:
        raise RuntimeError(f"unknown direction in channel: {channel_name}")
    base = direction_to_index[direction]
    if channel_name.startswith("out_"):
        return 5 + base
    return base


def topology_position_masks() -> dict[str, np.ndarray]:
    corner = np.zeros(16, dtype=bool)
    edge = np.zeros(16, dtype=bool)
    interior = np.zeros(16, dtype=bool)
    for router in range(16):
        row, column = divmod(router, 4)
        boundary_count = int(row in (0, 3)) + int(column in (0, 3))
        if boundary_count == 2:
            corner[router] = True
        elif boundary_count == 1:
            edge[router] = True
        else:
            interior[router] = True
    return {
        "corner": corner,
        "edge": edge,
        "interior": interior,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")
    require(data_link.is_symlink(), f"dataset symlink missing: {data_link}")
    data_root = data_link.resolve()
    require(data_root.is_dir(), f"dataset target missing: {data_root}")

    f0_dir = repo / "reports/v5/p3_experiments/f0_d70_feature_study"
    f0_lock_path = f0_dir / "V5_P3_F0_D70_FEATURE_STUDY_LOCK.json"
    group_contract_path = f0_dir / "FEATURE_GROUP_CONTRACT.json"
    data_access_path = f0_dir / "DATA_ACCESS_CONTRACT.json"
    f1_report_path = (
        f0_dir
        / "schema_audit/V5_P3_F1_DYNAMIC70_SCHEMA_CERTIFICATION_REPORT.json"
    )
    f1_lock_path = (
        f0_dir
        / "schema_audit/V5_P3_F1_DYNAMIC70_SCHEMA_CERTIFICATION_LOCK.json"
    )

    loader_path = repo / "src/data/v5_p3_tranche_a_guarded_loader.py"

    required = [
        f0_lock_path,
        group_contract_path,
        data_access_path,
        f1_report_path,
        f1_lock_path,
        loader_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"required artifacts missing: {missing}")

    f0_lock = json.loads(f0_lock_path.read_text(encoding="utf-8"))
    group_contract = json.loads(
        group_contract_path.read_text(encoding="utf-8")
    )
    data_access = json.loads(data_access_path.read_text(encoding="utf-8"))
    f1_report = json.loads(f1_report_path.read_text(encoding="utf-8"))
    f1_lock = json.loads(f1_lock_path.read_text(encoding="utf-8"))

    require(f0_lock.get("status") == "FROZEN", "F0 is not frozen")
    require(f1_report.get("status") == "PASS", "F1 is not PASS")
    require(
        f1_lock.get("report_sha256") == sha256_file(f1_report_path),
        "F1 report/lock mismatch",
    )
    require(
        f1_lock.get("feature_group_contract_sha256")
        == sha256_file(group_contract_path),
        "feature-group contract mismatch",
    )
    require(
        f1_lock.get("F2_authorized") is True,
        "F1 did not authorize F2",
    )
    require(
        data_access["F2_F3"].get("Tranche_A_train") is True,
        "F2 train access not authorized",
    )
    require(
        data_access["F2_F3"].get("Tranche_A_validation") is False,
        "F2 validation access must be false",
    )
    require(
        f0_lock["authorizations"].get("Tranche_A_sealed_test_access")
        is False,
        "sealed-test access must remain false",
    )

    channels = group_contract["channels"]
    require(len(channels) == 70, "feature-group contract is not Dynamic70")
    channel_names = [row["name"] for row in channels]
    physical_indices = np.asarray(
        [physical_mask_index(name) for name in channel_names],
        dtype=np.int64,
    )

    loader_module = import_source(
        loader_path,
        "_v5_p3_f2_guarded_loader",
    )
    Dataset = loader_module.GuardedV5P3TrancheAPreliminaryDataset
    try:
        dataset = Dataset(data_root, "train", active_only=False)
    except TypeError:
        dataset = Dataset(
            data_root=data_root,
            split="train",
            active_only=False,
        )

    item_count = len(dataset)
    require(item_count > 0, "training dataset is empty")
    first = dataset[0]
    require(isinstance(first, dict), "guarded-loader sample is not a dict")
    sample_keys = sorted(str(key) for key in first.keys())

    first_x = tensor_to_numpy(
        extract_first(first, ("x", "features", "input"))
    )
    require(
        tuple(first_x.shape) == (16, 70, 32),
        f"unexpected x shape: {first_x.shape}",
    )
    first_mask = tensor_to_numpy(
        extract_first(
            first,
            (
                "physical_port_mask",
                "port_mask",
                "physical_mask",
            ),
        )
    )
    require(
        tuple(first_mask.shape) == (16, 10),
        f"unexpected physical mask shape: {first_mask.shape}",
    )

    overall = RunningStats.create(70)
    by_attack = {
        0: RunningStats.create(70),
        1: RunningStats.create(70),
    }
    by_count = {
        count: RunningStats.create(70)
        for count in (1, 2, 3, 4)
    }
    by_position = {
        name: RunningStats.create(70)
        for name in ("corner", "edge", "interior")
    }

    valid_count = np.zeros(70, dtype=np.int64)
    invalid_count = np.zeros(70, dtype=np.int64)
    invalid_nonzero_count = np.zeros(70, dtype=np.int64)
    invalid_abs_max = np.zeros(70, dtype=np.float64)
    sample_active_count = np.zeros(70, dtype=np.int64)
    sample_router_active_count = np.zeros(70, dtype=np.int64)
    sample_router_valid_count = np.zeros(70, dtype=np.int64)
    router_ever_active = np.zeros((70, 16), dtype=bool)

    position_masks = topology_position_masks()
    sample_indices = set(
        np.linspace(
            0,
            item_count - 1,
            num=min(QUANTILE_SAMPLE_ITEMS, item_count),
            dtype=np.int64,
        ).tolist()
    )
    quantile_values: list[list[np.ndarray]] = [
        [] for _ in range(70)
    ]

    pair_activity: dict[str, np.ndarray] = {}
    pair_id_available = False
    attack_items = 0
    control_items = 0
    count_items = {1: 0, 2: 0, 3: 0, 4: 0}
    physical_mask_reference = first_mask.astype(np.uint8)
    physical_mask_variation_count = 0

    started = datetime.now(timezone.utc)
    for index in range(item_count):
        sample = first if index == 0 else dataset[index]
        x = tensor_to_numpy(
            extract_first(sample, ("x", "features", "input"))
        ).astype(np.float64, copy=False)
        physical_mask = tensor_to_numpy(
            extract_first(
                sample,
                (
                    "physical_port_mask",
                    "port_mask",
                    "physical_mask",
                ),
            )
        ).astype(np.uint8, copy=False)

        require(
            tuple(x.shape) == (16, 70, 32),
            f"item {index}: x shape changed to {x.shape}",
        )
        require(
            tuple(physical_mask.shape) == (16, 10),
            f"item {index}: physical mask shape changed to {physical_mask.shape}",
        )
        if not np.array_equal(physical_mask, physical_mask_reference):
            physical_mask_variation_count += 1

        attack = extract_attack_label(sample)
        require(attack in (0, 1), f"item {index}: invalid attack label {attack}")
        count_label = extract_count_label(sample)
        if attack:
            attack_items += 1
            require(
                count_label in (1, 2, 3, 4),
                f"item {index}: active count label {count_label}",
            )
            count_items[count_label] += 1
        else:
            control_items += 1

        # [routers, channels, epochs] -> [channels, routers, epochs]
        channel_values = np.transpose(x, (1, 0, 2))
        channel_valid_router = (
            physical_mask[:, physical_indices].T.astype(bool)
        )
        valid = np.broadcast_to(
            channel_valid_router[:, :, None],
            channel_values.shape,
        )

        overall.update(channel_values, valid)
        by_attack[attack].update(channel_values, valid)
        if attack:
            by_count[count_label].update(channel_values, valid)

        for position_name, router_mask in position_masks.items():
            position_valid = valid & router_mask[None, :, None]
            by_position[position_name].update(
                channel_values,
                position_valid,
            )

        valid_count += np.sum(valid, axis=(1, 2))
        invalid = ~valid
        invalid_count += np.sum(invalid, axis=(1, 2))
        invalid_nonzero_count += np.sum(
            invalid & (np.abs(channel_values) > ZERO_TOLERANCE),
            axis=(1, 2),
        )
        local_invalid_max = np.max(
            np.where(invalid, np.abs(channel_values), 0.0),
            axis=(1, 2),
        )
        invalid_abs_max = np.maximum(
            invalid_abs_max,
            local_invalid_max,
        )

        active_values = valid & (
            np.abs(channel_values) > ZERO_TOLERANCE
        )
        sample_active = np.any(active_values, axis=(1, 2))
        sample_active_count += sample_active

        sample_router_active = np.any(active_values, axis=2)
        sample_router_valid = channel_valid_router
        sample_router_active_count += np.sum(
            sample_router_active,
            axis=1,
        )
        sample_router_valid_count += np.sum(
            sample_router_valid,
            axis=1,
        )
        router_ever_active |= sample_router_active

        pair_id = extract_pair_id(sample)
        if pair_id is not None:
            pair_id_available = True
            if pair_id not in pair_activity:
                pair_activity[pair_id] = np.zeros(70, dtype=bool)
            pair_activity[pair_id] |= sample_active

        if index in sample_indices:
            for channel in range(70):
                values = channel_values[channel][
                    channel_valid_router[channel]
                ].reshape(-1)
                finite_values = values[np.isfinite(values)]
                if finite_values.size:
                    quantile_values[channel].append(
                        finite_values.astype(np.float32, copy=True)
                    )

        if (
            (index + 1) % PROGRESS_INTERVAL == 0
            or index + 1 == item_count
        ):
            print(
                f"F2_training_items={index + 1}/{item_count}",
                flush=True,
            )

    overall_final = overall.finalize()
    attack_final = {
        key: value.finalize()
        for key, value in by_attack.items()
    }
    count_final = {
        key: value.finalize()
        for key, value in by_count.items()
    }
    position_final = {
        key: value.finalize()
        for key, value in by_position.items()
    }

    sample_fraction = sample_active_count / float(item_count)
    sample_router_fraction = np.divide(
        sample_router_active_count,
        sample_router_valid_count,
        out=np.full(70, np.nan, dtype=np.float64),
        where=sample_router_valid_count > 0,
    )
    router_ever_fraction = np.mean(router_ever_active, axis=1)

    if pair_id_available and pair_activity:
        pair_matrix = np.stack(list(pair_activity.values()), axis=0)
        pair_fraction = np.mean(pair_matrix, axis=0)
        pair_count = pair_matrix.shape[0]
    else:
        pair_fraction = np.full(70, np.nan, dtype=np.float64)
        pair_count = 0

    quantile_rows = []
    quantiles_by_channel = {}
    for channel in range(70):
        if quantile_values[channel]:
            values = np.concatenate(quantile_values[channel])
            q = np.quantile(
                values,
                [0.25, 0.5, 0.75],
            )
            unique_sample_count = int(
                np.unique(values).size
            )
            quantiles_by_channel[channel] = {
                "q25": float(q[0]),
                "median": float(q[1]),
                "q75": float(q[2]),
                "iqr": float(q[2] - q[0]),
                "sampled_value_count": int(values.size),
                "sampled_unique_count": unique_sample_count,
            }
        else:
            quantiles_by_channel[channel] = {
                "q25": float("nan"),
                "median": float("nan"),
                "q75": float("nan"),
                "iqr": float("nan"),
                "sampled_value_count": 0,
                "sampled_unique_count": 0,
            }

    feature_rows: list[dict[str, Any]] = []
    constant_rows: list[dict[str, Any]] = []
    rare_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    for channel, metadata in enumerate(channels):
        std = float(overall_final["std"][channel])
        dynamic_range = float(overall_final["range"][channel])
        nonfinite = int(overall_final["nonfinite_count"][channel])
        active_fraction = float(sample_fraction[channel])
        topology_conditioned = metadata["direction"] != "local"

        if nonfinite > 0:
            activity_class = "INVALID"
        elif (
            not math.isfinite(std)
            or std <= CONSTANT_TOLERANCE
            or dynamic_range <= CONSTANT_TOLERANCE
        ):
            activity_class = "CONSTANT"
        elif active_fraction <= NEAR_CONSTANT_ACTIVE_FRACTION:
            activity_class = "NEAR_CONSTANT"
        elif (
            (
                math.isfinite(float(pair_fraction[channel]))
                and float(pair_fraction[channel]) < RARE_SAMPLE_FRACTION
            )
            or active_fraction < RARE_SAMPLE_FRACTION
        ):
            activity_class = "RARE"
        else:
            activity_class = "ACTIVE"

        topology_class = (
            "TOPOLOGY_CONDITIONAL"
            if topology_conditioned
            else "TOPOLOGY_GLOBAL_LOCAL_PORT"
        )
        row = {
            **metadata,
            "valid_value_count": int(overall_final["count"][channel]),
            "minimum": float(overall_final["minimum"][channel]),
            "maximum": float(overall_final["maximum"][channel]),
            "dynamic_range": dynamic_range,
            "mean": float(overall_final["mean"][channel]),
            "std": std,
            "zero_fraction_valid": float(
                overall_final["zero_fraction"][channel]
            ),
            "q25_sampled": quantiles_by_channel[channel]["q25"],
            "median_sampled": quantiles_by_channel[channel]["median"],
            "q75_sampled": quantiles_by_channel[channel]["q75"],
            "iqr_sampled": quantiles_by_channel[channel]["iqr"],
            "sampled_unique_count": quantiles_by_channel[channel][
                "sampled_unique_count"
            ],
            "sample_activity_fraction": active_fraction,
            "sample_router_activity_fraction": float(
                sample_router_fraction[channel]
            ),
            "router_ever_active_fraction": float(
                router_ever_fraction[channel]
            ),
            "pair_activity_fraction": float(pair_fraction[channel]),
            "nonfinite_count": nonfinite,
            "invalid_port_value_count": int(invalid_count[channel]),
            "invalid_port_nonzero_count": int(
                invalid_nonzero_count[channel]
            ),
            "invalid_port_nonzero_fraction": (
                float(invalid_nonzero_count[channel] / invalid_count[channel])
                if invalid_count[channel]
                else 0.0
            ),
            "invalid_port_abs_max": float(invalid_abs_max[channel]),
            "activity_class": activity_class,
            "topology_class": topology_class,
            "combined_class": (
                f"{topology_class}_{activity_class}"
                if topology_conditioned
                else activity_class
            ),
        }
        feature_rows.append(row)
        if activity_class in ("CONSTANT", "NEAR_CONSTANT"):
            constant_rows.append(row)
        if activity_class == "RARE":
            rare_rows.append(row)
        if activity_class == "INVALID":
            invalid_rows.append(row)

    control_attack_rows = []
    for channel, metadata in enumerate(channels):
        row = {
            **metadata,
            "control_count": int(
                attack_final[0]["count"][channel]
            ),
            "control_mean": float(
                attack_final[0]["mean"][channel]
            ),
            "control_std": float(
                attack_final[0]["std"][channel]
            ),
            "control_zero_fraction": float(
                attack_final[0]["zero_fraction"][channel]
            ),
            "attack_count": int(
                attack_final[1]["count"][channel]
            ),
            "attack_mean": float(
                attack_final[1]["mean"][channel]
            ),
            "attack_std": float(
                attack_final[1]["std"][channel]
            ),
            "attack_zero_fraction": float(
                attack_final[1]["zero_fraction"][channel]
            ),
            "attack_minus_control_mean": float(
                attack_final[1]["mean"][channel]
                - attack_final[0]["mean"][channel]
            ),
        }
        control_attack_rows.append(row)

    count_rows = []
    for count_label in (1, 2, 3, 4):
        final = count_final[count_label]
        for channel, metadata in enumerate(channels):
            count_rows.append({
                **metadata,
                "attacker_count": count_label,
                "value_count": int(final["count"][channel]),
                "mean": float(final["mean"][channel]),
                "std": float(final["std"][channel]),
                "zero_fraction": float(
                    final["zero_fraction"][channel]
                ),
            })

    position_rows = []
    for position_name in ("corner", "edge", "interior"):
        final = position_final[position_name]
        for channel, metadata in enumerate(channels):
            position_rows.append({
                **metadata,
                "router_position": position_name,
                "value_count": int(final["count"][channel]),
                "mean": float(final["mean"][channel]),
                "std": float(final["std"][channel]),
                "zero_fraction": float(
                    final["zero_fraction"][channel]
                ),
                "minimum": float(final["minimum"][channel]),
                "maximum": float(final["maximum"][channel]),
            })

    group_rows = []
    for group_name, group_indices in group_contract[
        "macro_groups"
    ].items():
        selected_rows = [
            feature_rows[int(index)] for index in group_indices
        ]
        group_rows.append({
            "macro_group": group_name,
            "channel_count": len(selected_rows),
            "active_channels": sum(
                row["activity_class"] == "ACTIVE"
                for row in selected_rows
            ),
            "rare_channels": sum(
                row["activity_class"] == "RARE"
                for row in selected_rows
            ),
            "near_constant_channels": sum(
                row["activity_class"] == "NEAR_CONSTANT"
                for row in selected_rows
            ),
            "constant_channels": sum(
                row["activity_class"] == "CONSTANT"
                for row in selected_rows
            ),
            "invalid_channels": sum(
                row["activity_class"] == "INVALID"
                for row in selected_rows
            ),
            "mean_sample_activity_fraction": float(
                np.mean(
                    [
                        row["sample_activity_fraction"]
                        for row in selected_rows
                    ]
                )
            ),
            "mean_zero_fraction_valid": float(
                np.mean(
                    [
                        row["zero_fraction_valid"]
                        for row in selected_rows
                    ]
                )
            ),
            "mean_invalid_port_nonzero_fraction": float(
                np.mean(
                    [
                        row["invalid_port_nonzero_fraction"]
                        for row in selected_rows
                    ]
                )
            ),
        })

    schema_audit_dir = output_dir / "schema_audit"
    schema_audit_dir.mkdir(parents=True, exist_ok=True)
    feature_csv = schema_audit_dir / "F2_FEATURE_STATISTICS.csv"
    position_csv = schema_audit_dir / "F2_ROUTER_POSITION_STATISTICS.csv"
    control_attack_csv = schema_audit_dir / "F2_CONTROL_ATTACK_STATISTICS.csv"
    count_csv = schema_audit_dir / "F2_ATTACKER_COUNT_STATISTICS.csv"
    topology_csv = schema_audit_dir / "F2_TOPOLOGY_MASK_AUDIT.csv"
    constant_csv = schema_audit_dir / "F2_CONSTANT_AND_NEAR_CONSTANT_CHANNELS.csv"
    rare_csv = schema_audit_dir / "F2_RARE_CHANNELS.csv"
    invalid_csv = schema_audit_dir / "F2_INVALID_CHANNELS.csv"
    group_csv = schema_audit_dir / "F2_GROUP_SUMMARY.csv"
    sample_keys_path = schema_audit_dir / "F2_GUARDED_LOADER_SAMPLE_KEYS.json"
    figure_png = schema_audit_dir / "F2_ACTIVITY_HEATMAP.png"
    figure_pdf = schema_audit_dir / "F2_ACTIVITY_HEATMAP.pdf"

    write_csv(feature_csv, feature_rows)
    write_csv(position_csv, position_rows)
    write_csv(control_attack_csv, control_attack_rows)
    write_csv(count_csv, count_rows)
    write_csv(
        topology_csv,
        [
            {
                "dynamic70_index": row["dynamic70_index"],
                "name": row["name"],
                "direction": row["direction"],
                "physical_mask_index": int(
                    physical_indices[row["dynamic70_index"]]
                ),
                "invalid_port_value_count": row[
                    "invalid_port_value_count"
                ],
                "invalid_port_nonzero_count": row[
                    "invalid_port_nonzero_count"
                ],
                "invalid_port_nonzero_fraction": row[
                    "invalid_port_nonzero_fraction"
                ],
                "invalid_port_abs_max": row[
                    "invalid_port_abs_max"
                ],
            }
            for row in feature_rows
        ],
    )
    write_csv(constant_csv, constant_rows)
    write_csv(rare_csv, rare_rows)
    write_csv(invalid_csv, invalid_rows)
    write_csv(group_csv, group_rows)
    atomic_json(
        sample_keys_path,
        {
            "sample_keys": sample_keys,
            "x_shape": list(first_x.shape),
            "physical_port_mask_shape": list(first_mask.shape),
            "pair_id_available": pair_id_available,
            "pair_count": pair_count,
        },
    )

    figure_status = {
        "generated": False,
        "png": None,
        "pdf": None,
        "exception": None,
    }
    try:
        import matplotlib.pyplot as plt

        values = np.asarray(
            [
                [
                    row["sample_activity_fraction"],
                    row["zero_fraction_valid"],
                    row["sample_router_activity_fraction"],
                    row["router_ever_active_fraction"],
                ]
                for row in feature_rows
            ],
            dtype=np.float64,
        )
        fig, ax = plt.subplots(figsize=(10, 16))
        image = ax.imshow(values, aspect="auto")
        ax.set_xlabel("Audit measure")
        ax.set_ylabel("Dynamic70 channel")
        ax.set_xticks(range(4))
        ax.set_xticklabels(
            [
                "sample activity",
                "zero fraction",
                "sample-router activity",
                "router ever active",
            ],
            rotation=30,
            ha="right",
        )
        ax.set_yticks(range(70))
        ax.set_yticklabels(
            [
                f"{row['dynamic70_index']:02d} {row['name']}"
                for row in feature_rows
            ],
            fontsize=5,
        )
        fig.colorbar(image, ax=ax, label="Fraction")
        fig.tight_layout()
        fig.savefig(figure_png, dpi=180)
        fig.savefig(figure_pdf)
        plt.close(fig)
        figure_status = {
            "generated": True,
            "png": str(figure_png),
            "pdf": str(figure_pdf),
            "exception": None,
        }
    except Exception as exc:
        figure_status["exception"] = repr(exc)

    nonfinite_total = int(
        sum(row["nonfinite_count"] for row in feature_rows)
    )
    constant_on_valid = [
        row["name"]
        for row in feature_rows
        if row["activity_class"] == "CONSTANT"
    ]
    near_constant = [
        row["name"]
        for row in feature_rows
        if row["activity_class"] == "NEAR_CONSTANT"
    ]
    rare = [
        row["name"]
        for row in feature_rows
        if row["activity_class"] == "RARE"
    ]

    # Scientific stage completion and promotion authorization are separate.
    f3_authorized = (
        nonfinite_total == 0
        and len(constant_on_valid) == 0
    )
    next_stage = (
        "V5_P3_F3_DYNAMIC70_TRAINING_ONLY_MATCHED_"
        "CONTROL_ATTACK_RESPONSE_ANALYSIS"
        if f3_authorized
        else "V5_P3_F2R_DYNAMIC70_DEGENERACY_FINDING_REVIEW"
    )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": "Tranche-A training-only descriptive audit",
        "analysis_domain": (
            "normalized Dynamic70 model input returned by the guarded loader; "
            "statistics exclude physically invalid ports using the separate "
            "16x10 physical-port mask"
        ),
        "dataset": {
            "split": "train",
            "active_only": False,
            "items": item_count,
            "control_items": control_items,
            "attack_items": attack_items,
            "active_count_items": count_items,
            "pair_id_available": pair_id_available,
            "distinct_pair_ids": pair_count,
        },
        "interface": {
            "x_shape": [16, 70, 32],
            "physical_port_mask_shape": [16, 10],
            "physical_mask_variation_count": (
                physical_mask_variation_count
            ),
        },
        "findings": {
            "active_channels": sum(
                row["activity_class"] == "ACTIVE"
                for row in feature_rows
            ),
            "rare_channels": rare,
            "near_constant_channels": near_constant,
            "constant_channels": constant_on_valid,
            "invalid_channels": [
                row["name"] for row in invalid_rows
            ],
            "nonfinite_value_count": nonfinite_total,
        },
        "artifacts": {
            "feature_statistics": str(feature_csv),
            "router_position_statistics": str(position_csv),
            "control_attack_statistics": str(control_attack_csv),
            "attacker_count_statistics": str(count_csv),
            "topology_mask_audit": str(topology_csv),
            "constant_channels": str(constant_csv),
            "rare_channels": str(rare_csv),
            "invalid_channels": str(invalid_csv),
            "group_summary": str(group_csv),
            "sample_keys": str(sample_keys_path),
            "figure": figure_status,
        },
        "decision": {
            "F2_complete": True,
            "F3_authorized": f3_authorized,
            "validation_access_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": next_stage,
        },
        "governance": {
            "model_loaded": False,
            "training_tensors_loaded": True,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
        },
        "provenance": {
            "F0_lock_sha256": sha256_file(f0_lock_path),
            "feature_group_contract_sha256": sha256_file(
                group_contract_path
            ),
            "data_access_contract_sha256": sha256_file(
                data_access_path
            ),
            "F1_report_sha256": sha256_file(f1_report_path),
            "F1_lock_sha256": sha256_file(f1_lock_path),
            "guarded_loader_sha256": sha256_file(loader_path),
            "installed_script_sha256": sha256_file(
                installed_script
            ),
        },
    }

    report_path = schema_audit_dir / f"{STAGE}_REPORT.json"
    lock_path = schema_audit_dir / f"{STAGE}_LOCK.json"
    complete_path = schema_audit_dir / f"{STAGE}_COMPLETE"
    atomic_json(report_path, report)
    atomic_json(
        lock_path,
        {
            "stage": STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(report_path),
            "feature_statistics_sha256": sha256_file(feature_csv),
            "router_position_statistics_sha256": sha256_file(
                position_csv
            ),
            "control_attack_statistics_sha256": sha256_file(
                control_attack_csv
            ),
            "attacker_count_statistics_sha256": sha256_file(
                count_csv
            ),
            "topology_mask_audit_sha256": sha256_file(
                topology_csv
            ),
            "group_summary_sha256": sha256_file(group_csv),
            "training_items": item_count,
            "nonfinite_value_count": nonfinite_total,
            "constant_channel_count": len(constant_on_valid),
            "F3_authorized": f3_authorized,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign={CAMPAIGN}")
    print("analysis_domain=normalized_guarded_loader_input")
    print(f"training_items={item_count}")
    print(f"control_items={control_items}")
    print(f"attack_items={attack_items}")
    print(f"active_count_items={count_items}")
    print(f"pair_id_available={str(pair_id_available).lower()}")
    print(f"distinct_pair_ids={pair_count}")
    print(
        "physical_mask_variation_count="
        f"{physical_mask_variation_count}"
    )
    print(f"active_channels={report['findings']['active_channels']}")
    print(f"rare_channel_count={len(rare)}")
    print(f"near_constant_channel_count={len(near_constant)}")
    print(f"constant_channel_count={len(constant_on_valid)}")
    print(f"invalid_channel_count={len(invalid_rows)}")
    print(f"nonfinite_value_count={nonfinite_total}")
    print(f"figure_generated={str(figure_status['generated']).lower()}")
    print("model_loaded=false")
    print("training_tensors_loaded=true")
    print("validation_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"F3_authorized={str(f3_authorized).lower()}")
    print(f"next_stage={next_stage}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
