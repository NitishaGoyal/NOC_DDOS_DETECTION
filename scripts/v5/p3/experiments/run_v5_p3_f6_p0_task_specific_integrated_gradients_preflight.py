from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = "V5_P3_F6_P0_TASK_SPECIFIC_INTEGRATED_GRADIENTS_PREFLIGHT"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "VALIDATION-EXPLORATORY"

EXPECTED_ITEMS = 13863
EXPECTED_PARAMETER_COUNT = 60553
EXPECTED_INPUT_SHAPE = [16, 70, 32]
EXPECTED_MASK_SHAPE = [16, 10]
EXPECTED_EDGE_COUNT = 48

GROUPS = {
    "directional_traffic_volume": {
        "start": 0,
        "end_exclusive": 10,
        "count": 10,
    },
    "inter_flit_timing": {
        "start": 10,
        "end_exclusive": 30,
        "count": 20,
    },
    "queue_activity": {
        "start": 30,
        "end_exclusive": 40,
        "count": 10,
    },
    "buffer_pressure": {
        "start": 40,
        "end_exclusive": 55,
        "count": 15,
    },
    "flow_control_stalls": {
        "start": 55,
        "end_exclusive": 70,
        "count": 15,
    },
}

TASKS = {
    "graph": {
        "target": "attack_logit",
        "sample_scope": "active_attack_items",
    },
    "count": {
        "target": (
            "true_count_class_logit_minus_mean_of_other_three_logits"
        ),
        "sample_scope": "active_attack_items",
    },
    "source": {
        "target": (
            "mean_true_positive_node_logit_minus_"
            "mean_true_negative_node_logit"
        ),
        "sample_scope": "active_attack_items_with_nonempty_source_target",
    },
    "transit": {
        "target": (
            "mean_true_positive_node_logit_minus_"
            "mean_true_negative_node_logit"
        ),
        "sample_scope": "active_attack_items_with_nonempty_transit_target",
    },
    "victim": {
        "target": (
            "mean_true_positive_node_logit_minus_"
            "mean_true_negative_node_logit"
        ),
        "sample_scope": "active_attack_items_with_nonempty_victim_target",
    },
    "path": {
        "target": (
            "mean_true_positive_node_logit_minus_"
            "mean_true_negative_node_logit"
        ),
        "sample_scope": "active_attack_items_with_nonempty_path_target",
    },
}

PAIR_ID_ALIASES = (
    "pair_id",
    "pairid",
    "matched_pair_id",
    "pair_index",
)
PAIR_MEMBER_ALIASES = (
    "pair_member",
    "member",
    "pair_role",
    "control_attack",
    "attack_control",
    "is_attack",
)
WINDOW_ALIASES = (
    "window_start",
    "window_start_epoch",
    "window_epoch_start",
    "start_epoch",
    "window_id",
)
SAMPLE_INDEX_ALIASES = (
    "sample_index",
    "item_index",
    "validation_index",
    "index",
)
ATTACK_LABEL_ALIASES = (
    "y_attack",
    "attack_label",
    "graph_label",
    "is_attack",
)
COUNT_LABEL_ALIASES = (
    "y_attacker_count",
    "attacker_count",
    "attack_count",
    "count_label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_report_lock(
    report_path: Path,
    lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = load_json(report_path)
    lock = load_json(lock_path)
    require(report.get("status") == "PASS", f"report not PASS: {report_path}")
    require(
        lock.get("report_sha256") == sha256_file(report_path),
        f"report/lock mismatch: {report_path}",
    )
    return report, lock


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def matching_key(keys: list[str], aliases: tuple[str, ...]) -> str | None:
    normalized = {normalize(key): key for key in keys}
    for alias in aliases:
        key = normalized.get(normalize(alias))
        if key is not None:
            return key
    return None


def read_npy_header_portable(
    handle: Any,
) -> tuple[tuple[int, int], tuple[int, ...], bool, np.dtype]:
    """
    Read only a .npy member's published header structure.

    NumPy 2.x removed numpy.lib.format._read_array_header because it is a
    private implementation detail. This parser supports the published 1.0,
    2.0 and 3.0 formats without reading the array payload.
    """
    major, minor = np.lib.format.read_magic(handle)
    version = (int(major), int(minor))

    if version == (1, 0):
        length_bytes = handle.read(2)
        require(
            len(length_bytes) == 2,
            "truncated NumPy 1.0 header-length field",
        )
        header_length = struct.unpack("<H", length_bytes)[0]
        encoding = "latin1"
    elif version in ((2, 0), (3, 0)):
        length_bytes = handle.read(4)
        require(
            len(length_bytes) == 4,
            f"truncated NumPy {major}.{minor} header-length field",
        )
        header_length = struct.unpack("<I", length_bytes)[0]
        encoding = "utf-8" if version == (3, 0) else "latin1"
    else:
        raise RuntimeError(
            f"unsupported NumPy array format version: {major}.{minor}"
        )

    require(
        0 < header_length <= 64 * 1024 * 1024,
        f"unsafe or invalid NumPy header length: {header_length}",
    )
    header_bytes = handle.read(header_length)
    require(
        len(header_bytes) == header_length,
        "truncated NumPy array header",
    )

    try:
        header_text = header_bytes.decode(encoding)
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"NumPy header decode failed for version {major}.{minor}"
        ) from exc

    try:
        header = ast.literal_eval(header_text.strip())
    except (SyntaxError, ValueError) as exc:
        raise RuntimeError("NumPy header literal parsing failed") from exc

    require(isinstance(header, dict), "NumPy header is not a dictionary")
    require(
        set(header) == {"descr", "fortran_order", "shape"},
        f"unexpected NumPy header keys: {sorted(header)}",
    )

    shape_value = header["shape"]
    require(
        isinstance(shape_value, tuple)
        and all(isinstance(item, int) and item >= 0 for item in shape_value),
        f"invalid NumPy shape in header: {shape_value!r}",
    )
    require(
        isinstance(header["fortran_order"], bool),
        "invalid NumPy fortran_order field",
    )

    try:
        dtype = np.dtype(header["descr"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"invalid NumPy dtype descriptor: {header['descr']!r}"
        ) from exc

    return (
        version,
        tuple(int(item) for item in shape_value),
        bool(header["fortran_order"]),
        dtype,
    )


def npz_inventory(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"NPZ missing: {path}")
    rows = {}
    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.infolist():
            if not member.filename.endswith(".npy"):
                continue
            key = member.filename[:-4]
            with archive.open(member, "r") as handle:
                version, shape, fortran_order, dtype = (
                    read_npy_header_portable(handle)
                )
            rows[key] = {
                "member": member.filename,
                "npy_format_version": [
                    int(version[0]),
                    int(version[1]),
                ],
                "shape": [int(item) for item in shape],
                "dtype": str(dtype),
                "compressed_size": int(member.compress_size),
                "uncompressed_size": int(member.file_size),
                "fortran_order": bool(fortran_order),
            }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "header_parser": (
            "portable_public_npy_format_parser_no_private_numpy_API"
        ),
        "keys": rows,
    }


def safe_load_npz_arrays(
    path: Path,
    selected_keys: list[str],
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: np.asarray(archive[key])
            for key in selected_keys
        }


def candidate_csv_files(repo: Path) -> list[Path]:
    roots = [
        repo / "reports/v5/p3_d1_immutable_validation_logit_export",
        repo / "reports/v5/p3_experiments/f0_d70_feature_study",
        repo / "reports/v5",
    ]
    files = []
    seen = set()

    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.csv"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                if path.stat().st_size > 256 * 1024 * 1024:
                    continue
            except OSError:
                continue
            files.append(resolved)

    return files


def inspect_csv(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return None
            fields = [str(field) for field in reader.fieldnames]
            rows = []
            for index, row in enumerate(reader):
                if index < 5:
                    rows.append(dict(row))
                if index + 1 > EXPECTED_ITEMS + 10:
                    break
            row_count = index + 1 if 'index' in locals() else 0
    except Exception:
        return None

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "fields": fields,
        "row_count": row_count,
        "preview": rows,
        "pair_id_key": matching_key(fields, PAIR_ID_ALIASES),
        "pair_member_key": matching_key(fields, PAIR_MEMBER_ALIASES),
        "window_key": matching_key(fields, WINDOW_ALIASES),
        "sample_index_key": matching_key(fields, SAMPLE_INDEX_ALIASES),
        "attack_label_key": matching_key(fields, ATTACK_LABEL_ALIASES),
        "count_label_key": matching_key(fields, COUNT_LABEL_ALIASES),
    }


def classify_pair_member(value: Any) -> str | None:
    text = normalize(str(value))
    if text in ("attack", "attacked", "active", "malicious", "1", "true"):
        return "attack"
    if text in ("control", "benign", "clean", "normal", "0", "false"):
        return "control"
    try:
        numeric = int(float(str(value)))
    except Exception:
        return None
    if numeric == 1:
        return "attack"
    if numeric == 0:
        return "control"
    return None


def validate_csv_pair_mapping(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    path = Path(candidate["path"])
    pair_key = candidate["pair_id_key"]
    member_key = candidate["pair_member_key"]
    window_key = candidate["window_key"]
    sample_key = candidate["sample_index_key"]

    if not pair_key or not member_key:
        return {
            "usable": False,
            "reason": "missing pair_id or pair_member column",
        }

    rows = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            rows.append({
                "row_index": row_index,
                "sample_index": (
                    int(float(row[sample_key]))
                    if sample_key and row.get(sample_key, "") != ""
                    else row_index
                ),
                "pair_id": str(row[pair_key]),
                "pair_member": classify_pair_member(row[member_key]),
                "window": (
                    str(row[window_key])
                    if window_key and row.get(window_key, "") != ""
                    else None
                ),
            })

    if len(rows) != EXPECTED_ITEMS:
        return {
            "usable": False,
            "reason": (
                f"row count is {len(rows)}, expected {EXPECTED_ITEMS}"
            ),
        }

    if any(row["pair_member"] is None for row in rows):
        return {
            "usable": False,
            "reason": "unrecognized pair_member values",
        }

    mapping_key = (
        "pair_id+window"
        if window_key is not None
        else "pair_id"
    )
    grouped: dict[tuple[str, str | None], dict[str, list[int]]] = {}
    for row in rows:
        key = (row["pair_id"], row["window"])
        bucket = grouped.setdefault(
            key,
            {"attack": [], "control": []},
        )
        bucket[row["pair_member"]].append(row["sample_index"])

    invalid = {
        str(key): value
        for key, value in grouped.items()
        if len(value["attack"]) != 1 or len(value["control"]) != 1
    }
    attack_to_control = {
        value["attack"][0]: value["control"][0]
        for value in grouped.values()
        if len(value["attack"]) == 1 and len(value["control"]) == 1
    }

    return {
        "usable": len(invalid) == 0 and len(attack_to_control) > 0,
        "mapping_key": mapping_key,
        "row_count": len(rows),
        "group_count": len(grouped),
        "matched_attack_item_count": len(attack_to_control),
        "invalid_group_count": len(invalid),
        "invalid_group_preview": dict(list(invalid.items())[:20]),
        "attack_to_control_preview": dict(
            list(sorted(attack_to_control.items()))[:20]
        ),
        "attack_to_control_mapping": attack_to_control,
    }


def infer_npz_pair_mapping(
    npz_path: Path,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    keys = list(inventory["keys"])
    pair_key = matching_key(keys, PAIR_ID_ALIASES)
    member_key = matching_key(keys, PAIR_MEMBER_ALIASES)
    window_key = matching_key(keys, WINDOW_ALIASES)
    sample_key = matching_key(keys, SAMPLE_INDEX_ALIASES)
    attack_key = matching_key(keys, ATTACK_LABEL_ALIASES)

    if not pair_key:
        return {
            "usable": False,
            "reason": "NPZ has no pair_id-like array",
            "keys": {
                "pair_id": pair_key,
                "pair_member": member_key,
                "window": window_key,
                "sample_index": sample_key,
                "attack_label": attack_key,
            },
        }

    selected = [pair_key]
    for key in (member_key, window_key, sample_key, attack_key):
        if key is not None and key not in selected:
            selected.append(key)

    arrays = safe_load_npz_arrays(npz_path, selected)
    length = int(np.asarray(arrays[pair_key]).reshape(-1).shape[0])
    if length != EXPECTED_ITEMS:
        return {
            "usable": False,
            "reason": f"NPZ pair array length is {length}",
        }

    pair_values = np.asarray(arrays[pair_key]).reshape(-1)
    window_values = (
        np.asarray(arrays[window_key]).reshape(-1)
        if window_key is not None
        else np.asarray([None] * length, dtype=object)
    )
    sample_values = (
        np.asarray(arrays[sample_key]).reshape(-1)
        if sample_key is not None
        else np.arange(length, dtype=np.int64)
    )

    if member_key is not None:
        member_values = np.asarray(arrays[member_key]).reshape(-1)
        members = [classify_pair_member(value) for value in member_values]
    elif attack_key is not None:
        attack_values = np.asarray(arrays[attack_key]).reshape(-1)
        members = [
            "attack" if int(value) == 1 else "control"
            for value in attack_values
        ]
    else:
        return {
            "usable": False,
            "reason": (
                "NPZ has pair_id but no pair_member or attack-label array"
            ),
        }

    if any(member is None for member in members):
        return {
            "usable": False,
            "reason": "unrecognized NPZ pair-member values",
        }

    grouped: dict[tuple[str, str], dict[str, list[int]]] = {}
    for index in range(length):
        key = (
            str(pair_values[index]),
            str(window_values[index]),
        )
        bucket = grouped.setdefault(
            key,
            {"attack": [], "control": []},
        )
        bucket[members[index]].append(int(sample_values[index]))

    invalid = {
        str(key): value
        for key, value in grouped.items()
        if len(value["attack"]) != 1 or len(value["control"]) != 1
    }
    attack_to_control = {
        value["attack"][0]: value["control"][0]
        for value in grouped.values()
        if len(value["attack"]) == 1 and len(value["control"]) == 1
    }

    return {
        "usable": len(invalid) == 0 and len(attack_to_control) > 0,
        "mapping_key": (
            "pair_id+window"
            if window_key is not None
            else "pair_id"
        ),
        "keys": {
            "pair_id": pair_key,
            "pair_member": member_key,
            "window": window_key,
            "sample_index": sample_key,
            "attack_label": attack_key,
        },
        "group_count": len(grouped),
        "matched_attack_item_count": len(attack_to_control),
        "invalid_group_count": len(invalid),
        "invalid_group_preview": dict(list(invalid.items())[:20]),
        "attack_to_control_preview": dict(
            list(sorted(attack_to_control.items()))[:20]
        ),
        "attack_to_control_mapping": attack_to_control,
    }


def count_strata_from_npz(
    npz_path: Path,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    keys = list(inventory["keys"])
    attack_key = matching_key(keys, ATTACK_LABEL_ALIASES)
    count_key = matching_key(keys, COUNT_LABEL_ALIASES)

    if not attack_key or not count_key:
        return {
            "resolved": False,
            "reason": "attack/count label arrays unavailable",
            "attack_key": attack_key,
            "count_key": count_key,
        }

    arrays = safe_load_npz_arrays(npz_path, [attack_key, count_key])
    attack = np.asarray(arrays[attack_key]).astype(np.int64).reshape(-1)
    count = np.asarray(arrays[count_key]).astype(np.int64).reshape(-1)

    require(attack.shape[0] == EXPECTED_ITEMS, "attack label length mismatch")
    require(count.shape[0] == EXPECTED_ITEMS, "count label length mismatch")

    active_counts = count[attack == 1]
    strata = {
        str(value): int(np.sum(active_counts == value))
        for value in sorted(np.unique(active_counts))
    }
    enough = all(strata.get(str(value), 0) >= 128 for value in (1, 2, 3, 4))
    return {
        "resolved": True,
        "attack_key": attack_key,
        "count_key": count_key,
        "active_item_count": int(np.sum(attack == 1)),
        "control_item_count": int(np.sum(attack == 0)),
        "active_count_strata": strata,
        "supports_128_per_count": enough,
    }


def protocol_contract() -> dict[str, Any]:
    covered = []
    for group, row in GROUPS.items():
        require(
            row["end_exclusive"] - row["start"] == row["count"],
            f"group count mismatch: {group}",
        )
        covered.extend(range(row["start"], row["end_exclusive"]))
    require(covered == list(range(70)), "groups do not cover 0..69 exactly")

    return {
        "classification": CLASSIFICATION,
        "tasks": TASKS,
        "baseline": {
            "type": "matched_control_feature_tensor",
            "matching_priority": [
                "pair_id+window_start",
                "pair_id",
            ],
            "attack_item_attributed": True,
            "control_item_attributed": False,
            "physical_port_mask": (
                "attack-item mask retained; baseline feature tensor only"
            ),
            "edge_index": "unchanged",
        },
        "sample_selection": {
            "scope": "active validation items only",
            "stratification": "attacker_count in {1,2,3,4}",
            "items_per_count": 128,
            "total_items": 512,
            "selection_seed": 6101,
            "without_replacement": True,
            "same_sample_set_for_all_tasks_when_target_is_defined": True,
        },
        "integrated_gradients": {
            "path": "straight_line_matched_control_to_attack",
            "integration_rule": "trapezoidal",
            "steps": 64,
            "step_endpoints_included": True,
            "gradient_microbatch_size": 16,
            "input_requires_grad_only": True,
            "model_parameters_require_grad": False,
            "runtime": {
                "deterministic_algorithms_enabled": False,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": True,
                "float32_matmul_precision": "highest",
            },
        },
        "completeness_gate": {
            "identity": (
                "sum(IG) approximately equals target(attack)-target(control)"
            ),
            "absolute_error_reported": True,
            "relative_error_denominator": (
                "max(abs(target_difference),1e-6)"
            ),
            "task_median_relative_error_max": 0.02,
            "task_p95_relative_error_max": 0.05,
            "fallback_if_failed": "repeat failed task with 128 steps",
        },
        "aggregation": {
            "signed_channel_mean": True,
            "absolute_channel_mean": True,
            "group_total_absolute_attribution": True,
            "group_mean_absolute_attribution_per_channel": True,
            "router_absolute_attribution": True,
            "time_step_absolute_attribution": True,
            "normalization": [
                "fraction_of_total_absolute_attribution",
                "per_channel_mean_to_control_for_group_width",
            ],
            "bootstrap_resamples": 1000,
            "bootstrap_seed": 6102,
            "confidence_interval": 0.95,
        },
        "macro_groups": GROUPS,
        "paper_claim_boundary": {
            "feature_importance_wording_authorized_after_F6": True,
            "feature_removal_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "compact_interface_equivalence_authorized": False,
        },
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")
    require(data_link.is_symlink(), f"dataset symlink missing: {data_link}")
    require(data_link.resolve().is_dir(), "dataset target missing")

    feature_root = repo / (
        "reports/v5/p3_experiments/f0_d70_feature_study"
    )
    f5r_dir = feature_root / "permutation_review"
    permutation_dir = feature_root / "permutation"
    metric_dir = feature_root / "metric_adapter"

    f5r_report_path = f5r_dir / (
        "V5_P3_F5R_GROUP_BLOCK_PERMUTATION_RESULT_REVIEW_REPORT.json"
    )
    f5r_lock_path = f5r_dir / (
        "V5_P3_F5R_GROUP_BLOCK_PERMUTATION_RESULT_REVIEW_LOCK.json"
    )
    f5_report_path = permutation_dir / (
        "V5_P3_F5_DYNAMIC70_GROUP_BLOCK_PERMUTATION_REPORT.json"
    )
    f5_lock_path = permutation_dir / (
        "V5_P3_F5_DYNAMIC70_GROUP_BLOCK_PERMUTATION_LOCK.json"
    )
    f4m_report_path = metric_dir / (
        "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION_FINAL_REPORT.json"
    )
    f4m_lock_path = metric_dir / (
        "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION_FINAL_LOCK.json"
    )
    route_path = permutation_dir / (
        "F5_P0_EXECUTION_ROUTE_SOURCE_INVENTORY.json"
    )
    runtime_recovery_path = permutation_dir / (
        "V5_P3_F5_R3B_OFFICIAL_RUNTIME_BASELINE_RECOVERY_REPORT.json"
    )

    f5r_report, f5r_lock = verify_report_lock(
        f5r_report_path,
        f5r_lock_path,
    )
    f5_report, f5_lock = verify_report_lock(
        f5_report_path,
        f5_lock_path,
    )
    f4m_report, f4m_lock = verify_report_lock(
        f4m_report_path,
        f4m_lock_path,
    )

    require(f5r_lock.get("F5R_complete") is True, "F5R incomplete")
    require(f5r_lock.get("F6_authorized") is True, "F6 not authorized")
    require(f5r_lock.get("F7_authorized") is False, "F7 must remain held")
    require(
        f5r_lock.get("feature_removal_authorized") is False,
        "feature removal must remain unauthorized",
    )
    require(
        f5r_lock.get("hardware_reduction_claim_authorized") is False,
        "hardware reduction claims must remain unauthorized",
    )
    require(f5_lock.get("F5_complete") is True, "F5 incomplete")
    require(f4m_lock.get("all_14_metrics_certified") is True, "F4M incomplete")
    require(route_path.is_file(), "execution route inventory missing")
    require(runtime_recovery_path.is_file(), "runtime recovery report missing")

    route = load_json(route_path)
    runtime_recovery = load_json(runtime_recovery_path)
    require(runtime_recovery.get("status") == "PASS", "runtime recovery not PASS")
    require(
        runtime_recovery["finding"]["confirmed_causal_runtime_field"]
        == "cudnn_allow_tf32",
        "runtime-recovery causal field changed",
    )

    certified = route["certified_files"]
    source_files = {}
    for label in ("model", "loader", "exporter", "checkpoint", "adapter"):
        row = certified[label]
        path = Path(row["path"]).resolve()
        require(path.is_file(), f"{label} missing: {path}")
        require(
            sha256_file(path) == row["actual_sha256"],
            f"{label} hash changed: {path}",
        )
        source_files[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    require(
        f5_report["execution"]["parameter_count"] == EXPECTED_PARAMETER_COUNT,
        "parameter count changed",
    )
    require(
        f5_report["execution"]["validation_items"] == EXPECTED_ITEMS,
        "validation item count changed",
    )

    fresh_cert_path = metric_dir / (
        "F4M_R3A_FRESH_REPLAY_14_METRIC_CERTIFICATION.json"
    )
    fresh_cert = load_json(fresh_cert_path)
    fresh_npz = Path(fresh_cert["npz"]).resolve()
    require(fresh_npz.is_file(), "fresh validation NPZ missing")
    require(
        sha256_file(fresh_npz) == fresh_cert["npz_sha256"],
        "fresh validation NPZ hash changed",
    )

    inventory = npz_inventory(fresh_npz)
    npz_pair_mapping = infer_npz_pair_mapping(
        fresh_npz,
        inventory,
    )
    count_strata = count_strata_from_npz(
        fresh_npz,
        inventory,
    )

    csv_candidates = []
    csv_mapping = None
    for path in candidate_csv_files(repo):
        candidate = inspect_csv(path)
        if candidate is None:
            continue
        csv_candidates.append(candidate)
        mapping = validate_csv_pair_mapping(candidate)
        candidate["pair_mapping_review"] = {
            key: value
            for key, value in mapping.items()
            if key != "attack_to_control_mapping"
        }
        if mapping.get("usable") and csv_mapping is None:
            csv_mapping = {
                "candidate": candidate,
                "mapping": mapping,
            }

    if npz_pair_mapping.get("usable"):
        pair_source = {
            "type": "fresh_validation_npz",
            "path": str(fresh_npz),
            "sha256": sha256_file(fresh_npz),
            "mapping": {
                key: value
                for key, value in npz_pair_mapping.items()
                if key != "attack_to_control_mapping"
            },
        }
        matched_attack_item_count = int(
            npz_pair_mapping["matched_attack_item_count"]
        )
    elif csv_mapping is not None:
        pair_source = {
            "type": "validation_manifest_csv",
            "path": csv_mapping["candidate"]["path"],
            "sha256": csv_mapping["candidate"]["sha256"],
            "mapping": {
                key: value
                for key, value in csv_mapping["mapping"].items()
                if key != "attack_to_control_mapping"
            },
        }
        matched_attack_item_count = int(
            csv_mapping["mapping"]["matched_attack_item_count"]
        )
    else:
        pair_source = None
        matched_attack_item_count = 0

    protocol = protocol_contract()

    matched_control_resolved = pair_source is not None
    count_strata_ready = bool(
        count_strata.get("resolved")
        and count_strata.get("supports_128_per_count")
    )
    sample_capacity_ready = matched_attack_item_count >= 512

    F6_execution_authorized = bool(
        matched_control_resolved
        and count_strata_ready
        and sample_capacity_ready
    )

    identity_path = output_dir / (
        "F6_P0_MATCHED_CONTROL_IDENTITY_ROUTE.json"
    )
    protocol_path = output_dir / (
        "F6_P0_TASK_SPECIFIC_INTEGRATED_GRADIENTS_PROTOCOL.json"
    )
    source_path = output_dir / (
        "F6_P0_FROZEN_EXECUTION_SOURCE_AND_RUNTIME_CONTRACT.json"
    )
    authorization_path = output_dir / (
        "F6_P0_EXECUTION_AUTHORIZATION.json"
    )

    atomic_json(
        identity_path,
        {
            "fresh_npz_inventory": inventory,
            "npz_pair_mapping": {
                key: value
                for key, value in npz_pair_mapping.items()
                if key != "attack_to_control_mapping"
            },
            "csv_candidates": csv_candidates,
            "selected_pair_source": pair_source,
            "matched_control_resolved": matched_control_resolved,
            "matched_attack_item_count": matched_attack_item_count,
            "count_strata": count_strata,
        },
    )
    atomic_json(protocol_path, protocol)
    atomic_json(
        source_path,
        {
            "source_files": source_files,
            "runtime": protocol["integrated_gradients"]["runtime"],
            "input_shape": EXPECTED_INPUT_SHAPE,
            "physical_mask_shape": EXPECTED_MASK_SHAPE,
            "directed_edge_count": EXPECTED_EDGE_COUNT,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "validation_items": EXPECTED_ITEMS,
            "model_class": f5_report["execution"]["model_class"],
            "model_construction": f5_report["execution"]["model_construction"],
        },
    )
    atomic_json(
        authorization_path,
        {
            "F5R_complete": True,
            "F6_authorized_by_F5R": True,
            "matched_control_resolved": matched_control_resolved,
            "count_strata_ready": count_strata_ready,
            "sample_capacity_ready": sample_capacity_ready,
            "actual_F6_execution_authorized": F6_execution_authorized,
            "F7_retraining_ablation_authorized": False,
            "feature_removal_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "sealed_test_access": False,
            "next_stage": (
                "V5_P3_F6_TASK_SPECIFIC_INTEGRATED_GRADIENTS"
                if F6_execution_authorized
                else "V5_P3_F6_P1_MATCHED_CONTROL_IDENTITY_REVIEW"
            ),
        },
    )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "scope": (
            "Freeze task-specific differentiable targets, matched-control "
            "baseline policy, deterministic validation sampling, integration "
            "and completeness gates, source hashes, official D1 runtime, and "
            "the F6 versus F7 authorization boundary."
        ),
        "recovery": {
            "stage": "V5_P3_F6_P0_R1_NUMPY_HEADER_API_RECOVERY",
            "failure_classification": (
                "private_NumPy_header_API_removed_in_installed_NumPy"
            ),
            "failed_private_API": (
                "numpy.lib.format._read_array_header"
            ),
            "replacement": (
                "portable published-format header parser supporting "
                "npy 1.0, 2.0 and 3.0"
            ),
            "validation_feature_tensor_replay_required": False,
            "model_or_checkpoint_replay_required": False,
        },
        "authorization": {
            "F5R_complete": True,
            "F6_authorized_by_F5R": True,
            "matched_control_resolved": matched_control_resolved,
            "count_strata_ready": count_strata_ready,
            "sample_capacity_ready": sample_capacity_ready,
            "actual_F6_execution_authorized": F6_execution_authorized,
            "F7_authorized": False,
        },
        "identity": {
            "selected_pair_source": pair_source,
            "matched_attack_item_count": matched_attack_item_count,
            "count_strata": count_strata,
        },
        "protocol": protocol,
        "decision": {
            "F6_P0_complete": True,
            "actual_F6_execution_authorized": F6_execution_authorized,
            "F7_retraining_ablation_authorized": False,
            "feature_removal_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F6_TASK_SPECIFIC_INTEGRATED_GRADIENTS"
                if F6_execution_authorized
                else "V5_P3_F6_P1_MATCHED_CONTROL_IDENTITY_REVIEW"
            ),
        },
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "validation_feature_tensors_loaded": False,
            "validation_output_identity_payloads_loaded": True,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "thresholds_changed": False,
            "metric_formulas_changed": False,
            "model_weights_changed": False,
        },
        "artifacts": {
            "identity_route": str(identity_path),
            "IG_protocol": str(protocol_path),
            "source_and_runtime_contract": str(source_path),
            "authorization": str(authorization_path),
        },
        "provenance": {
            "F5R_report_sha256": sha256_file(f5r_report_path),
            "F5R_lock_sha256": sha256_file(f5r_lock_path),
            "F5_report_sha256": sha256_file(f5_report_path),
            "F5_lock_sha256": sha256_file(f5_lock_path),
            "F4M_report_sha256": sha256_file(f4m_report_path),
            "F4M_lock_sha256": sha256_file(f4m_lock_path),
            "route_inventory_sha256": sha256_file(route_path),
            "runtime_recovery_sha256": sha256_file(runtime_recovery_path),
            "fresh_certification_sha256": sha256_file(fresh_cert_path),
            "fresh_npz_sha256": sha256_file(fresh_npz),
            "installed_script_sha256": sha256_file(installed_script),
            "identity_route_sha256": sha256_file(identity_path),
            "IG_protocol_sha256": sha256_file(protocol_path),
            "source_and_runtime_contract_sha256": sha256_file(source_path),
            "authorization_sha256": sha256_file(authorization_path),
        },
    }

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    atomic_json(report_path, report)
    atomic_json(
        lock_path,
        {
            "stage": STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(report_path),
            "identity_route_sha256": sha256_file(identity_path),
            "IG_protocol_sha256": sha256_file(protocol_path),
            "source_and_runtime_contract_sha256": sha256_file(source_path),
            "authorization_sha256": sha256_file(authorization_path),
            "actual_F6_execution_authorized": F6_execution_authorized,
            "F7_authorized": False,
            "feature_removal_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "validation_feature_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(
        "recovery_stage="
        "V5_P3_F6_P0_R1_NUMPY_HEADER_API_RECOVERY"
    )
    print(
        "failure_classification="
        "private_NumPy_header_API_removed_in_installed_NumPy"
    )
    print(
        "NPZ_header_parser="
        "portable_published_format_no_private_NumPy_API"
    )
    print(f"classification={CLASSIFICATION}")
    print("F5R_complete=true")
    print("F6_authorized_by_F5R=true")
    print(f"model_class={f5_report['execution']['model_class']}")
    print(f"parameter_count={EXPECTED_PARAMETER_COUNT}")
    print(f"validation_items={EXPECTED_ITEMS}")
    print(f"input_shape={EXPECTED_INPUT_SHAPE}")
    print(f"physical_mask_shape={EXPECTED_MASK_SHAPE}")
    print(f"directed_edge_count={EXPECTED_EDGE_COUNT}")
    print(
        "official_runtime="
        f"{protocol['integrated_gradients']['runtime']}"
    )
    print(f"matched_control_resolved={str(matched_control_resolved).lower()}")
    print(f"matched_attack_item_count={matched_attack_item_count}")
    print(f"selected_pair_source={pair_source}")
    print(f"count_strata={count_strata}")
    print(f"count_strata_ready={str(count_strata_ready).lower()}")
    print(f"sample_capacity_ready={str(sample_capacity_ready).lower()}")
    print("IG_task_count=6")
    for task, row in TASKS.items():
        print(
            f"IG_task_{task}=target={row['target']}:"
            f"scope={row['sample_scope']}"
        )
    print("IG_sample_total=512")
    print("IG_items_per_attacker_count=128")
    print("IG_sample_selection_seed=6101")
    print("IG_steps=64")
    print("IG_gradient_microbatch_size=16")
    print("IG_baseline=matched_control_feature_tensor")
    print("IG_completeness_median_relative_error_max=0.02")
    print("IG_completeness_p95_relative_error_max=0.05")
    print(
        "actual_F6_execution_authorized="
        f"{str(F6_execution_authorized).lower()}"
    )
    print("F7_retraining_ablation_authorized=false")
    print("feature_removal_authorized=false")
    print("hardware_reduction_claim_authorized=false")
    print("model_loaded=false")
    print("checkpoint_loaded=false")
    print("validation_feature_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={report['decision']['next_stage']}")
    print(f"identity_route={identity_path}")
    print(f"IG_protocol={protocol_path}")
    print(f"source_and_runtime_contract={source_path}")
    print(f"authorization={authorization_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
