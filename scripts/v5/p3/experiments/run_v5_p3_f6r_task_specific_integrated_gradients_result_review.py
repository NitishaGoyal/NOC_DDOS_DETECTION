from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = "V5_P3_F6R_TASK_SPECIFIC_INTEGRATED_GRADIENTS_RESULT_REVIEW"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "VALIDATION-EXPLORATORY"

TASKS = (
    "graph",
    "count",
    "source",
    "transit",
    "victim",
    "path",
)

GROUPS = (
    "directional_traffic_volume",
    "inter_flit_timing",
    "queue_activity",
    "buffer_pressure",
    "flow_control_stalls",
)

TASK_TO_F5_METRIC = {
    "graph": "graph_ap",
    "count": "count_active_macro_f1",
    "source": "source_ap",
    "transit": "transit_ap",
    "victim": "victim_ap",
    "path": "path_ap",
}

EXPECTED_SELECTED_ITEMS = 512
EXPECTED_COUNT_STRATA = {"1": 128, "2": 128, "3": 128, "4": 128}
EXPECTED_PARAMETER_COUNT = 60553
EXPECTED_MAPPING_FINGERPRINT = (
    "8b6657a8287c899d97437ed3531f6d43b3cce327f0c45f743a9ae324365d5ae8"
)
EXPECTED_CONSTANT_CHANNELS = {60, 65, 66, 67, 68, 69}
MICROBATCH_SIZE = 16

AGGREGATE_KEYS = (
    "attack_indices",
    "control_indices",
    "target_attack",
    "target_control",
    "target_delta",
    "ig_sum",
    "absolute_completeness_error",
    "relative_completeness_error",
    "channel_signed",
    "channel_absolute",
    "channel_fraction",
    "group_absolute",
    "group_per_channel",
    "group_fraction",
    "router_absolute",
    "time_absolute",
    "total_absolute",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    parser.add_argument("--run-dir", default="")
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


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(rows, f"cannot write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"CSV missing: {path}")
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON missing: {path}")
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
    import re
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def active_run(
    ig_root: Path,
    canonical_report: dict[str, Any],
    explicit: str,
) -> Path:
    report_run = Path(canonical_report["run_directory"]).resolve()

    if explicit:
        run_dir = Path(explicit).expanduser().resolve()
    else:
        pointer = ig_root / "F6_ACTIVE_RUN_PATH.txt"
        require(pointer.is_file(), f"F6 active-run pointer missing: {pointer}")
        run_dir = Path(
            pointer.read_text(encoding="utf-8").strip()
        ).expanduser().resolve()

    require(run_dir.is_dir(), f"F6 run directory missing: {run_dir}")
    require(run_dir == report_run, "active F6 run differs from canonical report")
    return run_dir


def load_npz(path: Path, keys: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    require(path.is_file(), f"NPZ missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        selected = archive.files if keys is None else keys
        return {key: np.asarray(archive[key]) for key in selected}


def compare_exact(
    left: np.ndarray,
    right: np.ndarray,
    label: str,
) -> dict[str, Any]:
    require(left.shape == right.shape, f"{label} shape mismatch")
    exact = bool(np.array_equal(left, right))
    if np.issubdtype(left.dtype, np.number) and np.issubdtype(
        right.dtype,
        np.number,
    ):
        difference = np.abs(
            np.asarray(left, dtype=np.float64)
            - np.asarray(right, dtype=np.float64)
        )
        return {
            "exact_equal": exact,
            "maximum_absolute_difference": float(np.max(difference)),
            "mean_absolute_difference": float(np.mean(difference)),
        }
    return {"exact_equal": exact}


def selected_attempt_directory(
    task_dir: Path,
    points: int,
) -> Path:
    path = task_dir / f"attempt_points_{points}"
    require(path.is_dir(), f"selected attempt directory missing: {path}")
    return path


def verify_selected_attempt(
    *,
    task: str,
    task_result: dict[str, Any],
    task_dir: Path,
) -> dict[str, Any]:
    require(task_result.get("status") == "PASS", f"{task} task not PASS")
    sample_count = int(task_result["sample_count"])
    points = int(task_result["selected_integration_points"])
    selected = task_result["selected_attempt"]

    require(
        selected["completeness"]["pass"] is True,
        f"{task} selected completeness did not pass",
    )
    require(
        float(selected["completeness"]["median_relative_error"]) <= 0.02,
        f"{task} median completeness exceeds 2%",
    )
    require(
        float(selected["completeness"]["p95_relative_error"]) <= 0.05,
        f"{task} p95 completeness exceeds 5%",
    )

    attempt_dir = selected_attempt_directory(task_dir, points)
    attempt_result_path = attempt_dir / "ATTEMPT_RESULT.json"
    attempt_complete_path = attempt_dir / "ATTEMPT_COMPLETE"
    aggregate_path = attempt_dir / "PER_SAMPLE_AGGREGATES.npz"

    require(attempt_complete_path.is_file(), f"{task} attempt marker missing")
    attempt_result = load_json(attempt_result_path)
    require(
        attempt_result == selected,
        f"{task} TASK_RESULT selected attempt differs from ATTEMPT_RESULT",
    )

    expected_hashes = selected["artifact_hashes"]
    artifact_paths = {
        "channel_csv_sha256": Path(selected["artifacts"]["channel_csv"]),
        "group_csv_sha256": Path(selected["artifacts"]["group_csv"]),
        "router_csv_sha256": Path(selected["artifacts"]["router_csv"]),
        "time_csv_sha256": Path(selected["artifacts"]["time_csv"]),
        "per_sample_aggregates_sha256": Path(
            selected["artifacts"]["per_sample_aggregates"]
        ),
    }
    for hash_key, path in artifact_paths.items():
        require(path.is_file(), f"{task} artifact missing: {path}")
        require(
            expected_hashes[hash_key] == sha256_file(path),
            f"{task} artifact hash mismatch: {path}",
        )
    require(
        artifact_paths["per_sample_aggregates_sha256"].resolve()
        == aggregate_path.resolve(),
        f"{task} selected aggregate path differs from attempt directory",
    )

    sample_manifest_path = Path(task_result["sample_manifest"]).resolve()
    sample_manifest = load_json(sample_manifest_path)
    require(
        int(sample_manifest["selected_sample_count"]) == sample_count,
        f"{task} sample-manifest count mismatch",
    )

    aggregate = load_npz(aggregate_path, AGGREGATE_KEYS)
    require(
        aggregate["attack_indices"].shape == (sample_count,),
        f"{task} aggregate sample count changed",
    )
    require(
        aggregate["attack_indices"].tolist()
        == [int(value) for value in sample_manifest["attack_indices"]],
        f"{task} attack-index order differs from sample manifest",
    )
    require(
        aggregate["control_indices"].tolist()
        == [int(value) for value in sample_manifest["control_indices"]],
        f"{task} control-index order differs from sample manifest",
    )

    batches_dir = attempt_dir / "batches"
    expected_batch_count = math.ceil(sample_count / MICROBATCH_SIZE)
    reconstructed_parts: dict[str, list[np.ndarray]] = {
        key: [] for key in AGGREGATE_KEYS
    }
    batch_hashes = []

    for batch_number in range(expected_batch_count):
        stem = f"batch_{batch_number:04d}"
        npz_path = batches_dir / f"{stem}.npz"
        manifest_path = batches_dir / f"{stem}.json"
        marker_path = batches_dir / f"{stem}.complete"

        require(npz_path.is_file(), f"{task} batch NPZ missing: {npz_path}")
        require(
            manifest_path.is_file(),
            f"{task} batch manifest missing: {manifest_path}",
        )
        require(
            marker_path.is_file(),
            f"{task} batch marker missing: {marker_path}",
        )

        manifest = load_json(manifest_path)
        require(manifest.get("status") == "PASS", f"{task} batch not PASS")
        require(manifest.get("task") == task, f"{task} batch task mismatch")
        require(
            int(manifest.get("integration_points")) == points,
            f"{task} batch integration-point mismatch",
        )
        require(
            manifest.get("npz_sha256") == sha256_file(npz_path),
            f"{task} batch NPZ hash mismatch",
        )

        batch = load_npz(npz_path, AGGREGATE_KEYS)
        for key in AGGREGATE_KEYS:
            reconstructed_parts[key].append(batch[key])

        batch_hashes.append({
            "batch_number": batch_number,
            "npz": str(npz_path),
            "npz_sha256": manifest["npz_sha256"],
            "sample_count": int(manifest["sample_count"]),
        })

    reconstructed = {
        key: np.concatenate(parts, axis=0)
        for key, parts in reconstructed_parts.items()
    }
    aggregate_comparison = {
        key: compare_exact(reconstructed[key], aggregate[key], f"{task}:{key}")
        for key in AGGREGATE_KEYS
    }
    require(
        all(row["exact_equal"] for row in aggregate_comparison.values()),
        f"{task} aggregate was not reconstructed exactly from selected batches",
    )

    relative = np.asarray(
        aggregate["relative_completeness_error"],
        dtype=np.float64,
    )
    absolute = np.asarray(
        aggregate["absolute_completeness_error"],
        dtype=np.float64,
    )
    recomputed = {
        "median_relative_error": float(np.median(relative)),
        "p95_relative_error": float(np.quantile(relative, 0.95)),
        "maximum_relative_error": float(np.max(relative)),
        "mean_absolute_error": float(np.mean(absolute)),
    }
    for key, value in recomputed.items():
        require(
            abs(value - float(selected["completeness"][key])) <= 1e-8,
            f"{task} completeness recomputation mismatch: {key}",
        )

    group_fraction = np.asarray(aggregate["group_fraction"], dtype=np.float64)
    channel_fraction = np.asarray(
        aggregate["channel_fraction"],
        dtype=np.float64,
    )
    group_absolute = np.asarray(aggregate["group_absolute"], dtype=np.float64)
    total_absolute = np.asarray(aggregate["total_absolute"], dtype=np.float64)

    group_fraction_max_deviation = float(
        np.max(np.abs(group_fraction.sum(axis=1) - 1.0))
    )
    channel_fraction_max_deviation = float(
        np.max(np.abs(channel_fraction.sum(axis=1) - 1.0))
    )
    group_total_relative_error = np.abs(
        group_absolute.sum(axis=1) - total_absolute
    ) / np.maximum(total_absolute, 1e-12)
    group_total_max_relative_error = float(
        np.max(group_total_relative_error)
    )

    require(
        group_fraction_max_deviation <= 2e-5,
        f"{task} group fractions do not sum to one",
    )
    require(
        channel_fraction_max_deviation <= 2e-5,
        f"{task} channel fractions do not sum to one",
    )
    require(
        group_total_max_relative_error <= 2e-5,
        f"{task} group absolute attribution does not reconstruct total",
    )

    group_rows = read_csv(artifact_paths["group_csv_sha256"])
    channel_rows = read_csv(artifact_paths["channel_csv_sha256"])
    router_rows = read_csv(artifact_paths["router_csv_sha256"])
    time_rows = read_csv(artifact_paths["time_csv_sha256"])

    require(len(group_rows) == 5, f"{task} group CSV row count changed")
    require(len(channel_rows) == 70, f"{task} channel CSV row count changed")
    require(len(router_rows) == 16, f"{task} router CSV row count changed")
    require(len(time_rows) == 32, f"{task} time CSV row count changed")

    result_group_rows = selected["group_ranking_by_total_absolute"]
    require(
        [row["group"] for row in result_group_rows]
        == [row["group"] for row in group_rows],
        f"{task} group ranking differs between JSON and CSV",
    )
    for left, right in zip(result_group_rows, group_rows):
        require(
            abs(
                float(left["mean_fraction_of_total_absolute"])
                - float(right["mean_fraction_of_total_absolute"])
            )
            <= 1e-9,
            f"{task} group fraction differs between JSON and CSV",
        )

    return {
        "task": task,
        "sample_count": sample_count,
        "selected_integration_points": points,
        "fallback_128_points_used": bool(
            task_result.get("fallback_128_points_used", False)
        ),
        "recovery_256_points_used": bool(
            task_result.get("recovery_256_points_used", False)
        ),
        "completeness": recomputed,
        "leader_group": result_group_rows[0]["group"],
        "leader_fraction": float(
            result_group_rows[0]["mean_fraction_of_total_absolute"]
        ),
        "group_rows": result_group_rows,
        "top_channels": selected["top_20_channels_by_absolute"],
        "aggregate_comparison": aggregate_comparison,
        "group_fraction_max_sum_deviation": group_fraction_max_deviation,
        "channel_fraction_max_sum_deviation": channel_fraction_max_deviation,
        "group_total_max_relative_error": group_total_max_relative_error,
        "batch_count": expected_batch_count,
        "batch_hashes": batch_hashes,
        "attempt_result_sha256": sha256_file(attempt_result_path),
        "aggregate_sha256": sha256_file(aggregate_path),
        "sample_manifest_sha256": sha256_file(sample_manifest_path),
    }


def rank_map(rows: list[dict[str, Any]], rank_key: str) -> dict[str, int]:
    return {
        str(row["group"]): int(row[rank_key])
        for row in rows
    }


def spearman_no_ties(
    left: dict[str, int],
    right: dict[str, int],
) -> float:
    require(set(left) == set(right) == set(GROUPS), "rank groups changed")
    n = len(GROUPS)
    sum_squared = sum(
        (left[group] - right[group]) ** 2
        for group in GROUPS
    )
    return float(1.0 - 6.0 * sum_squared / (n * (n * n - 1)))


def top_k_overlap(
    left: dict[str, int],
    right: dict[str, int],
    k: int,
) -> int:
    left_top = {group for group, rank in left.items() if rank <= k}
    right_top = {group for group, rank in right.items() if rank <= k}
    return len(left_top & right_top)


def parse_constant_channel_indices(path: Path) -> set[int]:
    rows = read_csv(path)
    require(rows, "F2R constant-channel CSV is empty")
    fields = list(rows[0])
    normalized = {normalize(field): field for field in fields}

    # F2R's canonical schema uses `dynamic70_index`. Earlier F6R code
    # recognized only older/generic aliases, so the review stopped before any
    # attribution interpretation. Keep all known aliases explicit.
    index_candidates = (
        "dynamic70_index",
        "dynamic70_channel_index",
        "channel_index",
        "feature_index",
        "index",
    )
    index_field = next(
        (
            normalized[candidate]
            for candidate in index_candidates
            if candidate in normalized
        ),
        None,
    )
    require(
        index_field is not None,
        f"could not resolve constant-channel index field: {fields}",
    )

    constant_flag_candidates = (
        "constant_zero",
        "is_constant_zero",
    )
    constant_flag_field = next(
        (
            normalized[candidate]
            for candidate in constant_flag_candidates
            if candidate in normalized
        ),
        None,
    )
    classification_field = normalized.get("classification")

    def truthy(value: Any) -> bool:
        token = normalize(str(value))
        return token in {
            "1",
            "true",
            "yes",
            "y",
            "constant_zero",
            "constant",
        }

    selected_rows = []
    for row in rows:
        if constant_flag_field is not None:
            if truthy(row.get(constant_flag_field)):
                selected_rows.append(row)
        elif classification_field is not None:
            classification = normalize(
                str(row.get(classification_field, ""))
            )
            if "constant_zero" in classification:
                selected_rows.append(row)
        else:
            # The canonical F2R_CONSTANT_CHANNEL_REVIEW.csv is itself a
            # constant-only review table. Preserve compatibility with that
            # compact representation.
            selected_rows.append(row)

    require(
        selected_rows,
        "F2R constant-channel CSV contains no constant-zero rows",
    )

    indices = set()
    for row in selected_rows:
        raw = str(row.get(index_field, "")).strip()
        require(raw != "", f"empty constant-channel index in {path}")
        try:
            numeric = float(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"invalid constant-channel index {raw!r} in {path}"
            ) from exc
        integer = int(numeric)
        require(
            abs(numeric - integer) <= 1e-9,
            f"non-integral constant-channel index {raw!r} in {path}",
        )
        require(
            0 <= integer < 70,
            f"Dynamic70 constant-channel index out of range: {integer}",
        )
        indices.add(integer)

    require(
        len(indices) == len(selected_rows),
        "duplicate Dynamic70 constant-channel index detected",
    )
    return indices


def make_figures(
    output_dir: Path,
    task_group_rows: list[dict[str, Any]],
    completeness_rows: list[dict[str, Any]],
) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    figures = []

    matrix = np.zeros((len(TASKS), len(GROUPS)), dtype=np.float64)
    for row in task_group_rows:
        matrix[TASKS.index(row["task"]), GROUPS.index(row["group"])] = float(
            row["mean_fraction_of_total_absolute"]
        )

    heatmap_path = output_dir / "F6R_TASK_BY_GROUP_ATTRIBUTION_FRACTION.png"
    figure = plt.figure(figsize=(12, 6))
    axis = figure.add_subplot(111)
    image = axis.imshow(matrix, aspect="auto")
    axis.set_xticks(range(len(GROUPS)))
    axis.set_xticklabels(
        [group.replace("_", "\n") for group in GROUPS]
    )
    axis.set_yticks(range(len(TASKS)))
    axis.set_yticklabels(TASKS)
    axis.set_title("F6 task-specific matched-control IG attribution")
    figure.colorbar(image, ax=axis, label="Mean fraction of absolute attribution")
    figure.tight_layout()
    figure.savefig(heatmap_path, dpi=180)
    plt.close(figure)
    figures.append(heatmap_path)

    completeness_path = output_dir / "F6R_TASK_COMPLETENESS_REVIEW.png"
    figure = plt.figure(figsize=(10, 5))
    axis = figure.add_subplot(111)
    x = np.arange(len(TASKS))
    median = [
        next(
            row["median_relative_error"]
            for row in completeness_rows
            if row["task"] == task
        )
        for task in TASKS
    ]
    p95 = [
        next(
            row["p95_relative_error"]
            for row in completeness_rows
            if row["task"] == task
        )
        for task in TASKS
    ]
    width = 0.35
    axis.bar(x - width / 2, median, width, label="Median")
    axis.bar(x + width / 2, p95, width, label="p95")
    axis.axhline(0.02, linewidth=1, linestyle="--", label="Median gate")
    axis.axhline(0.05, linewidth=1, linestyle=":", label="p95 gate")
    axis.set_xticks(x)
    axis.set_xticklabels(TASKS)
    axis.set_ylabel("Relative completeness error")
    axis.set_title("F6 selected-attempt completeness")
    axis.legend()
    figure.tight_layout()
    figure.savefig(completeness_path, dpi=180)
    plt.close(figure)
    figures.append(completeness_path)

    return figures


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")

    feature_root = repo / (
        "reports/v5/p3_experiments/f0_d70_feature_study"
    )
    ig_root = feature_root / "integrated_gradients"
    f5r_root = feature_root / "permutation_review"
    schema_root = feature_root / "schema_audit"

    f6_report_path = ig_root / (
        "V5_P3_F6_TASK_SPECIFIC_INTEGRATED_GRADIENTS_REPORT.json"
    )
    f6_lock_path = ig_root / (
        "V5_P3_F6_TASK_SPECIFIC_INTEGRATED_GRADIENTS_LOCK.json"
    )
    f6_report, f6_lock = verify_report_lock(f6_report_path, f6_lock_path)

    require(f6_lock.get("F6_complete") is True, "canonical F6 incomplete")
    require(f6_lock.get("F6R_authorized") is True, "F6R not authorized")
    require(f6_lock.get("F7_authorized") is False, "F7 already authorized")
    require(
        f6_lock.get("feature_removal_authorized") is False,
        "feature removal already authorized",
    )
    require(
        f6_lock.get("sealed_test_tensors_loaded") is False,
        "sealed-test access detected",
    )

    require(
        int(f6_report["parameter_count"]) == EXPECTED_PARAMETER_COUNT,
        "F6 model parameter count changed",
    )
    require(
        int(f6_report["selected_item_count"]) == EXPECTED_SELECTED_ITEMS,
        "F6 selected item count changed",
    )
    require(
        f6_report["attacker_count_strata"] == EXPECTED_COUNT_STRATA,
        "F6 attacker-count strata changed",
    )

    run_dir = active_run(ig_root, f6_report, args.run_dir)
    run_report_path = Path(f6_report["run_report"]).resolve()
    run_lock_path = Path(f6_report["run_lock"]).resolve()
    run_report, run_lock = verify_report_lock(run_report_path, run_lock_path)
    require(run_lock.get("F6_complete") is True, "run-level F6 incomplete")
    require(run_lock.get("F6R_authorized") is True, "run-level F6R held")
    require(
        sha256_file(run_report_path) == f6_report["run_report_sha256"],
        "canonical F6 report references a changed run report",
    )
    require(
        sha256_file(run_lock_path) == f6_report["run_lock_sha256"],
        "canonical F6 report references a changed run lock",
    )

    selection_path = run_dir / "F6_SELECTED_ATTACK_CONTROL_ITEMS.json"
    selection = load_json(selection_path)
    require(
        int(selection["selected_item_count"]) == EXPECTED_SELECTED_ITEMS,
        "frozen F6 sample selection changed",
    )
    require(
        selection["attacker_count_strata"] == EXPECTED_COUNT_STRATA,
        "frozen selection strata changed",
    )

    p3_mapping_path = ig_root / (
        "F6_P3_FROZEN_ATTACK_TO_CONTROL_MAPPING.json"
    )
    p3_mapping = load_json(p3_mapping_path)
    require(
        p3_mapping["mapping_fingerprint_sha256"]
        == EXPECTED_MAPPING_FINGERPRINT,
        "F6-P3 mapping fingerprint changed",
    )

    r1_report_path = ig_root / (
        "V5_P3_F6_R1_COUNT_COMPLETENESS_FAILURE_DIAGNOSTIC_REPORT.json"
    )
    r1_lock_path = ig_root / (
        "V5_P3_F6_R1_COUNT_COMPLETENESS_FAILURE_DIAGNOSTIC_LOCK.json"
    )
    r2_report_path = ig_root / (
        "V5_P3_F6_R2_COUNT_256_POINT_TARGETED_RETRY_REPORT.json"
    )
    r2_lock_path = ig_root / (
        "V5_P3_F6_R2_COUNT_256_POINT_TARGETED_RETRY_LOCK.json"
    )
    r1_report, r1_lock = verify_report_lock(r1_report_path, r1_lock_path)
    r2_report, r2_lock = verify_report_lock(r2_report_path, r2_lock_path)

    require(
        r1_lock.get("failure_classification")
        == "COUNT_IG_DISCRETIZATION_NONCONVERGENCE_AT_128_POINTS",
        "count R1 classification changed",
    )
    require(
        r2_lock.get("count_completeness_pass") is True,
        "count 256-point recovery did not pass",
    )
    require(
        r2_lock.get("count_task_complete") is True,
        "count task was not completed by R2",
    )
    require(
        r2_lock.get("completeness_gate_changed") is False,
        "count completeness gate changed",
    )

    task_reviews = {}
    completeness_rows = []
    task_group_rows = []
    top_channel_rows = []

    for task in TASKS:
        task_dir = run_dir / "tasks" / task
        require(
            (task_dir / "TASK_COMPLETE").is_file(),
            f"{task} TASK_COMPLETE missing",
        )
        task_result_path = task_dir / "TASK_RESULT.json"
        task_result = load_json(task_result_path)
        review = verify_selected_attempt(
            task=task,
            task_result=task_result,
            task_dir=task_dir,
        )
        task_reviews[task] = review

        if task == "count":
            require(
                review["selected_integration_points"] == 256,
                "count did not select the certified 256-point attempt",
            )
            require(
                review["recovery_256_points_used"] is True,
                "count task result does not record 256-point recovery",
            )
        else:
            require(
                review["selected_integration_points"] in (64, 128),
                f"{task} selected unexpected integration points",
            )

        completeness_rows.append({
            "task": task,
            "sample_count": review["sample_count"],
            "selected_integration_points": review[
                "selected_integration_points"
            ],
            "median_relative_error": review["completeness"][
                "median_relative_error"
            ],
            "p95_relative_error": review["completeness"][
                "p95_relative_error"
            ],
            "maximum_relative_error": review["completeness"][
                "maximum_relative_error"
            ],
            "mean_absolute_error": review["completeness"][
                "mean_absolute_error"
            ],
            "completeness_pass": True,
            "leader_group": review["leader_group"],
            "leader_group_fraction": review["leader_fraction"],
        })

        for row in review["group_rows"]:
            task_group_rows.append({
                "task": task,
                "sample_count": review["sample_count"],
                "selected_integration_points": review[
                    "selected_integration_points"
                ],
                **row,
            })

        for row in review["top_channels"]:
            top_channel_rows.append({
                "task": task,
                **row,
            })

    require(
        task_reviews["graph"]["sample_count"] == 512,
        "graph sample count changed",
    )
    require(
        task_reviews["count"]["sample_count"] == 512,
        "count sample count changed",
    )
    require(
        1 <= task_reviews["transit"]["sample_count"] <= 512,
        "transit sample count invalid",
    )

    f5r_report_path = f5r_root / (
        "V5_P3_F5R_GROUP_BLOCK_PERMUTATION_RESULT_REVIEW_REPORT.json"
    )
    f5r_lock_path = f5r_root / (
        "V5_P3_F5R_GROUP_BLOCK_PERMUTATION_RESULT_REVIEW_LOCK.json"
    )
    f5r_report, f5r_lock = verify_report_lock(
        f5r_report_path,
        f5r_lock_path,
    )
    require(f5r_lock.get("F5R_complete") is True, "F5R incomplete")

    f5_task_rankings_path = f5r_root / (
        "F5R_TASK_SPECIFIC_GROUP_RANKINGS.json"
    )
    f5_task_rankings = load_json(f5_task_rankings_path)
    f5_selection_rows = read_csv(
        f5r_root / "F5R_SELECTION_SCORE_RANKING.csv"
    )

    f2r_constant_path = schema_root / "F2R_CONSTANT_CHANNEL_REVIEW.csv"
    constant_channels = parse_constant_channel_indices(f2r_constant_path)
    require(
        constant_channels == EXPECTED_CONSTANT_CHANNELS,
        f"F2R constant channels changed: {sorted(constant_channels)}",
    )

    concordance_rows = []
    task_interpretations = []

    for task in TASKS:
        f5_metric = TASK_TO_F5_METRIC[task]
        f5_rows = f5_task_rankings[f5_metric]
        f6_rows = task_reviews[task]["group_rows"]

        f5_ranks = rank_map(f5_rows, "rank")
        f6_ranks = rank_map(f6_rows, "rank_by_total_absolute")
        rho = spearman_no_ties(f5_ranks, f6_ranks)
        top1_match = (
            min(f5_ranks, key=f5_ranks.get)
            == min(f6_ranks, key=f6_ranks.get)
        )
        top2_overlap = top_k_overlap(f5_ranks, f6_ranks, 2)

        concordance_rows.append({
            "task": task,
            "F5_metric": f5_metric,
            "F5_leader": min(f5_ranks, key=f5_ranks.get),
            "F6_leader": min(f6_ranks, key=f6_ranks.get),
            "top1_match": top1_match,
            "top2_overlap_of_2": top2_overlap,
            "spearman_rank_correlation": rho,
            "interpretation": (
                "F5 and F6 answer different questions: validation-wide "
                "performance dependence after group shuffling versus local "
                "matched-control path attribution for the frozen checkpoint."
            ),
        })

        task_interpretations.append({
            "task": task,
            "sample_count": task_reviews[task]["sample_count"],
            "integration_points": task_reviews[task][
                "selected_integration_points"
            ],
            "F6_leading_group": task_reviews[task]["leader_group"],
            "F6_leading_group_fraction": task_reviews[task][
                "leader_fraction"
            ],
            "F5_permutation_leading_group": min(f5_ranks, key=f5_ranks.get),
            "top1_concordance": top1_match,
            "top2_overlap": top2_overlap,
            "spearman_rank_correlation": rho,
        })

    cross_group_rows = []
    f5_overall_rank = {
        row["group"]: int(row["rank"])
        for row in f5_selection_rows
    }

    for group in GROUPS:
        fractions = [
            float(
                next(
                    row["mean_fraction_of_total_absolute"]
                    for row in task_reviews[task]["group_rows"]
                    if row["group"] == group
                )
            )
            for task in TASKS
        ]
        f6_ranks = [
            int(
                next(
                    row["rank_by_total_absolute"]
                    for row in task_reviews[task]["group_rows"]
                    if row["group"] == group
                )
            )
            for task in TASKS
        ]
        cross_group_rows.append({
            "group": group,
            "F5_overall_selection_rank": f5_overall_rank[group],
            "F6_mean_fraction_of_total_absolute": float(np.mean(fractions)),
            "F6_min_fraction": float(np.min(fractions)),
            "F6_max_fraction": float(np.max(fractions)),
            "F6_mean_rank": float(np.mean(f6_ranks)),
            "F6_task_leader_count": int(np.sum(np.asarray(f6_ranks) == 1)),
            "F6_task_last_count": int(np.sum(np.asarray(f6_ranks) == 5)),
            "F2R_constant_channel_count": (
                len(EXPECTED_CONSTANT_CHANNELS)
                if group == "flow_control_stalls"
                else 0
            ),
        })
    cross_group_rows.sort(
        key=lambda row: row["F6_mean_fraction_of_total_absolute"],
        reverse=True,
    )
    for rank, row in enumerate(cross_group_rows, start=1):
        row["F6_cross_task_mean_fraction_rank"] = rank

    flow_row = next(
        row for row in cross_group_rows
        if row["group"] == "flow_control_stalls"
    )
    flow_control_reduction_hypothesis_supported = bool(
        flow_row["F5_overall_selection_rank"] == 5
        and flow_row["F6_task_last_count"] >= 4
        and flow_row["F2R_constant_channel_count"] == 6
    )

    completeness_csv = output_dir / "F6R_TASK_COMPLETENESS_REVIEW.csv"
    group_csv = output_dir / "F6R_TASK_GROUP_ATTRIBUTION_REVIEW.csv"
    channel_csv = output_dir / "F6R_TASK_TOP_CHANNEL_REVIEW.csv"
    concordance_csv = output_dir / "F6R_F5_F6_RANK_CONCORDANCE.csv"
    cross_group_csv = output_dir / "F6R_CROSS_METHOD_GROUP_TRIANGULATION.csv"
    artifact_json = output_dir / "F6R_ARTIFACT_AND_COMPLETENESS_CERTIFICATION.json"
    interpretation_json = output_dir / (
        "F6R_FEATURE_INTERPRETATION_AND_F7_DECISION.json"
    )

    write_csv(completeness_csv, completeness_rows)
    write_csv(group_csv, task_group_rows)
    write_csv(channel_csv, top_channel_rows)
    write_csv(concordance_csv, concordance_rows)
    write_csv(cross_group_csv, cross_group_rows)

    figures = make_figures(
        output_dir,
        task_group_rows,
        completeness_rows,
    )

    artifact_certification = {
        "F6_run_directory": str(run_dir),
        "selected_pair_count": selection["selected_item_count"],
        "attacker_count_strata": selection["attacker_count_strata"],
        "mapping_fingerprint": p3_mapping[
            "mapping_fingerprint_sha256"
        ],
        "model_parameter_count": f6_report["parameter_count"],
        "task_reviews": task_reviews,
        "count_recovery": {
            "R1_classification": r1_lock["failure_classification"],
            "R2_count_completeness_pass": r2_lock[
                "count_completeness_pass"
            ],
            "R2_count_task_complete": r2_lock["count_task_complete"],
            "R2_completeness_gate_changed": r2_lock[
                "completeness_gate_changed"
            ],
            "selected_count_points": task_reviews["count"][
                "selected_integration_points"
            ],
        },
        "all_six_tasks_complete": True,
        "all_selected_attempts_pass_completeness": True,
        "all_selected_batch_hashes_verified": True,
        "all_selected_aggregates_exactly_reconstructed": True,
        "sealed_test_tensors_loaded": False,
    }
    atomic_json(artifact_json, artifact_certification)

    paper_safe_claim = (
        "Task-specific integrated gradients along frozen matched "
        "control-to-attack paths indicate that the checkpoint draws most "
        "strongly on inter-flit timing for graph detection, attacker-count "
        "prediction, source localization and victim localization, while "
        "buffer-pressure evidence is strongest for transit and path "
        "localization. These attribution results describe the frozen model's "
        "local pathwise dependence and do not establish causal necessity or "
        "authorize feature deletion."
    )

    interpretation = {
        "F6_task_interpretations": task_interpretations,
        "F5_F6_method_boundary": {
            "F5": (
                "Validation-wide performance degradation after shuffling an "
                "entire macro-group across samples."
            ),
            "F6": (
                "Local absolute integrated-gradient attribution along a "
                "matched control-to-attack input path on a frozen stratified "
                "subset."
            ),
            "leader_match_required": False,
            "disagreement_is_not_automatically_a_contradiction": True,
        },
        "cross_method_group_triangulation": cross_group_rows,
        "flow_control_reduction_hypothesis_supported": (
            flow_control_reduction_hypothesis_supported
        ),
        "flow_control_reduction_hypothesis_note": (
            "A low-impact reduction hypothesis is suitable for retrained "
            "ablation only when F2R constant-channel evidence, F5 grouped "
            "permutation and F6 matched-control attribution all point in the "
            "same direction. It still does not authorize deletion."
        ),
        "paper_safe_claim": paper_safe_claim,
        "claim_boundaries": {
            "task_specific_feature_importance_wording_authorized": True,
            "causal_feature_necessity_claim_authorized": False,
            "feature_removal_authorized": False,
            "compact_interface_equivalence_authorized": False,
            "hardware_area_power_latency_savings_claim_authorized": False,
        },
        "decision": {
            "F6R_complete": True,
            "F7_protocol_preflight_authorized": True,
            "actual_F7_retraining_authorized": False,
            "F7_primary_experiments": [
                "same-width normalized-neutral masking of directional traffic volume",
                "same-width normalized-neutral masking of inter-flit timing",
                "same-width normalized-neutral masking of queue activity",
                "same-width normalized-neutral masking of buffer pressure",
                "same-width normalized-neutral masking of flow-control stalls",
            ],
            "F7_priority_hypothesis": (
                "flow_control_stalls_first"
                if flow_control_reduction_hypothesis_supported
                else "no_single_group_priority_frozen"
            ),
            "F7_seed_policy": (
                "seed 107 for all five primary ablations; seeds 117 and 127 "
                "only for the two most decision-relevant groups after the "
                "primary matrix is reviewed"
            ),
            "feature_removal_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F7_P0_SAME_WIDTH_RETRAINED_GROUP_ABLATION_PREFLIGHT"
            ),
        },
    }
    atomic_json(interpretation_json, interpretation)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "scope": (
            "Verify canonical and run-level F6 report-lock chains; certify "
            "the frozen 512-pair mapping and count recovery lineage; verify "
            "all six task completion markers, selected attempt artifacts and "
            "batch hashes; reconstruct selected aggregate outputs exactly; "
            "recompute completeness; review task/group/channel attribution; "
            "compare F5 permutation ranks with F6 matched-control IG ranks; "
            "and authorize only the F7 protocol preflight."
        ),
        "recovery": {
            "stage": (
                "V5_P3_F6R_R1_CONSTANT_CHANNEL_SCHEMA_ALIAS_RECOVERY"
            ),
            "failure_classification": (
                "F2R_constant_channel_index_schema_alias_not_recognized"
            ),
            "observed_canonical_index_field": "dynamic70_index",
            "parser_change": (
                "recognize dynamic70_index and validate constant_zero rows, "
                "integer indices, range and uniqueness"
            ),
            "scientific_protocol_changed": False,
            "saved_F6_outputs_recomputed": False,
        },
        "verification": {
            "F6_complete": True,
            "F6R_authorized_by_F6": True,
            "selected_pair_count": EXPECTED_SELECTED_ITEMS,
            "mapping_fingerprint": EXPECTED_MAPPING_FINGERPRINT,
            "model_parameter_count": EXPECTED_PARAMETER_COUNT,
            "task_count": len(TASKS),
            "all_task_markers_verified": True,
            "all_selected_batch_hashes_verified": True,
            "all_selected_aggregates_exactly_reconstructed": True,
            "all_selected_completeness_gates_pass": True,
            "count_256_point_recovery_verified": True,
            "F2R_constant_channels_verified": sorted(constant_channels),
        },
        "results": {
            "task_completeness": completeness_rows,
            "task_interpretations": task_interpretations,
            "cross_method_group_triangulation": cross_group_rows,
            "flow_control_reduction_hypothesis_supported": (
                flow_control_reduction_hypothesis_supported
            ),
            "paper_safe_claim": paper_safe_claim,
        },
        "decision": interpretation["decision"],
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "validation_feature_tensors_loaded": False,
            "saved_F6_output_artifacts_loaded": True,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "model_weights_changed": False,
            "IG_targets_changed": False,
            "completeness_gates_changed": False,
            "feature_removal_authorized": False,
        },
        "artifacts": {
            "task_completeness_csv": str(completeness_csv),
            "task_group_attribution_csv": str(group_csv),
            "task_top_channel_csv": str(channel_csv),
            "F5_F6_rank_concordance_csv": str(concordance_csv),
            "cross_method_group_triangulation_csv": str(cross_group_csv),
            "artifact_certification": str(artifact_json),
            "interpretation_and_F7_decision": str(interpretation_json),
            "figures": [str(path) for path in figures],
        },
        "provenance": {
            "canonical_F6_report_sha256": sha256_file(f6_report_path),
            "canonical_F6_lock_sha256": sha256_file(f6_lock_path),
            "run_F6_report_sha256": sha256_file(run_report_path),
            "run_F6_lock_sha256": sha256_file(run_lock_path),
            "F6_selection_sha256": sha256_file(selection_path),
            "F6_P3_mapping_sha256": sha256_file(p3_mapping_path),
            "count_R1_report_sha256": sha256_file(r1_report_path),
            "count_R1_lock_sha256": sha256_file(r1_lock_path),
            "count_R2_report_sha256": sha256_file(r2_report_path),
            "count_R2_lock_sha256": sha256_file(r2_lock_path),
            "F5R_report_sha256": sha256_file(f5r_report_path),
            "F5R_lock_sha256": sha256_file(f5r_lock_path),
            "F5_task_rankings_sha256": sha256_file(f5_task_rankings_path),
            "F2R_constant_channel_csv_sha256": sha256_file(
                f2r_constant_path
            ),
            "installed_script_sha256": sha256_file(installed_script),
            "task_completeness_csv_sha256": sha256_file(completeness_csv),
            "task_group_attribution_csv_sha256": sha256_file(group_csv),
            "task_top_channel_csv_sha256": sha256_file(channel_csv),
            "F5_F6_rank_concordance_csv_sha256": sha256_file(
                concordance_csv
            ),
            "cross_method_group_triangulation_csv_sha256": sha256_file(
                cross_group_csv
            ),
            "artifact_certification_sha256": sha256_file(artifact_json),
            "interpretation_and_F7_decision_sha256": sha256_file(
                interpretation_json
            ),
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
            "task_completeness_csv_sha256": sha256_file(completeness_csv),
            "task_group_attribution_csv_sha256": sha256_file(group_csv),
            "task_top_channel_csv_sha256": sha256_file(channel_csv),
            "F5_F6_rank_concordance_csv_sha256": sha256_file(
                concordance_csv
            ),
            "cross_method_group_triangulation_csv_sha256": sha256_file(
                cross_group_csv
            ),
            "artifact_certification_sha256": sha256_file(artifact_json),
            "interpretation_and_F7_decision_sha256": sha256_file(
                interpretation_json
            ),
            "F6R_complete": True,
            "F7_protocol_preflight_authorized": True,
            "actual_F7_retraining_authorized": False,
            "feature_removal_authorized": False,
            "compact_interface_equivalence_authorized": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(
        "recovery_stage="
        "V5_P3_F6R_R1_CONSTANT_CHANNEL_SCHEMA_ALIAS_RECOVERY"
    )
    print(
        "failure_classification="
        "F2R_constant_channel_index_schema_alias_not_recognized"
    )
    print("F2R_index_field_alias_recovered=dynamic70_index")
    print(f"classification={CLASSIFICATION}")
    print(f"run_directory={run_dir}")
    print("F6_complete=true")
    print("all_six_task_markers_verified=true")
    print("all_selected_batch_hashes_verified=true")
    print("all_selected_aggregates_exactly_reconstructed=true")
    print("all_selected_completeness_gates_pass=true")
    print("count_256_point_recovery_verified=true")
    print(f"mapping_fingerprint={EXPECTED_MAPPING_FINGERPRINT}")
    print(f"F2R_constant_channels={sorted(constant_channels)}")

    for row in completeness_rows:
        print(
            f"reviewed_F6_task={row['task']}:"
            f"samples={row['sample_count']}:"
            f"points={row['selected_integration_points']}:"
            f"median_rel={row['median_relative_error']}:"
            f"p95_rel={row['p95_relative_error']}:"
            f"leader={row['leader_group']}:"
            f"leader_fraction={row['leader_group_fraction']}"
        )

    for row in concordance_rows:
        print(
            f"F5_F6_concordance={row['task']}:"
            f"F5_leader={row['F5_leader']}:"
            f"F6_leader={row['F6_leader']}:"
            f"top1_match={str(row['top1_match']).lower()}:"
            f"top2_overlap={row['top2_overlap_of_2']}:"
            f"spearman={row['spearman_rank_correlation']}"
        )

    for row in cross_group_rows:
        print(
            f"cross_method_group={row['group']}:"
            f"F5_rank={row['F5_overall_selection_rank']}:"
            f"F6_mean_fraction={row['F6_mean_fraction_of_total_absolute']}:"
            f"F6_mean_rank={row['F6_mean_rank']}:"
            f"F6_last_count={row['F6_task_last_count']}:"
            f"F2R_constant_count={row['F2R_constant_channel_count']}"
        )

    print(
        "flow_control_reduction_hypothesis_supported="
        f"{str(flow_control_reduction_hypothesis_supported).lower()}"
    )
    print("task_specific_feature_importance_wording_authorized=true")
    print("causal_feature_necessity_claim_authorized=false")
    print("F7_protocol_preflight_authorized=true")
    print("actual_F7_retraining_authorized=false")
    print("feature_removal_authorized=false")
    print("compact_interface_equivalence_authorized=false")
    print("hardware_reduction_claim_authorized=false")
    print("sealed_test_tensors_loaded=false")
    print(
        "next_stage="
        "V5_P3_F7_P0_SAME_WIDTH_RETRAINED_GROUP_ABLATION_PREFLIGHT"
    )
    print(f"task_completeness_csv={completeness_csv}")
    print(f"task_group_attribution_csv={group_csv}")
    print(f"task_top_channel_csv={channel_csv}")
    print(f"F5_F6_rank_concordance_csv={concordance_csv}")
    print(f"cross_method_group_triangulation_csv={cross_group_csv}")
    print(f"artifact_certification={artifact_json}")
    print(f"interpretation_and_F7_decision={interpretation_json}")
    for path in figures:
        print(f"figure={path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
