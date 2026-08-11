#!/usr/bin/env python3
"""Audit a chronological V3 temporal-graph dataset.

Subcommands:
  verify-extraction  Compare an NPZ archive with an extracted .npy directory.
  compare            Compare an old and chronological extracted dataset.

The script refuses to overwrite an existing output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EXPECTED_ARRAYS = (
    "x",
    "y_graph",
    "y_node",
    "edge_index",
    "run_id",
    "end_epoch",
    "feature_cols",
    "profile",
    "seed",
    "split",
    "strength",
    "active_cores",
    "attackers",
)

STRUCTURAL_ARRAYS = tuple(name for name in EXPECTED_ARRAYS if name != "x")


def sha256_stream(handle, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(chunk_bytes)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    with path.open("rb") as handle:
        return sha256_stream(handle, chunk_bytes)


def make_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.mkdir(parents=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def verify_extraction(args: argparse.Namespace) -> int:
    npz_path = args.npz.expanduser().resolve()
    extracted_dir = args.extracted_dir.expanduser().resolve()
    output = make_output_dir(args.output)

    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ archive not found: {npz_path}")
    if not extracted_dir.is_dir():
        raise NotADirectoryError(f"Extracted directory not found: {extracted_dir}")

    rows: list[dict[str, Any]] = []

    with zipfile.ZipFile(npz_path, "r") as archive:
        archive_members = sorted(
            name for name in archive.namelist()
            if not name.endswith("/")
        )
        extracted_members = sorted(
            path.name for path in extracted_dir.iterdir()
            if path.is_file()
        )

        archive_set = set(archive_members)
        extracted_set = set(extracted_members)

        for member in sorted(archive_set | extracted_set):
            archive_present = member in archive_set
            extracted_present = member in extracted_set
            archive_hash = ""
            extracted_hash = ""
            archive_size = None
            extracted_size = None

            if archive_present:
                archive_size = archive.getinfo(member).file_size
                with archive.open(member, "r") as handle:
                    archive_hash = sha256_stream(handle)

            extracted_path = extracted_dir / member
            if extracted_present:
                extracted_size = extracted_path.stat().st_size
                extracted_hash = sha256_file(extracted_path)

            match = (
                archive_present
                and extracted_present
                and archive_hash == extracted_hash
                and archive_size == extracted_size
            )

            rows.append(
                {
                    "member": member,
                    "archive_present": archive_present,
                    "extracted_present": extracted_present,
                    "archive_size": archive_size,
                    "extracted_size": extracted_size,
                    "archive_sha256": archive_hash,
                    "extracted_sha256": extracted_hash,
                    "match": match,
                }
            )

    all_match = bool(rows) and all(row["match"] for row in rows)
    expected_members = {f"{name}.npy" for name in EXPECTED_ARRAYS}
    expected_members_present = expected_members.issubset(
        {row["member"] for row in rows if row["match"]}
    )

    verdict = "VERIFIED" if all_match and expected_members_present else "FAILED"

    write_csv(
        output / "npz_vs_extracted_sha256.csv",
        rows,
        [
            "member",
            "archive_present",
            "extracted_present",
            "archive_size",
            "extracted_size",
            "archive_sha256",
            "extracted_sha256",
            "match",
        ],
    )
    write_json(
        output / "verification_summary.json",
        {
            "npz": str(npz_path),
            "extracted_dir": str(extracted_dir),
            "archive_file_sha256": sha256_file(npz_path),
            "member_count": len(rows),
            "all_members_match": all_match,
            "expected_members_present": expected_members_present,
            "verdict": verdict,
        },
    )
    (output / "VERDICT.txt").write_text(verdict + "\n", encoding="utf-8")

    print(f"verdict: {verdict}")
    print(f"members checked: {len(rows)}")
    print(f"output: {output}")
    return 0 if verdict == "VERIFIED" else 1


def load_array(directory: Path, name: str) -> np.ndarray:
    path = directory / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing array: {path}")
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError:
        return np.load(path, allow_pickle=True)


def exact_equal(a: np.ndarray, b: np.ndarray, chunk_size: int = 100_000) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False

    supports_nan = (
        np.issubdtype(a.dtype, np.floating)
        or np.issubdtype(a.dtype, np.complexfloating)
    )

    def arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
        if supports_nan:
            return bool(np.array_equal(left, right, equal_nan=True))
        return bool(np.array_equal(left, right))

    if a.ndim == 0:
        return arrays_equal(a, b)

    for start in range(0, a.shape[0], chunk_size):
        stop = min(a.shape[0], start + chunk_size)
        if not arrays_equal(a[start:stop], b[start:stop]):
            return False
    return True


def feature_category(name: str) -> str:
    if name in {"ifd_in_norm", "ifd_out_norm"}:
        return "aggregate_ifd"
    if name.startswith("ifd_in_norm_") or name.startswith("ifd_out_norm_"):
        return "port_directional_ifd"
    if name in {"input_flit_count_norm", "output_flit_count_norm"}:
        return "aggregate_count"
    if name.startswith("in_count_norm_") or name.startswith("out_count_norm_"):
        return "port_directional_count"
    return "other"


def compare_datasets(args: argparse.Namespace) -> int:
    old_dir = args.old.expanduser().resolve()
    chrono_dir = args.chrono.expanduser().resolve()
    output = make_output_dir(args.output)

    if not old_dir.is_dir():
        raise NotADirectoryError(f"Old dataset directory not found: {old_dir}")
    if not chrono_dir.is_dir():
        raise NotADirectoryError(f"Chrono dataset directory not found: {chrono_dir}")

    old_files = sorted(path.name for path in old_dir.glob("*.npy"))
    chrono_files = sorted(path.name for path in chrono_dir.glob("*.npy"))

    structural: dict[str, Any] = {
        "old_files": old_files,
        "chrono_files": chrono_files,
        "same_file_names": old_files == chrono_files,
        "arrays": {},
    }

    structural_equal = old_files == chrono_files

    for name in STRUCTURAL_ARRAYS:
        old = load_array(old_dir, name)
        chrono = load_array(chrono_dir, name)
        equal = exact_equal(old, chrono)
        structural["arrays"][name] = {
            "old_shape": list(old.shape),
            "chrono_shape": list(chrono.shape),
            "old_dtype": str(old.dtype),
            "chrono_dtype": str(chrono.dtype),
            "exact_equal": equal,
        }
        structural_equal = structural_equal and equal

    x_old = load_array(old_dir, "x")
    x_chrono = load_array(chrono_dir, "x")

    x_shape_dtype_equal = (
        x_old.shape == x_chrono.shape
        and x_old.dtype == x_chrono.dtype
    )
    structural["arrays"]["x"] = {
        "old_shape": list(x_old.shape),
        "chrono_shape": list(x_chrono.shape),
        "old_dtype": str(x_old.dtype),
        "chrono_dtype": str(x_chrono.dtype),
        "shape_dtype_equal": x_shape_dtype_equal,
    }

    if not x_shape_dtype_equal or x_old.ndim != 4:
        verdict = "STRUCTURALLY DIFFERENT DATASETS"
        structural["verdict"] = verdict
        write_json(output / "comparison_summary.json", structural)
        (output / "VERDICT.txt").write_text(verdict + "\n", encoding="utf-8")
        print(f"verdict: {verdict}")
        print(f"output: {output}")
        return 2

    sample_count, router_count, temporal_count, feature_count = x_old.shape

    features = np.asarray(load_array(old_dir, "feature_cols")).astype(str)
    if len(features) != feature_count:
        raise ValueError(
            f"feature_cols length {len(features)} does not match x width {feature_count}"
        )

    run_ids = np.asarray(load_array(old_dir, "run_id")).astype(str)

    changed_samples = 0
    changed_routers = 0
    changed_temporal_positions = 0
    changed_cells = 0
    total_abs_sum = 0.0
    changed_abs_sum = 0.0
    maximum_abs_difference = 0.0

    per_feature_changed = np.zeros(feature_count, dtype=np.int64)
    per_feature_abs_max = np.zeros(feature_count, dtype=np.float64)
    per_feature_abs_sum_changed = np.zeros(feature_count, dtype=np.float64)

    per_run_changed_cells: defaultdict[str, int] = defaultdict(int)
    per_run_changed_samples: defaultdict[str, int] = defaultdict(int)

    for start in range(0, sample_count, args.chunk_size):
        stop = min(sample_count, start + args.chunk_size)

        old_chunk = np.asarray(x_old[start:stop], dtype=np.float64)
        chrono_chunk = np.asarray(x_chrono[start:stop], dtype=np.float64)

        equal = np.isclose(
            old_chunk,
            chrono_chunk,
            rtol=args.rtol,
            atol=args.atol,
            equal_nan=True,
        )
        changed = ~equal
        absolute = np.abs(
            np.nan_to_num(old_chunk, copy=False)
            - np.nan_to_num(chrono_chunk, copy=False)
        )

        sample_changed_cells = changed.sum(axis=(1, 2, 3))
        sample_changed = sample_changed_cells > 0

        changed_samples += int(sample_changed.sum())
        changed_routers += int(changed.any(axis=(2, 3)).sum())
        changed_temporal_positions += int(changed.any(axis=3).sum())
        changed_cells += int(changed.sum())

        total_abs_sum += float(absolute.sum())
        if changed.any():
            changed_abs_sum += float(absolute[changed].sum())
        maximum_abs_difference = max(
            maximum_abs_difference,
            float(absolute.max(initial=0.0)),
        )

        per_feature_changed += changed.sum(axis=(0, 1, 2))
        per_feature_abs_sum_changed += (absolute * changed).sum(axis=(0, 1, 2))
        per_feature_abs_max = np.maximum(
            per_feature_abs_max,
            absolute.max(axis=(0, 1, 2)),
        )

        chunk_runs = run_ids[start:stop]
        for run in np.unique(chunk_runs):
            mask = chunk_runs == run
            counts = sample_changed_cells[mask]
            per_run_changed_cells[str(run)] += int(counts.sum())
            per_run_changed_samples[str(run)] += int((counts > 0).sum())

    total_cells = int(np.prod(x_old.shape))
    x_identical = changed_cells == 0

    per_feature_rows: list[dict[str, Any]] = []
    category_totals: defaultdict[str, int] = defaultdict(int)
    cells_per_feature = sample_count * router_count * temporal_count

    for index, feature in enumerate(features):
        changed_count = int(per_feature_changed[index])
        category = feature_category(feature)
        category_totals[category] += changed_count
        per_feature_rows.append(
            {
                "feature_index": index,
                "feature_name": feature,
                "category": category,
                "changed_cells": changed_count,
                "changed_fraction": changed_count / cells_per_feature,
                "mean_abs_difference_changed_cells": (
                    float(per_feature_abs_sum_changed[index] / changed_count)
                    if changed_count
                    else 0.0
                ),
                "maximum_abs_difference": float(per_feature_abs_max[index]),
            }
        )

    unique_runs, run_sample_counts = np.unique(run_ids, return_counts=True)
    per_run_rows = [
        {
            "run_id": str(run),
            "total_samples": int(total),
            "changed_samples": per_run_changed_samples[str(run)],
            "changed_cells": per_run_changed_cells[str(run)],
        }
        for run, total in zip(unique_runs, run_sample_counts)
    ]

    if structural_equal and x_identical:
        verdict = "IDENTICAL DATASETS"
    elif structural_equal:
        verdict = "SAME SAMPLES/LABELS/SPLITS, CORRECTED X VALUES ONLY"
    else:
        verdict = "STRUCTURALLY DIFFERENT DATASETS"

    summary = {
        **structural,
        "x_statistics": {
            "total_feature_cells": total_cells,
            "changed_feature_cells": changed_cells,
            "changed_cell_percentage": 100.0 * changed_cells / total_cells,
            "changed_samples": changed_samples,
            "changed_routers": changed_routers,
            "changed_temporal_positions": changed_temporal_positions,
            "maximum_abs_difference": maximum_abs_difference,
            "mean_abs_difference_all_cells": total_abs_sum / total_cells,
            "mean_abs_difference_changed_cells": (
                changed_abs_sum / changed_cells if changed_cells else 0.0
            ),
            "category_changed_cells": dict(sorted(category_totals.items())),
        },
        "structural_arrays_exact": structural_equal,
        "x_identical": x_identical,
        "verdict": verdict,
    }

    write_json(output / "comparison_summary.json", summary)
    write_csv(
        output / "per_feature_differences.csv",
        per_feature_rows,
        [
            "feature_index",
            "feature_name",
            "category",
            "changed_cells",
            "changed_fraction",
            "mean_abs_difference_changed_cells",
            "maximum_abs_difference",
        ],
    )
    write_csv(
        output / "per_run_differences.csv",
        per_run_rows,
        ["run_id", "total_samples", "changed_samples", "changed_cells"],
    )
    (output / "VERDICT.txt").write_text(verdict + "\n", encoding="utf-8")

    print(f"verdict: {verdict}")
    print(f"changed cells: {changed_cells}")
    print(f"changed samples: {changed_samples}")
    print(f"output: {output}")

    return 0 if verdict != "STRUCTURALLY DIFFERENT DATASETS" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser(
        "verify-extraction",
        help="Compare NPZ members with extracted .npy files.",
    )
    verify.add_argument("--npz", required=True, type=Path)
    verify.add_argument("--extracted-dir", required=True, type=Path)
    verify.add_argument("--output", required=True, type=Path)
    verify.set_defaults(func=verify_extraction)

    compare = subparsers.add_parser(
        "compare",
        help="Compare old and chronological extracted datasets.",
    )
    compare.add_argument("--old", required=True, type=Path)
    compare.add_argument("--chrono", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--chunk-size", type=int, default=256)
    compare.add_argument("--atol", type=float, default=0.0)
    compare.add_argument("--rtol", type=float, default=0.0)
    compare.set_defaults(func=compare_datasets)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
