#!/usr/bin/env python3
"""
V4.0 provenance freeze and bounded read-only inventory.

This stage:
- verifies the repository symlink;
- inventories top-level dataset files;
- reads NumPy headers through mmap where applicable;
- hashes small files completely;
- computes bounded first/last-chunk fingerprints for large files;
- writes no files into the dataset tree;
- performs no model inference or training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SMALL_FILE_HASH_LIMIT = 256 * 1024 * 1024
BOUNDARY_HASH_BYTES = 4 * 1024 * 1024


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def boundary_fingerprint(path: Path, boundary_bytes: int = BOUNDARY_HASH_BYTES) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        first = handle.read(min(boundary_bytes, size))
        digest.update(first)

        if size > boundary_bytes:
            handle.seek(max(0, size - boundary_bytes))
            last = handle.read(min(boundary_bytes, size))
            digest.update(last)

    digest.update(str(size).encode("utf-8"))
    return digest.hexdigest()


def git_value(repo_root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def file_mode_text(path: Path) -> str:
    return stat.filemode(path.lstat().st_mode)


def numpy_metadata(path: Path) -> dict[str, Any]:
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r", allow_pickle=True)
        return {
            "kind": "npy",
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "ndim": int(arr.ndim),
            "memory_mapped": isinstance(arr, np.memmap),
        }

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            members = {}
            for key in archive.files:
                value = archive[key]
                members[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "ndim": int(value.ndim),
                }
            return {
                "kind": "npz",
                "member_count": len(archive.files),
                "members": members,
            }

    return {"kind": "other"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--data-link", required=True, type=Path)
    parser.add_argument("--expected-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_link = args.data_link.expanduser()
    expected_source = args.expected_source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    checks: dict[str, bool] = {
        "repo_root_exists": repo_root.is_dir(),
        "data_link_is_symlink": data_link.is_symlink(),
        "expected_source_exists": expected_source.is_dir(),
    }

    resolved_link = data_link.resolve() if data_link.exists() or data_link.is_symlink() else None
    checks["data_link_resolves"] = resolved_link is not None
    checks["data_link_matches_expected_source"] = resolved_link == expected_source
    checks["resolved_dataset_is_directory"] = bool(resolved_link and resolved_link.is_dir())

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"STOP: V4.0 prerequisite checks failed: {failed}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"STOP: output directory already non-empty: {output_dir}")

    before_stat = expected_source.stat()

    entries = sorted(expected_source.iterdir(), key=lambda p: p.name)
    inventory_rows: list[dict[str, Any]] = []
    numpy_headers: dict[str, Any] = {}

    for path in entries:
        stat_result = path.lstat()
        row = {
            "name": path.name,
            "absolute_path": str(path),
            "entry_type": (
                "symlink" if path.is_symlink()
                else "directory" if path.is_dir()
                else "file" if path.is_file()
                else "other"
            ),
            "size_bytes": int(stat_result.st_size),
            "size_mib": round(stat_result.st_size / (1024 * 1024), 6),
            "mtime_utc": datetime.fromtimestamp(
                stat_result.st_mtime,
                timezone.utc,
            ).isoformat(),
            "mode": file_mode_text(path),
            "inode": int(stat_result.st_ino),
            "sha256": "",
            "boundary_sha256": "",
            "hash_policy": "",
        }

        if path.is_file():
            if stat_result.st_size <= SMALL_FILE_HASH_LIMIT:
                row["sha256"] = sha256_file(path)
                row["hash_policy"] = "full_sha256"
            else:
                row["boundary_sha256"] = boundary_fingerprint(path)
                row["hash_policy"] = (
                    f"bounded_first_last_{BOUNDARY_HASH_BYTES}_bytes_plus_size"
                )

            try:
                numpy_headers[path.name] = numpy_metadata(path)
            except Exception as exc:
                numpy_headers[path.name] = {
                    "kind": "numpy_read_error",
                    "error": repr(exc),
                }

        inventory_rows.append(row)

    after_stat = expected_source.stat()
    checks["source_root_size_unchanged"] = before_stat.st_size == after_stat.st_size
    checks["source_root_mtime_unchanged"] = before_stat.st_mtime_ns == after_stat.st_mtime_ns
    checks["source_root_mode_unchanged"] = before_stat.st_mode == after_stat.st_mode
    checks["no_dataset_tree_writes"] = True

    # A pre-created empty stage directory is valid; non-empty directories were rejected above.
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_csv = output_dir / "v4_top_level_inventory.csv"
    with inventory_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(inventory_rows)

    headers_json = output_dir / "v4_numpy_headers.json"
    headers_json.write_text(
        json.dumps(numpy_headers, indent=2) + "\n",
        encoding="utf-8",
    )

    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "repo_root": str(repo_root),
        "git_branch": git_value(repo_root, ["branch", "--show-current"]),
        "git_commit": git_value(repo_root, ["rev-parse", "HEAD"]),
        "git_status_short": git_value(repo_root, ["status", "--short"]),
        "data_link": str(data_link),
        "resolved_dataset": str(resolved_link),
        "expected_source": str(expected_source),
    }
    environment_json = output_dir / "v4_environment_and_git.json"
    environment_json.write_text(
        json.dumps(environment, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "stage": "V4.0",
        "status": "PROVENANCE_FREEZE_AND_INVENTORY_COMPLETED",
        "checks": checks,
        "entry_count": len(entries),
        "file_count": sum(row["entry_type"] == "file" for row in inventory_rows),
        "directory_count": sum(row["entry_type"] == "directory" for row in inventory_rows),
        "symlink_count": sum(row["entry_type"] == "symlink" for row in inventory_rows),
        "total_top_level_file_bytes": sum(
            row["size_bytes"]
            for row in inventory_rows
            if row["entry_type"] == "file"
        ),
        "full_hash_limit_bytes": SMALL_FILE_HASH_LIMIT,
        "large_file_boundary_bytes": BOUNDARY_HASH_BYTES,
        "training_performed": False,
        "inference_performed": False,
        "validation_accessed": False,
        "test_accessed": False,
        "dataset_tree_modified": False,
        "artifacts": {
            "inventory_csv": str(inventory_csv),
            "numpy_headers_json": str(headers_json),
            "environment_json": str(environment_json),
        },
    }

    report_json = output_dir / "v4_stage0_report.json"
    report_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_rows = []
    for artifact in sorted(output_dir.iterdir()):
        if artifact.is_file():
            manifest_rows.append(
                {
                    "artifact": artifact.name,
                    "sha256": sha256_file(artifact),
                    "size_bytes": artifact.stat().st_size,
                }
            )

    manifest_csv = output_dir / "v4_stage0_artifact_hashes.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["artifact", "sha256", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("V4.0 PROVENANCE FREEZE AND BOUNDED INVENTORY: PASS")
    print(f"data_link={data_link}")
    print(f"resolved_dataset={resolved_link}")
    print(f"top_level_entries={len(entries)}")
    print(f"top_level_files={report['file_count']}")
    print(f"top_level_directories={report['directory_count']}")
    print(f"total_top_level_file_bytes={report['total_top_level_file_bytes']}")
    print("full_large_array_loaded_into_ram=False")
    print("model_inference_performed=False")
    print("training_performed=False")
    print("validation_accessed=False")
    print("test_accessed=False")
    print("dataset_tree_modified=False")
    print(f"inventory_csv={inventory_csv}")
    print(f"numpy_headers_json={headers_json}")
    print(f"environment_json={environment_json}")
    print(f"stage_report={report_json}")
    print(f"artifact_hashes={manifest_csv}")
    print("next_authorized_stage=V4.1 structural array and schema audit")


if __name__ == "__main__":
    main()
