#!/usr/bin/env python3
"""Stage 8A: read-only inventory of the V3 gem5 experiment root.

This script inventories files without executing discovered code, following symbolic
links, loading complete NumPy arrays, or scanning unbounded file contents.
"""

from __future__ import annotations

import argparse
import ast
import csv
import errno
import json
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
import textwrap
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


SCRIPT_VERSION = "stage8a_v1"
DEFAULT_EXPERIMENT_ROOT = Path(
    "/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3"
)
DEFAULT_FAILURE_ROOT = Path("reports/v3_failure_analysis")

CATEGORY_RAW = "1_raw_gem5_simulation_output"
CATEGORY_INTERMEDIATE = "2_intermediate_epoch_or_router_port_records"
CATEGORY_WINDOWED = "3_processed_temporal_window_files"
CATEGORY_FINAL = "4_final_numpy_graph_dataset"
CATEGORY_LABELS = "5_labels_and_split_metadata"
CATEGORY_BUILDER = "6_builder_or_preprocessing_script"
CATEGORY_DOCS = "7_logs_manifests_or_documentation"
CATEGORY_UNKNOWN = "8_unknown_or_unclassified"
CATEGORY_DIRECTORY = "directory_container"
CATEGORY_SYMLINK = "symbolic_link_not_followed"

FILE_CATEGORIES = {
    CATEGORY_RAW,
    CATEGORY_INTERMEDIATE,
    CATEGORY_WINDOWED,
    CATEGORY_FINAL,
    CATEGORY_LABELS,
    CATEGORY_BUILDER,
    CATEGORY_DOCS,
    CATEGORY_UNKNOWN,
}

FINAL_NUMPY_NAMES = {
    "x.npy",
    "feature_cols.npy",
    "edge_index.npy",
    "edge_attr.npy",
    "graph_ptr.npy",
    "node_features.npy",
}

LABEL_METADATA_NAMES = {
    "y_graph.npy",
    "y_node.npy",
    "run_id.npy",
    "end_epoch.npy",
    "split.npy",
    "attacker_mask.npy",
    "victim_id.npy",
    "victim_mask.npy",
    "sample_index.npy",
    "metadata.npy",
}

KNOWN_RAW_BASENAMES = {
    "stats.txt",
    "config.ini",
    "config.json",
    "simout",
    "simerr",
    "system.pc.com_1.device",
    "system.terminal",
}

RAW_PATH_TOKENS = {
    "m5out",
    "raw",
    "trace",
    "traces",
    "gem5stats",
    "gem5_stats",
    "simulation_output",
    "sim_output",
    "router_trace",
    "network_trace",
    "noc_trace",
    "probe_output",
}

INTERMEDIATE_NAME_TOKENS = {
    "epoch",
    "router",
    "port",
    "ifd",
    "flit_count",
    "feature_record",
    "feature_records",
    "monitor",
    "snapshot",
    "counter",
    "traffic_record",
    "temporal_features",
}

WINDOW_NAME_TOKENS = {
    "window",
    "windows",
    "temporal_graph",
    "graph_samples",
    "sample_windows",
    "sequence",
    "sequences",
}

BUILDER_NAME_TOKENS = {
    "build",
    "builder",
    "preprocess",
    "extract",
    "generate",
    "create_dataset",
    "make_dataset",
    "convert",
    "window",
    "normalize",
}

DOC_NAME_TOKENS = {
    "readme",
    "manifest",
    "report",
    "summary",
    "notes",
    "metadata",
    "log",
    "command",
    "provenance",
}

TEXT_EXTENSIONS = {
    ".py",
    ".sh",
    ".bash",
    ".zsh",
    ".ipynb",
    ".md",
    ".txt",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".ini",
    ".cfg",
    ".conf",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".log",
    ".out",
    ".err",
    ".dat",
}

SCRIPT_EXTENSIONS = {".py", ".sh", ".bash", ".zsh", ".ipynb"}

BINARY_EXTENSIONS = {
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".blob",
    ".so",
    ".a",
    ".o",
    ".elf",
    ".gz",
    ".xz",
    ".bz2",
    ".zip",
    ".tar",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
}

GENERAL_KEYWORDS = (
    "paper1_temporal_graphs_ports_v3",
    "x.npy",
    "feature_cols.npy",
    "end_epoch.npy",
    "y_graph.npy",
    "y_node.npy",
    "ifd_in_norm",
    "ifd_out_norm",
    "input_flit_count_norm",
    "output_flit_count_norm",
    "3293",
    "window",
    "stride",
    "epoch",
    "clip",
    "normalize",
)

BUILDER_KEYWORDS = (
    "np.save",
    "numpy.save",
    "sliding_window",
    "window_length",
    "window_size",
    "window_stride",
    "stride",
    "feature_cols",
    "y_graph",
    "y_node",
    "end_epoch",
    "edge_index",
    "ifd_in_norm",
    "ifd_out_norm",
    "flit_count_norm",
    "normalize",
    "clip",
)

RAW_KEYWORDS = (
    "sim_ticks",
    "system.ruby",
    "system.network",
    "router",
    "flit",
    "packet",
    "stats.txt",
    "gem5",
    "m5out",
    "network trace",
    "router trace",
)

CSV_COLUMNS_INVENTORY = [
    "relative_path",
    "parent_directory",
    "entry_name",
    "entry_type",
    "classification",
    "probable_role",
    "classification_reason",
    "size_bytes",
    "size_mib",
    "modified_time_utc",
    "mode_octal",
    "is_symlink",
    "symlink_target",
    "symlink_target_within_root",
    "extension",
    "text_prefix_scanned",
    "text_prefix_bytes_read",
    "keyword_hits",
    "npy_shape",
    "npy_dtype",
    "npy_fortran_order",
    "npy_header_bytes_read",
    "npz_member_count",
    "npz_members_preview",
    "permission_or_read_error",
    "inspect_in_stage8b",
]

CSV_COLUMNS_DIRECTORY = [
    "relative_directory",
    "recursive_file_count",
    "recursive_directory_count",
    "recursive_symlink_count",
    "recursive_size_bytes",
    "recursive_size_mib",
    "raw_count",
    "intermediate_count",
    "windowed_count",
    "final_numpy_count",
    "labels_metadata_count",
    "builder_count",
    "logs_docs_count",
    "unknown_count",
]

CSV_COLUMNS_CANDIDATE = [
    "relative_path",
    "classification",
    "candidate_score",
    "candidate_reasons",
    "keyword_hits",
    "size_bytes",
    "modified_time_utc",
    "inspect_in_stage8b",
]

CSV_COLUMNS_UNKNOWN_LARGE = [
    "relative_path",
    "size_bytes",
    "size_mib",
    "extension",
    "probable_role",
    "classification_reason",
    "permission_or_read_error",
]


@dataclass(frozen=True)
class NpyHeader:
    shape: tuple[int, ...] | None
    dtype: str | None
    fortran_order: bool | None
    bytes_read: int
    error: str


@dataclass(frozen=True)
class NpzHeader:
    member_count: int | None
    member_preview: str
    error: str


@dataclass
class InventoryResult:
    inventory_rows: list[dict[str, Any]]
    directory_rows: list[dict[str, Any]]
    raw_candidates: list[dict[str, Any]]
    builder_candidates: list[dict[str, Any]]
    unknown_large: list[dict[str, Any]]
    errors: list[str]
    source_metadata_unchanged: bool
    source_metadata_change_details: list[str]
    traversal_within_root: bool
    external_symlink_followed: bool
    npy_headers_bounded: bool
    text_prefixes_bounded: bool
    summary: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Stage 8A inventory of the V3 experiment root. "
            "The script does not execute discovered files, follow symbolic links, "
            "or load complete NumPy arrays."
        )
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help=f"Experiment root to inventory (default: {DEFAULT_EXPERIMENT_ROOT})",
    )
    parser.add_argument(
        "--failure-root",
        type=Path,
        default=DEFAULT_FAILURE_ROOT,
        help=f"Failure-analysis root (default: {DEFAULT_FAILURE_ROOT})",
    )
    parser.add_argument(
        "--text-prefix-bytes",
        type=int,
        default=65536,
        help="Maximum bytes read from any text-like file (default: 65536)",
    )
    parser.add_argument(
        "--max-npy-header-bytes",
        type=int,
        default=1048576,
        help="Maximum NumPy header bytes accepted (default: 1048576)",
    )
    parser.add_argument(
        "--max-npz-members",
        type=int,
        default=200,
        help="Maximum NPZ member names retained in preview (default: 200)",
    )
    parser.add_argument(
        "--large-file-threshold-mib",
        type=float,
        default=256.0,
        help="Threshold for unknown-large-file reporting (default: 256 MiB)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Stage 8A outputs transactionally",
    )
    parser.add_argument(
        "--smoke-test-only",
        action="store_true",
        help="Run synthetic read-only tests without touching the real experiment root",
    )
    args = parser.parse_args()
    if args.text_prefix_bytes <= 0:
        parser.error("--text-prefix-bytes must be positive")
    if args.max_npy_header_bytes <= 0:
        parser.error("--max-npy-header-bytes must be positive")
    if args.max_npz_members <= 0:
        parser.error("--max-npz-members must be positive")
    if args.large_file_threshold_mib <= 0:
        parser.error("--large-file-threshold-mib must be positive")
    return args


def utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def mib(size_bytes: int) -> float:
    return float(size_bytes) / (1024.0 * 1024.0)


def safe_relative(path: Path, root: Path) -> str:
    return "." if path == root else path.relative_to(root).as_posix()


def within_root_lexically(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def open_readonly_noatime(path: Path):
    """Open a non-symlink file read-only, using O_NOATIME/O_NOFOLLOW when possible."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    noatime = getattr(os, "O_NOATIME", 0)
    attempted_flags = flags | noatime
    try:
        fd = os.open(path, attempted_flags)
    except OSError as exc:
        if noatime and exc.errno in {errno.EPERM, errno.EACCES, errno.EINVAL}:
            fd = os.open(path, flags)
        else:
            raise
    return os.fdopen(fd, "rb", closefd=True)


def read_bounded_prefix(path: Path, max_bytes: int) -> tuple[bytes, str]:
    try:
        with open_readonly_noatime(path) as handle:
            data = handle.read(max_bytes)
        return data, ""
    except Exception as exc:  # noqa: BLE001 - inventory should record and continue
        return b"", f"{type(exc).__name__}: {exc}"


def inspect_npy_header(path: Path, max_header_bytes: int) -> NpyHeader:
    """Read only the NPY header using an explicit size bound."""

    bytes_read = 0
    try:
        with open_readonly_noatime(path) as handle:
            magic = handle.read(6)
            bytes_read += len(magic)
            if magic != b"\x93NUMPY":
                return NpyHeader(None, None, None, bytes_read, "invalid NPY magic")
            version_raw = handle.read(2)
            bytes_read += len(version_raw)
            if len(version_raw) != 2:
                return NpyHeader(None, None, None, bytes_read, "truncated NPY version")
            major, minor = version_raw[0], version_raw[1]
            if (major, minor) == (1, 0):
                length_raw = handle.read(2)
                bytes_read += len(length_raw)
                if len(length_raw) != 2:
                    return NpyHeader(None, None, None, bytes_read, "truncated NPY header length")
                header_length = struct.unpack("<H", length_raw)[0]
                encoding = "latin1"
            elif major in {2, 3}:
                length_raw = handle.read(4)
                bytes_read += len(length_raw)
                if len(length_raw) != 4:
                    return NpyHeader(None, None, None, bytes_read, "truncated NPY header length")
                header_length = struct.unpack("<I", length_raw)[0]
                encoding = "utf-8" if major == 3 else "latin1"
            else:
                return NpyHeader(None, None, None, bytes_read, f"unsupported NPY version {major}.{minor}")
            if header_length > max_header_bytes:
                return NpyHeader(
                    None,
                    None,
                    None,
                    bytes_read,
                    f"NPY header length {header_length} exceeds bound {max_header_bytes}",
                )
            header_raw = handle.read(header_length)
            bytes_read += len(header_raw)
            if len(header_raw) != header_length:
                return NpyHeader(None, None, None, bytes_read, "truncated NPY header")
            header_text = header_raw.decode(encoding).strip()
            header_dict = ast.literal_eval(header_text)
            if not isinstance(header_dict, dict):
                return NpyHeader(None, None, None, bytes_read, "NPY header is not a dictionary")
            shape_value = header_dict.get("shape")
            shape = tuple(int(value) for value in shape_value) if isinstance(shape_value, tuple) else None
            dtype = str(np.dtype(header_dict.get("descr")))
            fortran_order = bool(header_dict.get("fortran_order"))
            return NpyHeader(shape, dtype, fortran_order, bytes_read, "")
    except Exception as exc:  # noqa: BLE001
        return NpyHeader(None, None, None, bytes_read, f"{type(exc).__name__}: {exc}")


def inspect_npz_header(path: Path, max_members: int) -> NpzHeader:
    """Inspect the ZIP central directory only; member payloads are not decompressed."""

    try:
        with open_readonly_noatime(path) as handle:
            with zipfile.ZipFile(handle, mode="r") as archive:
                names = archive.namelist()
        preview = ";".join(names[:max_members])
        if len(names) > max_members:
            preview += f";...(+{len(names) - max_members} more)"
        return NpzHeader(len(names), preview, "")
    except Exception as exc:  # noqa: BLE001
        return NpzHeader(None, "", f"{type(exc).__name__}: {exc}")


def decode_text_prefix(data: bytes) -> str:
    if b"\x00" in data[:4096]:
        return ""
    return data.decode("utf-8", errors="replace")


def find_keyword_hits(text: str, keywords: Sequence[str]) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in keywords if keyword.lower() in lowered]


def token_in_name(name: str, tokens: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in tokens)


def path_tokens(relative_path: str) -> set[str]:
    tokens: set[str] = set()
    for part in Path(relative_path).parts:
        for token in re.split(r"[^a-zA-Z0-9]+", part.lower()):
            if token:
                tokens.add(token)
    return tokens


def classify_file(
    relative_path: str,
    name: str,
    extension: str,
    text: str,
    keyword_hits: Sequence[str],
) -> tuple[str, str, str]:
    lower_name = name.lower()
    lower_path = relative_path.lower()
    tokens = path_tokens(relative_path)

    if lower_name in FINAL_NUMPY_NAMES:
        return (
            CATEGORY_FINAL,
            "Final graph-dataset array or feature/topology metadata",
            f"exact recognized final-array filename: {lower_name}",
        )
    if lower_name in LABEL_METADATA_NAMES:
        return (
            CATEGORY_LABELS,
            "Stored labels, run/sample metadata, or split assignment",
            f"exact recognized label/metadata filename: {lower_name}",
        )
    if lower_name in KNOWN_RAW_BASENAMES or bool(tokens & RAW_PATH_TOKENS):
        return (
            CATEGORY_RAW,
            "Probable raw gem5 simulation output or configuration",
            "recognized gem5 basename or raw-output path token",
        )
    if extension in SCRIPT_EXTENSIONS:
        builder_hits = find_keyword_hits(text, BUILDER_KEYWORDS)
        if token_in_name(lower_name, BUILDER_NAME_TOKENS) or builder_hits:
            return (
                CATEGORY_BUILDER,
                "Candidate builder, extractor, normalizer, or windowing script",
                "script/notebook filename or bounded prefix contains builder keywords",
            )
        return (
            CATEGORY_BUILDER,
            "Script or notebook requiring Stage 8B relevance ranking",
            "script/notebook file inside experiment root",
        )
    if token_in_name(lower_name, WINDOW_NAME_TOKENS) or "temporal_graph" in lower_path:
        return (
            CATEGORY_WINDOWED,
            "Probable processed temporal windows or graph samples",
            "filename/path contains temporal-window tokens",
        )
    if token_in_name(lower_name, INTERMEDIATE_NAME_TOKENS):
        return (
            CATEGORY_INTERMEDIATE,
            "Probable epoch-level, router-level, port-level, or feature record",
            "filename contains intermediate feature/statistic tokens",
        )
    if extension in {".csv", ".tsv", ".parquet", ".feather"}:
        return (
            CATEGORY_INTERMEDIATE,
            "Structured tabular record requiring Stage 8B inspection",
            "structured table extension without a more specific classification",
        )
    if token_in_name(lower_name, DOC_NAME_TOKENS) or extension in {
        ".md",
        ".log",
        ".out",
        ".err",
    }:
        return (
            CATEGORY_DOCS,
            "Log, report, manifest, command record, or documentation",
            "documentation/log filename or extension",
        )
    if keyword_hits and extension in TEXT_EXTENSIONS:
        return (
            CATEGORY_DOCS,
            "Text artifact containing relevant Stage 8 keywords",
            "bounded text prefix contains project/dataset keywords",
        )
    return (
        CATEGORY_UNKNOWN,
        "Unknown artifact requiring later targeted inspection",
        "no recognized filename, path, extension, or bounded-prefix signature",
    )


def candidate_raw_score(row: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    relative = str(row["relative_path"]).lower()
    name = str(row["entry_name"]).lower()
    hits = str(row["keyword_hits"]).lower()
    if row["classification"] == CATEGORY_RAW:
        score += 5
        reasons.append("classified_as_raw")
    if name in KNOWN_RAW_BASENAMES:
        score += 4
        reasons.append("recognized_gem5_basename")
    if any(token in path_tokens(relative) for token in RAW_PATH_TOKENS):
        score += 2
        reasons.append("raw_path_token")
    if any(keyword in hits for keyword in RAW_KEYWORDS):
        score += 1
        reasons.append("raw_keyword_hit")
    return score, reasons


def candidate_builder_score(row: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    name = str(row["entry_name"]).lower()
    extension = str(row["extension"]).lower()
    hits = str(row["keyword_hits"]).lower()
    if row["classification"] == CATEGORY_BUILDER:
        score += 4
        reasons.append("classified_as_builder")
    if extension in SCRIPT_EXTENSIONS:
        score += 2
        reasons.append("script_or_notebook")
    if token_in_name(name, BUILDER_NAME_TOKENS):
        score += 2
        reasons.append("builder_filename_token")
    hit_count = sum(1 for keyword in BUILDER_KEYWORDS if keyword.lower() in hits)
    if hit_count:
        score += min(hit_count, 4)
        reasons.append(f"builder_keyword_hits={hit_count}")
    return score, reasons


def snapshot_metadata(root: Path) -> dict[str, tuple[int, int, int, str]]:
    snapshot: dict[str, tuple[int, int, int, str]] = {}
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        safe_dirnames: list[str] = []
        for dirname in dirnames:
            path = current_path / dirname
            try:
                info = path.lstat()
            except OSError:
                safe_dirnames.append(dirname)
                continue
            rel = safe_relative(path, root)
            target = os.readlink(path) if stat.S_ISLNK(info.st_mode) else ""
            snapshot[rel] = (info.st_size, info.st_mtime_ns, info.st_mode, target)
            if not stat.S_ISLNK(info.st_mode):
                safe_dirnames.append(dirname)
        dirnames[:] = safe_dirnames
        for filename in filenames:
            path = current_path / filename
            try:
                info = path.lstat()
            except OSError:
                continue
            rel = safe_relative(path, root)
            target = os.readlink(path) if stat.S_ISLNK(info.st_mode) else ""
            snapshot[rel] = (info.st_size, info.st_mtime_ns, info.st_mode, target)
    root_info = root.lstat()
    snapshot["."] = (root_info.st_size, root_info.st_mtime_ns, root_info.st_mode, "")
    return snapshot


def compare_metadata(
    before: dict[str, tuple[int, int, int, str]],
    after: dict[str, tuple[int, int, int, str]],
) -> tuple[bool, list[str]]:
    details: list[str] = []
    before_keys = set(before)
    after_keys = set(after)
    for missing in sorted(before_keys - after_keys):
        details.append(f"source entry disappeared: {missing}")
    for added in sorted(after_keys - before_keys):
        details.append(f"source entry appeared: {added}")
    for key in sorted(before_keys & after_keys):
        if before[key] != after[key]:
            details.append(f"source metadata changed: {key}: before={before[key]} after={after[key]}")
    return not details, details


def build_directory_rows(
    root: Path,
    inventory_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "recursive_file_count": 0,
            "recursive_directory_count": 0,
            "recursive_symlink_count": 0,
            "recursive_size_bytes": 0,
            "category_counts": Counter(),
        }
    )
    directory_names = {"."}
    for row in inventory_rows:
        rel = str(row["relative_path"])
        entry_type = str(row["entry_type"])
        if entry_type == "directory":
            directory_names.add(rel)
        parent = Path(rel).parent
        ancestors = ["."]
        if str(parent) not in {".", ""}:
            parts = parent.parts
            for index in range(1, len(parts) + 1):
                ancestors.append(Path(*parts[:index]).as_posix())
        for ancestor in ancestors:
            agg = aggregates[ancestor]
            if entry_type == "file":
                agg["recursive_file_count"] += 1
                agg["recursive_size_bytes"] += int(row["size_bytes"] or 0)
                agg["category_counts"][str(row["classification"])] += 1
            elif entry_type == "directory":
                if rel != ancestor:
                    agg["recursive_directory_count"] += 1
            elif entry_type == "symlink":
                agg["recursive_symlink_count"] += 1
    rows: list[dict[str, Any]] = []
    for directory in sorted(directory_names):
        agg = aggregates[directory]
        counts: Counter[str] = agg["category_counts"]
        rows.append(
            {
                "relative_directory": directory,
                "recursive_file_count": agg["recursive_file_count"],
                "recursive_directory_count": agg["recursive_directory_count"],
                "recursive_symlink_count": agg["recursive_symlink_count"],
                "recursive_size_bytes": agg["recursive_size_bytes"],
                "recursive_size_mib": round(mib(agg["recursive_size_bytes"]), 6),
                "raw_count": counts[CATEGORY_RAW],
                "intermediate_count": counts[CATEGORY_INTERMEDIATE],
                "windowed_count": counts[CATEGORY_WINDOWED],
                "final_numpy_count": counts[CATEGORY_FINAL],
                "labels_metadata_count": counts[CATEGORY_LABELS],
                "builder_count": counts[CATEGORY_BUILDER],
                "logs_docs_count": counts[CATEGORY_DOCS],
                "unknown_count": counts[CATEGORY_UNKNOWN],
            }
        )
    return rows


def inventory_experiment_root(
    experiment_root: Path,
    *,
    text_prefix_bytes: int,
    max_npy_header_bytes: int,
    max_npz_members: int,
    large_file_threshold_bytes: int,
) -> InventoryResult:
    root = experiment_root.expanduser().absolute()
    if not root.exists():
        raise FileNotFoundError(f"experiment root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"experiment root is not a directory: {root}")
    if root.is_symlink():
        raise ValueError("experiment root itself must not be a symbolic link")

    before = snapshot_metadata(root)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    traversal_within_root = True
    external_symlink_followed = False
    npy_headers_bounded = True
    text_prefixes_bounded = True

    def on_walk_error(exc: OSError) -> None:
        errors.append(f"walk error: {type(exc).__name__}: {exc}")

    for current, dirnames, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=on_walk_error,
    ):
        current_path = Path(current).absolute()
        if not within_root_lexically(current_path, root):
            traversal_within_root = False
            errors.append(f"walk escaped root: {current_path}")
            dirnames[:] = []
            continue

        kept_dirs: list[str] = []
        for dirname in sorted(dirnames):
            path = current_path / dirname
            rel = safe_relative(path, root)
            try:
                info = path.lstat()
                is_link = stat.S_ISLNK(info.st_mode)
                target = os.readlink(path) if is_link else ""
                target_within = ""
                if is_link:
                    target_path = (path.parent / target).absolute()
                    target_within = str(within_root_lexically(target_path, root))
                rows.append(
                    {
                        "relative_path": rel,
                        "parent_directory": safe_relative(path.parent, root),
                        "entry_name": dirname,
                        "entry_type": "symlink" if is_link else "directory",
                        "classification": CATEGORY_SYMLINK if is_link else CATEGORY_DIRECTORY,
                        "probable_role": "Symbolic link recorded but not followed" if is_link else "Directory container",
                        "classification_reason": "os.walk followlinks=False; symlink removed from recursion" if is_link else "directory entry",
                        "size_bytes": int(info.st_size),
                        "size_mib": round(mib(info.st_size), 6),
                        "modified_time_utc": utc_iso(info.st_mtime),
                        "mode_octal": oct(stat.S_IMODE(info.st_mode)),
                        "is_symlink": is_link,
                        "symlink_target": target,
                        "symlink_target_within_root": target_within,
                        "extension": path.suffix.lower(),
                        "text_prefix_scanned": False,
                        "text_prefix_bytes_read": 0,
                        "keyword_hits": "",
                        "npy_shape": "",
                        "npy_dtype": "",
                        "npy_fortran_order": "",
                        "npy_header_bytes_read": 0,
                        "npz_member_count": "",
                        "npz_members_preview": "",
                        "permission_or_read_error": "",
                        "inspect_in_stage8b": False,
                    }
                )
                if not is_link:
                    kept_dirs.append(dirname)
            except OSError as exc:
                errors.append(f"directory lstat failed: {rel}: {type(exc).__name__}: {exc}")
                kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in sorted(filenames):
            path = current_path / filename
            rel = safe_relative(path, root)
            if not within_root_lexically(path.absolute(), root):
                traversal_within_root = False
                errors.append(f"file escaped root lexically: {path}")
                continue
            try:
                info = path.lstat()
            except OSError as exc:
                errors.append(f"file lstat failed: {rel}: {type(exc).__name__}: {exc}")
                continue

            is_link = stat.S_ISLNK(info.st_mode)
            extension = path.suffix.lower()
            symlink_target = os.readlink(path) if is_link else ""
            target_within = ""
            if is_link:
                target_path = (path.parent / symlink_target).absolute()
                target_within = str(within_root_lexically(target_path, root))
                rows.append(
                    {
                        "relative_path": rel,
                        "parent_directory": safe_relative(path.parent, root),
                        "entry_name": filename,
                        "entry_type": "symlink",
                        "classification": CATEGORY_SYMLINK,
                        "probable_role": "Symbolic link recorded but not followed",
                        "classification_reason": "file symlink; target content not opened",
                        "size_bytes": int(info.st_size),
                        "size_mib": round(mib(info.st_size), 6),
                        "modified_time_utc": utc_iso(info.st_mtime),
                        "mode_octal": oct(stat.S_IMODE(info.st_mode)),
                        "is_symlink": True,
                        "symlink_target": symlink_target,
                        "symlink_target_within_root": target_within,
                        "extension": extension,
                        "text_prefix_scanned": False,
                        "text_prefix_bytes_read": 0,
                        "keyword_hits": "",
                        "npy_shape": "",
                        "npy_dtype": "",
                        "npy_fortran_order": "",
                        "npy_header_bytes_read": 0,
                        "npz_member_count": "",
                        "npz_members_preview": "",
                        "permission_or_read_error": "",
                        "inspect_in_stage8b": False,
                    }
                )
                continue

            text = ""
            text_scanned = False
            prefix_bytes_read = 0
            keyword_hits: list[str] = []
            npy_header = NpyHeader(None, None, None, 0, "")
            npz_header = NpzHeader(None, "", "")
            read_error = ""

            lower_name = filename.lower()
            text_like = extension in TEXT_EXTENSIONS or lower_name in KNOWN_RAW_BASENAMES
            if extension == ".npy":
                npy_header = inspect_npy_header(path, max_npy_header_bytes)
                if npy_header.bytes_read > max_npy_header_bytes + 12:
                    npy_headers_bounded = False
                read_error = npy_header.error
            elif extension == ".npz":
                npz_header = inspect_npz_header(path, max_npz_members)
                read_error = npz_header.error
            elif text_like and extension not in BINARY_EXTENSIONS:
                prefix, prefix_error = read_bounded_prefix(path, text_prefix_bytes)
                text_scanned = True
                prefix_bytes_read = len(prefix)
                if prefix_bytes_read > text_prefix_bytes:
                    text_prefixes_bounded = False
                read_error = prefix_error
                text = decode_text_prefix(prefix)
                if text:
                    keyword_hits = find_keyword_hits(text, GENERAL_KEYWORDS + BUILDER_KEYWORDS + RAW_KEYWORDS)

            classification, probable_role, reason = classify_file(
                rel,
                filename,
                extension,
                text,
                keyword_hits,
            )
            inspect_stage8b = classification in {
                CATEGORY_RAW,
                CATEGORY_INTERMEDIATE,
                CATEGORY_WINDOWED,
                CATEGORY_BUILDER,
            } or bool(keyword_hits)
            if extension == ".npy" and filename.lower() in FINAL_NUMPY_NAMES | LABEL_METADATA_NAMES:
                inspect_stage8b = True

            rows.append(
                {
                    "relative_path": rel,
                    "parent_directory": safe_relative(path.parent, root),
                    "entry_name": filename,
                    "entry_type": "file",
                    "classification": classification,
                    "probable_role": probable_role,
                    "classification_reason": reason,
                    "size_bytes": int(info.st_size),
                    "size_mib": round(mib(info.st_size), 6),
                    "modified_time_utc": utc_iso(info.st_mtime),
                    "mode_octal": oct(stat.S_IMODE(info.st_mode)),
                    "is_symlink": False,
                    "symlink_target": "",
                    "symlink_target_within_root": "",
                    "extension": extension,
                    "text_prefix_scanned": text_scanned,
                    "text_prefix_bytes_read": prefix_bytes_read,
                    "keyword_hits": ";".join(sorted(set(keyword_hits))),
                    "npy_shape": "" if npy_header.shape is None else repr(npy_header.shape),
                    "npy_dtype": "" if npy_header.dtype is None else npy_header.dtype,
                    "npy_fortran_order": "" if npy_header.fortran_order is None else npy_header.fortran_order,
                    "npy_header_bytes_read": npy_header.bytes_read,
                    "npz_member_count": "" if npz_header.member_count is None else npz_header.member_count,
                    "npz_members_preview": npz_header.member_preview,
                    "permission_or_read_error": read_error,
                    "inspect_in_stage8b": inspect_stage8b,
                }
            )

    rows.sort(key=lambda row: str(row["relative_path"]))
    after = snapshot_metadata(root)
    metadata_unchanged, metadata_details = compare_metadata(before, after)

    raw_candidates: list[dict[str, Any]] = []
    builder_candidates: list[dict[str, Any]] = []
    unknown_large: list[dict[str, Any]] = []
    for row in rows:
        if row["entry_type"] != "file":
            continue
        raw_score, raw_reasons = candidate_raw_score(row)
        if raw_score > 0:
            raw_candidates.append(
                {
                    "relative_path": row["relative_path"],
                    "classification": row["classification"],
                    "candidate_score": raw_score,
                    "candidate_reasons": ";".join(raw_reasons),
                    "keyword_hits": row["keyword_hits"],
                    "size_bytes": row["size_bytes"],
                    "modified_time_utc": row["modified_time_utc"],
                    "inspect_in_stage8b": row["inspect_in_stage8b"],
                }
            )
        builder_score, builder_reasons = candidate_builder_score(row)
        if builder_score > 0:
            builder_candidates.append(
                {
                    "relative_path": row["relative_path"],
                    "classification": row["classification"],
                    "candidate_score": builder_score,
                    "candidate_reasons": ";".join(builder_reasons),
                    "keyword_hits": row["keyword_hits"],
                    "size_bytes": row["size_bytes"],
                    "modified_time_utc": row["modified_time_utc"],
                    "inspect_in_stage8b": row["inspect_in_stage8b"],
                }
            )
        if row["classification"] == CATEGORY_UNKNOWN and int(row["size_bytes"]) >= large_file_threshold_bytes:
            unknown_large.append(
                {
                    "relative_path": row["relative_path"],
                    "size_bytes": row["size_bytes"],
                    "size_mib": row["size_mib"],
                    "extension": row["extension"],
                    "probable_role": row["probable_role"],
                    "classification_reason": row["classification_reason"],
                    "permission_or_read_error": row["permission_or_read_error"],
                }
            )

    raw_candidates.sort(key=lambda row: (-int(row["candidate_score"]), str(row["relative_path"])))
    builder_candidates.sort(key=lambda row: (-int(row["candidate_score"]), str(row["relative_path"])))
    unknown_large.sort(key=lambda row: (-int(row["size_bytes"]), str(row["relative_path"])))
    directory_rows = build_directory_rows(root, rows)

    file_rows = [row for row in rows if row["entry_type"] == "file"]
    category_counts = Counter(str(row["classification"]) for row in file_rows)
    entry_counts = Counter(str(row["entry_type"]) for row in rows)
    largest_files = sorted(file_rows, key=lambda row: int(row["size_bytes"]), reverse=True)[:20]

    raw_present = category_counts[CATEGORY_RAW] > 0
    intermediate_present = category_counts[CATEGORY_INTERMEDIATE] > 0
    windowed_present = category_counts[CATEGORY_WINDOWED] > 0
    final_present = category_counts[CATEGORY_FINAL] > 0
    labels_present = category_counts[CATEGORY_LABELS] > 0
    builders_present = category_counts[CATEGORY_BUILDER] > 0
    present_layers = [
        label
        for label, present in [
            ("raw", raw_present),
            ("intermediate", intermediate_present),
            ("windowed", windowed_present),
            ("final_numpy", final_present),
            ("labels_metadata", labels_present),
            ("builder_scripts", builders_present),
        ]
        if present
    ]
    if raw_present and final_present:
        root_verdict = "mixture_including_raw_and_final_artifacts"
    elif raw_present:
        root_verdict = "raw_or_simulation_artifacts_present_without_recognized_final_arrays"
    elif final_present and (intermediate_present or windowed_present):
        root_verdict = "processed_and_final_artifacts_present_no_recognized_raw_outputs"
    elif final_present:
        root_verdict = "primarily_final_numpy_dataset_no_recognized_raw_outputs"
    else:
        root_verdict = "no_recognized_final_numpy_dataset_inventory_requires_manual_review"

    summary = {
        "script_version": SCRIPT_VERSION,
        "experiment_root": str(root),
        "inventory_generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "entry_count": len(rows),
        "file_count": entry_counts["file"],
        "directory_count": entry_counts["directory"],
        "symlink_count": entry_counts["symlink"],
        "total_file_bytes": sum(int(row["size_bytes"]) for row in file_rows),
        "total_file_mib": round(mib(sum(int(row["size_bytes"]) for row in file_rows)), 6),
        "category_counts": dict(sorted(category_counts.items())),
        "present_layers": present_layers,
        "experiment_root_content_verdict": root_verdict,
        "candidate_raw_artifact_count": len(raw_candidates),
        "candidate_builder_artifact_count": len(builder_candidates),
        "unknown_large_file_count": len(unknown_large),
        "largest_files": [
            {
                "relative_path": row["relative_path"],
                "size_bytes": row["size_bytes"],
                "classification": row["classification"],
                "npy_shape": row["npy_shape"],
                "npy_dtype": row["npy_dtype"],
            }
            for row in largest_files
        ],
        "walk_or_read_error_count": len(errors),
        "source_metadata_unchanged": metadata_unchanged,
        "source_metadata_change_details": metadata_details,
        "traversal_within_root": traversal_within_root,
        "external_symlink_followed": external_symlink_followed,
        "npy_headers_bounded": npy_headers_bounded,
        "text_prefixes_bounded": text_prefixes_bounded,
        "source_tree_write_operations": 0,
        "discovered_files_executed": False,
        "complete_numpy_arrays_loaded": False,
        "complete_text_files_scanned": False,
    }

    return InventoryResult(
        inventory_rows=rows,
        directory_rows=directory_rows,
        raw_candidates=raw_candidates,
        builder_candidates=builder_candidates,
        unknown_large=unknown_large,
        errors=errors,
        source_metadata_unchanged=metadata_unchanged,
        source_metadata_change_details=metadata_details,
        traversal_within_root=traversal_within_root,
        external_symlink_followed=external_symlink_followed,
        npy_headers_bounded=npy_headers_bounded,
        text_prefixes_bounded=text_prefixes_bounded,
        summary=summary,
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str], limit: int = 20) -> str:
    selected = list(rows[:limit])
    if not selected:
        return "_None found._\n"
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in selected:
        values = []
        for column in columns:
            value = str(row.get(column, "")).replace("|", "\\|").replace("\n", " ")
            values.append(value)
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body]) + "\n"


def build_report(result: InventoryResult) -> str:
    summary = result.summary
    counts = summary["category_counts"]
    largest = summary["largest_files"]
    lines = [
        "# Stage 8A — V3 Experiment-Root Inventory",
        "",
        "This stage performed a bounded, read-only inventory of the supplied experiment root.",
        "It did not execute discovered files, follow symbolic links, load complete NumPy arrays,",
        "or recursively search outside the experiment directory.",
        "",
        "## Root",
        "",
        f"- Experiment root: `{summary['experiment_root']}`",
        f"- Files: **{summary['file_count']}**",
        f"- Directories: **{summary['directory_count']}**",
        f"- Symbolic links recorded but not followed: **{summary['symlink_count']}**",
        f"- Total file size: **{summary['total_file_mib']:.3f} MiB**",
        f"- Initial content verdict: **{summary['experiment_root_content_verdict']}**",
        "",
        "## Classification counts",
        "",
    ]
    for category in sorted(FILE_CATEGORIES):
        lines.append(f"- `{category}`: **{counts.get(category, 0)}**")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Stage 8A establishes what artifacts exist and which files deserve targeted Stage 8B inspection.",
            "It does not yet prove feature equations, normalization scope, clipping behaviour, port semantics,",
            "window construction, label construction, or split provenance.",
            "",
            "## Candidate raw artifacts",
            "",
            markdown_table(
                result.raw_candidates,
                ["relative_path", "candidate_score", "classification", "candidate_reasons"],
                limit=30,
            ).rstrip(),
            "",
            "## Candidate builder artifacts",
            "",
            markdown_table(
                result.builder_candidates,
                ["relative_path", "candidate_score", "classification", "candidate_reasons"],
                limit=30,
            ).rstrip(),
            "",
            "## Largest files",
            "",
            markdown_table(
                largest,
                ["relative_path", "size_bytes", "classification", "npy_shape", "npy_dtype"],
                limit=20,
            ).rstrip(),
            "",
            "## Unknown large files",
            "",
            markdown_table(
                result.unknown_large,
                ["relative_path", "size_mib", "extension", "probable_role"],
                limit=30,
            ).rstrip(),
            "",
            "## Safety and integrity",
            "",
            f"- Traversal stayed lexically within root: **{result.traversal_within_root}**",
            f"- External symbolic link followed: **{result.external_symlink_followed}**",
            f"- Source size/mtime/mode metadata unchanged: **{result.source_metadata_unchanged}**",
            f"- NumPy header reads remained bounded: **{result.npy_headers_bounded}**",
            f"- Text-prefix reads remained bounded: **{result.text_prefixes_bounded}**",
            f"- Walk/read errors recorded: **{len(result.errors)}**",
            "",
            "## Next stage",
            "",
            "Stage 8B should inspect only the ranked candidate raw and builder artifacts, then trace",
            "the generation chain from simulator output to the final arrays.",
            "",
        ]
    )
    return "\n".join(lines)


def validation_checks(
    result: InventoryResult,
    *,
    text_prefix_bytes: int,
    max_npy_header_bytes: int,
) -> list[tuple[str, bool]]:
    file_rows = [row for row in result.inventory_rows if row["entry_type"] == "file"]
    file_paths = {str(row["relative_path"]) for row in file_rows}
    raw_paths = {str(row["relative_path"]) for row in result.raw_candidates}
    builder_paths = {str(row["relative_path"]) for row in result.builder_candidates}
    unknown_large_paths = {str(row["relative_path"]) for row in result.unknown_large}
    valid_file_categories = all(str(row["classification"]) in FILE_CATEGORIES for row in file_rows)
    candidate_subsets = raw_paths <= file_paths and builder_paths <= file_paths and unknown_large_paths <= file_paths
    text_bounds = all(
        int(row["text_prefix_bytes_read"] or 0) <= text_prefix_bytes
        for row in file_rows
    )
    npy_bounds = all(
        int(row["npy_header_bytes_read"] or 0) <= max_npy_header_bytes + 12
        for row in file_rows
        if str(row["extension"]) == ".npy"
    )
    return [
        ("experiment root exists and is a directory", True),
        ("traversal stayed within specified experiment root", result.traversal_within_root),
        ("symbolic links were not followed", not result.external_symlink_followed),
        ("source size/mtime/mode metadata unchanged", result.source_metadata_unchanged),
        ("no discovered file executed", result.summary["discovered_files_executed"] is False),
        ("no complete NumPy array loaded", result.summary["complete_numpy_arrays_loaded"] is False),
        ("text reads stayed within configured prefix bound", text_bounds and result.text_prefixes_bounded),
        ("NumPy header reads stayed within configured bound", npy_bounds and result.npy_headers_bounded),
        ("every regular file received one Stage 8A category", valid_file_categories),
        ("candidate and unknown-large tables are inventory subsets", candidate_subsets),
        ("directory summary contains root row", any(row["relative_directory"] == "." for row in result.directory_rows)),
        ("inventory row count matches summary", len(result.inventory_rows) == int(result.summary["entry_count"])),
        ("file row count matches summary", len(file_rows) == int(result.summary["file_count"])),
        ("source-tree write operations recorded as zero", result.summary["source_tree_write_operations"] == 0),
    ]


def output_paths(failure_root: Path) -> dict[str, Path]:
    return {
        "inventory": failure_root / "tables/stage8a_experiment_root_inventory.csv",
        "directory_summary": failure_root / "tables/stage8a_directory_summary.csv",
        "candidate_raw": failure_root / "tables/stage8a_candidate_raw_artifacts.csv",
        "candidate_builder": failure_root / "tables/stage8a_candidate_builder_artifacts.csv",
        "unknown_large": failure_root / "tables/stage8a_unknown_large_files.csv",
        "validation": failure_root / "logs/44_stage8a_validation.txt",
        "report": failure_root / "STAGE8A_EXPERIMENT_ROOT_INVENTORY_REPORT.md",
        "summary": failure_root / "tables/stage8a_summary.json",
    }


def stage_outputs(
    result: InventoryResult,
    failure_root: Path,
    *,
    text_prefix_bytes: int,
    max_npy_header_bytes: int,
) -> tuple[Path, dict[str, Path], list[tuple[str, bool]]]:
    failure_root.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=".stage8a_staging_", dir=failure_root))
    finals = output_paths(failure_root)
    staged = {name: stage_root / path.relative_to(failure_root) for name, path in finals.items()}
    checks = validation_checks(
        result,
        text_prefix_bytes=text_prefix_bytes,
        max_npy_header_bytes=max_npy_header_bytes,
    )
    overall = all(value for _, value in checks)
    validation_text = "\n".join(f"{name}: {value}" for name, value in checks)
    validation_text += f"\noverall success: {overall}\n"

    write_csv(staged["inventory"], result.inventory_rows, CSV_COLUMNS_INVENTORY)
    write_csv(staged["directory_summary"], result.directory_rows, CSV_COLUMNS_DIRECTORY)
    write_csv(staged["candidate_raw"], result.raw_candidates, CSV_COLUMNS_CANDIDATE)
    write_csv(staged["candidate_builder"], result.builder_candidates, CSV_COLUMNS_CANDIDATE)
    write_csv(staged["unknown_large"], result.unknown_large, CSV_COLUMNS_UNKNOWN_LARGE)
    write_text(staged["validation"], validation_text)
    write_text(staged["report"], build_report(result))
    summary = dict(result.summary)
    summary["validation_checks"] = {name: value for name, value in checks}
    summary["overall_success"] = overall
    write_json(staged["summary"], summary)
    return stage_root, staged, checks


def promote_outputs(
    stage_root: Path,
    staged: dict[str, Path],
    finals: dict[str, Path],
    *,
    overwrite: bool,
) -> None:
    existing = [path for path in finals.values() if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Stage 8A outputs already exist; inspect them or rerun intentionally with --overwrite:\n"
            + formatted
        )

    backup_root = stage_root / ".backup"
    promoted: list[Path] = []
    backups: list[tuple[Path, Path]] = []
    try:
        for name, final_path in finals.items():
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                backup_path = backup_root / final_path.relative_to(final_path.parents[2] if "tables" in final_path.parts or "logs" in final_path.parts else final_path.parent)
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(final_path, backup_path)
                backups.append((backup_path, final_path))
            os.replace(staged[name], final_path)
            promoted.append(final_path)
    except Exception:
        for promoted_path in reversed(promoted):
            try:
                promoted_path.unlink(missing_ok=True)
            except OSError:
                pass
        for backup_path, final_path in reversed(backups):
            if backup_path.exists():
                final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup_path, final_path)
        raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def execute_inventory(
    experiment_root: Path,
    failure_root: Path,
    *,
    text_prefix_bytes: int,
    max_npy_header_bytes: int,
    max_npz_members: int,
    large_file_threshold_mib: float,
    overwrite: bool,
) -> InventoryResult:
    threshold_bytes = int(large_file_threshold_mib * 1024 * 1024)
    result = inventory_experiment_root(
        experiment_root,
        text_prefix_bytes=text_prefix_bytes,
        max_npy_header_bytes=max_npy_header_bytes,
        max_npz_members=max_npz_members,
        large_file_threshold_bytes=threshold_bytes,
    )
    stage_root, staged, checks = stage_outputs(
        result,
        failure_root,
        text_prefix_bytes=text_prefix_bytes,
        max_npy_header_bytes=max_npy_header_bytes,
    )
    if not all(value for _, value in checks):
        failed = [name for name, value in checks if not value]
        shutil.rmtree(stage_root, ignore_errors=True)
        raise RuntimeError("Stage 8A validation failed before output promotion: " + "; ".join(failed))
    promote_outputs(
        stage_root,
        staged,
        output_paths(failure_root),
        overwrite=overwrite,
    )
    return result


def run_smoke_test() -> bool:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="stage8a_smoke_") as temporary:
        base = Path(temporary)
        experiment = base / "experiment"
        failure = base / "failure"
        external = base / "external"
        (experiment / "runA/m5out").mkdir(parents=True)
        (experiment / "intermediate").mkdir(parents=True)
        (experiment / "dataset").mkdir(parents=True)
        (experiment / "scripts").mkdir(parents=True)
        (experiment / "docs").mkdir(parents=True)
        external.mkdir()

        (experiment / "runA/m5out/stats.txt").write_text(
            "sim_ticks 1000\nsystem.network.flits_received 42\n",
            encoding="utf-8",
        )
        (experiment / "runA/m5out/config.ini").write_text(
            "[system]\nrouter_count=16\n",
            encoding="utf-8",
        )
        (experiment / "intermediate/epoch_router_features.csv").write_text(
            "epoch,router,ifd_in_norm\n0,0,0.5\n",
            encoding="utf-8",
        )
        np.save(experiment / "dataset/x.npy", np.zeros((3, 16, 8, 24), dtype=np.float32))
        np.save(experiment / "dataset/y_graph.npy", np.array([0, 1, 0], dtype=np.int64))
        np.save(
            experiment / "dataset/feature_cols.npy",
            np.array(["ifd_in_norm", "ifd_out_norm"], dtype="U32"),
        )
        (experiment / "scripts/build_temporal_graphs.py").write_text(
            "import numpy as np\nwindow_size = 8\nstride = 1\nnp.save('x.npy', [])\n",
            encoding="utf-8",
        )
        (experiment / "docs/README.md").write_text(
            "paper1_temporal_graphs_ports_v3 manifest\n",
            encoding="utf-8",
        )
        with (experiment / "mystery.blob").open("wb") as handle:
            handle.truncate(2 * 1024 * 1024)
        symlink_supported = True
        try:
            os.symlink(external, experiment / "external_link", target_is_directory=True)
        except (OSError, NotImplementedError):
            symlink_supported = False

        before = snapshot_metadata(experiment)
        result = execute_inventory(
            experiment,
            failure,
            text_prefix_bytes=4096,
            max_npy_header_bytes=65536,
            max_npz_members=20,
            large_file_threshold_mib=1.0,
            overwrite=False,
        )
        after = snapshot_metadata(experiment)
        row_by_path = {str(row["relative_path"]): row for row in result.inventory_rows}
        checks["read-only traversal smoke test"] = before == after and result.source_metadata_unchanged
        checks["raw artifact classification smoke test"] = (
            row_by_path["runA/m5out/stats.txt"]["classification"] == CATEGORY_RAW
        )
        checks["intermediate classification smoke test"] = (
            row_by_path["intermediate/epoch_router_features.csv"]["classification"]
            == CATEGORY_INTERMEDIATE
        )
        checks["final NumPy header smoke test"] = (
            row_by_path["dataset/x.npy"]["classification"] == CATEGORY_FINAL
            and row_by_path["dataset/x.npy"]["npy_shape"] == "(3, 16, 8, 24)"
            and row_by_path["dataset/x.npy"]["npy_dtype"] == "float32"
        )
        checks["builder candidate smoke test"] = any(
            row["relative_path"] == "scripts/build_temporal_graphs.py"
            for row in result.builder_candidates
        )
        checks["external symlink not followed"] = (
            not result.external_symlink_followed
            and (
                not symlink_supported
                or row_by_path["external_link"]["entry_type"] == "symlink"
            )
        )
        checks["large unknown file smoke test"] = any(
            row["relative_path"] == "mystery.blob" for row in result.unknown_large
        )
        required_outputs = output_paths(failure)
        checks["atomic output smoke test"] = all(path.exists() for path in required_outputs.values())

    for name, value in checks.items():
        print(f"{name}: {value}")
    success = all(checks.values())
    print(f"smoke-test-only success: {success}")
    return success


def print_run_summary(result: InventoryResult, failure_root: Path) -> None:
    summary = result.summary
    print(f"script version: {SCRIPT_VERSION}", flush=True)
    print(f"experiment root: {summary['experiment_root']}", flush=True)
    print(f"failure root: {failure_root.expanduser().absolute()}", flush=True)
    print("read-only source traversal: True", flush=True)
    print("symbolic links followed: False", flush=True)
    print("discovered files executed: False", flush=True)
    print("complete NumPy arrays loaded: False", flush=True)
    print(f"entries inventoried: {summary['entry_count']}", flush=True)
    print(f"regular files inventoried: {summary['file_count']}", flush=True)
    print(f"directories inventoried: {summary['directory_count']}", flush=True)
    print(f"symlinks recorded: {summary['symlink_count']}", flush=True)
    print(f"candidate raw artifacts: {summary['candidate_raw_artifact_count']}", flush=True)
    print(f"candidate builder artifacts: {summary['candidate_builder_artifact_count']}", flush=True)
    print(f"unknown large files: {summary['unknown_large_file_count']}", flush=True)
    print(f"experiment-root content verdict: {summary['experiment_root_content_verdict']}", flush=True)
    print("Stage 8A output files:", flush=True)
    for name, path in output_paths(failure_root).items():
        print(f"  {name}: {path}", flush=True)
    print("overall success: True", flush=True)


def main() -> None:
    args = parse_args()
    if args.smoke_test_only:
        if not run_smoke_test():
            raise SystemExit(1)
        return

    print("Stage 8A safety boundary:", flush=True)
    print("  source tree writes: False", flush=True)
    print("  discovered file execution: False", flush=True)
    print("  symbolic-link traversal: False", flush=True)
    print("  complete NumPy array loading: False", flush=True)
    print(f"  maximum text prefix bytes: {args.text_prefix_bytes}", flush=True)
    print(f"  maximum NPY header bytes: {args.max_npy_header_bytes}", flush=True)

    result = execute_inventory(
        args.experiment_root,
        args.failure_root,
        text_prefix_bytes=args.text_prefix_bytes,
        max_npy_header_bytes=args.max_npy_header_bytes,
        max_npz_members=args.max_npz_members,
        large_file_threshold_mib=args.large_file_threshold_mib,
        overwrite=args.overwrite,
    )
    print_run_summary(result, args.failure_root)


if __name__ == "__main__":
    main()
