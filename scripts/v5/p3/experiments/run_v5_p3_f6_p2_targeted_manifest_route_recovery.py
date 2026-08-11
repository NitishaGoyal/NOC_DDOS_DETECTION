from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STAGE = "V5_P3_F6_P2_TARGETED_MANIFEST_ROUTE_RECOVERY"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "VALIDATION-EXPLORATORY"

EXPECTED_ITEMS = 13863
EXPECTED_ACTIVE = 3822
EXPECTED_CONTROL = 10041

PAIR_ALIASES = (
    "pair_id",
    "pairid",
    "matched_pair_id",
    "pair_index",
    "pair_idx",
    "attack_control_pair",
    "control_attack_pair",
    "pair_key",
)
WINDOW_ALIASES = (
    "window_start",
    "window_start_epoch",
    "window_epoch_start",
    "sample_window_start",
    "start_epoch",
    "window_id",
    "window_index",
    "window_idx",
)
MEMBER_ALIASES = (
    "pair_member",
    "pair_role",
    "member",
    "run_role",
    "control_attack",
    "attack_control",
    "is_attack",
    "active",
)
INDEX_ALIASES = (
    "sample_index",
    "validation_index",
    "split_index",
    "item_index",
    "local_index",
    "index",
)
ATTACK_ALIASES = (
    "y_attack",
    "attack_label",
    "graph_label",
    "is_attack",
    "active",
)
COUNT_ALIASES = (
    "y_attacker_count",
    "attacker_count",
    "attack_count",
    "count_label",
)
SPLIT_ALIASES = (
    "split",
    "dataset_split",
    "partition",
    "subset",
)
RUN_ALIASES = (
    "run_id",
    "run_name",
    "run",
    "trace_id",
)

VALIDATION_VALUES = {
    "validation",
    "val",
    "valid",
    "dev",
    "a_validation",
    "tranche_a_validation",
}

TEXT_SUFFIXES = {
    ".csv",
    ".tsv",
    ".jsonl",
    ".ndjson",
}


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
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
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


def import_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def semantic_score(name: str, aliases: tuple[str, ...]) -> int:
    normalized = normalize(name)
    score = 0
    for alias in aliases:
        target = normalize(alias)
        if normalized == target:
            score = max(score, 100)
        elif normalized.endswith("_" + target):
            score = max(score, 90)
        elif target in normalized:
            score = max(score, 70)
        else:
            target_tokens = set(target.split("_"))
            name_tokens = set(normalized.split("_"))
            overlap = len(target_tokens & name_tokens)
            if overlap:
                score = max(score, 20 + 10 * overlap)
    return score


def canonical_scalar(value: Any) -> str | None:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            value = value.decode("latin1", errors="replace")
    if value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    if text == "" or normalize(text) in ("nan", "none", "null"):
        return None
    return text


def classify_member(value: Any) -> str | None:
    text = normalize(str(value))
    if text in (
        "attack",
        "attacked",
        "active",
        "malicious",
        "1",
        "true",
        "yes",
    ):
        return "attack"
    if text in (
        "control",
        "benign",
        "normal",
        "clean",
        "baseline",
        "0",
        "false",
        "no",
    ):
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


def parse_integer_array(values: np.ndarray) -> np.ndarray | None:
    array = np.asarray(values).reshape(-1)
    parsed = []
    for value in array:
        try:
            numeric = float(value)
        except Exception:
            return None
        if not np.isfinite(numeric):
            return None
        integer = int(numeric)
        if abs(numeric - integer) > 1e-9:
            return None
        parsed.append(integer)
    return np.asarray(parsed, dtype=np.int64)


def read_npy_header(
    handle: Any,
) -> tuple[tuple[int, int], tuple[int, ...], bool, np.dtype]:
    major, minor = np.lib.format.read_magic(handle)
    version = (int(major), int(minor))

    if version == (1, 0):
        raw = handle.read(2)
        require(len(raw) == 2, "truncated npy 1.0 header length")
        header_length = struct.unpack("<H", raw)[0]
        encoding = "latin1"
    elif version in ((2, 0), (3, 0)):
        raw = handle.read(4)
        require(len(raw) == 4, "truncated npy 2/3 header length")
        header_length = struct.unpack("<I", raw)[0]
        encoding = "utf-8" if version == (3, 0) else "latin1"
    else:
        raise RuntimeError(f"unsupported npy version: {version}")

    require(
        0 < header_length <= 64 * 1024 * 1024,
        f"invalid npy header length: {header_length}",
    )
    header_bytes = handle.read(header_length)
    require(len(header_bytes) == header_length, "truncated npy header")

    import ast
    header = ast.literal_eval(header_bytes.decode(encoding).strip())
    require(
        isinstance(header, dict)
        and set(header) == {"descr", "fortran_order", "shape"},
        "invalid npy header dictionary",
    )
    shape = tuple(int(item) for item in header["shape"])
    return (
        version,
        shape,
        bool(header["fortran_order"]),
        np.dtype(header["descr"]),
    )


def npy_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        version, shape, fortran, dtype = read_npy_header(handle)
    return {
        "version": list(version),
        "shape": list(shape),
        "fortran_order": fortran,
        "dtype": str(dtype),
    }


def npz_member_inventory(path: Path) -> list[dict[str, Any]]:
    rows = []
    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.infolist():
            if not member.filename.endswith(".npy"):
                continue
            with archive.open(member, "r") as handle:
                version, shape, fortran, dtype = read_npy_header(handle)
            rows.append({
                "key": member.filename[:-4],
                "shape": list(shape),
                "dtype": str(dtype),
                "fortran_order": fortran,
                "version": list(version),
                "compressed_size": int(member.compress_size),
                "uncompressed_size": int(member.file_size),
            })
    return rows


class CandidateStore:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._arrays: dict[int, np.ndarray] = {}
        self._next_id = 0

    def add(
        self,
        *,
        name: str,
        source_group: str,
        origin: str,
        trust: str,
        values: Any,
        path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        try:
            array = np.asarray(values)
        except Exception:
            return

        if array.ndim == 0:
            return
        if array.shape[0] != EXPECTED_ITEMS:
            return
        if array.ndim > 2:
            return
        if array.size > EXPECTED_ITEMS * 32:
            return

        identifier = self._next_id
        self._next_id += 1
        self._arrays[identifier] = array

        record = {
            "id": identifier,
            "name": name,
            "source_group": source_group,
            "origin": origin,
            "trust": trust,
            "path": path,
            "shape": [int(item) for item in array.shape],
            "dtype": str(array.dtype),
            "size": int(array.size),
            "pair_score": semantic_score(name, PAIR_ALIASES),
            "window_score": semantic_score(name, WINDOW_ALIASES),
            "member_score": semantic_score(name, MEMBER_ALIASES),
            "index_score": semantic_score(name, INDEX_ALIASES),
            "attack_score": semantic_score(name, ATTACK_ALIASES),
            "count_score": semantic_score(name, COUNT_ALIASES),
            "details": details or {},
        }
        self.records.append(record)

    def array(self, identifier: int) -> np.ndarray:
        return self._arrays[identifier]


def construct_validation_dataset(
    loader_path: Path,
    data_root: Path,
) -> tuple[Any, dict[str, Any]]:
    module = import_source(loader_path, "_v5_p3_f6_p2_guarded_loader")
    attempts = []

    if hasattr(module, "_load_original_class") and hasattr(
        module,
        "_construct_original",
    ):
        dataset_class = module._load_original_class(data_root)
        for split in ("validation", "val", "VALIDATION"):
            try:
                dataset = module._construct_original(
                    dataset_class,
                    data_root,
                    split,
                    {},
                )
                length = len(dataset)
                attempts.append({
                    "route": "_load_original_class/_construct_original",
                    "split": split,
                    "length": int(length),
                    "status": "SUCCESS",
                })
                if length == EXPECTED_ITEMS:
                    return dataset, {
                        "route": "_load_original_class/_construct_original",
                        "split": split,
                        "dataset_class": (
                            f"{dataset.__class__.__module__}."
                            f"{dataset.__class__.__name__}"
                        ),
                        "attempts": attempts,
                    }
            except BaseException as exc:
                attempts.append({
                    "route": "_load_original_class/_construct_original",
                    "split": split,
                    "status": "FAILED",
                    "error": repr(exc),
                })

    raise RuntimeError(
        "could not construct validation dataset without item access; "
        f"attempts={attempts}"
    )


def object_inventory_and_candidates(
    dataset: Any,
    store: CandidateStore,
) -> dict[str, Any]:
    seen = set()
    inventory = []

    def visit(value: Any, path: str, depth: int) -> None:
        if depth > 4:
            return

        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)

        if isinstance(value, np.ndarray):
            inventory.append({
                "path": path,
                "type": type(value).__name__,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            })
            store.add(
                name=path,
                source_group="dataset_object",
                origin="dataset_object_attribute",
                trust="validation_dataset_order",
                values=value,
                details={"object_path": path},
            )
            return

        try:
            import torch
            if isinstance(value, torch.Tensor):
                inventory.append({
                    "path": path,
                    "type": "torch.Tensor",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                })
                if value.device.type == "cpu":
                    store.add(
                        name=path,
                        source_group="dataset_object",
                        origin="dataset_object_tensor_attribute",
                        trust="validation_dataset_order",
                        values=value.detach().numpy(),
                        details={"object_path": path},
                    )
                return
        except Exception:
            pass

        # pandas DataFrame/Series without importing pandas as a hard dependency.
        class_name = (
            f"{value.__class__.__module__}."
            f"{value.__class__.__name__}"
        )
        if class_name.endswith(".DataFrame"):
            inventory.append({
                "path": path,
                "type": class_name,
                "shape": list(value.shape),
                "columns": [str(column) for column in value.columns],
            })
            if len(value) == EXPECTED_ITEMS:
                for column in value.columns:
                    store.add(
                        name=f"{path}.{column}",
                        source_group=f"dataset_object:{path}",
                        origin="dataset_object_dataframe",
                        trust="validation_dataset_order",
                        values=value[column].to_numpy(),
                        details={"object_path": path, "column": str(column)},
                    )
            return

        if class_name.endswith(".Series"):
            try:
                shape = list(value.shape)
            except Exception:
                shape = None
            inventory.append({
                "path": path,
                "type": class_name,
                "shape": shape,
            })
            try:
                store.add(
                    name=path,
                    source_group="dataset_object",
                    origin="dataset_object_series",
                    trust="validation_dataset_order",
                    values=value.to_numpy(),
                    details={"object_path": path},
                )
            except Exception:
                pass
            return

        if isinstance(value, (list, tuple)):
            inventory.append({
                "path": path,
                "type": type(value).__name__,
                "length": len(value),
            })
            if len(value) == EXPECTED_ITEMS:
                try:
                    array = np.asarray(value)
                    if array.ndim <= 2:
                        store.add(
                            name=path,
                            source_group="dataset_object",
                            origin="dataset_object_sequence",
                            trust="validation_dataset_order",
                            values=array,
                            details={"object_path": path},
                        )
                except Exception:
                    pass

                # List of records.
                if value and isinstance(value[0], dict):
                    fields = sorted(
                        set().union(
                            *[
                                set(item)
                                for item in value[: min(100, len(value))]
                                if isinstance(item, dict)
                            ]
                        )
                    )
                    for field in fields:
                        column = [
                            item.get(field)
                            if isinstance(item, dict)
                            else None
                            for item in value
                        ]
                        store.add(
                            name=f"{path}.{field}",
                            source_group=f"dataset_object:{path}",
                            origin="dataset_object_record_sequence",
                            trust="validation_dataset_order",
                            values=np.asarray(column, dtype=object),
                            details={"object_path": path, "field": field},
                        )
            # Do not recursively traverse long sample lists.
            if len(value) <= 64:
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]", depth + 1)
            return

        if isinstance(value, dict):
            inventory.append({
                "path": path,
                "type": "dict",
                "keys_preview": [str(key) for key in list(value)[:50]],
                "length": len(value),
            })
            for key, child in list(value.items())[:200]:
                visit(child, f"{path}.{key}", depth + 1)
            return

        if isinstance(value, (str, bytes, int, float, bool, type(None), Path)):
            inventory.append({
                "path": path,
                "type": type(value).__name__,
                "value": str(value)[:500],
            })
            return

        if hasattr(value, "__dict__"):
            attributes = {
                key: child
                for key, child in vars(value).items()
                if not key.startswith("__")
            }
            inventory.append({
                "path": path,
                "type": class_name,
                "attribute_names": sorted(attributes),
            })
            for key, child in attributes.items():
                visit(child, f"{path}.{key}", depth + 1)

    visit(dataset, "dataset", 0)
    return {
        "dataset_type": (
            f"{dataset.__class__.__module__}."
            f"{dataset.__class__.__name__}"
        ),
        "entries": inventory,
    }


def scan_array_files(
    roots: list[Path],
    store: CandidateStore,
) -> list[dict[str, Any]]:
    inventory = []
    seen = set()
    filename_terms = (
        "pair",
        "window",
        "manifest",
        "metadata",
        "sample",
        "index",
        "validation",
        "member",
        "attack",
        "count",
        "run",
    )

    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            suffix = path.suffix.lower()
            low_name = normalize(path.name)
            if suffix not in (".npy", ".npz"):
                continue
            if not any(term in low_name for term in filename_terms):
                # Dataset root arrays are still inventoried when metadata-sized.
                if path.parent.resolve() != root.resolve():
                    continue

            try:
                if suffix == ".npy":
                    header = npy_header(path)
                    shape = header["shape"]
                    inventory.append({
                        "path": str(resolved),
                        "sha256": sha256_file(path),
                        "kind": "npy",
                        **header,
                    })
                    if (
                        shape
                        and shape[0] == EXPECTED_ITEMS
                        and len(shape) <= 2
                        and int(np.prod(shape)) <= EXPECTED_ITEMS * 32
                    ):
                        array = np.load(path, allow_pickle=False, mmap_mode="r")
                        store.add(
                            name=path.stem,
                            source_group=str(resolved),
                            origin="npy_file",
                            trust=(
                                "validation_named_file"
                                if "val" in low_name
                                else "unverified_file_order"
                            ),
                            values=array,
                            path=str(resolved),
                        )

                elif suffix == ".npz":
                    members = npz_member_inventory(path)
                    inventory.append({
                        "path": str(resolved),
                        "sha256": sha256_file(path),
                        "kind": "npz",
                        "members": members,
                    })
                    selected = [
                        row["key"]
                        for row in members
                        if (
                            row["shape"]
                            and row["shape"][0] == EXPECTED_ITEMS
                            and len(row["shape"]) <= 2
                            and int(np.prod(row["shape"]))
                            <= EXPECTED_ITEMS * 32
                            and max(
                                semantic_score(row["key"], PAIR_ALIASES),
                                semantic_score(row["key"], WINDOW_ALIASES),
                                semantic_score(row["key"], MEMBER_ALIASES),
                                semantic_score(row["key"], INDEX_ALIASES),
                                semantic_score(row["key"], ATTACK_ALIASES),
                                semantic_score(row["key"], COUNT_ALIASES),
                            )
                            > 0
                        )
                    ]
                    if selected:
                        with np.load(path, allow_pickle=False) as archive:
                            for key in selected:
                                store.add(
                                    name=key,
                                    source_group=str(resolved),
                                    origin="npz_member",
                                    trust=(
                                        "validation_named_file"
                                        if "val" in low_name
                                        else "unverified_file_order"
                                    ),
                                    values=np.asarray(archive[key]),
                                    path=str(resolved),
                                    details={"npz_key": key},
                                )
            except Exception as exc:
                inventory.append({
                    "path": str(resolved),
                    "kind": suffix.lstrip("."),
                    "error": repr(exc),
                })

    return inventory


def sniff_delimiter(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    sample = path.read_text(
        encoding="utf-8",
        errors="replace",
    )[:65536]
    try:
        return csv.Sniffer().sniff(
            sample,
            delimiters=",\t;|",
        ).delimiter
    except Exception:
        return ","


def scan_text_tables(
    roots: list[Path],
    store: CandidateStore,
) -> list[dict[str, Any]]:
    inventory = []
    seen = set()
    filename_terms = (
        "pair",
        "window",
        "manifest",
        "metadata",
        "sample",
        "validation",
        "aligned",
        "index",
    )

    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            low_name = normalize(path.name)
            if not any(term in low_name for term in filename_terms):
                continue

            try:
                if path.stat().st_size > 512 * 1024 * 1024:
                    continue

                if path.suffix.lower() in (".jsonl", ".ndjson"):
                    rows = []
                    fields = []
                    field_set = set()
                    with path.open(
                        "r",
                        encoding="utf-8",
                        errors="replace",
                    ) as handle:
                        for line in handle:
                            if not line.strip():
                                continue
                            value = json.loads(line)
                            if not isinstance(value, dict):
                                raise RuntimeError("non-object JSONL row")
                            rows.append(value)
                            for key in value:
                                if key not in field_set:
                                    field_set.add(key)
                                    fields.append(str(key))
                    delimiter = "jsonl"
                else:
                    delimiter = sniff_delimiter(path)
                    with path.open(
                        "r",
                        encoding="utf-8",
                        errors="replace",
                        newline="",
                    ) as handle:
                        reader = csv.DictReader(handle, delimiter=delimiter)
                        if not reader.fieldnames:
                            continue
                        fields = [str(field) for field in reader.fieldnames]
                        rows = [dict(row) for row in reader]

                split_key = next(
                    (
                        field for field in fields
                        if semantic_score(field, SPLIT_ALIASES) >= 70
                    ),
                    None,
                )
                filtered = rows
                if split_key is not None:
                    validation_rows = [
                        row for row in rows
                        if normalize(str(row.get(split_key, "")))
                        in VALIDATION_VALUES
                    ]
                    if validation_rows:
                        filtered = validation_rows

                inventory.append({
                    "path": str(resolved),
                    "sha256": sha256_file(path),
                    "fields": fields,
                    "input_rows": len(rows),
                    "validation_rows": len(filtered),
                    "split_key": split_key,
                    "delimiter": delimiter,
                })

                if len(filtered) != EXPECTED_ITEMS:
                    continue

                source_group = str(resolved)
                for field in fields:
                    values = np.asarray(
                        [row.get(field) for row in filtered],
                        dtype=object,
                    )
                    store.add(
                        name=field,
                        source_group=source_group,
                        origin="text_table_column",
                        trust=(
                            "validation_filtered_table"
                            if split_key is not None
                            else "unverified_table_order"
                        ),
                        values=values,
                        path=str(resolved),
                        details={
                            "field": field,
                            "split_key": split_key,
                        },
                    )
            except Exception as exc:
                inventory.append({
                    "path": str(resolved),
                    "error": repr(exc),
                })

    return inventory


def source_mining(
    roots: list[Path],
) -> dict[str, Any]:
    token_sets = (
        ("pair_id", "window_start"),
        ("pair_member", "pair_id"),
        ("matched", "control", "attack"),
    )
    rows = []
    path_literals = set()

    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in (
                ".py",
                ".json",
                ".txt",
                ".md",
                ".yaml",
                ".yml",
            ):
                continue
            try:
                if path.stat().st_size > 16 * 1024 * 1024:
                    continue
                text = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception:
                continue

            low = text.lower()
            if not any(all(token in low for token in tokens) for tokens in token_sets):
                continue

            lines = text.splitlines()
            matching_lines = []
            for index, line in enumerate(lines, start=1):
                low_line = line.lower()
                if any(
                    token in low_line
                    for token in (
                        "pair_id",
                        "pair_member",
                        "window_start",
                        "matched_control",
                    )
                ):
                    matching_lines.append({
                        "line": index,
                        "text": line[:1000],
                    })

                for match in re.findall(
                    r"""["']([^"']+\.(?:csv|tsv|json|jsonl|npy|npz))["']""",
                    line,
                    flags=re.IGNORECASE,
                ):
                    path_literals.add(match)

            rows.append({
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "matching_lines": matching_lines[:100],
            })

    return {
        "source_files": rows,
        "referenced_path_literals": sorted(path_literals),
    }


def candidate_summary(store: CandidateStore) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in record.items()
            if key != "id"
        }
        for record in store.records
    ]


def aligned_values(
    store: CandidateStore,
    candidate: dict[str, Any],
    index_candidate: dict[str, Any] | None,
) -> np.ndarray | None:
    array = np.asarray(store.array(candidate["id"])).reshape(EXPECTED_ITEMS, -1)
    if array.shape[1] != 1:
        return None
    values = array[:, 0]

    if index_candidate is None:
        return values

    indices = parse_integer_array(store.array(index_candidate["id"]))
    if indices is None:
        return None
    if set(indices.tolist()) != set(range(EXPECTED_ITEMS)):
        return None

    reordered = np.empty(EXPECTED_ITEMS, dtype=values.dtype)
    reordered[indices] = values
    return reordered


def same_source_candidates(
    records: list[dict[str, Any]],
    source_group: str,
    score_key: str,
    minimum: int = 1,
) -> list[dict[str, Any]]:
    rows = [
        record for record in records
        if (
            record["source_group"] == source_group
            and record[score_key] >= minimum
        )
    ]
    rows.sort(key=lambda row: row[score_key], reverse=True)
    return rows


def evaluate_mapping_route(
    *,
    store: CandidateStore,
    pair_candidate: dict[str, Any],
    window_candidate: dict[str, Any] | None,
    index_candidate: dict[str, Any] | None,
    attack_candidate: dict[str, Any] | None,
    fresh_attack: np.ndarray,
    fresh_count: np.ndarray,
) -> dict[str, Any]:
    route = {
        "pair_candidate": {
            key: value for key, value in pair_candidate.items() if key != "id"
        },
        "window_candidate": (
            {
                key: value
                for key, value in window_candidate.items()
                if key != "id"
            }
            if window_candidate is not None
            else None
        ),
        "index_candidate": (
            {
                key: value
                for key, value in index_candidate.items()
                if key != "id"
            }
            if index_candidate is not None
            else None
        ),
        "attack_candidate": (
            {
                key: value
                for key, value in attack_candidate.items()
                if key != "id"
            }
            if attack_candidate is not None
            else None
        ),
        "usable": False,
    }

    pair_values = aligned_values(store, pair_candidate, index_candidate)
    if pair_values is None:
        route["failure"] = "pair_alignment_failed"
        return route

    if window_candidate is not None:
        window_values = aligned_values(
            store,
            window_candidate,
            index_candidate,
        )
        if window_values is None:
            route["failure"] = "window_alignment_failed"
            return route
    else:
        window_values = np.asarray([""] * EXPECTED_ITEMS, dtype=object)

    if attack_candidate is not None:
        attack_values = aligned_values(
            store,
            attack_candidate,
            index_candidate,
        )
        if attack_values is None:
            route["failure"] = "attack_alignment_failed"
            return route
        parsed_attack = np.asarray(
            [
                1 if classify_member(value) == "attack"
                else 0 if classify_member(value) == "control"
                else -1
                for value in attack_values
            ],
            dtype=np.int64,
        )
        if np.any(parsed_attack < 0):
            route["failure"] = "unrecognized_attack_labels"
            return route
        attack_alignment = bool(np.array_equal(parsed_attack, fresh_attack))
        route["attack_label_alignment_exact"] = attack_alignment
        if not attack_alignment:
            route["failure"] = "source_attack_labels_do_not_match_fresh_order"
            return route
    else:
        trust = pair_candidate["trust"]
        route["attack_label_alignment_exact"] = None
        route["order_trust"] = trust
        if trust not in (
            "validation_dataset_order",
            "validation_filtered_table",
            "validation_named_file",
        ):
            route["failure"] = "row_order_unproven"
            return route

    identities = []
    invalid_identity_values = 0
    for pair, window in zip(pair_values, window_values):
        pair_text = canonical_scalar(pair)
        window_text = canonical_scalar(window)
        if pair_text is None:
            invalid_identity_values += 1
            identities.append(None)
            continue
        identities.append((pair_text, window_text or ""))

    if invalid_identity_values:
        route["failure"] = "empty_or_invalid_pair_identity"
        route["invalid_identity_value_count"] = invalid_identity_values
        return route

    grouped: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: {"attack": [], "control": []}
    )
    for index, identity in enumerate(identities):
        member = "attack" if int(fresh_attack[index]) == 1 else "control"
        grouped[identity][member].append(index)

    attack_groups = {
        identity: bucket
        for identity, bucket in grouped.items()
        if bucket["attack"]
    }
    control_only_groups = sum(
        1
        for bucket in grouped.values()
        if not bucket["attack"] and bucket["control"]
    )

    invalid_groups = {}
    mapping = []
    for identity, bucket in attack_groups.items():
        if len(bucket["attack"]) != 1 or len(bucket["control"]) != 1:
            invalid_groups[str(identity)] = {
                "attack_count": len(bucket["attack"]),
                "control_count": len(bucket["control"]),
                "attack_indices": bucket["attack"][:20],
                "control_indices": bucket["control"][:20],
            }
            continue
        attack_index = bucket["attack"][0]
        control_index = bucket["control"][0]
        mapping.append({
            "attack_index": attack_index,
            "control_index": control_index,
            "pair_id": identity[0],
            "window_start": identity[1],
            "attacker_count": int(fresh_count[attack_index]),
            "source_group": pair_candidate["source_group"],
            "pair_source_name": pair_candidate["name"],
            "window_source_name": (
                window_candidate["name"]
                if window_candidate is not None
                else None
            ),
        })

    mapping.sort(key=lambda row: row["attack_index"])
    expected_attack_indices = np.flatnonzero(fresh_attack == 1).tolist()
    mapped_attack_indices = [row["attack_index"] for row in mapping]
    controls = [row["control_index"] for row in mapping]

    all_active_mapped = mapped_attack_indices == expected_attack_indices
    unique_controls = len(set(controls))
    strata = Counter(row["attacker_count"] for row in mapping)

    route.update({
        "identity_type": (
            "pair_id+window"
            if window_candidate is not None
            else "pair_id"
        ),
        "total_identity_groups": len(grouped),
        "attack_identity_groups": len(attack_groups),
        "control_only_identity_groups": control_only_groups,
        "invalid_attack_group_count": len(invalid_groups),
        "invalid_attack_group_preview": dict(
            list(invalid_groups.items())[:20]
        ),
        "mapped_attack_item_count": len(mapping),
        "all_active_items_mapped": all_active_mapped,
        "unique_control_item_count": unique_controls,
        "control_reuse_count": len(controls) - unique_controls,
        "attacker_count_strata": {
            str(key): int(value)
            for key, value in sorted(strata.items())
        },
    })

    usable = bool(
        len(mapping) == EXPECTED_ACTIVE
        and all_active_mapped
        and len(invalid_groups) == 0
        and unique_controls == EXPECTED_ACTIVE
        and all(strata.get(value, 0) >= 128 for value in (1, 2, 3, 4))
    )
    route["usable"] = usable
    if usable:
        route["mapping"] = mapping
    else:
        route["failure"] = "mapping_acceptance_gate_failed"
    return route


def resolve_routes(
    store: CandidateStore,
    fresh_attack: np.ndarray,
    fresh_count: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = store.records
    pair_candidates = sorted(
        [row for row in records if row["pair_score"] > 0],
        key=lambda row: row["pair_score"],
        reverse=True,
    )[:40]

    evaluated = []
    usable = []
    seen_signatures = set()

    for pair in pair_candidates:
        windows = same_source_candidates(
            records,
            pair["source_group"],
            "window_score",
        )[:10]
        indices = same_source_candidates(
            records,
            pair["source_group"],
            "index_score",
        )[:5]
        attacks = same_source_candidates(
            records,
            pair["source_group"],
            "attack_score",
        )[:5]

        window_options = [None] + windows
        index_options = [None] + indices
        attack_options = [None] + attacks

        for window in window_options:
            for index_candidate in index_options:
                for attack_candidate in attack_options:
                    signature = (
                        pair["id"],
                        window["id"] if window else None,
                        index_candidate["id"] if index_candidate else None,
                        attack_candidate["id"] if attack_candidate else None,
                    )
                    if signature in seen_signatures:
                        continue
                    seen_signatures.add(signature)

                    route = evaluate_mapping_route(
                        store=store,
                        pair_candidate=pair,
                        window_candidate=window,
                        index_candidate=index_candidate,
                        attack_candidate=attack_candidate,
                        fresh_attack=fresh_attack,
                        fresh_count=fresh_count,
                    )
                    evaluated.append(route)
                    if route["usable"]:
                        usable.append(route)

                    if len(evaluated) >= 5000:
                        return evaluated, usable

    return evaluated, usable


def mapping_fingerprint(
    mapping: list[dict[str, Any]],
) -> str:
    pairs = [
        [int(row["attack_index"]), int(row["control_index"])]
        for row in mapping
    ]
    payload = json.dumps(
        pairs,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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

    feature_root = repo / (
        "reports/v5/p3_experiments/f0_d70_feature_study"
    )
    ig_dir = feature_root / "integrated_gradients"
    metric_dir = feature_root / "metric_adapter"
    permutation_dir = feature_root / "permutation"

    p1_report_path = ig_dir / (
        "V5_P3_F6_P1_MATCHED_CONTROL_IDENTITY_REVIEW_REPORT.json"
    )
    p1_lock_path = ig_dir / (
        "V5_P3_F6_P1_MATCHED_CONTROL_IDENTITY_REVIEW_LOCK.json"
    )
    p0_protocol_path = ig_dir / (
        "F6_P0_TASK_SPECIFIC_INTEGRATED_GRADIENTS_PROTOCOL.json"
    )
    p0_source_path = ig_dir / (
        "F6_P0_FROZEN_EXECUTION_SOURCE_AND_RUNTIME_CONTRACT.json"
    )
    p1_candidate_path = ig_dir / (
        "F6_P1_MATCHED_CONTROL_CANDIDATE_EVALUATIONS.json"
    )
    route_path = permutation_dir / (
        "F5_P0_EXECUTION_ROUTE_SOURCE_INVENTORY.json"
    )
    fresh_cert_path = metric_dir / (
        "F4M_R3A_FRESH_REPLAY_14_METRIC_CERTIFICATION.json"
    )

    p1_report, p1_lock = verify_report_lock(
        p1_report_path,
        p1_lock_path,
    )
    require(
        p1_lock.get("mapping_resolved") is False,
        "F6-P2 expected unresolved P1 mapping",
    )
    require(
        p1_lock.get("actual_F6_execution_authorized") is False,
        "F6 execution unexpectedly authorized before P2",
    )
    require(p1_lock.get("F7_authorized") is False, "F7 must remain held")

    for path in (
        p0_protocol_path,
        p0_source_path,
        p1_candidate_path,
        route_path,
        fresh_cert_path,
    ):
        require(path.is_file(), f"required artifact missing: {path}")

    p0_protocol = load_json(p0_protocol_path)
    p0_source = load_json(p0_source_path)
    p1_candidates = load_json(p1_candidate_path)
    route = load_json(route_path)
    fresh_cert = load_json(fresh_cert_path)

    require(
        p0_protocol["baseline"]["type"]
        == "matched_control_feature_tensor",
        "matched-control protocol changed",
    )
    require(
        p0_protocol["sample_selection"]["total_items"] == 512,
        "F6 sample count changed",
    )
    require(
        p0_protocol["integrated_gradients"]["steps"] == 64,
        "F6 integration steps changed",
    )

    fresh_npz = Path(fresh_cert["npz"]).resolve()
    require(fresh_npz.is_file(), f"fresh validation NPZ missing: {fresh_npz}")
    require(
        sha256_file(fresh_npz) == fresh_cert["npz_sha256"],
        "fresh validation NPZ hash changed",
    )
    with np.load(fresh_npz, allow_pickle=False) as archive:
        fresh_attack = np.asarray(archive["y_attack"]).astype(np.int64)
        fresh_count = np.asarray(
            archive["y_attacker_count"]
        ).astype(np.int64)

    require(fresh_attack.shape == (EXPECTED_ITEMS,), "attack label shape changed")
    require(fresh_count.shape == (EXPECTED_ITEMS,), "count label shape changed")
    require(int(np.sum(fresh_attack == 1)) == EXPECTED_ACTIVE, "active count changed")
    require(int(np.sum(fresh_attack == 0)) == EXPECTED_CONTROL, "control count changed")

    loader_path = Path(
        route["certified_files"]["loader"]["path"]
    ).resolve()
    require(loader_path.is_file(), f"guarded loader missing: {loader_path}")
    require(
        sha256_file(loader_path)
        == route["certified_files"]["loader"]["actual_sha256"],
        "guarded loader hash changed",
    )

    dataset, dataset_provenance = construct_validation_dataset(
        loader_path,
        data_root,
    )
    require(len(dataset) == EXPECTED_ITEMS, "validation dataset length changed")

    store = CandidateStore()
    object_inventory = object_inventory_and_candidates(dataset, store)

    roots = [
        data_root,
        repo / "reports/v5/p3_experiments/f0_d70_feature_study",
        repo / "reports/v5/p3_d1_immutable_validation_logit_export",
        repo / "data/processed/v5",
    ]
    array_inventory = scan_array_files(roots, store)
    table_inventory = scan_text_tables(roots, store)

    mining_roots = [
        repo / "scripts/v5/p3",
        repo / "reports/v5/p3_experiments/f0_d70_feature_study/paired_analysis",
        repo / "src/data",
        data_root,
    ]
    source_evidence = source_mining(mining_roots)

    evaluated_routes, usable_routes = resolve_routes(
        store,
        fresh_attack,
        fresh_count,
    )

    fingerprint_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route_result in usable_routes:
        fingerprint = mapping_fingerprint(route_result["mapping"])
        route_result["mapping_fingerprint_sha256"] = fingerprint
        fingerprint_groups[fingerprint].append(route_result)

    mapping_resolved = len(fingerprint_groups) == 1
    selected_route = None
    selected_mapping = None
    selected_fingerprint = None

    if mapping_resolved:
        selected_fingerprint, equivalent_routes = next(
            iter(fingerprint_groups.items())
        )
        equivalent_routes.sort(
            key=lambda row: (
                0 if row["identity_type"] == "pair_id+window" else 1,
                0 if row["attack_label_alignment_exact"] is True else 1,
                row["pair_candidate"]["pair_score"] * -1,
                row["pair_candidate"]["name"],
            )
        )
        selected_route = equivalent_routes[0]
        selected_mapping = selected_route["mapping"]

    p1_failure_distribution = Counter(
        str(row.get("failure", "unknown"))
        for row in p1_candidates.get("evaluations", [])
    )
    p2_failure_distribution = Counter(
        str(row.get("failure", "usable" if row.get("usable") else "unknown"))
        for row in evaluated_routes
    )

    object_path = output_dir / "F6_P2_VALIDATION_DATASET_OBJECT_INVENTORY.json"
    array_path = output_dir / "F6_P2_METADATA_ARRAY_FILE_INVENTORY.json"
    table_path = output_dir / "F6_P2_METADATA_TABLE_INVENTORY.json"
    source_path = output_dir / "F6_P2_PAIRING_SOURCE_CODE_EVIDENCE.json"
    candidate_path = output_dir / "F6_P2_METADATA_CANDIDATE_INVENTORY.json"
    routes_path = output_dir / "F6_P2_MAPPING_ROUTE_EVALUATIONS.json"
    mapping_json_path = output_dir / "F6_P2_FROZEN_ATTACK_TO_CONTROL_MAPPING.json"
    mapping_csv_path = output_dir / "F6_P2_FROZEN_ATTACK_TO_CONTROL_MAPPING.csv"
    decision_path = output_dir / "F6_P2_TARGETED_MANIFEST_ROUTE_DECISION.json"

    atomic_json(
        object_path,
        {
            "dataset_provenance": dataset_provenance,
            "dataset_object_inventory": object_inventory,
            "validation_item_getitem_called": False,
            "validation_feature_tensors_loaded": False,
        },
    )
    atomic_json(array_path, array_inventory)
    atomic_json(table_path, table_inventory)
    atomic_json(source_path, source_evidence)
    atomic_json(
        candidate_path,
        {
            "candidate_count": len(store.records),
            "candidates": candidate_summary(store),
        },
    )

    compact_routes = []
    for row in evaluated_routes:
        compact_routes.append({
            key: value
            for key, value in row.items()
            if key != "mapping"
        })
    atomic_json(
        routes_path,
        {
            "evaluated_route_count": len(evaluated_routes),
            "usable_route_count": len(usable_routes),
            "unique_mapping_fingerprint_count": len(fingerprint_groups),
            "P1_failure_distribution": dict(p1_failure_distribution),
            "P2_failure_distribution": dict(p2_failure_distribution),
            "mapping_fingerprints": {
                fingerprint: [
                    {
                        "identity_type": row["identity_type"],
                        "pair_candidate": row["pair_candidate"],
                        "window_candidate": row["window_candidate"],
                        "index_candidate": row["index_candidate"],
                        "attack_candidate": row["attack_candidate"],
                    }
                    for row in routes
                ]
                for fingerprint, routes in fingerprint_groups.items()
            },
            "routes": compact_routes,
        },
    )

    if selected_mapping is not None and selected_route is not None:
        strata = Counter(
            row["attacker_count"] for row in selected_mapping
        )
        mapping_document = {
            "status": "FROZEN",
            "classification": CLASSIFICATION,
            "mapping_fingerprint_sha256": selected_fingerprint,
            "identity_type": selected_route["identity_type"],
            "pair_candidate": selected_route["pair_candidate"],
            "window_candidate": selected_route["window_candidate"],
            "index_candidate": selected_route["index_candidate"],
            "attack_candidate": selected_route["attack_candidate"],
            "attack_item_count": len(selected_mapping),
            "unique_control_item_count": len(
                set(row["control_index"] for row in selected_mapping)
            ),
            "attacker_count_strata": {
                str(key): int(value)
                for key, value in sorted(strata.items())
            },
            "all_active_items_mapped": True,
            "mapping": selected_mapping,
        }
        atomic_json(mapping_json_path, mapping_document)
        write_csv(mapping_csv_path, selected_mapping)

    actual_F6_execution_authorized = bool(
        mapping_resolved
        and selected_mapping is not None
        and len(selected_mapping) == EXPECTED_ACTIVE
        and len(
            set(row["control_index"] for row in selected_mapping)
        )
        == EXPECTED_ACTIVE
    )

    atomic_json(
        decision_path,
        {
            "F6_P1_complete": True,
            "prior_manifest_search_candidate_count": p1_candidates.get(
                "candidate_count"
            ),
            "dataset_object_constructed": True,
            "dataset_getitem_called": False,
            "metadata_candidate_count": len(store.records),
            "evaluated_route_count": len(evaluated_routes),
            "usable_route_count": len(usable_routes),
            "unique_mapping_fingerprint_count": len(fingerprint_groups),
            "mapping_resolved": mapping_resolved,
            "selected_mapping_fingerprint": selected_fingerprint,
            "mapped_attack_item_count": (
                len(selected_mapping)
                if selected_mapping is not None
                else 0
            ),
            "actual_F6_execution_authorized": actual_F6_execution_authorized,
            "F7_retraining_ablation_authorized": False,
            "feature_removal_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "sealed_test_access": False,
            "next_stage": (
                "V5_P3_F6_TASK_SPECIFIC_INTEGRATED_GRADIENTS"
                if actual_F6_execution_authorized
                else "V5_P3_F6_P3_DATASET_BUILDER_IDENTITY_ROUTE_RECOVERY"
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
            "Move beyond filename-based manifest discovery by constructing "
            "the guarded validation dataset without item access, inventorying "
            "its metadata-bearing object attributes, scanning metadata-sized "
            "NPY/NPZ/table artifacts, mining the prior F3 pairing source "
            "route, exhaustively evaluating pair/window/index/order "
            "combinations against preserved D1 labels, and freezing a unique "
            "attack-to-control mapping when one exists."
        ),
        "finding": {
            "P1_candidate_count": p1_candidates.get("candidate_count"),
            "P1_usable_candidate_count": p1_candidates.get(
                "usable_candidate_count"
            ),
            "P1_failure_distribution": dict(p1_failure_distribution),
            "dataset_provenance": dataset_provenance,
            "metadata_candidate_count": len(store.records),
            "evaluated_route_count": len(evaluated_routes),
            "usable_route_count": len(usable_routes),
            "unique_mapping_fingerprint_count": len(fingerprint_groups),
            "mapping_resolved": mapping_resolved,
            "selected_mapping_fingerprint": selected_fingerprint,
            "mapped_attack_item_count": (
                len(selected_mapping)
                if selected_mapping is not None
                else 0
            ),
        },
        "decision": {
            "F6_P2_complete": True,
            "actual_F6_execution_authorized": actual_F6_execution_authorized,
            "F7_retraining_ablation_authorized": False,
            "feature_removal_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F6_TASK_SPECIFIC_INTEGRATED_GRADIENTS"
                if actual_F6_execution_authorized
                else "V5_P3_F6_P3_DATASET_BUILDER_IDENTITY_ROUTE_RECOVERY"
            ),
        },
        "governance": {
            "validation_dataset_object_constructed": True,
            "validation_dataset_getitem_called": False,
            "validation_feature_tensors_loaded": False,
            "validation_metadata_arrays_loaded": True,
            "validation_label_payloads_loaded": True,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "model_weights_changed": False,
            "thresholds_changed": False,
            "metric_formulas_changed": False,
        },
        "artifacts": {
            "dataset_object_inventory": str(object_path),
            "array_file_inventory": str(array_path),
            "table_inventory": str(table_path),
            "pairing_source_evidence": str(source_path),
            "metadata_candidate_inventory": str(candidate_path),
            "mapping_route_evaluations": str(routes_path),
            "mapping_json": (
                str(mapping_json_path)
                if mapping_json_path.is_file()
                else None
            ),
            "mapping_csv": (
                str(mapping_csv_path)
                if mapping_csv_path.is_file()
                else None
            ),
            "decision": str(decision_path),
        },
        "provenance": {
            "F6_P1_report_sha256": sha256_file(p1_report_path),
            "F6_P1_lock_sha256": sha256_file(p1_lock_path),
            "F6_P0_protocol_sha256": sha256_file(p0_protocol_path),
            "F6_P0_source_contract_sha256": sha256_file(p0_source_path),
            "F6_P1_candidate_evaluations_sha256": sha256_file(
                p1_candidate_path
            ),
            "execution_route_inventory_sha256": sha256_file(route_path),
            "fresh_certification_sha256": sha256_file(fresh_cert_path),
            "fresh_npz_sha256": sha256_file(fresh_npz),
            "guarded_loader_sha256": sha256_file(loader_path),
            "installed_script_sha256": sha256_file(installed_script),
            "dataset_object_inventory_sha256": sha256_file(object_path),
            "array_file_inventory_sha256": sha256_file(array_path),
            "table_inventory_sha256": sha256_file(table_path),
            "pairing_source_evidence_sha256": sha256_file(source_path),
            "metadata_candidate_inventory_sha256": sha256_file(candidate_path),
            "mapping_route_evaluations_sha256": sha256_file(routes_path),
            "mapping_json_sha256": (
                sha256_file(mapping_json_path)
                if mapping_json_path.is_file()
                else None
            ),
            "mapping_csv_sha256": (
                sha256_file(mapping_csv_path)
                if mapping_csv_path.is_file()
                else None
            ),
            "decision_sha256": sha256_file(decision_path),
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
            "dataset_object_inventory_sha256": sha256_file(object_path),
            "array_file_inventory_sha256": sha256_file(array_path),
            "table_inventory_sha256": sha256_file(table_path),
            "pairing_source_evidence_sha256": sha256_file(source_path),
            "metadata_candidate_inventory_sha256": sha256_file(candidate_path),
            "mapping_route_evaluations_sha256": sha256_file(routes_path),
            "mapping_json_sha256": (
                sha256_file(mapping_json_path)
                if mapping_json_path.is_file()
                else None
            ),
            "mapping_csv_sha256": (
                sha256_file(mapping_csv_path)
                if mapping_csv_path.is_file()
                else None
            ),
            "decision_sha256": sha256_file(decision_path),
            "mapping_resolved": mapping_resolved,
            "actual_F6_execution_authorized": actual_F6_execution_authorized,
            "F7_authorized": False,
            "feature_removal_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "validation_dataset_getitem_called": False,
            "validation_feature_tensors_loaded": False,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print("F6_P1_complete=true")
    print(f"P1_candidate_count={p1_candidates.get('candidate_count')}")
    print(
        "P1_usable_candidate_count="
        f"{p1_candidates.get('usable_candidate_count')}"
    )
    print(f"P1_failure_distribution={dict(p1_failure_distribution)}")
    print("validation_dataset_object_constructed=true")
    print("validation_dataset_getitem_called=false")
    print("validation_feature_tensors_loaded=false")
    print(f"dataset_class={dataset_provenance['dataset_class']}")
    print(f"metadata_candidate_count={len(store.records)}")

    ranked_candidates = sorted(
        store.records,
        key=lambda row: max(
            row["pair_score"],
            row["window_score"],
            row["member_score"],
            row["index_score"],
            row["attack_score"],
        ),
        reverse=True,
    )
    for row in ranked_candidates[:40]:
        print(
            "metadata_candidate="
            f"name={row['name']}:origin={row['origin']}:"
            f"source_group={row['source_group']}:trust={row['trust']}:"
            f"shape={row['shape']}:dtype={row['dtype']}:"
            f"pair_score={row['pair_score']}:"
            f"window_score={row['window_score']}:"
            f"member_score={row['member_score']}:"
            f"index_score={row['index_score']}:"
            f"attack_score={row['attack_score']}"
        )

    print(f"evaluated_mapping_route_count={len(evaluated_routes)}")
    print(f"usable_mapping_route_count={len(usable_routes)}")
    print(
        "unique_mapping_fingerprint_count="
        f"{len(fingerprint_groups)}"
    )
    print(f"P2_failure_distribution={dict(p2_failure_distribution)}")
    print(f"mapping_resolved={str(mapping_resolved).lower()}")

    if selected_route is not None and selected_mapping is not None:
        print(
            "selected_identity_type="
            f"{selected_route['identity_type']}"
        )
        print(
            "selected_pair_candidate="
            f"{selected_route['pair_candidate']}"
        )
        print(
            "selected_window_candidate="
            f"{selected_route['window_candidate']}"
        )
        print(
            "selected_index_candidate="
            f"{selected_route['index_candidate']}"
        )
        print(
            "selected_attack_candidate="
            f"{selected_route['attack_candidate']}"
        )
        print(f"mapping_fingerprint_sha256={selected_fingerprint}")
        print(f"mapped_attack_item_count={len(selected_mapping)}")
        print(
            "unique_control_item_count="
            f"{len(set(row['control_index'] for row in selected_mapping))}"
        )
        print(
            "attacker_count_strata="
            f"{dict(Counter(row['attacker_count'] for row in selected_mapping))}"
        )
    else:
        print("selected_identity_type=None")
        print("mapped_attack_item_count=0")

    print(
        "actual_F6_execution_authorized="
        f"{str(actual_F6_execution_authorized).lower()}"
    )
    print("F7_retraining_ablation_authorized=false")
    print("feature_removal_authorized=false")
    print("hardware_reduction_claim_authorized=false")
    print("model_loaded=false")
    print("checkpoint_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={report['decision']['next_stage']}")
    print(f"dataset_object_inventory={object_path}")
    print(f"array_file_inventory={array_path}")
    print(f"table_inventory={table_path}")
    print(f"pairing_source_evidence={source_path}")
    print(f"metadata_candidate_inventory={candidate_path}")
    print(f"mapping_route_evaluations={routes_path}")
    if mapping_json_path.is_file():
        print(f"mapping_json={mapping_json_path}")
    if mapping_csv_path.is_file():
        print(f"mapping_csv={mapping_csv_path}")
    print(f"decision={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
