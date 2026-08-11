from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STAGE = "V5_P3_F6_P3_DATASET_BUILDER_IDENTITY_ROUTE_RECOVERY"
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
    "pair_key",
    "attack_control_pair",
    "control_attack_pair",
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
    "epoch_start",
)
RUN_ALIASES = (
    "run_id",
    "run_name",
    "run",
    "trace_id",
    "simulation_id",
    "sample_run_id",
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
INDEX_ALIASES = (
    "sample_index",
    "validation_index",
    "split_index",
    "item_index",
    "local_index",
    "index",
)
SPLIT_ALIASES = (
    "split",
    "dataset_split",
    "partition",
    "subset",
)
ATTACK_RUN_ALIASES = (
    "attack_run_id",
    "attack_run",
    "attacked_run_id",
    "active_run_id",
)
CONTROL_RUN_ALIASES = (
    "control_run_id",
    "control_run",
    "benign_run_id",
    "baseline_run_id",
)

VALIDATION_VALUES = {
    "validation",
    "val",
    "valid",
    "dev",
    "a_validation",
    "tranche_a_validation",
}

MAX_TABLE_ROWS = 100000
MAX_TABLE_COLUMNS = 256
MAX_ARRAY_ELEMENTS = 8_000_000


@dataclass
class Table:
    table_id: str
    origin: str
    trust: str
    path: str | None
    columns: dict[str, np.ndarray]
    details: dict[str, Any]

    @property
    def row_count(self) -> int:
        lengths = {int(np.asarray(value).shape[0]) for value in self.columns.values()}
        if len(lengths) != 1:
            return -1
        return next(iter(lengths))


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
    best = 0
    for alias in aliases:
        target = normalize(alias)
        if normalized == target:
            best = max(best, 100)
        elif normalized.endswith("_" + target):
            best = max(best, 90)
        elif target in normalized:
            best = max(best, 75)
        else:
            name_tokens = set(normalized.split("_"))
            target_tokens = set(target.split("_"))
            overlap = len(name_tokens & target_tokens)
            if overlap:
                best = max(best, 20 + 10 * overlap)
    return best


def top_fields(
    table: Table,
    aliases: tuple[str, ...],
    *,
    minimum: int = 30,
    limit: int = 4,
) -> list[str]:
    scored = [
        (semantic_score(name, aliases), name)
        for name in table.columns
    ]
    scored = [
        (score, name)
        for score, name in scored
        if score >= minimum
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in scored[:limit]]


def scalar_text(value: Any) -> str | None:
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
        if value.is_integer():
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


def parse_integer_vector(values: np.ndarray) -> np.ndarray | None:
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


def clean_column(values: Any) -> np.ndarray | None:
    try:
        array = np.asarray(values)
    except Exception:
        return None
    if array.ndim == 0 or array.ndim > 2:
        return None
    if array.shape[0] <= 0 or array.shape[0] > MAX_TABLE_ROWS:
        return None
    if array.size > MAX_ARRAY_ELEMENTS:
        return None
    if array.ndim == 2 and array.shape[1] != 1:
        return None
    return array.reshape(-1)


def table_summary(table: Table) -> dict[str, Any]:
    return {
        "table_id": table.table_id,
        "origin": table.origin,
        "trust": table.trust,
        "path": table.path,
        "row_count": table.row_count,
        "column_count": len(table.columns),
        "columns": {
            name: {
                "dtype": str(np.asarray(values).dtype),
                "shape": list(np.asarray(values).shape),
                "pair_score": semantic_score(name, PAIR_ALIASES),
                "window_score": semantic_score(name, WINDOW_ALIASES),
                "run_score": semantic_score(name, RUN_ALIASES),
                "member_score": semantic_score(name, MEMBER_ALIASES),
                "attack_score": semantic_score(name, ATTACK_ALIASES),
                "count_score": semantic_score(name, COUNT_ALIASES),
                "index_score": semantic_score(name, INDEX_ALIASES),
                "split_score": semantic_score(name, SPLIT_ALIASES),
                "attack_run_score": semantic_score(name, ATTACK_RUN_ALIASES),
                "control_run_score": semantic_score(name, CONTROL_RUN_ALIASES),
            }
            for name, values in table.columns.items()
        },
        "details": table.details,
    }


def add_table(
    tables: list[Table],
    *,
    table_id: str,
    origin: str,
    trust: str,
    path: str | None,
    columns: dict[str, Any],
    details: dict[str, Any] | None = None,
) -> None:
    cleaned = {}
    for name, values in columns.items():
        column = clean_column(values)
        if column is not None:
            cleaned[str(name)] = column
    if not cleaned:
        return
    lengths = {int(value.shape[0]) for value in cleaned.values()}
    if len(lengths) != 1:
        return
    if len(cleaned) > MAX_TABLE_COLUMNS:
        cleaned = dict(list(cleaned.items())[:MAX_TABLE_COLUMNS])
    tables.append(
        Table(
            table_id=table_id,
            origin=origin,
            trust=trust,
            path=path,
            columns=cleaned,
            details=details or {},
        )
    )


def construct_validation_dataset(
    loader_path: Path,
    data_root: Path,
) -> tuple[Any, dict[str, Any], Any]:
    module = import_source(loader_path, "_v5_p3_f6_p3_guarded_loader")
    require(
        hasattr(module, "_load_original_class")
        and hasattr(module, "_construct_original"),
        "guarded loader route changed",
    )
    dataset_class = module._load_original_class(data_root)
    attempts = []
    for split in ("validation", "val", "VALIDATION"):
        try:
            dataset = module._construct_original(
                dataset_class,
                data_root,
                split,
                {},
            )
            attempts.append({
                "split": split,
                "length": int(len(dataset)),
                "status": "SUCCESS",
            })
            if len(dataset) == EXPECTED_ITEMS:
                return dataset, {
                    "route": "_load_original_class/_construct_original",
                    "split": split,
                    "dataset_class": (
                        f"{dataset.__class__.__module__}."
                        f"{dataset.__class__.__name__}"
                    ),
                    "attempts": attempts,
                }, dataset_class
        except BaseException as exc:
            attempts.append({
                "split": split,
                "status": "FAILED",
                "error": repr(exc),
            })
    raise RuntimeError(
        "could not construct 13,863-item validation dataset; "
        f"attempts={attempts}"
    )


def source_review(dataset_class: Any) -> dict[str, Any]:
    source_path = inspect.getsourcefile(dataset_class)
    source_file = Path(source_path).resolve() if source_path else None
    class_source = ""
    try:
        class_source = inspect.getsource(dataset_class)
    except Exception:
        pass

    methods = {}
    for method_name in ("__init__", "__len__", "__getitem__"):
        method = getattr(dataset_class, method_name, None)
        if method is None:
            continue
        try:
            methods[method_name] = {
                "signature": str(inspect.signature(method)),
                "source": inspect.getsource(method),
            }
        except Exception as exc:
            methods[method_name] = {
                "signature": None,
                "source_error": repr(exc),
            }

    keywords = (
        "pair",
        "window",
        "run_id",
        "sample_index",
        "split",
        "control",
        "attack",
        "manifest",
        "metadata",
        "np.load",
        "memmap",
        "read_csv",
        "read_json",
    )
    keyword_lines = []
    for index, line in enumerate(class_source.splitlines(), start=1):
        low = line.lower()
        if any(keyword in low for keyword in keywords):
            keyword_lines.append({
                "line": index,
                "text": line[:1200],
            })

    path_literals = sorted(
        set(
            re.findall(
                r"""["']([^"']+\.(?:csv|tsv|json|jsonl|npy|npz|parquet))["']""",
                class_source,
                flags=re.IGNORECASE,
            )
        )
    )

    return {
        "dataset_class": (
            f"{dataset_class.__module__}.{dataset_class.__name__}"
        ),
        "source_file": str(source_file) if source_file else None,
        "source_file_sha256": (
            sha256_file(source_file)
            if source_file and source_file.is_file()
            else None
        ),
        "class_source": class_source,
        "methods": methods,
        "keyword_lines": keyword_lines,
        "referenced_path_literals": path_literals,
    }


def dataset_object_tables(dataset: Any) -> tuple[list[Table], dict[str, Any]]:
    tables: list[Table] = []
    inventory = []
    seen = set()

    def visit(value: Any, path: str, depth: int) -> None:
        if depth > 5:
            return
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)

        if isinstance(value, np.ndarray):
            inventory.append({
                "path": path,
                "type": "numpy.ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            })
            if value.ndim <= 2 and 1 <= value.shape[0] <= MAX_TABLE_ROWS:
                add_table(
                    tables,
                    table_id=f"dataset:{path}",
                    origin="dataset_object_array",
                    trust="guarded_dataset_object_order",
                    path=None,
                    columns={path.split(".")[-1]: value},
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
                if (
                    value.device.type == "cpu"
                    and value.ndim <= 2
                    and 1 <= value.shape[0] <= MAX_TABLE_ROWS
                ):
                    add_table(
                        tables,
                        table_id=f"dataset:{path}",
                        origin="dataset_object_tensor",
                        trust="guarded_dataset_object_order",
                        path=None,
                        columns={
                            path.split(".")[-1]: value.detach().numpy()
                        },
                        details={"object_path": path},
                    )
                return
        except Exception:
            pass

        class_name = (
            f"{value.__class__.__module__}."
            f"{value.__class__.__name__}"
        )

        if class_name.endswith(".DataFrame"):
            columns = {
                str(column): value[column].to_numpy()
                for column in value.columns
            }
            inventory.append({
                "path": path,
                "type": class_name,
                "shape": list(value.shape),
                "columns": list(columns),
            })
            add_table(
                tables,
                table_id=f"dataset:{path}",
                origin="dataset_object_dataframe",
                trust="guarded_dataset_object_order",
                path=None,
                columns=columns,
                details={"object_path": path},
            )
            return

        if class_name.endswith(".Series"):
            try:
                array = value.to_numpy()
            except Exception:
                array = None
            inventory.append({
                "path": path,
                "type": class_name,
                "shape": list(value.shape),
            })
            if array is not None:
                add_table(
                    tables,
                    table_id=f"dataset:{path}",
                    origin="dataset_object_series",
                    trust="guarded_dataset_object_order",
                    path=None,
                    columns={path.split(".")[-1]: array},
                    details={"object_path": path},
                )
            return

        if isinstance(value, dict):
            inventory.append({
                "path": path,
                "type": "dict",
                "length": len(value),
                "keys_preview": [str(key) for key in list(value)[:100]],
            })

            # Dictionary-of-columns table.
            candidate_columns = {}
            for key, child in value.items():
                column = clean_column(child)
                if column is not None:
                    candidate_columns[str(key)] = column
            if candidate_columns:
                add_table(
                    tables,
                    table_id=f"dataset:{path}",
                    origin="dataset_object_dict_columns",
                    trust="guarded_dataset_object_order",
                    path=None,
                    columns=candidate_columns,
                    details={"object_path": path},
                )

            for key, child in list(value.items())[:300]:
                visit(child, f"{path}.{key}", depth + 1)
            return

        if isinstance(value, (list, tuple)):
            inventory.append({
                "path": path,
                "type": type(value).__name__,
                "length": len(value),
            })

            if value and all(
                isinstance(item, dict)
                for item in value[: min(len(value), 100)]
            ):
                fields = sorted(
                    set().union(
                        *[
                            set(item)
                            for item in value[: min(len(value), 1000)]
                        ]
                    )
                )
                columns = {
                    str(field): np.asarray(
                        [item.get(field) for item in value],
                        dtype=object,
                    )
                    for field in fields
                }
                add_table(
                    tables,
                    table_id=f"dataset:{path}",
                    origin="dataset_object_record_list",
                    trust="guarded_dataset_object_order",
                    path=None,
                    columns=columns,
                    details={"object_path": path},
                )
            else:
                column = clean_column(value)
                if column is not None:
                    add_table(
                        tables,
                        table_id=f"dataset:{path}",
                        origin="dataset_object_sequence",
                        trust="guarded_dataset_object_order",
                        path=None,
                        columns={path.split(".")[-1]: column},
                        details={"object_path": path},
                    )

            if len(value) <= 100:
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]", depth + 1)
            return

        if isinstance(value, (str, bytes, int, float, bool, type(None), Path)):
            inventory.append({
                "path": path,
                "type": type(value).__name__,
                "value": str(value)[:1000],
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

            # Merge sibling one-dimensional attributes into one table when
            # their lengths agree. This is important when pair_id, run_id and
            # window_start are stored as separate dataset fields.
            sibling_columns = {}
            for key, child in attributes.items():
                column = clean_column(child)
                if column is not None:
                    sibling_columns[key] = column
            by_length: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
            for key, column in sibling_columns.items():
                by_length[int(column.shape[0])][key] = column
            for length, columns in by_length.items():
                add_table(
                    tables,
                    table_id=f"dataset:{path}:siblings:{length}",
                    origin="dataset_object_sibling_attributes",
                    trust="guarded_dataset_object_order",
                    path=None,
                    columns=columns,
                    details={"object_path": path, "length": length},
                )

            for key, child in attributes.items():
                visit(child, f"{path}.{key}", depth + 1)

    visit(dataset, "dataset", 0)
    return tables, {
        "dataset_type": (
            f"{dataset.__class__.__module__}."
            f"{dataset.__class__.__name__}"
        ),
        "entries": inventory,
    }


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


def table_from_records(
    *,
    table_id: str,
    origin: str,
    trust: str,
    path: Path,
    records: list[dict[str, Any]],
    details: dict[str, Any],
) -> Table | None:
    if not records or len(records) > MAX_TABLE_ROWS:
        return None
    fields = sorted(set().union(*[set(record) for record in records]))
    if len(fields) > MAX_TABLE_COLUMNS:
        return None
    columns = {
        str(field): np.asarray(
            [record.get(field) for record in records],
            dtype=object,
        )
        for field in fields
    }
    tables = []
    add_table(
        tables,
        table_id=table_id,
        origin=origin,
        trust=trust,
        path=str(path.resolve()),
        columns=columns,
        details=details,
    )
    return tables[0] if tables else None


def load_text_tables(path: Path) -> list[Table]:
    suffix = path.suffix.lower()
    tables = []

    try:
        if suffix in (".csv", ".tsv"):
            delimiter = sniff_delimiter(path)
            with path.open(
                "r",
                encoding="utf-8",
                errors="replace",
                newline="",
            ) as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                if not reader.fieldnames:
                    return []
                records = [dict(row) for row in reader]
            table = table_from_records(
                table_id=f"file:{path}",
                origin="text_table",
                trust=(
                    "validation_named_file"
                    if "val" in normalize(path.name)
                    else "file_order_unproven"
                ),
                path=path,
                records=records,
                details={"delimiter": delimiter},
            )
            if table:
                tables.append(table)

        elif suffix in (".jsonl", ".ndjson"):
            records = []
            with path.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if isinstance(value, dict):
                        records.append(value)
            table = table_from_records(
                table_id=f"file:{path}",
                origin="jsonl_table",
                trust=(
                    "validation_named_file"
                    if "val" in normalize(path.name)
                    else "file_order_unproven"
                ),
                path=path,
                records=records,
                details={},
            )
            if table:
                tables.append(table)

        elif suffix == ".json":
            value = json.loads(
                path.read_text(encoding="utf-8", errors="replace")
            )
            record_sets = []
            if isinstance(value, list) and value and all(
                isinstance(item, dict) for item in value
            ):
                record_sets.append(("root", value))
            elif isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(child, list) and child and all(
                        isinstance(item, dict) for item in child
                    ):
                        record_sets.append((str(key), child))
                    elif isinstance(child, dict):
                        columns = {
                            str(name): column
                            for name, column in child.items()
                            if clean_column(column) is not None
                        }
                        if columns:
                            candidate_tables = []
                            add_table(
                                candidate_tables,
                                table_id=f"file:{path}:{key}",
                                origin="json_dict_columns",
                                trust=(
                                    "validation_named_file"
                                    if "val" in normalize(path.name)
                                    else "file_order_unproven"
                                ),
                                path=str(path.resolve()),
                                columns=columns,
                                details={"json_key": str(key)},
                            )
                            tables.extend(candidate_tables)

            for key, records in record_sets:
                table = table_from_records(
                    table_id=f"file:{path}:{key}",
                    origin="json_record_table",
                    trust=(
                        "validation_named_file"
                        if "val" in normalize(path.name)
                        else "file_order_unproven"
                    ),
                    path=path,
                    records=records,
                    details={"json_key": key},
                )
                if table:
                    tables.append(table)

    except Exception:
        return []

    return tables


def load_array_tables(path: Path) -> list[Table]:
    tables = []
    suffix = path.suffix.lower()
    try:
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                by_length: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
                for key in archive.files:
                    column = clean_column(np.asarray(archive[key]))
                    if column is not None:
                        by_length[int(column.shape[0])][key] = column
                for length, columns in by_length.items():
                    add_table(
                        tables,
                        table_id=f"file:{path}:length:{length}",
                        origin="npz_columns",
                        trust=(
                            "validation_named_file"
                            if "val" in normalize(path.name)
                            else "file_order_unproven"
                        ),
                        path=str(path.resolve()),
                        columns=columns,
                        details={"length": length},
                    )

        elif suffix == ".npy":
            array = np.load(path, allow_pickle=False, mmap_mode="r")
            column = clean_column(array)
            if column is not None:
                add_table(
                    tables,
                    table_id=f"file:{path}",
                    origin="npy_array",
                    trust=(
                        "validation_named_file"
                        if "val" in normalize(path.name)
                        else "file_order_unproven"
                    ),
                    path=str(path.resolve()),
                    columns={path.stem: column},
                    details={},
                )
    except Exception:
        return []
    return tables


def discover_file_tables(roots: list[Path]) -> tuple[list[Table], list[dict[str, Any]]]:
    tables = []
    inventory = []
    seen = set()
    suffixes = {
        ".csv",
        ".tsv",
        ".json",
        ".jsonl",
        ".ndjson",
        ".npy",
        ".npz",
    }

    excluded_fragments = (
        "/permutation/f5_runs/",
        "/integrated_gradients/",
        "/models/",
        "/.git/",
    )

    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            resolved = path.resolve()
            resolved_text = str(resolved)
            if any(fragment in resolved_text for fragment in excluded_fragments):
                continue
            if resolved in seen:
                continue
            seen.add(resolved)

            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size <= 0 or size > 512 * 1024 * 1024:
                continue

            if path.suffix.lower() in (".npy", ".npz"):
                loaded = load_array_tables(path)
            else:
                loaded = load_text_tables(path)

            inventory.append({
                "path": resolved_text,
                "sha256": sha256_file(path),
                "size_bytes": int(size),
                "loaded_table_count": len(loaded),
                "loaded_tables": [
                    {
                        "table_id": table.table_id,
                        "row_count": table.row_count,
                        "columns": list(table.columns),
                    }
                    for table in loaded
                ],
            })
            tables.extend(loaded)

    # Merge sibling one-column NPY tables from the same directory and length.
    groups: dict[tuple[str, int], dict[str, np.ndarray]] = defaultdict(dict)
    for table in tables:
        if table.origin != "npy_array" or table.row_count <= 0:
            continue
        parent = str(Path(table.path).parent) if table.path else table.table_id
        for name, values in table.columns.items():
            groups[(parent, table.row_count)][name] = values

    for (parent, length), columns in groups.items():
        if len(columns) < 2:
            continue
        add_table(
            tables,
            table_id=f"npy_siblings:{parent}:{length}",
            origin="npy_sibling_columns",
            trust=(
                "validation_named_file"
                if "val" in normalize(parent)
                else "file_order_unproven"
            ),
            path=parent,
            columns=columns,
            details={"length": length},
        )

    return tables, inventory


def filter_validation(table: Table) -> Table | None:
    split_fields = top_fields(table, SPLIT_ALIASES, minimum=70, limit=2)
    if not split_fields:
        return table
    for split_field in split_fields:
        values = np.asarray(table.columns[split_field]).reshape(-1)
        mask = np.asarray(
            [normalize(str(value)) in VALIDATION_VALUES for value in values],
            dtype=bool,
        )
        if int(np.sum(mask)) <= 0:
            continue
        columns = {
            name: np.asarray(column).reshape(-1)[mask]
            for name, column in table.columns.items()
        }
        return Table(
            table_id=table.table_id + f":filtered:{split_field}",
            origin=table.origin,
            trust="explicit_validation_split_filter",
            path=table.path,
            columns=columns,
            details={
                **table.details,
                "split_field": split_field,
                "input_rows": table.row_count,
                "output_rows": int(np.sum(mask)),
            },
        )
    return None


def align_sample_table(
    table: Table,
    fresh_attack: np.ndarray,
) -> tuple[Table | None, dict[str, Any]]:
    filtered = filter_validation(table)
    if filtered is None:
        return None, {"failure": "validation_filter_empty"}
    table = filtered

    if table.row_count != EXPECTED_ITEMS:
        return None, {
            "failure": "not_validation_item_length",
            "row_count": table.row_count,
        }

    index_fields = top_fields(table, INDEX_ALIASES, minimum=70, limit=3)
    for index_field in index_fields:
        indices = parse_integer_vector(table.columns[index_field])
        if indices is None:
            continue
        if set(indices.tolist()) != set(range(EXPECTED_ITEMS)):
            continue
        reordered = {}
        for name, values in table.columns.items():
            output = np.empty(EXPECTED_ITEMS, dtype=np.asarray(values).dtype)
            output[indices] = np.asarray(values).reshape(-1)
            reordered[name] = output
        aligned = Table(
            table_id=table.table_id + f":aligned:{index_field}",
            origin=table.origin,
            trust="explicit_validation_index",
            path=table.path,
            columns=reordered,
            details={**table.details, "index_field": index_field},
        )
        return aligned, {
            "mode": "explicit_validation_index",
            "index_field": index_field,
        }

    attack_fields = top_fields(table, ATTACK_ALIASES, minimum=70, limit=4)
    for attack_field in attack_fields:
        parsed = np.asarray(
            [
                1 if classify_member(value) == "attack"
                else 0 if classify_member(value) == "control"
                else -1
                for value in table.columns[attack_field]
            ],
            dtype=np.int64,
        )
        if np.any(parsed < 0):
            continue
        if np.array_equal(parsed, fresh_attack):
            return table, {
                "mode": "attack_label_alignment",
                "attack_field": attack_field,
            }

    if table.trust in (
        "guarded_dataset_object_order",
        "explicit_validation_split_filter",
        "validation_named_file",
    ):
        return table, {
            "mode": "trusted_validation_order",
            "trust": table.trust,
        }

    return None, {"failure": "validation_row_order_unproven"}


def pair_mapping_from_identity(
    pair_values: np.ndarray,
    window_values: np.ndarray,
    fresh_attack: np.ndarray,
    fresh_count: np.ndarray,
    route_meta: dict[str, Any],
) -> dict[str, Any]:
    identities = []
    invalid = 0
    for pair, window in zip(pair_values, window_values):
        pair_text = scalar_text(pair)
        window_text = scalar_text(window)
        if pair_text is None:
            invalid += 1
            identities.append(None)
        else:
            identities.append((pair_text, window_text or ""))

    if invalid:
        return {
            **route_meta,
            "usable": False,
            "failure": "invalid_pair_identity_values",
            "invalid_identity_count": invalid,
        }

    grouped: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: {"attack": [], "control": []}
    )
    for index, identity in enumerate(identities):
        role = "attack" if int(fresh_attack[index]) == 1 else "control"
        grouped[identity][role].append(index)

    attack_groups = {
        identity: bucket
        for identity, bucket in grouped.items()
        if bucket["attack"]
    }

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
        })

    mapping.sort(key=lambda row: row["attack_index"])
    expected_attacks = np.flatnonzero(fresh_attack == 1).tolist()
    mapped_attacks = [row["attack_index"] for row in mapping]
    controls = [row["control_index"] for row in mapping]
    strata = Counter(row["attacker_count"] for row in mapping)

    usable = bool(
        len(mapping) == EXPECTED_ACTIVE
        and mapped_attacks == expected_attacks
        and len(invalid_groups) == 0
        and len(set(controls)) == EXPECTED_ACTIVE
        and all(strata.get(value, 0) >= 128 for value in (1, 2, 3, 4))
    )

    result = {
        **route_meta,
        "usable": usable,
        "total_identity_groups": len(grouped),
        "attack_identity_groups": len(attack_groups),
        "control_only_identity_groups": sum(
            1
            for bucket in grouped.values()
            if not bucket["attack"] and bucket["control"]
        ),
        "invalid_attack_group_count": len(invalid_groups),
        "invalid_attack_group_preview": dict(
            list(invalid_groups.items())[:20]
        ),
        "mapped_attack_item_count": len(mapping),
        "all_active_items_mapped": mapped_attacks == expected_attacks,
        "unique_control_item_count": len(set(controls)),
        "control_reuse_count": len(controls) - len(set(controls)),
        "attacker_count_strata": {
            str(key): int(value)
            for key, value in sorted(strata.items())
        },
    }
    if usable:
        result["mapping"] = mapping
    else:
        result["failure"] = "mapping_acceptance_gate_failed"
    return result


def evaluate_direct_routes(
    sample_tables: list[tuple[Table, dict[str, Any]]],
    fresh_attack: np.ndarray,
    fresh_count: np.ndarray,
) -> list[dict[str, Any]]:
    routes = []
    for table, alignment in sample_tables:
        pair_fields = top_fields(table, PAIR_ALIASES, minimum=50, limit=5)
        window_fields = top_fields(table, WINDOW_ALIASES, minimum=50, limit=5)

        for pair_field in pair_fields:
            options = [None] + window_fields
            for window_field in options:
                pair_values = np.asarray(table.columns[pair_field]).reshape(-1)
                window_values = (
                    np.asarray(table.columns[window_field]).reshape(-1)
                    if window_field is not None
                    else np.asarray([""] * EXPECTED_ITEMS, dtype=object)
                )
                routes.append(
                    pair_mapping_from_identity(
                        pair_values,
                        window_values,
                        fresh_attack,
                        fresh_count,
                        {
                            "route_type": "direct_sample_pair_identity",
                            "sample_table": table_summary(table),
                            "alignment": alignment,
                            "pair_field": pair_field,
                            "window_field": window_field,
                        },
                    )
                )
    return routes


def run_metadata_maps(table: Table) -> list[dict[str, Any]]:
    maps = []
    run_fields = top_fields(table, RUN_ALIASES, minimum=50, limit=5)
    pair_fields = top_fields(table, PAIR_ALIASES, minimum=50, limit=5)
    member_fields = top_fields(table, MEMBER_ALIASES, minimum=50, limit=5)
    attack_run_fields = top_fields(
        table,
        ATTACK_RUN_ALIASES,
        minimum=50,
        limit=4,
    )
    control_run_fields = top_fields(
        table,
        CONTROL_RUN_ALIASES,
        minimum=50,
        limit=4,
    )

    # Standard run_id + pair_id + member schema.
    for run_field in run_fields:
        for pair_field in pair_fields:
            for member_field in member_fields:
                mapping = {}
                invalid = 0
                for run, pair, member in zip(
                    table.columns[run_field],
                    table.columns[pair_field],
                    table.columns[member_field],
                ):
                    run_text = scalar_text(run)
                    pair_text = scalar_text(pair)
                    role = classify_member(member)
                    if run_text is None or pair_text is None or role is None:
                        invalid += 1
                        continue
                    value = (pair_text, role)
                    if run_text in mapping and mapping[run_text] != value:
                        invalid += 1
                    mapping[run_text] = value
                if mapping and invalid == 0:
                    maps.append({
                        "schema": "run_pair_member",
                        "table": table_summary(table),
                        "run_field": run_field,
                        "pair_field": pair_field,
                        "member_field": member_field,
                        "mapping": mapping,
                    })

    # Pair table with separate attack_run_id and control_run_id columns.
    for pair_field in pair_fields:
        for attack_run_field in attack_run_fields:
            for control_run_field in control_run_fields:
                mapping = {}
                invalid = 0
                for pair, attack_run, control_run in zip(
                    table.columns[pair_field],
                    table.columns[attack_run_field],
                    table.columns[control_run_field],
                ):
                    pair_text = scalar_text(pair)
                    attack_text = scalar_text(attack_run)
                    control_text = scalar_text(control_run)
                    if (
                        pair_text is None
                        or attack_text is None
                        or control_text is None
                    ):
                        invalid += 1
                        continue
                    for run_text, role in (
                        (attack_text, "attack"),
                        (control_text, "control"),
                    ):
                        value = (pair_text, role)
                        if run_text in mapping and mapping[run_text] != value:
                            invalid += 1
                        mapping[run_text] = value
                if mapping and invalid == 0:
                    maps.append({
                        "schema": "pair_attack_run_control_run",
                        "table": table_summary(table),
                        "pair_field": pair_field,
                        "attack_run_field": attack_run_field,
                        "control_run_field": control_run_field,
                        "mapping": mapping,
                    })

    return maps


def evaluate_run_join_routes(
    sample_tables: list[tuple[Table, dict[str, Any]]],
    all_tables: list[Table],
    fresh_attack: np.ndarray,
    fresh_count: np.ndarray,
) -> list[dict[str, Any]]:
    run_maps = []
    for table in all_tables:
        if 2 <= table.row_count <= 20000:
            run_maps.extend(run_metadata_maps(table))

    routes = []
    for sample_table, alignment in sample_tables:
        sample_run_fields = top_fields(
            sample_table,
            RUN_ALIASES,
            minimum=50,
            limit=5,
        )
        sample_window_fields = top_fields(
            sample_table,
            WINDOW_ALIASES,
            minimum=50,
            limit=5,
        )
        if not sample_run_fields:
            continue

        for sample_run_field in sample_run_fields:
            run_values = np.asarray(
                sample_table.columns[sample_run_field]
            ).reshape(-1)
            for run_map in run_maps:
                metadata = run_map["mapping"]
                pair_values = []
                role_values = []
                missing = 0
                for value in run_values:
                    run_text = scalar_text(value)
                    mapped = metadata.get(run_text) if run_text is not None else None
                    if mapped is None:
                        missing += 1
                        pair_values.append(None)
                        role_values.append(None)
                    else:
                        pair_values.append(mapped[0])
                        role_values.append(mapped[1])

                if missing:
                    routes.append({
                        "route_type": "sample_run_to_run_metadata_join",
                        "usable": False,
                        "failure": "run_metadata_join_missing_values",
                        "missing_sample_count": missing,
                        "sample_table": table_summary(sample_table),
                        "sample_run_field": sample_run_field,
                        "run_metadata": {
                            key: value
                            for key, value in run_map.items()
                            if key != "mapping"
                        },
                    })
                    continue

                derived_attack = np.asarray(
                    [1 if role == "attack" else 0 for role in role_values],
                    dtype=np.int64,
                )
                if not np.array_equal(derived_attack, fresh_attack):
                    routes.append({
                        "route_type": "sample_run_to_run_metadata_join",
                        "usable": False,
                        "failure": "joined_run_role_does_not_match_fresh_labels",
                        "role_mismatch_count": int(
                            np.sum(derived_attack != fresh_attack)
                        ),
                        "sample_table": table_summary(sample_table),
                        "sample_run_field": sample_run_field,
                        "run_metadata": {
                            key: value
                            for key, value in run_map.items()
                            if key != "mapping"
                        },
                    })
                    continue

                window_options = [None] + sample_window_fields
                for window_field in window_options:
                    window_values = (
                        np.asarray(
                            sample_table.columns[window_field]
                        ).reshape(-1)
                        if window_field is not None
                        else np.asarray(
                            [""] * EXPECTED_ITEMS,
                            dtype=object,
                        )
                    )
                    routes.append(
                        pair_mapping_from_identity(
                            np.asarray(pair_values, dtype=object),
                            window_values,
                            fresh_attack,
                            fresh_count,
                            {
                                "route_type": (
                                    "sample_run_to_run_metadata_join"
                                ),
                                "sample_table": table_summary(sample_table),
                                "alignment": alignment,
                                "sample_run_field": sample_run_field,
                                "sample_window_field": window_field,
                                "run_metadata": {
                                    key: value
                                    for key, value in run_map.items()
                                    if key != "mapping"
                                },
                            },
                        )
                    )

    return routes


def mapping_fingerprint(mapping: list[dict[str, Any]]) -> str:
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

    p2_report_path = ig_dir / (
        "V5_P3_F6_P2_TARGETED_MANIFEST_ROUTE_RECOVERY_REPORT.json"
    )
    p2_lock_path = ig_dir / (
        "V5_P3_F6_P2_TARGETED_MANIFEST_ROUTE_RECOVERY_LOCK.json"
    )
    p0_protocol_path = ig_dir / (
        "F6_P0_TASK_SPECIFIC_INTEGRATED_GRADIENTS_PROTOCOL.json"
    )
    route_path = permutation_dir / (
        "F5_P0_EXECUTION_ROUTE_SOURCE_INVENTORY.json"
    )
    fresh_cert_path = metric_dir / (
        "F4M_R3A_FRESH_REPLAY_14_METRIC_CERTIFICATION.json"
    )

    p2_report, p2_lock = verify_report_lock(
        p2_report_path,
        p2_lock_path,
    )
    require(
        p2_lock.get("mapping_resolved") is False,
        "F6-P3 expected unresolved P2 mapping",
    )
    require(
        p2_lock.get("actual_F6_execution_authorized") is False,
        "F6 unexpectedly authorized before P3",
    )
    require(p2_lock.get("F7_authorized") is False, "F7 must remain held")

    p0_protocol = load_json(p0_protocol_path)
    route_inventory = load_json(route_path)
    fresh_cert = load_json(fresh_cert_path)

    require(
        p0_protocol["baseline"]["type"]
        == "matched_control_feature_tensor",
        "F6 matched-control protocol changed",
    )
    require(
        p0_protocol["sample_selection"]["total_items"] == 512,
        "F6 sample-count protocol changed",
    )
    require(
        p0_protocol["integrated_gradients"]["steps"] == 64,
        "F6 integration-step protocol changed",
    )

    fresh_npz = Path(fresh_cert["npz"]).resolve()
    require(fresh_npz.is_file(), f"fresh NPZ missing: {fresh_npz}")
    require(
        sha256_file(fresh_npz) == fresh_cert["npz_sha256"],
        "fresh NPZ hash changed",
    )
    with np.load(fresh_npz, allow_pickle=False) as archive:
        fresh_attack = np.asarray(archive["y_attack"]).astype(np.int64)
        fresh_count = np.asarray(
            archive["y_attacker_count"]
        ).astype(np.int64)

    require(fresh_attack.shape == (EXPECTED_ITEMS,), "attack shape changed")
    require(fresh_count.shape == (EXPECTED_ITEMS,), "count shape changed")
    require(int(np.sum(fresh_attack == 1)) == EXPECTED_ACTIVE, "active count changed")
    require(int(np.sum(fresh_attack == 0)) == EXPECTED_CONTROL, "control count changed")

    loader_row = route_inventory["certified_files"]["loader"]
    loader_path = Path(loader_row["path"]).resolve()
    require(loader_path.is_file(), f"guarded loader missing: {loader_path}")
    require(
        sha256_file(loader_path) == loader_row["actual_sha256"],
        "guarded loader hash changed",
    )

    dataset, dataset_provenance, dataset_class = construct_validation_dataset(
        loader_path,
        data_root,
    )
    source = source_review(dataset_class)

    object_tables, object_inventory = dataset_object_tables(dataset)

    file_roots = [
        data_root,
        repo / "reports/v5/p3_d1_immutable_validation_logit_export",
        feature_root / "paired_analysis",
        repo / "data/processed/v5",
    ]
    file_tables, file_inventory = discover_file_tables(file_roots)

    all_tables = object_tables + file_tables

    sample_tables = []
    alignment_reviews = []
    for table in all_tables:
        aligned, review = align_sample_table(table, fresh_attack)
        alignment_reviews.append({
            "table_id": table.table_id,
            "row_count": table.row_count,
            "review": review,
        })
        if aligned is not None:
            sample_tables.append((aligned, review))

    direct_routes = evaluate_direct_routes(
        sample_tables,
        fresh_attack,
        fresh_count,
    )
    join_routes = evaluate_run_join_routes(
        sample_tables,
        all_tables,
        fresh_attack,
        fresh_count,
    )
    routes = direct_routes + join_routes
    usable = [route for route in routes if route.get("usable")]

    fingerprint_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route_result in usable:
        fingerprint = mapping_fingerprint(route_result["mapping"])
        route_result["mapping_fingerprint_sha256"] = fingerprint
        fingerprint_groups[fingerprint].append(route_result)

    mapping_resolved = len(fingerprint_groups) == 1
    selected_route = None
    selected_mapping = None
    selected_fingerprint = None

    if mapping_resolved:
        selected_fingerprint, equivalent = next(
            iter(fingerprint_groups.items())
        )
        equivalent.sort(
            key=lambda row: (
                0 if row["route_type"]
                == "sample_run_to_run_metadata_join" else 1,
                0 if row.get("sample_window_field")
                or row.get("window_field") else 1,
                json.dumps(row, sort_keys=True, default=json_default)[:500],
            )
        )
        selected_route = equivalent[0]
        selected_mapping = selected_route["mapping"]

    failure_distribution = Counter(
        route.get("failure", "usable")
        for route in routes
    )

    source_path = output_dir / "F6_P3_DATASET_BUILDER_SOURCE_REVIEW.json"
    object_path = output_dir / "F6_P3_DATASET_OBJECT_METADATA_TABLES.json"
    file_path = output_dir / "F6_P3_FILE_METADATA_TABLES.json"
    alignment_path = output_dir / "F6_P3_SAMPLE_TABLE_ALIGNMENT_REVIEW.json"
    routes_path = output_dir / "F6_P3_DIRECT_AND_RUN_JOIN_ROUTE_EVALUATIONS.json"
    mapping_json_path = output_dir / "F6_P3_FROZEN_ATTACK_TO_CONTROL_MAPPING.json"
    mapping_csv_path = output_dir / "F6_P3_FROZEN_ATTACK_TO_CONTROL_MAPPING.csv"
    decision_path = output_dir / "F6_P3_DATASET_BUILDER_IDENTITY_DECISION.json"

    atomic_json(source_path, source)
    atomic_json(
        object_path,
        {
            "dataset_provenance": dataset_provenance,
            "dataset_getitem_called": False,
            "validation_feature_tensors_loaded": False,
            "object_inventory": object_inventory,
            "tables": [table_summary(table) for table in object_tables],
        },
    )
    atomic_json(
        file_path,
        {
            "roots": [str(root) for root in file_roots],
            "inventory": file_inventory,
            "tables": [table_summary(table) for table in file_tables],
        },
    )
    atomic_json(
        alignment_path,
        {
            "all_table_count": len(all_tables),
            "aligned_validation_sample_table_count": len(sample_tables),
            "reviews": alignment_reviews,
        },
    )

    compact_routes = []
    for route_result in routes:
        compact_routes.append({
            key: value
            for key, value in route_result.items()
            if key != "mapping"
        })
    atomic_json(
        routes_path,
        {
            "direct_route_count": len(direct_routes),
            "run_join_route_count": len(join_routes),
            "evaluated_route_count": len(routes),
            "usable_route_count": len(usable),
            "unique_mapping_fingerprint_count": len(fingerprint_groups),
            "failure_distribution": dict(failure_distribution),
            "mapping_fingerprints": {
                fingerprint: [
                    {
                        key: value
                        for key, value in route_result.items()
                        if key not in ("mapping",)
                    }
                    for route_result in route_rows
                ]
                for fingerprint, route_rows in fingerprint_groups.items()
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
            "route": {
                key: value
                for key, value in selected_route.items()
                if key != "mapping"
            },
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
            "F6_P2_complete": True,
            "dataset_builder_source_recovered": bool(source["class_source"]),
            "dataset_object_table_count": len(object_tables),
            "file_table_count": len(file_tables),
            "aligned_validation_sample_table_count": len(sample_tables),
            "direct_route_count": len(direct_routes),
            "run_join_route_count": len(join_routes),
            "usable_route_count": len(usable),
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
                else "V5_P3_F6_P4_MATCHED_CONTROL_BASELINE_POLICY_REVIEW"
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
            "Inspect the exact guarded validation dataset builder and its "
            "metadata-bearing object state; discover sample-level and "
            "run-level tables from dataset artifacts; evaluate direct "
            "pair identities and relational joins from validation run/window "
            "metadata to run-level pair/member metadata; and freeze a unique "
            "complete attack-to-control map when scientifically recoverable."
        ),
        "finding": {
            "dataset_provenance": dataset_provenance,
            "dataset_source_file": source["source_file"],
            "dataset_object_table_count": len(object_tables),
            "file_table_count": len(file_tables),
            "aligned_validation_sample_table_count": len(sample_tables),
            "direct_route_count": len(direct_routes),
            "run_join_route_count": len(join_routes),
            "usable_route_count": len(usable),
            "unique_mapping_fingerprint_count": len(fingerprint_groups),
            "failure_distribution": dict(failure_distribution),
            "mapping_resolved": mapping_resolved,
            "selected_mapping_fingerprint": selected_fingerprint,
            "mapped_attack_item_count": (
                len(selected_mapping)
                if selected_mapping is not None
                else 0
            ),
        },
        "decision": {
            "F6_P3_complete": True,
            "actual_F6_execution_authorized": actual_F6_execution_authorized,
            "F7_retraining_ablation_authorized": False,
            "feature_removal_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F6_TASK_SPECIFIC_INTEGRATED_GRADIENTS"
                if actual_F6_execution_authorized
                else "V5_P3_F6_P4_MATCHED_CONTROL_BASELINE_POLICY_REVIEW"
            ),
        },
        "governance": {
            "validation_dataset_object_constructed": True,
            "validation_dataset_getitem_called": False,
            "validation_feature_tensors_loaded": False,
            "validation_metadata_loaded": True,
            "validation_labels_loaded": True,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "model_weights_changed": False,
            "thresholds_changed": False,
            "metric_formulas_changed": False,
        },
        "artifacts": {
            "dataset_builder_source_review": str(source_path),
            "dataset_object_metadata_tables": str(object_path),
            "file_metadata_tables": str(file_path),
            "sample_table_alignment_review": str(alignment_path),
            "route_evaluations": str(routes_path),
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
            "F6_P2_report_sha256": sha256_file(p2_report_path),
            "F6_P2_lock_sha256": sha256_file(p2_lock_path),
            "F6_P0_protocol_sha256": sha256_file(p0_protocol_path),
            "execution_route_inventory_sha256": sha256_file(route_path),
            "fresh_certification_sha256": sha256_file(fresh_cert_path),
            "fresh_npz_sha256": sha256_file(fresh_npz),
            "guarded_loader_sha256": sha256_file(loader_path),
            "installed_script_sha256": sha256_file(installed_script),
            "dataset_builder_source_review_sha256": sha256_file(source_path),
            "dataset_object_metadata_tables_sha256": sha256_file(object_path),
            "file_metadata_tables_sha256": sha256_file(file_path),
            "sample_table_alignment_review_sha256": sha256_file(alignment_path),
            "route_evaluations_sha256": sha256_file(routes_path),
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
            "dataset_builder_source_review_sha256": sha256_file(source_path),
            "dataset_object_metadata_tables_sha256": sha256_file(object_path),
            "file_metadata_tables_sha256": sha256_file(file_path),
            "sample_table_alignment_review_sha256": sha256_file(alignment_path),
            "route_evaluations_sha256": sha256_file(routes_path),
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
    print("F6_P2_complete=true")
    print("validation_dataset_object_constructed=true")
    print("validation_dataset_getitem_called=false")
    print("validation_feature_tensors_loaded=false")
    print(f"dataset_class={dataset_provenance['dataset_class']}")
    print(f"dataset_source_file={source['source_file']}")
    print(f"dataset_object_table_count={len(object_tables)}")
    print(f"file_table_count={len(file_tables)}")
    print(
        "aligned_validation_sample_table_count="
        f"{len(sample_tables)}"
    )
    print(f"direct_route_count={len(direct_routes)}")
    print(f"run_join_route_count={len(join_routes)}")
    print(f"evaluated_route_count={len(routes)}")
    print(f"usable_route_count={len(usable)}")
    print(
        "unique_mapping_fingerprint_count="
        f"{len(fingerprint_groups)}"
    )
    print(f"failure_distribution={dict(failure_distribution)}")
    print(f"mapping_resolved={str(mapping_resolved).lower()}")

    if selected_route is not None and selected_mapping is not None:
        print(f"selected_route_type={selected_route['route_type']}")
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
        print("selected_route_type=None")
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
    print(f"dataset_builder_source_review={source_path}")
    print(f"dataset_object_metadata_tables={object_path}")
    print(f"file_metadata_tables={file_path}")
    print(f"sample_table_alignment_review={alignment_path}")
    print(f"route_evaluations={routes_path}")
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
