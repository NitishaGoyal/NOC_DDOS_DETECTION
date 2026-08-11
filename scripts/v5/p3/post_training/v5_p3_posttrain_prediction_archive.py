#!/usr/bin/env python3
"""Validation-only prediction archive writer for V5-P3 post-training analysis.

This module intentionally does not import any dataset or model code. It is a
storage/validation primitive that future evaluation adapters call after they
have produced raw predictions on an explicitly authorized validation split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROUTERS = 16
ALLOWED_SPLIT = "validation"
FORBIDDEN_TOKENS = ("test", "sealed")

VECTOR16_FIELDS = (
    "source_truth", "source_probability",
    "transit_truth", "transit_probability",
    "victim_truth", "victim_probability",
    "path_truth", "path_probability",
)

OPTIONAL_VECTOR16_FIELDS = (
    "source_thresholded", "transit_thresholded",
    "victim_thresholded", "path_thresholded",
)

REQUIRED_SCALARS = (
    "sample_key", "pair_key", "window_start_epoch", "window_end_epoch",
    "graph_truth", "graph_probability", "count_truth",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite_number(x: Any, name: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be numeric")
    y = float(x)
    if not math.isfinite(y):
        raise ValueError(f"{name} must be finite")
    return y


def _probability(x: Any, name: str) -> float:
    y = _finite_number(x, name)
    if y < 0.0 or y > 1.0:
        raise ValueError(f"{name} must be within [0,1]")
    return y


def _binary(x: Any, name: str) -> int:
    if x in (0, 1, False, True):
        return int(x)
    raise ValueError(f"{name} must be binary")


def _vector16(value: Any, name: str, probability: bool) -> List[float | int]:
    if not isinstance(value, list) or len(value) != ROUTERS:
        raise ValueError(f"{name} must be a list of length {ROUTERS}")
    if probability:
        return [_probability(v, f"{name}[{i}]") for i, v in enumerate(value)]
    return [_binary(v, f"{name}[{i}]") for i, v in enumerate(value)]


def assert_validation_split(split: str) -> None:
    s = str(split).strip().lower()
    if any(tok in s for tok in FORBIDDEN_TOKENS):
        raise ValueError("sealed/test split is forbidden for this archive implementation")
    if s != ALLOWED_SPLIT:
        raise ValueError(f"only split='{ALLOWED_SPLIT}' is authorized; got {split!r}")


def validate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    split = record.get("split", ALLOWED_SPLIT)
    assert_validation_split(split)

    missing = [k for k in REQUIRED_SCALARS if k not in record]
    missing += [k for k in VECTOR16_FIELDS if k not in record]
    if missing:
        raise ValueError(f"missing required fields: {sorted(set(missing))}")

    out = dict(record)
    out["split"] = ALLOWED_SPLIT
    out["sample_key"] = str(record["sample_key"])
    out["pair_key"] = str(record["pair_key"])
    out["window_start_epoch"] = int(record["window_start_epoch"])
    out["window_end_epoch"] = int(record["window_end_epoch"])
    if out["window_end_epoch"] < out["window_start_epoch"]:
        raise ValueError("window_end_epoch must be >= window_start_epoch")
    out["graph_truth"] = _binary(record["graph_truth"], "graph_truth")
    out["graph_probability"] = _probability(record["graph_probability"], "graph_probability")
    out["count_truth"] = int(record["count_truth"])
    if out["count_truth"] < 0 or out["count_truth"] > 4:
        raise ValueError("count_truth must be in [0,4]")

    for name in VECTOR16_FIELDS:
        out[name] = _vector16(record[name], name, probability=name.endswith("probability"))
    for name in OPTIONAL_VECTOR16_FIELDS:
        if name in record and record[name] is not None:
            out[name] = _vector16(record[name], name, probability=False)

    if "count_probability" in record and record["count_probability"] is not None:
        cp = record["count_probability"]
        if not isinstance(cp, list) or len(cp) not in (4, 5):
            raise ValueError("count_probability must have length 4 or 5")
        out["count_probability"] = [_probability(v, f"count_probability[{i}]") for i, v in enumerate(cp)]

    for name in ("attack_onset_epoch", "attack_termination_epoch"):
        if name in record and record[name] is not None:
            out[name] = int(record[name])
    if out.get("attack_onset_epoch") is not None and out.get("attack_termination_epoch") is not None:
        if out["attack_termination_epoch"] < out["attack_onset_epoch"]:
            raise ValueError("attack termination precedes onset")

    for name in ("checkpoint_sha256", "model_sha256", "dataset_protocol_sha256"):
        if name in record and record[name] is not None:
            out[name] = str(record[name]).lower()

    return out


@dataclass
class ArchiveManifest:
    schema_version: str
    split: str
    record_count: int
    archive_sha256: str
    validation_only: bool
    threshold_tuning_performed: bool
    test_directory_enumerated: bool
    test_tensor_contents_accessed: bool
    test_evaluation_performed: bool
    sealed_test_access: bool


def write_archive(records: Iterable[Dict[str, Any]], output_jsonl: Path) -> ArchiveManifest:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            clean = validate_record(rec)
            f.write(json.dumps(clean, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    manifest = ArchiveManifest(
        schema_version="V5_P3_POSTTRAIN_P1_JSONL_V1",
        split=ALLOWED_SPLIT,
        record_count=count,
        archive_sha256=sha256_file(output_jsonl),
        validation_only=True,
        threshold_tuning_performed=False,
        test_directory_enumerated=False,
        test_tensor_contents_accessed=False,
        test_evaluation_performed=False,
        sealed_test_access=False,
    )
    manifest_path = output_jsonl.with_suffix(output_jsonl.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                validate_record(json.loads(line))
            except Exception as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
            count += 1
    return count


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate-jsonl")
    v.add_argument("path", type=Path)
    e = sub.add_parser("example")
    e.add_argument("output", type=Path)
    args = p.parse_args()

    if args.cmd == "validate-jsonl":
        count = validate_jsonl(args.path)
        print(f"VALIDATION_ARCHIVE_VALID records={count}")
        return

    example = {
        "split": "validation",
        "sample_key": "EXAMPLE_SAMPLE",
        "pair_key": "EXAMPLE_PAIR",
        "window_start_epoch": 0,
        "window_end_epoch": 31,
        "graph_truth": 1,
        "graph_probability": 0.9,
        "count_truth": 1,
        "count_probability": [0.01, 0.97, 0.01, 0.01],
        "source_truth": [1] + [0] * 15,
        "source_probability": [0.9] + [0.1] * 15,
        "transit_truth": [0] * 16,
        "transit_probability": [0.1] * 16,
        "victim_truth": [0] * 15 + [1],
        "victim_probability": [0.1] * 15 + [0.9],
        "path_truth": [1] + [0] * 14 + [1],
        "path_probability": [0.8] + [0.1] * 14 + [0.8],
        "attack_onset_epoch": 8,
        "attack_termination_epoch": 100,
    }
    m = write_archive([example], args.output)
    print(json.dumps(asdict(m), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
