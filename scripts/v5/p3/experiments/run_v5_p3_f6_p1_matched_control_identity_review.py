from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STAGE = "V5_P3_F6_P1_MATCHED_CONTROL_IDENTITY_REVIEW"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "VALIDATION-EXPLORATORY"

EXPECTED_ITEMS = 13863
EXPECTED_ACTIVE_ITEMS = 3822
EXPECTED_CONTROL_ITEMS = 10041

PAIR_ID_ALIASES = (
    "pair_id",
    "pairid",
    "matched_pair_id",
    "attack_control_pair_id",
    "control_attack_pair_id",
)
PAIR_MEMBER_ALIASES = (
    "pair_member",
    "member",
    "pair_role",
    "control_attack",
    "attack_control",
    "run_role",
    "is_attack",
)
WINDOW_ALIASES = (
    "window_start",
    "window_start_epoch",
    "window_epoch_start",
    "start_epoch",
    "window_id",
    "sample_window_start",
)
SAMPLE_INDEX_ALIASES = (
    "sample_index",
    "item_index",
    "validation_index",
    "split_index",
    "index",
)
SPLIT_ALIASES = (
    "split",
    "dataset_split",
    "partition",
    "subset",
)
ATTACK_LABEL_ALIASES = (
    "y_attack",
    "attack_label",
    "graph_label",
    "is_attack",
    "active",
)
COUNT_LABEL_ALIASES = (
    "y_attacker_count",
    "attacker_count",
    "attack_count",
    "count_label",
)
RUN_ID_ALIASES = (
    "run_id",
    "run",
    "run_name",
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

TEXT_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".txt",
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


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def matching_key(fields: Iterable[str], aliases: tuple[str, ...]) -> str | None:
    normalized = {normalize(str(field)): str(field) for field in fields}
    for alias in aliases:
        key = normalized.get(normalize(alias))
        if key is not None:
            return key
    return None


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        numeric = float(text)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    integer = int(numeric)
    if abs(numeric - integer) > 1e-9:
        return None
    return integer


def classify_split(value: Any) -> str:
    return normalize(str(value))


def classify_pair_member(value: Any) -> str | None:
    text = normalize(str(value))
    if text in (
        "attack",
        "attacked",
        "active",
        "malicious",
        "injected",
        "1",
        "true",
        "yes",
    ):
        return "attack"
    if text in (
        "control",
        "benign",
        "clean",
        "normal",
        "baseline",
        "0",
        "false",
        "no",
    ):
        return "control"

    integer = parse_int(value)
    if integer == 1:
        return "attack"
    if integer == 0:
        return "control"
    return None


def candidate_roots(repo: Path, data_root: Path) -> list[Path]:
    roots = [
        data_root,
        repo / "reports/v5",
        repo / "data/processed/v5",
        repo / "scripts/v5",
    ]
    unique = []
    seen = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            continue
        if not resolved.exists() or resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def candidate_files(
    repo: Path,
    data_root: Path,
    p0_identity: dict[str, Any],
) -> list[Path]:
    paths = []
    seen = set()

    for candidate in p0_identity.get("csv_candidates", []):
        raw = candidate.get("path")
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)

    filename_terms = (
        "pair",
        "manifest",
        "window",
        "sample",
        "validation",
        "metadata",
        "index",
        "aligned",
    )

    for root in candidate_roots(repo, data_root):
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size <= 0 or size > 512 * 1024 * 1024:
                continue

            low_name = normalize(path.name)
            if not any(term in low_name for term in filename_terms):
                # Retain ordinary CSVs from the exact dataset root because the
                # relevant schema may not be reflected in the file name.
                if path.parent.resolve() != data_root.resolve():
                    continue

            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)

    return sorted(paths)


def sniff_delimiter(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"

    sample = path.read_text(
        encoding="utf-8",
        errors="replace",
    )[:65536]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except Exception:
        return ","


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    suffix = path.suffix.lower()

    if suffix in (".jsonl", ".ndjson"):
        rows = []
        fields = []
        field_set = set()
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError("JSONL row is not an object")
                row = {str(key): str(item) for key, item in value.items()}
                rows.append(row)
                for key in row:
                    if key not in field_set:
                        field_set.add(key)
                        fields.append(key)
        return fields, rows, "jsonl"

    delimiter = sniff_delimiter(path)
    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise RuntimeError("table has no header")
        fields = [str(field) for field in reader.fieldnames]
        rows = [dict(row) for row in reader]
    return fields, rows, delimiter


def filter_validation_rows(
    fields: list[str],
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    split_key = matching_key(fields, SPLIT_ALIASES)
    if split_key is None:
        return rows, {
            "split_key": None,
            "filter_applied": False,
            "input_rows": len(rows),
            "output_rows": len(rows),
        }

    normalized_values = Counter(
        classify_split(row.get(split_key, ""))
        for row in rows
    )
    validation_rows = [
        row for row in rows
        if classify_split(row.get(split_key, "")) in VALIDATION_VALUES
    ]

    return validation_rows, {
        "split_key": split_key,
        "filter_applied": True,
        "input_rows": len(rows),
        "output_rows": len(validation_rows),
        "observed_split_values": dict(normalized_values),
    }


def assign_validation_indices(
    fields: list[str],
    rows: list[dict[str, str]],
) -> tuple[list[int] | None, dict[str, Any]]:
    sample_key = matching_key(fields, SAMPLE_INDEX_ALIASES)

    if sample_key is not None:
        parsed = [parse_int(row.get(sample_key)) for row in rows]
        if all(value is not None for value in parsed):
            values = [int(value) for value in parsed]
            if (
                len(values) == EXPECTED_ITEMS
                and set(values) == set(range(EXPECTED_ITEMS))
            ):
                return values, {
                    "mode": "explicit_validation_index",
                    "sample_index_key": sample_key,
                    "exact_0_to_13862": True,
                }

            minimum = min(values) if values else None
            maximum = max(values) if values else None
            unique = len(set(values))
            return None, {
                "mode": "explicit_index_not_validation_local",
                "sample_index_key": sample_key,
                "minimum": minimum,
                "maximum": maximum,
                "unique_count": unique,
            }

    if len(rows) == EXPECTED_ITEMS:
        return list(range(EXPECTED_ITEMS)), {
            "mode": "validated_row_order",
            "sample_index_key": None,
            "reason": (
                "validation-filtered table has exactly 13,863 rows; row order "
                "must still agree with preserved labels"
            ),
        }

    return None, {
        "mode": "unresolved",
        "sample_index_key": sample_key,
        "row_count": len(rows),
    }


def evaluate_candidate(
    path: Path,
    fresh_labels: dict[str, np.ndarray],
) -> dict[str, Any]:
    evaluation: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "usable": False,
    }

    try:
        fields, all_rows, delimiter = read_table(path)
    except Exception as exc:
        evaluation["failure"] = f"table_read_failed:{exc!r}"
        return evaluation

    evaluation["delimiter"] = delimiter
    evaluation["fields"] = fields
    evaluation["input_row_count"] = len(all_rows)

    pair_key = matching_key(fields, PAIR_ID_ALIASES)
    member_key = matching_key(fields, PAIR_MEMBER_ALIASES)
    window_key = matching_key(fields, WINDOW_ALIASES)
    attack_key = matching_key(fields, ATTACK_LABEL_ALIASES)
    count_key = matching_key(fields, COUNT_LABEL_ALIASES)
    run_key = matching_key(fields, RUN_ID_ALIASES)

    evaluation["resolved_columns"] = {
        "pair_id": pair_key,
        "pair_member": member_key,
        "window": window_key,
        "attack_label": attack_key,
        "count_label": count_key,
        "run_id": run_key,
    }

    if pair_key is None:
        evaluation["failure"] = "missing_pair_id_column"
        return evaluation
    if member_key is None and attack_key is None:
        evaluation["failure"] = "missing_pair_member_and_attack_label"
        return evaluation

    rows, split_review = filter_validation_rows(fields, all_rows)
    evaluation["split_review"] = split_review
    evaluation["validation_row_count"] = len(rows)

    indices, index_review = assign_validation_indices(fields, rows)
    evaluation["index_review"] = index_review
    if indices is None:
        evaluation["failure"] = "validation_index_unresolved"
        return evaluation

    require(len(rows) == len(indices), "row/index length mismatch")

    records = []
    for row_order, (row, sample_index) in enumerate(zip(rows, indices)):
        if not 0 <= sample_index < EXPECTED_ITEMS:
            evaluation["failure"] = "sample_index_out_of_range"
            return evaluation

        if member_key is not None:
            member = classify_pair_member(row.get(member_key))
        else:
            member = classify_pair_member(row.get(attack_key))

        if member is None:
            evaluation["failure"] = "unrecognized_pair_member"
            evaluation["failure_row_order"] = row_order
            evaluation["failure_value"] = (
                row.get(member_key)
                if member_key is not None
                else row.get(attack_key)
            )
            return evaluation

        pair_value = str(row.get(pair_key, "")).strip()
        if pair_value == "":
            evaluation["failure"] = "empty_pair_id"
            evaluation["failure_row_order"] = row_order
            return evaluation

        window_value = (
            str(row.get(window_key, "")).strip()
            if window_key is not None
            else ""
        )
        count_value = (
            parse_int(row.get(count_key))
            if count_key is not None
            else None
        )
        attack_value = (
            parse_int(row.get(attack_key))
            if attack_key is not None
            else None
        )
        run_value = (
            str(row.get(run_key, "")).strip()
            if run_key is not None
            else ""
        )

        records.append({
            "sample_index": int(sample_index),
            "row_order": row_order,
            "pair_id": pair_value,
            "window": window_value,
            "member": member,
            "attack_label": attack_value,
            "count_label": count_value,
            "run_id": run_value,
        })

    # Strong row-order/index validation against preserved D1 labels.
    manifest_attack = np.asarray(
        [1 if record["member"] == "attack" else 0 for record in records],
        dtype=np.int64,
    )
    fresh_attack = np.asarray(fresh_labels["y_attack"], dtype=np.int64)
    indexed_fresh_attack = np.empty(EXPECTED_ITEMS, dtype=np.int64)
    for record in records:
        indexed_fresh_attack[record["sample_index"]] = (
            1 if record["member"] == "attack" else 0
        )

    attack_labels_exact = bool(
        np.array_equal(indexed_fresh_attack, fresh_attack)
    )
    evaluation["attack_label_alignment"] = {
        "exact": attack_labels_exact,
        "manifest_attack_count": int(np.sum(indexed_fresh_attack == 1)),
        "fresh_attack_count": int(np.sum(fresh_attack == 1)),
        "mismatch_count": int(
            np.sum(indexed_fresh_attack != fresh_attack)
        ),
    }
    if not attack_labels_exact:
        evaluation["failure"] = "attack_label_order_alignment_failed"
        return evaluation

    if count_key is not None:
        count_alignment_known = True
        count_mismatches = 0
        fresh_count = np.asarray(
            fresh_labels["y_attacker_count"],
            dtype=np.int64,
        )
        for record in records:
            value = record["count_label"]
            if value is None:
                count_alignment_known = False
                break
            if int(value) != int(fresh_count[record["sample_index"]]):
                count_mismatches += 1
        evaluation["count_label_alignment"] = {
            "available": count_alignment_known,
            "mismatch_count": count_mismatches,
            "exact": bool(count_alignment_known and count_mismatches == 0),
        }
        if count_alignment_known and count_mismatches != 0:
            evaluation["failure"] = "count_label_alignment_failed"
            return evaluation
    else:
        evaluation["count_label_alignment"] = {
            "available": False,
            "exact": None,
        }

    # Attack-centric pairing rule:
    # - every ATTACK identity must have exactly one CONTROL;
    # - CONTROL-only identities are allowed because validation contains
    #   substantially more control windows than active attack windows.
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"attack": [], "control": []}
    )
    for record in records:
        identity = (record["pair_id"], record["window"])
        grouped[identity][record["member"]].append(record)

    attack_groups = {
        identity: bucket
        for identity, bucket in grouped.items()
        if bucket["attack"]
    }
    control_only_group_count = sum(
        1
        for bucket in grouped.values()
        if not bucket["attack"] and bucket["control"]
    )

    invalid_attack_groups = {}
    mapping_rows = []

    for identity, bucket in attack_groups.items():
        if len(bucket["attack"]) != 1 or len(bucket["control"]) != 1:
            invalid_attack_groups[str(identity)] = {
                "attack_count": len(bucket["attack"]),
                "control_count": len(bucket["control"]),
                "attack_indices": [
                    row["sample_index"] for row in bucket["attack"]
                ],
                "control_indices": [
                    row["sample_index"] for row in bucket["control"]
                ],
            }
            continue

        attack_record = bucket["attack"][0]
        control_record = bucket["control"][0]
        attack_index = int(attack_record["sample_index"])
        control_index = int(control_record["sample_index"])

        if int(fresh_attack[attack_index]) != 1:
            invalid_attack_groups[str(identity)] = {
                "reason": "mapped attack index is not active in fresh labels",
                "attack_index": attack_index,
            }
            continue
        if int(fresh_attack[control_index]) != 0:
            invalid_attack_groups[str(identity)] = {
                "reason": "mapped control index is not control in fresh labels",
                "control_index": control_index,
            }
            continue

        mapping_rows.append({
            "attack_index": attack_index,
            "control_index": control_index,
            "pair_id": identity[0],
            "window_start": identity[1],
            "attack_run_id": attack_record["run_id"],
            "control_run_id": control_record["run_id"],
            "attacker_count": int(
                fresh_labels["y_attacker_count"][attack_index]
            ),
            "source_manifest": str(path),
        })

    mapping_rows.sort(key=lambda row: row["attack_index"])
    mapped_attack_indices = [row["attack_index"] for row in mapping_rows]
    expected_attack_indices = np.flatnonzero(fresh_attack == 1).tolist()

    all_active_mapped = mapped_attack_indices == expected_attack_indices
    duplicate_attack_indices = len(mapped_attack_indices) - len(
        set(mapped_attack_indices)
    )
    control_reuse_counts = Counter(
        row["control_index"] for row in mapping_rows
    )
    reused_control_count = sum(
        1 for count in control_reuse_counts.values() if count > 1
    )
    maximum_control_reuse = max(control_reuse_counts.values(), default=0)

    evaluation["pairing_review"] = {
        "identity": (
            "pair_id+window_start"
            if window_key is not None
            else "pair_id"
        ),
        "total_identity_groups": len(grouped),
        "attack_identity_groups": len(attack_groups),
        "control_only_identity_groups_allowed": control_only_group_count,
        "invalid_attack_group_count": len(invalid_attack_groups),
        "invalid_attack_group_preview": dict(
            list(invalid_attack_groups.items())[:30]
        ),
        "mapped_attack_item_count": len(mapping_rows),
        "expected_active_item_count": len(expected_attack_indices),
        "all_active_items_mapped": all_active_mapped,
        "duplicate_attack_index_count": duplicate_attack_indices,
        "unique_control_index_count": len(control_reuse_counts),
        "reused_control_index_count": reused_control_count,
        "maximum_control_reuse": maximum_control_reuse,
        "mapping_preview": mapping_rows[:20],
    }

    usable = bool(
        len(records) == EXPECTED_ITEMS
        and len(expected_attack_indices) == EXPECTED_ACTIVE_ITEMS
        and len(mapping_rows) == EXPECTED_ACTIVE_ITEMS
        and all_active_mapped
        and duplicate_attack_indices == 0
        and len(invalid_attack_groups) == 0
    )

    evaluation["usable"] = usable
    if not usable:
        evaluation["failure"] = "attack_centric_pair_mapping_incomplete"
    else:
        evaluation["mapping_rows"] = mapping_rows

    return evaluation


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

    p0_report_path = ig_dir / (
        "V5_P3_F6_P0_TASK_SPECIFIC_INTEGRATED_GRADIENTS_PREFLIGHT_REPORT.json"
    )
    p0_lock_path = ig_dir / (
        "V5_P3_F6_P0_TASK_SPECIFIC_INTEGRATED_GRADIENTS_PREFLIGHT_LOCK.json"
    )
    p0_identity_path = ig_dir / (
        "F6_P0_MATCHED_CONTROL_IDENTITY_ROUTE.json"
    )
    p0_protocol_path = ig_dir / (
        "F6_P0_TASK_SPECIFIC_INTEGRATED_GRADIENTS_PROTOCOL.json"
    )
    p0_source_path = ig_dir / (
        "F6_P0_FROZEN_EXECUTION_SOURCE_AND_RUNTIME_CONTRACT.json"
    )
    p0_authorization_path = ig_dir / (
        "F6_P0_EXECUTION_AUTHORIZATION.json"
    )
    fresh_cert_path = metric_dir / (
        "F4M_R3A_FRESH_REPLAY_14_METRIC_CERTIFICATION.json"
    )

    p0_report, p0_lock = verify_report_lock(
        p0_report_path,
        p0_lock_path,
    )
    require(
        p0_lock.get("actual_F6_execution_authorized") is False,
        "F6-P1 expected unresolved F6 execution authorization",
    )
    require(p0_lock.get("F7_authorized") is False, "F7 must remain held")
    require(
        p0_lock.get("feature_removal_authorized") is False,
        "feature removal must remain held",
    )

    for path in (
        p0_identity_path,
        p0_protocol_path,
        p0_source_path,
        p0_authorization_path,
        fresh_cert_path,
    ):
        require(path.is_file(), f"required artifact missing: {path}")

    p0_identity = load_json(p0_identity_path)
    p0_protocol = load_json(p0_protocol_path)
    p0_source = load_json(p0_source_path)
    fresh_cert = load_json(fresh_cert_path)

    require(
        p0_protocol["baseline"]["type"]
        == "matched_control_feature_tensor",
        "F6 baseline protocol changed",
    )
    require(
        p0_protocol["sample_selection"]["total_items"] == 512,
        "F6 sample count changed",
    )
    require(
        p0_protocol["integrated_gradients"]["steps"] == 64,
        "F6 integration-step count changed",
    )

    fresh_npz = Path(fresh_cert["npz"]).resolve()
    require(fresh_npz.is_file(), f"fresh validation NPZ missing: {fresh_npz}")
    require(
        sha256_file(fresh_npz) == fresh_cert["npz_sha256"],
        "fresh validation NPZ hash changed",
    )

    with np.load(fresh_npz, allow_pickle=False) as archive:
        fresh_labels = {
            "y_attack": np.asarray(archive["y_attack"]).astype(np.int64),
            "y_attacker_count": np.asarray(
                archive["y_attacker_count"]
            ).astype(np.int64),
        }

    require(
        fresh_labels["y_attack"].shape == (EXPECTED_ITEMS,),
        "fresh attack-label shape changed",
    )
    require(
        fresh_labels["y_attacker_count"].shape == (EXPECTED_ITEMS,),
        "fresh count-label shape changed",
    )
    require(
        int(np.sum(fresh_labels["y_attack"] == 1)) == EXPECTED_ACTIVE_ITEMS,
        "active validation count changed",
    )
    require(
        int(np.sum(fresh_labels["y_attack"] == 0)) == EXPECTED_CONTROL_ITEMS,
        "control validation count changed",
    )

    candidates = candidate_files(repo, data_root, p0_identity)
    evaluations = []
    usable_candidates = []

    for index, path in enumerate(candidates, start=1):
        print(
            f"candidate_progress={index}/{len(candidates)}:{path}",
            flush=True,
        )
        evaluation = evaluate_candidate(path, fresh_labels)
        evaluations.append(evaluation)
        if evaluation.get("usable"):
            usable_candidates.append(evaluation)

    # Deduplicate exact equivalent mappings. Multiple copied manifests are
    # acceptable only when they produce the same attack->control mapping.
    mapping_fingerprints: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for evaluation in usable_candidates:
        pairs = [
            (row["attack_index"], row["control_index"])
            for row in evaluation["mapping_rows"]
        ]
        payload = json.dumps(pairs, separators=(",", ":")).encode("utf-8")
        fingerprint = hashlib.sha256(payload).hexdigest()
        mapping_fingerprints[fingerprint].append(evaluation)

    mapping_resolved = len(mapping_fingerprints) == 1
    selected = None
    selected_mapping = None
    selected_fingerprint = None

    if mapping_resolved:
        selected_fingerprint, equivalent = next(
            iter(mapping_fingerprints.items())
        )
        equivalent.sort(
            key=lambda row: (
                0 if row["pairing_review"]["identity"]
                == "pair_id+window_start" else 1,
                len(row["path"]),
                row["path"],
            )
        )
        selected = equivalent[0]
        selected_mapping = selected["mapping_rows"]

    inventory_path = output_dir / (
        "F6_P1_MATCHED_CONTROL_CANDIDATE_EVALUATIONS.json"
    )
    mapping_json_path = output_dir / (
        "F6_P1_FROZEN_ATTACK_TO_CONTROL_MAPPING.json"
    )
    mapping_csv_path = output_dir / (
        "F6_P1_FROZEN_ATTACK_TO_CONTROL_MAPPING.csv"
    )
    decision_path = output_dir / (
        "F6_P1_MATCHED_CONTROL_IDENTITY_DECISION.json"
    )

    compact_evaluations = []
    for evaluation in evaluations:
        compact = {
            key: value
            for key, value in evaluation.items()
            if key != "mapping_rows"
        }
        compact_evaluations.append(compact)

    atomic_json(
        inventory_path,
        {
            "candidate_count": len(candidates),
            "usable_candidate_count": len(usable_candidates),
            "unique_mapping_fingerprint_count": len(mapping_fingerprints),
            "mapping_fingerprints": {
                fingerprint: [
                    {
                        "path": evaluation["path"],
                        "sha256": evaluation["sha256"],
                        "identity": evaluation[
                            "pairing_review"
                        ]["identity"],
                    }
                    for evaluation in rows
                ]
                for fingerprint, rows in mapping_fingerprints.items()
            },
            "evaluations": compact_evaluations,
        },
    )

    if mapping_resolved and selected_mapping is not None:
        strata = Counter(
            row["attacker_count"] for row in selected_mapping
        )
        strata_ready = all(strata.get(value, 0) >= 128 for value in (1, 2, 3, 4))
        require(strata_ready, f"resolved mapping lacks count capacity: {strata}")

        mapping_document = {
            "status": "FROZEN",
            "classification": CLASSIFICATION,
            "source_manifest": selected["path"],
            "source_manifest_sha256": selected["sha256"],
            "mapping_fingerprint_sha256": selected_fingerprint,
            "identity": selected["pairing_review"]["identity"],
            "attack_item_count": len(selected_mapping),
            "unique_control_item_count": len(
                set(row["control_index"] for row in selected_mapping)
            ),
            "attacker_count_strata": {
                str(key): int(value)
                for key, value in sorted(strata.items())
            },
            "all_attack_items_mapped": True,
            "control_only_identity_groups_allowed": True,
            "rationale": (
                "Validation contains more control windows than active attack "
                "windows. Correct matching therefore requires one unique "
                "control for every attack identity while permitting unmatched "
                "control-only identities."
            ),
            "mapping": selected_mapping,
        }
        atomic_json(mapping_json_path, mapping_document)
        write_csv(mapping_csv_path, selected_mapping)
    else:
        strata_ready = False
        mapping_document = None

    actual_F6_execution_authorized = bool(
        mapping_resolved
        and selected_mapping is not None
        and len(selected_mapping) == EXPECTED_ACTIVE_ITEMS
        and strata_ready
    )

    atomic_json(
        decision_path,
        {
            "F6_P0_complete": True,
            "F6_authorized_by_F5R": True,
            "prior_failure": (
                "matched-control resolver incorrectly treated control-only "
                "identity groups as invalid"
            ),
            "attack_centric_rule": (
                "every attack identity must have exactly one control; "
                "control-only identities are allowed"
            ),
            "candidate_count": len(candidates),
            "usable_candidate_count": len(usable_candidates),
            "unique_mapping_fingerprint_count": len(mapping_fingerprints),
            "mapping_resolved": mapping_resolved,
            "selected_source": (
                {
                    "path": selected["path"],
                    "sha256": selected["sha256"],
                    "identity": selected["pairing_review"]["identity"],
                    "fingerprint": selected_fingerprint,
                }
                if selected is not None
                else None
            ),
            "mapped_attack_item_count": (
                len(selected_mapping)
                if selected_mapping is not None
                else 0
            ),
            "count_strata_ready": strata_ready,
            "actual_F6_execution_authorized": (
                actual_F6_execution_authorized
            ),
            "F7_retraining_ablation_authorized": False,
            "feature_removal_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "sealed_test_access": False,
            "next_stage": (
                "V5_P3_F6_TASK_SPECIFIC_INTEGRATED_GRADIENTS"
                if actual_F6_execution_authorized
                else "V5_P3_F6_P2_TARGETED_MANIFEST_ROUTE_RECOVERY"
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
            "Re-evaluate existing and newly discovered validation manifests "
            "with the scientifically correct attack-centric matching rule, "
            "prove validation row order against preserved D1 labels, freeze "
            "the complete attack-to-control mapping, and decide whether "
            "actual F6 execution may begin."
        ),
        "finding": {
            "prior_resolver_issue": (
                "The F6-P0 resolver required every identity group to contain "
                "one attack and one control. That is incompatible with a "
                "validation split containing 3,822 attack items and 10,041 "
                "control items because legitimate control-only windows exist."
            ),
            "correct_rule": (
                "Require exactly one control for every attack identity and "
                "allow control-only identities."
            ),
            "candidate_count": len(candidates),
            "usable_candidate_count": len(usable_candidates),
            "unique_mapping_fingerprint_count": len(mapping_fingerprints),
            "mapping_resolved": mapping_resolved,
            "selected_source": (
                {
                    "path": selected["path"],
                    "sha256": selected["sha256"],
                    "identity": selected["pairing_review"]["identity"],
                    "mapping_fingerprint": selected_fingerprint,
                }
                if selected is not None
                else None
            ),
            "mapped_attack_item_count": (
                len(selected_mapping)
                if selected_mapping is not None
                else 0
            ),
        },
        "decision": {
            "F6_P1_complete": True,
            "actual_F6_execution_authorized": (
                actual_F6_execution_authorized
            ),
            "F7_retraining_ablation_authorized": False,
            "feature_removal_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F6_TASK_SPECIFIC_INTEGRATED_GRADIENTS"
                if actual_F6_execution_authorized
                else "V5_P3_F6_P2_TARGETED_MANIFEST_ROUTE_RECOVERY"
            ),
        },
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "validation_feature_tensors_loaded": False,
            "validation_label_payloads_loaded": True,
            "validation_manifest_payloads_loaded": True,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "thresholds_changed": False,
            "metric_formulas_changed": False,
            "model_weights_changed": False,
        },
        "artifacts": {
            "candidate_evaluations": str(inventory_path),
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
            "F6_P0_report_sha256": sha256_file(p0_report_path),
            "F6_P0_lock_sha256": sha256_file(p0_lock_path),
            "F6_P0_identity_route_sha256": sha256_file(p0_identity_path),
            "F6_P0_protocol_sha256": sha256_file(p0_protocol_path),
            "F6_P0_source_contract_sha256": sha256_file(p0_source_path),
            "F6_P0_authorization_sha256": sha256_file(
                p0_authorization_path
            ),
            "fresh_certification_sha256": sha256_file(fresh_cert_path),
            "fresh_npz_sha256": sha256_file(fresh_npz),
            "installed_script_sha256": sha256_file(installed_script),
            "candidate_evaluations_sha256": sha256_file(inventory_path),
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
            "candidate_evaluations_sha256": sha256_file(inventory_path),
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
            "actual_F6_execution_authorized": (
                actual_F6_execution_authorized
            ),
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
    print(f"classification={CLASSIFICATION}")
    print("F6_P0_complete=true")
    print("F6_authorized_by_F5R=true")
    print(
        "prior_resolver_issue="
        "control_only_identity_groups_were_incorrectly_rejected"
    )
    print(
        "attack_centric_rule="
        "one_unique_control_per_attack_control_only_groups_allowed"
    )
    print(f"candidate_count={len(candidates)}")
    print(f"usable_candidate_count={len(usable_candidates)}")
    print(
        "unique_mapping_fingerprint_count="
        f"{len(mapping_fingerprints)}"
    )
    print(f"mapping_resolved={str(mapping_resolved).lower()}")

    if selected is not None and selected_mapping is not None:
        print(f"selected_manifest={selected['path']}")
        print(f"selected_manifest_sha256={selected['sha256']}")
        print(
            "selected_identity="
            f"{selected['pairing_review']['identity']}"
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
        print("selected_manifest=None")
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
    print("validation_feature_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={report['decision']['next_stage']}")
    print(f"candidate_evaluations={inventory_path}")
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
