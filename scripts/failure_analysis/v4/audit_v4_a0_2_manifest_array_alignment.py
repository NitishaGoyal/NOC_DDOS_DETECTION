#!/usr/bin/env python3
"""
V4-A0.2: manifest-to-array alignment audit.

Read-only audit that aligns the authoritative V4 manifest with:
- completed_runs.txt
- metadata.json
- run_index.npy
- split_id.npy
- profile_id.npy
- attack_kind_id.npy
- attacker_count.npy
- strength.npy
- y_graph.npy
- y_node.npy

The script:
- discovers or accepts the exact V4 manifest;
- supports reasonable manifest-column aliases;
- resolves categorical code mappings from metadata or observed one-to-one alignment;
- verifies all 908 runs and attacker masks;
- records optional family, active-core, victim, timing, and route metadata coverage;
- performs no model construction, inference, thresholding, or training;
- never modifies the dataset or manifest.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCRIPT_VERSION = "1.0.0"
MANIFEST_BASENAME = "v4_dataset_manifest_all16.csv"

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "run_id": (
        "run_id", "run", "simulation_id", "simulation", "run_name", "name",
    ),
    "split": (
        "split", "dataset_split", "split_name",
    ),
    "graph_label": (
        "y_graph", "graph_label", "label", "is_attack", "attack_label",
        "graph_target", "attack",
    ),
    "profile": (
        "profile", "workload_profile", "traffic_profile", "background_profile",
    ),
    "attack_kind": (
        "attack_kind", "attack_type", "kind", "attack_mode",
    ),
    "attacker_count": (
        "attacker_count", "num_attackers", "n_attackers", "number_of_attackers",
        "mip_count",
    ),
    "strength": (
        "strength", "attack_strength", "injection_strength", "rate",
    ),
    "attackers": (
        "attackers", "attacker_cores", "attacker_ids", "malicious_cores",
        "mip_nodes", "mips", "attacker_set",
    ),
    "background_id": (
        "background_id", "background", "background_family_id",
    ),
    "matched_benign_run_id": (
        "matched_benign_run_id", "matched_benign", "benign_run_id",
        "control_run_id", "matched_control_run_id",
    ),
    "pair_id": (
        "pair_id", "matched_pair_id", "control_pair_id",
    ),
    "scenario_family": (
        "scenario_family", "scenario_family_id", "family",
    ),
    "placement_family": (
        "placement_family", "placement_family_id",
    ),
    "active_cores": (
        "active_cores", "active_benign_cores", "benign_active_cores",
        "legitimate_active_cores", "workload_cores",
    ),
    "victims": (
        "victims", "victim_cores", "victim_ids", "vip_nodes", "vips",
        "victim_set",
    ),
    "victim_count": (
        "victim_count", "num_victims", "n_victims", "number_of_victims",
    ),
    "timing_family": (
        "timing_family", "timing_family_id", "temporal_family",
    ),
    "route_family": (
        "route_family", "route_family_id", "path_family",
    ),
    "seed": (
        "seed", "simulation_seed", "traffic_seed", "background_seed",
    ),
    "burst_length": (
        "burst_length", "burst_epochs", "burst_len", "on_length",
    ),
    "pause_length": (
        "pause_length", "pause_epochs", "pause_len", "off_length",
    ),
    "notes": (
        "notes", "note", "comments", "comment",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def normalize_name(value: Any) -> str:
    return str(value).strip()


def canonical_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def resolve_columns(fieldnames: Iterable[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    canonical_to_original: dict[str, str] = {}
    for original in fieldnames:
        canonical_to_original.setdefault(canonical_column_name(original), original)

    resolved: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}

    for logical, aliases in COLUMN_ALIASES.items():
        matches = []
        for alias in aliases:
            canonical = canonical_column_name(alias)
            if canonical in canonical_to_original:
                matches.append(canonical_to_original[canonical])

        unique_matches = list(dict.fromkeys(matches))
        if unique_matches:
            resolved[logical] = unique_matches[0]
        if len(unique_matches) > 1:
            collisions[logical] = unique_matches

    return resolved, collisions


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str], str]:
    encodings = ("utf-8-sig", "utf-8")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ValueError("Manifest has no CSV header")
                rows = [
                    {str(k): ("" if v is None else str(v)) for k, v in row.items()}
                    for row in reader
                ]
                return rows, list(reader.fieldnames), encoding
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Unable to parse manifest {path}: {last_error}")


def discover_manifest(
    explicit_manifest: Path | None,
    search_roots: list[Path],
    data_dir: Path,
) -> tuple[Path, list[str]]:
    if explicit_manifest is not None:
        resolved = explicit_manifest.expanduser().resolve()
        if not resolved.is_file():
            raise SystemExit(f"STOP: explicit manifest does not exist: {resolved}")
        return resolved, [str(resolved)]

    candidates: set[Path] = set()
    discovery_log: list[str] = []

    likely_direct = [
        data_dir / MANIFEST_BASENAME,
        data_dir.parent / MANIFEST_BASENAME,
        data_dir.parent / "dataset_v4" / MANIFEST_BASENAME,
        data_dir.parent.parent / "dataset_v4" / MANIFEST_BASENAME,
    ]
    for candidate in likely_direct:
        if candidate.is_file():
            candidates.add(candidate.resolve())

    for root in search_roots:
        resolved_root = root.expanduser().resolve()
        if not resolved_root.exists():
            discovery_log.append(f"MISSING_SEARCH_ROOT\t{resolved_root}")
            continue

        if resolved_root.is_file():
            if resolved_root.name == MANIFEST_BASENAME:
                candidates.add(resolved_root)
            continue

        for candidate in resolved_root.rglob(MANIFEST_BASENAME):
            if candidate.is_file():
                candidates.add(candidate.resolve())

    ordered = sorted(candidates)
    discovery_log.extend(str(candidate) for candidate in ordered)

    if len(ordered) == 0:
        raise SystemExit(
            "STOP: no V4 manifest found. Supply --manifest explicitly or add a valid --search-root."
        )

    if len(ordered) > 1:
        raise SystemExit(
            "STOP: multiple V4 manifests found; choose the exact one with --manifest:\n"
            + "\n".join(str(path) for path in ordered)
        )

    return ordered[0], discovery_log


def parse_bool_or_binary(value: Any) -> int | None:
    text = normalize_name(value).lower()
    if text in {"", "none", "null", "na", "n/a"}:
        return None
    if text in {"1", "true", "yes", "y", "attack", "malicious"}:
        return 1
    if text in {"0", "false", "no", "n", "normal", "benign"}:
        return 0
    try:
        number = float(text)
        if number in (0.0, 1.0):
            return int(number)
    except ValueError:
        pass
    return None


def parse_int(value: Any) -> int | None:
    text = normalize_name(value)
    if text.lower() in {"", "none", "null", "na", "n/a", "nan"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    text = normalize_name(value)
    if text.lower() in {"", "none", "null", "na", "n/a", "nan"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_int_set(value: Any) -> tuple[int, ...] | None:
    text = normalize_name(value)
    if text.lower() in {"", "none", "null", "na", "n/a", "[]", "()", "{}"}:
        return tuple()

    # First attempt Python/JSON-style structures.
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            values = [int(item) for item in parsed]
            return tuple(sorted(set(values)))
        if isinstance(parsed, (int, float)):
            return (int(parsed),)
    except (ValueError, SyntaxError, TypeError):
        pass

    # Accept 1,2; 1|2; 1 2; 1-2-3 and similar simple encodings.
    tokens = re.findall(r"-?\d+", text)
    if not tokens:
        return None

    values = [int(token) for token in tokens]
    return tuple(sorted(set(values)))


def extract_metadata_code_map(metadata: Any, field: str) -> dict[int, str]:
    if not isinstance(metadata, dict):
        return {}

    code_maps = metadata.get("code_maps")
    if not isinstance(code_maps, dict):
        return {}

    possible_keys = [
        field,
        f"{field}_id",
        field.replace("_kind", ""),
    ]

    raw = None
    for key in possible_keys:
        if key in code_maps:
            raw = code_maps[key]
            break

    if not isinstance(raw, dict):
        return {}

    result: dict[int, str] = {}
    for key, value in raw.items():
        try:
            result[int(value)] = normalize_name(key)
            continue
        except (TypeError, ValueError):
            pass
        try:
            result[int(key)] = normalize_name(value)
        except (TypeError, ValueError):
            continue
    return result


def derive_observed_code_map(
    manifest_values: list[str],
    array_codes: list[int],
) -> dict[str, Any]:
    name_to_codes: dict[str, set[int]] = defaultdict(set)
    code_to_names: dict[int, set[str]] = defaultdict(set)

    for name, code in zip(manifest_values, array_codes):
        normalized = normalize_name(name)
        name_to_codes[normalized].add(int(code))
        code_to_names[int(code)].add(normalized)

    normalized_name_to_codes = {
        name: sorted(codes) for name, codes in sorted(name_to_codes.items())
    }
    normalized_code_to_names = {
        str(code): sorted(names) for code, names in sorted(code_to_names.items())
    }

    one_to_one = (
        all(len(codes) == 1 for codes in name_to_codes.values())
        and all(len(names) == 1 for names in code_to_names.values())
    )

    return {
        "name_to_codes": normalized_name_to_codes,
        "code_to_names": normalized_code_to_names,
        "one_to_one": one_to_one,
        "empty_manifest_values": sum(name == "" for name in manifest_values),
    }


def compare_categorical(
    manifest_values: list[str],
    array_codes: list[int],
    metadata_map: dict[int, str],
) -> dict[str, Any]:
    observed = derive_observed_code_map(manifest_values, array_codes)

    metadata_comparisons = []
    metadata_match = True
    if metadata_map:
        for manifest_name, code in zip(manifest_values, array_codes):
            expected_name = metadata_map.get(int(code))
            matches = (
                expected_name is not None
                and normalize_name(expected_name) == normalize_name(manifest_name)
            )
            metadata_comparisons.append(matches)
        metadata_match = all(metadata_comparisons)

    return {
        "metadata_code_map": metadata_map,
        "observed_mapping": observed,
        "metadata_map_available": bool(metadata_map),
        "metadata_map_matches_all_runs": metadata_match if metadata_map else None,
        "alignment_pass": (
            metadata_match if metadata_map else observed["one_to_one"]
        ),
    }


def candidate_metadata_run_sequence(metadata: Any, expected_runs: int) -> tuple[str | None, list[str] | None]:
    if not isinstance(metadata, dict):
        return None, None

    preferred = (
        "selected_runs",
        "run_ids",
        "runs",
        "completed_runs",
        "run_order",
    )
    for key in preferred:
        value = metadata.get(key)
        if isinstance(value, list) and len(value) == expected_runs:
            if all(isinstance(item, (str, int, float)) for item in value):
                return key, [normalize_name(item) for item in value]

    for key, value in metadata.items():
        if (
            isinstance(value, list)
            and len(value) == expected_runs
            and all(isinstance(item, (str, int, float)) for item in value)
        ):
            return key, [normalize_name(item) for item in value]

    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=[],
        help="Root to search recursively for v4_dataset_manifest_all16.csv; repeatable.",
    )
    parser.add_argument("--expected-runs", type=int, default=908)
    parser.add_argument("--expected-samples-per-run", type=int, default=3293)
    parser.add_argument("--expected-nodes", type=int, default=16)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not data_dir.is_dir():
        raise SystemExit(f"STOP: data directory does not exist: {data_dir}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"STOP: output directory already non-empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    source_before = {
        "size": data_dir.stat().st_size,
        "mtime_ns": data_dir.stat().st_mtime_ns,
        "mode": data_dir.stat().st_mode,
    }

    started = utc_now()
    print("V4-A0.2 MANIFEST-TO-ARRAY ALIGNMENT START", flush=True)

    search_roots = list(args.search_root)
    if not search_roots:
        search_roots = [
            data_dir.parent,
            data_dir.parent.parent,
        ]

    manifest_path, discovery_log = discover_manifest(
        explicit_manifest=args.manifest,
        search_roots=search_roots,
        data_dir=data_dir,
    )
    print(f"manifest={manifest_path}", flush=True)

    (output_dir / "manifest_discovery.txt").write_text(
        "\n".join(discovery_log) + "\n",
        encoding="utf-8",
    )

    rows, fieldnames, encoding = read_manifest(manifest_path)
    resolved, collisions = resolve_columns(fieldnames)

    required_logical_fields = [
        "run_id",
        "split",
        "graph_label",
        "profile",
        "attack_kind",
        "attacker_count",
        "strength",
        "attackers",
    ]
    missing_required_fields = [
        field for field in required_logical_fields if field not in resolved
    ]

    schema = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "encoding": encoding,
        "row_count": len(rows),
        "field_count": len(fieldnames),
        "fieldnames": fieldnames,
        "resolved_columns": resolved,
        "alias_collisions": collisions,
        "required_logical_fields": required_logical_fields,
        "missing_required_logical_fields": missing_required_fields,
        "optional_field_presence": {
            field: field in resolved
            for field in COLUMN_ALIASES
            if field not in required_logical_fields
        },
    }
    write_json(output_dir / "manifest_schema.json", schema)

    if missing_required_fields:
        raise SystemExit(
            "STOP: manifest is missing required logical fields: "
            + ", ".join(missing_required_fields)
        )

    metadata_path = data_dir / "metadata.json"
    completed_path = data_dir / "completed_runs.txt"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    completed_runs = [
        line.strip()
        for line in completed_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line.strip()
    ]

    run_index = np.load(data_dir / "run_index.npy", mmap_mode="r")
    split_id = np.load(data_dir / "split_id.npy", mmap_mode="r")
    profile_id = np.load(data_dir / "profile_id.npy", mmap_mode="r")
    attack_kind_id = np.load(data_dir / "attack_kind_id.npy", mmap_mode="r")
    attacker_count = np.load(data_dir / "attacker_count.npy", mmap_mode="r")
    strength = np.load(data_dir / "strength.npy", mmap_mode="r")
    y_graph = np.load(data_dir / "y_graph.npy", mmap_mode="r")
    y_node = np.load(data_dir / "y_node.npy", mmap_mode="r")

    first_indices = np.arange(args.expected_runs, dtype=np.int64) * args.expected_samples_per_run

    run_index_first = np.asarray(run_index[first_indices], dtype=np.int64)
    expected_run_index = np.arange(args.expected_runs, dtype=np.int64)
    run_index_order_ok = np.array_equal(run_index_first, expected_run_index)

    completed_count_ok = len(completed_runs) == args.expected_runs
    completed_unique_ok = len(set(completed_runs)) == len(completed_runs)

    manifest_run_column = resolved["run_id"]
    manifest_ids = [normalize_name(row[manifest_run_column]) for row in rows]
    manifest_unique_ok = len(set(manifest_ids)) == len(manifest_ids)
    manifest_count_ok = len(rows) == args.expected_runs

    manifest_by_run: dict[str, dict[str, str]] = {}
    duplicate_manifest_ids = []
    for row, run_id_value in zip(rows, manifest_ids):
        if run_id_value in manifest_by_run:
            duplicate_manifest_ids.append(run_id_value)
        else:
            manifest_by_run[run_id_value] = row

    completed_set = set(completed_runs)
    manifest_set = set(manifest_ids)
    completed_missing_from_manifest = sorted(completed_set - manifest_set)
    manifest_extra_vs_completed = sorted(manifest_set - completed_set)

    metadata_run_key, metadata_runs = candidate_metadata_run_sequence(
        metadata,
        args.expected_runs,
    )
    metadata_order_matches_completed = (
        metadata_runs == completed_runs if metadata_runs is not None else None
    )

    run_order_rows = []
    manifest_values_by_field: dict[str, list[str]] = defaultdict(list)
    array_codes_by_field: dict[str, list[int]] = defaultdict(list)

    graph_mismatches: list[str] = []
    attacker_count_mismatches: list[str] = []
    attacker_mask_mismatches: list[str] = []
    strength_mismatches: list[str] = []
    invalid_attacker_ids: list[str] = []
    missing_completed_rows: list[str] = []

    for run_pos, run_id_value in enumerate(completed_runs):
        manifest_row = manifest_by_run.get(run_id_value)
        sample_index = run_pos * args.expected_samples_per_run

        if manifest_row is None:
            missing_completed_rows.append(run_id_value)
            continue

        array_graph = int(y_graph[sample_index])
        array_split = int(split_id[sample_index])
        array_profile = int(profile_id[sample_index])
        array_kind = int(attack_kind_id[sample_index])
        array_attacker_count = int(attacker_count[sample_index])
        array_strength = int(strength[sample_index])
        array_attackers = tuple(
            int(index)
            for index in np.flatnonzero(
                np.asarray(y_node[sample_index]) == 1.0
            ).tolist()
        )

        manifest_graph = parse_bool_or_binary(
            manifest_row[resolved["graph_label"]]
        )
        manifest_attacker_count = parse_int(
            manifest_row[resolved["attacker_count"]]
        )
        manifest_strength = parse_int(
            manifest_row[resolved["strength"]]
        )
        manifest_attackers = parse_int_set(
            manifest_row[resolved["attackers"]]
        )

        if manifest_graph != array_graph:
            graph_mismatches.append(run_id_value)

        if manifest_attacker_count != array_attacker_count:
            attacker_count_mismatches.append(run_id_value)

        if manifest_attackers is None:
            attacker_mask_mismatches.append(run_id_value)
        else:
            if any(
                attacker < 0 or attacker >= args.expected_nodes
                for attacker in manifest_attackers
            ):
                invalid_attacker_ids.append(run_id_value)
            if manifest_attackers != array_attackers:
                attacker_mask_mismatches.append(run_id_value)

        # Compare strength directly for attack runs. Normal sentinels are reported separately.
        strength_match = True
        if array_graph == 1:
            strength_match = manifest_strength == array_strength
            if not strength_match:
                strength_mismatches.append(run_id_value)

        manifest_values_by_field["split"].append(
            normalize_name(manifest_row[resolved["split"]])
        )
        array_codes_by_field["split"].append(array_split)

        manifest_values_by_field["profile"].append(
            normalize_name(manifest_row[resolved["profile"]])
        )
        array_codes_by_field["profile"].append(array_profile)

        manifest_values_by_field["attack_kind"].append(
            normalize_name(manifest_row[resolved["attack_kind"]])
        )
        array_codes_by_field["attack_kind"].append(array_kind)

        output_row = {
            "run_position": run_pos,
            "run_index_array": int(run_index[sample_index]),
            "completed_run_id": run_id_value,
            "manifest_row_found": True,
            "manifest_graph_label": manifest_graph,
            "array_graph_label": array_graph,
            "graph_label_match": manifest_graph == array_graph,
            "manifest_split": manifest_values_by_field["split"][-1],
            "array_split_id": array_split,
            "manifest_profile": manifest_values_by_field["profile"][-1],
            "array_profile_id": array_profile,
            "manifest_attack_kind": manifest_values_by_field["attack_kind"][-1],
            "array_attack_kind_id": array_kind,
            "manifest_attacker_count": manifest_attacker_count,
            "array_attacker_count": array_attacker_count,
            "attacker_count_match": manifest_attacker_count == array_attacker_count,
            "manifest_attackers": (
                "" if manifest_attackers is None
                else "-".join(str(v) for v in manifest_attackers)
            ),
            "array_attackers": "-".join(str(v) for v in array_attackers),
            "attacker_mask_match": manifest_attackers == array_attackers,
            "manifest_strength": manifest_strength,
            "array_strength": array_strength,
            "strength_match_for_attack": strength_match,
        }

        for optional_field in (
            "background_id",
            "matched_benign_run_id",
            "pair_id",
            "scenario_family",
            "placement_family",
            "active_cores",
            "victims",
            "victim_count",
            "timing_family",
            "route_family",
            "seed",
            "burst_length",
            "pause_length",
            "notes",
        ):
            column = resolved.get(optional_field)
            output_row[f"manifest_{optional_field}"] = (
                normalize_name(manifest_row[column])
                if column is not None else ""
            )

        run_order_rows.append(output_row)

    categorical_alignment = {}
    for field, array_name in (
        ("split", "split"),
        ("profile", "profile"),
        ("attack_kind", "attack_kind"),
    ):
        metadata_map = extract_metadata_code_map(metadata, array_name)
        categorical_alignment[field] = compare_categorical(
            manifest_values_by_field[field],
            array_codes_by_field[field],
            metadata_map,
        )

    write_json(
        output_dir / "manifest_metadata_code_maps.json",
        categorical_alignment,
    )

    run_order_csv = output_dir / "manifest_run_order_alignment.csv"
    if run_order_rows:
        with run_order_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(run_order_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(run_order_rows)

    optional_coverage = {}
    for logical in (
        "background_id",
        "matched_benign_run_id",
        "pair_id",
        "scenario_family",
        "placement_family",
        "active_cores",
        "victims",
        "victim_count",
        "timing_family",
        "route_family",
        "seed",
        "burst_length",
        "pause_length",
        "notes",
    ):
        column = resolved.get(logical)
        if column is None:
            optional_coverage[logical] = {
                "column_present": False,
                "nonempty_rows": 0,
                "empty_rows": len(rows),
                "unique_nonempty_values": 0,
            }
        else:
            values = [normalize_name(row[column]) for row in rows]
            nonempty = [value for value in values if value != ""]
            optional_coverage[logical] = {
                "column_present": True,
                "column_name": column,
                "nonempty_rows": len(nonempty),
                "empty_rows": len(values) - len(nonempty),
                "unique_nonempty_values": len(set(nonempty)),
                "preview": nonempty[:10],
            }

    write_json(
        output_dir / "manifest_optional_metadata_coverage.json",
        optional_coverage,
    )

    source_after = {
        "size": data_dir.stat().st_size,
        "mtime_ns": data_dir.stat().st_mtime_ns,
        "mode": data_dir.stat().st_mode,
    }

    hard_checks = {
        "exactly_one_manifest_selected": manifest_path.is_file(),
        "manifest_row_count_matches_expected": manifest_count_ok,
        "manifest_run_ids_unique": manifest_unique_ok,
        "completed_run_count_matches_expected": completed_count_ok,
        "completed_run_ids_unique": completed_unique_ok,
        "completed_and_manifest_run_sets_identical": (
            len(completed_missing_from_manifest) == 0
            and len(manifest_extra_vs_completed) == 0
        ),
        "run_index_first_samples_are_0_through_907": run_index_order_ok,
        "metadata_run_order_matches_completed_when_available": (
            metadata_order_matches_completed is not False
        ),
        "all_completed_runs_have_manifest_rows": len(missing_completed_rows) == 0,
        "graph_labels_align": len(graph_mismatches) == 0,
        "attacker_counts_align": len(attacker_count_mismatches) == 0,
        "attacker_masks_align": len(attacker_mask_mismatches) == 0,
        "all_manifest_attacker_ids_valid": len(invalid_attacker_ids) == 0,
        "attack_strengths_align": len(strength_mismatches) == 0,
        "split_code_alignment_pass": categorical_alignment["split"][
            "alignment_pass"
        ],
        "profile_code_alignment_pass": categorical_alignment["profile"][
            "alignment_pass"
        ],
        "attack_kind_code_alignment_pass": categorical_alignment["attack_kind"][
            "alignment_pass"
        ],
        "dataset_root_metadata_unchanged": source_before == source_after,
    }

    mismatch_summary = {
        "duplicate_manifest_run_ids": sorted(set(duplicate_manifest_ids)),
        "completed_missing_from_manifest": completed_missing_from_manifest,
        "manifest_extra_vs_completed": manifest_extra_vs_completed,
        "missing_completed_rows": missing_completed_rows,
        "graph_label_mismatches": graph_mismatches,
        "attacker_count_mismatches": attacker_count_mismatches,
        "attacker_mask_mismatches": attacker_mask_mismatches,
        "invalid_manifest_attacker_ids": invalid_attacker_ids,
        "attack_strength_mismatches": strength_mismatches,
    }
    write_json(
        output_dir / "manifest_array_alignment_mismatches.json",
        mismatch_summary,
    )

    manifest_alignment = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "metadata_sha256": sha256_file(metadata_path),
        "completed_runs_sha256": sha256_file(completed_path),
        "manifest_rows": len(rows),
        "completed_runs": len(completed_runs),
        "metadata_run_sequence_key": metadata_run_key,
        "metadata_run_order_matches_completed": metadata_order_matches_completed,
        "run_index_order_ok": run_index_order_ok,
        "hard_checks": hard_checks,
        "all_hard_checks_pass": all(hard_checks.values()),
        "mismatch_counts": {
            key: len(value) for key, value in mismatch_summary.items()
        },
    }
    write_json(
        output_dir / "manifest_array_alignment.json",
        manifest_alignment,
    )

    hard_pass = all(hard_checks.values())
    verdict = (
        "V4_A0_2_MANIFEST_ARRAY_ALIGNMENT_PASS"
        if hard_pass
        else "V4_A0_2_MANIFEST_ARRAY_ALIGNMENT_FAIL"
    )

    report_lines = [
        "# V4-A0.2 — Manifest-to-Array Alignment",
        "",
        f"- Generated: `{utc_now()}`",
        f"- Manifest: `{manifest_path}`",
        f"- Manifest SHA-256: `{sha256_file(manifest_path)}`",
        f"- Verdict: **{verdict}**",
        "",
        "## Hard checks",
        "",
    ]
    for name, passed in hard_checks.items():
        report_lines.append(f"- [{'PASS' if passed else 'FAIL'}] `{name}`")

    report_lines += [
        "",
        "## Optional metadata availability",
        "",
    ]
    for name, coverage in optional_coverage.items():
        report_lines.append(
            f"- `{name}`: present={coverage['column_present']}, "
            f"nonempty_rows={coverage['nonempty_rows']}"
        )

    report_lines += [
        "",
        "## Safety boundary",
        "",
        "- Model inference performed: **False**",
        "- Model training performed: **False**",
        "- Threshold selection performed: **False**",
        "- Dataset modified: **False**",
        "- Manifest modified: **False**",
        "",
        "A pass authorizes only V4-A0.3 background-family leakage analysis.",
        "It does not authorize model training.",
        "",
    ]

    report_path = output_dir / "v4_a0_2_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    stage_manifest = {
        "stage": "V4-A0.2",
        "script_version": SCRIPT_VERSION,
        "started_utc": started,
        "completed_utc": utc_now(),
        "dataset": str(data_dir),
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "hard_checks": hard_checks,
        "hard_pass": hard_pass,
        "verdict": verdict,
        "model_inference_performed": False,
        "training_performed": False,
        "threshold_selection_performed": False,
        "validation_predictions_accessed": False,
        "test_predictions_accessed": False,
        "dataset_modified": False,
        "manifest_modified": False,
        "training_authorized": False,
        "next_authorized_stage": (
            "V4-A0.3 background-family leakage audit"
            if hard_pass
            else "Repair or explain V4-A0.2 alignment failures"
        ),
        "artifacts": [
            "manifest_discovery.txt",
            "manifest_schema.json",
            "manifest_run_order_alignment.csv",
            "manifest_metadata_code_maps.json",
            "manifest_optional_metadata_coverage.json",
            "manifest_array_alignment_mismatches.json",
            "manifest_array_alignment.json",
            "v4_a0_2_report.md",
        ],
    }
    stage_manifest_path = output_dir / "v4_a0_2_manifest.json"
    write_json(stage_manifest_path, stage_manifest)

    artifact_rows = []
    for artifact in sorted(output_dir.iterdir()):
        if artifact.is_file() and artifact.name != "v4_a0_2_artifact_hashes.csv":
            artifact_rows.append({
                "artifact": artifact.name,
                "size_bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            })

    hashes_path = output_dir / "v4_a0_2_artifact_hashes.csv"
    with hashes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["artifact", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(artifact_rows)

    print("=" * 88)
    print(verdict)
    print(f"manifest={manifest_path}")
    print(f"manifest_rows={len(rows)}")
    print(f"completed_runs={len(completed_runs)}")
    print(f"hard_checks_passed={sum(hard_checks.values())}/{len(hard_checks)}")
    print(f"graph_label_mismatches={len(graph_mismatches)}")
    print(f"attacker_count_mismatches={len(attacker_count_mismatches)}")
    print(f"attacker_mask_mismatches={len(attacker_mask_mismatches)}")
    print(f"attack_strength_mismatches={len(strength_mismatches)}")
    print("model_inference_performed=False")
    print("training_performed=False")
    print("dataset_modified=False")
    print("manifest_modified=False")
    print("training_authorized=False")
    print(f"stage_manifest={stage_manifest_path}")
    print(f"next_authorized_stage={stage_manifest['next_authorized_stage']}")
    print("=" * 88)

    if args.strict and not hard_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
