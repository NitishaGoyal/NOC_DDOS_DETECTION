#!/usr/bin/env python3
"""
Consolidated V4 pretraining diagnostic.

Stages consolidated:
- V4-A0.2: embedded metadata-to-array alignment
- V4-A0.3: background-family leakage audit
- V4-A0.4: matched-control and active-core shortcut audit
- V4-A0.5-lite: targeted feature-semantic shortcut audit
- V4-A0.6-lite: attack-kind and victim-metadata audit
- V4-A0.7: training-readiness closure

Safety: read-only memmaps; no model training/inference/thresholding; bounded x.npy sampling.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCRIPT_VERSION = "1.0.1"
A00_REQUIRED = ("v4_stage0_report.json", "v4_top_level_inventory.csv", "v4_numpy_headers.json")
A01_REQUIRED = (
    "v4_a0_1_manifest.json", "v4_run_level_summary.csv", "v4_run_boundary_audit.json",
    "v4_label_count_audit.json", "v4_split_count_audit.json",
    "v4_scenario_distribution_audit.json", "v4_edge_index_audit.json",
    "v4_chunk_feature_stats.json",
)
ARRAY_FILES = (
    "x.npy", "y_graph.npy", "y_node.npy", "run_index.npy", "split_id.npy",
    "profile_id.npy", "attack_kind_id.npy", "attacker_count.npy", "strength.npy",
)
PREFERRED_RUN_FIELDS = (
    "run_id", "split", "background_id", "matched_normal_run_id", "matched_benign_run_id",
    "pair_id", "scenario_family", "placement_family", "profile", "active_cores",
    "active_count", "seed", "attackers", "attacker_count", "strength", "attack_kind",
    "burst_ops", "pause_iters", "timing_family", "victims", "victim_count", "route_family",
)
FEATURE_NAMES = (
    "ifd_in_norm", "ifd_out_norm", "input_flit_count_norm", "output_flit_count_norm",
    "in_count_norm_local", "in_count_norm_north", "in_count_norm_east",
    "in_count_norm_south", "in_count_norm_west", "out_count_norm_local",
    "out_count_norm_north", "out_count_norm_east", "out_count_norm_south",
    "out_count_norm_west", "ifd_in_norm_local", "ifd_in_norm_north",
    "ifd_in_norm_east", "ifd_in_norm_south", "ifd_in_norm_west",
    "ifd_out_norm_local", "ifd_out_norm_north", "ifd_out_norm_east",
    "ifd_out_norm_south", "ifd_out_norm_west",
)
DIRECTIONAL_CHANNELS = {
    5: ("input", "count", "north", None), 6: ("input", "count", "east", None),
    7: ("input", "count", "south", None), 8: ("input", "count", "west", None),
    10: ("output", "count", "north", None), 11: ("output", "count", "east", None),
    12: ("output", "count", "south", None), 13: ("output", "count", "west", None),
    15: ("input", "ifd", "north", 5), 16: ("input", "ifd", "east", 6),
    17: ("input", "ifd", "south", 7), 18: ("input", "ifd", "west", 8),
    20: ("output", "ifd", "north", 10), 21: ("output", "ifd", "east", 11),
    22: ("output", "ifd", "south", 12), 23: ("output", "ifd", "west", 13),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def canonical_text(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text(value).lower())


def safe_int(value: Any) -> int | None:
    text = normalize_text(value)
    if text.lower() in {"", "none", "null", "na", "n/a", "nan"}:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def parse_int_set(value: Any) -> tuple[int, ...] | None:
    """Parse router-ID sets from metadata.

    Router IDs are non-negative. The V4 metadata commonly serializes sets as
    hyphen-delimited strings such as ``0-3-7-8``. Therefore a hyphen must be
    treated as a separator, not as a unary minus sign.
    """
    text = normalize_text(value)
    if text.lower() in {"", "none", "null", "na", "n/a", "[]", "()", "{}"}:
        return tuple()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            values = tuple(sorted(set(int(item) for item in parsed)))
            return values if all(item >= 0 for item in values) else None
        if isinstance(parsed, (int, float)):
            item = int(parsed)
            return (item,) if item >= 0 else None
    except (ValueError, SyntaxError, TypeError):
        pass

    # Metadata router sets use separators such as '-', ',', ';', '|', or spaces.
    # Extract unsigned integer tokens because negative router IDs are invalid.
    tokens = re.findall(r"\d+", text)
    return tuple(sorted(set(int(token) for token in tokens))) if tokens else None


def set_text(values: tuple[int, ...] | None) -> str:
    return "<UNPARSEABLE>" if values is None else ",".join(str(value) for value in values)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_ready(row.get(key, "")) for key in fieldnames})


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_files(root: Path, names: Iterable[str], stage: str) -> None:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"STOP: {stage} artifacts missing from {root}: {missing}")


def validate_prior_stages(a00_dir: Path, a01_dir: Path) -> dict[str, Any]:
    require_files(a00_dir, A00_REQUIRED, "A0.0")
    require_files(a01_dir, A01_REQUIRED, "A0.1")
    a00 = load_json(a00_dir / "v4_stage0_report.json")
    a01 = load_json(a01_dir / "v4_a0_1_manifest.json")
    a00_pass = (
        a00.get("status") == "PROVENANCE_FREEZE_AND_INVENTORY_COMPLETED"
        and all(bool(v) for v in a00.get("checks", {}).values())
        and a00.get("dataset_tree_modified") is False
    )
    a01_pass = (
        a01.get("hard_pass") is True
        and a01.get("verdict") == "V4_A0_1_MEMMAP_INTEGRITY_PASS"
        and a01.get("training_performed") is False
        and a01.get("model_inference_performed") is False
        and a01.get("dataset_tree_modified") is False
    )
    if not a00_pass:
        raise SystemExit("STOP: A0.0 did not pass.")
    if not a01_pass:
        raise SystemExit("STOP: A0.1 did not pass.")
    return {
        "a00_pass": True, "a01_pass": True,
        "a00_report_sha256": sha256_file(a00_dir / "v4_stage0_report.json"),
        "a01_manifest_sha256": sha256_file(a01_dir / "v4_a0_1_manifest.json"),
    }


def decode_code_map(metadata: Mapping[str, Any], logical_name: str) -> dict[str, int]:
    code_maps = metadata.get("code_maps")
    if not isinstance(code_maps, Mapping):
        return {}
    raw = None
    for key in (logical_name, f"{logical_name}_id", logical_name.replace("_kind", "")):
        if isinstance(code_maps.get(key), Mapping):
            raw = code_maps[key]
            break
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, value in raw.items():
        try:
            name = canonical_text(key)
            if logical_name == "attack_kind" and name == "":
                name = "none"
            result[name] = int(value)
            continue
        except (TypeError, ValueError):
            pass
        try:
            result[canonical_text(value)] = int(key)
        except (TypeError, ValueError):
            continue
    return result


def normalize_category(field: str, value: Any) -> str:
    text = canonical_text(value)
    return "none" if field == "attack_kind" and text == "" else text


def derive_category_mapping(names: Sequence[str], codes: Sequence[int]) -> dict[str, Any]:
    name_to_codes: dict[str, set[int]] = defaultdict(set)
    code_to_names: dict[int, set[str]] = defaultdict(set)
    for name, code in zip(names, codes):
        name_to_codes[name].add(int(code))
        code_to_names[int(code)].add(name)
    one_to_one = all(len(v) == 1 for v in name_to_codes.values()) and all(
        len(v) == 1 for v in code_to_names.values()
    )
    return {
        "one_to_one": one_to_one,
        "derived_name_to_code": {k: next(iter(v)) for k, v in name_to_codes.items() if len(v) == 1},
        "name_to_codes": {k: sorted(v) for k, v in name_to_codes.items()},
        "code_to_names": {str(k): sorted(v) for k, v in code_to_names.items()},
    }


def graph_label_from_record(record: Mapping[str, Any]) -> int | None:
    for key in ("y_graph", "graph_label", "is_attack", "attack_label"):
        value = safe_int(record.get(key))
        if value in (0, 1):
            return value
    count = safe_int(record.get("attacker_count"))
    if count is not None:
        return int(count > 0)
    attackers = parse_int_set(record.get("attackers"))
    if attackers is not None:
        return int(len(attackers) > 0)
    run_id = normalize_text(record.get("run_id"))
    return 0 if run_id.startswith("V4-N-") else 1 if run_id.startswith("V4-A-") else None


def direct_export_embedded_manifest(runs: Sequence[Mapping[str, Any]], path: Path) -> list[str]:
    fields: set[str] = set()
    for record in runs:
        fields.update(str(key) for key in record.keys())
    fieldnames = [field for field in PREFERRED_RUN_FIELDS if field in fields]
    fieldnames.extend(sorted(fields - set(fieldnames)))
    write_csv(path, runs, fieldnames)
    return fieldnames


@dataclass
class RunRecord:
    position: int
    run_id: str
    raw: Mapping[str, Any]
    split: str
    graph_label: int | None
    background_id: str
    matched_normal_run_id: str
    pair_id: str
    scenario_family: str
    placement_family: str
    profile: str
    active_cores: tuple[int, ...] | None
    seed: str
    attackers: tuple[int, ...] | None
    attacker_count: int | None
    strength: int | None
    attack_kind: str
    burst_ops: str
    pause_iters: str
    timing_family: str
    victims: tuple[int, ...] | None
    victim_count: int | None
    route_family: str


def build_run_record(position: int, raw: Mapping[str, Any]) -> RunRecord:
    return RunRecord(
        position=position,
        run_id=normalize_text(raw.get("run_id")),
        raw=raw,
        split=normalize_category("split", raw.get("split")),
        graph_label=graph_label_from_record(raw),
        background_id=normalize_text(raw.get("background_id")),
        matched_normal_run_id=normalize_text(raw.get("matched_normal_run_id", raw.get("matched_benign_run_id", ""))),
        pair_id=normalize_text(raw.get("pair_id")),
        scenario_family=normalize_text(raw.get("scenario_family")),
        placement_family=normalize_text(raw.get("placement_family")),
        profile=normalize_category("profile", raw.get("profile")),
        active_cores=parse_int_set(raw.get("active_cores")),
        seed=normalize_text(raw.get("seed")),
        attackers=parse_int_set(raw.get("attackers")),
        attacker_count=safe_int(raw.get("attacker_count")),
        strength=safe_int(raw.get("strength")),
        attack_kind=normalize_category("attack_kind", raw.get("attack_kind")),
        burst_ops=normalize_text(raw.get("burst_ops")),
        pause_iters=normalize_text(raw.get("pause_iters")),
        timing_family=normalize_text(raw.get("timing_family")),
        victims=parse_int_set(raw.get("victims")),
        victim_count=safe_int(raw.get("victim_count")),
        route_family=normalize_text(raw.get("route_family")),
    )


def metadata_array_alignment(metadata, runs, completed_runs, arrays, expected_runs, samples_per_run, expected_nodes):
    first_indices = np.arange(expected_runs, dtype=np.int64) * samples_per_run
    run_ids = [r.run_id for r in runs]
    duplicate_ids = sorted(k for k, v in Counter(run_ids).items() if v > 1)
    completed_duplicates = sorted(k for k, v in Counter(completed_runs).items() if v > 1)
    mismatches: list[dict[str, Any]] = []
    category_names: dict[str, list[str]] = defaultdict(list)
    category_codes: dict[str, list[int]] = defaultdict(list)
    for r, sc, pc, kc in zip(
        runs,
        np.asarray(arrays["split_id.npy"][first_indices], dtype=np.int64),
        np.asarray(arrays["profile_id.npy"][first_indices], dtype=np.int64),
        np.asarray(arrays["attack_kind_id.npy"][first_indices], dtype=np.int64),
    ):
        category_names["split"].append(r.split); category_codes["split"].append(int(sc))
        category_names["profile"].append(r.profile); category_codes["profile"].append(int(pc))
        category_names["attack_kind"].append(r.attack_kind); category_codes["attack_kind"].append(int(kc))
    category_audit = {}; mappings = {}
    for field in ("split", "profile", "attack_kind"):
        metadata_map = decode_code_map(metadata, field)
        derived = derive_category_mapping(category_names[field], category_codes[field])
        mapping = dict(derived["derived_name_to_code"])
        mapping.update(metadata_map)
        mappings[field] = mapping
        bad = [i for i, (name, code) in enumerate(zip(category_names[field], category_codes[field]))
               if mapping.get(name) is None or int(mapping[name]) != int(code)]
        category_audit[field] = {
            "metadata_code_map": metadata_map, "derived_mapping": derived,
            "mapping_source": "metadata.code_maps" if metadata_map else "observed_one_to_one",
            "mismatch_count": len(bad), "mismatch_positions": bad[:100],
            "pass": len(bad) == 0 and (bool(metadata_map) or derived["one_to_one"]),
        }
    for pos, r in enumerate(runs):
        if pos >= len(completed_runs):
            mismatches.append({"run_position": pos, "run_id": r.run_id, "check": "completed_run_missing", "expected": r.run_id, "observed": ""})
            continue
        idx = pos * samples_per_run
        array_attackers = tuple(int(v) for v in np.flatnonzero(np.asarray(arrays["y_node.npy"][idx]) == 1.0).tolist())
        checks = [
            ("run_order", r.run_id, completed_runs[pos]),
            ("run_index", pos, int(arrays["run_index.npy"][idx])),
            ("graph_label", r.graph_label, int(arrays["y_graph.npy"][idx])),
            ("attacker_count", r.attacker_count, int(arrays["attacker_count.npy"][idx])),
            ("attacker_mask", set_text(r.attackers), set_text(array_attackers)),
        ]
        if int(arrays["y_graph.npy"][idx]) == 1:
            checks.append(("strength", r.strength, int(arrays["strength.npy"][idx])))
        for check, expected, observed in checks:
            if expected != observed:
                mismatches.append({"run_position": pos, "run_id": r.run_id, "check": check, "expected": expected, "observed": observed})
        for field, observed_code in (("split", int(arrays["split_id.npy"][idx])), ("profile", int(arrays["profile_id.npy"][idx])), ("attack_kind", int(arrays["attack_kind_id.npy"][idx]))):
            expected_code = mappings[field].get(getattr(r, field))
            if expected_code is None or int(expected_code) != observed_code:
                mismatches.append({"run_position": pos, "run_id": r.run_id, "check": f"{field}_code", "expected": expected_code, "observed": observed_code})
        for field, values in (("attackers", r.attackers), ("active_cores", r.active_cores), ("victims", r.victims)):
            if values is not None and any(v < 0 or v >= expected_nodes for v in values):
                mismatches.append({"run_position": pos, "run_id": r.run_id, "check": f"{field}_router_range", "expected": f"0..{expected_nodes - 1}", "observed": set_text(values)})
    checks = {
        "embedded_run_count_matches_expected": len(runs) == expected_runs,
        "completed_run_count_matches_expected": len(completed_runs) == expected_runs,
        "embedded_run_ids_unique": len(duplicate_ids) == 0,
        "completed_run_ids_unique": len(completed_duplicates) == 0,
        "embedded_and_completed_run_sets_identical": set(run_ids) == set(completed_runs),
        "embedded_and_completed_run_order_identical": run_ids == list(completed_runs),
        "run_index_sequence_valid": bool(np.array_equal(np.asarray(arrays["run_index.npy"][first_indices], dtype=np.int64), np.arange(expected_runs, dtype=np.int64))),
        "split_code_alignment": category_audit["split"]["pass"],
        "profile_code_alignment": category_audit["profile"]["pass"],
        "attack_kind_code_alignment": category_audit["attack_kind"]["pass"],
        "all_metadata_array_checks_match": len(mismatches) == 0,
    }
    summary = {
        "source": 'metadata.json["runs"]', "manual_reconstruction": False,
        "inference_from_x_npy": False, "original_external_csv_recovered": False,
        "run_count": len(runs), "completed_run_count": len(completed_runs),
        "duplicate_embedded_run_ids": duplicate_ids, "duplicate_completed_run_ids": completed_duplicates,
        "checks": checks, "category_alignment": category_audit,
        "mismatch_count": len(mismatches), "hard_pass": all(checks.values()),
    }
    return summary, mismatches


def composite_key(record: RunRecord, fields: Sequence[str]) -> str:
    parts = []
    for field in fields:
        value = getattr(record, field)
        parts.append(f"{field}={set_text(value) if isinstance(value, tuple) or value is None else normalize_text(value)}")
    return "|".join(parts)


def group_split_audit(runs, group_type, key_function, skip_empty=True):
    groups = defaultdict(list)
    for record in runs:
        key = normalize_text(key_function(record))
        if skip_empty and key == "":
            continue
        groups[key].append(record)
    summary = []; failures = []
    for key, members in sorted(groups.items()):
        splits = sorted(set(r.split for r in members))
        row = {"group_type": group_type, "group_key": key, "run_count": len(members),
               "splits": ",".join(splits), "split_count": len(splits),
               "cross_split": len(splits) > 1, "run_ids": ";".join(r.run_id for r in members)}
        summary.append(row)
        if row["cross_split"]:
            failures.append(row)
    return summary, failures


def family_leakage_audit(runs):
    near_fields = ("background_id", "profile", "placement_family", "active_cores", "seed", "timing_family", "route_family", "victims")
    members = [{
        "run_id": r.run_id, "split": r.split, "graph_label": r.graph_label,
        "background_id": r.background_id, "pair_id": r.pair_id,
        "matched_normal_run_id": r.matched_normal_run_id,
        "scenario_family": r.scenario_family, "placement_family": r.placement_family,
        "profile": r.profile, "active_cores": set_text(r.active_cores), "seed": r.seed,
        "timing_family": r.timing_family, "route_family": r.route_family,
        "victims": set_text(r.victims), "near_duplicate_signature": composite_key(r, near_fields),
    } for r in runs]
    bg_s, bg_f = group_split_audit(runs, "background_id", lambda r: r.background_id)
    pair_s, pair_f = group_split_audit(runs, "pair_id", lambda r: r.pair_id)
    scen_s, scen_f = group_split_audit(runs, "scenario_family", lambda r: r.scenario_family)
    near_s, near_f = group_split_audit(runs, "near_duplicate_signature", lambda r: composite_key(r, near_fields))
    by_id = {r.run_id: r for r in runs}
    matched_f = []
    for r in runs:
        if r.graph_label == 1 and r.matched_normal_run_id:
            m = by_id.get(r.matched_normal_run_id)
            if m is not None and m.split != r.split:
                matched_f.append({"attack_run_id": r.run_id, "attack_split": r.split,
                                  "matched_normal_run_id": m.run_id, "matched_normal_split": m.split,
                                  "background_id": r.background_id})
    counts = {"background_family_cross_split_count": len(bg_f), "pair_cross_split_count": len(pair_f),
              "matched_normal_cross_split_count": len(matched_f),
              "near_duplicate_signature_cross_split_count": len(near_f)}
    return {"hard_counts": counts, "hard_pass": all(v == 0 for v in counts.values()),
            "scenario_family_cross_split_count_report_only": len(scen_f), "near_duplicate_fields": list(near_fields)}, {
        "family_members": members, "family_split_summary": bg_s + pair_s + scen_s + near_s,
        "family_leakage_failures": bg_f + pair_f, "matched_pair_split_failures": matched_f,
        "near_duplicate_split_failures": near_f,
    }


def matched_control_audit(runs):
    by_id = {r.run_id: r for r in runs}; audit = []; failures = []; shortcuts = []
    attack_runs = [r for r in runs if r.graph_label == 1]
    for a in attack_runs:
        m = by_id.get(a.matched_normal_run_id) if a.matched_normal_run_id else None
        exists = m is not None; is_normal = bool(exists and m.graph_label == 0)
        cmp = {f: bool(exists and getattr(a, f) == getattr(m, f)) for f in
               ("background_id", "profile", "placement_family", "seed", "active_cores", "route_family", "victims")}
        split_match = bool(exists and a.split == m.split)
        subset = bool(exists and a.attackers is not None and m.active_cores is not None and set(a.attackers).issubset(set(m.active_cores)))
        reasons = []
        if not a.matched_normal_run_id: reasons.append("matched_normal_id_missing")
        elif not exists: reasons.append("matched_normal_not_found")
        else:
            if not is_normal: reasons.append("matched_reference_not_normal")
            if not split_match: reasons.append("split_mismatch")
            for f in ("background_id", "profile", "placement_family", "seed", "active_cores"):
                if not cmp[f]: reasons.append(f"{f}_mismatch")
        shortcut = exists and not subset
        row = {
            "attack_run_id": a.run_id, "attack_split": a.split, "matched_normal_run_id": a.matched_normal_run_id,
            "matched_exists": exists, "matched_is_normal": is_normal, "matched_split": m.split if exists else "",
            "split_match": split_match, "background_match": cmp["background_id"], "profile_match": cmp["profile"],
            "placement_family_match": cmp["placement_family"], "seed_match": cmp["seed"],
            "active_cores_match": cmp["active_cores"], "route_family_match": cmp["route_family"],
            "victims_match": cmp["victims"], "attack_active_cores": set_text(a.active_cores),
            "normal_active_cores": set_text(m.active_cores) if exists else "", "attackers": set_text(a.attackers),
            "attackers_subset_of_normal_active_cores": subset, "shortcut_risk": shortcut,
            "hard_failure": bool(reasons), "hard_failure_reasons": ",".join(reasons),
        }
        audit.append(row)
        if reasons: failures.append(row)
        if shortcut: shortcuts.append(row)
    semantic = sum(1 for row in failures if any(token in row["hard_failure_reasons"] for token in
                   ("background_id_mismatch", "profile_mismatch", "placement_family_mismatch", "seed_mismatch", "active_cores_mismatch", "matched_reference_not_normal")))
    summary = {
        "attack_run_count": len(attack_runs), "valid_matched_control_count": len(attack_runs) - len(failures),
        "hard_failure_count": len(failures), "semantic_control_mismatch_count": semantic,
        "reference_or_split_failure_count": len(failures) - semantic,
        "active_core_shortcut_case_count": len(shortcuts),
        "active_core_shortcut_rate": len(shortcuts) / len(attack_runs) if attack_runs else 0.0,
        "hard_pass": len(failures) == 0, "shortcut_free": len(shortcuts) == 0,
        "idle_background_note": "Idle normals are not failures by themselves; they are shortcut risks only when used as attack controls whose attackers are absent from the normal active set.",
    }
    return summary, {"matched_control_audit": audit, "matched_control_failures": failures, "active_core_shortcut_cases": shortcuts}


def router_direction_valid(router, direction, mesh_cols=4):
    row, col = divmod(router, mesh_cols)
    return {"north": row > 0, "east": col < mesh_cols - 1, "south": row < mesh_cols - 1, "west": col > 0}[direction]


class RateAccumulator:
    def __init__(self):
        self.observations = self.zero = self.one = 0; self.total = 0.0
    def add(self, values):
        a = np.asarray(values); self.observations += int(a.size)
        self.zero += int(np.count_nonzero(a == 0.0)); self.one += int(np.count_nonzero(a == 1.0))
        self.total += float(a.sum(dtype=np.float64))
    def row(self):
        d = self.observations or 1
        return {"observations": self.observations, "zero_count": self.zero, "one_count": self.one,
                "zero_rate": self.zero / d, "one_rate": self.one / d, "mean": self.total / d}


def deterministic_feature_indices(runs, samples_per_run, target):
    offsets = np.unique(np.linspace(0, samples_per_run - 1, num=min(target, samples_per_run), dtype=np.int64))
    indices = []; positions = []
    for r in runs:
        for offset in offsets:
            indices.append(r.position * samples_per_run + int(offset)); positions.append(r.position)
    return np.asarray(indices, dtype=np.int64), positions


def max_rate_difference(rows, group_field):
    grouped = defaultdict(list)
    for row in rows: grouped[(int(row["feature_index"]), int(row["router"]))].append(row)
    maximum = {"difference": 0.0, "feature_index": None, "router": None, "metric": None, "groups": None}
    for (feature, router), entries in grouped.items():
        groups = defaultdict(dict)
        for row in entries:
            groups[str(row[group_field])]["zero_rate"] = float(row["zero_rate"])
            groups[str(row[group_field])]["one_rate"] = float(row["one_rate"])
        names = sorted(groups)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                for metric in ("zero_rate", "one_rate"):
                    if metric in groups[left] and metric in groups[right]:
                        diff = abs(groups[left][metric] - groups[right][metric])
                        if diff > maximum["difference"]:
                            maximum = {"difference": diff, "feature_index": feature, "router": router, "metric": metric, "groups": [left, right]}
    return maximum


def feature_semantic_audit(data_dir, a01_dir, runs, arrays, samples_per_run, expected_nodes, sample_per_run, batch_size):
    a01_stats = load_json(a01_dir / "v4_chunk_feature_stats.json")
    x = arrays["x.npy"]
    sample_indices, run_positions = deterministic_feature_indices(runs, samples_per_run, sample_per_run)
    label_acc = defaultdict(RateAccumulator); split_acc = defaultdict(RateAccumulator); label_split_acc = defaultdict(RateAccumulator); port_acc = defaultdict(RateAccumulator)
    print(f"targeted_feature_scan_start samples={len(sample_indices)} batches={math.ceil(len(sample_indices)/batch_size)}", flush=True)
    for batch_no, start in enumerate(range(0, len(sample_indices), batch_size), 1):
        end = min(len(sample_indices), start + batch_size)
        batch = np.asarray(x[sample_indices[start:end]])
        for local, run_pos in enumerate(run_positions[start:end]):
            r = runs[run_pos]; sample = batch[local]; label = int(r.graph_label or 0)
            for router in range(expected_nodes):
                for feature in range(sample.shape[-1]):
                    values = sample[router, :, feature]
                    label_acc[(feature, router, label)].add(values)
                    split_acc[(feature, router, r.split)].add(values)
                    label_split_acc[(feature, router, label, r.split)].add(values)
                for feature, (_, measurement, direction, count_feature) in DIRECTIONAL_CHANNELS.items():
                    valid = router_direction_valid(router, direction); values = sample[router, :, feature]
                    if measurement == "ifd" and count_feature is not None:
                        counts = sample[router, :, count_feature]
                        if valid:
                            idle = counts == 0.0
                            if np.any(idle): port_acc[(feature, "valid_idle_port", direction, label)].add(values[idle])
                            if np.any(~idle): port_acc[(feature, "valid_active_port", direction, label)].add(values[~idle])
                        else: port_acc[(feature, "invalid_port", direction, label)].add(values)
                    else:
                        port_acc[(feature, "valid_port" if valid else "invalid_port", direction, label)].add(values)
        if batch_no == 1 or end == len(sample_indices) or batch_no % 10 == 0:
            print(f"targeted_feature_scan_progress samples={end}/{len(sample_indices)}", flush=True)
    rates = []
    for (feature, router, label), acc in sorted(label_acc.items()):
        row = {"group_type": "graph_label", "feature_index": feature, "feature_name": FEATURE_NAMES[feature], "router": router, "graph_label": label, "split": ""}; row.update(acc.row()); rates.append(row)
    for (feature, router, split), acc in sorted(split_acc.items()):
        row = {"group_type": "split", "feature_index": feature, "feature_name": FEATURE_NAMES[feature], "router": router, "graph_label": "", "split": split}; row.update(acc.row()); rates.append(row)
    for (feature, router, label, split), acc in sorted(label_split_acc.items()):
        row = {"group_type": "graph_label_split", "feature_index": feature, "feature_name": FEATURE_NAMES[feature], "router": router, "graph_label": label, "split": split}; row.update(acc.row()); rates.append(row)
    ports = []
    for (feature, status, direction, label), acc in sorted(port_acc.items()):
        side, measurement, _, count_feature = DIRECTIONAL_CHANNELS[feature]
        row = {"feature_index": feature, "feature_name": FEATURE_NAMES[feature], "traffic_side": side, "measurement": measurement,
               "direction": direction, "port_status": status, "graph_label": label,
               "corresponding_count_feature": "" if count_feature is None else count_feature}; row.update(acc.row()); ports.append(row)
    max_label = max_rate_difference([r for r in rates if r["group_type"] == "graph_label"], "graph_label")
    max_split = max_rate_difference([r for r in rates if r["group_type"] == "split"], "split")
    ambiguous_split_rows = [
        r for r in rates
        if r["group_type"] == "split" and int(r["feature_index"]) in range(14, 24)
    ]
    max_ambiguous_split = max_rate_difference(ambiguous_split_rows, "split")
    lookup = {(int(r["feature_index"]), r["port_status"], r["direction"], int(r["graph_label"])): r for r in ports}
    comparisons = []
    for feature, (_, measurement, direction, _) in DIRECTIONAL_CHANNELS.items():
        if measurement != "ifd": continue
        for label in (0, 1):
            invalid = lookup.get((feature, "invalid_port", direction, label)); idle = lookup.get((feature, "valid_idle_port", direction, label))
            if not invalid or not idle: continue
            zd = abs(float(invalid["zero_rate"]) - float(idle["zero_rate"])); od = abs(float(invalid["one_rate"]) - float(idle["one_rate"]))
            comparisons.append({"feature_index": feature, "feature_name": FEATURE_NAMES[feature], "direction": direction, "graph_label": label,
                                "invalid_zero_rate": invalid["zero_rate"], "valid_idle_zero_rate": idle["zero_rate"], "zero_rate_difference": zd,
                                "invalid_one_rate": invalid["one_rate"], "valid_idle_one_rate": idle["one_rate"], "one_rate_difference": od,
                                "near_identical_encoding": zd <= 0.02 and od <= 0.02})
    masks = {name: (data_dir / name).is_file() for name in ("valid_port_mask.npy", "flit_observed_mask.npy", "gap_observed_mask.npy", "ifd_clipped_mask.npy")}
    ifd_ones = sum(int(a01_stats["features"][i]["exact_one_count"]) for i in range(14, 24))
    ambiguity = any(c["near_identical_encoding"] for c in comparisons) or (not all(masks.values()) and ifd_ones > 0)

    invalid_label_differences = []
    invalid_grouped = defaultdict(dict)
    for row in ports:
        if row["port_status"] != "invalid_port":
            continue
        key = (int(row["feature_index"]), row["direction"])
        invalid_grouped[key][int(row["graph_label"])] = row
    for (feature, direction), by_label in sorted(invalid_grouped.items()):
        if 0 not in by_label or 1 not in by_label:
            continue
        for metric in ("zero_rate", "one_rate"):
            diff = abs(float(by_label[0][metric]) - float(by_label[1][metric]))
            invalid_label_differences.append({
                "feature_index": feature,
                "feature_name": FEATURE_NAMES[feature],
                "direction": direction,
                "metric": metric,
                "difference": diff,
                "normal_rate": by_label[0][metric],
                "attack_rate": by_label[1][metric],
            })
    max_invalid_label = max(
        invalid_label_differences,
        key=lambda row: row["difference"],
        default={"difference": 0.0, "feature_index": None, "feature_name": None, "direction": None, "metric": None},
    )

    invalid_label_diff = float(max_invalid_label["difference"])
    ambiguous_split_diff = float(max_ambiguous_split["difference"])
    # Desired attack-sensitive traffic features can differ strongly by class. The gate therefore
    # uses only invalid-port behavior and ambiguous IFD split dependence as shortcut evidence.
    if invalid_label_diff >= 0.90:
        verdict = "FEATURE_SHORTCUT_CONFIRMED"
    elif invalid_label_diff >= 0.20 or ambiguous_split_diff >= 0.20:
        verdict = "POTENTIAL_FEATURE_SHORTCUT_REQUIRES_REBUILD"
    elif ambiguity:
        verdict = "FEATURE_AMBIGUITY_PRESENT_NO_LABEL_SEPARATION"
    else:
        verdict = "NO_OBVIOUS_FEATURE_SHORTCUT"
    return {
        "targeted_sampling": {"samples_per_run": sample_per_run, "total_samples": len(sample_indices), "batch_size": batch_size, "selection": "deterministic evenly spaced windows within every run"},
        "a01_full_scan_reused": True, "a01_all_values_finite": a01_stats.get("all_values_finite"),
        "a01_all_values_in_0_1": a01_stats.get("all_values_in_documented_range_0_1"),
        "mask_files_present": masks, "ifd_exact_one_total_from_a01": ifd_ones,
        "invalid_vs_valid_idle_comparisons": comparisons, "invalid_idle_encoding_ambiguity_detected": ambiguity,
        "maximum_graph_label_rate_difference_all_features_report_only": max_label,
        "maximum_split_rate_difference_all_features_report_only": max_split,
        "maximum_ambiguous_ifd_split_rate_difference": max_ambiguous_split,
        "invalid_port_label_rate_differences": invalid_label_differences,
        "maximum_invalid_port_label_rate_difference": max_invalid_label,
        "maximum_graph_label_rate_difference": max_invalid_label,
        "maximum_split_rate_difference": max_ambiguous_split,
        "thresholds": {"confirmed_invalid_port_label_difference": 0.90, "potential_invalid_port_or_ambiguous_split_difference": 0.20, "near_identical_invalid_vs_idle_tolerance": 0.02},
        "verdict": verdict,
        "interpretation_note": "Global attack/normal feature differences are expected signal and are report-only. Shortcut gating uses invalid-port behavior and ambiguous IFD split dependence; no classifier was trained."
    }, rates, ports


def attack_kind_and_victim_audit(runs, expected_nodes):
    attacks = [r for r in runs if r.graph_label == 1]; groups = defaultdict(list)
    for r in attacks: groups[(normalize_text(r.strength), r.burst_ops, r.pause_iters, r.timing_family)].append(r)
    signatures = []; aliases = []
    for signature, members in sorted(groups.items()):
        kinds = sorted(set(m.attack_kind for m in members))
        row = {"strength": signature[0], "burst_ops": signature[1], "pause_iters": signature[2], "timing_family": signature[3],
               "run_count": len(members), "attack_kinds": ",".join(kinds), "attack_kind_count": len(kinds),
               "is_cross_label_alias": len(kinds) > 1, "run_ids": ";".join(m.run_id for m in members)}
        signatures.append(row)
        if row["is_cross_label_alias"]: aliases.append(row)
    victim_missing = victim_count_mismatch = victim_invalid = route_missing = 0
    for r in attacks:
        if r.victims is None: victim_missing += 1
        else:
            if r.victim_count is not None and len(r.victims) != r.victim_count: victim_count_mismatch += 1
            if any(v < 0 or v >= expected_nodes for v in r.victims): victim_invalid += 1
        if r.route_family == "": route_missing += 1
    kind = {"attack_run_count": len(attacks), "attack_kind_counts": dict(sorted(Counter(r.attack_kind for r in attacks).items())),
            "timing_family_counts": dict(sorted(Counter(r.timing_family for r in attacks).items())),
            "route_family_counts": dict(sorted(Counter(r.route_family for r in attacks).items())),
            "burst_ops_nonempty": sum(r.burst_ops != "" for r in attacks), "pause_iters_nonempty": sum(r.pause_iters != "" for r in attacks),
            "implementation_signature_count": len(signatures), "cross_label_alias_count": len(aliases),
            "internal_attack_metadata_consistent": all(r.attack_kind != "" and r.timing_family != "" for r in attacks)}
    victim = {"attack_run_count": len(attacks), "victim_metadata_unparseable_count": victim_missing,
              "victim_count_mismatch_count": victim_count_mismatch, "invalid_victim_router_count": victim_invalid,
              "route_family_missing_count": route_missing, "victim_metadata_blocks_attacker_only_a1": False,
              "victim_metadata_blocks_victim_head": victim_missing > 0 or victim_count_mismatch > 0 or victim_invalid > 0}
    return kind, signatures, aliases, victim


def choose_final_verdict(alignment, leakage, controls, feature, attack_kind, victim):
    remediation = []; limitations = []
    if not alignment["hard_pass"]:
        verdict = "V4_REQUIRES_SPLIT_OR_METADATA_REBUILD"; remediation.append("Repair embedded metadata or metadata arrays so every run aligns exactly.")
    elif not leakage["hard_pass"]:
        verdict = "V4_REQUIRES_SPLIT_OR_METADATA_REBUILD"; remediation.append("Rebuild splits at background-family level; do not edit split_id.npy in place.")
    elif not controls["hard_pass"]:
        if controls["semantic_control_mismatch_count"] > 0:
            verdict = "V4_REQUIRES_GEM5_REGENERATION"; remediation.append("Regenerate affected matched controls with identical legitimate backgrounds.")
        else:
            verdict = "V4_REQUIRES_SPLIT_OR_METADATA_REBUILD"; remediation.append("Repair matched-normal references or split assignments.")
    elif feature["verdict"] == "FEATURE_SHORTCUT_CONFIRMED":
        verdict = "V4_REQUIRES_FEATURE_REBUILD"; remediation.append("Rebuild features with explicit port/observation/clipping masks.")
    elif not controls["shortcut_free"] or feature["verdict"] == "POTENTIAL_FEATURE_SHORTCUT_REQUIRES_REBUILD":
        verdict = "V4_READY_ONLY_FOR_DIAGNOSTIC_BASELINE"; limitations.append("Potential active-core or feature shortcut prevents formal scientific claims.")
    elif feature["verdict"] == "FEATURE_AMBIGUITY_PRESENT_NO_LABEL_SEPARATION":
        verdict = "V4_READY_FOR_A1_WITH_DOCUMENTED_LIMITATIONS"; limitations.append("IFD/invalid-port semantics remain ambiguous, but targeted audit found no large label separation.")
    elif not attack_kind["internal_attack_metadata_consistent"]:
        verdict = "V4_READY_FOR_A1_WITH_DOCUMENTED_LIMITATIONS"; limitations.append("Attack-kind metadata is incomplete; do not publish per-kind claims.")
    else: verdict = "V4_READY_FOR_A1_TRAINING"
    if attack_kind["cross_label_alias_count"] > 0: limitations.append("Some attack-kind labels share implementation signatures; merge or qualify per-kind reporting.")
    if victim["victim_metadata_blocks_victim_head"]: limitations.append("Victim metadata is incomplete or inconsistent; victim-head training remains blocked.")
    authorized = verdict in {"V4_READY_FOR_A1_TRAINING", "V4_READY_FOR_A1_WITH_DOCUMENTED_LIMITATIONS", "V4_READY_ONLY_FOR_DIAGNOSTIC_BASELINE"}
    level = "formal_v4_a1" if verdict == "V4_READY_FOR_A1_TRAINING" else "formal_v4_a1_with_documented_limitations" if verdict == "V4_READY_FOR_A1_WITH_DOCUMENTED_LIMITATIONS" else "diagnostic_only" if verdict == "V4_READY_ONLY_FOR_DIAGNOSTIC_BASELINE" else "none"
    allowed = [] if not authorized else ["Create the frozen V4-A1 Conv1D-TemporalGCN memmap training script.", "Run loader, one-batch, and short training preflights.", "Train seed 7 within the stated authorization level."]
    if authorized and level != "diagnostic_only": allowed += ["Select thresholds on validation only.", "Transfer frozen thresholds once to the development blind test."]
    blocked = [] if authorized else ["Start V4-A1 training before required remediation."]
    if level == "diagnostic_only": blocked += ["Treat diagnostic metrics as formal evidence.", "Use the development test for architecture or threshold tuning."]
    blocked += ["Call the current test split an independent publication holdout.", "Train a victim head when victim metadata is blocked."]
    return {"verdict": verdict, "training_authorized": authorized, "authorization_level": level,
            "allowed_tasks": allowed, "blocked_tasks": blocked, "required_remediation": remediation,
            "documented_limitations": limitations, "development_test_status": "development blind test",
            "publication_holdout_status": "not established; independent family pool required"}


def make_report(decision, prior, alignment, leakage, controls, feature, attack_kind, victim, data_dir):
    lines = ["# V4 Consolidated Pretraining Diagnostic", "", f"- Generated: `{utc_now()}`", f"- Script version: `{SCRIPT_VERSION}`",
             f"- Dataset: `{data_dir}`", f"- Final verdict: **{decision['verdict']}**", f"- Training authorized: **{decision['training_authorized']}**",
             f"- Authorization level: **{decision['authorization_level']}**", "", "## Frozen prerequisite stages", "",
             f"- A0.0 pass: `{prior['a00_pass']}`", f"- A0.1 pass: `{prior['a01_pass']}`", "", "## Embedded metadata alignment", "",
             f"- Source: `{alignment['source']}`", f"- Run count: `{alignment['run_count']}`", f"- Mismatches: `{alignment['mismatch_count']}`", f"- Hard pass: `{alignment['hard_pass']}`", "", "## Family leakage", ""]
    lines += [f"- `{k}`: `{v}`" for k, v in leakage["hard_counts"].items()]
    lines += [f"- Hard pass: `{leakage['hard_pass']}`", "", "## Matched controls and active-core shortcut", "",
              f"- Attack runs: `{controls['attack_run_count']}`", f"- Hard control failures: `{controls['hard_failure_count']}`",
              f"- Shortcut cases: `{controls['active_core_shortcut_case_count']}`", f"- Shortcut rate: `{controls['active_core_shortcut_rate']:.6f}`", "",
              "## Feature-semantic audit", "", f"- Verdict: **{feature['verdict']}**",
              f"- Maximum graph-label rate difference: `{feature['maximum_graph_label_rate_difference']['difference']:.6f}`",
              f"- Maximum split rate difference: `{feature['maximum_split_rate_difference']['difference']:.6f}`", "",
              "## Attack-kind and victim metadata", "", f"- Attack-kind implementation aliases: `{attack_kind['cross_label_alias_count']}`",
              f"- Victim metadata blocks attacker-only A1: `{victim['victim_metadata_blocks_attacker_only_a1']}`",
              f"- Victim metadata blocks victim head: `{victim['victim_metadata_blocks_victim_head']}`", "", "## Authorization", "",
              f"- Development test status: `{decision['development_test_status']}`", f"- Publication holdout status: `{decision['publication_holdout_status']}`", "", "### Allowed tasks", ""]
    lines += [f"- {x}" for x in decision["allowed_tasks"]] or ["- None until remediation."]
    lines += ["", "### Blocked tasks", ""] + [f"- {x}" for x in decision["blocked_tasks"]]
    lines += ["", "### Required remediation", ""] + ([f"- {x}" for x in decision["required_remediation"]] or ["- None before the authorized V4-A1 scope."])
    lines += ["", "### Documented limitations", ""] + ([f"- {x}" for x in decision["documented_limitations"]] or ["- None identified by this diagnostic."])
    lines += ["", "## Safety boundary", "", "- Model training performed: **False**", "- Model inference performed: **False**",
              "- Threshold selection performed: **False**", "- Test predictions accessed: **False**", "- Full x.npy loaded into RAM: **False**",
              "- Dataset directory modified: **False**", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--a00-dir", required=True, type=Path)
    parser.add_argument("--a01-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-runs", type=int, default=908)
    parser.add_argument("--expected-samples-per-run", type=int, default=3293)
    parser.add_argument("--expected-nodes", type=int, default=16)
    parser.add_argument("--feature-samples-per-run", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve(); a00_dir = args.a00_dir.expanduser().resolve(); a01_dir = args.a01_dir.expanduser().resolve(); output_dir = args.output_dir.expanduser().resolve()
    if not data_dir.is_dir(): raise SystemExit(f"STOP: data directory does not exist: {data_dir}")
    if output_dir.exists() and any(output_dir.iterdir()): raise SystemExit(f"STOP: output directory already non-empty: {output_dir}")
    for name in ARRAY_FILES + ("metadata.json", "completed_runs.txt"):
        if not (data_dir / name).is_file(): raise SystemExit(f"STOP: required dataset file missing: {data_dir / name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_before = {"size": data_dir.stat().st_size, "mtime_ns": data_dir.stat().st_mtime_ns, "mode": data_dir.stat().st_mode}
    started = utc_now()
    print("V4 CONSOLIDATED PRETRAINING DIAGNOSTIC START", flush=True); print(f"data_dir={data_dir}", flush=True); print(f"output_dir={output_dir}", flush=True)
    print("[1/6] validating frozen A0.0 and A0.1 evidence", flush=True)
    prior = validate_prior_stages(a00_dir, a01_dir)
    metadata = load_json(data_dir / "metadata.json"); raw_runs = metadata.get("runs") if isinstance(metadata, Mapping) else None
    if not isinstance(raw_runs, list) or not all(isinstance(r, Mapping) for r in raw_runs): raise SystemExit('STOP: metadata.json["runs"] is missing or malformed.')
    runs = [build_run_record(i, r) for i, r in enumerate(raw_runs)]
    completed_runs = [line.strip() for line in (data_dir / "completed_runs.txt").read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    arrays = {name: np.load(data_dir / name, mmap_mode="r", allow_pickle=False) for name in ARRAY_FILES}
    print("[2/6] aligning embedded metadata with memmap arrays", flush=True)
    export_path = output_dir / "v4_embedded_metadata_manifest.csv"; exported_fields = direct_export_embedded_manifest(raw_runs, export_path)
    alignment, alignment_mismatches = metadata_array_alignment(metadata, runs, completed_runs, arrays, args.expected_runs, args.expected_samples_per_run, args.expected_nodes)
    alignment["embedded_manifest_export"] = {"path": str(export_path), "field_count": len(exported_fields), "fields": exported_fields,
        "sha256": sha256_file(export_path), "source": 'metadata.json["runs"]', "transformation": "direct field-preserving tabular export",
        "manual_reconstruction": False, "inference_from_model_inputs": False, "original_external_csv_recovered": False}
    write_json(output_dir / "metadata_array_alignment.json", alignment)
    write_csv(output_dir / "metadata_array_mismatches.csv", alignment_mismatches, ("run_position", "run_id", "check", "expected", "observed"))
    print("[3/6] auditing family and matched-pair leakage", flush=True)
    leakage, ft = family_leakage_audit(runs)
    write_csv(output_dir / "family_members.csv", ft["family_members"], ("run_id", "split", "graph_label", "background_id", "pair_id", "matched_normal_run_id", "scenario_family", "placement_family", "profile", "active_cores", "seed", "timing_family", "route_family", "victims", "near_duplicate_signature"))
    group_fields = ("group_type", "group_key", "run_count", "splits", "split_count", "cross_split", "run_ids")
    write_csv(output_dir / "family_split_summary.csv", ft["family_split_summary"], group_fields)
    write_csv(output_dir / "family_leakage_failures.csv", ft["family_leakage_failures"], group_fields)
    write_csv(output_dir / "matched_pair_split_failures.csv", ft["matched_pair_split_failures"], ("attack_run_id", "attack_split", "matched_normal_run_id", "matched_normal_split", "background_id"))
    write_csv(output_dir / "near_duplicate_split_failures.csv", ft["near_duplicate_split_failures"], group_fields)
    write_json(output_dir / "family_leakage_summary.json", leakage)
    print("[4/6] auditing matched controls and active-core shortcuts", flush=True)
    controls, ct = matched_control_audit(runs)
    control_fields = ("attack_run_id", "attack_split", "matched_normal_run_id", "matched_exists", "matched_is_normal", "matched_split", "split_match", "background_match", "profile_match", "placement_family_match", "seed_match", "active_cores_match", "route_family_match", "victims_match", "attack_active_cores", "normal_active_cores", "attackers", "attackers_subset_of_normal_active_cores", "shortcut_risk", "hard_failure", "hard_failure_reasons")
    write_csv(output_dir / "matched_control_audit.csv", ct["matched_control_audit"], control_fields)
    write_csv(output_dir / "matched_control_failures.csv", ct["matched_control_failures"], control_fields)
    write_csv(output_dir / "active_core_shortcut_cases.csv", ct["active_core_shortcut_cases"], control_fields)
    write_json(output_dir / "active_core_shortcut_summary.json", controls)
    print("[5/6] running targeted feature-semantic audit", flush=True)
    feature, rates, ports = feature_semantic_audit(data_dir, a01_dir, runs, arrays, args.expected_samples_per_run, args.expected_nodes, args.feature_samples_per_run, args.feature_batch_size)
    write_json(output_dir / "feature_semantic_summary.json", feature)
    write_csv(output_dir / "feature_router_label_rates.csv", rates, ("group_type", "feature_index", "feature_name", "router", "graph_label", "split", "observations", "zero_count", "one_count", "zero_rate", "one_rate", "mean"))
    write_csv(output_dir / "invalid_vs_valid_port_rates.csv", ports, ("feature_index", "feature_name", "traffic_side", "measurement", "direction", "port_status", "graph_label", "corresponding_count_feature", "observations", "zero_count", "one_count", "zero_rate", "one_rate", "mean"))
    print("[6/6] auditing attack-kind/victim metadata and closing readiness", flush=True)
    attack_kind, signatures, aliases, victim = attack_kind_and_victim_audit(runs, args.expected_nodes)
    sig_fields = ("strength", "burst_ops", "pause_iters", "timing_family", "run_count", "attack_kinds", "attack_kind_count", "is_cross_label_alias", "run_ids")
    write_csv(output_dir / "attack_kind_signatures.csv", signatures, sig_fields); write_csv(output_dir / "attack_kind_aliases.csv", aliases, sig_fields)
    write_json(output_dir / "attack_kind_metadata_summary.json", attack_kind); write_json(output_dir / "victim_metadata_summary.json", victim)
    source_after = {"size": data_dir.stat().st_size, "mtime_ns": data_dir.stat().st_mtime_ns, "mode": data_dir.stat().st_mode}; unchanged = source_before == source_after
    decision = choose_final_verdict(alignment, leakage, controls, feature, attack_kind, victim)
    decision.update({"stage": "V4-A0.2-to-A0.7", "script_version": SCRIPT_VERSION, "started_utc": started, "completed_utc": utc_now(), "dataset": str(data_dir), "a00_dir": str(a00_dir), "a01_dir": str(a01_dir), "output_dir": str(output_dir), "prior_stages": prior,
        "embedded_metadata_alignment": {"hard_pass": alignment["hard_pass"], "mismatch_count": alignment["mismatch_count"]}, "family_leakage": leakage, "matched_controls": controls,
        "feature_semantics": {"verdict": feature["verdict"], "maximum_graph_label_rate_difference": feature["maximum_graph_label_rate_difference"], "maximum_split_rate_difference": feature["maximum_split_rate_difference"]},
        "attack_kind_metadata": attack_kind, "victim_metadata": victim, "model_training_performed": False, "model_inference_performed": False, "threshold_selection_performed": False,
        "test_predictions_accessed": False, "complete_x_loaded_into_ram": False, "dataset_directory_modified": not unchanged, "embedded_manifest_export_sha256": sha256_file(export_path)})
    if not unchanged:
        decision["verdict"] = "V4_NOT_SAFE_TO_TRAIN"; decision["training_authorized"] = False; decision["authorization_level"] = "none"; decision["required_remediation"].append("Investigate unexpected dataset-root metadata change during read-only audit.")
    decision_path = output_dir / "v4_pretraining_diagnostic_decision.json"; write_json(decision_path, decision)
    (output_dir / "v4_pretraining_diagnostic_report.md").write_text(make_report(decision, prior, alignment, leakage, controls, feature, attack_kind, victim, data_dir) + "\n", encoding="utf-8")
    artifact_rows = [{"artifact": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(output_dir.iterdir()) if p.is_file() and p.name != "v4_pretraining_diagnostic_artifact_hashes.csv"]
    write_csv(output_dir / "v4_pretraining_diagnostic_artifact_hashes.csv", artifact_rows, ("artifact", "size_bytes", "sha256"))
    print("=" * 96); print(f"FINAL_VERDICT={decision['verdict']}"); print(f"training_authorized={decision['training_authorized']}"); print(f"authorization_level={decision['authorization_level']}")
    print(f"metadata_alignment_pass={alignment['hard_pass']}"); print(f"metadata_array_mismatch_count={alignment['mismatch_count']}"); print(f"family_leakage_pass={leakage['hard_pass']}")
    print(f"matched_control_hard_pass={controls['hard_pass']}"); print(f"active_core_shortcut_cases={controls['active_core_shortcut_case_count']}"); print(f"feature_semantic_verdict={feature['verdict']}")
    print(f"attack_kind_alias_count={attack_kind['cross_label_alias_count']}"); print(f"victim_head_blocked={victim['victim_metadata_blocks_victim_head']}")
    print("model_training_performed=False"); print("model_inference_performed=False"); print("test_predictions_accessed=False"); print(f"dataset_directory_modified={not unchanged}"); print(f"decision={decision_path}"); print("=" * 96)
    blocking = {"V4_REQUIRES_SPLIT_OR_METADATA_REBUILD", "V4_REQUIRES_FEATURE_REBUILD", "V4_REQUIRES_GEM5_REGENERATION", "V4_NOT_SAFE_TO_TRAIN"}
    if args.strict and decision["verdict"] in blocking: raise SystemExit(2)


if __name__ == "__main__":
    main()
