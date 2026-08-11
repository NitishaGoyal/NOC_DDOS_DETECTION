from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STAGE = "V5_P3_F2R_DYNAMIC70_DEGENERACY_FINDING_REVIEW"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
ZERO_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def parse_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def flatten_json(value: Any, prefix: str = "$") -> Iterable[tuple[str, Any]]:
    yield prefix, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_json(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_json(child, f"{prefix}[{index}]")


def collect_metadata_evidence(
    data_root: Path,
    constant_rows: list[dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    names = {row["name"] for row in constant_rows}
    raw_indices = {int(row["raw81_index"]) for row in constant_rows}
    evidence: dict[str, list[dict[str, Any]]] = {
        name: [] for name in sorted(names)
    }

    for path in sorted(data_root.rglob("*.json")):
        try:
            if path.stat().st_size > 32 * 1024 * 1024:
                continue
            document = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for json_path, value in flatten_json(document):
            text = str(value)
            for row in constant_rows:
                name = row["name"]
                raw_index = int(row["raw81_index"])
                name_hit = name in text
                index_hit = (
                    raw_index in raw_indices
                    and (
                        json_path.endswith(f"[{raw_index}]")
                        or json_path.endswith(f".{raw_index}")
                    )
                )
                if name_hit or index_hit:
                    if len(evidence[name]) < 100:
                        evidence[name].append({
                            "file": str(path),
                            "json_path": json_path,
                            "value_preview": text[:500],
                        })
    return evidence


def all_close_zero(values: list[float]) -> bool:
    return all(
        math.isfinite(value) and abs(value) <= ZERO_TOLERANCE
        for value in values
    )


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

    study_dir = repo / "reports/v5/p3_experiments/f0_d70_feature_study"
    audit_dir = study_dir / "schema_audit"

    f0_lock_path = study_dir / "V5_P3_F0_D70_FEATURE_STUDY_LOCK.json"
    group_contract_path = study_dir / "FEATURE_GROUP_CONTRACT.json"
    metric_contract_path = study_dir / "METRIC_CONTRACT.json"
    data_access_path = study_dir / "DATA_ACCESS_CONTRACT.json"
    f1_lock_path = audit_dir / "V5_P3_F1_DYNAMIC70_SCHEMA_CERTIFICATION_LOCK.json"
    f2_report_path = audit_dir / (
        "V5_P3_F2_DYNAMIC70_TRAINING_ONLY_DEGENERACY_"
        "AND_OBSERVABILITY_AUDIT_REPORT.json"
    )
    f2_lock_path = audit_dir / (
        "V5_P3_F2_DYNAMIC70_TRAINING_ONLY_DEGENERACY_"
        "AND_OBSERVABILITY_AUDIT_LOCK.json"
    )

    feature_csv = audit_dir / "F2_FEATURE_STATISTICS.csv"
    control_attack_csv = audit_dir / "F2_CONTROL_ATTACK_STATISTICS.csv"
    count_csv = audit_dir / "F2_ATTACKER_COUNT_STATISTICS.csv"
    position_csv = audit_dir / "F2_ROUTER_POSITION_STATISTICS.csv"
    topology_csv = audit_dir / "F2_TOPOLOGY_MASK_AUDIT.csv"
    group_csv = audit_dir / "F2_GROUP_SUMMARY.csv"

    required = [
        f0_lock_path,
        group_contract_path,
        metric_contract_path,
        data_access_path,
        f1_lock_path,
        f2_report_path,
        f2_lock_path,
        feature_csv,
        control_attack_csv,
        count_csv,
        position_csv,
        topology_csv,
        group_csv,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"required artifacts missing: {missing}")

    f0_lock = json.loads(f0_lock_path.read_text(encoding="utf-8"))
    f2_report = json.loads(f2_report_path.read_text(encoding="utf-8"))
    f2_lock = json.loads(f2_lock_path.read_text(encoding="utf-8"))

    require(f0_lock.get("status") == "FROZEN", "F0 is not frozen")
    require(f2_report.get("status") == "PASS", "F2 is not PASS")
    require(
        f2_lock.get("report_sha256") == sha256_file(f2_report_path),
        "F2 report/lock mismatch",
    )
    require(
        int(f2_lock.get("constant_channel_count", -1)) == 6,
        "this F2R package expects the observed six constant channels",
    )
    require(
        int(f2_lock.get("nonfinite_value_count", -1)) == 0,
        "nonfinite values require a different failure review",
    )
    require(
        f2_lock.get("F3_authorized") is False,
        "F2 unexpectedly already authorized F3",
    )

    feature_rows = read_csv(feature_csv)
    control_attack_rows = read_csv(control_attack_csv)
    count_rows = read_csv(count_csv)
    position_rows = read_csv(position_csv)
    topology_rows = read_csv(topology_csv)
    group_rows = read_csv(group_csv)

    constants = [
        row for row in feature_rows
        if row["activity_class"] == "CONSTANT"
    ]
    require(len(constants) == 6, f"expected 6 constants, found {len(constants)}")

    control_by_name = {row["name"]: row for row in control_attack_rows}
    topology_by_name = {row["name"]: row for row in topology_rows}
    count_by_name: dict[str, list[dict[str, str]]] = {}
    for row in count_rows:
        count_by_name.setdefault(row["name"], []).append(row)
    position_by_name: dict[str, list[dict[str, str]]] = {}
    for row in position_rows:
        position_by_name.setdefault(row["name"], []).append(row)

    reviewed_rows = []
    all_zero_constants = True
    all_valid_observed = True
    all_topology_clean = True
    all_strata_zero = True

    for row in constants:
        name = row["name"]
        minimum = parse_float(row["minimum"])
        maximum = parse_float(row["maximum"])
        mean = parse_float(row["mean"])
        std = parse_float(row["std"])
        zero_fraction = parse_float(row["zero_fraction_valid"])
        valid_count = parse_int(row["valid_value_count"])
        invalid_nonzero = parse_int(row["invalid_port_nonzero_count"])
        nonfinite = parse_int(row["nonfinite_count"])
        sampled_unique = parse_int(row["sampled_unique_count"])

        constant_zero = all_close_zero([minimum, maximum, mean, std])
        valid_observed = valid_count > 0 and nonfinite == 0
        topology_clean = invalid_nonzero == 0
        sampled_singleton = sampled_unique <= 1

        control = control_by_name[name]
        stratum_values = [
            parse_float(control["control_mean"]),
            parse_float(control["control_std"]),
            parse_float(control["attack_mean"]),
            parse_float(control["attack_std"]),
        ]
        for count_row in count_by_name.get(name, []):
            stratum_values.extend([
                parse_float(count_row["mean"]),
                parse_float(count_row["std"]),
            ])
        for position_row in position_by_name.get(name, []):
            stratum_values.extend([
                parse_float(position_row["mean"]),
                parse_float(position_row["std"]),
                parse_float(position_row["minimum"]),
                parse_float(position_row["maximum"]),
            ])
        strata_zero = all_close_zero(stratum_values)

        all_zero_constants &= constant_zero
        all_valid_observed &= valid_observed
        all_topology_clean &= topology_clean
        all_strata_zero &= strata_zero

        reviewed_rows.append({
            "dynamic70_index": int(row["dynamic70_index"]),
            "raw81_index": int(row["raw81_index"]),
            "name": name,
            "macro_group": row["macro_group"],
            "fine_group": row["fine_group"],
            "direction": row["direction"],
            "valid_value_count": valid_count,
            "minimum": minimum,
            "maximum": maximum,
            "mean": mean,
            "std": std,
            "zero_fraction_valid": zero_fraction,
            "sampled_unique_count": sampled_unique,
            "invalid_port_nonzero_count": invalid_nonzero,
            "nonfinite_count": nonfinite,
            "constant_zero": constant_zero,
            "valid_ports_observed": valid_observed,
            "invalid_port_values_clean": topology_clean,
            "all_control_attack_count_position_strata_zero": strata_zero,
            "classification": (
                "TRAINING_OBSERVED_ZERO_SIGNAL"
                if (
                    constant_zero
                    and valid_observed
                    and topology_clean
                    and sampled_singleton
                    and strata_zero
                )
                else "UNEXPLAINED_CONSTANT"
            ),
        })

    metadata_evidence = collect_metadata_evidence(
        data_root,
        constants,
    )

    group_counts: dict[str, int] = {}
    fine_counts: dict[str, int] = {}
    direction_counts: dict[str, int] = {}
    for row in reviewed_rows:
        group_counts[row["macro_group"]] = group_counts.get(row["macro_group"], 0) + 1
        fine_counts[row["fine_group"]] = fine_counts.get(row["fine_group"], 0) + 1
        direction_counts[row["direction"]] = direction_counts.get(row["direction"], 0) + 1

    observationally_resolved = (
        all_zero_constants
        and all_valid_observed
        and all_topology_clean
        and all_strata_zero
        and all(
            row["classification"] == "TRAINING_OBSERVED_ZERO_SIGNAL"
            for row in reviewed_rows
        )
    )

    # This review does not prove instrumentation correctness. It only proves
    # that the six channels are consistently zero in every training stratum.
    f3_authorized = observationally_resolved

    review_csv = audit_dir / "F2R_CONSTANT_CHANNEL_REVIEW.csv"
    with review_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(reviewed_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(reviewed_rows)

    evidence_path = audit_dir / "F2R_CONSTANT_CHANNEL_METADATA_EVIDENCE.json"
    atomic_json(evidence_path, metadata_evidence)

    amendment_path = audit_dir / "F2R_PROTOCOL_AMENDMENT_CONSTANT_CHANNEL_HANDLING.json"
    amendment = {
        "campaign": CAMPAIGN,
        "stage": STAGE,
        "status": "FROZEN" if f3_authorized else "HOLD",
        "reason": (
            "F2 discovered six channels that are constant on physically valid "
            "Tranche-A training inputs."
        ),
        "observed_classification": (
            "TRAINING_OBSERVED_ZERO_SIGNAL"
            if observationally_resolved
            else "UNRESOLVED_CONSTANT_CHANNELS"
        ),
        "handling_for_future_stages": {
            "official_Dynamic70_tensor_changed": False,
            "A4_A5_checkpoint_changed": False,
            "constant_channels_retained_in_tensor": True,
            "constant_channels_assigned_zero_descriptive_effect": True,
            "standardized_effect_denominator_policy": (
                "do not divide by zero; report undefined standardized effect "
                "and exact zero signed/absolute response"
            ),
            "permutation_policy": (
                "permuting a constant channel/group component is expected to "
                "produce zero change and remains auditable"
            ),
            "attribution_policy": (
                "retain channels; report zero/near-zero attribution rather than "
                "dropping them before attribution"
            ),
            "retraining_ablation_policy": (
                "macro-group ablations remain unchanged; no single-channel "
                "removal is authorized yet"
            ),
            "reduced_interface_removal_authorized": False,
            "instrumentation_correctness_proven": False,
        },
        "scientific_language": (
            "These channels are training-observed zero-signal fields, not yet "
            "proven permanently redundant across Tranche-B or future A+B."
        ),
        "promotion_boundary": [
            "frozen Tranche-B audit",
            "future fresh A+B audit",
        ],
    }
    atomic_json(amendment_path, amendment)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "scope": (
            "Review the six F2 constant-channel findings using frozen F2 "
            "outputs and dataset JSON metadata only."
        ),
        "constant_channel_count": len(reviewed_rows),
        "constant_channels": reviewed_rows,
        "distribution": {
            "macro_groups": group_counts,
            "fine_groups": fine_counts,
            "directions": direction_counts,
        },
        "review_checks": {
            "all_constants_exact_zero": all_zero_constants,
            "all_have_physically_valid_observations": all_valid_observed,
            "all_invalid_port_values_zero": all_topology_clean,
            "all_control_attack_count_position_strata_zero": all_strata_zero,
            "observationally_resolved": observationally_resolved,
            "instrumentation_correctness_proven": False,
        },
        "decision": {
            "F2R_complete": True,
            "F3_authorized": f3_authorized,
            "constant_channel_removal_authorized": False,
            "Dynamic70_interface_change_authorized": False,
            "validation_access_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": (
                "V5_P3_F3_DYNAMIC70_TRAINING_ONLY_MATCHED_"
                "CONTROL_ATTACK_RESPONSE_ANALYSIS"
                if f3_authorized
                else "V5_P3_F2R2_CONSTANT_CHANNEL_INSTRUMENTATION_AUDIT"
            ),
        },
        "artifacts": {
            "review_csv": str(review_csv),
            "review_csv_sha256": sha256_file(review_csv),
            "metadata_evidence": str(evidence_path),
            "metadata_evidence_sha256": sha256_file(evidence_path),
            "protocol_amendment": str(amendment_path),
            "protocol_amendment_sha256": sha256_file(amendment_path),
        },
        "governance": {
            "model_loaded": False,
            "training_tensors_loaded": False,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
            "dataset_JSON_metadata_read": True,
        },
        "provenance": {
            "F0_lock_sha256": sha256_file(f0_lock_path),
            "feature_group_contract_sha256": sha256_file(group_contract_path),
            "metric_contract_sha256": sha256_file(metric_contract_path),
            "data_access_contract_sha256": sha256_file(data_access_path),
            "F1_lock_sha256": sha256_file(f1_lock_path),
            "F2_report_sha256": sha256_file(f2_report_path),
            "F2_lock_sha256": sha256_file(f2_lock_path),
            "F2_feature_statistics_sha256": sha256_file(feature_csv),
            "F2_control_attack_statistics_sha256": sha256_file(control_attack_csv),
            "F2_attacker_count_statistics_sha256": sha256_file(count_csv),
            "F2_router_position_statistics_sha256": sha256_file(position_csv),
            "F2_topology_mask_audit_sha256": sha256_file(topology_csv),
            "installed_script_sha256": sha256_file(installed_script),
        },
    }

    report_path = audit_dir / f"{STAGE}_REPORT.json"
    lock_path = audit_dir / f"{STAGE}_LOCK.json"
    complete_path = audit_dir / f"{STAGE}_COMPLETE"
    atomic_json(report_path, report)
    atomic_json(
        lock_path,
        {
            "stage": STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(report_path),
            "review_csv_sha256": sha256_file(review_csv),
            "metadata_evidence_sha256": sha256_file(evidence_path),
            "protocol_amendment_sha256": sha256_file(amendment_path),
            "constant_channel_count": len(reviewed_rows),
            "observationally_resolved": observationally_resolved,
            "F3_authorized": f3_authorized,
            "constant_channel_removal_authorized": False,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"constant_channel_count={len(reviewed_rows)}")
    for row in reviewed_rows:
        print(
            "constant_channel="
            f"{row['dynamic70_index']}:{row['name']}:"
            f"{row['classification']}"
        )
    print(f"constant_macro_group_counts={group_counts}")
    print(f"constant_fine_group_counts={fine_counts}")
    print(f"constant_direction_counts={direction_counts}")
    print(f"all_constants_exact_zero={str(all_zero_constants).lower()}")
    print(
        "all_have_physically_valid_observations="
        f"{str(all_valid_observed).lower()}"
    )
    print(
        "all_invalid_port_values_zero="
        f"{str(all_topology_clean).lower()}"
    )
    print(
        "all_control_attack_count_position_strata_zero="
        f"{str(all_strata_zero).lower()}"
    )
    print(
        "instrumentation_correctness_proven=false"
    )
    print("constant_channel_removal_authorized=false")
    print("Dynamic70_interface_change_authorized=false")
    print("model_loaded=false")
    print("training_tensors_loaded=false")
    print("validation_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"F3_authorized={str(f3_authorized).lower()}")
    print(f"next_stage={report['decision']['next_stage']}")
    print(f"review_csv={review_csv}")
    print(f"protocol_amendment={amendment_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
