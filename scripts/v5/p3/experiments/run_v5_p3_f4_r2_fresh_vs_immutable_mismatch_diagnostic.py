from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STAGE = "V5_P3_F4_R2_FRESH_VS_IMMUTABLE_MISMATCH_DIAGNOSTIC"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"

MAX_LOGIT_ABS_DIFF = 1e-6
MAX_LOGIT_MEAN_ABS_DIFF = 1e-8
EXPECTED_VALIDATION_ITEMS = 13863

IDENTITY_NAME_TERMS = (
    "sample_id",
    "sample_index",
    "item_id",
    "item_index",
    "pair_id",
    "pair_index",
    "window_start",
    "window_index",
    "run_id",
    "pair_member",
    "index",
)

REFERENCE_MANIFEST_NAMES = (
    "V5_P3_D1_TRANCHE_A_VALIDATION_EXPORT_MANIFEST.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def flatten_json(value: Any, prefix: str = "$") -> Iterable[tuple[str, Any]]:
    yield prefix, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_json(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_json(child, f"{prefix}[{index}]")


def find_latest_failed_run(workspace: Path) -> Path:
    candidates = []
    for path in workspace.glob("run_*"):
        if not path.is_dir():
            continue
        comparison = path / "F4_FRESH_VS_IMMUTABLE_NPZ_COMPARISON.json"
        invocation = path / "F4_D1_INVOCATION_CONTRACT.json"
        if comparison.is_file() and invocation.is_file():
            candidates.append(path)
    require(candidates, f"no failed F4 replay runs found under {workspace}")
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return candidates[0]


def headers(path: Path) -> dict[str, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: {
                "shape": [int(item) for item in archive[key].shape],
                "dtype": str(archive[key].dtype),
            }
            for key in archive.files
        }


def select_fresh_npz(run_dir: Path, reference_path: Path) -> Path:
    reference_headers = headers(reference_path)
    reference_keys = set(reference_headers)
    rows = []

    for path in run_dir.rglob("*.npz"):
        if path.resolve() == reference_path.resolve():
            continue
        try:
            current = headers(path)
        except Exception:
            continue
        shared = set(current) & reference_keys
        rows.append({
            "path": path,
            "exact_keys": set(current) == reference_keys,
            "key_overlap": len(shared),
            "shape_matches": sum(
                current[key]["shape"] == reference_headers[key]["shape"]
                for key in shared
            ),
            "dtype_matches": sum(
                current[key]["dtype"] == reference_headers[key]["dtype"]
                for key in shared
            ),
        })

    require(rows, f"no readable fresh NPZ found under {run_dir}")
    rows.sort(
        key=lambda row: (
            row["exact_keys"],
            row["key_overlap"],
            row["shape_matches"],
            row["dtype_matches"],
            row["path"].stat().st_mtime_ns,
        ),
        reverse=True,
    )
    best = rows[0]
    if len(rows) > 1:
        first = (
            best["exact_keys"],
            best["key_overlap"],
            best["shape_matches"],
            best["dtype_matches"],
        )
        second = (
            rows[1]["exact_keys"],
            rows[1]["key_overlap"],
            rows[1]["shape_matches"],
            rows[1]["dtype_matches"],
        )
        require(
            first > second,
            "fresh NPZ selection ambiguous: "
            f"{[str(row['path']) for row in rows[:5]]}",
        )
    return best["path"]


def selected_logit_keys(contract: dict[str, Any]) -> dict[str, str]:
    result = {}
    selected = contract["immutable_reference"]["selected_logits"]
    for role, row in selected.items():
        member = row["member"]
        result[role] = member[:-4] if member.endswith(".npy") else member
    return result


def array_row_hashes(array: np.ndarray) -> list[str]:
    if array.ndim == 0:
        return [hashlib.sha256(array.tobytes()).hexdigest()]
    contiguous = np.ascontiguousarray(array)
    if contiguous.ndim == 1:
        rows = contiguous.reshape(contiguous.shape[0], 1)
    else:
        rows = contiguous.reshape(contiguous.shape[0], -1)
    return [
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in rows
    ]


def row_multiset_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.ndim == 0:
        return bool(np.array_equal(left, right))
    return Counter(array_row_hashes(left)) == Counter(array_row_hashes(right))


def scalar_tuple(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def identity_candidate_keys(
    archive: np.lib.npyio.NpzFile,
    shared_keys: set[str],
) -> list[str]:
    rows = []
    for key in sorted(shared_keys):
        array = archive[key]
        low = normalize(key)
        if not any(term in low for term in IDENTITY_NAME_TERMS):
            continue
        if array.ndim == 0 or array.shape[0] != EXPECTED_VALIDATION_ITEMS:
            continue
        if array.ndim > 2:
            continue
        width = 1 if array.ndim == 1 else int(np.prod(array.shape[1:]))
        if width > 4:
            continue
        rows.append(key)
    return rows


def make_identity_rows(
    archive: np.lib.npyio.NpzFile,
    keys: list[str],
) -> list[tuple[Any, ...]]:
    columns = []
    for key in keys:
        array = archive[key]
        if array.ndim == 1:
            column = [
                (scalar_tuple(value),)
                for value in array
            ]
        else:
            reshaped = array.reshape(array.shape[0], -1)
            column = [
                tuple(scalar_tuple(value) for value in row)
                for row in reshaped
            ]
        columns.append(column)

    rows = []
    for index in range(EXPECTED_VALIDATION_ITEMS):
        combined = []
        for column in columns:
            combined.extend(column[index])
        rows.append(tuple(combined))
    return rows


def discover_reordering(
    reference: np.lib.npyio.NpzFile,
    fresh: np.lib.npyio.NpzFile,
    shared_keys: set[str],
) -> dict[str, Any]:
    candidates = identity_candidate_keys(reference, shared_keys)
    candidates = [
        key for key in candidates
        if key in fresh.files
        and reference[key].shape == fresh[key].shape
        and reference[key].dtype == fresh[key].dtype
    ]
    if not candidates:
        return {
            "available": False,
            "candidate_keys": [],
            "reason": "no shared identifier-like arrays",
        }

    # Use all identifier-like arrays together. This avoids guessing which
    # single field is globally unique.
    ref_rows = make_identity_rows(reference, candidates)
    fresh_rows = make_identity_rows(fresh, candidates)

    ref_unique = len(set(ref_rows)) == len(ref_rows)
    fresh_unique = len(set(fresh_rows)) == len(fresh_rows)
    same_set = set(ref_rows) == set(fresh_rows)

    result = {
        "available": True,
        "candidate_keys": candidates,
        "reference_unique": ref_unique,
        "fresh_unique": fresh_unique,
        "same_identifier_set": same_set,
        "permutation_constructed": False,
    }

    if not (ref_unique and fresh_unique and same_set):
        return result

    fresh_position = {
        row: index for index, row in enumerate(fresh_rows)
    }
    permutation = np.asarray(
        [fresh_position[row] for row in ref_rows],
        dtype=np.int64,
    )
    result["permutation_constructed"] = True
    result["identity_order_already_equal"] = bool(
        np.array_equal(
            permutation,
            np.arange(EXPECTED_VALIDATION_ITEMS, dtype=np.int64),
        )
    )
    result["permutation"] = permutation
    return result


def compare_arrays(
    reference_path: Path,
    fresh_path: Path,
    logit_keys: dict[str, str],
) -> dict[str, Any]:
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        fresh_path,
        allow_pickle=False,
    ) as fresh:
        ref_keys = set(reference.files)
        fresh_keys = set(fresh.files)
        shared = ref_keys & fresh_keys
        logit_set = set(logit_keys.values())

        reorder = discover_reordering(reference, fresh, shared)
        permutation = reorder.get("permutation")

        rows = []
        rows_reordered = []
        schema_pass = ref_keys == fresh_keys
        direct_nonlogit_pass = True
        direct_logit_pass = True
        reordered_nonlogit_pass = True
        reordered_logit_pass = True

        for key in sorted(shared):
            ref = reference[key]
            out = fresh[key]
            is_logit = key in logit_set
            shape_match = ref.shape == out.shape
            dtype_match = ref.dtype == out.dtype
            direct_exact = False
            row_multiset = False
            max_abs = None
            mean_abs = None
            finite = True

            if shape_match and dtype_match:
                direct_exact = bool(np.array_equal(ref, out))
                if ref.ndim >= 1 and ref.shape[0] == EXPECTED_VALIDATION_ITEMS:
                    row_multiset = row_multiset_equal(ref, out)

                if is_logit:
                    ref64 = ref.astype(np.float64, copy=False)
                    out64 = out.astype(np.float64, copy=False)
                    finite = bool(
                        np.all(np.isfinite(ref64))
                        and np.all(np.isfinite(out64))
                    )
                    if finite:
                        diff = np.abs(out64 - ref64)
                        max_abs = float(np.max(diff) if diff.size else 0.0)
                        mean_abs = float(np.mean(diff) if diff.size else 0.0)
                    passed = (
                        finite
                        and max_abs is not None
                        and max_abs <= MAX_LOGIT_ABS_DIFF
                        and mean_abs is not None
                        and mean_abs <= MAX_LOGIT_MEAN_ABS_DIFF
                    )
                    direct_logit_pass &= passed
                else:
                    passed = direct_exact
                    direct_nonlogit_pass &= passed
            else:
                passed = False
                if is_logit:
                    direct_logit_pass = False
                else:
                    direct_nonlogit_pass = False

            rows.append({
                "key": key,
                "is_logit": is_logit,
                "reference_shape": list(ref.shape),
                "fresh_shape": list(out.shape),
                "reference_dtype": str(ref.dtype),
                "fresh_dtype": str(out.dtype),
                "shape_match": shape_match,
                "dtype_match": dtype_match,
                "direct_exact": direct_exact,
                "row_multiset_equal": row_multiset,
                "finite": finite,
                "max_absolute_difference": max_abs,
                "mean_absolute_difference": mean_abs,
                "direct_pass": passed,
            })

            if permutation is not None and shape_match and dtype_match:
                if out.ndim >= 1 and out.shape[0] == EXPECTED_VALIDATION_ITEMS:
                    reordered = out[permutation]
                else:
                    reordered = out

                reordered_exact = bool(np.array_equal(ref, reordered))
                reordered_max = None
                reordered_mean = None
                reordered_finite = True

                if is_logit:
                    ref64 = ref.astype(np.float64, copy=False)
                    out64 = reordered.astype(np.float64, copy=False)
                    reordered_finite = bool(
                        np.all(np.isfinite(ref64))
                        and np.all(np.isfinite(out64))
                    )
                    if reordered_finite:
                        diff = np.abs(out64 - ref64)
                        reordered_max = float(
                            np.max(diff) if diff.size else 0.0
                        )
                        reordered_mean = float(
                            np.mean(diff) if diff.size else 0.0
                        )
                    reordered_pass = (
                        reordered_finite
                        and reordered_max is not None
                        and reordered_max <= MAX_LOGIT_ABS_DIFF
                        and reordered_mean is not None
                        and reordered_mean <= MAX_LOGIT_MEAN_ABS_DIFF
                    )
                    reordered_logit_pass &= reordered_pass
                else:
                    reordered_pass = reordered_exact
                    reordered_nonlogit_pass &= reordered_pass

                rows_reordered.append({
                    "key": key,
                    "is_logit": is_logit,
                    "reordered_exact": reordered_exact,
                    "finite": reordered_finite,
                    "max_absolute_difference": reordered_max,
                    "mean_absolute_difference": reordered_mean,
                    "reordered_pass": reordered_pass,
                })

        if not schema_pass:
            classification = "OUTPUT_SCHEMA_MISMATCH"
        elif direct_nonlogit_pass and direct_logit_pass:
            classification = "COMPARISON_POLICY_OR_PRIOR_ARTIFACT_SELECTION_MISMATCH"
        elif (
            permutation is not None
            and not reorder.get("identity_order_already_equal", True)
            and reordered_nonlogit_pass
            and reordered_logit_pass
        ):
            classification = "SAMPLE_ORDER_MISMATCH_CONFIRMED"
        elif direct_nonlogit_pass and not direct_logit_pass:
            classification = "LOGIT_NUMERIC_DRIFT_ONLY"
        elif not direct_nonlogit_pass and direct_logit_pass:
            classification = "NON_LOGIT_CONTENT_OR_ORDER_MISMATCH"
        else:
            classification = "MIXED_OUTPUT_MISMATCH"

        return {
            "reference_keys": sorted(ref_keys),
            "fresh_keys": sorted(fresh_keys),
            "key_match": ref_keys == fresh_keys,
            "missing_keys": sorted(ref_keys - fresh_keys),
            "extra_keys": sorted(fresh_keys - ref_keys),
            "direct_comparisons": rows,
            "reordering_diagnostic": {
                key: value
                for key, value in reorder.items()
                if key != "permutation"
            },
            "reordered_comparisons": rows_reordered,
            "direct_nonlogit_pass": direct_nonlogit_pass,
            "direct_logit_pass": direct_logit_pass,
            "reordered_nonlogit_pass": (
                reordered_nonlogit_pass if permutation is not None else None
            ),
            "reordered_logit_pass": (
                reordered_logit_pass if permutation is not None else None
            ),
            "classification": classification,
        }


def extract_runtime_provenance(
    invocation: dict[str, Any],
    exporter_log: str,
    reference_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    command = invocation.get("command", [])
    mapping = invocation.get("mapping", {}).get("mapped_arguments", [])

    current = {
        "command": command,
        "mapped_arguments": mapping,
        "device": None,
        "batch_size": None,
        "num_workers": None,
        "seed": None,
    }
    for row in mapping:
        kind = row.get("kind")
        value = row.get("value")
        if kind in current:
            current[kind] = value

    log_lines = []
    for line in exporter_log.splitlines():
        low = normalize(line)
        if any(term in low for term in (
            "device",
            "batch",
            "worker",
            "seed",
            "determin",
            "cuda",
            "checkpoint",
            "validation",
        )):
            log_lines.append(line[:1000])
            if len(log_lines) >= 200:
                break

    reference_rows = []
    if reference_manifest is not None:
        for json_path, value in flatten_json(reference_manifest):
            low = normalize(json_path)
            if any(term in low for term in (
                "device",
                "batch",
                "worker",
                "seed",
                "determin",
                "cuda",
                "checkpoint",
                "model",
                "loader",
            )):
                if isinstance(value, (str, int, float, bool)) or value is None:
                    reference_rows.append({
                        "json_path": json_path,
                        "value": value,
                    })

    return {
        "current_invocation": current,
        "exporter_log_relevant_lines": log_lines,
        "reference_manifest_runtime_fields": reference_rows,
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")

    baseline_dir = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/"
        "baseline_reproduction"
    )
    final_contract_path = baseline_dir / "F4_P1R4_FINAL_F4_REPLAY_CONTRACT.json"
    p1r3_contract_path = baseline_dir / (
        "F4_P1R3_ACTUAL_F4_REPRODUCTION_CONTRACT.json"
    )
    require(final_contract_path.is_file(), "final F4 contract missing")
    require(p1r3_contract_path.is_file(), "F4-P1R3 contract missing")

    final_contract = json.loads(
        final_contract_path.read_text(encoding="utf-8")
    )
    p1r3_contract = json.loads(
        p1r3_contract_path.read_text(encoding="utf-8")
    )
    require(final_contract.get("F4_authorized") is True, "F4 contract not frozen")
    require(final_contract.get("F5_authorized") is False, "F5 must remain held")

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = find_latest_failed_run(
            baseline_dir / "f4_replay_runs"
        )
    require(run_dir.is_dir(), f"run directory missing: {run_dir}")

    comparison_path = run_dir / "F4_FRESH_VS_IMMUTABLE_NPZ_COMPARISON.json"
    invocation_path = run_dir / "F4_D1_INVOCATION_CONTRACT.json"
    exporter_log_path = run_dir / "F4_D1_EXPORTER.log"

    require(comparison_path.is_file(), "original F4 comparison missing")
    require(invocation_path.is_file(), "F4 invocation contract missing")
    require(exporter_log_path.is_file(), "D1 exporter log missing")

    original_comparison = json.loads(
        comparison_path.read_text(encoding="utf-8")
    )
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    exporter_log = exporter_log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    reference_path = Path(
        final_contract["immutable_reference"]["path"]
    ).resolve()
    require(reference_path.is_file(), "immutable reference NPZ missing")
    require(
        sha256_file(reference_path)
        == final_contract["immutable_reference"]["sha256"],
        "immutable reference hash changed",
    )

    fresh_path = select_fresh_npz(run_dir, reference_path)
    logit_keys = selected_logit_keys(p1r3_contract)
    deep = compare_arrays(reference_path, fresh_path, logit_keys)

    manifest_path = None
    for name in REFERENCE_MANIFEST_NAMES:
        candidate = reference_path.parent / name
        if candidate.is_file():
            manifest_path = candidate
            break

    reference_manifest = None
    if manifest_path is not None:
        reference_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

    runtime = extract_runtime_provenance(
        invocation,
        exporter_log,
        reference_manifest,
    )

    classification = deep["classification"]
    next_stage_by_classification = {
        "OUTPUT_SCHEMA_MISMATCH":
            "V5_P3_F4_R3_OUTPUT_SCHEMA_RECONCILIATION_REVIEW",
        "SAMPLE_ORDER_MISMATCH_CONFIRMED":
            "V5_P3_F4_R3_SAMPLE_ORDER_PROVENANCE_REVIEW",
        "LOGIT_NUMERIC_DRIFT_ONLY":
            "V5_P3_F4_R3_RUNTIME_DETERMINISM_AND_DEVICE_MATCH_REVIEW",
        "NON_LOGIT_CONTENT_OR_ORDER_MISMATCH":
            "V5_P3_F4_R3_LABEL_AND_SAMPLE_IDENTITY_REVIEW",
        "MIXED_OUTPUT_MISMATCH":
            "V5_P3_F4_R3_MIXED_OUTPUT_ROOT_CAUSE_REVIEW",
        "COMPARISON_POLICY_OR_PRIOR_ARTIFACT_SELECTION_MISMATCH":
            "V5_P3_F4_R3_COMPARISON_POLICY_REVIEW",
    }
    next_stage = next_stage_by_classification[classification]

    diagnostic_path = output_dir / "F4_R2_DEEP_ARRAY_DIAGNOSTIC.json"
    runtime_path = output_dir / "F4_R2_RUNTIME_AND_REFERENCE_PROVENANCE.json"
    decision_path = output_dir / "F4_R2_FAILURE_CLASSIFICATION_AND_NEXT_STAGE.json"

    atomic_json(
        diagnostic_path,
        {
            "run_directory": str(run_dir),
            "reference_npz": str(reference_path),
            "reference_npz_sha256": sha256_file(reference_path),
            "fresh_npz": str(fresh_path),
            "fresh_npz_sha256": sha256_file(fresh_path),
            "logit_keys": logit_keys,
            "original_comparison": original_comparison,
            "deep_comparison": deep,
            "validation_output_payloads_loaded": True,
            "dataset_tensors_loaded": False,
        },
    )
    atomic_json(runtime_path, runtime)
    atomic_json(
        decision_path,
        {
            "classification": classification,
            "next_stage": next_stage,
            "F4_complete": False,
            "F4M_authorized": False,
            "F5_authorized": False,
            "sealed_test_access": False,
            "repair_authorized": False,
            "reason": (
                "F4-R2 is diagnostic only. It does not reorder arrays, relax "
                "tolerances, replace the immutable reference, or authorize "
                "metric/permutation stages."
            ),
        },
    )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "scope": (
            "Diagnose the completed D1 fresh-versus-immutable validation "
            "output mismatch without rerunning the model or validation split."
        ),
        "run": {
            "directory": str(run_dir),
            "fresh_npz": str(fresh_path),
            "fresh_npz_sha256": sha256_file(fresh_path),
            "immutable_npz": str(reference_path),
            "immutable_npz_sha256": sha256_file(reference_path),
        },
        "classification": classification,
        "deep_comparison": {
            "key_match": deep["key_match"],
            "direct_nonlogit_pass": deep["direct_nonlogit_pass"],
            "direct_logit_pass": deep["direct_logit_pass"],
            "reordering_diagnostic": deep["reordering_diagnostic"],
            "reordered_nonlogit_pass": deep["reordered_nonlogit_pass"],
            "reordered_logit_pass": deep["reordered_logit_pass"],
        },
        "decision": {
            "F4_R2_complete": True,
            "F4_complete": False,
            "F4M_metric_adapter_certification_authorized": False,
            "F5_permutation_authorized": False,
            "repair_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": next_stage,
        },
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "training_tensors_loaded": False,
            "validation_dataset_tensors_loaded": False,
            "validation_output_artifact_payloads_loaded": True,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "immutable_reference_modified": False,
            "fresh_output_modified": False,
        },
        "artifacts": {
            "deep_array_diagnostic": str(diagnostic_path),
            "runtime_and_reference_provenance": str(runtime_path),
            "failure_classification": str(decision_path),
        },
        "provenance": {
            "final_F4_contract_sha256": sha256_file(final_contract_path),
            "F4_P1R3_contract_sha256": sha256_file(p1r3_contract_path),
            "original_comparison_sha256": sha256_file(comparison_path),
            "invocation_contract_sha256": sha256_file(invocation_path),
            "exporter_log_sha256": sha256_file(exporter_log_path),
            "reference_manifest_sha256": (
                sha256_file(manifest_path)
                if manifest_path is not None
                else None
            ),
            "installed_script_sha256": sha256_file(installed_script),
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
            "deep_array_diagnostic_sha256": sha256_file(diagnostic_path),
            "runtime_and_reference_provenance_sha256": sha256_file(
                runtime_path
            ),
            "failure_classification_sha256": sha256_file(decision_path),
            "classification": classification,
            "F4_complete": False,
            "F4M_authorized": False,
            "F5_authorized": False,
            "validation_dataset_tensors_loaded": False,
            "validation_output_artifact_payloads_loaded": True,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"run_directory={run_dir}")
    print(f"fresh_npz={fresh_path}")
    print(f"fresh_npz_sha256={sha256_file(fresh_path)}")
    print(f"immutable_npz={reference_path}")
    print(f"immutable_npz_sha256={sha256_file(reference_path)}")
    print(f"key_match={str(deep['key_match']).lower()}")
    print(
        "direct_nonlogit_pass="
        f"{str(deep['direct_nonlogit_pass']).lower()}"
    )
    print(
        "direct_logit_pass="
        f"{str(deep['direct_logit_pass']).lower()}"
    )
    print(
        "identity_reordering_available="
        f"{str(deep['reordering_diagnostic'].get('available', False)).lower()}"
    )
    print(
        "identity_permutation_constructed="
        f"{str(deep['reordering_diagnostic'].get('permutation_constructed', False)).lower()}"
    )
    print(
        "identity_order_already_equal="
        f"{deep['reordering_diagnostic'].get('identity_order_already_equal')}"
    )
    print(
        "reordered_nonlogit_pass="
        f"{deep['reordered_nonlogit_pass']}"
    )
    print(
        "reordered_logit_pass="
        f"{deep['reordered_logit_pass']}"
    )
    for row in deep["direct_comparisons"]:
        print(
            "array_diagnostic="
            f"{row['key']}:"
            f"logit={str(row['is_logit']).lower()}:"
            f"shape_match={str(row['shape_match']).lower()}:"
            f"dtype_match={str(row['dtype_match']).lower()}:"
            f"direct_exact={str(row['direct_exact']).lower()}:"
            f"row_multiset_equal={str(row['row_multiset_equal']).lower()}:"
            f"max_abs_diff={row['max_absolute_difference']}:"
            f"mean_abs_diff={row['mean_absolute_difference']}:"
            f"pass={str(row['direct_pass']).lower()}"
        )
    print(f"failure_classification={classification}")
    print("F4_complete=false")
    print("F4M_metric_adapter_certification_authorized=false")
    print("F5_permutation_authorized=false")
    print("repair_authorized=false")
    print("model_loaded=false")
    print("checkpoint_loaded=false")
    print("validation_dataset_tensors_loaded=false")
    print("validation_output_artifact_payloads_loaded=true")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={next_stage}")
    print(f"deep_array_diagnostic={diagnostic_path}")
    print(f"runtime_and_reference_provenance={runtime_path}")
    print(f"failure_classification_report={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
