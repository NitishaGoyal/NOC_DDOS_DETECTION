from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = "V5_P3_D4_R1_A1_LEXICOGRAPHIC_INFEASIBILITY_ROOT_CAUSE_AUDIT"
CAMPAIGN_LABEL = "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic"
EXPECTED_ITEMS = 13_863
D4_CHUNK_SIZE = 100
CANONICAL_REPEAT_COUNT = 3
CONTROL_LIMIT = 16
A1_MARGIN_THRESHOLD = 8.7205320882398425


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            key: jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def import_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def semantic_hypothesis(value: Any) -> dict[str, Any]:
    fields = (
        "margin",
        "attacker_count",
        "route_ids",
        "source_mask",
        "transit_mask",
        "victim_mask",
        "path_mask",
    )
    result = {}
    for field in fields:
        if not hasattr(value, field):
            raise RuntimeError(
                f"decoder hypothesis lacks semantic field {field}"
            )
        item = getattr(value, field)
        if field == "margin":
            item = float(item)
            if not math.isfinite(item):
                raise RuntimeError("non-finite hypothesis margin")
            result[field] = {
                "value": item,
                "hex": item.hex(),
            }
        elif field == "route_ids":
            result[field] = [int(entry) for entry in item]
        else:
            result[field] = int(item)
    return result


def result_trace_record(
    result: Any,
    call_index: int,
    matrix: Any,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    raw_fun = getattr(result, "fun", None)
    raw_gap = getattr(result, "mip_gap", None)
    return {
        "call_index": call_index,
        "phase": "primary" if call_index == 1 else "lexicographic",
        "success": bool(getattr(result, "success", False)),
        "status": int(getattr(result, "status", -1)),
        "message": str(getattr(result, "message", "")),
        "fun": (
            float(raw_fun)
            if raw_fun is not None and math.isfinite(float(raw_fun))
            else None
        ),
        "mip_gap": (
            float(raw_gap)
            if raw_gap is not None and math.isfinite(float(raw_gap))
            else None
        ),
        "mip_node_count": int(getattr(result, "mip_node_count", 0) or 0),
        "constraint_rows": int(matrix.shape[0]),
        "constraint_columns": int(matrix.shape[1]),
        "last_lower": float(lower[-1]) if len(lower) else None,
        "last_upper": float(upper[-1]) if len(upper) else None,
    }


def run_instrumented_decode(
    module: Any,
    arrays: dict[str, np.ndarray],
    dataset_index: int,
) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    original_solve = module._solve_milp

    def wrapped_solve(objective, model, matrix, lower, upper):
        result = original_solve(objective, model, matrix, lower, upper)
        trace.append(
            result_trace_record(
                result,
                len(trace) + 1,
                matrix,
                lower,
                upper,
            )
        )
        return result

    module._solve_milp = wrapped_solve
    started = time.perf_counter()
    try:
        hypothesis = module.decode_best_attack_hypothesis(
            float(arrays["attack_logits"][dataset_index]),
            arrays["count_logits"][dataset_index].astype(np.float64),
            arrays["source_logits"][dataset_index].astype(np.float64),
            arrays["transit_logits"][dataset_index].astype(np.float64),
            arrays["victim_logits"][dataset_index].astype(np.float64),
            arrays["path_logits"][dataset_index].astype(np.float64),
        )
    except Exception as exc:
        return {
            "dataset_index": int(dataset_index),
            "status": "FAIL",
            "elapsed_seconds": time.perf_counter() - started,
            "exception_type": type(exc).__name__,
            "exception": repr(exc),
            "solver_trace": trace,
            "lexicographic_solve_calls": max(0, len(trace) - 1),
            "successful_lexicographic_candidates_before_failure": int(
                sum(
                    1
                    for row in trace[1:]
                    if row["success"] and row["status"] == 0
                )
            ),
        }
    finally:
        module._solve_milp = original_solve

    return {
        "dataset_index": int(dataset_index),
        "status": "PASS",
        "elapsed_seconds": time.perf_counter() - started,
        "semantic": semantic_hypothesis(hypothesis),
        "solver_trace": trace,
        "lexicographic_solve_calls": max(0, len(trace) - 1),
    }


def patch_source(
    source: str,
    *,
    presolve_false: bool = False,
    recomputed_center: bool = False,
    outward_rounding: bool = False,
) -> str:
    patched = source

    if presolve_false:
        old = '"presolve": True,'
        if patched.count(old) != 1:
            raise RuntimeError("cannot uniquely patch presolve option")
        patched = patched.replace(old, '"presolve": False,', 1)

    if recomputed_center:
        old = "primary_fun = float(primary.fun)"
        new = (
            "primary_fun = float("
            "primary_certificate.recomputed_objective"
            ")"
        )
        if patched.count(old) != 1:
            raise RuntimeError(
                "cannot uniquely patch primary objective center"
            )
        patched = patched.replace(old, new, 1)

    if outward_rounding:
        old_lower = (
            "np.asarray([primary_fun - SEMANTIC_TIE_TOLERANCE], "
            "dtype=np.float64),"
        )
        new_lower = (
            "np.asarray([np.nextafter("
            "primary_fun - SEMANTIC_TIE_TOLERANCE, -np.inf"
            ")], dtype=np.float64),"
        )
        old_upper = (
            "np.asarray([primary_fun + SEMANTIC_TIE_TOLERANCE], "
            "dtype=np.float64),"
        )
        new_upper = (
            "np.asarray([np.nextafter("
            "primary_fun + SEMANTIC_TIE_TOLERANCE, np.inf"
            ")], dtype=np.float64),"
        )
        if patched.count(old_lower) != 1 or patched.count(old_upper) != 1:
            raise RuntimeError(
                "cannot uniquely patch semantic-face bound rounding"
            )
        patched = patched.replace(old_lower, new_lower, 1)
        patched = patched.replace(old_upper, new_upper, 1)

    return patched


def valid_d4_chunks(
    working_dir: Path,
    export_sha256: str,
    decoder_sha256: str,
) -> list[dict[str, Any]]:
    chunks = []
    for manifest_path in sorted(
        working_dir.glob("chunk_*.manifest.json")
    ):
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            chunk_path = Path(manifest["chunk_path"])
            valid = (
                manifest.get("status") == "COMPLETE"
                and chunk_path.is_file()
                and manifest.get("chunk_sha256") == sha256_file(chunk_path)
                and manifest.get("D1_export_sha256") == export_sha256
                and manifest.get("certified_decoder_sha256")
                == decoder_sha256
                and float(manifest.get("A1_margin_threshold"))
                == A1_MARGIN_THRESHOLD
            )
        except Exception:
            valid = False
            manifest = {"manifest_path": str(manifest_path)}
        if valid:
            chunks.append(
                {
                    "manifest_path": str(manifest_path),
                    "chunk_path": str(chunk_path),
                    "start": int(manifest["start"]),
                    "stop": int(manifest["stop"]),
                    "chunk_sha256": manifest["chunk_sha256"],
                }
            )
    chunks.sort(key=lambda row: row["start"])
    return chunks


def main() -> int:
    args = parse_args()

    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    failure_item_path = output_dir / (
        "V5_P3_D4_R1_ISOLATED_FAILURE_ITEM.json"
    )
    candidate_dir = output_dir / "CANDIDATE_MODULES"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / f"{STAGE}_COMPLETE"
    hold_path = output_dir / f"{STAGE}_HOLD"

    d4_dir = (
        repo
        / "reports/v5/p3_d4_resumable_a1_exact_validation_evaluation"
    )
    d4_working_dir = d4_dir / "A1_WORKING"
    d4_failure_path = d4_dir / (
        "V5_P3_D4_RESUMABLE_A1_EXACT_"
        "TRANCHE_A_VALIDATION_EVALUATION_FAILURE.json"
    )
    d3_dir = (
        repo
        / "reports/v5/p3_d3_resumable_a1_exact_decoder_chunk_preflight"
    )
    d3_lock_path = d3_dir / (
        "V5_P3_D3_RESUMABLE_A1_EXACT_"
        "DECODER_CHUNK_PREFLIGHT_LOCK.json"
    )
    d3_selection_path = d3_dir / (
        "V5_P3_D3_A1_EXACT_PREFLIGHT_SELECTION.json"
    )
    d2_dir = (
        repo
        / "reports/v5/p3_d2_raw_and_frozen_a0_validation_evaluation"
    )
    d2_lock_path = d2_dir / (
        "V5_P3_D2_RAW_AND_FROZEN_A0_"
        "TRANCHE_A_VALIDATION_EVALUATION_LOCK.json"
    )
    d1_dir = (
        repo
        / "reports/v5/p3_d1_immutable_validation_logit_export"
    )
    d1_export_path = d1_dir / (
        "V5_P3_D1_TRANCHE_A_VALIDATION_LOGITS_AND_LABELS.npz"
    )
    d1_lock_path = d1_dir / (
        "V5_P3_D1_IMMUTABLE_TRANCHE_A_"
        "VALIDATION_LOGIT_EXPORT_LOCK.json"
    )
    decoder_path = (
        repo
        / "src/decoders/v5_legal_xy_exact_decoder_certified.py"
    )
    preserved_decoder_path = (
        repo
        / "reports/v5/p2_canonical_preservation/decoder/"
        "v5_legal_xy_exact_decoder_certified.py"
    )

    required = [
        d4_working_dir,
        d4_failure_path,
        d3_lock_path,
        d3_selection_path,
        d2_lock_path,
        d1_export_path,
        d1_lock_path,
        decoder_path,
        preserved_decoder_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")

    d1_lock = json.loads(d1_lock_path.read_text(encoding="utf-8"))
    d2_lock = json.loads(d2_lock_path.read_text(encoding="utf-8"))
    d3_lock = json.loads(d3_lock_path.read_text(encoding="utf-8"))
    d3_selection = json.loads(
        d3_selection_path.read_text(encoding="utf-8")
    )
    d4_failure = json.loads(
        d4_failure_path.read_text(encoding="utf-8")
    )

    export_sha = sha256_file(d1_export_path)
    decoder_sha = sha256_file(decoder_path)
    if d1_lock.get("export_sha256") != export_sha:
        raise RuntimeError("D1 export SHA mismatch")
    if d3_lock.get("certified_decoder_sha256") != decoder_sha:
        raise RuntimeError("decoder differs from D3")
    if decoder_sha != sha256_file(preserved_decoder_path):
        raise RuntimeError(
            "working decoder differs from canonical preservation copy"
        )
    if d2_lock.get("test_tensor_loaded") is not False:
        raise RuntimeError("D2 lock reports test access")
    if d3_lock.get("test_tensor_loaded") is not False:
        raise RuntimeError("D3 lock reports test access")

    chunks = valid_d4_chunks(
        d4_working_dir,
        export_sha,
        decoder_sha,
    )
    contiguous_stop = 0
    contiguous_chunks = []
    for chunk in chunks:
        if chunk["start"] != contiguous_stop:
            break
        contiguous_chunks.append(chunk)
        contiguous_stop = chunk["stop"]

    expected_failure_start = int(d4_failure.get("start", contiguous_stop))
    expected_failure_stop = int(
        d4_failure.get(
            "stop",
            min(expected_failure_start + D4_CHUNK_SIZE, EXPECTED_ITEMS),
        )
    )
    if contiguous_stop != expected_failure_start:
        raise RuntimeError(
            "committed-prefix/failure-chunk mismatch: "
            f"prefix_stop={contiguous_stop}, "
            f"failure_start={expected_failure_start}"
        )

    with np.load(d1_export_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    canonical = import_source(
        decoder_path,
        "_v5_p3_d4_r1_canonical_decoder",
    )

    scan_results = []
    isolated_failure = None
    canonical_success_semantics: dict[int, dict[str, Any]] = {}
    print(
        f"scanning_first_incomplete_chunk="
        f"{expected_failure_start}:{expected_failure_stop}",
        flush=True,
    )
    for dataset_index in range(
        expected_failure_start,
        expected_failure_stop,
    ):
        result = run_instrumented_decode(
            canonical,
            arrays,
            dataset_index,
        )
        scan_results.append(result)
        if result["status"] == "PASS":
            canonical_success_semantics[dataset_index] = result["semantic"]
            if (
                dataset_index == expected_failure_start
                or (dataset_index - expected_failure_start + 1) % 20 == 0
            ):
                print(
                    f"scan_index={dataset_index} status=PASS",
                    flush=True,
                )
            continue
        isolated_failure = result
        print(
            f"isolated_failure_index={dataset_index} "
            f"exception={result['exception']}",
            flush=True,
        )
        break

    if isolated_failure is None:
        raise RuntimeError(
            "canonical decoder did not reproduce the D4 failure "
            "within the first incomplete chunk"
        )

    failure_index = int(isolated_failure["dataset_index"])
    repeat_results = [
        run_instrumented_decode(canonical, arrays, failure_index)
        for _ in range(CANONICAL_REPEAT_COUNT)
    ]
    deterministic_canonical_failure = all(
        result["status"] == "FAIL"
        and result["exception"] == isolated_failure["exception"]
        for result in repeat_results
    )

    # Controls: recent canonical successes plus evenly distributed D3 probes.
    control_indices = list(
        canonical_success_semantics.keys()
    )[-8:]
    for index in d3_selection["ordered_indices"]:
        index = int(index)
        if index not in control_indices and index != failure_index:
            control_indices.append(index)
        if len(control_indices) >= CONTROL_LIMIT:
            break

    for index in control_indices:
        if index not in canonical_success_semantics:
            result = run_instrumented_decode(canonical, arrays, index)
            if result["status"] != "PASS":
                raise RuntimeError(
                    f"canonical control unexpectedly failed at {index}: "
                    f"{result['exception']}"
                )
            canonical_success_semantics[index] = result["semantic"]

    source_text = decoder_path.read_text(encoding="utf-8")
    candidate_specs = {
        "presolve_false": {
            "presolve_false": True,
        },
        "outward_rounding": {
            "outward_rounding": True,
        },
        "recomputed_center": {
            "recomputed_center": True,
        },
        "recomputed_center_plus_outward_rounding": {
            "recomputed_center": True,
            "outward_rounding": True,
        },
        "recomputed_center_outward_rounding_presolve_false": {
            "recomputed_center": True,
            "outward_rounding": True,
            "presolve_false": True,
        },
    }

    candidate_results: dict[str, Any] = {}
    for name, options in candidate_specs.items():
        candidate_path = candidate_dir / f"{name}.py"
        candidate_source = patch_source(source_text, **options)
        candidate_path.write_text(candidate_source, encoding="utf-8")
        module = import_source(
            candidate_path,
            f"_v5_p3_d4_r1_candidate_{name}",
        )

        failure_result = run_instrumented_decode(
            module,
            arrays,
            failure_index,
        )
        control_rows = []
        all_controls_match = True
        for control_index in control_indices:
            candidate_control = run_instrumented_decode(
                module,
                arrays,
                control_index,
            )
            match = (
                candidate_control["status"] == "PASS"
                and candidate_control["semantic"]
                == canonical_success_semantics[control_index]
            )
            all_controls_match &= match
            control_rows.append(
                {
                    "dataset_index": control_index,
                    "status": candidate_control["status"],
                    "semantic_matches_canonical": match,
                    "exception": candidate_control.get("exception"),
                }
            )

        candidate_results[name] = {
            "options": options,
            "source_path": str(candidate_path),
            "source_sha256": sha256_file(candidate_path),
            "failure_item_result": failure_result,
            "failure_item_succeeds": (
                failure_result["status"] == "PASS"
            ),
            "control_items": control_rows,
            "all_control_semantics_match": all_controls_match,
            "viable_numerical_candidate": (
                failure_result["status"] == "PASS"
                and all_controls_match
            ),
        }
        print(
            f"candidate={name} "
            f"failure_item_status={failure_result['status']} "
            f"controls_match={all_controls_match}",
            flush=True,
        )

    viable = [
        name
        for name, result in candidate_results.items()
        if result["viable_numerical_candidate"]
    ]
    viable_failure_semantics = {
        json.dumps(
            candidate_results[name]["failure_item_result"]["semantic"],
            sort_keys=True,
        )
        for name in viable
    }
    viable_candidates_agree = (
        len(viable_failure_semantics) <= 1
    )

    if viable and viable_candidates_agree:
        status = "PASS"
        root_cause = (
            "LEXICOGRAPHIC_FLOAT64_FORMULATION_OR_PRESOLVE_"
            "INFEASIBILITY_WITH_CERTIFIED_SEMANTIC_RECOVERY_CANDIDATE"
        )
        next_stage = (
            "V5_P3_D4_R2_PROSPECTIVE_NUMERICAL_IMPLEMENTATION_"
            "FREEZE_AND_FULL_REGRESSION"
        )
        complete_path.write_text(
            f"{STAGE}_COMPLETE\n",
            encoding="utf-8",
        )
        if hold_path.exists():
            hold_path.unlink()
    else:
        status = "HOLD"
        root_cause = (
            "UNRESOLVED_OR_SEMANTICALLY_AMBIGUOUS_"
            "LEXICOGRAPHIC_INFEASIBILITY"
        )
        next_stage = (
            "D4_R1_HOLD_DEEPER_EXACT_DECODER_NUMERICAL_ANALYSIS"
        )
        hold_path.write_text(
            f"{STAGE}_HOLD\n",
            encoding="utf-8",
        )

    failure_item = {
        "stage": STAGE,
        "dataset_index": failure_index,
        "labels": {
            "graph": int(arrays["y_attack"][failure_index]),
            "attacker_count": int(
                arrays["y_attacker_count"][failure_index]
            ),
            "source": arrays["y_source"][failure_index].astype(int).tolist(),
            "transit": arrays["y_transit"][failure_index].astype(int).tolist(),
            "victim": arrays["y_victim"][failure_index].astype(int).tolist(),
            "path": arrays["y_attack_path"][failure_index].astype(int).tolist(),
        },
        "logits": {
            "attack": float(arrays["attack_logits"][failure_index]),
            "count": arrays["count_logits"][failure_index].tolist(),
            "source": arrays["source_logits"][failure_index].tolist(),
            "transit": arrays["transit_logits"][failure_index].tolist(),
            "victim": arrays["victim_logits"][failure_index].tolist(),
            "path": arrays["path_logits"][failure_index].tolist(),
        },
        "canonical_first_failure": isolated_failure,
        "canonical_repeats": repeat_results,
    }
    atomic_json(failure_item_path, failure_item)

    report = {
        "stage": STAGE,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "scope": (
            "Validation-only isolation and numerical root-cause audit of "
            "the first certified exact-decoder lexicographic infeasibility "
            "encountered during D4."
        ),
        "d4_prefix": {
            "hash_valid_contiguous_chunks": len(contiguous_chunks),
            "hash_valid_contiguous_items": contiguous_stop,
            "first_incomplete_chunk_start": expected_failure_start,
            "first_incomplete_chunk_stop": expected_failure_stop,
            "remaining_items_from_prefix": EXPECTED_ITEMS - contiguous_stop,
            "preserved": True,
        },
        "failure": {
            "isolated_dataset_index": failure_index,
            "canonical_exception": isolated_failure["exception"],
            "canonical_solver_trace": isolated_failure["solver_trace"],
            "canonical_repeat_count": CANONICAL_REPEAT_COUNT,
            "canonical_failure_deterministic": (
                deterministic_canonical_failure
            ),
            "artifact_path": str(failure_item_path),
            "artifact_sha256": sha256_file(failure_item_path),
        },
        "candidate_audit": {
            "control_indices": control_indices,
            "candidate_results": candidate_results,
            "viable_candidates": viable,
            "viable_candidates_agree_on_failure_semantics": (
                viable_candidates_agree
            ),
            "diagnostic_only": True,
            "repo_decoder_modified": False,
            "D4_chunks_modified": False,
        },
        "root_cause_disposition": root_cause,
        "decision": {
            "existing_D4_prefix_reusable": True,
            "canonical_decoder_full_validation_complete": False,
            "numerical_candidate_freeze_authorized": (
                status == "PASS"
            ),
            "candidate_directly_installable": False,
            "threshold_or_margin_retuning_authorized": False,
            "beam_decoder_selected": False,
            "sealed_test_evaluation_authorized": False,
            "next_stage": next_stage,
        },
        "sealed_test": {
            "test_dataset_instantiated": False,
            "test_length_computed": False,
            "test_tensor_loaded": False,
            "evaluation_authorized": False,
        },
        "provenance": {
            "D1_export_sha256": export_sha,
            "D2_lock_sha256": sha256_file(d2_lock_path),
            "D3_lock_sha256": sha256_file(d3_lock_path),
            "canonical_decoder_sha256": decoder_sha,
            "D4_failure_artifact_sha256": sha256_file(d4_failure_path),
            "installed_script_sha256": sha256_file(installed_script),
        },
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": status,
        "report_sha256": sha256_file(report_path),
        "failure_item_sha256": sha256_file(failure_item_path),
        "D1_export_sha256": export_sha,
        "canonical_decoder_sha256": decoder_sha,
        "isolated_failure_index": failure_index,
        "hash_valid_D4_prefix_items": contiguous_stop,
        "viable_candidates": viable,
        "test_tensor_loaded": False,
        "certified_dataset_modified": False,
    }
    atomic_json(lock_path, lock)

    print(f"{STAGE}_{'COMPLETE' if status == 'PASS' else 'HOLD'}")
    print(f"status={status}")
    print(f"hash_valid_D4_prefix_chunks={len(contiguous_chunks)}")
    print(f"hash_valid_D4_prefix_items={contiguous_stop}")
    print(f"remaining_items_from_prefix={EXPECTED_ITEMS - contiguous_stop}")
    print(f"isolated_failure_index={failure_index}")
    print(
        "canonical_failure_deterministic="
        f"{deterministic_canonical_failure}"
    )
    print(
        "canonical_exception="
        f"{isolated_failure['exception']}"
    )
    print(f"viable_candidates={viable}")
    print(
        "viable_candidates_agree_on_failure_semantics="
        f"{viable_candidates_agree}"
    )
    print("repo_decoder_modified=false")
    print("D4_chunks_modified=false")
    print("threshold_or_margin_retuning_authorized=false")
    print("beam_decoder_selected=false")
    print("test_tensor_loaded=false")
    print(f"next_stage={next_stage}")
    print(f"failure_item={failure_item_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
