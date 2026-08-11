from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

STAGE = "V5_P3_D4_R2_R1_PRESOLVE_FALSE_INTEGRALITY_BOUNDARY_AND_HYBRID_RETRY_AUDIT"
A1_MARGIN_THRESHOLD = 8.7205320882398425
EXPECTED_PREFIX_ITEMS = 11700
CHUNK_SIZE = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def import_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def float_bits(value: float) -> int:
    return int(np.asarray([value], dtype=np.float64).view(np.uint64)[0])


def decode(d4, decoder, arrays, index: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        hypothesis = decoder.decode_best_attack_hypothesis(
            float(arrays["attack_logits"][index]),
            arrays["count_logits"][index].astype(np.float64),
            arrays["source_logits"][index].astype(np.float64),
            arrays["transit_logits"][index].astype(np.float64),
            arrays["victim_logits"][index].astype(np.float64),
            arrays["path_logits"][index].astype(np.float64),
        )
        decoded = decoder.apply_margin_threshold(
            hypothesis,
            A1_MARGIN_THRESHOLD,
        )
        extracted, _ = d4.extract_decoded_output(decoded, hypothesis)
        margin = float(extracted["margin"])
        return {
            "status": "PASS",
            "dataset_index": index,
            "graph": int(extracted["graph"]),
            "count": int(extracted["count"]),
            "source": extracted["source"].astype(int).tolist(),
            "transit": extracted["transit"].astype(int).tolist(),
            "victim": extracted["victim"].astype(int).tolist(),
            "path": extracted["path"].astype(int).tolist(),
            "route_count": int(extracted["route_ids"].size),
            "route_ids": [int(v) for v in extracted["route_ids"]],
            "margin": margin,
            "margin_hex": margin.hex(),
            "margin_bits": float_bits(margin),
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "status": "FAIL",
            "dataset_index": index,
            "exception_type": type(exc).__name__,
            "exception": repr(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }


def stored_semantic(chunk, local: int, index: int) -> dict[str, Any]:
    margin = float(chunk["margin"][local])
    return {
        "status": "PASS",
        "dataset_index": index,
        "graph": int(chunk["graph_prediction"][local]),
        "count": int(chunk["count_prediction"][local]),
        "source": chunk["source_prediction"][local].astype(int).tolist(),
        "transit": chunk["transit_prediction"][local].astype(int).tolist(),
        "victim": chunk["victim_prediction"][local].astype(int).tolist(),
        "path": chunk["path_prediction"][local].astype(int).tolist(),
        "route_count": int(chunk["route_count"][local]),
        "margin": margin,
        "margin_hex": margin.hex(),
        "margin_bits": float_bits(margin),
    }


def semantic_matches(decoded, stored) -> bool:
    keys = (
        "graph", "count", "source", "transit", "victim",
        "path", "route_count", "margin_bits",
    )
    return (
        decoded.get("status") == "PASS"
        and all(decoded.get(key) == stored.get(key) for key in keys)
    )


def parse_boundary(exception: str) -> dict[str, Any]:
    match = re.search(
        r"maximum_error=([0-9eE+\-.]+), tolerance=([0-9eE+\-.]+)",
        exception,
    )
    if not match:
        return {"parsed": False}
    maximum_error = float(match.group(1))
    tolerance = float(match.group(2))
    return {
        "parsed": True,
        "maximum_error": maximum_error,
        "tolerance": tolerance,
        "absolute_excess": maximum_error - tolerance,
        "relative_excess": (
            (maximum_error - tolerance) / tolerance
            if tolerance else None
        ),
    }


def contiguous_r2_prefix(working_dir: Path, candidate_sha: str, export_sha: str):
    rows = []
    for manifest_path in sorted(working_dir.glob("chunk_*.manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
            chunk_path = Path(manifest["candidate_chunk_path"])
            valid = (
                manifest.get("status") == "COMPLETE"
                and chunk_path.is_file()
                and manifest.get("candidate_chunk_sha256") == sha256_file(chunk_path)
                and manifest.get("candidate_sha256") == candidate_sha
                and manifest.get("D1_export_sha256") == export_sha
                and manifest.get("all_items_exact_match") is True
            )
        except Exception:
            valid = False
        if valid:
            rows.append(
                {
                    "start": int(manifest["start"]),
                    "stop": int(manifest["stop"]),
                    "manifest": str(manifest_path),
                    "chunk": str(chunk_path),
                }
            )
    rows.sort(key=lambda row: row["start"])
    contiguous = []
    stop = 0
    for row in rows:
        if row["start"] != stop:
            break
        contiguous.append(row)
        stop = row["stop"]
    return contiguous, stop


def load_canonical_chunk(
    working_dir: Path,
    start: int,
    stop: int,
    export_sha: str,
    canonical_sha: str,
):
    matches = list(
        working_dir.glob(
            f"chunk_*_items_{start:05d}_{stop:05d}.manifest.json"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one canonical chunk for {start}:{stop}, found {matches}"
        )
    manifest_path = matches[0]
    manifest = json.loads(manifest_path.read_text())
    chunk_path = Path(manifest["chunk_path"])
    if manifest.get("chunk_sha256") != sha256_file(chunk_path):
        raise RuntimeError("canonical chunk SHA mismatch")
    if manifest.get("D1_export_sha256") != export_sha:
        raise RuntimeError("canonical export SHA mismatch")
    if manifest.get("certified_decoder_sha256") != canonical_sha:
        raise RuntimeError("canonical decoder SHA mismatch")
    with np.load(chunk_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    return manifest, arrays


def hybrid_decode(d4, canonical, fallback, arrays, index: int):
    canonical_result = decode(d4, canonical, arrays, index)
    if canonical_result["status"] == "PASS":
        return {
            "status": "PASS",
            "selected_path": "canonical_presolve_true",
            "semantic": canonical_result,
            "canonical_result": canonical_result,
            "fallback_result": None,
        }
    fallback_result = decode(d4, fallback, arrays, index)
    if fallback_result["status"] == "PASS":
        return {
            "status": "PASS",
            "selected_path": "certified_fallback_presolve_false",
            "semantic": fallback_result,
            "canonical_result": canonical_result,
            "fallback_result": fallback_result,
        }
    return {
        "status": "FAIL",
        "selected_path": "both_failed",
        "canonical_result": canonical_result,
        "fallback_result": fallback_result,
    }


def mask_int(bits):
    return int(sum(int(bit) << i for i, bit in enumerate(bits)))


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    isolated_path = output_dir / "V5_P3_D4_R2_R1_ISOLATED_ITEM.json"

    r2_dir = repo / "reports/v5/p3_d4_r2_prospective_numerical_freeze_and_full_regression"
    r2_working = r2_dir / "FULL_PREFIX_REGRESSION_WORKING"
    candidate_path = r2_dir / "V5_P3_A1_EXACT_DECODER_PRESOLVE_FALSE_FROZEN.py"
    contract_path = r2_dir / "V5_P3_D4_R2_NUMERICAL_IMPLEMENTATION_FREEZE_CONTRACT.json"

    d4_dir = repo / "reports/v5/p3_d4_resumable_a1_exact_validation_evaluation"
    d4_working = d4_dir / "A1_WORKING"
    d4_script = repo / "scripts/v5/p3/run_v5_p3_d4_resumable_a1_exact_validation_evaluation.py"

    r1_dir = repo / "reports/v5/p3_d4_r1_a1_lexicographic_infeasibility_root_cause_audit"
    r1_report_path = r1_dir / "V5_P3_D4_R1_A1_LEXICOGRAPHIC_INFEASIBILITY_ROOT_CAUSE_AUDIT_REPORT.json"
    r1_lock_path = r1_dir / "V5_P3_D4_R1_A1_LEXICOGRAPHIC_INFEASIBILITY_ROOT_CAUSE_AUDIT_LOCK.json"

    d1_dir = repo / "reports/v5/p3_d1_immutable_validation_logit_export"
    export_path = d1_dir / "V5_P3_D1_TRANCHE_A_VALIDATION_LOGITS_AND_LABELS.npz"
    d1_lock_path = d1_dir / "V5_P3_D1_IMMUTABLE_TRANCHE_A_VALIDATION_LOGIT_EXPORT_LOCK.json"

    canonical_path = repo / "src/decoders/v5_legal_xy_exact_decoder_certified.py"

    required = [
        r2_working, candidate_path, contract_path, d4_working, d4_script,
        r1_report_path, r1_lock_path, export_path, d1_lock_path, canonical_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")

    contract = json.loads(contract_path.read_text())
    r1_report = json.loads(r1_report_path.read_text())
    r1_lock = json.loads(r1_lock_path.read_text())
    d1_lock = json.loads(d1_lock_path.read_text())

    candidate_sha = sha256_file(candidate_path)
    canonical_sha = sha256_file(canonical_path)
    export_sha = sha256_file(export_path)

    if contract["frozen_candidate_sha256"] != candidate_sha:
        raise RuntimeError("candidate SHA mismatch")
    if contract["canonical_decoder_sha256"] != canonical_sha:
        raise RuntimeError("canonical SHA mismatch")
    if contract["D1_export_sha256"] != export_sha:
        raise RuntimeError("D1 export SHA differs from R2 contract")
    if d1_lock["export_sha256"] != export_sha:
        raise RuntimeError("D1 lock mismatch")
    if r1_lock["report_sha256"] != sha256_file(r1_report_path):
        raise RuntimeError("R1 report/lock mismatch")

    chunks, prefix_stop = contiguous_r2_prefix(
        r2_working,
        candidate_sha,
        export_sha,
    )
    if prefix_stop <= 0 or prefix_stop >= EXPECTED_PREFIX_ITEMS:
        raise RuntimeError(f"unexpected R2 prefix={prefix_stop}")

    start = prefix_stop
    stop = min(start + CHUNK_SIZE, EXPECTED_PREFIX_ITEMS)
    canonical_manifest, canonical_chunk = load_canonical_chunk(
        d4_working,
        start,
        stop,
        export_sha,
        canonical_sha,
    )

    with np.load(export_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    d4 = import_source(d4_script, "_v5_p3_r2_r1_d4")
    canonical = import_source(canonical_path, "_v5_p3_r2_r1_canonical")
    fallback = import_source(candidate_path, "_v5_p3_r2_r1_fallback")

    print(f"scanning_first_incomplete_R2_chunk={start}:{stop}", flush=True)
    failure = None
    scan = []
    for index in range(start, stop):
        result = decode(d4, fallback, arrays, index)
        scan.append(result)
        if result["status"] == "FAIL":
            failure = result
            print(
                f"isolated_failure_index={index} exception={result['exception']}",
                flush=True,
            )
            break
        if index == start or (index - start + 1) % 20 == 0:
            print(f"scan_index={index} status=PASS", flush=True)

    if failure is None:
        raise RuntimeError("did not reproduce failure in incomplete R2 chunk")

    failure_index = int(failure["dataset_index"])
    boundary = parse_boundary(failure["exception"])
    repeats = [decode(d4, fallback, arrays, failure_index) for _ in range(3)]
    deterministic = all(
        row["status"] == "FAIL" and row["exception"] == failure["exception"]
        for row in repeats
    )

    stored = stored_semantic(
        canonical_chunk,
        failure_index - start,
        failure_index,
    )
    canonical_result = decode(d4, canonical, arrays, failure_index)
    canonical_matches = semantic_matches(canonical_result, stored)

    hybrid_failure = hybrid_decode(
        d4,
        canonical,
        fallback,
        arrays,
        failure_index,
    )
    hybrid_failure_matches = (
        hybrid_failure["status"] == "PASS"
        and semantic_matches(hybrid_failure["semantic"], stored)
    )

    chunk_rows = []
    chunk_all_match = True
    fallback_uses = 0
    for index in range(start, stop):
        hybrid = hybrid_decode(d4, canonical, fallback, arrays, index)
        stored_row = stored_semantic(canonical_chunk, index - start, index)
        match = (
            hybrid["status"] == "PASS"
            and semantic_matches(hybrid["semantic"], stored_row)
        )
        chunk_all_match &= match
        fallback_uses += int(
            hybrid.get("selected_path") == "certified_fallback_presolve_false"
        )
        chunk_rows.append(
            {
                "dataset_index": index,
                "selected_path": hybrid.get("selected_path"),
                "matches_stored_canonical": match,
                "canonical_exception": (
                    hybrid.get("canonical_result", {}).get("exception")
                ),
                "fallback_exception": (
                    (hybrid.get("fallback_result") or {}).get("exception")
                ),
            }
        )

    r1_index = int(r1_lock["isolated_failure_index"])
    r1_hybrid = hybrid_decode(d4, canonical, fallback, arrays, r1_index)
    selected_r1 = r1_report["candidate_audit"]["candidate_results"][
        "presolve_false"
    ]["failure_item_result"]["semantic"]

    if r1_hybrid["status"] == "PASS":
        sem = r1_hybrid["semantic"]
        observed_r1 = {
            "margin": {"value": sem["margin"], "hex": sem["margin_hex"]},
            "attacker_count": sem["count"],
            "route_ids": sem["route_ids"],
            "source_mask": mask_int(sem["source"]),
            "transit_mask": mask_int(sem["transit"]),
            "victim_mask": mask_int(sem["victim"]),
            "path_mask": mask_int(sem["path"]),
        }
    else:
        observed_r1 = None

    r1_matches = (
        r1_hybrid["status"] == "PASS"
        and r1_hybrid.get("selected_path")
        == "certified_fallback_presolve_false"
        and observed_r1 == selected_r1
    )

    status = "PASS" if all(
        [
            deterministic,
            boundary.get("parsed") is True,
            canonical_matches,
            hybrid_failure_matches,
            chunk_all_match,
            r1_matches,
        ]
    ) else "HOLD"

    atomic_json(
        isolated_path,
        {
            "stage": STAGE,
            "dataset_index": failure_index,
            "failure": failure,
            "repeats": repeats,
            "integrality_boundary": boundary,
            "canonical_result": canonical_result,
            "stored_canonical": stored,
            "canonical_matches_stored": canonical_matches,
            "hybrid_result": hybrid_failure,
            "hybrid_matches_stored": hybrid_failure_matches,
            "labels": {
                "graph": int(arrays["y_attack"][failure_index]),
                "attacker_count": int(
                    arrays["y_attacker_count"][failure_index]
                ),
            },
        },
    )

    next_stage = (
        "V5_P3_D4_R2_R2_CANONICAL_FIRST_CERTIFIED_PRESOLVE_FALSE_FALLBACK_FREEZE"
        if status == "PASS"
        else "D4_R2_R1_HOLD_DEEPER_NUMERICAL_POLICY_ANALYSIS"
    )

    report = {
        "stage": STAGE,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "R2_prefix": {
            "hash_valid_chunks": len(chunks),
            "hash_valid_items": prefix_stop,
            "preserved": True,
            "first_incomplete_chunk": [start, stop],
        },
        "isolated_failure": {
            "dataset_index": failure_index,
            "deterministic": deterministic,
            "exception": failure["exception"],
            "integrality_boundary": boundary,
            "artifact": str(isolated_path),
            "artifact_sha256": sha256_file(isolated_path),
        },
        "canonical_reference": {
            "same_item_succeeds": canonical_result["status"] == "PASS",
            "matches_stored_D4_output": canonical_matches,
        },
        "hybrid_policy_audit": {
            "policy": (
                "canonical presolve=True first; on ExactDecoderError only, "
                "retry presolve=False; return only a fully certified result"
            ),
            "failure_item_selected_path": hybrid_failure.get("selected_path"),
            "failure_item_matches_stored": hybrid_failure_matches,
            "affected_100_row_chunk_exact_match": chunk_all_match,
            "fallback_uses_in_affected_chunk": fallback_uses,
            "affected_chunk_rows": chunk_rows,
            "R1_failure_index": r1_index,
            "R1_selected_path": r1_hybrid.get("selected_path"),
            "R1_fallback_matches_R1_semantic": r1_matches,
        },
        "root_cause_disposition": (
            "GLOBAL_PRESOLVE_FALSE_IS_NOT_NUMERICALLY_DOMINANT; "
            "PRESOLVE_TRUE_AND_FALSE_HAVE_COMPLEMENTARY_CERTIFIED_FAILURE_MODES"
        ),
        "decision": {
            "global_presolve_false_candidate_rejected": True,
            "canonical_first_certified_fallback_freeze_authorized": (
                status == "PASS"
            ),
            "integrality_tolerance_relaxation_authorized": False,
            "threshold_or_margin_retuning_authorized": False,
            "beam_decoder_selected": False,
            "sealed_test_evaluation_authorized": False,
            "next_stage": next_stage,
        },
        "sealed_test": {
            "test_dataset_instantiated": False,
            "test_length_computed": False,
            "test_tensor_loaded": False,
        },
        "provenance": {
            "R2_contract_sha256": sha256_file(contract_path),
            "R2_candidate_sha256": candidate_sha,
            "canonical_decoder_sha256": canonical_sha,
            "D1_export_sha256": export_sha,
            "canonical_chunk_sha256": canonical_manifest["chunk_sha256"],
            "R1_report_sha256": sha256_file(r1_report_path),
            "installed_script_sha256": sha256_file(installed_script),
        },
        "repo_decoder_modified": False,
        "D4_chunks_modified": False,
        "R2_chunks_modified": False,
        "threshold_tuning_performed": False,
        "generalization_claim_authorized": False,
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": status,
        "report_sha256": sha256_file(report_path),
        "isolated_artifact_sha256": sha256_file(isolated_path),
        "R2_prefix_items": prefix_stop,
        "failure_index": failure_index,
        "canonical_matches_stored": canonical_matches,
        "affected_chunk_exact_match": chunk_all_match,
        "R1_fallback_matches": r1_matches,
        "test_tensor_loaded": False,
    }
    atomic_json(lock_path, lock)

    marker = output_dir / f"{STAGE}_{'COMPLETE' if status == 'PASS' else 'HOLD'}"
    marker.write_text(marker.name + "\n")

    print(f"{STAGE}_{'COMPLETE' if status == 'PASS' else 'HOLD'}")
    print(f"status={status}")
    print(f"hash_valid_R2_prefix_chunks={len(chunks)}")
    print(f"hash_valid_R2_prefix_items={prefix_stop}")
    print(f"isolated_failure_index={failure_index}")
    print(f"presolve_false_failure_deterministic={deterministic}")
    print(f"integrality_maximum_error={boundary.get('maximum_error')}")
    print(f"integrality_tolerance={boundary.get('tolerance')}")
    print(f"integrality_absolute_excess={boundary.get('absolute_excess')}")
    print(f"canonical_same_item_succeeds={canonical_result['status'] == 'PASS'}")
    print(f"canonical_matches_stored_D4_output={canonical_matches}")
    print(
        "hybrid_failure_item_selected_path="
        f"{hybrid_failure.get('selected_path')}"
    )
    print(f"hybrid_full_100_row_chunk_exact_match={chunk_all_match}")
    print(
        "R1_failure_hybrid_selected_path="
        f"{r1_hybrid.get('selected_path')}"
    )
    print(f"R1_failure_fallback_matches_R1_semantic={r1_matches}")
    print("integrality_tolerance_relaxation_authorized=false")
    print("repo_decoder_modified=false")
    print("D4_chunks_modified=false")
    print("R2_chunks_modified=false")
    print("test_tensor_loaded=false")
    print(f"next_stage={next_stage}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
