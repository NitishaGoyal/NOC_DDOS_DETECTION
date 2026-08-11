from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = "V5_P3_D4_R2_PROSPECTIVE_NUMERICAL_IMPLEMENTATION_FREEZE_AND_FULL_REGRESSION"
CAMPAIGN_LABEL = "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic"
EXPECTED_ITEMS = 13_863
EXPECTED_PREFIX_ITEMS = 11_700
CHUNK_SIZE = 100
A1_MARGIN_THRESHOLD = 8.7205320882398425
SELECTED_CANDIDATE = "presolve_false"


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


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def import_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def float_bits(value: float) -> int:
    return int(np.asarray([value], dtype=np.float64).view(np.uint64)[0])


def semantic_from_extracted(extracted: dict[str, Any]) -> dict[str, Any]:
    return {
        "margin": {
            "value": float(extracted["margin"]),
            "hex": float(extracted["margin"]).hex(),
        },
        "attacker_count": int(extracted["count"]),
        "route_ids": [int(item) for item in extracted["route_ids"]],
        "source_mask": int(
            sum(
                int(bit) << index
                for index, bit in enumerate(extracted["source"])
            )
        ),
        "transit_mask": int(
            sum(
                int(bit) << index
                for index, bit in enumerate(extracted["transit"])
            )
        ),
        "victim_mask": int(
            sum(
                int(bit) << index
                for index, bit in enumerate(extracted["victim"])
            )
        ),
        "path_mask": int(
            sum(
                int(bit) << index
                for index, bit in enumerate(extracted["path"])
            )
        ),
    }


def load_valid_canonical_chunk(
    manifest_path: Path,
    export_sha256: str,
    decoder_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_path = Path(manifest["chunk_path"])
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError(f"canonical chunk not complete: {manifest_path}")
    if not chunk_path.is_file():
        raise RuntimeError(f"canonical chunk missing: {chunk_path}")
    if manifest.get("chunk_sha256") != sha256_file(chunk_path):
        raise RuntimeError(f"canonical chunk SHA mismatch: {chunk_path}")
    if manifest.get("D1_export_sha256") != export_sha256:
        raise RuntimeError(f"canonical chunk export SHA mismatch: {chunk_path}")
    if manifest.get("certified_decoder_sha256") != decoder_sha256:
        raise RuntimeError(f"canonical chunk decoder SHA mismatch: {chunk_path}")
    if float(manifest.get("A1_margin_threshold")) != A1_MARGIN_THRESHOLD:
        raise RuntimeError(f"canonical chunk margin mismatch: {chunk_path}")
    with np.load(chunk_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    return manifest, arrays


def regression_paths(
    working_dir: Path,
    chunk_id: int,
    start: int,
    stop: int,
) -> tuple[Path, Path]:
    stem = f"chunk_{chunk_id:04d}_items_{start:05d}_{stop:05d}"
    return (
        working_dir / f"{stem}.npz",
        working_dir / f"{stem}.manifest.json",
    )


def validate_regression_chunk(
    candidate_path: Path,
    manifest_path: Path,
    start: int,
    stop: int,
    frozen_candidate_sha256: str,
    export_sha256: str,
    canonical_chunk_sha256: str,
) -> dict[str, np.ndarray]:
    if not candidate_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError(f"regression manifest not complete: {manifest_path}")
    if int(manifest.get("start", -1)) != start:
        raise RuntimeError(f"regression start changed: {manifest_path}")
    if int(manifest.get("stop", -1)) != stop:
        raise RuntimeError(f"regression stop changed: {manifest_path}")
    if manifest.get("candidate_sha256") != frozen_candidate_sha256:
        raise RuntimeError(f"candidate hash changed: {manifest_path}")
    if manifest.get("D1_export_sha256") != export_sha256:
        raise RuntimeError(f"export hash changed: {manifest_path}")
    if manifest.get("canonical_chunk_sha256") != canonical_chunk_sha256:
        raise RuntimeError(f"canonical reference changed: {manifest_path}")
    if manifest.get("candidate_chunk_sha256") != sha256_file(candidate_path):
        raise RuntimeError(f"candidate chunk SHA mismatch: {candidate_path}")
    with np.load(candidate_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    expected_items = stop - start
    expected_shapes = {
        "dataset_index": (expected_items,),
        "graph_prediction": (expected_items,),
        "count_prediction": (expected_items,),
        "source_prediction": (expected_items, 16),
        "transit_prediction": (expected_items, 16),
        "victim_prediction": (expected_items, 16),
        "path_prediction": (expected_items, 16),
        "route_count": (expected_items,),
        "margin": (expected_items,),
        "decode_seconds": (expected_items,),
        "exact_match": (expected_items,),
    }
    for key, shape in expected_shapes.items():
        if key not in arrays or arrays[key].shape != shape:
            raise RuntimeError(
                f"regression chunk {candidate_path} has invalid {key}"
            )
    if not np.array_equal(
        arrays["dataset_index"],
        np.arange(start, stop, dtype=np.int32),
    ):
        raise RuntimeError(f"regression coverage changed: {candidate_path}")
    if not np.all(arrays["exact_match"] == 1):
        raise RuntimeError(f"regression mismatch recorded: {candidate_path}")
    return arrays


def main() -> int:
    args = parse_args()

    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    working_dir = output_dir / "FULL_PREFIX_REGRESSION_WORKING"
    working_dir.mkdir(parents=True, exist_ok=True)
    frozen_candidate_path = output_dir / (
        "V5_P3_A1_EXACT_DECODER_PRESOLVE_FALSE_FROZEN.py"
    )
    contract_path = output_dir / (
        "V5_P3_D4_R2_NUMERICAL_IMPLEMENTATION_FREEZE_CONTRACT.json"
    )
    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"
    hold_path = output_dir / f"{STAGE}_HOLD"

    r1_dir = (
        repo
        / "reports/v5/p3_d4_r1_a1_lexicographic_infeasibility_root_cause_audit"
    )
    r1_report_path = r1_dir / (
        "V5_P3_D4_R1_A1_LEXICOGRAPHIC_INFEASIBILITY_ROOT_CAUSE_AUDIT_REPORT.json"
    )
    r1_lock_path = r1_dir / (
        "V5_P3_D4_R1_A1_LEXICOGRAPHIC_INFEASIBILITY_ROOT_CAUSE_AUDIT_LOCK.json"
    )
    d4_dir = (
        repo
        / "reports/v5/p3_d4_resumable_a1_exact_validation_evaluation"
    )
    d4_working_dir = d4_dir / "A1_WORKING"
    d4_script_path = (
        repo
        / "scripts/v5/p3/run_v5_p3_d4_resumable_a1_exact_validation_evaluation.py"
    )
    d1_dir = (
        repo
        / "reports/v5/p3_d1_immutable_validation_logit_export"
    )
    d1_export_path = d1_dir / (
        "V5_P3_D1_TRANCHE_A_VALIDATION_LOGITS_AND_LABELS.npz"
    )
    d1_lock_path = d1_dir / (
        "V5_P3_D1_IMMUTABLE_TRANCHE_A_VALIDATION_LOGIT_EXPORT_LOCK.json"
    )
    canonical_decoder_path = (
        repo
        / "src/decoders/v5_legal_xy_exact_decoder_certified.py"
    )
    installed_candidate_path = (
        repo
        / "src/decoders/v5_p3_a1_exact_decoder_presolve_false.py"
    )

    required = [
        r1_report_path,
        r1_lock_path,
        d4_working_dir,
        d4_script_path,
        d1_export_path,
        d1_lock_path,
        canonical_decoder_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")

    r1_report = json.loads(r1_report_path.read_text(encoding="utf-8"))
    r1_lock = json.loads(r1_lock_path.read_text(encoding="utf-8"))
    d1_lock = json.loads(d1_lock_path.read_text(encoding="utf-8"))

    if r1_report.get("status") != "PASS":
        raise RuntimeError("R1 report is not PASS")
    if r1_lock.get("report_sha256") != sha256_file(r1_report_path):
        raise RuntimeError("R1 report/lock SHA mismatch")
    decision = r1_report.get("decision", {})
    if not decision.get("numerical_candidate_freeze_authorized"):
        raise RuntimeError("R1 did not authorize a numerical candidate freeze")
    candidate_audit = r1_report["candidate_audit"]
    if SELECTED_CANDIDATE not in candidate_audit["viable_candidates"]:
        raise RuntimeError("minimal presolve_false candidate is not viable")
    if not candidate_audit[
        "viable_candidates_agree_on_failure_semantics"
    ]:
        raise RuntimeError("R1 viable candidates disagree semantically")

    selected_result = candidate_audit["candidate_results"][
        SELECTED_CANDIDATE
    ]
    if not selected_result["failure_item_succeeds"]:
        raise RuntimeError("selected candidate does not solve the failure item")
    if not selected_result["all_control_semantics_match"]:
        raise RuntimeError("selected candidate changes an R1 control")
    candidate_source_path = Path(selected_result["source_path"])
    if not candidate_source_path.is_file():
        raise RuntimeError(f"R1 candidate source missing: {candidate_source_path}")
    if sha256_file(candidate_source_path) != selected_result["source_sha256"]:
        raise RuntimeError("R1 candidate source SHA mismatch")

    canonical_sha = sha256_file(canonical_decoder_path)
    export_sha = sha256_file(d1_export_path)
    if d1_lock.get("export_sha256") != export_sha:
        raise RuntimeError("D1 export SHA mismatch")
    if r1_lock.get("canonical_decoder_sha256") != canonical_sha:
        raise RuntimeError("canonical decoder differs from R1")
    if int(r1_lock.get("hash_valid_D4_prefix_items", -1)) != EXPECTED_PREFIX_ITEMS:
        raise RuntimeError("R1 prefix length is not 11,700")

    candidate_bytes = candidate_source_path.read_bytes()
    if frozen_candidate_path.is_file():
        if frozen_candidate_path.read_bytes() != candidate_bytes:
            raise RuntimeError("existing frozen candidate differs from R1")
    else:
        temporary = frozen_candidate_path.with_suffix(".py.tmp")
        temporary.write_bytes(candidate_bytes)
        os.replace(temporary, frozen_candidate_path)
    frozen_candidate_sha = sha256_file(frozen_candidate_path)

    contract = {
        "stage": STAGE,
        "campaign_label": CAMPAIGN_LABEL,
        "selected_candidate": SELECTED_CANDIDATE,
        "selection_reason": (
            "Minimal one-line numerical change: disable HiGHS presolve. "
            "R1 showed it recovers the deterministic failure and exactly "
            "matches all controls; the larger combined candidate is unnecessary."
        ),
        "canonical_decoder_sha256": canonical_sha,
        "frozen_candidate_sha256": frozen_candidate_sha,
        "candidate_change": {
            "solver_option_presolve": False,
            "semantic_tie_tolerance_changed": False,
            "objective_changed": False,
            "route_model_changed": False,
            "margin_changed": False,
            "threshold_changed": False,
        },
        "full_regression_prefix_items": EXPECTED_PREFIX_ITEMS,
        "chunk_size": CHUNK_SIZE,
        "D1_export_sha256": export_sha,
        "isolated_failure_index": int(r1_lock["isolated_failure_index"]),
        "test_accessed": False,
    }
    if contract_path.is_file():
        existing_contract = json.loads(
            contract_path.read_text(encoding="utf-8")
        )
        if existing_contract != contract:
            raise RuntimeError("R2 freeze contract changed")
    else:
        atomic_json(contract_path, contract)

    # The frozen candidate is executed from the report directory but imports
    # repository modules through absolute names such as `src.decoders...`.
    # Add the resolved repository root before executing either source module.
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    d4_module = import_source(
        d4_script_path,
        "_v5_p3_d4_r2_d4_semantics",
    )
    candidate = import_source(
        frozen_candidate_path,
        "_v5_p3_d4_r2_frozen_candidate",
    )

    with np.load(d1_export_path, allow_pickle=False) as loaded:
        source_arrays = {key: loaded[key].copy() for key in loaded.files}

    canonical_manifests = sorted(
        d4_working_dir.glob("chunk_*.manifest.json")
    )
    prefix_manifests = []
    contiguous_stop = 0
    for manifest_path in canonical_manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        start = int(manifest.get("start", -1))
        stop = int(manifest.get("stop", -1))
        if start != contiguous_stop:
            break
        if stop > EXPECTED_PREFIX_ITEMS:
            break
        prefix_manifests.append(manifest_path)
        contiguous_stop = stop
    if contiguous_stop != EXPECTED_PREFIX_ITEMS:
        raise RuntimeError(
            f"canonical prefix stops at {contiguous_stop}, expected 11700"
        )

    decoded_chunks_this_run = 0
    reused_chunks_this_run = 0
    all_regression_chunks = []
    started_wall = time.time()

    for chunk_id, canonical_manifest_path in enumerate(prefix_manifests):
        canonical_manifest, canonical_chunk = load_valid_canonical_chunk(
            canonical_manifest_path,
            export_sha,
            canonical_sha,
        )
        start = int(canonical_manifest["start"])
        stop = int(canonical_manifest["stop"])
        canonical_chunk_sha = canonical_manifest["chunk_sha256"]
        candidate_chunk_path, candidate_manifest_path = regression_paths(
            working_dir,
            chunk_id,
            start,
            stop,
        )
        try:
            candidate_chunk = validate_regression_chunk(
                candidate_chunk_path,
                candidate_manifest_path,
                start,
                stop,
                frozen_candidate_sha,
                export_sha,
                canonical_chunk_sha,
            )
        except FileNotFoundError:
            print(
                f"R2_chunk={chunk_id + 1}/{len(prefix_manifests)} "
                f"status=REGRESSING items={start}:{stop}",
                flush=True,
            )
            item_count = stop - start
            candidate_chunk = {
                "dataset_index": np.arange(start, stop, dtype=np.int32),
                "graph_prediction": np.empty(item_count, dtype=np.uint8),
                "count_prediction": np.empty(item_count, dtype=np.int8),
                "source_prediction": np.empty((item_count, 16), dtype=np.uint8),
                "transit_prediction": np.empty((item_count, 16), dtype=np.uint8),
                "victim_prediction": np.empty((item_count, 16), dtype=np.uint8),
                "path_prediction": np.empty((item_count, 16), dtype=np.uint8),
                "route_count": np.empty(item_count, dtype=np.int8),
                "margin": np.empty(item_count, dtype=np.float64),
                "decode_seconds": np.empty(item_count, dtype=np.float64),
                "exact_match": np.empty(item_count, dtype=np.uint8),
            }

            for local, dataset_index in enumerate(range(start, stop)):
                t0 = time.perf_counter()
                hypothesis = candidate.decode_best_attack_hypothesis(
                    float(source_arrays["attack_logits"][dataset_index]),
                    source_arrays["count_logits"][dataset_index].astype(np.float64),
                    source_arrays["source_logits"][dataset_index].astype(np.float64),
                    source_arrays["transit_logits"][dataset_index].astype(np.float64),
                    source_arrays["victim_logits"][dataset_index].astype(np.float64),
                    source_arrays["path_logits"][dataset_index].astype(np.float64),
                )
                decoded = candidate.apply_margin_threshold(
                    hypothesis,
                    A1_MARGIN_THRESHOLD,
                )
                extracted, _ = d4_module.extract_decoded_output(
                    decoded,
                    hypothesis,
                )
                elapsed = time.perf_counter() - t0

                candidate_chunk["graph_prediction"][local] = extracted["graph"]
                candidate_chunk["count_prediction"][local] = extracted["count"]
                for role in ("source", "transit", "victim", "path"):
                    candidate_chunk[f"{role}_prediction"][local] = extracted[role]
                candidate_chunk["route_count"][local] = int(
                    extracted["route_ids"].size
                )
                candidate_chunk["margin"][local] = float(extracted["margin"])
                candidate_chunk["decode_seconds"][local] = elapsed

                exact = (
                    int(extracted["graph"])
                    == int(canonical_chunk["graph_prediction"][local])
                    and int(extracted["count"])
                    == int(canonical_chunk["count_prediction"][local])
                    and np.array_equal(
                        extracted["source"],
                        canonical_chunk["source_prediction"][local],
                    )
                    and np.array_equal(
                        extracted["transit"],
                        canonical_chunk["transit_prediction"][local],
                    )
                    and np.array_equal(
                        extracted["victim"],
                        canonical_chunk["victim_prediction"][local],
                    )
                    and np.array_equal(
                        extracted["path"],
                        canonical_chunk["path_prediction"][local],
                    )
                    and int(extracted["route_ids"].size)
                    == int(canonical_chunk["route_count"][local])
                    and float_bits(float(extracted["margin"]))
                    == float_bits(float(canonical_chunk["margin"][local]))
                )
                candidate_chunk["exact_match"][local] = np.uint8(exact)
                if not exact:
                    raise RuntimeError(
                        "candidate semantic mismatch at "
                        f"dataset_index={dataset_index}"
                    )

                if (
                    local == 0
                    or (local + 1) % 25 == 0
                    or local + 1 == item_count
                ):
                    print(
                        f"R2_chunk_item={local + 1}/{item_count} "
                        f"dataset_index={dataset_index} "
                        f"seconds={elapsed:.6f}",
                        flush=True,
                    )

            atomic_npz(candidate_chunk_path, candidate_chunk)
            candidate_manifest = {
                "stage": STAGE,
                "status": "COMPLETE",
                "chunk_id": chunk_id,
                "start": start,
                "stop": stop,
                "candidate_chunk_path": str(candidate_chunk_path),
                "candidate_chunk_sha256": sha256_file(candidate_chunk_path),
                "candidate_sha256": frozen_candidate_sha,
                "D1_export_sha256": export_sha,
                "canonical_chunk_sha256": canonical_chunk_sha,
                "all_items_exact_match": True,
            }
            atomic_json(candidate_manifest_path, candidate_manifest)
            candidate_chunk = validate_regression_chunk(
                candidate_chunk_path,
                candidate_manifest_path,
                start,
                stop,
                frozen_candidate_sha,
                export_sha,
                canonical_chunk_sha,
            )
            decoded_chunks_this_run += 1
            print(
                f"R2_chunk={chunk_id + 1}/{len(prefix_manifests)} "
                f"status=COMMITTED",
                flush=True,
            )
        else:
            reused_chunks_this_run += 1
            print(
                f"R2_chunk={chunk_id + 1}/{len(prefix_manifests)} "
                f"status=REUSED",
                flush=True,
            )
        all_regression_chunks.append(candidate_chunk)

    merged = {
        key: np.concatenate([chunk[key] for chunk in all_regression_chunks])
        for key in all_regression_chunks[0]
    }
    if merged["dataset_index"].shape != (EXPECTED_PREFIX_ITEMS,):
        raise RuntimeError("merged R2 prefix length mismatch")
    if not np.all(merged["exact_match"] == 1):
        raise RuntimeError("merged R2 regression has mismatches")

    failure_index = int(r1_lock["isolated_failure_index"])
    failure_started = time.perf_counter()
    failure_hypothesis = candidate.decode_best_attack_hypothesis(
        float(source_arrays["attack_logits"][failure_index]),
        source_arrays["count_logits"][failure_index].astype(np.float64),
        source_arrays["source_logits"][failure_index].astype(np.float64),
        source_arrays["transit_logits"][failure_index].astype(np.float64),
        source_arrays["victim_logits"][failure_index].astype(np.float64),
        source_arrays["path_logits"][failure_index].astype(np.float64),
    )
    failure_elapsed = time.perf_counter() - failure_started
    failure_semantic = {
        "margin": {
            "value": float(failure_hypothesis.margin),
            "hex": float(failure_hypothesis.margin).hex(),
        },
        "attacker_count": int(failure_hypothesis.attacker_count),
        "route_ids": [int(item) for item in failure_hypothesis.route_ids],
        "source_mask": int(failure_hypothesis.source_mask),
        "transit_mask": int(failure_hypothesis.transit_mask),
        "victim_mask": int(failure_hypothesis.victim_mask),
        "path_mask": int(failure_hypothesis.path_mask),
    }
    expected_failure_semantic = selected_result["failure_item_result"]["semantic"]
    if failure_semantic != expected_failure_semantic:
        raise RuntimeError(
            "frozen candidate no longer reproduces the R1 failure recovery"
        )

    # Install only after the complete prefix regression and failure recovery.
    installed_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    if installed_candidate_path.is_file():
        if installed_candidate_path.read_bytes() != frozen_candidate_path.read_bytes():
            raise RuntimeError(
                "existing installed P3 numerical decoder differs from R2 freeze"
            )
    else:
        temporary = installed_candidate_path.with_suffix(".py.tmp")
        shutil.copyfile(frozen_candidate_path, temporary)
        os.replace(temporary, installed_candidate_path)
    installed_sha = sha256_file(installed_candidate_path)
    if installed_sha != frozen_candidate_sha:
        raise RuntimeError("installed candidate SHA mismatch")

    timing = merged["decode_seconds"].astype(np.float64)
    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "audit_script_revision": "v2_repository_import_bootstrap",
        "scope": (
            "Prospective freeze of the minimal presolve-disabled P3 exact "
            "decoder and exact semantic regression against all 11,700 "
            "previously successful canonical D4 validation outputs."
        ),
        "selection": {
            "candidate": SELECTED_CANDIDATE,
            "reason": contract["selection_reason"],
            "canonical_decoder_modified": False,
            "new_P3_decoder_path": str(installed_candidate_path),
            "new_P3_decoder_sha256": installed_sha,
            "frozen_report_copy": str(frozen_candidate_path),
            "frozen_report_copy_sha256": frozen_candidate_sha,
        },
        "full_prefix_regression": {
            "items": EXPECTED_PREFIX_ITEMS,
            "chunks": len(prefix_manifests),
            "decoded_chunks_this_run": decoded_chunks_this_run,
            "reused_chunks_this_run": reused_chunks_this_run,
            "exact_semantic_match_items": int(merged["exact_match"].sum()),
            "mismatch_items": int(np.sum(merged["exact_match"] == 0)),
            "graph_count_masks_route_count_margin_bit_exact": True,
            "candidate_mean_decode_seconds": float(timing.mean()),
            "candidate_p95_decode_seconds": float(
                np.quantile(timing, 0.95)
            ),
            "wall_seconds_this_run": time.time() - started_wall,
        },
        "failure_recovery": {
            "dataset_index": failure_index,
            "recovered": True,
            "semantic_matches_R1": True,
            "semantic": failure_semantic,
            "decode_seconds": failure_elapsed,
        },
        "freeze_contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
            "solver_presolve": False,
            "semantic_tie_tolerance_changed": False,
            "objective_changed": False,
            "route_model_changed": False,
            "A1_margin_threshold": A1_MARGIN_THRESHOLD,
            "margin_retuned": False,
        },
        "decision": {
            "prospective_P3_numerical_decoder_frozen": True,
            "existing_11700_D4_outputs_reusable": True,
            "D4_resume_with_P3_decoder_authorized": True,
            "canonical_P2_decoder_replaced": False,
            "threshold_or_margin_retuning_authorized": False,
            "beam_decoder_selected": False,
            "sealed_test_evaluation_authorized": False,
            "next_stage": (
                "V5_P3_D4_R3_RESUME_A1_EXACT_VALIDATION_WITH_"
                "FROZEN_PRESOLVE_FALSE_DECODER"
            ),
        },
        "sealed_test": {
            "test_dataset_instantiated": False,
            "test_length_computed": False,
            "test_tensor_loaded": False,
            "evaluation_authorized": False,
        },
        "provenance": {
            "R1_report_sha256": sha256_file(r1_report_path),
            "R1_lock_sha256": sha256_file(r1_lock_path),
            "D1_export_sha256": export_sha,
            "canonical_decoder_sha256": canonical_sha,
            "installed_script_sha256": sha256_file(installed_script),
        },
        "model_replayed": False,
        "model_trained": False,
        "threshold_tuning_performed": False,
        "certified_dataset_modified": False,
        "generalization_claim_authorized": False,
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(report_path),
        "freeze_contract_sha256": sha256_file(contract_path),
        "canonical_decoder_sha256": canonical_sha,
        "frozen_P3_decoder_sha256": frozen_candidate_sha,
        "installed_P3_decoder_sha256": installed_sha,
        "D1_export_sha256": export_sha,
        "regressed_prefix_items": EXPECTED_PREFIX_ITEMS,
        "exact_semantic_match_items": int(merged["exact_match"].sum()),
        "failure_index": failure_index,
        "failure_recovered": True,
        "test_tensor_loaded": False,
        "certified_dataset_modified": False,
    }
    atomic_json(lock_path, lock)

    complete_path.write_text(f"{STAGE}_COMPLETE\n", encoding="utf-8")
    if hold_path.exists():
        hold_path.unlink()

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print("audit_script_revision=v2_repository_import_bootstrap")
    print(f"selected_candidate={SELECTED_CANDIDATE}")
    print("selection_reason=minimal_presolve_false_change")
    print(f"regressed_prefix_items={EXPECTED_PREFIX_ITEMS}")
    print(f"regression_chunks={len(prefix_manifests)}")
    print(
        "exact_semantic_match_items="
        f"{int(merged['exact_match'].sum())}"
    )
    print("mismatch_items=0")
    print(f"failure_index={failure_index}")
    print("failure_recovered=true")
    print("failure_semantic_matches_R1=true")
    print("canonical_P2_decoder_modified=false")
    print(f"installed_P3_decoder={installed_candidate_path}")
    print(f"installed_P3_decoder_sha256={installed_sha}")
    print(
        "candidate_mean_decode_seconds="
        f"{float(timing.mean()):.6f}"
    )
    print(
        "candidate_p95_decode_seconds="
        f"{float(np.quantile(timing, 0.95)):.6f}"
    )
    print("threshold_or_margin_retuning_authorized=false")
    print("beam_decoder_selected=false")
    print("test_tensor_loaded=false")
    print("existing_11700_D4_outputs_reusable=true")
    print("D4_resume_with_P3_decoder_authorized=true")
    print(
        "next_stage="
        "V5_P3_D4_R3_RESUME_A1_EXACT_VALIDATION_WITH_"
        "FROZEN_PRESOLVE_FALSE_DECODER"
    )
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
