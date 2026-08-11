from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = (
    "V5_P3_D4_R2_R2_CANONICAL_FIRST_CERTIFIED_"
    "PRESOLVE_FALSE_FALLBACK_FREEZE"
)
A1_MARGIN_THRESHOLD = 8.7205320882398425
AFFECTED_START = 10_600
AFFECTED_STOP = 10_700
R1_FAILURE_INDEX = 11_760


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package-dir", required=True)
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


def stored_semantic(chunk, local: int) -> dict[str, Any]:
    margin = float(chunk["margin"][local])
    return {
        "graph": int(chunk["graph_prediction"][local]),
        "count": int(chunk["count_prediction"][local]),
        "source": chunk["source_prediction"][local].astype(int).tolist(),
        "transit": chunk["transit_prediction"][local].astype(int).tolist(),
        "victim": chunk["victim_prediction"][local].astype(int).tolist(),
        "path": chunk["path_prediction"][local].astype(int).tolist(),
        "route_count": int(chunk["route_count"][local]),
        "margin_bits": float_bits(margin),
    }


def decode_semantic(d4, decoder, arrays, index: int) -> dict[str, Any]:
    started = time.perf_counter()
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
        "graph": int(extracted["graph"]),
        "count": int(extracted["count"]),
        "source": extracted["source"].astype(int).tolist(),
        "transit": extracted["transit"].astype(int).tolist(),
        "victim": extracted["victim"].astype(int).tolist(),
        "path": extracted["path"].astype(int).tolist(),
        "route_count": int(extracted["route_ids"].size),
        "route_ids": [int(value) for value in extracted["route_ids"]],
        "margin": margin,
        "margin_hex": margin.hex(),
        "margin_bits": float_bits(margin),
        "selected_path": getattr(
            hypothesis,
            "selected_path",
            "unwrapped",
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def compare_stored(observed, expected) -> bool:
    keys = (
        "graph",
        "count",
        "source",
        "transit",
        "victim",
        "path",
        "route_count",
        "margin_bits",
    )
    return all(observed[key] == expected[key] for key in keys)


def mask_int(bits) -> int:
    return int(sum(int(bit) << index for index, bit in enumerate(bits)))


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    package_dir = Path(args.package_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    contract_path = output_dir / (
        "V5_P3_A1_HYBRID_NUMERICAL_POLICY_CONTRACT.json"
    )
    frozen_wrapper_path = output_dir / (
        "v5_p3_a1_exact_decoder_certified_hybrid.py"
    )
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    r2r1_dir = (
        repo
        / "reports/v5/p3_d4_r2_r1_integrality_boundary_and_hybrid_retry_audit"
    )
    r2r1_report_path = r2r1_dir / (
        "V5_P3_D4_R2_R1_PRESOLVE_FALSE_INTEGRALITY_BOUNDARY_"
        "AND_HYBRID_RETRY_AUDIT_REPORT.json"
    )
    r2r1_lock_path = r2r1_dir / (
        "V5_P3_D4_R2_R1_PRESOLVE_FALSE_INTEGRALITY_BOUNDARY_"
        "AND_HYBRID_RETRY_AUDIT_LOCK.json"
    )

    r2_dir = (
        repo
        / "reports/v5/p3_d4_r2_prospective_numerical_freeze_and_full_regression"
    )
    fallback_source_path = r2_dir / (
        "V5_P3_A1_EXACT_DECODER_PRESOLVE_FALSE_FROZEN.py"
    )
    r2_contract_path = r2_dir / (
        "V5_P3_D4_R2_NUMERICAL_IMPLEMENTATION_FREEZE_CONTRACT.json"
    )

    r1_dir = (
        repo
        / "reports/v5/p3_d4_r1_a1_lexicographic_infeasibility_root_cause_audit"
    )
    r1_report_path = r1_dir / (
        "V5_P3_D4_R1_A1_LEXICOGRAPHIC_INFEASIBILITY_"
        "ROOT_CAUSE_AUDIT_REPORT.json"
    )

    d3_dir = (
        repo
        / "reports/v5/p3_d3_resumable_a1_exact_decoder_chunk_preflight"
    )
    d3_selection_path = d3_dir / (
        "V5_P3_D3_A1_EXACT_PREFLIGHT_SELECTION.json"
    )

    d4_dir = (
        repo
        / "reports/v5/p3_d4_resumable_a1_exact_validation_evaluation"
    )
    d4_working = d4_dir / "A1_WORKING"
    d4_script_path = (
        repo
        / "scripts/v5/p3/"
        "run_v5_p3_d4_resumable_a1_exact_validation_evaluation.py"
    )

    d1_dir = (
        repo
        / "reports/v5/p3_d1_immutable_validation_logit_export"
    )
    export_path = d1_dir / (
        "V5_P3_D1_TRANCHE_A_VALIDATION_LOGITS_AND_LABELS.npz"
    )

    canonical_path = (
        repo
        / "src/decoders/v5_legal_xy_exact_decoder_certified.py"
    )
    installed_fallback_path = (
        repo
        / "src/decoders/v5_p3_a1_exact_decoder_presolve_false.py"
    )
    installed_wrapper_path = (
        repo
        / "src/decoders/v5_p3_a1_exact_decoder_certified_hybrid.py"
    )
    package_wrapper_path = package_dir / (
        "v5_p3_a1_exact_decoder_certified_hybrid.py"
    )

    required = [
        r2r1_report_path,
        r2r1_lock_path,
        fallback_source_path,
        r2_contract_path,
        r1_report_path,
        d3_selection_path,
        d4_working,
        d4_script_path,
        export_path,
        canonical_path,
        package_wrapper_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")

    r2r1_report = json.loads(r2r1_report_path.read_text())
    r2r1_lock = json.loads(r2r1_lock_path.read_text())
    r2_contract = json.loads(r2_contract_path.read_text())
    r1_report = json.loads(r1_report_path.read_text())
    d3_selection = json.loads(d3_selection_path.read_text())

    if r2r1_report.get("status") != "PASS":
        raise RuntimeError("R2-R1 report is not PASS")
    if r2r1_lock.get("report_sha256") != sha256_file(r2r1_report_path):
        raise RuntimeError("R2-R1 report/lock mismatch")
    if not r2r1_report["decision"].get(
        "canonical_first_certified_fallback_freeze_authorized"
    ):
        raise RuntimeError("R2-R1 did not authorize hybrid freeze")

    canonical_sha = sha256_file(canonical_path)
    fallback_sha = sha256_file(fallback_source_path)
    export_sha = sha256_file(export_path)
    wrapper_sha = sha256_file(package_wrapper_path)

    if r2_contract["canonical_decoder_sha256"] != canonical_sha:
        raise RuntimeError("canonical decoder differs from R2 contract")
    if r2_contract["frozen_candidate_sha256"] != fallback_sha:
        raise RuntimeError("fallback differs from R2 contract")
    if r2_contract["D1_export_sha256"] != export_sha:
        raise RuntimeError("D1 export differs from R2 contract")

    contract = {
        "stage": STAGE,
        "policy": (
            "canonical presolve=True first; retry presolve=False only "
            "after canonical ExactDecoderError; fail closed if fallback "
            "also raises its ExactDecoderError"
        ),
        "canonical_decoder_sha256": canonical_sha,
        "fallback_decoder_sha256": fallback_sha,
        "hybrid_wrapper_sha256": wrapper_sha,
        "A1_margin_threshold": A1_MARGIN_THRESHOLD,
        "MILP_changed": False,
        "objective_changed": False,
        "route_library_changed": False,
        "certificate_tolerances_changed": False,
        "threshold_or_margin_changed": False,
        "beam_decoder_selected": False,
        "test_accessed": False,
    }
    atomic_json(contract_path, contract)

    shutil.copy2(package_wrapper_path, frozen_wrapper_path)

    # Audit the fallback and wrapper without installing them into src/.
    # Register the fallback under its intended module name so the wrapper can
    # resolve it. Exact source files are committed only after all checks pass.
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import importlib

    decoder_package = importlib.import_module("src.decoders")
    fallback_module_name = (
        "src.decoders.v5_p3_a1_exact_decoder_presolve_false"
    )
    wrapper_module_name = (
        "src.decoders.v5_p3_a1_exact_decoder_certified_hybrid"
    )

    fallback_module = import_source(
        fallback_source_path,
        fallback_module_name,
    )
    setattr(
        decoder_package,
        "v5_p3_a1_exact_decoder_presolve_false",
        fallback_module,
    )

    d4 = import_source(d4_script_path, "_v5_p3_r2_r2_d4")
    canonical = import_source(canonical_path, "_v5_p3_r2_r2_canonical")
    hybrid = import_source(
        package_wrapper_path,
        wrapper_module_name,
    )
    setattr(
        decoder_package,
        "v5_p3_a1_exact_decoder_certified_hybrid",
        hybrid,
    )

    with np.load(export_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}

    # Load canonical 10600:10700 committed chunk.
    matches = list(
        d4_working.glob(
            "chunk_*_items_10600_10700.manifest.json"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one canonical 10600:10700 manifest, found {matches}"
        )
    canonical_manifest = json.loads(matches[0].read_text())
    canonical_chunk_path = Path(canonical_manifest["chunk_path"])
    if canonical_manifest["chunk_sha256"] != sha256_file(
        canonical_chunk_path
    ):
        raise RuntimeError("canonical affected chunk SHA mismatch")
    with np.load(canonical_chunk_path, allow_pickle=False) as loaded:
        canonical_chunk = {
            key: loaded[key].copy() for key in loaded.files
        }

    affected_rows = []
    affected_all_match = True
    for index in range(AFFECTED_START, AFFECTED_STOP):
        observed = decode_semantic(d4, hybrid, arrays, index)
        expected = stored_semantic(
            canonical_chunk,
            index - AFFECTED_START,
        )
        match = compare_stored(observed, expected)
        affected_all_match &= match
        affected_rows.append(
            {
                "dataset_index": index,
                "selected_path": observed["selected_path"],
                "matches_stored_canonical": match,
            }
        )
    if not affected_all_match:
        raise RuntimeError("hybrid differs in affected 100-row chunk")

    # D3 probe regression against direct canonical execution.
    probe_rows = []
    for index in [int(v) for v in d3_selection["ordered_indices"]]:
        observed = decode_semantic(d4, hybrid, arrays, index)
        expected = decode_semantic(d4, canonical, arrays, index)
        match = compare_stored(observed, expected)
        if observed["selected_path"] != "canonical_presolve_true":
            raise RuntimeError(
                f"D3 probe unexpectedly used fallback at {index}"
            )
        if not match:
            raise RuntimeError(
                f"D3 probe changed semantic output at {index}"
            )
        probe_rows.append(
            {
                "dataset_index": index,
                "selected_path": observed["selected_path"],
                "matches_canonical": match,
            }
        )

    # Complementary fallback case at 11760.
    fallback_observed = decode_semantic(
        d4,
        hybrid,
        arrays,
        R1_FAILURE_INDEX,
    )
    expected_fallback = r1_report[
        "candidate_audit"
    ]["candidate_results"]["presolve_false"][
        "failure_item_result"
    ]["semantic"]
    fallback_semantic = {
        "margin": {
            "value": fallback_observed["margin"],
            "hex": fallback_observed["margin_hex"],
        },
        "attacker_count": fallback_observed["count"],
        "route_ids": fallback_observed["route_ids"],
        "source_mask": mask_int(fallback_observed["source"]),
        "transit_mask": mask_int(fallback_observed["transit"]),
        "victim_mask": mask_int(fallback_observed["victim"]),
        "path_mask": mask_int(fallback_observed["path"]),
    }
    fallback_matches = (
        fallback_observed["selected_path"]
        == "certified_fallback_presolve_false"
        and fallback_semantic == expected_fallback
    )
    if not fallback_matches:
        raise RuntimeError(
            "hybrid does not reproduce R1 fallback semantic result"
        )

    # Ensure generic exceptions are not swallowed as numerical fallback.
    original = hybrid._canonical.decode_best_attack_hypothesis
    def raise_runtime(*args, **kwargs):
        raise RuntimeError("NON_DECODER_FAILURE_SENTINEL")
    hybrid._canonical.decode_best_attack_hypothesis = raise_runtime
    generic_propagated = False
    try:
        hybrid.decode_best_attack_hypothesis(
            float(arrays["attack_logits"][0]),
            arrays["count_logits"][0],
            arrays["source_logits"][0],
            arrays["transit_logits"][0],
            arrays["victim_logits"][0],
            arrays["path_logits"][0],
        )
    except RuntimeError as exc:
        generic_propagated = (
            str(exc) == "NON_DECODER_FAILURE_SENTINEL"
        )
    finally:
        hybrid._canonical.decode_best_attack_hypothesis = original
    if not generic_propagated:
        raise RuntimeError(
            "hybrid incorrectly catches generic runtime failures"
        )

    # Install exact source files only after all semantic and exception
    # boundary checks have passed.
    installed_fallback_path.parent.mkdir(parents=True, exist_ok=True)
    for destination, source, expected_sha in (
        (installed_fallback_path, fallback_source_path, fallback_sha),
        (installed_wrapper_path, package_wrapper_path, wrapper_sha),
    ):
        if destination.is_file():
            if sha256_file(destination) != expected_sha:
                raise RuntimeError(
                    f"existing installed file differs: {destination}"
                )
        else:
            temporary = destination.with_suffix(
                destination.suffix + ".tmp"
            )
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        if sha256_file(destination) != expected_sha:
            raise RuntimeError(
                f"installed source SHA mismatch: {destination}"
            )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Freeze and install the P3 canonical-first, certified "
            "presolve-disabled fallback numerical policy."
        ),
        "contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
        },
        "installed": {
            "canonical_decoder_modified": False,
            "fallback_path": str(installed_fallback_path),
            "fallback_sha256": sha256_file(installed_fallback_path),
            "hybrid_wrapper_path": str(installed_wrapper_path),
            "hybrid_wrapper_sha256": sha256_file(
                installed_wrapper_path
            ),
        },
        "regression": {
            "affected_chunk": {
                "range": [AFFECTED_START, AFFECTED_STOP],
                "items": AFFECTED_STOP - AFFECTED_START,
                "all_match_stored_canonical": affected_all_match,
                "rows": affected_rows,
            },
            "D3_probes": {
                "items": len(probe_rows),
                "all_canonical_path_and_exact_match": True,
                "rows": probe_rows,
            },
            "R1_failure": {
                "dataset_index": R1_FAILURE_INDEX,
                "selected_path": fallback_observed["selected_path"],
                "matches_R1_semantic": fallback_matches,
            },
            "generic_non_decoder_exception_propagated": (
                generic_propagated
            ),
        },
        "decision": {
            "hybrid_numerical_policy_frozen": True,
            "canonical_D4_prefix_11700_reusable": True,
            "R3_resume_authorized": True,
            "certificate_tolerance_relaxation_authorized": False,
            "threshold_or_margin_retuning_authorized": False,
            "beam_decoder_selected": False,
            "sealed_test_evaluation_authorized": False,
            "next_stage": (
                "V5_P3_D4_R3_RESUME_A1_EXACT_VALIDATION_WITH_"
                "FROZEN_CERTIFIED_HYBRID_DECODER"
            ),
        },
        "sealed_test": {
            "test_dataset_instantiated": False,
            "test_length_computed": False,
            "test_tensor_loaded": False,
        },
        "provenance": {
            "R2_R1_report_sha256": sha256_file(r2r1_report_path),
            "R2_R1_lock_sha256": sha256_file(r2r1_lock_path),
            "R2_contract_sha256": sha256_file(r2_contract_path),
            "R1_report_sha256": sha256_file(r1_report_path),
            "D1_export_sha256": export_sha,
            "canonical_decoder_sha256": canonical_sha,
            "fallback_decoder_sha256": fallback_sha,
            "installed_script_sha256": sha256_file(installed_script),
        },
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(report_path),
        "contract_sha256": sha256_file(contract_path),
        "canonical_decoder_sha256": canonical_sha,
        "fallback_decoder_sha256": fallback_sha,
        "hybrid_wrapper_sha256": wrapper_sha,
        "affected_chunk_exact_match": affected_all_match,
        "D3_probe_count": len(probe_rows),
        "R1_fallback_matches": fallback_matches,
        "test_tensor_loaded": False,
    }
    atomic_json(lock_path, lock)
    complete_path.write_text(f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(
        "policy=canonical_presolve_true_then_certified_"
        "presolve_false_fallback"
    )
    print("affected_chunk_items=100")
    print("affected_chunk_exact_match=true")
    print(f"D3_probe_items={len(probe_rows)}")
    print("D3_probes_exact_match=true")
    print(f"R1_failure_index={R1_FAILURE_INDEX}")
    print(
        "R1_failure_selected_path="
        f"{fallback_observed['selected_path']}"
    )
    print("R1_failure_matches_R1_semantic=true")
    print("generic_non_decoder_exception_propagated=true")
    print("canonical_decoder_modified=false")
    print(f"installed_fallback={installed_fallback_path}")
    print(f"installed_fallback_sha256={fallback_sha}")
    print(f"installed_hybrid_wrapper={installed_wrapper_path}")
    print(f"installed_hybrid_wrapper_sha256={wrapper_sha}")
    print("certificate_tolerance_relaxation_authorized=false")
    print("threshold_or_margin_retuning_authorized=false")
    print("beam_decoder_selected=false")
    print("test_tensor_loaded=false")
    print("canonical_D4_prefix_11700_reusable=true")
    print("R3_resume_authorized=true")
    print(
        "next_stage="
        "V5_P3_D4_R3_RESUME_A1_EXACT_VALIDATION_WITH_"
        "FROZEN_CERTIFIED_HYBRID_DECODER"
    )
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
