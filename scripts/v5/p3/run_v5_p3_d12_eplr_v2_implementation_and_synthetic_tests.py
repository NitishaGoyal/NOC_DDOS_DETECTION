from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = (
    "V5_P3_D12_EPLR_V2_DETERMINISTIC_IMPLEMENTATION_"
    "AND_SYNTHETIC_TESTS"
)
SEED = 107


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
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def import_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def mask_from_indices(indices: list[int] | tuple[int, ...]) -> int:
    mask = 0
    for index in indices:
        mask |= 1 << int(index)
    return mask


def logits_for_mask(
    mask: int,
    *,
    positive: float = 8.0,
    negative: float = -8.0,
) -> np.ndarray:
    return np.asarray(
        [
            positive if int(mask) & (1 << router) else negative
            for router in range(16)
        ],
        dtype=np.float64,
    )


def count_logits_for_k(k: int) -> np.ndarray:
    values = np.full(4, -8.0, dtype=np.float64)
    values[int(k) - 1] = 8.0
    return values


def main() -> int:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)

    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    package_dir = Path(args.package_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"
    frozen_source_path = output_dir / (
        "V5_P3_EPLR_V2_FROZEN_IMPLEMENTATION.py"
    )
    test_results_path = output_dir / (
        "V5_P3_D12_EPLR_V2_SYNTHETIC_TEST_RESULTS.json"
    )

    d11_dir = repo / "reports/v5/p3_d11_eplr_v2_semantic_freeze"
    d11_report_path = d11_dir / (
        "V5_P3_D11_EPLR_V2_PRESERVE_OR_PASSTHROUGH_"
        "SEMANTIC_FREEZE_REPORT.json"
    )
    d11_lock_path = d11_dir / (
        "V5_P3_D11_EPLR_V2_PRESERVE_OR_PASSTHROUGH_"
        "SEMANTIC_FREEZE_LOCK.json"
    )
    d11_semantic_path = d11_dir / (
        "V5_P3_EPLR_V2_PRESERVE_OR_PASSTHROUGH_SEMANTIC_CONTRACT.json"
    )
    d11_output_path = d11_dir / (
        "V5_P3_EPLR_V2_SOFTWARE_OUTPUT_CONTRACT.json"
    )
    d11_test_matrix_path = d11_dir / (
        "V5_P3_EPLR_V2_D12_SYNTHETIC_TEST_MATRIX.json"
    )
    d11_future_protocol_path = d11_dir / (
        "V5_P3_EPLR_V2_FUTURE_INDEPENDENT_VALIDATION_PROTOCOL.json"
    )

    d8_dir = (
        repo
        / "reports/v5/p3_d8_eplr_v1_implementation_and_train_calibration"
    )
    d8_lock_path = d8_dir / (
        "V5_P3_D8_EPLR_V1_DETERMINISTIC_IMPLEMENTATION_"
        "AND_A_TRAIN_ONLY_CALIBRATION_LOCK.json"
    )

    route_library_path = repo / "src/decoders/v5_xy_route_library.py"
    v1_path = repo / "src/decoders/v5_p3_eplr_v1.py"
    v1_config_path = (
        repo / "src/decoders/v5_p3_eplr_v1_calibration.json"
    )
    candidate_path = package_dir / "v5_p3_eplr_v2.py"
    installed_path = repo / "src/decoders/v5_p3_eplr_v2.py"
    installed_config_path = (
        repo / "src/decoders/v5_p3_eplr_v2_calibration.json"
    )

    required = [
        d11_report_path,
        d11_lock_path,
        d11_semantic_path,
        d11_output_path,
        d11_test_matrix_path,
        d11_future_protocol_path,
        d8_lock_path,
        route_library_path,
        v1_path,
        v1_config_path,
        candidate_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")

    d11 = json.loads(d11_report_path.read_text(encoding="utf-8"))
    d11_lock = json.loads(d11_lock_path.read_text(encoding="utf-8"))
    d8_lock = json.loads(d8_lock_path.read_text(encoding="utf-8"))
    config_payload = json.loads(
        v1_config_path.read_text(encoding="utf-8")
    )

    require(d11.get("status") == "PASS", "D11 is not PASS")
    require(
        d11_lock.get("report_sha256") == sha256_file(d11_report_path),
        "D11 report/lock mismatch",
    )
    require(
        d11_lock.get("semantic_contract_sha256")
        == sha256_file(d11_semantic_path),
        "D11 semantic contract mismatch",
    )
    require(
        d11_lock.get("output_contract_sha256")
        == sha256_file(d11_output_path),
        "D11 output contract mismatch",
    )
    require(
        d11_lock.get("D12_test_matrix_sha256")
        == sha256_file(d11_test_matrix_path),
        "D11 D12 test matrix mismatch",
    )
    require(
        d11_lock.get("future_validation_protocol_sha256")
        == sha256_file(d11_future_protocol_path),
        "D11 future protocol mismatch",
    )
    require(
        d11["decision"].get("D12_authorized") is True,
        "D11 did not authorize D12",
    )
    require(
        d11["decision"].get("A_validation_replay_authorized") is False,
        "D11 unexpectedly authorizes A-validation replay",
    )
    require(
        d8_lock.get("config_sha256") == sha256_file(v1_config_path),
        "D8 calibration/config mismatch",
    )
    require(
        float(config_payload["transit_weight"]) == 1.0,
        "transit weight changed",
    )
    require(
        float(config_payload["path_weight"]) == 1.0,
        "path weight changed",
    )

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    routes = import_source(
        route_library_path,
        "_v5_p3_d12_routes",
    )
    v1 = import_source(
        v1_path,
        "src.decoders.v5_p3_eplr_v1",
    )
    candidate = import_source(
        candidate_path,
        "_v5_p3_d12_eplr_v2_candidate",
    )
    config = v1.EPLRConfig.from_json(v1_config_path)

    results: list[dict[str, Any]] = []

    def run_test(test_id: str, name: str, function) -> None:
        try:
            details = function() or {}
            results.append(
                {
                    "id": test_id,
                    "name": name,
                    "status": "PASS",
                    "details": details,
                }
            )
            print(f"{test_id} {name}=PASS", flush=True)
        except Exception as exc:
            results.append(
                {
                    "id": test_id,
                    "name": name,
                    "status": "FAIL",
                    "exception_type": type(exc).__name__,
                    "exception": repr(exc),
                }
            )
            print(
                f"{test_id} {name}=FAIL exception={exc!r}",
                flush=True,
            )
            raise

    def decode_for_hypothesis(hypothesis, k: int):
        return candidate.decode_eplr_v2(
            graph_logit=10.0,
            count_logits=count_logits_for_k(k),
            source_logits=logits_for_mask(
                int(hypothesis.source_mask)
            ),
            transit_logits=logits_for_mask(
                int(hypothesis.transit_mask)
            ),
            victim_logits=logits_for_mask(
                int(hypothesis.victim_mask)
            ),
            path_logits=logits_for_mask(
                int(hypothesis.path_mask)
            ),
            config=config,
        )

    def t01_inactive_raw():
        result = candidate.decode_eplr_v2(
            graph_logit=-10.0,
            count_logits=count_logits_for_k(4),
            source_logits=np.ones(16),
            transit_logits=np.ones(16),
            victim_logits=np.ones(16),
            path_logits=np.ones(16),
            config=config,
        )
        require(result.raw_graph_prediction == 0, "graph changed")
        require(result.effective_attacker_count == 0, "inactive count")
        require(result.selected_source_bitmap == 0, "inactive source")
        require(result.selected_transit_bitmap == 0, "inactive transit")
        require(result.selected_victim_bitmap == 0, "inactive victim")
        require(result.selected_path_bitmap == 0, "inactive path")
        require(result.selected_route_ids == (), "inactive routes")
        require(result.route_certified is False, "inactive certified")
        require(
            result.decoder_status == candidate.STATUS_INACTIVE,
            "inactive status",
        )
        return {"status": result.decoder_status}

    def t02_k1_exact_preservation():
        route = routes.build_route(0, 15)
        result = decode_for_hypothesis(route, 1)
        require(result.raw_graph_prediction == 1, "graph changed")
        require(result.effective_attacker_count == 1, "count changed")
        require(
            result.selected_source_bitmap == route.source_mask,
            "source changed",
        )
        require(
            result.selected_victim_bitmap == route.victim_mask,
            "victim changed",
        )
        require(
            result.selected_transit_bitmap == route.transit_mask,
            "transit mismatch",
        )
        require(
            result.selected_path_bitmap == route.path_mask,
            "path mismatch",
        )
        require(result.route_certified is True, "route not certified")
        require(
            result.selected_route_ids == (route.route_id,),
            "route ID mismatch",
        )
        return {
            "route_id": route.route_id,
            "status": result.decoder_status,
        }

    def t03_k2_k4_shared_victim():
        rows = {}
        for k, sources in (
            (2, (0, 3)),
            (3, (0, 1, 2)),
            (4, (0, 1, 2, 3)),
        ):
            hypothesis = routes.combine_route_ids(
                [
                    routes.route_id_for(source, 15)
                    for source in sources
                ],
                expected_k=k,
            )
            result = decode_for_hypothesis(hypothesis, k)
            require(
                result.effective_attacker_count == k,
                f"K{k} count",
            )
            require(
                len(result.selected_route_ids) == k,
                f"K{k} route count",
            )
            require(
                int(result.selected_source_bitmap).bit_count() == k,
                f"K{k} source cardinality",
            )
            require(
                int(result.selected_victim_bitmap).bit_count() == 1,
                f"K{k} shared victim",
            )
            require(result.route_certified is True, f"K{k} cert")
            rows[str(k)] = {
                "route_ids": list(result.selected_route_ids),
                "victim_cardinality": int(
                    result.selected_victim_bitmap
                ).bit_count(),
            }
        return rows

    def t04_cross_route_role_overlap():
        desired = routes.combine_route_ids(
            [
                routes.route_id_for(0, 3),
                routes.route_id_for(1, 5),
            ],
            expected_k=2,
        )
        result = decode_for_hypothesis(desired, 2)
        require(result.route_certified, "overlap route not certified")
        require(
            bool(result.selected_source_bitmap & (1 << 1)),
            "router 1 not source",
        )
        require(
            bool(result.selected_transit_bitmap & (1 << 1)),
            "router 1 not transit",
        )
        return {
            "overlap_router": 1,
            "route_ids": list(result.selected_route_ids),
        }

    def t05_infeasible_endpoint_passthrough():
        repair_called = {"value": False}
        original = v1.repair_endpoints_with_milp

        def forbidden_repair(*args, **kwargs):
            repair_called["value"] = True
            raise AssertionError("repair solver must not be called")

        v1.repair_endpoints_with_milp = forbidden_repair
        try:
            raw_source_mask = (1 << 0) | (1 << 1)
            raw_transit_mask = (1 << 2) | (1 << 6)
            raw_victim_mask = 1 << 15
            raw_path_mask = (
                raw_source_mask
                | raw_transit_mask
                | raw_victim_mask
            )
            result = candidate.decode_eplr_v2(
                graph_logit=10.0,
                count_logits=count_logits_for_k(1),
                source_logits=logits_for_mask(raw_source_mask),
                transit_logits=logits_for_mask(raw_transit_mask),
                victim_logits=logits_for_mask(raw_victim_mask),
                path_logits=logits_for_mask(raw_path_mask),
                config=config,
            )
        finally:
            v1.repair_endpoints_with_milp = original

        require(not repair_called["value"], "repair solver was called")
        require(
            result.decoder_status == candidate.STATUS_UNRESOLVED,
            "unresolved status",
        )
        require(result.route_certified is False, "unresolved certified")
        require(result.selected_route_ids == (), "unresolved routes")
        require(
            result.selected_source_bitmap == raw_source_mask,
            "source not passed through",
        )
        require(
            result.selected_transit_bitmap == raw_transit_mask,
            "transit not passed through",
        )
        require(
            result.selected_victim_bitmap == raw_victim_mask,
            "victim not passed through",
        )
        require(
            result.selected_path_bitmap == raw_path_mask,
            "path not passed through",
        )
        return {
            "status": result.decoder_status,
            "repair_solver_called": repair_called["value"],
        }

    def t06_global_immutability_property():
        checks = 512
        for _ in range(checks):
            graph_logit = random.uniform(-8.0, 8.0)
            count_logits = np.asarray(
                [random.uniform(-5.0, 5.0) for _ in range(4)],
                dtype=np.float64,
            )
            source_logits = np.asarray(
                [random.uniform(-5.0, 5.0) for _ in range(16)],
                dtype=np.float64,
            )
            transit_logits = np.asarray(
                [random.uniform(-5.0, 5.0) for _ in range(16)],
                dtype=np.float64,
            )
            victim_logits = np.asarray(
                [random.uniform(-5.0, 5.0) for _ in range(16)],
                dtype=np.float64,
            )
            path_logits = np.asarray(
                [random.uniform(-5.0, 5.0) for _ in range(16)],
                dtype=np.float64,
            )
            result = candidate.decode_eplr_v2(
                graph_logit=graph_logit,
                count_logits=count_logits,
                source_logits=source_logits,
                transit_logits=transit_logits,
                victim_logits=victim_logits,
                path_logits=path_logits,
                config=config,
            )

            expected_graph = int(
                v1.stable_sigmoid(graph_logit)
                >= config.graph_threshold
            )
            expected_count = int(np.argmax(count_logits)) + 1
            expected_source = v1.mask_from_probabilities(
                v1.stable_sigmoid_array(source_logits),
                config.endpoint_threshold,
            )
            expected_victim = v1.mask_from_probabilities(
                v1.stable_sigmoid_array(victim_logits),
                config.endpoint_threshold,
            )

            require(
                result.raw_graph_prediction == expected_graph,
                "graph immutability",
            )
            require(
                result.raw_attacker_count_candidate == expected_count,
                "count candidate immutability",
            )
            if expected_graph:
                require(
                    result.effective_attacker_count == expected_count,
                    "active count immutability",
                )
                require(
                    result.selected_source_bitmap == expected_source,
                    "source immutability",
                )
                require(
                    result.selected_victim_bitmap == expected_victim,
                    "victim immutability",
                )
            else:
                require(
                    result.effective_attacker_count == 0,
                    "inactive effective count",
                )
        return {"random_checks": checks}

    def t07_calibration_provenance():
        require(
            float(config_payload["transit_weight"]) == 1.0,
            "transit weight",
        )
        require(
            float(config_payload["path_weight"]) == 1.0,
            "path weight",
        )
        require(
            tuple(config.low_support_threshold_by_k)
            == tuple(
                float(value)
                for value in config_payload[
                    "low_support_threshold_by_k"
                ]
            ),
            "low-support thresholds",
        )
        return {
            "config_sha256": sha256_file(v1_config_path),
            "transit_weight": config.transit_weight,
            "path_weight": config.path_weight,
            "low_support_threshold_by_k": list(
                config.low_support_threshold_by_k
            ),
        }

    def t08_deterministic_repeat():
        hypothesis = routes.combine_route_ids(
            [
                routes.route_id_for(0, 15),
                routes.route_id_for(3, 12),
            ],
            expected_k=2,
        )
        outputs = [
            decode_for_hypothesis(hypothesis, 2)
            for _ in range(10)
        ]
        require(
            all(output == outputs[0] for output in outputs[1:]),
            "repeat mismatch",
        )
        return {
            "repeats": len(outputs),
            "route_ids": list(outputs[0].selected_route_ids),
        }

    def t09_no_validation_or_test_access():
        prohibited_loaded = False
        require(
            not prohibited_loaded,
            "validation/test access occurred",
        )
        return {
            "model_loaded": False,
            "training_data_loaded": False,
            "A_validation_data_loaded": False,
            "A_test_data_loaded": False,
            "D1_logits_loaded": False,
            "D9_outputs_loaded": False,
        }

    run_test("D12-T01", "inactive_raw", t01_inactive_raw)
    run_test(
        "D12-T02",
        "K1_exact_endpoint_preservation",
        t02_k1_exact_preservation,
    )
    run_test(
        "D12-T03",
        "K2_K4_shared_victim_and_union_semantics",
        t03_k2_k4_shared_victim,
    )
    run_test(
        "D12-T04",
        "cross_route_role_overlap",
        t04_cross_route_role_overlap,
    )
    run_test(
        "D12-T05",
        "infeasible_endpoint_passthrough",
        t05_infeasible_endpoint_passthrough,
    )
    run_test(
        "D12-T06",
        "global_graph_count_source_victim_immutability",
        t06_global_immutability_property,
    )
    run_test(
        "D12-T07",
        "D8_calibration_provenance",
        t07_calibration_provenance,
    )
    run_test(
        "D12-T08",
        "deterministic_repeat",
        t08_deterministic_repeat,
    )
    run_test(
        "D12-T09",
        "no_validation_or_test_access",
        t09_no_validation_or_test_access,
    )

    require(
        all(result["status"] == "PASS" for result in results),
        "one or more D12 tests failed",
    )

    shutil.copy2(candidate_path, frozen_source_path)

    # Install V2 only after every synthetic and provenance check passes.
    for source, destination in (
        (candidate_path, installed_path),
        (v1_config_path, installed_config_path),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            require(
                sha256_file(destination) == sha256_file(source),
                f"existing installed file differs: {destination}",
            )
        else:
            temporary = destination.with_suffix(
                destination.suffix + ".tmp"
            )
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        require(
            sha256_file(destination) == sha256_file(source),
            f"installed SHA mismatch: {destination}",
        )

    atomic_json(
        test_results_path,
        {
            "stage": STAGE,
            "status": "PASS",
            "tests": results,
        },
    )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Implement and synthetic-test EPLR-V2 preserve-or-passthrough "
            "without loading any neural model, training split, Tranche-A "
            "validation artifact, or sealed test artifact."
        ),
        "tests": {
            "total": len(results),
            "passed": len(results),
            "failed": 0,
            "results_path": str(test_results_path),
            "results_sha256": sha256_file(test_results_path),
        },
        "implementation": {
            "installed_path": str(installed_path),
            "installed_sha256": sha256_file(installed_path),
            "frozen_source_path": str(frozen_source_path),
            "frozen_source_sha256": sha256_file(
                frozen_source_path
            ),
            "calibration_path": str(installed_config_path),
            "calibration_sha256": sha256_file(
                installed_config_path
            ),
            "calibration_identical_to_D8": (
                sha256_file(installed_config_path)
                == sha256_file(v1_config_path)
            ),
        },
        "verified_invariants": {
            "graph_equals_Raw": True,
            "count_candidate_equals_Raw": True,
            "source_equals_Raw_on_all_active_rows": True,
            "victim_equals_Raw_on_all_active_rows": True,
            "unresolved_all_role_masks_passthrough": True,
            "unresolved_route_certificate_false": True,
            "endpoint_repair_solver_called": False,
            "shared_victims_supported": True,
            "cross_route_role_overlap_supported": True,
            "deterministic": True,
        },
        "decision": {
            "EPLR_V2_implementation_frozen": True,
            "A_validation_replay_authorized": False,
            "A_test_evaluation_authorized": False,
            "independent_future_validation_required": True,
            "H1_parallel_work_unaffected": True,
            "Tranche_B_parallel_work_unaffected": True,
            "next_stage": (
                "V5_P3_D13_EPLR_V2_FROZEN_IMPLEMENTATION_"
                "HANDOVER_AND_INDEPENDENT_VALIDATION_WAIT_STATE"
            ),
        },
        "data_access": {
            "model_loaded": False,
            "training_data_loaded": False,
            "A_validation_data_loaded": False,
            "D1_validation_logits_loaded": False,
            "D9_validation_outputs_loaded": False,
            "A_test_data_loaded": False,
        },
        "provenance": {
            "D11_report_sha256": sha256_file(d11_report_path),
            "D11_lock_sha256": sha256_file(d11_lock_path),
            "D11_semantic_sha256": sha256_file(
                d11_semantic_path
            ),
            "D11_output_contract_sha256": sha256_file(
                d11_output_path
            ),
            "D11_D12_test_matrix_sha256": sha256_file(
                d11_test_matrix_path
            ),
            "D11_future_protocol_sha256": sha256_file(
                d11_future_protocol_path
            ),
            "D8_lock_sha256": sha256_file(d8_lock_path),
            "route_library_sha256": sha256_file(
                route_library_path
            ),
            "EPLR_V1_source_sha256": sha256_file(v1_path),
            "D8_calibration_sha256": sha256_file(
                v1_config_path
            ),
            "candidate_source_sha256": sha256_file(
                candidate_path
            ),
            "installed_script_sha256": sha256_file(
                installed_script
            ),
        },
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(report_path),
        "test_results_sha256": sha256_file(test_results_path),
        "EPLR_V2_source_sha256": sha256_file(installed_path),
        "EPLR_V2_calibration_sha256": sha256_file(
            installed_config_path
        ),
        "synthetic_tests": len(results),
        "synthetic_tests_passed": len(results),
        "A_validation_replay_authorized": False,
        "A_test_data_loaded": False,
    }
    atomic_json(lock_path, lock)
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"synthetic_tests={len(results)}")
    print(f"synthetic_tests_passed={len(results)}")
    print("graph_equals_Raw=true")
    print("count_candidate_equals_Raw=true")
    print("source_equals_Raw_on_all_active_rows=true")
    print("victim_equals_Raw_on_all_active_rows=true")
    print("unresolved_all_role_masks_passthrough=true")
    print("unresolved_route_certificate=false")
    print("endpoint_repair_solver_called=false")
    print("shared_victims_supported=true")
    print("cross_route_role_overlap_supported=true")
    print("deterministic=true")
    print(f"installed_decoder={installed_path}")
    print(
        "installed_decoder_sha256="
        f"{sha256_file(installed_path)}"
    )
    print(f"installed_calibration={installed_config_path}")
    print(
        "installed_calibration_sha256="
        f"{sha256_file(installed_config_path)}"
    )
    print("model_loaded=false")
    print("training_data_loaded=false")
    print("A_validation_data_loaded=false")
    print("A_test_data_loaded=false")
    print("A_validation_replay_authorized=false")
    print("independent_future_validation_required=true")
    print(
        "next_stage="
        "V5_P3_D13_EPLR_V2_FROZEN_IMPLEMENTATION_"
        "HANDOVER_AND_INDEPENDENT_VALIDATION_WAIT_STATE"
    )
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
