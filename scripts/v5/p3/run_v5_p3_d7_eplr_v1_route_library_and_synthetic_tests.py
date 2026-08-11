from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = "V5_P3_D7_EPLR_V1_ROUTE_LIBRARY_AND_SYNTHETIC_UNIT_TESTS"
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


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
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


def uniform_probabilities(value: float) -> list[float]:
    return [float(value)] * 16


def probabilities_for_mask(
    mask: int,
    *,
    positive: float = 0.9,
    negative: float = 0.1,
) -> list[float]:
    return [
        positive if mask & (1 << index) else negative
        for index in range(16)
    ]


def main() -> int:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)

    repo = Path(args.repo).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    package_dir = Path(args.package_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    d6_dir = repo / "reports/v5/p3_d6_eplr_v1_design_freeze"
    d6_report_path = d6_dir / (
        "V5_P3_D6_EPLR_V1_SEMANTIC_AND_OBJECTIVE_"
        "SPECIFICATION_FREEZE_REPORT.json"
    )
    d6_lock_path = d6_dir / (
        "V5_P3_D6_EPLR_V1_SEMANTIC_AND_OBJECTIVE_"
        "SPECIFICATION_FREEZE_LOCK.json"
    )
    semantic_path = d6_dir / "V5_P3_EPLR_V1_SEMANTIC_CONTRACT.json"
    objective_path = d6_dir / "V5_P3_EPLR_V1_OBJECTIVE_CONTRACT.json"
    output_contract_path = d6_dir / "V5_P3_EPLR_V1_OUTPUT_CONTRACT.json"
    d7_matrix_path = d6_dir / "V5_P3_EPLR_V1_D7_UNIT_TEST_MATRIX.json"

    route_library_path = repo / "src/decoders/v5_xy_route_library.py"
    package_helper_path = (
        package_dir / "v5_p3_eplr_v1_reference_semantics.py"
    )
    frozen_helper_path = out / (
        "V5_P3_EPLR_V1_D7_FROZEN_REFERENCE_SEMANTICS.py"
    )

    required = [
        d6_report_path,
        d6_lock_path,
        semantic_path,
        objective_path,
        output_contract_path,
        d7_matrix_path,
        route_library_path,
        package_helper_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")

    d6_report = json.loads(d6_report_path.read_text(encoding="utf-8"))
    d6_lock = json.loads(d6_lock_path.read_text(encoding="utf-8"))
    require(d6_report.get("status") == "PASS", "D6 is not PASS")
    require(
        d6_lock.get("report_sha256") == sha256_file(d6_report_path),
        "D6 report/lock mismatch",
    )
    require(
        d6_lock.get("semantic_sha256") == sha256_file(semantic_path),
        "D6 semantic contract mismatch",
    )
    require(
        d6_lock.get("objective_sha256") == sha256_file(objective_path),
        "D6 objective contract mismatch",
    )
    require(
        d6_lock.get("output_sha256")
        == sha256_file(output_contract_path),
        "D6 output contract mismatch",
    )
    require(
        d6_lock.get("D7_matrix_sha256") == sha256_file(d7_matrix_path),
        "D6 D7 matrix mismatch",
    )
    require(
        d6_report["decision"].get("D7_authorized") is True,
        "D6 did not authorize D7",
    )

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    routes = import_source(
        route_library_path,
        "_v5_p3_d7_route_library",
    )
    helper = import_source(
        package_helper_path,
        "_v5_p3_d7_reference_semantics",
    )

    test_results = []

    def run_test(test_id: str, name: str, function) -> None:
        try:
            details = function() or {}
            test_results.append(
                {
                    "id": test_id,
                    "name": name,
                    "status": "PASS",
                    "details": details,
                }
            )
            print(f"{test_id} {name}=PASS", flush=True)
        except Exception as exc:
            test_results.append(
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

    def t01_route_inventory():
        require(routes.MESH_ROWS == 4, "mesh rows changed")
        require(routes.MESH_COLUMNS == 4, "mesh columns changed")
        require(routes.ROUTER_COUNT == 16, "router count changed")
        require(
            routes.ORDERED_ROUTE_COUNT == 240,
            "ordered route count changed",
        )
        require(len(routes.ROUTE_TABLE) == 240, "route table length")
        require(
            [record.route_id for record in routes.ROUTE_TABLE]
            == list(range(240)),
            "route IDs are not contiguous",
        )
        pairs = {(record.source, record.victim) for record in routes.ROUTE_TABLE}
        expected = {
            (source, victim)
            for source in range(16)
            for victim in range(16)
            if source != victim
        }
        require(pairs == expected, "source/victim coverage mismatch")
        return {"routes": 240, "endpoint_pairs": len(pairs)}

    def t02_route_id_roundtrip():
        for source in range(16):
            for victim in range(16):
                if source == victim:
                    continue
                route_id = routes.route_id_for(source, victim)
                require(
                    routes.source_victim_for_route_id(route_id)
                    == (source, victim),
                    f"round trip failed for {source}->{victim}",
                )
        return {"round_trips": 240}

    def t03_topology_and_xy_legality():
        edges = routes.expected_directed_edges()
        require(len(edges) == 48, "expected edge count changed")
        for record in routes.ROUTE_TABLE:
            routes.validate_route_nodes(
                record.source,
                record.victim,
                record.nodes,
                require_xy=True,
            )
            require(
                record.nodes
                == routes.generate_xy_nodes(record.source, record.victim),
                f"XY nodes changed for route {record.route_id}",
            )
            require(
                record.path_mask
                == (
                    record.source_mask
                    | record.transit_mask
                    | record.victim_mask
                ),
                f"path identity failed at route {record.route_id}",
            )
        return {"directed_edges": 48, "routes_validated": 240}

    def t04_inactive_raw():
        result = helper.assemble_result(
            raw_graph_probability=0.2,
            raw_graph_prediction=0,
            raw_attacker_count_candidate=3,
            raw_source_mask=0x0003,
            raw_transit_mask=0x00F0,
            raw_victim_mask=0x8000,
            raw_path_mask=0x80F3,
            hypothesis=None,
            route_consistency_score=-12.0,
            endpoint_repaired=False,
            low_support=False,
        )
        require(result.raw_graph_prediction == 0, "graph changed")
        require(result.effective_attacker_count == 0, "inactive count")
        require(result.decoded_source_mask == 0, "inactive source")
        require(result.decoded_transit_mask == 0, "inactive transit")
        require(result.decoded_victim_mask == 0, "inactive victim")
        require(result.decoded_path_mask == 0, "inactive path")
        require(
            result.decoder_status == helper.STATUS_INACTIVE,
            "inactive status",
        )
        return {"status": result.decoder_status}

    def t05_k1_exact_preservation():
        route = routes.build_route(0, 15)
        candidates = helper.enumerate_exact_endpoint_hypotheses(
            routes,
            source_mask=route.source_mask,
            victim_mask=route.victim_mask,
            k=1,
        )
        require(len(candidates) == 1, "K1 candidate count")
        require(candidates[0].route_ids == (route.route_id,), "K1 route")
        require(candidates[0].transit_mask == route.transit_mask, "transit")
        require(candidates[0].path_mask == route.path_mask, "path")
        return {"route_id": route.route_id, "nodes": list(route.nodes)}

    def t06_k2_k4_union_semantics():
        examples = {
            2: [(0, 15), (3, 12)],
            3: [(0, 15), (1, 14), (2, 13)],
            4: [(0, 15), (1, 14), (2, 13), (3, 12)],
        }
        rows = {}
        for k, pairs in examples.items():
            hypothesis = routes.combine_route_ids(
                [routes.route_id_for(source, victim) for source, victim in pairs],
                expected_k=k,
            )
            require(hypothesis.attacker_count == k, f"K{k} count")
            require(hypothesis.source_mask.bit_count() == k, f"K{k} source")
            rows[str(k)] = {
                "route_ids": list(hypothesis.route_ids),
                "source_cardinality": hypothesis.source_mask.bit_count(),
                "victim_cardinality": hypothesis.victim_mask.bit_count(),
                "path_cardinality": hypothesis.path_mask.bit_count(),
            }
        return rows

    def t07_shared_victim():
        hypothesis = routes.combine_route_ids(
            [
                routes.route_id_for(0, 15),
                routes.route_id_for(3, 15),
            ],
            expected_k=2,
        )
        require(hypothesis.source_mask.bit_count() == 2, "shared source")
        require(hypothesis.victim_mask.bit_count() == 1, "shared victim")
        return {
            "route_ids": list(hypothesis.route_ids),
            "victim_mask": hypothesis.victim_mask,
        }

    def t08_role_overlap_across_routes():
        source_transit = routes.combine_route_ids(
            [
                routes.route_id_for(0, 3),
                routes.route_id_for(1, 5),
            ],
            expected_k=2,
        )
        require(
            bool(source_transit.source_mask & (1 << 1)),
            "router 1 not source",
        )
        require(
            bool(source_transit.transit_mask & (1 << 1)),
            "router 1 not transit",
        )
        source_victim = routes.combine_route_ids(
            [
                routes.route_id_for(0, 3),
                routes.route_id_for(3, 4),
            ],
            expected_k=2,
        )
        require(
            bool(source_victim.source_mask & (1 << 3)),
            "router 3 not source",
        )
        require(
            bool(source_victim.victim_mask & (1 << 3)),
            "router 3 not victim",
        )
        return {
            "source_transit_overlap_router": 1,
            "source_victim_overlap_router": 3,
        }

    def t09_overlapping_path_union():
        first = routes.get_route(routes.route_id_for(0, 15))
        second = routes.get_route(routes.route_id_for(1, 12))
        hypothesis = routes.combine_route_ids(
            [first.route_id, second.route_id],
            expected_k=2,
        )
        independent = first.path_mask.bit_count() + second.path_mask.bit_count()
        union = hypothesis.path_mask.bit_count()
        require(union < independent, "paths did not overlap")
        return {"independent_cardinality": independent, "union_cardinality": union}

    def t10_endpoint_assignment_and_tie_break():
        source_mask = (1 << 0) | (1 << 3)
        victim_mask = (1 << 12) | (1 << 15)
        candidates = helper.enumerate_exact_endpoint_hypotheses(
            routes,
            source_mask=source_mask,
            victim_mask=victim_mask,
            k=2,
        )
        require(len(candidates) == 2, "expected two endpoint assignments")
        transit = uniform_probabilities(0.5)
        path = uniform_probabilities(0.5)
        chosen, score = helper.choose_endpoint_preserving_hypothesis(
            routes,
            source_mask=source_mask,
            victim_mask=victim_mask,
            k=2,
            transit_probabilities=transit,
            path_probabilities=path,
        )
        expected = min(hypothesis.route_ids for hypothesis in candidates)
        require(chosen.route_ids == expected, "tie-break changed")
        return {
            "candidate_route_tuples": [
                list(hypothesis.route_ids) for hypothesis in candidates
            ],
            "selected": list(chosen.route_ids),
            "score": score,
        }

    def t11_transit_path_support_selection():
        source_mask = (1 << 0) | (1 << 3)
        victim_mask = (1 << 12) | (1 << 15)
        desired = routes.combine_route_ids(
            [
                routes.route_id_for(0, 15),
                routes.route_id_for(3, 12),
            ],
            expected_k=2,
        )
        transit_probabilities = probabilities_for_mask(
            desired.transit_mask,
            positive=0.97,
            negative=0.03,
        )
        path_probabilities = probabilities_for_mask(
            desired.path_mask,
            positive=0.97,
            negative=0.03,
        )
        chosen, score = helper.choose_endpoint_preserving_hypothesis(
            routes,
            source_mask=source_mask,
            victim_mask=victim_mask,
            k=2,
            transit_probabilities=transit_probabilities,
            path_probabilities=path_probabilities,
        )
        require(chosen.route_ids == desired.route_ids, "support selection")
        return {"selected": list(chosen.route_ids), "score": score}

    def t12_endpoint_repair_reference():
        raw_source_mask = (1 << 0) | (1 << 1)
        raw_victim_mask = 1 << 15
        source_probabilities = [
            0.98, 0.55, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05,
        ]
        victim_probabilities = probabilities_for_mask(
            1 << 15,
            positive=0.98,
            negative=0.02,
        )
        expected = routes.build_route(0, 15)
        transit_probabilities = probabilities_for_mask(
            expected.transit_mask,
            positive=0.9,
            negative=0.1,
        )
        path_probabilities = probabilities_for_mask(
            expected.path_mask,
            positive=0.9,
            negative=0.1,
        )
        hypothesis, diagnostics = helper.reference_minimum_repair_k1_k2(
            routes,
            raw_source_mask=raw_source_mask,
            raw_victim_mask=raw_victim_mask,
            k=1,
            source_probabilities=source_probabilities,
            victim_probabilities=victim_probabilities,
            transit_probabilities=transit_probabilities,
            path_probabilities=path_probabilities,
        )
        require(hypothesis is not None, "repair found no solution")
        require(hypothesis.route_ids == (expected.route_id,), "repair route")
        result = helper.assemble_result(
            raw_graph_probability=0.9,
            raw_graph_prediction=1,
            raw_attacker_count_candidate=1,
            raw_source_mask=raw_source_mask,
            raw_transit_mask=0,
            raw_victim_mask=raw_victim_mask,
            raw_path_mask=0,
            hypothesis=hypothesis,
            route_consistency_score=float(diagnostics["route_support"]),
            endpoint_repaired=True,
            low_support=False,
        )
        require(result.raw_graph_prediction == 1, "repair changed graph")
        require(result.effective_attacker_count == 1, "repair changed count")
        require(result.source_repair_count == 1, "source repair count")
        require(result.victim_repair_count == 0, "victim repair count")
        require(result.decoder_status == helper.STATUS_REPAIRED, "status")
        return {
            "selected": list(hypothesis.route_ids),
            "source_repair_count": result.source_repair_count,
            "victim_repair_count": result.victim_repair_count,
        }

    def t13_low_route_support_diagnostic():
        route = routes.build_route(0, 15)
        # Use an intentionally extreme synthetic contradiction so the route
        # support score is unambiguously below the fixed synthetic threshold.
        # D7 tests only status plumbing here; it does not calibrate or freeze
        # the future train-derived low-support threshold.
        hypothesis, score = helper.choose_endpoint_preserving_hypothesis(
            routes,
            source_mask=route.source_mask,
            victim_mask=route.victim_mask,
            k=1,
            transit_probabilities=uniform_probabilities(1e-6),
            path_probabilities=uniform_probabilities(1e-6),
        )
        require(hypothesis is not None, "legal route missing")
        synthetic_threshold = -80.0
        low_support = score < synthetic_threshold
        result = helper.assemble_result(
            raw_graph_probability=0.8,
            raw_graph_prediction=1,
            raw_attacker_count_candidate=1,
            raw_source_mask=route.source_mask,
            raw_transit_mask=0,
            raw_victim_mask=route.victim_mask,
            raw_path_mask=0,
            hypothesis=hypothesis,
            route_consistency_score=score,
            endpoint_repaired=False,
            low_support=low_support,
        )
        require(low_support, "synthetic low-support condition not reached")
        require(
            result.decoder_status == helper.STATUS_LEGAL_LOW_SUPPORT,
            "low-support status",
        )
        require(result.source_repair_count == 0, "source changed")
        require(result.victim_repair_count == 0, "victim changed")
        require(result.raw_graph_prediction == 1, "graph changed")
        return {
            "score": score,
            "synthetic_test_threshold": synthetic_threshold,
            "status": result.decoder_status,
            "note": "threshold is synthetic test input, not calibrated/frozen",
        }

    def t14_no_legal_explanation_fail_closed():
        route = routes.build_route(0, 15)
        chosen, score = helper.choose_endpoint_preserving_hypothesis(
            routes,
            source_mask=route.source_mask,
            victim_mask=route.victim_mask,
            k=1,
            transit_probabilities=uniform_probabilities(0.5),
            path_probabilities=uniform_probabilities(0.5),
            allowed_route_ids=set(),
        )
        require(chosen is None, "disallowed route unexpectedly selected")
        result = helper.assemble_result(
            raw_graph_probability=0.9,
            raw_graph_prediction=1,
            raw_attacker_count_candidate=1,
            raw_source_mask=route.source_mask,
            raw_transit_mask=0,
            raw_victim_mask=route.victim_mask,
            raw_path_mask=0,
            hypothesis=None,
            route_consistency_score=score,
            endpoint_repaired=False,
            low_support=False,
        )
        require(result.raw_graph_prediction == 1, "graph changed")
        require(result.effective_attacker_count == 1, "count changed")
        require(
            result.decoder_status == helper.STATUS_NO_SOLUTION,
            "fail-closed status",
        )
        require(result.decoded_path_mask == 0, "invented certified path")
        return {"status": result.decoder_status}

    def t15_graph_count_immutability_property():
        checks = 0
        for _ in range(256):
            graph = random.randint(0, 1)
            count = random.randint(1, 4)
            source = random.randrange(16)
            victim = random.randrange(15)
            if victim >= source:
                victim += 1
            hypothesis = routes.combine_route_ids(
                [routes.route_id_for(source, victim)],
                expected_k=1,
            )
            if graph == 1 and count != 1:
                # Use a legal K-count hypothesis for the active property.
                pairs = []
                used_sources = set()
                cursor = 0
                while len(pairs) < count:
                    s = cursor % 16
                    v = (15 - cursor) % 16
                    cursor += 1
                    if s == v or s in used_sources:
                        continue
                    used_sources.add(s)
                    pairs.append((s, v))
                hypothesis = routes.combine_route_ids(
                    [
                        routes.route_id_for(source_id, victim_id)
                        for source_id, victim_id in pairs
                    ],
                    expected_k=count,
                )
            result = helper.assemble_result(
                raw_graph_probability=0.75 if graph else 0.25,
                raw_graph_prediction=graph,
                raw_attacker_count_candidate=count,
                raw_source_mask=hypothesis.source_mask,
                raw_transit_mask=0,
                raw_victim_mask=hypothesis.victim_mask,
                raw_path_mask=0,
                hypothesis=hypothesis if graph else None,
                route_consistency_score=-1.0,
                endpoint_repaired=False,
                low_support=False,
            )
            require(
                result.raw_graph_prediction == graph,
                "property graph changed",
            )
            require(
                result.effective_attacker_count == (count if graph else 0),
                "property count changed",
            )
            checks += 1
        return {"random_property_checks": checks}

    run_test("D7-T01", "route_inventory", t01_route_inventory)
    run_test("D7-T02", "route_id_roundtrip", t02_route_id_roundtrip)
    run_test("D7-T03", "topology_and_xy_legality", t03_topology_and_xy_legality)
    run_test("D7-T04", "inactive_raw", t04_inactive_raw)
    run_test("D7-T05", "K1_exact_preservation", t05_k1_exact_preservation)
    run_test("D7-T06", "K2_K4_union_semantics", t06_k2_k4_union_semantics)
    run_test("D7-T07", "shared_victim", t07_shared_victim)
    run_test("D7-T08", "role_overlap_across_routes", t08_role_overlap_across_routes)
    run_test("D7-T09", "overlapping_path_union", t09_overlapping_path_union)
    run_test("D7-T10", "endpoint_assignment_tie_break", t10_endpoint_assignment_and_tie_break)
    run_test("D7-T11", "transit_path_support_selection", t11_transit_path_support_selection)
    run_test("D7-T12", "endpoint_repair_reference", t12_endpoint_repair_reference)
    run_test("D7-T13", "low_route_support_diagnostic", t13_low_route_support_diagnostic)
    run_test("D7-T14", "no_legal_explanation_fail_closed", t14_no_legal_explanation_fail_closed)
    run_test("D7-T15", "graph_count_immutability_property", t15_graph_count_immutability_property)

    require(
        all(result["status"] == "PASS" for result in test_results),
        "one or more D7 tests failed",
    )

    shutil.copy2(package_helper_path, frozen_helper_path)

    inventory = {
        "mesh_rows": routes.MESH_ROWS,
        "mesh_columns": routes.MESH_COLUMNS,
        "router_count": routes.ROUTER_COUNT,
        "ordered_route_count": routes.ORDERED_ROUTE_COUNT,
        "directed_edge_count": len(routes.expected_directed_edges()),
        "route_id_min": min(record.route_id for record in routes.ROUTE_TABLE),
        "route_id_max": max(record.route_id for record in routes.ROUTE_TABLE),
        "route_library_sha256": sha256_file(route_library_path),
    }
    inventory_path = out / "V5_P3_D7_EPLR_V1_ROUTE_LIBRARY_INVENTORY.json"
    atomic_json(inventory_path, inventory)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Audit the frozen canonical 4x4 XY route library and validate "
            "EPLR-V1 semantics using synthetic/unit tests only."
        ),
        "route_library": inventory,
        "tests": {
            "total": len(test_results),
            "passed": sum(
                result["status"] == "PASS" for result in test_results
            ),
            "failed": sum(
                result["status"] != "PASS" for result in test_results
            ),
            "results": test_results,
        },
        "frozen_reference_helper": {
            "path": str(frozen_helper_path),
            "sha256": sha256_file(frozen_helper_path),
            "classification": (
                "D7-tested reference semantics; not production decoder"
            ),
        },
        "decision": {
            "route_library_certified_for_EPLR": True,
            "shared_victim_semantics_pass": True,
            "role_overlap_semantics_pass": True,
            "graph_count_immutability_pass": True,
            "endpoint_preservation_feasibility_pass": True,
            "synthetic_endpoint_repair_pass": True,
            "fail_closed_behavior_pass": True,
            "D8_authorized": True,
            "validation_replay_authorized": False,
            "sealed_test_evaluation_authorized": False,
            "next_stage": (
                "V5_P3_D8_EPLR_V1_DETERMINISTIC_IMPLEMENTATION_"
                "AND_A_TRAIN_ONLY_CALIBRATION"
            ),
        },
        "governance": {
            "model_loaded": False,
            "logits_loaded": False,
            "training_data_loaded": False,
            "validation_data_loaded": False,
            "test_tensor_loaded": False,
            "threshold_tuning_performed": False,
            "production_decoder_installed": False,
        },
        "provenance": {
            "D6_report_sha256": sha256_file(d6_report_path),
            "D6_lock_sha256": sha256_file(d6_lock_path),
            "D6_semantic_sha256": sha256_file(semantic_path),
            "D6_objective_sha256": sha256_file(objective_path),
            "D6_output_contract_sha256": sha256_file(
                output_contract_path
            ),
            "D6_D7_matrix_sha256": sha256_file(d7_matrix_path),
            "route_library_sha256": sha256_file(route_library_path),
            "frozen_reference_helper_sha256": sha256_file(
                frozen_helper_path
            ),
            "installed_script_sha256": sha256_file(installed_script),
        },
    }

    report_path = out / f"{STAGE}_REPORT.json"
    lock_path = out / f"{STAGE}_LOCK.json"
    atomic_json(report_path, report)
    atomic_json(
        lock_path,
        {
            "stage": STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(report_path),
            "route_library_sha256": sha256_file(route_library_path),
            "route_inventory_sha256": sha256_file(inventory_path),
            "frozen_reference_helper_sha256": sha256_file(
                frozen_helper_path
            ),
            "test_count": len(test_results),
            "test_pass_count": len(test_results),
            "D8_authorized": True,
            "validation_data_loaded": False,
            "test_tensor_loaded": False,
        },
    )
    atomic_text(out / f"{STAGE}_COMPLETE", f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print("audit_script_revision=v2_low_support_synthetic_stimulus")
    print("route_library_routes=240")
    print("route_library_directed_edges=48")
    print(f"synthetic_unit_tests={len(test_results)}")
    print(f"synthetic_unit_tests_passed={len(test_results)}")
    print("shared_victim_semantics_pass=true")
    print("role_overlap_semantics_pass=true")
    print("graph_count_immutability_pass=true")
    print("endpoint_preservation_feasibility_pass=true")
    print("synthetic_endpoint_repair_pass=true")
    print("fail_closed_behavior_pass=true")
    print("production_decoder_installed=false")
    print("training_data_loaded=false")
    print("validation_data_loaded=false")
    print("test_tensor_loaded=false")
    print("D8_authorized=true")
    print("validation_replay_authorized=false")
    print(
        "next_stage="
        "V5_P3_D8_EPLR_V1_DETERMINISTIC_IMPLEMENTATION_"
        "AND_A_TRAIN_ONLY_CALIBRATION"
    )
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
