from __future__ import annotations
import argparse, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

STAGE = "V5_P3_D6_EPLR_V1_SEMANTIC_AND_OBJECTIVE_SPECIFICATION_FREEZE"
LABEL = "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic — EPLR-V1 Decoder Design Freeze"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)

def write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--installed-script", required=True)
    a = p.parse_args()

    repo = Path(a.repo).expanduser().resolve()
    out = Path(a.output_dir).expanduser().resolve()
    installed = Path(a.installed_script).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    d5 = repo / "reports/v5/p3_d5_tranche_a_decoder_review_and_4x4_expert_interface_freeze"
    d5_report = d5 / "V5_P3_D5_TRANCHE_A_DECODER_REVIEW_AND_4X4_EXPERT_INTERFACE_FREEZE_REPORT.json"
    d5_lock = d5 / "V5_P3_D5_TRANCHE_A_DECODER_REVIEW_AND_4X4_EXPERT_INTERFACE_FREEZE_LOCK.json"
    d5_if = d5 / "V5_P3_4X4_DYNAMIC70_EXPERT_INTERFACE_CONTRACT.json"
    d5_dec = d5 / "V5_P3_TRANCHE_A_DECODER_POLICY_DECISION.json"
    h0 = repo / "artifacts/rtl/v5_p3_h0_current_4x4_dynamic70_rtl_handoff/V5_P3_H0_CURRENT_4X4_DYNAMIC70_RTL_HANDOFF_PREQUANTIZATION_REFERENCE.zip"
    routes = repo / "src/decoders/v5_xy_route_library.py"
    canonical = repo / "src/decoders/v5_legal_xy_exact_decoder_certified.py"
    hybrid = repo / "src/decoders/v5_p3_a1_exact_decoder_certified_hybrid.py"

    required = [d5_report, d5_lock, d5_if, d5_dec, h0, routes, canonical, hybrid]
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")

    d5r = json.loads(d5_report.read_text())
    d5l = json.loads(d5_lock.read_text())
    if d5r.get("status") != "PASS":
        raise RuntimeError("D5 report is not PASS")
    if d5l.get("report_sha256") != sha(d5_report):
        raise RuntimeError("D5 report/lock mismatch")
    if d5l.get("interface_contract_sha256") != sha(d5_if):
        raise RuntimeError("D5 interface/lock mismatch")
    if d5l.get("decoder_decision_sha256") != sha(d5_dec):
        raise RuntimeError("D5 decoder decision/lock mismatch")

    semantic = {
      "stage": STAGE,
      "status": "FROZEN",
      "decoder": {"name": "V5-P3 EPLR-V1", "expansion": "Endpoint-Preserving Legal-XY Repair"},
      "hardware_boundary": {"inside_RTL": False, "input": "69 raw logits", "H0_sha256": sha(h0)},
      "raw_policy": {
        "graph": "sigmoid(graph_logit) >= 0.5; immutable",
        "active_count": "1 + argmax(count_logits); immutable",
        "inactive_effective_count": 0,
        "inactive_masks": "all zero",
        "source_mask": "sigmoid(source_logits) >= 0.5",
        "victim_mask": "sigmoid(victim_logits) >= 0.5"
      },
      "route_semantics": {
        "topology": "fixed 4x4 row-major mesh",
        "route_library": "canonical V5 legal XY route library",
        "active_route_count": "exactly K",
        "unique_attacker_sources": True,
        "shared_victims_allowed": True,
        "victim_union_cardinality_may_be_less_than_K": True,
        "source_mask": "union of route sources",
        "victim_mask": "union of route victims",
        "transit_mask": "union of internal transit routers",
        "path_mask": "union router bitmap, not ordered route",
        "global_endpoint_transit_disjointness": False,
        "endpoint_on_one_route_may_be_transit_on_another": True
      },
      "endpoint_policy": {
        "first": "preserve Raw source and victim masks exactly when K legal routes explain them",
        "repair_only_if_preservation_infeasible": True,
        "repair": "minimum confidence-sensitive source/victim edits",
        "graph_remains_immutable": True,
        "count_remains_immutable": True
      },
      "statuses": [
        "INACTIVE_RAW",
        "RAW_ENDPOINTS_LEGAL",
        "RAW_ENDPOINTS_LEGAL_LOW_ROUTE_SUPPORT",
        "ENDPOINT_REPAIR_APPLIED",
        "NO_CERTIFIED_LEGAL_EXPLANATION"
      ],
      "diagnostics": [
        "route_consistency_score",
        "selected_route_ids",
        "source_repair_count",
        "victim_repair_count",
        "endpoint_preserved_but_low_route_support"
      ],
      "failure_policy": {"fail_closed": True, "old_A1_fallback": False, "beam_selected": False}
    }

    objective = {
      "stage": STAGE,
      "status": "FROZEN",
      "style": "lexicographic",
      "priorities": [
        "P0 freeze Raw graph and active count",
        "P1 find exact endpoint-preserving K-route explanation",
        "P2 only if infeasible, minimize confidence-sensitive endpoint repair cost",
        "P3 minimize unweighted endpoint Hamming edits",
        "P4 maximize transit/path Bernoulli log-likelihood over route-union masks",
        "P5 deterministic lexicographically smallest route-ID tuple"
      ],
      "default_role_weights": {"transit": 1.0, "path": 1.0},
      "D8_train_only_calibration": {
        "role_weights": True,
        "endpoint_repair_cost": True,
        "low_route_support_threshold": True,
        "graph_threshold": False,
        "count_policy": False,
        "endpoint_thresholds": False,
        "validation_guided_selection": False
      },
      "top_k_endpoint_pruning": False,
      "deterministic": True,
      "fail_closed": True
    }

    outputs = {
      "stage": STAGE,
      "status": "FROZEN",
      "per_sample": [
        "raw_graph_probability",
        "raw_graph_prediction",
        "raw_attacker_count_candidate",
        "effective_attacker_count",
        "raw_source_bitmap",
        "raw_transit_bitmap",
        "raw_victim_bitmap",
        "raw_path_bitmap",
        "decoded_source_bitmap",
        "decoded_transit_bitmap",
        "decoded_victim_bitmap",
        "decoded_path_bitmap",
        "selected_route_ids",
        "route_consistency_score",
        "source_repair_count",
        "victim_repair_count",
        "endpoint_preserved_but_low_route_support",
        "decoder_status"
      ],
      "path_label_semantics": "union bitmap",
      "ordered_routes_are_metadata_only": True
    }

    d7 = {
      "stage": "V5_P3_D7_EPLR_V1_ROUTE_LIBRARY_AND_SYNTHETIC_UNIT_TESTS",
      "status": "PREFROZEN_TEST_MATRIX",
      "tests": [
        "inactive Raw output",
        "K1 exact route preservation",
        "K2-K4 route unions",
        "shared victim",
        "endpoint/transit overlap across routes",
        "overlapping path unions",
        "endpoint cardinality repair",
        "legal endpoints with low route support",
        "no legal explanation fail-closed",
        "deterministic tie-break",
        "graph/count immutability property",
        "complete canonical route-library inventory"
      ],
      "training_data_access": False,
      "validation_data_access": False,
      "test_data_access": False
    }

    governance = """# V5-P3 EPLR-V1 governance

D6 freezes design only. It loads no model tensors, logits, training data,
validation data, or sealed-test data.

Sequence:
1. D7 route-library and synthetic unit tests.
2. D8 deterministic implementation and A-train-only calibration.
3. D9 one frozen Tranche-A validation exploratory replay.
4. D10 Raw/A0/old-A1/EPLR comparison and carry-forward decision.

D9 exploratory gate:
- graph predictions exactly equal Raw;
- active count predictions exactly equal Raw;
- source exact drop <= 1 percentage point;
- victim exact drop <= 1 percentage point;
- transit exact gain >= 5 percentage points;
- strict exact gain >= 5 percentage points;
- path does not materially degrade;
- every decoder failure is explicit and fail closed.

This branch does not block H1 quantization, shared 8x8 wrapper engineering, or
Tranche-B preparation. A-test remains sealed.
"""

    semantic_path = out / "V5_P3_EPLR_V1_SEMANTIC_CONTRACT.json"
    objective_path = out / "V5_P3_EPLR_V1_OBJECTIVE_CONTRACT.json"
    output_path = out / "V5_P3_EPLR_V1_OUTPUT_CONTRACT.json"
    d7_path = out / "V5_P3_EPLR_V1_D7_UNIT_TEST_MATRIX.json"
    gov_path = out / "V5_P3_EPLR_V1_GOVERNANCE_NOTE.md"

    write_json(semantic_path, semantic)
    write_json(objective_path, objective)
    write_json(output_path, outputs)
    write_json(d7_path, d7)
    write_text(gov_path, governance)

    report = {
      "stage": STAGE,
      "status": "PASS",
      "created_utc": datetime.now(timezone.utc).isoformat(),
      "campaign_label": LABEL,
      "frozen": {
        "semantic_sha256": sha(semantic_path),
        "objective_sha256": sha(objective_path),
        "output_sha256": sha(output_path),
        "D7_matrix_sha256": sha(d7_path),
        "governance_sha256": sha(gov_path)
      },
      "decision": {
        "D7_authorized": True,
        "D8_authorized": False,
        "validation_replay_authorized": False,
        "sealed_test_evaluation_authorized": False,
        "H1_parallel_work": True,
        "Tranche_B_parallel_work": True,
        "next_stage": "V5_P3_D7_EPLR_V1_ROUTE_LIBRARY_AND_SYNTHETIC_UNIT_TESTS"
      },
      "provenance": {
        "D5_report_sha256": sha(d5_report),
        "D5_lock_sha256": sha(d5_lock),
        "D5_interface_sha256": sha(d5_if),
        "D5_decoder_decision_sha256": sha(d5_dec),
        "H0_archive_sha256": sha(h0),
        "route_library_sha256": sha(routes),
        "canonical_decoder_sha256": sha(canonical),
        "hybrid_decoder_sha256": sha(hybrid),
        "installed_script_sha256": sha(installed)
      },
      "model_loaded": False,
      "logits_loaded": False,
      "training_data_loaded": False,
      "validation_data_loaded": False,
      "test_tensor_loaded": False
    }

    report_path = out / f"{STAGE}_REPORT.json"
    lock_path = out / f"{STAGE}_LOCK.json"
    write_json(report_path, report)
    write_json(lock_path, {
      "stage": STAGE,
      "status": "PASS",
      "report_sha256": sha(report_path),
      "semantic_sha256": sha(semantic_path),
      "objective_sha256": sha(objective_path),
      "output_sha256": sha(output_path),
      "D7_matrix_sha256": sha(d7_path),
      "route_library_sha256": sha(routes),
      "validation_data_loaded": False,
      "test_tensor_loaded": False
    })
    write_text(out / f"{STAGE}_COMPLETE", f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print("decoder_name=V5-P3_EPLR-V1")
    print("graph_prediction_immutable=true")
    print("active_count_prediction_immutable=true")
    print("endpoint_preserving_first=true")
    print("repair_only_when_preservation_infeasible=true")
    print("shared_victims_allowed=true")
    print("global_role_disjointness=false")
    print("decoder_inside_RTL=false")
    print("validation_data_loaded=false")
    print("test_tensor_loaded=false")
    print("D7_authorized=true")
    print("D8_authorized=false")
    print("validation_replay_authorized=false")
    print("H1_parallel_work=true")
    print("Tranche_B_parallel_work=true")
    print("next_stage=V5_P3_D7_EPLR_V1_ROUTE_LIBRARY_AND_SYNTHETIC_UNIT_TESTS")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
