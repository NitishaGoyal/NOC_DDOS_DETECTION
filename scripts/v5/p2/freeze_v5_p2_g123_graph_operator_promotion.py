#!/usr/bin/env python3
"""Freeze the joint G1/G2/G3 review and Task-D challenger decision."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

STAGE = "V5_P2_G123_GRAPH_OPERATOR_PROMOTION"
COMPLETE = f"{STAGE}_COMPLETE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_by_operator(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["operator"]: row for row in rows}


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        print(f"STOP: output directory already exists: {out}")
        return 2
    out.mkdir(parents=True)

    g0 = repo / "reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol"
    g1 = repo / "reports/v5/p2_g1c_source_only_graph_matrix_aggregation"
    g2 = repo / "reports/v5/p2_g2_direct_graph_matrix_aggregation"
    g3 = repo / "reports/v5/p2_g3_role_aware_multilabel_matrix_aggregation"

    paths = {
        "stop_go": g0 / "V5_P2_G0_STOP_GO_CRITERIA.json",
        "stage_plan": g0 / "V5_P2_G0_STAGE_PLAN.json",
        "task_matrix": g0 / "V5_P2_G0_TASK_AND_MODEL_MATRIX.json",
        "seed_policy": g0 / "V5_P2_G0_TRAINING_SELECTION_AND_SEED_POLICY.json",
        "test_policy": g0 / "V5_P2_G0_P2_TEST_ACCESS_POLICY.json",
        "g1_report": g1 / "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION.json",
        "g1_lock": g1 / "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION_LOCK.json",
        "g1_summary": g1 / "V5_P2_G1C_THREE_SEED_SUMMARY.csv",
        "g2_report": g2 / "V5_P2_G2_DIRECT_GRAPH_MATRIX_AGGREGATION.json",
        "g2_lock": g2 / "V5_P2_G2_DIRECT_GRAPH_MATRIX_AGGREGATION_LOCK.json",
        "g2_summary": g2 / "V5_P2_G2_THREE_SEED_SUMMARY.csv",
        "g3_report": g3 / "V5_P2_G3_ROLE_AWARE_MULTILABEL_MATRIX_AGGREGATION.json",
        "g3_lock": g3 / "V5_P2_G3_ROLE_AWARE_MULTILABEL_MATRIX_AGGREGATION_LOCK.json",
        "g3_summary": g3 / "V5_P2_G3_THREE_SEED_SUMMARY.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        report = {"stage": STAGE, "status": "HOLD", "missing": missing}
        write_json(out / f"{STAGE}.json", report)
        (out / f"{STAGE}_HOLD").write_text(f"{STAGE}_HOLD\n", encoding="utf-8")
        return 1

    for prefix in ("g1", "g2", "g3"):
        lock = read_json(paths[f"{prefix}_lock"])
        if lock.get("architecture_selected") is not False:
            raise RuntimeError(f"{prefix.upper()} unexpectedly selected an architecture")
        if lock.get("p2_test_directory_enumerated") is not False:
            raise RuntimeError(f"{prefix.upper()} test directory was enumerated")
        if lock.get("p2_test_tensors_deserialized") is not False:
            raise RuntimeError(f"{prefix.upper()} test tensors were deserialized")
        if lock.get("report_sha256") != sha256_file(paths[f"{prefix}_report"]):
            raise RuntimeError(f"{prefix.upper()} report SHA mismatch")
        if lock.get("summary_csv_sha256") != sha256_file(paths[f"{prefix}_summary"]):
            raise RuntimeError(f"{prefix.upper()} summary SHA mismatch")

    task_matrix = read_json(paths["task_matrix"])
    seed_policy = read_json(paths["seed_policy"])
    test_policy = read_json(paths["test_policy"])
    task_d = next(task for task in task_matrix["tasks"] if task["task_id"] == "D_FULL_MULTITASK_SYSTEM")
    if task_d["models"] != [
        "EXISTING_B3_CONV1D_COUNT4",
        "CONV1D_PLUS_ONE_PROMOTED_GRAPH_OPERATOR_COUNT4",
    ]:
        raise RuntimeError("Task-D candidate contract changed")
    if seed_policy["training_base"]["finalist_stability_seeds"] != [107, 117, 127, 137, 147]:
        raise RuntimeError("Task-D finalist seed contract changed")
    if test_policy["g1_to_g6"]["p2_test_tensor_access_allowed"] is not False:
        raise RuntimeError("G0 unexpectedly authorizes test tensors")

    g1_rows = read_csv_by_operator(paths["g1_summary"])
    g2_rows = read_csv_by_operator(paths["g2_summary"])
    g3_rows = read_csv_by_operator(paths["g3_summary"])
    for rows in (g1_rows, g2_rows, g3_rows):
        if not {"conv1d", "gcnconv", "graphconv"}.issubset(rows):
            raise RuntimeError("required operators missing from aggregate summary")

    c1, gr1 = g1_rows["conv1d"], g1_rows["graphconv"]
    c2, gr2 = g2_rows["conv1d"], g2_rows["graphconv"]
    c3, gr3 = g3_rows["conv1d"], g3_rows["graphconv"]

    deltas = {
        "g1_source_f1": f(gr1, "source_f1_mean") - f(c1, "source_f1_mean"),
        "g1_exact_source_set_attack": f(gr1, "exact_set_attack_mean") - f(c1, "exact_set_attack_mean"),
        "g2_graph_balanced_accuracy": f(gr2, "graph_balanced_accuracy_mean") - f(c2, "graph_balanced_accuracy_mean"),
        "g2_graph_f1": f(gr2, "graph_f1_mean") - f(c2, "graph_f1_mean"),
        "g2_graph_fpr": f(gr2, "graph_fpr_mean") - f(c2, "graph_fpr_mean"),
        "g3_macro_role_f1": f(gr3, "macro_role_node_f1_mean") - f(c3, "macro_role_node_f1_mean"),
        "g3_macro_exact_attack": f(gr3, "macro_role_exact_set_attack_mean") - f(c3, "macro_role_exact_set_attack_mean"),
        "g3_joint_exact_all_roles_attack": f(gr3, "joint_exact_all_roles_attack_mean") - f(c3, "joint_exact_all_roles_attack_mean"),
        "g3_source_f1": f(gr3, "source_f1_mean") - f(c3, "source_f1_mean"),
        "g3_transit_f1": f(gr3, "transit_f1_mean") - f(c3, "transit_f1_mean"),
        "g3_victim_f1": f(gr3, "victim_f1_mean") - f(c3, "victim_f1_mean"),
        "g3_path_f1": f(gr3, "path_f1_mean") - f(c3, "path_f1_mean"),
    }

    role_deltas = [
        deltas["g3_source_f1"],
        deltas["g3_transit_f1"],
        deltas["g3_victim_f1"],
        deltas["g3_path_f1"],
    ]
    case2_checks = {
        "graph_balanced_accuracy_gain_at_least_0_02": deltas["g2_graph_balanced_accuracy"] >= 0.02,
        "fpr_reduced": deltas["g2_graph_fpr"] < 0.0,
        "at_least_three_role_f1_gains_at_least_0_02": sum(value >= 0.02 for value in role_deltas) >= 3,
        "no_role_degrades_more_than_0_02": min(role_deltas) >= -0.02,
        "strict_exactness_gain_at_least_0_03": deltas["g3_joint_exact_all_roles_attack"] >= 0.03,
    }
    case2_pass = all(case2_checks.values())
    case4_applies = (
        deltas["g2_graph_balanced_accuracy"] < 0.02
        or deltas["g3_macro_role_f1"] < 0.02
    )

    # The frozen stage plan still calls for one full-multitask comparison.
    # Therefore GraphConv is advanced only as the strongest reviewer challenger;
    # it is not promoted as the selected architecture.
    decision = {
        "task_d_challenger": "graphconv",
        "task_d_candidates": ["conv1d", "graphconv"],
        "promotion_scope": "TASK_D_REVIEWER_CHALLENGER_ONLY",
        "g0_case_2_material_promotion_pass": case2_pass,
        "g0_case_4_lightweight_path_applies": case4_applies,
        "main_lightweight_path_before_task_d": "EXISTING_B3_CONV1D_COUNT4",
        "reason": (
            "GraphConv is the strongest non-attention operator across G1-G3 and "
            "is the single frozen Task-D challenger. It does not satisfy every "
            "G0 case-2 material-gain condition, so B3 Conv1D remains the main "
            "lightweight path pending the prescribed full-multitask comparison."
        ),
        "architecture_selected": False,
    }

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "joint_review": {
            "deltas_graphconv_minus_conv1d": deltas,
            "case2_checks": case2_checks,
            "case2_pass": case2_pass,
            "case4_applies": case4_applies,
        },
        "decision": decision,
        "task_d": {
            "targets": task_d["targets"],
            "seeds": [107, 117, 127, 137, 147],
            "total_runs": 10,
            "loss_and_checkpoint_selection": "reuse frozen P2 B1 full multitask protocol exactly",
            "threshold_tuning_during_training": False,
        },
        "provenance": {name: sha256_file(path) for name, path in paths.items()},
        "security_boundary": {
            "test_directory_enumerated": False,
            "test_tensors_deserialized": False,
            "test_evaluation_performed": False,
            "quantization_performed": False,
            "rtl_generated": False,
        },
        "next_stage": "V5_P2_TASK_D_FULL_MULTITASK_SYSTEM",
    }
    report_path = out / f"{STAGE}.json"
    write_json(report_path, report)
    lock = {
        "status": COMPLETE,
        "report_sha256": sha256_file(report_path),
        "task_d_challenger": "graphconv",
        "promotion_scope": "TASK_D_REVIEWER_CHALLENGER_ONLY",
        "g0_case_2_material_promotion_pass": case2_pass,
        "g0_case_4_lightweight_path_applies": case4_applies,
        "task_d_seeds": [107, 117, 127, 137, 147],
        "task_d_total_runs": 10,
        "architecture_selected": False,
        "p2_test_directory_enumerated": False,
        "p2_test_tensors_deserialized": False,
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(out / f"{STAGE}_LOCK.json", lock)
    (out / COMPLETE).write_text(COMPLETE + "\n", encoding="utf-8")

    print("===== V5 P2 G1/G2/G3 JOINT REVIEW =====")
    print("status: COMPLETE")
    print("task_d_challenger: graphconv")
    print("promotion_scope: TASK_D_REVIEWER_CHALLENGER_ONLY")
    print("g0_case_2_material_promotion_pass:", str(case2_pass).lower())
    print("g0_case_4_lightweight_path_applies:", str(case4_applies).lower())
    print("architecture_selected: false")
    print("test_directory_enumerated: false")
    print("test_tensors_deserialized: false")
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
