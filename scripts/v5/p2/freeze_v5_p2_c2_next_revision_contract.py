#!/usr/bin/env python3
"""
V5 P2-C2 Next Dataset and Model Revision Contract

Reads only the frozen P2-C1 interpretation report and lock. It freezes the
requirements for the next fresh dataset and model study.

No P2 prediction cache, checkpoint, dataset root, original test tensor, or
test directory is accessed.

The next revision is centered on:
- hard legitimate/control NoC traffic;
- fresh matched ATTACK/CONTROL simulation pairs;
- chronological, leakage-safe splits;
- the frozen B3 model as the reference baseline;
- a matched-pair graph ranking objective to suppress false positives;
- targeted transit, K1, and K4 coverage;
- no redesign of the already solved four-class count head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


STAGE = "V5_P2_C2_NEXT_DATASET_AND_MODEL_REVISION_CONTRACT"
COMPLETE = f"{STAGE}_COMPLETE"

EXPECTED_C1_DECISION = (
    "FREEZE_P2_ROOT_CAUSE_AS_BROAD_CONTROL_FALSE_POSITIVE_"
    "DOMINANCE_AND_PRIORITIZE_HARD_LEGITIMATE_TRAFFIC"
)

EXPECTED_FACTS = {
    "false_positive_windows": 2307,
    "false_negative_windows": 35,
    "pairs_with_false_positive": 60,
    "false_positive_scope": "broad_across_control_scenarios",
    "false_positive_temporal_form": "persistent_false_alarm_episodes",
    "attacker_count_accuracy": 1.0,
    "weakest_localization_role": "transit",
    "weakest_k_graph_recall": 1,
    "weakest_k_all_task_exact": 4,
}

REFERENCE_ARCHITECTURE = {
    "name": "B3_CAUSAL_DEPTHWISE_SEPARABLE_CONV1D_ONLY_P2_COUNT4",
    "parameter_count": 43273,
    "selected_seed": 127,
    "selected_epoch": 59,
    "selected_checkpoint_sha256": (
        "7d4afae2214f67ecd9c65c6ff4614ba8238614234d7b07cdf1408b6d3efb07ef"
    ),
}


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    c1_dir = args.c1_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(
            f"STOP: output already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    c1_report_path = (
        c1_dir
        / "V5_P2_C1_SCENARIO_AND_ROOT_CAUSE_INTERPRETATION.json"
    )
    c1_lock_path = (
        c1_dir
        / "V5_P2_C1_SCENARIO_AND_ROOT_CAUSE_INTERPRETATION_LOCK.json"
    )
    c1_complete_path = (
        c1_dir
        / "V5_P2_C1_SCENARIO_AND_ROOT_CAUSE_INTERPRETATION_COMPLETE"
    )

    for path in (
        c1_report_path,
        c1_lock_path,
        c1_complete_path,
    ):
        if not path.is_file():
            failures.append(f"missing C1 prerequisite: {path}")

    if failures:
        hold = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "p2_prediction_cache_accessed": False,
            "p2_test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", hold)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        return 1

    c1_report = load_json(c1_report_path)
    c1_lock = load_json(c1_lock_path)

    if c1_report.get("status") != "COMPLETE":
        failures.append("C1 report is not COMPLETE")
    if c1_report.get("decision") != EXPECTED_C1_DECISION:
        failures.append("C1 decision changed")
    if (
        c1_lock.get("report_sha256")
        != sha256_file(c1_report_path)
    ):
        failures.append("C1 report SHA mismatch")
    if c1_lock.get("prediction_cache_opened") is not False:
        failures.append("C1 unexpectedly opened the prediction cache")
    if c1_lock.get("checkpoint_loaded") is not False:
        failures.append("C1 unexpectedly loaded the checkpoint")
    if c1_lock.get("test_inference_rerun") is not False:
        failures.append("C1 unexpectedly reran test inference")

    primary = c1_report["conclusions"]["primary_failure"]
    evidence = primary["evidence"]
    count = c1_report["conclusions"]["attacker_count"]
    localization = c1_report["conclusions"]["localization"]

    observed_facts = {
        "false_positive_windows": int(
            evidence["false_positive_windows"]
        ),
        "false_negative_windows": int(
            evidence["false_negative_windows"]
        ),
        "pairs_with_false_positive": int(
            evidence["pairs_with_false_positive"]
        ),
        "false_positive_scope": str(
            evidence["pair_scope"]
        ),
        "false_positive_temporal_form": str(
            evidence["temporal_form"]
        ),
        "attacker_count_accuracy": float(
            count["accuracy"]
        ),
        "weakest_localization_role": str(
            localization[
                "weakest_role_by_mean_supported_node_f1"
            ]
        ),
        "weakest_k_graph_recall": int(
            localization["weakest_k_for_graph_recall"]
        ),
        "weakest_k_all_task_exact": int(
            localization[
                "weakest_k_for_strict_attack_exact"
            ]
        ),
    }

    if observed_facts != EXPECTED_FACTS:
        failures.append(
            f"C1 frozen facts changed: {observed_facts}"
        )

    if failures:
        hold = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "p2_prediction_cache_accessed": False,
            "p2_test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", hold)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    contract_core = {
        "contract_name": (
            "V5_P2_C2_FRESH_HARD_LEGITIMATE_TRAFFIC_REVISION"
        ),
        "contract_version": 1,
        "root_cause": {
            **EXPECTED_FACTS,
            "interpretation": (
                "The main failure is persistent, broad false-positive "
                "activation on legitimate/control traffic. The next revision "
                "must improve legal/control-versus-attack separation before "
                "increasing model capacity."
            ),
        },
        "terminology": {
            "hard_legitimate_control": (
                "Fresh benign NoC traffic that produces congestion, "
                "hotspots, bursts, backpressure, or phase changes without "
                "any malicious injection."
            ),
            "legal_control_note": (
                "The project shorthand 'legal NoC traffic' means legitimate "
                "or benign control traffic, not a legal/compliance category."
            ),
        },
        "fresh_dataset_contract": {
            "working_name": (
                "V6_HARD_LEGITIMATE_MATCHED_CHRONO_GRAPH"
            ),
            "source_rule": (
                "Generate entirely fresh simulation runs. No P2 test tensor, "
                "P2 test window, or reconstructed P2 test feature may be "
                "copied into training, validation, or development analysis."
            ),
            "topology_scope": {
                "primary": "4x4 2D mesh, 16 routers",
                "reason": (
                    "Resolve the 4x4 regional-expert discrimination problem "
                    "before replication to larger meshes."
                ),
            },
            "feature_interface": {
                "stored_features": 81,
                "learned_features": "PRIMARY58",
                "topology_mask": "[16,10] corrected physical-port mask",
                "window": 32,
                "stride": 8,
                "second_normalization": False,
                "model_metadata_inputs": False,
            },
            "required_legitimate_control_families": [
                {
                    "family": "baseline_low_to_moderate_load",
                    "purpose": "preserve easy-normal coverage",
                },
                {
                    "family": "sustained_high_load",
                    "purpose": (
                        "teach high utilization without malicious injection"
                    ),
                },
                {
                    "family": "bursty_and_on_off",
                    "purpose": (
                        "separate legal bursts from attack onset"
                    ),
                },
                {
                    "family": "memory_controller_hotspot",
                    "purpose": (
                        "cover legitimate concentration near shared resources"
                    ),
                },
                {
                    "family": "phase_change_and_ramp",
                    "purpose": (
                        "cover persistent false-alarm episodes around "
                        "workload transitions"
                    ),
                },
                {
                    "family": "multi_source_synchronized_benign",
                    "purpose": (
                        "cover coordinated-looking but legitimate traffic"
                    ),
                },
                {
                    "family": "asymmetric_and_locality_skewed",
                    "purpose": (
                        "cover uneven router/port utilization"
                    ),
                },
                {
                    "family": "backpressure_and_queue_build_up",
                    "purpose": (
                        "cover non-malicious congestion signatures"
                    ),
                },
            ],
            "matched_pair_rule": {
                "required": True,
                "control_and_attack_share": [
                    "legal workload family",
                    "application/workload identity",
                    "mapping and active-core placement",
                    "simulation seed",
                    "start state",
                    "traffic phase",
                    "memory-controller/victim placement where applicable",
                    "collection duration",
                ],
                "only_attack_member_changes": [
                    "malicious injection enabled",
                    "attacker locations",
                    "attacker count K1/K2/K3/K4",
                    "attack strength/profile",
                ],
                "pair_alignment": (
                    "Use common-prefix chronological windows with identical "
                    "ordered starts for ATTACK and CONTROL."
                ),
            },
            "attack_coverage": {
                "attacker_counts": [1, 2, 3, 4],
                "priority_categories": {
                    "K1": (
                        "weakest graph-recall category in P2-C1"
                    ),
                    "K4": (
                        "weakest strict all-task exact category in P2-C1"
                    ),
                },
                "strengths": (
                    "Include low, medium, and high attack strengths, with "
                    "special emphasis on low-strength attacks overlapping "
                    "hard legitimate traffic."
                ),
                "victims": (
                    "Retain one- and multi-victim cases when supported by "
                    "the simulator and labels."
                ),
            },
            "localization_coverage": {
                "priority_role": "transit",
                "required": [
                    "balanced transit path lengths",
                    "edge, corner, and interior transit routers",
                    "overlapping and non-overlapping routes",
                    "short and long paths",
                    "all router IDs represented where physically possible",
                ],
            },
            "split_contract": {
                "split_unit": (
                    "matched simulation pair/run family, never individual "
                    "windows"
                ),
                "chronological": True,
                "no_overlap_across_splits": [
                    "run ID",
                    "matched pair",
                    "simulation seed",
                    "application mapping",
                    "generated trace",
                    "window lineage",
                ],
                "scenario_holdout": (
                    "Validation and final blind test must contain complete "
                    "unseen scenario families or workload/mapping groups, "
                    "not merely later windows from training runs."
                ),
                "test_policy": (
                    "Do not enumerate or deserialize the new blind-test "
                    "tensors before checkpoint and thresholds are frozen."
                ),
            },
            "balance_contract": {
                "pair_balance": "one ATTACK and one CONTROL member per pair",
                "scenario_balance": (
                    "No single legitimate family, attack count, strength, "
                    "victim placement, or router placement may dominate "
                    "training or validation."
                ),
                "hard_control_emphasis": (
                    "Hard legitimate controls must form a substantial, "
                    "explicitly reported portion of every non-test split."
                ),
                "shortcut_audit_required": True,
            },
            "generation_metadata": {
                "allowed_for_audit_and_split": True,
                "allowed_as_model_input": False,
                "required_fields": [
                    "run_id",
                    "pair_id",
                    "split_id",
                    "workload_family",
                    "workload_id",
                    "mapping_id",
                    "simulation_seed",
                    "traffic_phase",
                    "attack_enabled",
                    "attacker_count",
                    "attacker_nodes",
                    "victim_nodes",
                    "attack_strength",
                    "profile",
                    "start_epoch",
                    "end_epoch",
                ],
            },
        },
        "model_revision_contract": {
            "reference_baseline": REFERENCE_ARCHITECTURE,
            "baseline_first": (
                "First retrain the unchanged 43,273-parameter B3 model on "
                "the fresh dataset. This isolates dataset improvement from "
                "architecture improvement."
            ),
            "count_head": {
                "logits": 4,
                "classes": [1, 2, 3, 4],
                "change_allowed": False,
                "reason": (
                    "P2 frozen blind-test active-count accuracy is 1.0."
                ),
            },
            "primary_revision": {
                "name": (
                    "B3_COUNT4_PLUS_MATCHED_PAIR_GRAPH_MARGIN"
                ),
                "architecture_change": False,
                "parameter_change": 0,
                "inference_interface_change": False,
                "training_only_objective": (
                    "For every aligned ATTACK/CONTROL pair-window block, "
                    "enforce attack_logit >= control_logit + margin."
                ),
                "recommended_ablation_margins": [
                    0.25,
                    0.5,
                    1.0,
                ],
                "selection_rule": (
                    "Choose the margin using validation only; freeze the "
                    "search space before training."
                ),
                "reason": (
                    "Directly increases separation between matched attacks "
                    "and hard legitimate controls without adding hardware."
                ),
            },
            "secondary_revision_candidates": [
                {
                    "name": "control_false_positive_asymmetric_graph_loss",
                    "type": "training-only loss",
                    "purpose": (
                        "penalize false-positive controls more strongly while "
                        "preserving attack recall"
                    ),
                    "constraint": (
                        "must be compared against the unchanged-B3 and "
                        "matched-margin baselines"
                    ),
                },
                {
                    "name": "graph_node_evidence_consistency_regularization",
                    "type": "training-only regularizer",
                    "purpose": (
                        "discourage graph attack predictions when source/path "
                        "heads provide no coherent attack evidence"
                    ),
                    "constraint": (
                        "no separate second-stage inference system"
                    ),
                },
            ],
            "localization_priority": {
                "role": "transit",
                "allowed_change": (
                    "scenario balancing and modest loss-weight ablation"
                ),
                "forbidden": (
                    "sacrificing graph false-positive control to inflate "
                    "transit metrics"
                ),
            },
            "hardware_constraints": {
                "single_stage": True,
                "quantizable": True,
                "deterministic_interface": True,
                "regional_4x4_expert_compatible": True,
                "primary_revision_extra_parameters": 0,
            },
        },
        "experiment_order": [
            {
                "stage": "D0_FRESH_DATASET_GENERATION_AND_MANIFEST",
                "action": (
                    "Generate fresh matched legal-control/attack runs and "
                    "freeze metadata manifests."
                ),
            },
            {
                "stage": "D1_DATASET_AUDIT_AND_SHORTCUT_AUDIT",
                "action": (
                    "Audit pair alignment, labels, splits, hard-control "
                    "coverage, and forbidden metadata leakage."
                ),
            },
            {
                "stage": "D2_UNCHANGED_B3_BASELINE",
                "action": (
                    "Train unchanged B3-count4 on the new dataset with "
                    "multiple seeds."
                ),
            },
            {
                "stage": "D3_MATCHED_PAIR_MARGIN_ABLATION",
                "action": (
                    "Train the zero-parameter matched-pair graph-margin "
                    "variants using the frozen margin search."
                ),
            },
            {
                "stage": "D4_VALIDATION_SELECTION",
                "action": (
                    "Select dataset/model variant using validation graph "
                    "balanced accuracy, FPR, recall, and localization metrics."
                ),
            },
            {
                "stage": "D5_VALIDATION_THRESHOLD_FREEZE",
                "action": (
                    "Freeze one checkpoint and all thresholds."
                ),
            },
            {
                "stage": "D6_ONE_SHOT_BLIND_TEST",
                "action": (
                    "Authorize and execute exactly one fresh blind-test pass."
                ),
            },
        ],
        "acceptance_targets": {
            "development_gates": {
                "validation_graph_balanced_accuracy_min": 0.95,
                "validation_graph_fpr_max": 0.05,
                "validation_graph_recall_min": 0.95,
                "validation_count_macro_f1_min": 0.98,
                "validation_transit_node_f1_min": 0.75,
            },
            "blind_test_research_goal": {
                "graph_accuracy_min": 0.98,
                "graph_fpr_max": 0.05,
                "graph_recall_min": 0.95,
                "note": (
                    "These are project goals, not guarantees. The final paper "
                    "must report the frozen result even if a goal is missed."
                ),
            },
        },
        "prohibited_actions": [
            "Do not retrain on or copy P2 blind-test tensors/windows.",
            "Do not retune the frozen P2 checkpoint or thresholds.",
            "Do not use IDs, profiles, mappings, strengths, seeds, or split "
            "metadata as learned inputs.",
            "Do not start with a larger architecture before establishing the "
            "unchanged-B3 fresh-dataset baseline.",
            "Do not redesign the count head.",
            "Do not use a two-stage detector/localizer as the primary system.",
            "Do not inspect the new blind-test contents before authorization.",
        ],
        "security_boundary": {
            "p2_prediction_cache_accessed": False,
            "p2_checkpoint_loaded": False,
            "p2_dataset_root_accessed": False,
            "p2_test_directory_enumerated": False,
            "p2_test_tensor_contents_accessed": False,
            "p2_test_inference_rerun": False,
            "contract_source": (
                "P2-C1 frozen interpretation report and lock only"
            ),
        },
    }

    contract = {
        **contract_core,
        "contract_sha256": canonical_sha256(contract_core),
    }

    contract_path = (
        output_dir
        / "V5_P2_C2_NEXT_DATASET_AND_MODEL_REVISION_CONTRACT.json"
    )
    write_json(contract_path, contract)

    markdown = [
        "# V5 P2-C2 Next Dataset and Model Revision Contract",
        "",
        "## Frozen diagnosis",
        "",
        "- P2 graph false positives: **2,307**.",
        "- P2 graph false negatives: **35**.",
        "- Pairs with false positives: **60/69**.",
        "- False alarms are **broad** and occur in **persistent episodes**.",
        "- Active attacker-count accuracy is **100%**.",
        "- Weakest localization role: **transit**.",
        "- Weakest graph-recall category: **K1**.",
        "- Weakest strict exact category: **K4**.",
        "",
        "The next revision is therefore a **fresh hard-legitimate-traffic "
        "dataset first**, followed by a zero-extra-parameter matched-pair "
        "graph-margin objective.",
        "",
        "## Dataset",
        "",
        "Working name: `V6_HARD_LEGITIMATE_MATCHED_CHRONO_GRAPH`.",
        "",
        "Generate fresh legal/control runs covering sustained load, bursts, "
        "memory-controller hotspots, phase changes, synchronized benign "
        "sources, asymmetry, locality skew, and backpressure. Every control "
        "must be matched to attacks under the same workload, mapping, seed, "
        "phase, placement, and duration.",
        "",
        "Do not copy P2 test tensors or windows into the new dataset.",
        "",
        "## Model order",
        "",
        "1. Retrain the unchanged 43,273-parameter B3-count4 model.",
        "2. Add only a training-time matched ATTACK/CONTROL graph-margin loss.",
        "3. Compare frozen margins 0.25, 0.5, and 1.0 using validation only.",
        "4. Consider asymmetric graph loss or graph/node consistency only "
        "after the unchanged baseline and margin ablation.",
        "5. Keep the four-class count head unchanged.",
        "",
        "## Primary margin revision",
        "",
        "For aligned ATTACK/CONTROL windows, train the graph logits so that:",
        "",
        "`attack_logit >= control_logit + margin`",
        "",
        "This adds no inference parameters and directly targets the broad "
        "control false-positive problem.",
        "",
        "## Required split policy",
        "",
        "- Split complete run pairs, never windows.",
        "- Keep run ID, pair, seed, mapping, trace lineage, and scenario family "
        "disjoint across splits.",
        "- Hold out complete scenario/workload groups.",
        "- Do not inspect the new blind-test tensor contents before checkpoint "
        "and thresholds are frozen.",
        "",
        "## Required experiment sequence",
        "",
        "`D0 dataset generation -> D1 audits -> D2 unchanged B3 -> "
        "D3 matched-margin ablation -> D4 selection -> D5 threshold freeze -> "
        "D6 one-shot blind test`",
        "",
        "## Integrity",
        "",
        "- P2 prediction cache accessed: **no**.",
        "- P2 checkpoint loaded: **no**.",
        "- P2 test tensors accessed: **no**.",
        "- P2 inference rerun: **no**.",
        "",
    ]

    markdown_path = (
        output_dir
        / "V5_P2_C2_NEXT_DATASET_AND_MODEL_REVISION_CONTRACT.md"
    )
    atomic_write(markdown_path, "\n".join(markdown))

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "decision": (
            "FREEZE_FRESH_HARD_LEGITIMATE_MATCHED_DATASET_AND_"
            "ZERO_PARAMETER_PAIR_MARGIN_STUDY"
        ),
        "contract": contract,
        "artifacts": {
            "contract_json": [
                str(contract_path),
                sha256_file(contract_path),
            ],
            "contract_markdown": [
                str(markdown_path),
                sha256_file(markdown_path),
            ],
        },
        "provenance": {
            "c1_report_sha256": sha256_file(c1_report_path),
            "c1_lock_sha256": sha256_file(c1_lock_path),
        },
        "security_boundary": contract["security_boundary"],
        "failures": failures,
        "warnings": warnings,
        "next_stage": (
            "V6_D0_FRESH_HARD_LEGITIMATE_DATASET_GENERATION_PLAN"
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    lock = {
        "status": COMPLETE,
        "decision": (
            "FREEZE_FRESH_HARD_LEGITIMATE_MATCHED_DATASET_AND_"
            "ZERO_PARAMETER_PAIR_MARGIN_STUDY"
        ),
        "report_sha256": sha256_file(report_path),
        "contract_file_sha256": sha256_file(contract_path),
        "contract_sha256": contract["contract_sha256"],
        "contract_markdown_sha256": sha256_file(markdown_path),
        "root_cause": EXPECTED_FACTS,
        "reference_architecture": REFERENCE_ARCHITECTURE,
        "p2_prediction_cache_accessed": False,
        "p2_checkpoint_loaded": False,
        "p2_test_directory_enumerated": False,
        "p2_test_tensor_contents_accessed": False,
        "p2_test_inference_rerun": False,
        "next_stage": (
            "V6_D0_FRESH_HARD_LEGITIMATE_DATASET_GENERATION_PLAN"
        ),
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(
        output_dir / f"{STAGE}_LOCK.json",
        lock,
    )
    atomic_write(
        output_dir / COMPLETE,
        COMPLETE + "\n",
    )

    print("===== V5 P2-C2 NEXT DATASET + MODEL REVISION CONTRACT =====")
    print("status: COMPLETE")
    print(
        "decision: "
        "FREEZE_FRESH_HARD_LEGITIMATE_MATCHED_DATASET_AND_"
        "ZERO_PARAMETER_PAIR_MARGIN_STUDY"
    )
    print(
        "dataset_working_name: "
        "V6_HARD_LEGITIMATE_MATCHED_CHRONO_GRAPH"
    )
    print(
        "primary_dataset_target: "
        "hard_legitimate_control_traffic"
    )
    print(
        "primary_model_revision: "
        "B3_COUNT4_PLUS_MATCHED_PAIR_GRAPH_MARGIN"
    )
    print("primary_revision_extra_parameters: 0")
    print("count_head_change_allowed: false")
    print("priority_localization_role: transit")
    print("priority_attack_category_graph: K1")
    print("priority_attack_category_exact: K4")
    print(
        "contract_sha256:",
        contract["contract_sha256"],
    )
    print("p2_prediction_cache_accessed: false")
    print("p2_checkpoint_loaded: false")
    print("p2_test_directory_enumerated: false")
    print("p2_test_tensor_contents_accessed: false")
    print("p2_test_inference_rerun: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print(
        "next_stage: "
        "V6_D0_FRESH_HARD_LEGITIMATE_DATASET_GENERATION_PLAN"
    )
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
