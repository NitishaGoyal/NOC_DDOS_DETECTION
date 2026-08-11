#!/usr/bin/env python3
"""
V5 P2-G0 Graph-Baseline and RTL-Handoff Protocol

This append-only protocol stage inserts a controlled graph-baseline branch
between the completed C2A amendment and the Legal NoC decoder L0 stage.

It freezes:
- four distinct experimental tasks;
- the minimum operator suite and promotion rules;
- a fair common architecture/training policy;
- validation-only selection;
- stop/go criteria;
- RTL baseline selection and handoff requirements;
- an immutable G1-G6 stage plan;
- return to L0 after the graph-baseline branch.

This stage performs no training, inference, cache access, quantization,
checkpoint selection, RTL generation, or decoder implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


STAGE = "V5_P2_G0_GRAPH_BASELINE_AND_RTL_HANDOFF_PROTOCOL"
COMPLETE = f"{STAGE}_COMPLETE"

EXPECTED_C2A_COMPLETE = (
    "V5_P2_C2A_LEGAL_NOC_DECODER_AMENDMENT_COMPLETE"
)
EXPECTED_C2A_DECISION = (
    "PRESERVE_C2_AND_ADD_VALIDATION_FROZEN_LEGAL_NOC_"
    "STRUCTURED_DECODER_STUDY"
)
EXPECTED_C2A_AMENDMENT_SHA = (
    "09ac4d69d5e75b3839c9326260c1db6f12b3f8be36d7ba79e4457aded0ceeed4"
)
EXPECTED_C2A_EVALUATION_POLICY_SHA = (
    "47daeaf48999016d23a73718674b1c7536256dfce34b46fc40cb243bc05f91c7"
)
EXPECTED_C2A_STAGE_PLAN_SHA = (
    "fb991f0ed9e131cdeefa780d047bb938255bb72a33ae1f08de3150002621f211"
)
EXPECTED_C2_CONTRACT_SHA = (
    "d863a0b14d5113f283208bafa2e5804da52336f11f8ddf001e6c332c942f34b0"
)

FROZEN_TRAINING_BASE = {
    "initial_screening_seeds": [107, 117, 127],
    "finalist_stability_seeds": [107, 117, 127, 137, 147],
    "optimizer": "AdamW",
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-4,
    "max_epochs": 100,
    "minimum_epochs": 15,
    "early_stopping_patience": 12,
    "scheduler": "ReduceLROnPlateau",
    "amp": False,
    "threshold_tuning_during_training": False,
    "checkpoint_selection_split": "validation_only",
}

FROZEN_DATA_INTERFACE = {
    "dataset": "V5_P2_PAIR_ALIGNED_PRIMARY58",
    "splits": ["train", "validation"],
    "test_access": False,
    "learned_feature_set": "PRIMARY58",
    "physical_port_mask_shape": [16, 10],
    "physical_port_mask_true_count": 128,
    "num_nodes": 16,
    "topology": "4x4_2D_MESH",
    "directed_physical_edges": 48,
    "window": 32,
    "stride": 8,
    "second_normalization": False,
    "metadata_as_model_input": False,
    "pair_aligned_attack_control_batches": True,
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
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


def hold(
    output_dir: Path,
    failures: list[str],
    warnings: list[str],
) -> int:
    payload = {
        "stage": STAGE,
        "status": "HOLD",
        "failures": failures,
        "warnings": warnings,
        "training_performed": False,
        "inference_performed": False,
        "b4_validation_cache_accessed": False,
        "b6_test_cache_accessed": False,
        "p2_test_tensor_contents_accessed": False,
        "rtl_generated": False,
        "legal_decoder_implemented": False,
    }
    write_json(output_dir / f"{STAGE}.json", payload)
    atomic_write(
        output_dir / f"{STAGE}_HOLD",
        f"{STAGE}_HOLD\n",
    )
    print(f"{STAGE}_HOLD")
    for failure in failures:
        print("FAIL:", failure)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2a-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    c2a_dir = args.c2a_dir.expanduser().resolve()
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

    c2a_report_path = (
        c2a_dir
        / "V5_P2_C2A_LEGAL_NOC_DECODER_AMENDMENT.json"
    )
    c2a_lock_path = (
        c2a_dir
        / "V5_P2_C2A_LEGAL_NOC_DECODER_AMENDMENT_LOCK.json"
    )
    c2a_complete_path = (
        c2a_dir
        / "V5_P2_C2A_LEGAL_NOC_DECODER_AMENDMENT_COMPLETE"
    )
    c2a_eval_path = (
        c2a_dir
        / "V5_P2_C2A_P2_VS_V6_EVALUATION_POLICY.json"
    )
    c2a_plan_path = (
        c2a_dir
        / "V5_P2_C2A_DECODER_STAGE_PLAN.json"
    )

    prerequisite_paths = {
        "c2a_report": c2a_report_path,
        "c2a_lock": c2a_lock_path,
        "c2a_complete": c2a_complete_path,
        "c2a_evaluation_policy": c2a_eval_path,
        "c2a_stage_plan": c2a_plan_path,
    }
    for name, path in prerequisite_paths.items():
        if not path.is_file():
            failures.append(f"missing C2A prerequisite {name}: {path}")

    if failures:
        return hold(output_dir, failures, warnings)

    c2a_report = load_json(c2a_report_path)
    c2a_lock = load_json(c2a_lock_path)
    c2a_eval = load_json(c2a_eval_path)
    c2a_plan = load_json(c2a_plan_path)

    if (
        c2a_complete_path.read_text(encoding="utf-8").strip()
        != EXPECTED_C2A_COMPLETE
    ):
        failures.append("C2A COMPLETE marker content changed")
    if c2a_report.get("status") != "COMPLETE":
        failures.append("C2A report is not COMPLETE")
    if c2a_report.get("decision") != EXPECTED_C2A_DECISION:
        failures.append("C2A decision changed")
    if c2a_lock.get("amendment_file_sha256") != sha256_file(
        c2a_report_path
    ):
        failures.append("C2A amendment-file SHA mismatch")
    if (
        c2a_lock.get("amendment_sha256")
        != EXPECTED_C2A_AMENDMENT_SHA
    ):
        failures.append("C2A amendment canonical SHA changed")
    if (
        c2a_lock.get("evaluation_policy_file_sha256")
        != sha256_file(c2a_eval_path)
    ):
        failures.append("C2A evaluation-policy file SHA mismatch")
    if (
        c2a_lock.get("evaluation_policy_sha256")
        != EXPECTED_C2A_EVALUATION_POLICY_SHA
    ):
        failures.append("C2A evaluation-policy canonical SHA changed")
    if (
        c2a_lock.get("stage_plan_file_sha256")
        != sha256_file(c2a_plan_path)
    ):
        failures.append("C2A stage-plan file SHA mismatch")
    if (
        c2a_lock.get("stage_plan_sha256")
        != EXPECTED_C2A_STAGE_PLAN_SHA
    ):
        failures.append("C2A stage-plan canonical SHA changed")
    if (
        c2a_lock.get("c2_contract_sha256")
        != EXPECTED_C2_CONTRACT_SHA
    ):
        failures.append("C2 contract SHA changed through C2A")
    if c2a_lock.get("c2_modified") is not False:
        failures.append("C2A indicates C2 was modified")
    if (
        c2a_lock.get("p2_prediction_cache_accessed")
        is not False
    ):
        failures.append("C2A unexpectedly accessed a P2 cache")
    if (
        c2a_lock.get("p2_test_tensor_contents_accessed")
        is not False
    ):
        failures.append("C2A unexpectedly accessed P2 test tensors")
    if c2a_lock.get("p2_test_inference_rerun") is not False:
        failures.append("C2A unexpectedly reran P2 test inference")

    if failures:
        return hold(output_dir, failures, warnings)

    task_matrix_core = {
        "name": "V5_P2_G0_TASK_AND_MODEL_MATRIX",
        "version": 1,
        "tasks": [
            {
                "task_id": "A_SOURCE_ONLY_LOCALIZATION",
                "purpose": (
                    "UofF-equivalent binary malicious-source localization"
                ),
                "targets": ["source[16]"],
                "participating_losses": ["source_weighted_bce"],
                "excluded_losses": [
                    "graph",
                    "count",
                    "transit",
                    "victim",
                    "path",
                ],
                "graph_score": (
                    "maximum sigmoid source score across 16 routers"
                ),
                "graph_decision": (
                    "attack when at least one source prediction is positive"
                ),
                "models": [
                    "CONV1D_ONLY",
                    "CONV1D_PLUS_GCNCONV",
                    "CONV1D_PLUS_GRAPHCONV",
                    "CONV1D_PLUS_GAT_SCREENING",
                ],
            },
            {
                "task_id": "B_DIRECT_GRAPH_DETECTION",
                "purpose": (
                    "test whether learned spatial aggregation improves "
                    "attack-versus-control discrimination"
                ),
                "targets": ["graph_attack"],
                "participating_losses": ["graph_bce"],
                "excluded_losses": [
                    "count",
                    "source",
                    "transit",
                    "victim",
                    "path",
                ],
                "graph_score": "direct learned graph logit",
                "graph_readout": (
                    "identical pooling and graph head across operators"
                ),
                "models": [
                    "CONV1D_ONLY",
                    "CONV1D_PLUS_GCNCONV",
                    "CONV1D_PLUS_GRAPHCONV",
                    "CONV1D_PLUS_GAT_SCREENING",
                ],
            },
            {
                "task_id": "C_ROLE_AWARE_MULTILABEL_LOCALIZATION",
                "purpose": (
                    "measure preservation of overlapping source, transit, "
                    "victim, and path identities"
                ),
                "targets": [
                    "source[16]",
                    "transit[16]",
                    "victim[16]",
                    "path[16]",
                ],
                "label_type": "four_independent_binary_multilabel_heads",
                "mutually_exclusive_roles": False,
                "participating_losses": [
                    "source_weighted_bce",
                    "transit_weighted_bce",
                    "victim_weighted_bce",
                    "path_weighted_bce",
                ],
                "models": [
                    "CONV1D_ONLY",
                    "CONV1D_PLUS_GCNCONV",
                    "CONV1D_PLUS_GRAPHCONV",
                    "PROMOTED_GAT_ONLY",
                ],
            },
            {
                "task_id": "D_FULL_MULTITASK_SYSTEM",
                "purpose": (
                    "compare the complete B3 task interface against the "
                    "single strongest promoted graph operator"
                ),
                "targets": [
                    "graph_attack",
                    "count_K1_K4",
                    "source[16]",
                    "transit[16]",
                    "victim[16]",
                    "path[16]",
                ],
                "models": [
                    "EXISTING_B3_CONV1D_COUNT4",
                    "CONV1D_PLUS_ONE_PROMOTED_GRAPH_OPERATOR_COUNT4",
                ],
                "promotion_requirement": (
                    "operator must pass G1/G2 screening before full "
                    "multitask training"
                ),
            },
        ],
        "operator_suite": {
            "mandatory": [
                "CONV1D_ONLY",
                "GCNCONV",
                "GRAPHCONV",
            ],
            "screening_only": ["GAT"],
            "gat_promotion_rule": (
                "promote only if it beats GraphConv by at least 0.02 "
                "absolute F1 or balanced accuracy, or materially reduces "
                "multi-seed variance"
            ),
        },
    }
    task_matrix = {
        **task_matrix_core,
        "matrix_sha256": canonical_sha256(task_matrix_core),
    }

    operator_contracts_core = {
        "name": "V5_P2_G0_GRAPH_OPERATOR_CONTRACTS",
        "version": 1,
        "common_architecture": {
            "temporal_encoder": (
                "unchanged B3 causal depthwise-separable Conv1D encoder"
            ),
            "graph_layer_count": 2,
            "graph_width": (
                "equal to the B3 temporal encoder output width; G1 preflight "
                "must resolve and record the numeric width from frozen "
                "model source before training"
            ),
            "activation": (
                "same activation for every graph operator"
            ),
            "dropout": (
                "same probability and placement for every graph operator"
            ),
            "residual_policy": (
                "same residual policy for every graph operator"
            ),
            "heads": "identical task-specific heads",
            "edge_index": (
                "fixed 4x4 bidirectional physical mesh with 48 directed "
                "edges; no learned or scenario-dependent edges"
            ),
            "operator_specific_hyperparameter_search": False,
        },
        "gcnconv": {
            "equation": (
                "H' = D_hat^(-1/2) A_hat D_hat^(-1/2) H W + b"
            ),
            "self_loops": True,
            "normalization": "symmetric",
            "edge_weights": "unit",
            "bias": True,
        },
        "graphconv": {
            "equation": (
                "h'_i = W_root h_i + W_neigh "
                "sum_{j in N(i)} h_j + b"
            ),
            "self_feature": "explicit root transform",
            "neighbor_aggregation": "unnormalized sum",
            "edge_weights": "unit",
            "bias": True,
            "rtl_preference_reason": (
                "fixed neighbor lists, degree <= 4, additions plus two "
                "linear transforms, no attention softmax"
            ),
        },
        "gat": {
            "status": "screening_only",
            "head_count": (
                "single common head configuration frozen in G1 preflight"
            ),
            "self_loops": True,
            "attention_normalization": (
                "softmax over incoming neighbors including self"
            ),
            "negative_slope": 0.2,
            "promotion_required_for_later_tasks": True,
        },
        "hardware_reference_rule": {
            "pytorch_geometric_is_not_rtl_specification": True,
            "selected_operator_must_be_rewritten_explicitly": True,
            "float_equivalence_max_abs_error": 1.0e-6,
            "ranking_and_decision_equivalence": True,
        },
    }
    operator_contracts = {
        **operator_contracts_core,
        "contracts_sha256": canonical_sha256(
            operator_contracts_core
        ),
    }

    training_policy_core = {
        "name": "V5_P2_G0_TRAINING_SELECTION_AND_SEED_POLICY",
        "version": 1,
        "data_interface": FROZEN_DATA_INTERFACE,
        "training_base": FROZEN_TRAINING_BASE,
        "task_selection": {
            "A_SOURCE_ONLY_LOCALIZATION": {
                "loss": "class_weighted_source_bce_with_logits",
                "checkpoint_score": {
                    "source_average_precision": 0.50,
                    "source_auroc": 0.25,
                    "derived_graph_average_precision": 0.25,
                },
                "post_checkpoint_threshold_objective": {
                    "source_node_f1": 0.60,
                    "source_exact_set_attack": 0.40,
                },
            },
            "B_DIRECT_GRAPH_DETECTION": {
                "loss": "graph_bce_with_logits",
                "checkpoint_score": {
                    "graph_auroc": 0.50,
                    "graph_average_precision": 0.50,
                },
                "post_checkpoint_threshold_objective": (
                    "maximize validation balanced accuracy"
                ),
            },
            "C_ROLE_AWARE_MULTILABEL_LOCALIZATION": {
                "loss": (
                    "weighted source + transit + victim + path BCE losses"
                ),
                "checkpoint_score": {
                    "source_average_precision": 0.30,
                    "transit_average_precision": 0.25,
                    "victim_average_precision": 0.25,
                    "path_average_precision": 0.20,
                },
                "post_checkpoint_threshold_selection": (
                    "separate validation-only role thresholds using the "
                    "same node-F1/exact-set policy for every operator"
                ),
            },
            "D_FULL_MULTITASK_SYSTEM": {
                "loss_and_selection": (
                    "reuse frozen P2 B1 full multitask protocol"
                ),
                "operator_count": 1,
            },
        },
        "fairness_requirements": [
            "same train/validation pair manifests",
            "same batch construction",
            "same temporal encoder implementation",
            "same graph width/layer count across graph operators",
            "same seeds within each comparison",
            "same optimizer/scheduler/early stopping",
            "same checkpoint-selection metric per task",
            "same threshold-selection policy per task",
            "no test access",
        ],
        "reporting": {
            "per_seed_required": True,
            "mean_and_standard_deviation_required": True,
            "parameter_count_required": True,
            "operation_count_required": True,
            "peak_memory_estimate_required": True,
            "inference_latency_proxy_required": True,
        },
    }
    training_policy = {
        **training_policy_core,
        "policy_sha256": canonical_sha256(training_policy_core),
    }

    stop_go_core = {
        "name": "V5_P2_G0_STOP_GO_CRITERIA",
        "version": 1,
        "cases": [
            {
                "case": 1,
                "condition": (
                    "GraphConv source-only improves source F1 and exact "
                    "source set by at least 0.03 absolute, while role-aware "
                    "localization degrades or shows no all-role gain"
                ),
                "decision": (
                    "support temporal role heads plus Legal NoC decoder; "
                    "retain GraphConv as source-only/reviewer baseline"
                ),
            },
            {
                "case": 2,
                "condition": (
                    "GraphConv improves graph balanced accuracy by at least "
                    "0.02, reduces FPR without recall loss greater than 0.01, "
                    "improves at least 3 of 4 role F1 scores by at least 0.02, "
                    "degrades no role by more than 0.02, and improves strict "
                    "exactness by at least 0.03"
                ),
                "decision": (
                    "promote GraphConv into candidate final architecture; "
                    "still evaluate Legal NoC decoder as consistency module "
                    "and hardware-oriented alternative"
                ),
            },
            {
                "case": 3,
                "condition": (
                    "no graph operator beats Conv1D-only source F1 by 0.01 "
                    "or source-only results show high seed variance"
                ),
                "decision": (
                    "audit labels, edge_index, self-loops, class weighting, "
                    "positive prevalence, temporal observability, window "
                    "length, normalization, losses, and split integrity "
                    "before blaming graph learning"
                ),
            },
            {
                "case": 4,
                "condition": (
                    "GraphConv gain is below 0.02 absolute or cost is "
                    "disproportionate"
                ),
                "decision": (
                    "preserve GraphConv as reviewer baseline and continue "
                    "with B3 plus Legal NoC decoder as main lightweight path"
                ),
            },
        ],
        "return_to_l0_rule": (
            "G6 completion or an explicit G4 stop decision must return the "
            "project to V5_P2_L0_LEGAL_XY_DECODER_CONTRACT"
        ),
    }
    stop_go = {
        **stop_go_core,
        "criteria_sha256": canonical_sha256(stop_go_core),
    }

    rtl_requirements_core = {
        "name": "V5_P2_G0_RTL_HANDOFF_REQUIREMENTS",
        "version": 1,
        "handoff_purpose": (
            "provide a stable validation-selected temporal-plus-graph "
            "accelerator target while Legal NoC decoder research proceeds"
        ),
        "claim_label": (
            "VALIDATION-SELECTED GRAPH BASELINE — "
            "NOT A P2 BLIND-TESTED FINAL MODEL"
        ),
        "primary_candidate_policy": {
            "preferred": (
                "Conv1D plus GraphConv source-only localization with graph "
                "detection derived from source evidence"
            ),
            "selection_condition": (
                "must pass source-only validation, stability, and cost gates"
            ),
            "fallback": (
                "existing frozen B3 Conv1D-count4 temporal-only reference"
            ),
            "not_preselected": True,
        },
        "required_artifacts": [
            "architecture_contract.json",
            "tensor_interface.json",
            "graph_operator_equation.md",
            "fixed_4x4_topology.json",
            "selected_checkpoint.pt",
            "selected_checkpoint.sha256",
            "model_source.py",
            "explicit_graph_operator_reference.py",
            "parameter_manifest.csv",
            "primary58_indices.json",
            "feature_names.json",
            "physical_port_mask.npy",
            "edge_index.npy",
            "feature_contract.md",
            "quantization_contract.json",
            "calibration_manifest.csv",
            "fixed_point_weights.npz",
            "fixed_point_biases.npz",
            "overflow_saturation_policy.md",
            "pytorch_reference_inference.py",
            "hardware_explicit_float_reference.py",
            "integer_reference_inference.py",
            "layer_order.json",
            "golden_inputs.npz",
            "golden_outputs_float.npz",
            "golden_outputs_integer.npz",
            "layerwise_float_activations.npz",
            "layerwise_integer_activations.npz",
            "vector_manifest.csv",
            "width_contract.json",
            "accumulator_bounds.json",
            "operation_counts.json",
            "latency_throughput_targets.json",
            "memory_rom_requirements.json",
            "equivalence_policy.json",
            "expected_tolerances.json",
            "corner_cases.md",
            "README.md",
        ],
        "golden_vector_sources": [
            "validation examples",
            "hand-constructed topology cases",
        ],
        "forbidden_golden_vector_sources": [
            "P2 test tensors",
            "P2 B6 cache",
        ],
        "quantization_study_order": [
            "FP32 reference",
            "INT16 activations with INT8 weights",
            "INT8 activations with INT8 weights",
            "freeze selected fixed-point contract",
        ],
        "quantization_acceptance": {
            "graph_decision_agreement_min": 0.999,
            "source_node_decision_agreement_min": 0.995,
            "source_ranking_agreement_min": 0.995,
            "validation_source_f1_loss_max_absolute": 0.005,
            "accumulator_overflow_allowed": False,
        },
        "float_reference_acceptance": {
            "max_absolute_output_difference": 1.0e-6,
            "source_ranking_agreement": 1.0,
            "thresholded_source_decision_agreement": 1.0,
            "derived_graph_decision_agreement": 1.0,
        },
    }
    rtl_requirements = {
        **rtl_requirements_core,
        "requirements_sha256": canonical_sha256(
            rtl_requirements_core
        ),
    }

    test_access_policy_core = {
        "name": "V5_P2_G0_P2_TEST_ACCESS_POLICY",
        "version": 1,
        "g0": {
            "b4_validation_cache_accessed": False,
            "b6_test_cache_accessed": False,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
            "training_performed": False,
            "inference_performed": False,
        },
        "g1_to_g6": {
            "allowed_data": ["P2 train", "P2 validation"],
            "p2_test_model_inference_allowed": False,
            "p2_test_directory_enumeration_allowed": False,
            "p2_test_tensor_access_allowed": False,
            "b6_cache_model_comparison_allowed": False,
            "reason": (
                "B6 contains only frozen B3 outputs and cannot evaluate "
                "newly trained graph models"
            ),
        },
        "selection_boundary": {
            "architecture_selection": "validation_only",
            "checkpoint_selection": "validation_only",
            "threshold_selection": "validation_only",
            "quantization_calibration": "validation_only",
            "rtl_golden_vectors": "validation_and_hand_constructed_only",
        },
        "first_clean_blind_graph_comparison": "V6",
    }
    test_access_policy = {
        **test_access_policy_core,
        "policy_sha256": canonical_sha256(test_access_policy_core),
    }

    stage_plan_core = {
        "name": "V5_P2_G0_STAGE_PLAN",
        "version": 1,
        "stages": [
            {
                "stage": "V5_P2_G1_SOURCE_ONLY_GRAPH_BASELINES",
                "action": (
                    "train Conv1D, GCNConv, GraphConv, and screening GAT "
                    "for source-only localization using three seeds"
                ),
                "data": "P2 train/validation only",
            },
            {
                "stage": "V5_P2_G2_GRAPH_DETECTION_BASELINES",
                "action": (
                    "train direct graph-detection variants using identical "
                    "readout and three seeds"
                ),
                "data": "P2 train/validation only",
            },
            {
                "stage": "V5_P2_G3_ROLE_AWARE_GRAPH_BASELINES",
                "action": (
                    "promote justified operators into role-aware and one "
                    "full-multitask comparison"
                ),
                "data": "P2 train/validation only",
            },
            {
                "stage": (
                    "V5_P2_G4_GRAPH_OPERATOR_STABILITY_AND_SELECTION"
                ),
                "action": (
                    "run finalists across five seeds and freeze scientific "
                    "conclusion plus RTL candidate recommendation"
                ),
                "data": "P2 validation only for selection",
            },
            {
                "stage": (
                    "V5_P2_G5_RTL_GRAPH_BASELINE_SELECTION_CONTRACT"
                ),
                "action": (
                    "freeze selected validation-only architecture, seed, "
                    "checkpoint, operator equation, interfaces, and "
                    "quantization policy"
                ),
                "data": "no P2 test access",
            },
            {
                "stage": (
                    "V5_P2_G6_RTL_GRAPH_BASELINE_HANDOFF_FREEZE"
                ),
                "action": (
                    "build and freeze complete software/integer/golden-vector "
                    "handoff package for RTL team"
                ),
                "data": "validation and hand-constructed vectors only",
            },
            {
                "stage": "V5_P2_L0_LEGAL_XY_DECODER_CONTRACT",
                "action": (
                    "resume Legal NoC decoder branch from first principles"
                ),
                "data": "none",
            },
        ],
        "append_only": True,
        "c2_modified": False,
        "c2a_modified": False,
    }
    stage_plan = {
        **stage_plan_core,
        "plan_sha256": canonical_sha256(stage_plan_core),
    }

    protocol_core = {
        "stage": STAGE,
        "version": 1,
        "status": "COMPLETE",
        "decision": (
            "FREEZE_VALIDATION_ONLY_GRAPH_BASELINE_BRANCH_BEFORE_"
            "LEGAL_NOC_DECODER_IMPLEMENTATION"
        ),
        "scientific_questions": [
            "Can a properly designed GraphConv reproduce strong binary "
            "source localization?",
            "Does graph aggregation improve direct graph detection?",
            "Does graph aggregation degrade source/transit/victim/path "
            "role separation?",
            "Was prior localization weakness caused by vanilla GCN, task "
            "formulation, data/labels, or graph learning generally?",
        ],
        "relationship_to_existing_stages": {
            "c2_preserved": True,
            "c2a_preserved": True,
            "legal_decoder_l0_temporarily_deferred": True,
            "legal_decoder_cancelled": False,
            "return_to_l0_required": True,
        },
        "artifact_canonical_sha256": {
            "task_and_model_matrix": task_matrix["matrix_sha256"],
            "graph_operator_contracts": (
                operator_contracts["contracts_sha256"]
            ),
            "training_selection_and_seed_policy": (
                training_policy["policy_sha256"]
            ),
            "stop_go_criteria": stop_go["criteria_sha256"],
            "rtl_handoff_requirements": (
                rtl_requirements["requirements_sha256"]
            ),
            "p2_test_access_policy": (
                test_access_policy["policy_sha256"]
            ),
            "stage_plan": stage_plan["plan_sha256"],
        },
        "provenance": {
            "c2a_report_sha256": sha256_file(c2a_report_path),
            "c2a_lock_sha256": sha256_file(c2a_lock_path),
            "c2a_evaluation_policy_sha256": (
                sha256_file(c2a_eval_path)
            ),
            "c2a_stage_plan_sha256": sha256_file(c2a_plan_path),
            "c2a_amendment_sha256": EXPECTED_C2A_AMENDMENT_SHA,
            "c2_contract_sha256": EXPECTED_C2_CONTRACT_SHA,
        },
        "security_boundary": {
            "training_performed": False,
            "checkpoint_created": False,
            "architecture_selected": False,
            "quantization_performed": False,
            "rtl_generated": False,
            "b4_validation_cache_accessed": False,
            "b6_test_cache_accessed": False,
            "p2_test_directory_enumerated": False,
            "p2_test_tensor_contents_accessed": False,
            "p2_test_inference_rerun": False,
            "legal_decoder_implemented": False,
        },
        "failures": failures,
        "warnings": warnings,
        "next_stage": "V5_P2_G1_SOURCE_ONLY_GRAPH_BASELINES",
    }
    protocol = {
        **protocol_core,
        "protocol_sha256": canonical_sha256(protocol_core),
    }

    outputs = {
        "protocol": (
            output_dir
            / "V5_P2_G0_GRAPH_BASELINE_AND_RTL_HANDOFF_PROTOCOL.json"
        ),
        "protocol_markdown": (
            output_dir
            / "V5_P2_G0_GRAPH_BASELINE_AND_RTL_HANDOFF_PROTOCOL.md"
        ),
        "task_matrix": (
            output_dir
            / "V5_P2_G0_TASK_AND_MODEL_MATRIX.json"
        ),
        "operator_contracts": (
            output_dir
            / "V5_P2_G0_GRAPH_OPERATOR_CONTRACTS.json"
        ),
        "training_policy": (
            output_dir
            / "V5_P2_G0_TRAINING_SELECTION_AND_SEED_POLICY.json"
        ),
        "stop_go": (
            output_dir
            / "V5_P2_G0_STOP_GO_CRITERIA.json"
        ),
        "rtl_requirements": (
            output_dir
            / "V5_P2_G0_RTL_HANDOFF_REQUIREMENTS.json"
        ),
        "test_access": (
            output_dir
            / "V5_P2_G0_P2_TEST_ACCESS_POLICY.json"
        ),
        "stage_plan": (
            output_dir
            / "V5_P2_G0_STAGE_PLAN.json"
        ),
    }

    write_json(outputs["task_matrix"], task_matrix)
    write_json(outputs["operator_contracts"], operator_contracts)
    write_json(outputs["training_policy"], training_policy)
    write_json(outputs["stop_go"], stop_go)
    write_json(outputs["rtl_requirements"], rtl_requirements)
    write_json(outputs["test_access"], test_access_policy)
    write_json(outputs["stage_plan"], stage_plan)
    write_json(outputs["protocol"], protocol)

    markdown = [
        "# V5 P2-G0 Graph-Baseline and RTL-Handoff Protocol",
        "",
        "## Frozen decision",
        "",
        "Pause Legal NoC decoder implementation after C2A and run a "
        "time-boxed, validation-only graph-baseline branch first.",
        "",
        "## Scientific questions",
        "",
        "1. Can GraphConv reproduce strong binary source localization?",
        "2. Does graph aggregation improve direct graph detection?",
        "3. Does graph aggregation blur source/transit/victim/path roles?",
        "4. Was the earlier localization problem caused by vanilla GCN, "
        "task formulation, data/labels, or graph learning generally?",
        "",
        "## Required task order",
        "",
        "`G1 source-only -> G2 graph detection -> G3 role-aware/full "
        "multitask -> G4 five-seed selection -> G5 RTL selection contract -> "
        "G6 RTL handoff -> return to L0 Legal XY decoder contract`",
        "",
        "## Mandatory operators",
        "",
        "- Conv1D-only reference.",
        "- Conv1D + GCNConv.",
        "- Conv1D + GraphConv.",
        "- GAT is screening-only and requires a material validation or "
        "stability advantage for promotion.",
        "",
        "## Fairness",
        "",
        "All variants use the same P2 train/validation pair manifests, "
        "PRIMARY58 interface, physical-port mask, temporal encoder, graph "
        "width, layer count, seeds, optimizer, scheduler, early stopping, "
        "checkpoint-selection metric, and threshold-selection policy for a "
        "given task.",
        "",
        "## Test boundary",
        "",
        "- No P2 test inference.",
        "- No P2 test tensor access.",
        "- No B6-cache architecture comparison.",
        "- First clean blind graph-model comparison occurs on V6.",
        "",
        "## RTL direction",
        "",
        "The preferred candidate is Conv1D + GraphConv source-only, but it is "
        "not selected yet. It must pass validation, stability, and hardware-"
        "cost gates. Existing B3 Conv1D-count4 is the fallback temporal "
        "reference.",
        "",
        "## Immediate next stage",
        "",
        "`V5_P2_G1_SOURCE_ONLY_GRAPH_BASELINES`",
        "",
    ]
    atomic_write(
        outputs["protocol_markdown"],
        "\n".join(markdown),
    )

    lock = {
        "status": COMPLETE,
        "decision": (
            "FREEZE_VALIDATION_ONLY_GRAPH_BASELINE_BRANCH_BEFORE_"
            "LEGAL_NOC_DECODER_IMPLEMENTATION"
        ),
        "protocol_file_sha256": sha256_file(outputs["protocol"]),
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_markdown_sha256": sha256_file(
            outputs["protocol_markdown"]
        ),
        "artifact_file_sha256": {
            key: sha256_file(path)
            for key, path in outputs.items()
            if key not in {"protocol", "protocol_markdown"}
        },
        "artifact_canonical_sha256": (
            protocol["artifact_canonical_sha256"]
        ),
        "c2a_report_sha256": sha256_file(c2a_report_path),
        "c2a_lock_sha256": sha256_file(c2a_lock_path),
        "c2a_amendment_sha256": EXPECTED_C2A_AMENDMENT_SHA,
        "c2_contract_sha256": EXPECTED_C2_CONTRACT_SHA,
        "training_performed": False,
        "inference_performed": False,
        "architecture_selected": False,
        "checkpoint_created": False,
        "quantization_performed": False,
        "rtl_generated": False,
        "b4_validation_cache_accessed": False,
        "b6_test_cache_accessed": False,
        "p2_test_directory_enumerated": False,
        "p2_test_tensor_contents_accessed": False,
        "p2_test_inference_rerun": False,
        "legal_decoder_implemented": False,
        "next_stage": "V5_P2_G1_SOURCE_ONLY_GRAPH_BASELINES",
        "script_sha256": sha256_file(Path(__file__)),
    }
    lock_path = (
        output_dir
        / "V5_P2_G0_GRAPH_BASELINE_AND_RTL_HANDOFF_PROTOCOL_LOCK.json"
    )
    write_json(lock_path, lock)
    atomic_write(
        output_dir / COMPLETE,
        COMPLETE + "\n",
    )

    print("===== V5 P2-G0 GRAPH BASELINE + RTL HANDOFF PROTOCOL =====")
    print("status: COMPLETE")
    print(
        "decision: "
        "FREEZE_VALIDATION_ONLY_GRAPH_BASELINE_BRANCH_BEFORE_"
        "LEGAL_NOC_DECODER_IMPLEMENTATION"
    )
    print("c2_preserved: true")
    print("c2a_preserved: true")
    print("legal_decoder_cancelled: false")
    print("legal_decoder_l0_temporarily_deferred: true")
    print("mandatory_models: CONV1D, GCNCONV, GRAPHCONV")
    print("gat_status: SCREENING_ONLY")
    print("initial_seeds: 107,117,127")
    print("finalist_seeds: 107,117,127,137,147")
    print("p2_train_allowed: true")
    print("p2_validation_allowed: true")
    print("p2_test_allowed: false")
    print("preferred_rtl_candidate_preselected: false")
    print(
        "preferred_rtl_candidate_policy: "
        "CONV1D_PLUS_GRAPHCONV_SOURCE_ONLY_IF_GATES_PASS"
    )
    print("fallback_rtl_reference: EXISTING_B3_CONV1D_COUNT4")
    print("training_performed: false")
    print("inference_performed: false")
    print("architecture_selected: false")
    print("checkpoint_created: false")
    print("b4_validation_cache_accessed: false")
    print("b6_test_cache_accessed: false")
    print("p2_test_directory_enumerated: false")
    print("p2_test_tensor_contents_accessed: false")
    print("p2_test_inference_rerun: false")
    print("legal_decoder_implemented: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("protocol_sha256:", protocol["protocol_sha256"])
    print("next_stage: V5_P2_G1_SOURCE_ONLY_GRAPH_BASELINES")
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
