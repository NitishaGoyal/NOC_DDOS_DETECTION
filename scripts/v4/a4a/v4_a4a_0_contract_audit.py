#!/usr/bin/env python3
"""
V4-A4a-0
Structured Null-Aware Set Decoder Contract and Implementation Audit

Purpose
-------
Freeze the scientific and implementation contract before any A4a training.

This stage:
  * verifies A3.12, A3.13, and A3.15 provenance
  * verifies the exact frozen A3 checkpoint and source
  * locates the exact original A3 trainer by its checkpoint-recorded hash
  * audits V4 labels, split inventory, and required array shapes
  * freezes the decoder-only A4a-D0 architecture
  * computes exact parameter counts
  * freezes training/evaluation/safety rules
  * explicitly prohibits development-test selection

No model training, prediction generation, threshold search, or test evaluation
occurs in this script.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPECTED_MODEL_SOURCE_SHA = (
    "5ebcf80385b95f329faf935cad8039b64"
    "afd79f464903308eb07e674d78a1704"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value and all(
                isinstance(v, torch.Tensor) for v in value.values()
            ):
                return value
        if checkpoint and all(
            isinstance(v, torch.Tensor) for v in checkpoint.values()
        ):
            return checkpoint
    raise RuntimeError("state_dict not found")


def linear_parameters(in_features: int, out_features: int) -> int:
    return in_features * out_features + out_features


def locate_hash_matches(repo: Path, wanted_sha: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for path in sorted(repo.rglob("*.py")):
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        try:
            if path.stat().st_size > 2_000_000:
                continue
            current_sha = sha256_file(path)
        except OSError:
            continue
        if current_sha == wanted_sha:
            matches.append(
                {
                    "path": str(path),
                    "sha256": current_sha,
                    "size_bytes": path.stat().st_size,
                }
            )
    return matches


def source_class_inventory(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    rows: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = [
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assignments: list[str] = []
        for child in node.body:
            for subnode in ast.walk(child):
                if isinstance(subnode, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        subnode.targets
                        if isinstance(subnode, ast.Assign)
                        else [subnode.target]
                    )
                    for target in targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            assignments.append(target.attr)
        rows.append(
            {
                "class_name": node.name,
                "line_start": node.lineno,
                "line_end": node.end_lineno,
                "methods": ",".join(sorted(set(methods))),
                "self_modules_or_fields": ",".join(sorted(set(assignments))),
            }
        )
    return rows


def split_name_mapping(metadata: dict[str, Any]) -> dict[int, str]:
    mapping = metadata.get("code_maps", {}).get("split", {})
    result: dict[int, str] = {}
    if isinstance(mapping, dict):
        for name, value in mapping.items():
            try:
                result[int(value)] = str(name)
            except (TypeError, ValueError):
                continue
    return result


def array_inventory(data_dir: Path) -> list[dict[str, Any]]:
    entries = [
        ("x.npy", True),
        ("y_graph.npy", True),
        ("y_node.npy", True),
        ("run_index.npy", True),
        ("split.npy", True),
        ("edge_index.npy", True),
        ("attacker_count.npy", False),
        ("profile_id.npy", False),
        ("profile.npy", False),
        ("attack_kind_id.npy", False),
        ("attack_kind.npy", False),
        ("active_cores.npy", False),
        ("attackers.npy", False),
    ]
    rows: list[dict[str, Any]] = []
    for name, required in entries:
        path = data_dir / name
        if not path.exists():
            rows.append(
                {
                    "name": name,
                    "required": required,
                    "exists": False,
                    "path": str(path),
                }
            )
            continue
        array = np.load(path, mmap_mode="r", allow_pickle=True)
        rows.append(
            {
                "name": name,
                "required": required,
                "exists": True,
                "path": str(path),
                "shape": "x".join(str(value) for value in array.shape),
                "dtype": str(array.dtype),
                "size_bytes": path.stat().st_size,
                "sha256": (
                    sha256_file(path)
                    if path.stat().st_size <= 100_000_000
                    else "OMITTED_LARGE_ARRAY"
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--a3-12-summary", type=Path, required=True)
    parser.add_argument("--a3-12-lock", type=Path, required=True)
    parser.add_argument("--a3-13-summary", type=Path, required=True)
    parser.add_argument("--a3-13-lock", type=Path, required=True)
    parser.add_argument("--a3-15-summary", type=Path, required=True)
    parser.add_argument("--a3-15-lock", type=Path, required=True)
    parser.add_argument("--operational-policy", type=Path, required=True)
    parser.add_argument("--operational-policy-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    required = [
        args.repo,
        args.data_dir,
        args.metadata,
        args.checkpoint,
        args.model_source,
        args.a3_12_summary,
        args.a3_12_lock,
        args.a3_13_summary,
        args.a3_13_lock,
        args.a3_15_summary,
        args.a3_15_lock,
        args.operational_policy,
        args.operational_policy_lock,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("A4a-0 FAIL: missing required paths", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(
            f"A4a-0 FAIL: output exists: {args.output_dir}",
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True)

    metadata = load_json(args.metadata)
    a3_12_summary = load_json(args.a3_12_summary)
    a3_12_lock = load_json(args.a3_12_lock)
    a3_13_summary = load_json(args.a3_13_summary)
    a3_13_lock = load_json(args.a3_13_lock)
    a3_15_summary = load_json(args.a3_15_summary)
    a3_15_lock = load_json(args.a3_15_lock)
    operational_policy = load_json(args.operational_policy)
    operational_lock = load_json(args.operational_policy_lock)

    checkpoint_sha = sha256_file(args.checkpoint)
    model_source_sha = sha256_file(args.model_source)
    operational_sha = sha256_file(args.operational_policy)

    failures: list[str] = []
    if model_source_sha != EXPECTED_MODEL_SOURCE_SHA:
        failures.append("model source hash differs from resolved A3 source")
    if a3_12_lock.get("checkpoint_sha256") != checkpoint_sha:
        failures.append("A3.12 checkpoint mismatch")
    if a3_13_lock.get("checkpoint_sha256") != checkpoint_sha:
        failures.append("A3.13 checkpoint mismatch")
    if a3_13_lock.get("model_source_sha256") != model_source_sha:
        failures.append("A3.13 model source mismatch")
    if a3_15_lock.get("development_test_used_for_selection") is not False:
        failures.append("A3.15 reports development-test selection")
    if a3_12_summary.get("development_test_used_for_selection") is not False:
        failures.append("A3.12 reports development-test selection")
    if a3_13_summary.get("development_test_used_for_selection") is not False:
        failures.append("A3.13 reports development-test selection")
    if operational_lock.get("checkpoint_sha256") != checkpoint_sha:
        failures.append("operational-policy checkpoint mismatch")
    if operational_lock.get("operational_policy_family_sha256") != operational_sha:
        failures.append("operational-policy hash mismatch")

    if failures:
        payload = {"status": "FAIL", "failures": failures}
        (args.output_dir / "a4a_0_failure.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = extract_state_dict(checkpoint)
    recorded_hashes = checkpoint.get("source_hashes", {})
    recorded_trainer_sha = recorded_hashes.get("trainer")
    recorded_common_sha = recorded_hashes.get("common")

    if recorded_common_sha != model_source_sha:
        raise RuntimeError(
            "checkpoint-recorded common source hash does not match model source"
        )
    if not recorded_trainer_sha:
        raise RuntimeError("checkpoint does not record original trainer hash")

    trainer_matches = locate_hash_matches(args.repo, recorded_trainer_sha)
    write_csv(
        args.output_dir / "original_trainer_hash_matches.csv",
        trainer_matches,
    )
    if len(trainer_matches) != 1:
        failures.append(
            f"expected one exact trainer source match, found {len(trainer_matches)}"
        )

    array_rows = array_inventory(args.data_dir)
    write_csv(args.output_dir / "dataset_array_inventory.csv", array_rows)

    missing_arrays = [
        row["name"]
        for row in array_rows
        if row.get("required", False) and not row.get("exists", False)
    ]
    if missing_arrays:
        failures.append(
            "missing required dataset arrays: " + ", ".join(missing_arrays)
        )

    split_array = np.load(args.data_dir / "split.npy", mmap_mode="r")
    run_index = np.load(args.data_dir / "run_index.npy", mmap_mode="r")
    y_graph = np.load(args.data_dir / "y_graph.npy", mmap_mode="r")
    y_node = np.load(args.data_dir / "y_node.npy", mmap_mode="r")
    attacker_count_path = args.data_dir / "attacker_count.npy"
    if attacker_count_path.exists():
        attacker_count = np.load(
            attacker_count_path,
            mmap_mode="r",
        )
        attacker_count_source = "attacker_count.npy"
    else:
        attacker_count = np.sum(
            np.asarray(y_node, dtype=np.int8),
            axis=1,
            dtype=np.int16,
        )
        attacker_count_source = "derived_from_y_node_sum"

    split_map = split_name_mapping(metadata)
    split_rows: list[dict[str, Any]] = []
    for split_id in sorted(int(value) for value in np.unique(split_array)):
        sample_mask = np.asarray(split_array) == split_id
        split_runs = np.unique(np.asarray(run_index)[sample_mask])
        counts, frequencies = np.unique(
            np.asarray(attacker_count)[sample_mask],
            return_counts=True,
        )
        row: dict[str, Any] = {
            "split_id": split_id,
            "split_name": split_map.get(split_id, str(split_id)),
            "sample_count": int(np.sum(sample_mask)),
            "run_count": int(split_runs.size),
            "normal_sample_count": int(
                np.sum(np.asarray(y_graph)[sample_mask] == 0)
            ),
            "attack_sample_count": int(
                np.sum(np.asarray(y_graph)[sample_mask] == 1)
            ),
        }
        for count, frequency in zip(counts, frequencies):
            row[f"attacker_count_{int(count)}_samples"] = int(frequency)
        split_rows.append(row)
    write_csv(args.output_dir / "split_and_count_inventory.csv", split_rows)

    if tuple(y_node.shape[1:]) != (16,):
        failures.append(f"unexpected y_node shape {tuple(y_node.shape)}")
    observed_counts = sorted(
        int(value) for value in np.unique(attacker_count)
    )
    if observed_counts != [0, 1, 2, 3, 4]:
        failures.append(
            f"expected attacker counts [0,1,2,3,4], found {observed_counts}"
        )

    source_rows = source_class_inventory(args.model_source)
    write_csv(args.output_dir / "frozen_model_source_inventory.csv", source_rows)

    model_signature = checkpoint.get("model_signature", {})
    expected_parameter_count = int(
        model_signature.get("expected_parameter_count", -1)
    )
    actual_parameter_count = int(
        sum(tensor.numel() for tensor in state_dict.values())
    )
    if expected_parameter_count != actual_parameter_count:
        failures.append(
            "checkpoint parameter count does not match model signature"
        )

    # A4a-D0 exact decoder parameter contract.
    dims = {
        "routers": 16,
        "h_local": 16,
        "h_graph": 16,
        "h_node": 32,
        "regional_embedding": 64,
        "router_hidden": 16,
        "count_classes": 5,
    }

    router_membership_head = (
        linear_parameters(dims["h_node"], dims["router_hidden"])
        + linear_parameters(dims["router_hidden"], 1)
    )
    cardinality_head = linear_parameters(
        dims["regional_embedding"],
        dims["count_classes"],
    )
    null_head = linear_parameters(dims["regional_embedding"], 1)
    new_decoder_parameters = (
        router_membership_head + cardinality_head + null_head
    )

    frozen_representation_parameters = sum(
        int(tensor.numel())
        for key, tensor in state_dict.items()
        if key.startswith(
            (
                "temporal_conv.",
                "local_projection.",
                "gcn.",
            )
        )
    )
    frozen_graph_head_parameters = sum(
        int(tensor.numel())
        for key, tensor in state_dict.items()
        if key.startswith("graph_head.")
    )
    deployed_parameter_count = (
        frozen_representation_parameters
        + frozen_graph_head_parameters
        + new_decoder_parameters
    )

    parameter_rows = [
        {
            "component": "frozen_temporal_local_gcn_encoder",
            "trainable": False,
            "parameter_count": frozen_representation_parameters,
        },
        {
            "component": "frozen_graph_head",
            "trainable": False,
            "parameter_count": frozen_graph_head_parameters,
        },
        {
            "component": "new_router_membership_head_32_16_1",
            "trainable": True,
            "parameter_count": router_membership_head,
        },
        {
            "component": "new_cardinality_head_64_5",
            "trainable": True,
            "parameter_count": cardinality_head,
        },
        {
            "component": "new_null_head_64_1",
            "trainable": True,
            "parameter_count": null_head,
        },
        {
            "component": "a4a_d0_total_deployed",
            "trainable": "",
            "parameter_count": deployed_parameter_count,
        },
    ]
    write_csv(args.output_dir / "a4a_d0_parameter_budget.csv", parameter_rows)

    if new_decoder_parameters > 1200:
        failures.append(
            f"A4a-D0 decoder exceeds 1200 parameters: {new_decoder_parameters}"
        )
    if deployed_parameter_count > 3500:
        failures.append(
            f"A4a-D0 deployed model exceeds 3500 parameters: {deployed_parameter_count}"
        )

    candidate_policy = operational_policy["policy_roles"][
        "candidate_localization"
    ]["policy"]

    best_proxy = a3_15_summary.get("best_validation_directional_proxy")
    a3_12_validation = a3_12_summary["split_summaries"]["validation"]

    contract = {
        "status": "PASS" if not failures else "FAIL",
        "designation": (
            "V4-A4a-0 Structured Null-Aware Set Decoder Contract"
        ),
        "experiment_name": "v4_a4a_d0_decoder_only_seed7",
        "scientific_hypothesis": (
            "A structured null-aware and count-consistent decoder can recover "
            "part of the validation oracle-count localization gap without "
            "changing the strong frozen A3 local-plus-graph representation."
        ),
        "frozen_base": {
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "model_source_path": str(args.model_source),
            "model_source_sha256": model_source_sha,
            "original_parameter_count": actual_parameter_count,
            "original_trainer_sha256": recorded_trainer_sha,
            "original_trainer_matches": trainer_matches,
            "frozen_modules": [
                "temporal_conv",
                "local_projection",
                "gcn",
                "graph_head",
            ],
            "discarded_or_replaced_modules": [
                "attacker_hidden",
                "attacker_out",
                "count_head",
            ],
        },
        "a4a_d0_architecture": {
            "router_membership_head": [32, 16, 1],
            "cardinality_head": [64, 5],
            "null_head": [64, 1],
            "router_membership_input": "h_node=concat(h_local,h_graph)",
            "cardinality_input": "regional_embedding",
            "null_input": "regional_embedding",
            "new_trainable_parameter_count": new_decoder_parameters,
            "total_deployed_parameter_count": deployed_parameter_count,
            "parameter_budget_new_decoder_max": 1200,
            "parameter_budget_total_deployed_max": 3500,
            "directional_proxy_in_primary_model": False,
            "directional_proxy_policy": (
                "The A3.15 proxy may only be tested later as an auxiliary "
                "validation-selected feature. It is prohibited as a hard gate."
            ),
        },
        "training_contract": {
            "trainable_modules": [
                "router_membership_head",
                "cardinality_head",
                "null_head",
            ],
            "encoder_frozen": True,
            "graph_head_frozen": True,
            "training_split_only": True,
            "development_test_access_during_training": False,
            "seed": 7,
            "epochs_max": 75,
            "early_stopping_patience": 10,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size_primary": 512,
            "batch_size_fallback": 256,
            "losses": {
                "router_membership_bce": 1.0,
                "cardinality_cross_entropy": 1.0,
                "null_state_bce": 0.5,
                "router_rank_margin": 0.2,
                "count_router_consistency_smooth_l1": 0.25,
            },
            "consistency_definition": (
                "SmoothL1 between expected cardinality from the five-class "
                "softmax and the sum of sigmoid router-membership probabilities."
            ),
            "null_definition": "null target is attacker_count == 0",
        },
        "evaluation_contract": {
            "temporal_mode": candidate_policy["mode"],
            "temporal_horizon": candidate_policy["horizon"],
            "deoverlap_stride_epochs": 8,
            "frozen_outer_graph_threshold": candidate_policy[
                "graph_threshold"
            ],
            "graph_gate_source": "frozen A3 graph head",
            "node_and_null_policy_selection": "validation only",
            "development_test_transfer_count": 1,
            "development_test_used_for_selection": False,
            "primary_decoder": (
                "predict k in {0,1,2,3,4}; if null or k=0 output empty set; "
                "otherwise output top-k routers by membership score"
            ),
        },
        "validation_safety_and_quality_gates": {
            "attack_set_precision_min": 0.98,
            "attack_exact_localization_min": 0.83,
            "stable_exact_localization_min": 0.80,
            "attack_run_coverage_min": 0.90,
            "normal_window_false_isolation_rate_max": 0.00025,
            "normal_persistent_run_false_isolation_fraction_max": 0.0,
            "automatic_isolation_claim_allowed_only_if_all_gates_pass": True,
        },
        "diagnostic_baselines": {
            "a3_h32_validation_threshold_exact": a3_12_validation[
                "temporal_primary_attack_exact"
            ],
            "a3_h32_validation_oracle_count_exact": a3_12_validation[
                "temporal_oracle_attack_exact"
            ],
            "a3_h32_validation_oracle_gap": a3_12_validation[
                "temporal_oracle_minus_primary_exact_gap"
            ],
            "a3_13_decision": a3_13_summary[
                "validation_based_decision_class"
            ],
            "a3_15_decision": a3_15_summary[
                "validation_based_decision_class"
            ],
            "a3_15_best_proxy": best_proxy,
        },
        "explicit_prohibitions": [
            "No threshold, feature, persistence, or model selection on development test.",
            "No removal or weakening of graph context based on graph-only ablation.",
            "No hard source/transit rule from the moderate A3.15 proxy.",
            "No end-to-end encoder fine-tuning in A4a-D0.",
            "No automatic-isolation claim unless every validation and frozen-transfer safety gate passes.",
            "No publication claim using the current development test as an independent holdout.",
        ],
        "v5_requirements_preserved": [
            "explicit legitimate-source and transit-router labels",
            "explicit valid, idle, and clipped port semantics",
            "local injection and ejection quantities",
            "victim labels",
            "matched benign/attack counterfactual runs",
            "independent publication holdout",
        ],
        "provenance": {
            "metadata_sha256": sha256_file(args.metadata),
            "a3_12_summary_sha256": sha256_file(args.a3_12_summary),
            "a3_12_lock_sha256": sha256_file(args.a3_12_lock),
            "a3_13_summary_sha256": sha256_file(args.a3_13_summary),
            "a3_13_lock_sha256": sha256_file(args.a3_13_lock),
            "a3_15_summary_sha256": sha256_file(args.a3_15_summary),
            "a3_15_lock_sha256": sha256_file(args.a3_15_lock),
            "operational_policy_sha256": operational_sha,
        },
        "training_performed": False,
        "prediction_generation_performed": False,
        "threshold_search_performed": False,
        "development_test_accessed": False,
    }

    contract_path = args.output_dir / "A4A_D0_FROZEN_CONTRACT.json"
    contract_path.write_text(
        json.dumps(jsonable(contract), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    audit = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checkpoint_parameter_count": actual_parameter_count,
        "exact_original_trainer_match_count": len(trainer_matches),
        "observed_attacker_counts": observed_counts,
        "attacker_count_source": attacker_count_source,
        "new_decoder_parameter_count": new_decoder_parameters,
        "total_deployed_parameter_count": deployed_parameter_count,
        "contract_sha256": sha256_file(contract_path),
        "ready_for_a4a_d0_training_implementation": not failures,
    }
    audit_path = args.output_dir / "implementation_audit.json"
    audit_path.write_text(
        json.dumps(jsonable(audit), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lock = {
        "status": (
            "A4A_0_CONTRACT_AUDIT_COMPLETE"
            if not failures
            else "A4A_0_CONTRACT_AUDIT_FAILED"
        ),
        "checkpoint_sha256": checkpoint_sha,
        "model_source_sha256": model_source_sha,
        "contract_sha256": sha256_file(contract_path),
        "implementation_audit_sha256": sha256_file(audit_path),
        "training_performed": False,
        "development_test_accessed": False,
        "threshold_search_performed": False,
    }
    (args.output_dir / "A4A_0_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    marker = (
        "V4_A4A_0_CONTRACT_AUDIT_PASS"
        if not failures
        else "V4_A4A_0_CONTRACT_AUDIT_FAIL"
    )
    (args.output_dir / marker).write_text(
        marker + "\n",
        encoding="utf-8",
    )

    print(json.dumps(jsonable(audit), indent=2, sort_keys=True))
    print(marker)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
