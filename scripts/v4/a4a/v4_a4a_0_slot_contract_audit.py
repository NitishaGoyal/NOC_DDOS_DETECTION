#!/usr/bin/env python3
"""
V4-A4a-0S
Null-Aware Slot-Set Decoder Contract and Implementation Audit

This stage replaces the earlier count-plus-top-k A4a contract with a genuinely
set-structured primary decoder:

  * four unordered attacker slots
  * each slot predicts router 0..15 or NULL
  * permutation-invariant training
  * unique constrained set decoding
  * membership and cardinality derived from slot probabilities
  * no hard directional proxy gate

The script freezes provenance, parameter budgets, ablations, training losses,
validation gates, kill criteria, promotion criteria, and paper-claim limits.

No model training, prediction generation, threshold search, validation
selection, or development-test evaluation occurs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPECTED_MODEL_SOURCE_SHA = (
    "5ebcf80385b95f329faf935cad8039b64"
    "afd79f464903308eb07e674d78a1704"
)

REQUIRED_ARRAYS = (
    "x.npy",
    "y_graph.npy",
    "y_node.npy",
    "run_index.npy",
    "split_id.npy",
    "attacker_count.npy",
    "edge_index.npy",
)

OPTIONAL_ARRAYS = (
    "end_epoch.npy",
    "profile_id.npy",
    "attack_kind_id.npy",
    "strength.npy",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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
            candidate = checkpoint.get(key)
            if (
                isinstance(candidate, dict)
                and candidate
                and all(isinstance(v, torch.Tensor) for v in candidate.values())
            ):
                return candidate

        if checkpoint and all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            return checkpoint

    raise RuntimeError("state_dict not found in checkpoint")


def linear_parameters(in_features: int, out_features: int) -> int:
    return in_features * out_features + out_features


def locate_hash_matches(repo: Path, wanted_sha: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []

    for path in sorted(repo.rglob("*.py")):
        if any(
            part in {".git", ".venv", "__pycache__"}
            for part in path.parts
        ):
            continue

        try:
            if path.stat().st_size > 2_000_000:
                continue
            current_sha = sha256_file(path)
        except OSError:
            continue

        if current_sha != wanted_sha:
            continue

        path_text = str(path)
        canonical_score = 0
        if "/scripts/v4/train/" in path_text:
            canonical_score += 100
        if ".buggy_" in path_text or ".backup" in path_text:
            canonical_score -= 50

        matches.append(
            {
                "path": path_text,
                "sha256": current_sha,
                "size_bytes": path.stat().st_size,
                "canonical_score": canonical_score,
            }
        )

    matches.sort(
        key=lambda row: (
            -int(row["canonical_score"]),
            len(str(row["path"])),
            str(row["path"]),
        )
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
                if not isinstance(subnode, (ast.Assign, ast.AnnAssign)):
                    continue

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
                "self_modules_or_fields": ",".join(
                    sorted(set(assignments))
                ),
            }
        )

    return rows


def split_name_mapping(metadata: dict[str, Any]) -> dict[int, str]:
    mapping = metadata.get("code_maps", {}).get("split", {})
    result: dict[int, str] = {}

    if not isinstance(mapping, dict):
        return result

    for name, value in mapping.items():
        try:
            result[int(value)] = str(name)
        except (TypeError, ValueError):
            continue

    return result


def array_inventory(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for name in (*REQUIRED_ARRAYS, *OPTIONAL_ARRAYS):
        required = name in REQUIRED_ARRAYS
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


def get_validation_a3_12(summary: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        summary.get("split_summaries", {}).get("validation"),
        summary.get("splits", {}).get("validation"),
        summary.get("validation"),
    ]

    for candidate in candidates:
        if isinstance(candidate, dict):
            required = {
                "temporal_primary_attack_exact",
                "temporal_oracle_attack_exact",
                "temporal_oracle_minus_primary_exact_gap",
            }
            if required.issubset(candidate):
                return candidate

    raise KeyError(
        "A3.12 validation summary with temporal exact metrics not found"
    )


def parameter_count_by_prefix(
    state_dict: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> int:
    return sum(
        int(tensor.numel())
        for key, tensor in state_dict.items()
        if key.startswith(prefixes)
    )


def render_markdown(contract: dict[str, Any]) -> str:
    architecture = contract["a4a_slot_primary_architecture"]
    training = contract["training_contract"]
    validation = contract["validation_selection_contract"]
    kill = contract["kill_criteria"]
    promotion = contract["frozen_transfer_promotion_criteria"]

    lines = [
        "# V4-A4a-0S Frozen Contract",
        "",
        "## Primary decoder",
        "",
        "Four unordered attacker slots are used. Each slot predicts one of "
        "17 classes: router 0–15 or NULL.",
        "",
        f"- Slot count: {architecture['slot_count']}",
        f"- Classes per slot: {architecture['classes_per_slot']}",
        f"- Slot latent dimension: {architecture['slot_latent_dim']}",
        f"- New trainable parameters: "
        f"{architecture['new_trainable_parameter_count']}",
        f"- Total deployed parameters: "
        f"{architecture['total_deployed_parameter_count']}",
        "",
        "The primary predicted set comes from a unique constrained assignment. "
        "NULL may be reused; a real router may be assigned to at most one slot.",
        "",
        "## Training",
        "",
        f"- Encoder frozen: {training['encoder_frozen']}",
        f"- Graph head frozen: {training['graph_head_frozen']}",
        f"- Maximum epochs: {training['epochs_max']}",
        f"- Early-stopping patience: "
        f"{training['early_stopping_patience']}",
        f"- Seed: {training['seed']}",
        "",
        "## Validation selection",
        "",
    ]

    for key, value in validation.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Kill criteria", ""])
    for item in kill:
        lines.append(f"- {item}")

    lines.extend(["", "## Frozen-transfer promotion criteria", ""])
    for key, value in promotion.items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            contract["paper_claim_boundary"],
            "",
        ]
    )

    return "\n".join(lines)


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
    parser.add_argument(
        "--operational-policy-lock",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    required_paths = [
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

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        print("A4a-0S FAIL: missing required paths", file=sys.stderr)
        for path in missing_paths:
            print(f"  {path}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(
            f"A4a-0S FAIL: output exists: {args.output_dir}",
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
    operational_policy_sha = sha256_file(args.operational_policy)

    failures: list[str] = []

    if model_source_sha != EXPECTED_MODEL_SOURCE_SHA:
        failures.append(
            "model source hash differs from resolved frozen A3 source"
        )

    if a3_12_lock.get("checkpoint_sha256") != checkpoint_sha:
        failures.append("A3.12 checkpoint mismatch")

    if a3_13_lock.get("checkpoint_sha256") != checkpoint_sha:
        failures.append("A3.13 checkpoint mismatch")

    if a3_13_lock.get("model_source_sha256") != model_source_sha:
        failures.append("A3.13 model-source mismatch")

    for stage_name, stage_summary in (
        ("A3.12", a3_12_summary),
        ("A3.13", a3_13_summary),
        ("A3.15", a3_15_summary),
    ):
        if stage_summary.get("development_test_used_for_selection") is not False:
            failures.append(
                f"{stage_name} reports development-test selection"
            )

    if a3_13_summary.get("validation_based_decision_class") != (
        "local_graph_fusion_is_complementary"
    ):
        failures.append(
            "A3.13 did not resolve to complementary local/graph fusion"
        )

    if a3_15_summary.get("validation_based_decision_class") not in {
        "moderate_univariate_directional_proxy_only",
        "strong_univariate_directional_proxy_exists",
    }:
        failures.append(
            "A3.15 does not support auxiliary-only directional treatment"
        )

    if operational_lock.get("checkpoint_sha256") != checkpoint_sha:
        failures.append("operational-policy checkpoint mismatch")

    if operational_lock.get(
        "operational_policy_family_sha256"
    ) != operational_policy_sha:
        failures.append("operational-policy hash mismatch")

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
        failures.append(
            "checkpoint-recorded common source hash does not match model source"
        )

    if not recorded_trainer_sha:
        failures.append("checkpoint does not record original trainer hash")
        trainer_matches: list[dict[str, Any]] = []
    else:
        trainer_matches = locate_hash_matches(
            args.repo,
            recorded_trainer_sha,
        )

    if not trainer_matches:
        failures.append("no exact original trainer hash match found")

    write_csv(
        args.output_dir / "original_trainer_hash_matches.csv",
        trainer_matches,
    )

    canonical_trainer = (
        trainer_matches[0]["path"]
        if trainer_matches
        else None
    )

    array_rows = array_inventory(args.data_dir)
    write_csv(
        args.output_dir / "dataset_array_inventory.csv",
        array_rows,
    )

    missing_required_arrays = [
        row["name"]
        for row in array_rows
        if row["required"] and not row["exists"]
    ]

    if missing_required_arrays:
        failures.append(
            "missing required dataset arrays: "
            + ", ".join(missing_required_arrays)
        )

    if not missing_required_arrays:
        split_array = np.load(
            args.data_dir / "split_id.npy",
            mmap_mode="r",
        )
        run_index = np.load(
            args.data_dir / "run_index.npy",
            mmap_mode="r",
        )
        y_graph = np.load(
            args.data_dir / "y_graph.npy",
            mmap_mode="r",
        )
        y_node = np.load(
            args.data_dir / "y_node.npy",
            mmap_mode="r",
        )
        attacker_count = np.load(
            args.data_dir / "attacker_count.npy",
            mmap_mode="r",
        )

        sample_count = int(split_array.shape[0])
        aligned_lengths = {
            "split_id": int(split_array.shape[0]),
            "run_index": int(run_index.shape[0]),
            "y_graph": int(y_graph.shape[0]),
            "y_node": int(y_node.shape[0]),
            "attacker_count": int(attacker_count.shape[0]),
        }

        if len(set(aligned_lengths.values())) != 1:
            failures.append(
                f"dataset array lengths are not aligned: {aligned_lengths}"
            )

        if tuple(y_node.shape[1:]) != (16,):
            failures.append(
                f"unexpected y_node shape: {tuple(y_node.shape)}"
            )

        observed_counts = sorted(
            int(value)
            for value in np.unique(attacker_count)
        )

        if observed_counts != [0, 1, 2, 3, 4]:
            failures.append(
                "expected attacker counts [0,1,2,3,4], "
                f"found {observed_counts}"
            )

        split_map = split_name_mapping(metadata)
        split_rows: list[dict[str, Any]] = []

        split_values = np.asarray(split_array)
        run_values = np.asarray(run_index)
        graph_values = np.asarray(y_graph)
        count_values = np.asarray(attacker_count)

        for split_id in sorted(
            int(value)
            for value in np.unique(split_values)
        ):
            mask = split_values == split_id
            counts, frequencies = np.unique(
                count_values[mask],
                return_counts=True,
            )

            row: dict[str, Any] = {
                "split_id": split_id,
                "split_name": split_map.get(split_id, str(split_id)),
                "sample_count": int(np.sum(mask)),
                "run_count": int(np.unique(run_values[mask]).size),
                "normal_sample_count": int(
                    np.sum(graph_values[mask] == 0)
                ),
                "attack_sample_count": int(
                    np.sum(graph_values[mask] == 1)
                ),
            }

            for count, frequency in zip(counts, frequencies):
                row[f"attacker_count_{int(count)}_samples"] = int(
                    frequency
                )

            split_rows.append(row)

        write_csv(
            args.output_dir / "split_and_count_inventory.csv",
            split_rows,
        )
    else:
        sample_count = 0
        observed_counts = []
        aligned_lengths = {}
        split_rows = []
        write_csv(
            args.output_dir / "split_and_count_inventory.csv",
            [],
        )

    source_rows = source_class_inventory(args.model_source)
    write_csv(
        args.output_dir / "frozen_model_source_inventory.csv",
        source_rows,
    )

    actual_parameter_count = int(
        sum(tensor.numel() for tensor in state_dict.values())
    )
    expected_parameter_count = int(
        checkpoint.get("model_signature", {}).get(
            "expected_parameter_count",
            -1,
        )
    )

    if actual_parameter_count != expected_parameter_count:
        failures.append(
            "checkpoint parameter count differs from model signature"
        )

    frozen_encoder_parameters = parameter_count_by_prefix(
        state_dict,
        (
            "temporal_conv.",
            "local_projection.",
            "gcn.",
        ),
    )
    frozen_graph_head_parameters = parameter_count_by_prefix(
        state_dict,
        ("graph_head.",),
    )
    frozen_deployed_base = (
        frozen_encoder_parameters
        + frozen_graph_head_parameters
    )

    # Count-plus-top-k ablation baseline.
    topk_membership_head = (
        linear_parameters(32, 16)
        + linear_parameters(16, 1)
    )
    topk_cardinality_head = linear_parameters(64, 5)
    topk_null_head = linear_parameters(64, 1)
    topk_new_parameters = (
        topk_membership_head
        + topk_cardinality_head
        + topk_null_head
    )
    topk_total_parameters = (
        frozen_deployed_base
        + topk_new_parameters
    )

    # Primary low-rank four-slot decoder.
    slot_router_key_projection = linear_parameters(32, 8)
    slot_global_context_projection = linear_parameters(64, 8)
    slot_identity_embeddings = 4 * 8
    slot_null_head = linear_parameters(64, 4)

    slot_new_parameters = (
        slot_router_key_projection
        + slot_global_context_projection
        + slot_identity_embeddings
        + slot_null_head
    )
    slot_total_parameters = (
        frozen_deployed_base
        + slot_new_parameters
    )

    if slot_new_parameters > 1200:
        failures.append(
            "primary slot decoder exceeds 1200 new parameters"
        )

    if slot_total_parameters > 3500:
        failures.append(
            "primary slot model exceeds 3500 deployed parameters"
        )

    parameter_rows = [
        {
            "variant": "frozen_base",
            "component": "temporal_local_gcn_encoder",
            "trainable": False,
            "parameter_count": frozen_encoder_parameters,
        },
        {
            "variant": "frozen_base",
            "component": "graph_head",
            "trainable": False,
            "parameter_count": frozen_graph_head_parameters,
        },
        {
            "variant": "A4a-TopK",
            "component": "router_membership_32_16_1",
            "trainable": True,
            "parameter_count": topk_membership_head,
        },
        {
            "variant": "A4a-TopK",
            "component": "cardinality_64_5",
            "trainable": True,
            "parameter_count": topk_cardinality_head,
        },
        {
            "variant": "A4a-TopK",
            "component": "null_64_1",
            "trainable": True,
            "parameter_count": topk_null_head,
        },
        {
            "variant": "A4a-TopK",
            "component": "new_decoder_total",
            "trainable": True,
            "parameter_count": topk_new_parameters,
        },
        {
            "variant": "A4a-TopK",
            "component": "deployed_total",
            "trainable": "",
            "parameter_count": topk_total_parameters,
        },
        {
            "variant": "A4a-Slot-LR",
            "component": "router_key_projection_32_8",
            "trainable": True,
            "parameter_count": slot_router_key_projection,
        },
        {
            "variant": "A4a-Slot-LR",
            "component": "global_context_projection_64_8",
            "trainable": True,
            "parameter_count": slot_global_context_projection,
        },
        {
            "variant": "A4a-Slot-LR",
            "component": "slot_identity_embeddings_4_8",
            "trainable": True,
            "parameter_count": slot_identity_embeddings,
        },
        {
            "variant": "A4a-Slot-LR",
            "component": "slot_null_head_64_4",
            "trainable": True,
            "parameter_count": slot_null_head,
        },
        {
            "variant": "A4a-Slot-LR",
            "component": "new_decoder_total",
            "trainable": True,
            "parameter_count": slot_new_parameters,
        },
        {
            "variant": "A4a-Slot-LR",
            "component": "deployed_total",
            "trainable": "",
            "parameter_count": slot_total_parameters,
        },
    ]
    write_csv(
        args.output_dir / "a4a_slot_parameter_budget.csv",
        parameter_rows,
    )

    ablations = [
        {
            "order": 1,
            "name": "A4a-TopK",
            "primary_output": "count-controlled top-k router set",
            "losses": (
                "membership BCE + cardinality CE + null BCE + "
                "count/router consistency"
            ),
            "directional_proxy": False,
            "purpose": (
                "Controls for decoder retraining without slot structure."
            ),
        },
        {
            "order": 2,
            "name": "A4a-Slot",
            "primary_output": "four unordered router-or-NULL slots",
            "losses": "permutation-invariant slot matching only",
            "directional_proxy": False,
            "purpose": "Measures the value of direct set representation.",
        },
        {
            "order": 3,
            "name": "A4a-Slot+Membership",
            "primary_output": "four unordered router-or-NULL slots",
            "losses": (
                "slot matching + derived union-membership BCE"
            ),
            "directional_proxy": False,
            "purpose": "Tests router-level coverage supervision.",
        },
        {
            "order": 4,
            "name": "A4a-Slot+Count",
            "primary_output": "four unordered router-or-NULL slots",
            "losses": (
                "slot matching + derived slot-cardinality CE"
            ),
            "directional_proxy": False,
            "purpose": "Tests explicit variable-cardinality supervision.",
        },
        {
            "order": 5,
            "name": "A4a-Slot-Full",
            "primary_output": "four unordered router-or-NULL slots",
            "losses": (
                "slot matching + derived membership BCE + derived count CE "
                "+ duplicate penalty"
            ),
            "directional_proxy": False,
            "purpose": "Primary A4a candidate.",
        },
        {
            "order": 6,
            "name": "A4a-Slot-Full+DirectionalProxy",
            "primary_output": "four unordered router-or-NULL slots",
            "losses": (
                "full slot losses with auxiliary directional context"
            ),
            "directional_proxy": True,
            "purpose": (
                "Validation-only optional ablation; never a hard gate."
            ),
        },
    ]
    write_csv(
        args.output_dir / "a4a_slot_ablation_matrix.csv",
        ablations,
    )

    a3_12_validation = get_validation_a3_12(a3_12_summary)
    oracle_gap = float(
        a3_12_validation[
            "temporal_oracle_minus_primary_exact_gap"
        ]
    )

    if oracle_gap < 0.10:
        failures.append(
            "A3.12 validation oracle gap is below the 10-point A4a trigger"
        )

    candidate_policy = operational_policy["policy_roles"][
        "candidate_localization"
    ]["policy"]

    contract = {
        "status": "PASS" if not failures else "FAIL",
        "designation": (
            "V4-A4a-0S Null-Aware Slot-Set Decoder Frozen Contract"
        ),
        "experiment_family": "v4_a4a_null_aware_slot_set_decoder",
        "primary_experiment_name": "v4_a4a_slot_full_decoder_only_seed7",
        "scientific_hypothesis": (
            "A four-slot null-aware permutation-invariant decoder can "
            "recover a meaningful fraction of the A3.12 cardinality/set gap "
            "without changing the strong frozen A3 local-plus-graph encoder."
        ),
        "frozen_base": {
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "model_source_path": str(args.model_source),
            "model_source_sha256": model_source_sha,
            "original_parameter_count": actual_parameter_count,
            "original_trainer_sha256": recorded_trainer_sha,
            "original_trainer_hash_matches": trainer_matches,
            "canonical_original_trainer": canonical_trainer,
            "frozen_modules": [
                "temporal_conv",
                "local_projection",
                "gcn",
                "graph_head",
            ],
            "replaced_modules": [
                "attacker_hidden",
                "attacker_out",
                "count_head",
            ],
        },
        "a4a_slot_primary_architecture": {
            "slot_count": 4,
            "router_classes": 16,
            "null_classes": 1,
            "classes_per_slot": 17,
            "slot_latent_dim": 8,
            "router_key_projection": [32, 8],
            "global_context_projection": [64, 8],
            "slot_identity_embeddings": [4, 8],
            "null_head": [64, 4],
            "router_logit_definition": (
                "dot(global_context_8 + slot_embedding_s, "
                "router_key_projection(h_node_r)) / sqrt(8)"
            ),
            "slot_probability_definition": (
                "softmax over 16 router logits plus one slot-specific NULL logit"
            ),
            "new_trainable_parameter_count": slot_new_parameters,
            "total_deployed_parameter_count": slot_total_parameters,
            "new_decoder_parameter_budget_max": 1200,
            "total_deployed_parameter_budget_max": 3500,
            "directional_proxy_in_primary_model": False,
        },
        "set_training_and_inference": {
            "training_matching": (
                "Permutation-invariant exact/Hungarian matching between "
                "true routers and four slots; unmatched slots target NULL."
            ),
            "hardware_reference_inference": (
                "Unique constrained assignment maximizing combined slot "
                "log-probability; each real router may appear at most once; "
                "NULL may be assigned to multiple slots."
            ),
            "hardware_candidate_inference": (
                "Greedy unique assignment ordered by slot confidence; compare "
                "against exact constrained assignment before RTL freeze."
            ),
            "duplicate_handling": (
                "Duplicates are prohibited by constrained inference and "
                "discouraged during training by a differentiable duplicate loss."
            ),
            "derived_membership_probability": (
                "q_r = 1 - product_s(1 - p(slot_s = router_r))"
            ),
            "derived_cardinality_distribution": (
                "Poisson-binomial distribution of four non-NULL slot "
                "probabilities."
            ),
        },
        "training_contract": {
            "trainable_modules": [
                "router_key_projection",
                "global_context_projection",
                "slot_identity_embeddings",
                "slot_null_head",
            ],
            "encoder_frozen": True,
            "graph_head_frozen": True,
            "training_split_only": True,
            "development_test_access_during_training": False,
            "seed": 7,
            "epochs_max": 75,
            "early_stopping_patience": 10,
            "minimum_delta": 0.0001,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size_primary": 512,
            "batch_size_fallback": 256,
            "initialization": {
                "linear_weights": "xavier_uniform",
                "linear_biases": "zeros",
                "slot_identity_embeddings": "normal_std_0.02",
            },
            "primary_full_loss_weights": {
                "permutation_invariant_slot_matching": 1.0,
                "derived_union_membership_bce": 0.5,
                "derived_cardinality_cross_entropy": 0.5,
                "duplicate_slot_penalty": 0.1,
            },
            "null_target_definition": (
                "Unmatched slots target NULL; a normal sample assigns all "
                "four slots to NULL."
            ),
            "null_class_weighting": (
                "Freeze from training-split prevalence only; record the chosen "
                "value before validation selection."
            ),
        },
        "evaluation_contract": {
            "temporal_mode": candidate_policy["mode"],
            "temporal_horizon": candidate_policy["horizon"],
            "deoverlap_stride_epochs": 8,
            "outer_graph_gate": (
                "Frozen A3 graph head and validation-frozen graph threshold"
            ),
            "graph_threshold": candidate_policy["graph_threshold"],
            "primary_predicted_set_source": (
                "null-aware slot decoder, not count-controlled top-k"
            ),
            "validation_only_model_and_policy_selection": True,
            "development_test_transfer_count": 1,
            "development_test_used_for_selection": False,
            "run_level_paired_bootstrap_required": True,
            "bootstrap_unit": (
                "run or matched/scenario group; never independent temporal samples"
            ),
        },
        "validation_selection_contract": {
            "hard_gate_persistent_normal_run_false_isolation_fraction": 0.0,
            "hard_gate_attacker_set_precision_min": 0.98,
            "hard_gate_candidate_coverage_min": 0.90,
            "minimum_exact_improvement_over_a3_h32_percentage_points": 2.0,
            "selection_after_hard_gates": (
                "maximize stable exact localization, then attack exact "
                "localization, then minimize normal window false isolation"
            ),
            "a3_h32_reference_exact": a3_12_validation[
                "temporal_primary_attack_exact"
            ],
            "a3_h32_oracle_count_ceiling": a3_12_validation[
                "temporal_oracle_attack_exact"
            ],
        },
        "kill_criteria": [
            (
                "Reject further A4a development if the best validation-selected "
                "slot model improves attack exact localization by less than "
                "2 percentage points over A3.11-H32."
            ),
            (
                "Reject any A4a model that increases persistent normal-run "
                "false isolation."
            ),
            (
                "Reject a gain achieved only through material candidate-coverage "
                "collapse below 0.90."
            ),
            (
                "Reject the structured slot design if it does not outperform "
                "the retrained A4a-TopK ablation."
            ),
            (
                "After a kill decision, close V4 architecture tuning and move "
                "to V5 rather than introducing development-test-specific fixes."
            ),
        ],
        "frozen_transfer_promotion_criteria": {
            "desirable_exact_improvement_percentage_points_min": 3.0,
            "strong_exact_improvement_percentage_points": "3 to 5",
            "attacker_set_precision_min": 0.98,
            "persistent_normal_run_false_isolation_fraction": 0.0,
            "candidate_coverage_must_not_collapse": True,
            "parameter_budget_must_pass": True,
            "latency_budget_must_pass": True,
            "no_post_transfer_retuning": True,
        },
        "ablation_order": [row["name"] for row in ablations],
        "directional_proxy_contract": {
            "a3_15_decision": a3_15_summary[
                "validation_based_decision_class"
            ],
            "best_proxy": a3_15_summary.get(
                "best_validation_directional_proxy"
            ),
            "primary_model_use": False,
            "optional_ablation_use": True,
            "hard_gate_use": False,
            "selection_source": "validation only",
        },
        "hard_normal_claim_boundary": (
            "A4a is expected to recover part of the decoder/cardinality gap. "
            "It is not expected to fully solve the A3.15 router-3 hard-normal "
            "ambiguity because those cases are genuinely attacker-like under "
            "the saved V4 feature semantics. Remaining failures are a V4 "
            "representation/semantics limitation and must be carried into V5."
        ),
        "paper_claim_boundary": (
            "The strongest defensible V4 claim is high-precision candidate "
            "localization with temporally stable attacker-set estimation. "
            "Fully autonomous isolation remains conditional on strict "
            "persistent-normal safety gates."
        ),
        "statistical_reporting_contract": {
            "paired_baseline": "A3.11-H32 frozen candidate policy",
            "required_outputs": [
                "paired exact-localization difference",
                "95% run-level bootstrap confidence interval",
                "number of runs improved",
                "number of runs regressed",
                "number of runs unchanged",
            ],
        },
        "v5_requirements_preserved": [
            "explicit legitimate-source and transit-router labels",
            "explicit valid, idle, and clipped port semantics",
            "local injection and ejection quantities",
            "victim labels",
            "matched benign/attack counterfactual runs",
            "independent publication holdout",
        ],
        "diagnostic_evidence": {
            "a3_12_validation_threshold_exact": a3_12_validation[
                "temporal_primary_attack_exact"
            ],
            "a3_12_validation_oracle_count_exact": a3_12_validation[
                "temporal_oracle_attack_exact"
            ],
            "a3_12_validation_oracle_gap": oracle_gap,
            "a3_13_decision": a3_13_summary[
                "validation_based_decision_class"
            ],
            "a3_15_decision": a3_15_summary[
                "validation_based_decision_class"
            ],
        },
        "explicit_prohibitions": [
            "No model, loss, threshold, persistence, feature, or decoder selection on development test.",
            "No count-plus-top-k decoder as the primary A4a architecture.",
            "No naive duplicate removal as the reference set decoder.",
            "No removal or weakening of graph context based on graph-only ablation.",
            "No hard directional source/transit rule from A3.15 proxies.",
            "No end-to-end encoder fine-tuning in the first A4a decoder-only stage.",
            "No autonomous-isolation claim unless every safety gate passes.",
            "No publication claim that the current development test is an independent holdout.",
        ],
        "dataset_audit": {
            "total_samples": sample_count,
            "array_lengths": aligned_lengths,
            "observed_attacker_counts": observed_counts,
            "split_inventory": split_rows,
        },
        "provenance": {
            "metadata_sha256": sha256_file(args.metadata),
            "a3_12_summary_sha256": sha256_file(args.a3_12_summary),
            "a3_12_lock_sha256": sha256_file(args.a3_12_lock),
            "a3_13_summary_sha256": sha256_file(args.a3_13_summary),
            "a3_13_lock_sha256": sha256_file(args.a3_13_lock),
            "a3_15_summary_sha256": sha256_file(args.a3_15_summary),
            "a3_15_lock_sha256": sha256_file(args.a3_15_lock),
            "operational_policy_sha256": operational_policy_sha,
        },
        "training_performed": False,
        "prediction_generation_performed": False,
        "threshold_search_performed": False,
        "development_test_accessed": False,
    }

    contract_path = (
        args.output_dir / "A4A_SLOT_D0_FROZEN_CONTRACT.json"
    )
    contract_path.write_text(
        json.dumps(jsonable(contract), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    markdown_path = (
        args.output_dir / "A4A_SLOT_D0_FROZEN_CONTRACT.md"
    )
    markdown_path.write_text(
        render_markdown(contract),
        encoding="utf-8",
    )

    audit = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checkpoint_parameter_count": actual_parameter_count,
        "original_trainer_hash_match_count": len(trainer_matches),
        "canonical_original_trainer": canonical_trainer,
        "observed_attacker_counts": observed_counts,
        "authoritative_split_array": "split_id.npy",
        "a4a_topk_new_parameter_count": topk_new_parameters,
        "a4a_topk_total_deployed_parameter_count": topk_total_parameters,
        "a4a_slot_new_parameter_count": slot_new_parameters,
        "a4a_slot_total_deployed_parameter_count": slot_total_parameters,
        "contract_sha256": sha256_file(contract_path),
        "contract_markdown_sha256": sha256_file(markdown_path),
        "ready_for_a4a_slot_training_implementation": not failures,
        "training_performed": False,
        "development_test_accessed": False,
    }

    audit_path = args.output_dir / "implementation_audit.json"
    audit_path.write_text(
        json.dumps(jsonable(audit), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lock = {
        "status": (
            "A4A_0S_SLOT_CONTRACT_AUDIT_COMPLETE"
            if not failures
            else "A4A_0S_SLOT_CONTRACT_AUDIT_FAILED"
        ),
        "checkpoint_sha256": checkpoint_sha,
        "model_source_sha256": model_source_sha,
        "contract_sha256": sha256_file(contract_path),
        "contract_markdown_sha256": sha256_file(markdown_path),
        "implementation_audit_sha256": sha256_file(audit_path),
        "training_performed": False,
        "prediction_generation_performed": False,
        "threshold_search_performed": False,
        "development_test_accessed": False,
    }

    (args.output_dir / "A4A_0S_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    marker = (
        "V4_A4A_0S_SLOT_CONTRACT_AUDIT_PASS"
        if not failures
        else "V4_A4A_0S_SLOT_CONTRACT_AUDIT_FAIL"
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
