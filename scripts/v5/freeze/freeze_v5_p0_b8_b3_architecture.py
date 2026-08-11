#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

STAGE = "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(obj: Any) -> str:
    payload = json.dumps(
        obj, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def walk(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield p, v
            yield from walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            yield p, v
            yield from walk(v, p)


def find_key(obj: Any, name: str):
    for p, v in walk(obj):
        key = p.rsplit(".", 1)[-1].split("[", 1)[0]
        if key == name:
            return v
    return None


def bool_value(v: Any):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
    return None


def check_no_test_access(stage: str, report: Any, failures: list[str]):
    forbidden = {
        "test_split_accessed",
        "test_dataset_constructed",
        "test_tensors_read",
        "test_performance_evaluated",
        "test_directory_enumerated",
        "test_used_for_fitting",
        "test_used_for_selection",
        "test_used_for_threshold_selection",
    }
    for p, v in walk(report):
        key = p.rsplit(".", 1)[-1].split("[", 1)[0]
        if key in forbidden and bool_value(v) is True:
            failures.append(f"{stage}: forbidden test access at {p}")


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in (
        "a2", "a2_1", "a3", "r1c", "b1", "b2", "b3",
        "b4", "b5", "b6", "b7a", "b7b"
    ):
        ap.add_argument(f"--{name.replace('_', '-')}-dir",
                        dest=f"{name}_dir", type=Path, required=True)
    ap.add_argument("--wrapper", type=Path, required=True)
    ap.add_argument("--b3-script", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    out = args.output_dir.expanduser().resolve()
    if out.exists():
        print(f"STOP: output directory already exists: {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True)

    dirs = {
        key: getattr(args, f"{key}_dir").expanduser().resolve()
        for key in (
            "a2", "a2_1", "a3", "r1c", "b1", "b2", "b3",
            "b4", "b5", "b6", "b7a", "b7b"
        )
    }
    wrapper = args.wrapper.expanduser().resolve()
    b3_script = args.b3_script.expanduser().resolve()

    specs = {
        "a2": (
            "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT.json",
            None,
            "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_PASS",
            "PASS",
        ),
        "a2_1": (
            "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT.json",
            None,
            "V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_PASS",
            "PASS",
        ),
        "a3": (
            "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE.json",
            "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_LOCK.json",
            "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_PASS",
            "PASS",
        ),
        "r1c": (
            "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE.json",
            "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_LOCK.json",
            "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_PASS",
            "PASS",
        ),
        "b1": (
            "V5_P0_B1_STATIC_FINAL_EPOCH_MLP.json",
            "V5_P0_B1_STATIC_FINAL_EPOCH_MLP_LOCK.json",
            "V5_P0_B1_STATIC_FINAL_EPOCH_MLP_COMPLETE",
            "COMPLETE",
        ),
        "b2": (
            "V5_P0_B2_TEMPORAL_MEAN_POOLING.json",
            "V5_P0_B2_TEMPORAL_MEAN_POOLING_LOCK.json",
            "V5_P0_B2_TEMPORAL_MEAN_POOLING_COMPLETE",
            "COMPLETE",
        ),
        "b3": (
            "V5_P0_B3_CONV1D_ONLY.json",
            "V5_P0_B3_CONV1D_ONLY_LOCK.json",
            "V5_P0_B3_CONV1D_ONLY_COMPLETE",
            "COMPLETE",
        ),
        "b4": (
            "V5_P0_B4_GCN_ONLY.json",
            "V5_P0_B4_GCN_ONLY_LOCK.json",
            "V5_P0_B4_GCN_ONLY_COMPLETE",
            "COMPLETE",
        ),
        "b5": (
            "V5_P0_B5_CONV1D_GCN.json",
            "V5_P0_B5_CONV1D_GCN_LOCK.json",
            "V5_P0_B5_CONV1D_GCN_COMPLETE",
            "COMPLETE",
        ),
        "b6": (
            "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN.json",
            "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_LOCK.json",
            "V5_P0_B6_DUAL_TEMPORAL_FUSION_NO_GCN_COMPLETE",
            "COMPLETE",
        ),
        "b7a": (
            "V5_P0_B7A_DIRECTIONAL_GATED_GRAPH_DIAGNOSTIC.json",
            "V5_P0_B7A_DIRECTIONAL_GATED_GRAPH_DIAGNOSTIC_LOCK.json",
            "V5_P0_B7A_DIRECTIONAL_GATED_GRAPH_DIAGNOSTIC_COMPLETE",
            "COMPLETE",
        ),
        "b7b": (
            "V5_P0_B7B_DIRECTIONAL_GATED_GRAPH_MULTI_SEED_CONFIRMATION.json",
            "V5_P0_B7B_DIRECTIONAL_GATED_GRAPH_MULTI_SEED_CONFIRMATION_LOCK.json",
            "V5_P0_B7B_DIRECTIONAL_GATED_GRAPH_MULTI_SEED_CONFIRMATION_COMPLETE",
            "COMPLETE",
        ),
    }

    failures, warnings = [], []
    reports, locks, provenance = {}, {}, {}

    for p in (wrapper, b3_script):
        if not p.is_file():
            failures.append(f"missing required file: {p}")

    for key, (report_name, lock_name, marker_name, expected_status) in specs.items():
        d = dirs[key]
        report_path = d / report_name
        marker_path = d / marker_name
        if not report_path.is_file():
            failures.append(f"missing {key} report: {report_path}")
            continue
        if not marker_path.is_file():
            failures.append(f"missing {key} marker: {marker_path}")

        report = load(report_path)
        reports[key] = report
        provenance[f"{key}_report_sha256"] = sha(report_path)
        if report.get("status") != expected_status:
            failures.append(
                f"{key} status={report.get('status')!r}, "
                f"expected {expected_status!r}"
            )
        check_no_test_access(key, report, failures)

        if lock_name:
            lock_path = d / lock_name
            if not lock_path.is_file():
                failures.append(f"missing {key} lock: {lock_path}")
            else:
                lock = load(lock_path)
                locks[key] = lock
                provenance[f"{key}_lock_sha256"] = sha(lock_path)
                if lock.get("report_sha256") != sha(report_path):
                    failures.append(f"{key} report SHA does not match lock")

    if reports:
        a2_loader = reports.get("a2", {}).get("loader_contract", {})
        if a2_loader.get("primary_feature_variant") != "PRIMARY58":
            failures.append("A2 primary feature variant is not PRIMARY58")
        if a2_loader.get("metadata_concatenated_to_x") is not False:
            failures.append("A2 does not forbid metadata concatenation")
        if a2_loader.get("raw_mask_supplied_separately") is not True:
            failures.append("A2 does not require the separate raw mask")

        a3 = reports.get("a3", {})
        if a3.get("feature_variant") != "PRIMARY58":
            failures.append("A3 feature variant is not PRIMARY58")
        if int(a3.get("window", -1)) != 32:
            failures.append("A3 window is not 32")
        if int(a3.get("stride", -1)) != 8:
            failures.append("A3 stride is not 8")
        if a3.get("normalization_applied_by_wrapper") is not False:
            failures.append("A3 unexpectedly applies normalization")

        b3_count = reports.get("b3", {}).get("model", {}).get("parameter_count")
        if b3_count != 43208:
            failures.append(
                f"B3 parameter_count={b3_count!r}, expected 43208"
            )

        b7b = reports.get("b7b", {})
        decision = b7b.get("candidate_decision")
        if decision != "REJECT_B7A_AND_FREEZE_B3":
            failures.append(
                f"B7B decision={decision!r}; B3 freeze not authorized"
            )
        if b7b.get("next_stage") != STAGE:
            failures.append(
                f"B7B next_stage={b7b.get('next_stage')!r}, "
                f"expected {STAGE!r}"
            )

        seeds = find_key(b7b, "seeds")
        if seeds is None:
            warnings.append("B7B seed list not found in report")
        elif list(seeds) != [7, 17, 27]:
            failures.append(f"B7B seeds={seeds!r}, expected [7,17,27]")

        promotion = b7b.get("promotion_passed")
        if promotion is None:
            promotion = find_key(b7b, "promotion_passed")
        if promotion is not False:
            failures.append(
                f"B7B promotion_passed={promotion!r}, expected False"
            )

    if failures:
        hold = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "training_performed": False,
            "test_split_accessed": False,
            "provenance": provenance,
        }
        write_json(out / f"{STAGE}.json", hold)
        write_text(out / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for item in failures:
            print("FAIL:", item)
        return 1

    contract = {
        "contract_name": "V5_P0_FROZEN_B3_CONV1D_ONLY",
        "version": 1,
        "freeze_scope": "architecture_input_loss_and_interface",
        "weights_frozen": False,
        "selected_checkpoint_as_final": False,
        "selected_architecture": {
            "name": "B3_CAUSAL_DEPTHWISE_SEPARABLE_CONV1D_ONLY",
            "single_stage": True,
            "parameter_count": 43208,
            "graph_message_passing": False,
            "edge_index_as_model_input": False,
            "metadata_as_model_input": False,
        },
        "input": {
            "feature_variant": "PRIMARY58",
            "x_shape": ["B", 16, 58, 32],
            "window": 32,
            "stride": 8,
            "stored_x_already_standardized": True,
            "second_normalization_forbidden": True,
            "physical_port_mask_shape": ["B", 16, 10],
            "physical_port_mask_dtype": "bool",
            "physical_port_mask_source": "topology-derived recovered binary mask",
            "stored_standardized_mask_channels_forbidden": True,
            "active_only": False,
        },
        "temporal_encoder": {
            "input_projection": "Conv1d(58,64,kernel_size=1)",
            "block_type": "causal depthwise-separable residual Conv1D",
            "kernel_size": 3,
            "dilations": [1, 2, 4, 8],
            "channels": 64,
            "receptive_field": 31,
            "readout": "final causal encoded timestep",
        },
        "router_representation": {
            "input": "concat(temporal64, recovered_mask10)",
            "projection": "Linear(74,64)->ReLU",
        },
        "node_heads": {
            "tasks": ["source", "transit", "victim", "attack_path"],
            "pattern": "Linear(64,32)->ReLU->Linear(32,1)",
        },
        "graph_heads": {
            "pooling": ["router_mean", "router_max"],
            "encoder": "Linear(128,64)->ReLU",
            "attack": "Linear(64,1)",
            "attacker_count": "Linear(64,3)",
        },
        "losses": {
            "positive_weight_rule": "min(20, negatives/positives)",
            "attack": ["BCEWithLogitsLoss", 1.0],
            "attacker_count": ["CrossEntropyLoss", 0.5],
            "source": ["BCEWithLogitsLoss", 1.0],
            "transit": ["BCEWithLogitsLoss", 0.5],
            "victim": ["BCEWithLogitsLoss", 0.75],
            "attack_path": ["BCEWithLogitsLoss", 0.5],
        },
        "baseline_threshold": 0.5,
        "forbidden_inputs": [
            "run_id", "pair_id", "filename", "mode", "seed",
            "source/victim metadata", "scenario metadata",
            "serialization order", "run length", "window position",
            "router IDs", "router embeddings", "router coordinates",
            "status_flags", "stored standardized mask channels",
            "hashes or derivatives of quarantined provenance",
        ],
        "rejected_primary_alternatives": {
            "B4": "normalized one-hop GCN harmed localization",
            "B5": "Conv1D plus normalized GCN harmed generalization",
            "B6": "dual temporal fusion failed to preserve B2/B3 strengths",
            "B7A": "directional gated graph was not stable across fresh paired seeds",
        },
        "required_baselines": {
            "B1": "static final-epoch MLP",
            "B2": "temporal-mean MLP",
            "B3": "selected Conv1D-only architecture",
        },
    }
    contract["architecture_contract_sha256"] = canonical_sha(contract)

    p1 = {
        "architecture": "frozen B3 Conv1D-only",
        "before_training": [
            "audit P1 tensors, labels, masks, normalization, and leakage",
            "verify PRIMARY58, window 32, stride 8, and recovered Boolean mask",
            "lock P1 train/validation/test manifests",
            "keep P1 test inaccessible during development",
        ],
        "training": {
            "train_from_scratch": True,
            "fit_split": "P1 train only",
            "validation_may_control": [
                "early stopping",
                "checkpoint selection",
                "learning-rate scheduling",
                "threshold calibration",
            ],
            "multiple_preregistered_seeds_recommended": True,
        },
        "optional_zero_shot": {
            "source": "existing P0-trained B3 checkpoint",
            "target": "P1 validation or separate locked transfer split",
            "locked_P1_test_forbidden": True,
        },
        "test_boundary": {
            "development_access": False,
            "checkpoint_selection": False,
            "threshold_calibration": False,
            "final_evaluation_count": 1,
        },
        "required_P1_baseline": "B2 temporal-mean MLP",
        "architecture_search_reopening": (
            "requires a new explicitly named phase and cannot be based "
            "on P0 or P1 test performance"
        ),
    }

    contract_path = out / "V5_P0_B8_B3_CONV1D_ONLY_ARCHITECTURE_CONTRACT.json"
    p1_path = out / "V5_P0_B8_P1_PROTOCOL_BOUNDARY.json"
    spec_path = out / "V5_P0_B8_ARCHITECTURE_SPEC.md"
    p1_md_path = out / "V5_P0_B8_P1_PROTOCOL_BOUNDARY.md"

    write_json(contract_path, contract)
    write_json(p1_path, p1)

    write_text(spec_path, f"""# Frozen P0 Architecture

**Selected:** B3 causal depthwise-separable Conv1D-only  
**Parameters:** 43,208  
**Weights frozen:** No — this is an architecture freeze.

## Interface
- PRIMARY58 `x`: `[B,16,58,32]`
- recovered topology-derived Boolean mask: `[B,16,10]`
- no second normalization
- no metadata, router IDs, coordinates, or graph message passing

## Encoder
- `Conv1d(58,64,1)`
- causal depthwise-separable residual blocks
- kernel 3, dilations `[1,2,4,8]`
- receptive field 31
- final causal encoded timestep

## Heads
- router representation: `Linear(74,64)->ReLU`
- role heads: `Linear(64,32)->ReLU->Linear(32,1)`
- graph pooling: mean + max
- graph encoder: `Linear(128,64)->ReLU`
- attack head: `Linear(64,1)`
- count head: `Linear(64,3)`

## Decision evidence
B7B issued `REJECT_B7A_AND_FREEZE_B3`.

## Contract hash
`{contract["architecture_contract_sha256"]}`
""")

    write_text(p1_md_path, """# P1 Protocol Boundary

1. Audit P1 independently.
2. Verify the frozen PRIMARY58/window32/stride8/mask interface.
3. Lock P1 train, validation, and test manifests.
4. Train the frozen B3 architecture from scratch on P1 train.
5. Use only P1 validation for early stopping, checkpoint selection,
   scheduling, and threshold calibration.
6. Evaluate the locked P1 test exactly once.
7. Train B2 as the required simple baseline.
8. Do not reopen architecture search from test results.
""")

    provenance["wrapper_sha256"] = sha(wrapper)
    provenance["b3_training_script_sha256"] = sha(b3_script)
    provenance["b8_script_sha256"] = sha(Path(__file__))

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "decision": "FREEZE_B3_CONV1D_ONLY",
        "architecture_only_freeze": True,
        "weights_frozen": False,
        "training_performed": False,
        "test_split_accessed": False,
        "architecture_contract": contract,
        "p1_protocol_boundary": p1,
        "evidence": {
            "B7B_decision": "REJECT_B7A_AND_FREEZE_B3",
            "B7B_seeds": find_key(reports["b7b"], "seeds"),
            "B7B_promotion_passed": False,
        },
        "artifacts": {
            "contract": [str(contract_path), sha(contract_path)],
            "architecture_spec": [str(spec_path), sha(spec_path)],
            "p1_protocol_json": [str(p1_path), sha(p1_path)],
            "p1_protocol_markdown": [str(p1_md_path), sha(p1_md_path)],
        },
        "failures": failures,
        "warnings": warnings,
        "provenance": provenance,
        "next_stage": "V5_P1_INDEPENDENT_DATASET_AUDIT",
    }

    report_path = out / f"{STAGE}.json"
    write_json(report_path, report)

    lock = {
        "status": f"{STAGE}_COMPLETE",
        "decision": "FREEZE_B3_CONV1D_ONLY",
        "architecture_contract_sha256": contract["architecture_contract_sha256"],
        "report_sha256": sha(report_path),
        "contract_file_sha256": sha(contract_path),
        "architecture_spec_sha256": sha(spec_path),
        "p1_protocol_json_sha256": sha(p1_path),
        "p1_protocol_markdown_sha256": sha(p1_md_path),
        "b8_script_sha256": sha(Path(__file__)),
        "training_performed": False,
        "test_split_accessed": False,
        "next_stage": "V5_P1_INDEPENDENT_DATASET_AUDIT",
    }
    write_json(out / f"{STAGE}_LOCK.json", lock)
    write_text(out / f"{STAGE}_COMPLETE", f"{STAGE}_COMPLETE\n")

    print("===== V5 P0-B8 ARCHITECTURE FREEZE =====")
    print("decision: FREEZE_B3_CONV1D_ONLY")
    print("selected_architecture: B3_CAUSAL_DEPTHWISE_SEPARABLE_CONV1D_ONLY")
    print("parameter_count: 43208")
    print("feature_variant: PRIMARY58")
    print("window: 32")
    print("stride: 8")
    print("graph_message_passing: false")
    print("weights_frozen: false")
    print("architecture_only_freeze: true")
    print("architecture_contract_sha256:", contract["architecture_contract_sha256"])
    print("training_performed: false")
    print("test_split_accessed: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("next_stage: V5_P1_INDEPENDENT_DATASET_AUDIT")
    print(f"{STAGE}_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
