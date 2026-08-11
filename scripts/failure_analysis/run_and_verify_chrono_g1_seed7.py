#!/usr/bin/env python3
"""
G1.2: launch and verify the single authorized Chrono-G1 seed-7 training run.

This wrapper:
- requires successful G1.0 and G1.1 reports;
- verifies the frozen A1 and G1 sources/checkpoint;
- reads the frozen training arguments produced by G1.1;
- launches exactly one ordinary-sampler G1 training run;
- streams and saves the trainer output;
- verifies the produced checkpoint, summary, and splits;
- confirms exact split equality with Chrono-A1;
- confirms the checkpoint loads strictly into the 1,138-parameter G1 model;
- writes a post-training integrity report.

The inherited historical trainer may print default-threshold test diagnostics.
Those values are NOT used for model selection, threshold selection, or G1.3
advancement. G1.3 remains a validation-only gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPECTED_A1_SOURCE_SHA256 = (
    "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b"
)
EXPECTED_A1_CHECKPOINT_SHA256 = (
    "c29cdd2303669a61454ed118ef0be4b8809a635b12c0c42806f8b6992340e0ef"
)
EXPECTED_G1_SOURCE_SHA256 = (
    "3ebe7b8d37c5104c606daf0e65b488a3f553b815b01689ce172b12b8bbbffb5e"
)
EXPECTED_G1_PARAMETER_COUNT = 1138

REQUIRED_PLAN_KEYS = (
    "data",
    "model_output_option",
    "model_output_path",
    "split_mode",
    "epochs",
    "patience",
    "min_delta",
    "batch_size",
    "lr",
    "weight_decay",
    "temporal_dim",
    "gcn_hidden",
    "gcn_out",
    "node_loss_weight",
    "graph_threshold",
    "node_threshold",
    "seed",
    "sampler",
    "b1_sampler",
    "c1_readout",
    "graph_operator",
    "source",
    "source_sha256",
    "a1_splits",
    "a1_splits_sha256",
)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], str]:
    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
            "model_state",
        ):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value:
                if all(isinstance(v, torch.Tensor) for v in value.values()):
                    return value, key

        if checkpoint and all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            return checkpoint, "checkpoint_root"

    raise ValueError("Unable to locate a tensor state_dict in checkpoint.")


def exact_split_equality(
    reference_path: Path,
    candidate_path: Path,
) -> dict[str, bool]:
    with np.load(reference_path) as reference, np.load(candidate_path) as candidate:
        reference_keys = set(reference.files)
        candidate_keys = set(candidate.files)
        checks: dict[str, bool] = {
            "split_keys_equal": reference_keys == candidate_keys,
        }
        for key in sorted(reference_keys | candidate_keys):
            checks[f"{key}_present_in_both"] = (
                key in reference_keys and key in candidate_keys
            )
            if key in reference_keys and key in candidate_keys:
                checks[f"{key}_exactly_equal"] = np.array_equal(
                    np.asarray(reference[key]),
                    np.asarray(candidate[key]),
                )
        return checks


def compare_scalar(expected: Any, observed: Any) -> bool:
    if isinstance(expected, bool) or isinstance(observed, bool):
        return expected is observed
    if isinstance(expected, (int, float)) and isinstance(
        observed,
        (int, float),
    ):
        return abs(float(expected) - float(observed)) <= 1e-12
    return str(expected) == str(observed)


def stream_process(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return int(process.wait())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--a1-source", required=True, type=Path)
    parser.add_argument("--g1-source", required=True, type=Path)
    parser.add_argument("--g1-source-audit", required=True, type=Path)
    parser.add_argument("--g1-preflight-report", required=True, type=Path)
    parser.add_argument("--training-plan", required=True, type=Path)
    parser.add_argument("--a1-model-dir", required=True, type=Path)
    parser.add_argument("--g1-model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    a1_source = args.a1_source.resolve()
    g1_source = args.g1_source.resolve()
    source_audit_path = args.g1_source_audit.resolve()
    preflight_path = args.g1_preflight_report.resolve()
    training_plan_path = args.training_plan.resolve()
    a1_model_dir = args.a1_model_dir.resolve()
    g1_model_dir = args.g1_model_dir.resolve()
    output_dir = args.output_dir.resolve()

    a1_checkpoint = a1_model_dir / "best_model.pt"
    a1_splits = a1_model_dir / "splits.npz"

    required = {
        "repo_root": repo_root,
        "a1_source": a1_source,
        "g1_source": g1_source,
        "g1_source_audit": source_audit_path,
        "g1_preflight_report": preflight_path,
        "training_plan": training_plan_path,
        "a1_checkpoint": a1_checkpoint,
        "a1_splits": a1_splits,
    }
    for label, path in required.items():
        if label == "repo_root":
            if not path.is_dir():
                raise SystemExit(f"STOP: missing {label}: {path}")
        elif not path.is_file():
            raise SystemExit(f"STOP: missing {label}: {path}")

    if g1_model_dir.exists() and any(g1_model_dir.iterdir()):
        raise SystemExit(
            f"STOP: G1 model directory already non-empty: {g1_model_dir}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: G1.2 output directory already non-empty: {output_dir}"
        )

    source_audit = load_json(source_audit_path)
    preflight = load_json(preflight_path)
    plan = load_json(training_plan_path)

    missing_plan_keys = [
        key for key in REQUIRED_PLAN_KEYS if key not in plan
    ]
    if missing_plan_keys:
        raise SystemExit(
            f"STOP: frozen training plan missing keys: {missing_plan_keys}"
        )

    prerequisite_checks = {
        "g1_0_passed": (
            source_audit.get("stage") == "G1.0"
            and source_audit.get("status") == "SOURCE_DIFF_AUDIT_PASSED"
        ),
        "g1_1_passed": (
            preflight.get("stage") == "G1.1"
            and preflight.get("status")
            == "REAL_DATA_DRY_PREFLIGHT_PASSED"
        ),
        "g1_0_no_persistent_training": (
            source_audit.get("scope", {})
            .get("persistent_training_started") is False
        ),
        "g1_1_no_persistent_training": (
            preflight.get("persistent_training_started") is False
        ),
        "g1_1_no_checkpoint": (
            preflight.get("checkpoint_written") is False
        ),
        "a1_source_hash_frozen": (
            sha256(a1_source) == EXPECTED_A1_SOURCE_SHA256
        ),
        "a1_checkpoint_hash_frozen": (
            sha256(a1_checkpoint) == EXPECTED_A1_CHECKPOINT_SHA256
        ),
        "g1_source_hash_frozen": (
            sha256(g1_source) == EXPECTED_G1_SOURCE_SHA256
        ),
        "g1_source_matches_g1_0": (
            sha256(g1_source)
            == source_audit["sources"]["g1_sha256"]
        ),
        "g1_source_matches_g1_1": (
            sha256(g1_source)
            == preflight["hashes"]["g1_source"]
        ),
        "plan_hash_matches_g1_1": (
            sha256(training_plan_path)
            == preflight["frozen_training_plan"]["sha256"]
        ),
        "plan_source_path_matches": (
            Path(plan["source"]).resolve() == g1_source
        ),
        "plan_source_hash_matches": (
            plan["source_sha256"] == sha256(g1_source)
        ),
        "plan_a1_splits_path_matches": (
            Path(plan["a1_splits"]).resolve() == a1_splits
        ),
        "plan_a1_splits_hash_matches": (
            plan["a1_splits_sha256"] == sha256(a1_splits)
        ),
        "ordinary_sampler_frozen": (
            plan["sampler"] == "ordinary A1 shuffled training"
            and plan["b1_sampler"] is False
        ),
        "c1_readout_disabled": plan["c1_readout"] is False,
        "graphconv_operator_frozen": (
            "GraphConv" in plan["graph_operator"]
        ),
        "seed_is_7": int(plan["seed"]) == 7,
        "planned_model_path_matches": (
            Path(plan["model_output_path"]).resolve() == g1_model_dir
        ),
        "planned_output_option_is_out_dir": (
            plan["model_output_option"] == "--out-dir"
        ),
    }
    if not all(prerequisite_checks.values()):
        failed = [
            name for name, passed in prerequisite_checks.items()
            if not passed
        ]
        raise SystemExit(f"STOP: G1.2 prerequisite checks failed: {failed}")

    data_dir = Path(plan["data"]).resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"STOP: planned dataset missing: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=False)
    training_log = output_dir / "g1_training_console.log"

    command = [
        sys.executable,
        str(g1_source),
        "--data",
        str(data_dir),
        "--split-mode",
        str(plan["split_mode"]),
        "--epochs",
        str(plan["epochs"]),
        "--patience",
        str(plan["patience"]),
        "--min-delta",
        str(plan["min_delta"]),
        "--batch-size",
        str(plan["batch_size"]),
        "--lr",
        str(plan["lr"]),
        "--weight-decay",
        str(plan["weight_decay"]),
        "--temporal-dim",
        str(plan["temporal_dim"]),
        "--gcn-hidden",
        str(plan["gcn_hidden"]),
        "--gcn-out",
        str(plan["gcn_out"]),
        "--node-loss-weight",
        str(plan["node_loss_weight"]),
        "--graph-threshold",
        str(plan["graph_threshold"]),
        "--node-threshold",
        str(plan["node_threshold"]),
        "--seed",
        str(plan["seed"]),
        str(plan["model_output_option"]),
        str(g1_model_dir),
    ]

    command_path = output_dir / "g1_training_command.json"
    command_path.write_text(
        json.dumps(
            {
                "command": command,
                "cwd": str(repo_root),
                "source_sha256": sha256(g1_source),
                "training_plan_sha256": sha256(training_plan_path),
                "note": (
                    "Any default-threshold test metrics printed by the "
                    "historical trainer are diagnostic only and cannot "
                    "influence G1.3 advancement."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("G1.2 SINGLE SEED-7 TRAINING")
    print(f"source={g1_source}")
    print(f"source_sha256={sha256(g1_source)}")
    print(f"data={data_dir}")
    print(f"model_dir={g1_model_dir}")
    print("ordinary_a1_sampler=True")
    print("b1_sampler=False")
    print("c1_readout=False")
    print("seed=7")
    print("test_threshold_selection_performed=False")
    print("test_metrics_in_training_log_are_diagnostic_only=True")
    print("launching_training=True")
    print()

    return_code = stream_process(
        command,
        cwd=repo_root,
        log_path=training_log,
    )

    if return_code != 0:
        failure_report = {
            "stage": "G1.2",
            "status": "TRAINING_PROCESS_FAILED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "return_code": return_code,
            "command": command,
            "training_log": str(training_log),
            "training_log_sha256": sha256(training_log),
        }
        failure_path = output_dir / "g1_training_failure.json"
        failure_path.write_text(
            json.dumps(failure_report, indent=2) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(
            f"G1.2: FAIL — trainer return code {return_code}; "
            f"see {failure_path}"
        )

    produced = {
        "checkpoint": g1_model_dir / "best_model.pt",
        "summary": g1_model_dir / "summary.json",
        "splits": g1_model_dir / "splits.npz",
    }
    for label, path in produced.items():
        if not path.is_file():
            raise SystemExit(
                f"G1.2: FAIL — training returned 0 but missing {label}: {path}"
            )

    split_checks = exact_split_equality(
        a1_splits,
        produced["splits"],
    )

    checkpoint = torch.load(
        produced["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    state_dict, state_dict_location = extract_state_dict(checkpoint)

    saved_args = (
        checkpoint.get("args")
        if isinstance(checkpoint, dict)
        else None
    )
    saved_arg_checks: dict[str, bool] = {}
    if isinstance(saved_args, dict):
        for key in (
            "split_mode",
            "epochs",
            "patience",
            "min_delta",
            "batch_size",
            "lr",
            "weight_decay",
            "temporal_dim",
            "gcn_hidden",
            "gcn_out",
            "node_loss_weight",
            "graph_threshold",
            "node_threshold",
            "seed",
        ):
            saved_arg_checks[f"saved_arg_{key}_matches"] = (
                key in saved_args
                and compare_scalar(plan[key], saved_args[key])
            )
    else:
        saved_arg_checks["checkpoint_has_saved_args_dict"] = False

    module = load_module(
        g1_source,
        "chrono_g1_post_training_integrity",
    )
    x = np.load(data_dir / "x.npy", mmap_mode="r")
    edge_index = np.load(
        data_dir / "edge_index.npy",
    ).astype(np.int64)

    model = module.TemporalGCN(
        input_features=int(x.shape[-1]),
        temporal_dim=int(plan["temporal_dim"]),
        gcn_hidden=int(plan["gcn_hidden"]),
        gcn_out=int(plan["gcn_out"]),
    )
    strict_load_result = model.load_state_dict(
        state_dict,
        strict=True,
    )
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    adjacency = module.build_normalized_adjacency(
        edge_index,
        int(x.shape[1]),
    )

    summary = load_json(produced["summary"])
    summary_best_epoch = summary.get(
        "best_epoch",
        checkpoint.get("best_epoch")
        if isinstance(checkpoint, dict)
        else None,
    )

    integrity_checks = {
        **prerequisite_checks,
        **split_checks,
        **saved_arg_checks,
        "training_return_code_zero": return_code == 0,
        "checkpoint_exists": produced["checkpoint"].is_file(),
        "summary_exists": produced["summary"].is_file(),
        "splits_exist": produced["splits"].is_file(),
        "g1_source_hash_unchanged_after_training": (
            sha256(g1_source) == EXPECTED_G1_SOURCE_SHA256
        ),
        "checkpoint_state_loads_strictly": (
            len(strict_load_result.missing_keys) == 0
            and len(strict_load_result.unexpected_keys) == 0
        ),
        "parameter_count_is_1138": (
            parameter_count == EXPECTED_G1_PARAMETER_COUNT
        ),
        "adjacency_zero_diagonal": bool(
            torch.all(torch.diag(adjacency) == 0).item()
        ),
        "adjacency_has_48_entries": (
            int(adjacency.sum().item()) == 48
        ),
        "summary_is_nonempty": bool(summary),
        "best_epoch_is_available": summary_best_epoch is not None,
        "training_log_is_nonempty": training_log.stat().st_size > 0,
    }

    status = (
        "TRAINING_COMPLETED_AND_VERIFIED"
        if all(integrity_checks.values())
        else "POST_TRAINING_INTEGRITY_FAILED"
    )

    report = {
        "stage": "G1.2",
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "return_code": return_code,
        "protocol": {
            "single_authorized_seed": 7,
            "ordinary_a1_sampler": True,
            "b1_sampler": False,
            "c1_readout": False,
            "persistent_training_completed": True,
            "validation_threshold_selection_performed": False,
            "test_threshold_selection_performed": False,
            "historical_trainer_may_emit_default_threshold_test_diagnostics": True,
            "test_metrics_in_training_log_are_diagnostic_only": True,
            "next_stage_is_validation_only_gate": True,
        },
        "command": command,
        "paths": {
            "g1_source": str(g1_source),
            "training_plan": str(training_plan_path),
            "training_log": str(training_log),
            "g1_model_dir": str(g1_model_dir),
            "checkpoint": str(produced["checkpoint"]),
            "summary": str(produced["summary"]),
            "splits": str(produced["splits"]),
            "a1_splits": str(a1_splits),
        },
        "hashes": {
            "g1_source": sha256(g1_source),
            "training_plan": sha256(training_plan_path),
            "training_log": sha256(training_log),
            "checkpoint": sha256(produced["checkpoint"]),
            "summary": sha256(produced["summary"]),
            "splits": sha256(produced["splits"]),
            "a1_splits": sha256(a1_splits),
        },
        "checkpoint": {
            "state_dict_location": state_dict_location,
            "parameter_count": parameter_count,
            "best_epoch": summary_best_epoch,
            "state_tensor_count": len(state_dict),
        },
        "summary": summary,
        "checks": integrity_checks,
    }

    report_path = output_dir / "g1_post_training_integrity.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("G1.2 POST-TRAINING INTEGRITY")
    print(f"status={status}")
    print(f"return_code={return_code}")
    print(f"best_epoch={summary_best_epoch}")
    print(f"parameter_count={parameter_count}")
    print(f"checkpoint_sha256={sha256(produced['checkpoint'])}")
    print(f"source_sha256={sha256(g1_source)}")
    print(f"splits_identical_to_a1={all(split_checks.values())}")
    print(
        "saved_training_args_match="
        f"{bool(saved_arg_checks) and all(saved_arg_checks.values())}"
    )
    print("validation_threshold_selection_performed=False")
    print("test_threshold_selection_performed=False")
    print("test_metrics_in_training_log_are_diagnostic_only=True")
    print(f"training_log={training_log}")
    print(f"integrity_report={report_path}")

    if status != "TRAINING_COMPLETED_AND_VERIFIED":
        failed = [
            name
            for name, passed in integrity_checks.items()
            if not passed
        ]
        raise SystemExit(f"G1.2: FAIL — {failed}")

    print("G1.2 RESULT: PASS")
    print("next_authorized_stage=G1.3 validation-only threshold gate")


if __name__ == "__main__":
    main()
