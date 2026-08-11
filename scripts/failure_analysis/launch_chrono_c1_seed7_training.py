#!/usr/bin/env python3
"""
C1.3: launch and verify the seed-7 Chrono-C1 training run.

The launcher reads the actual output-directory CLI option and every frozen
training argument from the successful C1.2 preflight plan. It does not guess
or manually redefine hyperparameters.

The underlying A1-derived source may print default-threshold test diagnostics
at the end, as the historical A1/B1 trainer did. Those diagnostics are logged
but MUST NOT be used for the C1 advancement decision. Formal selection remains
validation-only; validation-selected thresholds are transferred later only if
C1 passes its validation gate.
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


ARGUMENT_FLAG_MAP = {
    "split_mode": "--split-mode",
    "epochs": "--epochs",
    "patience": "--patience",
    "min_delta": "--min-delta",
    "batch_size": "--batch-size",
    "lr": "--lr",
    "weight_decay": "--weight-decay",
    "temporal_dim": "--temporal-dim",
    "gcn_hidden": "--gcn-hidden",
    "gcn_out": "--gcn-out",
    "node_loss_weight": "--node-loss-weight",
    "graph_threshold": "--graph-threshold",
    "node_threshold": "--node-threshold",
    "seed": "--seed",
}

REQUIRED_MODEL_FILES = (
    "best_model.pt",
    "splits.npz",
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


def same_value(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-12
        except (TypeError, ValueError):
            return False
    return a == b


def state_dict_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    raise ValueError("Checkpoint contains no model_state_dict/state_dict.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--c1-source", required=True, type=Path)
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--training-plan", required=True, type=Path)
    parser.add_argument("--a1-splits", required=True, type=Path)
    parser.add_argument("--c1-model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    c1_source = args.c1_source.resolve()
    preflight_path = args.preflight_report.resolve()
    plan_path = args.training_plan.resolve()
    a1_splits_path = args.a1_splits.resolve()
    c1_model_dir = args.c1_model_dir.resolve()
    output_dir = args.output_dir.resolve()

    required_inputs = {
        "c1_source": c1_source,
        "preflight_report": preflight_path,
        "training_plan": plan_path,
        "a1_splits": a1_splits_path,
    }
    for label, path in required_inputs.items():
        if not path.is_file():
            raise SystemExit(f"STOP: missing {label}: {path}")

    if not repo_root.is_dir():
        raise SystemExit(f"STOP: repository root missing: {repo_root}")
    if c1_model_dir.exists() and any(c1_model_dir.iterdir()):
        raise SystemExit(
            f"STOP: C1 model directory already contains files: {c1_model_dir}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: C1.3 output directory already contains files: {output_dir}"
        )

    preflight = load_json(preflight_path)
    plan = load_json(plan_path)

    preflight_checks = {
        "stage_is_c1_2": preflight.get("stage") == "C1.2",
        "status_passed": preflight.get("status") == "DRY_PREFLIGHT_PASSED",
        "persistent_training_not_started": (
            preflight.get("persistent_training_started") is False
        ),
        "checkpoint_not_written": preflight.get("checkpoint_written") is False,
        "test_inference_not_performed_in_preflight": (
            preflight.get("test_inference_performed") is False
        ),
        "all_preflight_checks_true": all(
            bool(value) for value in preflight.get("checks", {}).values()
        ),
        "source_hash_matches_preflight": (
            sha256(c1_source) == preflight["hashes"]["c1_source"]
        ),
        "plan_hash_matches_preflight": (
            sha256(plan_path)
            == preflight["frozen_training_plan"]["sha256"]
        ),
    }
    if not all(preflight_checks.values()):
        failed = [
            name for name, passed in preflight_checks.items() if not passed
        ]
        raise SystemExit(f"STOP: C1.2 prerequisite failed: {failed}")

    output_option = str(plan.get("model_output_option", "")).strip()
    data_path = Path(str(plan.get("data", ""))).expanduser().resolve()
    planned_output = Path(
        str(plan.get("model_output_path", ""))
    ).expanduser().resolve()
    planned_source = Path(str(plan.get("source", ""))).expanduser().resolve()

    plan_checks = {
        "data_path_exists": data_path.exists(),
        "planned_source_matches": planned_source == c1_source,
        "planned_output_matches": planned_output == c1_model_dir,
        "source_hash_matches_plan": (
            sha256(c1_source) == plan.get("source_sha256")
        ),
        "output_option_is_cli_flag": output_option.startswith("--"),
        "b1_sampler_false": plan.get("b1_sampler") is False,
        "ordinary_a1_sampler": (
            plan.get("sampler") == "ordinary A1 shuffled training"
        ),
        "all_frozen_arguments_present": all(
            key in plan for key in ARGUMENT_FLAG_MAP
        ),
    }
    if not all(plan_checks.values()):
        failed = [name for name, passed in plan_checks.items() if not passed]
        raise SystemExit(f"STOP: frozen training plan invalid: {failed}")

    command = [
        sys.executable,
        str(c1_source),
        "--data",
        str(data_path),
        output_option,
        str(c1_model_dir),
    ]
    for key, flag in ARGUMENT_FLAG_MAP.items():
        command.extend([flag, str(plan[key])])

    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "c1_seed7_training.log"
    launch_manifest_path = output_dir / "c1_training_launch_manifest.json"

    launch_manifest = {
        "stage": "C1.3",
        "status": "TRAINING_LAUNCHED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "working_directory": str(repo_root),
        "source": {
            "path": str(c1_source),
            "sha256": sha256(c1_source),
        },
        "preflight": {
            "path": str(preflight_path),
            "sha256": sha256(preflight_path),
        },
        "training_plan": {
            "path": str(plan_path),
            "sha256": sha256(plan_path),
        },
        "a1_splits": {
            "path": str(a1_splits_path),
            "sha256": sha256(a1_splits_path),
        },
        "c1_model_dir": str(c1_model_dir),
        "log": str(log_path),
        "selection_protocol_note": (
            "Any default-threshold test diagnostics emitted by the historical "
            "trainer are not permitted for C1 model advancement. Advancement "
            "is determined later from validation-only analysis."
        ),
    }
    launch_manifest_path.write_text(
        json.dumps(launch_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("C1.3 TRAINING LAUNCH")
    print(f"source_sha256={sha256(c1_source)}")
    print(f"model_output_cli_option={output_option}")
    print(f"data={data_path}")
    print(f"model_dir={c1_model_dir}")
    print(f"seed={plan['seed']}")
    print(f"epochs={plan['epochs']}")
    print(f"patience={plan['patience']}")
    print(f"batch_size={plan['batch_size']}")
    print(f"sampler={plan['sampler']}")
    print("b1_sampler=False")
    print(f"log={log_path}")
    print("command=" + " ".join(command))
    print()

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()

    if return_code != 0:
        failure = {
            **launch_manifest,
            "status": "TRAINING_FAILED",
            "return_code": return_code,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        launch_manifest_path.write_text(
            json.dumps(failure, indent=2) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(
            f"C1.3 TRAINING: FAIL — trainer return code {return_code}"
        )

    missing_model_files = [
        name for name in REQUIRED_MODEL_FILES
        if not (c1_model_dir / name).is_file()
    ]
    if missing_model_files:
        raise SystemExit(
            f"STOP: training returned success but files are missing: "
            f"{missing_model_files}"
        )

    c1_splits_path = c1_model_dir / "splits.npz"
    with np.load(a1_splits_path) as a1_splits, np.load(c1_splits_path) as c1_splits:
        split_checks = {
            key: bool(np.array_equal(a1_splits[key], c1_splits[key]))
            for key in ("train_idx", "val_idx", "test_idx")
        }
    if not all(split_checks.values()):
        raise SystemExit(
            "STOP: C1 saved splits differ from A1: "
            + json.dumps(split_checks)
        )

    checkpoint_path = c1_model_dir / "best_model.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, dict):
        raise SystemExit("STOP: C1 checkpoint has no saved args dictionary.")

    argument_checks = {
        key: (
            key in checkpoint_args
            and same_value(checkpoint_args[key], plan[key])
        )
        for key in ARGUMENT_FLAG_MAP
    }
    if not all(argument_checks.values()):
        failed = [key for key, passed in argument_checks.items() if not passed]
        raise SystemExit(
            f"STOP: C1 checkpoint arguments differ from plan: {failed}"
        )

    module = load_module(c1_source, "chrono_c1_post_training_integrity")
    x = np.load(data_path / "x.npy", mmap_mode="r")
    model = module.TemporalGCN(
        input_features=int(x.shape[-1]),
        temporal_dim=int(plan["temporal_dim"]),
        gcn_hidden=int(plan["gcn_hidden"]),
        gcn_out=int(plan["gcn_out"]),
    )
    state_dict = state_dict_from_checkpoint(checkpoint)
    model.load_state_dict(state_dict, strict=True)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    graph_head_weight_shape = list(
        state_dict["graph_head.weight"].shape
    )
    architecture_checks = {
        "parameter_count_is_890": parameter_count == 890,
        "graph_head_weight_shape_is_1_by_16": (
            graph_head_weight_shape == [1, 16]
        ),
        "graph_head_input_width_is_16": model.graph_head.in_features == 16,
        "source_unchanged_after_training": (
            sha256(c1_source) == plan["source_sha256"]
        ),
    }
    if not all(architecture_checks.values()):
        failed = [
            name for name, passed in architecture_checks.items()
            if not passed
        ]
        raise SystemExit(
            f"STOP: post-training architecture integrity failed: {failed}"
        )

    produced_files = []
    for path in sorted(c1_model_dir.rglob("*")):
        if path.is_file():
            produced_files.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(c1_model_dir)),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256(path),
                }
            )

    summary_path = c1_model_dir / "summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else None

    integrity = {
        **launch_manifest,
        "status": "TRAINING_COMPLETED_AND_VERIFIED",
        "return_code": return_code,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "split_checks": split_checks,
        "checkpoint_argument_checks": argument_checks,
        "architecture_checks": architecture_checks,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "parameter_count": parameter_count,
            "graph_head_weight_shape": graph_head_weight_shape,
            "saved_epoch": checkpoint.get(
                "epoch",
                checkpoint.get("best_epoch"),
            ),
            "saved_validation_score": checkpoint.get(
                "best_validation_score",
                checkpoint.get("best_val_score"),
            ),
        },
        "summary": summary,
        "produced_files": produced_files,
        "formal_validation_gate_evaluated": False,
        "formal_test_transfer_performed": False,
    }
    integrity_path = output_dir / "c1_post_training_integrity.json"
    integrity_path.write_text(
        json.dumps(integrity, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("C1.3 TRAINING AND POST-TRAINING INTEGRITY: PASS")
    print(f"return_code={return_code}")
    print(f"a1_c1_splits_identical={all(split_checks.values())}")
    print(f"frozen_checkpoint_arguments_match=True")
    print(f"parameter_count={parameter_count}")
    print(f"graph_head_weight_shape={graph_head_weight_shape}")
    print(f"checkpoint_sha256={sha256(checkpoint_path)}")
    print(f"produced_file_count={len(produced_files)}")
    if summary is not None:
        print(f"summary_present=True")
        if "best_epoch" in summary:
            print(f"best_epoch={summary['best_epoch']}")
        elif "epoch" in summary:
            print(f"best_epoch={summary['epoch']}")
    else:
        print("summary_present=False")
    print("formal_validation_gate_evaluated=False")
    print("formal_test_transfer_performed=False")
    print(f"integrity_report={integrity_path}")


if __name__ == "__main__":
    main()
