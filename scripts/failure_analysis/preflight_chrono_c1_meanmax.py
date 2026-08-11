#!/usr/bin/env python3
"""
C1.2: dry preflight for the Chrono-C1 mean-plus-max experiment.

No checkpoint is written and no persistent training is performed.

Checks:
- C1.0 and C1.1 prerequisite integrity;
- A1/C1 source and audit hashes;
- no B1 sampler references;
- C1 destination model directory is absent or empty;
- expected A1 command-line arguments are still supported by C1;
- frozen A1 split integrity and dataset shapes;
- one real chronological-data batch passes through C1;
- graph/node logits and losses have the expected shapes;
- loss and gradients are finite;
- same-seed C1 initialization and outputs are deterministic;
- graph readout is exactly concatenated mean-plus-max;
- parameter count is the expected 890;
- no test threshold selection or test inference occurs.

The script writes a preflight report and a frozen training-argument plan only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


EXPECTED_A1_SOURCE_SHA256 = (
    "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b"
)

FORBIDDEN_TOKENS = (
    "chrono_b1_scenario_sampler",
    "scenario_balanced",
    "scenario-balanced",
    "weightedrandomsampler",
)

EXPECTED_CLI_OPTIONS = (
    "--data",
    "--model-dir",
    "--split-mode",
    "--epochs",
    "--patience",
    "--min-delta",
    "--batch-size",
    "--lr",
    "--weight-decay",
    "--temporal-dim",
    "--gcn-hidden",
    "--gcn-out",
    "--node-loss-weight",
    "--graph-threshold",
    "--node-threshold",
    "--seed",
)

FROZEN_ARGUMENT_KEYS = (
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


def finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def compute_pos_weight(labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    if positive <= 0:
        raise ValueError("Cannot compute positive-class weight with zero positives.")
    return float(negative / positive)


def compare_values(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return a == b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--a1-source", required=True, type=Path)
    parser.add_argument("--c1-source", required=True, type=Path)
    parser.add_argument("--c1-spec", required=True, type=Path)
    parser.add_argument("--c1-source-audit", required=True, type=Path)
    parser.add_argument("--a1-model-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--c1-model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-batch-size", type=int, default=8)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    a1_source = args.a1_source.resolve()
    c1_source = args.c1_source.resolve()
    spec_path = args.c1_spec.resolve()
    audit_path = args.c1_source_audit.resolve()
    a1_model_dir = args.a1_model_dir.resolve()
    data_dir = args.data_dir.resolve()
    c1_model_dir = args.c1_model_dir.resolve()
    output_dir = args.output_dir.resolve()

    checkpoint_path = a1_model_dir / "best_model.pt"
    splits_path = a1_model_dir / "splits.npz"

    required = {
        "a1_source": a1_source,
        "c1_source": c1_source,
        "c1_spec": spec_path,
        "c1_source_audit": audit_path,
        "a1_checkpoint": checkpoint_path,
        "a1_splits": splits_path,
        "x": data_dir / "x.npy",
        "y_graph": data_dir / "y_graph.npy",
        "y_node": data_dir / "y_node.npy",
        "edge_index": data_dir / "edge_index.npy",
    }
    for label, path in required.items():
        if not path.is_file():
            raise SystemExit(f"STOP: missing prerequisite {label}: {path}")

    if not repo_root.is_dir():
        raise SystemExit(f"STOP: repository root missing: {repo_root}")
    if not data_dir.is_dir():
        raise SystemExit(f"STOP: chronological dataset missing: {data_dir}")
    if args.dry_batch_size <= 0:
        raise SystemExit("STOP: dry batch size must be positive.")

    if c1_model_dir.exists() and any(c1_model_dir.iterdir()):
        raise SystemExit(
            f"STOP: C1 model directory is already non-empty: {c1_model_dir}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: C1.2 output directory is already non-empty: {output_dir}"
        )

    specification = load_json(spec_path)
    source_audit = load_json(audit_path)

    prerequisite_checks = {
        "spec_stage_is_c1_0": specification.get("stage") == "C1.0",
        "spec_frozen": specification.get("status") == "SPECIFICATION_FROZEN",
        "spec_training_not_started": specification.get("training_started") is False,
        "spec_test_not_used": (
            specification.get("test_data_used_for_selection") is False
        ),
        "source_audit_stage_is_c1_1": source_audit.get("stage") == "C1.1",
        "source_audit_status_passed": (
            source_audit.get("status") == "SOURCE_GENERATED_AND_AUDITED"
        ),
        "source_audit_training_not_started": (
            source_audit.get("training_started") is False
        ),
        "source_audit_all_checks_pass": all(
            bool(value)
            for value in source_audit.get("audit_checks", {}).values()
        ),
        "a1_source_hash_frozen": (
            sha256(a1_source) == EXPECTED_A1_SOURCE_SHA256
        ),
        "a1_source_hash_matches_spec": (
            sha256(a1_source)
            == specification["control"]["source_sha256"]
        ),
        "c1_source_hash_matches_audit": (
            sha256(c1_source)
            == source_audit["c1_source"]["sha256"]
        ),
        "spec_hash_matches_audit": (
            sha256(spec_path) == source_audit["c1_spec"]["sha256"]
        ),
    }
    if not all(prerequisite_checks.values()):
        failed = [
            name for name, passed in prerequisite_checks.items() if not passed
        ]
        raise SystemExit(f"STOP: C1 prerequisite checks failed: {failed}")

    source_text = c1_source.read_text(encoding="utf-8")
    forbidden_hits = [
        token for token in FORBIDDEN_TOKENS if token in source_text.lower()
    ]

    source_structure_checks = {
        "no_b1_or_weighted_sampler_tokens": not forbidden_hits,
        "meanmax_readout_present": (
            "graph_mean = h.mean(dim=1)" in source_text
            and "graph_max = h.max(dim=1).values" in source_text
            and (
                "graph_embedding = "
                "torch.cat([graph_mean, graph_max], dim=-1)"
            )
            in source_text
        ),
        "mean_only_graph_embedding_absent": (
            len(
                re.findall(
                    r"(?m)^\s*graph_embedding\s*=\s*h\.mean\(dim=1\)\s*$",
                    source_text,
                )
            )
            == 0
        ),
        "c1_graph_head_width_present": (
            "self.graph_head = nn.Linear(2 * gcn_out, 1)" in source_text
        ),
    }
    if not all(source_structure_checks.values()):
        failed = [
            name for name, passed in source_structure_checks.items()
            if not passed
        ]
        raise SystemExit(f"STOP: C1 source structure failed: {failed}")

    help_result = subprocess.run(
        [sys.executable, str(c1_source), "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if help_result.returncode != 0:
        raise SystemExit(
            "STOP: C1 --help failed:\n"
            + help_result.stdout
            + "\n"
            + help_result.stderr
        )
    help_text = help_result.stdout + "\n" + help_result.stderr
    cli_option_checks = {
        option: option in help_text for option in EXPECTED_CLI_OPTIONS
    }
    if not all(cli_option_checks.values()):
        missing = [
            option for option, present in cli_option_checks.items()
            if not present
        ]
        raise SystemExit(f"STOP: C1 CLI options missing: {missing}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, dict):
        raise SystemExit("STOP: A1 checkpoint does not contain a saved args dict.")

    spec_args = specification["control"]["saved_training_arguments"]
    frozen_argument_checks: dict[str, bool] = {}
    frozen_arguments: dict[str, Any] = {}
    for key in FROZEN_ARGUMENT_KEYS:
        if key not in saved_args or key not in spec_args:
            frozen_argument_checks[key] = False
            continue
        frozen_arguments[key] = saved_args[key]
        frozen_argument_checks[key] = compare_values(
            saved_args[key],
            spec_args[key],
        )

    if not all(frozen_argument_checks.values()):
        failed = [
            key for key, passed in frozen_argument_checks.items()
            if not passed
        ]
        raise SystemExit(f"STOP: frozen A1 arguments changed or missing: {failed}")

    with np.load(splits_path) as splits:
        train_idx = np.asarray(splits["train_idx"], dtype=np.int64)
        val_idx = np.asarray(splits["val_idx"], dtype=np.int64)
        test_idx = np.asarray(splits["test_idx"], dtype=np.int64)

    x = np.load(data_dir / "x.npy", mmap_mode="r")
    y_graph = np.load(data_dir / "y_graph.npy", mmap_mode="r")
    y_node = np.load(data_dir / "y_node.npy", mmap_mode="r")
    edge_index = np.load(data_dir / "edge_index.npy").astype(np.int64)

    split_checks = {
        "train_unique": len(np.unique(train_idx)) == len(train_idx),
        "val_unique": len(np.unique(val_idx)) == len(val_idx),
        "test_unique": len(np.unique(test_idx)) == len(test_idx),
        "train_val_disjoint": len(np.intersect1d(train_idx, val_idx)) == 0,
        "train_test_disjoint": len(np.intersect1d(train_idx, test_idx)) == 0,
        "val_test_disjoint": len(np.intersect1d(val_idx, test_idx)) == 0,
        "all_indices_in_range": all(
            bool(np.all(indices >= 0) and np.all(indices < len(x)))
            for indices in (train_idx, val_idx, test_idx)
        ),
    }
    if not all(split_checks.values()):
        failed = [name for name, passed in split_checks.items() if not passed]
        raise SystemExit(f"STOP: split checks failed: {failed}")

    dataset_checks = {
        "x_rank_is_4": x.ndim == 4,
        "y_graph_rows_match": len(y_graph) == len(x),
        "y_node_rows_match": len(y_node) == len(x),
        "node_counts_match": y_node.shape[1] == x.shape[1],
        "edge_index_shape": edge_index.ndim == 2 and edge_index.shape[0] == 2,
        "train_nonempty": len(train_idx) > 0,
        "validation_nonempty": len(val_idx) > 0,
        "test_nonempty": len(test_idx) > 0,
    }
    if not all(dataset_checks.values()):
        failed = [name for name, passed in dataset_checks.items() if not passed]
        raise SystemExit(f"STOP: dataset checks failed: {failed}")

    train_graph_labels = np.asarray(y_graph[train_idx], dtype=np.int64)
    train_node_labels = np.asarray(y_node[train_idx], dtype=np.int64)
    graph_pos_weight = compute_pos_weight(train_graph_labels)
    node_pos_weight = compute_pos_weight(train_node_labels.reshape(-1))

    module = load_module(c1_source, "chrono_c1_preflight_source")
    required_symbols = (
        "TemporalGCN",
        "build_normalized_adjacency",
        "set_seed",
    )
    symbol_checks = {
        symbol: hasattr(module, symbol) for symbol in required_symbols
    }
    if not all(symbol_checks.values()):
        missing = [
            symbol for symbol, present in symbol_checks.items()
            if not present
        ]
        raise SystemExit(f"STOP: required C1 source symbols missing: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_count = min(args.dry_batch_size, len(train_idx))
    batch_indices = train_idx[:batch_count]

    batch_x = torch.from_numpy(
        np.array(x[batch_indices], dtype=np.float32, copy=True)
    ).to(device)
    batch_y_graph = torch.from_numpy(
        np.array(y_graph[batch_indices], dtype=np.float32, copy=True)
    ).to(device)
    batch_y_node = torch.from_numpy(
        np.array(y_node[batch_indices], dtype=np.float32, copy=True)
    ).to(device)

    a_hat = module.build_normalized_adjacency(
        edge_index,
        int(x.shape[1]),
    ).to(device)

    seed = int(saved_args["seed"])
    module.set_seed(seed)
    model_a = module.TemporalGCN(
        input_features=int(x.shape[-1]),
        temporal_dim=int(saved_args["temporal_dim"]),
        gcn_hidden=int(saved_args["gcn_hidden"]),
        gcn_out=int(saved_args["gcn_out"]),
    ).to(device)

    module.set_seed(seed)
    model_b = module.TemporalGCN(
        input_features=int(x.shape[-1]),
        temporal_dim=int(saved_args["temporal_dim"]),
        gcn_hidden=int(saved_args["gcn_hidden"]),
        gcn_out=int(saved_args["gcn_out"]),
    ).to(device)

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        graph_a_eval, node_a_eval = model_a(batch_x, a_hat)
        graph_b_eval, node_b_eval = model_b(batch_x, a_hat)

    deterministic_checks = {
        "same_seed_state_dict_equal": all(
            torch.equal(
                model_a.state_dict()[key].detach().cpu(),
                model_b.state_dict()[key].detach().cpu(),
            )
            for key in model_a.state_dict()
        ),
        "same_seed_graph_outputs_equal": torch.equal(
            graph_a_eval,
            graph_b_eval,
        ),
        "same_seed_node_outputs_equal": torch.equal(
            node_a_eval,
            node_b_eval,
        ),
    }

    model_a.train()
    model_a.zero_grad(set_to_none=True)
    graph_logits, node_logits = model_a(batch_x, a_hat)

    graph_criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(graph_pos_weight, device=device)
    )
    node_criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(node_pos_weight, device=device)
    )

    graph_loss = graph_criterion(graph_logits, batch_y_graph)
    node_loss = node_criterion(node_logits, batch_y_node)
    total_loss = (
        graph_loss
        + float(saved_args["node_loss_weight"]) * node_loss
    )
    total_loss.backward()

    gradients = [
        parameter.grad
        for parameter in model_a.parameters()
        if parameter.requires_grad
    ]
    gradient_checks = {
        "all_trainable_parameters_have_gradients": all(
            gradient is not None for gradient in gradients
        ),
        "all_gradients_finite": all(
            gradient is not None and finite_tensor(gradient)
            for gradient in gradients
        ),
        "at_least_one_nonzero_gradient": any(
            gradient is not None
            and bool(torch.any(gradient != 0).item())
            for gradient in gradients
        ),
    }

    model_a.eval()
    with torch.no_grad():
        h = model_a.temporal(batch_x)
        h = F.relu(model_a.gcn1(h, a_hat))
        h = F.relu(model_a.gcn2(h, a_hat))
        expected_node = model_a.node_head(h).squeeze(-1)
        graph_mean = h.mean(dim=1)
        graph_max = h.max(dim=1).values
        expected_embedding = torch.cat(
            [graph_mean, graph_max],
            dim=-1,
        )
        expected_graph = model_a.graph_head(
            expected_embedding
        ).squeeze(-1)
        actual_graph, actual_node = model_a(batch_x, a_hat)

    parameter_count = sum(
        parameter.numel() for parameter in model_a.parameters()
    )
    expected_parameter_count = 890

    runtime_checks = {
        "device_available": True,
        "adjacency_shape_correct": (
            list(a_hat.shape) == [int(x.shape[1]), int(x.shape[1])]
        ),
        "batch_shape_correct": list(batch_x.shape) == [
            batch_count,
            int(x.shape[1]),
            int(x.shape[2]),
            int(x.shape[3]),
        ],
        "graph_logits_shape_correct": list(graph_logits.shape) == [batch_count],
        "node_logits_shape_correct": (
            list(node_logits.shape) == [batch_count, int(x.shape[1])]
        ),
        "graph_logits_finite": finite_tensor(graph_logits),
        "node_logits_finite": finite_tensor(node_logits),
        "graph_loss_finite": finite_tensor(graph_loss),
        "node_loss_finite": finite_tensor(node_loss),
        "total_loss_finite": finite_tensor(total_loss),
        "manual_meanmax_graph_matches_forward": torch.equal(
            actual_graph,
            expected_graph,
        ),
        "manual_node_matches_forward": torch.equal(
            actual_node,
            expected_node,
        ),
        "embedding_width_is_2_gcn_out": (
            expected_embedding.shape[-1]
            == 2 * int(saved_args["gcn_out"])
        ),
        "graph_head_input_width_is_2_gcn_out": (
            model_a.graph_head.in_features
            == 2 * int(saved_args["gcn_out"])
        ),
        "parameter_count_is_890": parameter_count == expected_parameter_count,
        "model_directory_absent_or_empty": (
            not c1_model_dir.exists() or not any(c1_model_dir.iterdir())
        ),
    }

    all_checks = {
        **prerequisite_checks,
        **source_structure_checks,
        **{f"cli_{key}": value for key, value in cli_option_checks.items()},
        **{f"arg_{key}": value for key, value in frozen_argument_checks.items()},
        **split_checks,
        **dataset_checks,
        **symbol_checks,
        **deterministic_checks,
        **gradient_checks,
        **runtime_checks,
    }
    if not all(all_checks.values()):
        failed = [name for name, passed in all_checks.items() if not passed]
        raise SystemExit(f"STOP: C1.2 preflight failed: {failed}")

    output_dir.mkdir(parents=True, exist_ok=False)

    training_plan = {
        "data": str(data_dir),
        "model_dir": str(c1_model_dir),
        **{
            key: saved_args[key]
            for key in FROZEN_ARGUMENT_KEYS
        },
        "sampler": "ordinary A1 shuffled training",
        "b1_sampler": False,
        "source": str(c1_source),
        "source_sha256": sha256(c1_source),
    }
    plan_path = output_dir / "c1_frozen_training_arguments.json"
    plan_path.write_text(
        json.dumps(training_plan, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "stage": "C1.2",
        "status": "DRY_PREFLIGHT_PASSED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "persistent_training_started": False,
        "checkpoint_written": False,
        "test_inference_performed": False,
        "test_threshold_selection_performed": False,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "paths": {
            "repo_root": str(repo_root),
            "a1_source": str(a1_source),
            "c1_source": str(c1_source),
            "c1_spec": str(spec_path),
            "c1_source_audit": str(audit_path),
            "a1_checkpoint": str(checkpoint_path),
            "a1_splits": str(splits_path),
            "data_dir": str(data_dir),
            "c1_model_dir": str(c1_model_dir),
        },
        "hashes": {
            "a1_source": sha256(a1_source),
            "c1_source": sha256(c1_source),
            "c1_spec": sha256(spec_path),
            "c1_source_audit": sha256(audit_path),
            "a1_checkpoint": sha256(checkpoint_path),
            "a1_splits": sha256(splits_path),
        },
        "checks": all_checks,
        "forbidden_token_hits": forbidden_hits,
        "dataset": {
            "x_shape": list(x.shape),
            "y_graph_shape": list(y_graph.shape),
            "y_node_shape": list(y_node.shape),
            "edge_index_shape": list(edge_index.shape),
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
            "dry_batch_size": batch_count,
        },
        "weights_and_loss": {
            "graph_pos_weight_recomputed_from_train": graph_pos_weight,
            "node_pos_weight_recomputed_from_train": node_pos_weight,
            "node_loss_weight": float(saved_args["node_loss_weight"]),
            "dry_graph_loss": float(graph_loss.detach().cpu()),
            "dry_node_loss": float(node_loss.detach().cpu()),
            "dry_total_loss": float(total_loss.detach().cpu()),
        },
        "model": {
            "name": "Conv1D-MeanMax-TemporalGCN",
            "parameter_count": parameter_count,
            "expected_parameter_count": expected_parameter_count,
            "graph_head_input_width": model_a.graph_head.in_features,
            "node_count": int(x.shape[1]),
            "input_features": int(x.shape[-1]),
        },
        "frozen_training_plan": {
            "path": str(plan_path),
            "sha256": sha256(plan_path),
        },
    }
    report_path = output_dir / "c1_dry_preflight.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    help_path = output_dir / "c1_cli_help.txt"
    help_path.write_text(help_text, encoding="utf-8")

    print("C1.2 REAL-DATA DRY PREFLIGHT: PASS")
    print(f"device={device}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"c1_source_sha256={sha256(c1_source)}")
    print(f"b1_sampler_references={len(forbidden_hits)}")
    print(f"frozen_arguments_match=True")
    print(f"split_integrity_pass={all(split_checks.values())}")
    print(f"train_rows={len(train_idx)}")
    print(f"validation_rows={len(val_idx)}")
    print(f"test_rows={len(test_idx)}")
    print(f"dry_batch_size={batch_count}")
    print(f"parameter_count={parameter_count}")
    print(f"graph_head_input_width={model_a.graph_head.in_features}")
    print(
        "same_seed_deterministic="
        f"{all(deterministic_checks.values())}"
    )
    print(
        "manual_meanmax_graph_matches_forward="
        f"{runtime_checks['manual_meanmax_graph_matches_forward']}"
    )
    print(
        "all_gradients_finite="
        f"{gradient_checks['all_gradients_finite']}"
    )
    print(f"dry_total_loss={float(total_loss.detach().cpu()):.6f}")
    print(f"c1_model_dir_empty=True")
    print(f"persistent_training_started=False")
    print(f"test_inference_performed=False")
    print(f"training_plan={plan_path}")
    print(f"preflight_report={report_path}")
    print(f"cli_help={help_path}")


if __name__ == "__main__":
    main()
