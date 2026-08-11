#!/usr/bin/env python3
"""
C1.0: freeze the Chrono-A1 mean-plus-max graph-readout experiment specification.

This stage performs no training and changes no source files. It:
- verifies the completed B1 closure;
- verifies the frozen Chrono-A1 source/checkpoint;
- records the exact A1 training arguments and split hashes;
- defines the sole permitted C1 change;
- freezes validation advancement criteria and stopping rules;
- writes JSON and Markdown experiment specifications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--c0-closure-manifest", required=True, type=Path)
    parser.add_argument("--a1-source", required=True, type=Path)
    parser.add_argument("--a1-model-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    closure_path = args.c0_closure_manifest.resolve()
    a1_source = args.a1_source.resolve()
    a1_model_dir = args.a1_model_dir.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    checkpoint_path = a1_model_dir / "best_model.pt"
    splits_path = a1_model_dir / "splits.npz"

    required_files = {
        "c0_closure_manifest": closure_path,
        "a1_source": a1_source,
        "a1_checkpoint": checkpoint_path,
        "a1_splits": splits_path,
        "dataset_x": data_dir / "x.npy",
        "dataset_y_graph": data_dir / "y_graph.npy",
        "dataset_y_node": data_dir / "y_node.npy",
        "dataset_edge_index": data_dir / "edge_index.npy",
    }

    if not repo_root.is_dir():
        raise SystemExit(f"STOP: repository root missing: {repo_root}")
    if not a1_model_dir.is_dir():
        raise SystemExit(f"STOP: A1 model directory missing: {a1_model_dir}")
    if not data_dir.is_dir():
        raise SystemExit(f"STOP: dataset directory missing: {data_dir}")

    for label, path in required_files.items():
        if not path.is_file():
            raise SystemExit(f"STOP: missing required file {label}: {path}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: C1.0 output directory already exists and is not empty: {output_dir}"
        )

    closure = load_json(closure_path)
    closure_checks = {
        "closure_stage_correct": closure.get("stage") == "C0_B1_closure",
        "closure_status_correct": closure.get("status") == "CLOSED_AND_REJECTED",
        "b1_verdict_reject": (
            closure.get("scientific_decision", {}).get("b1_verdict") == "reject"
        ),
        "a1_retained": (
            closure.get("scientific_decision", {}).get("retained_architecture")
            == "Chrono-A1 Conv1D-TemporalGCN"
        ),
        "b1_sampler_not_retained": (
            closure.get("scientific_decision", {})
            .get("scenario_balanced_sampler_retained")
            is False
        ),
    }
    if not all(closure_checks.values()):
        raise SystemExit(
            "STOP: B1 closure state is invalid: "
            + json.dumps(closure_checks, sort_keys=True)
        )

    source_hash = sha256(a1_source)
    checkpoint_hash = sha256(checkpoint_path)
    frozen_hash_checks = {
        "a1_source_hash_matches": source_hash == EXPECTED_A1_SOURCE_SHA256,
        "a1_checkpoint_hash_matches": (
            checkpoint_hash == EXPECTED_A1_CHECKPOINT_SHA256
        ),
    }
    if not all(frozen_hash_checks.values()):
        raise SystemExit(
            "STOP: frozen A1 provenance changed: "
            + json.dumps(frozen_hash_checks, sort_keys=True)
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "args" not in checkpoint:
        raise SystemExit("STOP: A1 checkpoint does not contain saved args.")

    saved_args = normalize_json(checkpoint["args"])
    best_epoch = checkpoint.get("epoch", checkpoint.get("best_epoch"))
    best_validation_score = checkpoint.get(
        "best_validation_score",
        checkpoint.get("best_val_score"),
    )

    with np.load(splits_path) as splits:
        split_arrays = {
            "train_idx": np.asarray(splits["train_idx"], dtype=np.int64),
            "val_idx": np.asarray(splits["val_idx"], dtype=np.int64),
            "test_idx": np.asarray(splits["test_idx"], dtype=np.int64),
        }

    split_checks = {
        "train_unique": (
            len(np.unique(split_arrays["train_idx"]))
            == len(split_arrays["train_idx"])
        ),
        "val_unique": (
            len(np.unique(split_arrays["val_idx"]))
            == len(split_arrays["val_idx"])
        ),
        "test_unique": (
            len(np.unique(split_arrays["test_idx"]))
            == len(split_arrays["test_idx"])
        ),
        "train_val_disjoint": (
            len(
                np.intersect1d(
                    split_arrays["train_idx"],
                    split_arrays["val_idx"],
                )
            )
            == 0
        ),
        "train_test_disjoint": (
            len(
                np.intersect1d(
                    split_arrays["train_idx"],
                    split_arrays["test_idx"],
                )
            )
            == 0
        ),
        "val_test_disjoint": (
            len(
                np.intersect1d(
                    split_arrays["val_idx"],
                    split_arrays["test_idx"],
                )
            )
            == 0
        ),
    }
    if not all(split_checks.values()):
        raise SystemExit(
            "STOP: A1 split integrity failed: "
            + json.dumps(split_checks, sort_keys=True)
        )

    x = np.load(data_dir / "x.npy", mmap_mode="r")
    y_graph = np.load(data_dir / "y_graph.npy", mmap_mode="r")
    y_node = np.load(data_dir / "y_node.npy", mmap_mode="r")
    edge_index = np.load(data_dir / "edge_index.npy", mmap_mode="r")

    dataset_checks = {
        "x_y_graph_rows_match": x.shape[0] == y_graph.shape[0],
        "x_y_node_rows_match": x.shape[0] == y_node.shape[0],
        "node_counts_match": x.shape[1] == y_node.shape[1],
        "edge_index_shape_valid": edge_index.ndim == 2 and edge_index.shape[0] == 2,
        "all_split_indices_in_range": all(
            bool(np.all(indices >= 0) and np.all(indices < x.shape[0]))
            for indices in split_arrays.values()
        ),
    }
    if not all(dataset_checks.values()):
        raise SystemExit(
            "STOP: dataset integrity failed: "
            + json.dumps(dataset_checks, sort_keys=True)
        )

    output_dir.mkdir(parents=True, exist_ok=False)

    specification = {
        "stage": "C1.0",
        "experiment_id": "chrono_c1_mean_plus_max_readout_seed7",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SPECIFICATION_FROZEN",
        "objective": (
            "Test whether concatenated mean-plus-max graph aggregation converts "
            "localized node evidence into a more robust graph-level attack "
            "decision than Chrono-A1 mean pooling."
        ),
        "control": {
            "name": "Chrono-A1 Conv1D-TemporalGCN",
            "source": str(a1_source),
            "source_sha256": source_hash,
            "model_dir": str(a1_model_dir),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "splits": str(splits_path),
            "splits_sha256": sha256(splits_path),
            "best_epoch": normalize_json(best_epoch),
            "best_validation_score": normalize_json(best_validation_score),
            "saved_training_arguments": saved_args,
        },
        "dataset": {
            "path": str(data_dir),
            "resolved_path": str(data_dir.resolve()),
            "x_shape": list(x.shape),
            "y_graph_shape": list(y_graph.shape),
            "y_node_shape": list(y_node.shape),
            "edge_index_shape": list(edge_index.shape),
            "split_counts": {
                key.replace("_idx", ""): int(len(value))
                for key, value in split_arrays.items()
            },
            "split_integrity": split_checks,
            "dataset_integrity": dataset_checks,
        },
        "single_permitted_change": {
            "component": "graph readout and required graph-head input width",
            "control_readout": "mean(h, dim=1)",
            "c1_readout": (
                "concat(mean(h, dim=1), max(h, dim=1).values), dim=-1"
            ),
            "control_graph_head_input_width": "gcn_out",
            "c1_graph_head_input_width": "2 * gcn_out",
        },
        "frozen_factors": [
            "corrected chronological V3 dataset",
            "exact train/validation/test split indices",
            "ordinary Chrono-A1 shuffled training loader",
            "no B1 sampler or scenario-balanced sampling",
            "Conv1D temporal encoder",
            "GCN layer count, widths and operations",
            "node head",
            "graph and node loss definitions",
            "graph and node class weights",
            "node loss weight",
            "optimizer",
            "learning rate",
            "weight decay",
            "batch size",
            "epoch budget",
            "early-stopping patience and minimum delta",
            "model seed",
            "checkpoint-selection score",
            "validation-only graph threshold protocol",
            "validation-only node threshold protocol",
            "blind-test threshold-transfer protocol",
        ],
        "forbidden_changes": [
            "B1 scenario-balanced sampler",
            "new sampler hierarchy",
            "loss-weight adjustment",
            "class-weight adjustment",
            "temporal encoder change",
            "GCN change",
            "node-head change",
            "hidden-width change",
            "learning-rate change",
            "additional auxiliary task",
            "test-set threshold selection",
        ],
        "validation_advancement_gate": {
            "delta_definition": "C1 minus Chrono-A1",
            "primary_improvement_any_one_required": {
                "graph_f1": ">= +0.010",
                "strength20_macro_recall": ">= +0.030",
                "worst_attack_run_recall": ">= +0.050",
            },
            "mandatory_safeguards_all_required": {
                "normal_run_macro_fpr": "<= +0.020",
                "worst_normal_run_fpr": "<= +0.050",
                "overall_graph_recall": ">= -0.020",
                "attack_run_macro_recall": ">= -0.020",
                "attack_only_node_f1": ">= -0.020",
                "attack_only_exact_localization": ">= -0.030",
                "top1_hit_rate": ">= -0.020",
                "top3_hit_rate": ">= -0.010",
            },
        },
        "decision_rules": {
            "reject_without_test": (
                "Reject if no primary improvement passes or any mandatory "
                "validation safeguard fails materially."
            ),
            "advance_to_blind_test": (
                "Advance only if at least one primary improvement and all "
                "mandatory safeguards pass on validation."
            ),
            "multi_seed_confirmation": (
                "Run matched A1/C1 additional seeds only if validation and "
                "blind-test transfer remain promising."
            ),
            "v3_stopping_rule": (
                "After the C1 verdict, or after matched multi-seed confirmation "
                "if C1 is promising, V3 model development ends permanently."
            ),
        },
        "closure_dependency": {
            "manifest": str(closure_path),
            "manifest_sha256": sha256(closure_path),
            "checks": closure_checks,
        },
        "provenance_checks": frozen_hash_checks,
        "training_started": False,
        "test_data_used_for_selection": False,
    }

    json_path = output_dir / "c1_experiment_spec.json"
    json_path.write_text(
        json.dumps(specification, indent=2) + "\n",
        encoding="utf-8",
    )

    markdown_lines = [
        "# C1 Mean-Plus-Max Readout Specification",
        "",
        "**Status:** SPECIFICATION FROZEN",
        "",
        "## Control",
        "",
        "`Chrono-A1 Conv1D-TemporalGCN`, ordinary shuffled training, seed 7.",
        "",
        "## Sole permitted change",
        "",
        "```text",
        "A1: mean graph pooling",
        "C1: concatenate mean and max graph pooling",
        "```",
        "",
        "The graph-head input width changes from `gcn_out` to `2 * gcn_out`.",
        "",
        "## Critical prohibition",
        "",
        "**The rejected B1 scenario-balanced sampler must not be imported or reused.**",
        "",
        "## Validation gate",
        "",
        "At least one primary improvement must pass and every mandatory safeguard "
        "must pass before blind-test transfer.",
        "",
        "## V3 stopping rule",
        "",
        "C1 is the final permitted V3 architecture intervention. After its verdict, "
        "V3 model development ends, except matched multi-seed confirmation if C1 "
        "is first judged promising.",
        "",
    ]
    markdown_path = output_dir / "c1_experiment_spec.md"
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    print("C1.0 EXPERIMENT SPECIFICATION: PASS")
    print("status=SPECIFICATION_FROZEN")
    print("control=Chrono-A1 Conv1D-TemporalGCN")
    print("single_change=mean pooling -> mean-plus-max concatenation")
    print("b1_sampler_allowed=False")
    print(f"a1_source_sha256={source_hash}")
    print(f"a1_checkpoint_sha256={checkpoint_hash}")
    print(f"split_integrity_pass={all(split_checks.values())}")
    print(f"dataset_integrity_pass={all(dataset_checks.values())}")
    print(f"training_started=False")
    print(f"test_data_used_for_selection=False")
    print(f"json_spec={json_path}")
    print(f"markdown_spec={markdown_path}")


if __name__ == "__main__":
    main()
