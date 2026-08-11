#!/usr/bin/env python3
"""
Chrono-B1 Stage B1.8: post-training integrity and preliminary comparison audit.

Reads A1/B1 checkpoints, summaries, histories and split files. It verifies:
- both experiments completed and required artifacts exist;
- train/val/test indices are identical;
- model parameter names and tensor shapes are identical;
- frozen training arguments match;
- B1 sampler metadata is present;
- default-threshold metric deltas and frozen safeguard checks.

This is not the final B1 verdict. Threshold selection and aligned run-level
analysis must still use validation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


FROZEN_ARGS = (
    "data",
    "split_mode",
    "splits_file",
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


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(d: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in d:
            return float(d[name])
    raise KeyError(f"None of the metric keys exist: {names}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1-dir", required=True, type=Path)
    parser.add_argument("--b1-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    a1_dir = args.a1_dir.resolve()
    b1_dir = args.b1_dir.resolve()
    output = args.output.resolve()

    required = ("best_model.pt", "history.json", "summary.json", "splits.npz")
    missing = []
    for label, root in (("A1", a1_dir), ("B1", b1_dir)):
        for filename in required:
            path = root / filename
            if not path.is_file():
                missing.append(f"{label}:{path}")
    if missing:
        raise SystemExit("STOP: missing artifacts:\n" + "\n".join(missing))

    if output.exists():
        raise SystemExit(f"STOP: output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    a1_summary = load_json(a1_dir / "summary.json")
    b1_summary = load_json(b1_dir / "summary.json")
    a1_history = load_json(a1_dir / "history.json")
    b1_history = load_json(b1_dir / "history.json")

    a1_ckpt = torch.load(
        a1_dir / "best_model.pt", map_location="cpu", weights_only=False
    )
    b1_ckpt = torch.load(
        b1_dir / "best_model.pt", map_location="cpu", weights_only=False
    )

    a1_splits = np.load(a1_dir / "splits.npz")
    b1_splits = np.load(b1_dir / "splits.npz")
    split_checks = {
        name: bool(np.array_equal(a1_splits[name], b1_splits[name]))
        for name in ("train_idx", "val_idx", "test_idx")
    }

    a1_state = a1_ckpt["model_state_dict"]
    b1_state = b1_ckpt["model_state_dict"]
    model_name_match = set(a1_state) == set(b1_state)
    shape_mismatches = {}
    if model_name_match:
        for name in a1_state:
            a_shape = tuple(a1_state[name].shape)
            b_shape = tuple(b1_state[name].shape)
            if a_shape != b_shape:
                shape_mismatches[name] = {"a1": a_shape, "b1": b_shape}

    a1_args = dict(a1_ckpt["args"])
    b1_args = dict(b1_ckpt["args"])
    frozen_arg_differences = {}
    for key in FROZEN_ARGS:
        a_value = a1_args.get(key)
        b_value = b1_args.get(key)
        if a_value != b_value:
            frozen_arg_differences[key] = {"a1": a_value, "b1": b_value}

    sampler_summary = b1_summary.get("sampling")
    sampler_history_present = bool(b1_history) and all(
        isinstance(row.get("sampler"), dict) for row in b1_history
    )

    a1_val = a1_summary["val"]
    b1_val = b1_summary["val"]
    a1_test = a1_summary["test"]
    b1_test = b1_summary["test"]

    comparisons = {}
    for split_name, baseline, candidate in (
        ("val", a1_val, b1_val),
        ("test", a1_test, b1_test),
    ):
        comparisons[split_name] = {
            "graph_f1": {
                "a1": metric(baseline, "f1", "g_f1"),
                "b1": metric(candidate, "f1", "g_f1"),
            },
            "graph_recall": {
                "a1": metric(baseline, "recall", "g_rec"),
                "b1": metric(candidate, "recall", "g_rec"),
            },
            "graph_fpr": {
                "a1": metric(baseline, "fpr", "g_fpr"),
                "b1": metric(candidate, "fpr", "g_fpr"),
            },
            "node_f1": {
                "a1": metric(baseline, "node_f1"),
                "b1": metric(candidate, "node_f1"),
            },
            "exact_localization": {
                "a1": metric(baseline, "exact_localization", "exact_loc"),
                "b1": metric(candidate, "exact_localization", "exact_loc"),
            },
        }
        for values in comparisons[split_name].values():
            values["delta_b1_minus_a1"] = values["b1"] - values["a1"]

    val_delta = comparisons["val"]
    default_threshold_safeguards = {
        "overall_graph_recall_drop_le_0_02": (
            val_delta["graph_recall"]["delta_b1_minus_a1"] >= -0.02
        ),
        "node_f1_drop_le_0_02": (
            val_delta["node_f1"]["delta_b1_minus_a1"] >= -0.02
        ),
        "exact_localization_drop_le_0_03": (
            val_delta["exact_localization"]["delta_b1_minus_a1"] >= -0.03
        ),
    }

    integrity_checks = {
        "split_indices_identical": all(split_checks.values()),
        "model_parameter_names_identical": model_name_match,
        "model_parameter_shapes_identical": not shape_mismatches,
        "frozen_arguments_identical": not frozen_arg_differences,
        "sampler_summary_present": isinstance(sampler_summary, dict),
        "sampler_history_present_every_epoch": sampler_history_present,
        "a1_history_nonempty": bool(a1_history),
        "b1_history_nonempty": bool(b1_history),
    }

    result = {
        "integrity_pass": all(integrity_checks.values()),
        "integrity_checks": integrity_checks,
        "artifact_hashes": {
            "a1_checkpoint": file_sha256(a1_dir / "best_model.pt"),
            "b1_checkpoint": file_sha256(b1_dir / "best_model.pt"),
            "a1_summary": file_sha256(a1_dir / "summary.json"),
            "b1_summary": file_sha256(b1_dir / "summary.json"),
            "a1_splits": file_sha256(a1_dir / "splits.npz"),
            "b1_splits": file_sha256(b1_dir / "splits.npz"),
        },
        "best_epochs": {
            "a1": int(a1_summary["best_epoch"]),
            "b1": int(b1_summary["best_epoch"]),
        },
        "best_validation_scores": {
            "a1": float(a1_summary["best_val_score"]),
            "b1": float(b1_summary["best_val_score"]),
        },
        "split_checks": split_checks,
        "model_parameter_shape_mismatches": shape_mismatches,
        "frozen_argument_differences": frozen_arg_differences,
        "b1_sampling_summary": sampler_summary,
        "default_threshold_comparison": comparisons,
        "default_threshold_safeguards": default_threshold_safeguards,
        "default_threshold_safeguards_pass": all(
            default_threshold_safeguards.values()
        ),
        "interpretation": (
            "This is a preliminary threshold-0.5 check only. Do not issue the "
            "final B1 verdict until validation-selected threshold transfer and "
            "aligned run/scenario comparisons are complete."
        ),
    }

    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        "B1.8 POST-TRAINING INTEGRITY: "
        + ("PASS" if result["integrity_pass"] else "FAIL")
    )
    print(f"a1_best_epoch={result['best_epochs']['a1']}")
    print(f"b1_best_epoch={result['best_epochs']['b1']}")
    print(
        "a1_best_val_score="
        f"{result['best_validation_scores']['a1']:.6f}"
    )
    print(
        "b1_best_val_score="
        f"{result['best_validation_scores']['b1']:.6f}"
    )
    print(f"splits_identical={all(split_checks.values())}")
    print(f"model_parameter_names_identical={model_name_match}")
    print(f"model_parameter_shapes_identical={not shape_mismatches}")
    print(f"frozen_arguments_identical={not frozen_arg_differences}")
    print(f"sampler_summary_present={isinstance(sampler_summary, dict)}")
    print(f"sampler_history_present={sampler_history_present}")

    for split_name in ("val", "test"):
        print(f"\n{split_name.upper()} DEFAULT-THRESHOLD DELTAS (B1 - A1)")
        for metric_name, values in comparisons[split_name].items():
            print(
                f"{metric_name}={values['delta_b1_minus_a1']:+.6f} "
                f"(A1={values['a1']:.6f}, B1={values['b1']:.6f})"
            )

    print("\nDEFAULT-THRESHOLD SAFEGUARDS")
    for name, passed in default_threshold_safeguards.items():
        print(f"{name}={passed}")
    print(
        "default_threshold_safeguards_pass="
        f"{result['default_threshold_safeguards_pass']}"
    )
    print(f"output={output}")


if __name__ == "__main__":
    main()
