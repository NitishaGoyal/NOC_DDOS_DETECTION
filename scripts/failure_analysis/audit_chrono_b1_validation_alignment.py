#!/usr/bin/env python3
"""
Chrono-B1 Stage B1.10: independent validation alignment audit.

Checks:
- A1 and B1 prediction files contain the required arrays;
- row counts, node counts, labels, metadata and real indices match exactly;
- exported real indices match the saved validation split;
- probabilities are finite and lie in [0, 1];
- labels are binary;
- validation indices are unique;
- every exported row is from the validation split.

No thresholds are selected and no test data is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = (
    "model_label",
    "split",
    "real_index",
    "graph_prob",
    "node_prob",
    "y_graph",
    "y_node",
    "run_id",
    "profile",
    "strength",
    "active_cores",
    "attackers",
    "end_epoch",
    "seed",
    "dataset_split",
)

ALIGNED_KEYS = (
    "real_index",
    "y_graph",
    "y_node",
    "run_id",
    "profile",
    "strength",
    "active_cores",
    "attackers",
    "end_epoch",
    "seed",
    "dataset_split",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_unit_interval(array: np.ndarray) -> bool:
    array = np.asarray(array)
    return bool(
        np.isfinite(array).all()
        and np.all(array >= 0.0)
        and np.all(array <= 1.0)
    )


def binary_array(array: np.ndarray) -> bool:
    values = np.unique(np.asarray(array))
    return bool(np.all(np.isin(values, [0, 1, 0.0, 1.0])))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1-predictions", required=True, type=Path)
    parser.add_argument("--b1-predictions", required=True, type=Path)
    parser.add_argument("--a1-splits", required=True, type=Path)
    parser.add_argument("--b1-splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    a1_path = args.a1_predictions.resolve()
    b1_path = args.b1_predictions.resolve()
    a1_splits_path = args.a1_splits.resolve()
    b1_splits_path = args.b1_splits.resolve()
    output = args.output.resolve()

    for path in (a1_path, b1_path, a1_splits_path, b1_splits_path):
        if not path.is_file():
            raise SystemExit(f"STOP: missing input file: {path}")

    if output.exists():
        raise SystemExit(f"STOP: output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    a1 = load_npz(a1_path)
    b1 = load_npz(b1_path)

    missing = {
        "a1": sorted(set(REQUIRED_KEYS) - set(a1)),
        "b1": sorted(set(REQUIRED_KEYS) - set(b1)),
    }

    if missing["a1"] or missing["b1"]:
        raise SystemExit(f"STOP: required prediction arrays missing: {missing}")

    with np.load(a1_splits_path) as d:
        a1_val_idx = np.asarray(d["val_idx"], dtype=np.int64)
    with np.load(b1_splits_path) as d:
        b1_val_idx = np.asarray(d["val_idx"], dtype=np.int64)

    a1_real = np.asarray(a1["real_index"], dtype=np.int64)
    b1_real = np.asarray(b1["real_index"], dtype=np.int64)

    alignment_checks = {
        key: bool(np.array_equal(a1[key], b1[key]))
        for key in ALIGNED_KEYS
    }

    checks: dict[str, bool] = {
        "required_keys_present": True,
        "a1_split_marker_is_val": str(a1["split"].item()) == "val",
        "b1_split_marker_is_val": str(b1["split"].item()) == "val",
        "row_count_equal": len(a1_real) == len(b1_real),
        "real_indices_equal": alignment_checks["real_index"],
        "a1_real_indices_unique": len(np.unique(a1_real)) == len(a1_real),
        "b1_real_indices_unique": len(np.unique(b1_real)) == len(b1_real),
        "a1_real_indices_match_saved_val_idx": bool(
            np.array_equal(a1_real, a1_val_idx)
        ),
        "b1_real_indices_match_saved_val_idx": bool(
            np.array_equal(b1_real, b1_val_idx)
        ),
        "saved_val_indices_equal": bool(
            np.array_equal(a1_val_idx, b1_val_idx)
        ),
        "a1_graph_prob_shape_valid": (
            a1["graph_prob"].shape == (len(a1_real),)
        ),
        "b1_graph_prob_shape_valid": (
            b1["graph_prob"].shape == (len(b1_real),)
        ),
        "a1_node_prob_shape_valid": (
            a1["node_prob"].ndim == 2
            and a1["node_prob"].shape[0] == len(a1_real)
        ),
        "b1_node_prob_shape_valid": (
            b1["node_prob"].ndim == 2
            and b1["node_prob"].shape[0] == len(b1_real)
        ),
        "node_probability_shapes_equal": (
            a1["node_prob"].shape == b1["node_prob"].shape
        ),
        "node_label_shapes_equal": (
            a1["y_node"].shape == b1["y_node"].shape
        ),
        "node_probability_and_label_shapes_match_a1": (
            a1["node_prob"].shape == a1["y_node"].shape
        ),
        "node_probability_and_label_shapes_match_b1": (
            b1["node_prob"].shape == b1["y_node"].shape
        ),
        "a1_graph_prob_finite_unit_interval": finite_unit_interval(
            a1["graph_prob"]
        ),
        "b1_graph_prob_finite_unit_interval": finite_unit_interval(
            b1["graph_prob"]
        ),
        "a1_node_prob_finite_unit_interval": finite_unit_interval(
            a1["node_prob"]
        ),
        "b1_node_prob_finite_unit_interval": finite_unit_interval(
            b1["node_prob"]
        ),
        "graph_labels_binary": binary_array(a1["y_graph"]),
        "node_labels_binary": binary_array(a1["y_node"]),
        "all_a1_rows_marked_val": bool(
            np.all(a1["dataset_split"].astype(str) == "val")
        ),
        "all_b1_rows_marked_val": bool(
            np.all(b1["dataset_split"].astype(str) == "val")
        ),
    }

    for key, passed in alignment_checks.items():
        checks[f"aligned_{key}"] = passed

    diagnostics: dict[str, Any] = {
        "a1_rows": int(len(a1_real)),
        "b1_rows": int(len(b1_real)),
        "a1_node_count": int(a1["node_prob"].shape[1]),
        "b1_node_count": int(b1["node_prob"].shape[1]),
        "a1_attack_samples": int(np.sum(a1["y_graph"] == 1)),
        "a1_normal_samples": int(np.sum(a1["y_graph"] == 0)),
        "graph_probability_mean": {
            "a1": float(np.mean(a1["graph_prob"])),
            "b1": float(np.mean(b1["graph_prob"])),
        },
        "graph_probability_std": {
            "a1": float(np.std(a1["graph_prob"])),
            "b1": float(np.std(b1["graph_prob"])),
        },
        "graph_probability_pearson": float(
            np.corrcoef(a1["graph_prob"], b1["graph_prob"])[0, 1]
        ),
        "mean_absolute_graph_probability_difference": float(
            np.mean(np.abs(a1["graph_prob"] - b1["graph_prob"]))
        ),
        "mean_absolute_node_probability_difference": float(
            np.mean(np.abs(a1["node_prob"] - b1["node_prob"]))
        ),
    }

    overall_pass = all(checks.values())

    report = {
        "stage": "B1.10",
        "split": "validation only",
        "overall_pass": overall_pass,
        "checks": checks,
        "alignment_checks": alignment_checks,
        "diagnostics": diagnostics,
        "files": {
            "a1_predictions": str(a1_path),
            "a1_predictions_sha256": sha256(a1_path),
            "b1_predictions": str(b1_path),
            "b1_predictions_sha256": sha256(b1_path),
            "a1_splits": str(a1_splits_path),
            "a1_splits_sha256": sha256(a1_splits_path),
            "b1_splits": str(b1_splits_path),
            "b1_splits_sha256": sha256(b1_splits_path),
        },
        "thresholds_selected": False,
        "test_data_read": False,
    }

    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        "B1.10 VALIDATION ALIGNMENT AUDIT: "
        + ("PASS" if overall_pass else "FAIL")
    )
    print(f"a1_rows={diagnostics['a1_rows']}")
    print(f"b1_rows={diagnostics['b1_rows']}")
    print(f"node_count={diagnostics['a1_node_count']}")
    print(f"attack_samples={diagnostics['a1_attack_samples']}")
    print(f"normal_samples={diagnostics['a1_normal_samples']}")
    print(f"real_indices_equal={checks['real_indices_equal']}")
    print(
        "saved_val_indices_equal="
        f"{checks['saved_val_indices_equal']}"
    )
    print(f"labels_equal={alignment_checks['y_graph'] and alignment_checks['y_node']}")
    print(
        "metadata_equal="
        f"{all(alignment_checks[k] for k in ALIGNED_KEYS if k not in {'real_index', 'y_graph', 'y_node'})}"
    )
    print(
        "probabilities_valid="
        f"{checks['a1_graph_prob_finite_unit_interval'] and checks['b1_graph_prob_finite_unit_interval'] and checks['a1_node_prob_finite_unit_interval'] and checks['b1_node_prob_finite_unit_interval']}"
    )
    print(
        "graph_probability_pearson="
        f"{diagnostics['graph_probability_pearson']:.6f}"
    )
    print(
        "mean_abs_graph_prob_difference="
        f"{diagnostics['mean_absolute_graph_probability_difference']:.6f}"
    )
    print(
        "mean_abs_node_prob_difference="
        f"{diagnostics['mean_absolute_node_probability_difference']:.6f}"
    )
    failed = [name for name, passed in checks.items() if not passed]
    print(f"failed_checks={failed}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
