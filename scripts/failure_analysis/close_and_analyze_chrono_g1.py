#!/usr/bin/env python3
"""
G1.4 targeted validation-only postmortem and closure freeze.

Uses only the already-exported A1/G1 validation predictions from G1.3.
It performs no model inference, no training, and no test access.

Outputs:
- targeted GraphConv postmortem;
- per-run graph delta table;
- router-degree node-activation table;
- artifact hash manifest;
- formal G1 closure manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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


def decode_strings(values: np.ndarray) -> np.ndarray:
    result = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return np.asarray(result, dtype=str)


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int | float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    def div(a: float, b: float) -> float:
        return float(a / b) if b else 0.0

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    f1 = div(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": div(fp, fp + tn),
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "q01": float(np.quantile(values, 0.01)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def node_error_profile(
    *,
    y_graph: np.ndarray,
    y_node: np.ndarray,
    node_prob: np.ndarray,
    threshold: float,
    adjacency: np.ndarray,
    degrees: np.ndarray,
) -> dict[str, Any]:
    prediction = (node_prob >= threshold).astype(np.int64)
    attack_mask = y_graph == 1
    normal_mask = y_graph == 0

    attack_true = y_node[attack_mask]
    attack_pred = prediction[attack_mask]
    attack_prob = node_prob[attack_mask]

    normal_pred = prediction[normal_mask]
    normal_prob = node_prob[normal_mask]

    attack_counts = binary_counts(attack_true.reshape(-1), attack_pred.reshape(-1))

    true_count = attack_true.sum(axis=1)
    predicted_count = attack_pred.sum(axis=1)

    adjacent_nonattacker_mask = np.zeros_like(attack_true, dtype=bool)
    for sample in range(len(attack_true)):
        attackers = np.flatnonzero(attack_true[sample] == 1)
        adjacent = np.zeros(attack_true.shape[1], dtype=bool)
        for attacker in attackers:
            adjacent |= adjacency[attacker].astype(bool)
        adjacent &= attack_true[sample] == 0
        adjacent_nonattacker_mask[sample] = adjacent

    other_nonattacker_mask = (attack_true == 0) & (~adjacent_nonattacker_mask)

    degree_rows = []
    for degree in sorted(np.unique(degrees).tolist()):
        nodes = np.flatnonzero(degrees == degree)
        normal_activation = float(np.mean(normal_pred[:, nodes]))
        normal_mean_prob = float(np.mean(normal_prob[:, nodes]))

        attack_nonattacker_mask_for_degree = np.zeros_like(attack_true, dtype=bool)
        attack_nonattacker_mask_for_degree[:, nodes] = True
        attack_nonattacker_mask_for_degree &= attack_true == 0

        degree_rows.append(
            {
                "degree": int(degree),
                "router_count": int(len(nodes)),
                "routers": "-".join(str(int(node)) for node in nodes),
                "normal_node_activation_rate": normal_activation,
                "normal_node_mean_probability": normal_mean_prob,
                "attack_nonattacker_activation_rate": float(
                    np.mean(attack_pred[attack_nonattacker_mask_for_degree])
                ),
                "attack_nonattacker_mean_probability": float(
                    np.mean(attack_prob[attack_nonattacker_mask_for_degree])
                ),
            }
        )

    true_attacker_mask = attack_true == 1
    nonattacker_mask = attack_true == 0

    return {
        "threshold": float(threshold),
        "attack_node_metrics": attack_counts,
        "exact_localization": float(np.mean(np.all(attack_true == attack_pred, axis=1))),
        "overprediction_rate": float(np.mean(predicted_count > true_count)),
        "underprediction_rate": float(np.mean(predicted_count < true_count)),
        "empty_prediction_rate": float(np.mean(predicted_count == 0)),
        "mean_true_attacker_probability": float(np.mean(attack_prob[true_attacker_mask])),
        "mean_nonattacker_probability_on_attack_samples": float(
            np.mean(attack_prob[nonattacker_mask])
        ),
        "adjacent_nonattacker_activation_rate": float(
            np.mean(attack_pred[adjacent_nonattacker_mask])
        ),
        "other_nonattacker_activation_rate": float(
            np.mean(attack_pred[other_nonattacker_mask])
        ),
        "adjacent_nonattacker_mean_probability": float(
            np.mean(attack_prob[adjacent_nonattacker_mask])
        ),
        "other_nonattacker_mean_probability": float(
            np.mean(attack_prob[other_nonattacker_mask])
        ),
        "normal_node_activation_rate": float(np.mean(normal_pred)),
        "normal_node_mean_probability": float(np.mean(normal_prob)),
        "degree_rows": degree_rows,
    }


def per_run_graph_metrics(
    *,
    y_graph: np.ndarray,
    graph_prob: np.ndarray,
    threshold: float,
    run_id: np.ndarray,
) -> dict[str, dict[str, Any]]:
    prediction = (graph_prob >= threshold).astype(np.int64)
    result: dict[str, dict[str, Any]] = {}

    for current_run in np.unique(run_id):
        mask = run_id == current_run
        labels = y_graph[mask]
        if not np.all(labels == labels[0]):
            raise ValueError(f"Mixed graph labels in run {current_run}")
        label = int(labels[0])
        positive_rate = float(np.mean(prediction[mask] == 1))
        result[str(current_run)] = {
            "run_id": str(current_run),
            "label": label,
            "sample_count": int(np.sum(mask)),
            "positive_prediction_rate": positive_rate,
            "fpr": positive_rate if label == 0 else None,
            "recall": positive_rate if label == 1 else None,
            "mean_graph_probability": float(np.mean(graph_prob[mask])),
            "median_graph_probability": float(np.median(graph_prob[mask])),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--validation-dir", required=True, type=Path)
    parser.add_argument("--g0-report", required=True, type=Path)
    parser.add_argument("--g1-source-audit", required=True, type=Path)
    parser.add_argument("--g1-preflight", required=True, type=Path)
    parser.add_argument("--g1-training-integrity", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    validation_dir = args.validation_dir.resolve()
    output_dir = args.output_dir.resolve()

    gate_path = validation_dir / "g1_validation_gate.json"
    a1_pred_path = validation_dir / "a1_validation_predictions.npz"
    g1_pred_path = validation_dir / "g1_validation_predictions.npz"

    required = {
        "data_dir": data_dir,
        "gate": gate_path,
        "a1_predictions": a1_pred_path,
        "g1_predictions": g1_pred_path,
        "g0_report": args.g0_report.resolve(),
        "g1_source_audit": args.g1_source_audit.resolve(),
        "g1_preflight": args.g1_preflight.resolve(),
        "g1_training_integrity": args.g1_training_integrity.resolve(),
        "edge_index": data_dir / "edge_index.npy",
    }
    for label, path in required.items():
        if label == "data_dir":
            if not path.is_dir():
                raise SystemExit(f"STOP: missing {label}: {path}")
        elif not path.is_file():
            raise SystemExit(f"STOP: missing {label}: {path}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"STOP: output directory already non-empty: {output_dir}")

    gate = load_json(gate_path)
    prerequisite_checks = {
        "gate_stage_is_g1_3": gate.get("stage") == "G1.3",
        "gate_completed": gate.get("status") == "VALIDATION_GATE_COMPLETED",
        "validation_only": gate.get("protocol", {}).get("validation_only") is True,
        "test_inference_false": (
            gate.get("protocol", {}).get("test_inference_performed") is False
        ),
        "test_threshold_selection_false": (
            gate.get("protocol", {}).get("test_threshold_selection_performed") is False
        ),
        "primary_any_pass_true": gate.get("gate", {}).get("primary_any_pass") is True,
        "mandatory_all_pass_false": (
            gate.get("gate", {}).get("mandatory_all_pass") is False
        ),
        "advance_false": (
            gate.get("gate", {}).get("advance_to_frozen_test_transfer") is False
        ),
        "verdict_rejected": (
            gate.get("gate", {}).get("verdict")
            == "VALIDATION_DOES_NOT_JUSTIFY_TEST_TRANSFER"
        ),
    }
    if not all(prerequisite_checks.values()):
        failed = [key for key, value in prerequisite_checks.items() if not value]
        raise SystemExit(f"STOP: closure prerequisites failed: {failed}")

    with np.load(a1_pred_path, allow_pickle=True) as a1_npz:
        a1 = {key: np.asarray(a1_npz[key]) for key in a1_npz.files}
    with np.load(g1_pred_path, allow_pickle=True) as g1_npz:
        g1 = {key: np.asarray(g1_npz[key]) for key in g1_npz.files}

    alignment_checks = {
        "prediction_keys_equal": set(a1) == set(g1),
        "sample_idx_equal": np.array_equal(a1["sample_idx"], g1["sample_idx"]),
        "y_graph_equal": np.array_equal(a1["y_graph"], g1["y_graph"]),
        "y_node_equal": np.array_equal(a1["y_node"], g1["y_node"]),
        "run_id_equal": np.array_equal(a1["run_id"], g1["run_id"]),
        "strength_equal": np.array_equal(a1["strength"], g1["strength"]),
        "validation_rows_42809": len(a1["sample_idx"]) == 42809,
        "node_count_16": a1["y_node"].shape[1] == 16,
    }
    if not all(alignment_checks.values()):
        failed = [key for key, value in alignment_checks.items() if not value]
        raise SystemExit(f"STOP: prediction alignment failed: {failed}")

    y_graph = np.asarray(a1["y_graph"], dtype=np.int64)
    y_node = np.asarray(a1["y_node"], dtype=np.int64)
    run_id = decode_strings(a1["run_id"])

    edge_index = np.load(data_dir / "edge_index.npy").astype(np.int64)
    adjacency = np.zeros((16, 16), dtype=np.int64)
    for source, target in zip(edge_index[0], edge_index[1]):
        source_i = int(source)
        target_i = int(target)
        if source_i != target_i:
            adjacency[source_i, target_i] = 1
    degrees = adjacency.sum(axis=1)

    expected_degrees = np.asarray(
        [2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 2, 3, 3, 2],
        dtype=np.int64,
    )
    topology_checks = {
        "nonself_directed_edges_48": int(adjacency.sum()) == 48,
        "router_degrees_match_4x4_mesh": np.array_equal(degrees, expected_degrees),
    }
    if not all(topology_checks.values()):
        failed = [key for key, value in topology_checks.items() if not value]
        raise SystemExit(f"STOP: topology checks failed: {failed}")

    a1_graph_threshold = float(gate["a1"]["graph_threshold"])
    g1_graph_threshold = float(gate["g1"]["graph_threshold"])
    a1_node_threshold = float(gate["a1"]["node_threshold"])
    g1_node_threshold = float(gate["g1"]["node_threshold"])

    a1_node = node_error_profile(
        y_graph=y_graph,
        y_node=y_node,
        node_prob=np.asarray(a1["node_prob"], dtype=np.float64),
        threshold=a1_node_threshold,
        adjacency=adjacency,
        degrees=degrees,
    )
    g1_node = node_error_profile(
        y_graph=y_graph,
        y_node=y_node,
        node_prob=np.asarray(g1["node_prob"], dtype=np.float64),
        threshold=g1_node_threshold,
        adjacency=adjacency,
        degrees=degrees,
    )

    a1_runs = per_run_graph_metrics(
        y_graph=y_graph,
        graph_prob=np.asarray(a1["graph_prob"], dtype=np.float64),
        threshold=a1_graph_threshold,
        run_id=run_id,
    )
    g1_runs = per_run_graph_metrics(
        y_graph=y_graph,
        graph_prob=np.asarray(g1["graph_prob"], dtype=np.float64),
        threshold=g1_graph_threshold,
        run_id=run_id,
    )

    run_rows = []
    for current_run in sorted(a1_runs):
        a = a1_runs[current_run]
        g = g1_runs[current_run]
        row = {
            "run_id": current_run,
            "label": a["label"],
            "sample_count": a["sample_count"],
            "a1_positive_prediction_rate": a["positive_prediction_rate"],
            "g1_positive_prediction_rate": g["positive_prediction_rate"],
            "delta_positive_prediction_rate": (
                g["positive_prediction_rate"] - a["positive_prediction_rate"]
            ),
            "a1_mean_graph_probability": a["mean_graph_probability"],
            "g1_mean_graph_probability": g["mean_graph_probability"],
            "delta_mean_graph_probability": (
                g["mean_graph_probability"] - a["mean_graph_probability"]
            ),
        }
        run_rows.append(row)

    normal_rows = [row for row in run_rows if row["label"] == 0]
    attack_rows = [row for row in run_rows if row["label"] == 1]

    normal_rows_sorted = sorted(
        normal_rows,
        key=lambda row: row["delta_positive_prediction_rate"],
        reverse=True,
    )
    attack_rows_sorted = sorted(
        attack_rows,
        key=lambda row: row["delta_positive_prediction_rate"],
        reverse=True,
    )

    a1_graph_prob = np.asarray(a1["graph_prob"], dtype=np.float64)
    g1_graph_prob = np.asarray(g1["graph_prob"], dtype=np.float64)

    graph_distribution = {
        "a1_normal": quantiles(a1_graph_prob[y_graph == 0]),
        "a1_attack": quantiles(a1_graph_prob[y_graph == 1]),
        "g1_normal": quantiles(g1_graph_prob[y_graph == 0]),
        "g1_attack": quantiles(g1_graph_prob[y_graph == 1]),
    }

    degree_rows = []
    a1_degree = {row["degree"]: row for row in a1_node["degree_rows"]}
    g1_degree = {row["degree"]: row for row in g1_node["degree_rows"]}
    for degree in sorted(a1_degree):
        a = a1_degree[degree]
        g = g1_degree[degree]
        degree_rows.append(
            {
                "degree": degree,
                "router_count": a["router_count"],
                "routers": a["routers"],
                "a1_normal_node_activation_rate": a["normal_node_activation_rate"],
                "g1_normal_node_activation_rate": g["normal_node_activation_rate"],
                "delta_normal_node_activation_rate": (
                    g["normal_node_activation_rate"]
                    - a["normal_node_activation_rate"]
                ),
                "a1_normal_node_mean_probability": a["normal_node_mean_probability"],
                "g1_normal_node_mean_probability": g["normal_node_mean_probability"],
                "delta_normal_node_mean_probability": (
                    g["normal_node_mean_probability"]
                    - a["normal_node_mean_probability"]
                ),
                "a1_attack_nonattacker_activation_rate": (
                    a["attack_nonattacker_activation_rate"]
                ),
                "g1_attack_nonattacker_activation_rate": (
                    g["attack_nonattacker_activation_rate"]
                ),
                "delta_attack_nonattacker_activation_rate": (
                    g["attack_nonattacker_activation_rate"]
                    - a["attack_nonattacker_activation_rate"]
                ),
            }
        )

    node_deltas = {
        "attacker_recall": (
            g1_node["attack_node_metrics"]["recall"]
            - a1_node["attack_node_metrics"]["recall"]
        ),
        "nonattacker_fpr_on_attack_samples": (
            g1_node["attack_node_metrics"]["fpr"]
            - a1_node["attack_node_metrics"]["fpr"]
        ),
        "exact_localization": (
            g1_node["exact_localization"] - a1_node["exact_localization"]
        ),
        "overprediction_rate": (
            g1_node["overprediction_rate"] - a1_node["overprediction_rate"]
        ),
        "underprediction_rate": (
            g1_node["underprediction_rate"] - a1_node["underprediction_rate"]
        ),
        "empty_prediction_rate": (
            g1_node["empty_prediction_rate"] - a1_node["empty_prediction_rate"]
        ),
        "adjacent_nonattacker_activation_rate": (
            g1_node["adjacent_nonattacker_activation_rate"]
            - a1_node["adjacent_nonattacker_activation_rate"]
        ),
        "other_nonattacker_activation_rate": (
            g1_node["other_nonattacker_activation_rate"]
            - a1_node["other_nonattacker_activation_rate"]
        ),
        "normal_node_activation_rate": (
            g1_node["normal_node_activation_rate"]
            - a1_node["normal_node_activation_rate"]
        ),
    }

    conclusion = {
        "g1_status": "REJECTED_ON_VALIDATION",
        "formal_test_transfer_authorized": False,
        "multiseed_authorized": False,
        "additional_v3_architecture_training_authorized": False,
        "final_v3_model": "Chrono-A1 Conv1D-TemporalGCN",
        "architectural_insight_retained": (
            "Separate self/root and neighbor transforms improved attacker "
            "localization, but raw unnormalized neighbor summation violated "
            "normal-run false-positive safeguards."
        ),
        "next_architecture_work_location": "V4 only",
        "v4_baseline_order": [
            "V4-A1: Conv1D + normalized GCN",
            (
                "V4-G2 only after V4-A1 failure analysis: separate root "
                "transform + normalized/mean neighbor transform"
            ),
        ],
        "default_threshold_test_metrics_policy": (
            "Document as diagnostic trainer output only; do not use for "
            "selection, advancement, or formal claims."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=False)

    per_run_path = output_dir / "g1_per_run_validation_deltas.csv"
    degree_path = output_dir / "g1_router_degree_node_activation.csv"
    write_csv(per_run_path, run_rows)
    write_csv(degree_path, degree_rows)

    report = {
        "stage": "G1.4",
        "status": "TARGETED_POSTMORTEM_COMPLETED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "validation_predictions_only": True,
            "model_inference_performed": False,
            "training_performed": False,
            "test_accessed": False,
            "test_transfer_performed": False,
        },
        "checks": {
            **prerequisite_checks,
            **alignment_checks,
            **topology_checks,
        },
        "thresholds": {
            "a1_graph": a1_graph_threshold,
            "g1_graph": g1_graph_threshold,
            "a1_node": a1_node_threshold,
            "g1_node": g1_node_threshold,
        },
        "graph_probability_distribution": graph_distribution,
        "node_error_profile": {
            "a1": {k: v for k, v in a1_node.items() if k != "degree_rows"},
            "g1": {k: v for k, v in g1_node.items() if k != "degree_rows"},
            "g1_minus_a1": node_deltas,
        },
        "run_level": {
            "normal_runs": len(normal_rows),
            "attack_runs": len(attack_rows),
            "largest_normal_fpr_regressions": normal_rows_sorted[:10],
            "largest_normal_fpr_improvements": list(reversed(normal_rows_sorted[-10:])),
            "largest_attack_recall_improvements": attack_rows_sorted[:10],
            "largest_attack_recall_regressions": list(reversed(attack_rows_sorted[-10:])),
        },
        "router_degree_analysis": degree_rows,
        "conclusion": conclusion,
    }

    report_path = output_dir / "g1_targeted_postmortem.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        "# G1 Targeted Validation-Only Postmortem",
        "",
        "## Formal closure",
        "",
        "- G1 is rejected on validation.",
        "- No formal test transfer is authorized.",
        "- No additional G1 seed or V3 architecture experiment is authorized.",
        "- Chrono-A1 remains the final V3 model.",
        "- The separate-root insight is retained for a later V4-only normalized operator.",
        "",
        "## Node behaviour, G1 minus A1",
        "",
    ]
    for key, value in node_deltas.items():
        md_lines.append(f"- `{key}`: `{value:+.6f}`")
    md_lines.extend(
        [
            "",
            "## Largest benign run regressions",
            "",
        ]
    )
    for row in normal_rows_sorted[:5]:
        md_lines.append(
            f"- `{row['run_id']}`: "
            f"`{row['delta_positive_prediction_rate']:+.6f}`"
        )
    md_lines.extend(
        [
            "",
            "## Next architecture policy",
            "",
            "1. Train V4-A1 with the normalized GCN first.",
            "2. Perform V4-A1 run-level/localization failure analysis.",
            "3. Test V4-G2 only if localization smearing remains.",
            "4. G2 may separate root and neighbour weights, but the neighbour path must use mean or degree normalization.",
            "",
            "Default-threshold V3 test metrics printed by the trainer are retained as diagnostic logs only.",
        ]
    )
    md_path = output_dir / "g1_targeted_postmortem.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    artifact_paths = {
        "g0_report": required["g0_report"],
        "g1_source_audit": required["g1_source_audit"],
        "g1_preflight": required["g1_preflight"],
        "g1_training_integrity": required["g1_training_integrity"],
        "g1_validation_gate": gate_path,
        "a1_validation_predictions": a1_pred_path,
        "g1_validation_predictions": g1_pred_path,
        "per_run_deltas": per_run_path,
        "router_degree_analysis": degree_path,
        "targeted_postmortem_json": report_path,
        "targeted_postmortem_md": md_path,
    }

    hash_rows = []
    for name, path in artifact_paths.items():
        hash_rows.append(
            {
                "artifact": name,
                "path": str(path),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    hashes_path = output_dir / "g1_closure_artifact_hashes.csv"
    write_csv(hashes_path, hash_rows)

    closure = {
        "stage": "G1.4",
        "status": "G1_CLOSED_AND_FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": conclusion,
        "artifact_hash_manifest": str(hashes_path),
        "artifact_hash_manifest_sha256": sha256(hashes_path),
        "next_immediate_track": (
            "V3.1 raw-trace inventory and semantic-mask reconstructability audit"
        ),
        "no_longer_authorized": [
            "G1 formal test transfer",
            "G1 multiseed confirmation",
            "Mean-GraphConv training on V3",
            "Normalized root-separated GraphConv training on V3",
            "GraphSAGE training on V3",
            "Further graph readout/loss/sampler tuning on V3",
        ],
    }
    closure_path = output_dir / "g1_closure_manifest.json"
    closure_path.write_text(
        json.dumps(closure, indent=2) + "\n",
        encoding="utf-8",
    )

    print("G1.4 TARGETED VALIDATION-ONLY POSTMORTEM: PASS")
    print("model_inference_performed=False")
    print("training_performed=False")
    print("test_accessed=False")
    print(f"normal_runs={len(normal_rows)}")
    print(f"attack_runs={len(attack_rows)}")
    print(f"attacker_recall_delta={node_deltas['attacker_recall']:+.6f}")
    print(
        "attack_nonattacker_fpr_delta="
        f"{node_deltas['nonattacker_fpr_on_attack_samples']:+.6f}"
    )
    print(
        "adjacent_nonattacker_activation_delta="
        f"{node_deltas['adjacent_nonattacker_activation_rate']:+.6f}"
    )
    print(
        "normal_node_activation_delta="
        f"{node_deltas['normal_node_activation_rate']:+.6f}"
    )
    print(
        "largest_normal_run_regression="
        f"{normal_rows_sorted[0]['run_id']}:"
        f"{normal_rows_sorted[0]['delta_positive_prediction_rate']:+.6f}"
    )
    print("formal_test_transfer_authorized=False")
    print("multiseed_authorized=False")
    print("additional_v3_architecture_training_authorized=False")
    print("final_v3_model=Chrono-A1 Conv1D-TemporalGCN")
    print("next_architecture_work_location=V4 only")
    print(f"postmortem_json={report_path}")
    print(f"postmortem_markdown={md_path}")
    print(f"artifact_hashes={hashes_path}")
    print(f"closure_manifest={closure_path}")
    print("G1.4 RESULT: G1_CLOSED_AND_FROZEN")


if __name__ == "__main__":
    main()
