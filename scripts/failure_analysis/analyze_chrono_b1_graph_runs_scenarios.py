#!/usr/bin/env python3
"""
Chrono-B1 Stage B1.13: graph-level run and scenario analysis.

Uses aligned A1/B1 validation and test prediction files plus the frozen
validation-selected graph thresholds.

Produces:
- per-run graph metrics for validation and test;
- grouped metrics by profile, strength, attacker count, active cores and seed;
- normal-run macro/worst FPR;
- attack-run macro/worst recall;
- Strength-20 recall;
- hard-normal-run analysis;
- A1/B1 per-run wins, ties and losses.

No threshold is selected or modified here.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


HARD_NORMAL_RUN = "N-3-7-8-12-Pmixed-R18-V3"


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def text(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if value is None:
        return "NA"
    value = str(value).strip()
    return value if value else "NA"


def canonical_strength(value: Any) -> str:
    raw = text(value).strip('"').strip("'")
    if raw.lower() in {"na", "nan", "none", "null", ""}:
        return "NA"
    try:
        number = float(raw)
        return str(int(number)) if number.is_integer() else f"{number:g}"
    except ValueError:
        return raw


def parse_sequence(value: Any) -> list[Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, (int, np.integer)):
        number = int(value)
        return [] if number < 0 else [number]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number) or number < 0:
            return []
        return [int(number)]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    raw = str(value).strip()
    if not raw or raw.lower() in {"na", "nan", "none", "null", "[]", "()"}:
        return []

    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple, set, np.ndarray)):
            return list(parsed)
        if isinstance(parsed, (int, float)):
            return [parsed]
    except (ValueError, SyntaxError):
        pass

    for separator in ("-", ",", "_", " "):
        if separator in raw:
            return [part.strip() for part in raw.split(separator) if part.strip()]

    return [raw]


def canonical_sequence(value: Any) -> str:
    items = parse_sequence(value)
    normalized: list[str] = []
    for item in items:
        raw = text(item)
        try:
            normalized.append(str(int(float(raw))))
        except ValueError:
            if raw != "NA":
                normalized.append(raw)

    if not normalized:
        return "NA"

    def sort_key(token: str) -> tuple[int, Any]:
        try:
            return (0, int(token))
        except ValueError:
            return (1, token)

    return "-".join(sorted(dict.fromkeys(normalized), key=sort_key))


def attacker_count(value: Any, graph_label: int) -> int:
    if int(graph_label) == 0:
        return 0
    return len(
        [
            item
            for item in parse_sequence(value)
            if text(item).lower() not in {"na", "nan", "none", "null", "-1"}
        ]
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as d:
        return {key: np.asarray(d[key]) for key in d.files}


def graph_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = (np.asarray(y_prob).reshape(-1) >= threshold).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)

    return {
        "sample_count": int(len(y_true)),
        "positive_count": int(np.sum(y_true == 1)),
        "negative_count": int(np.sum(y_true == 0)),
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": safe_div(fp, fp + tn),
        "tnr": safe_div(tn, tn + fp),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_group_values(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    y_graph = np.asarray(data["y_graph"], dtype=np.int64)
    return {
        "profile": np.asarray([text(v).lower() for v in data["profile"]]),
        "strength": np.asarray([canonical_strength(v) for v in data["strength"]]),
        "attacker_count": np.asarray(
            [
                str(attacker_count(v, int(label)))
                for v, label in zip(data["attackers"], y_graph)
            ]
        ),
        "active_cores": np.asarray(
            [canonical_sequence(v) for v in data["active_cores"]]
        ),
        "seed": np.asarray([text(v) for v in data["seed"]]),
        "run_id": np.asarray([text(v) for v in data["run_id"]]),
    }


def grouped_rows(
    *,
    split_name: str,
    model_name: str,
    data: dict[str, np.ndarray],
    threshold: float,
    dimension: str,
    values: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_true = np.asarray(data["y_graph"], dtype=np.int64)
    y_prob = np.asarray(data["graph_prob"], dtype=np.float64)

    for value in sorted(set(values.tolist())):
        mask = values == value
        metrics = graph_metrics(y_true[mask], y_prob[mask], threshold)
        rows.append(
            {
                "split": split_name,
                "model": model_name,
                "threshold": threshold,
                "dimension": dimension,
                "group": value,
                **metrics,
            }
        )
    return rows


def run_rows_for_model(
    *,
    split_name: str,
    model_name: str,
    data: dict[str, np.ndarray],
    threshold: float,
    groups: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = grouped_rows(
        split_name=split_name,
        model_name=model_name,
        data=data,
        threshold=threshold,
        dimension="run_id",
        values=groups["run_id"],
    )

    metadata_by_run: dict[str, dict[str, Any]] = {}
    for i, run in enumerate(groups["run_id"]):
        if run not in metadata_by_run:
            metadata_by_run[run] = {
                "profile": groups["profile"][i],
                "strength": groups["strength"][i],
                "attacker_count": groups["attacker_count"][i],
                "active_cores": groups["active_cores"][i],
                "seed": groups["seed"][i],
            }

    for row in rows:
        row.update(metadata_by_run[row["group"]])
        row["run_id"] = row.pop("group")
        if row["positive_count"] > 0 and row["negative_count"] == 0:
            row["run_class"] = "attack"
        elif row["negative_count"] > 0 and row["positive_count"] == 0:
            row["run_class"] = "normal"
        else:
            row["run_class"] = "mixed"
    return rows


def summarize_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normal_rows = [row for row in rows if row["run_class"] == "normal"]
    attack_rows = [row for row in rows if row["run_class"] == "attack"]

    normal_fprs = [float(row["fpr"]) for row in normal_rows]
    attack_recalls = [float(row["recall"]) for row in attack_rows]

    worst_normal = (
        max(normal_rows, key=lambda row: (row["fpr"], row["run_id"]))
        if normal_rows
        else None
    )
    worst_attack = (
        min(attack_rows, key=lambda row: (row["recall"], row["run_id"]))
        if attack_rows
        else None
    )

    strength20 = [
        row for row in attack_rows if canonical_strength(row["strength"]) == "20"
    ]

    hard = [row for row in normal_rows if row["run_id"] == HARD_NORMAL_RUN]

    return {
        "normal_run_count": len(normal_rows),
        "attack_run_count": len(attack_rows),
        "normal_run_macro_fpr": (
            float(np.mean(normal_fprs)) if normal_fprs else None
        ),
        "worst_normal_run_fpr": (
            float(worst_normal["fpr"]) if worst_normal else None
        ),
        "worst_normal_run_id": (
            worst_normal["run_id"] if worst_normal else None
        ),
        "attack_run_macro_recall": (
            float(np.mean(attack_recalls)) if attack_recalls else None
        ),
        "worst_attack_run_recall": (
            float(worst_attack["recall"]) if worst_attack else None
        ),
        "worst_attack_run_id": (
            worst_attack["run_id"] if worst_attack else None
        ),
        "strength20_attack_run_count": len(strength20),
        "strength20_macro_recall": (
            float(np.mean([row["recall"] for row in strength20]))
            if strength20
            else None
        ),
        "hard_normal_run_present": bool(hard),
        "hard_normal_run_fpr": (
            float(hard[0]["fpr"]) if hard else None
        ),
        "hard_normal_run_sample_count": (
            int(hard[0]["sample_count"]) if hard else 0
        ),
    }


def compare_run_rows(
    a1_rows: list[dict[str, Any]],
    b1_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    a_map = {row["run_id"]: row for row in a1_rows}
    b_map = {row["run_id"]: row for row in b1_rows}

    if set(a_map) != set(b_map):
        raise ValueError("A1/B1 run sets differ.")

    comparisons: list[dict[str, Any]] = []
    normal_wins = normal_ties = normal_losses = 0
    attack_wins = attack_ties = attack_losses = 0
    tolerance = 1e-12

    for run_id in sorted(a_map):
        a = a_map[run_id]
        b = b_map[run_id]

        if a["run_class"] != b["run_class"]:
            raise ValueError(f"Run-class mismatch for {run_id}")

        row = {
            "run_id": run_id,
            "run_class": a["run_class"],
            "profile": a["profile"],
            "strength": a["strength"],
            "attacker_count": a["attacker_count"],
            "active_cores": a["active_cores"],
            "seed": a["seed"],
            "sample_count": a["sample_count"],
            "a1_fpr": a["fpr"],
            "b1_fpr": b["fpr"],
            "delta_fpr_b1_minus_a1": b["fpr"] - a["fpr"],
            "a1_recall": a["recall"],
            "b1_recall": b["recall"],
            "delta_recall_b1_minus_a1": b["recall"] - a["recall"],
            "a1_accuracy": a["accuracy"],
            "b1_accuracy": b["accuracy"],
            "delta_accuracy_b1_minus_a1": b["accuracy"] - a["accuracy"],
        }

        if a["run_class"] == "normal":
            delta = row["delta_fpr_b1_minus_a1"]
            if delta < -tolerance:
                verdict = "B1_win"
                normal_wins += 1
            elif delta > tolerance:
                verdict = "B1_loss"
                normal_losses += 1
            else:
                verdict = "tie"
                normal_ties += 1
        elif a["run_class"] == "attack":
            delta = row["delta_recall_b1_minus_a1"]
            if delta > tolerance:
                verdict = "B1_win"
                attack_wins += 1
            elif delta < -tolerance:
                verdict = "B1_loss"
                attack_losses += 1
            else:
                verdict = "tie"
                attack_ties += 1
        else:
            verdict = "mixed_not_scored"

        row["run_verdict"] = verdict
        comparisons.append(row)

    counts = {
        "normal_runs": {
            "B1_wins": normal_wins,
            "ties": normal_ties,
            "B1_losses": normal_losses,
        },
        "attack_runs": {
            "B1_wins": attack_wins,
            "ties": attack_ties,
            "B1_losses": attack_losses,
        },
    }
    return comparisons, counts


def validate_alignment(a1: dict[str, np.ndarray], b1: dict[str, np.ndarray], split: str) -> None:
    for key in (
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
    ):
        if not np.array_equal(a1[key], b1[key]):
            raise ValueError(f"{split}: A1/B1 mismatch in {key}")

    if not np.all(a1["dataset_split"].astype(str) == split):
        raise ValueError(f"{split}: A1 contains rows outside the split")
    if not np.all(b1["dataset_split"].astype(str) == split):
        raise ValueError(f"{split}: B1 contains rows outside the split")


def analyze_split(
    *,
    split_name: str,
    a1: dict[str, np.ndarray],
    b1: dict[str, np.ndarray],
    a1_threshold: float,
    b1_threshold: float,
    output_dir: Path,
) -> dict[str, Any]:
    validate_alignment(a1, b1, split_name)

    groups = make_group_values(a1)
    all_group_rows: list[dict[str, Any]] = []

    for model_name, data, threshold in (
        ("A1", a1, a1_threshold),
        ("B1", b1, b1_threshold),
    ):
        for dimension in (
            "profile",
            "strength",
            "attacker_count",
            "active_cores",
            "seed",
        ):
            all_group_rows.extend(
                grouped_rows(
                    split_name=split_name,
                    model_name=model_name,
                    data=data,
                    threshold=threshold,
                    dimension=dimension,
                    values=groups[dimension],
                )
            )

    a1_runs = run_rows_for_model(
        split_name=split_name,
        model_name="A1",
        data=a1,
        threshold=a1_threshold,
        groups=groups,
    )
    b1_runs = run_rows_for_model(
        split_name=split_name,
        model_name="B1",
        data=b1,
        threshold=b1_threshold,
        groups=groups,
    )

    run_comparison, run_verdict_counts = compare_run_rows(a1_runs, b1_runs)

    write_csv(output_dir / f"{split_name}_grouped_graph_metrics.csv", all_group_rows)
    write_csv(output_dir / f"{split_name}_a1_per_run_graph_metrics.csv", a1_runs)
    write_csv(output_dir / f"{split_name}_b1_per_run_graph_metrics.csv", b1_runs)
    write_csv(output_dir / f"{split_name}_a1_vs_b1_per_run.csv", run_comparison)

    a_summary = summarize_runs(a1_runs)
    b_summary = summarize_runs(b1_runs)

    deltas = {
        "normal_run_macro_fpr": (
            b_summary["normal_run_macro_fpr"] - a_summary["normal_run_macro_fpr"]
            if a_summary["normal_run_macro_fpr"] is not None
            and b_summary["normal_run_macro_fpr"] is not None
            else None
        ),
        "worst_normal_run_fpr": (
            b_summary["worst_normal_run_fpr"] - a_summary["worst_normal_run_fpr"]
            if a_summary["worst_normal_run_fpr"] is not None
            and b_summary["worst_normal_run_fpr"] is not None
            else None
        ),
        "attack_run_macro_recall": (
            b_summary["attack_run_macro_recall"] - a_summary["attack_run_macro_recall"]
            if a_summary["attack_run_macro_recall"] is not None
            and b_summary["attack_run_macro_recall"] is not None
            else None
        ),
        "worst_attack_run_recall": (
            b_summary["worst_attack_run_recall"] - a_summary["worst_attack_run_recall"]
            if a_summary["worst_attack_run_recall"] is not None
            and b_summary["worst_attack_run_recall"] is not None
            else None
        ),
        "strength20_macro_recall": (
            b_summary["strength20_macro_recall"] - a_summary["strength20_macro_recall"]
            if a_summary["strength20_macro_recall"] is not None
            and b_summary["strength20_macro_recall"] is not None
            else None
        ),
        "hard_normal_run_fpr": (
            b_summary["hard_normal_run_fpr"] - a_summary["hard_normal_run_fpr"]
            if a_summary["hard_normal_run_fpr"] is not None
            and b_summary["hard_normal_run_fpr"] is not None
            else None
        ),
    }

    return {
        "split": split_name,
        "thresholds": {"a1": a1_threshold, "b1": b1_threshold},
        "a1": a_summary,
        "b1": b_summary,
        "deltas_b1_minus_a1": deltas,
        "run_verdict_counts": run_verdict_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1-validation", required=True, type=Path)
    parser.add_argument("--b1-validation", required=True, type=Path)
    parser.add_argument("--a1-test", required=True, type=Path)
    parser.add_argument("--b1-test", required=True, type=Path)
    parser.add_argument("--threshold-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    inputs = (
        args.a1_validation,
        args.b1_validation,
        args.a1_test,
        args.b1_test,
        args.threshold_report,
    )
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"STOP: missing input: {path}")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: output directory exists and is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_report = json.loads(
        args.threshold_report.read_text(encoding="utf-8")
    )
    frozen = threshold_report["frozen_thresholds_for_test_transfer"]

    validation = analyze_split(
        split_name="val",
        a1=load_npz(args.a1_validation),
        b1=load_npz(args.b1_validation),
        a1_threshold=float(frozen["a1_graph"]),
        b1_threshold=float(frozen["b1_graph"]),
        output_dir=output_dir,
    )
    test = analyze_split(
        split_name="test",
        a1=load_npz(args.a1_test),
        b1=load_npz(args.b1_test),
        a1_threshold=float(frozen["a1_graph"]),
        b1_threshold=float(frozen["b1_graph"]),
        output_dir=output_dir,
    )

    report = {
        "stage": "B1.13",
        "threshold_protocol": (
            "Model-specific graph thresholds selected on validation in B1.11 "
            "and transferred unchanged to test."
        ),
        "hard_normal_run": HARD_NORMAL_RUN,
        "validation": validation,
        "test": test,
    }

    report_path = output_dir / "graph_run_scenario_analysis.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("B1.13 GRAPH RUN/SCENARIO ANALYSIS: PASS")

    for split_name, result in (("VAL", validation), ("TEST", test)):
        print(f"\n{split_name}")
        for model_name in ("a1", "b1"):
            summary = result[model_name]
            print(f"{model_name.upper()}")
            print(
                "  normal_run_macro_fpr="
                f"{summary['normal_run_macro_fpr']:.6f}"
            )
            print(
                "  worst_normal_run_fpr="
                f"{summary['worst_normal_run_fpr']:.6f}"
            )
            print(
                "  worst_normal_run_id="
                f"{summary['worst_normal_run_id']}"
            )
            print(
                "  attack_run_macro_recall="
                f"{summary['attack_run_macro_recall']:.6f}"
            )
            print(
                "  worst_attack_run_recall="
                f"{summary['worst_attack_run_recall']:.6f}"
            )
            print(
                "  worst_attack_run_id="
                f"{summary['worst_attack_run_id']}"
            )
            print(
                "  strength20_macro_recall="
                f"{summary['strength20_macro_recall']:.6f}"
            )
            print(
                "  hard_normal_run_present="
                f"{summary['hard_normal_run_present']}"
            )
            if summary["hard_normal_run_present"]:
                print(
                    "  hard_normal_run_fpr="
                    f"{summary['hard_normal_run_fpr']:.6f}"
                )

        print("DELTAS B1 - A1")
        for key, value in result["deltas_b1_minus_a1"].items():
            if value is None:
                print(f"  {key}=NA")
            else:
                print(f"  {key}={value:+.6f}")

        print("RUN VERDICTS")
        print(
            "  normal="
            f"{result['run_verdict_counts']['normal_runs']}"
        )
        print(
            "  attack="
            f"{result['run_verdict_counts']['attack_runs']}"
        )

    print(f"\noutput={report_path}")


if __name__ == "__main__":
    main()
