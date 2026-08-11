#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

FORBIDDEN = ("test", "sealed")

GRAPH_REQUIRED = (
    "graph_auroc",
    "graph_average_precision",
    "graph_f1",
    "graph_precision",
    "graph_recall",
    "graph_false_positive_rate",
)
MULTITASK_REQUIRED = (
    "graph_auroc",
    "graph_average_precision",
    "count_active_macro_f1",
    "source_average_precision",
    "transit_average_precision",
    "victim_average_precision",
    "path_average_precision",
)
COST_FIELDS = (
    "parameter_count",
    "serialized_weight_bytes",
    "estimated_macs_or_ops",
    "batch1_cpu_latency_ms",
    "batch1_gpu_latency_ms",
)


def _finite_number(x: Any, name: str, allow_none: bool = False) -> Optional[float]:
    if x is None and allow_none:
        return None
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be numeric")
    y = float(x)
    if not math.isfinite(y):
        raise ValueError(f"{name} must be finite")
    return y


def _unit_metric(x: Any, name: str, allow_none: bool = False) -> Optional[float]:
    y = _finite_number(x, name, allow_none)
    if y is None:
        return None
    if not 0.0 <= y <= 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return y


def validate_result(r: Dict[str, Any]) -> Dict[str, Any]:
    required = (
        "experiment_id",
        "model_id",
        "model_family",
        "task_scope",
        "dataset_scope",
        "split",
        "seed",
        "screen_or_confirmation",
        "metrics",
        "provenance",
    )
    missing = [k for k in required if k not in r]
    if missing:
        raise ValueError(f"missing result fields: {missing}")

    split = str(r["split"]).strip().lower()
    if any(tok in split for tok in FORBIDDEN):
        raise ValueError("P4 rejects sealed/test result records")
    if split != "validation":
        raise ValueError("P4 accepts validation result records only")

    if r["dataset_scope"] not in ("V5_P3_TRANCHE_A_DYNAMIC70", "V5_P3_FINAL_AB_DYNAMIC70"):
        raise ValueError("unsupported dataset_scope")

    if int(r["seed"]) not in (107, 117, 127):
        raise ValueError("P4 comparison seeds are frozen to 107/117/127")

    if r["screen_or_confirmation"] not in ("SCREENING", "CONFIRMATION"):
        raise ValueError("screen_or_confirmation must be SCREENING or CONFIRMATION")

    if bool(r.get("threshold_tuning_performed", False)):
        raise ValueError("threshold tuning inside P4 result is forbidden")

    if bool(r.get("decoder_used_for_model_selection", False)):
        raise ValueError("decoder may not be used for model selection in P4")

    metrics = r["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dict")

    scope = r["task_scope"]
    if scope == "graph_detection_only":
        for k in GRAPH_REQUIRED:
            _unit_metric(metrics.get(k), f"metrics.{k}")
        for k in ("count_active_macro_f1", "source_average_precision", "transit_average_precision",
                  "victim_average_precision", "path_average_precision"):
            if metrics.get(k) not in (None, ""):
                raise ValueError(f"graph_detection_only model must leave {k} NA")
    elif scope == "multitask":
        for k in MULTITASK_REQUIRED:
            _unit_metric(metrics.get(k), f"metrics.{k}")
    else:
        raise ValueError("task_scope must be graph_detection_only or multitask")

    for k in COST_FIELDS:
        if k in r and r[k] not in (None, ""):
            _finite_number(r[k], k)

    prov = r["provenance"]
    if not isinstance(prov, dict):
        raise ValueError("provenance must be a dict")
    for k in ("dataset_protocol_sha256", "model_or_script_sha256"):
        if k not in prov:
            raise ValueError(f"provenance.{k} is required")

    out = dict(r)
    return out


def load_results(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text())
    rows = obj if isinstance(obj, list) else obj.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("results JSON must be a non-empty list or {'results': [...]}")
    return [validate_result(r) for r in rows]


def _mean_std(values: Sequence[float]) -> tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    sd = statistics.stdev(vals) if len(vals) >= 2 else None
    return mean, sd


def aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r["dataset_scope"], r["model_id"], r["task_scope"], r["screen_or_confirmation"])
        grouped[key].append(r)

    out: List[Dict[str, Any]] = []
    metric_keys = sorted(set(GRAPH_REQUIRED + MULTITASK_REQUIRED))
    for key in sorted(grouped):
        ds, model, scope, phase = key
        rs = sorted(grouped[key], key=lambda x: int(x["seed"]))
        row: Dict[str, Any] = {
            "dataset_scope": ds,
            "model_id": model,
            "task_scope": scope,
            "screen_or_confirmation": phase,
            "seeds": ",".join(str(int(r["seed"])) for r in rs),
            "n_seeds": len(rs),
        }
        for m in metric_keys:
            vals = [r["metrics"].get(m) for r in rs if r["metrics"].get(m) not in (None, "")]
            mean, sd = _mean_std(vals)
            row[f"{m}_mean"] = mean
            row[f"{m}_sample_std"] = sd
        for c in COST_FIELDS:
            vals = [r.get(c) for r in rs if r.get(c) not in (None, "")]
            mean, sd = _mean_std(vals)
            row[f"{c}_mean"] = mean
            row[f"{c}_sample_std"] = sd
        out.append(row)
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def execute(results_json: Path, output_csv: Path) -> None:
    rows = load_results(results_json)
    write_csv(output_csv, aggregate(rows))


def _synthetic() -> List[Dict[str, Any]]:
    prov = {"dataset_protocol_sha256": "a"*64, "model_or_script_sha256": "b"*64}
    graph = {
        "graph_auroc": 0.88, "graph_average_precision": 0.64,
        "graph_f1": 0.80, "graph_precision": 0.81,
        "graph_recall": 0.79, "graph_false_positive_rate": 0.04,
        "count_active_macro_f1": None, "source_average_precision": None,
        "transit_average_precision": None, "victim_average_precision": None,
        "path_average_precision": None,
    }
    multi1 = {
        "graph_auroc": 0.89, "graph_average_precision": 0.65,
        "graph_f1": 0.81, "graph_precision": 0.82,
        "graph_recall": 0.80, "graph_false_positive_rate": 0.03,
        "count_active_macro_f1": 0.99, "source_average_precision": 0.63,
        "transit_average_precision": 0.62, "victim_average_precision": 0.61,
        "path_average_precision": 0.60,
    }
    multi2 = dict(multi1)
    multi2["victim_average_precision"] = 0.63
    return [
        {
            "experiment_id": "P4_SELFTEST_LR_107", "model_id": "logistic_regression",
            "model_family": "classical", "task_scope": "graph_detection_only",
            "dataset_scope": "V5_P3_TRANCHE_A_DYNAMIC70", "split": "validation",
            "seed": 107, "screen_or_confirmation": "SCREENING", "metrics": graph,
            "parameter_count": 701, "provenance": prov,
            "threshold_tuning_performed": False, "decoder_used_for_model_selection": False,
        },
        {
            "experiment_id": "P4_SELFTEST_GRAPHCONV_107", "model_id": "conv1d_graphconv",
            "model_family": "neural_graph", "task_scope": "multitask",
            "dataset_scope": "V5_P3_TRANCHE_A_DYNAMIC70", "split": "validation",
            "seed": 107, "screen_or_confirmation": "CONFIRMATION", "metrics": multi1,
            "parameter_count": 60553, "provenance": prov,
            "threshold_tuning_performed": False, "decoder_used_for_model_selection": False,
        },
        {
            "experiment_id": "P4_SELFTEST_GRAPHCONV_117", "model_id": "conv1d_graphconv",
            "model_family": "neural_graph", "task_scope": "multitask",
            "dataset_scope": "V5_P3_TRANCHE_A_DYNAMIC70", "split": "validation",
            "seed": 117, "screen_or_confirmation": "CONFIRMATION", "metrics": multi2,
            "parameter_count": 60553, "provenance": prov,
            "threshold_tuning_performed": False, "decoder_used_for_model_selection": False,
        },
    ]


def selftest(tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    inp = tmp / "results.json"
    inp.write_text(json.dumps(_synthetic(), indent=2) + "\n")
    out = tmp / "aggregate.csv"
    execute(inp, out)

    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    g = next(r for r in rows if r["model_id"] == "conv1d_graphconv")
    assert g["n_seeds"] == "2"
    assert abs(float(g["victim_average_precision_mean"]) - 0.62) < 1e-12

    bad = _synthetic()[0]
    bad = dict(bad)
    bad["split"] = "sealed_test"
    denied = False
    try:
        validate_result(bad)
    except ValueError:
        denied = True
    assert denied

    bad2 = _synthetic()[0]
    bad2 = json.loads(json.dumps(bad2))
    bad2["metrics"]["victim_average_precision"] = 0.5
    denied2 = False
    try:
        validate_result(bad2)
    except ValueError:
        denied2 = True
    assert denied2

    print("P4_SELFTEST_PASS")
    print("aggregate_rows=2")
    print("graph_only_role_metrics_remain_na=true")
    print("confirmation_mean_std_path_exercised=true")
    print("sealed_test_alias_rejected=true")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--results-json", type=Path, required=True)
    r.add_argument("--output-csv", type=Path, required=True)

    s = sub.add_parser("selftest")
    s.add_argument("--tmp-dir", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "selftest":
        selftest(args.tmp_dir)
        return
    execute(args.results_json, args.output_csv)
    print("V5_P3_POSTTRAIN_P4_MODEL_COMPARISON_AGGREGATION_COMPLETE")
    print("sealed_test_access=false")


if __name__ == "__main__":
    main()
