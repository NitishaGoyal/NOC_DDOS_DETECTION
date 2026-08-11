#!/usr/bin/env python3
"""P1-B adapter from P0 scientific prediction fields to P1-A archive records.

This module intentionally does not import any model or dataset. A future
authorized validation evaluator may call `adapt_row()` / `write_adapted_rows()`
after it has produced raw validation predictions.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

FORBIDDEN_SPLIT_TOKENS = ("test", "sealed")
REQUIRED_SCIENTIFIC_FIELDS = (
    "sample_key", "pair_key", "window_start_epoch", "window_end_epoch",
    "graph_truth", "graph_probability", "count_truth",
    "source_truth_16", "source_probability_16",
    "transit_truth_16", "transit_probability_16",
    "victim_truth_16", "victim_probability_16",
    "path_truth_16", "path_probability_16",
)

ALIASES = {
    "source_truth_16": "source_truth",
    "source_probability_16": "source_probability",
    "transit_truth_16": "transit_truth",
    "transit_probability_16": "transit_probability",
    "victim_truth_16": "victim_truth",
    "victim_probability_16": "victim_probability",
    "path_truth_16": "path_truth",
    "path_probability_16": "path_probability",
    "observed_attack_onset_epoch": "attack_onset_epoch",
    "observed_attack_end_epoch": "attack_termination_epoch",
}

PASSTHROUGH_METADATA = (
    "run_key",
    "attacker_count",
    "attack_strength",
    "traffic_or_workload_class",
    "source_routers",
    "victim_routers",
    "route_metadata",
    "model_sha256",
    "checkpoint_sha256",
    "dataset_protocol_sha256",
)


def _assert_validation_split(split: str) -> None:
    s = str(split).strip().lower()
    if any(tok in s for tok in FORBIDDEN_SPLIT_TOKENS):
        raise ValueError("P1-B rejects sealed/test split")
    if s != "validation":
        raise ValueError(f"P1-B is validation-only; got split={split!r}")


def _finite_float(x: Any, name: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be numeric")
    y = float(x)
    if not math.isfinite(y):
        raise ValueError(f"{name} must be finite")
    return y


def _softmax(logits: List[Any]) -> List[float]:
    if not isinstance(logits, list) or len(logits) not in (4, 5):
        raise ValueError("count_logits must be a list of length 4 or 5")
    xs = [_finite_float(x, f"count_logits[{i}]") for i, x in enumerate(logits)]
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    z = sum(ex)
    return [x / z for x in ex]


def _json_safe(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if hasattr(x, "tolist"):
        return _json_safe(x.tolist())
    if hasattr(x, "item"):
        return _json_safe(x.item())
    raise TypeError(f"value is not JSON-safe: {type(x).__name__}")


def adapt_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("raw prediction row must be a dict")
    _assert_validation_split(raw.get("split", "validation"))

    missing = [k for k in REQUIRED_SCIENTIFIC_FIELDS if k not in raw]
    if missing:
        raise ValueError(f"missing P0 scientific fields: {missing}")

    out: Dict[str, Any] = {"split": "validation"}

    direct = (
        "sample_key", "pair_key", "window_start_epoch", "window_end_epoch",
        "graph_truth", "graph_probability", "count_truth",
    )
    for k in direct:
        out[k] = _json_safe(raw[k])

    for src, dst in ALIASES.items():
        if src in raw:
            out[dst] = _json_safe(raw[src])

    if "count_logits" in raw and raw["count_logits"] is not None:
        out["count_logits"] = _json_safe(raw["count_logits"])
        out["count_probability"] = _softmax(_json_safe(raw["count_logits"]))
    elif "count_probability" in raw and raw["count_probability"] is not None:
        out["count_probability"] = _json_safe(raw["count_probability"])

    for k in PASSTHROUGH_METADATA:
        if k in raw:
            out[k] = _json_safe(raw[k])

    # Preserve explicitly-declared extra analysis metadata without interpreting it.
    extras = raw.get("analysis_metadata")
    if extras is not None:
        out["analysis_metadata"] = _json_safe(extras)

    return out


def _load_p1a(module_path: Path):
    spec = importlib.util.spec_from_file_location("v5_p3_posttrain_prediction_archive", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load P1-A module: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_adapted_rows(
    rows: Iterable[Dict[str, Any]],
    output_jsonl: Path,
    p1a_module_path: Path,
):
    p1a = _load_p1a(p1a_module_path)
    return p1a.write_archive((adapt_row(r) for r in rows), output_jsonl)


def _example_row(split: str = "validation") -> Dict[str, Any]:
    return {
        "split": split,
        "sample_key": "P1B_SELFTEST_SAMPLE_000",
        "pair_key": "P1B_SELFTEST_PAIR_000",
        "run_key": "P1B_SELFTEST_ATTACK_RUN_000",
        "window_start_epoch": 8,
        "window_end_epoch": 39,
        "graph_truth": 1,
        "graph_probability": 0.91,
        "count_truth": 2,
        "count_logits": [-3.0, 3.0, 0.0, -1.0],
        "source_truth_16": [1, 0] + [0] * 14,
        "source_probability_16": [0.92, 0.08] + [0.02] * 14,
        "transit_truth_16": [0, 1] + [0] * 14,
        "transit_probability_16": [0.04, 0.83] + [0.03] * 14,
        "victim_truth_16": [0] * 15 + [1],
        "victim_probability_16": [0.02] * 15 + [0.94],
        "path_truth_16": [1, 1] + [0] * 13 + [1],
        "path_probability_16": [0.89, 0.82] + [0.03] * 13 + [0.93],
        "observed_attack_onset_epoch": 16,
        "observed_attack_end_epoch": 200,
        "attacker_count": 2,
        "attack_strength": "SELFTEST",
        "traffic_or_workload_class": "SELFTEST",
        "source_routers": [0],
        "victim_routers": [15],
        "route_metadata": {"routing": "XY", "selftest": True},
        "model_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "dataset_protocol_sha256": "3" * 64,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("adapt-json")
    a.add_argument("input_json", type=Path)
    a.add_argument("output_json", type=Path)

    s = sub.add_parser("selftest")
    s.add_argument("--p1a-module", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)

    args = p.parse_args()

    if args.cmd == "adapt-json":
        raw = json.loads(args.input_json.read_text())
        args.output_json.write_text(
            json.dumps(adapt_row(raw), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("P1B_ADAPT_JSON_COMPLETE")
        return

    manifest = write_adapted_rows(
        [_example_row()],
        args.output,
        args.p1a_module,
    )

    # Mandatory negative test: sealed/test aliases must be denied.
    denied = False
    try:
        adapt_row(_example_row("sealed_test"))
    except ValueError:
        denied = True
    if not denied:
        raise SystemExit("P1-B selftest failed: sealed/test split was not rejected")

    print("P1B_SELFTEST_PASS")
    print(f"records={manifest.record_count}")
    print(f"archive_sha256={manifest.archive_sha256}")
    print("sealed_test_alias_rejected=true")


if __name__ == "__main__":
    main()
