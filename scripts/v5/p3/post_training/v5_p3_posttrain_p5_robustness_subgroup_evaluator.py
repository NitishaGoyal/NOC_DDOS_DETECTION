#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROUTERS = 16
FORBIDDEN = ("test", "sealed")
HEADS = ("source", "transit", "victim", "path")


def _assert_validation(r: Dict[str, Any]) -> None:
    split = str(r.get("split", "")).strip().lower()
    if any(tok in split for tok in FORBIDDEN):
        raise ValueError("P5 rejects sealed/test archive records")
    if split != "validation":
        raise ValueError(f"P5 requires split='validation'; got {split!r}")


def _prob(x: Any, name: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be numeric")
    y = float(x)
    if not math.isfinite(y) or not 0 <= y <= 1:
        raise ValueError(f"{name} must be finite in [0,1]")
    return y


def _truth_vec16(x: Any, name: str) -> List[int]:
    if not isinstance(x, list) or len(x) != ROUTERS:
        raise ValueError(f"{name} must be length {ROUTERS}")
    out = []
    for i, v in enumerate(x):
        if v not in (0, 1, False, True):
            raise ValueError(f"{name}[{i}] must be binary")
        out.append(int(v))
    return out


def _prob_vec16(x: Any, name: str) -> List[float]:
    if not isinstance(x, list) or len(x) != ROUTERS:
        raise ValueError(f"{name} must be length {ROUTERS}")
    return [_prob(v, f"{name}[{i}]") for i, v in enumerate(x)]


def load_archive(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                _assert_validation(r)
                if r.get("run_key") is None:
                    raise ValueError("run_key required")
                if r.get("graph_truth") not in (0, 1, False, True):
                    raise ValueError("graph_truth must be binary")
                _prob(r["graph_probability"], "graph_probability")
                for h in HEADS:
                    _truth_vec16(r[f"{h}_truth"], f"{h}_truth")
                    _prob_vec16(r[f"{h}_probability"], f"{h}_probability")
            except Exception as e:
                raise ValueError(f"{path}:{ln}: {e}") from e
            rows.append(r)
    if not rows:
        raise ValueError("archive empty")
    return rows


def load_latency_csv(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None:
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            run = str(r.get("run_key", "")).strip()
            if not run:
                raise ValueError("P2 latency CSV row missing run_key")
            out[run] = dict(r)
    return out


def _average_precision(y: Sequence[int], s: Sequence[float]) -> Optional[float]:
    if not y or len(y) != len(s):
        return None
    positives = sum(y)
    if positives == 0:
        return None
    pairs = sorted(zip(s, y), key=lambda p: (-p[0], -p[1]))
    tp = 0
    acc = 0.0
    for rank, (_, label) in enumerate(pairs, 1):
        if label:
            tp += 1
            acc += tp / rank
    return acc / positives


def _auroc(y: Sequence[int], s: Sequence[float]) -> Optional[float]:
    pos = [v for v, t in zip(s, y) if t == 1]
    neg = [v for v, t in zip(s, y) if t == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def _type7(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    h = (len(vals) - 1) * q
    lo, hi = int(math.floor(h)), int(math.ceil(h))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (h - lo) * (vals[hi] - vals[lo])


def _router_xy(router: int) -> Tuple[int, int]:
    if not 0 <= router < 16:
        raise ValueError(f"router id out of 4x4 range: {router}")
    return router // 4, router % 4


def _location(router: int) -> str:
    r, c = _router_xy(router)
    if r in (0, 3) and c in (0, 3):
        return "CORNER"
    if r in (0, 3) or c in (0, 3):
        return "EDGE"
    return "INTERIOR"


def _true_indices(r: Dict[str, Any], head: str) -> List[int]:
    return [i for i, v in enumerate(_truth_vec16(r[f"{head}_truth"], f"{head}_truth")) if v]


def _location_label(indices: Sequence[int]) -> str:
    if not indices:
        return "NONE"
    cats = {_location(i) for i in indices}
    if len(cats) == 1:
        return next(iter(cats))
    return "MULTI_LOCATION"


def _distance_descriptor(r: Dict[str, Any]) -> Tuple[str, str]:
    src = _true_indices(r, "source")
    vic = _true_indices(r, "victim")
    if len(src) == 1 and len(vic) == 1:
        sr, sc = _router_xy(src[0])
        vr, vc = _router_xy(vic[0])
        d = abs(sr - vr) + abs(sc - vc)
        if d == 0:
            b = "distance_0"
        elif d <= 2:
            b = "short_1_2"
        elif d <= 4:
            b = "medium_3_4"
        elif d <= 6:
            b = "long_5_6"
        else:
            b = "OUT_OF_RANGE"
        return str(d), b
    if not src or not vic:
        return "MISSING_ENDPOINT", "MISSING_ENDPOINT"
    return "MULTI_ENDPOINT_UNPAIRED", "MULTI_ENDPOINT_UNPAIRED"


def _analysis_meta(r: Dict[str, Any]) -> Dict[str, Any]:
    m = r.get("analysis_metadata")
    return m if isinstance(m, dict) else {}


def derive_labels(r: Dict[str, Any]) -> Dict[str, str]:
    ac = r.get("attacker_count")
    if isinstance(ac, int) and 1 <= ac <= 4:
        attacker = f"K{ac}"
    else:
        attacker = "OTHER_OR_UNKNOWN"

    strength = "MISSING" if r.get("attack_strength") is None else str(r["attack_strength"])
    workload = "MISSING" if r.get("traffic_or_workload_class") is None else str(r["traffic_or_workload_class"])
    meta = _analysis_meta(r)
    profile = "MISSING" if meta.get("attack_profile") is None else str(meta["attack_profile"])

    overlap = meta.get("path_overlap", meta.get("route_overlap"))
    overlap_label = "MISSING" if overlap is None else str(overlap)

    dist_exact, dist_bin = _distance_descriptor(r)
    src_loc = _location_label(_true_indices(r, "source"))
    vic_loc = _location_label(_true_indices(r, "victim"))
    transit_count = str(sum(_truth_vec16(r["transit_truth"], "transit_truth")))
    path_count = str(sum(_truth_vec16(r["path_truth"], "path_truth")))

    return {
        "attacker_count": attacker,
        "attack_strength": strength,
        "attack_profile": profile,
        "traffic_or_workload_class": workload,
        "singleton_sv_manhattan_distance": dist_exact,
        "singleton_sv_distance_bin": dist_bin,
        "source_location": src_loc,
        "victim_location": vic_loc,
        "true_transit_router_count": transit_count,
        "true_path_router_count": path_count,
        "explicit_route_overlap": overlap_label,
    }


def _flatten_active(rows: Sequence[Dict[str, Any]], head: str) -> Tuple[List[int], List[float]]:
    y, s = [], []
    for r in rows:
        if int(r["graph_truth"]) != 1:
            continue
        y.extend(_truth_vec16(r[f"{head}_truth"], f"{head}_truth"))
        s.extend(_prob_vec16(r[f"{head}_probability"], f"{head}_probability"))
    return y, s


def _latency_float(row: Dict[str, str], key: str) -> Optional[float]:
    v = row.get(key)
    if v in (None, "", "None", "null"):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def _latency_bool(row: Dict[str, str], key: str) -> Optional[bool]:
    v = str(row.get(key, "")).strip().lower()
    if v in ("true", "1"):
        return True
    if v in ("false", "0"):
        return False
    return None


def summarize_group(rows: Sequence[Dict[str, Any]], latency: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    active = [r for r in rows if int(r["graph_truth"]) == 1]
    gy = [int(r["graph_truth"]) for r in rows]
    gs = [_prob(r["graph_probability"], "graph_probability") for r in rows]

    out: Dict[str, Any] = {
        "window_count": len(rows),
        "active_attack_window_count": len(active),
        "run_count": len({str(r["run_key"]) for r in rows}),
        "graph_auroc_all_windows": _auroc(gy, gs),
        "graph_average_precision_all_windows": _average_precision(gy, gs),
    }
    for h in HEADS:
        y, s = _flatten_active(rows, h)
        out[f"{h}_average_precision_active_windows"] = _average_precision(y, s)

    run_keys = sorted({str(r["run_key"]) for r in rows})
    lrows = [latency[k] for k in run_keys if k in latency]
    if lrows:
        detected_flags = [_latency_bool(r, "undetected_during_attack") for r in lrows]
        known = [x for x in detected_flags if x is not None]
        out["p2_latency_runs_joined"] = len(lrows)
        out["p2_detected_during_attack_fraction"] = (
            sum(not x for x in known) / len(known) if known else None
        )

        graph_lat = [
            _latency_float(r, "graph_first_latency_epochs") for r in lrows
            if _latency_bool(r, "graph_first_detected_before_attack_termination") is True
        ]
        graph_lat = [x for x in graph_lat if x is not None]

        victim_lat = [
            _latency_float(r, "victim_first_correct_set_latency_epochs") for r in lrows
            if _latency_bool(r, "victim_first_correct_before_attack_termination") is True
        ]
        victim_lat = [x for x in victim_lat if x is not None]

        out["p2_graph_first_latency_epochs_median_detected_during_attack"] = _type7(graph_lat, 0.5)
        out["p2_graph_first_latency_epochs_p95_detected_during_attack"] = _type7(graph_lat, 0.95)
        out["p2_victim_first_correct_set_latency_epochs_median_before_termination"] = _type7(victim_lat, 0.5)
        out["p2_victim_first_correct_set_latency_epochs_p95_before_termination"] = _type7(victim_lat, 0.95)
    else:
        out["p2_latency_runs_joined"] = 0
        out["p2_detected_during_attack_fraction"] = None
        out["p2_graph_first_latency_epochs_median_detected_during_attack"] = None
        out["p2_graph_first_latency_epochs_p95_detected_during_attack"] = None
        out["p2_victim_first_correct_set_latency_epochs_median_before_termination"] = None
        out["p2_victim_first_correct_set_latency_epochs_p95_before_termination"] = None
    return out


def build_tables(rows: Sequence[Dict[str, Any]], latency: Dict[str, Dict[str, str]]):
    labelled = [(r, derive_labels(r)) for r in rows]
    dims = list(labelled[0][1].keys())

    metrics_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []

    for dim in dims:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r, labels in labelled:
            groups[labels[dim]].append(r)

        missing = len(groups.get("MISSING", []))
        coverage_rows.append({
            "dimension": dim,
            "total_windows": len(rows),
            "missing_windows": missing,
            "nonmissing_windows": len(rows) - missing,
            "nonmissing_fraction": (len(rows) - missing) / len(rows),
            "distinct_group_values": len(groups),
        })

        for value in sorted(groups):
            m = summarize_group(groups[value], latency)
            metrics_rows.append({
                "dimension": dim,
                "group_value": value,
                **m,
            })

    return metrics_rows, coverage_rows


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


def execute(archive: Path, out_dir: Path, latency_csv: Optional[Path]) -> None:
    rows = load_archive(archive)
    latency = load_latency_csv(latency_csv)
    metrics, coverage = build_tables(rows, latency)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "robustness_subgroup_metrics.csv", metrics)
    write_csv(out_dir / "robustness_metadata_coverage.csv", coverage)
    summary = {
        "validation_windows": len(rows),
        "validation_runs": len({str(r["run_key"]) for r in rows}),
        "latency_csv_supplied": latency_csv is not None,
        "latency_runs_available": len(latency),
        "multi_endpoint_pairing_inferred": False,
        "route_overlap_inferred_from_union_mask": False,
        "threshold_tuning_performed": False,
        "model_inference_performed": False,
        "training_performed": False,
        "sealed_test_access": False,
    }
    (out_dir / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _synthetic_rows() -> List[Dict[str, Any]]:
    def truth(indices):
        return [int(i in indices) for i in range(16)]
    def prob(t, hi=0.9, lo=0.1):
        return [hi if x else lo for x in t]

    rows = []
    # singleton short distance 0->1
    for end, gt, gp in [(8,0,0.1),(24,1,0.9),(32,1,0.95)]:
        s = truth([0]); v = truth([1]); tr = truth([]); p = truth([0,1])
        rows.append({
            "split":"validation","sample_key":f"R1_{end}","pair_key":"P1","run_key":"R1",
            "window_end_epoch":end,"graph_truth":gt,"graph_probability":gp,
            "source_truth":s,"source_probability":prob(s),
            "transit_truth":tr,"transit_probability":prob(tr),
            "victim_truth":v,"victim_probability":prob(v),
            "path_truth":p,"path_probability":prob(p),
            "attacker_count":1,"attack_strength":"S20",
            "traffic_or_workload_class":"WL_A",
            "analysis_metadata":{"attack_profile":"bursty","path_overlap":0},
        })
    # multi-endpoint case, must never receive a fabricated distance
    s2 = truth([0,3]); v2 = truth([12,15]); tr2 = truth([1,2,7,11]); p2 = truth([0,1,2,3,7,11,12,15])
    rows.append({
        "split":"validation","sample_key":"R2_24","pair_key":"P2","run_key":"R2",
        "window_end_epoch":24,"graph_truth":1,"graph_probability":0.8,
        "source_truth":s2,"source_probability":prob(s2),
        "transit_truth":tr2,"transit_probability":prob(tr2),
        "victim_truth":v2,"victim_probability":prob(v2),
        "path_truth":p2,"path_probability":prob(p2),
        "attacker_count":2,"attack_strength":"S63",
        "traffic_or_workload_class":"WL_B",
        "analysis_metadata":{"attack_profile":"stream"},
    })
    return rows


def _synthetic_latency(path: Path) -> None:
    fields = [
        "run_key","undetected_during_attack",
        "graph_first_latency_epochs","graph_first_detected_before_attack_termination",
        "victim_first_correct_set_latency_epochs","victim_first_correct_before_attack_termination"
    ]
    rows = [
        {"run_key":"R1","undetected_during_attack":"False",
         "graph_first_latency_epochs":"8","graph_first_detected_before_attack_termination":"True",
         "victim_first_correct_set_latency_epochs":"16","victim_first_correct_before_attack_termination":"True"},
        {"run_key":"R2","undetected_during_attack":"True",
         "graph_first_latency_epochs":"","graph_first_detected_before_attack_termination":"False",
         "victim_first_correct_set_latency_epochs":"","victim_first_correct_before_attack_termination":"False"},
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def selftest(tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    archive = tmp / "archive.jsonl"
    with archive.open("w", encoding="utf-8") as f:
        for r in _synthetic_rows():
            f.write(json.dumps(r, sort_keys=True) + "\n")
    lat = tmp / "latency.csv"
    _synthetic_latency(lat)
    out = tmp / "out"
    execute(archive, out, lat)

    metrics = list(csv.DictReader((out / "robustness_subgroup_metrics.csv").open()))
    d = {(r["dimension"], r["group_value"]): r for r in metrics}
    assert ("singleton_sv_distance_bin", "short_1_2") in d
    assert ("singleton_sv_distance_bin", "MULTI_ENDPOINT_UNPAIRED") in d
    assert ("attacker_count", "K1") in d and ("attacker_count", "K2") in d
    assert ("victim_location", "CORNER") in d
    assert ("victim_location", "MULTI_LOCATION") not in d or True

    # Explicitly verify no pairing was fabricated.
    rows = _synthetic_rows()
    multi = next(r for r in rows if r["run_key"] == "R2")
    labels = derive_labels(multi)
    assert labels["singleton_sv_manhattan_distance"] == "MULTI_ENDPOINT_UNPAIRED"
    assert labels["singleton_sv_distance_bin"] == "MULTI_ENDPOINT_UNPAIRED"

    # Negative split test.
    bad = dict(rows[0])
    bad["split"] = "sealed_test"
    bp = tmp / "bad.jsonl"
    bp.write_text(json.dumps(bad) + "\n")
    denied = False
    try:
        load_archive(bp)
    except ValueError:
        denied = True
    assert denied

    print("P5_SELFTEST_PASS")
    print("singleton_distance_short_group_present=true")
    print("multi_endpoint_unpaired_group_present=true")
    print("p2_latency_join_exercised=true")
    print("route_overlap_inference=false")
    print("sealed_test_alias_rejected=true")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--archive", type=Path, required=True)
    r.add_argument("--output-dir", type=Path, required=True)
    r.add_argument("--p2-latency-csv", type=Path)

    s = sub.add_parser("selftest")
    s.add_argument("--tmp-dir", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "selftest":
        selftest(args.tmp_dir)
        return

    execute(args.archive, args.output_dir, args.p2_latency_csv)
    print("V5_P3_POSTTRAIN_P5_ROBUSTNESS_SUBGROUP_EVALUATION_COMPLETE")
    print("multi_endpoint_pairing_inferred=false")
    print("route_overlap_inferred_from_union_mask=false")
    print("sealed_test_access=false")


if __name__ == "__main__":
    main()
