#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

FORBIDDEN_SPLIT_TOKENS = ("test", "sealed")
ROUTERS = 16
Q_STABLE = 2


def _assert_validation_record(r: Dict[str, Any]) -> None:
    split = str(r.get("split", "")).strip().lower()
    if any(tok in split for tok in FORBIDDEN_SPLIT_TOKENS):
        raise ValueError("P2 rejects sealed/test archive records")
    if split != "validation":
        raise ValueError(f"P2 requires split='validation'; got {split!r}")


def _prob(x: Any, name: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be numeric")
    x = float(x)
    if not math.isfinite(x) or not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be finite in [0,1]")
    return x


def _binary_vec16(x: Any, name: str) -> List[int]:
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


def _threshold_vec(x: Sequence[float], threshold: float) -> List[int]:
    return [int(v >= threshold) for v in x]


def _exact_set(prob: Any, truth: Any, threshold: float, prefix: str) -> bool:
    p = _prob_vec16(prob, f"{prefix}_probability")
    t = _binary_vec16(truth, f"{prefix}_truth")
    return _threshold_vec(p, threshold) == t


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                _assert_validation_record(r)
            except Exception as e:
                raise ValueError(f"{path}:{line_no}: {e}") from e
            rows.append(r)
    if not rows:
        raise ValueError("prediction archive contains no records")
    return rows


def _require_run_metadata(run_key: str, rs: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    onsets = {r.get("attack_onset_epoch") for r in rs}
    terms = {r.get("attack_termination_epoch") for r in rs}
    if None in onsets or len(onsets) != 1:
        raise ValueError(f"run {run_key}: attack_onset_epoch must be present and constant")
    if None in terms or len(terms) != 1:
        raise ValueError(f"run {run_key}: attack_termination_epoch must be present and constant")
    onset = int(next(iter(onsets)))
    term = int(next(iter(terms)))
    if term < onset:
        raise ValueError(f"run {run_key}: attack termination precedes onset")
    return onset, term


def _first_epoch(flags: Sequence[Tuple[int, bool]]) -> Optional[int]:
    for epoch, flag in flags:
        if flag:
            return epoch
    return None


def _first_stable_epoch(flags: Sequence[Tuple[int, bool]], q: int = Q_STABLE) -> Optional[int]:
    streak = 0
    streak_start = None
    for epoch, flag in flags:
        if flag:
            if streak == 0:
                streak_start = epoch
            streak += 1
            if streak >= q:
                return streak_start
        else:
            streak = 0
            streak_start = None
    return None


def _latency(decision_epoch: Optional[int], onset: int) -> Optional[int]:
    if decision_epoch is None:
        return None
    return int(decision_epoch - onset)


def _type7_quantile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        return None
    if n == 1:
        return vals[0]
    h = (n - 1) * q
    lo = int(math.floor(h))
    hi = int(math.ceil(h))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (h - lo) * (vals[hi] - vals[lo])


def analyze_run(
    run_key: str,
    rs: Sequence[Dict[str, Any]],
    graph_threshold: float,
    source_threshold: float,
    victim_threshold: float,
    path_threshold: float,
    cycles_per_epoch: Optional[int],
) -> Dict[str, Any]:
    onset, term = _require_run_metadata(run_key, rs)
    ordered = sorted(rs, key=lambda r: (int(r["window_end_epoch"]), int(r["window_start_epoch"]), str(r["sample_key"])))

    ends = [int(r["window_end_epoch"]) for r in ordered]
    if len(ends) != len(set((int(r["window_end_epoch"]), str(r["sample_key"])) for r in ordered)):
        pass  # duplicate decision epochs are allowed only when sample keys differ; order remains deterministic

    pre = [r for r in ordered if int(r["window_end_epoch"]) < onset]
    post = [r for r in ordered if int(r["window_end_epoch"]) >= onset]
    during = [r for r in post if int(r["window_end_epoch"]) <= term]

    graph_post = [
        (int(r["window_end_epoch"]), _prob(r["graph_probability"], "graph_probability") >= graph_threshold)
        for r in post
    ]
    graph_during = [(e, f) for e, f in graph_post if e <= term]

    first_graph = _first_epoch(graph_post)
    stable_graph = _first_stable_epoch(graph_post, Q_STABLE)

    src_flags = [
        (int(r["window_end_epoch"]), _exact_set(r["source_probability"], r["source_truth"], source_threshold, "source"))
        for r in post
    ]
    vic_flags = [
        (int(r["window_end_epoch"]), _exact_set(r["victim_probability"], r["victim_truth"], victim_threshold, "victim"))
        for r in post
    ]
    path_flags = [
        (int(r["window_end_epoch"]), _exact_set(r["path_probability"], r["path_truth"], path_threshold, "path"))
        for r in post
    ]

    complete_flags = []
    for r in post:
        epoch = int(r["window_end_epoch"])
        ok = (
            _exact_set(r["source_probability"], r["source_truth"], source_threshold, "source")
            and _exact_set(r["path_probability"], r["path_truth"], path_threshold, "path")
            and _exact_set(r["victim_probability"], r["victim_truth"], victim_threshold, "victim")
        )
        complete_flags.append((epoch, ok))

    source_first = _first_epoch(src_flags)
    victim_first = _first_epoch(vic_flags)
    path_first = _first_epoch(path_flags)
    complete_first = _first_epoch(complete_flags)

    pre_false_epochs = [
        int(r["window_end_epoch"])
        for r in pre
        if _prob(r["graph_probability"], "graph_probability") >= graph_threshold
    ]
    first_during_graph = _first_epoch(graph_during)
    detected_during = first_during_graph is not None

    def before_term(epoch: Optional[int]) -> bool:
        return epoch is not None and epoch <= term

    out = {
        "run_key": run_key,
        "pair_key": str(ordered[0].get("pair_key", "")),
        "attack_onset_epoch": onset,
        "attack_termination_epoch": term,
        "pre_onset_window_count": len(pre),
        "pre_onset_false_alarm_windows": len(pre_false_epochs),
        "any_pre_onset_false_alarm": bool(pre_false_epochs),
        "earliest_pre_onset_false_alarm_epoch": min(pre_false_epochs) if pre_false_epochs else None,
        "graph_first_decision_epoch": first_graph,
        "graph_first_latency_epochs": _latency(first_graph, onset),
        "graph_stable_decision_epoch": stable_graph,
        "graph_stable_latency_epochs": _latency(stable_graph, onset),
        "source_first_correct_set_epoch": source_first,
        "source_first_correct_set_latency_epochs": _latency(source_first, onset),
        "victim_first_correct_set_epoch": victim_first,
        "victim_first_correct_set_latency_epochs": _latency(victim_first, onset),
        "path_first_correct_set_epoch": path_first,
        "path_first_correct_set_latency_epochs": _latency(path_first, onset),
        "complete_source_path_victim_first_correct_epoch": complete_first,
        "complete_source_path_victim_first_correct_latency_epochs": _latency(complete_first, onset),
        "graph_first_detected_before_attack_termination": before_term(first_graph),
        "graph_stable_detected_before_attack_termination": before_term(stable_graph),
        "source_first_correct_before_attack_termination": before_term(source_first),
        "victim_first_correct_before_attack_termination": before_term(victim_first),
        "path_first_correct_before_attack_termination": before_term(path_first),
        "complete_first_correct_before_attack_termination": before_term(complete_first),
        "undetected_during_attack": not detected_during,
        "post_onset_decision_count": len(post),
        "during_attack_decision_count": len(during),
    }

    for k in ("attacker_count", "attack_strength", "traffic_or_workload_class"):
        vals = {json.dumps(r.get(k), sort_keys=True) for r in ordered if r.get(k) is not None}
        out[k] = json.loads(next(iter(vals))) if len(vals) == 1 else None

    if cycles_per_epoch is not None:
        for k in list(out):
            if k.endswith("_latency_epochs") and out[k] is not None:
                out[k.replace("_epochs", "_cycles")] = int(out[k]) * cycles_per_epoch

    return out


def analyze_archive(
    rows: Sequence[Dict[str, Any]],
    graph_threshold: float,
    source_threshold: float,
    victim_threshold: float,
    path_threshold: float,
    cycles_per_epoch: Optional[int],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        run_key = r.get("run_key")
        if run_key is None or not str(run_key).strip():
            raise ValueError("P2 requires run_key on every archive record")
        grouped[str(run_key)].append(r)
    return [
        analyze_run(k, grouped[k], graph_threshold, source_threshold, victim_threshold, path_threshold, cycles_per_epoch)
        for k in sorted(grouped)
    ]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def build_summary(per_run: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(per_run)
    detected = [r for r in per_run if not r["undetected_during_attack"]]
    lat = [
        r["graph_first_latency_epochs"]
        for r in detected
        if r["graph_first_latency_epochs"] is not None
        and r["graph_first_detected_before_attack_termination"]
    ]
    false_alarm_runs = sum(bool(r["any_pre_onset_false_alarm"]) for r in per_run)
    false_alarm_windows = sum(int(r["pre_onset_false_alarm_windows"]) for r in per_run)

    return {
        "eligible_attack_runs": total,
        "detected_during_attack_runs": len(detected),
        "undetected_during_attack_runs": total - len(detected),
        "detected_during_attack_fraction": (len(detected) / total) if total else None,
        "pre_onset_false_alarm_runs": false_alarm_runs,
        "pre_onset_false_alarm_rate": (false_alarm_runs / total) if total else None,
        "false_alarm_windows_total": false_alarm_windows,
        "false_alarm_windows_per_run": (false_alarm_windows / total) if total else None,
        "graph_first_latency_epochs_median_detected_during_attack": _type7_quantile(lat, 0.5),
        "graph_first_latency_epochs_p95_detected_during_attack": _type7_quantile(lat, 0.95),
        "stable_detection_consecutive_decisions_q": Q_STABLE,
        "decision_epoch_semantics": "window_end_epoch",
        "cdf_denominator": "all eligible attack runs",
        "threshold_tuning_performed": False,
        "validation_only": True,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "test_evaluation_performed": False,
        "sealed_test_access": False,
    }


def build_cdf(per_run: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total = len(per_run)
    detected_lat = sorted(
        int(r["graph_first_latency_epochs"])
        for r in per_run
        if r["graph_first_latency_epochs"] is not None
        and r["graph_first_detected_before_attack_termination"]
    )
    out = []
    for x in sorted(set(detected_lat)):
        n = sum(v <= x for v in detected_lat)
        out.append({
            "latency_epochs": x,
            "detected_runs_at_or_below": n,
            "eligible_attack_runs": total,
            "cdf_fraction_all_attack_runs": (n / total) if total else None,
        })
    return out


def execute(
    archive: Path,
    out_dir: Path,
    graph_threshold: float,
    source_threshold: float,
    victim_threshold: float,
    path_threshold: float,
    cycles_per_epoch: Optional[int],
) -> None:
    for n, v in [
        ("graph_threshold", graph_threshold),
        ("source_threshold", source_threshold),
        ("victim_threshold", victim_threshold),
        ("path_threshold", path_threshold),
    ]:
        _prob(v, n)

    rows = load_jsonl(archive)
    per_run = analyze_archive(
        rows, graph_threshold, source_threshold, victim_threshold, path_threshold, cycles_per_epoch
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_run_onset_latency.csv", per_run)
    summary = build_summary(per_run)
    summary["thresholds_consumed"] = {
        "graph": graph_threshold,
        "source": source_threshold,
        "victim": victim_threshold,
        "path": path_threshold,
    }
    summary["cycles_per_epoch"] = cycles_per_epoch
    (out_dir / "onset_latency_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(out_dir / "graph_first_latency_cdf.csv", build_cdf(per_run))


def _synthetic_rows() -> List[Dict[str, Any]]:
    z = [0] * 16
    src = [1] + [0] * 15
    vic = [0] * 15 + [1]
    path = [1, 1] + [0] * 13 + [1]

    def rec(run, end, gp, src_ok, vic_ok, path_ok, onset=16, term=40):
        return {
            "split": "validation",
            "sample_key": f"{run}_{end}",
            "pair_key": f"PAIR_{run}",
            "run_key": run,
            "window_start_epoch": max(0, end - 31),
            "window_end_epoch": end,
            "graph_truth": int(end >= onset and end <= term),
            "graph_probability": gp,
            "count_truth": 1,
            "source_truth": src,
            "source_probability": [0.9 if (src_ok and t) else (0.1 if src_ok else 0.1 if t else 0.9) for t in src],
            "transit_truth": z,
            "transit_probability": [0.1] * 16,
            "victim_truth": vic,
            "victim_probability": [0.9 if (vic_ok and t) else (0.1 if vic_ok else 0.1 if t else 0.9) for t in vic],
            "path_truth": path,
            "path_probability": [0.9 if (path_ok and t) else (0.1 if path_ok else 0.1 if t else 0.9) for t in path],
            "attack_onset_epoch": onset,
            "attack_termination_epoch": term,
            "attacker_count": 1,
            "attack_strength": "SELFTEST",
            "traffic_or_workload_class": "SELFTEST",
        }

    # RUN_A: one pre-onset false alarm at epoch 8; graph positive at 16 and 24,
    # so first latency=0 and stable latency=0 (streak starts at epoch 16).
    a = [
        rec("RUN_A", 8, 0.7, False, False, False),
        rec("RUN_A", 16, 0.8, False, False, False),
        rec("RUN_A", 24, 0.9, True, False, True),
        rec("RUN_A", 32, 0.9, True, True, True),
        rec("RUN_A", 40, 0.9, True, True, True),
    ]
    # RUN_B: never graph-positive during attack.
    b = [
        rec("RUN_B", 8, 0.1, False, False, False),
        rec("RUN_B", 16, 0.2, False, False, False),
        rec("RUN_B", 24, 0.3, False, False, False),
        rec("RUN_B", 32, 0.4, False, False, False),
        rec("RUN_B", 40, 0.49, False, False, False),
    ]
    return a + b


def selftest(tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    archive = tmp / "synthetic_validation_archive.jsonl"
    with archive.open("w", encoding="utf-8") as f:
        for r in _synthetic_rows():
            f.write(json.dumps(r, sort_keys=True) + "\n")

    out = tmp / "out"
    execute(archive, out, 0.5, 0.5, 0.5, 0.5, 1000)
    per = list(csv.DictReader((out / "per_run_onset_latency.csv").open()))
    by = {r["run_key"]: r for r in per}
    assert by["RUN_A"]["graph_first_latency_epochs"] == "0"
    assert by["RUN_A"]["graph_stable_latency_epochs"] == "0"
    assert by["RUN_A"]["victim_first_correct_set_latency_epochs"] == "16"
    assert by["RUN_A"]["complete_source_path_victim_first_correct_latency_epochs"] == "16"
    assert by["RUN_A"]["any_pre_onset_false_alarm"] == "True"
    assert by["RUN_B"]["undetected_during_attack"] == "True"

    summary = json.loads((out / "onset_latency_summary.json").read_text())
    assert summary["eligible_attack_runs"] == 2
    assert summary["detected_during_attack_runs"] == 1
    assert abs(summary["pre_onset_false_alarm_rate"] - 0.5) < 1e-12
    assert summary["graph_first_latency_epochs_median_detected_during_attack"] == 0.0

    # Negative split test.
    bad = dict(_synthetic_rows()[0])
    bad["split"] = "sealed_test"
    bad_archive = tmp / "bad.jsonl"
    bad_archive.write_text(json.dumps(bad) + "\n")
    denied = False
    try:
        load_jsonl(bad_archive)
    except ValueError:
        denied = True
    assert denied

    print("P2_SELFTEST_PASS")
    print("eligible_attack_runs=2")
    print("detected_during_attack_runs=1")
    print("pre_onset_false_alarm_rate=0.5")
    print("stable_detection_q=2")
    print("sealed_test_alias_rejected=true")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--archive", type=Path, required=True)
    r.add_argument("--output-dir", type=Path, required=True)
    r.add_argument("--graph-threshold", type=float, required=True)
    r.add_argument("--source-threshold", type=float, required=True)
    r.add_argument("--victim-threshold", type=float, required=True)
    r.add_argument("--path-threshold", type=float, required=True)
    r.add_argument("--cycles-per-epoch", type=int)

    s = sub.add_parser("selftest")
    s.add_argument("--tmp-dir", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "selftest":
        selftest(args.tmp_dir)
        return

    execute(
        args.archive,
        args.output_dir,
        args.graph_threshold,
        args.source_threshold,
        args.victim_threshold,
        args.path_threshold,
        args.cycles_per_epoch,
    )
    print("V5_P3_POSTTRAIN_P2_ONSET_LATENCY_EVALUATION_COMPLETE")
    print("validation_only=true")
    print("threshold_tuning_performed=false")
    print("sealed_test_access=false")


if __name__ == "__main__":
    main()
