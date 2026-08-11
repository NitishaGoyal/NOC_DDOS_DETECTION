#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROUTERS = 16
FORBIDDEN_SPLIT_TOKENS = ("test", "sealed")
HEADS = ("source", "transit", "victim", "path")


def _assert_validation(r: Dict[str, Any]) -> None:
    split = str(r.get("split", "")).strip().lower()
    if any(tok in split for tok in FORBIDDEN_SPLIT_TOKENS):
        raise ValueError("P3 rejects sealed/test archive records")
    if split != "validation":
        raise ValueError(f"P3 requires split='validation'; got {split!r}")


def _prob(x: Any, name: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be numeric")
    y = float(x)
    if not math.isfinite(y) or not 0 <= y <= 1:
        raise ValueError(f"{name} must be finite in [0,1]")
    return y


def _binary(x: Any, name: str) -> int:
    if x not in (0, 1, False, True):
        raise ValueError(f"{name} must be binary")
    return int(x)


def _prob_vec16(x: Any, name: str) -> List[float]:
    if not isinstance(x, list) or len(x) != ROUTERS:
        raise ValueError(f"{name} must be length {ROUTERS}")
    return [_prob(v, f"{name}[{i}]") for i, v in enumerate(x)]


def _truth_vec16(x: Any, name: str) -> List[int]:
    if not isinstance(x, list) or len(x) != ROUTERS:
        raise ValueError(f"{name} must be length {ROUTERS}")
    return [_binary(v, f"{name}[{i}]") for i, v in enumerate(x)]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                _assert_validation(r)
                _binary(r["graph_truth"], "graph_truth")
                for h in HEADS:
                    _truth_vec16(r[f"{h}_truth"], f"{h}_truth")
                    _prob_vec16(r[f"{h}_probability"], f"{h}_probability")
            except Exception as e:
                raise ValueError(f"{path}:{line_no}: {e}") from e
            rows.append(r)
    if not rows:
        raise ValueError("prediction archive contains no records")
    return rows


def _average_precision(y: Sequence[int], s: Sequence[float]) -> Optional[float]:
    if len(y) != len(s) or not y:
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
    if len(y) != len(s) or not y:
        return None
    pos = [v for v, t in zip(s, y) if t == 1]
    neg = [v for v, t in zip(s, y) if t == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _prf(y: Sequence[int], s: Sequence[float], threshold: float) -> Dict[str, Optional[float]]:
    pred = [int(v >= threshold) for v in s]
    tp = sum(a == 1 and b == 1 for a, b in zip(y, pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, pred))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None or precision + recall == 0:
        f1 = 0.0 if (precision == 0 or recall == 0) else None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _exact(prob: Sequence[float], truth: Sequence[int], threshold: float) -> bool:
    return [int(v >= threshold) for v in prob] == list(truth)


def _flatten(rows: Sequence[Dict[str, Any]], head: str) -> Tuple[List[int], List[float]]:
    y: List[int] = []
    s: List[float] = []
    for r in rows:
        y.extend(_truth_vec16(r[f"{head}_truth"], f"{head}_truth"))
        s.extend(_prob_vec16(r[f"{head}_probability"], f"{head}_probability"))
    return y, s


def _victim_ranking(rows: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    eligible = 0
    top1_correct = 0
    recall_sums = {1: 0.0, 2: 0.0, 3: 0.0}
    rr_sum = 0.0

    for r in rows:
        truth = _truth_vec16(r["victim_truth"], "victim_truth")
        true_idx = {i for i, t in enumerate(truth) if t}
        if not true_idx:
            continue
        eligible += 1
        probs = _prob_vec16(r["victim_probability"], "victim_probability")
        order = sorted(range(ROUTERS), key=lambda i: (-probs[i], i))
        if order[0] in true_idx:
            top1_correct += 1
        for k in (1, 2, 3):
            hit = len(true_idx.intersection(order[:k]))
            recall_sums[k] += hit / len(true_idx)
        first_rank = min(order.index(i) + 1 for i in true_idx)
        rr_sum += 1.0 / first_rank

    return {
        "eligible_victim_windows": eligible,
        "top1_victim_accuracy": top1_correct / eligible if eligible else None,
        "victim_recall_at_1": recall_sums[1] / eligible if eligible else None,
        "victim_recall_at_2": recall_sums[2] / eligible if eligible else None,
        "victim_recall_at_3": recall_sums[3] / eligible if eligible else None,
        "victim_mean_reciprocal_rank": rr_sum / eligible if eligible else None,
    }


def _cross_head_confusion(
    active: Sequence[Dict[str, Any]],
    source_threshold: float,
    transit_threshold: float,
    victim_threshold: float,
) -> List[Dict[str, Any]]:
    specs = [
        ("true_victim", "transit", "victim", transit_threshold),
        ("true_victim", "source", "victim", source_threshold),
        ("true_transit", "victim", "transit", victim_threshold),
        ("true_source", "victim", "source", victim_threshold),
    ]
    out = []
    for population, predicted_head, truth_head, th in specs:
        vals = []
        positives = 0
        for r in active:
            truth = _truth_vec16(r[f"{truth_head}_truth"], f"{truth_head}_truth")
            prob = _prob_vec16(r[f"{predicted_head}_probability"], f"{predicted_head}_probability")
            for t, p in zip(truth, prob):
                if t:
                    vals.append(p)
                    positives += int(p >= th)
        out.append({
            "population": population,
            "cross_head": predicted_head,
            "router_entries": len(vals),
            "cross_head_positive_rate": positives / len(vals) if vals else None,
            "mean_cross_head_probability": sum(vals) / len(vals) if vals else None,
            "threshold": th,
        })
    return out


def evaluate(
    rows: Sequence[Dict[str, Any]],
    thresholds: Dict[str, float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    for h in HEADS:
        _prob(thresholds[h], f"{h}_threshold")

    active = [r for r in rows if int(r["graph_truth"]) == 1]
    control = [r for r in rows if int(r["graph_truth"]) == 0]
    if not active:
        raise ValueError("P3 requires at least one active attack window")

    summary: Dict[str, Any] = {
        "validation_windows_total": len(rows),
        "active_attack_windows": len(active),
        "control_windows": len(control),
        "thresholds_consumed": thresholds,
        "threshold_tuning_performed": False,
        "decoder_applied": False,
        "validation_only": True,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "test_evaluation_performed": False,
        "sealed_test_access": False,
    }

    per_window: List[Dict[str, Any]] = []
    joint_sv = joint_spv = joint_all = 0
    exact_counts = {h: 0 for h in HEADS}

    for r in active:
        exacts = {}
        for h in HEADS:
            truth = _truth_vec16(r[f"{h}_truth"], f"{h}_truth")
            prob = _prob_vec16(r[f"{h}_probability"], f"{h}_probability")
            exacts[h] = _exact(prob, truth, thresholds[h])
            exact_counts[h] += int(exacts[h])
        joint_sv += int(exacts["source"] and exacts["victim"])
        joint_spv += int(exacts["source"] and exacts["path"] and exacts["victim"])
        joint_all += int(all(exacts.values()))
        per_window.append({
            "sample_key": r.get("sample_key"),
            "pair_key": r.get("pair_key"),
            "run_key": r.get("run_key"),
            "window_end_epoch": r.get("window_end_epoch"),
            "source_exact": exacts["source"],
            "transit_exact": exacts["transit"],
            "victim_exact": exacts["victim"],
            "path_exact": exacts["path"],
            "joint_source_victim_exact": exacts["source"] and exacts["victim"],
            "complete_source_path_victim_exact": exacts["source"] and exacts["path"] and exacts["victim"],
            "all_four_role_sets_exact": all(exacts.values()),
        })

    for h in HEADS:
        ya, sa = _flatten(active, h)
        summary[f"{h}_average_precision_active_windows"] = _average_precision(ya, sa)
        summary[f"{h}_auroc_active_windows"] = _auroc(ya, sa)
        prf = _prf(ya, sa, thresholds[h])
        summary[f"{h}_precision_active_windows"] = prf["precision"]
        summary[f"{h}_recall_active_windows"] = prf["recall"]
        summary[f"{h}_f1_active_windows"] = prf["f1"]
        summary[f"{h}_exact_set_accuracy_active_windows"] = exact_counts[h] / len(active)

        yall, sall = _flatten(rows, h)
        summary[f"{h}_average_precision_all_windows"] = _average_precision(yall, sall)
        summary[f"{h}_auroc_all_windows"] = _auroc(yall, sall)

    summary["joint_source_victim_exact_accuracy_active_windows"] = joint_sv / len(active)
    summary["complete_source_path_victim_exact_accuracy_active_windows"] = joint_spv / len(active)
    summary["all_four_role_sets_exact_accuracy_active_windows"] = joint_all / len(active)

    summary.update(_victim_ranking(active))

    # Explicit control-window victim false-alarm metrics.
    if control:
        any_alarm = 0
        router_fp = 0
        router_total = 0
        for r in control:
            truth = _truth_vec16(r["victim_truth"], "victim_truth")
            prob = _prob_vec16(r["victim_probability"], "victim_probability")
            pred = [int(v >= thresholds["victim"]) for v in prob]
            # Control truth should normally be all zero, but compute FPs against truth rather than assume.
            fps = [p == 1 and t == 0 for p, t in zip(pred, truth)]
            any_alarm += int(any(fps))
            router_fp += sum(fps)
            router_total += sum(t == 0 for t in truth)
        summary["control_any_victim_alarm_rate"] = any_alarm / len(control)
        summary["control_victim_router_false_positive_rate"] = router_fp / router_total if router_total else None
    else:
        summary["control_any_victim_alarm_rate"] = None
        summary["control_victim_router_false_positive_rate"] = None

    confusion = _cross_head_confusion(
        active,
        thresholds["source"],
        thresholds["transit"],
        thresholds["victim"],
    )
    return summary, per_window, confusion


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def execute(archive: Path, output_dir: Path, thresholds: Dict[str, float]) -> None:
    rows = load_jsonl(archive)
    summary, per_window, confusion = evaluate(rows, thresholds)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "victim_role_path_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "per_window_role_exactness.csv", per_window)
    write_csv(output_dir / "cross_head_role_confusion.csv", confusion)


def _synthetic_rows() -> List[Dict[str, Any]]:
    src = [1] + [0] * 15
    tr = [0, 1] + [0] * 14
    vic = [0] * 15 + [1]
    path = [1, 1] + [0] * 13 + [1]
    z = [0] * 16

    def probs_for_truth(t, correct=True):
        if correct:
            return [0.9 if x else 0.1 for x in t]
        return [0.1 if x else 0.9 for x in t]

    rows = []
    # Active window, fully correct.
    rows.append({
        "split": "validation", "sample_key": "A1", "pair_key": "P1", "run_key": "R1",
        "window_end_epoch": 24, "graph_truth": 1, "graph_probability": 0.9,
        "source_truth": src, "source_probability": probs_for_truth(src, True),
        "transit_truth": tr, "transit_probability": probs_for_truth(tr, True),
        "victim_truth": vic, "victim_probability": probs_for_truth(vic, True),
        "path_truth": path, "path_probability": probs_for_truth(path, True),
    })
    # Active window: victim still ranks true victim first, but thresholded victim set wrong via one extra FP.
    vp = probs_for_truth(vic, True)
    vp[2] = 0.8
    tp = probs_for_truth(tr, True)
    tp[15] = 0.7  # cross-head transit activation on true victim
    rows.append({
        "split": "validation", "sample_key": "A2", "pair_key": "P1", "run_key": "R1",
        "window_end_epoch": 32, "graph_truth": 1, "graph_probability": 0.9,
        "source_truth": src, "source_probability": probs_for_truth(src, True),
        "transit_truth": tr, "transit_probability": tp,
        "victim_truth": vic, "victim_probability": vp,
        "path_truth": path, "path_probability": probs_for_truth(path, True),
    })
    # Control window with one victim false positive.
    cp = [0.1] * 16
    cp[4] = 0.8
    rows.append({
        "split": "validation", "sample_key": "C1", "pair_key": "P2", "run_key": "RC",
        "window_end_epoch": 8, "graph_truth": 0, "graph_probability": 0.1,
        "source_truth": z, "source_probability": [0.1] * 16,
        "transit_truth": z, "transit_probability": [0.1] * 16,
        "victim_truth": z, "victim_probability": cp,
        "path_truth": z, "path_probability": [0.1] * 16,
    })
    return rows


def selftest(tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    archive = tmp / "synthetic_validation_archive.jsonl"
    with archive.open("w", encoding="utf-8") as f:
        for r in _synthetic_rows():
            f.write(json.dumps(r, sort_keys=True) + "\n")

    out = tmp / "out"
    execute(archive, out, {h: 0.5 for h in HEADS})
    s = json.loads((out / "victim_role_path_summary.json").read_text())

    assert s["active_attack_windows"] == 2
    assert s["control_windows"] == 1
    assert abs(s["victim_exact_set_accuracy_active_windows"] - 0.5) < 1e-12
    assert abs(s["top1_victim_accuracy"] - 1.0) < 1e-12
    assert abs(s["victim_recall_at_1"] - 1.0) < 1e-12
    assert abs(s["control_any_victim_alarm_rate"] - 1.0) < 1e-12
    assert abs(s["joint_source_victim_exact_accuracy_active_windows"] - 0.5) < 1e-12

    # Negative split test.
    bad = dict(_synthetic_rows()[0])
    bad["split"] = "sealed_test"
    badp = tmp / "bad.jsonl"
    badp.write_text(json.dumps(bad) + "\n")
    denied = False
    try:
        load_jsonl(badp)
    except ValueError:
        denied = True
    assert denied

    print("P3_SELFTEST_PASS")
    print("active_attack_windows=2")
    print("control_windows=1")
    print("victim_exact_set_accuracy=0.5")
    print("top1_victim_accuracy=1.0")
    print("control_any_victim_alarm_rate=1.0")
    print("sealed_test_alias_rejected=true")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--archive", type=Path, required=True)
    r.add_argument("--output-dir", type=Path, required=True)
    r.add_argument("--source-threshold", type=float, required=True)
    r.add_argument("--transit-threshold", type=float, required=True)
    r.add_argument("--victim-threshold", type=float, required=True)
    r.add_argument("--path-threshold", type=float, required=True)

    s = sub.add_parser("selftest")
    s.add_argument("--tmp-dir", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "selftest":
        selftest(args.tmp_dir)
        return

    thresholds = {
        "source": args.source_threshold,
        "transit": args.transit_threshold,
        "victim": args.victim_threshold,
        "path": args.path_threshold,
    }
    execute(args.archive, args.output_dir, thresholds)
    print("V5_P3_POSTTRAIN_P3_VICTIM_ROLE_PATH_EVALUATION_COMPLETE")
    print("validation_only=true")
    print("threshold_tuning_performed=false")
    print("decoder_applied=false")
    print("sealed_test_access=false")


if __name__ == "__main__":
    main()
