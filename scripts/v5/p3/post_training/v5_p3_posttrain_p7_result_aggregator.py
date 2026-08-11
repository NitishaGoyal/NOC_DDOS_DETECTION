#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

FORBIDDEN_EVIDENCE_KEYS = (
    "sealed_test_access",
    "test_evaluation_performed",
    "test_tensor_contents_accessed",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a JSON object")
    for k in FORBIDDEN_EVIDENCE_KEYS:
        if obj.get(k) is True:
            raise ValueError(f"{path}: forbidden sealed/test evidence flag {k}=true")
    return obj


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        seen = set()
        ordered: List[str] = []
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    ordered.append(k)
        fields = ordered
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def safe_float(x: Any) -> Optional[float]:
    if x in (None, "", "None", "null", "NA", "NOT_AVAILABLE"):
        return None
    try:
        y = float(x)
    except Exception:
        return None
    return y if math.isfinite(y) else None


def maybe(path: Path) -> Optional[Path]:
    return path if path.is_file() else None


def build_response_latency(p2_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary = read_json(p2_dir / "onset_latency_summary.json")
    table = [{
        "Eligible attack runs": summary.get("eligible_attack_runs"),
        "Detected during attack": summary.get("detected_during_attack_runs"),
        "Undetected during attack": summary.get("undetected_during_attack_runs"),
        "Detected fraction": summary.get("detected_during_attack_fraction"),
        "Pre-onset false-alarm rate": summary.get("pre_onset_false_alarm_rate"),
        "False-alarm windows/run": summary.get("false_alarm_windows_per_run"),
        "Median graph latency (epochs)": summary.get("graph_first_latency_epochs_median_detected_during_attack"),
        "P95 graph latency (epochs)": summary.get("graph_first_latency_epochs_p95_detected_during_attack"),
        "Stable detection q": summary.get("stable_detection_consecutive_decisions_q"),
        "Evidence class": "VALIDATION-EXPLORATORY",
    }]
    cdf = read_csv(p2_dir / "graph_first_latency_cdf.csv")
    return table, cdf, summary


def build_victim_table(p3_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    s = read_json(p3_dir / "victim_role_path_summary.json")
    table = [{
        "Victim AP (active)": s.get("victim_average_precision_active_windows"),
        "Victim AUROC (active)": s.get("victim_auroc_active_windows"),
        "Victim Precision (active)": s.get("victim_precision_active_windows"),
        "Victim Recall (active)": s.get("victim_recall_active_windows"),
        "Victim F1 (active)": s.get("victim_f1_active_windows"),
        "Exact victim set": s.get("victim_exact_set_accuracy_active_windows"),
        "Top-1 victim accuracy": s.get("top1_victim_accuracy"),
        "Victim Recall@1": s.get("victim_recall_at_1"),
        "Victim Recall@2": s.get("victim_recall_at_2"),
        "Victim Recall@3": s.get("victim_recall_at_3"),
        "Victim MRR": s.get("victim_mean_reciprocal_rank"),
        "Exact source set": s.get("source_exact_set_accuracy_active_windows"),
        "Exact transit set": s.get("transit_exact_set_accuracy_active_windows"),
        "Exact path set": s.get("path_exact_set_accuracy_active_windows"),
        "Joint source+victim exact": s.get("joint_source_victim_exact_accuracy_active_windows"),
        "Complete source+path+victim exact": s.get("complete_source_path_victim_exact_accuracy_active_windows"),
        "Control any-victim-alarm rate": s.get("control_any_victim_alarm_rate"),
        "Evidence class": "VALIDATION-EXPLORATORY",
    }]
    ranking = [{
        "metric": "Top-1 accuracy",
        "value": s.get("top1_victim_accuracy"),
    }, {
        "metric": "Recall@1",
        "value": s.get("victim_recall_at_1"),
    }, {
        "metric": "Recall@2",
        "value": s.get("victim_recall_at_2"),
    }, {
        "metric": "Recall@3",
        "value": s.get("victim_recall_at_3"),
    }, {
        "metric": "MRR",
        "value": s.get("victim_mean_reciprocal_rank"),
    }]
    confusion = read_csv(p3_dir / "cross_head_role_confusion.csv")
    return table, ranking, confusion, s


def build_model_table(p4_file: Path) -> list[dict[str, Any]]:
    return read_csv(p4_file)


def build_robustness(p5_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(p5_dir / "robustness_subgroup_metrics.csv")
    coverage = read_csv(p5_dir / "robustness_metadata_coverage.csv")
    summary = read_json(p5_dir / "robustness_summary.json")

    preferred = {
        "attacker_count",
        "attack_strength",
        "attack_profile",
        "traffic_or_workload_class",
        "singleton_sv_distance_bin",
        "source_location",
        "victim_location",
        "true_transit_router_count",
        "true_path_router_count",
        "explicit_route_overlap",
    }
    table = [r for r in rows if r.get("dimension") in preferred]
    return table, coverage, summary


def build_inference_table(p6_file: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    s = read_json(p6_file)
    table = [{
        "Device": s.get("device"),
        "Precision": s.get("precision"),
        "Batch size": s.get("batch_size"),
        "Warmup iterations": s.get("warmup_iterations"),
        "Measured iterations": s.get("measured_iterations"),
        "Median latency (ms)": s.get("median_latency_ms"),
        "Mean latency (ms)": s.get("mean_latency_ms"),
        "P95 latency (ms)": s.get("p95_latency_ms"),
        "Sample SD (ms)": s.get("sample_standard_deviation_ms"),
        "Min latency (ms)": s.get("minimum_latency_ms"),
        "Max latency (ms)": s.get("maximum_latency_ms"),
        "Parameter count": s.get("parameter_count"),
        "Weight bytes": s.get("weight_bytes"),
        "Peak CUDA allocated bytes": s.get("peak_cuda_allocated_bytes"),
        "Evidence class": "ENGINEERING",
    }]
    return table, s


def build_f7(path: Optional[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if path is None or not path.is_file():
        na = [{
            "status": "NOT_AVAILABLE",
            "note": "Optional F7 summary CSV not supplied to this P7 execution.",
            "feature_removal_authorized": False,
        }]
        return na, na
    rows = read_csv(path)
    # Preserve all provided F7 fields; P7 does not reinterpret the evidence.
    out = []
    for r in rows:
        x = dict(r)
        x["Evidence class"] = x.get("Evidence class") or "VALIDATION-EXPLORATORY"
        x["feature_removal_authorized"] = False
        out.append(x)
    return out, out


def execute(
    p2_dir: Path,
    p3_dir: Path,
    p4_file: Path,
    p5_dir: Path,
    p6_file: Path,
    output_dir: Path,
    f7_file: Optional[Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    response_table, cdf, p2_summary = build_response_latency(p2_dir)
    victim_table, victim_ranking, role_confusion, p3_summary = build_victim_table(p3_dir)
    model_table = build_model_table(p4_file)
    robustness_table, coverage, p5_summary = build_robustness(p5_dir)
    inference_table, p6_summary = build_inference_table(p6_file)
    f7_table, f7_fig = build_f7(f7_file)

    write_csv(output_dir / "TABLE_model_comparison.csv", model_table)
    write_csv(output_dir / "TABLE_response_latency.csv", response_table)
    write_csv(output_dir / "TABLE_victim_role_path.csv", victim_table)
    write_csv(output_dir / "TABLE_robustness.csv", robustness_table)
    write_csv(output_dir / "TABLE_inference_cost.csv", inference_table)
    write_csv(output_dir / "TABLE_feature_group_ablation.csv", f7_table)

    write_csv(output_dir / "FIG_latency_cdf.csv", cdf)
    write_csv(output_dir / "FIG_victim_ranking.csv", victim_ranking)
    write_csv(output_dir / "FIG_role_confusion.csv", role_confusion)
    write_csv(output_dir / "FIG_robustness.csv", robustness_table)
    write_csv(output_dir / "FIG_feature_group_ablation.csv", f7_fig)

    manuscript = {
        "evidence_class": "VALIDATION-EXPLORATORY unless separately promoted",
        "sealed_test_claims_created": False,
        "feature_removal_authorized_from_f7": False,
        "response_latency": p2_summary,
        "victim_role_path": p3_summary,
        "robustness": p5_summary,
        "inference_latency": p6_summary,
        "model_comparison_rows": len(model_table),
        "f7_supplied": f7_file is not None and f7_file.is_file(),
    }
    (output_dir / "manuscript_numbers.json").write_text(
        json.dumps(manuscript, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest_rows = []
    for p in sorted(output_dir.iterdir()):
        if p.is_file() and p.name != "artifact_manifest.json":
            manifest_rows.append({
                "file": p.name,
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
            })
    manifest = {
        "status": "PASS",
        "classification": "ENGINEERING",
        "sealed_test_access": False,
        "sealed_test_claims_created": False,
        "files": manifest_rows,
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def create_selftest_tree(root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    p2 = root / "p2"; p3 = root / "p3"; p5 = root / "p5"
    p2.mkdir(parents=True); p3.mkdir(); p5.mkdir()

    (p2 / "onset_latency_summary.json").write_text(json.dumps({
        "eligible_attack_runs": 4,
        "detected_during_attack_runs": 3,
        "undetected_during_attack_runs": 1,
        "detected_during_attack_fraction": 0.75,
        "pre_onset_false_alarm_rate": 0.25,
        "false_alarm_windows_per_run": 0.5,
        "graph_first_latency_epochs_median_detected_during_attack": 8.0,
        "graph_first_latency_epochs_p95_detected_during_attack": 15.2,
        "stable_detection_consecutive_decisions_q": 2,
        "sealed_test_access": False,
    }, indent=2) + "\n")
    _write_csv(p2 / "graph_first_latency_cdf.csv",
               ["latency_epochs","detected_runs_at_or_below","eligible_attack_runs","cdf_fraction_all_attack_runs"],
               [{"latency_epochs":0,"detected_runs_at_or_below":1,"eligible_attack_runs":4,"cdf_fraction_all_attack_runs":0.25},
                {"latency_epochs":8,"detected_runs_at_or_below":3,"eligible_attack_runs":4,"cdf_fraction_all_attack_runs":0.75}])
    _write_csv(p2 / "per_run_onset_latency.csv", ["run_key"], [{"run_key":"R1"}])

    (p3 / "victim_role_path_summary.json").write_text(json.dumps({
        "victim_average_precision_active_windows": 0.71,
        "victim_auroc_active_windows": 0.91,
        "victim_precision_active_windows": 0.70,
        "victim_recall_active_windows": 0.72,
        "victim_f1_active_windows": 0.71,
        "victim_exact_set_accuracy_active_windows": 0.61,
        "top1_victim_accuracy": 0.84,
        "victim_recall_at_1": 0.78,
        "victim_recall_at_2": 0.90,
        "victim_recall_at_3": 0.95,
        "victim_mean_reciprocal_rank": 0.88,
        "source_exact_set_accuracy_active_windows": 0.60,
        "transit_exact_set_accuracy_active_windows": 0.57,
        "path_exact_set_accuracy_active_windows": 0.55,
        "joint_source_victim_exact_accuracy_active_windows": 0.48,
        "complete_source_path_victim_exact_accuracy_active_windows": 0.42,
        "control_any_victim_alarm_rate": 0.03,
        "sealed_test_access": False,
    }, indent=2) + "\n")
    _write_csv(p3 / "cross_head_role_confusion.csv",
               ["population","cross_head","router_entries","cross_head_positive_rate","mean_cross_head_probability","threshold"],
               [{"population":"true_victim","cross_head":"transit","router_entries":10,"cross_head_positive_rate":0.2,"mean_cross_head_probability":0.3,"threshold":0.5}])
    _write_csv(p3 / "per_window_role_exactness.csv", ["sample_key"], [{"sample_key":"S1"}])

    p4 = root / "model_comparison_aggregate.csv"
    _write_csv(p4,
               ["dataset_scope","model_id","task_scope","screen_or_confirmation","seeds","n_seeds","graph_auroc_mean","victim_average_precision_mean"],
               [{"dataset_scope":"V5_P3_TRANCHE_A_DYNAMIC70","model_id":"conv1d_graphconv","task_scope":"multitask",
                 "screen_or_confirmation":"SCREENING","seeds":"107","n_seeds":1,"graph_auroc_mean":0.89,"victim_average_precision_mean":0.61}])

    _write_csv(p5 / "robustness_subgroup_metrics.csv",
               ["dimension","group_value","window_count","active_attack_window_count","run_count","graph_auroc_all_windows","victim_average_precision_active_windows"],
               [{"dimension":"attacker_count","group_value":"K1","window_count":20,"active_attack_window_count":10,"run_count":2,"graph_auroc_all_windows":0.9,"victim_average_precision_active_windows":0.7}])
    _write_csv(p5 / "robustness_metadata_coverage.csv",
               ["dimension","total_windows","missing_windows","nonmissing_windows","nonmissing_fraction","distinct_group_values"],
               [{"dimension":"attack_strength","total_windows":20,"missing_windows":0,"nonmissing_windows":20,"nonmissing_fraction":1.0,"distinct_group_values":2}])
    (p5 / "robustness_summary.json").write_text(json.dumps({
        "validation_windows":20,"validation_runs":2,"sealed_test_access":False
    }, indent=2) + "\n")

    p6 = root / "inference_latency_result.json"
    p6.write_text(json.dumps({
        "device":"cpu","precision":"fp32","batch_size":1,"warmup_iterations":100,"measured_iterations":1000,
        "median_latency_ms":0.2,"mean_latency_ms":0.21,"p95_latency_ms":0.25,
        "sample_standard_deviation_ms":0.03,"minimum_latency_ms":0.18,"maximum_latency_ms":0.40,
        "parameter_count":60553,"weight_bytes":242212,"peak_cuda_allocated_bytes":None,
        "sealed_test_access":False
    }, indent=2) + "\n")

    f7 = root / "feature_group_ablation_summary.csv"
    _write_csv(f7,
               ["variant","selection_score","delta_vs_control"],
               [{"variant":"control_dynamic70","selection_score":0.7601,"delta_vs_control":0.0},
                {"variant":"ablate_buffer_pressure","selection_score":0.7434,"delta_vs_control":-0.0167}])
    return p2, p3, p4, p5, p6, f7


def selftest(tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    p2,p3,p4,p5,p6,f7 = create_selftest_tree(tmp / "inputs")
    out = tmp / "out"
    execute(p2,p3,p4,p5,p6,out,f7)

    expected = {
        "TABLE_model_comparison.csv",
        "TABLE_response_latency.csv",
        "TABLE_victim_role_path.csv",
        "TABLE_robustness.csv",
        "TABLE_inference_cost.csv",
        "TABLE_feature_group_ablation.csv",
        "FIG_latency_cdf.csv",
        "FIG_victim_ranking.csv",
        "FIG_role_confusion.csv",
        "FIG_robustness.csv",
        "FIG_feature_group_ablation.csv",
        "manuscript_numbers.json",
        "artifact_manifest.json",
    }
    produced = {p.name for p in out.iterdir() if p.is_file()}
    assert expected == produced

    m = json.loads((out / "manuscript_numbers.json").read_text())
    assert m["sealed_test_claims_created"] is False
    assert m["feature_removal_authorized_from_f7"] is False
    assert m["response_latency"]["eligible_attack_runs"] == 4

    a = json.loads((out / "artifact_manifest.json").read_text())
    assert a["status"] == "PASS"
    assert a["sealed_test_access"] is False
    assert len(a["files"]) == 12

    print("P7_SELFTEST_PASS")
    print("paper_table_files=6")
    print("figure_ready_files=5")
    print("machine_readable_files=2")
    print("sealed_test_claims_created=false")
    print("feature_removal_authorized=false")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--p2-dir", type=Path, required=True)
    r.add_argument("--p3-dir", type=Path, required=True)
    r.add_argument("--p4-model-comparison-csv", type=Path, required=True)
    r.add_argument("--p5-dir", type=Path, required=True)
    r.add_argument("--p6-inference-json", type=Path, required=True)
    r.add_argument("--output-dir", type=Path, required=True)
    r.add_argument("--f7-summary-csv", type=Path)

    s = sub.add_parser("selftest")
    s.add_argument("--tmp-dir", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "selftest":
        selftest(args.tmp_dir)
        return

    execute(
        args.p2_dir,
        args.p3_dir,
        args.p4_model_comparison_csv,
        args.p5_dir,
        args.p6_inference_json,
        args.output_dir,
        args.f7_summary_csv,
    )
    print("V5_P3_POSTTRAIN_P7_RESULT_AGGREGATION_COMPLETE")
    print("sealed_test_claims_created=false")
    print("sealed_test_access=false")


if __name__ == "__main__":
    main()
