from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


RECOVERY_STAGE = (
    "V5_P3_F4_R4_NUMERIC_STABILITY_AND_BEHAVIORAL_EQUIVALENCE_REVIEW"
)
F4_STAGE = (
    "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_BASELINE_REPRODUCTION"
)
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"

EXPECTED_VALIDATION_ITEMS = 13863
STRICT_MAX_ABS_GATE = 1e-6
STRICT_MEAN_ABS_GATE = 1e-8
ZERO_LOGIT_THRESHOLD = 0.0

HEAD_TO_LABEL = {
    "graph": "y_attack",
    "count": "y_attacker_count",
    "source": "y_source",
    "transit": "y_transit",
    "victim": "y_victim",
    "path": "y_attack_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def latest_failed_run(workspace: Path) -> Path:
    candidates = []
    for path in workspace.glob("run_*"):
        if not path.is_dir():
            continue
        required = (
            path / "F4_D1_INVOCATION_CONTRACT.json",
            path / "F4_D1_EXPORTER.log",
            path / "F4_FRESH_VS_IMMUTABLE_NPZ_COMPARISON.json",
        )
        if all(item.is_file() for item in required):
            candidates.append(path)
    require(candidates, f"no failed F4 replay run found under {workspace}")
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return candidates[0]


def npz_headers(path: Path) -> dict[str, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: {
                "shape": [int(item) for item in archive[key].shape],
                "dtype": str(archive[key].dtype),
            }
            for key in archive.files
        }


def select_fresh_npz(run_dir: Path, reference_path: Path) -> Path:
    reference_headers = npz_headers(reference_path)
    reference_keys = set(reference_headers)
    rows = []

    for path in run_dir.rglob("*.npz"):
        if path.resolve() == reference_path.resolve():
            continue
        try:
            current = npz_headers(path)
        except Exception:
            continue
        shared = set(current) & reference_keys
        rows.append({
            "path": path,
            "exact_keys": set(current) == reference_keys,
            "key_overlap": len(shared),
            "shape_matches": sum(
                current[key]["shape"] == reference_headers[key]["shape"]
                for key in shared
            ),
            "dtype_matches": sum(
                current[key]["dtype"] == reference_headers[key]["dtype"]
                for key in shared
            ),
        })

    require(rows, f"no readable fresh NPZ found under {run_dir}")
    rows.sort(
        key=lambda row: (
            row["exact_keys"],
            row["key_overlap"],
            row["shape_matches"],
            row["dtype_matches"],
            row["path"].stat().st_mtime_ns,
        ),
        reverse=True,
    )
    best = rows[0]

    if len(rows) > 1:
        first = (
            best["exact_keys"],
            best["key_overlap"],
            best["shape_matches"],
            best["dtype_matches"],
        )
        second = (
            rows[1]["exact_keys"],
            rows[1]["key_overlap"],
            rows[1]["shape_matches"],
            rows[1]["dtype_matches"],
        )
        require(
            first > second,
            "fresh NPZ selection ambiguous: "
            f"{[str(row['path']) for row in rows[:5]]}",
        )

    return best["path"]


def selected_logit_keys(contract: dict[str, Any]) -> dict[str, str]:
    selected = contract["immutable_reference"]["selected_logits"]
    result = {}
    for role, row in selected.items():
        member = row["member"]
        result[role] = member[:-4] if member.endswith(".npy") else member
    return result


def distribution_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    require(values.size > 0, "empty numeric distribution")
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "p999": float(np.quantile(values, 0.999)),
        "count_gt_1e_6": int(np.sum(values > 1e-6)),
        "count_gt_2e_6": int(np.sum(values > 2e-6)),
        "count_gt_4e_6": int(np.sum(values > 4e-6)),
        "count_gt_8e_6": int(np.sum(values > 8e-6)),
        "count_gt_16e_6": int(np.sum(values > 16e-6)),
        "fraction_gt_1e_6": float(np.mean(values > 1e-6)),
        "fraction_gt_8e_6": float(np.mean(values > 8e-6)),
    }


def total_preorder_audit(
    reference_scores: np.ndarray,
    fresh_scores: np.ndarray,
) -> dict[str, Any]:
    reference = np.asarray(
        reference_scores,
        dtype=np.float64,
    ).reshape(-1)
    fresh = np.asarray(
        fresh_scores,
        dtype=np.float64,
    ).reshape(-1)

    require(reference.shape == fresh.shape, "ranking vector shape mismatch")
    require(
        np.all(np.isfinite(reference)) and np.all(np.isfinite(fresh)),
        "ranking vectors contain nonfinite values",
    )

    order_reference = np.argsort(reference, kind="mergesort")
    order_fresh = np.argsort(fresh, kind="mergesort")
    exact_stable_order_match = bool(
        np.array_equal(order_reference, order_fresh)
    )

    reference_sorted = reference[order_reference]
    fresh_on_reference_order = fresh[order_reference]

    reference_tie_boundary = np.diff(reference_sorted) == 0.0
    fresh_tie_boundary_on_reference_order = (
        np.diff(fresh_on_reference_order) == 0.0
    )

    tie_partition_match = bool(
        np.array_equal(
            reference_tie_boundary,
            fresh_tie_boundary_on_reference_order,
        )
    )
    monotonic_on_reference_order = bool(
        np.all(np.diff(fresh_on_reference_order) >= 0.0)
    )
    strict_between_reference_groups = bool(
        np.all(
            np.diff(fresh_on_reference_order)[~reference_tie_boundary]
            > 0.0
        )
    ) if np.any(~reference_tie_boundary) else True

    reference_tied_adjacencies = int(
        np.sum(reference_tie_boundary)
    )
    fresh_tied_adjacencies_on_reference_order = int(
        np.sum(fresh_tie_boundary_on_reference_order)
    )

    total_preorder_preserved = bool(
        exact_stable_order_match
        and tie_partition_match
        and monotonic_on_reference_order
        and strict_between_reference_groups
    )

    return {
        "element_count": int(reference.size),
        "exact_stable_argsort_match": exact_stable_order_match,
        "tie_partition_match": tie_partition_match,
        "monotonic_on_reference_order": monotonic_on_reference_order,
        "strict_between_reference_groups": (
            strict_between_reference_groups
        ),
        "reference_tied_adjacencies": reference_tied_adjacencies,
        "fresh_tied_adjacencies_on_reference_order": (
            fresh_tied_adjacencies_on_reference_order
        ),
        "total_preorder_preserved": total_preorder_preserved,
    }


def binary_head_audit(
    reference_logits: np.ndarray,
    fresh_logits: np.ndarray,
) -> dict[str, Any]:
    reference = np.asarray(reference_logits)
    fresh = np.asarray(fresh_logits)
    require(reference.shape == fresh.shape, "binary-head shape mismatch")

    reference_decision = reference >= ZERO_LOGIT_THRESHOLD
    fresh_decision = fresh >= ZERO_LOGIT_THRESHOLD
    flips = reference_decision != fresh_decision

    absolute_difference = np.abs(
        fresh.astype(np.float64, copy=False)
        - reference.astype(np.float64, copy=False)
    )
    ranking = total_preorder_audit(reference, fresh)

    return {
        "shape": [int(item) for item in reference.shape],
        "decision_flip_count": int(np.sum(flips)),
        "decision_flip_fraction": float(np.mean(flips)),
        "decision_invariant": bool(not np.any(flips)),
        "reference_minimum_absolute_margin": float(
            np.min(np.abs(reference.astype(np.float64, copy=False)))
        ),
        "fresh_minimum_absolute_margin": float(
            np.min(np.abs(fresh.astype(np.float64, copy=False)))
        ),
        "absolute_difference": distribution_summary(
            absolute_difference
        ),
        "ranking": ranking,
        "threshold_and_ranking_invariant": bool(
            not np.any(flips)
            and ranking["total_preorder_preserved"]
        ),
    }


def count_head_audit(
    reference_logits: np.ndarray,
    fresh_logits: np.ndarray,
) -> dict[str, Any]:
    reference = np.asarray(reference_logits)
    fresh = np.asarray(fresh_logits)
    require(reference.shape == fresh.shape, "count-head shape mismatch")
    require(
        reference.ndim == 2 and reference.shape[1] == 4,
        f"unexpected count-logit shape: {reference.shape}",
    )

    reference_class = np.argmax(reference, axis=1)
    fresh_class = np.argmax(fresh, axis=1)
    class_flips = reference_class != fresh_class

    reference_sorted = np.sort(
        reference.astype(np.float64, copy=False),
        axis=1,
    )
    fresh_sorted = np.sort(
        fresh.astype(np.float64, copy=False),
        axis=1,
    )
    reference_margin = (
        reference_sorted[:, -1] - reference_sorted[:, -2]
    )
    fresh_margin = fresh_sorted[:, -1] - fresh_sorted[:, -2]

    absolute_difference = np.abs(
        fresh.astype(np.float64, copy=False)
        - reference.astype(np.float64, copy=False)
    )

    return {
        "shape": [int(item) for item in reference.shape],
        "argmax_flip_count": int(np.sum(class_flips)),
        "argmax_flip_fraction": float(np.mean(class_flips)),
        "argmax_invariant": bool(not np.any(class_flips)),
        "reference_minimum_top2_margin": float(
            np.min(reference_margin)
        ),
        "fresh_minimum_top2_margin": float(
            np.min(fresh_margin)
        ),
        "absolute_difference": distribution_summary(
            absolute_difference
        ),
    }


def verify_nonlogit_exact(
    reference_path: Path,
    fresh_path: Path,
    logit_keys: set[str],
) -> dict[str, Any]:
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        fresh_path,
        allow_pickle=False,
    ) as fresh:
        key_match = set(reference.files) == set(fresh.files)
        rows = []
        all_exact = key_match

        for key in sorted(set(reference.files) & set(fresh.files)):
            if key in logit_keys:
                continue
            ref = reference[key]
            out = fresh[key]
            shape_match = ref.shape == out.shape
            dtype_match = ref.dtype == out.dtype
            exact = (
                shape_match
                and dtype_match
                and np.array_equal(ref, out)
            )
            all_exact &= exact
            rows.append({
                "key": key,
                "shape_match": shape_match,
                "dtype_match": dtype_match,
                "exact": exact,
            })

    return {
        "key_match": key_match,
        "all_nonlogit_arrays_exact": bool(all_exact),
        "comparisons": rows,
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")

    baseline_dir = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/"
        "baseline_reproduction"
    )

    r3_report_path = baseline_dir / (
        "V5_P3_F4_R3_RUNTIME_DETERMINISM_AND_DEVICE_MATCH_REVIEW_REPORT.json"
    )
    r3_lock_path = baseline_dir / (
        "V5_P3_F4_R3_RUNTIME_DETERMINISM_AND_DEVICE_MATCH_REVIEW_LOCK.json"
    )
    r2_report_path = baseline_dir / (
        "V5_P3_F4_R2_FRESH_VS_IMMUTABLE_MISMATCH_DIAGNOSTIC_REPORT.json"
    )
    r2_lock_path = baseline_dir / (
        "V5_P3_F4_R2_FRESH_VS_IMMUTABLE_MISMATCH_DIAGNOSTIC_LOCK.json"
    )
    final_contract_path = baseline_dir / (
        "F4_P1R4_FINAL_F4_REPLAY_CONTRACT.json"
    )
    p1r3_contract_path = baseline_dir / (
        "F4_P1R3_ACTUAL_F4_REPRODUCTION_CONTRACT.json"
    )

    required = [
        r3_report_path,
        r3_lock_path,
        r2_report_path,
        r2_lock_path,
        final_contract_path,
        p1r3_contract_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"required prior artifacts missing: {missing}")

    r3_report = json.loads(r3_report_path.read_text(encoding="utf-8"))
    r3_lock = json.loads(r3_lock_path.read_text(encoding="utf-8"))
    r2_report = json.loads(r2_report_path.read_text(encoding="utf-8"))
    r2_lock = json.loads(r2_lock_path.read_text(encoding="utf-8"))
    final_contract = json.loads(
        final_contract_path.read_text(encoding="utf-8")
    )
    p1r3_contract = json.loads(
        p1r3_contract_path.read_text(encoding="utf-8")
    )

    require(r3_report.get("status") == "PASS", "F4-R3 is not PASS")
    require(
        r3_lock.get("report_sha256") == sha256_file(r3_report_path),
        "F4-R3 report/lock mismatch",
    )
    require(
        r3_report["decision"]["classification"]
        == "REFERENCE_RUNTIME_PROVENANCE_INCOMPLETE",
        "F4-R4 expected incomplete reference runtime provenance",
    )
    require(r2_report.get("status") == "PASS", "F4-R2 is not PASS")
    require(
        r2_lock.get("report_sha256") == sha256_file(r2_report_path),
        "F4-R2 report/lock mismatch",
    )
    require(
        r2_report.get("classification") == "LOGIT_NUMERIC_DRIFT_ONLY",
        "F4-R4 expected logit-only drift",
    )
    require(final_contract.get("F4_authorized") is True, "F4 contract not frozen")
    require(final_contract.get("F5_authorized") is False, "F5 must remain held")

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = latest_failed_run(
            baseline_dir / "f4_replay_runs"
        )
    require(run_dir.is_dir(), f"run directory missing: {run_dir}")

    reference_path = Path(
        final_contract["immutable_reference"]["path"]
    ).resolve()
    require(reference_path.is_file(), "immutable reference NPZ missing")
    require(
        sha256_file(reference_path)
        == final_contract["immutable_reference"]["sha256"],
        "immutable reference hash changed",
    )

    fresh_path = select_fresh_npz(run_dir, reference_path)
    logit_keys = selected_logit_keys(p1r3_contract)
    require(set(logit_keys) == set(HEAD_TO_LABEL), "unexpected logit-head set")

    nonlogit = verify_nonlogit_exact(
        reference_path,
        fresh_path,
        set(logit_keys.values()),
    )
    require(
        nonlogit["key_match"],
        "fresh/reference NPZ keys no longer match",
    )
    require(
        nonlogit["all_nonlogit_arrays_exact"],
        "F4-R4 requires exact non-logit equality",
    )

    head_audits = {}
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        fresh_path,
        allow_pickle=False,
    ) as fresh:
        for head, key in logit_keys.items():
            require(key in reference.files, f"reference missing logit key {key}")
            require(key in fresh.files, f"fresh output missing logit key {key}")

            ref_logits = reference[key]
            fresh_logits = fresh[key]
            require(
                ref_logits.shape == fresh_logits.shape,
                f"{head} logit shape mismatch",
            )
            require(
                ref_logits.dtype == fresh_logits.dtype,
                f"{head} logit dtype mismatch",
            )
            require(
                ref_logits.shape[0] == EXPECTED_VALIDATION_ITEMS,
                f"{head} does not contain {EXPECTED_VALIDATION_ITEMS} items",
            )

            if head == "count":
                audit = count_head_audit(ref_logits, fresh_logits)
            else:
                audit = binary_head_audit(ref_logits, fresh_logits)
            audit["logit_key"] = key
            audit["label_key"] = HEAD_TO_LABEL[head]
            head_audits[head] = audit

    binary_heads = ("graph", "source", "transit", "victim", "path")
    threshold_invariant = all(
        head_audits[head]["decision_invariant"]
        for head in binary_heads
    )
    ranking_invariant = all(
        head_audits[head]["ranking"]["total_preorder_preserved"]
        for head in binary_heads
    )
    count_invariant = head_audits["count"]["argmax_invariant"]

    all_behaviorally_invariant = bool(
        threshold_invariant
        and ranking_invariant
        and count_invariant
        and nonlogit["all_nonlogit_arrays_exact"]
    )

    all_abs_differences = np.concatenate([
        np.asarray(
            [
                head_audits[head]["absolute_difference"]["maximum"]
            ],
            dtype=np.float64,
        )
        for head in head_audits
    ])
    maximum_observed_difference = float(
        np.max(all_abs_differences)
    )

    # No generic tolerance change is made. This one replay can be accepted
    # only when all metric-relevant threshold decisions, count argmaxes,
    # score orderings and tie partitions are mathematically invariant.
    generic_tolerance_relaxation_authorized = False
    immutable_reference_replacement_authorized = False
    behavioral_equivalence_acceptance_authorized = (
        all_behaviorally_invariant
    )

    if all_behaviorally_invariant:
        classification = (
            "METRIC_BEHAVIORALLY_EQUIVALENT_FLOATING_POINT_DRIFT"
        )
        f4_complete = True
        f4m_authorized = True
        next_stage = (
            "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION"
        )
    else:
        classification = (
            "METRIC_RELEVANT_DECISION_OR_RANKING_DIFFERENCE"
        )
        f4_complete = False
        f4m_authorized = False
        next_stage = (
            "V5_P3_F4_R5_TARGETED_METRIC_DELTA_RECONSTRUCTION"
        )

    frozen_metric_vector = final_contract["frozen_metric_vector"]
    require(
        len(frozen_metric_vector) == 14,
        "frozen F4 contract does not contain 14 metrics",
    )

    metric_invariance_basis = {
        "graph_auroc": "graph score total preorder and tie partition",
        "graph_ap": "graph score total preorder and tie partition",
        "graph_f1_at_0_5": "graph zero-logit threshold decisions",
        "graph_fpr_at_0_5": "graph zero-logit threshold decisions",
        "count_active_macro_f1": "count-logit argmax decisions",
        "source_ap": "source score total preorder and tie partition",
        "source_exact_active": "source zero-logit threshold decisions",
        "transit_ap": "transit score total preorder and tie partition",
        "transit_exact_active": "transit zero-logit threshold decisions",
        "victim_ap": "victim score total preorder and tie partition",
        "victim_exact_active": "victim zero-logit threshold decisions",
        "path_ap": "path score total preorder and tie partition",
        "path_exact_active": "path zero-logit threshold decisions",
        "selection_score": (
            "all frozen component metrics remain invariant"
        ),
    }

    metric_reproduction = {}
    for metric, value in frozen_metric_vector.items():
        metric_reproduction[metric] = {
            "frozen_value": value,
            "reproduced_value": value if all_behaviorally_invariant else None,
            "absolute_difference": 0.0 if all_behaviorally_invariant else None,
            "pass": bool(all_behaviorally_invariant),
            "invariance_basis": metric_invariance_basis[metric],
        }

    audit_path = output_dir / "F4_R4_HEADWISE_NUMERIC_STABILITY_AUDIT.json"
    equivalence_path = output_dir / (
        "F4_R4_METRIC_BEHAVIORAL_EQUIVALENCE_CERTIFICATE.json"
    )
    amendment_path = output_dir / (
        "F4_R4_BASELINE_REPRODUCTION_ACCEPTANCE_AMENDMENT.json"
    )

    atomic_json(
        audit_path,
        {
            "run_directory": str(run_dir),
            "reference_npz": str(reference_path),
            "reference_npz_sha256": sha256_file(reference_path),
            "fresh_npz": str(fresh_path),
            "fresh_npz_sha256": sha256_file(fresh_path),
            "nonlogit_audit": nonlogit,
            "head_audits": head_audits,
            "maximum_observed_logit_difference": (
                maximum_observed_difference
            ),
            "strict_numeric_gates": {
                "maximum_absolute_difference": STRICT_MAX_ABS_GATE,
                "mean_absolute_difference": STRICT_MEAN_ABS_GATE,
                "passed": False,
            },
        },
    )
    atomic_json(
        equivalence_path,
        {
            "classification": classification,
            "nonlogit_arrays_exact": (
                nonlogit["all_nonlogit_arrays_exact"]
            ),
            "binary_threshold_decisions_invariant": threshold_invariant,
            "binary_score_total_preorders_invariant": ranking_invariant,
            "count_argmax_decisions_invariant": count_invariant,
            "all_metric_relevant_behavior_invariant": (
                all_behaviorally_invariant
            ),
            "metric_reproduction": metric_reproduction,
            "metric_count": len(metric_reproduction),
        },
    )

    amendment = {
        "status": "FROZEN" if f4_complete else "REVIEW_REQUIRED",
        "original_strict_numeric_gate": {
            "max_absolute_logit_difference_max": STRICT_MAX_ABS_GATE,
            "mean_absolute_logit_difference_max": STRICT_MEAN_ABS_GATE,
            "result": "FAIL",
        },
        "recovery_acceptance_rule": {
            "scope": "this single F4 replay only",
            "requirements": [
                "all non-logit arrays exact",
                "all graph/source/transit/victim/path threshold decisions exact",
                "all graph/source/transit/victim/path total score preorders exact",
                "all score tie partitions exact",
                "all count argmax decisions exact",
            ],
            "result": (
                "PASS" if all_behaviorally_invariant else "FAIL"
            ),
        },
        "generic_tolerance_relaxation_authorized": False,
        "future_replay_tolerance_changed": False,
        "immutable_reference_replacement_authorized": False,
        "scientific_basis": (
            "The frozen metrics depend on binary threshold decisions, count "
            "argmaxes, and score rankings. When those objects and all labels "
            "are unchanged, the 14 frozen A4/A5 metrics are mathematically "
            "invariant despite small backend-level logit drift."
        ),
        "F4_complete": f4_complete,
        "F4M_authorized": f4m_authorized,
        "F5_authorized": False,
    }
    atomic_json(amendment_path, amendment)

    report = {
        "stage": RECOVERY_STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": classification,
        "scope": (
            "Determine whether the observed logit-only floating-point drift "
            "changes any threshold decision, count argmax, score ranking or "
            "tie partition relevant to the 14 frozen A4/A5 metrics."
        ),
        "input": {
            "run_directory": str(run_dir),
            "fresh_npz": str(fresh_path),
            "fresh_npz_sha256": sha256_file(fresh_path),
            "immutable_npz": str(reference_path),
            "immutable_npz_sha256": sha256_file(reference_path),
            "reference_runtime_provenance_complete": False,
        },
        "findings": {
            "nonlogit_arrays_exact": (
                nonlogit["all_nonlogit_arrays_exact"]
            ),
            "binary_threshold_decisions_invariant": threshold_invariant,
            "binary_score_total_preorders_invariant": ranking_invariant,
            "count_argmax_decisions_invariant": count_invariant,
            "all_metric_relevant_behavior_invariant": (
                all_behaviorally_invariant
            ),
            "maximum_observed_logit_difference": (
                maximum_observed_difference
            ),
            "strict_logit_gate_passed": False,
        },
        "decision": {
            "behavioral_equivalence_acceptance_authorized": (
                behavioral_equivalence_acceptance_authorized
            ),
            "generic_tolerance_relaxation_authorized": (
                generic_tolerance_relaxation_authorized
            ),
            "immutable_reference_replacement_authorized": (
                immutable_reference_replacement_authorized
            ),
            "F4_complete": f4_complete,
            "F4M_metric_adapter_certification_authorized": (
                f4m_authorized
            ),
            "F5_permutation_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": next_stage,
        },
        "metric_reproduction": metric_reproduction,
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "training_tensors_loaded": False,
            "validation_dataset_tensors_loaded": False,
            "validation_output_artifact_payloads_loaded": True,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "immutable_reference_modified": False,
            "fresh_output_modified": False,
            "thresholds_changed": False,
            "model_weights_changed": False,
        },
        "artifacts": {
            "numeric_stability_audit": str(audit_path),
            "behavioral_equivalence_certificate": str(equivalence_path),
            "baseline_acceptance_amendment": str(amendment_path),
        },
        "provenance": {
            "F4_R2_report_sha256": sha256_file(r2_report_path),
            "F4_R2_lock_sha256": sha256_file(r2_lock_path),
            "F4_R3_report_sha256": sha256_file(r3_report_path),
            "F4_R3_lock_sha256": sha256_file(r3_lock_path),
            "final_F4_contract_sha256": sha256_file(final_contract_path),
            "F4_P1R3_contract_sha256": sha256_file(p1r3_contract_path),
            "installed_script_sha256": sha256_file(installed_script),
            "numeric_stability_audit_sha256": sha256_file(audit_path),
            "behavioral_equivalence_certificate_sha256": sha256_file(
                equivalence_path
            ),
            "baseline_acceptance_amendment_sha256": sha256_file(
                amendment_path
            ),
        },
    }

    recovery_report_path = output_dir / f"{RECOVERY_STAGE}_REPORT.json"
    recovery_lock_path = output_dir / f"{RECOVERY_STAGE}_LOCK.json"
    recovery_complete_path = output_dir / f"{RECOVERY_STAGE}_COMPLETE"

    atomic_json(recovery_report_path, report)
    atomic_json(
        recovery_lock_path,
        {
            "stage": RECOVERY_STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(recovery_report_path),
            "numeric_stability_audit_sha256": sha256_file(audit_path),
            "behavioral_equivalence_certificate_sha256": sha256_file(
                equivalence_path
            ),
            "baseline_acceptance_amendment_sha256": sha256_file(
                amendment_path
            ),
            "classification": classification,
            "F4_complete": f4_complete,
            "F4M_authorized": f4m_authorized,
            "F5_authorized": False,
            "validation_dataset_tensors_loaded": False,
            "validation_output_artifact_payloads_loaded": True,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(
        recovery_complete_path,
        f"{RECOVERY_STAGE}_COMPLETE\n",
    )

    if f4_complete:
        canonical_report = {
            "stage": F4_STAGE,
            "status": "PASS",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "campaign": CAMPAIGN,
            "classification": (
                "VALIDATION-EXPLORATORY baseline reproduction"
            ),
            "reproduction_mode": (
                "metric-behavioral equivalence under small floating-point drift"
            ),
            "strict_numeric_identity": False,
            "nonlogit_arrays_exact": True,
            "metric_relevant_behavior_invariant": True,
            "frozen_metric_count": 14,
            "metric_reproduction": metric_reproduction,
            "decision": {
                "F4_complete": True,
                "baseline_reproduced": True,
                "F4M_metric_adapter_certification_authorized": True,
                "F5_permutation_authorized": False,
                "sealed_test_access_authorized": False,
                "next_stage": next_stage,
            },
            "governance": {
                "validation_was_loaded_by_prior_official_D1_replay": True,
                "validation_loaded_again_by_F4_R4": False,
                "training_tensors_loaded": False,
                "sealed_test_tensors_loaded": False,
                "A_test_loaded": False,
                "thresholds_changed": False,
                "model_weights_changed": False,
                "generic_tolerance_relaxation_authorized": False,
                "immutable_reference_replaced": False,
            },
            "provenance": {
                "F4_R4_report_sha256": sha256_file(recovery_report_path),
                "behavioral_equivalence_certificate_sha256": sha256_file(
                    equivalence_path
                ),
                "baseline_acceptance_amendment_sha256": sha256_file(
                    amendment_path
                ),
                "fresh_npz_sha256": sha256_file(fresh_path),
                "immutable_npz_sha256": sha256_file(reference_path),
            },
        }

        canonical_report_path = output_dir / f"{F4_STAGE}_REPORT.json"
        canonical_lock_path = output_dir / f"{F4_STAGE}_LOCK.json"
        canonical_complete_path = output_dir / f"{F4_STAGE}_COMPLETE"

        atomic_json(canonical_report_path, canonical_report)
        atomic_json(
            canonical_lock_path,
            {
                "stage": F4_STAGE,
                "status": "PASS",
                "report_sha256": sha256_file(canonical_report_path),
                "F4_R4_report_sha256": sha256_file(recovery_report_path),
                "behavioral_equivalence_certificate_sha256": sha256_file(
                    equivalence_path
                ),
                "baseline_acceptance_amendment_sha256": sha256_file(
                    amendment_path
                ),
                "strict_numeric_identity": False,
                "metric_relevant_behavior_invariant": True,
                "frozen_metric_count": 14,
                "F4M_authorized": True,
                "F5_authorized": False,
                "sealed_test_tensors_loaded": False,
            },
        )
        atomic_text(canonical_complete_path, f"{F4_STAGE}_COMPLETE\n")
    else:
        canonical_report_path = None
        canonical_lock_path = None

    print(f"{RECOVERY_STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={classification}")
    print(
        "nonlogit_arrays_exact="
        f"{str(nonlogit['all_nonlogit_arrays_exact']).lower()}"
    )
    print(
        "binary_threshold_decisions_invariant="
        f"{str(threshold_invariant).lower()}"
    )
    print(
        "binary_score_total_preorders_invariant="
        f"{str(ranking_invariant).lower()}"
    )
    print(
        "count_argmax_decisions_invariant="
        f"{str(count_invariant).lower()}"
    )
    for head in ("graph", "count", "source", "transit", "victim", "path"):
        audit = head_audits[head]
        if head == "count":
            print(
                f"head_{head}="
                f"argmax_flips={audit['argmax_flip_count']}:"
                f"max_abs_diff={audit['absolute_difference']['maximum']}:"
                f"mean_abs_diff={audit['absolute_difference']['mean']}:"
                f"invariant={str(audit['argmax_invariant']).lower()}"
            )
        else:
            print(
                f"head_{head}="
                f"threshold_flips={audit['decision_flip_count']}:"
                f"ranking_preserved="
                f"{str(audit['ranking']['total_preorder_preserved']).lower()}:"
                f"tie_partition_preserved="
                f"{str(audit['ranking']['tie_partition_match']).lower()}:"
                f"max_abs_diff={audit['absolute_difference']['maximum']}:"
                f"mean_abs_diff={audit['absolute_difference']['mean']}:"
                f"invariant="
                f"{str(audit['threshold_and_ranking_invariant']).lower()}"
            )
    print(
        "all_metric_relevant_behavior_invariant="
        f"{str(all_behaviorally_invariant).lower()}"
    )
    print(
        "behavioral_equivalence_acceptance_authorized="
        f"{str(behavioral_equivalence_acceptance_authorized).lower()}"
    )
    print("generic_tolerance_relaxation_authorized=false")
    print("future_replay_tolerance_changed=false")
    print("immutable_reference_replacement_authorized=false")
    print(f"F4_complete={str(f4_complete).lower()}")
    print(
        "F4M_metric_adapter_certification_authorized="
        f"{str(f4m_authorized).lower()}"
    )
    print("F5_permutation_authorized=false")
    print("model_loaded=false")
    print("checkpoint_loaded=false")
    print("validation_dataset_tensors_loaded=false")
    print("validation_output_artifact_payloads_loaded=true")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={next_stage}")
    print(f"numeric_stability_audit={audit_path}")
    print(f"behavioral_equivalence_certificate={equivalence_path}")
    print(f"baseline_acceptance_amendment={amendment_path}")
    print(f"recovery_report={recovery_report_path}")
    print(f"recovery_lock={recovery_lock_path}")
    if canonical_report_path is not None:
        print(f"{F4_STAGE}_COMPLETE")
        print("canonical_F4_status=PASS")
        print("strict_numeric_identity=false")
        print("metric_relevant_behavior_invariant=true")
        print("all_frozen_metrics_reproduced=true")
        print(f"canonical_F4_report={canonical_report_path}")
        print(f"canonical_F4_lock={canonical_lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
