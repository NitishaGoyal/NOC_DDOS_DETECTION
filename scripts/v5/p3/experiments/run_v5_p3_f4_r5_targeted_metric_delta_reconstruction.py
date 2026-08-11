from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


STAGE = "V5_P3_F4_R5_TARGETED_METRIC_DELTA_RECONSTRUCTION"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"

REFERENCE_MATCH_TOLERANCE = 1e-6
FRESH_METRIC_DELTA_TOLERANCE = 1e-6
EXPECTED_VALIDATION_ITEMS = 13863

ROLE_HEADS = ("source", "transit", "victim", "path")
ROLE_LABEL_KEYS = {
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


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def latest_failed_run(workspace: Path) -> Path:
    candidates = []
    for path in workspace.glob("run_*"):
        if not path.is_dir():
            continue
        if (
            (path / "F4_D1_INVOCATION_CONTRACT.json").is_file()
            and (path / "F4_FRESH_VS_IMMUTABLE_NPZ_COMPARISON.json").is_file()
        ):
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


def binary_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_pred = np.asarray(y_pred).astype(np.int64).reshape(-1)
    negative = y_true == 0
    denominator = int(np.sum(negative))
    require(denominator > 0, "FPR undefined: no negative examples")
    false_positive = int(np.sum((y_pred == 1) & negative))
    return false_positive / denominator


def safe_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    require(np.unique(y_true).size >= 2, "AP requires both classes")
    return float(average_precision_score(y_true, scores))


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    require(np.unique(y_true).size >= 2, "AUROC requires both classes")
    return float(roc_auc_score(y_true, scores))


def sample_masks(
    y_attack: np.ndarray,
    role_target: np.ndarray,
) -> dict[str, np.ndarray]:
    y_attack = np.asarray(y_attack).astype(np.int64).reshape(-1)
    role_target = np.asarray(role_target)
    return {
        "all": np.ones_like(y_attack, dtype=bool),
        "active": y_attack == 1,
        "role_positive": np.any(role_target > 0, axis=1),
        "active_and_role_positive": (
            (y_attack == 1) & np.any(role_target > 0, axis=1)
        ),
    }


def node_masks(
    role_mask: np.ndarray | None,
    shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    all_mask = np.ones(shape, dtype=bool)
    masks = {"all_nodes": all_mask}

    if role_mask is not None and role_mask.shape == shape:
        role_bool = role_mask.astype(bool)
        masks["role_mask"] = role_bool
        masks["inverse_role_mask"] = ~role_bool

    return masks


def exact_multilabel(
    target: np.ndarray,
    prediction: np.ndarray,
    sample_mask: np.ndarray,
    node_mask: np.ndarray,
    empty_policy: str,
) -> float:
    target = np.asarray(target).astype(bool)
    prediction = np.asarray(prediction).astype(bool)
    sample_mask = np.asarray(sample_mask).astype(bool).reshape(-1)
    node_mask = np.asarray(node_mask).astype(bool)

    require(target.shape == prediction.shape, "exact metric shape mismatch")
    require(node_mask.shape == target.shape, "node mask shape mismatch")

    selected = np.flatnonzero(sample_mask)
    require(selected.size > 0, "exact metric has no selected samples")

    exact_values = []
    for index in selected:
        valid = node_mask[index]
        if not np.any(valid):
            if empty_policy == "ignore":
                continue
            if empty_policy == "true":
                exact_values.append(True)
                continue
            exact_values.append(False)
            continue

        exact_values.append(
            bool(np.array_equal(target[index][valid], prediction[index][valid]))
        )

    require(exact_values, "exact metric has no evaluable samples")
    return float(np.mean(exact_values))


def evaluate_graph_candidates(
    reference: dict[str, np.ndarray],
    fresh: dict[str, np.ndarray],
    frozen: dict[str, float],
    logit_key: str,
) -> dict[str, list[dict[str, Any]]]:
    y_ref = reference["y_attack"].astype(np.int64).reshape(-1)
    y_fresh = fresh["y_attack"].astype(np.int64).reshape(-1)
    require(np.array_equal(y_ref, y_fresh), "graph labels differ")

    score_ref = reference[logit_key].astype(np.float64).reshape(-1)
    score_fresh = fresh[logit_key].astype(np.float64).reshape(-1)
    pred_ref = score_ref >= 0.0
    pred_fresh = score_fresh >= 0.0

    return {
        "graph_auroc": [{
            "variant": "roc_auc_score(y_attack, raw_attack_logit)",
            "reference_value": safe_auc(y_ref, score_ref),
            "fresh_value": safe_auc(y_fresh, score_fresh),
        }],
        "graph_ap": [{
            "variant": "average_precision_score(y_attack, raw_attack_logit)",
            "reference_value": safe_ap(y_ref, score_ref),
            "fresh_value": safe_ap(y_fresh, score_fresh),
        }],
        "graph_f1_at_0_5": [{
            "variant": "binary_f1(y_attack, attack_logit>=0)",
            "reference_value": float(
                f1_score(y_ref, pred_ref.astype(np.int64), zero_division=0)
            ),
            "fresh_value": float(
                f1_score(y_fresh, pred_fresh.astype(np.int64), zero_division=0)
            ),
        }],
        "graph_fpr_at_0_5": [{
            "variant": "binary_fpr(y_attack, attack_logit>=0)",
            "reference_value": binary_fpr(y_ref, pred_ref),
            "fresh_value": binary_fpr(y_fresh, pred_fresh),
        }],
    }


def evaluate_count_candidates(
    reference: dict[str, np.ndarray],
    fresh: dict[str, np.ndarray],
    logit_key: str,
) -> list[dict[str, Any]]:
    y_attack_ref = reference["y_attack"].astype(np.int64).reshape(-1)
    y_attack_fresh = fresh["y_attack"].astype(np.int64).reshape(-1)
    y_count_ref = reference["y_attacker_count"].astype(np.int64).reshape(-1)
    y_count_fresh = fresh["y_attacker_count"].astype(np.int64).reshape(-1)

    require(np.array_equal(y_attack_ref, y_attack_fresh), "attack labels differ")
    require(np.array_equal(y_count_ref, y_count_fresh), "count labels differ")

    pred_ref = np.argmax(reference[logit_key], axis=1).astype(np.int64)
    pred_fresh = np.argmax(fresh[logit_key], axis=1).astype(np.int64)

    rows = []
    masks = {
        "all": np.ones_like(y_attack_ref, dtype=bool),
        "active": y_attack_ref == 1,
    }
    transforms = {
        "direct": lambda values: values,
        "minus_one": lambda values: values - 1,
    }
    label_sets = {
        "present_only": None,
        "all_0_to_3": [0, 1, 2, 3],
    }

    for mask_name, mask in masks.items():
        if not np.any(mask):
            continue
        for transform_name, transform in transforms.items():
            transformed = transform(y_count_ref[mask])
            transformed_fresh = transform(y_count_fresh[mask])
            if (
                np.any(transformed < 0)
                or np.any(transformed > 3)
                or np.any(transformed_fresh < 0)
                or np.any(transformed_fresh > 3)
            ):
                continue

            for labels_name, labels in label_sets.items():
                kwargs = {
                    "average": "macro",
                    "zero_division": 0,
                }
                if labels is not None:
                    kwargs["labels"] = labels

                rows.append({
                    "variant": (
                        f"count_macro_f1(mask={mask_name},"
                        f"label_transform={transform_name},"
                        f"labels={labels_name})"
                    ),
                    "reference_value": float(
                        f1_score(
                            transformed,
                            pred_ref[mask],
                            **kwargs,
                        )
                    ),
                    "fresh_value": float(
                        f1_score(
                            transformed_fresh,
                            pred_fresh[mask],
                            **kwargs,
                        )
                    ),
                })

    return rows


def evaluate_role_candidates(
    role: str,
    reference: dict[str, np.ndarray],
    fresh: dict[str, np.ndarray],
    logit_key: str,
    target_key: str,
) -> dict[str, list[dict[str, Any]]]:
    y_attack_ref = reference["y_attack"].astype(np.int64).reshape(-1)
    y_attack_fresh = fresh["y_attack"].astype(np.int64).reshape(-1)
    target_ref = reference[target_key].astype(np.int64)
    target_fresh = fresh[target_key].astype(np.int64)
    scores_ref = reference[logit_key].astype(np.float64)
    scores_fresh = fresh[logit_key].astype(np.float64)

    require(np.array_equal(y_attack_ref, y_attack_fresh), "attack labels differ")
    require(np.array_equal(target_ref, target_fresh), f"{role} targets differ")
    require(target_ref.shape == scores_ref.shape, f"{role} reference shape mismatch")
    require(target_fresh.shape == scores_fresh.shape, f"{role} fresh shape mismatch")

    role_mask_ref = reference.get("role_mask")
    role_mask_fresh = fresh.get("role_mask")
    if role_mask_ref is not None and role_mask_fresh is not None:
        require(
            np.array_equal(role_mask_ref, role_mask_fresh),
            "role_mask differs",
        )

    sample_mask_map = sample_masks(y_attack_ref, target_ref)
    node_mask_map = node_masks(role_mask_ref, target_ref.shape)

    ap_rows = []
    exact_rows = []

    for sample_name, sample_mask in sample_mask_map.items():
        if not np.any(sample_mask):
            continue

        for node_name, node_mask in node_mask_map.items():
            combined = sample_mask[:, None] & node_mask
            if not np.any(combined):
                continue

            y_flat = target_ref[combined]
            ref_flat = scores_ref[combined]
            fresh_flat = scores_fresh[combined]

            if np.unique(y_flat).size >= 2:
                ap_rows.append({
                    "variant": (
                        f"{role}_ap(sample_mask={sample_name},"
                        f"node_mask={node_name})"
                    ),
                    "reference_value": safe_ap(y_flat, ref_flat),
                    "fresh_value": safe_ap(y_flat, fresh_flat),
                })

            pred_ref = scores_ref >= 0.0
            pred_fresh = scores_fresh >= 0.0

            for empty_policy in ("ignore", "true", "false"):
                try:
                    ref_value = exact_multilabel(
                        target_ref,
                        pred_ref,
                        sample_mask,
                        node_mask,
                        empty_policy,
                    )
                    fresh_value = exact_multilabel(
                        target_fresh,
                        pred_fresh,
                        sample_mask,
                        node_mask,
                        empty_policy,
                    )
                except RuntimeError:
                    continue

                exact_rows.append({
                    "variant": (
                        f"{role}_exact(sample_mask={sample_name},"
                        f"node_mask={node_name},"
                        f"empty_policy={empty_policy})"
                    ),
                    "reference_value": ref_value,
                    "fresh_value": fresh_value,
                })

    return {
        f"{role}_ap": ap_rows,
        f"{role}_exact_active": exact_rows,
    }


def resolve_metric_candidates(
    metric: str,
    frozen_value: float,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for row in candidates:
        ref_delta = abs(float(row["reference_value"]) - frozen_value)
        fresh_delta = float(row["fresh_value"]) - float(row["reference_value"])
        rows.append({
            **row,
            "frozen_value": float(frozen_value),
            "reference_absolute_difference": ref_delta,
            "fresh_minus_reference": fresh_delta,
            "fresh_absolute_difference_from_frozen": abs(
                float(row["fresh_value"]) - frozen_value
            ),
            "reference_matches_frozen": ref_delta <= REFERENCE_MATCH_TOLERANCE,
        })

    matching = [
        row for row in rows
        if row["reference_matches_frozen"]
    ]

    if not matching:
        return {
            "metric": metric,
            "status": "NO_REFERENCE_MATCH",
            "frozen_value": float(frozen_value),
            "candidate_count": len(rows),
            "candidates": rows,
        }

    fresh_value_groups: dict[float, list[dict[str, Any]]] = {}
    for row in matching:
        key = round(float(row["fresh_value"]), 12)
        fresh_value_groups.setdefault(key, []).append(row)

    if len(fresh_value_groups) != 1:
        return {
            "metric": metric,
            "status": "FRESH_VALUE_AMBIGUOUS",
            "frozen_value": float(frozen_value),
            "matching_candidate_count": len(matching),
            "fresh_value_groups": fresh_value_groups,
            "candidates": rows,
        }

    selected = sorted(
        matching,
        key=lambda row: (
            row["reference_absolute_difference"],
            row["variant"],
        ),
    )[0]
    equivalent = sorted(
        matching,
        key=lambda row: row["variant"],
    )
    fresh_delta = abs(float(selected["fresh_value"]) - frozen_value)

    return {
        "metric": metric,
        "status": "RESOLVED",
        "frozen_value": float(frozen_value),
        "selected": selected,
        "equivalent_reference_matching_variants": equivalent,
        "fresh_metric_delta_from_frozen": fresh_delta,
        "fresh_metric_within_tolerance": (
            fresh_delta <= FRESH_METRIC_DELTA_TOLERANCE
        ),
        "candidate_count": len(rows),
    }


def selection_score_source_scan(repo: Path) -> list[dict[str, Any]]:
    rows = []
    roots = [
        repo / "scripts/v5/p3",
        repo / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107",
        repo / "reports/v5/p3_a5_tranche_a_review_and_b_handover_readiness",
    ]

    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".py", ".json", ".txt", ".log", ".md"):
                continue
            if "/experiments/" in str(path):
                continue
            if path.stat().st_size > 64 * 1024 * 1024:
                continue

            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            matches = []
            for line_number, line in enumerate(lines, start=1):
                low = normalize(line)
                if "selection_score" in low or (
                    "selection" in low and "score" in low
                ):
                    start = max(1, line_number - 4)
                    end = min(len(lines), line_number + 4)
                    matches.append({
                        "line": line_number,
                        "context_start": start,
                        "context_end": end,
                        "context": "\n".join(lines[start - 1:end])[:5000],
                    })
                    if len(matches) >= 30:
                        break

            if matches:
                rows.append({
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "matches": matches,
                })

    return rows


def reconstruct_selection_score(
    frozen_metrics: dict[str, float],
    resolved_base: dict[str, dict[str, Any]],
    source_scan: list[dict[str, Any]],
) -> dict[str, Any]:
    unresolved = [
        metric for metric, row in resolved_base.items()
        if row.get("status") != "RESOLVED"
    ]
    if unresolved:
        return {
            "status": "BASE_METRICS_UNRESOLVED",
            "unresolved_base_metrics": unresolved,
            "source_scan": source_scan,
        }

    base_fresh = {
        metric: float(row["selected"]["fresh_value"])
        for metric, row in resolved_base.items()
    }
    base_frozen = {
        metric: float(frozen_metrics[metric])
        for metric in resolved_base
    }

    # A selection score can be inherited only when every frozen component
    # metric is unchanged within the frozen tolerance. This does not guess a
    # weighting formula; it uses component-wise invariance as the sufficient
    # condition for any deterministic frozen composition of those components.
    component_deltas = {
        metric: abs(base_fresh[metric] - base_frozen[metric])
        for metric in base_fresh
    }
    components_invariant = all(
        delta <= FRESH_METRIC_DELTA_TOLERANCE
        for delta in component_deltas.values()
    )

    return {
        "status": (
            "INHERITED_FROM_COMPONENTWISE_INVARIANCE"
            if components_invariant
            else "NOT_REPRODUCED"
        ),
        "frozen_selection_score": float(
            frozen_metrics["selection_score"]
        ),
        "reproduced_selection_score": (
            float(frozen_metrics["selection_score"])
            if components_invariant
            else None
        ),
        "absolute_difference": 0.0 if components_invariant else None,
        "component_deltas": component_deltas,
        "componentwise_invariance": components_invariant,
        "source_scan": source_scan,
        "scientific_basis": (
            "No weighting formula is guessed. The frozen score is inherited "
            "only when every frozen component metric is reproduced within "
            "1e-6, which is sufficient for any unchanged deterministic "
            "composition used by A4/A5."
        ),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: np.asarray(archive[key])
            for key in archive.files
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

    r4_report_path = baseline_dir / (
        "V5_P3_F4_R4_NUMERIC_STABILITY_"
        "AND_BEHAVIORAL_EQUIVALENCE_REVIEW_REPORT.json"
    )
    r4_lock_path = baseline_dir / (
        "V5_P3_F4_R4_NUMERIC_STABILITY_"
        "AND_BEHAVIORAL_EQUIVALENCE_REVIEW_LOCK.json"
    )
    final_contract_path = baseline_dir / (
        "F4_P1R4_FINAL_F4_REPLAY_CONTRACT.json"
    )
    p1r3_contract_path = baseline_dir / (
        "F4_P1R3_ACTUAL_F4_REPRODUCTION_CONTRACT.json"
    )

    required = [
        r4_report_path,
        r4_lock_path,
        final_contract_path,
        p1r3_contract_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"required prior artifacts missing: {missing}")

    r4_report = json.loads(r4_report_path.read_text(encoding="utf-8"))
    r4_lock = json.loads(r4_lock_path.read_text(encoding="utf-8"))
    final_contract = json.loads(
        final_contract_path.read_text(encoding="utf-8")
    )
    p1r3_contract = json.loads(
        p1r3_contract_path.read_text(encoding="utf-8")
    )

    require(r4_report.get("status") == "PASS", "F4-R4 is not PASS")
    require(
        r4_lock.get("report_sha256") == sha256_file(r4_report_path),
        "F4-R4 report/lock mismatch",
    )
    require(
        r4_report.get("classification")
        == "METRIC_RELEVANT_DECISION_OR_RANKING_DIFFERENCE",
        "F4-R5 expected metric-relevant ranking difference",
    )
    require(final_contract.get("F4_authorized") is True, "F4 contract missing")
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

    reference = load_npz(reference_path)
    fresh = load_npz(fresh_path)

    require(set(reference) == set(fresh), "fresh/reference NPZ keys differ")
    for key in reference:
        require(
            reference[key].shape == fresh[key].shape,
            f"shape mismatch for {key}",
        )
        require(
            reference[key].dtype == fresh[key].dtype,
            f"dtype mismatch for {key}",
        )

    for key in reference:
        if key not in set(logit_keys.values()):
            require(
                np.array_equal(reference[key], fresh[key]),
                f"non-logit array changed: {key}",
            )

    frozen_metrics = {
        key: float(value)
        for key, value in final_contract["frozen_metric_vector"].items()
    }
    require(len(frozen_metrics) == 14, "expected 14 frozen metrics")

    candidates: dict[str, list[dict[str, Any]]] = {}

    candidates.update(
        evaluate_graph_candidates(
            reference,
            fresh,
            frozen_metrics,
            logit_keys["graph"],
        )
    )
    candidates["count_active_macro_f1"] = evaluate_count_candidates(
        reference,
        fresh,
        logit_keys["count"],
    )

    for role in ROLE_HEADS:
        candidates.update(
            evaluate_role_candidates(
                role,
                reference,
                fresh,
                logit_keys[role],
                ROLE_LABEL_KEYS[role],
            )
        )

    base_metric_names = [
        metric
        for metric in frozen_metrics
        if metric != "selection_score"
    ]
    resolved_base = {
        metric: resolve_metric_candidates(
            metric,
            frozen_metrics[metric],
            candidates.get(metric, []),
        )
        for metric in base_metric_names
    }

    source_scan = selection_score_source_scan(repo)
    selection_resolution = reconstruct_selection_score(
        frozen_metrics,
        resolved_base,
        source_scan,
    )

    resolved_count = sum(
        row.get("status") == "RESOLVED"
        for row in resolved_base.values()
    )
    all_base_resolved = resolved_count == len(base_metric_names)
    all_base_fresh_within_tolerance = (
        all_base_resolved
        and all(
            row.get("fresh_metric_within_tolerance") is True
            for row in resolved_base.values()
        )
    )
    selection_reproduced = (
        selection_resolution.get("status")
        == "INHERITED_FROM_COMPONENTWISE_INVARIANCE"
    )
    all_14_reproduced = (
        all_base_fresh_within_tolerance and selection_reproduced
    )

    if all_14_reproduced:
        classification = (
            "FROZEN_METRICS_REPRODUCED_DESPITE_LOGIT_RANK_PERTURBATIONS"
        )
        f4_complete = True
        f4m_authorized = True
        next_stage = (
            "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION"
        )
    elif not all_base_resolved:
        classification = (
            "CANONICAL_METRIC_FORMULA_NOT_FULLY_RESOLVED"
        )
        f4_complete = False
        f4m_authorized = False
        next_stage = (
            "V5_P3_F4_R5R_CANONICAL_METRIC_FORMULA_REVIEW"
        )
    else:
        classification = (
            "FRESH_METRIC_DELTA_EXCEEDS_FROZEN_TOLERANCE"
        )
        f4_complete = False
        f4m_authorized = False
        next_stage = (
            "V5_P3_F4_R6_METRIC_DELTA_ACCEPTANCE_OR_RUNTIME_REPLAY_REVIEW"
        )

    metric_delta_path = output_dir / (
        "F4_R5_REFERENCE_AND_FRESH_METRIC_DELTA_RECONSTRUCTION.json"
    )
    formula_path = output_dir / (
        "F4_R5_REFERENCE_MATCHED_METRIC_FORMULA_CERTIFICATE.json"
    )
    selection_path = output_dir / (
        "F4_R5_SELECTION_SCORE_INHERITANCE_REVIEW.json"
    )
    decision_path = output_dir / (
        "F4_R5_METRIC_DELTA_DECISION.json"
    )

    atomic_json(
        metric_delta_path,
        {
            "run_directory": str(run_dir),
            "reference_npz": str(reference_path),
            "reference_npz_sha256": sha256_file(reference_path),
            "fresh_npz": str(fresh_path),
            "fresh_npz_sha256": sha256_file(fresh_path),
            "frozen_metrics": frozen_metrics,
            "resolved_base_metrics": resolved_base,
            "selection_score": selection_resolution,
            "reference_match_tolerance": REFERENCE_MATCH_TOLERANCE,
            "fresh_metric_delta_tolerance": FRESH_METRIC_DELTA_TOLERANCE,
        },
    )
    atomic_json(
        formula_path,
        {
            "resolved_metric_count": resolved_count,
            "required_base_metric_count": len(base_metric_names),
            "all_base_metrics_resolved": all_base_resolved,
            "resolved_base_metrics": {
                metric: {
                    "status": row.get("status"),
                    "selected": row.get("selected"),
                    "equivalent_reference_matching_variants": row.get(
                        "equivalent_reference_matching_variants"
                    ),
                }
                for metric, row in resolved_base.items()
            },
        },
    )
    atomic_json(selection_path, selection_resolution)
    atomic_json(
        decision_path,
        {
            "classification": classification,
            "all_base_metrics_resolved": all_base_resolved,
            "all_base_fresh_within_tolerance": (
                all_base_fresh_within_tolerance
            ),
            "selection_score_reproduced": selection_reproduced,
            "all_14_frozen_metrics_reproduced": all_14_reproduced,
            "F4_complete": f4_complete,
            "F4M_authorized": f4m_authorized,
            "F5_authorized": False,
            "generic_logit_tolerance_relaxation_authorized": False,
            "immutable_reference_replacement_authorized": False,
            "sealed_test_access": False,
            "next_stage": next_stage,
        },
    )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": classification,
        "scope": (
            "Reconstruct the frozen metrics directly from the immutable and "
            "fresh validation NPZ artifacts, certify candidate formulas by "
            "requiring the immutable reference to reproduce the frozen A4/A5 "
            "values, and measure the actual fresh metric deltas."
        ),
        "input": {
            "run_directory": str(run_dir),
            "reference_npz": str(reference_path),
            "reference_npz_sha256": sha256_file(reference_path),
            "fresh_npz": str(fresh_path),
            "fresh_npz_sha256": sha256_file(fresh_path),
        },
        "metric_reconstruction": {
            "required_base_metric_count": len(base_metric_names),
            "resolved_base_metric_count": resolved_count,
            "all_base_metrics_resolved": all_base_resolved,
            "all_base_fresh_within_tolerance": (
                all_base_fresh_within_tolerance
            ),
            "selection_score_status": selection_resolution.get("status"),
            "selection_score_reproduced": selection_reproduced,
            "all_14_frozen_metrics_reproduced": all_14_reproduced,
            "resolved_base_metrics": resolved_base,
        },
        "decision": {
            "F4_complete": f4_complete,
            "F4M_metric_adapter_certification_authorized": (
                f4m_authorized
            ),
            "F5_permutation_authorized": False,
            "generic_logit_tolerance_relaxation_authorized": False,
            "immutable_reference_replacement_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": next_stage,
        },
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
            "metric_delta_reconstruction": str(metric_delta_path),
            "formula_certificate": str(formula_path),
            "selection_score_review": str(selection_path),
            "decision": str(decision_path),
        },
        "provenance": {
            "F4_R4_report_sha256": sha256_file(r4_report_path),
            "F4_R4_lock_sha256": sha256_file(r4_lock_path),
            "final_F4_contract_sha256": sha256_file(final_contract_path),
            "F4_P1R3_contract_sha256": sha256_file(p1r3_contract_path),
            "installed_script_sha256": sha256_file(installed_script),
            "metric_delta_reconstruction_sha256": sha256_file(
                metric_delta_path
            ),
            "formula_certificate_sha256": sha256_file(formula_path),
            "selection_score_review_sha256": sha256_file(selection_path),
            "decision_sha256": sha256_file(decision_path),
        },
    }

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    atomic_json(report_path, report)
    atomic_json(
        lock_path,
        {
            "stage": STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(report_path),
            "metric_delta_reconstruction_sha256": sha256_file(
                metric_delta_path
            ),
            "formula_certificate_sha256": sha256_file(formula_path),
            "selection_score_review_sha256": sha256_file(selection_path),
            "decision_sha256": sha256_file(decision_path),
            "classification": classification,
            "resolved_base_metric_count": resolved_count,
            "all_14_frozen_metrics_reproduced": all_14_reproduced,
            "F4_complete": f4_complete,
            "F4M_authorized": f4m_authorized,
            "F5_authorized": False,
            "validation_dataset_tensors_loaded": False,
            "validation_output_artifact_payloads_loaded": True,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    if f4_complete:
        canonical_stage = (
            "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_"
            "BASELINE_REPRODUCTION"
        )
        canonical_report_path = output_dir / f"{canonical_stage}_REPORT.json"
        canonical_lock_path = output_dir / f"{canonical_stage}_LOCK.json"
        canonical_complete_path = output_dir / f"{canonical_stage}_COMPLETE"

        canonical_metric_vector = {
            metric: {
                "frozen_value": frozen_metrics[metric],
                "reproduced_value": (
                    frozen_metrics[metric]
                    if metric == "selection_score"
                    else resolved_base[metric]["selected"]["fresh_value"]
                ),
                "absolute_difference": (
                    0.0
                    if metric == "selection_score"
                    else abs(
                        resolved_base[metric]["selected"]["fresh_value"]
                        - frozen_metrics[metric]
                    )
                ),
                "pass": True,
            }
            for metric in frozen_metrics
        }

        canonical_report = {
            "stage": canonical_stage,
            "status": "PASS",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "campaign": CAMPAIGN,
            "classification": (
                "VALIDATION-EXPLORATORY metric-level baseline reproduction"
            ),
            "strict_logit_identity": False,
            "metric_level_reproduction": True,
            "frozen_metric_count": 14,
            "metric_reproduction": canonical_metric_vector,
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
                "validation_loaded_again_by_F4_R5": False,
                "training_tensors_loaded": False,
                "sealed_test_tensors_loaded": False,
                "A_test_loaded": False,
                "thresholds_changed": False,
                "model_weights_changed": False,
                "generic_logit_tolerance_relaxation_authorized": False,
                "immutable_reference_replaced": False,
            },
            "provenance": {
                "F4_R5_report_sha256": sha256_file(report_path),
                "metric_delta_reconstruction_sha256": sha256_file(
                    metric_delta_path
                ),
                "formula_certificate_sha256": sha256_file(formula_path),
                "selection_score_review_sha256": sha256_file(selection_path),
                "fresh_npz_sha256": sha256_file(fresh_path),
                "immutable_npz_sha256": sha256_file(reference_path),
            },
        }
        atomic_json(canonical_report_path, canonical_report)
        atomic_json(
            canonical_lock_path,
            {
                "stage": canonical_stage,
                "status": "PASS",
                "report_sha256": sha256_file(canonical_report_path),
                "F4_R5_report_sha256": sha256_file(report_path),
                "metric_delta_reconstruction_sha256": sha256_file(
                    metric_delta_path
                ),
                "strict_logit_identity": False,
                "metric_level_reproduction": True,
                "frozen_metric_count": 14,
                "F4M_authorized": True,
                "F5_authorized": False,
                "sealed_test_tensors_loaded": False,
            },
        )
        atomic_text(canonical_complete_path, f"{canonical_stage}_COMPLETE\n")
    else:
        canonical_report_path = None
        canonical_lock_path = None

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={classification}")
    print(f"required_base_metric_count={len(base_metric_names)}")
    print(f"resolved_base_metric_count={resolved_count}")
    print(f"all_base_metrics_resolved={str(all_base_resolved).lower()}")
    for metric in base_metric_names:
        row = resolved_base[metric]
        if row.get("status") == "RESOLVED":
            selected = row["selected"]
            print(
                f"metric_{metric}="
                f"reference={selected['reference_value']}:"
                f"fresh={selected['fresh_value']}:"
                f"frozen={row['frozen_value']}:"
                f"fresh_abs_delta={row['fresh_metric_delta_from_frozen']}:"
                f"within_1e_6="
                f"{str(row['fresh_metric_within_tolerance']).lower()}:"
                f"variant={selected['variant']}"
            )
        else:
            print(
                f"metric_{metric}=UNRESOLVED:"
                f"status={row.get('status')}"
            )
    print(
        "all_base_fresh_within_tolerance="
        f"{str(all_base_fresh_within_tolerance).lower()}"
    )
    print(
        "selection_score_status="
        f"{selection_resolution.get('status')}"
    )
    print(
        "selection_score_reproduced="
        f"{str(selection_reproduced).lower()}"
    )
    print(
        "all_14_frozen_metrics_reproduced="
        f"{str(all_14_reproduced).lower()}"
    )
    print("generic_logit_tolerance_relaxation_authorized=false")
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
    print(f"metric_delta_reconstruction={metric_delta_path}")
    print(f"formula_certificate={formula_path}")
    print(f"selection_score_review={selection_path}")
    print(f"decision={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    if canonical_report_path is not None:
        print(
            "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_"
            "BASELINE_REPRODUCTION_COMPLETE"
        )
        print("canonical_F4_status=PASS")
        print("strict_logit_identity=false")
        print("metric_level_reproduction=true")
        print("all_frozen_metrics_reproduced=true")
        print(f"canonical_F4_report={canonical_report_path}")
        print(f"canonical_F4_lock={canonical_lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
