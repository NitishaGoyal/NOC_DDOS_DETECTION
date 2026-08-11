from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = (
    "V5_P3_D10_RAW_A0_A1_EPLR_V1_COMPARISON_"
    "AND_CARRY_FORWARD_DECISION"
)
CAMPAIGN_LABEL = (
    "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic "
    "— EPLR-V1 Comparison and Carry-Forward Review"
)
TOTAL_ITEMS = 13_863
REFERENCE_TOLERANCE = 5e-7

STATUS = {
    0: "INACTIVE_RAW",
    1: "RAW_ENDPOINTS_LEGAL",
    2: "RAW_ENDPOINTS_LEGAL_LOW_ROUTE_SUPPORT",
    3: "ENDPOINT_REPAIR_APPLIED",
    4: "NO_CERTIFIED_LEGAL_EXPLANATION",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def binary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.uint8).reshape(-1)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    accuracy = (tp + tn) / labels.size
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
    }


def macro_f1_active_count(
    labels: np.ndarray,
    predictions: np.ndarray,
    active: np.ndarray,
) -> float:
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    active = np.asarray(active, dtype=bool).reshape(-1)
    values = []
    for klass in (1, 2, 3, 4):
        y = (labels[active] == klass).astype(np.uint8)
        p = (predictions[active] == klass).astype(np.uint8)
        values.append(binary_metrics(y, p)["f1"])
    return float(np.mean(values))


def exact_active(
    labels: np.ndarray,
    predictions: np.ndarray,
    active: np.ndarray,
) -> float:
    labels = np.asarray(labels, dtype=np.uint8)
    predictions = np.asarray(predictions, dtype=np.uint8)
    active = np.asarray(active, dtype=bool).reshape(-1)
    if not np.any(active):
        return float("nan")
    return float(
        np.mean(
            np.all(
                labels[active] == predictions[active],
                axis=1,
            )
        )
    )


def strict_exact_vector(
    y_graph: np.ndarray,
    y_count: np.ndarray,
    y_source: np.ndarray,
    y_transit: np.ndarray,
    y_victim: np.ndarray,
    y_path: np.ndarray,
    p_graph: np.ndarray,
    p_count: np.ndarray,
    p_source: np.ndarray,
    p_transit: np.ndarray,
    p_victim: np.ndarray,
    p_path: np.ndarray,
) -> np.ndarray:
    y_graph = np.asarray(y_graph, dtype=np.uint8).reshape(-1)
    active = y_graph == 1
    graph_ok = np.asarray(p_graph, dtype=np.uint8).reshape(-1) == y_graph
    count_ok = np.asarray(p_count).reshape(-1) == np.asarray(
        y_count
    ).reshape(-1)
    role_ok = (
        np.all(p_source == y_source, axis=1)
        & np.all(p_transit == y_transit, axis=1)
        & np.all(p_victim == y_victim, axis=1)
        & np.all(p_path == y_path, axis=1)
    )
    inactive_roles_zero = (
        np.all(p_source == 0, axis=1)
        & np.all(p_transit == 0, axis=1)
        & np.all(p_victim == 0, axis=1)
        & np.all(p_path == 0, axis=1)
    )
    return graph_ok & np.where(
        active,
        count_ok & role_ok,
        inactive_roles_zero,
    )


def evaluate(
    y: dict[str, np.ndarray],
    graph: np.ndarray,
    count: np.ndarray,
    source: np.ndarray,
    transit: np.ndarray,
    victim: np.ndarray,
    path: np.ndarray,
) -> dict[str, Any]:
    active = y["graph"] == 1
    strict = strict_exact_vector(
        y["graph"],
        y["count"],
        y["source"],
        y["transit"],
        y["victim"],
        y["path"],
        graph,
        count,
        source,
        transit,
        victim,
        path,
    )
    return {
        "graph": binary_metrics(y["graph"], graph),
        "count_active_macro_f1": macro_f1_active_count(
            y["count"],
            count,
            active,
        ),
        "source_exact_active": exact_active(
            y["source"],
            source,
            active,
        ),
        "transit_exact_active": exact_active(
            y["transit"],
            transit,
            active,
        ),
        "victim_exact_active": exact_active(
            y["victim"],
            victim,
            active,
        ),
        "path_exact_active": exact_active(
            y["path"],
            path,
            active,
        ),
        "strict_exact": float(np.mean(strict)),
        "strict_vector": strict,
    }


def subgroup_audit(
    name: str,
    selector: np.ndarray,
    y: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    eplr: dict[str, np.ndarray],
) -> dict[str, Any]:
    selector = np.asarray(selector, dtype=bool)
    active_selector = selector & (y["graph"] == 1)
    count = int(np.sum(selector))
    active_count = int(np.sum(active_selector))

    result: dict[str, Any] = {
        "name": name,
        "items": count,
        "true_active_items": active_count,
    }
    if active_count == 0:
        result["active_metrics_available"] = False
        return result

    result["active_metrics_available"] = True
    for role in ("source", "transit", "victim", "path"):
        result[f"Raw_{role}_exact"] = exact_active(
            y[role],
            raw[role],
            active_selector,
        )
        result[f"EPLR_{role}_exact"] = exact_active(
            y[role],
            eplr[role],
            active_selector,
        )
        result[f"EPLR_minus_Raw_{role}_exact"] = (
            result[f"EPLR_{role}_exact"]
            - result[f"Raw_{role}_exact"]
        )

    raw_joint = (
        (raw["count"] == y["count"])
        & np.all(raw["source"] == y["source"], axis=1)
        & np.all(raw["transit"] == y["transit"], axis=1)
        & np.all(raw["victim"] == y["victim"], axis=1)
        & np.all(raw["path"] == y["path"], axis=1)
    )
    eplr_joint = (
        (eplr["count"] == y["count"])
        & np.all(eplr["source"] == y["source"], axis=1)
        & np.all(eplr["transit"] == y["transit"], axis=1)
        & np.all(eplr["victim"] == y["victim"], axis=1)
        & np.all(eplr["path"] == y["path"], axis=1)
    )
    result["Raw_active_joint_exact"] = float(
        np.mean(raw_joint[active_selector])
    )
    result["EPLR_active_joint_exact"] = float(
        np.mean(eplr_joint[active_selector])
    )
    result["EPLR_minus_Raw_active_joint_exact"] = (
        result["EPLR_active_joint_exact"]
        - result["Raw_active_joint_exact"]
    )
    return result


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    comparison_path = output_dir / (
        "V5_P3_D10_RAW_A0_A1_EPLR_V1_COMPARISON.json"
    )
    subgroup_path = output_dir / (
        "V5_P3_D10_EPLR_V1_STATUS_AND_REPAIR_ROOT_CAUSE_AUDIT.json"
    )
    decision_note_path = output_dir / (
        "V5_P3_D10_EPLR_V1_CARRY_FORWARD_DECISION.md"
    )
    v2_note_path = output_dir / (
        "V5_P3_D10_EPLR_V2_PRESERVE_OR_PASSTHROUGH_DESIGN_INPUT.md"
    )
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    d9_dir = (
        repo
        / "reports/v5/p3_d9_eplr_v1_one_shot_validation_replay"
    )
    d9_report_path = d9_dir / (
        "V5_P3_D9_EPLR_V1_ONE_SHOT_TRANCHE_A_"
        "VALIDATION_EXPLORATORY_REPLAY_REPORT.json"
    )
    d9_lock_path = d9_dir / (
        "V5_P3_D9_EPLR_V1_ONE_SHOT_TRANCHE_A_"
        "VALIDATION_EXPLORATORY_REPLAY_LOCK.json"
    )
    d9_merged_path = d9_dir / (
        "V5_P3_D9_EPLR_V1_COMPLETE_VALIDATION_OUTPUTS.npz"
    )

    d1_dir = (
        repo
        / "reports/v5/p3_d1_immutable_validation_logit_export"
    )
    d1_export_path = d1_dir / (
        "V5_P3_D1_TRANCHE_A_VALIDATION_LOGITS_AND_LABELS.npz"
    )
    d1_lock_path = d1_dir / (
        "V5_P3_D1_IMMUTABLE_TRANCHE_A_VALIDATION_LOGIT_EXPORT_LOCK.json"
    )

    r3_dir = (
        repo
        / "reports/v5/p3_d4_r3_resume_a1_exact_validation_hybrid"
    )
    r3_report_path = r3_dir / (
        "V5_P3_D4_R3_RESUME_A1_EXACT_VALIDATION_WITH_"
        "FROZEN_CERTIFIED_HYBRID_DECODER_REPORT.json"
    )
    r3_lock_path = r3_dir / (
        "V5_P3_D4_R3_RESUME_A1_EXACT_VALIDATION_WITH_"
        "FROZEN_CERTIFIED_HYBRID_DECODER_LOCK.json"
    )

    d8_dir = (
        repo
        / "reports/v5/p3_d8_eplr_v1_implementation_and_train_calibration"
    )
    d8_report_path = d8_dir / (
        "V5_P3_D8_EPLR_V1_DETERMINISTIC_IMPLEMENTATION_"
        "AND_A_TRAIN_ONLY_CALIBRATION_REPORT.json"
    )
    d8_lock_path = d8_dir / (
        "V5_P3_D8_EPLR_V1_DETERMINISTIC_IMPLEMENTATION_"
        "AND_A_TRAIN_ONLY_CALIBRATION_LOCK.json"
    )

    decoder_path = repo / "src/decoders/v5_p3_eplr_v1.py"
    config_path = (
        repo / "src/decoders/v5_p3_eplr_v1_calibration.json"
    )

    required = [
        d9_report_path,
        d9_lock_path,
        d9_merged_path,
        d1_export_path,
        d1_lock_path,
        r3_report_path,
        r3_lock_path,
        d8_report_path,
        d8_lock_path,
        decoder_path,
        config_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")

    d9 = json.loads(d9_report_path.read_text(encoding="utf-8"))
    d9_lock = json.loads(d9_lock_path.read_text(encoding="utf-8"))
    d1_lock = json.loads(d1_lock_path.read_text(encoding="utf-8"))
    r3 = json.loads(r3_report_path.read_text(encoding="utf-8"))
    r3_lock = json.loads(r3_lock_path.read_text(encoding="utf-8"))
    d8 = json.loads(d8_report_path.read_text(encoding="utf-8"))
    d8_lock = json.loads(d8_lock_path.read_text(encoding="utf-8"))

    if d9.get("status") != "PASS":
        raise RuntimeError("D9 is not PASS")
    if d9_lock.get("report_sha256") != sha256_file(d9_report_path):
        raise RuntimeError("D9 report/lock mismatch")
    if d9_lock.get("merged_outputs_sha256") != sha256_file(
        d9_merged_path
    ):
        raise RuntimeError("D9 merged output/lock mismatch")
    if d1_lock.get("export_sha256") != sha256_file(d1_export_path):
        raise RuntimeError("D1 export/lock mismatch")
    if r3.get("status") != "PASS":
        raise RuntimeError("R3 is not PASS")
    if r3_lock.get("report_sha256") != sha256_file(r3_report_path):
        raise RuntimeError("R3 report/lock mismatch")
    if d8.get("status") != "PASS":
        raise RuntimeError("D8 is not PASS")
    if d8_lock.get("report_sha256") != sha256_file(d8_report_path):
        raise RuntimeError("D8 report/lock mismatch")
    if d8_lock.get("decoder_sha256") != sha256_file(decoder_path):
        raise RuntimeError("decoder differs from D8 lock")
    if d8_lock.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("config differs from D8 lock")
    if d9["acceptance_gate"].get("status") != "FAIL":
        raise RuntimeError(
            "D10 package is scoped to the observed EPLR-V1 gate failure"
        )

    with np.load(d1_export_path, allow_pickle=False) as loaded:
        d1 = {key: loaded[key].copy() for key in loaded.files}
    with np.load(d9_merged_path, allow_pickle=False) as loaded:
        merged = {
            key: loaded[key].copy()
            for key in loaded.files
        }

    if d1["attack_logits"].shape[0] != TOTAL_ITEMS:
        raise RuntimeError("D1 item count changed")
    if merged["dataset_index"].shape[0] != TOTAL_ITEMS:
        raise RuntimeError("D9 item count changed")
    if not np.array_equal(
        merged["dataset_index"],
        np.arange(TOTAL_ITEMS),
    ):
        raise RuntimeError("D9 dataset order changed")

    y = {
        "graph": d1["y_attack"].astype(np.uint8),
        "count": d1["y_attacker_count"].astype(np.int8),
        "source": d1["y_source"].astype(np.uint8),
        "transit": d1["y_transit"].astype(np.uint8),
        "victim": d1["y_victim"].astype(np.uint8),
        "path": d1["y_attack_path"].astype(np.uint8),
    }
    raw = {
        "graph": (
            stable_sigmoid(d1["attack_logits"]) >= 0.5
        ).astype(np.uint8),
        "count": (
            np.argmax(d1["count_logits"], axis=1).astype(np.int8) + 1
        ),
        "source": (
            stable_sigmoid(d1["source_logits"]) >= 0.5
        ).astype(np.uint8),
        "transit": (
            stable_sigmoid(d1["transit_logits"]) >= 0.5
        ).astype(np.uint8),
        "victim": (
            stable_sigmoid(d1["victim_logits"]) >= 0.5
        ).astype(np.uint8),
        "path": (
            stable_sigmoid(d1["path_logits"]) >= 0.5
        ).astype(np.uint8),
    }
    eplr = {
        "graph": merged["raw_graph_prediction"].astype(np.uint8),
        "count": merged["raw_count_candidate"].astype(np.int8),
        "source": merged["decoded_source"].astype(np.uint8),
        "transit": merged["decoded_transit"].astype(np.uint8),
        "victim": merged["decoded_victim"].astype(np.uint8),
        "path": merged["decoded_path"].astype(np.uint8),
    }

    if not np.array_equal(eplr["graph"], raw["graph"]):
        raise RuntimeError("EPLR graph no longer equals Raw")
    if not np.array_equal(eplr["count"], raw["count"]):
        raise RuntimeError("EPLR count no longer equals Raw")

    raw_eval = evaluate(
        y,
        raw["graph"],
        raw["count"],
        raw["source"],
        raw["transit"],
        raw["victim"],
        raw["path"],
    )
    eplr_eval = evaluate(
        y,
        eplr["graph"],
        eplr["count"],
        eplr["source"],
        eplr["transit"],
        eplr["victim"],
        eplr["path"],
    )

    # Diagnostic only: preserve legal EPLR transit/path cleanup, but pass
    # through all Raw role masks when endpoint preservation was infeasible.
    preserved_selector = np.isin(
        merged["status_code"],
        np.asarray([1, 2], dtype=np.uint8),
    )
    repair_or_failure_selector = np.isin(
        merged["status_code"],
        np.asarray([3, 4], dtype=np.uint8),
    )

    v2_diag = {
        "graph": raw["graph"].copy(),
        "count": raw["count"].copy(),
        "source": raw["source"].copy(),
        "transit": raw["transit"].copy(),
        "victim": raw["victim"].copy(),
        "path": raw["path"].copy(),
    }
    v2_diag["transit"][preserved_selector] = eplr["transit"][
        preserved_selector
    ]
    v2_diag["path"][preserved_selector] = eplr["path"][
        preserved_selector
    ]

    v2_diag_eval = evaluate(
        y,
        v2_diag["graph"],
        v2_diag["count"],
        v2_diag["source"],
        v2_diag["transit"],
        v2_diag["victim"],
        v2_diag["path"],
    )

    def strip_vector(metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in metrics.items()
            if key != "strict_vector"
        }

    raw_metrics = strip_vector(raw_eval)
    eplr_metrics = strip_vector(eplr_eval)
    v2_diag_metrics = strip_vector(v2_diag_eval)

    # Reproduce D9 load-bearing metrics exactly.
    d9_raw = d9["metrics"]["Raw"]
    d9_eplr = d9["metrics"]["EPLR_V1"]
    reproduction = {
        "Raw_source_exact_delta": (
            raw_metrics["source_exact_active"]
            - float(d9_raw["source_exact_active"])
        ),
        "Raw_transit_exact_delta": (
            raw_metrics["transit_exact_active"]
            - float(d9_raw["transit_exact_active"])
        ),
        "Raw_victim_exact_delta": (
            raw_metrics["victim_exact_active"]
            - float(d9_raw["victim_exact_active"])
        ),
        "Raw_path_exact_delta": (
            raw_metrics["path_exact_active"]
            - float(d9_raw["path_exact_active"])
        ),
        "Raw_strict_exact_delta": (
            raw_metrics["strict_exact"]
            - float(d9_raw["strict_exact"])
        ),
        "EPLR_source_exact_delta": (
            eplr_metrics["source_exact_active"]
            - float(d9_eplr["source_exact_active"])
        ),
        "EPLR_transit_exact_delta": (
            eplr_metrics["transit_exact_active"]
            - float(d9_eplr["transit_exact_active"])
        ),
        "EPLR_victim_exact_delta": (
            eplr_metrics["victim_exact_active"]
            - float(d9_eplr["victim_exact_active"])
        ),
        "EPLR_path_exact_delta": (
            eplr_metrics["path_exact_active"]
            - float(d9_eplr["path_exact_active"])
        ),
        "EPLR_strict_exact_delta": (
            eplr_metrics["strict_exact"]
            - float(d9_eplr["strict_exact"])
        ),
    }
    max_reproduction_delta = max(
        abs(value) for value in reproduction.values()
    )
    if max_reproduction_delta > REFERENCE_TOLERANCE:
        raise RuntimeError(
            "D9 metric reproduction exceeded tolerance: "
            f"{max_reproduction_delta}"
        )

    status_code = merged["status_code"].astype(np.uint8)
    subgroup_rows = []
    for code, name in STATUS.items():
        subgroup_rows.append(
            subgroup_audit(
                name,
                status_code == code,
                y,
                raw,
                eplr,
            )
        )
    subgroup_rows.extend(
        [
            subgroup_audit(
                "ENDPOINT_PRESERVED_COMBINED",
                preserved_selector,
                y,
                raw,
                eplr,
            ),
            subgroup_audit(
                "REPAIR_OR_NO_SOLUTION_COMBINED",
                repair_or_failure_selector,
                y,
                raw,
                eplr,
            ),
        ]
    )

    preserved_source_equal_raw = bool(
        np.array_equal(
            eplr["source"][preserved_selector],
            raw["source"][preserved_selector],
        )
    )
    preserved_victim_equal_raw = bool(
        np.array_equal(
            eplr["victim"][preserved_selector],
            raw["victim"][preserved_selector],
        )
    )

    comparison = {
        "classification": CAMPAIGN_LABEL,
        "Raw": raw_metrics,
        "A0": r3["metrics"]["A0"],
        "old_A1": r3["metrics"]["A1"],
        "EPLR_V1": eplr_metrics,
        "post_hoc_diagnostic_only_EPLR_V2_preserve_or_passthrough": (
            v2_diag_metrics
        ),
        "EPLR_V1_minus_Raw": {
            key: eplr_metrics[key] - raw_metrics[key]
            for key in (
                "count_active_macro_f1",
                "source_exact_active",
                "transit_exact_active",
                "victim_exact_active",
                "path_exact_active",
                "strict_exact",
            )
        },
        "V2_diagnostic_minus_Raw": {
            key: v2_diag_metrics[key] - raw_metrics[key]
            for key in (
                "count_active_macro_f1",
                "source_exact_active",
                "transit_exact_active",
                "victim_exact_active",
                "path_exact_active",
                "strict_exact",
            )
        },
        "warning": (
            "The V2 preserve-or-passthrough row is a post-hoc diagnostic "
            "constructed after observing A-validation. It is not a frozen "
            "candidate evaluation and cannot support a validation claim."
        ),
    }
    atomic_json(comparison_path, comparison)

    subgroup_audit_payload = {
        "stage": STAGE,
        "status_counts": {
            STATUS[code]: int(np.sum(status_code == code))
            for code in sorted(STATUS)
        },
        "endpoint_preserved_rows": int(
            np.sum(preserved_selector)
        ),
        "repair_or_no_solution_rows": int(
            np.sum(repair_or_failure_selector)
        ),
        "preserved_source_outputs_equal_Raw": (
            preserved_source_equal_raw
        ),
        "preserved_victim_outputs_equal_Raw": (
            preserved_victim_equal_raw
        ),
        "subgroups": subgroup_rows,
        "root_cause": (
            "Endpoint-preserving rows retain Raw source/victim predictions. "
            "The observed source/victim collapse is therefore attributable "
            "to the endpoint-repair/no-solution path, while legal route "
            "cleanup is responsible for the transit and strict-structure gain."
        ),
    }
    atomic_json(subgroup_path, subgroup_audit_payload)

    v1_strict_gain = (
        eplr_metrics["strict_exact"]
        - raw_metrics["strict_exact"]
    )
    v1_source_delta = (
        eplr_metrics["source_exact_active"]
        - raw_metrics["source_exact_active"]
    )
    v1_victim_delta = (
        eplr_metrics["victim_exact_active"]
        - raw_metrics["victim_exact_active"]
    )
    v1_path_delta = (
        eplr_metrics["path_exact_active"]
        - raw_metrics["path_exact_active"]
    )
    v1_transit_delta = (
        eplr_metrics["transit_exact_active"]
        - raw_metrics["transit_exact_active"]
    )

    decision_note = f"""# V5-P3 D10 EPLR-V1 carry-forward decision

## Classification

**{CAMPAIGN_LABEL}**

## Decision

EPLR-V1 is **not carried forward as implemented**.

It satisfied the intended immutable graph/count contract and improved:

- transit exactness by `{v1_transit_delta:+.8f}`;
- strict exactness by `{v1_strict_gain:+.8f}`.

It failed the predeclared Pareto gate because it changed endpoint/path quality:

- source exactness by `{v1_source_delta:+.8f}`;
- victim exactness by `{v1_victim_delta:+.8f}`;
- path exactness by `{v1_path_delta:+.8f}`.

The decoder and calibration must not be retuned using Tranche-A validation.

## Root-cause disposition

The endpoint-preserving branch is not the cause of the endpoint collapse:
source and victim outputs on preserved rows are bit-identical to Raw.

The damaging behavior is concentrated in:

- `ENDPOINT_REPAIR_APPLIED`;
- `NO_CERTIFIED_LEGAL_EXPLANATION`.

Therefore:

- legal route cleanup remains promising;
- endpoint reassignment is rejected for the next version;
- zero-mask fail-closed output must not be used as the prediction returned for
  metric evaluation when a legal explanation cannot be certified.

## Carry-forward

A new decoder version may be designed:

**EPLR-V2 Preserve-or-Passthrough**

- graph remains Raw;
- count remains Raw;
- source remains Raw;
- victim remains Raw;
- when Raw endpoints admit a legal explanation, legal routing may clean
  transit/path;
- otherwise all Raw role masks pass through unchanged and the decoder reports
  an unresolved diagnostic status;
- no endpoint-repair MILP is used;
- no second EPLR validation replay on Tranche-A is authorized.

A future confirmatory evaluation must use an independent future validation
boundary such as frozen Tranche-B validation or the final A+B protocol.
"""
    atomic_text(decision_note_path, decision_note)

    v2_note = """# EPLR-V2 Preserve-or-Passthrough design input

This is a D10 design input, not a frozen implementation.

## Proposed immutable outputs

- graph: Raw
- count: Raw
- source: Raw
- victim: Raw

## Feasible endpoint case

When exactly K legal XY routes preserve the Raw source and victim masks:

- select the best legal route tuple using train-frozen transit/path support;
- output route-derived transit and path union bitmaps;
- retain Raw source and victim bitmaps;
- report legal or low-support status.

## Infeasible endpoint case

Do not repair or reassign endpoints.

- pass through Raw source, transit, victim and path masks;
- report `RAW_ENDPOINTS_UNRESOLVED_PASSTHROUGH`;
- return no certified route IDs;
- do not claim that the passthrough masks form a legal route explanation.

## Governance

The D10 counterfactual metrics are post-hoc diagnostics only. They must not be
reported as a frozen A-validation result. EPLR-V2 requires a new semantic and
unit-test freeze, A-train-only implementation work, and independent future
validation. A-test remains sealed.
"""
    atomic_text(v2_note_path, v2_note)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "scope": (
            "Compare Raw, A0, old A1 and EPLR-V1, localize the EPLR-V1 "
            "failure mechanism, reject or carry forward V1, and record a "
            "governed design direction for a distinct future decoder version."
        ),
        "comparison": {
            "path": str(comparison_path),
            "sha256": sha256_file(comparison_path),
        },
        "root_cause_audit": {
            "path": str(subgroup_path),
            "sha256": sha256_file(subgroup_path),
            "preserved_source_outputs_equal_Raw": (
                preserved_source_equal_raw
            ),
            "preserved_victim_outputs_equal_Raw": (
                preserved_victim_equal_raw
            ),
            "endpoint_repair_or_failure_rows": int(
                np.sum(repair_or_failure_selector)
            ),
        },
        "scientific_decision": {
            "EPLR_V1_stage_completed_correctly": True,
            "EPLR_V1_acceptance_gate_pass": False,
            "EPLR_V1_carry_forward_as_implemented": False,
            "EPLR_V1_retuning_on_A_validation_authorized": False,
            "endpoint_preserving_route_cleanup_principle_retained": True,
            "endpoint_repair_policy_retained": False,
            "zero_mask_no_solution_prediction_retained": False,
            "EPLR_V2_preserve_or_passthrough_design_authorized": True,
            "EPLR_V2_A_validation_replay_authorized": False,
            "next_stage": (
                "V5_P3_D11_EPLR_V2_PRESERVE_OR_PASSTHROUGH_"
                "SEMANTIC_FREEZE"
            ),
        },
        "post_hoc_diagnostic": {
            "performed": True,
            "name": "EPLR-V2 preserve-or-passthrough counterfactual",
            "metrics": v2_diag_metrics,
            "claim_authorized": False,
            "candidate_selected": False,
            "purpose": (
                "Mechanism diagnosis and future-design input only."
            ),
        },
        "reference_reproduction": {
            "status": "PASS",
            "tolerance": REFERENCE_TOLERANCE,
            "max_absolute_delta": max_reproduction_delta,
            "deltas": reproduction,
        },
        "artifacts": {
            "decision_note_path": str(decision_note_path),
            "decision_note_sha256": sha256_file(decision_note_path),
            "V2_design_input_path": str(v2_note_path),
            "V2_design_input_sha256": sha256_file(v2_note_path),
        },
        "governance": {
            "decoder_changed": False,
            "config_changed": False,
            "neural_model_replayed": False,
            "A_validation_replayed": False,
            "new_validation_candidate_evaluated": False,
            "A_test_tensor_loaded": False,
            "sealed_test_evaluation_authorized": False,
            "H1_parallel_work_unaffected": True,
            "Tranche_B_parallel_work_unaffected": True,
        },
        "provenance": {
            "D9_report_sha256": sha256_file(d9_report_path),
            "D9_lock_sha256": sha256_file(d9_lock_path),
            "D9_merged_outputs_sha256": sha256_file(d9_merged_path),
            "D1_export_sha256": sha256_file(d1_export_path),
            "D1_lock_sha256": sha256_file(d1_lock_path),
            "R3_report_sha256": sha256_file(r3_report_path),
            "R3_lock_sha256": sha256_file(r3_lock_path),
            "D8_report_sha256": sha256_file(d8_report_path),
            "D8_lock_sha256": sha256_file(d8_lock_path),
            "decoder_sha256": sha256_file(decoder_path),
            "config_sha256": sha256_file(config_path),
            "installed_script_sha256": sha256_file(installed_script),
        },
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(report_path),
        "comparison_sha256": sha256_file(comparison_path),
        "root_cause_audit_sha256": sha256_file(subgroup_path),
        "decision_note_sha256": sha256_file(decision_note_path),
        "V2_design_input_sha256": sha256_file(v2_note_path),
        "EPLR_V1_acceptance_gate_pass": False,
        "EPLR_V1_carry_forward": False,
        "EPLR_V2_design_authorized": True,
        "EPLR_V2_A_validation_replay_authorized": False,
        "A_test_tensor_loaded": False,
    }
    atomic_json(lock_path, lock)
    complete_path.write_text(
        f"{STAGE}_COMPLETE\n",
        encoding="utf-8",
    )

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign_label={CAMPAIGN_LABEL}")
    print("EPLR_V1_acceptance_gate_pass=false")
    print("EPLR_V1_carry_forward_as_implemented=false")
    print("EPLR_V1_retuning_on_A_validation_authorized=false")
    print(
        "endpoint_preserving_route_cleanup_principle_retained=true"
    )
    print("endpoint_repair_policy_retained=false")
    print("zero_mask_no_solution_prediction_retained=false")
    print(
        "preserved_source_outputs_equal_Raw="
        f"{preserved_source_equal_raw}"
    )
    print(
        "preserved_victim_outputs_equal_Raw="
        f"{preserved_victim_equal_raw}"
    )
    print(
        "endpoint_repair_or_failure_rows="
        f"{int(np.sum(repair_or_failure_selector))}"
    )
    print(
        "post_hoc_V2_diagnostic_source_exact="
        f"{v2_diag_metrics['source_exact_active']:.8f}"
    )
    print(
        "post_hoc_V2_diagnostic_transit_exact="
        f"{v2_diag_metrics['transit_exact_active']:.8f}"
    )
    print(
        "post_hoc_V2_diagnostic_victim_exact="
        f"{v2_diag_metrics['victim_exact_active']:.8f}"
    )
    print(
        "post_hoc_V2_diagnostic_path_exact="
        f"{v2_diag_metrics['path_exact_active']:.8f}"
    )
    print(
        "post_hoc_V2_diagnostic_strict_exact="
        f"{v2_diag_metrics['strict_exact']:.8f}"
    )
    print("post_hoc_V2_diagnostic_claim_authorized=false")
    print("EPLR_V2_preserve_or_passthrough_design_authorized=true")
    print("EPLR_V2_A_validation_replay_authorized=false")
    print("A_test_tensor_loaded=false")
    print(
        "next_stage="
        "V5_P3_D11_EPLR_V2_PRESERVE_OR_PASSTHROUGH_"
        "SEMANTIC_FREEZE"
    )
    print(f"comparison={comparison_path}")
    print(f"root_cause_audit={subgroup_path}")
    print(f"decision_note={decision_note_path}")
    print(f"V2_design_input={v2_note_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
