from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = (
    "V5_P3_D4_R3_RESUME_A1_EXACT_VALIDATION_WITH_"
    "FROZEN_CERTIFIED_HYBRID_DECODER"
)
CAMPAIGN_LABEL = (
    "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic "
    "— Frozen P2 Decoder Transfer Evaluation"
)
TOTAL_ITEMS = 13_863
PREFIX_ITEMS = 11_700
CHUNK_SIZE = 100
A1_MARGIN_THRESHOLD = 8.7205320882398425

GRAPH_THRESHOLD_A0 = 0.47174675035328983
SOURCE_THRESHOLD_A0 = 0.94960549299285935
TRANSIT_THRESHOLD_A0 = 0.85703332488359107
VICTIM_THRESHOLD_A0 = 0.82601148026998117
PATH_THRESHOLD_A0 = 0.8339347466090468

RAW_REFERENCE = {
    "graph_accuracy": 0.80213518,
    "graph_f1": 0.70116570,
    "graph_fpr": 0.21302659,
    "strict_exact": 0.63211426,
}
A0_REFERENCE = {
    "graph_accuracy": 0.80401068,
    "graph_f1": 0.70718827,
    "graph_fpr": 0.21671148,
    "strict_exact": 0.62583856,
}
REFERENCE_TOLERANCE = 5e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def import_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
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
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)

    start = 0
    while start < scores.size:
        stop = start + 1
        while (
            stop < scores.size
            and sorted_scores[stop] == sorted_scores[start]
        ):
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop

    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    ranks = np.arange(1, labels.size + 1)
    precision = cumulative / ranks
    return float(np.sum(precision * sorted_labels) / positives)


def macro_f1_active_count(
    labels: np.ndarray,
    predictions: np.ndarray,
    active_mask: np.ndarray,
) -> float:
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    values = []
    for klass in (1, 2, 3, 4):
        y = (labels[active_mask] == klass).astype(np.uint8)
        p = (predictions[active_mask] == klass).astype(np.uint8)
        values.append(binary_metrics(y, p)["f1"])
    return float(np.mean(values))


def role_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    active_mask: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.uint8)
    predictions = np.asarray(predictions, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    exact_active = float(
        np.mean(
            np.all(
                predictions[active_mask] == labels[active_mask],
                axis=1,
            )
        )
    )
    return {
        "average_precision_all_router_labels": average_precision(
            labels.reshape(-1),
            scores.reshape(-1),
        ),
        "exact_active": exact_active,
    }


def strict_exact(
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

    role_ok = (
        np.all(p_source == y_source, axis=1)
        & np.all(p_transit == y_transit, axis=1)
        & np.all(p_victim == y_victim, axis=1)
        & np.all(p_path == y_path, axis=1)
    )
    count_ok = np.asarray(p_count).reshape(-1) == np.asarray(
        y_count
    ).reshape(-1)

    # Count is defined only for active samples. For inactive samples, strict
    # correctness requires an inactive graph decision and zero role masks.
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


def evaluate_policy(
    name: str,
    y: dict[str, np.ndarray],
    graph_scores: np.ndarray,
    graph_predictions: np.ndarray,
    count_predictions: np.ndarray,
    role_scores: dict[str, np.ndarray],
    role_predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    active = y["graph"] == 1
    strict = strict_exact(
        y["graph"],
        y["count"],
        y["source"],
        y["transit"],
        y["victim"],
        y["path"],
        graph_predictions,
        count_predictions,
        role_predictions["source"],
        role_predictions["transit"],
        role_predictions["victim"],
        role_predictions["path"],
    )
    result = {
        "name": name,
        "graph": binary_metrics(y["graph"], graph_predictions),
        "graph_auroc": binary_auroc(y["graph"], graph_scores),
        "graph_average_precision": average_precision(
            y["graph"],
            graph_scores,
        ),
        "strict_exact": float(np.mean(strict)),
        "strict_exact_count": int(np.sum(strict)),
        "count_active_macro_f1": macro_f1_active_count(
            y["count"],
            count_predictions,
            active,
        ),
        "roles": {},
        "fully_consistent_predicted_active": float(
            np.mean(
                (graph_predictions == 1)
                & (count_predictions >= 1)
                & (count_predictions <= 4)
                & (
                    np.sum(role_predictions["source"], axis=1)
                    == count_predictions
                )
                & (
                    np.sum(role_predictions["victim"], axis=1)
                    == count_predictions
                )
            )
        ),
    }
    for role in ("source", "transit", "victim", "path"):
        result["roles"][role] = role_metrics(
            y[role],
            role_predictions[role],
            role_scores[role],
            active,
        )
    return result


def manifest_range(path: Path, manifest: dict[str, Any]) -> tuple[int, int]:
    if "start" in manifest and "stop" in manifest:
        return int(manifest["start"]), int(manifest["stop"])
    match = re.search(r"items_(\d+)_(\d+)", path.name)
    if not match:
        raise RuntimeError(f"cannot resolve chunk range from {path}")
    return int(match.group(1)), int(match.group(2))


def resolve_chunk_path(
    manifest_path: Path,
    manifest: dict[str, Any],
    possible_keys: tuple[str, ...],
) -> Path:
    for key in possible_keys:
        value = manifest.get(key)
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = manifest_path.parent / path
            return path.resolve()
    raise RuntimeError(f"chunk path missing in {manifest_path}")


def load_and_validate_canonical_prefix(
    working_dir: Path,
    export_sha: str,
    canonical_sha: str,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    records = []
    for manifest_path in working_dir.glob("chunk_*.manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            start, stop = manifest_range(manifest_path, manifest)
        except Exception:
            continue
        if start >= PREFIX_ITEMS or stop > PREFIX_ITEMS:
            continue
        chunk_path = resolve_chunk_path(
            manifest_path,
            manifest,
            ("chunk_path", "canonical_chunk_path"),
        )
        expected_sha = (
            manifest.get("chunk_sha256")
            or manifest.get("canonical_chunk_sha256")
        )
        if not chunk_path.is_file():
            raise RuntimeError(f"canonical chunk missing: {chunk_path}")
        if expected_sha != sha256_file(chunk_path):
            raise RuntimeError(
                f"canonical chunk SHA mismatch: {chunk_path}"
            )
        if manifest.get("D1_export_sha256") != export_sha:
            raise RuntimeError(
                f"canonical chunk D1 export mismatch: {manifest_path}"
            )
        decoder_sha = manifest.get("certified_decoder_sha256")
        if decoder_sha is not None and decoder_sha != canonical_sha:
            raise RuntimeError(
                f"canonical decoder SHA mismatch: {manifest_path}"
            )
        records.append(
            {
                "start": start,
                "stop": stop,
                "manifest_path": manifest_path,
                "chunk_path": chunk_path,
                "chunk_sha256": expected_sha,
            }
        )

    records.sort(key=lambda row: row["start"])
    expected_start = 0
    for record in records:
        if record["start"] != expected_start:
            raise RuntimeError(
                f"canonical prefix gap: expected {expected_start}, "
                f"found {record['start']}"
            )
        expected_start = record["stop"]
    if expected_start != PREFIX_ITEMS:
        raise RuntimeError(
            f"canonical prefix ends at {expected_start}, "
            f"expected {PREFIX_ITEMS}"
        )

    fields = {
        "dataset_index": [],
        "graph_prediction": [],
        "count_prediction": [],
        "source_prediction": [],
        "transit_prediction": [],
        "victim_prediction": [],
        "path_prediction": [],
        "margin": [],
        "route_count": [],
        "selected_path_code": [],
    }

    for record in records:
        with np.load(record["chunk_path"], allow_pickle=False) as loaded:
            count = record["stop"] - record["start"]
            fields["dataset_index"].append(
                loaded["dataset_index"].copy()
                if "dataset_index" in loaded.files
                else np.arange(
                    record["start"],
                    record["stop"],
                    dtype=np.int32,
                )
            )
            for key in (
                "graph_prediction",
                "count_prediction",
                "source_prediction",
                "transit_prediction",
                "victim_prediction",
                "path_prediction",
                "margin",
                "route_count",
            ):
                if key not in loaded.files:
                    raise RuntimeError(
                        f"canonical chunk missing {key}: "
                        f"{record['chunk_path']}"
                    )
                fields[key].append(loaded[key].copy())
            fields["selected_path_code"].append(
                np.zeros(count, dtype=np.uint8)
            )

    merged = {
        key: np.concatenate(parts, axis=0)
        for key, parts in fields.items()
    }
    if merged["dataset_index"].shape[0] != PREFIX_ITEMS:
        raise RuntimeError("canonical prefix item count mismatch")
    if not np.array_equal(
        merged["dataset_index"],
        np.arange(PREFIX_ITEMS),
    ):
        raise RuntimeError("canonical prefix dataset order mismatch")
    return records, merged


def load_valid_suffix_chunk(
    manifest_path: Path,
    expected: dict[str, Any],
) -> dict[str, np.ndarray] | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if manifest.get(key) != value:
                return None
        chunk_path = resolve_chunk_path(
            manifest_path,
            manifest,
            ("chunk_path",),
        )
        if (
            manifest.get("status") != "COMPLETE"
            or not chunk_path.is_file()
            or manifest.get("chunk_sha256") != sha256_file(chunk_path)
        ):
            return None
        with np.load(chunk_path, allow_pickle=False) as loaded:
            return {key: loaded[key].copy() for key in loaded.files}
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    merged_path = output_dir / (
        "V5_P3_D4_R3_COMPLETE_A1_HYBRID_VALIDATION_OUTPUTS.npz"
    )
    suffix_working = output_dir / "HYBRID_SUFFIX_WORKING"
    suffix_working.mkdir(parents=True, exist_ok=True)

    r2r2_dir = repo / "reports/v5/p3_d4_r2_r2_hybrid_decoder_freeze"
    r2r2_report_path = r2r2_dir / (
        "V5_P3_D4_R2_R2_CANONICAL_FIRST_CERTIFIED_"
        "PRESOLVE_FALSE_FALLBACK_FREEZE_REPORT.json"
    )
    r2r2_lock_path = r2r2_dir / (
        "V5_P3_D4_R2_R2_CANONICAL_FIRST_CERTIFIED_"
        "PRESOLVE_FALSE_FALLBACK_FREEZE_LOCK.json"
    )

    d4_dir = repo / "reports/v5/p3_d4_resumable_a1_exact_validation_evaluation"
    d4_working = d4_dir / "A1_WORKING"
    d4_script_path = repo / (
        "scripts/v5/p3/"
        "run_v5_p3_d4_resumable_a1_exact_validation_evaluation.py"
    )

    d1_dir = repo / "reports/v5/p3_d1_immutable_validation_logit_export"
    export_path = d1_dir / (
        "V5_P3_D1_TRANCHE_A_VALIDATION_LOGITS_AND_LABELS.npz"
    )
    d1_lock_path = d1_dir / (
        "V5_P3_D1_IMMUTABLE_TRANCHE_A_VALIDATION_LOGIT_EXPORT_LOCK.json"
    )

    d2_dir = repo / "reports/v5/p3_d2_raw_and_frozen_a0_validation_evaluation"
    d2_report_candidates = list(d2_dir.glob("*REPORT.json"))

    canonical_path = repo / "src/decoders/v5_legal_xy_exact_decoder_certified.py"
    fallback_path = repo / "src/decoders/v5_p3_a1_exact_decoder_presolve_false.py"
    hybrid_path = repo / "src/decoders/v5_p3_a1_exact_decoder_certified_hybrid.py"

    required = [
        r2r2_report_path,
        r2r2_lock_path,
        d4_working,
        d4_script_path,
        export_path,
        d1_lock_path,
        canonical_path,
        fallback_path,
        hybrid_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")

    r2r2_report = json.loads(r2r2_report_path.read_text(encoding="utf-8"))
    r2r2_lock = json.loads(r2r2_lock_path.read_text(encoding="utf-8"))
    d1_lock = json.loads(d1_lock_path.read_text(encoding="utf-8"))

    if r2r2_report.get("status") != "PASS":
        raise RuntimeError("R2-R2 report is not PASS")
    if r2r2_lock.get("report_sha256") != sha256_file(r2r2_report_path):
        raise RuntimeError("R2-R2 report/lock mismatch")
    if not r2r2_report["decision"].get("R3_resume_authorized"):
        raise RuntimeError("R2-R2 did not authorize R3")

    export_sha = sha256_file(export_path)
    canonical_sha = sha256_file(canonical_path)
    fallback_sha = sha256_file(fallback_path)
    hybrid_sha = sha256_file(hybrid_path)

    if d1_lock.get("export_sha256") != export_sha:
        raise RuntimeError("D1 export/lock mismatch")
    if r2r2_lock.get("canonical_decoder_sha256") != canonical_sha:
        raise RuntimeError("canonical decoder SHA differs from R2-R2")
    if r2r2_lock.get("fallback_decoder_sha256") != fallback_sha:
        raise RuntimeError("fallback decoder SHA differs from R2-R2")
    if r2r2_lock.get("hybrid_wrapper_sha256") != hybrid_sha:
        raise RuntimeError("hybrid decoder SHA differs from R2-R2")

    canonical_records, prefix = load_and_validate_canonical_prefix(
        d4_working,
        export_sha,
        canonical_sha,
    )

    with np.load(export_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}

    if arrays["attack_logits"].shape[0] != TOTAL_ITEMS:
        raise RuntimeError(
            f"D1 item count={arrays['attack_logits'].shape[0]}, "
            f"expected={TOTAL_ITEMS}"
        )

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    d4 = import_source(d4_script_path, "_v5_p3_d4_r3_semantics")
    hybrid = import_source(hybrid_path, "_v5_p3_d4_r3_hybrid")

    suffix_parts: dict[str, list[np.ndarray]] = {
        "dataset_index": [],
        "graph_prediction": [],
        "count_prediction": [],
        "source_prediction": [],
        "transit_prediction": [],
        "victim_prediction": [],
        "path_prediction": [],
        "margin": [],
        "route_count": [],
        "route_ids_padded": [],
        "selected_path_code": [],
        "canonical_error_present": [],
        "decode_seconds": [],
    }

    chunk_records = []
    fallback_uses = 0
    for chunk_number, start in enumerate(
        range(PREFIX_ITEMS, TOTAL_ITEMS, CHUNK_SIZE),
        start=1,
    ):
        stop = min(start + CHUNK_SIZE, TOTAL_ITEMS)
        chunk_path = suffix_working / (
            f"chunk_{chunk_number:04d}_items_{start:05d}_{stop:05d}.npz"
        )
        manifest_path = suffix_working / (
            f"chunk_{chunk_number:04d}_items_{start:05d}_{stop:05d}"
            ".manifest.json"
        )
        expected_manifest = {
            "start": start,
            "stop": stop,
            "D1_export_sha256": export_sha,
            "canonical_decoder_sha256": canonical_sha,
            "fallback_decoder_sha256": fallback_sha,
            "hybrid_decoder_sha256": hybrid_sha,
            "A1_margin_threshold": A1_MARGIN_THRESHOLD,
        }

        chunk = load_valid_suffix_chunk(
            manifest_path,
            expected_manifest,
        )
        if chunk is not None:
            print(
                f"R3_suffix_chunk={chunk_number} status=REUSED "
                f"items={start}:{stop}",
                flush=True,
            )
        else:
            print(
                f"R3_suffix_chunk={chunk_number} status=DECODING "
                f"items={start}:{stop}",
                flush=True,
            )
            graph = []
            count = []
            source = []
            transit = []
            victim = []
            path = []
            margin = []
            route_count = []
            route_ids_padded = []
            selected_path_code = []
            canonical_error_present = []
            decode_seconds = []

            for local, index in enumerate(range(start, stop), start=1):
                started = time.perf_counter()
                try:
                    hypothesis = hybrid.decode_best_attack_hypothesis(
                        float(arrays["attack_logits"][index]),
                        arrays["count_logits"][index].astype(np.float64),
                        arrays["source_logits"][index].astype(np.float64),
                        arrays["transit_logits"][index].astype(np.float64),
                        arrays["victim_logits"][index].astype(np.float64),
                        arrays["path_logits"][index].astype(np.float64),
                    )
                    decoded = hybrid.apply_margin_threshold(
                        hypothesis,
                        A1_MARGIN_THRESHOLD,
                    )
                    extracted, diagnostics = d4.extract_decoded_output(
                        decoded,
                        hypothesis,
                    )
                except Exception as exc:
                    hold = {
                        "stage": STAGE,
                        "status": "HOLD",
                        "failed_dataset_index": index,
                        "exception_type": type(exc).__name__,
                        "exception": repr(exc),
                        "completed_suffix_chunks": chunk_number - 1,
                        "D4_prefix_items_preserved": PREFIX_ITEMS,
                        "test_tensor_loaded": False,
                    }
                    atomic_json(
                        output_dir / f"{STAGE}_HOLD.json",
                        hold,
                    )
                    raise

                routes = np.asarray(
                    extracted["route_ids"],
                    dtype=np.int32,
                ).reshape(-1)
                padded = np.full(4, -1, dtype=np.int32)
                if routes.size > 4:
                    raise RuntimeError(
                        f"route count exceeds 4 at index {index}"
                    )
                padded[: routes.size] = routes

                selected = getattr(
                    hypothesis,
                    "selected_path",
                    None,
                )
                if selected == "canonical_presolve_true":
                    selected_code = 0
                elif selected == "certified_fallback_presolve_false":
                    selected_code = 1
                else:
                    raise RuntimeError(
                        f"unknown hybrid selected path at {index}: "
                        f"{selected}"
                    )

                graph.append(int(extracted["graph"]))
                count.append(int(extracted["count"]))
                source.append(
                    extracted["source"].astype(np.uint8)
                )
                transit.append(
                    extracted["transit"].astype(np.uint8)
                )
                victim.append(
                    extracted["victim"].astype(np.uint8)
                )
                path.append(
                    extracted["path"].astype(np.uint8)
                )
                margin.append(float(extracted["margin"]))
                route_count.append(int(routes.size))
                route_ids_padded.append(padded)
                selected_path_code.append(selected_code)
                canonical_error_present.append(
                    int(getattr(hypothesis, "canonical_error", None) is not None)
                )
                decode_seconds.append(time.perf_counter() - started)

                if (
                    local == 1
                    or local % 25 == 0
                    or index + 1 == stop
                ):
                    print(
                        f"R3_chunk_item={local}/{stop-start} "
                        f"dataset_index={index} "
                        f"selected_path={selected} "
                        f"seconds={decode_seconds[-1]:.6f}",
                        flush=True,
                    )

            chunk = {
                "dataset_index": np.arange(
                    start,
                    stop,
                    dtype=np.int32,
                ),
                "graph_prediction": np.asarray(graph, dtype=np.uint8),
                "count_prediction": np.asarray(count, dtype=np.int8),
                "source_prediction": np.stack(source).astype(np.uint8),
                "transit_prediction": np.stack(transit).astype(np.uint8),
                "victim_prediction": np.stack(victim).astype(np.uint8),
                "path_prediction": np.stack(path).astype(np.uint8),
                "margin": np.asarray(margin, dtype=np.float64),
                "route_count": np.asarray(route_count, dtype=np.int8),
                "route_ids_padded": np.stack(route_ids_padded).astype(
                    np.int32
                ),
                "selected_path_code": np.asarray(
                    selected_path_code,
                    dtype=np.uint8,
                ),
                "canonical_error_present": np.asarray(
                    canonical_error_present,
                    dtype=np.uint8,
                ),
                "decode_seconds": np.asarray(
                    decode_seconds,
                    dtype=np.float64,
                ),
            }
            atomic_npz(chunk_path, **chunk)
            manifest = {
                "stage": STAGE,
                "status": "COMPLETE",
                **expected_manifest,
                "chunk_path": str(chunk_path),
                "chunk_sha256": sha256_file(chunk_path),
                "items": stop - start,
                "canonical_path_items": int(
                    np.sum(chunk["selected_path_code"] == 0)
                ),
                "fallback_path_items": int(
                    np.sum(chunk["selected_path_code"] == 1)
                ),
            }
            atomic_json(manifest_path, manifest)
            print(
                f"R3_suffix_chunk={chunk_number} status=COMMITTED",
                flush=True,
            )

        if not np.array_equal(
            chunk["dataset_index"],
            np.arange(start, stop),
        ):
            raise RuntimeError(
                f"suffix chunk order mismatch at {start}:{stop}"
            )
        fallback_uses += int(np.sum(chunk["selected_path_code"] == 1))
        for key in suffix_parts:
            suffix_parts[key].append(chunk[key])
        chunk_records.append(
            {
                "start": start,
                "stop": stop,
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "chunk_path": str(chunk_path),
                "chunk_sha256": sha256_file(chunk_path),
            }
        )

    suffix = {
        key: np.concatenate(parts, axis=0)
        for key, parts in suffix_parts.items()
    }
    if suffix["dataset_index"].shape[0] != TOTAL_ITEMS - PREFIX_ITEMS:
        raise RuntimeError("suffix item count mismatch")
    if not np.array_equal(
        suffix["dataset_index"],
        np.arange(PREFIX_ITEMS, TOTAL_ITEMS),
    ):
        raise RuntimeError("suffix dataset order mismatch")

    complete = {}
    for key in (
        "dataset_index",
        "graph_prediction",
        "count_prediction",
        "source_prediction",
        "transit_prediction",
        "victim_prediction",
        "path_prediction",
        "margin",
        "route_count",
        "selected_path_code",
    ):
        complete[key] = np.concatenate(
            [prefix[key], suffix[key]],
            axis=0,
        )
    complete["route_ids_padded"] = np.concatenate(
        [
            np.full((PREFIX_ITEMS, 4), -1, dtype=np.int32),
            suffix["route_ids_padded"],
        ],
        axis=0,
    )
    complete["canonical_error_present"] = np.concatenate(
        [
            np.zeros(PREFIX_ITEMS, dtype=np.uint8),
            suffix["canonical_error_present"],
        ]
    )
    complete["decode_seconds_suffix"] = suffix["decode_seconds"]

    if not np.array_equal(
        complete["dataset_index"],
        np.arange(TOTAL_ITEMS),
    ):
        raise RuntimeError("complete A1 output order mismatch")

    atomic_npz(merged_path, **complete)

    y = {
        "graph": arrays["y_attack"].astype(np.uint8),
        "count": arrays["y_attacker_count"].astype(np.int8),
        "source": arrays["y_source"].astype(np.uint8),
        "transit": arrays["y_transit"].astype(np.uint8),
        "victim": arrays["y_victim"].astype(np.uint8),
        "path": arrays["y_attack_path"].astype(np.uint8),
    }

    graph_prob = sigmoid(arrays["attack_logits"])
    role_prob = {
        "source": sigmoid(arrays["source_logits"]),
        "transit": sigmoid(arrays["transit_logits"]),
        "victim": sigmoid(arrays["victim_logits"]),
        "path": sigmoid(arrays["path_logits"]),
    }
    count_argmax = (
        np.argmax(arrays["count_logits"], axis=1).astype(np.int8) + 1
    )

    raw = evaluate_policy(
        "Raw_0.5",
        y,
        graph_prob,
        (graph_prob >= 0.5).astype(np.uint8),
        count_argmax,
        role_prob,
        {
            role: (scores >= 0.5).astype(np.uint8)
            for role, scores in role_prob.items()
        },
    )
    a0 = evaluate_policy(
        "Frozen_P2_A0_transfer",
        y,
        graph_prob,
        (graph_prob >= GRAPH_THRESHOLD_A0).astype(np.uint8),
        count_argmax,
        role_prob,
        {
            "source": (
                role_prob["source"] >= SOURCE_THRESHOLD_A0
            ).astype(np.uint8),
            "transit": (
                role_prob["transit"] >= TRANSIT_THRESHOLD_A0
            ).astype(np.uint8),
            "victim": (
                role_prob["victim"] >= VICTIM_THRESHOLD_A0
            ).astype(np.uint8),
            "path": (
                role_prob["path"] >= PATH_THRESHOLD_A0
            ).astype(np.uint8),
        },
    )
    a1 = evaluate_policy(
        "Frozen_P2_exact_decoder_transfer_with_P3_hybrid_numerical_policy",
        y,
        complete["margin"],
        complete["graph_prediction"],
        complete["count_prediction"],
        {
            "source": complete["source_prediction"].astype(np.float64),
            "transit": complete["transit_prediction"].astype(np.float64),
            "victim": complete["victim_prediction"].astype(np.float64),
            "path": complete["path_prediction"].astype(np.float64),
        },
        {
            "source": complete["source_prediction"],
            "transit": complete["transit_prediction"],
            "victim": complete["victim_prediction"],
            "path": complete["path_prediction"],
        },
    )

    reference_checks = {
        "Raw": {
            "graph_accuracy_delta": (
                raw["graph"]["accuracy"]
                - RAW_REFERENCE["graph_accuracy"]
            ),
            "graph_f1_delta": (
                raw["graph"]["f1"] - RAW_REFERENCE["graph_f1"]
            ),
            "graph_fpr_delta": (
                raw["graph"]["fpr"] - RAW_REFERENCE["graph_fpr"]
            ),
            "strict_exact_delta": (
                raw["strict_exact"] - RAW_REFERENCE["strict_exact"]
            ),
        },
        "A0": {
            "graph_accuracy_delta": (
                a0["graph"]["accuracy"]
                - A0_REFERENCE["graph_accuracy"]
            ),
            "graph_f1_delta": (
                a0["graph"]["f1"] - A0_REFERENCE["graph_f1"]
            ),
            "graph_fpr_delta": (
                a0["graph"]["fpr"] - A0_REFERENCE["graph_fpr"]
            ),
            "strict_exact_delta": (
                a0["strict_exact"] - A0_REFERENCE["strict_exact"]
            ),
        },
    }
    max_reference_delta = max(
        abs(value)
        for section in reference_checks.values()
        for value in section.values()
    )
    if max_reference_delta > REFERENCE_TOLERANCE:
        raise RuntimeError(
            "Raw/A0 metric reproduction exceeded tolerance: "
            f"{max_reference_delta}"
        )

    suffix_seconds = suffix["decode_seconds"]
    selected_codes = complete["selected_path_code"]
    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "scope": (
            "Complete the preliminary Tranche-A validation-only A1 exact "
            "decoder transfer evaluation by reusing the certified canonical "
            "11,700-item prefix and decoding the remaining 2,163 items with "
            "the prospectively frozen P3 hybrid numerical policy."
        ),
        "decoder_policy": {
            "semantic_decoder": (
                "Frozen P2 route library, objective, certificate, thresholds, "
                "and A1 margin"
            ),
            "P3_numerical_policy": (
                "canonical presolve=True first; certified presolve=False "
                "fallback only on canonical ExactDecoderError"
            ),
            "A1_margin_threshold": A1_MARGIN_THRESHOLD,
            "certificate_tolerance_changed": False,
            "threshold_or_margin_retuned": False,
            "beam_decoder_selected": False,
        },
        "coverage": {
            "total_validation_items": TOTAL_ITEMS,
            "canonical_prefix_items_reused": PREFIX_ITEMS,
            "hybrid_suffix_items_decoded": TOTAL_ITEMS - PREFIX_ITEMS,
            "canonical_prefix_chunks": len(canonical_records),
            "hybrid_suffix_chunks": len(chunk_records),
            "complete": True,
        },
        "hybrid_path_counts": {
            "canonical_presolve_true": int(
                np.sum(selected_codes == 0)
            ),
            "certified_fallback_presolve_false": int(
                np.sum(selected_codes == 1)
            ),
            "fallback_uses_in_suffix": fallback_uses,
        },
        "metrics": {
            "Raw": raw,
            "A0": a0,
            "A1": a1,
            "comparison": {
                "A1_minus_Raw_graph_accuracy": (
                    a1["graph"]["accuracy"]
                    - raw["graph"]["accuracy"]
                ),
                "A1_minus_Raw_graph_f1": (
                    a1["graph"]["f1"] - raw["graph"]["f1"]
                ),
                "A1_minus_Raw_graph_fpr": (
                    a1["graph"]["fpr"] - raw["graph"]["fpr"]
                ),
                "A1_minus_Raw_strict_exact": (
                    a1["strict_exact"] - raw["strict_exact"]
                ),
                "A1_minus_A0_strict_exact": (
                    a1["strict_exact"] - a0["strict_exact"]
                ),
            },
        },
        "reference_reproduction": {
            "tolerance": REFERENCE_TOLERANCE,
            "max_absolute_delta": max_reference_delta,
            "checks": reference_checks,
            "status": "PASS",
        },
        "timing_suffix_only": {
            "items": int(suffix_seconds.size),
            "mean_seconds": float(np.mean(suffix_seconds)),
            "median_seconds": float(np.median(suffix_seconds)),
            "p95_seconds": float(np.quantile(suffix_seconds, 0.95)),
            "p99_seconds": float(np.quantile(suffix_seconds, 0.99)),
            "max_seconds": float(np.max(suffix_seconds)),
            "total_seconds": float(np.sum(suffix_seconds)),
        },
        "artifacts": {
            "merged_outputs_path": str(merged_path),
            "merged_outputs_sha256": sha256_file(merged_path),
            "suffix_working_directory": str(suffix_working),
            "chunk_records": chunk_records,
        },
        "decision": {
            "A1_validation_evaluation_complete": True,
            "result_classification": (
                "preliminary validation-only decoder-transfer diagnostic"
            ),
            "final_model_claim_authorized": False,
            "P2_result_claim_authorized": False,
            "test_evaluation_authorized": False,
            "D5_review_authorized": True,
            "next_stage": (
                "V5_P3_D5_TRANCHE_A_DECODER_REVIEW_AND_"
                "4X4_EXPERT_INTERFACE_FREEZE"
            ),
        },
        "sealed_test": {
            "test_dataset_instantiated": False,
            "test_length_computed": False,
            "test_tensor_loaded": False,
            "evaluation_authorized": False,
        },
        "provenance": {
            "R2_R2_report_sha256": sha256_file(r2r2_report_path),
            "R2_R2_lock_sha256": sha256_file(r2r2_lock_path),
            "D1_export_sha256": export_sha,
            "D2_report_sha256": (
                sha256_file(d2_report_candidates[0])
                if len(d2_report_candidates) == 1
                else None
            ),
            "canonical_decoder_sha256": canonical_sha,
            "fallback_decoder_sha256": fallback_sha,
            "hybrid_decoder_sha256": hybrid_sha,
            "installed_script_sha256": sha256_file(installed_script),
        },
        "canonical_D4_chunks_modified": False,
        "R2_chunks_modified": False,
        "model_replayed": False,
        "model_trained": False,
        "threshold_tuning_performed": False,
        "test_tensor_loaded": False,
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(report_path),
        "merged_outputs_sha256": sha256_file(merged_path),
        "D1_export_sha256": export_sha,
        "canonical_decoder_sha256": canonical_sha,
        "fallback_decoder_sha256": fallback_sha,
        "hybrid_decoder_sha256": hybrid_sha,
        "total_items": TOTAL_ITEMS,
        "canonical_prefix_items": PREFIX_ITEMS,
        "hybrid_suffix_items": TOTAL_ITEMS - PREFIX_ITEMS,
        "A1_graph_accuracy": a1["graph"]["accuracy"],
        "A1_graph_f1": a1["graph"]["f1"],
        "A1_graph_fpr": a1["graph"]["fpr"],
        "A1_strict_exact": a1["strict_exact"],
        "fallback_path_items": int(np.sum(selected_codes == 1)),
        "test_tensor_loaded": False,
    }
    atomic_json(lock_path, lock)
    (
        output_dir / f"{STAGE}_COMPLETE"
    ).write_text(f"{STAGE}_COMPLETE\n", encoding="utf-8")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign_label={CAMPAIGN_LABEL}")
    print(f"total_validation_items={TOTAL_ITEMS}")
    print(f"canonical_prefix_items_reused={PREFIX_ITEMS}")
    print(
        "hybrid_suffix_items_decoded="
        f"{TOTAL_ITEMS - PREFIX_ITEMS}"
    )
    print(
        "hybrid_path_canonical_items="
        f"{int(np.sum(selected_codes == 0))}"
    )
    print(
        "hybrid_path_fallback_items="
        f"{int(np.sum(selected_codes == 1))}"
    )
    print(f"Raw_graph_accuracy={raw['graph']['accuracy']:.8f}")
    print(f"Raw_graph_f1={raw['graph']['f1']:.8f}")
    print(f"Raw_graph_fpr={raw['graph']['fpr']:.8f}")
    print(f"Raw_strict_exact={raw['strict_exact']:.8f}")
    print(f"A0_graph_accuracy={a0['graph']['accuracy']:.8f}")
    print(f"A0_graph_f1={a0['graph']['f1']:.8f}")
    print(f"A0_graph_fpr={a0['graph']['fpr']:.8f}")
    print(f"A0_strict_exact={a0['strict_exact']:.8f}")
    print(f"A1_graph_accuracy={a1['graph']['accuracy']:.8f}")
    print(f"A1_graph_auroc={a1['graph_auroc']:.8f}")
    print(
        "A1_graph_average_precision="
        f"{a1['graph_average_precision']:.8f}"
    )
    print(f"A1_graph_f1={a1['graph']['f1']:.8f}")
    print(f"A1_graph_fpr={a1['graph']['fpr']:.8f}")
    print(f"A1_strict_exact={a1['strict_exact']:.8f}")
    print(
        "A1_count_active_macro_f1="
        f"{a1['count_active_macro_f1']:.8f}"
    )
    for role in ("source", "transit", "victim", "path"):
        print(
            f"A1_{role}_exact_active="
            f"{a1['roles'][role]['exact_active']:.8f}"
        )
    print(
        "A1_minus_Raw_strict_exact="
        f"{a1['strict_exact'] - raw['strict_exact']:.8f}"
    )
    print(
        "A1_minus_A0_strict_exact="
        f"{a1['strict_exact'] - a0['strict_exact']:.8f}"
    )
    print(
        "Raw_A0_reference_max_delta="
        f"{max_reference_delta:.12g}"
    )
    print("threshold_or_margin_retuning=false")
    print("beam_decoder_selected=false")
    print("canonical_D4_chunks_modified=false")
    print("test_tensor_loaded=false")
    print("D5_review_authorized=true")
    print(
        "next_stage="
        "V5_P3_D5_TRANCHE_A_DECODER_REVIEW_AND_"
        "4X4_EXPERT_INTERFACE_FREEZE"
    )
    print(f"merged_outputs={merged_path}")
    print(f"merged_outputs_sha256={sha256_file(merged_path)}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
