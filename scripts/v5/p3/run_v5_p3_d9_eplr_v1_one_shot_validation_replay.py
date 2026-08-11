from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import json
import math
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STAGE = (
    "V5_P3_D9_EPLR_V1_ONE_SHOT_TRANCHE_A_"
    "VALIDATION_EXPLORATORY_REPLAY"
)
CAMPAIGN_LABEL = (
    "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic "
    "— EPLR Decoder Exploratory Evaluation"
)
TOTAL_ITEMS = 13_863
CHUNK_SIZE = 128
REFERENCE_TOLERANCE = 5e-7

STATUS_TO_CODE = {
    "INACTIVE_RAW": 0,
    "RAW_ENDPOINTS_LEGAL": 1,
    "RAW_ENDPOINTS_LEGAL_LOW_ROUTE_SUPPORT": 2,
    "ENDPOINT_REPAIR_APPLIED": 3,
    "NO_CERTIFIED_LEGAL_EXPLANATION": 4,
}
CODE_TO_STATUS = {
    value: key for key, value in STATUS_TO_CODE.items()
}

# Frozen before this one-shot validation replay.
ACCEPTANCE = {
    "maximum_source_exact_drop": 0.01,
    "maximum_victim_exact_drop": 0.01,
    "minimum_transit_exact_gain": 0.05,
    "minimum_strict_exact_gain": 0.05,
    "maximum_path_exact_drop": 0.01,
}

_WORKER_DECODER = None
_WORKER_CONFIG = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    parser.add_argument("--workers", type=int, default=4)
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


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def bitmap_to_array(mask: int) -> np.ndarray:
    return np.asarray(
        [
            1 if int(mask) & (1 << router) else 0
            for router in range(16)
        ],
        dtype=np.uint8,
    )


def mask_from_array(values: np.ndarray) -> int:
    mask = 0
    for index, value in enumerate(
        np.asarray(values).reshape(-1).tolist()
    ):
        if int(value):
            mask |= 1 << index
    return mask


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
    return float(
        np.mean(
            np.all(
                labels[active] == predictions[active],
                axis=1,
            )
        )
    )


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


def _worker_init(repo: str, config_path: str) -> None:
    global _WORKER_DECODER, _WORKER_CONFIG
    repo_path = str(Path(repo).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    _WORKER_DECODER = importlib.import_module(
        "src.decoders.v5_p3_eplr_v1"
    )
    _WORKER_CONFIG = _WORKER_DECODER.EPLRConfig.from_json(
        config_path
    )


def _worker_decode(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        dataset_index,
        graph_logit,
        count_logits,
        source_logits,
        transit_logits,
        victim_logits,
        path_logits,
    ) = task
    started = time.perf_counter()
    try:
        result = _WORKER_DECODER.decode_eplr_v1(
            graph_logit=float(graph_logit),
            count_logits=count_logits,
            source_logits=source_logits,
            transit_logits=transit_logits,
            victim_logits=victim_logits,
            path_logits=path_logits,
            config=_WORKER_CONFIG,
        )
        payload = result.to_dict()
        payload["dataset_index"] = int(dataset_index)
        payload["decode_seconds"] = (
            time.perf_counter() - started
        )
        payload["worker_exception"] = None
        return payload
    except Exception as exc:
        return {
            "dataset_index": int(dataset_index),
            "worker_exception": repr(exc),
            "worker_exception_type": type(exc).__name__,
            "decode_seconds": time.perf_counter() - started,
        }


def load_valid_chunk(
    manifest_path: Path,
    expected: dict[str, Any],
) -> dict[str, np.ndarray] | None:
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        for key, value in expected.items():
            if manifest.get(key) != value:
                return None
        chunk_path = Path(manifest["chunk_path"])
        if (
            manifest.get("status") != "COMPLETE"
            or not chunk_path.is_file()
            or manifest.get("chunk_sha256")
            != sha256_file(chunk_path)
        ):
            return None
        with np.load(chunk_path, allow_pickle=False) as loaded:
            return {
                key: loaded[key].copy()
                for key in loaded.files
            }
    except Exception:
        return None


def check_route_semantics(
    decoder,
    record: dict[str, Any],
) -> None:
    status = record["decoder_status"]
    graph = int(record["raw_graph_prediction"])
    count = int(record["effective_attacker_count"])
    route_ids = tuple(
        int(value) for value in record["selected_route_ids"]
    )

    if status == decoder.STATUS_INACTIVE:
        if graph != 0 or count != 0 or route_ids:
            raise RuntimeError("inactive semantic contract failure")
        for key in (
            "decoded_source_bitmap",
            "decoded_transit_bitmap",
            "decoded_victim_bitmap",
            "decoded_path_bitmap",
        ):
            if int(record[key]) != 0:
                raise RuntimeError(
                    f"inactive nonzero output: {key}"
                )
        return

    if status == decoder.STATUS_NO_SOLUTION:
        if graph != 1 or count not in (1, 2, 3, 4):
            raise RuntimeError("no-solution graph/count failure")
        if route_ids:
            raise RuntimeError("no-solution returned route IDs")
        return

    if graph != 1 or count not in (1, 2, 3, 4):
        raise RuntimeError("active semantic graph/count failure")
    if len(route_ids) != count:
        raise RuntimeError(
            f"route count {len(route_ids)} != Raw K {count}"
        )

    hypothesis = decoder.routes.combine_route_ids(
        route_ids,
        expected_k=count,
    )
    expected = {
        "decoded_source_bitmap": int(hypothesis.source_mask),
        "decoded_transit_bitmap": int(hypothesis.transit_mask),
        "decoded_victim_bitmap": int(hypothesis.victim_mask),
        "decoded_path_bitmap": int(hypothesis.path_mask),
    }
    for key, value in expected.items():
        if int(record[key]) != value:
            raise RuntimeError(
                f"route-union mismatch for {key}"
            )

    raw_source = int(record["raw_source_bitmap"])
    raw_victim = int(record["raw_victim_bitmap"])
    source_repairs = int(record["source_repair_count"])
    victim_repairs = int(record["victim_repair_count"])
    if source_repairs != (raw_source ^ expected[
        "decoded_source_bitmap"
    ]).bit_count():
        raise RuntimeError("source repair count mismatch")
    if victim_repairs != (raw_victim ^ expected[
        "decoded_victim_bitmap"
    ]).bit_count():
        raise RuntimeError("victim repair count mismatch")

    if status in (
        decoder.STATUS_LEGAL,
        decoder.STATUS_LEGAL_LOW_SUPPORT,
    ):
        if source_repairs != 0 or victim_repairs != 0:
            raise RuntimeError(
                "endpoint-preserving status changed endpoints"
            )
    elif status == decoder.STATUS_REPAIRED:
        if source_repairs + victim_repairs <= 0:
            raise RuntimeError(
                "repair status contains no endpoint edits"
            )
    else:
        raise RuntimeError(f"unknown status: {status}")


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    workers = max(1, int(args.workers))
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"
    merged_path = output_dir / (
        "V5_P3_D9_EPLR_V1_COMPLETE_VALIDATION_OUTPUTS.npz"
    )
    working_dir = output_dir / "EPLR_VALIDATION_WORKING"
    working_dir.mkdir(parents=True, exist_ok=True)

    if complete_path.is_file():
        print(f"{STAGE}_ALREADY_COMPLETE")
        print(report_path.read_text(encoding="utf-8"))
        return 0

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

    d1_dir = (
        repo
        / "reports/v5/p3_d1_immutable_validation_logit_export"
    )
    export_path = d1_dir / (
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

    d5_dir = (
        repo
        / "reports/v5/p3_d5_tranche_a_decoder_review_and_4x4_expert_interface_freeze"
    )
    d5_lock_path = d5_dir / (
        "V5_P3_D5_TRANCHE_A_DECODER_REVIEW_AND_"
        "4X4_EXPERT_INTERFACE_FREEZE_LOCK.json"
    )

    decoder_path = repo / "src/decoders/v5_p3_eplr_v1.py"
    config_path = (
        repo / "src/decoders/v5_p3_eplr_v1_calibration.json"
    )

    required = [
        d8_report_path,
        d8_lock_path,
        export_path,
        d1_lock_path,
        r3_report_path,
        r3_lock_path,
        d5_lock_path,
        decoder_path,
        config_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")

    d8 = json.loads(d8_report_path.read_text(encoding="utf-8"))
    d8_lock = json.loads(d8_lock_path.read_text(encoding="utf-8"))
    d1_lock = json.loads(d1_lock_path.read_text(encoding="utf-8"))
    r3 = json.loads(r3_report_path.read_text(encoding="utf-8"))
    r3_lock = json.loads(r3_lock_path.read_text(encoding="utf-8"))
    d5_lock = json.loads(d5_lock_path.read_text(encoding="utf-8"))

    if d8.get("status") != "PASS":
        raise RuntimeError("D8 is not PASS")
    if d8_lock.get("report_sha256") != sha256_file(d8_report_path):
        raise RuntimeError("D8 report/lock mismatch")
    if not d8["decision"].get("D9_authorized"):
        raise RuntimeError("D8 did not authorize D9")
    if not d8["decision"].get("validation_replay_must_be_one_shot"):
        raise RuntimeError("D8 one-shot contract missing")
    if d8_lock.get("decoder_sha256") != sha256_file(decoder_path):
        raise RuntimeError("EPLR decoder differs from D8 lock")
    if d8_lock.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("EPLR config differs from D8 lock")
    if d1_lock.get("export_sha256") != sha256_file(export_path):
        raise RuntimeError("D1 export/lock mismatch")
    if r3.get("status") != "PASS":
        raise RuntimeError("R3 is not PASS")
    if r3_lock.get("report_sha256") != sha256_file(r3_report_path):
        raise RuntimeError("R3 report/lock mismatch")
    if d5_lock.get("test_tensor_loaded") is not False:
        raise RuntimeError("D5 lock reports test access")

    export_sha = sha256_file(export_path)
    decoder_sha = sha256_file(decoder_path)
    config_sha = sha256_file(config_path)

    with np.load(export_path, allow_pickle=False) as loaded:
        arrays = {
            key: loaded[key].copy()
            for key in loaded.files
        }
    if arrays["attack_logits"].shape[0] != TOTAL_ITEMS:
        raise RuntimeError("D1 validation item count changed")

    repo_string = str(repo)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    decoder = importlib.import_module(
        "src.decoders.v5_p3_eplr_v1"
    )

    graph_prob = stable_sigmoid(arrays["attack_logits"])
    raw_graph = (graph_prob >= 0.5).astype(np.uint8)
    raw_count = (
        np.argmax(arrays["count_logits"], axis=1).astype(np.int8)
        + 1
    )
    raw_source = (
        stable_sigmoid(arrays["source_logits"]) >= 0.5
    ).astype(np.uint8)
    raw_transit = (
        stable_sigmoid(arrays["transit_logits"]) >= 0.5
    ).astype(np.uint8)
    raw_victim = (
        stable_sigmoid(arrays["victim_logits"]) >= 0.5
    ).astype(np.uint8)
    raw_path = (
        stable_sigmoid(arrays["path_logits"]) >= 0.5
    ).astype(np.uint8)

    field_parts: dict[str, list[np.ndarray]] = {
        "dataset_index": [],
        "raw_graph_prediction": [],
        "raw_count_candidate": [],
        "effective_count": [],
        "raw_source_bitmap": [],
        "raw_transit_bitmap": [],
        "raw_victim_bitmap": [],
        "raw_path_bitmap": [],
        "decoded_source": [],
        "decoded_transit": [],
        "decoded_victim": [],
        "decoded_path": [],
        "route_ids_padded": [],
        "route_count": [],
        "route_consistency_score": [],
        "source_repair_count": [],
        "victim_repair_count": [],
        "low_support": [],
        "status_code": [],
        "repair_solver_used": [],
        "decode_seconds": [],
    }

    chunk_records = []
    context = multiprocessing.get_context("spawn")
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=(str(repo), str(config_path)),
    )

    try:
        for chunk_number, start in enumerate(
            range(0, TOTAL_ITEMS, CHUNK_SIZE),
            start=1,
        ):
            stop = min(start + CHUNK_SIZE, TOTAL_ITEMS)
            chunk_path = working_dir / (
                f"chunk_{chunk_number:04d}_"
                f"items_{start:05d}_{stop:05d}.npz"
            )
            manifest_path = working_dir / (
                f"chunk_{chunk_number:04d}_"
                f"items_{start:05d}_{stop:05d}.manifest.json"
            )
            expected_manifest = {
                "start": start,
                "stop": stop,
                "D1_export_sha256": export_sha,
                "decoder_sha256": decoder_sha,
                "config_sha256": config_sha,
                "chunk_size_contract": CHUNK_SIZE,
            }

            chunk = load_valid_chunk(
                manifest_path,
                expected_manifest,
            )
            if chunk is not None:
                print(
                    f"D9_chunk={chunk_number} status=REUSED "
                    f"items={start}:{stop}",
                    flush=True,
                )
            else:
                print(
                    f"D9_chunk={chunk_number} status=DECODING "
                    f"items={start}:{stop}",
                    flush=True,
                )
                tasks = [
                    (
                        index,
                        float(arrays["attack_logits"][index]),
                        arrays["count_logits"][index],
                        arrays["source_logits"][index],
                        arrays["transit_logits"][index],
                        arrays["victim_logits"][index],
                        arrays["path_logits"][index],
                    )
                    for index in range(start, stop)
                ]
                records = list(
                    executor.map(
                        _worker_decode,
                        tasks,
                        chunksize=1,
                    )
                )

                for record in records:
                    if record.get("worker_exception") is not None:
                        hold = {
                            "stage": STAGE,
                            "status": "HOLD",
                            "failed_dataset_index": int(
                                record["dataset_index"]
                            ),
                            "worker_exception_type": record.get(
                                "worker_exception_type"
                            ),
                            "worker_exception": record.get(
                                "worker_exception"
                            ),
                            "completed_chunks": chunk_number - 1,
                            "test_tensor_loaded": False,
                        }
                        atomic_json(
                            output_dir / f"{STAGE}_HOLD.json",
                            hold,
                        )
                        raise RuntimeError(
                            "EPLR worker failed at validation index "
                            f"{record['dataset_index']}: "
                            f"{record['worker_exception']}"
                        )
                    check_route_semantics(decoder, record)

                count = stop - start
                route_ids_padded = np.full(
                    (count, 4),
                    -1,
                    dtype=np.int32,
                )
                route_count = np.zeros(count, dtype=np.int8)
                decoded_source = np.zeros(
                    (count, 16),
                    dtype=np.uint8,
                )
                decoded_transit = np.zeros(
                    (count, 16),
                    dtype=np.uint8,
                )
                decoded_victim = np.zeros(
                    (count, 16),
                    dtype=np.uint8,
                )
                decoded_path = np.zeros(
                    (count, 16),
                    dtype=np.uint8,
                )

                for local, record in enumerate(records):
                    route_ids = tuple(
                        int(value)
                        for value in record["selected_route_ids"]
                    )
                    route_count[local] = len(route_ids)
                    route_ids_padded[
                        local, : len(route_ids)
                    ] = route_ids
                    decoded_source[local] = bitmap_to_array(
                        int(record["decoded_source_bitmap"])
                    )
                    decoded_transit[local] = bitmap_to_array(
                        int(record["decoded_transit_bitmap"])
                    )
                    decoded_victim[local] = bitmap_to_array(
                        int(record["decoded_victim_bitmap"])
                    )
                    decoded_path[local] = bitmap_to_array(
                        int(record["decoded_path_bitmap"])
                    )

                chunk = {
                    "dataset_index": np.arange(
                        start,
                        stop,
                        dtype=np.int32,
                    ),
                    "raw_graph_prediction": np.asarray(
                        [
                            record["raw_graph_prediction"]
                            for record in records
                        ],
                        dtype=np.uint8,
                    ),
                    "raw_count_candidate": np.asarray(
                        [
                            record[
                                "raw_attacker_count_candidate"
                            ]
                            for record in records
                        ],
                        dtype=np.int8,
                    ),
                    "effective_count": np.asarray(
                        [
                            record["effective_attacker_count"]
                            for record in records
                        ],
                        dtype=np.int8,
                    ),
                    "raw_source_bitmap": np.asarray(
                        [
                            record["raw_source_bitmap"]
                            for record in records
                        ],
                        dtype=np.uint16,
                    ),
                    "raw_transit_bitmap": np.asarray(
                        [
                            record["raw_transit_bitmap"]
                            for record in records
                        ],
                        dtype=np.uint16,
                    ),
                    "raw_victim_bitmap": np.asarray(
                        [
                            record["raw_victim_bitmap"]
                            for record in records
                        ],
                        dtype=np.uint16,
                    ),
                    "raw_path_bitmap": np.asarray(
                        [
                            record["raw_path_bitmap"]
                            for record in records
                        ],
                        dtype=np.uint16,
                    ),
                    "decoded_source": decoded_source,
                    "decoded_transit": decoded_transit,
                    "decoded_victim": decoded_victim,
                    "decoded_path": decoded_path,
                    "route_ids_padded": route_ids_padded,
                    "route_count": route_count,
                    "route_consistency_score": np.asarray(
                        [
                            record["route_consistency_score"]
                            for record in records
                        ],
                        dtype=np.float64,
                    ),
                    "source_repair_count": np.asarray(
                        [
                            record["source_repair_count"]
                            for record in records
                        ],
                        dtype=np.int8,
                    ),
                    "victim_repair_count": np.asarray(
                        [
                            record["victim_repair_count"]
                            for record in records
                        ],
                        dtype=np.int8,
                    ),
                    "low_support": np.asarray(
                        [
                            record[
                                "endpoint_preserved_but_low_route_support"
                            ]
                            for record in records
                        ],
                        dtype=np.uint8,
                    ),
                    "status_code": np.asarray(
                        [
                            STATUS_TO_CODE[
                                record["decoder_status"]
                            ]
                            for record in records
                        ],
                        dtype=np.uint8,
                    ),
                    "repair_solver_used": np.asarray(
                        [
                            record["repair_solver_used"]
                            for record in records
                        ],
                        dtype=np.uint8,
                    ),
                    "decode_seconds": np.asarray(
                        [
                            record["decode_seconds"]
                            for record in records
                        ],
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
                    "items": count,
                    "workers": workers,
                    "status_counts": {
                        CODE_TO_STATUS[code]: int(
                            np.sum(chunk["status_code"] == code)
                        )
                        for code in sorted(CODE_TO_STATUS)
                    },
                }
                atomic_json(manifest_path, manifest)
                print(
                    f"D9_chunk={chunk_number} status=COMMITTED "
                    f"repair_items="
                    f"{int(np.sum(chunk['repair_solver_used']))}",
                    flush=True,
                )

            if not np.array_equal(
                chunk["dataset_index"],
                np.arange(start, stop),
            ):
                raise RuntimeError(
                    f"chunk order mismatch at {start}:{stop}"
                )
            for key in field_parts:
                field_parts[key].append(chunk[key])
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
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    merged = {
        key: np.concatenate(parts, axis=0)
        for key, parts in field_parts.items()
    }
    if not np.array_equal(
        merged["dataset_index"],
        np.arange(TOTAL_ITEMS),
    ):
        raise RuntimeError("merged validation order mismatch")

    if not np.array_equal(
        merged["raw_graph_prediction"],
        raw_graph,
    ):
        raise RuntimeError("EPLR graph predictions differ from Raw")
    if not np.array_equal(
        merged["raw_count_candidate"],
        raw_count,
    ):
        raise RuntimeError(
            "EPLR Raw count candidates differ from Raw argmax"
        )

    # Confirm the decoder's retained Raw endpoint bitmaps.
    for index in range(TOTAL_ITEMS):
        if int(merged["raw_source_bitmap"][index]) != mask_from_array(
            raw_source[index]
        ):
            raise RuntimeError("Raw source bitmap mismatch")
        if int(merged["raw_victim_bitmap"][index]) != mask_from_array(
            raw_victim[index]
        ):
            raise RuntimeError("Raw victim bitmap mismatch")

    atomic_npz(merged_path, **merged)

    y_graph = arrays["y_attack"].astype(np.uint8)
    y_count = arrays["y_attacker_count"].astype(np.int8)
    y_source = arrays["y_source"].astype(np.uint8)
    y_transit = arrays["y_transit"].astype(np.uint8)
    y_victim = arrays["y_victim"].astype(np.uint8)
    y_path = arrays["y_attack_path"].astype(np.uint8)
    active = y_graph == 1

    raw_strict = strict_exact(
        y_graph,
        y_count,
        y_source,
        y_transit,
        y_victim,
        y_path,
        raw_graph,
        raw_count,
        raw_source,
        raw_transit,
        raw_victim,
        raw_path,
    )
    eplr_strict = strict_exact(
        y_graph,
        y_count,
        y_source,
        y_transit,
        y_victim,
        y_path,
        merged["raw_graph_prediction"],
        merged["raw_count_candidate"],
        merged["decoded_source"],
        merged["decoded_transit"],
        merged["decoded_victim"],
        merged["decoded_path"],
    )

    raw_metrics = {
        "graph": binary_metrics(y_graph, raw_graph),
        "count_active_macro_f1": macro_f1_active_count(
            y_count,
            raw_count,
            active,
        ),
        "source_exact_active": exact_active(
            y_source,
            raw_source,
            active,
        ),
        "transit_exact_active": exact_active(
            y_transit,
            raw_transit,
            active,
        ),
        "victim_exact_active": exact_active(
            y_victim,
            raw_victim,
            active,
        ),
        "path_exact_active": exact_active(
            y_path,
            raw_path,
            active,
        ),
        "strict_exact": float(np.mean(raw_strict)),
    }
    eplr_metrics = {
        "graph": binary_metrics(
            y_graph,
            merged["raw_graph_prediction"],
        ),
        "count_active_macro_f1": macro_f1_active_count(
            y_count,
            merged["raw_count_candidate"],
            active,
        ),
        "source_exact_active": exact_active(
            y_source,
            merged["decoded_source"],
            active,
        ),
        "transit_exact_active": exact_active(
            y_transit,
            merged["decoded_transit"],
            active,
        ),
        "victim_exact_active": exact_active(
            y_victim,
            merged["decoded_victim"],
            active,
        ),
        "path_exact_active": exact_active(
            y_path,
            merged["decoded_path"],
            active,
        ),
        "strict_exact": float(np.mean(eplr_strict)),
    }

    r3_raw = r3["metrics"]["Raw"]
    reproduction = {
        "graph_accuracy_delta": (
            raw_metrics["graph"]["accuracy"]
            - float(r3_raw["graph"]["accuracy"])
        ),
        "graph_f1_delta": (
            raw_metrics["graph"]["f1"]
            - float(r3_raw["graph"]["f1"])
        ),
        "graph_fpr_delta": (
            raw_metrics["graph"]["fpr"]
            - float(r3_raw["graph"]["fpr"])
        ),
        "strict_exact_delta": (
            raw_metrics["strict_exact"]
            - float(r3_raw["strict_exact"])
        ),
    }
    max_reproduction_delta = max(
        abs(value) for value in reproduction.values()
    )
    if max_reproduction_delta > REFERENCE_TOLERANCE:
        raise RuntimeError(
            "Raw reference reproduction exceeded tolerance: "
            f"{max_reproduction_delta}"
        )

    deltas = {
        "graph_accuracy": (
            eplr_metrics["graph"]["accuracy"]
            - raw_metrics["graph"]["accuracy"]
        ),
        "graph_f1": (
            eplr_metrics["graph"]["f1"]
            - raw_metrics["graph"]["f1"]
        ),
        "graph_fpr": (
            eplr_metrics["graph"]["fpr"]
            - raw_metrics["graph"]["fpr"]
        ),
        "count_active_macro_f1": (
            eplr_metrics["count_active_macro_f1"]
            - raw_metrics["count_active_macro_f1"]
        ),
        "source_exact_active": (
            eplr_metrics["source_exact_active"]
            - raw_metrics["source_exact_active"]
        ),
        "transit_exact_active": (
            eplr_metrics["transit_exact_active"]
            - raw_metrics["transit_exact_active"]
        ),
        "victim_exact_active": (
            eplr_metrics["victim_exact_active"]
            - raw_metrics["victim_exact_active"]
        ),
        "path_exact_active": (
            eplr_metrics["path_exact_active"]
            - raw_metrics["path_exact_active"]
        ),
        "strict_exact": (
            eplr_metrics["strict_exact"]
            - raw_metrics["strict_exact"]
        ),
    }

    gates = {
        "graph_predictions_exactly_Raw": bool(
            np.array_equal(
                merged["raw_graph_prediction"],
                raw_graph,
            )
        ),
        "count_candidates_exactly_Raw": bool(
            np.array_equal(
                merged["raw_count_candidate"],
                raw_count,
            )
        ),
        "source_exact_drop_within_1_point": (
            deltas["source_exact_active"]
            >= -ACCEPTANCE["maximum_source_exact_drop"]
        ),
        "victim_exact_drop_within_1_point": (
            deltas["victim_exact_active"]
            >= -ACCEPTANCE["maximum_victim_exact_drop"]
        ),
        "transit_exact_gain_at_least_5_points": (
            deltas["transit_exact_active"]
            >= ACCEPTANCE["minimum_transit_exact_gain"]
        ),
        "strict_exact_gain_at_least_5_points": (
            deltas["strict_exact"]
            >= ACCEPTANCE["minimum_strict_exact_gain"]
        ),
        "path_exact_drop_within_1_point": (
            deltas["path_exact_active"]
            >= -ACCEPTANCE["maximum_path_exact_drop"]
        ),
        "all_rows_explicitly_classified": bool(
            np.all(
                np.isin(
                    merged["status_code"],
                    np.asarray(
                        sorted(CODE_TO_STATUS),
                        dtype=np.uint8,
                    ),
                )
            )
        ),
    }
    acceptance_pass = all(gates.values())

    status_counts = {
        CODE_TO_STATUS[code]: int(
            np.sum(merged["status_code"] == code)
        )
        for code in sorted(CODE_TO_STATUS)
    }
    raw_positive = int(np.sum(raw_graph == 1))
    repair_items = int(np.sum(merged["repair_solver_used"]))
    no_solution_items = status_counts[
        "NO_CERTIFIED_LEGAL_EXPLANATION"
    ]
    low_support_items = int(np.sum(merged["low_support"]))

    decode_seconds = merged["decode_seconds"]
    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "classification": (
            "one-shot Tranche-A validation exploratory decoder replay; "
            "not final V5-P3 result and not sealed-test evaluation"
        ),
        "frozen_inputs": {
            "D1_validation_items": TOTAL_ITEMS,
            "decoder_sha256": decoder_sha,
            "config_sha256": config_sha,
            "D1_export_sha256": export_sha,
            "workers": workers,
            "chunk_size": CHUNK_SIZE,
        },
        "metrics": {
            "Raw": raw_metrics,
            "EPLR_V1": eplr_metrics,
            "EPLR_minus_Raw": deltas,
            "old_A1_from_R3": r3["metrics"]["A1"],
            "A0_from_R3": r3["metrics"]["A0"],
        },
        "decoder_behavior": {
            "raw_positive_items": raw_positive,
            "status_counts": status_counts,
            "repair_solver_items": repair_items,
            "repair_solver_fraction_of_raw_positive": (
                repair_items / raw_positive
                if raw_positive
                else 0.0
            ),
            "no_solution_items": no_solution_items,
            "no_solution_fraction_of_raw_positive": (
                no_solution_items / raw_positive
                if raw_positive
                else 0.0
            ),
            "low_support_items": low_support_items,
            "mean_source_repairs_on_repaired": (
                float(
                    np.mean(
                        merged["source_repair_count"][
                            merged["repair_solver_used"] == 1
                        ]
                    )
                )
                if repair_items
                else 0.0
            ),
            "mean_victim_repairs_on_repaired": (
                float(
                    np.mean(
                        merged["victim_repair_count"][
                            merged["repair_solver_used"] == 1
                        ]
                    )
                )
                if repair_items
                else 0.0
            ),
        },
        "acceptance_gate": {
            "status": "PASS" if acceptance_pass else "FAIL",
            "thresholds": ACCEPTANCE,
            "checks": gates,
        },
        "reference_reproduction": {
            "status": "PASS",
            "tolerance": REFERENCE_TOLERANCE,
            "max_absolute_delta": max_reproduction_delta,
            "deltas": reproduction,
        },
        "timing": {
            "items": TOTAL_ITEMS,
            "mean_seconds": float(np.mean(decode_seconds)),
            "median_seconds": float(np.median(decode_seconds)),
            "p95_seconds": float(
                np.quantile(decode_seconds, 0.95)
            ),
            "p99_seconds": float(
                np.quantile(decode_seconds, 0.99)
            ),
            "max_seconds": float(np.max(decode_seconds)),
            "sum_worker_seconds": float(
                np.sum(decode_seconds)
            ),
        },
        "artifacts": {
            "merged_outputs_path": str(merged_path),
            "merged_outputs_sha256": sha256_file(merged_path),
            "working_directory": str(working_dir),
            "chunks": chunk_records,
        },
        "decision": {
            "one_shot_validation_replay_complete": True,
            "EPLR_V1_acceptance_gate_pass": acceptance_pass,
            "EPLR_V1_retuning_authorized": False,
            "EPLR_V1_carry_forward_authorized": acceptance_pass,
            "D10_review_authorized": True,
            "sealed_test_evaluation_authorized": False,
            "next_stage": (
                "V5_P3_D10_RAW_A0_A1_EPLR_V1_COMPARISON_"
                "AND_CARRY_FORWARD_DECISION"
            ),
        },
        "governance": {
            "model_replayed": False,
            "neural_weights_changed": False,
            "decoder_or_config_changed": False,
            "threshold_or_weight_retuned_after_validation": False,
            "validation_replay_count": 1,
            "test_tensor_loaded": False,
        },
        "provenance": {
            "D8_report_sha256": sha256_file(d8_report_path),
            "D8_lock_sha256": sha256_file(d8_lock_path),
            "D1_lock_sha256": sha256_file(d1_lock_path),
            "R3_report_sha256": sha256_file(r3_report_path),
            "R3_lock_sha256": sha256_file(r3_lock_path),
            "D5_lock_sha256": sha256_file(d5_lock_path),
            "installed_script_sha256": sha256_file(
                installed_script
            ),
        },
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(report_path),
        "merged_outputs_sha256": sha256_file(merged_path),
        "D1_export_sha256": export_sha,
        "decoder_sha256": decoder_sha,
        "config_sha256": config_sha,
        "acceptance_gate_status": (
            "PASS" if acceptance_pass else "FAIL"
        ),
        "Raw_strict_exact": raw_metrics["strict_exact"],
        "EPLR_strict_exact": eplr_metrics["strict_exact"],
        "EPLR_minus_Raw_strict_exact": deltas[
            "strict_exact"
        ],
        "repair_solver_items": repair_items,
        "no_solution_items": no_solution_items,
        "validation_replay_count": 1,
        "test_tensor_loaded": False,
    }
    atomic_json(lock_path, lock)
    complete_path.write_text(
        f"{STAGE}_COMPLETE\n",
        encoding="utf-8",
    )

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign_label={CAMPAIGN_LABEL}")
    print(f"validation_items={TOTAL_ITEMS}")
    print(f"workers={workers}")
    print(f"raw_positive_items={raw_positive}")
    for status, count in status_counts.items():
        print(f"status_{status}={count}")
    print(f"repair_solver_items={repair_items}")
    print(f"no_solution_items={no_solution_items}")
    print(f"low_support_items={low_support_items}")
    print(
        "graph_predictions_exactly_Raw="
        f"{gates['graph_predictions_exactly_Raw']}"
    )
    print(
        "count_candidates_exactly_Raw="
        f"{gates['count_candidates_exactly_Raw']}"
    )
    print(
        f"Raw_source_exact_active="
        f"{raw_metrics['source_exact_active']:.8f}"
    )
    print(
        f"EPLR_source_exact_active="
        f"{eplr_metrics['source_exact_active']:.8f}"
    )
    print(
        f"Raw_transit_exact_active="
        f"{raw_metrics['transit_exact_active']:.8f}"
    )
    print(
        f"EPLR_transit_exact_active="
        f"{eplr_metrics['transit_exact_active']:.8f}"
    )
    print(
        f"Raw_victim_exact_active="
        f"{raw_metrics['victim_exact_active']:.8f}"
    )
    print(
        f"EPLR_victim_exact_active="
        f"{eplr_metrics['victim_exact_active']:.8f}"
    )
    print(
        f"Raw_path_exact_active="
        f"{raw_metrics['path_exact_active']:.8f}"
    )
    print(
        f"EPLR_path_exact_active="
        f"{eplr_metrics['path_exact_active']:.8f}"
    )
    print(
        f"Raw_strict_exact="
        f"{raw_metrics['strict_exact']:.8f}"
    )
    print(
        f"EPLR_strict_exact="
        f"{eplr_metrics['strict_exact']:.8f}"
    )
    print(
        "EPLR_minus_Raw_source_exact="
        f"{deltas['source_exact_active']:.8f}"
    )
    print(
        "EPLR_minus_Raw_transit_exact="
        f"{deltas['transit_exact_active']:.8f}"
    )
    print(
        "EPLR_minus_Raw_victim_exact="
        f"{deltas['victim_exact_active']:.8f}"
    )
    print(
        "EPLR_minus_Raw_path_exact="
        f"{deltas['path_exact_active']:.8f}"
    )
    print(
        "EPLR_minus_Raw_strict_exact="
        f"{deltas['strict_exact']:.8f}"
    )
    print(
        "acceptance_gate_status="
        f"{'PASS' if acceptance_pass else 'FAIL'}"
    )
    for name, value in gates.items():
        print(f"gate_{name}={value}")
    print("EPLR_V1_retuning_authorized=false")
    print("test_tensor_loaded=false")
    print("D10_review_authorized=true")
    print(
        "next_stage="
        "V5_P3_D10_RAW_A0_A1_EPLR_V1_COMPARISON_"
        "AND_CARRY_FORWARD_DECISION"
    )
    print(f"merged_outputs={merged_path}")
    print(f"merged_outputs_sha256={sha256_file(merged_path)}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
