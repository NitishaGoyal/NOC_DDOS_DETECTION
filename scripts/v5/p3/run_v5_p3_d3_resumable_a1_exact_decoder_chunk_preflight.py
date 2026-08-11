from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = "V5_P3_D3_RESUMABLE_A1_EXACT_DECODER_CHUNK_PREFLIGHT"
CAMPAIGN_LABEL = "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic"
EXPECTED_ITEMS = 13_863
CHUNK_SIZE = 16
INACTIVE_PROBES = 16
ACTIVE_PROBES_PER_COUNT = 12
EXPECTED_SELECTED_ITEMS = (
    INACTIVE_PROBES + 4 * ACTIVE_PROBES_PER_COUNT
)
A1_MARGIN_THRESHOLD = 8.7205320882398425


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


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {
            str(key): jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(
                f"non-finite float encountered in decoder output: {value}"
            )
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            key: jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def select_evenly(
    indices: np.ndarray,
    requested: int,
) -> list[int]:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size < requested:
        raise RuntimeError(
            f"need {requested} indices, found {indices.size}"
        )
    positions = np.linspace(
        0,
        indices.size - 1,
        requested,
        dtype=np.int64,
    )
    selected = indices[positions].tolist()
    if len(set(selected)) != requested:
        raise RuntimeError(
            "evenly spaced probe selection produced duplicates"
        )
    return [int(value) for value in selected]


def choose_probe_indices(
    graph: np.ndarray,
    count: np.ndarray,
) -> tuple[list[int], dict[str, list[int]]]:
    inactive = np.flatnonzero(graph == 0)
    groups: dict[str, list[int]] = {
        "inactive": select_evenly(
            inactive,
            INACTIVE_PROBES,
        )
    }
    for attacker_count in (1, 2, 3, 4):
        active = np.flatnonzero(
            (graph == 1) & (count == attacker_count)
        )
        groups[f"K{attacker_count}"] = select_evenly(
            active,
            ACTIVE_PROBES_PER_COUNT,
        )
    ordered = (
        groups["inactive"]
        + groups["K1"]
        + groups["K2"]
        + groups["K3"]
        + groups["K4"]
    )
    if len(ordered) != EXPECTED_SELECTED_ITEMS:
        raise RuntimeError("selected-item count mismatch")
    if len(set(ordered)) != len(ordered):
        raise RuntimeError("probe groups overlap")
    return ordered, groups


def chunk_ranges(total: int):
    for chunk_id, start in enumerate(
        range(0, total, CHUNK_SIZE)
    ):
        stop = min(start + CHUNK_SIZE, total)
        yield chunk_id, start, stop


def chunk_paths(
    working_dir: Path,
    chunk_id: int,
    start: int,
    stop: int,
) -> tuple[Path, Path]:
    stem = (
        f"chunk_{chunk_id:03d}_"
        f"selected_{start:04d}_{stop:04d}"
    )
    return (
        working_dir / f"{stem}.json",
        working_dir / f"{stem}.manifest.json",
    )


def validate_chunk(
    chunk_path: Path,
    manifest_path: Path,
    expected_indices: list[int],
) -> dict[str, Any]:
    if not chunk_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    actual_sha = sha256_file(chunk_path)
    if manifest.get("chunk_sha256") != actual_sha:
        raise RuntimeError(
            f"chunk SHA mismatch: {chunk_path}"
        )
    payload = json.loads(
        chunk_path.read_text(encoding="utf-8")
    )
    observed_indices = [
        int(item["dataset_index"])
        for item in payload["items"]
    ]
    if observed_indices != expected_indices:
        raise RuntimeError(
            f"chunk coverage changed for {chunk_path}: "
            f"{observed_indices} != {expected_indices}"
        )
    if payload.get("status") != "COMPLETE":
        raise RuntimeError(
            f"chunk is not COMPLETE: {chunk_path}"
        )
    if int(payload.get("item_count", -1)) != len(
        expected_indices
    ):
        raise RuntimeError(
            f"chunk item count changed: {chunk_path}"
        )
    return payload


def decode_item(
    decoder,
    arrays: dict[str, np.ndarray],
    dataset_index: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    hypothesis = decoder.decode_best_attack_hypothesis(
        float(arrays["attack_logits"][dataset_index]),
        arrays["count_logits"][dataset_index].astype(
            np.float64
        ),
        arrays["source_logits"][dataset_index].astype(
            np.float64
        ),
        arrays["transit_logits"][dataset_index].astype(
            np.float64
        ),
        arrays["victim_logits"][dataset_index].astype(
            np.float64
        ),
        arrays["path_logits"][dataset_index].astype(
            np.float64
        ),
    )
    decoded = decoder.apply_margin_threshold(
        hypothesis,
        A1_MARGIN_THRESHOLD,
    )
    elapsed = time.perf_counter() - started

    hypothesis_value = jsonable(hypothesis)
    decoded_value = jsonable(decoded)
    margin = getattr(hypothesis, "margin", None)
    if margin is None:
        margin = hypothesis_value.get("margin")
    if margin is None or not math.isfinite(float(margin)):
        raise RuntimeError(
            f"missing or non-finite margin at item {dataset_index}"
        )

    semantic = {
        "hypothesis": hypothesis_value,
        "decoded": decoded_value,
    }
    return {
        "dataset_index": int(dataset_index),
        "graph_truth": int(
            arrays["y_attack"][dataset_index]
        ),
        "attacker_count_truth": int(
            arrays["y_attacker_count"][dataset_index]
        ),
        "margin": float(margin),
        "decode_seconds": elapsed,
        "semantic_sha256": sha256_json(semantic),
        **semantic,
    }


def main() -> int:
    args = parse_args()

    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(
        args.installed_script
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    working_dir = output_dir / "A1_PREFLIGHT_WORKING"
    working_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    selection_path = output_dir / (
        "V5_P3_D3_A1_EXACT_PREFLIGHT_SELECTION.json"
    )
    merged_path = output_dir / (
        "V5_P3_D3_A1_EXACT_PREFLIGHT_MERGED_RESULTS.json"
    )
    complete_path = output_dir / f"{STAGE}_COMPLETE"
    hold_path = output_dir / f"{STAGE}_HOLD"

    d2_dir = (
        repo
        / "reports/v5/p3_d2_raw_and_frozen_a0_validation_evaluation"
    )
    d2_report_path = d2_dir / (
        "V5_P3_D2_RAW_AND_FROZEN_A0_"
        "TRANCHE_A_VALIDATION_EVALUATION_REPORT.json"
    )
    d2_lock_path = d2_dir / (
        "V5_P3_D2_RAW_AND_FROZEN_A0_"
        "TRANCHE_A_VALIDATION_EVALUATION_LOCK.json"
    )
    d1_dir = (
        repo
        / "reports/v5/p3_d1_immutable_validation_logit_export"
    )
    d1_export_path = d1_dir / (
        "V5_P3_D1_TRANCHE_A_VALIDATION_LOGITS_AND_LABELS.npz"
    )
    d1_lock_path = d1_dir / (
        "V5_P3_D1_IMMUTABLE_TRANCHE_A_"
        "VALIDATION_LOGIT_EXPORT_LOCK.json"
    )
    d0_dir = (
        repo
        / "reports/v5/p3_d0_frozen_decoder_transfer_preflight"
    )
    d0_lock_path = d0_dir / (
        "V5_P3_D0_FROZEN_DECODER_TRANSFER_PREFLIGHT_LOCK.json"
    )
    decoder_path = (
        repo
        / "src/decoders/v5_legal_xy_exact_decoder_certified.py"
    )
    preserved_decoder_path = (
        repo
        / "reports/v5/p2_canonical_preservation/decoder/"
        "v5_legal_xy_exact_decoder_certified.py"
    )

    required = [
        d2_report_path,
        d2_lock_path,
        d1_export_path,
        d1_lock_path,
        d0_lock_path,
        decoder_path,
        preserved_decoder_path,
    ]
    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            f"required artifacts missing: {missing}"
        )

    d2_report = json.loads(
        d2_report_path.read_text(encoding="utf-8")
    )
    d2_lock = json.loads(
        d2_lock_path.read_text(encoding="utf-8")
    )
    d1_lock = json.loads(
        d1_lock_path.read_text(encoding="utf-8")
    )
    d0_lock = json.loads(
        d0_lock_path.read_text(encoding="utf-8")
    )

    if d2_report.get("status") != "PASS":
        raise RuntimeError("D2 report is not PASS")
    if not d2_report.get("decision", {}).get(
        "A1_exact_resumable_preflight_authorized"
    ):
        raise RuntimeError(
            "D2 did not authorize the A1 chunk preflight"
        )
    if d2_lock.get("report_sha256") != sha256_file(
        d2_report_path
    ):
        raise RuntimeError("D2 report/lock SHA mismatch")
    if d1_lock.get("export_sha256") != sha256_file(
        d1_export_path
    ):
        raise RuntimeError("D1 export SHA mismatch")
    decoder_sha = sha256_file(decoder_path)
    preserved_sha = sha256_file(preserved_decoder_path)
    if decoder_sha != preserved_sha:
        raise RuntimeError(
            "working certified decoder differs from "
            "canonical preservation copy"
        )
    if d0_lock.get(
        "certified_decoder_sha256"
    ) != decoder_sha:
        raise RuntimeError(
            "certified decoder differs from D0"
        )
    if float(
        d0_lock.get("A1_margin_threshold")
    ) != A1_MARGIN_THRESHOLD:
        raise RuntimeError(
            "A1 margin threshold differs from D0"
        )

    with np.load(
        d1_export_path,
        allow_pickle=False,
    ) as loaded:
        arrays = {
            key: loaded[key].copy()
            for key in loaded.files
        }

    expected_shapes = {
        "attack_logits": (EXPECTED_ITEMS,),
        "count_logits": (EXPECTED_ITEMS, 4),
        "source_logits": (EXPECTED_ITEMS, 16),
        "transit_logits": (EXPECTED_ITEMS, 16),
        "victim_logits": (EXPECTED_ITEMS, 16),
        "path_logits": (EXPECTED_ITEMS, 16),
        "y_attack": (EXPECTED_ITEMS,),
        "y_attacker_count": (EXPECTED_ITEMS,),
    }
    for key, expected_shape in expected_shapes.items():
        if key not in arrays:
            raise RuntimeError(
                f"D1 export missing {key}"
            )
        if arrays[key].shape != expected_shape:
            raise RuntimeError(
                f"{key} shape={arrays[key].shape}, "
                f"expected={expected_shape}"
            )
    for key in (
        "attack_logits",
        "count_logits",
        "source_logits",
        "transit_logits",
        "victim_logits",
        "path_logits",
    ):
        if not np.isfinite(arrays[key]).all():
            raise RuntimeError(
                f"non-finite values in {key}"
            )

    selected_indices, selection_groups = (
        choose_probe_indices(
            arrays["y_attack"].astype(np.int64),
            arrays["y_attacker_count"].astype(
                np.int64
            ),
        )
    )
    selection = {
        "stage": STAGE,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "selection_policy": (
            "16 evenly spaced inactive items and 12 "
            "evenly spaced active items for each K1-K4"
        ),
        "selected_items": len(selected_indices),
        "chunk_size": CHUNK_SIZE,
        "groups": selection_groups,
        "ordered_indices": selected_indices,
        "D1_export_sha256": sha256_file(
            d1_export_path
        ),
        "certified_decoder_sha256": decoder_sha,
        "A1_margin_threshold": A1_MARGIN_THRESHOLD,
    }
    if selection_path.is_file():
        existing = json.loads(
            selection_path.read_text(encoding="utf-8")
        )
        if existing != selection:
            raise RuntimeError(
                "existing D3 selection contract changed"
            )
    else:
        atomic_json(selection_path, selection)

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    decoder = importlib.import_module(
        "src.decoders."
        "v5_legal_xy_exact_decoder_certified"
    )

    solver_identity = jsonable(
        decoder.solver_identity()
    )
    certification_policy = jsonable(
        decoder.certification_policy()
    )

    first_pass_decoded_chunks = 0
    first_pass_reused_chunks = 0
    all_items: list[dict[str, Any]] = []

    for chunk_id, start, stop in chunk_ranges(
        len(selected_indices)
    ):
        expected_indices = selected_indices[start:stop]
        chunk_path, manifest_path = chunk_paths(
            working_dir,
            chunk_id,
            start,
            stop,
        )
        try:
            payload = validate_chunk(
                chunk_path,
                manifest_path,
                expected_indices,
            )
        except FileNotFoundError:
            items = []
            for position, dataset_index in enumerate(
                expected_indices,
                start=1,
            ):
                item = decode_item(
                    decoder,
                    arrays,
                    dataset_index,
                )
                items.append(item)
                print(
                    f"chunk={chunk_id + 1}/"
                    f"{math.ceil(len(selected_indices) / CHUNK_SIZE)} "
                    f"item={position}/{len(expected_indices)} "
                    f"dataset_index={dataset_index} "
                    f"seconds={item['decode_seconds']:.6f}",
                    flush=True,
                )
            payload = {
                "stage": STAGE,
                "status": "COMPLETE",
                "chunk_id": chunk_id,
                "selected_start": start,
                "selected_stop": stop,
                "item_count": len(items),
                "dataset_indices": expected_indices,
                "items": items,
            }
            atomic_json(chunk_path, payload)
            manifest = {
                "stage": STAGE,
                "chunk_id": chunk_id,
                "selected_start": start,
                "selected_stop": stop,
                "dataset_indices": expected_indices,
                "chunk_path": str(chunk_path),
                "chunk_sha256": sha256_file(
                    chunk_path
                ),
                "item_count": len(items),
            }
            atomic_json(manifest_path, manifest)
            payload = validate_chunk(
                chunk_path,
                manifest_path,
                expected_indices,
            )
            first_pass_decoded_chunks += 1
        else:
            first_pass_reused_chunks += 1

        all_items.extend(payload["items"])

    if [
        int(item["dataset_index"])
        for item in all_items
    ] != selected_indices:
        raise RuntimeError(
            "merged D3 chunk coverage is not exact"
        )

    # Simulate a restart/resume pass: all chunks must be
    # reusable solely from their manifests and hashes.
    second_pass_reused_chunks = 0
    for chunk_id, start, stop in chunk_ranges(
        len(selected_indices)
    ):
        expected_indices = selected_indices[start:stop]
        chunk_path, manifest_path = chunk_paths(
            working_dir,
            chunk_id,
            start,
            stop,
        )
        validate_chunk(
            chunk_path,
            manifest_path,
            expected_indices,
        )
        second_pass_reused_chunks += 1

    expected_chunk_count = math.ceil(
        len(selected_indices) / CHUNK_SIZE
    )
    if second_pass_reused_chunks != expected_chunk_count:
        raise RuntimeError(
            "resume-validation chunk count mismatch"
        )

    # Re-decode first and last item of every chunk and
    # require exact semantic identity.
    stored_by_index = {
        int(item["dataset_index"]): item
        for item in all_items
    }
    deterministic_indices = []
    for _, start, stop in chunk_ranges(
        len(selected_indices)
    ):
        deterministic_indices.append(
            selected_indices[start]
        )
        deterministic_indices.append(
            selected_indices[stop - 1]
        )
    deterministic_indices = list(
        dict.fromkeys(deterministic_indices)
    )
    deterministic_redecode = []
    for dataset_index in deterministic_indices:
        repeated = decode_item(
            decoder,
            arrays,
            dataset_index,
        )
        expected_sha = stored_by_index[
            dataset_index
        ]["semantic_sha256"]
        match = (
            repeated["semantic_sha256"]
            == expected_sha
        )
        deterministic_redecode.append({
            "dataset_index": dataset_index,
            "expected_semantic_sha256": (
                expected_sha
            ),
            "repeated_semantic_sha256": (
                repeated["semantic_sha256"]
            ),
            "exact_match": match,
            "repeat_decode_seconds": (
                repeated["decode_seconds"]
            ),
        })
        if not match:
            raise RuntimeError(
                "certified decoder semantic output "
                f"changed on repeat for {dataset_index}"
            )

    decode_seconds = np.asarray(
        [
            float(item["decode_seconds"])
            for item in all_items
        ],
        dtype=np.float64,
    )
    margins = np.asarray(
        [
            float(item["margin"])
            for item in all_items
        ],
        dtype=np.float64,
    )
    if not np.isfinite(decode_seconds).all():
        raise RuntimeError(
            "non-finite D3 decode time"
        )
    if not np.isfinite(margins).all():
        raise RuntimeError(
            "non-finite D3 decoder margin"
        )

    estimated_full_seconds = float(
        decode_seconds.mean() * EXPECTED_ITEMS
    )
    merged = {
        "stage": STAGE,
        "status": "COMPLETE",
        "campaign_label": CAMPAIGN_LABEL,
        "selected_items": len(all_items),
        "selected_indices": selected_indices,
        "selection_groups": selection_groups,
        "chunk_size": CHUNK_SIZE,
        "chunk_count": expected_chunk_count,
        "first_pass_decoded_chunks": (
            first_pass_decoded_chunks
        ),
        "first_pass_reused_chunks": (
            first_pass_reused_chunks
        ),
        "second_pass_reused_chunks": (
            second_pass_reused_chunks
        ),
        "deterministic_redecode": (
            deterministic_redecode
        ),
        "timing": {
            "mean_seconds": float(
                decode_seconds.mean()
            ),
            "median_seconds": float(
                np.median(decode_seconds)
            ),
            "p95_seconds": float(
                np.quantile(
                    decode_seconds,
                    0.95,
                )
            ),
            "maximum_seconds": float(
                decode_seconds.max()
            ),
            "estimated_full_13863_seconds": (
                estimated_full_seconds
            ),
            "estimated_full_13863_hours": (
                estimated_full_seconds / 3600.0
            ),
        },
        "margin": {
            "minimum": float(margins.min()),
            "median": float(
                np.median(margins)
            ),
            "maximum": float(margins.max()),
            "finite": True,
        },
        "items": all_items,
    }
    atomic_json(merged_path, merged)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "scope": (
            "Resumable atomic-chunk preflight of the "
            "frozen certified P2 A1 exact decoder on a "
            "balanced 64-item subset of the immutable "
            "V5-P3 A-validation export."
        ),
        "selection": {
            "path": str(selection_path),
            "sha256": sha256_file(
                selection_path
            ),
            "items": len(selected_indices),
            "groups": selection_groups,
        },
        "decoder": {
            "path": str(decoder_path),
            "sha256": decoder_sha,
            "working_matches_canonical_preservation": (
                True
            ),
            "solver_identity": solver_identity,
            "certification_policy": (
                certification_policy
            ),
            "A1_margin_threshold": (
                A1_MARGIN_THRESHOLD
            ),
        },
        "chunk_protocol": {
            "chunk_size": CHUNK_SIZE,
            "chunk_count": expected_chunk_count,
            "working_directory": str(
                working_dir
            ),
            "atomic_json_chunks": True,
            "per_chunk_hash_manifests": True,
            "first_pass_decoded_chunks": (
                first_pass_decoded_chunks
            ),
            "first_pass_reused_chunks": (
                first_pass_reused_chunks
            ),
            "second_pass_reused_chunks": (
                second_pass_reused_chunks
            ),
            "resume_validation_pass": (
                second_pass_reused_chunks
                == expected_chunk_count
            ),
        },
        "numerical_and_determinism": {
            "all_selected_items_decoded": True,
            "certification_exceptions": 0,
            "all_margins_finite": True,
            "repeat_items": len(
                deterministic_redecode
            ),
            "repeat_semantic_outputs_exact": (
                all(
                    row["exact_match"]
                    for row in deterministic_redecode
                )
            ),
            "merged_results_path": str(
                merged_path
            ),
            "merged_results_sha256": (
                sha256_file(merged_path)
            ),
        },
        "timing": merged["timing"],
        "decision": {
            "resumable_A1_chunk_protocol_certified": (
                True
            ),
            "A1_exact_full_validation_evaluation_authorized": (
                True
            ),
            "A1_margin_retuning_authorized": False,
            "beam_decoder_selected": False,
            "sealed_test_evaluation_authorized": (
                False
            ),
            "next_stage": (
                "V5_P3_D4_RESUMABLE_A1_EXACT_"
                "TRANCHE_A_VALIDATION_EVALUATION"
            ),
        },
        "sealed_test": {
            "test_dataset_instantiated": False,
            "test_length_computed": False,
            "test_tensor_loaded": False,
            "evaluation_authorized": False,
        },
        "provenance": {
            "D2_report_sha256": sha256_file(
                d2_report_path
            ),
            "D2_lock_sha256": sha256_file(
                d2_lock_path
            ),
            "D1_export_sha256": sha256_file(
                d1_export_path
            ),
            "D0_lock_sha256": sha256_file(
                d0_lock_path
            ),
            "installed_script_sha256": (
                sha256_file(installed_script)
            ),
            "python_version": (
                platform.python_version()
            ),
            "numpy_version": np.__version__,
        },
        "model_replayed": False,
        "model_trained": False,
        "threshold_tuning_performed": False,
        "certified_dataset_modified": False,
        "generalization_claim_authorized": False,
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(
            report_path
        ),
        "selection_sha256": sha256_file(
            selection_path
        ),
        "merged_results_sha256": sha256_file(
            merged_path
        ),
        "D1_export_sha256": sha256_file(
            d1_export_path
        ),
        "certified_decoder_sha256": (
            decoder_sha
        ),
        "A1_margin_threshold": (
            A1_MARGIN_THRESHOLD
        ),
        "selected_items": len(
            selected_indices
        ),
        "chunk_size": CHUNK_SIZE,
        "chunk_count": expected_chunk_count,
        "second_pass_reused_chunks": (
            second_pass_reused_chunks
        ),
        "deterministic_repeat_items": len(
            deterministic_redecode
        ),
        "test_tensor_loaded": False,
        "certified_dataset_modified": False,
    }
    atomic_json(lock_path, lock)

    complete_path.write_text(
        f"{STAGE}_COMPLETE\n",
        encoding="utf-8",
    )
    if hold_path.exists():
        hold_path.unlink()

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign_label={CAMPAIGN_LABEL}")
    print(
        f"selected_items={len(selected_indices)}"
    )
    print(
        "selection_distribution="
        "inactive16,K1_12,K2_12,K3_12,K4_12"
    )
    print(f"chunk_size={CHUNK_SIZE}")
    print(
        f"chunk_count={expected_chunk_count}"
    )
    print(
        "first_pass_decoded_chunks="
        f"{first_pass_decoded_chunks}"
    )
    print(
        "first_pass_reused_chunks="
        f"{first_pass_reused_chunks}"
    )
    print(
        "second_pass_reused_chunks="
        f"{second_pass_reused_chunks}"
    )
    print(
        "resume_validation_pass=true"
    )
    print(
        "certification_exceptions=0"
    )
    print("all_margins_finite=true")
    print(
        "deterministic_repeat_items="
        f"{len(deterministic_redecode)}"
    )
    print(
        "repeat_semantic_outputs_exact=true"
    )
    print(
        "mean_exact_decode_seconds="
        f"{decode_seconds.mean():.6f}"
    )
    print(
        "p95_exact_decode_seconds="
        f"{np.quantile(decode_seconds, 0.95):.6f}"
    )
    print(
        "maximum_exact_decode_seconds="
        f"{decode_seconds.max():.6f}"
    )
    print(
        "estimated_full_validation_hours="
        f"{estimated_full_seconds / 3600.0:.3f}"
    )
    print("threshold_tuning_performed=false")
    print("A1_margin_retuning_authorized=false")
    print("beam_decoder_selected=false")
    print("test_dataset_instantiated=false")
    print("test_length_computed=false")
    print("test_tensor_loaded=false")
    print(
        "A1_exact_full_validation_"
        "evaluation_authorized=true"
    )
    print(
        "sealed_test_evaluation_authorized=false"
    )
    print(
        "next_stage="
        "V5_P3_D4_RESUMABLE_A1_EXACT_"
        "TRANCHE_A_VALIDATION_EVALUATION"
    )
    print(f"selection={selection_path}")
    print(f"merged_results={merged_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
