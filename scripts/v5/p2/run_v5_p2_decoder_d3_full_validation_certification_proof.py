#!/usr/bin/env python3
"""V5 P2 D3 full validation certification proof.

Runs the D2 certified exact decoder over all 12,528 frozen validation-logit
items, persists chunk artifacts atomically, supports safe resume, and freezes a
full validation numerical-certificate report.

This stage:
- reads the immutable validation-logit archive only;
- uses the versioned D2 certified decoder only;
- applies the already frozen A1 margin threshold only for output accounting;
- performs no model inference and no threshold selection;
- never forms or inspects the P2 test path;
- creates no evaluation authorization.

Interrupted executions are resumed from hash-verified chunk artifacts. Final
D3 artifacts are created only after complete coverage and deterministic replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np


EXPECTED = {
    "d1_contract": (
        "b63efbfe1272fb7f6ab922645182fba86e28d7c8f42a70e0f3ac453b88bec71f"
    ),
    "d2_artifact": (
        "fcb78ae4059f49e4c78d4e3cf603160f7d8bad72857d6ef6b777cc4de0ab7055"
    ),
    "certified_decoder": (
        "30309376cb0990a0f482045e2dac413d4370729cf94f9e556f895af2c5c8c71d"
    ),
    "legacy_decoder": (
        "8da32b3ca3915365b8b01049a4ddc6a043583aa7193589ccf4af8116a4b0783c"
    ),
    "route_library": (
        "3ae860817ddef7cdbc0360057c51dc86908337b40b1df4c7df8fc2c8b82ea0d7"
    ),
    "validation_manifest": (
        "ac7a38fd1f4b00b578254d9253418655aa78b6cc7a7a1d65f78ba15487b14604"
    ),
}

EXPECTED_POLICY_ID = "V5P2-D1-6ce8cc7c76b8d2cff367631077d65f20"
EXPECTED_VALIDATION_ITEMS = 12_528
EXPECTED_A1_MARGIN_THRESHOLD = 8.7205320882398425
OLD_REPORTING_CEILING = 1.4210854715202004e-14
D3_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: observed={observed}, expected={expected}"
        )
    return observed


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def mask_to_int(value: Any) -> int:
    integer = int(value)
    if integer < 0 or integer > 0xFFFF:
        raise RuntimeError(f"router mask outside uint16 range: {integer}")
    return integer


def padded_routes(route_ids: tuple[int, ...]) -> np.ndarray:
    result = np.full(4, -1, dtype=np.int16)
    if len(route_ids) > 4:
        raise RuntimeError(f"too many routes: {route_ids}")
    if route_ids:
        result[: len(route_ids)] = np.asarray(route_ids, dtype=np.int16)
    return result


def evenly_spaced_indices(total: int, requested: int) -> np.ndarray:
    count = min(max(0, int(requested)), int(total))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == total:
        return np.arange(total, dtype=np.int64)
    values = np.rint(
        np.linspace(0, total - 1, num=count, endpoint=True)
    ).astype(np.int64)
    values = np.unique(values)
    if values.size < count:
        present = set(values.tolist())
        supplement = [
            index for index in range(total)
            if index not in present
        ][: count - values.size]
        values = np.concatenate(
            [values, np.asarray(supplement, dtype=np.int64)]
        )
    return np.sort(values[:count])


def fingerprint_hypothesis(value: Any) -> str:
    payload = {
        "margin_hex": float(value.margin).hex(),
        "attacker_count": int(value.attacker_count),
        "route_ids": [int(item) for item in value.route_ids],
        "source_mask": int(value.source_mask),
        "transit_mask": int(value.transit_mask),
        "victim_mask": int(value.victim_mask),
        "path_mask": int(value.path_mask),
        "primary_gap_hex": float(
            value.primary_certificate.raw_reported_mip_gap
        ).hex(),
        "lexicographic_gap_hex": float(
            value.lexicographic_certificate.raw_reported_mip_gap
        ).hex(),
        "high_precision_semantic_face_difference": str(
            value.high_precision_semantic_face_difference
        ),
        "high_precision_rejections": int(
            value.lexicographic_candidates_rejected_by_high_precision
        ),
        "certificate_policy_id": str(value.certificate_policy_id),
        "certificate_status": str(value.certificate_status),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def chunk_ranges(total: int, chunk_size: int) -> Iterator[tuple[int, int, int]]:
    chunk_id = 0
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        yield chunk_id, start, stop
        chunk_id += 1


def expected_chunk_name(chunk_id: int, start: int, stop: int) -> str:
    return f"chunk_{chunk_id:05d}_{start:05d}_{stop:05d}.npz"


def validate_chunk(
    path: Path,
    *,
    expected_start: int,
    expected_stop: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as value:
        required = {
            "validation_index",
            "margin",
            "attacker_count",
            "route_ids",
            "source_mask",
            "transit_mask",
            "victim_mask",
            "path_mask",
            "attack_after_threshold",
            "primary_gap",
            "lexicographic_gap",
            "primary_integrality_error",
            "lexicographic_integrality_error",
            "primary_bound_violation",
            "lexicographic_bound_violation",
            "primary_linear_violation",
            "lexicographic_linear_violation",
            "primary_objective_difference",
            "lexicographic_objective_difference",
            "high_precision_rejections",
            "fingerprint_digest",
        }
        missing = required - set(value.files)
        if missing:
            raise RuntimeError(
                f"chunk missing arrays {sorted(missing)}: {path}"
            )

        indices = np.asarray(value["validation_index"], dtype=np.int64)
        expected_indices = np.arange(
            expected_start,
            expected_stop,
            dtype=np.int64,
        )
        if not np.array_equal(indices, expected_indices):
            raise RuntimeError(
                f"chunk index coverage mismatch: {path}"
            )

        rows = expected_stop - expected_start
        if np.asarray(value["route_ids"]).shape != (rows, 4):
            raise RuntimeError(f"chunk route shape mismatch: {path}")
        if np.asarray(value["fingerprint_digest"]).shape != (rows,):
            raise RuntimeError(f"chunk fingerprint shape mismatch: {path}")

        summary = {
            "rows": rows,
            "minimum_primary_gap": float(
                np.min(value["primary_gap"])
            ),
            "maximum_primary_gap": float(
                np.max(value["primary_gap"])
            ),
            "minimum_lexicographic_gap": float(
                np.min(value["lexicographic_gap"])
            ),
            "maximum_lexicographic_gap": float(
                np.max(value["lexicographic_gap"])
            ),
            "maximum_integrality_error": float(
                max(
                    np.max(value["primary_integrality_error"]),
                    np.max(value["lexicographic_integrality_error"]),
                )
            ),
            "maximum_bound_violation": float(
                max(
                    np.max(value["primary_bound_violation"]),
                    np.max(value["lexicographic_bound_violation"]),
                )
            ),
            "maximum_linear_violation": float(
                max(
                    np.max(value["primary_linear_violation"]),
                    np.max(value["lexicographic_linear_violation"]),
                )
            ),
            "maximum_objective_difference": float(
                max(
                    np.max(value["primary_objective_difference"]),
                    np.max(value["lexicographic_objective_difference"]),
                )
            ),
            "attack_after_threshold_count": int(
                np.sum(value["attack_after_threshold"])
            ),
            "positive_gap_count": int(
                np.count_nonzero(
                    np.asarray(value["primary_gap"]) > 0.0
                )
                + np.count_nonzero(
                    np.asarray(value["lexicographic_gap"]) > 0.0
                )
            ),
            "above_old_gate_count": int(
                np.count_nonzero(
                    np.asarray(value["primary_gap"])
                    > OLD_REPORTING_CEILING
                )
                + np.count_nonzero(
                    np.asarray(value["lexicographic_gap"])
                    > OLD_REPORTING_CEILING
                )
            ),
            "high_precision_rejection_total": int(
                np.sum(value["high_precision_rejections"])
            ),
        }
    return summary


def decode_chunk(
    *,
    certified: Any,
    arrays: dict[str, np.ndarray],
    start: int,
    stop: int,
    threshold: float,
) -> dict[str, np.ndarray]:
    rows = stop - start
    validation_index = np.arange(start, stop, dtype=np.int64)
    margin = np.empty(rows, dtype=np.float64)
    attacker_count = np.empty(rows, dtype=np.uint8)
    route_ids = np.empty((rows, 4), dtype=np.int16)
    source_mask = np.empty(rows, dtype=np.uint16)
    transit_mask = np.empty(rows, dtype=np.uint16)
    victim_mask = np.empty(rows, dtype=np.uint16)
    path_mask = np.empty(rows, dtype=np.uint16)
    attack_after_threshold = np.empty(rows, dtype=np.uint8)

    primary_gap = np.empty(rows, dtype=np.float64)
    lexicographic_gap = np.empty(rows, dtype=np.float64)
    primary_integrality_error = np.empty(rows, dtype=np.float64)
    lexicographic_integrality_error = np.empty(rows, dtype=np.float64)
    primary_bound_violation = np.empty(rows, dtype=np.float64)
    lexicographic_bound_violation = np.empty(rows, dtype=np.float64)
    primary_linear_violation = np.empty(rows, dtype=np.float64)
    lexicographic_linear_violation = np.empty(rows, dtype=np.float64)
    primary_objective_difference = np.empty(rows, dtype=np.float64)
    lexicographic_objective_difference = np.empty(rows, dtype=np.float64)
    high_precision_rejections = np.empty(rows, dtype=np.uint16)
    fingerprint_digest = np.empty(rows, dtype="S64")

    for local, index in enumerate(range(start, stop)):
        hypothesis = certified.decode_best_attack_hypothesis(
            float(arrays["graph_logits"][index]),
            np.asarray(
                arrays["count_logits"][index],
                dtype=np.float64,
            ),
            np.asarray(
                arrays["source_logits"][index],
                dtype=np.float64,
            ),
            np.asarray(
                arrays["transit_logits"][index],
                dtype=np.float64,
            ),
            np.asarray(
                arrays["victim_logits"][index],
                dtype=np.float64,
            ),
            np.asarray(
                arrays["path_logits"][index],
                dtype=np.float64,
            ),
        )
        if hypothesis.certificate_status != "CERTIFIED":
            raise RuntimeError(
                f"uncertified hypothesis at validation index {index}"
            )
        if hypothesis.certificate_policy_id != EXPECTED_POLICY_ID:
            raise RuntimeError(
                f"wrong policy at validation index {index}"
            )

        primary = hypothesis.primary_certificate
        lexicographic = hypothesis.lexicographic_certificate
        if (
            primary.certificate_status != "CERTIFIED"
            or lexicographic.certificate_status != "CERTIFIED"
        ):
            raise RuntimeError(
                f"uncertified MILP at validation index {index}"
            )

        decoded = certified.apply_margin_threshold(
            hypothesis,
            threshold,
        )

        margin[local] = float(hypothesis.margin)
        attacker_count[local] = int(hypothesis.attacker_count)
        route_ids[local] = padded_routes(hypothesis.route_ids)
        source_mask[local] = mask_to_int(hypothesis.source_mask)
        transit_mask[local] = mask_to_int(hypothesis.transit_mask)
        victim_mask[local] = mask_to_int(hypothesis.victim_mask)
        path_mask[local] = mask_to_int(hypothesis.path_mask)
        attack_after_threshold[local] = int(decoded.attack)

        primary_gap[local] = float(primary.raw_reported_mip_gap)
        lexicographic_gap[local] = float(
            lexicographic.raw_reported_mip_gap
        )
        primary_integrality_error[local] = float(
            primary.maximum_integrality_error
        )
        lexicographic_integrality_error[local] = float(
            lexicographic.maximum_integrality_error
        )
        primary_bound_violation[local] = float(
            primary.maximum_bound_violation
        )
        lexicographic_bound_violation[local] = float(
            lexicographic.maximum_bound_violation
        )
        primary_linear_violation[local] = float(
            primary.maximum_linear_constraint_violation
        )
        lexicographic_linear_violation[local] = float(
            lexicographic.maximum_linear_constraint_violation
        )
        primary_objective_difference[local] = float(
            primary.objective_absolute_difference
        )
        lexicographic_objective_difference[local] = float(
            lexicographic.objective_absolute_difference
        )
        high_precision_rejections[local] = int(
            hypothesis
            .lexicographic_candidates_rejected_by_high_precision
        )
        fingerprint_digest[local] = fingerprint_hypothesis(
            hypothesis
        ).encode("ascii")

    return {
        "validation_index": validation_index,
        "margin": margin,
        "attacker_count": attacker_count,
        "route_ids": route_ids,
        "source_mask": source_mask,
        "transit_mask": transit_mask,
        "victim_mask": victim_mask,
        "path_mask": path_mask,
        "attack_after_threshold": attack_after_threshold,
        "primary_gap": primary_gap,
        "lexicographic_gap": lexicographic_gap,
        "primary_integrality_error": primary_integrality_error,
        "lexicographic_integrality_error": (
            lexicographic_integrality_error
        ),
        "primary_bound_violation": primary_bound_violation,
        "lexicographic_bound_violation": (
            lexicographic_bound_violation
        ),
        "primary_linear_violation": primary_linear_violation,
        "lexicographic_linear_violation": (
            lexicographic_linear_violation
        ),
        "primary_objective_difference": (
            primary_objective_difference
        ),
        "lexicographic_objective_difference": (
            lexicographic_objective_difference
        ),
        "high_precision_rejections": high_precision_rejections,
        "fingerprint_digest": fingerprint_digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument(
        "--determinism-replay-items",
        type=int,
        default=64,
    )
    args = parser.parse_args()

    root = args.repo_root.expanduser().resolve()
    chunk_size = max(1, int(args.chunk_size))

    paths = {
        "d1_contract": (
            root
            / "artifacts/v5/p2_decoder_d1_prospective_numerical_policy_freeze"
            / "V5_P2_D1_PROSPECTIVE_NUMERICAL_CERTIFICATION_POLICY.json"
        ),
        "d2_artifact": (
            root
            / "artifacts/v5/p2_decoder_d2_certified_implementation"
            / "V5_P2_DECODER_D2_CERTIFIED_IMPLEMENTATION.json"
        ),
        "certified_decoder": (
            root
            / "src/decoders/v5_legal_xy_exact_decoder_certified.py"
        ),
        "legacy_decoder": (
            root / "src/decoders/v5_legal_xy_exact_decoder.py"
        ),
        "route_library": (
            root / "src/decoders/v5_xy_route_library.py"
        ),
        "validation_manifest": (
            root
            / "artifacts/v5/p2_e1_immutable_validation_logit_archive"
            / "ARCHIVE_MANIFEST.json"
        ),
        "l5_report": (
            root
            / "reports/v5/p2_l5_final_pretest_freeze"
            / "V5_P2_L5_FINAL_PRETEST_FREEZE.json"
        ),
    }
    archive = (
        root / "artifacts/v5/p2_e1_immutable_validation_logit_archive"
    )

    observed_hashes = {
        label: require_hash(paths[label], EXPECTED[label], label)
        for label in (
            "d1_contract",
            "d2_artifact",
            "certified_decoder",
            "legacy_decoder",
            "route_library",
            "validation_manifest",
        )
    }
    if not paths["l5_report"].is_file():
        raise FileNotFoundError(
            f"L5 final pretest report missing: {paths['l5_report']}"
        )

    d1 = read_json(paths["d1_contract"])
    d2 = read_json(paths["d2_artifact"])
    l5 = read_json(paths["l5_report"])

    if d1.get("policy_id") != EXPECTED_POLICY_ID:
        raise RuntimeError("D1 policy ID changed")
    if d2.get("policy_id") != EXPECTED_POLICY_ID:
        raise RuntimeError("D2 policy ID changed")
    if not bool(d2["implementation"]["legacy_decoder_preserved"]):
        raise RuntimeError("D2 does not preserve the legacy decoder")

    threshold_values = []
    for key, value in l5.items():
        if key == "A1_margin_threshold" and isinstance(
            value,
            (int, float),
        ):
            threshold_values.append(float(value))
    if not threshold_values:
        # Search nested JSON without depending on a specific report layout.
        stack = [l5]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key, child in value.items():
                    if (
                        key == "A1_margin_threshold"
                        and isinstance(child, (int, float))
                    ):
                        threshold_values.append(float(child))
                    else:
                        stack.append(child)
            elif isinstance(value, list):
                stack.extend(value)
    if not any(
        math.isclose(
            value,
            EXPECTED_A1_MARGIN_THRESHOLD,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for value in threshold_values
    ):
        raise RuntimeError(
            "frozen A1 margin threshold missing or changed in L5"
        )

    final_report_dir = (
        root
        / "reports/v5/p2_decoder_d3_full_validation_certification_proof"
    )
    final_artifact_dir = (
        root
        / "artifacts/v5/p2_decoder_d3_full_validation_certification_proof"
    )
    working_dir = (
        root
        / "artifacts/v5/p2_decoder_d3_full_validation_certification_working"
    )
    chunks_dir = working_dir / "chunks"
    state_path = working_dir / "STATE.json"

    if final_report_dir.exists() or final_artifact_dir.exists():
        raise RuntimeError(
            "final D3 output already exists; do not overwrite it"
        )

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from src.decoders import (
        v5_legal_xy_exact_decoder_certified as certified,
    )

    policy = certified.certification_policy()
    if policy.get("policy_id") != EXPECTED_POLICY_ID:
        raise RuntimeError("runtime certified-decoder policy changed")

    arrays = {
        name: np.load(
            archive / f"{name}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        for name in (
            "graph_logits",
            "count_logits",
            "source_logits",
            "transit_logits",
            "victim_logits",
            "path_logits",
            "validation_item_index",
        )
    }
    total = int(arrays["graph_logits"].shape[0])
    if total != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(
            f"validation item count={total}, expected={EXPECTED_VALIDATION_ITEMS}"
        )
    if not np.array_equal(
        arrays["validation_item_index"],
        np.arange(total, dtype=np.int64),
    ):
        raise RuntimeError("validation item ordering changed")

    expected_shapes = {
        "count_logits": (total, 4),
        "source_logits": (total, 16),
        "transit_logits": (total, 16),
        "victim_logits": (total, 16),
        "path_logits": (total, 16),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise RuntimeError(
                f"{name} shape={arrays[name].shape}, expected={shape}"
            )

    run_contract = {
        "schema_version": D3_SCHEMA_VERSION,
        "stage": (
            "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF"
        ),
        "policy_id": EXPECTED_POLICY_ID,
        "total_validation_items": total,
        "chunk_size": chunk_size,
        "A1_margin_threshold": EXPECTED_A1_MARGIN_THRESHOLD,
        "determinism_replay_items": int(
            max(0, args.determinism_replay_items)
        ),
        "bindings": observed_hashes,
        "l5_report_sha256": sha256_file(paths["l5_report"]),
    }
    contract_digest = hashlib.sha256(
        json.dumps(
            run_contract,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    if state_path.is_file():
        state = read_json(state_path)
        if state.get("contract_sha256") != contract_digest:
            raise RuntimeError(
                "existing D3 working state belongs to another contract"
            )
    else:
        working_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "stage": run_contract["stage"],
            "status": "IN_PROGRESS",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "contract": run_contract,
            "contract_sha256": contract_digest,
            "completed_chunks": [],
        }
        atomic_write_json(state_path, state)

    completed_by_name = {
        item["filename"]: item
        for item in state.get("completed_chunks", [])
    }

    ranges = list(chunk_ranges(total, chunk_size))
    for ordinal, (chunk_id, start, stop) in enumerate(ranges, start=1):
        filename = expected_chunk_name(chunk_id, start, stop)
        chunk_path = chunks_dir / filename

        if filename in completed_by_name:
            record = completed_by_name[filename]
            if record.get("sha256") != sha256_file(chunk_path):
                raise RuntimeError(
                    f"completed chunk hash changed: {chunk_path}"
                )
            validate_chunk(
                chunk_path,
                expected_start=start,
                expected_stop=stop,
            )
            print(
                f"D3_chunk={ordinal}/{len(ranges)} "
                f"coverage={stop}/{total} status=RESUMED"
            )
            continue

        if chunk_path.exists():
            raise RuntimeError(
                f"unregistered chunk exists: {chunk_path}"
            )

        chunk_arrays = decode_chunk(
            certified=certified,
            arrays=arrays,
            start=start,
            stop=stop,
            threshold=EXPECTED_A1_MARGIN_THRESHOLD,
        )
        atomic_savez(chunk_path, **chunk_arrays)
        summary = validate_chunk(
            chunk_path,
            expected_start=start,
            expected_stop=stop,
        )
        record = {
            "chunk_id": chunk_id,
            "filename": filename,
            "start": start,
            "stop": stop,
            "rows": stop - start,
            "sha256": sha256_file(chunk_path),
            "summary": summary,
        }
        state["completed_chunks"].append(record)
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(state_path, state)
        completed_by_name[filename] = record

        print(
            f"D3_chunk={ordinal}/{len(ranges)} "
            f"coverage={stop}/{total} status=COMMITTED"
        )

    # Re-verify every chunk after complete coverage.
    state = read_json(state_path)
    records = sorted(
        state["completed_chunks"],
        key=lambda item: int(item["start"]),
    )
    if len(records) != len(ranges):
        raise RuntimeError("D3 chunk count incomplete")

    global_stats = {
        "minimum_primary_gap": math.inf,
        "maximum_primary_gap": 0.0,
        "minimum_lexicographic_gap": math.inf,
        "maximum_lexicographic_gap": 0.0,
        "maximum_integrality_error": 0.0,
        "maximum_bound_violation": 0.0,
        "maximum_linear_violation": 0.0,
        "maximum_objective_difference": 0.0,
        "attack_after_threshold_count": 0,
        "positive_gap_count": 0,
        "above_old_gate_count": 0,
        "high_precision_rejection_total": 0,
    }
    attacker_count_histogram = {str(k): 0 for k in (1, 2, 3, 4)}
    fingerprint_hasher = hashlib.sha256()
    chunk_hash_hasher = hashlib.sha256()
    all_margins = np.empty(total, dtype=np.float64)
    all_primary_gaps = np.empty(total, dtype=np.float64)
    all_lex_gaps = np.empty(total, dtype=np.float64)

    cursor = 0
    for record, (chunk_id, start, stop) in zip(records, ranges):
        if (
            int(record["chunk_id"]) != chunk_id
            or int(record["start"]) != start
            or int(record["stop"]) != stop
        ):
            raise RuntimeError("D3 chunk coverage is not contiguous")
        chunk_path = chunks_dir / record["filename"]
        observed = sha256_file(chunk_path)
        if observed != record["sha256"]:
            raise RuntimeError(f"chunk hash changed: {chunk_path}")
        summary = validate_chunk(
            chunk_path,
            expected_start=start,
            expected_stop=stop,
        )

        global_stats["minimum_primary_gap"] = min(
            global_stats["minimum_primary_gap"],
            summary["minimum_primary_gap"],
        )
        global_stats["maximum_primary_gap"] = max(
            global_stats["maximum_primary_gap"],
            summary["maximum_primary_gap"],
        )
        global_stats["minimum_lexicographic_gap"] = min(
            global_stats["minimum_lexicographic_gap"],
            summary["minimum_lexicographic_gap"],
        )
        global_stats["maximum_lexicographic_gap"] = max(
            global_stats["maximum_lexicographic_gap"],
            summary["maximum_lexicographic_gap"],
        )
        for key in (
            "maximum_integrality_error",
            "maximum_bound_violation",
            "maximum_linear_violation",
            "maximum_objective_difference",
        ):
            global_stats[key] = max(
                global_stats[key],
                summary[key],
            )
        for key in (
            "attack_after_threshold_count",
            "positive_gap_count",
            "above_old_gate_count",
            "high_precision_rejection_total",
        ):
            global_stats[key] += int(summary[key])

        with np.load(chunk_path, allow_pickle=False) as value:
            rows = stop - start
            all_margins[cursor : cursor + rows] = value["margin"]
            all_primary_gaps[cursor : cursor + rows] = (
                value["primary_gap"]
            )
            all_lex_gaps[cursor : cursor + rows] = (
                value["lexicographic_gap"]
            )
            for count in (1, 2, 3, 4):
                attacker_count_histogram[str(count)] += int(
                    np.count_nonzero(
                        np.asarray(value["attacker_count"]) == count
                    )
                )
            for digest in np.asarray(value["fingerprint_digest"]):
                fingerprint_hasher.update(bytes(digest))
                fingerprint_hasher.update(b"\n")
        chunk_hash_hasher.update(record["sha256"].encode("ascii"))
        chunk_hash_hasher.update(b"\n")
        cursor += stop - start

    if cursor != total:
        raise RuntimeError("aggregated D3 row count mismatch")

    # Deterministic replay against committed fingerprints.
    replay_indices = evenly_spaced_indices(
        total,
        int(max(0, args.determinism_replay_items)),
    )
    replay_passed = 0
    for ordinal, index in enumerate(replay_indices.tolist(), start=1):
        chunk_id = index // chunk_size
        start = chunk_id * chunk_size
        stop = min(total, start + chunk_size)
        chunk_path = chunks_dir / expected_chunk_name(
            chunk_id,
            start,
            stop,
        )
        with np.load(chunk_path, allow_pickle=False) as value:
            stored = bytes(
                value["fingerprint_digest"][index - start]
            ).decode("ascii")

        replay = certified.decode_best_attack_hypothesis(
            float(arrays["graph_logits"][index]),
            np.asarray(arrays["count_logits"][index], dtype=np.float64),
            np.asarray(arrays["source_logits"][index], dtype=np.float64),
            np.asarray(arrays["transit_logits"][index], dtype=np.float64),
            np.asarray(arrays["victim_logits"][index], dtype=np.float64),
            np.asarray(arrays["path_logits"][index], dtype=np.float64),
        )
        observed = fingerprint_hypothesis(replay)
        if observed != stored:
            raise RuntimeError(
                f"D3 deterministic replay mismatch at index {index}"
            )
        replay_passed += 1
        if ordinal % 8 == 0 or ordinal == replay_indices.size:
            print(
                f"D3_determinism_replay={ordinal}/"
                f"{replay_indices.size} validation_index={index}"
            )

    if global_stats["maximum_integrality_error"] > float(
        policy["integrality_abs_tolerance"]
    ):
        raise RuntimeError("D3 integrality maximum exceeds D1 policy")
    if global_stats["maximum_bound_violation"] > float(
        policy["variable_bound_abs_tolerance"]
    ):
        raise RuntimeError("D3 bound maximum exceeds D1 policy")
    if global_stats["maximum_linear_violation"] > float(
        policy["linear_constraint_abs_tolerance"]
    ):
        raise RuntimeError("D3 linear maximum exceeds D1 policy")
    if (
        global_stats["maximum_primary_gap"]
        > float(policy["mip_gap_reporting_ceiling"])
        or global_stats["maximum_lexicographic_gap"]
        > float(policy["mip_gap_reporting_ceiling"])
    ):
        raise RuntimeError("D3 reported gap exceeds D1 policy")

    quantiles = {
        "margin": {
            str(q): float(np.quantile(all_margins, q))
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
        "primary_gap": {
            str(q): float(np.quantile(all_primary_gaps, q))
            for q in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
        "lexicographic_gap": {
            str(q): float(np.quantile(all_lex_gaps, q))
            for q in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
    }

    final_artifact_dir.mkdir(parents=True, exist_ok=False)
    final_report_dir.mkdir(parents=True, exist_ok=False)

    certified_outputs_dir = final_artifact_dir / "certified_outputs"
    certified_outputs_dir.mkdir()
    for record in records:
        source = chunks_dir / record["filename"]
        destination = certified_outputs_dir / record["filename"]
        os.link(source, destination)

    proof = {
        "stage": (
            "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF"
        ),
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy_id": EXPECTED_POLICY_ID,
        "coverage": {
            "validation_items_total": total,
            "validation_items_certified": total,
            "primary_MILPs_certified": total,
            "lexicographic_MILPs_certified": total,
            "all_outputs_certified": True,
            "A1_margin_threshold": EXPECTED_A1_MARGIN_THRESHOLD,
            "threshold_selection_performed": False,
        },
        "numerical_summary": global_stats,
        "attacker_count_hypothesis_histogram": attacker_count_histogram,
        "quantiles": quantiles,
        "determinism": {
            "replay_items": int(replay_indices.size),
            "replay_items_passed": replay_passed,
            "passed": replay_passed == int(replay_indices.size),
            "replay_indices_sha256": hashlib.sha256(
                replay_indices.astype(np.int64).tobytes()
            ).hexdigest(),
        },
        "immutable_output_archive": {
            "directory": str(certified_outputs_dir),
            "chunks": len(records),
            "rows": total,
            "chunk_hash_sequence_sha256": (
                chunk_hash_hasher.hexdigest()
            ),
            "semantic_fingerprint_sequence_sha256": (
                fingerprint_hasher.hexdigest()
            ),
        },
        "bindings": {
            label: {
                "path": str(paths[label]),
                "sha256": observed_hashes[label],
            }
            for label in sorted(observed_hashes)
        },
        "l5_binding": {
            "path": str(paths["l5_report"]),
            "sha256": sha256_file(paths["l5_report"]),
            "A1_margin_threshold_verified": True,
        },
        "solver_identity": {
            key: value
            for key, value in certified.solver_identity().__dict__.items()
        },
        "scientific_interpretation": {
            "full_validation_numerical_certificate_proven": True,
            "model_performance_result": False,
            "test_result": False,
            "legacy_decoder_replaced": False,
            "certified_decoder_ready_for_staged_evaluator": True,
        },
        "security_boundary": {
            "validation_archive_read": True,
            "test_path_formed": False,
            "test_directory_checked": False,
            "test_directory_enumerated": False,
            "test_tensors_deserialized": False,
            "model_checkpoint_loaded": False,
            "neural_inference_performed": False,
            "threshold_selection_performed": False,
            "decoder_source_modified": False,
            "evaluation_authorization_created": False,
        },
        "next_stage": (
            "V5_P2_EVALUATOR_D4_STAGED_RAW_A0_A1_IMPLEMENTATION_AND_VALIDATION"
        ),
    }

    artifact_json = (
        final_artifact_dir
        / "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF.json"
    )
    atomic_write_json(artifact_json, proof)
    artifact_sha = sha256_file(artifact_json)

    report_json = (
        final_report_dir
        / "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF.json"
    )
    report_value = dict(proof)
    report_value["artifact"] = {
        "path": str(artifact_json),
        "sha256": artifact_sha,
    }
    atomic_write_json(report_json, report_value)

    report_md = (
        final_report_dir
        / "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF.md"
    )
    atomic_write_text(
        report_md,
        f"""# V5 P2 Decoder D3 Full Validation Certification Proof

## Status

- Status: **COMPLETE**
- Policy: `{EXPECTED_POLICY_ID}`
- Certified validation items: **{total}/{total}**
- Primary MILPs certified: **{total}/{total}**
- Lexicographic MILPs certified: **{total}/{total}**
- Deterministic replay: **{replay_passed}/{replay_indices.size} PASS**

## Numerical maxima

| Certificate quantity | Maximum |
|---|---:|
| Primary reported MIP gap | `{global_stats['maximum_primary_gap']:.17g}` |
| Lexicographic reported MIP gap | `{global_stats['maximum_lexicographic_gap']:.17g}` |
| Integrality error | `{global_stats['maximum_integrality_error']:.17g}` |
| Variable-bound violation | `{global_stats['maximum_bound_violation']:.17g}` |
| Linear-constraint violation | `{global_stats['maximum_linear_violation']:.17g}` |
| Objective difference | `{global_stats['maximum_objective_difference']:.17g}` |

- Positive reported-gap certificates:
  **{global_stats['positive_gap_count']}**
- Certificates above the old 64-epsilon gate:
  **{global_stats['above_old_gate_count']}**
- High-precision candidate rejections:
  **{global_stats['high_precision_rejection_total']}**
- Validation hypotheses classified as attack after the already frozen A1
  threshold: **{global_stats['attack_after_threshold_count']}/{total}**

## Interpretation

D3 proves that every frozen validation-logit item can be decoded under the D1
policy with independent primary and lexicographic numerical certificates.

This is a decoder-correctness and numerical-stability result. It is not a model
performance result and is not a test result. No threshold was selected or
changed.

## Immutable output archive

- Chunks: **{len(records)}**
- Rows: **{total}**
- Chunk-hash sequence SHA-256:
  `{chunk_hash_hasher.hexdigest()}`
- Semantic-fingerprint sequence SHA-256:
  `{fingerprint_hasher.hexdigest()}`

## Safety boundary

- Test path formed: **false**
- Test directory checked/enumerated: **false**
- Test tensors deserialized: **false**
- Model/checkpoint loaded: **false**
- Neural inference performed: **false**
- Threshold selection performed: **false**
- Evaluation authorization created: **false**

## Next stage

**V5 P2 evaluator D4 staged Raw/A0/A1 implementation and validation**

Artifact SHA-256: `{artifact_sha}`
""",
    )

    lock = {
        "stage": (
            "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF_LOCK"
        ),
        "status": "LOCKED",
        "policy_id": EXPECTED_POLICY_ID,
        "artifact_path": str(artifact_json),
        "artifact_sha256": artifact_sha,
        "report_json_sha256": sha256_file(report_json),
        "report_markdown_sha256": sha256_file(report_md),
        "certified_validation_items": total,
        "all_outputs_certified": True,
        "determinism_passed": True,
        "test_access_performed": False,
        "authorization_created": False,
        "next_stage": proof["next_stage"],
    }
    lock_path = (
        final_report_dir
        / "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF_LOCK.json"
    )
    atomic_write_json(lock_path, lock)

    marker = (
        final_report_dir
        / "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF_COMPLETE"
    )
    atomic_write_text(
        marker,
        "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF_COMPLETE\n",
    )

    checksum_paths = [
        artifact_json,
        report_json,
        report_md,
        lock_path,
        marker,
        *sorted(certified_outputs_dir.glob("*.npz")),
    ]
    checksum_path = (
        final_report_dir
        / "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF_SHA256SUMS.txt"
    )
    atomic_write_text(
        checksum_path,
        "".join(
            f"{sha256_file(path)}  {path}\n"
            for path in checksum_paths
        ),
    )

    state["status"] = "COMPLETE"
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    state["final_artifact_sha256"] = artifact_sha
    state["final_report_directory"] = str(final_report_dir)
    atomic_write_json(state_path, state)

    print(
        "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF_COMPLETE"
    )
    print("status=COMPLETE")
    print(f"policy_id={EXPECTED_POLICY_ID}")
    print(f"validation_items_certified={total}/{total}")
    print(f"primary_MILPs_certified={total}/{total}")
    print(f"lexicographic_MILPs_certified={total}/{total}")
    print("all_outputs_certified=true")
    print(
        f"determinism_replay_passed="
        f"{replay_passed}/{replay_indices.size}"
    )
    print(
        f"maximum_primary_reported_mip_gap="
        f"{global_stats['maximum_primary_gap']:.17g}"
    )
    print(
        f"maximum_lexicographic_reported_mip_gap="
        f"{global_stats['maximum_lexicographic_gap']:.17g}"
    )
    print(
        f"maximum_integrality_error="
        f"{global_stats['maximum_integrality_error']:.17g}"
    )
    print(
        f"maximum_bound_violation="
        f"{global_stats['maximum_bound_violation']:.17g}"
    )
    print(
        f"maximum_linear_constraint_violation="
        f"{global_stats['maximum_linear_violation']:.17g}"
    )
    print(
        f"maximum_objective_difference="
        f"{global_stats['maximum_objective_difference']:.17g}"
    )
    print(
        f"positive_reported_gap_certificate_count="
        f"{global_stats['positive_gap_count']}"
    )
    print(
        f"certificate_count_above_old_gate="
        f"{global_stats['above_old_gate_count']}"
    )
    print("model_performance_result=false")
    print("test_result=false")
    print("test_path_formed=false")
    print("test_directory_checked=false")
    print("test_directory_enumerated=false")
    print("test_tensors_deserialized=false")
    print("model_checkpoint_loaded=false")
    print("neural_inference_performed=false")
    print("threshold_selection_performed=false")
    print("authorization_created=false")
    print(
        "next_stage="
        "V5_P2_EVALUATOR_D4_STAGED_RAW_A0_A1_IMPLEMENTATION_AND_VALIDATION"
    )
    print(f"artifact={artifact_json}")
    print(f"artifact_sha256={artifact_sha}")
    print(f"report_markdown={report_md}")
    print(f"report_json={report_json}")
    print(f"sha256s={checksum_path}")


if __name__ == "__main__":
    main()
