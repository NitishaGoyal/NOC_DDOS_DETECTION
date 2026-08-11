#!/usr/bin/env python3
"""Freeze the V5 P2 D2 certified decoder implementation.

This stage verifies the D1 policy binding, executes unit tests before launch
(via installer), and compares the legacy and certified decoders on a fixed,
evenly-spaced validation-only smoke set.

No test path is formed. No model/checkpoint is loaded. No authorization is
created. The legacy decoder is not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED = {
    "f1": "0265de528df00fc84186688ad64621a1ac688a7aae65b2381f93fb8757c372e8",
    "d0": "671c6524810a765a0fc54116afd1fe3fd2e1599780310f3b379b6920c8708d81",
    "d1": "b63efbfe1272fb7f6ab922645182fba86e28d7c8f42a70e0f3ac453b88bec71f",
    "legacy_decoder": "8da32b3ca3915365b8b01049a4ddc6a043583aa7193589ccf4af8116a4b0783c",
    "route_library": "3ae860817ddef7cdbc0360057c51dc86908337b40b1df4c7df8fc2c8b82ea0d7",
}
EXPECTED_VALIDATION_ITEMS = 12_528
EXPECTED_POLICY_ID = "V5P2-D1-6ce8cc7c76b8d2cff367631077d65f20"


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
            f"{label} hash mismatch: observed={observed}, expected={expected}"
        )
    return observed


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evenly_spaced_indices(total: int, count: int) -> np.ndarray:
    count = min(max(1, int(count)), total)
    values = np.rint(
        np.linspace(0, total - 1, num=count, endpoint=True)
    ).astype(np.int64)
    values = np.unique(values)
    if values.size != count:
        selected = set(values.tolist())
        extra = [index for index in range(total) if index not in selected]
        values = np.sort(
            np.concatenate(
                [values, np.asarray(extra[: count - values.size], dtype=np.int64)]
            )
        )
    return values


def semantic_fingerprint(value: Any) -> dict[str, Any]:
    float_fields = (
        "margin",
        "graph_contribution",
        "count_contribution",
        "source_contribution",
        "transit_contribution",
        "victim_contribution",
        "path_contribution",
        "semantic_face_score_difference",
        "semantic_face_numerical_allowance",
        "semantic_face_certification_tolerance",
    )
    fingerprint: dict[str, Any] = {
        "attacker_count": int(value.attacker_count),
        "route_ids": [int(item) for item in value.route_ids],
        "source_mask": int(value.source_mask),
        "transit_mask": int(value.transit_mask),
        "victim_mask": int(value.victim_mask),
        "path_mask": int(value.path_mask),
        "high_precision_primary_variable_score": str(
            value.high_precision_primary_variable_score
        ),
        "high_precision_selected_variable_score": str(
            value.high_precision_selected_variable_score
        ),
        "high_precision_semantic_face_difference": str(
            value.high_precision_semantic_face_difference
        ),
        "high_precision_semantic_face_certified": bool(
            value.high_precision_semantic_face_certified
        ),
        "lexicographic_candidates_rejected_by_high_precision": int(
            value.lexicographic_candidates_rejected_by_high_precision
        ),
    }
    for field in float_fields:
        fingerprint[field] = float(getattr(value, field)).hex()
    return fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--smoke-items", type=int, default=32)
    parser.add_argument("--determinism-repeats", type=int, default=4)
    args = parser.parse_args()

    root = args.repo_root.expanduser().resolve()
    paths = {
        "f1": root / "artifacts/v5/p2_final_f1_post_inference_numerical_failure_disposition/V5_P2_FINAL_F1_POST_INFERENCE_NUMERICAL_FAILURE_DISPOSITION.json",
        "d0": root / "artifacts/v5/p2_decoder_d0_numerical_root_cause_audit/V5_P2_DECODER_D0_NUMERICAL_ROOT_CAUSE_AUDIT.json",
        "d1": root / "artifacts/v5/p2_decoder_d1_prospective_numerical_policy_freeze/V5_P2_D1_PROSPECTIVE_NUMERICAL_CERTIFICATION_POLICY.json",
        "legacy_decoder": root / "src/decoders/v5_legal_xy_exact_decoder.py",
        "certified_decoder": root / "src/decoders/v5_legal_xy_exact_decoder_certified.py",
        "route_library": root / "src/decoders/v5_xy_route_library.py",
        "certified_test": root / "tests/decoders/test_v5_legal_xy_exact_decoder_certified.py",
        "validation_manifest": root / "artifacts/v5/p2_e1_immutable_validation_logit_archive/ARCHIVE_MANIFEST.json",
    }
    archive = root / "artifacts/v5/p2_e1_immutable_validation_logit_archive"

    observed = {
        label: require_hash(paths[label], EXPECTED[label], label)
        for label in ("f1", "d0", "d1", "legacy_decoder", "route_library")
    }
    for label in ("certified_decoder", "certified_test", "validation_manifest"):
        if not paths[label].is_file():
            raise FileNotFoundError(f"{label} missing: {paths[label]}")

    d1 = read_json(paths["d1"])
    if d1.get("policy_id") != EXPECTED_POLICY_ID:
        raise RuntimeError("D1 policy ID changed")

    report_dir = root / "reports/v5/p2_decoder_d2_certified_implementation"
    artifact_dir = root / "artifacts/v5/p2_decoder_d2_certified_implementation"
    if report_dir.exists() or artifact_dir.exists():
        raise RuntimeError("D2 output already exists; do not overwrite it")

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    from src.decoders import v5_legal_xy_exact_decoder as legacy
    from src.decoders import v5_legal_xy_exact_decoder_certified as certified

    runtime_policy = certified.certification_policy()
    expected_values = {
        "policy_id": d1["policy_id"],
        "mip_gap_reporting_ceiling": d1["numerical_context"]["new_reporting_ceiling"],
        "integrality_abs_tolerance": d1["mandatory_acceptance_sequence"][3]["requirements"]["integrality_abs_tolerance"],
        "variable_bound_abs_tolerance": d1["mandatory_acceptance_sequence"][3]["requirements"]["variable_bound_abs_tolerance"],
        "linear_constraint_abs_tolerance": d1["mandatory_acceptance_sequence"][4]["requirements"]["maximum_abs_violation"],
        "objective_abs_tolerance": d1["mandatory_acceptance_sequence"][5]["requirements"]["absolute_tolerance"],
        "objective_rel_tolerance": d1["mandatory_acceptance_sequence"][5]["requirements"]["relative_tolerance"],
        "semantic_face_abs_tolerance": d1["mandatory_acceptance_sequence"][7]["requirements"]["semantic_face_abs_tolerance"],
        "high_precision_decimal_digits": d1["mandatory_acceptance_sequence"][7]["requirements"]["decimal_precision_digits"],
    }
    for key, expected_value in expected_values.items():
        if runtime_policy.get(key) != expected_value:
            raise RuntimeError(
                f"certified decoder policy mismatch for {key}: "
                f"runtime={runtime_policy.get(key)!r}, D1={expected_value!r}"
            )

    arrays = {
        name: np.load(archive / f"{name}.npy", mmap_mode="r", allow_pickle=False)
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
            f"validation count={total}, expected={EXPECTED_VALIDATION_ITEMS}"
        )
    if not np.array_equal(
        arrays["validation_item_index"],
        np.arange(total, dtype=np.int64),
    ):
        raise RuntimeError("validation item order changed")

    indices = evenly_spaced_indices(total, args.smoke_items)
    determinism_count = min(max(0, args.determinism_repeats), indices.size)

    max_primary_gap = 0.0
    max_lex_gap = 0.0
    max_integrality = 0.0
    max_bound = 0.0
    max_linear = 0.0
    max_objective_difference = 0.0
    semantic_fingerprints: list[str] = []

    for ordinal, index in enumerate(indices.tolist(), start=1):
        decode_args = (
            float(arrays["graph_logits"][index]),
            np.asarray(arrays["count_logits"][index], dtype=np.float64),
            np.asarray(arrays["source_logits"][index], dtype=np.float64),
            np.asarray(arrays["transit_logits"][index], dtype=np.float64),
            np.asarray(arrays["victim_logits"][index], dtype=np.float64),
            np.asarray(arrays["path_logits"][index], dtype=np.float64),
        )
        old = legacy.decode_best_attack_hypothesis(*decode_args)
        new = certified.decode_best_attack_hypothesis(*decode_args)

        old_fingerprint = semantic_fingerprint(old)
        new_fingerprint = semantic_fingerprint(new)
        if old_fingerprint != new_fingerprint:
            raise RuntimeError(
                f"legacy/certified semantic mismatch at validation index {index}"
            )
        semantic_fingerprints.append(
            hashlib.sha256(
                json.dumps(
                    new_fingerprint,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )

        if new.certificate_status != "CERTIFIED":
            raise RuntimeError(f"uncertified output at validation index {index}")
        if new.certificate_policy_id != EXPECTED_POLICY_ID:
            raise RuntimeError(f"wrong policy at validation index {index}")

        for certificate in (
            new.primary_certificate,
            new.lexicographic_certificate,
        ):
            if certificate.certificate_status != "CERTIFIED":
                raise RuntimeError(
                    f"MILP certificate failure at validation index {index}"
                )
            max_integrality = max(
                max_integrality,
                certificate.maximum_integrality_error,
            )
            max_bound = max(max_bound, certificate.maximum_bound_violation)
            max_linear = max(
                max_linear,
                certificate.maximum_linear_constraint_violation,
            )
            max_objective_difference = max(
                max_objective_difference,
                certificate.objective_absolute_difference,
            )
        max_primary_gap = max(
            max_primary_gap,
            new.primary_certificate.raw_reported_mip_gap,
        )
        max_lex_gap = max(
            max_lex_gap,
            new.lexicographic_certificate.raw_reported_mip_gap,
        )

        if ordinal <= determinism_count:
            repeated = certified.decode_best_attack_hypothesis(*decode_args)
            if new != repeated:
                raise RuntimeError(
                    f"certified decoder nondeterminism at validation index {index}"
                )

        print(
            f"D2_validation_smoke={ordinal}/{indices.size} "
            f"validation_index={index}"
        )

    artifact_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    snapshot_dir = artifact_dir / "source_snapshot"
    snapshot_dir.mkdir()
    for key in ("certified_decoder", "certified_test"):
        shutil.copy2(paths[key], snapshot_dir / paths[key].name)

    implementation = {
        "stage": "V5_P2_DECODER_D2_CERTIFIED_DECODER_IMPLEMENTATION",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy_id": EXPECTED_POLICY_ID,
        "implementation": {
            "legacy_decoder_preserved": True,
            "certified_decoder_path": str(paths["certified_decoder"]),
            "certified_decoder_sha256": sha256_file(paths["certified_decoder"]),
            "certified_test_path": str(paths["certified_test"]),
            "certified_test_sha256": sha256_file(paths["certified_test"]),
            "runtime_policy": runtime_policy,
            "independent_certificates": [
                "solver termination and finite payload",
                "reported MIP-gap ceiling",
                "integrality",
                "variable bounds",
                "all active linear constraints",
                "objective recomputation",
                "route IDs and unique sources",
                "count and source/transit/victim/path union masks",
                "high-precision semantic face and lexicographic tie break",
            ],
            "fail_hard_taxonomy_implemented": True,
        },
        "validation_smoke": {
            "total_validation_items": total,
            "audited_items": int(indices.size),
            "selection": "deterministic_evenly_spaced",
            "determinism_repeated_items": int(determinism_count),
            "legacy_certified_semantic_equivalence": True,
            "all_outputs_certified": True,
            "maximum_primary_reported_mip_gap": max_primary_gap,
            "maximum_lexicographic_reported_mip_gap": max_lex_gap,
            "maximum_integrality_error": max_integrality,
            "maximum_bound_violation": max_bound,
            "maximum_linear_constraint_violation": max_linear,
            "maximum_objective_difference": max_objective_difference,
            "audited_indices_sha256": hashlib.sha256(
                indices.astype(np.int64).tobytes()
            ).hexdigest(),
            "semantic_fingerprint_set_sha256": hashlib.sha256(
                "\n".join(semantic_fingerprints).encode("utf-8")
            ).hexdigest(),
        },
        "bindings": {
            label: {"path": str(paths[label]), "sha256": observed[label]}
            for label in ("f1", "d0", "d1", "legacy_decoder", "route_library")
        },
        "security_boundary": {
            "validation_archive_read": True,
            "test_path_formed": False,
            "test_directory_checked": False,
            "test_directory_enumerated": False,
            "test_tensors_deserialized": False,
            "model_checkpoint_loaded": False,
            "neural_inference_performed": False,
            "legacy_decoder_modified": False,
            "evaluation_authorization_created": False,
        },
        "next_stage": "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF",
    }

    artifact_json = artifact_dir / "V5_P2_DECODER_D2_CERTIFIED_IMPLEMENTATION.json"
    write_json(artifact_json, implementation)
    artifact_sha = sha256_file(artifact_json)

    report_json = report_dir / "V5_P2_DECODER_D2_CERTIFIED_IMPLEMENTATION.json"
    report_value = dict(implementation)
    report_value["artifact"] = {"path": str(artifact_json), "sha256": artifact_sha}
    write_json(report_json, report_value)

    report_md = report_dir / "V5_P2_DECODER_D2_CERTIFIED_IMPLEMENTATION.md"
    markdown = f"""# V5 P2 Decoder D2 Certified Implementation

## Status

- Status: **COMPLETE**
- D1 policy: `{EXPECTED_POLICY_ID}`
- Legacy decoder modified: **false**
- Next stage: **D3 full validation certification proof**

## Implementation

The versioned certified decoder adds independent solver-payload, gap,
integrality, bound, linear-feasibility, objective, and legal-route
certificates while preserving the frozen structured score and tie-break
semantics.

- Certified source SHA-256: `{sha256_file(paths["certified_decoder"])}`
- Certified test SHA-256: `{sha256_file(paths["certified_test"])}`

## Validation-only smoke equivalence

- Items audited: **{indices.size} / {total}**
- Legacy/certified semantic equivalence: **PASS**
- All primary and lexicographic results certified: **PASS**
- Repeated deterministic decodes: **{determinism_count} PASS**
- Maximum primary reported gap: `{max_primary_gap:.17g}`
- Maximum lexicographic reported gap: `{max_lex_gap:.17g}`
- Maximum integrality error: `{max_integrality:.17g}`
- Maximum bound violation: `{max_bound:.17g}`
- Maximum linear-constraint violation: `{max_linear:.17g}`
- Maximum objective difference: `{max_objective_difference:.17g}`

## Safety boundary

- Test path formed: **false**
- Test directory checked/enumerated: **false**
- Test tensors deserialized: **false**
- Model/checkpoint loaded: **false**
- Neural inference performed: **false**
- Evaluation authorization created: **false**

Artifact SHA-256: `{artifact_sha}`
"""
    report_md.write_text(markdown, encoding="utf-8")

    lock = {
        "stage": "V5_P2_DECODER_D2_CERTIFIED_IMPLEMENTATION_LOCK",
        "status": "LOCKED",
        "policy_id": EXPECTED_POLICY_ID,
        "artifact_sha256": artifact_sha,
        "certified_decoder_sha256": sha256_file(paths["certified_decoder"]),
        "certified_test_sha256": sha256_file(paths["certified_test"]),
        "legacy_decoder_sha256": observed["legacy_decoder"],
        "legacy_decoder_modified": False,
        "test_access_performed": False,
        "authorization_created": False,
        "next_stage": implementation["next_stage"],
    }
    lock_path = report_dir / "V5_P2_DECODER_D2_CERTIFIED_IMPLEMENTATION_LOCK.json"
    write_json(lock_path, lock)

    marker = report_dir / "V5_P2_DECODER_D2_CERTIFIED_IMPLEMENTATION_COMPLETE"
    marker.write_text(
        "V5_P2_DECODER_D2_CERTIFIED_IMPLEMENTATION_COMPLETE\n",
        encoding="utf-8",
    )

    checksum_paths = [artifact_json, report_json, report_md, lock_path, marker]
    checksum_path = report_dir / "V5_P2_DECODER_D2_CERTIFIED_IMPLEMENTATION_SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path}\n" for path in checksum_paths),
        encoding="utf-8",
    )

    print("V5_P2_DECODER_D2_CERTIFIED_IMPLEMENTATION_COMPLETE")
    print("status=COMPLETE")
    print(f"policy_id={EXPECTED_POLICY_ID}")
    print("legacy_decoder_preserved=true")
    print(f"certified_decoder={paths['certified_decoder']}")
    print(f"certified_decoder_sha256={sha256_file(paths['certified_decoder'])}")
    print(f"validation_items_total={total}")
    print(f"validation_smoke_items={indices.size}")
    print("legacy_certified_semantic_equivalence=true")
    print("all_outputs_certified=true")
    print("validation_determinism_passed=true")
    print(f"maximum_primary_reported_mip_gap={max_primary_gap:.17g}")
    print(f"maximum_lexicographic_reported_mip_gap={max_lex_gap:.17g}")
    print(f"maximum_integrality_error={max_integrality:.17g}")
    print(f"maximum_bound_violation={max_bound:.17g}")
    print(f"maximum_linear_constraint_violation={max_linear:.17g}")
    print(f"maximum_objective_difference={max_objective_difference:.17g}")
    print("test_path_formed=false")
    print("test_directory_checked=false")
    print("test_directory_enumerated=false")
    print("test_tensors_deserialized=false")
    print("model_checkpoint_loaded=false")
    print("neural_inference_performed=false")
    print("authorization_created=false")
    print("next_stage=V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF")
    print(f"artifact={artifact_json}")
    print(f"artifact_sha256={artifact_sha}")
    print(f"report_markdown={report_md}")
    print(f"report_json={report_json}")
    print(f"sha256s={checksum_path}")


if __name__ == "__main__":
    main()
