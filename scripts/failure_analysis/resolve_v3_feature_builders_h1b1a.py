#!/usr/bin/env python3
"""
H1B.1A — Resolve likely V3 feature-builder sources from H1B.0R.

Reads the source-focused candidate inventory, selects files that actually:
- write x.npy;
- write feature_cols.npy; or
- define feature_cols and stack/concatenate the feature tensor.

It then:
- copies and hashes the selected source files;
- extracts exact line contexts for feature creation, normalization, clipping,
  direction mapping, missing-value handling, and dataset output;
- identifies referenced local scripts/configs that may be upstream builders;
- produces a compact manual-resolution report.

No training, model inference, or test access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REFERENCE_PATTERNS = [
    re.compile(r"""(?:python|python3)\s+["']?([^"' \\\n]+\.py)"""),
    re.compile(r"""(?:source|bash|sh)\s+["']?([^"' \\\n]+\.(?:sh|bash))"""),
    re.compile(r"""from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import"""),
    re.compile(r"""import\s+([A-Za-z_][A-Za-z0-9_.]*)"""),
]

EVIDENCE_TERMS = [
    "feature_cols",
    "x.npy",
    "np.save",
    "numpy.save",
    "open_memmap",
    "memmap",
    "np.stack",
    "np.column_stack",
    "np.concatenate",
    "_norm",
    "normalize",
    "normalization",
    "np.clip",
    "torch.clamp",
    "clip",
    "scale",
    "divisor",
    "ifd",
    "flit",
    "count",
    "missing",
    "invalid",
    "valid_port",
    "flit_observed",
    "gap_observed",
    "nan_to_num",
    "where(",
    "local",
    "north",
    "east",
    "south",
    "west",
    "window_epochs",
    "stride_epochs",
    "paper1_temporal_graphs_ports_v3",
]

FEATURE_FAMILIES = [
    "ifd_in_norm",
    "ifd_out_norm",
    "input_flit_count_norm",
    "output_flit_count_norm",
    "in_count_norm_",
    "out_count_norm_",
    "ifd_in_norm_",
    "ifd_out_norm_",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def line_context(lines: list[str], index: int, radius: int = 4) -> str:
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    return "\n".join(
        f"{'>>' if pos == index else '  '}{pos + 1}: {lines[pos]}"
        for pos in range(start, end)
    )


def resolve_reference(
    token: str,
    source_path: Path,
    known_roots: list[Path],
) -> list[Path]:
    token = token.strip().strip("'\"")
    candidates: list[Path] = []

    if token.endswith(".py") or token.endswith(".sh") or token.endswith(".bash"):
        raw = Path(token).expanduser()
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append((source_path.parent / raw).resolve())
            for root in known_roots:
                candidates.append((root / raw).resolve())
    elif "." in token and "/" not in token:
        module_path = Path(*token.split("."))
        candidates.append((source_path.parent / module_path).with_suffix(".py"))
        for root in known_roots:
            candidates.append((root / module_path).with_suffix(".py"))

    result = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            result.append(candidate)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1b0r-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        default=[],
        help="Optional roots used to resolve referenced local scripts.",
    )
    args = parser.parse_args()

    h1b0r_dir = args.h1b0r_dir.resolve()
    output_dir = args.output_dir.resolve()
    search_roots = [path.expanduser().resolve() for path in args.search_root]

    candidates_path = (
        h1b0r_dir / "reports" / "source_focused_candidates.csv"
    )
    evidence_path = (
        h1b0r_dir / "reports" / "source_focused_evidence.csv"
    )
    summary_path = h1b0r_dir / "h1b0r_source_refinement.json"

    for path in (candidates_path, evidence_path, summary_path):
        if not path.is_file():
            raise SystemExit(f"STOP: missing H1B.0R artifact: {path}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"STOP: output directory is non-empty: {output_dir}")

    with candidates_path.open(newline="", encoding="utf-8") as handle:
        candidate_rows = list(csv.DictReader(handle))

    with evidence_path.open(newline="", encoding="utf-8") as handle:
        evidence_rows = list(csv.DictReader(handle))

    likely_rows = []
    for row in candidate_rows:
        is_likely = (
            truthy(row.get("writes_x_npy"))
            or truthy(row.get("writes_feature_cols"))
            or (
                truthy(row.get("defines_feature_cols"))
                and truthy(row.get("stacks_features"))
            )
        )
        if is_likely:
            likely_rows.append(row)

    if not likely_rows:
        raise SystemExit("STOP: H1B.0R identified no likely builder sources")

    likely_rows.sort(
        key=lambda row: (
            -int(row.get("source_creation_score", 0)),
            str(row.get("file_path", "")),
        )
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    copied_dir = output_dir / "candidate_sources"
    reports_dir = output_dir / "reports"
    manifests_dir = output_dir / "manifests"
    copied_dir.mkdir()
    reports_dir.mkdir()
    manifests_dir.mkdir()

    candidate_summary_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    copied_paths: dict[Path, Path] = {}

    known_roots = list(search_roots)
    for row in likely_rows:
        source_path = Path(row["file_path"]).resolve()
        known_roots.append(source_path.parent)

    for rank, row in enumerate(likely_rows, start=1):
        source_path = Path(row["file_path"]).resolve()
        if not source_path.is_file():
            candidate_summary_rows.append({
                "rank": rank,
                "source_path": str(source_path),
                "exists": False,
                "classification": "MISSING_SOURCE",
            })
            continue

        copy_name = f"{rank:02d}_{source_path.name}"
        copied_path = copied_dir / copy_name
        shutil.copy2(source_path, copied_path)
        copied_paths[source_path] = copied_path

        text = source_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        exact_feature_mentions = sorted({
            family
            for family in FEATURE_FAMILIES
            if family.lower() in text.lower()
        })
        term_counts = {
            term: text.lower().count(term.lower())
            for term in EVIDENCE_TERMS
            if term.lower() in text.lower()
        }

        candidate_summary_rows.append({
            "rank": rank,
            "source_path": str(source_path),
            "copied_path": str(copied_path),
            "exists": True,
            "sha256": sha256(source_path),
            "size_bytes": source_path.stat().st_size,
            "source_creation_score": row.get("source_creation_score", ""),
            "matched_feature_count": row.get("matched_feature_count", ""),
            "writes_x_npy": row.get("writes_x_npy", ""),
            "writes_feature_cols": row.get("writes_feature_cols", ""),
            "defines_feature_cols": row.get("defines_feature_cols", ""),
            "stacks_features": row.get("stacks_features", ""),
            "normalizes": row.get("normalizes", ""),
            "clips": row.get("clips", ""),
            "reads_stats": row.get("reads_stats", ""),
            "windowing": row.get("windowing", ""),
            "feature_family_mentions": "|".join(exact_feature_mentions),
            "evidence_term_counts_json": json.dumps(term_counts, sort_keys=True),
            "manual_role_classification": "",
            "authoritative_for_feature_values": "",
            "authoritative_for_normalization": "",
            "authoritative_for_missing_semantics": "",
            "authoritative_for_direction_mapping": "",
            "upstream_source_required": "",
            "review_notes": "",
        })

        for line_index, line in enumerate(lines):
            matched_terms = [
                term for term in EVIDENCE_TERMS
                if term.lower() in line.lower()
            ]
            if not matched_terms:
                continue
            context_rows.append({
                "candidate_rank": rank,
                "source_path": str(source_path),
                "line_number": line_index + 1,
                "matched_terms": "|".join(matched_terms),
                "line_text": line.strip(),
                "context": line_context(lines, line_index),
            })

        for pattern in REFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                token = match.group(1)
                resolved = resolve_reference(
                    token,
                    source_path,
                    known_roots,
                )
                if resolved:
                    for target in resolved:
                        reference_rows.append({
                            "candidate_rank": rank,
                            "source_path": str(source_path),
                            "reference_token": token,
                            "resolved_path": str(target),
                            "resolved_sha256": sha256(target),
                            "same_as_likely_candidate": (
                                target in copied_paths
                            ),
                        })
                else:
                    reference_rows.append({
                        "candidate_rank": rank,
                        "source_path": str(source_path),
                        "reference_token": token,
                        "resolved_path": "",
                        "resolved_sha256": "",
                        "same_as_likely_candidate": False,
                    })

    # Include H1B.0R evidence rows corresponding to the likely sources.
    likely_paths = {str(Path(row["file_path"]).resolve()) for row in likely_rows}
    inherited_evidence = [
        row for row in evidence_rows
        if str(Path(row["file_path"]).resolve()) in likely_paths
    ]

    write_csv(
        reports_dir / "likely_builder_candidates.csv",
        candidate_summary_rows,
    )
    write_csv(
        reports_dir / "likely_builder_line_contexts.csv",
        context_rows,
    )
    write_csv(
        reports_dir / "referenced_upstream_sources.csv",
        reference_rows,
    )
    write_csv(
        reports_dir / "inherited_h1b0r_evidence.csv",
        inherited_evidence,
    )

    # Human-readable compact review.
    markdown = [
        "# H1B.1A Likely Builder Resolution",
        "",
        f"Likely builder candidates: {len(candidate_summary_rows)}",
        "",
    ]
    for row in candidate_summary_rows:
        markdown.extend([
            f"## Candidate {row['rank']}: `{Path(row['source_path']).name}`",
            "",
            f"- Source: `{row['source_path']}`",
            f"- SHA-256: `{row.get('sha256', '')}`",
            f"- Score: `{row.get('source_creation_score', '')}`",
            f"- Writes x.npy: `{row.get('writes_x_npy', '')}`",
            f"- Writes feature_cols.npy: `{row.get('writes_feature_cols', '')}`",
            f"- Defines feature_cols: `{row.get('defines_feature_cols', '')}`",
            f"- Stacks features: `{row.get('stacks_features', '')}`",
            f"- Normalizes: `{row.get('normalizes', '')}`",
            f"- Clips: `{row.get('clips', '')}`",
            "",
        ])
    markdown.extend([
        "## Resolution rule",
        "",
        "A file may be declared authoritative only when its code directly "
        "supports the raw field, temporal aggregation, normalization, clipping, "
        "zero/missing semantics, direction mapping, and output feature order. "
        "A wrapper that only copies or saves an existing tensor is not the "
        "authoritative feature-value source.",
        "",
    ])
    (output_dir / "H1B1A_REVIEW.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    artifact_rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            artifact_rows.append({
                "relative_path": str(path.relative_to(output_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    artifact_hashes = manifests_dir / "h1b1a_artifact_hashes.csv"
    write_csv(artifact_hashes, artifact_rows)

    release = {
        "stage": "H1B.1A",
        "status": "LIKELY_BUILDERS_EXTRACTED_FOR_REVIEW",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "likely_builder_count": len(candidate_summary_rows),
        "candidate_sources_copied": len(copied_paths),
        "context_row_count": len(context_rows),
        "reference_row_count": len(reference_rows),
        "training_performed": False,
        "model_inference_performed": False,
        "test_accessed": False,
        "feature_extraction_rtl_freeze_authorized": False,
        "model_core_rtl_work_authorized": True,
        "next_stage": (
            "H1B.1B classify candidates as builder, wrapper, augmenter, "
            "or audit consumer and resolve upstream source chain"
        ),
        "artifact_hash_manifest": str(artifact_hashes),
        "artifact_hash_manifest_sha256": sha256(artifact_hashes),
    }
    release_path = manifests_dir / "h1b1a_release_manifest.json"
    release_path.write_text(
        json.dumps(release, indent=2) + "\n",
        encoding="utf-8",
    )

    print("H1B.1A LIKELY V3 FEATURE BUILDERS: PASS")
    print(f"output_dir={output_dir}")
    print(f"likely_builder_count={len(candidate_summary_rows)}")
    print(f"candidate_sources_copied={len(copied_paths)}")
    print(f"context_row_count={len(context_rows)}")
    print(f"reference_row_count={len(reference_rows)}")
    print("training_performed=False")
    print("model_inference_performed=False")
    print("test_accessed=False")
    print("feature_extraction_rtl_freeze_authorized=False")
    print("model_core_rtl_work_authorized=True")
    print("likely_builders:")
    for row in candidate_summary_rows:
        print(
            f"  rank={row['rank']} "
            f"x={int(truthy(row.get('writes_x_npy')))} "
            f"feature_cols={int(truthy(row.get('writes_feature_cols')))} "
            f"defines={int(truthy(row.get('defines_feature_cols')))} "
            f"stacks={int(truthy(row.get('stacks_features')))} "
            f"path={row['source_path']}"
        )
    print(
        "candidate_report="
        f"{reports_dir / 'likely_builder_candidates.csv'}"
    )
    print(
        "context_report="
        f"{reports_dir / 'likely_builder_line_contexts.csv'}"
    )
    print(
        "upstream_references="
        f"{reports_dir / 'referenced_upstream_sources.csv'}"
    )
    print(f"review_markdown={output_dir / 'H1B1A_REVIEW.md'}")
    print(f"release_manifest={release_path}")
    print("next_stage=H1B.1B_CANDIDATE_CLASSIFICATION")


if __name__ == "__main__":
    main()
