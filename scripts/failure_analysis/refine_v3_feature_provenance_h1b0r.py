#!/usr/bin/env python3
"""
H1B.0R — source-focused refinement of the V3 feature provenance inventory.

This stage deliberately excludes generated reports, tables, models, artifacts,
processed datasets, logs, and caches. It ranks only implementation/configuration
files that could have created the frozen 24-feature V3 tensor.

No training, no model inference, no test access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_SUFFIXES = {
    ".py", ".sh", ".bash", ".zsh", ".cpp", ".cc", ".c",
    ".hpp", ".hh", ".h", ".json", ".yaml", ".yml", ".toml",
    ".ini", ".cfg",
}

EXCLUDED_DIR_NAMES = {
    ".git", ".venv", "venv", "__pycache__", ".mypy_cache",
    ".pytest_cache", "node_modules", "build", "dist", "models",
    "checkpoints", "artifacts", "reports", "logs", "tables",
    "figures", "plots", "results", "output", "outputs",
}

EXCLUDED_PATH_FRAGMENTS = (
    "/data/processed/",
    "/data/raw/",
    "/data/interim/",
    "/reports/",
    "/artifacts/",
    "/models/",
    "/logs/",
)

CREATION_SIGNALS = {
    "writes_x_npy": re.compile(
        r"(?:np\.save|numpy\.save)\s*\([^,\n]*[\"']x\.npy[\"']",
        re.IGNORECASE,
    ),
    "writes_feature_cols": re.compile(
        r"(?:np\.save|numpy\.save)\s*\([^,\n]*[\"']feature_cols\.npy[\"']",
        re.IGNORECASE,
    ),
    "defines_feature_cols": re.compile(
        r"\bfeature_cols\s*=",
        re.IGNORECASE,
    ),
    "creates_memmap": re.compile(
        r"\bopen_memmap\b|\bmemmap\b",
        re.IGNORECASE,
    ),
    "stacks_features": re.compile(
        r"\b(?:np|numpy)\.(?:stack|column_stack|concatenate)\s*\(",
        re.IGNORECASE,
    ),
    "normalizes": re.compile(
        r"\b(?:norm|normalize|normalization|divisor|scale)\b",
        re.IGNORECASE,
    ),
    "clips": re.compile(
        r"\b(?:np\.clip|numpy\.clip|torch\.clamp|clip_value|clip_max)\b",
        re.IGNORECASE,
    ),
    "reads_stats": re.compile(
        r"\b(?:stats\.txt|stats\.json|gem5|router|flit|ifd)\b",
        re.IGNORECASE,
    ),
    "windowing": re.compile(
        r"\b(?:window_epochs|stride_epochs|sliding window|window_size)\b",
        re.IGNORECASE,
    ),
    "labels": re.compile(
        r"\b(?:y_graph|y_node|attackers|victim|run_id|split)\b",
        re.IGNORECASE,
    ),
}

FORMULA_PATTERNS = [
    re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*_norm\s*=", re.IGNORECASE),
    re.compile(r"/\s*(?:[0-9]+(?:\.[0-9]+)?|[A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"\b(?:np\.clip|numpy\.clip|torch\.clamp)\s*\(", re.IGNORECASE),
    re.compile(r"\b(?:clip|max|scale|divisor)\s*=", re.IGNORECASE),
    re.compile(r"\b(?:where|isnan|nan_to_num|fillna)\s*\(", re.IGNORECASE),
]

LIKELY_FILENAME_TERMS = (
    "build", "create", "generate", "preprocess", "dataset",
    "temporal", "graph", "feature", "port", "v3",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def context(lines: list[str], index: int, radius: int = 3) -> str:
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    return "\n".join(
        f"{'>>' if i == index else '  '}{i + 1}: {lines[i]}"
        for i in range(start, end)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--search-root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    roots = [p.expanduser().resolve() for p in args.search_root]
    output_dir = args.output_dir.resolve()

    feature_path = data_dir / "feature_cols.npy"
    x_path = data_dir / "x.npy"
    if not feature_path.is_file() or not x_path.is_file():
        raise SystemExit("STOP: frozen V3 feature_cols.npy or x.npy missing")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"STOP: output directory non-empty: {output_dir}")

    existing_roots = [p for p in roots if p.exists()]
    if not existing_roots:
        raise SystemExit("STOP: no supplied search roots exist")

    features = np.load(feature_path, allow_pickle=True)
    feature_names = [
        v.decode("utf-8") if isinstance(v, bytes) else str(v)
        for v in np.asarray(features).reshape(-1)
    ]
    feature_regexes = {
        name: re.compile(re.escape(name), re.IGNORECASE)
        for name in feature_names
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    reports = output_dir / "reports"
    manifests = output_dir / "manifests"
    reports.mkdir()
    manifests.mkdir()

    candidate_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    files_scanned = 0

    for root in existing_roots:
        if root.is_file():
            candidates = [root]
        else:
            candidates = []
            for current, dirnames, filenames in os.walk(root):
                dirnames[:] = [
                    d for d in dirnames if d not in EXCLUDED_DIR_NAMES
                ]
                current_path = Path(current)
                for filename in filenames:
                    candidates.append(current_path / filename)

        for path in candidates:
            try:
                resolved = path.resolve()
                resolved_text = str(resolved)
                if resolved in seen:
                    continue
                seen.add(resolved)
                if path.suffix.lower() not in SOURCE_SUFFIXES:
                    continue
                if any(fragment in resolved_text for fragment in EXCLUDED_PATH_FRAGMENTS):
                    continue
                if path.stat().st_size > args.max_file_bytes:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            files_scanned += 1
            lines = text.splitlines()

            matched_features = [
                name for name, regex in feature_regexes.items()
                if regex.search(text)
            ]
            signal_hits = {
                name: bool(regex.search(text))
                for name, regex in CREATION_SIGNALS.items()
            }
            formula_line_count = sum(
                1
                for line in lines
                if any(regex.search(line) for regex in FORMULA_PATTERNS)
            )
            filename_bonus = sum(
                1 for term in LIKELY_FILENAME_TERMS
                if term in path.name.lower()
            )

            source_creation_score = (
                25 * int(signal_hits["writes_x_npy"])
                + 25 * int(signal_hits["writes_feature_cols"])
                + 18 * int(signal_hits["defines_feature_cols"])
                + 8 * int(signal_hits["creates_memmap"])
                + 8 * int(signal_hits["stacks_features"])
                + 6 * int(signal_hits["normalizes"])
                + 6 * int(signal_hits["clips"])
                + 4 * int(signal_hits["reads_stats"])
                + 4 * int(signal_hits["windowing"])
                + 3 * int(signal_hits["labels"])
                + 3 * len(matched_features)
                + min(formula_line_count, 20)
                + 2 * filename_bonus
            )

            # Require at least meaningful feature-generation evidence.
            if source_creation_score < 10:
                continue

            candidate_rows.append({
                "search_root": str(root),
                "file_path": str(resolved),
                "filename": path.name,
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "matched_feature_count": len(matched_features),
                "matched_features": "|".join(matched_features),
                "formula_line_count": formula_line_count,
                "filename_bonus": filename_bonus,
                **{key: value for key, value in signal_hits.items()},
                "source_creation_score": source_creation_score,
            })

            for line_index, line in enumerate(lines):
                feature_hits = [
                    name for name, regex in feature_regexes.items()
                    if regex.search(line)
                ]
                creation_hits = [
                    name for name, regex in CREATION_SIGNALS.items()
                    if regex.search(line)
                ]
                formula_hits = [
                    regex.pattern for regex in FORMULA_PATTERNS
                    if regex.search(line)
                ]
                if not feature_hits and not creation_hits and not formula_hits:
                    continue
                evidence_rows.append({
                    "file_path": str(resolved),
                    "line_number": line_index + 1,
                    "feature_hits": "|".join(feature_hits),
                    "creation_signal_hits": "|".join(creation_hits),
                    "formula_signal_hits": "|".join(formula_hits),
                    "line_text": line.strip(),
                    "context": context(lines, line_index),
                })

    candidate_rows.sort(
        key=lambda row: (
            -int(row["source_creation_score"]),
            -int(row["matched_feature_count"]),
            str(row["file_path"]),
        )
    )
    evidence_rows.sort(
        key=lambda row: (str(row["file_path"]), int(row["line_number"]))
    )

    write_csv(reports / "source_focused_candidates.csv", candidate_rows)
    write_csv(reports / "source_focused_evidence.csv", evidence_rows)

    top_rows = candidate_rows[:30]
    review_rows = []
    for rank, row in enumerate(top_rows, start=1):
        review_rows.append({
            "rank": rank,
            "file_path": row["file_path"],
            "sha256": row["sha256"],
            "source_creation_score": row["source_creation_score"],
            "matched_feature_count": row["matched_feature_count"],
            "writes_x_npy": row["writes_x_npy"],
            "writes_feature_cols": row["writes_feature_cols"],
            "defines_feature_cols": row["defines_feature_cols"],
            "normalizes": row["normalizes"],
            "clips": row["clips"],
            "reads_stats": row["reads_stats"],
            "windowing": row["windowing"],
            "manual_classification": "",
            "authoritative_builder": "",
            "authoritative_normalization_source": "",
            "authoritative_direction_mapping_source": "",
            "notes": "",
        })
    write_csv(reports / "top30_manual_review.csv", review_rows)

    likely_builders = [
        row for row in candidate_rows
        if (
            row["writes_x_npy"]
            or row["writes_feature_cols"]
            or (
                row["defines_feature_cols"]
                and row["stacks_features"]
            )
        )
    ]

    summary = {
        "stage": "H1B.0R",
        "status": "SOURCE_FOCUSED_REFINEMENT_COMPLETED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "model_inference_performed": False,
        "test_accessed": False,
        "reports_and_generated_tables_excluded": True,
        "search_roots": [str(p) for p in existing_roots],
        "files_scanned": files_scanned,
        "candidate_source_count": len(candidate_rows),
        "likely_builder_count": len(likely_builders),
        "top_candidate_sources": candidate_rows[:10],
        "feature_extraction_rtl_freeze_authorized": False,
        "model_core_rtl_work_authorized": True,
        "next_stage": (
            "H1B.1 inspect the highest-ranked implementation sources and "
            "freeze authoritative normalization and missing-value rules"
        ),
    }
    summary_path = output_dir / "h1b0r_source_refinement.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    (output_dir / "README.md").write_text(
        "# H1B.0R Source-Focused Refinement\n\n"
        "Generated reports, tables, artifacts, models, logs, and processed data "
        "were excluded. Review `reports/top30_manual_review.csv` and the matching "
        "line contexts in `reports/source_focused_evidence.csv`.\n",
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
    hashes_path = manifests / "h1b0r_artifact_hashes.csv"
    write_csv(hashes_path, artifact_rows)

    release_path = manifests / "h1b0r_release_manifest.json"
    release_path.write_text(
        json.dumps({
            "stage": "H1B.0R",
            "status": "SOURCE_FOCUSED_REFINEMENT_COMPLETED",
            "candidate_source_count": len(candidate_rows),
            "likely_builder_count": len(likely_builders),
            "feature_extraction_rtl_freeze_authorized": False,
            "model_core_rtl_work_authorized": True,
            "artifact_hash_manifest": str(hashes_path),
            "artifact_hash_manifest_sha256": sha256(hashes_path),
            "next_stage": "H1B.1_AUTHORITATIVE_SOURCE_RESOLUTION",
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print("H1B.0R SOURCE-FOCUSED PROVENANCE REFINEMENT: PASS")
    print(f"output_dir={output_dir}")
    print(f"files_scanned={files_scanned}")
    print(f"candidate_source_count={len(candidate_rows)}")
    print(f"likely_builder_count={len(likely_builders)}")
    print("reports_and_generated_tables_excluded=True")
    print("training_performed=False")
    print("model_inference_performed=False")
    print("test_accessed=False")
    print("feature_extraction_rtl_freeze_authorized=False")
    print("model_core_rtl_work_authorized=True")
    print("top_source_candidates:")
    for row in candidate_rows[:15]:
        print(
            f"  score={row['source_creation_score']} "
            f"features={row['matched_feature_count']} "
            f"x={int(bool(row['writes_x_npy']))} "
            f"feature_cols={int(bool(row['writes_feature_cols']))} "
            f"path={row['file_path']}"
        )
    print(f"manual_review={reports / 'top30_manual_review.csv'}")
    print(f"evidence={reports / 'source_focused_evidence.csv'}")
    print(f"summary={summary_path}")
    print("next_stage=H1B.1_AUTHORITATIVE_SOURCE_RESOLUTION")


if __name__ == "__main__":
    main()
