#!/usr/bin/env python3
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

EXPECTED_X_SHAPE = (233803, 16, 8, 24)
TEXT_SUFFIXES = {
    ".py", ".sh", ".bash", ".md", ".txt", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".csv", ".cpp", ".cc", ".c", ".hpp",
    ".hh", ".h", ".sv", ".svh", ".v", ".vh",
}
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", "build",
    "dist", "models", "checkpoints", "artifacts",
}
SEARCH_TERMS = [
    "feature_cols", "x.npy", "ifd_in_norm", "ifd_out_norm",
    "input_flit_count_norm", "output_flit_count_norm",
    "in_count_norm_", "out_count_norm_", "ifd_in_norm_", "ifd_out_norm_",
    "clip", "clipping", "normalize", "normalization", "divisor", "scale",
    "missing", "invalid", "valid_port", "flit_observed", "gap_observed",
    "raw_ifd", "preclip", "pre_clip", "port_order", "direction",
    "local", "north", "east", "south", "west",
]


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


def classify_feature(name: str) -> tuple[str, str]:
    if name in ("ifd_in_norm", "ifd_out_norm"):
        return "aggregate_ifd", "aggregate"
    if name in ("input_flit_count_norm", "output_flit_count_norm"):
        return "aggregate_flit_count", "aggregate"
    for prefix, family in (
        ("in_count_norm_", "per_port_input_flit_count"),
        ("out_count_norm_", "per_port_output_flit_count"),
        ("ifd_in_norm_", "per_port_input_ifd"),
        ("ifd_out_norm_", "per_port_output_ifd"),
    ):
        if name.startswith(prefix):
            return family, name[len(prefix):]
    return "unknown", "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--search-root", action="append", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--sample-windows", type=int, default=20000)
    p.add_argument("--max-file-bytes", type=int, default=5_000_000)
    args = p.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    roots = [r.expanduser().resolve() for r in args.search_root]

    x_path = data_dir / "x.npy"
    feature_path = data_dir / "feature_cols.npy"
    edge_path = data_dir / "edge_index.npy"
    for path in (x_path, feature_path, edge_path):
        if not path.is_file():
            raise SystemExit(f"STOP: missing {path}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"STOP: output directory is non-empty: {output_dir}")

    existing_roots = [r for r in roots if r.exists()]
    missing_roots = [r for r in roots if not r.exists()]
    if not existing_roots:
        raise SystemExit("STOP: no search root exists")

    x = np.load(x_path, mmap_mode="r")
    features = np.load(feature_path, allow_pickle=True)
    feature_names = [
        v.decode("utf-8") if isinstance(v, bytes) else str(v)
        for v in np.asarray(features).reshape(-1)
    ]
    if tuple(x.shape) != EXPECTED_X_SHAPE:
        raise SystemExit(f"STOP: unexpected x shape {x.shape}")
    if len(feature_names) != 24:
        raise SystemExit(f"STOP: expected 24 features, got {len(feature_names)}")

    output_dir.mkdir(parents=True, exist_ok=False)
    reports = output_dir / "reports"
    manifests = output_dir / "manifests"
    reports.mkdir()
    manifests.mkdir()

    compiled_terms = [
        (term, re.compile(re.escape(term), re.IGNORECASE))
        for term in SEARCH_TERMS
    ]
    feature_patterns = {
        name: re.compile(re.escape(name), re.IGNORECASE)
        for name in feature_names
    }

    line_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    files_scanned = 0
    seen_paths: set[Path] = set()

    for root in existing_roots:
        candidates: list[Path] = []
        if root.is_file():
            candidates = [root]
        else:
            for current, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                current_path = Path(current)
                for filename in filenames:
                    candidates.append(current_path / filename)

        for path in candidates:
            try:
                resolved = path.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                if path.suffix.lower() not in TEXT_SUFFIXES:
                    continue
                if path.stat().st_size > args.max_file_bytes:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            files_scanned += 1
            lines = text.splitlines()
            matches = 0
            matched_features: set[str] = set()
            formula_signals = 0

            for line_no, line in enumerate(lines, start=1):
                term_hits = [
                    term for term, pattern in compiled_terms
                    if pattern.search(line)
                ]
                feature_hits = [
                    name for name, pattern in feature_patterns.items()
                    if pattern.search(line)
                ]
                if not term_hits and not feature_hits:
                    continue

                matches += 1
                matched_features.update(feature_hits)
                formula_signals += int(
                    bool(re.search(
                        r"np\.clip|torch\.clamp|/\s*[A-Za-z0-9_.]+|"
                        r"\bscale\s*=|\bclip\s*=|\bdivisor\s*=|\bwhere\s*\(",
                        line,
                        re.IGNORECASE,
                    ))
                )

                start = max(0, line_no - 3)
                end = min(len(lines), line_no + 2)
                context = "\n".join(
                    f"{'>>' if i + 1 == line_no else '  '}{i + 1}: {lines[i]}"
                    for i in range(start, end)
                )
                line_rows.append({
                    "search_root": str(root),
                    "file_path": str(resolved),
                    "line_number": line_no,
                    "matched_features": "|".join(feature_hits),
                    "matched_terms": "|".join(term_hits),
                    "line_text": line.strip(),
                    "context": context,
                })

            if matches:
                file_rows.append({
                    "search_root": str(root),
                    "file_path": str(resolved),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "match_count": matches,
                    "matched_feature_count": len(matched_features),
                    "matched_features": "|".join(sorted(matched_features)),
                    "formula_signal_count": formula_signals,
                    "rank_score": (
                        10 * len(matched_features)
                        + 3 * formula_signals
                        + matches
                    ),
                })

    file_rows.sort(
        key=lambda row: (-int(row["rank_score"]), str(row["file_path"]))
    )
    line_rows.sort(
        key=lambda row: (str(row["file_path"]), int(row["line_number"]))
    )

    sample_count = min(args.sample_windows, x.shape[0])
    sample_indices = np.linspace(
        0, x.shape[0] - 1, num=sample_count, dtype=np.int64
    )
    sampled = np.asarray(x[sample_indices], dtype=np.float32)

    stats_rows: list[dict[str, Any]] = []
    resolution_rows: list[dict[str, Any]] = []
    for index, name in enumerate(feature_names):
        values = sampled[..., index].reshape(-1).astype(np.float64)
        family, scope = classify_feature(name)
        stats_rows.append({
            "feature_index": index,
            "feature_name": name,
            "minimum": float(np.min(values)),
            "p001": float(np.quantile(values, 0.001)),
            "p01": float(np.quantile(values, 0.01)),
            "p50": float(np.quantile(values, 0.50)),
            "p99": float(np.quantile(values, 0.99)),
            "p999": float(np.quantile(values, 0.999)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "zero_fraction": float(np.mean(values == 0.0)),
            "one_fraction": float(np.mean(values == 1.0)),
            "negative_fraction": float(np.mean(values < 0.0)),
            "above_one_fraction": float(np.mean(values > 1.0)),
            "note": "Descriptive only; not normalization inference.",
        })

        matching = [
            row for row in file_rows
            if name in str(row["matched_features"]).split("|")
        ]
        resolution_rows.append({
            "feature_index": index,
            "feature_name": name,
            "family": family,
            "port_or_scope": scope,
            "candidate_file_count": len(matching),
            "top_candidate_file_1": (
                matching[0]["file_path"] if len(matching) > 0 else ""
            ),
            "top_candidate_file_2": (
                matching[1]["file_path"] if len(matching) > 1 else ""
            ),
            "raw_source_field": "",
            "aggregation_window": "",
            "normalization_equation": "",
            "normalization_divisor_or_scale": "",
            "clipping_rule": "",
            "pre_clipping_value_available": "",
            "zero_semantics": "",
            "missing_value_semantics": "",
            "valid_port_rule": "",
            "direction_mapping": "",
            "authoritative_source_file": "",
            "authoritative_line_range": "",
            "evidence_status": "UNRESOLVED_REQUIRES_MANUAL_REVIEW",
            "review_notes": "",
        })

    write_csv(reports / "candidate_source_files.csv", file_rows)
    write_csv(reports / "candidate_line_matches.csv", line_rows)
    write_csv(reports / "sampled_feature_statistics.csv", stats_rows)
    write_csv(
        reports / "feature_provenance_resolution_template.csv",
        resolution_rows,
    )
    np.save(reports / "sampled_window_indices.npy", sample_indices)

    summary = {
        "stage": "H1B.0",
        "status": "PROVENANCE_INVENTORY_COMPLETED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "model_inference_performed": False,
        "test_accessed": False,
        "normalization_inferred_from_values": False,
        "data_dir": str(data_dir),
        "x_shape": list(x.shape),
        "x_sha256": sha256(x_path),
        "feature_cols_sha256": sha256(feature_path),
        "edge_index_sha256": sha256(edge_path),
        "feature_names": feature_names,
        "requested_search_roots": [str(r) for r in roots],
        "existing_search_roots": [str(r) for r in existing_roots],
        "missing_search_roots": [str(r) for r in missing_roots],
        "files_scanned": files_scanned,
        "candidate_file_count": len(file_rows),
        "line_match_count": len(line_rows),
        "sampled_windows": sample_count,
        "resolution_status": {
            "normalization_equations": "UNRESOLVED",
            "clipping_rules": "UNRESOLVED",
            "zero_semantics": "UNRESOLVED",
            "missing_value_semantics": "UNRESOLVED",
            "valid_port_rules": "UNRESOLVED",
            "direction_mapping": "UNRESOLVED",
            "raw_or_preclipping_ifd": "UNRESOLVED",
        },
        "feature_extraction_rtl_freeze_authorized": False,
        "model_core_rtl_work_authorized": True,
        "next_stage": "H1B.1_AUTHORITATIVE_SOURCE_RESOLUTION",
    }
    summary_path = output_dir / "h1b0_provenance_inventory.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    (output_dir / "README.md").write_text(
        "# H1B.0 V3 Feature Provenance Inventory\n\n"
        "Review candidate_source_files.csv first, then candidate_line_matches.csv, "
        "and complete feature_provenance_resolution_template.csv only from "
        "authoritative code/configuration evidence. Processed-value statistics "
        "must not be used to invent normalization equations.\n",
        encoding="utf-8",
    )

    artifact_rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            artifact_rows.append({
                "relative_path": str(path.relative_to(output_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    hashes_path = manifests / "h1b0_artifact_hashes.csv"
    write_csv(hashes_path, artifact_rows)

    release_path = manifests / "h1b0_release_manifest.json"
    release_path.write_text(json.dumps({
        "stage": "H1B.0",
        "status": "PROVENANCE_INVENTORY_COMPLETED",
        "candidate_file_count": len(file_rows),
        "line_match_count": len(line_rows),
        "features_requiring_manual_resolution": len(feature_names),
        "feature_extraction_rtl_freeze_authorized": False,
        "model_core_rtl_work_authorized": True,
        "next_stage": "H1B.1_AUTHORITATIVE_SOURCE_RESOLUTION",
        "artifact_hash_manifest": str(hashes_path),
        "artifact_hash_manifest_sha256": sha256(hashes_path),
    }, indent=2) + "\n", encoding="utf-8")

    print("H1B.0 V3 FEATURE PROVENANCE INVENTORY: PASS")
    print(f"output_dir={output_dir}")
    print(f"search_roots_requested={len(roots)}")
    print(f"search_roots_existing={len(existing_roots)}")
    print(f"search_roots_missing={len(missing_roots)}")
    print(f"files_scanned={files_scanned}")
    print(f"candidate_file_count={len(file_rows)}")
    print(f"line_match_count={len(line_rows)}")
    print(f"sampled_windows={sample_count}")
    print("training_performed=False")
    print("model_inference_performed=False")
    print("test_accessed=False")
    print("normalization_inferred_from_values=False")
    print("feature_extraction_rtl_freeze_authorized=False")
    print("model_core_rtl_work_authorized=True")
    if file_rows:
        print("top_candidate_sources:")
        for row in file_rows[:10]:
            print(
                f"  score={row['rank_score']} "
                f"features={row['matched_feature_count']} "
                f"path={row['file_path']}"
            )
    print(
        "resolution_template="
        f"{reports / 'feature_provenance_resolution_template.csv'}"
    )
    print(f"inventory_report={summary_path}")
    print(f"artifact_hashes={hashes_path}")
    print(f"release_manifest={release_path}")
    print("next_stage=H1B.1_AUTHORITATIVE_SOURCE_RESOLUTION")


if __name__ == "__main__":
    main()
