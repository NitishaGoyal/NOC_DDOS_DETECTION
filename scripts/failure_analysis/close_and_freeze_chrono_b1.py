#!/usr/bin/env python3
"""
Stage C0: close and freeze the rejected Chrono-B1 experiment.

This script does not train, evaluate, move, delete, or modify existing artifacts.
It verifies the final B1 verdict, inventories the experiment, hashes the relevant
sources/checkpoints/reports, records repository/environment provenance, and
writes a closure package in a new output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_HASHES = {
    "a1_source": "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b",
    "b1_source": "9a0d4ce709e7109388915a291f219cd653f38cb53aae5a1fffde2d01f892aafd",
    "b1_sampler": "742d8c1ea8e1d8e54993cfa8df8447598fd10bb4c325c2760ed5a45ae2b42404",
    "a1_checkpoint": "c29cdd2303669a61454ed118ef0be4b8809a635b12c0c42806f8b6992340e0ef",
}

REQUIRED_REPORTS = {
    "validation_export_manifest": "b1_9_validation_predictions/validation_export_manifest.json",
    "validation_alignment_audit": "b1_10_validation_alignment/validation_alignment_audit.json",
    "validation_threshold_selection": "b1_11_validation_threshold_selection/validation_threshold_selection.json",
    "test_threshold_transfer": "b1_12_test_threshold_transfer/test_threshold_transfer.json",
    "graph_run_scenario_analysis": "b1_13_graph_run_scenario_analysis/graph_run_scenario_analysis.json",
    "final_b1_json": "b1_14_17_combined_final_analysis/final_b1_report.json",
    "final_b1_markdown": "b1_14_17_combined_final_analysis/final_b1_report.md",
}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
        }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def inventory_files(
    roots: list[tuple[str, Path]],
    excluded_roots: set[Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()

    for category, root in roots:
        root = root.resolve()
        if not root.exists():
            continue

        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            path = path.resolve()

            if any(excluded == path or excluded in path.parents for excluded in excluded_roots):
                continue
            if not path.is_file() or path in seen:
                continue

            seen.add(path)
            stat = path.stat()
            rows.append(
                {
                    "category": category,
                    "path": str(path),
                    "size_bytes": int(stat.st_size),
                    "mtime_utc": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "sha256": sha256(path),
                }
            )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--a1-source", required=True, type=Path)
    parser.add_argument("--b1-source", required=True, type=Path)
    parser.add_argument("--b1-sampler", required=True, type=Path)
    parser.add_argument("--a1-model-dir", required=True, type=Path)
    parser.add_argument("--b1-model-dir", required=True, type=Path)
    parser.add_argument("--b1-report-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    a1_source = args.a1_source.resolve()
    b1_source = args.b1_source.resolve()
    b1_sampler = args.b1_sampler.resolve()
    a1_model_dir = args.a1_model_dir.resolve()
    b1_model_dir = args.b1_model_dir.resolve()
    report_root = args.b1_report_root.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: closure output directory already exists and is not empty: {output_dir}"
        )

    required_paths = {
        "repo_root": repo_root,
        "a1_source": a1_source,
        "b1_source": b1_source,
        "b1_sampler": b1_sampler,
        "a1_model_dir": a1_model_dir,
        "b1_model_dir": b1_model_dir,
        "b1_report_root": report_root,
        "a1_checkpoint": a1_model_dir / "best_model.pt",
        "b1_checkpoint": b1_model_dir / "best_model.pt",
    }

    for label, path in required_paths.items():
        if label.endswith("_dir") or label in {
            "repo_root",
            "a1_model_dir",
            "b1_model_dir",
            "b1_report_root",
        }:
            if not path.is_dir():
                raise SystemExit(f"STOP: missing directory {label}: {path}")
        elif not path.is_file():
            raise SystemExit(f"STOP: missing file {label}: {path}")

    required_report_paths = {
        label: report_root / relative
        for label, relative in REQUIRED_REPORTS.items()
    }
    for label, path in required_report_paths.items():
        if not path.is_file():
            raise SystemExit(f"STOP: missing required B1 report {label}: {path}")

    final_report = load_json(required_report_paths["final_b1_json"])
    verdict = str(final_report.get("formal_verdict", "")).strip().lower()
    retained_architecture = str(
        final_report.get("final_retained_architecture", "")
    ).strip()
    sampler_retained = final_report.get("scenario_balanced_sampler_retained")

    verdict_checks = {
        "formal_verdict_is_reject": verdict == "reject",
        "retained_architecture_is_chrono_a1": (
            retained_architecture == "Chrono-A1 Conv1D-TemporalGCN"
        ),
        "scenario_balanced_sampler_not_retained": sampler_retained is False,
    }

    if not all(verdict_checks.values()):
        raise SystemExit(
            "STOP: final B1 verdict does not match the expected rejected state: "
            + json.dumps(verdict_checks)
        )

    source_hashes = {
        "a1_source": sha256(a1_source),
        "b1_source": sha256(b1_source),
        "b1_sampler": sha256(b1_sampler),
        "a1_checkpoint": sha256(a1_model_dir / "best_model.pt"),
        "b1_checkpoint": sha256(b1_model_dir / "best_model.pt"),
    }

    frozen_hash_checks = {
        key: source_hashes[key] == expected
        for key, expected in EXPECTED_SOURCE_HASHES.items()
    }

    if not all(frozen_hash_checks.values()):
        raise SystemExit(
            "STOP: one or more historically frozen A1/B1 hashes changed: "
            + json.dumps(frozen_hash_checks)
        )

    output_dir.mkdir(parents=True, exist_ok=False)

    inventory = inventory_files(
        roots=[
            ("a1_source", a1_source),
            ("b1_source", b1_source),
            ("b1_sampler", b1_sampler),
            ("a1_model", a1_model_dir),
            ("b1_model", b1_model_dir),
            ("b1_reports", report_root),
        ],
        excluded_roots={output_dir},
    )

    inventory_csv = output_dir / "b1_artifact_inventory_sha256.csv"
    write_csv(inventory_csv, inventory)

    git_head = run_command(["git", "rev-parse", "HEAD"], repo_root)
    git_branch = run_command(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root
    )
    git_status = run_command(["git", "status", "--short"], repo_root)
    python_version = sys.version.replace("\n", " ")

    torch_info: dict[str, Any]
    try:
        import torch

        torch_info = {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        }
    except Exception as exc:
        torch_info = {"error": repr(exc)}

    closure = {
        "stage": "C0_B1_closure",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CLOSED_AND_REJECTED",
        "scientific_decision": {
            "b1_verdict": verdict,
            "retained_architecture": retained_architecture,
            "scenario_balanced_sampler_retained": sampler_retained,
            "next_experiment": final_report.get(
                "next_experiment_recommendation"
            ),
        },
        "verdict_checks": verdict_checks,
        "historically_frozen_hash_checks": frozen_hash_checks,
        "key_hashes": source_hashes,
        "required_reports": {
            label: {
                "path": str(path),
                "sha256": sha256(path),
            }
            for label, path in required_report_paths.items()
        },
        "artifact_inventory": {
            "file_count": len(inventory),
            "total_bytes": int(sum(row["size_bytes"] for row in inventory)),
            "csv": str(inventory_csv),
            "csv_sha256": sha256(inventory_csv),
        },
        "repository": {
            "root": str(repo_root),
            "git_head": git_head,
            "git_branch": git_branch,
            "git_status_short": git_status,
        },
        "environment": {
            "platform": platform.platform(),
            "python": python_version,
            "executable": sys.executable,
            "working_directory": os.getcwd(),
            "torch": torch_info,
        },
        "immutability_note": (
            "This closure package hashes but does not copy or modify the original "
            "model, source, prediction, or report artifacts."
        ),
    }

    manifest_path = output_dir / "b1_closure_manifest.json"
    manifest_path.write_text(
        json.dumps(closure, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# Chrono-B1 Closure",
        "",
        "**Status:** CLOSED AND REJECTED",
        "",
        f"- Formal verdict: `{verdict}`",
        f"- Retained architecture: `{retained_architecture}`",
        f"- Scenario-balanced sampler retained: `{sampler_retained}`",
        f"- Frozen artifact count: `{len(inventory)}`",
        f"- Inventory bytes: `{sum(row['size_bytes'] for row in inventory)}`",
        "",
        "## Scientific consequence",
        "",
        "Chrono-B1 is closed as a rejected sampling-only ablation. "
        "Chrono-A1 remains the frozen control. The B1 sampler must not be reused "
        "for C1.",
        "",
        "## Next experiment",
        "",
        str(final_report.get("next_experiment_recommendation", "")),
        "",
        "## Closure files",
        "",
        "- `b1_closure_manifest.json`",
        "- `b1_artifact_inventory_sha256.csv`",
        "- `b1_closure_summary.md`",
        "",
    ]
    summary_path = output_dir / "b1_closure_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    closure["closure_outputs"] = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
    }
    manifest_path.write_text(
        json.dumps(closure, indent=2) + "\n",
        encoding="utf-8",
    )

    print("C0 B1 CLOSURE: PASS")
    print(f"formal_verdict={verdict}")
    print(f"retained_architecture={retained_architecture}")
    print(f"scenario_balanced_sampler_retained={sampler_retained}")
    print(f"historical_hashes_match={all(frozen_hash_checks.values())}")
    print(f"required_reports_present={len(required_report_paths)}")
    print(f"artifact_count={len(inventory)}")
    print(f"artifact_bytes={sum(row['size_bytes'] for row in inventory)}")
    print(f"git_head={git_head['stdout']}")
    print(f"git_branch={git_branch['stdout']}")
    print(f"git_status_entries={len(git_status['stdout'].splitlines()) if git_status['stdout'] else 0}")
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")
    print(f"inventory={inventory_csv}")


if __name__ == "__main__":
    main()
