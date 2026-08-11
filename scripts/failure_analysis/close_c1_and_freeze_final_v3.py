#!/usr/bin/env python3
"""
C1.6: close the rejected Chrono-C1 experiment and freeze the final V3 model.

This script performs no training and no evaluation. It verifies:
- C1 passed the validation gate;
- C1 failed blind-test corroboration;
- multiseed confirmation is not authorized;
- Chrono-A1 remains the final V3 architecture;
- B1 and C1 are both closed as rejected ablations.

It then writes a final V3 architecture-freeze package containing:
- decision manifest;
- artifact hashes;
- concise scientific summary;
- next-phase handoff to V3.1 representation auditing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED = {
    "a1_source": "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b",
    "a1_checkpoint": "c29cdd2303669a61454ed118ef0be4b8809a635b12c0c42806f8b6992340e0ef",
    "c1_checkpoint": "9c7890a7a2fd1c389564621f4036f8ef1ecdc5cd40aeb82443410e5e32ee0865",
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], cwd: Path) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["category", "path", "size_bytes", "sha256"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--b1-closure", required=True, type=Path)
    parser.add_argument("--c1-spec", required=True, type=Path)
    parser.add_argument("--c1-training-integrity", required=True, type=Path)
    parser.add_argument("--c1-validation-report", required=True, type=Path)
    parser.add_argument("--c1-test-report", required=True, type=Path)
    parser.add_argument("--a1-source", required=True, type=Path)
    parser.add_argument("--c1-source", required=True, type=Path)
    parser.add_argument("--a1-model-dir", required=True, type=Path)
    parser.add_argument("--c1-model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    b1_closure_path = args.b1_closure.resolve()
    c1_spec_path = args.c1_spec.resolve()
    c1_training_path = args.c1_training_integrity.resolve()
    c1_validation_path = args.c1_validation_report.resolve()
    c1_test_path = args.c1_test_report.resolve()
    a1_source = args.a1_source.resolve()
    c1_source = args.c1_source.resolve()
    a1_model_dir = args.a1_model_dir.resolve()
    c1_model_dir = args.c1_model_dir.resolve()
    output_dir = args.output_dir.resolve()

    a1_checkpoint = a1_model_dir / "best_model.pt"
    a1_summary = a1_model_dir / "summary.json"
    a1_splits = a1_model_dir / "splits.npz"
    c1_checkpoint = c1_model_dir / "best_model.pt"
    c1_summary = c1_model_dir / "summary.json"
    c1_splits = c1_model_dir / "splits.npz"

    required_files = {
        "b1_closure": b1_closure_path,
        "c1_spec": c1_spec_path,
        "c1_training_integrity": c1_training_path,
        "c1_validation_report": c1_validation_path,
        "c1_test_report": c1_test_path,
        "a1_source": a1_source,
        "c1_source": c1_source,
        "a1_checkpoint": a1_checkpoint,
        "a1_summary": a1_summary,
        "a1_splits": a1_splits,
        "c1_checkpoint": c1_checkpoint,
        "c1_summary": c1_summary,
        "c1_splits": c1_splits,
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise SystemExit(f"STOP: missing {label}: {path}")

    if not repo_root.is_dir():
        raise SystemExit(f"STOP: repository root missing: {repo_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: final V3 freeze directory already non-empty: {output_dir}"
        )

    b1 = load_json(b1_closure_path)
    c1_spec = load_json(c1_spec_path)
    c1_training = load_json(c1_training_path)
    c1_validation = load_json(c1_validation_path)
    c1_test = load_json(c1_test_path)

    decision_checks = {
        "b1_closed_and_rejected": b1.get("status") == "CLOSED_AND_REJECTED",
        "b1_sampler_not_retained": (
            b1.get("scientific_decision", {})
            .get("scenario_balanced_sampler_retained") is False
        ),
        "c1_spec_frozen": c1_spec.get("status") == "SPECIFICATION_FROZEN",
        "c1_training_verified": (
            c1_training.get("status") == "TRAINING_COMPLETED_AND_VERIFIED"
        ),
        "c1_validation_gate_completed": (
            c1_validation.get("status") == "VALIDATION_GATE_COMPLETED"
        ),
        "c1_validation_advanced": (
            c1_validation.get("gate", {}).get("advance_to_blind_test") is True
        ),
        "c1_test_transfer_completed": (
            c1_test.get("status") == "BLIND_TEST_TRANSFER_COMPLETED"
        ),
        "c1_test_not_corroborated": (
            c1_test.get("gate", {}).get("formal_test_verdict")
            == "TEST_DOES_NOT_CORROBORATE_C1"
        ),
        "multiseed_not_authorized": (
            c1_test.get("gate", {})
            .get("advance_to_multiseed_confirmation") is False
        ),
        "test_thresholds_not_reselected": (
            c1_test.get("protocol", {})
            .get("thresholds_reselected_on_test") is False
        ),
        "test_sweep_not_performed": (
            c1_test.get("protocol", {})
            .get("test_threshold_sweep_performed") is False
        ),
    }
    if not all(decision_checks.values()):
        failed = [key for key, value in decision_checks.items() if not value]
        raise SystemExit(f"STOP: final V3 decision checks failed: {failed}")

    hash_checks = {
        "a1_source_hash_matches": sha256(a1_source) == EXPECTED["a1_source"],
        "a1_checkpoint_hash_matches": (
            sha256(a1_checkpoint) == EXPECTED["a1_checkpoint"]
        ),
        "c1_checkpoint_hash_matches": (
            sha256(c1_checkpoint) == EXPECTED["c1_checkpoint"]
        ),
        "c1_source_matches_validation_manifest": (
            sha256(c1_source)
            == c1_validation["c1_manifest"]["source_sha256"]
        ),
        "a1_source_matches_validation_manifest": (
            sha256(a1_source)
            == c1_validation["a1_manifest"]["source_sha256"]
        ),
    }
    if not all(hash_checks.values()):
        failed = [key for key, value in hash_checks.items() if not value]
        raise SystemExit(f"STOP: final V3 hash checks failed: {failed}")

    validation_deltas = c1_validation["gate"]["deltas"]
    test_deltas = c1_test["gate"]["deltas"]

    final_decision = {
        "retained_model": "Chrono-A1 Conv1D-TemporalGCN",
        "retained_model_source": str(a1_source),
        "retained_model_checkpoint": str(a1_checkpoint),
        "rejected_ablation_b1": "scenario-balanced sampling",
        "rejected_ablation_c1": "mean-plus-max graph readout",
        "c1_multiseed_required": False,
        "further_v3_architecture_search_allowed": False,
        "next_phase": "V3.1 raw-trace and feature-semantics audit",
        "scientific_interpretation": (
            "C1 improved validation graph F1 and normal-run FPR, but the "
            "improvement did not transfer. On blind test, graph F1 decreased "
            "and normal false-positive rate increased substantially. The "
            "remaining failure is therefore not solved by a simple readout "
            "change and is more consistent with representation and scenario "
            "distribution mismatch."
        ),
    }

    key_metrics = {
        "validation": {
            "a1_graph_f1": c1_validation["a1"]["graph"]["f1"],
            "c1_graph_f1": c1_validation["c1"]["graph"]["f1"],
            "graph_f1_delta": validation_deltas["graph_f1"],
            "a1_normal_run_macro_fpr": (
                c1_validation["a1"]["run_summary"]["normal_run_macro_fpr"]
            ),
            "c1_normal_run_macro_fpr": (
                c1_validation["c1"]["run_summary"]["normal_run_macro_fpr"]
            ),
            "normal_run_macro_fpr_delta": (
                validation_deltas["normal_run_macro_fpr"]
            ),
        },
        "test": {
            "a1_graph_f1": c1_test["a1"]["graph"]["f1"],
            "c1_graph_f1": c1_test["c1"]["graph"]["f1"],
            "graph_f1_delta": test_deltas["graph_f1"],
            "a1_graph_fpr": c1_test["a1"]["graph"]["fpr"],
            "c1_graph_fpr": c1_test["c1"]["graph"]["fpr"],
            "graph_fpr_delta": test_deltas["graph_fpr"],
            "a1_normal_run_macro_fpr": (
                c1_test["a1"]["run_summary"]["normal_run_macro_fpr"]
            ),
            "c1_normal_run_macro_fpr": (
                c1_test["c1"]["run_summary"]["normal_run_macro_fpr"]
            ),
            "normal_run_macro_fpr_delta": (
                test_deltas["normal_run_macro_fpr"]
            ),
            "strength20_macro_recall_delta": (
                test_deltas["strength20_macro_recall"]
            ),
            "top1_hit_rate_delta": test_deltas["top1_hit_rate"],
        },
    }

    artifact_paths = [
        ("source", a1_source),
        ("source", c1_source),
        ("model", a1_checkpoint),
        ("model", a1_summary),
        ("model", a1_splits),
        ("model", c1_checkpoint),
        ("model", c1_summary),
        ("model", c1_splits),
        ("decision", b1_closure_path),
        ("decision", c1_spec_path),
        ("decision", c1_training_path),
        ("decision", c1_validation_path),
        ("decision", c1_test_path),
    ]
    inventory = [
        {
            "category": category,
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }
        for category, path in artifact_paths
    ]

    output_dir.mkdir(parents=True, exist_ok=False)
    inventory_path = output_dir / "final_v3_artifact_hashes.csv"
    write_csv(inventory_path, inventory)

    git_head = run(["git", "rev-parse", "HEAD"], repo_root)
    git_branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    git_status = run(["git", "status", "--short"], repo_root)

    manifest = {
        "stage": "C1.6_FINAL_V3_FREEZE",
        "status": "V3_ARCHITECTURE_FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision_checks": decision_checks,
        "hash_checks": hash_checks,
        "final_decision": final_decision,
        "key_metrics": key_metrics,
        "artifact_inventory": {
            "path": str(inventory_path),
            "sha256": sha256(inventory_path),
            "file_count": len(inventory),
        },
        "repository": {
            "root": str(repo_root),
            "git_head": git_head,
            "git_branch": git_branch,
            "git_status_short": git_status,
        },
        "prohibitions_after_freeze": [
            "No B1 multiseed training",
            "No C1 multiseed training",
            "No additional V3 sampler search",
            "No additional V3 readout search",
            "No V3 loss-weight search",
            "No V3 hidden-width or depth search",
            "No V3 attention variant",
            "No victim or path head on V3",
            "No hardware optimization before V3.1/V4 model freeze",
            "No 16x16 scaling before V4 baseline stability",
        ],
        "authorized_next_work": [
            "Audit raw trace availability and feature semantics for V3.1",
            "Define validity, observation, and clipping masks",
            "Build deterministic V3.1 representation with unchanged samples and splits",
            "Compare retained A1 architecture on V3 versus V3.1",
            "Prepare V4 scenario and metadata specification",
        ],
    }

    manifest_path = output_dir / "final_v3_architecture_freeze.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = f"""# Final V3 Architecture Freeze

**Status:** V3 ARCHITECTURE FROZEN

## Retained model

`Chrono-A1 Conv1D-TemporalGCN`

## Rejected ablations

- B1 scenario-balanced sampling
- C1 mean-plus-max graph readout

## Why C1 was rejected

C1 appeared promising on validation:

- graph F1 delta: `{validation_deltas['graph_f1']:+.6f}`
- normal-run macro FPR delta: `{validation_deltas['normal_run_macro_fpr']:+.6f}`

But it failed blind-test corroboration:

- graph F1 delta: `{test_deltas['graph_f1']:+.6f}`
- graph FPR delta: `{test_deltas['graph_fpr']:+.6f}`
- normal-run macro FPR delta: `{test_deltas['normal_run_macro_fpr']:+.6f}`

Its weak-attack recall and node ranking improved slightly, but the graph-level
false-positive cost was unacceptable.

## Final decision

- No C1 multiseed training.
- No further V3 architecture or sampler experiments.
- Chrono-A1 is the final V3 model.
- The next phase is the V3.1 raw-trace and feature-semantics audit.

## Scientific interpretation

The remaining blind-test failure is not solved by scenario-balanced sampling
or by replacing mean pooling with mean-plus-max pooling. The evidence now points
more strongly toward feature semantics, representation ambiguity, and scenario
distribution mismatch.
"""
    summary_path = output_dir / "final_v3_architecture_freeze.md"
    summary_path.write_text(summary, encoding="utf-8")

    print("C1.6 FINAL V3 ARCHITECTURE FREEZE: PASS")
    print("status=V3_ARCHITECTURE_FROZEN")
    print("retained_model=Chrono-A1 Conv1D-TemporalGCN")
    print("b1_status=REJECTED")
    print("c1_status=REJECTED")
    print("c1_multiseed_authorized=False")
    print("further_v3_architecture_search_allowed=False")
    print(
        f"validation_graph_f1_delta={validation_deltas['graph_f1']:+.6f}"
    )
    print(
        "validation_normal_run_macro_fpr_delta="
        f"{validation_deltas['normal_run_macro_fpr']:+.6f}"
    )
    print(f"test_graph_f1_delta={test_deltas['graph_f1']:+.6f}")
    print(f"test_graph_fpr_delta={test_deltas['graph_fpr']:+.6f}")
    print(
        "test_normal_run_macro_fpr_delta="
        f"{test_deltas['normal_run_macro_fpr']:+.6f}"
    )
    print("next_phase=V3.1 raw-trace and feature-semantics audit")
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")
    print(f"inventory={inventory_path}")


if __name__ == "__main__":
    main()
