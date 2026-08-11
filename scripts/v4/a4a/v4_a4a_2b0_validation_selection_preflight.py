#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

EXPECTED_PRIMARY_TRAINER_SHA = "c424a4877c39cdbf1df54dba89d152913c7aef3d7a265930f21c9af4ec52f42d"
EXPECTED_BASE_CHECKPOINT_SHA = "f61c1add6c057f7f53dd34fb1f9f4e95b01cefd5053e0e42bc100c76dac7f923"

SEARCH_TERMS = (
    "h32",
    "persistent",
    "false isolation",
    "candidate coverage",
    "candidate_coverage",
    "exact localization",
    "exact_localization",
    "run_index",
    "end_epoch",
    "graph_threshold",
    "hard_normal",
    "hard-normal",
    "a3.11",
    "a3.12",
    "oracle",
)

REPORT_NAME_TERMS = (
    "h32",
    "persistent",
    "policy",
    "threshold",
    "hard_normal",
    "hard-normal",
    "oracle",
    "localization",
    "coverage",
    "operational",
    "latency",
    "a3_11",
    "a3_12",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_checkpoint(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    row: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "top_level_type": type(ckpt).__name__,
    }
    if not isinstance(ckpt, dict):
        return row

    row["top_level_keys"] = sorted(ckpt.keys())
    for key in (
        "epoch",
        "best_epoch_after_epoch",
        "best_validation_loss_after_epoch",
        "improved_loss_monitor",
        "test_loader_constructed",
        "test_evaluated",
        "development_test_accessed",
    ):
        if key in ckpt:
            value = ckpt[key]
            row[key] = value if isinstance(value, (str, int, float, bool)) or value is None else type(value).__name__

    for key in ("validation_losses", "validation_metrics", "validation_extra"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            row[f"{key}_keys"] = sorted(value.keys())
            row[key] = {
                k: v
                for k, v in value.items()
                if isinstance(v, (str, int, float, bool)) or v is None
            }

    state = ckpt.get("model_state_dict")
    if isinstance(state, dict):
        row["model_state_tensor_count"] = len(state)
        row["model_state_keys"] = sorted(state.keys())
        row["model_state_numel"] = int(
            sum(v.numel() for v in state.values() if isinstance(v, torch.Tensor))
        )
    return row


def inspect_small_report(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "suffix": path.suffix.lower(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    try:
        if path.suffix.lower() == ".json":
            value = load_json(path)
            row["type"] = type(value).__name__
            if isinstance(value, dict):
                row["keys"] = sorted(value.keys())
                row["selected_values"] = {
                    key: child
                    for key, child in value.items()
                    if isinstance(child, (str, int, float, bool))
                    and any(
                        term in key.lower()
                        for term in (
                            "status",
                            "decision",
                            "threshold",
                            "persistent",
                            "coverage",
                            "precision",
                            "recall",
                            "exact",
                            "test",
                            "validation",
                        )
                    )
                }
            elif isinstance(value, list):
                row["length"] = len(value)
        elif path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                row["header"] = next(reader, [])
                row["first_row"] = next(reader, None)
        elif path.suffix.lower() == ".npy":
            arr = np.load(path, mmap_mode="r")
            row["shape"] = list(arr.shape)
            row["dtype"] = str(arr.dtype)
        elif path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as z:
                row["keys"] = list(z.files)
                row["arrays"] = {
                    key: {
                        "shape": list(z[key].shape),
                        "dtype": str(z[key].dtype),
                    }
                    for key in z.files
                }
        elif path.suffix.lower() in {".md", ".txt"}:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            row["line_count"] = len(lines)
            row["first_30_lines"] = "\n".join(lines[:30])
    except Exception as exc:
        row["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return row


def source_hits(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

    hits = []
    for number, line in enumerate(lines, start=1):
        lowered = line.lower()
        matched = [term for term in SEARCH_TERMS if term in lowered]
        if matched:
            start = max(1, number - 2)
            end = min(len(lines), number + 2)
            hits.append(
                {
                    "line": number,
                    "matched_terms": matched,
                    "excerpt": "\n".join(
                        f"{idx:5d}: {lines[idx - 1]}"
                        for idx in range(start, end + 1)
                    ),
                }
            )
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--primary-trainer", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--a3-report-root", type=Path, required=True)
    parser.add_argument("--a3-script-root", type=Path, required=True)
    parser.add_argument("--a4a-contract-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    required = [
        args.primary_trainer,
        args.base_checkpoint,
        args.training_dir / "summary.json",
        args.training_dir / "config.json",
        args.training_dir / "training_history.json",
        args.training_dir / "validation_checkpoint_manifest.json",
        args.training_dir / "V4_A4A_2A_PRIMARY_TRAINING_PASS",
        args.a4a_contract_dir / "A4A_1C_FULL_TRAINING_CONTRACT.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("A4a-2B0 FAIL: missing paths", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 2
    if args.output_dir.exists():
        print(f"A4a-2B0 FAIL: output exists: {args.output_dir}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True)

    failures = []
    trainer_sha = sha256_file(args.primary_trainer)
    base_sha = sha256_file(args.base_checkpoint)
    if trainer_sha != EXPECTED_PRIMARY_TRAINER_SHA:
        failures.append(f"primary trainer hash mismatch: {trainer_sha}")
    if base_sha != EXPECTED_BASE_CHECKPOINT_SHA:
        failures.append(f"base checkpoint hash mismatch: {base_sha}")

    summary = load_json(args.training_dir / "summary.json")
    config = load_json(args.training_dir / "config.json")
    history = load_json(args.training_dir / "training_history.json")
    manifest = load_json(args.training_dir / "validation_checkpoint_manifest.json")
    contract = load_json(args.a4a_contract_dir / "A4A_1C_FULL_TRAINING_CONTRACT.json")

    if summary.get("status") != "PASS":
        failures.append("training summary is not PASS")
    if summary.get("test_evaluated") is not False:
        failures.append("training summary does not affirm no test evaluation")
    if summary.get("validation_checkpoint_count") != len(manifest):
        failures.append("validation checkpoint count mismatch")

    checkpoints = []
    for entry in manifest:
        rel = entry.get("filename")
        if not rel:
            failures.append("manifest entry lacks filename")
            continue
        path = args.training_dir / rel
        if not path.is_file():
            failures.append(f"missing checkpoint: {path}")
            continue
        row = summarize_checkpoint(path)
        row["manifest"] = entry
        row["manifest_hash_matches"] = entry.get("sha256") == row["sha256"]
        if not row["manifest_hash_matches"]:
            failures.append(f"manifest hash mismatch: {path}")
        checkpoints.append(row)

    source_candidates = []
    if args.a3_script_root.exists():
        for path in sorted(args.a3_script_root.rglob("*")):
            if (
                path.is_file()
                and path.suffix.lower() in {".py", ".sh", ".md", ".txt"}
                and path.stat().st_size <= 5 * 1024 * 1024
            ):
                hits = source_hits(path)
                if hits:
                    source_candidates.append(
                        {
                            "path": str(path),
                            "sha256": sha256_file(path),
                            "hit_count": len(hits),
                            "hits": hits,
                        }
                    )

    report_candidates = []
    if args.a3_report_root.exists():
        for path in sorted(args.a3_report_root.rglob("*")):
            if (
                path.is_file()
                and path.suffix.lower() in {".json", ".csv", ".npy", ".npz", ".md", ".txt"}
                and path.stat().st_size <= 20 * 1024 * 1024
                and any(term in path.name.lower() for term in REPORT_NAME_TERMS)
            ):
                report_candidates.append(inspect_small_report(path))

    dataset_headers = {}
    immutable = config.get("immutable_config", {})
    data_dir_value = immutable.get("data_dir")
    if data_dir_value:
        data_dir = Path(data_dir_value)
        for name in ("metadata.json", "run_index.npy", "end_epoch.npy", "split_id.npy", "attacker_count.npy"):
            path = data_dir / name
            dataset_headers[name] = (
                inspect_small_report(path)
                if path.exists()
                else {"path": str(path), "exists": False}
            )

    full_val_epochs = [
        row.get("epoch")
        for row in history
        if isinstance(row, dict) and row.get("full_validation_performed") is True
    ]

    report = {
        "status": "FAIL" if failures else "PASS",
        "designation": "V4-A4a-2B0 Validation-Selection Preflight",
        "failures": failures,
        "training_summary": summary,
        "training_history_record_count": len(history),
        "full_validation_epochs": full_val_epochs,
        "validation_checkpoint_count": len(checkpoints),
        "validation_checkpoints": checkpoints,
        "a4a_validation_contract": contract.get("checkpoint_and_validation_contract"),
        "a4a_kill_criteria": contract.get("kill_criteria"),
        "a4a_promotion_criteria": contract.get("promotion_criteria"),
        "a3_source_candidates": source_candidates,
        "a3_report_candidates": report_candidates,
        "dataset_metadata_headers": dataset_headers,
        "provenance": {
            "primary_trainer_sha256": trainer_sha,
            "base_checkpoint_sha256": base_sha,
            "training_summary_sha256": sha256_file(args.training_dir / "summary.json"),
            "training_history_sha256": sha256_file(args.training_dir / "training_history.json"),
            "checkpoint_manifest_sha256": sha256_file(args.training_dir / "validation_checkpoint_manifest.json"),
            "a4a_contract_sha256": sha256_file(args.a4a_contract_dir / "A4A_1C_FULL_TRAINING_CONTRACT.json"),
            "preflight_script_sha256": sha256_file(Path(__file__)),
        },
        "dataset_samples_loaded": False,
        "model_inference_performed": False,
        "checkpoint_selection_performed": False,
        "threshold_search_performed": False,
        "validation_loader_constructed": False,
        "test_loader_constructed": False,
        "development_test_accessed": False,
        "ready_to_build_a4a_2b_selector": not failures,
    }

    report_path = args.output_dir / "validation_selection_preflight.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    text = [
        "V4-A4a-2B0 VALIDATION-SELECTION PREFLIGHT",
        f"status: {report['status']}",
        f"ready_to_build_a4a_2b_selector: {report['ready_to_build_a4a_2b_selector']}",
        f"validation_checkpoint_count: {len(checkpoints)}",
        f"full_validation_epochs: {full_val_epochs}",
        f"a3_source_candidate_count: {len(source_candidates)}",
        f"a3_report_candidate_count: {len(report_candidates)}",
        "dataset_samples_loaded: false",
        "model_inference_performed: false",
        "checkpoint_selection_performed: false",
        "threshold_search_performed: false",
        "development_test_accessed: false",
        "",
        "CHECKPOINTS",
    ]
    for row in checkpoints:
        text.append(
            f"- epoch={row.get('epoch')} path={row['path']} "
            f"sha256={row['sha256']} manifest_hash_matches={row['manifest_hash_matches']}"
        )
    text.extend(["", "A3 SOURCE CANDIDATES"])
    for row in source_candidates:
        text.append(f"- {row['path']} hits={row['hit_count']}")
        for hit in row["hits"][:6]:
            text.append(f"  line={hit['line']} terms={hit['matched_terms']}")
    text.extend(["", "A3 REPORT CANDIDATES"])
    for row in report_candidates:
        text.append(f"- {row['path']} suffix={row['suffix']} bytes={row['bytes']}")

    text_path = args.output_dir / "validation_selection_preflight.txt"
    text_path.write_text("\n".join(text) + "\n", encoding="utf-8")

    lock = {
        "status": (
            "A4A_2B0_VALIDATION_SELECTION_PREFLIGHT_COMPLETE"
            if not failures
            else "A4A_2B0_VALIDATION_SELECTION_PREFLIGHT_FAILED"
        ),
        "preflight_report_sha256": sha256_file(report_path),
        "preflight_text_sha256": sha256_file(text_path),
        "validation_checkpoint_count": len(checkpoints),
        "dataset_samples_loaded": False,
        "model_inference_performed": False,
        "checkpoint_selection_performed": False,
        "threshold_search_performed": False,
        "development_test_accessed": False,
    }
    (args.output_dir / "A4A_2B0_PREFLIGHT_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    marker = (
        "V4_A4A_2B0_VALIDATION_SELECTION_PREFLIGHT_PASS"
        if not failures
        else "V4_A4A_2B0_VALIDATION_SELECTION_PREFLIGHT_FAIL"
    )
    (args.output_dir / marker).write_text(marker + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(marker)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
