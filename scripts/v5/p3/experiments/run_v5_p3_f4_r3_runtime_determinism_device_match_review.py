from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STAGE = "V5_P3_F4_R3_RUNTIME_DETERMINISM_AND_DEVICE_MATCH_REVIEW"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"

RUNTIME_TERMS = (
    "device",
    "cuda",
    "gpu",
    "cpu",
    "torch",
    "pytorch",
    "cudnn",
    "cublas",
    "tf32",
    "float32_matmul_precision",
    "deterministic",
    "benchmark",
    "seed",
    "batch",
    "worker",
    "python",
    "numpy",
    "scipy",
    "sklearn",
    "driver",
)

SOURCE_CONTROL_TERMS = (
    "manual_seed",
    "cuda.manual_seed_all",
    "use_deterministic_algorithms",
    "cudnn.deterministic",
    "cudnn.benchmark",
    "allow_tf32",
    "set_float32_matmul_precision",
    "model.eval",
    "inference_mode",
    "no_grad",
    "shuffle=",
    "num_workers",
    "batch_size",
    "map_location",
    "torch.load",
)

REFERENCE_FILE_SUFFIXES = (
    ".json",
    ".log",
    ".txt",
    ".md",
    ".yaml",
    ".yml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
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
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def flatten_json(value: Any, prefix: str = "$") -> Iterable[tuple[str, Any]]:
    yield prefix, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_json(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_json(child, f"{prefix}[{index}]")


def latest_failed_run(workspace: Path) -> Path:
    candidates = []
    for path in workspace.glob("run_*"):
        if not path.is_dir():
            continue
        required = (
            path / "F4_D1_INVOCATION_CONTRACT.json",
            path / "F4_D1_EXPORTER.log",
            path / "F4_FRESH_VS_IMMUTABLE_NPZ_COMPARISON.json",
        )
        if all(item.is_file() for item in required):
            candidates.append(path)
    require(candidates, f"no failed replay run found under {workspace}")
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return candidates[0]


def extract_runtime_lines(text: str, limit: int = 500) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        low = normalize(line)
        if any(term in low for term in RUNTIME_TERMS):
            rows.append({
                "line": line_number,
                "text": line[:1500],
            })
            if len(rows) >= limit:
                break
    return rows


def extract_runtime_json(document: Any, limit: int = 1000) -> list[dict[str, Any]]:
    rows = []
    for json_path, value in flatten_json(document):
        low = normalize(json_path)
        value_text = normalize(str(value))
        if any(term in low or term in value_text for term in RUNTIME_TERMS):
            if isinstance(value, (str, int, float, bool)) or value is None:
                rows.append({
                    "json_path": json_path,
                    "value": value,
                })
                if len(rows) >= limit:
                    break
    return rows


def scan_reference_directory(root: Path) -> dict[str, Any]:
    inventory = []
    extracted = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in REFERENCE_FILE_SUFFIXES:
            continue
        if path.stat().st_size > 128 * 1024 * 1024:
            continue
        record = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
            "suffix": path.suffix.lower(),
        }
        inventory.append(record)

        try:
            if path.suffix.lower() == ".json":
                document = json.loads(path.read_text(encoding="utf-8"))
                rows = extract_runtime_json(document)
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
                rows = extract_runtime_lines(text)
        except Exception as exc:
            rows = [{"error": repr(exc)}]

        if rows:
            extracted.append({
                **record,
                "runtime_evidence": rows,
            })

    return {
        "inventory": inventory,
        "runtime_evidence": extracted,
    }


def source_control_audit(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    controls = {}
    for term in SOURCE_CONTROL_TERMS:
        matches = []
        term_low = term.lower()
        for line_number, line in enumerate(lines, start=1):
            if term_low in line.lower():
                matches.append({
                    "line": line_number,
                    "text": line.strip()[:1200],
                })
                if len(matches) >= 30:
                    break
        controls[term] = matches

    functions = []
    classes = []
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    "name": node.name,
                    "line": int(node.lineno),
                })
            elif isinstance(node, ast.ClassDef):
                classes.append({
                    "name": node.name,
                    "line": int(node.lineno),
                })
    except SyntaxError:
        pass

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "controls": controls,
        "functions": functions,
        "classes": classes,
    }


def current_runtime_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    try:
        import numpy as np
        snapshot["numpy_version"] = np.__version__
    except Exception as exc:
        snapshot["numpy_error"] = repr(exc)

    try:
        import torch

        snapshot.update({
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cudnn_version": (
                int(torch.backends.cudnn.version())
                if torch.backends.cudnn.is_available()
                else None
            ),
            "cudnn_available": bool(torch.backends.cudnn.is_available()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
        })

        devices = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append({
                    "index": index,
                    "name": properties.name,
                    "major": int(properties.major),
                    "minor": int(properties.minor),
                    "total_memory": int(properties.total_memory),
                    "multi_processor_count": int(
                        properties.multi_processor_count
                    ),
                })
        snapshot["cuda_devices"] = devices
    except Exception as exc:
        snapshot["torch_error"] = repr(exc)

    return snapshot


def invocation_runtime(invocation: dict[str, Any]) -> dict[str, Any]:
    mapped = invocation.get("mapping", {}).get("mapped_arguments", [])
    values = {}
    for row in mapped:
        if row.get("included"):
            values[row.get("kind", "unknown")] = row.get("value")

    environment = invocation.get("environment", {})
    return {
        "command": invocation.get("command", []),
        "mapped_values": values,
        "environment": environment,
        "recovery": invocation.get("recovery"),
    }


def reference_runtime_fields(reference_scan: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(reference_scan["runtime_evidence"], sort_keys=True)
    normalized = normalize(text)

    result = {
        "cpu_mentioned": "cpu" in normalized,
        "cuda_mentioned": "cuda" in normalized,
        "gpu_mentioned": "gpu" in normalized,
        "deterministic_mentioned": "deterministic" in normalized,
        "tf32_mentioned": "tf32" in normalized,
        "batch_mentioned": "batch" in normalized,
        "seed_mentioned": "seed" in normalized,
        "worker_mentioned": "worker" in normalized,
        "torch_mentioned": "torch" in normalized,
        "python_mentioned": "python" in normalized,
    }

    device_candidates = []
    batch_candidates = []
    worker_candidates = []
    seed_candidates = []
    version_candidates = []

    for source in reference_scan["runtime_evidence"]:
        for row in source["runtime_evidence"]:
            value = str(row.get("value", row.get("text", "")))
            low = normalize(
                f"{row.get('json_path', '')} {value}"
            )

            if "device" in low:
                device_candidates.append({
                    "source": source["path"],
                    "evidence": row,
                })
            if "batch" in low:
                batch_candidates.append({
                    "source": source["path"],
                    "evidence": row,
                })
            if "worker" in low:
                worker_candidates.append({
                    "source": source["path"],
                    "evidence": row,
                })
            if "seed" in low:
                seed_candidates.append({
                    "source": source["path"],
                    "evidence": row,
                })
            if any(term in low for term in (
                "torch_version",
                "pytorch_version",
                "python_version",
                "cuda_version",
                "cudnn_version",
                "numpy_version",
            )):
                version_candidates.append({
                    "source": source["path"],
                    "evidence": row,
                })

    result.update({
        "device_candidates": device_candidates,
        "batch_candidates": batch_candidates,
        "worker_candidates": worker_candidates,
        "seed_candidates": seed_candidates,
        "version_candidates": version_candidates,
    })
    return result


def infer_reference_device(fields: dict[str, Any]) -> str | None:
    joined = normalize(json.dumps(fields["device_candidates"]))
    has_cuda = "cuda" in joined or "gpu" in joined
    has_cpu = "cpu" in joined

    if has_cuda and not has_cpu:
        return "cuda"
    if has_cpu and not has_cuda:
        return "cpu"
    return None


def infer_current_device(invocation: dict[str, Any], exporter_log: str) -> str | None:
    mapped = invocation.get("mapped_values", {})
    value = normalize(str(mapped.get("device", "")))
    if "cuda" in value or "gpu" in value:
        return "cuda"
    if "cpu" in value:
        return "cpu"

    low = normalize(exporter_log)
    if "device_cuda" in low or "using_cuda" in low:
        return "cuda"
    if "device_cpu" in low or "using_cpu" in low:
        return "cpu"
    return None


def classify(
    *,
    reference_device: str | None,
    current_device: str | None,
    source_audit: dict[str, Any],
    reference_fields: dict[str, Any],
    current_snapshot: dict[str, Any],
) -> dict[str, Any]:
    source_controls = source_audit["controls"]

    deterministic_controls_present = any(
        source_controls[term]
        for term in (
            "use_deterministic_algorithms",
            "cudnn.deterministic",
            "cudnn.benchmark",
            "allow_tf32",
            "set_float32_matmul_precision",
        )
    )
    seed_controls_present = bool(
        source_controls["manual_seed"]
        or source_controls["cuda.manual_seed_all"]
    )
    eval_mode_present = bool(source_controls["model.eval"])
    inference_context_present = bool(
        source_controls["inference_mode"]
        or source_controls["no_grad"]
    )

    reference_runtime_complete = bool(
        reference_device
        and reference_fields["version_candidates"]
        and reference_fields["batch_candidates"]
    )

    reasons = []

    if reference_device and current_device and reference_device != current_device:
        classification = "DEVICE_MISMATCH_CONFIRMED"
        reasons.append(
            f"reference device={reference_device}, current device={current_device}"
        )
        replay_authorized = True
        next_stage = "V5_P3_F4_R4_REFERENCE_DEVICE_MATCHED_REPLAY"
    elif (
        current_device == "cuda"
        and not deterministic_controls_present
    ):
        classification = "CUDA_BACKEND_POLICY_NOT_FROZEN"
        reasons.append(
            "exporter source does not explicitly freeze deterministic/TF32/backend policy"
        )
        replay_authorized = False
        next_stage = "V5_P3_F4_R4_DETERMINISTIC_BACKEND_PROTOCOL_REVIEW"
    elif not reference_runtime_complete:
        classification = "REFERENCE_RUNTIME_PROVENANCE_INCOMPLETE"
        reasons.append(
            "reference device/version/batch provenance is not complete enough "
            "for a strict runtime-matched replay"
        )
        replay_authorized = False
        next_stage = "V5_P3_F4_R4_NUMERIC_STABILITY_AND_TOLERANCE_REVIEW"
    elif current_device == "cuda":
        classification = "CUDA_NUMERIC_DRIFT_UNDER_RECORDED_RUNTIME"
        reasons.append(
            "labels and sample order match exactly; only small CUDA logit drift remains"
        )
        replay_authorized = False
        next_stage = "V5_P3_F4_R4_NUMERIC_STABILITY_AND_TOLERANCE_REVIEW"
    else:
        classification = "NUMERIC_DRIFT_UNDER_APPARENTLY_MATCHED_RUNTIME"
        reasons.append(
            "labels and sample order match; device mismatch was not established"
        )
        replay_authorized = False
        next_stage = "V5_P3_F4_R4_NUMERIC_STABILITY_AND_TOLERANCE_REVIEW"

    return {
        "classification": classification,
        "reasons": reasons,
        "deterministic_controls_present": deterministic_controls_present,
        "seed_controls_present": seed_controls_present,
        "eval_mode_present": eval_mode_present,
        "inference_context_present": inference_context_present,
        "reference_runtime_complete": reference_runtime_complete,
        "runtime_matched_replay_authorized": replay_authorized,
        "next_stage": next_stage,
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")

    baseline_dir = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/"
        "baseline_reproduction"
    )
    r2_report_path = baseline_dir / (
        "V5_P3_F4_R2_FRESH_VS_IMMUTABLE_MISMATCH_DIAGNOSTIC_REPORT.json"
    )
    r2_lock_path = baseline_dir / (
        "V5_P3_F4_R2_FRESH_VS_IMMUTABLE_MISMATCH_DIAGNOSTIC_LOCK.json"
    )
    r2_runtime_path = baseline_dir / (
        "F4_R2_RUNTIME_AND_REFERENCE_PROVENANCE.json"
    )
    final_contract_path = baseline_dir / "F4_P1R4_FINAL_F4_REPLAY_CONTRACT.json"

    required = [
        r2_report_path,
        r2_lock_path,
        r2_runtime_path,
        final_contract_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"required F4-R2 artifacts missing: {missing}")

    r2_report = json.loads(r2_report_path.read_text(encoding="utf-8"))
    r2_lock = json.loads(r2_lock_path.read_text(encoding="utf-8"))
    final_contract = json.loads(
        final_contract_path.read_text(encoding="utf-8")
    )

    require(r2_report.get("status") == "PASS", "F4-R2 is not PASS")
    require(
        r2_lock.get("report_sha256") == sha256_file(r2_report_path),
        "F4-R2 report/lock mismatch",
    )
    require(
        r2_report.get("classification") == "LOGIT_NUMERIC_DRIFT_ONLY",
        "F4-R3 expected LOGIT_NUMERIC_DRIFT_ONLY",
    )
    require(final_contract.get("F4_authorized") is True, "F4 contract missing")
    require(final_contract.get("F5_authorized") is False, "F5 must remain held")

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = latest_failed_run(
            baseline_dir / "f4_replay_runs"
        )
    require(run_dir.is_dir(), f"run directory missing: {run_dir}")

    invocation_path = run_dir / "F4_D1_INVOCATION_CONTRACT.json"
    exporter_log_path = run_dir / "F4_D1_EXPORTER.log"
    require(invocation_path.is_file(), "invocation contract missing")
    require(exporter_log_path.is_file(), "exporter log missing")

    invocation_document = json.loads(
        invocation_path.read_text(encoding="utf-8")
    )
    invocation = invocation_runtime(invocation_document)
    exporter_log = exporter_log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    exporter_path = Path(final_contract["exporter"]["path"]).resolve()
    reference_npz = Path(
        final_contract["immutable_reference"]["path"]
    ).resolve()
    require(exporter_path.is_file(), "official D1 exporter missing")
    require(reference_npz.is_file(), "immutable reference NPZ missing")
    require(
        sha256_file(exporter_path) == final_contract["exporter"]["sha256"],
        "official D1 exporter hash changed",
    )
    require(
        sha256_file(reference_npz)
        == final_contract["immutable_reference"]["sha256"],
        "immutable reference NPZ hash changed",
    )

    reference_dir = reference_npz.parent
    reference_scan = scan_reference_directory(reference_dir)
    reference_fields = reference_runtime_fields(reference_scan)
    source_audit = source_control_audit(exporter_path)
    runtime_snapshot = current_runtime_snapshot()

    reference_device = infer_reference_device(reference_fields)
    current_device = infer_current_device(invocation, exporter_log)

    decision = classify(
        reference_device=reference_device,
        current_device=current_device,
        source_audit=source_audit,
        reference_fields=reference_fields,
        current_snapshot=runtime_snapshot,
    )

    exporter_log_runtime = extract_runtime_lines(exporter_log)

    source_path = output_dir / "F4_R3_D1_EXPORTER_DETERMINISM_AUDIT.json"
    reference_path = output_dir / "F4_R3_IMMUTABLE_REFERENCE_RUNTIME_PROVENANCE.json"
    current_path = output_dir / "F4_R3_CURRENT_REPLAY_RUNTIME_PROVENANCE.json"
    decision_path = output_dir / "F4_R3_RUNTIME_MISMATCH_DECISION.json"

    atomic_json(source_path, source_audit)
    atomic_json(
        reference_path,
        {
            "reference_directory": str(reference_dir),
            "reference_npz": str(reference_npz),
            "reference_npz_sha256": sha256_file(reference_npz),
            "scan": reference_scan,
            "fields": reference_fields,
            "inferred_reference_device": reference_device,
        },
    )
    atomic_json(
        current_path,
        {
            "run_directory": str(run_dir),
            "invocation": invocation,
            "exporter_log_runtime_lines": exporter_log_runtime,
            "current_runtime_snapshot": runtime_snapshot,
            "inferred_current_device": current_device,
        },
    )
    atomic_json(
        decision_path,
        {
            **decision,
            "reference_device": reference_device,
            "current_device": current_device,
            "F4_complete": False,
            "F4M_authorized": False,
            "F5_authorized": False,
            "sealed_test_access": False,
            "tolerance_relaxation_authorized": False,
            "immutable_reference_replacement_authorized": False,
        },
    )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "scope": (
            "Determine whether the F4 logit-only drift is explained by "
            "device, backend-policy, batch/runtime, or incomplete reference "
            "provenance without rerunning validation."
        ),
        "input_finding": {
            "F4_R2_classification": "LOGIT_NUMERIC_DRIFT_ONLY",
            "nonlogit_arrays_exact": True,
            "sample_order_equal": True,
            "maximum_observed_logit_difference": 1.52587890625e-05,
            "mean_differences_order": "approximately 1e-7",
        },
        "runtime": {
            "reference_device": reference_device,
            "current_device": current_device,
            "current_snapshot": runtime_snapshot,
        },
        "decision": {
            "classification": decision["classification"],
            "reasons": decision["reasons"],
            "runtime_matched_replay_authorized": decision[
                "runtime_matched_replay_authorized"
            ],
            "F4_complete": False,
            "F4M_metric_adapter_certification_authorized": False,
            "F5_permutation_authorized": False,
            "tolerance_relaxation_authorized": False,
            "sealed_test_access_authorized": False,
            "next_stage": decision["next_stage"],
        },
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "training_tensors_loaded": False,
            "validation_dataset_tensors_loaded": False,
            "validation_output_artifact_payloads_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "immutable_reference_modified": False,
        },
        "artifacts": {
            "exporter_determinism_audit": str(source_path),
            "reference_runtime_provenance": str(reference_path),
            "current_runtime_provenance": str(current_path),
            "runtime_mismatch_decision": str(decision_path),
        },
        "provenance": {
            "F4_R2_report_sha256": sha256_file(r2_report_path),
            "F4_R2_lock_sha256": sha256_file(r2_lock_path),
            "F4_R2_runtime_provenance_sha256": sha256_file(
                r2_runtime_path
            ),
            "final_F4_contract_sha256": sha256_file(final_contract_path),
            "invocation_contract_sha256": sha256_file(invocation_path),
            "exporter_log_sha256": sha256_file(exporter_log_path),
            "official_exporter_sha256": sha256_file(exporter_path),
            "installed_script_sha256": sha256_file(installed_script),
        },
    }

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    atomic_json(report_path, report)
    atomic_json(
        lock_path,
        {
            "stage": STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(report_path),
            "exporter_determinism_audit_sha256": sha256_file(source_path),
            "reference_runtime_provenance_sha256": sha256_file(reference_path),
            "current_runtime_provenance_sha256": sha256_file(current_path),
            "runtime_mismatch_decision_sha256": sha256_file(decision_path),
            "classification": decision["classification"],
            "runtime_matched_replay_authorized": decision[
                "runtime_matched_replay_authorized"
            ],
            "F4_complete": False,
            "F4M_authorized": False,
            "F5_authorized": False,
            "validation_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"run_directory={run_dir}")
    print(f"reference_device={reference_device}")
    print(f"current_device={current_device}")
    print(
        "deterministic_controls_present="
        f"{str(decision['deterministic_controls_present']).lower()}"
    )
    print(
        "seed_controls_present="
        f"{str(decision['seed_controls_present']).lower()}"
    )
    print(
        "eval_mode_present="
        f"{str(decision['eval_mode_present']).lower()}"
    )
    print(
        "inference_context_present="
        f"{str(decision['inference_context_present']).lower()}"
    )
    print(
        "reference_runtime_complete="
        f"{str(decision['reference_runtime_complete']).lower()}"
    )
    print(f"runtime_classification={decision['classification']}")
    print(f"runtime_reasons={decision['reasons']}")
    print(
        "runtime_matched_replay_authorized="
        f"{str(decision['runtime_matched_replay_authorized']).lower()}"
    )
    print("tolerance_relaxation_authorized=false")
    print("immutable_reference_replacement_authorized=false")
    print("F4_complete=false")
    print("F4M_metric_adapter_certification_authorized=false")
    print("F5_permutation_authorized=false")
    print("model_loaded=false")
    print("checkpoint_loaded=false")
    print("validation_dataset_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={decision['next_stage']}")
    print(f"exporter_determinism_audit={source_path}")
    print(f"reference_runtime_provenance={reference_path}")
    print(f"current_runtime_provenance={current_path}")
    print(f"runtime_mismatch_decision={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
