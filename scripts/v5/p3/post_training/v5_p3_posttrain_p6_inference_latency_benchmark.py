#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

DEFAULT_WARMUP = 100
DEFAULT_MEASURED = 1000
BATCH_SIZE = 1


def _type7(values: List[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        raise ValueError("no timing samples")
    if len(vals) == 1:
        return vals[0]
    h = (len(vals) - 1) * q
    lo = int(math.floor(h))
    hi = int(math.ceil(h))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (h - lo) * (vals[hi] - vals[lo])


def summarize_ms(samples_ms: List[float]) -> Dict[str, float]:
    if not samples_ms:
        raise ValueError("no latency samples")
    return {
        "median_latency_ms": statistics.median(samples_ms),
        "mean_latency_ms": statistics.mean(samples_ms),
        "p95_latency_ms": _type7(samples_ms, 0.95),
        "sample_standard_deviation_ms": statistics.stdev(samples_ms) if len(samples_ms) >= 2 else 0.0,
        "minimum_latency_ms": min(samples_ms),
        "maximum_latency_ms": max(samples_ms),
    }


def _load_case_module(path: Path):
    name = "_v5_p3_p6_benchmark_case"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark case module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_benchmark_case"):
        raise ValueError("benchmark case module lacks build_benchmark_case()")
    return mod


def _torch_runtime():
    import torch
    return torch


def validate_case(case: Dict[str, Any]) -> None:
    if not isinstance(case, dict):
        raise ValueError("build_benchmark_case() must return a dict")
    required = (
        "callable",
        "parameter_count",
        "weight_bytes",
        "model_sha256",
        "checkpoint_sha256",
        "input_contract",
    )
    missing = [k for k in required if k not in case]
    if missing:
        raise ValueError(f"benchmark case missing fields: {missing}")
    if not callable(case["callable"]):
        raise ValueError("benchmark case 'callable' must be callable")
    for k in ("parameter_count", "weight_bytes"):
        if isinstance(case[k], bool) or not isinstance(case[k], int) or case[k] < 0:
            raise ValueError(f"{k} must be a non-negative integer")
    for k in ("model_sha256", "checkpoint_sha256"):
        v = str(case[k])
        if len(v) != 64 or any(c not in "0123456789abcdefABCDEF" for c in v):
            raise ValueError(f"{k} must be a 64-character SHA-256 hex string")
    if not isinstance(case["input_contract"], (dict, str)):
        raise ValueError("input_contract must be a dict or string")


def benchmark_callable(
    fn: Callable[[], Any],
    device: str,
    warmup: int,
    measured: int,
    ack_gpu_idle: bool,
) -> Dict[str, Any]:
    if warmup < 0 or measured <= 0:
        raise ValueError("warmup must be >=0 and measured must be >0")

    torch = _torch_runtime()
    dev = device.strip().lower()
    use_cuda = dev.startswith("cuda")

    if use_cuda:
        if not ack_gpu_idle:
            raise ValueError("CUDA benchmark requires explicit --ack-gpu-idle")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    sync = torch.cuda.synchronize if use_cuda else (lambda: None)

    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        if use_cuda:
            sync()

        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        samples_ms: List[float] = []
        for _ in range(measured):
            sync()
            t0 = time.perf_counter_ns()
            fn()
            sync()
            t1 = time.perf_counter_ns()
            samples_ms.append((t1 - t0) / 1_000_000.0)

    out = summarize_ms(samples_ms)
    out.update({
        "batch_size": BATCH_SIZE,
        "warmup_iterations": warmup,
        "measured_iterations": measured,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if use_cuda else None,
        "cuda_synchronized_per_measurement": bool(use_cuda),
    })
    return out


def execute(
    case_module: Path,
    output_json: Path,
    device: str,
    precision: str,
    warmup: int,
    measured: int,
    ack_gpu_idle: bool,
) -> None:
    mod = _load_case_module(case_module)
    case = mod.build_benchmark_case(device=device, precision=precision)
    validate_case(case)

    result = benchmark_callable(
        case["callable"],
        device=device,
        warmup=warmup,
        measured=measured,
        ack_gpu_idle=ack_gpu_idle,
    )
    result.update({
        "precision": precision,
        "parameter_count": case["parameter_count"],
        "weight_bytes": case["weight_bytes"],
        "model_sha256": case["model_sha256"],
        "checkpoint_sha256": case["checkpoint_sha256"],
        "input_contract": case["input_contract"],
        "timed_region": "one complete neural forward only",
        "decoder_in_timed_region": False,
        "thresholding_in_timed_region": False,
        "archive_write_in_timed_region": False,
        "security_response_latency_claimed": False,
        "sealed_test_access": False,
    })
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_selftest_case(path: Path) -> None:
    text = """\
def build_benchmark_case(device: str, precision: str):
    import torch
    x = torch.ones((1, 16), dtype=torch.float32, device="cpu")
    layer = torch.nn.Linear(16, 8, bias=True).eval()
    def forward():
        return layer(x)
    return {
        "callable": forward,
        "parameter_count": sum(p.numel() for p in layer.parameters()),
        "weight_bytes": sum(p.numel() * p.element_size() for p in layer.parameters()),
        "model_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "input_contract": {"shape": [1, 16], "synthetic": True},
    }
"""
    path.write_text(text, encoding="utf-8")


def selftest(tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    case = tmp / "selftest_case.py"
    out = tmp / "selftest_result.json"
    _write_selftest_case(case)

    execute(
        case_module=case,
        output_json=out,
        device="cpu",
        precision="fp32-selftest",
        warmup=5,
        measured=20,
        ack_gpu_idle=False,
    )
    result = json.loads(out.read_text())
    assert result["batch_size"] == 1
    assert result["warmup_iterations"] == 5
    assert result["measured_iterations"] == 20
    assert result["peak_cuda_allocated_bytes"] is None
    assert result["security_response_latency_claimed"] is False
    assert result["decoder_in_timed_region"] is False

    # Negative contract test: CUDA path must require explicit idle acknowledgement
    denied = False
    try:
        benchmark_callable(lambda: None, "cuda", 0, 1, ack_gpu_idle=False)
    except ValueError:
        denied = True
    assert denied

    print("P6_SELFTEST_PASS")
    print("selftest_device=cpu")
    print("selftest_warmup_iterations=5")
    print("selftest_measured_iterations=20")
    print("cuda_idle_ack_guard_exercised=true")
    print("final_model_used=false")
    print("dataset_used=false")
    print("cuda_used=false")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--case-module", type=Path, required=True)
    r.add_argument("--output-json", type=Path, required=True)
    r.add_argument("--device", required=True)
    r.add_argument("--precision", required=True)
    r.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    r.add_argument("--measured", type=int, default=DEFAULT_MEASURED)
    r.add_argument("--ack-gpu-idle", action="store_true")

    s = sub.add_parser("selftest")
    s.add_argument("--tmp-dir", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "selftest":
        selftest(args.tmp_dir)
        return

    execute(
        case_module=args.case_module,
        output_json=args.output_json,
        device=args.device,
        precision=args.precision,
        warmup=args.warmup,
        measured=args.measured,
        ack_gpu_idle=args.ack_gpu_idle,
    )
    print("V5_P3_POSTTRAIN_P6_INFERENCE_LATENCY_BENCHMARK_COMPLETE")
    print("security_response_latency_claimed=false")
    print("sealed_test_access=false")


if __name__ == "__main__":
    main()
