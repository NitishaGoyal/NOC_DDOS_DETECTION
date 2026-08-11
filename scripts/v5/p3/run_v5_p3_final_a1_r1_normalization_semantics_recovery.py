from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STAGE = (
    "V5_P3_FINAL_A1_R1_TRAIN_ONLY_NORMALIZATION_"
    "SEMANTICS_RECOVERY_AND_NATIVE_LOADER_CERTIFICATION"
)
CAMPAIGN_LABEL = (
    "V5-P3-1500-D70 Final A+B Dataset — "
    "Train-Only Normalization-Semantics Recovery"
)
A0_STAGE = (
    "V5_P3_FINAL_A0_DYNAMIC70_INTEGRITY_AND_"
    "GUARDED_LOADER_PREFLIGHT"
)

FEATURE_HINTS = (
    "x_log1p",
    "features_log1p",
    "dynamic70_log1p",
    "x_dynamic70_log1p",
    "x",
    "features",
    "dynamic70",
)

ROOT_NAMES = {
    "root", "dataset_root", "data_root", "dataset_dir", "root_dir", "path"
}
SPLIT_NAMES = {"split", "partition", "subset"}
NORM_NAMES = {
    "normalization_path", "normalization_file", "norm_path", "norm_file"
}
CACHE_NAMES = {
    "cache_size", "lru_size", "lru_capacity", "max_cache_size", "cache_runs"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dataset-link", required=True)
    parser.add_argument("--physical-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def import_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def walk(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from walk(child, f"{path}[{index}]")


def load_torch(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def choose_feature_tensor(payload: Any) -> tuple[str, Any]:
    import torch

    candidates = []
    for path, value in walk(payload):
        if not isinstance(value, torch.Tensor):
            continue
        shape = tuple(int(item) for item in value.shape)
        if value.ndim < 2 or shape.count(70) != 1:
            continue

        lowered = path.lower()
        score = 0
        for rank, hint in enumerate(FEATURE_HINTS):
            if lowered.endswith("." + hint):
                score = max(score, 1000 - rank)
            elif hint in lowered:
                score = max(score, 500 - rank)
        if 16 in shape:
            score += 100
        if value.ndim == 3:
            score += 50
        if value.is_floating_point():
            score += 20
        candidates.append((score, path, value))

    require(candidates, "no unambiguous D70 feature tensor found")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    top_score = candidates[0][0]
    top = [item for item in candidates if item[0] == top_score]
    require(
        len(top) == 1,
        "ambiguous top-ranked D70 tensors: "
        + ", ".join(path for _, path, _ in top),
    )
    _, path, value = top[0]
    return path, value


def to_feature_matrix(tensor: Any) -> np.ndarray:
    import torch

    value = tensor.detach().cpu()
    shape = tuple(int(item) for item in value.shape)
    axes = [index for index, size in enumerate(shape) if size == 70]
    require(len(axes) == 1, f"expected one D70 axis, got {shape}")
    value = torch.movedim(value, axes[0], -1)
    return value.reshape(-1, 70).to(torch.float64).numpy()


@dataclass
class Moments:
    count: int
    total: np.ndarray
    total_sq: np.ndarray

    @classmethod
    def empty(cls) -> "Moments":
        return cls(
            count=0,
            total=np.zeros(70, dtype=np.float64),
            total_sq=np.zeros(70, dtype=np.float64),
        )

    def update(self, matrix: np.ndarray) -> None:
        require(
            matrix.ndim == 2 and matrix.shape[1] == 70,
            f"bad feature matrix {matrix.shape}",
        )
        require(np.isfinite(matrix).all(), "non-finite train feature value")
        self.count += int(matrix.shape[0])
        self.total += matrix.sum(axis=0, dtype=np.float64)
        self.total_sq += np.square(
            matrix, dtype=np.float64
        ).sum(axis=0, dtype=np.float64)

    def merge(self, other: "Moments") -> "Moments":
        return Moments(
            count=self.count + other.count,
            total=self.total + other.total,
            total_sq=self.total_sq + other.total_sq,
        )

    def finalize(self) -> dict[str, Any]:
        require(self.count > 0, "empty moments")
        mean = self.total / float(self.count)
        variance = self.total_sq / float(self.count) - np.square(mean)
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance)
        return {
            "count": int(self.count),
            "mean": mean,
            "variance": variance,
            "std": std,
        }


def length70_vectors(payload: Any) -> list[tuple[str, np.ndarray]]:
    import torch

    result = []
    for path, value in walk(payload):
        array = None
        if isinstance(value, torch.Tensor) and int(value.numel()) == 70:
            array = (
                value.detach()
                .cpu()
                .reshape(-1)
                .to(torch.float64)
                .numpy()
            )
        elif isinstance(value, np.ndarray) and value.size == 70:
            array = value.reshape(-1).astype(np.float64, copy=False)
        elif (
            isinstance(value, list)
            and len(value) == 70
            and all(isinstance(item, (int, float)) for item in value)
        ):
            array = np.asarray(value, dtype=np.float64)
        if array is not None:
            result.append((path, array))
    return result


def scalar_candidates(payload: Any) -> list[tuple[str, float]]:
    result = []
    for path, value in walk(payload):
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            numeric = float(value)
            if math.isfinite(numeric) and 0.0 < numeric <= 1.0:
                result.append((path, numeric))
    return result


def compare(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    difference = np.abs(actual - expected)
    scale = np.maximum(np.abs(expected), 1.0)
    scaled = difference / scale
    return {
        "max_abs": float(np.max(difference)),
        "mean_abs": float(np.mean(difference)),
        "max_scaled_relative": float(np.max(scaled)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "allclose_rtol5e-6_atol5e-7": bool(
            np.allclose(actual, expected, rtol=5e-6, atol=5e-7)
        ),
    }


def build_representations(
    mean: np.ndarray,
    variance: np.ndarray,
    std: np.ndarray,
    eps_values: list[float],
) -> dict[str, np.ndarray]:
    representations = {
        "mean": mean,
        "variance_population": variance,
        "std_population": std,
        "std_zero_to_one": np.where(std == 0.0, 1.0, std),
        "inverse_std_zero_to_one": np.where(std == 0.0, 1.0, 1.0 / std),
    }

    for eps in sorted(set(eps_values)):
        tag = format(eps, ".17g")
        representations[
            f"inverse_max_std_eps[{tag}]"
        ] = 1.0 / np.maximum(std, eps)
        representations[
            f"inverse_std_plus_eps[{tag}]"
        ] = 1.0 / (std + eps)
        representations[
            f"rsqrt_variance_plus_eps[{tag}]"
        ] = 1.0 / np.sqrt(variance + eps)
        representations[
            f"rsqrt_max_variance_eps[{tag}]"
        ] = 1.0 / np.sqrt(np.maximum(variance, eps))

    return representations


def audit_normalization(
    guarded: Any,
    normalization_path: Path,
    moments_npz_path: Path,
) -> dict[str, Any]:
    tranche_moments = {"A": Moments.empty(), "B": Moments.empty()}
    feature_paths = {}
    run_counts = {"A": 0, "B": 0}

    train_runs = guarded.list_runs("train")
    require(len(train_runs) == 2400, "train run count mismatch")

    for index, run_path in enumerate(train_runs, start=1):
        tranche = run_path.name[2]
        require(tranche in tranche_moments, f"bad tranche {run_path.name}")
        payload = guarded.load_run("train", run_path.name)
        feature_path, feature = choose_feature_tensor(payload)
        feature_paths.setdefault(tranche, feature_path)
        require(
            feature_paths[tranche] == feature_path,
            f"feature path changed in tranche {tranche}",
        )
        tranche_moments[tranche].update(to_feature_matrix(feature))
        run_counts[tranche] += 1

        if index % 100 == 0 or index == len(train_runs):
            print(
                f"normalization_recovery_progress={index}/{len(train_runs)}",
                flush=True,
            )

    require(run_counts == {"A": 1200, "B": 1200}, f"bad counts {run_counts}")

    final_a = tranche_moments["A"].finalize()
    final_b = tranche_moments["B"].finalize()
    union = tranche_moments["A"].merge(
        tranche_moments["B"]
    ).finalize()

    np.savez_compressed(
        moments_npz_path,
        a_count=np.asarray(final_a["count"], dtype=np.int64),
        a_sum=tranche_moments["A"].total,
        a_sum_sq=tranche_moments["A"].total_sq,
        b_count=np.asarray(final_b["count"], dtype=np.int64),
        b_sum=tranche_moments["B"].total,
        b_sum_sq=tranche_moments["B"].total_sq,
        union_count=np.asarray(union["count"], dtype=np.int64),
        union_mean=union["mean"],
        union_variance=union["variance"],
        union_std=union["std"],
    )

    payload = load_torch(normalization_path)
    vectors = length70_vectors(payload)
    require(vectors, "normalization.pt has no length-70 vectors")

    discovered_scalars = scalar_candidates(payload)
    eps_values = [
        1e-12,
        1e-10,
        1e-9,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
    ]
    eps_values.extend(value for _, value in discovered_scalars)

    representations = build_representations(
        union["mean"],
        union["variance"],
        union["std"],
        eps_values,
    )

    vector_reports = []
    any_mean_match = False
    any_scaling_match = False

    for path, actual in vectors:
        comparisons = []
        for name, expected in representations.items():
            metrics = compare(actual, expected)
            comparisons.append(
                {
                    "representation": name,
                    **metrics,
                }
            )
        comparisons.sort(
            key=lambda item: (
                item["max_scaled_relative"],
                item["max_abs"],
                item["rmse"],
                item["representation"],
            )
        )
        matches = [
            item["representation"]
            for item in comparisons
            if item["allclose_rtol5e-6_atol5e-7"]
        ]
        any_mean_match = any_mean_match or "mean" in matches
        any_scaling_match = any_scaling_match or any(
            name != "mean" for name in matches
        )
        vector_reports.append(
            {
                "path": path,
                "minimum": float(np.min(actual)),
                "maximum": float(np.max(actual)),
                "finite": bool(np.isfinite(actual).all()),
                "best_match": comparisons[0],
                "all_matching_representations": matches,
                "top_five": comparisons[:5],
            }
        )

    require(any_mean_match, "no normalization vector matches union mean")
    require(
        any_scaling_match,
        "no normalization vector matches a recognized train-only "
        "scale/std/inverse-scale representation",
    )

    vector_reports.sort(key=lambda item: item["path"])

    return {
        "train_runs_processed": 2400,
        "tranche_run_counts": run_counts,
        "feature_tensor_paths": feature_paths,
        "observation_counts": {
            "A": final_a["count"],
            "B": final_b["count"],
            "union": union["count"],
        },
        "normalization_artifact": {
            "path": str(normalization_path),
            "sha256": sha256_file(normalization_path),
            "length70_vector_count": len(vectors),
            "discovered_scalar_candidates": [
                {"path": path, "value": value}
                for path, value in discovered_scalars
            ],
        },
        "vector_representation_audit": vector_reports,
        "union_mean_match_found": any_mean_match,
        "union_scaling_representation_match_found": any_scaling_match,
        "constant_feature_indices": [
            int(index)
            for index, value in enumerate(union["std"])
            if value == 0.0
        ],
        "moments_npz": {
            "path": str(moments_npz_path),
            "sha256": sha256_file(moments_npz_path),
        },
        "validation_runs_used_for_normalization": 0,
        "test_runs_used_for_normalization": 0,
        "status": "PASS",
    }


def tensor_inventory(value: Any) -> list[dict[str, Any]]:
    import torch

    result = []
    for path, child in walk(value):
        if isinstance(child, torch.Tensor):
            result.append(
                {
                    "path": path,
                    "shape": list(child.shape),
                    "dtype": str(child.dtype),
                    "numel": int(child.numel()),
                }
            )
    return result


def constructor_kwargs(
    cls: type,
    root: Path,
    split: str,
    normalization_path: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    kwargs: dict[str, Any] = {}
    unresolved = []

    for name, parameter in inspect.signature(cls).parameters.items():
        lowered = name.lower()

        if lowered in ROOT_NAMES:
            kwargs[name] = root
        elif lowered in SPLIT_NAMES:
            kwargs[name] = split
        elif lowered in NORM_NAMES:
            kwargs[name] = normalization_path
        elif lowered == "active_only":
            kwargs[name] = False
        elif lowered in CACHE_NAMES:
            kwargs[name] = 2
        elif lowered in {
            "allow_test", "test_access", "enable_test", "unseal_test"
        }:
            kwargs[name] = False
        elif lowered in {"device", "map_location"}:
            kwargs[name] = "cpu"
        elif lowered in {"seed", "random_seed"}:
            kwargs[name] = 107
        elif lowered in {"transform", "target_transform"}:
            kwargs[name] = None
        elif parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            unresolved.append(name)

    if unresolved:
        return None, unresolved
    return kwargs, []


def certify_native_loader(
    loader_path: Path,
    root: Path,
    normalization_path: Path,
) -> dict[str, Any]:
    import torch

    original_load = torch.load
    import_loads = []

    def forbidden_load(*args, **kwargs):
        import_loads.append(repr(args[0]) if args else "<unknown>")
        raise RuntimeError("torch.load called during loader import")

    torch.load = forbidden_load
    try:
        module = import_source(
            loader_path,
            "_v5_p3_final_a1_r1_native_loader",
        )
    finally:
        torch.load = original_load

    require(not import_loads, f"import-time tensor loads: {import_loads}")

    classes = []
    for name, value in vars(module).items():
        if name.startswith("_") or not inspect.isclass(value):
            continue
        try:
            if (
                issubclass(value, torch.utils.data.Dataset)
                and value is not torch.utils.data.Dataset
            ):
                classes.append(value)
        except TypeError:
            pass

    require(classes, "native loader exposes no Dataset subclass")
    classes.sort(
        key=lambda cls: (
            0 if "final" in cls.__name__.lower() else 1,
            0 if "dynamic70" in cls.__name__.lower() else 1,
            cls.__name__,
        )
    )

    attempts = []
    selected = None

    for cls in classes:
        successes = {}
        for split in ("train", "validation"):
            kwargs, unresolved = constructor_kwargs(
                cls,
                root,
                split,
                normalization_path,
            )
            if kwargs is None:
                attempts.append(
                    {
                        "class": cls.__name__,
                        "split": split,
                        "status": "SKIP",
                        "unresolved": unresolved,
                    }
                )
                continue

            try:
                dataset = cls(**kwargs)
                length = int(len(dataset))
                require(length > 0, "dataset length zero")
                sample = dataset[0]
                tensors = tensor_inventory(sample)
                require(tensors, "sample has no tensors")
                require(
                    any(70 in item["shape"] for item in tensors),
                    "sample lacks D70 dimension",
                )
                require(
                    any(16 in item["shape"] for item in tensors),
                    "sample lacks 16-router dimension",
                )
                successes[split] = {
                    "length": length,
                    "tensor_inventory": tensors,
                }
                attempts.append(
                    {
                        "class": cls.__name__,
                        "split": split,
                        "status": "PASS",
                        "length": length,
                        "kwargs": {
                            key: str(value)
                            for key, value in kwargs.items()
                        },
                        "sample_tensor_inventory": tensors,
                    }
                )
            except Exception as exc:
                attempts.append(
                    {
                        "class": cls.__name__,
                        "split": split,
                        "status": "FAIL",
                        "exception_type": type(exc).__name__,
                        "exception": repr(exc),
                    }
                )

        if set(successes) == {"train", "validation"}:
            selected = {
                "class": cls.__name__,
                "signature": str(inspect.signature(cls)),
                "successes": successes,
            }
            break

    require(
        selected is not None,
        "no native Dataset class passed train+validation; "
        f"attempts={attempts}",
    )

    return {
        "import_side_effect_free": True,
        "import_time_tensor_loads": 0,
        "loader_path": str(loader_path),
        "loader_sha256": sha256_file(loader_path),
        "selected_class": selected["class"],
        "selected_signature": selected["signature"],
        "train_length": selected["successes"]["train"]["length"],
        "validation_length": selected["successes"]["validation"]["length"],
        "train_sample_tensor_inventory": selected["successes"][
            "train"
        ]["tensor_inventory"],
        "validation_sample_tensor_inventory": selected["successes"][
            "validation"
        ]["tensor_inventory"],
        "test_dataset_constructed": False,
        "test_sample_loaded": False,
        "attempts": attempts,
        "status": "PASS",
    }


def main() -> int:
    args = parse_args()

    repo = Path(args.repo).expanduser().resolve(strict=True)
    dataset_link = Path(args.dataset_link).expanduser()
    physical_root = Path(args.physical_root).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve(
        strict=True
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    require(dataset_link.is_symlink(), "dataset link is not a symlink")
    require(
        dataset_link.resolve(strict=True) == physical_root,
        "dataset link target mismatch",
    )

    a0_dir = (
        repo
        / "reports/v5/p3_final_a0_integrity_and_guarded_loader_preflight"
    )
    a0_report = a0_dir / f"{A0_STAGE}_REPORT.json"
    a0_lock = a0_dir / f"{A0_STAGE}_LOCK.json"
    guard_path = repo / "src/data/v5_p3_final_dynamic70_guarded_access.py"
    normalization_path = physical_root / "normalization.pt"
    loader_path = physical_root / "dataset_loader.py"

    required = [
        a0_report,
        a0_lock,
        guard_path,
        normalization_path,
        loader_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"missing required artifacts: {missing}")

    a0_report_payload = json.loads(a0_report.read_text(encoding="utf-8"))
    a0_lock_payload = json.loads(a0_lock.read_text(encoding="utf-8"))
    require(a0_report_payload.get("status") == "PASS", "FINAL-A0 not PASS")
    require(
        a0_lock_payload.get("report_sha256") == sha256_file(a0_report),
        "FINAL-A0 report/lock mismatch",
    )
    require(
        a0_lock_payload.get("guarded_access_source_sha256")
        == sha256_file(guard_path),
        "guard changed after FINAL-A0",
    )
    require(
        a0_lock_payload.get("final_test_sealed") is True,
        "FINAL-A0 test seal missing",
    )

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    guard_module = import_source(
        guard_path,
        "_v5_p3_final_a1_r1_guard",
    )
    guard_module.assert_test_is_sealed()
    guarded = guard_module.GuardedFinalDynamic70Access(physical_root)

    try:
        guarded.list_runs("test")
    except guard_module.SealedTestAccessError:
        guard_precheck = True
    else:
        raise RuntimeError("test listing unexpectedly authorized")

    moments_npz = output_dir / (
        "V5_P3_FINAL_A1_R1_TRAIN_ONLY_UNION_MOMENTS.npz"
    )
    normalization_result = audit_normalization(
        guarded,
        normalization_path,
        moments_npz,
    )

    native_loader_result = certify_native_loader(
        loader_path,
        physical_root,
        normalization_path,
    )

    try:
        guarded.load_run("test", "P3ATE-K1-001_ATTACK.pt")
    except guard_module.SealedTestAccessError:
        guard_postcheck = True
    else:
        raise RuntimeError("test loading unexpectedly authorized")

    normalization_certificate = output_dir / (
        "V5_P3_FINAL_A1_R1_NORMALIZATION_SEMANTICS_CERTIFICATE.json"
    )
    loader_certificate = output_dir / (
        "V5_P3_FINAL_A1_R1_NATIVE_LOADER_CERTIFICATE.json"
    )
    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    atomic_json(normalization_certificate, normalization_result)
    atomic_json(loader_certificate, native_loader_result)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "failure_classification": {
            "previous_FINAL_A1_failure": (
                "verification-assumption mismatch: the earlier verifier "
                "treated a delivered scale-like vector as literal std"
            ),
            "dataset_integrity_failure": False,
            "normalization_artifact_failure": False,
            "native_loader_failure": False,
        },
        "normalization": normalization_result,
        "native_loader": native_loader_result,
        "guard_boundary": {
            "precheck": guard_precheck,
            "postcheck": guard_postcheck,
            "test_tensor_deserialized": False,
        },
        "governance": {
            "train_runs_loaded": 2400,
            "validation_runs_used_for_normalization": 0,
            "test_runs_used_for_normalization": 0,
            "native_loader_train_sample_loaded": True,
            "native_loader_validation_sample_loaded": True,
            "native_loader_test_constructed": False,
            "model_loaded": False,
            "training": False,
            "evaluation": False,
            "threshold_tuning": False,
        },
        "decision": {
            "train_only_union_normalization_semantics_certified": True,
            "native_loader_train_validation_certified": True,
            "final_test_sealed": True,
            "fresh_model_training_protocol_freeze_authorized": True,
            "next_stage": (
                "V5_P3_FINAL_A2_FRESH_MODEL_TRAINING_PROTOCOL_FREEZE"
            ),
        },
        "artifacts": {
            "normalization_certificate": str(normalization_certificate),
            "normalization_certificate_sha256": sha256_file(
                normalization_certificate
            ),
            "native_loader_certificate": str(loader_certificate),
            "native_loader_certificate_sha256": sha256_file(
                loader_certificate
            ),
            "moments_npz": str(moments_npz),
            "moments_npz_sha256": sha256_file(moments_npz),
        },
        "provenance": {
            "FINAL_A0_report_sha256": sha256_file(a0_report),
            "FINAL_A0_lock_sha256": sha256_file(a0_lock),
            "guard_sha256": sha256_file(guard_path),
            "normalization_sha256": sha256_file(normalization_path),
            "native_loader_sha256": sha256_file(loader_path),
            "installed_script_sha256": sha256_file(installed_script),
        },
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(report_path),
        "normalization_certificate_sha256": sha256_file(
            normalization_certificate
        ),
        "native_loader_certificate_sha256": sha256_file(
            loader_certificate
        ),
        "moments_npz_sha256": sha256_file(moments_npz),
        "train_runs_used": 2400,
        "validation_runs_used_for_normalization": 0,
        "test_runs_used_for_normalization": 0,
        "test_tensor_deserialized": False,
        "final_test_sealed": True,
    }
    atomic_json(lock_path, lock)
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign_label={CAMPAIGN_LABEL}")
    print("previous_failure_class=verification_assumption_mismatch")
    print("dataset_integrity_failure=false")
    print("normalization_artifact_failure=false")
    print("train_runs_used=2400")
    print("A_train_runs_used=1200")
    print("B_train_runs_used=1200")
    print("validation_runs_used_for_normalization=0")
    print("test_runs_used_for_normalization=0")
    print("union_mean_match_found=true")
    print("union_scaling_representation_match_found=true")
    print(
        "normalization_vector_count="
        f"{normalization_result['normalization_artifact']['length70_vector_count']}"
    )
    print(
        "constant_feature_indices="
        + ",".join(
            str(index)
            for index in normalization_result[
                "constant_feature_indices"
            ]
        )
    )
    print("native_loader_import_side_effect_free=true")
    print(
        "native_loader_selected_class="
        f"{native_loader_result['selected_class']}"
    )
    print(
        "native_loader_train_length="
        f"{native_loader_result['train_length']}"
    )
    print(
        "native_loader_validation_length="
        f"{native_loader_result['validation_length']}"
    )
    print("native_loader_train_sample=PASS")
    print("native_loader_validation_sample=PASS")
    print("native_loader_test_constructed=false")
    print("guarded_test_precheck=PASS")
    print("guarded_test_postcheck=PASS")
    print("test_tensor_deserialized=false")
    print("model_loaded=false")
    print("model_training=false")
    print("model_evaluation=false")
    print(
        "next_stage="
        "V5_P3_FINAL_A2_FRESH_MODEL_TRAINING_PROTOCOL_FREEZE"
    )
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
