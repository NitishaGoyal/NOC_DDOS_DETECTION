from __future__ import annotations

import argparse
import ast
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
    "V5_P3_FINAL_A1_TRAIN_ONLY_UNION_NORMALIZATION_"
    "AND_NATIVE_LOADER_CERTIFICATION"
)
CAMPAIGN_LABEL = (
    "V5-P3-1500-D70 Final A+B Dataset "
    "— Train-Only Union Normalization and Native-Loader Certification"
)

A0_STAGE = (
    "V5_P3_FINAL_A0_DYNAMIC70_INTEGRITY_AND_"
    "GUARDED_LOADER_PREFLIGHT"
)

FEATURE_NAME_HINTS = (
    "x_log1p",
    "features_log1p",
    "dynamic70_log1p",
    "x_dynamic70_log1p",
    "x",
    "features",
    "dynamic70",
)

ROOT_PARAMETER_NAMES = {
    "root",
    "dataset_root",
    "data_root",
    "dataset_dir",
    "root_dir",
    "path",
}
SPLIT_PARAMETER_NAMES = {
    "split",
    "partition",
    "subset",
}
NORMALIZATION_PARAMETER_NAMES = {
    "normalization_path",
    "normalization_file",
    "norm_path",
    "norm_file",
}
CACHE_PARAMETER_NAMES = {
    "cache_size",
    "lru_size",
    "lru_capacity",
    "max_cache_size",
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
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def recursive_values(
    value: Any,
    path: str = "$",
) -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from recursive_values(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from recursive_values(child, f"{path}[{index}]")


def tensor_inventory(value: Any) -> list[dict[str, Any]]:
    import torch

    inventory = []
    for path, child in recursive_values(value):
        if isinstance(child, torch.Tensor):
            inventory.append(
                {
                    "path": path,
                    "shape": list(child.shape),
                    "dtype": str(child.dtype),
                    "numel": int(child.numel()),
                }
            )
    return inventory


def choose_feature_tensor(payload: Any) -> tuple[str, Any]:
    import torch

    candidates = []
    for path, child in recursive_values(payload):
        if not isinstance(child, torch.Tensor):
            continue
        shape = tuple(int(v) for v in child.shape)
        if child.ndim < 2 or 70 not in shape:
            continue

        lowered = path.lower()
        score = 0
        for rank, hint in enumerate(FEATURE_NAME_HINTS):
            if lowered.endswith("." + hint) or lowered == "$." + hint:
                score = 1000 - rank
                break
            if hint in lowered:
                score = max(score, 500 - rank)

        if 16 in shape:
            score += 100
        if child.ndim == 3:
            score += 50
        if child.is_floating_point():
            score += 20

        candidates.append((score, path, child))

    require(candidates, "no tensor containing a D70 axis was found")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score = candidates[0][0]
    best = [item for item in candidates if item[0] == best_score]
    require(
        len(best) == 1,
        "ambiguous D70 feature tensors at equal priority: "
        + ", ".join(path for _, path, _ in best),
    )
    _, path, tensor = best[0]
    return path, tensor


def feature_matrix_2d(tensor: Any) -> np.ndarray:
    import torch

    array = tensor.detach().cpu()
    shape = tuple(int(v) for v in array.shape)
    axes = [index for index, size in enumerate(shape) if size == 70]
    require(
        len(axes) == 1,
        f"feature tensor must have exactly one D70 axis, got {shape}",
    )
    feature_axis = axes[0]
    array = torch.movedim(array, feature_axis, -1)
    return (
        array.reshape(-1, 70)
        .to(dtype=torch.float64)
        .numpy()
    )


@dataclass
class RunningMoments:
    count: int
    total: np.ndarray
    total_sq: np.ndarray

    @classmethod
    def empty(cls) -> "RunningMoments":
        return cls(
            count=0,
            total=np.zeros(70, dtype=np.float64),
            total_sq=np.zeros(70, dtype=np.float64),
        )

    def update(self, matrix: np.ndarray) -> None:
        require(
            matrix.ndim == 2 and matrix.shape[1] == 70,
            f"invalid feature matrix shape {matrix.shape}",
        )
        require(
            np.isfinite(matrix).all(),
            "non-finite values found in train feature matrix",
        )
        self.count += int(matrix.shape[0])
        self.total += matrix.sum(axis=0, dtype=np.float64)
        self.total_sq += np.square(
            matrix,
            dtype=np.float64,
        ).sum(axis=0, dtype=np.float64)

    def combine(self, other: "RunningMoments") -> "RunningMoments":
        return RunningMoments(
            count=self.count + other.count,
            total=self.total + other.total,
            total_sq=self.total_sq + other.total_sq,
        )

    def finalize(self) -> dict[str, np.ndarray | int]:
        require(self.count > 0, "empty running moments")
        mean = self.total / float(self.count)
        variance = self.total_sq / float(self.count) - np.square(mean)
        variance = np.maximum(variance, 0.0)
        std_population = np.sqrt(variance)
        std_zero_to_one = np.where(std_population == 0.0, 1.0, std_population)
        return {
            "count": self.count,
            "mean": mean,
            "std_population": std_population,
            "std_zero_to_one": std_zero_to_one,
        }


def load_torch(path: Path) -> Any:
    import torch

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def vector70_candidates(
    payload: Any,
) -> list[tuple[str, np.ndarray]]:
    import torch

    candidates = []
    for path, child in recursive_values(payload):
        array = None
        if isinstance(child, torch.Tensor) and int(child.numel()) == 70:
            array = (
                child.detach()
                .cpu()
                .reshape(-1)
                .to(dtype=torch.float64)
                .numpy()
            )
        elif isinstance(child, np.ndarray) and child.size == 70:
            array = child.reshape(-1).astype(np.float64, copy=False)
        elif (
            isinstance(child, list)
            and len(child) == 70
            and all(isinstance(item, (int, float)) for item in child)
        ):
            array = np.asarray(child, dtype=np.float64)

        if array is not None:
            candidates.append((path, array))
    return candidates


def choose_mean_and_std(
    normalization_payload: Any,
) -> tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]]:
    vectors = vector70_candidates(normalization_payload)
    require(vectors, "normalization artifact has no length-70 vectors")

    mean_candidates = [
        item for item in vectors if "mean" in item[0].lower()
    ]
    std_candidates = [
        item
        for item in vectors
        if any(
            token in item[0].lower()
            for token in ("std", "scale", "sigma")
        )
    ]

    require(
        len(mean_candidates) == 1,
        "expected exactly one mean vector, found "
        + ", ".join(path for path, _ in mean_candidates),
    )
    require(
        std_candidates,
        "no std/scale vector found in normalization artifact",
    )

    # Prefer an explicitly effective/clamped std when multiple vectors exist.
    std_candidates.sort(
        key=lambda item: (
            0
            if any(
                token in item[0].lower()
                for token in ("effective", "clamp", "safe")
            )
            else 1,
            item[0],
        )
    )
    return mean_candidates[0], std_candidates[0]


def error_summary(
    actual: np.ndarray,
    expected: np.ndarray,
) -> dict[str, float]:
    difference = np.abs(actual - expected)
    scale = np.maximum(np.abs(expected), 1.0)
    relative = difference / scale
    return {
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()),
        "max_scaled_relative": float(relative.max(initial=0.0)),
    }


def verify_normalization(
    root: Path,
    guarded: Any,
    normalization_path: Path,
) -> dict[str, Any]:
    moments = {
        "A": RunningMoments.empty(),
        "B": RunningMoments.empty(),
    }
    feature_paths = {}
    run_counts = {"A": 0, "B": 0}

    train_runs = guarded.list_runs("train")
    require(len(train_runs) == 2400, "guarded train run count mismatch")

    for index, run_path in enumerate(train_runs, start=1):
        tranche = run_path.name[2]
        require(tranche in moments, f"unexpected tranche in {run_path.name}")

        payload = guarded.load_run("train", run_path.name)
        feature_path, feature_tensor = choose_feature_tensor(payload)
        matrix = feature_matrix_2d(feature_tensor)
        moments[tranche].update(matrix)
        feature_paths.setdefault(tranche, feature_path)
        require(
            feature_paths[tranche] == feature_path,
            f"feature tensor path changed within tranche {tranche}: "
            f"{feature_paths[tranche]} vs {feature_path}",
        )
        run_counts[tranche] += 1

        if index % 100 == 0 or index == len(train_runs):
            print(
                f"normalization_progress={index}/{len(train_runs)}",
                flush=True,
            )

    require(run_counts == {"A": 1200, "B": 1200}, f"bad run counts {run_counts}")

    finalized_a = moments["A"].finalize()
    finalized_b = moments["B"].finalize()
    union = moments["A"].combine(moments["B"]).finalize()

    normalization_payload = load_torch(normalization_path)
    (mean_path, artifact_mean), (
        std_path,
        artifact_std,
    ) = choose_mean_and_std(normalization_payload)

    mean_error = error_summary(artifact_mean, union["mean"])
    std_raw_error = error_summary(
        artifact_std,
        union["std_population"],
    )
    std_safe_error = error_summary(
        artifact_std,
        union["std_zero_to_one"],
    )
    if std_safe_error["max_abs"] <= std_raw_error["max_abs"]:
        selected_std_reference = "population_std_zero_to_one"
        selected_std_error = std_safe_error
    else:
        selected_std_reference = "population_std_raw"
        selected_std_error = std_raw_error

    tolerance = 1.0e-5
    require(
        mean_error["max_abs"] <= tolerance
        and mean_error["max_scaled_relative"] <= tolerance,
        f"union mean mismatch: {mean_error}",
    )
    require(
        selected_std_error["max_abs"] <= tolerance
        and selected_std_error["max_scaled_relative"] <= tolerance,
        "union std mismatch: "
        f"raw={std_raw_error}, safe={std_safe_error}",
    )

    a_mean_error = error_summary(artifact_mean, finalized_a["mean"])
    b_mean_error = error_summary(artifact_mean, finalized_b["mean"])

    return {
        "train_runs_processed": len(train_runs),
        "tranche_run_counts": run_counts,
        "feature_tensor_paths": feature_paths,
        "observation_counts": {
            "A": int(finalized_a["count"]),
            "B": int(finalized_b["count"]),
            "union": int(union["count"]),
        },
        "normalization_artifact": {
            "path": str(normalization_path),
            "sha256": sha256_file(normalization_path),
            "mean_path": mean_path,
            "std_path": std_path,
        },
        "union_comparison": {
            "mean_error": mean_error,
            "std_raw_error": std_raw_error,
            "std_safe_error": std_safe_error,
            "selected_std_reference": selected_std_reference,
            "selected_std_error": selected_std_error,
            "tolerance": tolerance,
            "status": "PASS",
        },
        "diagnostic_nonselected_comparisons": {
            "artifact_vs_A_only_mean": a_mean_error,
            "artifact_vs_B_only_mean": b_mean_error,
        },
        "constant_feature_indices": [
            int(index)
            for index, value in enumerate(union["std_population"])
            if value == 0.0
        ],
        "validation_runs_loaded": 0,
        "test_runs_loaded": 0,
    }


def public_loader_inventory(module: Any, loader_path: Path) -> dict[str, Any]:
    import torch

    dataset_classes = []
    public_functions = []

    for name, value in sorted(vars(module).items()):
        if name.startswith("_"):
            continue
        if inspect.isclass(value):
            try:
                is_dataset = issubclass(value, torch.utils.data.Dataset)
            except TypeError:
                is_dataset = False
            if is_dataset and value is not torch.utils.data.Dataset:
                dataset_classes.append(
                    {
                        "name": name,
                        "signature": str(inspect.signature(value)),
                    }
                )
        elif inspect.isfunction(value):
            public_functions.append(
                {
                    "name": name,
                    "signature": str(inspect.signature(value)),
                }
            )

    return {
        "path": str(loader_path),
        "sha256": sha256_file(loader_path),
        "dataset_classes": dataset_classes,
        "public_functions": public_functions,
    }


def build_constructor_kwargs(
    cls: type,
    root: Path,
    split: str,
    normalization_path: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    signature = inspect.signature(cls)
    kwargs: dict[str, Any] = {}
    unresolved = []

    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        normalized = name.lower()

        if normalized in ROOT_PARAMETER_NAMES:
            kwargs[name] = root
        elif normalized in SPLIT_PARAMETER_NAMES:
            kwargs[name] = split
        elif normalized in NORMALIZATION_PARAMETER_NAMES:
            kwargs[name] = normalization_path
        elif normalized == "active_only":
            kwargs[name] = False
        elif normalized in CACHE_PARAMETER_NAMES:
            kwargs[name] = 2
        elif normalized in {
            "allow_test",
            "test_access",
            "enable_test",
            "unseal_test",
        }:
            kwargs[name] = False
        elif normalized in {"map_location", "device"}:
            kwargs[name] = "cpu"
        elif normalized in {"seed", "random_seed"}:
            kwargs[name] = 107
        elif normalized in {"transform", "target_transform"}:
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
    dataset_link: Path,
    normalization_path: Path,
) -> dict[str, Any]:
    import torch

    original_torch_load = torch.load
    import_load_calls = []

    def forbidden_import_load(*args, **kwargs):
        import_load_calls.append(repr(args[0]) if args else "<unknown>")
        raise RuntimeError(
            "dataset_loader.py attempted tensor deserialization at import time"
        )

    torch.load = forbidden_import_load
    try:
        module = import_source(
            loader_path,
            "_v5_p3_final_native_dataset_loader",
        )
    finally:
        torch.load = original_torch_load

    require(
        not import_load_calls,
        f"native loader had import-time tensor loads: {import_load_calls}",
    )

    inventory = public_loader_inventory(module, loader_path)
    dataset_class_names = [
        entry["name"] for entry in inventory["dataset_classes"]
    ]
    require(dataset_class_names, "native loader exposes no Dataset subclass")

    class_values = [
        getattr(module, name) for name in dataset_class_names
    ]
    class_values.sort(
        key=lambda cls: (
            0 if "final" in cls.__name__.lower() else 1,
            0 if "dynamic70" in cls.__name__.lower() else 1,
            cls.__name__,
        )
    )

    attempts = []
    selected = None

    for cls in class_values:
        for split in ("train", "validation"):
            kwargs, unresolved = build_constructor_kwargs(
                cls,
                dataset_link,
                split,
                normalization_path,
            )
            if kwargs is None:
                attempts.append(
                    {
                        "class": cls.__name__,
                        "split": split,
                        "status": "SKIP",
                        "unresolved_required_parameters": unresolved,
                    }
                )
                continue
            try:
                dataset = cls(**kwargs)
                length = len(dataset)
                require(length > 0, "native dataset length is zero")
                sample = dataset[0]
                tensors = tensor_inventory(sample)
                require(tensors, "native loader sample contains no tensors")
                require(
                    any(70 in item["shape"] for item in tensors),
                    "native loader sample has no D70 tensor",
                )
                require(
                    any(16 in item["shape"] for item in tensors),
                    "native loader sample has no 16-router tensor",
                )
                attempts.append(
                    {
                        "class": cls.__name__,
                        "split": split,
                        "status": "PASS",
                        "kwargs": {
                            key: str(value)
                            for key, value in kwargs.items()
                        },
                        "length": int(length),
                        "sample_tensor_inventory": tensors,
                    }
                )
                if selected is None:
                    selected = {
                        "class": cls,
                        "kwargs_by_split": {},
                        "length_by_split": {},
                        "sample_by_split": {},
                    }
                if selected["class"] is cls:
                    selected["kwargs_by_split"][split] = kwargs
                    selected["length_by_split"][split] = int(length)
                    selected["sample_by_split"][split] = tensors
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

        if (
            selected is not None
            and selected["class"] is cls
            and set(selected["length_by_split"]) == {"train", "validation"}
        ):
            break
        if selected is not None and selected["class"] is cls:
            selected = None

    require(
        selected is not None,
        "no native Dataset class passed both train and validation "
        f"certification; attempts={attempts}",
    )

    # Confirm the repository guard rejects test before the native loader is
    # ever invoked for that split.
    guard_path = (
        dataset_link.parents[3]
        / "src/data/v5_p3_final_dynamic70_guarded_access.py"
    )
    # The computed path above is not reliable for arbitrary repo layouts;
    # caller records the actual guard separately. This certification uses
    # the already imported guarded module supplied by main.

    return {
        "import_side_effect_free": True,
        "import_time_tensor_loads": 0,
        "inventory": inventory,
        "attempts": attempts,
        "selected_class": selected["class"].__name__,
        "selected_signature": str(
            inspect.signature(selected["class"])
        ),
        "length_by_split": selected["length_by_split"],
        "sample_tensor_inventory_by_split": selected["sample_by_split"],
        "train_sample_loaded": True,
        "validation_sample_loaded": True,
        "test_dataset_constructed": False,
        "test_sample_loaded": False,
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

    require(dataset_link.is_symlink(), "canonical dataset path is not a symlink")
    require(
        dataset_link.resolve(strict=True) == physical_root,
        "canonical dataset link target changed after FINAL-A0",
    )

    a0_dir = (
        repo
        / "reports/v5/p3_final_a0_integrity_and_guarded_loader_preflight"
    )
    a0_report_path = a0_dir / f"{A0_STAGE}_REPORT.json"
    a0_lock_path = a0_dir / f"{A0_STAGE}_LOCK.json"
    guard_path = repo / "src/data/v5_p3_final_dynamic70_guarded_access.py"

    loader_path = physical_root / "dataset_loader.py"
    normalization_path = physical_root / "normalization.pt"
    split_manifest_path = physical_root / "split_manifest.json"
    final_cert_path = physical_root / "FINAL_HANDOVER_CERTIFICATION.json"

    required = [
        a0_report_path,
        a0_lock_path,
        guard_path,
        loader_path,
        normalization_path,
        split_manifest_path,
        final_cert_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"required artifacts missing: {missing}")

    a0_report = json.loads(a0_report_path.read_text(encoding="utf-8"))
    a0_lock = json.loads(a0_lock_path.read_text(encoding="utf-8"))
    require(a0_report.get("status") == "PASS", "FINAL-A0 is not PASS")
    require(
        a0_lock.get("report_sha256") == sha256_file(a0_report_path),
        "FINAL-A0 report/lock mismatch",
    )
    require(
        a0_lock.get("guarded_access_source_sha256")
        == sha256_file(guard_path),
        "guarded-access source changed after FINAL-A0",
    )
    require(
        a0_lock.get("final_test_sealed") is True,
        "FINAL-A0 does not preserve test seal",
    )
    require(
        a0_lock.get("test_tensor_deserialized") is False,
        "FINAL-A0 reports test tensor access",
    )

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    guarded_module = import_source(
        guard_path,
        "_v5_p3_final_a1_guarded_access",
    )
    guarded_module.assert_test_is_sealed()
    guarded = guarded_module.GuardedFinalDynamic70Access(
        physical_root
    )

    # Explicitly prove the guard still rejects test before all later work.
    try:
        guarded.list_runs("test")
    except guarded_module.SealedTestAccessError:
        test_guard_precheck = True
    else:
        raise RuntimeError("test listing unexpectedly authorized")

    normalization_result = verify_normalization(
        physical_root,
        guarded,
        normalization_path,
    )

    native_loader_result = certify_native_loader(
        loader_path,
        dataset_link.resolve(strict=True),
        normalization_path,
    )

    try:
        guarded.load_run(
            "test",
            "P3ATE-K1-001_ATTACK.pt",
        )
    except guarded_module.SealedTestAccessError:
        test_guard_postcheck = True
    else:
        raise RuntimeError("test loading unexpectedly authorized")

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    normalization_report_path = output_dir / (
        "V5_P3_FINAL_A1_TRAIN_ONLY_UNION_NORMALIZATION_CERTIFICATE.json"
    )
    loader_report_path = output_dir / (
        "V5_P3_FINAL_A1_NATIVE_LOADER_CERTIFICATE.json"
    )
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    atomic_json(normalization_report_path, normalization_result)
    atomic_json(loader_report_path, native_loader_result)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "scope": (
            "Recompute Dynamic70 moments from all 2400 final A+B training "
            "run tensors only; verify the delivered normalization artifact; "
            "import and certify the delivered native dataset loader on train "
            "and validation only; preserve the final test seal."
        ),
        "dataset": {
            "physical_root": str(physical_root),
            "dataset_link": str(dataset_link),
            "symlink_target_exact": True,
        },
        "normalization": normalization_result,
        "native_loader": native_loader_result,
        "guard_boundary": {
            "guard_path": str(guard_path),
            "guard_sha256": sha256_file(guard_path),
            "test_guard_precheck": test_guard_precheck,
            "test_guard_postcheck": test_guard_postcheck,
            "test_tensor_deserialized": False,
        },
        "governance": {
            "train_runs_loaded": 2400,
            "validation_runs_loaded_for_normalization": 0,
            "test_runs_loaded_for_normalization": 0,
            "native_loader_train_sample_loaded": True,
            "native_loader_validation_sample_loaded": True,
            "native_loader_test_constructed": False,
            "native_loader_test_sample_loaded": False,
            "model_loaded": False,
            "model_training": False,
            "model_evaluation": False,
            "threshold_tuning": False,
        },
        "decision": {
            "union_normalization_certified": True,
            "native_loader_train_validation_certified": True,
            "guarded_test_boundary_retained": True,
            "final_training_protocol_freeze_authorized": True,
            "next_stage": (
                "V5_P3_FINAL_A2_FRESH_MODEL_TRAINING_PROTOCOL_FREEZE"
            ),
        },
        "artifacts": {
            "normalization_certificate": str(
                normalization_report_path
            ),
            "normalization_certificate_sha256": sha256_file(
                normalization_report_path
            ),
            "native_loader_certificate": str(loader_report_path),
            "native_loader_certificate_sha256": sha256_file(
                loader_report_path
            ),
        },
        "provenance": {
            "FINAL_A0_report_sha256": sha256_file(a0_report_path),
            "FINAL_A0_lock_sha256": sha256_file(a0_lock_path),
            "normalization_pt_sha256": sha256_file(normalization_path),
            "dataset_loader_py_sha256": sha256_file(loader_path),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "final_handover_certification_sha256": sha256_file(
                final_cert_path
            ),
            "installed_script_sha256": sha256_file(
                installed_script
            ),
        },
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(report_path),
        "normalization_certificate_sha256": sha256_file(
            normalization_report_path
        ),
        "native_loader_certificate_sha256": sha256_file(
            loader_report_path
        ),
        "normalization_pt_sha256": sha256_file(normalization_path),
        "dataset_loader_py_sha256": sha256_file(loader_path),
        "guard_sha256": sha256_file(guard_path),
        "train_runs_used_for_normalization": 2400,
        "validation_runs_used_for_normalization": 0,
        "test_runs_used_for_normalization": 0,
        "test_tensor_deserialized": False,
        "final_test_sealed": True,
    }
    atomic_json(lock_path, lock)
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    comparison = normalization_result["union_comparison"]
    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign_label={CAMPAIGN_LABEL}")
    print("train_runs_used_for_normalization=2400")
    print("A_train_runs_used=1200")
    print("B_train_runs_used=1200")
    print("validation_runs_used_for_normalization=0")
    print("test_runs_used_for_normalization=0")
    print("union_normalization_mean_match=true")
    print("union_normalization_std_match=true")
    print(
        "normalization_mean_max_abs_error="
        f"{comparison['mean_error']['max_abs']:.17g}"
    )
    print(
        "normalization_std_max_abs_error="
        f"{comparison['selected_std_error']['max_abs']:.17g}"
    )
    print(
        "normalization_std_reference="
        f"{comparison['selected_std_reference']}"
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
        f"{native_loader_result['length_by_split']['train']}"
    )
    print(
        "native_loader_validation_length="
        f"{native_loader_result['length_by_split']['validation']}"
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
