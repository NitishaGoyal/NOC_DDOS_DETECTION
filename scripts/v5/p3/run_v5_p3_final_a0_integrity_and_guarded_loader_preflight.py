from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STAGE = (
    "V5_P3_FINAL_A0_DYNAMIC70_INTEGRITY_AND_"
    "GUARDED_LOADER_PREFLIGHT"
)
CAMPAIGN_LABEL = (
    "V5-P3-1500-D70 Final A+B Dataset "
    "— Integrity and Guarded-Loader Preflight"
)
EXPECTED_ARCHIVE_SHA256 = (
    "371137b88618db129c5ab32503c9081188cd64e88a0f9503d29c452d7705c3fa"
)
EXPECTED_RUN_COUNTS = {
    "train": {"A": 1200, "B": 1200, "total": 2400},
    "validation": {"A": 150, "B": 150, "total": 300},
    "test": {"A": 150, "B": 150, "total": 300},
}
SPLIT_CODES = {
    "train": "TR",
    "validation": "VA",
    "test": "TE",
}
RUN_PATTERN = re.compile(
    r"^P3(?P<tranche>[AB])"
    r"(?P<split_code>TR|VA|TE)"
    r"-K(?P<k>[1-4])"
    r"-(?P<index>[0-9]{3})"
    r"_(?P<label>ATTACK|CONTROL)[.]pt$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dataset-link", required=True)
    parser.add_argument("--physical-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package-dir", required=True)
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def recursive_items(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from recursive_items(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from recursive_items(child, f"{path}[{index}]")


def feature_schema_evidence(payload: Any) -> dict[str, Any]:
    list_candidates = []
    numeric_candidates = []
    mask_candidates = []

    for path, value in recursive_items(payload):
        lowered = path.lower()
        if isinstance(value, list) and len(value) == 70:
            list_candidates.append(
                {
                    "path": path,
                    "length": 70,
                    "all_strings": all(
                        isinstance(item, str) for item in value
                    ),
                }
            )
        if (
            isinstance(value, int)
            and value == 70
            and any(token in lowered for token in ("feature", "dynamic", "d70"))
        ):
            numeric_candidates.append(path)
        if (
            isinstance(value, int)
            and value == 10
            and "mask" in lowered
        ):
            mask_candidates.append(path)

    return {
        "D70_list_candidates": list_candidates,
        "D70_numeric_candidates": numeric_candidates,
        "mask10_numeric_candidates": mask_candidates,
        "D70_supported": bool(list_candidates or numeric_candidates),
    }


def parse_sha256sums(path: Path, root: Path) -> list[tuple[str, Path, str]]:
    entries = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line:
            continue
        match = re.match(r"^([0-9a-fA-F]{64})[ \t]+[* ]?(.+)$", line)
        if match is None:
            raise RuntimeError(
                f"invalid SHA256SUMS line {line_number}: {raw!r}"
            )
        expected = match.group(1).lower()
        rel_text = match.group(2).strip()
        if rel_text.startswith("./"):
            rel_text = rel_text[2:]
        prefix = root.name + "/"
        if rel_text.startswith(prefix):
            rel_text = rel_text[len(prefix):]

        relative = Path(rel_text)
        candidate = (root / relative).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"SHA256SUMS entry escapes dataset root: {rel_text}"
            ) from exc
        entries.append((rel_text, candidate, expected))
    require(entries, "SHA256SUMS.txt contains no entries")
    return entries


def verify_internal_hashes(
    entries: list[tuple[str, Path, str]],
    root: Path,
) -> dict[str, Any]:
    failures = []
    test_files_hashed = 0
    bytes_hashed = 0

    for index, (relative, path, expected) in enumerate(entries, start=1):
        if not path.is_file():
            failures.append(
                {
                    "path": relative,
                    "reason": "missing",
                }
            )
            continue
        actual = sha256_file(path)
        bytes_hashed += path.stat().st_size
        if "/runs/test/" in f"/{relative}":
            test_files_hashed += 1
        if actual != expected:
            failures.append(
                {
                    "path": relative,
                    "reason": "sha256_mismatch",
                    "expected": expected,
                    "actual": actual,
                }
            )
        if index % 100 == 0 or index == len(entries):
            print(
                f"hash_progress={index}/{len(entries)}",
                flush=True,
            )

    require(
        not failures,
        f"internal SHA256 verification failed: {failures[:5]}",
    )
    return {
        "entries_verified": len(entries),
        "bytes_hashed": bytes_hashed,
        "test_files_hashed_for_integrity": test_files_hashed,
        "failures": failures,
    }


def inventory_runs(root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    global_basenames: set[str] = set()

    for split, expected in EXPECTED_RUN_COUNTS.items():
        split_dir = root / "runs" / split
        require(split_dir.is_dir(), f"missing split directory: {split_dir}")

        runs = sorted(split_dir.glob("*.pt"))
        parsed = []
        pairs: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
        tranche_counts = Counter()
        k_run_counts = Counter()
        k_pair_counts = Counter()

        for path in runs:
            require(
                path.name not in global_basenames,
                f"duplicate run basename across splits: {path.name}",
            )
            global_basenames.add(path.name)

            match = RUN_PATTERN.fullmatch(path.name)
            require(match is not None, f"invalid run filename: {path.name}")
            fields = match.groupdict()
            require(
                fields["split_code"] == SPLIT_CODES[split],
                f"split code mismatch for {path.name}",
            )

            tranche = fields["tranche"]
            k = int(fields["k"])
            pair_index = int(fields["index"])
            label = fields["label"]
            pair_key = (
                tranche,
                fields["split_code"],
                k,
                pair_index,
            )
            pairs[pair_key].add(label)
            tranche_counts[tranche] += 1
            k_run_counts[str(k)] += 1
            parsed.append(path.name)

        malformed_pairs = {
            "|".join(map(str, key)): sorted(labels)
            for key, labels in pairs.items()
            if labels != {"ATTACK", "CONTROL"}
        }
        require(
            not malformed_pairs,
            f"attack/control pairing failure in {split}: "
            f"{list(malformed_pairs.items())[:5]}",
        )

        for tranche in ("A", "B"):
            require(
                tranche_counts[tranche] == expected[tranche],
                f"{split} tranche {tranche} run count "
                f"{tranche_counts[tranche]} != {expected[tranche]}",
            )
        require(
            len(runs) == expected["total"],
            f"{split} total run count {len(runs)} != {expected['total']}",
        )

        for tranche, _, k, _ in pairs:
            k_pair_counts[f"{tranche}:K{k}"] += 1

        report[split] = {
            "run_count": len(runs),
            "pair_count": len(pairs),
            "tranche_run_counts": dict(sorted(tranche_counts.items())),
            "K_run_counts": dict(sorted(k_run_counts.items())),
            "tranche_K_pair_counts": dict(sorted(k_pair_counts.items())),
            "attack_control_pairing": "PASS",
            "filename_split_namespace": "PASS",
            "sample_first": parsed[:3],
            "sample_last": parsed[-3:],
        }

    return report


def static_loader_inventory(loader_path: Path) -> dict[str, Any]:
    source = loader_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(loader_path))
    classes = [
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    ]
    functions = [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return {
        "path": str(loader_path),
        "sha256": sha256_file(loader_path),
        "python_syntax": "PASS",
        "classes": classes,
        "functions": functions,
        "source_bytes": len(source.encode("utf-8")),
        "imported": False,
        "dataset_accessed": False,
    }


def recursive_tensor_shapes(value: Any, path: str = "$") -> list[dict[str, Any]]:
    import torch

    found = []
    if isinstance(value, torch.Tensor):
        found.append(
            {
                "path": path,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": int(value.numel()),
            }
        )
    elif isinstance(value, dict):
        for key, child in value.items():
            found.extend(
                recursive_tensor_shapes(child, f"{path}.{key}")
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(
                recursive_tensor_shapes(child, f"{path}[{index}]")
            )
    return found


def select_sample_run(guarded, split: str, tranche: str) -> Path:
    attack = guarded.list_runs(
        split,
        tranche=tranche,
        label="ATTACK",
    )
    require(attack, f"no {split} tranche {tranche} attack runs")
    return attack[0]


def sample_allowed_tensors(
    guarded,
) -> dict[str, Any]:
    samples = []
    d70_observed = False
    router16_observed = False

    for split in ("train", "validation"):
        for tranche in ("A", "B"):
            path = select_sample_run(guarded, split, tranche)
            payload = guarded.load_run(split, path.name)
            shapes = recursive_tensor_shapes(payload)
            require(shapes, f"no tensors found in allowed run {path}")
            d70_observed = d70_observed or any(
                70 in item["shape"] for item in shapes
            )
            router16_observed = router16_observed or any(
                16 in item["shape"] for item in shapes
            )
            samples.append(
                {
                    "split": split,
                    "tranche": tranche,
                    "run": path.name,
                    "tensor_count": len(shapes),
                    "tensor_shapes": shapes,
                }
            )

    require(
        d70_observed,
        "no dimension of size 70 observed in allowed sample tensors",
    )
    require(
        router16_observed,
        "no dimension of size 16 observed in allowed sample tensors",
    )
    return {
        "samples_loaded": len(samples),
        "sample_details": samples,
        "D70_dimension_observed": d70_observed,
        "router16_dimension_observed": router16_observed,
        "test_tensor_deserialized": False,
    }


def normalization_inventory(path: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    shapes = recursive_tensor_shapes(payload)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "tensor_shapes": shapes,
        "D70_dimension_observed": any(
            70 in item["shape"] for item in shapes
        ),
        "support_artifact_loaded": True,
        "test_tensor_loaded": False,
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve(strict=True)
    dataset_link = Path(args.dataset_link).expanduser()
    physical_root = Path(args.physical_root).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    package_dir = Path(args.package_dir).expanduser().resolve(strict=True)
    installed_script = Path(args.installed_script).expanduser().resolve(
        strict=True
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    inventory_path = output_dir / (
        "V5_P3_FINAL_DYNAMIC70_SPLIT_AND_PAIR_INVENTORY.json"
    )
    hash_path = output_dir / (
        "V5_P3_FINAL_DYNAMIC70_INTERNAL_SHA256_VERIFICATION.json"
    )
    loader_path = output_dir / (
        "V5_P3_FINAL_DYNAMIC70_GUARDED_ACCESS_PREFLIGHT.json"
    )
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    require(dataset_link.is_symlink(), f"dataset link is not a symlink: {dataset_link}")
    resolved_link = dataset_link.resolve(strict=True)
    require(
        resolved_link == physical_root,
        f"dataset link target mismatch: {resolved_link} != {physical_root}",
    )
    require(physical_root.is_dir(), f"physical root is not a directory: {physical_root}")

    support_names = [
        "feature_schema.json",
        "dataset_summary.json",
        "README.md",
        "split_manifest.json",
        "topology.pt",
        "normalization.pt",
        "FINAL_HANDOVER_CERTIFICATION.json",
        "SHA256SUMS.txt",
        "tensor_manifest.json",
        "campaign_manifest.json",
        "dataset_loader.py",
        "FINAL_DATASET_COMPLETE.json",
    ]
    support_paths = {
        name: physical_root / name for name in support_names
    }
    missing = [
        str(path) for path in support_paths.values() if not path.is_file()
    ]
    require(not missing, f"missing final dataset support artifacts: {missing}")

    feature_schema = load_json(support_paths["feature_schema.json"])
    dataset_summary = load_json(support_paths["dataset_summary.json"])
    split_manifest = load_json(support_paths["split_manifest.json"])
    tensor_manifest = load_json(support_paths["tensor_manifest.json"])
    campaign_manifest = load_json(support_paths["campaign_manifest.json"])
    final_certification = load_json(
        support_paths["FINAL_HANDOVER_CERTIFICATION.json"]
    )
    final_complete = load_json(
        support_paths["FINAL_DATASET_COMPLETE.json"]
    )

    schema_evidence = feature_schema_evidence(feature_schema)
    require(
        schema_evidence["D70_supported"],
        "feature_schema.json does not expose a verifiable D70 contract",
    )

    run_inventory = inventory_runs(physical_root)
    atomic_json(inventory_path, run_inventory)

    sha_entries = parse_sha256sums(
        support_paths["SHA256SUMS.txt"],
        physical_root,
    )
    hash_verification = verify_internal_hashes(
        sha_entries,
        physical_root,
    )
    atomic_json(hash_path, hash_verification)

    native_loader_inventory = static_loader_inventory(
        support_paths["dataset_loader.py"]
    )

    candidate_guard = package_dir / (
        "v5_p3_final_dynamic70_guarded_access.py"
    )
    require(candidate_guard.is_file(), f"candidate guard missing: {candidate_guard}")
    guard_module = import_source(
        candidate_guard,
        "_v5_p3_final_dynamic70_guard_candidate",
    )
    guard_module.assert_test_is_sealed()
    guarded = guard_module.GuardedFinalDynamic70Access(physical_root)

    require(
        len(guarded.list_runs("train")) == 2400,
        "guarded train listing mismatch",
    )
    require(
        len(guarded.list_runs("validation")) == 300,
        "guarded validation listing mismatch",
    )

    sealed_test_name = next(
        (physical_root / "runs" / "test").glob("*.pt")
    ).name
    test_guard_checks = {}

    try:
        guarded.list_runs("test")
    except guard_module.SealedTestAccessError:
        test_guard_checks["list_runs_test_denied"] = True
    else:
        raise RuntimeError("guarded list_runs unexpectedly exposed test")

    try:
        guarded.load_run("test", sealed_test_name)
    except guard_module.SealedTestAccessError:
        test_guard_checks["load_run_test_denied_before_deserialization"] = True
    else:
        raise RuntimeError("guarded load_run unexpectedly exposed test")

    sample_inventory = sample_allowed_tensors(guarded)
    normalization = normalization_inventory(
        support_paths["normalization.pt"]
    )

    loader_preflight = {
        "native_dataset_loader": native_loader_inventory,
        "candidate_guard_sha256": sha256_file(candidate_guard),
        "test_guard_checks": test_guard_checks,
        "allowed_sample_inventory": sample_inventory,
        "normalization_inventory": normalization,
        "test_directory_enumerated_for_filename_count_only": True,
        "test_bytes_hashed_for_integrity_only": (
            hash_verification["test_files_hashed_for_integrity"] > 0
        ),
        "test_tensor_deserialized": False,
        "native_loader_imported": False,
    }
    atomic_json(loader_path, loader_preflight)

    installed_guard = (
        repo / "src/data/v5_p3_final_dynamic70_guarded_access.py"
    )
    installed_guard.parent.mkdir(parents=True, exist_ok=True)
    if installed_guard.exists():
        require(
            sha256_file(installed_guard) == sha256_file(candidate_guard),
            f"existing guarded access module differs: {installed_guard}",
        )
    else:
        temporary = installed_guard.with_suffix(
            installed_guard.suffix + ".tmp"
        )
        shutil.copy2(candidate_guard, temporary)
        os.replace(temporary, installed_guard)
    require(
        sha256_file(installed_guard) == sha256_file(candidate_guard),
        "installed guarded access SHA mismatch",
    )

    top_level_support_hashes = {
        name: sha256_file(path)
        for name, path in sorted(support_paths.items())
    }
    support_text = "\n".join(
        [
            json.dumps(final_certification, sort_keys=True),
            json.dumps(final_complete, sort_keys=True),
            json.dumps(campaign_manifest, sort_keys=True),
            support_paths["README.md"].read_text(
                encoding="utf-8",
                errors="replace",
            ),
        ]
    )
    archive_sha_reference_found = (
        EXPECTED_ARCHIVE_SHA256 in support_text
    )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "scope": (
            "Read-only final A+B dataset integrity, filename-level split and "
            "attack/control pair verification, D70 schema evidence, internal "
            "checksum verification, limited train/validation tensor sampling, "
            "and installation of a test-denying guarded access boundary."
        ),
        "dataset": {
            "physical_root": str(physical_root),
            "dataset_link": str(dataset_link),
            "resolved_link": str(resolved_link),
            "symlink_target_exact": True,
            "expected_archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "archive_sha_reference_found_in_support_metadata": (
                archive_sha_reference_found
            ),
            "root_modified": False,
        },
        "schema": schema_evidence,
        "splits": run_inventory,
        "integrity": {
            "internal_SHA256SUMS_path": str(
                support_paths["SHA256SUMS.txt"]
            ),
            "internal_SHA256SUMS_sha256": sha256_file(
                support_paths["SHA256SUMS.txt"]
            ),
            **hash_verification,
        },
        "guarded_access": {
            "installed_path": str(installed_guard),
            "installed_sha256": sha256_file(installed_guard),
            "train_run_listing": 2400,
            "validation_run_listing": 300,
            "test_listing_denied": True,
            "test_load_denied_before_deserialization": True,
            "native_dataset_loader_imported": False,
            "native_dataset_loader_static_inventory": (
                native_loader_inventory
            ),
            "allowed_sample_tensor_loads": sample_inventory,
        },
        "normalization_support": normalization,
        "support_artifact_hashes": top_level_support_hashes,
        "manifest_parse": {
            "dataset_summary_loaded": True,
            "split_manifest_loaded": True,
            "tensor_manifest_loaded": True,
            "campaign_manifest_loaded": True,
            "final_handover_certification_loaded": True,
            "final_dataset_complete_loaded": True,
            "test_tensor_loaded": False,
        },
        "governance": {
            "A_B_train_access": "authorized for later governed stages",
            "A_B_validation_access": (
                "not evaluated here; only two representative allowed "
                "run tensors per tranche were structurally loaded"
            ),
            "A_B_test_access": "sealed",
            "test_tensor_deserialized": False,
            "test_file_bytes_hashed_for_integrity": True,
            "test_filenames_counted_for_inventory": True,
            "threshold_tuning": False,
            "model_training": False,
            "model_evaluation": False,
        },
        "decision": {
            "final_dataset_integrity_preflight_passed": True,
            "guarded_access_boundary_installed": True,
            "native_loader_integration_not_yet_frozen": True,
            "final_test_remains_sealed": True,
            "next_stage": (
                "V5_P3_FINAL_A1_TRAIN_ONLY_UNION_NORMALIZATION_"
                "AND_NATIVE_LOADER_CERTIFICATION"
            ),
        },
        "artifacts": {
            "split_inventory": str(inventory_path),
            "split_inventory_sha256": sha256_file(inventory_path),
            "hash_verification": str(hash_path),
            "hash_verification_sha256": sha256_file(hash_path),
            "loader_preflight": str(loader_path),
            "loader_preflight_sha256": sha256_file(loader_path),
        },
        "provenance": {
            "candidate_guard_sha256": sha256_file(candidate_guard),
            "installed_script_sha256": sha256_file(installed_script),
        },
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(report_path),
        "split_inventory_sha256": sha256_file(inventory_path),
        "hash_verification_sha256": sha256_file(hash_path),
        "loader_preflight_sha256": sha256_file(loader_path),
        "guarded_access_source_sha256": sha256_file(installed_guard),
        "dataset_link": str(dataset_link),
        "physical_root": str(physical_root),
        "internal_SHA256SUMS_sha256": sha256_file(
            support_paths["SHA256SUMS.txt"]
        ),
        "expected_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "test_tensor_deserialized": False,
        "final_test_sealed": True,
    }
    atomic_json(lock_path, lock)
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign_label={CAMPAIGN_LABEL}")
    print("dataset_symlink_target_exact=true")
    print("D70_schema_evidence=true")
    print("train_runs=2400")
    print("train_pairs=1200")
    print("validation_runs=300")
    print("validation_pairs=150")
    print("test_runs=300")
    print("test_pairs=150")
    print("attack_control_pairing=PASS")
    print(
        "internal_sha256_entries_verified="
        f"{hash_verification['entries_verified']}"
    )
    print("guarded_train_listing=PASS")
    print("guarded_validation_listing=PASS")
    print("guarded_test_listing_denied=true")
    print("guarded_test_load_denied_before_deserialization=true")
    print("allowed_sample_tensor_loads=4")
    print("D70_dimension_observed_in_allowed_samples=true")
    print("router16_dimension_observed_in_allowed_samples=true")
    print("native_dataset_loader_imported=false")
    print("test_filenames_counted_for_inventory=true")
    print("test_bytes_hashed_for_integrity=true")
    print("test_tensor_deserialized=false")
    print("model_training=false")
    print("model_evaluation=false")
    print(f"installed_guarded_access={installed_guard}")
    print(
        "installed_guarded_access_sha256="
        f"{sha256_file(installed_guard)}"
    )
    print(
        "next_stage="
        "V5_P3_FINAL_A1_TRAIN_ONLY_UNION_NORMALIZATION_"
        "AND_NATIVE_LOADER_CERTIFICATION"
    )
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
