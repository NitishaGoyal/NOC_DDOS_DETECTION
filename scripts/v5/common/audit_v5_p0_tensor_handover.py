#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import py_compile
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

REQUIRED_TOP_LEVEL = [
    "runs",
    "topology.pt",
    "normalization.pt",
    "feature_schema.json",
    "split_manifest.json",
    "tensor_manifest.json",
    "dataset_summary.json",
    "dataset_loader.py",
    "README.md",
]

REQUIRED_RUN_KEYS = [
    "x",
    "epoch_id",
    "y_attack",
    "y_attacker_count",
    "y_source",
    "y_transit",
    "y_victim",
    "y_attack_path",
    "role_mask",
    "case_id",
    "pair_id",
    "run_id",
    "split",
    "mode",
]

EXPECTED_SPLITS = {"train": 32, "validation": 8, "test": 8}
EXPECTED_MODES = {"control": 24, "attack": 24}

FORBIDDEN_EXACT_NAMES = {
    "snapshot_tick", "tick", "epoch", "epoch_id", "router", "router_id",
    "case", "case_id", "pair", "pair_id", "run", "run_id", "seed",
    "mode", "split", "attack_active", "attacker_count",
    "active_attack_count", "source_router", "victim_router", "packet_id",
    "flit_id", "address", "candidate_getx_count", "role_mask", "y_attack",
    "y_attacker_count", "y_source", "y_transit", "y_victim",
    "y_attack_path",
}

FORBIDDEN_SUBSTRINGS = [
    "pair_id", "run_id", "case_id", "packet_id", "flit_id",
    "candidate_getx", "active_attack_count", "attack_active", "y_source",
    "y_transit", "y_victim", "y_attack_path", "role_mask",
    "victim_router", "source_router",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        write_text_atomic(path, "")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def scalar_text(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"expected scalar tensor, got shape {tuple(value.shape)}")
        value = value.detach().cpu().item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_leaves(value: Any, prefix: str = "") -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value.detach().cpu()
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from tensor_leaves(child, child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            yield from tensor_leaves(child, child_prefix)


def feature_name(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("name", "feature_name", "column", "original_name"):
            if key in item:
                return str(item[key])
    return ""


def find_feature_list(value: Any) -> list[Any] | None:
    candidates: list[list[Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, list):
            if len(node) == 81:
                names = [feature_name(item) for item in node]
                if all(names):
                    candidates.append(node)
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            for child in node.values():
                visit(child)

    visit(value)
    if not candidates:
        return None

    signatures: dict[tuple[str, ...], list[Any]] = {}
    for candidate in candidates:
        signatures.setdefault(
            tuple(feature_name(item) for item in candidate),
            candidate,
        )
    return next(iter(signatures.values())) if len(signatures) == 1 else None


def check_binary(tensor: torch.Tensor, name: str, failures: list[str]) -> None:
    unique = torch.unique(tensor.detach().cpu())
    bad = [value.item() for value in unique if value.item() not in (0, 1)]
    if bad:
        failures.append(f"{name} contains non-binary values: {bad[:20]}")


def verify_sha256sums(root: Path, sums_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    pattern = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+)$")

    for line_number, raw in enumerate(
        sums_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line:
            continue
        match = pattern.match(line)
        if not match:
            failures.append(f"malformed SHA256SUMS line {line_number}: {raw!r}")
            continue

        expected, relative = match.groups()
        path = root / relative
        if not path.is_file():
            failures.append(f"SHA256SUMS file missing: {relative}")
            rows.append({
                "path": relative,
                "expected_sha256": expected.lower(),
                "actual_sha256": "",
                "matches": False,
            })
            continue

        actual = sha256_file(path)
        matches = actual == expected.lower()
        if not matches:
            failures.append(
                f"SHA256 mismatch: {relative}; "
                f"expected={expected.lower()} actual={actual}"
            )
        rows.append({
            "path": relative,
            "expected_sha256": expected.lower(),
            "actual_sha256": actual,
            "matches": matches,
        })
    return rows, failures


def audit_run(path: Path, directory_split: str) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    row: dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "directory_split": directory_split,
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }

    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        row["load_error"] = repr(exc)
        return row, [f"failed to load {path.name}: {exc!r}"]

    if not isinstance(value, dict):
        row["object_type"] = type(value).__name__
        return row, [f"{path.name} does not contain a dictionary"]

    missing = [key for key in REQUIRED_RUN_KEYS if key not in value]
    if missing:
        row["missing_keys"] = ",".join(missing)
        return row, [f"{path.name} missing required keys: {missing}"]

    try:
        case_id = scalar_text(value["case_id"])
        pair_id = scalar_text(value["pair_id"])
        run_id = scalar_text(value["run_id"])
        split = scalar_text(value["split"]).lower()
        mode = scalar_text(value["mode"]).lower()
    except Exception as exc:
        return row, [f"{path.name} metadata decoding failed: {exc!r}"]

    row.update({
        "case_id": case_id,
        "pair_id": pair_id,
        "run_id": run_id,
        "split": split,
        "mode": mode,
    })

    if split != directory_split:
        failures.append(
            f"{path.name}: metadata split={split!r} "
            f"but directory split={directory_split!r}"
        )
    if split not in EXPECTED_SPLITS:
        failures.append(f"{path.name}: unknown split={split!r}")
    if mode not in EXPECTED_MODES:
        failures.append(f"{path.name}: unknown mode={mode!r}")

    tensor_keys = [
        "x", "epoch_id", "y_attack", "y_attacker_count", "y_source",
        "y_transit", "y_victim", "y_attack_path", "role_mask",
    ]
    tensors: dict[str, torch.Tensor] = {}
    for key in tensor_keys:
        tensor = value[key]
        if not isinstance(tensor, torch.Tensor):
            return row, failures + [f"{path.name}: {key} is not a tensor"]
        tensors[key] = tensor.detach().cpu()

    x = tensors["x"]
    if x.ndim != 3:
        return row, failures + [
            f"{path.name}: x shape={tuple(x.shape)}, expected rank 3"
        ]

    T = int(x.shape[0])
    expected_shapes = {
        "x": (T, 16, 81),
        "epoch_id": (T,),
        "y_attack": (T,),
        "y_attacker_count": (T,),
        "y_source": (T, 16),
        "y_transit": (T, 16),
        "y_victim": (T, 16),
        "y_attack_path": (T, 16),
        "role_mask": (T, 16),
    }

    for key, expected in expected_shapes.items():
        actual = tuple(tensors[key].shape)
        if actual != expected:
            failures.append(
                f"{path.name}: {key} shape={actual}, expected={expected}"
            )

    if T <= 32:
        failures.append(f"{path.name}: T={T}, expected T>32")

    for key, tensor in tensors.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            failures.append(f"{path.name}: {key} contains non-finite values")

    epoch_id = tensors["epoch_id"]
    if epoch_id.numel() > 1 and not bool(torch.all(epoch_id[1:] > epoch_id[:-1])):
        failures.append(f"{path.name}: epoch_id is not strictly increasing")

    for key in ("y_attack", "y_source", "y_transit", "y_victim", "y_attack_path"):
        check_binary(tensors[key], f"{path.name}:{key}", failures)

    y_attack = tensors["y_attack"].bool()
    y_count = tensors["y_attacker_count"].long()
    y_source = tensors["y_source"].bool()
    y_transit = tensors["y_transit"].bool()
    y_victim = tensors["y_victim"].bool()
    y_path = tensors["y_attack_path"].bool()
    role_mask = tensors["role_mask"].long()

    if y_count.numel():
        if int(y_count.min().item()) < 0:
            failures.append(f"{path.name}: negative attacker count")
        if int(y_count.max().item()) > 2:
            failures.append(
                f"{path.name}: attacker count exceeds P0 maximum 2: "
                f"{int(y_count.max().item())}"
            )

    expected_path = y_source | y_transit | y_victim
    if not torch.equal(y_path, expected_path):
        mismatch = int((y_path != expected_path).sum().item())
        failures.append(
            f"{path.name}: y_attack_path union mismatch at {mismatch} entries"
        )

    expected_role = (
        y_source.long()
        + 2 * y_transit.long()
        + 4 * y_victim.long()
    )
    if not torch.equal(role_mask, expected_role):
        mismatch = int((role_mask != expected_role).sum().item())
        failures.append(
            f"{path.name}: role_mask mismatch at {mismatch} entries"
        )

    inactive = ~y_attack
    active = y_attack

    if bool(inactive.any()):
        inactive_values = {
            "y_attacker_count": y_count[inactive],
            "y_source": y_source[inactive],
            "y_transit": y_transit[inactive],
            "y_victim": y_victim[inactive],
            "y_attack_path": y_path[inactive],
            "role_mask": role_mask[inactive],
        }
        for key, tensor in inactive_values.items():
            if bool((tensor != 0).any()):
                failures.append(
                    f"{path.name}: inactive epochs contain nonzero {key}"
                )

    if bool(active.any()):
        active_counts = y_count[active]
        if bool((active_counts < 1).any()):
            failures.append(
                f"{path.name}: active epoch with attacker_count<1"
            )
        source_counts = y_source[active].long().sum(dim=1)
        if not torch.equal(source_counts, active_counts):
            mismatch = int((source_counts != active_counts).sum().item())
            failures.append(
                f"{path.name}: active source count disagrees with "
                f"attacker count in {mismatch} epochs"
            )

    if mode == "control":
        for key in (
            "y_attack", "y_attacker_count", "y_source", "y_transit",
            "y_victim", "y_attack_path", "role_mask",
        ):
            if bool((tensors[key] != 0).any()):
                failures.append(
                    f"{path.name}: control run contains nonzero {key}"
                )

    source_victim = y_source & y_victim
    invalid_overlap = source_victim & (y_transit | (role_mask != 5))
    if bool(invalid_overlap.any()):
        failures.append(
            f"{path.name}: source=victim overlap violates role_mask=5 semantics"
        )

    row.update({
        "T": T,
        "num_nodes": int(x.shape[1]),
        "num_features": int(x.shape[2]),
        "x_dtype": str(x.dtype),
        "epoch_dtype": str(epoch_id.dtype),
        "y_attack_dtype": str(tensors["y_attack"].dtype),
        "y_count_dtype": str(tensors["y_attacker_count"].dtype),
        "node_label_dtype": str(tensors["y_source"].dtype),
        "role_mask_dtype": str(tensors["role_mask"].dtype),
        "active_epoch_count": int(active.sum().item()),
        "inactive_epoch_count": int(inactive.sum().item()),
        "maximum_attacker_count": (
            int(y_count.max().item()) if y_count.numel() else 0
        ),
        "source_victim_overlap_entries": int(source_victim.sum().item()),
        "x_min": float(x.min().item()),
        "x_max": float(x.max().item()),
        "x_mean": float(x.mean().item()),
        "x_std": float(x.std(unbiased=False).item()),
    })
    return row, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-sha256-file", type=Path)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not root.is_dir():
        print(f"STOP: dataset root missing: {root}", file=sys.stderr)
        return 2
    if output_dir.exists():
        print(
            f"STOP: output directory already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    report: dict[str, Any] = {
        "stage": "V5_P0_A1_TENSOR_HANDOVER_AUDIT",
        "dataset_root": str(root),
        "audit_boundary": {
            "model_training_performed": False,
            "model_inference_performed": False,
            "threshold_selection_performed": False,
            "feature_selection_performed": False,
            "checkpoint_selection_performed": False,
            "test_performance_evaluated": False,
            "raw_gem5_trace_accessed": False,
        },
    }

    external_archive: dict[str, Any] = {
        "requested": bool(args.archive or args.archive_sha256_file),
        "verified": False,
    }
    if bool(args.archive) != bool(args.archive_sha256_file):
        failures.append(
            "--archive and --archive-sha256-file must be supplied together"
        )
    elif args.archive and args.archive_sha256_file:
        archive = args.archive.expanduser().resolve()
        checksum_file = args.archive_sha256_file.expanduser().resolve()
        if not archive.is_file():
            failures.append(f"archive missing: {archive}")
        if not checksum_file.is_file():
            failures.append(
                f"archive checksum file missing: {checksum_file}"
            )
        if archive.is_file() and checksum_file.is_file():
            tokens = checksum_file.read_text(
                encoding="utf-8"
            ).strip().split()
            if not tokens:
                failures.append("archive checksum file is empty")
            else:
                expected = tokens[0].lower()
                actual = sha256_file(archive)
                external_archive.update({
                    "archive": str(archive),
                    "checksum_file": str(checksum_file),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "verified": expected == actual,
                })
                if expected != actual:
                    failures.append("external archive SHA-256 mismatch")
    report["external_archive"] = external_archive

    top_level_rows: list[dict[str, Any]] = []
    for name in REQUIRED_TOP_LEVEL:
        path = root / name
        expected_type = "directory" if name == "runs" else "file"
        correct_type = (
            path.is_dir()
            if expected_type == "directory"
            else path.is_file()
        )
        top_level_rows.append({
            "name": name,
            "expected_type": expected_type,
            "exists": path.exists(),
            "correct_type": correct_type,
        })
        if not path.exists() or not correct_type:
            failures.append(
                f"required {expected_type} missing: {path}"
            )

    for name in (
        "feature_schema.json",
        "split_manifest.json",
        "tensor_manifest.json",
        "dataset_summary.json",
    ):
        path = root / name
        if path.is_file():
            try:
                load_json(path)
            except Exception as exc:
                failures.append(f"{name} is invalid JSON: {exc!r}")

    loader_path = root / "dataset_loader.py"
    if loader_path.is_file():
        try:
            py_compile.compile(str(loader_path), doraise=True)
        except Exception as exc:
            failures.append(
                f"dataset_loader.py does not compile: {exc!r}"
            )

    checksum_rows: list[dict[str, Any]] = []
    sums_path = root / "SHA256SUMS.txt"
    if sums_path.is_file():
        checksum_rows, checksum_failures = verify_sha256sums(
            root, sums_path
        )
        failures.extend(checksum_failures)
    else:
        warnings.append(
            "SHA256SUMS.txt is absent; internal package hashes "
            "were not verified"
        )

    feature_rows: list[dict[str, Any]] = []
    feature_names: list[str] = []
    schema_path = root / "feature_schema.json"
    if schema_path.is_file():
        schema = load_json(schema_path)
        feature_list = find_feature_list(schema)
        if feature_list is None:
            failures.append(
                "could not resolve a unique 81-entry feature list "
                "from feature_schema.json"
            )
        else:
            for index, item in enumerate(feature_list):
                name = feature_name(item)
                feature_names.append(name)
                feature_rows.append({
                    "index": index,
                    "name": name,
                    "definition_json": (
                        json.dumps(item, sort_keys=True)
                        if not isinstance(item, str)
                        else json.dumps({"name": item})
                    ),
                })

            normalized = [name.strip().lower() for name in feature_names]
            duplicates = sorted(
                name
                for name, count in Counter(normalized).items()
                if count > 1
            )
            if duplicates:
                failures.append(
                    f"duplicate feature names: {duplicates}"
                )

            exact_forbidden = sorted(
                set(normalized) & FORBIDDEN_EXACT_NAMES
            )
            if exact_forbidden:
                failures.append(
                    "forbidden metadata/truth features present: "
                    f"{exact_forbidden}"
                )

            suspicious = sorted({
                name
                for name in normalized
                if any(token in name for token in FORBIDDEN_SUBSTRINGS)
            })
            if suspicious:
                failures.append(
                    "suspicious metadata/truth feature names present: "
                    f"{suspicious}"
                )

    normalization_rows: list[dict[str, Any]] = []
    normalization_path = root / "normalization.pt"
    if normalization_path.is_file():
        try:
            normalization = torch.load(
                normalization_path,
                map_location="cpu",
                weights_only=False,
            )
            leaves = list(tensor_leaves(normalization))
            if not leaves:
                warnings.append(
                    "normalization.pt has no tensor leaves"
                )
            for key, tensor in leaves:
                finite = (
                    bool(torch.isfinite(tensor).all())
                    if tensor.is_floating_point()
                    else True
                )
                normalization_rows.append({
                    "key": key,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": int(tensor.numel()),
                    "finite": finite,
                    "min": (
                        float(tensor.min().item())
                        if tensor.numel() and tensor.is_floating_point()
                        else ""
                    ),
                    "max": (
                        float(tensor.max().item())
                        if tensor.numel() and tensor.is_floating_point()
                        else ""
                    ),
                })
                if not finite:
                    failures.append(
                        f"normalization tensor {key!r} "
                        "contains non-finite values"
                    )
        except Exception as exc:
            failures.append(
                f"failed to load normalization.pt: {exc!r}"
            )

    topology_report: dict[str, Any] = {}
    topology_path = root / "topology.pt"
    if topology_path.is_file():
        try:
            topology = torch.load(
                topology_path,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(topology, dict):
                failures.append(
                    "topology.pt is not a dictionary"
                )
            else:
                edge_index = topology.get("edge_index")
                if not isinstance(edge_index, torch.Tensor):
                    failures.append(
                        "topology.pt missing tensor edge_index"
                    )
                else:
                    edge_index = edge_index.detach().cpu().long()
                    topology_report["edge_index_shape"] = list(
                        edge_index.shape
                    )
                    if tuple(edge_index.shape) != (2, 48):
                        failures.append(
                            "edge_index shape="
                            f"{tuple(edge_index.shape)}, expected=(2,48)"
                        )

                    edges = [
                        (int(src), int(dst))
                        for src, dst in edge_index.t().tolist()
                    ]
                    edge_set = set(edges)
                    duplicate_count = len(edges) - len(edge_set)
                    self_loop_count = sum(
                        src == dst for src, dst in edges
                    )
                    missing_reverse = [
                        (src, dst)
                        for src, dst in edges
                        if (dst, src) not in edge_set
                    ]

                    if edges:
                        if min(min(edge) for edge in edges) < 0:
                            failures.append(
                                "edge_index contains negative node IDs"
                            )
                        if max(max(edge) for edge in edges) > 15:
                            failures.append(
                                "edge_index contains node IDs greater than 15"
                            )
                    if duplicate_count:
                        failures.append(
                            f"topology has {duplicate_count} "
                            "duplicate directed edges"
                        )
                    if self_loop_count:
                        failures.append(
                            f"topology has {self_loop_count} "
                            "stored self-loops"
                        )
                    if missing_reverse:
                        failures.append(
                            f"topology has {len(missing_reverse)} "
                            "edges without reverse pairs"
                        )

                    degrees = Counter(src for src, _ in edges)
                    degree_multiset = sorted(
                        degrees.get(node, 0)
                        for node in range(16)
                    )
                    expected_multiset = sorted(
                        [2] * 4 + [3] * 8 + [4] * 4
                    )
                    if degree_multiset != expected_multiset:
                        failures.append(
                            "incorrect 4x4 mesh degree multiset: "
                            f"{degree_multiset}"
                        )

                    topology_report.update({
                        "directed_edge_count": len(edges),
                        "duplicate_directed_edge_count": duplicate_count,
                        "self_loop_count": self_loop_count,
                        "missing_reverse_count": len(missing_reverse),
                        "out_degrees": {
                            str(node): degrees.get(node, 0)
                            for node in range(16)
                        },
                    })

                raw_num_nodes = topology.get("num_nodes")
                try:
                    num_nodes = (
                        int(raw_num_nodes.item())
                        if isinstance(raw_num_nodes, torch.Tensor)
                        else int(raw_num_nodes)
                    )
                except Exception:
                    num_nodes = None
                topology_report["num_nodes"] = num_nodes
                if num_nodes != 16:
                    failures.append(
                        f"topology num_nodes={num_nodes}, expected 16"
                    )

                coordinates = topology.get("router_coordinates")
                if coordinates is not None:
                    if not isinstance(coordinates, torch.Tensor):
                        failures.append(
                            "router_coordinates is not a tensor"
                        )
                    elif tuple(coordinates.shape) != (16, 2):
                        failures.append(
                            "router_coordinates shape="
                            f"{tuple(coordinates.shape)}, expected=(16,2)"
                        )
                    else:
                        topology_report[
                            "router_coordinates_shape"
                        ] = [16, 2]
        except Exception as exc:
            failures.append(
                f"failed to audit topology.pt: {exc!r}"
            )

    run_rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    pair_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_ids: list[str] = []

    runs_root = root / "runs"
    if runs_root.is_dir():
        for split in EXPECTED_SPLITS:
            split_dir = runs_root / split
            if not split_dir.is_dir():
                failures.append(
                    f"missing split directory: {split_dir}"
                )
                continue

            for path in sorted(split_dir.glob("*.pt")):
                row, run_failures = audit_run(path, split)
                failures.extend(run_failures)
                run_rows.append(row)

                if "split" in row:
                    split_counts[row["split"]] += 1
                if "mode" in row:
                    mode_counts[row["mode"]] += 1
                if "pair_id" in row:
                    pair_members[row["pair_id"]].append(row)
                if "run_id" in row:
                    run_ids.append(row["run_id"])

    if len(run_rows) != 48:
        failures.append(
            f"run tensor count={len(run_rows)}, expected 48"
        )
    if dict(split_counts) != EXPECTED_SPLITS:
        failures.append(
            f"split counts={dict(split_counts)}, "
            f"expected={EXPECTED_SPLITS}"
        )
    if dict(mode_counts) != EXPECTED_MODES:
        failures.append(
            f"mode counts={dict(mode_counts)}, "
            f"expected={EXPECTED_MODES}"
        )

    duplicate_run_ids = sorted(
        run_id
        for run_id, count in Counter(run_ids).items()
        if count > 1
    )
    if duplicate_run_ids:
        failures.append(
            f"duplicate run IDs: {duplicate_run_ids}"
        )

    if len(pair_members) != 24:
        failures.append(
            f"unique pair count={len(pair_members)}, expected 24"
        )

    pair_rows: list[dict[str, Any]] = []
    for pair_id, members in sorted(pair_members.items()):
        modes = sorted(
            member.get("mode", "")
            for member in members
        )
        splits = sorted(set(
            member.get("split", "")
            for member in members
        ))
        cases = sorted(set(
            member.get("case_id", "")
            for member in members
        ))
        valid = (
            len(members) == 2
            and modes == ["attack", "control"]
            and len(splits) == 1
            and len(cases) == 1
        )
        if not valid:
            failures.append(
                f"pair integrity failure: {pair_id}; "
                f"members={len(members)} modes={modes} "
                f"splits={splits} cases={cases}"
            )
        pair_rows.append({
            "pair_id": pair_id,
            "member_count": len(members),
            "modes": ",".join(modes),
            "splits": ",".join(splits),
            "case_ids": ",".join(cases),
            "valid": valid,
        })

    local_overlap_run_count = sum(
        int(row.get("source_victim_overlap_entries", 0)) > 0
        for row in run_rows
    )
    dual_attacker_run_count = sum(
        int(row.get("maximum_attacker_count", 0)) == 2
        for row in run_rows
    )

    if local_overlap_run_count == 0:
        failures.append(
            "no source=victim overlap run found"
        )
    if dual_attacker_run_count == 0:
        failures.append(
            "no dual-attacker run found"
        )

    report.update({
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "warnings": warnings,
        "top_level_inventory": top_level_rows,
        "internal_checksum_file_present": sums_path.is_file(),
        "internal_checksum_entry_count": len(checksum_rows),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "normalization_tensor_count": len(normalization_rows),
        "topology": topology_report,
        "run_count": len(run_rows),
        "split_counts": dict(split_counts),
        "mode_counts": dict(mode_counts),
        "pair_count": len(pair_members),
        "local_overlap_run_count": local_overlap_run_count,
        "dual_attacker_run_count": dual_attacker_run_count,
        "training_authorized": not failures,
        "next_stage": (
            "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT"
            if not failures
            else "HOLD_FIX_TENSOR_HANDOVER"
        ),
    })

    write_csv(
        output_dir / "top_level_inventory.csv",
        top_level_rows,
    )
    write_csv(
        output_dir / "internal_checksum_inventory.csv",
        checksum_rows,
    )
    write_csv(
        output_dir / "feature_inventory.csv",
        feature_rows,
    )
    write_csv(
        output_dir / "normalization_inventory.csv",
        normalization_rows,
    )
    write_csv(
        output_dir / "tensor_inventory.csv",
        run_rows,
    )
    write_csv(
        output_dir / "pair_inventory.csv",
        pair_rows,
    )

    report_path = (
        output_dir / "tensor_handover_audit.json"
    )
    write_json(report_path, report)

    lock = {
        "status": (
            "V5_P0_A1_TENSOR_HANDOVER_AUDIT_PASS"
            if not failures
            else "V5_P0_A1_TENSOR_HANDOVER_AUDIT_HOLD"
        ),
        "dataset_root": str(root),
        "report_sha256": sha256_file(report_path),
        "audit_script_sha256": sha256_file(Path(__file__)),
        "run_count": len(run_rows),
        "pair_count": len(pair_members),
        "split_counts": dict(split_counts),
        "mode_counts": dict(mode_counts),
        "model_training_performed": False,
        "model_inference_performed": False,
        "test_performance_evaluated": False,
        "training_authorized": not failures,
        "next_stage": report["next_stage"],
    }
    write_json(
        output_dir / "V5_P0_A1_TENSOR_HANDOVER_LOCK.json",
        lock,
    )

    print(
        "===== V5 P0-A1 TENSOR HANDOVER AUDIT ====="
    )
    print(f"dataset_root: {root}")
    print(f"run_count: {len(run_rows)}")
    print(f"split_counts: {dict(split_counts)}")
    print(f"mode_counts: {dict(mode_counts)}")
    print(f"pair_count: {len(pair_members)}")
    print(f"feature_count: {len(feature_names)}")
    print(
        "local_overlap_run_count: "
        f"{local_overlap_run_count}"
    )
    print(
        "dual_attacker_run_count: "
        f"{dual_attacker_run_count}"
    )
    print(f"failure_count: {len(failures)}")
    print(f"warning_count: {len(warnings)}")

    if warnings:
        print("\n===== WARNINGS =====")
        for warning in warnings:
            print(f"WARNING: {warning}")

    if failures:
        print("\n===== FAILURES =====", file=sys.stderr)
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        write_text_atomic(
            output_dir
            / "V5_P0_A1_TENSOR_HANDOVER_AUDIT_HOLD",
            "V5_P0_A1_TENSOR_HANDOVER_AUDIT_HOLD\n",
        )
        print("training_authorized: false")
        print("next_stage: HOLD_FIX_TENSOR_HANDOVER")
        print(
            "V5_P0_A1_TENSOR_HANDOVER_AUDIT_HOLD"
        )
        return 1

    write_text_atomic(
        output_dir
        / "V5_P0_A1_TENSOR_HANDOVER_AUDIT_PASS",
        "V5_P0_A1_TENSOR_HANDOVER_AUDIT_PASS\n",
    )
    print("training_authorized: true")
    print(
        "next_stage: "
        "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT"
    )
    print("V5_P0_A1_TENSOR_HANDOVER_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
