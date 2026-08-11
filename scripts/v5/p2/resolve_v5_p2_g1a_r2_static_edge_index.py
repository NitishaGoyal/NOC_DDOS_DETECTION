#!/usr/bin/env python3
"""
V5 P2-G1A-R2 Static Edge-Index Resolution

The G1A-R1 run successfully resolved:
- PyTorch Geometric;
- the required pair manifest;
- train/validation dataset construction;
- one train and one validation sample.

R1 stopped only because it assumed edge_index must be present in every sample.
For a fixed 4x4 NoC, the loader may store topology once on the dataset/module
or as a separate file.

R2 resolves that static topology without deleting or overwriting G1A/G1A-R1.

Allowed:
- read G0 and the historical R1 HOLD report;
- reuse the exact R1-resolved pair manifests;
- instantiate P2 train/validation datasets;
- deserialize one train and one validation item;
- inspect dataset/module attributes;
- inspect non-test files inside the P2 data root for edge_index;
- write a resolved static edge_index artifact and contract.

Forbidden:
- enumerate or deserialize the P2 test split;
- load checkpoints;
- access B4/B6 prediction caches;
- train, optimize, tune, select, quantize, generate RTL, or implement the
  Legal NoC decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


STAGE = "V5_P2_G1A_R2_STATIC_EDGE_INDEX_RESOLUTION"
COMPLETE = f"{STAGE}_COMPLETE"

EXPECTED_G0_PROTOCOL_SHA = (
    "22e3d9828edaba9323a0c5206806ea93c149a239b8d9043ad79129de76882def"
)
EXPECTED_R1_HOLD = (
    "V5_P2_G1A_R1_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT_HOLD"
)

EDGE_ATTRIBUTE_NAMES = (
    "edge_index",
    "graph_edge_index",
    "static_edge_index",
    "mesh_edge_index",
    "edges",
)
EDGE_FILE_NAMES = (
    "edge_index.npy",
    "edge_index.npz",
    "edge_index.pt",
    "edge_index.pth",
    "edge_index.json",
    "edges.npy",
    "edges.pt",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def find_dataset_class(module, class_name: str | None):
    candidates = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        try:
            if issubclass(obj, Dataset):
                candidates.append(obj)
        except TypeError:
            pass

    if class_name:
        for cls in candidates:
            if cls.__name__ == class_name:
                return cls
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        f"unable to uniquely select dataset class; "
        f"requested={class_name}, candidates={[c.__name__ for c in candidates]}"
    )


def build_dataset_kwargs(
    cls,
    root: Path,
    split: str,
    pair_manifest: Path,
) -> dict[str, Any]:
    aliases: dict[str, Any] = {
        "root": root,
        "dataset_root": root,
        "data_root": root,
        "data_dir": root,
        "path": root,
        "split": split,
        "pair_manifest": pair_manifest,
        "pair_manifest_path": pair_manifest,
        "window": 32,
        "window_size": 32,
        "stride": 8,
        "active_only": False,
    }
    kwargs: dict[str, Any] = {}
    unresolved = []
    for name, parameter in inspect.signature(cls).parameters.items():
        if name in aliases:
            kwargs[name] = aliases[name]
        elif parameter.default is not inspect._empty:
            continue
        elif name in {"self", "args", "kwargs"}:
            continue
        else:
            unresolved.append(name)
    if unresolved:
        raise TypeError(
            f"unresolved dataset constructor parameters: {unresolved}"
        )
    return kwargs


def as_edge_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None

    if torch.is_tensor(value):
        tensor = value.detach().cpu()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif isinstance(value, (list, tuple)):
        try:
            tensor = torch.as_tensor(value)
        except Exception:
            return None
    else:
        return None

    if tensor.ndim != 2:
        return None

    if tensor.shape[0] == 2:
        edge_index = tensor
    elif tensor.shape[1] == 2:
        edge_index = tensor.t()
    else:
        return None

    if edge_index.numel() == 0:
        return None

    return edge_index.to(dtype=torch.long).contiguous()


def load_edge_file(path: Path) -> torch.Tensor | None:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        return as_edge_tensor(np.load(path, allow_pickle=False))

    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        preferred = (
            "edge_index",
            "edges",
            "arr_0",
        )
        for key in preferred:
            if key in archive:
                resolved = as_edge_tensor(archive[key])
                if resolved is not None:
                    return resolved
        for key in archive.files:
            resolved = as_edge_tensor(archive[key])
            if resolved is not None:
                return resolved
        return None

    if suffix in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(value, dict):
            for key in ("edge_index", "edges"):
                if key in value:
                    resolved = as_edge_tensor(value[key])
                    if resolved is not None:
                        return resolved
        return as_edge_tensor(value)

    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            for key in ("edge_index", "edges"):
                if key in value:
                    resolved = as_edge_tensor(value[key])
                    if resolved is not None:
                        return resolved
        return as_edge_tensor(value)

    return None


def edge_summary(edge_index: torch.Tensor) -> dict[str, Any]:
    edges = edge_index.t().contiguous()
    unique_edges = torch.unique(edges, dim=0)

    return {
        "shape": list(edge_index.shape),
        "dtype": str(edge_index.dtype),
        "minimum_node": int(edge_index.min().item()),
        "maximum_node": int(edge_index.max().item()),
        "directed_edge_count": int(edge_index.shape[1]),
        "unique_directed_edge_count": int(unique_edges.shape[0]),
        "self_loop_count": int(
            (edge_index[0] == edge_index[1]).sum().item()
        ),
        "sha256_int64_bytes": hashlib.sha256(
            edge_index.numpy().astype(np.int64, copy=False).tobytes()
        ).hexdigest(),
    }


def validate_4x4_physical_mesh(
    edge_index: torch.Tensor,
) -> list[str]:
    failures: list[str] = []
    summary = edge_summary(edge_index)

    if summary["shape"] != [2, 48]:
        failures.append(
            f"edge_index shape is {summary['shape']}, expected [2, 48]"
        )
    if summary["minimum_node"] != 0 or summary["maximum_node"] != 15:
        failures.append(
            "edge_index node range is not exactly 0..15"
        )
    if summary["unique_directed_edge_count"] != 48:
        failures.append(
            "edge_index does not contain 48 unique directed edges"
        )
    if summary["self_loop_count"] != 0:
        failures.append("stored physical topology contains self-loops")

    edge_set = {
        (int(src), int(dst))
        for src, dst in edge_index.t().tolist()
    }

    for src, dst in edge_set:
        src_row, src_col = divmod(src, 4)
        dst_row, dst_col = divmod(dst, 4)
        manhattan = abs(src_row - dst_row) + abs(src_col - dst_col)
        if manhattan != 1:
            failures.append(
                f"non-physical mesh edge found: {src}->{dst}"
            )
            break

    missing_reverse = [
        (src, dst)
        for src, dst in sorted(edge_set)
        if (dst, src) not in edge_set
    ]
    if missing_reverse:
        failures.append(
            f"missing reverse directed edges, first={missing_reverse[0]}"
        )

    expected_undirected = set()
    for row in range(4):
        for col in range(4):
            node = row * 4 + col
            if col + 1 < 4:
                expected_undirected.add(
                    tuple(sorted((node, node + 1)))
                )
            if row + 1 < 4:
                expected_undirected.add(
                    tuple(sorted((node, node + 4)))
                )

    actual_undirected = {
        tuple(sorted((src, dst)))
        for src, dst in edge_set
    }
    if actual_undirected != expected_undirected:
        failures.append(
            "edge_index does not match all 24 physical links of the "
            "row-major 4x4 mesh"
        )

    return failures


def inspect_attributes(
    owner_name: str,
    owner: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in EDGE_ATTRIBUTE_NAMES:
        if not hasattr(owner, name):
            continue
        try:
            value = getattr(owner, name)
        except Exception as exc:
            rows.append(
                {
                    "owner": owner_name,
                    "attribute": name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        resolved = as_edge_tensor(value)
        rows.append(
            {
                "owner": owner_name,
                "attribute": name,
                "resolved": resolved is not None,
                "summary": (
                    edge_summary(resolved)
                    if resolved is not None
                    else {
                        "type": type(value).__name__,
                        "repr": repr(value)[:300],
                    }
                ),
                "_tensor": resolved,
            }
        )
    return rows


def find_edge_files(data_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for current, dirs, files in os.walk(data_root):
        current_path = Path(current)
        lowered = "/".join(part.lower() for part in current_path.parts)
        if "test" in lowered:
            dirs[:] = []
            continue
        for filename in files:
            lower = filename.lower()
            if (
                lower in EDGE_FILE_NAMES
                or "edge_index" in lower
            ):
                candidates.append(current_path / filename)
    return sorted(set(candidates), key=lambda path: str(path))


def select_consistent_edge(
    candidates: list[dict[str, Any]],
) -> tuple[torch.Tensor | None, str | None, list[str]]:
    valid = []
    errors = []

    for candidate in candidates:
        tensor = candidate.get("_tensor")
        if tensor is None:
            continue
        mesh_failures = validate_4x4_physical_mesh(tensor)
        if mesh_failures:
            errors.append(
                f"{candidate['source']}: {'; '.join(mesh_failures)}"
            )
            continue
        valid.append(candidate)

    if not valid:
        return None, None, errors

    hashes = {
        edge_summary(candidate["_tensor"])["sha256_int64_bytes"]
        for candidate in valid
    }
    if len(hashes) != 1:
        errors.append(
            "multiple valid topology sources disagree in edge ordering/content"
        )
        return None, None, errors

    selected = valid[0]
    return (
        selected["_tensor"],
        selected["source"],
        errors,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--g0-dir", type=Path, required=True)
    parser.add_argument("--r1-hold-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    g0_dir = args.g0_dir.expanduser().resolve()
    r1_hold_dir = args.r1_hold_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    loader_path = args.loader.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(
            f"STOP: output already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    g0_lock_path = (
        g0_dir
        / "V5_P2_G0_GRAPH_BASELINE_AND_RTL_HANDOFF_PROTOCOL_LOCK.json"
    )
    r1_report_path = (
        r1_hold_dir
        / "V5_P2_G1A_R1_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT.json"
    )
    r1_lock_path = (
        r1_hold_dir
        / "V5_P2_G1A_R1_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT_LOCK.json"
    )
    r1_hold_marker = (
        r1_hold_dir
        / "V5_P2_G1A_R1_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT_HOLD"
    )

    for path in (
        g0_lock_path,
        r1_report_path,
        r1_lock_path,
        r1_hold_marker,
        loader_path,
    ):
        if not path.is_file():
            failures.append(f"missing prerequisite: {path}")
    if not data_root.is_dir():
        failures.append(f"missing data root: {data_root}")

    r1_report: dict[str, Any] = {}
    if not failures:
        g0_lock = json.loads(
            g0_lock_path.read_text(encoding="utf-8")
        )
        r1_report = json.loads(
            r1_report_path.read_text(encoding="utf-8")
        )
        r1_lock = json.loads(
            r1_lock_path.read_text(encoding="utf-8")
        )

        if g0_lock.get("protocol_sha256") != EXPECTED_G0_PROTOCOL_SHA:
            failures.append("G0 protocol SHA changed")
        if (
            r1_hold_marker.read_text(encoding="utf-8").strip()
            != EXPECTED_R1_HOLD
        ):
            failures.append("R1 HOLD marker changed")
        if r1_lock.get("report_sha256") != sha256_file(r1_report_path):
            failures.append("R1 report SHA mismatch")
        if r1_report.get("status") != "HOLD":
            failures.append("R1 report is not HOLD")

    selected = (
        r1_report.get("manifest_discovery", {}).get("selected", {})
        if r1_report
        else {}
    )
    train_manifest_text = selected.get("train_manifest")
    validation_manifest_text = selected.get("validation_manifest")
    selected_class_name = (
        r1_report.get("loader", {}).get("selected_class")
        if r1_report
        else None
    )

    if not train_manifest_text:
        failures.append("R1 report has no selected train manifest")
    if not validation_manifest_text:
        failures.append("R1 report has no selected validation manifest")

    train_dataset = None
    validation_dataset = None
    train_sample = None
    validation_sample = None
    dataset_class = None

    if not failures:
        train_manifest = Path(train_manifest_text).resolve()
        validation_manifest = Path(validation_manifest_text).resolve()

        if not train_manifest.is_file():
            failures.append(
                f"selected train manifest missing: {train_manifest}"
            )
        if not validation_manifest.is_file():
            failures.append(
                "selected validation manifest missing: "
                f"{validation_manifest}"
            )

    if not failures:
        loader_module = import_module_from_path(
            "v5_p2_g1a_r2_loader",
            loader_path,
        )
        dataset_class = find_dataset_class(
            loader_module,
            selected_class_name,
        )

        train_kwargs = build_dataset_kwargs(
            dataset_class,
            data_root,
            "train",
            train_manifest,
        )
        validation_kwargs = build_dataset_kwargs(
            dataset_class,
            data_root,
            "validation",
            validation_manifest,
        )
        train_dataset = dataset_class(**train_kwargs)
        validation_dataset = dataset_class(**validation_kwargs)
        train_sample = train_dataset[0]
        validation_sample = validation_dataset[0]

    topology_candidates: list[dict[str, Any]] = []
    attribute_inspection: list[dict[str, Any]] = []

    if not failures:
        owners = (
            ("train_dataset", train_dataset),
            ("validation_dataset", validation_dataset),
            ("loader_module", loader_module),
        )

        for owner_name, owner in owners:
            rows = inspect_attributes(owner_name, owner)
            for row in rows:
                attribute_inspection.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key != "_tensor"
                    }
                )
                if row.get("_tensor") is not None:
                    topology_candidates.append(
                        {
                            "source": (
                                f"{owner_name}.{row['attribute']}"
                            ),
                            "_tensor": row["_tensor"],
                        }
                    )

        if isinstance(train_sample, dict):
            for key in EDGE_ATTRIBUTE_NAMES:
                if key in train_sample:
                    tensor = as_edge_tensor(train_sample[key])
                    if tensor is not None:
                        topology_candidates.append(
                            {
                                "source": f"train_sample[{key}]",
                                "_tensor": tensor,
                            }
                        )

        if isinstance(validation_sample, dict):
            for key in EDGE_ATTRIBUTE_NAMES:
                if key in validation_sample:
                    tensor = as_edge_tensor(validation_sample[key])
                    if tensor is not None:
                        topology_candidates.append(
                            {
                                "source": f"validation_sample[{key}]",
                                "_tensor": tensor,
                            }
                        )

    file_inspection: list[dict[str, Any]] = []
    if not failures:
        for path in find_edge_files(data_root):
            try:
                tensor = load_edge_file(path)
                file_inspection.append(
                    {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "resolved": tensor is not None,
                        "summary": (
                            edge_summary(tensor)
                            if tensor is not None
                            else None
                        ),
                    }
                )
                if tensor is not None:
                    topology_candidates.append(
                        {
                            "source": f"file:{path}",
                            "_tensor": tensor,
                        }
                    )
            except Exception as exc:
                file_inspection.append(
                    {
                        "path": str(path),
                        "resolved": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    selected_edge = None
    selected_source = None
    candidate_errors: list[str] = []

    if not failures:
        (
            selected_edge,
            selected_source,
            candidate_errors,
        ) = select_consistent_edge(topology_candidates)

        if selected_edge is None:
            failures.append(
                "no validated static 4x4 physical edge_index was resolved"
            )
            failures.extend(candidate_errors)

    edge_artifact_path = (
        output_dir
        / "V5_P2_G1A_R2_RESOLVED_STATIC_EDGE_INDEX.npy"
    )
    edge_contract_path = (
        output_dir
        / "V5_P2_G1A_R2_STATIC_EDGE_INDEX_CONTRACT.json"
    )

    edge_contract: dict[str, Any] = {}
    if selected_edge is not None:
        np.save(
            edge_artifact_path,
            selected_edge.numpy().astype(np.int64, copy=False),
            allow_pickle=False,
        )
        edge_contract_core = {
            "name": "V5_P2_G1A_R2_STATIC_EDGE_INDEX_CONTRACT",
            "source": selected_source,
            "topology": "4x4_2D_MESH",
            "router_numbering": "row_major_0_to_15",
            "physical_self_loops": False,
            "directed": True,
            "bidirectional_physical_links": True,
            "summary": edge_summary(selected_edge),
            "artifact_filename": edge_artifact_path.name,
            "artifact_sha256": sha256_file(edge_artifact_path),
            "runtime_batching_rule": (
                "replicate or offset this fixed edge_index per graph batch; "
                "do not learn or alter topology"
            ),
            "gcn_self_loop_rule": (
                "GCNConv may add its own self-loops according to the frozen "
                "operator contract; the stored physical edge_index has none"
            ),
        }
        edge_contract = {
            **edge_contract_core,
            "contract_sha256": canonical_sha256(edge_contract_core),
        }
        write_json(edge_contract_path, edge_contract)

    report = {
        "stage": STAGE,
        "status": "COMPLETE" if not failures else "HOLD",
        "decision": (
            "FREEZE_STATIC_EDGE_INDEX_AND_AUTHORIZE_G1_SOURCE_ONLY_IMPLEMENTATION"
            if not failures
            else "HOLD_G1_IMPLEMENTATION_PENDING_STATIC_EDGE_INDEX_RESOLUTION"
        ),
        "historical_g1a_hold_preserved": True,
        "historical_g1a_r1_hold_preserved": True,
        "r1_resolved_pair_manifest_reused": True,
        "pair_manifests": {
            "train": train_manifest_text,
            "validation": validation_manifest_text,
        },
        "dataset": {
            "class": (
                dataset_class.__name__
                if dataset_class is not None
                else None
            ),
            "train_length": (
                len(train_dataset)
                if train_dataset is not None
                else None
            ),
            "validation_length": (
                len(validation_dataset)
                if validation_dataset is not None
                else None
            ),
            "train_sample_keys": (
                list(train_sample.keys())
                if isinstance(train_sample, dict)
                else None
            ),
            "validation_sample_keys": (
                list(validation_sample.keys())
                if isinstance(validation_sample, dict)
                else None
            ),
            "edge_index_expected_per_sample": False,
            "edge_index_is_static_graph_input": True,
        },
        "attribute_inspection": attribute_inspection,
        "file_inspection": file_inspection,
        "topology_candidate_sources": [
            candidate["source"]
            for candidate in topology_candidates
        ],
        "selected_topology_source": selected_source,
        "edge_contract": edge_contract,
        "security_boundary": {
            "train_samples_deserialized": (
                1 if train_sample is not None else 0
            ),
            "validation_samples_deserialized": (
                1 if validation_sample is not None else 0
            ),
            "test_directory_enumerated": False,
            "test_tensors_deserialized": False,
            "checkpoint_loaded": False,
            "b4_validation_cache_accessed": False,
            "b6_test_cache_accessed": False,
            "training_performed": False,
            "optimization_steps": 0,
            "threshold_tuning_performed": False,
            "architecture_selected": False,
            "quantization_performed": False,
            "rtl_generated": False,
            "legal_decoder_implemented": False,
        },
        "failures": failures,
        "warnings": warnings,
        "next_stage": (
            "V5_P2_G1_SOURCE_ONLY_GRAPH_BASELINE_IMPLEMENTATION_AND_TRAINING"
            if not failures
            else "V5_P2_G1A_R2A_STATIC_TOPOLOGY_SOURCE_AUDIT"
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    markdown = [
        "# V5 P2-G1A-R2 Static Edge-Index Resolution",
        "",
        f"- Status: **{report['status']}**",
        f"- Dataset class: `{report['dataset']['class']}`",
        f"- Train manifest: `{train_manifest_text}`",
        f"- Validation manifest: `{validation_manifest_text}`",
        f"- Selected topology source: `{selected_source}`",
        "",
        "The P2 loader does not need to duplicate a fixed 4x4 topology in "
        "every sample. R2 resolves the static edge_index from the dataset, "
        "loader module, or a separate data-root artifact and freezes it for "
        "G1.",
        "",
        "No P2 test access, checkpoint loading, training, threshold tuning, "
        "quantization, RTL generation, or Legal NoC decoder implementation "
        "was performed.",
        "",
    ]
    atomic_write(
        output_dir
        / "V5_P2_G1A_R2_STATIC_EDGE_INDEX_RESOLUTION.md",
        "\n".join(markdown),
    )

    lock = {
        "status": COMPLETE if not failures else f"{STAGE}_HOLD",
        "report_sha256": sha256_file(report_path),
        "g0_protocol_sha256": EXPECTED_G0_PROTOCOL_SHA,
        "r1_report_sha256": sha256_file(r1_report_path),
        "historical_holds_preserved": True,
        "edge_artifact_sha256": (
            sha256_file(edge_artifact_path)
            if edge_artifact_path.is_file()
            else None
        ),
        "edge_contract_file_sha256": (
            sha256_file(edge_contract_path)
            if edge_contract_path.is_file()
            else None
        ),
        "edge_contract_sha256": (
            edge_contract.get("contract_sha256")
            if edge_contract
            else None
        ),
        "test_directory_enumerated": False,
        "test_tensors_deserialized": False,
        "checkpoint_loaded": False,
        "b4_validation_cache_accessed": False,
        "b6_test_cache_accessed": False,
        "training_performed": False,
        "optimization_steps": 0,
        "architecture_selected": False,
        "quantization_performed": False,
        "rtl_generated": False,
        "legal_decoder_implemented": False,
        "next_stage": report["next_stage"],
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(
        output_dir
        / "V5_P2_G1A_R2_STATIC_EDGE_INDEX_RESOLUTION_LOCK.json",
        lock,
    )

    if failures:
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print("===== V5 P2-G1A-R2 STATIC EDGE-INDEX RESOLUTION =====")
        print("status: HOLD")
        print("historical_holds_preserved: true")
        print("pair_manifest_reused: true")
        print("topology_candidate_count:", len(topology_candidates))
        print("failure_count:", len(failures))
        print("warning_count:", len(warnings))
        for failure in failures:
            print("FAIL:", failure)
        print(
            "next_stage: "
            "V5_P2_G1A_R2A_STATIC_TOPOLOGY_SOURCE_AUDIT"
        )
        print(f"{STAGE}_HOLD")
        return 1

    atomic_write(
        output_dir / COMPLETE,
        COMPLETE + "\n",
    )

    print("===== V5 P2-G1A-R2 STATIC EDGE-INDEX RESOLUTION =====")
    print("status: COMPLETE")
    print(
        "decision: "
        "FREEZE_STATIC_EDGE_INDEX_AND_AUTHORIZE_G1_SOURCE_ONLY_IMPLEMENTATION"
    )
    print("historical_g1a_hold_preserved: true")
    print("historical_g1a_r1_hold_preserved: true")
    print("pair_manifest_reused: true")
    print("train_pair_manifest:", train_manifest_text)
    print("validation_pair_manifest:", validation_manifest_text)
    print("dataset_class:", dataset_class.__name__)
    print("train_items:", len(train_dataset))
    print("validation_items:", len(validation_dataset))
    print("edge_index_per_sample: false")
    print("static_edge_index_source:", selected_source)
    print(
        "static_edge_index_shape:",
        edge_summary(selected_edge)["shape"],
    )
    print(
        "static_edge_index_directed_edges:",
        edge_summary(selected_edge)["directed_edge_count"],
    )
    print(
        "static_edge_index_artifact_sha256:",
        sha256_file(edge_artifact_path),
    )
    print(
        "static_edge_index_contract_sha256:",
        edge_contract["contract_sha256"],
    )
    print("train_samples_deserialized: 1")
    print("validation_samples_deserialized: 1")
    print("test_directory_enumerated: false")
    print("test_tensors_deserialized: false")
    print("checkpoint_loaded: false")
    print("b4_validation_cache_accessed: false")
    print("b6_test_cache_accessed: false")
    print("training_performed: false")
    print("optimization_steps: 0")
    print("architecture_selected: false")
    print("failure_count: 0")
    print("warning_count:", len(warnings))
    print(
        "next_stage: "
        "V5_P2_G1_SOURCE_ONLY_GRAPH_BASELINE_IMPLEMENTATION_AND_TRAINING"
    )
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
