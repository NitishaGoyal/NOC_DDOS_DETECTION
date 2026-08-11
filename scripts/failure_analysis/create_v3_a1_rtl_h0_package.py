#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

EXPECTED_SOURCE = "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b"
EXPECTED_CHECKPOINT = "c29cdd2303669a61454ed118ef0be4b8809a635b12c0c42806f8b6992340e0ef"
EXPECTED_PARAMS = 882
EXPECTED_X = (233803, 16, 8, 24)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_module(path: Path):
    spec = importlib.util.spec_from_file_location("v3_a1_handoff", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "model_state"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value and all(
                isinstance(v, torch.Tensor) for v in value.values()
            ):
                return value, key
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint, "checkpoint_root"
    raise RuntimeError("No tensor state_dict found in checkpoint")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()

    source = args.source.resolve()
    model_dir = args.model_dir.resolve()
    data_dir = args.data_dir.resolve()
    out = args.output_dir.resolve()

    checkpoint_path = model_dir / "best_model.pt"
    summary_path = model_dir / "summary.json"
    history_path = model_dir / "history.json"
    splits_path = model_dir / "splits.npz"
    x_path = data_dir / "x.npy"
    edge_path = data_dir / "edge_index.npy"
    features_path = data_dir / "feature_cols.npy"

    required = [
        source, checkpoint_path, summary_path, splits_path,
        x_path, edge_path, features_path
    ]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"STOP: missing {path}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"STOP: output directory is non-empty: {out}")

    if sha(source) != EXPECTED_SOURCE:
        raise SystemExit("STOP: frozen A1 source hash mismatch")
    if sha(checkpoint_path) != EXPECTED_CHECKPOINT:
        raise SystemExit("STOP: frozen A1 checkpoint hash mismatch")

    x = np.load(x_path, mmap_mode="r")
    if tuple(x.shape) != EXPECTED_X:
        raise SystemExit(f"STOP: unexpected x shape {x.shape}")

    edge_index = np.load(edge_path).astype(np.int64)
    feature_cols = np.load(features_path, allow_pickle=True)

    module = import_module(source)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, state_location = state_dict_from_checkpoint(checkpoint)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

    model = module.TemporalGCN(
        input_features=24,
        temporal_dim=int(saved_args.get("temporal_dim", 8)),
        gcn_hidden=int(saved_args.get("gcn_hidden", 16)),
        gcn_out=int(saved_args.get("gcn_out", 8)),
    )
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise SystemExit("STOP: checkpoint does not strictly load")

    params = sum(p.numel() for p in model.parameters())
    if params != EXPECTED_PARAMS:
        raise SystemExit(f"STOP: expected 882 parameters, found {params}")

    adjacency = module.build_normalized_adjacency(edge_index, 16)
    adjacency = adjacency.detach().cpu().numpy().astype(np.float32)
    if int(np.count_nonzero(adjacency)) != 64:
        raise SystemExit("STOP: expected 64 nonzero adjacency entries")

    dirs = {
        "docs": out / "docs",
        "model": out / "model",
        "features": out / "features",
        "adjacency": out / "adjacency",
        "weights": out / "weights" / "fp32",
        "golden": out / "golden_vectors" / "PENDING_H1",
        "verification": out / "verification",
        "quant": out / "quantization",
        "manifest": out / "manifests",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    copies = [
        (source, dirs["model"] / "train_temporal_gcn_v3.py"),
        (checkpoint_path, dirs["model"] / "best_model.pt"),
        (summary_path, dirs["model"] / "summary.json"),
        (splits_path, dirs["model"] / "splits.npz"),
        (edge_path, dirs["adjacency"] / "edge_index.npy"),
        (features_path, dirs["features"] / "feature_cols.npy"),
    ]
    if history_path.is_file():
        copies.append((history_path, dirs["model"] / "history.json"))
    for src, dst in copies:
        shutil.copy2(src, dst)

    features = [
        v.decode() if isinstance(v, bytes) else str(v)
        for v in np.asarray(feature_cols).reshape(-1)
    ]
    (dirs["features"] / "feature_order.json").write_text(
        json.dumps(
            {
                "feature_count": len(features),
                "index_to_name": {str(i): name for i, name in enumerate(features)},
                "status": "Frozen for V3 bring-up; semantics may change in V3.1/V4",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    np.save(dirs["adjacency"] / "normalized_adjacency_fp32.npy", adjacency)
    np.savetxt(
        dirs["adjacency"] / "normalized_adjacency_fp32.csv",
        adjacency, delimiter=",", fmt="%.9g"
    )
    sparse = []
    for dst in range(16):
        for src in range(16):
            value = float(adjacency[dst, src])
            if value != 0:
                sparse.append({
                    "destination_router": dst,
                    "source_router": src,
                    "coefficient_fp32": value,
                })
    write_csv(dirs["adjacency"] / "normalized_adjacency_sparse.csv", sparse)

    arrays = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
    np.savez(dirs["weights"] / "state_dict_fp32.npz", **arrays)
    weight_rows = []
    for name, array in arrays.items():
        weight_rows.append({
            "name": name,
            "shape": "x".join(map(str, array.shape)),
            "dtype": str(array.dtype),
            "numel": int(array.size),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
            "status": "PROVISIONAL_V3_BRINGUP_WEIGHT",
        })
    write_csv(dirs["weights"] / "weight_manifest.csv", weight_rows)

    module_rows = []
    for name, layer in model.named_modules():
        if not name:
            continue
        row = {"name": name, "class": layer.__class__.__name__}
        for attr in (
            "in_channels", "out_channels", "kernel_size", "stride",
            "padding", "dilation", "groups", "in_features", "out_features"
        ):
            if hasattr(layer, attr):
                value = getattr(layer, attr)
                row[attr] = str(value)
        if hasattr(layer, "bias"):
            row["bias"] = getattr(layer, "bias") is not None
        row["direct_parameter_count"] = sum(
            p.numel() for p in layer.parameters(recurse=False)
        )
        module_rows.append(row)
    write_csv(dirs["docs"] / "module_manifest.csv", module_rows)
    (dirs["docs"] / "model_repr.txt").write_text(repr(model) + "\n")

    architecture = {
        "status": "V3_A1_RTL_BRINGUP_ONLY",
        "software_tensor_order": ["batch", "router", "epoch", "feature"],
        "input_shape": ["B", 16, 8, 24],
        "elements_per_window": 3072,
        "storage_bytes": {"fp32": 12288, "fp16_or_int16": 6144, "int8": 3072},
        "parameter_count": params,
        "source_sha256": sha(source),
        "checkpoint_sha256": sha(checkpoint_path),
        "checkpoint_state_location": state_location,
        "saved_args": saved_args,
        "adjacency_nonzero_entries": int(np.count_nonzero(adjacency)),
        "adjacency_weight_sum": float(adjacency.sum()),
        "module_manifest": module_rows,
        "not_final": [
            "weights", "thresholds", "normalization", "quantization",
            "ROM contents", "publication golden vectors", "accuracy claims"
        ],
    }
    (dirs["docs"] / "architecture_contract.json").write_text(
        json.dumps(architecture, indent=2, default=str) + "\n"
    )

    (out / "README.md").write_text(
        "# V3-A1 RTL Bring-Up Package\n\n"
        "**NOT FINAL WEIGHTS — FOR RTL/FRONT-END BRING-UP ONLY**\n\n"
        "Use for datapath, interface, buffering, FP32 verification, and early "
        "quantization exploration. V4 must replace weights, thresholds, final "
        "golden vectors, quantization assets, and reported metrics.\n"
    )
    (dirs["docs"] / "OPEN_ITEMS_BEFORE_H1.md").write_text(
        "# Open items before golden-vector release\n\n"
        "1. Confirm Conv1D kernel/padding/stride/dilation/bias/activation.\n"
        "2. Confirm exact GCN arithmetic order from frozen source.\n"
        "3. Export threshold provenance.\n"
        "4. Export per-feature normalization equations/constants.\n"
        "5. Define missing/invalid observation semantics.\n"
        "6. Generate 20–32 balanced validation-only layer traces.\n"
        "7. Define FP32 and fixed-point tolerances.\n"
        "8. Approve stream order, mask sideband, and register map.\n"
    )
    (dirs["golden"] / "README.md").write_text(
        "H1 must generate validation-only layer-by-layer golden vectors.\n"
    )
    (dirs["quant"] / "README.md").write_text(
        "Exploratory only: FP32 → FP16/BF16 → INT16 → mixed INT8/INT16. "
        "Do not freeze formats or ROM contents from V3.\n"
    )

    artifact_rows = []
    for path in sorted(out.rglob("*")):
        if path.is_file():
            artifact_rows.append({
                "relative_path": str(path.relative_to(out)),
                "size_bytes": path.stat().st_size,
                "sha256": sha(path),
            })
    hashes_path = dirs["manifest"] / "artifact_hashes.csv"
    write_csv(hashes_path, artifact_rows)

    release = {
        "stage": "H0",
        "status": "V3_A1_RTL_BRINGUP_PACKAGE_CREATED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_verified": True,
        "checkpoint_verified": True,
        "strict_checkpoint_load": True,
        "parameter_count": params,
        "input_shape": [16, 8, 24],
        "weights_status": "PROVISIONAL_V3_BRINGUP_ONLY",
        "golden_vectors_status": "PENDING_H1",
        "quantization_status": "EXPLORATORY_ONLY",
        "final_v4_replacement_required": True,
        "artifact_hash_manifest": str(hashes_path),
    }
    release_path = dirs["manifest"] / "release_manifest.json"
    release_path.write_text(json.dumps(release, indent=2) + "\n")

    print("H0 V3-A1 RTL BRING-UP PACKAGE: PASS")
    print(f"output_dir={out}")
    print(f"source_sha256={sha(source)}")
    print(f"checkpoint_sha256={sha(checkpoint_path)}")
    print(f"parameter_count={params}")
    print("input_contract=[16,8,24]")
    print("elements_per_window=3072")
    print("fp32_window_bytes=12288")
    print(f"adjacency_nonzero_entries={int(np.count_nonzero(adjacency))}")
    print(f"adjacency_weight_sum={float(adjacency.sum()):.9f}")
    print("weights_status=PROVISIONAL_V3_BRINGUP_ONLY")
    print("golden_vectors_status=PENDING_H1")
    print("quantization_status=EXPLORATORY_ONLY")
    print(f"artifact_manifest={hashes_path}")
    print(f"release_manifest={release_path}")


if __name__ == "__main__":
    main()
