#!/usr/bin/env python3
"""
Chrono-B1 Stage B1.9: export aligned validation predictions for A1 and B1.

The script:
- loads the corrected chronological V3 dataset;
- verifies each checkpoint's saved validation indices;
- reconstructs the frozen Conv1D-TemporalGCN from the A1 source;
- exports graph probabilities, node probabilities, labels and metadata;
- writes A1 and B1 validation NPZ files plus an export manifest.

No thresholds are selected here. No test predictions are read or exported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("chrono_a1_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class IndexedTemporalDataset(Dataset):
    def __init__(self, data_dir: Path, indices: np.ndarray):
        self.x = np.load(data_dir / "x.npy", mmap_mode="r")
        self.y_graph = np.load(data_dir / "y_graph.npy", mmap_mode="r")
        self.y_node = np.load(data_dir / "y_node.npy", mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        real_index = int(self.indices[position])
        return {
            "x": torch.from_numpy(
                np.asarray(self.x[real_index], dtype=np.float32)
            ),
            "y_graph": torch.tensor(
                float(self.y_graph[real_index]), dtype=torch.float32
            ),
            "y_node": torch.from_numpy(
                np.asarray(self.y_node[real_index], dtype=np.float32)
            ),
            "real_index": torch.tensor(real_index, dtype=torch.int64),
        }


def read_metadata(data_dir: Path, indices: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in (
        "run_id",
        "profile",
        "strength",
        "active_cores",
        "attackers",
        "end_epoch",
        "seed",
        "split",
    ):
        path = data_dir / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing metadata file: {path}")
        array = np.load(
            path,
            mmap_mode="r" if name == "end_epoch" else None,
            allow_pickle=True,
        )
        result[name] = np.asarray(array[indices])
    return result


def export_model(
    *,
    label: str,
    model_dir: Path,
    source_module,
    data_dir: Path,
    output_path: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = model_dir / "best_model.pt"
    splits_path = model_dir / "splits.npz"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    splits = np.load(splits_path)
    val_idx = splits["val_idx"].astype(np.int64)

    saved_args = checkpoint["args"]
    x = np.load(data_dir / "x.npy", mmap_mode="r")
    edge_index = np.load(data_dir / "edge_index.npy").astype(np.int64)

    if len(x.shape) != 4:
        raise ValueError(f"Expected x shape [S,N,T,F], got {x.shape}")

    num_nodes = int(x.shape[1])
    input_features = int(x.shape[-1])

    model = source_module.TemporalGCN(
        input_features=input_features,
        temporal_dim=int(saved_args["temporal_dim"]),
        gcn_hidden=int(saved_args["gcn_hidden"]),
        gcn_out=int(saved_args["gcn_out"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    a_hat = source_module.build_normalized_adjacency(
        edge_index,
        num_nodes,
    ).to(device)

    dataset = IndexedTemporalDataset(data_dir, val_idx)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    graph_prob_parts: list[np.ndarray] = []
    node_prob_parts: list[np.ndarray] = []
    y_graph_parts: list[np.ndarray] = []
    y_node_parts: list[np.ndarray] = []
    real_index_parts: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"export {label} val", leave=False):
            batch_x = batch["x"].to(device, non_blocking=True)
            graph_logits, node_logits = model(batch_x, a_hat)

            graph_prob_parts.append(
                torch.sigmoid(graph_logits).cpu().numpy().astype(np.float32)
            )
            node_prob_parts.append(
                torch.sigmoid(node_logits).cpu().numpy().astype(np.float32)
            )
            y_graph_parts.append(
                batch["y_graph"].numpy().astype(np.float32)
            )
            y_node_parts.append(
                batch["y_node"].numpy().astype(np.float32)
            )
            real_index_parts.append(
                batch["real_index"].numpy().astype(np.int64)
            )

    graph_prob = np.concatenate(graph_prob_parts, axis=0)
    node_prob = np.concatenate(node_prob_parts, axis=0)
    y_graph = np.concatenate(y_graph_parts, axis=0)
    y_node = np.concatenate(y_node_parts, axis=0)
    real_index = np.concatenate(real_index_parts, axis=0)

    if not np.array_equal(real_index, val_idx):
        raise RuntimeError(
            f"{label}: exported row order does not match saved val_idx"
        )

    metadata = read_metadata(data_dir, real_index)

    np.savez_compressed(
        output_path,
        model_label=np.asarray(label),
        split=np.asarray("val"),
        real_index=real_index,
        graph_prob=graph_prob,
        node_prob=node_prob,
        y_graph=y_graph,
        y_node=y_node,
        run_id=metadata["run_id"],
        profile=metadata["profile"],
        strength=metadata["strength"],
        active_cores=metadata["active_cores"],
        attackers=metadata["attackers"],
        end_epoch=metadata["end_epoch"],
        seed=metadata["seed"],
        dataset_split=metadata["split"],
    )

    return {
        "model_label": label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "splits_sha256": sha256(splits_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "rows": int(len(real_index)),
        "nodes": int(node_prob.shape[1]),
        "graph_prob_shape": list(graph_prob.shape),
        "node_prob_shape": list(node_prob.shape),
        "real_index_first": int(real_index[0]),
        "real_index_last": int(real_index[-1]),
        "graph_prob_min": float(graph_prob.min()),
        "graph_prob_max": float(graph_prob.max()),
        "node_prob_min": float(node_prob.min()),
        "node_prob_max": float(node_prob.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model-source", required=True, type=Path)
    parser.add_argument("--a1-dir", required=True, type=Path)
    parser.add_argument("--b1-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    source_path = args.model_source.resolve()
    a1_dir = args.a1_dir.resolve()
    b1_dir = args.b1_dir.resolve()
    output_dir = args.output_dir.resolve()

    required_data = (
        "x.npy",
        "y_graph.npy",
        "y_node.npy",
        "edge_index.npy",
        "run_id.npy",
        "profile.npy",
        "strength.npy",
        "active_cores.npy",
        "attackers.npy",
        "end_epoch.npy",
        "seed.npy",
        "split.npy",
    )
    for name in required_data:
        if not (data_dir / name).is_file():
            raise SystemExit(f"STOP: missing dataset file: {data_dir / name}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: output directory already exists and is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_module = load_module(source_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    a1_output = output_dir / "a1_validation_predictions.npz"
    b1_output = output_dir / "b1_validation_predictions.npz"

    a1_manifest = export_model(
        label="A1",
        model_dir=a1_dir,
        source_module=source_module,
        data_dir=data_dir,
        output_path=a1_output,
        batch_size=args.batch_size,
        device=device,
    )
    b1_manifest = export_model(
        label="B1",
        model_dir=b1_dir,
        source_module=source_module,
        data_dir=data_dir,
        output_path=b1_output,
        batch_size=args.batch_size,
        device=device,
    )

    a1 = np.load(a1_output, allow_pickle=True)
    b1 = np.load(b1_output, allow_pickle=True)

    alignment_checks = {
        "row_count_equal": len(a1["real_index"]) == len(b1["real_index"]),
        "real_index_equal": bool(
            np.array_equal(a1["real_index"], b1["real_index"])
        ),
        "y_graph_equal": bool(np.array_equal(a1["y_graph"], b1["y_graph"])),
        "y_node_equal": bool(np.array_equal(a1["y_node"], b1["y_node"])),
        "run_id_equal": bool(np.array_equal(a1["run_id"], b1["run_id"])),
        "profile_equal": bool(np.array_equal(a1["profile"], b1["profile"])),
        "strength_equal": bool(np.array_equal(a1["strength"], b1["strength"])),
        "active_cores_equal": bool(
            np.array_equal(a1["active_cores"], b1["active_cores"])
        ),
        "attackers_equal": bool(
            np.array_equal(a1["attackers"], b1["attackers"])
        ),
        "end_epoch_equal": bool(
            np.array_equal(a1["end_epoch"], b1["end_epoch"])
        ),
        "dataset_split_all_val_a1": bool(
            np.all(a1["dataset_split"].astype(str) == "val")
        ),
        "dataset_split_all_val_b1": bool(
            np.all(b1["dataset_split"].astype(str) == "val")
        ),
    }

    manifest = {
        "stage": "B1.9",
        "split": "validation only",
        "model_source": str(source_path),
        "model_source_sha256": sha256(source_path),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "a1": a1_manifest,
        "b1": b1_manifest,
        "alignment_checks": alignment_checks,
        "alignment_pass": all(alignment_checks.values()),
        "test_predictions_exported": False,
        "thresholds_selected": False,
    }

    manifest_path = output_dir / "validation_export_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "B1.9 VALIDATION PREDICTION EXPORT: "
        + ("PASS" if manifest["alignment_pass"] else "FAIL")
    )
    print(f"a1_rows={a1_manifest['rows']}")
    print(f"b1_rows={b1_manifest['rows']}")
    print(f"node_count={a1_manifest['nodes']}")
    print(f"real_index_equal={alignment_checks['real_index_equal']}")
    print(f"y_graph_equal={alignment_checks['y_graph_equal']}")
    print(f"y_node_equal={alignment_checks['y_node_equal']}")
    print(f"run_id_equal={alignment_checks['run_id_equal']}")
    print(
        "all_rows_validation="
        f"{alignment_checks['dataset_split_all_val_a1'] and alignment_checks['dataset_split_all_val_b1']}"
    )
    print(f"a1_output={a1_output}")
    print(f"b1_output={b1_output}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
