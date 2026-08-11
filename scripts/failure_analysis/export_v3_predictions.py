#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


# Make "scripts.train_..." importable when this file is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GRAPH_THRESHOLD = 0.50
NODE_THRESHOLD = 0.50


@dataclass(frozen=True)
class ModelSpec:
    name: str
    module_name: str
    class_name: str
    model_dir: Path
    output_filename: str


MODEL_SPECS = [
    ModelSpec(
        name="conv1d_gcn",
        module_name="scripts.train_temporal_gcn_v3",
        class_name="TemporalGCN",
        model_dir=Path(
            "models/v3/temporal_gcn_ports_v3_builtin_split"
        ),
        output_filename="conv1d_gcn_predictions.npz",
    ),
    ModelSpec(
        name="tcn_attention_gcn",
        module_name="scripts.train_temporal_tcn_gcn_v3",
        class_name="TemporalTCNGCN",
        model_dir=Path(
            "models/v3/temporal_tcn_gcn_ports_v3_builtin_split"
        ),
        output_filename="tcn_attention_gcn_predictions.npz",
    ),
    ModelSpec(
        name="tcn_meanpool_gcn",
        module_name="scripts.train_temporal_tcn_meanpool_gcn_v3",
        class_name="TemporalTCNGCN",
        model_dir=Path(
            "models/v3/temporal_tcn_meanpool_gcn_ports_v3_builtin_split"
        ),
        output_filename="tcn_meanpool_gcn_predictions.npz",
    ),
    ModelSpec(
        name="tcn_maxpool_gcn",
        module_name="scripts.train_temporal_tcn_maxpool_gcn_v3",
        class_name="TemporalTCNGCN",
        model_dir=Path(
            "models/v3/temporal_tcn_maxpool_gcn_ports_v3_builtin_split"
        ),
        output_filename="tcn_maxpool_gcn_predictions.npz",
    ),
]


def load_array(
    path: Path,
    *,
    mmap: bool = False,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    if mmap:
        try:
            return np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )
        except ValueError:
            pass

    try:
        return np.load(path, allow_pickle=False)
    except ValueError:
        return np.load(path, allow_pickle=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, torch.device):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if hasattr(value, "__dict__"):
        return json_safe(vars(value))

    return value


def atomic_write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    text = json.dumps(
        json_safe(data),
        indent=2,
        sort_keys=True,
    ) + "\n"

    atomic_write_text(path, text)


def atomic_savez(
    path: Path,
    **arrays: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    try:
        with temporary.open(
            "w",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()

            for row in rows:
                writer.writerow(row)

        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_int_array(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(
        np.asarray(array, dtype=np.int64).reshape(-1)
    )

    return hashlib.sha256(canonical.tobytes()).hexdigest()


def sha256_string_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()

    for value in np.asarray(array).astype(str).reshape(-1):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file missing: {path}")

    value = json.loads(path.read_text())

    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")

    return value


def checkpoint_args_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return dict(value)

    if hasattr(value, "__dict__"):
        return dict(vars(value))

    raise TypeError(
        "Unsupported checkpoint argument type: "
        f"{type(value).__name__}"
    )


def git_value(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        return None

    value = result.stdout.strip()
    return value or None


def require_output_paths_available(
    paths: list[Path],
    overwrite: bool,
) -> None:
    existing = [path for path in paths if path.exists()]

    if existing and not overwrite:
        joined = "\n".join(f"  {path}" for path in existing)

        raise FileExistsError(
            "Output files already exist. Remove them, choose another "
            "output directory, or rerun with --overwrite:\n"
            f"{joined}"
        )


def build_sample_ids(
    run_id: np.ndarray,
    end_epoch: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        [
            f"{run}:{epoch}"
            for run, epoch in zip(run_id, end_epoch)
        ],
        dtype=str,
    )

    if len(np.unique(values)) != len(values):
        duplicate_values, duplicate_counts = np.unique(
            values,
            return_counts=True,
        )
        duplicates = duplicate_values[duplicate_counts > 1]

        raise RuntimeError(
            "run_id:end_epoch is not unique. Duplicates include: "
            f"{duplicates[:20].tolist()}"
        )

    return values


def validate_model_splits(
    data_split: np.ndarray,
    specs: list[ModelSpec],
) -> dict[str, np.ndarray]:
    expected = {
        "train_idx": np.where(data_split == "train")[0].astype(
            np.int64
        ),
        "val_idx": np.where(data_split == "val")[0].astype(
            np.int64
        ),
        "test_idx": np.where(data_split == "test")[0].astype(
            np.int64
        ),
    }

    for spec in specs:
        split_path = PROJECT_ROOT / spec.model_dir / "splits.npz"

        if not split_path.is_file():
            raise FileNotFoundError(
                f"Split file missing for {spec.name}: {split_path}"
            )

        stored = np.load(split_path, allow_pickle=False)

        for key, expected_values in expected.items():
            if key not in stored.files:
                raise KeyError(
                    f"{split_path} does not contain {key}"
                )

            actual_values = np.asarray(
                stored[key],
                dtype=np.int64,
            ).reshape(-1)

            if not np.array_equal(
                actual_values,
                expected_values,
            ):
                raise RuntimeError(
                    f"{spec.name} {key} does not match split.npy"
                )

    return expected


def constructor_value_map(
    *,
    checkpoint_args: dict[str, Any],
    input_feature_dim: int,
    num_nodes: int,
) -> dict[str, Any]:
    temporal_dim = int(checkpoint_args.get("temporal_dim", 8))
    gcn_hidden = int(checkpoint_args.get("gcn_hidden", 16))
    gcn_out = int(checkpoint_args.get("gcn_out", 8))
    dropout = float(
        checkpoint_args.get(
            "tcn_dropout",
            checkpoint_args.get("dropout", 0.1),
        )
    )

    return {
        # Input-feature aliases
        "input_feature_dim": input_feature_dim,
        "input_dim": input_feature_dim,
        "in_dim": input_feature_dim,
        "in_features": input_feature_dim,
        "num_features": input_feature_dim,
        "feature_dim": input_feature_dim,
        "input_channels": input_feature_dim,

        # Temporal embedding aliases
        "temporal_dim": temporal_dim,
        "temporal_hidden": temporal_dim,
        "temporal_hidden_dim": temporal_dim,

        # Graph hidden aliases
        "gcn_hidden": gcn_hidden,
        "graph_hidden": gcn_hidden,
        "gcn_hidden_dim": gcn_hidden,

        # Graph output aliases
        "gcn_out": gcn_out,
        "graph_out": gcn_out,
        "gcn_output_dim": gcn_out,
        "node_embedding_dim": gcn_out,

        # Dropout aliases
        "dropout": dropout,
        "tcn_dropout": dropout,

        # Topology aliases
        "num_nodes": num_nodes,
        "n_nodes": num_nodes,
    }


def instantiate_model(
    model_class: type[torch.nn.Module],
    *,
    checkpoint_args: dict[str, Any],
    input_feature_dim: int,
    num_nodes: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    signature = inspect.signature(model_class.__init__)
    value_map = constructor_value_map(
        checkpoint_args=checkpoint_args,
        input_feature_dim=input_feature_dim,
        num_nodes=num_nodes,
    )

    positional_arguments: list[Any] = []
    keyword_arguments: dict[str, Any] = {}

    fallback_values = [
        input_feature_dim,
        int(checkpoint_args.get("temporal_dim", 8)),
        int(checkpoint_args.get("gcn_hidden", 16)),
        int(checkpoint_args.get("gcn_out", 8)),
        float(checkpoint_args.get("tcn_dropout", 0.1)),
        num_nodes,
    ]
    fallback_index = 0

    for name, parameter in signature.parameters.items():
        if name == "self":
            continue

        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        if name in value_map:
            value = value_map[name]
        elif name in checkpoint_args:
            value = checkpoint_args[name]
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            # Last-resort positional fallback for simple constructors.
            if fallback_index >= len(fallback_values):
                raise TypeError(
                    f"Cannot supply required constructor argument {name!r} "
                    f"for {model_class.__name__}. Signature: {signature}"
                )

            value = fallback_values[fallback_index]
            fallback_index += 1

        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional_arguments.append(value)
        else:
            keyword_arguments[name] = value

    model = model_class(
        *positional_arguments,
        **keyword_arguments,
    )

    construction_record = {
        "class": model_class.__name__,
        "signature": str(signature),
        "positional_arguments": positional_arguments,
        "keyword_arguments": keyword_arguments,
    }

    return model, construction_record


def load_model(
    spec: ModelSpec,
    *,
    edge_index: np.ndarray,
    input_feature_dim: int,
    num_nodes: int,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    torch.Tensor,
    Any,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    model_dir = (PROJECT_ROOT / spec.model_dir).resolve()
    checkpoint_path = model_dir / "best_model.pt"
    summary_path = model_dir / "summary.json"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint missing: {checkpoint_path}"
        )

    module = importlib.import_module(spec.module_name)

    if not hasattr(module, spec.class_name):
        raise AttributeError(
            f"{spec.module_name} has no class {spec.class_name}"
        )

    if not hasattr(module, "build_normalized_adjacency"):
        raise AttributeError(
            f"{spec.module_name} has no "
            "build_normalized_adjacency function"
        )

    model_class = getattr(module, spec.class_name)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint is not a dictionary: {checkpoint_path}"
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"model_state_dict missing from {checkpoint_path}"
        )

    checkpoint_args = checkpoint_args_to_dict(
        checkpoint.get("args")
    )

    model, construction_record = instantiate_model(
        model_class,
        checkpoint_args=checkpoint_args,
        input_feature_dim=input_feature_dim,
        num_nodes=num_nodes,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.to(device)
    model.eval()

    adjacency = module.build_normalized_adjacency(
        edge_index,
        num_nodes,
    ).to(device)

    summary = read_json(summary_path)

    record = {
        "name": spec.name,
        "module": spec.module_name,
        "class": spec.class_name,
        "model_dir": str(model_dir),
        "checkpoint_path": str(checkpoint_path),
        "summary_path": str(summary_path),
        "checkpoint_model_name": checkpoint.get("model_name"),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "summary_best_epoch": summary.get("best_epoch"),
        "checkpoint_args": checkpoint_args,
        "construction": construction_record,
        "stored_tensor_elements": int(
            sum(
                value.numel()
                for value in checkpoint[
                    "model_state_dict"
                ].values()
                if torch.is_tensor(value)
            )
        ),
    }

    return (
        model,
        adjacency,
        module,
        checkpoint,
        summary,
        record,
    )


def unpack_model_output(
    output: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, dict):
        graph_candidates = [
            "graph_logits",
            "graph_logit",
            "graph_output",
        ]
        node_candidates = [
            "node_logits",
            "node_logit",
            "node_output",
        ]

        graph_logits = next(
            (
                output[name]
                for name in graph_candidates
                if name in output
            ),
            None,
        )
        node_logits = next(
            (
                output[name]
                for name in node_candidates
                if name in output
            ),
            None,
        )

        if graph_logits is None or node_logits is None:
            raise RuntimeError(
                "Dictionary model output does not contain recognizable "
                "graph and node logits."
            )

        return graph_logits, node_logits

    if isinstance(output, (tuple, list)) and len(output) >= 2:
        return output[0], output[1]

    raise RuntimeError(
        "Model output must be a dictionary or tuple/list containing "
        "graph and node logits."
    )


def run_logits(
    *,
    model: torch.nn.Module,
    adjacency: torch.Tensor,
    x: np.ndarray,
    indices: np.ndarray,
    num_nodes: int,
    batch_size: int,
    device: torch.device,
    show_progress: bool,
    progress_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    graph_logits_out = np.empty(
        len(indices),
        dtype=np.float32,
    )
    node_logits_out = np.empty(
        (len(indices), num_nodes),
        dtype=np.float32,
    )

    number_of_batches = (
        len(indices) + batch_size - 1
    ) // batch_size

    progress_interval = max(1, number_of_batches // 20)

    with torch.inference_mode():
        for batch_number, start in enumerate(
            range(0, len(indices), batch_size),
            start=1,
        ):
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]

            # Advanced indexing produces only the current batch in RAM.
            batch_x_numpy = np.asarray(
                x[batch_indices],
                dtype=np.float32,
            )

            batch_x = torch.from_numpy(
                batch_x_numpy
            ).to(
                device,
                non_blocking=device.type == "cuda",
            )

            output = model(batch_x, adjacency)
            graph_logits, node_logits = unpack_model_output(output)

            graph_logits = graph_logits.detach().float()

            if graph_logits.ndim == 2 and graph_logits.shape[-1] == 1:
                graph_logits = graph_logits.squeeze(-1)

            graph_logits = graph_logits.reshape(-1)

            node_logits = node_logits.detach().float()

            if node_logits.ndim == 3 and node_logits.shape[-1] == 1:
                node_logits = node_logits.squeeze(-1)

            if node_logits.shape != (
                len(batch_indices),
                num_nodes,
            ):
                raise RuntimeError(
                    "Unexpected node-logit shape. "
                    f"Expected {(len(batch_indices), num_nodes)}, "
                    f"received {tuple(node_logits.shape)}"
                )

            if graph_logits.shape != (len(batch_indices),):
                raise RuntimeError(
                    "Unexpected graph-logit shape. "
                    f"Expected {(len(batch_indices),)}, "
                    f"received {tuple(graph_logits.shape)}"
                )

            graph_logits_out[start:end] = (
                graph_logits.cpu().numpy()
            )
            node_logits_out[start:end] = (
                node_logits.cpu().numpy()
            )

            if show_progress and (
                batch_number == 1
                or batch_number == number_of_batches
                or batch_number % progress_interval == 0
            ):
                print(
                    f"{progress_prefix}: "
                    f"batch {batch_number}/{number_of_batches}, "
                    f"samples {end}/{len(indices)}",
                    flush=True,
                )

    return graph_logits_out, node_logits_out


def verify_determinism(
    *,
    model: torch.nn.Module,
    adjacency: torch.Tensor,
    x: np.ndarray,
    indices: np.ndarray,
    num_nodes: int,
    device: torch.device,
) -> dict[str, Any]:
    subset = indices[: min(128, len(indices))]

    first_graph, first_node = run_logits(
        model=model,
        adjacency=adjacency,
        x=x,
        indices=subset,
        num_nodes=num_nodes,
        batch_size=min(128, len(subset)),
        device=device,
        show_progress=False,
        progress_prefix="determinism-pass-1",
    )

    second_graph, second_node = run_logits(
        model=model,
        adjacency=adjacency,
        x=x,
        indices=subset,
        num_nodes=num_nodes,
        batch_size=min(128, len(subset)),
        device=device,
        show_progress=False,
        progress_prefix="determinism-pass-2",
    )

    graph_difference = float(
        np.max(np.abs(first_graph - second_graph))
    )
    node_difference = float(
        np.max(np.abs(first_node - second_node))
    )

    return {
        "sample_count": int(len(subset)),
        "graph_max_abs_difference": graph_difference,
        "node_max_abs_difference": node_difference,
        "deterministic_within_1e_7": (
            graph_difference <= 1e-7
            and node_difference <= 1e-7
        ),
    }


def manual_binary_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = (
        np.asarray(y_probability).reshape(-1) >= threshold
    ).astype(np.int64)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    accuracy = (tn + tp) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        2 * precision * recall / max(1e-15, precision + recall)
    )
    fpr = fp / max(1, fp + tn)

    return {
        "acc": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def manual_node_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = (
        np.asarray(y_probability) >= threshold
    ).astype(np.int64)

    flattened_true = y_true.reshape(-1)
    flattened_pred = y_pred.reshape(-1)

    tn = int(
        (
            (flattened_true == 0)
            & (flattened_pred == 0)
        ).sum()
    )
    fp = int(
        (
            (flattened_true == 0)
            & (flattened_pred == 1)
        ).sum()
    )
    fn = int(
        (
            (flattened_true == 1)
            & (flattened_pred == 0)
        ).sum()
    )
    tp = int(
        (
            (flattened_true == 1)
            & (flattened_pred == 1)
        ).sum()
    )

    accuracy = (tn + tp) / max(1, len(flattened_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        2 * precision * recall / max(1e-15, precision + recall)
    )

    attack_mask = y_true.sum(axis=1) > 0

    if attack_mask.any():
        exact = float(
            np.mean(
                np.all(
                    y_pred[attack_mask] == y_true[attack_mask],
                    axis=1,
                )
            )
        )
    else:
        exact = float("nan")

    return {
        "node_acc": float(accuracy),
        "node_precision": float(precision),
        "node_recall": float(recall),
        "node_f1": float(f1),
        "exact_localization": exact,
    }


def numeric_mapping(
    value: Any,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    return {
        str(key): json_safe(item)
        for key, item in value.items()
    }


def module_metrics_or_manual(
    *,
    module: Any,
    y_graph: np.ndarray,
    graph_probability: np.ndarray,
    y_node: np.ndarray,
    node_probability: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    fallback_graph = manual_binary_metrics(
        y_graph,
        graph_probability,
        GRAPH_THRESHOLD,
    )
    fallback_node = manual_node_metrics(
        y_node,
        node_probability,
        NODE_THRESHOLD,
    )

    if not hasattr(module, "binary_metrics") or not hasattr(
        module,
        "node_metrics",
    ):
        return fallback_graph, fallback_node, "manual"

    try:
        graph_raw = module.binary_metrics(
            y_graph,
            graph_probability,
            GRAPH_THRESHOLD,
        )
        node_raw = module.node_metrics(
            y_node,
            node_probability,
            NODE_THRESHOLD,
        )
    except Exception as error:
        print(
            "WARNING: module metric functions failed; "
            f"using manual metrics: {error!r}",
            flush=True,
        )
        return fallback_graph, fallback_node, "manual_fallback"

    graph_mapping = numeric_mapping(graph_raw)
    node_mapping = numeric_mapping(node_raw)

    if graph_mapping is None or node_mapping is None:
        return fallback_graph, fallback_node, "manual_fallback"

    graph = dict(fallback_graph)
    node = dict(fallback_node)

    graph_aliases = {
        "acc": ["acc", "accuracy"],
        "precision": ["precision"],
        "recall": ["recall"],
        "f1": ["f1"],
        "fpr": ["fpr"],
        "tn": ["tn"],
        "fp": ["fp"],
        "fn": ["fn"],
        "tp": ["tp"],
    }

    node_aliases = {
        "node_acc": ["node_acc", "acc", "accuracy"],
        "node_precision": ["node_precision", "precision"],
        "node_recall": ["node_recall", "recall"],
        "node_f1": ["node_f1", "f1"],
        "exact_localization": [
            "exact_localization",
            "exact_loc",
        ],
    }

    for output_key, candidates in graph_aliases.items():
        for candidate in candidates:
            if candidate in graph_mapping:
                graph[output_key] = graph_mapping[candidate]
                break

    for output_key, candidates in node_aliases.items():
        for candidate in candidates:
            if candidate in node_mapping:
                node[output_key] = node_mapping[candidate]
                break

    return graph, node, "module"


def summary_difference(
    summary_metrics: dict[str, Any],
    graph_metrics: dict[str, Any],
    node_metrics: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    comparisons = {
        "acc": graph_metrics["acc"],
        "precision": graph_metrics["precision"],
        "recall": graph_metrics["recall"],
        "f1": graph_metrics["f1"],
        "fpr": graph_metrics["fpr"],
        "tn": graph_metrics["tn"],
        "fp": graph_metrics["fp"],
        "fn": graph_metrics["fn"],
        "tp": graph_metrics["tp"],
        "node_acc": node_metrics["node_acc"],
        "node_precision": node_metrics["node_precision"],
        "node_recall": node_metrics["node_recall"],
        "node_f1": node_metrics["node_f1"],
        "exact_localization": node_metrics[
            "exact_localization"
        ],
    }

    differences: dict[str, float] = {}

    for key, recomputed in comparisons.items():
        if key not in summary_metrics:
            continue

        saved = summary_metrics[key]

        try:
            differences[key] = abs(
                float(recomputed) - float(saved)
            )
        except (TypeError, ValueError):
            continue

    maximum = max(differences.values(), default=float("nan"))
    return float(maximum), differences


def create_metadata_rows(
    *,
    export_indices: np.ndarray,
    sample_ids: np.ndarray,
    run_id: np.ndarray,
    split: np.ndarray,
    end_epoch: np.ndarray,
    y_graph: np.ndarray,
    y_node: np.ndarray,
    profile: np.ndarray,
    active_cores: np.ndarray,
    attackers: np.ndarray,
    strength: np.ndarray,
    seed: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    attacker_counts = y_node[export_indices].sum(axis=1).astype(
        np.int64
    )

    for order, sample_index in enumerate(export_indices):
        rows.append(
            {
                "sample_order_within_export": order,
                "sample_index": int(sample_index),
                "sample_id": sample_ids[order],
                "run_id": str(run_id[sample_index]),
                "split": str(split[sample_index]),
                "end_epoch": int(end_epoch[sample_index]),
                "true_graph": int(y_graph[sample_index]),
                "true_attacker_count": int(
                    attacker_counts[order]
                ),
                "profile": str(profile[sample_index]),
                "active_cores": str(
                    active_cores[sample_index]
                ),
                "attackers": str(attackers[sample_index]),
                "strength": str(strength[sample_index]),
                "seed": str(seed[sample_index]),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export aligned V3 validation and test predictions for "
            "all four final models."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--tables-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--logs-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. Omit for the final full export."
        ),
    )
    parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=1e-5,
    )
    args = parser.parse_args()

    started_at = time.time()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    tables_dir = args.tables_dir.resolve()
    logs_dir = args.logs_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "prediction_metadata.csv"
    manifest_path = output_dir / "prediction_manifest.json"
    metric_path = tables_dir / "exported_metric_reproduction.csv"
    alignment_path = logs_dir / "25_prediction_alignment_check.txt"

    model_output_paths = [
        output_dir / spec.output_filename
        for spec in MODEL_SPECS
    ]

    expected_outputs = [
        metadata_path,
        manifest_path,
        metric_path,
        alignment_path,
        *model_output_paths,
    ]

    require_output_paths_available(
        expected_outputs,
        overwrite=args.overwrite,
    )

    device = torch.device(
        "cpu"
        if args.cpu or not torch.cuda.is_available()
        else "cuda"
    )

    print("project_root:", PROJECT_ROOT)
    print("data_dir:", data_dir)
    print("output_dir:", output_dir)
    print("device:", device)
    print("batch_size:", args.batch_size)
    print(
        "max_samples_per_split:",
        args.max_samples_per_split,
    )

    # Memory-map large numeric arrays.
    x = load_array(data_dir / "x.npy", mmap=True)
    y_graph = np.asarray(
        load_array(data_dir / "y_graph.npy", mmap=True)
    ).astype(np.int64)
    y_node = np.asarray(
        load_array(data_dir / "y_node.npy", mmap=True)
    ).astype(np.int64)
    edge_index = np.asarray(
        load_array(data_dir / "edge_index.npy")
    ).astype(np.int64)

    run_id = np.asarray(
        load_array(data_dir / "run_id.npy")
    ).astype(str)
    split = np.asarray(
        load_array(data_dir / "split.npy")
    ).astype(str)
    end_epoch = np.asarray(
        load_array(data_dir / "end_epoch.npy", mmap=True)
    ).astype(np.int64)

    profile = np.asarray(
        load_array(data_dir / "profile.npy")
    ).astype(str)
    active_cores = np.asarray(
        load_array(data_dir / "active_cores.npy")
    ).astype(str)
    attackers = np.asarray(
        load_array(data_dir / "attackers.npy")
    ).astype(str)
    strength = np.asarray(
        load_array(data_dir / "strength.npy")
    ).astype(str)
    seed = np.asarray(
        load_array(data_dir / "seed.npy")
    ).astype(str)

    sample_count = int(x.shape[0])
    num_nodes = int(x.shape[1])
    temporal_length = int(x.shape[2])
    input_feature_dim = int(x.shape[3])

    required_lengths = {
        "y_graph": len(y_graph),
        "y_node": len(y_node),
        "run_id": len(run_id),
        "split": len(split),
        "end_epoch": len(end_epoch),
        "profile": len(profile),
        "active_cores": len(active_cores),
        "attackers": len(attackers),
        "strength": len(strength),
        "seed": len(seed),
    }

    for name, length in required_lengths.items():
        if length != sample_count:
            raise RuntimeError(
                f"Length mismatch: {name}={length}, "
                f"x={sample_count}"
            )

    if y_node.shape != (sample_count, num_nodes):
        raise RuntimeError(
            f"Unexpected y_node shape: {y_node.shape}"
        )

    expected_splits = validate_model_splits(
        split,
        MODEL_SPECS,
    )

    val_indices = expected_splits["val_idx"]
    test_indices = expected_splits["test_idx"]

    if args.max_samples_per_split is not None:
        limit = args.max_samples_per_split

        if limit <= 0:
            raise ValueError(
                "--max-samples-per-split must be positive"
            )

        val_indices = val_indices[:limit]
        test_indices = test_indices[:limit]

    export_indices = np.concatenate(
        [val_indices, test_indices]
    ).astype(np.int64)

    export_split = split[export_indices].astype(str)
    export_run_id = run_id[export_indices].astype(str)
    export_end_epoch = end_epoch[export_indices].astype(
        np.int64
    )

    sample_ids = build_sample_ids(
        export_run_id,
        export_end_epoch,
    )
    sample_order = np.arange(
        len(export_indices),
        dtype=np.int64,
    )

    export_true_graph = y_graph[export_indices].astype(
        np.int8
    )
    export_true_node = y_node[export_indices].astype(
        np.int8
    )
    export_true_attacker_count = export_true_node.sum(
        axis=1
    ).astype(np.int8)

    metadata_rows = create_metadata_rows(
        export_indices=export_indices,
        sample_ids=sample_ids,
        run_id=run_id,
        split=split,
        end_epoch=end_epoch,
        y_graph=y_graph,
        y_node=y_node,
        profile=profile,
        active_cores=active_cores,
        attackers=attackers,
        strength=strength,
        seed=seed,
    )

    metadata_fields = [
        "sample_order_within_export",
        "sample_index",
        "sample_id",
        "run_id",
        "split",
        "end_epoch",
        "true_graph",
        "true_attacker_count",
        "profile",
        "active_cores",
        "attackers",
        "strength",
        "seed",
    ]

    atomic_write_csv(
        metadata_path,
        metadata_fields,
        metadata_rows,
    )

    full_export = args.max_samples_per_split is None

    metric_rows: list[dict[str, Any]] = []
    model_manifest: dict[str, Any] = {}
    alignment_lines: list[str] = []

    all_finite = True
    all_shapes_valid = True
    all_sample_indices_aligned = True
    all_sample_ids_aligned = True
    all_split_labels_aligned = True
    all_true_labels_aligned = True
    all_metrics_reproduced = True
    all_deterministic = True

    for model_number, spec in enumerate(
        MODEL_SPECS,
        start=1,
    ):
        print()
        print("=" * 100)
        print(
            f"MODEL {model_number}/{len(MODEL_SPECS)}: "
            f"{spec.name}"
        )
        print("=" * 100)

        (
            model,
            adjacency,
            module,
            checkpoint,
            summary,
            model_record,
        ) = load_model(
            spec,
            edge_index=edge_index,
            input_feature_dim=input_feature_dim,
            num_nodes=num_nodes,
            device=device,
        )

        print(
            "constructor:",
            model_record["construction"],
            flush=True,
        )

        determinism = verify_determinism(
            model=model,
            adjacency=adjacency,
            x=x,
            indices=export_indices,
            num_nodes=num_nodes,
            device=device,
        )

        print("determinism:", determinism, flush=True)

        if not determinism["deterministic_within_1e_7"]:
            all_deterministic = False

        graph_logits, node_logits = run_logits(
            model=model,
            adjacency=adjacency,
            x=x,
            indices=export_indices,
            num_nodes=num_nodes,
            batch_size=args.batch_size,
            device=device,
            show_progress=True,
            progress_prefix=spec.name,
        )

        graph_probability = (
            1.0 / (1.0 + np.exp(-graph_logits))
        ).astype(np.float32)

        node_probability = (
            1.0 / (1.0 + np.exp(-node_logits))
        ).astype(np.float32)

        pred_graph = (
            graph_probability >= GRAPH_THRESHOLD
        ).astype(np.int8)
        pred_node = (
            node_probability >= NODE_THRESHOLD
        ).astype(np.int8)

        pred_attacker_count = pred_node.sum(
            axis=1
        ).astype(np.int8)

        exact_localization_per_sample = np.all(
            pred_node == export_true_node,
            axis=1,
        ).astype(np.int8)

        finite = (
            np.isfinite(graph_logits).all()
            and np.isfinite(node_logits).all()
            and np.isfinite(graph_probability).all()
            and np.isfinite(node_probability).all()
        )

        shapes_valid = (
            graph_logits.shape == (len(export_indices),)
            and graph_probability.shape
            == (len(export_indices),)
            and node_logits.shape
            == (len(export_indices), num_nodes)
            and node_probability.shape
            == (len(export_indices), num_nodes)
        )

        if not finite:
            all_finite = False

        if not shapes_valid:
            all_shapes_valid = False

        model_output_path = output_dir / spec.output_filename

        atomic_savez(
            model_output_path,
            model_name=np.asarray([spec.name]),
            sample_order_within_export=sample_order,
            sample_index=export_indices,
            sample_id=sample_ids,
            run_id=export_run_id,
            split=export_split,
            end_epoch=export_end_epoch,
            true_graph=export_true_graph,
            true_attacker_mask=export_true_node,
            true_attacker_count=export_true_attacker_count,
            graph_logit=graph_logits,
            graph_probability=graph_probability,
            pred_graph_at_0_50=pred_graph,
            graph_threshold=np.asarray(
                [GRAPH_THRESHOLD],
                dtype=np.float32,
            ),
            node_logits=node_logits,
            node_probabilities=node_probability,
            pred_attacker_mask_at_0_50=pred_node,
            node_threshold_0_50=np.asarray(
                [NODE_THRESHOLD],
                dtype=np.float32,
            ),
            pred_attacker_count_at_0_50=(
                pred_attacker_count
            ),
            exact_localization_at_0_50=(
                exact_localization_per_sample
            ),
        )

        # Reload the saved file and verify alignment.
        saved = np.load(
            model_output_path,
            allow_pickle=False,
        )

        index_aligned = np.array_equal(
            saved["sample_index"],
            export_indices,
        )
        id_aligned = np.array_equal(
            saved["sample_id"].astype(str),
            sample_ids,
        )
        split_aligned = np.array_equal(
            saved["split"].astype(str),
            export_split,
        )
        labels_aligned = (
            np.array_equal(
                saved["true_graph"],
                export_true_graph,
            )
            and np.array_equal(
                saved["true_attacker_mask"],
                export_true_node,
            )
        )

        all_sample_indices_aligned &= index_aligned
        all_sample_ids_aligned &= id_aligned
        all_split_labels_aligned &= split_aligned
        all_true_labels_aligned &= labels_aligned

        model_split_metrics: dict[str, Any] = {}

        for split_name in ["val", "test"]:
            split_mask = export_split == split_name

            graph_metrics, node_metrics, metric_source = (
                module_metrics_or_manual(
                    module=module,
                    y_graph=export_true_graph[split_mask],
                    graph_probability=graph_probability[
                        split_mask
                    ],
                    y_node=export_true_node[split_mask],
                    node_probability=node_probability[
                        split_mask
                    ],
                )
            )

            if full_export:
                saved_summary = summary.get(split_name, {})

                if not isinstance(saved_summary, dict):
                    saved_summary = {}

                maximum_difference, differences = (
                    summary_difference(
                        saved_summary,
                        graph_metrics,
                        node_metrics,
                    )
                )

                metrics_match = (
                    np.isfinite(maximum_difference)
                    and maximum_difference
                    <= args.metric_tolerance
                )
            else:
                maximum_difference = float("nan")
                differences = {}
                metrics_match = True

            if not metrics_match:
                all_metrics_reproduced = False

            metric_row = {
                "model": spec.name,
                "split": split_name,
                "accuracy": graph_metrics["acc"],
                "precision": graph_metrics["precision"],
                "recall": graph_metrics["recall"],
                "f1": graph_metrics["f1"],
                "fpr": graph_metrics["fpr"],
                "tn": graph_metrics["tn"],
                "fp": graph_metrics["fp"],
                "fn": graph_metrics["fn"],
                "tp": graph_metrics["tp"],
                "node_accuracy": node_metrics["node_acc"],
                "node_precision": node_metrics[
                    "node_precision"
                ],
                "node_recall": node_metrics["node_recall"],
                "node_f1": node_metrics["node_f1"],
                "exact_localization": node_metrics[
                    "exact_localization"
                ],
                "metric_source": metric_source,
                "saved_summary_max_abs_difference": (
                    maximum_difference
                ),
                "metrics_match_tolerance": metrics_match,
            }

            metric_rows.append(metric_row)

            model_split_metrics[split_name] = {
                "metrics": metric_row,
                "per_field_saved_summary_differences": (
                    differences
                ),
            }

            print(
                f"{spec.name} {split_name}: "
                f"acc={graph_metrics['acc']:.6f} "
                f"f1={graph_metrics['f1']:.6f} "
                f"fpr={graph_metrics['fpr']:.6f} "
                f"node_f1={node_metrics['node_f1']:.6f} "
                "exact="
                f"{node_metrics['exact_localization']:.6f} "
                f"summary_diff={maximum_difference}",
                flush=True,
            )

        model_record.update(
            {
                "output_file": str(model_output_path),
                "output_file_bytes": int(
                    model_output_path.stat().st_size
                ),
                "graph_threshold": GRAPH_THRESHOLD,
                "node_threshold": NODE_THRESHOLD,
                "validation_selected_node_threshold": None,
                "output_shapes": {
                    "graph_logits": list(graph_logits.shape),
                    "node_logits": list(node_logits.shape),
                },
                "finite_outputs": finite,
                "shapes_valid": shapes_valid,
                "sample_index_aligned": index_aligned,
                "sample_id_aligned": id_aligned,
                "split_aligned": split_aligned,
                "labels_aligned": labels_aligned,
                "determinism_check": determinism,
                "split_metrics": model_split_metrics,
            }
        )

        model_manifest[spec.name] = model_record

        alignment_lines.extend(
            [
                f"MODEL: {spec.name}",
                f"  output_file: {model_output_path}",
                f"  sample_index_aligned: {index_aligned}",
                f"  sample_id_aligned: {id_aligned}",
                f"  split_aligned: {split_aligned}",
                f"  labels_aligned: {labels_aligned}",
                f"  finite_outputs: {finite}",
                f"  shapes_valid: {shapes_valid}",
                "  deterministic_within_1e_7: "
                f"{determinism['deterministic_within_1e_7']}",
                "",
            ]
        )

        # Release GPU memory before the next model.
        del model
        del adjacency

        if device.type == "cuda":
            torch.cuda.empty_cache()

    metric_fields = [
        "model",
        "split",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
        "tn",
        "fp",
        "fn",
        "tp",
        "node_accuracy",
        "node_precision",
        "node_recall",
        "node_f1",
        "exact_localization",
        "metric_source",
        "saved_summary_max_abs_difference",
        "metrics_match_tolerance",
    ]

    atomic_write_csv(
        metric_path,
        metric_fields,
        metric_rows,
    )

    overall_success = all(
        [
            all_finite,
            all_shapes_valid,
            all_sample_indices_aligned,
            all_sample_ids_aligned,
            all_split_labels_aligned,
            all_true_labels_aligned,
            all_metrics_reproduced,
            all_deterministic,
        ]
    )

    split_hashes = {
        key: sha256_int_array(value)
        for key, value in expected_splits.items()
    }

    manifest = {
        "status": "complete" if overall_success else "validation_failed",
        "normalization_window_construction_provenance": "unresolved",
        "dataset": {
            "path": str(data_dir),
            "x_shape": list(x.shape),
            "x_dtype": str(x.dtype),
            "sample_count": sample_count,
            "num_nodes": num_nodes,
            "temporal_length": temporal_length,
            "input_feature_dim": input_feature_dim,
        },
        "export": {
            "exported_splits": ["val", "test"],
            "validation_sample_count": int(len(val_indices)),
            "test_sample_count": int(len(test_indices)),
            "total_sample_count": int(len(export_indices)),
            "full_export": full_export,
            "max_samples_per_split": args.max_samples_per_split,
            "batch_size": args.batch_size,
            "graph_threshold": GRAPH_THRESHOLD,
            "node_threshold": NODE_THRESHOLD,
            "sample_id_format": "run_id:end_epoch",
            "sample_order_sha256": sha256_int_array(
                export_indices
            ),
            "sample_id_sha256": sha256_string_array(
                sample_ids
            ),
            "metadata_file": str(metadata_path),
            "metrics_file": str(metric_path),
            "alignment_file": str(alignment_path),
        },
        "split_hashes": split_hashes,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else None
            ),
            "git_branch": git_value(
                "branch",
                "--show-current",
            ),
            "git_commit": git_value(
                "rev-parse",
                "HEAD",
            ),
        },
        "checks": {
            "all_sample_indices_aligned": (
                all_sample_indices_aligned
            ),
            "all_sample_ids_aligned": all_sample_ids_aligned,
            "all_split_labels_aligned": (
                all_split_labels_aligned
            ),
            "all_true_labels_aligned": (
                all_true_labels_aligned
            ),
            "all_prediction_arrays_finite": all_finite,
            "all_output_shapes_valid": all_shapes_valid,
            "all_aggregate_metrics_reproduced": (
                all_metrics_reproduced
            ),
            "all_determinism_checks_passed": (
                all_deterministic
            ),
            "metric_tolerance": args.metric_tolerance,
            "overall_success": overall_success,
        },
        "models": model_manifest,
        "runtime_seconds": float(time.time() - started_at),
    }

    atomic_write_json(
        manifest_path,
        manifest,
    )

    alignment_lines.extend(
        [
            "=" * 80,
            "GLOBAL ALIGNMENT CHECK",
            "=" * 80,
            "all sample indices aligned: "
            f"{all_sample_indices_aligned}",
            "all sample IDs aligned: "
            f"{all_sample_ids_aligned}",
            "all split labels aligned: "
            f"{all_split_labels_aligned}",
            "all true labels aligned: "
            f"{all_true_labels_aligned}",
            "all prediction arrays finite: "
            f"{all_finite}",
            "all output shapes valid: "
            f"{all_shapes_valid}",
            "all aggregate metrics reproduced: "
            f"{all_metrics_reproduced}",
            "all determinism checks passed: "
            f"{all_deterministic}",
            f"full export: {full_export}",
            f"overall success: {overall_success}",
            "",
            f"sample count: {len(export_indices)}",
            f"validation samples: {len(val_indices)}",
            f"test samples: {len(test_indices)}",
            "expected full sample count: 85618",
            "sample order SHA-256: "
            f"{sha256_int_array(export_indices)}",
            "sample ID SHA-256: "
            f"{sha256_string_array(sample_ids)}",
        ]
    )

    atomic_write_text(
        alignment_path,
        "\n".join(alignment_lines) + "\n",
    )

    print()
    print("=" * 100)
    print("EXPORT COMPLETE")
    print("=" * 100)
    print("metadata:", metadata_path)
    print("manifest:", manifest_path)
    print("metrics:", metric_path)
    print("alignment:", alignment_path)

    for path in model_output_paths:
        print("predictions:", path)

    print("overall_success:", overall_success)

    if not overall_success:
        raise SystemExit(
            "Prediction files were written, but at least one validation "
            "check failed. Inspect the logs before continuing."
        )


if __name__ == "__main__":
    main()
