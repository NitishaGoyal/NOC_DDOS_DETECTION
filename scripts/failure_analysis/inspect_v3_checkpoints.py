#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_MODEL_DIRS = [
    Path("models/v3/temporal_gcn_ports_v3_builtin_split"),
    Path("models/v3/temporal_tcn_gcn_ports_v3_builtin_split"),
    Path("models/v3/temporal_tcn_meanpool_gcn_ports_v3_builtin_split"),
    Path("models/v3/temporal_tcn_maxpool_gcn_ports_v3_builtin_split"),
]


INTERESTING_NAMES = {
    "model",
    "model_name",
    "data",
    "data_path",
    "dataset",
    "seed",
    "best_epoch",
    "split_mode",
    "splits_file",
    "epochs",
    "patience",
    "min_delta",
    "batch_size",
    "lr",
    "learning_rate",
    "weight_decay",
    "temporal_dim",
    "gcn_hidden",
    "gcn_out",
    "tcn_dropout",
    "node_loss_weight",
    "graph_threshold",
    "node_threshold",
    "num_nodes",
    "num_features",
    "feature_count",
    "temporal_length",
}


def object_to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, Namespace):
        return vars(value)

    if isinstance(value, dict):
        return value

    if hasattr(value, "__dict__"):
        return vars(value)

    return {"value": repr(value)}


def flatten_mapping(
    mapping: dict[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            flattened.update(
                flatten_mapping(value, prefix=full_key)
            )
        else:
            flattened[full_key] = value

    return flattened


def safe_json_load(path: Path) -> Any:
    if not path.is_file():
        return None

    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"ERROR": repr(exc)}


def print_selected_mapping(
    title: str,
    mapping: dict[str, Any],
) -> None:
    print(title)

    flattened = flatten_mapping(mapping)
    selected = []

    for full_key, value in flattened.items():
        leaf = full_key.split(".")[-1]

        if leaf in INTERESTING_NAMES:
            selected.append((full_key, value))

    if not selected:
        print("  no selected fields found")
    else:
        for key, value in sorted(selected):
            print(f"  {key}: {value!r}")


def count_state_dict_elements(
    state_dict: dict[str, Any],
) -> tuple[int, int]:
    tensor_count = 0
    element_count = 0

    for value in state_dict.values():
        if torch.is_tensor(value):
            tensor_count += 1
            element_count += int(value.numel())

    return tensor_count, element_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect V3 checkpoint metadata without printing weights."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    model_dirs = args.model_dir or DEFAULT_MODEL_DIRS

    x_path = data_dir / "x.npy"

    print("data_dir:", data_dir)

    if x_path.is_file():
        x = np.load(x_path, mmap_mode="r")

        print("dataset_x_shape:", x.shape)
        print("dataset_x_dtype:", x.dtype)

        if x.ndim == 4:
            print("dataset_sample_count:", x.shape[0])
            print("dataset_router_count:", x.shape[1])
            print("dataset_temporal_length:", x.shape[2])
            print("dataset_feature_count:", x.shape[3])
    else:
        print("dataset_x_missing:", x_path)

    for model_dir in model_dirs:
        model_dir = model_dir.resolve()

        print("\n" + "=" * 100)
        print("MODEL DIRECTORY:", model_dir)
        print("=" * 100)

        summary_path = model_dir / "summary.json"
        history_path = model_dir / "history.json"
        checkpoint_path = model_dir / "best_model.pt"

        summary = safe_json_load(summary_path)
        history = safe_json_load(history_path)

        print("summary_path:", summary_path)
        print("history_path:", history_path)
        print("checkpoint_path:", checkpoint_path)

        if isinstance(summary, dict):
            print("summary_top_level_keys:", sorted(summary.keys()))
            print_selected_mapping(
                "selected_summary_fields:",
                summary,
            )
        else:
            print("summary_value:", summary)

        if isinstance(history, dict):
            print("history_top_level_keys:", sorted(history.keys()))

            for key, value in history.items():
                if isinstance(value, list):
                    print(f"history_length.{key}:", len(value))
        elif isinstance(history, list):
            print("history_length:", len(history))
        else:
            print("history_value:", history)

        if not checkpoint_path.is_file():
            print("CRITICAL: checkpoint missing")
            continue

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            print(
                "checkpoint_type:",
                type(checkpoint).__name__,
            )
            continue

        print("checkpoint_top_level_keys:", sorted(checkpoint.keys()))

        state_dict = checkpoint.get("model_state_dict")

        if isinstance(state_dict, dict):
            tensor_count, element_count = count_state_dict_elements(
                state_dict
            )

            print("state_dict_tensor_count:", tensor_count)
            print(
                "state_dict_total_tensor_elements:",
                element_count,
            )
            print(
                "state_dict_approx_fp32_bytes:",
                element_count * 4,
            )
            print(
                "state_dict_approx_int8_bytes:",
                element_count,
            )
            print(
                "state_dict_key_preview:",
                sorted(state_dict.keys())[:30],
            )
        else:
            print("model_state_dict_missing_or_invalid")

        for candidate_key in [
            "args",
            "config",
            "model_args",
            "training_args",
            "hyperparameters",
        ]:
            if candidate_key not in checkpoint:
                continue

            mapping = object_to_mapping(checkpoint[candidate_key])

            print_selected_mapping(
                f"selected_checkpoint_fields.{candidate_key}:",
                mapping,
            )

            print(
                f"all_checkpoint_fields.{candidate_key}:"
            )

            for key, value in sorted(
                flatten_mapping(mapping).items()
            ):
                print(f"  {key}: {value!r}")

        if "epoch" in checkpoint:
            print("checkpoint_epoch:", checkpoint["epoch"])

        if "best_epoch" in checkpoint:
            print(
                "checkpoint_best_epoch:",
                checkpoint["best_epoch"],
            )

        if "best_val_loss" in checkpoint:
            print(
                "checkpoint_best_val_loss:",
                checkpoint["best_val_loss"],
            )


if __name__ == "__main__":
    main()
