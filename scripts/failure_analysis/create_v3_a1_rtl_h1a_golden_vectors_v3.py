#!/usr/bin/env python3
"""
H1A: Generate validation-only, layer-by-layer FP32 golden vectors for
Chrono-A1 RTL bring-up.

Uses the already-exported A1 validation predictions from G1.3 only for
deterministic vector selection. Runs model inference only on the selected
validation samples. It never accesses the test split.

Outputs 24 balanced vectors with:
- input tensor;
- Conv1D pre/post activation;
- temporal pooled embedding;
- GCN1 pre/post activation;
- GCN2 pre/post activation;
- node logits/probabilities;
- graph embedding/logit/probability;
- decisions and true labels;
- .npy, .npz, CSV metadata, and IEEE-754 FP32 .hex files.

Normalization equations are NOT inferred here. H1B must recover them from the
authoritative dataset-builder source or simulation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


EXPECTED_SOURCE_SHA256 = (
    "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "c29cdd2303669a61454ed118ef0be4b8809a635b12c0c42806f8b6992340e0ef"
)
EXPECTED_PARAMETER_COUNT = 882
EXPECTED_GRAPH_THRESHOLD = 0.410
EXPECTED_NODE_THRESHOLD = 0.765
EXPECTED_VECTOR_COUNT = 24


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], str]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "model_state"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value and all(
                isinstance(v, torch.Tensor) for v in value.values()
            ):
                return value, key
        if checkpoint and all(
            isinstance(v, torch.Tensor) for v in checkpoint.values()
        ):
            return checkpoint, "checkpoint_root"
    raise ValueError("Could not locate tensor state_dict in checkpoint.")


def decode_strings(values: np.ndarray) -> np.ndarray:
    result = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return np.asarray(result, dtype=str)


def profile_from_run_id(run_id: str) -> str:
    match = re.search(r"-P(bursty|stream|mixed)-", run_id)
    return match.group(1) if match else "unknown"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def float32_hex_lines(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
    words = values.view(np.uint32)
    return "".join(f"{int(word):08x}\n" for word in words)


def bitmap_from_binary(values: np.ndarray) -> int:
    bitmap = 0
    for index, value in enumerate(np.asarray(values).reshape(-1)):
        if int(value) != 0:
            bitmap |= 1 << index
    return bitmap


def select_unique(
    candidates: np.ndarray,
    quota: int,
    selected: list[int],
    selected_set: set[int],
    category: str,
    category_for_index: dict[int, str],
) -> None:
    taken = 0
    for index in candidates.tolist():
        index = int(index)
        if index in selected_set:
            continue
        selected.append(index)
        selected_set.add(index)
        category_for_index[index] = category
        taken += 1
        if taken >= quota:
            break


def prioritize_run_diversity(
    indices: np.ndarray,
    run_id: np.ndarray,
    score: np.ndarray,
    descending: bool,
) -> np.ndarray:
    order = indices[np.argsort(score[indices])]
    if descending:
        order = order[::-1]

    first_per_run = []
    remaining = []
    seen_runs = set()
    for index in order.tolist():
        current_run = str(run_id[index])
        if current_run not in seen_runs:
            first_per_run.append(index)
            seen_runs.add(current_run)
        else:
            remaining.append(index)
    return np.asarray(first_per_run + remaining, dtype=np.int64)


def select_vectors(
    *,
    y_graph: np.ndarray,
    y_node: np.ndarray,
    run_id: np.ndarray,
    strength: np.ndarray,
    graph_prob: np.ndarray,
    node_prob: np.ndarray,
    graph_threshold: float,
    node_threshold: float,
    count: int,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    graph_pred = (graph_prob >= graph_threshold).astype(np.int64)
    node_pred = (node_prob >= node_threshold).astype(np.int64)
    exact = np.all(node_pred == y_node, axis=1)
    attacker_count = y_node.sum(axis=1)
    profile = np.asarray(
        [profile_from_run_id(value) for value in run_id],
        dtype=str,
    )

    selected: list[int] = []
    selected_set: set[int] = set()
    category_for_index: dict[int, str] = {}

    normal = np.flatnonzero(y_graph == 0)
    attack = np.flatnonzero(y_graph == 1)

    # 2 easy normal examples.
    easy_normal = normal[np.argsort(graph_prob[normal])]
    select_unique(
        easy_normal, 2, selected, selected_set,
        "easy_normal", category_for_index,
    )

    # 4 hard normal examples; force the historically hardest run first.
    hard_run = "N-2-6-10-14-Pmixed-R13-V3"
    hard_run_indices = np.flatnonzero(
        (y_graph == 0) & (run_id == hard_run)
    )
    hard_run_indices = hard_run_indices[
        np.argsort(graph_prob[hard_run_indices])[::-1]
    ]
    select_unique(
        hard_run_indices, 1, selected, selected_set,
        "hard_normal_known_run", category_for_index,
    )
    hard_normal = prioritize_run_diversity(
        normal, run_id, graph_prob, descending=True
    )
    select_unique(
        hard_normal, 3, selected, selected_set,
        "hard_normal", category_for_index,
    )

    # 3 correctly localized strong single-attacker cases.
    strong_single = np.flatnonzero(
        (y_graph == 1)
        & (strength >= 63)
        & (attacker_count == 1)
        & exact
    )
    strong_single = prioritize_run_diversity(
        strong_single, run_id, graph_prob, descending=True
    )
    select_unique(
        strong_single, 3, selected, selected_set,
        "strong_single_exact", category_for_index,
    )

    # 4 weak S20 cases, prioritizing threshold-boundary and run diversity.
    weak = np.flatnonzero((y_graph == 1) & np.isclose(strength, 20.0))
    weak_score = np.abs(graph_prob - graph_threshold)
    weak = prioritize_run_diversity(
        weak, run_id, -weak_score, descending=True
    )
    select_unique(
        weak, 4, selected, selected_set,
        "weak_strength20", category_for_index,
    )

    # 2 each from the three attack profiles, emphasizing difficult samples.
    for current_profile in ("bursty", "stream", "mixed"):
        candidates = np.flatnonzero(
            (y_graph == 1) & (profile == current_profile)
        )
        candidates = prioritize_run_diversity(
            candidates, run_id, graph_prob, descending=False
        )
        select_unique(
            candidates, 2, selected, selected_set,
            f"profile_{current_profile}", category_for_index,
        )

    # 2 multi-attacker cases, highest attacker count first.
    multi = np.flatnonzero((y_graph == 1) & (attacker_count >= 2))
    multi_order = sorted(
        multi.tolist(),
        key=lambda index: (
            -int(attacker_count[index]),
            float(graph_prob[index]),
            str(run_id[index]),
            int(index),
        ),
    )
    select_unique(
        np.asarray(multi_order, dtype=np.int64),
        2,
        selected,
        selected_set,
        "multi_attacker",
        category_for_index,
    )

    # One graph FP.
    graph_fp = np.flatnonzero((y_graph == 0) & (graph_pred == 1))
    graph_fp = graph_fp[np.argsort(graph_prob[graph_fp])[::-1]]
    select_unique(
        graph_fp, 1, selected, selected_set,
        "graph_false_positive", category_for_index,
    )

    # One graph FN.
    graph_fn = np.flatnonzero((y_graph == 1) & (graph_pred == 0))
    graph_fn = graph_fn[np.argsort(graph_prob[graph_fn])]
    select_unique(
        graph_fn, 1, selected, selected_set,
        "graph_false_negative", category_for_index,
    )

    # One localization error.
    localization_error = np.flatnonzero((y_graph == 1) & (~exact))
    loc_distance = np.abs(node_prob - y_node).mean(axis=1)
    localization_error = localization_error[
        np.argsort(loc_distance[localization_error])[::-1]
    ]
    select_unique(
        localization_error, 1, selected, selected_set,
        "localization_error", category_for_index,
    )

    # Fill any shortfall deterministically with diverse validation examples.
    if len(selected) < count:
        all_indices = np.arange(len(y_graph), dtype=np.int64)
        fallback_order = sorted(
            all_indices.tolist(),
            key=lambda index: (
                int(y_graph[index]),
                str(run_id[index]),
                int(index),
            ),
        )
        select_unique(
            np.asarray(fallback_order, dtype=np.int64),
            count - len(selected),
            selected,
            selected_set,
            "fallback_balanced", category_for_index,
        )

    if len(selected) < count:
        raise RuntimeError(
            f"Could select only {len(selected)} unique vectors, expected {count}"
        )

    selected = selected[:count]
    categories = [category_for_index[index] for index in selected]
    coverage: dict[str, int] = {}
    for category in categories:
        coverage[category] = coverage.get(category, 0) + 1

    return (
        np.asarray(selected, dtype=np.int64),
        categories,
        coverage,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--h0-dir", required=True, type=Path)
    parser.add_argument("--threshold-report", required=True, type=Path)
    parser.add_argument(
        "--validation-predictions",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--vector-count",
        type=int,
        default=EXPECTED_VECTOR_COUNT,
    )
    parser.add_argument(
        "--reference-batch-size",
        type=int,
        default=512,
        help=(
            "Replay the frozen validation inference batch context. "
            "G1.3 A1 predictions were generated with batch size 512."
        ),
    )
    args = parser.parse_args()

    source = args.source.resolve()
    model_dir = args.model_dir.resolve()
    data_dir = args.data_dir.resolve()
    h0_dir = args.h0_dir.resolve()
    threshold_report_path = args.threshold_report.resolve()
    prediction_path = args.validation_predictions.resolve()
    output_dir = args.output_dir.resolve()

    checkpoint_path = model_dir / "best_model.pt"
    splits_path = model_dir / "splits.npz"
    x_path = data_dir / "x.npy"
    edge_path = data_dir / "edge_index.npy"
    feature_path = data_dir / "feature_cols.npy"
    h0_release_path = h0_dir / "manifests" / "release_manifest.json"

    required = {
        "source": source,
        "checkpoint": checkpoint_path,
        "splits": splits_path,
        "x": x_path,
        "edge_index": edge_path,
        "feature_cols": feature_path,
        "h0_release": h0_release_path,
        "threshold_report": threshold_report_path,
        "validation_predictions": prediction_path,
    }
    for label, path in required.items():
        if not path.is_file():
            raise SystemExit(f"STOP: missing {label}: {path}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"STOP: H1A output directory non-empty: {output_dir}")
    if args.vector_count != EXPECTED_VECTOR_COUNT:
        raise SystemExit(
            f"STOP: this release requires exactly {EXPECTED_VECTOR_COUNT} vectors"
        )

    h0_release = load_json(h0_release_path)
    threshold_report = load_json(threshold_report_path)

    prerequisite_checks = {
        "source_hash_frozen": sha256(source) == EXPECTED_SOURCE_SHA256,
        "checkpoint_hash_frozen": (
            sha256(checkpoint_path) == EXPECTED_CHECKPOINT_SHA256
        ),
        "h0_stage": h0_release.get("stage") == "H0",
        "h0_passed": (
            h0_release.get("status")
            == "V3_A1_RTL_BRINGUP_PACKAGE_CREATED"
        ),
        "h0_weights_provisional": (
            h0_release.get("weights_status")
            == "PROVISIONAL_V3_BRINGUP_ONLY"
        ),
        "g1_3_validation_only": (
            threshold_report.get("stage") == "G1.3"
            and threshold_report.get("protocol", {}).get("validation_only")
            is True
        ),
        "g1_3_no_test_inference": (
            threshold_report.get("protocol", {})
            .get("test_inference_performed") is False
        ),
    }
    if not all(prerequisite_checks.values()):
        failed = [
            key for key, value in prerequisite_checks.items()
            if not value
        ]
        raise SystemExit(f"STOP: H1A prerequisites failed: {failed}")

    graph_threshold = float(
        threshold_report["a1"]["graph_threshold"]
    )
    node_threshold = float(
        threshold_report["a1"]["node_threshold"]
    )
    if not math.isclose(
        graph_threshold,
        EXPECTED_GRAPH_THRESHOLD,
        abs_tol=1e-12,
    ):
        raise SystemExit(
            f"STOP: unexpected graph threshold {graph_threshold}"
        )
    if not math.isclose(
        node_threshold,
        EXPECTED_NODE_THRESHOLD,
        abs_tol=1e-12,
    ):
        raise SystemExit(
            f"STOP: unexpected node threshold {node_threshold}"
        )

    with np.load(prediction_path, allow_pickle=True) as prediction_npz:
        predictions = {
            key: np.asarray(prediction_npz[key])
            for key in prediction_npz.files
        }

    prediction_required = {
        "sample_idx",
        "y_graph",
        "y_node",
        "run_id",
        "strength",
        "graph_prob",
        "node_prob",
    }
    if not prediction_required.issubset(predictions):
        missing = sorted(prediction_required - set(predictions))
        raise SystemExit(
            f"STOP: validation prediction fields missing: {missing}"
        )

    sample_idx = np.asarray(predictions["sample_idx"], dtype=np.int64)
    y_graph = np.asarray(predictions["y_graph"], dtype=np.int64)
    y_node = np.asarray(predictions["y_node"], dtype=np.int64)
    run_id = decode_strings(predictions["run_id"])
    strength = np.asarray(predictions["strength"], dtype=np.float64)
    graph_prob_reference = np.asarray(
        predictions["graph_prob"],
        dtype=np.float32,
    )
    node_prob_reference = np.asarray(
        predictions["node_prob"],
        dtype=np.float32,
    )

    with np.load(splits_path) as split_npz:
        validation_idx = np.asarray(
            split_npz["val_idx"],
            dtype=np.int64,
        )

    alignment_checks = {
        "validation_rows_42809": len(sample_idx) == 42809,
        "sample_idx_matches_frozen_validation_split": np.array_equal(
            sample_idx,
            validation_idx,
        ),
        "y_graph_shape": tuple(y_graph.shape) == (42809,),
        "y_node_shape": tuple(y_node.shape) == (42809, 16),
        "graph_prob_shape": (
            tuple(graph_prob_reference.shape) == (42809,)
        ),
        "node_prob_shape": (
            tuple(node_prob_reference.shape) == (42809, 16)
        ),
    }
    if not all(alignment_checks.values()):
        failed = [
            key for key, value in alignment_checks.items()
            if not value
        ]
        raise SystemExit(f"STOP: validation alignment failed: {failed}")

    selected_local, categories, coverage = select_vectors(
        y_graph=y_graph,
        y_node=y_node,
        run_id=run_id,
        strength=strength,
        graph_prob=graph_prob_reference,
        node_prob=node_prob_reference,
        graph_threshold=graph_threshold,
        node_threshold=node_threshold,
        count=args.vector_count,
    )
    selected_global = sample_idx[selected_local]

    x = np.load(x_path, mmap_mode="r")
    edge_index = np.load(edge_path).astype(np.int64)
    feature_cols = np.load(feature_path, allow_pickle=True)
    feature_names = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(feature_cols).reshape(-1)
    ]

    module = load_module(source, "v3_a1_h1a_golden_source")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict, state_location = extract_state_dict(checkpoint)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

    model = module.TemporalGCN(
        input_features=24,
        temporal_dim=int(saved_args.get("temporal_dim", 8)),
        gcn_hidden=int(saved_args.get("gcn_hidden", 16)),
        gcn_out=int(saved_args.get("gcn_out", 8)),
    )
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise SystemExit("STOP: strict A1 checkpoint load failed")

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise SystemExit(
            f"STOP: expected 882 parameters, found {parameter_count}"
        )

    conv_layers = [
        layer
        for layer in model.temporal.modules()
        if isinstance(layer, torch.nn.Conv1d)
    ]
    if len(conv_layers) != 1:
        raise SystemExit(
            f"STOP: expected exactly one temporal Conv1d, found {len(conv_layers)}"
        )
    conv_layer = conv_layers[0]

    temporal_forward_source = inspect.getsource(
        module.TemporalEncoder.forward
    )
    model_forward_source = inspect.getsource(
        module.TemporalGCN.forward
    )

    if not torch.cuda.is_available():
        raise SystemExit(
            "STOP: CUDA is required because the frozen G1.3 validation "
            "predictions were generated by CUDA FP32 inference."
        )

    reference_batch_size = int(args.reference_batch_size)
    if reference_batch_size != 512:
        raise SystemExit(
            "STOP: H1A-v3 requires --reference-batch-size 512 to reproduce "
            "the frozen G1.3 validation inference context."
        )

    # Group selected validation-local indices by the exact original
    # validation batch that contained them.
    replay_groups: dict[int, list[tuple[int, int]]] = {}
    for output_position, local_index in enumerate(selected_local.tolist()):
        batch_start = (int(local_index) // reference_batch_size) * reference_batch_size
        replay_groups.setdefault(batch_start, []).append(
            (output_position, int(local_index))
        )

    tensor_names = (
        "input_tensor",
        "conv1d_pre_activation",
        "conv1d_post_activation",
        "temporal_pooled",
        "gcn1_pre_activation",
        "gcn1_post_activation",
        "gcn2_pre_activation",
        "gcn2_post_activation",
        "node_logits",
        "node_probabilities",
        "graph_embedding",
        "graph_logits",
        "graph_probabilities",
    )
    canonical_slots: dict[str, list[np.ndarray | None]] = {
        name: [None] * args.vector_count for name in tensor_names
    }

    canonical_internal_checks = {
        "manual_temporal_pool_matches": True,
        "manual_graph_logits_match_full_forward": True,
        "manual_node_logits_match_full_forward": True,
        "all_outputs_finite": True,
    }

    canonical_device = torch.device("cuda")
    model.to(canonical_device)
    model.eval()
    adjacency_cuda = module.build_normalized_adjacency(
        edge_index,
        16,
    ).to(canonical_device)

    conv_layer_cuda = [
        layer
        for layer in model.temporal.modules()
        if isinstance(layer, torch.nn.Conv1d)
    ][0]

    for batch_start in sorted(replay_groups):
        batch_end = min(
            batch_start + reference_batch_size,
            len(sample_idx),
        )
        batch_global_indices = sample_idx[batch_start:batch_end]
        batch_input = torch.from_numpy(
            np.array(
                x[batch_global_indices],
                dtype=np.float32,
                copy=True,
            )
        ).to(canonical_device)

        captured_conv: list[torch.Tensor] = []

        def capture_conv(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured_conv.append(output.detach())

        hook = conv_layer_cuda.register_forward_hook(capture_conv)

        with torch.no_grad():
            temporal_pooled_batch = model.temporal(batch_input)
            if len(captured_conv) != 1:
                raise SystemExit(
                    "STOP: expected one Conv1D hook output in CUDA replay, "
                    f"got {len(captured_conv)}"
                )

            conv_pre_flat = captured_conv[0]
            conv_post_flat = torch.relu(conv_pre_flat)

            current_batch_size = batch_input.shape[0]
            router_count = batch_input.shape[1]
            conv_pre_batch = conv_pre_flat.reshape(
                current_batch_size,
                router_count,
                conv_pre_flat.shape[1],
                conv_pre_flat.shape[2],
            )
            conv_post_batch = conv_post_flat.reshape(
                current_batch_size,
                router_count,
                conv_post_flat.shape[1],
                conv_post_flat.shape[2],
            )
            manual_temporal_pool = conv_post_batch.max(dim=-1).values

            gcn1_pre_batch = model.gcn1(
                temporal_pooled_batch,
                adjacency_cuda,
            )
            gcn1_post_batch = torch.relu(gcn1_pre_batch)

            gcn2_pre_batch = model.gcn2(
                gcn1_post_batch,
                adjacency_cuda,
            )
            gcn2_post_batch = torch.relu(gcn2_pre_batch)

            node_logits_batch = model.node_head(
                gcn2_post_batch
            ).squeeze(-1)
            node_prob_batch = torch.sigmoid(node_logits_batch)

            graph_embedding_batch = gcn2_post_batch.mean(dim=1)
            graph_logits_batch = model.graph_head(
                graph_embedding_batch
            ).squeeze(-1)
            graph_prob_batch = torch.sigmoid(graph_logits_batch)

            full_graph_logits, full_node_logits = model(
                batch_input,
                adjacency_cuda,
            )

        hook.remove()

        canonical_internal_checks["manual_temporal_pool_matches"] &= torch.equal(
            temporal_pooled_batch,
            manual_temporal_pool,
        )
        canonical_internal_checks[
            "manual_graph_logits_match_full_forward"
        ] &= torch.equal(graph_logits_batch, full_graph_logits)
        canonical_internal_checks[
            "manual_node_logits_match_full_forward"
        ] &= torch.equal(node_logits_batch, full_node_logits)
        canonical_internal_checks["all_outputs_finite"] &= all(
            bool(torch.isfinite(value).all().item())
            for value in (
                batch_input,
                conv_pre_batch,
                conv_post_batch,
                temporal_pooled_batch,
                gcn1_pre_batch,
                gcn1_post_batch,
                gcn2_pre_batch,
                gcn2_post_batch,
                node_logits_batch,
                node_prob_batch,
                graph_embedding_batch,
                graph_logits_batch,
                graph_prob_batch,
            )
        )

        batch_tensors = {
            "input_tensor": batch_input,
            "conv1d_pre_activation": conv_pre_batch,
            "conv1d_post_activation": conv_post_batch,
            "temporal_pooled": temporal_pooled_batch,
            "gcn1_pre_activation": gcn1_pre_batch,
            "gcn1_post_activation": gcn1_post_batch,
            "gcn2_pre_activation": gcn2_pre_batch,
            "gcn2_post_activation": gcn2_post_batch,
            "node_logits": node_logits_batch,
            "node_probabilities": node_prob_batch,
            "graph_embedding": graph_embedding_batch,
            "graph_logits": graph_logits_batch,
            "graph_probabilities": graph_prob_batch,
        }

        for output_position, local_index in replay_groups[batch_start]:
            within_batch = local_index - batch_start
            for tensor_name, batch_tensor in batch_tensors.items():
                canonical_slots[tensor_name][output_position] = (
                    batch_tensor[within_batch]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )

    if any(
        value is None
        for slots in canonical_slots.values()
        for value in slots
    ):
        raise SystemExit("STOP: one or more selected canonical tensors were not filled")

    tensor_batch = {
        name: np.stack(slots, axis=0).astype(np.float32, copy=False)
        for name, slots in canonical_slots.items()
    }

    canonical_graph_prob = tensor_batch["graph_probabilities"]
    canonical_node_prob = tensor_batch["node_probabilities"]
    saved_graph_prob = graph_prob_reference[selected_local]
    saved_node_prob = node_prob_reference[selected_local]

    graph_exact_match = np.array_equal(
        canonical_graph_prob,
        saved_graph_prob,
    )
    node_exact_match = np.array_equal(
        canonical_node_prob,
        saved_node_prob,
    )

    graph_max_abs_error = float(
        np.max(
            np.abs(
                canonical_graph_prob.astype(np.float64)
                - saved_graph_prob.astype(np.float64)
            )
        )
    )
    node_max_abs_error = float(
        np.max(
            np.abs(
                canonical_node_prob.astype(np.float64)
                - saved_node_prob.astype(np.float64)
            )
        )
    )

    # Secondary CPU replay using the same original validation batches. These
    # values are diagnostic only and are not the golden reference.
    cpu_model = module.TemporalGCN(
        input_features=24,
        temporal_dim=int(saved_args.get("temporal_dim", 8)),
        gcn_hidden=int(saved_args.get("gcn_hidden", 16)),
        gcn_out=int(saved_args.get("gcn_out", 8)),
    )
    cpu_load = cpu_model.load_state_dict(state_dict, strict=True)
    if cpu_load.missing_keys or cpu_load.unexpected_keys:
        raise SystemExit("STOP: strict CPU shadow checkpoint load failed")
    cpu_model.eval()
    adjacency_cpu = module.build_normalized_adjacency(
        edge_index,
        16,
    ).cpu()

    cpu_graph_slots: list[np.ndarray | None] = [None] * args.vector_count
    cpu_node_slots: list[np.ndarray | None] = [None] * args.vector_count

    for batch_start in sorted(replay_groups):
        batch_end = min(
            batch_start + reference_batch_size,
            len(sample_idx),
        )
        batch_global_indices = sample_idx[batch_start:batch_end]
        batch_input_cpu = torch.from_numpy(
            np.array(
                x[batch_global_indices],
                dtype=np.float32,
                copy=True,
            )
        )
        with torch.no_grad():
            cpu_graph_logits, cpu_node_logits = cpu_model(
                batch_input_cpu,
                adjacency_cpu,
            )
            cpu_graph_prob_batch = torch.sigmoid(
                cpu_graph_logits
            ).cpu().numpy()
            cpu_node_prob_batch = torch.sigmoid(
                cpu_node_logits
            ).cpu().numpy()

        for output_position, local_index in replay_groups[batch_start]:
            within_batch = local_index - batch_start
            cpu_graph_slots[output_position] = np.asarray(
                cpu_graph_prob_batch[within_batch],
                dtype=np.float32,
            )
            cpu_node_slots[output_position] = np.asarray(
                cpu_node_prob_batch[within_batch],
                dtype=np.float32,
            )

    cpu_graph_prob = np.stack(cpu_graph_slots, axis=0).astype(np.float32)
    cpu_node_prob = np.stack(cpu_node_slots, axis=0).astype(np.float32)

    cuda_graph_decision = (
        canonical_graph_prob >= graph_threshold
    ).astype(np.uint8)
    saved_graph_decision = (
        saved_graph_prob >= graph_threshold
    ).astype(np.uint8)
    cpu_graph_decision = (
        cpu_graph_prob >= graph_threshold
    ).astype(np.uint8)

    cuda_node_decision = (
        canonical_node_prob >= node_threshold
    ).astype(np.uint8)
    saved_node_decision = (
        saved_node_prob >= node_threshold
    ).astype(np.uint8)
    cpu_node_decision = (
        cpu_node_prob >= node_threshold
    ).astype(np.uint8)

    cuda_top1 = np.argmax(canonical_node_prob, axis=1)
    saved_top1 = np.argmax(saved_node_prob, axis=1)
    cpu_top1 = np.argmax(cpu_node_prob, axis=1)

    cpu_cuda_graph_max_abs_error = float(
        np.max(
            np.abs(
                cpu_graph_prob.astype(np.float64)
                - canonical_graph_prob.astype(np.float64)
            )
        )
    )
    cpu_cuda_node_max_abs_error = float(
        np.max(
            np.abs(
                cpu_node_prob.astype(np.float64)
                - canonical_node_prob.astype(np.float64)
            )
        )
    )

    cross_device_sensitive = (
        (cpu_graph_decision != cuda_graph_decision)
        | np.any(cpu_node_decision != cuda_node_decision, axis=1)
        | (cpu_top1 != cuda_top1)
    )

    forward_checks = {
        **canonical_internal_checks,
        "canonical_graph_prob_exactly_matches_saved_cuda": graph_exact_match,
        "canonical_node_prob_exactly_matches_saved_cuda": node_exact_match,
        "graph_threshold_decisions_match_saved_cuda": np.array_equal(
            cuda_graph_decision,
            saved_graph_decision,
        ),
        "node_threshold_decisions_match_saved_cuda": np.array_equal(
            cuda_node_decision,
            saved_node_decision,
        ),
        "top1_router_matches_saved_cuda": np.array_equal(
            cuda_top1,
            saved_top1,
        ),
    }
    if not all(forward_checks.values()):
        failed = [
            key for key, value in forward_checks.items()
            if not value
        ]
        raise SystemExit(f"STOP: H1A-v3 forward checks failed: {failed}")

    output_dir.mkdir(parents=True, exist_ok=False)
    vectors_dir = output_dir / "vectors"
    combined_dir = output_dir / "combined"
    docs_dir = output_dir / "docs"
    manifests_dir = output_dir / "manifests"
    for path in (vectors_dir, combined_dir, docs_dir, manifests_dir):
        path.mkdir(parents=True, exist_ok=True)

    combined_metadata: dict[str, np.ndarray] = {
        "validation_local_index": selected_local,
        "global_sample_index": selected_global,
        "category": np.asarray(categories, dtype=str),
        "run_id": run_id[selected_local],
        "strength": strength[selected_local],
        "true_graph_label": y_graph[selected_local],
        "true_attacker_bitmap": np.asarray(
            [
                bitmap_from_binary(row)
                for row in y_node[selected_local]
            ],
            dtype=np.uint16,
        ),
        "true_node_labels": y_node[selected_local],
        "graph_threshold": np.full(
            args.vector_count,
            graph_threshold,
            dtype=np.float32,
        ),
        "node_threshold": np.full(
            args.vector_count,
            node_threshold,
            dtype=np.float32,
        ),
    }

    graph_decision = cuda_graph_decision
    node_decision = cuda_node_decision
    node_bitmap = np.asarray(
        [
            bitmap_from_binary(row)
            for row in node_decision
        ],
        dtype=np.uint16,
    )
    top1_router = cuda_top1.astype(np.uint8)

    combined_metadata.update(
        {
            "graph_decision": graph_decision,
            "node_decision": node_decision,
            "predicted_router_bitmap": node_bitmap,
            "top1_router": top1_router,
            "cpu_shadow_graph_probability": cpu_graph_prob,
            "cpu_shadow_node_probability": cpu_node_prob,
            "cpu_shadow_graph_decision": cpu_graph_decision,
            "cpu_shadow_node_decision": cpu_node_decision,
            "cpu_shadow_top1_router": cpu_top1.astype(np.uint8),
            "cross_device_sensitive": cross_device_sensitive.astype(np.uint8),
        }
    )

    np.savez_compressed(
        combined_dir / "golden_vectors_fp32.npz",
        **tensor_batch,
        **combined_metadata,
    )

    manifest_rows: list[dict[str, Any]] = []
    tensor_shape_rows: list[dict[str, Any]] = []

    for name, array in tensor_batch.items():
        tensor_shape_rows.append(
            {
                "tensor": name,
                "batch_shape": "x".join(str(v) for v in array.shape),
                "per_vector_shape": "x".join(
                    str(v) for v in array.shape[1:]
                ),
                "dtype": str(array.dtype),
                "flatten_order": "C row-major",
            }
        )
    write_csv(
        docs_dir / "tensor_shapes.csv",
        tensor_shape_rows,
    )

    for vector_position in range(args.vector_count):
        vector_id = f"v{vector_position:02d}"
        vector_dir = vectors_dir / vector_id
        vector_dir.mkdir(parents=True, exist_ok=False)

        per_vector_tensors = {
            name: np.asarray(array[vector_position])
            for name, array in tensor_batch.items()
        }

        np.savez_compressed(
            vector_dir / f"{vector_id}_all_tensors.npz",
            **per_vector_tensors,
        )

        tensor_manifest = []
        for tensor_name, tensor in per_vector_tensors.items():
            tensor = np.ascontiguousarray(tensor, dtype=np.float32)
            np.save(vector_dir / f"{tensor_name}.npy", tensor)
            (vector_dir / f"{tensor_name}.hex").write_text(
                float32_hex_lines(tensor),
                encoding="utf-8",
            )
            tensor_manifest.append(
                {
                    "tensor": tensor_name,
                    "shape": list(tensor.shape),
                    "dtype": "float32",
                    "numel": int(tensor.size),
                    "flatten_order": "C row-major",
                    "hex_word_format": (
                        "one IEEE-754 binary32 word per line, "
                        "8 hexadecimal digits"
                    ),
                    "minimum": float(tensor.min()),
                    "maximum": float(tensor.max()),
                }
            )

        true_nodes = y_node[selected_local[vector_position]]
        metadata = {
            "vector_id": vector_id,
            "selection_category": categories[vector_position],
            "validation_local_index": int(
                selected_local[vector_position]
            ),
            "global_sample_index": int(
                selected_global[vector_position]
            ),
            "run_id": str(run_id[selected_local[vector_position]]),
            "profile": profile_from_run_id(
                str(run_id[selected_local[vector_position]])
            ),
            "strength": float(
                strength[selected_local[vector_position]]
            ),
            "attacker_count": int(true_nodes.sum()),
            "true_graph_label": int(
                y_graph[selected_local[vector_position]]
            ),
            "true_attacker_bitmap_hex": (
                f"0x{bitmap_from_binary(true_nodes):04x}"
            ),
            "graph_threshold": graph_threshold,
            "node_threshold": node_threshold,
            "graph_logit": float(tensor_batch["graph_logits"][vector_position]),
            "graph_probability": float(canonical_graph_prob[vector_position]),
            "graph_decision": int(graph_decision[vector_position]),
            "predicted_router_bitmap_hex": (
                f"0x{int(node_bitmap[vector_position]):04x}"
            ),
            "top1_router": int(top1_router[vector_position]),
            "exact_localization": bool(
                np.array_equal(
                    node_decision[vector_position],
                    true_nodes,
                )
            ),
            "canonical_reference_device": "CUDA_FP32",
            "reference_batch_size": reference_batch_size,
            "cpu_shadow_graph_probability": float(
                cpu_graph_prob[vector_position]
            ),
            "cpu_shadow_top1_router": int(cpu_top1[vector_position]),
            "cross_device_sensitive": bool(
                cross_device_sensitive[vector_position]
            ),
            "tensor_manifest": tensor_manifest,
        }
        (vector_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        manifest_rows.append(
            {
                "vector_id": vector_id,
                "category": categories[vector_position],
                "validation_local_index": int(
                    selected_local[vector_position]
                ),
                "global_sample_index": int(
                    selected_global[vector_position]
                ),
                "run_id": str(
                    run_id[selected_local[vector_position]]
                ),
                "profile": metadata["profile"],
                "strength": metadata["strength"],
                "attacker_count": metadata["attacker_count"],
                "true_graph_label": metadata["true_graph_label"],
                "true_attacker_bitmap_hex": (
                    metadata["true_attacker_bitmap_hex"]
                ),
                "graph_probability": metadata["graph_probability"],
                "graph_decision": metadata["graph_decision"],
                "predicted_router_bitmap_hex": (
                    metadata["predicted_router_bitmap_hex"]
                ),
                "top1_router": metadata["top1_router"],
                "exact_localization": metadata["exact_localization"],
                "cross_device_sensitive": metadata["cross_device_sensitive"],
            }
        )

    write_csv(output_dir / "golden_vector_manifest.csv", manifest_rows)

    feature_range_rows = []
    selected_input = tensor_batch["input_tensor"]
    for feature_index, feature_name in enumerate(feature_names):
        values = selected_input[..., feature_index].reshape(-1)
        feature_range_rows.append(
            {
                "feature_index": feature_index,
                "feature_name": feature_name,
                "selected_vector_minimum": float(values.min()),
                "selected_vector_maximum": float(values.max()),
                "selected_vector_mean": float(values.mean()),
                "selected_vector_std": float(values.std()),
                "selected_vector_zero_fraction": float(
                    np.mean(values == 0)
                ),
                "normalization_equation_status": (
                    "NOT_INFERRED_FROM_PROCESSED_VALUES"
                ),
            }
        )
    write_csv(
        docs_dir / "selected_vector_feature_ranges.csv",
        feature_range_rows,
    )

    architecture_details = {
        "conv1d": {
            "in_channels": conv_layer.in_channels,
            "out_channels": conv_layer.out_channels,
            "kernel_size": list(conv_layer.kernel_size),
            "stride": list(conv_layer.stride),
            "padding": list(conv_layer.padding),
            "dilation": list(conv_layer.dilation),
            "groups": conv_layer.groups,
            "bias": conv_layer.bias is not None,
            "selected_output_shape": list(tensor_batch["conv1d_pre_activation"].shape),
        },
        "temporal_pooled_shape": list(tensor_batch["temporal_pooled"].shape),
        "gcn1_pre_shape": list(tensor_batch["gcn1_pre_activation"].shape),
        "gcn2_pre_shape": list(tensor_batch["gcn2_pre_activation"].shape),
        "graph_embedding_shape": list(tensor_batch["graph_embedding"].shape),
        "node_logits_shape": list(tensor_batch["node_logits"].shape),
        "graph_logits_shape": list(tensor_batch["graph_logits"].shape),
        "temporal_forward_source": temporal_forward_source,
        "model_forward_source": model_forward_source,
    }
    (docs_dir / "exact_forward_contract.json").write_text(
        json.dumps(architecture_details, indent=2) + "\n",
        encoding="utf-8",
    )

    readme = f"""# H1A Validation-Only FP32 Golden Vectors

Status: **V3 RTL bring-up only; not final V4 vectors**

Vector count: {args.vector_count}

Source split: frozen chronological validation split only.

No test data was accessed.

## Software tensor order

Input tensor per vector: `[router, epoch, feature] = [16, 8, 24]`.

All `.hex` files flatten tensors in NumPy C row-major order. Each line is one
IEEE-754 binary32 bit pattern written as eight hexadecimal digits.

## Thresholds

- Graph probability threshold: {graph_threshold:.3f}
- Node probability threshold: {node_threshold:.3f}

These are provisional V3 validation thresholds.

## Canonical floating-point reference

Layer-by-layer golden tensors are generated using CUDA FP32 because the frozen
G1.3 validation predictions and selected thresholds were produced using CUDA
FP32. Each selected sample is replayed inside its original validation batch
context using batch size 512 before its tensors are extracted.

Fresh CUDA probabilities must match the saved G1.3 probabilities bit-for-bit.

CPU FP32 final probabilities are exported only as a portability shadow. Any
vector whose graph decision, node bitmap, or Top-1 router differs between CPU
and CUDA is marked `cross_device_sensitive` and must be treated as a boundary
stress vector rather than an exact cross-platform decision oracle.

## Important normalization limitation

This stage exports exact processed input values and observed selected-vector
ranges. It does not infer normalization equations from those values.

H1B must recover normalization equations, clipping constants, missing-value
semantics, and mask semantics from the authoritative V3 dataset builder or raw
trace pipeline before the feature-extraction RTL is frozen.
"""
    (output_dir / "README.md").write_text(
        readme,
        encoding="utf-8",
    )

    artifact_rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            artifact_rows.append(
                {
                    "relative_path": str(
                        path.relative_to(output_dir)
                    ),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    artifact_hash_path = manifests_dir / "h1a_artifact_hashes.csv"
    write_csv(artifact_hash_path, artifact_rows)

    release = {
        "stage": "H1A",
        "status": "V3_A1_FP32_GOLDEN_VECTORS_CREATED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(source),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_state_location": state_location,
        "parameter_count": parameter_count,
        "validation_only": True,
        "test_accessed": False,
        "vector_count": args.vector_count,
        "selection_coverage": coverage,
        "graph_threshold": graph_threshold,
        "node_threshold": node_threshold,
        "forward_checks": forward_checks,
        "reference_policy": {
            "canonical_golden_device": "CUDA_FP32",
            "saved_validation_prediction_device": "CUDA_FP32",
            "reference_batch_size": reference_batch_size,
            "exact_saved_probability_match_required": True,
            "graph_max_abs_error_cuda_vs_saved": graph_max_abs_error,
            "node_max_abs_error_cuda_vs_saved": node_max_abs_error,
            "cpu_shadow_graph_max_abs_error": (
                cpu_cuda_graph_max_abs_error
            ),
            "cpu_shadow_node_max_abs_error": (
                cpu_cuda_node_max_abs_error
            ),
            "cross_device_sensitive_vector_count": int(
                np.sum(cross_device_sensitive)
            ),
            "required_saved_output_agreement": [
                "graph probability bit pattern",
                "16 node probability bit patterns",
                "graph threshold decision",
                "16-node threshold decision",
                "Top-1 router",
            ],
        },
        "normalization_equations_status": "PENDING_H1B_PROVENANCE",
        "weights_status": "PROVISIONAL_V3_BRINGUP_ONLY",
        "vectors_status": "PROVISIONAL_V3_BRINGUP_ONLY",
        "final_v4_replacement_required": True,
        "artifact_hash_manifest": str(artifact_hash_path),
        "artifact_hash_manifest_sha256": sha256(
            artifact_hash_path
        ),
    }
    release_path = manifests_dir / "h1a_release_manifest.json"
    release_path.write_text(
        json.dumps(release, indent=2) + "\n",
        encoding="utf-8",
    )

    print("H1A V3-A1 FP32 GOLDEN VECTOR EXPORT: PASS")
    print(f"output_dir={output_dir}")
    print(f"vector_count={args.vector_count}")
    print("validation_only=True")
    print("test_accessed=False")
    print(f"graph_threshold={graph_threshold:.3f}")
    print(f"node_threshold={node_threshold:.3f}")
    print(
        "conv1d="
        f"{conv_layer.in_channels}->{conv_layer.out_channels},"
        f"kernel={tuple(conv_layer.kernel_size)},"
        f"stride={tuple(conv_layer.stride)},"
        f"padding={tuple(conv_layer.padding)},"
        f"dilation={tuple(conv_layer.dilation)},"
        f"bias={conv_layer.bias is not None}"
    )
    print(
        "conv1d_output_shape="
        f"{list(tensor_batch['conv1d_pre_activation'].shape)}"
    )
    print(
        "temporal_pooled_shape="
        f"{list(tensor_batch['temporal_pooled'].shape)}"
    )
    print(
        "gcn1_shape="
        f"{list(tensor_batch['gcn1_pre_activation'].shape)}"
    )
    print(
        "gcn2_shape="
        f"{list(tensor_batch['gcn2_pre_activation'].shape)}"
    )
    print(
        "node_logits_shape="
        f"{list(tensor_batch['node_logits'].shape)}"
    )
    print(
        "graph_embedding_shape="
        f"{list(tensor_batch['graph_embedding'].shape)}"
    )
    print("canonical_golden_device=CUDA_FP32")
    print("saved_validation_prediction_device=CUDA_FP32")
    print(f"reference_batch_size={reference_batch_size}")
    print(
        "graph_max_abs_error_cuda_vs_saved="
        f"{graph_max_abs_error:.9e}"
    )
    print(
        "node_max_abs_error_cuda_vs_saved="
        f"{node_max_abs_error:.9e}"
    )
    print(
        "saved_probability_exact_match="
        f"{graph_exact_match and node_exact_match}"
    )
    print(
        "saved_discrete_outputs_match="
        f"{forward_checks['graph_threshold_decisions_match_saved_cuda'] and forward_checks['node_threshold_decisions_match_saved_cuda'] and forward_checks['top1_router_matches_saved_cuda']}"
    )
    print(
        "cpu_shadow_graph_max_abs_error="
        f"{cpu_cuda_graph_max_abs_error:.9e}"
    )
    print(
        "cpu_shadow_node_max_abs_error="
        f"{cpu_cuda_node_max_abs_error:.9e}"
    )
    print(
        "cross_device_sensitive_vector_count="
        f"{int(np.sum(cross_device_sensitive))}"
    )
    print("normalization_equations_status=PENDING_H1B_PROVENANCE")
    print("weights_status=PROVISIONAL_V3_BRINGUP_ONLY")
    print("vectors_status=PROVISIONAL_V3_BRINGUP_ONLY")
    print(f"vector_manifest={output_dir / 'golden_vector_manifest.csv'}")
    print(f"artifact_hashes={artifact_hash_path}")
    print(f"release_manifest={release_path}")


if __name__ == "__main__":
    main()
