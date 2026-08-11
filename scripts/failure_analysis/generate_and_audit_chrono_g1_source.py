#!/usr/bin/env python3
"""
G1.0: generate the Chrono-G1 GraphConv source and perform a structural audit.

No training, no checkpoint creation, and no validation/test inference occur.
Only synthetic tensors are used after source generation.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
import py_compile
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPECTED_A1_SOURCE_SHA256 = (
    "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b"
)
EXPECTED_G0_SCRIPT_SHA256 = (
    "7b262678da7f2c3ef42c873c8507d7ebbcbabc689712cd0f8f9a6c4a062dd6c3"
)
EXPECTED_A1_PARAMETERS = 882
EXPECTED_G1_PARAMETERS = 1138
EXPECTED_PARAMETER_DELTA = 256

A1_MODEL_NAME = "Conv1D-TemporalGCN"
G1_MODEL_NAME = "Conv1D-GraphConv-TemporalGNN"

NEW_ADJACENCY_FUNCTION = (
    "def build_normalized_adjacency(\n"
    "    edge_index: np.ndarray,\n"
    "    num_nodes: int,\n"
    ") -> torch.Tensor:\n"
    "    # Binary incoming-neighbor adjacency for GraphConv.\n"
    "    # Self-loops are excluded because the root transform handles self.\n"
    "    adjacency = torch.zeros(\n"
    "        (num_nodes, num_nodes),\n"
    "        dtype=torch.float32,\n"
    "    )\n"
    "\n"
    "    sources = edge_index[0]\n"
    "    targets = edge_index[1]\n"
    "\n"
    "    for source, target in zip(sources, targets):\n"
    "        source_i = int(source)\n"
    "        target_i = int(target)\n"
    "        if source_i == target_i:\n"
    "            continue\n"
    "        adjacency[target_i, source_i] = 1.0\n"
    "\n"
    "    return adjacency\n"
)

NEW_GRAPH_LAYER_CLASS = (
    "class GraphConvLayer(nn.Module):\n"
    "    # Morris/PyG-style GraphConv with add aggregation:\n"
    "    # h_out_i = W_root h_i + W_neighbor sum(j in N(i)) h_j + b.\n"
    "\n"
    "    def __init__(self, in_dim: int, out_dim: int):\n"
    "        super().__init__()\n"
    "        self.lin_neighbor = nn.Linear(in_dim, out_dim, bias=True)\n"
    "        self.lin_root = nn.Linear(in_dim, out_dim, bias=False)\n"
    "\n"
    "    def forward(\n"
    "        self,\n"
    "        h: torch.Tensor,\n"
    "        neighbor_adjacency: torch.Tensor,\n"
    "    ) -> torch.Tensor:\n"
    "        neighbor_sum = torch.einsum(\n"
    "            \"ij,bjf->bif\",\n"
    "            neighbor_adjacency,\n"
    "            h,\n"
    "        )\n"
    "        return (\n"
    "            self.lin_neighbor(neighbor_sum)\n"
    "            + self.lin_root(h)\n"
    "        )\n"
)


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


def find_top_level_node(
    tree: ast.Module,
    node_type: type[ast.AST],
    name: str,
) -> ast.AST:
    matches = [
        node
        for node in tree.body
        if isinstance(node, node_type) and getattr(node, "name", None) == name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one top-level {node_type.__name__} named {name!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def source_segment(source: str, node: ast.AST) -> str:
    lines = source.splitlines(keepends=True)
    return "".join(lines[int(node.lineno) - 1:int(node.end_lineno)])


def splice_nodes(
    source: str,
    replacements: list[tuple[ast.AST, str]],
) -> str:
    lines = source.splitlines(keepends=True)
    for node, replacement in sorted(
        replacements,
        key=lambda item: int(item[0].lineno),
        reverse=True,
    ):
        start = int(node.lineno) - 1
        end = int(node.end_lineno)
        if not replacement.endswith("\n"):
            replacement += "\n"
        lines[start:end] = [replacement]
    return "".join(lines)


def manual_graphconv(
    layer: torch.nn.Module,
    h: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_nodes, _ = h.shape
    output = torch.empty(
        batch_size,
        num_nodes,
        layer.lin_neighbor.out_features,
        dtype=h.dtype,
    )
    for batch in range(batch_size):
        for target in range(num_nodes):
            aggregate = torch.zeros(h.shape[-1], dtype=h.dtype)
            for source in range(num_nodes):
                if adjacency[target, source] != 0:
                    aggregate += h[batch, source]
            output[batch, target] = (
                layer.lin_neighbor(aggregate)
                + layer.lin_root(h[batch, target])
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1-source", required=True, type=Path)
    parser.add_argument("--g0-report", required=True, type=Path)
    parser.add_argument("--g0-script", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--g1-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    a1_source = args.a1_source.resolve()
    g0_report_path = args.g0_report.resolve()
    g0_script = args.g0_script.resolve()
    data_dir = args.data_dir.resolve()
    g1_source = args.g1_source.resolve()
    output_dir = args.output_dir.resolve()

    required = {
        "a1_source": a1_source,
        "g0_report": g0_report_path,
        "g0_script": g0_script,
        "x": data_dir / "x.npy",
        "edge_index": data_dir / "edge_index.npy",
    }
    for label, path in required.items():
        if not path.is_file():
            raise SystemExit(f"STOP: missing {label}: {path}")

    if g1_source.exists():
        raise SystemExit(f"STOP: refusing to overwrite G1 source: {g1_source}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: G1.0 output directory already non-empty: {output_dir}"
        )

    g0 = load_json(g0_report_path)
    prerequisite_checks = {
        "g0_stage_is_g0": g0.get("stage") == "G0",
        "g0_status_passed": g0.get("status") == "EQUIVALENCE_AUDIT_PASSED",
        "g0_training_not_started": (
            g0.get("scope", {}).get("training_started") is False
        ),
        "g0_validation_not_inferred": (
            g0.get("scope", {}).get("validation_inference_performed") is False
        ),
        "g0_test_not_inferred": (
            g0.get("scope", {}).get("test_inference_performed") is False
        ),
        "g0_operator_different": (
            g0.get("decision_if_passed", {})
            .get("operator_meaningfully_different") is True
        ),
        "g0_script_hash_matches": (
            sha256(g0_script) == EXPECTED_G0_SCRIPT_SHA256
        ),
        "a1_source_hash_matches_frozen": (
            sha256(a1_source) == EXPECTED_A1_SOURCE_SHA256
        ),
        "g0_a1_source_hash_matches": (
            g0.get("frozen_a1", {}).get("source_sha256")
            == EXPECTED_A1_SOURCE_SHA256
        ),
    }
    if not all(prerequisite_checks.values()):
        failed = [name for name, passed in prerequisite_checks.items() if not passed]
        raise SystemExit(f"STOP: G1.0 prerequisite checks failed: {failed}")

    original_source = a1_source.read_text(encoding="utf-8")
    original_tree = ast.parse(original_source)

    original_function_node = find_top_level_node(
        original_tree,
        ast.FunctionDef,
        "build_normalized_adjacency",
    )
    original_class_node = find_top_level_node(
        original_tree,
        ast.ClassDef,
        "GCNLayer",
    )

    original_function = source_segment(original_source, original_function_node)
    original_class = source_segment(original_source, original_class_node)

    generated_source = splice_nodes(
        original_source,
        [
            (original_function_node, NEW_ADJACENCY_FUNCTION),
            (original_class_node, NEW_GRAPH_LAYER_CLASS),
        ],
    )

    constructor_count = generated_source.count("GCNLayer(")
    if constructor_count != 2:
        raise SystemExit(
            "STOP: expected exactly two GCNLayer constructor references; "
            f"found {constructor_count}."
        )
    generated_source = generated_source.replace(
        "GCNLayer(",
        "GraphConvLayer(",
    )

    model_name_count = generated_source.count(A1_MODEL_NAME)
    if model_name_count < 1:
        raise SystemExit(
            f"STOP: expected model identifier {A1_MODEL_NAME!r}."
        )
    generated_source = generated_source.replace(
        A1_MODEL_NAME,
        G1_MODEL_NAME,
    )
    ast.parse(generated_source)

    generated_tree = ast.parse(generated_source)
    generated_function_node = find_top_level_node(
        generated_tree,
        ast.FunctionDef,
        "build_normalized_adjacency",
    )
    generated_class_node = find_top_level_node(
        generated_tree,
        ast.ClassDef,
        "GraphConvLayer",
    )

    reconstructed = splice_nodes(
        generated_source,
        [
            (generated_function_node, original_function),
            (generated_class_node, original_class),
        ],
    )
    reverse_reference_count = reconstructed.count("GraphConvLayer(")
    if reverse_reference_count != 2:
        raise SystemExit(
            "STOP: reverse audit expected two GraphConvLayer references; "
            f"found {reverse_reference_count}."
        )
    reconstructed = reconstructed.replace(
        "GraphConvLayer(",
        "GCNLayer(",
    ).replace(
        G1_MODEL_NAME,
        A1_MODEL_NAME,
    )
    reverse_patch_exact = reconstructed == original_source

    g1_source.parent.mkdir(parents=True, exist_ok=True)
    g1_source.write_text(generated_source, encoding="utf-8")
    py_compile.compile(str(g1_source), doraise=True)

    a1_module = load_module(a1_source, "g1_a1_source_audit")
    g1_module = load_module(g1_source, "g1_generated_source_audit")

    x = np.load(data_dir / "x.npy", mmap_mode="r")
    edge_index = np.load(data_dir / "edge_index.npy").astype(np.int64)
    input_features = int(x.shape[-1])
    num_nodes = int(x.shape[1])

    adjacency = g1_module.build_normalized_adjacency(
        edge_index,
        num_nodes,
    )

    edges = {
        (int(source), int(target))
        for source, target in zip(edge_index[0], edge_index[1])
    }
    nonself = {edge for edge in edges if edge[0] != edge[1]}
    self_edges = {edge for edge in edges if edge[0] == edge[1]}
    expected_adjacency = torch.zeros_like(adjacency)
    for source, target in nonself:
        expected_adjacency[target, source] = 1.0

    topology_checks = {
        "raw_unique_edge_count_is_64": len(edges) == 64,
        "raw_self_loop_count_is_16": len(self_edges) == 16,
        "raw_nonself_directed_edge_count_is_48": len(nonself) == 48,
        "g1_adjacency_shape_is_16_by_16": list(adjacency.shape) == [16, 16],
        "g1_adjacency_is_binary": bool(
            torch.all((adjacency == 0) | (adjacency == 1)).item()
        ),
        "g1_adjacency_has_zero_diagonal": bool(
            torch.all(torch.diag(adjacency) == 0).item()
        ),
        "g1_adjacency_has_48_entries": int(adjacency.sum().item()) == 48,
        "g1_adjacency_matches_nonself_edges": bool(
            torch.equal(adjacency, expected_adjacency)
        ),
        "g1_adjacency_is_symmetric": bool(
            torch.equal(adjacency, adjacency.T)
        ),
    }

    torch.manual_seed(7007)
    a1_model = a1_module.TemporalGCN(
        input_features=input_features,
        temporal_dim=8,
        gcn_hidden=16,
        gcn_out=8,
    )
    torch.manual_seed(7007)
    g1_model = g1_module.TemporalGCN(
        input_features=input_features,
        temporal_dim=8,
        gcn_hidden=16,
        gcn_out=8,
    )

    a1_parameters = sum(p.numel() for p in a1_model.parameters())
    g1_parameters = sum(p.numel() for p in g1_model.parameters())

    a1_state = a1_model.state_dict()
    g1_state = g1_model.state_dict()

    shared_prefixes = ("temporal.", "node_head.", "graph_head.")
    a1_shared = {
        key: value
        for key, value in a1_state.items()
        if key.startswith(shared_prefixes)
    }
    g1_shared = {
        key: value
        for key, value in g1_state.items()
        if key.startswith(shared_prefixes)
    }

    shared_names_equal = set(a1_shared) == set(g1_shared)
    shared_shapes_equal = shared_names_equal and all(
        tuple(a1_shared[key].shape) == tuple(g1_shared[key].shape)
        for key in a1_shared
    )

    with torch.no_grad():
        for key, value in a1_shared.items():
            g1_state[key].copy_(value)
    g1_model.load_state_dict(g1_state, strict=True)

    shared_equal_after_transplant = all(
        torch.equal(a1_model.state_dict()[key], g1_model.state_dict()[key])
        for key in a1_shared
    )

    torch.manual_seed(9001)
    synthetic_x = torch.randn(4, num_nodes, 8, input_features)

    a1_model.eval()
    g1_model.eval()
    with torch.no_grad():
        temporal_a1 = a1_model.temporal(synthetic_x)
        temporal_g1 = g1_model.temporal(synthetic_x)
        temporal_difference = float(
            torch.max(torch.abs(temporal_a1 - temporal_g1)).item()
        )
        graph_logits, node_logits = g1_model(synthetic_x, adjacency)

    torch.manual_seed(9011)
    random_h = torch.randn(3, num_nodes, 8)
    direct = g1_model.gcn1(random_h, adjacency)
    manual = manual_graphconv(g1_model.gcn1, random_h, adjacency)
    formula_error = float(torch.max(torch.abs(direct - manual)).item())

    g1_model.train()
    graph_target = torch.tensor([0.0, 1.0, 0.0, 1.0])
    node_target = torch.zeros((4, num_nodes), dtype=torch.float32)
    node_target[1, 5] = 1.0
    node_target[3, 12] = 1.0
    graph_train, node_train = g1_model(synthetic_x, adjacency)
    synthetic_loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            graph_train,
            graph_target,
        )
        + torch.nn.functional.binary_cross_entropy_with_logits(
            node_train,
            node_target,
        )
    )
    synthetic_loss.backward()
    all_gradients_finite = all(
        p.grad is not None and bool(torch.isfinite(p.grad).all().item())
        for p in g1_model.parameters()
    )

    g1_graph_keys = sorted(
        key
        for key in g1_state
        if key.startswith(("gcn1.", "gcn2."))
    )

    structural_checks = {
        "reverse_patch_exact": reverse_patch_exact,
        "generated_source_compiles": True,
        "a1_parameter_count_is_882": a1_parameters == EXPECTED_A1_PARAMETERS,
        "g1_parameter_count_is_1138": g1_parameters == EXPECTED_G1_PARAMETERS,
        "parameter_delta_is_256": (
            g1_parameters - a1_parameters == EXPECTED_PARAMETER_DELTA
        ),
        "shared_parameter_names_equal": shared_names_equal,
        "shared_parameter_shapes_equal": shared_shapes_equal,
        "shared_parameters_equal_after_transplant": (
            shared_equal_after_transplant
        ),
        "temporal_outputs_identical_after_transplant": (
            temporal_difference == 0.0
        ),
        "graphconv_formula_matches_manual_reference": formula_error <= 1e-6,
        "g1_graph_output_shape_is_batch": list(graph_logits.shape) == [4],
        "g1_node_output_shape_is_batch_by_nodes": (
            list(node_logits.shape) == [4, num_nodes]
        ),
        "g1_outputs_are_finite": bool(
            torch.isfinite(graph_logits).all().item()
            and torch.isfinite(node_logits).all().item()
        ),
        "synthetic_loss_is_finite": bool(torch.isfinite(synthetic_loss).item()),
        "all_synthetic_gradients_finite": all_gradients_finite,
        "g1_has_separate_root_and_neighbor_parameters": all(
            any(fragment in key for key in g1_graph_keys)
            for fragment in (
                "lin_neighbor.weight",
                "lin_neighbor.bias",
                "lin_root.weight",
            )
        ),
        "g1_has_no_a1_linear_graph_parameters": not any(
            ".linear." in key for key in g1_graph_keys
        ),
        "model_identifier_updated": (
            G1_MODEL_NAME in generated_source
            and A1_MODEL_NAME not in generated_source
        ),
    }

    checks = {
        **prerequisite_checks,
        **topology_checks,
        **structural_checks,
    }
    status = (
        "SOURCE_DIFF_AUDIT_PASSED"
        if all(checks.values())
        else "SOURCE_DIFF_AUDIT_FAILED"
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    diff_path = output_dir / "a1_vs_g1_source.diff"
    diff_path.write_text(
        "".join(
            difflib.unified_diff(
                original_source.splitlines(keepends=True),
                generated_source.splitlines(keepends=True),
                fromfile=str(a1_source),
                tofile=str(g1_source),
            )
        ),
        encoding="utf-8",
    )

    report = {
        "stage": "G1.0",
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "persistent_training_started": False,
            "checkpoint_written": False,
            "validation_inference_performed": False,
            "test_inference_performed": False,
            "synthetic_forward_backward_only": True,
        },
        "single_changed_factor": (
            "normalized shared-transform GCN -> separate-root-and-neighbor "
            "add-aggregation GraphConv"
        ),
        "sources": {
            "a1_path": str(a1_source),
            "a1_sha256": sha256(a1_source),
            "g1_path": str(g1_source),
            "g1_sha256": sha256(g1_source),
            "diff_path": str(diff_path),
            "diff_sha256": sha256(diff_path),
        },
        "source_edit_counts": {
            "adjacency_function_replaced": 1,
            "graph_layer_class_replaced": 1,
            "constructor_references_replaced": constructor_count,
            "model_identifier_occurrences_replaced": model_name_count,
        },
        "parameters": {
            "a1_total": a1_parameters,
            "g1_total": g1_parameters,
            "delta": g1_parameters - a1_parameters,
            "shared_state_keys": sorted(a1_shared),
            "g1_graph_state_keys": g1_graph_keys,
        },
        "synthetic_audit": {
            "temporal_max_abs_difference": temporal_difference,
            "graphconv_formula_max_abs_error": formula_error,
            "graph_output_shape": list(graph_logits.shape),
            "node_output_shape": list(node_logits.shape),
            "synthetic_loss": float(synthetic_loss.item()),
        },
        "checks": checks,
        "next_authorized_stage_if_passed": (
            "G1.1 real-data dry preflight only"
        ),
    }

    report_path = output_dir / "g1_source_diff_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("G1.0 SOURCE GENERATION AND STRUCTURAL DIFF AUDIT")
    print(f"status={status}")
    print(f"a1_source_sha256={sha256(a1_source)}")
    print(f"g1_source_sha256={sha256(g1_source)}")
    print(f"reverse_patch_exact={reverse_patch_exact}")
    print(f"graph_layer_constructor_replacements={constructor_count}")
    print(f"model_identifier_replacements={model_name_count}")
    print(
        "g1_adjacency_has_zero_diagonal="
        f"{topology_checks['g1_adjacency_has_zero_diagonal']}"
    )
    print(
        "g1_adjacency_has_48_entries="
        f"{topology_checks['g1_adjacency_has_48_entries']}"
    )
    print(f"a1_parameter_count={a1_parameters}")
    print(f"g1_parameter_count={g1_parameters}")
    print(f"parameter_delta={g1_parameters - a1_parameters}")
    print(f"shared_parameter_names_equal={shared_names_equal}")
    print(f"shared_parameter_shapes_equal={shared_shapes_equal}")
    print(
        "shared_parameters_equal_after_transplant="
        f"{shared_equal_after_transplant}"
    )
    print(
        "temporal_outputs_identical_after_transplant="
        f"{temporal_difference == 0.0}"
    )
    print(f"graphconv_formula_max_abs_error={formula_error:.9g}")
    print(f"all_synthetic_gradients_finite={all_gradients_finite}")
    print("persistent_training_started=False")
    print("checkpoint_written=False")
    print("validation_inference_performed=False")
    print("test_inference_performed=False")
    print(f"g1_source={g1_source}")
    print(f"diff={diff_path}")
    print(f"report={report_path}")

    if status != "SOURCE_DIFF_AUDIT_PASSED":
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"G1.0: FAIL — {failed}")

    print("G1.0 RESULT: PASS")
    print("next_authorized_stage=G1.1 real-data dry preflight only")


if __name__ == "__main__":
    main()
