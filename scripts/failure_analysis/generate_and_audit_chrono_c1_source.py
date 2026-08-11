#!/usr/bin/env python3
"""
C1.1: generate and independently audit the Chrono-C1 mean-plus-max source.

The script creates the C1 training source from the frozen A1 source only when:
- the A1 SHA-256 matches the frozen value;
- the C1.0 experiment specification is valid;
- the expected A1 graph-head and mean-pooling statements each occur once;
- reversing the permitted edits reproduces the A1 source byte-for-byte;
- model parameter names and shapes differ only where permitted;
- common initialized parameters and node outputs remain identical;
- the C1 forward path exactly implements concatenated mean-plus-max pooling;
- no B1 sampler reference is present.

No training is performed.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


EXPECTED_A1_SOURCE_SHA256 = (
    "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b"
)

A1_MODEL_NAME = "Conv1D-TemporalGCN"
C1_MODEL_NAME = "Conv1D-MeanMax-TemporalGCN"

FORBIDDEN_B1_TOKENS = (
    "chrono_b1_scenario_sampler",
    "scenario_balanced",
    "scenario-balanced",
    "weightedrandomsampler",
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


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def patch_source(a1_text: str) -> tuple[str, dict[str, int]]:
    graph_head_pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)self\.graph_head = "
        r"nn\.Linear\(gcn_out, 1\)[ \t]*$"
    )
    pooling_pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)graph_embedding = "
        r"h\.mean\(dim=1\)[ \t]*$"
    )

    graph_head_matches = list(graph_head_pattern.finditer(a1_text))
    pooling_matches = list(pooling_pattern.finditer(a1_text))
    model_name_count = a1_text.count(A1_MODEL_NAME)

    counts = {
        "graph_head_statement": len(graph_head_matches),
        "mean_pooling_statement": len(pooling_matches),
        "model_name_occurrences": model_name_count,
    }

    if counts["graph_head_statement"] != 1:
        raise ValueError(
            "Expected exactly one A1 graph-head statement; found "
            f"{counts['graph_head_statement']}."
        )
    if counts["mean_pooling_statement"] != 1:
        raise ValueError(
            "Expected exactly one A1 mean-pooling statement; found "
            f"{counts['mean_pooling_statement']}."
        )
    if counts["model_name_occurrences"] < 1:
        raise ValueError(
            f"Expected at least one {A1_MODEL_NAME!r} provenance label."
        )

    c1_text = graph_head_pattern.sub(
        lambda match: (
            f"{match.group('indent')}"
            "self.graph_head = nn.Linear(2 * gcn_out, 1)"
        ),
        a1_text,
        count=1,
    )
    c1_text = pooling_pattern.sub(
        lambda match: (
            f"{match.group('indent')}graph_mean = h.mean(dim=1)\n"
            f"{match.group('indent')}graph_max = h.max(dim=1).values\n"
            f"{match.group('indent')}"
            "graph_embedding = torch.cat([graph_mean, graph_max], dim=-1)"
        ),
        c1_text,
        count=1,
    )
    c1_text = c1_text.replace(A1_MODEL_NAME, C1_MODEL_NAME)

    return c1_text, counts


def reverse_patch(c1_text: str) -> str:
    reversed_text = c1_text.replace(C1_MODEL_NAME, A1_MODEL_NAME)

    new_head_pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)self\.graph_head = "
        r"nn\.Linear\(2 \* gcn_out, 1\)[ \t]*$"
    )
    new_pool_pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)graph_mean = h\.mean\(dim=1\)\n"
        r"(?P=indent)graph_max = h\.max\(dim=1\)\.values\n"
        r"(?P=indent)graph_embedding = "
        r"torch\.cat\(\[graph_mean, graph_max\], dim=-1\)[ \t]*$"
    )

    reversed_text, head_count = new_head_pattern.subn(
        lambda match: (
            f"{match.group('indent')}"
            "self.graph_head = nn.Linear(gcn_out, 1)"
        ),
        reversed_text,
        count=1,
    )
    reversed_text, pool_count = new_pool_pattern.subn(
        lambda match: (
            f"{match.group('indent')}graph_embedding = h.mean(dim=1)"
        ),
        reversed_text,
        count=1,
    )

    if head_count != 1 or pool_count != 1:
        raise ValueError(
            "Could not reverse the generated C1 patch exactly: "
            f"head_count={head_count}, pool_count={pool_count}"
        )

    return reversed_text


def tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.detach().cpu(), b.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1-source", required=True, type=Path)
    parser.add_argument("--c1-source", required=True, type=Path)
    parser.add_argument("--c1-spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    a1_source = args.a1_source.resolve()
    c1_source = args.c1_source.resolve()
    spec_path = args.c1_spec.resolve()
    output_dir = args.output_dir.resolve()

    if not a1_source.is_file():
        raise SystemExit(f"STOP: A1 source missing: {a1_source}")
    if not spec_path.is_file():
        raise SystemExit(f"STOP: C1.0 specification missing: {spec_path}")
    if c1_source.exists():
        raise SystemExit(
            f"STOP: C1 destination source already exists: {c1_source}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: C1.1 output directory is already non-empty: {output_dir}"
        )

    a1_hash = sha256(a1_source)
    if a1_hash != EXPECTED_A1_SOURCE_SHA256:
        raise SystemExit(
            "STOP: frozen A1 source hash changed: "
            f"expected={EXPECTED_A1_SOURCE_SHA256}, actual={a1_hash}"
        )

    specification = load_json(spec_path)
    spec_checks = {
        "stage_is_c1_0": specification.get("stage") == "C1.0",
        "status_is_frozen": (
            specification.get("status") == "SPECIFICATION_FROZEN"
        ),
        "control_is_a1": (
            specification.get("control", {}).get("name")
            == "Chrono-A1 Conv1D-TemporalGCN"
        ),
        "training_not_started": (
            specification.get("training_started") is False
        ),
        "test_not_used_for_selection": (
            specification.get("test_data_used_for_selection") is False
        ),
        "b1_sampler_forbidden": any(
            "B1 scenario-balanced sampler" in item
            for item in specification.get("forbidden_changes", [])
        ),
        "single_change_is_graph_readout": (
            specification.get("single_permitted_change", {}).get("component")
            == "graph readout and required graph-head input width"
        ),
    }
    if not all(spec_checks.values()):
        raise SystemExit(
            "STOP: invalid C1.0 specification: "
            + json.dumps(spec_checks, sort_keys=True)
        )

    a1_text = a1_source.read_text(encoding="utf-8")
    c1_text, replacement_counts = patch_source(a1_text)

    compile(c1_text, str(c1_source), "exec")

    reverse_exact = reverse_patch(c1_text) == a1_text
    if not reverse_exact:
        raise SystemExit(
            "STOP: reversing permitted C1 edits did not reproduce A1 exactly."
        )

    forbidden_hits = [
        token for token in FORBIDDEN_B1_TOKENS
        if token in c1_text.lower()
    ]
    if forbidden_hits:
        raise SystemExit(
            f"STOP: generated C1 source contains B1/sampler tokens: {forbidden_hits}"
        )

    unified_diff = "".join(
        difflib.unified_diff(
            a1_text.splitlines(keepends=True),
            c1_text.splitlines(keepends=True),
            fromfile=str(a1_source),
            tofile=str(c1_source),
        )
    )

    saved_args = specification["control"]["saved_training_arguments"]
    input_features = int(
        specification["dataset"]["x_shape"][-1]
    )
    temporal_dim = int(saved_args["temporal_dim"])
    gcn_hidden = int(saved_args["gcn_hidden"])
    gcn_out = int(saved_args["gcn_out"])
    seed = int(saved_args["seed"])
    node_count = int(specification["dataset"]["x_shape"][1])
    time_steps = int(specification["dataset"]["x_shape"][2])

    with tempfile.TemporaryDirectory(prefix="chrono_c1_audit_") as temp_dir:
        temp_path = Path(temp_dir) / c1_source.name
        temp_path.write_text(c1_text, encoding="utf-8")

        compile_result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(temp_path)],
            text=True,
            capture_output=True,
            check=False,
        )
        if compile_result.returncode != 0:
            raise SystemExit(
                "STOP: generated C1 source failed py_compile:\n"
                + compile_result.stderr
            )

        a1_module = load_module(a1_source, "chrono_c1_audit_a1")
        c1_module = load_module(temp_path, "chrono_c1_audit_c1")

        torch.manual_seed(seed)
        a1_model = a1_module.TemporalGCN(
            input_features=input_features,
            temporal_dim=temporal_dim,
            gcn_hidden=gcn_hidden,
            gcn_out=gcn_out,
        )
        torch.manual_seed(seed)
        c1_model = c1_module.TemporalGCN(
            input_features=input_features,
            temporal_dim=temporal_dim,
            gcn_hidden=gcn_hidden,
            gcn_out=gcn_out,
        )

        a1_state = a1_model.state_dict()
        c1_state = c1_model.state_dict()

        state_names_equal = list(a1_state) == list(c1_state)
        differing_shapes = {
            key: {
                "a1": list(a1_state[key].shape),
                "c1": list(c1_state[key].shape),
            }
            for key in a1_state
            if a1_state[key].shape != c1_state[key].shape
        }
        expected_shape_difference = {
            "graph_head.weight": {
                "a1": [1, gcn_out],
                "c1": [1, 2 * gcn_out],
            }
        }
        shape_difference_only_permitted = (
            differing_shapes == expected_shape_difference
        )

        common_parameter_equality = {
            key: tensor_equal(a1_state[key], c1_state[key])
            for key in a1_state
            if key != "graph_head.weight"
        }
        common_initialized_parameters_equal = all(
            common_parameter_equality.values()
        )

        a1_parameter_count = sum(
            parameter.numel() for parameter in a1_model.parameters()
        )
        c1_parameter_count = sum(
            parameter.numel() for parameter in c1_model.parameters()
        )
        expected_parameter_delta = gcn_out
        parameter_delta = c1_parameter_count - a1_parameter_count
        parameter_delta_correct = parameter_delta == expected_parameter_delta

        graph_head_widths_correct = (
            a1_model.graph_head.in_features == gcn_out
            and c1_model.graph_head.in_features == 2 * gcn_out
            and a1_model.graph_head.out_features == 1
            and c1_model.graph_head.out_features == 1
        )

        torch.manual_seed(104729)
        x = torch.randn(3, node_count, time_steps, input_features)
        a_hat = torch.eye(node_count)

        a1_model.eval()
        c1_model.eval()
        with torch.no_grad():
            a1_graph_logits, a1_node_logits = a1_model(x, a_hat)
            c1_graph_logits, c1_node_logits = c1_model(x, a_hat)

            h = c1_model.temporal(x)
            h = F.relu(c1_model.gcn1(h, a_hat))
            h = F.relu(c1_model.gcn2(h, a_hat))
            expected_node_logits = c1_model.node_head(h).squeeze(-1)
            graph_mean = h.mean(dim=1)
            graph_max = h.max(dim=1).values
            expected_graph_embedding = torch.cat(
                [graph_mean, graph_max],
                dim=-1,
            )
            expected_graph_logits = c1_model.graph_head(
                expected_graph_embedding
            ).squeeze(-1)

        forward_checks = {
            "a1_graph_shape_correct": list(a1_graph_logits.shape) == [3],
            "c1_graph_shape_correct": list(c1_graph_logits.shape) == [3],
            "a1_node_shape_correct": (
                list(a1_node_logits.shape) == [3, node_count]
            ),
            "c1_node_shape_correct": (
                list(c1_node_logits.shape) == [3, node_count]
            ),
            "node_outputs_identical_a1_c1": torch.equal(
                a1_node_logits,
                c1_node_logits,
            ),
            "c1_node_forward_matches_manual": torch.equal(
                c1_node_logits,
                expected_node_logits,
            ),
            "c1_graph_forward_matches_manual_meanmax": torch.equal(
                c1_graph_logits,
                expected_graph_logits,
            ),
            "c1_embedding_width_correct": (
                expected_graph_embedding.shape[-1] == 2 * gcn_out
            ),
        }

    audit_checks = {
        "a1_hash_matches": a1_hash == EXPECTED_A1_SOURCE_SHA256,
        **spec_checks,
        "expected_replacement_counts": (
            replacement_counts["graph_head_statement"] == 1
            and replacement_counts["mean_pooling_statement"] == 1
            and replacement_counts["model_name_occurrences"] >= 1
        ),
        "reverse_patch_reproduces_a1_exactly": reverse_exact,
        "no_b1_sampler_tokens": not forbidden_hits,
        "generated_source_compiles": True,
        "state_parameter_names_equal": state_names_equal,
        "shape_difference_only_graph_head_weight": (
            shape_difference_only_permitted
        ),
        "common_initialized_parameters_equal": (
            common_initialized_parameters_equal
        ),
        "graph_head_widths_correct": graph_head_widths_correct,
        "parameter_delta_correct": parameter_delta_correct,
        **forward_checks,
    }

    if not all(audit_checks.values()):
        failed = [
            key for key, passed in audit_checks.items() if not passed
        ]
        raise SystemExit(f"STOP: C1.1 source audit failed: {failed}")

    output_dir.mkdir(parents=True, exist_ok=False)
    c1_source.parent.mkdir(parents=True, exist_ok=True)
    c1_source.write_text(c1_text, encoding="utf-8")
    os.chmod(c1_source, 0o555)

    installed_compile = subprocess.run(
        [sys.executable, "-m", "py_compile", str(c1_source)],
        text=True,
        capture_output=True,
        check=False,
    )
    if installed_compile.returncode != 0:
        c1_source.unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        raise SystemExit(
            "STOP: installed C1 source failed py_compile:\n"
            + installed_compile.stderr
        )

    diff_path = output_dir / "a1_vs_c1_source.diff"
    diff_path.write_text(unified_diff, encoding="utf-8")

    audit = {
        "stage": "C1.1",
        "status": "SOURCE_GENERATED_AND_AUDITED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_started": False,
        "a1_source": {
            "path": str(a1_source),
            "sha256": a1_hash,
        },
        "c1_source": {
            "path": str(c1_source),
            "sha256": sha256(c1_source),
            "mode_octal": oct(c1_source.stat().st_mode & 0o777),
        },
        "c1_spec": {
            "path": str(spec_path),
            "sha256": sha256(spec_path),
        },
        "permitted_changes": {
            "graph_head": (
                "nn.Linear(gcn_out, 1) -> "
                "nn.Linear(2 * gcn_out, 1)"
            ),
            "graph_readout": (
                "mean pooling -> concatenated mean-plus-max pooling"
            ),
            "provenance_label": (
                f"{A1_MODEL_NAME} -> {C1_MODEL_NAME}"
            ),
        },
        "replacement_counts": replacement_counts,
        "audit_checks": audit_checks,
        "model_audit": {
            "input_features": input_features,
            "temporal_dim": temporal_dim,
            "gcn_hidden": gcn_hidden,
            "gcn_out": gcn_out,
            "a1_parameter_count": a1_parameter_count,
            "c1_parameter_count": c1_parameter_count,
            "parameter_delta": parameter_delta,
            "expected_parameter_delta": expected_parameter_delta,
            "differing_parameter_shapes": differing_shapes,
            "common_parameter_equality": common_parameter_equality,
        },
        "forbidden_token_hits": forbidden_hits,
        "diff": {
            "path": str(diff_path),
            "sha256": sha256(diff_path),
        },
    }

    audit_path = output_dir / "c1_source_diff_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    print("C1.1 SOURCE GENERATION AND DIFF AUDIT: PASS")
    print(f"a1_source_sha256={a1_hash}")
    print(f"c1_source_sha256={sha256(c1_source)}")
    print("single_model_change=mean -> mean-plus-max graph readout")
    print("graph_head_input_change=gcn_out -> 2*gcn_out")
    print(f"model_name={C1_MODEL_NAME}")
    print(f"b1_sampler_references=0")
    print(f"reverse_patch_exact={reverse_exact}")
    print(f"state_parameter_names_equal={state_names_equal}")
    print(
        "shape_difference_only_graph_head_weight="
        f"{shape_difference_only_permitted}"
    )
    print(
        "common_initialized_parameters_equal="
        f"{common_initialized_parameters_equal}"
    )
    print(f"a1_parameter_count={a1_parameter_count}")
    print(f"c1_parameter_count={c1_parameter_count}")
    print(f"parameter_delta={parameter_delta}")
    print(
        "node_outputs_identical_a1_c1="
        f"{forward_checks['node_outputs_identical_a1_c1']}"
    )
    print(
        "c1_graph_forward_matches_manual_meanmax="
        f"{forward_checks['c1_graph_forward_matches_manual_meanmax']}"
    )
    print(f"training_started=False")
    print(f"c1_source={c1_source}")
    print(f"diff={diff_path}")
    print(f"audit={audit_path}")


if __name__ == "__main__":
    main()
