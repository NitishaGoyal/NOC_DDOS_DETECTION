from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


STAGE = "V5_P3_F7_E2_R4_R3_R1_FROZEN_BATCH_GEOMETRY_RECOVERY"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "TRAIN-DERIVED"

EXPECTED_COMPUTE_LOSS_SHA256 = (
    "d9629317801a72a1905dbcd49eace1418f15014df3a2f55d097c25ae4997155f"
)
EXPECTED_EARLY_STOP = (
    "epoch >= 15 and early_stop_patience_counter >= 12"
)
EXPECTED_PARAMETER_COUNT = 60553
EXPECTED_SEED = 107
EXPECTED_TRAIN_ITEMS = 110855
EXPECTED_VALIDATION_ITEMS = 13863
INTERCEPT_EXIT_CODE = 86
EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE = 256
EXPECTED_TRAIN_BATCHES = 434
EXPECTED_VALIDATION_BATCHES = 55

GROUPS = {
    "control_dynamic70": None,
    "ablate_directional_traffic_volume": (0, 10),
    "ablate_inter_flit_timing": (10, 30),
    "ablate_queue_activity": (30, 40),
    "ablate_buffer_pressure": (40, 55),
    "ablate_flow_control_stalls": (55, 70),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(
    path: Path,
    value: str,
    mode: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)
    if mode is not None:
        path.chmod(mode)




def create_corrected_cardinality_trainer(
    original_path: Path,
    corrected_path: Path,
    expected_original_sha256: str,
) -> dict[str, Any]:
    """Patch the two legacy P2 cardinalities by parsed integer value.

    The failed R3 package searched for the exact textual token ``70166``.
    Python source may legally spell the same integer as ``70_166``. The runtime
    message prints the evaluated value without underscores, so the console can
    report 70166 even when that exact character sequence does not occur in the
    file.

    This recovery parses the trainer and replaces only ast.Constant integer
    nodes whose evaluated values are exactly 70166 or 12528.
    """
    require(
        sha256_file(original_path) == expected_original_sha256,
        "R5 generated trainer SHA changed before cardinality correction",
    )

    original = original_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    original_tree = ast.parse(original)
    offsets = line_offsets(original)
    lines = original.splitlines()

    replacements_by_value = {
        70166: 110855,
        12528: 13863,
    }
    replacements: list[tuple[int, int, str]] = []
    contexts: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for old_value, new_value in replacements_by_value.items():
        nodes = [
            node
            for node in ast.walk(original_tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
            and node.value == old_value
        ]
        require(
            nodes,
            f"legacy cardinality AST integer {old_value} not found",
        )
        require(
            len(nodes) <= 20,
            f"unexpectedly many AST occurrences of legacy cardinality "
            f"{old_value}: {len(nodes)}",
        )

        counts[str(old_value)] = len(nodes)

        for node in nodes:
            start_offset, end_offset = absolute_span(node, offsets)
            source_token = original[start_offset:end_offset]
            line_number = int(node.lineno)
            context_start = max(1, line_number - 3)
            context_end = min(len(lines), int(node.end_lineno) + 3)

            replacements.append(
                (
                    start_offset,
                    end_offset,
                    str(new_value),
                )
            )
            contexts.append({
                "old_evaluated_value": old_value,
                "new_evaluated_value": new_value,
                "original_source_token": source_token,
                "line_start": int(node.lineno),
                "line_end": int(node.end_lineno),
                "context_start": context_start,
                "context_end": context_end,
                "context": "\n".join(
                    f"{number:05d}: {lines[number - 1]}"
                    for number in range(context_start, context_end + 1)
                ),
            })

    corrected = apply_replacements(original, replacements)
    corrected_tree = ast.parse(corrected)
    compile(corrected, str(corrected_path), "exec")

    remaining_old_values = {
        value: [
            {
                "line_start": int(node.lineno),
                "line_end": int(node.end_lineno),
                "source": ast.get_source_segment(corrected, node),
            }
            for node in ast.walk(corrected_tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
            and node.value == value
        ]
        for value in replacements_by_value
    }
    require(
        all(not rows for rows in remaining_old_values.values()),
        f"legacy cardinality AST constants survived: {remaining_old_values}",
    )

    corrected_values = [
        node.value
        for node in ast.walk(corrected_tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ]
    require(
        110855 in corrected_values,
        "certified P3 train cardinality missing after AST correction",
    )
    require(
        13863 in corrected_values,
        "certified P3 validation cardinality missing after AST correction",
    )

    atomic_text(corrected_path, corrected, mode=0o555)

    return {
        "original_trainer": str(original_path),
        "original_trainer_sha256": sha256_file(original_path),
        "corrected_trainer": str(corrected_path),
        "corrected_trainer_sha256": sha256_file(corrected_path),
        "failure_classification": (
            "R3_searched_exact_text_70166_but_source_used_an_equivalent_"
            "Python_numeric_literal_spelling"
        ),
        "legacy_train_items": 70166,
        "certified_F7_train_items": 110855,
        "legacy_validation_items": 12528,
        "certified_F7_validation_items": 13863,
        "AST_integer_replacement_counts": counts,
        "exact_source_contexts": contexts,
        "patch_scope": (
            "ast.Constant integer nodes only; evaluated_value "
            "70166_to_110855 and 12528_to_13863"
        ),
        "textual_spelling_independent": True,
        "compute_loss_not_targeted": True,
        "optimizer_scheduler_clipping_not_targeted": True,
        "early_stop_not_targeted": True,
        "model_architecture_not_targeted": True,
        "feature_masking_not_targeted": True,
    }



def create_item_batch_route_trainer(
    cardinality_corrected_path: Path,
    final_path: Path,
    expected_cardinality_sha256: str,
) -> dict[str, Any]:
    """Replace pair-block batching using frozen batch geometry.

    R4 failed because it assumed that the pair-count constructor parameter
    would contain the words "pair" plus "batch" or "block". The actual
    PairBlockBatchSampler API is free to call that parameter ``batch_size`` or
    use another neutral name.

    The effective item batch size does not need to be inferred from a parameter
    name. It is uniquely recoverable from the frozen P2 cardinality and batch
    count:

        ceil(70_166 / B) = 275

    The only positive integer B satisfying that equation is 256. The frozen
    validation route is consistent:

        ceil(12_528 / 256) = 49

    This patch therefore:
      1. certifies B=256 mathematically from frozen execution geometry;
      2. audits sampler signatures and call bindings without depending on names;
      3. preserves train/validation shuffle roles and deterministic seed route;
      4. replaces only the active pair-block construction and DataLoader
         batch_sampler binding.
    """
    require(
        sha256_file(cardinality_corrected_path)
        == expected_cardinality_sha256,
        "cardinality-corrected trainer SHA changed before sampler recovery",
    )

    text = cardinality_corrected_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    tree = ast.parse(text)
    offsets = line_offsets(text)

    LEGACY_TRAIN_ITEMS = 70_166
    LEGACY_VALIDATION_ITEMS = 12_528
    LEGACY_TRAIN_BATCHES = 275
    LEGACY_VALIDATION_BATCHES = 49

    def admissible_batch_sizes(item_count: int, batch_count: int) -> list[int]:
        require(item_count > 0, "item_count must be positive")
        require(batch_count > 0, "batch_count must be positive")
        lower = math.ceil(item_count / batch_count)
        if batch_count == 1:
            upper = item_count
        else:
            upper = (item_count - 1) // (batch_count - 1)
        return [
            value
            for value in range(lower, upper + 1)
            if math.ceil(item_count / value) == batch_count
        ]

    train_candidates = admissible_batch_sizes(
        LEGACY_TRAIN_ITEMS,
        LEGACY_TRAIN_BATCHES,
    )
    validation_candidates = admissible_batch_sizes(
        LEGACY_VALIDATION_ITEMS,
        LEGACY_VALIDATION_BATCHES,
    )
    intersection = sorted(
        set(train_candidates).intersection(validation_candidates)
    )

    require(
        train_candidates == [EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE],
        "frozen train geometry does not uniquely resolve item batch size: "
        f"{train_candidates}",
    )
    require(
        EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE in validation_candidates,
        "frozen validation geometry is inconsistent with item batch size 256: "
        f"{validation_candidates}",
    )
    require(
        intersection == [EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE],
        "frozen train/validation geometry intersection changed: "
        f"{intersection}",
    )

    def assignment_nodes(target_name: str):
        rows = []
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    names.extend(target_names(target))
            elif isinstance(node, ast.AnnAssign):
                names.extend(target_names(node.target))
            else:
                continue
            if target_name in names:
                rows.append(node)
        return rows

    def assignment_value(node):
        value = node.value
        require(value is not None, "assignment has no value")
        return value

    def simple_constant_environment() -> dict[str, Any]:
        candidates: dict[str, list[ast.AST]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                target_list = []
                for target in node.targets:
                    target_list.extend(target_names(target))
                for name in target_list:
                    if "." not in name:
                        candidates.setdefault(name, []).append(node.value)
            elif isinstance(node, ast.AnnAssign):
                for name in target_names(node.target):
                    if "." not in name and node.value is not None:
                        candidates.setdefault(name, []).append(node.value)

        environment: dict[str, Any] = {}

        def evaluate(node: ast.AST | None, stack: set[str] | None = None):
            if node is None:
                return None
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                if node.id in environment:
                    return environment[node.id]
                stack = set() if stack is None else set(stack)
                if node.id in stack:
                    return None
                rows = candidates.get(node.id, [])
                if len(rows) != 1:
                    return None
                stack.add(node.id)
                return evaluate(rows[0], stack)
            if isinstance(node, ast.UnaryOp):
                operand = evaluate(node.operand, stack)
                if operand is None:
                    return None
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.UAdd):
                    return +operand
                if isinstance(node.op, ast.Not):
                    return not operand
                return None
            if isinstance(node, ast.BinOp):
                left = evaluate(node.left, stack)
                right = evaluate(node.right, stack)
                if left is None or right is None:
                    return None
                try:
                    if isinstance(node.op, ast.Add):
                        return left + right
                    if isinstance(node.op, ast.Sub):
                        return left - right
                    if isinstance(node.op, ast.Mult):
                        return left * right
                    if isinstance(node.op, ast.FloorDiv):
                        return left // right
                    if isinstance(node.op, ast.Div):
                        return left / right
                    if isinstance(node.op, ast.Mod):
                        return left % right
                    if isinstance(node.op, ast.Pow):
                        return left ** right
                except Exception:
                    return None
            return None

        changed = True
        while changed:
            changed = False
            for name, rows in candidates.items():
                if name in environment or len(rows) != 1:
                    continue
                value = evaluate(rows[0])
                if isinstance(value, (int, float, bool, str)):
                    environment[name] = value
                    changed = True

        environment["_evaluate"] = evaluate
        return environment

    constant_environment = simple_constant_environment()
    evaluate = constant_environment.pop("_evaluate")

    def class_init_signature(class_name: str):
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == class_name
        ]
        require(
            len(classes) == 1,
            f"{class_name} definition is not unique",
        )
        initializers = [
            node
            for node in classes[0].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__init__"
        ]
        require(
            len(initializers) == 1,
            f"{class_name}.__init__ is not unique",
        )
        init = initializers[0]

        positional_nodes = (
            list(init.args.posonlyargs)
            + list(init.args.args)
        )
        positional_names = [argument.arg for argument in positional_nodes]
        if positional_names and positional_names[0] == "self":
            positional_names = positional_names[1:]

        defaults: dict[str, ast.AST] = {}
        positional_defaults = list(init.args.defaults)
        if positional_defaults:
            for name, default in zip(
                positional_names[-len(positional_defaults):],
                positional_defaults,
            ):
                defaults[name] = default

        keyword_only_names = [
            argument.arg for argument in init.args.kwonlyargs
        ]
        for name, default in zip(
            keyword_only_names,
            init.args.kw_defaults,
        ):
            if default is not None:
                defaults[name] = default

        return {
            "class_node": classes[0],
            "init_node": init,
            "positional_names": positional_names,
            "keyword_only_names": keyword_only_names,
            "all_names": positional_names + keyword_only_names,
            "defaults": defaults,
            "signature_source": ast.unparse(init.args),
        }

    sampler_signature = class_init_signature(
        "PairBlockBatchSampler"
    )

    def bind_call(call: ast.Call) -> dict[str, ast.AST]:
        bound = dict(sampler_signature["defaults"])
        for index, argument in enumerate(call.args):
            require(
                index < len(sampler_signature["positional_names"]),
                "PairBlockBatchSampler call has too many positional args",
            )
            bound[
                sampler_signature["positional_names"][index]
            ] = argument
        for keyword in call.keywords:
            require(
                keyword.arg is not None,
                "PairBlockBatchSampler **kwargs route is unsupported",
            )
            bound[keyword.arg] = keyword.value
        return bound

    def resolve_shuffle(
        bound: dict[str, ast.AST],
        target: str,
    ) -> tuple[bool, str, str]:
        named_candidates = []
        for name, node in bound.items():
            if "shuffle" not in name.lower():
                continue
            value = evaluate(node)
            if isinstance(value, bool):
                named_candidates.append(
                    (name, value, ast.unparse(node))
                )

        unique_values = sorted(
            set(value for _, value, _ in named_candidates)
        )
        if len(unique_values) == 1:
            return (
                unique_values[0],
                "signature_or_call_binding",
                ";".join(
                    f"{name}={source}"
                    for name, _, source in named_candidates
                ),
            )

        expected = target == "train_sampler"
        return (
            expected,
            "certified_target_role",
            (
                "train_shuffle_true"
                if expected
                else "validation_shuffle_false"
            ),
        )

    def resolve_seed_source(
        bound: dict[str, ast.AST],
    ) -> tuple[str, str]:
        seed_candidates = [
            (name, node)
            for name, node in bound.items()
            if "seed" in name.lower()
        ]
        if len(seed_candidates) == 1:
            name, node = seed_candidates[0]
            return ast.unparse(node), f"bound_parameter:{name}"
        return "seed", "trainer_seed_argument"

    controller_source = (
        "\nclass F7EpochShuffleController:\n"
        "    # Deterministic epoch-level controller for item-level shuffle.\n"
        "    def __init__(self, base_seed):\n"
        "        self.base_seed = int(base_seed)\n"
        "        self.epoch = 0\n"
        "        self.generator = torch.Generator()\n"
        "        self.set_epoch(0)\n"
        "\n"
        "    def set_epoch(self, epoch):\n"
        "        self.epoch = int(epoch)\n"
        "        self.generator.manual_seed(self.base_seed + self.epoch)\n"
    )
    controller_insertion = insertion_offset_after_future_imports(
        text,
        tree,
        offsets,
    )

    sampler_targets = ("train_sampler", "validation_sampler")
    sampler_rows: dict[str, dict[str, Any]] = {}
    replacements: list[tuple[int, int, str]] = [
        (controller_insertion, controller_insertion, controller_source)
    ]

    for target in sampler_targets:
        assignments = assignment_nodes(target)
        require(
            len(assignments) == 1,
            f"{target} assignment is not unique: "
            f"{[(node.lineno, ast.unparse(node)) for node in assignments]}",
        )
        assignment = assignments[0]
        value = assignment_value(assignment)
        require(
            isinstance(value, ast.Call)
            and call_name(value.func).rsplit(".", 1)[-1]
            == "PairBlockBatchSampler",
            f"{target} is not constructed by PairBlockBatchSampler: "
            f"{ast.unparse(value)}",
        )

        bound = bind_call(value)
        evaluated_bindings = {}
        for name, node in bound.items():
            resolved = evaluate(node)
            evaluated_bindings[name] = {
                "source": ast.unparse(node),
                "resolved_value": resolved,
                "resolved_type": (
                    type(resolved).__name__
                    if resolved is not None
                    else None
                ),
            }

        # Diagnostic only: identify any argument/default equal to 128. This is
        # not used to authorize the patch.
        inferred_pair_count_candidates = [
            {
                "parameter": name,
                "source": row["source"],
                "resolved_value": row["resolved_value"],
            }
            for name, row in evaluated_bindings.items()
            if row["resolved_value"]
            == EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE // 2
        ]

        shuffle_value, shuffle_resolution, shuffle_source = (
            resolve_shuffle(bound, target)
        )
        seed_source, seed_resolution = resolve_seed_source(bound)

        if target == "train_sampler":
            require(
                shuffle_value is True,
                "train PairBlockBatchSampler is not shuffled",
            )
            replacement_value = (
                f"F7EpochShuffleController({seed_source})"
            )
        else:
            require(
                shuffle_value is False,
                "validation PairBlockBatchSampler unexpectedly shuffles",
            )
            replacement_value = "None"

        value_start, value_end = absolute_span(value, offsets)
        replacements.append(
            (value_start, value_end, replacement_value)
        )

        sampler_rows[target] = {
            "replacement_value": replacement_value,
            "assignment_line": int(assignment.lineno),
            "original_call": ast.unparse(value),
            "signature": sampler_signature["signature_source"],
            "bound_arguments": evaluated_bindings,
            "pair_count_name_required_for_patch": False,
            "diagnostic_128_candidates": inferred_pair_count_candidates,
            "effective_item_batch_size": (
                EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE
            ),
            "effective_item_batch_size_resolution": (
                "unique_frozen_batch_geometry"
            ),
            "shuffle": shuffle_value,
            "shuffle_source": shuffle_source,
            "shuffle_resolution": shuffle_resolution,
            "seed_source": seed_source,
            "seed_resolution": seed_resolution,
        }

    loader_targets = {
        "train_loader": "train_sampler",
        "validation_loader": "validation_sampler",
    }
    loader_rows: dict[str, dict[str, Any]] = {}

    for loader_target, sampler_target in loader_targets.items():
        assignments = assignment_nodes(loader_target)
        require(
            len(assignments) == 1,
            f"{loader_target} assignment is not unique: "
            f"{[(node.lineno, ast.unparse(node)) for node in assignments]}",
        )
        assignment = assignments[0]
        value = assignment_value(assignment)
        require(
            isinstance(value, ast.Call)
            and call_name(value.func).rsplit(".", 1)[-1]
            == "DataLoader",
            f"{loader_target} is not a DataLoader call: "
            f"{ast.unparse(value)}",
        )

        original_call = ast.unparse(value)
        keyword_map = {
            keyword.arg: keyword
            for keyword in value.keywords
            if keyword.arg is not None
        }
        require(
            "batch_sampler" in keyword_map,
            f"{loader_target} has no inherited batch_sampler route",
        )
        require(
            ast.unparse(keyword_map["batch_sampler"].value)
            == sampler_target,
            f"{loader_target} batch_sampler does not reference "
            f"{sampler_target}",
        )
        require(
            "collate_fn" not in keyword_map,
            f"{loader_target} does not use default collate",
        )

        new_call = copy.deepcopy(value)
        new_call.keywords = [
            keyword
            for keyword in new_call.keywords
            if keyword.arg not in {
                "batch_sampler",
                "sampler",
                "batch_size",
                "shuffle",
                "drop_last",
                "generator",
            }
        ]

        shuffle_value = sampler_rows[sampler_target]["shuffle"]
        new_call.keywords.extend([
            ast.keyword(
                arg="batch_size",
                value=ast.Constant(
                    value=EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE
                ),
            ),
            ast.keyword(
                arg="shuffle",
                value=ast.Constant(value=shuffle_value),
            ),
            ast.keyword(
                arg="drop_last",
                value=ast.Constant(value=False),
            ),
        ])

        if shuffle_value:
            seed_source = sampler_rows[sampler_target]["seed_source"]
            generator_expression = ast.parse(
                f"{sampler_target}.generator",
                mode="eval",
            ).body
            new_call.keywords.append(
                ast.keyword(
                    arg="generator",
                    value=generator_expression,
                )
            )

        ast.fix_missing_locations(new_call)
        new_call_source = ast.unparse(new_call)
        value_start, value_end = absolute_span(value, offsets)
        replacements.append(
            (value_start, value_end, new_call_source)
        )

        loader_rows[loader_target] = {
            "assignment_line": int(assignment.lineno),
            "original_call": original_call,
            "corrected_call": new_call_source,
            "batch_size": EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE,
            "shuffle": shuffle_value,
            "drop_last": False,
            "default_collate": True,
            "generator_seeded": bool(shuffle_value),
        }

    corrected = apply_replacements(text, replacements)
    corrected_tree = ast.parse(corrected)
    compile(corrected, str(final_path), "exec")

    active_pair_calls = [
        {
            "line": int(node.lineno),
            "source": ast.unparse(node),
        }
        for node in ast.walk(corrected_tree)
        if isinstance(node, ast.Call)
        and call_name(node.func).rsplit(".", 1)[-1]
        == "PairBlockBatchSampler"
    ]
    require(
        not active_pair_calls,
        f"active PairBlockBatchSampler calls survived: "
        f"{active_pair_calls}",
    )

    controller_classes = [
        node
        for node in corrected_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "F7EpochShuffleController"
    ]
    require(
        len(controller_classes) == 1,
        "F7EpochShuffleController definition is not unique",
    )

    corrected_parent_map: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(corrected_tree):
        for child in ast.iter_child_nodes(parent):
            corrected_parent_map[child] = parent

    train_sampler_loads = [
        node
        for node in ast.walk(corrected_tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == "train_sampler"
    ]
    validation_sampler_loads = [
        node
        for node in ast.walk(corrected_tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == "validation_sampler"
    ]

    set_epoch_uses = []
    generator_uses = []
    unexpected_train_sampler_uses = []

    for node in train_sampler_loads:
        parent = corrected_parent_map.get(node)
        grandparent = (
            corrected_parent_map.get(parent)
            if parent is not None
            else None
        )
        row = {
            "line": int(node.lineno),
            "parent": (
                ast.unparse(parent)
                if parent is not None
                else None
            ),
            "grandparent": (
                ast.unparse(grandparent)
                if grandparent is not None
                else None
            ),
        }

        if (
            isinstance(parent, ast.Attribute)
            and parent.value is node
            and parent.attr == "set_epoch"
            and isinstance(grandparent, ast.Call)
            and grandparent.func is parent
        ):
            set_epoch_uses.append(row)
        elif (
            isinstance(parent, ast.Attribute)
            and parent.value is node
            and parent.attr == "generator"
        ):
            generator_uses.append(row)
        else:
            unexpected_train_sampler_uses.append(row)

    require(
        len(set_epoch_uses) == 1,
        f"expected one train_sampler.set_epoch use, "
        f"found {set_epoch_uses}",
    )
    require(
        len(generator_uses) == 1,
        f"expected one train_sampler.generator use, "
        f"found {generator_uses}",
    )
    require(
        not unexpected_train_sampler_uses,
        "unexpected train_sampler use survived: "
        f"{unexpected_train_sampler_uses}",
    )
    require(
        not validation_sampler_loads,
        "validation_sampler remains active: "
        f"{[(node.lineno, ast.unparse(corrected_parent_map.get(node))) for node in validation_sampler_loads]}",
    )

    def evaluated_constant(node: ast.AST | None):
        if isinstance(node, ast.Constant):
            return node.value
        return None

    corrected_loader_rows = {}
    for loader_target in loader_targets:
        assignments = []
        for node in ast.walk(corrected_tree):
            names = []
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    names.extend(target_names(target))
            elif isinstance(node, ast.AnnAssign):
                names.extend(target_names(node.target))
            else:
                continue
            if loader_target in names:
                assignments.append(node)

        require(len(assignments) == 1, f"{loader_target} changed")
        call = assignments[0].value
        require(isinstance(call, ast.Call), f"{loader_target} call missing")
        keywords = {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg is not None
        }
        require("batch_sampler" not in keywords, "batch_sampler survived")
        require("sampler" not in keywords, "sampler keyword survived")
        require(
            evaluated_constant(keywords.get("batch_size"))
            == EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE,
            f"{loader_target} batch_size is not "
            f"{EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE}",
        )
        require(
            evaluated_constant(keywords.get("drop_last")) is False,
            f"{loader_target} drop_last is not false",
        )
        require(
            "collate_fn" not in keywords,
            f"{loader_target} default collate was not preserved",
        )
        corrected_loader_rows[loader_target] = {
            "source": ast.unparse(call),
            "batch_size": EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE,
            "shuffle": evaluated_constant(keywords.get("shuffle")),
            "drop_last": False,
            "default_collate": True,
            "generator_present": "generator" in keywords,
            "generator_source": (
                ast.unparse(keywords["generator"])
                if "generator" in keywords
                else None
            ),
        }

    require(
        corrected_loader_rows["train_loader"]["shuffle"] is True,
        "corrected train loader is not shuffled",
    )
    require(
        corrected_loader_rows["validation_loader"]["shuffle"] is False,
        "corrected validation loader is shuffled",
    )
    require(
        corrected_loader_rows["train_loader"]["generator_source"]
        == "train_sampler.generator",
        "corrected train loader is not bound to "
        "train_sampler.generator",
    )
    require(
        corrected_loader_rows["validation_loader"]["generator_present"]
        is False,
        "validation loader unexpectedly has a random generator",
    )

    train_batches = math.ceil(
        EXPECTED_TRAIN_ITEMS / EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE
    )
    validation_batches = math.ceil(
        EXPECTED_VALIDATION_ITEMS
        / EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE
    )
    require(
        train_batches == EXPECTED_TRAIN_BATCHES,
        f"train batch count changed: {train_batches}",
    )
    require(
        validation_batches == EXPECTED_VALIDATION_BATCHES,
        f"validation batch count changed: {validation_batches}",
    )

    remaining_pair_text_rows = []
    for line_number, line in enumerate(
        corrected.splitlines(),
        start=1,
    ):
        if "pair" in line.lower():
            remaining_pair_text_rows.append({
                "line": line_number,
                "text": line,
            })

    atomic_text(final_path, corrected, mode=0o555)

    return {
        "input_cardinality_corrected_trainer": str(
            cardinality_corrected_path
        ),
        "input_cardinality_corrected_trainer_sha256": (
            sha256_file(cardinality_corrected_path)
        ),
        "final_item_batch_trainer": str(final_path),
        "final_item_batch_trainer_sha256": sha256_file(final_path),
        "failure_classification": (
            "R4_assumed_semantic_pair_count_parameter_name_but_sampler_"
            "signature_uses_a_neutral_parameter_name"
        ),
        "recovery_method": (
            "derive_effective_item_batch_size_from_frozen_cardinality_and_"
            "batch_count_geometry_not_parameter_naming"
        ),
        "frozen_batch_geometry": {
            "legacy_train_items": LEGACY_TRAIN_ITEMS,
            "legacy_train_batches": LEGACY_TRAIN_BATCHES,
            "legacy_train_admissible_item_batch_sizes": train_candidates,
            "legacy_validation_items": LEGACY_VALIDATION_ITEMS,
            "legacy_validation_batches": LEGACY_VALIDATION_BATCHES,
            "legacy_validation_admissible_item_batch_sizes": (
                validation_candidates
            ),
            "intersection": intersection,
            "unique_effective_item_batch_size": (
                EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE
            ),
        },
        "sampler_signature": sampler_signature["signature_source"],
        "sampler_routes": sampler_rows,
        "loader_routes": loader_rows,
        "corrected_loader_routes": corrected_loader_rows,
        "active_pair_block_sampler_call_count": 0,
        "epoch_shuffle_controller": {
            "class": "F7EpochShuffleController",
            "seed_formula": "base_seed_plus_epoch",
            "set_epoch_use_count": len(set_epoch_uses),
            "generator_use_count": len(generator_uses),
            "unexpected_use_count": len(
                unexpected_train_sampler_uses
            ),
        },
        "active_validation_sampler_load_count": 0,
        "effective_item_batch_size": (
            EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE
        ),
        "effective_item_batch_size_resolution": (
            "unique_frozen_batch_geometry"
        ),
        "train_items": EXPECTED_TRAIN_ITEMS,
        "validation_items": EXPECTED_VALIDATION_ITEMS,
        "train_batches": train_batches,
        "validation_batches": validation_batches,
        "drop_last": False,
        "default_collate": True,
        "train_shuffle": True,
        "validation_shuffle": False,
        "deterministic_train_shuffle_generator": True,
        "deterministic_epoch_specific_train_shuffle": True,
        "legacy_set_epoch_use_preserved": True,
        "restart_at_epoch_boundary_reproducible": True,
        "all_odd_training_items_retained": True,
        "remaining_pair_text_inventory": remaining_pair_text_rows,
        "compute_loss_not_targeted": True,
        "optimizer_scheduler_clipping_not_targeted": True,
        "early_stop_not_targeted": True,
        "model_architecture_not_targeted": True,
        "feature_masking_not_targeted": True,
    }


def create_corrected_matrix_launcher(
    original_path: Path,
    corrected_path: Path,
    expected_original_sha256: str,
    corrected_trainer_path: Path,
) -> dict[str, Any]:
    require(
        sha256_file(original_path) == expected_original_sha256,
        "R5 matrix launcher SHA changed before correction",
    )

    original = original_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    old_line = '  mkdir -p "$MODEL_DIR" "$REPORT_DIR"\n'
    require(
        original.count(old_line) == 1,
        "expected exactly one premature model/report mkdir route",
    )

    replacement = (
        '  mkdir -p "$RUN_DIR"\n'
        '\n'
        '  if [[ -e "$MODEL_DIR" || -e "$REPORT_DIR" ]]; then\n'
        '    echo "STOP: incomplete F7 run output already exists: '
        '$RUN_DIR" >&2\n'
        '    echo "Remove or archive the incomplete run directory before '
        'retrying; no output will be overwritten." >&2\n'
        '    exit 1\n'
        '  fi\n'
    )
    corrected = original.replace(old_line, replacement, 1)

    trainer_lines = [
        line
        for line in corrected.splitlines()
        if line.startswith("TRAINER=")
    ]
    require(
        len(trainer_lines) == 1,
        f"matrix launcher TRAINER binding is not unique: {trainer_lines}",
    )
    corrected = corrected.replace(
        trainer_lines[0],
        "TRAINER=" + shlex.quote(str(corrected_trainer_path)),
        1,
    )

    require(
        'mkdir -p "$MODEL_DIR" "$REPORT_DIR"' not in corrected,
        "premature model/report mkdir survived launcher correction",
    )
    require(
        'if [[ -e "$MODEL_DIR" || -e "$REPORT_DIR" ]]' in corrected,
        "partial-run overwrite guard missing",
    )

    subprocess.run(
        ["bash", "-n"],
        input=corrected,
        text=True,
        check=True,
    )
    atomic_text(corrected_path, corrected, mode=0o555)

    return {
        "original_launcher": str(original_path),
        "original_launcher_sha256": sha256_file(original_path),
        "corrected_launcher": str(corrected_path),
        "corrected_launcher_sha256": sha256_file(corrected_path),
        "premature_model_report_mkdir_removed": True,
        "run_directory_only_precreated": True,
        "partial_run_overwrite_guard_added": True,
        "completed_run_skip_preserved": True,
        "trainer_binding": str(corrected_trainer_path),
        "trainer_binding_sha256": sha256_file(corrected_trainer_path),
        "corrected_trainer_binding_verified": True,
    }


def import_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(
        spec is not None and spec.loader is not None,
        f"cannot import {path}",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [ast.unparse(node)]
    if isinstance(node, (ast.Tuple, ast.List)):
        output = []
        for child in node.elts:
            output.extend(target_names(child))
        return output
    return []


def line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def absolute_span(
    node: ast.AST,
    offsets: list[int],
) -> tuple[int, int]:
    start = offsets[int(node.lineno) - 1] + int(node.col_offset)
    end = (
        offsets[int(node.end_lineno) - 1]
        + int(node.end_col_offset)
    )
    return start, end


def apply_replacements(
    text: str,
    replacements: list[tuple[int, int, str]],
) -> str:
    output = text
    for start, end, value in sorted(
        replacements,
        key=lambda row: row[0],
        reverse=True,
    ):
        output = output[:start] + value + output[end:]
    return output


def exact_function_source(
    text: str,
    tree: ast.Module,
    name: str,
) -> str:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    require(len(matches) == 1, f"{name} function is not unique")
    source = ast.get_source_segment(text, matches[0])
    require(source is not None, f"could not extract {name}")
    return source


def insertion_offset_after_future_imports(
    text: str,
    tree: ast.Module,
    offsets: list[int],
) -> int:
    insertion_line = 1
    body = list(tree.body)

    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        insertion_line = int(body[0].end_lineno) + 1

    for node in body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
        ):
            insertion_line = max(
                insertion_line,
                int(node.end_lineno) + 1,
            )

    return offsets[insertion_line - 1]


def parameter_state_hash(model: torch.nn.Module) -> str:
    """Canonical hash matching the runtime-adapter certificate."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def flatten_arrays(
    value: Any,
    path: str = "root",
    seen: set[int] | None = None,
) -> dict[str, Any]:
    if seen is None:
        seen = set()

    output: dict[str, Any] = {}

    if isinstance(value, torch.Tensor):
        output[path] = value.detach().cpu()
        return output

    if isinstance(value, np.ndarray):
        output[path] = np.array(value, copy=True)
        return output

    object_id = id(value)
    if object_id in seen:
        return output

    if isinstance(value, dict):
        seen.add(object_id)
        for key, child in value.items():
            output.update(
                flatten_arrays(
                    child,
                    f"{path}[{key!r}]",
                    seen,
                )
            )
        return output

    if isinstance(value, (list, tuple)):
        seen.add(object_id)
        for index, child in enumerate(value):
            output.update(
                flatten_arrays(
                    child,
                    f"{path}[{index}]",
                    seen,
                )
            )
        return output

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        seen.add(object_id)
        for field in dataclasses.fields(value):
            output.update(
                flatten_arrays(
                    getattr(value, field.name),
                    f"{path}.{field.name}",
                    seen,
                )
            )
        return output

    if hasattr(value, "__dict__"):
        seen.add(object_id)
        for key, child in vars(value).items():
            output.update(
                flatten_arrays(
                    child,
                    f"{path}.{key}",
                    seen,
                )
            )

    return output


def array_shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in value.shape)


def array_allclose_exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            return False
        return bool(torch.equal(left, right))

    if isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray):
            return False
        return bool(np.array_equal(left, right))

    return False


def array_nonzero_count(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(torch.count_nonzero(value).item())
    return int(np.count_nonzero(value))


def array_group_zero(
    value: Any,
    start: int,
    end: int,
) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.count_nonzero(value[:, start:end, :]).item() == 0)
    return bool(np.count_nonzero(value[:, start:end, :]) == 0)


def array_outside_equal(
    control: Any,
    masked: Any,
    start: int,
    end: int,
) -> bool:
    if isinstance(control, torch.Tensor):
        return bool(
            torch.equal(control[:, :start, :], masked[:, :start, :])
            and torch.equal(control[:, end:, :], masked[:, end:, :])
        )

    return bool(
        np.array_equal(control[:, :start, :], masked[:, :start, :])
        and np.array_equal(control[:, end:, :], masked[:, end:, :])
    )


def unique_shape_path(
    flattened: dict[str, Any],
    shape: tuple[int, ...],
) -> tuple[str, Any]:
    matches = [
        (path, value)
        for path, value in flattened.items()
        if array_shape(value) == shape
    ]
    require(
        len(matches) == 1,
        f"expected one leaf with shape {shape}, found "
        f"{[(path, array_shape(value)) for path, value in matches]}",
    )
    return matches[0]


def create_instrumented_trainer(
    trainer_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    text = trainer_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    tree = ast.parse(text)
    offsets = line_offsets(text)

    optimizer_assignments = []
    optimizer_step_calls = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = []
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    names.extend(target_names(target))
            else:
                names.extend(target_names(node.target))
            if "optimizer" in names:
                optimizer_assignments.append(node)

        if isinstance(node, ast.Call):
            if call_name(node.func) == "optimizer.step":
                optimizer_step_calls.append(node)

    require(
        len(optimizer_assignments) == 1,
        f"optimizer assignment not unique: "
        f"{[(node.lineno, ast.unparse(node)) for node in optimizer_assignments]}",
    )
    require(
        len(optimizer_step_calls) == 1,
        f"optimizer.step call not unique: "
        f"{[(node.lineno, ast.unparse(node)) for node in optimizer_step_calls]}",
    )

    helper_source = r'''
_F7_E2_INITIAL_PARAMETER_HASH = None


def _f7_e2_optimizer_parameter_hash(optimizer):
    import hashlib as _hashlib
    _digest = _hashlib.sha256()
    for _group_index, _group in enumerate(optimizer.param_groups):
        _digest.update(str(_group_index).encode("utf-8"))
        for _parameter_index, _parameter in enumerate(_group["params"]):
            _tensor = _parameter.detach().cpu().contiguous()
            _digest.update(str(_parameter_index).encode("utf-8"))
            _digest.update(str(_tensor.dtype).encode("utf-8"))
            _digest.update(str(tuple(_tensor.shape)).encode("utf-8"))
            _digest.update(_tensor.numpy().tobytes())
    return _digest.hexdigest()


def _f7_e2_capture_initial_optimizer_state(optimizer):
    global _F7_E2_INITIAL_PARAMETER_HASH
    _F7_E2_INITIAL_PARAMETER_HASH = (
        _f7_e2_optimizer_parameter_hash(optimizer)
    )


def _f7_e2_intercept_optimizer_step(optimizer):
    import json as _json
    import math as _math
    import os as _os
    from pathlib import Path as _Path

    _gradient_square_sum = 0.0
    _gradient_tensor_count = 0
    _all_gradients_finite = True

    for _group in optimizer.param_groups:
        for _parameter in _group["params"]:
            if _parameter.grad is None:
                continue
            _gradient = _parameter.grad.detach()
            _gradient_tensor_count += 1
            _all_gradients_finite = (
                _all_gradients_finite
                and bool(torch.isfinite(_gradient).all().item())
            )
            _gradient_square_sum += float(
                torch.sum(_gradient.float() * _gradient.float()).item()
            )

    _current_hash = _f7_e2_optimizer_parameter_hash(optimizer)
    _payload = {
        "sentinel": "F7_E2_OPTIMIZER_STEP_INTERCEPTED",
        "optimizer_step_performed": False,
        "initial_parameter_hash": _F7_E2_INITIAL_PARAMETER_HASH,
        "intercept_parameter_hash": _current_hash,
        "parameters_unchanged_before_step": (
            _F7_E2_INITIAL_PARAMETER_HASH == _current_hash
        ),
        "gradient_tensor_count": _gradient_tensor_count,
        "gradient_l2": _math.sqrt(_gradient_square_sum),
        "all_gradients_finite": _all_gradients_finite,
    }

    _path = _Path(_os.environ["F7_E2_INTERCEPT_JSON"])
    _path.parent.mkdir(parents=True, exist_ok=True)
    _path.write_text(
        _json.dumps(_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("F7_E2_OPTIMIZER_STEP_INTERCEPTED", flush=True)
    raise SystemExit(86)
'''

    replacements: list[tuple[int, int, str]] = []

    insertion = insertion_offset_after_future_imports(
        text,
        tree,
        offsets,
    )
    replacements.append((insertion, insertion, helper_source))

    optimizer_assignment = optimizer_assignments[0]
    _, assignment_end = absolute_span(
        optimizer_assignment,
        offsets,
    )
    source_line_start = text.rfind("\n", 0, assignment_end) + 1
    prefix = text[source_line_start:assignment_end]
    match = re.match(r"[ \t]*", prefix)
    require(match is not None, "could not derive optimizer indentation")
    indentation = match.group(0)

    replacements.append(
        (
            assignment_end,
            assignment_end,
            (
                "\n"
                + indentation
                + "_f7_e2_capture_initial_optimizer_state(optimizer)"
            ),
        )
    )

    step_call = optimizer_step_calls[0]
    step_start, step_end = absolute_span(step_call, offsets)
    replacements.append(
        (
            step_start,
            step_end,
            "_f7_e2_intercept_optimizer_step(optimizer)",
        )
    )

    instrumented = apply_replacements(text, replacements)
    ast.parse(instrumented)
    compile(instrumented, str(output_path), "exec")
    atomic_text(output_path, instrumented, mode=0o555)

    return {
        "optimizer_assignment_line": int(optimizer_assignment.lineno),
        "optimizer_step_line": int(step_call.lineno),
        "instrumented_trainer": str(output_path),
        "instrumented_trainer_sha256": sha256_file(output_path),
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"missing repository: {repo}")
    require(data_link.is_symlink(), f"missing dataset symlink: {data_link}")
    require(data_link.resolve().is_dir(), "dataset target missing")

    r5_root = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/"
        "retrained_group_ablation"
    )
    r5_report_path = r5_root / (
        "V5_P3_F7_E1_R5_RUNTIME_IMPORT_USE_SITE_PATCH_"
        "AND_GENERATED_TRAINER_RECOVERY_REPORT.json"
    )
    r5_lock_path = r5_root / (
        "V5_P3_F7_E1_R5_RUNTIME_IMPORT_USE_SITE_PATCH_"
        "AND_GENERATED_TRAINER_RECOVERY_LOCK.json"
    )

    r5_report = load_json(r5_report_path)
    r5_lock = load_json(r5_lock_path)

    require(r5_report.get("status") == "PASS", "R5 report not PASS")
    require(
        r5_lock.get("report_sha256") == sha256_file(r5_report_path),
        "R5 report/lock mismatch",
    )
    require(
        r5_lock.get("E2_generated_trainer_preflight_authorized") is True,
        "E2 preflight is not authorized",
    )
    require(
        r5_lock.get("primary_matrix_execution_authorized") is False,
        "matrix execution unexpectedly authorized before E2",
    )
    require(
        r5_lock.get("sealed_test_tensors_loaded") is False,
        "sealed-test access detected in R5",
    )

    artifacts = r5_report["artifacts"]
    trainer_path = Path(artifacts["generated_trainer"]).resolve()
    adapter_path = Path(artifacts["runtime_adapter"]).resolve()
    launcher_path = Path(artifacts["matrix_launcher"]).resolve()
    matrix_manifest_path = Path(
        artifacts["matrix_manifest"]
    ).resolve()

    require(trainer_path.is_file(), "generated trainer missing")
    require(adapter_path.is_file(), "runtime adapter missing")
    require(launcher_path.is_file(), "matrix launcher missing")
    require(matrix_manifest_path.is_file(), "matrix manifest missing")

    original_trainer_path = trainer_path
    require(
        sha256_file(original_trainer_path)
        == r5_lock["generated_trainer_sha256"],
        "generated trainer SHA changed",
    )

    cardinality_corrected_trainer_path = (
        original_trainer_path.parent
        / "run_v5_p3_f7_reconstructed_group_ablation_trainer_e2_r4_r3_cardinality.py"
    )
    cardinality_patch = create_corrected_cardinality_trainer(
        original_trainer_path,
        cardinality_corrected_trainer_path,
        r5_lock["generated_trainer_sha256"],
    )

    corrected_trainer_path = (
        original_trainer_path.parent
        / "run_v5_p3_f7_reconstructed_group_ablation_trainer_e2_r4_r3.py"
    )
    item_batch_patch = create_item_batch_route_trainer(
        cardinality_corrected_trainer_path,
        corrected_trainer_path,
        sha256_file(cardinality_corrected_trainer_path),
    )
    trainer_path = corrected_trainer_path

    require(
        sha256_file(adapter_path)
        == r5_lock["runtime_adapter_sha256"],
        "runtime adapter SHA changed",
    )
    original_launcher_path = launcher_path
    corrected_launcher_path = (
        original_launcher_path.parent
        / "run_v5_p3_f7_primary_seed107_matrix_e2_r4_r3.sh"
    )
    require(
        "shlex" in globals(),
        "shlex import is unavailable before launcher correction",
    )
    launcher_patch = create_corrected_matrix_launcher(
        original_launcher_path,
        corrected_launcher_path,
        r5_lock["matrix_launcher_sha256"],
        trainer_path,
    )
    launcher_patch["shlex_import_verified"] = True
    launcher_path = corrected_launcher_path

    trainer_text = trainer_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    trainer_tree = ast.parse(trainer_text)
    adapter_text = adapter_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    ast.parse(adapter_text)

    compute_loss_source = exact_function_source(
        trainer_text,
        trainer_tree,
        "compute_loss",
    )
    require(
        sha256_text(compute_loss_source)
        == EXPECTED_COMPUTE_LOSS_SHA256,
        "generated compute_loss SHA changed",
    )

    early_conditions = [
        ast.unparse(node.test)
        for node in ast.walk(trainer_tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Break)
            for child in ast.walk(node)
        )
        and "early_stop_patience_counter"
        in ast.unparse(node.test)
    ]
    require(
        early_conditions == [EXPECTED_EARLY_STOP],
        f"early-stop condition changed: {early_conditions}",
    )

    forbidden_imports = [
        ast.unparse(node)
        for node in ast.walk(trainer_tree)
        if isinstance(node, ast.Call)
        and call_name(node.func).rsplit(".", 1)[-1] == "import_module"
        and (
            "loader_path" in ast.unparse(node)
            or "model_path" in ast.unparse(node)
        )
    ]
    require(
        not forbidden_imports,
        f"legacy runtime import survived: {forbidden_imports}",
    )

    adapter_call_counts = {
        "F7Dynamic70ModelFactory": 0,
        "F7TrainDatasetAdapter": 0,
        "F7ValidationDatasetAdapter": 0,
    }
    for node in ast.walk(trainer_tree):
        if not isinstance(node, ast.Call):
            continue
        short = call_name(node.func).rsplit(".", 1)[-1]
        if short in adapter_call_counts:
            adapter_call_counts[short] += 1

    require(
        adapter_call_counts == {
            "F7Dynamic70ModelFactory": 1,
            "F7TrainDatasetAdapter": 1,
            "F7ValidationDatasetAdapter": 1,
        },
        f"adapter call counts changed: {adapter_call_counts}",
    )

    optimizer_step_calls = [
        node
        for node in ast.walk(trainer_tree)
        if isinstance(node, ast.Call)
        and call_name(node.func) == "optimizer.step"
    ]
    clip_calls = [
        node
        for node in ast.walk(trainer_tree)
        if isinstance(node, ast.Call)
        and call_name(node.func).rsplit(".", 1)[-1] == "clip_grad_norm_"
    ]
    require(
        len(optimizer_step_calls) == 1,
        "optimizer.step is not unique",
    )
    require(
        len(clip_calls) == 1,
        "clip_grad_norm_ is not unique",
    )
    require(
        int(clip_calls[0].lineno)
        < int(optimizer_step_calls[0].lineno),
        "gradient clipping is not before optimizer.step",
    )

    launcher_text = launcher_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    require(
        "primary_matrix_execution_authorized" in launcher_text,
        "launcher execution guard missing",
    )
    require(
        "sealed_test_tensors_loaded" in launcher_text,
        "launcher sealed-test guard missing",
    )
    subprocess.run(
        ["bash", "-n", str(launcher_path)],
        check=True,
    )

    scratch = output_dir / "F7_E2_R4_R3_RUNTIME_PREFLIGHT_SCRATCH"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    environment = os.environ.copy()
    environment.update({
        "F7_REPO": str(repo),
        "F7_DATA_LINK": str(data_link),
        "F7_SEED": str(EXPECTED_SEED),
        "F7_ABLATION_LABEL": "control_dynamic70",
    })

    adapter = import_source(
        adapter_path,
        "_v5_p3_f7_e2_runtime_adapter",
    )

    model_hashes = []
    model_certificates = []

    for run_number in (1, 2):
        model_run_dir = scratch / f"model_determinism_{run_number}"
        model_run_dir.mkdir(parents=True)
        os.environ.update(environment)
        os.environ["F7_RUN_DIR"] = str(model_run_dir)
        model = adapter.F7Dynamic70ModelFactory()

        parameter_count = int(
            sum(parameter.numel() for parameter in model.parameters())
        )
        require(
            parameter_count == EXPECTED_PARAMETER_COUNT,
            f"parameter count changed: {parameter_count}",
        )

        state_hash = parameter_state_hash(model)
        model_hashes.append(state_hash)

        certificate_path = (
            model_run_dir / "F7_MODEL_INITIALIZATION_CERTIFICATION.json"
        )
        require(
            certificate_path.is_file(),
            "model initialization certificate missing",
        )
        certificate = load_json(certificate_path)
        require(
            certificate["parameter_count"] == EXPECTED_PARAMETER_COUNT,
            "certificate parameter count changed",
        )
        require(
            certificate["seed"] == EXPECTED_SEED,
            "certificate seed changed",
        )
        require(
            certificate["fresh_reset"]["fresh_state_hash"]
            == state_hash,
            "certificate fresh-state hash mismatch",
        )
        model_certificates.append({
            "path": str(certificate_path),
            "sha256": sha256_file(certificate_path),
            "fresh_state_hash": state_hash,
        })
        del model

    require(
        model_hashes[0] == model_hashes[1],
        "fresh seed-107 model initialization is not deterministic",
    )

    dataset_run_dir = scratch / "dataset_mask_audit"
    dataset_run_dir.mkdir(parents=True)
    os.environ.update(environment)
    os.environ["F7_RUN_DIR"] = str(dataset_run_dir)
    os.environ["F7_ABLATION_LABEL"] = "control_dynamic70"

    control_train = adapter.F7TrainDatasetAdapter()
    control_validation = adapter.F7ValidationDatasetAdapter()

    require(
        len(control_train) == EXPECTED_TRAIN_ITEMS,
        f"train length changed: {len(control_train)}",
    )
    require(
        len(control_validation) == EXPECTED_VALIDATION_ITEMS,
        f"validation length changed: {len(control_validation)}",
    )

    validation_sample = control_validation[0]
    validation_flat = flatten_arrays(validation_sample)
    validation_x_path, validation_x = unique_shape_path(
        validation_flat,
        (16, 70, 32),
    )
    validation_mask_path, validation_mask = unique_shape_path(
        validation_flat,
        (16, 10),
    )

    masking_results = []

    for label, channel_range in GROUPS.items():
        if channel_range is None:
            continue

        start, end = channel_range
        os.environ["F7_ABLATION_LABEL"] = label
        masked_train = adapter.F7TrainDatasetAdapter()

        selected = None
        for index in range(min(512, len(control_train))):
            control_item = control_train[index]
            masked_item = masked_train[index]

            control_flat = flatten_arrays(control_item)
            masked_flat = flatten_arrays(masked_item)

            control_x_path, control_x = unique_shape_path(
                control_flat,
                (16, 70, 32),
            )
            masked_x_path, masked_x = unique_shape_path(
                masked_flat,
                (16, 70, 32),
            )

            require(
                control_x_path == masked_x_path,
                "feature tensor path changed under masking",
            )

            if array_nonzero_count(control_x[:, start:end, :]) == 0:
                continue

            require(
                array_group_zero(masked_x, start, end),
                f"{label} target group is not all zero",
            )
            require(
                array_outside_equal(
                    control_x,
                    masked_x,
                    start,
                    end,
                ),
                f"{label} changed values outside its group",
            )

            control_mask_path, control_mask = unique_shape_path(
                control_flat,
                (16, 10),
            )
            masked_mask_path, masked_mask = unique_shape_path(
                masked_flat,
                (16, 10),
            )
            require(
                control_mask_path == masked_mask_path,
                "physical-mask path changed",
            )
            require(
                array_allclose_exact(control_mask, masked_mask),
                "physical port mask changed under ablation",
            )

            control_paths = set(control_flat)
            masked_paths = set(masked_flat)
            require(
                control_paths == masked_paths,
                "tensor/array leaf structure changed under masking",
            )

            for path in sorted(control_paths):
                if path == control_x_path:
                    continue
                require(
                    array_allclose_exact(
                        control_flat[path],
                        masked_flat[path],
                    ),
                    f"{label} changed non-feature leaf {path}",
                )

            changed_count = array_nonzero_count(
                control_x[:, start:end, :]
            )
            selected = {
                "label": label,
                "channel_range": [start, end],
                "sample_index": index,
                "feature_path": control_x_path,
                "physical_mask_path": control_mask_path,
                "nonzero_values_zeroed": changed_count,
                "outside_group_exact": True,
                "physical_mask_exact": True,
                "all_other_array_leaves_exact": True,
            }
            break

        require(
            selected is not None,
            f"could not find a nonzero audit sample for {label}",
        )
        masking_results.append(selected)
        del masked_train

    del control_train
    del control_validation

    instrumented_path = scratch / (
        "run_v5_p3_f7_instrumented_first_batch_preflight.py"
    )
    instrumentation = create_instrumented_trainer(
        trainer_path,
        instrumented_path,
    )

    runtime_run_dir = scratch / "instrumented_trainer_runtime"
    runtime_run_dir.mkdir(parents=True)
    model_dir = runtime_run_dir / "model"
    report_dir = runtime_run_dir / "report"
    require(
        not model_dir.exists(),
        f"preflight model directory unexpectedly exists: {model_dir}",
    )
    require(
        not report_dir.exists(),
        f"preflight report directory unexpectedly exists: {report_dir}",
    )

    intercept_json = runtime_run_dir / "F7_E2_INTERCEPT_RESULT.json"

    legacy = r5_report["finding"]["patch_manifest"][
        "legacy_compatibility"
    ]
    b1_dir = Path(legacy["b1_dir"]).resolve()
    b0_r3_dir = Path(legacy["b0_r3_dir"]).resolve()
    legacy_loader = Path(legacy["loader_path"]).resolve()
    legacy_model = Path(legacy["model_path"]).resolve()

    runtime_environment = environment.copy()
    runtime_environment.update({
        "F7_RUN_DIR": str(runtime_run_dir),
        "F7_ABLATION_LABEL": "control_dynamic70",
        "F7_E2_INTERCEPT_JSON": str(intercept_json),
        "PYTHONPATH": (
            str(repo)
            + os.pathsep
            + runtime_environment.get("PYTHONPATH", "")
        ),
    })

    command = [
        sys.executable,
        str(instrumented_path),
        "--root",
        str(repo),
        "--b1-dir",
        str(b1_dir),
        "--b0-r3-dir",
        str(b0_r3_dir),
        "--loader-path",
        str(legacy_loader),
        "--model-path",
        str(legacy_model),
        "--model-dir",
        str(model_dir),
        "--report-dir",
        str(report_dir),
        "--seed",
        str(EXPECTED_SEED),
    ]

    process = subprocess.run(
        command,
        cwd=repo,
        env=runtime_environment,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    console_path = runtime_run_dir / "console.log"
    atomic_text(
        console_path,
        process.stdout + "\n--- STDERR ---\n" + process.stderr,
    )

    require(
        process.returncode == INTERCEPT_EXIT_CODE,
        "instrumented trainer did not reach the intercepted optimizer step; "
        f"returncode={process.returncode}; console={console_path}",
    )
    require(
        "F7_E2_OPTIMIZER_STEP_INTERCEPTED"
        in process.stdout + process.stderr,
        "optimizer-step intercept sentinel missing",
    )
    require(
        intercept_json.is_file(),
        "optimizer-step intercept JSON missing",
    )

    intercept = load_json(intercept_json)
    require(
        intercept["optimizer_step_performed"] is False,
        "optimizer step was performed",
    )
    require(
        intercept["parameters_unchanged_before_step"] is True,
        "parameters changed before intercepted optimizer step",
    )
    require(
        intercept["gradient_tensor_count"] > 0,
        "no parameter gradients were produced",
    )
    require(
        intercept["gradient_l2"] > 0.0
        and math.isfinite(intercept["gradient_l2"]),
        "gradient L2 is invalid",
    )
    require(
        intercept["all_gradients_finite"] is True,
        "nonfinite gradients detected",
    )

    scientific_checkpoint_candidates = []
    for suffix in ("*.pt", "*.pth", "*.ckpt"):
        scientific_checkpoint_candidates.extend(
            runtime_run_dir.rglob(suffix)
        )
    require(
        not scientific_checkpoint_candidates,
        "instrumented preflight created a scientific checkpoint: "
        f"{scientific_checkpoint_candidates}",
    )

    runtime_certificate = (
        runtime_run_dir / "F7_MODEL_INITIALIZATION_CERTIFICATION.json"
    )
    require(
        runtime_certificate.is_file(),
        "instrumented trainer model certificate missing",
    )

    static_results = {
        "recovery_classification": (
            "R4_R2_expected_433_train_batches_but_110855_items_at_batch_size_256_require_434"
        ),
        "prior_AST_numeric_literal_recovery_preserved": True,
        "prior_output_directory_recovery_preserved": True,
        "cardinality_patch": cardinality_patch,
        "item_batch_patch": item_batch_patch,
        "launcher_patch": launcher_patch,
        "generated_trainer_sha256": sha256_file(trainer_path),
        "runtime_adapter_sha256": sha256_file(adapter_path),
        "matrix_launcher_sha256": sha256_file(launcher_path),
        "compute_loss_sha256": sha256_text(compute_loss_source),
        "early_stop_condition": early_conditions[0],
        "legacy_dynamic_import_call_count": 0,
        "adapter_call_counts": adapter_call_counts,
        "optimizer_step_call_count": len(optimizer_step_calls),
        "gradient_clip_call_count": len(clip_calls),
        "gradient_clip_before_optimizer_step": True,
        "launcher_execution_guard_present": True,
        "launcher_sealed_test_guard_present": True,
    }

    runtime_results = {
        "recovery_classification": (
            "E2_compared_parameter_only_hash_against_complete_state_dict_hash"
        ),
        "canonical_state_hash_algorithm": (
            "sorted_state_dict_key_dtype_shape_and_contiguous_cpu_bytes"
        ),
        "R2_recovery_classification": (
            "E2_R1_precreated_model_and_report_directories_before_trainer"
        ),
        "R3A_recovery_classification": (
            "prior_AST_numeric_literal_cardinality_recovery"
        ),
        "corrected_trainer_train_items": EXPECTED_TRAIN_ITEMS,
        "corrected_trainer_validation_items": EXPECTED_VALIDATION_ITEMS,
        "effective_item_batch_size": EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE,
        "train_batch_arithmetic": {
            "full_batches": 110855 // 256,
            "remainder_items": 110855 % 256,
            "ceil_batches": math.ceil(110855 / 256),
            "proof": "433_full_batches_plus_one_partial_batch_of_7_items",
        },
        "train_batches": EXPECTED_TRAIN_BATCHES,
        "validation_batches": EXPECTED_VALIDATION_BATCHES,
        "pair_block_sampler_active": False,
        "epoch_shuffle_controller_active": True,
        "legacy_set_epoch_use_preserved": True,
        "deterministic_epoch_specific_shuffle": True,
        "all_odd_training_items_retained": True,
        "instrumented_model_dir_absent_at_launch": True,
        "instrumented_report_dir_absent_at_launch": True,
        "model_class": "V6P0Dynamic70GraphConvCount4",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "seed": EXPECTED_SEED,
        "deterministic_fresh_state_hash": model_hashes[0],
        "two_model_initializations_exact_match": True,
        "model_certificates": model_certificates,
        "train_items": EXPECTED_TRAIN_ITEMS,
        "validation_items": EXPECTED_VALIDATION_ITEMS,
        "validation_sample_feature_path": validation_x_path,
        "validation_sample_feature_shape": list(
            array_shape(validation_x)
        ),
        "validation_sample_mask_path": validation_mask_path,
        "validation_sample_mask_shape": list(
            array_shape(validation_mask)
        ),
        "masking_results": masking_results,
        "instrumentation": instrumentation,
        "instrumented_command": command,
        "instrumented_return_code": process.returncode,
        "intercept": intercept,
        "console": str(console_path),
        "console_sha256": sha256_file(console_path),
        "runtime_model_certificate": str(runtime_certificate),
        "runtime_model_certificate_sha256": sha256_file(
            runtime_certificate
        ),
        "scientific_checkpoint_saved": False,
        "optimizer_step_performed": False,
    }

    cardinality_patch_path = output_dir / (
        "F7_E2_R4_R3_DATASET_CARDINALITY_GUARD_PATCH.json"
    )
    atomic_json(cardinality_patch_path, cardinality_patch)

    item_batch_patch_path = output_dir / (
        "F7_E2_R4_R3_PAIRBLOCK_TO_ITEMBATCH_ROUTE_PATCH.json"
    )
    atomic_json(item_batch_patch_path, item_batch_patch)

    launcher_patch_path = output_dir / (
        "F7_E2_R4_R3_CORRECTED_MATRIX_LAUNCHER_PATCH.json"
    )
    atomic_json(launcher_patch_path, launcher_patch)

    static_path = output_dir / (
        "F7_E2_R4_R3_GENERATED_TRAINER_STATIC_CERTIFICATION.json"
    )
    runtime_path = output_dir / (
        "F7_E2_R4_R3_RUNTIME_ADAPTER_AND_FIRST_BATCH_CERTIFICATION.json"
    )
    decision_path = output_dir / (
        "F7_E2_R4_R3_PRIMARY_MATRIX_EXECUTION_DECISION.json"
    )

    atomic_json(static_path, static_results)
    atomic_json(runtime_path, runtime_results)

    decision = {
        "E2_R3A_R1_complete": True,
        "missing_shlex_import_recovered": True,
        "prior_AST_numeric_literal_recovery_preserved": True,
        "E2_R1_complete": True,
        "recovery_classification": (
            "E2_compared_parameter_only_hash_against_complete_state_dict_hash"
        ),
        "canonical_state_hash_match_verified": True,
        "R2_nonexistent_output_directory_contract_verified": True,
        "R3A_dataset_cardinality_guard_recovered": True,
        "corrected_trainer_generated": True,
        "pair_block_sampler_removed": True,
        "item_level_batch_route_verified": True,
        "frozen_batch_geometry_uniquely_resolves_256": True,
        "sampler_parameter_name_not_used_for_authorization": True,
        "epoch_shuffle_controller_verified": True,
        "legacy_set_epoch_use_preserved": True,
        "deterministic_epoch_specific_shuffle_verified": True,
        "train_batch_arithmetic_corrected": True,
        "train_full_batches": 433,
        "train_remainder_items": 7,
        "train_total_batches": 434,
        "effective_item_batch_size_preserved": True,
        "odd_training_cardinality_supported": True,
        "corrected_trainer_bound_in_matrix_launcher": True,
        "AST_numeric_literal_patch_verified": True,
        "corrected_matrix_launcher_generated": True,
        "partial_run_overwrite_guard_verified": True,
        "E2_complete": True,
        "generated_trainer_static_preflight_passed": True,
        "runtime_adapter_preflight_passed": True,
        "model_determinism_verified": True,
        "train_validation_split_lengths_verified": True,
        "all_five_group_masks_verified": True,
        "physical_port_mask_preserved": True,
        "first_batch_forward_loss_backward_reached": True,
        "finite_nonzero_gradients_verified": True,
        "optimizer_step_intercepted_before_execution": True,
        "scientific_checkpoint_saved": False,
        "primary_matrix_execution_authorized": True,
        "actual_scientific_training_started": False,
        "actual_F7_primary_matrix_execution_started": False,
        "F8_multi_seed_authorized": False,
        "feature_removal_authorized": False,
        "compact_interface_equivalence_authorized": False,
        "hardware_reduction_claim_authorized": False,
        "sealed_test_access": False,
        "next_stage": (
            "V5_P3_F7_PRIMARY_SEED107_SAME_WIDTH_RETRAINED_"
            "GROUP_ABLATION_MATRIX_EXECUTION"
        ),
    }
    atomic_json(decision_path, decision)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "scope": (
            "Verify the generated trainer, runtime adapter and guarded matrix "
            "launcher statically; instantiate the certified Dynamic70 model "
            "twice to prove deterministic fresh seed-107 initialization; "
            "construct guarded train and validation datasets; verify all five "
            "same-width group masks on real training samples while preserving "
            "the physical mask and every non-feature array leaf; and execute "
            "an instrumented copy of the generated trainer through the first "
            "forward, exact compute_loss, backward and gradient-clipping path. "
            "The unique optimizer.step call is replaced by a sentinel that "
            "hashes parameters, records finite gradients and exits before any "
            "parameter update. No scientific checkpoint or sealed-test data "
            "is created or accessed."
        ),
        "finding": {
            "static": static_results,
            "runtime": runtime_results,
        },
        "decision": decision,
        "governance": {
            "model_loaded": True,
            "checkpoint_loaded_for_architecture_capture": True,
            "checkpoint_weights_retained_for_training": False,
            "training_dataset_constructed": True,
            "validation_dataset_constructed": True,
            "training_feature_tensors_accessed": True,
            "validation_feature_tensors_accessed": True,
            "legacy_loader_module_executed": False,
            "legacy_model_module_executed": False,
            "optimizer_step_performed": False,
            "scientific_checkpoint_saved": False,
            "scientific_training_started": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
        },
        "artifacts": {
            "static_certification": str(static_path),
            "runtime_certification": str(runtime_path),
            "execution_decision": str(decision_path),
            "corrected_trainer": str(trainer_path),
            "cardinality_patch": str(cardinality_patch_path),
            "item_batch_patch": str(item_batch_patch_path),
            "corrected_matrix_launcher": str(launcher_path),
            "launcher_patch": str(launcher_patch_path),
            "scratch": str(scratch),
        },
        "provenance": {
            "R5_report_sha256": sha256_file(r5_report_path),
            "R5_lock_sha256": sha256_file(r5_lock_path),
            "original_R5_generated_trainer_sha256": (
                cardinality_patch["original_trainer_sha256"]
            ),
            "generated_trainer_sha256": sha256_file(trainer_path),
            "cardinality_patch_sha256": sha256_file(cardinality_patch_path),
            "item_batch_patch_sha256": sha256_file(item_batch_patch_path),
            "runtime_adapter_sha256": sha256_file(adapter_path),
            "original_R5_matrix_launcher_sha256": (
                launcher_patch["original_launcher_sha256"]
            ),
            "matrix_launcher_sha256": sha256_file(launcher_path),
            "launcher_patch_sha256": sha256_file(launcher_patch_path),
            "matrix_manifest_sha256": sha256_file(
                matrix_manifest_path
            ),
            "installed_script_sha256": sha256_file(
                installed_script
            ),
            "static_certification_sha256": sha256_file(
                static_path
            ),
            "runtime_certification_sha256": sha256_file(
                runtime_path
            ),
            "execution_decision_sha256": sha256_file(
                decision_path
            ),
        },
    }

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    atomic_json(report_path, report)
    atomic_json(
        lock_path,
        {
            "stage": STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(report_path),
            "static_certification_sha256": sha256_file(static_path),
            "runtime_certification_sha256": sha256_file(runtime_path),
            "execution_decision_sha256": sha256_file(decision_path),
            "generated_trainer_sha256": sha256_file(trainer_path),
            "cardinality_patch_sha256": sha256_file(cardinality_patch_path),
            "item_batch_patch_sha256": sha256_file(item_batch_patch_path),
            "runtime_adapter_sha256": sha256_file(adapter_path),
            "matrix_launcher_sha256": sha256_file(launcher_path),
            "launcher_patch_sha256": sha256_file(launcher_patch_path),
            "R2_nonexistent_output_directory_contract_verified": True,
            "R3A_dataset_cardinality_guard_recovered": True,
            "corrected_trainer_bound_in_matrix_launcher": True,
            "E2_complete": True,
            "primary_matrix_execution_authorized": True,
            "actual_scientific_training_started": False,
            "actual_F7_primary_matrix_execution_started": False,
            "optimizer_step_performed": False,
            "scientific_checkpoint_saved": False,
            "F8_multi_seed_authorized": False,
            "feature_removal_authorized": False,
            "compact_interface_equivalence_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    canonical_lock_path = output_dir / (
        "V5_P3_F7_E2_GENERATED_TRAINER_STATIC_AND_RUNTIME_PREFLIGHT_LOCK.json"
    )
    require(
        not canonical_lock_path.exists(),
        f"canonical E2 authorization lock already exists: {canonical_lock_path}",
    )
    atomic_json(
        canonical_lock_path,
        {
            "stage": (
                "V5_P3_F7_E2_GENERATED_TRAINER_"
                "STATIC_AND_RUNTIME_PREFLIGHT"
            ),
            "authorization_source_stage": STAGE,
            "status": "PASS",
            "source_report": str(report_path),
            "source_report_sha256": sha256_file(report_path),
            "source_lock": str(lock_path),
            "source_lock_sha256": sha256_file(lock_path),
            "generated_trainer_sha256": sha256_file(trainer_path),
            "cardinality_patch_sha256": sha256_file(cardinality_patch_path),
            "item_batch_patch_sha256": sha256_file(item_batch_patch_path),
            "runtime_adapter_sha256": sha256_file(adapter_path),
            "matrix_launcher": str(launcher_path),
            "matrix_launcher_sha256": sha256_file(launcher_path),
            "launcher_patch_sha256": sha256_file(launcher_patch_path),
            "cardinality_patch_sha256": sha256_file(cardinality_patch_path),
            "item_batch_patch_sha256": sha256_file(item_batch_patch_path),
            "R3A_dataset_cardinality_guard_recovered": True,
            "R4_R2_pair_block_sampler_removed": True,
            "R4_R2_item_level_batch_route_verified": True,
            "effective_item_batch_size": EXPECTED_EFFECTIVE_ITEM_BATCH_SIZE,
            "train_batches": EXPECTED_TRAIN_BATCHES,
            "validation_batches": EXPECTED_VALIDATION_BATCHES,
            "corrected_trainer_bound_in_matrix_launcher": True,
            "primary_matrix_execution_authorized": True,
            "actual_scientific_training_started": False,
            "actual_F7_primary_matrix_execution_started": False,
            "optimizer_step_performed": False,
            "scientific_checkpoint_saved": False,
            "F8_multi_seed_authorized": False,
            "feature_removal_authorized": False,
            "compact_interface_equivalence_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print(
        "recovery_classification="
        "R4_assumed_semantic_pair_count_parameter_name_but_sampler_"
        "signature_uses_a_neutral_parameter_name"
    )
    print("missing_shlex_import_recovered=true")
    print("shlex_import_verified=true")
    print("prior_AST_numeric_literal_recovery_preserved=true")
    print(
        "prior_recoveries_preserved="
        "canonical_state_hash_and_nonexistent_output_directory_contract"
    )
    print(
        "legacy_split_cardinality_replacement="
        "train_70166_to_110855:validation_12528_to_13863"
    )
    print(
        "cardinality_patch_scope="
        "AST_integer_constants_by_evaluated_value_not_textual_spelling"
    )
    print("textual_spelling_independent=true")
    print(
        "canonical_state_hash_algorithm="
        "sorted_state_dict_key_dtype_shape_and_contiguous_cpu_bytes"
    )
    print("canonical_state_hash_match_verified=true")
    print(
        "original_R5_generated_trainer="
        f"{cardinality_patch['original_trainer']}"
    )
    print(
        "original_R5_generated_trainer_sha256="
        f"{cardinality_patch['original_trainer_sha256']}"
    )
    print(f"corrected_trainer={trainer_path}")
    print(f"generated_trainer_sha256={sha256_file(trainer_path)}")
    print("corrected_trainer_train_items=110855")
    print("corrected_trainer_validation_items=13863")
    print("pair_block_sampler_active=false")
    print("item_level_batch_route_verified=true")
    print("epoch_shuffle_controller_active=true")
    print("legacy_set_epoch_use_preserved=true")
    print("train_loader_generator=train_sampler.generator")
    print("epoch_seed_formula=base_seed_plus_epoch")
    print("deterministic_epoch_specific_shuffle=true")
    print("restart_at_epoch_boundary_reproducible=true")
    print("batch_size_resolution=unique_frozen_batch_geometry")
    print("sampler_parameter_name_required_for_patch=false")
    print("legacy_train_geometry=ceil_70166_over_256_equals_275")
    print("legacy_validation_geometry=ceil_12528_over_256_equals_49")
    print("effective_item_batch_size=256")
    print("train_full_batches=433")
    print("train_remainder_items=7")
    print("train_total_batches=434")
    print("train_batch_arithmetic=433_full_plus_1_partial_equals_434")
    print("effective_item_batch_size_derivation=2_times_128_pairs")
    print("train_batches=434")
    print("validation_batches=55")
    print("drop_last=false")
    print("default_collate=true")
    print("all_odd_training_items_retained=true")
    print(f"runtime_adapter={adapter_path}")
    print(f"runtime_adapter_sha256={sha256_file(adapter_path)}")
    print(
        "original_R5_matrix_launcher="
        f"{launcher_patch['original_launcher']}"
    )
    print(
        "original_R5_matrix_launcher_sha256="
        f"{launcher_patch['original_launcher_sha256']}"
    )
    print(f"corrected_matrix_launcher={launcher_path}")
    print(f"matrix_launcher_sha256={sha256_file(launcher_path)}")
    print("premature_model_report_mkdir_removed=true")
    print("partial_run_overwrite_guard_added=true")
    print("instrumented_model_dir_absent_at_launch=true")
    print("instrumented_report_dir_absent_at_launch=true")
    print(f"compute_loss_sha256={sha256_text(compute_loss_source)}")
    print(f"early_stop_condition={early_conditions[0]}")
    print("legacy_dynamic_import_call_count=0")
    print(f"adapter_call_counts={adapter_call_counts}")
    print(f"parameter_count={EXPECTED_PARAMETER_COUNT}")
    print(f"seed={EXPECTED_SEED}")
    print(
        "deterministic_fresh_state_hash="
        f"{model_hashes[0]}"
    )
    print("two_model_initializations_exact_match=true")
    print(f"train_items={EXPECTED_TRAIN_ITEMS}")
    print(f"validation_items={EXPECTED_VALIDATION_ITEMS}")

    for row in masking_results:
        print(
            "group_mask_verified="
            f"{row['label']}:"
            f"channels={row['channel_range']}:"
            f"sample_index={row['sample_index']}:"
            f"nonzero_values_zeroed={row['nonzero_values_zeroed']}:"
            "outside_group_exact=true:"
            "physical_mask_exact=true:"
            "other_array_leaves_exact=true"
        )

    print("first_batch_forward_loss_backward_reached=true")
    print(
        "gradient_tensor_count="
        f"{intercept['gradient_tensor_count']}"
    )
    print(f"gradient_l2={intercept['gradient_l2']}")
    print("all_gradients_finite=true")
    print("parameters_unchanged_before_step=true")
    print("optimizer_step_performed=false")
    print("scientific_checkpoint_saved=false")
    print("primary_matrix_execution_authorized=true")
    print("actual_scientific_training_started=false")
    print("actual_F7_primary_matrix_execution_started=false")
    print("F8_multi_seed_authorized=false")
    print("feature_removal_authorized=false")
    print("compact_interface_equivalence_authorized=false")
    print("hardware_reduction_claim_authorized=false")
    print("sealed_test_tensors_loaded=false")
    print(
        "next_stage="
        "V5_P3_F7_PRIMARY_SEED107_SAME_WIDTH_RETRAINED_"
        "GROUP_ABLATION_MATRIX_EXECUTION"
    )
    print(f"static_certification={static_path}")
    print(f"runtime_certification={runtime_path}")
    print(f"execution_decision={decision_path}")
    print(f"cardinality_patch={cardinality_patch_path}")
    print(f"item_batch_patch={item_batch_patch_path}")
    print(f"launcher_patch={launcher_patch_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    print(f"canonical_E2_execution_lock={canonical_lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
