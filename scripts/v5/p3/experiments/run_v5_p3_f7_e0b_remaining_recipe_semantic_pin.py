from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


STAGE = "V5_P3_F7_E0B_REMAINING_RECIPE_SEMANTIC_PIN"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "TRAIN-DERIVED"

SKELETON_RELATIVE_PATH = Path(
    "scripts/v5/p2/train_v5_p2_b2_single_seed.py"
)

EXPECTED_MODEL_CLASS = "V6P0Dynamic70GraphConvCount4"
EXPECTED_PARAMETER_COUNT = 60553
EXPECTED_PRIMARY_SEED = 107
EXPECTED_TRAIN_ITEMS = 110855
EXPECTED_VALIDATION_ITEMS = 13863

OPTIMIZER_CLASSES = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
    "Adagrad": torch.optim.Adagrad,
}

SCHEDULER_CLASSES = {
    name: getattr(torch.optim.lr_scheduler, name)
    for name in (
        "StepLR",
        "MultiStepLR",
        "ExponentialLR",
        "CosineAnnealingLR",
        "CosineAnnealingWarmRestarts",
        "ReduceLROnPlateau",
        "OneCycleLR",
        "LambdaLR",
        "LinearLR",
        "ConstantLR",
        "SequentialLR",
        "CyclicLR",
    )
    if hasattr(torch.optim.lr_scheduler, name)
}

LOSS_PRIMITIVES = {
    "binary_cross_entropy_with_logits",
    "cross_entropy",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "mse_loss",
    "l1_loss",
    "smooth_l1_loss",
}

EARLY_STOP_TOKENS = (
    "patience",
    "early_stop",
    "early_stopping",
    "bad_epoch",
    "no_improve",
    "epochs_without_improvement",
    "stale_epoch",
)

CLIP_CALLS = {
    "clip_grad_norm_",
    "clip_grad_value_",
}

CHECKPOINT_CALLS = {
    "save",
    "save_checkpoint",
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_report_lock(
    report_path: Path,
    lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = load_json(report_path)
    lock = load_json(lock_path)
    require(report.get("status") == "PASS", f"report not PASS: {report_path}")
    require(
        lock.get("report_sha256") == sha256_file(report_path),
        f"report/lock mismatch: {report_path}",
    )
    return report, lock


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def source_record(path: Path, lines: list[str], node: ast.AST) -> dict[str, Any]:
    start = int(getattr(node, "lineno", 0))
    end = int(getattr(node, "end_lineno", start))
    excerpt = "\n".join(lines[max(0, start - 1):end])
    return {
        "path": str(path.resolve()),
        "line_start": start,
        "line_end": end,
        "source": ast.unparse(node),
        "excerpt": excerpt,
        "excerpt_sha256": sha256_text(excerpt),
    }


def literal_or_source(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return {
            "literal": False,
            "source": ast.unparse(node),
        }


class StaticResolver:
    def __init__(self, tree: ast.AST, path: Path) -> None:
        self.tree = tree
        self.path = path
        self.lines = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()

        self.assignments: dict[str, list[ast.AST]] = {}
        self.assignment_nodes: dict[str, list[ast.AST]] = {}
        self.function_defaults: dict[str, dict[str, ast.AST]] = {}
        self.argparse_defaults: dict[str, ast.AST] = {}
        self.argparse_records: dict[str, dict[str, Any]] = {}
        self.parent: dict[ast.AST, ast.AST] = {}

        self._build_parent_map()
        self._collect_function_defaults()
        self._collect_assignments()
        self._collect_argparse_defaults()

    def _build_parent_map(self) -> None:
        for parent in ast.walk(self.tree):
            for child in ast.iter_child_nodes(parent):
                self.parent[child] = parent

    def _collect_function_defaults(self) -> None:
        for node in ast.walk(self.tree):
            if not isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue

            positional = list(node.args.args)
            defaults = list(node.args.defaults)
            offset = len(positional) - len(defaults)
            mapping = {}

            for index, default in enumerate(defaults):
                mapping[positional[offset + index].arg] = default

            for argument, default in zip(
                node.args.kwonlyargs,
                node.args.kw_defaults,
            ):
                if default is not None:
                    mapping[argument.arg] = default

            self.function_defaults[node.name] = mapping

    def _collect_assignments(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    self._register_target(target, node.value, node)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                self._register_target(node.target, node.value, node)

    def _register_target(
        self,
        target: ast.AST,
        value: ast.AST,
        assignment_node: ast.AST,
    ) -> None:
        if isinstance(target, ast.Name):
            key = target.id
            self.assignments.setdefault(key, []).append(value)
            self.assignment_nodes.setdefault(key, []).append(assignment_node)
        elif isinstance(target, ast.Attribute):
            key = ast.unparse(target)
            self.assignments.setdefault(key, []).append(value)
            self.assignment_nodes.setdefault(key, []).append(assignment_node)
        elif isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(value, (ast.Tuple, ast.List)):
                for child_target, child_value in zip(
                    target.elts,
                    value.elts,
                ):
                    self._register_target(
                        child_target,
                        child_value,
                        assignment_node,
                    )

    def _collect_argparse_defaults(self) -> None:
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            if call_name(node.func).rsplit(".", 1)[-1] != "add_argument":
                continue

            option_strings = [
                argument.value
                for argument in node.args
                if isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
            ]
            if not option_strings:
                continue

            keyword_map = {
                keyword.arg: keyword.value
                for keyword in node.keywords
                if keyword.arg is not None
            }
            default = keyword_map.get("default")
            if default is None:
                continue

            dest_node = keyword_map.get("dest")
            if (
                isinstance(dest_node, ast.Constant)
                and isinstance(dest_node.value, str)
            ):
                dest = dest_node.value
            else:
                long_options = [
                    option
                    for option in option_strings
                    if option.startswith("--")
                ]
                selected = long_options[-1] if long_options else option_strings[-1]
                dest = selected.lstrip("-").replace("-", "_")

            self.argparse_defaults[dest] = default
            self.argparse_records[dest] = {
                "options": option_strings,
                **source_record(self.path, self.lines, node),
            }

    def enclosing_function(self, node: ast.AST) -> str | None:
        current = node
        while current in self.parent:
            current = self.parent[current]
            if isinstance(
                current,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                return current.name
        return None

    def ancestors(self, node: ast.AST) -> list[ast.AST]:
        output = []
        current = node
        while current in self.parent:
            current = self.parent[current]
            output.append(current)
        return output

    def resolve(
        self,
        node: ast.AST,
        *,
        function_name: str | None = None,
        seen: set[str] | None = None,
    ) -> dict[str, Any]:
        if seen is None:
            seen = set()

        expression = ast.unparse(node)
        identity = repr((function_name or "<module>", expression))
        if identity in seen:
            return {
                "resolved": False,
                "expression": expression,
                "reason": "cyclic_symbol_resolution",
            }
        seen = set(seen)
        seen.add(identity)

        if isinstance(node, ast.Constant):
            return {
                "resolved": True,
                "value": node.value,
                "expression": expression,
                "route": "literal",
            }

        if isinstance(node, ast.UnaryOp):
            child = self.resolve(
                node.operand,
                function_name=function_name,
                seen=seen,
            )
            if not child["resolved"]:
                return child
            value = child["value"]
            if isinstance(node.op, ast.USub):
                value = -value
            elif isinstance(node.op, ast.UAdd):
                value = +value
            elif isinstance(node.op, ast.Not):
                value = not value
            else:
                return {
                    "resolved": False,
                    "expression": expression,
                    "reason": "unsupported_unary_operator",
                }
            return {
                "resolved": True,
                "value": value,
                "expression": expression,
                "route": "unary_expression",
                "child": child,
            }

        if isinstance(node, ast.BinOp):
            left = self.resolve(
                node.left,
                function_name=function_name,
                seen=seen,
            )
            right = self.resolve(
                node.right,
                function_name=function_name,
                seen=seen,
            )
            if not left["resolved"] or not right["resolved"]:
                return {
                    "resolved": False,
                    "expression": expression,
                    "reason": "unresolved_binary_operand",
                    "left": left,
                    "right": right,
                }
            try:
                if isinstance(node.op, ast.Add):
                    value = left["value"] + right["value"]
                elif isinstance(node.op, ast.Sub):
                    value = left["value"] - right["value"]
                elif isinstance(node.op, ast.Mult):
                    value = left["value"] * right["value"]
                elif isinstance(node.op, ast.Div):
                    value = left["value"] / right["value"]
                elif isinstance(node.op, ast.FloorDiv):
                    value = left["value"] // right["value"]
                elif isinstance(node.op, ast.Pow):
                    value = left["value"] ** right["value"]
                elif isinstance(node.op, ast.Mod):
                    value = left["value"] % right["value"]
                else:
                    raise ValueError("unsupported binary operator")
            except Exception as exc:
                return {
                    "resolved": False,
                    "expression": expression,
                    "reason": f"binary_evaluation_failed:{exc!r}",
                }
            return {
                "resolved": True,
                "value": value,
                "expression": expression,
                "route": "binary_expression",
                "left": left,
                "right": right,
            }

        if isinstance(node, ast.Name):
            name = node.id

            if function_name is not None:
                defaults = self.function_defaults.get(function_name, {})
                if name in defaults:
                    child = self.resolve(
                        defaults[name],
                        function_name=function_name,
                        seen=seen,
                    )
                    return {
                        **child,
                        "expression": expression,
                        "route": "function_parameter_default",
                        "parameter": name,
                        "function": function_name,
                        "child": child,
                    }

            values = self.assignments.get(name, [])
            literal_candidates = []
            for value_node in values:
                child = self.resolve(
                    value_node,
                    function_name=function_name,
                    seen=seen,
                )
                if child.get("resolved"):
                    literal_candidates.append(child)

            unique_values = []
            for child in literal_candidates:
                value = child["value"]
                if value not in unique_values:
                    unique_values.append(value)

            if len(unique_values) == 1:
                return {
                    "resolved": True,
                    "value": unique_values[0],
                    "expression": expression,
                    "route": "unique_resolved_assignment_value",
                    "symbol": name,
                    "candidate_count": len(values),
                    "children": literal_candidates,
                }

            return {
                "resolved": False,
                "expression": expression,
                "reason": (
                    "symbol_not_found"
                    if not values
                    else "symbol_assignments_not_unique"
                ),
                "symbol": name,
                "assignment_count": len(values),
                "resolved_candidate_values": unique_values,
            }

        if isinstance(node, ast.Attribute):
            full = ast.unparse(node)

            if (
                isinstance(node.value, ast.Name)
                and node.value.id in {"args", "config", "cfg", "options"}
            ):
                dest = node.attr
                if dest in self.argparse_defaults:
                    child = self.resolve(
                        self.argparse_defaults[dest],
                        function_name=function_name,
                        seen=seen,
                    )
                    return {
                        **child,
                        "expression": expression,
                        "route": "argparse_default",
                        "dest": dest,
                        "argparse_record": self.argparse_records[dest],
                        "child": child,
                    }

            values = self.assignments.get(full, [])
            if len(values) == 1:
                child = self.resolve(
                    values[0],
                    function_name=function_name,
                    seen=seen,
                )
                return {
                    **child,
                    "expression": expression,
                    "route": "unique_attribute_assignment",
                    "symbol": full,
                    "child": child,
                }

            return {
                "resolved": False,
                "expression": expression,
                "reason": "attribute_not_resolved",
                "symbol": full,
            }

        if isinstance(node, (ast.List, ast.Tuple, ast.Dict)):
            try:
                value = ast.literal_eval(node)
                return {
                    "resolved": True,
                    "value": value,
                    "expression": expression,
                    "route": "literal_container",
                }
            except Exception:
                return {
                    "resolved": False,
                    "expression": expression,
                    "reason": "container_not_literal",
                }

        if isinstance(node, ast.Subscript):
            container = self.resolve(
                node.value,
                function_name=function_name,
                seen=seen,
            )
            index = self.resolve(
                node.slice,
                function_name=function_name,
                seen=seen,
            )
            if not container["resolved"] or not index["resolved"]:
                return {
                    "resolved": False,
                    "expression": expression,
                    "reason": "unresolved_subscript",
                    "container": container,
                    "index": index,
                }
            try:
                value = container["value"][index["value"]]
            except Exception as exc:
                return {
                    "resolved": False,
                    "expression": expression,
                    "reason": f"subscript_failed:{exc!r}",
                }
            return {
                "resolved": True,
                "value": value,
                "expression": expression,
                "route": "resolved_subscript",
                "container": container,
                "index": index,
            }

        if isinstance(node, ast.Call):
            short = call_name(node.func).rsplit(".", 1)[-1]
            if short in {"int", "float", "str", "bool"} and len(node.args) == 1:
                child = self.resolve(
                    node.args[0],
                    function_name=function_name,
                    seen=seen,
                )
                if not child["resolved"]:
                    return child
                caster = {
                    "int": int,
                    "float": float,
                    "str": str,
                    "bool": bool,
                }[short]
                return {
                    "resolved": True,
                    "value": caster(child["value"]),
                    "expression": expression,
                    "route": f"{short}_cast",
                    "child": child,
                }

            if short in {"max", "min"} and node.args:
                children = [
                    self.resolve(
                        argument,
                        function_name=function_name,
                        seen=seen,
                    )
                    for argument in node.args
                ]
                if all(child["resolved"] for child in children):
                    function = max if short == "max" else min
                    return {
                        "resolved": True,
                        "value": function(
                            child["value"] for child in children
                        ),
                        "expression": expression,
                        "route": f"{short}_call",
                        "children": children,
                    }

            return {
                "resolved": False,
                "expression": expression,
                "reason": "unsupported_call_expression",
                "call_name": call_name(node.func),
            }

        return {
            "resolved": False,
            "expression": expression,
            "reason": f"unsupported_AST_node:{type(node).__name__}",
        }


def target_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    return None


def assigned_target(parent: ast.AST | None) -> str | None:
    if isinstance(parent, ast.Assign) and len(parent.targets) == 1:
        return target_name(parent.targets[0])
    if isinstance(parent, ast.AnnAssign):
        return target_name(parent.target)
    return None


def keyword_node(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def resolve_argument(
    resolver: StaticResolver,
    call: ast.Call,
    *,
    keyword: str,
    positional_index: int | None,
    runtime_callable: Any | None,
) -> dict[str, Any]:
    function_name = resolver.enclosing_function(call)
    node = keyword_node(call, keyword)

    if node is not None:
        result = resolver.resolve(
            node,
            function_name=function_name,
        )
        return {
            **result,
            "argument_route": "keyword",
            "keyword": keyword,
        }

    if positional_index is not None and len(call.args) > positional_index:
        result = resolver.resolve(
            call.args[positional_index],
            function_name=function_name,
        )
        return {
            **result,
            "argument_route": "positional",
            "position": positional_index,
            "keyword": keyword,
        }

    if runtime_callable is not None:
        signature = inspect.signature(runtime_callable)
        parameter = signature.parameters[keyword]
        require(
            parameter.default is not inspect._empty,
            f"runtime default missing: {runtime_callable}.{keyword}",
        )
        return {
            "resolved": True,
            "value": parameter.default,
            "expression": None,
            "route": "runtime_signature_default",
            "argument_route": "runtime_default",
            "keyword": keyword,
            "signature": str(signature),
            "callable": (
                f"{runtime_callable.__module__}."
                f"{runtime_callable.__qualname__}"
            ),
            "torch_version": torch.__version__,
        }

    return {
        "resolved": False,
        "argument_route": "absent",
        "keyword": keyword,
        "reason": "argument_absent_and_no_runtime_default",
    }


def compare_operator(node: ast.cmpop) -> str:
    mapping = {
        ast.Gt: ">",
        ast.GtE: ">=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Eq: "==",
        ast.NotEq: "!=",
    }
    return mapping.get(type(node), type(node).__name__)


def contains_break(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Break) for child in ast.walk(node))


def contains_checkpoint_call(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if call_name(child.func).rsplit(".", 1)[-1] in CHECKPOINT_CALLS:
            return True
    return False


def nearest_loop_semantics(
    resolver: StaticResolver,
    node: ast.AST,
) -> list[dict[str, Any]]:
    loops = []
    for ancestor in resolver.ancestors(node):
        if isinstance(ancestor, ast.For):
            loops.append({
                "type": "for",
                "target": ast.unparse(ancestor.target),
                "iterator": ast.unparse(ancestor.iter),
                "line_start": int(ancestor.lineno),
            })
        elif isinstance(ancestor, ast.While):
            loops.append({
                "type": "while",
                "condition": ast.unparse(ancestor.test),
                "line_start": int(ancestor.lineno),
            })
    return loops


def range_semantics(
    resolver: StaticResolver,
    loop: ast.For,
) -> dict[str, Any]:
    if not isinstance(loop.iter, ast.Call):
        return {
            "resolved": False,
            "reason": "epoch_iterator_is_not_a_call",
            "iterator": ast.unparse(loop.iter),
        }

    if call_name(loop.iter.func).rsplit(".", 1)[-1] != "range":
        return {
            "resolved": False,
            "reason": "epoch_iterator_is_not_range",
            "iterator": ast.unparse(loop.iter),
        }

    function_name = resolver.enclosing_function(loop)
    arguments = [
        resolver.resolve(
            argument,
            function_name=function_name,
        )
        for argument in loop.iter.args
    ]
    if not all(result.get("resolved") for result in arguments):
        return {
            "resolved": False,
            "reason": "range_argument_unresolved",
            "arguments": arguments,
            "iterator": ast.unparse(loop.iter),
        }

    values = [result["value"] for result in arguments]
    if len(values) == 1:
        start, stop, step = 0, int(values[0]), 1
    elif len(values) == 2:
        start, stop = int(values[0]), int(values[1])
        step = 1
    elif len(values) == 3:
        start, stop, step = map(int, values)
    else:
        return {
            "resolved": False,
            "reason": "range_argument_count_invalid",
            "values": values,
        }

    require(step != 0, "epoch range step is zero")
    epochs = list(range(start, stop, step))

    return {
        "resolved": True,
        "start": start,
        "stop_exclusive": stop,
        "step": step,
        "executed_epoch_count": len(epochs),
        "first_epoch_index": epochs[0] if epochs else None,
        "last_epoch_index": epochs[-1] if epochs else None,
        "indices_sha256": sha256_text(
            json.dumps(epochs, separators=(",", ":"))
        ),
        "arguments": arguments,
        "iterator": ast.unparse(loop.iter),
    }


def early_stop_policy(
    resolver: StaticResolver,
    tree: ast.AST,
    path: Path,
    lines: list[str],
) -> dict[str, Any]:
    candidates = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not contains_break(node):
            continue

        text = ast.unparse(node)
        token = normalize(text)
        if not any(term in token for term in EARLY_STOP_TOKENS):
            continue

        comparisons = [
            child
            for child in ast.walk(node.test)
            if isinstance(child, ast.Compare)
            and len(child.ops) == 1
            and len(child.comparators) == 1
        ]

        comparison_rows = []
        for comparison in comparisons:
            left = comparison.left
            right = comparison.comparators[0]
            left_result = resolver.resolve(
                left,
                function_name=resolver.enclosing_function(node),
            )
            right_result = resolver.resolve(
                right,
                function_name=resolver.enclosing_function(node),
            )

            left_name = ast.unparse(left)
            right_name = ast.unparse(right)

            comparison_rows.append({
                "left_expression": left_name,
                "left_resolution": left_result,
                "operator": compare_operator(comparison.ops[0]),
                "right_expression": right_name,
                "right_resolution": right_result,
            })

        candidates.append({
            "record": source_record(path, lines, node),
            "comparison_rows": comparison_rows,
            "loop_ancestors": nearest_loop_semantics(resolver, node),
        })

    if not candidates:
        token_evidence = []
        for node in ast.walk(tree):
            text = ast.unparse(node) if isinstance(
                node,
                (
                    ast.Assign,
                    ast.AnnAssign,
                    ast.AugAssign,
                    ast.If,
                    ast.Call,
                ),
            ) else ""
            token = normalize(text)
            if text and any(term in token for term in EARLY_STOP_TOKENS):
                token_evidence.append(source_record(path, lines, node))

        return {
            "active": False,
            "classification": "NO_ACTIVE_BREAK_BASED_EARLY_STOPPING",
            "stop_candidates": [],
            "token_evidence": token_evidence,
            "semantics_resolved": True,
        }

    if len(candidates) != 1:
        return {
            "active": True,
            "classification": "MULTIPLE_EARLY_STOP_CONDITIONS",
            "stop_candidates": candidates,
            "semantics_resolved": False,
        }

    candidate = candidates[0]
    numeric_thresholds = []
    counter_names = []
    operators = []

    for row in candidate["comparison_rows"]:
        left = row["left_resolution"]
        right = row["right_resolution"]
        operator = row["operator"]

        if (
            right.get("resolved")
            and isinstance(right.get("value"), (int, float))
            and not isinstance(right.get("value"), bool)
        ):
            numeric_thresholds.append(float(right["value"]))
            counter_names.append(row["left_expression"])
            operators.append(operator)
        elif (
            left.get("resolved")
            and isinstance(left.get("value"), (int, float))
            and not isinstance(left.get("value"), bool)
        ):
            numeric_thresholds.append(float(left["value"]))
            counter_names.append(row["right_expression"])
            inverse = {
                ">": "<",
                ">=": "<=",
                "<": ">",
                "<=": ">=",
                "==": "==",
                "!=": "!=",
            }[operator]
            operators.append(inverse)

    unique_thresholds = []
    for value in numeric_thresholds:
        if value not in unique_thresholds:
            unique_thresholds.append(value)

    unique_counters = []
    for name in counter_names:
        if name not in unique_counters:
            unique_counters.append(name)

    unique_operators = []
    for operator in operators:
        if operator not in unique_operators:
            unique_operators.append(operator)

    if (
        len(unique_thresholds) != 1
        or len(unique_counters) != 1
        or len(unique_operators) != 1
    ):
        return {
            "active": True,
            "classification": "EARLY_STOP_THRESHOLD_OR_COUNTER_UNRESOLVED",
            "stop_candidates": candidates,
            "numeric_threshold_candidates": unique_thresholds,
            "counter_candidates": unique_counters,
            "operator_candidates": unique_operators,
            "semantics_resolved": False,
        }

    threshold = unique_thresholds[0]
    counter = unique_counters[0]
    operator = unique_operators[0]

    counter_events = []
    improvement_branches = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [ast.unparse(target) for target in node.targets]
            if counter in targets:
                result = resolver.resolve(
                    node.value,
                    function_name=resolver.enclosing_function(node),
                )
                counter_events.append({
                    "event": "assign",
                    "value_resolution": result,
                    **source_record(path, lines, node),
                })

        elif isinstance(node, ast.AnnAssign):
            if ast.unparse(node.target) == counter and node.value is not None:
                result = resolver.resolve(
                    node.value,
                    function_name=resolver.enclosing_function(node),
                )
                counter_events.append({
                    "event": "assign",
                    "value_resolution": result,
                    **source_record(path, lines, node),
                })

        elif isinstance(node, ast.AugAssign):
            if ast.unparse(node.target) == counter:
                result = resolver.resolve(
                    node.value,
                    function_name=resolver.enclosing_function(node),
                )
                counter_events.append({
                    "event": "increment"
                    if isinstance(node.op, ast.Add)
                    else "augmented_assignment",
                    "operator": type(node.op).__name__,
                    "value_resolution": result,
                    **source_record(path, lines, node),
                })

        if isinstance(node, ast.If) and node is not None:
            body_source = "\n".join(ast.unparse(child) for child in node.body)
            else_source = "\n".join(ast.unparse(child) for child in node.orelse)
            body_token = normalize(body_source)
            else_token = normalize(else_source)

            counter_reset_in_body = (
                normalize(f"{counter} = 0") in body_token
                or normalize(f"{counter}=0") in body_token
            )
            counter_increment_in_else = (
                normalize(f"{counter} += 1") in else_token
                or normalize(f"{counter}={counter}+1") in else_token
            )

            if counter_reset_in_body or counter_increment_in_else:
                improvement_branches.append({
                    "test": ast.unparse(node.test),
                    "counter_reset_in_body": counter_reset_in_body,
                    "counter_increment_in_else": counter_increment_in_else,
                    "checkpoint_call_in_body": any(
                        contains_checkpoint_call(child)
                        for child in node.body
                    ),
                    **source_record(path, lines, node),
                })

    increment_values = []
    reset_values = []

    for event in counter_events:
        result = event.get("value_resolution", {})
        if not result.get("resolved"):
            continue
        value = result.get("value")
        if event["event"] == "increment":
            increment_values.append(value)
        elif event["event"] == "assign":
            reset_values.append(value)

    unique_increment_values = []
    for value in increment_values:
        if value not in unique_increment_values:
            unique_increment_values.append(value)

    numeric_reset_values = [
        value
        for value in reset_values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]

    stop_line = candidate["record"]["line_start"]
    increment_lines = [
        event["line_start"]
        for event in counter_events
        if event["event"] == "increment"
    ]
    increment_before_stop = any(
        line < stop_line
        for line in increment_lines
    )

    effective_patience = None
    if (
        threshold >= 0
        and unique_increment_values == [1]
        and 0 in numeric_reset_values
        and increment_before_stop
    ):
        if operator == ">=":
            effective_patience = int(threshold)
        elif operator == ">":
            effective_patience = int(threshold) + 1

    semantics_resolved = bool(
        threshold >= 0
        and operator in {">=", ">"}
        and unique_increment_values == [1]
        and 0 in numeric_reset_values
        and increment_before_stop
        and effective_patience is not None
        and len(improvement_branches) >= 1
    )

    return {
        "active": True,
        "classification": (
            "ACTIVE_CONSECUTIVE_NON_IMPROVEMENT_EARLY_STOPPING"
            if semantics_resolved
            else "EARLY_STOPPING_PRESENT_BUT_SEMANTICS_INCOMPLETE"
        ),
        "counter": counter,
        "threshold": threshold,
        "operator": operator,
        "increment_value": (
            unique_increment_values[0]
            if len(unique_increment_values) == 1
            else None
        ),
        "reset_values": sorted(set(numeric_reset_values)),
        "increment_before_stop_check": increment_before_stop,
        "effective_consecutive_non_improving_epochs": effective_patience,
        "stop_condition": candidate,
        "counter_events": counter_events,
        "improvement_branches": improvement_branches,
        "semantics_resolved": semantics_resolved,
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"repository missing: {repo}")
    require(data_link.is_symlink(), f"dataset symlink missing: {data_link}")
    require(data_link.resolve().is_dir(), "dataset target missing")

    feature_root = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study"
    )
    f7_root = feature_root / "retrained_group_ablation"

    e0a_report_path = f7_root / (
        "V5_P3_F7_E0A_EXACT_HYPERPARAMETER_AND_LOSS_ROUTE_PIN_REPORT.json"
    )
    e0a_lock_path = f7_root / (
        "V5_P3_F7_E0A_EXACT_HYPERPARAMETER_AND_LOSS_ROUTE_PIN_LOCK.json"
    )
    e0a_pin_path = f7_root / (
        "F7_E0A_EXACT_HYPERPARAMETER_PIN.json"
    )
    p2_report_path = f7_root / (
        "V5_P3_F7_P2_RECONSTRUCTED_TRAINER_ADAPTER_AND_DRY_RUN_PREFLIGHT_REPORT.json"
    )
    p2_lock_path = f7_root / (
        "V5_P3_F7_P2_RECONSTRUCTED_TRAINER_ADAPTER_AND_DRY_RUN_PREFLIGHT_LOCK.json"
    )
    protocol_path = f7_root / (
        "F7_P0_FROZEN_SAME_WIDTH_ABLATION_PROTOCOL.json"
    )
    reconstructed_route_path = f7_root / (
        "F7_P1B_FROZEN_RECONSTRUCTED_TRAINING_ROUTE.json"
    )

    e0a_report, e0a_lock = verify_report_lock(
        e0a_report_path,
        e0a_lock_path,
    )
    p2_report, p2_lock = verify_report_lock(
        p2_report_path,
        p2_lock_path,
    )
    e0a_pin = load_json(e0a_pin_path)
    protocol = load_json(protocol_path)
    reconstructed_route = load_json(reconstructed_route_path)

    require(
        e0a_lock.get("E0A_complete") is True,
        "E0A incomplete",
    )
    require(
        e0a_lock.get("E1_trainer_generation_authorized") is False,
        "E1 already authorized before E0B",
    )
    require(
        p2_lock.get("actual_F7_retraining_authorized") is True,
        "P2 did not authorize F7 execution",
    )
    require(
        p2_lock.get("feature_removal_authorized") is False,
        "feature removal already authorized",
    )
    require(
        p2_lock.get("sealed_test_tensors_loaded") is False,
        "sealed-test access detected",
    )
    require(
        reconstructed_route["status"] == "FROZEN",
        "reconstructed route not frozen",
    )
    require(
        reconstructed_route["model"]["class"] == EXPECTED_MODEL_CLASS,
        "model class changed",
    )
    require(
        int(reconstructed_route["model"]["parameter_count"])
        == EXPECTED_PARAMETER_COUNT,
        "parameter count changed",
    )
    require(
        int(reconstructed_route["dataset"]["train_items"])
        == EXPECTED_TRAIN_ITEMS,
        "train count changed",
    )
    require(
        int(reconstructed_route["dataset"]["validation_items"])
        == EXPECTED_VALIDATION_ITEMS,
        "validation count changed",
    )

    skeleton_path = (repo / SKELETON_RELATIVE_PATH).resolve()
    require(skeleton_path.is_file(), f"skeleton missing: {skeleton_path}")
    require(
        sha256_file(skeleton_path)
        == e0a_report["finding"]["skeleton_sha256"],
        "skeleton hash differs from E0A",
    )

    text = skeleton_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    lines = text.splitlines()
    tree = ast.parse(text)
    resolver = StaticResolver(tree, skeleton_path)

    optimizer_constructors = []
    scheduler_constructors = []
    optimizer_steps = []
    scheduler_steps = []
    unclassified_steps = []
    clip_calls = []
    backward_calls = []
    compute_loss_functions = []
    compute_loss_calls = []
    primitive_loss_calls = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "compute_loss":
                record = source_record(skeleton_path, lines, node)
                record["function_name"] = node.name
                compute_loss_functions.append(record)

        if not isinstance(node, ast.Call):
            continue

        full_name = call_name(node.func)
        short = full_name.rsplit(".", 1)[-1]
        parent = resolver.parent.get(node)
        assigned = assigned_target(parent)

        if short in OPTIMIZER_CLASSES:
            optimizer_class = OPTIMIZER_CLASSES[short]
            optimizer_constructors.append({
                "variable": assigned,
                "implementation": (
                    f"{optimizer_class.__module__}."
                    f"{optimizer_class.__qualname__}"
                ),
                "learning_rate": resolve_argument(
                    resolver,
                    node,
                    keyword="lr",
                    positional_index=1,
                    runtime_callable=optimizer_class,
                ),
                "weight_decay": resolve_argument(
                    resolver,
                    node,
                    keyword="weight_decay",
                    positional_index=None,
                    runtime_callable=optimizer_class,
                ),
                **source_record(skeleton_path, lines, node),
            })

        if short in SCHEDULER_CLASSES:
            scheduler_class = SCHEDULER_CLASSES[short]
            keywords = {}
            signature = inspect.signature(scheduler_class)
            for parameter_name, parameter in signature.parameters.items():
                if parameter_name == "optimizer":
                    continue
                if (
                    keyword_node(node, parameter_name) is not None
                    or parameter.default is not inspect._empty
                ):
                    keywords[parameter_name] = resolve_argument(
                        resolver,
                        node,
                        keyword=parameter_name,
                        positional_index=None,
                        runtime_callable=scheduler_class,
                    )

            scheduler_constructors.append({
                "variable": assigned,
                "implementation": (
                    f"{scheduler_class.__module__}."
                    f"{scheduler_class.__qualname__}"
                ),
                "runtime_signature": str(signature),
                "resolved_parameters": keywords,
                **source_record(skeleton_path, lines, node),
            })

        if short == "step" and isinstance(node.func, ast.Attribute):
            base = ast.unparse(node.func.value)
            record = {
                "base": base,
                "arguments": [
                    resolver.resolve(
                        argument,
                        function_name=resolver.enclosing_function(node),
                    )
                    for argument in node.args
                ],
                "argument_expressions": [
                    ast.unparse(argument)
                    for argument in node.args
                ],
                "loop_ancestors": nearest_loop_semantics(resolver, node),
                "enclosing_function": resolver.enclosing_function(node),
                **source_record(skeleton_path, lines, node),
            }
            optimizer_variables = {
                row["variable"]
                for row in optimizer_constructors
                if row["variable"] is not None
            }
            scheduler_variables = {
                row["variable"]
                for row in scheduler_constructors
                if row["variable"] is not None
            }

            if base in optimizer_variables:
                optimizer_steps.append(record)
            elif base in scheduler_variables:
                scheduler_steps.append(record)
            else:
                unclassified_steps.append(record)

        if short == "backward":
            backward_calls.append(
                source_record(skeleton_path, lines, node)
            )

        if short in CLIP_CALLS:
            threshold_keyword = (
                "max_norm"
                if short == "clip_grad_norm_"
                else "clip_value"
            )
            threshold_position = 1
            clip_calls.append({
                "implementation": full_name,
                "threshold": resolve_argument(
                    resolver,
                    node,
                    keyword=threshold_keyword,
                    positional_index=threshold_position,
                    runtime_callable=None,
                ),
                "loop_ancestors": nearest_loop_semantics(resolver, node),
                "enclosing_function": resolver.enclosing_function(node),
                **source_record(skeleton_path, lines, node),
            })

        if short == "compute_loss":
            compute_loss_calls.append({
                "arguments": [
                    ast.unparse(argument)
                    for argument in node.args
                ],
                "keywords": {
                    keyword.arg: ast.unparse(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg is not None
                },
                "enclosing_function": resolver.enclosing_function(node),
                **source_record(skeleton_path, lines, node),
            })

        if short in LOSS_PRIMITIVES:
            primitive_loss_calls.append({
                "implementation": full_name,
                "arguments": [
                    ast.unparse(argument)
                    for argument in node.args
                ],
                "keywords": {
                    keyword.arg: ast.unparse(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg is not None
                },
                "enclosing_function": resolver.enclosing_function(node),
                **source_record(skeleton_path, lines, node),
            })

    require(
        len(optimizer_constructors) == 1,
        "optimizer constructor is not unique",
    )
    optimizer_variable = optimizer_constructors[0]["variable"]
    require(
        optimizer_variable is not None,
        "optimizer constructor is not assigned to a variable",
    )

    # Reclassify step calls after all constructors are known.
    optimizer_steps = []
    scheduler_steps = []
    unclassified_steps = []
    optimizer_variables = {optimizer_variable}
    scheduler_variables = {
        row["variable"]
        for row in scheduler_constructors
        if row["variable"] is not None
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if call_name(node.func).rsplit(".", 1)[-1] != "step":
            continue
        if not isinstance(node.func, ast.Attribute):
            continue

        base = ast.unparse(node.func.value)
        record = {
            "base": base,
            "arguments": [
                resolver.resolve(
                    argument,
                    function_name=resolver.enclosing_function(node),
                )
                for argument in node.args
            ],
            "argument_expressions": [
                ast.unparse(argument)
                for argument in node.args
            ],
            "loop_ancestors": nearest_loop_semantics(resolver, node),
            "enclosing_function": resolver.enclosing_function(node),
            **source_record(skeleton_path, lines, node),
        }

        if base in optimizer_variables:
            optimizer_steps.append(record)
        elif base in scheduler_variables:
            scheduler_steps.append(record)
        else:
            unclassified_steps.append(record)

    require(
        len(optimizer_steps) == 1,
        f"expected one optimizer.step route, found {len(optimizer_steps)}",
    )

    epoch_loops = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        if "epoch" not in ast.unparse(node.target).lower():
            continue
        semantics = range_semantics(resolver, node)
        epoch_loops.append({
            "target": ast.unparse(node.target),
            "semantics": semantics,
            **source_record(skeleton_path, lines, node),
        })

    require(len(epoch_loops) == 1, "epoch loop is not unique")
    epoch_semantics = epoch_loops[0]["semantics"]
    require(
        epoch_semantics.get("resolved") is True,
        "epoch range semantics unresolved",
    )

    early_stopping = early_stop_policy(
        resolver,
        tree,
        skeleton_path,
        lines,
    )

    # Scheduler semantic pin.
    scheduler_semantics_resolved = True
    scheduler_policy = {
        "constructor_count": len(scheduler_constructors),
        "constructors": scheduler_constructors,
        "step_count": len(scheduler_steps),
        "steps": scheduler_steps,
        "policy": None,
    }

    if len(scheduler_constructors) == 0:
        scheduler_policy["policy"] = "NO_SCHEDULER"
        require(
            len(scheduler_steps) == 0,
            "scheduler.step exists without scheduler constructor",
        )
    elif len(scheduler_constructors) == 1:
        require(
            scheduler_constructors[0]["variable"] is not None,
            "scheduler variable unresolved",
        )
        if len(scheduler_steps) != 1:
            scheduler_semantics_resolved = False
            scheduler_policy["policy"] = (
                "SCHEDULER_PRESENT_STEP_ROUTE_AMBIGUOUS"
            )
        else:
            loops = scheduler_steps[0]["loop_ancestors"]
            epoch_ancestor = any(
                "epoch" in normalize(
                    row.get("target", "") + row.get("iterator", "")
                )
                for row in loops
            )
            batch_ancestor = any(
                any(
                    term in normalize(
                        row.get("target", "") + row.get("iterator", "")
                    )
                    for term in ("batch", "loader", "train_loader")
                )
                for row in loops
            )

            if epoch_ancestor and not batch_ancestor:
                cadence = "ONCE_PER_EPOCH"
            elif epoch_ancestor and batch_ancestor:
                cadence = "PER_TRAINING_BATCH"
            else:
                cadence = "OUTSIDE_RECOGNIZED_EPOCH_LOOP"

            scheduler_policy["policy"] = (
                "ACTIVE_SCHEDULER_WITH_RESOLVED_SINGLE_STEP"
            )
            scheduler_policy["step_cadence"] = cadence
            scheduler_policy["step_argument_expressions"] = (
                scheduler_steps[0]["argument_expressions"]
            )
            scheduler_policy["step_after_optimizer_step"] = (
                scheduler_steps[0]["line_start"]
                > optimizer_steps[0]["line_start"]
            )
            scheduler_semantics_resolved = cadence in {
                "ONCE_PER_EPOCH",
                "PER_TRAINING_BATCH",
            }
    else:
        scheduler_policy["policy"] = (
            "MULTIPLE_SCHEDULER_CONSTRUCTORS"
        )
        scheduler_semantics_resolved = False

    # Gradient clipping semantic pin.
    clip_semantics_resolved = True
    clip_policy = {
        "clip_call_count": len(clip_calls),
        "calls": clip_calls,
        "policy": None,
    }

    if len(clip_calls) == 0:
        clip_policy["policy"] = "NO_GRADIENT_CLIPPING"
    elif len(clip_calls) == 1:
        threshold = clip_calls[0]["threshold"]
        clip_policy["policy"] = "ACTIVE_SINGLE_GRADIENT_CLIP"
        clip_policy["threshold_resolved"] = threshold.get("resolved")
        clip_policy["after_backward"] = (
            bool(backward_calls)
            and clip_calls[0]["line_start"]
            > backward_calls[0]["line_start"]
        )
        clip_policy["before_optimizer_step"] = (
            clip_calls[0]["line_start"]
            < optimizer_steps[0]["line_start"]
        )
        clip_semantics_resolved = bool(
            threshold.get("resolved")
            and isinstance(threshold.get("value"), (int, float))
            and not isinstance(threshold.get("value"), bool)
            and float(threshold["value"]) > 0
            and clip_policy["after_backward"]
            and clip_policy["before_optimizer_step"]
        )
    else:
        clip_policy["policy"] = "MULTIPLE_GRADIENT_CLIP_CALLS"
        clip_semantics_resolved = False

    # Loss route semantic pin.
    loss_semantics_resolved = bool(
        len(compute_loss_functions) == 1
        and len(compute_loss_calls) >= 1
        and len(primitive_loss_calls) >= 1
    )
    loss_policy = {
        "compute_loss_function_count": len(compute_loss_functions),
        "compute_loss_functions": compute_loss_functions,
        "compute_loss_call_count": len(compute_loss_calls),
        "compute_loss_calls": compute_loss_calls,
        "primitive_loss_call_count": len(primitive_loss_calls),
        "primitive_loss_calls": primitive_loss_calls,
        "source_copy_policy": (
            "E1 must copy the exact compute_loss function excerpt and verify "
            "its SHA-256 rather than reconstructing loss weights manually."
            if len(compute_loss_functions) == 1
            else None
        ),
        "semantics_resolved": loss_semantics_resolved,
    }

    # Optimizer step cadence.
    optimizer_step_loops = optimizer_steps[0]["loop_ancestors"]
    optimizer_step_inside_batch_loop = any(
        any(
            term in normalize(
                row.get("target", "") + row.get("iterator", "")
            )
            for term in ("batch", "loader", "train_loader")
        )
        for row in optimizer_step_loops
    )
    optimizer_step_inside_epoch_loop = any(
        "epoch" in normalize(
            row.get("target", "") + row.get("iterator", "")
        )
        for row in optimizer_step_loops
    )
    optimizer_semantics_resolved = bool(
        optimizer_step_inside_batch_loop
        and optimizer_step_inside_epoch_loop
    )

    optimizer_policy = {
        "constructor": optimizer_constructors[0],
        "variable": optimizer_variable,
        "optimizer_step": optimizer_steps[0],
        "optimizer_step_inside_batch_loop": (
            optimizer_step_inside_batch_loop
        ),
        "optimizer_step_inside_epoch_loop": (
            optimizer_step_inside_epoch_loop
        ),
        "unclassified_step_calls": unclassified_steps,
        "semantics_resolved": optimizer_semantics_resolved,
    }

    unresolved = []

    if not early_stopping["semantics_resolved"]:
        unresolved.append({
            "field": "early_stopping_policy",
            "reason": early_stopping["classification"],
        })
    if not scheduler_semantics_resolved:
        unresolved.append({
            "field": "scheduler_policy",
            "reason": scheduler_policy["policy"],
        })
    if not clip_semantics_resolved:
        unresolved.append({
            "field": "gradient_clipping_policy",
            "reason": clip_policy["policy"],
        })
    if not loss_semantics_resolved:
        unresolved.append({
            "field": "loss_route",
            "reason": (
                "compute_loss function/call or primitive-loss mapping is "
                "not unique"
            ),
        })
    if not optimizer_semantics_resolved:
        unresolved.append({
            "field": "optimizer_step_cadence",
            "reason": (
                "optimizer.step was not uniquely placed inside both the "
                "training-batch and epoch loops"
            ),
        })

    # Correct E0A's range-stop wording.
    e0a_range_stop = e0a_pin.get("max_epochs")
    executed_epoch_budget = epoch_semantics[
        "executed_epoch_count"
    ]
    range_stop_exclusive = epoch_semantics["stop_exclusive"]
    epoch_budget_correction_required = bool(
        e0a_range_stop == range_stop_exclusive
        and executed_epoch_budget != range_stop_exclusive
    )

    exact_recipe = {
        "status": "FROZEN" if not unresolved else "PARTIALLY_RESOLVED",
        "route_type": (
            "reconstructed_F7_recipe_from_exact_skeleton_semantics"
        ),
        "not_claimed": (
            "byte-identical historical A4 trainer reproduction"
        ),
        "primary_seed": EXPECTED_PRIMARY_SEED,
        "train_batch_size": e0a_pin["train_batch_size"],
        "validation_batch_size": e0a_pin[
            "validation_batch_size"
        ],
        "epoch_loop": epoch_semantics,
        "executed_epoch_budget": executed_epoch_budget,
        "range_stop_exclusive": range_stop_exclusive,
        "E0A_max_epochs_field": e0a_range_stop,
        "E0A_epoch_budget_correction_required": (
            epoch_budget_correction_required
        ),
        "optimizer": optimizer_policy,
        "early_stopping": early_stopping,
        "scheduler": scheduler_policy,
        "gradient_clipping": clip_policy,
        "loss": loss_policy,
        "selection_formula": protocol["metric_protocol"][
            "primary_selection_formula"
        ],
        "threshold_policy": protocol["metric_protocol"][
            "threshold_policy"
        ],
        "candidate_reduction_gates": protocol["metric_protocol"][
            "candidate_reduction_gates"
        ],
        "unresolved_fields": unresolved,
    }

    early_path = output_dir / (
        "F7_E0B_EARLY_STOPPING_SEMANTIC_PIN.json"
    )
    scheduler_path = output_dir / (
        "F7_E0B_SCHEDULER_AND_OPTIMIZER_STEP_SEMANTIC_PIN.json"
    )
    clip_path = output_dir / (
        "F7_E0B_GRADIENT_CLIPPING_SEMANTIC_PIN.json"
    )
    loss_path = output_dir / (
        "F7_E0B_EXACT_LOSS_SOURCE_PIN.json"
    )
    epoch_path = output_dir / (
        "F7_E0B_EXACT_EPOCH_RANGE_SEMANTICS.json"
    )
    recipe_path = output_dir / (
        "F7_E0B_FINAL_RECONSTRUCTED_EXECUTION_RECIPE.json"
    )
    decision_path = output_dir / (
        "F7_E0B_TRAINER_GENERATION_DECISION.json"
    )

    atomic_json(early_path, early_stopping)
    atomic_json(
        scheduler_path,
        {
            "optimizer": optimizer_policy,
            "scheduler": scheduler_policy,
        },
    )
    atomic_json(clip_path, clip_policy)
    atomic_json(loss_path, loss_policy)
    atomic_json(epoch_path, {
        "epoch_loop_count": len(epoch_loops),
        "epoch_loop": epoch_loops[0],
        "E0A_max_epochs_field": e0a_range_stop,
        "executed_epoch_budget": executed_epoch_budget,
        "range_stop_exclusive": range_stop_exclusive,
        "correction_required": epoch_budget_correction_required,
    })
    atomic_json(recipe_path, exact_recipe)

    E1_authorized = len(unresolved) == 0
    next_stage = (
        "V5_P3_F7_E1_RECONSTRUCTED_CONTROL_AND_GROUP_ABLATION_TRAINER_GENERATION"
        if E1_authorized
        else "V5_P3_F7_E0C_TARGETED_CONTROL_FLOW_RECOVERY"
    )

    decision = {
        "E0A_complete": True,
        "exact_epoch_semantics_resolved": True,
        "executed_epoch_budget": executed_epoch_budget,
        "E0A_epoch_budget_correction_required": (
            epoch_budget_correction_required
        ),
        "early_stopping_semantics_resolved": early_stopping[
            "semantics_resolved"
        ],
        "scheduler_semantics_resolved": scheduler_semantics_resolved,
        "gradient_clipping_semantics_resolved": (
            clip_semantics_resolved
        ),
        "loss_semantics_resolved": loss_semantics_resolved,
        "optimizer_step_semantics_resolved": (
            optimizer_semantics_resolved
        ),
        "unresolved_fields": unresolved,
        "E1_trainer_generation_authorized": E1_authorized,
        "actual_scientific_training_started": False,
        "F8_multi_seed_authorized": False,
        "feature_removal_authorized": False,
        "compact_interface_equivalence_authorized": False,
        "hardware_reduction_claim_authorized": False,
        "sealed_test_access": False,
        "next_stage": next_stage,
    }
    atomic_json(decision_path, decision)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "scope": (
            "Resolve the final F7 recipe semantics without executing training: "
            "distinguish the exclusive range stop from the actual epoch count; "
            "pin the early-stopping counter, threshold, comparison operator, "
            "increment/reset behavior and effective patience; separate the "
            "true optimizer step from scheduler.step; pin scheduler "
            "construction and cadence; pin gradient clipping and its placement "
            "between backward and optimizer.step; hash the exact compute_loss "
            "source; and authorize E1 only when every control-flow route is "
            "unambiguous."
        ),
        "finding": {
            "skeleton": str(skeleton_path),
            "skeleton_sha256": sha256_file(skeleton_path),
            "epoch_semantics": epoch_semantics,
            "E0A_epoch_budget_correction_required": (
                epoch_budget_correction_required
            ),
            "early_stopping": early_stopping,
            "optimizer": optimizer_policy,
            "scheduler": scheduler_policy,
            "gradient_clipping": clip_policy,
            "loss": loss_policy,
            "unresolved_fields": unresolved,
        },
        "decision": decision,
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "training_dataset_constructed": False,
            "validation_dataset_constructed": False,
            "training_feature_tensors_loaded": False,
            "validation_feature_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "optimizer_step_performed": False,
            "scientific_checkpoint_saved": False,
            "scientific_training_started": False,
            "historical_A4_recipe_claimed": False,
        },
        "artifacts": {
            "early_stopping_pin": str(early_path),
            "scheduler_and_optimizer_step_pin": str(scheduler_path),
            "gradient_clipping_pin": str(clip_path),
            "loss_source_pin": str(loss_path),
            "epoch_semantics": str(epoch_path),
            "final_recipe": str(recipe_path),
            "decision": str(decision_path),
        },
        "provenance": {
            "E0A_report_sha256": sha256_file(e0a_report_path),
            "E0A_lock_sha256": sha256_file(e0a_lock_path),
            "E0A_pin_sha256": sha256_file(e0a_pin_path),
            "P2_report_sha256": sha256_file(p2_report_path),
            "P2_lock_sha256": sha256_file(p2_lock_path),
            "F7_protocol_sha256": sha256_file(protocol_path),
            "reconstructed_route_sha256": sha256_file(
                reconstructed_route_path
            ),
            "skeleton_sha256": sha256_file(skeleton_path),
            "installed_script_sha256": sha256_file(installed_script),
            "early_stopping_pin_sha256": sha256_file(early_path),
            "scheduler_pin_sha256": sha256_file(scheduler_path),
            "gradient_clipping_pin_sha256": sha256_file(clip_path),
            "loss_source_pin_sha256": sha256_file(loss_path),
            "epoch_semantics_sha256": sha256_file(epoch_path),
            "final_recipe_sha256": sha256_file(recipe_path),
            "decision_sha256": sha256_file(decision_path),
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
            "early_stopping_pin_sha256": sha256_file(early_path),
            "scheduler_pin_sha256": sha256_file(scheduler_path),
            "gradient_clipping_pin_sha256": sha256_file(clip_path),
            "loss_source_pin_sha256": sha256_file(loss_path),
            "epoch_semantics_sha256": sha256_file(epoch_path),
            "final_recipe_sha256": sha256_file(recipe_path),
            "decision_sha256": sha256_file(decision_path),
            "E0B_complete": True,
            "E1_trainer_generation_authorized": E1_authorized,
            "actual_scientific_training_started": False,
            "F8_multi_seed_authorized": False,
            "feature_removal_authorized": False,
            "compact_interface_equivalence_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "training_feature_tensors_loaded": False,
            "validation_feature_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print(f"training_skeleton={skeleton_path}")
    print(f"training_skeleton_sha256={sha256_file(skeleton_path)}")
    print(
        "epoch_range="
        f"start={epoch_semantics['start']}:"
        f"stop_exclusive={epoch_semantics['stop_exclusive']}:"
        f"step={epoch_semantics['step']}:"
        f"executed_count={epoch_semantics['executed_epoch_count']}:"
        f"first={epoch_semantics['first_epoch_index']}:"
        f"last={epoch_semantics['last_epoch_index']}"
    )
    print(f"E0A_max_epochs_field={e0a_range_stop}")
    print(
        "E0A_epoch_budget_correction_required="
        f"{str(epoch_budget_correction_required).lower()}"
    )
    print(
        "early_stopping_classification="
        f"{early_stopping['classification']}"
    )
    print(
        "early_stopping_semantics_resolved="
        f"{str(early_stopping['semantics_resolved']).lower()}"
    )
    if early_stopping.get("active"):
        print(
            "early_stopping_policy="
            f"counter={early_stopping.get('counter')}:"
            f"threshold={early_stopping.get('threshold')}:"
            f"operator={early_stopping.get('operator')}:"
            f"increment={early_stopping.get('increment_value')}:"
            f"effective_patience="
            f"{early_stopping.get('effective_consecutive_non_improving_epochs')}"
        )
    else:
        print("early_stopping_policy=inactive")
    print(f"optimizer_variable={optimizer_variable}")
    print(f"optimizer_step_count={len(optimizer_steps)}")
    print(
        "optimizer_step_inside_training_batch="
        f"{str(optimizer_step_inside_batch_loop).lower()}"
    )
    print(f"scheduler_constructor_count={len(scheduler_constructors)}")
    print(f"scheduler_step_count={len(scheduler_steps)}")
    print(f"scheduler_policy={scheduler_policy['policy']}")
    if scheduler_policy.get("step_cadence"):
        print(f"scheduler_step_cadence={scheduler_policy['step_cadence']}")
    print(f"gradient_clip_call_count={len(clip_calls)}")
    print(f"gradient_clip_policy={clip_policy['policy']}")
    if len(clip_calls) == 1:
        print(
            "gradient_clip_threshold="
            f"{clip_calls[0]['threshold']}"
        )
        print(
            "gradient_clip_after_backward="
            f"{str(clip_policy['after_backward']).lower()}"
        )
        print(
            "gradient_clip_before_optimizer_step="
            f"{str(clip_policy['before_optimizer_step']).lower()}"
        )
    print(f"compute_loss_function_count={len(compute_loss_functions)}")
    print(f"compute_loss_call_count={len(compute_loss_calls)}")
    print(f"primitive_loss_call_count={len(primitive_loss_calls)}")
    if len(compute_loss_functions) == 1:
        print(
            "compute_loss_source_sha256="
            f"{compute_loss_functions[0]['excerpt_sha256']}"
        )
    print(f"unresolved_execution_field_count={len(unresolved)}")
    for row in unresolved:
        print(
            f"unresolved_execution_field={row['field']}:"
            f"reason={row['reason']}"
        )
    print(
        "E1_trainer_generation_authorized="
        f"{str(E1_authorized).lower()}"
    )
    print("actual_scientific_training_started=false")
    print("F8_multi_seed_authorized=false")
    print("feature_removal_authorized=false")
    print("compact_interface_equivalence_authorized=false")
    print("hardware_reduction_claim_authorized=false")
    print("model_loaded=false")
    print("checkpoint_loaded=false")
    print("training_feature_tensors_loaded=false")
    print("validation_feature_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={next_stage}")
    print(f"early_stopping_pin={early_path}")
    print(f"scheduler_and_optimizer_step_pin={scheduler_path}")
    print(f"gradient_clipping_pin={clip_path}")
    print(f"loss_source_pin={loss_path}")
    print(f"epoch_semantics={epoch_path}")
    print(f"final_recipe={recipe_path}")
    print(f"decision={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
