from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


STAGE = "V5_P3_F7_E0C_TARGETED_CONTROL_FLOW_RECOVERY"
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

SCHEDULER_CLASS_NAMES = {
    name
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

CLIP_CALLS = {
    "clip_grad_norm_",
    "clip_grad_value_",
}

EARLY_STOP_TOKENS = (
    "patience",
    "early_stop",
    "early_stopping",
    "bad_epoch",
    "bad_epochs",
    "no_improve",
    "epochs_without_improvement",
    "stale_epoch",
    "stale_epochs",
    "best_epoch",
)

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


def invert_operator(operator: str) -> str:
    return {
        ">": "<",
        ">=": "<=",
        "<": ">",
        "<=": ">=",
        "==": "==",
        "!=": "!=",
    }[operator]


def contains_break(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Break) for child in ast.walk(node))


def contains_checkpoint_call(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if call_name(child.func).rsplit(".", 1)[-1] in CHECKPOINT_CALLS:
                return True
    return False


class ProgramIndex:
    def __init__(self, tree: ast.Module, path: Path) -> None:
        self.tree = tree
        self.path = path
        self.lines = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()

        self.parent: dict[ast.AST, ast.AST] = {}
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.function_parameters: dict[str, list[str]] = {}
        self.function_defaults: dict[str, dict[str, ast.AST]] = {}
        self.callsites: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.assignments: dict[
            tuple[str | None, str],
            list[dict[str, Any]]
        ] = defaultdict(list)
        self.argparse_defaults: dict[str, ast.AST] = {}
        self.argparse_records: dict[str, dict[str, Any]] = {}

        self._build_parent_map()
        self._collect_functions()
        self._collect_argparse_defaults()
        self._collect_assignments()
        self._collect_callsites()

    def _build_parent_map(self) -> None:
        for parent in ast.walk(self.tree):
            for child in ast.iter_child_nodes(parent):
                self.parent[child] = parent

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

    def _collect_functions(self) -> None:
        for node in ast.walk(self.tree):
            if not isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue

            self.functions[node.name] = node

            positional = [
                argument.arg
                for argument in (
                    list(node.args.posonlyargs) + list(node.args.args)
                )
            ]
            keyword_only = [argument.arg for argument in node.args.kwonlyargs]
            self.function_parameters[node.name] = positional + keyword_only

            defaults: dict[str, ast.AST] = {}
            positional_defaults = list(node.args.defaults)
            offset = len(positional) - len(positional_defaults)
            for index, default in enumerate(positional_defaults):
                defaults[positional[offset + index]] = default

            for argument, default in zip(
                node.args.kwonlyargs,
                node.args.kw_defaults,
            ):
                if default is not None:
                    defaults[argument.arg] = default

            self.function_defaults[node.name] = defaults

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

    def _collect_assignments(self) -> None:
        for node in ast.walk(self.tree):
            function_name = self.enclosing_function(node)

            if isinstance(node, ast.Assign):
                for target in node.targets:
                    for name in target_names(target):
                        self.assignments[(function_name, name)].append({
                            "value": node.value,
                            "node": node,
                            "line": int(node.lineno),
                            "kind": "assign",
                        })

            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                for name in target_names(node.target):
                    self.assignments[(function_name, name)].append({
                        "value": node.value,
                        "node": node,
                        "line": int(node.lineno),
                        "kind": "assign",
                    })

            elif isinstance(node, ast.AugAssign):
                for name in target_names(node.target):
                    self.assignments[(function_name, name)].append({
                        "value": node.value,
                        "node": node,
                        "line": int(node.lineno),
                        "kind": "augassign",
                        "operator": type(node.op).__name__,
                    })

    def _collect_callsites(self) -> None:
        local_names = set(self.functions)

        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue

            short = call_name(node.func).rsplit(".", 1)[-1]
            if short not in local_names:
                continue

            caller = self.enclosing_function(node)
            self.callsites[short].append({
                "node": node,
                "caller": caller,
                "line": int(node.lineno),
                "record": source_record(self.path, self.lines, node),
            })

    def bind_parameter(
        self,
        *,
        callee: str,
        parameter: str,
        call: ast.Call,
    ) -> ast.AST | None:
        parameters = self.function_parameters[callee]

        for keyword in call.keywords:
            if keyword.arg == parameter:
                return keyword.value

        if parameter in parameters:
            position = parameters.index(parameter)
            if position < len(call.args):
                return call.args[position]

        return self.function_defaults.get(callee, {}).get(parameter)

    def loop_context(self, node: ast.AST) -> dict[str, Any]:
        loops = []
        for ancestor in self.ancestors(node):
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

        token = normalize(
            " ".join(
                (
                    row.get("target", "")
                    + " "
                    + row.get("iterator", "")
                    + " "
                    + row.get("condition", "")
                )
                for row in loops
            )
        )

        return {
            "loops": loops,
            "epoch": "epoch" in token,
            "batch": any(
                term in token
                for term in (
                    "batch",
                    "loader",
                    "train_loader",
                    "dataloader",
                )
            ),
        }


class InterproceduralResolver:
    def __init__(self, index: ProgramIndex) -> None:
        self.index = index

    def resolve(
        self,
        node: ast.AST,
        *,
        function_name: str | None,
        use_line: int | None,
        stack: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        expression = ast.unparse(node)
        identity = (
            function_name or "<module>",
            expression,
            use_line,
        )
        identity_key = repr(identity)
        if identity_key in stack:
            return {
                "resolved": False,
                "expression": expression,
                "reason": "cyclic_resolution",
            }
        stack = stack + (identity_key,)

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
                use_line=use_line,
                stack=stack,
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
                use_line=use_line,
                stack=stack,
            )
            right = self.resolve(
                node.right,
                function_name=function_name,
                use_line=use_line,
                stack=stack,
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
                    raise ValueError("unsupported operator")
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
                parameters = self.index.function_parameters.get(
                    function_name,
                    [],
                )
                if name in parameters:
                    callsites = self.index.callsites.get(function_name, [])
                    resolved_calls = []

                    for callsite in callsites:
                        bound = self.index.bind_parameter(
                            callee=function_name,
                            parameter=name,
                            call=callsite["node"],
                        )
                        if bound is None:
                            continue

                        child = self.resolve(
                            bound,
                            function_name=callsite["caller"],
                            use_line=callsite["line"],
                            stack=stack,
                        )
                        resolved_calls.append({
                            "callsite": callsite["record"],
                            "caller": callsite["caller"],
                            "resolution": child,
                        })

                    resolved_values = []
                    for row in resolved_calls:
                        child = row["resolution"]
                        if child.get("resolved"):
                            value = child.get("value")
                            if value not in resolved_values:
                                resolved_values.append(value)

                    if len(resolved_values) == 1:
                        return {
                            "resolved": True,
                            "value": resolved_values[0],
                            "expression": expression,
                            "route": "unique_interprocedural_parameter_binding",
                            "function": function_name,
                            "parameter": name,
                            "callsites": resolved_calls,
                        }

                    default = self.index.function_defaults.get(
                        function_name,
                        {},
                    ).get(name)
                    if default is not None:
                        child = self.resolve(
                            default,
                            function_name=function_name,
                            use_line=use_line,
                            stack=stack,
                        )
                        if child.get("resolved"):
                            return {
                                **child,
                                "expression": expression,
                                "route": "function_parameter_default",
                                "function": function_name,
                                "parameter": name,
                                "child": child,
                            }

                    return {
                        "resolved": False,
                        "expression": expression,
                        "reason": (
                            "parameter_binding_not_unique_or_unresolved"
                        ),
                        "function": function_name,
                        "parameter": name,
                        "resolved_values": resolved_values,
                        "callsites": resolved_calls,
                    }

            assignment_rows = []

            for scope in (function_name, None):
                rows = self.index.assignments.get((scope, name), [])
                for row in rows:
                    if (
                        scope == function_name
                        and use_line is not None
                        and row["line"] >= use_line
                    ):
                        continue
                    if row["kind"] != "assign":
                        continue

                    child = self.resolve(
                        row["value"],
                        function_name=scope,
                        use_line=row["line"],
                        stack=stack,
                    )
                    assignment_rows.append({
                        "scope": scope,
                        "record": source_record(
                            self.index.path,
                            self.index.lines,
                            row["node"],
                        ),
                        "resolution": child,
                    })

            values = []
            for row in assignment_rows:
                child = row["resolution"]
                if child.get("resolved"):
                    value = child.get("value")
                    if value not in values:
                        values.append(value)

            if len(values) == 1:
                return {
                    "resolved": True,
                    "value": values[0],
                    "expression": expression,
                    "route": "unique_prior_assignment_value",
                    "symbol": name,
                    "assignments": assignment_rows,
                }

            return {
                "resolved": False,
                "expression": expression,
                "reason": (
                    "symbol_not_found"
                    if not assignment_rows
                    else "prior_assignment_values_not_unique"
                ),
                "symbol": name,
                "resolved_values": values,
                "assignments": assignment_rows,
            }

        if isinstance(node, ast.Attribute):
            full = ast.unparse(node)

            if (
                isinstance(node.value, ast.Name)
                and node.value.id in {"args", "config", "cfg", "options"}
            ):
                dest = node.attr
                if dest in self.index.argparse_defaults:
                    child = self.resolve(
                        self.index.argparse_defaults[dest],
                        function_name=function_name,
                        use_line=use_line,
                        stack=stack,
                    )
                    return {
                        **child,
                        "expression": expression,
                        "route": "argparse_default",
                        "dest": dest,
                        "argparse_record": self.index.argparse_records[dest],
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
                use_line=use_line,
                stack=stack,
            )
            index = self.resolve(
                node.slice,
                function_name=function_name,
                use_line=use_line,
                stack=stack,
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
                    use_line=use_line,
                    stack=stack,
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
                        use_line=use_line,
                        stack=stack,
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


class ExecutionContextResolver:
    def __init__(self, index: ProgramIndex) -> None:
        self.index = index
        self.memo: dict[str | None, list[dict[str, Any]]] = {}

    def function_contexts(
        self,
        function_name: str | None,
        stack: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        if function_name in self.memo:
            return self.memo[function_name]

        key = function_name or "<module>"
        if key in stack:
            return []
        stack = stack + (key,)

        if function_name is None:
            contexts = [{
                "path": ["<module>"],
                "epoch": False,
                "batch": False,
                "callsites": [],
            }]
            self.memo[function_name] = contexts
            return contexts

        callsites = self.index.callsites.get(function_name, [])
        contexts = []

        for callsite in callsites:
            direct = self.index.loop_context(callsite["node"])
            caller_contexts = self.function_contexts(
                callsite["caller"],
                stack,
            )
            if not caller_contexts:
                caller_contexts = [{
                    "path": [callsite["caller"] or "<module>"],
                    "epoch": False,
                    "batch": False,
                    "callsites": [],
                }]

            for caller_context in caller_contexts:
                contexts.append({
                    "path": (
                        caller_context["path"] + [function_name]
                    ),
                    "epoch": (
                        caller_context["epoch"] or direct["epoch"]
                    ),
                    "batch": (
                        caller_context["batch"] or direct["batch"]
                    ),
                    "callsites": (
                        caller_context["callsites"]
                        + [{
                            "callee": function_name,
                            "caller": callsite["caller"],
                            "direct_loop_context": direct,
                            "record": callsite["record"],
                        }]
                    ),
                })

        self.memo[function_name] = contexts
        return contexts

    def node_context(self, node: ast.AST) -> dict[str, Any]:
        function_name = self.index.enclosing_function(node)
        direct = self.index.loop_context(node)
        function_contexts = self.function_contexts(function_name)

        if not function_contexts:
            function_contexts = [{
                "path": [function_name or "<module>"],
                "epoch": False,
                "batch": False,
                "callsites": [],
            }]

        combined = []
        for context in function_contexts:
            combined.append({
                **context,
                "epoch": context["epoch"] or direct["epoch"],
                "batch": context["batch"] or direct["batch"],
                "direct_node_context": direct,
            })

        return {
            "function": function_name,
            "direct_context": direct,
            "execution_paths": combined,
            "all_paths_epoch": bool(
                combined
                and all(row["epoch"] for row in combined)
            ),
            "all_paths_batch": bool(
                combined
                and all(row["batch"] for row in combined)
            ),
            "any_path_epoch": any(row["epoch"] for row in combined),
            "any_path_batch": any(row["batch"] for row in combined),
        }


def keyword_node(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def resolve_argument(
    *,
    resolver: InterproceduralResolver,
    index: ProgramIndex,
    call: ast.Call,
    keyword: str,
    positional_index: int | None,
    runtime_callable: Any | None,
) -> dict[str, Any]:
    function_name = index.enclosing_function(call)
    use_line = int(call.lineno)

    node = keyword_node(call, keyword)
    if node is not None:
        result = resolver.resolve(
            node,
            function_name=function_name,
            use_line=use_line,
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
            use_line=use_line,
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


def classify_staleness_expression(node: ast.AST) -> dict[str, Any]:
    expression = ast.unparse(node)
    names = sorted({
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
    })

    if isinstance(node, ast.Name):
        return {
            "classification": "DIRECT_STALE_COUNTER",
            "expression": expression,
            "counter": node.id,
            "names": names,
        }

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        left = ast.unparse(node.left)
        right = ast.unparse(node.right)
        token = normalize(expression)
        if "epoch" in token and "best_epoch" in token:
            return {
                "classification": "CURRENT_EPOCH_MINUS_BEST_EPOCH",
                "expression": expression,
                "current_epoch_expression": left,
                "best_epoch_expression": right,
                "names": names,
            }

    token = normalize(expression)
    if "epoch" in token and "best_epoch" in token:
        return {
            "classification": "EPOCH_DISTANCE_EXPRESSION",
            "expression": expression,
            "names": names,
        }

    return {
        "classification": "GENERIC_STALENESS_EXPRESSION",
        "expression": expression,
        "names": names,
    }


def find_early_stop(
    *,
    tree: ast.Module,
    index: ProgramIndex,
    resolver: InterproceduralResolver,
    context_resolver: ExecutionContextResolver,
) -> dict[str, Any]:
    candidates = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not contains_break(node):
            continue

        token = normalize(ast.unparse(node))
        if not any(term in token for term in EARLY_STOP_TOKENS):
            continue

        context = context_resolver.node_context(node)

        comparisons = [
            child
            for child in ast.walk(node.test)
            if isinstance(child, ast.Compare)
            and len(child.ops) == 1
            and len(child.comparators) == 1
        ]

        comparison_rows = []
        for comparison in comparisons:
            function_name = index.enclosing_function(comparison)
            line = int(comparison.lineno)

            left_node = comparison.left
            right_node = comparison.comparators[0]
            operator = compare_operator(comparison.ops[0])

            left = resolver.resolve(
                left_node,
                function_name=function_name,
                use_line=line,
            )
            right = resolver.resolve(
                right_node,
                function_name=function_name,
                use_line=line,
            )

            threshold = None
            threshold_side = None
            staleness_node = None
            normalized_operator = operator

            if (
                right.get("resolved")
                and isinstance(right.get("value"), (int, float))
                and not isinstance(right.get("value"), bool)
            ):
                threshold = float(right["value"])
                threshold_side = "right"
                staleness_node = left_node
            elif (
                left.get("resolved")
                and isinstance(left.get("value"), (int, float))
                and not isinstance(left.get("value"), bool)
            ):
                threshold = float(left["value"])
                threshold_side = "left"
                staleness_node = right_node
                normalized_operator = invert_operator(operator)

            comparison_rows.append({
                "left_expression": ast.unparse(left_node),
                "left_resolution": left,
                "operator": operator,
                "right_expression": ast.unparse(right_node),
                "right_resolution": right,
                "numeric_threshold": threshold,
                "numeric_threshold_side": threshold_side,
                "normalized_operator": normalized_operator,
                "staleness": (
                    classify_staleness_expression(staleness_node)
                    if staleness_node is not None
                    else None
                ),
                "record": source_record(
                    index.path,
                    index.lines,
                    comparison,
                ),
            })

        viable = [
            row
            for row in comparison_rows
            if row["numeric_threshold"] is not None
            and row["staleness"] is not None
            and row["normalized_operator"] in {">=", ">"}
        ]

        candidates.append({
            "record": source_record(index.path, index.lines, node),
            "context": context,
            "comparison_rows": comparison_rows,
            "viable_comparisons": viable,
        })

    if len(candidates) != 1:
        return {
            "active": bool(candidates),
            "classification": (
                "NO_BREAK_BASED_EARLY_STOP"
                if not candidates
                else "MULTIPLE_BREAK_BASED_EARLY_STOP_CANDIDATES"
            ),
            "candidates": candidates,
            "semantics_resolved": not candidates,
        }

    candidate = candidates[0]
    viable = candidate["viable_comparisons"]
    if len(viable) != 1:
        return {
            "active": True,
            "classification": (
                "EARLY_STOP_THRESHOLD_OR_STALENESS_EXPRESSION_UNRESOLVED"
            ),
            "candidate": candidate,
            "semantics_resolved": False,
        }

    comparison = viable[0]
    threshold = comparison["numeric_threshold"]
    operator = comparison["normalized_operator"]
    staleness = comparison["staleness"]

    effective_patience = (
        int(threshold)
        if operator == ">="
        else int(threshold) + 1
    )

    support = {
        "counter_events": [],
        "best_epoch_updates": [],
        "improvement_branches": [],
    }

    if staleness["classification"] == "DIRECT_STALE_COUNTER":
        counter = staleness["counter"]

        for row in index.assignments.get(
            (candidate["context"]["function"], counter),
            [],
        ):
            resolution = resolver.resolve(
                row["value"],
                function_name=candidate["context"]["function"],
                use_line=row["line"],
            )
            support["counter_events"].append({
                "kind": row["kind"],
                "operator": row.get("operator"),
                "resolution": resolution,
                "record": source_record(
                    index.path,
                    index.lines,
                    row["node"],
                ),
            })

        reset_zero = any(
            event["kind"] == "assign"
            and event["resolution"].get("resolved")
            and event["resolution"].get("value") == 0
            for event in support["counter_events"]
        )
        increment_one = any(
            event["kind"] == "augassign"
            and event.get("operator") == "Add"
            and event["resolution"].get("resolved")
            and event["resolution"].get("value") == 1
            for event in support["counter_events"]
        )

        semantics_resolved = bool(
            threshold >= 0
            and float(threshold).is_integer()
            and reset_zero
            and increment_one
            and candidate["context"]["any_path_epoch"]
        )

        classification = (
            "ACTIVE_DIRECT_CONSECUTIVE_NON_IMPROVEMENT_COUNTER"
            if semantics_resolved
            else "DIRECT_COUNTER_SUPPORT_EVENTS_INCOMPLETE"
        )

    elif staleness["classification"] in {
        "CURRENT_EPOCH_MINUS_BEST_EPOCH",
        "EPOCH_DISTANCE_EXPRESSION",
    }:
        names = staleness["names"]
        best_epoch_names = [
            name for name in names
            if "best_epoch" in normalize(name)
        ]

        for best_name in best_epoch_names:
            for scope in (
                candidate["context"]["function"],
                None,
            ):
                for row in index.assignments.get((scope, best_name), []):
                    record = source_record(
                        index.path,
                        index.lines,
                        row["node"],
                    )
                    value_expression = ast.unparse(row["value"])
                    if "epoch" in normalize(value_expression):
                        support["best_epoch_updates"].append({
                            "name": best_name,
                            "scope": scope,
                            "value_expression": value_expression,
                            "record": record,
                        })

        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue

            node_text = ast.unparse(node)
            node_token = normalize(node_text)
            updates_best_epoch = any(
                normalize(best_name) in node_token
                and "epoch" in node_token
                for best_name in best_epoch_names
            )
            if not updates_best_epoch:
                continue

            support["improvement_branches"].append({
                "test": ast.unparse(node.test),
                "checkpoint_call_in_body": any(
                    contains_checkpoint_call(child)
                    for child in node.body
                ),
                "record": source_record(index.path, index.lines, node),
            })

        semantics_resolved = bool(
            threshold >= 0
            and float(threshold).is_integer()
            and candidate["context"]["any_path_epoch"]
            and len(support["best_epoch_updates"]) >= 1
            and len(support["improvement_branches"]) >= 1
        )

        classification = (
            "ACTIVE_EPOCH_DISTANCE_FROM_LAST_IMPROVEMENT"
            if semantics_resolved
            else "EPOCH_DISTANCE_SUPPORT_EVENTS_INCOMPLETE"
        )

    else:
        semantics_resolved = False
        classification = "GENERIC_STALENESS_EXPRESSION_NOT_CERTIFIED"

    return {
        "active": True,
        "classification": classification,
        "threshold": threshold,
        "operator": operator,
        "effective_consecutive_non_improving_epochs": effective_patience,
        "staleness": staleness,
        "candidate": candidate,
        "support": support,
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

    e0b_report_path = f7_root / (
        "V5_P3_F7_E0B_REMAINING_RECIPE_SEMANTIC_PIN_REPORT.json"
    )
    e0b_lock_path = f7_root / (
        "V5_P3_F7_E0B_REMAINING_RECIPE_SEMANTIC_PIN_LOCK.json"
    )
    e0b_recipe_path = f7_root / (
        "F7_E0B_FINAL_RECONSTRUCTED_EXECUTION_RECIPE.json"
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

    e0b_report, e0b_lock = verify_report_lock(
        e0b_report_path,
        e0b_lock_path,
    )
    p2_report, p2_lock = verify_report_lock(
        p2_report_path,
        p2_lock_path,
    )
    e0b_recipe = load_json(e0b_recipe_path)
    e0a_pin = load_json(e0a_pin_path)
    protocol = load_json(protocol_path)
    reconstructed_route = load_json(reconstructed_route_path)

    require(e0b_lock.get("E0B_complete") is True, "E0B incomplete")
    require(
        e0b_lock.get("E1_trainer_generation_authorized") is False,
        "E1 already authorized before E0C",
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
    require(reconstructed_route["status"] == "FROZEN", "route not frozen")
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
        "train item count changed",
    )
    require(
        int(reconstructed_route["dataset"]["validation_items"])
        == EXPECTED_VALIDATION_ITEMS,
        "validation item count changed",
    )

    skeleton_path = (repo / SKELETON_RELATIVE_PATH).resolve()
    require(skeleton_path.is_file(), f"skeleton missing: {skeleton_path}")
    require(
        sha256_file(skeleton_path)
        == e0b_report["finding"]["skeleton_sha256"],
        "skeleton hash differs from E0B",
    )

    text = skeleton_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    tree = ast.parse(text)
    index = ProgramIndex(tree, skeleton_path)
    resolver = InterproceduralResolver(index)
    context_resolver = ExecutionContextResolver(index)

    optimizer_constructors = []
    scheduler_constructors = []
    optimizer_steps = []
    scheduler_steps = []
    clip_calls = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        full_name = call_name(node.func)
        short = full_name.rsplit(".", 1)[-1]
        function_name = index.enclosing_function(node)
        parent = index.parent.get(node)

        assigned = None
        if isinstance(parent, ast.Assign) and len(parent.targets) == 1:
            names = target_names(parent.targets[0])
            assigned = names[0] if len(names) == 1 else None
        elif isinstance(parent, ast.AnnAssign):
            names = target_names(parent.target)
            assigned = names[0] if len(names) == 1 else None

        if short in OPTIMIZER_CLASSES:
            optimizer_constructors.append({
                "variable": assigned,
                "implementation": (
                    f"{OPTIMIZER_CLASSES[short].__module__}."
                    f"{OPTIMIZER_CLASSES[short].__qualname__}"
                ),
                "record": source_record(skeleton_path, index.lines, node),
            })

        if short in SCHEDULER_CLASS_NAMES:
            scheduler_constructors.append({
                "variable": assigned,
                "implementation": full_name,
                "record": source_record(skeleton_path, index.lines, node),
            })

        if short == "step" and isinstance(node.func, ast.Attribute):
            base = ast.unparse(node.func.value)
            record = {
                "base": base,
                "function": function_name,
                "context": context_resolver.node_context(node),
                "record": source_record(skeleton_path, index.lines, node),
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

        if short in CLIP_CALLS:
            threshold_keyword = (
                "max_norm"
                if short == "clip_grad_norm_"
                else "clip_value"
            )
            threshold = resolve_argument(
                resolver=resolver,
                index=index,
                call=node,
                keyword=threshold_keyword,
                positional_index=1,
                runtime_callable=None,
            )
            clip_calls.append({
                "implementation": full_name,
                "function": function_name,
                "threshold": threshold,
                "context": context_resolver.node_context(node),
                "record": source_record(skeleton_path, index.lines, node),
            })

    require(
        len(optimizer_constructors) == 1,
        "optimizer constructor is not unique",
    )

    optimizer_variable = optimizer_constructors[0]["variable"]
    require(optimizer_variable is not None, "optimizer variable unresolved")

    # Reclassify step calls after constructors are fully known.
    optimizer_steps = []
    scheduler_steps = []
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
            "function": index.enclosing_function(node),
            "context": context_resolver.node_context(node),
            "record": source_record(skeleton_path, index.lines, node),
        }

        if base in optimizer_variables:
            optimizer_steps.append(record)
        elif base in scheduler_variables:
            scheduler_steps.append(record)

    require(
        len(optimizer_steps) == 1,
        f"expected one optimizer.step, found {len(optimizer_steps)}",
    )

    optimizer_context = optimizer_steps[0]["context"]
    optimizer_cadence_resolved = bool(
        optimizer_context["all_paths_epoch"]
        and optimizer_context["all_paths_batch"]
    )

    require(
        len(clip_calls) == 1,
        f"expected one gradient clip call, found {len(clip_calls)}",
    )
    clip = clip_calls[0]
    threshold = clip["threshold"]

    clip_order = {
        "after_backward": False,
        "before_optimizer_step": False,
    }
    clip_function = clip["function"]
    clip_line = clip["record"]["line_start"]

    backward_lines = []
    optimizer_step_lines_same_function = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        short = call_name(node.func).rsplit(".", 1)[-1]
        function_name = index.enclosing_function(node)

        if short == "backward" and function_name == clip_function:
            backward_lines.append(int(node.lineno))

        if (
            short == "step"
            and isinstance(node.func, ast.Attribute)
            and ast.unparse(node.func.value) == optimizer_variable
            and function_name == clip_function
        ):
            optimizer_step_lines_same_function.append(int(node.lineno))

    clip_order["after_backward"] = any(
        line < clip_line
        for line in backward_lines
    )
    clip_order["before_optimizer_step"] = any(
        line > clip_line
        for line in optimizer_step_lines_same_function
    )

    clip_semantics_resolved = bool(
        threshold.get("resolved")
        and isinstance(threshold.get("value"), (int, float))
        and not isinstance(threshold.get("value"), bool)
        and float(threshold["value"]) > 0
        and clip["context"]["all_paths_epoch"]
        and clip["context"]["all_paths_batch"]
        and clip_order["after_backward"]
        and clip_order["before_optimizer_step"]
    )

    early_stopping = find_early_stop(
        tree=tree,
        index=index,
        resolver=resolver,
        context_resolver=context_resolver,
    )

    scheduler_semantics_resolved = bool(
        e0b_report["decision"]["scheduler_semantics_resolved"]
    )
    loss_semantics_resolved = bool(
        e0b_report["decision"]["loss_semantics_resolved"]
    )
    exact_epoch_semantics_resolved = bool(
        e0b_report["decision"]["exact_epoch_semantics_resolved"]
    )

    unresolved = []
    if not early_stopping["semantics_resolved"]:
        unresolved.append({
            "field": "early_stopping_policy",
            "reason": early_stopping["classification"],
        })
    if not clip_semantics_resolved:
        unresolved.append({
            "field": "gradient_clipping_policy",
            "reason": (
                "interprocedural threshold, execution context or ordering "
                "remains unresolved"
            ),
        })
    if not optimizer_cadence_resolved:
        unresolved.append({
            "field": "optimizer_step_cadence",
            "reason": (
                "not every reachable optimizer.step path is inside both "
                "the epoch and training-batch contexts"
            ),
        })
    if not scheduler_semantics_resolved:
        unresolved.append({
            "field": "scheduler_policy",
            "reason": "E0B scheduler semantics did not pass",
        })
    if not loss_semantics_resolved:
        unresolved.append({
            "field": "loss_route",
            "reason": "E0B loss semantics did not pass",
        })
    if not exact_epoch_semantics_resolved:
        unresolved.append({
            "field": "epoch_semantics",
            "reason": "E0B exact epoch semantics did not pass",
        })

    optimizer_path = output_dir / (
        "F7_E0C_INTERPROCEDURAL_OPTIMIZER_STEP_CADENCE_RECOVERY.json"
    )
    clip_path = output_dir / (
        "F7_E0C_INTERPROCEDURAL_GRADIENT_CLIP_THRESHOLD_RECOVERY.json"
    )
    early_path = output_dir / (
        "F7_E0C_EARLY_STOP_THRESHOLD_AND_STALENESS_RECOVERY.json"
    )
    call_graph_path = output_dir / (
        "F7_E0C_LOCAL_CALL_GRAPH_AND_PARAMETER_BINDINGS.json"
    )
    recipe_path = output_dir / (
        "F7_E0C_FINAL_EXECUTION_RECIPE_LOCK.json"
    )
    decision_path = output_dir / (
        "F7_E0C_TRAINER_GENERATION_DECISION.json"
    )

    atomic_json(
        optimizer_path,
        {
            "optimizer_constructor": optimizer_constructors[0],
            "optimizer_step_count": len(optimizer_steps),
            "optimizer_step": optimizer_steps[0],
            "all_paths_epoch": optimizer_context["all_paths_epoch"],
            "all_paths_batch": optimizer_context["all_paths_batch"],
            "cadence": (
                "ONE_STEP_PER_TRAINING_BATCH_WITHIN_EPOCH"
                if optimizer_cadence_resolved
                else "UNRESOLVED"
            ),
            "semantics_resolved": optimizer_cadence_resolved,
        },
    )
    atomic_json(
        clip_path,
        {
            "clip_call": clip,
            "threshold": threshold,
            "ordering": clip_order,
            "all_paths_epoch": clip["context"]["all_paths_epoch"],
            "all_paths_batch": clip["context"]["all_paths_batch"],
            "semantics_resolved": clip_semantics_resolved,
        },
    )
    atomic_json(early_path, early_stopping)

    call_graph = {
        "functions": {
            name: {
                "parameters": index.function_parameters[name],
                "defaults": {
                    key: ast.unparse(value)
                    for key, value in index.function_defaults[name].items()
                },
                "callsites": [
                    {
                        "caller": row["caller"],
                        "line": row["line"],
                        "record": row["record"],
                        "direct_loop_context": index.loop_context(
                            row["node"]
                        ),
                    }
                    for row in index.callsites.get(name, [])
                ],
                "execution_contexts": context_resolver.function_contexts(
                    name
                ),
            }
            for name in sorted(index.functions)
        }
    }
    atomic_json(call_graph_path, call_graph)

    final_recipe = {
        "status": "FROZEN" if not unresolved else "PARTIALLY_RESOLVED",
        "route_type": (
            "reconstructed_F7_recipe_with_interprocedural_control_flow_pin"
        ),
        "not_claimed": (
            "byte-identical historical A4 trainer reproduction"
        ),
        "primary_seed": EXPECTED_PRIMARY_SEED,
        "train_batch_size": e0a_pin["train_batch_size"],
        "validation_batch_size": e0a_pin[
            "validation_batch_size"
        ],
        "optimizer": e0a_pin["optimizer"],
        "learning_rate": e0a_pin["learning_rate"],
        "weight_decay": e0a_pin["weight_decay"],
        "executed_epoch_budget": e0b_recipe[
            "executed_epoch_budget"
        ],
        "epoch_loop": e0b_recipe["epoch_loop"],
        "early_stopping": early_stopping,
        "optimizer_step": {
            "cadence": (
                "ONE_STEP_PER_TRAINING_BATCH_WITHIN_EPOCH"
                if optimizer_cadence_resolved
                else "UNRESOLVED"
            ),
            "source": optimizer_steps[0],
        },
        "scheduler": e0b_recipe["scheduler"],
        "gradient_clipping": {
            "implementation": clip["implementation"],
            "threshold": threshold,
            "ordering": clip_order,
            "execution_context": clip["context"],
        },
        "loss": e0b_recipe["loss"],
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
    atomic_json(recipe_path, final_recipe)

    E1_authorized = len(unresolved) == 0
    next_stage = (
        "V5_P3_F7_E1_RECONSTRUCTED_CONTROL_AND_GROUP_ABLATION_TRAINER_GENERATION"
        if E1_authorized
        else "V5_P3_F7_E0D_MANUAL_SOURCE_EXCERPT_PIN"
    )

    decision = {
        "E0B_complete": True,
        "interprocedural_parameter_binding_complete": (
            threshold.get("resolved") is True
        ),
        "optimizer_step_cadence_resolved": optimizer_cadence_resolved,
        "gradient_clipping_semantics_resolved": clip_semantics_resolved,
        "early_stopping_semantics_resolved": early_stopping[
            "semantics_resolved"
        ],
        "scheduler_semantics_resolved": scheduler_semantics_resolved,
        "loss_semantics_resolved": loss_semantics_resolved,
        "exact_epoch_semantics_resolved": exact_epoch_semantics_resolved,
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
            "Recover E0B's three false-negative or unresolved control-flow "
            "fields without executing training: build a local function-call "
            "graph; propagate loop context through helper calls; resolve "
            "function parameters from unique call-site bindings; certify that "
            "optimizer.step executes once per training batch inside the epoch "
            "loop; resolve the gradient-clip threshold and ordering; and parse "
            "the break-bearing early-stop condition as either a stale counter "
            "or current-epoch minus best-epoch expression."
        ),
        "finding": {
            "skeleton": str(skeleton_path),
            "skeleton_sha256": sha256_file(skeleton_path),
            "optimizer_step": optimizer_steps[0],
            "optimizer_step_cadence_resolved": optimizer_cadence_resolved,
            "gradient_clip": clip,
            "gradient_clip_order": clip_order,
            "gradient_clip_semantics_resolved": clip_semantics_resolved,
            "early_stopping": early_stopping,
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
            "optimizer_step_recovery": str(optimizer_path),
            "gradient_clip_recovery": str(clip_path),
            "early_stop_recovery": str(early_path),
            "call_graph_and_bindings": str(call_graph_path),
            "final_recipe_lock": str(recipe_path),
            "decision": str(decision_path),
        },
        "provenance": {
            "E0B_report_sha256": sha256_file(e0b_report_path),
            "E0B_lock_sha256": sha256_file(e0b_lock_path),
            "E0B_recipe_sha256": sha256_file(e0b_recipe_path),
            "E0A_pin_sha256": sha256_file(e0a_pin_path),
            "P2_report_sha256": sha256_file(p2_report_path),
            "P2_lock_sha256": sha256_file(p2_lock_path),
            "F7_protocol_sha256": sha256_file(protocol_path),
            "reconstructed_route_sha256": sha256_file(
                reconstructed_route_path
            ),
            "skeleton_sha256": sha256_file(skeleton_path),
            "installed_script_sha256": sha256_file(installed_script),
            "optimizer_recovery_sha256": sha256_file(optimizer_path),
            "clip_recovery_sha256": sha256_file(clip_path),
            "early_stop_recovery_sha256": sha256_file(early_path),
            "call_graph_sha256": sha256_file(call_graph_path),
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
            "optimizer_recovery_sha256": sha256_file(optimizer_path),
            "clip_recovery_sha256": sha256_file(clip_path),
            "early_stop_recovery_sha256": sha256_file(early_path),
            "call_graph_sha256": sha256_file(call_graph_path),
            "final_recipe_sha256": sha256_file(recipe_path),
            "decision_sha256": sha256_file(decision_path),
            "E0C_complete": True,
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
    print(f"optimizer_variable={optimizer_variable}")
    print(f"optimizer_step_count={len(optimizer_steps)}")
    print(
        "optimizer_step_all_paths_epoch="
        f"{str(optimizer_context['all_paths_epoch']).lower()}"
    )
    print(
        "optimizer_step_all_paths_batch="
        f"{str(optimizer_context['all_paths_batch']).lower()}"
    )
    print(
        "optimizer_step_cadence_resolved="
        f"{str(optimizer_cadence_resolved).lower()}"
    )
    print(f"gradient_clip_implementation={clip['implementation']}")
    print(f"gradient_clip_threshold={threshold}")
    print(
        "gradient_clip_all_paths_epoch="
        f"{str(clip['context']['all_paths_epoch']).lower()}"
    )
    print(
        "gradient_clip_all_paths_batch="
        f"{str(clip['context']['all_paths_batch']).lower()}"
    )
    print(
        "gradient_clip_after_backward="
        f"{str(clip_order['after_backward']).lower()}"
    )
    print(
        "gradient_clip_before_optimizer_step="
        f"{str(clip_order['before_optimizer_step']).lower()}"
    )
    print(
        "gradient_clipping_semantics_resolved="
        f"{str(clip_semantics_resolved).lower()}"
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
            f"threshold={early_stopping.get('threshold')}:"
            f"operator={early_stopping.get('operator')}:"
            f"effective_patience="
            f"{early_stopping.get('effective_consecutive_non_improving_epochs')}:"
            f"staleness={early_stopping.get('staleness')}"
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
    print(f"optimizer_step_recovery={optimizer_path}")
    print(f"gradient_clip_recovery={clip_path}")
    print(f"early_stop_recovery={early_path}")
    print(f"call_graph_and_bindings={call_graph_path}")
    print(f"final_recipe_lock={recipe_path}")
    print(f"decision={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
