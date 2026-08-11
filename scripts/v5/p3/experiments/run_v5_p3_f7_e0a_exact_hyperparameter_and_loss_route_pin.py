from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


STAGE = "V5_P3_F7_E0A_EXACT_HYPERPARAMETER_AND_LOSS_ROUTE_PIN"
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

LOSS_NAMES = {
    "binary_cross_entropy_with_logits",
    "cross_entropy",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "focal_loss",
}

EARLY_STOP_TOKENS = (
    "patience",
    "early_stop",
    "early_stopping",
    "bad_epoch",
    "no_improve",
    "epochs_without_improvement",
)

GRADIENT_CLIP_TOKENS = (
    "clip_grad",
    "max_grad_norm",
    "gradient_clip",
)

SCHEDULER_TOKENS = (
    "scheduler",
    "lr_scheduler",
    "reduce_lr",
    "cosineannealing",
    "steplr",
    "onecycle",
)


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
    return {
        "path": str(path.resolve()),
        "line_start": start,
        "line_end": end,
        "source": ast.unparse(node),
        "excerpt": "\n".join(lines[max(0, start - 1):end]),
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

            arguments = list(node.args.args)
            defaults = list(node.args.defaults)
            offset = len(arguments) - len(defaults)
            mapping = {}

            for index, default in enumerate(defaults):
                mapping[arguments[offset + index].arg] = default

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
                    self._register_target(target, node.value)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                self._register_target(node.target, node.value)

    def _register_target(self, target: ast.AST, value: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.assignments.setdefault(target.id, []).append(value)
        elif isinstance(target, ast.Attribute):
            self.assignments.setdefault(
                ast.unparse(target),
                [],
            ).append(value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(value, (ast.Tuple, ast.List)):
                for child_target, child_value in zip(
                    target.elts,
                    value.elts,
                ):
                    self._register_target(child_target, child_value)

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
        identity = (
            function_name or "<module>",
            expression,
        )
        identity_key = repr(identity)
        if identity_key in seen:
            return {
                "resolved": False,
                "expression": expression,
                "reason": "cyclic_symbol_resolution",
            }
        seen = set(seen)
        seen.add(identity_key)

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
                "route": "unary_literal",
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
                    raise ValueError("unsupported operator")
            except Exception as exc:
                return {
                    "resolved": False,
                    "expression": expression,
                    "reason": f"binary_evaluation_failed:{exc!r}",
                    "left": left,
                    "right": right,
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
            if len(values) == 1:
                child = self.resolve(
                    values[0],
                    function_name=function_name,
                    seen=seen,
                )
                return {
                    **child,
                    "expression": expression,
                    "route": "unique_assignment",
                    "symbol": name,
                    "child": child,
                }

            return {
                "resolved": False,
                "expression": expression,
                "reason": (
                    "symbol_not_found"
                    if not values
                    else "multiple_symbol_assignments"
                ),
                "symbol": name,
                "assignment_count": len(values),
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

        if isinstance(node, (ast.Dict, ast.List, ast.Tuple)):
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
                    "container": container,
                    "index": index,
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
            name = call_name(node.func).rsplit(".", 1)[-1]
            if name in {"int", "float", "str", "bool"} and len(node.args) == 1:
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
                }[name]
                return {
                    "resolved": True,
                    "value": caster(child["value"]),
                    "expression": expression,
                    "route": f"{name}_cast",
                    "child": child,
                }

            if name in {"max", "min"} and node.args:
                children = [
                    self.resolve(
                        argument,
                        function_name=function_name,
                        seen=seen,
                    )
                    for argument in node.args
                ]
                if all(child["resolved"] for child in children):
                    function = max if name == "max" else min
                    return {
                        "resolved": True,
                        "value": function(
                            child["value"] for child in children
                        ),
                        "expression": expression,
                        "route": f"{name}_call",
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


def keyword_node(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def runtime_default(
    callable_object: Any,
    parameter_name: str,
) -> dict[str, Any]:
    signature = inspect.signature(callable_object)
    parameter = signature.parameters[parameter_name]
    require(
        parameter.default is not inspect._empty,
        f"runtime default missing for {parameter_name}",
    )
    return {
        "resolved": True,
        "value": parameter.default,
        "route": "runtime_signature_default",
        "signature": str(signature),
        "callable": (
            f"{callable_object.__module__}."
            f"{callable_object.__qualname__}"
        ),
        "torch_version": torch.__version__,
    }


def resolve_call_argument(
    *,
    resolver: StaticResolver,
    call: ast.Call,
    keyword: str,
    positional_index: int | None,
    runtime_callable: Any | None,
    function_name: str | None,
) -> dict[str, Any]:
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
        result = runtime_default(runtime_callable, keyword)
        return {
            **result,
            "argument_route": "runtime_default",
            "keyword": keyword,
        }

    return {
        "resolved": False,
        "argument_route": "absent",
        "keyword": keyword,
        "reason": "argument_absent_and_no_runtime_default",
    }


def classify_loader(call: ast.Call) -> str:
    dataset_expression = (
        ast.unparse(call.args[0])
        if call.args
        else ast.unparse(keyword_node(call, "dataset"))
        if keyword_node(call, "dataset") is not None
        else ""
    )
    token = normalize(dataset_expression)

    if any(term in token for term in ("validation", "valid", "val", "dev")):
        return "validation"
    if any(term in token for term in ("train", "training")):
        return "train"

    shuffle = keyword_node(call, "shuffle")
    if isinstance(shuffle, ast.Constant) and shuffle.value is True:
        return "train"
    if isinstance(shuffle, ast.Constant) and shuffle.value is False:
        return "validation_or_evaluation"

    return "unclassified"


def unique_numeric(
    rows: list[dict[str, Any]],
    field: str,
    *,
    integer: bool = False,
    positive: bool = False,
) -> tuple[Any | None, list[Any]]:
    values = []
    for row in rows:
        result = row[field]
        if not result.get("resolved"):
            continue
        value = result.get("value")
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        numeric = int(value) if integer and float(value).is_integer() else value
        if integer and not isinstance(numeric, int):
            continue
        if positive and numeric <= 0:
            continue
        values.append(numeric)

    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)

    return (unique[0] if len(unique) == 1 else None, unique)


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

    e0_report_path = f7_root / (
        "V5_P3_F7_E0_RECONSTRUCTED_EXECUTION_RECIPE_AND_SOURCE_MAP_REPORT.json"
    )
    e0_lock_path = f7_root / (
        "V5_P3_F7_E0_RECONSTRUCTED_EXECUTION_RECIPE_AND_SOURCE_MAP_LOCK.json"
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

    e0_report, e0_lock = verify_report_lock(e0_report_path, e0_lock_path)
    p2_report, p2_lock = verify_report_lock(p2_report_path, p2_lock_path)
    protocol = load_json(protocol_path)
    reconstructed_route = load_json(reconstructed_route_path)

    require(
        e0_lock.get("P2_execution_authorization_verified") is True,
        "E0 did not verify P2 authorization",
    )
    require(
        e0_lock.get("E1_trainer_generation_authorized") is False,
        "E1 already authorized before E0A",
    )
    require(
        p2_lock.get("actual_F7_retraining_authorized") is True,
        "P2 did not authorize F7",
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
        == e0_report["finding"]["skeleton_sha256"],
        "skeleton hash differs from E0",
    )

    source_text = skeleton_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    lines = source_text.splitlines()
    tree = ast.parse(source_text)
    resolver = StaticResolver(tree, skeleton_path)

    dataloader_rows = []
    optimizer_rows = []
    loss_rows = []
    epoch_rows = []
    early_stop_rows = []
    gradient_clip_rows = []
    scheduler_rows = []
    selection_rows = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            full_name = call_name(node.func)
            short = full_name.rsplit(".", 1)[-1]
            function_name = resolver.enclosing_function(node)

            if short == "DataLoader":
                loader_class = classify_loader(node)
                batch_size = resolve_call_argument(
                    resolver=resolver,
                    call=node,
                    keyword="batch_size",
                    positional_index=1,
                    runtime_callable=DataLoader,
                    function_name=function_name,
                )
                shuffle = resolve_call_argument(
                    resolver=resolver,
                    call=node,
                    keyword="shuffle",
                    positional_index=None,
                    runtime_callable=DataLoader,
                    function_name=function_name,
                )
                num_workers = resolve_call_argument(
                    resolver=resolver,
                    call=node,
                    keyword="num_workers",
                    positional_index=None,
                    runtime_callable=DataLoader,
                    function_name=function_name,
                )
                drop_last = resolve_call_argument(
                    resolver=resolver,
                    call=node,
                    keyword="drop_last",
                    positional_index=None,
                    runtime_callable=DataLoader,
                    function_name=function_name,
                )

                dataloader_rows.append({
                    "classification": loader_class,
                    "dataset_expression": (
                        ast.unparse(node.args[0])
                        if node.args
                        else (
                            ast.unparse(keyword_node(node, "dataset"))
                            if keyword_node(node, "dataset") is not None
                            else None
                        )
                    ),
                    "batch_size": batch_size,
                    "shuffle": shuffle,
                    "num_workers": num_workers,
                    "drop_last": drop_last,
                    "enclosing_function": function_name,
                    **source_record(skeleton_path, lines, node),
                })

            if short in OPTIMIZER_CLASSES:
                optimizer_class = OPTIMIZER_CLASSES[short]
                lr = resolve_call_argument(
                    resolver=resolver,
                    call=node,
                    keyword="lr",
                    positional_index=1,
                    runtime_callable=optimizer_class,
                    function_name=function_name,
                )
                weight_decay = resolve_call_argument(
                    resolver=resolver,
                    call=node,
                    keyword="weight_decay",
                    positional_index=None,
                    runtime_callable=optimizer_class,
                    function_name=function_name,
                )
                optimizer_rows.append({
                    "implementation": (
                        f"{optimizer_class.__module__}."
                        f"{optimizer_class.__qualname__}"
                    ),
                    "runtime_signature": str(
                        inspect.signature(optimizer_class)
                    ),
                    "torch_version": torch.__version__,
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "enclosing_function": function_name,
                    **source_record(skeleton_path, lines, node),
                })

            if short in LOSS_NAMES or "loss" in normalize(full_name):
                assignment_target = None
                parent = resolver.parent.get(node)
                if isinstance(parent, ast.Assign):
                    assignment_target = ", ".join(
                        ast.unparse(target)
                        for target in parent.targets
                    )
                elif isinstance(parent, ast.AnnAssign):
                    assignment_target = ast.unparse(parent.target)

                loss_rows.append({
                    "call_name": full_name,
                    "assignment_target": assignment_target,
                    "enclosing_function": function_name,
                    "arguments": [ast.unparse(argument) for argument in node.args],
                    "keywords": {
                        keyword.arg: ast.unparse(keyword.value)
                        for keyword in node.keywords
                        if keyword.arg is not None
                    },
                    **source_record(skeleton_path, lines, node),
                })

            token = normalize(full_name)
            if any(term in token for term in GRADIENT_CLIP_TOKENS):
                gradient_clip_rows.append(
                    source_record(skeleton_path, lines, node)
                )
            if any(term in token for term in SCHEDULER_TOKENS):
                scheduler_rows.append(
                    source_record(skeleton_path, lines, node)
                )

        if isinstance(node, (ast.For, ast.While)):
            source = ast.unparse(node)
            if "epoch" in source.lower():
                if isinstance(node, ast.For):
                    iterator = node.iter
                    resolved_stop = None
                    range_name = call_name(iterator.func) if isinstance(iterator, ast.Call) else ""
                    if isinstance(iterator, ast.Call) and range_name.rsplit(".", 1)[-1] == "range":
                        if len(iterator.args) == 1:
                            stop_node = iterator.args[0]
                        elif len(iterator.args) >= 2:
                            stop_node = iterator.args[1]
                        else:
                            stop_node = None
                        if stop_node is not None:
                            resolved_stop = resolver.resolve(
                                stop_node,
                                function_name=resolver.enclosing_function(node),
                            )
                    epoch_rows.append({
                        "loop_type": "for",
                        "range_stop_resolution": resolved_stop,
                        "enclosing_function": resolver.enclosing_function(node),
                        **source_record(skeleton_path, lines, node),
                    })
                else:
                    epoch_rows.append({
                        "loop_type": "while",
                        "condition": ast.unparse(node.test),
                        "enclosing_function": resolver.enclosing_function(node),
                        **source_record(skeleton_path, lines, node),
                    })

        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.If, ast.Compare, ast.AugAssign)):
            text = ast.unparse(node)
            token = normalize(text)
            if any(term in token for term in EARLY_STOP_TOKENS):
                early_stop_rows.append(
                    source_record(skeleton_path, lines, node)
                )
            if any(
                term in token
                for term in (
                    "selection_score",
                    "best_score",
                    "validation_score",
                    "val_score",
                )
            ):
                selection_rows.append(
                    source_record(skeleton_path, lines, node)
                )

    require(len(dataloader_rows) >= 2, "fewer than two DataLoader calls found")
    require(len(optimizer_rows) == 1, "optimizer constructor is not unique")
    require(loss_rows, "no loss route found")
    require(epoch_rows, "no epoch loop found")

    train_loader_rows = [
        row for row in dataloader_rows
        if row["classification"] == "train"
    ]
    validation_loader_rows = [
        row for row in dataloader_rows
        if row["classification"] in {
            "validation",
            "validation_or_evaluation",
        }
    ]

    train_batch_size, train_batch_values = unique_numeric(
        train_loader_rows,
        "batch_size",
        integer=True,
        positive=True,
    )
    validation_batch_size, validation_batch_values = unique_numeric(
        validation_loader_rows,
        "batch_size",
        integer=True,
        positive=True,
    )

    optimizer_lr, optimizer_lr_values = unique_numeric(
        optimizer_rows,
        "learning_rate",
        positive=True,
    )
    weight_decay, weight_decay_values = unique_numeric(
        optimizer_rows,
        "weight_decay",
        positive=False,
    )

    max_epoch_candidates = []
    for row in epoch_rows:
        resolution = row.get("range_stop_resolution")
        if (
            isinstance(resolution, dict)
            and resolution.get("resolved")
            and isinstance(resolution.get("value"), (int, float))
            and not isinstance(resolution.get("value"), bool)
        ):
            value = resolution["value"]
            if float(value).is_integer() and int(value) > 0:
                max_epoch_candidates.append(int(value))

    max_epoch_values = []
    for value in max_epoch_candidates:
        if value not in max_epoch_values:
            max_epoch_values.append(value)
    max_epochs = (
        max_epoch_values[0]
        if len(max_epoch_values) == 1
        else None
    )

    early_stopping_present = bool(early_stop_rows)
    gradient_clipping_present = bool(gradient_clip_rows)
    scheduler_present = bool(scheduler_rows)

    loss_call_names = sorted(
        {
            row["call_name"]
            for row in loss_rows
        }
    )

    unresolved = []
    if train_batch_size is None:
        unresolved.append({
            "field": "train_batch_size",
            "candidate_values": train_batch_values,
            "reason": "train DataLoader batch_size did not resolve uniquely",
        })
    if validation_batch_size is None:
        unresolved.append({
            "field": "validation_batch_size",
            "candidate_values": validation_batch_values,
            "reason": (
                "validation DataLoader batch_size did not resolve uniquely"
            ),
        })
    if optimizer_lr is None:
        unresolved.append({
            "field": "learning_rate",
            "candidate_values": optimizer_lr_values,
            "reason": "optimizer learning rate did not resolve uniquely",
        })
    if weight_decay is None:
        unresolved.append({
            "field": "weight_decay",
            "candidate_values": weight_decay_values,
            "reason": "optimizer weight decay did not resolve uniquely",
        })
    if max_epochs is None:
        unresolved.append({
            "field": "max_epochs",
            "candidate_values": max_epoch_values,
            "reason": "epoch-loop stop did not resolve uniquely",
        })
    if early_stopping_present:
        unresolved.append({
            "field": "early_stopping_policy",
            "candidate_values": [],
            "reason": (
                "early-stopping-related source exists and requires a dedicated "
                "semantic pin before trainer generation"
            ),
        })
    if len(loss_call_names) == 0:
        unresolved.append({
            "field": "loss_route",
            "candidate_values": [],
            "reason": "no loss calls found",
        })

    exact_pin_path = output_dir / (
        "F7_E0A_EXACT_HYPERPARAMETER_PIN.json"
    )
    dataloader_path = output_dir / (
        "F7_E0A_DATALOADER_ARGUMENT_RESOLUTION.json"
    )
    optimizer_path = output_dir / (
        "F7_E0A_OPTIMIZER_ARGUMENT_AND_RUNTIME_DEFAULT_RESOLUTION.json"
    )
    loss_path = output_dir / (
        "F7_E0A_LOSS_SELECTION_AND_TRAINING_CONTROL_FLOW_MAP.json"
    )
    decision_path = output_dir / (
        "F7_E0A_TRAINER_GENERATION_DECISION.json"
    )

    exact_pin = {
        "status": "FROZEN" if not unresolved else "PARTIALLY_RESOLVED",
        "route_type": "reconstructed_F7_recipe_from_exact_skeleton_behavior",
        "not_claimed": (
            "byte-identical historical A4 recipe reproduction"
        ),
        "primary_seed": EXPECTED_PRIMARY_SEED,
        "train_batch_size": train_batch_size,
        "validation_batch_size": validation_batch_size,
        "max_epochs": max_epochs,
        "optimizer": optimizer_rows[0]["implementation"],
        "learning_rate": optimizer_lr,
        "weight_decay": weight_decay,
        "weight_decay_resolution_route": optimizer_rows[0][
            "weight_decay"
        ]["route"],
        "loss_call_names": loss_call_names,
        "loss_call_count": len(loss_rows),
        "early_stopping_present": early_stopping_present,
        "gradient_clipping_present": gradient_clipping_present,
        "scheduler_present": scheduler_present,
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

    atomic_json(exact_pin_path, exact_pin)
    atomic_json(
        dataloader_path,
        {
            "DataLoader_runtime_signature": str(
                inspect.signature(DataLoader)
            ),
            "torch_version": torch.__version__,
            "rows": dataloader_rows,
            "train_batch_candidate_values": train_batch_values,
            "validation_batch_candidate_values": (
                validation_batch_values
            ),
        },
    )
    atomic_json(
        optimizer_path,
        {
            "optimizer_rows": optimizer_rows,
            "learning_rate_candidate_values": optimizer_lr_values,
            "weight_decay_candidate_values": weight_decay_values,
            "runtime_default_note": (
                "When weight_decay is omitted from torch.optim.AdamW, the "
                "runtime signature default is pinned rather than guessed."
            ),
        },
    )
    atomic_json(
        loss_path,
        {
            "loss_rows": loss_rows,
            "loss_call_names": loss_call_names,
            "epoch_rows": epoch_rows,
            "early_stopping_rows": early_stop_rows,
            "gradient_clipping_rows": gradient_clip_rows,
            "scheduler_rows": scheduler_rows,
            "selection_rows": selection_rows,
        },
    )

    E1_authorized = len(unresolved) == 0
    next_stage = (
        "V5_P3_F7_E1_RECONSTRUCTED_CONTROL_AND_GROUP_ABLATION_TRAINER_GENERATION"
        if E1_authorized
        else "V5_P3_F7_E0B_REMAINING_RECIPE_SEMANTIC_PIN"
    )

    decision = {
        "E0_complete": True,
        "exact_skeleton_reparse_complete": True,
        "train_batch_size_resolved": train_batch_size is not None,
        "validation_batch_size_resolved": (
            validation_batch_size is not None
        ),
        "learning_rate_resolved": optimizer_lr is not None,
        "weight_decay_resolved": weight_decay is not None,
        "max_epochs_resolved": max_epochs is not None,
        "loss_route_mapped": bool(loss_rows),
        "early_stopping_present": early_stopping_present,
        "gradient_clipping_present": gradient_clipping_present,
        "scheduler_present": scheduler_present,
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
            "Resolve E0's missing batch-size and weight-decay fields by "
            "re-parsing the exact frozen V5-P2 training skeleton; symbolically "
            "resolve argparse defaults, assignments, function defaults, "
            "DataLoader positional/keyword arguments and optimizer arguments; "
            "pin omitted runtime defaults directly from the installed PyTorch "
            "signatures; map loss, epoch, early-stopping, clipping, scheduler "
            "and selection source lines; and authorize E1 only when the full "
            "reconstructed execution recipe is unambiguous."
        ),
        "finding": {
            "skeleton": str(skeleton_path),
            "skeleton_sha256": sha256_file(skeleton_path),
            "train_batch_size": train_batch_size,
            "validation_batch_size": validation_batch_size,
            "max_epochs": max_epochs,
            "optimizer": optimizer_rows[0]["implementation"],
            "learning_rate": optimizer_lr,
            "weight_decay": weight_decay,
            "weight_decay_resolution_route": optimizer_rows[0][
                "weight_decay"
            ]["route"],
            "loss_call_names": loss_call_names,
            "loss_call_count": len(loss_rows),
            "early_stopping_present": early_stopping_present,
            "gradient_clipping_present": gradient_clipping_present,
            "scheduler_present": scheduler_present,
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
            "exact_hyperparameter_pin": str(exact_pin_path),
            "dataloader_resolution": str(dataloader_path),
            "optimizer_resolution": str(optimizer_path),
            "loss_and_control_flow_map": str(loss_path),
            "decision": str(decision_path),
        },
        "provenance": {
            "E0_report_sha256": sha256_file(e0_report_path),
            "E0_lock_sha256": sha256_file(e0_lock_path),
            "P2_report_sha256": sha256_file(p2_report_path),
            "P2_lock_sha256": sha256_file(p2_lock_path),
            "F7_protocol_sha256": sha256_file(protocol_path),
            "reconstructed_route_sha256": sha256_file(
                reconstructed_route_path
            ),
            "skeleton_sha256": sha256_file(skeleton_path),
            "installed_script_sha256": sha256_file(installed_script),
            "exact_pin_sha256": sha256_file(exact_pin_path),
            "dataloader_resolution_sha256": sha256_file(dataloader_path),
            "optimizer_resolution_sha256": sha256_file(optimizer_path),
            "loss_map_sha256": sha256_file(loss_path),
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
            "exact_pin_sha256": sha256_file(exact_pin_path),
            "dataloader_resolution_sha256": sha256_file(dataloader_path),
            "optimizer_resolution_sha256": sha256_file(optimizer_path),
            "loss_map_sha256": sha256_file(loss_path),
            "decision_sha256": sha256_file(decision_path),
            "E0A_complete": True,
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
    print(f"torch_version={torch.__version__}")
    print(f"dataloader_call_count={len(dataloader_rows)}")
    for index, row in enumerate(dataloader_rows, start=1):
        print(
            f"dataloader_{index}="
            f"classification={row['classification']}:"
            f"dataset={row['dataset_expression']}:"
            f"batch_size={row['batch_size']}:"
            f"shuffle={row['shuffle']}:"
            f"line={row['line_start']}"
        )
    print(f"train_batch_size={train_batch_size}")
    print(f"validation_batch_size={validation_batch_size}")
    print(f"max_epochs={max_epochs}")
    print(f"optimizer={optimizer_rows[0]['implementation']}")
    print(f"learning_rate={optimizer_lr}")
    print(f"weight_decay={weight_decay}")
    print(
        "weight_decay_resolution_route="
        f"{optimizer_rows[0]['weight_decay']['route']}"
    )
    print(f"loss_call_count={len(loss_rows)}")
    print(f"loss_call_names={loss_call_names}")
    print(
        "early_stopping_present="
        f"{str(early_stopping_present).lower()}"
    )
    print(
        "gradient_clipping_present="
        f"{str(gradient_clipping_present).lower()}"
    )
    print(f"scheduler_present={str(scheduler_present).lower()}")
    print(f"unresolved_execution_field_count={len(unresolved)}")
    for row in unresolved:
        print(
            f"unresolved_execution_field={row['field']}:"
            f"values={row['candidate_values']}:"
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
    print(f"exact_hyperparameter_pin={exact_pin_path}")
    print(f"dataloader_resolution={dataloader_path}")
    print(f"optimizer_resolution={optimizer_path}")
    print(f"loss_and_control_flow_map={loss_path}")
    print(f"decision={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
