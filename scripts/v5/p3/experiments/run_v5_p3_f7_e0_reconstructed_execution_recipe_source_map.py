from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


STAGE = "V5_P3_F7_E0_RECONSTRUCTED_EXECUTION_RECIPE_AND_SOURCE_MAP"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "TRAIN-DERIVED"

EXPECTED_MODEL_CLASS = "V6P0Dynamic70GraphConvCount4"
EXPECTED_PARAMETER_COUNT = 60553
EXPECTED_SEED = 107
EXPECTED_TRAIN_ITEMS = 110855
EXPECTED_VALIDATION_ITEMS = 13863

SKELETON_RELATIVE_PATH = Path(
    "scripts/v5/p2/train_v5_p2_b2_single_seed.py"
)

RECIPE_ALIASES = {
    "seed": (
        "seed",
        "random_seed",
    ),
    "batch_size": (
        "batch_size",
        "train_batch_size",
    ),
    "validation_batch_size": (
        "validation_batch_size",
        "val_batch_size",
        "eval_batch_size",
    ),
    "max_epochs": (
        "max_epochs",
        "epochs",
        "epoch_budget",
        "num_epochs",
    ),
    "learning_rate": (
        "learning_rate",
        "lr",
    ),
    "weight_decay": (
        "weight_decay",
        "wd",
    ),
    "early_stopping_patience": (
        "early_stopping_patience",
        "early_stop_patience",
        "patience",
    ),
    "gradient_clip_norm": (
        "gradient_clip_norm",
        "max_grad_norm",
        "clip_grad_norm",
    ),
    "num_workers": (
        "num_workers",
        "workers",
    ),
}

OPTIMIZER_NAMES = {
    "Adam",
    "AdamW",
    "SGD",
    "RMSprop",
    "Adagrad",
}

LOSS_CALL_NAMES = {
    "binary_cross_entropy_with_logits",
    "cross_entropy",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "focal_loss",
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


def literal_or_source(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return {
            "source": ast.unparse(node),
            "literal": False,
        }


def line_record(
    path: Path,
    source_lines: list[str],
    node: ast.AST,
) -> dict[str, Any]:
    start = int(getattr(node, "lineno", 0))
    end = int(getattr(node, "end_lineno", start))
    excerpt = "\n".join(source_lines[max(0, start - 1):end])
    return {
        "path": str(path.resolve()),
        "line_start": start,
        "line_end": end,
        "source": ast.unparse(node),
        "excerpt": excerpt,
    }


class SkeletonMapVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, source_lines: list[str]) -> None:
        self.path = path
        self.source_lines = source_lines

        self.imports = []
        self.functions = []
        self.classes = []
        self.argparse_options = []
        self.optimizer_calls = []
        self.dataloader_calls = []
        self.model_calls = []
        self.loss_calls = []
        self.backward_calls = []
        self.optimizer_step_calls = []
        self.zero_grad_calls = []
        self.checkpoint_calls = []
        self.epoch_loops = []
        self.selection_expressions = []
        self.split_literals = []
        self.assignments = []
        self.main_guards = []

    def visit_Import(self, node: ast.Import) -> Any:
        self.imports.append(line_record(self.path, self.source_lines, node))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        self.imports.append(line_record(self.path, self.source_lines, node))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.functions.append({
            "name": node.name,
            **line_record(self.path, self.source_lines, node),
        })
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.functions.append({
            "name": node.name,
            **line_record(self.path, self.source_lines, node),
        })
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.classes.append({
            "name": node.name,
            **line_record(self.path, self.source_lines, node),
        })
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        target_text = ", ".join(ast.unparse(target) for target in node.targets)
        self.assignments.append({
            "targets": target_text,
            "value": literal_or_source(node.value),
            **line_record(self.path, self.source_lines, node),
        })
        token = normalize(target_text)
        if any(
            term in token
            for term in (
                "selection",
                "score",
                "best_metric",
                "validation_metric",
            )
        ):
            self.selection_expressions.append(
                line_record(self.path, self.source_lines, node)
            )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        target_text = ast.unparse(node.target)
        value = (
            literal_or_source(node.value)
            if node.value is not None
            else None
        )
        self.assignments.append({
            "targets": target_text,
            "value": value,
            **line_record(self.path, self.source_lines, node),
        })
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        name = call_name(node.func)
        short = name.rsplit(".", 1)[-1]

        if short == "add_argument":
            option_strings = []
            for argument in node.args:
                if (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                ):
                    option_strings.append(argument.value)
            keywords = {
                keyword.arg: literal_or_source(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
            self.argparse_options.append({
                "options": option_strings,
                "keywords": keywords,
                **line_record(self.path, self.source_lines, node),
            })

        if short in OPTIMIZER_NAMES:
            self.optimizer_calls.append({
                "call_name": name,
                "args": [literal_or_source(argument) for argument in node.args],
                "keywords": {
                    keyword.arg: literal_or_source(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg is not None
                },
                **line_record(self.path, self.source_lines, node),
            })

        if short == "DataLoader":
            self.dataloader_calls.append({
                "call_name": name,
                "args": [literal_or_source(argument) for argument in node.args],
                "keywords": {
                    keyword.arg: literal_or_source(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg is not None
                },
                **line_record(self.path, self.source_lines, node),
            })

        if (
            "model" in normalize(name)
            or short.startswith("V5")
            or short.startswith("V6")
        ):
            self.model_calls.append({
                "call_name": name,
                "args": [literal_or_source(argument) for argument in node.args],
                "keywords": {
                    keyword.arg: literal_or_source(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg is not None
                },
                **line_record(self.path, self.source_lines, node),
            })

        if short in LOSS_CALL_NAMES or "loss" in normalize(name):
            self.loss_calls.append({
                "call_name": name,
                "args": [ast.unparse(argument) for argument in node.args],
                "keywords": {
                    keyword.arg: ast.unparse(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg is not None
                },
                **line_record(self.path, self.source_lines, node),
            })

        if short == "backward":
            self.backward_calls.append(
                line_record(self.path, self.source_lines, node)
            )
        elif short == "step":
            self.optimizer_step_calls.append(
                line_record(self.path, self.source_lines, node)
            )
        elif short == "zero_grad":
            self.zero_grad_calls.append(
                line_record(self.path, self.source_lines, node)
            )
        elif short in ("save", "save_checkpoint"):
            self.checkpoint_calls.append(
                line_record(self.path, self.source_lines, node)
            )

        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        text = ast.unparse(node.target) + " " + ast.unparse(node.iter)
        if "epoch" in text.lower():
            self.epoch_loops.append(
                line_record(self.path, self.source_lines, node)
            )
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> Any:
        if "epoch" in ast.unparse(node.test).lower():
            self.epoch_loops.append(
                line_record(self.path, self.source_lines, node)
            )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, str):
            token = normalize(node.value)
            if token in {
                "train",
                "training",
                "validation",
                "valid",
                "val",
                "dev",
            }:
                self.split_literals.append({
                    "value": node.value,
                    **line_record(self.path, self.source_lines, node),
                })
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> Any:
        test = ast.unparse(node.test)
        if "__name__" in test and "__main__" in test:
            self.main_guards.append(
                line_record(self.path, self.source_lines, node)
            )
        self.generic_visit(node)


def map_skeleton(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    source_lines = text.splitlines()
    tree = ast.parse(text)
    visitor = SkeletonMapVisitor(path, source_lines)
    visitor.visit(tree)

    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "line_count": len(source_lines),
        "imports": visitor.imports,
        "functions": visitor.functions,
        "classes": visitor.classes,
        "argparse_options": visitor.argparse_options,
        "optimizer_calls": visitor.optimizer_calls,
        "dataloader_calls": visitor.dataloader_calls,
        "model_calls": visitor.model_calls,
        "loss_calls": visitor.loss_calls,
        "backward_calls": visitor.backward_calls,
        "optimizer_step_calls": visitor.optimizer_step_calls,
        "zero_grad_calls": visitor.zero_grad_calls,
        "checkpoint_calls": visitor.checkpoint_calls,
        "epoch_loops": visitor.epoch_loops,
        "selection_expressions": visitor.selection_expressions,
        "split_literals": visitor.split_literals,
        "assignments": visitor.assignments,
        "main_guards": visitor.main_guards,
    }


def recursive_scalar_candidates(
    value: Any,
    aliases: tuple[str, ...],
    path: str = "root",
) -> list[dict[str, Any]]:
    rows = []
    alias_tokens = tuple(normalize(alias) for alias in aliases)

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            key_token = normalize(str(key))
            if any(alias in key_token for alias in alias_tokens):
                if isinstance(child, (str, int, float, bool)) or child is None:
                    rows.append({
                        "path": child_path,
                        "value": child,
                    })
            rows.extend(
                recursive_scalar_candidates(
                    child,
                    aliases,
                    child_path,
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(
                recursive_scalar_candidates(
                    child,
                    aliases,
                    f"{path}[{index}]",
                )
            )

    return rows


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    require(isinstance(checkpoint, dict), "checkpoint is not a dictionary")

    metadata = {}
    tensor_key_count = 0
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            tensor_key_count += 1
            continue
        if isinstance(value, dict):
            compact = {}
            for child_key, child in value.items():
                if isinstance(child, torch.Tensor):
                    continue
                if isinstance(child, (str, int, float, bool)) or child is None:
                    compact[str(child_key)] = child
            if compact:
                metadata[str(key)] = compact
        elif isinstance(value, (str, int, float, bool)) or value is None:
            metadata[str(key)] = value

    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "top_level_keys": sorted(str(key) for key in checkpoint),
        "top_level_tensor_key_count": tensor_key_count,
        "non_tensor_metadata": metadata,
    }


def argparse_defaults(skeleton_map: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for row in skeleton_map["argparse_options"]:
        defaults = row["keywords"]
        if "default" not in defaults:
            continue
        for option in row["options"]:
            output[normalize(option.lstrip("-"))] = {
                "option": option,
                "default": defaults["default"],
                "line_start": row["line_start"],
                "source": row["source"],
            }
    return output


def assignment_candidates(
    skeleton_map: dict[str, Any],
    aliases: tuple[str, ...],
) -> list[dict[str, Any]]:
    alias_tokens = tuple(normalize(alias) for alias in aliases)
    rows = []
    for row in skeleton_map["assignments"]:
        token = normalize(row["targets"])
        if any(alias in token for alias in alias_tokens):
            rows.append(row)
    return rows


def deduplicate_scalar_candidates(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for row in rows:
        value = row.get("value")
        if isinstance(value, dict):
            key = json.dumps(value, sort_keys=True)
        else:
            key = repr(value)
        identity = (row.get("source", ""), row.get("path", ""), key)
        if identity in seen:
            continue
        seen.add(identity)
        output.append(row)
    return output


def recipe_review(
    *,
    skeleton_map: dict[str, Any],
    lineage_documents: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    defaults = argparse_defaults(skeleton_map)
    result = {}

    combined = {
        "lineage": lineage_documents,
        "checkpoint_metadata": checkpoint["non_tensor_metadata"],
    }

    for field, aliases in RECIPE_ALIASES.items():
        rows = []

        for alias in aliases:
            token = normalize(alias)
            if token in defaults:
                rows.append({
                    "source": "skeleton_argparse_default",
                    "path": f"argparse.{token}",
                    "value": defaults[token]["default"],
                    "line_start": defaults[token]["line_start"],
                    "expression": defaults[token]["source"],
                })

        for row in assignment_candidates(skeleton_map, aliases):
            rows.append({
                "source": "skeleton_assignment",
                "path": (
                    f"line_{row['line_start']}."
                    f"{row['targets']}"
                ),
                "value": row["value"],
                "expression": row["source"],
            })

        for row in recursive_scalar_candidates(combined, aliases):
            rows.append({
                "source": "frozen_lineage_or_checkpoint",
                **row,
            })

        result[field] = deduplicate_scalar_candidates(rows)

    return result


def count_unique_literal_values(rows: list[dict[str, Any]]) -> list[Any]:
    values = []
    seen = set()

    for row in rows:
        value = row.get("value")
        if isinstance(value, dict):
            continue
        if isinstance(value, bool) or value is None:
            key = (type(value).__name__, repr(value))
        elif isinstance(value, (int, float)):
            numeric = float(value)
            key = ("numeric", numeric)
        elif isinstance(value, str):
            stripped = value.strip()
            try:
                numeric = float(stripped)
                key = ("numeric", numeric)
                value = numeric
            except ValueError:
                key = ("string", normalize(stripped))
                value = stripped
        else:
            continue

        if key in seen:
            continue
        seen.add(key)
        values.append(value)

    return values


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
    f6r_root = feature_root / "integrated_gradients_review"
    baseline_root = feature_root / "baseline_reproduction"
    metric_root = feature_root / "metric_adapter"

    p2_report_path = f7_root / (
        "V5_P3_F7_P2_RECONSTRUCTED_TRAINER_ADAPTER_AND_DRY_RUN_PREFLIGHT_REPORT.json"
    )
    p2_lock_path = f7_root / (
        "V5_P3_F7_P2_RECONSTRUCTED_TRAINER_ADAPTER_AND_DRY_RUN_PREFLIGHT_LOCK.json"
    )
    p1b_route_path = f7_root / (
        "F7_P1B_FROZEN_RECONSTRUCTED_TRAINING_ROUTE.json"
    )
    p1b_recipe_path = f7_root / (
        "F7_P1B_ARTIFACT_ANCHORED_TRAINING_RECIPE_EVIDENCE.json"
    )
    p0_protocol_path = f7_root / (
        "F7_P0_FROZEN_SAME_WIDTH_ABLATION_PROTOCOL.json"
    )
    f4_report_path = baseline_root / (
        "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_BASELINE_REPRODUCTION_REPORT.json"
    )
    f4m_report_path = metric_root / (
        "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION_FINAL_REPORT.json"
    )
    f6r_report_path = f6r_root / (
        "V5_P3_F6R_TASK_SPECIFIC_INTEGRATED_GRADIENTS_RESULT_REVIEW_REPORT.json"
    )

    p2_report, p2_lock = verify_report_lock(
        p2_report_path,
        p2_lock_path,
    )
    require(
        p2_lock.get("actual_F7_retraining_authorized") is True,
        "P2 did not authorize F7 execution",
    )
    require(
        p2_lock.get("all_dry_run_gates_pass") is True,
        "P2 dry-run gates did not pass",
    )
    require(
        p2_lock.get("feature_removal_authorized") is False,
        "feature removal already authorized",
    )
    require(
        p2_lock.get("sealed_test_tensors_loaded") is False,
        "sealed-test access detected",
    )

    reconstructed_route = load_json(p1b_route_path)
    p1b_recipe = load_json(p1b_recipe_path)
    protocol = load_json(p0_protocol_path)
    f4_report = load_json(f4_report_path)
    f4m_report = load_json(f4m_report_path)
    f6r_report = load_json(f6r_report_path)

    require(reconstructed_route["status"] == "FROZEN", "P1B route not frozen")
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
        == reconstructed_route["components"]["training_loop_skeleton"][
            "sha256"
        ],
        "training skeleton hash changed",
    )

    skeleton_map = map_skeleton(skeleton_path)
    require(
        len(skeleton_map["optimizer_calls"]) >= 1,
        "no optimizer construction found in skeleton",
    )
    require(
        len(skeleton_map["dataloader_calls"]) >= 1,
        "no DataLoader construction found in skeleton",
    )
    require(
        len(skeleton_map["backward_calls"]) >= 1,
        "no backward call found in skeleton",
    )
    require(
        len(skeleton_map["optimizer_step_calls"]) >= 1,
        "no optimizer step found in skeleton",
    )
    require(
        len(skeleton_map["zero_grad_calls"]) >= 1,
        "no zero_grad call found in skeleton",
    )
    require(
        len(skeleton_map["checkpoint_calls"]) >= 1,
        "no checkpoint save call found in skeleton",
    )
    require(
        len(skeleton_map["epoch_loops"]) >= 1,
        "no epoch loop found in skeleton",
    )

    checkpoint_path = Path(
        reconstructed_route["components"]["checkpoint"]["path"]
    ).resolve()
    require(checkpoint_path.is_file(), f"checkpoint missing: {checkpoint_path}")
    require(
        sha256_file(checkpoint_path)
        == reconstructed_route["components"]["checkpoint"]["sha256"],
        "checkpoint hash changed",
    )
    checkpoint_review = checkpoint_metadata(checkpoint_path)

    lineage = {
        "P1B_recipe_evidence": p1b_recipe,
        "P1B_route": reconstructed_route,
        "F7_protocol": protocol,
        "F4_report": f4_report,
        "F4M_report": f4m_report,
        "F6R_report": f6r_report,
    }
    recipe_candidates = recipe_review(
        skeleton_map=skeleton_map,
        lineage_documents=lineage,
        checkpoint=checkpoint_review,
    )

    recipe_value_summary = {
        field: count_unique_literal_values(rows)
        for field, rows in recipe_candidates.items()
    }

    seed_values = recipe_value_summary["seed"]
    require(
        any(
            (
                isinstance(value, (int, float))
                and float(value) == EXPECTED_SEED
            )
            or str(value).strip() == str(EXPECTED_SEED)
            for value in seed_values
        ),
        "seed-107 evidence missing from execution recipe",
    )

    static_source_map_path = output_dir / (
        "F7_E0_TRAINING_SKELETON_STATIC_SOURCE_MAP.json"
    )
    recipe_candidates_path = output_dir / (
        "F7_E0_EXECUTION_RECIPE_CANDIDATE_EVIDENCE.json"
    )
    checkpoint_metadata_path = output_dir / (
        "F7_E0_CHECKPOINT_NON_TENSOR_METADATA_REVIEW.json"
    )
    adapter_plan_path = output_dir / (
        "F7_E0_RECONSTRUCTED_TRAINER_ADAPTER_PLAN.json"
    )
    decision_path = output_dir / (
        "F7_E0_EXECUTION_SOURCE_FREEZE_DECISION.json"
    )

    atomic_json(static_source_map_path, skeleton_map)
    atomic_json(
        recipe_candidates_path,
        {
            "candidate_evidence": recipe_candidates,
            "unique_literal_values": recipe_value_summary,
        },
    )
    atomic_json(checkpoint_metadata_path, checkpoint_review)

    adapter_plan = {
        "status": "FROZEN",
        "base_training_loop_skeleton": {
            "path": str(skeleton_path),
            "sha256": sha256_file(skeleton_path),
        },
        "certified_replacements": {
            "model": reconstructed_route["components"]["model"],
            "loader": reconstructed_route["components"]["loader"],
            "metric_adapter": reconstructed_route["components"]["adapter"],
            "checkpoint_reference_only": reconstructed_route["components"][
                "checkpoint"
            ],
        },
        "fresh_initialization_policy": p2_report["finding"]["model"][
            "fresh_reset"
        ]["policy"],
        "mask_adapter": protocol["masking_protocol"],
        "run_matrix": protocol["run_matrix"],
        "primary_seed": EXPECTED_SEED,
        "training_loop_source_sections": {
            "optimizer_calls": skeleton_map["optimizer_calls"],
            "dataloader_calls": skeleton_map["dataloader_calls"],
            "loss_calls": skeleton_map["loss_calls"],
            "backward_calls": skeleton_map["backward_calls"],
            "optimizer_step_calls": skeleton_map[
                "optimizer_step_calls"
            ],
            "zero_grad_calls": skeleton_map["zero_grad_calls"],
            "checkpoint_calls": skeleton_map["checkpoint_calls"],
            "epoch_loops": skeleton_map["epoch_loops"],
            "selection_expressions": skeleton_map[
                "selection_expressions"
            ],
        },
        "required_E1_behavior": [
            "generate an isolated F7 trainer rather than editing the frozen skeleton",
            "use the certified V6P0 model and guarded loader",
            "use the canonical F4M metric adapter",
            "preserve the P2 deterministic fresh-reset route",
            "implement the six seed-107 runs in append-only directories",
            "make every epoch and run restart-safe",
            "save train/validation metrics only",
            "never construct sealed-test or A-test datasets",
            "compare all five ablations with the freshly retrained control",
        ],
    }
    atomic_json(adapter_plan_path, adapter_plan)

    exact_recipe_fields = {
        "seed": EXPECTED_SEED,
        "train_items": EXPECTED_TRAIN_ITEMS,
        "validation_items": EXPECTED_VALIDATION_ITEMS,
        "model_class": EXPECTED_MODEL_CLASS,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "selection_formula": protocol["metric_protocol"][
            "primary_selection_formula"
        ],
        "reporting_threshold": protocol["metric_protocol"][
            "threshold_policy"
        ]["graph_reporting_threshold"],
        "threshold_tuning_per_run": protocol["metric_protocol"][
            "threshold_policy"
        ]["threshold_tuning_per_run"],
        "candidate_reduction_gates": protocol["metric_protocol"][
            "candidate_reduction_gates"
        ],
    }

    unresolved_execution_fields = []
    for field in (
        "batch_size",
        "max_epochs",
        "learning_rate",
        "weight_decay",
    ):
        values = recipe_value_summary[field]
        if len(values) != 1:
            unresolved_execution_fields.append({
                "field": field,
                "unique_literal_values": values,
                "reason": (
                    "zero or multiple literal candidates remain after "
                    "skeleton, A4/F4/F4M/P1B and checkpoint review"
                ),
            })

    optimizer_implementations = sorted(
        {
            row["call_name"]
            for row in skeleton_map["optimizer_calls"]
        }
    )
    if len(optimizer_implementations) != 1:
        unresolved_execution_fields.append({
            "field": "optimizer_implementation",
            "unique_literal_values": optimizer_implementations,
            "reason": "optimizer constructor is not unique",
        })

    loss_route_present = len(skeleton_map["loss_calls"]) > 0
    if not loss_route_present:
        unresolved_execution_fields.append({
            "field": "loss_implementation",
            "unique_literal_values": [],
            "reason": "no loss call was statically resolved in the skeleton",
        })

    E1_trainer_generation_authorized = (
        len(unresolved_execution_fields) == 0
    )
    next_stage = (
        "V5_P3_F7_E1_RECONSTRUCTED_CONTROL_AND_GROUP_ABLATION_TRAINER_GENERATION"
        if E1_trainer_generation_authorized
        else "V5_P3_F7_E0A_EXACT_HYPERPARAMETER_AND_LOSS_ROUTE_PIN"
    )

    decision = {
        "P2_execution_authorization_verified": True,
        "training_skeleton_source_mapped": True,
        "checkpoint_metadata_reviewed": True,
        "adapter_plan_frozen": True,
        "exact_recipe_fields": exact_recipe_fields,
        "optimizer_implementations": optimizer_implementations,
        "loss_call_count": len(skeleton_map["loss_calls"]),
        "unresolved_execution_fields": unresolved_execution_fields,
        "E1_trainer_generation_authorized": (
            E1_trainer_generation_authorized
        ),
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
            "Materialize the authorized F7 execution route without starting "
            "scientific training: verify P2 authorization and the P1B "
            "reconstructed route; statically map the frozen V5-P2 training "
            "skeleton by exact source lines; inventory optimizer, DataLoader, "
            "loss, backward, step, checkpoint and epoch-loop routes; review "
            "non-tensor checkpoint metadata and frozen A4/F4/F4M/P1B evidence "
            "for exact hyperparameters; freeze the isolated F7 adapter plan; "
            "and authorize trainer generation only when all execution-critical "
            "fields are uniquely pinned."
        ),
        "finding": {
            "skeleton_path": str(skeleton_path),
            "skeleton_sha256": sha256_file(skeleton_path),
            "optimizer_call_count": len(
                skeleton_map["optimizer_calls"]
            ),
            "dataloader_call_count": len(
                skeleton_map["dataloader_calls"]
            ),
            "loss_call_count": len(skeleton_map["loss_calls"]),
            "backward_call_count": len(
                skeleton_map["backward_calls"]
            ),
            "optimizer_step_call_count": len(
                skeleton_map["optimizer_step_calls"]
            ),
            "checkpoint_call_count": len(
                skeleton_map["checkpoint_calls"]
            ),
            "epoch_loop_count": len(skeleton_map["epoch_loops"]),
            "recipe_unique_literal_values": recipe_value_summary,
            "unresolved_execution_fields": unresolved_execution_fields,
        },
        "decision": decision,
        "governance": {
            "model_loaded": False,
            "checkpoint_tensors_loaded": True,
            "checkpoint_non_tensor_metadata_retained_only": True,
            "training_dataset_constructed": False,
            "validation_dataset_constructed": False,
            "training_feature_tensors_loaded": False,
            "validation_feature_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "optimizer_step_performed": False,
            "scientific_checkpoint_saved": False,
            "scientific_training_started": False,
        },
        "artifacts": {
            "static_source_map": str(static_source_map_path),
            "recipe_candidate_evidence": str(recipe_candidates_path),
            "checkpoint_metadata_review": str(checkpoint_metadata_path),
            "adapter_plan": str(adapter_plan_path),
            "decision": str(decision_path),
        },
        "provenance": {
            "P2_report_sha256": sha256_file(p2_report_path),
            "P2_lock_sha256": sha256_file(p2_lock_path),
            "P1B_route_sha256": sha256_file(p1b_route_path),
            "P1B_recipe_sha256": sha256_file(p1b_recipe_path),
            "F7_protocol_sha256": sha256_file(p0_protocol_path),
            "F4_report_sha256": sha256_file(f4_report_path),
            "F4M_report_sha256": sha256_file(f4m_report_path),
            "F6R_report_sha256": sha256_file(f6r_report_path),
            "skeleton_sha256": sha256_file(skeleton_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "installed_script_sha256": sha256_file(installed_script),
            "static_source_map_sha256": sha256_file(
                static_source_map_path
            ),
            "recipe_candidate_evidence_sha256": sha256_file(
                recipe_candidates_path
            ),
            "checkpoint_metadata_review_sha256": sha256_file(
                checkpoint_metadata_path
            ),
            "adapter_plan_sha256": sha256_file(adapter_plan_path),
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
            "static_source_map_sha256": sha256_file(
                static_source_map_path
            ),
            "recipe_candidate_evidence_sha256": sha256_file(
                recipe_candidates_path
            ),
            "checkpoint_metadata_review_sha256": sha256_file(
                checkpoint_metadata_path
            ),
            "adapter_plan_sha256": sha256_file(adapter_plan_path),
            "decision_sha256": sha256_file(decision_path),
            "P2_execution_authorization_verified": True,
            "adapter_plan_frozen": True,
            "E1_trainer_generation_authorized": (
                E1_trainer_generation_authorized
            ),
            "actual_scientific_training_started": False,
            "F8_multi_seed_authorized": False,
            "feature_removal_authorized": False,
            "compact_interface_equivalence_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "model_loaded": False,
            "training_feature_tensors_loaded": False,
            "validation_feature_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print("P2_execution_authorization_verified=true")
    print(f"training_skeleton={skeleton_path}")
    print(f"training_skeleton_sha256={sha256_file(skeleton_path)}")
    print(
        "optimizer_implementations="
        f"{optimizer_implementations}"
    )
    print(f"optimizer_call_count={len(skeleton_map['optimizer_calls'])}")
    print(f"dataloader_call_count={len(skeleton_map['dataloader_calls'])}")
    print(f"loss_call_count={len(skeleton_map['loss_calls'])}")
    print(f"backward_call_count={len(skeleton_map['backward_calls'])}")
    print(
        "optimizer_step_call_count="
        f"{len(skeleton_map['optimizer_step_calls'])}"
    )
    print(
        "checkpoint_call_count="
        f"{len(skeleton_map['checkpoint_calls'])}"
    )
    print(f"epoch_loop_count={len(skeleton_map['epoch_loops'])}")
    for field, values in recipe_value_summary.items():
        print(f"recipe_candidates_{field}={values}")
    print(
        "unresolved_execution_field_count="
        f"{len(unresolved_execution_fields)}"
    )
    for row in unresolved_execution_fields:
        print(
            f"unresolved_execution_field={row['field']}:"
            f"values={row['unique_literal_values']}:"
            f"reason={row['reason']}"
        )
    print("adapter_plan_frozen=true")
    print(
        "E1_trainer_generation_authorized="
        f"{str(E1_trainer_generation_authorized).lower()}"
    )
    print("actual_scientific_training_started=false")
    print("F8_multi_seed_authorized=false")
    print("feature_removal_authorized=false")
    print("compact_interface_equivalence_authorized=false")
    print("hardware_reduction_claim_authorized=false")
    print("model_loaded=false")
    print("training_feature_tensors_loaded=false")
    print("validation_feature_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={next_stage}")
    print(f"static_source_map={static_source_map_path}")
    print(f"recipe_candidate_evidence={recipe_candidates_path}")
    print(f"checkpoint_metadata_review={checkpoint_metadata_path}")
    print(f"adapter_plan={adapter_plan_path}")
    print(f"decision={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
