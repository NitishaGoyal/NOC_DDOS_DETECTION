from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_TRAIN_ITEMS = 110855
EXPECTED_VALIDATION_ITEMS = 13863
EXPECTED_MODEL_CLASS = "V6P0Dynamic70GraphConvCount4"
EXPECTED_PARAMETER_COUNT = 60553

STAGE = "V5_P3_F7_P1_OFFICIAL_TRAINING_ROUTE_RECOVERY"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "TRAIN-DERIVED"

REJECTED_PATH_TERMS = (
    "preflight",
    "review",
    "permutation",
    "integrated_gradient",
    "integrated-gradients",
    "diagnostic",
    "recovery",
    "audit",
    "metric_adapter",
    "evaluation",
    "export",
    "feature_study",
    "feature-study",
    "f6r",
    "f7_p0",
    "f7-p0",
)

TEXT_SUFFIXES = {
    ".json",
    ".txt",
    ".log",
    ".md",
    ".yaml",
    ".yml",
    ".sh",
    ".csv",
}

PYTHON_PATH_PATTERN = re.compile(
    r"""(?P<path>(?:/|\.{0,2}/)?[A-Za-z0-9_./-]+\.py)"""
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


def import_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def path_rejected(path: Path) -> tuple[bool, list[str]]:
    normalized = normalize(str(path))
    hits = [term for term in REJECTED_PATH_TERMS if normalize(term) in normalized]
    return bool(hits), hits


def resolve_reference(
    raw: str,
    *,
    repo: Path,
    evidence_file: Path,
) -> list[Path]:
    raw = raw.strip().strip("'\"`),:;[]{}")
    candidate = Path(raw).expanduser()
    attempts = []

    if candidate.is_absolute():
        attempts.append(candidate)
    else:
        attempts.extend([
            evidence_file.parent / candidate,
            repo / candidate,
            repo / "scripts" / candidate,
            repo / "scripts/v5/p3" / candidate,
            repo / "scripts/v5/p3/experiments" / candidate,
        ])

    resolved = []
    seen = set()
    for path in attempts:
        try:
            path = path.resolve()
        except Exception:
            continue
        if path in seen:
            continue
        seen.add(path)
        if path.is_file() and path.suffix.lower() == ".py":
            resolved.append(path)
    return resolved


def mine_a4_references(repo: Path) -> dict[str, Any]:
    a4_root = (
        repo
        / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107"
    )
    require(a4_root.is_dir(), f"A4 report directory missing: {a4_root}")

    evidence_files = []
    resolved_references: dict[str, dict[str, Any]] = {}

    for path in sorted(a4_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.stat().st_size > 64 * 1024 * 1024:
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        matches = []
        for match in PYTHON_PATH_PATTERN.finditer(text):
            raw = match.group("path")
            paths = resolve_reference(raw, repo=repo, evidence_file=path)
            matches.append({
                "raw": raw,
                "resolved": [str(candidate) for candidate in paths],
            })
            for candidate in paths:
                key = str(candidate)
                row = resolved_references.setdefault(
                    key,
                    {
                        "path": key,
                        "sha256": sha256_file(candidate),
                        "referenced_by": [],
                    },
                )
                row["referenced_by"].append(str(path.resolve()))

        command_lines = [
            {
                "line": index,
                "text": line[:2000],
            }
            for index, line in enumerate(text.splitlines(), start=1)
            if any(
                token in line.lower()
                for token in (
                    "python ",
                    "python3 ",
                    "trainer",
                    "train",
                    "epoch",
                    "optimizer",
                    "checkpoint",
                    "seed",
                )
            )
        ][:500]

        evidence_files.append({
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "python_path_matches": matches,
            "relevant_lines": command_lines,
        })

    return {
        "A4_root": str(a4_root.resolve()),
        "evidence_file_count": len(evidence_files),
        "evidence_files": evidence_files,
        "resolved_python_references": sorted(
            resolved_references.values(),
            key=lambda row: row["path"],
        ),
    }


class TrainingASTVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.model_class_reference = False
        self.dataloader_reference = False
        self.backward_calls = 0
        self.step_calls = 0
        self.zero_grad_calls = 0
        self.torch_save_calls = 0
        self.train_split_literals = 0
        self.validation_split_literals = 0
        self.epoch_loops = 0
        self.loss_names = set()
        self.optimizer_names = set()
        self.functions = []
        self.classes = []

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id == EXPECTED_MODEL_CLASS:
            self.model_class_reference = True
        if node.id == "DataLoader":
            self.dataloader_reference = True
        lowered = node.id.lower()
        if "loss" in lowered:
            self.loss_names.add(node.id)
        if "optim" in lowered:
            self.optimizer_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if node.attr == EXPECTED_MODEL_CLASS:
            self.model_class_reference = True
        if node.attr == "DataLoader":
            self.dataloader_reference = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr == "backward":
                self.backward_calls += 1
            elif func.attr == "step":
                self.step_calls += 1
            elif func.attr == "zero_grad":
                self.zero_grad_calls += 1
            elif (
                func.attr == "save"
                and isinstance(func.value, ast.Name)
                and func.value.id == "torch"
            ):
                self.torch_save_calls += 1
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, str):
            token = normalize(node.value)
            if token in {"train", "training"}:
                self.train_split_literals += 1
            if token in {"validation", "valid", "val", "dev"}:
                self.validation_split_literals += 1
            if EXPECTED_MODEL_CLASS in node.value:
                self.model_class_reference = True
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        text = ast.unparse(node.target).lower() + " " + ast.unparse(node.iter).lower()
        if "epoch" in text:
            self.epoch_loops += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.functions.append(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.functions.append(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.classes.append(node.name)
        self.generic_visit(node)


def inspect_python(path: Path, a4_reference_count: int) -> dict[str, Any]:
    rejected, rejection_terms = path_rejected(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
        syntax_ok = True
        syntax_error = None
    except SyntaxError as exc:
        tree = None
        syntax_ok = False
        syntax_error = repr(exc)

    visitor = TrainingASTVisitor()
    if tree is not None:
        visitor.visit(tree)

    exact_training_loop = bool(
        visitor.model_class_reference
        and visitor.dataloader_reference
        and visitor.backward_calls > 0
        and visitor.step_calls > 0
        and visitor.zero_grad_calls > 0
        and visitor.torch_save_calls > 0
        and visitor.train_split_literals > 0
        and visitor.validation_split_literals > 0
        and visitor.epoch_loops > 0
    )

    score = 0
    score += 25 if visitor.model_class_reference else 0
    score += 15 if visitor.dataloader_reference else 0
    score += min(visitor.backward_calls, 3) * 10
    score += min(visitor.step_calls, 3) * 8
    score += min(visitor.zero_grad_calls, 3) * 4
    score += min(visitor.torch_save_calls, 3) * 5
    score += 5 if visitor.train_split_literals else 0
    score += 5 if visitor.validation_split_literals else 0
    score += 5 if visitor.epoch_loops else 0
    score += min(a4_reference_count, 5) * 20
    score -= 1000 if rejected else 0

    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "syntax_ok": syntax_ok,
        "syntax_error": syntax_error,
        "rejected": rejected,
        "rejection_terms": rejection_terms,
        "A4_reference_count": a4_reference_count,
        "model_class_reference": visitor.model_class_reference,
        "dataloader_reference": visitor.dataloader_reference,
        "backward_call_count": visitor.backward_calls,
        "optimizer_step_call_count": visitor.step_calls,
        "zero_grad_call_count": visitor.zero_grad_calls,
        "torch_save_call_count": visitor.torch_save_calls,
        "train_split_literal_count": visitor.train_split_literals,
        "validation_split_literal_count": visitor.validation_split_literals,
        "epoch_loop_count": visitor.epoch_loops,
        "loss_names": sorted(visitor.loss_names),
        "optimizer_names": sorted(visitor.optimizer_names),
        "functions": sorted(set(visitor.functions)),
        "classes": sorted(set(visitor.classes)),
        "exact_training_loop": exact_training_loop,
        "score": score,
    }


def repository_candidates(
    repo: Path,
    a4_references: dict[str, Any],
) -> list[dict[str, Any]]:
    reference_counts = {
        row["path"]: len(set(row["referenced_by"]))
        for row in a4_references["resolved_python_references"]
    }

    roots = [
        repo / "scripts/v5/p3",
        repo / "scripts/v6",
        repo / "src",
        repo / "scripts",
    ]
    seen = set()
    rows = []

    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            try:
                row = inspect_python(
                    resolved,
                    reference_counts.get(str(resolved), 0),
                )
            except Exception as exc:
                rows.append({
                    "path": str(resolved),
                    "inspection_error": repr(exc),
                    "exact_training_loop": False,
                    "rejected": False,
                    "score": -9999,
                })
                continue

            if (
                row["model_class_reference"]
                or row["A4_reference_count"] > 0
                or row["backward_call_count"] > 0
            ):
                rows.append(row)

    rows.sort(key=lambda row: (-row.get("score", -9999), row["path"]))
    return rows


def git_history_candidates(repo: Path) -> dict[str, Any]:
    if not (repo / ".git").exists():
        return {
            "git_repository": False,
            "candidate_paths": [],
            "error": None,
        }

    try:
        process = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "log",
                "--all",
                "--name-only",
                "--pretty=format:",
                "--",
                "*.py",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        return {
            "git_repository": True,
            "candidate_paths": [],
            "error": repr(exc),
        }

    names = sorted(
        {
            line.strip()
            for line in process.stdout.splitlines()
            if line.strip().endswith(".py")
            and any(
                token in normalize(line)
                for token in ("a4", "train", "dynamic70", "task_d")
            )
        }
    )

    existing = []
    deleted = []
    for name in names:
        path = (repo / name).resolve()
        if path.is_file():
            existing.append(str(path))
        else:
            deleted.append(name)

    return {
        "git_repository": True,
        "candidate_paths": names,
        "existing_candidate_paths": existing,
        "deleted_historical_candidate_paths": deleted,
        "error": None,
    }


def select_trainer(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    strict = [
        row
        for row in candidates
        if row.get("exact_training_loop") is True
        and row.get("rejected") is False
        and row.get("syntax_ok") is True
    ]

    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strict:
        by_sha[row["sha256"]].append(row)

    groups = []
    for sha256, rows in by_sha.items():
        rows.sort(
            key=lambda row: (
                -row["A4_reference_count"],
                -row["score"],
                len(row["path"]),
                row["path"],
            )
        )
        groups.append({
            "sha256": sha256,
            "canonical": rows[0],
            "copies": rows,
            "A4_reference_count": max(row["A4_reference_count"] for row in rows),
            "score": max(row["score"] for row in rows),
        })

    groups.sort(
        key=lambda group: (
            -group["A4_reference_count"],
            -group["score"],
            group["canonical"]["path"],
        )
    )

    decision = {
        "strict_candidate_count": len(strict),
        "unique_strict_sha256_count": len(groups),
        "groups": groups,
        "selection_rule": (
            "Select one unique strict training implementation SHA. Prefer "
            "implementations explicitly referenced by A4 evidence; copied "
            "files with the same SHA are one implementation."
        ),
    }

    if not groups:
        decision["classification"] = "no_strict_official_training_route_found"
        return None, decision

    referenced_groups = [
        group for group in groups
        if group["A4_reference_count"] > 0
    ]

    if len(referenced_groups) == 1:
        decision["classification"] = "unique_A4_referenced_strict_training_route"
        return referenced_groups[0]["canonical"], decision

    if len(referenced_groups) > 1:
        decision["classification"] = "multiple_A4_referenced_strict_training_routes"
        return None, decision

    if len(groups) == 1:
        decision["classification"] = "unique_repository_strict_training_route"
        return groups[0]["canonical"], decision

    top = groups[0]
    second = groups[1]
    if top["score"] - second["score"] >= 20:
        decision["classification"] = "unique_high_margin_repository_training_route"
        return top["canonical"], decision

    decision["classification"] = "multiple_unresolved_strict_training_routes"
    return None, decision


def construct_split_lengths(
    loader_path: Path,
    data_root: Path,
) -> dict[str, Any]:
    module = import_source(
        loader_path,
        "_v5_p3_f7_p1_guarded_loader",
    )
    require(
        hasattr(module, "_load_original_class")
        and hasattr(module, "_construct_original"),
        "guarded loader helper route changed",
    )
    dataset_class = module._load_original_class(data_root)

    results = {}
    for canonical, aliases in {
        "train": ("train", "training", "TRAIN"),
        "validation": ("validation", "val", "VALIDATION"),
    }.items():
        attempts = []
        resolved = None
        for split in aliases:
            try:
                dataset = module._construct_original(
                    dataset_class,
                    data_root,
                    split,
                    {},
                )
                length = int(len(dataset))
                attempts.append({
                    "split": split,
                    "status": "SUCCESS",
                    "length": length,
                    "dataset_type": (
                        f"{dataset.__class__.__module__}."
                        f"{dataset.__class__.__name__}"
                    ),
                })
                if resolved is None:
                    resolved = {
                        "split": split,
                        "length": length,
                        "dataset_type": attempts[-1]["dataset_type"],
                    }
                    break
            except BaseException as exc:
                attempts.append({
                    "split": split,
                    "status": "FAILED",
                    "error": repr(exc),
                })

        results[canonical] = {
            "resolved": resolved,
            "attempts": attempts,
        }

    train_length = (
        results["train"]["resolved"]["length"]
        if results["train"]["resolved"] is not None
        else None
    )
    validation_length = (
        results["validation"]["resolved"]["length"]
        if results["validation"]["resolved"] is not None
        else None
    )

    return {
        "loader": str(loader_path.resolve()),
        "loader_sha256": sha256_file(loader_path),
        "dataset_class": (
            f"{dataset_class.__module__}.{dataset_class.__name__}"
        ),
        "train": results["train"],
        "validation": results["validation"],
        "train_length": train_length,
        "validation_length": validation_length,
        "train_count_certified": train_length == EXPECTED_TRAIN_ITEMS,
        "validation_count_certified": (
            validation_length == EXPECTED_VALIDATION_ITEMS
        ),
        "dataset_getitem_called": False,
        "training_feature_tensors_loaded": False,
        "validation_feature_tensors_loaded": False,
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
    data_root = data_link.resolve()
    require(data_root.is_dir(), f"dataset target missing: {data_root}")

    feature_root = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study"
    )
    f7_root = feature_root / "retrained_group_ablation"
    f6r_root = feature_root / "integrated_gradients_review"
    permutation_root = feature_root / "permutation"

    p0_report_path = f7_root / (
        "V5_P3_F7_P0_SAME_WIDTH_RETRAINED_GROUP_ABLATION_PREFLIGHT_REPORT.json"
    )
    p0_lock_path = f7_root / (
        "V5_P3_F7_P0_SAME_WIDTH_RETRAINED_GROUP_ABLATION_PREFLIGHT_LOCK.json"
    )
    f6r_report_path = f6r_root / (
        "V5_P3_F6R_TASK_SPECIFIC_INTEGRATED_GRADIENTS_RESULT_REVIEW_REPORT.json"
    )
    f6r_lock_path = f6r_root / (
        "V5_P3_F6R_TASK_SPECIFIC_INTEGRATED_GRADIENTS_RESULT_REVIEW_LOCK.json"
    )
    route_inventory_path = permutation_root / (
        "F5_P0_EXECUTION_ROUTE_SOURCE_INVENTORY.json"
    )

    p0_report, p0_lock = verify_report_lock(p0_report_path, p0_lock_path)
    f6r_report, f6r_lock = verify_report_lock(
        f6r_report_path,
        f6r_lock_path,
    )
    route_inventory = load_json(route_inventory_path)

    require(p0_lock.get("F7_P0_complete") is True, "F7-P0 incomplete")
    require(p0_lock.get("F7_protocol_frozen") is True, "F7 protocol not frozen")
    require(
        p0_lock.get("actual_F7_retraining_authorized") is False,
        "F7 unexpectedly authorized before P1",
    )
    require(p0_lock.get("F8_multi_seed_authorized") is False, "F8 must remain held")
    require(
        p0_lock.get("feature_removal_authorized") is False,
        "feature removal must remain held",
    )
    require(f6r_lock.get("F6R_complete") is True, "F6R incomplete")

    p0_selected = p0_report["finding"].get("selected_trainer")
    false_positive_detected = bool(
        p0_selected is not None
        and (
            "f7_p0_same_width_retrained_group_ablation_preflight"
            in normalize(p0_selected["path"])
            or p0_selected["path"] == str(installed_script)
        )
    )

    certified_loader = route_inventory["certified_files"]["loader"]
    loader_path = Path(certified_loader["path"]).resolve()
    require(loader_path.is_file(), f"certified loader missing: {loader_path}")
    require(
        sha256_file(loader_path) == certified_loader["actual_sha256"],
        "certified loader hash changed",
    )

    split_counts = construct_split_lengths(loader_path, data_root)
    a4_references = mine_a4_references(repo)
    candidates = repository_candidates(repo, a4_references)
    selected_trainer, selection = select_trainer(candidates)
    git_history = git_history_candidates(repo)

    trainer_resolved = selected_trainer is not None
    counts_certified = bool(
        split_counts["train_count_certified"]
        and split_counts["validation_count_certified"]
    )

    adapter_preflight_authorized = bool(
        trainer_resolved
        and counts_certified
        and f6r_lock.get("F7_protocol_preflight_authorized") is True
    )

    next_stage = (
        "V5_P3_F7_P2_FROZEN_TRAINER_ADAPTER_AND_DRY_RUN_PREFLIGHT"
        if adapter_preflight_authorized
        else "V5_P3_F7_P1A_MANUAL_TRAINING_ROUTE_PIN"
    )

    a4_path = output_dir / "F7_P1_A4_TRAINING_SOURCE_REFERENCE_REVIEW.json"
    candidate_path = output_dir / "F7_P1_STRICT_TRAINER_CANDIDATE_REVIEW.json"
    git_path = output_dir / "F7_P1_GIT_HISTORY_TRAINER_ROUTE_REVIEW.json"
    count_path = output_dir / "F7_P1_GUARDED_SPLIT_LENGTH_CERTIFICATION.json"
    route_path = output_dir / "F7_P1_FROZEN_OFFICIAL_TRAINING_ROUTE.json"
    decision_path = output_dir / "F7_P1_EXECUTION_ROUTE_DECISION.json"

    atomic_json(a4_path, a4_references)
    atomic_json(
        candidate_path,
        {
            "P0_selected_trainer": p0_selected,
            "P0_false_positive_detected": false_positive_detected,
            "candidate_count": len(candidates),
            "selection": selection,
            "selected_trainer": selected_trainer,
            "candidates": candidates,
        },
    )
    atomic_json(git_path, git_history)
    atomic_json(count_path, split_counts)

    frozen_route = {
        "status": "FROZEN" if trainer_resolved else "UNRESOLVED",
        "trainer": selected_trainer,
        "selection_classification": selection["classification"],
        "model_class": EXPECTED_MODEL_CLASS,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "guarded_loader": {
            "path": str(loader_path),
            "sha256": sha256_file(loader_path),
        },
        "dataset_target": str(data_root),
        "dataset_symlink": str(data_link),
        "train_items": split_counts["train_length"],
        "validation_items": split_counts["validation_length"],
        "training_recipe_source": (
            "official A4 route and selected strict training implementation"
            if trainer_resolved
            else None
        ),
        "allowed_next_modification": (
            "P2 may wrap the frozen trainer with one post-loader group-zeroing "
            "operation and must verify a no-mask dry run before training."
            if trainer_resolved
            else None
        ),
    }
    atomic_json(route_path, frozen_route)

    decision = {
        "F7_P0_complete": True,
        "P0_false_positive_trainer_detected": false_positive_detected,
        "P0_false_positive_reason": (
            "The selected file was the F7-P0 preflight itself. It contains "
            "training-related words and AST symbols for discovery logic but "
            "does not execute the official A4 training loop."
            if false_positive_detected
            else None
        ),
        "train_count_certified": split_counts["train_count_certified"],
        "validation_count_certified": split_counts[
            "validation_count_certified"
        ],
        "official_trainer_route_resolved": trainer_resolved,
        "trainer_selection_classification": selection["classification"],
        "F7_adapter_preflight_authorized": adapter_preflight_authorized,
        "actual_F7_retraining_authorized": False,
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
            "Correct the F7-P0 trainer-discovery false positive; mine A4 "
            "artifacts for explicit Python source references; inspect repository "
            "candidates with strict AST evidence for a real optimization loop; "
            "group copied implementations by SHA; review git history; construct "
            "the guarded train and validation dataset objects without item "
            "access; certify exact split lengths; freeze the official trainer "
            "route when unique; and authorize only the trainer-adapter dry-run "
            "preflight."
        ),
        "finding": {
            "P0_selected_trainer": p0_selected,
            "P0_false_positive_detected": false_positive_detected,
            "train_items": split_counts["train_length"],
            "validation_items": split_counts["validation_length"],
            "train_count_certified": split_counts["train_count_certified"],
            "validation_count_certified": split_counts[
                "validation_count_certified"
            ],
            "trainer_candidate_count": len(candidates),
            "trainer_selection": selection,
            "selected_trainer": selected_trainer,
        },
        "decision": decision,
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "training_dataset_object_constructed": True,
            "validation_dataset_object_constructed": True,
            "dataset_getitem_called": False,
            "training_feature_tensors_loaded": False,
            "validation_feature_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "scientific_protocol_changed": False,
            "actual_F7_retraining_authorized": False,
        },
        "artifacts": {
            "A4_training_source_reference_review": str(a4_path),
            "strict_trainer_candidate_review": str(candidate_path),
            "git_history_review": str(git_path),
            "guarded_split_length_certification": str(count_path),
            "frozen_official_training_route": str(route_path),
            "execution_route_decision": str(decision_path),
        },
        "provenance": {
            "F7_P0_report_sha256": sha256_file(p0_report_path),
            "F7_P0_lock_sha256": sha256_file(p0_lock_path),
            "F6R_report_sha256": sha256_file(f6r_report_path),
            "F6R_lock_sha256": sha256_file(f6r_lock_path),
            "route_inventory_sha256": sha256_file(route_inventory_path),
            "certified_loader_sha256": sha256_file(loader_path),
            "installed_script_sha256": sha256_file(installed_script),
            "A4_reference_review_sha256": sha256_file(a4_path),
            "candidate_review_sha256": sha256_file(candidate_path),
            "git_history_review_sha256": sha256_file(git_path),
            "split_length_certification_sha256": sha256_file(count_path),
            "frozen_training_route_sha256": sha256_file(route_path),
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
            "A4_reference_review_sha256": sha256_file(a4_path),
            "candidate_review_sha256": sha256_file(candidate_path),
            "git_history_review_sha256": sha256_file(git_path),
            "split_length_certification_sha256": sha256_file(count_path),
            "frozen_training_route_sha256": sha256_file(route_path),
            "decision_sha256": sha256_file(decision_path),
            "P0_false_positive_trainer_detected": false_positive_detected,
            "train_count_certified": split_counts["train_count_certified"],
            "validation_count_certified": split_counts[
                "validation_count_certified"
            ],
            "official_trainer_route_resolved": trainer_resolved,
            "F7_adapter_preflight_authorized": adapter_preflight_authorized,
            "actual_F7_retraining_authorized": False,
            "F8_multi_seed_authorized": False,
            "feature_removal_authorized": False,
            "compact_interface_equivalence_authorized": False,
            "hardware_reduction_claim_authorized": False,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "dataset_getitem_called": False,
            "training_feature_tensors_loaded": False,
            "validation_feature_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print(
        "P0_false_positive_trainer_detected="
        f"{str(false_positive_detected).lower()}"
    )
    if p0_selected is not None:
        print(f"P0_false_positive_path={p0_selected['path']}")
    print("dataset_getitem_called=false")
    print("training_feature_tensors_loaded=false")
    print("validation_feature_tensors_loaded=false")
    print(f"train_items={split_counts['train_length']}")
    print(f"validation_items={split_counts['validation_length']}")
    print(
        "train_count_certified="
        f"{str(split_counts['train_count_certified']).lower()}"
    )
    print(
        "validation_count_certified="
        f"{str(split_counts['validation_count_certified']).lower()}"
    )
    print(f"strict_candidate_count={selection['strict_candidate_count']}")
    print(
        "unique_strict_sha256_count="
        f"{selection['unique_strict_sha256_count']}"
    )
    print(
        "trainer_selection_classification="
        f"{selection['classification']}"
    )
    if selected_trainer is not None:
        print(f"selected_trainer={selected_trainer['path']}")
        print(f"selected_trainer_sha256={selected_trainer['sha256']}")
        print(
            "selected_trainer_A4_reference_count="
            f"{selected_trainer['A4_reference_count']}"
        )
    else:
        print("selected_trainer=None")
    print(
        "official_trainer_route_resolved="
        f"{str(trainer_resolved).lower()}"
    )
    print(
        "F7_adapter_preflight_authorized="
        f"{str(adapter_preflight_authorized).lower()}"
    )
    print("actual_F7_retraining_authorized=false")
    print("F8_multi_seed_authorized=false")
    print("feature_removal_authorized=false")
    print("compact_interface_equivalence_authorized=false")
    print("hardware_reduction_claim_authorized=false")
    print("model_loaded=false")
    print("checkpoint_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={next_stage}")
    print(f"A4_training_source_reference_review={a4_path}")
    print(f"strict_trainer_candidate_review={candidate_path}")
    print(f"git_history_review={git_path}")
    print(f"guarded_split_length_certification={count_path}")
    print(f"frozen_official_training_route={route_path}")
    print(f"execution_route_decision={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
