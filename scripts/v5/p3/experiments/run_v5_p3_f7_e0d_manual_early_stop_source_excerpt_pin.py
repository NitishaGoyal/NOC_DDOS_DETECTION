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

STAGE = "V5_P3_F7_E0D_MANUAL_EARLY_STOP_SOURCE_EXCERPT_PIN"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "TRAIN-DERIVED"
SKELETON = Path("scripts/v5/p2/train_v5_p2_b2_single_seed.py")

TOKENS = (
    "patience", "early_stop", "early_stopping", "stale_epoch",
    "bad_epoch", "no_improve", "best_epoch", "best_score",
    "best_metric", "best_selection",
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


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def verify_report_lock(report: Path, lock: Path) -> tuple[dict, dict]:
    report_data = load_json(report)
    lock_data = load_json(lock)
    require(report_data.get("status") == "PASS", "E0C report not PASS")
    require(
        lock_data.get("report_sha256") == sha256_file(report),
        "E0C report/lock mismatch",
    )
    return report_data, lock_data


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def contains_break(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Break) for child in ast.walk(node))


def target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [ast.unparse(node)]
    if isinstance(node, (ast.Tuple, ast.List)):
        result = []
        for child in node.elts:
            result.extend(target_names(child))
        return result
    return []


def enclosing_function(
    node: ast.AST,
    parent: dict[ast.AST, ast.AST],
) -> str | None:
    current = node
    while current in parent:
        current = parent[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
    return None


def excerpt(
    path: Path,
    lines: list[str],
    start: int,
    end: int,
    context: int,
) -> dict[str, Any]:
    context_start = max(1, start - context)
    context_end = min(len(lines), end + context)
    selected = lines[context_start - 1:context_end]
    raw = "\n".join(selected)
    numbered = "\n".join(
        f"{number:05d}: {line}"
        for number, line in enumerate(selected, start=context_start)
    )
    return {
        "path": str(path),
        "source_start": start,
        "source_end": end,
        "context_start": context_start,
        "context_end": context_end,
        "raw": raw,
        "numbered": numbered,
        "raw_sha256": sha256_text(raw),
        "numbered_sha256": sha256_text(numbered),
    }


def condition_parts(node: ast.AST) -> list[dict[str, str | None]]:
    rows = []

    def visit(child: ast.AST, boolean_parent: str | None = None) -> None:
        if isinstance(child, ast.BoolOp):
            operator = "AND" if isinstance(child.op, ast.And) else "OR"
            for value in child.values:
                visit(value, operator)
            return
        rows.append({
            "boolean_parent": boolean_parent,
            "node_type": type(child).__name__,
            "expression": ast.unparse(child),
        })

    visit(node)
    return rows


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"missing repo: {repo}")
    require(data_link.is_symlink(), f"missing dataset symlink: {data_link}")
    require(data_link.resolve().is_dir(), "dataset target missing")

    f7 = repo / "reports/v5/p3_experiments/f0_d70_feature_study/retrained_group_ablation"
    e0c_report_path = f7 / "V5_P3_F7_E0C_TARGETED_CONTROL_FLOW_RECOVERY_REPORT.json"
    e0c_lock_path = f7 / "V5_P3_F7_E0C_TARGETED_CONTROL_FLOW_RECOVERY_LOCK.json"
    e0c_report, e0c_lock = verify_report_lock(
        e0c_report_path,
        e0c_lock_path,
    )

    require(e0c_lock.get("E0C_complete") is True, "E0C incomplete")
    require(
        e0c_lock.get("E1_trainer_generation_authorized") is False,
        "E1 already authorized",
    )
    require(
        e0c_lock.get("feature_removal_authorized") is False,
        "feature removal unexpectedly authorized",
    )
    require(
        e0c_lock.get("sealed_test_tensors_loaded") is False,
        "sealed-test access detected",
    )

    skeleton = (repo / SKELETON).resolve()
    require(skeleton.is_file(), f"missing skeleton: {skeleton}")
    require(
        sha256_file(skeleton) == e0c_report["finding"]["skeleton_sha256"],
        "skeleton hash changed since E0C",
    )

    text = skeleton.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    tree = ast.parse(text)

    parent = {}
    for ancestor in ast.walk(tree):
        for child in ast.iter_child_nodes(ancestor):
            parent[child] = ancestor

    break_candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not contains_break(node):
            continue
        source = ast.unparse(node)
        token = normalize(source)
        hits = [name for name in TOKENS if normalize(name) in token]
        symbols = sorted({
            child.id for child in ast.walk(node.test)
            if isinstance(child, ast.Name)
        })
        attributes = sorted({
            ast.unparse(child) for child in ast.walk(node.test)
            if isinstance(child, ast.Attribute)
        })
        relevant = sorted({
            symbol for symbol in symbols + attributes
            if any(normalize(name) in normalize(symbol) for name in TOKENS)
            or "epoch" in normalize(symbol)
            or "score" in normalize(symbol)
            or "metric" in normalize(symbol)
        })
        row = {
            "function": enclosing_function(node, parent),
            "line_start": int(node.lineno),
            "line_end": int(getattr(node, "end_lineno", node.lineno)),
            "condition": ast.unparse(node.test),
            "condition_parts": condition_parts(node.test),
            "token_hits": hits,
            "symbols": symbols,
            "attributes": attributes,
            "relevant_symbols": relevant,
            "full_source": source,
            "full_source_sha256": sha256_text(source),
            "excerpt": excerpt(
                skeleton,
                lines,
                int(node.lineno),
                int(getattr(node, "end_lineno", node.lineno)),
                16,
            ),
        }
        break_candidates.append(row)

    require(break_candidates, "no break-bearing if statement found")
    early_candidates = [
        row for row in break_candidates
        if row["token_hits"] or row["relevant_symbols"]
    ]
    require(early_candidates, "no early-stop-like break candidate found")

    assignment_rows = []
    relevant_symbols = sorted({
        symbol for row in early_candidates
        for symbol in row["relevant_symbols"]
    })
    simple_symbols = {symbol.rsplit(".", 1)[-1] for symbol in relevant_symbols}

    for node in ast.walk(tree):
        names = []
        value = None
        kind = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.extend(target_names(target))
            value = ast.unparse(node.value)
            kind = "assign"
        elif isinstance(node, ast.AnnAssign):
            names.extend(target_names(node.target))
            value = ast.unparse(node.value) if node.value is not None else None
            kind = "annotated_assign"
        elif isinstance(node, ast.AugAssign):
            names.extend(target_names(node.target))
            value = ast.unparse(node.value)
            kind = f"augassign_{type(node.op).__name__}"
        else:
            continue

        if not any(name.rsplit(".", 1)[-1] in simple_symbols for name in names):
            continue
        assignment_rows.append({
            "function": enclosing_function(node, parent),
            "kind": kind,
            "names": names,
            "value": value,
            "line_start": int(node.lineno),
            "line_end": int(getattr(node, "end_lineno", node.lineno)),
            "source": ast.unparse(node),
            "excerpt": excerpt(
                skeleton,
                lines,
                int(node.lineno),
                int(getattr(node, "end_lineno", node.lineno)),
                5,
            ),
        })

    argparse_rows = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        options = [
            argument.value for argument in node.args
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
        ]
        if not options:
            continue
        if not any(
            option.lstrip("-").replace("-", "_") in simple_symbols
            for option in options
        ):
            continue
        argparse_rows.append({
            "options": options,
            "line_start": int(node.lineno),
            "line_end": int(getattr(node, "end_lineno", node.lineno)),
            "source": ast.unparse(node),
            "excerpt": excerpt(
                skeleton,
                lines,
                int(node.lineno),
                int(getattr(node, "end_lineno", node.lineno)),
                4,
            ),
        })

    improvement_rows = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        token = normalize(ast.unparse(node))
        if not any(
            marker in token for marker in (
                "best_epoch", "best_score", "best_metric",
                "best_selection", "improv",
            )
        ):
            continue
        improvement_rows.append({
            "function": enclosing_function(node, parent),
            "line_start": int(node.lineno),
            "line_end": int(getattr(node, "end_lineno", node.lineno)),
            "condition": ast.unparse(node.test),
            "source": ast.unparse(node),
            "source_sha256": sha256_text(ast.unparse(node)),
            "excerpt": excerpt(
                skeleton,
                lines,
                int(node.lineno),
                int(getattr(node, "end_lineno", node.lineno)),
                12,
            ),
        })

    candidates_path = output_dir / "F7_E0D_EARLY_STOP_BREAK_CANDIDATE_EXCERPTS.json"
    symbols_path = output_dir / "F7_E0D_SYMBOL_ASSIGNMENT_AND_ARGPARSE_EVIDENCE.json"
    improvements_path = output_dir / "F7_E0D_IMPROVEMENT_BRANCH_EVIDENCE.json"
    pin_path = output_dir / "F7_E0D_MANUAL_SOURCE_EXCERPT_PIN.json"
    decision_path = output_dir / "F7_E0D_TRAINER_GENERATION_DECISION.json"

    atomic_json(candidates_path, {
        "break_candidate_count": len(break_candidates),
        "early_stop_candidate_count": len(early_candidates),
        "candidates": early_candidates,
    })
    atomic_json(symbols_path, {
        "relevant_symbols": relevant_symbols,
        "assignments": assignment_rows,
        "argparse": argparse_rows,
    })
    atomic_json(improvements_path, {
        "candidate_count": len(improvement_rows),
        "candidates": improvement_rows,
    })

    unique_candidate = len(early_candidates) == 1
    selected = early_candidates[0] if unique_candidate else None
    atomic_json(pin_path, {
        "status": "EXCERPT_FROZEN",
        "unique_candidate": unique_candidate,
        "selected_candidate": selected,
        "selected_excerpt_sha256": (
            selected["excerpt"]["raw_sha256"]
            if selected is not None else None
        ),
        "automatic_semantic_interpretation_authorized": False,
        "early_stopping_semantics_resolved": False,
        "E1_trainer_generation_authorized": False,
    })

    next_stage = (
        "V5_P3_F7_E0D_R1_EXACT_EARLY_STOP_POLICY_PIN"
        if unique_candidate
        else "V5_P3_F7_E0D_R0_MULTIPLE_BREAK_CANDIDATE_REVIEW"
    )
    decision = {
        "E0C_complete": True,
        "optimizer_step_cadence_resolved": True,
        "gradient_clipping_semantics_resolved": True,
        "scheduler_semantics_resolved": True,
        "loss_semantics_resolved": True,
        "epoch_semantics_resolved": True,
        "source_excerpt_frozen": True,
        "unique_early_stop_candidate": unique_candidate,
        "early_stopping_semantics_resolved": False,
        "E1_trainer_generation_authorized": False,
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
            "Freeze exact line-numbered and hashed early-stop source evidence "
            "after E0C resolved all other control-flow fields."
        ),
        "finding": {
            "skeleton": str(skeleton),
            "skeleton_sha256": sha256_file(skeleton),
            "break_candidate_count": len(break_candidates),
            "early_stop_candidate_count": len(early_candidates),
            "unique_candidate": unique_candidate,
            "relevant_symbols": relevant_symbols,
            "assignment_record_count": len(assignment_rows),
            "argparse_record_count": len(argparse_rows),
            "improvement_candidate_count": len(improvement_rows),
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
        },
        "artifacts": {
            "candidate_excerpts": str(candidates_path),
            "symbol_evidence": str(symbols_path),
            "improvement_evidence": str(improvements_path),
            "manual_pin": str(pin_path),
            "decision": str(decision_path),
        },
        "provenance": {
            "E0C_report_sha256": sha256_file(e0c_report_path),
            "E0C_lock_sha256": sha256_file(e0c_lock_path),
            "skeleton_sha256": sha256_file(skeleton),
            "installed_script_sha256": sha256_file(installed_script),
            "candidate_excerpts_sha256": sha256_file(candidates_path),
            "symbol_evidence_sha256": sha256_file(symbols_path),
            "improvement_evidence_sha256": sha256_file(improvements_path),
            "manual_pin_sha256": sha256_file(pin_path),
            "decision_sha256": sha256_file(decision_path),
        },
    }

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"

    atomic_json(report_path, report)
    atomic_json(lock_path, {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(report_path),
        "candidate_excerpts_sha256": sha256_file(candidates_path),
        "symbol_evidence_sha256": sha256_file(symbols_path),
        "improvement_evidence_sha256": sha256_file(improvements_path),
        "manual_pin_sha256": sha256_file(pin_path),
        "decision_sha256": sha256_file(decision_path),
        "E0D_complete": True,
        "source_excerpt_frozen": True,
        "unique_early_stop_candidate": unique_candidate,
        "early_stopping_semantics_resolved": False,
        "E1_trainer_generation_authorized": False,
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
    })
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print(f"training_skeleton={skeleton}")
    print(f"training_skeleton_sha256={sha256_file(skeleton)}")
    print(f"break_candidate_count={len(break_candidates)}")
    print(f"early_stop_candidate_count={len(early_candidates)}")
    print(f"unique_early_stop_candidate={str(unique_candidate).lower()}")

    for number, row in enumerate(early_candidates, start=1):
        print(
            f"early_stop_candidate_{number}="
            f"function={row['function']}:"
            f"lines={row['line_start']}-{row['line_end']}:"
            f"condition={row['condition']}:"
            f"symbols={row['relevant_symbols']}:"
            f"source_sha256={row['full_source_sha256']}"
        )
        print(f"early_stop_candidate_{number}_excerpt_begin")
        print(row["excerpt"]["numbered"])
        print(f"early_stop_candidate_{number}_excerpt_end")

    print(f"relevant_symbols={relevant_symbols}")
    for row in assignment_rows:
        print(
            "early_stop_assignment="
            f"function={row['function']}:"
            f"line={row['line_start']}:"
            f"kind={row['kind']}:"
            f"source={row['source']}"
        )
    for row in argparse_rows:
        print(
            "early_stop_argparse="
            f"line={row['line_start']}:"
            f"source={row['source']}"
        )

    print("early_stopping_semantics_resolved=false")
    print("E1_trainer_generation_authorized=false")
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
    print(f"candidate_excerpts={candidates_path}")
    print(f"symbol_evidence={symbols_path}")
    print(f"improvement_evidence={improvements_path}")
    print(f"manual_pin={pin_path}")
    print(f"decision={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
