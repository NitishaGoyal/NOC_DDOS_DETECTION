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


STAGE = "V5_P3_F7_E0D_R1_EXACT_EARLY_STOP_POLICY_PIN"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "TRAIN-DERIVED"

SKELETON_RELATIVE_PATH = Path(
    "scripts/v5/p2/train_v5_p2_b2_single_seed.py"
)

EXPECTED_CONDITION_NORMALIZED = (
    "epoch_15_and_early_stop_patience_counter_12"
)
EXPECTED_COUNTER = "early_stop_patience_counter"
EXPECTED_MINIMUM_EPOCH = 15
EXPECTED_PATIENCE = 12
EXPECTED_INCREMENT = 1
EXPECTED_RESET = 0
EXPECTED_EPOCH_START = 1
EXPECTED_EPOCH_STOP_EXCLUSIVE = 101
EXPECTED_EXECUTED_EPOCH_BUDGET = 100


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


def source_record(
    path: Path,
    lines: list[str],
    node: ast.AST,
    context: int = 4,
) -> dict[str, Any]:
    start = int(getattr(node, "lineno", 0))
    end = int(getattr(node, "end_lineno", start))
    context_start = max(1, start - context)
    context_end = min(len(lines), end + context)
    excerpt_lines = lines[context_start - 1:context_end]
    raw = "\n".join(excerpt_lines)
    numbered = "\n".join(
        f"{line_number:05d}: {line}"
        for line_number, line in enumerate(
            excerpt_lines,
            start=context_start,
        )
    )
    source = ast.unparse(node)
    return {
        "path": str(path.resolve()),
        "line_start": start,
        "line_end": end,
        "context_start": context_start,
        "context_end": context_end,
        "source": source,
        "source_sha256": sha256_text(source),
        "raw_excerpt": raw,
        "raw_excerpt_sha256": sha256_text(raw),
        "numbered_excerpt": numbered,
        "numbered_excerpt_sha256": sha256_text(numbered),
    }


def contains_break(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Break) for child in ast.walk(node))


def contains_checkpoint_save(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        short = call_name(child.func).rsplit(".", 1)[-1]
        if short in {"save", "save_checkpoint"}:
            return True
    return False


def numeric_constant(node: ast.AST) -> int | float | None:
    if isinstance(node, ast.Constant):
        if (
            isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        child = numeric_constant(node.operand)
        return -child if child is not None else None
    return None


def parse_comparison(node: ast.Compare) -> dict[str, Any]:
    require(
        len(node.ops) == 1 and len(node.comparators) == 1,
        "chained comparison unsupported",
    )
    operator = {
        ast.GtE: ">=",
        ast.Gt: ">",
        ast.LtE: "<=",
        ast.Lt: "<",
        ast.Eq: "==",
        ast.NotEq: "!=",
    }.get(type(node.ops[0]))
    require(operator is not None, "comparison operator unsupported")

    left = ast.unparse(node.left)
    right = ast.unparse(node.comparators[0])
    left_numeric = numeric_constant(node.left)
    right_numeric = numeric_constant(node.comparators[0])

    return {
        "left": left,
        "operator": operator,
        "right": right,
        "left_numeric": left_numeric,
        "right_numeric": right_numeric,
        "source": ast.unparse(node),
    }


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

    f7_root = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/"
        "retrained_group_ablation"
    )

    e0d_report_path = f7_root / (
        "V5_P3_F7_E0D_MANUAL_EARLY_STOP_SOURCE_EXCERPT_PIN_REPORT.json"
    )
    e0d_lock_path = f7_root / (
        "V5_P3_F7_E0D_MANUAL_EARLY_STOP_SOURCE_EXCERPT_PIN_LOCK.json"
    )
    e0d_candidates_path = f7_root / (
        "F7_E0D_EARLY_STOP_BREAK_CANDIDATE_EXCERPTS.json"
    )
    e0c_recipe_path = f7_root / (
        "F7_E0C_FINAL_EXECUTION_RECIPE_LOCK.json"
    )
    e0c_report_path = f7_root / (
        "V5_P3_F7_E0C_TARGETED_CONTROL_FLOW_RECOVERY_REPORT.json"
    )
    e0c_lock_path = f7_root / (
        "V5_P3_F7_E0C_TARGETED_CONTROL_FLOW_RECOVERY_LOCK.json"
    )
    protocol_path = f7_root / (
        "F7_P0_FROZEN_SAME_WIDTH_ABLATION_PROTOCOL.json"
    )
    p2_lock_path = f7_root / (
        "V5_P3_F7_P2_RECONSTRUCTED_TRAINER_ADAPTER_AND_DRY_RUN_PREFLIGHT_LOCK.json"
    )

    e0d_report, e0d_lock = verify_report_lock(
        e0d_report_path,
        e0d_lock_path,
    )
    e0c_report, e0c_lock = verify_report_lock(
        e0c_report_path,
        e0c_lock_path,
    )
    candidates_document = load_json(e0d_candidates_path)
    e0c_recipe = load_json(e0c_recipe_path)
    protocol = load_json(protocol_path)
    p2_lock = load_json(p2_lock_path)

    require(e0d_lock.get("E0D_complete") is True, "E0D incomplete")
    require(
        e0d_lock.get("unique_early_stop_candidate") is True,
        "E0D did not freeze one unique candidate",
    )
    require(
        e0d_lock.get("E1_trainer_generation_authorized") is False,
        "E1 already authorized before E0D-R1",
    )
    require(e0c_lock.get("E0C_complete") is True, "E0C incomplete")
    require(
        p2_lock.get("actual_F7_retraining_authorized") is True,
        "P2 did not authorize F7 execution",
    )
    require(
        p2_lock.get("sealed_test_tensors_loaded") is False,
        "sealed-test access detected",
    )

    require(
        int(candidates_document["early_stop_candidate_count"]) == 1,
        "candidate document is not unique",
    )
    frozen_candidate = candidates_document["candidates"][0]

    skeleton_path = (repo / SKELETON_RELATIVE_PATH).resolve()
    require(skeleton_path.is_file(), f"missing skeleton: {skeleton_path}")
    require(
        sha256_file(skeleton_path)
        == e0d_report["finding"]["skeleton_sha256"],
        "skeleton hash changed since E0D",
    )

    text = skeleton_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    tree = ast.parse(text)

    parent: dict[ast.AST, ast.AST] = {}
    for ancestor in ast.walk(tree):
        for child in ast.iter_child_nodes(ancestor):
            parent[child] = ancestor

    candidate_nodes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not contains_break(node):
            continue
        if int(node.lineno) != int(frozen_candidate["line_start"]):
            continue
        if int(getattr(node, "end_lineno", node.lineno)) != int(
            frozen_candidate["line_end"]
        ):
            continue
        candidate_nodes.append(node)

    require(
        len(candidate_nodes) == 1,
        f"could not recover unique candidate AST node: {len(candidate_nodes)}",
    )
    candidate_node = candidate_nodes[0]
    require(
        ast.unparse(candidate_node) == frozen_candidate["full_source"],
        "candidate source changed",
    )
    require(
        sha256_text(ast.unparse(candidate_node))
        == frozen_candidate["full_source_sha256"],
        "candidate source SHA changed",
    )
    require(
        enclosing_function(candidate_node, parent) == "main",
        "early-stop candidate moved outside main",
    )

    require(
        isinstance(candidate_node.test, ast.BoolOp)
        and isinstance(candidate_node.test.op, ast.And),
        "early-stop condition is not a two-gate AND",
    )
    require(
        len(candidate_node.test.values) == 2,
        "early-stop condition does not have exactly two gates",
    )

    comparisons = []
    for value in candidate_node.test.values:
        require(
            isinstance(value, ast.Compare),
            "early-stop gate is not a comparison",
        )
        comparisons.append(parse_comparison(value))

    epoch_gate = None
    patience_gate = None
    for row in comparisons:
        if row["left"] == "epoch":
            epoch_gate = row
        elif row["left"] == EXPECTED_COUNTER:
            patience_gate = row

    require(epoch_gate is not None, "epoch warm-up gate missing")
    require(patience_gate is not None, "patience counter gate missing")
    require(epoch_gate["operator"] == ">=", "epoch gate operator changed")
    require(
        epoch_gate["right_numeric"] == EXPECTED_MINIMUM_EPOCH,
        "minimum epoch changed",
    )
    require(
        patience_gate["operator"] == ">=",
        "patience gate operator changed",
    )
    require(
        patience_gate["right_numeric"] == EXPECTED_PATIENCE,
        "patience threshold changed",
    )

    assignments = []
    for node in ast.walk(tree):
        function_name = enclosing_function(node, parent)
        if function_name != "main":
            continue

        if isinstance(node, ast.Assign):
            names = []
            for target in node.targets:
                names.extend(target_names(target))
            if EXPECTED_COUNTER in names:
                assignments.append({
                    "kind": "assign",
                    "value": numeric_constant(node.value),
                    "record": source_record(
                        skeleton_path,
                        lines,
                        node,
                        context=6,
                    ),
                    "node": node,
                })

        elif isinstance(node, ast.AugAssign):
            names = target_names(node.target)
            if EXPECTED_COUNTER in names:
                assignments.append({
                    "kind": (
                        "increment"
                        if isinstance(node.op, ast.Add)
                        else f"augassign_{type(node.op).__name__}"
                    ),
                    "value": numeric_constant(node.value),
                    "record": source_record(
                        skeleton_path,
                        lines,
                        node,
                        context=6,
                    ),
                    "node": node,
                })

    reset_rows = [
        row
        for row in assignments
        if row["kind"] == "assign"
        and row["value"] == EXPECTED_RESET
    ]
    increment_rows = [
        row
        for row in assignments
        if row["kind"] == "increment"
        and row["value"] == EXPECTED_INCREMENT
    ]

    require(
        len(reset_rows) >= 2,
        "expected initialization and improvement reset to zero",
    )
    require(
        len(increment_rows) == 1,
        "expected exactly one increment-by-one event",
    )

    improvement_branches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        body_nodes = list(ast.walk(ast.Module(body=node.body, type_ignores=[])))
        else_nodes = list(ast.walk(ast.Module(body=node.orelse, type_ignores=[])))

        reset_in_body = any(
            isinstance(child, ast.Assign)
            and EXPECTED_COUNTER
            in [
                name
                for target in child.targets
                for name in target_names(target)
            ]
            and numeric_constant(child.value) == EXPECTED_RESET
            for child in body_nodes
        )
        increment_in_else = any(
            isinstance(child, ast.AugAssign)
            and EXPECTED_COUNTER in target_names(child.target)
            and isinstance(child.op, ast.Add)
            and numeric_constant(child.value) == EXPECTED_INCREMENT
            for child in else_nodes
        )

        if reset_in_body and increment_in_else:
            improvement_branches.append({
                "condition": ast.unparse(node.test),
                "checkpoint_saved_in_improvement_body": (
                    contains_checkpoint_save(
                        ast.Module(body=node.body, type_ignores=[])
                    )
                ),
                "record": source_record(
                    skeleton_path,
                    lines,
                    node,
                    context=8,
                ),
            })

    require(
        len(improvement_branches) == 1,
        f"improvement/reset branch is not unique: {len(improvement_branches)}",
    )
    improvement_branch = improvement_branches[0]

    increment_line = increment_rows[0]["record"]["line_start"]
    stop_line = int(candidate_node.lineno)
    require(
        increment_line < stop_line,
        "counter increment does not occur before stop check",
    )

    epoch_semantics = e0c_recipe["epoch_loop"]
    require(
        int(epoch_semantics["start"]) == EXPECTED_EPOCH_START,
        "epoch start changed",
    )
    require(
        int(epoch_semantics["stop_exclusive"])
        == EXPECTED_EPOCH_STOP_EXCLUSIVE,
        "epoch stop changed",
    )
    require(
        int(epoch_semantics["executed_epoch_count"])
        == EXPECTED_EXECUTED_EPOCH_BUDGET,
        "executed epoch budget changed",
    )

    policy = {
        "classification": (
            "DUAL_GATE_CONSECUTIVE_NON_IMPROVEMENT_EARLY_STOP"
        ),
        "condition": ast.unparse(candidate_node.test),
        "condition_source_sha256": sha256_text(
            ast.unparse(candidate_node.test)
        ),
        "minimum_epoch_gate": {
            "variable": "epoch",
            "operator": ">=",
            "threshold": EXPECTED_MINIMUM_EPOCH,
            "meaning": (
                "early stopping cannot occur before epoch 15"
            ),
        },
        "patience_gate": {
            "counter": EXPECTED_COUNTER,
            "operator": ">=",
            "threshold": EXPECTED_PATIENCE,
            "reset_value": EXPECTED_RESET,
            "increment_value": EXPECTED_INCREMENT,
            "increment_occurs_before_stop_check": True,
            "meaning": (
                "stop after 12 consecutive validation epochs without "
                "improvement, subject to the epoch>=15 warm-up gate"
            ),
        },
        "effective_patience_epochs": EXPECTED_PATIENCE,
        "earliest_possible_stop_epoch": EXPECTED_MINIMUM_EPOCH,
        "maximum_executed_epoch_budget": (
            EXPECTED_EXECUTED_EPOCH_BUDGET
        ),
        "improvement_branch": improvement_branch,
        "initialization_and_reset_records": [
            {
                key: value
                for key, value in row.items()
                if key != "node"
            }
            for row in reset_rows
        ],
        "increment_record": {
            key: value
            for key, value in increment_rows[0].items()
            if key != "node"
        },
        "stop_record": source_record(
            skeleton_path,
            lines,
            candidate_node,
            context=10,
        ),
        "semantics_resolved": True,
    }

    final_recipe = dict(e0c_recipe)
    final_recipe["status"] = "FROZEN"
    final_recipe["early_stopping"] = policy
    final_recipe["unresolved_fields"] = []
    final_recipe["trainer_generation_authorized"] = True
    final_recipe["historical_A4_trainer_claimed"] = False

    policy_path = output_dir / (
        "F7_E0D_R1_EXACT_EARLY_STOP_POLICY.json"
    )
    final_recipe_path = output_dir / (
        "F7_E0D_R1_FINAL_RECONSTRUCTED_EXECUTION_RECIPE.json"
    )
    evidence_path = output_dir / (
        "F7_E0D_R1_COUNTER_RESET_INCREMENT_AND_IMPROVEMENT_EVIDENCE.json"
    )
    decision_path = output_dir / (
        "F7_E0D_R1_TRAINER_GENERATION_AUTHORIZATION.json"
    )

    atomic_json(policy_path, policy)
    atomic_json(final_recipe_path, final_recipe)
    atomic_json(
        evidence_path,
        {
            "candidate_source_sha256": (
                frozen_candidate["full_source_sha256"]
            ),
            "candidate_condition": ast.unparse(candidate_node.test),
            "candidate_comparisons": comparisons,
            "reset_records": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "node"
                }
                for row in reset_rows
            ],
            "increment_record": {
                key: value
                for key, value in increment_rows[0].items()
                if key != "node"
            },
            "improvement_branch": improvement_branch,
            "increment_line": increment_line,
            "stop_check_line": stop_line,
        },
    )

    decision = {
        "E0D_complete": True,
        "unique_early_stop_candidate_verified": True,
        "condition_source_hash_verified": True,
        "minimum_epoch_gate_verified": True,
        "patience_threshold_verified": True,
        "counter_initialization_verified": True,
        "counter_reset_on_improvement_verified": True,
        "counter_increment_on_non_improvement_verified": True,
        "increment_before_stop_check_verified": True,
        "early_stopping_semantics_resolved": True,
        "final_execution_recipe_frozen": True,
        "E1_trainer_generation_authorized": True,
        "actual_scientific_training_started": False,
        "actual_F7_primary_matrix_execution_started": False,
        "F8_multi_seed_authorized": False,
        "feature_removal_authorized": False,
        "compact_interface_equivalence_authorized": False,
        "hardware_reduction_claim_authorized": False,
        "sealed_test_access": False,
        "next_stage": (
            "V5_P3_F7_E1_RECONSTRUCTED_CONTROL_AND_GROUP_ABLATION_"
            "TRAINER_GENERATION"
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
            "Use the unique E0D source excerpt and exact AST to freeze the "
            "remaining early-stopping policy: verify the epoch>=15 warm-up "
            "gate, the early_stop_patience_counter>=12 gate, counter "
            "initialization/reset to zero, increment by one on non-improvement, "
            "increment-before-stop ordering and the unique improvement branch; "
            "merge the resolved policy into the E0C execution recipe; and "
            "authorize only E1 trainer generation."
        ),
        "finding": {
            "skeleton": str(skeleton_path),
            "skeleton_sha256": sha256_file(skeleton_path),
            "policy": policy,
            "final_recipe_status": final_recipe["status"],
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
            "historical_A4_trainer_claimed": False,
        },
        "artifacts": {
            "early_stop_policy": str(policy_path),
            "final_execution_recipe": str(final_recipe_path),
            "counter_and_improvement_evidence": str(evidence_path),
            "trainer_generation_authorization": str(decision_path),
        },
        "provenance": {
            "E0D_report_sha256": sha256_file(e0d_report_path),
            "E0D_lock_sha256": sha256_file(e0d_lock_path),
            "E0D_candidates_sha256": sha256_file(
                e0d_candidates_path
            ),
            "E0C_report_sha256": sha256_file(e0c_report_path),
            "E0C_lock_sha256": sha256_file(e0c_lock_path),
            "E0C_recipe_sha256": sha256_file(e0c_recipe_path),
            "F7_protocol_sha256": sha256_file(protocol_path),
            "P2_lock_sha256": sha256_file(p2_lock_path),
            "skeleton_sha256": sha256_file(skeleton_path),
            "installed_script_sha256": sha256_file(installed_script),
            "policy_sha256": sha256_file(policy_path),
            "final_recipe_sha256": sha256_file(final_recipe_path),
            "evidence_sha256": sha256_file(evidence_path),
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
            "policy_sha256": sha256_file(policy_path),
            "final_recipe_sha256": sha256_file(final_recipe_path),
            "evidence_sha256": sha256_file(evidence_path),
            "decision_sha256": sha256_file(decision_path),
            "E0D_R1_complete": True,
            "early_stopping_semantics_resolved": True,
            "final_execution_recipe_frozen": True,
            "E1_trainer_generation_authorized": True,
            "actual_scientific_training_started": False,
            "actual_F7_primary_matrix_execution_started": False,
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
        "early_stopping_classification="
        "DUAL_GATE_CONSECUTIVE_NON_IMPROVEMENT_EARLY_STOP"
    )
    print(
        "early_stopping_condition="
        f"{ast.unparse(candidate_node.test)}"
    )
    print("minimum_early_stop_epoch=15")
    print("patience_counter=early_stop_patience_counter")
    print("patience_threshold=12")
    print("patience_counter_initial_value=0")
    print("patience_counter_reset_value=0")
    print("patience_counter_increment=1")
    print("patience_increment_before_stop_check=true")
    print("counter_reset_on_improvement_verified=true")
    print("counter_increment_on_non_improvement_verified=true")
    print(
        "effective_policy="
        "stop_after_12_consecutive_non_improving_validation_epochs_"
        "but_not_before_epoch_15"
    )
    print("executed_epoch_budget=100")
    print("early_stopping_semantics_resolved=true")
    print("final_execution_recipe_frozen=true")
    print("E1_trainer_generation_authorized=true")
    print("actual_scientific_training_started=false")
    print("actual_F7_primary_matrix_execution_started=false")
    print("F8_multi_seed_authorized=false")
    print("feature_removal_authorized=false")
    print("compact_interface_equivalence_authorized=false")
    print("hardware_reduction_claim_authorized=false")
    print("model_loaded=false")
    print("checkpoint_loaded=false")
    print("training_feature_tensors_loaded=false")
    print("validation_feature_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(
        "next_stage="
        "V5_P3_F7_E1_RECONSTRUCTED_CONTROL_AND_GROUP_ABLATION_"
        "TRAINER_GENERATION"
    )
    print(f"early_stop_policy={policy_path}")
    print(f"final_execution_recipe={final_recipe_path}")
    print(f"counter_and_improvement_evidence={evidence_path}")
    print(f"trainer_generation_authorization={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
