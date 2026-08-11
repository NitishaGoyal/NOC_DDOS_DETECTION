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


STAGE = "V5_P3_F7_P1B_RECONSTRUCTED_TRAINING_RECIPE_PIN"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "TRAIN-DERIVED"

EXPECTED_MODEL_CLASS = "V6P0Dynamic70GraphConvCount4"
EXPECTED_PARAMETER_COUNT = 60553
EXPECTED_TRAIN_ITEMS = 110855
EXPECTED_VALIDATION_ITEMS = 13863
EXPECTED_SEED = 107

SKELETON_RELATIVE_PATH = Path(
    "scripts/v5/p2/train_v5_p2_b2_single_seed.py"
)

REQUIRED_SKELETON_EVIDENCE = {
    "dataloader": True,
    "backward": True,
    "optimizer_step": True,
    "zero_grad": True,
    "checkpoint_save": True,
    "epoch_loop": True,
    "train_split": True,
    "validation_split": True,
}

EXPECTED_SELECTION_FORMULA = (
    "(graph_auroc + graph_ap + source_ap + transit_ap + victim_ap "
    "+ path_ap + count_active_macro_f1) / 7"
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


class SkeletonVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.evidence = {
            "dataloader": False,
            "backward": False,
            "optimizer_step": False,
            "zero_grad": False,
            "checkpoint_save": False,
            "epoch_loop": False,
            "train_split": False,
            "validation_split": False,
        }
        self.argparse_defaults: dict[str, Any] = {}
        self.functions: list[str] = []
        self.classes: list[str] = []

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id == "DataLoader":
            self.evidence["dataloader"] = True
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if node.attr == "DataLoader":
            self.evidence["dataloader"] = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        function = node.func
        if isinstance(function, ast.Attribute):
            if function.attr == "backward":
                self.evidence["backward"] = True
            elif function.attr == "step":
                self.evidence["optimizer_step"] = True
            elif function.attr == "zero_grad":
                self.evidence["zero_grad"] = True
            elif function.attr in ("save", "save_checkpoint"):
                self.evidence["checkpoint_save"] = True

            if function.attr == "add_argument" and node.args:
                option = None
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    option = first.value
                if option:
                    default_value = None
                    default_found = False
                    for keyword in node.keywords:
                        if keyword.arg == "default":
                            default_found = True
                            try:
                                default_value = ast.literal_eval(keyword.value)
                            except Exception:
                                default_value = ast.unparse(keyword.value)
                    if default_found:
                        self.argparse_defaults[option] = default_value

        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        text = (
            ast.unparse(node.target).lower()
            + " "
            + ast.unparse(node.iter).lower()
        )
        if "epoch" in text:
            self.evidence["epoch_loop"] = True
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> Any:
        if "epoch" in ast.unparse(node.test).lower():
            self.evidence["epoch_loop"] = True
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, str):
            token = normalize(node.value)
            if token in {"train", "training"} or "split_train" in token:
                self.evidence["train_split"] = True
            if token in {"validation", "valid", "val", "dev"}:
                self.evidence["validation_split"] = True
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


def inspect_skeleton(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text)
    visitor = SkeletonVisitor()
    visitor.visit(tree)

    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "evidence": visitor.evidence,
        "argparse_defaults": visitor.argparse_defaults,
        "functions": sorted(set(visitor.functions)),
        "classes": sorted(set(visitor.classes)),
        "contains_expected_model_class": EXPECTED_MODEL_CLASS in text,
    }


def find_candidate(
    candidate_document: dict[str, Any],
    skeleton_path: Path,
) -> dict[str, Any] | None:
    expected = str(skeleton_path.resolve())
    for row in candidate_document.get("candidates", []):
        if row.get("entrypoint") == expected:
            return row
    return None


def recursive_scalar_candidates(
    value: Any,
    aliases: tuple[str, ...],
    path: str = "root",
) -> list[dict[str, Any]]:
    rows = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            key_token = normalize(str(key))
            if any(normalize(alias) in key_token for alias in aliases):
                if isinstance(child, (str, int, float, bool)) or child is None:
                    rows.append({"path": child_path, "value": child})
            rows.extend(recursive_scalar_candidates(child, aliases, child_path))
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


def recipe_evidence(
    a4_evidence: dict[str, Any],
    f4_report: dict[str, Any],
    f4m_report: dict[str, Any],
    protocol: dict[str, Any],
    skeleton: dict[str, Any],
) -> dict[str, Any]:
    combined = {
        "A4_evidence": a4_evidence,
        "F4_report": f4_report,
        "F4M_report": f4m_report,
        "F7_protocol": protocol,
    }

    categories = {
        "seed": ("seed",),
        "epochs": ("max_epochs", "epochs", "epoch_budget"),
        "batch_size": ("batch_size", "batch"),
        "learning_rate": ("learning_rate", "lr"),
        "weight_decay": ("weight_decay",),
        "optimizer": ("optimizer",),
        "patience": ("patience", "early_stopping"),
        "selection_score": ("selection_score",),
    }

    results = {}
    for category, aliases in categories.items():
        results[category] = recursive_scalar_candidates(
            combined,
            aliases,
        )

    results["skeleton_argparse_defaults"] = skeleton[
        "argparse_defaults"
    ]
    results["seed_107_evidence_count"] = sum(
        1
        for row in results["seed"]
        if str(row["value"]) in {"107", "107.0"}
    )
    results["selection_formula_frozen"] = protocol[
        "metric_protocol"
    ]["primary_selection_formula"]
    results["selection_formula_matches_expected"] = (
        normalize(results["selection_formula_frozen"])
        == normalize(EXPECTED_SELECTION_FORMULA)
    )
    return results


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
    metric_root = feature_root / "metric_adapter"
    baseline_root = feature_root / "baseline_reproduction"
    permutation_root = feature_root / "permutation"

    p1a_report_path = f7_root / (
        "V5_P3_F7_P1A_MANUAL_TRAINING_ROUTE_PIN_REPORT.json"
    )
    p1a_lock_path = f7_root / (
        "V5_P3_F7_P1A_MANUAL_TRAINING_ROUTE_PIN_LOCK.json"
    )
    p1_report_path = f7_root / (
        "V5_P3_F7_P1_OFFICIAL_TRAINING_ROUTE_RECOVERY_REPORT.json"
    )
    p1_lock_path = f7_root / (
        "V5_P3_F7_P1_OFFICIAL_TRAINING_ROUTE_RECOVERY_LOCK.json"
    )
    f6r_report_path = f6r_root / (
        "V5_P3_F6R_TASK_SPECIFIC_INTEGRATED_GRADIENTS_RESULT_REVIEW_REPORT.json"
    )
    f6r_lock_path = f6r_root / (
        "V5_P3_F6R_TASK_SPECIFIC_INTEGRATED_GRADIENTS_RESULT_REVIEW_LOCK.json"
    )
    f4_report_path = baseline_root / (
        "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_BASELINE_REPRODUCTION_REPORT.json"
    )
    f4_lock_path = baseline_root / (
        "V5_P3_F4_FROZEN_CHECKPOINT_VALIDATION_BASELINE_REPRODUCTION_LOCK.json"
    )
    f4m_report_path = metric_root / (
        "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION_FINAL_REPORT.json"
    )
    f4m_lock_path = metric_root / (
        "V5_P3_F4M_CANONICAL_METRIC_ADAPTER_CERTIFICATION_FINAL_LOCK.json"
    )

    p1a_report, p1a_lock = verify_report_lock(
        p1a_report_path,
        p1a_lock_path,
    )
    p1_report, p1_lock = verify_report_lock(
        p1_report_path,
        p1_lock_path,
    )
    f6r_report, f6r_lock = verify_report_lock(
        f6r_report_path,
        f6r_lock_path,
    )
    f4_report, f4_lock = verify_report_lock(
        f4_report_path,
        f4_lock_path,
    )
    f4m_report, f4m_lock = verify_report_lock(
        f4m_report_path,
        f4m_lock_path,
    )

    require(
        p1a_lock.get("official_trainer_route_resolved") is False,
        "P1A unexpectedly resolved an official trainer",
    )
    require(
        p1a_lock.get("F7_adapter_preflight_authorized") is False,
        "P1A unexpectedly authorized adapter preflight",
    )
    require(
        p1_lock.get("train_count_certified") is True
        and p1_lock.get("validation_count_certified") is True,
        "P1 split counts are not certified",
    )
    require(f6r_lock.get("F6R_complete") is True, "F6R incomplete")
    require(
        f6r_lock.get("F7_protocol_preflight_authorized") is True,
        "F6R did not authorize the F7 protocol preflight",
    )

    skeleton_path = (repo / SKELETON_RELATIVE_PATH).resolve()
    require(
        skeleton_path.is_file(),
        f"reconstructed trainer skeleton missing: {skeleton_path}",
    )
    skeleton = inspect_skeleton(skeleton_path)

    for key, expected in REQUIRED_SKELETON_EVIDENCE.items():
        require(
            skeleton["evidence"].get(key) is expected,
            f"trainer skeleton lacks required evidence: {key}",
        )

    candidate_document_path = f7_root / (
        "F7_P1A_TRANSITIVE_TRAINING_ROUTE_CANDIDATES.json"
    )
    candidate_document = load_json(candidate_document_path)
    skeleton_candidate = find_candidate(
        candidate_document,
        skeleton_path,
    )
    require(
        skeleton_candidate is not None,
        "P1A candidate inventory does not contain the pinned skeleton",
    )
    for key, expected in REQUIRED_SKELETON_EVIDENCE.items():
        require(
            skeleton_candidate["aggregate_evidence"].get(key) is expected,
            f"P1A skeleton candidate lacks required evidence: {key}",
        )
    require(
        skeleton_candidate["aggregate_evidence"].get("model_class") is False,
        "skeleton unexpectedly contains the V6P0 model route",
    )

    route_inventory_path = permutation_root / (
        "F5_P0_EXECUTION_ROUTE_SOURCE_INVENTORY.json"
    )
    route_inventory = load_json(route_inventory_path)
    certified = route_inventory["certified_files"]

    component_rows = {}
    for component in ("model", "loader", "adapter", "checkpoint"):
        row = certified[component]
        path = Path(row["path"]).resolve()
        require(path.is_file(), f"certified {component} missing: {path}")
        require(
            sha256_file(path) == row["actual_sha256"],
            f"certified {component} hash changed",
        )
        component_rows[component] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    model_text = Path(
        component_rows["model"]["path"]
    ).read_text(encoding="utf-8", errors="replace")
    require(
        f"class {EXPECTED_MODEL_CLASS}" in model_text
        or EXPECTED_MODEL_CLASS in model_text,
        "certified model source does not expose expected class",
    )

    split_cert_path = f7_root / (
        "F7_P1_GUARDED_SPLIT_LENGTH_CERTIFICATION.json"
    )
    split_cert = load_json(split_cert_path)
    require(
        int(split_cert["train_length"]) == EXPECTED_TRAIN_ITEMS,
        "train item count changed",
    )
    require(
        int(split_cert["validation_length"]) == EXPECTED_VALIDATION_ITEMS,
        "validation item count changed",
    )

    protocol_path = f7_root / (
        "F7_P0_FROZEN_SAME_WIDTH_ABLATION_PROTOCOL.json"
    )
    protocol = load_json(protocol_path)
    require(
        protocol["training_recipe_policy"]["seed_107_primary_matrix"] is True,
        "primary seed policy changed",
    )
    require(
        protocol["masking_protocol"]["model_parameter_count_preserved"]
        == EXPECTED_PARAMETER_COUNT,
        "parameter count changed",
    )

    a4_evidence_path = f7_root / (
        "F7_P0_A4_TRAINING_PROTOCOL_EVIDENCE.json"
    )
    a4_evidence = load_json(a4_evidence_path)

    recipe = recipe_evidence(
        a4_evidence,
        f4_report,
        f4m_report,
        protocol,
        skeleton,
    )
    require(
        recipe["seed_107_evidence_count"] > 0,
        "no seed-107 evidence found in frozen lineage",
    )
    require(
        recipe["selection_formula_matches_expected"] is True,
        "selection formula changed",
    )

    composite_route = {
        "route_type": "artifact_anchored_reconstructed_F7_training_route",
        "status": "FROZEN",
        "scientific_interpretation": (
            "The exact monolithic A4 trainer source is unavailable. F7 will "
            "therefore use a separately declared reconstructed route composed "
            "of the verified V5-P2 optimization-loop skeleton plus the "
            "certified V6P0 Dynamic70 model, guarded Tranche-A loader, "
            "canonical metric adapter, A4/F4 checkpoint-selection lineage and "
            "the frozen F7 masking protocol."
        ),
        "not_claimed": [
            "byte-identical reproduction of the missing A4 trainer",
            "independent validation",
            "feature-removal authorization",
        ],
        "components": {
            "training_loop_skeleton": {
                "path": str(skeleton_path),
                "sha256": skeleton["sha256"],
                "evidence": skeleton["evidence"],
            },
            **component_rows,
        },
        "dataset": {
            "symlink": str(data_link),
            "target": str(data_link.resolve()),
            "train_items": EXPECTED_TRAIN_ITEMS,
            "validation_items": EXPECTED_VALIDATION_ITEMS,
            "sealed_test_access": False,
        },
        "model": {
            "class": EXPECTED_MODEL_CLASS,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
        },
        "seed": EXPECTED_SEED,
        "selection_formula": EXPECTED_SELECTION_FORMULA,
        "masking_protocol": protocol["masking_protocol"],
        "run_matrix": protocol["run_matrix"],
        "allowed_next_action": (
            "Construct an isolated F7 adapter around this composite route and "
            "perform no-mask plus one masked single-batch forward/backward dry "
            "runs. No scientific checkpoint or metric may be produced in P2."
        ),
    }

    route_path = output_dir / (
        "F7_P1B_FROZEN_RECONSTRUCTED_TRAINING_ROUTE.json"
    )
    recipe_path = output_dir / (
        "F7_P1B_ARTIFACT_ANCHORED_TRAINING_RECIPE_EVIDENCE.json"
    )
    decision_path = output_dir / (
        "F7_P1B_RECONSTRUCTED_ROUTE_DECISION.json"
    )

    atomic_json(route_path, composite_route)
    atomic_json(recipe_path, recipe)

    adapter_preflight_authorized = True
    next_stage = (
        "V5_P3_F7_P2_RECONSTRUCTED_TRAINER_ADAPTER_AND_DRY_RUN_PREFLIGHT"
    )

    decision = {
        "P1A_official_route_unresolved": True,
        "reconstructed_route_frozen": True,
        "reconstructed_route_is_declared_non_byte_identical_to_A4": True,
        "training_loop_skeleton_verified": True,
        "certified_model_verified": True,
        "certified_loader_verified": True,
        "canonical_metric_adapter_verified": True,
        "split_counts_certified": True,
        "seed_107_lineage_verified": True,
        "selection_formula_verified": True,
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
            "Resolve the absence of a monolithic official A4 trainer by "
            "pinning a transparently reconstructed, artifact-anchored F7 "
            "training route: verify the complete V5-P2 optimization-loop "
            "skeleton, certified V6P0 model, guarded loader, canonical metric "
            "adapter, checkpoint, split counts, seed-107 lineage, selection "
            "formula and frozen same-width masking protocol; then authorize "
            "only an isolated adapter dry-run preflight."
        ),
        "finding": {
            "official_A4_trainer_source_available": False,
            "skeleton": skeleton,
            "P1A_skeleton_candidate": skeleton_candidate,
            "components": component_rows,
            "recipe_evidence": recipe,
            "train_items": EXPECTED_TRAIN_ITEMS,
            "validation_items": EXPECTED_VALIDATION_ITEMS,
        },
        "decision": decision,
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "dataset_getitem_called": False,
            "training_feature_tensors_loaded": False,
            "validation_feature_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "scientific_protocol_changed": False,
            "missing_A4_trainer_concealed": False,
            "actual_F7_retraining_authorized": False,
        },
        "artifacts": {
            "frozen_reconstructed_route": str(route_path),
            "training_recipe_evidence": str(recipe_path),
            "decision": str(decision_path),
        },
        "provenance": {
            "P1A_report_sha256": sha256_file(p1a_report_path),
            "P1A_lock_sha256": sha256_file(p1a_lock_path),
            "P1_report_sha256": sha256_file(p1_report_path),
            "P1_lock_sha256": sha256_file(p1_lock_path),
            "F6R_report_sha256": sha256_file(f6r_report_path),
            "F6R_lock_sha256": sha256_file(f6r_lock_path),
            "F4_report_sha256": sha256_file(f4_report_path),
            "F4_lock_sha256": sha256_file(f4_lock_path),
            "F4M_report_sha256": sha256_file(f4m_report_path),
            "F4M_lock_sha256": sha256_file(f4m_lock_path),
            "P1A_candidate_document_sha256": sha256_file(
                candidate_document_path
            ),
            "route_inventory_sha256": sha256_file(route_inventory_path),
            "split_certification_sha256": sha256_file(split_cert_path),
            "F7_protocol_sha256": sha256_file(protocol_path),
            "A4_evidence_sha256": sha256_file(a4_evidence_path),
            "installed_script_sha256": sha256_file(installed_script),
            "frozen_route_sha256": sha256_file(route_path),
            "recipe_evidence_sha256": sha256_file(recipe_path),
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
            "frozen_route_sha256": sha256_file(route_path),
            "recipe_evidence_sha256": sha256_file(recipe_path),
            "decision_sha256": sha256_file(decision_path),
            "reconstructed_route_frozen": True,
            "official_A4_trainer_source_available": False,
            "F7_adapter_preflight_authorized": True,
            "actual_F7_retraining_authorized": False,
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
    print("official_A4_trainer_source_available=false")
    print("reconstructed_route_frozen=true")
    print(
        "training_loop_skeleton="
        f"{skeleton_path}"
    )
    print(
        "training_loop_skeleton_sha256="
        f"{skeleton['sha256']}"
    )
    print(
        "training_loop_skeleton_evidence="
        f"{skeleton['evidence']}"
    )
    print(f"certified_model={component_rows['model']['path']}")
    print(f"certified_loader={component_rows['loader']['path']}")
    print(f"canonical_metric_adapter={component_rows['adapter']['path']}")
    print(f"certified_checkpoint={component_rows['checkpoint']['path']}")
    print(f"train_items={EXPECTED_TRAIN_ITEMS}")
    print(f"validation_items={EXPECTED_VALIDATION_ITEMS}")
    print("seed_107_lineage_verified=true")
    print("selection_formula_verified=true")
    print("reconstructed_route_is_byte_identical_to_A4=false")
    print("F7_adapter_preflight_authorized=true")
    print("actual_F7_retraining_authorized=false")
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
        "V5_P3_F7_P2_RECONSTRUCTED_TRAINER_ADAPTER_AND_DRY_RUN_PREFLIGHT"
    )
    print(f"frozen_reconstructed_route={route_path}")
    print(f"training_recipe_evidence={recipe_path}")
    print(f"decision={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
