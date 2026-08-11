from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "V5_P3_F7_E1_R5_P0_DYNAMIC_IMPORT_AND_HASH_GATE_PIN"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "TRAIN-DERIVED"

SKELETON_RELATIVE_PATH = Path(
    "scripts/v5/p2/train_v5_p2_b2_single_seed.py"
)
EXPECTED_SKELETON_SHA256 = (
    "8fa354698056653a4ff9d467047fced850ff36c3bdef557b08cbd58aa3d3ae1c"
)

B1_REPORT_NAME = "V5_P2_B1_TRAINING_PROTOCOL_LOCK.json"
B1_LOCK_NAME = "V5_P2_B1_TRAINING_PROTOCOL_LOCK_LOCK.json"
B0_R3_REPORT_NAME = (
    "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT.json"
)
B0_R3_LOCK_NAME = (
    "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT_LOCK.json"
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


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


def source_excerpt(
    path: Path,
    lines: list[str],
    node: ast.AST,
    context: int = 5,
) -> dict[str, Any]:
    start = int(getattr(node, "lineno", 0))
    end = int(getattr(node, "end_lineno", start))
    context_start = max(1, start - context)
    context_end = min(len(lines), end + context)
    excerpt_lines = lines[context_start - 1:context_end]
    raw = "\n".join(excerpt_lines)
    numbered = "\n".join(
        f"{line_no:05d}: {line}"
        for line_no, line in enumerate(
            excerpt_lines,
            start=context_start,
        )
    )
    return {
        "path": str(path),
        "line_start": start,
        "line_end": end,
        "context_start": context_start,
        "context_end": context_end,
        "raw": raw,
        "raw_sha256": sha256_text(raw),
        "numbered": numbered,
        "numbered_sha256": sha256_text(numbered),
    }


def verify_latest_r4_r1(
    f7_root: Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    reports = sorted(
        f7_root.glob(
            "V5_P3_F7_E1_R4_R1*ALIAS_PROPAGATION_RECOVERY_REPORT.json"
        )
    )
    locks = sorted(
        f7_root.glob(
            "V5_P3_F7_E1_R4_R1*ALIAS_PROPAGATION_RECOVERY_LOCK.json"
        )
    )
    require(
        len(reports) == 1,
        f"expected one corrected R4-R1 report, found {reports}",
    )
    require(
        len(locks) == 1,
        f"expected one corrected R4-R1 lock, found {locks}",
    )

    report = load_json(reports[0])
    lock = load_json(locks[0])
    require(report.get("status") == "PASS", "corrected R4-R1 report not PASS")
    require(
        lock.get("report_sha256") == sha256_file(reports[0]),
        "corrected R4-R1 report/lock mismatch",
    )
    require(
        report.get("finding", {}).get(
            "alias_propagation_fix"
        ) is not None,
        "corrected R4-R1 alias fix evidence missing",
    )
    return reports[0], locks[0], report, lock


def find_unique_directory_with_files(
    root: Path,
    names: tuple[str, ...],
) -> Path:
    candidates = []
    first_name = names[0]
    for first in root.rglob(first_name):
        if not first.is_file():
            continue
        directory = first.parent
        if all((directory / name).is_file() for name in names):
            candidates.append(directory.resolve())

    candidates = sorted(set(candidates))
    require(
        len(candidates) == 1,
        f"directory for {names} is not unique: {candidates}",
    )
    return candidates[0]


def canonical_hash_match(
    repo: Path,
    expected_sha256: str,
    *,
    kind: str,
) -> tuple[Path, list[dict[str, Any]]]:
    require(kind in {"loader", "model"}, f"unknown kind: {kind}")
    matches = []

    for path in repo.rglob("*.py"):
        if not path.is_file():
            continue
        try:
            digest = sha256_file(path)
        except OSError:
            continue
        if digest != expected_sha256:
            continue

        relative = str(path.relative_to(repo))
        token = relative.lower()
        score = 0

        if kind == "loader":
            if token.startswith("src/data/"):
                score += 100
            if "loader" in token or "dataset" in token:
                score += 30
        else:
            if token.startswith("src/models/"):
                score += 100
            if "model" in token:
                score += 30

        if "v5_p2" in token:
            score += 40
        if "/reports/" in f"/{token}" or "/artifacts/" in f"/{token}":
            score -= 50
        if "/scripts/" in f"/{token}":
            score -= 20

        matches.append({
            "path": str(path.resolve()),
            "relative": relative,
            "sha256": digest,
            "score": score,
        })

    require(
        matches,
        f"no repository Python file matches {kind} SHA {expected_sha256}",
    )
    matches.sort(key=lambda row: (-row["score"], row["relative"]))
    best_score = matches[0]["score"]
    best = [row for row in matches if row["score"] == best_score]
    require(
        len(best) == 1,
        f"canonical {kind} SHA match is ambiguous: {best}",
    )
    return Path(best[0]["path"]), matches


def assignment_rows(
    tree: ast.Module,
    path: Path,
    lines: list[str],
    targets: set[str],
) -> list[dict[str, Any]]:
    rows = []

    for node in ast.walk(tree):
        value = None
        names = []

        if isinstance(node, ast.Assign):
            value = node.value
            for target in node.targets:
                names.extend(target_names(target))
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            names.extend(target_names(node.target))
        else:
            continue

        selected = sorted(set(names) & targets)
        if not selected or value is None:
            continue

        rows.append({
            "targets": selected,
            "source": ast.unparse(node),
            "value_source": ast.unparse(value),
            "excerpt": source_excerpt(path, lines, node),
        })

    rows.sort(
        key=lambda row: (
            row["excerpt"]["line_start"],
            row["excerpt"]["line_end"],
        )
    )
    return rows


def dynamic_import_rows(
    tree: ast.Module,
    path: Path,
    lines: list[str],
) -> list[dict[str, Any]]:
    rows = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = call_name(node.func)
        if function.rsplit(".", 1)[-1] != "import_module":
            continue
        if not node.args:
            continue

        first = ast.unparse(node.args[0])
        if first not in {"loader_path", "model_path"}:
            continue

        rows.append({
            "path_argument": first,
            "function": function,
            "source": ast.unparse(node),
            "excerpt": source_excerpt(path, lines, node),
        })

    rows.sort(key=lambda row: row["excerpt"]["line_start"])
    return rows


def constructor_rows(
    tree: ast.Module,
    path: Path,
    lines: list[str],
) -> list[dict[str, Any]]:
    rows = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        short = call_name(node.func).rsplit(".", 1)[-1]
        if short not in {"DatasetClass", "ModelClass"}:
            continue

        rows.append({
            "constructor": short,
            "source": ast.unparse(node),
            "excerpt": source_excerpt(path, lines, node),
        })

    rows.sort(key=lambda row: row["excerpt"]["line_start"])
    return rows


def relevant_compare_rows(
    tree: ast.Module,
    path: Path,
    lines: list[str],
) -> list[dict[str, Any]]:
    rows = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        source = ast.unparse(node)
        if not (
            "loader_sha256" in source
            or "model_sha256" in source
            or "corrected_model_sha256" in source
            or "paths['loader']" in source
            or "paths['model']" in source
            or 'paths["loader"]' in source
            or 'paths["model"]' in source
        ):
            continue

        rows.append({
            "source": source,
            "excerpt": source_excerpt(path, lines, node),
        })

    rows.sort(key=lambda row: row["excerpt"]["line_start"])
    return rows


def provenance_hash_rows(
    tree: ast.Module,
    path: Path,
    lines: list[str],
) -> list[dict[str, Any]]:
    rows = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue

        selected_keys = []
        for key in node.keys:
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value in {
                    "loader_sha256",
                    "model_sha256",
                    "model_source_sha256",
                }
            ):
                selected_keys.append(key.value)

        if not selected_keys:
            continue

        source = ast.unparse(node)
        if "loader_path" not in source and "model_path" not in source:
            continue

        rows.append({
            "keys": selected_keys,
            "source": source,
            "excerpt": source_excerpt(path, lines, node),
        })

    rows.sort(key=lambda row: row["excerpt"]["line_start"])
    return rows


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require(repo.is_dir(), f"missing repository: {repo}")
    require(data_link.is_symlink(), f"missing dataset symlink: {data_link}")
    require(data_link.resolve().is_dir(), "dataset symlink target missing")

    f7_root = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/"
        "retrained_group_ablation"
    )

    (
        r4_r1_report_path,
        r4_r1_lock_path,
        r4_r1_report,
        r4_r1_lock,
    ) = verify_latest_r4_r1(f7_root)

    skeleton_path = (repo / SKELETON_RELATIVE_PATH).resolve()
    require(skeleton_path.is_file(), f"missing skeleton: {skeleton_path}")
    require(
        sha256_file(skeleton_path) == EXPECTED_SKELETON_SHA256,
        "skeleton SHA changed",
    )

    b1_dir = (repo / "reports/v5/p2_b1_training_protocol_lock").resolve()
    require(b1_dir.is_dir(), f"missing fixed B1 directory: {b1_dir}")
    require((b1_dir / B1_REPORT_NAME).is_file(), "B1 report missing")
    require((b1_dir / B1_LOCK_NAME).is_file(), "B1 lock missing")

    b0_r3_dir = find_unique_directory_with_files(
        repo / "reports/v5",
        (B0_R3_REPORT_NAME, B0_R3_LOCK_NAME),
    )

    b1_report = load_json(b1_dir / B1_REPORT_NAME)
    b1_lock = load_json(b1_dir / B1_LOCK_NAME)
    b0_r3_report = load_json(b0_r3_dir / B0_R3_REPORT_NAME)
    b0_r3_lock = load_json(b0_r3_dir / B0_R3_LOCK_NAME)

    require(
        b1_report.get("status") == "COMPLETE",
        "B1 report status changed",
    )
    require(
        b0_r3_report.get("status") == "COMPLETE",
        "B0-R3 report status changed",
    )
    require(
        b1_lock.get("report_sha256")
        == sha256_file(b1_dir / B1_REPORT_NAME),
        "B1 report/lock mismatch",
    )
    require(
        b0_r3_lock.get("report_sha256")
        == sha256_file(b0_r3_dir / B0_R3_REPORT_NAME),
        "B0-R3 report/lock mismatch",
    )

    expected_loader_sha = b1_lock.get("loader_sha256")
    expected_b1_model_sha = b1_lock.get("model_sha256")
    expected_b0_model_sha = b0_r3_lock.get("corrected_model_sha256")

    for label, value in (
        ("loader_sha256", expected_loader_sha),
        ("B1 model_sha256", expected_b1_model_sha),
        ("B0-R3 corrected_model_sha256", expected_b0_model_sha),
    ):
        require(
            isinstance(value, str) and len(value) == 64,
            f"invalid frozen {label}: {value!r}",
        )

    require(
        expected_b1_model_sha == expected_b0_model_sha,
        "B1 and B0-R3 model source hashes disagree",
    )

    legacy_loader_path, loader_matches = canonical_hash_match(
        repo,
        expected_loader_sha,
        kind="loader",
    )
    legacy_model_path, model_matches = canonical_hash_match(
        repo,
        expected_b1_model_sha,
        kind="model",
    )

    text = skeleton_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    lines = text.splitlines()
    tree = ast.parse(text)

    imports = dynamic_import_rows(tree, skeleton_path, lines)
    require(
        [row["path_argument"] for row in imports]
        == ["loader_path", "model_path"],
        f"dynamic import route changed: {imports}",
    )

    assignments = assignment_rows(
        tree,
        skeleton_path,
        lines,
        {
            "loader_module",
            "model_module",
            "DatasetClass",
            "ModelClass",
            "train_dataset",
            "validation_dataset",
            "model",
        },
    )

    class_assignments = [
        row
        for row in assignments
        if set(row["targets"]) & {
            "DatasetClass",
            "ModelClass",
        }
    ]
    require(
        any(
            "V5P2PairAlignedPrimary58Dataset" in row["value_source"]
            for row in class_assignments
        ),
        "legacy DatasetClass extraction changed",
    )
    require(
        any(
            "P2B3Conv1DOnlyCount4" in row["value_source"]
            for row in class_assignments
        ),
        "legacy ModelClass extraction changed",
    )

    constructors = constructor_rows(tree, skeleton_path, lines)
    require(
        sum(row["constructor"] == "DatasetClass" for row in constructors) == 2,
        "expected two DatasetClass constructor calls",
    )
    require(
        sum(row["constructor"] == "ModelClass" for row in constructors) == 1,
        "expected one ModelClass constructor call",
    )

    comparisons = relevant_compare_rows(tree, skeleton_path, lines)
    require(
        any("loader_sha256" in row["source"] for row in comparisons),
        "loader SHA identity gate not found",
    )
    require(
        any("model_sha256" in row["source"] for row in comparisons),
        "B1 model SHA identity gate not found",
    )
    require(
        any(
            "corrected_model_sha256" in row["source"]
            for row in comparisons
        ),
        "B0-R3 corrected model SHA identity gate not found",
    )

    provenance_rows = provenance_hash_rows(
        tree,
        skeleton_path,
        lines,
    )
    require(
        provenance_rows,
        "loader/model output provenance hash rows not found",
    )

    patch_plan_path = output_dir / (
        "F7_E1_R5_P0_FROZEN_RUNTIME_IMPORT_PATCH_PLAN.json"
    )
    hash_resolution_path = output_dir / (
        "F7_E1_R5_P0_EXACT_LEGACY_HASH_MATCH_RESOLUTION.json"
    )
    source_map_path = output_dir / (
        "F7_E1_R5_P0_EXACT_DYNAMIC_IMPORT_CLASS_AND_GATE_SOURCE_MAP.json"
    )
    decision_path = output_dir / (
        "F7_E1_R5_P0_RUNTIME_PATCH_GENERATION_DECISION.json"
    )

    hash_resolution = {
        "b1_dir": str(b1_dir),
        "b0_r3_dir": str(b0_r3_dir),
        "expected_loader_sha256": expected_loader_sha,
        "expected_model_sha256": expected_b1_model_sha,
        "B1_and_B0_R3_model_hashes_equal": True,
        "canonical_legacy_loader_path": str(legacy_loader_path),
        "canonical_legacy_model_path": str(legacy_model_path),
        "loader_matches": loader_matches,
        "model_matches": model_matches,
        "resolution_method": (
            "exact SHA-256 equality against frozen B1/B0-R3 locks; "
            "canonical repository source chosen only among exact-content matches"
        ),
    }
    atomic_json(hash_resolution_path, hash_resolution)

    source_map = {
        "dynamic_import_calls": imports,
        "relevant_assignments": assignments,
        "constructor_calls": constructors,
        "frozen_hash_identity_gates": comparisons,
        "output_provenance_hash_rows": provenance_rows,
        "corrected_R4_R1_loader_classification_override": {
            "reported": r4_r1_report["finding"]["arguments"][
                "loader_path"
            ]["classification"],
            "corrected": "DYNAMIC_RUNTIME_DEPENDENCY",
            "reason": (
                "exact source contains import_module(loader_path, ...); "
                "the R4-R1 call classifier failed to recognize the local "
                "helper name import_module as a dynamic import"
            ),
        },
        "corrected_R4_R1_model_classification": (
            "DYNAMIC_RUNTIME_DEPENDENCY"
        ),
        "model_dir_classification": "ISOLATED_OUTPUT_DIRECTORY",
    }
    atomic_json(source_map_path, source_map)

    patch_plan = {
        "status": "FROZEN",
        "historical_A4_trainer_claimed": False,
        "patch_scope": [
            {
                "operation": "remove_runtime_dynamic_import",
                "target": "loader_module = import_module(loader_path, ...)",
                "replacement": (
                    "direct use of F7TrainDatasetAdapter and "
                    "F7ValidationDatasetAdapter"
                ),
            },
            {
                "operation": "remove_runtime_dynamic_import",
                "target": "model_module = import_module(model_path, ...)",
                "replacement": "direct use of F7Dynamic70ModelFactory",
            },
            {
                "operation": "replace_legacy_class_extraction",
                "target": (
                    "DatasetClass=V5P2PairAlignedPrimary58Dataset; "
                    "ModelClass=P2B3Conv1DOnlyCount4"
                ),
                "replacement": (
                    "F7 fixed-split guarded dataset adapters and "
                    "fresh Dynamic70 model factory"
                ),
            },
            {
                "operation": "replace_constructor_calls",
                "targets": [
                    "train_dataset = DatasetClass(...)",
                    "validation_dataset = DatasetClass(...)",
                    "model = ModelClass().to(device)",
                ],
                "replacement": [
                    "train_dataset = F7TrainDatasetAdapter(...)",
                    "validation_dataset = F7ValidationDatasetAdapter(...)",
                    "model = F7Dynamic70ModelFactory().to(device)",
                ],
            },
            {
                "operation": "retain_frozen_legacy_identity_gates",
                "reason": (
                    "B1/B0-R3 source integrity checks remain useful as "
                    "compatibility provenance, but their files are not executed"
                ),
                "loader_path": str(legacy_loader_path),
                "model_path": str(legacy_model_path),
                "loader_sha256": expected_loader_sha,
                "model_sha256": expected_b1_model_sha,
            },
            {
                "operation": "disambiguate_output_provenance",
                "requirement": (
                    "generated checkpoints/reports/locks must label frozen "
                    "P2 source hashes as legacy_compatibility_* and separately "
                    "record the actual F7 runtime-adapter SHA-256"
                ),
            },
            {
                "operation": "isolate_outputs",
                "requirement": (
                    "model_dir and report_dir must be distinct append-only "
                    "subdirectories inside each F7 run directory"
                ),
            },
        ],
        "must_preserve": [
            "compute_loss exact source SHA-256",
            "100 executed epoch maximum",
            "AdamW lr=0.001 weight_decay=0.0001",
            "ReduceLROnPlateau once per epoch",
            "clip_grad_norm_ max_norm=1.0",
            "early stop after 12 consecutive non-improvements, not before epoch 15",
            "train length 110855",
            "validation length 13863",
            "seed 107 initial primary matrix",
            "sealed-test quarantine",
        ],
        "must_remove_from_runtime": [
            "execution of legacy loader module",
            "execution of legacy model module",
            "construction of V5P2PairAlignedPrimary58Dataset",
            "construction of P2B3Conv1DOnlyCount4",
        ],
    }
    atomic_json(patch_plan_path, patch_plan)

    decision = {
        "R5_P0_complete": True,
        "loader_path_runtime_classification": (
            "DYNAMIC_RUNTIME_DEPENDENCY"
        ),
        "model_path_runtime_classification": (
            "DYNAMIC_RUNTIME_DEPENDENCY"
        ),
        "model_dir_classification": "ISOLATED_OUTPUT_DIRECTORY",
        "exact_legacy_source_hashes_resolved": True,
        "runtime_import_patch_plan_frozen": True,
        "R5_runtime_patch_generation_authorized": True,
        "E2_generated_trainer_preflight_authorized": False,
        "primary_matrix_execution_authorized": False,
        "actual_scientific_training_started": False,
        "actual_F7_primary_matrix_execution_started": False,
        "F8_multi_seed_authorized": False,
        "feature_removal_authorized": False,
        "compact_interface_equivalence_authorized": False,
        "hardware_reduction_claim_authorized": False,
        "sealed_test_access": False,
        "next_stage": (
            "V5_P3_F7_E1_R5_RUNTIME_IMPORT_USE_SITE_PATCH_"
            "AND_GENERATED_TRAINER_RECOVERY"
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
            "Pin the exact legacy loader/model dynamic-import calls, class "
            "extractions, constructor calls, B1/B0-R3 source-hash gates and "
            "output provenance uses. Resolve the exact compatibility source "
            "files by frozen SHA-256 equality rather than path guessing. "
            "Freeze a patch plan that removes legacy source execution while "
            "retaining frozen identity checks as clearly labelled compatibility "
            "provenance. No model, checkpoint, dataset or feature tensor is loaded."
        ),
        "finding": {
            "hash_resolution": hash_resolution,
            "source_map": source_map,
            "patch_plan": patch_plan,
        },
        "decision": decision,
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "training_dataset_constructed": False,
            "validation_dataset_constructed": False,
            "training_feature_tensors_loaded": False,
            "validation_feature_tensors_loaded": False,
            "legacy_loader_module_executed": False,
            "legacy_model_module_executed": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "optimizer_step_performed": False,
            "scientific_checkpoint_saved": False,
            "scientific_training_started": False,
            "primary_matrix_execution_authorized": False,
        },
        "artifacts": {
            "patch_plan": str(patch_plan_path),
            "hash_resolution": str(hash_resolution_path),
            "source_map": str(source_map_path),
            "decision": str(decision_path),
        },
        "provenance": {
            "corrected_R4_R1_report": str(r4_r1_report_path),
            "corrected_R4_R1_report_sha256": sha256_file(
                r4_r1_report_path
            ),
            "corrected_R4_R1_lock": str(r4_r1_lock_path),
            "corrected_R4_R1_lock_sha256": sha256_file(
                r4_r1_lock_path
            ),
            "skeleton_sha256": sha256_file(skeleton_path),
            "B1_report_sha256": sha256_file(b1_dir / B1_REPORT_NAME),
            "B1_lock_sha256": sha256_file(b1_dir / B1_LOCK_NAME),
            "B0_R3_report_sha256": sha256_file(
                b0_r3_dir / B0_R3_REPORT_NAME
            ),
            "B0_R3_lock_sha256": sha256_file(
                b0_r3_dir / B0_R3_LOCK_NAME
            ),
            "installed_script_sha256": sha256_file(installed_script),
            "patch_plan_sha256": sha256_file(patch_plan_path),
            "hash_resolution_sha256": sha256_file(
                hash_resolution_path
            ),
            "source_map_sha256": sha256_file(source_map_path),
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
            "patch_plan_sha256": sha256_file(patch_plan_path),
            "hash_resolution_sha256": sha256_file(
                hash_resolution_path
            ),
            "source_map_sha256": sha256_file(source_map_path),
            "decision_sha256": sha256_file(decision_path),
            "R5_P0_complete": True,
            "R5_runtime_patch_generation_authorized": True,
            "E2_generated_trainer_preflight_authorized": False,
            "primary_matrix_execution_authorized": False,
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
            "legacy_loader_module_executed": False,
            "legacy_model_module_executed": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print(
        "corrected_loader_path_classification="
        "DYNAMIC_RUNTIME_DEPENDENCY"
    )
    print(
        "corrected_model_path_classification="
        "DYNAMIC_RUNTIME_DEPENDENCY"
    )
    print("model_dir_classification=ISOLATED_OUTPUT_DIRECTORY")
    print(f"skeleton={skeleton_path}")
    print(f"skeleton_sha256={sha256_file(skeleton_path)}")
    print(f"b1_dir={b1_dir}")
    print(f"b0_r3_dir={b0_r3_dir}")
    print(f"expected_loader_sha256={expected_loader_sha}")
    print(f"expected_model_sha256={expected_b1_model_sha}")
    print(f"canonical_legacy_loader_path={legacy_loader_path}")
    print(f"canonical_legacy_model_path={legacy_model_path}")

    for row in imports:
        print(
            "dynamic_import_pin="
            f"argument={row['path_argument']}:"
            f"lines={row['excerpt']['line_start']}-"
            f"{row['excerpt']['line_end']}:"
            f"source={row['source']}"
        )

    for row in comparisons:
        print(
            "frozen_hash_gate_pin="
            f"lines={row['excerpt']['line_start']}-"
            f"{row['excerpt']['line_end']}:"
            f"source={row['source']}"
        )

    print("runtime_import_patch_plan_frozen=true")
    print("R5_runtime_patch_generation_authorized=true")
    print("E2_generated_trainer_preflight_authorized=false")
    print("primary_matrix_execution_authorized=false")
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
    print("legacy_loader_module_executed=false")
    print("legacy_model_module_executed=false")
    print("sealed_test_tensors_loaded=false")
    print(
        "next_stage="
        "V5_P3_F7_E1_R5_RUNTIME_IMPORT_USE_SITE_PATCH_"
        "AND_GENERATED_TRAINER_RECOVERY"
    )
    print(f"patch_plan={patch_plan_path}")
    print(f"hash_resolution={hash_resolution_path}")
    print(f"source_map={source_map_path}")
    print(f"decision={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
