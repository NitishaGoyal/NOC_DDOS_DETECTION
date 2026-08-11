from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "V5_P3_F7_E1_R4_R1_R1_ALIAS_PROPAGATION_RECOVERY"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "TRAIN-DERIVED"

SKELETON_RELATIVE_PATH = Path(
    "scripts/v5/p2/train_v5_p2_b2_single_seed.py"
)
EXPECTED_SKELETON_SHA256 = (
    "8fa354698056653a4ff9d467047fced850ff36c3bdef557b08cbd58aa3d3ae1c"
)

TARGET_ARGUMENTS = (
    "root",
    "b1_dir",
    "b0_r3_dir",
    "loader_path",
    "model_path",
    "model_dir",
)

NAMESPACE_NAMES = {
    "args",
    "arg",
    "cfg",
    "config",
    "options",
    "opts",
    "cli",
    "parsed",
    "namespace",
}

PATH_PRESERVING_CALLS = {
    "Path",
    "str",
    "os.fspath",
    "abspath",
    "realpath",
    "resolve",
    "expanduser",
    "absolute",
}

DYNAMIC_IMPORT_CALLS = {
    "importlib.import_module",
    "importlib.util.spec_from_file_location",
    "spec_from_file_location",
    "SourceFileLoader",
    "load_module",
    "exec_module",
    "runpy.run_path",
    "run_path",
}

PYTHON_EXECUTION_CALLS = {
    "exec",
    "eval",
    "compile",
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
    "os.system",
}

MODEL_OR_CHECKPOINT_LOAD_CALLS = {
    "torch.load",
    "load_checkpoint",
    "load_model",
    "load_state_dict",
    "load_state_dict_from_file",
}

FILE_READ_CALLS = {
    "open",
    "read_text",
    "read_bytes",
    "json.load",
    "json.loads",
    "yaml.safe_load",
    "np.load",
    "numpy.load",
    "torch.load",
}

HASH_CALLS = {
    "sha256_file",
    "sha256",
    "hashlib.sha256",
    "md5",
    "hash_file",
}

COPY_CALLS = {
    "shutil.copy",
    "shutil.copy2",
    "copy",
    "copy2",
    "install",
}

OUTPUT_CALLS = {
    "mkdir",
    "makedirs",
    "write_text",
    "write_bytes",
    "json.dump",
    "torch.save",
    "save",
    "save_checkpoint",
    "unlink",
    "replace",
    "rename",
}

EXISTENCE_CALLS = {
    "exists",
    "is_file",
    "is_dir",
    "resolve",
    "expanduser",
    "stat",
    "samefile",
}

PROVENANCE_CALLS = {
    "append",
    "extend",
    "update",
    "setdefault",
    "json.dumps",
    "json.dump",
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


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
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


def target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [ast.unparse(node)]
    if isinstance(node, (ast.Tuple, ast.List)):
        output: list[str] = []
        for child in node.elts:
            output.extend(target_names(child))
        return output
    return []


def build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    output: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            output[child] = parent
    return output


def enclosing_function(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> str | None:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
    return None


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
        f"{line_number:05d}: {line}"
        for line_number, line in enumerate(
            excerpt_lines,
            start=context_start,
        )
    )
    return {
        "path": str(path.resolve()),
        "line_start": start,
        "line_end": end,
        "context_start": context_start,
        "context_end": context_end,
        "raw": raw,
        "raw_sha256": sha256_text(raw),
        "numbered": numbered,
        "numbered_sha256": sha256_text(numbered),
    }


def argparse_inventory(tree: ast.Module) -> list[dict[str, Any]]:
    rows = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if call_name(node.func).rsplit(".", 1)[-1] != "add_argument":
            continue

        options = [
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
        ]
        keywords = {
            keyword.arg: keyword.value
            for keyword in node.keywords
            if keyword.arg is not None
        }

        dest_node = keywords.get("dest")
        if (
            isinstance(dest_node, ast.Constant)
            and isinstance(dest_node.value, str)
        ):
            dest = dest_node.value
        elif options:
            long_options = [
                option
                for option in options
                if option.startswith("--")
            ]
            selected = long_options[-1] if long_options else options[-1]
            dest = selected.lstrip("-").replace("-", "_")
        else:
            continue

        default_node = keywords.get("default")
        try:
            default = (
                ast.literal_eval(default_node)
                if default_node is not None
                else None
            )
        except Exception:
            default = {
                "source": ast.unparse(default_node)
            } if default_node is not None else None

        required_node = keywords.get("required")
        required = bool(
            isinstance(required_node, ast.Constant)
            and required_node.value is True
        )

        rows.append({
            "dest": dest,
            "options": options,
            "required": required,
            "default": default,
            "line_start": int(node.lineno),
            "line_end": int(getattr(node, "end_lineno", node.lineno)),
            "source": ast.unparse(node),
        })

    return rows


def namespace_argument_tag(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in NAMESPACE_NAMES
            and node.attr in TARGET_ARGUMENTS
        ):
            return node.attr

    if isinstance(node, ast.Subscript):
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in NAMESPACE_NAMES
        ):
            try:
                key = ast.literal_eval(node.slice)
            except Exception:
                key = None
            if key in TARGET_ARGUMENTS:
                return str(key)

    return None


def expression_tags(
    node: ast.AST,
    aliases: dict[str, set[str]],
) -> set[str]:
    tags: set[str] = set()

    direct = namespace_argument_tag(node)
    if direct is not None:
        tags.add(direct)

    if isinstance(node, ast.Name):
        tags.update(aliases.get(node.id, set()))
        return tags

    if isinstance(node, ast.Attribute):
        full = ast.unparse(node)
        tags.update(aliases.get(full, set()))

    if isinstance(node, ast.Call):
        function = call_name(node.func)
        short = function.rsplit(".", 1)[-1]

        if short in PATH_PRESERVING_CALLS or function in PATH_PRESERVING_CALLS:
            # Method-style path transformations carry provenance through the
            # receiver, e.g. args.loader_path.expanduser().resolve().
            # R4 omitted this receiver and therefore lost the tag before the
            # assignment to loader_path/model_path/model_dir.
            if isinstance(node.func, ast.Attribute):
                tags.update(
                    expression_tags(
                        node.func.value,
                        aliases,
                    )
                )
            for argument in node.args:
                tags.update(expression_tags(argument, aliases))
            for keyword in node.keywords:
                tags.update(expression_tags(keyword.value, aliases))
            return tags

    for child in ast.iter_child_nodes(node):
        tags.update(expression_tags(child, aliases))

    return tags


def build_alias_map(
    tree: ast.Module,
) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    aliases: dict[str, set[str]] = {}
    assignment_rows: list[dict[str, Any]] = []

    changed = True
    iteration = 0

    while changed and iteration < 30:
        iteration += 1
        changed = False

        for node in ast.walk(tree):
            value = None
            targets: list[str] = []
            kind = None

            if isinstance(node, ast.Assign):
                value = node.value
                for target in node.targets:
                    targets.extend(target_names(target))
                kind = "assign"

            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                value = node.value
                targets.extend(target_names(node.target))
                kind = "annotated_assign"

            elif isinstance(node, ast.NamedExpr):
                value = node.value
                targets.extend(target_names(node.target))
                kind = "named_expression"

            if value is None or not targets:
                continue

            tags = expression_tags(value, aliases)
            if not tags:
                continue

            for target in targets:
                previous = aliases.get(target, set())
                combined = previous | tags
                if combined != previous:
                    aliases[target] = combined
                    changed = True

                row = {
                    "kind": kind,
                    "target": target,
                    "tags": sorted(tags),
                    "line_start": int(getattr(node, "lineno", 0)),
                    "line_end": int(
                        getattr(node, "end_lineno", getattr(node, "lineno", 0))
                    ),
                    "source": ast.unparse(node),
                }
                if row not in assignment_rows:
                    assignment_rows.append(row)

    return aliases, assignment_rows


def classify_call(function: str) -> list[str]:
    short = function.rsplit(".", 1)[-1]
    labels: list[str] = []

    if function in DYNAMIC_IMPORT_CALLS or short in DYNAMIC_IMPORT_CALLS:
        labels.append("dynamic_import")

    if function in PYTHON_EXECUTION_CALLS or short in PYTHON_EXECUTION_CALLS:
        labels.append("python_execution")

    if (
        function in MODEL_OR_CHECKPOINT_LOAD_CALLS
        or short in MODEL_OR_CHECKPOINT_LOAD_CALLS
    ):
        labels.append("model_or_checkpoint_load")

    if function in FILE_READ_CALLS or short in FILE_READ_CALLS:
        labels.append("file_read_or_parse")

    if function in HASH_CALLS or short in HASH_CALLS:
        labels.append("hash_or_identity")

    if function in COPY_CALLS or short in COPY_CALLS:
        labels.append("copy_or_install")

    if function in OUTPUT_CALLS or short in OUTPUT_CALLS:
        labels.append("output_or_mutation")

    if function in EXISTENCE_CALLS or short in EXISTENCE_CALLS:
        labels.append("existence_or_path_normalization")

    if function in PROVENANCE_CALLS or short in PROVENANCE_CALLS:
        labels.append("provenance_or_manifest_recording")

    if not labels:
        labels.append("other_call")

    return labels


def collect_use_sites(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
    aliases: dict[str, set[str]],
    path: Path,
    lines: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()

    for node in ast.walk(tree):
        row: dict[str, Any] | None = None

        if isinstance(node, ast.Call):
            tagged_arguments = []
            all_tags: set[str] = set()

            for index, argument in enumerate(node.args):
                tags = expression_tags(argument, aliases)
                if tags:
                    tagged_arguments.append({
                        "route": "positional",
                        "position": index,
                        "expression": ast.unparse(argument),
                        "tags": sorted(tags),
                    })
                    all_tags.update(tags)

            for keyword in node.keywords:
                tags = expression_tags(keyword.value, aliases)
                if tags:
                    tagged_arguments.append({
                        "route": "keyword",
                        "keyword": keyword.arg,
                        "expression": ast.unparse(keyword.value),
                        "tags": sorted(tags),
                    })
                    all_tags.update(tags)

            receiver_tags: set[str] = set()
            if isinstance(node.func, ast.Attribute):
                receiver_tags = expression_tags(node.func.value, aliases)
                all_tags.update(receiver_tags)

            if all_tags:
                function = call_name(node.func)
                row = {
                    "node_type": "Call",
                    "function": function,
                    "classifications": classify_call(function),
                    "tags": sorted(all_tags),
                    "receiver_tags": sorted(receiver_tags),
                    "tagged_arguments": tagged_arguments,
                    "source": ast.unparse(node),
                }

        elif isinstance(node, ast.Compare):
            tags = expression_tags(node, aliases)
            if tags:
                row = {
                    "node_type": "Compare",
                    "classifications": ["comparison_or_manifest_gate"],
                    "tags": sorted(tags),
                    "source": ast.unparse(node),
                }

        elif isinstance(node, ast.JoinedStr):
            tags = expression_tags(node, aliases)
            if tags:
                row = {
                    "node_type": "JoinedStr",
                    "classifications": ["string_or_log_recording"],
                    "tags": sorted(tags),
                    "source": ast.unparse(node),
                }

        elif isinstance(node, ast.Dict):
            tags = expression_tags(node, aliases)
            if tags:
                row = {
                    "node_type": "Dict",
                    "classifications": ["provenance_or_manifest_recording"],
                    "tags": sorted(tags),
                    "source": ast.unparse(node),
                }

        elif isinstance(node, ast.Return):
            tags = expression_tags(node, aliases)
            if tags:
                row = {
                    "node_type": "Return",
                    "classifications": ["return_or_propagation"],
                    "tags": sorted(tags),
                    "source": ast.unparse(node),
                }

        if row is None:
            continue

        key = (
            int(getattr(node, "lineno", 0)),
            int(getattr(node, "end_lineno", getattr(node, "lineno", 0))),
            row["source"],
        )
        if key in seen:
            continue
        seen.add(key)

        row.update({
            "function_scope": enclosing_function(node, parents),
            "line_start": key[0],
            "line_end": key[1],
            "excerpt": source_excerpt(path, lines, node),
        })
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["line_start"],
            row["line_end"],
            row["node_type"],
        )
    )
    return rows


def summarize_argument(
    argument: str,
    use_sites: list[dict[str, Any]],
    aliases: dict[str, set[str]],
    assignment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [
        row
        for row in use_sites
        if argument in row["tags"]
    ]
    assignments = [
        row
        for row in assignment_rows
        if argument in row["tags"]
    ]
    alias_names = sorted(
        name
        for name, tags in aliases.items()
        if argument in tags
    )

    classifications = sorted({
        classification
        for row in rows
        for classification in row["classifications"]
    })

    dynamic = bool({
        "dynamic_import",
        "python_execution",
        "model_or_checkpoint_load",
    } & set(classifications))

    comparison = "comparison_or_manifest_gate" in classifications
    hash_use = "hash_or_identity" in classifications
    read_use = "file_read_or_parse" in classifications
    copy_use = "copy_or_install" in classifications
    output_use = "output_or_mutation" in classifications
    existence_use = (
        "existence_or_path_normalization" in classifications
    )
    provenance_use = bool({
        "provenance_or_manifest_recording",
        "string_or_log_recording",
    } & set(classifications))

    if not rows and not assignments:
        classification = "UNUSED_REQUIRED_CLI_ARGUMENT"

    elif argument == "model_dir" and output_use and not dynamic:
        classification = "ISOLATED_OUTPUT_DIRECTORY"

    elif dynamic:
        classification = "DYNAMIC_RUNTIME_DEPENDENCY"

    elif comparison:
        classification = "FROZEN_MANIFEST_OR_IDENTITY_GATE"

    elif (
        (hash_use or read_use or copy_use or existence_use or provenance_use)
        and not output_use
    ):
        classification = "PROVENANCE_OR_COMPATIBILITY_ONLY"

    elif (
        output_use
        and argument in {"root", "b1_dir", "b0_r3_dir"}
        and not dynamic
    ):
        classification = "MIXED_DIRECTORY_COMPATIBILITY_ROUTE"

    else:
        classification = "MIXED_OR_UNRESOLVED_USE"

    return {
        "argument": argument,
        "classification": classification,
        "alias_names": alias_names,
        "assignment_count": len(assignments),
        "assignments": assignments,
        "use_site_count": len(rows),
        "classifications_observed": classifications,
        "flags": {
            "dynamic_runtime_use": dynamic,
            "comparison_or_manifest_gate": comparison,
            "hash_use": hash_use,
            "file_read_use": read_use,
            "copy_use": copy_use,
            "output_or_mutation_use": output_use,
            "existence_or_path_normalization": existence_use,
            "provenance_or_log_use": provenance_use,
        },
        "use_sites": rows,
    }


def decide_next_stage(
    summaries: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    loader_model = [
        summaries["loader_path"],
        summaries["model_path"],
    ]

    if any(
        row["classification"] == "DYNAMIC_RUNTIME_DEPENDENCY"
        for row in loader_model
    ):
        return (
            "V5_P3_F7_E1_R5_RUNTIME_IMPORT_USE_SITE_PATCH",
            "loader_path or model_path participates in dynamic import, "
            "execution, or model/checkpoint loading",
        )

    if any(
        row["classification"] == "FROZEN_MANIFEST_OR_IDENTITY_GATE"
        for row in loader_model
    ):
        return (
            "V5_P3_F7_E1_R5_EXACT_MANIFEST_EXPECTATION_EXTRACTION",
            "loader_path or model_path is checked by a comparison or "
            "frozen identity gate",
        )

    if all(
        row["classification"] in {
            "PROVENANCE_OR_COMPATIBILITY_ONLY",
            "UNUSED_REQUIRED_CLI_ARGUMENT",
        }
        for row in loader_model
    ):
        return (
            "V5_P3_F7_E1_R5_COMPATIBILITY_PROVENANCE_BINDING_AND_E2_GENERATION",
            "loader_path and model_path are compatibility/provenance-only "
            "or unused after constructor replacement",
        )

    return (
        "V5_P3_F7_E1_R5_TARGETED_MANUAL_BINDING_REVIEW",
        "loader/model path use remains mixed or unresolved",
    )


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

    f7_root = (
        repo
        / "reports/v5/p3_experiments/f0_d70_feature_study/"
        "retrained_group_ablation"
    )

    r1_report_path = f7_root / (
        "V5_P3_F7_E0D_R1_EXACT_EARLY_STOP_POLICY_PIN_REPORT.json"
    )
    r1_lock_path = f7_root / (
        "V5_P3_F7_E0D_R1_EXACT_EARLY_STOP_POLICY_PIN_LOCK.json"
    )
    p2_lock_path = f7_root / (
        "V5_P3_F7_P2_RECONSTRUCTED_TRAINER_ADAPTER_AND_DRY_RUN_PREFLIGHT_LOCK.json"
    )
    prior_r4_report_path = f7_root / (
        "V5_P3_F7_E1_R4_LEGACY_PATH_ARGUMENT_USE_SITE_AUDIT_REPORT.json"
    )
    prior_r4_lock_path = f7_root / (
        "V5_P3_F7_E1_R4_LEGACY_PATH_ARGUMENT_USE_SITE_AUDIT_LOCK.json"
    )

    prior_r4_report, prior_r4_lock = verify_report_lock(
        prior_r4_report_path,
        prior_r4_lock_path,
    )
    require(
        prior_r4_lock.get("R4_complete") is True,
        "prior R4 audit is incomplete",
    )
    require(
        prior_r4_report["finding"]["arguments"]["loader_path"][
            "alias_names"
        ] == [],
        "prior R4 did not exhibit the alias-propagation false negative",
    )

    _, r1_lock = verify_report_lock(
        r1_report_path,
        r1_lock_path,
    )
    p2_lock = load_json(p2_lock_path)

    require(
        r1_lock.get("final_execution_recipe_frozen") is True,
        "final execution recipe is not frozen",
    )
    require(
        r1_lock.get("E1_trainer_generation_authorized") is True,
        "E1 generation is not authorized",
    )
    require(
        r1_lock.get("actual_scientific_training_started") is False,
        "scientific training already started",
    )
    require(
        p2_lock.get("actual_F7_retraining_authorized") is True,
        "P2 did not authorize F7",
    )
    require(
        p2_lock.get("sealed_test_tensors_loaded") is False,
        "sealed-test access detected",
    )

    skeleton_path = (repo / SKELETON_RELATIVE_PATH).resolve()
    require(skeleton_path.is_file(), f"missing skeleton: {skeleton_path}")
    require(
        sha256_file(skeleton_path) == EXPECTED_SKELETON_SHA256,
        "training skeleton hash changed",
    )

    text = skeleton_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    lines = text.splitlines()
    tree = ast.parse(text)
    parents = build_parent_map(tree)

    cli_rows = argparse_inventory(tree)
    target_cli = {
        argument: [
            row for row in cli_rows
            if normalize(row["dest"]) == normalize(argument)
        ]
        for argument in TARGET_ARGUMENTS
    }

    for argument, rows in target_cli.items():
        require(
            len(rows) == 1,
            f"required CLI argument {argument!r} is not unique: {rows}",
        )

    aliases, assignment_rows = build_alias_map(tree)

    for argument in TARGET_ARGUMENTS:
        require(
            argument in aliases,
            f"alias propagation failed to create local variable {argument!r}",
        )
        require(
            argument in aliases[argument],
            f"local variable {argument!r} lost its originating CLI tag: "
            f"{sorted(aliases[argument])}",
        )

    use_sites = collect_use_sites(
        tree,
        parents,
        aliases,
        skeleton_path,
        lines,
    )

    summaries = {
        argument: summarize_argument(
            argument,
            use_sites,
            aliases,
            assignment_rows,
        )
        for argument in TARGET_ARGUMENTS
    }

    next_stage, next_reason = decide_next_stage(summaries)

    summary_path = output_dir / (
        "F7_E1_R4_R1_LEGACY_PATH_ARGUMENT_USE_SITE_SUMMARY.json"
    )
    full_audit_path = output_dir / (
        "F7_E1_R4_R1_LEGACY_PATH_ARGUMENT_FULL_STATIC_AUDIT.json"
    )
    excerpts_path = output_dir / (
        "F7_E1_R4_R1_LEGACY_PATH_ARGUMENT_EXACT_SOURCE_EXCERPTS.txt"
    )
    decision_path = output_dir / (
        "F7_E1_R4_R1_NEXT_BINDING_ROUTE_DECISION.json"
    )

    summary_document = {
        "recovery_classification": (
            "R4_path_preserving_method_receiver_was_not_traversed"
        ),
        "alias_propagation_fix": (
            "propagate tags through node.func.value for method-style "
            "expanduser/resolve/absolute/path-normalization calls"
        ),
        "direct_local_aliases_verified": {
            argument: sorted(aliases[argument])
            for argument in TARGET_ARGUMENTS
        },
        "skeleton": str(skeleton_path),
        "skeleton_sha256": sha256_file(skeleton_path),
        "arguments": {
            argument: {
                "cli": target_cli[argument][0],
                "classification": summaries[argument]["classification"],
                "flags": summaries[argument]["flags"],
                "alias_names": summaries[argument]["alias_names"],
                "assignment_count": summaries[argument][
                    "assignment_count"
                ],
                "use_site_count": summaries[argument]["use_site_count"],
                "classifications_observed": summaries[argument][
                    "classifications_observed"
                ],
            }
            for argument in TARGET_ARGUMENTS
        },
        "next_stage": next_stage,
        "next_stage_reason": next_reason,
    }
    atomic_json(summary_path, summary_document)

    full_audit = {
        "argparse_inventory": cli_rows,
        "target_cli_arguments": target_cli,
        "alias_map": {
            name: sorted(tags)
            for name, tags in sorted(aliases.items())
        },
        "alias_assignments": assignment_rows,
        "use_sites": use_sites,
        "argument_summaries": summaries,
    }
    atomic_json(full_audit_path, full_audit)

    excerpt_blocks = []
    for argument in TARGET_ARGUMENTS:
        summary = summaries[argument]
        excerpt_blocks.append(
            "=" * 88
            + "\n"
            + f"ARGUMENT: {argument}\n"
            + f"CLASSIFICATION: {summary['classification']}\n"
            + f"ALIASES: {summary['alias_names']}\n"
            + f"USE SITES: {summary['use_site_count']}\n"
            + "=" * 88
        )

        if not summary["use_sites"]:
            excerpt_blocks.append("<no use site found>")
            continue

        for index, row in enumerate(summary["use_sites"], start=1):
            excerpt_blocks.append(
                f"\n[{argument} use {index}]\n"
                f"scope={row['function_scope']}\n"
                f"lines={row['line_start']}-{row['line_end']}\n"
                f"node_type={row['node_type']}\n"
                f"classifications={row['classifications']}\n"
                f"source={row['source']}\n"
                f"{row['excerpt']['numbered']}"
            )

    atomic_text(
        excerpts_path,
        "\n\n".join(excerpt_blocks) + "\n",
    )

    decision = {
        "R4_static_audit_complete": True,
        "loader_path_classification": summaries["loader_path"][
            "classification"
        ],
        "model_path_classification": summaries["model_path"][
            "classification"
        ],
        "model_dir_classification": summaries["model_dir"][
            "classification"
        ],
        "next_stage": next_stage,
        "next_stage_reason": next_reason,
        "E2_generated_trainer_preflight_authorized": False,
        "primary_matrix_execution_authorized": False,
        "actual_scientific_training_started": False,
        "actual_F7_primary_matrix_execution_started": False,
        "F8_multi_seed_authorized": False,
        "feature_removal_authorized": False,
        "compact_interface_equivalence_authorized": False,
        "hardware_reduction_claim_authorized": False,
        "sealed_test_access": False,
    }
    atomic_json(decision_path, decision)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "scope": (
            "Recover the R4 alias-analysis false negative by propagating "
            "argument provenance through method receivers in chained path "
            "normalization expressions such as "
            "args.loader_path.expanduser().resolve(); verify that all six "
            "local path variables retain their originating CLI tags; then "
            "repeat the complete static sink, comparison, import, read, hash, "
            "copy, output and manifest audit. No pointed-to path is opened or "
            "executed."
        ),
        "finding": summary_document,
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
            "legacy_loader_path_opened": False,
            "legacy_model_path_opened": False,
            "optimizer_step_performed": False,
            "scientific_checkpoint_saved": False,
            "scientific_training_started": False,
            "primary_matrix_execution_authorized": False,
        },
        "artifacts": {
            "summary": str(summary_path),
            "full_static_audit": str(full_audit_path),
            "exact_source_excerpts": str(excerpts_path),
            "decision": str(decision_path),
        },
        "provenance": {
            "prior_R4_report_sha256": sha256_file(
                prior_r4_report_path
            ),
            "prior_R4_lock_sha256": sha256_file(
                prior_r4_lock_path
            ),
            "E0D_R1_report_sha256": sha256_file(r1_report_path),
            "E0D_R1_lock_sha256": sha256_file(r1_lock_path),
            "P2_lock_sha256": sha256_file(p2_lock_path),
            "skeleton_sha256": sha256_file(skeleton_path),
            "installed_script_sha256": sha256_file(installed_script),
            "summary_sha256": sha256_file(summary_path),
            "full_audit_sha256": sha256_file(full_audit_path),
            "excerpts_sha256": sha256_file(excerpts_path),
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
            "summary_sha256": sha256_file(summary_path),
            "full_audit_sha256": sha256_file(full_audit_path),
            "excerpts_sha256": sha256_file(excerpts_path),
            "decision_sha256": sha256_file(decision_path),
            "R4_R1_complete": True,
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
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print(
        "recovery_classification="
        "R4_path_preserving_method_receiver_was_not_traversed"
    )
    print("alias_receiver_propagation_fixed=true")
    for argument in TARGET_ARGUMENTS:
        print(
            "verified_direct_alias="
            f"{argument}:tags={sorted(aliases[argument])}"
        )
    print(f"skeleton={skeleton_path}")
    print(f"skeleton_sha256={sha256_file(skeleton_path)}")

    for argument in TARGET_ARGUMENTS:
        summary = summaries[argument]
        cli = target_cli[argument][0]
        print(
            "legacy_argument_summary="
            f"{argument}:"
            f"option={cli['options']}:"
            f"required={str(cli['required']).lower()}:"
            f"classification={summary['classification']}:"
            f"aliases={summary['alias_names']}:"
            f"use_sites={summary['use_site_count']}:"
            f"observed={summary['classifications_observed']}"
        )

    for argument in ("loader_path", "model_path", "model_dir"):
        summary = summaries[argument]
        print(f"{argument}_exact_use_sites_begin")
        if not summary["use_sites"]:
            print("<no use site found>")
        for index, row in enumerate(summary["use_sites"], start=1):
            print(
                f"use_{index}:"
                f"scope={row['function_scope']}:"
                f"lines={row['line_start']}-{row['line_end']}:"
                f"classifications={row['classifications']}:"
                f"source={row['source']}"
            )
            print(row["excerpt"]["numbered"])
        print(f"{argument}_exact_use_sites_end")

    print(f"next_stage={next_stage}")
    print(f"next_stage_reason={next_reason}")
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
    print("sealed_test_tensors_loaded=false")
    print(f"summary={summary_path}")
    print(f"full_static_audit={full_audit_path}")
    print(f"exact_source_excerpts={excerpts_path}")
    print(f"decision={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
