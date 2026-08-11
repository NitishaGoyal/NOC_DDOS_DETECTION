#!/usr/bin/env python3
"""
V4-A3.13-P
Frozen Local-vs-Graph Attribution Preflight

This preflight discovers the exact frozen A3 model implementation and checkpoint
schema before any branch-level instrumentation is attempted.

Why this is a separate gate
---------------------------
A3.13 must compare the actual h_local and h_graph tensors from the frozen model.
Reimplementing the architecture from memory or guessing module names would break
checkpoint provenance. This script therefore:

  * verifies A3.12 and frozen checkpoint provenance
  * inventories the checkpoint/state_dict
  * searches the repository for candidate model classes
  * parses candidate Python files with AST
  * reports likely temporal/local/graph/node-head module families
  * determines whether exact instrumentation can be generated safely

No model training occurs.
No validation or test predictions are evaluated.
No policy selection occurs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch


SEARCH_TOKENS = (
    "sourcepreserve",
    "countaware",
    "rank_mask",
    "local_proj",
    "local_projection",
    "node_head",
    "attacker_head",
    "graph_head",
    "count_head",
    "gcn",
    "graphconv",
    "temporal",
)

MODULE_FAMILIES = {
    "temporal": (
        "conv",
        "temporal",
        "time",
    ),
    "local_projection": (
        "local_proj",
        "local_projection",
        "pre_gcn",
        "node_proj",
    ),
    "graph": (
        "gcn",
        "graph",
        "conv_g",
        "message",
    ),
    "node_head": (
        "node_head",
        "attacker_head",
        "node_classifier",
        "localization",
    ),
    "graph_head": (
        "graph_head",
        "graph_classifier",
        "detector",
    ),
    "count_head": (
        "count_head",
        "count_classifier",
        "cardinality",
    ),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, Path):
        return str(value)
    return value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def top_level_inventory(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return {
            str(key): jsonable(value)
            for key, value in obj.items()
            if not (
                isinstance(value, dict)
                and value
                and all(isinstance(v, torch.Tensor) for v in value.values())
            )
        }
    return {"checkpoint_object_type": type(obj).__name__}


def extract_state_dict(obj: Any) -> tuple[dict[str, torch.Tensor] | None, str]:
    if isinstance(obj, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
            "net",
            "network",
        ):
            value = obj.get(key)
            if isinstance(value, dict) and value and all(
                isinstance(v, torch.Tensor) for v in value.values()
            ):
                return value, key
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj, "<checkpoint_is_state_dict>"
    if hasattr(obj, "state_dict"):
        value = obj.state_dict()
        if isinstance(value, dict):
            return value, "<pickled_model.state_dict>"
    return None, ""


def common_prefix_families(keys: list[str]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    params: dict[str, int] = {}
    for key in keys:
        parts = key.split(".")
        prefix = parts[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return [
        {
            "top_level_prefix": prefix,
            "tensor_count": counts[prefix],
        }
        for prefix in sorted(counts)
    ]


def likely_module_families(keys: list[str]) -> dict[str, list[str]]:
    lowered = {key: key.lower() for key in keys}
    result: dict[str, list[str]] = {}
    for family, tokens in MODULE_FAMILIES.items():
        matches = sorted(
            {
                key.split(".")[0]
                for key, lower in lowered.items()
                if any(token in lower for token in tokens)
            }
        )
        result[family] = matches
    return result


def iter_python_files(repo: Path) -> list[Path]:
    roots = [
        repo / "scripts",
        repo / "src",
        repo / "models",
        repo,
    ]
    found: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root == repo:
            candidates = list(root.glob("*.py"))
        else:
            candidates = list(root.rglob("*.py"))
        for path in candidates:
            if any(part in {".venv", "__pycache__", ".git"} for part in path.parts):
                continue
            try:
                if path.stat().st_size > 2_000_000:
                    continue
            except OSError:
                continue
            found.add(path.resolve())
    return sorted(found)


def ast_inventory(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text, filename=str(path))

    classes: list[dict[str, Any]] = []
    functions: list[str] = []
    assignments: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                try:
                    bases.append(ast.unparse(base))
                except Exception:
                    bases.append(type(base).__name__)
            methods = [
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            self_assignments: list[str] = []
            forward_returns: list[str] = []
            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for subnode in ast.walk(child):
                    if isinstance(subnode, (ast.Assign, ast.AnnAssign)):
                        targets = (
                            subnode.targets
                            if isinstance(subnode, ast.Assign)
                            else [subnode.target]
                        )
                        for target in targets:
                            if (
                                isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"
                            ):
                                self_assignments.append(target.attr)
                    if child.name == "forward" and isinstance(subnode, ast.Return):
                        try:
                            forward_returns.append(ast.unparse(subnode.value))
                        except Exception:
                            forward_returns.append(type(subnode.value).__name__)
            classes.append(
                {
                    "name": node.name,
                    "bases": bases,
                    "methods": sorted(set(methods)),
                    "self_assignments": sorted(set(self_assignments)),
                    "forward_returns": forward_returns,
                }
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                assignments.append(ast.unparse(node))
            except Exception:
                pass

    lowered = text.lower()
    token_hits = [token for token in SEARCH_TOKENS if token in lowered]
    nn_module_classes = [
        item["name"]
        for item in classes
        if any(
            "module" in base.lower()
            for base in item["bases"]
        )
    ]

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "token_hits": token_hits,
        "classes": classes,
        "nn_module_classes": nn_module_classes,
        "top_level_functions": functions,
        "top_level_assignment_preview": assignments[:20],
    }


def candidate_score(record: dict[str, Any], state_prefixes: list[str]) -> int:
    score = 0
    hits = set(record["token_hits"])
    score += 3 * len(hits)
    if record["nn_module_classes"]:
        score += 5
    assignments = set()
    for cls in record["classes"]:
        assignments.update(cls["self_assignments"])
        if "forward" in cls["methods"]:
            score += 3
    for prefix in state_prefixes:
        if prefix in assignments:
            score += 5
        if any(prefix.lower() in hit.lower() for hit in hits):
            score += 2
    path_lower = record["path"].lower()
    for token in ("a3", "v4", "source", "count", "rank"):
        if token in path_lower:
            score += 1
    return score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--a3-12-lock", type=Path, required=True)
    parser.add_argument("--a3-12-summary", type=Path, required=True)
    parser.add_argument("--operational-policy-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    required = [
        args.repo,
        args.checkpoint,
        args.a3_12_lock,
        args.a3_12_summary,
        args.operational_policy_lock,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("A3.13-P FAIL: missing required paths", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(
            f"A3.13-P FAIL: output exists: {args.output_dir}",
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True)

    a3_12_lock = load_json(args.a3_12_lock)
    a3_12_summary = load_json(args.a3_12_summary)
    operational_lock = load_json(args.operational_policy_lock)

    checkpoint_sha = sha256_file(args.checkpoint)
    failures: list[str] = []
    if a3_12_lock.get("checkpoint_sha256") != checkpoint_sha:
        failures.append("A3.12 checkpoint hash mismatch")
    if operational_lock.get("checkpoint_sha256") != checkpoint_sha:
        failures.append("operational checkpoint hash mismatch")
    if a3_12_summary.get("development_test_used_for_selection") is not False:
        failures.append("A3.12 reports development-test selection")

    if failures:
        payload = {"status": "FAIL", "failures": failures}
        (args.output_dir / "preflight_failure.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    state_dict, state_location = extract_state_dict(checkpoint)
    if state_dict is None:
        payload = {
            "status": "HOLD",
            "reason": "state_dict_not_found",
            "checkpoint_object_type": type(checkpoint).__name__,
        }
        (args.output_dir / "preflight_summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2))
        return 0

    state_rows: list[dict[str, Any]] = []
    total_parameters = 0
    for key, tensor in state_dict.items():
        count = int(tensor.numel())
        total_parameters += count
        state_rows.append(
            {
                "key": key,
                "shape": "x".join(str(x) for x in tensor.shape),
                "dtype": str(tensor.dtype),
                "parameter_count": count,
                "top_level_prefix": key.split(".")[0],
            }
        )
    write_csv(args.output_dir / "checkpoint_state_dict.csv", state_rows)

    state_keys = list(state_dict.keys())
    state_prefixes = sorted({key.split(".")[0] for key in state_keys})
    families = likely_module_families(state_keys)

    python_files = iter_python_files(args.repo)
    source_records: list[dict[str, Any]] = []
    parse_failures: list[dict[str, str]] = []

    for path in python_files:
        try:
            record = ast_inventory(path)
        except (SyntaxError, UnicodeDecodeError, OSError) as exc:
            parse_failures.append(
                {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        if not record["token_hits"] and not record["nn_module_classes"]:
            continue
        record["score"] = candidate_score(record, state_prefixes)
        source_records.append(record)

    source_records.sort(
        key=lambda item: (-int(item["score"]), item["path"])
    )

    candidate_rows: list[dict[str, Any]] = []
    for rank, record in enumerate(source_records[:50], start=1):
        class_names = [item["name"] for item in record["classes"]]
        assignments = sorted(
            {
                value
                for item in record["classes"]
                for value in item["self_assignments"]
            }
        )
        candidate_rows.append(
            {
                "rank": rank,
                "score": record["score"],
                "path": record["path"],
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
                "token_hits": ",".join(record["token_hits"]),
                "class_names": ",".join(class_names),
                "nn_module_classes": ",".join(record["nn_module_classes"]),
                "self_assignments": ",".join(assignments),
            }
        )
    write_csv(args.output_dir / "model_source_candidates.csv", candidate_rows)
    write_csv(args.output_dir / "python_parse_failures.csv", parse_failures)

    likely_candidate = source_records[0] if source_records else None
    likely_classes = []
    likely_assignments: set[str] = set()
    if likely_candidate:
        likely_classes = likely_candidate["nn_module_classes"]
        for item in likely_candidate["classes"]:
            likely_assignments.update(item["self_assignments"])

    required_families = (
        "temporal",
        "local_projection",
        "graph",
        "node_head",
    )
    family_presence = {
        family: bool(families.get(family))
        for family in required_families
    }

    source_matches_prefix = False
    if likely_candidate:
        source_matches_prefix = any(
            prefix in likely_assignments
            for prefix in state_prefixes
        )

    ready = bool(
        likely_candidate
        and likely_classes
        and source_matches_prefix
        and all(family_presence.values())
    )

    summary = {
        "status": "PASS" if ready else "HOLD",
        "designation": "V4-A3.13-P Frozen Local-vs-Graph Attribution Preflight",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_object_type": type(checkpoint).__name__,
        "checkpoint_top_level_inventory": top_level_inventory(checkpoint),
        "state_dict_location": state_location,
        "state_dict_tensor_count": len(state_dict),
        "state_dict_total_parameter_count": total_parameters,
        "state_dict_prefix_inventory": common_prefix_families(state_keys),
        "likely_module_families": families,
        "required_family_presence": family_presence,
        "python_file_count_scanned": len(python_files),
        "candidate_source_count": len(source_records),
        "likely_model_source": (
            {
                "path": likely_candidate["path"],
                "sha256": likely_candidate["sha256"],
                "score": likely_candidate["score"],
                "nn_module_classes": likely_candidate["nn_module_classes"],
                "classes": likely_candidate["classes"],
            }
            if likely_candidate
            else None
        ),
        "source_matches_checkpoint_prefix": source_matches_prefix,
        "exact_instrumentation_ready": ready,
        "interpretation": (
            "Exact frozen-model instrumentation may proceed."
            if ready
            else (
                "Do not implement branch attribution yet. Resolve the exact "
                "model class/module names from the reported checkpoint and "
                "source inventories."
            )
        ),
        "development_test_accessed": False,
        "training_performed": False,
        "policy_selection_performed": False,
        "provenance": {
            "a3_12_lock_sha256": sha256_file(args.a3_12_lock),
            "a3_12_summary_sha256": sha256_file(args.a3_12_summary),
            "operational_policy_lock_sha256": sha256_file(
                args.operational_policy_lock
            ),
        },
    }

    summary_path = args.output_dir / "preflight_summary.json"
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lock = {
        "status": (
            "A3_13_PREFLIGHT_READY"
            if ready
            else "A3_13_PREFLIGHT_HOLD"
        ),
        "checkpoint_sha256": checkpoint_sha,
        "preflight_summary_sha256": sha256_file(summary_path),
        "development_test_accessed": False,
        "training_performed": False,
        "policy_selection_performed": False,
    }
    (args.output_dir / "A3_13_PREFLIGHT_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    marker_name = (
        "V4_A3_13_PREFLIGHT_READY"
        if ready
        else "V4_A3_13_PREFLIGHT_HOLD"
    )
    (args.output_dir / marker_name).write_text(
        marker_name + "\n",
        encoding="utf-8",
    )

    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    print(marker_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
