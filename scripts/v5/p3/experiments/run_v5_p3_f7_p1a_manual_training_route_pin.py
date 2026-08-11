from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "V5_P3_F7_P1A_MANUAL_TRAINING_ROUTE_PIN"
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "TRAIN-DERIVED"

EXPECTED_MODEL_CLASS = "V6P0Dynamic70GraphConvCount4"
EXPECTED_TRAIN_ITEMS = 110855
EXPECTED_VALIDATION_ITEMS = 13863
EXPECTED_PARAMETER_COUNT = 60553

TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".json",
    ".txt",
    ".log",
    ".md",
    ".yaml",
    ".yml",
}

ENTRYPOINT_NAME_TERMS = (
    "a4",
    "train",
    "training",
    "diagnostic",
    "tranche",
    "task_d",
    "dynamic70",
    "seed107",
)

REJECT_PATH_TERMS = (
    "f7_p0",
    "f7_p1",
    "f7_p1a",
    "preflight",
    "feature_study",
    "integrated_gradient",
    "permutation",
    "result_review",
    "metric_adapter",
    "export",
    "audit",
    "recovery_package",
)

EVIDENCE_KEYS = (
    "model_class",
    "dataloader",
    "backward",
    "optimizer_step",
    "zero_grad",
    "checkpoint_save",
    "epoch_loop",
    "train_split",
    "validation_split",
)

PYTHON_COMMAND_RE = re.compile(
    r"""(?:^|\s)(?:python|python3|[A-Za-z0-9_./-]+/python)\s+
        (?P<script>(?:"[^"]+\.py"|'[^']+\.py'|[^\s;&|]+\.py))
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

PYTHON_PATH_RE = re.compile(
    r"""(?P<script>(?:/|\.{0,2}/)?[A-Za-z0-9_./-]+\.py)"""
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    parser.add_argument(
        "--trainer-entrypoint",
        default="",
        help=(
            "Optional explicit current-repository Python entrypoint. "
            "The route is still rejected unless transitive evidence proves "
            "the complete training loop."
        ),
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def path_is_rejected(path: str | Path) -> tuple[bool, list[str]]:
    token = normalize(str(path))
    hits = [
        term
        for term in REJECT_PATH_TERMS
        if normalize(term) in token
    ]
    return bool(hits), hits


def resolve_path_reference(
    raw: str,
    *,
    repo: Path,
    source_file: Path,
) -> list[Path]:
    cleaned = raw.strip().strip("'\"`),:;[]{}")
    candidate = Path(cleaned).expanduser()

    attempts = []
    if candidate.is_absolute():
        attempts.append(candidate)
    else:
        attempts.extend(
            [
                source_file.parent / candidate,
                repo / candidate,
                repo / "scripts" / candidate,
                repo / "scripts/v5/p3" / candidate,
                repo / "scripts/v5/p3/experiments" / candidate,
            ]
        )

    output = []
    seen = set()
    for attempt in attempts:
        try:
            resolved = attempt.resolve()
        except Exception:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file() and resolved.suffix.lower() == ".py":
            output.append(resolved)
    return output


class SourceEvidenceVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: list[tuple[str, int]] = []
        self.import_from: list[tuple[str | None, int, list[str]]] = []
        self.functions: list[str] = []
        self.classes: list[str] = []
        self.evidence = {key: False for key in EVIDENCE_KEYS}
        self.counts = {key: 0 for key in EVIDENCE_KEYS}
        self.string_literals: list[str] = []

    def set_evidence(self, key: str) -> None:
        self.evidence[key] = True
        self.counts[key] += 1

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            self.imports.append((alias.name, 0))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        self.import_from.append(
            (
                node.module,
                int(node.level or 0),
                [alias.name for alias in node.names],
            )
        )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.functions.append(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.functions.append(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.classes.append(node.name)
        if node.name == EXPECTED_MODEL_CLASS:
            self.set_evidence("model_class")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id == EXPECTED_MODEL_CLASS:
            self.set_evidence("model_class")
        if node.id == "DataLoader":
            self.set_evidence("dataloader")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if node.attr == EXPECTED_MODEL_CLASS:
            self.set_evidence("model_class")
        if node.attr == "DataLoader":
            self.set_evidence("dataloader")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        function = node.func
        if isinstance(function, ast.Attribute):
            if function.attr == "backward":
                self.set_evidence("backward")
            elif function.attr == "step":
                self.set_evidence("optimizer_step")
            elif function.attr == "zero_grad":
                self.set_evidence("zero_grad")
            elif function.attr in ("save", "save_checkpoint"):
                self.set_evidence("checkpoint_save")
        elif isinstance(function, ast.Name):
            if function.id in ("save_checkpoint", "checkpoint_save"):
                self.set_evidence("checkpoint_save")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        target = ast.unparse(node.target).lower()
        iterator = ast.unparse(node.iter).lower()
        if "epoch" in target or "epoch" in iterator:
            self.set_evidence("epoch_loop")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> Any:
        condition = ast.unparse(node.test).lower()
        if "epoch" in condition:
            self.set_evidence("epoch_loop")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, str):
            value = node.value
            token = normalize(value)
            if len(value) <= 1000:
                self.string_literals.append(value)
            if EXPECTED_MODEL_CLASS in value:
                self.set_evidence("model_class")
            if token in {"train", "training"} or "split_train" in token:
                self.set_evidence("train_split")
            if token in {
                "validation",
                "valid",
                "val",
                "dev",
            } or "split_validation" in token:
                self.set_evidence("validation_split")
        self.generic_visit(node)


def module_name_for_path(repo: Path, path: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(repo.resolve())
    except ValueError:
        return None

    if relative.suffix != ".py":
        return None

    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def build_module_map(repo: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    exact = {}
    suffix_map: dict[str, list[Path]] = defaultdict(list)

    for path in repo.rglob("*.py"):
        if not path.is_file():
            continue
        module = module_name_for_path(repo, path)
        if not module:
            continue
        exact[module] = path.resolve()
        components = module.split(".")
        for start in range(len(components)):
            suffix_map[".".join(components[start:])].append(path.resolve())

    return exact, suffix_map


def resolve_imports(
    *,
    repo: Path,
    path: Path,
    visitor: SourceEvidenceVisitor,
    module_map: dict[str, Path],
    suffix_map: dict[str, list[Path]],
) -> list[Path]:
    resolved = set()
    current_module = module_name_for_path(repo, path)
    current_parts = current_module.split(".") if current_module else []

    def add_module(module: str) -> None:
        if module in module_map:
            resolved.add(module_map[module])
            return
        matches = suffix_map.get(module, [])
        if len(matches) == 1:
            resolved.add(matches[0])

    for module, _ in visitor.imports:
        add_module(module)

    for module, level, names in visitor.import_from:
        prefix_parts = list(current_parts[:-1])
        if level > 0:
            keep = max(0, len(prefix_parts) - (level - 1))
            prefix_parts = prefix_parts[:keep]

        module_parts = module.split(".") if module else []
        base = ".".join(prefix_parts + module_parts)
        if base:
            add_module(base)

        for name in names:
            candidate = ".".join(
                [part for part in (base, name) if part]
            )
            add_module(candidate)

    return sorted(resolved)


def inspect_current_source(
    *,
    repo: Path,
    path: Path,
    module_map: dict[str, Path],
    suffix_map: dict[str, list[Path]],
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
        syntax_ok = True
        syntax_error = None
    except SyntaxError as exc:
        tree = None
        syntax_ok = False
        syntax_error = repr(exc)

    visitor = SourceEvidenceVisitor()
    if tree is not None:
        visitor.visit(tree)

    imports = (
        resolve_imports(
            repo=repo,
            path=path,
            visitor=visitor,
            module_map=module_map,
            suffix_map=suffix_map,
        )
        if tree is not None
        else []
    )
    rejected, rejection_terms = path_is_rejected(path)

    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "syntax_ok": syntax_ok,
        "syntax_error": syntax_error,
        "rejected": rejected,
        "rejection_terms": rejection_terms,
        "module": module_name_for_path(repo, path),
        "imports": [str(candidate) for candidate in imports],
        "functions": sorted(set(visitor.functions)),
        "classes": sorted(set(visitor.classes)),
        "direct_evidence": visitor.evidence,
        "direct_evidence_counts": visitor.counts,
        "string_literals": visitor.string_literals[:200],
        "size_bytes": int(path.stat().st_size),
    }


def collect_current_sources(repo: Path) -> dict[str, dict[str, Any]]:
    module_map, suffix_map = build_module_map(repo)
    output = {}

    roots = [
        repo / "scripts",
        repo / "src",
    ]
    seen = set()

    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                output[str(resolved)] = inspect_current_source(
                    repo=repo,
                    path=resolved,
                    module_map=module_map,
                    suffix_map=suffix_map,
                )
            except Exception as exc:
                output[str(resolved)] = {
                    "path": str(resolved),
                    "inspection_error": repr(exc),
                    "syntax_ok": False,
                    "rejected": False,
                    "imports": [],
                    "direct_evidence": {
                        key: False for key in EVIDENCE_KEYS
                    },
                    "direct_evidence_counts": {
                        key: 0 for key in EVIDENCE_KEYS
                    },
                    "sha256": sha256_file(resolved),
                }

    return output


def mine_text_evidence(repo: Path) -> dict[str, Any]:
    roots = [
        repo / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107",
        repo / "reports/v5",
        repo / "scripts/v5/p3",
    ]
    evidence_files = []
    entrypoint_references: dict[str, set[str]] = defaultdict(set)

    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if (
                not path.is_file()
                or path.suffix.lower() not in TEXT_SUFFIXES
                or path.stat().st_size > 64 * 1024 * 1024
            ):
                continue
            resolved_file = path.resolve()
            if resolved_file in seen:
                continue
            seen.add(resolved_file)

            text = path.read_text(encoding="utf-8", errors="replace")
            command_hits = []
            path_hits = []

            for match in PYTHON_COMMAND_RE.finditer(text):
                raw = match.group("script")
                resolved_paths = resolve_path_reference(
                    raw,
                    repo=repo,
                    source_file=path,
                )
                command_hits.append({
                    "raw": raw,
                    "resolved": [str(candidate) for candidate in resolved_paths],
                })
                for candidate in resolved_paths:
                    entrypoint_references[str(candidate)].add(
                        str(resolved_file)
                    )

            for match in PYTHON_PATH_RE.finditer(text):
                raw = match.group("script")
                resolved_paths = resolve_path_reference(
                    raw,
                    repo=repo,
                    source_file=path,
                )
                path_hits.append({
                    "raw": raw,
                    "resolved": [str(candidate) for candidate in resolved_paths],
                })
                for candidate in resolved_paths:
                    entrypoint_references[str(candidate)].add(
                        str(resolved_file)
                    )

            relevant_lines = [
                {
                    "line": index,
                    "text": line[:2000],
                }
                for index, line in enumerate(text.splitlines(), start=1)
                if any(
                    term in line.lower()
                    for term in (
                        "python ",
                        "python3 ",
                        "trainer",
                        "training",
                        "seed 107",
                        "seed=107",
                        "epoch",
                        "checkpoint",
                        EXPECTED_MODEL_CLASS.lower(),
                    )
                )
            ][:500]

            if command_hits or path_hits or relevant_lines:
                evidence_files.append({
                    "path": str(resolved_file),
                    "sha256": sha256_file(path),
                    "command_hits": command_hits,
                    "path_hits": path_hits,
                    "relevant_lines": relevant_lines,
                })

    return {
        "evidence_file_count": len(evidence_files),
        "evidence_files": evidence_files,
        "entrypoint_reference_counts": {
            path: len(files)
            for path, files in entrypoint_references.items()
        },
        "entrypoint_referenced_by": {
            path: sorted(files)
            for path, files in entrypoint_references.items()
        },
    }


def transitive_route(
    entrypoint: str,
    sources: dict[str, dict[str, Any]],
    *,
    max_depth: int = 12,
) -> dict[str, Any]:
    queue = deque([(entrypoint, 0)])
    visited = []
    seen = set()

    aggregate_evidence = {key: False for key in EVIDENCE_KEYS}
    evidence_sources: dict[str, list[str]] = defaultdict(list)

    while queue:
        path, depth = queue.popleft()
        if path in seen or depth > max_depth or path not in sources:
            continue
        seen.add(path)
        visited.append(path)

        row = sources[path]
        for key, value in row["direct_evidence"].items():
            if value:
                aggregate_evidence[key] = True
                evidence_sources[key].append(path)

        for imported in row.get("imports", []):
            queue.append((imported, depth + 1))

    source_hashes = sorted(
        (path, sources[path]["sha256"])
        for path in visited
    )
    fingerprint = sha256_bytes(
        json.dumps(source_hashes, separators=(",", ":")).encode("utf-8")
    )

    return {
        "entrypoint": entrypoint,
        "entrypoint_sha256": sources[entrypoint]["sha256"],
        "source_count": len(visited),
        "sources": visited,
        "source_hashes": [
            {"path": path, "sha256": sha}
            for path, sha in source_hashes
        ],
        "aggregate_evidence": aggregate_evidence,
        "evidence_sources": {
            key: sorted(set(paths))
            for key, paths in evidence_sources.items()
        },
        "all_required_evidence": all(aggregate_evidence.values()),
        "route_fingerprint_sha256": fingerprint,
    }


def candidate_entrypoints(
    *,
    repo: Path,
    sources: dict[str, dict[str, Any]],
    text_evidence: dict[str, Any],
    explicit: str,
) -> list[dict[str, Any]]:
    reference_counts = text_evidence["entrypoint_reference_counts"]
    explicit_path = None
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if not explicit_path.is_absolute():
            explicit_path = repo / explicit_path
        explicit_path = explicit_path.resolve()
        require(
            explicit_path.is_file() and explicit_path.suffix == ".py",
            f"explicit trainer entrypoint missing: {explicit_path}",
        )

    rows = []
    for path, source in sources.items():
        path_obj = Path(path)
        name_token = normalize(path_obj.name)
        referenced = int(reference_counts.get(path, 0))
        entrypoint_name_hit = any(
            normalize(term) in name_token
            for term in ENTRYPOINT_NAME_TERMS
        )
        direct_main = "main" in source.get("functions", [])
        explicit_match = explicit_path is not None and path_obj == explicit_path

        if not (
            explicit_match
            or referenced > 0
            or entrypoint_name_hit
            or direct_main
        ):
            continue

        rejected, rejection_terms = path_is_rejected(path)
        route = transitive_route(path, sources)

        score = 0
        score += 200 if explicit_match else 0
        score += 50 * referenced
        score += 25 if entrypoint_name_hit else 0
        score += 10 if direct_main else 0
        score += 20 * sum(route["aggregate_evidence"].values())
        score += 100 if route["all_required_evidence"] else 0
        score -= 1000 if rejected else 0

        rows.append({
            **route,
            "explicit_match": explicit_match,
            "reference_count": referenced,
            "referenced_by": text_evidence[
                "entrypoint_referenced_by"
            ].get(path, []),
            "entrypoint_name_hit": entrypoint_name_hit,
            "direct_main_function": direct_main,
            "rejected": rejected,
            "rejection_terms": rejection_terms,
            "score": score,
        })

    rows.sort(key=lambda row: (-row["score"], row["entrypoint"]))
    return rows


def git_historical_inventory(repo: Path) -> dict[str, Any]:
    if not (repo / ".git").exists():
        return {
            "git_repository": False,
            "records": [],
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
                "--format=COMMIT:%H",
                "--name-only",
                "--",
                "*.py",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception as exc:
        return {
            "git_repository": True,
            "records": [],
            "error": repr(exc),
        }

    records = []
    commit = None
    seen = set()

    for line in process.stdout.splitlines():
        if line.startswith("COMMIT:"):
            commit = line.split(":", 1)[1].strip()
            continue
        path = line.strip()
        if not path or commit is None or not path.endswith(".py"):
            continue

        token = normalize(path)
        if not any(normalize(term) in token for term in ENTRYPOINT_NAME_TERMS):
            continue
        if any(normalize(term) in token for term in REJECT_PATH_TERMS):
            continue

        key = (commit, path)
        if key in seen:
            continue
        seen.add(key)

        try:
            show = subprocess.run(
                ["git", "-C", str(repo), "show", f"{commit}:{path}"],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except Exception:
            continue

        payload = show.stdout
        text = payload.decode("utf-8", errors="replace")
        evidence = {
            "model_class": EXPECTED_MODEL_CLASS in text,
            "dataloader": "DataLoader" in text,
            "backward": ".backward(" in text,
            "optimizer_step": ".step(" in text,
            "zero_grad": ".zero_grad(" in text,
            "checkpoint_save": (
                "torch.save(" in text or "save_checkpoint(" in text
            ),
            "epoch_loop": (
                "for epoch" in text or "while epoch" in text
            ),
            "train_split": (
                '"train"' in text or "'train'" in text
            ),
            "validation_split": any(
                literal in text
                for literal in (
                    '"validation"',
                    "'validation'",
                    '"val"',
                    "'val'",
                )
            ),
        }
        records.append({
            "commit": commit,
            "path": path,
            "blob_sha256": sha256_bytes(payload),
            "evidence": evidence,
            "all_required_evidence_in_single_blob": all(evidence.values()),
            "current_path_exists": (repo / path).is_file(),
        })

        if len(records) >= 500:
            break

    return {
        "git_repository": True,
        "record_count": len(records),
        "records": records,
        "error": None,
    }


def choose_route(
    candidates: list[dict[str, Any]],
    explicit: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    eligible = [
        row
        for row in candidates
        if row["all_required_evidence"]
        and not row["rejected"]
    ]

    by_fingerprint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_fingerprint[row["route_fingerprint_sha256"]].append(row)

    groups = []
    for fingerprint, rows in by_fingerprint.items():
        rows.sort(
            key=lambda row: (
                -row["explicit_match"],
                -row["reference_count"],
                -row["score"],
                row["entrypoint"],
            )
        )
        groups.append({
            "fingerprint": fingerprint,
            "canonical": rows[0],
            "equivalent_entrypoints": rows,
            "maximum_reference_count": max(
                row["reference_count"] for row in rows
            ),
            "maximum_score": max(row["score"] for row in rows),
            "contains_explicit_entrypoint": any(
                row["explicit_match"] for row in rows
            ),
        })

    groups.sort(
        key=lambda group: (
            -group["contains_explicit_entrypoint"],
            -group["maximum_reference_count"],
            -group["maximum_score"],
            group["canonical"]["entrypoint"],
        )
    )

    decision = {
        "eligible_entrypoint_count": len(eligible),
        "unique_route_fingerprint_count": len(groups),
        "route_groups": groups,
        "selection_rule": (
            "An explicit entrypoint is accepted only if transitive repository "
            "imports prove all frozen training-loop evidence. Otherwise choose "
            "one uniquely referenced complete route fingerprint."
        ),
    }

    if explicit:
        explicit_groups = [
            group
            for group in groups
            if group["contains_explicit_entrypoint"]
        ]
        if len(explicit_groups) == 1:
            decision["classification"] = (
                "explicit_entrypoint_transitively_verified"
            )
            return explicit_groups[0]["canonical"], decision
        decision["classification"] = (
            "explicit_entrypoint_failed_transitive_verification"
        )
        return None, decision

    referenced = [
        group
        for group in groups
        if group["maximum_reference_count"] > 0
    ]
    if len(referenced) == 1:
        decision["classification"] = (
            "unique_referenced_transitive_training_route"
        )
        return referenced[0]["canonical"], decision

    if len(referenced) > 1:
        top = referenced[0]
        second = referenced[1]
        if (
            top["maximum_reference_count"]
            > second["maximum_reference_count"]
            and top["maximum_score"] - second["maximum_score"] >= 20
        ):
            decision["classification"] = (
                "high_margin_referenced_transitive_training_route"
            )
            return top["canonical"], decision
        decision["classification"] = (
            "multiple_referenced_transitive_training_routes"
        )
        return None, decision

    if len(groups) == 1:
        decision["classification"] = (
            "unique_repository_transitive_training_route"
        )
        return groups[0]["canonical"], decision

    if len(groups) > 1:
        top = groups[0]
        second = groups[1]
        if top["maximum_score"] - second["maximum_score"] >= 40:
            decision["classification"] = (
                "high_margin_repository_transitive_training_route"
            )
            return top["canonical"], decision

    decision["classification"] = "training_route_still_unresolved"
    return None, decision


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

    p1_report_path = f7_root / (
        "V5_P3_F7_P1_OFFICIAL_TRAINING_ROUTE_RECOVERY_REPORT.json"
    )
    p1_lock_path = f7_root / (
        "V5_P3_F7_P1_OFFICIAL_TRAINING_ROUTE_RECOVERY_LOCK.json"
    )
    p0_protocol_path = f7_root / (
        "F7_P0_FROZEN_SAME_WIDTH_ABLATION_PROTOCOL.json"
    )
    split_cert_path = f7_root / (
        "F7_P1_GUARDED_SPLIT_LENGTH_CERTIFICATION.json"
    )
    f6r_report_path = f6r_root / (
        "V5_P3_F6R_TASK_SPECIFIC_INTEGRATED_GRADIENTS_RESULT_REVIEW_REPORT.json"
    )
    f6r_lock_path = f6r_root / (
        "V5_P3_F6R_TASK_SPECIFIC_INTEGRATED_GRADIENTS_RESULT_REVIEW_LOCK.json"
    )

    p1_report, p1_lock = verify_report_lock(
        p1_report_path,
        p1_lock_path,
    )
    f6r_report, f6r_lock = verify_report_lock(
        f6r_report_path,
        f6r_lock_path,
    )
    protocol = load_json(p0_protocol_path)
    split_cert = load_json(split_cert_path)

    require(
        p1_lock.get("official_trainer_route_resolved") is False,
        "P1 already resolved the trainer",
    )
    require(
        p1_lock.get("F7_adapter_preflight_authorized") is False,
        "P1 unexpectedly authorized P2",
    )
    require(
        p1_lock.get("actual_F7_retraining_authorized") is False,
        "actual F7 unexpectedly authorized",
    )
    require(
        p1_lock.get("feature_removal_authorized") is False,
        "feature removal unexpectedly authorized",
    )
    require(f6r_lock.get("F6R_complete") is True, "F6R incomplete")
    require(
        split_cert["train_count_certified"] is True
        and split_cert["validation_count_certified"] is True,
        "P1 split-length certification did not pass",
    )
    require(
        int(split_cert["train_length"]) == EXPECTED_TRAIN_ITEMS,
        "train item count changed",
    )
    require(
        int(split_cert["validation_length"]) == EXPECTED_VALIDATION_ITEMS,
        "validation item count changed",
    )
    require(
        protocol["masking_protocol"]["model_parameter_count_preserved"]
        == EXPECTED_PARAMETER_COUNT,
        "F7 model parameter count changed",
    )

    current_sources = collect_current_sources(repo)
    text_evidence = mine_text_evidence(repo)
    candidates = candidate_entrypoints(
        repo=repo,
        sources=current_sources,
        text_evidence=text_evidence,
        explicit=args.trainer_entrypoint,
    )
    selected_route, selection = choose_route(
        candidates,
        args.trainer_entrypoint,
    )
    git_inventory = git_historical_inventory(repo)

    route_resolved = selected_route is not None
    adapter_preflight_authorized = bool(
        route_resolved
        and split_cert["train_count_certified"]
        and split_cert["validation_count_certified"]
        and f6r_lock.get("F7_protocol_preflight_authorized") is True
    )

    next_stage = (
        "V5_P3_F7_P2_FROZEN_TRAINER_ADAPTER_AND_DRY_RUN_PREFLIGHT"
        if adapter_preflight_authorized
        else "V5_P3_F7_P1B_MANUAL_ENTRYPOINT_INPUT_REQUIRED"
    )

    source_inventory_path = output_dir / (
        "F7_P1A_CURRENT_SOURCE_AND_IMPORT_GRAPH_INVENTORY.json"
    )
    text_evidence_path = output_dir / (
        "F7_P1A_A4_AND_REPORT_ENTRYPOINT_REFERENCE_EVIDENCE.json"
    )
    candidate_path = output_dir / (
        "F7_P1A_TRANSITIVE_TRAINING_ROUTE_CANDIDATES.json"
    )
    git_path = output_dir / (
        "F7_P1A_HISTORICAL_TRAINER_BLOB_INVENTORY.json"
    )
    frozen_route_path = output_dir / (
        "F7_P1A_FROZEN_MANUALLY_VERIFIED_TRAINING_ROUTE.json"
    )
    decision_path = output_dir / (
        "F7_P1A_TRAINING_ROUTE_PIN_DECISION.json"
    )

    atomic_json(
        source_inventory_path,
        {
            "source_count": len(current_sources),
            "sources": list(current_sources.values()),
        },
    )
    atomic_json(text_evidence_path, text_evidence)
    atomic_json(
        candidate_path,
        {
            "explicit_entrypoint_argument": (
                str(Path(args.trainer_entrypoint).expanduser())
                if args.trainer_entrypoint
                else None
            ),
            "candidate_count": len(candidates),
            "selection": selection,
            "selected_route": selected_route,
            "candidates": candidates[:200],
        },
    )
    atomic_json(git_path, git_inventory)

    frozen_route = {
        "status": "FROZEN" if route_resolved else "UNRESOLVED",
        "selection_classification": selection["classification"],
        "selected_route": selected_route,
        "train_items": EXPECTED_TRAIN_ITEMS,
        "validation_items": EXPECTED_VALIDATION_ITEMS,
        "model_class": EXPECTED_MODEL_CLASS,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "F7_protocol": str(p0_protocol_path),
        "F7_protocol_sha256": sha256_file(p0_protocol_path),
        "allowed_next_change": (
            "P2 may add only the frozen post-loader single-group zeroing "
            "adapter and must first prove a no-mask dry run."
            if route_resolved
            else None
        ),
    }
    atomic_json(frozen_route_path, frozen_route)

    decision = {
        "F7_P1_complete": True,
        "split_counts_certified": True,
        "manual_entrypoint_argument_supplied": bool(
            args.trainer_entrypoint
        ),
        "current_source_count": len(current_sources),
        "candidate_entrypoint_count": len(candidates),
        "eligible_complete_route_count": selection[
            "eligible_entrypoint_count"
        ],
        "unique_route_fingerprint_count": selection[
            "unique_route_fingerprint_count"
        ],
        "selection_classification": selection["classification"],
        "official_trainer_route_resolved": route_resolved,
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
            "Recover the A4 trainer when no monolithic strict script exists "
            "by building a repository-local Python import graph, aggregating "
            "training-loop evidence transitively across entrypoint and helper "
            "modules, mining A4/report command references, inventorying "
            "historical Git blobs, optionally verifying an explicitly supplied "
            "entrypoint, freezing one unique complete route fingerprint, and "
            "authorizing only the P2 trainer-adapter dry-run preflight."
        ),
        "finding": {
            "P1_split_counts_certified": True,
            "current_source_count": len(current_sources),
            "text_evidence_file_count": text_evidence[
                "evidence_file_count"
            ],
            "candidate_entrypoint_count": len(candidates),
            "selection": selection,
            "selected_route": selected_route,
            "historical_git_record_count": git_inventory.get(
                "record_count",
                0,
            ),
        },
        "decision": decision,
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "dataset_object_constructed": False,
            "dataset_getitem_called": False,
            "training_feature_tensors_loaded": False,
            "validation_feature_tensors_loaded": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "source_code_scanned": True,
            "git_history_scanned": True,
            "scientific_protocol_changed": False,
            "actual_F7_retraining_authorized": False,
        },
        "artifacts": {
            "current_source_inventory": str(source_inventory_path),
            "entrypoint_reference_evidence": str(text_evidence_path),
            "transitive_route_candidates": str(candidate_path),
            "historical_git_inventory": str(git_path),
            "frozen_training_route": str(frozen_route_path),
            "decision": str(decision_path),
        },
        "provenance": {
            "F7_P1_report_sha256": sha256_file(p1_report_path),
            "F7_P1_lock_sha256": sha256_file(p1_lock_path),
            "F6R_report_sha256": sha256_file(f6r_report_path),
            "F6R_lock_sha256": sha256_file(f6r_lock_path),
            "F7_protocol_sha256": sha256_file(p0_protocol_path),
            "split_certification_sha256": sha256_file(split_cert_path),
            "installed_script_sha256": sha256_file(installed_script),
            "source_inventory_sha256": sha256_file(source_inventory_path),
            "entrypoint_evidence_sha256": sha256_file(text_evidence_path),
            "candidate_review_sha256": sha256_file(candidate_path),
            "git_inventory_sha256": sha256_file(git_path),
            "frozen_route_sha256": sha256_file(frozen_route_path),
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
            "source_inventory_sha256": sha256_file(
                source_inventory_path
            ),
            "entrypoint_evidence_sha256": sha256_file(
                text_evidence_path
            ),
            "candidate_review_sha256": sha256_file(candidate_path),
            "git_inventory_sha256": sha256_file(git_path),
            "frozen_route_sha256": sha256_file(frozen_route_path),
            "decision_sha256": sha256_file(decision_path),
            "split_counts_certified": True,
            "official_trainer_route_resolved": route_resolved,
            "F7_adapter_preflight_authorized": (
                adapter_preflight_authorized
            ),
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
    print("P1_split_counts_certified=true")
    print(f"current_source_count={len(current_sources)}")
    print(
        "text_evidence_file_count="
        f"{text_evidence['evidence_file_count']}"
    )
    print(f"candidate_entrypoint_count={len(candidates)}")
    print(
        "eligible_complete_route_count="
        f"{selection['eligible_entrypoint_count']}"
    )
    print(
        "unique_route_fingerprint_count="
        f"{selection['unique_route_fingerprint_count']}"
    )
    print(
        "selection_classification="
        f"{selection['classification']}"
    )
    if selected_route is not None:
        print(f"selected_entrypoint={selected_route['entrypoint']}")
        print(
            "selected_entrypoint_sha256="
            f"{selected_route['entrypoint_sha256']}"
        )
        print(
            "selected_route_fingerprint_sha256="
            f"{selected_route['route_fingerprint_sha256']}"
        )
        print(
            "selected_route_source_count="
            f"{selected_route['source_count']}"
        )
        print(
            "selected_route_all_required_evidence="
            f"{selected_route['all_required_evidence']}"
        )
    else:
        print("selected_entrypoint=None")
        top = candidates[:10]
        for index, row in enumerate(top, start=1):
            print(
                f"candidate_{index}="
                f"path={row['entrypoint']}:"
                f"score={row['score']}:"
                f"references={row['reference_count']}:"
                f"complete={str(row['all_required_evidence']).lower()}:"
                f"rejected={str(row['rejected']).lower()}:"
                f"evidence={row['aggregate_evidence']}"
            )
    print(
        "official_trainer_route_resolved="
        f"{str(route_resolved).lower()}"
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
    print("training_feature_tensors_loaded=false")
    print("validation_feature_tensors_loaded=false")
    print("sealed_test_tensors_loaded=false")
    print(f"next_stage={next_stage}")
    print(f"current_source_inventory={source_inventory_path}")
    print(f"entrypoint_reference_evidence={text_evidence_path}")
    print(f"transitive_route_candidates={candidate_path}")
    print(f"historical_git_inventory={git_path}")
    print(f"frozen_training_route={frozen_route_path}")
    print(f"decision={decision_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
