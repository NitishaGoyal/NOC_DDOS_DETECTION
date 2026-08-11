#!/usr/bin/env python3
"""
V4-A4a-1B Trainer Interface Preflight

Read-only inspection of the exact original A3 trainer before constructing the
A4a smoke-training adapter.

The script:
- verifies the exact trainer/source/checkpoint/A4a-1A provenance
- parses imports, classes, functions and argparse options
- imports the trainer without executing main()
- records runtime signatures
- identifies likely Dataset/DataLoader/train/eval/checkpoint helpers
- records checkpoint keys and saved configuration
- emits targeted source excerpts for training-pipeline symbols

It does not open x.npy, split arrays, validation predictions, test predictions
or any development-test artifact. It performs no training.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch


EXPECTED_TRAINER_SHA = (
    "e5f9a48e9f7ca873c351743df92e3bb0"
    "0d00a893333434c7fd0e04734e010525"
)
EXPECTED_MODEL_SOURCE_SHA = (
    "5ebcf80385b95f329faf935cad8039b64"
    "afd79f464903308eb07e674d78a1704"
)
EXPECTED_CHECKPOINT_SHA = (
    "f61c1add6c057f7f53dd34fb1f9f4e95"
    "b01cefd5053e0e42bc100c76dac7f923"
)
EXPECTED_SLOT_MODULE_SHA = (
    "97b56aabb3239dfc7543c241665cacf99"
    "8eb6b1a1f5f67f931cdeb70e75c111c"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def literal_or_text(node: ast.AST, source: str) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        segment = ast.get_source_segment(source, node)
        return segment if segment is not None else ast.dump(node)


def ast_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    arguments = []
    positional = [*node.args.posonlyargs, *node.args.args]
    default_offset = len(positional) - len(node.args.defaults)

    for index, argument in enumerate(positional):
        text = argument.arg
        if argument.annotation is not None:
            text += ":" + ast.unparse(argument.annotation)
        if index >= default_offset:
            text += "=" + ast.unparse(
                node.args.defaults[index - default_offset]
            )
        arguments.append(text)

    if node.args.vararg is not None:
        arguments.append("*" + node.args.vararg.arg)
    elif node.args.kwonlyargs:
        arguments.append("*")

    for argument, default in zip(
        node.args.kwonlyargs,
        node.args.kw_defaults,
    ):
        text = argument.arg
        if argument.annotation is not None:
            text += ":" + ast.unparse(argument.annotation)
        if default is not None:
            text += "=" + ast.unparse(default)
        arguments.append(text)

    if node.args.kwarg is not None:
        arguments.append("**" + node.args.kwarg.arg)

    result = f"{node.name}({', '.join(arguments)})"
    if node.returns is not None:
        result += " -> " + ast.unparse(node.returns)
    return result


def parse_argparse_options(
    tree: ast.AST,
    source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue

        flags = [
            literal_or_text(argument, source)
            for argument in node.args
        ]
        keywords = {
            keyword.arg or "**": literal_or_text(keyword.value, source)
            for keyword in node.keywords
        }
        rows.append(
            {
                "line": node.lineno,
                "flags": flags,
                "keywords": keywords,
            }
        )

    rows.sort(key=lambda row: int(row["line"]))
    return rows


def source_excerpt(
    lines: list[str],
    start: int,
    end: int,
) -> str:
    start = max(1, start)
    end = min(len(lines), end)
    return "\n".join(
        f"{line_number:5d}: {lines[line_number - 1]}"
        for line_number in range(start, end + 1)
    )


def checkpoint_inventory(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {"type": type(checkpoint).__name__}

    result: dict[str, Any] = {
        "top_level_keys": sorted(checkpoint.keys()),
    }

    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            result[key] = {
                "type": "Tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        elif isinstance(value, dict):
            child: dict[str, Any] = {
                "type": "dict",
                "keys": sorted(value.keys()),
            }
            if value and all(
                isinstance(item, torch.Tensor)
                for item in value.values()
            ):
                child["tensor_inventory"] = {
                    name: {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                    }
                    for name, tensor in value.items()
                }
            result[key] = child
        elif isinstance(value, (str, int, float, bool, type(None))):
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = {
                "type": type(value).__name__,
                "length": len(value),
                "preview": jsonable(list(value[:10])),
            }
        else:
            result[key] = {
                "type": type(value).__name__,
                "repr": repr(value)[:1000],
            }

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--slot-module", type=Path, required=True)
    parser.add_argument("--a4a-1a-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    required = [
        args.repo,
        args.trainer,
        args.model_source,
        args.checkpoint,
        args.slot_module,
        args.a4a_1a_dir / "V4_A4A_1A_SLOT_IMPLEMENTATION_PASS",
        args.a4a_1a_dir / "A4A_1A_LOCK.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("A4a-1B preflight FAIL: missing paths", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(
            f"A4a-1B preflight FAIL: output exists: {args.output_dir}",
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True)

    hashes = {
        "trainer_sha256": sha256_file(args.trainer),
        "model_source_sha256": sha256_file(args.model_source),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "slot_module_sha256": sha256_file(args.slot_module),
    }

    failures: list[str] = []
    expected = {
        "trainer_sha256": EXPECTED_TRAINER_SHA,
        "model_source_sha256": EXPECTED_MODEL_SOURCE_SHA,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "slot_module_sha256": EXPECTED_SLOT_MODULE_SHA,
    }
    for key, expected_value in expected.items():
        if hashes[key] != expected_value:
            failures.append(
                f"{key} mismatch: {hashes[key]} != {expected_value}"
            )

    a4a_1a_lock = load_json(args.a4a_1a_dir / "A4A_1A_LOCK.json")
    if a4a_1a_lock.get("status") != "A4A_1A_SLOT_IMPLEMENTATION_COMPLETE":
        failures.append("A4a-1A lock status is not complete")
    if a4a_1a_lock.get("development_test_accessed") is not False:
        failures.append("A4a-1A reports development-test access")

    if failures:
        payload = {"status": "FAIL", "failures": failures}
        (args.output_dir / "preflight_failure.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1

    source = args.trainer.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(args.trainer))

    imports: list[dict[str, Any]] = []
    functions_ast: list[dict[str, Any]] = []
    classes_ast: list[dict[str, Any]] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.append(
                {
                    "line": node.lineno,
                    "kind": "import",
                    "modules": [
                        {
                            "name": alias.name,
                            "asname": alias.asname,
                        }
                        for alias in node.names
                    ],
                }
            )
        elif isinstance(node, ast.ImportFrom):
            imports.append(
                {
                    "line": node.lineno,
                    "kind": "from",
                    "module": node.module,
                    "level": node.level,
                    "names": [
                        {
                            "name": alias.name,
                            "asname": alias.asname,
                        }
                        for alias in node.names
                    ],
                }
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions_ast.append(
                {
                    "name": node.name,
                    "signature": ast_signature(node),
                    "line_start": node.lineno,
                    "line_end": node.end_lineno,
                }
            )
        elif isinstance(node, ast.ClassDef):
            methods = []
            for child in node.body:
                if isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    methods.append(
                        {
                            "name": child.name,
                            "signature": ast_signature(child),
                            "line_start": child.lineno,
                            "line_end": child.end_lineno,
                        }
                    )
            classes_ast.append(
                {
                    "name": node.name,
                    "bases": [ast.unparse(base) for base in node.bases],
                    "line_start": node.lineno,
                    "line_end": node.end_lineno,
                    "methods": methods,
                }
            )

    argparse_options = parse_argparse_options(tree, source)

    for import_root in (
        args.repo,
        args.repo / "scripts",
        args.trainer.parent,
        args.model_source.parent,
        args.slot_module.parent,
    ):
        text = str(import_root.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)

    trainer_module = load_module(
        args.trainer,
        "resolved_original_v4_a3_trainer_for_a4a_1b_preflight",
    )

    runtime_symbols: list[dict[str, Any]] = []
    likely_symbols: list[dict[str, Any]] = []
    keywords = (
        "dataset",
        "data",
        "loader",
        "split",
        "train",
        "eval",
        "metric",
        "checkpoint",
        "mask",
        "adj",
        "graph",
        "early",
        "seed",
        "loss",
        "collate",
    )

    for name in sorted(vars(trainer_module)):
        if name.startswith("_"):
            continue
        value = getattr(trainer_module, name)
        if not (inspect.isclass(value) or inspect.isfunction(value)):
            continue
        if getattr(value, "__module__", None) != trainer_module.__name__:
            continue

        try:
            signature = str(inspect.signature(value))
        except Exception:
            signature = "<unavailable>"

        try:
            source_lines, start_line = inspect.getsourcelines(value)
            end_line = start_line + len(source_lines) - 1
        except Exception:
            start_line = None
            end_line = None

        row = {
            "name": name,
            "kind": "class" if inspect.isclass(value) else "function",
            "signature": signature,
            "line_start": start_line,
            "line_end": end_line,
        }
        runtime_symbols.append(row)

        lowered = name.lower()
        if any(keyword in lowered for keyword in keywords):
            likely_symbols.append(row)

    likely_class_methods: list[dict[str, Any]] = []
    for symbol in likely_symbols:
        if symbol["kind"] != "class":
            continue
        value = getattr(trainer_module, symbol["name"])
        for method_name, method in inspect.getmembers(
            value,
            predicate=inspect.isfunction,
        ):
            if method_name.startswith("_") and method_name != "__init__":
                continue
            try:
                signature = str(inspect.signature(method))
            except Exception:
                signature = "<unavailable>"
            try:
                source_lines, start_line = inspect.getsourcelines(method)
                end_line = start_line + len(source_lines) - 1
            except Exception:
                start_line = None
                end_line = None
            likely_class_methods.append(
                {
                    "class": symbol["name"],
                    "method": method_name,
                    "signature": signature,
                    "line_start": start_line,
                    "line_end": end_line,
                }
            )

    search_terms = [
        "split_id.npy",
        "run_index.npy",
        "x.npy",
        "y_graph.npy",
        "y_node.npy",
        "attacker_count.npy",
        "np.load",
        "mmap_mode",
        "Dataset",
        "DataLoader",
        "Subset",
        "WeightedRandomSampler",
        "A_hat",
        "physical_valid_port_mask",
        "edge_index",
        "optimizer",
        "Adam",
        "AdamW",
        "clip_grad",
        "early",
        "patience",
        "best_model",
        "torch.save",
        "load_state_dict",
        "train_loader",
        "val_loader",
        "test_loader",
        "test",
    ]

    term_hits: dict[str, list[int]] = {}
    excerpt_ranges: list[tuple[int, int]] = []

    for term in search_terms:
        hits = [
            line_number
            for line_number, line in enumerate(lines, start=1)
            if term in line
        ]
        term_hits[term] = hits
        for line_number in hits:
            excerpt_ranges.append(
                (
                    max(1, line_number - 4),
                    min(len(lines), line_number + 6),
                )
            )

    for symbol in likely_symbols:
        start = symbol.get("line_start")
        end = symbol.get("line_end")
        if start is None or end is None:
            continue
        if end - start <= 160:
            excerpt_ranges.append((int(start), int(end)))
        else:
            excerpt_ranges.append((int(start), int(start) + 80))
            excerpt_ranges.append((int(end) - 40, int(end)))

    excerpt_ranges.sort()
    merged_ranges: list[list[int]] = []
    for start, end in excerpt_ranges:
        if not merged_ranges or start > merged_ranges[-1][1] + 1:
            merged_ranges.append([start, end])
        else:
            merged_ranges[-1][1] = max(merged_ranges[-1][1], end)

    # Keep the report bounded while preserving the high-value sections.
    merged_ranges = merged_ranges[:40]
    excerpts = [
        {
            "line_start": start,
            "line_end": end,
            "text": source_excerpt(lines, start, end),
        }
        for start, end in merged_ranges
    ]

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    report = {
        "status": "PASS",
        "designation": "V4-A4a-1B Original Trainer Interface Preflight",
        "trainer_path": str(args.trainer),
        "hashes": hashes,
        "imports": imports,
        "functions_ast": functions_ast,
        "classes_ast": classes_ast,
        "argparse_options": argparse_options,
        "runtime_symbols": runtime_symbols,
        "likely_pipeline_symbols": likely_symbols,
        "likely_class_methods": likely_class_methods,
        "search_term_hits": term_hits,
        "source_excerpts": excerpts,
        "checkpoint_inventory": checkpoint_inventory(checkpoint),
        "dataset_accessed": False,
        "validation_accessed": False,
        "development_test_accessed": False,
        "training_performed": False,
        "ready_to_build_smoke_adapter": True,
    }

    report_path = args.output_dir / "trainer_interface_preflight.json"
    report_path.write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    text_report = args.output_dir / "trainer_interface_preflight.txt"
    with text_report.open("w", encoding="utf-8") as handle:
        handle.write("===== STATUS =====\n")
        handle.write("status: PASS\n")
        handle.write(f"trainer: {args.trainer}\n")
        for key, value in hashes.items():
            handle.write(f"{key}: {value}\n")

        handle.write("\n===== ARGPARSE OPTIONS =====\n")
        for row in argparse_options:
            handle.write(
                f"line={row['line']} flags={row['flags']} "
                f"keywords={row['keywords']}\n"
            )

        handle.write("\n===== LIKELY PIPELINE SYMBOLS =====\n")
        for row in likely_symbols:
            handle.write(
                f"{row['kind']} {row['name']}{row['signature']} "
                f"lines={row['line_start']}-{row['line_end']}\n"
            )

        handle.write("\n===== LIKELY CLASS METHODS =====\n")
        for row in likely_class_methods:
            handle.write(
                f"{row['class']}.{row['method']}{row['signature']} "
                f"lines={row['line_start']}-{row['line_end']}\n"
            )

        handle.write("\n===== SEARCH TERM HITS =====\n")
        for term, hits in term_hits.items():
            handle.write(f"{term}: {hits}\n")

        handle.write("\n===== CHECKPOINT INVENTORY =====\n")
        handle.write(
            json.dumps(
                jsonable(checkpoint_inventory(checkpoint)),
                indent=2,
                sort_keys=True,
            )
        )
        handle.write("\n")

        handle.write("\n===== TARGETED SOURCE EXCERPTS =====\n")
        for excerpt in excerpts:
            handle.write(
                f"\n--- lines {excerpt['line_start']}-"
                f"{excerpt['line_end']} ---\n"
            )
            handle.write(excerpt["text"])
            handle.write("\n")

    lock = {
        "status": "A4A_1B_TRAINER_INTERFACE_PREFLIGHT_COMPLETE",
        **hashes,
        "trainer_interface_preflight_sha256": sha256_file(report_path),
        "trainer_interface_text_sha256": sha256_file(text_report),
        "dataset_accessed": False,
        "validation_accessed": False,
        "development_test_accessed": False,
        "training_performed": False,
    }
    (args.output_dir / "A4A_1B_PREFLIGHT_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "V4_A4A_1B_TRAINER_INTERFACE_PREFLIGHT_PASS").write_text(
        "V4_A4A_1B_TRAINER_INTERFACE_PREFLIGHT_PASS\n",
        encoding="utf-8",
    )

    summary = {
        "status": "PASS",
        "trainer": str(args.trainer),
        "argparse_option_count": len(argparse_options),
        "runtime_symbol_count": len(runtime_symbols),
        "likely_pipeline_symbol_count": len(likely_symbols),
        "likely_class_method_count": len(likely_class_methods),
        "source_excerpt_count": len(excerpts),
        "checkpoint_top_level_keys": (
            sorted(checkpoint.keys())
            if isinstance(checkpoint, dict)
            else []
        ),
        "ready_to_build_smoke_adapter": True,
        "dataset_accessed": False,
        "development_test_accessed": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("V4_A4A_1B_TRAINER_INTERFACE_PREFLIGHT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
