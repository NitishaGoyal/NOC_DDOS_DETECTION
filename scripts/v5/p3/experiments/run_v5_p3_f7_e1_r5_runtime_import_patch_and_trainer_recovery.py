from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = (
    "V5_P3_F7_E1_R5_RUNTIME_IMPORT_USE_SITE_PATCH_"
    "AND_GENERATED_TRAINER_RECOVERY"
)
CAMPAIGN = "V5-P3-F0-D70-FEATURE-STUDY"
CLASSIFICATION = "TRAIN-DERIVED"

SKELETON_RELATIVE_PATH = Path(
    "scripts/v5/p2/train_v5_p2_b2_single_seed.py"
)
GENERATED_TRAINER_RELATIVE_PATH = Path(
    "scripts/v5/p3/experiments/"
    "run_v5_p3_f7_reconstructed_group_ablation_trainer.py"
)
RUNTIME_ADAPTER_RELATIVE_PATH = Path(
    "src/models/v5_p3_f7_r5_runtime_adapter.py"
)
MATRIX_LAUNCHER_RELATIVE_PATH = Path(
    "scripts/v5/p3/experiments/"
    "run_v5_p3_f7_primary_seed107_matrix.sh"
)

EXPECTED_SKELETON_SHA256 = (
    "8fa354698056653a4ff9d467047fced850ff36c3bdef557b08cbd58aa3d3ae1c"
)
EXPECTED_COMPUTE_LOSS_SHA256 = (
    "d9629317801a72a1905dbcd49eace1418f15014df3a2f55d097c25ae4997155f"
)
EXPECTED_EARLY_STOP = (
    "epoch >= 15 and early_stop_patience_counter >= 12"
)
EXPECTED_PARAMETER_COUNT = 60553
EXPECTED_SEED = 107
EXPECTED_TRAIN_ITEMS = 110855
EXPECTED_VALIDATION_ITEMS = 13863

GROUPS = {
    "control_dynamic70": None,
    "ablate_directional_traffic_volume": [0, 10],
    "ablate_inter_flit_timing": [10, 30],
    "ablate_queue_activity": [30, 40],
    "ablate_buffer_pressure": [40, 55],
    "ablate_flow_control_stalls": [55, 70],
}

METHOD_WRAPPERS = {
    "to",
    "cuda",
    "cpu",
    "eval",
    "train",
    "float",
    "double",
    "half",
    "bfloat16",
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


def atomic_text(
    path: Path,
    value: str,
    mode: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)
    if mode is not None:
        path.chmod(mode)


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


def line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def absolute_span(
    node: ast.AST,
    offsets: list[int],
) -> tuple[int, int]:
    start = offsets[int(node.lineno) - 1] + int(node.col_offset)
    end = (
        offsets[int(node.end_lineno) - 1]
        + int(node.end_col_offset)
    )
    return start, end


def apply_replacements(
    text: str,
    replacements: list[tuple[int, int, str]],
) -> str:
    output = text
    for start, end, value in sorted(
        replacements,
        key=lambda row: row[0],
        reverse=True,
    ):
        output = output[:start] + value + output[end:]
    return output


def exact_function_source(
    text: str,
    tree: ast.Module,
    name: str,
) -> str:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    require(len(matches) == 1, f"{name} function is not unique")
    source = ast.get_source_segment(text, matches[0])
    require(source is not None, f"could not extract {name}")
    return source


def deepest_constructor_call(call: ast.Call) -> ast.Call:
    current = call
    while (
        isinstance(current.func, ast.Attribute)
        and current.func.attr in METHOD_WRAPPERS
        and isinstance(current.func.value, ast.Call)
    ):
        current = current.func.value
    return current


def assignment_matches(
    tree: ast.Module,
    target: str,
) -> list[ast.Assign | ast.AnnAssign]:
    matches = []

    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Assign):
            for target_node in node.targets:
                names.extend(target_names(target_node))
        elif isinstance(node, ast.AnnAssign):
            names.extend(target_names(node.target))
        else:
            continue

        if target in names:
            matches.append(node)

    return matches


def unique_assignment(
    tree: ast.Module,
    target: str,
) -> ast.Assign | ast.AnnAssign:
    matches = assignment_matches(tree, target)
    require(
        len(matches) == 1,
        f"assignment {target!r} is not unique: "
        f"{[(node.lineno, ast.unparse(node)) for node in matches]}",
    )
    return matches[0]


def assignment_value(
    node: ast.Assign | ast.AnnAssign,
) -> ast.AST:
    value = node.value
    require(value is not None, "assignment has no value")
    return value


def insertion_offset_after_future_imports(
    text: str,
    tree: ast.Module,
    offsets: list[int],
) -> int:
    insertion_line = 1
    body = list(tree.body)

    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        insertion_line = int(body[0].end_lineno) + 1

    for node in body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
        ):
            insertion_line = max(
                insertion_line,
                int(node.end_lineno) + 1,
            )

    return offsets[insertion_line - 1]


def dictionary_key_replacements(
    assignment: ast.Assign | ast.AnnAssign,
    offsets: list[int],
) -> tuple[
    list[tuple[int, int, str]],
    list[dict[str, Any]],
]:
    value = assignment_value(assignment)
    require(
        isinstance(value, ast.Dict),
        f"{ast.unparse(assignment)} is not a dictionary assignment",
    )

    replacements = []
    rows = []

    for dictionary in ast.walk(value):
        if not isinstance(dictionary, ast.Dict):
            continue

        for key, child in zip(dictionary.keys, dictionary.values):
            if not (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
            ):
                continue

            child_source = ast.unparse(child)
            replacement_key = None
            route = None

            if "sha256_file(loader_path)" in child_source:
                replacement_key = (
                    "legacy_compatibility_loader_sha256"
                )
                route = "loader_path"

            elif "sha256_file(model_path)" in child_source:
                replacement_key = (
                    "legacy_compatibility_model_sha256"
                )
                route = "model_path"

            if replacement_key is None:
                continue

            start, end = absolute_span(key, offsets)
            replacements.append(
                (
                    start,
                    end,
                    json.dumps(replacement_key),
                )
            )
            rows.append({
                "old_key": key.value,
                "new_key": replacement_key,
                "value_source": child_source,
                "route": route,
                "line_start": int(key.lineno),
                "line_end": int(key.end_lineno),
            })

    return replacements, rows


def top_level_dictionary_insertion(
    text: str,
    assignment: ast.Assign | ast.AnnAssign,
    offsets: list[int],
    field_source: str,
) -> tuple[int, int, str]:
    value = assignment_value(assignment)
    require(isinstance(value, ast.Dict), "target is not a dictionary")

    _, end = absolute_span(value, offsets)
    require(text[end - 1] == "}", "dictionary closing brace changed")

    line_start = text.rfind("\n", 0, end - 1) + 1
    closing_prefix = text[line_start:end - 1]
    match = re.match(r"[ \t]*", closing_prefix)
    require(match is not None, "could not derive dictionary indentation")
    indent = match.group(0)

    insertion = (
        "\n"
        + indent
        + "    "
        + field_source
        + "\n"
        + indent
    )
    return end - 1, end - 1, insertion


def argparse_dests(tree: ast.Module) -> dict[str, list[str]]:
    output = {}

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
        if not options:
            continue

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
        else:
            long_options = [
                option for option in options
                if option.startswith("--")
            ]
            selected = long_options[-1] if long_options else options[-1]
            dest = selected.lstrip("-").replace("-", "_")

        output[dest] = options

    return output


def runtime_adapter_source() -> str:
    return r'''from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


EXPECTED_PARAMETER_COUNT = 60553
EXPECTED_MODEL_CLASS = "V6P0Dynamic70GraphConvCount4"

GROUP_CHANNELS = {
    "control_dynamic70": None,
    "ablate_directional_traffic_volume": (0, 10),
    "ablate_inter_flit_timing": (10, 30),
    "ablate_queue_activity": (30, 40),
    "ablate_buffer_pressure": (40, 55),
    "ablate_flow_control_stalls": (55, 70),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _import_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    _require(
        spec is not None and spec.loader is not None,
        f"cannot import source: {path}",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parameter_count(model: torch.nn.Module) -> int:
    return int(
        sum(parameter.numel() for parameter in model.parameters())
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _reset_leaf_modules(
    model: torch.nn.Module,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    checkpoint_state_hash = _state_hash(model)
    reset_rows = []

    for name, module in model.named_modules():
        if name == "" or list(module.children()):
            continue

        reset = getattr(module, "reset_parameters", None)
        if not callable(reset):
            continue

        reset()
        reset_rows.append({
            "name": name,
            "type": (
                f"{module.__class__.__module__}."
                f"{module.__class__.__name__}"
            ),
        })

    fresh_state_hash = _state_hash(model)

    _require(reset_rows, "no reset_parameters routes found")
    _require(
        fresh_state_hash != checkpoint_state_hash,
        "fresh reset equals checkpoint state",
    )
    _require(
        all(
            torch.isfinite(value).all().item()
            for value in model.state_dict().values()
        ),
        "fresh reset produced a nonfinite state tensor",
    )

    return {
        "seed": seed,
        "checkpoint_state_hash": checkpoint_state_hash,
        "fresh_state_hash": fresh_state_hash,
        "reset_leaf_module_count": len(reset_rows),
        "reset_leaf_modules": reset_rows,
    }


def _build_fresh_model() -> torch.nn.Module:
    repo = Path(os.environ["F7_REPO"]).expanduser().resolve()
    data_link = Path(os.environ["F7_DATA_LINK"]).expanduser()
    run_dir = Path(os.environ["F7_RUN_DIR"]).expanduser().resolve()
    seed = int(os.environ.get("F7_SEED", "107"))

    exporter = (
        repo
        / "scripts/v5/p3/"
        "run_v5_p3_d1_immutable_validation_logit_export.py"
    )
    checkpoint = (
        repo
        / "reports/v5/"
        "p3_a4_tranche_a_preliminary_diagnostic_seed107/"
        "V5_P3_A4_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_"
        "SEED107_BEST.pt"
    )
    f6_source = (
        repo
        / "scripts/v5/p3/experiments/"
        "run_v5_p3_f6_task_specific_integrated_gradients.py"
    )

    for path in (exporter, checkpoint, f6_source):
        _require(
            path.is_file(),
            f"required certified model artifact missing: {path}",
        )
    _require(
        data_link.is_symlink(),
        f"dataset symlink missing: {data_link}",
    )

    f6 = _import_source(
        f6_source,
        "_v5_p3_f7_r5_model_capture",
    )

    try:
        checkpoint_object = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint_object = torch.load(
            checkpoint,
            map_location="cpu",
        )

    checkpoint_state = f6.extract_state_dict(
        checkpoint_object
    )
    capture_dir = run_dir / "model_capture"
    capture_dir.mkdir(parents=True, exist_ok=True)

    model, capture = f6.capture_model_via_official_exporter(
        exporter_path=exporter,
        repo=repo,
        data_link=data_link,
        run_dir=capture_dir,
        checkpoint_state_dict=checkpoint_state,
    )
    model.to("cpu")

    _require(
        model.__class__.__name__ == EXPECTED_MODEL_CLASS,
        f"captured model class changed: "
        f"{model.__class__.__name__}",
    )
    _require(
        f6.state_dict_exact_match(model, checkpoint_state),
        "captured model does not match certified checkpoint",
    )
    _require(
        _parameter_count(model) == EXPECTED_PARAMETER_COUNT,
        "captured parameter count changed",
    )

    reset = _reset_leaf_modules(model, seed)

    _require(
        _parameter_count(model) == EXPECTED_PARAMETER_COUNT,
        "fresh parameter count changed",
    )

    if not hasattr(model, "architecture_name"):
        model.architecture_name = EXPECTED_MODEL_CLASS

    _atomic_json(
        run_dir / "F7_MODEL_INITIALIZATION_CERTIFICATION.json",
        {
            "model_class": model.__class__.__name__,
            "architecture_name": model.architecture_name,
            "parameter_count": _parameter_count(model),
            "seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "exporter": str(exporter),
            "exporter_sha256": _sha256_file(exporter),
            "capture": capture,
            "fresh_reset": reset,
            "historical_A4_initialization_claimed": False,
            "checkpoint_used_only_to_certify_and_capture_architecture": True,
            "checkpoint_weights_retained_for_training": False,
        },
    )
    return model


class F7Dynamic70ModelFactory:
    def __new__(cls, *args, **kwargs):
        return _build_fresh_model()


def _mask_feature_payload(
    value: Any,
    channel_range: tuple[int, int] | None,
) -> tuple[Any, int]:
    if channel_range is None:
        return value, 0

    start, end = channel_range

    if isinstance(value, torch.Tensor):
        if tuple(value.shape) == (16, 70, 32):
            output = value.clone()
            output[:, start:end, :] = 0
            return output, 1
        return value, 0

    if isinstance(value, np.ndarray):
        if tuple(value.shape) == (16, 70, 32):
            output = np.array(value, copy=True)
            output[:, start:end, :] = 0
            return output, 1
        return value, 0

    if isinstance(value, dict):
        output = {}
        count = 0
        for key, child in value.items():
            transformed, child_count = _mask_feature_payload(
                child,
                channel_range,
            )
            output[key] = transformed
            count += child_count
        return output, count

    if isinstance(value, list):
        output = []
        count = 0
        for child in value:
            transformed, child_count = _mask_feature_payload(
                child,
                channel_range,
            )
            output.append(transformed)
            count += child_count
        return output, count

    if isinstance(value, tuple):
        output = []
        count = 0
        for child in value:
            transformed, child_count = _mask_feature_payload(
                child,
                channel_range,
            )
            output.append(transformed)
            count += child_count

        if hasattr(value, "_fields"):
            return type(value)(*output), count
        return tuple(output), count

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        updates = {}
        count = 0
        for field in dataclasses.fields(value):
            transformed, child_count = _mask_feature_payload(
                getattr(value, field.name),
                channel_range,
            )
            updates[field.name] = transformed
            count += child_count
        return dataclasses.replace(value, **updates), count

    if hasattr(value, "__dict__"):
        output = copy.copy(value)
        count = 0
        for key, child in vars(value).items():
            transformed, child_count = _mask_feature_payload(
                child,
                channel_range,
            )
            try:
                setattr(output, key, transformed)
            except Exception:
                pass
            count += child_count
        return output, count

    return value, 0


class _F7FixedSplitDatasetAdapter(Dataset):
    FIXED_SPLIT = ""

    def __init__(self, *args, **kwargs):
        repo = Path(os.environ["F7_REPO"]).expanduser().resolve()
        data_link = Path(os.environ["F7_DATA_LINK"]).expanduser()
        run_dir = Path(os.environ["F7_RUN_DIR"]).expanduser().resolve()
        label = os.environ.get(
            "F7_ABLATION_LABEL",
            "control_dynamic70",
        )

        _require(
            label in GROUP_CHANNELS,
            f"unknown ablation label: {label}",
        )

        loader_path = (
            repo / "src/data/v5_p3_tranche_a_guarded_loader.py"
        )
        _require(
            loader_path.is_file(),
            f"guarded loader missing: {loader_path}",
        )

        loader = _import_source(
            loader_path,
            f"_v5_p3_f7_r5_loader_{self.FIXED_SPLIT}",
        )

        data_root = data_link.resolve()
        dataset_class = loader._load_original_class(data_root)
        self._inner = loader._construct_original(
            dataset_class,
            data_root,
            self.FIXED_SPLIT,
            {},
        )
        self._channel_range = GROUP_CHANNELS[label]
        self.pair_manifest = (
            run_dir
            / "F7_FIXED_GUARDED_SPLIT_NO_LEGACY_PAIR_MANIFEST.csv"
        )

        expected = (
            110855
            if self.FIXED_SPLIT == "train"
            else 13863
        )
        _require(
            len(self._inner) == expected,
            f"{self.FIXED_SPLIT} length changed: "
            f"{len(self._inner)}",
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int):
        item = self._inner[index]
        transformed, count = _mask_feature_payload(
            item,
            self._channel_range,
        )

        if self._channel_range is None:
            _require(
                count == 0,
                "control path unexpectedly transformed a feature tensor",
            )
        else:
            _require(
                count == 1,
                "expected exactly one [16,70,32] feature tensor; "
                f"found {count}",
            )

        return transformed

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)


class F7TrainDatasetAdapter(_F7FixedSplitDatasetAdapter):
    FIXED_SPLIT = "train"


class F7ValidationDatasetAdapter(_F7FixedSplitDatasetAdapter):
    FIXED_SPLIT = "validation"
'''


def matrix_launcher_source(
    *,
    trainer_path: Path,
    adapter_path: Path,
    matrix_dir: Path,
    run_root: Path,
    repo: Path,
    b1_dir: Path,
    b0_r3_dir: Path,
    legacy_loader_path: Path,
    legacy_model_path: Path,
) -> str:
    lock_check = (
        'import json,sys; from pathlib import Path; '
        'x=json.loads(Path(sys.argv[1]).read_text()); '
        'assert x.get("status")=="PASS"; '
        'assert x.get("primary_matrix_execution_authorized") is True; '
        'assert x.get("sealed_test_tensors_loaded") is False'
    )
    label_read = (
        'import json,sys; '
        'print(json.load(open(sys.argv[1]))["label"])'
    )

    return f'''#!/usr/bin/env bash
set -euo pipefail

REPO="${{1:-{shlex.quote(str(repo))}}}"
DATA_LINK="${{2:-$REPO/data/processed/v5/p3_1500_d70_tranche_a_preliminary}}"
PYTHON="${{PYTHON:-$REPO/.venv/bin/python}}"

E2_LOCK="$REPO/reports/v5/p3_experiments/f0_d70_feature_study/retrained_group_ablation/V5_P3_F7_E2_GENERATED_TRAINER_STATIC_AND_RUNTIME_PREFLIGHT_LOCK.json"
TRAINER={shlex.quote(str(trainer_path))}
ADAPTER={shlex.quote(str(adapter_path))}
MATRIX_DIR={shlex.quote(str(matrix_dir))}
RUN_ROOT={shlex.quote(str(run_root))}
B1_DIR={shlex.quote(str(b1_dir))}
B0_R3_DIR={shlex.quote(str(b0_r3_dir))}
LEGACY_LOADER={shlex.quote(str(legacy_loader_path))}
LEGACY_MODEL={shlex.quote(str(legacy_model_path))}

[[ -f "$E2_LOCK" ]] || {{
  echo "F7 E2 execution lock missing: $E2_LOCK" >&2
  exit 1
}}
"$PYTHON" -c {shlex.quote(lock_check)} "$E2_LOCK"

[[ -f "$TRAINER" ]] || {{
  echo "generated trainer missing: $TRAINER" >&2
  exit 1
}}
[[ -f "$ADAPTER" ]] || {{
  echo "runtime adapter missing: $ADAPTER" >&2
  exit 1
}}

mkdir -p "$RUN_ROOT"

for SPEC in "$MATRIX_DIR"/*.json; do
  LABEL="$("$PYTHON" -c {shlex.quote(label_read)} "$SPEC")"
  RUN_DIR="$RUN_ROOT/$LABEL"
  MODEL_DIR="$RUN_DIR/model"
  REPORT_DIR="$RUN_DIR/report"

  if [[ -f "$REPORT_DIR/F7_RUN_COMPLETE" ]]; then
    echo "skip complete run: $LABEL"
    continue
  fi

  mkdir -p "$MODEL_DIR" "$REPORT_DIR"

  export F7_REPO="$REPO"
  export F7_DATA_LINK="$DATA_LINK"
  export F7_RUN_DIR="$RUN_DIR"
  export F7_SEED="107"
  export F7_ABLATION_LABEL="$LABEL"

  CMD=(
    "$PYTHON"
    "$TRAINER"
    --root "$REPO"
    --b1-dir "$B1_DIR"
    --b0-r3-dir "$B0_R3_DIR"
    --loader-path "$LEGACY_LOADER"
    --model-path "$LEGACY_MODEL"
    --model-dir "$MODEL_DIR"
    --report-dir "$REPORT_DIR"
    --seed "107"
  )

  printf 'launching %s\\n' "$LABEL"
  printf 'command:'
  printf ' %q' "${{CMD[@]}}"
  printf '\\n'

  "${{CMD[@]}}" 2>&1 | tee "$RUN_DIR/console.log"

  touch "$REPORT_DIR/F7_RUN_COMPLETE"
done
'''


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

    p0_report_path = f7_root / (
        "V5_P3_F7_E1_R5_P0_DYNAMIC_IMPORT_AND_HASH_GATE_PIN_REPORT.json"
    )
    p0_lock_path = f7_root / (
        "V5_P3_F7_E1_R5_P0_DYNAMIC_IMPORT_AND_HASH_GATE_PIN_LOCK.json"
    )
    require(p0_report_path.is_file(), "R5-P0 report missing")
    require(p0_lock_path.is_file(), "R5-P0 lock missing")

    p0_report = load_json(p0_report_path)
    p0_lock = load_json(p0_lock_path)

    require(p0_report.get("status") == "PASS", "R5-P0 report not PASS")
    require(
        p0_lock.get("report_sha256") == sha256_file(p0_report_path),
        "R5-P0 report/lock mismatch",
    )
    require(
        p0_lock.get("R5_runtime_patch_generation_authorized") is True,
        "R5 runtime patch generation not authorized",
    )
    require(
        p0_lock.get("primary_matrix_execution_authorized") is False,
        "primary matrix unexpectedly authorized",
    )
    require(
        p0_lock.get("sealed_test_tensors_loaded") is False,
        "sealed-test access detected",
    )

    finding = p0_report["finding"]
    hash_resolution = finding["hash_resolution"]
    patch_plan = finding["patch_plan"]

    require(patch_plan.get("status") == "FROZEN", "patch plan not frozen")

    b1_dir = Path(hash_resolution["b1_dir"]).resolve()
    b0_r3_dir = Path(hash_resolution["b0_r3_dir"]).resolve()
    legacy_loader_path = Path(
        hash_resolution["canonical_legacy_loader_path"]
    ).resolve()
    legacy_model_path = Path(
        hash_resolution["canonical_legacy_model_path"]
    ).resolve()

    require(b1_dir.is_dir(), "frozen B1 directory missing")
    require(b0_r3_dir.is_dir(), "frozen B0-R3 directory missing")
    require(legacy_loader_path.is_file(), "legacy loader missing")
    require(legacy_model_path.is_file(), "legacy model missing")

    require(
        sha256_file(legacy_loader_path)
        == hash_resolution["expected_loader_sha256"],
        "legacy loader hash changed",
    )
    require(
        sha256_file(legacy_model_path)
        == hash_resolution["expected_model_sha256"],
        "legacy model hash changed",
    )

    skeleton_path = (repo / SKELETON_RELATIVE_PATH).resolve()
    require(skeleton_path.is_file(), f"missing skeleton: {skeleton_path}")
    require(
        sha256_file(skeleton_path) == EXPECTED_SKELETON_SHA256,
        "skeleton hash changed",
    )

    original_text = skeleton_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    original_tree = ast.parse(original_text)
    offsets = line_offsets(original_text)

    compute_loss = exact_function_source(
        original_text,
        original_tree,
        "compute_loss",
    )
    require(
        sha256_text(compute_loss) == EXPECTED_COMPUTE_LOSS_SHA256,
        "compute_loss hash changed",
    )

    required_dests = {
        "root",
        "b1_dir",
        "b0_r3_dir",
        "loader_path",
        "model_path",
        "model_dir",
        "report_dir",
        "seed",
    }
    cli_dests = argparse_dests(original_tree)
    require(
        required_dests.issubset(cli_dests),
        f"required CLI route changed: missing "
        f"{sorted(required_dests - set(cli_dests))}",
    )

    replacements: list[tuple[int, int, str]] = []
    patch_rows = []

    import_block = (
        "import pathlib as _f7_pathlib\n"
        "import src.models.v5_p3_f7_r5_runtime_adapter "
        "as _f7_runtime_adapter\n"
        "from src.models.v5_p3_f7_r5_runtime_adapter import (\n"
        "    F7Dynamic70ModelFactory,\n"
        "    F7TrainDatasetAdapter,\n"
        "    F7ValidationDatasetAdapter,\n"
        ")\n"
        "_F7_RUNTIME_ADAPTER_PATH = _f7_pathlib.Path(\n"
        "    _f7_runtime_adapter.__file__\n"
        ").resolve()\n"
    )
    insertion = insertion_offset_after_future_imports(
        original_text,
        original_tree,
        offsets,
    )
    replacements.append((insertion, insertion, import_block))

    for target in (
        "loader_module",
        "model_module",
        "DatasetClass",
        "ModelClass",
    ):
        assignment = unique_assignment(original_tree, target)
        value = assignment_value(assignment)
        start, end = absolute_span(value, offsets)
        replacements.append((start, end, "None"))
        patch_rows.append({
            "target": target,
            "operation": "disable_legacy_runtime_route",
            "old_value": ast.unparse(value),
            "new_value": "None",
            "line_start": int(value.lineno),
            "line_end": int(value.end_lineno),
        })

    constructor_targets = (
        (
            "train_dataset",
            "F7TrainDatasetAdapter",
        ),
        (
            "validation_dataset",
            "F7ValidationDatasetAdapter",
        ),
        (
            "model",
            "F7Dynamic70ModelFactory",
        ),
    )

    for target, replacement_name in constructor_targets:
        assignment = unique_assignment(original_tree, target)
        value = assignment_value(assignment)
        require(
            isinstance(value, ast.Call),
            f"{target} assignment is not a call",
        )
        constructor_call = deepest_constructor_call(value)
        old_function = ast.unparse(constructor_call.func)
        start, end = absolute_span(constructor_call.func, offsets)
        replacements.append((start, end, replacement_name))
        patch_rows.append({
            "target": target,
            "operation": "replace_constructor_function",
            "old_function": old_function,
            "new_function": replacement_name,
            "line_start": int(constructor_call.func.lineno),
            "line_end": int(constructor_call.func.end_lineno),
        })

    stage_assignment = unique_assignment(original_tree, "STAGE")
    stage_value = assignment_value(stage_assignment)
    start, end = absolute_span(stage_value, offsets)
    replacements.append(
        (
            start,
            end,
            json.dumps(
                "V5_P3_F7_SAME_WIDTH_RETRAINED_GROUP_ABLATION"
            ),
        )
    )

    provenance_patch_rows = []
    for target in ("checkpoint_payload", "seed_manifest", "lock"):
        assignment = unique_assignment(original_tree, target)
        key_replacements, rows = dictionary_key_replacements(
            assignment,
            offsets,
        )
        replacements.extend(key_replacements)
        provenance_patch_rows.extend(
            [
                {
                    "dictionary_target": target,
                    **row,
                }
                for row in rows
            ]
        )
        replacements.append(
            top_level_dictionary_insertion(
                original_text,
                assignment,
                offsets,
                (
                    '"runtime_adapter_sha256": '
                    "sha256_file(_F7_RUNTIME_ADAPTER_PATH),"
                ),
            )
        )

    generated_text = apply_replacements(
        original_text,
        replacements,
    )

    literal_replacements = {
        "===== V5 P2-B2 SINGLE-SEED TRAINING =====": (
            "===== V5-P3 F7 SAME-WIDTH RETRAINED GROUP ABLATION ====="
        ),
        "V5_P2_B2_SEED_{seed}_TRAINING_REPORT.json": (
            "V5_P3_F7_SEED_{seed}_TRAINING_REPORT.json"
        ),
        "V5_P2_B2_SEED_{seed}_LOCK.json": (
            "V5_P3_F7_SEED_{seed}_LOCK.json"
        ),
        "V5_P2_B2_MULTI_SEED_FINALIZATION": (
            "V5_P3_F7_PRIMARY_MATRIX_AGGREGATION"
        ),
    }
    for old, new in literal_replacements.items():
        generated_text = generated_text.replace(old, new)

    generated_text = (
        "# AUTO-GENERATED BY V5-P3 F7-E1-R5.\n"
        "# Legacy P2 source hashes are compatibility provenance only.\n"
        "# Legacy loader/model modules are never executed by this trainer.\n"
        + generated_text
    )

    generated_tree = ast.parse(generated_text)

    generated_compute_loss = exact_function_source(
        generated_text,
        generated_tree,
        "compute_loss",
    )
    require(
        sha256_text(generated_compute_loss)
        == EXPECTED_COMPUTE_LOSS_SHA256,
        "generated trainer changed compute_loss",
    )

    early_conditions = [
        ast.unparse(node.test)
        for node in ast.walk(generated_tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Break)
            for child in ast.walk(node)
        )
        and "early_stop_patience_counter"
        in ast.unparse(node.test)
    ]
    require(
        early_conditions == [EXPECTED_EARLY_STOP],
        f"early-stop condition changed: {early_conditions}",
    )

    forbidden_dynamic_imports = []
    for node in ast.walk(generated_tree):
        if not isinstance(node, ast.Call):
            continue
        short = call_name(node.func).rsplit(".", 1)[-1]
        if short != "import_module":
            continue
        source = ast.unparse(node)
        if "loader_path" in source or "model_path" in source:
            forbidden_dynamic_imports.append(source)
    require(
        not forbidden_dynamic_imports,
        f"legacy dynamic import survived: {forbidden_dynamic_imports}",
    )

    forbidden_legacy_classes = [
        node.attr
        for node in ast.walk(generated_tree)
        if isinstance(node, ast.Attribute)
        and node.attr in {
            "V5P2PairAlignedPrimary58Dataset",
            "P2B3Conv1DOnlyCount4",
        }
    ]
    require(
        not forbidden_legacy_classes,
        f"legacy class extraction survived: {forbidden_legacy_classes}",
    )

    adapter_call_counts = {
        "F7Dynamic70ModelFactory": 0,
        "F7TrainDatasetAdapter": 0,
        "F7ValidationDatasetAdapter": 0,
    }
    for node in ast.walk(generated_tree):
        if not isinstance(node, ast.Call):
            continue
        short = call_name(node.func).rsplit(".", 1)[-1]
        if short in adapter_call_counts:
            adapter_call_counts[short] += 1

    require(
        adapter_call_counts == {
            "F7Dynamic70ModelFactory": 1,
            "F7TrainDatasetAdapter": 1,
            "F7ValidationDatasetAdapter": 1,
        },
        f"adapter call count changed: {adapter_call_counts}",
    )

    generated_source = ast.unparse(generated_tree)
    require(
        "b1_lock.get('loader_sha256')" in generated_source
        or 'b1_lock.get("loader_sha256")' in generated_source,
        "frozen B1 loader identity gate was removed",
    )
    require(
        "b1_lock.get('model_sha256')" in generated_source
        or 'b1_lock.get("model_sha256")' in generated_source,
        "frozen B1 model identity gate was removed",
    )
    require(
        "corrected_model_sha256" in generated_source,
        "frozen B0-R3 corrected-model identity gate was removed",
    )

    ambiguous_runtime_keys = []
    runtime_adapter_key_count = 0

    for dictionary in ast.walk(generated_tree):
        if not isinstance(dictionary, ast.Dict):
            continue

        for key, value in zip(dictionary.keys, dictionary.values):
            if not (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
            ):
                continue

            value_source = ast.unparse(value)

            if key.value == "runtime_adapter_sha256":
                runtime_adapter_key_count += 1

            if (
                key.value
                in {
                    "loader_sha256",
                    "model_sha256",
                    "model_source_sha256",
                }
                and (
                    "loader_path" in value_source
                    or "model_path" in value_source
                )
            ):
                ambiguous_runtime_keys.append({
                    "key": key.value,
                    "value": value_source,
                    "line": int(key.lineno),
                })

    require(
        not ambiguous_runtime_keys,
        f"ambiguous legacy provenance keys survived: "
        f"{ambiguous_runtime_keys}",
    )
    require(
        runtime_adapter_key_count == 3,
        "runtime adapter SHA must appear in checkpoint, report and lock; "
        f"found {runtime_adapter_key_count}",
    )

    generated_trainer_path = (
        repo / GENERATED_TRAINER_RELATIVE_PATH
    ).resolve()
    runtime_adapter_path = (
        repo / RUNTIME_ADAPTER_RELATIVE_PATH
    ).resolve()
    matrix_launcher_path = (
        repo / MATRIX_LAUNCHER_RELATIVE_PATH
    ).resolve()

    adapter_text = runtime_adapter_source()
    ast.parse(adapter_text)

    atomic_text(
        generated_trainer_path,
        generated_text,
        mode=0o555,
    )
    atomic_text(
        runtime_adapter_path,
        adapter_text,
        mode=0o444,
    )

    compile(
        generated_text,
        str(generated_trainer_path),
        "exec",
    )
    compile(
        adapter_text,
        str(runtime_adapter_path),
        "exec",
    )

    matrix_dir = (
        f7_root / "F7_E1_R5_PRIMARY_SEED107_RUN_SPECS"
    )
    matrix_dir.mkdir(parents=True, exist_ok=True)
    spec_paths = []

    for label, channel_range in GROUPS.items():
        spec = {
            "label": label,
            "seed": EXPECTED_SEED,
            "classification": CLASSIFICATION,
            "input_channels": 70,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "channel_range_zeroed": channel_range,
            "train_items": EXPECTED_TRAIN_ITEMS,
            "validation_items": EXPECTED_VALIDATION_ITEMS,
            "legacy_loader_module_executed": False,
            "legacy_model_module_executed": False,
            "physical_port_mask_changed": False,
            "edge_index_changed": False,
            "sealed_test_access": False,
        }
        path = matrix_dir / f"{label}.json"
        atomic_json(path, spec)
        spec_paths.append(path)

    run_root = f7_root / "f7_runs/seed107"
    launcher_text = matrix_launcher_source(
        trainer_path=generated_trainer_path,
        adapter_path=runtime_adapter_path,
        matrix_dir=matrix_dir,
        run_root=run_root,
        repo=repo,
        b1_dir=b1_dir,
        b0_r3_dir=b0_r3_dir,
        legacy_loader_path=legacy_loader_path,
        legacy_model_path=legacy_model_path,
    )
    subprocess.run(
        ["bash", "-n"],
        input=launcher_text,
        text=True,
        check=True,
    )
    atomic_text(
        matrix_launcher_path,
        launcher_text,
        mode=0o555,
    )

    patch_manifest_path = output_dir / (
        "F7_E1_R5_GENERATED_TRAINER_PATCH_MANIFEST.json"
    )
    generated_source_map_path = output_dir / (
        "F7_E1_R5_GENERATED_SOURCE_STATIC_CERTIFICATION.json"
    )
    matrix_manifest_path = output_dir / (
        "F7_E1_R5_PRIMARY_SEED107_MATRIX_MANIFEST.json"
    )
    e2_authorization_path = output_dir / (
        "F7_E1_R5_E2_PREFLIGHT_AUTHORIZATION.json"
    )

    patch_manifest = {
        "skeleton": str(skeleton_path),
        "skeleton_sha256": sha256_file(skeleton_path),
        "patch_rows": patch_rows,
        "provenance_key_relabels": provenance_patch_rows,
        "literal_replacements": literal_replacements,
        "legacy_compatibility": {
            "b1_dir": str(b1_dir),
            "b0_r3_dir": str(b0_r3_dir),
            "loader_path": str(legacy_loader_path),
            "loader_sha256": sha256_file(legacy_loader_path),
            "model_path": str(legacy_model_path),
            "model_sha256": sha256_file(legacy_model_path),
            "modules_executed_by_generated_trainer": False,
        },
    }
    atomic_json(patch_manifest_path, patch_manifest)

    static_certification = {
        "generated_trainer": str(generated_trainer_path),
        "generated_trainer_sha256": sha256_file(
            generated_trainer_path
        ),
        "runtime_adapter": str(runtime_adapter_path),
        "runtime_adapter_sha256": sha256_file(
            runtime_adapter_path
        ),
        "compute_loss_sha256": sha256_text(
            generated_compute_loss
        ),
        "early_stop_condition": early_conditions[0],
        "legacy_dynamic_import_call_count": 0,
        "legacy_class_extraction_count": 0,
        "adapter_call_counts": adapter_call_counts,
        "frozen_legacy_hash_gates_preserved": True,
        "ambiguous_runtime_provenance_key_count": 0,
        "runtime_adapter_sha_key_count": (
            runtime_adapter_key_count
        ),
        "historical_A4_trainer_claimed": False,
    }
    atomic_json(generated_source_map_path, static_certification)

    matrix_manifest = {
        "seed": EXPECTED_SEED,
        "run_count": len(spec_paths),
        "execution_order": list(GROUPS),
        "run_specs": [
            {
                "label": load_json(path)["label"],
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in spec_paths
        ],
        "model_dir_policy": (
            "<run_dir>/model; distinct and append-only"
        ),
        "report_dir_policy": (
            "<run_dir>/report; distinct and append-only"
        ),
        "launcher": str(matrix_launcher_path),
        "launcher_sha256": sha256_file(
            matrix_launcher_path
        ),
        "execution_guard": (
            "requires E2 lock with "
            "primary_matrix_execution_authorized=true"
        ),
        "F8_included": False,
        "sealed_test_access": False,
    }
    atomic_json(matrix_manifest_path, matrix_manifest)

    e2_authorization = {
        "R5_P0_verified": True,
        "runtime_import_patch_plan_applied": True,
        "legacy_loader_module_execution_removed": True,
        "legacy_model_module_execution_removed": True,
        "legacy_class_construction_removed": True,
        "frozen_legacy_hash_gates_preserved": True,
        "runtime_adapter_provenance_disambiguated": True,
        "compute_loss_hash_preserved": True,
        "early_stop_policy_preserved": True,
        "six_run_matrix_generated": True,
        "E2_generated_trainer_preflight_authorized": True,
        "primary_matrix_execution_authorized": False,
        "actual_scientific_training_started": False,
        "actual_F7_primary_matrix_execution_started": False,
        "F8_multi_seed_authorized": False,
        "feature_removal_authorized": False,
        "compact_interface_equivalence_authorized": False,
        "hardware_reduction_claim_authorized": False,
        "sealed_test_access": False,
        "next_stage": (
            "V5_P3_F7_E2_GENERATED_TRAINER_"
            "STATIC_AND_RUNTIME_PREFLIGHT"
        ),
    }
    atomic_json(e2_authorization_path, e2_authorization)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "scope": (
            "Generate an isolated F7 trainer by removing both legacy "
            "runtime imports and legacy class construction; preserve the "
            "frozen B1/B0-R3 source-hash identity gates strictly as "
            "compatibility provenance; relabel legacy source hashes in "
            "generated checkpoints/reports/locks; separately record the "
            "actual F7 runtime-adapter hash; preserve compute_loss and the "
            "frozen execution recipe; generate six seed-107 run specs and "
            "an E2-guarded sequential launcher. No generated code is "
            "executed and no model, checkpoint or dataset is loaded."
        ),
        "finding": {
            "patch_manifest": patch_manifest,
            "static_certification": static_certification,
            "matrix_manifest": matrix_manifest,
        },
        "decision": e2_authorization,
        "governance": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "training_dataset_constructed": False,
            "validation_dataset_constructed": False,
            "training_feature_tensors_loaded": False,
            "validation_feature_tensors_loaded": False,
            "legacy_loader_module_executed": False,
            "legacy_model_module_executed": False,
            "generated_trainer_executed": False,
            "runtime_adapter_executed": False,
            "sealed_test_tensors_loaded": False,
            "A_test_loaded": False,
            "optimizer_step_performed": False,
            "scientific_checkpoint_saved": False,
            "scientific_training_started": False,
            "primary_matrix_execution_authorized": False,
        },
        "artifacts": {
            "generated_trainer": str(generated_trainer_path),
            "runtime_adapter": str(runtime_adapter_path),
            "matrix_launcher": str(matrix_launcher_path),
            "matrix_spec_directory": str(matrix_dir),
            "patch_manifest": str(patch_manifest_path),
            "static_certification": str(
                generated_source_map_path
            ),
            "matrix_manifest": str(matrix_manifest_path),
            "E2_authorization": str(e2_authorization_path),
        },
        "provenance": {
            "R5_P0_report_sha256": sha256_file(p0_report_path),
            "R5_P0_lock_sha256": sha256_file(p0_lock_path),
            "skeleton_sha256": sha256_file(skeleton_path),
            "installed_script_sha256": sha256_file(
                installed_script
            ),
            "generated_trainer_sha256": sha256_file(
                generated_trainer_path
            ),
            "runtime_adapter_sha256": sha256_file(
                runtime_adapter_path
            ),
            "matrix_launcher_sha256": sha256_file(
                matrix_launcher_path
            ),
            "patch_manifest_sha256": sha256_file(
                patch_manifest_path
            ),
            "static_certification_sha256": sha256_file(
                generated_source_map_path
            ),
            "matrix_manifest_sha256": sha256_file(
                matrix_manifest_path
            ),
            "E2_authorization_sha256": sha256_file(
                e2_authorization_path
            ),
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
            "generated_trainer_sha256": sha256_file(
                generated_trainer_path
            ),
            "runtime_adapter_sha256": sha256_file(
                runtime_adapter_path
            ),
            "matrix_launcher_sha256": sha256_file(
                matrix_launcher_path
            ),
            "patch_manifest_sha256": sha256_file(
                patch_manifest_path
            ),
            "static_certification_sha256": sha256_file(
                generated_source_map_path
            ),
            "matrix_manifest_sha256": sha256_file(
                matrix_manifest_path
            ),
            "E2_authorization_sha256": sha256_file(
                e2_authorization_path
            ),
            "R5_complete": True,
            "E2_generated_trainer_preflight_authorized": True,
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
            "generated_trainer_executed": False,
            "runtime_adapter_executed": False,
            "sealed_test_tensors_loaded": False,
        },
    )
    atomic_text(complete_path, f"{STAGE}_COMPLETE\n")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"classification={CLASSIFICATION}")
    print(f"skeleton={skeleton_path}")
    print(f"skeleton_sha256={sha256_file(skeleton_path)}")
    print(
        "compute_loss_sha256="
        f"{sha256_text(generated_compute_loss)}"
    )
    print(f"early_stop_condition={early_conditions[0]}")
    print("legacy_dynamic_import_call_count=0")
    print("legacy_class_extraction_count=0")
    print(f"adapter_call_counts={adapter_call_counts}")
    print("frozen_legacy_hash_gates_preserved=true")
    print("ambiguous_runtime_provenance_key_count=0")
    print(
        "runtime_adapter_sha_key_count="
        f"{runtime_adapter_key_count}"
    )
    print(f"generated_trainer={generated_trainer_path}")
    print(
        "generated_trainer_sha256="
        f"{sha256_file(generated_trainer_path)}"
    )
    print(f"runtime_adapter={runtime_adapter_path}")
    print(
        "runtime_adapter_sha256="
        f"{sha256_file(runtime_adapter_path)}"
    )
    print(f"matrix_launcher={matrix_launcher_path}")
    print(
        "matrix_launcher_sha256="
        f"{sha256_file(matrix_launcher_path)}"
    )
    print(f"primary_seed={EXPECTED_SEED}")
    print(f"primary_run_count={len(spec_paths)}")

    for path in spec_paths:
        spec = load_json(path)
        print(
            "generated_run_spec="
            f"{spec['label']}:"
            f"channels={spec['channel_range_zeroed']}:"
            f"sha256={sha256_file(path)}"
        )

    print("E2_generated_trainer_preflight_authorized=true")
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
    print("generated_trainer_executed=false")
    print("runtime_adapter_executed=false")
    print("sealed_test_tensors_loaded=false")
    print(
        "next_stage="
        "V5_P3_F7_E2_GENERATED_TRAINER_"
        "STATIC_AND_RUNTIME_PREFLIGHT"
    )
    print(f"patch_manifest={patch_manifest_path}")
    print(
        "static_certification="
        f"{generated_source_map_path}"
    )
    print(f"matrix_manifest={matrix_manifest_path}")
    print(f"E2_authorization={e2_authorization_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
