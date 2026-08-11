from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "V5_P3_F7_E1_RECONSTRUCTED_CONTROL_AND_GROUP_ABLATION_TRAINER_GENERATION"
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
    "src/models/v5_p3_f7_reconstructed_runtime_adapter_e1.py"
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
EXPECTED_MODEL_CLASS = "V6P0Dynamic70GraphConvCount4"
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


def atomic_text(path: Path, value: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)
    if mode is not None:
        path.chmod(mode)


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


def imported_local_name(alias: ast.alias) -> str:
    return alias.asname or alias.name


def reconstruct_import(
    module: str,
    level: int,
    aliases: list[ast.alias],
) -> str:
    dots = "." * int(level)
    rendered = ", ".join(
        (
            f"{alias.name} as {alias.asname}"
            if alias.asname
            else alias.name
        )
        for alias in aliases
    )
    return f"from {dots}{module} import {rendered}"


def replace_line_ranges(
    lines: list[str],
    replacements: list[tuple[int, int, list[str]]],
) -> list[str]:
    output = list(lines)
    for start, end, new_lines in sorted(
        replacements,
        key=lambda row: row[0],
        reverse=True,
    ):
        output[start - 1:end] = new_lines
    return output


def exact_function_source(
    text: str,
    tree: ast.Module,
    name: str,
) -> tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    require(len(matches) == 1, f"{name} function is not unique")
    source = ast.get_source_segment(text, matches[0])
    require(source is not None, f"could not extract {name} source")
    return source, matches[0]


def find_selected_import(
    tree: ast.Module,
    *,
    module_token: str,
    used_constructor_names: set[str],
) -> tuple[ast.ImportFrom, ast.alias]:
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if module_token not in module:
            continue
        for alias in node.names:
            local = imported_local_name(alias)
            if local in used_constructor_names:
                candidates.append((node, alias))

    require(
        len(candidates) == 1,
        f"expected one {module_token} constructor import, found "
        f"{[(row[0].module, row[1].name, imported_local_name(row[1])) for row in candidates]}",
    )
    return candidates[0]


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
                option for option in options if option.startswith("--")
            ]
            selected = long_options[-1] if long_options else options[-1]
            dest = selected.lstrip("-").replace("-", "_")
        else:
            continue

        required_node = keywords.get("required")
        required = bool(
            isinstance(required_node, ast.Constant)
            and required_node.value is True
        )
        default_node = keywords.get("default")
        try:
            default = (
                ast.literal_eval(default_node)
                if default_node is not None
                else None
            )
        except Exception:
            default = (
                {"source": ast.unparse(default_node)}
                if default_node is not None
                else None
            )

        rows.append({
            "dest": dest,
            "options": options,
            "required": required,
            "default": default,
            "line_start": int(node.lineno),
            "source": ast.unparse(node),
        })
    return rows


def choose_cli_option(
    rows: list[dict[str, Any]],
    *,
    exact_dests: tuple[str, ...] = (),
    token_sets: tuple[tuple[str, ...], ...] = (),
    optional: bool = False,
) -> dict[str, Any] | None:
    candidates = []

    for row in rows:
        dest = normalize(row["dest"])
        score = 0

        if dest in {normalize(value) for value in exact_dests}:
            score += 100

        for tokens in token_sets:
            normalized_tokens = tuple(normalize(token) for token in tokens)
            if all(token in dest for token in normalized_tokens):
                score += 20 * len(tokens)

        if score > 0:
            candidates.append((score, row))

    candidates.sort(key=lambda item: (-item[0], item[1]["line_start"]))
    if not candidates:
        if optional:
            return None
        raise RuntimeError(
            f"could not resolve CLI option for exact={exact_dests}, "
            f"tokens={token_sets}"
        )

    best_score = candidates[0][0]
    best = [row for score, row in candidates if score == best_score]
    require(
        len(best) == 1,
        f"CLI option ambiguous: {[row['dest'] for row in best]}",
    )
    row = dict(best[0])
    require(row["options"], f"CLI option has no option strings: {row}")
    preferred = [
        option for option in row["options"] if option.startswith("--")
    ]
    row["selected_option"] = preferred[-1] if preferred else row["options"][0]
    return row


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
    spec = importlib.util.spec_from_file_location(module_name, path)
    _require(spec is not None and spec.loader is not None, f"cannot import {path}")
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
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _reset_leaf_modules(model: torch.nn.Module, seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    checkpoint_hash = _state_hash(model)
    reset_rows = []

    for name, module in model.named_modules():
        if name == "":
            continue
        if list(module.children()):
            continue
        reset = getattr(module, "reset_parameters", None)
        if not callable(reset):
            continue
        reset()
        reset_rows.append({
            "name": name,
            "type": f"{module.__class__.__module__}.{module.__class__.__name__}",
        })

    fresh_hash = _state_hash(model)
    _require(reset_rows, "no leaf reset_parameters routes found")
    _require(fresh_hash != checkpoint_hash, "fresh state equals checkpoint state")
    _require(
        all(torch.isfinite(value).all().item() for value in model.state_dict().values()),
        "fresh reset produced nonfinite state",
    )
    return {
        "seed": seed,
        "checkpoint_state_hash": checkpoint_hash,
        "fresh_state_hash": fresh_hash,
        "reset_leaf_module_count": len(reset_rows),
        "reset_leaf_modules": reset_rows,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _build_fresh_model() -> torch.nn.Module:
    repo = Path(os.environ["F7_REPO"]).expanduser().resolve()
    data_link = Path(os.environ["F7_DATA_LINK"]).expanduser()
    run_dir = Path(os.environ["F7_RUN_DIR"]).expanduser().resolve()
    seed = int(os.environ.get("F7_SEED", "107"))

    exporter = (
        repo
        / "scripts/v5/p3/run_v5_p3_d1_immutable_validation_logit_export.py"
    )
    checkpoint = (
        repo
        / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107/"
        "V5_P3_A4_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_SEED107_BEST.pt"
    )
    f6_source = (
        repo
        / "scripts/v5/p3/experiments/"
        "run_v5_p3_f6_task_specific_integrated_gradients.py"
    )

    for path in (exporter, checkpoint, f6_source):
        _require(path.is_file(), f"required model-construction artifact missing: {path}")
    _require(data_link.is_symlink(), f"dataset symlink missing: {data_link}")

    f6 = _import_source(
        f6_source,
        "_v5_p3_f7_runtime_model_capture",
    )

    try:
        checkpoint_object = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint_object = torch.load(checkpoint, map_location="cpu")

    checkpoint_state = f6.extract_state_dict(checkpoint_object)
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
        f"captured model class changed: {model.__class__.__name__}",
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

    _atomic_json(
        run_dir / "F7_MODEL_INITIALIZATION_CERTIFICATION.json",
        {
            "model_class": model.__class__.__name__,
            "parameter_count": _parameter_count(model),
            "seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "exporter": str(exporter),
            "exporter_sha256": _sha256_file(exporter),
            "capture": capture,
            "fresh_reset": reset,
            "historical_A4_initialization_claimed": False,
        },
    )
    return model


class F7Dynamic70ModelFactory:
    def __new__(cls, *args, **kwargs):
        return _build_fresh_model()


def _find_split(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    candidates = []

    for key, value in kwargs.items():
        if key.lower() in {"split", "partition", "subset", "mode"}:
            candidates.append(str(value))

    for value in args:
        if isinstance(value, str):
            candidates.append(value)
        elif isinstance(value, Path):
            candidates.append(str(value))

    for candidate in candidates:
        token = candidate.lower()
        if "train" in token:
            return "train"
        if "validation" in token or token.endswith("/val") or token == "val":
            return "validation"

    raise RuntimeError(
        "could not infer train/validation split from dataset constructor "
        f"args={args!r}, kwargs={kwargs!r}"
    )


def _mask_feature_payload(value: Any, channel_range: tuple[int, int] | None) -> tuple[Any, int]:
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
            transformed, child_count = _mask_feature_payload(child, channel_range)
            output[key] = transformed
            count += child_count
        return output, count

    if isinstance(value, list):
        output = []
        count = 0
        for child in value:
            transformed, child_count = _mask_feature_payload(child, channel_range)
            output.append(transformed)
            count += child_count
        return output, count

    if isinstance(value, tuple):
        transformed_values = []
        count = 0
        for child in value:
            transformed, child_count = _mask_feature_payload(child, channel_range)
            transformed_values.append(transformed)
            count += child_count
        if hasattr(value, "_fields"):
            return type(value)(*transformed_values), count
        return tuple(transformed_values), count

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
            transformed, child_count = _mask_feature_payload(child, channel_range)
            try:
                setattr(output, key, transformed)
            except Exception:
                pass
            count += child_count
        return output, count

    return value, 0


class F7Dynamic70DatasetAdapter(Dataset):
    def __init__(self, *args, **kwargs):
        repo = Path(os.environ["F7_REPO"]).expanduser().resolve()
        data_link = Path(os.environ["F7_DATA_LINK"]).expanduser()
        label = os.environ.get("F7_ABLATION_LABEL", "control_dynamic70")
        _require(label in GROUP_CHANNELS, f"unknown F7 ablation label: {label}")

        split = _find_split(args, kwargs)
        loader_path = repo / "src/data/v5_p3_tranche_a_guarded_loader.py"
        _require(loader_path.is_file(), f"guarded loader missing: {loader_path}")
        loader = _import_source(
            loader_path,
            "_v5_p3_f7_runtime_guarded_loader",
        )

        data_root = data_link.resolve()
        dataset_class = loader._load_original_class(data_root)
        self._inner = loader._construct_original(
            dataset_class,
            data_root,
            split,
            {},
        )
        self._split = split
        self._label = label
        self._channel_range = GROUP_CHANNELS[label]

        expected = 110855 if split == "train" else 13863
        _require(
            len(self._inner) == expected,
            f"{split} dataset length changed: {len(self._inner)}",
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
            _require(count == 0, "control path unexpectedly transformed a feature tensor")
        else:
            _require(
                count == 1,
                f"expected exactly one Dynamic70 feature tensor, found {count}",
            )
        return transformed

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)
'''


def matrix_launcher_source(
    *,
    trainer_path: Path,
    matrix_dir: Path,
    cli_map: dict[str, Any],
    output_root: Path,
) -> str:
    seed_option = shlex.quote(cli_map["seed"]["selected_option"])
    data_option = shlex.quote(cli_map["data"]["selected_option"])
    output_option = shlex.quote(cli_map["output"]["selected_option"])
    device_option = (
        shlex.quote(cli_map["device"]["selected_option"])
        if cli_map.get("device")
        else ""
    )

    device_fragment = (
        f'  CMD+=({device_option} "$DEVICE")\n'
        if device_option
        else ""
    )

    return f'''#!/usr/bin/env bash
set -euo pipefail

REPO="${{1:-$HOME/research/projects/GNN-2d}}"
DATA_LINK="${{2:-$REPO/data/processed/v5/p3_1500_d70_tranche_a_preliminary}}"
PYTHON="${{PYTHON:-$REPO/.venv/bin/python}}"
DEVICE="${{DEVICE:-cuda}}"

E2_LOCK="$REPO/reports/v5/p3_experiments/f0_d70_feature_study/retrained_group_ablation/V5_P3_F7_E2_GENERATED_TRAINER_STATIC_AND_RUNTIME_PREFLIGHT_LOCK.json"
TRAINER={shlex.quote(str(trainer_path))}
MATRIX_DIR={shlex.quote(str(matrix_dir))}
RUN_ROOT={shlex.quote(str(output_root))}

[[ -f "$E2_LOCK" ]] || {{
  echo "F7 E2 execution lock missing: $E2_LOCK" >&2
  exit 1
}}

"$PYTHON" - "$E2_LOCK" <<'PY_E2_LOCK'
import json
import sys
from pathlib import Path
lock = json.loads(Path(sys.argv[1]).read_text())
assert lock.get("status") == "PASS"
assert lock.get("primary_matrix_execution_authorized") is True
assert lock.get("sealed_test_tensors_loaded") is False
PY_E2_LOCK

mkdir -p "$RUN_ROOT"

for SPEC in "$MATRIX_DIR"/*.json; do
  LABEL="$("$PYTHON" - "$SPEC" <<'PY_SPEC'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["label"])
PY_SPEC
)"
  RUN_DIR="$RUN_ROOT/$LABEL"
  COMPLETE="$RUN_DIR/F7_RUN_COMPLETE"
  if [[ -f "$COMPLETE" ]]; then
    echo "skip complete run: $LABEL"
    continue
  fi

  mkdir -p "$RUN_DIR"
  export F7_REPO="$REPO"
  export F7_DATA_LINK="$DATA_LINK"
  export F7_RUN_DIR="$RUN_DIR"
  export F7_SEED="107"
  export F7_ABLATION_LABEL="$LABEL"

  CMD=(
    "$PYTHON"
    "$TRAINER"
    {seed_option} "107"
    {data_option} "$DATA_LINK"
    {output_option} "$RUN_DIR"
  )
{device_fragment}
  printf 'launching %s\\n' "$LABEL"
  printf 'command:'
  printf ' %q' "${{CMD[@]}}"
  printf '\\n'
  "${{CMD[@]}}" 2>&1 | tee "$RUN_DIR/console.log"
done
'''


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

    r1_report_path = f7_root / (
        "V5_P3_F7_E0D_R1_EXACT_EARLY_STOP_POLICY_PIN_REPORT.json"
    )
    r1_lock_path = f7_root / (
        "V5_P3_F7_E0D_R1_EXACT_EARLY_STOP_POLICY_PIN_LOCK.json"
    )
    final_recipe_path = f7_root / (
        "F7_E0D_R1_FINAL_RECONSTRUCTED_EXECUTION_RECIPE.json"
    )
    p2_lock_path = f7_root / (
        "V5_P3_F7_P2_RECONSTRUCTED_TRAINER_ADAPTER_AND_DRY_RUN_PREFLIGHT_LOCK.json"
    )

    r1_report, r1_lock = verify_report_lock(
        r1_report_path,
        r1_lock_path,
    )
    final_recipe = load_json(final_recipe_path)
    p2_lock = load_json(p2_lock_path)

    require(
        r1_lock.get("E0D_R1_complete") is True,
        "E0D-R1 incomplete",
    )
    require(
        r1_lock.get("final_execution_recipe_frozen") is True,
        "final execution recipe not frozen",
    )
    require(
        r1_lock.get("E1_trainer_generation_authorized") is True,
        "E1 trainer generation not authorized",
    )
    require(
        r1_lock.get("actual_scientific_training_started") is False,
        "scientific training already started",
    )
    require(
        p2_lock.get("actual_F7_retraining_authorized") is True,
        "P2 did not authorize the F7 experiment",
    )
    require(
        p2_lock.get("sealed_test_tensors_loaded") is False,
        "sealed-test access detected",
    )
    require(final_recipe["status"] == "FROZEN", "recipe is not frozen")
    require(final_recipe["unresolved_fields"] == [], "recipe still unresolved")

    skeleton_path = (repo / SKELETON_RELATIVE_PATH).resolve()
    require(skeleton_path.is_file(), f"missing skeleton: {skeleton_path}")
    require(
        sha256_file(skeleton_path) == EXPECTED_SKELETON_SHA256,
        "training skeleton hash changed",
    )

    skeleton_text = skeleton_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    skeleton_lines = skeleton_text.splitlines()
    skeleton_tree = ast.parse(skeleton_text)

    compute_loss_source, _ = exact_function_source(
        skeleton_text,
        skeleton_tree,
        "compute_loss",
    )
    require(
        sha256_text(compute_loss_source) == EXPECTED_COMPUTE_LOSS_SHA256,
        "compute_loss source hash changed",
    )

    used_constructor_names = {
        call_name(node.func)
        for node in ast.walk(skeleton_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
    }

    model_import_node, model_alias = find_selected_import(
        skeleton_tree,
        module_token="models",
        used_constructor_names=used_constructor_names,
    )
    data_import_node, data_alias = find_selected_import(
        skeleton_tree,
        module_token="data",
        used_constructor_names=used_constructor_names,
    )

    model_local = imported_local_name(model_alias)
    data_local = imported_local_name(data_alias)

    replacements = []

    for node, selected_alias, adapter_symbol in (
        (
            model_import_node,
            model_alias,
            "F7Dynamic70ModelFactory",
        ),
        (
            data_import_node,
            data_alias,
            "F7Dynamic70DatasetAdapter",
        ),
    ):
        remaining = [
            alias
            for alias in node.names
            if alias is not selected_alias
        ]
        new_lines = []
        if remaining:
            new_lines.append(
                reconstruct_import(
                    node.module or "",
                    int(node.level or 0),
                    remaining,
                )
            )
        new_lines.append(
            "from src.models.v5_p3_f7_reconstructed_runtime_adapter_e1 "
            f"import {adapter_symbol} as {imported_local_name(selected_alias)}"
        )
        replacements.append(
            (
                int(node.lineno),
                int(getattr(node, "end_lineno", node.lineno)),
                new_lines,
            )
        )

    generated_lines = replace_line_ranges(
        skeleton_lines,
        replacements,
    )
    provenance_header = [
        "# AUTO-GENERATED BY V5-P3 F7-E1.",
        "# Source skeleton SHA-256: " + EXPECTED_SKELETON_SHA256,
        "# This is a reconstructed F7 route, not the missing historical A4 trainer.",
        "",
    ]
    generated_text = "\n".join(
        provenance_header + generated_lines
    ) + "\n"
    generated_tree = ast.parse(generated_text)

    generated_compute_loss, _ = exact_function_source(
        generated_text,
        generated_tree,
        "compute_loss",
    )
    require(
        sha256_text(generated_compute_loss) == EXPECTED_COMPUTE_LOSS_SHA256,
        "generated trainer changed compute_loss",
    )

    early_conditions = [
        ast.unparse(node.test)
        for node in ast.walk(generated_tree)
        if isinstance(node, ast.If)
        and any(isinstance(child, ast.Break) for child in ast.walk(node))
        and "early_stop_patience_counter" in ast.unparse(node.test)
    ]
    require(
        early_conditions == [
            "epoch >= 15 and early_stop_patience_counter >= 12"
        ],
        f"generated early-stop policy changed: {early_conditions}",
    )

    generated_trainer_path = (
        repo / GENERATED_TRAINER_RELATIVE_PATH
    ).resolve()
    adapter_path = (
        repo / RUNTIME_ADAPTER_RELATIVE_PATH
    ).resolve()
    matrix_launcher_path = (
        repo / MATRIX_LAUNCHER_RELATIVE_PATH
    ).resolve()

    adapter_text = runtime_adapter_source()
    ast.parse(adapter_text)

    cli_rows = argparse_inventory(generated_tree)
    cli_map = {
        "seed": choose_cli_option(
            cli_rows,
            exact_dests=("seed",),
            token_sets=(("seed",),),
        ),
        "data": choose_cli_option(
            cli_rows,
            exact_dests=(
                "data",
                "data_root",
                "dataset",
                "dataset_root",
            ),
            token_sets=(
                ("data",),
                ("dataset",),
            ),
        ),
        "output": choose_cli_option(
            cli_rows,
            exact_dests=(
                "output_dir",
                "report_dir",
                "run_dir",
                "output",
            ),
            token_sets=(
                ("output",),
                ("report", "dir"),
                ("run", "dir"),
            ),
        ),
        "device": choose_cli_option(
            cli_rows,
            exact_dests=("device",),
            token_sets=(("device",),),
            optional=True,
        ),
    }

    selected_dests = {
        row["dest"]
        for row in cli_map.values()
        if row is not None
    }
    unbound_required = [
        row
        for row in cli_rows
        if row["required"] and row["dest"] not in selected_dests
    ]
    require(
        not unbound_required,
        f"required CLI arguments remain unbound: "
        f"{[row['dest'] for row in unbound_required]}",
    )

    atomic_text(
        generated_trainer_path,
        generated_text,
        mode=0o555,
    )
    atomic_text(
        adapter_path,
        adapter_text,
        mode=0o444,
    )

    matrix_dir = f7_root / "F7_E1_PRIMARY_SEED107_RUN_SPECS"
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
            "mask_operation": (
                "none"
                if channel_range is None
                else (
                    f"post-loader Dynamic70 x[:,{channel_range[0]}:"
                    f"{channel_range[1]},:] = 0"
                )
            ),
            "physical_port_mask_changed": False,
            "edge_index_changed": False,
            "train_items": EXPECTED_TRAIN_ITEMS,
            "validation_items": EXPECTED_VALIDATION_ITEMS,
            "sealed_test_access": False,
        }
        spec_path = matrix_dir / f"{label}.json"
        atomic_json(spec_path, spec)
        spec_paths.append(spec_path)

    run_root = f7_root / "f7_runs/seed107"
    launcher_text = matrix_launcher_source(
        trainer_path=generated_trainer_path,
        matrix_dir=matrix_dir,
        cli_map=cli_map,
        output_root=run_root,
    )
    atomic_text(
        matrix_launcher_path,
        launcher_text,
        mode=0o555,
    )

    compile(
        generated_text,
        str(generated_trainer_path),
        "exec",
    )
    compile(
        adapter_text,
        str(adapter_path),
        "exec",
    )

    source_map_path = output_dir / (
        "F7_E1_GENERATED_SOURCE_AND_PATCH_MAP.json"
    )
    matrix_manifest_path = output_dir / (
        "F7_E1_PRIMARY_SEED107_MATRIX_MANIFEST.json"
    )
    cli_map_path = output_dir / (
        "F7_E1_GENERATED_TRAINER_CLI_BINDING_MAP.json"
    )
    authorization_path = output_dir / (
        "F7_E1_E2_PREFLIGHT_AUTHORIZATION.json"
    )

    source_map = {
        "skeleton": str(skeleton_path),
        "skeleton_sha256": sha256_file(skeleton_path),
        "generated_trainer": str(generated_trainer_path),
        "generated_trainer_sha256": sha256_file(generated_trainer_path),
        "runtime_adapter": str(adapter_path),
        "runtime_adapter_sha256": sha256_file(adapter_path),
        "matrix_launcher": str(matrix_launcher_path),
        "matrix_launcher_sha256": sha256_file(matrix_launcher_path),
        "model_import_patch": {
            "original_module": model_import_node.module,
            "original_symbol": model_alias.name,
            "local_symbol": model_local,
            "replacement_factory": "F7Dynamic70ModelFactory",
        },
        "dataset_import_patch": {
            "original_module": data_import_node.module,
            "original_symbol": data_alias.name,
            "local_symbol": data_local,
            "replacement_adapter": "F7Dynamic70DatasetAdapter",
        },
        "compute_loss_sha256": sha256_text(generated_compute_loss),
        "early_stop_condition": early_conditions[0],
        "historical_A4_trainer_claimed": False,
    }
    atomic_json(source_map_path, source_map)

    matrix_manifest = {
        "primary_seed": EXPECTED_SEED,
        "run_count": len(spec_paths),
        "run_specs": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "label": path.stem,
            }
            for path in spec_paths
        ],
        "execution_order": list(GROUPS),
        "control_run_included": True,
        "F8_multi_seed_included": False,
        "sealed_test_access": False,
    }
    atomic_json(matrix_manifest_path, matrix_manifest)
    atomic_json(
        cli_map_path,
        {
            "argparse_inventory": cli_rows,
            "selected_bindings": cli_map,
            "unbound_required_arguments": unbound_required,
        },
    )

    authorization = {
        "E0D_R1_recipe_verified": True,
        "generated_trainer_created": True,
        "runtime_adapter_created": True,
        "compute_loss_hash_preserved": True,
        "early_stop_policy_preserved": True,
        "six_run_matrix_generated": True,
        "primary_seed": EXPECTED_SEED,
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
            "V5_P3_F7_E2_GENERATED_TRAINER_STATIC_AND_RUNTIME_PREFLIGHT"
        ),
    }
    atomic_json(authorization_path, authorization)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN,
        "classification": CLASSIFICATION,
        "scope": (
            "Generate, but do not execute, the frozen reconstructed F7 "
            "trainer: copy the verified V5-P2 training skeleton; replace only "
            "the selected model and dataset imports with isolated Dynamic70 "
            "runtime adapters; preserve the exact compute_loss and early-stop "
            "source; create six seed-107 same-width run specifications; create "
            "an execution launcher guarded by a future E2 lock; statically "
            "compile all generated Python; and authorize only E2 preflight."
        ),
        "finding": {
            "source_map": source_map,
            "matrix_manifest": matrix_manifest,
            "cli_map": cli_map,
        },
        "decision": authorization,
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
            "generated_launcher_execution_guarded_by_E2": True,
            "historical_A4_trainer_claimed": False,
        },
        "artifacts": {
            "generated_trainer": str(generated_trainer_path),
            "runtime_adapter": str(adapter_path),
            "matrix_launcher": str(matrix_launcher_path),
            "run_spec_directory": str(matrix_dir),
            "source_map": str(source_map_path),
            "matrix_manifest": str(matrix_manifest_path),
            "cli_map": str(cli_map_path),
            "E2_authorization": str(authorization_path),
        },
        "provenance": {
            "E0D_R1_report_sha256": sha256_file(r1_report_path),
            "E0D_R1_lock_sha256": sha256_file(r1_lock_path),
            "final_recipe_sha256": sha256_file(final_recipe_path),
            "P2_lock_sha256": sha256_file(p2_lock_path),
            "skeleton_sha256": sha256_file(skeleton_path),
            "installed_script_sha256": sha256_file(installed_script),
            "generated_trainer_sha256": sha256_file(generated_trainer_path),
            "runtime_adapter_sha256": sha256_file(adapter_path),
            "matrix_launcher_sha256": sha256_file(matrix_launcher_path),
            "source_map_sha256": sha256_file(source_map_path),
            "matrix_manifest_sha256": sha256_file(matrix_manifest_path),
            "cli_map_sha256": sha256_file(cli_map_path),
            "authorization_sha256": sha256_file(authorization_path),
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
            "generated_trainer_sha256": sha256_file(generated_trainer_path),
            "runtime_adapter_sha256": sha256_file(adapter_path),
            "matrix_launcher_sha256": sha256_file(matrix_launcher_path),
            "source_map_sha256": sha256_file(source_map_path),
            "matrix_manifest_sha256": sha256_file(matrix_manifest_path),
            "cli_map_sha256": sha256_file(cli_map_path),
            "authorization_sha256": sha256_file(authorization_path),
            "E1_complete": True,
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
        "model_import_patch="
        f"{model_import_node.module}.{model_alias.name}"
        f"->{model_local}=F7Dynamic70ModelFactory"
    )
    print(
        "dataset_import_patch="
        f"{data_import_node.module}.{data_alias.name}"
        f"->{data_local}=F7Dynamic70DatasetAdapter"
    )
    print(f"compute_loss_sha256={sha256_text(generated_compute_loss)}")
    print(f"early_stop_condition={early_conditions[0]}")
    print(f"generated_trainer={generated_trainer_path}")
    print(f"generated_trainer_sha256={sha256_file(generated_trainer_path)}")
    print(f"runtime_adapter={adapter_path}")
    print(f"runtime_adapter_sha256={sha256_file(adapter_path)}")
    print(f"matrix_launcher={matrix_launcher_path}")
    print(f"matrix_launcher_sha256={sha256_file(matrix_launcher_path)}")
    print(f"primary_seed={EXPECTED_SEED}")
    print(f"primary_run_count={len(spec_paths)}")
    for path in spec_paths:
        spec = load_json(path)
        print(
            f"generated_run_spec="
            f"{spec['label']}:"
            f"channels={spec['channel_range_zeroed']}:"
            f"sha256={sha256_file(path)}"
        )
    print(
        "cli_seed_option="
        f"{cli_map['seed']['selected_option']}"
    )
    print(
        "cli_data_option="
        f"{cli_map['data']['selected_option']}"
    )
    print(
        "cli_output_option="
        f"{cli_map['output']['selected_option']}"
    )
    print(
        "cli_device_option="
        f"{cli_map['device']['selected_option'] if cli_map['device'] else None}"
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
    print("sealed_test_tensors_loaded=false")
    print(
        "next_stage="
        "V5_P3_F7_E2_GENERATED_TRAINER_STATIC_AND_RUNTIME_PREFLIGHT"
    )
    print(f"source_map={source_map_path}")
    print(f"matrix_manifest={matrix_manifest_path}")
    print(f"cli_map={cli_map_path}")
    print(f"E2_authorization={authorization_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
