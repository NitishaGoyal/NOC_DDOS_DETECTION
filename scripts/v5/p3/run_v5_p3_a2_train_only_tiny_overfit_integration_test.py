from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


STAGE = "V5_P3_A2_TRAIN_ONLY_TINY_OVERFIT_INTEGRATION_TEST"
CAMPAIGN_LABEL = "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic"
SEED = 107
EXPECTED_PARAMETER_COUNT = 60_553
EXPECTED_X_SHAPE = (16, 70, 32)
EXPECTED_MASK_SHAPE = (16, 10)
EXPECTED_EDGE_SHAPE = (2, 48)
EXPECTED_OUTPUT_SHAPES = {
    "attack_logits": (24,),
    "count_logits": (24, 4),
    "source_logits": (24, 16),
    "transit_logits": (24, 16),
    "victim_logits": (24, 16),
    "path_logits": (24, 16),
}
ACTIVE_PER_COUNT = 3
NEGATIVE_COUNT = 12
SUBSET_SIZE = ACTIVE_PER_COUNT * 4 + NEGATIVE_COUNT
MAX_PER_PAIR_PER_GROUP = 2
MAX_EPOCHS = 600
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def walk(value: Any, prefix: str = "sample"):
    if isinstance(value, dict):
        for key in sorted(value, key=lambda item: str(item)):
            yield from walk(value[key], f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from walk(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def as_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, (int, float, bool, np.number)):
        return torch.as_tensor(value)
    return None


def tensor_leaves(sample: Any) -> list[tuple[str, torch.Tensor]]:
    leaves = []
    for path, value in walk(sample):
        tensor = as_tensor(value)
        if tensor is not None:
            leaves.append((path, tensor))
    return leaves


def find_unique_shape(
    leaves: list[tuple[str, torch.Tensor]],
    shape: tuple[int, ...],
    role: str,
) -> torch.Tensor:
    matches = [
        tensor
        for _, tensor in leaves
        if tuple(tensor.shape) == shape
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{role}: expected exactly one tensor with shape={shape}; "
            f"found={len(matches)}"
        )
    return matches[0]


def find_token_tensor(
    leaves: list[tuple[str, torch.Tensor]],
    tokens: Iterable[str],
    *,
    shape: tuple[int, ...] | None = None,
    scalar: bool = False,
    exclude: Iterable[str] = (),
    role: str,
) -> torch.Tensor:
    tokens = tuple(token.lower() for token in tokens)
    exclude = tuple(token.lower() for token in exclude)
    matches = []
    for path, tensor in leaves:
        lowered = path.lower()
        if not any(token in lowered for token in tokens):
            continue
        if any(token in lowered for token in exclude):
            continue
        if shape is not None and tuple(tensor.shape) != shape:
            continue
        if scalar and tensor.numel() != 1:
            continue
        matches.append((path, tensor))
    if len(matches) != 1:
        inventory = [
            (path, tuple(tensor.shape), str(tensor.dtype))
            for path, tensor in leaves
        ]
        raise RuntimeError(
            f"{role}: expected one matching tensor; found={matches}; "
            f"inventory={inventory}"
        )
    return matches[0][1]


def find_exact_suffix_tensor(
    leaves: list[tuple[str, torch.Tensor]],
    suffix: str,
    *,
    shape: tuple[int, ...] | None = None,
    scalar: bool = False,
    role: str,
) -> torch.Tensor:
    """Resolve a certified loader field by its exact terminal key.

    This prevents metadata tensors such as ``source_target_router`` from
    being mistaken for the supervised target ``y_source``.
    """
    normalized_suffix = suffix.lower()
    matches = []
    for path, tensor in leaves:
        lowered = path.lower()
        if not (
            lowered == normalized_suffix
            or lowered.endswith("." + normalized_suffix)
        ):
            continue
        if shape is not None and tuple(tensor.shape) != shape:
            continue
        if scalar and tensor.numel() != 1:
            continue
        matches.append((path, tensor))

    if len(matches) != 1:
        inventory = [
            (path, tuple(tensor.shape), str(tensor.dtype))
            for path, tensor in leaves
        ]
        raise RuntimeError(
            f"{role}: expected exact field {suffix!r}; "
            f"found={matches}; inventory={inventory}"
        )
    return matches[0][1]


def find_string_value(
    sample: Any,
    tokens: Iterable[str],
    fallback: str,
) -> str:
    tokens = tuple(token.lower() for token in tokens)
    candidates = []
    for path, value in walk(sample):
        if not any(token in path.lower() for token in tokens):
            continue
        if isinstance(value, (str, int, np.integer)):
            candidates.append(str(value))
    return candidates[0] if candidates else fallback


def canonicalize_sample(
    sample: Any,
    dataset_name: str,
    dataset_index: int,
) -> dict[str, Any]:
    leaves = tensor_leaves(sample)
    x = find_unique_shape(leaves, EXPECTED_X_SHAPE, "x")
    mask = find_unique_shape(leaves, EXPECTED_MASK_SHAPE, "mask")
    edge = find_unique_shape(leaves, EXPECTED_EDGE_SHAPE, "edge_index")

    # Resolve the exact certified supervised-target fields.
    # Broad token matching is unsafe because the sample also includes
    # metadata such as source_target_router and role_mask.
    graph = find_exact_suffix_tensor(
        leaves,
        "y_attack",
        scalar=True,
        role="graph attack",
    )
    count = find_exact_suffix_tensor(
        leaves,
        "y_attacker_count",
        scalar=True,
        role="attacker count",
    )
    source = find_exact_suffix_tensor(
        leaves,
        "y_source",
        shape=(16,),
        role="source",
    )
    transit = find_exact_suffix_tensor(
        leaves,
        "y_transit",
        shape=(16,),
        role="transit",
    )
    victim = find_exact_suffix_tensor(
        leaves,
        "y_victim",
        shape=(16,),
        role="victim",
    )
    path = find_exact_suffix_tensor(
        leaves,
        "y_attack_path",
        shape=(16,),
        role="path",
    )

    graph_value = int(graph.reshape(-1)[0].item())
    count_value = int(count.reshape(-1)[0].item())
    if graph_value not in (0, 1):
        raise RuntimeError(f"invalid graph label={graph_value}")
    if count_value not in (0, 1, 2, 3, 4):
        raise RuntimeError(f"invalid count label={count_value}")
    if graph_value == 1 and count_value not in (1, 2, 3, 4):
        raise RuntimeError(
            f"active sample has invalid conditional count={count_value}"
        )

    pair_id = find_string_value(
        sample,
        ("pair_id", "pair_key", "pair"),
        fallback=f"{dataset_name}:{dataset_index}",
    )
    run_id = find_string_value(
        sample,
        ("run_id", "run_key", "run_name"),
        fallback=f"{dataset_name}:{dataset_index}",
    )

    return {
        "x": x.float().contiguous(),
        "mask": mask.float().contiguous(),
        "edge": edge.long().contiguous(),
        "graph": torch.tensor(graph_value, dtype=torch.float32),
        "count": torch.tensor(count_value, dtype=torch.long),
        "source": source.float().contiguous(),
        "transit": transit.float().contiguous(),
        "victim": victim.float().contiguous(),
        "path": path.float().contiguous(),
        "pair_id": pair_id,
        "run_id": run_id,
        "dataset_name": dataset_name,
        "dataset_index": int(dataset_index),
    }


def candidate_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    count = min(length, count)
    values = np.linspace(0, length - 1, num=count, dtype=np.int64)
    return sorted(set(int(value) for value in values.tolist()))


def select_active_examples(
    dataset,
    initial_candidates: int = 1024,
) -> list[dict[str, Any]]:
    selected: dict[int, list[dict[str, Any]]] = {
        1: [], 2: [], 3: [], 4: []
    }
    pair_counts: Counter[str] = Counter()
    inspected = set()

    for candidate_count in (initial_candidates, 2048, 4096):
        for index in candidate_indices(len(dataset), candidate_count):
            if index in inspected:
                continue
            inspected.add(index)
            item = canonicalize_sample(
                dataset[index],
                "train_active_only",
                index,
            )
            count = int(item["count"].item())
            if int(item["graph"].item()) != 1:
                continue
            if count not in selected:
                continue
            if len(selected[count]) >= ACTIVE_PER_COUNT:
                continue
            if pair_counts[item["pair_id"]] >= MAX_PER_PAIR_PER_GROUP:
                continue
            selected[count].append(item)
            pair_counts[item["pair_id"]] += 1
            if all(
                len(selected[value]) >= ACTIVE_PER_COUNT
                for value in (1, 2, 3, 4)
            ):
                result = []
                for value in (1, 2, 3, 4):
                    result.extend(selected[value])
                return result

    counts = {key: len(value) for key, value in selected.items()}
    raise RuntimeError(
        "unable to select active count-balanced subset; "
        f"selected={counts}; inspected={len(inspected)}"
    )


def select_negative_examples(
    dataset,
    selected_active: list[dict[str, Any]],
    initial_candidates: int = 1024,
) -> list[dict[str, Any]]:
    selected = []
    pair_counts: Counter[str] = Counter()
    inspected = set()
    active_pairs = {item["pair_id"] for item in selected_active}

    for candidate_count in (initial_candidates, 2048, 4096):
        for index in candidate_indices(len(dataset), candidate_count):
            if index in inspected:
                continue
            inspected.add(index)
            item = canonicalize_sample(
                dataset[index],
                "train_all",
                index,
            )
            if int(item["graph"].item()) != 0:
                continue
            if pair_counts[item["pair_id"]] >= MAX_PER_PAIR_PER_GROUP:
                continue

            # Prefer matched pair IDs when available, but do not require
            # them because control and attack readiness may differ.
            preference = item["pair_id"] in active_pairs
            item["matched_to_selected_active_pair"] = preference
            selected.append(item)
            pair_counts[item["pair_id"]] += 1
            if len(selected) >= NEGATIVE_COUNT:
                selected.sort(
                    key=lambda row: (
                        not row["matched_to_selected_active_pair"],
                        row["pair_id"],
                        row["dataset_index"],
                    )
                )
                return selected[:NEGATIVE_COUNT]

    raise RuntimeError(
        "unable to select deterministic inactive subset; "
        f"selected={len(selected)}; inspected={len(inspected)}"
    )


def stack_subset(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != SUBSET_SIZE:
        raise RuntimeError(
            f"subset size={len(items)}, expected={SUBSET_SIZE}"
        )
    base_edge = items[0]["edge"]
    for item in items[1:]:
        if not torch.equal(base_edge, item["edge"]):
            raise RuntimeError("edge_index differs inside tiny subset")
    return {
        "x": torch.stack([item["x"] for item in items]),
        "mask": torch.stack([item["mask"] for item in items]),
        "edge": base_edge,
        "graph": torch.stack([item["graph"] for item in items]),
        "count": torch.stack([item["count"] for item in items]),
        "source": torch.stack([item["source"] for item in items]),
        "transit": torch.stack([item["transit"] for item in items]),
        "victim": torch.stack([item["victim"] for item in items]),
        "path": torch.stack([item["path"] for item in items]),
        "metadata": [
            {
                key: item[key]
                for key in (
                    "dataset_name",
                    "dataset_index",
                    "pair_id",
                    "run_id",
                )
            }
            for item in items
        ],
    }


def pos_weight(target: torch.Tensor, maximum: float = 20.0) -> float:
    positive = float(target.sum().item())
    total = float(target.numel())
    negative = total - positive
    if positive <= 0:
        raise RuntimeError("target has zero positive entries")
    return min(maximum, max(1.0, negative / positive))


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    active = batch["graph"] > 0.5
    if not bool(active.any()):
        raise RuntimeError("tiny subset has no active samples")

    losses = {
        "graph": F.binary_cross_entropy_with_logits(
            outputs["attack_logits"],
            batch["graph"],
            pos_weight=weights["graph"],
        ),
        "count": F.cross_entropy(
            outputs["count_logits"][active],
            batch["count"][active] - 1,
        ),
        "source": F.binary_cross_entropy_with_logits(
            outputs["source_logits"],
            batch["source"],
            pos_weight=weights["source"],
        ),
        "transit": F.binary_cross_entropy_with_logits(
            outputs["transit_logits"],
            batch["transit"],
            pos_weight=weights["transit"],
        ),
        "victim": F.binary_cross_entropy_with_logits(
            outputs["victim_logits"],
            batch["victim"],
            pos_weight=weights["victim"],
        ),
        "path": F.binary_cross_entropy_with_logits(
            outputs["path_logits"],
            batch["path"],
            pos_weight=weights["path"],
        ),
    }
    losses["total"] = sum(losses.values())
    return losses


def binary_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = (torch.sigmoid(logits) >= 0.5).to(torch.int64)
    truth = target.to(torch.int64)
    tp = int(((prediction == 1) & (truth == 1)).sum().item())
    fp = int(((prediction == 1) & (truth == 0)).sum().item())
    fn = int(((prediction == 0) & (truth == 1)).sum().item())
    tn = int(((prediction == 0) & (truth == 0)).sum().item())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    if target.ndim == 2:
        exact = float(
            (prediction == truth).all(dim=1).float().mean().item()
        )
    else:
        exact = accuracy
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": exact,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    active = batch["graph"] > 0.5
    count_prediction = outputs["count_logits"][active].argmax(dim=1) + 1
    count_truth = batch["count"][active]
    count_accuracy = float(
        (count_prediction == count_truth).float().mean().item()
    )
    return {
        "graph": binary_metrics(
            outputs["attack_logits"],
            batch["graph"],
        ),
        "count_active_accuracy": count_accuracy,
        "source": binary_metrics(
            outputs["source_logits"],
            batch["source"],
        ),
        "transit": binary_metrics(
            outputs["transit_logits"],
            batch["transit"],
        ),
        "victim": binary_metrics(
            outputs["victim_logits"],
            batch["victim"],
        ),
        "path": binary_metrics(
            outputs["path_logits"],
            batch["path"],
        ),
    }


def gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    groups = {
        "input_projection": ("input_projection.",),
        "temporal_blocks": ("temporal_blocks.",),
        "node_projection": ("node_projection.",),
        "graph1": ("graph1.",),
        "graph2": ("graph2.",),
        "graph_projection": ("graph_projection.",),
        "attack_head": ("attack_head.",),
        "count_head": ("count_head.",),
        "source_head": ("source_head.",),
        "transit_head": ("transit_head.",),
        "victim_head": ("victim_head.",),
        "path_head": ("path_head.",),
    }
    results = {}
    named = list(model.named_parameters())
    for group, prefixes in groups.items():
        square_sum = 0.0
        found = False
        for name, parameter in named:
            if not name.startswith(prefixes):
                continue
            found = True
            if parameter.grad is not None:
                square_sum += float(
                    parameter.grad.detach().float().pow(2).sum().item()
                )
        if not found:
            raise RuntimeError(f"gradient group missing: {group}")
        results[group] = math.sqrt(square_sum)
    return results


def gate_pass(
    initial_losses: dict[str, float],
    final_losses: dict[str, float],
    metrics: dict[str, Any],
) -> tuple[bool, dict[str, bool]]:
    checks = {
        "total_loss_ratio_le_0p25": (
            final_losses["total"]
            <= 0.25 * initial_losses["total"]
        ),
        "every_component_loss_decreased": all(
            final_losses[name] < initial_losses[name]
            for name in (
                "graph",
                "count",
                "source",
                "transit",
                "victim",
                "path",
            )
        ),
        "graph_accuracy_ge_0p95": (
            metrics["graph"]["accuracy"] >= 0.95
        ),
        "count_accuracy_ge_0p90": (
            metrics["count_active_accuracy"] >= 0.90
        ),
        "source_f1_ge_0p85": metrics["source"]["f1"] >= 0.85,
        "transit_f1_ge_0p85": metrics["transit"]["f1"] >= 0.85,
        "victim_f1_ge_0p85": metrics["victim"]["f1"] >= 0.85,
        "path_f1_ge_0p85": metrics["path"]["f1"] >= 0.85,
    }
    return all(checks.values()), checks


def to_float_losses(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        name: float(value.detach().item())
        for name, value in losses.items()
    }


def move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def main() -> int:
    args = parse_args()
    set_seed(SEED)

    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    data_root = data_link.resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    checkpoint_path = output_dir / f"{STAGE}_BEST.pt"
    epoch_csv_path = output_dir / f"{STAGE}_EPOCHS.csv"
    subset_json_path = output_dir / f"{STAGE}_SUBSET.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"
    hold_path = output_dir / f"{STAGE}_HOLD"

    a1_dir = (
        repo
        / "reports/v5/p3_a1_guarded_loader_schema_certification"
    )
    a1_report_path = (
        a1_dir
        / "V5_P3_A1_GUARDED_LOADER_SCHEMA_CERTIFICATION_REPORT.json"
    )
    a1_lock_path = (
        a1_dir
        / "V5_P3_A1_GUARDED_LOADER_SCHEMA_CERTIFICATION_LOCK.json"
    )
    wrapper_path = (
        repo / "src/data/v5_p3_tranche_a_guarded_loader.py"
    )
    b3_path = repo / "src/models/v5_p2_b3_conv1d_only_count4.py"
    canonical_path = (
        repo / "src/models/v5_p2_task_d_full_multitask_count4.py"
    )
    dynamic_path = (
        repo
        / "src/models/v6_p0_dynamic70_task_d_full_multitask_count4.py"
    )
    source_freeze_report_path = (
        repo
        / "reports/v6/p0_m8a_r1_dynamic70_model_source_freeze/"
        "V6_P0_M8A_R1_DYNAMIC70_MODEL_SOURCE_FREEZE_REPORT.json"
    )

    required = [
        a1_report_path,
        a1_lock_path,
        wrapper_path,
        b3_path,
        canonical_path,
        dynamic_path,
        source_freeze_report_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required files missing: {missing}")
    if not data_link.is_symlink():
        raise RuntimeError(
            f"canonical data path is not a symlink: {data_link}"
        )

    a1_report = json.loads(
        a1_report_path.read_text(encoding="utf-8")
    )
    a1_lock = json.loads(
        a1_lock_path.read_text(encoding="utf-8")
    )
    if a1_report.get("status") != "PASS":
        raise RuntimeError("A1 report is not PASS")
    if not a1_report.get("decision", {}).get(
        "tiny_overfit_authorized"
    ):
        raise RuntimeError("A1 did not authorize tiny overfit")
    if a1_report.get("sealed_test", {}).get(
        "test_dataset_instantiated"
    ):
        raise RuntimeError("A1 reports test dataset instantiation")
    if a1_lock.get("guarded_wrapper_sha256") != sha256_file(
        wrapper_path
    ):
        raise RuntimeError("A1 guarded-wrapper SHA mismatch")

    source_freeze = json.loads(
        source_freeze_report_path.read_text(encoding="utf-8")
    )
    expected_dynamic_sha = source_freeze.get(
        "source_generation", {}
    ).get("versioned_source_sha256")
    if expected_dynamic_sha != sha256_file(dynamic_path):
        raise RuntimeError(
            "Dynamic70 source differs from frozen M8A-R1 source"
        )

    # Process-wide deserialization guard. Any attempt to torch.load a
    # file under runs/test is rejected and recorded.
    original_torch_load = torch.load
    loaded_paths: list[str] = []

    def guarded_torch_load(file, *load_args, **load_kwargs):
        try:
            candidate = Path(os.fspath(file)).expanduser().resolve()
        except TypeError:
            candidate = None
        if candidate is not None:
            candidate_text = str(candidate)
            loaded_paths.append(candidate_text)
            parts = candidate.parts
            if (
                "runs" in parts
                and "test" in parts
                and parts.index("runs") < parts.index("test")
            ):
                raise PermissionError(
                    "A2 sealed-test deserialization guard blocked "
                    f"{candidate}"
                )
        return original_torch_load(file, *load_args, **load_kwargs)

    torch.load = guarded_torch_load

    wrapper_mod = import_module(
        wrapper_path,
        "_v5_p3_a2_guarded_loader",
    )
    b3_mod = import_module(b3_path, "_v5_p3_a2_b3")
    canonical_mod = import_module(
        canonical_path,
        "_v5_p3_a2_canonical",
    )
    dynamic_mod = import_module(
        dynamic_path,
        "_v5_p3_a2_dynamic70",
    )

    GuardedDataset = (
        wrapper_mod.GuardedV5P3TrancheAPreliminaryDataset
    )
    SealedTestAccessError = wrapper_mod.SealedTestAccessError

    try:
        GuardedDataset(data_root, "test")
    except SealedTestAccessError:
        test_negative_check = True
    else:
        test_negative_check = False
    if not test_negative_check:
        raise RuntimeError("A1 wrapper failed to reject A_test")

    train_all = GuardedDataset(
        data_root,
        "train",
        active_only=False,
    )
    train_active = GuardedDataset(
        data_root,
        "train",
        active_only=True,
    )

    active_items = select_active_examples(train_active)
    negative_items = select_negative_examples(
        train_all,
        active_items,
    )
    items = active_items + negative_items
    batch_cpu = stack_subset(items)

    count_distribution = Counter(
        int(item["count"].item()) for item in active_items
    )
    if count_distribution != Counter(
        {1: ACTIVE_PER_COUNT, 2: ACTIVE_PER_COUNT,
         3: ACTIVE_PER_COUNT, 4: ACTIVE_PER_COUNT}
    ):
        raise RuntimeError(
            f"active count distribution changed: {count_distribution}"
        )
    if int(batch_cpu["graph"].sum().item()) != len(active_items):
        raise RuntimeError("graph-label distribution mismatch")

    pair_ids = [item["pair_id"] for item in items]
    unique_pair_count = len(set(pair_ids))
    if unique_pair_count < 8:
        raise RuntimeError(
            f"tiny subset has only {unique_pair_count} unique pairs"
        )

    role_positive_counts = {
        role: int(batch_cpu[role].sum().item())
        for role in ("source", "transit", "victim", "path")
    }
    if any(value <= 0 for value in role_positive_counts.values()):
        raise RuntimeError(
            f"role-positive coverage failure: {role_positive_counts}"
        )

    atomic_json(
        subset_json_path,
        {
            "stage": STAGE,
            "campaign_label": CAMPAIGN_LABEL,
            "seed": SEED,
            "subset_size": len(items),
            "active_items": len(active_items),
            "inactive_items": len(negative_items),
            "active_count_distribution": dict(
                sorted(count_distribution.items())
            ),
            "unique_pair_count": unique_pair_count,
            "role_positive_counts": role_positive_counts,
            "items": [
                {
                    **batch_cpu["metadata"][index],
                    "graph_label": int(items[index]["graph"].item()),
                    "count_label": int(items[index]["count"].item()),
                }
                for index in range(len(items))
            ],
            "validation_accessed": False,
            "test_accessed": False,
        },
    )

    edge = batch_cpu["edge"]
    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    canonical_model = canonical_mod.P2TaskDGraphConvCount4(
        reference_b3,
        edge,
    )
    model = (
        dynamic_mod
        .build_v6_p0_dynamic70_from_canonical_structure(
            canonical_model,
            seed=SEED,
        )
    )

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Dynamic70 parameter_count={parameter_count}, "
            f"expected={EXPECTED_PARAMETER_COUNT}"
        )
    if model.input_projection.in_channels != 70:
        raise RuntimeError(
            "Dynamic70 input projection does not accept 70 features"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    batch = move_batch(batch_cpu, device)

    weights_float = {
        "graph": pos_weight(batch_cpu["graph"]),
        "source": pos_weight(batch_cpu["source"]),
        "transit": pos_weight(batch_cpu["transit"]),
        "victim": pos_weight(batch_cpu["victim"]),
        "path": pos_weight(batch_cpu["path"]),
    }
    weights = {
        key: torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in weights_float.items()
    }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )

    model.train()
    outputs = model(batch["x"], batch["mask"])
    actual_output_shapes = {
        key: tuple(value.shape)
        for key, value in outputs.items()
    }
    if actual_output_shapes != EXPECTED_OUTPUT_SHAPES:
        raise RuntimeError(
            f"output shapes={actual_output_shapes}, "
            f"expected={EXPECTED_OUTPUT_SHAPES}"
        )

    initial_loss_tensors = compute_losses(outputs, batch, weights)
    initial_losses = to_float_losses(initial_loss_tensors)

    optimizer.zero_grad(set_to_none=True)
    initial_loss_tensors["total"].backward()
    first_gradient_norms = gradient_norms(model)
    zero_gradient_groups = [
        key
        for key, value in first_gradient_norms.items()
        if not math.isfinite(value) or value <= 0.0
    ]
    if zero_gradient_groups:
        raise RuntimeError(
            "non-positive first-backward gradient groups: "
            f"{zero_gradient_groups}; norms={first_gradient_norms}"
        )
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRADIENT_CLIP,
    )
    optimizer.step()

    # Track the numerically lowest-loss state for diagnostics, but keep a
    # separate gate-passing state for certification. A lower total loss does
    # not necessarily satisfy every per-head memorization threshold.
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_optimizer_state = None
    success_epoch = None
    success_state = None
    success_optimizer_state = None
    success_losses = None
    success_metrics = None
    history: list[dict[str, Any]] = []
    final_gate_checks = {}

    start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["x"], batch["mask"])
        losses = compute_losses(outputs, batch, weights)
        total_loss = losses["total"]
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"non-finite loss at epoch={epoch}"
            )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
        )
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_outputs = model(batch["x"], batch["mask"])
            eval_losses_t = compute_losses(
                eval_outputs,
                batch,
                weights,
            )
            eval_losses = to_float_losses(eval_losses_t)
            metrics = evaluate_metrics(eval_outputs, batch)

        if eval_losses["total"] < best_loss:
            best_loss = eval_losses["total"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(
                optimizer.state_dict()
            )

        passed, checks = gate_pass(
            initial_losses,
            eval_losses,
            metrics,
        )
        row = {
            "epoch": epoch,
            "total_loss": eval_losses["total"],
            "graph_loss": eval_losses["graph"],
            "count_loss": eval_losses["count"],
            "source_loss": eval_losses["source"],
            "transit_loss": eval_losses["transit"],
            "victim_loss": eval_losses["victim"],
            "path_loss": eval_losses["path"],
            "graph_accuracy": metrics["graph"]["accuracy"],
            "graph_f1": metrics["graph"]["f1"],
            "count_active_accuracy": (
                metrics["count_active_accuracy"]
            ),
            "source_f1": metrics["source"]["f1"],
            "transit_f1": metrics["transit"]["f1"],
            "victim_f1": metrics["victim"]["f1"],
            "path_f1": metrics["path"]["f1"],
            "gate_pass": passed,
        }
        history.append(row)

        if epoch == 1 or epoch % 25 == 0 or passed:
            print(
                f"epoch={epoch:03d} "
                f"loss={eval_losses['total']:.6f} "
                f"g_acc={metrics['graph']['accuracy']:.4f} "
                f"count_acc={metrics['count_active_accuracy']:.4f} "
                f"src_f1={metrics['source']['f1']:.4f} "
                f"tr_f1={metrics['transit']['f1']:.4f} "
                f"vic_f1={metrics['victim']['f1']:.4f} "
                f"path_f1={metrics['path']['f1']:.4f} "
                f"gate={passed}"
            )

        if passed and epoch >= 20:
            success_epoch = epoch
            success_state = copy.deepcopy(model.state_dict())
            success_optimizer_state = copy.deepcopy(
                optimizer.state_dict()
            )
            success_losses = copy.deepcopy(eval_losses)
            success_metrics = copy.deepcopy(metrics)
            final_gate_checks = copy.deepcopy(checks)
            break

    write_csv(epoch_csv_path, history)

    if best_state is None or best_optimizer_state is None:
        raise RuntimeError("no lowest-loss state captured")

    if (
        success_state is not None
        and success_optimizer_state is not None
        and success_epoch is not None
    ):
        selected_state = success_state
        selected_optimizer_state = success_optimizer_state
        selected_epoch = success_epoch
        selected_reason = "FIRST_GATE_PASSING_EPOCH"
    else:
        selected_state = best_state
        selected_optimizer_state = best_optimizer_state
        selected_epoch = best_epoch
        selected_reason = "LOWEST_LOSS_FALLBACK_NO_GATE_PASS"

    model.load_state_dict(selected_state)
    model.eval()
    with torch.no_grad():
        selected_outputs = model(batch["x"], batch["mask"])
        selected_losses_t = compute_losses(
            selected_outputs,
            batch,
            weights,
        )
        selected_losses = to_float_losses(selected_losses_t)
        selected_metrics = evaluate_metrics(
            selected_outputs,
            batch,
        )
    passed, final_gate_checks = gate_pass(
        initial_losses,
        selected_losses,
        selected_metrics,
    )

    # Internal consistency: any state captured at a gate-passing epoch must
    # still pass when restored and reevaluated deterministically.
    if success_epoch is not None and not passed:
        raise RuntimeError(
            "captured gate-passing state failed after deterministic reload"
        )

    checkpoint_payload = {
        "stage": STAGE,
        "campaign_label": CAMPAIGN_LABEL,
        "seed": SEED,
        "epoch": selected_epoch,
        "selection_reason": selected_reason,
        "model_state_dict": selected_state,
        "optimizer_state_dict": selected_optimizer_state,
        "parameter_count": parameter_count,
        "input_features": 70,
        "subset_manifest_sha256": sha256_file(subset_json_path),
        "a1_report_sha256": sha256_file(a1_report_path),
        "dynamic70_source_sha256": sha256_file(dynamic_path),
        "test_tensor_loaded": False,
        "validation_accessed": False,
    }
    torch.save(checkpoint_payload, checkpoint_path)

    # Exact checkpoint reload into a fresh 60,553-parameter model.
    fresh_reference = b3_mod.P2B3Conv1DOnlyCount4()
    fresh_canonical = canonical_mod.P2TaskDGraphConvCount4(
        fresh_reference,
        edge,
    )
    reloaded_model = (
        dynamic_mod
        .build_v6_p0_dynamic70_from_canonical_structure(
            fresh_canonical,
            seed=SEED,
        )
        .to(device)
    )
    reloaded_optimizer = torch.optim.AdamW(
        reloaded_model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    reloaded_model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    reloaded_optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )
    reloaded_model.eval()
    with torch.no_grad():
        reload_outputs = reloaded_model(
            batch["x"],
            batch["mask"],
        )
    reload_max_abs_diff = max(
        float(
            (
                reload_outputs[key] - selected_outputs[key]
            )
            .abs()
            .max()
            .item()
        )
        for key in selected_outputs
    )
    checkpoint_reload_exact = reload_max_abs_diff <= 1e-7

    # Prove optimizer resume changes trainable state and stays finite.
    before_resume_digest = state_digest(
        reloaded_model.state_dict()
    )
    reloaded_model.train()
    reloaded_optimizer.zero_grad(set_to_none=True)
    resume_outputs = reloaded_model(
        batch["x"],
        batch["mask"],
    )
    resume_losses = compute_losses(
        resume_outputs,
        batch,
        weights,
    )
    resume_losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(
        reloaded_model.parameters(),
        GRADIENT_CLIP,
    )
    reloaded_optimizer.step()
    after_resume_digest = state_digest(
        reloaded_model.state_dict()
    )
    optimizer_resume_functional = (
        before_resume_digest != after_resume_digest
        and math.isfinite(
            float(resume_losses["total"].detach().item())
        )
    )

    test_loaded_paths = [
        path for path in loaded_paths
        if "/runs/test/" in path
    ]
    if test_loaded_paths:
        raise RuntimeError(
            f"sealed-test paths were deserialized: {test_loaded_paths}"
        )

    all_integration_checks = {
        "tiny_overfit_gate": passed,
        "first_backward_all_groups_nonzero": (
            not zero_gradient_groups
        ),
        "checkpoint_reload_exact": checkpoint_reload_exact,
        "optimizer_resume_functional": (
            optimizer_resume_functional
        ),
        "parameter_count_exact": (
            parameter_count == EXPECTED_PARAMETER_COUNT
        ),
        "input_width_exact": (
            model.input_projection.in_channels == 70
        ),
        "output_shapes_exact": (
            actual_output_shapes == EXPECTED_OUTPUT_SHAPES
        ),
        "sealed_test_negative_check": test_negative_check,
        "test_tensor_loaded_false": not test_loaded_paths,
        "validation_accessed_false": True,
    }
    overall_pass = all(all_integration_checks.values())

    report = {
        "stage": STAGE,
        "status": "PASS" if overall_pass else "HOLD",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "audit_script_revision": "v3_gate_passing_checkpoint_selection",
        "interpretation": (
            "Train-only fixed-subset memorization and integration "
            "sanity test. It is not a generalization result."
        ),
        "repository": str(repo),
        "canonical_data_link": str(data_link.absolute()),
        "resolved_dataset_root": str(data_root),
        "device": str(device),
        "seed": SEED,
        "model": {
            "class": model.__class__.__name__,
            "builder": (
                "build_v6_p0_dynamic70_from_"
                "canonical_structure"
            ),
            "input_shape": [16, 70, 32],
            "mask_shape": [16, 10],
            "parameter_count": parameter_count,
            "expected_parameter_count": (
                EXPECTED_PARAMETER_COUNT
            ),
            "output_shapes": {
                key: list(value)
                for key, value in actual_output_shapes.items()
            },
            "fresh_initialization": True,
            "historical_checkpoint_used_for_training": False,
        },
        "data": {
            "split": "train",
            "train_all_length": len(train_all),
            "train_active_only_length": len(train_active),
            "subset_size": SUBSET_SIZE,
            "active_examples": len(active_items),
            "inactive_examples": len(negative_items),
            "active_count_distribution": dict(
                sorted(count_distribution.items())
            ),
            "unique_pair_count": unique_pair_count,
            "role_positive_counts": role_positive_counts,
            "pair_capped_selection": True,
            "maximum_items_per_pair_per_group": (
                MAX_PER_PAIR_PER_GROUP
            ),
            "subset_manifest": str(subset_json_path),
            "validation_accessed": False,
            "test_accessed": False,
        },
        "loss": {
            "components": [
                "graph",
                "count_active_only",
                "source",
                "transit",
                "victim",
                "path",
            ],
            "component_coefficients": {
                "graph": 1.0,
                "count": 1.0,
                "source": 1.0,
                "transit": 1.0,
                "victim": 1.0,
                "path": 1.0,
            },
            "positive_weights": weights_float,
            "count_mapping": {
                "inactive": "masked",
                "active_targets": "y_count_minus_1",
                "logits": 4,
                "class_values": [1, 2, 3, 4],
            },
            "initial": initial_losses,
            "selected_gate_state": selected_losses,
            "lowest_total_loss_value": best_loss,
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "max_epochs": MAX_EPOCHS,
            "epochs_run": len(history),
            "lowest_loss_epoch": best_epoch,
            "success_epoch": success_epoch,
            "selected_checkpoint_epoch": selected_epoch,
            "selected_checkpoint_reason": selected_reason,
            "elapsed_seconds": time.time() - start,
        },
        "first_backward_gradient_norms": (
            first_gradient_norms
        ),
        "selected_metrics": selected_metrics,
        "tiny_overfit_gate_checks": final_gate_checks,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "selected_epoch": selected_epoch,
            "selection_reason": selected_reason,
            "reload_max_abs_diff": reload_max_abs_diff,
            "reload_exact": checkpoint_reload_exact,
            "optimizer_resume_functional": (
                optimizer_resume_functional
            ),
        },
        "sealed_test": {
            "guard_negative_check": test_negative_check,
            "test_dataset_instantiated": False,
            "test_length_computed": False,
            "test_tensor_loaded": False,
            "test_loaded_paths": test_loaded_paths,
            "sealed_test_evaluation_authorized": False,
        },
        "integration_checks": all_integration_checks,
        "decision": {
            "tiny_overfit_integration_passed": overall_pass,
            "preliminary_baseline_preflight_authorized": (
                overall_pass
            ),
            "preliminary_baseline_training_authorized": False,
            "sealed_test_evaluation_authorized": False,
            "next_stage": (
                "V5_P3_A3_PRELIMINARY_BASELINE_PREFLIGHT"
                if overall_pass
                else "A2_HOLD_REVIEW"
            ),
        },
        "provenance": {
            "a1_report": str(a1_report_path),
            "a1_report_sha256": sha256_file(a1_report_path),
            "a1_lock_sha256": sha256_file(a1_lock_path),
            "guarded_wrapper_sha256": sha256_file(
                wrapper_path
            ),
            "b3_source_sha256": sha256_file(b3_path),
            "canonical_model_source_sha256": sha256_file(
                canonical_path
            ),
            "dynamic70_source_sha256": sha256_file(
                dynamic_path
            ),
            "source_freeze_report_sha256": sha256_file(
                source_freeze_report_path
            ),
            "installed_script_sha256": sha256_file(
                installed_script
            ),
        },
        "source_modified": True,
        "certified_dataset_modified": False,
        "model_trained": True,
        "generalization_claim_authorized": False,
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": report["status"],
        "campaign_label": CAMPAIGN_LABEL,
        "report_sha256": sha256_file(report_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "epoch_csv_sha256": sha256_file(epoch_csv_path),
        "subset_manifest_sha256": sha256_file(
            subset_json_path
        ),
        "installed_script_sha256": sha256_file(
            installed_script
        ),
        "dynamic70_source_sha256": sha256_file(
            dynamic_path
        ),
        "guarded_wrapper_sha256": sha256_file(
            wrapper_path
        ),
        "parameter_count": parameter_count,
        "input_features": 70,
        "selected_checkpoint_epoch": selected_epoch,
        "selected_checkpoint_reason": selected_reason,
        "validation_accessed": False,
        "test_tensor_loaded": False,
        "certified_dataset_modified": False,
    }
    atomic_json(lock_path, lock)

    if overall_pass:
        complete_path.write_text(
            f"{STAGE}_COMPLETE\n",
            encoding="utf-8",
        )
        if hold_path.exists():
            hold_path.unlink()
    else:
        hold_path.write_text(
            f"{STAGE}_HOLD\n",
            encoding="utf-8",
        )

    print(f"{STAGE}_{'COMPLETE' if overall_pass else 'HOLD'}")
    print(f"status={report['status']}")
    print("campaign_label=" + CAMPAIGN_LABEL)
    print("audit_script_revision=v3_gate_passing_checkpoint_selection")
    print("data_split=train_only")
    print(f"train_all_length={len(train_all)}")
    print(f"train_active_only_length={len(train_active)}")
    print(f"subset_size={SUBSET_SIZE}")
    print(f"unique_pair_count={unique_pair_count}")
    print(
        "active_count_distribution="
        + json.dumps(dict(sorted(count_distribution.items())))
    )
    print(f"model_class={model.__class__.__name__}")
    print(f"input_features={model.input_projection.in_channels}")
    print(f"parameter_count={parameter_count}")
    print(
        "initial_total_loss="
        f"{initial_losses['total']:.8f}"
    )
    print(f"lowest_total_loss={best_loss:.8f}")
    print(f"lowest_loss_epoch={best_epoch}")
    print(
        "selected_checkpoint_total_loss="
        f"{selected_losses['total']:.8f}"
    )
    print(f"selected_checkpoint_epoch={selected_epoch}")
    print(f"selected_checkpoint_reason={selected_reason}")
    print(f"success_epoch={success_epoch}")
    print(
        "first_backward_all_groups_nonzero="
        f"{not zero_gradient_groups}"
    )
    print(
        "checkpoint_reload_exact="
        f"{checkpoint_reload_exact}"
    )
    print(
        "optimizer_resume_functional="
        f"{optimizer_resume_functional}"
    )
    print("validation_accessed=false")
    print("test_dataset_instantiated=false")
    print("test_length_computed=false")
    print("test_tensor_loaded=false")
    print(
        "preliminary_baseline_preflight_authorized="
        f"{overall_pass}"
    )
    print("preliminary_baseline_training_authorized=false")
    print("sealed_test_evaluation_authorized=false")
    print(
        "next_stage="
        + report["decision"]["next_stage"]
    )
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    print(f"checkpoint={checkpoint_path}")
    print(f"epoch_csv={epoch_csv_path}")
    print(f"subset_manifest={subset_json_path}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
