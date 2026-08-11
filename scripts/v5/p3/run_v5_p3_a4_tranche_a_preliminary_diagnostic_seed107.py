from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler


STAGE = "V5_P3_A4_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_SEED107"
CAMPAIGN_LABEL = "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic"
SEED = 107
EXPECTED_PARAMETER_COUNT = 60_553
EXPECTED_TRAIN_ITEMS = 110_855
EXPECTED_VALIDATION_ITEMS = 13_863
COUNT_CLASS_VALUES = (1, 2, 3, 4)


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
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class EpochShuffleSampler(Sampler[int]):
    """Deterministic epoch-indexed full permutation for exact resume."""

    def __init__(self, length: int, seed: int) -> None:
        self.length = int(length)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(
            self.length,
            generator=generator,
        )
        return iter(permutation.tolist())

    def __len__(self) -> int:
        return self.length


def move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    required = {
        "x",
        "physical_port_mask",
        "y_attack",
        "y_attacker_count",
        "y_source",
        "y_transit",
        "y_victim",
        "y_attack_path",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise RuntimeError(
            f"batch missing certified keys: {missing}; "
            f"keys={sorted(batch)}"
        )

    result = {
        "x": batch["x"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=(device.type == "cuda"),
        ),
        "mask": batch["physical_port_mask"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=(device.type == "cuda"),
        ),
        "graph": batch["y_attack"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=(device.type == "cuda"),
        ).reshape(-1),
        "count": batch["y_attacker_count"].to(
            device=device,
            dtype=torch.long,
            non_blocking=(device.type == "cuda"),
        ).reshape(-1),
        "source": batch["y_source"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=(device.type == "cuda"),
        ),
        "transit": batch["y_transit"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=(device.type == "cuda"),
        ),
        "victim": batch["y_victim"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=(device.type == "cuda"),
        ),
        "path": batch["y_attack_path"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=(device.type == "cuda"),
        ),
    }

    batch_size = result["x"].shape[0]
    expected_shapes = {
        "x": (batch_size, 16, 70, 32),
        "mask": (batch_size, 16, 10),
        "graph": (batch_size,),
        "count": (batch_size,),
        "source": (batch_size, 16),
        "transit": (batch_size, 16),
        "victim": (batch_size, 16),
        "path": (batch_size, 16),
    }
    for key, expected in expected_shapes.items():
        if tuple(result[key].shape) != expected:
            raise RuntimeError(
                f"{key} shape={tuple(result[key].shape)}, "
                f"expected={expected}"
            )
    return result


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    graph_pos_weight: torch.Tensor,
    role_pos_weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    active = batch["graph"] > 0.5
    if not bool(active.any()):
        count_loss = outputs["count_logits"].sum() * 0.0
    else:
        active_targets = batch["count"][active] - 1
        if not bool(
            ((active_targets >= 0) & (active_targets <= 3)).all()
        ):
            raise RuntimeError(
                "active Count4 targets are outside 0..3"
            )
        count_loss = F.cross_entropy(
            outputs["count_logits"][active],
            active_targets,
        )

    losses = {
        "graph": F.binary_cross_entropy_with_logits(
            outputs["attack_logits"],
            batch["graph"],
            pos_weight=graph_pos_weight,
        ),
        "count": count_loss,
        "source": F.binary_cross_entropy_with_logits(
            outputs["source_logits"],
            batch["source"],
            pos_weight=role_pos_weights["source"],
        ),
        "transit": F.binary_cross_entropy_with_logits(
            outputs["transit_logits"],
            batch["transit"],
            pos_weight=role_pos_weights["transit"],
        ),
        "victim": F.binary_cross_entropy_with_logits(
            outputs["victim_logits"],
            batch["victim"],
            pos_weight=role_pos_weights["victim"],
        ),
        "path": F.binary_cross_entropy_with_logits(
            outputs["path_logits"],
            batch["path"],
            pos_weight=role_pos_weights["path"],
        ),
    }
    losses["total"] = sum(losses.values())
    if not all(torch.isfinite(value) for value in losses.values()):
        raise RuntimeError("non-finite multitask loss")
    return losses


def binary_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = (scores >= threshold).astype(np.int64)
    tp = int(np.sum((predictions == 1) & (targets == 1)))
    fp = int(np.sum((predictions == 1) & (targets == 0)))
    fn = int(np.sum((predictions == 0) & (targets == 1)))
    tn = int(np.sum((predictions == 0) & (targets == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while (
            end < len(values)
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def binary_auroc(
    scores: np.ndarray,
    targets: np.ndarray,
) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    positive = targets == 1
    negative = targets == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = average_ranks(scores)
    rank_sum_positive = float(ranks[positive].sum())
    return (
        rank_sum_positive
        - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def binary_average_precision(
    scores: np.ndarray,
    targets: np.ndarray,
) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    positive_count = int((targets == 1).sum())
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered_targets = targets[order]
    cumulative_positive = np.cumsum(ordered_targets == 1)
    positions = np.arange(1, len(targets) + 1)
    precision_at_k = cumulative_positive / positions
    return float(
        precision_at_k[ordered_targets == 1].sum()
        / positive_count
    )


def multiclass_macro_f1(
    predictions: np.ndarray,
    targets: np.ndarray,
    classes: tuple[int, ...],
) -> float:
    values = []
    for class_value in classes:
        prediction_positive = predictions == class_value
        target_positive = targets == class_value
        tp = int(np.sum(prediction_positive & target_positive))
        fp = int(np.sum(prediction_positive & ~target_positive))
        fn = int(np.sum(~prediction_positive & target_positive))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        value = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        values.append(value)
    return float(np.mean(values))


def exact_multilabel(
    scores: np.ndarray,
    targets: np.ndarray,
    active_mask: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    predictions = (scores >= threshold).astype(np.int64)
    all_exact = float(
        np.mean(np.all(predictions == targets, axis=1))
    )
    if np.any(active_mask):
        active_exact = float(
            np.mean(
                np.all(
                    predictions[active_mask]
                    == targets[active_mask],
                    axis=1,
                )
            )
        )
    else:
        active_exact = float("nan")
    return {
        "exact_all": all_exact,
        "exact_active": active_exact,
    }


def validation_metrics(
    accum: dict[str, list[np.ndarray]],
    loss_totals: dict[str, float],
    item_count: int,
) -> dict[str, Any]:
    graph_logits = np.concatenate(accum["attack_logits"])
    graph_scores = 1.0 / (1.0 + np.exp(-graph_logits))
    graph_targets = np.concatenate(accum["graph_targets"]).astype(
        np.int64
    )
    active_mask = graph_targets == 1

    graph = binary_metrics(graph_scores, graph_targets)
    graph["auroc"] = binary_auroc(
        graph_scores,
        graph_targets,
    )
    graph["average_precision"] = binary_average_precision(
        graph_scores,
        graph_targets,
    )

    count_logits = np.concatenate(accum["count_logits"])
    count_targets = np.concatenate(accum["count_targets"]).astype(
        np.int64
    )
    active_count_logits = count_logits[active_mask]
    active_count_targets = count_targets[active_mask]
    count_predictions = (
        np.argmax(active_count_logits, axis=1) + 1
    )
    count_accuracy = float(
        np.mean(count_predictions == active_count_targets)
    )
    count_macro_f1 = multiclass_macro_f1(
        count_predictions,
        active_count_targets,
        COUNT_CLASS_VALUES,
    )

    roles = {}
    for role in ("source", "transit", "victim", "path"):
        logits = np.concatenate(accum[f"{role}_logits"])
        scores = 1.0 / (1.0 + np.exp(-logits))
        targets = np.concatenate(
            accum[f"{role}_targets"]
        ).astype(np.int64)
        flattened = binary_metrics(
            scores.reshape(-1),
            targets.reshape(-1),
        )
        flattened["auroc"] = binary_auroc(
            scores.reshape(-1),
            targets.reshape(-1),
        )
        flattened["average_precision"] = (
            binary_average_precision(
                scores.reshape(-1),
                targets.reshape(-1),
            )
        )
        flattened.update(
            exact_multilabel(
                scores,
                targets,
                active_mask,
            )
        )
        roles[role] = flattened

    components = [
        graph["auroc"],
        graph["average_precision"],
        count_macro_f1,
        roles["source"]["average_precision"],
        roles["transit"]["average_precision"],
        roles["victim"]["average_precision"],
        roles["path"]["average_precision"],
    ]
    if not all(math.isfinite(value) for value in components):
        raise RuntimeError(
            f"non-finite validation selection component: "
            f"{components}"
        )
    selection_score = float(np.mean(components))

    return {
        "loss": {
            key: value / item_count
            for key, value in loss_totals.items()
        },
        "selection_score": selection_score,
        "selection_components": {
            "graph_auroc": graph["auroc"],
            "graph_average_precision": (
                graph["average_precision"]
            ),
            "count_active_macro_f1": count_macro_f1,
            "source_average_precision": (
                roles["source"]["average_precision"]
            ),
            "transit_average_precision": (
                roles["transit"]["average_precision"]
            ),
            "victim_average_precision": (
                roles["victim"]["average_precision"]
            ),
            "path_average_precision": (
                roles["path"]["average_precision"]
            ),
        },
        "graph": graph,
        "count_active": {
            "accuracy": count_accuracy,
            "macro_f1": count_macro_f1,
            "items": int(active_mask.sum()),
        },
        "roles": roles,
        "threshold_policy": {
            "reporting_threshold": 0.5,
            "threshold_tuning_performed": False,
            "selection_metrics_are_threshold_free_except_count_macro_f1": True,
        },
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    graph_pos_weight: torch.Tensor,
    role_pos_weights: dict[str, torch.Tensor],
    gradient_clip: float,
) -> dict[str, float]:
    model.train()
    loss_totals = Counter()
    item_count = 0

    for batch in loader:
        moved = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(moved["x"], moved["mask"])
        losses = compute_losses(
            outputs,
            moved,
            graph_pos_weight,
            role_pos_weights,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip,
        )
        optimizer.step()

        batch_size = int(moved["x"].shape[0])
        item_count += batch_size
        for key, value in losses.items():
            loss_totals[key] += (
                float(value.detach().item()) * batch_size
            )

    if item_count != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(
            f"train epoch saw {item_count} items, "
            f"expected={EXPECTED_TRAIN_ITEMS}"
        )
    return {
        key: value / item_count
        for key, value in loss_totals.items()
    }


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    graph_pos_weight: torch.Tensor,
    role_pos_weights: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model.eval()
    loss_totals = Counter()
    item_count = 0
    accum = {
        "attack_logits": [],
        "graph_targets": [],
        "count_logits": [],
        "count_targets": [],
        "source_logits": [],
        "source_targets": [],
        "transit_logits": [],
        "transit_targets": [],
        "victim_logits": [],
        "victim_targets": [],
        "path_logits": [],
        "path_targets": [],
    }

    with torch.no_grad():
        for batch in loader:
            moved = move_batch(batch, device)
            outputs = model(moved["x"], moved["mask"])
            losses = compute_losses(
                outputs,
                moved,
                graph_pos_weight,
                role_pos_weights,
            )
            batch_size = int(moved["x"].shape[0])
            item_count += batch_size
            for key, value in losses.items():
                loss_totals[key] += (
                    float(value.detach().item()) * batch_size
                )

            accum["attack_logits"].append(
                outputs["attack_logits"].detach().cpu().numpy()
            )
            accum["graph_targets"].append(
                moved["graph"].detach().cpu().numpy()
            )
            accum["count_logits"].append(
                outputs["count_logits"].detach().cpu().numpy()
            )
            accum["count_targets"].append(
                moved["count"].detach().cpu().numpy()
            )
            for role in (
                "source",
                "transit",
                "victim",
                "path",
            ):
                accum[f"{role}_logits"].append(
                    outputs[f"{role}_logits"]
                    .detach()
                    .cpu()
                    .numpy()
                )
                accum[f"{role}_targets"].append(
                    moved[role].detach().cpu().numpy()
                )

    if item_count != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(
            f"validation saw {item_count} items, "
            f"expected={EXPECTED_VALIDATION_ITEMS}"
        )
    return validation_metrics(
        accum,
        dict(loss_totals),
        item_count,
    )


def flatten_history_row(
    epoch: int,
    learning_rate: float,
    train_metrics: dict[str, float],
    validation: dict[str, Any],
    elapsed_seconds: float,
    best_epoch: int | None,
    patience_counter: int,
    is_best: bool,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_total_loss": train_metrics["total"],
        "train_graph_loss": train_metrics["graph"],
        "train_count_loss": train_metrics["count"],
        "train_source_loss": train_metrics["source"],
        "train_transit_loss": train_metrics["transit"],
        "train_victim_loss": train_metrics["victim"],
        "train_path_loss": train_metrics["path"],
        "validation_total_loss": validation["loss"]["total"],
        "selection_score": validation["selection_score"],
        "graph_auroc": validation["graph"]["auroc"],
        "graph_average_precision": (
            validation["graph"]["average_precision"]
        ),
        "graph_accuracy_at_0p5": (
            validation["graph"]["accuracy"]
        ),
        "graph_f1_at_0p5": validation["graph"]["f1"],
        "graph_fpr_at_0p5": validation["graph"]["fpr"],
        "count_active_accuracy": (
            validation["count_active"]["accuracy"]
        ),
        "count_active_macro_f1": (
            validation["count_active"]["macro_f1"]
        ),
        "source_average_precision": (
            validation["roles"]["source"]["average_precision"]
        ),
        "transit_average_precision": (
            validation["roles"]["transit"]["average_precision"]
        ),
        "victim_average_precision": (
            validation["roles"]["victim"]["average_precision"]
        ),
        "path_average_precision": (
            validation["roles"]["path"]["average_precision"]
        ),
        "source_exact_active_at_0p5": (
            validation["roles"]["source"]["exact_active"]
        ),
        "transit_exact_active_at_0p5": (
            validation["roles"]["transit"]["exact_active"]
        ),
        "victim_exact_active_at_0p5": (
            validation["roles"]["victim"]["exact_active"]
        ),
        "path_exact_active_at_0p5": (
            validation["roles"]["path"]["exact_active"]
        ),
        "is_best_checkpoint": is_best,
        "best_epoch_after_epoch": best_epoch,
        "early_stop_patience_counter": patience_counter,
        "elapsed_seconds": elapsed_seconds,
    }


def main() -> int:
    args = parse_args()
    set_seed(SEED)

    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    data_root = data_link.resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(
        args.installed_script
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    history_path = output_dir / f"{STAGE}_EPOCHS.csv"
    progress_path = output_dir / f"{STAGE}_PROGRESS.json"
    best_checkpoint_path = output_dir / f"{STAGE}_BEST.pt"
    last_checkpoint_path = output_dir / f"{STAGE}_LAST.pt"
    complete_path = output_dir / f"{STAGE}_COMPLETE"
    hold_path = output_dir / f"{STAGE}_HOLD"

    a3_dir = (
        repo
        / "reports/v5/p3_a3_preliminary_baseline_preflight"
    )
    a3_report_path = (
        a3_dir
        / "V5_P3_A3_PRELIMINARY_BASELINE_PREFLIGHT_REPORT.json"
    )
    a3_lock_path = (
        a3_dir
        / "V5_P3_A3_PRELIMINARY_BASELINE_PREFLIGHT_LOCK.json"
    )
    protocol_path = (
        a3_dir
        / "V5_P3_TRANCHE_A_PRELIMINARY_BASELINE_PROTOCOL.json"
    )
    wrapper_path = (
        repo / "src/data/v5_p3_tranche_a_guarded_loader.py"
    )
    b3_path = (
        repo / "src/models/v5_p2_b3_conv1d_only_count4.py"
    )
    canonical_path = (
        repo / "src/models/v5_p2_task_d_full_multitask_count4.py"
    )
    dynamic_path = (
        repo
        / "src/models/v6_p0_dynamic70_task_d_full_multitask_count4.py"
    )
    normalization_path = (
        data_root / "normalization_tranche_a_provisional.pt"
    )

    required = [
        a3_report_path,
        a3_lock_path,
        protocol_path,
        wrapper_path,
        b3_path,
        canonical_path,
        dynamic_path,
        normalization_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required files missing: {missing}")
    if not data_link.is_symlink():
        raise RuntimeError(
            f"dataset path is not canonical symlink: {data_link}"
        )

    a3_report = json.loads(
        a3_report_path.read_text(encoding="utf-8")
    )
    a3_lock = json.loads(
        a3_lock_path.read_text(encoding="utf-8")
    )
    protocol = json.loads(
        protocol_path.read_text(encoding="utf-8")
    )

    if a3_report.get("status") != "PASS":
        raise RuntimeError("A3 report is not PASS")
    if not a3_report.get("decision", {}).get(
        "preliminary_baseline_training_authorized"
    ):
        raise RuntimeError(
            "A3 did not authorize preliminary training"
        )
    if a3_report.get("sealed_test", {}).get(
        "test_tensor_loaded"
    ):
        raise RuntimeError(
            "A3 reports sealed-test tensor access"
        )
    if a3_lock.get("report_sha256") != sha256_file(
        a3_report_path
    ):
        raise RuntimeError("A3 report/lock SHA mismatch")
    if a3_lock.get("protocol_sha256") != sha256_file(
        protocol_path
    ):
        raise RuntimeError("A3 protocol/lock SHA mismatch")

    # Process-wide guard against any test-run deserialization.
    original_torch_load = torch.load
    loaded_paths: list[str] = []

    def guarded_torch_load(file, *load_args, **load_kwargs):
        try:
            candidate = Path(
                os.fspath(file)
            ).expanduser().resolve()
        except TypeError:
            candidate = None
        if candidate is not None:
            candidate_text = str(candidate)
            loaded_paths.append(candidate_text)
            if "/runs/test/" in candidate_text:
                raise PermissionError(
                    "A4 sealed-test guard blocked "
                    f"{candidate_text}"
                )
        return original_torch_load(
            file,
            *load_args,
            **load_kwargs,
        )

    torch.load = guarded_torch_load

    wrapper_mod = import_module(
        wrapper_path,
        "_v5_p3_a4_guarded_loader",
    )
    b3_mod = import_module(b3_path, "_v5_p3_a4_b3")
    canonical_mod = import_module(
        canonical_path,
        "_v5_p3_a4_canonical",
    )
    dynamic_mod = import_module(
        dynamic_path,
        "_v5_p3_a4_dynamic70",
    )

    GuardedDataset = (
        wrapper_mod.GuardedV5P3TrancheAPreliminaryDataset
    )
    SealedTestAccessError = (
        wrapper_mod.SealedTestAccessError
    )
    try:
        GuardedDataset(data_root, "test")
    except SealedTestAccessError:
        test_negative_check = True
    else:
        test_negative_check = False
    if not test_negative_check:
        raise RuntimeError("guard failed to reject A_test")

    train_dataset = GuardedDataset(
        data_root,
        "train",
        active_only=False,
    )
    validation_dataset = GuardedDataset(
        data_root,
        "validation",
        active_only=False,
    )
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(
            f"train length={len(train_dataset)}"
        )
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(
            f"validation length={len(validation_dataset)}"
        )

    optimization = protocol["optimization"]
    batch_size = int(optimization["batch_size"])
    num_workers = int(optimization["num_workers"])
    maximum_epochs = int(optimization["maximum_epochs"])
    gradient_clip = float(
        optimization["gradient_clip_norm"]
    )

    sampler = EpochShuffleSampler(
        len(train_dataset),
        SEED,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    first_train_sample = train_dataset[0]
    edge = torch.as_tensor(
        first_train_sample["edge_index"]
    ).long()
    if tuple(edge.shape) != (2, 48):
        raise RuntimeError(
            f"edge shape={tuple(edge.shape)}"
        )

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    canonical_model = (
        canonical_mod.P2TaskDGraphConvCount4(
            reference_b3,
            edge,
        )
    )
    model = dynamic_mod.build_v6_p0_dynamic70_from_canonical_structure(
        canonical_model,
        seed=SEED,
    )
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"parameter count={parameter_count}, "
            f"expected={EXPECTED_PARAMETER_COUNT}"
        )
    if not all(
        parameter.requires_grad
        for parameter in model.parameters()
    ):
        raise RuntimeError("some model parameters are frozen")
    model = model.to(device)

    loss_protocol = protocol["loss"]
    graph_pos_weight = torch.tensor(
        float(loss_protocol["graph_positive_weight"]),
        dtype=torch.float32,
        device=device,
    )
    role_pos_weights = {
        role: torch.tensor(
            float(
                loss_protocol["role_positive_weights"][role]
            ),
            dtype=torch.float32,
            device=device,
        )
        for role in (
            "source",
            "transit",
            "victim",
            "path",
        )
    }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        betas=tuple(optimization["betas"]),
        eps=float(optimization["epsilon"]),
        weight_decay=float(
            optimization["weight_decay"]
        ),
    )
    scheduler_cfg = optimization["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_cfg["mode"],
        factor=float(scheduler_cfg["factor"]),
        patience=int(scheduler_cfg["patience"]),
        threshold=float(scheduler_cfg["threshold"]),
        threshold_mode="abs",
        cooldown=0,
        min_lr=float(scheduler_cfg["minimum_lr"]),
    )

    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_rank = None
    best_epoch = None
    best_validation = None
    best_checkpoint_sha256 = None
    best_early_stop_score = float("-inf")
    patience_counter = 0
    stopped_early = False
    resumed = False

    if last_checkpoint_path.is_file():
        checkpoint = torch.load(
            last_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        expected_provenance = {
            "protocol_sha256": sha256_file(protocol_path),
            "dynamic70_model_sha256": sha256_file(
                dynamic_path
            ),
            "guarded_loader_sha256": sha256_file(
                wrapper_path
            ),
            "normalization_sha256": sha256_file(
                normalization_path
            ),
        }
        for key, expected in expected_provenance.items():
            if checkpoint.get(key) != expected:
                raise RuntimeError(
                    f"resume provenance mismatch for {key}"
                )
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_rank = checkpoint["best_rank"]
        best_epoch = checkpoint["best_epoch"]
        best_validation = checkpoint["best_validation"]
        best_checkpoint_sha256 = checkpoint[
            "best_checkpoint_sha256"
        ]
        best_early_stop_score = float(
            checkpoint["best_early_stop_score"]
        )
        patience_counter = int(
            checkpoint["patience_counter"]
        )
        history = checkpoint["history"]
        resumed = True
        print(
            f"resume_from_epoch={start_epoch - 1}; "
            f"next_epoch={start_epoch}"
        )

    run_start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, maximum_epochs + 1):
        epoch_start = time.time()
        sampler.set_epoch(epoch)
        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            graph_pos_weight,
            role_pos_weights,
            gradient_clip,
        )
        validation = validate(
            model,
            validation_loader,
            device,
            graph_pos_weight,
            role_pos_weights,
        )

        score = float(validation["selection_score"])
        validation_loss = float(
            validation["loss"]["total"]
        )
        rank = (
            score,
            -validation_loss,
            -epoch,
        )
        is_best = best_rank is None or rank > tuple(best_rank)

        if is_best:
            best_rank = list(rank)
            best_epoch = epoch
            best_validation = validation
            best_payload = {
                "stage": STAGE,
                "campaign_label": CAMPAIGN_LABEL,
                "seed": SEED,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "validation_metrics": validation,
                "parameter_count": parameter_count,
                "protocol_sha256": sha256_file(
                    protocol_path
                ),
                "a3_report_sha256": sha256_file(
                    a3_report_path
                ),
                "dynamic70_model_sha256": sha256_file(
                    dynamic_path
                ),
                "guarded_loader_sha256": sha256_file(
                    wrapper_path
                ),
                "normalization_sha256": sha256_file(
                    normalization_path
                ),
                "threshold_tuning_performed": False,
                "test_tensor_loaded": False,
            }
            atomic_torch_save(
                best_payload,
                best_checkpoint_path,
            )
            best_checkpoint_sha256 = sha256_file(
                best_checkpoint_path
            )

        if score > best_early_stop_score + float(
            optimization["early_stopping"][
                "minimum_improvement"
            ]
        ):
            best_early_stop_score = score
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step(score)

        row = flatten_history_row(
            epoch=epoch,
            learning_rate=current_lr,
            train_metrics=train_metrics,
            validation=validation,
            elapsed_seconds=time.time() - epoch_start,
            best_epoch=best_epoch,
            patience_counter=patience_counter,
            is_best=is_best,
        )
        history.append(row)
        write_csv(history_path, history)

        last_payload = {
            "stage": STAGE,
            "campaign_label": CAMPAIGN_LABEL,
            "seed": SEED,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_rank": best_rank,
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "best_checkpoint_sha256": (
                best_checkpoint_sha256
            ),
            "best_early_stop_score": (
                best_early_stop_score
            ),
            "patience_counter": patience_counter,
            "history": history,
            "protocol_sha256": sha256_file(protocol_path),
            "dynamic70_model_sha256": sha256_file(
                dynamic_path
            ),
            "guarded_loader_sha256": sha256_file(
                wrapper_path
            ),
            "normalization_sha256": sha256_file(
                normalization_path
            ),
            "threshold_tuning_performed": False,
            "test_tensor_loaded": False,
        }
        atomic_torch_save(
            last_payload,
            last_checkpoint_path,
        )

        atomic_json(
            progress_path,
            {
                "stage": STAGE,
                "status": "RUNNING",
                "campaign_label": CAMPAIGN_LABEL,
                "seed": SEED,
                "completed_epoch": epoch,
                "best_epoch": best_epoch,
                "best_checkpoint_sha256": (
                    best_checkpoint_sha256
                ),
                "best_validation_metrics": (
                    best_validation
                ),
                "current_validation_metrics": validation,
                "current_learning_rate_after_scheduler": (
                    float(optimizer.param_groups[0]["lr"])
                ),
                "early_stop_patience_counter": (
                    patience_counter
                ),
                "resumed": resumed,
                "threshold_tuning_performed": False,
                "test_dataset_instantiated": False,
                "test_tensor_loaded": False,
            },
        )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['total']:.6f} "
            f"val_loss={validation_loss:.6f} "
            f"score={score:.6f} "
            f"g_auc={validation['graph']['auroc']:.6f} "
            f"g_ap={validation['graph']['average_precision']:.6f} "
            f"g_fpr={validation['graph']['fpr']:.6f} "
            f"count_f1={validation['count_active']['macro_f1']:.6f} "
            f"src_ap={validation['roles']['source']['average_precision']:.6f} "
            f"tr_ap={validation['roles']['transit']['average_precision']:.6f} "
            f"vic_ap={validation['roles']['victim']['average_precision']:.6f} "
            f"path_ap={validation['roles']['path']['average_precision']:.6f} "
            f"lr={current_lr:.8g} "
            f"best_epoch={best_epoch} "
            f"patience={patience_counter}"
        )

        early_cfg = optimization["early_stopping"]
        if (
            epoch >= int(early_cfg["minimum_epochs"])
            and patience_counter
            >= int(early_cfg["patience"])
        ):
            stopped_early = True
            print(
                f"early_stopping_epoch={epoch}; "
                f"patience={patience_counter}"
            )
            break

    if (
        best_epoch is None
        or best_validation is None
        or not best_checkpoint_path.is_file()
    ):
        raise RuntimeError("no best checkpoint was produced")

    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    test_loaded_paths = [
        path
        for path in loaded_paths
        if "/runs/test/" in path
    ]
    if test_loaded_paths:
        raise RuntimeError(
            f"sealed-test paths loaded: {test_loaded_paths}"
        )

    completed_epoch = history[-1]["epoch"]
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    elapsed = time.time() - run_start

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "interpretation": (
            "Tranche-A train/validation preliminary diagnostic. "
            "Not a sealed-test or final 1500-pair result."
        ),
        "seed": SEED,
        "device": str(device),
        "resumed": resumed,
        "model": {
            "class": model.__class__.__name__,
            "builder": (
                "build_v6_p0_dynamic70_from_"
                "canonical_structure"
            ),
            "input_shape": [16, 70, 32],
            "parameter_count": parameter_count,
            "all_parameters_trainable": True,
            "fresh_initialization_at_original_start": True,
            "historical_checkpoint_reused": False,
        },
        "data": {
            "representation": "Dynamic70",
            "train_items": len(train_dataset),
            "validation_items": len(
                validation_dataset
            ),
            "train_pairs": 600,
            "validation_pairs": 75,
            "normalization_scope": (
                "Tranche-A provisional A_train-only"
            ),
            "test_dataset_instantiated": False,
            "test_tensor_loaded": False,
        },
        "training": {
            "epochs_completed": completed_epoch,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "batch_size": batch_size,
            "elapsed_seconds_this_invocation": elapsed,
            "peak_cuda_memory_bytes": peak_memory,
        },
        "selection": {
            "formula": (
                "mean(graph_AUROC, graph_AP, "
                "count_active_macro_F1, source_AP, "
                "transit_AP, victim_AP, path_AP)"
            ),
            "threshold_tuning_performed": False,
            "best_validation_selection_score": (
                best_validation["selection_score"]
            ),
            "best_validation_metrics": best_validation,
        },
        "artifacts": {
            "best_checkpoint": str(
                best_checkpoint_path
            ),
            "best_checkpoint_sha256": sha256_file(
                best_checkpoint_path
            ),
            "last_checkpoint": str(
                last_checkpoint_path
            ),
            "last_checkpoint_sha256": sha256_file(
                last_checkpoint_path
            ),
            "history_csv": str(history_path),
            "history_csv_sha256": sha256_file(
                history_path
            ),
            "progress_json": str(progress_path),
            "progress_json_sha256": sha256_file(
                progress_path
            ),
        },
        "sealed_test": {
            "negative_guard_check": test_negative_check,
            "test_dataset_instantiated": False,
            "test_length_computed": False,
            "test_tensor_loaded": False,
            "test_loaded_paths": test_loaded_paths,
            "evaluation_authorized": False,
        },
        "decision": {
            "tranche_a_preliminary_diagnostic_complete": True,
            "result_label": CAMPAIGN_LABEL,
            "architecture_change_authorized": False,
            "additional_tranche_a_seed_sweep_authorized": False,
            "sealed_test_evaluation_authorized": False,
            "next_stage": (
                "V5_P3_A5_TRANCHE_A_PRELIMINARY_"
                "DIAGNOSTIC_REVIEW_AND_B_HANDOVER_READINESS"
            ),
        },
        "provenance": {
            "a3_report_sha256": sha256_file(
                a3_report_path
            ),
            "a3_lock_sha256": sha256_file(
                a3_lock_path
            ),
            "protocol_sha256": sha256_file(
                protocol_path
            ),
            "guarded_loader_sha256": sha256_file(
                wrapper_path
            ),
            "b3_source_sha256": sha256_file(b3_path),
            "canonical_model_sha256": sha256_file(
                canonical_path
            ),
            "dynamic70_model_sha256": sha256_file(
                dynamic_path
            ),
            "normalization_sha256": sha256_file(
                normalization_path
            ),
            "installed_script_sha256": sha256_file(
                installed_script
            ),
        },
        "source_modified": True,
        "certified_dataset_modified": False,
        "generalization_claim_authorized": (
            "A_validation preliminary only"
        ),
        "final_1500_pair_claim_authorized": False,
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "campaign_label": CAMPAIGN_LABEL,
        "seed": SEED,
        "report_sha256": sha256_file(report_path),
        "best_checkpoint_sha256": sha256_file(
            best_checkpoint_path
        ),
        "last_checkpoint_sha256": sha256_file(
            last_checkpoint_path
        ),
        "history_csv_sha256": sha256_file(
            history_path
        ),
        "progress_json_sha256": sha256_file(
            progress_path
        ),
        "protocol_sha256": sha256_file(
            protocol_path
        ),
        "dynamic70_model_sha256": sha256_file(
            dynamic_path
        ),
        "guarded_loader_sha256": sha256_file(
            wrapper_path
        ),
        "normalization_sha256": sha256_file(
            normalization_path
        ),
        "best_epoch": best_epoch,
        "best_validation_selection_score": (
            best_validation["selection_score"]
        ),
        "parameter_count": parameter_count,
        "input_features": 70,
        "test_tensor_loaded": False,
        "certified_dataset_modified": False,
    }
    atomic_json(lock_path, lock)

    complete_path.write_text(
        f"{STAGE}_COMPLETE\n",
        encoding="utf-8",
    )
    if hold_path.exists():
        hold_path.unlink()

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign_label={CAMPAIGN_LABEL}")
    print(f"seed={SEED}")
    print(f"device={device}")
    print("representation=Dynamic70")
    print("input_features=70")
    print(f"parameter_count={parameter_count}")
    print(f"epochs_completed={completed_epoch}")
    print(f"best_epoch={best_epoch}")
    print(
        "best_validation_selection_score="
        f"{best_validation['selection_score']:.8f}"
    )
    print(
        "best_graph_auroc="
        f"{best_validation['graph']['auroc']:.8f}"
    )
    print(
        "best_graph_average_precision="
        f"{best_validation['graph']['average_precision']:.8f}"
    )
    print(
        "best_graph_f1_at_0p5="
        f"{best_validation['graph']['f1']:.8f}"
    )
    print(
        "best_graph_fpr_at_0p5="
        f"{best_validation['graph']['fpr']:.8f}"
    )
    print(
        "best_count_active_macro_f1="
        f"{best_validation['count_active']['macro_f1']:.8f}"
    )
    for role in (
        "source",
        "transit",
        "victim",
        "path",
    ):
        print(
            f"best_{role}_average_precision="
            f"{best_validation['roles'][role]['average_precision']:.8f}"
        )
        print(
            f"best_{role}_exact_active_at_0p5="
            f"{best_validation['roles'][role]['exact_active']:.8f}"
        )
    print("threshold_tuning_performed=false")
    print("test_dataset_instantiated=false")
    print("test_length_computed=false")
    print("test_tensor_loaded=false")
    print("sealed_test_evaluation_authorized=false")
    print(
        "next_stage="
        "V5_P3_A5_TRANCHE_A_PRELIMINARY_"
        "DIAGNOSTIC_REVIEW_AND_B_HANDOVER_READINESS"
    )
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    print(f"best_checkpoint={best_checkpoint_path}")
    print(f"last_checkpoint={last_checkpoint_path}")
    print(f"history_csv={history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
