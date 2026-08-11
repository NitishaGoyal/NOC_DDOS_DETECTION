#!/usr/bin/env python3
"""
V5 P0-A4 Tiny-Subset Overfit Diagnostic

Purpose
-------
Prove that the frozen A3 loader, PRIMARY58 inputs, graph topology, structural
mask, multitask labels, loss functions, optimizer, and checkpoint path can
learn a deliberately tiny TRAIN-only subset.

This is not a baseline result and must not be cited as generalization
performance.

Data boundary
-------------
- Reads TRAIN tensors/windows only.
- Selects one single-attacker matched pair and one dual-attacker matched pair.
- Does not read validation or test tensors.
- Performs no threshold search; all binary heads use threshold 0.5.
"""

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
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


ROLE_KEYS = (
    "y_source",
    "y_transit",
    "y_victim",
    "y_attack_path",
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


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def scalar_text(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"expected scalar tensor, got {tuple(value.shape)}"
            )
        value = value.detach().cpu().item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def import_contract_dataset(wrapper_path: Path):
    spec = importlib.util.spec_from_file_location(
        "v5_p0_contract_loader",
        wrapper_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import wrapper: {wrapper_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset_class = getattr(
        module,
        "V5P0ContractWindowDataset",
        None,
    )
    if dataset_class is None:
        raise AttributeError(
            "wrapper does not define V5P0ContractWindowDataset"
        )
    return dataset_class


def verify_a3(
    a3_dir: Path,
    failures: list[str],
) -> dict[str, Any]:
    marker = (
        a3_dir
        / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_PASS"
    )
    report_path = (
        a3_dir
        / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE.json"
    )
    lock_path = (
        a3_dir
        / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_LOCK.json"
    )

    for path in (marker, report_path, lock_path):
        if not path.is_file():
            failures.append(f"required A3 artifact missing: {path}")

    if failures:
        return {}

    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )
    lock = json.loads(
        lock_path.read_text(encoding="utf-8")
    )

    if report.get("status") != "PASS":
        failures.append("A3 report is not PASS")
    if lock.get("status") != (
        "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_PASS"
    ):
        failures.append("A3 lock is not PASS")
    if lock.get("report_sha256") != sha256_file(report_path):
        failures.append("A3 report SHA does not match lock")
    if report.get("feature_variant") != "PRIMARY58":
        failures.append("A3 did not freeze PRIMARY58")
    if int(report.get("window")) != 32:
        failures.append("A3 window is not 32")
    if int(report.get("stride")) != 8:
        failures.append("A3 stride is not 8")
    if bool(report.get("normalization_applied_by_wrapper")):
        failures.append("A3 applied normalization in wrapper")

    return {
        "report_sha256": sha256_file(report_path),
        "lock_sha256": sha256_file(lock_path),
        "report": report,
    }


def discover_pairs(
    train_dir: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Path],
]:
    pairs: dict[str, dict[str, Any]] = defaultdict(dict)
    run_paths: dict[str, Path] = {}

    for path in sorted(train_dir.glob("*.pt")):
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        pair_id = scalar_text(payload["pair_id"])
        run_id = scalar_text(payload["run_id"])
        mode = scalar_text(payload["mode"]).lower()

        pairs[pair_id][mode] = {
            "run_id": run_id,
            "case_id": scalar_text(payload["case_id"]),
            "T": int(payload["x"].shape[0]),
            "max_attacker_count": int(
                payload["y_attacker_count"].max().item()
            ),
            "active_epochs": int(
                payload["y_attack"].bool().sum().item()
            ),
            "source_positive_entries": int(
                payload["y_source"].bool().sum().item()
            ),
            "transit_positive_entries": int(
                payload["y_transit"].bool().sum().item()
            ),
            "victim_positive_entries": int(
                payload["y_victim"].bool().sum().item()
            ),
            "path_positive_entries": int(
                payload["y_attack_path"].bool().sum().item()
            ),
            "source_victim_overlap_entries": int(
                (
                    payload["y_source"].bool()
                    & payload["y_victim"].bool()
                ).sum().item()
            ),
            "path": str(path),
        }
        run_paths[run_id] = path

    return dict(pairs), run_paths


def select_pairs(
    pairs: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    candidates = []

    for pair_id, members in pairs.items():
        attack = members.get("attack")
        control = members.get("control")
        if attack is None or control is None:
            continue

        role_coverage = sum(
            int(attack[key] > 0)
            for key in (
                "source_positive_entries",
                "transit_positive_entries",
                "victim_positive_entries",
                "path_positive_entries",
            )
        )
        score = (
            role_coverage,
            int(attack["transit_positive_entries"] > 0),
            int(attack["victim_positive_entries"] > 0),
            attack["active_epochs"],
        )
        candidates.append(
            (
                pair_id,
                attack["max_attacker_count"],
                score,
            )
        )

    single = sorted(
        (
            candidate
            for candidate in candidates
            if candidate[1] == 1
        ),
        key=lambda item: item[2],
        reverse=True,
    )
    dual = sorted(
        (
            candidate
            for candidate in candidates
            if candidate[1] == 2
        ),
        key=lambda item: item[2],
        reverse=True,
    )

    if not single:
        raise RuntimeError(
            "no complete single-attacker pair found in train split"
        )
    if not dual:
        raise RuntimeError(
            "no complete dual-attacker pair found in train split"
        )

    return single[0][0], dual[0][0]


class DenseTemporalGraphOverfitNet(nn.Module):
    """
    Deliberately high-capacity diagnostic model.

    This is not the final architecture. It exists only to prove the tiny
    subset and multitask training path are learnable.
    """

    def __init__(
        self,
        feature_count: int = 58,
        window: int = 32,
        node_hidden: int = 256,
    ) -> None:
        super().__init__()

        flattened = feature_count * window

        self.temporal = nn.Sequential(
            nn.Linear(flattened, 512),
            nn.ReLU(),
            nn.Linear(512, node_hidden),
            nn.ReLU(),
        )
        self.structure_fuse = nn.Sequential(
            nn.Linear(node_hidden + 10, node_hidden),
            nn.ReLU(),
        )

        self.g1_self = nn.Linear(node_hidden, node_hidden)
        self.g1_neigh = nn.Linear(node_hidden, node_hidden)
        self.g2_self = nn.Linear(node_hidden, node_hidden)
        self.g2_neigh = nn.Linear(node_hidden, node_hidden)

        node_input = node_hidden * 2
        self.source_head = nn.Sequential(
            nn.Linear(node_input, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.transit_head = nn.Sequential(
            nn.Linear(node_input, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.victim_head = nn.Sequential(
            nn.Linear(node_input, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.path_head = nn.Sequential(
            nn.Linear(node_input, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        graph_input = node_hidden * 2
        self.attack_head = nn.Sequential(
            nn.Linear(graph_input, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.count_head = nn.Sequential(
            nn.Linear(graph_input, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    @staticmethod
    def normalized_adjacency(
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if edge_index.ndim == 3:
            edge_index = edge_index[0]
        edge_index = edge_index.to(device=device)

        adjacency = torch.zeros(
            num_nodes,
            num_nodes,
            dtype=dtype,
            device=device,
        )
        adjacency[
            edge_index[1].long(),
            edge_index[0].long(),
        ] = 1
        adjacency.fill_diagonal_(1)

        degree = adjacency.sum(dim=1).clamp_min(1)
        inv_sqrt = degree.rsqrt()
        return (
            inv_sqrt[:, None]
            * adjacency
            * inv_sqrt[None, :]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, nodes, features, window = x.shape

        local = self.temporal(
            x.reshape(batch * nodes, features * window)
        ).reshape(batch, nodes, -1)

        mask = physical_port_mask.to(
            dtype=local.dtype,
            device=local.device,
        )
        local = self.structure_fuse(
            torch.cat([local, mask], dim=-1)
        )

        adjacency = self.normalized_adjacency(
            edge_index=edge_index,
            num_nodes=nodes,
            device=local.device,
            dtype=local.dtype,
        )

        neighbor1 = torch.einsum(
            "ij,bjh->bih",
            adjacency,
            local,
        )
        graph1 = F.relu(
            self.g1_self(local)
            + self.g1_neigh(neighbor1)
        )

        neighbor2 = torch.einsum(
            "ij,bjh->bih",
            adjacency,
            graph1,
        )
        graph2 = F.relu(
            self.g2_self(graph1)
            + self.g2_neigh(neighbor2)
        )

        node_repr = torch.cat(
            [local, graph2],
            dim=-1,
        )
        graph_repr = torch.cat(
            [
                graph2.mean(dim=1),
                graph2.max(dim=1).values,
            ],
            dim=-1,
        )

        return {
            "attack": self.attack_head(
                graph_repr
            ).squeeze(-1),
            "count": self.count_head(graph_repr),
            "source": self.source_head(
                node_repr
            ).squeeze(-1),
            "transit": self.transit_head(
                node_repr
            ).squeeze(-1),
            "victim": self.victim_head(
                node_repr
            ).squeeze(-1),
            "path": self.path_head(
                node_repr
            ).squeeze(-1),
        }


def collect_targets(
    dataset: Subset,
) -> dict[str, torch.Tensor]:
    values: dict[str, list[torch.Tensor]] = {
        "attack": [],
        "count": [],
        "source": [],
        "transit": [],
        "victim": [],
        "path": [],
    }

    key_map = {
        "attack": "y_attack",
        "count": "y_attacker_count",
        "source": "y_source",
        "transit": "y_transit",
        "victim": "y_victim",
        "path": "y_attack_path",
    }

    for index in range(len(dataset)):
        item = dataset[index]
        for target_name, item_key in key_map.items():
            value = item[item_key]
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            values[target_name].append(
                value.detach().cpu()
            )

    return {
        key: torch.stack(items)
        for key, items in values.items()
    }


def positive_weight(
    target: torch.Tensor,
    cap: float = 20.0,
) -> float:
    target = target.float()
    positives = float(target.sum().item())
    negatives = float(target.numel() - positives)

    if positives <= 0:
        return 1.0
    return min(cap, negatives / positives)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["attack"].device

    attack_target = batch["y_attack"].to(
        device=device,
        dtype=torch.float32,
    )
    count_target = batch["y_attacker_count"].to(
        device=device,
        dtype=torch.long,
    )
    source_target = batch["y_source"].to(
        device=device,
        dtype=torch.float32,
    )
    transit_target = batch["y_transit"].to(
        device=device,
        dtype=torch.float32,
    )
    victim_target = batch["y_victim"].to(
        device=device,
        dtype=torch.float32,
    )
    path_target = batch["y_attack_path"].to(
        device=device,
        dtype=torch.float32,
    )

    losses = {
        "attack": F.binary_cross_entropy_with_logits(
            outputs["attack"],
            attack_target,
            pos_weight=pos_weights["attack"],
        ),
        "count": F.cross_entropy(
            outputs["count"],
            count_target,
        ),
        "source": F.binary_cross_entropy_with_logits(
            outputs["source"],
            source_target,
            pos_weight=pos_weights["source"],
        ),
        "transit": F.binary_cross_entropy_with_logits(
            outputs["transit"],
            transit_target,
            pos_weight=pos_weights["transit"],
        ),
        "victim": F.binary_cross_entropy_with_logits(
            outputs["victim"],
            victim_target,
            pos_weight=pos_weights["victim"],
        ),
        "path": F.binary_cross_entropy_with_logits(
            outputs["path"],
            path_target,
            pos_weight=pos_weights["path"],
        ),
    }

    total = (
        1.00 * losses["attack"]
        + 0.50 * losses["count"]
        + 1.00 * losses["source"]
        + 0.50 * losses["transit"]
        + 0.75 * losses["victim"]
        + 0.50 * losses["path"]
    )

    return total, {
        key: float(value.detach().item())
        for key, value in losses.items()
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    totals = {
        "samples": 0,
        "attack_correct": 0,
        "count_correct": 0,
        "source_exact": 0,
        "transit_exact": 0,
        "victim_exact": 0,
        "path_exact": 0,
        "all_tasks_exact": 0,
    }

    node_counts = {
        role: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for role in (
            "source",
            "transit",
            "victim",
            "path",
        )
    }

    for batch in loader:
        x = batch["x"].to(device=device)
        edge_index = batch["edge_index"].to(
            device=device
        )
        mask = batch["physical_port_mask"].to(
            device=device
        )

        outputs = model(
            x=x,
            edge_index=edge_index,
            physical_port_mask=mask,
        )

        attack_pred = (
            torch.sigmoid(outputs["attack"]) >= 0.5
        )
        count_pred = outputs["count"].argmax(dim=-1)

        attack_target = batch["y_attack"].to(
            device=device
        ).bool()
        count_target = batch["y_attacker_count"].to(
            device=device
        ).long()

        role_exact = {}
        role_predictions = {}
        role_targets = {}

        for role, item_key in (
            ("source", "y_source"),
            ("transit", "y_transit"),
            ("victim", "y_victim"),
            ("path", "y_attack_path"),
        ):
            prediction = (
                torch.sigmoid(outputs[role]) >= 0.5
            )
            target = batch[item_key].to(
                device=device
            ).bool()

            exact = (
                prediction == target
            ).all(dim=1)

            role_exact[role] = exact
            role_predictions[role] = prediction
            role_targets[role] = target

            node_counts[role]["tp"] += int(
                (prediction & target).sum().item()
            )
            node_counts[role]["fp"] += int(
                (prediction & ~target).sum().item()
            )
            node_counts[role]["fn"] += int(
                (~prediction & target).sum().item()
            )

        batch_size = int(x.shape[0])
        totals["samples"] += batch_size
        totals["attack_correct"] += int(
            (attack_pred == attack_target).sum().item()
        )
        totals["count_correct"] += int(
            (count_pred == count_target).sum().item()
        )

        for role in role_exact:
            totals[f"{role}_exact"] += int(
                role_exact[role].sum().item()
            )

        all_exact = (
            (attack_pred == attack_target)
            & (count_pred == count_target)
            & role_exact["source"]
            & role_exact["transit"]
            & role_exact["victim"]
            & role_exact["path"]
        )
        totals["all_tasks_exact"] += int(
            all_exact.sum().item()
        )

    samples = max(1, totals["samples"])
    metrics = {
        "attack_accuracy": (
            totals["attack_correct"] / samples
        ),
        "count_accuracy": (
            totals["count_correct"] / samples
        ),
        "source_exact_set": (
            totals["source_exact"] / samples
        ),
        "transit_exact_set": (
            totals["transit_exact"] / samples
        ),
        "victim_exact_set": (
            totals["victim_exact"] / samples
        ),
        "path_exact_set": (
            totals["path_exact"] / samples
        ),
        "all_tasks_exact": (
            totals["all_tasks_exact"] / samples
        ),
        "samples": totals["samples"],
    }

    for role, counts in node_counts.items():
        denominator = (
            2 * counts["tp"]
            + counts["fp"]
            + counts["fn"]
        )
        metrics[f"{role}_node_f1"] = (
            2 * counts["tp"] / denominator
            if denominator
            else 1.0
        )

    return metrics


def gate_passed(metrics: dict[str, float]) -> bool:
    return (
        metrics["attack_accuracy"] >= 0.995
        and metrics["count_accuracy"] >= 0.995
        and metrics["source_exact_set"] >= 0.98
        and metrics["transit_exact_set"] >= 0.95
        and metrics["victim_exact_set"] >= 0.98
        and metrics["path_exact_set"] >= 0.95
        and metrics["all_tasks_exact"] >= 0.90
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--a2-1-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    a2_dir = args.a2_dir.expanduser().resolve()
    a2_1_dir = args.a2_1_dir.expanduser().resolve()
    a3_dir = args.a3_dir.expanduser().resolve()
    wrapper_path = args.wrapper.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()

    if output_dir.exists():
        print(
            f"STOP: output directory already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    if model_dir.exists():
        print(
            f"STOP: model directory already exists: {model_dir}",
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    a3_provenance = verify_a3(
        a3_dir,
        failures,
    )

    if not wrapper_path.is_file():
        failures.append(
            f"contract loader missing: {wrapper_path}"
        )

    if failures:
        report = {
            "stage": "V5_P0_A4_TINY_SUBSET_OVERFIT",
            "status": "FAIL",
            "failures": failures,
            "warnings": warnings,
        }
        report_path = (
            output_dir
            / "V5_P0_A4_TINY_SUBSET_OVERFIT.json"
        )
        write_json(report_path, report)
        atomic_write(
            output_dir
            / "V5_P0_A4_TINY_SUBSET_OVERFIT_HOLD",
            "V5_P0_A4_TINY_SUBSET_OVERFIT_HOLD\n",
        )
        print("V5_P0_A4_TINY_SUBSET_OVERFIT_HOLD")
        return 1

    set_seed(args.seed)

    ContractDataset = import_contract_dataset(
        wrapper_path
    )
    base_dataset = ContractDataset(
        root=root,
        split="train",
        contract_dir=a2_dir,
        mask_audit_dir=a2_1_dir,
        window=32,
        stride=8,
        active_only=False,
        feature_variant="PRIMARY58",
    )

    pairs, run_paths = discover_pairs(
        root / "runs" / "train"
    )
    single_pair, dual_pair = select_pairs(pairs)

    selected_pairs = [single_pair, dual_pair]
    selected_run_ids = set()
    selected_pair_report = {}

    for pair_id in selected_pairs:
        members = pairs[pair_id]
        selected_pair_report[pair_id] = members
        for mode in ("control", "attack"):
            selected_run_ids.add(
                members[mode]["run_id"]
            )

    subset_indices = []
    run_window_counts = defaultdict(int)

    for index in range(len(base_dataset)):
        item = base_dataset[index]
        run_id = scalar_text(item["run_id"])
        if run_id in selected_run_ids:
            subset_indices.append(index)
            run_window_counts[run_id] += 1

    if not subset_indices:
        failures.append(
            "selected tiny subset contains zero windows"
        )

    for run_id in sorted(selected_run_ids):
        if run_window_counts[run_id] == 0:
            failures.append(
                f"selected run has zero windows: {run_id}"
            )

    tiny_dataset = Subset(
        base_dataset,
        subset_indices,
    )

    targets = collect_targets(tiny_dataset)
    target_coverage = {
        "samples": len(tiny_dataset),
        "attack_positive_samples": int(
            targets["attack"].bool().sum().item()
        ),
        "count_values": sorted(
            int(value)
            for value in torch.unique(
                targets["count"]
            ).tolist()
        ),
        "source_positive_entries": int(
            targets["source"].bool().sum().item()
        ),
        "transit_positive_entries": int(
            targets["transit"].bool().sum().item()
        ),
        "victim_positive_entries": int(
            targets["victim"].bool().sum().item()
        ),
        "path_positive_entries": int(
            targets["path"].bool().sum().item()
        ),
    }

    for role in (
        "source",
        "transit",
        "victim",
        "path",
    ):
        if target_coverage[
            f"{role}_positive_entries"
        ] == 0:
            failures.append(
                f"tiny subset has no positive {role} labels"
            )

    if set(target_coverage["count_values"]) != {0, 1, 2}:
        failures.append(
            "tiny subset attacker-count labels do not cover 0,1,2: "
            f"{target_coverage['count_values']}"
        )

    if failures:
        report = {
            "stage": "V5_P0_A4_TINY_SUBSET_OVERFIT",
            "status": "FAIL",
            "failures": failures,
            "warnings": warnings,
            "selected_pairs": selected_pair_report,
            "target_coverage": target_coverage,
        }
        report_path = (
            output_dir
            / "V5_P0_A4_TINY_SUBSET_OVERFIT.json"
        )
        write_json(report_path, report)
        atomic_write(
            output_dir
            / "V5_P0_A4_TINY_SUBSET_OVERFIT_HOLD",
            "V5_P0_A4_TINY_SUBSET_OVERFIT_HOLD\n",
        )
        print("V5_P0_A4_TINY_SUBSET_OVERFIT_HOLD")
        return 1

    pos_weight_values = {
        "attack": positive_weight(
            targets["attack"]
        ),
        "source": positive_weight(
            targets["source"]
        ),
        "transit": positive_weight(
            targets["transit"]
        ),
        "victim": positive_weight(
            targets["victim"]
        ),
        "path": positive_weight(
            targets["path"]
        ),
    }

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        tiny_dataset,
        batch_size=min(
            args.batch_size,
            len(tiny_dataset),
        ),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    eval_loader = DataLoader(
        tiny_dataset,
        batch_size=min(
            args.batch_size,
            len(tiny_dataset),
        ),
        shuffle=False,
        num_workers=0,
    )

    model = DenseTemporalGraphOverfitNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    pos_weights = {
        key: torch.tensor(
            value,
            dtype=torch.float32,
            device=device,
        )
        for key, value in pos_weight_values.items()
    }

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    history = []
    best_metrics = None
    best_epoch = 0
    best_score = -math.inf

    print("===== V5 P0-A4 TINY-SUBSET OVERFIT =====")
    print(f"device: {device}")
    print(f"parameter_count: {parameter_count}")
    print(f"selected_single_pair: {single_pair}")
    print(f"selected_dual_pair: {dual_pair}")
    print(f"selected_run_count: {len(selected_run_ids)}")
    print(f"tiny_window_count: {len(tiny_dataset)}")
    print(f"target_coverage: {target_coverage}")
    print(f"positive_weights: {pos_weight_values}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_samples = 0
        component_totals = defaultdict(float)

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            x = batch["x"].to(
                device=device,
                dtype=torch.float32,
            )
            edge_index = batch["edge_index"].to(
                device=device
            )
            mask = batch["physical_port_mask"].to(
                device=device
            )

            outputs = model(
                x=x,
                edge_index=edge_index,
                physical_port_mask=mask,
            )
            loss, components = compute_loss(
                outputs,
                batch,
                pos_weights,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch {epoch}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=10.0,
            )
            optimizer.step()

            batch_size = int(x.shape[0])
            epoch_total += float(
                loss.detach().item()
            ) * batch_size
            epoch_samples += batch_size

            for key, value in components.items():
                component_totals[key] += (
                    value * batch_size
                )

        metrics = evaluate(
            model=model,
            loader=eval_loader,
            device=device,
        )

        mean_loss = epoch_total / max(
            1,
            epoch_samples,
        )
        row = {
            "epoch": epoch,
            "loss": mean_loss,
            **{
                f"loss_{key}": (
                    value / max(1, epoch_samples)
                )
                for key, value in component_totals.items()
            },
            **metrics,
        }
        history.append(row)

        score = (
            metrics["all_tasks_exact"]
            + metrics["source_exact_set"]
            + metrics["victim_exact_set"]
            + metrics["path_exact_set"]
            + metrics["attack_accuracy"]
            + metrics["count_accuracy"]
            - mean_loss
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = {
                **metrics,
                "loss": mean_loss,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "epoch": epoch,
                    "metrics": best_metrics,
                    "selected_pairs": selected_pair_report,
                    "selected_run_ids": sorted(
                        selected_run_ids
                    ),
                    "feature_variant": "PRIMARY58",
                    "window": 32,
                    "stride": 8,
                    "seed": args.seed,
                },
                model_dir / "best.pt",
            )

        if (
            epoch == 1
            or epoch % 10 == 0
            or gate_passed(metrics)
        ):
            print(
                f"epoch={epoch:03d} "
                f"loss={mean_loss:.6f} "
                f"attack_acc="
                f"{metrics['attack_accuracy']:.4f} "
                f"count_acc="
                f"{metrics['count_accuracy']:.4f} "
                f"source_exact="
                f"{metrics['source_exact_set']:.4f} "
                f"transit_exact="
                f"{metrics['transit_exact_set']:.4f} "
                f"victim_exact="
                f"{metrics['victim_exact_set']:.4f} "
                f"path_exact="
                f"{metrics['path_exact_set']:.4f} "
                f"all_exact="
                f"{metrics['all_tasks_exact']:.4f}"
            )

        if gate_passed(metrics):
            print(
                f"overfit_gate_reached_at_epoch: {epoch}"
            )
            break

    final_metrics = evaluate(
        model=model,
        loader=eval_loader,
        device=device,
    )
    final_pass = gate_passed(final_metrics)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": history[-1]["epoch"],
            "metrics": final_metrics,
            "selected_pairs": selected_pair_report,
            "selected_run_ids": sorted(selected_run_ids),
            "feature_variant": "PRIMARY58",
            "window": 32,
            "stride": 8,
            "seed": args.seed,
        },
        model_dir / "final.pt",
    )

    history_path = (
        output_dir / "training_history.csv"
    )
    with history_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)

    report = {
        "stage": "V5_P0_A4_TINY_SUBSET_OVERFIT",
        "status": "PASS" if final_pass else "HOLD",
        "purpose": (
            "train-only pipeline and optimization diagnostic; "
            "not a generalization result"
        ),
        "device": str(device),
        "parameter_count": parameter_count,
        "selected_pairs": selected_pair_report,
        "selected_run_ids": sorted(selected_run_ids),
        "run_window_counts": dict(
            sorted(run_window_counts.items())
        ),
        "tiny_window_count": len(tiny_dataset),
        "target_coverage": target_coverage,
        "positive_weights": pos_weight_values,
        "training": {
            "seed": args.seed,
            "maximum_epochs": args.epochs,
            "epochs_completed": history[-1]["epoch"],
            "batch_size": min(
                args.batch_size,
                len(tiny_dataset),
            ),
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "binary_threshold": 0.5,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "final_metrics": final_metrics,
        },
        "overfit_gate": {
            "attack_accuracy_min": 0.995,
            "count_accuracy_min": 0.995,
            "source_exact_set_min": 0.98,
            "transit_exact_set_min": 0.95,
            "victim_exact_set_min": 0.98,
            "path_exact_set_min": 0.95,
            "all_tasks_exact_min": 0.90,
            "passed": final_pass,
        },
        "provenance": {
            "a3_report_sha256": a3_provenance.get(
                "report_sha256"
            ),
            "a3_lock_sha256": a3_provenance.get(
                "lock_sha256"
            ),
            "wrapper_sha256": sha256_file(
                wrapper_path
            ),
            "script_sha256": sha256_file(
                Path(__file__)
            ),
            "history_sha256": sha256_file(
                history_path
            ),
            "best_checkpoint_sha256": sha256_file(
                model_dir / "best.pt"
            ),
            "final_checkpoint_sha256": sha256_file(
                model_dir / "final.pt"
            ),
        },
        "audit_boundary": {
            "train_split_only": True,
            "validation_tensors_read": False,
            "test_tensors_read": False,
            "threshold_search_performed": False,
            "validation_performance_evaluated": False,
            "test_performance_evaluated": False,
        },
        "failures": (
            []
            if final_pass
            else [
                "tiny-subset overfit gate was not reached"
            ]
        ),
        "warnings": warnings,
        "next_stage": (
            "V5_P0_B0_SHORTCUT_AND_TRIVIAL_BASELINES"
            if final_pass
            else "HOLD_DEBUG_A4_OVERFIT"
        ),
    }

    report_path = (
        output_dir
        / "V5_P0_A4_TINY_SUBSET_OVERFIT.json"
    )
    write_json(report_path, report)
    write_json(
        output_dir / "selected_pairs.json",
        selected_pair_report,
    )

    lock = {
        "status": (
            "V5_P0_A4_TINY_SUBSET_OVERFIT_PASS"
            if final_pass
            else "V5_P0_A4_TINY_SUBSET_OVERFIT_HOLD"
        ),
        "report_sha256": sha256_file(report_path),
        "script_sha256": sha256_file(Path(__file__)),
        "wrapper_sha256": sha256_file(wrapper_path),
        "history_sha256": sha256_file(history_path),
        "best_checkpoint_sha256": sha256_file(
            model_dir / "best.pt"
        ),
        "final_checkpoint_sha256": sha256_file(
            model_dir / "final.pt"
        ),
        "train_split_only": True,
        "validation_performance_evaluated": False,
        "test_performance_evaluated": False,
        "next_stage": report["next_stage"],
    }
    write_json(
        output_dir
        / "V5_P0_A4_TINY_SUBSET_OVERFIT_LOCK.json",
        lock,
    )

    print("\n===== FINAL TINY-OVERFIT RESULT =====")
    for key, value in final_metrics.items():
        print(f"{key}: {value}")
    print(f"best_epoch: {best_epoch}")
    print(f"epochs_completed: {history[-1]['epoch']}")
    print(f"failure_count: {0 if final_pass else 1}")
    print(f"warning_count: {len(warnings)}")

    if not final_pass:
        atomic_write(
            output_dir
            / "V5_P0_A4_TINY_SUBSET_OVERFIT_HOLD",
            "V5_P0_A4_TINY_SUBSET_OVERFIT_HOLD\n",
        )
        print("next_stage: HOLD_DEBUG_A4_OVERFIT")
        print("V5_P0_A4_TINY_SUBSET_OVERFIT_HOLD")
        return 1

    atomic_write(
        output_dir
        / "V5_P0_A4_TINY_SUBSET_OVERFIT_PASS",
        "V5_P0_A4_TINY_SUBSET_OVERFIT_PASS\n",
    )
    print(
        "next_stage: "
        "V5_P0_B0_SHORTCUT_AND_TRIVIAL_BASELINES"
    )
    print("V5_P0_A4_TINY_SUBSET_OVERFIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
