#!/usr/bin/env python3
"""
V5 P0-B0-R1B Numeric Provenance Shortcut Audit

This stage resolves the remaining B0-R1 HOLD caused by four TRAIN
attack/control pairs having different run lengths.

It fits small probes using only non-traffic numeric provenance:
- run length;
- run window count;
- within-run position;
- dataset/run serialization position;
- all numeric fields combined.

TRAIN is used for fitting, VALIDATION for evaluation, and TEST is never
enumerated or read. No identifier strings, mode, filenames, traffic x,
edge_index, or physical masks are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


ROLE_KEYS = {
    "source": "y_source",
    "transit": "y_transit",
    "victim": "y_victim",
    "path": "y_attack_path",
}
ROLES = tuple(ROLE_KEYS)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def scalar_text(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"expected scalar tensor, got {tuple(value.shape)}")
        value = value.detach().cpu().item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def load_split(
    root: Path,
    split: str,
    window: int,
    stride: int,
) -> dict[str, Any]:
    files = sorted((root / "runs" / split).glob("*.pt"))
    rows: dict[str, Any] = {
        "attack": [],
        "count": [],
        "roles": {role: [] for role in ROLES},
        "run_id": [],
        "pair_id": [],
        "mode": [],
        "run_length": [],
        "run_window_count": [],
        "window_start": [],
        "window_target": [],
        "window_ordinal": [],
        "run_file_order": [],
        "dataset_index": [],
    }

    dataset_index = 0
    run_records = []

    for run_file_order, path in enumerate(files):
        payload = torch.load(path, map_location="cpu", weights_only=False)

        run_id = scalar_text(payload["run_id"])
        pair_id = scalar_text(payload["pair_id"])
        mode = scalar_text(payload["mode"]).lower()
        run_length = int(payload["x"].shape[0])

        starts = list(range(0, run_length - window + 1, stride))
        run_window_count = len(starts)

        run_records.append(
            {
                "run_id": run_id,
                "pair_id": pair_id,
                "mode": mode,
                "run_length": run_length,
                "run_window_count": run_window_count,
                "run_file_order": run_file_order,
            }
        )

        for window_ordinal, start in enumerate(starts):
            target = start + window - 1

            rows["attack"].append(
                int(payload["y_attack"][target].detach().cpu().item())
            )
            rows["count"].append(
                int(payload["y_attacker_count"][target].detach().cpu().item())
            )

            for role, key in ROLE_KEYS.items():
                rows["roles"][role].append(
                    payload[key][target].detach().cpu().bool()
                )

            rows["run_id"].append(run_id)
            rows["pair_id"].append(pair_id)
            rows["mode"].append(mode)
            rows["run_length"].append(float(run_length))
            rows["run_window_count"].append(float(run_window_count))
            rows["window_start"].append(float(start))
            rows["window_target"].append(float(target))
            rows["window_ordinal"].append(float(window_ordinal))
            rows["run_file_order"].append(float(run_file_order))
            rows["dataset_index"].append(float(dataset_index))
            dataset_index += 1

    return {
        "attack": torch.tensor(rows["attack"], dtype=torch.long),
        "count": torch.tensor(rows["count"], dtype=torch.long),
        "roles": {
            role: torch.stack(rows["roles"][role]).bool()
            for role in ROLES
        },
        **{
            key: torch.tensor(rows[key], dtype=torch.float32)
            for key in (
                "run_length",
                "run_window_count",
                "window_start",
                "window_target",
                "window_ordinal",
                "run_file_order",
                "dataset_index",
            )
        },
        "run_id": rows["run_id"],
        "pair_id": rows["pair_id"],
        "mode": rows["mode"],
        "run_records": run_records,
        "file_count": len(files),
    }


def standardize(
    train: torch.Tensor,
    validation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False)
    constant = std < 1e-8
    safe_std = torch.where(constant, torch.ones_like(std), std)
    return (
        (train - mean) / safe_std,
        (validation - mean) / safe_std,
        {
            "train_mean": mean.tolist(),
            "train_std": std.tolist(),
            "constant_columns": constant.tolist(),
        },
    )


def build_features(
    train: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]:
    train_length = train["run_length"].clamp_min(1.0)
    validation_length = validation["run_length"].clamp_min(1.0)

    raw = {
        "run_length_only": (
            train["run_length"].unsqueeze(1),
            validation["run_length"].unsqueeze(1),
            ["run_length"],
        ),
        "run_window_count_only": (
            train["run_window_count"].unsqueeze(1),
            validation["run_window_count"].unsqueeze(1),
            ["run_window_count"],
        ),
        "within_run_position_only": (
            torch.stack(
                [
                    train["window_start"],
                    train["window_target"],
                    train["window_ordinal"],
                    train["window_start"] / train_length,
                    train["window_target"] / train_length,
                ],
                dim=1,
            ),
            torch.stack(
                [
                    validation["window_start"],
                    validation["window_target"],
                    validation["window_ordinal"],
                    validation["window_start"] / validation_length,
                    validation["window_target"] / validation_length,
                ],
                dim=1,
            ),
            [
                "window_start",
                "window_target",
                "window_ordinal",
                "start_fraction",
                "target_fraction",
            ],
        ),
        "serialization_only": (
            torch.stack(
                [
                    train["run_file_order"],
                    train["dataset_index"],
                ],
                dim=1,
            ),
            torch.stack(
                [
                    validation["run_file_order"],
                    validation["dataset_index"],
                ],
                dim=1,
            ),
            ["run_file_order", "dataset_index"],
        ),
        "all_numeric_provenance": (
            torch.stack(
                [
                    train["run_length"],
                    train["run_window_count"],
                    train["window_start"],
                    train["window_target"],
                    train["window_ordinal"],
                    train["window_start"] / train_length,
                    train["window_target"] / train_length,
                    train["run_file_order"],
                    train["dataset_index"],
                ],
                dim=1,
            ),
            torch.stack(
                [
                    validation["run_length"],
                    validation["run_window_count"],
                    validation["window_start"],
                    validation["window_target"],
                    validation["window_ordinal"],
                    validation["window_start"] / validation_length,
                    validation["window_target"] / validation_length,
                    validation["run_file_order"],
                    validation["dataset_index"],
                ],
                dim=1,
            ),
            [
                "run_length",
                "run_window_count",
                "window_start",
                "window_target",
                "window_ordinal",
                "start_fraction",
                "target_fraction",
                "run_file_order",
                "dataset_index",
            ],
        ),
    }

    result = {}
    for name, (train_x, validation_x, fields) in raw.items():
        train_x, validation_x, normalization = standardize(
            train_x.float(),
            validation_x.float(),
        )
        result[name] = (
            train_x,
            validation_x,
            {
                "fields": fields,
                "normalization": normalization,
            },
        )
    return result


class Probe(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.graph_head = nn.Linear(hidden, 1)
        self.count_head = nn.Linear(hidden, 3)
        self.role_heads = nn.ModuleDict(
            {role: nn.Linear(hidden, 16) for role in ROLES}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(x)
        return {
            "graph": self.graph_head(hidden).squeeze(-1),
            "count": self.count_head(hidden),
            **{
                role: self.role_heads[role](hidden)
                for role in ROLES
            },
        }


def positive_weight(target: torch.Tensor, cap: float = 20.0) -> float:
    target = target.float()
    positives = float(target.sum().item())
    negatives = float(target.numel() - positives)
    if positives <= 0:
        return 1.0
    return min(cap, negatives / positives)


def train_probe(
    name: str,
    train_x: torch.Tensor,
    validation_x: torch.Tensor,
    train: dict[str, Any],
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    set_seed(seed)

    model = Probe(int(train_x.shape[1])).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    graph_pos_weight = torch.tensor(
        positive_weight(train["attack"]),
        device=device,
    )
    role_pos_weights = {
        role: torch.tensor(
            positive_weight(train["roles"][role]),
            device=device,
        )
        for role in ROLES
    }

    batch_size = min(256, int(train_x.shape[0]))
    generator = torch.Generator().manual_seed(seed)
    final_loss = math.nan

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(
            train_x.shape[0],
            generator=generator,
        )
        weighted_loss = 0.0
        seen = 0

        for start in range(0, len(permutation), batch_size):
            indices = permutation[start:start + batch_size]
            outputs = model(train_x[indices].to(device))

            graph_target = train["attack"][indices].float().to(device)
            count_target = train["count"][indices].long().to(device)

            loss = F.binary_cross_entropy_with_logits(
                outputs["graph"],
                graph_target,
                pos_weight=graph_pos_weight,
            )
            loss = loss + 0.5 * F.cross_entropy(
                outputs["count"],
                count_target,
            )

            coefficients = {
                "source": 1.0,
                "transit": 0.5,
                "victim": 0.75,
                "path": 0.5,
            }
            for role in ROLES:
                role_target = (
                    train["roles"][role][indices]
                    .float()
                    .to(device)
                )
                loss = loss + coefficients[role] * (
                    F.binary_cross_entropy_with_logits(
                        outputs[role],
                        role_target,
                        pos_weight=role_pos_weights[role],
                    )
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            size = int(indices.shape[0])
            weighted_loss += float(loss.detach().item()) * size
            seen += size

        final_loss = weighted_loss / max(1, seen)
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(
                f"{name}: epoch={epoch:03d}/{epochs} "
                f"loss={final_loss:.6f}"
            )

    torch.save(
        {
            "name": name,
            "model_state_dict": model.state_dict(),
            "input_dim": int(train_x.shape[1]),
            "epochs": epochs,
            "seed": seed,
        },
        checkpoint_path,
    )

    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, validation_x.shape[0], 512):
            output = model(
                validation_x[start:start + 512].to(device)
            )
            batches.append(
                {
                    key: value.detach().cpu()
                    for key, value in output.items()
                }
            )

    logits = {
        key: torch.cat(
            [batch[key] for batch in batches],
            dim=0,
        )
        for key in batches[0]
    }

    prediction = {
        "graph": torch.sigmoid(logits["graph"]) >= 0.5,
        "count": logits["count"].argmax(dim=1),
        "roles": {
            role: torch.sigmoid(logits[role]) >= 0.5
            for role in ROLES
        },
    }

    details = {
        "final_train_loss": final_loss,
        "parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    return prediction, details


def graph_metrics(
    truth: torch.Tensor,
    prediction: torch.Tensor,
) -> dict[str, Any]:
    truth = truth.bool()
    prediction = prediction.bool()

    tp = int((truth & prediction).sum().item())
    tn = int((~truth & ~prediction).sum().item())
    fp = int((~truth & prediction).sum().item())
    fn = int((truth & ~prediction).sum().item())

    recall = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = (
        2 * precision * recall / max(1e-12, precision + recall)
    )

    return {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "balanced_accuracy": 0.5 * (recall + tnr),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / max(1, fp + tn),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def count_metrics(
    truth: torch.Tensor,
    prediction: torch.Tensor,
) -> dict[str, Any]:
    truth = truth.long()
    prediction = prediction.long()

    confusion = torch.zeros(3, 3, dtype=torch.long)
    for actual, predicted in zip(
        truth.tolist(),
        prediction.tolist(),
    ):
        confusion[actual, predicted] += 1

    f1_values = []
    for label in range(3):
        tp = int(confusion[label, label].item())
        fp = int(confusion[:, label].sum().item()) - tp
        fn = int(confusion[label, :].sum().item()) - tp

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1_values.append(
            2 * precision * recall
            / max(1e-12, precision + recall)
        )

    return {
        "accuracy": float(
            (truth == prediction).float().mean().item()
        ),
        "macro_f1": sum(f1_values) / 3.0,
        "confusion_matrix": confusion.tolist(),
    }


def role_metrics(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    attack: torch.Tensor,
) -> dict[str, Any]:
    active = attack.bool()
    truth = truth.bool()[active]
    prediction = prediction.bool()[active]

    tp = int((truth & prediction).sum().item())
    fp = int((~truth & prediction).sum().item())
    fn = int((truth & ~prediction).sum().item())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        2 * precision * recall
        / max(1e-12, precision + recall)
    )
    exact = float(
        (truth == prediction)
        .all(dim=1)
        .float()
        .mean()
        .item()
    )

    return {
        "node_f1": f1,
        "exact_set": exact,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate(
    validation: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    return {
        "graph": graph_metrics(
            validation["attack"],
            prediction["graph"],
        ),
        "count": count_metrics(
            validation["count"],
            prediction["count"],
        ),
        "roles": {
            role: role_metrics(
                validation["roles"][role],
                prediction["roles"][role],
                validation["attack"],
            )
            for role in ROLES
        },
    }


def paired_feature_equality(
    validation: dict[str, Any],
) -> dict[str, Any]:
    records = validation["run_records"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["pair_id"]].append(record)

    unequal_length_pairs = []
    unequal_window_count_pairs = []
    malformed_pairs = []

    for pair_id, members in grouped.items():
        modes = sorted(member["mode"] for member in members)
        if len(members) != 2 or modes != ["attack", "control"]:
            malformed_pairs.append(pair_id)
            continue
        if len({member["run_length"] for member in members}) != 1:
            unequal_length_pairs.append(pair_id)
        if len(
            {member["run_window_count"] for member in members}
        ) != 1:
            unequal_window_count_pairs.append(pair_id)

    return {
        "pair_count": len(grouped),
        "malformed_pairs": malformed_pairs,
        "unequal_length_pairs": unequal_length_pairs,
        "unequal_window_count_pairs": unequal_window_count_pairs,
        "run_length_and_window_position_are_pair_matched": (
            not malformed_pairs
            and not unequal_length_pairs
            and not unequal_window_count_pairs
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--b0-dir", type=Path, required=True)
    parser.add_argument("--r1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    b0_dir = args.b0_dir.expanduser().resolve()
    r1_dir = args.r1_dir.expanduser().resolve()
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
    hold_reasons: list[str] = []

    required = [
        b0_dir / "V5_P0_B0_SHORTCUT_AUDIT_SUITE_HOLD",
        b0_dir / "V5_P0_B0_SHORTCUT_AUDIT_SUITE.json",
        r1_dir / "V5_P0_B0_R1_IDENTIFIER_QUARANTINE_HOLD",
        r1_dir / "V5_P0_B0_R1_IDENTIFIER_QUARANTINE.json",
    ]
    for path in required:
        if not path.is_file():
            failures.append(f"required artifact missing: {path}")

    if failures:
        report = {
            "stage": "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT",
            "status": "FAIL",
            "failures": failures,
        }
        write_json(
            output_dir
            / "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT.json",
            report,
        )
        atomic_write(
            output_dir
            / "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_HOLD",
            "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_HOLD\n",
        )
        return 1

    print("collecting TRAIN labels and numeric provenance...")
    train = load_split(
        root=root,
        split="train",
        window=args.window,
        stride=args.stride,
    )

    print("collecting VALIDATION labels and numeric provenance...")
    validation = load_split(
        root=root,
        split="validation",
        window=args.window,
        stride=args.stride,
    )

    paired_check = paired_feature_equality(validation)
    if not paired_check[
        "run_length_and_window_position_are_pair_matched"
    ]:
        hold_reasons.append(
            "validation attack/control pairs are not matched for run "
            "length and window position"
        )

    feature_sets = build_features(train, validation)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    results = {}
    details = {}

    print("===== V5 P0-B0-R1B NUMERIC PROVENANCE AUDIT =====")
    print("device:", device)
    print("test_split_accessed: false")

    for offset, (
        name,
        (train_x, validation_x, feature_details),
    ) in enumerate(feature_sets.items()):
        print(f"\n===== {name} =====")

        prediction, training_details = train_probe(
            name=name,
            train_x=train_x,
            validation_x=validation_x,
            train=train,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed + offset,
            device=device,
            checkpoint_path=model_dir / f"{name}.pt",
        )

        results[name] = evaluate(
            validation=validation,
            prediction=prediction,
        )
        details[name] = {
            **feature_details,
            **training_details,
            "input_dim": int(train_x.shape[1]),
        }

    critical_variants = (
        "run_length_only",
        "run_window_count_only",
        "within_run_position_only",
        "all_numeric_provenance",
    )

    for name in critical_variants:
        metric = results[name]

        if metric["graph"]["balanced_accuracy"] >= 0.70:
            hold_reasons.append(
                f"{name} graph balanced accuracy too high: "
                f"{metric['graph']['balanced_accuracy']:.4f}"
            )

        if metric["count"]["macro_f1"] >= 0.55:
            hold_reasons.append(
                f"{name} count macro F1 too high: "
                f"{metric['count']['macro_f1']:.4f}"
            )

        if (
            metric["roles"]["source"]["exact_set"] >= 0.50
            or metric["roles"]["victim"]["exact_set"] >= 0.50
        ):
            hold_reasons.append(
                f"{name} source/victim exact localization too high"
            )

    serialization_metric = results["serialization_only"]
    if (
        serialization_metric["graph"]["balanced_accuracy"] >= 0.70
        or serialization_metric["count"]["macro_f1"] >= 0.55
        or serialization_metric["roles"]["source"]["exact_set"] >= 0.50
        or serialization_metric["roles"]["victim"]["exact_set"] >= 0.50
    ):
        warnings.append(
            "dataset/run serialization position is predictive; this field "
            "is non-physical and remains strictly quarantined"
        )

    passed = not failures and not hold_reasons

    rows = []
    for name, metric in results.items():
        rows.append(
            {
                "variant": name,
                "graph_balanced_accuracy": (
                    metric["graph"]["balanced_accuracy"]
                ),
                "graph_f1": metric["graph"]["f1"],
                "count_macro_f1": metric["count"]["macro_f1"],
                "source_node_f1_attack": (
                    metric["roles"]["source"]["node_f1"]
                ),
                "source_exact_attack": (
                    metric["roles"]["source"]["exact_set"]
                ),
                "victim_node_f1_attack": (
                    metric["roles"]["victim"]["node_f1"]
                ),
                "victim_exact_attack": (
                    metric["roles"]["victim"]["exact_set"]
                ),
            }
        )

    summary_path = (
        output_dir
        / "V5_P0_B0_R1B_NUMERIC_PROVENANCE_SUMMARY.csv"
    )
    write_csv(summary_path, rows)

    decision = {
        "status": "PASS" if passed else "HOLD",
        "classification": (
            "NUMERIC_PROVENANCE_NOT_A_VALIDATION_SHORTCUT"
            if passed
            else "NUMERIC_PROVENANCE_SHORTCUT_UNRESOLVED"
        ),
        "hold_reasons": hold_reasons,
        "warnings": warnings,
        "paired_validation_check": paired_check,
        "release_thresholds": {
            "critical_graph_balanced_accuracy_max_exclusive": 0.70,
            "critical_count_macro_f1_max_exclusive": 0.55,
            "critical_source_or_victim_exact_max_exclusive": 0.50,
        },
        "serialization_policy": (
            "serialization remains forbidden regardless of probe result"
        ),
        "next_stage": (
            "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE"
            if passed
            else "HOLD_REMEDIATE_B0_R1B"
        ),
    }

    report = {
        "stage": "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT",
        "status": "PASS" if passed else "HOLD",
        "train_window_count": int(train["attack"].shape[0]),
        "validation_window_count": int(
            validation["attack"].shape[0]
        ),
        "train_run_count": int(train["file_count"]),
        "validation_run_count": int(validation["file_count"]),
        "window": args.window,
        "stride": args.stride,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "results": results,
        "details": details,
        "decision": decision,
        "failures": failures,
        "warnings": warnings,
        "audit_boundary": {
            "traffic_x_values_read": False,
            "identifier_strings_used_as_probe_inputs": False,
            "mode_used_as_probe_input": False,
            "train_used_for_fitting": True,
            "validation_used_for_evaluation": True,
            "validation_checkpoint_selection": False,
            "test_directory_enumerated": False,
            "test_tensors_read": False,
            "test_performance_evaluated": False,
        },
        "provenance": {
            "b0_report_sha256": sha256_file(
                b0_dir
                / "V5_P0_B0_SHORTCUT_AUDIT_SUITE.json"
            ),
            "r1_report_sha256": sha256_file(
                r1_dir
                / "V5_P0_B0_R1_IDENTIFIER_QUARANTINE.json"
            ),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "next_stage": decision["next_stage"],
    }

    report_path = (
        output_dir
        / "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT.json"
    )
    decision_path = (
        output_dir
        / "V5_P0_B0_R1B_DECISION.json"
    )
    write_json(report_path, report)
    write_json(decision_path, decision)

    lock = {
        "status": (
            "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_PASS"
            if passed
            else "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_HOLD"
        ),
        "report_sha256": sha256_file(report_path),
        "decision_sha256": sha256_file(decision_path),
        "summary_csv_sha256": sha256_file(summary_path),
        "script_sha256": sha256_file(Path(__file__)),
        "test_split_accessed": False,
        "next_stage": decision["next_stage"],
    }

    write_json(
        output_dir
        / "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_LOCK.json",
        lock,
    )

    print("\n===== B0-R1B SUMMARY =====")
    for row in rows:
        print(
            f"{row['variant']}: "
            f"g_bal_acc={row['graph_balanced_accuracy']:.4f} "
            f"count_macro_f1={row['count_macro_f1']:.4f} "
            f"src_exact={row['source_exact_attack']:.4f} "
            f"victim_exact={row['victim_exact_attack']:.4f}"
        )

    print("\n===== B0-R1B DECISION =====")
    print(
        "validation_pair_count:",
        paired_check["pair_count"],
    )
    print(
        "validation_pair_matched_numeric_features:",
        paired_check[
            "run_length_and_window_position_are_pair_matched"
        ],
    )
    print("hold_reason_count:", len(hold_reasons))
    print("warning_count:", len(warnings))
    print("test_split_accessed: false")

    for reason in hold_reasons:
        print("HOLD:", reason)
    for warning in warnings:
        print("WARNING:", warning)

    if not passed:
        atomic_write(
            output_dir
            / "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_HOLD",
            "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_HOLD\n",
        )
        print("next_stage: HOLD_REMEDIATE_B0_R1B")
        print(
            "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_HOLD"
        )
        return 1

    atomic_write(
        output_dir
        / "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_PASS",
        "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_PASS\n",
    )
    print(
        "classification: "
        "NUMERIC_PROVENANCE_NOT_A_VALIDATION_SHORTCUT"
    )
    print("next_stage: V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE")
    print(
        "V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
