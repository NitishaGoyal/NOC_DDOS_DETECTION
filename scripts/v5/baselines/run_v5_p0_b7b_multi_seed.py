#!/usr/bin/env python3
"""V5 P0-B7B paired multi-seed confirmation for B3 versus B7A.

Fresh paired seeds: 7, 17, 27.
Only the random seed changes. Architecture, data contract, optimizer,
losses, threshold, epoch count, and final-epoch evaluation remain frozen.
TEST is never constructed, enumerated, read, or evaluated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SEEDS = (7, 17, 27)
ROLE_KEYS = {
    "source": "y_source",
    "transit": "y_transit",
    "victim": "y_victim",
    "path": "y_attack_path",
}
ROLES = tuple(ROLE_KEYS)


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def import_module(path: Path):
    spec = importlib.util.spec_from_file_location("v5_p0_b7a_b7b_frozen", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class B3Conv1DOnlyModel(nn.Module):
    """Exact B3 architecture with an evaluation-compatible interface."""

    def __init__(
        self,
        frozen_module,
        expected_edge_index: torch.Tensor,
        feature_count: int = 58,
        temporal_channels: int = 64,
        mask_count: int = 10,
        graph_hidden: int = 64,
        role_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "expected_edge_index",
            expected_edge_index.detach().cpu().long().contiguous(),
        )
        self.input_projection = nn.Conv1d(
            feature_count,
            temporal_channels,
            kernel_size=1,
        )
        self.temporal_blocks = nn.ModuleList(
            [
                frozen_module.CausalDepthwiseResidualBlock(
                    channels=temporal_channels,
                    kernel_size=3,
                    dilation=dilation,
                )
                for dilation in (1, 2, 4, 8)
            ]
        )
        self.node_post = nn.Sequential(
            nn.Linear(temporal_channels + mask_count, temporal_channels),
            nn.ReLU(),
        )
        self.role_heads = nn.ModuleDict(
            {
                role: nn.Sequential(
                    nn.Linear(temporal_channels, role_hidden),
                    nn.ReLU(),
                    nn.Linear(role_hidden, 1),
                )
                for role in ROLES
            }
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(temporal_channels * 2, graph_hidden),
            nn.ReLU(),
        )
        self.attack_head = nn.Linear(graph_hidden, 1)
        self.count_head = nn.Linear(graph_hidden, 3)

    def verify_edge_index(self, edge_index: torch.Tensor) -> None:
        if edge_index.ndim == 3:
            reference = edge_index[0]
            if not torch.equal(
                edge_index,
                reference.unsqueeze(0).expand_as(edge_index),
            ):
                raise ValueError("batch contains non-identical edge_index tensors")
        elif edge_index.ndim == 2:
            reference = edge_index
        else:
            raise ValueError(f"unexpected edge_index shape: {tuple(edge_index.shape)}")
        if not torch.equal(reference.long(), self.expected_edge_index):
            raise ValueError("runtime edge_index differs from audited topology")

    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
        edge_index: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        if x.ndim != 4 or x.shape[1:] != (16, 58, 32):
            raise ValueError(f"expected x [B,16,58,32], got {tuple(x.shape)}")
        if physical_port_mask.ndim != 3 or physical_port_mask.shape[1:] != (16, 10):
            raise ValueError(
                "expected physical_port_mask [B,16,10], got "
                f"{tuple(physical_port_mask.shape)}"
            )
        self.verify_edge_index(edge_index)
        batch_size, router_count, feature_count, window = x.shape
        temporal = x.reshape(batch_size * router_count, feature_count, window)
        temporal = F.relu(self.input_projection(temporal))
        for block in self.temporal_blocks:
            temporal = block(temporal)
        node_temporal = temporal[..., -1].reshape(batch_size, router_count, -1)
        mask = physical_port_mask.to(device=x.device, dtype=x.dtype)
        h_local = self.node_post(torch.cat([node_temporal, mask], dim=-1))
        graph_repr = torch.cat(
            [h_local.mean(dim=1), h_local.max(dim=1).values],
            dim=-1,
        )
        graph_repr = self.graph_encoder(graph_repr)
        outputs = {
            "attack": self.attack_head(graph_repr).squeeze(-1),
            "count": self.count_head(graph_repr),
        }
        for role in ROLES:
            outputs[role] = self.role_heads[role](h_local).squeeze(-1)
        if return_diagnostics:
            diagnostics = {
                "alpha": torch.tensor(0.0, device=x.device),
                "gate_mean_north": torch.tensor(0.0, device=x.device),
                "gate_mean_east": torch.tensor(0.0, device=x.device),
                "gate_mean_south": torch.tensor(0.0, device=x.device),
                "gate_mean_west": torch.tensor(0.0, device=x.device),
            }
            return outputs, diagnostics
        return outputs


def key_metrics(module, metrics: dict[str, Any]) -> dict[str, float]:
    return module.key_metrics(metrics)


def metric_row(module, model_name: str, seed: int, metrics: dict[str, Any]) -> dict[str, Any]:
    return {"model": model_name, "seed": seed, **key_metrics(module, metrics)}


def mean_std(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in rows]
        output[metric] = {
            "mean": float(statistics.mean(values)),
            "std_sample": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "values": values,
        }
    return output


def train_one(
    *,
    module,
    model_name: str,
    seed: int,
    train_dataset,
    validation_dataset,
    edge_index: torch.Tensor,
    train_stats: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    model_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    module.set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    if model_name == "B3":
        model = B3Conv1DOnlyModel(module, edge_index)
        expected_parameters = 43208
    elif model_name == "B7A":
        model = module.DirectionalGatedGraphDiagnosticModel(
            expected_edge_index=edge_index,
        )
        expected_parameters = 50325
    else:
        raise ValueError(model_name)

    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != expected_parameters:
        raise RuntimeError(
            f"{model_name} seed {seed}: parameter_count={parameter_count}, "
            f"expected {expected_parameters}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    pos_weights, pos_weight_values = module.build_positive_weights(
        train_stats,
        device=device,
    )

    print(f"\n===== B7B {model_name} SEED {seed} =====")
    print("device:", device)
    print("parameter_count:", parameter_count)
    print("validation_during_training: false")
    print("test_split_accessed: false")

    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_samples = 0
        component_totals = {key: 0.0 for key in ("attack", "count", *ROLES)}

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                x=batch["x"].to(device=device, dtype=torch.float32),
                physical_port_mask=batch["physical_port_mask"].to(device=device),
                edge_index=batch["edge_index"].to(device=device, dtype=torch.long),
            )
            loss, components = module.compute_loss(
                outputs=outputs,
                batch=batch,
                pos_weights=pos_weights,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{model_name} seed {seed}: non-finite loss at epoch {epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            current_batch = int(batch["x"].shape[0])
            epoch_total += float(loss.detach().item()) * current_batch
            epoch_samples += current_batch
            for key, value in components.items():
                component_totals[key] += value * current_batch

        mean_loss = epoch_total / max(1, epoch_samples)
        row = {
            "epoch": epoch,
            "train_loss": mean_loss,
            **{
                f"train_loss_{key}": value / max(1, epoch_samples)
                for key, value in component_totals.items()
            },
        }
        if model_name == "B7A":
            row["alpha"] = float(
                torch.sigmoid(model.alpha_logit).detach().cpu().item()
            )
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            suffix = ""
            if model_name == "B7A":
                suffix = f" alpha={row['alpha']:.6f}"
            print(
                f"model={model_name} seed={seed:03d} "
                f"epoch={epoch:03d}/{epochs} train_loss={mean_loss:.6f}{suffix}"
            )

    train_metrics, _, train_diagnostics = module.evaluate(
        model=model,
        loader=train_eval_loader,
        device=device,
    )
    validation_metrics, validation_predictions, validation_diagnostics = module.evaluate(
        model=model,
        loader=validation_loader,
        device=device,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    history_path = output_dir / "HISTORY.csv"
    metrics_path = output_dir / "METRICS.csv"
    report_path = output_dir / "REPORT.json"
    checkpoint_path = model_dir / "final_model.pt"
    predictions_path = model_dir / "validation_predictions_and_targets.pt"

    write_csv(history_path, history)
    write_csv(
        metrics_path,
        module.flatten_metric_rows("train", train_metrics)
        + module.flatten_metric_rows("validation", validation_metrics),
    )

    alpha_final = 0.0
    if model_name == "B7A":
        alpha_final = float(torch.sigmoid(model.alpha_logit).detach().cpu().item())

    torch.save(
        {
            "stage": "V5_P0_B7B_DIRECTIONAL_GATED_GRAPH_MULTI_SEED_CONFIRMATION",
            "model_name": model_name,
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_epoch": epochs,
            "feature_variant": "PRIMARY58",
            "window": 32,
            "stride": 8,
            "binary_threshold": 0.5,
            "parameter_count": parameter_count,
            "alpha_final": alpha_final,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "validation_graph_diagnostics": validation_diagnostics,
            "test_split_accessed": False,
        },
        checkpoint_path,
    )
    torch.save(
        {
            "stage": "V5_P0_B7B_DIRECTIONAL_GATED_GRAPH_MULTI_SEED_CONFIRMATION",
            "model_name": model_name,
            "seed": seed,
            "split": "validation",
            "binary_threshold": 0.5,
            "predictions_and_targets": validation_predictions,
            "metadata_included": False,
        },
        predictions_path,
    )

    report = {
        "stage": "V5_P0_B7B_DIRECTIONAL_GATED_GRAPH_MULTI_SEED_CONFIRMATION",
        "status": "COMPLETE",
        "model_name": model_name,
        "seed": seed,
        "parameter_count": parameter_count,
        "protocol": {
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_clip_norm": 5.0,
            "binary_threshold": 0.5,
            "early_stopping": False,
            "validation_checkpoint_selection": False,
            "validation_evaluated_only_after_training": True,
            "threshold_search": False,
            "test_split_accessed": False,
        },
        "positive_weights": pos_weight_values,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "train_graph_diagnostics": train_diagnostics,
        "validation_graph_diagnostics": validation_diagnostics,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "validation_predictions": str(predictions_path),
            "validation_predictions_sha256": sha256_file(predictions_path),
            "history_csv": str(history_path),
            "history_csv_sha256": sha256_file(history_path),
            "metrics_csv": str(metrics_path),
            "metrics_csv_sha256": sha256_file(metrics_path),
        },
        "audit_boundary": {
            "train_used_for_fitting": True,
            "validation_used_for_final_evaluation": True,
            "validation_used_for_checkpoint_selection": False,
            "validation_used_for_threshold_selection": False,
            "test_directory_enumerated": False,
            "test_dataset_constructed": False,
            "test_tensors_read": False,
            "test_performance_evaluated": False,
        },
    }
    write_json(report_path, report)

    graph = validation_metrics["graph"]
    print(
        f"B7B_RESULT model={model_name} seed={seed} "
        f"g_bal_acc={graph['balanced_accuracy']:.4f} "
        f"g_recall={graph['recall']:.4f} "
        f"g_f1={graph['f1']:.4f} g_fpr={graph['fpr']:.4f} "
        f"graph_fn={graph['fn']} "
        f"count_macro_f1={validation_metrics['count']['macro_f1']:.4f} "
        f"source_f1={validation_metrics['roles']['source']['attack_windows']['node_f1']:.4f} "
        f"victim_f1={validation_metrics['roles']['victim']['attack_windows']['node_f1']:.4f} "
        f"all_exact={validation_metrics['all_tasks_exact']:.4f}"
    )

    return {
        "model": model_name,
        "seed": seed,
        "parameter_count": parameter_count,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_graph_diagnostics": validation_diagnostics,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--a2-1-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--r1c-dir", type=Path, required=True)
    parser.add_argument("--b1-dir", type=Path, required=True)
    parser.add_argument("--b2-dir", type=Path, required=True)
    parser.add_argument("--b3-dir", type=Path, required=True)
    parser.add_argument("--b4-dir", type=Path, required=True)
    parser.add_argument("--b5-dir", type=Path, required=True)
    parser.add_argument("--b6-dir", type=Path, required=True)
    parser.add_argument("--b7a-dir", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--frozen-module", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    a2_dir = args.a2_dir.expanduser().resolve()
    a2_1_dir = args.a2_1_dir.expanduser().resolve()
    a3_dir = args.a3_dir.expanduser().resolve()
    r1c_dir = args.r1c_dir.expanduser().resolve()
    wrapper = args.wrapper.expanduser().resolve()
    frozen_module_path = args.frozen_module.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    b7a_dir = args.b7a_dir.expanduser().resolve()

    if output_dir.exists() or model_dir.exists():
        print("STOP: B7B output/model directory already exists", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    module = import_module(frozen_module_path)
    baseline_dirs = {
        "b1": args.b1_dir.expanduser().resolve(),
        "b2": args.b2_dir.expanduser().resolve(),
        "b3": args.b3_dir.expanduser().resolve(),
        "b4": args.b4_dir.expanduser().resolve(),
        "b5": args.b5_dir.expanduser().resolve(),
        "b6": args.b6_dir.expanduser().resolve(),
    }
    failures, warnings, provenance, _ = module.verify_prerequisites(
        a2_dir=a2_dir,
        a3_dir=a3_dir,
        r1c_dir=r1c_dir,
        baseline_dirs=baseline_dirs,
        wrapper_path=wrapper,
    )

    b7a_report_path = (
        b7a_dir / "V5_P0_B7A_DIRECTIONAL_GATED_GRAPH_DIAGNOSTIC.json"
    )
    b7a_lock_path = (
        b7a_dir / "V5_P0_B7A_DIRECTIONAL_GATED_GRAPH_DIAGNOSTIC_LOCK.json"
    )
    b7a_marker = (
        b7a_dir / "V5_P0_B7A_DIRECTIONAL_GATED_GRAPH_DIAGNOSTIC_COMPLETE"
    )
    for path in (b7a_report_path, b7a_lock_path, b7a_marker):
        if not path.is_file():
            failures.append(f"missing B7A trigger artifact: {path}")
    if not failures:
        b7a_report = load_json(b7a_report_path)
        b7a_lock = load_json(b7a_lock_path)
        if b7a_report.get("status") != "COMPLETE":
            failures.append("B7A trigger report is not COMPLETE")
        if b7a_report.get("candidate_decision") != "PROMOTE_TO_MULTI_SEED_CONFIRMATION":
            failures.append("B7A trigger decision did not promote to multi-seed confirmation")
        if b7a_report.get("success_gate", {}).get("passed") is not True:
            failures.append("B7A trigger success gate did not pass")
        if b7a_lock.get("report_sha256") != sha256_file(b7a_report_path):
            failures.append("B7A trigger report SHA does not match lock")
        if b7a_report.get("protocol", {}).get("test_split_accessed") is not False:
            failures.append("B7A trigger indicates test access")
        provenance.update(
            {
                "b7a_trigger_report_sha256": sha256_file(b7a_report_path),
                "b7a_trigger_lock_sha256": sha256_file(b7a_lock_path),
                "b7a_trigger_script_sha256": b7a_report.get("provenance", {}).get(
                    "script_sha256"
                ),
            }
        )

    if failures:
        write_json(
            output_dir / "V5_P0_B7B_HOLD.json",
            {
                "stage": "V5_P0_B7B_DIRECTIONAL_GATED_GRAPH_MULTI_SEED_CONFIRMATION",
                "status": "HOLD",
                "failures": failures,
                "warnings": warnings,
                "test_split_accessed": False,
            },
        )
        atomic_write(output_dir / "V5_P0_B7B_HOLD", "V5_P0_B7B_HOLD\n")
        print("V5_P0_B7B_HOLD")
        return 1

    ContractDataset = module.import_contract_dataset(wrapper)
    train_dataset = ContractDataset(
        root=root,
        split="train",
        contract_dir=a2_dir,
        mask_audit_dir=a2_1_dir,
        window=32,
        stride=8,
        active_only=False,
        feature_variant="PRIMARY58",
    )
    validation_dataset = ContractDataset(
        root=root,
        split="validation",
        contract_dir=a2_dir,
        mask_audit_dir=a2_1_dir,
        window=32,
        stride=8,
        active_only=False,
        feature_variant="PRIMARY58",
    )
    sample = train_dataset[0]
    if tuple(sample["x"].shape) != (16, 58, 32):
        raise RuntimeError(f"unexpected x shape: {tuple(sample['x'].shape)}")
    if sample["physical_port_mask"].dtype != torch.bool:
        raise RuntimeError("physical_port_mask is not Boolean")
    module.validate_mesh_edge_index(sample["edge_index"])

    stats_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    train_stats = module.target_statistics(stats_loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("===== V5 P0-B7B PAIRED MULTI-SEED CONFIRMATION =====")
    print("seeds:", list(SEEDS))
    print("device:", device)
    print("train_window_count:", len(train_dataset))
    print("validation_window_count:", len(validation_dataset))
    print("paired_models: [B3, B7A]")
    print("architecture_tuning_during_b7b: false")
    print("validation_during_training: false")
    print("test_split_accessed: false")
    print("corrected_gate_diagnostics: true")

    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_root = output_dir / f"seed_{seed:03d}"
        seed_models = model_dir / f"seed_{seed:03d}"
        for model_name in ("B3", "B7A"):
            result = train_one(
                module=module,
                model_name=model_name,
                seed=seed,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                edge_index=sample["edge_index"],
                train_stats=train_stats,
                device=device,
                output_dir=seed_root / model_name.lower(),
                model_dir=seed_models / model_name.lower(),
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            results.append(result)

    metric_names = list(key_metrics(module, results[0]["validation_metrics"]).keys())
    metric_rows = [
        metric_row(module, result["model"], result["seed"], result["validation_metrics"])
        for result in results
    ]
    b3_rows = [row for row in metric_rows if row["model"] == "B3"]
    b7a_rows = [row for row in metric_rows if row["model"] == "B7A"]
    b3_by_seed = {int(row["seed"]): row for row in b3_rows}
    b7a_by_seed = {int(row["seed"]): row for row in b7a_rows}

    paired_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        row = {"seed": seed}
        for metric in metric_names:
            row[f"b3_{metric}"] = b3_by_seed[seed][metric]
            row[f"b7a_{metric}"] = b7a_by_seed[seed][metric]
            row[f"delta_{metric}"] = (
                float(b7a_by_seed[seed][metric])
                - float(b3_by_seed[seed][metric])
            )
        paired_rows.append(row)

    aggregate = {
        "B3": mean_std(b3_rows, metric_names),
        "B7A": mean_std(b7a_rows, metric_names),
    }
    mean_delta = {
        metric: aggregate["B7A"][metric]["mean"] - aggregate["B3"][metric]["mean"]
        for metric in metric_names
    }

    mean_requirements = {
        "graph_f1": (
            aggregate["B7A"]["graph_f1"]["mean"]
            >= aggregate["B3"]["graph_f1"]["mean"]
        ),
        "graph_recall": (
            aggregate["B7A"]["graph_recall"]["mean"]
            >= aggregate["B3"]["graph_recall"]["mean"] - 0.01
        ),
        "graph_fpr": (
            aggregate["B7A"]["graph_fpr"]["mean"]
            <= aggregate["B3"]["graph_fpr"]["mean"] + 0.02
        ),
        "count_macro_f1": (
            aggregate["B7A"]["count_macro_f1"]["mean"]
            >= aggregate["B3"]["count_macro_f1"]["mean"] - 0.02
        ),
        "source_node_f1_attack": (
            aggregate["B7A"]["source_node_f1_attack"]["mean"]
            >= aggregate["B3"]["source_node_f1_attack"]["mean"] - 0.02
        ),
        "victim_node_f1_attack": (
            aggregate["B7A"]["victim_node_f1_attack"]["mean"]
            >= aggregate["B3"]["victim_node_f1_attack"]["mean"] - 0.02
        ),
        "all_tasks_exact": (
            aggregate["B7A"]["all_tasks_exact"]["mean"]
            >= aggregate["B3"]["all_tasks_exact"]["mean"] + 0.03
        ),
    }

    all_exact_positive = sum(
        1 for row in paired_rows if row["delta_all_tasks_exact"] > 0.0
    )
    graph_f1_positive = sum(
        1 for row in paired_rows if row["delta_graph_f1"] > 0.0
    )
    no_large_source_drop = all(
        row["delta_source_node_f1_attack"] >= -0.05 for row in paired_rows
    )
    no_large_victim_drop = all(
        row["delta_victim_node_f1_attack"] >= -0.05 for row in paired_rows
    )
    no_large_recall_drop = all(
        row["delta_graph_recall"] >= -0.03 for row in paired_rows
    )
    stability_requirements = {
        "all_exact_improves_in_at_least_2_of_3": all_exact_positive >= 2,
        "graph_f1_improves_in_at_least_2_of_3": graph_f1_positive >= 2,
        "no_seed_source_drop_gt_0_05": no_large_source_drop,
        "no_seed_victim_drop_gt_0_05": no_large_victim_drop,
        "no_seed_graph_recall_drop_gt_0_03": no_large_recall_drop,
    }

    promotion_passed = all(mean_requirements.values()) and all(
        stability_requirements.values()
    )
    failed_requirements = [
        f"mean::{name}" for name, passed in mean_requirements.items() if not passed
    ] + [
        f"stability::{name}"
        for name, passed in stability_requirements.items()
        if not passed
    ]

    if promotion_passed:
        candidate_decision = "PROMOTE_B7A_TO_P0_ARCHITECTURE_FREEZE"
        next_stage = "V5_P0_B8_FREEZE_DIRECTIONAL_GATED_GRAPH_ARCHITECTURE"
    else:
        candidate_decision = "REJECT_B7A_AND_FREEZE_B3"
        next_stage = "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE"

    metrics_csv = output_dir / "V5_P0_B7B_VALIDATION_METRICS_BY_SEED.csv"
    paired_csv = output_dir / "V5_P0_B7B_PAIRED_DELTAS_BY_SEED.csv"
    report_path = output_dir / "V5_P0_B7B_DIRECTIONAL_GATED_GRAPH_MULTI_SEED_CONFIRMATION.json"
    write_csv(metrics_csv, metric_rows)
    write_csv(paired_csv, paired_rows)

    report = {
        "stage": "V5_P0_B7B_DIRECTIONAL_GATED_GRAPH_MULTI_SEED_CONFIRMATION",
        "status": "COMPLETE",
        "candidate_decision": candidate_decision,
        "promotion_passed": promotion_passed,
        "failed_requirements": failed_requirements,
        "fresh_paired_seeds": list(SEEDS),
        "models": {
            "B3": "exact causal Conv1D-only comparator",
            "B7A": "frozen directional gated residual graph diagnostic",
        },
        "gate_diagnostic_patch": {
            "original_seed_101_run_preserved": True,
            "training_or_predictions_affected": False,
            "change": (
                "expanded the valid-direction mask across the batch before "
                "computing post-sigmoid gate means"
            ),
            "corrected_values_must_be_in_0_1": True,
            "frozen_module_sha256": sha256_file(frozen_module_path),
        },
        "protocol": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": 5.0,
            "binary_threshold": 0.5,
            "early_stopping": False,
            "validation_checkpoint_selection": False,
            "validation_evaluated_only_after_training": True,
            "threshold_search": False,
            "architecture_tuning_during_b7b": False,
            "test_split_accessed": False,
        },
        "aggregate_validation_metrics": aggregate,
        "mean_b7a_minus_b3": mean_delta,
        "paired_seed_results": paired_rows,
        "promotion_rule": {
            "mean_requirements": mean_requirements,
            "stability_requirements": stability_requirements,
            "all_requirements_must_pass": True,
        },
        "run_artifacts": results,
        "artifacts": {
            "validation_metrics_csv": str(metrics_csv),
            "validation_metrics_csv_sha256": sha256_file(metrics_csv),
            "paired_deltas_csv": str(paired_csv),
            "paired_deltas_csv_sha256": sha256_file(paired_csv),
        },
        "warnings": warnings,
        "failures": [],
        "audit_boundary": {
            "train_used_for_fitting": True,
            "validation_used_for_final_evaluation": True,
            "validation_used_for_checkpoint_selection": False,
            "validation_used_for_threshold_selection": False,
            "test_directory_enumerated": False,
            "test_dataset_constructed": False,
            "test_tensors_read": False,
            "test_performance_evaluated": False,
        },
        "provenance": {
            **provenance,
            "b7b_script_sha256": sha256_file(Path(__file__)),
            "contract_wrapper_sha256": sha256_file(wrapper),
        },
        "next_stage": next_stage,
    }
    write_json(report_path, report)

    lock = {
        "status": "V5_P0_B7B_DIRECTIONAL_GATED_GRAPH_MULTI_SEED_CONFIRMATION_COMPLETE",
        "candidate_decision": candidate_decision,
        "promotion_passed": promotion_passed,
        "report_sha256": sha256_file(report_path),
        "validation_metrics_csv_sha256": sha256_file(metrics_csv),
        "paired_deltas_csv_sha256": sha256_file(paired_csv),
        "b7b_script_sha256": sha256_file(Path(__file__)),
        "frozen_module_sha256": sha256_file(frozen_module_path),
        "test_split_accessed": False,
        "next_stage": next_stage,
    }
    write_json(
        output_dir / "V5_P0_B7B_DIRECTIONAL_GATED_GRAPH_MULTI_SEED_CONFIRMATION_LOCK.json",
        lock,
    )
    atomic_write(
        output_dir / "V5_P0_B7B_DIRECTIONAL_GATED_GRAPH_MULTI_SEED_CONFIRMATION_COMPLETE",
        "V5_P0_B7B_DIRECTIONAL_GATED_GRAPH_MULTI_SEED_CONFIRMATION_COMPLETE\n",
    )

    print("\n===== B7B FINAL AGGREGATE =====")
    for metric in (
        "graph_balanced_accuracy",
        "graph_recall",
        "graph_f1",
        "graph_fpr",
        "count_macro_f1",
        "source_node_f1_attack",
        "victim_node_f1_attack",
        "all_tasks_exact",
    ):
        print(
            f"{metric}: "
            f"B3={aggregate['B3'][metric]['mean']:.4f}±{aggregate['B3'][metric]['std_sample']:.4f} "
            f"B7A={aggregate['B7A'][metric]['mean']:.4f}±{aggregate['B7A'][metric]['std_sample']:.4f} "
            f"delta={mean_delta[metric]:+.4f}"
        )
    print("promotion_passed:", str(promotion_passed).lower())
    print("failed_requirements:", failed_requirements)
    print("candidate_decision:", candidate_decision)
    print("test_split_accessed: false")
    print("next_stage:", next_stage)
    print("V5_P0_B7B_DIRECTIONAL_GATED_GRAPH_MULTI_SEED_CONFIRMATION_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
