#!/usr/bin/env python3
"""V5 P0-B4 GCN-only, parameter-matched against B2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROLE_KEYS = {
    "source": "y_source",
    "transit": "y_transit",
    "victim": "y_victim",
    "path": "y_attack_path",
}
ROLES = tuple(ROLE_KEYS)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    atomic_write(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def import_dataset(wrapper: Path):
    spec = importlib.util.spec_from_file_location("v5_p0_contract_loader_b4", wrapper)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import wrapper: {wrapper}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, "V5P0ContractWindowDataset", None)
    if cls is None:
        raise AttributeError("wrapper lacks V5P0ContractWindowDataset")
    return cls


def verify_report_lock(report: Path, lock: Path, stage: str) -> list[str]:
    failures: list[str] = []
    if not report.is_file():
        failures.append(f"missing {stage} report: {report}")
        return failures
    if not lock.is_file():
        failures.append(f"missing {stage} lock: {lock}")
        return failures
    lock_obj = load_json(lock)
    if lock_obj.get("report_sha256") != sha256_file(report):
        failures.append(f"{stage} report SHA mismatch")
    return failures


def verify_prerequisites(a2: Path, a3: Path, r1c: Path, b1: Path, b2: Path, b3: Path, wrapper: Path):
    failures: list[str] = []
    required_markers = [
        a2 / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_PASS",
        a3 / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_PASS",
        r1c / "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_PASS",
        b1 / "V5_P0_B1_STATIC_FINAL_EPOCH_MLP_COMPLETE",
        b2 / "V5_P0_B2_TEMPORAL_MEAN_POOLING_COMPLETE",
        b3 / "V5_P0_B3_CONV1D_ONLY_COMPLETE",
        wrapper,
    ]
    for path in required_markers:
        if not path.is_file():
            failures.append(f"missing prerequisite: {path}")

    reports = {
        "a2": a2 / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT.json",
        "a3": a3 / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE.json",
        "r1c": r1c / "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE.json",
        "b1": b1 / "V5_P0_B1_STATIC_FINAL_EPOCH_MLP.json",
        "b2": b2 / "V5_P0_B2_TEMPORAL_MEAN_POOLING.json",
        "b3": b3 / "V5_P0_B3_CONV1D_ONLY.json",
    }
    locks = {
        "a3": a3 / "V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_LOCK.json",
        "r1c": r1c / "V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_LOCK.json",
        "b1": b1 / "V5_P0_B1_STATIC_FINAL_EPOCH_MLP_LOCK.json",
        "b2": b2 / "V5_P0_B2_TEMPORAL_MEAN_POOLING_LOCK.json",
        "b3": b3 / "V5_P0_B3_CONV1D_ONLY_LOCK.json",
    }
    for name, lock in locks.items():
        failures += verify_report_lock(reports[name], lock, name.upper())

    loaded: dict[str, Any] = {}
    if not failures:
        loaded = {name: load_json(path) for name, path in reports.items()}
        expected_status = {
            "a2": "PASS",
            "a3": "PASS",
            "r1c": "PASS",
            "b1": "COMPLETE",
            "b2": "COMPLETE",
            "b3": "COMPLETE",
        }
        for name, expected in expected_status.items():
            if loaded[name].get("status") != expected:
                failures.append(f"{name.upper()} status is not {expected}")

        a2_contract = loaded["a2"].get("loader_contract", {})
        if a2_contract.get("primary_feature_variant") != "PRIMARY58":
            failures.append("A2 primary feature variant is not PRIMARY58")
        if a2_contract.get("metadata_concatenated_to_x") is not False:
            failures.append("A2 does not forbid metadata concatenation")
        if loaded["a3"].get("feature_variant") != "PRIMARY58":
            failures.append("A3 did not validate PRIMARY58")
        if loaded["b3"].get("next_stage") != "V5_P0_B4_GCN_ONLY":
            failures.append("B3 did not authorize B4")
        if loaded["b2"].get("model", {}).get("parameter_count") != 25544:
            failures.append("B2 parameter count is not 25,544")

    provenance = {f"{name}_report_sha256": sha256_file(path) for name, path in reports.items() if path.is_file()}
    provenance["wrapper_sha256"] = sha256_file(wrapper) if wrapper.is_file() else None
    return failures, loaded, provenance


def normalized_adjacency(edge_index: torch.Tensor, node_count: int = 16) -> torch.Tensor:
    edge_index = edge_index.detach().cpu().long()
    if edge_index.ndim != 2 or tuple(edge_index.shape)[0] != 2:
        raise ValueError(f"edge_index must be [2,E], got {tuple(edge_index.shape)}")
    src, dst = edge_index[0], edge_index[1]
    if int(src.min()) < 0 or int(dst.min()) < 0 or int(src.max()) >= node_count or int(dst.max()) >= node_count:
        raise ValueError("edge_index node index out of range")
    a = torch.zeros(node_count, node_count, dtype=torch.float32)
    a[dst, src] = 1.0
    a.fill_diagonal_(1.0)
    deg = a.sum(dim=1).clamp_min(1.0)
    inv = deg.pow(-0.5)
    return inv[:, None] * a * inv[None, :]


class DenseGCNLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        support = x @ self.weight
        return torch.einsum("ij,bjf->bif", adjacency, support) + self.bias


class GCNOnlyModel(nn.Module):
    def __init__(self, edge_index: torch.Tensor) -> None:
        super().__init__()
        edge_index = edge_index.detach().cpu().long().contiguous()
        self.register_buffer("expected_edge_index", edge_index)
        self.register_buffer("adjacency", normalized_adjacency(edge_index))
        self.input_projection = nn.Linear(68, 64)
        self.gcn = DenseGCNLayer(64)
        self.role_heads = nn.ModuleDict({
            role: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
            for role in ROLES
        })
        self.graph_encoder = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.attack_head = nn.Linear(64, 1)
        self.count_head = nn.Linear(64, 3)

    def _verify_edge_index(self, edge_index: torch.Tensor) -> None:
        if edge_index.ndim == 3:
            ref = edge_index[0]
            if not torch.equal(edge_index, ref.unsqueeze(0).expand_as(edge_index)):
                raise ValueError("non-identical edge_index tensors within batch")
        elif edge_index.ndim == 2:
            ref = edge_index
        else:
            raise ValueError(f"unexpected edge_index shape: {tuple(edge_index.shape)}")
        if not torch.equal(ref.long(), self.expected_edge_index):
            raise ValueError("runtime edge_index differs from audited topology")

    def forward(self, x: torch.Tensor, physical_port_mask: torch.Tensor, edge_index: torch.Tensor):
        if x.ndim != 4 or tuple(x.shape[1:]) != (16, 58, 32):
            raise ValueError(f"expected x [B,16,58,32], got {tuple(x.shape)}")
        if physical_port_mask.ndim != 3 or tuple(physical_port_mask.shape[1:]) != (16, 10):
            raise ValueError(f"expected mask [B,16,10], got {tuple(physical_port_mask.shape)}")
        self._verify_edge_index(edge_index)
        temporal_mean = x.mean(dim=-1)
        mask = physical_port_mask.to(device=x.device, dtype=x.dtype)
        h = F.relu(self.input_projection(torch.cat([temporal_mean, mask], dim=-1)))
        h = F.relu(self.gcn(h, self.adjacency.to(device=x.device, dtype=x.dtype)))
        graph = self.graph_encoder(torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1))
        out = {"attack": self.attack_head(graph).squeeze(-1), "count": self.count_head(graph)}
        for role in ROLES:
            out[role] = self.role_heads[role](h).squeeze(-1)
        return out


def target_statistics(loader: DataLoader) -> dict[str, Any]:
    windows = graph_pos = 0
    count_counts = torch.zeros(3, dtype=torch.long)
    role_pos = {r: 0 for r in ROLES}
    role_total = {r: 0 for r in ROLES}
    for batch in loader:
        y = batch["y_attack"].bool()
        windows += int(y.numel())
        graph_pos += int(y.sum())
        count_counts += torch.bincount(batch["y_attacker_count"].long(), minlength=3)
        for role, key in ROLE_KEYS.items():
            t = batch[key].bool()
            role_pos[role] += int(t.sum())
            role_total[role] += int(t.numel())
    return {
        "window_count": windows,
        "graph_positive_windows": graph_pos,
        "graph_negative_windows": windows - graph_pos,
        "count_class_counts": count_counts.tolist(),
        "role_positive_entries": role_pos,
        "role_total_entries": role_total,
    }


def positive_weight(positives: int, total: int, cap: float = 20.0) -> float:
    if positives <= 0:
        return 1.0
    return min(cap, (total - positives) / positives)


def build_pos_weights(stats: dict[str, Any], device: torch.device):
    values = {"attack": positive_weight(stats["graph_positive_windows"], stats["window_count"])}
    for role in ROLES:
        values[role] = positive_weight(stats["role_positive_entries"][role], stats["role_total_entries"][role])
    tensors = {k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in values.items()}
    return tensors, values


def compute_loss(outputs, batch, pos_weights):
    device = outputs["attack"].device
    losses = {
        "attack": F.binary_cross_entropy_with_logits(outputs["attack"], batch["y_attack"].to(device=device, dtype=torch.float32), pos_weight=pos_weights["attack"]),
        "count": F.cross_entropy(outputs["count"], batch["y_attacker_count"].to(device=device, dtype=torch.long)),
    }
    for role, key in ROLE_KEYS.items():
        losses[role] = F.binary_cross_entropy_with_logits(outputs[role], batch[key].to(device=device, dtype=torch.float32), pos_weight=pos_weights[role])
    total = losses["attack"] + 0.5 * losses["count"] + losses["source"] + 0.5 * losses["transit"] + 0.75 * losses["victim"] + 0.5 * losses["path"]
    return total, {k: float(v.detach()) for k, v in losses.items()}


def binary_metrics(truth: torch.Tensor, pred: torch.Tensor) -> dict[str, Any]:
    truth, pred = truth.bool(), pred.bool()
    tp = int((truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "balanced_accuracy": 0.5 * (recall + tnr),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / max(1, fp + tn),
        "tnr": tnr,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "support_negative": tn + fp,
        "support_positive": tp + fn,
    }


def multiclass_metrics(truth: torch.Tensor, pred: torch.Tensor, classes: int = 3) -> dict[str, Any]:
    truth, pred = truth.long(), pred.long()
    cm = torch.zeros(classes, classes, dtype=torch.long)
    for a, p in zip(truth.tolist(), pred.tolist()):
        cm[a, p] += 1
    per_class, f1s = {}, []
    for c in range(classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        support = int(cm[c, :].sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        f1s.append(f1)
        per_class[str(c)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {
        "accuracy": float((truth == pred).float().mean()),
        "macro_f1": sum(f1s) / classes,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def node_metrics(truth: torch.Tensor, pred: torch.Tensor) -> dict[str, Any]:
    truth, pred = truth.bool(), pred.bool()
    tp = int((truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "node_accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "node_precision": precision,
        "node_recall": recall,
        "node_f1": f1,
        "exact_set": float((truth == pred).all(dim=1).float().mean()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "positive_entries": int(truth.sum()),
        "window_count": int(truth.shape[0]),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    acc: dict[str, list[torch.Tensor]] = {
        "attack_truth": [], "attack_prediction": [], "attack_probability": [],
        "count_truth": [], "count_prediction": [],
    }
    for role in ROLES:
        acc[f"{role}_truth"] = []
        acc[f"{role}_prediction"] = []
        acc[f"{role}_probability"] = []
    for batch in loader:
        out = model(
            batch["x"].to(device=device, dtype=torch.float32),
            batch["physical_port_mask"].to(device=device),
            batch["edge_index"].to(device=device, dtype=torch.long),
        )
        ap = torch.sigmoid(out["attack"])
        acc["attack_truth"].append(batch["y_attack"].cpu().bool())
        acc["attack_probability"].append(ap.cpu())
        acc["attack_prediction"].append((ap >= 0.5).cpu())
        acc["count_truth"].append(batch["y_attacker_count"].cpu().long())
        acc["count_prediction"].append(out["count"].argmax(dim=-1).cpu())
        for role, key in ROLE_KEYS.items():
            p = torch.sigmoid(out[role])
            acc[f"{role}_truth"].append(batch[key].cpu().bool())
            acc[f"{role}_probability"].append(p.cpu())
            acc[f"{role}_prediction"].append((p >= 0.5).cpu())
    combined = {k: torch.cat(v, dim=0) for k, v in acc.items()}
    attack_truth = combined["attack_truth"]
    metrics = {
        "graph": binary_metrics(attack_truth, combined["attack_prediction"]),
        "count": multiclass_metrics(combined["count_truth"], combined["count_prediction"]),
        "roles": {},
    }
    all_exact = (attack_truth == combined["attack_prediction"]) & (combined["count_truth"] == combined["count_prediction"])
    active = attack_truth.bool()
    for role in ROLES:
        truth, pred = combined[f"{role}_truth"], combined[f"{role}_prediction"]
        metrics["roles"][role] = {
            "all_windows": node_metrics(truth, pred),
            "attack_windows": node_metrics(truth[active], pred[active]),
        }
        all_exact &= (truth == pred).all(dim=1)
    metrics["all_tasks_exact"] = float(all_exact.float().mean())
    metrics["window_count"] = int(attack_truth.shape[0])
    return metrics, combined


def key_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "graph_balanced_accuracy": metrics["graph"]["balanced_accuracy"],
        "graph_f1": metrics["graph"]["f1"],
        "graph_fpr": metrics["graph"]["fpr"],
        "count_macro_f1": metrics["count"]["macro_f1"],
        "source_node_f1_attack": metrics["roles"]["source"]["attack_windows"]["node_f1"],
        "transit_node_f1_attack": metrics["roles"]["transit"]["attack_windows"]["node_f1"],
        "victim_node_f1_attack": metrics["roles"]["victim"]["attack_windows"]["node_f1"],
        "path_node_f1_attack": metrics["roles"]["path"]["attack_windows"]["node_f1"],
        "all_tasks_exact": metrics["all_tasks_exact"],
    }


def metric_delta(current: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    a, b = key_metrics(current), key_metrics(reference)
    return {k: a[k] - b[k] for k in a}


def flatten_rows(split: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {"split": split, "task": "graph", **metrics["graph"]},
        {"split": split, "task": "attacker_count", "accuracy": metrics["count"]["accuracy"], "macro_f1": metrics["count"]["macro_f1"]},
        {"split": split, "task": "all_tasks", "exact_set": metrics["all_tasks_exact"]},
    ]
    for role in ROLES:
        for scope in ("all_windows", "attack_windows"):
            rows.append({"split": split, "task": role, "scope": scope, **metrics["roles"][role][scope]})
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--a2-dir", type=Path, required=True)
    p.add_argument("--a2-1-dir", type=Path, required=True)
    p.add_argument("--a3-dir", type=Path, required=True)
    p.add_argument("--r1c-dir", type=Path, required=True)
    p.add_argument("--b1-dir", type=Path, required=True)
    p.add_argument("--b2-dir", type=Path, required=True)
    p.add_argument("--b3-dir", type=Path, required=True)
    p.add_argument("--wrapper", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--seed", type=int, default=101)
    args = p.parse_args()

    root = args.root.expanduser().resolve()
    a2, a21, a3 = args.a2_dir.expanduser().resolve(), args.a2_1_dir.expanduser().resolve(), args.a3_dir.expanduser().resolve()
    r1c, b1, b2, b3 = args.r1c_dir.expanduser().resolve(), args.b1_dir.expanduser().resolve(), args.b2_dir.expanduser().resolve(), args.b3_dir.expanduser().resolve()
    wrapper = args.wrapper.expanduser().resolve()
    out_dir, model_dir = args.output_dir.expanduser().resolve(), args.model_dir.expanduser().resolve()

    if out_dir.exists():
        print(f"STOP: output directory already exists: {out_dir}", file=sys.stderr)
        return 2
    if model_dir.exists():
        print(f"STOP: model directory already exists: {model_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    failures, prior, provenance = verify_prerequisites(a2, a3, r1c, b1, b2, b3, wrapper)
    if failures:
        report = {"stage": "V5_P0_B4_GCN_ONLY", "status": "HOLD", "failures": failures, "test_split_accessed": False}
        write_json(out_dir / "V5_P0_B4_GCN_ONLY.json", report)
        atomic_write(out_dir / "V5_P0_B4_GCN_ONLY_HOLD", "V5_P0_B4_GCN_ONLY_HOLD\n")
        print("V5_P0_B4_GCN_ONLY_HOLD")
        return 1

    set_seed(args.seed)
    Dataset = import_dataset(wrapper)
    train_ds = Dataset(root=root, split="train", contract_dir=a2, mask_audit_dir=a21, window=32, stride=8, active_only=False, feature_variant="PRIMARY58")
    val_ds = Dataset(root=root, split="validation", contract_dir=a2, mask_audit_dir=a21, window=32, stride=8, active_only=False, feature_variant="PRIMARY58")
    sample = train_ds[0]
    if tuple(sample["x"].shape) != (16, 58, 32):
        failures.append(f"unexpected x shape: {tuple(sample['x'].shape)}")
    if tuple(sample["physical_port_mask"].shape) != (16, 10) or sample["physical_port_mask"].dtype != torch.bool:
        failures.append("unexpected physical_port_mask contract")
    if tuple(sample["edge_index"].shape) != (2, 48):
        failures.append(f"unexpected edge_index shape: {tuple(sample['edge_index'].shape)}")
    if failures:
        write_json(out_dir / "V5_P0_B4_GCN_ONLY.json", {"stage": "V5_P0_B4_GCN_ONLY", "status": "HOLD", "failures": failures, "test_split_accessed": False})
        atomic_write(out_dir / "V5_P0_B4_GCN_ONLY_HOLD", "V5_P0_B4_GCN_ONLY_HOLD\n")
        print("V5_P0_B4_GCN_ONLY_HOLD")
        return 1

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    train_eval = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    stats = target_statistics(train_eval)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNOnlyModel(sample["edge_index"]).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != 25544:
        raise RuntimeError(f"parameter count mismatch: {parameter_count} != 25544")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    pos_weights, pos_weight_values = build_pos_weights(stats, device)

    print("===== V5 P0-B4 GCN-ONLY =====")
    print("device:", device)
    print("parameter_count:", parameter_count)
    print("parameter_matched_to_B2: true")
    print("train_window_count:", len(train_ds))
    print("validation_window_count:", len(val_ds))
    print("input_temporal_reduction: x.mean(dim=-1)")
    print("temporal_order_used: false")
    print("gcn_layers: 1")
    print("message_passing_hops: 1")
    print("edge_index_used: true")
    print("metadata_used_as_model_input: false")
    print("validation_during_training: false")
    print("test_split_accessed: false")
    print("positive_weights:", pos_weight_values)

    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, samples = 0.0, 0
        components_total = {k: 0.0 for k in ("attack", "count", *ROLES)}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(
                batch["x"].to(device=device, dtype=torch.float32),
                batch["physical_port_mask"].to(device=device),
                batch["edge_index"].to(device=device, dtype=torch.long),
            )
            loss, components = compute_loss(out, batch, pos_weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            size = int(batch["x"].shape[0])
            total += float(loss.detach()) * size
            samples += size
            for k, v in components.items():
                components_total[k] += v * size
        mean_loss = total / max(1, samples)
        row = {"epoch": epoch, "train_loss": mean_loss}
        row.update({f"train_loss_{k}": v / max(1, samples) for k, v in components_total.items()})
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d}/{args.epochs} train_loss={mean_loss:.6f}")

    train_metrics, _ = evaluate(model, train_eval, device)
    val_metrics, val_predictions = evaluate(model, val_loader, device)

    ckpt = model_dir / "final_model.pt"
    torch.save({
        "stage": "V5_P0_B4_GCN_ONLY",
        "model_class": "GCNOnlyModel",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_epoch": args.epochs,
        "seed": args.seed,
        "feature_variant": "PRIMARY58",
        "window": 32,
        "stride": 8,
        "temporal_reduction": "mean_over_32_epochs",
        "temporal_order_used": False,
        "gcn_layers": 1,
        "message_passing_hops": 1,
        "edge_index_used": True,
        "metadata_used_as_model_input": False,
        "binary_threshold": 0.5,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "parameter_count": parameter_count,
    }, ckpt)
    pred_path = model_dir / "validation_predictions_and_targets.pt"
    torch.save({"stage": "V5_P0_B4_GCN_ONLY", "split": "validation", "binary_threshold": 0.5, "predictions_and_targets": val_predictions, "metadata_included": False}, pred_path)

    history_path = out_dir / "V5_P0_B4_GCN_ONLY_HISTORY.csv"
    metrics_path = out_dir / "V5_P0_B4_GCN_ONLY_METRICS.csv"
    write_csv(history_path, history)
    write_csv(metrics_path, flatten_rows("train", train_metrics) + flatten_rows("validation", val_metrics))

    comparison = {
        "validation_b4_minus_b2": metric_delta(val_metrics, prior["b2"]["validation_metrics"]),
        "validation_b4_minus_b3": metric_delta(val_metrics, prior["b3"]["validation_metrics"]),
        "key_metrics": {
            "b1_validation": key_metrics(prior["b1"]["validation_metrics"]),
            "b2_validation": key_metrics(prior["b2"]["validation_metrics"]),
            "b3_validation": key_metrics(prior["b3"]["validation_metrics"]),
            "b4_validation": key_metrics(val_metrics),
        },
    }
    comparison_path = out_dir / "V5_P0_B4_VS_B1_B2_B3_COMPARISON.json"
    write_json(comparison_path, comparison)

    report = {
        "stage": "V5_P0_B4_GCN_ONLY",
        "status": "COMPLETE",
        "scientific_question": "Does one-hop topology-aware message passing improve over B2's parameter-matched temporal-mean MLP?",
        "model": {
            "class": "GCNOnlyModel",
            "parameter_count": parameter_count,
            "parameter_matched_to_b2": True,
            "temporal_reduction": "x.mean(dim=-1)",
            "temporal_order_used": False,
            "gcn_layers": 1,
            "message_passing_hops": 1,
            "edge_index_used": True,
            "metadata_used_as_model_input": False,
        },
        "protocol": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "binary_threshold": 0.5,
            "early_stopping": False,
            "validation_checkpoint_selection": False,
            "validation_evaluated_only_after_training": True,
            "test_split_accessed": False,
        },
        "fairness_against_b2": {
            "same_parameter_count": True,
            "same_temporal_reduction": True,
            "same_seed_and_optimizer_protocol": True,
            "only_intended_change": "B2 independent Linear(64,64) replaced by normalized one-hop GCN(64,64)",
        },
        "train_target_statistics": stats,
        "positive_weights": pos_weight_values,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "comparison_to_b1_b2_b3": comparison,
        "artifacts": {
            "checkpoint": str(ckpt), "checkpoint_sha256": sha256_file(ckpt),
            "validation_predictions": str(pred_path), "validation_predictions_sha256": sha256_file(pred_path),
            "history_csv": str(history_path), "history_csv_sha256": sha256_file(history_path),
            "metrics_csv": str(metrics_path), "metrics_csv_sha256": sha256_file(metrics_path),
            "comparison_json": str(comparison_path), "comparison_json_sha256": sha256_file(comparison_path),
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
        "provenance": {**provenance, "script_sha256": sha256_file(Path(__file__))},
        "failures": [],
        "warnings": [],
        "next_stage": "V5_P0_B5_CONV1D_GCN",
    }
    report_path = out_dir / "V5_P0_B4_GCN_ONLY.json"
    write_json(report_path, report)
    lock = {
        "status": "V5_P0_B4_GCN_ONLY_COMPLETE",
        "report_sha256": sha256_file(report_path),
        "checkpoint_sha256": sha256_file(ckpt),
        "validation_predictions_sha256": sha256_file(pred_path),
        "history_csv_sha256": sha256_file(history_path),
        "metrics_csv_sha256": sha256_file(metrics_path),
        "comparison_json_sha256": sha256_file(comparison_path),
        "script_sha256": sha256_file(Path(__file__)),
        "test_split_accessed": False,
        "next_stage": "V5_P0_B5_CONV1D_GCN",
    }
    write_json(out_dir / "V5_P0_B4_GCN_ONLY_LOCK.json", lock)
    atomic_write(out_dir / "V5_P0_B4_GCN_ONLY_COMPLETE", "V5_P0_B4_GCN_ONLY_COMPLETE\n")

    d = comparison["validation_b4_minus_b2"]
    print("\n===== B4 FINAL RESULTS =====")
    print("train:", f"g_bal_acc={train_metrics['graph']['balanced_accuracy']:.4f}", f"g_f1={train_metrics['graph']['f1']:.4f}", f"g_fpr={train_metrics['graph']['fpr']:.4f}", f"count_macro_f1={train_metrics['count']['macro_f1']:.4f}", f"source_f1_attack={train_metrics['roles']['source']['attack_windows']['node_f1']:.4f}", f"victim_f1_attack={train_metrics['roles']['victim']['attack_windows']['node_f1']:.4f}", f"all_exact={train_metrics['all_tasks_exact']:.4f}")
    print("validation:", f"g_bal_acc={val_metrics['graph']['balanced_accuracy']:.4f}", f"g_f1={val_metrics['graph']['f1']:.4f}", f"g_fpr={val_metrics['graph']['fpr']:.4f}", f"count_macro_f1={val_metrics['count']['macro_f1']:.4f}", f"source_f1_attack={val_metrics['roles']['source']['attack_windows']['node_f1']:.4f}", f"victim_f1_attack={val_metrics['roles']['victim']['attack_windows']['node_f1']:.4f}", f"all_exact={val_metrics['all_tasks_exact']:.4f}")
    print("validation_delta_B4_minus_B2:", f"g_bal_acc={d['graph_balanced_accuracy']:+.4f}", f"g_f1={d['graph_f1']:+.4f}", f"g_fpr={d['graph_fpr']:+.4f}", f"count_macro_f1={d['count_macro_f1']:+.4f}", f"source_f1={d['source_node_f1_attack']:+.4f}", f"transit_f1={d['transit_node_f1_attack']:+.4f}", f"victim_f1={d['victim_node_f1_attack']:+.4f}", f"path_f1={d['path_node_f1_attack']:+.4f}", f"all_exact={d['all_tasks_exact']:+.4f}")
    print("failure_count: 0")
    print("warning_count: 0")
    print("test_split_accessed: false")
    print("next_stage: V5_P0_B5_CONV1D_GCN")
    print("V5_P0_B4_GCN_ONLY_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
