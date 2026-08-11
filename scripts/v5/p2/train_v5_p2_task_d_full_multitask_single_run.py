#!/usr/bin/env python3
"""One frozen Task-D full-multitask run for Conv1D or GraphConv."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

STAGE = "V5_P2_TASK_D_FULL_MULTITASK_SINGLE_RUN"
COMPLETE = f"{STAGE}_COMPLETE"
SEEDS = [107, 117, 127, 137, 147]
CANDIDATES = ["conv1d", "graphconv"]
EXPECTED_PARAMS = {"conv1d": 43_273, "graphconv": 59_785}
EXPECTED_TRAIN_ITEMS = 70_166
EXPECTED_VALIDATION_ITEMS = 12_528
EXPECTED_B1_PROTOCOL_SHA = "0817ae7812f91e3c75260589acaf52bd8a74b07b2114a451b4773578efde4b60"
COUNT_CLASS_VALUES = [1, 2, 3, 4]


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    # Python 3.12 dataclasses resolve annotations through sys.modules while
    # the module body is executing. Register the dynamic module first.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def measure_latency(model, batch: dict[str, torch.Tensor], device: torch.device) -> float:
    x = batch["x"].to(device, non_blocking=(device.type == "cuda"))
    mask = batch["physical_port_mask"].to(device, non_blocking=(device.type == "cuda"))
    model.eval()
    warmup, repeats = 10, 40
    with torch.no_grad():
        for _ in range(warmup):
            model(x, mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(x, mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return elapsed * 1e6 / (repeats * int(x.shape[0]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--b1-dir", type=Path, required=True)
    parser.add_argument("--b0-r3-dir", type=Path, required=True)
    parser.add_argument("--promotion-dir", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--topology-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--b3-model-path", type=Path, required=True)
    parser.add_argument("--task-d-model-path", type=Path, required=True)
    parser.add_argument("--b2-training-script-path", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate = args.candidate
    seed = int(args.seed)
    if seed not in SEEDS:
        print(f"STOP: seed={seed} is not in {SEEDS}", file=sys.stderr)
        return 2

    resolved = {k: v.expanduser().resolve() if isinstance(v, Path) else v for k, v in vars(args).items()}
    root: Path = resolved["root"]
    model_dir: Path = resolved["model_dir"]
    report_dir: Path = resolved["report_dir"]
    if model_dir.exists() or report_dir.exists():
        print("STOP: model/report output already exists", file=sys.stderr)
        print(f"model_dir={model_dir}", file=sys.stderr)
        print(f"report_dir={report_dir}", file=sys.stderr)
        return 2
    model_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    paths = {
        "b1_protocol": resolved["b1_dir"] / "V5_P2_B1_TRAINING_PROTOCOL.json",
        "b1_report": resolved["b1_dir"] / "V5_P2_B1_TRAINING_PROTOCOL_LOCK.json",
        "b1_lock": resolved["b1_dir"] / "V5_P2_B1_TRAINING_PROTOCOL_LOCK_LOCK.json",
        "b0_report": resolved["b0_r3_dir"] / "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT.json",
        "b0_lock": resolved["b0_r3_dir"] / "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT_LOCK.json",
        "promotion_report": resolved["promotion_dir"] / "V5_P2_G123_GRAPH_OPERATOR_PROMOTION.json",
        "promotion_lock": resolved["promotion_dir"] / "V5_P2_G123_GRAPH_OPERATOR_PROMOTION_LOCK.json",
        "preflight_report": resolved["preflight_dir"] / "V5_P2_TASK_D_FULL_MULTITASK_PREFLIGHT.json",
        "preflight_lock": resolved["preflight_dir"] / "V5_P2_TASK_D_FULL_MULTITASK_PREFLIGHT_LOCK.json",
        "edge": resolved["topology_dir"] / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
        "pair_manifest": resolved["pair_manifest"],
        "loader": resolved["loader_path"],
        "b3_model": resolved["b3_model_path"],
        "task_d_model": resolved["task_d_model_path"],
        "b2_train": resolved["b2_training_script_path"],
    }
    failures = [f"missing {name}: {path}" for name, path in paths.items() if not path.is_file()]
    warnings: list[str] = []
    if not root.is_dir():
        failures.append(f"dataset root missing: {root}")
    if failures:
        write_json(report_dir / f"{STAGE}.json", {
            "stage": STAGE, "status": "HOLD", "candidate": candidate,
            "seed": seed, "failures": failures,
            "test_directory_enumerated": False, "test_tensors_deserialized": False,
        })
        (report_dir / f"{STAGE}_HOLD").write_text(f"{STAGE}_HOLD\n", encoding="utf-8")
        return 1

    protocol = json.loads(paths["b1_protocol"].read_text())
    b1_report = json.loads(paths["b1_report"].read_text())
    b1_lock = json.loads(paths["b1_lock"].read_text())
    b0_report = json.loads(paths["b0_report"].read_text())
    b0_lock = json.loads(paths["b0_lock"].read_text())
    promotion = json.loads(paths["promotion_report"].read_text())
    promotion_lock = json.loads(paths["promotion_lock"].read_text())
    preflight = json.loads(paths["preflight_report"].read_text())
    preflight_lock = json.loads(paths["preflight_lock"].read_text())

    if protocol.get("protocol_sha256") != EXPECTED_B1_PROTOCOL_SHA:
        failures.append("B1 protocol SHA changed")
    if b1_lock.get("protocol_file_sha256") != sha256_file(paths["b1_protocol"]):
        failures.append("B1 protocol file SHA mismatch")
    if b1_lock.get("model_sha256") != sha256_file(paths["b3_model"]):
        failures.append("B3 model SHA mismatch")
    if b1_lock.get("loader_sha256") != sha256_file(paths["loader"]):
        failures.append("loader SHA mismatch")
    if b0_lock.get("report_sha256") != sha256_file(paths["b0_report"]):
        failures.append("B0-R3 report SHA mismatch")
    if b0_lock.get("shortcut_block_count") != 0 or b0_lock.get("label_integrity_pass") is not True:
        failures.append("B0-R3 label/shortcut audit is not clean")
    if promotion_lock.get("report_sha256") != sha256_file(paths["promotion_report"]):
        failures.append("promotion report SHA mismatch")
    if promotion["decision"]["task_d_challenger"] != "graphconv":
        failures.append("promotion challenger changed")
    if preflight_lock.get("report_sha256") != sha256_file(paths["preflight_report"]):
        failures.append("preflight report SHA mismatch")
    if preflight.get("scientific_training_authorized") is not True:
        failures.append("Task-D preflight did not authorize training")
    if candidate == "graphconv" and promotion["decision"]["promotion_scope"] != "TASK_D_REVIEWER_CHALLENGER_ONLY":
        failures.append("GraphConv promotion scope changed")

    if failures:
        write_json(report_dir / f"{STAGE}.json", {
            "stage": STAGE, "status": "HOLD", "candidate": candidate,
            "seed": seed, "failures": failures, "warnings": warnings,
            "test_directory_enumerated": False, "test_tensors_deserialized": False,
        })
        (report_dir / f"{STAGE}_HOLD").write_text(f"{STAGE}_HOLD\n", encoding="utf-8")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    b2 = import_module(paths["b2_train"], f"taskd_b2_{candidate}_{seed}")
    loader_mod = import_module(paths["loader"], f"taskd_loader_{candidate}_{seed}")
    b3_mod = import_module(paths["b3_model"], f"taskd_b3_{candidate}_{seed}")
    td_mod = import_module(paths["task_d_model"], f"taskd_model_{candidate}_{seed}")

    b2.EXPECTED_SEEDS = SEEDS
    b2.EXPECTED_TRAIN_ITEMS = EXPECTED_TRAIN_ITEMS
    b2.EXPECTED_VALIDATION_ITEMS = EXPECTED_VALIDATION_ITEMS
    b2.set_seed(seed)

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation length={len(validation_dataset)}")

    train_summary = b0_report["label_summaries"]["train"]
    graph_positive = int(train_summary["graph_positive"])
    graph_negative = int(train_summary["graph_negative"])
    graph_pos_weight_value = graph_negative / graph_positive

    role_distribution_keys = {
        "source": "active_source_count_distribution",
        "transit": "active_transit_count_distribution",
        "victim": "active_victim_count_distribution",
        "path": "active_path_count_distribution",
    }
    role_weight_manifest: dict[str, Any] = {}
    for role, key in role_distribution_keys.items():
        distribution = b2.parse_distribution(train_summary[key])
        positives = b2.role_positive_count(distribution)
        negatives = EXPECTED_TRAIN_ITEMS * 16 - positives
        raw = negatives / positives
        role_weight_manifest[role] = {
            "positive_entries": positives,
            "negative_entries": negatives,
            "raw_positive_weight": raw,
            "clamped_positive_weight": min(20.0, max(1.0, raw)),
        }
    count_distribution = b2.parse_distribution(train_summary["active_count_distribution"])
    if sorted(count_distribution) != COUNT_CLASS_VALUES:
        raise RuntimeError("count distribution is not 1..4")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
    if candidate == "conv1d":
        model = reference_b3
    else:
        model = td_mod.P2TaskDGraphConvCount4(reference_b3, edge)
    model = model.to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != EXPECTED_PARAMS[candidate]:
        raise RuntimeError(f"parameter count={parameter_count}, expected {EXPECTED_PARAMS[candidate]}")

    full_initial_state_sha256 = b2.sha256_state_dict(model.state_dict())
    if candidate == "conv1d":
        common_initial_state = OrderedDict((k, v) for k, v in model.state_dict().items())
    else:
        common_initial_state = OrderedDict(td_mod.common_state_dict(model).items())
    common_initial_state_sha256 = b2.sha256_state_dict(common_initial_state)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, threshold=1e-4,
        threshold_mode="abs", cooldown=0, min_lr=1e-5,
    )
    graph_pos_weight = torch.tensor(graph_pos_weight_value, dtype=torch.float32, device=device)
    role_pos_weights = {
        role: torch.tensor(details["clamped_positive_weight"], dtype=torch.float32, device=device)
        for role, details in role_weight_manifest.items()
    }

    checkpoint_path = model_dir / "best_checkpoint.pt"
    history_path = report_dir / f"V5_P2_TASK_D_{candidate.upper()}_SEED_{seed}_HISTORY.csv"
    progress_path = report_dir / f"V5_P2_TASK_D_{candidate.upper()}_SEED_{seed}_PROGRESS.json"
    start = time.time()
    best_rank = None
    best_epoch = None
    best_validation_metrics = None
    best_checkpoint_sha256 = None
    best_early_stop_score = float("-inf")
    patience_counter = 0
    history_rows: list[dict[str, Any]] = []
    completed_epoch = 0
    stopped_early = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print("===== V5 P2 TASK-D FULL MULTITASK SINGLE RUN =====")
    print("candidate:", candidate)
    print("seed:", seed)
    print("device:", device)
    print("train_items:", len(train_dataset))
    print("validation_items:", len(validation_dataset))
    print("parameter_count:", parameter_count)
    print("b1_protocol_sha256:", protocol["protocol_sha256"])
    print("threshold_tuning_performed: false")
    print("test_directory_enumerated: false")
    print("test_tensors_deserialized: false")

    for epoch in range(1, 101):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = b2.train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, device=device,
            graph_pos_weight=graph_pos_weight, role_pos_weights=role_pos_weights,
            gradient_clip=1.0,
        )
        validation_metrics = b2.validate(
            model=model, loader=validation_loader, device=device,
            graph_pos_weight=graph_pos_weight, role_pos_weights=role_pos_weights,
        )
        rank = b2.checkpoint_rank(validation_metrics, epoch)
        is_best = best_rank is None or rank > best_rank
        if is_best:
            best_rank = rank
            best_epoch = epoch
            best_validation_metrics = validation_metrics
            payload = {
                "stage": STAGE,
                "candidate": candidate,
                "seed": seed,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "b1_protocol_sha256": protocol["protocol_sha256"],
                "promotion_report_sha256": sha256_file(paths["promotion_report"]),
                "loader_sha256": sha256_file(paths["loader"]),
                "b3_model_sha256": sha256_file(paths["b3_model"]),
                "task_d_model_sha256": sha256_file(paths["task_d_model"]),
                "full_initial_state_sha256": full_initial_state_sha256,
                "common_initial_state_sha256": common_initial_state_sha256,
                "train_label_weight_manifest": {
                    "graph_positive_weight": graph_pos_weight_value,
                    "role_positive_weights": role_weight_manifest,
                    "count_distribution": count_distribution,
                    "count_class_mapping": b2.COUNT_CLASS_MAPPING,
                },
                "validation_metrics": validation_metrics,
                "threshold_tuning_performed": False,
                "test_tensors_deserialized": False,
            }
            b2.atomic_torch_save(payload, checkpoint_path)
            best_checkpoint_sha256 = sha256_file(checkpoint_path)

        score = float(validation_metrics["selection_score"])
        if score > best_early_stop_score + 1e-4:
            best_early_stop_score = score
            patience_counter = 0
        else:
            patience_counter += 1
        scheduler.step(score)
        completed_epoch = epoch
        row = b2.flatten_epoch_row(
            epoch=epoch, learning_rate=current_lr, train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            elapsed_seconds=time.time() - epoch_start,
            is_best_checkpoint=is_best,
            early_stop_patience_counter=patience_counter,
        )
        row = {"candidate": candidate, "seed": seed, **row}
        history_rows.append(row)
        b2.write_csv(history_path, history_rows)
        write_json(progress_path, {
            "stage": STAGE, "status": "RUNNING", "candidate": candidate,
            "seed": seed, "completed_epoch": completed_epoch, "best_epoch": best_epoch,
            "best_checkpoint_sha256": best_checkpoint_sha256,
            "best_validation_metrics": best_validation_metrics,
            "current_validation_metrics": validation_metrics,
            "current_learning_rate_after_scheduler": float(optimizer.param_groups[0]["lr"]),
            "early_stop_patience_counter": patience_counter,
            "test_directory_enumerated": False, "test_tensors_deserialized": False,
        })
        print(
            f"candidate={candidate} seed={seed} epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} val_loss={validation_metrics['loss']:.6f} "
            f"score={score:.6f} g_auc={validation_metrics['graph']['auroc']:.6f} "
            f"g_ap={validation_metrics['graph']['average_precision']:.6f} "
            f"count_f1={validation_metrics['count_active']['macro_f1']:.6f} "
            f"src_ap={validation_metrics['roles']['source']['average_precision']:.6f} "
            f"tr_ap={validation_metrics['roles']['transit']['average_precision']:.6f} "
            f"vic_ap={validation_metrics['roles']['victim']['average_precision']:.6f} "
            f"path_ap={validation_metrics['roles']['path']['average_precision']:.6f} "
            f"lr={current_lr:.8g} best_epoch={best_epoch} patience={patience_counter}"
        )
        if epoch >= 15 and patience_counter >= 12:
            stopped_early = True
            print(f"early stopping at epoch {epoch}; patience reached 12")
            break

    if best_epoch is None or best_validation_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("no best checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    first_validation_batch = next(iter(validation_loader))
    latency_us = measure_latency(model, first_validation_batch, device)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    operation_proxy = td_mod.operation_count_proxy(candidate)

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "candidate": candidate,
        "seed": seed,
        "architecture": {
            "name": model.architecture_name,
            "parameter_count": parameter_count,
            "full_initial_state_sha256": full_initial_state_sha256,
            "common_initial_state_sha256": common_initial_state_sha256,
            "b3_model_sha256": sha256_file(paths["b3_model"]),
            "task_d_model_sha256": sha256_file(paths["task_d_model"]),
            "loader_sha256": sha256_file(paths["loader"]),
            "b1_protocol_sha256": protocol["protocol_sha256"],
            "promotion_report_sha256": sha256_file(paths["promotion_report"]),
        },
        "data": {
            "train_items": len(train_dataset), "validation_items": len(validation_dataset),
            "pair_block_batch_size": 128, "item_batch_size": 256,
            "train_batches": len(train_loader), "validation_batches": len(validation_loader),
            "pair_keys_returned_to_model": False,
        },
        "label_weights": {
            "graph_positive_weight": graph_pos_weight_value,
            "role_positive_weights": role_weight_manifest,
            "count_distribution": count_distribution,
            "count_class_weights_applied": False,
        },
        "training": {
            "device": str(device), "completed_epoch": completed_epoch,
            "stopped_early": stopped_early, "maximum_epochs": 100,
            "minimum_epochs_before_stop": 15, "early_stopping_patience": 12,
            "optimizer": "AdamW", "initial_learning_rate": 1e-3,
            "weight_decay": 1e-4, "gradient_clip_global_norm": 1.0,
            "scheduler": "ReduceLROnPlateau", "automatic_mixed_precision": False,
            "total_elapsed_seconds": time.time() - start,
        },
        "best": {
            "epoch": best_epoch, "validation_metrics": best_validation_metrics,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "ranking_tuple": list(best_rank),
        },
        "deployment_proxies": {
            "inference_microseconds_per_item": latency_us,
            "peak_cuda_memory_bytes": peak_memory,
            **operation_proxy,
        },
        "artifacts": {
            "history_csv": [str(history_path), sha256_file(history_path)],
            "progress_json": [str(progress_path), sha256_file(progress_path)],
            "best_checkpoint": [str(checkpoint_path), sha256_file(checkpoint_path)],
        },
        "security_boundary": {
            "train_tensor_contents_accessed": True,
            "validation_tensor_contents_accessed": True,
            "threshold_tuning_performed": False,
            "test_directory_existence_checked": False,
            "test_directory_enumerated": False,
            "test_tensor_files_opened": False,
            "test_tensors_deserialized": False,
            "test_dataset_constructed": False,
            "test_evaluation_performed": False,
        },
        "failures": [], "warnings": warnings,
        "next_stage": "V5_P2_TASK_D_FULL_MULTITASK_MATRIX_AGGREGATION",
    }
    report_path = report_dir / f"V5_P2_TASK_D_{candidate.upper()}_SEED_{seed}_REPORT.json"
    write_json(report_path, report)
    lock = {
        "status": COMPLETE, "candidate": candidate, "seed": seed,
        "report_sha256": sha256_file(report_path),
        "history_sha256": sha256_file(history_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "common_initial_state_sha256": common_initial_state_sha256,
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_validation_metrics["selection_score"],
        "threshold_tuning_performed": False,
        "architecture_selected": False,
        "test_directory_enumerated": False,
        "test_tensors_deserialized": False,
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(report_dir / f"V5_P2_TASK_D_{candidate.upper()}_SEED_{seed}_LOCK.json", lock)
    (report_dir / COMPLETE).write_text(COMPLETE + "\n", encoding="utf-8")

    print("===== V5 P2 TASK-D SINGLE RUN COMPLETE =====")
    print("candidate:", candidate)
    print("seed:", seed)
    print("best_epoch:", best_epoch)
    print("best_validation_selection_score:", best_validation_metrics["selection_score"])
    print("inference_microseconds_per_item:", latency_us)
    print("threshold_tuning_performed: false")
    print("architecture_selected: false")
    print("test_directory_enumerated: false")
    print("test_tensors_deserialized: false")
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
