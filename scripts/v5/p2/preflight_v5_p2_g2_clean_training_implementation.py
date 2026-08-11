#!/usr/bin/env python3
"""Clean-production preflight for V5 P2-G2 direct graph detection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

STAGE = "V5_P2_G2_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT"
COMPLETE = f"{STAGE}_COMPLETE"
HOLD = f"{STAGE}_HOLD"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stack_items(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items], dim=0) for key in items[0]}


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    trainer_path = args.trainer_path.expanduser().resolve()
    trainer = import_module(trainer_path, "v5_p2_g2_clean_preflight_trainer")
    contracts = trainer.verify_contracts(args)
    loader_module = import_module(args.loader_path.expanduser().resolve(), "v5_p2_g2_clean_preflight_loader")
    b3_module = import_module(args.b3_model_path.expanduser().resolve(), "v5_p2_g2_clean_preflight_b3")
    DatasetClass = loader_module.V5P2PairAlignedPrimary58Dataset
    B3Class = b3_module.P2B3Conv1DOnlyCount4

    train_dataset = DatasetClass(root=args.root.expanduser().resolve(), split="train", pair_manifest=args.pair_manifest.expanduser().resolve(), window=32, stride=8)
    validation_dataset = DatasetClass(root=args.root.expanduser().resolve(), split="validation", pair_manifest=args.pair_manifest.expanduser().resolve(), window=32, stride=8)
    if len(train_dataset) != trainer.EXPECTED_TRAIN_ITEMS:
        raise RuntimeError("G2 preflight train length changed")
    if len(validation_dataset) != trainer.EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError("G2 preflight validation length changed")

    train_batch_cpu = stack_items([train_dataset[index] for index in range(8)])
    validation_batch_cpu = stack_items([validation_dataset[index] for index in range(8)])
    if tuple(train_batch_cpu["x"].shape) != (8, 16, 58, 32):
        raise RuntimeError("G2 preflight train input shape changed")
    if tuple(validation_batch_cpu["x"].shape) != (8, 16, 58, 32):
        raise RuntimeError("G2 preflight validation input shape changed")
    if tuple(train_batch_cpu["y_attack"].reshape(-1).shape) != (8,):
        raise RuntimeError("G2 preflight train graph target shape changed")
    if tuple(validation_batch_cpu["y_attack"].reshape(-1).shape) != (8,):
        raise RuntimeError("G2 preflight validation graph target shape changed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, Any] = {}
    reference_b3_hashes: set[str] = set()
    graph_projection_hashes: set[str] = set()
    attack_head_hashes: set[str] = set()
    encoder_hashes: set[str] = set()

    for operator in trainer.OPERATORS:
        trainer.set_seed(107)
        reference_b3 = B3Class()
        model = trainer.CleanDirectGraphBaseline(reference_b3, operator, contracts["edge_index"])
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != trainer.EXPECTED_CLEAN_PARAMETERS[operator]:
            raise RuntimeError(f"{operator}: parameter_count={parameter_count}, expected {trainer.EXPECTED_CLEAN_PARAMETERS[operator]}")

        reference_b3_hash = trainer.sha256_state_dict(reference_b3.state_dict())
        graph_projection_hash = trainer.sha256_state_dict(model.graph_projection.state_dict())
        attack_head_hash = trainer.sha256_state_dict(model.attack_head.state_dict())
        encoder_state = {
            **{f"input_projection.{key}": value for key, value in model.input_projection.state_dict().items()},
            **{f"temporal_blocks.{key}": value for key, value in model.temporal_blocks.state_dict().items()},
            **{f"node_projection.{key}": value for key, value in model.node_projection.state_dict().items()},
        }
        encoder_hash = trainer.sha256_state_dict(encoder_state)
        reference_b3_hashes.add(reference_b3_hash)
        graph_projection_hashes.add(graph_projection_hash)
        attack_head_hashes.add(attack_head_hash)
        encoder_hashes.add(encoder_hash)

        reference_b3.eval()
        model.eval()
        with torch.no_grad():
            reference_logits = reference_b3(train_batch_cpu["x"], train_batch_cpu["physical_port_mask"])["attack_logits"]
            clean_logits = model.graph_logits_without_message_passing(train_batch_cpu["x"], train_batch_cpu["physical_port_mask"])
        equivalence_error = float((reference_logits - clean_logits).abs().max())
        if equivalence_error != 0.0:
            raise RuntimeError(f"{operator}: clean graph-logit extraction equivalence error={equivalence_error}")
        del reference_b3

        model = model.to(device)
        train_batch = trainer.move_batch(train_batch_cpu, device)
        validation_batch = trainer.move_batch(validation_batch_cpu, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(train_batch["x"], train_batch["physical_port_mask"])
        train_target = train_batch["y_attack"].float().reshape(-1)
        train_loss = F.binary_cross_entropy_with_logits(train_logits, train_target)
        if not torch.isfinite(train_loss):
            raise RuntimeError(f"{operator}: non-finite train loss")
        train_loss.backward()

        trainable_tensors = 0
        gradient_tensors = 0
        finite_gradient_tensors = 0
        total_absolute_gradient = 0.0
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            trainable_tensors += 1
            if parameter.grad is None:
                continue
            gradient_tensors += 1
            if torch.isfinite(parameter.grad).all():
                finite_gradient_tensors += 1
            total_absolute_gradient += float(parameter.grad.abs().sum().item())
        if gradient_tensors == 0 or gradient_tensors != finite_gradient_tensors:
            raise RuntimeError(f"{operator}: missing or non-finite gradients")
        if total_absolute_gradient <= 0.0:
            raise RuntimeError(f"{operator}: zero total gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_batch["x"], validation_batch["physical_port_mask"])
            validation_target = validation_batch["y_attack"].float().reshape(-1)
            validation_loss = F.binary_cross_entropy_with_logits(validation_logits, validation_target)
        if tuple(train_logits.shape) != (8,) or tuple(validation_logits.shape) != (8,):
            raise RuntimeError(f"{operator}: graph logits shape changed")
        if not torch.isfinite(validation_loss):
            raise RuntimeError(f"{operator}: non-finite validation loss")

        results[operator] = {
            "status": "PASS",
            "parameter_count": parameter_count,
            "clean_equivalence_max_abs_error": equivalence_error,
            "train_logits_shape": list(train_logits.shape),
            "validation_logits_shape": list(validation_logits.shape),
            "train_loss_before_step": float(train_loss.item()),
            "validation_loss_after_step": float(validation_loss.item()),
            "trainable_parameter_tensors": trainable_tensors,
            "gradient_parameter_tensors": gradient_tensors,
            "finite_gradient_parameter_tensors": finite_gradient_tensors,
            "total_absolute_gradient": total_absolute_gradient,
            "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "reference_b3_initial_state_sha256": reference_b3_hash,
            "common_encoder_initial_state_sha256": encoder_hash,
            "graph_projection_initial_state_sha256": graph_projection_hash,
            "attack_head_initial_state_sha256": attack_head_hash,
        }

    if len(reference_b3_hashes) != 1:
        raise RuntimeError("operators did not receive identical seeded B3 initialization")
    if len(encoder_hashes) != 1 or len(graph_projection_hashes) != 1 or len(attack_head_hashes) != 1:
        raise RuntimeError("common encoder/readout initialization differs across operators")

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "device": str(device),
        "operator_results": results,
        "operator_pass_count": sum(value["status"] == "PASS" for value in results.values()),
        "scientific_training_authorized": len(results) == len(trainer.OPERATORS),
        "contracts": {
            "trainer_sha256": sha256_file(trainer_path),
            "loader_sha256": sha256_file(args.loader_path.expanduser().resolve()),
            "b3_model_sha256": sha256_file(args.b3_model_path.expanduser().resolve()),
            "edge_index_sha256": trainer.EXPECTED_EDGE_SHA,
            "g1_aggregation_report_sha256": trainer.EXPECTED_G1_AGGREGATION_REPORT_SHA,
            "g1_decision_sha256": trainer.EXPECTED_G1_DECISION_SHA,
        },
        "security_boundary": {
            "p2_train_used": True,
            "p2_validation_used": True,
            "p2_test_directory_existence_checked": False,
            "p2_test_directory_enumerated": False,
            "p2_test_tensor_files_opened": False,
            "p2_test_tensors_deserialized": False,
            "p2_test_evaluation_performed": False,
            "architecture_selected": False,
        },
        "next_stage": "V5_P2_G2_DIRECT_GRAPH_TRAINING_MATRIX",
    }
    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)
    lock = {
        "status": COMPLETE,
        "scientific_training_authorized": report["scientific_training_authorized"],
        "trainer_sha256": sha256_file(trainer_path),
        "report_sha256": sha256_file(report_path),
        "edge_index_sha256": trainer.EXPECTED_EDGE_SHA,
        "operator_pass_count": report["operator_pass_count"],
        "p2_test_directory_enumerated": False,
        "p2_test_tensors_deserialized": False,
        "architecture_selected": False,
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-G2 CLEAN TRAINING IMPLEMENTATION PREFLIGHT =====")
    print("status: COMPLETE")
    print("device:", device)
    print("operator_pass_count:", report["operator_pass_count"])
    for operator in trainer.OPERATORS:
        result = results[operator]
        print(f"{operator} PASS train_loss={result['train_loss_before_step']:.8f} val_loss={result['validation_loss_after_step']:.8f} params={result['parameter_count']}")
    print("scientific_training_authorized: true")
    print("test_directory_enumerated: false")
    print("test_tensors_deserialized: false")
    print(COMPLETE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--g0-dir", type=Path, required=True)
    parser.add_argument("--topology-dir", type=Path, required=True)
    parser.add_argument("--b0-r3-dir", type=Path, required=True)
    parser.add_argument("--g1-freeze-dir", type=Path, required=True)
    parser.add_argument("--g1-aggregation-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--b3-model-path", type=Path, required=True)
    parser.add_argument("--trainer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        return run(args)
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = f"{type(exc).__name__}: {exc}"
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failure": failure,
            "scientific_training_authorized": False,
            "p2_test_directory_enumerated": False,
            "p2_test_tensors_deserialized": False,
            "architecture_selected": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / HOLD, HOLD + "\n")
        print(HOLD)
        print("FAIL:", failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
