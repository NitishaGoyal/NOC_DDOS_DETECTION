#!/usr/bin/env python3
"""Clean implementation preflight for V5 P2-G3 role-aware multilabel training."""

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


STAGE = "V5_P2_G3_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT"
COMPLETE = f"{STAGE}_COMPLETE"
OPERATORS = ("conv1d", "gcnconv", "graphconv")
EXPECTED_PARAMETERS = {
    "conv1d": 34_692,
    "gcnconv": 43_012,
    "graphconv": 51_204,
}


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
    return {
        key: torch.stack([item[key] for item in items], dim=0)
        for key in items[0]
    }


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def common_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    prefixes = (
        "input_projection.",
        "temporal_blocks.",
        "node_projection.",
        "source_head.",
        "transit_head.",
        "victim_head.",
        "path_head.",
    )
    return {
        key: value
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    trainer_path = args.trainer_path.expanduser().resolve()
    trainer = import_module(trainer_path, "v5_p2_g3_clean_preflight_trainer")
    contracts = trainer.verify_contracts(args)

    loader_module = import_module(
        args.loader_path.expanduser().resolve(),
        "v5_p2_g3_clean_preflight_loader",
    )
    b3_module = import_module(
        args.b3_model_path.expanduser().resolve(),
        "v5_p2_g3_clean_preflight_b3",
    )
    DatasetClass = loader_module.V5P2PairAlignedPrimary58Dataset
    B3Class = b3_module.P2B3Conv1DOnlyCount4

    train_dataset = DatasetClass(
        root=args.root.expanduser().resolve(),
        split="train",
        pair_manifest=args.pair_manifest.expanduser().resolve(),
        window=32,
        stride=8,
    )
    validation_dataset = DatasetClass(
        root=args.root.expanduser().resolve(),
        split="validation",
        pair_manifest=args.pair_manifest.expanduser().resolve(),
        window=32,
        stride=8,
    )
    if len(train_dataset) != trainer.EXPECTED_TRAIN_ITEMS:
        raise RuntimeError("G3 preflight train length changed")
    if len(validation_dataset) != trainer.EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError("G3 preflight validation length changed")

    train_batch_cpu = stack_items(
        [train_dataset[index] for index in range(8)]
    )
    validation_batch_cpu = stack_items(
        [validation_dataset[index] for index in range(8)]
    )
    for name, batch in (
        ("train", train_batch_cpu),
        ("validation", validation_batch_cpu),
    ):
        if tuple(batch["x"].shape) != (8, 16, 58, 32):
            raise RuntimeError(f"G3 preflight {name} input shape changed")
        if tuple(batch["physical_port_mask"].shape) != (8, 16, 10):
            raise RuntimeError(f"G3 preflight {name} mask shape changed")
        if tuple(batch["y_attack"].reshape(-1).shape) != (8,):
            raise RuntimeError(f"G3 preflight {name} attack target shape changed")
        for role in trainer.ROLES:
            if tuple(batch[trainer.TARGET_KEYS[role]].shape) != (8, 16):
                raise RuntimeError(
                    f"G3 preflight {name} {role} target shape changed"
                )

    role_weights = trainer.role_positive_weights(contracts["b0_report"])
    expected_role_weights = {
        "source": 20.0,
        "transit": 15.7433147902343,
        "victim": 20.0,
        "path": 7.375342240922689,
    }
    for role, expected in expected_role_weights.items():
        if abs(role_weights[role] - expected) > 1e-12:
            raise RuntimeError(
                f"G3 preflight {role} positive weight changed: "
                f"{role_weights[role]} != {expected}"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, Any] = {}
    common_hashes: set[str] = set()
    role_head_hashes = {role: set() for role in trainer.ROLES}
    reference_hashes: set[str] = set()

    for operator in OPERATORS:
        trainer.set_seed(107)
        reference_b3 = B3Class()
        model = trainer.CleanRoleAwareBaseline(
            reference_b3,
            operator,
            contracts["edge_index"],
        )
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        if parameter_count != EXPECTED_PARAMETERS[operator]:
            raise RuntimeError(
                f"{operator}: parameter_count={parameter_count}, "
                f"expected {EXPECTED_PARAMETERS[operator]}"
            )

        reference_hashes.add(state_hash(reference_b3))
        common_digest = hashlib.sha256()
        for key, value in common_state(model).items():
            common_digest.update(key.encode("utf-8"))
            tensor = value.detach().cpu().contiguous()
            common_digest.update(tensor.numpy().tobytes())
        common_hashes.add(common_digest.hexdigest())
        for role in trainer.ROLES:
            role_head_hashes[role].add(
                state_hash(getattr(model, f"{role}_head"))
            )

        reference_b3.eval()
        model.eval()
        with torch.no_grad():
            reference_output = reference_b3(
                train_batch_cpu["x"],
                train_batch_cpu["physical_port_mask"],
            )
            clean_output = model.logits_without_graph(
                train_batch_cpu["x"],
                train_batch_cpu["physical_port_mask"],
            )
        equivalence = {
            "source": float(
                (
                    reference_output["source_logits"]
                    - clean_output["source"]
                ).abs().max()
            ),
            "transit": float(
                (
                    reference_output["transit_logits"]
                    - clean_output["transit"]
                ).abs().max()
            ),
            "victim": float(
                (
                    reference_output["victim_logits"]
                    - clean_output["victim"]
                ).abs().max()
            ),
            "path": float(
                (
                    reference_output["path_logits"]
                    - clean_output["path"]
                ).abs().max()
            ),
        }
        if any(value != 0.0 for value in equivalence.values()):
            raise RuntimeError(
                f"{operator}: clean Conv1D role equivalence failed: {equivalence}"
            )

        model = model.to(device)
        train_batch = trainer.move_batch(train_batch_cpu, device)
        validation_batch = trainer.move_batch(
            validation_batch_cpu, device
        )
        pos_weights = {
            role: torch.tensor(
                role_weights[role], dtype=torch.float32, device=device
            )
            for role in trainer.ROLES
        }
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.001, weight_decay=0.0001
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(
            train_batch["x"], train_batch["physical_port_mask"]
        )
        train_loss, train_role_losses = trainer.weighted_role_losses(
            train_logits, train_batch, pos_weights
        )
        if not torch.isfinite(train_loss):
            raise RuntimeError(f"{operator}: non-finite train loss")
        train_loss.backward()
        finite_gradients = True
        nonzero_gradient = False
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            finite_gradients = finite_gradients and bool(
                torch.isfinite(parameter.grad).all()
            )
            nonzero_gradient = nonzero_gradient or bool(
                parameter.grad.abs().max() > 0
            )
        if not finite_gradients or not nonzero_gradient:
            raise RuntimeError(
                f"{operator}: invalid gradient state "
                f"finite={finite_gradients} nonzero={nonzero_gradient}"
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(
                validation_batch["x"],
                validation_batch["physical_port_mask"],
            )
            validation_loss, validation_role_losses = (
                trainer.weighted_role_losses(
                    validation_logits,
                    validation_batch,
                    pos_weights,
                )
            )
        if not torch.isfinite(validation_loss):
            raise RuntimeError(f"{operator}: non-finite validation loss")
        for role in trainer.ROLES:
            if tuple(validation_logits[role].shape) != (8, 16):
                raise RuntimeError(
                    f"{operator}: {role} output shape changed"
                )

        results[operator] = {
            "status": "PASS",
            "parameter_count": parameter_count,
            "train_loss": float(train_loss.item()),
            "validation_loss": float(validation_loss.item()),
            "train_role_losses": {
                role: float(train_role_losses[role].item())
                for role in trainer.ROLES
            },
            "validation_role_losses": {
                role: float(validation_role_losses[role].item())
                for role in trainer.ROLES
            },
            "clean_role_equivalence_max_abs_error": equivalence,
            "finite_gradients": finite_gradients,
            "nonzero_gradient": nonzero_gradient,
            "output_shapes": {
                role: list(validation_logits[role].shape)
                for role in trainer.ROLES
            },
        }

    if len(reference_hashes) != 1:
        raise RuntimeError("seeded B3 reference initial state differs by operator")
    if len(common_hashes) != 1:
        raise RuntimeError("common encoder/role-head initial state differs by operator")
    for role, hashes in role_head_hashes.items():
        if len(hashes) != 1:
            raise RuntimeError(f"{role} head initial state differs by operator")

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "device": str(device),
        "operators": results,
        "operator_pass_count": len(results),
        "role_positive_weights": role_weights,
        "common_initialization_equal_across_operators": True,
        "reference_b3_initialization_equal_across_operators": True,
        "scientific_training_authorized": True,
        "trainer_sha256": sha256_file(trainer_path),
        "loader_sha256": trainer.EXPECTED_LOADER_SHA,
        "b3_model_sha256": trainer.EXPECTED_B3_MODEL_SHA,
        "edge_index_sha256": trainer.EXPECTED_EDGE_SHA,
        "g1_aggregation_report_sha256": trainer.EXPECTED_G1_REPORT_SHA,
        "g2_aggregation_report_sha256": trainer.EXPECTED_G2_REPORT_SHA,
        "p2_test_directory_enumerated": False,
        "p2_test_tensors_deserialized": False,
        "architecture_selected": False,
        "quantization_performed": False,
        "rtl_generated": False,
    }
    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)
    lock = {
        "status": COMPLETE,
        "report_sha256": sha256_file(report_path),
        "trainer_sha256": sha256_file(trainer_path),
        "loader_sha256": trainer.EXPECTED_LOADER_SHA,
        "b3_model_sha256": trainer.EXPECTED_B3_MODEL_SHA,
        "edge_index_sha256": trainer.EXPECTED_EDGE_SHA,
        "operator_pass_count": len(results),
        "scientific_training_authorized": True,
        "p2_test_directory_enumerated": False,
        "p2_test_tensors_deserialized": False,
        "architecture_selected": False,
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-G3 CLEAN TRAINING IMPLEMENTATION PREFLIGHT =====")
    print("status: COMPLETE")
    print(f"device: {device}")
    print(f"operator_pass_count: {len(results)}")
    for operator in OPERATORS:
        result = results[operator]
        print(
            f"{operator} PASS "
            f"train_loss={result['train_loss']:.8f} "
            f"val_loss={result['validation_loss']:.8f} "
            f"params={result['parameter_count']}"
        )
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
    parser.add_argument("--g1-aggregation-dir", type=Path, required=True)
    parser.add_argument("--g2-aggregation-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--b3-model-path", type=Path, required=True)
    parser.add_argument("--trainer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"HOLD: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
