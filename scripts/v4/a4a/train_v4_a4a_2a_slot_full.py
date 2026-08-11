#!/usr/bin/env python3
"""
V4-A4a-2A
Primary A4a-Slot-Full Trainer and Source Audit

Modes
-----
source-audit
    Verify provenance, model structure, trainable parameters, synthetic
    forward/backward behavior, frozen gradients and the batch-512 resource
    authorization. No dataset is opened.

train
    Train the decoder-only A4a-Slot-Full model under the frozen A4a-1C
    contract:
      * frozen A3 temporal/local/graph representation and graph head
      * 1,076 trainable slot-decoder parameters
      * batch size 512
      * AdamW, lr 1e-3, weight decay 1e-4
      * gradient clipping at norm 5
      * maximum 75 epochs, patience 10, minimum delta 1e-4
      * full validation at epoch 1, every 3 epochs and patience boundaries
      * early-stop monitor: minimum full-validation total slot loss

Every full-validation checkpoint is retained for the later A4a-2B
validation-only model-selection stage. No exact set decoding, threshold search
or development-test access occurs during optimizer training.
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
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from v4_a4a_slot_model import (
    A4aFrozenSlotModel,
    NULL_CLASS,
    NUM_ROUTERS,
    SlotLossWeights,
    count_parameters,
)


EXPECTED_TRAINER_SHA = (
    "e5f9a48e9f7ca873c351743df92e3bb0"
    "0d00a893333434c7fd0e04734e010525"
)
EXPECTED_MODEL_SOURCE_SHA = (
    "5ebcf80385b95f329faf935cad8039b64"
    "afd79f464903308eb07e674d78a1704"
)
EXPECTED_BASE_CHECKPOINT_SHA = (
    "f61c1add6c057f7f53dd34fb1f9f4e95"
    "b01cefd5053e0e42bc100c76dac7f923"
)
EXPECTED_SLOT_MODULE_SHA = (
    "97b56aabb3239dfc7543c241665cacf99"
    "8eb6b1a1f5f67f931cdeb70e75c111c"
)
EXPECTED_SMOKE_SCRIPT_SHA = (
    "a1c93ed64198c0bd946e64da116b3438"
    "012d742c0c5fafc092e8517b4761a35d"
)
EXPECTED_CONTRACT_FREEZE_SCRIPT_SHA = (
    "b5b05b2200a41484949a3a813323d80a"
    "08bce6c6957999679890ef4cd3dbbfab"
)
EXPECTED_RESOURCE_PROBE_SCRIPT_SHA = (
    "479b290ad97b0d1ceb4424c6e343f319"
    "bc31ff64e53981f98a5ce23f5e34dbb4"
)

EXPECTED_TOTAL_PARAMETERS = 3189
EXPECTED_TRAINABLE_PARAMETERS = 1076
EXPECTED_BATCH_SIZE = 512
EXPECTED_MAX_EPOCHS = 75
EXPECTED_PATIENCE = 10
EXPECTED_FULL_VAL_EVERY = 3


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if (
                isinstance(value, dict)
                and value
                and all(isinstance(v, torch.Tensor) for v in value.values())
            ):
                return value
    raise RuntimeError("checkpoint model_state_dict not found")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    observed: set[str] = set()
    for row in rows:
        for key in row:
            if key not in observed:
                observed.add(key)
                fieldnames.append(key)

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def loader_generator(loader) -> torch.Generator | None:
    direct = getattr(loader, "generator", None)
    if isinstance(direct, torch.Generator):
        return direct

    sampler = getattr(loader, "sampler", None)
    sampler_generator = getattr(sampler, "generator", None)
    if isinstance(sampler_generator, torch.Generator):
        return sampler_generator

    batch_sampler = getattr(loader, "batch_sampler", None)
    nested_sampler = getattr(batch_sampler, "sampler", None)
    nested_generator = getattr(nested_sampler, "generator", None)
    if isinstance(nested_generator, torch.Generator):
        return nested_generator

    return None


def capture_rng_state(train_loader) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "train_loader_generator": None,
    }
    generator = loader_generator(train_loader)
    if generator is not None:
        state["train_loader_generator"] = generator.get_state()
    return state


def restore_rng_state(
    state: dict[str, Any],
    train_loader,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    if (
        torch.cuda.is_available()
        and state.get("torch_cuda") is not None
    ):
        torch.cuda.set_rng_state_all(state["torch_cuda"])

    generator_state = state.get("train_loader_generator")
    generator = loader_generator(train_loader)
    if generator_state is not None and generator is not None:
        generator.set_state(generator_state)


def frozen_snapshot(model: A4aFrozenSlotModel) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.encoder.state_dict().items()
    }


def compare_frozen(
    before: dict[str, torch.Tensor],
    model: A4aFrozenSlotModel,
) -> dict[str, Any]:
    changed: dict[str, float] = {}
    maximum = 0.0

    for name, before_tensor in before.items():
        after_tensor = model.encoder.state_dict()[name].detach().cpu()
        difference = float(
            torch.max(torch.abs(before_tensor - after_tensor)).item()
        )
        maximum = max(maximum, difference)
        if difference != 0.0:
            changed[name] = difference

    return {
        "unchanged": not changed,
        "maximum_absolute_difference": maximum,
        "changed_tensors": changed,
    }


def flatten_metrics(
    prefix: str,
    value: Any,
    output: dict[str, Any],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}_{key}" if prefix else str(key)
            flatten_metrics(child_prefix, child, output)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        output[prefix] = value
    else:
        output[prefix] = json.dumps(value, sort_keys=True)


class CheapMetricAccumulator:
    """Threshold-free training diagnostics plus membership@0.5."""

    def __init__(self) -> None:
        self.samples = 0
        self.attack_samples = 0
        self.normal_samples = 0

        self.membership_tp = 0
        self.membership_fp = 0
        self.membership_fn = 0
        self.membership_exact = 0

        self.cardinality_correct = 0
        self.raw_duplicate_samples = 0
        self.raw_all_null_samples = 0

        self.slot_probability_count = 0
        self.slot_null_probability_sum = 0.0
        self.membership_probability_sum = 0.0
        self.membership_probability_count = 0

    def update(
        self,
        slot_logits: torch.Tensor,
        membership_probabilities: torch.Tensor,
        cardinality_probabilities: torch.Tensor,
        y_node: torch.Tensor,
        attacker_count: torch.Tensor,
    ) -> None:
        truth = y_node > 0.5
        membership_prediction = membership_probabilities >= 0.5

        self.samples += int(y_node.shape[0])
        count_cpu = attacker_count.detach().cpu().numpy()
        self.attack_samples += int(np.sum(count_cpu > 0))
        self.normal_samples += int(np.sum(count_cpu == 0))

        self.membership_tp += int(
            torch.logical_and(
                membership_prediction,
                truth,
            ).sum().item()
        )
        self.membership_fp += int(
            torch.logical_and(
                membership_prediction,
                ~truth,
            ).sum().item()
        )
        self.membership_fn += int(
            torch.logical_and(
                ~membership_prediction,
                truth,
            ).sum().item()
        )
        self.membership_exact += int(
            torch.all(
                membership_prediction == truth,
                dim=1,
            ).sum().item()
        )

        predicted_count = cardinality_probabilities.argmax(dim=1)
        self.cardinality_correct += int(
            (predicted_count == attacker_count).sum().item()
        )

        raw_assignment = slot_logits.argmax(dim=-1).detach().cpu().numpy()
        for row in raw_assignment:
            real = [
                int(value)
                for value in row
                if int(value) != NULL_CLASS
            ]
            self.raw_duplicate_samples += int(
                len(real) != len(set(real))
            )
            self.raw_all_null_samples += int(not real)

        slot_probabilities = torch.softmax(slot_logits, dim=-1)
        self.slot_null_probability_sum += float(
            slot_probabilities[..., NULL_CLASS].sum().item()
        )
        self.slot_probability_count += int(
            slot_probabilities.shape[0]
            * slot_probabilities.shape[1]
        )
        self.membership_probability_sum += float(
            membership_probabilities.sum().item()
        )
        self.membership_probability_count += int(
            membership_probabilities.numel()
        )

    def finalize(self) -> dict[str, Any]:
        precision = (
            self.membership_tp
            / (self.membership_tp + self.membership_fp)
            if self.membership_tp + self.membership_fp
            else 1.0
        )
        recall = (
            self.membership_tp
            / (self.membership_tp + self.membership_fn)
            if self.membership_tp + self.membership_fn
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        return {
            "samples": self.samples,
            "attack_samples": self.attack_samples,
            "normal_samples": self.normal_samples,
            "membership_threshold": 0.5,
            "membership_precision": precision,
            "membership_recall": recall,
            "membership_f1": f1,
            "membership_exact_fraction": (
                self.membership_exact / self.samples
                if self.samples
                else 0.0
            ),
            "cardinality_accuracy": (
                self.cardinality_correct / self.samples
                if self.samples
                else 0.0
            ),
            "raw_duplicate_proposal_fraction": (
                self.raw_duplicate_samples / self.samples
                if self.samples
                else 0.0
            ),
            "raw_all_null_proposal_fraction": (
                self.raw_all_null_samples / self.samples
                if self.samples
                else 0.0
            ),
            "mean_slot_null_probability": (
                self.slot_null_probability_sum
                / self.slot_probability_count
                if self.slot_probability_count
                else 0.0
            ),
            "mean_router_membership_probability": (
                self.membership_probability_sum
                / self.membership_probability_count
                if self.membership_probability_count
                else 0.0
            ),
            "membership_tp": self.membership_tp,
            "membership_fp": self.membership_fp,
            "membership_fn": self.membership_fn,
        }


def run_epoch(
    model: A4aFrozenSlotModel,
    loader,
    adjacency: torch.Tensor,
    physical_mask: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    smoke_module,
    *,
    null_weight: float,
    loss_weights: SlotLossWeights,
    gradient_clip_norm: float,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    training = optimizer is not None
    model.train(training)

    totals = {
        "total": 0.0,
        "matching": 0.0,
        "membership": 0.0,
        "cardinality": 0.0,
        "duplicate": 0.0,
    }
    samples = 0
    batches = 0
    gradient_norm_sum = 0.0
    gradient_norm_max = 0.0
    metric = CheapMetricAccumulator()
    logit_min = float("inf")
    logit_max = -float("inf")

    for batch in loader:
        batches += 1
        x = batch["x"].to(
            device,
            non_blocking=True,
            dtype=torch.float32,
        )
        y_node = batch["y_node"].to(
            device,
            non_blocking=True,
            dtype=torch.float32,
        )
        attacker_count = batch["attacker_count"].to(
            device,
            non_blocking=True,
            dtype=torch.long,
        )

        with torch.set_grad_enabled(training):
            output = model(x, adjacency, physical_mask)
            losses = smoke_module.compute_smoke_losses(
                output["slot_logits"],
                y_node,
                null_weight=null_weight,
                weights=loss_weights,
            )

            for name, value in losses.items():
                if not torch.isfinite(value):
                    raise FloatingPointError(
                        f"non-finite {name} loss at batch {batches}"
                    )

            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()

                for name, parameter in model.encoder.named_parameters():
                    if parameter.grad is not None:
                        raise RuntimeError(
                            f"frozen encoder gradient appeared in {name}"
                        )

                for name, parameter in model.slot_decoder.named_parameters():
                    if parameter.grad is None:
                        raise RuntimeError(
                            f"missing gradient in trainable parameter {name}"
                        )
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(
                            f"non-finite gradient in {name}"
                        )

                norm = torch.nn.utils.clip_grad_norm_(
                    model.slot_decoder.parameters(),
                    max_norm=gradient_clip_norm,
                )
                norm_value = float(norm.detach().cpu())
                if not math.isfinite(norm_value):
                    raise FloatingPointError(
                        f"non-finite gradient norm at batch {batches}"
                    )
                gradient_norm_sum += norm_value
                gradient_norm_max = max(
                    gradient_norm_max,
                    norm_value,
                )
                optimizer.step()

        batch_size = int(x.shape[0])
        samples += batch_size
        for name, value in losses.items():
            totals[name] += float(value.detach().cpu()) * batch_size

        metric.update(
            output["slot_logits"].detach(),
            output["membership_probabilities"].detach(),
            output["cardinality_probabilities"].detach(),
            y_node,
            attacker_count,
        )
        logit_min = min(
            logit_min,
            float(output["slot_logits"].detach().min().cpu()),
        )
        logit_max = max(
            logit_max,
            float(output["slot_logits"].detach().max().cpu()),
        )

    if samples == 0:
        raise RuntimeError("epoch processed zero samples")

    losses_average = {
        name: value / samples
        for name, value in totals.items()
    }
    extra = {
        "batches": batches,
        "samples": samples,
        "gradient_norm_before_clip_mean": (
            gradient_norm_sum / batches
            if training and batches
            else None
        ),
        "gradient_norm_before_clip_max": (
            gradient_norm_max if training else None
        ),
        "gradient_clip_norm": (
            gradient_clip_norm if training else None
        ),
        "slot_logit_min": logit_min,
        "slot_logit_max": logit_max,
    }
    return losses_average, metric.finalize(), extra


def verify_common(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
]:
    contract_path = (
        args.contract_dir / "A4A_1C_FULL_TRAINING_CONTRACT.json"
    )
    contract_lock_path = args.contract_dir / "A4A_1C_LOCK.json"
    contract_pass_path = (
        args.contract_dir / "V4_A4A_1C_FULL_TRAINING_CONTRACT_PASS"
    )
    probe_report_path = (
        args.probe_dir / "batch512_resource_probe.json"
    )
    probe_lock_path = args.probe_dir / "A4A_1D_LOCK.json"
    probe_pass_path = (
        args.probe_dir
        / "V4_A4A_1D_BATCH512_RESOURCE_PROBE_PASS"
    )

    required = [
        args.repo,
        args.trainer,
        args.model_source,
        args.base_checkpoint,
        args.slot_module,
        args.smoke_script,
        args.contract_freeze_script,
        args.resource_probe_script,
        contract_path,
        contract_lock_path,
        contract_pass_path,
        probe_report_path,
        probe_lock_path,
        probe_pass_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing required paths:\n  " + "\n  ".join(missing)
        )

    hashes = {
        "trainer_sha256": sha256_file(args.trainer),
        "model_source_sha256": sha256_file(args.model_source),
        "base_checkpoint_sha256": sha256_file(
            args.base_checkpoint
        ),
        "slot_module_sha256": sha256_file(args.slot_module),
        "smoke_script_sha256": sha256_file(args.smoke_script),
        "contract_freeze_script_sha256": sha256_file(
            args.contract_freeze_script
        ),
        "resource_probe_script_sha256": sha256_file(
            args.resource_probe_script
        ),
        "full_training_contract_sha256": sha256_file(
            contract_path
        ),
        "full_training_contract_lock_sha256": sha256_file(
            contract_lock_path
        ),
        "resource_probe_report_sha256": sha256_file(
            probe_report_path
        ),
        "resource_probe_lock_sha256": sha256_file(
            probe_lock_path
        ),
        "primary_trainer_sha256": sha256_file(Path(__file__)),
    }
    expected = {
        "trainer_sha256": EXPECTED_TRAINER_SHA,
        "model_source_sha256": EXPECTED_MODEL_SOURCE_SHA,
        "base_checkpoint_sha256": EXPECTED_BASE_CHECKPOINT_SHA,
        "slot_module_sha256": EXPECTED_SLOT_MODULE_SHA,
        "smoke_script_sha256": EXPECTED_SMOKE_SCRIPT_SHA,
        "contract_freeze_script_sha256": (
            EXPECTED_CONTRACT_FREEZE_SCRIPT_SHA
        ),
        "resource_probe_script_sha256": (
            EXPECTED_RESOURCE_PROBE_SCRIPT_SHA
        ),
    }
    mismatch = [
        f"{key}: {hashes[key]} != {expected_value}"
        for key, expected_value in expected.items()
        if hashes[key] != expected_value
    ]
    if mismatch:
        raise RuntimeError(
            "provenance hash mismatch:\n  " + "\n  ".join(mismatch)
        )

    contract = load_json(contract_path)
    contract_lock = load_json(contract_lock_path)
    probe = load_json(probe_report_path)
    probe_lock = load_json(probe_lock_path)

    if contract.get("status") != "PASS":
        raise RuntimeError("full-training contract is not PASS")
    if contract_lock.get("status") != (
        "A4A_1C_FULL_TRAINING_CONTRACT_COMPLETE"
    ):
        raise RuntimeError("full-training contract lock is incomplete")
    if contract_lock.get("contract_sha256") != hashes[
        "full_training_contract_sha256"
    ]:
        raise RuntimeError("contract hash differs from contract lock")

    if probe.get("status") != "PASS":
        raise RuntimeError("batch-512 resource probe is not PASS")
    if probe.get("decision") != "batch512_primary_authorized":
        raise RuntimeError("batch 512 is not authorized")
    if probe.get(
        "ready_for_a4a_2a_primary_training_implementation"
    ) is not True:
        raise RuntimeError("resource probe does not authorize A4a-2A")
    if probe_lock.get("status") != (
        "A4A_1D_BATCH512_RESOURCE_PROBE_COMPLETE"
    ):
        raise RuntimeError("resource-probe lock is incomplete")
    if probe_lock.get("resource_probe_report_sha256") != hashes[
        "resource_probe_report_sha256"
    ]:
        raise RuntimeError("probe report hash differs from probe lock")

    for stage_name, payload in (
        ("contract", contract),
        ("resource probe", probe),
    ):
        if payload.get("development_test_accessed") is not False:
            raise RuntimeError(
                f"{stage_name} does not affirm no development-test access"
            )

    runtime = contract["runtime_contract"]
    frozen_runtime_checks = {
        "batch_size_primary": (
            runtime["batch_size_primary"] == EXPECTED_BATCH_SIZE
        ),
        "maximum_epochs": (
            runtime["maximum_epochs"] == EXPECTED_MAX_EPOCHS
        ),
        "patience": (
            runtime["early_stopping_patience_epochs"]
            == EXPECTED_PATIENCE
        ),
        "full_validation_every": (
            runtime["full_validation_every_epochs"]
            == EXPECTED_FULL_VAL_EVERY
        ),
        "seed": runtime["seed"] == 7,
        "optimizer": runtime["optimizer"] == "AdamW",
        "amp": runtime["automatic_mixed_precision"] is False,
    }
    if not all(frozen_runtime_checks.values()):
        raise RuntimeError(
            f"frozen runtime mismatch: {frozen_runtime_checks}"
        )

    return contract, contract_lock, probe, hashes


def import_project_modules(args: argparse.Namespace):
    for import_root in (
        args.repo,
        args.repo / "scripts",
        args.trainer.parent,
        args.model_source.parent,
        args.slot_module.parent,
        args.smoke_script.parent,
    ):
        text = str(import_root.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)

    trainer_module = load_module(
        args.trainer,
        "resolved_v4_a3_trainer_for_a4a_2a",
    )
    a3_module = load_module(
        args.model_source,
        "resolved_v4_a3_model_for_a4a_2a",
    )
    smoke_module = load_module(
        args.smoke_script,
        "resolved_a4a_1b_smoke_for_a4a_2a",
    )
    return trainer_module, a3_module, smoke_module


def make_model(
    a3_module,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[A4aFrozenSlotModel, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    base_model = a3_module.A3SourcePreserveModel()
    base_model.load_state_dict(
        extract_state_dict(checkpoint),
        strict=True,
    )
    base_model.eval()

    model = A4aFrozenSlotModel(base_model).to(device)
    return model, checkpoint


def source_audit(args: argparse.Namespace) -> int:
    if args.out_dir.exists():
        raise RuntimeError(
            f"source-audit output already exists: {args.out_dir}"
        )
    args.out_dir.mkdir(parents=True)

    contract, _, probe, hashes = verify_common(args)
    trainer_module, a3_module, smoke_module = import_project_modules(args)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, checkpoint = make_model(
        a3_module,
        args.base_checkpoint,
        device,
    )
    model.train(True)

    parameters = count_parameters(model)
    if parameters["total"] != EXPECTED_TOTAL_PARAMETERS:
        raise RuntimeError(f"unexpected total parameters: {parameters}")
    if parameters["trainable"] != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(f"unexpected trainable parameters: {parameters}")

    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    unexpected_trainable = [
        name
        for name in trainable_names
        if not name.startswith("slot_decoder.")
    ]
    if unexpected_trainable:
        raise RuntimeError(
            f"unexpected trainable parameters: {unexpected_trainable}"
        )

    x = torch.zeros(
        4,
        16,
        8,
        24,
        dtype=torch.float32,
        device=device,
    )
    y_node = torch.zeros(
        4,
        NUM_ROUTERS,
        dtype=torch.float32,
        device=device,
    )
    y_node[1, 3] = 1
    y_node[2, [3, 7]] = 1
    y_node[3, [1, 3, 7, 12]] = 1

    adjacency = checkpoint["A_hat"].to(device)
    physical_mask = checkpoint[
        "physical_valid_port_mask"
    ].to(device)

    loss_contract = contract["loss_contract"]
    weights = SlotLossWeights(
        matching=float(
            loss_contract["permutation_invariant_matching_weight"]
        ),
        membership=float(
            loss_contract["derived_union_membership_bce_weight"]
        ),
        cardinality=float(
            loss_contract[
                "derived_cardinality_cross_entropy_weight"
            ]
        ),
        duplicate=float(
            loss_contract["duplicate_slot_penalty_weight"]
        ),
    )
    output = model(x, adjacency, physical_mask)
    losses = smoke_module.compute_smoke_losses(
        output["slot_logits"],
        y_node,
        null_weight=float(loss_contract["null_target_weight"]),
        weights=weights,
    )
    losses["total"].backward()

    frozen_gradient_violations = [
        name
        for name, parameter in model.encoder.named_parameters()
        if parameter.grad is not None
    ]
    missing_trainable_gradients = [
        name
        for name, parameter in model.slot_decoder.named_parameters()
        if parameter.grad is None
    ]
    nonfinite_trainable_gradients = [
        name
        for name, parameter in model.slot_decoder.named_parameters()
        if (
            parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
        )
    ]

    if frozen_gradient_violations:
        raise RuntimeError(
            f"frozen gradient violations: {frozen_gradient_violations}"
        )
    if missing_trainable_gradients:
        raise RuntimeError(
            f"missing trainable gradients: {missing_trainable_gradients}"
        )
    if nonfinite_trainable_gradients:
        raise RuntimeError(
            f"non-finite trainable gradients: "
            f"{nonfinite_trainable_gradients}"
        )

    audit = {
        "status": "PASS",
        "designation": "V4-A4a-2A Primary Trainer Source Audit",
        "device": str(device),
        "parameter_count": parameters,
        "trainable_parameter_names": trainable_names,
        "synthetic_output_shapes": {
            key: list(value.shape)
            for key, value in output.items()
            if isinstance(value, torch.Tensor)
        },
        "synthetic_losses": {
            key: float(value.detach().cpu())
            for key, value in losses.items()
        },
        "frozen_gradient_violations": frozen_gradient_violations,
        "missing_trainable_gradients": missing_trainable_gradients,
        "nonfinite_trainable_gradients": (
            nonfinite_trainable_gradients
        ),
        "batch512_resource_probe_decision": probe["decision"],
        "runtime_contract": contract["runtime_contract"],
        "loss_contract": contract["loss_contract"],
        "training_performed": False,
        "dataset_accessed": False,
        "validation_accessed": False,
        "test_loader_constructed": False,
        "development_test_accessed": False,
        "ready_for_a4a_2a_primary_training": True,
        "provenance": hashes,
    }

    audit_path = args.out_dir / "source_audit.json"
    atomic_json(audit_path, audit)
    lock = {
        "status": "A4A_2A_PRIMARY_TRAINER_SOURCE_AUDIT_COMPLETE",
        "source_audit_sha256": sha256_file(audit_path),
        "primary_trainer_sha256": hashes["primary_trainer_sha256"],
        "full_training_contract_sha256": hashes[
            "full_training_contract_sha256"
        ],
        "resource_probe_report_sha256": hashes[
            "resource_probe_report_sha256"
        ],
        "training_performed": False,
        "development_test_accessed": False,
    }
    atomic_json(args.out_dir / "A4A_2A_SOURCE_AUDIT_LOCK.json", lock)
    (
        args.out_dir / "V4_A4A_2A_PRIMARY_TRAINER_SOURCE_AUDIT_PASS"
    ).write_text(
        "V4_A4A_2A_PRIMARY_TRAINER_SOURCE_AUDIT_PASS\n",
        encoding="utf-8",
    )

    print(json.dumps(audit, indent=2, sort_keys=True))
    print("V4_A4A_2A_PRIMARY_TRAINER_SOURCE_AUDIT_PASS")
    return 0


def immutable_config(
    contract: dict[str, Any],
    hashes: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    runtime = contract["runtime_contract"]
    loss_contract = contract["loss_contract"]
    return {
        "designation": "V4-A4a-2A A4a-Slot-Full Primary Training",
        "data_dir": str(args.data_dir.resolve()),
        "seed": runtime["seed"],
        "maximum_epochs": runtime["maximum_epochs"],
        "patience": runtime["early_stopping_patience_epochs"],
        "minimum_delta": runtime["minimum_delta"],
        "full_validation_every_epochs": (
            runtime["full_validation_every_epochs"]
        ),
        "optimizer": runtime["optimizer"],
        "learning_rate": runtime["learning_rate"],
        "weight_decay": runtime["weight_decay"],
        "gradient_clip_norm": runtime["gradient_clip_norm"],
        "batch_size": runtime["batch_size_primary"],
        "num_workers": runtime["num_workers"],
        "pin_memory": runtime["pin_memory"],
        "persistent_workers": runtime["persistent_workers"],
        "prefetch_factor": runtime["prefetch_factor"],
        "automatic_mixed_precision": (
            runtime["automatic_mixed_precision"]
        ),
        "loss_contract": loss_contract,
        "expected_total_parameters": EXPECTED_TOTAL_PARAMETERS,
        "expected_trainable_parameters": (
            EXPECTED_TRAINABLE_PARAMETERS
        ),
        "full_training_contract_sha256": hashes[
            "full_training_contract_sha256"
        ],
        "resource_probe_report_sha256": hashes[
            "resource_probe_report_sha256"
        ],
        "primary_trainer_sha256": hashes[
            "primary_trainer_sha256"
        ],
    }


def validate_source_audit(
    source_audit_dir: Path,
    hashes: dict[str, str],
) -> dict[str, Any]:
    marker = (
        source_audit_dir
        / "V4_A4A_2A_PRIMARY_TRAINER_SOURCE_AUDIT_PASS"
    )
    report_path = source_audit_dir / "source_audit.json"
    lock_path = source_audit_dir / "A4A_2A_SOURCE_AUDIT_LOCK.json"

    for path in (marker, report_path, lock_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing source-audit artifact: {path}")

    report = load_json(report_path)
    lock = load_json(lock_path)

    if report.get("status") != "PASS":
        raise RuntimeError("source audit is not PASS")
    if report.get("ready_for_a4a_2a_primary_training") is not True:
        raise RuntimeError("source audit does not authorize training")
    if lock.get("status") != (
        "A4A_2A_PRIMARY_TRAINER_SOURCE_AUDIT_COMPLETE"
    ):
        raise RuntimeError("source-audit lock is incomplete")
    if lock.get("source_audit_sha256") != sha256_file(report_path):
        raise RuntimeError("source-audit report hash differs from lock")
    if lock.get("primary_trainer_sha256") != hashes[
        "primary_trainer_sha256"
    ]:
        raise RuntimeError("primary trainer changed after source audit")

    return report


def training(args: argparse.Namespace) -> int:
    if args.data_dir is None:
        raise ValueError("--data-dir is required for train mode")
    if args.source_audit_dir is None:
        raise ValueError("--source-audit-dir is required for train mode")

    contract, _, _, hashes = verify_common(args)
    validate_source_audit(args.source_audit_dir, hashes)
    trainer_module, a3_module, smoke_module = import_project_modules(args)

    runtime = contract["runtime_contract"]
    loss_contract = contract["loss_contract"]
    frozen_config = immutable_config(contract, hashes, args)

    if args.resume:
        required = [
            args.out_dir / "config.json",
            args.out_dir / "last_model.pt",
            args.out_dir / "training_history.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "resume artifacts missing:\n  " + "\n  ".join(missing)
            )
        stored_config = load_json(args.out_dir / "config.json")
        if stored_config.get("immutable_config") != frozen_config:
            raise RuntimeError(
                "resume configuration differs from frozen configuration"
            )
    else:
        if args.out_dir.exists() and any(args.out_dir.iterdir()):
            raise RuntimeError(
                f"training output is non-empty: {args.out_dir}"
            )
        args.out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(runtime["seed"]))

    loader_args = argparse.Namespace(
        data_dir=str(args.data_dir),
        batch_size=int(runtime["batch_size_primary"]),
        num_workers=int(runtime["num_workers"]),
        pin_memory=bool(runtime["pin_memory"]),
        persistent_workers=bool(runtime["persistent_workers"]),
        prefetch_factor=int(runtime["prefetch_factor"]),
        seed=int(runtime["seed"]),
    )
    objects = trainer_module.prepare_objects(
        loader_args,
        include_val=True,
    )
    if objects["val_loader"] is None:
        raise RuntimeError("validation loader was not constructed")

    device = objects["device"]
    if device.type != "cuda":
        raise RuntimeError(
            f"authorized primary training requires CUDA, got {device}"
        )

    unused_original_model = objects.pop("model", None)
    del unused_original_model
    torch.cuda.empty_cache()

    model, base_checkpoint = make_model(
        a3_module,
        args.base_checkpoint,
        device,
    )
    parameters = count_parameters(model)
    if parameters["total"] != EXPECTED_TOTAL_PARAMETERS:
        raise RuntimeError(f"unexpected total parameters: {parameters}")
    if parameters["trainable"] != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(f"unexpected trainable parameters: {parameters}")

    adjacency = objects["adjacency"].to(device)
    physical_mask = objects["mask"].to(device)
    if not torch.equal(
        adjacency.detach().cpu(),
        base_checkpoint["A_hat"].detach().cpu(),
    ):
        raise RuntimeError("adjacency differs from frozen checkpoint")
    if not torch.equal(
        physical_mask.detach().cpu(),
        base_checkpoint[
            "physical_valid_port_mask"
        ].detach().cpu(),
    ):
        raise RuntimeError("physical mask differs from frozen checkpoint")

    loss_weights = SlotLossWeights(
        matching=float(
            loss_contract["permutation_invariant_matching_weight"]
        ),
        membership=float(
            loss_contract["derived_union_membership_bce_weight"]
        ),
        cardinality=float(
            loss_contract[
                "derived_cardinality_cross_entropy_weight"
            ]
        ),
        duplicate=float(
            loss_contract["duplicate_slot_penalty_weight"]
        ),
    )
    null_weight = float(loss_contract["null_target_weight"])
    gradient_clip_norm = float(runtime["gradient_clip_norm"])

    optimizer = torch.optim.AdamW(
        model.slot_decoder.parameters(),
        lr=float(runtime["learning_rate"]),
        weight_decay=float(runtime["weight_decay"]),
    )

    start_epoch = 1
    best_validation_loss = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    last_full_validation: dict[str, Any] | None = None

    if args.resume:
        checkpoint = torch.load(
            args.out_dir / "last_model.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(
            checkpoint["best_validation_loss"]
        )
        best_epoch = int(checkpoint["best_epoch"])
        last_full_validation = checkpoint[
            "last_full_validation"
        ]
        history = load_json(
            args.out_dir / "training_history.json"
        )
        restore_rng_state(
            checkpoint["rng_state"],
            objects["train_loader"],
        )
        print(
            f"RESUME epoch={start_epoch} "
            f"best_epoch={best_epoch} "
            f"best_val_loss={best_validation_loss:.8f}",
            flush=True,
        )
    else:
        config = {
            "immutable_config": frozen_config,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "parameter_count": parameters,
            "source_audit_dir": str(args.source_audit_dir.resolve()),
            "source_audit_sha256": sha256_file(
                args.source_audit_dir / "source_audit.json"
            ),
            "test_loader_constructed": False,
            "test_evaluated": False,
            "development_test_accessed": False,
        }
        atomic_json(args.out_dir / "config.json", config)
        atomic_json(
            args.out_dir / "split_counts.json",
            {
                "train": int(len(objects["splits"]["train"])),
                "validation": int(len(objects["splits"]["val"])),
            },
        )
        checkpoints_dir = args.out_dir / "validation_checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=False)

    maximum_epochs = int(runtime["maximum_epochs"])
    patience = int(runtime["early_stopping_patience_epochs"])
    minimum_delta = float(runtime["minimum_delta"])
    full_val_every = int(runtime["full_validation_every_epochs"])

    if start_epoch > maximum_epochs:
        raise RuntimeError(
            f"resume start epoch {start_epoch} exceeds {maximum_epochs}"
        )

    frozen_before = frozen_snapshot(model)
    stopped_early = False

    for epoch in range(start_epoch, maximum_epochs + 1):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)

        train_losses, train_metrics, train_extra = run_epoch(
            model,
            objects["train_loader"],
            adjacency,
            physical_mask,
            device,
            optimizer,
            smoke_module,
            null_weight=null_weight,
            loss_weights=loss_weights,
            gradient_clip_norm=gradient_clip_norm,
        )

        patience_boundary_due = (
            best_epoch >= 1
            and (epoch - best_epoch) >= patience
        )
        full_validation_performed = (
            epoch == 1
            or epoch % full_val_every == 0
            or patience_boundary_due
        )

        validation_losses = None
        validation_metrics = None
        validation_extra = None
        improved = False

        if full_validation_performed:
            validation_losses, validation_metrics, validation_extra = (
                run_epoch(
                    model,
                    objects["val_loader"],
                    adjacency,
                    physical_mask,
                    device,
                    None,
                    smoke_module,
                    null_weight=null_weight,
                    loss_weights=loss_weights,
                    gradient_clip_norm=gradient_clip_norm,
                )
            )
            current_loss = float(validation_losses["total"])
            improved = (
                current_loss
                < best_validation_loss - minimum_delta
            )
            if improved:
                best_validation_loss = current_loss
                best_epoch = epoch

            validation_checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "A_hat": adjacency.detach().cpu(),
                "physical_valid_port_mask": physical_mask.detach().cpu(),
                "parameter_count": parameters,
                "validation_losses": validation_losses,
                "validation_metrics": validation_metrics,
                "validation_extra": validation_extra,
                "best_validation_loss_after_epoch": (
                    best_validation_loss
                ),
                "best_epoch_after_epoch": best_epoch,
                "improved_loss_monitor": improved,
                "immutable_config": frozen_config,
                "test_loader_constructed": False,
                "test_evaluated": False,
                "development_test_accessed": False,
            }
            validation_path = (
                args.out_dir
                / "validation_checkpoints"
                / f"epoch_{epoch:03d}.pt"
            )
            trainer_module.atomic_torch_save(
                validation_path,
                validation_checkpoint,
            )

            if improved:
                trainer_module.atomic_torch_save(
                    args.out_dir / "best_loss_model.pt",
                    validation_checkpoint,
                )

            last_full_validation = {
                "epoch": epoch,
                "validation_losses": validation_losses,
                "validation_metrics": validation_metrics,
                "validation_extra": validation_extra,
                "improved": improved,
                "checkpoint_path": str(validation_path),
                "checkpoint_sha256": sha256_file(validation_path),
            }

        frozen_comparison = compare_frozen(frozen_before, model)
        if frozen_comparison["unchanged"] is not True:
            raise RuntimeError(
                f"frozen encoder changed at epoch {epoch}: "
                f"{frozen_comparison}"
            )

        duration = time.perf_counter() - started
        gpu_peak = (
            torch.cuda.max_memory_allocated(device) / 1024**2
        )
        epochs_since_best = (
            0
            if best_epoch < 1
            else max(0, epoch - best_epoch)
        )

        record = {
            "epoch": epoch,
            "duration_seconds": duration,
            "full_validation_performed": (
                full_validation_performed
            ),
            "best_epoch_after_epoch": best_epoch,
            "best_validation_loss_after_epoch": (
                best_validation_loss
            ),
            "epochs_since_best": epochs_since_best,
            "improved_loss_monitor": improved,
            "gpu_peak_allocated_mib": gpu_peak,
            "train_losses": train_losses,
            "train_metrics": train_metrics,
            "train_extra": train_extra,
            "validation_losses": validation_losses,
            "validation_metrics": validation_metrics,
            "validation_extra": validation_extra,
        }
        history.append(record)

        last_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "last_full_validation": last_full_validation,
            "immutable_config": frozen_config,
            "rng_state": capture_rng_state(
                objects["train_loader"]
            ),
            "A_hat": adjacency.detach().cpu(),
            "physical_valid_port_mask": physical_mask.detach().cpu(),
            "parameter_count": parameters,
            "test_loader_constructed": False,
            "test_evaluated": False,
            "development_test_accessed": False,
        }
        trainer_module.atomic_torch_save(
            args.out_dir / "last_model.pt",
            last_checkpoint,
        )
        atomic_json(
            args.out_dir / "training_history.json",
            history,
        )

        flat_record: dict[str, Any] = {
            "epoch": epoch,
            "duration_seconds": duration,
            "full_validation_performed": (
                full_validation_performed
            ),
            "best_epoch_after_epoch": best_epoch,
            "best_validation_loss_after_epoch": (
                best_validation_loss
            ),
            "epochs_since_best": epochs_since_best,
            "improved_loss_monitor": improved,
            "gpu_peak_allocated_mib": gpu_peak,
        }
        flatten_metrics("train_losses", train_losses, flat_record)
        flatten_metrics("train_metrics", train_metrics, flat_record)
        flatten_metrics("train_extra", train_extra, flat_record)
        flatten_metrics(
            "validation_losses",
            validation_losses,
            flat_record,
        )
        flatten_metrics(
            "validation_metrics",
            validation_metrics,
            flat_record,
        )
        flatten_metrics(
            "validation_extra",
            validation_extra,
            flat_record,
        )

        csv_rows: list[dict[str, Any]] = []
        for history_record in history:
            row: dict[str, Any] = {
                "epoch": history_record["epoch"],
                "duration_seconds": history_record[
                    "duration_seconds"
                ],
                "full_validation_performed": history_record[
                    "full_validation_performed"
                ],
                "best_epoch_after_epoch": history_record[
                    "best_epoch_after_epoch"
                ],
                "best_validation_loss_after_epoch": history_record[
                    "best_validation_loss_after_epoch"
                ],
                "epochs_since_best": history_record[
                    "epochs_since_best"
                ],
                "improved_loss_monitor": history_record[
                    "improved_loss_monitor"
                ],
                "gpu_peak_allocated_mib": history_record[
                    "gpu_peak_allocated_mib"
                ],
            }
            flatten_metrics(
                "train_losses",
                history_record["train_losses"],
                row,
            )
            flatten_metrics(
                "train_metrics",
                history_record["train_metrics"],
                row,
            )
            flatten_metrics(
                "train_extra",
                history_record["train_extra"],
                row,
            )
            flatten_metrics(
                "validation_losses",
                history_record["validation_losses"],
                row,
            )
            flatten_metrics(
                "validation_metrics",
                history_record["validation_metrics"],
                row,
            )
            flatten_metrics(
                "validation_extra",
                history_record["validation_extra"],
                row,
            )
            csv_rows.append(row)

        atomic_csv(args.out_dir / "training_history.csv", csv_rows)
        atomic_json(
            args.out_dir / "latest_epoch.json",
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "epochs_since_best": epochs_since_best,
                "full_validation_performed": (
                    full_validation_performed
                ),
                "record": record,
            },
        )

        if full_validation_performed:
            print(
                f"epoch={epoch:03d} "
                f"duration={duration:.1f}s "
                f"train_loss={train_losses['total']:.6f} "
                f"val_loss={validation_losses['total']:.6f} "
                f"val_member_f1={validation_metrics['membership_f1']:.4f} "
                f"val_count_acc={validation_metrics['cardinality_accuracy']:.4f} "
                f"best_epoch={best_epoch} "
                f"epochs_since_best={epochs_since_best}/{patience} "
                f"improved={improved}",
                flush=True,
            )
        else:
            print(
                f"epoch={epoch:03d} "
                f"duration={duration:.1f}s "
                f"train_loss={train_losses['total']:.6f} "
                f"full_val=False "
                f"train_member_f1={train_metrics['membership_f1']:.4f} "
                f"train_count_acc={train_metrics['cardinality_accuracy']:.4f} "
                f"best_epoch={best_epoch} "
                f"epochs_since_best={epochs_since_best}/{patience}",
                flush=True,
            )

        if (
            full_validation_performed
            and best_epoch >= 1
            and epochs_since_best >= patience
        ):
            stopped_early = True
            print(f"EARLY_STOP epoch={epoch}", flush=True)
            break

    best_path = args.out_dir / "best_loss_model.pt"
    if not best_path.is_file():
        raise RuntimeError(
            "training completed without best_loss_model.pt"
        )

    completed_epoch = int(
        load_json(args.out_dir / "latest_epoch.json")["epoch"]
    )
    validation_checkpoints = sorted(
        (args.out_dir / "validation_checkpoints").glob(
            "epoch_*.pt"
        )
    )
    checkpoint_manifest = [
        {
            "filename": str(path.relative_to(args.out_dir)),
            "epoch": int(path.stem.split("_")[-1]),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in validation_checkpoints
    ]
    atomic_json(
        args.out_dir / "validation_checkpoint_manifest.json",
        checkpoint_manifest,
    )

    summary = {
        "status": "PASS",
        "designation": "V4-A4a-2A A4a-Slot-Full Primary Training",
        "primary_variant": "A4a-Slot-Full",
        "parameter_count": parameters,
        "epochs_completed": completed_epoch,
        "maximum_epochs": maximum_epochs,
        "stopped_early": stopped_early,
        "best_loss_epoch": best_epoch,
        "best_validation_total_loss": best_validation_loss,
        "best_loss_checkpoint_sha256": sha256_file(best_path),
        "validation_checkpoint_count": len(
            validation_checkpoints
        ),
        "early_stop_monitor": (
            "minimum full-validation total slot loss"
        ),
        "primary_model_selection_performed": False,
        "threshold_search_performed": False,
        "validation_exact_decoding_performed": False,
        "test_loader_constructed": False,
        "test_evaluated": False,
        "development_test_accessed": False,
        "next_stage": (
            "A4a-2B validation-only exact-decoder checkpoint selection"
        ),
        "provenance": hashes,
    }
    summary_path = args.out_dir / "summary.json"
    atomic_json(summary_path, summary)

    artifact_rows = []
    for path in sorted(args.out_dir.rglob("*")):
        if (
            path.is_file()
            and path.name != "artifact_manifest.csv"
        ):
            artifact_rows.append(
                {
                    "filename": str(path.relative_to(args.out_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    atomic_csv(
        args.out_dir / "artifact_manifest.csv",
        artifact_rows,
    )

    (
        args.out_dir / "V4_A4A_2A_PRIMARY_TRAINING_PASS"
    ).write_text(
        "V4_A4A_2A_PRIMARY_TRAINING_PASS\n",
        encoding="utf-8",
    )

    print("V4_A4A_2A_PRIMARY_TRAINING_PASS")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("source-audit", "train"),
        required=True,
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--slot-module", type=Path, required=True)
    parser.add_argument("--smoke-script", type=Path, required=True)
    parser.add_argument(
        "--contract-freeze-script",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--resource-probe-script",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--contract-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--source-audit-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "source-audit":
        if args.resume:
            raise RuntimeError(
                "--resume is invalid for source-audit mode"
            )
        return source_audit(args)
    return training(args)


if __name__ == "__main__":
    raise SystemExit(main())
