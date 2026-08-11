#!/usr/bin/env python3
"""
V4-A4a-1B
Two-Epoch Slot-Decoder Smoke Training

This script reuses the exact original A3 trainer's data pipeline:
  * load_metadata
  * load_split_indices
  * V4A3MemmapDataset
  * build_loader
  * build_normalized_adjacency
  * build_physical_valid_port_mask

It loads the frozen A3 checkpoint, constructs the audited 3,189-parameter A4a
slot model, and trains only the 1,076-parameter slot decoder for two bounded
epochs.

Smoke purpose only:
  * verify finite real-data losses and gradients
  * verify frozen encoder weights remain unchanged
  * verify exact unique decoding on real validation batches
  * monitor NULL use, duplicate proposals and set metrics
  * verify checkpoint save/reload
  * verify the vectorized matching loss against the audited exact loss

No development-test loader is constructed. No threshold/model selection occurs.
Smoke artifacts must not be used as the final A4a model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from v4_a4a_slot_model import (
    A4aFrozenSlotModel,
    NULL_CLASS,
    NUM_ROUTERS,
    NUM_SLOTS,
    SlotLossWeights,
    count_parameters,
    derived_cardinality_probabilities,
    derived_membership_probabilities,
    duplicate_slot_penalty,
    exact_unique_constrained_decode,
    greedy_unique_decode,
    permutation_invariant_slot_matching_loss,
)


EXPECTED_TRAINER_SHA = (
    "e5f9a48e9f7ca873c351743df92e3bb0"
    "0d00a893333434c7fd0e04734e010525"
)
EXPECTED_MODEL_SOURCE_SHA = (
    "5ebcf80385b95f329faf935cad8039b64"
    "afd79f464903308eb07e674d78a1704"
)
EXPECTED_CHECKPOINT_SHA = (
    "f61c1add6c057f7f53dd34fb1f9f4e95"
    "b01cefd5053e0e42bc100c76dac7f923"
)
EXPECTED_SLOT_MODULE_SHA = (
    "97b56aabb3239dfc7543c241665cacf99"
    "8eb6b1a1f5f67f931cdeb70e75c111c"
)
EXPECTED_A4A_1A_AUDIT_SHA = (
    "7075ae14697c24bf359c2aa214b1af894"
    "8c3c6a27ed9e500c51623b4cadb28b5"
)
EXPECTED_TOTAL_PARAMETERS = 3189
EXPECTED_TRAINABLE_PARAMETERS = 1076


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
    raise RuntimeError("checkpoint state dict not found")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def permutations_for_count(
    count: int,
    device: torch.device,
) -> torch.Tensor:
    if count == 0:
        return torch.empty(
            1,
            0,
            dtype=torch.long,
            device=device,
        )
    return torch.tensor(
        list(itertools.permutations(range(NUM_SLOTS), count)),
        dtype=torch.long,
        device=device,
    )


def vectorized_matching_loss(
    slot_logits: torch.Tensor,
    y_node: torch.Tensor,
    *,
    null_weight: float,
) -> torch.Tensor:
    """Vectorized equivalent of the audited exact matching loss."""
    if slot_logits.ndim != 3 or tuple(slot_logits.shape[1:]) != (4, 17):
        raise ValueError("slot_logits must be [B,4,17]")
    if y_node.ndim != 2 or y_node.shape[1] != NUM_ROUTERS:
        raise ValueError("y_node must be [B,16]")
    if null_weight <= 0:
        raise ValueError("null_weight must be positive")

    truth = y_node > 0.5
    counts = truth.sum(dim=1)
    if torch.any(counts > NUM_SLOTS):
        raise ValueError("found more than four true attackers")

    log_probabilities = F.log_softmax(slot_logits, dim=-1)
    per_sample = torch.empty(
        slot_logits.shape[0],
        dtype=slot_logits.dtype,
        device=slot_logits.device,
    )

    for count in range(NUM_SLOTS + 1):
        group_mask = counts == count
        if not bool(group_mask.any()):
            continue

        group_log_probabilities = log_probabilities[group_mask]
        group_truth = truth[group_mask]
        group_size = int(group_log_probabilities.shape[0])
        patterns = permutations_for_count(count, slot_logits.device)
        assignment_count = int(patterns.shape[0])

        targets = torch.full(
            (group_size, assignment_count, NUM_SLOTS),
            NULL_CLASS,
            dtype=torch.long,
            device=slot_logits.device,
        )

        if count > 0:
            attacker_ids = torch.nonzero(
                group_truth,
                as_tuple=False,
            )[:, 1].reshape(group_size, count)

            for pattern_index in range(assignment_count):
                for attacker_position in range(count):
                    target_slot = int(
                        patterns[
                            pattern_index,
                            attacker_position,
                        ].item()
                    )
                    targets[
                        :,
                        pattern_index,
                        target_slot,
                    ] = attacker_ids[:, attacker_position]

        expanded = group_log_probabilities.unsqueeze(1).expand(
            group_size,
            assignment_count,
            NUM_SLOTS,
            17,
        )
        selected = torch.gather(
            expanded,
            dim=3,
            index=targets.unsqueeze(-1),
        ).squeeze(-1)

        target_weights = torch.where(
            targets == NULL_CLASS,
            torch.full_like(selected, float(null_weight)),
            torch.ones_like(selected),
        )
        assignment_nll = -(
            selected * target_weights
        ).sum(dim=2) / target_weights.sum(dim=2)

        per_sample[group_mask] = assignment_nll.min(dim=1).values

    return per_sample.mean()


def compute_smoke_losses(
    slot_logits: torch.Tensor,
    y_node: torch.Tensor,
    *,
    null_weight: float,
    weights: SlotLossWeights,
) -> dict[str, torch.Tensor]:
    truth = (y_node > 0.5).to(slot_logits.dtype)
    slot_probabilities = torch.softmax(slot_logits, dim=-1)
    membership_probabilities = derived_membership_probabilities(
        slot_probabilities
    )
    cardinality_probabilities = derived_cardinality_probabilities(
        slot_probabilities
    )
    true_count = truth.sum(dim=1).long()

    matching = vectorized_matching_loss(
        slot_logits,
        truth,
        null_weight=null_weight,
    )
    membership = F.binary_cross_entropy(
        membership_probabilities,
        truth,
    )
    cardinality = F.nll_loss(
        torch.log(cardinality_probabilities.clamp_min(1e-8)),
        true_count,
    )
    duplicate = duplicate_slot_penalty(slot_probabilities)

    total = (
        weights.matching * matching
        + weights.membership * membership
        + weights.cardinality * cardinality
        + weights.duplicate * duplicate
    )

    return {
        "total": total,
        "matching": matching,
        "membership": membership,
        "cardinality": cardinality,
        "duplicate": duplicate,
    }


def compute_null_weight(
    data_dir: Path,
    train_indices: np.ndarray,
) -> dict[str, Any]:
    attacker_count = np.load(
        data_dir / "attacker_count.npy",
        mmap_mode="r",
    )
    train_counts = np.asarray(
        attacker_count[np.asarray(train_indices, dtype=np.int64)],
        dtype=np.int64,
    )

    non_null_targets = int(train_counts.sum())
    total_slots = int(train_counts.size * NUM_SLOTS)
    null_targets = int(total_slots - non_null_targets)

    if non_null_targets <= 0 or null_targets <= 0:
        raise RuntimeError(
            "invalid training prevalence for NULL-weight computation"
        )

    raw_ratio = non_null_targets / null_targets
    applied = min(1.0, max(0.05, raw_ratio))

    return {
        "definition": (
            "non_null_slot_targets / null_slot_targets, "
            "clamped to [0.05,1.0]"
        ),
        "training_sample_count": int(train_counts.size),
        "non_null_slot_targets": non_null_targets,
        "null_slot_targets": null_targets,
        "raw_ratio": raw_ratio,
        "applied_null_weight": applied,
        "development_test_used": False,
    }


def true_sets_from_y_node(y_node: torch.Tensor) -> list[list[int]]:
    truth = y_node.detach().cpu().numpy() > 0.5
    return [
        np.flatnonzero(row).astype(int).tolist()
        for row in truth
    ]


class SetMetricAccumulator:
    def __init__(self) -> None:
        self.samples = 0
        self.attack_samples = 0
        self.normal_samples = 0
        self.attack_exact = 0
        self.all_exact = 0
        self.attack_nonempty = 0
        self.normal_false_set = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.count_correct = 0
        self.raw_duplicate_samples = 0
        self.raw_all_null_samples = 0
        self.greedy_exact_disagreement = 0
        self.slot_count = 0
        self.slot_null_argmax = 0

    def update(
        self,
        slot_logits: torch.Tensor,
        graph_logits: torch.Tensor,
        y_node: torch.Tensor,
        attacker_count: torch.Tensor,
        *,
        graph_threshold: float,
        compare_greedy: bool,
    ) -> None:
        exact = exact_unique_constrained_decode(slot_logits)
        greedy = (
            greedy_unique_decode(slot_logits)
            if compare_greedy
            else None
        )

        graph_probability = torch.sigmoid(
            graph_logits
        ).detach().cpu().numpy()
        true_sets = true_sets_from_y_node(y_node)
        true_count = attacker_count.detach().cpu().numpy().astype(int)

        cardinality = derived_cardinality_probabilities(
            torch.softmax(slot_logits, dim=-1)
        ).argmax(dim=1).detach().cpu().numpy()

        raw_argmax = slot_logits.argmax(dim=-1).detach().cpu().numpy()
        self.slot_count += int(raw_argmax.size)
        self.slot_null_argmax += int(
            np.sum(raw_argmax == NULL_CLASS)
        )

        for sample, (true_set_list, predicted_set_list) in enumerate(
            zip(true_sets, exact["predicted_sets"])
        ):
            predicted_set = (
                set(predicted_set_list)
                if graph_probability[sample] >= graph_threshold
                else set()
            )
            true_set = set(true_set_list)

            raw_real = [
                int(value)
                for value in raw_argmax[sample]
                if int(value) != NULL_CLASS
            ]
            if len(raw_real) != len(set(raw_real)):
                self.raw_duplicate_samples += 1
            if not raw_real:
                self.raw_all_null_samples += 1

            self.samples += 1
            self.all_exact += int(predicted_set == true_set)
            self.count_correct += int(
                int(cardinality[sample]) == int(true_count[sample])
            )

            self.tp += len(predicted_set & true_set)
            self.fp += len(predicted_set - true_set)
            self.fn += len(true_set - predicted_set)

            if true_set:
                self.attack_samples += 1
                self.attack_exact += int(predicted_set == true_set)
                self.attack_nonempty += int(bool(predicted_set))
            else:
                self.normal_samples += 1
                self.normal_false_set += int(bool(predicted_set))

            if greedy is not None:
                greedy_set = (
                    set(greedy["predicted_sets"][sample])
                    if graph_probability[sample] >= graph_threshold
                    else set()
                )
                self.greedy_exact_disagreement += int(
                    greedy_set != predicted_set
                )

    def finalize(self) -> dict[str, Any]:
        precision = (
            self.tp / (self.tp + self.fp)
            if self.tp + self.fp > 0
            else 1.0
        )
        recall = (
            self.tp / (self.tp + self.fn)
            if self.tp + self.fn > 0
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return {
            "samples": self.samples,
            "attack_samples": self.attack_samples,
            "normal_samples": self.normal_samples,
            "attack_exact_localization": (
                self.attack_exact / self.attack_samples
                if self.attack_samples
                else 0.0
            ),
            "all_exact_set": (
                self.all_exact / self.samples
                if self.samples
                else 0.0
            ),
            "candidate_coverage": (
                self.attack_nonempty / self.attack_samples
                if self.attack_samples
                else 0.0
            ),
            "normal_false_set_fraction": (
                self.normal_false_set / self.normal_samples
                if self.normal_samples
                else 0.0
            ),
            "set_precision": precision,
            "set_recall": recall,
            "set_f1": f1,
            "cardinality_accuracy": (
                self.count_correct / self.samples
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
            "slot_null_argmax_fraction": (
                self.slot_null_argmax / self.slot_count
                if self.slot_count
                else 0.0
            ),
            "greedy_exact_set_disagreement_fraction": (
                self.greedy_exact_disagreement / self.samples
                if self.samples
                else 0.0
            ),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def frozen_state_snapshot(model: A4aFrozenSlotModel) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.encoder.state_dict().items()
    }


def compare_state(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, Any]:
    differences = {}
    maximum = 0.0

    for name in before:
        difference = float(
            torch.max(
                torch.abs(before[name] - after[name].detach().cpu())
            ).item()
        )
        if difference != 0.0:
            differences[name] = difference
        maximum = max(maximum, difference)

    return {
        "maximum_absolute_difference": maximum,
        "changed_tensors": differences,
        "unchanged": not differences,
    }


def matching_crosscheck(
    slot_logits: torch.Tensor,
    y_node: torch.Tensor,
    null_weight: float,
) -> dict[str, Any]:
    subset_size = min(12, int(slot_logits.shape[0]))
    logits_subset = slot_logits[:subset_size]
    truth_subset = y_node[:subset_size]

    audited = permutation_invariant_slot_matching_loss(
        logits_subset,
        truth_subset,
        null_weight=null_weight,
    )
    vectorized = vectorized_matching_loss(
        logits_subset,
        truth_subset,
        null_weight=null_weight,
    )
    difference = float(
        torch.abs(audited - vectorized).detach().cpu().item()
    )

    if difference > 1e-5:
        raise RuntimeError(
            "vectorized matching does not match audited implementation: "
            f"difference={difference}"
        )

    return {
        "sample_count": subset_size,
        "audited_loss": float(audited.detach().cpu()),
        "vectorized_loss": float(vectorized.detach().cpu()),
        "absolute_difference": difference,
        "pass": True,
    }


def run_limited_epoch(
    model: A4aFrozenSlotModel,
    loader,
    adjacency: torch.Tensor,
    physical_mask: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    max_batches: int,
    null_weight: float,
    loss_weights: SlotLossWeights,
    graph_threshold: float,
    grad_clip_norm: float,
    crosscheck_matching: bool,
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
    metric = SetMetricAccumulator()
    gradient_norms: list[float] = []
    logit_min = float("inf")
    logit_max = -float("inf")
    crosscheck_result = None
    first_probe = None

    for batch_number, batch in enumerate(loader, start=1):
        if batch_number > max_batches:
            break

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
        y_graph = batch["y_graph"].to(
            device,
            non_blocking=True,
        )
        attacker_count = batch["attacker_count"].to(
            device,
            non_blocking=True,
            dtype=torch.long,
        )

        if first_probe is None:
            first_probe = {
                "x": x[: min(8, x.shape[0])].detach().cpu(),
                "y_node": y_node[: min(8, y_node.shape[0])].detach().cpu(),
            }

        with torch.set_grad_enabled(training):
            output = model(x, adjacency, physical_mask)
            losses = compute_smoke_losses(
                output["slot_logits"],
                y_node,
                null_weight=null_weight,
                weights=loss_weights,
            )

            for name, value in losses.items():
                if not torch.isfinite(value):
                    raise FloatingPointError(
                        f"non-finite {name} loss at batch {batch_number}"
                    )

            if crosscheck_matching and crosscheck_result is None:
                crosscheck_result = matching_crosscheck(
                    output["slot_logits"],
                    y_node,
                    null_weight,
                )

            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()

                for name, parameter in model.named_parameters():
                    if (
                        parameter.grad is not None
                        and not torch.isfinite(parameter.grad).all()
                    ):
                        raise FloatingPointError(
                            f"non-finite gradient in {name}"
                        )

                norm = torch.nn.utils.clip_grad_norm_(
                    model.slot_decoder.parameters(),
                    max_norm=grad_clip_norm,
                )
                norm_value = float(norm.detach().cpu())
                if not math.isfinite(norm_value):
                    raise FloatingPointError(
                        f"non-finite gradient norm at batch {batch_number}"
                    )
                gradient_norms.append(norm_value)
                optimizer.step()

        batch_size = int(x.shape[0])
        samples += batch_size
        for name, value in losses.items():
            totals[name] += float(value.detach().cpu()) * batch_size

        current_logits = output["slot_logits"].detach()
        logit_min = min(logit_min, float(current_logits.min().cpu()))
        logit_max = max(logit_max, float(current_logits.max().cpu()))

        metric.update(
            output["slot_logits"],
            output["graph_logits"],
            y_node,
            attacker_count,
            graph_threshold=graph_threshold,
            compare_greedy=not training,
        )

    if samples == 0:
        raise RuntimeError("limited epoch processed zero samples")

    losses_average = {
        name: value / samples
        for name, value in totals.items()
    }
    extra = {
        "batches": min(max_batches, batch_number),
        "samples": samples,
        "gradient_norm_before_clip_max": (
            max(gradient_norms) if gradient_norms else None
        ),
        "gradient_norm_before_clip_mean": (
            sum(gradient_norms) / len(gradient_norms)
            if gradient_norms
            else None
        ),
        "gradient_clip_norm": grad_clip_norm,
        "slot_logit_min": logit_min,
        "slot_logit_max": logit_max,
        "matching_crosscheck": crosscheck_result,
        "probe": first_probe,
    }
    return losses_average, metric.finalize(), extra


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--slot-module", type=Path, required=True)
    parser.add_argument("--a4a-1a-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=12)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.epochs != 2:
        raise RuntimeError("A4a-1B smoke is frozen to exactly two epochs")
    if args.train_batches <= 0 or args.val_batches <= 0:
        raise ValueError("batch limits must be positive")
    if args.grad_clip_norm <= 0:
        raise ValueError("gradient clipping norm must be positive")
    if args.num_workers == 0 and args.persistent_workers:
        raise ValueError(
            "persistent workers require num_workers > 0"
        )

    required = [
        args.repo,
        args.data_dir,
        args.trainer,
        args.model_source,
        args.base_checkpoint,
        args.slot_module,
        args.a4a_1a_dir / "V4_A4A_1A_SLOT_IMPLEMENTATION_PASS",
        args.a4a_1a_dir / "A4A_1A_LOCK.json",
        args.a4a_1a_dir / "implementation_audit.json",
        args.contract,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("A4a-1B FAIL: missing paths", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(
            f"A4a-1B FAIL: output exists: {args.output_dir}",
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True)

    provenance = {
        "trainer_sha256": sha256_file(args.trainer),
        "model_source_sha256": sha256_file(args.model_source),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "slot_module_sha256": sha256_file(args.slot_module),
        "a4a_1a_audit_sha256": sha256_file(
            args.a4a_1a_dir / "implementation_audit.json"
        ),
        "contract_sha256": sha256_file(args.contract),
        "smoke_script_sha256": sha256_file(Path(__file__)),
    }
    expected = {
        "trainer_sha256": EXPECTED_TRAINER_SHA,
        "model_source_sha256": EXPECTED_MODEL_SOURCE_SHA,
        "base_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "slot_module_sha256": EXPECTED_SLOT_MODULE_SHA,
        "a4a_1a_audit_sha256": EXPECTED_A4A_1A_AUDIT_SHA,
    }
    failures = [
        f"{key} mismatch"
        for key, value in expected.items()
        if provenance[key] != value
    ]

    contract = load_json(args.contract)
    if contract.get("status") != "PASS":
        failures.append("A4a slot contract is not PASS")

    a4a_1a = load_json(
        args.a4a_1a_dir / "implementation_audit.json"
    )
    if a4a_1a.get("ready_for_a4a_1b_smoke_training") is not True:
        failures.append("A4a-1A did not authorize smoke training")
    if a4a_1a.get("development_test_accessed") is not False:
        failures.append("A4a-1A reports development-test access")

    if failures:
        payload = {"status": "FAIL", "failures": failures}
        (args.output_dir / "smoke_failure.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1

    set_seed(args.seed)

    for import_root in (
        args.repo,
        args.repo / "scripts",
        args.trainer.parent,
        args.model_source.parent,
        args.slot_module.parent,
    ):
        text = str(import_root.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)

    trainer_module = load_module(
        args.trainer,
        "resolved_original_v4_a3_trainer_for_a4a_1b_smoke",
    )
    a3_module = load_module(
        args.model_source,
        "resolved_v4_a3_model_for_a4a_1b_smoke",
    )

    original_args = argparse.Namespace(
        data_dir=str(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
    )

    objects = trainer_module.prepare_objects(
        original_args,
        include_val=True,
    )
    if objects["val_loader"] is None:
        raise RuntimeError("original pipeline did not create validation loader")

    device = objects["device"]
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")

    # The original prepare_objects constructs an unused A3 model. Release it
    # before constructing the frozen-checkpoint slot model.
    original_prepared_model = objects.pop("model", None)
    del original_prepared_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checkpoint = torch.load(
        args.base_checkpoint,
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
    model.train(True)
    counts = count_parameters(model)

    if counts["total"] != EXPECTED_TOTAL_PARAMETERS:
        raise RuntimeError(f"unexpected total parameters: {counts}")
    if counts["trainable"] != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(f"unexpected trainable parameters: {counts}")

    adjacency = objects["adjacency"].to(device)
    physical_mask = objects["mask"].to(device)

    if not torch.equal(
        adjacency.detach().cpu(),
        checkpoint["A_hat"].detach().cpu(),
    ):
        raise RuntimeError(
            "original pipeline adjacency differs from frozen checkpoint"
        )
    if not torch.equal(
        physical_mask.detach().cpu(),
        checkpoint["physical_valid_port_mask"].detach().cpu(),
    ):
        raise RuntimeError(
            "original pipeline physical mask differs from checkpoint"
        )

    null_weight_report = compute_null_weight(
        args.data_dir,
        objects["splits"]["train"],
    )
    null_weight = float(
        null_weight_report["applied_null_weight"]
    )

    loss_weights = SlotLossWeights(
        matching=1.0,
        membership=0.5,
        cardinality=0.5,
        duplicate=0.1,
    )
    graph_threshold = float(
        contract["evaluation_contract"]["graph_threshold"]
    )

    optimizer = torch.optim.AdamW(
        model.slot_decoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    frozen_before = frozen_state_snapshot(model)

    history: list[dict[str, Any]] = []
    final_probe = None

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_losses, train_metrics, train_extra = run_limited_epoch(
            model,
            objects["train_loader"],
            adjacency,
            physical_mask,
            device,
            optimizer,
            max_batches=args.train_batches,
            null_weight=null_weight,
            loss_weights=loss_weights,
            graph_threshold=graph_threshold,
            grad_clip_norm=args.grad_clip_norm,
            crosscheck_matching=(epoch == 1),
        )
        val_losses, val_metrics, val_extra = run_limited_epoch(
            model,
            objects["val_loader"],
            adjacency,
            physical_mask,
            device,
            None,
            max_batches=args.val_batches,
            null_weight=null_weight,
            loss_weights=loss_weights,
            graph_threshold=graph_threshold,
            grad_clip_norm=args.grad_clip_norm,
            crosscheck_matching=False,
        )

        if val_extra["probe"] is not None:
            final_probe = val_extra["probe"]

        duration = time.perf_counter() - started
        gpu_peak = (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else 0.0
        )

        row = {
            "epoch": epoch,
            "duration_seconds": duration,
            "train_losses": train_losses,
            "train_metrics": train_metrics,
            "train_extra": {
                key: value
                for key, value in train_extra.items()
                if key != "probe"
            },
            "validation_losses": val_losses,
            "validation_metrics": val_metrics,
            "validation_extra": {
                key: value
                for key, value in val_extra.items()
                if key != "probe"
            },
            "gpu_peak_allocated_mib": gpu_peak,
        }
        history.append(row)

        print(
            f"epoch={epoch} "
            f"train_loss={train_losses['total']:.6f} "
            f"val_loss={val_losses['total']:.6f} "
            f"val_exact={val_metrics['attack_exact_localization']:.4f} "
            f"val_precision={val_metrics['set_precision']:.4f} "
            f"val_coverage={val_metrics['candidate_coverage']:.4f} "
            f"val_normal_false={val_metrics['normal_false_set_fraction']:.6f} "
            f"raw_dup={val_metrics['raw_duplicate_proposal_fraction']:.4f} "
            f"duration={duration:.1f}s",
            flush=True,
        )

    frozen_after = {
        name: tensor.detach().cpu()
        for name, tensor in model.encoder.state_dict().items()
    }
    frozen_comparison = compare_state(
        frozen_before,
        frozen_after,
    )
    if not frozen_comparison["unchanged"]:
        raise RuntimeError(
            f"frozen encoder changed: {frozen_comparison}"
        )

    if final_probe is None:
        raise RuntimeError("validation probe was not captured")

    probe_x = final_probe["x"].to(device)
    with torch.no_grad():
        model.eval()
        pre_save_output = model(
            probe_x,
            adjacency,
            physical_mask,
        )
        pre_save_slot_logits = (
            pre_save_output["slot_logits"].detach().cpu()
        )
        pre_save_graph_logits = (
            pre_save_output["graph_logits"].detach().cpu()
        )

    smoke_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "base_checkpoint_sha256": provenance[
            "base_checkpoint_sha256"
        ],
        "slot_module_sha256": provenance["slot_module_sha256"],
        "contract_sha256": provenance["contract_sha256"],
        "smoke_script_sha256": provenance["smoke_script_sha256"],
        "args": vars(args),
        "parameter_count": counts,
        "null_weight": null_weight_report,
        "loss_weights": {
            "matching": loss_weights.matching,
            "membership": loss_weights.membership,
            "cardinality": loss_weights.cardinality,
            "duplicate": loss_weights.duplicate,
        },
        "A_hat": adjacency.detach().cpu(),
        "physical_valid_port_mask": physical_mask.detach().cpu(),
        "history": history,
        "smoke_only": True,
        "test_loader_constructed": False,
        "test_evaluated": False,
    }
    checkpoint_path = args.output_dir / "smoke_last_model.pt"
    trainer_module.atomic_torch_save(
        checkpoint_path,
        smoke_checkpoint,
    )

    reloaded_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    reloaded_base = a3_module.A3SourcePreserveModel()
    reloaded_base.load_state_dict(
        extract_state_dict(checkpoint),
        strict=True,
    )
    reloaded_model = A4aFrozenSlotModel(
        reloaded_base
    ).to(device)
    reloaded_model.load_state_dict(
        reloaded_checkpoint["model_state_dict"],
        strict=True,
    )
    reloaded_model.eval()

    with torch.no_grad():
        reloaded_output = reloaded_model(
            probe_x,
            adjacency,
            physical_mask,
        )

    slot_reload_diff = float(
        torch.max(
            torch.abs(
                pre_save_slot_logits
                - reloaded_output["slot_logits"].detach().cpu()
            )
        ).item()
    )
    graph_reload_diff = float(
        torch.max(
            torch.abs(
                pre_save_graph_logits
                - reloaded_output["graph_logits"].detach().cpu()
            )
        ).item()
    )
    if slot_reload_diff != 0.0 or graph_reload_diff != 0.0:
        raise RuntimeError(
            "checkpoint reload changed outputs: "
            f"slot={slot_reload_diff}, graph={graph_reload_diff}"
        )

    first_train_loss = history[0]["train_losses"]["total"]
    last_train_loss = history[-1]["train_losses"]["total"]

    report = {
        "status": "PASS",
        "designation": "V4-A4a-1B Two-Epoch Slot Smoke Training",
        "device": str(device),
        "parameter_count": counts,
        "configuration": {
            "epochs": args.epochs,
            "train_batches_per_epoch": args.train_batches,
            "validation_batches_per_epoch": args.val_batches,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.grad_clip_norm,
            "graph_threshold": graph_threshold,
            "loss_weights": {
                "matching": loss_weights.matching,
                "membership": loss_weights.membership,
                "cardinality": loss_weights.cardinality,
                "duplicate": loss_weights.duplicate,
            },
        },
        "null_weight": null_weight_report,
        "history": history,
        "integrity_checks": {
            "vectorized_matching_crosscheck_pass": bool(
                history[0]["train_extra"][
                    "matching_crosscheck"
                ]["pass"]
            ),
            "frozen_encoder_unchanged": frozen_comparison,
            "checkpoint_reload_pass": True,
            "checkpoint_reload_slot_logit_max_abs_diff": slot_reload_diff,
            "checkpoint_reload_graph_logit_max_abs_diff": graph_reload_diff,
            "all_losses_finite": True,
            "all_gradients_finite": True,
            "gradient_clipping_exercised": True,
            "exact_unique_decode_exercised": True,
            "greedy_decode_compared_on_validation": True,
            "parameter_count_pass": (
                counts["total"] == EXPECTED_TOTAL_PARAMETERS
                and counts["trainable"]
                == EXPECTED_TRAINABLE_PARAMETERS
            ),
        },
        "smoke_observations": {
            "train_loss_epoch1": first_train_loss,
            "train_loss_epoch2": last_train_loss,
            "train_loss_decreased": (
                last_train_loss < first_train_loss
            ),
            "final_validation_metrics": history[-1][
                "validation_metrics"
            ],
            "final_validation_losses": history[-1][
                "validation_losses"
            ],
        },
        "split_usage": {
            "train_samples_available": int(
                len(objects["splits"]["train"])
            ),
            "validation_samples_available": int(
                len(objects["splits"]["val"])
            ),
            "train_loader_constructed": True,
            "validation_loader_constructed": True,
            "test_loader_constructed": False,
            "test_samples_loaded": False,
            "test_evaluated": False,
        },
        "provenance": provenance,
        "smoke_only": True,
        "must_not_be_used_as_final_model": True,
        "validation_selection_performed": False,
        "threshold_search_performed": False,
        "development_test_accessed": False,
        "ready_for_a4a_1c_training_contract_freeze": True,
    }

    report_path = args.output_dir / "smoke_report.json"
    trainer_module.atomic_json_dump(report_path, report)
    trainer_module.atomic_json_dump(
        args.output_dir / "smoke_history.json",
        history,
    )
    trainer_module.atomic_json_dump(
        args.output_dir / "null_weight.json",
        null_weight_report,
    )

    lock = {
        "status": "A4A_1B_TWO_EPOCH_SMOKE_COMPLETE",
        "base_checkpoint_sha256": provenance[
            "base_checkpoint_sha256"
        ],
        "trainer_sha256": provenance["trainer_sha256"],
        "model_source_sha256": provenance["model_source_sha256"],
        "slot_module_sha256": provenance["slot_module_sha256"],
        "contract_sha256": provenance["contract_sha256"],
        "smoke_script_sha256": provenance["smoke_script_sha256"],
        "smoke_checkpoint_sha256": sha256_file(checkpoint_path),
        "smoke_report_sha256": sha256_file(report_path),
        "test_loader_constructed": False,
        "test_evaluated": False,
        "development_test_accessed": False,
        "validation_selection_performed": False,
    }
    trainer_module.atomic_json_dump(
        args.output_dir / "A4A_1B_LOCK.json",
        lock,
    )
    (
        args.output_dir / "V4_A4A_1B_TWO_EPOCH_SMOKE_PASS"
    ).write_text(
        "V4_A4A_1B_TWO_EPOCH_SMOKE_PASS\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    print("V4_A4A_1B_TWO_EPOCH_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
