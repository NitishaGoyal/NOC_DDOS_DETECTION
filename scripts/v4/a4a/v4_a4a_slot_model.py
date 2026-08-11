#!/usr/bin/env python3
"""
V4-A4a slot-set decoder primitives.

This module implements:
- the trimmed frozen A3 temporal/local/graph encoder and graph head
- a 1,076-parameter low-rank four-slot router-or-NULL decoder
- exact permutation-invariant slot matching
- derived router-membership probabilities
- derived cardinality probabilities
- duplicate-slot regularization
- exact unique constrained decoding
- a hardware-oriented greedy unique decoder

No dataset split, threshold selection, or training loop is contained here.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_ROUTERS = 16
NUM_SLOTS = 4
NULL_CLASS = 16
NUM_SLOT_CLASSES = 17
SLOT_DIM = 8


@dataclass(frozen=True)
class SlotLossWeights:
    matching: float = 1.0
    membership: float = 0.5
    cardinality: float = 0.5
    duplicate: float = 0.1


def _validate_y_node(y_node: torch.Tensor) -> torch.Tensor:
    if y_node.ndim != 2 or y_node.shape[1] != NUM_ROUTERS:
        raise ValueError(
            f"y_node must have shape [B,{NUM_ROUTERS}], "
            f"got {tuple(y_node.shape)}"
        )
    truth = y_node > 0.5
    count = truth.sum(dim=1)
    if torch.any(count > NUM_SLOTS):
        raise ValueError("A4a supports at most four attacker routers")
    return truth


class FrozenA3EncoderGraphHead(nn.Module):
    """Trimmed deployment path from the frozen A3 model.

    Only the temporal encoder, local projection, GCN and graph head are copied.
    The old attacker and count heads are deliberately excluded.
    """

    def __init__(self, frozen_a3_model: nn.Module):
        super().__init__()
        self.temporal_conv = copy.deepcopy(frozen_a3_model.temporal_conv)
        self.local_projection = copy.deepcopy(
            frozen_a3_model.local_projection
        )
        self.gcn = copy.deepcopy(frozen_a3_model.gcn)
        self.graph_head = copy.deepcopy(frozen_a3_model.graph_head)

        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        # Frozen deterministic path: keep it in eval mode even while the
        # decoder is trained.
        super().train(False)
        return self

    def forward(
        self,
        x: torch.Tensor,
        a_hat: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (16, 8, 24):
            raise ValueError(
                f"expected x [B,16,8,24], got {tuple(x.shape)}"
            )

        batch_size = int(x.shape[0])
        temporal_input = x.reshape(
            batch_size * NUM_ROUTERS,
            8,
            24,
        ).permute(0, 2, 1)

        temporal_sequence = F.relu(
            self.temporal_conv(temporal_input)
        )
        temporal_mean = temporal_sequence.mean(dim=2)
        temporal_max = temporal_sequence.max(dim=2).values
        temporal_pooled = torch.cat(
            [temporal_mean, temporal_max],
            dim=1,
        ).reshape(batch_size, NUM_ROUTERS, 32)

        if tuple(physical_port_mask.shape) == (NUM_ROUTERS, 5):
            mask = physical_port_mask.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
        elif tuple(physical_port_mask.shape) == (
            batch_size,
            NUM_ROUTERS,
            5,
        ):
            mask = physical_port_mask
        else:
            raise ValueError(
                "physical_port_mask must be [16,5] or [B,16,5], "
                f"got {tuple(physical_port_mask.shape)}"
            )

        local_input = torch.cat(
            [
                temporal_pooled,
                mask.to(temporal_pooled.dtype),
            ],
            dim=2,
        )
        h_local = F.relu(self.local_projection(local_input))
        h_graph = F.relu(self.gcn(h_local, a_hat))
        h_node = torch.cat([h_local, h_graph], dim=2)

        regional_mean = h_node.mean(dim=1)
        regional_max = h_node.max(dim=1).values
        regional_embedding = torch.cat(
            [regional_mean, regional_max],
            dim=1,
        )
        graph_logits = self.graph_head(
            regional_embedding
        ).squeeze(-1)

        return {
            "graph_logits": graph_logits,
            "h_local": h_local,
            "h_graph": h_graph,
            "h_node": h_node,
            "regional_embedding": regional_embedding,
        }


class A4aLowRankSlotDecoder(nn.Module):
    """Four unordered router-or-NULL slots, 1,076 parameters."""

    def __init__(self, slot_dim: int = SLOT_DIM):
        super().__init__()
        if slot_dim != SLOT_DIM:
            raise ValueError(
                f"frozen A4a-0S contract requires slot_dim={SLOT_DIM}"
            )

        self.slot_dim = slot_dim
        self.router_key_projection = nn.Linear(32, slot_dim)
        self.global_context_projection = nn.Linear(64, slot_dim)
        self.slot_identity_embeddings = nn.Parameter(
            torch.empty(NUM_SLOTS, slot_dim)
        )
        self.slot_null_head = nn.Linear(64, NUM_SLOTS)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.router_key_projection.weight)
        nn.init.zeros_(self.router_key_projection.bias)
        nn.init.xavier_uniform_(self.global_context_projection.weight)
        nn.init.zeros_(self.global_context_projection.bias)
        nn.init.normal_(
            self.slot_identity_embeddings,
            mean=0.0,
            std=0.02,
        )
        nn.init.xavier_uniform_(self.slot_null_head.weight)
        nn.init.zeros_(self.slot_null_head.bias)

    def forward(
        self,
        h_node: torch.Tensor,
        regional_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if h_node.ndim != 3 or tuple(h_node.shape[1:]) != (16, 32):
            raise ValueError(
                f"h_node must be [B,16,32], got {tuple(h_node.shape)}"
            )
        if (
            regional_embedding.ndim != 2
            or regional_embedding.shape[1] != 64
            or regional_embedding.shape[0] != h_node.shape[0]
        ):
            raise ValueError(
                "regional_embedding must be [B,64] with the same B"
            )

        router_keys = self.router_key_projection(h_node)
        global_context = self.global_context_projection(
            regional_embedding
        )
        slot_queries = (
            global_context.unsqueeze(1)
            + self.slot_identity_embeddings.unsqueeze(0)
        )

        router_logits = torch.einsum(
            "bsd,brd->bsr",
            slot_queries,
            router_keys,
        ) / math.sqrt(float(self.slot_dim))

        null_logits = self.slot_null_head(
            regional_embedding
        ).unsqueeze(-1)

        slot_logits = torch.cat(
            [router_logits, null_logits],
            dim=-1,
        )
        slot_probabilities = torch.softmax(slot_logits, dim=-1)

        membership_probabilities = derived_membership_probabilities(
            slot_probabilities
        )
        cardinality_probabilities = derived_cardinality_probabilities(
            slot_probabilities
        )

        return {
            "slot_logits": slot_logits,
            "slot_probabilities": slot_probabilities,
            "membership_probabilities": membership_probabilities,
            "cardinality_probabilities": cardinality_probabilities,
        }


class A4aFrozenSlotModel(nn.Module):
    """Frozen A3 representation plus the trainable A4a slot decoder."""

    def __init__(self, frozen_a3_model: nn.Module):
        super().__init__()
        self.encoder = FrozenA3EncoderGraphHead(frozen_a3_model)
        self.slot_decoder = A4aLowRankSlotDecoder()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.train(False)
        self.slot_decoder.train(mode)
        return self

    def forward(
        self,
        x: torch.Tensor,
        a_hat: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            representation = self.encoder(
                x,
                a_hat,
                physical_port_mask,
            )

        decoded = self.slot_decoder(
            representation["h_node"],
            representation["regional_embedding"],
        )
        return {**representation, **decoded}


def derived_membership_probabilities(
    slot_probabilities: torch.Tensor,
) -> torch.Tensor:
    """q_r = 1 - product_s (1 - p(slot_s = router_r))."""
    if (
        slot_probabilities.ndim != 3
        or tuple(slot_probabilities.shape[1:]) != (
            NUM_SLOTS,
            NUM_SLOT_CLASSES,
        )
    ):
        raise ValueError(
            "slot_probabilities must have shape [B,4,17]"
        )

    router_probabilities = slot_probabilities[..., :NUM_ROUTERS]
    return 1.0 - torch.prod(
        1.0 - router_probabilities,
        dim=1,
    )


def derived_cardinality_probabilities(
    slot_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Poisson-binomial P(K=0..4) from slot non-NULL probabilities."""
    if (
        slot_probabilities.ndim != 3
        or tuple(slot_probabilities.shape[1:]) != (
            NUM_SLOTS,
            NUM_SLOT_CLASSES,
        )
    ):
        raise ValueError(
            "slot_probabilities must have shape [B,4,17]"
        )

    non_null = 1.0 - slot_probabilities[..., NULL_CLASS]
    batch_size = int(non_null.shape[0])

    distribution = torch.zeros(
        batch_size,
        NUM_SLOTS + 1,
        dtype=slot_probabilities.dtype,
        device=slot_probabilities.device,
    )
    distribution[:, 0] = 1.0

    for slot in range(NUM_SLOTS):
        probability = non_null[:, slot]
        updated = distribution * (1.0 - probability).unsqueeze(1)
        updated[:, 1:] = (
            updated[:, 1:]
            + distribution[:, :-1] * probability.unsqueeze(1)
        )
        distribution = updated

    return distribution


def duplicate_slot_penalty(
    slot_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Mean pairwise overlap over real-router probabilities."""
    router_probabilities = slot_probabilities[..., :NUM_ROUTERS]
    penalties = []

    for left in range(NUM_SLOTS):
        for right in range(left + 1, NUM_SLOTS):
            penalties.append(
                (
                    router_probabilities[:, left]
                    * router_probabilities[:, right]
                ).sum(dim=1)
            )

    return torch.stack(penalties, dim=1).mean()


def _slot_patterns_for_count(
    count: int,
    device: torch.device,
) -> torch.Tensor:
    if count < 0 or count > NUM_SLOTS:
        raise ValueError("count must be between zero and four")

    if count == 0:
        return torch.empty(
            1,
            0,
            dtype=torch.long,
            device=device,
        )

    patterns = list(
        itertools.permutations(range(NUM_SLOTS), count)
    )
    return torch.tensor(
        patterns,
        dtype=torch.long,
        device=device,
    )


def permutation_invariant_slot_matching_loss(
    slot_logits: torch.Tensor,
    y_node: torch.Tensor,
    *,
    null_weight: float = 1.0,
) -> torch.Tensor:
    """Exact unordered-set NLL by enumerating at most 4! assignments.

    For k true attackers, all injective mappings from the sorted k router IDs
    to the four slots are evaluated. Remaining slots target NULL.
    """
    truth = _validate_y_node(y_node)

    if (
        slot_logits.ndim != 3
        or tuple(slot_logits.shape[1:]) != (
            NUM_SLOTS,
            NUM_SLOT_CLASSES,
        )
        or slot_logits.shape[0] != truth.shape[0]
    ):
        raise ValueError("slot_logits must have shape [B,4,17]")

    if null_weight <= 0:
        raise ValueError("null_weight must be positive")

    log_probabilities = F.log_softmax(slot_logits, dim=-1)
    per_sample_losses = []

    for sample in range(int(slot_logits.shape[0])):
        attacker_ids = torch.nonzero(
            truth[sample],
            as_tuple=False,
        ).flatten()
        count = int(attacker_ids.numel())
        patterns = _slot_patterns_for_count(
            count,
            slot_logits.device,
        )

        assignment_count = int(patterns.shape[0])
        targets = torch.full(
            (assignment_count, NUM_SLOTS),
            NULL_CLASS,
            dtype=torch.long,
            device=slot_logits.device,
        )

        for attacker_position in range(count):
            targets[
                torch.arange(
                    assignment_count,
                    device=slot_logits.device,
                ),
                patterns[:, attacker_position],
            ] = attacker_ids[attacker_position]

        expanded_log_probabilities = log_probabilities[
            sample
        ].unsqueeze(0).expand(assignment_count, -1, -1)

        selected = torch.gather(
            expanded_log_probabilities,
            dim=2,
            index=targets.unsqueeze(-1),
        ).squeeze(-1)

        weights = torch.ones_like(selected)
        weights = torch.where(
            targets == NULL_CLASS,
            torch.full_like(weights, float(null_weight)),
            weights,
        )
        assignment_nll = -(
            selected * weights
        ).sum(dim=1) / weights.sum(dim=1)

        per_sample_losses.append(assignment_nll.min())

    return torch.stack(per_sample_losses).mean()


def slot_set_losses(
    slot_logits: torch.Tensor,
    y_node: torch.Tensor,
    *,
    weights: SlotLossWeights = SlotLossWeights(),
    null_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    truth = _validate_y_node(y_node).to(slot_logits.dtype)
    slot_probabilities = torch.softmax(slot_logits, dim=-1)
    membership = derived_membership_probabilities(
        slot_probabilities
    )
    cardinality = derived_cardinality_probabilities(
        slot_probabilities
    )
    true_count = truth.sum(dim=1).long()

    matching = permutation_invariant_slot_matching_loss(
        slot_logits,
        truth,
        null_weight=null_weight,
    )
    membership_loss = F.binary_cross_entropy(
        membership,
        truth,
    )
    cardinality_loss = F.nll_loss(
        torch.log(cardinality.clamp_min(1e-8)),
        true_count,
    )
    duplicate = duplicate_slot_penalty(slot_probabilities)

    total = (
        weights.matching * matching
        + weights.membership * membership_loss
        + weights.cardinality * cardinality_loss
        + weights.duplicate * duplicate
    )

    return {
        "total": total,
        "matching": matching,
        "membership": membership_loss,
        "cardinality": cardinality_loss,
        "duplicate": duplicate,
    }


def exact_unique_constrained_decode(
    slot_logits: torch.Tensor,
) -> dict[str, Any]:
    """Exact maximum-score slot assignment using a linear assignment solver.

    Four distinct NULL dummy columns are added. All have the slot-specific
    NULL logit, so NULL can be selected by any number of slots while each real
    router remains unique.
    """
    if (
        slot_logits.ndim != 3
        or tuple(slot_logits.shape[1:]) != (
            NUM_SLOTS,
            NUM_SLOT_CLASSES,
        )
    ):
        raise ValueError("slot_logits must have shape [B,4,17]")

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for exact constrained reference decoding"
        ) from exc

    logits = slot_logits.detach().cpu().numpy()
    assignments: list[list[int]] = []
    sets: list[list[int]] = []
    scores: list[float] = []

    for sample_logits in logits:
        real_router_scores = sample_logits[:, :NUM_ROUTERS]
        null_scores = np.repeat(
            sample_logits[:, NULL_CLASS : NULL_CLASS + 1],
            NUM_SLOTS,
            axis=1,
        )
        augmented = np.concatenate(
            [real_router_scores, null_scores],
            axis=1,
        )

        row_indices, column_indices = linear_sum_assignment(
            -augmented
        )
        ordered_columns = np.full(
            NUM_SLOTS,
            -1,
            dtype=np.int64,
        )
        ordered_columns[row_indices] = column_indices

        decoded_assignment = [
            int(column)
            if int(column) < NUM_ROUTERS
            else NULL_CLASS
            for column in ordered_columns
        ]
        decoded_set = sorted(
            {
                value
                for value in decoded_assignment
                if value != NULL_CLASS
            }
        )
        total_score = float(
            sum(
                augmented[slot, column]
                for slot, column in enumerate(ordered_columns)
            )
        )

        assignments.append(decoded_assignment)
        sets.append(decoded_set)
        scores.append(total_score)

    return {
        "slot_assignments": assignments,
        "predicted_sets": sets,
        "assignment_scores": scores,
    }


def greedy_unique_decode(
    slot_logits: torch.Tensor,
) -> dict[str, Any]:
    """Hardware-oriented greedy unique router assignment."""
    if (
        slot_logits.ndim != 3
        or tuple(slot_logits.shape[1:]) != (
            NUM_SLOTS,
            NUM_SLOT_CLASSES,
        )
    ):
        raise ValueError("slot_logits must have shape [B,4,17]")

    logits = slot_logits.detach().cpu().numpy()
    assignments: list[list[int]] = []
    sets: list[list[int]] = []

    for sample_logits in logits:
        slot_order = np.argsort(
            -np.max(sample_logits, axis=1),
            kind="stable",
        )
        used_routers: set[int] = set()
        result = [NULL_CLASS] * NUM_SLOTS

        for slot in slot_order:
            class_order = np.argsort(
                -sample_logits[slot],
                kind="stable",
            )
            for class_id in class_order:
                class_id = int(class_id)
                if class_id == NULL_CLASS:
                    result[int(slot)] = NULL_CLASS
                    break
                if class_id not in used_routers:
                    result[int(slot)] = class_id
                    used_routers.add(class_id)
                    break

        assignments.append(result)
        sets.append(
            sorted(
                value
                for value in result
                if value != NULL_CLASS
            )
        )

    return {
        "slot_assignments": assignments,
        "predicted_sets": sets,
    }


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
    }
