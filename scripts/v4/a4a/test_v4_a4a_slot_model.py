#!/usr/bin/env python3
"""Pytest unit tests for the V4-A4a slot primitives."""

from __future__ import annotations

import torch

from v4_a4a_slot_model import (
    A4aLowRankSlotDecoder,
    NULL_CLASS,
    derived_cardinality_probabilities,
    exact_unique_constrained_decode,
    permutation_invariant_slot_matching_loss,
)


def favored(assignments):
    logits = torch.full((len(assignments), 4, 17), -8.0)
    for sample, assignment in enumerate(assignments):
        for slot, class_id in enumerate(assignment):
            logits[sample, slot, class_id] = 8.0
    return logits


def test_decoder_shapes_and_normalization():
    decoder = A4aLowRankSlotDecoder()
    output = decoder(
        torch.randn(3, 16, 32),
        torch.randn(3, 64),
    )
    assert output["slot_logits"].shape == (3, 4, 17)
    assert output["membership_probabilities"].shape == (3, 16)
    assert output["cardinality_probabilities"].shape == (3, 5)
    assert torch.allclose(
        output["slot_probabilities"].sum(dim=-1),
        torch.ones(3, 4),
        atol=1e-6,
    )
    assert torch.allclose(
        output["cardinality_probabilities"].sum(dim=1),
        torch.ones(3),
        atol=1e-6,
    )


def test_permutation_invariant_matching():
    truth = torch.zeros(1, 16)
    truth[0, 3] = 1
    truth[0, 7] = 1

    left = permutation_invariant_slot_matching_loss(
        favored([[3, 7, NULL_CLASS, NULL_CLASS]]),
        truth,
    )
    right = permutation_invariant_slot_matching_loss(
        favored([[7, NULL_CLASS, 3, NULL_CLASS]]),
        truth,
    )
    assert torch.allclose(left, right, atol=1e-5)


def test_all_null_cardinality():
    probabilities = torch.zeros(2, 4, 17)
    probabilities[..., NULL_CLASS] = 1.0
    cardinality = derived_cardinality_probabilities(probabilities)
    expected = torch.zeros_like(cardinality)
    expected[:, 0] = 1.0
    assert torch.allclose(cardinality, expected, atol=1e-7)


def test_exact_decode_suppresses_duplicates():
    logits = torch.full((1, 4, 17), -10.0)
    logits[0, 0, 3] = 10.0
    logits[0, 1, 3] = 9.0
    logits[0, 1, NULL_CLASS] = 8.0
    logits[0, 2, 7] = 10.0
    logits[0, 3, NULL_CLASS] = 10.0

    decoded = exact_unique_constrained_decode(logits)
    assert decoded["predicted_sets"][0] == [3, 7]
