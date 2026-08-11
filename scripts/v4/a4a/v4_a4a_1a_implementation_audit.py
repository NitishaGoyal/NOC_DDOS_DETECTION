#!/usr/bin/env python3
"""
V4-A4a-1A slot implementation correctness audit.

This audit imports the exact frozen A3 source, loads the exact checkpoint,
constructs the trimmed 3,189-parameter A4a slot model and runs deterministic
synthetic correctness tests.

No dataset split is read. No training loop, validation selection, threshold
search, or development-test evaluation occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from v4_a4a_slot_model import (
    A4aFrozenSlotModel,
    A4aLowRankSlotDecoder,
    NULL_CLASS,
    SlotLossWeights,
    count_parameters,
    derived_cardinality_probabilities,
    derived_membership_probabilities,
    duplicate_slot_penalty,
    exact_unique_constrained_decode,
    greedy_unique_decode,
    permutation_invariant_slot_matching_loss,
    slot_set_losses,
)


EXPECTED_MODEL_SOURCE_SHA = (
    "5ebcf80385b95f329faf935cad8039b64"
    "afd79f464903308eb07e674d78a1704"
)
EXPECTED_BASE_CHECKPOINT_SHA = (
    "f61c1add6c057f7f53dd34fb1f9f4e95"
    "b01cefd5053e0e42bc100c76dac7f923"
)
EXPECTED_SLOT_DECODER_PARAMETERS = 1076
EXPECTED_DEPLOYED_PARAMETERS = 3189


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


def assert_close(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    tolerance: float = 1e-6,
    message: str,
) -> None:
    if not torch.allclose(left, right, atol=tolerance, rtol=0.0):
        raise AssertionError(message)


def make_favored_logits(
    assignments: list[list[int]],
    high: float = 8.0,
    low: float = -8.0,
) -> torch.Tensor:
    logits = torch.full(
        (len(assignments), 4, 17),
        low,
        dtype=torch.float32,
    )
    for sample, assignment in enumerate(assignments):
        for slot, class_id in enumerate(assignment):
            logits[sample, slot, class_id] = high
    return logits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-lock", type=Path, required=True)
    parser.add_argument("--slot-module", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()

    required = [
        args.repo,
        args.checkpoint,
        args.model_source,
        args.contract,
        args.contract_lock,
        args.slot_module,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("A4a-1A FAIL: missing paths", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(
            f"A4a-1A FAIL: output exists: {args.output_dir}",
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True)

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7)

    contract = load_json(args.contract)
    contract_lock = load_json(args.contract_lock)

    checkpoint_sha = sha256_file(args.checkpoint)
    model_source_sha = sha256_file(args.model_source)
    slot_module_sha = sha256_file(args.slot_module)
    contract_sha = sha256_file(args.contract)

    failures: list[str] = []
    tests: dict[str, Any] = {}

    if checkpoint_sha != EXPECTED_BASE_CHECKPOINT_SHA:
        failures.append("unexpected frozen A3 checkpoint hash")
    if model_source_sha != EXPECTED_MODEL_SOURCE_SHA:
        failures.append("unexpected frozen A3 model-source hash")
    if contract_lock.get("contract_sha256") != contract_sha:
        failures.append("slot contract hash mismatch")
    if contract.get("status") != "PASS":
        failures.append("slot contract did not pass")

    if failures:
        payload = {"status": "FAIL", "failures": failures}
        (args.output_dir / "implementation_failure.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    for import_root in (
        args.repo,
        args.repo / "scripts",
        args.model_source.parent,
        args.slot_module.parent,
    ):
        import_text = str(import_root.resolve())
        if import_text not in sys.path:
            sys.path.insert(0, import_text)

    a3_module = load_module(
        args.model_source,
        "resolved_v4_a3_sourcepreserve_for_a4a_1a",
    )
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    base_model = a3_module.A3SourcePreserveModel()
    base_model.load_state_dict(
        extract_state_dict(checkpoint),
        strict=True,
    )
    base_model.eval()

    slot_model = A4aFrozenSlotModel(base_model).to(device)
    slot_model.train(True)

    parameter_counts = count_parameters(slot_model)
    decoder_counts = count_parameters(slot_model.slot_decoder)

    tests["parameter_count"] = {
        "model": parameter_counts,
        "slot_decoder": decoder_counts,
        "expected_model_total": EXPECTED_DEPLOYED_PARAMETERS,
        "expected_model_trainable": EXPECTED_SLOT_DECODER_PARAMETERS,
    }

    if parameter_counts["total"] != EXPECTED_DEPLOYED_PARAMETERS:
        failures.append(
            f"deployed parameter count {parameter_counts['total']} "
            f"!= {EXPECTED_DEPLOYED_PARAMETERS}"
        )
    if parameter_counts["trainable"] != EXPECTED_SLOT_DECODER_PARAMETERS:
        failures.append(
            f"trainable parameter count {parameter_counts['trainable']} "
            f"!= {EXPECTED_SLOT_DECODER_PARAMETERS}"
        )
    if decoder_counts["total"] != EXPECTED_SLOT_DECODER_PARAMETERS:
        failures.append("slot decoder parameter count mismatch")

    decoder = A4aLowRankSlotDecoder()
    decoder_output = decoder(
        torch.randn(5, 16, 32),
        torch.randn(5, 64),
    )

    probability_sums = decoder_output["slot_probabilities"].sum(dim=-1)
    assert_close(
        probability_sums,
        torch.ones_like(probability_sums),
        message="slot probabilities do not sum to one",
    )
    tests["slot_probability_normalization"] = True

    cardinality_sums = decoder_output[
        "cardinality_probabilities"
    ].sum(dim=1)
    assert_close(
        cardinality_sums,
        torch.ones_like(cardinality_sums),
        message="cardinality probabilities do not sum to one",
    )
    tests["cardinality_probability_normalization"] = True

    membership = decoder_output["membership_probabilities"]
    if torch.any(membership < 0) or torch.any(membership > 1):
        failures.append("derived membership probabilities leave [0,1]")
    tests["membership_probability_range"] = True

    all_null_probabilities = torch.zeros(2, 4, 17)
    all_null_probabilities[..., NULL_CLASS] = 1.0
    all_null_cardinality = derived_cardinality_probabilities(
        all_null_probabilities
    )
    expected_zero = torch.zeros_like(all_null_cardinality)
    expected_zero[:, 0] = 1.0
    assert_close(
        all_null_cardinality,
        expected_zero,
        message="all-NULL slots do not imply cardinality zero",
    )
    tests["all_null_cardinality"] = True

    all_router_probabilities = torch.zeros(1, 4, 17)
    for slot, router in enumerate([1, 3, 7, 12]):
        all_router_probabilities[0, slot, router] = 1.0
    all_router_cardinality = derived_cardinality_probabilities(
        all_router_probabilities
    )
    expected_four = torch.zeros_like(all_router_cardinality)
    expected_four[:, 4] = 1.0
    assert_close(
        all_router_cardinality,
        expected_four,
        message="four non-NULL slots do not imply cardinality four",
    )
    tests["four_router_cardinality"] = True

    y_two = torch.zeros(1, 16)
    y_two[0, 3] = 1
    y_two[0, 7] = 1

    ordered_logits = make_favored_logits(
        [[3, 7, NULL_CLASS, NULL_CLASS]]
    )
    permuted_logits = make_favored_logits(
        [[7, NULL_CLASS, 3, NULL_CLASS]]
    )
    ordered_loss = permutation_invariant_slot_matching_loss(
        ordered_logits,
        y_two,
    )
    permuted_loss = permutation_invariant_slot_matching_loss(
        permuted_logits,
        y_two,
    )
    assert_close(
        ordered_loss,
        permuted_loss,
        tolerance=1e-5,
        message="matching loss is not slot-permutation invariant",
    )
    tests["permutation_invariant_matching"] = {
        "ordered_loss": float(ordered_loss),
        "permuted_loss": float(permuted_loss),
    }

    normal_logits = make_favored_logits(
        [[NULL_CLASS] * 4]
    )
    normal_decoded = exact_unique_constrained_decode(
        normal_logits
    )
    if normal_decoded["predicted_sets"][0] != []:
        failures.append("all-NULL exact decoding did not produce empty set")
    tests["normal_all_null_decode"] = normal_decoded

    duplicate_logits = torch.full((1, 4, 17), -10.0)
    duplicate_logits[0, 0, 3] = 10.0
    duplicate_logits[0, 1, 3] = 9.0
    duplicate_logits[0, 1, NULL_CLASS] = 8.0
    duplicate_logits[0, 2, 7] = 10.0
    duplicate_logits[0, 3, NULL_CLASS] = 10.0

    exact_duplicate_decode = exact_unique_constrained_decode(
        duplicate_logits
    )
    predicted_set = exact_duplicate_decode["predicted_sets"][0]
    if predicted_set != [3, 7]:
        failures.append(
            "exact unique constrained decode failed duplicate case: "
            f"{predicted_set}"
        )
    if len(predicted_set) != len(set(predicted_set)):
        failures.append("exact decode contains duplicate router")
    tests["exact_duplicate_suppression"] = exact_duplicate_decode

    greedy_duplicate_decode = greedy_unique_decode(
        duplicate_logits
    )
    greedy_set = greedy_duplicate_decode["predicted_sets"][0]
    if len(greedy_set) != len(set(greedy_set)):
        failures.append("greedy decode contains duplicate router")
    tests["greedy_duplicate_suppression"] = greedy_duplicate_decode

    duplicate_high = torch.softmax(
        make_favored_logits(
            [[3, 3, NULL_CLASS, NULL_CLASS]],
            high=8.0,
            low=-8.0,
        ),
        dim=-1,
    )
    duplicate_low = torch.softmax(
        make_favored_logits(
            [[3, 7, NULL_CLASS, NULL_CLASS]],
            high=8.0,
            low=-8.0,
        ),
        dim=-1,
    )
    penalty_high = duplicate_slot_penalty(duplicate_high)
    penalty_low = duplicate_slot_penalty(duplicate_low)
    if not float(penalty_high) > float(penalty_low):
        failures.append("duplicate penalty does not punish slot collision")
    tests["duplicate_penalty_ordering"] = {
        "collision": float(penalty_high),
        "unique": float(penalty_low),
    }

    batch_size = 4
    x = torch.randn(
        batch_size,
        16,
        8,
        24,
        device=device,
    )
    a_hat = checkpoint["A_hat"].to(
        device=device,
        dtype=torch.float32,
    )
    physical_mask = checkpoint["physical_valid_port_mask"].to(
        device=device,
        dtype=torch.float32,
    )
    y_node = torch.zeros(
        batch_size,
        16,
        device=device,
    )
    y_node[1, 3] = 1
    y_node[2, [3, 7]] = 1
    y_node[3, [1, 3, 7, 12]] = 1

    output = slot_model(x, a_hat, physical_mask)
    losses = slot_set_losses(
        output["slot_logits"],
        y_node,
        weights=SlotLossWeights(),
    )

    if not torch.isfinite(losses["total"]):
        failures.append("synthetic total loss is not finite")
    losses["total"].backward()

    frozen_gradient_violations = []
    for name, parameter in slot_model.encoder.named_parameters():
        if parameter.grad is not None:
            frozen_gradient_violations.append(name)

    trainable_without_gradient = []
    nonfinite_trainable_gradients = []
    trainable_gradient_norms = {}

    for name, parameter in slot_model.slot_decoder.named_parameters():
        if parameter.grad is None:
            trainable_without_gradient.append(name)
            continue
        norm = float(parameter.grad.detach().norm().cpu())
        trainable_gradient_norms[name] = norm
        if not math.isfinite(norm):
            nonfinite_trainable_gradients.append(name)

    if frozen_gradient_violations:
        failures.append(
            "frozen encoder received gradients: "
            + ", ".join(frozen_gradient_violations)
        )
    if trainable_without_gradient:
        failures.append(
            "slot parameters missing gradients: "
            + ", ".join(trainable_without_gradient)
        )
    if nonfinite_trainable_gradients:
        failures.append(
            "non-finite slot gradients: "
            + ", ".join(nonfinite_trainable_gradients)
        )

    tests["synthetic_backward"] = {
        "losses": {
            key: float(value.detach().cpu())
            for key, value in losses.items()
        },
        "frozen_gradient_violations": frozen_gradient_violations,
        "trainable_without_gradient": trainable_without_gradient,
        "nonfinite_trainable_gradients": nonfinite_trainable_gradients,
        "trainable_gradient_norms": trainable_gradient_norms,
    }

    report = {
        "status": "PASS" if not failures else "FAIL",
        "designation": "V4-A4a-1A Slot Implementation Correctness Audit",
        "device": str(device),
        "failures": failures,
        "tests": tests,
        "provenance": {
            "checkpoint_sha256": checkpoint_sha,
            "model_source_sha256": model_source_sha,
            "contract_sha256": contract_sha,
            "contract_lock_sha256": sha256_file(args.contract_lock),
            "slot_module_sha256": slot_module_sha,
            "audit_script_sha256": sha256_file(Path(__file__)),
        },
        "training_loop_executed": False,
        "dataset_accessed": False,
        "validation_accessed": False,
        "development_test_accessed": False,
        "threshold_search_performed": False,
        "ready_for_a4a_1b_smoke_training": not failures,
    }

    report_path = args.output_dir / "implementation_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lock = {
        "status": (
            "A4A_1A_SLOT_IMPLEMENTATION_COMPLETE"
            if not failures
            else "A4A_1A_SLOT_IMPLEMENTATION_FAILED"
        ),
        "checkpoint_sha256": checkpoint_sha,
        "model_source_sha256": model_source_sha,
        "contract_sha256": contract_sha,
        "slot_module_sha256": slot_module_sha,
        "implementation_audit_sha256": sha256_file(report_path),
        "dataset_accessed": False,
        "validation_accessed": False,
        "development_test_accessed": False,
        "training_loop_executed": False,
    }
    (args.output_dir / "A4A_1A_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    marker = (
        "V4_A4A_1A_SLOT_IMPLEMENTATION_PASS"
        if not failures
        else "V4_A4A_1A_SLOT_IMPLEMENTATION_FAIL"
    )
    (args.output_dir / marker).write_text(
        marker + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    print(marker)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
