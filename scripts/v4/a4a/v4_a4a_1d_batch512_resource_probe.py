#!/usr/bin/env python3
"""
V4-A4a-1D
Primary Batch-512 Resource Probe

This bounded probe exercises the frozen A4a-Slot-Full training path for exactly
two optimizer steps at batch size 512.

It verifies:
- exact original A3 train-only memmap/loader path
- 3,189 total / 1,076 trainable parameters
- finite real-data losses and gradients
- gradient clipping at norm 5
- bitwise-frozen A3 encoder and graph head
- GPU peak allocated/reserved memory
- host peak RSS
- checkpoint serialization and exact reload reproduction

No validation loader, test loader, threshold search, checkpoint selection or
development-test artifact is accessed.

A CUDA out-of-memory condition is recorded as an explicit HOLD authorizing the
already-frozen batch-256 fallback probe. Other errors remain failures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from v4_a4a_slot_model import (
    A4aFrozenSlotModel,
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

EXPECTED_TOTAL_PARAMETERS = 3189
EXPECTED_TRAINABLE_PARAMETERS = 1076
EXPECTED_BATCH_SIZE = 512
EXPECTED_PROBE_BATCHES = 2


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


def frozen_snapshot(model: A4aFrozenSlotModel) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.encoder.state_dict().items()
    }


def compare_frozen(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, Any]:
    changed: dict[str, float] = {}
    maximum = 0.0

    for name, tensor_before in before.items():
        difference = float(
            torch.max(
                torch.abs(
                    tensor_before - after[name].detach().cpu()
                )
            ).item()
        )
        maximum = max(maximum, difference)
        if difference != 0.0:
            changed[name] = difference

    return {
        "unchanged": not changed,
        "maximum_absolute_difference": maximum,
        "changed_tensors": changed,
    }


def host_peak_rss_mib() -> float:
    # Linux ru_maxrss is KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_marker(path: Path, text: str) -> None:
    path.write_text(text + "\n", encoding="utf-8")


def cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        isinstance(exc, torch.cuda.OutOfMemoryError)
        or "cuda out of memory" in message
        or "cublas_status_alloc_failed" in message
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--slot-module", type=Path, required=True)
    parser.add_argument("--smoke-script", type=Path, required=True)
    parser.add_argument(
        "--contract-freeze-script",
        type=Path,
        required=True,
    )
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--probe-batches", type=int, default=2)
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
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()

    if args.batch_size != EXPECTED_BATCH_SIZE:
        raise RuntimeError(
            "A4a-1D is frozen to batch size 512"
        )
    if args.probe_batches != EXPECTED_PROBE_BATCHES:
        raise RuntimeError(
            "A4a-1D is frozen to exactly two probe batches"
        )
    if args.num_workers == 0 and args.persistent_workers:
        raise ValueError(
            "persistent workers require num_workers > 0"
        )

    contract_path = (
        args.contract_dir / "A4A_1C_FULL_TRAINING_CONTRACT.json"
    )
    contract_lock_path = args.contract_dir / "A4A_1C_LOCK.json"
    contract_pass_path = (
        args.contract_dir / "V4_A4A_1C_FULL_TRAINING_CONTRACT_PASS"
    )

    required = [
        args.repo,
        args.data_dir,
        args.trainer,
        args.model_source,
        args.base_checkpoint,
        args.slot_module,
        args.smoke_script,
        args.contract_freeze_script,
        contract_path,
        contract_lock_path,
        contract_pass_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("A4a-1D FAIL: missing required paths", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(
            f"A4a-1D FAIL: output exists: {args.output_dir}",
            file=sys.stderr,
        )
        return 2

    args.output_dir.mkdir(parents=True)

    provenance = {
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
        "full_training_contract_sha256": sha256_file(
            contract_path
        ),
        "full_training_contract_lock_sha256": sha256_file(
            contract_lock_path
        ),
        "resource_probe_script_sha256": sha256_file(
            Path(__file__)
        ),
    }

    expected_hashes = {
        "trainer_sha256": EXPECTED_TRAINER_SHA,
        "model_source_sha256": EXPECTED_MODEL_SOURCE_SHA,
        "base_checkpoint_sha256": EXPECTED_BASE_CHECKPOINT_SHA,
        "slot_module_sha256": EXPECTED_SLOT_MODULE_SHA,
        "smoke_script_sha256": EXPECTED_SMOKE_SCRIPT_SHA,
        "contract_freeze_script_sha256": (
            EXPECTED_CONTRACT_FREEZE_SCRIPT_SHA
        ),
    }

    failures = [
        f"{key} mismatch: {provenance[key]} != {expected}"
        for key, expected in expected_hashes.items()
        if provenance[key] != expected
    ]

    contract = load_json(contract_path)
    contract_lock = load_json(contract_lock_path)

    if contract.get("status") != "PASS":
        failures.append("full-training contract is not PASS")
    if contract.get(
        "ready_for_a4a_1d_batch512_resource_probe"
    ) is not True:
        failures.append("contract does not authorize A4a-1D")
    if contract_lock.get("status") != (
        "A4A_1C_FULL_TRAINING_CONTRACT_COMPLETE"
    ):
        failures.append("A4a-1C lock is incomplete")
    if contract_lock.get("contract_sha256") != provenance[
        "full_training_contract_sha256"
    ]:
        failures.append("A4a-1C contract hash differs from lock")
    if contract.get("development_test_accessed") is not False:
        failures.append(
            "A4a-1C does not affirm no development-test access"
        )

    runtime = contract.get("runtime_contract", {})
    resource_contract = contract.get("resource_probe_contract", {})
    loss_contract = contract.get("loss_contract", {})

    frozen_checks = {
        "contract_batch_size_primary": (
            runtime.get("batch_size_primary") == EXPECTED_BATCH_SIZE
        ),
        "contract_probe_batch_size": (
            resource_contract.get(
                "primary_batch_size_to_probe"
            ) == EXPECTED_BATCH_SIZE
        ),
        "contract_probe_batches": (
            resource_contract.get("probe_batches")
            == EXPECTED_PROBE_BATCHES
        ),
        "contract_gradient_clip_norm": (
            float(runtime.get("gradient_clip_norm", -1.0)) == 5.0
        ),
        "contract_seed": runtime.get("seed") == 7,
        "contract_optimizer": runtime.get("optimizer") == "AdamW",
        "contract_learning_rate": (
            float(runtime.get("learning_rate", -1.0)) == 0.001
        ),
        "contract_weight_decay": (
            float(runtime.get("weight_decay", -1.0)) == 0.0001
        ),
    }
    for key, passed in frozen_checks.items():
        if not passed:
            failures.append(f"frozen contract check failed: {key}")

    if failures:
        failure = {
            "status": "FAIL",
            "failures": failures,
            "provenance": provenance,
            "training_performed": False,
            "validation_loader_constructed": False,
            "test_loader_constructed": False,
            "development_test_accessed": False,
        }
        write_json(
            args.output_dir / "resource_probe_failure.json",
            failure,
        )
        write_marker(
            args.output_dir
            / "V4_A4A_1D_BATCH512_RESOURCE_PROBE_FAIL",
            "V4_A4A_1D_BATCH512_RESOURCE_PROBE_FAIL",
        )
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        failure = {
            "status": "FAIL",
            "failures": [
                "CUDA is unavailable; batch-512 GPU viability cannot be probed"
            ],
            "training_performed": False,
            "development_test_accessed": False,
        }
        write_json(
            args.output_dir / "resource_probe_failure.json",
            failure,
        )
        write_marker(
            args.output_dir
            / "V4_A4A_1D_BATCH512_RESOURCE_PROBE_FAIL",
            "V4_A4A_1D_BATCH512_RESOURCE_PROBE_FAIL",
        )
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1

    set_seed(7)

    for import_root in (
        args.repo,
        args.repo / "scripts",
        args.trainer.parent,
        args.model_source.parent,
        args.slot_module.parent,
        args.smoke_script.parent,
    ):
        root_text = str(import_root.resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

    trainer_module = load_module(
        args.trainer,
        "resolved_v4_a3_trainer_for_a4a_1d",
    )
    a3_module = load_module(
        args.model_source,
        "resolved_v4_a3_model_for_a4a_1d",
    )
    smoke_module = load_module(
        args.smoke_script,
        "resolved_a4a_1b_smoke_for_a4a_1d",
    )

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total_memory = torch.cuda.mem_get_info(device)

    try:
        loader_args = argparse.Namespace(
            data_dir=str(args.data_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            seed=7,
        )
        objects = trainer_module.prepare_objects(
            loader_args,
            include_val=False,
        )

        if objects.get("val_loader") is not None:
            raise RuntimeError(
                "resource probe unexpectedly constructed validation loader"
            )

        unused_original_model = objects.pop("model", None)
        del unused_original_model
        torch.cuda.empty_cache()

        base_checkpoint = torch.load(
            args.base_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        base_model = a3_module.A3SourcePreserveModel()
        base_model.load_state_dict(
            extract_state_dict(base_checkpoint),
            strict=True,
        )
        base_model.eval()

        model = A4aFrozenSlotModel(base_model).to(device)
        model.train(True)

        parameter_count = count_parameters(model)
        if parameter_count["total"] != EXPECTED_TOTAL_PARAMETERS:
            raise RuntimeError(
                f"unexpected total parameters: {parameter_count}"
            )
        if (
            parameter_count["trainable"]
            != EXPECTED_TRAINABLE_PARAMETERS
        ):
            raise RuntimeError(
                f"unexpected trainable parameters: {parameter_count}"
            )

        adjacency = objects["adjacency"].to(
            device=device,
            dtype=torch.float32,
        )
        physical_mask = objects["mask"].to(
            device=device,
            dtype=torch.float32,
        )

        if not torch.equal(
            adjacency.detach().cpu(),
            base_checkpoint["A_hat"].detach().cpu(),
        ):
            raise RuntimeError(
                "pipeline adjacency differs from frozen checkpoint"
            )
        if not torch.equal(
            physical_mask.detach().cpu(),
            base_checkpoint[
                "physical_valid_port_mask"
            ].detach().cpu(),
        ):
            raise RuntimeError(
                "pipeline physical mask differs from frozen checkpoint"
            )

        loss_weights = SlotLossWeights(
            matching=float(
                loss_contract[
                    "permutation_invariant_matching_weight"
                ]
            ),
            membership=float(
                loss_contract[
                    "derived_union_membership_bce_weight"
                ]
            ),
            cardinality=float(
                loss_contract[
                    "derived_cardinality_cross_entropy_weight"
                ]
            ),
            duplicate=float(
                loss_contract[
                    "duplicate_slot_penalty_weight"
                ]
            ),
        )
        null_weight = float(
            loss_contract["null_target_weight"]
        )
        grad_clip_norm = float(
            runtime["gradient_clip_norm"]
        )

        optimizer = torch.optim.AdamW(
            model.slot_decoder.parameters(),
            lr=float(runtime["learning_rate"]),
            weight_decay=float(runtime["weight_decay"]),
        )

        frozen_before = frozen_snapshot(model)
        batch_reports: list[dict[str, Any]] = []
        last_probe_x = None

        started = time.perf_counter()
        for batch_number, batch in enumerate(
            objects["train_loader"],
            start=1,
        ):
            if batch_number > EXPECTED_PROBE_BATCHES:
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

            if int(x.shape[0]) != EXPECTED_BATCH_SIZE:
                raise RuntimeError(
                    f"probe batch {batch_number} has "
                    f"{x.shape[0]} samples, expected 512"
                )

            last_probe_x = x[:8].detach().cpu()

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
                        f"non-finite {name} loss in batch {batch_number}"
                    )

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()

            frozen_gradient_violations = []
            for name, parameter in model.encoder.named_parameters():
                if parameter.grad is not None:
                    frozen_gradient_violations.append(name)

            if frozen_gradient_violations:
                raise RuntimeError(
                    "frozen parameters received gradients: "
                    + ", ".join(frozen_gradient_violations)
                )

            missing_gradients = []
            nonfinite_gradients = []
            for name, parameter in model.slot_decoder.named_parameters():
                if parameter.grad is None:
                    missing_gradients.append(name)
                elif not torch.isfinite(parameter.grad).all():
                    nonfinite_gradients.append(name)

            if missing_gradients:
                raise RuntimeError(
                    "trainable parameters missing gradients: "
                    + ", ".join(missing_gradients)
                )
            if nonfinite_gradients:
                raise FloatingPointError(
                    "non-finite gradients: "
                    + ", ".join(nonfinite_gradients)
                )

            preclip_norm = torch.nn.utils.clip_grad_norm_(
                model.slot_decoder.parameters(),
                max_norm=grad_clip_norm,
            )
            preclip_norm_value = float(preclip_norm.detach().cpu())
            if not math.isfinite(preclip_norm_value):
                raise FloatingPointError(
                    f"non-finite gradient norm in batch {batch_number}"
                )

            optimizer.step()
            torch.cuda.synchronize(device)

            free_now, _ = torch.cuda.mem_get_info(device)
            batch_reports.append(
                {
                    "batch_number": batch_number,
                    "batch_size": int(x.shape[0]),
                    "losses": {
                        key: float(value.detach().cpu())
                        for key, value in losses.items()
                    },
                    "gradient_norm_before_clip": (
                        preclip_norm_value
                    ),
                    "gradient_clip_norm": grad_clip_norm,
                    "gpu_allocated_mib": (
                        torch.cuda.memory_allocated(device) / 1024**2
                    ),
                    "gpu_reserved_mib": (
                        torch.cuda.memory_reserved(device) / 1024**2
                    ),
                    "gpu_free_mib_after_batch": (
                        free_now / 1024**2
                    ),
                    "host_peak_rss_mib": host_peak_rss_mib(),
                    "slot_logit_min": float(
                        output["slot_logits"].detach().min().cpu()
                    ),
                    "slot_logit_max": float(
                        output["slot_logits"].detach().max().cpu()
                    ),
                }
            )

        elapsed = time.perf_counter() - started

        if len(batch_reports) != EXPECTED_PROBE_BATCHES:
            raise RuntimeError(
                f"processed {len(batch_reports)} batches, expected 2"
            )
        if last_probe_x is None:
            raise RuntimeError("probe input was not captured")

        frozen_after = {
            name: tensor.detach().cpu()
            for name, tensor in model.encoder.state_dict().items()
        }
        frozen_comparison = compare_frozen(
            frozen_before,
            frozen_after,
        )
        if frozen_comparison["unchanged"] is not True:
            raise RuntimeError(
                f"frozen encoder changed: {frozen_comparison}"
            )

        model.eval()
        with torch.no_grad():
            pre_save = model(
                last_probe_x.to(device),
                adjacency,
                physical_mask,
            )
            pre_save_slot = (
                pre_save["slot_logits"].detach().cpu()
            )
            pre_save_graph = (
                pre_save["graph_logits"].detach().cpu()
            )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "A_hat": adjacency.detach().cpu(),
            "physical_valid_port_mask": (
                physical_mask.detach().cpu()
            ),
            "parameter_count": parameter_count,
            "batch_size": EXPECTED_BATCH_SIZE,
            "probe_batches": EXPECTED_PROBE_BATCHES,
            "provenance": provenance,
            "resource_probe_only": True,
            "validation_loader_constructed": False,
            "test_loader_constructed": False,
            "test_evaluated": False,
        }
        checkpoint_path = (
            args.output_dir / "batch512_probe_checkpoint.pt"
        )
        trainer_module.atomic_torch_save(
            checkpoint_path,
            checkpoint,
        )

        reloaded_checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        reloaded_base = a3_module.A3SourcePreserveModel()
        reloaded_base.load_state_dict(
            extract_state_dict(base_checkpoint),
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
                last_probe_x.to(device),
                adjacency,
                physical_mask,
            )

        slot_reload_difference = float(
            torch.max(
                torch.abs(
                    pre_save_slot
                    - reloaded_output[
                        "slot_logits"
                    ].detach().cpu()
                )
            ).item()
        )
        graph_reload_difference = float(
            torch.max(
                torch.abs(
                    pre_save_graph
                    - reloaded_output[
                        "graph_logits"
                    ].detach().cpu()
                )
            ).item()
        )

        if (
            slot_reload_difference != 0.0
            or graph_reload_difference != 0.0
        ):
            raise RuntimeError(
                "checkpoint reload changed outputs: "
                f"slot={slot_reload_difference}, "
                f"graph={graph_reload_difference}"
            )

        torch.cuda.synchronize(device)
        free_after, _ = torch.cuda.mem_get_info(device)

        report = {
            "status": "PASS",
            "designation": (
                "V4-A4a-1D Primary Batch-512 Resource Probe"
            ),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "parameter_count": parameter_count,
            "configuration": {
                "batch_size": EXPECTED_BATCH_SIZE,
                "probe_batches": EXPECTED_PROBE_BATCHES,
                "seed": 7,
                "optimizer": "AdamW",
                "learning_rate": float(runtime["learning_rate"]),
                "weight_decay": float(runtime["weight_decay"]),
                "gradient_clip_norm": grad_clip_norm,
                "null_target_weight": null_weight,
                "num_workers": args.num_workers,
                "pin_memory": args.pin_memory,
                "persistent_workers": args.persistent_workers,
                "prefetch_factor": args.prefetch_factor,
                "automatic_mixed_precision": False,
            },
            "batch_reports": batch_reports,
            "resource_summary": {
                "elapsed_seconds": elapsed,
                "gpu_total_mib": total_memory / 1024**2,
                "gpu_free_mib_before": free_before / 1024**2,
                "gpu_free_mib_after": free_after / 1024**2,
                "gpu_peak_allocated_mib": (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                ),
                "gpu_peak_reserved_mib": (
                    torch.cuda.max_memory_reserved(device) / 1024**2
                ),
                "host_peak_rss_mib": host_peak_rss_mib(),
            },
            "integrity_checks": {
                "processed_exactly_two_batches": True,
                "every_batch_size_is_512": True,
                "all_losses_finite": True,
                "all_gradients_finite": True,
                "gradient_clipping_exercised": True,
                "frozen_encoder_unchanged": frozen_comparison,
                "checkpoint_serialization_pass": True,
                "checkpoint_reload_slot_logit_max_abs_diff": (
                    slot_reload_difference
                ),
                "checkpoint_reload_graph_logit_max_abs_diff": (
                    graph_reload_difference
                ),
                "parameter_count_pass": True,
                "validation_loader_constructed": False,
                "test_loader_constructed": False,
            },
            "decision": "batch512_primary_authorized",
            "fallback_batch256_authorized": False,
            "training_performed": True,
            "training_scope": "two optimizer steps for resource probe only",
            "validation_accessed": False,
            "validation_selection_performed": False,
            "threshold_search_performed": False,
            "test_loader_constructed": False,
            "test_evaluated": False,
            "development_test_accessed": False,
            "resource_probe_checkpoint_is_not_a_model_candidate": True,
            "ready_for_a4a_2a_primary_training_implementation": True,
            "provenance": provenance,
        }

        report_path = (
            args.output_dir / "batch512_resource_probe.json"
        )
        write_json(report_path, report)

        lock = {
            "status": "A4A_1D_BATCH512_RESOURCE_PROBE_COMPLETE",
            "decision": "batch512_primary_authorized",
            "resource_probe_report_sha256": sha256_file(
                report_path
            ),
            "resource_probe_checkpoint_sha256": sha256_file(
                checkpoint_path
            ),
            "full_training_contract_sha256": provenance[
                "full_training_contract_sha256"
            ],
            "base_checkpoint_sha256": provenance[
                "base_checkpoint_sha256"
            ],
            "slot_module_sha256": provenance[
                "slot_module_sha256"
            ],
            "validation_accessed": False,
            "test_loader_constructed": False,
            "development_test_accessed": False,
        }
        write_json(
            args.output_dir / "A4A_1D_LOCK.json",
            lock,
        )
        write_marker(
            args.output_dir
            / "V4_A4A_1D_BATCH512_RESOURCE_PROBE_PASS",
            "V4_A4A_1D_BATCH512_RESOURCE_PROBE_PASS",
        )

        print(json.dumps(report, indent=2, sort_keys=True))
        print("V4_A4A_1D_BATCH512_RESOURCE_PROBE_PASS")
        return 0

    except BaseException as exc:
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        if cuda_oom(exc):
            hold = {
                "status": "HOLD",
                "designation": (
                    "V4-A4a-1D Primary Batch-512 Resource Probe"
                ),
                "reason": "cuda_out_of_memory_at_batch512",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "decision": (
                    "batch512_not_authorized; run the frozen "
                    "batch256 fallback resource probe"
                ),
                "fallback_batch_size": 256,
                "training_scope": (
                    "bounded resource probe only; no candidate produced"
                ),
                "validation_accessed": False,
                "test_loader_constructed": False,
                "development_test_accessed": False,
                "provenance": provenance,
            }
            hold_path = (
                args.output_dir / "batch512_resource_probe_hold.json"
            )
            write_json(hold_path, hold)
            write_json(
                args.output_dir / "A4A_1D_HOLD_LOCK.json",
                {
                    "status": (
                        "A4A_1D_BATCH512_RESOURCE_PROBE_HOLD"
                    ),
                    "reason": hold["reason"],
                    "hold_report_sha256": sha256_file(hold_path),
                    "fallback_batch_size": 256,
                    "development_test_accessed": False,
                },
            )
            write_marker(
                args.output_dir
                / "V4_A4A_1D_BATCH512_RESOURCE_PROBE_HOLD",
                "V4_A4A_1D_BATCH512_RESOURCE_PROBE_HOLD",
            )
            print(json.dumps(hold, indent=2), file=sys.stderr)
            print("V4_A4A_1D_BATCH512_RESOURCE_PROBE_HOLD")
            return 3

        failure = {
            "status": "FAIL",
            "designation": (
                "V4-A4a-1D Primary Batch-512 Resource Probe"
            ),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "validation_accessed": False,
            "test_loader_constructed": False,
            "development_test_accessed": False,
            "provenance": provenance,
        }
        failure_path = (
            args.output_dir / "batch512_resource_probe_failure.json"
        )
        write_json(failure_path, failure)
        write_json(
            args.output_dir / "A4A_1D_FAILURE_LOCK.json",
            {
                "status": "A4A_1D_BATCH512_RESOURCE_PROBE_FAILED",
                "failure_report_sha256": sha256_file(failure_path),
                "development_test_accessed": False,
            },
        )
        write_marker(
            args.output_dir
            / "V4_A4A_1D_BATCH512_RESOURCE_PROBE_FAIL",
            "V4_A4A_1D_BATCH512_RESOURCE_PROBE_FAIL",
        )
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
