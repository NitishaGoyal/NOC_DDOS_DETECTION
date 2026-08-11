#!/usr/bin/env python3
"""V5 P2 D5 finalized-architecture E1 integration and validation.

D5 reconnects the actual finalized architecture and frozen checkpoint to the
validation loader, regenerates every E1 validation logit on CPU, and proves
bitwise equality with the immutable E1 archive.

It then feeds the regenerated arrays through the already frozen D4 Raw/A0
functions and proves exact equality with the committed D4 E2/E3 outputs and
metrics. Because the regenerated logits are bitwise identical to the E1 archive,
D5 also binds the completed D3 certified A1 archive to the actual finalized
architecture and verifies exact equality with D4 E4.

This stage is validation-only. It never forms or inspects the P2 test path,
creates no authorization, performs no threshold/model/decoder selection, and
does not execute the decoder.

The inference pass is interruption-safe. Each validation batch is committed
atomically and hash-recorded. Re-running the same command resumes after
verifying every existing chunk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader


STAGE = "V5_P2_EVALUATOR_D5_FINALIZED_ARCHITECTURE_E1_INTEGRATION"
COMPLETE = f"{STAGE}_COMPLETE"

SELECTED_SEED = 107
SELECTED_EPOCH = 25
EXPECTED_ITEMS = 12_528
EXPECTED_BATCHES = 49
EXPECTED_PARAMETER_COUNT = 59_785
EXPECTED_STATE_TENSOR_COUNT = 49
EXPECTED_STATE_NUMEL = 59_881
PAIR_BLOCK_BATCH_SIZE = 128
ITEM_BATCH_SIZE = 256

REQUIRED_ITEM_KEYS = {
    "x",
    "physical_port_mask",
    "y_attack",
    "y_attacker_count",
    "y_source",
    "y_transit",
    "y_victim",
    "y_attack_path",
    "role_mask",
}
REQUIRED_OUTPUT_KEYS = {
    "attack_logits",
    "count_logits",
    "source_logits",
    "transit_logits",
    "victim_logits",
    "path_logits",
}
ROLE_NAMES = ("source", "transit", "victim", "path")
LOGIT_OUTPUT_KEY = {
    "graph_logits": "attack_logits",
    "count_logits": "count_logits",
    "source_logits": "source_logits",
    "transit_logits": "transit_logits",
    "victim_logits": "victim_logits",
    "path_logits": "path_logits",
}
LABEL_BATCH_KEY = {
    "y_graph": "y_attack",
    "y_attacker_count": "y_attacker_count",
    "y_source": "y_source",
    "y_transit": "y_transit",
    "y_victim": "y_victim",
    "y_path": "y_attack_path",
    "role_mask": "role_mask",
}

EXPECTED_HASHES = {
    "loader": "2725ff993f4f03ebee3d5ffb775b1fedd6b131a9c4c89ed8049f45b249c24ac2",
    "b3_model": "56ee3207d039b60e8e3a898a689cd8e361247c86b7450a3a5e95f423ebe30def",
    "task_d_model": "ccdfcb74c1ddab7f20b98c922444873ad76c204ec4b5fd6827d52fb716f11fd9",
    "b2_training_source": "8fa354698056653a4ff9d467047fced850ff36c3bdef557b08cbd58aa3d3ae1c",
    "checkpoint": "82314baedb4bf3969842abf9076679682369daae2e57db07d4b4aeaa8623f7cc",
    "edge_index": "f6b8050bc158de509b0ff1c5d1d7cb1ffe32c08b0f2287398270b1b891b57aff",
    "physical_mask": "a93f81a9ce4315d0eef911f9b9c45529ceb9b2da726dd578a247b46d260e6a3a",
    "pair_manifest": "f42f40446d03161a6932ee060f6d8c859f894cb5075d1fc926ea161461413ab5",
    "e1_manifest": "ac7a38fd1f4b00b578254d9253418655aa78b6cc7a7a1d65f78ba15487b14604",
    "d1_contract": "b63efbfe1272fb7f6ab922645182fba86e28d7c8f42a70e0f3ac453b88bec71f",
    "d2_artifact": "fcb78ae4059f49e4c78d4e3cf603160f7d8bad72857d6ef6b777cc4de0ab7055",
    "d3_artifact": "11339c6afcec5275a310219e05a360638121cdfabd2f02589273e2c84af56def",
    "d4_artifact": "0be3b06a31b30fabda6e7f6f5582be9cf4bfde105a4d01bf893dd4bcc0545a3b",
    "d4_runner": "ddb61f04a4b16da4725d365e43526d05d3cb8eafa64d62eda8a460cda6aec70c",
    "threshold_contract": "b9acb9dac24cc1d0fe2cdae32525a4c0bdddc40e1607824599ad5bc3a446d957",
}
EXPECTED_POLICY_ID = "V5P2-D1-6ce8cc7c76b8d2cff367631077d65f20"
EXPECTED_E1_ARRAYS = {
    "graph_logits": ((EXPECTED_ITEMS,), np.dtype("float32")),
    "count_logits": ((EXPECTED_ITEMS, 4), np.dtype("float32")),
    "source_logits": ((EXPECTED_ITEMS, 16), np.dtype("float32")),
    "transit_logits": ((EXPECTED_ITEMS, 16), np.dtype("float32")),
    "victim_logits": ((EXPECTED_ITEMS, 16), np.dtype("float32")),
    "path_logits": ((EXPECTED_ITEMS, 16), np.dtype("float32")),
    "y_graph": ((EXPECTED_ITEMS,), np.dtype("uint8")),
    "y_attacker_count": ((EXPECTED_ITEMS,), np.dtype("int64")),
    "y_source": ((EXPECTED_ITEMS, 16), np.dtype("uint8")),
    "y_transit": ((EXPECTED_ITEMS, 16), np.dtype("uint8")),
    "y_victim": ((EXPECTED_ITEMS, 16), np.dtype("uint8")),
    "y_path": ((EXPECTED_ITEMS, 16), np.dtype("uint8")),
    "role_mask": ((EXPECTED_ITEMS, 16), np.dtype("uint8")),
    "validation_item_index": ((EXPECTED_ITEMS,), np.dtype("int64")),
    "manifest_row_index": ((EXPECTED_ITEMS,), np.dtype("int64")),
}
EXPECTED_STATIC_MASK_SHA = (
    "3d18729ec5c1e72ff166853ca92b2d9d37c892bf4b8e40bfa46c930adbdb33af"
)
EXPECTED_STABLE_ITEMS_SHA = (
    "f44938b8fb10217d62c0823621a3d1419926f6833e705f17c2884805ba67195b"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=True,
        ).encode("utf-8")
    ).hexdigest()


def sha256_array_payload(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: observed={observed}, expected={expected}"
        )
    return observed


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n",
    )


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **{key: np.asarray(value) for key, value in arrays.items()},
    )
    os.replace(temporary, path)


def import_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def repository_paths(root: Path) -> dict[str, Path]:
    reports = root / "reports/v5"
    artifacts = root / "artifacts/v5"
    return {
        "loader": root / "src/data/v5_p2_pair_aligned_primary58_dataset.py",
        "b3_model": root / "src/models/v5_p2_b3_conv1d_only_count4.py",
        "task_d_model": root / "src/models/v5_p2_task_d_full_multitask_count4.py",
        "b2_training_source": root / "scripts/v5/p2/train_v5_p2_b2_single_seed.py",
        "checkpoint": (
            reports
            / "p2_task_d_checkpoint_selection_freeze"
            / "selected_graphconv_checkpoint.pt"
        ),
        "edge_index": (
            reports
            / "p2_g1a_r2a_canonical_static_topology_contract"
            / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy"
        ),
        "physical_mask": (
            reports
            / "p2_a2_r2_feature_normalization_mask_contract"
            / "V5_P2_A2_R2_TOPOLOGY_DERIVED_BOOLEAN_PORT_MASK.pt"
        ),
        "pair_manifest": (
            reports
            / "p2_a1_r2_pair_aligned_window_contract"
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
        ),
        "e1_manifest": (
            artifacts
            / "p2_e1_immutable_validation_logit_archive"
            / "ARCHIVE_MANIFEST.json"
        ),
        "d1_contract": (
            artifacts
            / "p2_decoder_d1_prospective_numerical_policy_freeze"
            / "V5_P2_D1_PROSPECTIVE_NUMERICAL_CERTIFICATION_POLICY.json"
        ),
        "d2_artifact": (
            artifacts
            / "p2_decoder_d2_certified_implementation"
            / "V5_P2_DECODER_D2_CERTIFIED_IMPLEMENTATION.json"
        ),
        "d3_artifact": (
            artifacts
            / "p2_decoder_d3_full_validation_certification_proof"
            / "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF.json"
        ),
        "d4_artifact": (
            artifacts
            / "p2_evaluator_d4_staged_raw_a0_a1_validation"
            / "V5_P2_EVALUATOR_D4_STAGED_RAW_A0_A1_VALIDATION.json"
        ),
        "d4_runner": (
            root
            / "scripts/v5/p2"
            / "run_v5_p2_evaluator_d4_staged_raw_a0_a1_validation.py"
        ),
        "threshold_contract": (
            artifacts
            / "p2_e2_raw_threshold_freeze"
            / "V5_P2_E2_FROZEN_RAW_THRESHOLDS.json"
        ),
    }


def load_edge_index(path: Path) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.int64 or tuple(array.shape) != (2, 48):
        raise RuntimeError(
            f"edge index contract changed: dtype={array.dtype}, shape={array.shape}"
        )
    tensor = torch.from_numpy(np.ascontiguousarray(array)).long()
    pairs = {(int(a), int(b)) for a, b in tensor.t().tolist()}
    if len(pairs) != 48:
        raise RuntimeError("edge index does not contain 48 unique directed edges")
    return tensor.contiguous()


def load_frozen_physical_mask(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or not isinstance(value.get("mask"), torch.Tensor):
        raise RuntimeError("frozen physical-mask artifact lacks tensor key 'mask'")
    mask = value["mask"].detach().cpu().bool().contiguous()
    if tuple(mask.shape) != (16, 10) or int(mask.sum().item()) != 128:
        raise RuntimeError("frozen physical-mask contract changed")
    return mask


def build_model(
    *,
    paths: dict[str, Path],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    b3_module = import_source(paths["b3_model"], "v5_p2_d5_b3_model")
    task_module = import_source(paths["task_d_model"], "v5_p2_d5_task_d_model")
    edge = load_edge_index(paths["edge_index"])

    reference = b3_module.P2B3Conv1DOnlyCount4()
    model = task_module.P2TaskDGraphConvCount4(reference, edge).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"model parameter count={parameter_count}, expected={EXPECTED_PARAMETER_COUNT}"
        )

    checkpoint = torch.load(
        paths["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise RuntimeError("checkpoint payload is not a dictionary")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("checkpoint model_state_dict is absent")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError("checkpoint state contains non-tensors")
    if len(state) != EXPECTED_STATE_TENSOR_COUNT:
        raise RuntimeError(
            f"checkpoint tensor count={len(state)}, "
            f"expected={EXPECTED_STATE_TENSOR_COUNT}"
        )
    state_numel = sum(int(value.numel()) for value in state.values())
    if state_numel != EXPECTED_STATE_NUMEL:
        raise RuntimeError(
            f"checkpoint state numel={state_numel}, expected={EXPECTED_STATE_NUMEL}"
        )

    required_metadata = {
        "stage": "V5_P2_TASK_D_FULL_MULTITASK_SINGLE_RUN",
        "candidate": "graphconv",
        "seed": SELECTED_SEED,
        "epoch": SELECTED_EPOCH,
        "threshold_tuning_performed": False,
        "test_tensors_deserialized": False,
        "loader_sha256": EXPECTED_HASHES["loader"],
        "b3_model_sha256": EXPECTED_HASHES["b3_model"],
        "task_d_model_sha256": EXPECTED_HASHES["task_d_model"],
    }
    for key, expected in required_metadata.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(
                f"checkpoint metadata mismatch: {key}={checkpoint.get(key)!r}, "
                f"expected={expected!r}"
            )

    checkpoint_edge = state.get("base_edge_index")
    if not isinstance(checkpoint_edge, torch.Tensor):
        raise RuntimeError("checkpoint base_edge_index is absent")
    if not torch.equal(checkpoint_edge.detach().cpu().long(), edge):
        raise RuntimeError("checkpoint and frozen topology edge indices differ")

    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint load returned incompatible keys")
    model.eval()

    return model, {
        "architecture": (
            "Causal Depthwise-Separable Conv1D Temporal Encoder "
            "+ Two-Layer GraphConv"
        ),
        "model_class": "P2TaskDGraphConvCount4",
        "selected_seed": SELECTED_SEED,
        "selected_epoch": SELECTED_EPOCH,
        "parameter_count": parameter_count,
        "state_tensor_count": len(state),
        "state_numel": state_numel,
        "checkpoint_top_level_keys": sorted(str(key) for key in checkpoint),
    }


def manifest_row_mapping(path: Path) -> tuple[dict[str, int], int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping: dict[str, int] = {}
    count = 0
    for row_index, row in enumerate(rows):
        if row.get("split") != "validation":
            continue
        pair_key = row.get("pair_key", "")
        if not pair_key:
            raise RuntimeError("validation manifest row has empty pair_key")
        if pair_key in mapping:
            raise RuntimeError(f"duplicate validation pair_key: {pair_key}")
        mapping[pair_key] = row_index
        count += 1
    if count != 74:
        raise RuntimeError(f"validation manifest rows={count}, expected=74")
    return mapping, count


def expected_batches_from_sampler(sampler: Any) -> list[list[int]]:
    batches = [list(batch) for batch in sampler]
    if len(batches) != EXPECTED_BATCHES:
        raise RuntimeError(
            f"validation batch count={len(batches)}, expected={EXPECTED_BATCHES}"
        )
    flattened = [index for batch in batches for index in batch]
    if flattened != list(range(EXPECTED_ITEMS)):
        raise RuntimeError(
            "validation sampler order is not exact dataset order 0..12527"
        )
    if any(len(batch) > ITEM_BATCH_SIZE for batch in batches):
        raise RuntimeError("validation batch exceeds 256 items")
    return batches


def batch_chunk_name(batch_index: int, indices: list[int]) -> str:
    return (
        f"batch_{batch_index:03d}_"
        f"{indices[0]:05d}_{indices[-1] + 1:05d}.npz"
    )


def validate_chunk(
    path: Path,
    *,
    indices: list[int],
    expected_manifest_rows: np.ndarray,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as value:
        required = set(EXPECTED_E1_ARRAYS)
        missing = required - set(value.files)
        if missing:
            raise RuntimeError(f"{path}: missing arrays {sorted(missing)}")
        rows = len(indices)
        expected_local_shapes = {
            "graph_logits": (rows,),
            "count_logits": (rows, 4),
            "source_logits": (rows, 16),
            "transit_logits": (rows, 16),
            "victim_logits": (rows, 16),
            "path_logits": (rows, 16),
            "y_graph": (rows,),
            "y_attacker_count": (rows,),
            "y_source": (rows, 16),
            "y_transit": (rows, 16),
            "y_victim": (rows, 16),
            "y_path": (rows, 16),
            "role_mask": (rows, 16),
            "validation_item_index": (rows,),
            "manifest_row_index": (rows,),
        }
        for name, shape in expected_local_shapes.items():
            array = np.asarray(value[name])
            if array.shape != shape:
                raise RuntimeError(
                    f"{path}: {name} shape={array.shape}, expected={shape}"
                )
            expected_dtype = EXPECTED_E1_ARRAYS[name][1]
            if array.dtype != expected_dtype:
                raise RuntimeError(
                    f"{path}: {name} dtype={array.dtype}, expected={expected_dtype}"
                )
            if not np.all(np.isfinite(array)):
                raise RuntimeError(f"{path}: {name} contains non-finite values")

        expected_indices = np.asarray(indices, dtype=np.int64)
        if not np.array_equal(value["validation_item_index"], expected_indices):
            raise RuntimeError(f"{path}: validation item indices changed")
        if not np.array_equal(
            value["manifest_row_index"],
            expected_manifest_rows,
        ):
            raise RuntimeError(f"{path}: manifest row indices changed")

        return {
            "rows": rows,
            "first_index": int(indices[0]),
            "last_index": int(indices[-1]),
            "graph_logit_min": float(np.min(value["graph_logits"])),
            "graph_logit_max": float(np.max(value["graph_logits"])),
        }


def create_batch_arrays(
    *,
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    indices: list[int],
    manifest_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    if set(batch) != REQUIRED_ITEM_KEYS:
        raise RuntimeError(
            f"validation batch keys changed: {sorted(batch)}"
        )
    if set(outputs) != REQUIRED_OUTPUT_KEYS:
        raise RuntimeError(
            f"model output keys changed: {sorted(outputs)}"
        )

    result: dict[str, np.ndarray] = {}
    for archive_name, model_name in LOGIT_OUTPUT_KEY.items():
        result[archive_name] = (
            outputs[model_name]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    for archive_name, batch_name in LABEL_BATCH_KEY.items():
        dtype = (
            np.int64
            if archive_name == "y_attacker_count"
            else np.uint8
        )
        result[archive_name] = (
            batch[batch_name].detach().cpu().numpy().astype(dtype, copy=False)
        )
    result["validation_item_index"] = np.asarray(indices, dtype=np.int64)
    result["manifest_row_index"] = np.asarray(
        manifest_rows,
        dtype=np.int64,
    )
    return result


def compare_arrays_exact(
    regenerated: dict[str, np.ndarray],
    archive_dir: Path,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    all_equal = True
    for name, (shape, dtype) in EXPECTED_E1_ARRAYS.items():
        frozen = np.load(
            archive_dir / f"{name}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        value = np.asarray(regenerated[name])
        if value.shape != shape or value.dtype != dtype:
            raise RuntimeError(
                f"regenerated {name} schema changed: "
                f"shape={value.shape}, dtype={value.dtype}"
            )
        if frozen.shape != shape or frozen.dtype != dtype:
            raise RuntimeError(
                f"frozen {name} schema changed: "
                f"shape={frozen.shape}, dtype={frozen.dtype}"
            )

        equal = bool(np.array_equal(value, frozen))
        all_equal = all_equal and equal
        mismatch_count = int(np.count_nonzero(value != frozen))
        if np.issubdtype(dtype, np.floating):
            absolute = np.abs(
                value.astype(np.float64) - frozen.astype(np.float64)
            )
            maximum_absolute_difference = float(np.max(absolute))
        else:
            maximum_absolute_difference = 0.0 if equal else float("nan")

        comparison[name] = {
            "bitwise_equal": equal,
            "mismatch_count": mismatch_count,
            "maximum_absolute_difference": maximum_absolute_difference,
            "regenerated_payload_sha256": sha256_array_payload(value),
            "frozen_payload_sha256": sha256_array_payload(frozen),
        }
    if not all_equal:
        failures = [
            name
            for name, record in comparison.items()
            if not record["bitwise_equal"]
        ]
        raise RuntimeError(
            "regenerated E1 arrays are not bitwise identical: "
            + ", ".join(failures)
        )
    return comparison


def compare_output_npz(
    regenerated: dict[str, np.ndarray],
    path: Path,
    label: str,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as frozen:
        if set(regenerated) != set(frozen.files):
            raise RuntimeError(
                f"{label} output keys differ: "
                f"regenerated={sorted(regenerated)}, frozen={sorted(frozen.files)}"
            )
        records: dict[str, Any] = {}
        for key in sorted(regenerated):
            value = np.asarray(regenerated[key])
            reference = np.asarray(frozen[key])
            equal = bool(np.array_equal(value, reference))
            records[key] = {
                "bitwise_equal": equal,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "regenerated_payload_sha256": sha256_array_payload(value),
                "frozen_payload_sha256": sha256_array_payload(reference),
            }
            if not equal:
                raise RuntimeError(f"{label} output mismatch: {key}")
    return records


def compare_metrics_exact(
    regenerated: dict[str, Any],
    path: Path,
    label: str,
) -> str:
    frozen = read_json(path)
    regenerated_digest = sha256_json(regenerated)
    frozen_digest = sha256_json(frozen)
    if regenerated_digest != frozen_digest:
        raise RuntimeError(
            f"{label} metric mismatch: regenerated={regenerated_digest}, "
            f"frozen={frozen_digest}"
        )
    return regenerated_digest


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def tree_digest(root: Path) -> str:
    return sha256_json(tree_hashes(root))


def validate_stable_items(
    *,
    archive_dir: Path,
    dataset: Any,
    manifest_rows: np.ndarray,
) -> dict[str, Any]:
    path = archive_dir / "stable_items.jsonl"
    require_hash(path, EXPECTED_STABLE_ITEMS_SHA, "E1 stable-items archive")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    if len(records) != EXPECTED_ITEMS:
        raise RuntimeError(
            f"stable item count={len(records)}, expected={EXPECTED_ITEMS}"
        )

    stable_ids = [
        record.get("stable_item_id")
        for record in records
    ]
    if any(not isinstance(value, str) or not value for value in stable_ids):
        raise RuntimeError("stable item ID missing or invalid")
    if len(set(stable_ids)) != EXPECTED_ITEMS:
        raise RuntimeError("stable item IDs are not unique")

    checked_fields: dict[str, bool] = {}
    candidate_fields = {
        "validation_item_index": lambda i, entry: i,
        "manifest_row_index": lambda i, entry: int(manifest_rows[i]),
        "split": lambda i, entry: "validation",
        "pair_key": lambda i, entry: entry.pair_key,
        "mode": lambda i, entry: entry.mode,
        "start": lambda i, entry: int(entry.start),
        "target": lambda i, entry: int(entry.target),
        "common_length": lambda i, entry: int(entry.common_length),
        "file_name": lambda i, entry: entry.file_path.name,
    }
    for field, expected_function in candidate_fields.items():
        present = all(field in record for record in records)
        checked_fields[field] = present
        if not present:
            continue
        for index, (record, entry) in enumerate(
            zip(records, dataset._index)
        ):
            expected = expected_function(index, entry)
            if record[field] != expected:
                raise RuntimeError(
                    f"stable item field mismatch: index={index}, "
                    f"field={field}, observed={record[field]!r}, "
                    f"expected={expected!r}"
                )
    return {
        "record_count": len(records),
        "stable_item_ids_unique": True,
        "archive_sha256": sha256_file(path),
        "fields_present_and_verified": checked_fields,
        "stable_id_sequence_sha256": hashlib.sha256(
            ("\n".join(stable_ids) + "\n").encode("utf-8")
        ).hexdigest(),
    }


def d3_chunk_sequence_digest(chunks_dir: Path) -> tuple[str, int]:
    paths = sorted(chunks_dir.glob("chunk_*.npz"))
    if len(paths) != 126:
        raise RuntimeError(f"D3 chunk count={len(paths)}, expected=126")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    root = args.repo_root.expanduser().resolve()
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else root / "data/processed/v5/p2_complete558_dynamic_graph"
    )

    paths = repository_paths(root)
    observed_hashes = {
        label: require_hash(paths[label], expected, label)
        for label, expected in EXPECTED_HASHES.items()
    }

    e1_manifest = read_json(paths["e1_manifest"])
    d1_contract = read_json(paths["d1_contract"])
    d2_artifact = read_json(paths["d2_artifact"])
    d3_artifact = read_json(paths["d3_artifact"])
    d4_artifact = read_json(paths["d4_artifact"])

    if e1_manifest.get("status") != "COMPLETE":
        raise RuntimeError("E1 archive is not complete")
    if e1_manifest.get("split") != "validation":
        raise RuntimeError("E1 archive split changed")
    if int(e1_manifest.get("item_count", -1)) != EXPECTED_ITEMS:
        raise RuntimeError("E1 item count changed")
    if int(e1_manifest.get("batch_count", -1)) != EXPECTED_BATCHES:
        raise RuntimeError("E1 batch count changed")
    if e1_manifest.get("inference", {}).get("device") != "cpu":
        raise RuntimeError("E1 was not frozen on CPU")
    if e1_manifest.get("inference", {}).get(
        "torch_deterministic_algorithms"
    ) is not True:
        raise RuntimeError("E1 deterministic-algorithms flag changed")
    if e1_manifest.get("selected_checkpoint") != {
        "epoch": SELECTED_EPOCH,
        "seed": SELECTED_SEED,
        "sha256": EXPECTED_HASHES["checkpoint"],
    }:
        raise RuntimeError("E1 selected checkpoint changed")

    expected_e1_sources = {
        "loader_sha256": EXPECTED_HASHES["loader"],
        "b3_model_sha256": EXPECTED_HASHES["b3_model"],
        "task_d_model_sha256": EXPECTED_HASHES["task_d_model"],
        "b2_training_source_sha256": EXPECTED_HASHES["b2_training_source"],
        "checkpoint_sha256": EXPECTED_HASHES["checkpoint"],
        "edge_index_sha256": EXPECTED_HASHES["edge_index"],
        "pair_manifest_sha256": EXPECTED_HASHES["pair_manifest"],
    }
    if e1_manifest.get("source_hashes") != expected_e1_sources:
        raise RuntimeError("E1 source-hash contract changed")

    if d1_contract.get("policy_id") != EXPECTED_POLICY_ID:
        raise RuntimeError("D1 policy ID changed")
    if d2_artifact.get("policy_id") != EXPECTED_POLICY_ID:
        raise RuntimeError("D2 policy ID changed")
    if d3_artifact.get("policy_id") != EXPECTED_POLICY_ID:
        raise RuntimeError("D3 policy ID changed")
    if d4_artifact.get("status") != "COMPLETE":
        raise RuntimeError("D4 is not complete")
    if d4_artifact.get("scientific_status", {}).get(
        "Raw_A0_persist_before_A1"
    ) is not True:
        raise RuntimeError("D4 persistence contract changed")
    if d4_artifact.get("failure_isolation_proof", {}).get(
        "status"
    ) != "PASS":
        raise RuntimeError("D4 failure-isolation proof changed")

    if not data_root.is_dir():
        raise FileNotFoundError(f"validation dataset root missing: {data_root}")

    archive_dir = paths["e1_manifest"].parent
    archive_records = e1_manifest.get("artifacts")
    if not isinstance(archive_records, dict):
        raise RuntimeError("E1 archive artifact map is missing")
    for file_name, record in archive_records.items():
        canonical = archive_dir / file_name
        if not canonical.is_file():
            raise FileNotFoundError(canonical)
        expected = record.get("sha256")
        if not isinstance(expected, str):
            raise RuntimeError(f"E1 SHA-256 missing for {file_name}")
        require_hash(canonical, expected, f"E1 archive {file_name}")

    if archive_records["physical_port_mask_static.npy"]["sha256"] != (
        EXPECTED_STATIC_MASK_SHA
    ):
        raise RuntimeError("E1 static-mask hash changed")
    if archive_records["stable_items.jsonl"]["sha256"] != (
        EXPECTED_STABLE_ITEMS_SHA
    ):
        raise RuntimeError("E1 stable-items hash changed")

    d4_output_root = (
        root / "artifacts/v5/p2_evaluator_d4_staged_raw_a0_a1_validation"
    )
    for stage, expected_digest in d4_artifact["stage_tree_sha256"].items():
        stage_dir = d4_output_root / stage
        observed = tree_digest(stage_dir)
        if observed != expected_digest:
            raise RuntimeError(
                f"D4 stage-tree hash changed: {stage}, "
                f"observed={observed}, expected={expected_digest}"
            )

    d3_chunks_dir = (
        root
        / "artifacts/v5/p2_decoder_d3_full_validation_certification_proof"
        / "certified_outputs"
    )
    observed_d3_sequence, d3_chunk_count = d3_chunk_sequence_digest(
        d3_chunks_dir
    )
    expected_d3_sequence = d3_artifact[
        "immutable_output_archive"
    ]["chunk_hash_sequence_sha256"]
    if observed_d3_sequence != expected_d3_sequence:
        raise RuntimeError("D3 certified-output chunk sequence changed")

    torch.manual_seed(0)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    loader_module = import_source(
        paths["loader"],
        "v5_p2_d5_validation_loader",
    )
    b2_module = import_source(
        paths["b2_training_source"],
        "v5_p2_d5_b2_sampler",
    )
    d4_module = import_source(
        paths["d4_runner"],
        "v5_p2_d5_d4_functions",
    )

    DatasetClass = loader_module.V5P2PairAlignedPrimary58Dataset
    dataset = DatasetClass(
        root=data_root,
        split="validation",
        pair_manifest=paths["pair_manifest"],
    )
    if len(dataset) != EXPECTED_ITEMS:
        raise RuntimeError(
            f"validation dataset length={len(dataset)}, expected={EXPECTED_ITEMS}"
        )
    if dataset.pair_count != 74:
        raise RuntimeError(
            f"validation pair count={dataset.pair_count}, expected=74"
        )

    frozen_mask = load_frozen_physical_mask(paths["physical_mask"])
    if not torch.equal(dataset.physical_port_mask, frozen_mask):
        raise RuntimeError(
            "loader-derived and frozen physical-port masks differ"
        )
    e1_static_mask = np.load(
        archive_dir / "physical_port_mask_static.npy",
        allow_pickle=False,
    )
    if not np.array_equal(
        dataset.physical_port_mask.numpy().astype(np.uint8),
        e1_static_mask,
    ):
        raise RuntimeError("loader mask and E1 static mask differ")

    pair_to_manifest_row, validation_manifest_rows = manifest_row_mapping(
        paths["pair_manifest"]
    )
    all_manifest_rows = np.asarray(
        [
            pair_to_manifest_row[entry.pair_key]
            for entry in dataset._index
        ],
        dtype=np.int64,
    )

    # Fail before inference if the exact validation item/manifest ordering no
    # longer matches the immutable E1 archive.
    frozen_validation_item_index = np.load(
        archive_dir / "validation_item_index.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    frozen_manifest_row_index = np.load(
        archive_dir / "manifest_row_index.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if not np.array_equal(
        frozen_validation_item_index,
        np.arange(EXPECTED_ITEMS, dtype=np.int64),
    ):
        raise RuntimeError("frozen validation-item order changed")
    if not np.array_equal(frozen_manifest_row_index, all_manifest_rows):
        raise RuntimeError(
            "loader-derived manifest-row order differs from immutable E1"
        )

    sampler = b2_module.PairBlockBatchSampler(
        dataset,
        block_batch_size=PAIR_BLOCK_BATCH_SIZE,
        shuffle=False,
        seed=SELECTED_SEED,
    )
    expected_batches = expected_batches_from_sampler(sampler)

    # Recreate sampler because the exact DataLoader owns its iterator.
    sampler_for_loader = b2_module.PairBlockBatchSampler(
        dataset,
        block_batch_size=PAIR_BLOCK_BATCH_SIZE,
        shuffle=False,
        seed=SELECTED_SEED,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler_for_loader,
        num_workers=0,
        pin_memory=False,
    )
    if len(loader) != EXPECTED_BATCHES:
        raise RuntimeError(
            f"DataLoader batch count={len(loader)}, expected={EXPECTED_BATCHES}"
        )

    device = torch.device("cpu")
    model, model_summary = build_model(paths=paths, device=device)

    working_dir = (
        root
        / "artifacts/v5/p2_evaluator_d5_finalized_architecture_working"
    )
    chunks_dir = working_dir / "validation_inference_chunks"
    state_path = working_dir / "STATE.json"
    final_artifact_dir = (
        root
        / "artifacts/v5/p2_evaluator_d5_finalized_architecture_integration"
    )
    final_report_dir = (
        root
        / "reports/v5/p2_evaluator_d5_finalized_architecture_integration"
    )

    if final_artifact_dir.exists() or final_report_dir.exists():
        raise RuntimeError(
            "final D5 output already exists; do not delete or overwrite it"
        )

    run_contract = {
        "stage": STAGE,
        "data_root": str(data_root),
        "item_count": EXPECTED_ITEMS,
        "batch_count": EXPECTED_BATCHES,
        "pair_block_batch_size": PAIR_BLOCK_BATCH_SIZE,
        "selected_seed": SELECTED_SEED,
        "selected_epoch": SELECTED_EPOCH,
        "source_hashes": observed_hashes,
        "D3_chunk_sequence_sha256": observed_d3_sequence,
        "D4_stage_tree_sha256": d4_artifact["stage_tree_sha256"],
    }
    contract_sha = sha256_json(run_contract)

    if state_path.is_file():
        state = read_json(state_path)
        if state.get("contract_sha256") != contract_sha:
            raise RuntimeError(
                "existing D5 working state belongs to another contract"
            )
    else:
        working_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "stage": STAGE,
            "status": "IN_PROGRESS",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "contract": run_contract,
            "contract_sha256": contract_sha,
            "completed_batches": [],
        }
        atomic_write_json(state_path, state)

    completed = {
        item["filename"]: item
        for item in state.get("completed_batches", [])
    }

    model.eval()
    emitted = 0
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader):
            indices = expected_batches[batch_number]
            if indices[0] != emitted:
                raise RuntimeError("validation emitted order changed")
            emitted += len(indices)
            manifest_rows = all_manifest_rows[np.asarray(indices)]
            filename = batch_chunk_name(batch_number, indices)
            chunk_path = chunks_dir / filename

            if filename in completed:
                record = completed[filename]
                if not chunk_path.is_file():
                    raise RuntimeError(
                        f"registered D5 chunk is missing: {chunk_path}"
                    )
                observed = sha256_file(chunk_path)
                if observed != record["sha256"]:
                    raise RuntimeError(
                        f"registered D5 chunk hash changed: {chunk_path}"
                    )
                validate_chunk(
                    chunk_path,
                    indices=indices,
                    expected_manifest_rows=manifest_rows,
                )
                print(
                    f"D5_inference_batch={batch_number + 1}/{EXPECTED_BATCHES} "
                    f"emitted={emitted}/{EXPECTED_ITEMS} status=RESUMED",
                    flush=True,
                )
                continue

            if chunk_path.exists():
                # Atomic-save recovery: a crash may occur after the chunk was
                # replaced but before STATE.json was updated.
                summary = validate_chunk(
                    chunk_path,
                    indices=indices,
                    expected_manifest_rows=manifest_rows,
                )
                record = {
                    "batch_number": batch_number,
                    "filename": filename,
                    "first_index": indices[0],
                    "stop_index": indices[-1] + 1,
                    "rows": len(indices),
                    "sha256": sha256_file(chunk_path),
                    "summary": summary,
                    "reconciled_unregistered_chunk": True,
                }
                state["completed_batches"].append(record)
                state["updated_utc"] = datetime.now(timezone.utc).isoformat()
                atomic_write_json(state_path, state)
                completed[filename] = record
                print(
                    f"D5_inference_batch={batch_number + 1}/{EXPECTED_BATCHES} "
                    f"emitted={emitted}/{EXPECTED_ITEMS} "
                    f"status=RECONCILED",
                    flush=True,
                )
                continue

            x = batch["x"].to(device=device, dtype=torch.float32)
            mask = batch["physical_port_mask"].to(device=device)
            outputs = model(x, mask)
            arrays = create_batch_arrays(
                batch=batch,
                outputs=outputs,
                indices=indices,
                manifest_rows=manifest_rows,
            )
            atomic_savez(chunk_path, **arrays)
            summary = validate_chunk(
                chunk_path,
                indices=indices,
                expected_manifest_rows=manifest_rows,
            )
            record = {
                "batch_number": batch_number,
                "filename": filename,
                "first_index": indices[0],
                "stop_index": indices[-1] + 1,
                "rows": len(indices),
                "sha256": sha256_file(chunk_path),
                "summary": summary,
                "reconciled_unregistered_chunk": False,
            }
            state["completed_batches"].append(record)
            state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(state_path, state)
            completed[filename] = record

            print(
                f"D5_inference_batch={batch_number + 1}/{EXPECTED_BATCHES} "
                f"emitted={emitted}/{EXPECTED_ITEMS} status=COMMITTED",
                flush=True,
            )

    if emitted != EXPECTED_ITEMS:
        raise RuntimeError(
            f"D5 emitted items={emitted}, expected={EXPECTED_ITEMS}"
        )

    state = read_json(state_path)
    records = sorted(
        state["completed_batches"],
        key=lambda item: int(item["batch_number"]),
    )
    if len(records) != EXPECTED_BATCHES:
        raise RuntimeError(
            f"D5 committed batch count={len(records)}, expected={EXPECTED_BATCHES}"
        )

    parts: dict[str, list[np.ndarray]] = {
        name: [] for name in EXPECTED_E1_ARRAYS
    }
    chunk_hash_sequence = hashlib.sha256()
    for batch_number, indices in enumerate(expected_batches):
        filename = batch_chunk_name(batch_number, indices)
        record = records[batch_number]
        if record["filename"] != filename:
            raise RuntimeError("D5 batch records are not contiguous")
        path = chunks_dir / filename
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"D5 chunk hash changed: {path}")
        validate_chunk(
            path,
            indices=indices,
            expected_manifest_rows=all_manifest_rows[np.asarray(indices)],
        )
        with np.load(path, allow_pickle=False) as value:
            for name in parts:
                parts[name].append(np.asarray(value[name]).copy())
        chunk_hash_sequence.update(record["sha256"].encode("ascii"))
        chunk_hash_sequence.update(b"\n")

    regenerated = {
        name: np.concatenate(value, axis=0)
        for name, value in parts.items()
    }

    # Repeat the exact first validation batch and compare both repeated forwards
    # and the committed first chunk.
    first_indices = expected_batches[0]
    first_loader = DataLoader(
        dataset,
        batch_sampler=[first_indices],
        num_workers=0,
        pin_memory=False,
    )
    first_batch = next(iter(first_loader))
    with torch.inference_mode():
        first_output = model(
            first_batch["x"].to(device=device, dtype=torch.float32),
            first_batch["physical_port_mask"].to(device=device),
        )
        second_output = model(
            first_batch["x"].to(device=device, dtype=torch.float32),
            first_batch["physical_port_mask"].to(device=device),
        )
    for key in REQUIRED_OUTPUT_KEYS:
        if not torch.equal(first_output[key], second_output[key]):
            raise RuntimeError(f"first-batch repeat failed: {key}")
        archive_name = next(
            name
            for name, output_key in LOGIT_OUTPUT_KEY.items()
            if output_key == key
        )
        first_value = (
            first_output[key].cpu().numpy().astype(np.float32, copy=False)
        )
        if not np.array_equal(
            first_value,
            regenerated[archive_name][: len(first_indices)],
        ):
            raise RuntimeError(
                f"first-batch replay differs from committed chunk: {key}"
            )

    e1_comparison = compare_arrays_exact(regenerated, archive_dir)
    stable_item_proof = validate_stable_items(
        archive_dir=archive_dir,
        dataset=dataset,
        manifest_rows=all_manifest_rows,
    )

    labels = {
        "graph": regenerated["y_graph"],
        "count": regenerated["y_attacker_count"],
        "source": regenerated["y_source"],
        "transit": regenerated["y_transit"],
        "victim": regenerated["y_victim"],
        "path": regenerated["y_path"],
    }
    raw_outputs, raw_metrics, a0_outputs, a0_metrics = d4_module.raw_and_a0(
        regenerated,
        labels,
    )

    d4_raw_dir = d4_output_root / "E2_RAW_NEURAL"
    d4_a0_dir = d4_output_root / "E3_A0_LIGHTWEIGHT"
    d4_a1_dir = d4_output_root / "E4_A1_CERTIFIED_STRUCTURED"

    raw_output_comparison = compare_output_npz(
        raw_outputs,
        d4_raw_dir / "outputs.npz",
        "D4 Raw",
    )
    raw_metric_digest = compare_metrics_exact(
        raw_metrics,
        d4_raw_dir / "metrics.json",
        "D4 Raw",
    )
    a0_output_comparison = compare_output_npz(
        a0_outputs,
        d4_a0_dir / "outputs.npz",
        "D4 A0",
    )
    a0_metric_digest = compare_metrics_exact(
        a0_metrics,
        d4_a0_dir / "metrics.json",
        "D4 A0",
    )

    d3_outputs = d4_module.load_d3_outputs(d3_chunks_dir)
    a1_outputs, a1_metrics = d4_module.a1_from_d3(d3_outputs, labels)
    a1_output_comparison = compare_output_npz(
        a1_outputs,
        d4_a1_dir / "outputs.npz",
        "D4 A1",
    )
    a1_metric_digest = compare_metrics_exact(
        a1_metrics,
        d4_a1_dir / "metrics.json",
        "D4 A1",
    )

    integration_proof = {
        "E1_all_arrays_bitwise_equal": all(
            item["bitwise_equal"] for item in e1_comparison.values()
        ),
        "E1_six_logit_arrays_bitwise_equal": all(
            e1_comparison[name]["bitwise_equal"]
            for name in (
                "graph_logits",
                "count_logits",
                "source_logits",
                "transit_logits",
                "victim_logits",
                "path_logits",
            )
        ),
        "E1_labels_and_order_bitwise_equal": all(
            e1_comparison[name]["bitwise_equal"]
            for name in (
                "y_graph",
                "y_attacker_count",
                "y_source",
                "y_transit",
                "y_victim",
                "y_path",
                "role_mask",
                "validation_item_index",
                "manifest_row_index",
            )
        ),
        "D4_Raw_outputs_bitwise_equal": all(
            value["bitwise_equal"]
            for value in raw_output_comparison.values()
        ),
        "D4_A0_outputs_bitwise_equal": all(
            value["bitwise_equal"]
            for value in a0_output_comparison.values()
        ),
        "D4_A1_outputs_bitwise_equal": all(
            value["bitwise_equal"]
            for value in a1_output_comparison.values()
        ),
        "D4_Raw_metrics_exact": True,
        "D4_A0_metrics_exact": True,
        "D4_A1_metrics_exact": True,
        "first_batch_exact_repeat": True,
        "D3_transitive_binding_to_regenerated_E1": True,
        "reason": (
            "D3 consumed the immutable E1 logits, and D5 proved that all "
            "regenerated finalized-architecture logits are bitwise identical "
            "to that immutable E1 archive."
        ),
    }
    if not all(
        value is True
        for key, value in integration_proof.items()
        if isinstance(value, bool)
    ):
        raise RuntimeError("D5 integration proof is incomplete")

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "finalized_system": model_summary,
        "validation_data": {
            "data_root": str(data_root),
            "items": EXPECTED_ITEMS,
            "batches": EXPECTED_BATCHES,
            "validation_manifest_rows": validation_manifest_rows,
            "pair_count": dataset.pair_count,
            "item_order": (
                "PairBlockBatchSampler(block_batch_size=128, "
                "shuffle=False, seed=107), exact 0..12527"
            ),
        },
        "inference": {
            "device": "cpu",
            "dtype": "float32",
            "amp": False,
            "model_eval": True,
            "torch_inference_mode": True,
            "torch_deterministic_algorithms": True,
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "first_batch_exact_repeat": True,
            "single_complete_validation_inference_pass": True,
            "resume_safe_batch_commits": True,
        },
        "integration_proof": integration_proof,
        "E1_array_comparison": e1_comparison,
        "stable_item_proof": stable_item_proof,
        "D4_equivalence": {
            "Raw": {
                "outputs": raw_output_comparison,
                "metrics_sha256": raw_metric_digest,
            },
            "A0": {
                "outputs": a0_output_comparison,
                "metrics_sha256": a0_metric_digest,
            },
            "A1": {
                "outputs": a1_output_comparison,
                "metrics_sha256": a1_metric_digest,
            },
        },
        "validation_metrics": {
            "Raw_graph_accuracy": raw_metrics["graph"]["accuracy"],
            "A0_graph_accuracy": a0_metrics["graph"]["accuracy"],
            "A1_graph_accuracy": a1_metrics["graph"]["accuracy"],
            "Raw_strict_all_task_exactness": (
                raw_metrics["strict_all_task_exactness"]
            ),
            "A0_strict_all_task_exactness": (
                a0_metrics["strict_all_task_exactness"]
            ),
            "A1_strict_all_task_exactness": (
                a1_metrics["strict_all_task_exactness"]
            ),
        },
        "archive_bindings": {
            "E1_manifest_sha256": observed_hashes["e1_manifest"],
            "E1_stable_items_sha256": EXPECTED_STABLE_ITEMS_SHA,
            "D3_chunk_count": d3_chunk_count,
            "D3_chunk_hash_sequence_sha256": observed_d3_sequence,
            "D4_artifact_sha256": observed_hashes["d4_artifact"],
            "D4_stage_tree_sha256": d4_artifact["stage_tree_sha256"],
            "regenerated_batch_chunk_hash_sequence_sha256": (
                chunk_hash_sequence.hexdigest()
            ),
        },
        "source_hashes": observed_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_geometric": importlib.metadata.version(
                "torch-geometric"
            ),
        },
        "scientific_status": {
            "finalized_architecture_connected_to_E1": True,
            "finalized_checkpoint_connected_to_E1": True,
            "immutable_E1_reproduced_bitwise": True,
            "D4_staged_pipeline_reproduced_exactly": True,
            "D3_certified_A1_transitively_bound": True,
            "validation_model_performance_metrics_computed": True,
            "test_result": False,
            "threshold_selection_performed": False,
            "model_or_checkpoint_selection_performed": False,
            "decoder_selection_performed": False,
            "decoder_execution_performed": False,
            "P2_future_status": "post_hoc_test_informed_only",
        },
        "security_boundary": {
            "validation_tensor_files_opened": True,
            "validation_tensors_deserialized": True,
            "validation_model_inference_performed": True,
            "test_path_formed": False,
            "test_directory_checked": False,
            "test_directory_enumerated": False,
            "test_tensor_files_opened": False,
            "test_tensors_deserialized": False,
            "threshold_selection_performed": False,
            "decoder_execution_performed": False,
            "evaluation_authorization_created": False,
        },
        "next_stage": (
            "V5_P2_D6_POST_HOC_NUMERICAL_RECOVERY_READINESS_AND_DISCLOSURE_FREEZE"
        ),
    }

    # Build every final payload before creating final directories.
    final_artifact_dir.mkdir(parents=True, exist_ok=False)
    final_report_dir.mkdir(parents=True, exist_ok=False)

    chunk_archive_dir = final_artifact_dir / "regenerated_e1_batch_chunks"
    chunk_archive_dir.mkdir()
    for record in records:
        source = chunks_dir / record["filename"]
        destination = chunk_archive_dir / record["filename"]
        os.link(source, destination)

    regenerated_npz = final_artifact_dir / "REGENERATED_E1_ARRAYS.npz"
    atomic_savez(regenerated_npz, **regenerated)

    artifact_json = (
        final_artifact_dir
        / "V5_P2_EVALUATOR_D5_FINALIZED_ARCHITECTURE_E1_INTEGRATION.json"
    )
    atomic_write_json(artifact_json, report)
    artifact_sha = sha256_file(artifact_json)

    report_json = (
        final_report_dir
        / "V5_P2_EVALUATOR_D5_FINALIZED_ARCHITECTURE_E1_INTEGRATION.json"
    )
    report_copy = dict(report)
    report_copy["artifact"] = {
        "path": str(artifact_json),
        "sha256": artifact_sha,
    }
    atomic_write_json(report_json, report_copy)

    report_md = (
        final_report_dir
        / "V5_P2_EVALUATOR_D5_FINALIZED_ARCHITECTURE_E1_INTEGRATION.md"
    )
    atomic_write_text(
        report_md,
        f"""# V5 P2 D5 Finalized-Architecture E1 Integration

## Status

- Status: **COMPLETE**
- Final architecture:
  **Causal Depthwise-Separable Conv1D Temporal Encoder + Two-Layer GraphConv**
- Checkpoint: **seed 107, epoch 25**
- Validation items regenerated: **{EXPECTED_ITEMS}/{EXPECTED_ITEMS}**
- Validation batches: **{EXPECTED_BATCHES}/{EXPECTED_BATCHES}**

## Exact E1 reproduction

- Six neural-logit arrays bitwise identical: **PASS**
- Labels, role mask, item order, and manifest-row order identical: **PASS**
- Physical-port mask identical: **PASS**
- Stable-item sequence verified: **PASS**
- First-batch repeated inference bitwise identical: **PASS**

The actual finalized model, frozen checkpoint, audited PRIMARY58 loader,
physical-port mask, static 4×4 edge index, and pair-block validation order
therefore reproduce the immutable E1 archive exactly.

## Staged-pipeline equivalence

- Raw outputs bitwise identical to D4 E2: **PASS**
- Raw metrics identical to D4 E2: **PASS**
- A0 outputs bitwise identical to D4 E3: **PASS**
- A0 metrics identical to D4 E3: **PASS**
- Certified A1 outputs bitwise identical to D4 E4: **PASS**
- Certified A1 metrics identical to D4 E4: **PASS**

D3 is transitively bound to the actual finalized model because D3 consumed the
immutable E1 logits and D5 reproduced every one of those logits bitwise.

## Validation metrics

| Group | Graph accuracy | Strict all-task exactness |
|---|---:|---:|
| Raw | `{raw_metrics['graph']['accuracy']:.6f}` | `{raw_metrics['strict_all_task_exactness']:.6f}` |
| A0 | `{a0_metrics['graph']['accuracy']:.6f}` | `{a0_metrics['strict_all_task_exactness']:.6f}` |
| Certified A1 | `{a1_metrics['graph']['accuracy']:.6f}` | `{a1_metrics['strict_all_task_exactness']:.6f}` |

These remain validation metrics, not test results.

## Safety boundary

- Validation model inference performed: **true**
- Test path formed: **false**
- Test directory checked/enumerated: **false**
- Test tensor files opened: **false**
- Test tensors deserialized: **false**
- Threshold/model/checkpoint/decoder selection performed: **false**
- Decoder execution performed: **false**
- Evaluation authorization created: **false**

Any future P2 evaluation remains explicitly post hoc and test-informed.

## Next stage

**V5 P2 D6 post-hoc numerical-recovery readiness and disclosure freeze**

Artifact SHA-256: `{artifact_sha}`
""",
    )

    lock = {
        "stage": f"{STAGE}_LOCK",
        "status": "LOCKED",
        "artifact_path": str(artifact_json),
        "artifact_sha256": artifact_sha,
        "report_json_sha256": sha256_file(report_json),
        "report_markdown_sha256": sha256_file(report_md),
        "regenerated_E1_npz_sha256": sha256_file(regenerated_npz),
        "E1_bitwise_reproduction_passed": True,
        "D4_exact_equivalence_passed": True,
        "D3_transitive_binding_passed": True,
        "test_access_performed": False,
        "authorization_created": False,
        "next_stage": report["next_stage"],
    }
    lock_path = (
        final_report_dir
        / "V5_P2_EVALUATOR_D5_FINALIZED_ARCHITECTURE_E1_INTEGRATION_LOCK.json"
    )
    atomic_write_json(lock_path, lock)

    marker = final_report_dir / COMPLETE
    atomic_write_text(marker, COMPLETE + "\n")

    checksum_paths = [
        artifact_json,
        regenerated_npz,
        report_json,
        report_md,
        lock_path,
        marker,
        *sorted(chunk_archive_dir.glob("*.npz")),
    ]
    checksum_path = (
        final_report_dir
        / "V5_P2_EVALUATOR_D5_FINALIZED_ARCHITECTURE_E1_INTEGRATION_SHA256SUMS.txt"
    )
    atomic_write_text(
        checksum_path,
        "".join(
            f"{sha256_file(path)}  {path}\n"
            for path in checksum_paths
        ),
    )

    state["status"] = "COMPLETE"
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    state["final_artifact_sha256"] = artifact_sha
    atomic_write_json(state_path, state)

    print("V5_P2_EVALUATOR_D5_FINALIZED_ARCHITECTURE_E1_INTEGRATION_COMPLETE")
    print("status=COMPLETE")
    print("finalized_architecture_connected_to_E1=true")
    print("finalized_checkpoint_connected_to_E1=true")
    print(f"validation_items_regenerated={EXPECTED_ITEMS}/{EXPECTED_ITEMS}")
    print(f"validation_batches_regenerated={EXPECTED_BATCHES}/{EXPECTED_BATCHES}")
    print("E1_six_logit_arrays_bitwise_equal=true")
    print("E1_labels_and_order_bitwise_equal=true")
    print("physical_port_mask_identical=true")
    print("stable_item_sequence_verified=true")
    print("first_batch_exact_repeat=true")
    print("D4_Raw_outputs_and_metrics_exact=true")
    print("D4_A0_outputs_and_metrics_exact=true")
    print("D4_A1_outputs_and_metrics_exact=true")
    print("D3_transitive_binding_to_finalized_model=true")
    print(
        f"Raw_graph_accuracy={raw_metrics['graph']['accuracy']:.17g}"
    )
    print(
        f"A0_graph_accuracy={a0_metrics['graph']['accuracy']:.17g}"
    )
    print(
        f"A1_graph_accuracy={a1_metrics['graph']['accuracy']:.17g}"
    )
    print(
        f"Raw_strict_all_task_exactness="
        f"{raw_metrics['strict_all_task_exactness']:.17g}"
    )
    print(
        f"A0_strict_all_task_exactness="
        f"{a0_metrics['strict_all_task_exactness']:.17g}"
    )
    print(
        f"A1_strict_all_task_exactness="
        f"{a1_metrics['strict_all_task_exactness']:.17g}"
    )
    print("validation_metrics_only=true")
    print("test_result=false")
    print("test_path_formed=false")
    print("test_directory_checked=false")
    print("test_directory_enumerated=false")
    print("test_tensor_files_opened=false")
    print("test_tensors_deserialized=false")
    print("threshold_selection_performed=false")
    print("decoder_execution_performed=false")
    print("authorization_created=false")
    print(
        "next_stage="
        "V5_P2_D6_POST_HOC_NUMERICAL_RECOVERY_READINESS_AND_DISCLOSURE_FREEZE"
    )
    print(f"artifact={artifact_json}")
    print(f"artifact_sha256={artifact_sha}")
    print(f"report_markdown={report_md}")
    print(f"report_json={report_json}")
    print(f"sha256s={checksum_path}")


if __name__ == "__main__":
    main()
