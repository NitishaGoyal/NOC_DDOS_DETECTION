from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import shutil
import struct
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


STAGE = "V5_P3_H0_CURRENT_4X4_DYNAMIC70_RTL_HANDOFF_PREQUANTIZATION_REFERENCE"
CAMPAIGN_LABEL = "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic"
SEED = 107
EXPECTED_PARAMETER_COUNT = 60_553
EXPECTED_INPUT_SHAPE = (16, 70, 32)
EXPECTED_MASK_SHAPE = (16, 10)
EXPECTED_EDGE_SHAPE = (2, 48)
EXPECTED_OUTPUT_SCALARS = 69
EXPECTED_CHECKPOINT_EPOCH = 14

HEAD_LAYOUT = {
    "graph": {"offset": 0, "length": 1},
    "count": {"offset": 1, "length": 4, "class_order": [1, 2, 3, 4]},
    "source": {"offset": 5, "length": 16, "router_order": list(range(16))},
    "transit": {"offset": 21, "length": 16, "router_order": list(range(16))},
    "victim": {"offset": 37, "length": 16, "router_order": list(range(16))},
    "path": {"offset": 53, "length": 16, "router_order": list(range(16))},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def import_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_float32_memh(path: Path, values: np.ndarray) -> None:
    array = np.asarray(values, dtype="<f4").reshape(-1)
    bits = array.view("<u4")
    path.write_text(
        "\n".join(f"{int(value):08x}" for value in bits) + "\n",
        encoding="utf-8",
    )


def write_uint8_memh(path: Path, values: np.ndarray) -> None:
    array = np.asarray(values, dtype=np.uint8).reshape(-1)
    path.write_text(
        "\n".join(f"{int(value):02x}" for value in array) + "\n",
        encoding="utf-8",
    )


def flatten_outputs(outputs: dict[str, torch.Tensor]) -> np.ndarray:
    graph = outputs["attack_logits"].detach().cpu().numpy().reshape(-1, 1)
    count = outputs["count_logits"].detach().cpu().numpy()
    source = outputs["source_logits"].detach().cpu().numpy()
    transit = outputs["transit_logits"].detach().cpu().numpy()
    victim = outputs["victim_logits"].detach().cpu().numpy()
    path = outputs["path_logits"].detach().cpu().numpy()
    flat = np.concatenate(
        [graph, count, source, transit, victim, path],
        axis=1,
    ).astype(np.float32)
    if flat.shape[1] != EXPECTED_OUTPUT_SCALARS:
        raise RuntimeError(
            f"flattened output width={flat.shape[1]}, "
            f"expected={EXPECTED_OUTPUT_SCALARS}"
        )
    return flat


def select_probe_indices(dataset, d0_report_path: Path) -> list[int]:
    if d0_report_path.is_file():
        report = json.loads(d0_report_path.read_text(encoding="utf-8"))
        indices = report.get("data", {}).get("probe_indices")
        if indices is not None and len(indices) == 8:
            return [int(index) for index in indices]

    inactive = []
    active_by_count: dict[int, int] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        graph = int(torch.as_tensor(sample["y_attack"]).item())
        count = int(torch.as_tensor(sample["y_attacker_count"]).item())
        if graph == 0 and len(inactive) < 4:
            inactive.append(index)
        if graph == 1 and count in (1, 2, 3, 4):
            active_by_count.setdefault(count, index)
        if len(inactive) == 4 and sorted(active_by_count) == [1, 2, 3, 4]:
            return inactive + [active_by_count[value] for value in (1, 2, 3, 4)]
    raise RuntimeError("could not select 4 inactive and K1-K4 validation probes")


def safe_metadata(sample: dict[str, Any], key: str) -> Any:
    if key not in sample:
        return None
    value = sample[key]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def zip_directory(root: Path, zip_path: Path) -> None:
    temporary = zip_path.with_suffix(zip_path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(root.parent))
    os.replace(temporary, zip_path)


def main() -> int:
    args = parse_args()
    set_seed(SEED)

    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    data_root = data_link.resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    package_dir = Path(args.package_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()

    artifact_dir = output_dir / "handoff"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)

    weights_dir = artifact_dir / "weights"
    golden_dir = artifact_dir / "golden"
    topology_dir = artifact_dir / "topology"
    rtl_dir = artifact_dir / "rtl"
    source_dir = artifact_dir / "source_snapshot"
    docs_dir = artifact_dir / "docs"
    for directory in (
        weights_dir,
        golden_dir,
        topology_dir,
        rtl_dir,
        source_dir,
        docs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    wrapper_path = repo / "src/data/v5_p3_tranche_a_guarded_loader.py"
    b3_path = repo / "src/models/v5_p2_b3_conv1d_only_count4.py"
    canonical_path = repo / "src/models/v5_p2_task_d_full_multitask_count4.py"
    dynamic_path = repo / "src/models/v6_p0_dynamic70_task_d_full_multitask_count4.py"
    checkpoint_path = (
        repo
        / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107/"
        "V5_P3_A4_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_SEED107_BEST.pt"
    )
    d0_report_path = (
        repo
        / "reports/v5/p3_d0_frozen_decoder_transfer_preflight/"
        "V5_P3_D0_FROZEN_DECODER_TRANSFER_PREFLIGHT_REPORT.json"
    )
    d1_lock_path = (
        repo
        / "reports/v5/p3_d1_immutable_validation_logit_export/"
        "V5_P3_D1_IMMUTABLE_TRANCHE_A_VALIDATION_LOGIT_EXPORT_LOCK.json"
    )

    required = [
        wrapper_path,
        b3_path,
        canonical_path,
        dynamic_path,
        checkpoint_path,
        d0_report_path,
        d1_lock_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")
    if not data_link.is_symlink():
        raise RuntimeError(f"dataset path is not canonical symlink: {data_link}")

    # Guard against any A-test deserialization.
    original_torch_load = torch.load
    loaded_paths: list[str] = []

    def guarded_torch_load(file, *load_args, **load_kwargs):
        try:
            candidate = Path(os.fspath(file)).expanduser().resolve()
        except TypeError:
            candidate = None
        if candidate is not None:
            text = str(candidate)
            loaded_paths.append(text)
            if "/runs/test/" in text:
                raise PermissionError(
                    f"H0 sealed-test guard blocked {text}"
                )
        return original_torch_load(file, *load_args, **load_kwargs)

    torch.load = guarded_torch_load

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    wrapper_mod = import_source(wrapper_path, "_v5_p3_h0_guarded_loader")
    b3_mod = import_source(b3_path, "_v5_p3_h0_b3")
    canonical_mod = import_source(canonical_path, "_v5_p3_h0_canonical")
    dynamic_mod = import_source(dynamic_path, "_v5_p3_h0_dynamic70")

    GuardedDataset = wrapper_mod.GuardedV5P3TrancheAPreliminaryDataset
    SealedTestAccessError = wrapper_mod.SealedTestAccessError
    try:
        GuardedDataset(data_root, "test")
    except SealedTestAccessError:
        test_negative_check = True
    else:
        test_negative_check = False
    if not test_negative_check:
        raise RuntimeError("guarded loader failed to reject A-test")

    validation_dataset = GuardedDataset(
        data_root,
        "validation",
        active_only=False,
    )
    probe_indices = select_probe_indices(validation_dataset, d0_report_path)
    samples = [validation_dataset[index] for index in probe_indices]

    edge_index = torch.as_tensor(
        samples[0]["edge_index"],
        dtype=torch.long,
    )
    if tuple(edge_index.shape) != EXPECTED_EDGE_SHAPE:
        raise RuntimeError(
            f"edge_index shape={tuple(edge_index.shape)}, "
            f"expected={EXPECTED_EDGE_SHAPE}"
        )

    reference = b3_mod.P2B3Conv1DOnlyCount4()
    canonical_model = canonical_mod.P2TaskDGraphConvCount4(
        reference,
        edge_index,
    )
    model = dynamic_mod.build_v6_p0_dynamic70_from_canonical_structure(
        canonical_model,
        seed=SEED,
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"parameter_count={parameter_count}, "
            f"expected={EXPECTED_PARAMETER_COUNT}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint.get("epoch", -1)) != EXPECTED_CHECKPOINT_EPOCH:
        raise RuntimeError(
            f"checkpoint epoch={checkpoint.get('epoch')}, "
            f"expected={EXPECTED_CHECKPOINT_EPOCH}"
        )
    if int(checkpoint.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("checkpoint parameter-count contract changed")
    if checkpoint.get("test_tensor_loaded") is not False:
        raise RuntimeError("checkpoint reports test access")

    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = torch.stack(
        [torch.as_tensor(sample["x"], dtype=torch.float32) for sample in samples]
    )
    mask = torch.stack(
        [
            torch.as_tensor(
                sample["physical_port_mask"],
                dtype=torch.float32,
            )
            for sample in samples
        ]
    )
    if tuple(x.shape[1:]) != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(
            f"input shape={tuple(x.shape[1:])}, "
            f"expected={EXPECTED_INPUT_SHAPE}"
        )
    if tuple(mask.shape[1:]) != EXPECTED_MASK_SHAPE:
        raise RuntimeError(
            f"mask shape={tuple(mask.shape[1:])}, "
            f"expected={EXPECTED_MASK_SHAPE}"
        )

    started = time.time()
    with torch.no_grad():
        outputs = model(x.to(device), mask.to(device))
    flat_outputs = flatten_outputs(outputs)

    expected_output_shapes = {
        "attack_logits": (8,),
        "count_logits": (8, 4),
        "source_logits": (8, 16),
        "transit_logits": (8, 16),
        "victim_logits": (8, 16),
        "path_logits": (8, 16),
    }
    actual_output_shapes = {
        key: tuple(value.shape) for key, value in outputs.items()
    }
    if actual_output_shapes != expected_output_shapes:
        raise RuntimeError(
            f"output shapes={actual_output_shapes}, "
            f"expected={expected_output_shapes}"
        )

    # ------------------------------------------------------------------
    # Weight export
    # ------------------------------------------------------------------
    # Export parameters separately from registered buffers. The model carries
    # its fixed graph edge_index as a [2,48] buffer (96 elements); that buffer
    # belongs to the topology contract, not the 60,553-parameter weight stream.
    named_parameters = list(model.named_parameters())
    named_buffers = list(model.named_buffers())

    weight_rows = []
    flattened_weights = []
    npz_tensors = {}
    offset = 0

    for ordinal, (name, tensor) in enumerate(named_parameters):
        array = tensor.detach().cpu().numpy().astype("<f4", copy=False)
        flat = array.reshape(-1)
        sanitized = name.replace(".", "__")
        npz_tensors[sanitized] = array
        np.save(weights_dir / f"{ordinal:03d}_{sanitized}.npy", array)
        flattened_weights.append(flat)
        weight_rows.append(
            {
                "ordinal": ordinal,
                "parameter_name": name,
                "npz_key": sanitized,
                "shape": list(array.shape),
                "elements": int(array.size),
                "float32_offset": offset,
                "float32_end_exclusive": offset + int(array.size),
                "bytes": int(array.nbytes),
                "requires_grad": bool(tensor.requires_grad),
            }
        )
        offset += int(array.size)

    flat_weights = np.concatenate(flattened_weights).astype("<f4")
    if flat_weights.size != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"flattened parameter size={flat_weights.size}, "
            f"expected={EXPECTED_PARAMETER_COUNT}"
        )

    np.savez_compressed(
        weights_dir / "model_parameters_float32.npz",
        **npz_tensors,
    )

    buffer_rows = []
    buffer_npz = {}
    for ordinal, (name, tensor) in enumerate(named_buffers):
        array = tensor.detach().cpu().numpy()
        sanitized = name.replace(".", "__")
        buffer_npz[sanitized] = array
        buffer_rows.append(
            {
                "ordinal": ordinal,
                "buffer_name": name,
                "shape": list(array.shape),
                "elements": int(array.size),
                "dtype": str(array.dtype),
                "bytes": int(array.nbytes),
            }
        )

    np.savez_compressed(
        weights_dir / "model_buffers.npz",
        **buffer_npz,
    )
    atomic_json(
        weights_dir / "buffers_manifest.json",
        {
            "classification": (
                "non-parameter registered buffers; fixed graph topology is "
                "also exported under topology/"
            ),
            "buffer_count": len(buffer_rows),
            "buffer_elements": int(
                sum(row["elements"] for row in buffer_rows)
            ),
            "buffers": buffer_rows,
        },
    )
    flat_weights.tofile(weights_dir / "weights_float32_le.bin")
    write_float32_memh(
        weights_dir / "weights_float32_bits.memh",
        flat_weights,
    )
    atomic_json(
        weights_dir / "weights_manifest.json",
        {
            "ordering": "PyTorch named_parameters iteration order",
            "dtype": "IEEE-754 float32 little-endian",
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "total_bytes": int(flat_weights.nbytes),
            "tensors": weight_rows,
        },
    )
    with (weights_dir / "weights_manifest.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ordinal",
                "parameter_name",
                "npz_key",
                "shape",
                "elements",
                "float32_offset",
                "float32_end_exclusive",
                "bytes",
                "requires_grad",
            ],
        )
        writer.writeheader()
        for row in weight_rows:
            csv_row = dict(row)
            csv_row["shape"] = "x".join(str(value) for value in row["shape"])
            writer.writerow(csv_row)

    # ------------------------------------------------------------------
    # Topology export
    # ------------------------------------------------------------------
    edge_array = edge_index.cpu().numpy().astype(np.int16)
    np.save(topology_dir / "edge_index_int16.npy", edge_array)
    edge_array.T.astype("<i2").tofile(
        topology_dir / "directed_edges_src_dst_int16_le.bin"
    )
    with (topology_dir / "directed_edges.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["edge_id", "source_router", "destination_router"])
        for edge_id, (source, destination) in enumerate(edge_array.T.tolist()):
            writer.writerow([edge_id, source, destination])

    adjacency = np.zeros((16, 16), dtype=np.uint8)
    for source, destination in edge_array.T.tolist():
        adjacency[source, destination] = 1
    np.save(topology_dir / "adjacency_uint8.npy", adjacency)
    write_uint8_memh(
        topology_dir / "adjacency_uint8.memh",
        adjacency,
    )
    atomic_json(
        topology_dir / "router_mapping.json",
        {
            "mesh": "4x4",
            "router_numbering": "row-major",
            "router_id_formula": "router_id = row * 4 + column",
            "coordinates": [
                {
                    "router_id": router,
                    "row": router // 4,
                    "column": router % 4,
                }
                for router in range(16)
            ],
            "directed_edges": 48,
            "edge_order": "exact dataset/model edge_index order",
        },
    )

    # ------------------------------------------------------------------
    # Golden vector export
    # ------------------------------------------------------------------
    x_np = x.cpu().numpy().astype("<f4")
    mask_u8 = mask.cpu().numpy().astype(np.uint8)
    raw_outputs = {
        key: value.detach().cpu().numpy().astype("<f4")
        for key, value in outputs.items()
    }

    np.savez_compressed(
        golden_dir / "golden_vectors_float32.npz",
        probe_indices=np.asarray(probe_indices, dtype=np.int32),
        x=x_np,
        physical_port_mask=mask_u8,
        output_flat=flat_outputs.astype("<f4"),
        attack_logits=raw_outputs["attack_logits"],
        count_logits=raw_outputs["count_logits"],
        source_logits=raw_outputs["source_logits"],
        transit_logits=raw_outputs["transit_logits"],
        victim_logits=raw_outputs["victim_logits"],
        path_logits=raw_outputs["path_logits"],
        y_attack=np.asarray(
            [
                int(torch.as_tensor(sample["y_attack"]).item())
                for sample in samples
            ],
            dtype=np.uint8,
        ),
        y_attacker_count=np.asarray(
            [
                int(torch.as_tensor(sample["y_attacker_count"]).item())
                for sample in samples
            ],
            dtype=np.int8,
        ),
        y_source=np.stack(
            [
                torch.as_tensor(sample["y_source"]).cpu().numpy()
                for sample in samples
            ]
        ).astype(np.uint8),
        y_transit=np.stack(
            [
                torch.as_tensor(sample["y_transit"]).cpu().numpy()
                for sample in samples
            ]
        ).astype(np.uint8),
        y_victim=np.stack(
            [
                torch.as_tensor(sample["y_victim"]).cpu().numpy()
                for sample in samples
            ]
        ).astype(np.uint8),
        y_path=np.stack(
            [
                torch.as_tensor(sample["y_attack_path"]).cpu().numpy()
                for sample in samples
            ]
        ).astype(np.uint8),
    )

    sample_manifest = []
    for local_index, dataset_index in enumerate(probe_indices):
        sample_dir = golden_dir / f"sample_{local_index:02d}_val_{dataset_index:05d}"
        sample_dir.mkdir()

        input_array = x_np[local_index]
        mask_array = mask_u8[local_index]
        output_array = flat_outputs[local_index].astype("<f4")

        input_array.tofile(sample_dir / "input_x_float32_le.bin")
        mask_array.tofile(sample_dir / "physical_port_mask_uint8.bin")
        output_array.tofile(sample_dir / "expected_output_logits_float32_le.bin")
        write_float32_memh(
            sample_dir / "input_x_float32_bits.memh",
            input_array,
        )
        write_uint8_memh(
            sample_dir / "physical_port_mask_uint8.memh",
            mask_array,
        )
        write_float32_memh(
            sample_dir / "expected_output_logits_float32_bits.memh",
            output_array,
        )

        metadata = {
            "sample_slot": local_index,
            "validation_dataset_index": dataset_index,
            "input_layout": "x[router][feature][time], C-order",
            "input_shape": list(EXPECTED_INPUT_SHAPE),
            "mask_layout": "physical_port_mask[router][mask_feature], C-order",
            "mask_shape": list(EXPECTED_MASK_SHAPE),
            "output_layout": HEAD_LAYOUT,
            "output_scalars": EXPECTED_OUTPUT_SCALARS,
            "labels": {
                "graph": int(torch.as_tensor(samples[local_index]["y_attack"]).item()),
                "attacker_count": int(
                    torch.as_tensor(
                        samples[local_index]["y_attacker_count"]
                    ).item()
                ),
            },
            "provenance": {
                key: safe_metadata(samples[local_index], key)
                for key in (
                    "pair_id",
                    "run_id",
                    "window_start",
                    "epoch_id",
                    "epoch_tick",
                )
            },
        }
        atomic_json(sample_dir / "sample_manifest.json", metadata)
        sample_manifest.append(metadata)

    atomic_json(
        golden_dir / "golden_vector_manifest.json",
        {
            "sample_count": len(probe_indices),
            "selection": "4 inactive plus one active sample for K1-K4",
            "validation_indices": probe_indices,
            "comparison_rule": (
                "Compare raw logits before sigmoid or any decoder. "
                "Float32 software-reference tolerance should be explicitly "
                "declared by the implementation under test."
            ),
            "head_layout": HEAD_LAYOUT,
            "samples": sample_manifest,
        },
    )

    # ------------------------------------------------------------------
    # Source and interface snapshot
    # ------------------------------------------------------------------
    for source_path in (
        b3_path,
        canonical_path,
        dynamic_path,
    ):
        shutil.copy2(source_path, source_dir / source_path.name)

    for static_path in (
        package_dir / "v5_p3_dynamic70_model_if_pkg.sv",
        package_dir / "v5_p3_dynamic70_accel_blackbox.sv",
    ):
        shutil.copy2(static_path, rtl_dir / static_path.name)
    shutil.copy2(package_dir / "REGISTER_MAP.md", docs_dir / "REGISTER_MAP.md")

    architecture_contract = {
        "model_identity": (
            "Causal Depthwise-Separable Conv1D Temporal Encoder "
            "+ Two-Layer GraphConv, Dynamic70 adaptation"
        ),
        "temporal_frontend_reference": "B3",
        "full_neural_architecture": (
            "B3 temporal frontend + two-layer GraphConv"
        ),
        "scope": "single shared 4x4 regional expert",
        "input": {
            "x_shape": ["batch", 16, 70, 32],
            "x_semantics": "x[batch][router][feature][time]",
            "physical_port_mask_shape": ["batch", 16, 10],
            "feature_representation": "Dynamic70",
            "temporal_window": 32,
        },
        "graph": {
            "routers": 16,
            "mesh": "4x4",
            "directed_edges": 48,
            "edge_index_fixed": True,
        },
        "outputs": {
            "graph_attack_logit": ["batch"],
            "attacker_count_logits": ["batch", 4],
            "attacker_count_class_order": [1, 2, 3, 4],
            "source_logits": ["batch", 16],
            "transit_logits": ["batch", 16],
            "victim_logits": ["batch", 16],
            "path_logits": ["batch", 16],
            "flattened_output_scalars": 69,
            "flattened_head_layout": HEAD_LAYOUT,
        },
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "checkpoint": {
            "epoch": EXPECTED_CHECKPOINT_EPOCH,
            "classification": CAMPAIGN_LABEL,
            "final_model_status": (
                "engineering handoff of the current Tranche-A model; "
                "not the future fresh A+B official model"
            ),
        },
        "decoder_boundary": {
            "included": False,
            "reason": (
                "The exact MILP decoder is software-only and is currently "
                "under numerical freeze/regression. H0 freezes the neural "
                "expert interface and raw logits only."
            ),
        },
        "quantization_boundary": {
            "quantized": False,
            "current_numeric_reference": "IEEE-754 float32",
            "synthesis_numeric_format": "not yet frozen",
            "next_required_stage": (
                "calibration-driven per-layer fixed-point/int8-int16 "
                "quantization and accumulator-width audit"
            ),
        },
        "memory_estimates_bytes": {
            "weights_float32": EXPECTED_PARAMETER_COUNT * 4,
            "weights_int16_candidate": EXPECTED_PARAMETER_COUNT * 2,
            "weights_int8_candidate": EXPECTED_PARAMETER_COUNT,
            "single_input_window_float32": 16 * 70 * 32 * 4,
            "single_input_window_int16_candidate": 16 * 70 * 32 * 2,
            "single_input_window_int8_candidate": 16 * 70 * 32,
            "mask_uint8": 16 * 10,
            "output_logits_float32": 69 * 4,
        },
        "8x8_reuse_contract": {
            "supported_as_wrapper_next": True,
            "shared_weights": True,
            "regional_invocations": 4,
            "regional_input_shape": ["batch", 4, 16, 70, 32],
            "regional_mask_shape": ["batch", 4, 16, 10],
            "cross_region_reasoning_included": False,
        },
    }
    atomic_json(
        docs_dir / "architecture_contract.json",
        architecture_contract,
    )

    d1_lock = json.loads(d1_lock_path.read_text(encoding="utf-8"))
    test_loaded_paths = [
        path for path in loaded_paths if "/runs/test/" in path
    ]
    if test_loaded_paths:
        raise RuntimeError(
            f"sealed-test tensors were loaded: {test_loaded_paths}"
        )

    # Artifact-level SHA manifest.
    sha_rows = []
    for path in sorted(artifact_dir.rglob("*")):
        if path.is_file():
            sha_rows.append(
                {
                    "path": str(path.relative_to(artifact_dir)),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    atomic_json(
        artifact_dir / "SHA256_MANIFEST.json",
        {
            "stage": STAGE,
            "files": sha_rows,
        },
    )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "audit_script_revision": (
            "v3_parameter_manifest_csv_schema_fix"
        ),
        "scope": (
            "Pre-quantization RTL handoff for the current 4x4 Dynamic70 "
            "neural regional expert."
        ),
        "artifact": {
            "directory": str(artifact_dir),
            "manifest": str(artifact_dir / "SHA256_MANIFEST.json"),
            "files": len(sha_rows) + 1,
        },
        "model": {
            "checkpoint_epoch": EXPECTED_CHECKPOINT_EPOCH,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "parameter_count": parameter_count,
            "input_shape": list(EXPECTED_INPUT_SHAPE),
            "mask_shape": list(EXPECTED_MASK_SHAPE),
            "output_scalars": EXPECTED_OUTPUT_SCALARS,
            "source_sha256": {
                b3_path.name: sha256_file(b3_path),
                canonical_path.name: sha256_file(canonical_path),
                dynamic_path.name: sha256_file(dynamic_path),
            },
        },
        "weights": {
            "float32_elements": int(flat_weights.size),
            "registered_buffer_count": len(buffer_rows),
            "registered_buffer_elements": int(
                sum(row["elements"] for row in buffer_rows)
            ),
            "float32_bytes": int(flat_weights.nbytes),
            "flat_binary_sha256": sha256_file(
                weights_dir / "weights_float32_le.bin"
            ),
            "flat_memh_sha256": sha256_file(
                weights_dir / "weights_float32_bits.memh"
            ),
            "npz_sha256": sha256_file(
                weights_dir / "model_parameters_float32.npz"
            ),
        },
        "golden": {
            "validation_indices": probe_indices,
            "samples": len(probe_indices),
            "golden_npz_sha256": sha256_file(
                golden_dir / "golden_vectors_float32.npz"
            ),
            "raw_neural_logits_only": True,
        },
        "topology": {
            "edge_index_shape": list(edge_array.shape),
            "directed_edges": int(edge_array.shape[1]),
            "edge_csv_sha256": sha256_file(
                topology_dir / "directed_edges.csv"
            ),
        },
        "boundaries": {
            "decoder_included": False,
            "quantization_frozen": False,
            "RTL_compute_kernel_implemented": False,
            "interface_contract_frozen": True,
            "weights_and_golden_vectors_exported": True,
            "current_model_is_Tranche_A_preliminary": True,
        },
        "sealed_test": {
            "guard_negative_check": test_negative_check,
            "test_dataset_instantiated": False,
            "test_length_computed": False,
            "test_tensor_loaded": False,
            "test_loaded_paths": test_loaded_paths,
            "D1_lock_test_tensor_loaded": d1_lock.get("test_tensor_loaded"),
        },
        "runtime": {
            "device": str(device),
            "elapsed_seconds": time.time() - started,
        },
        "provenance": {
            "installed_script_sha256": sha256_file(installed_script),
            "D0_report_sha256": sha256_file(d0_report_path),
            "D1_lock_sha256": sha256_file(d1_lock_path),
        },
        "next_stage": (
            "V5_P3_H1_DYNAMIC70_QUANTIZATION_AND_ACCUMULATOR_WIDTH_FREEZE"
        ),
    }
    atomic_json(
        output_dir / f"{STAGE}_REPORT.json",
        report,
    )

    archive_path = output_dir / (
        "V5_P3_H0_CURRENT_4X4_DYNAMIC70_RTL_HANDOFF_PREQUANTIZATION_REFERENCE.zip"
    )
    zip_directory(artifact_dir, archive_path)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(
            output_dir / f"{STAGE}_REPORT.json"
        ),
        "handoff_archive_sha256": sha256_file(archive_path),
        "handoff_archive_size_bytes": archive_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_count": parameter_count,
        "input_features": 70,
        "window_length": 32,
        "routers": 16,
        "directed_edges": 48,
        "output_scalars": 69,
        "quantization_frozen": False,
        "decoder_included": False,
        "test_tensor_loaded": False,
    }
    atomic_json(
        output_dir / f"{STAGE}_LOCK.json",
        lock,
    )
    (
        output_dir / f"{STAGE}_COMPLETE"
    ).write_text(f"{STAGE}_COMPLETE\n", encoding="utf-8")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(
        "audit_script_revision="
        "v3_parameter_manifest_csv_schema_fix"
    )
    print(f"campaign_label={CAMPAIGN_LABEL}")
    print(f"checkpoint_epoch={EXPECTED_CHECKPOINT_EPOCH}")
    print(f"parameter_count={parameter_count}")
    print("input_shape=16x70x32")
    print("mask_shape=16x10")
    print("directed_edges=48")
    print("output_scalars=69")
    print(f"golden_samples={len(probe_indices)}")
    print(f"golden_validation_indices={probe_indices}")
    print(f"weights_float32_bytes={flat_weights.nbytes}")
    print(f"registered_buffer_count={len(buffer_rows)}")
    print(
        "registered_buffer_elements="
        f"{sum(row['elements'] for row in buffer_rows)}"
    )
    print("decoder_included=false")
    print("quantization_frozen=false")
    print("RTL_compute_kernel_implemented=false")
    print("interface_contract_frozen=true")
    print("test_dataset_instantiated=false")
    print("test_tensor_loaded=false")
    print(f"handoff_archive={archive_path}")
    print(f"handoff_archive_sha256={sha256_file(archive_path)}")
    print(
        "next_stage="
        "V5_P3_H1_DYNAMIC70_QUANTIZATION_AND_ACCUMULATOR_WIDTH_FREEZE"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
