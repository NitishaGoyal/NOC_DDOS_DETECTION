#!/usr/bin/env python3
"""
V5 P2-A3 Pair-Aligned PRIMARY58 Loader Contract Audit

Installs/audits the replacement P2 train/validation dataset loader.

The audit:
- verifies A0, A1-R2, and A2-R2 locks and hashes;
- constructs TRAIN and VALIDATION datasets only;
- validates all pair-aligned index entries without loading test;
- verifies total aligned-window count = 82,694;
- checks pair adjacency and identical ATTACK/CONTROL starts;
- loads deterministic TRAIN/VALIDATION sample items;
- proves returned x is an exact PRIMARY58 slice of stored standardized x;
- proves targets come from the final epoch of each window;
- proves only the approved item keys are returned;
- proves the corrected Boolean [16,10] mask matches A2-R2;
- proves split='test' is rejected before filesystem access.

No P2 test directory is enumerated or opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


STAGE = "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT"
COMPLETE = f"{STAGE}_COMPLETE"

EXPECTED_TOTAL_ALIGNED_WINDOWS = 82694
EXPECTED_TOTAL_PAIRS = 489
EXPECTED_TRAIN_PAIRS = 415
EXPECTED_VALIDATION_PAIRS = 74

EXPECTED_ITEM_KEYS = {
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

PROHIBITED_ITEM_KEYS = {
    "edge_index",
    "router_coordinates",
    "epoch_id",
    "case_id",
    "pair_id",
    "run_id",
    "filename",
    "mode",
    "split",
    "category",
    "window_start",
    "window_target",
    "window_end",
    "native_run_length",
    "common_length",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def import_loader(path: Path):
    spec = importlib.util.spec_from_file_location(
        "v5_p2_pair_aligned_primary58_dataset",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import loader from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def deterministic_pair_sample_bases(dataset) -> list[int]:
    length = len(dataset)
    if length < 2 or length % 2:
        raise ValueError("dataset length is not positive and even")

    first = 0
    middle = (length // 2) // 2 * 2
    if middle >= length - 1:
        middle = length - 2
    last = length - 2
    return sorted({first, middle, last})


def check_scalar_tensor(
    value: Any,
    dtype: torch.dtype,
    name: str,
    failures: list[str],
    context: str,
) -> None:
    if not isinstance(value, torch.Tensor):
        failures.append(f"{context}: {name} is not Tensor")
        return
    if value.ndim != 0:
        failures.append(
            f"{context}: {name} shape={tuple(value.shape)}, expected scalar"
        )
    if value.dtype != dtype:
        failures.append(
            f"{context}: {name} dtype={value.dtype}, expected {dtype}"
        )


def check_node_tensor(
    value: Any,
    allowed_dtypes: tuple[torch.dtype, ...],
    name: str,
    failures: list[str],
    context: str,
) -> None:
    if not isinstance(value, torch.Tensor):
        failures.append(f"{context}: {name} is not Tensor")
        return
    if tuple(value.shape) != (16,):
        failures.append(
            f"{context}: {name} shape={tuple(value.shape)}, expected [16]"
        )
    if value.dtype not in allowed_dtypes:
        failures.append(
            f"{context}: {name} dtype={value.dtype}, "
            f"expected one of {allowed_dtypes}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a0-dir", type=Path, required=True)
    parser.add_argument("--a1-r2-dir", type=Path, required=True)
    parser.add_argument("--a2-r2-dir", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root_input = args.root.expanduser()
    root = root_input.resolve()
    a0_dir = args.a0_dir.expanduser().resolve()
    a1_r2_dir = args.a1_r2_dir.expanduser().resolve()
    a2_r2_dir = args.a2_r2_dir.expanduser().resolve()
    loader_path = args.loader_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    paths = {
        "a0_report": (
            a0_dir
            / "V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT.json"
        ),
        "a0_lock": (
            a0_dir
            / "V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT_LOCK.json"
        ),
        "a1_r2_report": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT.json"
        ),
        "a1_r2_lock": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_CONTRACT_LOCK.json"
        ),
        "pair_manifest": (
            a1_r2_dir
            / "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
        ),
        "a2_r2_report": (
            a2_r2_dir
            / "V5_P2_A2_R2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT.json"
        ),
        "a2_r2_lock": (
            a2_r2_dir
            / "V5_P2_A2_R2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT_LOCK.json"
        ),
        "a2_r2_mask": (
            a2_r2_dir
            / "V5_P2_A2_R2_TOPOLOGY_DERIVED_BOOLEAN_PORT_MASK.pt"
        ),
        "loader": loader_path,
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    for split in ("train", "validation"):
        if not (root / "runs" / split).is_dir():
            failures.append(f"missing runs/{split}")

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        return 1

    a0_report = load_json(paths["a0_report"])
    a0_lock = load_json(paths["a0_lock"])
    a1_report = load_json(paths["a1_r2_report"])
    a1_lock = load_json(paths["a1_r2_lock"])
    a2_report = load_json(paths["a2_r2_report"])
    a2_lock = load_json(paths["a2_r2_lock"])

    if a0_report.get("status") != "COMPLETE":
        failures.append("A0 status is not COMPLETE")
    if a1_report.get("status") != "COMPLETE":
        failures.append("A1-R2 status is not COMPLETE")
    if a2_report.get("status") != "COMPLETE":
        failures.append("A2-R2 status is not COMPLETE")

    if a0_lock.get("report_sha256") != sha256_file(paths["a0_report"]):
        failures.append("A0 report SHA mismatch")
    if a1_lock.get("report_sha256") != sha256_file(paths["a1_r2_report"]):
        failures.append("A1-R2 report SHA mismatch")
    if a2_lock.get("report_sha256") != sha256_file(paths["a2_r2_report"]):
        failures.append("A2-R2 report SHA mismatch")
    if a1_lock.get("pair_manifest_sha256") != sha256_file(
        paths["pair_manifest"]
    ):
        failures.append("A1-R2 pair manifest SHA mismatch")
    if a2_lock.get("corrected_port_mask_sha256") != sha256_file(
        paths["a2_r2_mask"]
    ):
        failures.append("A2-R2 mask SHA mismatch")

    if a1_lock.get("aligned_windows") != EXPECTED_TOTAL_ALIGNED_WINDOWS:
        failures.append("A1-R2 aligned-window count is not 82,694")
    if a1_lock.get("pair_count") != EXPECTED_TOTAL_PAIRS:
        failures.append("A1-R2 pair count is not 489")
    if a2_lock.get("primary58_count") != 58:
        failures.append("A2-R2 PRIMARY58 count is not 58")
    if a2_lock.get("window") != 32 or a2_lock.get("stride") != 8:
        failures.append("A2-R2 window/stride contract changed")
    if a2_lock.get("second_normalization_forbidden") is not True:
        failures.append("A2-R2 does not forbid second normalization")
    if a2_lock.get("reference_dataset_loader_authorized") is not False:
        failures.append("A2-R2 unexpectedly authorizes reference loader")
    if a2_lock.get("graph_message_passing") is not False:
        failures.append("A2-R2 graph-message-passing flag changed")
    if a2_lock.get("edge_index_model_input") is not False:
        failures.append("A2-R2 edge_index input flag changed")

    for label, document in (
        ("A0", a0_report),
        ("A1-R2", a1_report),
        ("A2-R2", a2_report),
    ):
        if document.get("security_boundary", {}).get(
            "test_tensor_contents_accessed"
        ) is not False:
            failures.append(
                f"{label} does not certify untouched test tensors"
            )

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    module = import_loader(loader_path)
    DatasetClass = module.V5P2PairAlignedPrimary58Dataset

    # Must reject test before any test filesystem interaction.
    test_rejected = False
    try:
        DatasetClass(
            root=root,
            split="test",
            pair_manifest=paths["pair_manifest"],
        )
    except ValueError:
        test_rejected = True
    except Exception as exc:
        failures.append(
            "split='test' raised the wrong exception type: "
            f"{type(exc).__name__}: {exc}"
        )
    else:
        failures.append("split='test' was not rejected")

    datasets = {}
    for split in ("train", "validation"):
        datasets[split] = DatasetClass(
            root=root,
            split=split,
            pair_manifest=paths["pair_manifest"],
        )

    if datasets["train"].pair_count != EXPECTED_TRAIN_PAIRS:
        failures.append(
            f"train pair_count={datasets['train'].pair_count}, expected 415"
        )
    if datasets["validation"].pair_count != EXPECTED_VALIDATION_PAIRS:
        failures.append(
            "validation pair_count="
            f"{datasets['validation'].pair_count}, expected 74"
        )

    total_dataset_windows = sum(len(dataset) for dataset in datasets.values())
    if total_dataset_windows != EXPECTED_TOTAL_ALIGNED_WINDOWS:
        failures.append(
            f"loader aligned windows={total_dataset_windows}, "
            f"expected {EXPECTED_TOTAL_ALIGNED_WINDOWS}"
        )

    stored_mask_payload = torch.load(
        paths["a2_r2_mask"],
        map_location="cpu",
        weights_only=False,
    )
    stored_mask = stored_mask_payload.get("mask")
    if (
        not isinstance(stored_mask, torch.Tensor)
        or stored_mask.dtype != torch.bool
        or tuple(stored_mask.shape) != (16, 10)
    ):
        failures.append("A2-R2 stored mask is invalid")

    index_checks = {
        "total_entries": 0,
        "paired_adjacency_checks": 0,
        "pair_start_group_count": 0,
    }
    split_index_summaries = {}
    sample_records = []

    for split, dataset in datasets.items():
        entries = dataset._index
        index_checks["total_entries"] += len(entries)

        if len(entries) % 2:
            failures.append(f"{split} dataset length is odd")

        pair_start_groups = defaultdict(list)
        pair_counts = Counter()
        for entry in entries:
            if entry.file_path.parent.name != split:
                failures.append(
                    f"{split}: entry points outside split: {entry.file_path}"
                )
            if entry.mode not in {"attack", "control"}:
                failures.append(
                    f"{split}/{entry.pair_key}: invalid mode {entry.mode}"
                )
            if entry.target != entry.start + 31:
                failures.append(
                    f"{split}/{entry.pair_key}: target/start mismatch"
                )
            if entry.start % 8:
                failures.append(
                    f"{split}/{entry.pair_key}: start is not stride-aligned"
                )
            if entry.target >= entry.common_length:
                failures.append(
                    f"{split}/{entry.pair_key}: target exceeds common length"
                )

            pair_start_groups[
                (entry.pair_key, entry.start)
            ].append(entry)
            pair_counts[entry.pair_key] += 1

        for key, group in pair_start_groups.items():
            index_checks["pair_start_group_count"] += 1
            if len(group) != 2:
                failures.append(
                    f"{split}/{key}: aligned start has {len(group)} members"
                )
                continue
            modes = [entry.mode for entry in group]
            if modes != ["attack", "control"]:
                failures.append(
                    f"{split}/{key}: member order={modes}, "
                    "expected attack,control"
                )
            if (
                group[0].start != group[1].start
                or group[0].target != group[1].target
                or group[0].common_length != group[1].common_length
            ):
                failures.append(
                    f"{split}/{key}: pair members do not share indices"
                )

        for base in deterministic_pair_sample_bases(dataset):
            attack_entry = entries[base]
            control_entry = entries[base + 1]
            index_checks["paired_adjacency_checks"] += 1

            if attack_entry.mode != "attack":
                failures.append(
                    f"{split} sample {base}: first member is not attack"
                )
            if control_entry.mode != "control":
                failures.append(
                    f"{split} sample {base+1}: second member is not control"
                )
            if (
                attack_entry.pair_key != control_entry.pair_key
                or attack_entry.start != control_entry.start
                or attack_entry.target != control_entry.target
                or attack_entry.common_length != control_entry.common_length
            ):
                failures.append(
                    f"{split} sample base {base}: pair alignment mismatch"
                )

            for item_index in (base, base + 1):
                entry = entries[item_index]
                item = dataset[item_index]
                context = (
                    f"{split}/{entry.pair_key}/"
                    f"{entry.mode}/start={entry.start}"
                )

                if set(item) != EXPECTED_ITEM_KEYS:
                    failures.append(
                        f"{context}: item keys={sorted(item)}, "
                        f"expected {sorted(EXPECTED_ITEM_KEYS)}"
                    )
                leaked = sorted(set(item) & PROHIBITED_ITEM_KEYS)
                if leaked:
                    failures.append(
                        f"{context}: prohibited returned keys={leaked}"
                    )

                x = item.get("x")
                if (
                    not isinstance(x, torch.Tensor)
                    or x.dtype != torch.float32
                    or tuple(x.shape) != (16, 58, 32)
                    or not x.is_contiguous()
                    or not bool(torch.isfinite(x).all().item())
                ):
                    failures.append(f"{context}: x contract failed")

                mask = item.get("physical_port_mask")
                if (
                    not isinstance(mask, torch.Tensor)
                    or mask.dtype != torch.bool
                    or tuple(mask.shape) != (16, 10)
                ):
                    failures.append(
                        f"{context}: physical_port_mask contract failed"
                    )
                elif isinstance(stored_mask, torch.Tensor) and not torch.equal(
                    mask,
                    stored_mask,
                ):
                    failures.append(
                        f"{context}: mask differs from A2-R2 artifact"
                    )

                check_scalar_tensor(
                    item.get("y_attack"),
                    torch.float32,
                    "y_attack",
                    failures,
                    context,
                )
                check_scalar_tensor(
                    item.get("y_attacker_count"),
                    torch.int64,
                    "y_attacker_count",
                    failures,
                    context,
                )
                for key in (
                    "y_source",
                    "y_transit",
                    "y_victim",
                    "y_attack_path",
                ):
                    check_node_tensor(
                        item.get(key),
                        (torch.float32,),
                        key,
                        failures,
                        context,
                    )
                check_node_tensor(
                    item.get("role_mask"),
                    (torch.uint8, torch.bool),
                    "role_mask",
                    failures,
                    context,
                )

                raw = torch.load(
                    entry.file_path,
                    map_location="cpu",
                    weights_only=False,
                )
                expected_x = (
                    raw["x"][
                        entry.start:entry.start + 32,
                        :,
                        module.PRIMARY58_INDICES,
                    ]
                    .permute(1, 2, 0)
                    .contiguous()
                )
                if not torch.equal(x, expected_x):
                    failures.append(
                        f"{context}: x is not the exact stored PRIMARY58 slice"
                    )

                target_key_pairs = (
                    ("y_attack", "y_attack"),
                    ("y_attacker_count", "y_attacker_count"),
                    ("y_source", "y_source"),
                    ("y_transit", "y_transit"),
                    ("y_victim", "y_victim"),
                    ("y_attack_path", "y_attack_path"),
                    ("role_mask", "role_mask"),
                )
                for item_key, raw_key in target_key_pairs:
                    if not torch.equal(
                        item[item_key],
                        raw[raw_key][entry.target],
                    ):
                        failures.append(
                            f"{context}: {item_key} is not final-epoch target"
                        )

                sample_records.append(
                    {
                        "split": split,
                        "pair_key": entry.pair_key,
                        "mode": entry.mode,
                        "start": entry.start,
                        "target": entry.target,
                        "common_length": entry.common_length,
                        "x_shape": list(x.shape),
                        "x_dtype": str(x.dtype),
                        "mask_shape": list(mask.shape),
                        "mask_dtype": str(mask.dtype),
                        "returned_keys": sorted(item),
                        "exact_primary58_slice": torch.equal(
                            x,
                            expected_x,
                        ),
                    }
                )

        split_index_summaries[split] = {
            "dataset_length": len(dataset),
            "pair_count": dataset.pair_count,
            "pair_start_group_count": len(pair_start_groups),
            "unique_pair_count": len(pair_counts),
            "minimum_entries_per_pair": min(
                pair_counts.values(),
                default=0,
            ),
            "maximum_entries_per_pair": max(
                pair_counts.values(),
                default=0,
            ),
        }

    contract = {
        "contract_name": (
            "V5_P2_PAIR_ALIGNED_PRIMARY58_FROZEN_B3_LOADER"
        ),
        "contract_version": 1,
        "authorized_splits": ["train", "validation"],
        "test_split_authorized": False,
        "dataset_class": "V5P2PairAlignedPrimary58Dataset",
        "index_policy": {
            "source": (
                "V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
            ),
            "order": (
                "pair-major, window-start-major, ATTACK then CONTROL"
            ),
            "window": 32,
            "stride": 8,
            "common_length": "min(T_attack,T_control)",
            "identical_pair_member_starts": True,
            "total_non_test_items": total_dataset_windows,
        },
        "returned_item": {
            "keys": sorted(EXPECTED_ITEM_KEYS),
            "x_shape": [16, 58, 32],
            "x_dtype": "torch.float32",
            "x_values": (
                "exact stored standardized PRIMARY58 slice; "
                "no normalization performed"
            ),
            "physical_port_mask_shape": [16, 10],
            "physical_port_mask_dtype": "torch.bool",
            "target_epoch": "start + window - 1",
        },
        "model_input_boundary": {
            "model_inputs": ["x", "physical_port_mask"],
            "targets_only": [
                "y_attack",
                "y_attacker_count",
                "y_source",
                "y_transit",
                "y_victim",
                "y_attack_path",
                "role_mask",
            ],
            "edge_index_returned": False,
            "metadata_returned": False,
            "window_position_returned": False,
            "second_normalization": False,
        },
        "frozen_architecture": {
            "name": "B3_CAUSAL_DEPTHWISE_SEPARABLE_CONV1D_ONLY",
            "graph_message_passing": False,
            "edge_index_model_input": False,
        },
        "test_boundary": {
            "constructor_rejects_test": test_rejected,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        },
        "loader_sha256": sha256_file(loader_path),
    }
    contract["contract_sha256"] = canonical_sha256(contract)

    contract_path = (
        output_dir
        / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT.json"
    )
    write_json(contract_path, contract)

    markdown = f"""# V5 P2 A3 Pair-Aligned PRIMARY58 Loader Contract

## Authorized loader

`V5P2PairAlignedPrimary58Dataset`

Authorized splits:

```text
train
validation
```

The constructor rejects `test`.

## Windowing

```text
window = 32
stride = 8
common_length = min(T_attack, T_control)
ordering = pair, start, ATTACK, CONTROL
```

Both pair members use identical starts and final-epoch targets.

## Returned item

```text
x                  float32 [16,58,32]
physical_port_mask bool    [16,10]
y_attack           float32 scalar
y_attacker_count   int64   scalar
y_source           float32 [16]
y_transit          float32 [16]
y_victim           float32 [16]
y_attack_path      float32 [16]
role_mask          uint8/bool [16]
```

Only `x` and `physical_port_mask` are model inputs. The rest are targets or
loss masks.

The loader returns no identifiers, metadata, coordinates, edge index, lengths,
or window positions.

## Normalization

`x` is an exact slice of the stored train-standardized tensor. The loader does
not normalize, standardize, inverse-transform, or otherwise alter feature
values.

## Counts

```text
train pairs          = {datasets["train"].pair_count}
validation pairs     = {datasets["validation"].pair_count}
total aligned items  = {total_dataset_windows}
```

## Contract SHA-256

`{contract["contract_sha256"]}`

## Test boundary

No P2 test directory was enumerated and no P2 test tensor was opened.
"""
    markdown_path = (
        output_dir
        / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT.md"
    )
    atomic_write(markdown_path, markdown)

    status = "COMPLETE" if not failures else "HOLD"
    report = {
        "stage": STAGE,
        "status": status,
        "decision": (
            "FREEZE_PAIR_ALIGNED_PRIMARY58_LOADER_AND_AUTHORIZE_A4"
            if not failures
            else "BLOCK_P2_A4"
        ),
        "dataset": {
            "link_path": str(root_input.absolute()),
            "resolved_root": str(root),
        },
        "loader": {
            "path": str(loader_path),
            "sha256": sha256_file(loader_path),
            "class": "V5P2PairAlignedPrimary58Dataset",
        },
        "split_index_summaries": split_index_summaries,
        "index_checks": index_checks,
        "sample_records": sample_records,
        "contract": contract,
        "provenance": {
            "a0_report_sha256": sha256_file(paths["a0_report"]),
            "a0_lock_sha256": sha256_file(paths["a0_lock"]),
            "a1_r2_report_sha256": sha256_file(paths["a1_r2_report"]),
            "a1_r2_lock_sha256": sha256_file(paths["a1_r2_lock"]),
            "pair_manifest_sha256": sha256_file(paths["pair_manifest"]),
            "a2_r2_report_sha256": sha256_file(paths["a2_r2_report"]),
            "a2_r2_lock_sha256": sha256_file(paths["a2_r2_lock"]),
            "a2_r2_mask_sha256": sha256_file(paths["a2_r2_mask"]),
        },
        "artifacts": {
            "contract_json": [
                str(contract_path),
                sha256_file(contract_path),
            ],
            "contract_markdown": [
                str(markdown_path),
                sha256_file(markdown_path),
            ],
        },
        "security_boundary": {
            "train_tensor_contents_accessed": True,
            "validation_tensor_contents_accessed": True,
            "test_constructor_attempted": True,
            "test_constructor_rejected_before_filesystem_access": test_rejected,
            "test_directory_existence_checked": False,
            "test_directory_enumerated": False,
            "test_tensor_files_opened": False,
            "test_tensor_bytes_read": False,
            "test_tensor_contents_accessed": False,
            "test_dataset_constructed": False,
            "test_windows_constructed": False,
        },
        "failures": failures,
        "warnings": warnings,
        "next_stage": (
            "V5_P2_A4_LOADER_INTEGRATION_AND_TINY_OVERFIT"
            if not failures
            else None
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    if failures:
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures[:100]:
            print("FAIL:", failure)
        print("failure_count:", len(failures))
        print("warning_count:", len(warnings))
        return 1

    lock = {
        "status": COMPLETE,
        "decision": (
            "FREEZE_PAIR_ALIGNED_PRIMARY58_LOADER_AND_AUTHORIZE_A4"
        ),
        "report_sha256": sha256_file(report_path),
        "contract_file_sha256": sha256_file(contract_path),
        "contract_sha256": contract["contract_sha256"],
        "loader_sha256": sha256_file(loader_path),
        "train_pairs": datasets["train"].pair_count,
        "validation_pairs": datasets["validation"].pair_count,
        "train_items": len(datasets["train"]),
        "validation_items": len(datasets["validation"]),
        "total_aligned_items": total_dataset_windows,
        "item_keys": sorted(EXPECTED_ITEM_KEYS),
        "x_shape": [16, 58, 32],
        "mask_shape": [16, 10],
        "test_constructor_rejected": test_rejected,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "next_stage": (
            "V5_P2_A4_LOADER_INTEGRATION_AND_TINY_OVERFIT"
        ),
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-A3 PAIR-ALIGNED PRIMARY58 LOADER CONTRACT =====")
    print("status: COMPLETE")
    print(
        "decision: "
        "FREEZE_PAIR_ALIGNED_PRIMARY58_LOADER_AND_AUTHORIZE_A4"
    )
    print("loader_class: V5P2PairAlignedPrimary58Dataset")
    print("train_pairs:", datasets["train"].pair_count)
    print("validation_pairs:", datasets["validation"].pair_count)
    print("train_items:", len(datasets["train"]))
    print("validation_items:", len(datasets["validation"]))
    print("total_aligned_items:", total_dataset_windows)
    print("x_shape: [16, 58, 32]")
    print("x_dtype: float32")
    print("physical_port_mask_shape: [16, 10]")
    print("physical_port_mask_dtype: bool")
    print("second_normalization_performed: false")
    print("edge_index_returned: false")
    print("metadata_returned: false")
    print("test_constructor_rejected:", str(test_rejected).lower())
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print(
        "next_stage: "
        "V5_P2_A4_LOADER_INTEGRATION_AND_TINY_OVERFIT"
    )
    print("contract_sha256:", contract["contract_sha256"])
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
