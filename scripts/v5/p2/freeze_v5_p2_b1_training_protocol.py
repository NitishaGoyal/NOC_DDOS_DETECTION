#!/usr/bin/env python3
"""
V5 P2-B1 Training Protocol Lock

This stage freezes the complete non-test training and checkpoint-selection
protocol for the corrected 43,273-parameter P2 B3 Conv1D-only model.

It reads only prior reports, locks, the audited loader source, and the corrected
model source. It does not deserialize any run tensor, construct any dataset,
enumerate runs/test, or perform training.

The next stage is:
    V5_P2_B2_MULTI_SEED_TRAINING
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


STAGE = "V5_P2_B1_TRAINING_PROTOCOL_LOCK"
COMPLETE = f"{STAGE}_COMPLETE"

EXPECTED_MODEL_PARAMETERS = 43_273
EXPECTED_COUNT_CLASSES = [1, 2, 3, 4]
EXPECTED_TRAIN_ITEMS = 70_166
EXPECTED_VALIDATION_ITEMS = 12_528
EXPECTED_TOTAL_NONTEST_ITEMS = 82_694

SEEDS = [107, 117, 127]

LOSS_WEIGHTS = {
    "attack": 1.0,
    "count": 0.5,
    "source": 1.0,
    "transit": 0.5,
    "victim": 0.75,
    "path": 0.5,
}

SELECTION_WEIGHTS = {
    "graph_auroc": 0.30,
    "graph_average_precision": 0.15,
    "count_macro_f1_active": 0.15,
    "source_average_precision": 0.10,
    "transit_average_precision": 0.10,
    "victim_average_precision": 0.10,
    "path_average_precision": 0.10,
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


def import_model(path: Path):
    spec = importlib.util.spec_from_file_location(
        "v5_p2_b3_count4_b1_protocol",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import corrected model from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def explicit_false(document: dict[str, Any], key: str) -> bool:
    boundary = document.get("security_boundary")
    if isinstance(boundary, dict) and key in boundary:
        return boundary[key] is False
    return document.get(key) is False


def normalized_inverse_frequency(
    counts: dict[int, int],
) -> dict[int, float]:
    total = sum(counts.values())
    class_count = len(counts)
    return {
        label: total / (class_count * frequency)
        for label, frequency in counts.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1-r2-dir", type=Path, required=True)
    parser.add_argument("--a2-r2-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--b0-r2-dir", type=Path, required=True)
    parser.add_argument("--b0-r3-dir", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    a1_r2_dir = args.a1_r2_dir.expanduser().resolve()
    a2_r2_dir = args.a2_r2_dir.expanduser().resolve()
    a3_dir = args.a3_dir.expanduser().resolve()
    b0_r2_dir = args.b0_r2_dir.expanduser().resolve()
    b0_r3_dir = args.b0_r3_dir.expanduser().resolve()
    loader_path = args.loader_path.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    paths = {
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
        "a3_report": (
            a3_dir
            / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT.json"
        ),
        "a3_lock": (
            a3_dir
            / "V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT_LOCK.json"
        ),
        "b0_r2_report": (
            b0_r2_dir
            / "V5_P2_B0_R2_COUNT_HEAD_EXPANSION_AND_A4_REPEAT.json"
        ),
        "b0_r2_lock": (
            b0_r2_dir
            / "V5_P2_B0_R2_COUNT_HEAD_EXPANSION_AND_A4_REPEAT_LOCK.json"
        ),
        "b0_r3_report": (
            b0_r3_dir
            / "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT.json"
        ),
        "b0_r3_lock": (
            b0_r3_dir
            / "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT_LOCK.json"
        ),
        "loader": loader_path,
        "model": model_path,
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "run_tensors_deserialized": False,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        return 1

    reports = {
        "A1-R2": load_json(paths["a1_r2_report"]),
        "A2-R2": load_json(paths["a2_r2_report"]),
        "A3": load_json(paths["a3_report"]),
        "B0-R2": load_json(paths["b0_r2_report"]),
        "B0-R3": load_json(paths["b0_r3_report"]),
    }
    locks = {
        "A1-R2": load_json(paths["a1_r2_lock"]),
        "A2-R2": load_json(paths["a2_r2_lock"]),
        "A3": load_json(paths["a3_lock"]),
        "B0-R2": load_json(paths["b0_r2_lock"]),
        "B0-R3": load_json(paths["b0_r3_lock"]),
    }
    report_paths = {
        "A1-R2": paths["a1_r2_report"],
        "A2-R2": paths["a2_r2_report"],
        "A3": paths["a3_report"],
        "B0-R2": paths["b0_r2_report"],
        "B0-R3": paths["b0_r3_report"],
    }

    for label in reports:
        if reports[label].get("status") != "COMPLETE":
            failures.append(f"{label} status is not COMPLETE")
        if (
            locks[label].get("report_sha256")
            != sha256_file(report_paths[label])
        ):
            failures.append(f"{label} report SHA mismatch")
        if not explicit_false(
            reports[label],
            "test_tensor_contents_accessed",
        ):
            failures.append(
                f"{label} does not certify untouched test tensors"
            )

    if (
        locks["A1-R2"].get("pair_manifest_sha256")
        != sha256_file(paths["pair_manifest"])
    ):
        failures.append("A1-R2 pair-manifest SHA mismatch")
    if locks["A1-R2"].get("window") != 32:
        failures.append("A1-R2 window changed")
    if locks["A1-R2"].get("stride") != 8:
        failures.append("A1-R2 stride changed")
    if locks["A1-R2"].get("aligned_windows") != 82_694:
        failures.append("A1-R2 aligned-window count changed")

    if locks["A2-R2"].get("primary58_count") != 58:
        failures.append("A2-R2 PRIMARY58 count changed")
    if locks["A2-R2"].get(
        "second_normalization_forbidden"
    ) is not True:
        failures.append("A2-R2 normalization contract changed")
    if locks["A2-R2"].get("graph_message_passing") is not False:
        failures.append("A2-R2 message-passing contract changed")
    if locks["A2-R2"].get("edge_index_model_input") is not False:
        failures.append("A2-R2 edge-index contract changed")

    if (
        locks["A3"].get("loader_sha256")
        != sha256_file(paths["loader"])
    ):
        failures.append("A3 loader SHA mismatch")
    if locks["A3"].get("train_items") != EXPECTED_TRAIN_ITEMS:
        failures.append("A3 train-item count changed")
    if (
        locks["A3"].get("validation_items")
        != EXPECTED_VALIDATION_ITEMS
    ):
        failures.append("A3 validation-item count changed")
    if (
        locks["A3"].get("total_aligned_items")
        != EXPECTED_TOTAL_NONTEST_ITEMS
    ):
        failures.append("A3 total-item count changed")
    if locks["A3"].get("test_constructor_rejected") is not True:
        failures.append("A3 test-constructor rejection changed")

    if locks["B0-R2"].get("parameter_count") != EXPECTED_MODEL_PARAMETERS:
        failures.append("B0-R2 parameter count changed")
    if locks["B0-R2"].get("count_logits") != 4:
        failures.append("B0-R2 count-head width changed")
    if (
        locks["B0-R2"].get("count_class_values")
        != EXPECTED_COUNT_CLASSES
    ):
        failures.append("B0-R2 count-class values changed")
    if locks["B0-R2"].get("all_tasks_all_exact") is not True:
        failures.append("B0-R2 integration repeat was not exact")
    if locks["B0-R2"].get("audit_weights_saved") is not False:
        failures.append("B0-R2 unexpectedly saved audit weights")
    if (
        locks["B0-R2"].get("model_sha256")
        != sha256_file(paths["model"])
    ):
        failures.append("B0-R2 corrected-model SHA mismatch")

    if (
        locks["B0-R3"].get("decision")
        != "AUTHORIZE_P2_B1_TRAINING_PROTOCOL_LOCK"
    ):
        failures.append("B0-R3 did not authorize B1")
    if locks["B0-R3"].get("shortcut_block_count") != 0:
        failures.append("B0-R3 shortcut-block count is not zero")
    if locks["B0-R3"].get("label_integrity_pass") is not True:
        failures.append("B0-R3 label-integrity gate did not pass")
    if locks["B0-R3"].get("count_head_logits") != 4:
        failures.append("B0-R3 count-head width changed")
    if (
        locks["B0-R3"].get("role_mask_semantics")
        != "bitfield3_source1_transit2_victim4"
    ):
        failures.append("B0-R3 role-mask semantics changed")
    if (
        locks["B0-R3"].get("corrected_model_sha256")
        != sha256_file(paths["model"])
    ):
        failures.append("B0-R3 corrected-model SHA mismatch")

    model_module = import_model(model_path)
    model = model_module.P2B3Conv1DOnlyCount4()
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if parameter_count != EXPECTED_MODEL_PARAMETERS:
        failures.append(
            f"imported corrected model has {parameter_count} parameters"
        )
    if model.count_head.out_features != 4:
        failures.append("imported corrected model count head is not 4")
    if list(model.count_class_values) != EXPECTED_COUNT_CLASSES:
        failures.append("imported model count-class contract changed")

    b0_r3_report = reports["B0-R3"]
    label_summaries = b0_r3_report.get("label_summaries", {})
    train_summary = label_summaries.get("train", {})
    validation_summary = label_summaries.get("validation", {})

    train_positive = int(train_summary.get("graph_positive", -1))
    train_negative = int(train_summary.get("graph_negative", -1))
    validation_positive = int(
        validation_summary.get("graph_positive", -1)
    )
    validation_negative = int(
        validation_summary.get("graph_negative", -1)
    )

    if train_positive <= 0 or train_negative <= 0:
        failures.append("B0-R3 train graph counts are invalid")
    if validation_positive <= 0 or validation_negative <= 0:
        failures.append("B0-R3 validation graph counts are invalid")
    if train_positive + train_negative != EXPECTED_TRAIN_ITEMS:
        failures.append("B0-R3 train graph count total changed")
    if (
        validation_positive + validation_negative
        != EXPECTED_VALIDATION_ITEMS
    ):
        failures.append("B0-R3 validation graph count total changed")

    raw_count_distribution = train_summary.get(
        "active_count_distribution",
        {},
    )
    try:
        train_count_distribution = {
            int(label): int(frequency)
            for label, frequency in raw_count_distribution.items()
        }
    except Exception as exc:
        failures.append(
            f"cannot parse train active-count distribution: {exc}"
        )
        train_count_distribution = {}

    if sorted(train_count_distribution) != EXPECTED_COUNT_CLASSES:
        failures.append(
            "B0-R3 train active-count classes are not [1,2,3,4]"
        )
    if any(value <= 0 for value in train_count_distribution.values()):
        failures.append("B0-R3 train count frequencies are invalid")

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "run_tensors_deserialized": False,
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    graph_pos_weight = train_negative / train_positive
    count_inverse_frequency = normalized_inverse_frequency(
        train_count_distribution
    )

    protocol = {
        "contract_name": "V5_P2_B3_COUNT4_MULTI_SEED_TRAINING_PROTOCOL",
        "contract_version": 1,
        "architecture": {
            "class": "P2B3Conv1DOnlyCount4",
            "name": model.architecture_name,
            "parameter_count": EXPECTED_MODEL_PARAMETERS,
            "count_logits": 4,
            "count_class_values": EXPECTED_COUNT_CLASSES,
            "count_class_mapping": {
                "1": 0,
                "2": 1,
                "3": 2,
                "4": 3,
            },
            "graph_message_passing": False,
            "edge_index_model_input": False,
            "role_mask_model_input": False,
            "role_mask_model_output": False,
            "role_mask_semantics": (
                "source + 2*transit + 4*victim"
            ),
            "model_sha256": sha256_file(model_path),
        },
        "data": {
            "loader_class": "V5P2PairAlignedPrimary58Dataset",
            "loader_sha256": sha256_file(loader_path),
            "authorized_splits": ["train", "validation"],
            "test_split_authorized": False,
            "window": 32,
            "stride": 8,
            "pair_aligned_common_prefix": True,
            "primary58": True,
            "second_normalization": False,
            "train_items": EXPECTED_TRAIN_ITEMS,
            "validation_items": EXPECTED_VALIDATION_ITEMS,
            "batch_size_items": 256,
            "batch_size_pair_windows": 128,
            "batching": (
                "shuffle aligned ATTACK/CONTROL two-item blocks; "
                "preserve adjacency inside each block"
            ),
            "drop_last_train": False,
            "shuffle_validation": False,
            "num_workers": 0,
            "pin_memory_on_cuda": True,
        },
        "reproducibility": {
            "seeds": SEEDS,
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "precision": "float32",
            "automatic_mixed_precision": False,
            "model_initialization": "PyTorch module defaults under seed",
            "one_training_run_per_seed": True,
            "resume_after_protocol_change": False,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 1e-4,
            "gradient_clip_global_norm": 1.0,
        },
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": "max",
            "factor": 0.5,
            "patience_epochs": 4,
            "threshold": 1e-4,
            "threshold_mode": "abs",
            "cooldown_epochs": 0,
            "minimum_learning_rate": 1e-5,
        },
        "epochs_and_early_stopping": {
            "maximum_epochs": 100,
            "minimum_epochs_before_stop": 15,
            "early_stopping_patience_epochs": 12,
            "early_stopping_minimum_delta": 1e-4,
            "monitor": "validation_selection_score",
            "mode": "max",
        },
        "losses": {
            "total": (
                "1.0*attack_bce + 0.5*count_ce + "
                "1.0*source_bce + 0.5*transit_bce + "
                "0.75*victim_bce + 0.5*path_bce"
            ),
            "weights": LOSS_WEIGHTS,
            "attack": {
                "type": "BCEWithLogitsLoss",
                "scope": "all items",
                "positive_weight": graph_pos_weight,
                "positive_weight_formula": (
                    "train_graph_negative/train_graph_positive"
                ),
                "train_positive_items": train_positive,
                "train_negative_items": train_negative,
            },
            "count": {
                "type": "CrossEntropyLoss",
                "scope": "graph-positive attack items only",
                "classes": EXPECTED_COUNT_CLASSES,
                "class_mapping": {
                    "1": 0,
                    "2": 1,
                    "3": 2,
                    "4": 3,
                },
                "class_weighting": "none",
                "observed_train_distribution": {
                    str(label): frequency
                    for label, frequency in sorted(
                        train_count_distribution.items()
                    )
                },
                "diagnostic_inverse_frequency_weights_not_applied": {
                    str(label): value
                    for label, value in sorted(
                        count_inverse_frequency.items()
                    )
                },
                "label_smoothing": 0.0,
            },
            "source": {
                "type": "BCEWithLogitsLoss",
                "scope": "all items and nodes",
                "positive_weight": (
                    "compute once from complete train aligned labels as "
                    "negative_nodes/positive_nodes; clamp to [1,20]; "
                    "save in each seed manifest"
                ),
            },
            "transit": {
                "type": "BCEWithLogitsLoss",
                "scope": "all items and nodes",
                "positive_weight": (
                    "compute once from complete train aligned labels as "
                    "negative_nodes/positive_nodes; clamp to [1,20]; "
                    "save in each seed manifest"
                ),
            },
            "victim": {
                "type": "BCEWithLogitsLoss",
                "scope": "all items and nodes",
                "positive_weight": (
                    "compute once from complete train aligned labels as "
                    "negative_nodes/positive_nodes; clamp to [1,20]; "
                    "save in each seed manifest"
                ),
            },
            "path": {
                "type": "BCEWithLogitsLoss",
                "scope": "all items and nodes",
                "positive_weight": (
                    "compute once from complete train aligned labels as "
                    "negative_nodes/positive_nodes; clamp to [1,20]; "
                    "save in each seed manifest"
                ),
            },
            "role_mask_loss": False,
        },
        "validation": {
            "frequency": "every epoch",
            "split": "validation",
            "full_split": True,
            "threshold_tuning_during_training": False,
            "binary_head_metrics": [
                "AUROC",
                "average_precision",
                "fixed_0.5_precision",
                "fixed_0.5_recall",
                "fixed_0.5_f1",
            ],
            "count_metrics": [
                "active_macro_f1",
                "active_accuracy",
                "active_confusion_matrix",
            ],
            "selection_score": {
                "formula": (
                    "0.30*graph_AUROC + "
                    "0.15*graph_average_precision + "
                    "0.15*count_active_macro_f1 + "
                    "0.10*source_average_precision + "
                    "0.10*transit_average_precision + "
                    "0.10*victim_average_precision + "
                    "0.10*path_average_precision"
                ),
                "weights": SELECTION_WEIGHTS,
                "threshold_free_except_count_argmax": True,
                "undefined_metric_policy": (
                    "HOLD; no metric may be silently replaced or skipped"
                ),
            },
        },
        "checkpointing": {
            "save_initial_state_hash": True,
            "save_best_checkpoint_per_seed": True,
            "best_checkpoint_rule": (
                "highest validation_selection_score"
            ),
            "within_seed_tie_breakers": [
                "higher graph_AUROC",
                "higher graph_average_precision",
                "lower validation_total_loss",
                "earlier epoch",
            ],
            "cross_seed_selection": (
                "same ranking rule applied to each seed's locked best epoch"
            ),
            "save_last_checkpoint": False,
            "save_optimizer_state_for_best": True,
            "checkpoint_contains": [
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "epoch",
                "seed",
                "protocol_sha256",
                "loader_sha256",
                "model_source_sha256",
                "train_label_weight_manifest",
                "validation_metrics",
            ],
        },
        "forbidden_actions": [
            "constructing or reading the P2 test dataset",
            "using mode, category, case_id, run_id, pair_id, filenames, "
            "lengths, window positions, epoch IDs, coordinates, edge_index, "
            "status_flags, or role_mask as learned inputs",
            "second normalization",
            "hyperparameter changes after B1 lock",
            "seed-specific hyperparameter changes",
            "threshold tuning during B2 training",
            "selecting a checkpoint using test data",
            "reusing A4 or B0-R2 audit weights",
        ],
        "post_training_sequence": [
            "B2: run all three seeds and freeze each best checkpoint",
            "B3: compare locked seed results and select one checkpoint",
            "B4: tune attack/source/transit/victim/path thresholds on "
            "validation only and freeze thresholds",
            "B5: authorize exactly one P2 test evaluation",
        ],
        "baseline_reporting_obligations": {
            "highest_final_epoch_traffic_router_source_hit_at_1": (
                locks["B0-R3"].get(
                    "highest_traffic_source_validation_hit_at_1"
                )
            ),
            "most_common_source_hit_at_1": (
                locks["B0-R3"].get(
                    "most_common_source_validation_hit_at_1"
                )
            ),
            "epoch_position_graph_balanced_accuracy": (
                locks["B0-R3"].get(
                    "epoch_position_validation_balanced_accuracy"
                )
            ),
            "mask_only_role_validation_auroc": (
                locks["B0-R3"].get(
                    "mask_role_validation_auroc"
                )
            ),
            "quarantined_mode_and_category_baselines": (
                "report as leakage diagnostics only; never compare as "
                "deployable baselines"
            ),
        },
        "test_boundary": {
            "test_directory_enumerated": False,
            "test_tensor_contents_accessed": False,
            "test_evaluation_authorized": False,
        },
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)

    protocol_path = (
        output_dir / "V5_P2_B1_TRAINING_PROTOCOL.json"
    )
    write_json(protocol_path, protocol)

    markdown = f"""# V5 P2 B1 Training Protocol Lock

## Corrected architecture

```text
model       = {model.architecture_name}
parameters  = {EXPECTED_MODEL_PARAMETERS}
count head  = 4 active-count logits
classes     = 1, 2, 3, 4
```

Only the P0 count head changed from `Linear(64,3)` to `Linear(64,4)`.
The temporal encoder, graph detector, and four localization heads remain
unchanged.

## Dataset

```text
train items       = {EXPECTED_TRAIN_ITEMS}
validation items  = {EXPECTED_VALIDATION_ITEMS}
window / stride   = 32 / 8
input             = [B,16,58,32]
mask              = [B,16,10]
batch             = 256 items = 128 aligned pair-windows
```

Aligned ATTACK/CONTROL blocks are shuffled as units. The order inside each
two-item block remains ATTACK then CONTROL.

## Runs

```text
seeds       = {SEEDS}
max epochs  = 100
early stop  = patience 12 after minimum epoch 15
precision   = float32
AMP         = disabled
```

## Optimizer

```text
AdamW
lr           = 1e-3
weight decay = 1e-4
gradient clip global norm = 1.0
```

`ReduceLROnPlateau` monitors the validation selection score with factor 0.5,
patience 4, and minimum learning rate `1e-5`.

## Checkpoint-selection score

```text
0.30 graph AUROC
0.15 graph average precision
0.15 active count macro F1
0.10 source average precision
0.10 transit average precision
0.10 victim average precision
0.10 path average precision
```

No threshold tuning occurs during B2. The score is threshold-free except for
count-head argmax.

## Loss

```text
1.00 attack BCE
0.50 active-only four-class count CE
1.00 source BCE
0.50 transit BCE
0.75 victim BCE
0.50 path BCE
```

`role_mask` is the bit-field `source + 2*transit + 4*victim`. It is
bookkeeping only and receives no independent loss.

## Test boundary

P2 test remains unauthorized. It may be evaluated once only after multi-seed
training, seed selection, checkpoint freeze, and validation-only threshold
freeze.

## Protocol SHA-256

`{protocol["protocol_sha256"]}`
"""
    markdown_path = (
        output_dir / "V5_P2_B1_TRAINING_PROTOCOL.md"
    )
    atomic_write(markdown_path, markdown)

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "decision": "FREEZE_P2_TRAINING_PROTOCOL_AND_AUTHORIZE_B2",
        "protocol": protocol,
        "source_statistics": {
            "train_graph_positive": train_positive,
            "train_graph_negative": train_negative,
            "validation_graph_positive": validation_positive,
            "validation_graph_negative": validation_negative,
            "graph_positive_weight": graph_pos_weight,
            "train_active_count_distribution": {
                str(label): frequency
                for label, frequency in sorted(
                    train_count_distribution.items()
                )
            },
        },
        "artifacts": {
            "protocol_json": [
                str(protocol_path),
                sha256_file(protocol_path),
            ],
            "protocol_markdown": [
                str(markdown_path),
                sha256_file(markdown_path),
            ],
        },
        "provenance": {
            name: sha256_file(path)
            for name, path in paths.items()
        },
        "security_boundary": {
            "run_tensors_deserialized": False,
            "train_dataset_constructed": False,
            "validation_dataset_constructed": False,
            "test_directory_existence_checked": False,
            "test_directory_enumerated": False,
            "test_tensor_files_opened": False,
            "test_tensor_bytes_read": False,
            "test_tensor_contents_accessed": False,
            "test_dataset_constructed": False,
            "training_performed": False,
        },
        "failures": failures,
        "warnings": warnings,
        "next_stage": "V5_P2_B2_MULTI_SEED_TRAINING",
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    lock = {
        "status": COMPLETE,
        "decision": "FREEZE_P2_TRAINING_PROTOCOL_AND_AUTHORIZE_B2",
        "report_sha256": sha256_file(report_path),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_markdown_sha256": sha256_file(markdown_path),
        "model_sha256": sha256_file(model_path),
        "loader_sha256": sha256_file(loader_path),
        "parameter_count": EXPECTED_MODEL_PARAMETERS,
        "count_logits": 4,
        "seeds": SEEDS,
        "train_items": EXPECTED_TRAIN_ITEMS,
        "validation_items": EXPECTED_VALIDATION_ITEMS,
        "batch_size_items": 256,
        "maximum_epochs": 100,
        "early_stopping_patience_epochs": 12,
        "test_directory_enumerated": False,
        "test_tensor_contents_accessed": False,
        "test_evaluation_authorized": False,
        "next_stage": "V5_P2_B2_MULTI_SEED_TRAINING",
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-B1 TRAINING PROTOCOL LOCK =====")
    print("status: COMPLETE")
    print(
        "decision: "
        "FREEZE_P2_TRAINING_PROTOCOL_AND_AUTHORIZE_B2"
    )
    print("architecture:", model.architecture_name)
    print("parameter_count:", EXPECTED_MODEL_PARAMETERS)
    print("count_logits: 4")
    print("count_class_values:", EXPECTED_COUNT_CLASSES)
    print("train_items:", EXPECTED_TRAIN_ITEMS)
    print("validation_items:", EXPECTED_VALIDATION_ITEMS)
    print("seeds:", SEEDS)
    print("batch_size_items: 256")
    print("batch_size_pair_windows: 128")
    print("optimizer: AdamW")
    print("learning_rate: 0.001")
    print("weight_decay: 0.0001")
    print("maximum_epochs: 100")
    print("minimum_epochs_before_stop: 15")
    print("early_stopping_patience_epochs: 12")
    print("scheduler: ReduceLROnPlateau")
    print("automatic_mixed_precision: false")
    print("threshold_tuning_during_training: false")
    print("role_mask_loss: false")
    print("run_tensors_deserialized: false")
    print("training_performed: false")
    print("test_directory_enumerated: false")
    print("test_tensor_contents_accessed: false")
    print("test_evaluation_authorized: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print("protocol_sha256:", protocol["protocol_sha256"])
    print("next_stage: V5_P2_B2_MULTI_SEED_TRAINING")
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
