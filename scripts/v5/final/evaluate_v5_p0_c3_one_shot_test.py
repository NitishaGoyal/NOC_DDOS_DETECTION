#!/usr/bin/env python3
"""
V5 P0-C3 One-Shot Locked Test Evaluation

This is the first authorized access to the P0 test split.

The script:
- verifies B8, C0, C1, and C2 reports/locks/hashes;
- verifies the frozen seed-117 epoch-101 B3 checkpoint;
- verifies the frozen C2 thresholds;
- writes TEST_ACCESS_STARTED before constructing the test dataset;
- constructs only the P0 test dataset;
- performs exactly one forward evaluation;
- performs no training, checkpoint selection, threshold sweep, or calibration;
- writes immutable test predictions, metrics, report, lock, and completion marker.

The output directory is a one-shot guard. If evaluation starts and later fails,
the existing output directory prevents an automatic rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


STAGE = "V5_P0_C3_ONE_SHOT_LOCKED_TEST_EVALUATION"
COMPLETE = f"{STAGE}_COMPLETE"
ACCESS_STARTED = f"{STAGE}_TEST_ACCESS_STARTED"
ROLES = ("source", "transit", "victim", "path")
ROLE_KEYS = {
    "source": "y_source",
    "transit": "y_transit",
    "victim": "y_victim",
    "path": "y_attack_path",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def import_symbol(module_path: Path, symbol: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = getattr(module, symbol, None)
    if value is None:
        raise AttributeError(f"{module_path} lacks {symbol}")
    return value


def binary_metrics(
    truth: torch.Tensor,
    prediction: torch.Tensor,
) -> dict[str, Any]:
    truth = truth.detach().cpu().bool()
    prediction = prediction.detach().cpu().bool()

    tp = int((truth & prediction).sum().item())
    tn = int((~truth & ~prediction).sum().item())
    fp = int((~truth & prediction).sum().item())
    fn = int((truth & ~prediction).sum().item())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    f1 = (
        2.0 * precision * recall
        / max(1e-12, precision + recall)
    )

    return {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "balanced_accuracy": 0.5 * (recall + tnr),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / max(1, fp + tn),
        "tnr": tnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support_negative": tn + fp,
        "support_positive": tp + fn,
    }


def multiclass_metrics(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    class_count: int = 3,
) -> dict[str, Any]:
    truth = truth.detach().cpu().long()
    prediction = prediction.detach().cpu().long()

    confusion = torch.zeros(
        class_count,
        class_count,
        dtype=torch.long,
    )
    for actual, predicted in zip(truth.tolist(), prediction.tolist()):
        confusion[actual, predicted] += 1

    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in range(class_count):
        tp = int(confusion[label, label].item())
        fp = int(confusion[:, label].sum().item()) - tp
        fn = int(confusion[label, :].sum().item()) - tp
        support = int(confusion[label, :].sum().item())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (
            2.0 * precision * recall
            / max(1e-12, precision + recall)
        )
        f1_values.append(f1)
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "accuracy": float(
            (truth == prediction).float().mean().item()
        ),
        "macro_f1": sum(f1_values) / class_count,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def node_metrics(
    truth: torch.Tensor,
    prediction: torch.Tensor,
) -> dict[str, Any]:
    truth = truth.detach().cpu().bool()
    prediction = prediction.detach().cpu().bool()

    tp = int((truth & prediction).sum().item())
    tn = int((~truth & ~prediction).sum().item())
    fp = int((~truth & prediction).sum().item())
    fn = int((truth & ~prediction).sum().item())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        2.0 * precision * recall
        / max(1e-12, precision + recall)
    )
    exact = float(
        (truth == prediction)
        .all(dim=1)
        .float()
        .mean()
        .item()
    )

    return {
        "node_accuracy": (
            (tp + tn) / max(1, tp + tn + fp + fn)
        ),
        "node_precision": precision,
        "node_recall": recall,
        "node_f1": f1,
        "exact_set": exact,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_entries": int(truth.sum().item()),
        "window_count": int(truth.shape[0]),
    }


@torch.inference_mode()
def evaluate_once(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()

    values: dict[str, list[torch.Tensor]] = {
        "attack_truth": [],
        "attack_probability": [],
        "attack_prediction": [],
        "count_truth": [],
        "count_prediction": [],
    }
    for role in ROLES:
        values[f"{role}_truth"] = []
        values[f"{role}_probability"] = []
        values[f"{role}_prediction"] = []

    forward_batches = 0
    for batch in loader:
        outputs = model(
            batch["x"].to(device=device, dtype=torch.float32),
            batch["physical_port_mask"].to(device=device),
        )
        forward_batches += 1

        attack_probability = torch.sigmoid(outputs["attack"])
        attack_prediction = (
            attack_probability >= thresholds["attack"]
        )
        count_prediction = outputs["count"].argmax(dim=-1)

        values["attack_truth"].append(
            batch["y_attack"].detach().cpu().bool()
        )
        values["attack_probability"].append(
            attack_probability.detach().cpu()
        )
        values["attack_prediction"].append(
            attack_prediction.detach().cpu()
        )
        values["count_truth"].append(
            batch["y_attacker_count"].detach().cpu().long()
        )
        values["count_prediction"].append(
            count_prediction.detach().cpu()
        )

        for role, target_key in ROLE_KEYS.items():
            probability = torch.sigmoid(outputs[role])
            prediction = probability >= thresholds[role]

            values[f"{role}_truth"].append(
                batch[target_key].detach().cpu().bool()
            )
            values[f"{role}_probability"].append(
                probability.detach().cpu()
            )
            values[f"{role}_prediction"].append(
                prediction.detach().cpu()
            )

    combined = {
        key: torch.cat(parts, dim=0)
        for key, parts in values.items()
    }

    attack_truth = combined["attack_truth"]
    attack_windows = attack_truth.bool()

    metrics: dict[str, Any] = {
        "graph": binary_metrics(
            attack_truth,
            combined["attack_prediction"],
        ),
        "count": multiclass_metrics(
            combined["count_truth"],
            combined["count_prediction"],
        ),
        "roles": {},
        "window_count": int(attack_truth.shape[0]),
        "forward_batch_count": forward_batches,
    }

    all_exact = (
        combined["attack_truth"]
        == combined["attack_prediction"]
    ) & (
        combined["count_truth"]
        == combined["count_prediction"]
    )

    for role in ROLES:
        truth = combined[f"{role}_truth"]
        prediction = combined[f"{role}_prediction"]

        metrics["roles"][role] = {
            "all_windows": node_metrics(truth, prediction),
            "attack_windows": node_metrics(
                truth[attack_windows],
                prediction[attack_windows],
            ),
        }
        all_exact &= (truth == prediction).all(dim=1)

    metrics["all_tasks_exact"] = float(
        all_exact.float().mean().item()
    )
    return metrics, combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--a2-1-dir", type=Path, required=True)
    parser.add_argument("--b8-dir", type=Path, required=True)
    parser.add_argument("--c0-dir", type=Path, required=True)
    parser.add_argument("--c1-report-dir", type=Path, required=True)
    parser.add_argument("--c1-model-dir", type=Path, required=True)
    parser.add_argument("--c2-report-dir", type=Path, required=True)
    parser.add_argument("--c2-model-dir", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--c1-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    a2_dir = args.a2_dir.expanduser().resolve()
    a2_1_dir = args.a2_1_dir.expanduser().resolve()
    b8_dir = args.b8_dir.expanduser().resolve()
    c0_dir = args.c0_dir.expanduser().resolve()
    c1_report_dir = args.c1_report_dir.expanduser().resolve()
    c1_model_dir = args.c1_model_dir.expanduser().resolve()
    c2_report_dir = args.c2_report_dir.expanduser().resolve()
    c2_model_dir = args.c2_model_dir.expanduser().resolve()
    wrapper_path = args.wrapper.expanduser().resolve()
    c1_script_path = args.c1_script.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()

    if output_dir.exists():
        print(
            f"STOP: one-shot output already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    if model_dir.exists():
        print(
            f"STOP: one-shot model directory already exists: {model_dir}",
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    paths = {
        "b8_report": (
            b8_dir
            / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE.json"
        ),
        "b8_lock": (
            b8_dir
            / "V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE_LOCK.json"
        ),
        "b8_contract": (
            b8_dir
            / "V5_P0_B8_B3_CONV1D_ONLY_ARCHITECTURE_CONTRACT.json"
        ),
        "c0_report": (
            c0_dir / "V5_P0_C0_FINAL_B3_TRAINING_PROTOCOL_LOCK.json"
        ),
        "c0_lock": (
            c0_dir
            / "V5_P0_C0_FINAL_B3_TRAINING_PROTOCOL_LOCK_LOCK.json"
        ),
        "c0_protocol": (
            c0_dir / "V5_P0_C0_FINAL_B3_PROTOCOL.json"
        ),
        "c1_report": (
            c1_report_dir
            / "V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING.json"
        ),
        "c1_lock": (
            c1_report_dir
            / "V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING_LOCK.json"
        ),
        "c2_report": (
            c2_report_dir
            / "V5_P0_C2_FINAL_CHECKPOINT_THRESHOLD_FREEZE.json"
        ),
        "c2_lock": (
            c2_report_dir
            / "V5_P0_C2_FINAL_CHECKPOINT_THRESHOLD_FREEZE_LOCK.json"
        ),
        "c2_manifest": (
            c2_report_dir
            / "V5_P0_C2_FROZEN_CHECKPOINT_THRESHOLDS.json"
        ),
        "c2_checkpoint": (
            c2_model_dir / "final_locked_b3_checkpoint.pt"
        ),
        "wrapper": wrapper_path,
        "c1_script": c1_script_path,
    }

    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing prerequisite {name}: {path}")

    if not root.is_dir():
        failures.append(f"P0 dataset root missing: {root}")
    if not a2_dir.is_dir():
        failures.append(f"A2 directory missing: {a2_dir}")
    if not a2_1_dir.is_dir():
        failures.append(f"A2-1 directory missing: {a2_1_dir}")

    documents: dict[str, Any] = {}
    if not failures:
        for name in (
            "b8_report",
            "b8_lock",
            "b8_contract",
            "c0_report",
            "c0_lock",
            "c0_protocol",
            "c1_report",
            "c1_lock",
            "c2_report",
            "c2_lock",
            "c2_manifest",
        ):
            documents[name] = load_json(paths[name])

        for label, name in (
            ("B8", "b8_report"),
            ("C0", "c0_report"),
            ("C1", "c1_report"),
            ("C2", "c2_report"),
        ):
            report = documents[name]
            if report.get("status") != "COMPLETE":
                failures.append(
                    f"{label} status={report.get('status')!r}, "
                    "expected COMPLETE"
                )
            if report.get("test_split_accessed") is not False:
                failures.append(
                    f"{label} does not certify untouched test"
                )

        if (
            documents["b8_lock"].get("report_sha256")
            != sha256_file(paths["b8_report"])
        ):
            failures.append("B8 report SHA mismatch")
        if (
            documents["c0_lock"].get("report_sha256")
            != sha256_file(paths["c0_report"])
        ):
            failures.append("C0 report SHA mismatch")
        if (
            documents["c1_lock"].get("report_sha256")
            != sha256_file(paths["c1_report"])
        ):
            failures.append("C1 report SHA mismatch")
        if (
            documents["c2_lock"].get("report_sha256")
            != sha256_file(paths["c2_report"])
        ):
            failures.append("C2 report SHA mismatch")

        if (
            documents["c2_lock"].get(
                "threshold_manifest_file_sha256"
            )
            != sha256_file(paths["c2_manifest"])
        ):
            failures.append("C2 threshold-manifest file SHA mismatch")
        if (
            documents["c2_lock"].get("final_checkpoint_sha256")
            != sha256_file(paths["c2_checkpoint"])
        ):
            failures.append("C2 final-checkpoint SHA mismatch")

        c1_provenance = documents["c1_report"].get("provenance", {})
        if (
            c1_provenance.get("c1_script_sha256")
            != sha256_file(c1_script_path)
        ):
            failures.append("C1 training script changed after C1 lock")
        if (
            c1_provenance.get("wrapper_sha256")
            != sha256_file(wrapper_path)
        ):
            failures.append("dataset wrapper changed after C1 lock")

        manifest = documents["c2_manifest"]
        thresholds = manifest.get("frozen_thresholds")
        if thresholds != documents["c2_lock"].get("frozen_thresholds"):
            failures.append("C2 frozen thresholds mismatch between files")
        if manifest.get("test_split_accessed") is not False:
            failures.append("C2 manifest does not certify untouched test")
        if manifest.get("checkpoint", {}).get("seed") != 117:
            failures.append("C2 frozen checkpoint seed is not 117")
        if manifest.get("checkpoint", {}).get("epoch") != 101:
            failures.append("C2 frozen checkpoint epoch is not 101")
        if manifest.get("checkpoint", {}).get("parameter_count") != 43208:
            failures.append("C2 parameter count is not 43,208")

        architecture_hash = documents["b8_contract"].get(
            "architecture_contract_sha256"
        )
        protocol_hash = documents["c0_protocol"].get("protocol_sha256")
        if (
            documents["c2_lock"].get("architecture_contract_sha256")
            != architecture_hash
        ):
            failures.append("C2 architecture-contract hash mismatch")
        if (
            documents["c2_lock"].get("protocol_sha256")
            != protocol_hash
        ):
            failures.append("C2 protocol hash mismatch")

    if failures:
        hold = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
            "training_performed": False,
            "test_split_constructed": False,
            "test_split_accessed": False,
        }
        write_json(output_dir / f"{STAGE}.json", hold)
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    manifest = documents["c2_manifest"]
    thresholds = {
        key: float(value)
        for key, value in manifest["frozen_thresholds"].items()
    }

    access_record = {
        "stage": STAGE,
        "event": "TEST_ACCESS_STARTED",
        "authorized_checkpoint_sha256": sha256_file(
            paths["c2_checkpoint"]
        ),
        "authorized_threshold_manifest_file_sha256": sha256_file(
            paths["c2_manifest"]
        ),
        "frozen_thresholds": thresholds,
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "threshold_calibration_allowed": False,
        "test_evaluation_count_authorized": 1,
    }
    write_json(
        output_dir / f"{ACCESS_STARTED}.json",
        access_record,
    )
    atomic_write(
        output_dir / ACCESS_STARTED,
        ACCESS_STARTED + "\n",
    )

    # First test construction occurs only after the irreversible access marker.
    ContractDataset = import_symbol(
        wrapper_path,
        "V5P0ContractWindowDataset",
        "v5_p0_contract_loader_c3",
    )
    FrozenB3Model = import_symbol(
        c1_script_path,
        "FrozenB3Model",
        "v5_p0_c1_model_c3",
    )

    test_dataset = ContractDataset(
        root=root,
        split="test",
        contract_dir=a2_dir,
        mask_audit_dir=a2_1_dir,
        window=32,
        stride=8,
        active_only=False,
        feature_variant="PRIMARY58",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    sample = test_dataset[0]
    if tuple(sample["x"].shape) != (16, 58, 32):
        raise RuntimeError(
            f"test sample x shape={tuple(sample['x'].shape)}"
        )
    if tuple(sample["physical_port_mask"].shape) != (16, 10):
        raise RuntimeError(
            "test physical_port_mask shape="
            f"{tuple(sample['physical_port_mask'].shape)}"
        )
    if sample["physical_port_mask"].dtype != torch.bool:
        raise RuntimeError("test physical_port_mask is not Boolean")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = FrozenB3Model().to(device)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    if parameter_count != 43208:
        raise RuntimeError(
            f"parameter_count={parameter_count}, expected 43208"
        )

    checkpoint = torch.load(
        paths["c2_checkpoint"],
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("seed") != 117:
        raise RuntimeError("checkpoint seed changed")
    if checkpoint.get("epoch") != 101:
        raise RuntimeError("checkpoint epoch changed")
    if checkpoint.get("parameter_count") != 43208:
        raise RuntimeError("checkpoint parameter count changed")

    model.load_state_dict(checkpoint["model_state_dict"])

    print("===== V5 P0-C3 ONE-SHOT LOCKED TEST EVALUATION =====")
    print("device:", device)
    print("test_window_count:", len(test_dataset))
    print("selected_seed: 117")
    print("selected_epoch: 101")
    print("parameter_count: 43208")
    print("frozen_thresholds:", thresholds)
    print(
        "checkpoint_sha256:",
        sha256_file(paths["c2_checkpoint"]),
    )
    print("training_performed: false")
    print("checkpoint_selection_performed: false")
    print("threshold_calibration_performed: false")
    print("test_evaluation_count: 1")

    metrics, predictions = evaluate_once(
        model,
        test_loader,
        device,
        thresholds,
    )

    predictions_path = (
        model_dir / "locked_test_predictions_and_targets.pt"
    )
    torch.save(
        {
            "stage": STAGE,
            "checkpoint_sha256": sha256_file(
                paths["c2_checkpoint"]
            ),
            "threshold_manifest_file_sha256": sha256_file(
                paths["c2_manifest"]
            ),
            "frozen_thresholds": thresholds,
            "predictions_and_targets": predictions,
            "test_evaluation_count": 1,
        },
        predictions_path,
    )

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "decision": "FINAL_P0_TEST_RESULT_RECORDED",
        "architecture": {
            "name": "B3_CAUSAL_DEPTHWISE_SEPARABLE_CONV1D_ONLY",
            "parameter_count": 43208,
            "architecture_contract_sha256": (
                documents["b8_contract"][
                    "architecture_contract_sha256"
                ]
            ),
        },
        "checkpoint": {
            "seed": 117,
            "epoch": 101,
            "path": str(paths["c2_checkpoint"]),
            "sha256": sha256_file(paths["c2_checkpoint"]),
        },
        "frozen_thresholds": thresholds,
        "test_metrics": metrics,
        "test_window_count": len(test_dataset),
        "test_forward_batch_count": metrics["forward_batch_count"],
        "artifacts": {
            "test_predictions_and_targets": [
                str(predictions_path),
                sha256_file(predictions_path),
            ],
        },
        "provenance": {
            "b8_report_sha256": sha256_file(paths["b8_report"]),
            "b8_lock_sha256": sha256_file(paths["b8_lock"]),
            "b8_contract_file_sha256": sha256_file(
                paths["b8_contract"]
            ),
            "c0_report_sha256": sha256_file(paths["c0_report"]),
            "c0_lock_sha256": sha256_file(paths["c0_lock"]),
            "c0_protocol_file_sha256": sha256_file(
                paths["c0_protocol"]
            ),
            "c1_report_sha256": sha256_file(paths["c1_report"]),
            "c1_lock_sha256": sha256_file(paths["c1_lock"]),
            "c2_report_sha256": sha256_file(paths["c2_report"]),
            "c2_lock_sha256": sha256_file(paths["c2_lock"]),
            "c2_threshold_manifest_file_sha256": sha256_file(
                paths["c2_manifest"]
            ),
            "c2_checkpoint_sha256": sha256_file(
                paths["c2_checkpoint"]
            ),
            "wrapper_sha256": sha256_file(wrapper_path),
            "c1_script_sha256": sha256_file(c1_script_path),
            "c3_script_sha256": sha256_file(Path(__file__)),
        },
        "training_performed": False,
        "validation_split_constructed": False,
        "checkpoint_selection_performed": False,
        "threshold_sweep_performed": False,
        "threshold_calibration_performed": False,
        "test_split_constructed": True,
        "test_split_accessed": True,
        "test_evaluation_count": 1,
        "retraining_after_test": False,
        "failures": failures,
        "warnings": warnings,
        "next_stage": "V5_P0_FINAL_RESULTS_ARCHIVE_AND_P1_HANDOFF",
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    lock = {
        "status": COMPLETE,
        "decision": "FINAL_P0_TEST_RESULT_RECORDED",
        "report_sha256": sha256_file(report_path),
        "checkpoint_sha256": sha256_file(paths["c2_checkpoint"]),
        "threshold_manifest_file_sha256": sha256_file(
            paths["c2_manifest"]
        ),
        "test_predictions_sha256": sha256_file(predictions_path),
        "frozen_thresholds": thresholds,
        "test_evaluation_count": 1,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "threshold_calibration_performed": False,
        "test_split_accessed": True,
        "c3_script_sha256": sha256_file(Path(__file__)),
        "next_stage": "V5_P0_FINAL_RESULTS_ARCHIVE_AND_P1_HANDOFF",
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    graph = metrics["graph"]
    roles = metrics["roles"]

    print("\n===== V5 P0-C3 FINAL P0 TEST RESULT =====")
    print(
        "graph_balanced_accuracy:",
        f"{graph['balanced_accuracy']:.4f}",
    )
    print("graph_accuracy:", f"{graph['accuracy']:.4f}")
    print("graph_precision:", f"{graph['precision']:.4f}")
    print("graph_recall:", f"{graph['recall']:.4f}")
    print("graph_f1:", f"{graph['f1']:.4f}")
    print("graph_fpr:", f"{graph['fpr']:.4f}")
    print(
        "graph_confusion:",
        {
            "tn": graph["tn"],
            "fp": graph["fp"],
            "fn": graph["fn"],
            "tp": graph["tp"],
        },
    )
    print(
        "count_macro_f1:",
        f"{metrics['count']['macro_f1']:.4f}",
    )
    print(
        "source_f1_attack:",
        f"{roles['source']['attack_windows']['node_f1']:.4f}",
    )
    print(
        "transit_f1_attack:",
        f"{roles['transit']['attack_windows']['node_f1']:.4f}",
    )
    print(
        "victim_f1_attack:",
        f"{roles['victim']['attack_windows']['node_f1']:.4f}",
    )
    print(
        "path_f1_attack:",
        f"{roles['path']['attack_windows']['node_f1']:.4f}",
    )
    print(
        "source_exact_attack:",
        f"{roles['source']['attack_windows']['exact_set']:.4f}",
    )
    print(
        "victim_exact_attack:",
        f"{roles['victim']['attack_windows']['exact_set']:.4f}",
    )
    print(
        "all_tasks_exact:",
        f"{metrics['all_tasks_exact']:.4f}",
    )
    print("test_window_count:", len(test_dataset))
    print("test_evaluation_count: 1")
    print("training_performed: false")
    print("checkpoint_selection_performed: false")
    print("threshold_calibration_performed: false")
    print("test_split_accessed: true")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    print(
        "next_stage: V5_P0_FINAL_RESULTS_ARCHIVE_AND_P1_HANDOFF"
    )
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
