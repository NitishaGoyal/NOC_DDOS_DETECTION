from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


STAGE = "V5_P3_D1_IMMUTABLE_TRANCHE_A_VALIDATION_LOGIT_EXPORT"
CAMPAIGN_LABEL = "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic"
SEED = 107
EXPECTED_ITEMS = 13_863
EXPECTED_PARAMETER_COUNT = 60_553
METRIC_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-dir", required=True)
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


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else ["dataset_index"]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def import_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while (
            end < len(values)
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def binary_auroc(scores: np.ndarray, targets: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    positive = targets == 1
    negative = targets == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = average_ranks(scores)
    rank_sum_positive = float(ranks[positive].sum())
    return (
        rank_sum_positive
        - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def binary_average_precision(
    scores: np.ndarray,
    targets: np.ndarray,
) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    positive_count = int((targets == 1).sum())
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered_targets = targets[order]
    cumulative_positive = np.cumsum(ordered_targets == 1)
    positions = np.arange(1, len(targets) + 1)
    precision_at_k = cumulative_positive / positions
    return float(
        precision_at_k[ordered_targets == 1].sum()
        / positive_count
    )


def binary_threshold_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    ).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(np.sum((predictions == 1) & (targets == 1)))
    fp = int(np.sum((predictions == 1) & (targets == 0)))
    fn = int(np.sum((predictions == 0) & (targets == 1)))
    tn = int(np.sum((predictions == 0) & (targets == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def multiclass_macro_f1(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    values = []
    for class_value in (1, 2, 3, 4):
        prediction_positive = predictions == class_value
        target_positive = targets == class_value
        tp = int(np.sum(prediction_positive & target_positive))
        fp = int(np.sum(prediction_positive & ~target_positive))
        fn = int(np.sum(~prediction_positive & target_positive))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return float(np.mean(values))


def exact_active(
    probabilities: np.ndarray,
    targets: np.ndarray,
    active_mask: np.ndarray,
) -> float:
    predictions = (
        np.asarray(probabilities) >= 0.5
    ).astype(np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    return float(
        np.mean(
            np.all(
                predictions[active_mask] == targets[active_mask],
                axis=1,
            )
        )
    )


def scalar_batch_values(
    batch: dict[str, Any],
    key: str,
    batch_size: int,
    fallback_start: int,
) -> list[Any]:
    if key not in batch:
        if key == "dataset_index":
            return list(range(fallback_start, fallback_start + batch_size))
        return [""] * batch_size

    value = batch[key]
    if isinstance(value, torch.Tensor):
        flattened = value.detach().cpu().reshape(batch_size, -1)
        if flattened.shape[1] == 1:
            return [item.item() for item in flattened[:, 0]]
        return [json.dumps(row.tolist()) for row in flattened]
    if isinstance(value, np.ndarray):
        array = value.reshape(batch_size, -1)
        if array.shape[1] == 1:
            return [item.item() for item in array[:, 0]]
        return [json.dumps(row.tolist()) for row in array]
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size:
            return [str(item) for item in value]
    return [str(value)] * batch_size


def main() -> int:
    args = parse_args()
    set_seed(SEED)

    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    data_root = data_link.resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    export_path = output_dir / (
        "V5_P3_D1_TRANCHE_A_VALIDATION_LOGITS_AND_LABELS.npz"
    )
    metadata_path = output_dir / (
        "V5_P3_D1_TRANCHE_A_VALIDATION_ROW_METADATA.csv"
    )
    manifest_path = output_dir / (
        "V5_P3_D1_TRANCHE_A_VALIDATION_EXPORT_MANIFEST.json"
    )
    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"
    hold_path = output_dir / f"{STAGE}_HOLD"

    d0_dir = repo / "reports/v5/p3_d0_frozen_decoder_transfer_preflight"
    d0_report_path = d0_dir / (
        "V5_P3_D0_FROZEN_DECODER_TRANSFER_PREFLIGHT_REPORT.json"
    )
    d0_lock_path = d0_dir / (
        "V5_P3_D0_FROZEN_DECODER_TRANSFER_PREFLIGHT_LOCK.json"
    )
    a5_dir = repo / "reports/v5/p3_a5_tranche_a_review_and_b_handover_readiness"
    a5_metrics_path = a5_dir / (
        "V5_P3_A5_STABLE_BEST_CHECKPOINT_VALIDATION_METRICS.json"
    )
    a4_dir = repo / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107"
    checkpoint_path = a4_dir / (
        "V5_P3_A4_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_SEED107_BEST.pt"
    )

    wrapper_path = repo / "src/data/v5_p3_tranche_a_guarded_loader.py"
    b3_path = repo / "src/models/v5_p2_b3_conv1d_only_count4.py"
    canonical_path = repo / "src/models/v5_p2_task_d_full_multitask_count4.py"
    dynamic_path = repo / "src/models/v6_p0_dynamic70_task_d_full_multitask_count4.py"

    required = [
        d0_report_path,
        d0_lock_path,
        a5_metrics_path,
        checkpoint_path,
        wrapper_path,
        b3_path,
        canonical_path,
        dynamic_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")
    if not data_link.is_symlink():
        raise RuntimeError(
            f"dataset path is not the canonical symlink: {data_link}"
        )

    d0_report = json.loads(d0_report_path.read_text(encoding="utf-8"))
    d0_lock = json.loads(d0_lock_path.read_text(encoding="utf-8"))
    a5_metrics = json.loads(a5_metrics_path.read_text(encoding="utf-8"))

    if d0_report.get("status") != "PASS":
        raise RuntimeError("D0 report is not PASS")
    if not d0_report.get("decision", {}).get(
        "immutable_validation_logit_export_authorized"
    ):
        raise RuntimeError("D0 did not authorize D1 export")
    if d0_lock.get("report_sha256") != sha256_file(d0_report_path):
        raise RuntimeError("D0 report/lock SHA mismatch")
    if d0_report.get("sealed_test", {}).get("test_tensor_loaded"):
        raise RuntimeError("D0 reports sealed-test access")

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
                    f"D1 sealed-test deserialization guard blocked {text}"
                )
        return original_torch_load(file, *load_args, **load_kwargs)

    torch.load = guarded_torch_load

    wrapper_mod = import_source(wrapper_path, "_v5_p3_d1_guarded_loader")
    b3_mod = import_source(b3_path, "_v5_p3_d1_b3")
    canonical_mod = import_source(canonical_path, "_v5_p3_d1_canonical")
    dynamic_mod = import_source(dynamic_path, "_v5_p3_d1_dynamic70")

    GuardedDataset = wrapper_mod.GuardedV5P3TrancheAPreliminaryDataset
    SealedTestAccessError = wrapper_mod.SealedTestAccessError
    try:
        GuardedDataset(data_root, "test")
    except SealedTestAccessError:
        test_negative_check = True
    else:
        test_negative_check = False
    if not test_negative_check:
        raise RuntimeError("guarded loader failed to reject A_test")

    validation_dataset = GuardedDataset(
        data_root,
        "validation",
        active_only=False,
    )
    if len(validation_dataset) != EXPECTED_ITEMS:
        raise RuntimeError(
            f"validation length={len(validation_dataset)}, "
            f"expected={EXPECTED_ITEMS}"
        )

    loader = DataLoader(
        validation_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    first_sample = validation_dataset[0]
    edge_index = torch.as_tensor(
        first_sample["edge_index"],
        dtype=torch.long,
    )
    physical_mask_template = torch.as_tensor(
        first_sample["physical_port_mask"],
        dtype=torch.uint8,
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
            f"parameter count={parameter_count}, "
            f"expected={EXPECTED_PARAMETER_COUNT}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint.get("epoch", -1)) != 14:
        raise RuntimeError(
            f"checkpoint epoch={checkpoint.get('epoch')}, expected=14"
        )
    if int(checkpoint.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("checkpoint parameter-count contract changed")
    if checkpoint.get("threshold_tuning_performed") is not False:
        raise RuntimeError("checkpoint unexpectedly reports threshold tuning")
    if checkpoint.get("test_tensor_loaded") is not False:
        raise RuntimeError("checkpoint reports sealed-test access")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    arrays: dict[str, np.ndarray] = {
        "dataset_index": np.arange(EXPECTED_ITEMS, dtype=np.int32),
        "attack_logits": np.empty((EXPECTED_ITEMS,), dtype=np.float32),
        "count_logits": np.empty((EXPECTED_ITEMS, 4), dtype=np.float32),
        "source_logits": np.empty((EXPECTED_ITEMS, 16), dtype=np.float32),
        "transit_logits": np.empty((EXPECTED_ITEMS, 16), dtype=np.float32),
        "victim_logits": np.empty((EXPECTED_ITEMS, 16), dtype=np.float32),
        "path_logits": np.empty((EXPECTED_ITEMS, 16), dtype=np.float32),
        "y_attack": np.empty((EXPECTED_ITEMS,), dtype=np.uint8),
        "y_attacker_count": np.empty((EXPECTED_ITEMS,), dtype=np.int8),
        "y_source": np.empty((EXPECTED_ITEMS, 16), dtype=np.uint8),
        "y_transit": np.empty((EXPECTED_ITEMS, 16), dtype=np.uint8),
        "y_victim": np.empty((EXPECTED_ITEMS, 16), dtype=np.uint8),
        "y_attack_path": np.empty((EXPECTED_ITEMS, 16), dtype=np.uint8),
        "edge_index": edge_index.cpu().numpy().astype(np.int64),
        "physical_port_mask_template": (
            physical_mask_template.cpu().numpy().astype(np.uint8)
        ),
    }

    optional_specs = {
        "epoch_id": (np.int32, (EXPECTED_ITEMS,)),
        "epoch_tick": (np.int64, (EXPECTED_ITEMS,)),
        "window_start": (np.int32, (EXPECTED_ITEMS,)),
        "window_target": (np.int32, (EXPECTED_ITEMS,)),
        "role_mask": (np.uint8, (EXPECTED_ITEMS, 16)),
        "source_target_router": (np.int8, (EXPECTED_ITEMS, 16)),
    }
    optional_present = {
        key: key in first_sample for key in optional_specs
    }
    for key, present in optional_present.items():
        if present:
            dtype, shape = optional_specs[key]
            arrays[key] = np.empty(shape, dtype=dtype)

    metadata_rows: list[dict[str, Any]] = []
    offset = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_number, batch in enumerate(loader, start=1):
            x = batch["x"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=(device.type == "cuda"),
            )
            mask = batch["physical_port_mask"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=(device.type == "cuda"),
            )
            outputs = model(x, mask)
            batch_size = int(x.shape[0])
            stop = offset + batch_size

            expected_shapes = {
                "attack_logits": (batch_size,),
                "count_logits": (batch_size, 4),
                "source_logits": (batch_size, 16),
                "transit_logits": (batch_size, 16),
                "victim_logits": (batch_size, 16),
                "path_logits": (batch_size, 16),
            }
            actual_shapes = {
                key: tuple(value.shape)
                for key, value in outputs.items()
            }
            if actual_shapes != expected_shapes:
                raise RuntimeError(
                    f"output shapes={actual_shapes}, "
                    f"expected={expected_shapes}"
                )

            for key in (
                "attack_logits",
                "count_logits",
                "source_logits",
                "transit_logits",
                "victim_logits",
                "path_logits",
            ):
                arrays[key][offset:stop] = (
                    outputs[key].detach().cpu().numpy().astype(np.float32)
                )

            arrays["y_attack"][offset:stop] = (
                batch["y_attack"].detach().cpu().numpy()
                .reshape(-1).astype(np.uint8)
            )
            arrays["y_attacker_count"][offset:stop] = (
                batch["y_attacker_count"].detach().cpu().numpy()
                .reshape(-1).astype(np.int8)
            )
            for destination, source in (
                ("y_source", "y_source"),
                ("y_transit", "y_transit"),
                ("y_victim", "y_victim"),
                ("y_attack_path", "y_attack_path"),
            ):
                arrays[destination][offset:stop] = (
                    batch[source].detach().cpu().numpy().astype(np.uint8)
                )

            if "edge_index" in batch:
                batched_edge = (
                    batch["edge_index"].detach().cpu().numpy()
                )
                expected_edge = arrays["edge_index"]
                if not np.all(
                    batched_edge
                    == expected_edge.reshape(1, 2, 48)
                ):
                    raise RuntimeError("edge_index varies within validation")
            if "physical_port_mask" in batch:
                batched_mask = (
                    batch["physical_port_mask"].detach().cpu().numpy()
                )
                expected_mask = arrays["physical_port_mask_template"]
                if not np.all(
                    batched_mask
                    == expected_mask.reshape(1, 16, 10)
                ):
                    raise RuntimeError(
                        "physical_port_mask varies within validation"
                    )

            for key, present in optional_present.items():
                if not present:
                    continue
                value = batch[key].detach().cpu().numpy()
                arrays[key][offset:stop] = value.astype(
                    arrays[key].dtype
                )

            pair_values = scalar_batch_values(
                batch,
                "pair_id",
                batch_size,
                offset,
            )
            run_values = scalar_batch_values(
                batch,
                "run_id",
                batch_size,
                offset,
            )
            window_values = scalar_batch_values(
                batch,
                "window_start",
                batch_size,
                offset,
            )
            epoch_values = scalar_batch_values(
                batch,
                "epoch_id",
                batch_size,
                offset,
            )

            for local_index in range(batch_size):
                metadata_rows.append({
                    "dataset_index": offset + local_index,
                    "pair_id": pair_values[local_index],
                    "run_id": run_values[local_index],
                    "window_start": window_values[local_index],
                    "epoch_id": epoch_values[local_index],
                })

            offset = stop
            if batch_number % 20 == 0 or offset == EXPECTED_ITEMS:
                print(
                    f"validation_batches_exported="
                    f"{batch_number}/{len(loader)}; "
                    f"items={offset}/{EXPECTED_ITEMS}"
                )

    if offset != EXPECTED_ITEMS:
        raise RuntimeError(
            f"exported items={offset}, expected={EXPECTED_ITEMS}"
        )
    if len(metadata_rows) != EXPECTED_ITEMS:
        raise RuntimeError("metadata row count mismatch")

    array_finiteness = {
        key: bool(np.isfinite(value).all())
        for key, value in arrays.items()
        if key.endswith("_logits")
    }
    if not all(array_finiteness.values()):
        raise RuntimeError(
            f"non-finite exported logits: {array_finiteness}"
        )

    graph_targets = arrays["y_attack"].astype(np.int64)
    active_mask = graph_targets == 1
    graph_probabilities = torch.sigmoid(
        torch.from_numpy(arrays["attack_logits"])
    ).numpy()
    graph_metrics = binary_threshold_metrics(
        graph_probabilities,
        graph_targets,
    )
    graph_auroc = binary_auroc(
        arrays["attack_logits"],
        graph_targets,
    )
    graph_ap = binary_average_precision(
        arrays["attack_logits"],
        graph_targets,
    )

    count_targets = arrays["y_attacker_count"][active_mask].astype(
        np.int64
    )
    count_predictions = (
        np.argmax(arrays["count_logits"][active_mask], axis=1) + 1
    )
    count_macro_f1 = multiclass_macro_f1(
        count_predictions,
        count_targets,
    )

    role_metrics: dict[str, dict[str, float]] = {}
    for role, label_key in (
        ("source", "y_source"),
        ("transit", "y_transit"),
        ("victim", "y_victim"),
        ("path", "y_attack_path"),
    ):
        role_logits = arrays[f"{role}_logits"]
        role_targets = arrays[label_key].astype(np.int64)
        role_probabilities = torch.sigmoid(
            torch.from_numpy(role_logits)
        ).numpy()
        role_metrics[role] = {
            "average_precision": binary_average_precision(
                role_logits.reshape(-1),
                role_targets.reshape(-1),
            ),
            "exact_active_at_0p5": exact_active(
                role_probabilities,
                role_targets,
                active_mask,
            ),
        }

    selection_components = {
        "graph_auroc": graph_auroc,
        "graph_average_precision": graph_ap,
        "count_macro_f1": count_macro_f1,
        "source_average_precision": role_metrics["source"][
            "average_precision"
        ],
        "transit_average_precision": role_metrics["transit"][
            "average_precision"
        ],
        "victim_average_precision": role_metrics["victim"][
            "average_precision"
        ],
        "path_average_precision": role_metrics["path"][
            "average_precision"
        ],
    }
    selection_score = float(
        np.mean(list(selection_components.values()))
    )

    comparisons = {
        "selection_score": (
            selection_score,
            float(a5_metrics["selection_score"]),
        ),
        "graph_auroc": (
            graph_auroc,
            float(a5_metrics["graph"]["auroc_raw_logits"]),
        ),
        "graph_average_precision": (
            graph_ap,
            float(
                a5_metrics["graph"][
                    "average_precision_raw_logits"
                ]
            ),
        ),
        "graph_f1_at_0p5": (
            graph_metrics["f1"],
            float(a5_metrics["graph"]["f1"]),
        ),
        "graph_fpr_at_0p5": (
            graph_metrics["fpr"],
            float(a5_metrics["graph"]["fpr"]),
        ),
        "count_macro_f1": (
            count_macro_f1,
            float(a5_metrics["count_active"]["macro_f1"]),
        ),
    }
    for role in ("source", "transit", "victim", "path"):
        comparisons[f"{role}_average_precision"] = (
            role_metrics[role]["average_precision"],
            float(
                a5_metrics["roles"][role][
                    "average_precision_raw_logits"
                ]
            ),
        )
        comparisons[f"{role}_exact_active"] = (
            role_metrics[role]["exact_active_at_0p5"],
            float(a5_metrics["roles"][role]["exact_active"]),
        )

    metric_deltas = {
        key: abs(exported - reference)
        for key, (exported, reference) in comparisons.items()
    }
    maximum_metric_delta = max(metric_deltas.values())
    metric_consistency_pass = (
        maximum_metric_delta <= METRIC_TOLERANCE
    )
    if not metric_consistency_pass:
        raise RuntimeError(
            "export metrics differ from A5 stable replay: "
            f"max_delta={maximum_metric_delta}; "
            f"deltas={metric_deltas}"
        )

    atomic_npz(export_path, arrays)
    atomic_csv(metadata_path, metadata_rows)

    array_manifest = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": (
                bool(np.isfinite(value).all())
                if np.issubdtype(value.dtype, np.number)
                else None
            ),
        }
        for key, value in arrays.items()
    }
    manifest = {
        "stage": STAGE,
        "status": "FROZEN_BY_HASH",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "split": "validation",
        "row_order": "guarded_validation_dataset_index_ascending",
        "items": EXPECTED_ITEMS,
        "model_checkpoint_epoch": 14,
        "parameter_count": parameter_count,
        "input_features": 70,
        "arrays": array_manifest,
        "export_path": str(export_path),
        "export_sha256": sha256_file(export_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "metric_replay": {
            "selection_score": selection_score,
            "selection_components": selection_components,
            "graph_threshold_0p5": graph_metrics,
            "count_macro_f1": count_macro_f1,
            "roles": role_metrics,
            "maximum_absolute_delta_vs_A5": maximum_metric_delta,
            "tolerance": METRIC_TOLERANCE,
            "consistency_pass": metric_consistency_pass,
        },
        "decoder_executed": False,
        "threshold_tuning_performed": False,
        "test_accessed": False,
    }
    atomic_json(manifest_path, manifest)

    # Reopen the frozen export without pickle and recheck names/shapes.
    with np.load(export_path, allow_pickle=False) as reopened:
        reopened_names = sorted(reopened.files)
        expected_names = sorted(arrays)
        if reopened_names != expected_names:
            raise RuntimeError(
                f"reopened array names={reopened_names}, "
                f"expected={expected_names}"
            )
        for key, expected in arrays.items():
            actual = reopened[key]
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise RuntimeError(
                    f"reopened {key}: shape={actual.shape}, "
                    f"dtype={actual.dtype}; expected shape={expected.shape}, "
                    f"dtype={expected.dtype}"
                )
            if key.endswith("_logits") and not np.array_equal(
                actual,
                expected,
            ):
                raise RuntimeError(
                    f"reopened logits differ for {key}"
                )

    test_loaded_paths = [
        path for path in loaded_paths if "/runs/test/" in path
    ]
    if test_loaded_paths:
        raise RuntimeError(
            f"sealed-test tensors were deserialized: {test_loaded_paths}"
        )

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "scope": (
            "Single deterministic replay of the A4 epoch-14 model on "
            "A_validation, producing an immutable-by-hash raw-logit and "
            "label export for decoder evaluation."
        ),
        "export": {
            "path": str(export_path),
            "sha256": sha256_file(export_path),
            "size_bytes": export_path.stat().st_size,
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "items": EXPECTED_ITEMS,
            "arrays": array_manifest,
            "pickle_required": False,
            "immutable_by_hash": True,
        },
        "model": {
            "class": model.__class__.__name__,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "parameter_count": parameter_count,
            "input_features": 70,
        },
        "metric_certification": {
            "selection_score": selection_score,
            "selection_components": selection_components,
            "graph_threshold_0p5": graph_metrics,
            "count_macro_f1": count_macro_f1,
            "roles": role_metrics,
            "maximum_absolute_delta_vs_A5": maximum_metric_delta,
            "tolerance": METRIC_TOLERANCE,
            "consistency_pass": metric_consistency_pass,
        },
        "data": {
            "split": "validation",
            "items": len(validation_dataset),
            "batches": len(loader),
            "ordering": "ascending_dataset_index",
            "optional_metadata_arrays_present": optional_present,
            "test_dataset_instantiated": False,
            "test_tensor_loaded": False,
        },
        "runtime": {
            "device": str(device),
            "elapsed_seconds": time.time() - start_time,
        },
        "decision": {
            "immutable_validation_export_complete": True,
            "raw_and_A0_evaluation_authorized": True,
            "A1_exact_full_evaluation_authorized": False,
            "A1_exact_resumable_preflight_authorized": False,
            "sealed_test_evaluation_authorized": False,
            "next_stage": (
                "V5_P3_D2_RAW_AND_FROZEN_A0_"
                "TRANCHE_A_VALIDATION_EVALUATION"
            ),
        },
        "sealed_test": {
            "guard_negative_check": test_negative_check,
            "test_dataset_instantiated": False,
            "test_length_computed": False,
            "test_tensor_loaded": False,
            "test_loaded_paths": test_loaded_paths,
            "evaluation_authorized": False,
        },
        "provenance": {
            "d0_report_sha256": sha256_file(d0_report_path),
            "d0_lock_sha256": sha256_file(d0_lock_path),
            "a5_stable_metrics_sha256": sha256_file(a5_metrics_path),
            "a4_checkpoint_sha256": sha256_file(checkpoint_path),
            "guarded_loader_sha256": sha256_file(wrapper_path),
            "dynamic70_model_sha256": sha256_file(dynamic_path),
            "installed_script_sha256": sha256_file(installed_script),
        },
        "model_trained": False,
        "decoder_executed": False,
        "threshold_tuning_performed": False,
        "certified_dataset_modified": False,
        "generalization_claim_authorized": False,
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(report_path),
        "export_sha256": sha256_file(export_path),
        "metadata_sha256": sha256_file(metadata_path),
        "manifest_sha256": sha256_file(manifest_path),
        "d0_report_sha256": sha256_file(d0_report_path),
        "a4_checkpoint_sha256": sha256_file(checkpoint_path),
        "items": EXPECTED_ITEMS,
        "parameter_count": parameter_count,
        "input_features": 70,
        "metric_maximum_absolute_delta_vs_A5": maximum_metric_delta,
        "decoder_executed": False,
        "test_tensor_loaded": False,
        "certified_dataset_modified": False,
    }
    atomic_json(lock_path, lock)

    complete_path.write_text(
        f"{STAGE}_COMPLETE\n",
        encoding="utf-8",
    )
    if hold_path.exists():
        hold_path.unlink()

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"campaign_label={CAMPAIGN_LABEL}")
    print(f"validation_items={EXPECTED_ITEMS}")
    print(f"validation_batches={len(loader)}")
    print("row_order=ascending_dataset_index")
    print(f"model_checkpoint_epoch={checkpoint['epoch']}")
    print(f"parameter_count={parameter_count}")
    print("input_features=70")
    print(f"export_size_bytes={export_path.stat().st_size}")
    print(f"export_sha256={sha256_file(export_path)}")
    print(
        "metric_maximum_absolute_delta_vs_A5="
        f"{maximum_metric_delta:.10f}"
    )
    print(f"metric_consistency_pass={metric_consistency_pass}")
    print("decoder_executed=false")
    print("threshold_tuning_performed=false")
    print("test_dataset_instantiated=false")
    print("test_length_computed=false")
    print("test_tensor_loaded=false")
    print("raw_and_A0_evaluation_authorized=true")
    print("A1_exact_full_evaluation_authorized=false")
    print("sealed_test_evaluation_authorized=false")
    print(
        "next_stage="
        "V5_P3_D2_RAW_AND_FROZEN_A0_"
        "TRANCHE_A_VALIDATION_EVALUATION"
    )
    print(f"export={export_path}")
    print(f"metadata={metadata_path}")
    print(f"manifest={manifest_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
