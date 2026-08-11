from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


STAGE = (
    "V5_P3_D8_EPLR_V1_DETERMINISTIC_IMPLEMENTATION_"
    "AND_A_TRAIN_ONLY_CALIBRATION"
)
SEED = 107
PARAMETERS = 60_553
CHECKPOINT_EPOCH = 14
QUOTA_PER_TRUE_K = 256
CALIBRATION_MAX_ITEMS = 4 * QUOTA_PER_TRUE_K

WEIGHT_CANDIDATES = (
    (0.25, 1.0),
    (0.5, 1.0),
    (1.0, 1.0),
    (2.0, 1.0),
    (4.0, 1.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-link", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def import_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def mask_from_array(values: np.ndarray) -> int:
    mask = 0
    for index, value in enumerate(
        np.asarray(values).reshape(-1).tolist()
    ):
        if int(value):
            mask |= 1 << index
    return mask


def exact_mask(mask: int, label: np.ndarray) -> bool:
    return mask == mask_from_array(label)


def select_train_items(dataset) -> tuple[list[int], dict[int, int]]:
    selected: list[int] = []
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for index in range(len(dataset)):
        sample = dataset[index]
        true_k = int(torch.as_tensor(sample["y_attacker_count"]).item())
        if true_k in counts and counts[true_k] < QUOTA_PER_TRUE_K:
            selected.append(index)
            counts[true_k] += 1
        if all(value >= QUOTA_PER_TRUE_K for value in counts.values()):
            break
    return selected, counts


def batched_inference(
    dataset,
    indices: list[int],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    outputs = {
        "dataset_index": [],
        "attack_logits": [],
        "count_logits": [],
        "source_logits": [],
        "transit_logits": [],
        "victim_logits": [],
        "path_logits": [],
        "y_count": [],
        "y_source": [],
        "y_transit": [],
        "y_victim": [],
        "y_path": [],
    }

    model.eval()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        samples = [dataset[index] for index in batch_indices]
        x = torch.stack(
            [
                torch.as_tensor(sample["x"], dtype=torch.float32)
                for sample in samples
            ]
        ).to(device)
        mask = torch.stack(
            [
                torch.as_tensor(
                    sample["physical_port_mask"],
                    dtype=torch.float32,
                )
                for sample in samples
            ]
        ).to(device)

        with torch.no_grad():
            prediction = model(x, mask)

        outputs["dataset_index"].append(
            np.asarray(batch_indices, dtype=np.int32)
        )
        outputs["attack_logits"].append(
            prediction["attack_logits"]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        for key in (
            "count_logits",
            "source_logits",
            "transit_logits",
            "victim_logits",
            "path_logits",
        ):
            outputs[key].append(
                prediction[key].detach().cpu().numpy()
            )
        outputs["y_count"].append(
            np.asarray(
                [
                    int(
                        torch.as_tensor(
                            sample["y_attacker_count"]
                        ).item()
                    )
                    for sample in samples
                ],
                dtype=np.int8,
            )
        )
        for output_key, sample_key in (
            ("y_source", "y_source"),
            ("y_transit", "y_transit"),
            ("y_victim", "y_victim"),
            ("y_path", "y_attack_path"),
        ):
            outputs[output_key].append(
                np.stack(
                    [
                        torch.as_tensor(sample[sample_key])
                        .cpu()
                        .numpy()
                        for sample in samples
                    ]
                ).astype(np.uint8)
            )

        print(
            f"train_inference_items={min(start + batch_size, len(indices))}/"
            f"{len(indices)}",
            flush=True,
        )

    return {
        key: np.concatenate(parts, axis=0)
        for key, parts in outputs.items()
    }


def main() -> int:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    data_root = data_link.resolve()
    out = Path(args.output_dir).expanduser().resolve()
    package_dir = Path(args.package_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    d7_dir = (
        repo / "reports/v5/p3_d7_eplr_v1_route_library_and_synthetic_tests"
    )
    d7_report_path = d7_dir / (
        "V5_P3_D7_EPLR_V1_ROUTE_LIBRARY_AND_"
        "SYNTHETIC_UNIT_TESTS_REPORT.json"
    )
    d7_lock_path = d7_dir / (
        "V5_P3_D7_EPLR_V1_ROUTE_LIBRARY_AND_"
        "SYNTHETIC_UNIT_TESTS_LOCK.json"
    )
    d6_dir = repo / "reports/v5/p3_d6_eplr_v1_design_freeze"
    d6_semantic_path = d6_dir / "V5_P3_EPLR_V1_SEMANTIC_CONTRACT.json"
    d6_objective_path = d6_dir / "V5_P3_EPLR_V1_OBJECTIVE_CONTRACT.json"

    loader_path = repo / "src/data/v5_p3_tranche_a_guarded_loader.py"
    b3_path = repo / "src/models/v5_p2_b3_conv1d_only_count4.py"
    canonical_model_path = (
        repo / "src/models/v5_p2_task_d_full_multitask_count4.py"
    )
    dynamic_model_path = (
        repo / "src/models/v6_p0_dynamic70_task_d_full_multitask_count4.py"
    )
    checkpoint_path = repo / (
        "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107/"
        "V5_P3_A4_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_SEED107_BEST.pt"
    )
    package_decoder_path = package_dir / "v5_p3_eplr_v1.py"
    installed_decoder_path = repo / "src/decoders/v5_p3_eplr_v1.py"
    installed_config_path = (
        repo / "src/decoders/v5_p3_eplr_v1_calibration.json"
    )

    required = [
        d7_report_path,
        d7_lock_path,
        d6_semantic_path,
        d6_objective_path,
        loader_path,
        b3_path,
        canonical_model_path,
        dynamic_model_path,
        checkpoint_path,
        package_decoder_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")
    if not data_link.is_symlink():
        raise RuntimeError(f"dataset link is not canonical symlink: {data_link}")

    d7_report = json.loads(d7_report_path.read_text(encoding="utf-8"))
    d7_lock = json.loads(d7_lock_path.read_text(encoding="utf-8"))
    if d7_report.get("status") != "PASS":
        raise RuntimeError("D7 is not PASS")
    if d7_lock.get("report_sha256") != sha256_file(d7_report_path):
        raise RuntimeError("D7 report/lock mismatch")
    if not d7_report["decision"].get("D8_authorized"):
        raise RuntimeError("D7 did not authorize D8")

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    loader_mod = import_source(loader_path, "_v5_p3_d8_loader")
    b3_mod = import_source(b3_path, "_v5_p3_d8_b3")
    canonical_mod = import_source(
        canonical_model_path,
        "_v5_p3_d8_canonical_model",
    )
    dynamic_mod = import_source(
        dynamic_model_path,
        "_v5_p3_d8_dynamic_model",
    )
    decoder_mod = import_source(
        package_decoder_path,
        "_v5_p3_d8_eplr_candidate",
    )

    Dataset = loader_mod.GuardedV5P3TrancheAPreliminaryDataset
    train_dataset = Dataset(
        data_root,
        "train",
        active_only=True,
    )

    indices, true_k_counts = select_train_items(train_dataset)
    if len(indices) < CALIBRATION_MAX_ITEMS:
        raise RuntimeError(
            f"insufficient calibration items: {len(indices)}; "
            f"counts={true_k_counts}"
        )

    first = train_dataset[indices[0]]
    edge_index = torch.as_tensor(
        first["edge_index"],
        dtype=torch.long,
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
    if parameter_count != PARAMETERS:
        raise RuntimeError(
            f"parameter_count={parameter_count}, expected={PARAMETERS}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint.get("epoch", -1)) != CHECKPOINT_EPOCH:
        raise RuntimeError("checkpoint epoch changed")
    if checkpoint.get("test_tensor_loaded") is not False:
        raise RuntimeError("checkpoint reports test access")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    calibration = batched_inference(
        train_dataset,
        indices,
        model,
        device,
    )

    graph_prob = stable_sigmoid(calibration["attack_logits"])
    source_prob = stable_sigmoid(calibration["source_logits"])
    transit_prob = stable_sigmoid(calibration["transit_logits"])
    victim_prob = stable_sigmoid(calibration["victim_logits"])
    path_prob = stable_sigmoid(calibration["path_logits"])
    predicted_k = (
        np.argmax(calibration["count_logits"], axis=1).astype(np.int8) + 1
    )

    candidate_rows = []
    selected_rows_by_candidate: dict[
        tuple[float, float], list[dict[str, Any]]
    ] = {
        candidate: [] for candidate in WEIGHT_CANDIDATES
    }

    raw_positive = 0
    preservation_feasible = 0
    for row in range(len(indices)):
        if graph_prob[row] < 0.5:
            continue
        raw_positive += 1
        raw_source_mask = decoder_mod.mask_from_probabilities(
            source_prob[row],
            0.5,
        )
        raw_victim_mask = decoder_mod.mask_from_probabilities(
            victim_prob[row],
            0.5,
        )
        k = int(predicted_k[row])

        candidates = decoder_mod.enumerate_endpoint_preserving_hypotheses(
            raw_source_mask,
            raw_victim_mask,
            k,
        )
        if not candidates:
            continue
        preservation_feasible += 1

        for transit_weight, path_weight in WEIGHT_CANDIDATES:
            config = decoder_mod.EPLRConfig(
                transit_weight=transit_weight,
                path_weight=path_weight,
            )
            chosen, support = (
                decoder_mod.choose_endpoint_preserving_hypothesis(
                    source_mask=raw_source_mask,
                    victim_mask=raw_victim_mask,
                    k=k,
                    transit_probabilities=transit_prob[row],
                    path_probabilities=path_prob[row],
                    config=config,
                )
            )
            selected_rows_by_candidate[
                (transit_weight, path_weight)
            ].append(
                {
                    "dataset_index": int(
                        calibration["dataset_index"][row]
                    ),
                    "predicted_k": k,
                    "true_k": int(calibration["y_count"][row]),
                    "support": float(support),
                    "source_exact": exact_mask(
                        chosen.source_mask,
                        calibration["y_source"][row],
                    ),
                    "victim_exact": exact_mask(
                        chosen.victim_mask,
                        calibration["y_victim"][row],
                    ),
                    "transit_exact": exact_mask(
                        chosen.transit_mask,
                        calibration["y_transit"][row],
                    ),
                    "path_exact": exact_mask(
                        chosen.path_mask,
                        calibration["y_path"][row],
                    ),
                    "joint_transit_path_exact": (
                        exact_mask(
                            chosen.transit_mask,
                            calibration["y_transit"][row],
                        )
                        and exact_mask(
                            chosen.path_mask,
                            calibration["y_path"][row],
                        )
                    ),
                }
            )

    if preservation_feasible < 128:
        raise RuntimeError(
            "too few exact-preservation train rows for calibration: "
            f"{preservation_feasible}"
        )

    for candidate in WEIGHT_CANDIDATES:
        rows = selected_rows_by_candidate[candidate]
        if not rows:
            raise RuntimeError(f"candidate has no rows: {candidate}")
        candidate_rows.append(
            {
                "transit_weight": candidate[0],
                "path_weight": candidate[1],
                "items": len(rows),
                "source_exact": float(
                    np.mean([row["source_exact"] for row in rows])
                ),
                "victim_exact": float(
                    np.mean([row["victim_exact"] for row in rows])
                ),
                "transit_exact": float(
                    np.mean([row["transit_exact"] for row in rows])
                ),
                "path_exact": float(
                    np.mean([row["path_exact"] for row in rows])
                ),
                "joint_transit_path_exact": float(
                    np.mean(
                        [
                            row["joint_transit_path_exact"]
                            for row in rows
                        ]
                    )
                ),
            }
        )

    # Train-only lexicographic selection:
    # joint transit/path exact, transit exact, path exact, then closest to 1:1.
    selected_candidate = max(
        candidate_rows,
        key=lambda row: (
            row["joint_transit_path_exact"],
            row["transit_exact"],
            row["path_exact"],
            -abs(np.log2(row["transit_weight"])),
        ),
    )
    selected_key = (
        float(selected_candidate["transit_weight"]),
        float(selected_candidate["path_weight"]),
    )
    selected_train_rows = selected_rows_by_candidate[selected_key]

    thresholds = []
    threshold_support_counts = {}
    for k in (1, 2, 3, 4):
        correct_scores = [
            row["support"]
            for row in selected_train_rows
            if row["predicted_k"] == k
            and row["joint_transit_path_exact"]
        ]
        if len(correct_scores) >= 8:
            threshold = float(np.quantile(correct_scores, 0.05))
        else:
            all_scores = [
                row["support"]
                for row in selected_train_rows
                if row["predicted_k"] == k
            ]
            if not all_scores:
                threshold = float("-inf")
            else:
                threshold = float(np.quantile(all_scores, 0.01))
        thresholds.append(threshold)
        threshold_support_counts[str(k)] = {
            "joint_correct_scores": len(correct_scores),
            "all_scores": sum(
                row["predicted_k"] == k
                for row in selected_train_rows
            ),
        }

    config_payload = {
        "stage": STAGE,
        "status": "FROZEN_A_TRAIN_ONLY",
        "transit_weight": selected_key[0],
        "path_weight": selected_key[1],
        "low_support_threshold_by_k": thresholds,
        "graph_threshold": 0.5,
        "endpoint_threshold": 0.5,
        "objective_freeze_tolerance": 1e-8,
        "integrality_tolerance": 1e-7,
        "calibration_split": "Tranche-A train only",
        "calibration_items_selected": len(indices),
        "raw_positive_items": raw_positive,
        "endpoint_preservation_feasible_items": preservation_feasible,
        "validation_accessed": False,
        "test_accessed": False,
    }
    config_path = out / "V5_P3_EPLR_V1_A_TRAIN_ONLY_CALIBRATION.json"
    atomic_json(config_path, config_payload)

    # Candidate production decoder regressions.
    config = decoder_mod.EPLRConfig.from_json(config_path)

    inactive = decoder_mod.decode_eplr_v1(
        graph_logit=-10.0,
        count_logits=[0.0, 1.0, 2.0, 3.0],
        source_logits=np.zeros(16),
        transit_logits=np.zeros(16),
        victim_logits=np.zeros(16),
        path_logits=np.zeros(16),
        config=config,
    )
    if (
        inactive.raw_graph_prediction != 0
        or inactive.effective_attacker_count != 0
        or inactive.decoded_path_bitmap != 0
        or inactive.decoder_status != decoder_mod.STATUS_INACTIVE
    ):
        raise RuntimeError("production inactive regression failed")

    route = decoder_mod.routes.build_route(0, 15)
    endpoint_logits = np.full(16, -8.0)
    endpoint_logits[0] = 8.0
    victim_logits = np.full(16, -8.0)
    victim_logits[15] = 8.0
    transit_logits = np.full(16, -8.0)
    path_logits = np.full(16, -8.0)
    for router in decoder_mod.indices_from_mask(route.transit_mask):
        transit_logits[router] = 8.0
    for router in decoder_mod.indices_from_mask(route.path_mask):
        path_logits[router] = 8.0

    legal_outputs = [
        decoder_mod.decode_eplr_v1(
            graph_logit=10.0,
            count_logits=[10.0, 0.0, 0.0, 0.0],
            source_logits=endpoint_logits,
            transit_logits=transit_logits,
            victim_logits=victim_logits,
            path_logits=path_logits,
            config=config,
        )
        for _ in range(3)
    ]
    if not all(output == legal_outputs[0] for output in legal_outputs[1:]):
        raise RuntimeError("production deterministic repeat failed")
    if legal_outputs[0].selected_route_ids != (route.route_id,):
        raise RuntimeError("production legal route regression failed")
    if legal_outputs[0].repair_solver_used:
        raise RuntimeError("legal endpoint case unexpectedly used repair")

    # Force one deterministic K1 repair: Raw predicts two sources while K=1.
    repair_source_logits = np.full(16, -8.0)
    repair_source_logits[0] = 8.0
    repair_source_logits[1] = 0.2
    repaired = decoder_mod.decode_eplr_v1(
        graph_logit=10.0,
        count_logits=[10.0, 0.0, 0.0, 0.0],
        source_logits=repair_source_logits,
        transit_logits=transit_logits,
        victim_logits=victim_logits,
        path_logits=path_logits,
        config=config,
    )
    if repaired.raw_graph_prediction != 1:
        raise RuntimeError("repair changed graph")
    if repaired.effective_attacker_count != 1:
        raise RuntimeError("repair changed count")
    if not repaired.repair_solver_used:
        raise RuntimeError("repair case did not use repair solver")
    if repaired.decoder_status not in (
        decoder_mod.STATUS_REPAIRED,
        decoder_mod.STATUS_LEGAL_LOW_SUPPORT,
    ):
        raise RuntimeError(
            f"unexpected repair status: {repaired.decoder_status}"
        )

    # Install only after calibration and regressions pass.
    for source, destination in (
        (package_decoder_path, installed_decoder_path),
        (config_path, installed_config_path),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            if sha256_file(destination) != sha256_file(source):
                raise RuntimeError(
                    f"existing installed file differs: {destination}"
                )
        else:
            temporary = destination.with_suffix(
                destination.suffix + ".tmp"
            )
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)

    calibration_export_path = out / (
        "V5_P3_D8_A_TRAIN_CALIBRATION_LOGITS_AND_LABELS.npz"
    )
    np.savez_compressed(calibration_export_path, **calibration)

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Install deterministic EPLR-V1 and calibrate transit/path role "
            "weights plus low-support diagnostics using Tranche-A train only."
        ),
        "model": {
            "checkpoint_epoch": CHECKPOINT_EPOCH,
            "parameter_count": parameter_count,
            "device": str(device),
        },
        "calibration_sample": {
            "active_only": True,
            "selected_items": len(indices),
            "quota_per_true_k": QUOTA_PER_TRUE_K,
            "true_k_counts": true_k_counts,
            "raw_positive_items": raw_positive,
            "endpoint_preservation_feasible_items": preservation_feasible,
        },
        "candidate_metrics": candidate_rows,
        "selected_calibration": {
            "transit_weight": selected_key[0],
            "path_weight": selected_key[1],
            "low_support_threshold_by_k": thresholds,
            "threshold_support_counts": threshold_support_counts,
            "selection_rule": (
                "train-only lexicographic joint transit/path exact, transit "
                "exact, path exact, then closest to 1:1"
            ),
        },
        "implementation_regression": {
            "inactive_pass": True,
            "legal_endpoint_repeat_count": 3,
            "legal_endpoint_deterministic": True,
            "repair_case_pass": True,
            "graph_count_immutability_pass": True,
        },
        "installed": {
            "decoder_path": str(installed_decoder_path),
            "decoder_sha256": sha256_file(installed_decoder_path),
            "config_path": str(installed_config_path),
            "config_sha256": sha256_file(installed_config_path),
        },
        "artifacts": {
            "calibration_config_path": str(config_path),
            "calibration_config_sha256": sha256_file(config_path),
            "train_calibration_export_path": str(
                calibration_export_path
            ),
            "train_calibration_export_sha256": sha256_file(
                calibration_export_path
            ),
        },
        "decision": {
            "production_decoder_installed": True,
            "A_train_only_calibration_complete": True,
            "D9_authorized": True,
            "validation_replay_authorized": True,
            "validation_replay_must_be_one_shot": True,
            "threshold_or_weight_retuning_after_D9": False,
            "sealed_test_evaluation_authorized": False,
            "next_stage": (
                "V5_P3_D9_EPLR_V1_ONE_SHOT_TRANCHE_A_"
                "VALIDATION_EXPLORATORY_REPLAY"
            ),
        },
        "governance": {
            "training_split_loaded": True,
            "validation_split_loaded": False,
            "test_split_loaded": False,
            "validation_metrics_observed": False,
            "test_metrics_observed": False,
        },
        "provenance": {
            "D7_report_sha256": sha256_file(d7_report_path),
            "D7_lock_sha256": sha256_file(d7_lock_path),
            "D6_semantic_sha256": sha256_file(d6_semantic_path),
            "D6_objective_sha256": sha256_file(d6_objective_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "candidate_decoder_sha256": sha256_file(
                package_decoder_path
            ),
            "installed_script_sha256": sha256_file(installed_script),
        },
    }

    report_path = out / f"{STAGE}_REPORT.json"
    lock_path = out / f"{STAGE}_LOCK.json"
    atomic_json(report_path, report)
    atomic_json(
        lock_path,
        {
            "stage": STAGE,
            "status": "PASS",
            "report_sha256": sha256_file(report_path),
            "decoder_sha256": sha256_file(installed_decoder_path),
            "config_sha256": sha256_file(installed_config_path),
            "train_calibration_export_sha256": sha256_file(
                calibration_export_path
            ),
            "selected_transit_weight": selected_key[0],
            "selected_path_weight": selected_key[1],
            "low_support_threshold_by_k": thresholds,
            "validation_split_loaded": False,
            "test_split_loaded": False,
            "D9_authorized": True,
        },
    )
    (
        out / f"{STAGE}_COMPLETE"
    ).write_text(f"{STAGE}_COMPLETE\n", encoding="utf-8")

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(f"calibration_items={len(indices)}")
    print(f"calibration_true_k_counts={true_k_counts}")
    print(f"raw_positive_items={raw_positive}")
    print(
        "endpoint_preservation_feasible_items="
        f"{preservation_feasible}"
    )
    print(f"selected_transit_weight={selected_key[0]}")
    print(f"selected_path_weight={selected_key[1]}")
    print(f"low_support_threshold_by_k={thresholds}")
    print("inactive_regression_pass=true")
    print("legal_endpoint_deterministic_pass=true")
    print("repair_regression_pass=true")
    print("graph_count_immutability_pass=true")
    print(f"installed_decoder={installed_decoder_path}")
    print(
        "installed_decoder_sha256="
        f"{sha256_file(installed_decoder_path)}"
    )
    print(f"installed_config={installed_config_path}")
    print(
        "installed_config_sha256="
        f"{sha256_file(installed_config_path)}"
    )
    print("training_split_loaded=true")
    print("validation_split_loaded=false")
    print("test_split_loaded=false")
    print("D9_authorized=true")
    print("validation_replay_authorized=true")
    print(
        "next_stage="
        "V5_P3_D9_EPLR_V1_ONE_SHOT_TRANCHE_A_"
        "VALIDATION_EXPLORATORY_REPLAY"
    )
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
