from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.util
import inspect
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


STAGE = "V5_P3_D0_FROZEN_DECODER_TRANSFER_PREFLIGHT"
CAMPAIGN_LABEL = "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic"
SEED = 107
EXPECTED_PARAMETER_COUNT = 60_553
EXPECTED_VALIDATION_ITEMS = 13_863
EXPECTED_A0_THRESHOLDS = {
    "graph": 0.47174675035328983,
    "source": 0.94960549299285935,
    "transit": 0.85703332488359107,
    "victim": 0.82601148026998117,
    "path": 0.8339347466090468,
}
EXPECTED_A1_MARGIN_THRESHOLD = 8.7205320882398425


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


def import_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def as_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return as_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            key: as_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def stable_sigmoid(logit: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(logit, dtype=torch.float64)
    return torch.sigmoid(tensor).cpu().numpy()


def walk_json(value: Any, path: str = "$"):
    """Yield every JSON node together with its structural path."""
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_json(item, f"{path}[{index}]")


def resolve_frozen_threshold_contract(
    document: dict[str, Any],
) -> tuple[dict[str, float], float, dict[str, Any]]:
    """Resolve the frozen P2 constants from any nested contract schema.

    The historical contract changed nesting across freezes. Certification is
    based on exact numeric agreement with the already known frozen constants,
    not on one particular JSON key layout.
    """
    required_roles = set(EXPECTED_A0_THRESHOLDS)
    threshold_candidates: list[dict[str, Any]] = []
    margin_candidates: list[dict[str, Any]] = []

    for path, value in walk_json(document):
        if isinstance(value, dict) and required_roles <= set(value):
            try:
                normalized = {
                    key: float(value[key])
                    for key in sorted(required_roles)
                }
            except (TypeError, ValueError):
                pass
            else:
                threshold_candidates.append({
                    "path": path,
                    "values": normalized,
                })

        if isinstance(value, dict):
            for key, item in value.items():
                normalized_key = str(key).strip().lower()
                if normalized_key not in {
                    "a1_margin_threshold",
                    "margin_threshold",
                }:
                    continue
                try:
                    numeric = float(item)
                except (TypeError, ValueError):
                    continue
                margin_candidates.append({
                    "path": f"{path}.{key}",
                    "value": numeric,
                })

    matching_thresholds = [
        candidate
        for candidate in threshold_candidates
        if all(
            candidate["values"][key] == expected
            for key, expected in EXPECTED_A0_THRESHOLDS.items()
        )
    ]
    matching_margins = [
        candidate
        for candidate in margin_candidates
        if candidate["value"] == EXPECTED_A1_MARGIN_THRESHOLD
    ]

    audit = {
        "threshold_candidates": threshold_candidates,
        "margin_candidates": margin_candidates,
        "matching_threshold_candidate_count": len(matching_thresholds),
        "matching_margin_candidate_count": len(matching_margins),
    }

    if not matching_thresholds or not matching_margins:
        raise RuntimeError(
            "could not resolve exact frozen P2 decoder constants; "
            f"audit={json.dumps(audit, sort_keys=True)}"
        )

    return (
        dict(EXPECTED_A0_THRESHOLDS),
        EXPECTED_A1_MARGIN_THRESHOLD,
        {
            **audit,
            "selected_threshold_path": matching_thresholds[0]["path"],
            "selected_margin_path": matching_margins[0]["path"],
        },
    )


def find_validation_probe_indices(dataset) -> list[int]:
    inactive: list[int] = []
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
            return inactive + [active_by_count[count] for count in (1, 2, 3, 4)]
    raise RuntimeError(
        "could not select decoder probes with four inactive examples "
        f"and K1-K4 active coverage; inactive={inactive}; "
        f"active_counts={sorted(active_by_count)}"
    )


def main() -> int:
    args = parse_args()
    set_seed(SEED)

    repo = Path(args.repo).expanduser().resolve()
    data_link = Path(args.data_link).expanduser()
    data_root = data_link.resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    installed_script = Path(args.installed_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    probe_path = output_dir / f"{STAGE}_PROBES.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"
    hold_path = output_dir / f"{STAGE}_HOLD"

    a5_dir = repo / "reports/v5/p3_a5_tranche_a_review_and_b_handover_readiness"
    a5_report_path = a5_dir / (
        "V5_P3_A5_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_"
        "REVIEW_AND_B_HANDOVER_READINESS_REPORT.json"
    )
    a5_lock_path = a5_dir / (
        "V5_P3_A5_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_"
        "REVIEW_AND_B_HANDOVER_READINESS_LOCK.json"
    )
    a4_dir = repo / "reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107"
    a4_checkpoint_path = a4_dir / (
        "V5_P3_A4_TRANCHE_A_PRELIMINARY_"
        "DIAGNOSTIC_SEED107_BEST.pt"
    )
    wrapper_path = repo / "src/data/v5_p3_tranche_a_guarded_loader.py"
    b3_path = repo / "src/models/v5_p2_b3_conv1d_only_count4.py"
    canonical_model_path = repo / "src/models/v5_p2_task_d_full_multitask_count4.py"
    dynamic_model_path = repo / "src/models/v6_p0_dynamic70_task_d_full_multitask_count4.py"

    certified_decoder_path = repo / "src/decoders/v5_legal_xy_exact_decoder_certified.py"
    preserved_decoder_path = (
        repo
        / "reports/v5/p2_canonical_preservation/decoder/"
        "v5_legal_xy_exact_decoder_certified.py"
    )
    l5_contract_path = (
        repo
        / "reports/v5/p2_l5_final_pretest_freeze/"
        "V5_P2_ONE_SHOT_BLIND_EVALUATION_CONTRACT.json"
    )
    l4_selection_path = (
        repo
        / "reports/v5/p2_l4_decoder_validation_selection/"
        "V5_P2_L4_DECODER_VALIDATION_SELECTION.json"
    )
    l4_complete_path = (
        repo
        / "reports/v5/p2_l4_decoder_validation_selection/"
        "V5_P2_L4_DECODER_VALIDATION_SELECTION_COMPLETE"
    )
    d3_report_path = (
        repo
        / "reports/v5/p2_decoder_d3_full_validation_certification_proof/"
        "V5_P2_DECODER_D3_FULL_VALIDATION_CERTIFICATION_PROOF.json"
    )

    required = [
        a5_report_path,
        a5_lock_path,
        a4_checkpoint_path,
        wrapper_path,
        b3_path,
        canonical_model_path,
        dynamic_model_path,
        certified_decoder_path,
        preserved_decoder_path,
        l5_contract_path,
        l4_selection_path,
        l4_complete_path,
        d3_report_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")
    if not data_link.is_symlink():
        raise RuntimeError(f"dataset path is not the canonical symlink: {data_link}")

    a5_report = json.loads(a5_report_path.read_text(encoding="utf-8"))
    a5_lock = json.loads(a5_lock_path.read_text(encoding="utf-8"))
    if a5_report.get("status") != "PASS":
        raise RuntimeError("A5 report is not PASS")
    if a5_lock.get("report_sha256") != sha256_file(a5_report_path):
        raise RuntimeError("A5 report/lock SHA mismatch")
    if a5_report.get("sealed_test", {}).get("test_tensor_loaded"):
        raise RuntimeError("A5 reports sealed-test access")

    decoder_sha = sha256_file(certified_decoder_path)
    preserved_decoder_sha = sha256_file(preserved_decoder_path)
    if decoder_sha != preserved_decoder_sha:
        raise RuntimeError(
            "working certified decoder differs from canonical preservation copy"
        )

    l5_contract = json.loads(l5_contract_path.read_text(encoding="utf-8"))
    contract_a0, contract_a1, threshold_resolution_audit = (
        resolve_frozen_threshold_contract(l5_contract)
    )

    l4_selection = json.loads(l4_selection_path.read_text(encoding="utf-8"))
    l4_complete_text = l4_complete_path.read_text(encoding="utf-8")
    eligible_beam_zero = (
        "eligible_beam_configurations=0" in l4_complete_text
        or l4_selection.get("eligible_beam_configurations") == 0
        or l4_selection.get("A2_beam_revision", {}).get(
            "eligible_configuration_count"
        ) == 0
    )
    if not eligible_beam_zero:
        raise RuntimeError(
            "could not certify the frozen P2 disposition that no beam "
            "configuration passed the eligibility gates"
        )

    d3_report = json.loads(d3_report_path.read_text(encoding="utf-8"))
    d3_threshold = (
        d3_report.get("A1_margin_threshold")
        or d3_report.get("decoder", {}).get("A1_margin_threshold")
        or d3_report.get("thresholds", {}).get("A1_margin_threshold")
    )
    if d3_threshold is not None and float(d3_threshold) != EXPECTED_A1_MARGIN_THRESHOLD:
        raise RuntimeError("D3 certified A1 threshold differs from L5")

    # Process-wide guard: no A_test tensor may be deserialized.
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
                    f"D0 sealed-test deserialization guard blocked {text}"
                )
        return original_torch_load(file, *load_args, **load_kwargs)

    torch.load = guarded_torch_load

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    wrapper_mod = import_source(wrapper_path, "_v5_p3_d0_guarded_loader")
    b3_mod = import_source(b3_path, "_v5_p3_d0_b3")
    canonical_mod = import_source(canonical_model_path, "_v5_p3_d0_canonical")
    dynamic_mod = import_source(dynamic_model_path, "_v5_p3_d0_dynamic70")
    certified = importlib.import_module(
        "src.decoders.v5_legal_xy_exact_decoder_certified"
    )

    required_decoder_functions = {
        "solver_identity",
        "certification_policy",
        "decode_best_attack_hypothesis",
        "apply_margin_threshold",
    }
    missing_functions = sorted(
        name for name in required_decoder_functions
        if not hasattr(certified, name)
    )
    if missing_functions:
        raise RuntimeError(
            f"certified decoder missing functions: {missing_functions}"
        )

    decode_signature = str(
        inspect.signature(certified.decode_best_attack_hypothesis)
    )
    threshold_signature = str(
        inspect.signature(certified.apply_margin_threshold)
    )
    solver_identity = as_jsonable(certified.solver_identity())
    certification_policy = as_jsonable(certified.certification_policy())

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
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(
            f"validation length={len(validation_dataset)}, "
            f"expected={EXPECTED_VALIDATION_ITEMS}"
        )

    probe_indices = find_validation_probe_indices(validation_dataset)
    samples = [validation_dataset[index] for index in probe_indices]
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
    edge = torch.as_tensor(samples[0]["edge_index"], dtype=torch.long)
    graph_truth = np.asarray(
        [int(torch.as_tensor(sample["y_attack"]).item()) for sample in samples],
        dtype=np.int64,
    )
    count_truth = np.asarray(
        [
            int(torch.as_tensor(sample["y_attacker_count"]).item())
            for sample in samples
        ],
        dtype=np.int64,
    )

    reference = b3_mod.P2B3Conv1DOnlyCount4()
    canonical_model = canonical_mod.P2TaskDGraphConvCount4(reference, edge)
    model = dynamic_mod.build_v6_p0_dynamic70_from_canonical_structure(
        canonical_model,
        seed=SEED,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Dynamic70 parameter count={parameter_count}, "
            f"expected={EXPECTED_PARAMETER_COUNT}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    checkpoint = torch.load(
        a4_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint.get("epoch", -1)) != 14:
        raise RuntimeError(
            f"A4 best checkpoint epoch={checkpoint.get('epoch')}, expected=14"
        )
    if int(checkpoint.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("A4 checkpoint parameter-count contract changed")
    if checkpoint.get("threshold_tuning_performed") is not False:
        raise RuntimeError("A4 checkpoint unexpectedly reports threshold tuning")
    if checkpoint.get("test_tensor_loaded") is not False:
        raise RuntimeError("A4 checkpoint reports test access")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        outputs = model(x.to(device), mask.to(device))
    output_arrays = {
        key: value.detach().cpu().double().numpy()
        for key, value in outputs.items()
    }

    expected_shapes = {
        "attack_logits": (8,),
        "count_logits": (8, 4),
        "source_logits": (8, 16),
        "transit_logits": (8, 16),
        "victim_logits": (8, 16),
        "path_logits": (8, 16),
    }
    actual_shapes = {key: tuple(value.shape) for key, value in output_arrays.items()}
    if actual_shapes != expected_shapes:
        raise RuntimeError(
            f"output shapes={actual_shapes}, expected={expected_shapes}"
        )

    probabilities = {
        key: stable_sigmoid(value)
        for key, value in output_arrays.items()
        if key != "count_logits"
    }

    probes = []
    exact_times = []
    for local_index, dataset_index in enumerate(probe_indices):
        start = time.perf_counter()
        hypothesis = certified.decode_best_attack_hypothesis(
            float(output_arrays["attack_logits"][local_index]),
            output_arrays["count_logits"][local_index],
            output_arrays["source_logits"][local_index],
            output_arrays["transit_logits"][local_index],
            output_arrays["victim_logits"][local_index],
            output_arrays["path_logits"][local_index],
        )
        decoded = certified.apply_margin_threshold(
            hypothesis,
            EXPECTED_A1_MARGIN_THRESHOLD,
        )
        elapsed = time.perf_counter() - start
        exact_times.append(elapsed)

        hypothesis_json = as_jsonable(hypothesis)
        decoded_json = as_jsonable(decoded)
        margin = hypothesis_json.get("margin")
        if margin is None and hasattr(hypothesis, "margin"):
            margin = float(hypothesis.margin)
        if margin is None or not math.isfinite(float(margin)):
            raise RuntimeError(
                f"probe {dataset_index} produced non-finite/missing margin"
            )

        a0 = {
            "graph": bool(
                probabilities["attack_logits"][local_index]
                >= EXPECTED_A0_THRESHOLDS["graph"]
            ),
            "source": (
                probabilities["source_logits"][local_index]
                >= EXPECTED_A0_THRESHOLDS["source"]
            ).astype(np.int64).tolist(),
            "transit": (
                probabilities["transit_logits"][local_index]
                >= EXPECTED_A0_THRESHOLDS["transit"]
            ).astype(np.int64).tolist(),
            "victim": (
                probabilities["victim_logits"][local_index]
                >= EXPECTED_A0_THRESHOLDS["victim"]
            ).astype(np.int64).tolist(),
            "path": (
                probabilities["path_logits"][local_index]
                >= EXPECTED_A0_THRESHOLDS["path"]
            ).astype(np.int64).tolist(),
            "count": int(
                np.argmax(output_arrays["count_logits"][local_index]) + 1
            ),
        }

        probes.append({
            "dataset_index": int(dataset_index),
            "graph_truth": int(graph_truth[local_index]),
            "count_truth": int(count_truth[local_index]),
            "raw": {
                "graph_probability": float(
                    probabilities["attack_logits"][local_index]
                ),
                "count_prediction": int(
                    np.argmax(output_arrays["count_logits"][local_index]) + 1
                ),
            },
            "A0_frozen_P2_transfer": a0,
            "A1_hypothesis": hypothesis_json,
            "A1_decoded": decoded_json,
            "exact_decode_seconds": elapsed,
        })

    atomic_json(
        probe_path,
        {
            "stage": STAGE,
            "campaign_label": CAMPAIGN_LABEL,
            "probe_indices": probe_indices,
            "probes": probes,
        },
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
            "Compatibility and numerical smoke preflight for transferring "
            "the frozen P2 Raw/A0/A1 decoder policy to the V5-P3 "
            "Dynamic70 Tranche-A validation outputs."
        ),
        "audit_script_revision": (
            "v2_recursive_exact_threshold_contract_resolution"
        ),
        "frozen_transfer_policy": {
            "A0_thresholds": EXPECTED_A0_THRESHOLDS,
            "A1_margin_threshold": EXPECTED_A1_MARGIN_THRESHOLD,
            "threshold_resolution_audit": threshold_resolution_audit,
            "threshold_tuning_performed": False,
            "decoder_retraining_performed": False,
            "beam_decoder_selected": False,
            "beam_disposition": (
                "P2 validation selected no eligible hardware beam "
                "configuration; exact certified A1 remains diagnostic."
            ),
        },
        "decoder": {
            "working_path": str(certified_decoder_path),
            "working_sha256": decoder_sha,
            "canonical_preservation_path": str(preserved_decoder_path),
            "canonical_preservation_sha256": preserved_decoder_sha,
            "working_matches_canonical_preservation": True,
            "decode_signature": decode_signature,
            "threshold_signature": threshold_signature,
            "solver_identity": solver_identity,
            "certification_policy": certification_policy,
        },
        "model": {
            "class": model.__class__.__name__,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "parameter_count": parameter_count,
            "input_shape": [16, 70, 32],
            "output_shapes": {
                key: list(value) for key, value in actual_shapes.items()
            },
        },
        "data": {
            "split": "validation",
            "validation_items": len(validation_dataset),
            "probe_items": len(probe_indices),
            "probe_indices": probe_indices,
            "probe_graph_distribution": {
                "inactive": int(np.sum(graph_truth == 0)),
                "active": int(np.sum(graph_truth == 1)),
            },
            "probe_active_count_values": sorted(
                int(value) for value in set(count_truth[graph_truth == 1].tolist())
            ),
            "test_dataset_instantiated": False,
            "test_tensor_loaded": False,
        },
        "numerical_smoke": {
            "all_exact_decodes_completed": True,
            "exact_decode_seconds": exact_times,
            "mean_exact_decode_seconds": float(np.mean(exact_times)),
            "median_exact_decode_seconds": float(np.median(exact_times)),
            "maximum_exact_decode_seconds": float(np.max(exact_times)),
            "probe_artifact": str(probe_path),
            "probe_artifact_sha256": sha256_file(probe_path),
        },
        "decision": {
            "frozen_decoder_transfer_compatible": True,
            "immutable_validation_logit_export_authorized": True,
            "full_A0_evaluation_authorized_from_export": False,
            "resumable_A1_exact_evaluation_authorized_from_export": False,
            "sealed_test_evaluation_authorized": False,
            "next_stage": "V5_P3_D1_IMMUTABLE_TRANCHE_A_VALIDATION_LOGIT_EXPORT",
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
            "a5_report_sha256": sha256_file(a5_report_path),
            "a5_lock_sha256": sha256_file(a5_lock_path),
            "a4_best_checkpoint_sha256": sha256_file(a4_checkpoint_path),
            "guarded_loader_sha256": sha256_file(wrapper_path),
            "dynamic70_model_sha256": sha256_file(dynamic_model_path),
            "l5_contract_sha256": sha256_file(l5_contract_path),
            "l4_selection_sha256": sha256_file(l4_selection_path),
            "d3_certification_report_sha256": sha256_file(d3_report_path),
            "installed_script_sha256": sha256_file(installed_script),
        },
        "model_trained": False,
        "validation_replayed": True,
        "decoder_smoke_executed": True,
        "certified_dataset_modified": False,
        "generalization_claim_authorized": False,
    }
    atomic_json(report_path, report)

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(report_path),
        "probe_artifact_sha256": sha256_file(probe_path),
        "certified_decoder_sha256": decoder_sha,
        "a4_best_checkpoint_sha256": sha256_file(a4_checkpoint_path),
        "A0_thresholds": EXPECTED_A0_THRESHOLDS,
        "A1_margin_threshold": EXPECTED_A1_MARGIN_THRESHOLD,
        "parameter_count": parameter_count,
        "input_features": 70,
        "validation_probe_items": len(probe_indices),
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
    print(
        "audit_script_revision="
        "v2_recursive_exact_threshold_contract_resolution"
    )
    print(
        "selected_A0_threshold_path="
        f"{threshold_resolution_audit['selected_threshold_path']}"
    )
    print(
        "selected_A1_margin_path="
        f"{threshold_resolution_audit['selected_margin_path']}"
    )
    print(f"certified_decoder_sha256={decoder_sha}")
    print("working_matches_canonical_preservation=true")
    print(f"solver_identity={json.dumps(solver_identity, sort_keys=True)}")
    print(f"validation_items={len(validation_dataset)}")
    print(f"probe_items={len(probe_indices)}")
    print(f"probe_indices={probe_indices}")
    print("probe_active_count_values=1,2,3,4")
    print(f"parameter_count={parameter_count}")
    print("input_features=70")
    print(
        "mean_exact_decode_seconds="
        f"{float(np.mean(exact_times)):.6f}"
    )
    print(
        "maximum_exact_decode_seconds="
        f"{float(np.max(exact_times)):.6f}"
    )
    print("threshold_tuning_performed=false")
    print("beam_decoder_selected=false")
    print("test_dataset_instantiated=false")
    print("test_length_computed=false")
    print("test_tensor_loaded=false")
    print("immutable_validation_logit_export_authorized=true")
    print("full_A0_evaluation_authorized_from_export=false")
    print("resumable_A1_exact_evaluation_authorized_from_export=false")
    print("sealed_test_evaluation_authorized=false")
    print(
        "next_stage="
        "V5_P3_D1_IMMUTABLE_TRANCHE_A_VALIDATION_LOGIT_EXPORT"
    )
    print(f"probe_artifact={probe_path}")
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
