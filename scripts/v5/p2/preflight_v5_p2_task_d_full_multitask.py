#!/usr/bin/env python3
"""Synthetic and provenance preflight for the frozen Task-D 10-run matrix."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

STAGE = "V5_P2_TASK_D_FULL_MULTITASK_PREFLIGHT"
COMPLETE = f"{STAGE}_COMPLETE"
SEEDS = [107, 117, 127, 137, 147]
EXPECTED_PARAMS = {"conv1d": 43_273, "graphconv": 59_785}
EXPECTED_B1_PROTOCOL_SHA = "0817ae7812f91e3c75260589acaf52bd8a74b07b2114a451b4773578efde4b60"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_state_dict(state: dict[str, torch.Tensor]) -> str:
    buf = io.BytesIO()
    cpu = OrderedDict((k, v.detach().cpu().contiguous()) for k, v in state.items())
    torch.save(cpu, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_edge_index(edge: torch.Tensor) -> None:
    if tuple(edge.shape) != (2, 48):
        raise RuntimeError(f"edge_index shape={tuple(edge.shape)}")
    values = edge.detach().cpu().numpy()
    if values.min() != 0 or values.max() != 15:
        raise RuntimeError("edge_index router range is not 0..15")
    pairs = {tuple(map(int, x)) for x in values.T.tolist()}
    if len(pairs) != 48:
        raise RuntimeError("edge_index does not contain 48 unique directed edges")
    for s, d in pairs:
        sr, sc = divmod(s, 4)
        dr, dc = divmod(d, 4)
        if abs(sr - dr) + abs(sc - dc) != 1:
            raise RuntimeError(f"non-cardinal edge {s}->{d}")
        if (d, s) not in pairs:
            raise RuntimeError(f"missing reverse edge for {s}->{d}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        print(f"STOP: output already exists: {out}")
        return 2
    out.mkdir(parents=True)

    paths = {
        "promotion_report": repo / "reports/v5/p2_g123_graph_operator_promotion/V5_P2_G123_GRAPH_OPERATOR_PROMOTION.json",
        "promotion_lock": repo / "reports/v5/p2_g123_graph_operator_promotion/V5_P2_G123_GRAPH_OPERATOR_PROMOTION_LOCK.json",
        "promotion_marker": repo / "reports/v5/p2_g123_graph_operator_promotion/V5_P2_G123_GRAPH_OPERATOR_PROMOTION_COMPLETE",
        "b1_protocol": repo / "reports/v5/p2_b1_training_protocol_lock/V5_P2_B1_TRAINING_PROTOCOL.json",
        "b1_report": repo / "reports/v5/p2_b1_training_protocol_lock/V5_P2_B1_TRAINING_PROTOCOL_LOCK.json",
        "b1_lock": repo / "reports/v5/p2_b1_training_protocol_lock/V5_P2_B1_TRAINING_PROTOCOL_LOCK_LOCK.json",
        "g0_seed_policy": repo / "reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol/V5_P2_G0_TRAINING_SELECTION_AND_SEED_POLICY.json",
        "g0_operator_contracts": repo / "reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol/V5_P2_G0_GRAPH_OPERATOR_CONTRACTS.json",
        "test_policy": repo / "reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol/V5_P2_G0_P2_TEST_ACCESS_POLICY.json",
        "edge": repo / "reports/v5/p2_g1a_r2a_canonical_static_topology_contract/V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
        "b3_model": repo / "src/models/v5_p2_b3_conv1d_only_count4.py",
        "task_d_model": repo / "src/models/v5_p2_task_d_full_multitask_count4.py",
        "b2_train": repo / "scripts/v5/p2/train_v5_p2_b2_single_seed.py",
        "loader": repo / "src/data/v5_p2_pair_aligned_primary58_dataset.py",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    failures: list[str] = []
    if missing:
        failures.extend(f"missing: {p}" for p in missing)

    results: dict[str, Any] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if failures:
            raise RuntimeError("required files missing")
        promotion = json.loads(paths["promotion_report"].read_text())
        promotion_lock = json.loads(paths["promotion_lock"].read_text())
        if promotion_lock["report_sha256"] != sha256_file(paths["promotion_report"]):
            raise RuntimeError("promotion report SHA mismatch")
        if promotion["decision"]["task_d_challenger"] != "graphconv":
            raise RuntimeError("Task-D challenger is not graphconv")
        if promotion["decision"]["architecture_selected"] is not False:
            raise RuntimeError("architecture was selected before Task D")

        protocol = json.loads(paths["b1_protocol"].read_text())
        b1_report = json.loads(paths["b1_report"].read_text())
        b1_lock = json.loads(paths["b1_lock"].read_text())
        if protocol["protocol_sha256"] != EXPECTED_B1_PROTOCOL_SHA:
            raise RuntimeError("B1 protocol SHA changed")
        if b1_lock["protocol_file_sha256"] != sha256_file(paths["b1_protocol"]):
            raise RuntimeError("B1 protocol file SHA mismatch")
        if b1_lock["model_sha256"] != sha256_file(paths["b3_model"]):
            raise RuntimeError("B3 model SHA mismatch")
        if b1_lock["loader_sha256"] != sha256_file(paths["loader"]):
            raise RuntimeError("loader SHA mismatch")

        seed_policy = json.loads(paths["g0_seed_policy"].read_text())
        if seed_policy["training_base"]["finalist_stability_seeds"] != SEEDS:
            raise RuntimeError("finalist seed set changed")
        operator_contracts = json.loads(paths["g0_operator_contracts"].read_text())
        if operator_contracts["common_architecture"]["graph_layer_count"] != 2:
            raise RuntimeError("graph layer count changed")
        test_policy = json.loads(paths["test_policy"].read_text())
        if test_policy["g1_to_g6"]["p2_test_directory_enumeration_allowed"] is not False:
            raise RuntimeError("test directory enumeration unexpectedly allowed")

        edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
        validate_edge_index(edge)

        b3_mod = import_module(paths["b3_model"], "taskd_preflight_b3")
        td_mod = import_module(paths["task_d_model"], "taskd_preflight_model")
        b2_mod = import_module(paths["b2_train"], "taskd_preflight_b2")

        # Matched-seed common initialization proof.
        set_seed(107)
        conv_reference = b3_mod.P2B3Conv1DOnlyCount4()
        set_seed(107)
        graph_reference = b3_mod.P2B3Conv1DOnlyCount4()
        graph_model = td_mod.P2TaskDGraphConvCount4(graph_reference, edge)
        conv_hash = sha256_state_dict(conv_reference.state_dict())
        graph_common_hash = sha256_state_dict(td_mod.common_state_dict(graph_model))
        if conv_hash != graph_common_hash:
            raise RuntimeError("matched-seed common initialization is not identical")

        # Independent duplicate B3 instantiation must be exact under the seed.
        set_seed(107)
        duplicate_b3 = b3_mod.P2B3Conv1DOnlyCount4()
        if sha256_state_dict(duplicate_b3.state_dict()) != conv_hash:
            raise RuntimeError("B3 deterministic initialization failed")

        batch_size = 4
        x = torch.randn(batch_size, 16, 58, 32, device=device)
        mask = torch.ones(batch_size, 16, 10, dtype=torch.bool, device=device)
        batch = {
            "x": x,
            "physical_port_mask": mask,
            "y_attack": torch.tensor([0, 1, 1, 1], dtype=torch.float32, device=device),
            "y_attacker_count": torch.tensor([0, 1, 2, 4], dtype=torch.int64, device=device),
            "y_source": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_transit": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_victim": torch.randint(0, 2, (batch_size, 16), device=device),
            "y_attack_path": torch.randint(0, 2, (batch_size, 16), device=device),
        }
        graph_pos_weight = torch.tensor(2.45339108179939, device=device)
        role_weights = {
            "source": torch.tensor(20.0, device=device),
            "transit": torch.tensor(15.7433147902343, device=device),
            "victim": torch.tensor(20.0, device=device),
            "path": torch.tensor(7.375342240922689, device=device),
        }

        for candidate, model in (
            ("conv1d", conv_reference.to(device)),
            ("graphconv", graph_model.to(device)),
        ):
            params = sum(p.numel() for p in model.parameters())
            if params != EXPECTED_PARAMS[candidate]:
                raise RuntimeError(f"{candidate} params={params}, expected {EXPECTED_PARAMS[candidate]}")
            model.train()
            outputs = model(x, mask)
            loss, components = b2_mod.compute_loss(
                outputs,
                batch,
                graph_pos_weight=graph_pos_weight,
                role_pos_weights=role_weights,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"{candidate} non-finite loss")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"{candidate} non-finite gradients")
            optimizer.step()
            results[candidate] = {
                "status": "PASS",
                "parameter_count": params,
                "synthetic_total_loss": float(loss.item()),
                "component_losses": {k: float(v.item()) for k, v in components.items()},
                "operation_count_proxy": td_mod.operation_count_proxy(candidate),
            }

        status = "COMPLETE"
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        status = "HOLD"

    report = {
        "stage": STAGE,
        "status": status,
        "device": str(device),
        "seeds": SEEDS,
        "candidates": ["conv1d", "graphconv"],
        "total_runs": 10,
        "candidate_results": results,
        "matched_seed_common_initialization_sha256": (
            locals().get("conv_hash") if status == "COMPLETE" else None
        ),
        "scientific_training_authorized": status == "COMPLETE",
        "failures": failures,
        "architecture_selected": False,
        "test_directory_enumerated": False,
        "test_tensors_deserialized": False,
        "test_evaluation_performed": False,
        "provenance": {k: sha256_file(v) for k, v in paths.items() if v.is_file()},
    }
    report_path = out / f"{STAGE}.json"
    write_json(report_path, report)
    lock = {
        "status": f"{STAGE}_{status}",
        "report_sha256": sha256_file(report_path),
        "scientific_training_authorized": status == "COMPLETE",
        "candidate_parameter_counts": EXPECTED_PARAMS,
        "seeds": SEEDS,
        "architecture_selected": False,
        "p2_test_directory_enumerated": False,
        "p2_test_tensors_deserialized": False,
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(out / f"{STAGE}_LOCK.json", lock)
    marker = out / (COMPLETE if status == "COMPLETE" else f"{STAGE}_HOLD")
    marker.write_text(marker.name + "\n", encoding="utf-8")

    print("===== V5 P2 TASK-D FULL MULTITASK PREFLIGHT =====")
    print("status:", status)
    print("device:", device)
    for candidate, value in results.items():
        print(candidate, value["status"], "params=" + str(value["parameter_count"]), "loss=" + f"{value['synthetic_total_loss']:.8f}")
    print("scientific_training_authorized:", str(status == "COMPLETE").lower())
    print("architecture_selected: false")
    print("test_directory_enumerated: false")
    print("test_tensors_deserialized: false")
    print(marker.name)
    return 0 if status == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
