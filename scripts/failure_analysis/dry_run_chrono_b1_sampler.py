#!/usr/bin/env python3
"""
Chrono-B1 Stage B1.5: dry-run one complete sampler epoch.

This script:
- reads only metadata from the corrected chronological dataset;
- instantiates the frozen Chrono-B1 batch sampler;
- generates one full epoch of training indices;
- verifies split isolation, class balance, scenario coverage, run coverage,
  duplicate frequency, and temporal collision rate;
- writes JSON/CSV audit artifacts.

It does not instantiate the model and does not train.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from samplers.chrono_b1_scenario_sampler import (  # noqa: E402
    ScenarioBalancedBatchSampler,
)


REQUIRED_FILES = (
    "x.npy",
    "y_graph.npy",
    "split.npy",
    "profile.npy",
    "active_cores.npy",
    "strength.npy",
    "attackers.npy",
    "run_id.npy",
    "end_epoch.npy",
)


def load_array(data_dir: Path, name: str, *, mmap: bool = True) -> np.ndarray:
    path = data_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return np.load(
        path,
        mmap_mode="r" if mmap else None,
        allow_pickle=True,
    )


def text(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--samples-per-epoch", type=int, default=148185)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epoch-index", type=int, default=0)
    parser.add_argument("--max-temporal-retries", type=int, default=32)
    parser.add_argument("--max-collision-rate", type=float, default=0.05)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    for name in REQUIRED_FILES:
        if not (data_dir / name).is_file():
            raise SystemExit(f"STOP: required dataset file missing: {data_dir / name}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: output directory already exists and is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    x = load_array(data_dir, "x.npy")
    y_graph = load_array(data_dir, "y_graph.npy")
    split = load_array(data_dir, "split.npy", mmap=False).astype(str)
    profile = load_array(data_dir, "profile.npy", mmap=False)
    active_cores = load_array(data_dir, "active_cores.npy", mmap=False)
    strength = load_array(data_dir, "strength.npy", mmap=False)
    attackers = load_array(data_dir, "attackers.npy", mmap=False)
    run_id = load_array(data_dir, "run_id.npy", mmap=False).astype(str)
    end_epoch = load_array(data_dir, "end_epoch.npy")

    lengths = {
        "x": len(x),
        "y_graph": len(y_graph),
        "split": len(split),
        "profile": len(profile),
        "active_cores": len(active_cores),
        "strength": len(strength),
        "attackers": len(attackers),
        "run_id": len(run_id),
        "end_epoch": len(end_epoch),
    }
    if len(set(lengths.values())) != 1:
        raise SystemExit(f"STOP: metadata lengths differ: {lengths}")

    train_idx = np.flatnonzero(split == "train").astype(np.int64)
    val_idx = np.flatnonzero(split == "val").astype(np.int64)
    test_idx = np.flatnonzero(split == "test").astype(np.int64)

    if len(train_idx) == 0:
        raise SystemExit("STOP: training split contains zero samples")

    train_runs_expected = set(run_id[train_idx].tolist())
    train_profiles_expected = set(text(v).lower() for v in profile[train_idx])
    normal_profiles_expected = set(
        text(profile[i]).lower()
        for i in train_idx
        if int(y_graph[i]) == 0
    )
    attack_profiles_expected = set(
        text(profile[i]).lower()
        for i in train_idx
        if int(y_graph[i]) == 1
    )

    attack_strengths_expected = set()
    attack_counts_expected = set()
    for i in train_idx:
        if int(y_graph[i]) != 1:
            continue

        raw_strength = text(strength[i]).strip('"').strip("'")
        try:
            numeric_strength = float(raw_strength)
            canonical_strength = (
                str(int(numeric_strength))
                if numeric_strength.is_integer()
                else f"{numeric_strength:g}"
            )
        except ValueError:
            canonical_strength = raw_strength
        attack_strengths_expected.add(canonical_strength)

        raw_attackers = attackers[i]
        if isinstance(raw_attackers, np.ndarray):
            count = len(raw_attackers.tolist())
        else:
            count = None

        if count is None:
            import ast

            raw_text = text(raw_attackers)
            try:
                parsed = ast.literal_eval(raw_text)
                if isinstance(parsed, (list, tuple, set, np.ndarray)):
                    count = len(parsed)
                elif raw_text.lower() in {"", "na", "nan", "none", "null", "[]"}:
                    count = 0
                else:
                    count = 1
            except (ValueError, SyntaxError):
                for separator in ("-", ",", "_", " "):
                    if separator in raw_text:
                        count = len(
                            [p for p in raw_text.split(separator) if p.strip()]
                        )
                        break
                if count is None:
                    count = 1
        attack_counts_expected.add(int(count))

    sampler = ScenarioBalancedBatchSampler(
        y_graph=y_graph[train_idx],
        profile=profile[train_idx],
        active_cores=active_cores[train_idx],
        strength=strength[train_idx],
        attackers=attackers[train_idx],
        run_id=run_id[train_idx],
        end_epoch=end_epoch[train_idx],
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        temporal_window_length=int(x.shape[2]),
        seed=args.seed,
        drop_last=False,
        max_temporal_retries=args.max_temporal_retries,
    )
    sampler.set_epoch(args.epoch_index)

    local_index_counter: Counter[int] = Counter()
    real_index_counter: Counter[int] = Counter()
    batch_sizes: list[int] = []

    for batch in sampler:
        batch_sizes.append(len(batch))
        for local_index in batch:
            local_index = int(local_index)
            if local_index < 0 or local_index >= len(train_idx):
                raise SystemExit(
                    f"STOP: sampler produced invalid local index {local_index}"
                )
            real_index = int(train_idx[local_index])
            local_index_counter[local_index] += 1
            real_index_counter[real_index] += 1

    stats = sampler.last_epoch_stats()
    sampled_real_indices = np.fromiter(
        real_index_counter.keys(),
        dtype=np.int64,
    )

    val_overlap = int(np.intersect1d(sampled_real_indices, val_idx).size)
    test_overlap = int(np.intersect1d(sampled_real_indices, test_idx).size)
    non_train_count = int(np.sum(split[sampled_real_indices] != "train"))

    observed_runs = set(stats["run_counts"])
    observed_profiles = set(stats["profile_counts"])
    observed_strengths = {
        value
        for value in stats["strength_counts"]
        if value != "NA"
    }
    observed_attack_counts = {
        int(value)
        for value in stats["attacker_count_counts"]
        if int(value) > 0
    }

    normal_count = int(stats["class_counts"].get("normal", 0))
    attack_count = int(stats["class_counts"].get("attack", 0))
    samples_drawn = int(stats["samples_drawn"])
    normal_fraction = float(stats["normal_fraction"])

    checks = {
        "sample_count_exact": samples_drawn == args.samples_per_epoch,
        "normal_fraction_in_range": 0.49 <= normal_fraction <= 0.51,
        "normal_and_attack_present": normal_count > 0 and attack_count > 0,
        "all_training_runs_observed": observed_runs == train_runs_expected,
        "all_normal_profiles_observed": (
            normal_profiles_expected.issubset(observed_profiles)
        ),
        "all_attack_profiles_observed": (
            attack_profiles_expected.issubset(observed_profiles)
        ),
        "all_attack_strengths_observed": (
            attack_strengths_expected.issubset(observed_strengths)
        ),
        "all_attack_counts_observed": (
            attack_counts_expected.issubset(observed_attack_counts)
        ),
        "no_validation_indices": val_overlap == 0,
        "no_test_indices": test_overlap == 0,
        "all_indices_are_train": non_train_count == 0,
        "temporal_collision_rate_within_limit": (
            float(stats["temporal_collision_rate"])
            <= args.max_collision_rate
        ),
        "all_batches_nonempty": all(size > 0 for size in batch_sizes),
        "batch_count_matches_sampler": len(batch_sizes) == len(sampler),
    }
    overall_success = all(checks.values())

    summary = {
        "overall_success": overall_success,
        "checks": checks,
        "configuration": {
            "data_dir": str(data_dir),
            "batch_size": args.batch_size,
            "samples_per_epoch": args.samples_per_epoch,
            "seed": args.seed,
            "epoch_index": args.epoch_index,
            "temporal_window_length": int(x.shape[2]),
            "max_temporal_retries": args.max_temporal_retries,
            "max_collision_rate": args.max_collision_rate,
        },
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "coverage": {
            "expected_training_runs": len(train_runs_expected),
            "observed_training_runs": len(observed_runs),
            "missing_training_runs": sorted(
                train_runs_expected - observed_runs
            ),
            "expected_profiles": sorted(train_profiles_expected),
            "observed_profiles": sorted(observed_profiles),
            "expected_attack_strengths": sorted(attack_strengths_expected),
            "observed_attack_strengths": sorted(observed_strengths),
            "expected_attack_counts": sorted(attack_counts_expected),
            "observed_attack_counts": sorted(observed_attack_counts),
        },
        "split_isolation": {
            "validation_overlap": val_overlap,
            "test_overlap": test_overlap,
            "non_train_sample_count": non_train_count,
        },
        "sampler_stats": stats,
        "batch_statistics": {
            "batch_count": len(batch_sizes),
            "minimum_batch_size": min(batch_sizes) if batch_sizes else 0,
            "maximum_batch_size": max(batch_sizes) if batch_sizes else 0,
            "final_batch_size": batch_sizes[-1] if batch_sizes else 0,
        },
    }

    summary_path = output_dir / "sampler_dry_run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "sampled_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_id", "sampled_count"],
        )
        writer.writeheader()
        for key, value in sorted(
            stats["run_counts"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            writer.writerow({"run_id": key, "sampled_count": value})

    with (output_dir / "sampled_distribution.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dimension", "category", "sampled_count"],
        )
        writer.writeheader()
        for dimension, mapping in (
            ("class", stats["class_counts"]),
            ("profile", stats["profile_counts"]),
            ("strength", stats["strength_counts"]),
            ("attacker_count", stats["attacker_count_counts"]),
        ):
            for key, value in sorted(mapping.items()):
                writer.writerow(
                    {
                        "dimension": dimension,
                        "category": key,
                        "sampled_count": value,
                    }
                )

    print(
        "B1.5 SAMPLER DRY RUN: "
        + ("PASS" if overall_success else "FAIL")
    )
    print(f"samples_drawn={samples_drawn}")
    print(f"batches_drawn={stats['batches_drawn']}")
    print(f"normal_samples={normal_count}")
    print(f"attack_samples={attack_count}")
    print(f"normal_fraction={normal_fraction:.6f}")
    print(f"unique_indices_sampled={stats['unique_indices_sampled']}")
    print(f"duplicate_index_draws={stats['duplicate_index_draws']}")
    print(
        "duplicate_index_fraction="
        f"{stats['duplicate_index_fraction']:.6f}"
    )
    print(f"expected_training_runs={len(train_runs_expected)}")
    print(f"observed_training_runs={len(observed_runs)}")
    print(f"validation_overlap={val_overlap}")
    print(f"test_overlap={test_overlap}")
    print(
        "temporal_collision_count="
        f"{stats['temporal_collision_count']}"
    )
    print(
        "temporal_collision_rate="
        f"{stats['temporal_collision_rate']:.6f}"
    )
    print(
        "minimum_within_run_gap="
        f"{stats['minimum_observed_within_run_gap']}"
    )
    print(
        "median_within_run_gap="
        f"{stats['median_observed_within_run_gap']}"
    )
    print(f"output_dir={output_dir}")

    if not overall_success:
        failed = [name for name, passed in checks.items() if not passed]
        print("failed_checks=" + ",".join(failed))


if __name__ == "__main__":
    main()
