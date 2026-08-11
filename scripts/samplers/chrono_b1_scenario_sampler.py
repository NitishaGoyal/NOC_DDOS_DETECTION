#!/usr/bin/env python3
"""
Hierarchical scenario-balanced batch sampler for Chrono-B1.

The sampler yields dataset-local indices for the Chrono-A1 training Dataset.
It changes training exposure only. It does not modify dataset contents,
labels, model architecture, loss functions, or validation/test loaders.
"""

from __future__ import annotations

import ast
import math
from collections import Counter, defaultdict
from typing import Any, Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


def _to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _text(value: Any) -> str:
    value = _to_python(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if value is None:
        return "NA"
    text = str(value).strip()
    return text if text else "NA"


def _canonical_strength(value: Any) -> str:
    value = _to_python(value)
    if value is None:
        return "NA"
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "NA"
        return str(int(number)) if number.is_integer() else f"{number:g}"

    text = str(value).strip().strip('"').strip("'")
    if not text or text.lower() in {"na", "nan", "none", "null"}:
        return "NA"
    try:
        number = float(text)
        if math.isfinite(number):
            return str(int(number)) if number.is_integer() else f"{number:g}"
    except ValueError:
        pass
    return text


def _parse_sequence(value: Any) -> list[Any]:
    value = _to_python(value)
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, (int, np.integer)):
        number = int(value)
        return [] if number < 0 else [number]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number) or number < 0:
            return []
        return [int(number)]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "none", "null", "[]", "()"}:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set, np.ndarray)):
            return list(parsed)
        if isinstance(parsed, (int, float)):
            return [parsed]
    except (ValueError, SyntaxError):
        pass

    for separator in ("-", ",", "_", " "):
        if separator in text:
            return [part.strip() for part in text.split(separator) if part.strip()]

    return [text]


def _active_core_count(value: Any) -> int:
    return len(_parse_sequence(value))


def _attacker_count(value: Any, graph_label: int) -> int:
    if graph_label == 0:
        return 0
    return len(
        [
            item
            for item in _parse_sequence(value)
            if _text(item).lower() not in {"na", "nan", "none", "null", "-1"}
        ]
    )


def _choice(rng: np.random.Generator, values: Sequence[Any]) -> Any:
    if not values:
        raise RuntimeError("Cannot sample from an empty collection.")
    return values[int(rng.integers(0, len(values)))]


class ScenarioBalancedBatchSampler(Sampler[list[int]]):
    """
    Yield balanced batches of dataset-local training indices.

    Normal hierarchy:
        profile -> active-core count -> run -> temporally separated window

    Attack hierarchy:
        strength -> attacker count -> profile -> run ->
        temporally separated window
    """

    def __init__(
        self,
        *,
        y_graph: np.ndarray,
        profile: np.ndarray,
        active_cores: np.ndarray,
        strength: np.ndarray,
        attackers: np.ndarray,
        run_id: np.ndarray,
        end_epoch: np.ndarray,
        batch_size: int,
        samples_per_epoch: int,
        temporal_window_length: int,
        seed: int,
        drop_last: bool = False,
        max_temporal_retries: int = 32,
    ) -> None:
        super().__init__()

        arrays = {
            "y_graph": np.asarray(y_graph),
            "profile": np.asarray(profile),
            "active_cores": np.asarray(active_cores),
            "strength": np.asarray(strength),
            "attackers": np.asarray(attackers),
            "run_id": np.asarray(run_id),
            "end_epoch": np.asarray(end_epoch),
        }
        lengths = {name: len(array) for name, array in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Sampler metadata lengths differ: {lengths}")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive.")
        if temporal_window_length <= 0:
            raise ValueError("temporal_window_length must be positive.")
        if max_temporal_retries <= 0:
            raise ValueError("max_temporal_retries must be positive.")

        self.y_graph = arrays["y_graph"].astype(np.int64, copy=False)
        self.profile = np.asarray([_text(v).lower() for v in arrays["profile"]])
        self.active_core_count = np.asarray(
            [_active_core_count(v) for v in arrays["active_cores"]],
            dtype=np.int64,
        )
        self.strength = np.asarray(
            [_canonical_strength(v) for v in arrays["strength"]]
        )
        self.attacker_count = np.asarray(
            [
                _attacker_count(v, int(label))
                for v, label in zip(arrays["attackers"], self.y_graph)
            ],
            dtype=np.int64,
        )
        self.run_id = np.asarray([_text(v) for v in arrays["run_id"]])
        self.end_epoch = arrays["end_epoch"].astype(np.int64, copy=False)

        self.batch_size = int(batch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.temporal_window_length = int(temporal_window_length)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.max_temporal_retries = int(max_temporal_retries)
        self.epoch = 0
        self._last_stats: dict[str, Any] = {}

        self._normal_tree: dict[str, dict[int, dict[str, np.ndarray]]] = {}
        self._attack_tree: dict[
            str, dict[int, dict[str, dict[str, np.ndarray]]]
        ] = {}
        self._build_trees()

        if not self._normal_tree:
            raise ValueError("No normal training samples were available.")
        if not self._attack_tree:
            raise ValueError("No attack training samples were available.")

    def _build_trees(self) -> None:
        normal_temp: dict[
            str, dict[int, dict[str, list[int]]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        attack_temp: dict[
            str, dict[int, dict[str, dict[str, list[int]]]]
        ] = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(list))
            )
        )

        for local_index in range(len(self.y_graph)):
            label = int(self.y_graph[local_index])
            profile = str(self.profile[local_index])
            run = str(self.run_id[local_index])

            if label == 0:
                active_count = int(self.active_core_count[local_index])
                normal_temp[profile][active_count][run].append(local_index)
            elif label == 1:
                strength = str(self.strength[local_index])
                count = int(self.attacker_count[local_index])
                attack_temp[strength][count][profile][run].append(local_index)
            else:
                raise ValueError(
                    f"Expected binary y_graph labels, found {label} "
                    f"at local index {local_index}."
                )

        self._normal_tree = {
            profile: {
                active_count: {
                    run: np.asarray(indices, dtype=np.int64)
                    for run, indices in runs.items()
                }
                for active_count, runs in active_groups.items()
            }
            for profile, active_groups in normal_temp.items()
        }
        self._attack_tree = {
            strength: {
                count: {
                    profile: {
                        run: np.asarray(indices, dtype=np.int64)
                        for run, indices in runs.items()
                    }
                    for profile, runs in profiles.items()
                }
                for count, profiles in count_groups.items()
            }
            for strength, count_groups in attack_temp.items()
        }

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative.")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self.samples_per_epoch // self.batch_size
        return math.ceil(self.samples_per_epoch / self.batch_size)

    def last_epoch_stats(self) -> dict[str, Any]:
        return dict(self._last_stats)

    def _choose_normal_group(
        self,
        rng: np.random.Generator,
        used_runs: set[str],
    ) -> tuple[str, int, str, np.ndarray]:
        profiles = sorted(self._normal_tree)
        profile = _choice(rng, profiles)

        active_groups = self._normal_tree[profile]
        active_count = int(_choice(rng, sorted(active_groups)))

        runs = active_groups[active_count]
        run_names = sorted(runs)
        unused = [run for run in run_names if run not in used_runs]
        run = _choice(rng, unused if unused else run_names)
        return profile, active_count, run, runs[run]

    def _choose_attack_group(
        self,
        rng: np.random.Generator,
        used_runs: set[str],
    ) -> tuple[str, int, str, str, np.ndarray]:
        strengths = sorted(self._attack_tree)
        strength = _choice(rng, strengths)

        count_groups = self._attack_tree[strength]
        count = int(_choice(rng, sorted(count_groups)))

        profiles = count_groups[count]
        profile = _choice(rng, sorted(profiles))

        runs = profiles[profile]
        run_names = sorted(runs)
        unused = [run for run in run_names if run not in used_runs]
        run = _choice(rng, unused if unused else run_names)
        return strength, count, profile, run, runs[run]

    def _choose_temporally_separated(
        self,
        *,
        rng: np.random.Generator,
        candidates: np.ndarray,
        selected_epochs: list[int],
    ) -> tuple[int, bool, int | None]:
        if candidates.size == 0:
            raise RuntimeError("A sampler hierarchy leaf contains no samples.")

        if not selected_epochs:
            local_index = int(candidates[int(rng.integers(0, candidates.size))])
            return local_index, False, None

        for _ in range(self.max_temporal_retries):
            local_index = int(candidates[int(rng.integers(0, candidates.size))])
            epoch_value = int(self.end_epoch[local_index])
            minimum_gap = min(abs(epoch_value - old) for old in selected_epochs)
            if minimum_gap >= self.temporal_window_length:
                return local_index, False, minimum_gap

        candidate_epochs = self.end_epoch[candidates].astype(np.int64, copy=False)
        gaps = np.asarray(
            [
                min(abs(int(value) - old) for old in selected_epochs)
                for value in candidate_epochs
            ],
            dtype=np.int64,
        )
        best_gap = int(gaps.max())
        best_positions = np.flatnonzero(gaps == best_gap)
        chosen_position = int(_choice(rng, best_positions.tolist()))
        local_index = int(candidates[chosen_position])
        collision = best_gap < self.temporal_window_length
        return local_index, collision, best_gap

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)

        remaining = self.samples_per_epoch
        batch_number = 0

        class_counter: Counter[str] = Counter()
        profile_counter: Counter[str] = Counter()
        strength_counter: Counter[str] = Counter()
        attacker_count_counter: Counter[str] = Counter()
        run_counter: Counter[str] = Counter()
        index_counter: Counter[int] = Counter()

        temporal_collisions = 0
        repeated_run_draws = 0
        observed_gaps: list[int] = []

        while remaining > 0:
            current_batch_size = min(self.batch_size, remaining)
            if self.drop_last and current_batch_size < self.batch_size:
                break

            normal_count = current_batch_size // 2
            if current_batch_size % 2:
                normal_count += 1 if batch_number % 2 == 0 else 0
            attack_count = current_batch_size - normal_count

            labels = [0] * normal_count + [1] * attack_count
            rng.shuffle(labels)

            used_normal_runs: set[str] = set()
            used_attack_runs: set[str] = set()
            selected_epochs_by_run: dict[str, list[int]] = defaultdict(list)
            batch: list[int] = []

            for label in labels:
                if label == 0:
                    profile, _, run, candidates = self._choose_normal_group(
                        rng, used_normal_runs
                    )
                    used_before = run in used_normal_runs
                    used_normal_runs.add(run)
                    strength = "NA"
                    count = 0
                else:
                    (
                        strength,
                        count,
                        profile,
                        run,
                        candidates,
                    ) = self._choose_attack_group(rng, used_attack_runs)
                    used_before = run in used_attack_runs
                    used_attack_runs.add(run)

                if used_before:
                    repeated_run_draws += 1

                local_index, collision, gap = self._choose_temporally_separated(
                    rng=rng,
                    candidates=candidates,
                    selected_epochs=selected_epochs_by_run[run],
                )
                epoch_value = int(self.end_epoch[local_index])
                selected_epochs_by_run[run].append(epoch_value)

                if collision:
                    temporal_collisions += 1
                if gap is not None:
                    observed_gaps.append(int(gap))

                batch.append(local_index)
                class_counter["normal" if label == 0 else "attack"] += 1
                profile_counter[str(profile)] += 1
                strength_counter[str(strength)] += 1
                attacker_count_counter[str(count)] += 1
                run_counter[str(run)] += 1
                index_counter[local_index] += 1

            yield batch
            remaining -= current_batch_size
            batch_number += 1

        total_draws = sum(class_counter.values())
        duplicate_draws = sum(count - 1 for count in index_counter.values() if count > 1)
        normal_draws = class_counter.get("normal", 0)

        self._last_stats = {
            "epoch_index": self.epoch,
            "seed_used": self.seed + self.epoch,
            "samples_requested": self.samples_per_epoch,
            "samples_drawn": total_draws,
            "batches_drawn": batch_number,
            "class_counts": dict(sorted(class_counter.items())),
            "normal_fraction": normal_draws / total_draws if total_draws else 0.0,
            "profile_counts": dict(sorted(profile_counter.items())),
            "strength_counts": dict(sorted(strength_counter.items())),
            "attacker_count_counts": dict(sorted(attacker_count_counter.items())),
            "unique_runs_sampled": len(run_counter),
            "run_counts": dict(sorted(run_counter.items())),
            "unique_indices_sampled": len(index_counter),
            "duplicate_index_draws": duplicate_draws,
            "duplicate_index_fraction": (
                duplicate_draws / total_draws if total_draws else 0.0
            ),
            "repeated_run_draws_within_batches": repeated_run_draws,
            "temporal_collision_count": temporal_collisions,
            "temporal_collision_rate": (
                temporal_collisions / repeated_run_draws
                if repeated_run_draws
                else 0.0
            ),
            "minimum_observed_within_run_gap": (
                min(observed_gaps) if observed_gaps else None
            ),
            "median_observed_within_run_gap": (
                float(np.median(observed_gaps)) if observed_gaps else None
            ),
            "temporal_window_length": self.temporal_window_length,
        }
