"""Deterministic hardware-oriented beam decoder for V5 P2."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from src.decoders.v5_xy_route_library import ROUTE_TABLE, RouteRecord, combine_route_ids, empty_hypothesis

ALPHA_GRAPH = 1.0
ALPHA_COUNT = 0.5
ALPHA_SOURCE = 1.0
ALPHA_TRANSIT = 0.5
ALPHA_VICTIM = 0.75
ALPHA_PATH = 0.5
ROUTER_COUNT = 16
K_VALUES = (1, 2, 3, 4)
ALLOWED_PER_SOURCE_TOP_B = (1, 2, 4)
ALLOWED_BEAM_WIDTHS = (4, 8, 16)
SCORE_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class BeamState:
    route_ids: tuple[int, ...]
    source_mask: int
    transit_mask: int
    victim_mask: int
    path_mask: int
    role_score: float


@dataclass(frozen=True, slots=True)
class BeamAttackHypothesis:
    margin: float
    attacker_count: int
    route_ids: tuple[int, ...]
    source_mask: int
    transit_mask: int
    victim_mask: int
    path_mask: int
    per_source_top_b: int
    beam_width: int
    candidate_route_count: int
    expanded_state_count: int
    maximum_beam_occupancy: int
    completion_feasibility_pruned_state_count: int


@dataclass(frozen=True, slots=True)
class BeamDecodedOutput:
    attack: int
    margin: float
    attacker_count: int
    route_ids: tuple[int, ...]
    source_mask: int
    transit_mask: int
    victim_mask: int
    path_mask: int


def _vec(name: str, value: Sequence[float], size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _scalar(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _log_softmax(values: np.ndarray) -> np.ndarray:
    maximum = float(values.max())
    return values - (maximum + math.log(float(np.exp(values - maximum).sum())))


def _mask_sum(logits: np.ndarray, mask: int) -> float:
    return sum(float(logits[i]) for i in range(ROUTER_COUNT) if (mask >> i) & 1)


def _role_score(source: np.ndarray, transit: np.ndarray, victim: np.ndarray, path: np.ndarray,
                source_mask: int, transit_mask: int, victim_mask: int, path_mask: int) -> float:
    return float(
        ALPHA_SOURCE * _mask_sum(source, source_mask)
        + ALPHA_TRANSIT * _mask_sum(transit, transit_mask)
        + ALPHA_VICTIM * _mask_sum(victim, victim_mask)
        + ALPHA_PATH * _mask_sum(path, path_mask)
    )


def single_route_role_score(route: RouteRecord, source_logits: Sequence[float],
                            transit_logits: Sequence[float], victim_logits: Sequence[float],
                            path_logits: Sequence[float]) -> float:
    source = _vec("source_logits", source_logits, 16)
    transit = _vec("transit_logits", transit_logits, 16)
    victim = _vec("victim_logits", victim_logits, 16)
    path = _vec("path_logits", path_logits, 16)
    return _role_score(source, transit, victim, path, route.source_mask, route.transit_mask,
                       route.victim_mask, route.path_mask)


def rank_candidates_by_source(source_logits: Sequence[float], transit_logits: Sequence[float],
                              victim_logits: Sequence[float], path_logits: Sequence[float], *,
                              per_source_top_b: int) -> dict[int, tuple[int, ...]]:
    if per_source_top_b not in ALLOWED_PER_SOURCE_TOP_B:
        raise ValueError(f"per_source_top_b must be one of {ALLOWED_PER_SOURCE_TOP_B}")
    source = _vec("source_logits", source_logits, 16)
    transit = _vec("transit_logits", transit_logits, 16)
    victim = _vec("victim_logits", victim_logits, 16)
    path = _vec("path_logits", path_logits, 16)
    ranked: dict[int, tuple[int, ...]] = {}
    for src in range(ROUTER_COUNT):
        scored = []
        for route in ROUTE_TABLE:
            if route.source != src:
                continue
            scored.append((
                _role_score(source, transit, victim, path, route.source_mask,
                            route.transit_mask, route.victim_mask, route.path_mask),
                route.route_id,
            ))
        scored.sort(key=lambda item: (-item[0], item[1]))
        ranked[src] = tuple(route_id for _, route_id in scored[:per_source_top_b])
    return ranked


def _expand(state: BeamState, route: RouteRecord, source: np.ndarray, transit: np.ndarray,
            victim: np.ndarray, path: np.ndarray) -> BeamState:
    if state.route_ids and route.route_id <= state.route_ids[-1]:
        raise ValueError("route IDs must be strictly increasing")
    if state.source_mask & route.source_mask:
        raise ValueError("duplicate source")
    new_source = route.source_mask & ~state.source_mask
    new_transit = route.transit_mask & ~state.transit_mask
    new_victim = route.victim_mask & ~state.victim_mask
    new_path = route.path_mask & ~state.path_mask
    increment = _role_score(source, transit, victim, path, new_source, new_transit, new_victim, new_path)
    return BeamState(
        route_ids=state.route_ids + (route.route_id,),
        source_mask=state.source_mask | route.source_mask,
        transit_mask=state.transit_mask | route.transit_mask,
        victim_mask=state.victim_mask | route.victim_mask,
        path_mask=state.path_mask | route.path_mask,
        role_score=state.role_score + increment,
    )



def _can_complete_to_k(state: BeamState, candidates: tuple[int, ...], target_k: int) -> bool:
    """Return whether a partial state can still reach exactly target_k routes.

    Route IDs are frozen in source-major order. A partial state is extendable
    only when enough distinct unused sources remain among candidate routes with
    IDs greater than the current final route ID. This guard prevents high-score
    dead-end states from occupying the entire beam.
    """
    if target_k not in K_VALUES:
        raise ValueError(f"target_k must be one of {K_VALUES}")
    selected = len(state.route_ids)
    if selected > target_k:
        return False
    remaining = target_k - selected
    if remaining == 0:
        return True
    last = state.route_ids[-1] if state.route_ids else -1
    available_sources = {
        ROUTE_TABLE[route_id].source
        for route_id in candidates
        if route_id > last
        and not (state.source_mask & ROUTE_TABLE[route_id].source_mask)
    }
    return len(available_sources) >= remaining

def recompute_state_role_score(state: BeamState, source_logits: Sequence[float],
                               transit_logits: Sequence[float], victim_logits: Sequence[float],
                               path_logits: Sequence[float]) -> float:
    source = _vec("source_logits", source_logits, 16)
    transit = _vec("transit_logits", transit_logits, 16)
    victim = _vec("victim_logits", victim_logits, 16)
    path = _vec("path_logits", path_logits, 16)
    return _role_score(source, transit, victim, path, state.source_mask, state.transit_mask,
                       state.victim_mask, state.path_mask)


def decode_beam_attack_hypothesis(graph_logit: float, count_logits: Sequence[float],
                                  source_logits: Sequence[float], transit_logits: Sequence[float],
                                  victim_logits: Sequence[float], path_logits: Sequence[float], *,
                                  per_source_top_b: int, beam_width: int) -> BeamAttackHypothesis:
    if per_source_top_b not in ALLOWED_PER_SOURCE_TOP_B:
        raise ValueError(f"per_source_top_b must be one of {ALLOWED_PER_SOURCE_TOP_B}")
    if beam_width not in ALLOWED_BEAM_WIDTHS:
        raise ValueError(f"beam_width must be one of {ALLOWED_BEAM_WIDTHS}")

    graph = _scalar("graph_logit", graph_logit)
    count = _vec("count_logits", count_logits, 4)
    source = _vec("source_logits", source_logits, 16)
    transit = _vec("transit_logits", transit_logits, 16)
    victim = _vec("victim_logits", victim_logits, 16)
    path = _vec("path_logits", path_logits, 16)
    count_log_probs = _log_softmax(count)

    ranked = rank_candidates_by_source(source, transit, victim, path, per_source_top_b=per_source_top_b)
    candidates = tuple(sorted(route_id for values in ranked.values() for route_id in values))
    if len(candidates) != ROUTER_COUNT * per_source_top_b or len(set(candidates)) != len(candidates):
        raise RuntimeError("candidate generation did not retain exactly B unique routes per source")

    best_margin = -math.inf
    best_state: BeamState | None = None
    expanded = 0
    max_occupancy = 1
    completion_pruned = 0

    for k in K_VALUES:
        beam = [BeamState((), 0, 0, 0, 0, 0.0)]
        for _ in range(k):
            next_states: dict[tuple[int, ...], BeamState] = {}
            for state in beam:
                last = state.route_ids[-1] if state.route_ids else -1
                for route_id in candidates:
                    if route_id <= last:
                        continue
                    route = ROUTE_TABLE[route_id]
                    if state.source_mask & route.source_mask:
                        continue
                    new_state = _expand(state, route, source, transit, victim, path)
                    expanded += 1
                    if not _can_complete_to_k(new_state, candidates, k):
                        completion_pruned += 1
                        continue
                    next_states[new_state.route_ids] = new_state
            if not next_states:
                raise RuntimeError(f"beam search produced no feasible states for K={k}")
            beam = sorted(next_states.values(), key=lambda state: (-state.role_score, state.route_ids))[:beam_width]
            max_occupancy = max(max_occupancy, len(beam))

        count_term = ALPHA_COUNT * float(count_log_probs[k - 1])
        for state in beam:
            margin = ALPHA_GRAPH * graph + count_term + state.role_score
            better = margin > best_margin + SCORE_TIE_TOLERANCE
            tied = abs(margin - best_margin) <= SCORE_TIE_TOLERANCE
            if better or (tied and (best_state is None or (len(state.route_ids), state.route_ids) <
                                    (len(best_state.route_ids), best_state.route_ids))):
                best_margin = float(margin)
                best_state = state

    if best_state is None:
        raise RuntimeError("beam decoder produced no hypothesis")

    masks = combine_route_ids(best_state.route_ids, expected_k=len(best_state.route_ids))
    if (masks.source_mask, masks.transit_mask, masks.victim_mask, masks.path_mask) != (
        best_state.source_mask, best_state.transit_mask, best_state.victim_mask, best_state.path_mask
    ):
        raise RuntimeError("beam-state masks differ from frozen route unions")

    recomputed = recompute_state_role_score(best_state, source, transit, victim, path)
    if abs(recomputed - best_state.role_score) > SCORE_TIE_TOLERANCE:
        raise RuntimeError("incremental union score differs from recomputed score")

    return BeamAttackHypothesis(
        margin=best_margin,
        attacker_count=masks.attacker_count,
        route_ids=best_state.route_ids,
        source_mask=masks.source_mask,
        transit_mask=masks.transit_mask,
        victim_mask=masks.victim_mask,
        path_mask=masks.path_mask,
        per_source_top_b=per_source_top_b,
        beam_width=beam_width,
        candidate_route_count=len(candidates),
        expanded_state_count=expanded,
        maximum_beam_occupancy=max_occupancy,
        completion_feasibility_pruned_state_count=completion_pruned,
    )


def decode_all_grid_configurations(graph_logit: float, count_logits: Sequence[float],
                                   source_logits: Sequence[float], transit_logits: Sequence[float],
                                   victim_logits: Sequence[float], path_logits: Sequence[float]) -> dict[tuple[int, int], BeamAttackHypothesis]:
    return {
        (b, width): decode_beam_attack_hypothesis(
            graph_logit, count_logits, source_logits, transit_logits, victim_logits, path_logits,
            per_source_top_b=b, beam_width=width,
        )
        for b in ALLOWED_PER_SOURCE_TOP_B
        for width in ALLOWED_BEAM_WIDTHS
    }


def apply_beam_margin_threshold(hypothesis: BeamAttackHypothesis, threshold: float) -> BeamDecodedOutput:
    tau = _scalar("threshold", threshold)
    if hypothesis.margin < tau:
        normal = empty_hypothesis()
        return BeamDecodedOutput(0, hypothesis.margin, normal.attacker_count, normal.route_ids,
                                 normal.source_mask, normal.transit_mask, normal.victim_mask, normal.path_mask)
    return BeamDecodedOutput(1, hypothesis.margin, hypothesis.attacker_count, hypothesis.route_ids,
                             hypothesis.source_mask, hypothesis.transit_mask,
                             hypothesis.victim_mask, hypothesis.path_mask)
