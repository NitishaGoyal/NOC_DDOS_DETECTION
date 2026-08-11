"""V5-P3 EPLR-V1: Endpoint-Preserving Legal-XY Repair.

Software-side decoder for the existing 69 neural logits.

Frozen V1 behavior:
  - graph decision is Raw sigmoid(graph_logit) >= 0.5;
  - active count is Raw argmax(count_logits) + 1;
  - inactive outputs have effective count zero and zero role masks;
  - exact Raw source/victim preservation is attempted first;
  - endpoint repair is used only when exact preservation is infeasible;
  - graph and count never change;
  - exactly K canonical XY routes are selected;
  - attacker sources are unique and victims may be shared;
  - role outputs are union bitmaps;
  - no global endpoint/transit disjointness is imposed;
  - failures are explicit and fail closed.

The production repair path uses SciPy MILP. Exact endpoint-preserving route
selection uses finite enumeration over Raw endpoints and is much cheaper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import json
from math import exp, log
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_array, vstack

from src.decoders import v5_xy_route_library as routes


STATUS_INACTIVE = "INACTIVE_RAW"
STATUS_LEGAL = "RAW_ENDPOINTS_LEGAL"
STATUS_LEGAL_LOW_SUPPORT = "RAW_ENDPOINTS_LEGAL_LOW_ROUTE_SUPPORT"
STATUS_REPAIRED = "ENDPOINT_REPAIR_APPLIED"
STATUS_NO_SOLUTION = "NO_CERTIFIED_LEGAL_EXPLANATION"

GRAPH_THRESHOLD = 0.5
ENDPOINT_THRESHOLD = 0.5
OBJECTIVE_FREEZE_TOLERANCE = 1e-8
INTEGRALITY_TOLERANCE = 1e-7

Q_OFFSET = 0
S_OFFSET = Q_OFFSET + routes.ORDERED_ROUTE_COUNT
V_OFFSET = S_OFFSET + routes.ROUTER_COUNT
T_OFFSET = V_OFFSET + routes.ROUTER_COUNT
P_OFFSET = T_OFFSET + routes.ROUTER_COUNT
VARIABLE_COUNT = P_OFFSET + routes.ROUTER_COUNT


class EPLRDecodeError(RuntimeError):
    """EPLR could not return a certified legal explanation."""


@dataclass(frozen=True)
class EPLRConfig:
    transit_weight: float = 1.0
    path_weight: float = 1.0
    low_support_threshold_by_k: tuple[
        float, float, float, float
    ] = (
        float("-inf"),
        float("-inf"),
        float("-inf"),
        float("-inf"),
    )
    graph_threshold: float = GRAPH_THRESHOLD
    endpoint_threshold: float = ENDPOINT_THRESHOLD
    objective_freeze_tolerance: float = OBJECTIVE_FREEZE_TOLERANCE
    integrality_tolerance: float = INTEGRALITY_TOLERANCE

    @classmethod
    def from_json(cls, path: str | Path) -> "EPLRConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            transit_weight=float(payload["transit_weight"]),
            path_weight=float(payload["path_weight"]),
            low_support_threshold_by_k=tuple(
                float(value)
                for value in payload["low_support_threshold_by_k"]
            ),
            graph_threshold=float(
                payload.get("graph_threshold", GRAPH_THRESHOLD)
            ),
            endpoint_threshold=float(
                payload.get("endpoint_threshold", ENDPOINT_THRESHOLD)
            ),
            objective_freeze_tolerance=float(
                payload.get(
                    "objective_freeze_tolerance",
                    OBJECTIVE_FREEZE_TOLERANCE,
                )
            ),
            integrality_tolerance=float(
                payload.get(
                    "integrality_tolerance",
                    INTEGRALITY_TOLERANCE,
                )
            ),
        )


@dataclass(frozen=True)
class EPLRResult:
    raw_graph_probability: float
    raw_graph_prediction: int
    raw_attacker_count_candidate: int
    effective_attacker_count: int
    raw_source_bitmap: int
    raw_transit_bitmap: int
    raw_victim_bitmap: int
    raw_path_bitmap: int
    decoded_source_bitmap: int
    decoded_transit_bitmap: int
    decoded_victim_bitmap: int
    decoded_path_bitmap: int
    selected_route_ids: tuple[int, ...]
    route_consistency_score: float
    source_repair_count: int
    victim_repair_count: int
    endpoint_preserved_but_low_route_support: bool
    decoder_status: str
    repair_solver_used: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_sigmoid(value: float) -> float:
    value = float(value)
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    exp_value = exp(value)
    return exp_value / (1.0 + exp_value)


def stable_sigmoid_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    positive = array >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_values = np.exp(array[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def mask_from_probabilities(
    probabilities: Sequence[float],
    threshold: float,
) -> int:
    mask = 0
    for index, probability in enumerate(probabilities):
        if float(probability) >= float(threshold):
            mask |= 1 << index
    return mask


def indices_from_mask(mask: int) -> tuple[int, ...]:
    return tuple(
        index
        for index in range(routes.ROUTER_COUNT)
        if int(mask) & (1 << index)
    )


def hamming_distance(left: int, right: int) -> int:
    return int(left ^ right).bit_count()


def bernoulli_mask_log_likelihood(
    mask: int,
    probabilities: Sequence[float],
    *,
    epsilon: float = 1e-12,
) -> float:
    score = 0.0
    for index, probability in enumerate(probabilities):
        probability = min(max(float(probability), epsilon), 1.0 - epsilon)
        score += (
            log(probability)
            if int(mask) & (1 << index)
            else log(1.0 - probability)
        )
    return score


def route_support_score(
    hypothesis: Any,
    transit_probabilities: Sequence[float],
    path_probabilities: Sequence[float],
    config: EPLRConfig,
) -> float:
    return (
        config.transit_weight
        * bernoulli_mask_log_likelihood(
            hypothesis.transit_mask,
            transit_probabilities,
        )
        + config.path_weight
        * bernoulli_mask_log_likelihood(
            hypothesis.path_mask,
            path_probabilities,
        )
    )


def enumerate_endpoint_preserving_hypotheses(
    source_mask: int,
    victim_mask: int,
    k: int,
) -> tuple[Any, ...]:
    if k not in (1, 2, 3, 4):
        raise ValueError("attacker count must be in 1..4")

    sources = indices_from_mask(source_mask)
    victims = indices_from_mask(victim_mask)
    if len(sources) != k or not victims or len(victims) > k:
        return ()

    hypotheses: dict[tuple[int, ...], Any] = {}
    for assignment in product(victims, repeat=k):
        if set(assignment) != set(victims):
            continue
        if any(
            source == victim
            for source, victim in zip(sources, assignment)
        ):
            continue
        route_ids = [
            routes.route_id_for(source, victim)
            for source, victim in zip(sources, assignment)
        ]
        hypothesis = routes.combine_route_ids(
            route_ids,
            expected_k=k,
        )
        if (
            hypothesis.source_mask == source_mask
            and hypothesis.victim_mask == victim_mask
        ):
            hypotheses[hypothesis.route_ids] = hypothesis

    return tuple(
        hypotheses[key] for key in sorted(hypotheses)
    )


def choose_endpoint_preserving_hypothesis(
    *,
    source_mask: int,
    victim_mask: int,
    k: int,
    transit_probabilities: Sequence[float],
    path_probabilities: Sequence[float],
    config: EPLRConfig,
) -> tuple[Any | None, float]:
    candidates = enumerate_endpoint_preserving_hypotheses(
        source_mask,
        victim_mask,
        k,
    )
    if not candidates:
        return None, float("-inf")

    rows = [
        (
            route_support_score(
                hypothesis,
                transit_probabilities,
                path_probabilities,
                config,
            ),
            hypothesis.route_ids,
            hypothesis,
        )
        for hypothesis in candidates
    ]
    best_score = max(row[0] for row in rows)
    best = min(
        (row for row in rows if row[0] == best_score),
        key=lambda row: row[1],
    )
    return best[2], float(best[0])


def _append_constraint(
    rows: list[dict[int, float]],
    lower: list[float],
    upper: list[float],
    coefficients: dict[int, float],
    lb: float,
    ub: float,
) -> None:
    rows.append(coefficients)
    lower.append(float(lb))
    upper.append(float(ub))


def _rows_to_constraint(
    rows: list[dict[int, float]],
    lower: list[float],
    upper: list[float],
) -> LinearConstraint:
    data = []
    row_indices = []
    column_indices = []
    for row_index, coefficients in enumerate(rows):
        for column, value in coefficients.items():
            if value != 0.0:
                row_indices.append(row_index)
                column_indices.append(column)
                data.append(float(value))
    matrix = csc_array(
        (
            np.asarray(data, dtype=np.float64),
            (
                np.asarray(row_indices, dtype=np.int32),
                np.asarray(column_indices, dtype=np.int32),
            ),
        ),
        shape=(len(rows), VARIABLE_COUNT),
    )
    return LinearConstraint(
        matrix,
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )


def _base_repair_constraints(k: int) -> LinearConstraint:
    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    _append_constraint(
        rows,
        lower,
        upper,
        {
            Q_OFFSET + route_id: 1.0
            for route_id in range(routes.ORDERED_ROUTE_COUNT)
        },
        k,
        k,
    )

    for router in range(routes.ROUTER_COUNT):
        source_route_ids = [
            record.route_id
            for record in routes.ROUTE_TABLE
            if record.source == router
        ]
        coefficients = {S_OFFSET + router: 1.0}
        coefficients.update(
            {
                Q_OFFSET + route_id: -1.0
                for route_id in source_route_ids
            }
        )
        _append_constraint(
            rows,
            lower,
            upper,
            coefficients,
            0.0,
            0.0,
        )

    for router in range(routes.ROUTER_COUNT):
        victim_route_ids = [
            record.route_id
            for record in routes.ROUTE_TABLE
            if record.victim == router
        ]
        for route_id in victim_route_ids:
            _append_constraint(
                rows,
                lower,
                upper,
                {
                    Q_OFFSET + route_id: 1.0,
                    V_OFFSET + router: -1.0,
                },
                -np.inf,
                0.0,
            )
        coefficients = {V_OFFSET + router: 1.0}
        coefficients.update(
            {
                Q_OFFSET + route_id: -1.0
                for route_id in victim_route_ids
            }
        )
        _append_constraint(
            rows,
            lower,
            upper,
            coefficients,
            -np.inf,
            0.0,
        )

    for router in range(routes.ROUTER_COUNT):
        transit_route_ids = [
            record.route_id
            for record in routes.ROUTE_TABLE
            if record.transit_mask & (1 << router)
        ]
        if transit_route_ids:
            for route_id in transit_route_ids:
                _append_constraint(
                    rows,
                    lower,
                    upper,
                    {
                        Q_OFFSET + route_id: 1.0,
                        T_OFFSET + router: -1.0,
                    },
                    -np.inf,
                    0.0,
                )
            coefficients = {T_OFFSET + router: 1.0}
            coefficients.update(
                {
                    Q_OFFSET + route_id: -1.0
                    for route_id in transit_route_ids
                }
            )
            _append_constraint(
                rows,
                lower,
                upper,
                coefficients,
                -np.inf,
                0.0,
            )
        else:
            _append_constraint(
                rows,
                lower,
                upper,
                {T_OFFSET + router: 1.0},
                0.0,
                0.0,
            )

    for router in range(routes.ROUTER_COUNT):
        path_route_ids = [
            record.route_id
            for record in routes.ROUTE_TABLE
            if record.path_mask & (1 << router)
        ]
        for route_id in path_route_ids:
            _append_constraint(
                rows,
                lower,
                upper,
                {
                    Q_OFFSET + route_id: 1.0,
                    P_OFFSET + router: -1.0,
                },
                -np.inf,
                0.0,
            )
        coefficients = {P_OFFSET + router: 1.0}
        coefficients.update(
            {
                Q_OFFSET + route_id: -1.0
                for route_id in path_route_ids
            }
        )
        _append_constraint(
            rows,
            lower,
            upper,
            coefficients,
            -np.inf,
            0.0,
        )

    return _rows_to_constraint(rows, lower, upper)


def _binary_cost_coefficients(
    probabilities: Sequence[float],
    offset: int,
) -> np.ndarray:
    coefficients = np.zeros(VARIABLE_COUNT, dtype=np.float64)
    epsilon = 1e-12
    for router, probability in enumerate(probabilities):
        probability = min(max(float(probability), epsilon), 1.0 - epsilon)
        cost_if_zero = -log(1.0 - probability)
        cost_if_one = -log(probability)
        coefficients[offset + router] = cost_if_one - cost_if_zero
    return coefficients


def _hamming_coefficients(raw_mask: int, offset: int) -> np.ndarray:
    coefficients = np.zeros(VARIABLE_COUNT, dtype=np.float64)
    for router in range(routes.ROUTER_COUNT):
        coefficients[offset + router] = (
            -1.0 if raw_mask & (1 << router) else 1.0
        )
    return coefficients


def _negative_support_coefficients(
    probabilities: Sequence[float],
    offset: int,
    weight: float,
) -> np.ndarray:
    # Minimizing this coefficient vector maximizes Bernoulli log-likelihood.
    coefficients = np.zeros(VARIABLE_COUNT, dtype=np.float64)
    epsilon = 1e-12
    for router, probability in enumerate(probabilities):
        probability = min(max(float(probability), epsilon), 1.0 - epsilon)
        ll_if_zero = log(1.0 - probability)
        ll_if_one = log(probability)
        coefficients[offset + router] = (
            -float(weight) * (ll_if_one - ll_if_zero)
        )
    return coefficients


def _solve_milp(
    objective: np.ndarray,
    constraints: list[LinearConstraint],
) -> np.ndarray:
    result = milp(
        c=np.asarray(objective, dtype=np.float64),
        integrality=np.ones(VARIABLE_COUNT, dtype=np.uint8),
        bounds=Bounds(
            np.zeros(VARIABLE_COUNT, dtype=np.float64),
            np.ones(VARIABLE_COUNT, dtype=np.float64),
        ),
        constraints=constraints,
        options={
            "presolve": True,
            "time_limit": 60.0,
            "mip_rel_gap": 0.0,
        },
    )
    if not result.success or result.x is None:
        # Numerical fallback is deterministic and fail-closed.
        result = milp(
            c=np.asarray(objective, dtype=np.float64),
            integrality=np.ones(VARIABLE_COUNT, dtype=np.uint8),
            bounds=Bounds(
                np.zeros(VARIABLE_COUNT, dtype=np.float64),
                np.ones(VARIABLE_COUNT, dtype=np.float64),
            ),
            constraints=constraints,
            options={
                "presolve": False,
                "time_limit": 60.0,
                "mip_rel_gap": 0.0,
            },
        )
    if not result.success or result.x is None:
        raise EPLRDecodeError(
            f"MILP_FAILED: success={result.success}, "
            f"status={result.status}, message={result.message!r}"
        )

    maximum_error = float(
        np.max(np.abs(result.x - np.rint(result.x)))
    )
    if maximum_error > INTEGRALITY_TOLERANCE:
        raise EPLRDecodeError(
            "MILP_INTEGRALITY_FAILURE: "
            f"maximum_error={maximum_error}, "
            f"tolerance={INTEGRALITY_TOLERANCE}"
        )
    return np.rint(result.x).astype(np.uint8)


def _freeze_objective(
    constraints: list[LinearConstraint],
    objective: np.ndarray,
    optimum: float,
    tolerance: float,
) -> list[LinearConstraint]:
    row = csc_array(
        objective.reshape(1, -1),
        dtype=np.float64,
    )
    delta = float(tolerance) * max(1.0, abs(float(optimum)))
    return constraints + [
        LinearConstraint(
            row,
            np.asarray([optimum - delta], dtype=np.float64),
            np.asarray([optimum + delta], dtype=np.float64),
        )
    ]


def _feasible_with_extra_rows(
    constraints: list[LinearConstraint],
    extra_rows: list[dict[int, float]],
    extra_lb: list[float],
    extra_ub: list[float],
) -> bool:
    extra = _rows_to_constraint(extra_rows, extra_lb, extra_ub)
    try:
        _solve_milp(
            np.zeros(VARIABLE_COUNT, dtype=np.float64),
            constraints + [extra],
        )
        return True
    except EPLRDecodeError:
        return False


def _lexicographically_smallest_route_set(
    constraints: list[LinearConstraint],
    k: int,
) -> tuple[int, ...]:
    fixed_rows: list[dict[int, float]] = []
    fixed_lb: list[float] = []
    fixed_ub: list[float] = []
    selected: list[int] = []
    lower_route_id = 0

    for _position in range(k):
        low = lower_route_id
        high = routes.ORDERED_ROUTE_COUNT - 1

        while low < high:
            middle = (low + high) // 2
            candidate_rows = list(fixed_rows)
            candidate_lb = list(fixed_lb)
            candidate_ub = list(fixed_ub)
            candidate_rows.append(
                {
                    Q_OFFSET + route_id: 1.0
                    for route_id in range(lower_route_id, middle + 1)
                }
            )
            candidate_lb.append(1.0)
            candidate_ub.append(np.inf)
            if _feasible_with_extra_rows(
                constraints,
                candidate_rows,
                candidate_lb,
                candidate_ub,
            ):
                high = middle
            else:
                low = middle + 1

        chosen = low
        for route_id in range(lower_route_id, chosen):
            fixed_rows.append({Q_OFFSET + route_id: 1.0})
            fixed_lb.append(0.0)
            fixed_ub.append(0.0)
        fixed_rows.append({Q_OFFSET + chosen: 1.0})
        fixed_lb.append(1.0)
        fixed_ub.append(1.0)

        if not _feasible_with_extra_rows(
            constraints,
            fixed_rows,
            fixed_lb,
            fixed_ub,
        ):
            raise EPLRDecodeError(
                f"LEXICOGRAPHIC_ROUTE_FREEZE_FAILED at route {chosen}"
            )

        selected.append(chosen)
        lower_route_id = chosen + 1

    return tuple(selected)


def repair_endpoints_with_milp(
    *,
    raw_source_mask: int,
    raw_victim_mask: int,
    k: int,
    source_probabilities: Sequence[float],
    victim_probabilities: Sequence[float],
    transit_probabilities: Sequence[float],
    path_probabilities: Sequence[float],
    config: EPLRConfig,
) -> tuple[Any, float]:
    base = _base_repair_constraints(k)
    constraints = [base]

    endpoint_objective = (
        _binary_cost_coefficients(source_probabilities, S_OFFSET)
        + _binary_cost_coefficients(victim_probabilities, V_OFFSET)
    )
    solution = _solve_milp(endpoint_objective, constraints)
    endpoint_optimum = float(endpoint_objective @ solution)
    constraints = _freeze_objective(
        constraints,
        endpoint_objective,
        endpoint_optimum,
        config.objective_freeze_tolerance,
    )

    hamming_objective = (
        _hamming_coefficients(raw_source_mask, S_OFFSET)
        + _hamming_coefficients(raw_victim_mask, V_OFFSET)
    )
    solution = _solve_milp(hamming_objective, constraints)
    hamming_optimum = float(hamming_objective @ solution)
    constraints = _freeze_objective(
        constraints,
        hamming_objective,
        hamming_optimum,
        config.objective_freeze_tolerance,
    )

    support_objective = (
        _negative_support_coefficients(
            transit_probabilities,
            T_OFFSET,
            config.transit_weight,
        )
        + _negative_support_coefficients(
            path_probabilities,
            P_OFFSET,
            config.path_weight,
        )
    )
    solution = _solve_milp(support_objective, constraints)
    support_optimum = float(support_objective @ solution)
    constraints = _freeze_objective(
        constraints,
        support_objective,
        support_optimum,
        config.objective_freeze_tolerance,
    )

    route_ids = _lexicographically_smallest_route_set(
        constraints,
        k,
    )
    hypothesis = routes.combine_route_ids(
        route_ids,
        expected_k=k,
    )
    score = route_support_score(
        hypothesis,
        transit_probabilities,
        path_probabilities,
        config,
    )
    return hypothesis, score


def decode_eplr_v1(
    *,
    graph_logit: float,
    count_logits: Sequence[float],
    source_logits: Sequence[float],
    transit_logits: Sequence[float],
    victim_logits: Sequence[float],
    path_logits: Sequence[float],
    config: EPLRConfig,
) -> EPLRResult:
    count_logits = np.asarray(count_logits, dtype=np.float64).reshape(4)
    source_probabilities = stable_sigmoid_array(source_logits).reshape(16)
    transit_probabilities = stable_sigmoid_array(transit_logits).reshape(16)
    victim_probabilities = stable_sigmoid_array(victim_logits).reshape(16)
    path_probabilities = stable_sigmoid_array(path_logits).reshape(16)

    graph_probability = stable_sigmoid(float(graph_logit))
    graph_prediction = int(
        graph_probability >= config.graph_threshold
    )
    raw_count = int(np.argmax(count_logits)) + 1

    raw_source_mask = mask_from_probabilities(
        source_probabilities,
        config.endpoint_threshold,
    )
    raw_transit_mask = mask_from_probabilities(
        transit_probabilities,
        config.endpoint_threshold,
    )
    raw_victim_mask = mask_from_probabilities(
        victim_probabilities,
        config.endpoint_threshold,
    )
    raw_path_mask = mask_from_probabilities(
        path_probabilities,
        config.endpoint_threshold,
    )

    if graph_prediction == 0:
        return EPLRResult(
            raw_graph_probability=graph_probability,
            raw_graph_prediction=0,
            raw_attacker_count_candidate=raw_count,
            effective_attacker_count=0,
            raw_source_bitmap=raw_source_mask,
            raw_transit_bitmap=raw_transit_mask,
            raw_victim_bitmap=raw_victim_mask,
            raw_path_bitmap=raw_path_mask,
            decoded_source_bitmap=0,
            decoded_transit_bitmap=0,
            decoded_victim_bitmap=0,
            decoded_path_bitmap=0,
            selected_route_ids=(),
            route_consistency_score=float("-inf"),
            source_repair_count=0,
            victim_repair_count=0,
            endpoint_preserved_but_low_route_support=False,
            decoder_status=STATUS_INACTIVE,
            repair_solver_used=False,
        )

    hypothesis, support = choose_endpoint_preserving_hypothesis(
        source_mask=raw_source_mask,
        victim_mask=raw_victim_mask,
        k=raw_count,
        transit_probabilities=transit_probabilities,
        path_probabilities=path_probabilities,
        config=config,
    )
    repair_solver_used = False
    endpoint_repaired = False

    if hypothesis is None:
        repair_solver_used = True
        endpoint_repaired = True
        try:
            hypothesis, support = repair_endpoints_with_milp(
                raw_source_mask=raw_source_mask,
                raw_victim_mask=raw_victim_mask,
                k=raw_count,
                source_probabilities=source_probabilities,
                victim_probabilities=victim_probabilities,
                transit_probabilities=transit_probabilities,
                path_probabilities=path_probabilities,
                config=config,
            )
        except Exception:
            return EPLRResult(
                raw_graph_probability=graph_probability,
                raw_graph_prediction=1,
                raw_attacker_count_candidate=raw_count,
                effective_attacker_count=raw_count,
                raw_source_bitmap=raw_source_mask,
                raw_transit_bitmap=raw_transit_mask,
                raw_victim_bitmap=raw_victim_mask,
                raw_path_bitmap=raw_path_mask,
                decoded_source_bitmap=0,
                decoded_transit_bitmap=0,
                decoded_victim_bitmap=0,
                decoded_path_bitmap=0,
                selected_route_ids=(),
                route_consistency_score=float("-inf"),
                source_repair_count=0,
                victim_repair_count=0,
                endpoint_preserved_but_low_route_support=False,
                decoder_status=STATUS_NO_SOLUTION,
                repair_solver_used=True,
            )

    if hypothesis.attacker_count != raw_count:
        raise EPLRDecodeError("decoder changed immutable Raw count")

    low_threshold = config.low_support_threshold_by_k[raw_count - 1]
    low_support = bool(support < low_threshold)

    if endpoint_repaired:
        status = STATUS_REPAIRED
    elif low_support:
        status = STATUS_LEGAL_LOW_SUPPORT
    else:
        status = STATUS_LEGAL

    return EPLRResult(
        raw_graph_probability=graph_probability,
        raw_graph_prediction=1,
        raw_attacker_count_candidate=raw_count,
        effective_attacker_count=raw_count,
        raw_source_bitmap=raw_source_mask,
        raw_transit_bitmap=raw_transit_mask,
        raw_victim_bitmap=raw_victim_mask,
        raw_path_bitmap=raw_path_mask,
        decoded_source_bitmap=int(hypothesis.source_mask),
        decoded_transit_bitmap=int(hypothesis.transit_mask),
        decoded_victim_bitmap=int(hypothesis.victim_mask),
        decoded_path_bitmap=int(hypothesis.path_mask),
        selected_route_ids=tuple(hypothesis.route_ids),
        route_consistency_score=float(support),
        source_repair_count=hamming_distance(
            raw_source_mask,
            int(hypothesis.source_mask),
        ),
        victim_repair_count=hamming_distance(
            raw_victim_mask,
            int(hypothesis.victim_mask),
        ),
        endpoint_preserved_but_low_route_support=low_support,
        decoder_status=status,
        repair_solver_used=repair_solver_used,
    )


def bitmap_to_array(mask: int) -> np.ndarray:
    return np.asarray(
        [
            1 if int(mask) & (1 << index) else 0
            for index in range(routes.ROUTER_COUNT)
        ],
        dtype=np.uint8,
    )
