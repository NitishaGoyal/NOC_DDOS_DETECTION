"""Exact deterministic legal-route decoder for V5 P2.

The decoder implements the protocol frozen by V5 P2 L0 and consumes the
240-route table frozen by L1. It uses a two-phase binary MILP:

1. maximize the structured attack-hypothesis score exactly;
2. restrict the solution to the semantic optimum face and minimize the frozen
   tie-break code: lower K, then lexicographically lower route-ID tuple.

No time limit, node limit, candidate pruning, or beam approximation is
permitted. The solver is requested to use a zero relative MIP gap. A residual
reported gap no larger than the frozen float64 numerical-zero tolerance is
accepted only when HiGHS returns optimal status. The semantic tie tolerance remains exactly 1e-12. Semantic-face membership
is evaluated with Decimal arithmetic over the exact float64 objective
coefficients. Any lexicographic candidate outside that high-precision face is
excluded with a deterministic no-good cut and the lexicographic MILP is solved
again. No validation-derived tolerance expansion is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from functools import lru_cache
import math
import os
import platform
import warnings
from typing import Sequence

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack
from threadpoolctl import threadpool_limits

from src.decoders.v5_xy_route_library import (
    ROUTE_TABLE,
    HypothesisMasks,
    RouteRecord,
    combine_route_ids,
    empty_hypothesis,
)

ALPHA_GRAPH = 1.0
ALPHA_COUNT = 0.5
ALPHA_SOURCE = 1.0
ALPHA_TRANSIT = 0.5
ALPHA_VICTIM = 0.75
ALPHA_PATH = 0.5

ROUTER_COUNT = 16
K_VALUES = (1, 2, 3, 4)
K_MAX = 4
ROUTE_ID_BASE = 241
SEMANTIC_TIE_TOLERANCE = 1e-12
MIP_GAP_REPORTING_ULP_MULTIPLIER = 64
MIP_GAP_REPORTING_TOLERANCE = (
    MIP_GAP_REPORTING_ULP_MULTIPLIER * np.finfo(np.float64).eps
)
FACE_SCORE_ACCUMULATION_ULP_MULTIPLIER = 1024
FACE_SCORE_FORWARD_ERROR_SAFETY_FACTOR = 2.0
FACE_SCORE_NUMERICAL_ALLOWANCE_CAP = SEMANTIC_TIE_TOLERANCE
HIGH_PRECISION_DECIMAL_DIGITS = 80
MAX_HIGH_PRECISION_FACE_REJECTIONS = 1024

SUPPORTED_MILP_OPTIONS = {
    "disp": False,
    "presolve": True,
    "mip_rel_gap": 0.0,
}


class ExactDecoderError(RuntimeError):
    """Raised whenever exactness or deterministic tie-breaking is not proven."""


@dataclass(frozen=True, slots=True)
class SolverIdentity:
    python_version: str
    numpy_version: str
    scipy_version: str
    highs_version: str
    backend: str


@dataclass(frozen=True, slots=True)
class ExactAttackHypothesis:
    margin: float
    attacker_count: int
    route_ids: tuple[int, ...]
    source_mask: int
    transit_mask: int
    victim_mask: int
    path_mask: int
    graph_contribution: float
    count_contribution: float
    source_contribution: float
    transit_contribution: float
    victim_contribution: float
    path_contribution: float
    primary_mip_gap: float
    lexicographic_mip_gap: float
    primary_mip_node_count: int
    lexicographic_mip_node_count: int
    primary_status: int
    lexicographic_status: int
    optimality_proven: bool
    lexicographic_tie_break_certified: bool
    semantic_face_score_difference: float
    semantic_face_numerical_allowance: float
    semantic_face_certification_tolerance: float
    high_precision_primary_variable_score: str
    high_precision_selected_variable_score: str
    high_precision_semantic_face_difference: str
    high_precision_semantic_face_certified: bool
    lexicographic_candidates_rejected_by_high_precision: int
    solver_identity: SolverIdentity


@dataclass(frozen=True, slots=True)
class DecodedStructuredOutput:
    attack: int
    margin: float
    attacker_count: int
    route_ids: tuple[int, ...]
    source_mask: int
    transit_mask: int
    victim_mask: int
    path_mask: int


@dataclass(frozen=True, slots=True)
class _Model:
    route_records: tuple[RouteRecord, ...]
    route_count: int
    variable_count: int
    z_offset: int
    active_offset: int
    count_class_offset: int
    source_union_offset: int
    transit_union_offset: int
    victim_union_offset: int
    path_union_offset: int
    matrix: csr_matrix
    lower: np.ndarray
    upper: np.ndarray
    bounds: Bounds
    integrality: np.ndarray
    lexicographic_objective: np.ndarray


def solver_identity() -> SolverIdentity:
    try:
        from scipy.optimize._highspy import _core as highs_core

        highs_version = (
            f"{highs_core.HIGHS_VERSION_MAJOR}."
            f"{highs_core.HIGHS_VERSION_MINOR}."
            f"{highs_core.HIGHS_VERSION_PATCH}"
        )
    except Exception as exc:  # pragma: no cover - fail closed in freezer/tests
        raise ExactDecoderError(
            "Unable to resolve the embedded HiGHS version"
        ) from exc

    return SolverIdentity(
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        highs_version=highs_version,
        backend="scipy.optimize.milp_with_embedded_HiGHS",
    )


def _validate_vector(name: str, value: Sequence[float], size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _validate_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _log_softmax(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    normalizer = maximum + math.log(float(np.exp(values - maximum).sum()))
    return values - normalizer


def _append_row(
    rows: list[tuple[list[int], list[float]]],
    lower: list[float],
    upper: list[float],
    indices: list[int],
    coefficients: list[float],
    row_lower: float,
    row_upper: float,
) -> None:
    rows.append((indices, coefficients))
    lower.append(row_lower)
    upper.append(row_upper)


@lru_cache(maxsize=16)
def _build_model(route_records: tuple[RouteRecord, ...]) -> _Model:
    if not route_records:
        raise ValueError("route_records must not be empty")

    route_ids = [record.route_id for record in route_records]
    if route_ids != sorted(route_ids):
        raise ValueError("route_records must be ordered by ascending route_id")
    if len(set(route_ids)) != len(route_ids):
        raise ValueError("route_records contain duplicate route IDs")

    route_count = len(route_records)
    z_offset = 0
    active_offset = K_MAX * route_count
    count_class_offset = active_offset + K_MAX
    source_union_offset = count_class_offset + len(K_VALUES)
    transit_union_offset = source_union_offset + ROUTER_COUNT
    victim_union_offset = transit_union_offset + ROUTER_COUNT
    path_union_offset = victim_union_offset + ROUTER_COUNT
    variable_count = path_union_offset + ROUTER_COUNT

    rows: list[tuple[list[int], list[float]]] = []
    lower: list[float] = []
    upper: list[float] = []

    # Each active route position selects exactly one route.
    for position in range(K_MAX):
        indices = [
            z_offset + position * route_count + route_index
            for route_index in range(route_count)
        ]
        coefficients = [1.0] * route_count
        indices.append(active_offset + position)
        coefficients.append(-1.0)
        _append_row(rows, lower, upper, indices, coefficients, 0.0, 0.0)

    # Active positions form a prefix.
    for position in range(K_MAX - 1):
        _append_row(
            rows,
            lower,
            upper,
            [active_offset + position, active_offset + position + 1],
            [-1.0, 1.0],
            -np.inf,
            0.0,
        )

    # Exactly one count class K1-K4.
    _append_row(
        rows,
        lower,
        upper,
        [count_class_offset + index for index in range(len(K_VALUES))],
        [1.0] * len(K_VALUES),
        1.0,
        1.0,
    )

    # Number of active positions equals the chosen K.
    _append_row(
        rows,
        lower,
        upper,
        [active_offset + position for position in range(K_MAX)]
        + [count_class_offset + index for index in range(len(K_VALUES))],
        [1.0] * K_MAX + [-float(k) for k in K_VALUES],
        0.0,
        0.0,
    )

    # Route IDs in active positions must be strictly increasing.
    maximum_route_id = max(route_ids)
    big_m = float(maximum_route_id + 1)
    for position in range(K_MAX - 1):
        indices: list[int] = []
        coefficients: list[float] = []

        for route_index, record in enumerate(route_records):
            indices.append(z_offset + position * route_count + route_index)
            coefficients.append(float(record.route_id))

        for route_index, record in enumerate(route_records):
            indices.append(z_offset + (position + 1) * route_count + route_index)
            coefficients.append(-float(record.route_id))

        indices.append(active_offset + position + 1)
        coefficients.append(big_m)

        _append_row(
            rows,
            lower,
            upper,
            indices,
            coefficients,
            -np.inf,
            big_m - 1.0,
        )

    # At most one route per source.
    sources = sorted({record.source for record in route_records})
    for source in sources:
        matching = [
            route_index
            for route_index, record in enumerate(route_records)
            if record.source == source
        ]
        indices = [
            z_offset + position * route_count + route_index
            for position in range(K_MAX)
            for route_index in matching
        ]
        _append_row(
            rows,
            lower,
            upper,
            indices,
            [1.0] * len(indices),
            -np.inf,
            1.0,
        )

    role_specs = (
        ("source_mask", source_union_offset),
        ("transit_mask", transit_union_offset),
        ("victim_mask", victim_union_offset),
        ("path_mask", path_union_offset),
    )

    # Exact binary OR constraints for all four role unions.
    for mask_attribute, union_offset in role_specs:
        for router_id in range(ROUTER_COUNT):
            matching = [
                route_index
                for route_index, record in enumerate(route_records)
                if (int(getattr(record, mask_attribute)) >> router_id) & 1
            ]

            z_indices = [
                z_offset + position * route_count + route_index
                for position in range(K_MAX)
                for route_index in matching
            ]
            y_index = union_offset + router_id

            # y <= sum(z)
            _append_row(
                rows,
                lower,
                upper,
                z_indices + [y_index],
                [-1.0] * len(z_indices) + [1.0],
                -np.inf,
                0.0,
            )

            # sum(z) <= 4y
            _append_row(
                rows,
                lower,
                upper,
                z_indices + [y_index],
                [1.0] * len(z_indices) + [-float(K_MAX)],
                -np.inf,
                0.0,
            )

    row_indices: list[int] = []
    column_indices: list[int] = []
    data: list[float] = []

    for row_index, (indices, coefficients) in enumerate(rows):
        row_indices.extend([row_index] * len(indices))
        column_indices.extend(indices)
        data.extend(coefficients)

    matrix = coo_matrix(
        (data, (row_indices, column_indices)),
        shape=(len(rows), variable_count),
        dtype=np.float64,
    ).tocsr()

    lexicographic_objective = np.zeros(variable_count, dtype=np.float64)

    # K dominates every route-tuple digit.
    lexicographic_objective[
        active_offset : active_offset + K_MAX
    ] = float(ROUTE_ID_BASE ** K_MAX)

    for position in range(K_MAX):
        positional_weight = float(ROUTE_ID_BASE ** (K_MAX - 1 - position))
        for route_index, record in enumerate(route_records):
            lexicographic_objective[
                z_offset + position * route_count + route_index
            ] = positional_weight * float(record.route_id)

    return _Model(
        route_records=route_records,
        route_count=route_count,
        variable_count=variable_count,
        z_offset=z_offset,
        active_offset=active_offset,
        count_class_offset=count_class_offset,
        source_union_offset=source_union_offset,
        transit_union_offset=transit_union_offset,
        victim_union_offset=victim_union_offset,
        path_union_offset=path_union_offset,
        matrix=matrix,
        lower=np.asarray(lower, dtype=np.float64),
        upper=np.asarray(upper, dtype=np.float64),
        bounds=Bounds(
            np.zeros(variable_count, dtype=np.float64),
            np.ones(variable_count, dtype=np.float64),
        ),
        integrality=np.ones(variable_count, dtype=np.uint8),
        lexicographic_objective=lexicographic_objective,
    )


def mip_gap_is_numerical_zero(mip_gap: float) -> bool:
    """Return whether a solver-reported relative gap is float64 numerical zero.

    This does not change the requested HiGHS gap (0.0), the semantic objective,
    or the 1e-12 semantic tie tolerance. It only prevents a machine-roundoff
    residual from being misclassified as a material optimization gap after the
    solver has returned optimal status.
    """
    value = float(mip_gap)
    return (
        math.isfinite(value)
        and value >= 0.0
        and value <= MIP_GAP_REPORTING_TOLERANCE
    )


def semantic_face_numerical_allowance(
    primary_objective: Sequence[float],
    primary_variable_score: float,
    selected_variable_score: float,
) -> float:
    """Return a bounded float64 allowance for optimum-face score checking.

    The 1e-12 semantic tie tolerance is not changed. This allowance covers only
    finite-precision disagreement between the solver objective evaluation and
    the independently recomputed Python score. It is derived from the active
    objective coefficient count and magnitude, has a 1024-ULP accumulation
    floor, and is hard-capped at the semantic tie tolerance.
    """
    objective = np.asarray(primary_objective, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(objective)):
        raise ValueError("primary_objective contains a non-finite value")

    primary_score = float(primary_variable_score)
    selected_score = float(selected_variable_score)
    if not math.isfinite(primary_score) or not math.isfinite(selected_score):
        raise ValueError("semantic face scores must be finite")

    active = objective[np.flatnonzero(objective)]
    operation_count = max(1, int(active.size))
    eps = float(np.finfo(np.float64).eps)
    gamma = (operation_count * eps) / (1.0 - operation_count * eps)
    coefficient_l1 = float(np.abs(active).sum())
    scale = max(
        1.0,
        coefficient_l1,
        abs(primary_score),
        abs(selected_score),
    )

    forward_error_bound = (
        FACE_SCORE_FORWARD_ERROR_SAFETY_FACTOR * gamma * scale
    )
    accumulation_floor = (
        FACE_SCORE_ACCUMULATION_ULP_MULTIPLIER * eps * scale
    )
    allowance = max(forward_error_bound, accumulation_floor)
    return float(min(FACE_SCORE_NUMERICAL_ALLOWANCE_CAP, allowance))


def semantic_face_difference_is_certified(
    score_difference: float,
    numerical_allowance: float,
) -> bool:
    difference = float(score_difference)
    allowance = float(numerical_allowance)
    return (
        math.isfinite(difference)
        and difference >= 0.0
        and math.isfinite(allowance)
        and allowance >= 0.0
        and allowance <= FACE_SCORE_NUMERICAL_ALLOWANCE_CAP
        and difference <= SEMANTIC_TIE_TOLERANCE + allowance
    )


def _check_optimal_result(name: str, result: object) -> None:
    success = bool(getattr(result, "success", False))
    status = int(getattr(result, "status", -1))
    mip_gap = float(getattr(result, "mip_gap", np.inf))

    if not success or status != 0:
        raise ExactDecoderError(
            f"{name} MILP did not terminate optimally: "
            f"status={status}, message={getattr(result, 'message', None)!r}"
        )

    if not mip_gap_is_numerical_zero(mip_gap):
        raise ExactDecoderError(
            f"{name} MILP has material or invalid MIP gap: {mip_gap!r}; "
            f"float64 numerical-zero tolerance="
            f"{MIP_GAP_REPORTING_TOLERANCE:.17g}"
        )

    if getattr(result, "x", None) is None or getattr(result, "fun", None) is None:
        raise ExactDecoderError(f"{name} MILP returned no solution")


def _solve_milp(
    objective: np.ndarray,
    model: _Model,
    matrix: csr_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
) -> object:
    # threadpool_limits protects BLAS/OpenMP-backed libraries used by the process.
    with threadpool_limits(limits=1):
        return milp(
            c=objective,
            integrality=model.integrality,
            bounds=model.bounds,
            constraints=LinearConstraint(matrix, lower, upper),
            options=SUPPORTED_MILP_OPTIONS,
        )


def _extract_route_ids(model: _Model, solution: np.ndarray) -> tuple[int, ...]:
    active = np.rint(
        solution[model.active_offset : model.active_offset + K_MAX]
    ).astype(np.int64)
    attacker_count = int(active.sum())

    if attacker_count not in K_VALUES:
        raise ExactDecoderError(
            f"solver produced invalid attacker count: {attacker_count}"
        )

    route_ids: list[int] = []
    for position in range(attacker_count):
        start = model.z_offset + position * model.route_count
        stop = start + model.route_count
        selected = np.flatnonzero(np.rint(solution[start:stop]).astype(np.int64))
        if selected.size != 1:
            raise ExactDecoderError(
                f"route position {position} selected {selected.size} routes"
            )
        route_ids.append(model.route_records[int(selected[0])].route_id)

    route_tuple = tuple(route_ids)
    if route_tuple != tuple(sorted(route_tuple)):
        raise ExactDecoderError("selected route IDs are not strictly ordered")

    return route_tuple


def _mask_contribution(logits: np.ndarray, mask: int, weight: float) -> float:
    total = 0.0
    for router_id in range(ROUTER_COUNT):
        if (mask >> router_id) & 1:
            total += weight * float(logits[router_id])
    return total


def _score_route_ids(
    graph_logit: float,
    count_log_probabilities: np.ndarray,
    source_logits: np.ndarray,
    transit_logits: np.ndarray,
    victim_logits: np.ndarray,
    path_logits: np.ndarray,
    route_ids: tuple[int, ...],
) -> tuple[float, HypothesisMasks, tuple[float, float, float, float, float, float]]:
    masks = combine_route_ids(route_ids, expected_k=len(route_ids))
    k = masks.attacker_count

    graph_contribution = ALPHA_GRAPH * graph_logit
    count_contribution = ALPHA_COUNT * float(count_log_probabilities[k - 1])
    source_contribution = _mask_contribution(
        source_logits, masks.source_mask, ALPHA_SOURCE
    )
    transit_contribution = _mask_contribution(
        transit_logits, masks.transit_mask, ALPHA_TRANSIT
    )
    victim_contribution = _mask_contribution(
        victim_logits, masks.victim_mask, ALPHA_VICTIM
    )
    path_contribution = _mask_contribution(
        path_logits, masks.path_mask, ALPHA_PATH
    )

    margin = (
        graph_contribution
        + count_contribution
        + source_contribution
        + transit_contribution
        + victim_contribution
        + path_contribution
    )

    return (
        margin,
        masks,
        (
            graph_contribution,
            count_contribution,
            source_contribution,
            transit_contribution,
            victim_contribution,
            path_contribution,
        ),
    )



def _decimal_from_float(value: float) -> Decimal:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError("high-precision score input must be finite")
    return Decimal.from_float(scalar)


def _decimal_mask_contribution(
    logits: np.ndarray,
    mask: int,
    weight: float,
) -> Decimal:
    total = Decimal(0)
    decimal_weight = _decimal_from_float(weight)
    for router_id in range(ROUTER_COUNT):
        if (mask >> router_id) & 1:
            total += decimal_weight * _decimal_from_float(logits[router_id])
    return total


def _high_precision_variable_score(
    count_log_probabilities: np.ndarray,
    source_logits: np.ndarray,
    transit_logits: np.ndarray,
    victim_logits: np.ndarray,
    path_logits: np.ndarray,
    route_ids: tuple[int, ...],
) -> Decimal:
    """Evaluate the frozen float64 semantic coefficients using Decimal.

    Decimal.from_float preserves each binary64 coefficient exactly. Therefore,
    this removes summation-order disagreement without redefining the neural
    logits, weights, count log-probabilities, or 1e-12 semantic tolerance.
    """
    masks = combine_route_ids(route_ids, expected_k=len(route_ids))
    k = masks.attacker_count
    with localcontext() as context:
        context.prec = HIGH_PRECISION_DECIMAL_DIGITS
        return (
            _decimal_from_float(ALPHA_COUNT)
            * _decimal_from_float(count_log_probabilities[k - 1])
            + _decimal_mask_contribution(
                source_logits, masks.source_mask, ALPHA_SOURCE
            )
            + _decimal_mask_contribution(
                transit_logits, masks.transit_mask, ALPHA_TRANSIT
            )
            + _decimal_mask_contribution(
                victim_logits, masks.victim_mask, ALPHA_VICTIM
            )
            + _decimal_mask_contribution(
                path_logits, masks.path_mask, ALPHA_PATH
            )
        )


def high_precision_semantic_face_difference(
    primary_variable_score: Decimal,
    selected_variable_score: Decimal,
) -> Decimal:
    with localcontext() as context:
        context.prec = HIGH_PRECISION_DECIMAL_DIGITS
        return abs(selected_variable_score - primary_variable_score)


def high_precision_semantic_face_is_certified(
    difference: Decimal,
) -> bool:
    if not isinstance(difference, Decimal) or not difference.is_finite():
        return False
    return (
        difference >= Decimal(0)
        and difference <= _decimal_from_float(SEMANTIC_TIE_TOLERANCE)
    )


def _route_tuple_nogood_row(
    model: _Model,
    route_ids: tuple[int, ...],
) -> tuple[csr_matrix, float, float]:
    """Exclude exactly one ordered route tuple from a later lexicographic MILP."""
    route_index_by_id = {
        record.route_id: index
        for index, record in enumerate(model.route_records)
    }
    columns: list[int] = []
    for position, route_id in enumerate(route_ids):
        if route_id not in route_index_by_id:
            raise ExactDecoderError(f"unknown route ID in no-good cut: {route_id}")
        columns.append(
            model.z_offset
            + position * model.route_count
            + route_index_by_id[route_id]
        )
    columns.append(model.count_class_offset + len(route_ids) - 1)
    row = coo_matrix(
        (
            np.ones(len(columns), dtype=np.float64),
            (np.zeros(len(columns), dtype=np.int64), np.asarray(columns)),
        ),
        shape=(1, model.variable_count),
        dtype=np.float64,
    ).tocsr()
    # The exact tuple has K selected route-position variables plus its K-class
    # variable, for a sum of K+1. A longer tuple with the same prefix has a
    # different count-class variable and remains feasible.
    return row, -np.inf, float(len(route_ids))


def _decode_with_routes(
    graph_logit: float,
    count_logits: Sequence[float],
    source_logits: Sequence[float],
    transit_logits: Sequence[float],
    victim_logits: Sequence[float],
    path_logits: Sequence[float],
    route_records: tuple[RouteRecord, ...],
) -> ExactAttackHypothesis:
    graph = _validate_scalar("graph_logit", graph_logit)
    count = _validate_vector("count_logits", count_logits, 4)
    source = _validate_vector("source_logits", source_logits, 16)
    transit = _validate_vector("transit_logits", transit_logits, 16)
    victim = _validate_vector("victim_logits", victim_logits, 16)
    path = _validate_vector("path_logits", path_logits, 16)

    count_log_probabilities = _log_softmax(count)
    model = _build_model(route_records)

    primary_objective = np.zeros(model.variable_count, dtype=np.float64)
    primary_objective[
        model.count_class_offset : model.count_class_offset + 4
    ] = -ALPHA_COUNT * count_log_probabilities
    primary_objective[
        model.source_union_offset : model.source_union_offset + 16
    ] = -ALPHA_SOURCE * source
    primary_objective[
        model.transit_union_offset : model.transit_union_offset + 16
    ] = -ALPHA_TRANSIT * transit
    primary_objective[
        model.victim_union_offset : model.victim_union_offset + 16
    ] = -ALPHA_VICTIM * victim
    primary_objective[
        model.path_union_offset : model.path_union_offset + 16
    ] = -ALPHA_PATH * path

    primary = _solve_milp(
        primary_objective,
        model,
        model.matrix,
        model.lower,
        model.upper,
    )
    _check_optimal_result("primary", primary)

    primary_fun = float(primary.fun)

    nonzero = np.flatnonzero(primary_objective)
    optimum_row = coo_matrix(
        (
            primary_objective[nonzero],
            ([0] * int(nonzero.size), nonzero),
        ),
        shape=(1, model.variable_count),
        dtype=np.float64,
    ).tocsr()

    lex_matrix = vstack([model.matrix, optimum_row], format="csr")
    lex_lower = np.concatenate(
        [
            model.lower,
            np.asarray([primary_fun - SEMANTIC_TIE_TOLERANCE], dtype=np.float64),
        ]
    )
    lex_upper = np.concatenate(
        [
            model.upper,
            np.asarray([primary_fun + SEMANTIC_TIE_TOLERANCE], dtype=np.float64),
        ]
    )

    primary_route_ids = _extract_route_ids(
        model,
        np.asarray(primary.x, dtype=np.float64),
    )
    primary_high_precision_score = _high_precision_variable_score(
        count_log_probabilities,
        source,
        transit,
        victim,
        path,
        primary_route_ids,
    )

    current_lex_matrix = lex_matrix
    current_lex_lower = lex_lower
    current_lex_upper = lex_upper
    rejected_candidates = 0

    while True:
        lexicographic = _solve_milp(
            model.lexicographic_objective,
            model,
            current_lex_matrix,
            current_lex_lower,
            current_lex_upper,
        )
        _check_optimal_result("lexicographic", lexicographic)

        route_ids = _extract_route_ids(
            model,
            np.asarray(lexicographic.x, dtype=np.float64),
        )
        (
            margin,
            masks,
            contributions,
        ) = _score_route_ids(
            graph,
            count_log_probabilities,
            source,
            transit,
            victim,
            path,
            route_ids,
        )

        selected_high_precision_score = _high_precision_variable_score(
            count_log_probabilities,
            source,
            transit,
            victim,
            path,
            route_ids,
        )
        high_precision_difference = high_precision_semantic_face_difference(
            primary_high_precision_score,
            selected_high_precision_score,
        )
        if high_precision_semantic_face_is_certified(
            high_precision_difference
        ):
            break

        rejected_candidates += 1
        if rejected_candidates > MAX_HIGH_PRECISION_FACE_REJECTIONS:
            raise ExactDecoderError(
                "High-precision semantic-face filtering exceeded the frozen "
                f"limit of {MAX_HIGH_PRECISION_FACE_REJECTIONS} rejected "
                "lexicographic candidates"
            )

        nogood_row, nogood_lower, nogood_upper = _route_tuple_nogood_row(
            model,
            route_ids,
        )
        current_lex_matrix = vstack(
            [current_lex_matrix, nogood_row],
            format="csr",
        )
        current_lex_lower = np.concatenate(
            [
                current_lex_lower,
                np.asarray([nogood_lower], dtype=np.float64),
            ]
        )
        current_lex_upper = np.concatenate(
            [
                current_lex_upper,
                np.asarray([nogood_upper], dtype=np.float64),
            ]
        )

    # Retain the legacy float64 diagnostic fields for provenance, but exact
    # acceptance is now based exclusively on the Decimal evaluation above.
    selected_variable_score = margin - ALPHA_GRAPH * graph
    primary_variable_score = -primary_fun
    score_difference = abs(selected_variable_score - primary_variable_score)
    face_numerical_allowance = semantic_face_numerical_allowance(
        primary_objective,
        primary_variable_score,
        selected_variable_score,
    )
    face_certification_tolerance = (
        SEMANTIC_TIE_TOLERANCE + face_numerical_allowance
    )

    graph_c, count_c, source_c, transit_c, victim_c, path_c = contributions

    return ExactAttackHypothesis(
        margin=float(margin),
        attacker_count=masks.attacker_count,
        route_ids=route_ids,
        source_mask=masks.source_mask,
        transit_mask=masks.transit_mask,
        victim_mask=masks.victim_mask,
        path_mask=masks.path_mask,
        graph_contribution=graph_c,
        count_contribution=count_c,
        source_contribution=source_c,
        transit_contribution=transit_c,
        victim_contribution=victim_c,
        path_contribution=path_c,
        primary_mip_gap=float(primary.mip_gap),
        lexicographic_mip_gap=float(lexicographic.mip_gap),
        primary_mip_node_count=int(getattr(primary, "mip_node_count", 0)),
        lexicographic_mip_node_count=int(
            getattr(lexicographic, "mip_node_count", 0)
        ),
        primary_status=int(primary.status),
        lexicographic_status=int(lexicographic.status),
        optimality_proven=True,
        lexicographic_tie_break_certified=True,
        semantic_face_score_difference=float(score_difference),
        semantic_face_numerical_allowance=float(face_numerical_allowance),
        semantic_face_certification_tolerance=float(
            face_certification_tolerance
        ),
        high_precision_primary_variable_score=str(
            primary_high_precision_score
        ),
        high_precision_selected_variable_score=str(
            selected_high_precision_score
        ),
        high_precision_semantic_face_difference=str(
            high_precision_difference
        ),
        high_precision_semantic_face_certified=True,
        lexicographic_candidates_rejected_by_high_precision=(
            rejected_candidates
        ),
        solver_identity=solver_identity(),
    )


def decode_best_attack_hypothesis(
    graph_logit: float,
    count_logits: Sequence[float],
    source_logits: Sequence[float],
    transit_logits: Sequence[float],
    victim_logits: Sequence[float],
    path_logits: Sequence[float],
) -> ExactAttackHypothesis:
    """Decode the exact best non-normal attack hypothesis over all 240 routes."""
    return _decode_with_routes(
        graph_logit,
        count_logits,
        source_logits,
        transit_logits,
        victim_logits,
        path_logits,
        ROUTE_TABLE,
    )


def apply_margin_threshold(
    hypothesis: ExactAttackHypothesis,
    threshold: float,
) -> DecodedStructuredOutput:
    tau = _validate_scalar("threshold", threshold)

    if hypothesis.margin < tau:
        normal = empty_hypothesis()
        return DecodedStructuredOutput(
            attack=0,
            margin=hypothesis.margin,
            attacker_count=normal.attacker_count,
            route_ids=normal.route_ids,
            source_mask=normal.source_mask,
            transit_mask=normal.transit_mask,
            victim_mask=normal.victim_mask,
            path_mask=normal.path_mask,
        )

    return DecodedStructuredOutput(
        attack=1,
        margin=hypothesis.margin,
        attacker_count=hypothesis.attacker_count,
        route_ids=hypothesis.route_ids,
        source_mask=hypothesis.source_mask,
        transit_mask=hypothesis.transit_mask,
        victim_mask=hypothesis.victim_mask,
        path_mask=hypothesis.path_mask,
    )


def exhaustive_oracle_for_routes(
    graph_logit: float,
    count_logits: Sequence[float],
    source_logits: Sequence[float],
    transit_logits: Sequence[float],
    victim_logits: Sequence[float],
    path_logits: Sequence[float],
    route_records: Sequence[RouteRecord],
) -> tuple[float, int, tuple[int, ...]]:
    """Small-candidate exhaustive oracle used only by L2 unit tests."""
    from itertools import combinations

    graph = _validate_scalar("graph_logit", graph_logit)
    count = _validate_vector("count_logits", count_logits, 4)
    source = _validate_vector("source_logits", source_logits, 16)
    transit = _validate_vector("transit_logits", transit_logits, 16)
    victim = _validate_vector("victim_logits", victim_logits, 16)
    path = _validate_vector("path_logits", path_logits, 16)
    count_log_probabilities = _log_softmax(count)

    records = tuple(sorted(route_records, key=lambda record: record.route_id))
    best_score = -math.inf
    best_k = -1
    best_routes: tuple[int, ...] | None = None

    for k in K_VALUES:
        for selected in combinations(records, k):
            sources = [record.source for record in selected]
            if len(set(sources)) != k:
                continue

            route_ids = tuple(record.route_id for record in selected)
            score, _, _ = _score_route_ids(
                graph,
                count_log_probabilities,
                source,
                transit,
                victim,
                path,
                route_ids,
            )

            if score > best_score + SEMANTIC_TIE_TOLERANCE:
                best_score = score
                best_k = k
                best_routes = route_ids
            elif abs(score - best_score) <= SEMANTIC_TIE_TOLERANCE:
                assert best_routes is not None
                if (k, route_ids) < (best_k, best_routes):
                    best_score = score
                    best_k = k
                    best_routes = route_ids

    if best_routes is None:
        raise ExactDecoderError("exhaustive oracle found no feasible hypothesis")

    return float(best_score), best_k, best_routes
