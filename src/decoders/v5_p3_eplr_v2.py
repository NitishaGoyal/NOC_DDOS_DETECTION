"""V5-P3 EPLR-V2 Preserve-or-Passthrough decoder.

This software-side decoder consumes the existing 69 neural logits.

Frozen V2 policy:
  - graph, active count, source, and victim are exactly Raw;
  - when Raw endpoints admit exactly K canonical XY routes, select the
    best endpoint-preserving legal route tuple and use its transit/path unions;
  - when Raw endpoints are not legally explainable, pass through all Raw role
    masks unchanged and explicitly report an unresolved, non-certified status;
  - no endpoint repair, endpoint reassignment, or zero-mask fallback exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from src.decoders import v5_p3_eplr_v1 as v1


STATUS_INACTIVE = "INACTIVE_RAW"
STATUS_LEGAL = "RAW_ENDPOINTS_LEGAL"
STATUS_LEGAL_LOW_SUPPORT = "RAW_ENDPOINTS_LEGAL_LOW_ROUTE_SUPPORT"
STATUS_UNRESOLVED = "RAW_ENDPOINTS_UNRESOLVED_PASSTHROUGH"


@dataclass(frozen=True)
class EPLRV2Result:
    raw_graph_probability: float
    raw_graph_prediction: int
    raw_attacker_count_candidate: int
    effective_attacker_count: int

    raw_source_bitmap: int
    raw_transit_bitmap: int
    raw_victim_bitmap: int
    raw_path_bitmap: int

    selected_source_bitmap: int
    selected_transit_bitmap: int
    selected_victim_bitmap: int
    selected_path_bitmap: int

    selected_route_ids: tuple[int, ...]
    route_consistency_score: float
    route_certified: bool
    endpoint_preserved: bool
    decoder_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decode_eplr_v2(
    *,
    graph_logit: float,
    count_logits: Sequence[float],
    source_logits: Sequence[float],
    transit_logits: Sequence[float],
    victim_logits: Sequence[float],
    path_logits: Sequence[float],
    config: v1.EPLRConfig,
) -> EPLRV2Result:
    count_logits_array = np.asarray(
        count_logits,
        dtype=np.float64,
    ).reshape(4)
    source_probabilities = v1.stable_sigmoid_array(
        source_logits
    ).reshape(16)
    transit_probabilities = v1.stable_sigmoid_array(
        transit_logits
    ).reshape(16)
    victim_probabilities = v1.stable_sigmoid_array(
        victim_logits
    ).reshape(16)
    path_probabilities = v1.stable_sigmoid_array(
        path_logits
    ).reshape(16)

    graph_probability = v1.stable_sigmoid(float(graph_logit))
    graph_prediction = int(
        graph_probability >= config.graph_threshold
    )
    raw_count = int(np.argmax(count_logits_array)) + 1

    raw_source_mask = v1.mask_from_probabilities(
        source_probabilities,
        config.endpoint_threshold,
    )
    raw_transit_mask = v1.mask_from_probabilities(
        transit_probabilities,
        config.endpoint_threshold,
    )
    raw_victim_mask = v1.mask_from_probabilities(
        victim_probabilities,
        config.endpoint_threshold,
    )
    raw_path_mask = v1.mask_from_probabilities(
        path_probabilities,
        config.endpoint_threshold,
    )

    if graph_prediction == 0:
        return EPLRV2Result(
            raw_graph_probability=graph_probability,
            raw_graph_prediction=0,
            raw_attacker_count_candidate=raw_count,
            effective_attacker_count=0,
            raw_source_bitmap=raw_source_mask,
            raw_transit_bitmap=raw_transit_mask,
            raw_victim_bitmap=raw_victim_mask,
            raw_path_bitmap=raw_path_mask,
            selected_source_bitmap=0,
            selected_transit_bitmap=0,
            selected_victim_bitmap=0,
            selected_path_bitmap=0,
            selected_route_ids=(),
            route_consistency_score=float("nan"),
            route_certified=False,
            endpoint_preserved=False,
            decoder_status=STATUS_INACTIVE,
        )

    hypothesis, support = v1.choose_endpoint_preserving_hypothesis(
        source_mask=raw_source_mask,
        victim_mask=raw_victim_mask,
        k=raw_count,
        transit_probabilities=transit_probabilities,
        path_probabilities=path_probabilities,
        config=config,
    )

    if hypothesis is None:
        return EPLRV2Result(
            raw_graph_probability=graph_probability,
            raw_graph_prediction=1,
            raw_attacker_count_candidate=raw_count,
            effective_attacker_count=raw_count,
            raw_source_bitmap=raw_source_mask,
            raw_transit_bitmap=raw_transit_mask,
            raw_victim_bitmap=raw_victim_mask,
            raw_path_bitmap=raw_path_mask,
            selected_source_bitmap=raw_source_mask,
            selected_transit_bitmap=raw_transit_mask,
            selected_victim_bitmap=raw_victim_mask,
            selected_path_bitmap=raw_path_mask,
            selected_route_ids=(),
            route_consistency_score=float("nan"),
            route_certified=False,
            endpoint_preserved=True,
            decoder_status=STATUS_UNRESOLVED,
        )

    if int(hypothesis.attacker_count) != raw_count:
        raise RuntimeError(
            "EPLR-V2 legal route selection changed immutable Raw count"
        )
    if int(hypothesis.source_mask) != raw_source_mask:
        raise RuntimeError(
            "EPLR-V2 legal route selection changed immutable Raw source"
        )
    if int(hypothesis.victim_mask) != raw_victim_mask:
        raise RuntimeError(
            "EPLR-V2 legal route selection changed immutable Raw victim"
        )

    threshold = config.low_support_threshold_by_k[raw_count - 1]
    low_support = bool(float(support) < float(threshold))
    status = (
        STATUS_LEGAL_LOW_SUPPORT
        if low_support
        else STATUS_LEGAL
    )

    return EPLRV2Result(
        raw_graph_probability=graph_probability,
        raw_graph_prediction=1,
        raw_attacker_count_candidate=raw_count,
        effective_attacker_count=raw_count,
        raw_source_bitmap=raw_source_mask,
        raw_transit_bitmap=raw_transit_mask,
        raw_victim_bitmap=raw_victim_mask,
        raw_path_bitmap=raw_path_mask,
        selected_source_bitmap=raw_source_mask,
        selected_transit_bitmap=int(hypothesis.transit_mask),
        selected_victim_bitmap=raw_victim_mask,
        selected_path_bitmap=int(hypothesis.path_mask),
        selected_route_ids=tuple(hypothesis.route_ids),
        route_consistency_score=float(support),
        route_certified=True,
        endpoint_preserved=True,
        decoder_status=status,
    )


def bitmap_to_array(mask: int) -> np.ndarray:
    return np.asarray(
        [
            1 if int(mask) & (1 << router) else 0
            for router in range(16)
        ],
        dtype=np.uint8,
    )
