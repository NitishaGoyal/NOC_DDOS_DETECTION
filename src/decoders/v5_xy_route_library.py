"""Frozen deterministic 4x4 X-then-Y route library for V5 P2.

This module contains no learned parameters and reads no dataset metadata.
It implements the route and mask semantics frozen by the V5 P2 L0 contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

MESH_ROWS = 4
MESH_COLUMNS = 4
ROUTER_COUNT = 16
ORDERED_ROUTE_COUNT = 240
FULL_ROUTER_MASK = (1 << ROUTER_COUNT) - 1

PHYSICAL_PORT_MASK_ORDER = (
    "input_local",
    "input_north",
    "input_east",
    "input_south",
    "input_west",
    "output_local",
    "output_north",
    "output_east",
    "output_south",
    "output_west",
)

DIRECTION_TO_PORTS = {
    "north": ("output_north", "input_south"),
    "east": ("output_east", "input_west"),
    "south": ("output_south", "input_north"),
    "west": ("output_west", "input_east"),
}


@dataclass(frozen=True, slots=True)
class RouteRecord:
    route_id: int
    source: int
    victim: int
    nodes: tuple[int, ...]
    hop_directions: tuple[str, ...]
    source_mask: int
    transit_mask: int
    victim_mask: int
    path_mask: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "route_id": self.route_id,
            "source": self.source,
            "victim": self.victim,
            "source_coordinate": coordinate_dict(self.source),
            "victim_coordinate": coordinate_dict(self.victim),
            "nodes": list(self.nodes),
            "hop_directions": list(self.hop_directions),
            "hop_ports": [
                {
                    "direction": direction,
                    "source_output_port": DIRECTION_TO_PORTS[direction][0],
                    "destination_input_port": DIRECTION_TO_PORTS[direction][1],
                }
                for direction in self.hop_directions
            ],
            "source_mask": self.source_mask,
            "source_mask_hex": f"0x{self.source_mask:04x}",
            "transit_mask": self.transit_mask,
            "transit_mask_hex": f"0x{self.transit_mask:04x}",
            "victim_mask": self.victim_mask,
            "victim_mask_hex": f"0x{self.victim_mask:04x}",
            "path_mask": self.path_mask,
            "path_mask_hex": f"0x{self.path_mask:04x}",
        }


@dataclass(frozen=True, slots=True)
class HypothesisMasks:
    route_ids: tuple[int, ...]
    attacker_count: int
    source_mask: int
    transit_mask: int
    victim_mask: int
    path_mask: int


def validate_router_id(router_id: int) -> None:
    if not isinstance(router_id, int):
        raise TypeError(f"router_id must be int, got {type(router_id).__name__}")
    if not 0 <= router_id < ROUTER_COUNT:
        raise ValueError(f"router_id out of range: {router_id}")


def router_to_coordinate(router_id: int) -> tuple[int, int]:
    """Return (row, column), with x=column and y=row."""
    validate_router_id(router_id)
    return divmod(router_id, MESH_COLUMNS)


def coordinate_to_router(row: int, column: int) -> int:
    if not isinstance(row, int) or not isinstance(column, int):
        raise TypeError("row and column must be integers")
    if not 0 <= row < MESH_ROWS:
        raise ValueError(f"row out of range: {row}")
    if not 0 <= column < MESH_COLUMNS:
        raise ValueError(f"column out of range: {column}")
    return row * MESH_COLUMNS + column


def coordinate_dict(router_id: int) -> dict[str, int]:
    row, column = router_to_coordinate(router_id)
    return {"row": row, "column": column, "x": column, "y": row}


def route_id_for(source: int, victim: int) -> int:
    validate_router_id(source)
    validate_router_id(victim)
    if source == victim:
        raise ValueError("source and victim must differ")
    return 15 * source + (victim if victim < source else victim - 1)


def source_victim_for_route_id(route_id: int) -> tuple[int, int]:
    if not isinstance(route_id, int):
        raise TypeError("route_id must be int")
    if not 0 <= route_id < ORDERED_ROUTE_COUNT:
        raise ValueError(f"route_id out of range: {route_id}")
    source, offset = divmod(route_id, 15)
    victim = offset if offset < source else offset + 1
    return source, victim


def hop_direction(source_router: int, destination_router: int) -> str:
    source_row, source_column = router_to_coordinate(source_router)
    destination_row, destination_column = router_to_coordinate(destination_router)
    delta_row = destination_row - source_row
    delta_column = destination_column - source_column

    if (delta_row, delta_column) == (-1, 0):
        return "north"
    if (delta_row, delta_column) == (0, 1):
        return "east"
    if (delta_row, delta_column) == (1, 0):
        return "south"
    if (delta_row, delta_column) == (0, -1):
        return "west"

    raise ValueError(
        "routers are not physically adjacent cardinal neighbors: "
        f"{source_router}->{destination_router}"
    )


def expected_directed_edges() -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    for source in range(ROUTER_COUNT):
        source_row, source_column = router_to_coordinate(source)
        for delta_row, delta_column in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            row = source_row + delta_row
            column = source_column + delta_column
            if 0 <= row < MESH_ROWS and 0 <= column < MESH_COLUMNS:
                edges.append((source, coordinate_to_router(row, column)))
    return tuple(sorted(edges))


def generate_xy_nodes(source: int, victim: int) -> tuple[int, ...]:
    """Generate the unique deterministic X-then-Y path, endpoints included."""
    validate_router_id(source)
    validate_router_id(victim)
    if source == victim:
        raise ValueError("source and victim must differ")

    source_row, source_column = router_to_coordinate(source)
    victim_row, victim_column = router_to_coordinate(victim)

    row = source_row
    column = source_column
    nodes = [source]

    while column != victim_column:
        column += 1 if victim_column > column else -1
        nodes.append(coordinate_to_router(row, column))

    while row != victim_row:
        row += 1 if victim_row > row else -1
        nodes.append(coordinate_to_router(row, column))

    return tuple(nodes)


def _mask_from_nodes(nodes: Iterable[int]) -> int:
    mask = 0
    for router_id in nodes:
        validate_router_id(router_id)
        mask |= 1 << router_id
    return mask


def validate_route_nodes(
    source: int,
    victim: int,
    nodes: Sequence[int],
    *,
    require_xy: bool = True,
) -> None:
    validate_router_id(source)
    validate_router_id(victim)

    if source == victim:
        raise ValueError("source and victim must differ")
    if len(nodes) < 2:
        raise ValueError("a non-self route must contain at least two nodes")
    if nodes[0] != source:
        raise ValueError("route does not start at source")
    if nodes[-1] != victim:
        raise ValueError("route does not end at victim")

    for router_id in nodes:
        validate_router_id(router_id)

    if len(set(nodes)) != len(nodes):
        raise ValueError("route contains a repeated router")

    for current, following in zip(nodes, nodes[1:]):
        direction = hop_direction(current, following)
        if direction not in DIRECTION_TO_PORTS:
            raise ValueError(f"unsupported hop direction: {direction}")

    if require_xy and tuple(nodes) != generate_xy_nodes(source, victim):
        raise ValueError("route does not follow the frozen X-then-Y order")


def build_route(source: int, victim: int) -> RouteRecord:
    nodes = generate_xy_nodes(source, victim)
    validate_route_nodes(source, victim, nodes, require_xy=True)
    directions = tuple(
        hop_direction(current, following)
        for current, following in zip(nodes, nodes[1:])
    )

    source_mask = 1 << source
    victim_mask = 1 << victim
    transit_mask = _mask_from_nodes(nodes[1:-1])
    path_mask = _mask_from_nodes(nodes)

    if path_mask != (source_mask | transit_mask | victim_mask):
        raise AssertionError("path mask identity violated")

    return RouteRecord(
        route_id=route_id_for(source, victim),
        source=source,
        victim=victim,
        nodes=nodes,
        hop_directions=directions,
        source_mask=source_mask,
        transit_mask=transit_mask,
        victim_mask=victim_mask,
        path_mask=path_mask,
    )


def generate_route_table() -> tuple[RouteRecord, ...]:
    routes = tuple(
        build_route(source, victim)
        for source in range(ROUTER_COUNT)
        for victim in range(ROUTER_COUNT)
        if source != victim
    )
    if len(routes) != ORDERED_ROUTE_COUNT:
        raise AssertionError(f"expected 240 routes, got {len(routes)}")
    if tuple(route.route_id for route in routes) != tuple(range(ORDERED_ROUTE_COUNT)):
        raise AssertionError("route IDs are not contiguous in frozen order")
    return routes


ROUTE_TABLE = generate_route_table()


def get_route(route_id: int) -> RouteRecord:
    source, victim = source_victim_for_route_id(route_id)
    route = ROUTE_TABLE[route_id]
    if (route.source, route.victim) != (source, victim):
        raise AssertionError("route table ordering mismatch")
    return route


def empty_hypothesis() -> HypothesisMasks:
    return HypothesisMasks(
        route_ids=(),
        attacker_count=0,
        source_mask=0,
        transit_mask=0,
        victim_mask=0,
        path_mask=0,
    )


def combine_route_ids(
    route_ids: Sequence[int],
    *,
    expected_k: int | None = None,
) -> HypothesisMasks:
    normalized = tuple(sorted(int(route_id) for route_id in route_ids))

    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate route IDs are forbidden")
    if expected_k is not None and len(normalized) != expected_k:
        raise ValueError(
            f"count/source mismatch: expected K={expected_k}, got {len(normalized)}"
        )
    if len(normalized) not in (1, 2, 3, 4):
        raise ValueError("attack hypotheses must select K in {1,2,3,4}")

    routes = tuple(get_route(route_id) for route_id in normalized)
    sources = tuple(route.source for route in routes)
    if len(set(sources)) != len(sources):
        raise ValueError("duplicate source routes are forbidden")

    source_mask = 0
    transit_mask = 0
    victim_mask = 0
    path_mask = 0

    for route in routes:
        source_mask |= route.source_mask
        transit_mask |= route.transit_mask
        victim_mask |= route.victim_mask
        path_mask |= route.path_mask

    if source_mask.bit_count() != len(routes):
        raise AssertionError("source count does not equal K")

    return HypothesisMasks(
        route_ids=normalized,
        attacker_count=len(routes),
        source_mask=source_mask,
        transit_mask=transit_mask,
        victim_mask=victim_mask,
        path_mask=path_mask,
    )


def choose_equal_score_hypothesis(
    candidates: Sequence[tuple[float, Sequence[int]]],
    *,
    tolerance: float = 1e-12,
) -> tuple[float, tuple[int, ...]]:
    """Apply the frozen equal-score tie break: lower K, then route tuple."""
    if not candidates:
        raise ValueError("at least one candidate is required")

    best_score = max(float(score) for score, _ in candidates)
    tied = [
        (float(score), tuple(sorted(int(x) for x in route_ids)))
        for score, route_ids in candidates
        if abs(float(score) - best_score) <= tolerance
    ]
    tied.sort(key=lambda item: (len(item[1]), item[1]))
    return tied[0]
