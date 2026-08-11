from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.decoders.v5_xy_route_library import (
    DIRECTION_TO_PORTS,
    ORDERED_ROUTE_COUNT,
    PHYSICAL_PORT_MASK_ORDER,
    ROUTE_TABLE,
    build_route,
    choose_equal_score_hypothesis,
    combine_route_ids,
    coordinate_to_router,
    empty_hypothesis,
    expected_directed_edges,
    generate_xy_nodes,
    get_route,
    hop_direction,
    route_id_for,
    router_to_coordinate,
    source_victim_for_route_id,
    validate_route_nodes,
)

ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_DIR = (
    ROOT
    / "reports/v5/p2_g1a_r2a_canonical_static_topology_contract"
)
ROUTE_TABLE_PATH = ROOT / "artifacts/v5/decoders/xy4x4_route_table.json"


class TestV5XYRouteLibrary(unittest.TestCase):
    def test_route_count_and_contiguous_ids(self) -> None:
        self.assertEqual(len(ROUTE_TABLE), 240)
        self.assertEqual(
            [route.route_id for route in ROUTE_TABLE],
            list(range(240)),
        )

    def test_route_id_round_trip(self) -> None:
        for source in range(16):
            for victim in range(16):
                if source == victim:
                    continue
                route_id = route_id_for(source, victim)
                self.assertEqual(
                    source_victim_for_route_id(route_id),
                    (source, victim),
                )

    def test_coordinate_round_trip(self) -> None:
        for router_id in range(16):
            row, column = router_to_coordinate(router_id)
            self.assertEqual(coordinate_to_router(row, column), router_id)

    def test_coordinates_match_contract_array(self) -> None:
        coordinates = np.load(
            TOPOLOGY_DIR / "V5_P2_G1A_R2A_ROUTER_COORDINATES_ROW_MAJOR.npy",
            allow_pickle=False,
        )
        expected = np.asarray(
            [router_to_coordinate(router_id) for router_id in range(16)],
            dtype=coordinates.dtype,
        )
        np.testing.assert_array_equal(coordinates, expected)

    def test_expected_edges_match_contract_arrays(self) -> None:
        edges = expected_directed_edges()
        self.assertEqual(len(edges), 48)

        edge_index = np.load(
            TOPOLOGY_DIR / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
            allow_pickle=False,
        )
        actual_edges = tuple(
            (int(source), int(destination))
            for source, destination in zip(edge_index[0], edge_index[1])
        )
        self.assertEqual(actual_edges, edges)

        adjacency = np.load(
            TOPOLOGY_DIR / "V5_P2_G1A_R2A_CANONICAL_ADJACENCY_MATRIX.npy",
            allow_pickle=False,
        )
        expected_adjacency = np.zeros((16, 16), dtype=adjacency.dtype)
        for source, destination in edges:
            expected_adjacency[source, destination] = 1
        np.testing.assert_array_equal(adjacency, expected_adjacency)

    def test_all_routes_are_legal_X_then_Y(self) -> None:
        for route in ROUTE_TABLE:
            validate_route_nodes(
                route.source,
                route.victim,
                route.nodes,
                require_xy=True,
            )
            self.assertEqual(
                route.nodes,
                generate_xy_nodes(route.source, route.victim),
            )
            self.assertEqual(len(route.hop_directions), len(route.nodes) - 1)
            for direction in route.hop_directions:
                self.assertIn(direction, DIRECTION_TO_PORTS)

    def test_mask_semantics(self) -> None:
        for route in ROUTE_TABLE:
            self.assertEqual(route.source_mask, 1 << route.source)
            self.assertEqual(route.victim_mask, 1 << route.victim)
            expected_transit = 0
            for router_id in route.nodes[1:-1]:
                expected_transit |= 1 << router_id
            expected_path = 0
            for router_id in route.nodes:
                expected_path |= 1 << router_id
            self.assertEqual(route.transit_mask, expected_transit)
            self.assertEqual(route.path_mask, expected_path)
            self.assertEqual(
                route.path_mask,
                route.source_mask | route.transit_mask | route.victim_mask,
            )

    def test_one_hop_K1(self) -> None:
        route = build_route(0, 1)
        self.assertEqual(route.nodes, (0, 1))
        self.assertEqual(route.transit_mask, 0)
        self.assertEqual(route.hop_directions, ("east",))

    def test_corner_to_corner_K1(self) -> None:
        route = build_route(0, 15)
        self.assertEqual(route.nodes, (0, 1, 2, 3, 7, 11, 15))
        self.assertEqual(
            route.hop_directions,
            ("east", "east", "east", "south", "south", "south"),
        )

    def test_reverse_direction(self) -> None:
        route = build_route(15, 0)
        self.assertEqual(route.nodes, (15, 14, 13, 12, 8, 4, 0))
        self.assertEqual(
            route.hop_directions,
            ("west", "west", "west", "north", "north", "north"),
        )

    def test_edge_source(self) -> None:
        route = build_route(3, 12)
        self.assertEqual(route.nodes, (3, 2, 1, 0, 4, 8, 12))

    def test_normal_empty_output(self) -> None:
        normal = empty_hypothesis()
        self.assertEqual(normal.attacker_count, 0)
        self.assertEqual(normal.route_ids, ())
        self.assertEqual(normal.source_mask, 0)
        self.assertEqual(normal.transit_mask, 0)
        self.assertEqual(normal.victim_mask, 0)
        self.assertEqual(normal.path_mask, 0)

    def test_K2_disjoint_paths(self) -> None:
        hypothesis = combine_route_ids(
            [route_id_for(0, 1), route_id_for(14, 15)],
            expected_k=2,
        )
        self.assertEqual(hypothesis.attacker_count, 2)
        self.assertEqual(hypothesis.source_mask.bit_count(), 2)

    def test_K2_shared_victim(self) -> None:
        hypothesis = combine_route_ids(
            [route_id_for(0, 15), route_id_for(3, 15)],
            expected_k=2,
        )
        self.assertEqual(hypothesis.victim_mask.bit_count(), 1)
        self.assertTrue(hypothesis.victim_mask & (1 << 15))

    def test_K2_overlapping_paths(self) -> None:
        first = get_route(route_id_for(0, 15))
        second = get_route(route_id_for(1, 12))
        hypothesis = combine_route_ids(
            [first.route_id, second.route_id],
            expected_k=2,
        )
        total = first.path_mask.bit_count() + second.path_mask.bit_count()
        self.assertLess(hypothesis.path_mask.bit_count(), total)

    def test_K4_distinct_sources(self) -> None:
        hypothesis = combine_route_ids(
            [
                route_id_for(0, 15),
                route_id_for(1, 14),
                route_id_for(2, 13),
                route_id_for(3, 12),
            ],
            expected_k=4,
        )
        self.assertEqual(hypothesis.attacker_count, 4)
        self.assertEqual(hypothesis.source_mask.bit_count(), 4)

    def test_source_also_transit(self) -> None:
        hypothesis = combine_route_ids(
            [route_id_for(0, 3), route_id_for(1, 5)],
            expected_k=2,
        )
        self.assertTrue(hypothesis.source_mask & (1 << 1))
        self.assertTrue(hypothesis.transit_mask & (1 << 1))

    def test_source_also_victim(self) -> None:
        hypothesis = combine_route_ids(
            [route_id_for(0, 3), route_id_for(3, 4)],
            expected_k=2,
        )
        self.assertTrue(hypothesis.source_mask & (1 << 3))
        self.assertTrue(hypothesis.victim_mask & (1 << 3))

    def test_diagonal_hop_rejected(self) -> None:
        with self.assertRaises(ValueError):
            hop_direction(0, 5)

    def test_nonadjacent_hop_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_route_nodes(0, 2, (0, 2), require_xy=False)

    def test_wrong_dimension_order_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_route_nodes(0, 5, (0, 4, 5), require_xy=True)

    def test_invalid_physical_port_rejected_by_boundary(self) -> None:
        with self.assertRaises(ValueError):
            hop_direction(0, -1)

    def test_duplicate_source_routes_rejected(self) -> None:
        with self.assertRaises(ValueError):
            combine_route_ids(
                [route_id_for(0, 1), route_id_for(0, 2)],
                expected_k=2,
            )

    def test_count_source_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            combine_route_ids(
                [route_id_for(0, 1), route_id_for(4, 5)],
                expected_k=3,
            )

    def test_equal_score_tie_break_is_deterministic(self) -> None:
        score, route_ids = choose_equal_score_hypothesis(
            [
                (1.0, [route_id_for(3, 4), route_id_for(0, 1)]),
                (1.0, [route_id_for(2, 3)]),
                (1.0, [route_id_for(1, 2)]),
            ]
        )
        self.assertEqual(score, 1.0)
        self.assertEqual(route_ids, (route_id_for(1, 2),))

    def test_port_order_is_frozen(self) -> None:
        self.assertEqual(
            PHYSICAL_PORT_MASK_ORDER,
            (
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
            ),
        )

    def test_frozen_json_route_table_matches_module(self) -> None:
        with ROUTE_TABLE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.assertEqual(payload["ordered_route_count"], ORDERED_ROUTE_COUNT)
        self.assertEqual(len(payload["routes"]), ORDERED_ROUTE_COUNT)

        for expected, observed in zip(ROUTE_TABLE, payload["routes"]):
            self.assertEqual(observed, expected.to_json_dict())


if __name__ == "__main__":
    unittest.main()
