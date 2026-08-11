from __future__ import annotations

import math
import unittest

import numpy as np

from src.decoders.v5_legal_xy_exact_decoder import (
    ExactDecoderError,
    SEMANTIC_TIE_TOLERANCE,
    _decode_with_routes,
    apply_margin_threshold,
    decode_best_attack_hypothesis,
    exhaustive_oracle_for_routes,
    solver_identity,
)
from src.decoders.v5_xy_route_library import (
    ROUTE_TABLE,
    combine_route_ids,
    route_id_for,
)


class TestV5LegalXYExactDecoder(unittest.TestCase):
    def setUp(self) -> None:
        self.zeros16 = np.zeros(16, dtype=np.float64)

    def test_solver_identity_is_resolved(self) -> None:
        identity = solver_identity()
        self.assertTrue(identity.python_version)
        self.assertTrue(identity.numpy_version)
        self.assertTrue(identity.scipy_version)
        self.assertTrue(identity.highs_version)
        self.assertEqual(
            identity.backend,
            "scipy.optimize.milp_with_embedded_HiGHS",
        )

    def test_equal_score_prefers_lower_K_then_route_tuple(self) -> None:
        result = decode_best_attack_hypothesis(
            0.0,
            np.zeros(4),
            self.zeros16,
            self.zeros16,
            self.zeros16,
            self.zeros16,
        )
        self.assertEqual(result.attacker_count, 1)
        self.assertEqual(result.route_ids, (0,))
        self.assertTrue(result.optimality_proven)
        self.assertTrue(result.lexicographic_tie_break_certified)
        self.assertEqual(result.primary_mip_gap, 0.0)
        self.assertEqual(result.lexicographic_mip_gap, 0.0)

    def test_full_count_logits_can_select_K2(self) -> None:
        result = decode_best_attack_hypothesis(
            0.0,
            np.asarray([-20.0, 20.0, -20.0, -20.0]),
            self.zeros16,
            self.zeros16,
            self.zeros16,
            self.zeros16,
        )
        self.assertEqual(result.attacker_count, 2)
        self.assertEqual(result.route_ids, (0, 15))

    def test_known_K1_route_is_selected(self) -> None:
        source = np.full(16, -20.0)
        transit = np.full(16, -20.0)
        victim = np.full(16, -20.0)
        path = np.full(16, -20.0)

        expected_route_id = route_id_for(0, 15)
        expected = ROUTE_TABLE[expected_route_id]

        source[expected.source] = 30.0
        victim[expected.victim] = 30.0

        for router_id in range(16):
            if (expected.transit_mask >> router_id) & 1:
                transit[router_id] = 30.0
            if (expected.path_mask >> router_id) & 1:
                path[router_id] = 30.0

        result = decode_best_attack_hypothesis(
            0.0,
            np.asarray([30.0, -30.0, -30.0, -30.0]),
            source,
            transit,
            victim,
            path,
        )
        self.assertEqual(result.attacker_count, 1)
        self.assertEqual(result.route_ids, (expected_route_id,))
        self.assertEqual(result.source_mask, expected.source_mask)
        self.assertEqual(result.transit_mask, expected.transit_mask)
        self.assertEqual(result.victim_mask, expected.victim_mask)
        self.assertEqual(result.path_mask, expected.path_mask)

    def test_output_masks_equal_route_union(self) -> None:
        result = decode_best_attack_hypothesis(
            0.5,
            np.asarray([-10.0, 10.0, -10.0, -10.0]),
            np.linspace(-1.0, 1.0, 16),
            np.linspace(1.0, -1.0, 16),
            np.linspace(-0.5, 0.5, 16),
            np.linspace(0.5, -0.5, 16),
        )
        expected = combine_route_ids(
            result.route_ids,
            expected_k=result.attacker_count,
        )
        self.assertEqual(result.source_mask, expected.source_mask)
        self.assertEqual(result.transit_mask, expected.transit_mask)
        self.assertEqual(result.victim_mask, expected.victim_mask)
        self.assertEqual(result.path_mask, expected.path_mask)

    def test_margin_threshold_emits_H0(self) -> None:
        result = decode_best_attack_hypothesis(
            -5.0,
            np.zeros(4),
            self.zeros16,
            self.zeros16,
            self.zeros16,
            self.zeros16,
        )
        decoded = apply_margin_threshold(result, result.margin + 1.0)
        self.assertEqual(decoded.attack, 0)
        self.assertEqual(decoded.attacker_count, 0)
        self.assertEqual(decoded.route_ids, ())
        self.assertEqual(decoded.source_mask, 0)
        self.assertEqual(decoded.transit_mask, 0)
        self.assertEqual(decoded.victim_mask, 0)
        self.assertEqual(decoded.path_mask, 0)

    def test_margin_threshold_accepts_exact_boundary(self) -> None:
        result = decode_best_attack_hypothesis(
            0.0,
            np.zeros(4),
            self.zeros16,
            self.zeros16,
            self.zeros16,
            self.zeros16,
        )
        decoded = apply_margin_threshold(result, result.margin)
        self.assertEqual(decoded.attack, 1)
        self.assertEqual(decoded.route_ids, result.route_ids)

    def test_nonfinite_input_is_rejected(self) -> None:
        bad = np.zeros(16)
        bad[3] = np.nan
        with self.assertRaises(ValueError):
            decode_best_attack_hypothesis(
                0.0,
                np.zeros(4),
                bad,
                self.zeros16,
                self.zeros16,
                self.zeros16,
            )

    def test_restricted_MILP_matches_exhaustive_oracle(self) -> None:
        route_ids = [
            route_id_for(0, 1),
            route_id_for(0, 5),
            route_id_for(1, 0),
            route_id_for(1, 6),
            route_id_for(4, 5),
            route_id_for(4, 10),
            route_id_for(5, 4),
            route_id_for(5, 15),
        ]
        records = tuple(ROUTE_TABLE[route_id] for route_id in route_ids)

        for seed in (7, 17, 27):
            rng = np.random.default_rng(seed)
            graph = float(rng.normal())
            count = rng.normal(size=4)
            source = rng.normal(size=16)
            transit = rng.normal(size=16)
            victim = rng.normal(size=16)
            path = rng.normal(size=16)

            exact = _decode_with_routes(
                graph,
                count,
                source,
                transit,
                victim,
                path,
                records,
            )
            oracle_score, oracle_k, oracle_routes = exhaustive_oracle_for_routes(
                graph,
                count,
                source,
                transit,
                victim,
                path,
                records,
            )

            self.assertAlmostEqual(
                exact.margin,
                oracle_score,
                delta=SEMANTIC_TIE_TOLERANCE,
            )
            self.assertEqual(exact.attacker_count, oracle_k)
            self.assertEqual(exact.route_ids, oracle_routes)

    def test_repeat_execution_is_deterministic(self) -> None:
        args = (
            0.25,
            np.asarray([0.1, 0.2, 0.3, 0.4]),
            np.linspace(-0.8, 0.8, 16),
            np.linspace(0.6, -0.6, 16),
            np.linspace(-0.4, 0.4, 16),
            np.linspace(0.3, -0.3, 16),
        )
        first = decode_best_attack_hypothesis(*args)
        second = decode_best_attack_hypothesis(*args)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
