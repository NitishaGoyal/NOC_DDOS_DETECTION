from __future__ import annotations

import unittest
import numpy as np

from src.decoders.v5_legal_xy_beam_decoder import (
    ALLOWED_BEAM_WIDTHS,
    ALLOWED_PER_SOURCE_TOP_B,
    BeamState,
    SCORE_TIE_TOLERANCE,
    _expand,
    apply_beam_margin_threshold,
    decode_all_grid_configurations,
    decode_beam_attack_hypothesis,
    rank_candidates_by_source,
    recompute_state_role_score,
)
from src.decoders.v5_legal_xy_exact_decoder import decode_best_attack_hypothesis
from src.decoders.v5_xy_route_library import ROUTE_TABLE, combine_route_ids, route_id_for


class TestV5LegalXYBeamDecoder(unittest.TestCase):
    def setUp(self) -> None:
        self.zeros16 = np.zeros(16, dtype=np.float64)

    def test_exactly_B_candidates_per_source(self) -> None:
        for b in ALLOWED_PER_SOURCE_TOP_B:
            ranked = rank_candidates_by_source(
                self.zeros16, self.zeros16, self.zeros16, self.zeros16,
                per_source_top_b=b,
            )
            self.assertEqual(set(ranked), set(range(16)))
            self.assertTrue(all(len(values) == b for values in ranked.values()))
            self.assertEqual(len({route_id for values in ranked.values() for route_id in values}), 16 * b)

    def test_candidate_ties_use_lower_route_id(self) -> None:
        ranked = rank_candidates_by_source(
            self.zeros16, self.zeros16, self.zeros16, self.zeros16,
            per_source_top_b=4,
        )
        for source in range(16):
            expected = tuple(route.route_id for route in ROUTE_TABLE if route.source == source)[:4]
            self.assertEqual(ranked[source], expected)

    def test_incremental_union_score_matches_recomputation(self) -> None:
        source = np.linspace(-0.8, 0.8, 16)
        transit = np.linspace(0.6, -0.6, 16)
        victim = np.linspace(-0.4, 0.4, 16)
        path = np.linspace(0.3, -0.3, 16)
        state = BeamState((), 0, 0, 0, 0, 0.0)
        for route_id in sorted((route_id_for(0, 15), route_id_for(1, 12))):
            state = _expand(state, ROUTE_TABLE[route_id], source, transit, victim, path)
        recomputed = recompute_state_role_score(state, source, transit, victim, path)
        self.assertAlmostEqual(state.role_score, recomputed, delta=SCORE_TIE_TOLERANCE)

    def test_all_nine_grid_configurations_execute(self) -> None:
        results = decode_all_grid_configurations(
            0.0, np.zeros(4), self.zeros16, self.zeros16, self.zeros16, self.zeros16
        )
        self.assertEqual(len(results), 9)
        self.assertEqual(
            set(results),
            {(b, width) for b in ALLOWED_PER_SOURCE_TOP_B for width in ALLOWED_BEAM_WIDTHS},
        )
        for (b, width), result in results.items():
            self.assertEqual(result.candidate_route_count, 16 * b)
            self.assertEqual(result.per_source_top_b, b)
            self.assertEqual(result.beam_width, width)

    def test_count_logits_can_select_K2(self) -> None:
        result = decode_beam_attack_hypothesis(
            0.0,
            np.asarray([-20.0, 20.0, -20.0, -20.0]),
            self.zeros16, self.zeros16, self.zeros16, self.zeros16,
            per_source_top_b=4,
            beam_width=16,
        )
        self.assertEqual(result.attacker_count, 2)
        self.assertEqual(result.route_ids, (0, 15))

    def test_known_K1_matches_exact(self) -> None:
        source = np.full(16, -100.0)
        transit = np.full(16, -100.0)
        victim = np.full(16, -100.0)
        path = np.full(16, -100.0)
        expected_route = ROUTE_TABLE[route_id_for(0, 15)]
        source[expected_route.source] = 100.0
        victim[expected_route.victim] = 100.0
        for router_id in range(16):
            if (expected_route.transit_mask >> router_id) & 1:
                transit[router_id] = 100.0
            if (expected_route.path_mask >> router_id) & 1:
                path[router_id] = 100.0
        args = (0.0, np.asarray([100.0, -100.0, -100.0, -100.0]), source, transit, victim, path)
        exact = decode_best_attack_hypothesis(*args)
        beam = decode_beam_attack_hypothesis(*args, per_source_top_b=4, beam_width=16)
        self.assertEqual(beam.route_ids, exact.route_ids)
        self.assertEqual(beam.attacker_count, exact.attacker_count)
        self.assertAlmostEqual(beam.margin, exact.margin, delta=SCORE_TIE_TOLERANCE)

    def test_known_K2_matches_exact(self) -> None:
        source = np.full(16, -100.0)
        transit = np.full(16, -100.0)
        victim = np.full(16, -100.0)
        path = np.full(16, -100.0)
        expected = combine_route_ids((route_id_for(0, 1), route_id_for(14, 15)), expected_k=2)
        for router_id in range(16):
            if (expected.source_mask >> router_id) & 1:
                source[router_id] = 100.0
            if (expected.transit_mask >> router_id) & 1:
                transit[router_id] = 100.0
            if (expected.victim_mask >> router_id) & 1:
                victim[router_id] = 100.0
            if (expected.path_mask >> router_id) & 1:
                path[router_id] = 100.0
        args = (0.0, np.asarray([-100.0, 100.0, -100.0, -100.0]), source, transit, victim, path)
        exact = decode_best_attack_hypothesis(*args)
        beam = decode_beam_attack_hypothesis(*args, per_source_top_b=4, beam_width=16)
        self.assertEqual(beam.route_ids, exact.route_ids)
        self.assertEqual(beam.attacker_count, exact.attacker_count)
        self.assertAlmostEqual(beam.margin, exact.margin, delta=SCORE_TIE_TOLERANCE)

    def test_distinct_sources_and_union_masks(self) -> None:
        rng = np.random.default_rng(73)
        result = decode_beam_attack_hypothesis(
            float(rng.normal()), rng.normal(size=4), rng.normal(size=16),
            rng.normal(size=16), rng.normal(size=16), rng.normal(size=16),
            per_source_top_b=4, beam_width=16,
        )
        sources = [ROUTE_TABLE[route_id].source for route_id in result.route_ids]
        self.assertEqual(len(sources), len(set(sources)))
        expected = combine_route_ids(result.route_ids, expected_k=result.attacker_count)
        self.assertEqual((result.source_mask, result.transit_mask, result.victim_mask, result.path_mask),
                         (expected.source_mask, expected.transit_mask, expected.victim_mask, expected.path_mask))

    def test_margin_threshold_emits_H0(self) -> None:
        result = decode_beam_attack_hypothesis(
            -5.0, np.zeros(4), self.zeros16, self.zeros16, self.zeros16, self.zeros16,
            per_source_top_b=1, beam_width=4,
        )
        decoded = apply_beam_margin_threshold(result, result.margin + 1.0)
        self.assertEqual(decoded.attack, 0)
        self.assertEqual(decoded.attacker_count, 0)
        self.assertEqual(decoded.route_ids, ())
        self.assertEqual(decoded.source_mask | decoded.transit_mask | decoded.victim_mask | decoded.path_mask, 0)

    def test_repeat_execution_is_deterministic(self) -> None:
        args = (
            0.25,
            np.asarray([0.1, 0.2, 0.3, 0.4]),
            np.linspace(-0.8, 0.8, 16),
            np.linspace(0.6, -0.6, 16),
            np.linspace(-0.4, 0.4, 16),
            np.linspace(0.3, -0.3, 16),
        )
        first = decode_beam_attack_hypothesis(*args, per_source_top_b=4, beam_width=16)
        second = decode_beam_attack_hypothesis(*args, per_source_top_b=4, beam_width=16)
        self.assertEqual(first, second)

    def test_invalid_grid_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            decode_beam_attack_hypothesis(
                0.0, np.zeros(4), self.zeros16, self.zeros16, self.zeros16, self.zeros16,
                per_source_top_b=3, beam_width=8,
            )
        with self.assertRaises(ValueError):
            decode_beam_attack_hypothesis(
                0.0, np.zeros(4), self.zeros16, self.zeros16, self.zeros16, self.zeros16,
                per_source_top_b=2, beam_width=5,
            )

    def test_nonfinite_input_is_rejected(self) -> None:
        bad = np.zeros(16)
        bad[2] = np.inf
        with self.assertRaises(ValueError):
            decode_beam_attack_hypothesis(
                0.0, np.zeros(4), bad, self.zeros16, self.zeros16, self.zeros16,
                per_source_top_b=1, beam_width=4,
            )


if __name__ == "__main__":
    unittest.main()
