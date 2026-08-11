from __future__ import annotations

import unittest
import numpy as np

from src.decoders.v5_legal_xy_beam_decoder import (
    ALLOWED_BEAM_WIDTHS,
    ALLOWED_PER_SOURCE_TOP_B,
    BeamState,
    _can_complete_to_k,
    decode_all_grid_configurations,
    decode_beam_attack_hypothesis,
)
from src.decoders.v5_xy_route_library import ROUTE_TABLE, route_id_for


class TestV5LegalXYBeamDecoderCompletionFeasibility(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates_b1 = tuple(route_id_for(source, 0 if source != 0 else 1) for source in range(16))

    def test_empty_state_can_reach_k4(self) -> None:
        state = BeamState((), 0, 0, 0, 0, 0.0)
        self.assertTrue(_can_complete_to_k(state, self.candidates_b1, 4))

    def test_late_source_dead_end_is_rejected_for_k4(self) -> None:
        route_id = self.candidates_b1[15]
        route = ROUTE_TABLE[route_id]
        state = BeamState(
            (route_id,),
            route.source_mask,
            route.transit_mask,
            route.victim_mask,
            route.path_mask,
            0.0,
        )
        self.assertFalse(_can_complete_to_k(state, self.candidates_b1, 4))

    def test_completed_state_is_accepted(self) -> None:
        route_ids = tuple(self.candidates_b1[:4])
        source_mask = 0
        transit_mask = 0
        victim_mask = 0
        path_mask = 0
        for route_id in route_ids:
            route = ROUTE_TABLE[route_id]
            source_mask |= route.source_mask
            transit_mask |= route.transit_mask
            victim_mask |= route.victim_mask
            path_mask |= route.path_mask
        state = BeamState(
            route_ids,
            source_mask,
            transit_mask,
            victim_mask,
            path_mask,
            0.0,
        )
        self.assertTrue(_can_complete_to_k(state, self.candidates_b1, 4))

    def test_adversarial_high_source_scores_execute_all_nine_configs(self) -> None:
        # Without the completion-feasibility guard, the narrow beam is filled by
        # routes from the highest-numbered sources and eventually has no legal
        # increasing-ID continuation for K4.
        source = np.arange(16, dtype=np.float64) * 100.0
        zeros = np.zeros(16, dtype=np.float64)
        count = np.asarray([-100.0, -100.0, -100.0, 100.0], dtype=np.float64)
        results = decode_all_grid_configurations(
            0.0,
            count,
            source,
            zeros,
            zeros,
            zeros,
        )
        expected = {
            (b, width)
            for b in ALLOWED_PER_SOURCE_TOP_B
            for width in ALLOWED_BEAM_WIDTHS
        }
        self.assertEqual(set(results), expected)
        for result in results.values():
            self.assertEqual(result.attacker_count, 4)
            sources = [ROUTE_TABLE[route_id].source for route_id in result.route_ids]
            self.assertEqual(len(sources), 4)
            self.assertEqual(len(set(sources)), 4)
            self.assertGreater(result.completion_feasibility_pruned_state_count, 0)

    def test_narrowest_grid_is_deterministic_under_dead_end_pressure(self) -> None:
        source = np.arange(16, dtype=np.float64) * 100.0
        zeros = np.zeros(16, dtype=np.float64)
        count = np.asarray([-100.0, -100.0, -100.0, 100.0], dtype=np.float64)
        first = decode_beam_attack_hypothesis(
            0.0,
            count,
            source,
            zeros,
            zeros,
            zeros,
            per_source_top_b=1,
            beam_width=4,
        )
        second = decode_beam_attack_hypothesis(
            0.0,
            count,
            source,
            zeros,
            zeros,
            zeros,
            per_source_top_b=1,
            beam_width=4,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
