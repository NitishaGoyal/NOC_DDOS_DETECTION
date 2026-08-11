from __future__ import annotations

import math
import unittest

import numpy as np

from src.decoders.v5_legal_xy_exact_decoder import (
    FACE_SCORE_ACCUMULATION_ULP_MULTIPLIER,
    FACE_SCORE_NUMERICAL_ALLOWANCE_CAP,
    SEMANTIC_TIE_TOLERANCE,
    decode_best_attack_hypothesis,
    semantic_face_difference_is_certified,
    semantic_face_numerical_allowance,
)


TRIGGERING_SCORE_DIFFERENCE = 1.1439738045737613e-12


class TestV5ExactDecoderOptimumFaceNumericalCertification(unittest.TestCase):
    def test_observed_roundoff_excess_is_certified(self) -> None:
        objective = np.linspace(-0.5, 0.5, 68, dtype=np.float64)
        allowance = semantic_face_numerical_allowance(
            objective,
            1.0,
            1.0 + TRIGGERING_SCORE_DIFFERENCE,
        )
        self.assertTrue(
            semantic_face_difference_is_certified(
                TRIGGERING_SCORE_DIFFERENCE,
                allowance,
            )
        )

    def test_allowance_is_bounded_by_semantic_tolerance(self) -> None:
        objective = np.full(68, 100.0, dtype=np.float64)
        allowance = semantic_face_numerical_allowance(
            objective,
            10.0,
            10.0,
        )
        self.assertGreaterEqual(allowance, 0.0)
        self.assertLessEqual(allowance, FACE_SCORE_NUMERICAL_ALLOWANCE_CAP)
        self.assertEqual(
            FACE_SCORE_NUMERICAL_ALLOWANCE_CAP,
            SEMANTIC_TIE_TOLERANCE,
        )

    def test_accumulation_floor_is_machine_scale(self) -> None:
        self.assertEqual(FACE_SCORE_ACCUMULATION_ULP_MULTIPLIER, 1024)
        objective = np.asarray([1.0], dtype=np.float64)
        allowance = semantic_face_numerical_allowance(
            objective,
            0.0,
            0.0,
        )
        self.assertGreaterEqual(
            allowance,
            1024 * np.finfo(np.float64).eps,
        )

    def test_material_face_difference_is_rejected(self) -> None:
        allowance = FACE_SCORE_NUMERICAL_ALLOWANCE_CAP
        material = np.nextafter(
            SEMANTIC_TIE_TOLERANCE + allowance,
            math.inf,
        )
        self.assertFalse(
            semantic_face_difference_is_certified(material, allowance)
        )

    def test_invalid_face_values_are_rejected(self) -> None:
        for difference in (-1.0, np.inf, np.nan):
            with self.subTest(difference=difference):
                self.assertFalse(
                    semantic_face_difference_is_certified(
                        difference,
                        0.0,
                    )
                )
        for allowance in (-1.0, np.inf, np.nan):
            with self.subTest(allowance=allowance):
                self.assertFalse(
                    semantic_face_difference_is_certified(
                        0.0,
                        allowance,
                    )
                )

    def test_normal_decode_records_face_certificate(self) -> None:
        zeros = np.zeros(16, dtype=np.float64)
        result = decode_best_attack_hypothesis(
            0.0,
            np.zeros(4, dtype=np.float64),
            zeros,
            zeros,
            zeros,
            zeros,
        )
        self.assertTrue(result.lexicographic_tie_break_certified)
        self.assertTrue(
            semantic_face_difference_is_certified(
                result.semantic_face_score_difference,
                result.semantic_face_numerical_allowance,
            )
        )
        self.assertAlmostEqual(
            result.semantic_face_certification_tolerance,
            SEMANTIC_TIE_TOLERANCE
            + result.semantic_face_numerical_allowance,
            delta=0.0,
        )


if __name__ == "__main__":
    unittest.main()
