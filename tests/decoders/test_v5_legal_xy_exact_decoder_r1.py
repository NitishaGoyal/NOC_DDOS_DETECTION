from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import numpy as np

from src.decoders.v5_legal_xy_exact_decoder import (
    ExactDecoderError,
    MIP_GAP_REPORTING_TOLERANCE,
    MIP_GAP_REPORTING_ULP_MULTIPLIER,
    SEMANTIC_TIE_TOLERANCE,
    _check_optimal_result,
    decode_best_attack_hypothesis,
    mip_gap_is_numerical_zero,
)


class TestV5ExactDecoderNumericalGapCertification(unittest.TestCase):
    @staticmethod
    def result(gap: float, *, success: bool = True, status: int = 0):
        return SimpleNamespace(
            success=success,
            status=status,
            mip_gap=gap,
            x=np.zeros(1, dtype=np.float64),
            fun=0.0,
            message="HiGHS Status 7: Optimal",
        )

    def test_triggering_machine_residual_is_accepted(self) -> None:
        gap = 1.2458674689278237e-16
        self.assertTrue(mip_gap_is_numerical_zero(gap))
        _check_optimal_result("primary", self.result(gap))

    def test_tolerance_is_machine_scale_and_below_semantic_tolerance(self) -> None:
        self.assertEqual(MIP_GAP_REPORTING_ULP_MULTIPLIER, 64)
        self.assertAlmostEqual(
            MIP_GAP_REPORTING_TOLERANCE,
            64 * np.finfo(np.float64).eps,
            delta=0.0,
        )
        self.assertLess(MIP_GAP_REPORTING_TOLERANCE, SEMANTIC_TIE_TOLERANCE)

    def test_boundary_is_accepted(self) -> None:
        _check_optimal_result(
            "primary",
            self.result(MIP_GAP_REPORTING_TOLERANCE),
        )

    def test_next_float_above_tolerance_is_rejected(self) -> None:
        material = np.nextafter(MIP_GAP_REPORTING_TOLERANCE, math.inf)
        self.assertFalse(mip_gap_is_numerical_zero(material))
        with self.assertRaises(ExactDecoderError):
            _check_optimal_result("primary", self.result(material))

    def test_invalid_and_nonoptimal_results_remain_rejected(self) -> None:
        for gap in (-np.finfo(np.float64).eps, np.inf, np.nan):
            with self.subTest(gap=gap):
                with self.assertRaises(ExactDecoderError):
                    _check_optimal_result("primary", self.result(gap))
        with self.assertRaises(ExactDecoderError):
            _check_optimal_result(
                "primary",
                self.result(0.0, success=False, status=1),
            )

    def test_normal_decode_remains_certified(self) -> None:
        zeros = np.zeros(16, dtype=np.float64)
        result = decode_best_attack_hypothesis(
            0.0,
            np.zeros(4, dtype=np.float64),
            zeros,
            zeros,
            zeros,
            zeros,
        )
        self.assertTrue(result.optimality_proven)
        self.assertTrue(result.lexicographic_tie_break_certified)
        self.assertTrue(mip_gap_is_numerical_zero(result.primary_mip_gap))
        self.assertTrue(mip_gap_is_numerical_zero(result.lexicographic_mip_gap))


if __name__ == "__main__":
    unittest.main()
