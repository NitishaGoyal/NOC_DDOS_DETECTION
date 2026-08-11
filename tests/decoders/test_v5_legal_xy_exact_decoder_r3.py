from __future__ import annotations

from decimal import Decimal
import math
import unittest

import numpy as np

from src.decoders.v5_legal_xy_exact_decoder import (
    HIGH_PRECISION_DECIMAL_DIGITS,
    MAX_HIGH_PRECISION_FACE_REJECTIONS,
    ROUTE_TABLE,
    SEMANTIC_TIE_TOLERANCE,
    _build_model,
    _route_tuple_nogood_row,
    decode_best_attack_hypothesis,
    high_precision_semantic_face_is_certified,
)


class TestV5ExactDecoderHighPrecisionSemanticFace(unittest.TestCase):
    def test_decimal_boundary_is_accepted(self) -> None:
        tolerance = Decimal.from_float(SEMANTIC_TIE_TOLERANCE)
        self.assertTrue(high_precision_semantic_face_is_certified(tolerance))

    def test_decimal_next_value_above_boundary_is_rejected(self) -> None:
        tolerance = Decimal.from_float(SEMANTIC_TIE_TOLERANCE)
        above = tolerance + Decimal("1e-40")
        self.assertFalse(high_precision_semantic_face_is_certified(above))

    def test_invalid_decimal_values_are_rejected(self) -> None:
        for value in (
            Decimal("-1"),
            Decimal("NaN"),
            Decimal("Infinity"),
        ):
            with self.subTest(value=value):
                self.assertFalse(
                    high_precision_semantic_face_is_certified(value)
                )

    def test_no_good_cut_excludes_only_the_exact_count_and_tuple(self) -> None:
        model = _build_model(ROUTE_TABLE)
        first = ROUTE_TABLE[0].route_id
        second = next(
            record.route_id
            for record in ROUTE_TABLE
            if record.source != ROUTE_TABLE[0].source
        )
        row, lower, upper = _route_tuple_nogood_row(model, (first,))
        coefficients = row.toarray().reshape(-1)

        route_index = {
            record.route_id: index
            for index, record in enumerate(model.route_records)
        }

        exact = np.zeros(model.variable_count, dtype=np.float64)
        exact[
            model.z_offset + route_index[first]
        ] = 1.0
        exact[model.count_class_offset] = 1.0
        self.assertGreater(float(coefficients @ exact), upper)

        longer = np.zeros(model.variable_count, dtype=np.float64)
        longer[
            model.z_offset + route_index[first]
        ] = 1.0
        longer[
            model.z_offset + model.route_count + route_index[second]
        ] = 1.0
        longer[model.count_class_offset + 1] = 1.0
        self.assertLessEqual(float(coefficients @ longer), upper)
        self.assertTrue(math.isinf(lower) and lower < 0.0)

    def test_normal_decode_records_high_precision_certificate(self) -> None:
        zeros = np.zeros(16, dtype=np.float64)
        result = decode_best_attack_hypothesis(
            0.0,
            np.zeros(4, dtype=np.float64),
            zeros,
            zeros,
            zeros,
            zeros,
        )
        self.assertTrue(result.high_precision_semantic_face_certified)
        difference = Decimal(result.high_precision_semantic_face_difference)
        self.assertTrue(high_precision_semantic_face_is_certified(difference))
        self.assertGreaterEqual(
            result.lexicographic_candidates_rejected_by_high_precision,
            0,
        )

    def test_global_constants_are_not_validation_derived_tolerances(self) -> None:
        self.assertEqual(HIGH_PRECISION_DECIMAL_DIGITS, 80)
        self.assertEqual(MAX_HIGH_PRECISION_FACE_REJECTIONS, 1024)
        self.assertEqual(SEMANTIC_TIE_TOLERANCE, 1e-12)


if __name__ == "__main__":
    unittest.main()
