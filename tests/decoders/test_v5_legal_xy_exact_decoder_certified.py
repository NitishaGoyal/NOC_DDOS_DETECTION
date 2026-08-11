from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from src.decoders import v5_legal_xy_exact_decoder as legacy
from src.decoders import v5_legal_xy_exact_decoder_certified as certified
from src.decoders.v5_xy_route_library import ROUTE_TABLE


class TestV5CertifiedExactDecoder(unittest.TestCase):
    def test_policy_is_frozen(self) -> None:
        policy = certified.certification_policy()
        self.assertEqual(
            policy["policy_id"],
            "V5P2-D1-6ce8cc7c76b8d2cff367631077d65f20",
        )
        self.assertEqual(policy["mip_gap_reporting_ceiling"], 1e-12)
        self.assertEqual(policy["integrality_abs_tolerance"], 1e-9)
        self.assertTrue(policy["independent_certificate_gates_mandatory"])

    def test_observed_p2_residue_passes_gap_gate(self) -> None:
        self.assertTrue(
            certified.mip_gap_is_numerical_zero(
                2.4064574342771067e-14
            )
        )
        self.assertFalse(
            certified.mip_gap_is_numerical_zero(
                float(np.nextafter(1e-12, np.inf))
            )
        )
        self.assertFalse(certified.mip_gap_is_numerical_zero(-1e-16))
        self.assertFalse(certified.mip_gap_is_numerical_zero(np.inf))

    def test_decode_emits_two_full_certificates(self) -> None:
        zeros = np.zeros(16, dtype=np.float64)
        result = certified.decode_best_attack_hypothesis(
            0.0,
            np.zeros(4, dtype=np.float64),
            zeros,
            zeros,
            zeros,
            zeros,
        )
        self.assertEqual(result.attacker_count, 1)
        self.assertEqual(result.route_ids, (0,))
        self.assertEqual(result.certificate_status, "CERTIFIED")
        self.assertEqual(
            result.certificate_policy_id,
            "V5P2-D1-6ce8cc7c76b8d2cff367631077d65f20",
        )
        for certificate in (
            result.primary_certificate,
            result.lexicographic_certificate,
        ):
            self.assertEqual(certificate.certificate_status, "CERTIFIED")
            self.assertTrue(certificate.solver_success)
            self.assertEqual(certificate.solver_status, 0)
            self.assertLessEqual(
                certificate.raw_reported_mip_gap,
                certificate.gap_ceiling,
            )
            self.assertLessEqual(
                certificate.maximum_integrality_error,
                certificate.integrality_tolerance,
            )
            self.assertLessEqual(
                certificate.maximum_bound_violation,
                certificate.bound_tolerance,
            )
            self.assertLessEqual(
                certificate.maximum_linear_constraint_violation,
                certificate.linear_constraint_tolerance,
            )
            self.assertLessEqual(
                certificate.objective_absolute_difference,
                certificate.objective_allowed_difference,
            )
            self.assertEqual(certificate.attacker_count, 1)
        self.assertEqual(
            result.lexicographic_certificate.route_ids,
            result.route_ids,
        )

    def test_legacy_and_certified_semantics_match(self) -> None:
        for seed in (11, 21, 31):
            rng = np.random.default_rng(seed)
            args = (
                float(rng.normal()),
                rng.normal(size=4),
                rng.normal(size=16),
                rng.normal(size=16),
                rng.normal(size=16),
                rng.normal(size=16),
            )
            old = legacy.decode_best_attack_hypothesis(*args)
            new = certified.decode_best_attack_hypothesis(*args)
            self.assertEqual(old.attacker_count, new.attacker_count)
            self.assertEqual(old.route_ids, new.route_ids)
            self.assertEqual(old.source_mask, new.source_mask)
            self.assertEqual(old.transit_mask, new.transit_mask)
            self.assertEqual(old.victim_mask, new.victim_mask)
            self.assertEqual(old.path_mask, new.path_mask)
            self.assertEqual(old.margin.hex(), new.margin.hex())

    def _base_result(self):
        model = certified._build_model(tuple(ROUTE_TABLE))
        objective = np.zeros(model.variable_count, dtype=np.float64)
        result = certified._solve_milp(
            objective,
            model,
            model.matrix,
            model.lower,
            model.upper,
        )
        certificate = certified._certify_milp_result(
            "synthetic",
            result,
            objective,
            model,
            model.matrix,
            model.lower,
            model.upper,
        )
        self.assertEqual(certificate.certificate_status, "CERTIFIED")
        return model, objective, result

    def test_observed_gap_is_accepted_only_with_full_certificate(self) -> None:
        model, objective, result = self._base_result()
        forged = SimpleNamespace(
            success=result.success,
            status=result.status,
            message=result.message,
            mip_gap=2.4064574342771067e-14,
            x=np.asarray(result.x).copy(),
            fun=float(result.fun),
        )
        certificate = certified._certify_milp_result(
            "observed-gap",
            forged,
            objective,
            model,
            model.matrix,
            model.lower,
            model.upper,
        )
        self.assertEqual(certificate.certificate_status, "CERTIFIED")

    def test_material_gap_fails_hard(self) -> None:
        model, objective, result = self._base_result()
        forged = SimpleNamespace(
            success=True,
            status=0,
            message="synthetic",
            mip_gap=1e-6,
            x=np.asarray(result.x).copy(),
            fun=float(result.fun),
        )
        with self.assertRaisesRegex(
            certified.ExactDecoderError,
            "MATERIAL_REPORTED_MIP_GAP",
        ):
            certified._certify_milp_result(
                "material-gap",
                forged,
                objective,
                model,
                model.matrix,
                model.lower,
                model.upper,
            )

    def test_integrality_failure_fails_hard(self) -> None:
        model, objective, result = self._base_result()
        bad_x = np.asarray(result.x).copy()
        bad_x[0] = 0.5
        forged = SimpleNamespace(
            success=True,
            status=0,
            message="synthetic",
            mip_gap=0.0,
            x=bad_x,
            fun=float(np.dot(objective, bad_x)),
        )
        with self.assertRaisesRegex(
            certified.ExactDecoderError,
            "INTEGRALITY_CERTIFICATE_FAILURE",
        ):
            certified._certify_milp_result(
                "integrality",
                forged,
                objective,
                model,
                model.matrix,
                model.lower,
                model.upper,
            )

    def test_linear_feasibility_failure_fails_hard(self) -> None:
        model, objective, _ = self._base_result()
        bad_x = np.zeros(model.variable_count, dtype=np.float64)
        forged = SimpleNamespace(
            success=True,
            status=0,
            message="synthetic",
            mip_gap=0.0,
            x=bad_x,
            fun=0.0,
        )
        with self.assertRaisesRegex(
            certified.ExactDecoderError,
            "LINEAR_FEASIBILITY_CERTIFICATE_FAILURE",
        ):
            certified._certify_milp_result(
                "linear",
                forged,
                objective,
                model,
                model.matrix,
                model.lower,
                model.upper,
            )

    def test_objective_recomputation_failure_fails_hard(self) -> None:
        model, objective, result = self._base_result()
        forged = SimpleNamespace(
            success=True,
            status=0,
            message="synthetic",
            mip_gap=0.0,
            x=np.asarray(result.x).copy(),
            fun=float(result.fun) + 1.0,
        )
        with self.assertRaisesRegex(
            certified.ExactDecoderError,
            "OBJECTIVE_RECOMPUTATION_FAILURE",
        ):
            certified._certify_milp_result(
                "objective",
                forged,
                objective,
                model,
                model.matrix,
                model.lower,
                model.upper,
            )


if __name__ == "__main__":
    unittest.main()
