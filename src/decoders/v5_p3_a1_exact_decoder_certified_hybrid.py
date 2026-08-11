"""V5-P3 certified exact-decoder numerical policy.

Policy:
  1. Run the immutable canonical P2-certified decoder with HiGHS presolve.
  2. Only if that decoder raises its ExactDecoderError, retry the separately
     frozen presolve-disabled implementation.
  3. Return a result only when the selected implementation completes its
     existing full certificate. If both certified paths fail, fail closed.

This module does not change the MILP, objective, routes, tolerances,
thresholds, or A1 margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.decoders import v5_legal_xy_exact_decoder_certified as _canonical
from src.decoders import v5_p3_a1_exact_decoder_presolve_false as _fallback


class HybridExactDecoderError(RuntimeError):
    """Both certified numerical implementations failed."""


@dataclass(frozen=True)
class HybridHypothesis:
    hypothesis: Any
    selected_path: str
    canonical_error: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.hypothesis, name)


def decode_best_attack_hypothesis(*args, **kwargs) -> HybridHypothesis:
    try:
        value = _canonical.decode_best_attack_hypothesis(*args, **kwargs)
        return HybridHypothesis(
            hypothesis=value,
            selected_path="canonical_presolve_true",
            canonical_error=None,
        )
    except _canonical.ExactDecoderError as canonical_error:
        try:
            value = _fallback.decode_best_attack_hypothesis(*args, **kwargs)
            return HybridHypothesis(
                hypothesis=value,
                selected_path="certified_fallback_presolve_false",
                canonical_error=repr(canonical_error),
            )
        except _fallback.ExactDecoderError as fallback_error:
            raise HybridExactDecoderError(
                "BOTH_CERTIFIED_NUMERICAL_PATHS_FAILED: "
                f"canonical={canonical_error!r}; "
                f"fallback={fallback_error!r}"
            ) from fallback_error


def apply_margin_threshold(
    hypothesis: HybridHypothesis,
    margin_threshold: float,
):
    if not isinstance(hypothesis, HybridHypothesis):
        raise TypeError(
            "hybrid apply_margin_threshold requires HybridHypothesis"
        )
    if hypothesis.selected_path == "canonical_presolve_true":
        return _canonical.apply_margin_threshold(
            hypothesis.hypothesis,
            margin_threshold,
        )
    if hypothesis.selected_path == "certified_fallback_presolve_false":
        return _fallback.apply_margin_threshold(
            hypothesis.hypothesis,
            margin_threshold,
        )
    raise HybridExactDecoderError(
        f"unknown selected path: {hypothesis.selected_path}"
    )


def solver_identity():
    return {
        "policy": (
            "canonical presolve=True first; certified presolve=False "
            "fallback only on canonical ExactDecoderError"
        ),
        "canonical": _canonical.solver_identity(),
        "fallback": _fallback.solver_identity(),
    }


def certification_policy():
    return {
        "policy": "fail_closed_two_numerical_paths",
        "canonical": _canonical.certification_policy(),
        "fallback": _fallback.certification_policy(),
        "tolerance_relaxation": False,
    }
