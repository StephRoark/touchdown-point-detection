"""Calibrated fusion of estimator outputs (Task 18.1).

This module implements the concrete :class:`~tdz.models.FusionEnsemble` that
combines the touchdown-time estimates produced by the three estimator families
(physics, change-point, learned) into a single fused estimate carrying a 90 %
predictive interval, plus the per-flight traceability the safety analysis needs:
which estimators contributed and which were excluded (and why).

Scope (Task 18.1)
-----------------
This file implements the *combination* logic only:

* the calibrated weighted blend / stacking selected by ``fusion.method``;
* the fused ``t_td``, ``sigma_t`` and 90 % predictive interval;
* the ``contributing_estimators`` / ``excluded_estimators`` lists and the
  ``per_estimator_results`` map.

The detailed *gating* policy -- down-weighting or excluding estimators whose
``sigma_t`` exceeds ``fusion.confidence_threshold_sigma`` or that report a
failure diagnostic, giving the on-ground flag zero weight, and raising the
``WIDE_CONFIDENCE_INTERVAL`` / ``ESTIMATOR_DISAGREEMENT`` low-confidence flags --
is implemented here (Task 18.2). It is layered on top of the 18.1 combination
maths: the eligibility partition lives in :func:`_classify_estimate` (so it is
decided in exactly one place) and the post-combination confidence flags are
decided in :meth:`CalibratedFusion._confidence_flags`. Confidence is reported as
``"normal"`` for a clean fusion, ``"low-confidence"`` (with a ``reason_code``)
when the fused interval is wide or the estimators disagree, and ``"no-estimate"``
(``ALL_ESTIMATORS_FAILED``) when no estimator is eligible.

Gating policy (Task 18.2, Requirements 5.5, 5.6, 18.4)
------------------------------------------------------
* **High-sigma / failed estimators are excluded** (zero weight) from the blend:
  any estimate reporting ``confidence == "failed"``, a non-finite ``t_td``, a
  non-positive/non-finite ``sigma_t``, or ``sigma_t`` strictly above
  ``fusion.confidence_threshold_sigma`` is dropped, with the reason recorded in
  :attr:`FusedEstimate.excluded_estimators` (Req 5.5; Property 14). Excluding
  entirely is the strongest form of "weight strictly below nominal".
* **The on-ground flag gets zero weight** (Req 18.4). The flag is never emitted
  as a :class:`TDEstimate` in the normal pipeline (the physics layer applies it
  as an upper *bound*, never as a measurement), but if an on-ground-flag pseudo-
  estimate (method name in :data:`ON_GROUND_FLAG_METHOD_NAMES`) ever reaches the
  fusion it is excluded so it can never carry weight.
* **No-estimate** (``ALL_ESTIMATORS_FAILED``) is emitted when *every* estimator
  is failed or below-threshold, i.e. nothing is eligible (Req 5.6).
* **WIDE_CONFIDENCE_INTERVAL** flags the fused estimate low-confidence when the
  fused 90 % CI width exceeds ``fusion.low_confidence_ci_width_s`` (the time-
  domain analog of the ``low_confidence_ci_width_ft`` distance threshold, which
  is applied later in the mapping layer once the distance CI exists).
* **ESTIMATOR_DISAGREEMENT** flags the fused estimate low-confidence when the
  inter-estimator spread (the weighted between-estimator 1-sigma) exceeds
  ``fusion.disagreement_threshold_s`` and at least two estimators contributed.
  Disagreement is checked first: it is the root cause that also widens the CI,
  so it is the more diagnostic single ``reason_code`` to surface.

Combination maths
-----------------
Given the eligible estimates ``(t_i, sigma_i)`` and non-negative blend weights
``a_i`` (normalised to sum to 1):

* fused time            ``t = sum_i a_i * t_i`` (a convex combination, so it
  always lies within ``[min t_i, max t_i]``);
* within-estimator var  ``sum_i a_i^2 * sigma_i^2`` (variance of the weighted
  mean; for inverse-variance weights this reduces to the classic
  ``1 / sum_i 1/sigma_i^2``);
* between-estimator var  ``sum_i a_i * (t_i - t)^2`` (the weighted spread, so
  disagreeing estimators honestly widen the interval rather than producing a
  falsely tight one);
* fused 1-sigma          ``sqrt(within + between)``.

The two weighting schemes:

* ``"weighted_blend"`` -- inverse-variance weights ``a_i ~ 1 / sigma_i^2`` (the
  sound default: more certain estimators count for more).
* ``"stacking"`` -- calibrated per-method coefficients learned on the calibration
  split (Task 19) and supplied via ``stacking_weights``. Until those coefficients
  exist this falls back to inverse-variance weighting (documented), so the method
  is selectable now and the calibrated coefficients can be dropped in later
  without an interface change.

Predictive interval
--------------------
The 90 % interval is ``t +/- z * sigma`` where ``z`` is the two-sided standard-
normal quantile for :data:`CI_COVERAGE`. ``CI_COVERAGE`` is structural -- it
defines what the ``ci_90_*`` fields *mean* -- not an estimation tunable, so it is
a module constant rather than a config value.

Units: SI throughout. ``t_td``/``sigma_t`` and the CI bounds are seconds.
"""

from __future__ import annotations

import math
from typing import Final, Optional

from scipy.stats import norm

from tdz.config.schema import FusionConfig
from tdz.estimators.physics.base import (
    CONFIDENCE_FAILED,
    CONFIDENCE_LOW,
    CONFIDENCE_NORMAL,
)
from tdz.models import (
    FailureReason,
    FlightRecord,
    FusedEstimate,
    FusionEnsemble,
    TDEstimate,
)

__all__ = [
    "CI_COVERAGE",
    "CONFIDENCE_NO_ESTIMATE",
    "ON_GROUND_FLAG_METHOD_NAMES",
    "METHOD_WEIGHTED_BLEND",
    "METHOD_STACKING",
    "CalibratedFusion",
    "build_fusion",
]

#: Coverage of the predictive interval reported on :class:`FusedEstimate`. This
#: is structural (it defines the meaning of the ``ci_90_lower``/``ci_90_upper``
#: fields), not an estimation tunable, hence a module constant.
CI_COVERAGE: Final[float] = 0.90

#: Two-sided standard-normal quantile for :data:`CI_COVERAGE` (~1.645 for 90 %).
#: Derived from the coverage rather than hard-coded so the interval and the
#: coverage can never drift apart.
_CI_Z: Final[float] = float(norm.ppf((1.0 + CI_COVERAGE) / 2.0))

#: The fusion output confidence class meaning "no estimate could be produced".
#: The ``"normal"``/``"low-confidence"`` classes are reused from the estimator
#: base so the strings are written in exactly one place.
CONFIDENCE_NO_ESTIMATE: Final[str] = "no-estimate"

#: ``fusion.method`` values understood by :class:`CalibratedFusion`.
METHOD_WEIGHTED_BLEND: Final[str] = "weighted_blend"
METHOD_STACKING: Final[str] = "stacking"

#: Method names identifying an on-ground-flag pseudo-estimate. The on-ground
#: flag is an *upper bound* on ``t_td`` (handled in the estimator layer), never a
#: measurement that is averaged in, so any estimate carrying one of these names
#: is given zero weight in the fusion (Req 18.4). In the normal pipeline no such
#: estimate is ever produced; this is a defensive guard so the flag can never
#: acquire weight even if one were passed in.
ON_GROUND_FLAG_METHOD_NAMES: Final[frozenset[str]] = frozenset(
    {"on_ground_flag", "on_ground"}
)


def _is_finite(value: Optional[float]) -> bool:
    """Return ``True`` when ``value`` is a finite real number."""
    return value is not None and not math.isnan(value) and not math.isinf(value)


def _classify_estimate(
    estimate: TDEstimate, *, confidence_threshold_sigma: float
) -> Optional[str]:
    """Decide whether an estimate may enter the blend.

    Returns ``None`` when the estimate is eligible, or a short reason string when
    it must be excluded (zero weight). The reason is surfaced in
    :attr:`FusedEstimate.excluded_estimators` for traceability (Req 5.5).

    An estimate is excluded when it:

    * is an on-ground-flag pseudo-estimate -- the flag is an upper bound, never a
      weighted measurement (Req 18.4);
    * reported ``confidence == "failed"`` (its ``t_td`` is ``NaN`` / ``sigma_t``
      is ``inf``);
    * has a non-finite ``t_td``;
    * has a non-finite or non-positive ``sigma_t`` (a zero/negative uncertainty
      makes the inverse-variance weight ill-defined);
    * reports ``sigma_t`` strictly above ``confidence_threshold_sigma`` -- the
      estimator is too uncertain to trust, so it is dropped entirely rather than
      diluting the blend (Req 5.5; Property 14). Excluding entirely is the
      strongest form of "a weight strictly below its nominal weight".
    """
    if estimate.method_name in ON_GROUND_FLAG_METHOD_NAMES:
        return "on_ground_flag_zero_weight"
    if estimate.confidence == CONFIDENCE_FAILED:
        reason = estimate.diagnostics.get("reason_code") if estimate.diagnostics else None
        return reason or "failed"
    if not _is_finite(estimate.t_td):
        return "non_finite_t_td"
    if not _is_finite(estimate.sigma_t) or estimate.sigma_t <= 0.0:
        return "non_positive_sigma_t"
    if estimate.sigma_t > confidence_threshold_sigma:
        return (
            f"sigma_t_above_threshold "
            f"(sigma_t={estimate.sigma_t:.4g} > {confidence_threshold_sigma:.4g})"
        )
    return None


def _inverse_variance_weights(sigmas: list[float]) -> list[float]:
    """Inverse-variance (precision) weights ``1 / sigma^2`` for each estimate."""
    return [1.0 / (sigma * sigma) for sigma in sigmas]


def _combine(
    times: list[float], sigmas: list[float], weights: list[float]
) -> tuple[float, float, float]:
    """Blend eligible estimates into a fused ``(t_td, sigma_t, between_std)``.

    ``weights`` are non-negative and need not be normalised; they are normalised
    here. The fused time is the convex combination of ``times``; the fused
    variance is the within-estimator variance of the weighted mean plus the
    between-estimator (disagreement) variance. ``between_std`` is the weighted
    between-estimator 1-sigma (``sqrt`` of the between-variance) -- the spread
    used to decide ``ESTIMATOR_DISAGREEMENT``. See the module docstring.
    """
    total = math.fsum(weights)
    norm_w = [w / total for w in weights]

    t_fused = math.fsum(w * t for w, t in zip(norm_w, times))
    within_var = math.fsum((w * w) * (s * s) for w, s in zip(norm_w, sigmas))
    between_var = math.fsum(w * (t - t_fused) ** 2 for w, t in zip(norm_w, times))
    sigma_fused = math.sqrt(within_var + between_var)
    between_std = math.sqrt(between_var)
    return t_fused, sigma_fused, between_std


class CalibratedFusion(FusionEnsemble):
    """Concrete :class:`FusionEnsemble`: calibrated weighted blend / stacking.

    Parameters
    ----------
    config:
        The resolved ``fusion`` configuration. ``config.method`` selects the
        weighting scheme (:data:`METHOD_WEIGHTED_BLEND` or
        :data:`METHOD_STACKING`).
    stacking_weights:
        Optional calibrated per-method coefficients
        (``{method_name: non_negative_weight}``) learned on the calibration split
        (Task 19). Used only when ``config.method == "stacking"``; when absent (or
        when none of the eligible estimators have a coefficient) the blend falls
        back to inverse-variance weighting.
    """

    def __init__(
        self,
        config: FusionConfig,
        stacking_weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.config = config
        self.stacking_weights = dict(stacking_weights) if stacking_weights else {}

    # -- weighting ---------------------------------------------------------

    def _blend_weights(
        self, names: list[str], sigmas: list[float]
    ) -> list[float]:
        """Compute the (un-normalised) blend weight for each eligible estimate."""
        inverse_variance = _inverse_variance_weights(sigmas)
        if self.config.method != METHOD_STACKING or not self.stacking_weights:
            return inverse_variance

        # Stacking: use a calibrated coefficient where available, falling back to
        # the inverse-variance weight for any method without a calibrated entry.
        weights = [
            self.stacking_weights.get(name, iv)
            for name, iv in zip(names, inverse_variance)
        ]
        # Degenerate calibration (every eligible estimator assigned zero weight)
        # would make the blend's normalisation ill-defined; fall back to
        # inverse-variance weighting rather than emit NaNs.
        if math.fsum(weights) <= 0.0:
            return inverse_variance
        return weights

    # -- gating / confidence ----------------------------------------------

    def _confidence_flags(
        self, ci_width: float, between_std: float, n_eligible: int
    ) -> tuple[str, Optional[str]]:
        """Decide the fused confidence class and reason code (Task 18.2).

        Both checks degrade the estimate to ``"low-confidence"`` with a reason
        code; the estimate is still produced (the interval is reported with the
        flag, never suppressed).

        ``ESTIMATOR_DISAGREEMENT`` is checked first: a high inter-estimator
        spread is the root cause that also inflates the fused CI, so it is the
        more diagnostic single ``reason_code`` to surface. It requires at least
        two contributing estimators (a single estimator cannot "disagree").
        ``WIDE_CONFIDENCE_INTERVAL`` then catches a fused interval that is wide
        even when the estimators broadly agree (e.g. all individually uncertain).
        """
        if (
            n_eligible >= 2
            and between_std > self.config.disagreement_threshold_s
        ):
            return CONFIDENCE_LOW, FailureReason.ESTIMATOR_DISAGREEMENT.value
        if ci_width > self.config.low_confidence_ci_width_s:
            return CONFIDENCE_LOW, FailureReason.WIDE_CONFIDENCE_INTERVAL.value
        return CONFIDENCE_NORMAL, None

    # -- fusion ------------------------------------------------------------

    def fuse(
        self, estimates: list[TDEstimate], context: FlightRecord
    ) -> FusedEstimate:
        """Combine estimator outputs into a calibrated fused estimate.

        Applies the Task-18.2 gating policy: high-sigma/failed estimators and the
        on-ground flag are excluded (zero weight) with reasons recorded; a
        no-estimate (``ALL_ESTIMATORS_FAILED``) result is emitted when nothing is
        eligible; and the fused estimate is flagged ``WIDE_CONFIDENCE_INTERVAL``
        or ``ESTIMATOR_DISAGREEMENT`` low-confidence as appropriate.

        ``context`` (the flight record) is accepted for interface compatibility;
        the gating implemented here reads only the per-estimator outputs.
        """
        per_estimator_results: dict[str, TDEstimate] = {
            est.method_name: est for est in estimates
        }

        eligible: list[TDEstimate] = []
        excluded: list[str] = []
        for est in estimates:
            reason = _classify_estimate(
                est, confidence_threshold_sigma=self.config.confidence_threshold_sigma
            )
            if reason is None:
                eligible.append(est)
            else:
                excluded.append(f"{est.method_name}: {reason}")

        if not eligible:
            # Every estimator failed or was below the confidence threshold: emit a
            # no-estimate result rather than an unreliable fused value (Req 5.6).
            return FusedEstimate(
                t_td=float("nan"),
                sigma_t=float("inf"),
                ci_90_lower=float("nan"),
                ci_90_upper=float("nan"),
                confidence=CONFIDENCE_NO_ESTIMATE,
                reason_code=FailureReason.ALL_ESTIMATORS_FAILED.value,
                contributing_estimators=[],
                excluded_estimators=excluded,
                per_estimator_results=per_estimator_results,
            )

        names = [est.method_name for est in eligible]
        times = [float(est.t_td) for est in eligible]
        sigmas = [float(est.sigma_t) for est in eligible]

        weights = self._blend_weights(names, sigmas)
        t_fused, sigma_fused, between_std = _combine(times, sigmas, weights)

        ci_lower = t_fused - _CI_Z * sigma_fused
        ci_upper = t_fused + _CI_Z * sigma_fused
        ci_width = ci_upper - ci_lower

        confidence, reason_code = self._confidence_flags(
            ci_width, between_std, len(eligible)
        )

        return FusedEstimate(
            t_td=t_fused,
            sigma_t=sigma_fused,
            ci_90_lower=ci_lower,
            ci_90_upper=ci_upper,
            confidence=confidence,
            reason_code=reason_code,
            contributing_estimators=names,
            excluded_estimators=excluded,
            per_estimator_results=per_estimator_results,
        )


def build_fusion(
    config: FusionConfig,
    stacking_weights: Optional[dict[str, float]] = None,
) -> FusionEnsemble:
    """Construct the configured fusion ensemble.

    A thin factory mirroring the estimator construction pattern: callers pass the
    resolved ``fusion`` config (and optionally calibrated stacking coefficients)
    and receive the concrete :class:`FusionEnsemble`. Kept separate from the
    class so later methods can be dispatched here without changing call sites.
    """
    return CalibratedFusion(config, stacking_weights=stacking_weights)
