"""Tests for the fusion ensemble (Tasks 18.1 and 18.2).

Covers the calibrated weighted blend / stacking that combines the three
estimator families (physics, change-point, learned) into a single fused
estimate with a 90 % predictive interval (Req 5.4) and the per-flight
traceability of contributing/excluded estimators (Task 18.1), plus the gating
policy (Task 18.2): high-sigma / failed-estimator exclusion (Req 5.5), the
on-ground-flag zero-weight guard (Req 18.4), the no-estimate
(``ALL_ESTIMATORS_FAILED``) result when nothing is eligible (Req 5.6), and the
``WIDE_CONFIDENCE_INTERVAL`` / ``ESTIMATOR_DISAGREEMENT`` low-confidence flags.

The P5/P14 property tests (Task 18.3) are at the bottom of this module.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from tdz.config.schema import FusionConfig
from tdz.estimators.physics.base import (
    CONFIDENCE_LOW,
    CONFIDENCE_NORMAL,
    failed_estimate,
    make_estimate,
)
from tdz.fusion import (
    CONFIDENCE_NO_ESTIMATE,
    METHOD_STACKING,
    METHOD_WEIGHTED_BLEND,
    ON_GROUND_FLAG_METHOD_NAMES,
    CalibratedFusion,
    build_fusion,
)
from tdz.models import FailureReason, FlightRecord, RunwayReference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fusion_config(
    method: str = METHOD_WEIGHTED_BLEND,
    *,
    confidence_threshold_sigma: float = 5.0,
    low_confidence_ci_width_s: float = 1000.0,
    disagreement_threshold_s: float = 1000.0,
) -> FusionConfig:
    """A resolved fusion config (mirrors the loader defaults).

    The 18.2 low-confidence thresholds default to large values here so the 18.1
    combination tests observe ``"normal"`` confidence; the 18.2 gating tests pass
    explicit, tighter thresholds.
    """
    return FusionConfig(
        method=method,
        confidence_threshold_sigma=confidence_threshold_sigma,
        low_confidence_ci_width_ft=600.0,
        low_confidence_ci_width_s=low_confidence_ci_width_s,
        disagreement_threshold_s=disagreement_threshold_s,
    )


def _context() -> FlightRecord:
    """Minimal FlightRecord; the 18.1 combination does not read it."""
    runway = RunwayReference(
        threshold_lat=33.94,
        threshold_lon=-118.40,
        heading_deg=250.0,
        elevation_m=30.0,
        elevation_datum="HAE",
        geoid_undulation_m=0.0,
        length_m=3500.0,
        width_m=45.0,
        displaced=False,
    )
    empty = np.array([], dtype=float)
    return FlightRecord(
        flight_id="FUSE",
        aircraft_type="B738",
        ads_b_source="aireon",
        position_times=empty,
        velocity_times=empty,
        latitudes=empty,
        longitudes=empty,
        geometric_altitudes=empty,
        barometric_altitudes=empty,
        groundspeeds=empty,
        tracks=empty,
        baro_vertical_rates=empty,
        on_ground_flags=np.array([], dtype=bool),
        on_ground_transition_time=None,
        runway=runway,
    )


def _estimate(method_name: str, t_td: float, sigma_t: float):
    """A normal-confidence TDEstimate for a given family."""
    return make_estimate(
        t_td=t_td,
        sigma_t=sigma_t,
        confidence=CONFIDENCE_NORMAL,
        method_name=method_name,
        diagnostics={},
    )


def _three_family_estimates():
    """One estimate from each of the three families (physics/change-point/learned)."""
    return [
        _estimate("decel_knee", t_td=1000.0, sigma_t=1.0),       # physics
        _estimate("pelt", t_td=1002.0, sigma_t=2.0),             # change-point
        _estimate("lightgbm", t_td=1001.0, sigma_t=1.5),         # learned
    ]


# ---------------------------------------------------------------------------
# Weighted blend: fused t_td lies within the input range
# ---------------------------------------------------------------------------


def test_weighted_blend_t_td_between_min_and_max():
    fusion = CalibratedFusion(_fusion_config(METHOD_WEIGHTED_BLEND))
    estimates = _three_family_estimates()

    fused = fusion.fuse(estimates, _context())

    times = [e.t_td for e in estimates]
    # A convex combination of the inputs must lie within their span.
    assert min(times) <= fused.t_td <= max(times)
    assert fused.confidence == CONFIDENCE_NORMAL
    assert fused.reason_code is None


def test_inverse_variance_pulls_toward_most_certain_estimator():
    """The lowest-sigma (most certain) estimator dominates the blend."""
    fusion = CalibratedFusion(_fusion_config(METHOD_WEIGHTED_BLEND))
    estimates = [
        _estimate("decel_knee", t_td=1000.0, sigma_t=0.5),   # very certain
        _estimate("pelt", t_td=1010.0, sigma_t=5.0),         # uncertain
        _estimate("lightgbm", t_td=1010.0, sigma_t=5.0),     # uncertain
    ]

    fused = fusion.fuse(estimates, _context())

    midpoint = 0.5 * (1000.0 + 1010.0)
    # Far more precision on the t=1000 estimate -> fused time below the midpoint.
    assert fused.t_td < midpoint


def test_three_equal_estimates_average_to_their_common_value():
    fusion = CalibratedFusion(_fusion_config(METHOD_WEIGHTED_BLEND))
    estimates = [
        _estimate("decel_knee", t_td=500.0, sigma_t=1.0),
        _estimate("cusum", t_td=500.0, sigma_t=1.0),
        _estimate("sequence_model", t_td=500.0, sigma_t=1.0),
    ]

    fused = fusion.fuse(estimates, _context())

    assert fused.t_td == pytest.approx(500.0)
    # No disagreement -> fused sigma equals the equal-weight within-variance term.
    # within = sum (1/3)^2 * 1^2 = 1/3 ; between = 0 -> sigma = sqrt(1/3).
    assert fused.sigma_t == pytest.approx(math.sqrt(1.0 / 3.0))


# ---------------------------------------------------------------------------
# Predictive interval: lower < t_td < upper with positive width
# ---------------------------------------------------------------------------


def test_ci_bounds_bracket_t_td_with_positive_width():
    fusion = CalibratedFusion(_fusion_config(METHOD_WEIGHTED_BLEND))
    fused = fusion.fuse(_three_family_estimates(), _context())

    assert fused.ci_90_lower < fused.t_td < fused.ci_90_upper
    assert fused.ci_90_upper - fused.ci_90_lower > 0.0
    # Interval is symmetric about the fused time.
    assert (fused.t_td - fused.ci_90_lower) == pytest.approx(
        fused.ci_90_upper - fused.t_td
    )


def test_ci_width_scales_with_uncertainty():
    """Higher input uncertainty -> wider predictive interval."""
    fusion = CalibratedFusion(_fusion_config(METHOD_WEIGHTED_BLEND))

    tight = fusion.fuse(
        [
            _estimate("decel_knee", t_td=1000.0, sigma_t=0.5),
            _estimate("pelt", t_td=1000.0, sigma_t=0.5),
            _estimate("lightgbm", t_td=1000.0, sigma_t=0.5),
        ],
        _context(),
    )
    wide = fusion.fuse(
        [
            _estimate("decel_knee", t_td=1000.0, sigma_t=4.0),
            _estimate("pelt", t_td=1000.0, sigma_t=4.0),
            _estimate("lightgbm", t_td=1000.0, sigma_t=4.0),
        ],
        _context(),
    )

    tight_width = tight.ci_90_upper - tight.ci_90_lower
    wide_width = wide.ci_90_upper - wide.ci_90_lower
    assert wide_width > tight_width


def test_disagreement_widens_interval_beyond_agreement():
    """Spread between estimators inflates the fused sigma (between-variance)."""
    fusion = CalibratedFusion(_fusion_config(METHOD_WEIGHTED_BLEND))

    agree = fusion.fuse(
        [
            _estimate("decel_knee", t_td=1000.0, sigma_t=2.0),
            _estimate("pelt", t_td=1000.0, sigma_t=2.0),
            _estimate("lightgbm", t_td=1000.0, sigma_t=2.0),
        ],
        _context(),
    )
    disagree = fusion.fuse(
        [
            _estimate("decel_knee", t_td=990.0, sigma_t=2.0),
            _estimate("pelt", t_td=1000.0, sigma_t=2.0),
            _estimate("lightgbm", t_td=1010.0, sigma_t=2.0),
        ],
        _context(),
    )

    assert disagree.sigma_t > agree.sigma_t


# ---------------------------------------------------------------------------
# Traceability: contributing / excluded lists and per_estimator_results
# ---------------------------------------------------------------------------


def test_contributing_and_per_estimator_results_populated():
    estimates = _three_family_estimates()
    fused = CalibratedFusion(_fusion_config()).fuse(estimates, _context())

    assert set(fused.contributing_estimators) == {"decel_knee", "pelt", "lightgbm"}
    assert fused.excluded_estimators == []
    # Every estimate is recorded for traceability, keyed by method name.
    assert set(fused.per_estimator_results) == {"decel_knee", "pelt", "lightgbm"}
    for name, est in fused.per_estimator_results.items():
        assert est.method_name == name


def test_failed_estimator_is_excluded_with_reason_but_recorded():
    estimates = [
        _estimate("decel_knee", t_td=1000.0, sigma_t=1.0),
        _estimate("pelt", t_td=1001.0, sigma_t=1.5),
        failed_estimate("flare_crossing", FailureReason.INSUFFICIENT_FLARE_SAMPLES),
    ]

    fused = CalibratedFusion(_fusion_config()).fuse(estimates, _context())

    assert set(fused.contributing_estimators) == {"decel_knee", "pelt"}
    assert len(fused.excluded_estimators) == 1
    excluded_entry = fused.excluded_estimators[0]
    assert excluded_entry.startswith("flare_crossing:")
    assert FailureReason.INSUFFICIENT_FLARE_SAMPLES.value in excluded_entry
    # The excluded estimate is still recorded for full traceability.
    assert "flare_crossing" in fused.per_estimator_results
    # A failed estimator does not perturb the fused time (only the two good ones).
    assert 1000.0 <= fused.t_td <= 1001.0


def test_non_positive_sigma_estimate_is_excluded():
    estimates = [
        _estimate("decel_knee", t_td=1000.0, sigma_t=1.0),
        _estimate("pelt", t_td=1001.0, sigma_t=1.0),
        _estimate("lightgbm", t_td=999.0, sigma_t=0.0),  # invalid uncertainty
    ]

    fused = CalibratedFusion(_fusion_config()).fuse(estimates, _context())

    assert "lightgbm" not in fused.contributing_estimators
    assert any(e.startswith("lightgbm:") for e in fused.excluded_estimators)
    assert math.isfinite(fused.t_td)


def test_all_estimators_failed_yields_no_estimate():
    estimates = [
        failed_estimate("decel_knee", FailureReason.NO_GROUNDSPEED),
        failed_estimate("pelt", FailureReason.INSUFFICIENT_SAMPLES),
        failed_estimate("lightgbm", FailureReason.ALL_ESTIMATORS_FAILED),
    ]

    fused = CalibratedFusion(_fusion_config()).fuse(estimates, _context())

    assert fused.confidence == CONFIDENCE_NO_ESTIMATE
    assert fused.reason_code == FailureReason.ALL_ESTIMATORS_FAILED.value
    assert fused.contributing_estimators == []
    assert len(fused.excluded_estimators) == 3
    assert math.isnan(fused.t_td)


# ---------------------------------------------------------------------------
# Stacking method
# ---------------------------------------------------------------------------


def test_stacking_with_calibrated_weights_dominates_blend():
    """Calibrated stacking coefficients override inverse-variance weighting."""
    config = _fusion_config(METHOD_STACKING)
    # Put almost all weight on the physics estimate despite equal sigmas.
    stacking_weights = {"decel_knee": 100.0, "pelt": 1.0, "lightgbm": 1.0}
    fusion = CalibratedFusion(config, stacking_weights=stacking_weights)

    estimates = [
        _estimate("decel_knee", t_td=1000.0, sigma_t=2.0),
        _estimate("pelt", t_td=1010.0, sigma_t=2.0),
        _estimate("lightgbm", t_td=1010.0, sigma_t=2.0),
    ]
    fused = fusion.fuse(estimates, _context())

    # Heavily weighted toward t=1000 -> fused time very close to it.
    assert fused.t_td < 1001.0


def test_stacking_without_weights_falls_back_to_inverse_variance():
    estimates = _three_family_estimates()
    stacking = CalibratedFusion(_fusion_config(METHOD_STACKING))
    blend = CalibratedFusion(_fusion_config(METHOD_WEIGHTED_BLEND))

    fused_stacking = stacking.fuse(estimates, _context())
    fused_blend = blend.fuse(estimates, _context())

    assert fused_stacking.t_td == pytest.approx(fused_blend.t_td)
    assert fused_stacking.sigma_t == pytest.approx(fused_blend.sigma_t)


def test_stacking_all_zero_weights_falls_back_to_inverse_variance():
    """Degenerate calibration (all eligible coefficients zero) must not NaN out.

    If every eligible estimator is assigned a zero stacking coefficient the
    normalisation would divide by zero; the blend falls back to inverse-variance
    weighting and still produces a finite fused estimate.
    """
    estimates = _three_family_estimates()
    zero_weights = {est.method_name: 0.0 for est in estimates}
    stacking = CalibratedFusion(
        _fusion_config(METHOD_STACKING), stacking_weights=zero_weights
    )
    blend = CalibratedFusion(_fusion_config(METHOD_WEIGHTED_BLEND))

    fused_stacking = stacking.fuse(estimates, _context())
    fused_blend = blend.fuse(estimates, _context())

    assert math.isfinite(fused_stacking.t_td)
    assert math.isfinite(fused_stacking.sigma_t)
    assert fused_stacking.t_td == pytest.approx(fused_blend.t_td)
    assert fused_stacking.sigma_t == pytest.approx(fused_blend.sigma_t)


def test_build_fusion_returns_configured_ensemble():
    fusion = build_fusion(_fusion_config(METHOD_WEIGHTED_BLEND))
    assert isinstance(fusion, CalibratedFusion)
    fused = fusion.fuse(_three_family_estimates(), _context())
    assert fused.confidence == CONFIDENCE_NORMAL


# ---------------------------------------------------------------------------
# Task 18.2: high-sigma / failure gating (Req 5.5, Property 14)
# ---------------------------------------------------------------------------


def test_estimator_above_sigma_threshold_is_excluded_with_reason():
    """An estimate with sigma_t above the threshold is dropped (zero weight)."""
    config = _fusion_config(confidence_threshold_sigma=3.0)
    estimates = [
        _estimate("decel_knee", t_td=1000.0, sigma_t=1.0),
        _estimate("pelt", t_td=1001.0, sigma_t=1.5),
        _estimate("lightgbm", t_td=1050.0, sigma_t=9.0),  # too uncertain
    ]

    fused = CalibratedFusion(config).fuse(estimates, _context())

    assert "lightgbm" not in fused.contributing_estimators
    assert any(
        e.startswith("lightgbm:") and "sigma_t_above_threshold" in e
        for e in fused.excluded_estimators
    )
    # The dropped high-sigma estimate does not pull the fused time toward 1050.
    assert 1000.0 <= fused.t_td <= 1001.0


def test_estimator_at_threshold_is_retained():
    """Exclusion is strict (>) -- an estimate exactly at the threshold survives."""
    config = _fusion_config(confidence_threshold_sigma=2.0)
    estimates = [
        _estimate("decel_knee", t_td=1000.0, sigma_t=1.0),
        _estimate("pelt", t_td=1001.0, sigma_t=2.0),  # exactly at threshold
    ]

    fused = CalibratedFusion(config).fuse(estimates, _context())

    assert set(fused.contributing_estimators) == {"decel_knee", "pelt"}
    assert fused.excluded_estimators == []


def test_all_estimators_above_threshold_yields_no_estimate():
    """If every estimator is below-threshold-confidence -> no-estimate (Req 5.6)."""
    config = _fusion_config(confidence_threshold_sigma=2.0)
    estimates = [
        _estimate("decel_knee", t_td=1000.0, sigma_t=5.0),
        _estimate("pelt", t_td=1001.0, sigma_t=6.0),
        _estimate("lightgbm", t_td=1002.0, sigma_t=7.0),
    ]

    fused = CalibratedFusion(config).fuse(estimates, _context())

    assert fused.confidence == CONFIDENCE_NO_ESTIMATE
    assert fused.reason_code == FailureReason.ALL_ESTIMATORS_FAILED.value
    assert fused.contributing_estimators == []
    assert len(fused.excluded_estimators) == 3
    assert math.isnan(fused.t_td)


def test_mixed_failed_and_above_threshold_yields_no_estimate():
    config = _fusion_config(confidence_threshold_sigma=2.0)
    estimates = [
        failed_estimate("decel_knee", FailureReason.NO_GROUNDSPEED),
        _estimate("pelt", t_td=1001.0, sigma_t=9.0),  # above threshold
    ]

    fused = CalibratedFusion(config).fuse(estimates, _context())

    assert fused.confidence == CONFIDENCE_NO_ESTIMATE
    assert fused.reason_code == FailureReason.ALL_ESTIMATORS_FAILED.value


# ---------------------------------------------------------------------------
# Task 18.2: on-ground flag gets zero weight (Req 18.4)
# ---------------------------------------------------------------------------


def test_on_ground_flag_pseudo_estimate_is_excluded():
    """An on-ground-flag pseudo-estimate is given zero weight (Req 18.4)."""
    flag_name = sorted(ON_GROUND_FLAG_METHOD_NAMES)[0]
    estimates = [
        _estimate("decel_knee", t_td=1000.0, sigma_t=1.0),
        _estimate("pelt", t_td=1001.0, sigma_t=1.0),
        # The flag transition is far later; if it carried weight it would pull
        # the fused time up substantially.
        _estimate(flag_name, t_td=1100.0, sigma_t=1.0),
    ]

    fused = CalibratedFusion(_fusion_config()).fuse(estimates, _context())

    assert flag_name not in fused.contributing_estimators
    assert any(
        e.startswith(f"{flag_name}:") and "on_ground_flag_zero_weight" in e
        for e in fused.excluded_estimators
    )
    # Fused time is determined only by the two real estimators.
    assert 1000.0 <= fused.t_td <= 1001.0


# ---------------------------------------------------------------------------
# Task 18.2: WIDE_CONFIDENCE_INTERVAL flag
# ---------------------------------------------------------------------------


def test_wide_confidence_interval_flagged_when_ci_exceeds_threshold():
    """A wide fused interval (agreeing but uncertain estimators) is flagged."""
    config = _fusion_config(low_confidence_ci_width_s=2.0, disagreement_threshold_s=1000.0)
    estimates = [
        _estimate("decel_knee", t_td=1000.0, sigma_t=4.0),
        _estimate("pelt", t_td=1000.0, sigma_t=4.0),
        _estimate("lightgbm", t_td=1000.0, sigma_t=4.0),
    ]

    fused = CalibratedFusion(config).fuse(estimates, _context())

    assert (fused.ci_90_upper - fused.ci_90_lower) > config.low_confidence_ci_width_s
    assert fused.confidence == CONFIDENCE_LOW
    assert fused.reason_code == FailureReason.WIDE_CONFIDENCE_INTERVAL.value
    # The estimate is still produced (interval reported, not suppressed).
    assert math.isfinite(fused.t_td)


def test_narrow_confidence_interval_is_normal():
    config = _fusion_config(low_confidence_ci_width_s=100.0, disagreement_threshold_s=100.0)
    estimates = [
        _estimate("decel_knee", t_td=1000.0, sigma_t=0.5),
        _estimate("pelt", t_td=1000.0, sigma_t=0.5),
        _estimate("lightgbm", t_td=1000.0, sigma_t=0.5),
    ]

    fused = CalibratedFusion(config).fuse(estimates, _context())

    assert fused.confidence == CONFIDENCE_NORMAL
    assert fused.reason_code is None


# ---------------------------------------------------------------------------
# Task 18.2: ESTIMATOR_DISAGREEMENT flag
# ---------------------------------------------------------------------------


def test_estimator_disagreement_flagged_on_high_spread():
    """High inter-estimator spread flags ESTIMATOR_DISAGREEMENT."""
    config = _fusion_config(disagreement_threshold_s=2.0, low_confidence_ci_width_s=1000.0)
    estimates = [
        _estimate("decel_knee", t_td=980.0, sigma_t=1.0),
        _estimate("pelt", t_td=1000.0, sigma_t=1.0),
        _estimate("lightgbm", t_td=1020.0, sigma_t=1.0),
    ]

    fused = CalibratedFusion(config).fuse(estimates, _context())

    assert fused.confidence == CONFIDENCE_LOW
    assert fused.reason_code == FailureReason.ESTIMATOR_DISAGREEMENT.value


def test_disagreement_takes_precedence_over_wide_ci():
    """When both fire, disagreement (the root cause) is the surfaced reason."""
    config = _fusion_config(disagreement_threshold_s=2.0, low_confidence_ci_width_s=2.0)
    estimates = [
        _estimate("decel_knee", t_td=980.0, sigma_t=1.0),
        _estimate("pelt", t_td=1000.0, sigma_t=1.0),
        _estimate("lightgbm", t_td=1020.0, sigma_t=1.0),
    ]

    fused = CalibratedFusion(config).fuse(estimates, _context())

    assert fused.confidence == CONFIDENCE_LOW
    assert fused.reason_code == FailureReason.ESTIMATOR_DISAGREEMENT.value


def test_single_eligible_estimator_does_not_trigger_disagreement():
    """A single contributor cannot 'disagree' -> no ESTIMATOR_DISAGREEMENT."""
    config = _fusion_config(
        confidence_threshold_sigma=3.0,
        disagreement_threshold_s=0.0,  # any spread would trip a multi-estimator case
        low_confidence_ci_width_s=1000.0,
    )
    estimates = [
        _estimate("decel_knee", t_td=1000.0, sigma_t=1.0),
        _estimate("pelt", t_td=1050.0, sigma_t=9.0),  # excluded (above threshold)
    ]

    fused = CalibratedFusion(config).fuse(estimates, _context())

    assert fused.contributing_estimators == ["decel_knee"]
    assert fused.confidence == CONFIDENCE_NORMAL
    assert fused.reason_code is None


# ---------------------------------------------------------------------------
# Task 18.3: property tests P14 (high-sigma down-weighting) and P5 (fused
# t_td respects the on-ground upper bound).
# ---------------------------------------------------------------------------

# A pool of real estimator method names (none of which is an on-ground-flag
# pseudo-estimate) to draw eligible estimators from.
_METHOD_POOL = (
    "decel_knee",
    "flare_crossing",
    "pelt",
    "cusum",
    "glrt",
    "jerk_onset",
    "lightgbm",
    "sequence_model",
    "hybrid_residual",
)

# Fixed confidence threshold used by the property tests below.
_P14_THRESHOLD_SIGMA = 5.0


def _context_with_transition(transition):
    """A FlightRecord like ``_context()`` but with a known on-ground transition."""
    return dataclasses.replace(_context(), on_ground_transition_time=transition)


@st.composite
def _mixed_sigma_estimates(draw):
    """A set of normal-confidence estimates spanning the confidence threshold.

    Every estimate is finite with a strictly-positive ``sigma_t`` (so the only
    reason any is excluded is the high-sigma gate), drawn so that at least one
    estimator reports ``sigma_t`` strictly above :data:`_P14_THRESHOLD_SIGMA`.
    """
    specs = draw(
        st.lists(
            st.tuples(
                st.sampled_from(_METHOD_POOL),
                st.floats(min_value=0.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.01, max_value=20.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=2,
            max_size=6,
            unique_by=lambda spec: spec[0],
        )
    )
    # P14's precondition: one or more estimators exceed the threshold.
    assume(any(sigma > _P14_THRESHOLD_SIGMA for _, _, sigma in specs))
    return specs


@pytest.mark.property
@given(specs=_mixed_sigma_estimates())
def test_p14_high_sigma_estimators_are_down_weighted_and_recorded(specs):
    """Feature: touchdown-point-detection, Property 14: High-Sigma Down-Weighting

    For any set of estimator outputs where one or more report ``sigma_t``
    exceeding the configured confidence threshold, the fusion ensemble excludes
    those estimators (weight strictly below nominal -> zero) and records the
    exclusion with a reason in ``excluded_estimators``; every estimator at or
    below the threshold still contributes.

    Validates: Requirements 5.5
    """
    config = _fusion_config(confidence_threshold_sigma=_P14_THRESHOLD_SIGMA)
    estimates = [_estimate(name, t_td, sigma) for name, t_td, sigma in specs]

    fused = CalibratedFusion(config).fuse(estimates, _context())

    for name, _t_td, sigma in specs:
        if sigma > _P14_THRESHOLD_SIGMA:
            # Above-threshold estimators carry no weight ...
            assert name not in fused.contributing_estimators
            # ... and the exclusion is recorded with a high-sigma reason.
            assert any(
                entry.startswith(f"{name}:") and "sigma_t_above_threshold" in entry
                for entry in fused.excluded_estimators
            )
        else:
            # At/below-threshold estimators (finite, positive sigma) contribute.
            assert name in fused.contributing_estimators
            assert all(
                not entry.startswith(f"{name}:") for entry in fused.excluded_estimators
            )


@st.composite
def _bounded_estimates_and_transition(draw):
    """A known on-ground transition plus eligible estimates at/below it.

    Mirrors what reaches the fusion in the real pipeline: each estimator's
    candidate ``t_td`` has already been clamped to ``<= transition`` by the
    estimator layer. Optionally includes an on-ground-flag pseudo-estimate
    (placed *after* the transition) to check it is given zero weight.
    """
    transition = draw(
        st.floats(min_value=100.0, max_value=2000.0, allow_nan=False, allow_infinity=False)
    )
    specs = draw(
        st.lists(
            st.tuples(
                st.sampled_from(_METHOD_POOL),
                # offset below (or at) the transition
                st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
                # sigma strictly below the threshold -> eligible
                st.floats(min_value=0.1, max_value=4.9, allow_nan=False, allow_infinity=False),
            ),
            min_size=1,
            max_size=5,
            unique_by=lambda spec: spec[0],
        )
    )
    include_flag = draw(st.booleans())
    return transition, specs, include_flag


@pytest.mark.property
@given(data=_bounded_estimates_and_transition())
def test_p5_fused_t_td_respects_on_ground_upper_bound(data):
    """Feature: touchdown-point-detection, Property 5: On-Ground Flag Upper Bound

    For any flight with a known on-ground transition time, when every
    contributing estimator's candidate ``t_td`` respects the upper bound
    (``t_td <= transition``), the fused ``t_td`` also respects it. Any
    on-ground-flag pseudo-estimate is given zero weight (excluded), so the flag
    can never pull the fused time past the bound.

    Validates: Requirements 18.4
    """
    transition, specs, include_flag = data

    estimates = [
        _estimate(name, transition - offset, sigma) for name, offset, sigma in specs
    ]
    flag_name = sorted(ON_GROUND_FLAG_METHOD_NAMES)[0]
    if include_flag:
        # A flag "estimate" well past the transition: if it carried any weight it
        # would push the fused time above the bound.
        estimates.append(_estimate(flag_name, transition + 100.0, 1.0))

    fused = CalibratedFusion(_fusion_config()).fuse(estimates, _context_with_transition(transition))

    # At least one real estimator is eligible, so a fused value is produced.
    assert fused.confidence != CONFIDENCE_NO_ESTIMATE
    assert math.isfinite(fused.t_td)
    # The fused touchdown time never exceeds the on-ground upper bound (a small
    # epsilon absorbs floating-point rounding in the weighted mean).
    assert fused.t_td <= transition + 1e-6

    if include_flag:
        # The on-ground flag is zero-weighted (excluded), never a contributor.
        assert flag_name not in fused.contributing_estimators
        assert any(
            entry.startswith(f"{flag_name}:") and "on_ground_flag_zero_weight" in entry
            for entry in fused.excluded_estimators
        )
