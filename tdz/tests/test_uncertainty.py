"""Tests for uncertainty quantification and calibration (Task 19).

Covers the conformal interval calibration (Req 4.3, 4.4), the gap-proportional
(Req 9.2), post-transition-starvation (Req 9.6) and missing-lever-arm (Req 7.5)
widening, the low-confidence-not-suppress behavior (Req 4.5), and the two
property tests:

* **P6** -- CI validity: for any normal/low-confidence estimate the time and
  distance 90 % intervals satisfy ``lower < point < upper`` with positive width.
* **P7** -- gap-proportional widening: a gap of duration ``G`` within +/-30 s of
  ``t_td`` at nominal cadence ``C`` widens the interval by at least ``G / C``.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from tdz.config.schema import UncertaintyConfig
from tdz.estimators.physics.base import CONFIDENCE_LOW, CONFIDENCE_NORMAL
from tdz.fusion.ensemble import CONFIDENCE_NO_ESTIMATE
from tdz.models import FailureReason, FlightRecord, FusedEstimate, RunwayReference
from tdz.timebase import KNOTS_TO_MPS
from tdz.uncertainty import (
    ConformalCalibrator,
    UncertaintyQuantifier,
    gaussian_multiplier,
)
from tdz.uncertainty.quantifier import M_TO_FT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uncertainty_config(
    *,
    coverage_target: float = 0.90,
    nominal_cadence_s: float = 5.0,
    gap_window_half_width_s: float = 30.0,
    gap_min_duration_s: float = 10.0,
    missing_lever_arm_widening_factor: float = 1.5,
    post_transition_window_s: float = 15.0,
    min_post_transition_samples: int = 2,
    starvation_widening_factor: float = 1.5,
) -> UncertaintyConfig:
    return UncertaintyConfig(
        coverage_target=coverage_target,
        nominal_cadence_s=nominal_cadence_s,
        gap_window_half_width_s=gap_window_half_width_s,
        gap_min_duration_s=gap_min_duration_s,
        missing_lever_arm_widening_factor=missing_lever_arm_widening_factor,
        post_transition_window_s=post_transition_window_s,
        min_post_transition_samples=min_post_transition_samples,
        starvation_widening_factor=starvation_widening_factor,
    )


def _runway() -> RunwayReference:
    return RunwayReference(
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


def _flight(
    position_times: np.ndarray,
    *,
    transition=None,
    latitudes=None,
) -> FlightRecord:
    """Minimal FlightRecord carrying the fields the widening logic reads."""
    position_times = np.asarray(position_times, dtype=float)
    n = position_times.size
    if latitudes is None:
        latitudes = np.full(n, 33.94, dtype=float)
    latitudes = np.asarray(latitudes, dtype=float)
    empty = np.array([], dtype=float)
    return FlightRecord(
        flight_id="UNC",
        aircraft_type="B738",
        ads_b_source="aireon",
        position_times=position_times,
        velocity_times=position_times.copy(),
        latitudes=latitudes,
        longitudes=np.full(n, -118.40, dtype=float),
        geometric_altitudes=np.zeros(n, dtype=float),
        barometric_altitudes=np.zeros(n, dtype=float),
        groundspeeds=empty,
        tracks=empty,
        baro_vertical_rates=empty,
        on_ground_flags=np.zeros(n, dtype=bool),
        on_ground_transition_time=transition,
        runway=_runway(),
    )


def _fused(
    t_td: float,
    sigma_t: float,
    *,
    confidence: str = CONFIDENCE_NORMAL,
    reason_code=None,
) -> FusedEstimate:
    return FusedEstimate(
        t_td=t_td,
        sigma_t=sigma_t,
        ci_90_lower=t_td - 1.645 * sigma_t,
        ci_90_upper=t_td + 1.645 * sigma_t,
        confidence=confidence,
        reason_code=reason_code,
        contributing_estimators=["decel_knee", "pelt"],
        excluded_estimators=[],
        per_estimator_results={},
    )


def _uniform_times(t_td: float, cadence: float, n_each_side: int) -> np.ndarray:
    """Uniformly-spaced position times centered on ``t_td`` at ``cadence``."""
    ks = np.arange(-n_each_side, n_each_side + 1, dtype=float)
    return t_td + ks * cadence


# ---------------------------------------------------------------------------
# Conformal calibration (Req 4.3, 4.4)
# ---------------------------------------------------------------------------


def test_gaussian_multiplier_is_z_for_90pct():
    assert gaussian_multiplier(0.90) == pytest.approx(1.6448536, rel=1e-5)


def test_conformal_fit_falls_back_to_gaussian_without_data():
    cal = ConformalCalibrator.fit([], [], [], coverage_target=0.90)
    assert cal.n_calibration == 0
    assert cal.multiplier == pytest.approx(gaussian_multiplier(0.90))


def test_conformal_fit_recovers_multiplier_and_achieves_coverage():
    """A conformal multiplier fit on residuals achieves >= target coverage."""
    rng = np.random.default_rng(0)
    n = 2000
    sigmas = np.full(n, 1.0)
    # True residuals are ~2x the reported sigma -> raw z would under-cover.
    residuals = rng.normal(0.0, 2.0, size=n)
    points = np.zeros(n)
    truths = points + residuals

    cal = ConformalCalibrator.fit(points, truths, sigmas, coverage_target=0.90)

    # Empirical coverage of point +/- multiplier*sigma on a fresh sample.
    fresh = rng.normal(0.0, 2.0, size=n)
    covered = np.abs(fresh) <= cal.multiplier * 1.0
    coverage = float(np.mean(covered))
    assert 0.85 <= coverage <= 0.97
    # And it is wider than the naive Gaussian z (which would under-cover here).
    assert cal.multiplier > gaussian_multiplier(0.90)


def test_conformal_scaled_by_sigma():
    cal = ConformalCalibrator(multiplier=2.0, coverage_target=0.90, n_calibration=10)
    assert cal.half_width(3.0) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Gap-proportional widening (Req 9.2) -- worked example
# ---------------------------------------------------------------------------


def test_gap_10s_at_5s_cadence_doubles_width():
    """A 10 s gap at 5 s cadence doubles the interval width (Req 9.2 example)."""
    config = _uncertainty_config(nominal_cadence_s=5.0, gap_min_duration_s=8.0)
    q = UncertaintyQuantifier(config)
    t_td = 1000.0

    base = q.quantify(
        _fused(t_td, 1.0),
        _flight(_uniform_times(t_td, 5.0, 5)),
        groundspeed_at_td_mps=60.0,
        along_runway_distance_m=500.0,
    )
    # A single 10 s gap straddling t_td (samples at +/-5 s and beyond at 5 s).
    gapped_times = np.array(
        [t_td - 15.0, t_td - 10.0, t_td - 5.0, t_td + 5.0, t_td + 10.0, t_td + 15.0]
    )
    gapped = q.quantify(
        _fused(t_td, 1.0),
        _flight(gapped_times),
        groundspeed_at_td_mps=60.0,
        along_runway_distance_m=500.0,
    )

    base_w = base.time_ci_90_upper_s - base.time_ci_90_lower_s
    gap_w = gapped.time_ci_90_upper_s - gapped.time_ci_90_lower_s
    # 10 s gap / 5 s cadence = 2x.
    assert gap_w == pytest.approx(2.0 * base_w, rel=1e-9)
    assert gapped.diagnostics["gap_widening_factor"] == pytest.approx(2.0)


def test_normal_cadence_does_not_widen():
    config = _uncertainty_config(nominal_cadence_s=5.0, gap_min_duration_s=8.0)
    q = UncertaintyQuantifier(config)
    t_td = 1000.0
    res = q.quantify(
        _fused(t_td, 1.0),
        _flight(_uniform_times(t_td, 5.0, 6)),
        groundspeed_at_td_mps=60.0,
        along_runway_distance_m=500.0,
    )
    assert res.diagnostics["gap_widening_factor"] == pytest.approx(1.0)


def test_gap_outside_window_does_not_widen():
    """A gap far from t_td (outside +/-window) does not widen the interval."""
    config = _uncertainty_config(
        nominal_cadence_s=5.0, gap_min_duration_s=8.0, gap_window_half_width_s=30.0
    )
    q = UncertaintyQuantifier(config)
    t_td = 1000.0
    # Dense sampling across the +/-30 s window (cadence 5); the only large gap
    # ends at t_td-60, well before the window lower bound (t_td-30).
    dense = np.arange(t_td - 60.0, t_td + 30.0 + 1e-6, 5.0)
    times = np.concatenate([np.array([t_td - 300.0, t_td - 200.0]), dense])
    res = q.quantify(
        _fused(t_td, 1.0),
        _flight(times),
        groundspeed_at_td_mps=60.0,
        along_runway_distance_m=500.0,
    )
    assert res.diagnostics["gap_widening_factor"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Post-transition starvation widening (Req 9.6)
# ---------------------------------------------------------------------------


def test_post_transition_starvation_widens_and_flags():
    config = _uncertainty_config(
        starvation_widening_factor=2.0,
        post_transition_window_s=15.0,
        min_post_transition_samples=2,
    )
    q = UncertaintyQuantifier(config)
    t_td = 1000.0
    transition = 1002.0
    # Only ONE valid position sample after the transition within 15 s.
    times = np.array([t_td - 5.0, t_td, transition + 3.0])
    res = q.quantify(
        _fused(t_td, 1.0),
        _flight(times, transition=transition),
        groundspeed_at_td_mps=60.0,
        along_runway_distance_m=500.0,
    )
    assert res.diagnostics["post_transition_starved"] is True
    assert res.diagnostics["starvation_widening_factor"] == pytest.approx(2.0)
    assert res.confidence == CONFIDENCE_LOW
    assert res.reason_code == FailureReason.NO_GROUND_ROLL_CONFIRMATION.value


def test_sufficient_post_transition_samples_not_starved():
    config = _uncertainty_config(min_post_transition_samples=2, post_transition_window_s=15.0)
    q = UncertaintyQuantifier(config)
    t_td = 1000.0
    transition = 1002.0
    times = np.array([t_td, transition + 3.0, transition + 7.0, transition + 11.0])
    res = q.quantify(
        _fused(t_td, 1.0),
        _flight(times, transition=transition),
        groundspeed_at_td_mps=60.0,
        along_runway_distance_m=500.0,
    )
    assert res.diagnostics["post_transition_starved"] is False
    assert res.confidence == CONFIDENCE_NORMAL


# ---------------------------------------------------------------------------
# Missing-lever-arm widening (Req 7.5) -- distance only
# ---------------------------------------------------------------------------


def test_missing_lever_arm_widens_distance_only_and_flags():
    config = _uncertainty_config(missing_lever_arm_widening_factor=2.0)
    q = UncertaintyQuantifier(config)
    t_td = 1000.0
    times = _uniform_times(t_td, 5.0, 5)

    normal = q.quantify(
        _fused(t_td, 1.0),
        _flight(times),
        groundspeed_at_td_mps=60.0,
        along_runway_distance_m=500.0,
        lever_arm_missing=False,
    )
    missing = q.quantify(
        _fused(t_td, 1.0),
        _flight(times),
        groundspeed_at_td_mps=60.0,
        along_runway_distance_m=500.0,
        lever_arm_missing=True,
    )

    # Distance interval doubles; time interval unchanged.
    normal_dist_w = normal.distance_ci_90_upper_ft - normal.distance_ci_90_lower_ft
    missing_dist_w = missing.distance_ci_90_upper_ft - missing.distance_ci_90_lower_ft
    normal_time_w = normal.time_ci_90_upper_s - normal.time_ci_90_lower_s
    missing_time_w = missing.time_ci_90_upper_s - missing.time_ci_90_lower_s

    assert missing_dist_w == pytest.approx(2.0 * normal_dist_w, rel=1e-9)
    assert missing_time_w == pytest.approx(normal_time_w, rel=1e-9)
    assert missing.confidence == CONFIDENCE_LOW
    assert missing.reason_code == FailureReason.MISSING_LEVER_ARM.value


# ---------------------------------------------------------------------------
# Req 4.5: flag low-confidence rather than suppress; no-estimate -> None
# ---------------------------------------------------------------------------


def test_no_estimate_input_returns_none():
    q = UncertaintyQuantifier(_uncertainty_config())
    res = q.quantify(
        _fused(float("nan"), float("inf"), confidence=CONFIDENCE_NO_ESTIMATE,
               reason_code=FailureReason.ALL_ESTIMATORS_FAILED.value),
        _flight(np.array([1000.0])),
        groundspeed_at_td_mps=60.0,
        along_runway_distance_m=500.0,
    )
    assert res is None


def test_degenerate_groundspeed_flags_low_confidence_but_still_outputs_interval():
    """Req 4.5: a distance interval that cannot be reliably propagated is still
    emitted (with positive width) and flagged low-confidence, not suppressed."""
    q = UncertaintyQuantifier(_uncertainty_config())
    t_td = 1000.0
    res = q.quantify(
        _fused(t_td, 1.0),
        _flight(_uniform_times(t_td, 5.0, 5)),
        groundspeed_at_td_mps=0.0,  # cannot propagate timing -> distance
        along_runway_distance_m=500.0,
    )
    assert res is not None
    assert res.confidence == CONFIDENCE_LOW
    assert res.distance_ci_90_upper_ft - res.distance_ci_90_lower_ft > 0.0
    assert res.diagnostics["reliable_interval"] is False


def test_low_confidence_fused_input_preserved():
    q = UncertaintyQuantifier(_uncertainty_config())
    t_td = 1000.0
    res = q.quantify(
        _fused(t_td, 1.0, confidence=CONFIDENCE_LOW,
               reason_code=FailureReason.WIDE_CONFIDENCE_INTERVAL.value),
        _flight(_uniform_times(t_td, 5.0, 5)),
        groundspeed_at_td_mps=60.0,
        along_runway_distance_m=500.0,
    )
    assert res.confidence == CONFIDENCE_LOW
    assert res.reason_code == FailureReason.WIDE_CONFIDENCE_INTERVAL.value


def test_distance_converted_to_feet():
    q = UncertaintyQuantifier(_uncertainty_config())
    t_td = 1000.0
    res = q.quantify(
        _fused(t_td, 1.0),
        _flight(_uniform_times(t_td, 5.0, 5)),
        groundspeed_at_td_mps=60.0,
        along_runway_distance_m=500.0,
    )
    assert res.along_runway_distance_ft == pytest.approx(500.0 * M_TO_FT)


# ---------------------------------------------------------------------------
# Property 6: Confidence Interval Validity
# ---------------------------------------------------------------------------


@st.composite
def _flight_and_estimate(draw):
    t_td = draw(st.floats(min_value=100.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
    sigma_t = draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))
    gs_kt = draw(st.floats(min_value=50.0, max_value=220.0, allow_nan=False, allow_infinity=False))
    distance_m = draw(st.floats(min_value=0.0, max_value=3000.0, allow_nan=False, allow_infinity=False))
    confidence = draw(st.sampled_from([CONFIDENCE_NORMAL, CONFIDENCE_LOW]))
    lever_arm_missing = draw(st.booleans())
    include_transition = draw(st.booleans())
    n_side = draw(st.integers(min_value=1, max_value=6))
    cadence = draw(st.floats(min_value=3.0, max_value=6.0, allow_nan=False, allow_infinity=False))
    times = _uniform_times(t_td, cadence, n_side)
    transition = (t_td + draw(st.floats(min_value=0.5, max_value=10.0))) if include_transition else None
    return t_td, sigma_t, gs_kt, distance_m, confidence, lever_arm_missing, times, transition


@pytest.mark.property
@given(data=_flight_and_estimate())
def test_p6_confidence_interval_validity(data):
    """Feature: touchdown-point-detection, Property 6: Confidence Interval Validity

    For any flight producing a "normal" or "low-confidence" estimate, the output
    contains a 90% confidence interval for both time and distance where the lower
    bound is strictly below the point estimate, the point estimate is strictly
    below the upper bound, and the interval width is positive.

    Validates: Requirements 4.1, 4.2
    """
    t_td, sigma_t, gs_kt, distance_m, confidence, lever_missing, times, transition = data

    q = UncertaintyQuantifier(_uncertainty_config())
    res = q.quantify(
        _fused(t_td, sigma_t, confidence=confidence),
        _flight(times, transition=transition),
        groundspeed_at_td_mps=gs_kt * KNOTS_TO_MPS,
        along_runway_distance_m=distance_m,
        lever_arm_missing=lever_missing,
    )

    assert res is not None
    assert res.confidence in (CONFIDENCE_NORMAL, CONFIDENCE_LOW)

    # Time interval: lower < point < upper, positive width.
    assert res.time_ci_90_lower_s < res.t_td < res.time_ci_90_upper_s
    assert res.time_ci_90_upper_s - res.time_ci_90_lower_s > 0.0

    # Distance interval: lower < point < upper, positive width.
    assert res.distance_ci_90_lower_ft < res.along_runway_distance_ft < res.distance_ci_90_upper_ft
    assert res.distance_ci_90_upper_ft - res.distance_ci_90_lower_ft > 0.0


# ---------------------------------------------------------------------------
# Property 7: Gap-Proportional Uncertainty Widening
# ---------------------------------------------------------------------------


@st.composite
def _gap_scenario(draw):
    cadence = draw(st.floats(min_value=3.0, max_value=6.0, allow_nan=False, allow_infinity=False))
    gap = draw(st.floats(min_value=10.0, max_value=40.0, allow_nan=False, allow_infinity=False))
    sigma_t = draw(st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False))
    n_side = draw(st.integers(min_value=2, max_value=5))
    # gap_min sits strictly between cadence (<=6) and gap (>=10).
    assume(gap > cadence)
    return cadence, gap, sigma_t, n_side


@pytest.mark.property
@given(scenario=_gap_scenario())
def test_p7_gap_proportional_widening(scenario):
    """Feature: touchdown-point-detection, Property 7: Gap-Proportional Uncertainty Widening

    For any trajectory containing a data gap of duration G within +/-30 s of the
    estimated touchdown time, where nominal cadence is C, the reported confidence
    interval width is at least (G/C) times the width reported for an identical
    trajectory without the gap.

    Validates: Requirements 9.2
    """
    cadence, gap, sigma_t, n_side = scenario
    gap_min = 8.0  # strictly between cadence (<=6) and gap (>=10)
    config = _uncertainty_config(
        nominal_cadence_s=cadence,
        gap_min_duration_s=gap_min,
        gap_window_half_width_s=30.0,
    )
    q = UncertaintyQuantifier(config)
    t_td = 1000.0

    # Baseline: uniformly-sampled at cadence C (no gap exceeding gap_min).
    base_times = _uniform_times(t_td, cadence, n_side)
    base = q.quantify(
        _fused(t_td, sigma_t),
        _flight(base_times),
        groundspeed_at_td_mps=60.0,
        along_runway_distance_m=500.0,
    )

    # Gapped: a single gap of duration G straddling t_td, cadence C elsewhere.
    left = t_td - gap / 2.0 - np.arange(0, n_side) * cadence
    right = t_td + gap / 2.0 + np.arange(0, n_side) * cadence
    gapped_times = np.sort(np.concatenate([left, right]))
    gapped = q.quantify(
        _fused(t_td, sigma_t),
        _flight(gapped_times),
        groundspeed_at_td_mps=60.0,
        along_runway_distance_m=500.0,
    )

    base_w = base.time_ci_90_upper_s - base.time_ci_90_lower_s
    gap_w = gapped.time_ci_90_upper_s - gapped.time_ci_90_lower_s

    ratio = gap / cadence
    # Widened width is at least (G/C) times the un-gapped width.
    assert gap_w >= ratio * base_w * (1.0 - 1e-9)
