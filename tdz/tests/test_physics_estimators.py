"""Tests for the physics estimators (Task 12).

Covers the three physics estimators and the shared on-ground-flag upper bound:

* **Property 5 -- On-Ground Flag Upper Bound** (Req 18.1-18.4): a Hypothesis
  property over randomized landings asserting every estimator's (non-failed)
  ``t_td`` is strictly less than the on-ground transition time, plus direct
  unit tests of :func:`apply_on_ground_bound`.
* **Deceleration-knee** (Task 12.1, Req 5.1/6.1): known-breakpoint recovery on a
  two-slope speed profile, the FR24/velocity-only path, the implausible-fit
  low-confidence path (ESTIMATOR_DISAGREEMENT), and the no-groundspeed failure.
* **Vertical flare-crossing** (Task 12.2/12.5, Req 17.1-17.5): the flare-
  starvation edge case (one sub-50-ft sample but >=3 in the extended region
  still fits), the FR24/no-geometric failure (GEOMETRIC_ALT_UNAVAILABLE), the
  under-determined failure (INSUFFICIENT_FLARE_SAMPLES), known-crossing recovery,
  and the main-gear vertical-offset direction.
* **IMM filter + RTS smoother** (Task 12.3): the mode-probability crossover
  recovers the touchdown to sub-sample resolution with positive sigma and the
  documented diagnostics, and degrades to failed on too-few samples.

Tolerances are kept honest to the 4-5 s ADS-B cadence: a touchdown cannot be
located more finely than roughly a sample interval, so example tolerances are
stated as multiples of the cadence (commonly ~1.5x cadence) and documented at
each assertion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pyproj import Geod

from tdz.estimators.physics import (
    DecelKneeEstimator,
    FlareCrossingEstimator,
    ImmRtsEstimator,
    OnGroundBoundResult,
    apply_on_ground_bound,
)
from tdz.estimators.physics.base import ON_GROUND_BOUND_GUARD_S
from tdz.models import FailureReason, FlightRecord, RunwayReference
from tdz.timebase.interpolation import KNOTS_TO_MPS

_GEOD = Geod(ellps="WGS84")
FT_TO_M = 0.3048


# ---------------------------------------------------------------------------
# Synthetic landing helpers
# ---------------------------------------------------------------------------


def _runway(*, elevation_m: float = 30.0, datum: str = "HAE") -> RunwayReference:
    """A simple HAE runway (geoid undulation 0 so geo altitude is comparable)."""
    return RunwayReference(
        threshold_lat=33.94,
        threshold_lon=-118.40,
        heading_deg=250.0,
        elevation_m=elevation_m,
        elevation_datum=datum,
        geoid_undulation_m=0.0,
        length_m=3500.0,
        width_m=45.0,
        displaced=False,
    )


def _approach_latlon(runway: RunwayReference, speed_mps: float, dt_to_td: float):
    """Lat/lon of a point ``dt_to_td`` seconds (along-track) before threshold.

    Walks back along the reciprocal landing heading by ``speed * dt_to_td``;
    after touchdown (``dt_to_td < 0``) walks forward down the runway. Used only
    so the FlightRecord carries plausible positions -- the physics estimators
    under test consume altitude/speed, not lat/lon.
    """
    d = speed_mps * dt_to_td
    if d >= 0.0:
        az = (runway.heading_deg + 180.0) % 360.0
        lon, lat, _ = _GEOD.fwd(runway.threshold_lon, runway.threshold_lat, az, d)
    else:
        lon, lat, _ = _GEOD.fwd(
            runway.threshold_lon, runway.threshold_lat, runway.heading_deg, -d
        )
    return lat, lon


def synthetic_landing(
    *,
    dt: float = 4.5,
    t_td: float = 200.0,
    n_before: int = 12,
    n_after: int = 8,
    v_td_mps: float = 65.0,
    approach_decel: float = 0.5,
    rollout_decel: float = 2.5,
    glide_rate_mps: float = 3.5,
    flare_duration_s: float = 6.0,
    phase_s: float = 0.0,
    velocity_offset_s: float = 0.0,
    omit_geometric: bool = False,
    on_ground_delay_samples: float = 2.0,
    flight_id: str = "SYN",
    ads_b_source: str = "aireon",
) -> FlightRecord:
    """Build a synthetic completed-landing :class:`FlightRecord`.

    A constant ~3 deg glideslope, a quadratic flare flattening to the runway at
    ``t_td``, and a constant-deceleration ground roll, sampled at the 4-5 s ADS-B
    cadence. The groundspeed is a clean two-slope profile with the knee at
    ``t_td`` (gentle ``approach_decel`` -> steep ``rollout_decel``); the height
    above the runway follows the glideslope down to the flare onset (~50 ft),
    then a quadratic flattening to 0 at ``t_td``, then ~0 on the ground.

    Knobs
    -----
    omit_geometric:
        Replace geometric altitude with all-NaN (FR24-like velocity-only source).
    velocity_offset_s:
        Shift the velocity timebase relative to the position timebase (async
        Aireon); ``0`` co-times them (FR24-like).
    phase_s:
        Shift all sample times so touchdown falls between samples differently.
    on_ground_delay_samples:
        The on-ground flag transitions this many samples AFTER ``t_td``.
    """
    runway = _runway()
    position_times = phase_s + t_td + np.arange(-n_before, n_after + 1) * dt
    velocity_times = position_times + velocity_offset_s

    h_flare = 50.0 * FT_TO_M
    t_flare = t_td - flare_duration_s

    def height(t: float) -> float:
        if t <= t_flare:
            return h_flare + glide_rate_mps * (t_flare - t)
        if t <= t_td:
            return h_flare * ((t_td - t) / (t_td - t_flare)) ** 2
        return 0.0

    def speed_mps(t: float) -> float:
        if t <= t_td:
            return v_td_mps + approach_decel * (t_td - t)
        return v_td_mps - rollout_decel * (t - t_td)

    heights = np.array([height(float(t)) for t in position_times])
    geo_alt = runway.elevation_m + heights
    if omit_geometric:
        geo_alt = np.full(position_times.size, np.nan)

    v_mps = np.array([max(speed_mps(float(t)), 1.0) for t in velocity_times])
    gs_kt = v_mps / KNOTS_TO_MPS

    lats, lons = [], []
    for t in position_times:
        lat, lon = _approach_latlon(runway, max(speed_mps(float(t)), 1.0), t_td - float(t))
        lats.append(lat)
        lons.append(lon)

    transition = t_td + on_ground_delay_samples * dt
    on_ground_flags = position_times >= transition

    return FlightRecord(
        flight_id=flight_id,
        aircraft_type="B738",
        ads_b_source=ads_b_source,
        position_times=position_times,
        velocity_times=velocity_times,
        latitudes=np.array(lats),
        longitudes=np.array(lons),
        geometric_altitudes=geo_alt,
        barometric_altitudes=np.full(position_times.size, np.nan),
        groundspeeds=gs_kt,
        tracks=np.full(velocity_times.size, runway.heading_deg),
        baro_vertical_rates=np.full(velocity_times.size, np.nan),
        on_ground_flags=on_ground_flags,
        on_ground_transition_time=float(transition),
        runway=runway,
    )


def _flight_from_heights(
    *,
    position_times: np.ndarray,
    heights_above_runway_m: np.ndarray,
    runway: RunwayReference,
    on_ground_transition_time: float | None = None,
    groundspeeds_kt: np.ndarray | None = None,
    velocity_times: np.ndarray | None = None,
) -> FlightRecord:
    """Minimal FlightRecord from explicit height-above-runway samples.

    Used by the flare-crossing tests, which consume only ``position_times``,
    ``geometric_altitudes`` and the runway. Lat/lon are set on the threshold and
    a benign decelerating groundspeed is supplied for completeness.
    """
    position_times = np.asarray(position_times, dtype=float)
    geo_alt = runway.elevation_m + np.asarray(heights_above_runway_m, dtype=float)
    if velocity_times is None:
        velocity_times = position_times
    if groundspeeds_kt is None:
        groundspeeds_kt = np.linspace(140.0, 120.0, velocity_times.size)
    return FlightRecord(
        flight_id="FLR",
        aircraft_type="B738",
        ads_b_source="aireon",
        position_times=position_times,
        velocity_times=np.asarray(velocity_times, dtype=float),
        latitudes=np.full(position_times.size, runway.threshold_lat),
        longitudes=np.full(position_times.size, runway.threshold_lon),
        geometric_altitudes=geo_alt,
        barometric_altitudes=np.full(position_times.size, np.nan),
        groundspeeds=np.asarray(groundspeeds_kt, dtype=float),
        tracks=np.full(velocity_times.size, runway.heading_deg),
        baro_vertical_rates=np.full(velocity_times.size, np.nan),
        on_ground_flags=np.zeros(position_times.size, dtype=bool),
        on_ground_transition_time=on_ground_transition_time,
        runway=runway,
    )


# ===========================================================================
# Property 5: On-Ground Flag Upper Bound
# ===========================================================================


@pytest.mark.property
@given(
    cadence=st.floats(min_value=4.0, max_value=5.0),
    n_before=st.integers(min_value=8, max_value=16),
    n_after=st.integers(min_value=5, max_value=12),
    v_td_mps=st.floats(min_value=55.0, max_value=80.0),
    approach_decel=st.floats(min_value=0.3, max_value=0.9),
    rollout_decel=st.floats(min_value=1.6, max_value=3.5),
    glide_rate=st.floats(min_value=2.5, max_value=4.5),
    velocity_offset=st.floats(min_value=0.0, max_value=2.5),
    on_ground_delay=st.floats(min_value=1.0, max_value=3.0),
    omit_geometric=st.booleans(),
)
def test_property_on_ground_flag_upper_bound(
    cadence,
    n_before,
    n_after,
    v_td_mps,
    approach_decel,
    rollout_decel,
    glide_rate,
    velocity_offset,
    on_ground_delay,
    omit_geometric,
):
    """Feature: touchdown-point-detection, Property 5: On-Ground Flag Upper Bound

    For randomized landings with a known on-ground transition time, every
    estimator's reported ``t_td`` (when it is not a failed estimate) is strictly
    less than the transition time and never equal to it (Req 18.1, 18.2).
    """
    flight = synthetic_landing(
        dt=cadence,
        n_before=n_before,
        n_after=n_after,
        v_td_mps=v_td_mps,
        approach_decel=approach_decel,
        rollout_decel=rollout_decel,
        glide_rate_mps=glide_rate,
        velocity_offset_s=velocity_offset,
        on_ground_delay_samples=on_ground_delay,
        omit_geometric=omit_geometric,
    )
    transition = flight.on_ground_transition_time
    assert transition is not None

    estimators = [DecelKneeEstimator(), FlareCrossingEstimator(), ImmRtsEstimator()]
    for estimator in estimators:
        estimate = estimator.estimate(flight)
        if estimate.confidence == "failed":
            continue
        assert math.isfinite(estimate.t_td)
        # Req 18.2: t_td <= transition; Req 18.1: never equal to the transition.
        assert estimate.t_td < transition, (
            f"{estimator.name()} t_td={estimate.t_td} !< transition={transition}"
        )


@pytest.mark.unit
def test_apply_on_ground_bound_before_bound_unchanged():
    """A candidate strictly before the transition is returned unchanged (Req 18.4)."""
    result = apply_on_ground_bound(100.0, 130.0)
    assert isinstance(result, OnGroundBoundResult)
    assert result.clamped is False
    assert result.t_td == 100.0
    assert result.bound == 130.0


@pytest.mark.unit
def test_apply_on_ground_bound_at_or_after_clamped_strictly_below():
    """A candidate at/after the transition is clamped strictly below it (Req 18.1-18.3)."""
    at = apply_on_ground_bound(130.0, 130.0)
    assert at.clamped is True
    assert at.t_td < 130.0
    assert at.t_td == pytest.approx(130.0 - ON_GROUND_BOUND_GUARD_S)

    after = apply_on_ground_bound(145.0, 130.0)
    assert after.clamped is True
    assert after.t_td < 130.0
    assert after.t_td == pytest.approx(130.0 - ON_GROUND_BOUND_GUARD_S)
    assert after.pre_clamp_t_td == 145.0


@pytest.mark.unit
def test_apply_on_ground_bound_no_transition_unchanged():
    """With no transition time the flag has zero weight (Req 18.4)."""
    result = apply_on_ground_bound(100.0, None)
    assert result.clamped is False
    assert result.bound is None
    assert result.t_td == 100.0


@pytest.mark.unit
def test_on_ground_bound_clamps_estimator_output_and_widens_sigma():
    """An estimator whose raw t_td is after the transition is clamped by the base.

    Here the on-ground flag is forced to transition BEFORE the true knee, so the
    decel-knee candidate lands after the bound and must be clamped strictly below
    it with sigma widened (Req 18.3; base scaffolding).
    """
    flight = synthetic_landing(on_ground_delay_samples=-3.0)
    transition = flight.on_ground_transition_time
    estimate = DecelKneeEstimator().estimate(flight)
    assert estimate.t_td < transition
    assert estimate.diagnostics["on_ground_clamped"] is True
    assert estimate.t_td == pytest.approx(transition - ON_GROUND_BOUND_GUARD_S)
    assert "pre_clamp_t_td" in estimate.diagnostics


# ===========================================================================
# Deceleration-knee estimator (Task 12.1)
# ===========================================================================


@pytest.mark.unit
def test_decel_knee_recovers_known_touchdown():
    """The breakpoint of a clean two-slope speed profile recovers the knee.

    Tolerance ~1.5x cadence: the breakpoint cannot be located more finely than
    about a sample interval at 4-5 s cadence.
    """
    dt = 4.5
    flight = synthetic_landing(dt=dt, t_td=200.0, on_ground_delay_samples=4.0)
    estimate = DecelKneeEstimator().estimate(flight)
    assert estimate.confidence == "normal"
    assert abs(estimate.t_td - 200.0) <= 1.5 * dt
    assert estimate.sigma_t > 0.0
    assert "breakpoint_time" in estimate.diagnostics


@pytest.mark.unit
def test_decel_knee_runs_on_fr24_velocity_only():
    """Decel-knee runs without geometric altitude (velocity-stream only)."""
    dt = 4.5
    flight = synthetic_landing(
        dt=dt, omit_geometric=True, velocity_offset_s=0.0, ads_b_source="flightradar24"
    )
    estimate = DecelKneeEstimator().estimate(flight)
    assert estimate.confidence in ("normal", "low-confidence")
    assert math.isfinite(estimate.t_td)
    assert abs(estimate.t_td - 200.0) <= 1.5 * dt


@pytest.mark.unit
def test_decel_knee_implausible_fit_low_confidence_disagreement():
    """An implausible fitted regime flags low-confidence with ESTIMATOR_DISAGREEMENT.

    A tiny aircraft-class deceleration envelope makes the (physically normal)
    fitted rollout deceleration fall outside the prior, so the estimate is kept
    but down-weighted (Req 6.1).
    """
    from tdz.estimators.physics.decel_knee import DecelPrior

    flight = synthetic_landing()
    # Implausibly narrow envelope the real fit cannot satisfy.
    tiny = {"narrowbody": DecelPrior(31.0, 32.0, 0.10, 0.20)}
    estimate = DecelKneeEstimator(aircraft_class="narrowbody", priors=tiny).estimate(flight)
    assert estimate.confidence == "low-confidence"
    assert estimate.diagnostics["reason_code"] == FailureReason.ESTIMATOR_DISAGREEMENT.value
    assert estimate.diagnostics["prior_influence"]["within_prior"] is False


@pytest.mark.unit
def test_decel_knee_no_groundspeed_failed():
    """No groundspeed at all -> failed with NO_GROUNDSPEED."""
    flight = synthetic_landing()
    flight.groundspeeds = np.full(flight.velocity_times.size, np.nan)
    estimate = DecelKneeEstimator().estimate(flight)
    assert estimate.confidence == "failed"
    assert estimate.diagnostics["reason_code"] == FailureReason.NO_GROUNDSPEED.value


# ===========================================================================
# Vertical flare-crossing estimator (Task 12.2 / 12.5)
# ===========================================================================


@pytest.mark.unit
def test_flare_starvation_one_sample_below_50ft_still_fits():
    """Task 12.5: one sub-50-ft sample but >=3 in the extended region still fits.

    A realistic descent at 4-5 s cadence places only a single geometric sample
    below 50 ft, yet >=3 fall in the extended ~0-250 ft region, so the joint
    glideslope+flare fit produces an estimate rather than failing
    INSUFFICIENT_FLARE_SAMPLES (Req 17.1, 17.5).
    """
    runway = _runway()
    # Heights (m above runway): four samples above 50 ft (15.24 m) within 250 ft
    # (76.2 m), and exactly one below 50 ft. The geometric stream ends near
    # touchdown (no flat ground tail), as is realistic for the vertical fit.
    fifty_ft = 50.0 * FT_TO_M
    heights = np.array([72.0, 56.0, 40.0, 24.0, 8.0])  # only 8.0 m < 15.24 m
    position_times = np.arange(heights.size) * 4.5

    n_below_50 = int(np.count_nonzero(heights < fifty_ft))
    n_in_region = int(np.count_nonzero((heights >= 0.0) & (heights <= 250.0 * FT_TO_M)))
    assert n_below_50 == 1
    assert n_in_region >= 3

    flight = _flight_from_heights(
        position_times=position_times,
        heights_above_runway_m=heights,
        runway=runway,
    )
    estimate = FlareCrossingEstimator().estimate(flight)
    assert estimate.confidence == "normal"
    assert estimate.diagnostics["reason_code"] != FailureReason.INSUFFICIENT_FLARE_SAMPLES.value
    assert estimate.diagnostics["n_samples_in_fit_region"] >= 3
    assert math.isfinite(estimate.t_td)


@pytest.mark.unit
def test_flare_fr24_no_geometric_failed_geometric_unavailable():
    """Task 12.5: a source without geometric altitude self-disables the estimator."""
    flight = synthetic_landing(omit_geometric=True, ads_b_source="flightradar24")
    estimate = FlareCrossingEstimator().estimate(flight)
    assert estimate.confidence == "failed"
    assert (
        estimate.diagnostics["reason_code"]
        == FailureReason.GEOMETRIC_ALT_UNAVAILABLE.value
    )


@pytest.mark.unit
def test_flare_too_few_extended_region_samples_failed():
    """Fewer than 3 samples in the extended region -> INSUFFICIENT_FLARE_SAMPLES."""
    runway = _runway()
    # Only two samples in [0, 250 ft]; the rest are well above the region.
    heights = np.array([400.0, 300.0, 70.0, 30.0])  # only 70 and 30 in region
    position_times = np.arange(heights.size) * 4.5
    flight = _flight_from_heights(
        position_times=position_times,
        heights_above_runway_m=heights,
        runway=runway,
    )
    estimate = FlareCrossingEstimator().estimate(flight)
    assert estimate.confidence == "failed"
    assert (
        estimate.diagnostics["reason_code"]
        == FailureReason.INSUFFICIENT_FLARE_SAMPLES.value
    )
    assert estimate.diagnostics["n_samples_in_fit_region"] < 3


@pytest.mark.unit
def test_flare_recovers_known_crossing():
    """A clean quadratic flare crossing 0 at a known time is recovered.

    Heights follow ``h(t) = a (t_cross - t)^2 + b (t_cross - t)`` with ``b>0`` so
    the model is descending (nonzero rate) at the crossing. The OLS quadratic fit
    recovers the antenna (V=0) crossing within ~1 sample at 4.5 s cadence.
    """
    runway = _runway()
    t_cross = 24.0
    a, b = 0.05, 2.0  # curvature + descent term (m over seconds-to-crossing)
    position_times = np.arange(1.5, t_cross, 4.5)
    tau = t_cross - position_times
    heights = a * tau**2 + b * tau  # all positive, inside the extended region
    assert np.all(heights <= 250.0 * FT_TO_M)  # every sample within ~0-250 ft
    assert position_times.size >= 3

    flight = _flight_from_heights(
        position_times=position_times,
        heights_above_runway_m=heights,
        runway=runway,
    )
    estimate = FlareCrossingEstimator().estimate(flight)
    assert estimate.confidence == "normal"
    assert abs(estimate.t_td - t_cross) <= 4.5  # within ~one sample interval
    assert estimate.sigma_t > 0.0


@pytest.mark.unit
def test_flare_main_gear_offset_shifts_crossing_earlier():
    """Task 12.5(e): a positive vertical offset V moves the crossing earlier.

    The antenna sits V above the main gear, so the gear contacts the runway
    before the antenna reaches it: solving ``h = V`` (V>0) on a descending
    profile yields an earlier time than ``h = 0`` (Req 17.4).
    """
    runway = _runway()
    t_cross = 24.0
    a, b = 0.05, 2.0
    position_times = np.arange(1.5, t_cross, 4.5)
    tau = t_cross - position_times
    heights = a * tau**2 + b * tau

    flight = _flight_from_heights(
        position_times=position_times,
        heights_above_runway_m=heights,
        runway=runway,
    )
    t_antenna = FlareCrossingEstimator(vertical_offset_m=0.0).estimate(flight).t_td
    t_gear = FlareCrossingEstimator(vertical_offset_m=2.5).estimate(flight).t_td
    assert math.isfinite(t_antenna) and math.isfinite(t_gear)
    # Gear contact (V > 0) is earlier than antenna-at-runway (V = 0).
    assert t_gear < t_antenna


# ===========================================================================
# IMM filter + RTS smoother (Task 12.3)
# ===========================================================================


@pytest.mark.unit
def test_imm_crossover_recovers_touchdown_sub_sample():
    """The mode-probability crossover recovers the touchdown to sub-sample resolution.

    On a clean async landing the smoothed ``P(mode 2)=0.5`` crossing is within
    ~1.5x cadence of the true touchdown, is strictly between sample times (not
    equal to any sample), reports a positive sigma, and carries the documented
    diagnostics (mode probabilities + crossover).
    """
    dt = 4.5
    t_td = 200.0
    flight = synthetic_landing(dt=dt, t_td=t_td, velocity_offset_s=1.7)
    estimate = ImmRtsEstimator().estimate(flight)

    assert estimate.confidence == "normal"
    assert abs(estimate.t_td - t_td) <= 1.5 * dt
    assert estimate.sigma_t > 0.0

    # Sub-sample: not exactly equal to any velocity/position sample time.
    sample_times = np.concatenate([flight.velocity_times, flight.position_times])
    assert np.min(np.abs(sample_times - estimate.t_td)) > 1e-6

    diag = estimate.diagnostics
    assert "crossover_time" in diag
    assert "crossover_sharpness_per_s" in diag
    assert "mode_probabilities_smoothed" in diag
    assert "mode_probability_times" in diag
    assert len(diag["mode_probabilities_smoothed"]) == len(diag["mode_probability_times"])
    # Crossover brackets 0.5.
    p_before, p_after = diag["p_ground_at_crossover"]
    assert p_before < 0.5 <= p_after


@pytest.mark.unit
def test_imm_runs_velocity_only_without_vertical_channel():
    """The IMM still runs on a velocity-only source (vertical channel dropped)."""
    dt = 4.5
    flight = synthetic_landing(dt=dt, omit_geometric=True, ads_b_source="flightradar24")
    estimate = ImmRtsEstimator().estimate(flight)
    assert estimate.confidence == "normal"
    assert estimate.diagnostics["vertical_channel_used"] is False
    assert abs(estimate.t_td - 200.0) <= 1.5 * dt


@pytest.mark.unit
def test_imm_degrades_to_failed_on_too_few_samples():
    """Too few usable samples -> failed with INSUFFICIENT_SAMPLES."""
    flight = synthetic_landing(n_before=1, n_after=1)
    estimate = ImmRtsEstimator().estimate(flight)
    assert estimate.confidence == "failed"
    assert estimate.diagnostics["reason_code"] == FailureReason.INSUFFICIENT_SAMPLES.value


@pytest.mark.unit
def test_imm_no_groundspeed_failed():
    """No groundspeed at all -> failed with NO_GROUNDSPEED."""
    flight = synthetic_landing()
    flight.groundspeeds = np.full(flight.velocity_times.size, np.nan)
    estimate = ImmRtsEstimator().estimate(flight)
    assert estimate.confidence == "failed"
    assert estimate.diagnostics["reason_code"] == FailureReason.NO_GROUNDSPEED.value
