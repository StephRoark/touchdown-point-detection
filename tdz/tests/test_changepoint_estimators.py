"""Tests for the change-point estimators (Task 13).

Covers the four velocity-stream corroborators of the approach -> ground-roll
deceleration-regime transition (Req 5.2) and the jerk-onset corroborating role
(Req 16.3):

* **PELT / CUSUM / GLRT** -- known-answer recovery of a two-regime deceleration
  knee within ~1.5x the ADS-B cadence, positive ``sigma_t``, and ``t_td`` at or
  before the on-ground transition.
* **PELT property test** -- randomized change time / slopes / noise / cadence:
  the change point is recovered within a cadence-and-noise-scaled tolerance
  (documented at the assertion).
* **Jerk-onset** -- on a profile whose peak jerk *lags* the onset, the returned
  time is nearer the onset than the peak, and the estimate carries the
  corroborating-only role (low-confidence + ``corroborating_only`` flag).
* **Failure paths** -- no groundspeed -> NO_GROUNDSPEED; too few samples ->
  INSUFFICIENT_SAMPLES (parametrized across all four).
* **On-ground bound** -- a detector whose raw change point lands after a
  deliberately-early on-ground transition is clamped strictly below it
  (inherited from :class:`PhysicsEstimator`).

Tolerances are kept honest to the 4-5 s ADS-B cadence: a regime change cannot be
located more finely than roughly a sample interval, so example tolerances are
stated as multiples of the cadence and documented at each assertion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tdz.estimators.changepoint import (
    CusumEstimator,
    GlrtEstimator,
    JerkOnsetEstimator,
    PeltEstimator,
)
from tdz.estimators.physics.base import ON_GROUND_BOUND_GUARD_S
from tdz.models import FailureReason, FlightRecord, RunwayReference
from tdz.timebase.interpolation import KNOTS_TO_MPS

ALL_ESTIMATORS = [PeltEstimator, CusumEstimator, GlrtEstimator, JerkOnsetEstimator]
REGIME_ESTIMATORS = [PeltEstimator, CusumEstimator, GlrtEstimator]


# ---------------------------------------------------------------------------
# Synthetic two-regime deceleration helpers (velocity-stream only)
# ---------------------------------------------------------------------------


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


def _flight_from_groundspeed(
    *,
    times: np.ndarray,
    groundspeeds_kt: np.ndarray,
    on_ground_transition_time: float | None,
) -> FlightRecord:
    """Minimal velocity-stream FlightRecord (geometric altitude not required).

    The change-point detectors consume only ``velocity_times`` and
    ``groundspeeds``; geometric altitude/position are filled with benign values.
    """
    times = np.asarray(times, dtype=float)
    gs = np.asarray(groundspeeds_kt, dtype=float)
    runway = _runway()
    if on_ground_transition_time is not None:
        on_ground_flags = times >= float(on_ground_transition_time)
    else:
        on_ground_flags = np.zeros(times.size, dtype=bool)
    return FlightRecord(
        flight_id="CPT",
        aircraft_type="B738",
        ads_b_source="aireon",
        position_times=times,
        velocity_times=times,
        latitudes=np.full(times.size, runway.threshold_lat),
        longitudes=np.full(times.size, runway.threshold_lon),
        geometric_altitudes=np.full(times.size, np.nan),
        barometric_altitudes=np.full(times.size, np.nan),
        groundspeeds=gs,
        tracks=np.full(times.size, runway.heading_deg),
        baro_vertical_rates=np.full(times.size, np.nan),
        on_ground_flags=on_ground_flags,
        on_ground_transition_time=on_ground_transition_time,
        runway=runway,
    )


def two_regime_landing(
    *,
    dt: float = 4.5,
    t_td: float = 200.0,
    n_before: int = 12,
    n_after: int = 8,
    v_td_mps: float = 65.0,
    approach_decel: float = 0.4,
    rollout_decel: float = 2.5,
    on_ground_delay_samples: float = 2.0,
    noise_mps: float = 0.0,
    seed: int = 0,
) -> tuple[FlightRecord, float]:
    """Two-slope groundspeed profile with the deceleration knee at ``t_td``.

    Gentle approach deceleration (``approach_decel``) steepening to ground-roll
    braking (``rollout_decel``) at the knee; sampled at the 4-5 s cadence with an
    on-ground transition ``on_ground_delay_samples`` samples after the knee.
    Returns ``(flight, transition_time)``.
    """
    times = t_td + np.arange(-n_before, n_after + 1) * dt

    def speed_mps(t: float) -> float:
        if t <= t_td:
            return v_td_mps + approach_decel * (t_td - t)
        return v_td_mps - rollout_decel * (t - t_td)

    v = np.array([max(speed_mps(float(t)), 1.0) for t in times])
    if noise_mps > 0.0:
        rng = np.random.default_rng(seed)
        v = np.maximum(v + rng.normal(0.0, noise_mps, size=v.size), 1.0)
    gs_kt = v / KNOTS_TO_MPS

    transition = t_td + on_ground_delay_samples * dt
    flight = _flight_from_groundspeed(
        times=times, groundspeeds_kt=gs_kt, on_ground_transition_time=float(transition)
    )
    return flight, float(transition)


def jerk_lag_landing(
    *,
    dt: float = 4.5,
    t_onset: float = 200.0,
    braking_duration_s: float = 30.0,
    n_before: int = 12,
    n_after: int = 10,
    v0_mps: float = 70.0,
    cubic_c: float = 0.0016,
    approach_decel: float = 0.3,
) -> tuple[FlightRecord, float]:
    """Profile whose peak jerk LAGS the onset.

    Before ``t_onset`` the speed decelerates gently (constant ``approach_decel``).
    After onset the speed follows a **cubic** ``v0 - c*(t - t_onset)^3``, so the
    deceleration grows quadratically and the jerk magnitude grows linearly,
    peaking near the end of the braking window -- well after the onset. Returns
    ``(flight, t_onset)``.
    """
    times = t_onset + np.arange(-n_before, n_after + 1) * dt
    v_onset = v0_mps

    def speed_mps(t: float) -> float:
        if t <= t_onset:
            return v_onset + approach_decel * (t_onset - t)
        tau = min(t - t_onset, braking_duration_s)
        return v_onset - cubic_c * tau**3

    v = np.array([max(speed_mps(float(t)), 1.0) for t in times])
    gs_kt = v / KNOTS_TO_MPS
    # On-ground transition well after onset so it does not clamp the estimate.
    transition = t_onset + (n_after - 1) * dt
    flight = _flight_from_groundspeed(
        times=times, groundspeeds_kt=gs_kt, on_ground_transition_time=float(transition)
    )
    return flight, float(t_onset)


# ===========================================================================
# Known-answer recovery (PELT / CUSUM / GLRT)
# ===========================================================================


@pytest.mark.unit
@pytest.mark.parametrize("estimator_cls", REGIME_ESTIMATORS)
def test_regime_estimator_recovers_known_knee(estimator_cls):
    """Each regime detector recovers the deceleration knee within ~1.5x cadence.

    A clean two-slope profile has its knee at ``t_td=200``; at 4.5 s cadence the
    change cannot be located more finely than about a sample interval, so the
    tolerance is ~1.5x cadence. ``sigma_t`` is positive and ``t_td`` lies at or
    before the on-ground transition.
    """
    dt = 4.5
    flight, transition = two_regime_landing(dt=dt, t_td=200.0, on_ground_delay_samples=3.0)
    estimate = estimator_cls().estimate(flight)
    assert estimate.confidence == "normal"
    assert abs(estimate.t_td - 200.0) <= 1.5 * dt, (
        f"{estimator_cls.__name__} t_td={estimate.t_td}"
    )
    assert estimate.sigma_t > 0.0
    assert estimate.t_td <= transition
    assert "change_index" in estimate.diagnostics


@pytest.mark.unit
def test_jerk_onset_recovers_known_onset():
    """Jerk-onset recovers the braking onset within ~1.5x cadence (corroborating).

    On the two-slope profile the smoothed-jerk transient sits at the knee; the
    onset (leading edge) is recovered within ~1.5x cadence, with positive sigma,
    at/before the on-ground transition, and flagged corroborating-only.
    """
    dt = 4.5
    flight, transition = two_regime_landing(dt=dt, t_td=200.0, on_ground_delay_samples=3.0)
    estimate = JerkOnsetEstimator().estimate(flight)
    assert estimate.confidence == "low-confidence"
    assert estimate.diagnostics["corroborating_only"] is True
    assert abs(estimate.t_td - 200.0) <= 1.5 * dt
    assert estimate.sigma_t > 0.0
    assert estimate.t_td <= transition


# ===========================================================================
# Property test: PELT change-point recovery
# ===========================================================================


@pytest.mark.property
@given(
    dt=st.floats(min_value=4.0, max_value=5.0),
    knee_offset=st.integers(min_value=-3, max_value=3),
    n_before=st.integers(min_value=8, max_value=16),
    n_after=st.integers(min_value=6, max_value=12),
    v_td_mps=st.floats(min_value=55.0, max_value=80.0),
    approach_decel=st.floats(min_value=0.2, max_value=0.8),
    rollout_decel=st.floats(min_value=1.8, max_value=3.5),
    noise_mps=st.floats(min_value=0.0, max_value=0.4),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_property_pelt_recovers_changepoint(
    dt,
    knee_offset,
    n_before,
    n_after,
    v_td_mps,
    approach_decel,
    rollout_decel,
    noise_mps,
    seed,
):
    """Feature: touchdown-point-detection, Req 5.2: PELT change-point recovery

    For randomized two-regime deceleration profiles (cadence, knee location,
    approach/rollout slopes, and speed noise all varied), PELT recovers the
    regime change within a cadence-and-noise-scaled tolerance. The tolerance is
    ``2.0*cadence`` (the change cannot be located more finely than ~a sample
    interval) plus ``noise/rollout_decel`` seconds -- speed scatter of
    ``noise`` m/s around a knee whose deceleration changes by ``rollout_decel``
    maps to that much extra time ambiguity.
    """
    # Place the true knee a few samples off-centre so recovery is not trivial.
    t_td = 200.0 + knee_offset * dt
    flight, transition = two_regime_landing(
        dt=dt,
        t_td=t_td,
        n_before=n_before,
        n_after=n_after,
        v_td_mps=v_td_mps,
        approach_decel=approach_decel,
        rollout_decel=rollout_decel,
        on_ground_delay_samples=float(n_after - 1),
        noise_mps=noise_mps,
        seed=seed,
    )
    estimate = PeltEstimator().estimate(flight)
    assert estimate.confidence == "normal"
    assert math.isfinite(estimate.t_td)
    assert estimate.sigma_t > 0.0

    tol = 2.0 * dt + noise_mps / rollout_decel
    assert abs(estimate.t_td - t_td) <= tol, (
        f"t_td={estimate.t_td} true={t_td} tol={tol}"
    )


# ===========================================================================
# Jerk-onset: onset-not-peak + corroborating role
# ===========================================================================


@pytest.mark.unit
def test_jerk_onset_returns_onset_not_peak():
    """On a peak-lags-onset profile the returned time is nearer the onset.

    The cubic braking profile drives the jerk magnitude to grow through the
    braking window, so the smoothed-jerk extremum (peak) lags the true onset.
    The detector must return a time nearer the ONSET than the peak (Req design
    Errors 9.5: peak braking lags touchdown).
    """
    dt = 4.5
    flight, t_onset = jerk_lag_landing(dt=dt, t_onset=200.0)
    estimate = JerkOnsetEstimator().estimate(flight)

    diag = estimate.diagnostics
    peak_time = diag["peak_jerk_time"]
    # The peak genuinely lags the onset for this profile.
    assert peak_time > estimate.t_td
    assert diag["onset_lead_s"] > 0.0
    # The returned time is nearer the onset than the (lagging) peak.
    assert abs(estimate.t_td - t_onset) < abs(estimate.t_td - peak_time)
    # And reasonably close to the true onset (within ~2x cadence).
    assert abs(estimate.t_td - t_onset) <= 2.0 * dt


@pytest.mark.unit
def test_jerk_onset_is_corroborating_only():
    """Jerk-onset never presents as a standalone primary (Req 16.3)."""
    flight, _ = two_regime_landing()
    estimate = JerkOnsetEstimator().estimate(flight)
    assert estimate.confidence == "low-confidence"
    assert estimate.diagnostics["corroborating_only"] is True


# ===========================================================================
# Failure paths
# ===========================================================================


@pytest.mark.unit
@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_no_groundspeed_failed(estimator_cls):
    """No groundspeed at all -> failed with NO_GROUNDSPEED (each detector)."""
    flight, _ = two_regime_landing()
    flight.groundspeeds = np.full(flight.velocity_times.size, np.nan)
    estimate = estimator_cls().estimate(flight)
    assert estimate.confidence == "failed"
    assert estimate.diagnostics["reason_code"] == FailureReason.NO_GROUNDSPEED.value


@pytest.mark.unit
@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_too_few_samples_failed(estimator_cls):
    """Fewer than the minimum samples -> failed with INSUFFICIENT_SAMPLES."""
    flight, _ = two_regime_landing(n_before=1, n_after=1)
    estimate = estimator_cls().estimate(flight)
    assert estimate.confidence == "failed"
    assert estimate.diagnostics["reason_code"] == FailureReason.INSUFFICIENT_SAMPLES.value


# ===========================================================================
# On-ground upper bound (inherited from PhysicsEstimator)
# ===========================================================================


@pytest.mark.unit
def test_on_ground_bound_clamps_changepoint_output():
    """A raw change point after a deliberately-early on-ground flag is clamped.

    The on-ground flag is forced to transition BEFORE the true knee, so the
    detected regime change lands after the bound and must be clamped strictly
    below it with sigma widened (Req 18.3; base scaffolding).
    """
    dt = 4.5
    # Transition 3 samples BEFORE the knee at t_td=200.
    flight, transition = two_regime_landing(dt=dt, t_td=200.0, on_ground_delay_samples=-3.0)
    estimate = GlrtEstimator().estimate(flight)
    assert estimate.t_td < transition
    assert estimate.diagnostics["on_ground_clamped"] is True
    assert estimate.t_td == pytest.approx(transition - ON_GROUND_BOUND_GUARD_S)
    assert "pre_clamp_t_td" in estimate.diagnostics
