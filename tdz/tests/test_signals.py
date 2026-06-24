"""Tests for the signals/feature module (Task 11).

Covers:

* Segmented (piecewise) regression on raw groundspeed (Task 11.1, Req 16.1):
  a known-answer two-slope recovery test plus a property test over randomized
  breakpoints / slopes / noise (breakpoint recovery within a cadence/noise-scaled
  tolerance).
* Corroborating smoothed derivatives (Task 11.2, Req 16.2-16.6): derivative
  known-answer tests (constant deceleration -> constant 1st derivative / zero
  jerk; quadratic speed -> linear derivative) reporting RMS error vs the
  analytical derivative; GP posterior-std availability; and the <5-valid-samples
  reliability flag.
* Feature channels (Task 11.3): distance-to-threshold monotone toward ~0 on an
  approach, time-delta channel matching ``compute_time_deltas``, documented
  timebases, and FlightRecord slot population.
* The QAR-vs-smoothed-deceleration RMS-discrepancy harness (Task 11.4, Req 16.7).
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pyproj import Geod

from tdz.config.schema import SignalsConfig
from tdz.models import FlightRecord, RunwayReference
from tdz.signals import (
    MIN_VALID_SAMPLES_IN_WINDOW,
    build_feature_channels,
    deceleration_rms_discrepancy,
    fit_segmented_groundspeed,
    populate_flight_record,
    smoothed_derivatives,
)
from tdz.timebase.interpolation import KNOTS_TO_MPS, compute_time_deltas

_GEOD = Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signals_config(method: str = "savgol", **overrides) -> SignalsConfig:
    base = dict(
        smoothing_method=method,
        savgol_window_samples=7,
        savgol_poly_order=3,
        gp_length_scale_s=8.0,
        gp_noise_variance=0.5,
    )
    base.update(overrides)
    return SignalsConfig(**base)


def _runway(heading_deg: float = 250.0) -> RunwayReference:
    return RunwayReference(
        threshold_lat=33.94,
        threshold_lon=-118.40,
        heading_deg=heading_deg,
        elevation_m=30.0,
        elevation_datum="HAE",
        geoid_undulation_m=0.0,
        length_m=3000.0,
        width_m=45.0,
        displaced=False,
    )


def _point_before_threshold(runway: RunwayReference, distance_m: float):
    """Lat/lon ``distance_m`` metres BEFORE the threshold along the centerline.

    Walking from the threshold along the reciprocal of the landing heading puts
    the point on the approach side, so its signed along-runway distance is
    ``~ -distance_m`` (negative = before threshold).
    """
    az = (runway.heading_deg + 180.0) % 360.0
    lon, lat, _back = _GEOD.fwd(runway.threshold_lon, runway.threshold_lat, az, distance_m)
    return lat, lon


def _two_slope_speed(t, tau, v0_mps, slope1, slope2):
    """Continuous two-slope speed profile (m/s) with the knee at ``tau``."""
    v_at_tau = v0_mps + slope1 * tau
    return np.where(t <= tau, v0_mps + slope1 * t, v_at_tau + slope2 * (t - tau))


def _make_flight(
    *,
    position_times,
    latitudes,
    longitudes,
    velocity_times,
    groundspeeds_kt,
    runway: RunwayReference,
    geometric_altitudes=None,
) -> FlightRecord:
    position_times = np.asarray(position_times, dtype=float)
    velocity_times = np.asarray(velocity_times, dtype=float)
    npos = position_times.size
    nvel = velocity_times.size
    if geometric_altitudes is None:
        geometric_altitudes = np.full(npos, np.nan)
    return FlightRecord(
        flight_id="SIG",
        aircraft_type="B738",
        ads_b_source="aireon",
        position_times=position_times,
        velocity_times=velocity_times,
        latitudes=np.asarray(latitudes, dtype=float),
        longitudes=np.asarray(longitudes, dtype=float),
        geometric_altitudes=np.asarray(geometric_altitudes, dtype=float),
        barometric_altitudes=np.full(npos, np.nan),
        groundspeeds=np.asarray(groundspeeds_kt, dtype=float),
        tracks=np.full(nvel, runway.heading_deg),
        baro_vertical_rates=np.full(nvel, np.nan),
        on_ground_flags=np.zeros(npos, dtype=bool),
        on_ground_transition_time=None,
        runway=runway,
    )


# ---------------------------------------------------------------------------
# 11.1 Segmented regression: known-answer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_segmented_recovers_known_breakpoint():
    """Feature: touchdown-point-detection, Req 16.1 -- segmented regression.

    A synthetic two-slope groundspeed profile (gentle approach decel -> steep
    ground-roll decel) with a known breakpoint is recovered to within one
    sample cadence, and both segment decelerations are recovered closely.
    """
    dt = 4.5
    n_before, n_after = 6, 7
    t = np.arange(n_before + n_after) * dt
    tau = n_before * dt
    slope1, slope2 = -0.5, -2.5  # m/s^2
    v0 = 80.0  # m/s
    v_mps = _two_slope_speed(t, tau, v0, slope1, slope2)
    gs_kt = v_mps / KNOTS_TO_MPS

    fit = fit_segmented_groundspeed(t, gs_kt, n_segments=2)

    assert abs(fit.breakpoint_time - tau) <= dt
    assert fit.n_segments == 2
    assert len(fit.slopes_mps2) == 2
    assert fit.slopes_mps2[0] == pytest.approx(slope1, abs=0.1)
    assert fit.slopes_mps2[1] == pytest.approx(slope2, abs=0.1)
    assert fit.residual_rms_mps < 0.5


@pytest.mark.unit
def test_segmented_three_segment_fit():
    """A 3-segment fit returns two breakpoints and three slopes (Req 16.1)."""
    dt = 4.0
    t = np.arange(18) * dt
    # approach (gentle) -> hard braking -> easing rollout
    knots = (24.0, 48.0)
    v = np.empty_like(t)
    v0, s = 85.0, [-0.4, -3.0, -1.2]
    cur = v0
    last_t = 0.0
    seg = 0
    for i, ti in enumerate(t):
        if seg < 2 and ti > knots[seg]:
            cur = cur + s[seg] * (knots[seg] - last_t)
            last_t = knots[seg]
            seg += 1
        v[i] = cur + s[seg] * (ti - last_t)
    fit = fit_segmented_groundspeed(t, v / KNOTS_TO_MPS, n_segments=3)
    assert fit.n_segments == 3
    assert len(fit.breakpoint_times) == 2
    assert len(fit.slopes_mps2) == 3
    # Primary breakpoint is the steepest-deceleration transition (~first knot).
    assert abs(fit.breakpoint_time - knots[0]) <= 2 * dt


@pytest.mark.unit
def test_segmented_rejects_too_few_samples():
    """Too few valid samples for the requested segments -> ValueError."""
    t = np.array([0.0, 4.0, 8.0])
    gs = np.array([150.0, 140.0, 100.0])
    with pytest.raises(ValueError):
        fit_segmented_groundspeed(t, gs, n_segments=2)


# ---------------------------------------------------------------------------
# 11.1 Segmented regression: property (randomized breakpoints/slopes/noise)
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    cadence=st.floats(min_value=4.0, max_value=5.0),
    n_before=st.integers(min_value=4, max_value=10),
    n_after=st.integers(min_value=4, max_value=10),
    v0_mps=st.floats(min_value=60.0, max_value=90.0),
    slope1=st.floats(min_value=-1.0, max_value=-0.2),
    contrast=st.floats(min_value=0.8, max_value=2.5),
    noise_sigma=st.floats(min_value=0.0, max_value=0.4),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_segmented_breakpoint_recovery_property(
    cadence, n_before, n_after, v0_mps, slope1, contrast, noise_sigma, seed
):
    """Feature: touchdown-point-detection, Req 16.1 -- breakpoint recovery.

    For a randomized continuous two-slope groundspeed profile (clear slope
    contrast) sampled at the 4-5 s ADS-B cadence with additive noise, the
    segmented fit recovers the breakpoint within a tolerance scaled to the
    cadence and the noise-to-contrast ratio.
    """
    slope2 = slope1 - contrast  # steeper (more negative) ground-roll decel
    t = np.arange(n_before + n_after) * cadence
    tau = n_before * cadence
    v_mps = _two_slope_speed(t, tau, v0_mps, slope1, slope2)

    rng = np.random.default_rng(seed)
    v_noisy = v_mps + rng.normal(0.0, noise_sigma, size=v_mps.shape)
    gs_kt = v_noisy / KNOTS_TO_MPS

    fit = fit_segmented_groundspeed(t, gs_kt, n_segments=2)

    tol_s = 1.2 * cadence + 6.0 * noise_sigma / contrast
    assert abs(fit.breakpoint_time - tau) <= tol_s, (
        f"breakpoint={fit.breakpoint_time:.2f} tau={tau:.2f} tol={tol_s:.2f}"
    )


# ---------------------------------------------------------------------------
# 11.2 Derivative quality known-answer (Req 16.2, 16.7 spirit)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("method", ["savgol", "gp"])
def test_derivative_constant_deceleration_known_answer(method):
    """Constant deceleration -> constant 1st derivative and ~zero jerk.

    Reports the RMS error of the smoothed deceleration vs the analytical value
    (Req 16.7 spirit). A constant-deceleration (linear-speed) profile has a
    constant analytical derivative ``-a`` and zero jerk.
    """
    dt = 4.5
    t = np.arange(14) * dt
    a = 1.5  # m/s^2
    v_mps = 95.0 - a * t
    gs_kt = v_mps / KNOTS_TO_MPS

    result = smoothed_derivatives(t, gs_kt, _signals_config(method))

    analytic = np.full_like(t, -a)
    rms = deceleration_rms_discrepancy(result.deceleration_mps2, analytic)
    assert rms < 1e-2, f"deceleration RMS error = {rms:.2e} m/s^2"
    # Jerk of a constant-deceleration profile is ~0.
    assert np.nanmax(np.abs(result.jerk_mps3)) < 1e-2
    assert result.reliable


@pytest.mark.unit
@pytest.mark.parametrize("method", ["savgol", "gp"])
def test_derivative_quadratic_speed_known_answer(method):
    """Quadratic speed profile -> linear analytical derivative.

    ``v(t) = v0 - a t - 0.5 j t^2`` has ``v'(t) = -a - j t`` (linear) and
    ``v''(t) = -j`` (constant). The smoothed deceleration matches the linear
    analytical derivative within a documented RMS tolerance.
    """
    dt = 4.0
    t = np.arange(16) * dt
    a, j = 1.0, 0.05  # m/s^2 and m/s^3
    v_mps = 90.0 - a * t - 0.5 * j * t**2
    gs_kt = v_mps / KNOTS_TO_MPS

    result = smoothed_derivatives(t, gs_kt, _signals_config(method))

    analytic_decel = -a - j * t
    rms = deceleration_rms_discrepancy(result.deceleration_mps2, analytic_decel)
    assert rms < 5e-2, f"deceleration RMS error = {rms:.2e} m/s^2"
    # Second derivative recovers ~ -j.
    assert np.nanmedian(result.jerk_mps3) == pytest.approx(-j, abs=1e-2)


@pytest.mark.unit
def test_gp_reports_posterior_std():
    """The GP path reports a positive per-sample derivative posterior std (Req 16.4)."""
    dt = 4.5
    t = np.arange(12) * dt
    v_mps = 90.0 - 1.2 * t
    gs_kt = v_mps / KNOTS_TO_MPS

    result = smoothed_derivatives(t, gs_kt, _signals_config("gp"))

    assert result.gp_length_scale_s == pytest.approx(8.0)
    finite = np.isfinite(result.deceleration_mps2)
    assert np.all(result.derivative_uncertainty[finite] > 0.0)


@pytest.mark.unit
def test_piecewise_smoothing_flag_set_with_breakpoint():
    """Supplying a breakpoint enables piecewise (non-stationary) smoothing (Req 16.4)."""
    dt = 4.0
    t = np.arange(16) * dt
    v_mps = _two_slope_speed(t, 32.0, 85.0, -0.4, -2.6)
    gs_kt = v_mps / KNOTS_TO_MPS

    no_bp = smoothed_derivatives(t, gs_kt, _signals_config())
    with_bp = smoothed_derivatives(t, gs_kt, _signals_config(), breakpoint_time=32.0)

    assert not no_bp.piecewise
    assert with_bp.piecewise


# ---------------------------------------------------------------------------
# 11.2 Reliability flag (Req 16.6)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reliability_flag_fewer_than_five_samples():
    """Fewer than 5 valid samples in the window -> reliable=False (Req 16.6)."""
    dt = 4.5
    t = np.arange(4) * dt  # only 4 samples, < MIN_VALID_SAMPLES_IN_WINDOW
    gs_kt = (90.0 - 1.5 * t) / KNOTS_TO_MPS

    result = smoothed_derivatives(t, gs_kt, _signals_config(savgol_window_samples=7))

    assert MIN_VALID_SAMPLES_IN_WINDOW == 5
    assert result.min_valid_in_window < 5
    assert result.reliable is False


@pytest.mark.unit
def test_reliability_flag_true_with_enough_samples():
    """A window with >=5 valid samples is reliable (Req 16.6)."""
    dt = 4.5
    t = np.arange(12) * dt
    gs_kt = (90.0 - 1.5 * t) / KNOTS_TO_MPS

    result = smoothed_derivatives(t, gs_kt, _signals_config())

    assert result.min_valid_in_window >= 5
    assert result.reliable is True
    # Configured smoothing window reported in diagnostics (Req 16.5).
    assert result.window_samples == 7


# ---------------------------------------------------------------------------
# 11.3 Feature channels
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_distance_to_threshold_monotone_toward_zero_on_approach():
    """distance_to_threshold shrinks toward ~0 along an approach to the threshold.

    Position samples are placed on the centerline at decreasing distances before
    the threshold; their signed along-runway distance is negative and its
    magnitude decreases monotonically toward ~0 near the threshold (Req: distance
    feature channel, on the POSITION timebase).
    """
    runway = _runway()
    distances_before = [1500.0, 1000.0, 600.0, 300.0, 100.0, 20.0]
    lats, lons = [], []
    for d in distances_before:
        lat, lon = _point_before_threshold(runway, d)
        lats.append(lat)
        lons.append(lon)
    position_times = np.arange(len(distances_before)) * 4.0
    velocity_times = position_times + 1.0
    gs_kt = np.linspace(150.0, 135.0, len(distances_before))

    flight = _make_flight(
        position_times=position_times,
        latitudes=lats,
        longitudes=lons,
        velocity_times=velocity_times,
        groundspeeds_kt=gs_kt,
        runway=runway,
    )
    channels = build_feature_channels(flight, _signals_config())

    dist = channels.distance_to_threshold_m
    # On the position timebase.
    assert dist.shape == flight.position_times.shape
    # Before the threshold: negative signed distance, decreasing magnitude.
    assert np.all(dist < 5.0)
    assert np.all(np.diff(np.abs(dist)) < 0.0)
    # Near the threshold at the end (~ -20 m, ~0 relative to a 3000 m runway).
    assert abs(dist[-1]) < 50.0
    # Lateral offset ~0 on the centerline.
    assert np.all(np.abs(channels.lateral_offset_m) < 1.0)


@pytest.mark.unit
def test_time_delta_channels_match_compute_time_deltas():
    """The feature time-delta channels match compute_time_deltas on each timebase."""
    runway = _runway()
    position_times = np.array([0.0, 4.0, 9.0, 13.5, 18.0])
    velocity_times = np.array([1.0, 5.5, 10.0, 14.0, 19.5, 24.0])
    lats = np.full(position_times.size, runway.threshold_lat)
    lons = np.full(position_times.size, runway.threshold_lon)
    gs_kt = np.linspace(150.0, 70.0, velocity_times.size)

    flight = _make_flight(
        position_times=position_times,
        latitudes=lats,
        longitudes=lons,
        velocity_times=velocity_times,
        groundspeeds_kt=gs_kt,
        runway=runway,
    )
    channels = build_feature_channels(flight, _signals_config())

    np.testing.assert_allclose(
        channels.position_time_deltas, compute_time_deltas(position_times)
    )
    np.testing.assert_allclose(
        channels.velocity_time_deltas, compute_time_deltas(velocity_times)
    )
    # Documented timebases: derivative channels on velocity, distance on position.
    assert channels.deceleration_mps2.shape == velocity_times.shape
    assert channels.distance_to_threshold_m.shape == position_times.shape


@pytest.mark.unit
def test_height_above_runway_channel():
    """height_above_runway = geometric altitude - geoid-corrected runway HAE."""
    runway = _runway()  # HAE datum, elevation 30 m
    position_times = np.arange(5) * 4.0
    velocity_times = position_times + 1.0
    lats = np.full(position_times.size, runway.threshold_lat)
    lons = np.full(position_times.size, runway.threshold_lon)
    heights = np.array([200.0, 150.0, 100.0, 50.0, 5.0])
    geo_alt = runway.elevation_m + heights
    gs_kt = np.linspace(150.0, 130.0, velocity_times.size)

    flight = _make_flight(
        position_times=position_times,
        latitudes=lats,
        longitudes=lons,
        velocity_times=velocity_times,
        groundspeeds_kt=gs_kt,
        runway=runway,
        geometric_altitudes=geo_alt,
    )
    channels = build_feature_channels(flight, _signals_config())
    np.testing.assert_allclose(channels.height_above_runway_m, heights)


@pytest.mark.unit
def test_populate_flight_record_sets_slots():
    """populate_flight_record fills the derived-signal slots on the right timebases."""
    runway = _runway()
    position_times = np.arange(10) * 4.0
    velocity_times = position_times + 1.0
    lats, lons = [], []
    for i in range(position_times.size):
        lat, lon = _point_before_threshold(runway, 1200.0 - 120.0 * i)
        lats.append(lat)
        lons.append(lon)
    gs_kt = np.linspace(150.0, 70.0, velocity_times.size)

    flight = _make_flight(
        position_times=position_times,
        latitudes=lats,
        longitudes=lons,
        velocity_times=velocity_times,
        groundspeeds_kt=gs_kt,
        runway=runway,
    )
    assert flight.smoothed_deceleration is None  # not yet populated

    channels = populate_flight_record(flight, _signals_config())

    # Velocity-timebase slots.
    assert flight.smoothed_deceleration is not None
    assert flight.smoothed_deceleration.shape == velocity_times.shape
    assert flight.smoothed_jerk.shape == velocity_times.shape
    assert flight.derivative_uncertainties.shape == velocity_times.shape
    assert flight.time_deltas.shape == velocity_times.shape
    np.testing.assert_allclose(flight.time_deltas, compute_time_deltas(velocity_times))
    # Position-timebase slot.
    assert flight.distance_to_threshold.shape == position_times.shape
    # Returned channels are consistent with the populated slots.
    np.testing.assert_allclose(
        flight.smoothed_deceleration, channels.deceleration_mps2, equal_nan=True
    )


# ---------------------------------------------------------------------------
# 11.4 QAR-vs-smoothed-deceleration RMS-discrepancy harness (Req 16.7)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_qar_vs_smoothed_rms_discrepancy_harness():
    """Smoothed ADS-B deceleration vs synthetic 'QAR' acceleration -> small RMS.

    Reusable Req 16.7 harness: the full held-out-sample comparison against real
    QAR acceleration runs in the validation harness (Task 22); here we exercise
    the RMS-discrepancy core on a synthetic constant-deceleration 'QAR' truth.
    """
    dt = 4.5
    t = np.arange(14) * dt
    a = 1.3  # m/s^2 constant deceleration "truth"
    gs_kt = (92.0 - a * t) / KNOTS_TO_MPS

    result = smoothed_derivatives(t, gs_kt, _signals_config())
    qar_accel = np.full_like(t, -a)  # QAR-derived acceleration (m/s^2)

    rms = deceleration_rms_discrepancy(result.deceleration_mps2, qar_accel)
    assert rms < 1e-2, f"QAR-vs-smoothed deceleration RMS discrepancy = {rms:.3e} m/s^2"


@pytest.mark.unit
def test_rms_discrepancy_ignores_nan_and_handles_empty():
    """The RMS-discrepancy harness ignores NaN pairs and returns NaN when empty."""
    a = np.array([1.0, np.nan, 3.0, 4.0])
    b = np.array([1.0, 2.0, np.nan, 4.5])
    # Only indices 0 and 3 are finite in both: diffs 0.0 and -0.5.
    rms = deceleration_rms_discrepancy(a, b)
    assert rms == pytest.approx(np.sqrt((0.0**2 + 0.5**2) / 2))

    all_nan = np.array([np.nan, np.nan])
    assert np.isnan(deceleration_rms_discrepancy(all_nan, all_nan))
