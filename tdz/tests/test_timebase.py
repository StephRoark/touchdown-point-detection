"""Tests for the timebase module (Task 8).

Covers Property 3 (Kinematic Interpolation Accuracy Bound) and Property 4
(Asynchronous Timestamp Preservation), plus known-answer and edge unit tests:
exact straight-line dead-reckoning, kinematic-beats-naive-merge, track 0/360
wraparound, velocity-missing linear fallback with DEGRADED_INTERPOLATION,
common-grid resample spacing + time-delta channel, no-overshoot monotone
altitude interpolation, and the knots->m/s conversion constant.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pyproj import Geod

from tdz.models import FailureReason
from tdz.timebase import (
    KNOTS_TO_MPS,
    MAX_TIMESTAMP_MISALIGNMENT_ERROR_M,
    ContinuousTimebase,
    compute_time_deltas,
    interpolate_position_at,
    interpolate_track_deg,
    interpolate_velocity_at,
    monotone_interpolate,
    resample_to_grid,
)

_GEOD = Geod(ellps="WGS84")


def _geodesic_distance_m(lat1, lon1, lat2, lon2) -> float:
    """Geodesic distance between two points (meters)."""
    _fwd, _back, dist = _GEOD.inv(lon1, lat1, lon2, lat2)
    return float(dist)


def _truth_position(lat0, lon0, heading_deg, speed_mps, t):
    """Analytic truth: advance origin along a constant azimuth by speed*t."""
    lon2, lat2, _back = _GEOD.fwd(lon0, lat0, heading_deg, speed_mps * t)
    return lat2, lon2


# ---------------------------------------------------------------------------
# Property 3: Kinematic Interpolation Accuracy Bound
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    lat0=st.floats(min_value=-60.0, max_value=60.0),
    lon0=st.floats(min_value=-150.0, max_value=150.0),
    gs_kt=st.floats(min_value=120.0, max_value=150.0),
    heading=st.floats(min_value=0.0, max_value=360.0),
    vel_offset=st.floats(min_value=0.0, max_value=5.0),
    query_t=st.floats(min_value=5.0, max_value=15.0),
)
def test_kinematic_accuracy_bound(lat0, lon0, gs_kt, heading, vel_offset, query_t):
    """Feature: touchdown-point-detection, Property 3: Kinematic Interpolation Accuracy Bound

    For a straight, constant-velocity trajectory with position and velocity
    sampled on OFFSET timebases (offset 0-5 s) at 120-150 kt, kinematic
    dead-reckoning recovers the position at an arbitrary query time within the
    30 ft / 9.14 m timestamp-misalignment bound (Req 10.2, 10.3).
    """
    speed_mps = gs_kt * KNOTS_TO_MPS

    # Position sampled every 4 s; velocity sampled on a DISTINCT offset timebase.
    position_times = np.arange(0.0, 20.0001, 4.0)
    lats = np.empty_like(position_times)
    lons = np.empty_like(position_times)
    for i, t in enumerate(position_times):
        lats[i], lons[i] = _truth_position(lat0, lon0, heading, speed_mps, t)

    velocity_times = position_times + vel_offset
    gs = np.full(velocity_times.shape, gs_kt)
    tracks = np.full(velocity_times.shape, heading % 360.0)

    result = interpolate_position_at(
        position_times, lats, lons, velocity_times, gs, tracks, query_t
    )

    true_lat, true_lon = _truth_position(lat0, lon0, heading, speed_mps, query_t)
    error_m = _geodesic_distance_m(result.lat, result.lon, true_lat, true_lon)

    assert error_m < MAX_TIMESTAMP_MISALIGNMENT_ERROR_M, (
        f"kinematic error {error_m:.3f} m exceeds 9.14 m bound"
    )
    assert not result.degraded


# ---------------------------------------------------------------------------
# Property 4: Asynchronous Timestamp Preservation
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    pos_times=st.lists(
        st.floats(min_value=0.0, max_value=1000.0),
        min_size=2,
        max_size=40,
        unique=True,
    ),
    vel_times=st.lists(
        st.floats(min_value=0.0, max_value=1000.0),
        min_size=2,
        max_size=40,
        unique=True,
    ),
)
def test_async_timestamp_preservation(pos_times, vel_times):
    """Feature: touchdown-point-detection, Property 4: Asynchronous Timestamp Preservation

    The position and velocity timebases are kept as separate arrays and never
    merged: the count of distinct position timestamps and of distinct velocity
    timestamps are each conserved, and no single merged sample-time array is
    formed (Req 8.3, 10.1).
    """
    pos_times = np.sort(np.array(pos_times, dtype=float))
    vel_times = np.sort(np.array(vel_times, dtype=float))

    n_pos = pos_times.size
    n_vel = vel_times.size
    lats = np.linspace(0.0, 0.1, n_pos)
    lons = np.linspace(0.0, 0.1, n_pos)
    gs = np.full(n_vel, 130.0)
    tracks = np.full(n_vel, 90.0)

    tb = ContinuousTimebase(
        position_times=pos_times,
        latitudes=lats,
        longitudes=lons,
        velocity_times=vel_times,
        groundspeeds_kt=gs,
        tracks_deg=tracks,
    )

    # Distinct-timestamp counts conserved on each separate timebase.
    assert set(tb.position_times.tolist()) == set(pos_times.tolist())
    assert set(tb.velocity_times.tolist()) == set(vel_times.tolist())
    # The two arrays remain distinct (no merge into one sample-time array).
    assert tb.position_times.size == n_pos
    assert tb.velocity_times.size == n_vel
    # The time-delta channels track each native timebase independently.
    assert tb.position_time_deltas.size == n_pos
    assert tb.velocity_time_deltas.size == n_vel


# ---------------------------------------------------------------------------
# Known-answer: exact straight-line dead-reckoning along a meridian
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_meridian_constant_speed_is_exact():
    """A due-north (meridian) constant-speed leg dead-reckons exactly.

    Meridians are geodesics of constant azimuth 0, so composing forward steps is
    exact; the kinematic query matches analytic truth to sub-centimeter even
    with an offset velocity timebase.
    """
    lat0, lon0, heading, gs_kt = 10.0, 20.0, 0.0, 140.0
    speed_mps = gs_kt * KNOTS_TO_MPS

    position_times = np.arange(0.0, 20.0001, 4.0)
    lats = np.empty_like(position_times)
    lons = np.empty_like(position_times)
    for i, t in enumerate(position_times):
        lats[i], lons[i] = _truth_position(lat0, lon0, heading, speed_mps, t)

    velocity_times = position_times + 2.0  # distinct, offset timebase
    gs = np.full(velocity_times.shape, gs_kt)
    tracks = np.full(velocity_times.shape, heading)

    result = interpolate_position_at(
        position_times, lats, lons, velocity_times, gs, tracks, 10.0
    )
    true_lat, true_lon = _truth_position(lat0, lon0, heading, speed_mps, 10.0)
    error_m = _geodesic_distance_m(result.lat, result.lon, true_lat, true_lon)
    assert error_m < 0.05


@pytest.mark.unit
def test_kinematic_beats_naive_nearest_sample():
    """Dead-reckoning vastly beats naive nearest-sample selection mid-interval.

    Demonstrates the silent-bias trap: at ~140 kt a 2 s timestamp misalignment
    injects ~140 m of position error when the nearest sample is taken as-is,
    while kinematic interpolation stays sub-meter.
    """
    lat0, lon0, heading, gs_kt = 0.0, 0.0, 90.0, 140.0  # equator, due east (geodesic)
    speed_mps = gs_kt * KNOTS_TO_MPS

    position_times = np.arange(0.0, 20.0001, 4.0)
    lats = np.empty_like(position_times)
    lons = np.empty_like(position_times)
    for i, t in enumerate(position_times):
        lats[i], lons[i] = _truth_position(lat0, lon0, heading, speed_mps, t)

    velocity_times = position_times.copy()
    gs = np.full(velocity_times.shape, gs_kt)
    tracks = np.full(velocity_times.shape, heading)

    query_t = 10.0  # exactly between samples at 8 and 12
    true_lat, true_lon = _truth_position(lat0, lon0, heading, speed_mps, query_t)

    kin = interpolate_position_at(
        position_times, lats, lons, velocity_times, gs, tracks, query_t
    )
    kin_err = _geodesic_distance_m(kin.lat, kin.lon, true_lat, true_lon)

    nearest = int(np.argmin(np.abs(position_times - query_t)))
    naive_err = _geodesic_distance_m(
        lats[nearest], lons[nearest], true_lat, true_lon
    )

    assert kin_err < 1.0
    assert naive_err > 100.0
    assert naive_err > kin_err


# ---------------------------------------------------------------------------
# Velocity interpolation: linear groundspeed, wrap-aware track
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_groundspeed_linear_interpolation():
    """Groundspeed interpolates linearly between velocity samples."""
    gs_kt, track = interpolate_velocity_at(
        np.array([0.0, 10.0]), np.array([100.0, 140.0]), np.array([90.0, 90.0]), 2.5
    )
    assert gs_kt == pytest.approx(110.0)
    assert track == pytest.approx(90.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "a0,a1,expected",
    [
        (350.0, 10.0, 0.0),   # forward across 0/360
        (10.0, 350.0, 0.0),   # backward across 0/360
        (10.0, 40.0, 25.0),   # no wrap
        (170.0, 190.0, 180.0),  # no wrap near south
    ],
)
def test_track_wraparound(a0, a1, expected):
    """Track interpolation follows the shortest angular path across 0/360 deg."""
    result = interpolate_track_deg(np.array([0.0, 10.0]), np.array([a0, a1]), 5.0)
    # Compare on the circle (0 and 360 are equal).
    circular_err = min(abs(result - expected), 360.0 - abs(result - expected))
    assert circular_err < 1e-6


@pytest.mark.unit
def test_velocity_interp_clamps_outside_range():
    """Queries outside the velocity range hold the nearest endpoint value."""
    gs_kt, track = interpolate_velocity_at(
        np.array([0.0, 10.0]), np.array([100.0, 140.0]), np.array([80.0, 80.0]), 50.0
    )
    assert gs_kt == pytest.approx(140.0)
    assert track == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Degraded fallback (Req 10.4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_velocity_missing_falls_back_to_linear_degraded():
    """Null velocity where kinematic interp needs it -> linear fallback + flag.

    The query is flagged degraded with DEGRADED_INTERPOLATION and falls back to
    linear positional interpolation between the two nearest valid position
    messages (Req 10.4) rather than raising.
    """
    position_times = np.array([0.0, 10.0])
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 2.0])
    velocity_times = np.array([0.0, 10.0])
    gs = np.array([np.nan, np.nan])  # velocity unavailable
    tracks = np.array([np.nan, np.nan])

    result = interpolate_position_at(
        position_times, lats, lons, velocity_times, gs, tracks, 5.0
    )

    assert result.degraded is True
    assert result.reason is FailureReason.DEGRADED_INTERPOLATION
    # Linear midpoint between the two nearest valid position messages.
    assert result.lat == pytest.approx(0.5)
    assert result.lon == pytest.approx(1.0)


@pytest.mark.unit
def test_empty_velocity_falls_back_degraded():
    """Empty velocity timebase degrades to linear positional interpolation."""
    position_times = np.array([0.0, 10.0])
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 2.0])
    result = interpolate_position_at(
        position_times,
        lats,
        lons,
        np.array([]),
        np.array([]),
        np.array([]),
        5.0,
    )
    assert result.degraded is True
    assert result.reason is FailureReason.DEGRADED_INTERPOLATION


@pytest.mark.unit
def test_linear_method_is_not_degraded():
    """An intentional method='linear' query is NOT flagged degraded."""
    position_times = np.array([0.0, 10.0])
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 2.0])
    result = interpolate_position_at(
        position_times,
        lats,
        lons,
        np.array([0.0, 10.0]),
        np.array([130.0, 130.0]),
        np.array([90.0, 90.0]),
        5.0,
        method="linear",
    )
    assert result.degraded is False
    assert result.reason is None
    assert result.lat == pytest.approx(0.5)
    assert result.lon == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Time-delta channel
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_time_deltas():
    """Time-delta channel reports inter-sample gaps; first element is 0."""
    times = np.array([0.0, 4.0, 9.0, 11.0])
    deltas = compute_time_deltas(times)
    assert deltas.tolist() == [0.0, 4.0, 5.0, 2.0]
    assert deltas.shape == times.shape


@pytest.mark.unit
def test_compute_time_deltas_single_sample():
    """A single timestamp yields a single zero delta."""
    assert compute_time_deltas(np.array([7.0])).tolist() == [0.0]


# ---------------------------------------------------------------------------
# Common-grid resampling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resample_to_grid_spacing_and_time_deltas():
    """common_grid resamples onto fixed spacing with a correct time-delta channel."""
    heading, gs_kt = 0.0, 130.0
    speed_mps = gs_kt * KNOTS_TO_MPS
    position_times = np.arange(0.0, 20.0001, 4.0)
    lats = np.empty_like(position_times)
    lons = np.empty_like(position_times)
    for i, t in enumerate(position_times):
        lats[i], lons[i] = _truth_position(0.0, 0.0, heading, speed_mps, t)
    alts = 1000.0 - 5.0 * position_times  # gentle monotone descent
    velocity_times = position_times + 1.5
    gs = np.full(velocity_times.shape, gs_kt)
    tracks = np.full(velocity_times.shape, heading)

    grid_interval = 2.0
    result = resample_to_grid(
        position_times,
        lats,
        lons,
        alts,
        velocity_times,
        gs,
        tracks,
        grid_interval_s=grid_interval,
        t_start=0.0,
        t_end=20.0,
    )

    # Grid is evenly spaced at the configured interval.
    assert np.allclose(np.diff(result.times), grid_interval)
    assert result.times[0] == 0.0
    assert result.times[-1] == pytest.approx(20.0)
    # Time-delta channel matches the grid spacing (first element 0).
    assert result.time_deltas[0] == 0.0
    assert np.allclose(result.time_deltas[1:], grid_interval)
    # All output channels share the grid length.
    assert result.latitudes.shape == result.times.shape
    assert result.geometric_altitudes.shape == result.times.shape
    assert result.groundspeeds_kt.shape == result.times.shape
    # No degradation: velocity is available throughout.
    assert not result.degraded_mask.any()
    # Groundspeed channel recovers the constant speed.
    assert np.allclose(result.groundspeeds_kt, gs_kt)


@pytest.mark.unit
def test_resample_degraded_mask_when_velocity_missing():
    """Off-sample grid points where velocity is null are flagged degraded.

    A grid point coinciding exactly with a position sample needs no velocity
    (zero dead-reckoning leg) so it is not degraded; the intermediate grid
    points require velocity, are unavailable, and fall back to linear (flagged).
    """
    position_times = np.array([0.0, 4.0, 8.0])
    lats = np.array([0.0, 0.01, 0.02])
    lons = np.array([0.0, 0.0, 0.0])
    alts = np.array([300.0, 200.0, 100.0])
    velocity_times = np.array([0.0, 4.0, 8.0])
    gs = np.array([np.nan, np.nan, np.nan])
    tracks = np.array([np.nan, np.nan, np.nan])

    result = resample_to_grid(
        position_times,
        lats,
        lons,
        alts,
        velocity_times,
        gs,
        tracks,
        grid_interval_s=2.0,
        t_start=0.0,
        t_end=8.0,
    )
    # Grid: [0, 2, 4, 6, 8]; t=0,4,8 land on position samples (not degraded);
    # t=2,6 require velocity that is null -> degraded linear fallback.
    assert result.degraded_mask.tolist() == [False, True, False, True, False]


# ---------------------------------------------------------------------------
# Monotone altitude interpolation (no overshoot)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_monotone_interpolation_no_overshoot():
    """A monotone input stays monotone with no over/undershoot (PCHIP)."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 0.0, 0.0, 1.0, 10.0])  # nondecreasing, sharp knee
    xq = np.linspace(0.0, 4.0, 200)
    yq = monotone_interpolate(x, y, xq)
    # Bounded by the data range (no overshoot).
    assert yq.min() >= y.min() - 1e-9
    assert yq.max() <= y.max() + 1e-9
    # Monotone nondecreasing output.
    assert np.all(np.diff(yq) >= -1e-9)


@pytest.mark.unit
def test_monotone_interpolation_passes_through_nodes():
    """Monotone interpolation reproduces the data values at the nodes."""
    x = np.array([0.0, 2.0, 5.0, 9.0])
    y = np.array([100.0, 80.0, 60.0, 0.0])
    yq = monotone_interpolate(x, y, x)
    assert np.allclose(yq, y)


@pytest.mark.unit
def test_monotone_interpolation_drops_nan_nodes():
    """NaN nodes are ignored; a clean monotone fit results."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, np.nan, 2.0, 3.0])
    yq = monotone_interpolate(x, y, np.array([2.5]))
    assert yq[0] == pytest.approx(2.5, abs=0.2)


# ---------------------------------------------------------------------------
# Continuous-time strategy
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_continuous_timebase_queries():
    """ContinuousTimebase exposes kinematic position and velocity queries."""
    heading, gs_kt = 0.0, 130.0
    speed_mps = gs_kt * KNOTS_TO_MPS
    position_times = np.arange(0.0, 16.0001, 4.0)
    lats = np.empty_like(position_times)
    lons = np.empty_like(position_times)
    for i, t in enumerate(position_times):
        lats[i], lons[i] = _truth_position(5.0, 5.0, heading, speed_mps, t)
    velocity_times = position_times + 1.0
    gs = np.full(velocity_times.shape, gs_kt)
    tracks = np.full(velocity_times.shape, heading)

    tb = ContinuousTimebase(
        position_times=position_times,
        latitudes=lats,
        longitudes=lons,
        velocity_times=velocity_times,
        groundspeeds_kt=gs,
        tracks_deg=tracks,
    )

    pos = tb.position_at(6.0)
    true_lat, true_lon = _truth_position(5.0, 5.0, heading, speed_mps, 6.0)
    assert _geodesic_distance_m(pos.lat, pos.lon, true_lat, true_lon) < 0.05

    g, a = tb.velocity_at(6.0)
    assert g == pytest.approx(gs_kt)
    assert a == pytest.approx(heading)


# ---------------------------------------------------------------------------
# Units: knots -> m/s
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_knots_to_mps_constant():
    """The knots->m/s conversion constant is exact and applied correctly."""
    assert KNOTS_TO_MPS == 0.514444
    # 1 international knot = 1852 m / 3600 s.
    assert KNOTS_TO_MPS == pytest.approx(1852.0 / 3600.0, abs=1e-6)
    assert 100.0 * KNOTS_TO_MPS == pytest.approx(51.4444)
