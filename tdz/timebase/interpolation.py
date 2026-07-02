"""Async-timestamp-preserving kinematic interpolation & resampling (Task 8).

ADS-B position and velocity messages from asynchronous sources (Aireon) carry
**separate** emission timestamps. Naively pairing the nearest position and
velocity sample as if they were simultaneous injects a position error of
``velocity x dt`` -- at 130 kt with a 2 s offset that is ~130 m, comparable to
the touchdown distance the system exists to estimate (design "Separate
timestamps preserved (no naive merge)"). This module therefore:

* operates on the **separate** ``position_times`` and ``velocity_times`` arrays
  and never merges them into a single sample time (Req 8.3, 10.1; Property 4);
* aligns position and velocity to a common query time by **interpolation**,
  never by pairing nearest samples;
* queries position at an arbitrary time by **dead-reckoning** -- advancing a
  bracketing position fix along the interpolated track by ``speed x dt`` using
  :meth:`pyproj.Geod.fwd` on the WGS-84 ellipsoid, NOT by linear lat/lon
  interpolation (Req 10.2; Property 3). For a straight, constant-velocity
  trajectory this is exact, so timestamp-misalignment error stays well under
  the 30 ft / 9.14 m bound at 120-150 kt (Req 10.3; Property 3);
* falls back to linear positional interpolation between the two nearest valid
  position messages when velocity is unavailable where kinematic interpolation
  needs it, flagging the query degraded with
  :attr:`~tdz.models.FailureReason.DEGRADED_INTERPOLATION` (Req 10.4). This is a
  per-sample degradation surfaced as a flag, not a raised exception / flight
  rejection;
* emits an explicit time-delta channel (:func:`compute_time_deltas`) so learned
  models see irregular sample spacing (maps to
  :attr:`tdz.models.FlightRecord.time_deltas`);
* supports the two configured strategies (:class:`tdz.config.schema.TimebaseConfig`):
  ``common_grid`` resamples onto a fixed grid at ``grid_interval_s``
  (:func:`resample_to_grid`), and ``continuous_time`` keeps native timestamps
  and exposes query-at-time functions (:class:`ContinuousTimebase`).

Units convention
----------------
SI internally: meters, meters/second, seconds, radians. Groundspeed is ingested
in knots and converted with the documented constant :data:`KNOTS_TO_MPS`
(1 kt = 1852/3600 m/s); track is degrees true. Latitude/longitude stay in decimal
degrees (the geodetic coordinate, not a "unit" to convert). No conversion to
feet happens here -- that is the output boundary (Task 20).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

import numpy as np
from pyproj import Geod

from tdz.models import FailureReason

__all__ = [
    "KNOTS_TO_MPS",
    "MAX_TIMESTAMP_MISALIGNMENT_ERROR_M",
    "PositionQuery",
    "ResampleResult",
    "ContinuousTimebase",
    "compute_time_deltas",
    "interpolate_groundspeed_at",
    "interpolate_track_deg",
    "interpolate_velocity_at",
    "interpolate_position_at",
    "monotone_interpolate",
    "resample_to_grid",
]

#: Exact knots -> meters/second conversion (1 international knot = 1852 m / 3600 s).
KNOTS_TO_MPS: Final[float] = 1852.0 / 3600.0

#: Documented accuracy bound for timestamp-misalignment position error
#: (30 ft = 9.14 m), Req 10.3 / Property 3. Informational; enforced by tests.
MAX_TIMESTAMP_MISALIGNMENT_ERROR_M: Final[float] = 9.14

# WGS-84 ellipsoid; shared and thread-safe for forward (dead-reckoning) geodesy.
_GEOD: Final[Geod] = Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# Result value objects (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionQuery:
    """Result of a position query at an arbitrary time.

    Attributes
    ----------
    lat, lon:
        Interpolated latitude/longitude in decimal degrees.
    degraded:
        ``True`` when kinematic interpolation could not be applied because
        velocity was unavailable/null where it was needed, and the query fell
        back to linear positional interpolation (Req 10.4).
    reason:
        :attr:`FailureReason.DEGRADED_INTERPOLATION` when ``degraded`` is set,
        else ``None``. A configured ``method="linear"`` query is *not* degraded
        (it is an intentional choice, not a fallback).
    """

    lat: float
    lon: float
    degraded: bool
    reason: Optional[FailureReason]


@dataclass(frozen=True)
class ResampleResult:
    """Arrays resampled onto a fixed common grid plus the time-delta channel.

    All arrays share the grid length. ``time_deltas`` and ``degraded_mask`` map
    to the equally-named per-sample channels consumed downstream.
    """

    times: np.ndarray              # Grid times (epoch seconds)
    latitudes: np.ndarray          # Decimal degrees
    longitudes: np.ndarray         # Decimal degrees
    geometric_altitudes: np.ndarray  # Meters (HAE); NaN where unavailable
    groundspeeds_kt: np.ndarray    # Knots
    tracks_deg: np.ndarray         # Degrees true
    time_deltas: np.ndarray        # Seconds since previous grid sample
    degraded_mask: np.ndarray      # Bool: position query fell back to linear


# ---------------------------------------------------------------------------
# Time-delta channel
# ---------------------------------------------------------------------------


def compute_time_deltas(times: np.ndarray) -> np.ndarray:
    """Inter-sample time gaps for a (sorted) timestamp array, in seconds.

    Returns an array of the same length as ``times`` where element ``i`` is
    ``times[i] - times[i-1]`` and the first element is ``0.0`` (no preceding
    sample). This explicit irregular-spacing channel lets learned models see the
    native cadence (design: "Both strategies emit an explicit time-delta
    channel"). It maps to :attr:`tdz.models.FlightRecord.time_deltas`.
    """
    t = np.asarray(times, dtype=float)
    deltas = np.zeros(t.shape, dtype=float)
    if t.size >= 2:
        deltas[1:] = np.diff(t)
    return deltas


# ---------------------------------------------------------------------------
# Velocity interpolation (linear groundspeed; wrap-aware track)
# ---------------------------------------------------------------------------


def _bracket(times: np.ndarray, t: float) -> tuple[int, int, float]:
    """Return (i0, i1, frac) bracketing ``t`` in sorted ``times``.

    ``frac`` is the fractional position of ``t`` in ``[times[i0], times[i1]]``.
    Queries outside the range clamp to the nearest endpoint (held value);
    duplicate-time degenerate intervals yield ``frac = 0``.
    """
    n = times.size
    if n == 1:
        return 0, 0, 0.0
    if t <= times[0]:
        return 0, 0, 0.0
    if t >= times[-1]:
        return n - 1, n - 1, 0.0
    i1 = int(np.searchsorted(times, t, side="left"))
    i0 = i1 - 1
    span = times[i1] - times[i0]
    frac = 0.0 if span <= 0.0 else float((t - times[i0]) / span)
    return i0, i1, frac


def interpolate_groundspeed_at(
    velocity_times: np.ndarray, gs_kt: np.ndarray, t: float
) -> float:
    """Linearly interpolate groundspeed (knots) at time ``t``.

    Returns ``NaN`` when the bracketing velocity samples are missing/null, which
    callers treat as "velocity unavailable" for the degraded fallback.
    """
    vt = np.asarray(velocity_times, dtype=float)
    gs = np.asarray(gs_kt, dtype=float)
    if vt.size == 0:
        return float("nan")
    i0, i1, frac = _bracket(vt, t)
    g0, g1 = gs[i0], gs[i1]
    if i0 == i1:
        return float(g0)
    if np.isnan(g0) or np.isnan(g1):
        return float("nan")
    return float(g0 + frac * (g1 - g0))


def interpolate_track_deg(
    velocity_times: np.ndarray, track_deg: np.ndarray, t: float
) -> float:
    """Interpolate track (degrees true) at ``t`` with 0/360 wrap handling.

    Interpolates along the **shortest** angular path so a track crossing the
    0/360 boundary (e.g. 359 deg -> 1 deg) does not swing the long way round.
    Returns a value in ``[0, 360)``; ``NaN`` when bracketing samples are null.
    """
    vt = np.asarray(velocity_times, dtype=float)
    tr = np.asarray(track_deg, dtype=float)
    if vt.size == 0:
        return float("nan")
    i0, i1, frac = _bracket(vt, t)
    a0, a1 = tr[i0], tr[i1]
    if i0 == i1:
        return float(a0 % 360.0)
    if np.isnan(a0) or np.isnan(a1):
        return float("nan")
    # Shortest signed angular difference in (-180, 180].
    diff = ((a1 - a0 + 180.0) % 360.0) - 180.0
    return float((a0 + frac * diff) % 360.0)


def interpolate_velocity_at(
    velocity_times: np.ndarray,
    gs_kt: np.ndarray,
    track_deg: np.ndarray,
    t: float,
) -> tuple[float, float]:
    """Interpolate velocity at ``t`` -> ``(groundspeed_kt, track_deg)``.

    Groundspeed is interpolated linearly; track is interpolated along the
    shortest angular path (wrap-aware). Operates only on the velocity timebase
    -- it never consults position timestamps (Property 4). Either component is
    ``NaN`` when its bracketing velocity samples are unavailable.
    """
    return (
        interpolate_groundspeed_at(velocity_times, gs_kt, t),
        interpolate_track_deg(velocity_times, track_deg, t),
    )


# ---------------------------------------------------------------------------
# Position interpolation: kinematic dead-reckoning + linear fallback
# ---------------------------------------------------------------------------


def _nearest_index(times: np.ndarray, t: float) -> int:
    """Index of the position sample nearest ``t`` (ties -> earlier sample)."""
    return int(np.argmin(np.abs(times - t)))


def _two_nearest_valid(
    position_times: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    t: float,
) -> tuple[int, int]:
    """Indices of the two nearest position samples with valid (non-NaN) coords.

    Prefers a pair that brackets ``t``; otherwise the two closest valid samples.
    """
    valid = np.where(~(np.isnan(lats) | np.isnan(lons)))[0]
    if valid.size == 0:
        raise ValueError("no valid position samples for fallback interpolation")
    if valid.size == 1:
        return int(valid[0]), int(valid[0])
    vt = position_times[valid]
    j = int(np.searchsorted(vt, t, side="left"))
    if j <= 0:
        return int(valid[0]), int(valid[1])
    if j >= valid.size:
        return int(valid[-2]), int(valid[-1])
    return int(valid[j - 1]), int(valid[j])


def _linear_position(
    position_times: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    t: float,
) -> tuple[float, float]:
    """Linear lat/lon interpolation between the two nearest valid samples."""
    i0, i1 = _two_nearest_valid(position_times, lats, lons, t)
    if i0 == i1:
        return float(lats[i0]), float(lons[i0])
    span = position_times[i1] - position_times[i0]
    frac = 0.0 if span == 0.0 else float((t - position_times[i0]) / span)
    lat = float(lats[i0] + frac * (lats[i1] - lats[i0]))
    lon = float(lons[i0] + frac * (lons[i1] - lons[i0]))
    return lat, lon


def _velocity_available_at(
    velocity_times: np.ndarray,
    gs_kt: np.ndarray,
    track_deg: np.ndarray,
    t: float,
) -> bool:
    """Whether a non-NaN velocity can be interpolated at ``t``."""
    gs = interpolate_groundspeed_at(velocity_times, gs_kt, t)
    tr = interpolate_track_deg(velocity_times, track_deg, t)
    return not (np.isnan(gs) or np.isnan(tr))


def interpolate_position_at(
    position_times: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    velocity_times: np.ndarray,
    gs_kt: np.ndarray,
    track_deg: np.ndarray,
    t: float,
    *,
    method: str = "kinematic",
) -> PositionQuery:
    """Query position at arbitrary time ``t`` (default: kinematic dead-reckoning).

    Kinematic mode advances the **nearest** position fix to ``t`` along the
    interpolated track by ``speed x dt`` using :meth:`pyproj.Geod.fwd`. Velocity
    and track are evaluated on the velocity timebase at the interval midpoint
    (exact for constant velocity, second-order accurate otherwise), so position
    and velocity are aligned by interpolation -- never by pairing nearest
    samples (Req 10.2). The separate position/velocity time arrays are never
    merged (Req 10.1; Property 4).

    Parameters
    ----------
    position_times, lats, lons:
        The position timebase and coordinates (decimal degrees).
    velocity_times, gs_kt, track_deg:
        The velocity timebase, groundspeed (knots) and track (degrees true).
    t:
        Query time (epoch seconds).
    method:
        ``"kinematic"`` (default) dead-reckons with velocity; ``"linear"``
        intentionally interpolates lat/lon linearly (not flagged degraded).

    Returns
    -------
    PositionQuery
        ``(lat, lon, degraded, reason)``. ``degraded`` is set with
        :attr:`FailureReason.DEGRADED_INTERPOLATION` only when kinematic
        interpolation was requested but velocity was unavailable and the query
        fell back to linear positional interpolation (Req 10.4).
    """
    pt = np.asarray(position_times, dtype=float)
    la = np.asarray(lats, dtype=float)
    lo = np.asarray(lons, dtype=float)
    if pt.size == 0:
        raise ValueError("position arrays are empty")

    if method == "linear":
        lat, lon = _linear_position(pt, la, lo, t)
        return PositionQuery(lat=lat, lon=lon, degraded=False, reason=None)

    if method != "kinematic":
        raise ValueError(f"unknown interpolation method: {method!r}")

    base = _nearest_index(pt, t)
    base_t = float(pt[base])
    dt = t - base_t

    # No advance needed (query coincides with a sample): return it directly.
    if dt == 0.0 and not (np.isnan(la[base]) or np.isnan(lo[base])):
        return PositionQuery(
            lat=float(la[base]), lon=float(lo[base]), degraded=False, reason=None
        )

    # Velocity is sampled at the midpoint of the dead-reckoning interval.
    midpoint = base_t + dt / 2.0
    vt = np.asarray(velocity_times, dtype=float)
    gs = np.asarray(gs_kt, dtype=float)
    tr = np.asarray(track_deg, dtype=float)

    base_valid = not (np.isnan(la[base]) or np.isnan(lo[base]))
    if base_valid and _velocity_available_at(vt, gs, tr, midpoint):
        speed_mps = interpolate_groundspeed_at(vt, gs, midpoint) * KNOTS_TO_MPS
        track = interpolate_track_deg(vt, tr, midpoint)
        distance_m = speed_mps * dt
        azimuth_deg = track
        # pyproj advances along +azimuth; a negative dt means travel backward,
        # equivalently a positive distance along the reciprocal azimuth.
        if distance_m < 0.0:
            distance_m = -distance_m
            azimuth_deg = (azimuth_deg + 180.0) % 360.0
        lon2, lat2, _back = _GEOD.fwd(
            float(lo[base]), float(la[base]), azimuth_deg, distance_m
        )
        return PositionQuery(lat=float(lat2), lon=float(lon2), degraded=False, reason=None)

    # Velocity unavailable where kinematic interpolation needs it (Req 10.4):
    # flag degraded and fall back to linear positional interpolation.
    lat, lon = _linear_position(pt, la, lo, t)
    return PositionQuery(
        lat=lat, lon=lon, degraded=True, reason=FailureReason.DEGRADED_INTERPOLATION
    )


# ---------------------------------------------------------------------------
# Shape-aware (monotone) altitude interpolation -- PCHIP, numpy-only
# ---------------------------------------------------------------------------


def _pchip_edge_slope(h0: float, h1: float, m0: float, m1: float) -> float:
    """Shape-preserving one-sided endpoint slope (Fritsch-Carlson / PCHIP).

    Non-centered three-point estimate, limited so the endpoint never introduces
    a local extremum (which would overshoot a monotone input).
    """
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if np.sign(d) != np.sign(m0):
        d = 0.0
    elif (np.sign(m0) != np.sign(m1)) and (abs(d) > 3.0 * abs(m0)):
        d = 3.0 * m0
    return float(d)


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Monotone (Fritsch-Carlson) cubic Hermite slopes at the data nodes.

    Where consecutive secants change sign (a local extremum) the node slope is
    forced to zero; equal-sign secants use the weighted-harmonic mean. This
    guarantees no overshoot beyond the data, so a monotone input stays monotone.
    """
    n = x.size
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n, dtype=float)

    if n == 2:
        d[:] = delta[0]
        return d

    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] <= 0.0:
            d[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    d[0] = _pchip_edge_slope(h[0], h[1], delta[0], delta[1])
    d[-1] = _pchip_edge_slope(h[-1], h[-2], delta[-1], delta[-2])
    return d


def monotone_interpolate(
    x: np.ndarray, y: np.ndarray, x_query: np.ndarray
) -> np.ndarray:
    """Shape-aware monotone cubic (PCHIP) interpolation -- no overshoot.

    Used for altitude resampling (design: "interpolate altitude with shape-aware
    (monotone spline) interpolator"): a monotonic input yields a monotonic
    output with no spurious over/undershoot, unlike a natural cubic spline.
    Implemented in pure numpy (Fritsch-Carlson) to avoid a SciPy dependency.

    NaN nodes are dropped before fitting. Queries outside ``[x[0], x[-1]]`` are
    clamped to the endpoint (held value). Falls back to constant (1 node) or
    linear (NaN-only / 1 valid node) behavior gracefully.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.atleast_1d(np.asarray(x_query, dtype=float))

    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return np.full(xq.shape, np.nan)
    if x.size == 1:
        return np.full(xq.shape, float(y[0]))

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    d = _pchip_slopes(x, y)
    h = np.diff(x)

    xc = np.clip(xq, x[0], x[-1])
    idx = np.searchsorted(x, xc, side="right") - 1
    idx = np.clip(idx, 0, x.size - 2)

    # Cubic Hermite basis on each local interval.
    x0 = x[idx]
    hh = h[idx]
    s = (xc - x0) / hh
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    out = (
        h00 * y[idx]
        + h10 * hh * d[idx]
        + h01 * y[idx + 1]
        + h11 * hh * d[idx + 1]
    )
    return out


# ---------------------------------------------------------------------------
# Strategy 1: common-grid resampling
# ---------------------------------------------------------------------------


def _build_grid(
    grid_interval_s: float,
    t_start: Optional[float],
    t_end: Optional[float],
    position_times: np.ndarray,
    velocity_times: np.ndarray,
) -> np.ndarray:
    """Construct the fixed common grid from start/end and interval."""
    if grid_interval_s <= 0.0:
        raise ValueError("grid_interval_s must be positive")
    if t_start is None:
        t_start = float(min(position_times[0], velocity_times[0]))
    if t_end is None:
        t_end = float(max(position_times[-1], velocity_times[-1]))
    if t_end < t_start:
        raise ValueError("t_end must be >= t_start")
    n = int(np.floor((t_end - t_start) / grid_interval_s + 1e-9)) + 1
    return t_start + grid_interval_s * np.arange(n, dtype=float)


def resample_to_grid(
    position_times: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    geometric_altitudes: np.ndarray,
    velocity_times: np.ndarray,
    gs_kt: np.ndarray,
    track_deg: np.ndarray,
    *,
    grid_interval_s: float,
    method: str = "kinematic",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> ResampleResult:
    """Resample asynchronous samples onto a fixed common grid (``common_grid``).

    Position is resampled by kinematic dead-reckoning (or linear when
    ``method="linear"``), velocity by wrap-aware interpolation, and altitude by
    the shape-aware monotone interpolator (:func:`monotone_interpolate`). The
    distinct position and velocity timebases drive their respective channels
    independently -- they are never merged (Req 8.3, 10.1). A per-grid-point
    ``degraded_mask`` records where kinematic position interpolation fell back to
    linear (Req 10.4). ``time_deltas`` is the explicit spacing channel for the
    emitted grid.

    ``interpolation_method`` from :class:`~tdz.config.schema.TimebaseConfig`
    selects ``method``; ``grid_interval_s`` sets the spacing.
    """
    pt = np.asarray(position_times, dtype=float)
    vt = np.asarray(velocity_times, dtype=float)
    if pt.size == 0 or vt.size == 0:
        raise ValueError("position and velocity timebases must be non-empty")

    la = np.asarray(lats, dtype=float)
    lo = np.asarray(lons, dtype=float)
    alt = np.asarray(geometric_altitudes, dtype=float)
    gs = np.asarray(gs_kt, dtype=float)
    tr = np.asarray(track_deg, dtype=float)

    grid = _build_grid(grid_interval_s, t_start, t_end, pt, vt)

    out_lat = np.empty(grid.shape, dtype=float)
    out_lon = np.empty(grid.shape, dtype=float)
    out_gs = np.empty(grid.shape, dtype=float)
    out_tr = np.empty(grid.shape, dtype=float)
    degraded = np.zeros(grid.shape, dtype=bool)

    for i, t in enumerate(grid):
        pos = interpolate_position_at(pt, la, lo, vt, gs, tr, float(t), method=method)
        out_lat[i] = pos.lat
        out_lon[i] = pos.lon
        degraded[i] = pos.degraded
        g, a = interpolate_velocity_at(vt, gs, tr, float(t))
        out_gs[i] = g
        out_tr[i] = a

    out_alt = monotone_interpolate(pt, alt, grid)

    return ResampleResult(
        times=grid,
        latitudes=out_lat,
        longitudes=out_lon,
        geometric_altitudes=out_alt,
        groundspeeds_kt=out_gs,
        tracks_deg=out_tr,
        time_deltas=compute_time_deltas(grid),
        degraded_mask=degraded,
    )


# ---------------------------------------------------------------------------
# Strategy 2: continuous-time consumption (no resampling)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuousTimebase:
    """Keep native timestamps; expose query-at-time functions (``continuous_time``).

    No resampling occurs: position and velocity stay on their **distinct**
    native timebases (Req 10.1) and continuous-time estimators consume each at
    its own time. The per-channel time-delta arrays expose the native irregular
    spacing. Query methods delegate to the same kinematic/linear interpolation
    used by the grid strategy, so timestamp preservation and the dead-reckoning
    accuracy bound hold identically.
    """

    position_times: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    velocity_times: np.ndarray
    groundspeeds_kt: np.ndarray
    tracks_deg: np.ndarray

    @property
    def position_time_deltas(self) -> np.ndarray:
        """Native inter-sample gaps on the position timebase (seconds)."""
        return compute_time_deltas(self.position_times)

    @property
    def velocity_time_deltas(self) -> np.ndarray:
        """Native inter-sample gaps on the velocity timebase (seconds)."""
        return compute_time_deltas(self.velocity_times)

    def position_at(self, t: float, *, method: str = "kinematic") -> PositionQuery:
        """Kinematic (default) position query at native-time ``t``."""
        return interpolate_position_at(
            self.position_times,
            self.latitudes,
            self.longitudes,
            self.velocity_times,
            self.groundspeeds_kt,
            self.tracks_deg,
            t,
            method=method,
        )

    def velocity_at(self, t: float) -> tuple[float, float]:
        """Velocity query at native-time ``t`` -> ``(groundspeed_kt, track_deg)``."""
        return interpolate_velocity_at(
            self.velocity_times, self.groundspeeds_kt, self.tracks_deg, t
        )
