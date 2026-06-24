"""Module 2: Timebase.

Preserve asynchronous timestamps, perform kinematic (dead-reckoning)
interpolation, and optionally resample onto a common grid.

Public API (Task 8 -- async-timestamp-preserving interpolation & resampling):

* :func:`interpolate_velocity_at` -- interpolate ``(groundspeed_kt, track_deg)``
  at a query time; groundspeed linear, track wrap-aware at 0/360 deg. Helpers
  :func:`interpolate_groundspeed_at` and :func:`interpolate_track_deg` expose the
  components.
* :func:`interpolate_position_at` -- query position at an arbitrary time by
  kinematic dead-reckoning (:meth:`pyproj.Geod.fwd`, ``speed x dt`` along the
  interpolated track), with linear fallback flagged
  :attr:`~tdz.models.FailureReason.DEGRADED_INTERPOLATION` when velocity is
  unavailable (Req 10.2-10.4; Property 3). Returns a :class:`PositionQuery`.
* :func:`resample_to_grid` -- ``common_grid`` strategy: resample onto a fixed
  grid at ``grid_interval_s`` (kinematic position, wrap-aware velocity,
  shape-aware monotone altitude), emitting a :class:`ResampleResult` with the
  time-delta channel and a per-sample degraded mask.
* :class:`ContinuousTimebase` -- ``continuous_time`` strategy: keep native
  timestamps and expose query-at-time functions plus per-channel time-deltas.
* :func:`compute_time_deltas` -- the explicit irregular-spacing channel
  (maps to :attr:`tdz.models.FlightRecord.time_deltas`).
* :func:`monotone_interpolate` -- numpy-only PCHIP (no-overshoot) interpolation.
* :data:`KNOTS_TO_MPS` -- documented knots->m/s constant (1 kt = 0.514444 m/s).
* :data:`MAX_TIMESTAMP_MISALIGNMENT_ERROR_M` -- the 9.14 m (30 ft) accuracy bound.

The separate ``position_times`` and ``velocity_times`` arrays are never merged
into a single sample time; alignment to a common query time is always done by
interpolation, never by pairing nearest samples (Req 8.3, 10.1; Property 4).
"""

from tdz.timebase.interpolation import (
    KNOTS_TO_MPS,
    MAX_TIMESTAMP_MISALIGNMENT_ERROR_M,
    ContinuousTimebase,
    PositionQuery,
    ResampleResult,
    compute_time_deltas,
    interpolate_groundspeed_at,
    interpolate_position_at,
    interpolate_track_deg,
    interpolate_velocity_at,
    monotone_interpolate,
    resample_to_grid,
)

__all__ = [
    "KNOTS_TO_MPS",
    "MAX_TIMESTAMP_MISALIGNMENT_ERROR_M",
    "ContinuousTimebase",
    "PositionQuery",
    "ResampleResult",
    "compute_time_deltas",
    "interpolate_groundspeed_at",
    "interpolate_position_at",
    "interpolate_track_deg",
    "interpolate_velocity_at",
    "monotone_interpolate",
    "resample_to_grid",
]
