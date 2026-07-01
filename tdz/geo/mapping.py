"""Time -> position/speed mapping at the fused touchdown time (Task 20).

This module is the geometric half of the output boundary. Given a fused
touchdown *time* ``t_td`` (and its 1-sigma ``sigma_t``), it produces the
touchdown *geometry* -- along-runway distance and lateral offset -- and the
touchdown *groundspeed*, all in SI, by:

* **interpolating the horizontal trajectory** at ``t_td`` with the timebase's
  kinematic dead-reckoning (never pairing the nearest sample) and projecting the
  interpolated point onto the runway centerline (Req 2.1, 2.2);
* **applying the pitch-resolved lever-arm correction** to the along-runway
  distance -- subtracting ``X·cos θ + V·sin θ`` so the reported distance is at
  the main-gear contact point, not the antenna (Req 2.3 / Task 6 convention).
  The vertical offset ``V`` is the correction applied to the altitude crossing
  upstream (Task 6); it is recorded on the output but does not enter the
  horizontal distance again here;
* **interpolating the groundspeed** at ``t_td`` from the kinematic velocity
  interpolant (Req 3.2, *not* the nearest ADS-B sample) and **propagating**
  ``sigma_t`` into a groundspeed 1-sigma via the local slope of the interpolated
  speed profile, ``sigma_v = |dv/dt| · sigma_t`` (Req 3.3);
* **evaluating the position gates** (out-of-bounds along-runway, suspected wrong
  runway) as non-fatal flags (Req 2.4, 2.5 / Task 7);
* **flagging the speed low-confidence** when the interpolated speed is outside
  the plausible band or no velocity sample exists within a configured window of
  ``t_td`` (Req 3.4).

Everything here stays in SI (meters, m/s, seconds, radians). The SI->feet/knots
conversion, rounding, CI assembly, and the final :class:`~tdz.models.TouchdownResult`
are the responsibility of :mod:`tdz.assemble`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from tdz.geo.gates import PositionGateResult, evaluate_position_gates
from tdz.geo.lever_arm import LeverArmCorrection
from tdz.geo.projection import ProjectedPosition, RunwayProjector
from tdz.models import FailureReason, FlightRecord
from tdz.timebase.interpolation import (
    KNOTS_TO_MPS,
    interpolate_groundspeed_at,
    interpolate_position_at,
)

__all__ = [
    "TouchdownMapping",
    "groundspeed_slope_mps2",
    "velocity_samples_within",
    "map_touchdown",
]


@dataclass(frozen=True)
class TouchdownMapping:
    """SI geometry + speed at the fused touchdown time (Task 20).

    Immutable value object. All distances are meters, speeds m/s. The
    presentation-unit conversion happens in :mod:`tdz.assemble`.

    Attributes
    ----------
    along_runway_distance_m:
        Lever-arm-corrected along-runway distance from the threshold (meters,
        signed; positive = past the threshold in the landing direction).
    lateral_offset_m:
        Signed lateral offset from the centerline (meters).
    antenna_along_runway_distance_m:
        The along-runway distance *before* the lever-arm correction (the raw
        antenna projection), retained for diagnostics.
    groundspeed_mps:
        Interpolated groundspeed at ``t_td`` (m/s), from the kinematic velocity
        interpolant (Req 3.2). ``NaN`` when velocity is entirely unavailable.
    groundspeed_sigma_mps:
        Propagated groundspeed 1-sigma, ``|dv/dt| · sigma_t`` (m/s; Req 3.3).
    gates:
        The non-fatal position-gate result (out-of-bounds / wrong-runway).
    lever_arm:
        The :class:`LeverArmCorrection` applied.
    degraded_interpolation:
        ``True`` when the kinematic position query fell back to linear
        interpolation because velocity was unavailable (Req 10.4).
    speed_plausible:
        ``True`` when the interpolated speed is finite and within the plausible
        band ``[speed_min_mps, speed_max_mps]``.
    velocity_samples_present:
        ``True`` when at least one velocity sample lies within the configured
        window of ``t_td`` (Req 3.4).
    speed_low_confidence:
        ``True`` when the speed estimate should be flagged low-confidence
        (implausible speed or missing nearby velocity samples; Req 3.4).
    speed_reason_code:
        The reason code when ``speed_low_confidence`` is set, else ``None``.
    diagnostics:
        Free-form diagnostics for traceability.
    """

    along_runway_distance_m: float
    lateral_offset_m: float
    antenna_along_runway_distance_m: float
    groundspeed_mps: float
    groundspeed_sigma_mps: float
    gates: PositionGateResult
    lever_arm: LeverArmCorrection
    degraded_interpolation: bool
    speed_plausible: bool
    velocity_samples_present: bool
    speed_low_confidence: bool
    speed_reason_code: Optional[FailureReason]
    diagnostics: dict


def groundspeed_slope_mps2(
    velocity_times: np.ndarray, groundspeeds_kt: np.ndarray, t: float
) -> float:
    """Local slope ``dv/dt`` (m/s^2) of the interpolated groundspeed at ``t``.

    The timebase interpolates groundspeed **linearly** between bracketing
    velocity samples, so the exact derivative of the interpolant on the bracket
    containing ``t`` is the segment slope ``(v1 - v0) / (t1 - t0)``. This is used
    (Req 3.3) to propagate the touchdown-time uncertainty into a groundspeed
    uncertainty via the delta method, ``sigma_v = |dv/dt| · sigma_t``.

    In the held (clamped) region outside the sample range, and where fewer than
    two finite bracketing samples exist, the interpolant is flat, so the slope is
    ``0.0`` (timing uncertainty then induces no speed uncertainty).
    """
    vt = np.asarray(velocity_times, dtype=float)
    gs = np.asarray(groundspeeds_kt, dtype=float)
    n = vt.size
    if n < 2:
        return 0.0
    # Outside the sampled range the interpolant holds the endpoint -> flat.
    if t <= vt[0] or t >= vt[-1]:
        return 0.0
    i1 = int(np.searchsorted(vt, t, side="left"))
    i0 = i1 - 1
    if i0 < 0 or i1 >= n:
        return 0.0
    span = float(vt[i1] - vt[i0])
    if span <= 0.0:
        return 0.0
    g0 = float(gs[i0])
    g1 = float(gs[i1])
    if math.isnan(g0) or math.isnan(g1):
        return 0.0
    # Convert the knots-per-second slope to m/s^2.
    return (g1 - g0) * KNOTS_TO_MPS / span


def velocity_samples_within(
    velocity_times: np.ndarray, t: float, window_s: float
) -> bool:
    """Whether any velocity sample lies within ``window_s`` of ``t`` (Req 3.4)."""
    vt = np.asarray(velocity_times, dtype=float)
    if vt.size == 0:
        return False
    return bool(np.any(np.abs(vt - float(t)) <= float(window_s)))


def map_touchdown(
    flight: FlightRecord,
    t_td: float,
    sigma_t: float,
    *,
    lever_arm: LeverArmCorrection,
    speed_min_mps: float,
    speed_max_mps: float,
    velocity_gap_max_s: float,
    validation_config=None,
    wrong_runway_margin_m: Optional[float] = None,
    interpolation_method: str = "kinematic",
) -> TouchdownMapping:
    """Map a fused ``t_td`` (+/- ``sigma_t``) to SI touchdown geometry and speed.

    Parameters
    ----------
    flight:
        The flight record (position/velocity timebases, runway geometry).
    t_td:
        Fused touchdown time (epoch seconds).
    sigma_t:
        Fused 1-sigma touchdown-time uncertainty (seconds); propagated into the
        groundspeed sigma.
    lever_arm:
        The resolved :class:`LeverArmCorrection` (Task 6). Its along-runway shift
        is subtracted from the antenna-projected distance (main-gear contact).
    speed_min_mps, speed_max_mps:
        The plausible touchdown-speed band (m/s), converted from the config
        knots band by the caller (Req 3.1).
    velocity_gap_max_s:
        Flag the speed low-confidence when no velocity sample lies within this
        window of ``t_td`` (Req 3.4).
    validation_config, wrong_runway_margin_m:
        Passed through to :func:`~tdz.geo.gates.evaluate_position_gates` to size
        the wrong-runway lateral threshold (one of them must be supplied).
    interpolation_method:
        ``"kinematic"`` (default) or ``"linear"`` position query method.

    Returns
    -------
    TouchdownMapping
        The SI geometry, groundspeed + propagated sigma, gate flags, and speed
        confidence flag/reason.
    """
    # --- Horizontal position at t_td (kinematic dead-reckoning) -----------
    query = interpolate_position_at(
        flight.position_times,
        flight.latitudes,
        flight.longitudes,
        flight.velocity_times,
        flight.groundspeeds,
        flight.tracks,
        float(t_td),
        method=interpolation_method,
    )

    projector = RunwayProjector(flight.runway)
    projected = projector.project(query.lat, query.lon)
    antenna_along_m = projected.along_runway_distance_m
    # Subtract the pitch-resolved along-runway lever-arm shift so the reported
    # distance is at main-gear contact, not the antenna (Task 6 convention).
    along_m = antenna_along_m - lever_arm.along_runway_shift_m
    lateral_m = projected.lateral_offset_m

    # --- Position gates (non-fatal flags) ---------------------------------
    # Evaluate the gates on the lever-arm-corrected along-runway distance so the
    # out-of-bounds flag reflects the reported value.
    corrected_projection = ProjectedPosition(
        along_runway_distance_m=along_m, lateral_offset_m=lateral_m
    )
    gates = evaluate_position_gates(
        corrected_projection,
        flight.runway,
        validation_config=validation_config,
        wrong_runway_margin_m=wrong_runway_margin_m,
    )

    # --- Groundspeed at t_td (kinematic interpolation, Req 3.2) -----------
    gs_kt = interpolate_groundspeed_at(
        flight.velocity_times, flight.groundspeeds, float(t_td)
    )
    groundspeed_mps = gs_kt * KNOTS_TO_MPS if not math.isnan(gs_kt) else float("nan")

    # Propagate t_td uncertainty into a groundspeed sigma (Req 3.3).
    slope = groundspeed_slope_mps2(
        flight.velocity_times, flight.groundspeeds, float(t_td)
    )
    sigma_v = abs(slope) * abs(float(sigma_t)) if math.isfinite(sigma_t) else float("nan")
    if math.isnan(sigma_v):
        sigma_v = 0.0

    # --- Speed confidence (Req 3.4) ---------------------------------------
    velocity_present = velocity_samples_within(
        flight.velocity_times, float(t_td), velocity_gap_max_s
    )
    speed_plausible = (
        math.isfinite(groundspeed_mps)
        and speed_min_mps <= groundspeed_mps <= speed_max_mps
    )
    speed_low_confidence = (not speed_plausible) or (not velocity_present)
    speed_reason_code: Optional[FailureReason] = None
    if not velocity_present:
        speed_reason_code = FailureReason.DEGRADED_INTERPOLATION
    elif not speed_plausible:
        speed_reason_code = FailureReason.IMPLAUSIBLE_SPEED

    diagnostics = {
        "interpolation_degraded": query.degraded,
        "interpolation_method": interpolation_method,
        "antenna_along_runway_distance_m": antenna_along_m,
        "lever_arm_along_runway_shift_m": lever_arm.along_runway_shift_m,
        "lever_arm_altitude_target_shift_m": lever_arm.altitude_target_shift_m,
        "groundspeed_slope_mps2": slope,
        "velocity_samples_present": velocity_present,
        "speed_plausible": speed_plausible,
    }

    return TouchdownMapping(
        along_runway_distance_m=along_m,
        lateral_offset_m=lateral_m,
        antenna_along_runway_distance_m=antenna_along_m,
        groundspeed_mps=groundspeed_mps,
        groundspeed_sigma_mps=sigma_v,
        gates=gates,
        lever_arm=lever_arm,
        degraded_interpolation=bool(query.degraded),
        speed_plausible=speed_plausible,
        velocity_samples_present=velocity_present,
        speed_low_confidence=speed_low_confidence,
        speed_reason_code=speed_reason_code,
        diagnostics=diagnostics,
    )
