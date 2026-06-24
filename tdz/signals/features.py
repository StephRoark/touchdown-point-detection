"""Feature-channel construction for the learned estimators (Task 11.3).

Builds the per-sample feature channels the learned models (Tasks 15/16) consume
and populates the derived-signal slots on :class:`~tdz.models.FlightRecord`.

Two timebases, kept separate (no naive merge)
---------------------------------------------
ADS-B position and velocity messages have **distinct** timebases for async
sources (Req 8.3, 10.1), so the feature channels live on whichever timebase is
natural for each quantity and are never merged into one sample time:

* **Position timebase** (``flight.position_times``):
  ``distance_to_threshold_m`` (along-runway distance of each position sample via
  the runway-centerline projection), ``lateral_offset_m``, and
  ``height_above_runway_m`` (geometric altitude minus the geoid-corrected runway
  HAE elevation, when geometric altitude is available).
* **Velocity timebase** (``flight.velocity_times``):
  ``groundspeed_mps`` and the smoothed ``deceleration_mps2`` / ``jerk_mps3`` /
  ``derivative_uncertainty`` channels (from :mod:`tdz.signals.derivatives`).

The time-delta channel (:func:`~tdz.timebase.interpolation.compute_time_deltas`)
is exposed for **both** timebases; the slot
:attr:`FlightRecord.time_deltas` is populated from the **velocity** timebase so
it is co-located with the derivative channels that the sequence model consumes
alongside it.

Units convention
----------------
SI throughout: distances/heights in meters, groundspeed m/s, deceleration
m/s^2, jerk m/s^3, time deltas seconds. Groundspeed knots -> m/s via
:data:`~tdz.timebase.interpolation.KNOTS_TO_MPS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tdz.geo.datum import resolve_threshold_elevation_hae
from tdz.geo.projection import RunwayProjector
from tdz.models import FlightRecord
from tdz.signals.derivatives import DerivativeResult, smoothed_derivatives
from tdz.signals.segmented import fit_segmented_groundspeed
from tdz.timebase.interpolation import KNOTS_TO_MPS, compute_time_deltas

__all__ = [
    "FeatureChannels",
    "build_feature_channels",
    "populate_flight_record",
]


@dataclass(frozen=True)
class FeatureChannels:
    """Per-sample feature channels for the learned estimators (SI units).

    Position-timebase and velocity-timebase channels are held in separate arrays
    of (generally) different lengths; each channel's docstring names its
    timebase so downstream code never merges them.
    """

    # Timebases (epoch seconds)
    position_times: np.ndarray
    velocity_times: np.ndarray

    # Velocity-timebase channels
    groundspeed_mps: np.ndarray
    deceleration_mps2: np.ndarray
    jerk_mps3: np.ndarray
    derivative_uncertainty: np.ndarray
    velocity_time_deltas: np.ndarray

    # Position-timebase channels
    distance_to_threshold_m: np.ndarray
    lateral_offset_m: np.ndarray
    height_above_runway_m: np.ndarray
    position_time_deltas: np.ndarray

    # Diagnostics
    smoothing_method: str
    derivative_window_samples: int
    derivative_reliable: bool
    segmented_breakpoint_time: Optional[float]


def _distance_and_lateral(
    flight: FlightRecord,
) -> tuple[np.ndarray, np.ndarray]:
    """Project each position sample onto the runway centerline (position timebase).

    ``distance_to_threshold_m`` is the signed along-runway distance from the
    landing threshold (positive past the threshold in the landing direction);
    on an approach toward the threshold it decreases monotonically toward ~0.
    """
    projector = RunwayProjector(flight.runway)
    lats = np.asarray(flight.latitudes, dtype=float)
    lons = np.asarray(flight.longitudes, dtype=float)
    distance = np.full(lats.shape, np.nan)
    lateral = np.full(lats.shape, np.nan)
    for i in range(lats.size):
        if np.isnan(lats[i]) or np.isnan(lons[i]):
            continue
        projected = projector.project(float(lats[i]), float(lons[i]))
        distance[i] = projected.along_runway_distance_m
        lateral[i] = projected.lateral_offset_m
    return distance, lateral


def _height_above_runway(flight: FlightRecord) -> np.ndarray:
    """Geometric altitude (HAE) minus geoid-corrected runway elevation (meters).

    Returns an all-NaN array when geometric altitude is unavailable or the
    runway datum cannot be resolved (height-above-runway is simply omitted then,
    consistent with sources that lack true geometric altitude).
    """
    alt = getattr(flight, "geometric_altitudes", None)
    if alt is None:
        return np.array([], dtype=float)
    alt = np.asarray(alt, dtype=float)
    if alt.size == 0 or np.all(np.isnan(alt)):
        return np.full(alt.shape, np.nan)
    try:
        runway_hae = resolve_threshold_elevation_hae(flight.runway)
    except Exception:
        return np.full(alt.shape, np.nan)
    return alt - runway_hae


def _resolve_breakpoint(
    flight: FlightRecord, breakpoint_time: Optional[float]
) -> Optional[float]:
    """Use the supplied breakpoint, else fit one from raw groundspeed (best effort)."""
    if breakpoint_time is not None:
        return breakpoint_time
    try:
        fit = fit_segmented_groundspeed(
            flight.velocity_times, flight.groundspeeds, n_segments=2
        )
        return fit.breakpoint_time
    except (ValueError, np.linalg.LinAlgError):
        return None


def build_feature_channels(
    flight: FlightRecord,
    config: object,
    *,
    breakpoint_time: Optional[float] = None,
) -> FeatureChannels:
    """Build all learned-estimator feature channels for ``flight``.

    Parameters
    ----------
    flight:
        The aligned per-flight record (async timebases preserved).
    config:
        A :class:`~tdz.config.schema.SignalsConfig`-like object (smoothing
        method/window and GP hyperparameters) passed through to
        :func:`~tdz.signals.derivatives.smoothed_derivatives`.
    breakpoint_time:
        Optional regime-transition time to drive piecewise derivative smoothing.
        When ``None``, a 2-segment fit on the raw groundspeed supplies one (and
        the channels are smoothed piecewise about it); if that fit cannot be
        formed, smoothing falls back to non-piecewise.

    Returns
    -------
    FeatureChannels
        Position- and velocity-timebase channels plus diagnostics.
    """
    velocity_times = np.asarray(flight.velocity_times, dtype=float)
    position_times = np.asarray(flight.position_times, dtype=float)
    groundspeed_mps = np.asarray(flight.groundspeeds, dtype=float) * KNOTS_TO_MPS

    bp = _resolve_breakpoint(flight, breakpoint_time)
    derivatives: DerivativeResult = smoothed_derivatives(
        velocity_times, flight.groundspeeds, config, breakpoint_time=bp
    )

    distance, lateral = _distance_and_lateral(flight)
    height = _height_above_runway(flight)

    return FeatureChannels(
        position_times=position_times,
        velocity_times=velocity_times,
        groundspeed_mps=groundspeed_mps,
        deceleration_mps2=derivatives.deceleration_mps2,
        jerk_mps3=derivatives.jerk_mps3,
        derivative_uncertainty=derivatives.derivative_uncertainty,
        velocity_time_deltas=compute_time_deltas(velocity_times),
        distance_to_threshold_m=distance,
        lateral_offset_m=lateral,
        height_above_runway_m=height,
        position_time_deltas=compute_time_deltas(position_times),
        smoothing_method=derivatives.method,
        derivative_window_samples=derivatives.window_samples,
        derivative_reliable=derivatives.reliable,
        segmented_breakpoint_time=bp,
    )


def populate_flight_record(
    flight: FlightRecord,
    config: object,
    *,
    breakpoint_time: Optional[float] = None,
) -> FeatureChannels:
    """Populate ``flight``'s derived-signal slots in place; return the channels.

    Sets, on the velocity timebase, ``smoothed_deceleration``, ``smoothed_jerk``,
    ``derivative_uncertainties`` and ``time_deltas`` (co-located with the
    derivative channels); and, on the position timebase,
    ``distance_to_threshold``. The returned :class:`FeatureChannels` carries the
    full channel set (including the position-timebase channels that have no
    dedicated slot) plus diagnostics.
    """
    channels = build_feature_channels(flight, config, breakpoint_time=breakpoint_time)

    flight.smoothed_deceleration = channels.deceleration_mps2
    flight.smoothed_jerk = channels.jerk_mps3
    flight.derivative_uncertainties = channels.derivative_uncertainty
    flight.distance_to_threshold = channels.distance_to_threshold_m
    # time_deltas is the velocity-timebase spacing, co-located with the
    # derivative channels the sequence model consumes alongside it.
    flight.time_deltas = channels.velocity_time_deltas

    return channels
