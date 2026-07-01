"""Data-driven confidence-interval widening factors (Task 19).

The calibrated interval half-width from :mod:`tdz.uncertainty.conformal` is the
*baseline* width for a nominally-sampled flight. Three data-quality conditions
make an individual flight's estimate less certain than that baseline, and each
multiplies the reported interval width:

* **Gap-proportional widening (Req 9.2, Property 7).** A data gap of duration
  ``G`` within +/-``gap_window_half_width_s`` of ``t_td`` inflates the interval
  by ``G / nominal_cadence_s`` (a 10 s gap at 5 s cadence doubles the width).
  Only gaps strictly exceeding ``gap_min_duration_s`` count; the factor is
  floored at ``1.0`` so a normally-sampled trajectory is never *narrowed*.
* **Missing-lever-arm widening (Req 7.5).** When a type-specific lever arm is
  absent and the class-median default was substituted, the *distance* interval
  is widened by ``missing_lever_arm_widening_factor`` to span the class range.
  This affects distance only (the substitution shifts along-runway position,
  not time).
* **Post-transition starvation widening (Req 9.6).** When the on-ground flag
  transitions but fewer than ``min_post_transition_samples`` valid position
  samples exist within ``post_transition_window_s`` after it, ground-roll
  kinematics cannot be confirmed, so both intervals are widened by
  ``starvation_widening_factor``.

Each helper returns a factor ``>= 1.0`` so they compose by multiplication.
The gap and starvation factors apply to both time and distance; the lever-arm
factor applies to distance only. Keeping the factors as pure functions of the
:class:`~tdz.models.FlightRecord` makes them directly unit- and property-
testable (Property 7).

Units: seconds throughout for gaps/windows. The factors are dimensionless.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from tdz.config.schema import UncertaintyConfig
from tdz.models import FlightRecord

__all__ = [
    "gap_widening_factor",
    "post_transition_starved",
    "starvation_widening_factor",
    "missing_lever_arm_widening_factor",
]


def _finite_sorted(times: Optional[np.ndarray]) -> np.ndarray:
    """Return the finite entries of ``times`` as a sorted float array."""
    if times is None:
        return np.empty(0, dtype=float)
    arr = np.asarray(times, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr.sort()
    return arr


def gap_widening_factor(
    flight: FlightRecord, t_td: float, config: UncertaintyConfig
) -> float:
    """Gap-proportional widening factor for the interval around ``t_td``.

    Scans the position-message timestamps for the largest consecutive spacing
    (gap) whose interval overlaps ``[t_td - H, t_td + H]`` (with ``H =
    config.gap_window_half_width_s``). A gap ``G`` strictly exceeding
    ``config.gap_min_duration_s`` contributes a factor ``G /
    config.nominal_cadence_s``; the returned value is the maximum such factor,
    floored at ``1.0`` (no gap in the window -> ``1.0``, i.e. no widening).

    This realizes Req 9.2 / Property 7: for an identical trajectory *without*
    the gap the largest in-window spacing is the nominal cadence, giving factor
    ``1.0``, so the gapped interval is at least ``G / C`` times as wide.
    """
    if not math.isfinite(t_td):
        return 1.0

    times = _finite_sorted(flight.position_times)
    if times.size < 2:
        return 1.0

    half_window = config.gap_window_half_width_s
    lo = t_td - half_window
    hi = t_td + half_window
    cadence = config.nominal_cadence_s

    factor = 1.0
    starts = times[:-1]
    ends = times[1:]
    gaps = ends - starts
    # A gap [start, end] is "within +/-H of t_td" if it overlaps [lo, hi].
    overlaps = (ends >= lo) & (starts <= hi)
    for gap, overlap in zip(gaps, overlaps):
        if not overlap:
            continue
        if gap <= config.gap_min_duration_s:
            continue
        candidate = gap / cadence
        if candidate > factor:
            factor = candidate
    return factor


def post_transition_starved(
    flight: FlightRecord, config: UncertaintyConfig
) -> bool:
    """Whether post-transition ground-roll samples are starved (Req 9.6).

    ``True`` when the on-ground flag transition time is known and fewer than
    ``config.min_post_transition_samples`` valid position samples (finite
    timestamp *and* finite latitude) fall in the window
    ``(transition, transition + config.post_transition_window_s]``.
    """
    transition = flight.on_ground_transition_time
    if transition is None or not math.isfinite(transition):
        return False

    times = np.asarray(flight.position_times, dtype=float)
    lats = np.asarray(flight.latitudes, dtype=float)
    if times.size == 0:
        # No position samples at all after a transition -> starved.
        return True

    # Align lengths defensively (position arrays are parallel by construction).
    n = min(times.size, lats.size)
    times = times[:n]
    lats = lats[:n]

    window_end = transition + config.post_transition_window_s
    in_window = (
        np.isfinite(times)
        & np.isfinite(lats)
        & (times > transition)
        & (times <= window_end)
    )
    return int(np.count_nonzero(in_window)) < config.min_post_transition_samples


def starvation_widening_factor(
    flight: FlightRecord, config: UncertaintyConfig
) -> float:
    """Widening factor from post-transition sample starvation (Req 9.6).

    Returns ``config.starvation_widening_factor`` when
    :func:`post_transition_starved` holds, else ``1.0``.
    """
    if post_transition_starved(flight, config):
        return config.starvation_widening_factor
    return 1.0


def missing_lever_arm_widening_factor(
    lever_arm_missing: bool, config: UncertaintyConfig
) -> float:
    """Distance-CI widening factor for a class-median lever-arm default (Req 7.5).

    Returns ``config.missing_lever_arm_widening_factor`` when the type-specific
    lever arm was missing (class-median substituted), else ``1.0``. Applies to
    the distance interval only.
    """
    return config.missing_lever_arm_widening_factor if lever_arm_missing else 1.0
