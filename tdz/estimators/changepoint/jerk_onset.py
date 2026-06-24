"""Jerk-onset corroborating detector (Task 13).

Detects the **onset** of smoothed groundspeed jerk -- the brake/spoiler-onset
signature of the ground-roll regime -- and reports it as a CORROBORATING-only
touchdown indicator (Req 16.3; design "Jerk-Onset Detector").

Onset, not peak (Req design Errors 9.5)
---------------------------------------
Peak braking (the jerk extremum and, even later, the deceleration peak) *lags*
touchdown by several seconds, so this detector uses the jerk **onset** -- the
moment the smoothed jerk first rises through a fraction of its peak magnitude --
not the peak itself. At ground-roll onset the deceleration becomes rapidly more
negative, so the jerk (its time derivative) dips to a negative extremum; the
onset is the leading edge of that dip. The reported time is therefore the first
sub-sample crossing of ``onset_fraction * peak_jerk`` *before* the jerk
extremum, which sits earlier than (or at) the extremum by construction.

Smoothed jerk only (Req 16.2 / 16.3)
------------------------------------
The detector consumes ``signal.jerk_mps3`` from
:func:`prepare_decel_signal`, which comes from the local-polynomial / GP
smoother -- never raw finite-difference jerk (which is noise-dominated at 4-5 s
cadence and would be unusable).

Corroborating-only role (Req 16.3)
----------------------------------
Smoothed-jerk onset is **never** a standalone primary basis for ``t_td``. This
estimator makes that role explicit: it reports ``confidence = "low-confidence"``
and sets the diagnostics flag ``"corroborating_only": True``. The fusion layer
(Task 18) enforces the not-sole-basis rule; this estimator carries the role so
it can never be mistaken for a primary anchor. The on-ground upper bound is
applied by :class:`PhysicsEstimator`.
"""

from __future__ import annotations

from typing import Final, Optional

import numpy as np

from tdz.estimators.changepoint.common import (
    CONFIDENCE_LOW,
    PhysicsEstimator,
    failed_estimate,
    localization_sigma,
    make_estimate,
    prepare_decel_signal,
)
from tdz.models import FailureReason, FlightRecord, TDEstimate

__all__ = [
    "JerkOnsetEstimator",
    "METHOD_NAME",
    "DEFAULT_ONSET_FRACTION",
    "MIN_JERK_PEAK_MPS3",
]

#: Estimator identifier.
METHOD_NAME: Final[str] = "jerk_onset"

#: Fraction of the (signed) peak jerk that defines the onset level: the onset is
#: the first crossing of ``onset_fraction * peak_jerk`` ahead of the extremum.
DEFAULT_ONSET_FRACTION: Final[float] = 0.5

#: Floor on the peak jerk magnitude (m/s^3): below this there is no discernible
#: braking transient to locate.
MIN_JERK_PEAK_MPS3: Final[float] = 1e-3


class JerkOnsetEstimator(PhysicsEstimator):
    """Smoothed-jerk-onset CORROBORATING touchdown indicator (Req 16.3).

    Parameters
    ----------
    onset_fraction:
        Fraction of the peak jerk defining the onset crossing (default
        :data:`DEFAULT_ONSET_FRACTION`).
    config:
        Optional smoothing config forwarded to :func:`prepare_decel_signal`.
    """

    method_name = METHOD_NAME

    def __init__(
        self,
        *,
        onset_fraction: float = DEFAULT_ONSET_FRACTION,
        config=None,
    ) -> None:
        self.onset_fraction = float(onset_fraction)
        self._config = config

    def _raw_estimate(self, flight: FlightRecord) -> TDEstimate:
        if self._config is not None:
            signal, reason = prepare_decel_signal(flight, self._config)
        else:
            signal, reason = prepare_decel_signal(flight)
        if signal is None:
            return failed_estimate(self.method_name, reason)

        times = signal.times
        jerk = signal.jerk_mps3
        finite = np.isfinite(jerk)
        if int(np.count_nonzero(finite)) < 3:
            return failed_estimate(self.method_name, FailureReason.INSUFFICIENT_SAMPLES)

        t = times[finite]
        j = jerk[finite]
        n = j.size

        # Braking onset drives the jerk to its negative extremum: locate the most
        # negative smoothed jerk as the "peak" of the braking transient.
        peak_idx = int(np.argmin(j))
        peak_jerk = float(j[peak_idx])
        if abs(peak_jerk) < MIN_JERK_PEAK_MPS3 or peak_idx == 0:
            return failed_estimate(
                self.method_name,
                FailureReason.NO_GROUND_ROLL_CONFIRMATION,
                diagnostics={
                    "peak_jerk_mps3": peak_jerk,
                    "cadence_s": signal.cadence_s,
                },
            )

        onset_level = self.onset_fraction * peak_jerk  # negative
        # First index (scanning toward the peak) where the jerk drops THROUGH the
        # onset level -- the leading edge of the dip, ahead of the extremum.
        t_onset, onset_index = self._onset_crossing(t, j, peak_idx, onset_level)

        # sigma_t from the jerk scatter away from the transient and the jerk
        # excursion (localization_sigma is a generic cadence*scatter/|shift|).
        residual_std = self._baseline_jerk_std(j, onset_index, peak_idx)
        sigma_t = localization_sigma(signal.cadence_s, residual_std, peak_jerk)

        diagnostics = {
            "corroborating_only": True,
            "onset_time": float(t_onset),
            "onset_index": int(onset_index),
            "peak_jerk_index": peak_idx,
            "peak_jerk_mps3": peak_jerk,
            "peak_jerk_time": float(t[peak_idx]),
            "onset_level_mps3": float(onset_level),
            "onset_fraction": self.onset_fraction,
            "onset_lead_s": float(t[peak_idx] - t_onset),
            "residual_jerk_std_mps3": float(residual_std),
            "cadence_s": signal.cadence_s,
            "derivative_reliable": bool(signal.derivative.reliable),
        }
        # Corroborating-only: report low-confidence so fusion never treats the
        # jerk onset as a standalone primary (Req 16.3).
        return make_estimate(
            t_td=float(t_onset),
            sigma_t=sigma_t,
            confidence=CONFIDENCE_LOW,
            method_name=self.method_name,
            diagnostics=diagnostics,
            reason=FailureReason.NO_GROUND_ROLL_CONFIRMATION,
        )

    def _onset_crossing(
        self, t: np.ndarray, j: np.ndarray, peak_idx: int, onset_level: float
    ) -> tuple[float, int]:
        """Sub-sample time the jerk first drops through ``onset_level`` before the peak.

        Scans backward from the extremum to the first sample still above (less
        negative than) ``onset_level`` and linearly interpolates the crossing.
        Falls back to the peak sample time if no clean crossing exists.
        """
        crossing_idx: Optional[int] = None
        for i in range(peak_idx, 0, -1):
            if j[i] <= onset_level and j[i - 1] > onset_level:
                crossing_idx = i
                break
        if crossing_idx is None:
            return float(t[peak_idx]), peak_idx

        i = crossing_idx
        denom = j[i] - j[i - 1]
        if denom == 0.0:
            return float(0.5 * (t[i - 1] + t[i])), i
        frac = float((onset_level - j[i - 1]) / denom)
        frac = min(max(frac, 0.0), 1.0)
        return float(t[i - 1] + frac * (t[i] - t[i - 1])), i

    def _baseline_jerk_std(self, j: np.ndarray, onset_index: int, peak_idx: int) -> float:
        """Std of the jerk away from the braking transient (its noise scale)."""
        mask = np.ones(j.size, dtype=bool)
        lo = max(onset_index - 1, 0)
        hi = min(peak_idx + 2, j.size)
        mask[lo:hi] = False
        baseline = j[mask]
        if baseline.size <= 1:
            return float(np.std(j)) if j.size > 1 else 0.0
        return float(np.std(baseline))
