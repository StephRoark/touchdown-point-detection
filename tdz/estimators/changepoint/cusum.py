"""CUSUM change-point estimator (Task 13).

A two-sided tabular CUSUM detector for a shift in the **mean of the smoothed
deceleration** of groundspeed (from :func:`prepare_decel_signal`). CUSUM is the
online-capable corroborator of the approach -> ground-roll regime change (Req
5.2; design "Change-Point Estimators Detail"): it accumulates standardized
deviations from a reference mean and alarms when the running sum crosses a
threshold, then the alarm is mapped back to the regime onset.

Statistic
---------
The signal is standardized to ``z_i = (x_i - mu0) / sigma`` where ``mu0`` is the
overall mean deceleration and ``sigma`` a shift-robust noise scale (from first
differences). Two cumulative sums accumulate excursions beyond a **reference
value** ``k`` (the slack, in sigma units -- half the smallest shift worth
detecting)::

    S_hi[i] = max(0, S_hi[i-1] + z_i - k)      # upward shift
    S_lo[i] = min(0, S_lo[i-1] + z_i + k)      # downward shift

An alarm fires at the first ``i`` with ``S_hi[i] > h`` or ``S_lo[i] < -h`` for a
**threshold** ``h`` (in sigma units). For a landing the deceleration becomes
more negative at ground-roll onset, so the downward arm (``S_lo``) normally
alarms. ``k`` (:data:`DEFAULT_REFERENCE_K`) and ``h``
(:data:`DEFAULT_THRESHOLD_H`) are exposed as constructor params.

Mapping the alarm back to the onset
-----------------------------------
A CUSUM alarm *lags* the change because the sum must build past ``h``. The change
index is recovered as the **last reset point** of the alarming arm before the
alarm (the index where its cumulative sum last sat at zero), the textbook CUSUM
change-time estimate. This still sits on the leading edge of the smoothed step;
:func:`subsample_transition_time` then refines it to the half-level crossing of
the smoothed deceleration, removing the leading-edge bias. ``sigma_t`` follows
from :func:`localization_sigma` (within-segment scatter and the deceleration
shift). The on-ground upper bound is applied by :class:`PhysicsEstimator`.
"""

from __future__ import annotations

from typing import Final, Optional

import numpy as np

from tdz.estimators.changepoint.common import (
    CONFIDENCE_NORMAL,
    PhysicsEstimator,
    failed_estimate,
    localization_sigma,
    make_estimate,
    prepare_decel_signal,
    segment_residual_std,
    subsample_transition_time,
)
from tdz.models import FailureReason, FlightRecord, TDEstimate

__all__ = [
    "CusumEstimator",
    "METHOD_NAME",
    "DEFAULT_REFERENCE_K",
    "DEFAULT_THRESHOLD_H",
    "MIN_NOISE_STD_MPS2",
]

#: Estimator identifier.
METHOD_NAME: Final[str] = "cusum"

#: Reference value ``k`` (slack), in sigma units: half the smallest standardized
#: mean shift worth detecting. The classic ``k = 0.5`` detects a 1-sigma shift.
DEFAULT_REFERENCE_K: Final[float] = 0.5

#: Decision threshold ``h``, in sigma units. ``h = 5`` is the textbook
#: low-false-alarm choice for a tabular CUSUM.
DEFAULT_THRESHOLD_H: Final[float] = 5.0

#: Floor on the noise std (m/s^2) so standardization never divides by ~0 on a
#: perfectly clean signal.
MIN_NOISE_STD_MPS2: Final[float] = 1e-3


class CusumEstimator(PhysicsEstimator):
    """Two-sided CUSUM estimate of the deceleration-regime transition (Req 5.2).

    Parameters
    ----------
    reference_k:
        CUSUM slack ``k`` in sigma units (default :data:`DEFAULT_REFERENCE_K`).
    threshold_h:
        Decision threshold ``h`` in sigma units (default
        :data:`DEFAULT_THRESHOLD_H`).
    config:
        Optional smoothing config forwarded to :func:`prepare_decel_signal`.
    """

    method_name = METHOD_NAME

    def __init__(
        self,
        *,
        reference_k: float = DEFAULT_REFERENCE_K,
        threshold_h: float = DEFAULT_THRESHOLD_H,
        config=None,
    ) -> None:
        self.reference_k = float(reference_k)
        self.threshold_h = float(threshold_h)
        self._config = config

    def _raw_estimate(self, flight: FlightRecord) -> TDEstimate:
        if self._config is not None:
            signal, reason = prepare_decel_signal(flight, self._config)
        else:
            signal, reason = prepare_decel_signal(flight)
        if signal is None:
            return failed_estimate(self.method_name, reason)

        times = signal.times
        decel = signal.deceleration_mps2
        n = decel.size

        mu0 = float(np.mean(decel))
        sigma = max(float(np.sqrt(np.var(np.diff(decel)) / 2.0)), MIN_NOISE_STD_MPS2)
        z = (decel - mu0) / sigma

        s_hi = np.zeros(n)
        s_lo = np.zeros(n)
        reset_hi = 0  # last index where S_hi was at zero
        reset_lo = 0
        alarm_idx: Optional[int] = None
        alarm_arm: Optional[str] = None
        change_index = 0
        for i in range(1, n):
            s_hi[i] = max(0.0, s_hi[i - 1] + z[i] - self.reference_k)
            s_lo[i] = min(0.0, s_lo[i - 1] + z[i] + self.reference_k)
            if s_hi[i] == 0.0:
                reset_hi = i
            if s_lo[i] == 0.0:
                reset_lo = i
            if s_lo[i] < -self.threshold_h:
                alarm_idx, alarm_arm, change_index = i, "lo", reset_lo + 1
                break
            if s_hi[i] > self.threshold_h:
                alarm_idx, alarm_arm, change_index = i, "hi", reset_hi + 1
                break

        if alarm_idx is None:
            # No regime shift crossed the threshold: the deceleration mean did
            # not change enough to confirm a ground-roll transition.
            return failed_estimate(
                self.method_name,
                FailureReason.NO_GROUND_ROLL_CONFIRMATION,
                diagnostics={
                    "cadence_s": signal.cadence_s,
                    "reference_k": self.reference_k,
                    "threshold_h": self.threshold_h,
                    "max_s_hi": float(np.max(s_hi)),
                    "min_s_lo": float(np.min(s_lo)),
                },
            )

        change_index = int(np.clip(change_index, 1, n - 1))
        mean_before = float(np.mean(decel[:change_index]))
        mean_after = float(np.mean(decel[change_index:]))

        t_td = subsample_transition_time(
            times, decel, change_index, mean_before, mean_after
        )
        residual_std = segment_residual_std(decel, change_index)
        shift = mean_after - mean_before
        sigma_t = localization_sigma(signal.cadence_s, residual_std, shift)

        diagnostics = {
            "change_index": change_index,
            "alarm_index": int(alarm_idx),
            "alarm_arm": alarm_arm,
            "alarm_lag_samples": int(alarm_idx - (change_index - 1)),
            "mean_decel_before_mps2": mean_before,
            "mean_decel_after_mps2": mean_after,
            "decel_shift_mps2": shift,
            "reference_k": self.reference_k,
            "threshold_h": self.threshold_h,
            "noise_std_mps2": float(sigma),
            "residual_std_mps2": float(residual_std),
            "cadence_s": signal.cadence_s,
            "derivative_reliable": bool(signal.derivative.reliable),
        }
        return make_estimate(
            t_td=t_td,
            sigma_t=sigma_t,
            confidence=CONFIDENCE_NORMAL,
            method_name=self.method_name,
            diagnostics=diagnostics,
        )
