"""GLRT change-point estimator (Task 13).

A single-change Generalized Likelihood Ratio Test for a shift in the mean of the
**smoothed deceleration** of groundspeed (from :func:`prepare_decel_signal`).
The GLRT supplies a principled detection statistic for the approach ->
ground-roll regime change (Req 5.2; design "Change-Point Estimators Detail").

Statistic
---------
Under the null the deceleration is a single Gaussian (constant mean); under the
alternative there is one change at index ``k`` separating two constant-mean
segments. With a common variance the generalized log-likelihood ratio for a
split at ``k`` reduces to the standardized between-segment contrast::

    G(k) = [ n_b * n_a / n ] * (mean_before - mean_after)^2 / sigma^2

where ``n_b = k``, ``n_a = n - k`` and ``sigma^2`` is a shift-robust noise
variance (from first differences). ``G(k)`` is exactly the L2 cost reduction of
the split scaled by the noise variance, and is monotone in the classic
``2 * log Lambda`` GLR. The change index is ``argmax_k G(k)`` over admissible
splits (each segment at least :data:`~tdz.estimators.changepoint.common.MIN_SEG_SAMPLES`
samples), and the peak ``G`` is reported as the detection statistic.

``t_td`` / ``sigma_t``
----------------------
The argmax change *index* is refined to a sub-sample ``t_td`` by
:func:`subsample_transition_time` (half-level crossing of the smoothed step) and
``sigma_t`` derived by :func:`localization_sigma` from the within-segment
residual scatter and the deceleration shift. The on-ground upper bound is
applied by :class:`PhysicsEstimator`.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from tdz.estimators.changepoint.common import (
    CONFIDENCE_NORMAL,
    MIN_SEG_SAMPLES,
    PhysicsEstimator,
    failed_estimate,
    localization_sigma,
    make_estimate,
    prepare_decel_signal,
    segment_residual_std,
    subsample_transition_time,
)
from tdz.models import FlightRecord, TDEstimate

__all__ = [
    "GlrtEstimator",
    "METHOD_NAME",
    "MIN_NOISE_VAR_MPS2_SQ",
    "glr_profile",
]

#: Estimator identifier.
METHOD_NAME: Final[str] = "glrt"

#: Floor on the shift-robust noise-variance estimate ((m/s^2)^2) used to
#: standardize the GLR statistic.
MIN_NOISE_VAR_MPS2_SQ: Final[float] = 1e-4


def glr_profile(x: np.ndarray, min_seg: int) -> tuple[np.ndarray, np.ndarray]:
    """Single-change GLR profile ``G(k)`` over admissible split indices.

    Returns ``(indices, statistic)`` where ``indices[j]`` is a candidate change
    index (first index of the post-change segment) and ``statistic[j]`` the
    standardized between-segment contrast ``n_b*n_a/n*(mean_b-mean_a)^2 /
    sigma^2``. Empty arrays when the signal is too short to admit a split.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2 * min_seg:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    var_noise = max(float(np.var(np.diff(x)) / 2.0), MIN_NOISE_VAR_MPS2_SQ)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    total = cs[n]

    ks = np.arange(min_seg, n - min_seg + 1)
    sum_b = cs[ks]
    mean_b = sum_b / ks
    mean_a = (total - sum_b) / (n - ks)
    stat = ks * (n - ks) / n * (mean_b - mean_a) ** 2 / var_noise
    return ks, stat


class GlrtEstimator(PhysicsEstimator):
    """Single-change GLRT estimate of the deceleration-regime transition (Req 5.2).

    Parameters
    ----------
    min_seg:
        Minimum samples per segment (default :data:`MIN_SEG_SAMPLES`).
    config:
        Optional smoothing config forwarded to :func:`prepare_decel_signal`.
    """

    method_name = METHOD_NAME

    def __init__(self, *, min_seg: int = MIN_SEG_SAMPLES, config=None) -> None:
        self.min_seg = int(min_seg)
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

        ks, stat = glr_profile(decel, self.min_seg)
        if ks.size == 0:
            return failed_estimate(self.method_name, reason)

        best = int(np.argmax(stat))
        change_index = int(ks[best])
        glr_stat = float(stat[best])

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
            "glr_statistic": glr_stat,
            "mean_decel_before_mps2": mean_before,
            "mean_decel_after_mps2": mean_after,
            "decel_shift_mps2": shift,
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
