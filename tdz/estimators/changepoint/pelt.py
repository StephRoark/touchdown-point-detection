"""PELT change-point estimator (Task 13).

Exact penalized change-point detection on the **smoothed deceleration** of
groundspeed (from :func:`tdz.estimators.changepoint.common.prepare_decel_signal`).
PELT (Pruned Exact Linear Time, Killick et al. 2012) finds the segmentation that
minimises a penalised cost; here the per-segment cost is the piecewise-constant
**mean L2** cost and the penalty is BIC-like, so PELT recovers the approach ->
ground-roll deceleration-regime change without re-wrapping the segmented-
regression primary (Req 5.2; design "Change-Point Estimators Detail").

Cost
----
For a contiguous segment ``[s, t)`` of the deceleration signal ``x`` the cost is
the residual sum of squares about the segment mean::

    cost(s, t) = sum_{i in [s,t)} (x_i - mean)^2
               = (sum x_i^2) - (sum x_i)^2 / (t - s)

computed in O(1) from prefix sums. This is the Gaussian (constant-variance)
negative-log-likelihood up to a constant, i.e. a piecewise-constant-mean model
of the deceleration regimes.

Penalty
-------
A BIC-like penalty ``penalty = beta * var_noise * log(n)`` is charged per added
change point, where ``var_noise`` is a **shift-robust** noise-variance estimate
(half the variance of first differences, which is insensitive to the regime
mean shift itself) and ``beta`` (default :data:`DEFAULT_PENALTY_BETA`) is the
strength multiplier. A clean, low-noise signal carries a small penalty (the
knee is found readily); a noisy signal raises the bar so spurious splits are
suppressed. ``beta`` / the noise floor are exposed as constructor params.

Picking ``t_td``
---------------
PELT may return several change points (e.g. approach -> transition -> rollout).
The touchdown candidate is the change point with the **largest deceleration
increase** -- the boundary whose post-segment mean deceleration is most more
negative than its pre-segment mean (braking sharpens at the regime change). The
change *index* is then refined to a sub-sample ``t_td`` by
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
    "PeltEstimator",
    "METHOD_NAME",
    "DEFAULT_PENALTY_BETA",
    "MIN_NOISE_VAR_MPS2_SQ",
    "pelt_changepoints",
]

#: Estimator identifier (matches the ``pelt`` id in ``ALLOWED_ESTIMATORS``).
METHOD_NAME: Final[str] = "pelt"

#: Default penalty strength multiplier ``beta`` in ``beta * var_noise * log(n)``.
DEFAULT_PENALTY_BETA: Final[float] = 2.0

#: Floor on the shift-robust noise-variance estimate ((m/s^2)^2) so a perfectly
#: clean signal still carries a non-zero penalty and PELT does not over-segment
#: float-rounding noise.
MIN_NOISE_VAR_MPS2_SQ: Final[float] = 1e-4


def _shift_robust_noise_var(x: np.ndarray) -> float:
    """Half the variance of first differences -- insensitive to the mean shift.

    For a piecewise-constant mean plus white noise, ``Var(diff) = 2*sigma^2``
    everywhere except the single step, so ``Var(diff)/2`` estimates the noise
    variance ``sigma^2`` without being inflated by the regime change itself.
    """
    if x.size < 2:
        return MIN_NOISE_VAR_MPS2_SQ
    return max(float(np.var(np.diff(x)) / 2.0), MIN_NOISE_VAR_MPS2_SQ)


def pelt_changepoints(x: np.ndarray, penalty: float, min_seg: int) -> list[int]:
    """Exact PELT segmentation under the piecewise-constant-mean L2 cost.

    Returns the sorted change indices (each is the first index of a new
    segment); an empty list means a single segment (no change).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2 * min_seg:
        return []

    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])

    def seg_cost(s: int, t: int) -> float:
        length = t - s
        ssum = cs[t] - cs[s]
        ssum2 = cs2[t] - cs2[s]
        return float(ssum2 - ssum * ssum / length)

    f = np.full(n + 1, np.inf)
    f[0] = -penalty
    last = np.zeros(n + 1, dtype=int)
    candidates: list[int] = [0]

    for t in range(1, n + 1):
        best_val = np.inf
        best_s = 0
        for s in candidates:
            if t - s < min_seg:
                continue
            c = f[s] + seg_cost(s, t) + penalty
            if c < best_val:
                best_val = c
                best_s = s
        if not np.isfinite(best_val):
            # No eligible candidate yet (t < min_seg): leave f[t] = inf.
            continue
        f[t] = best_val
        last[t] = best_s

        # Pruning: drop candidates that can never beat the current optimum.
        pruned = [
            s
            for s in candidates
            if (t - s < min_seg) or (f[s] + seg_cost(s, t) <= f[t])
        ]
        pruned.append(t)
        candidates = pruned

    # Backtrack the optimal change indices.
    cps: list[int] = []
    t = n
    while t > 0:
        s = int(last[t])
        if s > 0:
            cps.append(s)
        t = s
    cps.sort()
    return cps


class PeltEstimator(PhysicsEstimator):
    """PELT change-point estimate of the deceleration-regime transition (Req 5.2).

    Parameters
    ----------
    penalty_beta:
        Penalty strength ``beta`` (default :data:`DEFAULT_PENALTY_BETA`).
    min_seg:
        Minimum samples per segment (default :data:`MIN_SEG_SAMPLES`).
    config:
        Optional smoothing config forwarded to :func:`prepare_decel_signal`.
    """

    method_name = METHOD_NAME

    def __init__(
        self,
        *,
        penalty_beta: float = DEFAULT_PENALTY_BETA,
        min_seg: int = MIN_SEG_SAMPLES,
        config=None,
    ) -> None:
        self.penalty_beta = float(penalty_beta)
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
        n = decel.size

        var_noise = _shift_robust_noise_var(decel)
        penalty = self.penalty_beta * var_noise * float(np.log(max(n, 2)))

        cps = pelt_changepoints(decel, penalty, self.min_seg)

        fallback_single = False
        if not cps:
            # No penalised split survived: fall back to the single best L2 split
            # so the corroborator still reports a candidate (down the increase).
            best = self._best_single_split(decel)
            if best is None:
                return failed_estimate(self.method_name, reason)
            cps = [best]
            fallback_single = True

        # Boundaries: [0, cp1, ..., n]. Choose the change with the largest
        # deceleration increase (post-mean most more-negative than pre-mean).
        boundaries = [0, *cps, n]
        best_idx = 0
        best_increase = -np.inf
        best_means = (0.0, 0.0)
        for j, cp in enumerate(cps):
            prev_b = boundaries[j]
            next_b = boundaries[j + 2]
            mean_before = float(np.mean(decel[prev_b:cp]))
            mean_after = float(np.mean(decel[cp:next_b]))
            increase = mean_before - mean_after  # > 0 when braking sharpens
            if increase > best_increase:
                best_increase = increase
                best_idx = cp
                best_means = (mean_before, mean_after)

        mean_before, mean_after = best_means
        t_td = subsample_transition_time(times, decel, best_idx, mean_before, mean_after)
        residual_std = segment_residual_std(decel, best_idx)
        shift = mean_after - mean_before
        sigma_t = localization_sigma(signal.cadence_s, residual_std, shift)

        diagnostics = {
            "change_index": int(best_idx),
            "change_indices": [int(c) for c in cps],
            "n_changepoints": len(cps),
            "mean_decel_before_mps2": mean_before,
            "mean_decel_after_mps2": mean_after,
            "decel_shift_mps2": shift,
            "penalty": float(penalty),
            "penalty_beta": self.penalty_beta,
            "noise_var_mps2_sq": float(var_noise),
            "residual_std_mps2": float(residual_std),
            "cadence_s": signal.cadence_s,
            "derivative_reliable": bool(signal.derivative.reliable),
            "fallback_single_change": fallback_single,
            "cost_model": "piecewise_constant_mean_l2",
        }
        return make_estimate(
            t_td=t_td,
            sigma_t=sigma_t,
            confidence=CONFIDENCE_NORMAL,
            method_name=self.method_name,
            diagnostics=diagnostics,
        )

    def _best_single_split(self, x: np.ndarray) -> int | None:
        """Argmax-reduction single change point (PELT fallback when none found)."""
        n = x.size
        if n < 2 * self.min_seg:
            return None
        cs = np.concatenate([[0.0], np.cumsum(x)])
        total = cs[n]
        best_k = None
        best_gain = -np.inf
        for k in range(self.min_seg, n - self.min_seg + 1):
            sum_b = cs[k]
            mean_b = sum_b / k
            mean_a = (total - sum_b) / (n - k)
            # Between-group sum of squares (the L2 cost reduction of the split).
            gain = k * (n - k) / n * (mean_b - mean_a) ** 2
            if gain > best_gain:
                best_gain = gain
                best_k = k
        return best_k
