"""Corroborating smoothed derivatives of groundspeed (Task 11.2).

Deceleration (1st derivative) and jerk (2nd derivative) of groundspeed are
**corroborating** signals only -- never the sole basis for ``t_td`` (Req 16.3).
The primary deceleration-regime estimate comes from segmented regression on the
raw speed (:mod:`tdz.signals.segmented`); these smoothed derivatives back it up
and feed the learned models.

Two smoothing methods, selected by :class:`~tdz.config.schema.SignalsConfig`:

* ``"savgol"`` -- a Savitzky-Golay-style **local polynomial** fit. Classic
  Savitzky-Golay assumes a uniform sample grid; ADS-B at 4-5 s is only
  approximately uniform, so this is implemented as an equally-weighted local
  polynomial regression over the configured window using the **actual** sample
  time offsets (a generalised SavGol). The derivative at each point is read off
  the fitted polynomial coefficients (poly order <= 3, window >= 5 samples per
  Req 16.2).
* ``"gp"`` -- a numpy-only **Gaussian-process surrogate**: a kernel-weighted
  local polynomial regression whose Gaussian weights have length scale
  ``gp_length_scale_s`` and whose observation noise floor is
  ``gp_noise_variance``. This is the locally-weighted-regression view of a GP
  posterior mean; it yields a posterior-style standard deviation per derivative
  (Req 16.4) without a heavyweight GP library. (Documented surrogate: it is not
  a full marginal-likelihood GP, but provides the inverse-variance weights
  downstream estimators need.)

Non-stationarity (Req 16.4)
---------------------------
A single stationary length scale must NOT be assumed across the whole landing:
the speed signal is smooth on approach and sharply non-stationary at the
deceleration knee. When a ``breakpoint_time`` is supplied (from the segmented
fit), smoothing is applied **piecewise** -- each point's window is restricted to
samples on its own side of the breakpoint, so the transition is never
over-smoothed. This is the "applied piecewise so the transition is not
over-smoothed" option of Req 16.4.

Posterior standard deviation
----------------------------
For both methods the per-point derivative uncertainty is the standard deviation
of the fitted derivative coefficient: ``cov = s^2 * (X^T W X)^-1`` where ``s^2``
is the (weighted) residual variance with a noise-variance floor for the GP path,
and ``X`` is the local polynomial design in centred time. ``std(deceleration) =
sqrt(cov[1, 1])`` and ``std(jerk) = 2 * sqrt(cov[2, 2])``.

Reliability (Req 16.5, 16.6)
----------------------------
The configured smoothing window is reported in the diagnostics. If fewer than
:data:`MIN_VALID_SAMPLES_IN_WINDOW` (5) valid groundspeed samples lie in any
evaluated window, the whole derivative channel is flagged unreliable
(``reliable=False``) so any estimator depending on it inherits a low-confidence
indicator.

Units convention
----------------
Groundspeed knots -> m/s via :data:`~tdz.timebase.interpolation.KNOTS_TO_MPS`;
deceleration is m/s^2, jerk is m/s^3, times are epoch seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

import numpy as np

from tdz.timebase.interpolation import KNOTS_TO_MPS

__all__ = [
    "DerivativeResult",
    "smoothed_derivatives",
    "deceleration_rms_discrepancy",
    "MIN_VALID_SAMPLES_IN_WINDOW",
    "GP_KERNEL_SUPPORT_SIGMAS",
    "GP_POLY_ORDER",
]

#: Minimum valid groundspeed samples that must lie in the smoothing window for
#: the derivative to be considered reliable (Req 16.6).
MIN_VALID_SAMPLES_IN_WINDOW: Final[int] = 5

#: GP-surrogate kernel support, in multiples of the length scale: samples beyond
#: this many length scales carry negligible weight and are excluded.
GP_KERNEL_SUPPORT_SIGMAS: Final[float] = 3.0

#: Local polynomial order used by the GP surrogate (>=2 so jerk is available).
GP_POLY_ORDER: Final[int] = 2


@dataclass(frozen=True)
class DerivativeResult:
    """Smoothed first/second derivatives of groundspeed with uncertainty (SI).

    All per-sample arrays are on the **velocity** timebase (the same ``times``
    passed in). ``NaN`` marks points where no valid derivative could be fit
    (too few samples in the local window).

    Attributes
    ----------
    times:
        Echo of the input velocity timebase (epoch seconds).
    deceleration_mps2:
        First derivative of groundspeed (m/s^2); negative while slowing.
    jerk_mps3:
        Second derivative of groundspeed (m/s^3).
    derivative_uncertainty:
        1-sigma posterior standard deviation of ``deceleration_mps2`` (m/s^2),
        for inverse-variance weighting downstream (Req 16.4).
    jerk_uncertainty:
        1-sigma standard deviation of ``jerk_mps3`` (m/s^3).
    method:
        ``"savgol"`` or ``"gp"``.
    window_samples:
        The configured smoothing window reported in diagnostics (Req 16.5). For
        ``savgol`` this is ``savgol_window_samples``; for ``gp`` it is the median
        number of samples falling inside the kernel support.
    gp_length_scale_s:
        The GP kernel length scale (seconds); ``None`` for the SavGol path.
    piecewise:
        ``True`` when smoothing was applied piecewise about a breakpoint so the
        deceleration knee was not over-smoothed (Req 16.4).
    reliable:
        ``False`` when any evaluated window held fewer than
        :data:`MIN_VALID_SAMPLES_IN_WINDOW` valid samples (Req 16.6).
    min_valid_in_window:
        Minimum count of valid samples across all evaluated windows.
    """

    times: np.ndarray
    deceleration_mps2: np.ndarray
    jerk_mps3: np.ndarray
    derivative_uncertainty: np.ndarray
    jerk_uncertainty: np.ndarray
    method: str
    window_samples: int
    gp_length_scale_s: Optional[float]
    piecewise: bool
    reliable: bool
    min_valid_in_window: int


def _weighted_local_fit(
    du: np.ndarray, y: np.ndarray, weights: np.ndarray, poly_order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted local polynomial fit in centred time; return ``(beta, cov)``.

    ``du`` are time offsets relative to the query point (so the polynomial is
    ``y = b0 + b1*du + b2*du^2 + ...`` and ``b1`` is the first derivative,
    ``2*b2`` the second). Returns the coefficient vector and its covariance
    ``s^2 * (X^T W X)^-1``.
    """
    n = du.size
    p = poly_order + 1
    x = np.vander(du, N=p, increasing=True)  # columns: 1, du, du^2, ...
    w = weights
    xtw = x.T * w  # (p, n)
    xtwx = xtw @ x  # (p, p)
    xtwy = xtw @ y  # (p,)
    xtwx_inv = np.linalg.pinv(xtwx)
    beta = xtwx_inv @ xtwy

    resid = y - x @ beta
    dof = max(n - p, 1)
    s2 = float((w @ (resid ** 2)) / (np.sum(w) / n * dof))
    cov = s2 * xtwx_inv
    return beta, cov


def smoothed_derivatives(
    times: np.ndarray,
    groundspeeds_kt: np.ndarray,
    config: object,
    *,
    breakpoint_time: Optional[float] = None,
) -> DerivativeResult:
    """Compute corroborating smoothed deceleration and jerk of groundspeed.

    Parameters
    ----------
    times:
        Velocity-timebase sample times (epoch seconds). Sorted ascending is
        assumed; NaN speeds are treated as invalid (excluded from every window).
    groundspeeds_kt:
        Groundspeed at each time (knots).
    config:
        A :class:`~tdz.config.schema.SignalsConfig`-like object providing
        ``smoothing_method`` (``"savgol"``/``"gp"``), ``savgol_window_samples``,
        ``savgol_poly_order``, ``gp_length_scale_s`` and ``gp_noise_variance``.
    breakpoint_time:
        Optional regime-transition time (epoch seconds), typically from
        :func:`~tdz.signals.segmented.fit_segmented_groundspeed`. When given,
        smoothing is applied piecewise about it so the knee is not over-smoothed
        (Req 16.4).

    Returns
    -------
    DerivativeResult
        Deceleration (m/s^2), jerk (m/s^3), their 1-sigma uncertainties, the
        reported window, and the reliability flag.
    """
    method = str(getattr(config, "smoothing_method", "savgol")).lower()
    savgol_window = int(getattr(config, "savgol_window_samples", 7))
    savgol_order = int(getattr(config, "savgol_poly_order", 3))
    gp_length_scale = float(getattr(config, "gp_length_scale_s", 8.0))
    gp_noise_variance = float(getattr(config, "gp_noise_variance", 0.5))

    t = np.asarray(times, dtype=float)
    v = np.asarray(groundspeeds_kt, dtype=float) * KNOTS_TO_MPS
    n = t.size

    decel = np.full(n, np.nan)
    jerk = np.full(n, np.nan)
    decel_std = np.full(n, np.nan)
    jerk_std = np.full(n, np.nan)

    valid = ~(np.isnan(t) | np.isnan(v))

    # Segment id per sample for piecewise smoothing (windows never cross it).
    if breakpoint_time is not None:
        segment = (t >= float(breakpoint_time)).astype(int)
        piecewise = True
    else:
        segment = np.zeros(n, dtype=int)
        piecewise = False

    min_valid_in_window = n if n > 0 else 0
    gp_support_counts: list[int] = []

    for i in range(n):
        if not valid[i]:
            continue

        same_segment = valid & (segment == segment[i])
        idx_pool = np.where(same_segment)[0]
        if idx_pool.size == 0:
            continue

        dt_pool = np.abs(t[idx_pool] - t[i])

        if method == "gp":
            poly_order = GP_POLY_ORDER
            support = GP_KERNEL_SUPPORT_SIGMAS * gp_length_scale
            in_support = dt_pool <= support
            sel = idx_pool[in_support]
            if sel.size == 0:
                sel = idx_pool
            du = t[sel] - t[i]
            weights = np.exp(-0.5 * (du / gp_length_scale) ** 2)
            gp_support_counts.append(int(sel.size))
        else:  # savgol-style equally-weighted local polynomial
            poly_order = savgol_order
            order = np.argsort(dt_pool)
            sel = idx_pool[order[: min(savgol_window, idx_pool.size)]]
            du = t[sel] - t[i]
            weights = np.ones(sel.size, dtype=float)

        count = int(sel.size)
        min_valid_in_window = min(min_valid_in_window, count)

        # Cap the polynomial order to what the available points can support.
        eff_order = min(poly_order, count - 1)
        if eff_order < 1:
            continue

        y = v[sel]
        beta, cov = _weighted_local_fit(du, y, weights, eff_order)

        decel[i] = float(beta[1])
        decel_var = float(cov[1, 1])
        if method == "gp":
            decel_var = max(decel_var, gp_noise_variance / max(np.sum(weights), 1e-9))
        decel_std[i] = float(np.sqrt(max(decel_var, 0.0)))

        if eff_order >= 2:
            jerk[i] = float(2.0 * beta[2])
            jerk_std[i] = float(2.0 * np.sqrt(max(float(cov[2, 2]), 0.0)))
        else:
            jerk[i] = 0.0
            jerk_std[i] = float("nan")

    if method == "gp":
        window_samples = int(np.median(gp_support_counts)) if gp_support_counts else 0
        gp_ls: Optional[float] = gp_length_scale
    else:
        window_samples = savgol_window
        gp_ls = None

    n_valid = int(np.sum(valid))
    effective_min = min_valid_in_window if n_valid > 0 else 0
    reliable = bool(n_valid >= MIN_VALID_SAMPLES_IN_WINDOW
                    and effective_min >= MIN_VALID_SAMPLES_IN_WINDOW)

    return DerivativeResult(
        times=t,
        deceleration_mps2=decel,
        jerk_mps3=jerk,
        derivative_uncertainty=decel_std,
        jerk_uncertainty=jerk_std,
        method=method,
        window_samples=window_samples,
        gp_length_scale_s=gp_ls,
        piecewise=piecewise,
        reliable=reliable,
        min_valid_in_window=int(effective_min),
    )


def deceleration_rms_discrepancy(
    smoothed_mps2: np.ndarray, reference_mps2: np.ndarray
) -> float:
    """RMS discrepancy (m/s^2) between smoothed and reference deceleration.

    Reusable harness for Req 16.7: compares the smoothed ADS-B deceleration
    against a reference acceleration series (e.g. QAR-derived) sampled at the
    same times and reports the RMS difference in m/s^2. ``NaN`` pairs (points
    with no valid smoothed derivative) are ignored. The full held-out-sample
    comparison against real QAR acceleration runs in the validation harness
    (Task 22); this helper is the unit-testable core.

    Returns ``NaN`` when no overlapping finite pairs exist.
    """
    a = np.asarray(smoothed_mps2, dtype=float)
    b = np.asarray(reference_mps2, dtype=float)
    if a.shape != b.shape:
        raise ValueError("smoothed and reference series must have the same shape")
    both = np.isfinite(a) & np.isfinite(b)
    if not np.any(both):
        return float("nan")
    diff = a[both] - b[both]
    return float(np.sqrt(np.mean(diff ** 2)))
