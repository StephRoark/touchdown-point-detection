"""Segmented (piecewise-linear) regression on raw groundspeed (Task 11.1).

The **primary** deceleration-regime estimate is obtained by fitting a segmented
(piecewise-linear) model **directly to the raw groundspeed-vs-time series** and
taking the breakpoint as the regime transition, rather than differentiating the
signal and then detecting a change point (Req 16.1; design key decision
"Segmented regression on raw groundspeed for the primary decel estimate").

Rationale (Req 16.1, 16.5)
--------------------------
Differentiate-then-detect forces a smoothing-window trade-off: a wide window
suppresses the 4-5 s cadence noise but blurs the very breakpoint being located,
while a narrow window keeps the transition sharp but leaves the derivative
noise-dominated. Fitting line segments to the *raw* speed signal localizes the
knee without any smoothing window at all -- the breakpoint is a free parameter
of the fit, not the argmax of a smoothed derivative.

Model
-----
A **continuous** piecewise-linear model is fit. For two segments with a single
breakpoint :math:`\\tau` the model is

    v(t) = beta0 + beta1 * t + beta2 * max(0, t - tau)

which is linear in ``(beta0, beta1, beta2)`` for a fixed ``tau`` (ordinary least
squares) and continuous at ``tau`` by construction (the segments meet -- no jump
discontinuity, so a clean two-slope "knee"). The slope is ``beta1`` before the
knot and ``beta1 + beta2`` after it. The breakpoint is found by **minimising the
total least-squares residual over a fine grid of candidate breakpoints** (a
sub-sample grid, so the located knee is not quantised to the sample cadence).
The optional three-segment variant adds a second hinge and searches candidate
breakpoint *pairs*.

Units convention
----------------
Groundspeed is ingested in knots and converted to SI (m/s) up front with
:data:`~tdz.timebase.interpolation.KNOTS_TO_MPS`, so every fitted quantity is
SI: segment slopes are decelerations in m/s^2, intercepts are m/s, and the
residual RMS is m/s. Times are epoch seconds; the fit is performed in a local
time frame (``t - times[0]``) for numerical conditioning and the breakpoint is
reported back in epoch seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from tdz.timebase.interpolation import KNOTS_TO_MPS

__all__ = [
    "SegmentedFit",
    "fit_segmented_groundspeed",
    "MIN_SAMPLES_PER_SEGMENT",
]

#: Minimum number of samples required on each side of every breakpoint for a
#: well-posed continuous piecewise-linear fit (two points define a line; fewer
#: would leave a segment slope under-determined / noise-dominated).
MIN_SAMPLES_PER_SEGMENT: Final[int] = 2

#: Upper bound on the number of single-breakpoint candidates evaluated on the
#: sub-sample grid. Keeps the 3-segment pair search (O(n_cand^2)) tractable.
_MAX_CANDIDATES_2SEG: Final[int] = 400
_MAX_CANDIDATES_3SEG: Final[int] = 60


@dataclass(frozen=True)
class SegmentedFit:
    """Result of a segmented piecewise-linear groundspeed fit (SI units).

    Attributes
    ----------
    breakpoint_time:
        The **primary** regime-transition time (epoch seconds): the breakpoint
        across which the deceleration steepens the most (approach decel ->
        ground-roll decel). For a 2-segment fit this is the single breakpoint.
        This is the quantity the decel-knee estimator (Task 12) consumes as the
        candidate ``t_td``.
    breakpoint_times:
        All fitted breakpoint times (epoch seconds), ordered in time. Length
        ``n_segments - 1``.
    slopes_mps2:
        Per-segment slopes -- the fitted decelerations in m/s^2 (negative while
        the aircraft is slowing). Length ``n_segments``.
    intercepts_mps:
        Per-segment line constant terms in m/s, expressed in the **local** time
        frame where ``t = 0`` at ``times[0]`` (i.e. the value each segment line
        extrapolates to at ``times[0]``). Diagnostic only.
    residual_rms_mps:
        Root-mean-square fit residual against the raw groundspeed (m/s); a
        smaller value means the two-slope model explains the speed profile well.
    n_segments:
        Number of fitted segments (2 or 3).
    n_samples:
        Number of valid samples the fit used.
    """

    breakpoint_time: float
    breakpoint_times: tuple[float, ...]
    slopes_mps2: tuple[float, ...]
    intercepts_mps: tuple[float, ...]
    residual_rms_mps: float
    n_segments: int
    n_samples: int


def _design_matrix(u: np.ndarray, knots: tuple[float, ...]) -> np.ndarray:
    """Build the continuous-piecewise-linear design matrix ``[1, u, (u-k)_+...]``."""
    cols = [np.ones_like(u), u]
    for k in knots:
        cols.append(np.clip(u - k, 0.0, None))
    return np.column_stack(cols)


def _fit_for_knots(
    u: np.ndarray, v: np.ndarray, knots: tuple[float, ...]
) -> tuple[np.ndarray, float]:
    """Least-squares fit for fixed knots; return ``(beta, residual_sum_squares)``."""
    x = _design_matrix(u, knots)
    beta, _residuals, _rank, _sv = np.linalg.lstsq(x, v, rcond=None)
    fitted = x @ beta
    rss = float(np.sum((v - fitted) ** 2))
    return beta, rss


def _segment_params(
    beta: np.ndarray, knots: tuple[float, ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Convert hinge coefficients to per-segment ``(slopes, intercepts)`` (SI).

    The slope of segment ``i`` is the cumulative sum of hinge coefficients;
    intercepts are propagated from segment to segment using continuity at each
    knot (``a_{i+1} = a_i + (s_i - s_{i+1}) * k_i``).
    """
    slopes = [float(beta[1])]
    for j in range(len(knots)):
        slopes.append(slopes[-1] + float(beta[2 + j]))

    intercepts = [float(beta[0])]
    for i, k in enumerate(knots):
        intercepts.append(intercepts[i] + (slopes[i] - slopes[i + 1]) * float(k))

    return tuple(slopes), tuple(intercepts)


def _candidate_knots(u: np.ndarray, n_candidates: int) -> np.ndarray:
    """Sub-sample grid of candidate breakpoints with >=2 points on each side."""
    lo, hi = float(u[1]), float(u[-2])
    if hi <= lo:
        return np.array([], dtype=float)
    grid = np.linspace(lo, hi, n_candidates)
    # Keep only candidates that leave at least MIN_SAMPLES_PER_SEGMENT points on
    # both sides so neither adjacent segment slope is under-determined.
    left = np.searchsorted(u, grid, side="left")
    right = u.size - np.searchsorted(u, grid, side="right")
    keep = (left >= MIN_SAMPLES_PER_SEGMENT) & (right >= MIN_SAMPLES_PER_SEGMENT)
    return grid[keep]


def _best_two_segment(u: np.ndarray, v: np.ndarray) -> tuple[tuple[float, ...], np.ndarray]:
    """Search single breakpoints; return ``(best_knots, best_beta)``."""
    n_cand = min(_MAX_CANDIDATES_2SEG, max(60, 12 * u.size))
    candidates = _candidate_knots(u, n_cand)
    if candidates.size == 0:
        raise ValueError("no admissible breakpoint candidates for a 2-segment fit")

    best_rss = np.inf
    best_knots: tuple[float, ...] = (float(candidates[0]),)
    best_beta = np.zeros(3)
    for c in candidates:
        beta, rss = _fit_for_knots(u, v, (float(c),))
        if rss < best_rss:
            best_rss = rss
            best_knots = (float(c),)
            best_beta = beta
    return best_knots, best_beta


def _best_three_segment(u: np.ndarray, v: np.ndarray) -> tuple[tuple[float, ...], np.ndarray]:
    """Search breakpoint pairs; return ``(best_knots, best_beta)``."""
    n_cand = min(_MAX_CANDIDATES_3SEG, max(20, 4 * u.size))
    candidates = _candidate_knots(u, n_cand)
    if candidates.size < 2:
        raise ValueError("too few admissible breakpoint candidates for a 3-segment fit")

    best_rss = np.inf
    best_knots: tuple[float, ...] | None = None
    best_beta = np.zeros(4)
    for i in range(candidates.size):
        c1 = float(candidates[i])
        for j in range(i + 1, candidates.size):
            c2 = float(candidates[j])
            # Require >=2 points strictly inside the middle segment.
            middle = int(np.sum((u > c1) & (u < c2)))
            if middle < MIN_SAMPLES_PER_SEGMENT:
                continue
            beta, rss = _fit_for_knots(u, v, (c1, c2))
            if rss < best_rss:
                best_rss = rss
                best_knots = (c1, c2)
                best_beta = beta
    if best_knots is None:
        raise ValueError("no admissible breakpoint pair for a 3-segment fit")
    return best_knots, best_beta


def _primary_breakpoint(
    knots_epoch: tuple[float, ...], slopes: tuple[float, ...]
) -> float:
    """Pick the knot where deceleration steepens most (largest slope drop).

    Approach deceleration is gentle (slope slightly negative); ground-roll
    braking is steep (slope strongly negative). The regime transition the
    decel-knee estimator wants is the breakpoint with the largest *decrease* in
    slope (``slope_before - slope_after``). With a single breakpoint this is
    simply that breakpoint.
    """
    if len(knots_epoch) == 1:
        return knots_epoch[0]
    drops = [slopes[i] - slopes[i + 1] for i in range(len(knots_epoch))]
    return knots_epoch[int(np.argmax(drops))]


def fit_segmented_groundspeed(
    times: np.ndarray,
    groundspeeds_kt: np.ndarray,
    *,
    n_segments: int = 2,
) -> SegmentedFit:
    """Fit a continuous piecewise-linear model to raw groundspeed vs time.

    The breakpoint(s) are found by minimising the total least-squares residual
    over a sub-sample grid of candidate breakpoints (Req 16.1). Groundspeed is
    converted knots -> m/s before fitting, so all returned quantities are SI.

    Parameters
    ----------
    times:
        Sample times (epoch seconds), the **velocity** timebase. Need not be
        pre-sorted; NaN speeds are dropped.
    groundspeeds_kt:
        Groundspeed at each time (knots).
    n_segments:
        ``2`` (default; approach decel -> ground-roll decel) or ``3``
        (approach / transition / rollout).

    Returns
    -------
    SegmentedFit
        Breakpoint time(s), per-segment slopes (decelerations, m/s^2),
        intercepts (m/s), residual RMS (m/s), and bookkeeping.

    Raises
    ------
    ValueError
        If ``n_segments`` is not 2 or 3, or there are too few valid samples to
        support the requested number of segments.
    """
    if n_segments not in (2, 3):
        raise ValueError(f"n_segments must be 2 or 3, got {n_segments}")

    t = np.asarray(times, dtype=float)
    v_kt = np.asarray(groundspeeds_kt, dtype=float)
    if t.shape != v_kt.shape:
        raise ValueError("times and groundspeeds_kt must have the same shape")

    valid = ~(np.isnan(t) | np.isnan(v_kt))
    t = t[valid]
    v_kt = v_kt[valid]

    order = np.argsort(t)
    t = t[order]
    v_kt = v_kt[order]

    # A continuous PWL fit with k knots has (k + 2) free parameters; require at
    # least one more sample than parameters plus the per-segment minimum.
    min_samples = max(n_segments + 2, MIN_SAMPLES_PER_SEGMENT * n_segments)
    if t.size < min_samples:
        raise ValueError(
            f"need at least {min_samples} valid samples for a {n_segments}-segment "
            f"fit, got {t.size}"
        )

    t0 = float(t[0])
    u = t - t0
    v = v_kt * KNOTS_TO_MPS  # fit entirely in SI (m/s)

    if n_segments == 2:
        knots_local, beta = _best_two_segment(u, v)
    else:
        knots_local, beta = _best_three_segment(u, v)

    _x = _design_matrix(u, knots_local)
    rss = float(np.sum((v - _x @ beta) ** 2))
    residual_rms = float(np.sqrt(rss / u.size))

    slopes, intercepts = _segment_params(beta, knots_local)
    knots_epoch = tuple(t0 + k for k in knots_local)
    primary = _primary_breakpoint(knots_epoch, slopes)

    return SegmentedFit(
        breakpoint_time=primary,
        breakpoint_times=knots_epoch,
        slopes_mps2=slopes,
        intercepts_mps=intercepts,
        residual_rms_mps=residual_rms,
        n_segments=n_segments,
        n_samples=int(u.size),
    )
