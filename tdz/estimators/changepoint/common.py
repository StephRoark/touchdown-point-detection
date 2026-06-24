"""Shared scaffolding for the change-point estimators (Task 13).

The four change-point estimators (PELT, CUSUM, GLRT, jerk-onset) are independent
**corroborators** of the deceleration-regime transition: each detects the
approach -> ground-roll change on the *velocity stream* with its own statistic
(they do **not** re-wrap the segmented-regression primary estimate). They are
robust without geometric altitude and so run on every source (Aireon and FR24).

Reuse of the physics scaffolding (cross-package import)
-------------------------------------------------------
The Requirement-18 on-ground-flag upper bound, the :class:`TDEstimate`
constructors, and the confidence constants already live in
:mod:`tdz.estimators.physics.base`. Rather than duplicate them, the change-point
estimators import :class:`~tdz.estimators.physics.base.PhysicsEstimator` (and the
helpers) directly from that module and re-export them here so the rest of this
package has a single import site. This is a deliberate cross-package import: the
"physics" base is really the *shared estimator* base (it bakes in the on-ground
bound that every estimator -- physics or change-point -- must respect, Property
5); the change-point estimators subclass it unchanged.

The deceleration signal the detectors run on
--------------------------------------------
PELT / CUSUM / GLRT detect a shift in the **deceleration** of groundspeed. To
avoid the noise-dominated naive-finite-difference trap at 4-5 s cadence (Req
16.2), the deceleration is taken from :func:`tdz.signals.derivatives.smoothed_derivatives`
(a local-polynomial / GP-surrogate smoother), not raw differencing. A modest
window keeps the knee sharp; :data:`DEFAULT_SIGNAL_CONFIG` documents the default.
The jerk-onset detector likewise uses the *smoothed* jerk (Req 16.2/16.3).

Sub-sample localization
-----------------------
A detector returns a change *index* ``k`` (samples ``[k:]`` are the post-change
regime). The reported ``t_td`` is refined to sub-sample resolution by
:func:`subsample_transition_time`: it finds where the smoothed deceleration
crosses the half-level between the two regime means and linearly interpolates the
crossing time. For a smoothed step this crossing sits at the knee centre, so the
estimate is unbiased regardless of which edge of the smoothed ramp the raw
detector alarmed on (this matters most for CUSUM, whose alarm sits on the ramp's
leading edge). It falls back to the bracketing-sample midpoint when no clean
crossing exists.

Uncertainty (``sigma_t``)
-------------------------
:func:`localization_sigma` mirrors the physics estimators' derivation: a fit/
detector term ``cadence * residual_std / |shift|`` (a deceleration scatter of
``residual_std`` m/s^2 around a knee whose deceleration changes by ``shift``
m/s^2 maps, over one cadence, to a time ambiguity in seconds) combined in
quadrature with a **cadence floor** ``CADENCE_SIGMA_FRACTION * cadence`` and
clamped to an **absolute floor** :data:`MIN_SIGMA_T_S`. A sharp, low-noise change
localizes tightly; a soft or noisy change widens.

Units: SI throughout. Groundspeed knots -> m/s is handled inside
:func:`smoothed_derivatives` via :data:`~tdz.timebase.interpolation.KNOTS_TO_MPS`;
deceleration is m/s^2, jerk m/s^3, times epoch seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

import numpy as np

# Cross-package reuse of the shared estimator base + helpers (see module docstring).
from tdz.estimators.physics.base import (
    CONFIDENCE_FAILED,
    CONFIDENCE_LOW,
    CONFIDENCE_NORMAL,
    PhysicsEstimator,
    failed_estimate,
    make_estimate,
)
from tdz.models import FailureReason
from tdz.signals.derivatives import DerivativeResult, smoothed_derivatives

__all__ = [
    # Re-exported shared scaffolding.
    "PhysicsEstimator",
    "make_estimate",
    "failed_estimate",
    "CONFIDENCE_NORMAL",
    "CONFIDENCE_LOW",
    "CONFIDENCE_FAILED",
    # Constants.
    "MIN_SAMPLES_FOR_CHANGEPOINT",
    "MIN_SEG_SAMPLES",
    "CADENCE_SIGMA_FRACTION",
    "MIN_SIGMA_T_S",
    "MIN_DECEL_SHIFT_MPS2",
    "ChangePointSignalConfig",
    "DEFAULT_SIGNAL_CONFIG",
    # Helpers.
    "DecelSignal",
    "prepare_decel_signal",
    "median_cadence",
    "subsample_transition_time",
    "segment_residual_std",
    "localization_sigma",
]

#: Minimum number of valid groundspeed samples required to run a change-point
#: detector. Two regimes need at least :data:`MIN_SEG_SAMPLES` on each side plus
#: headroom for the smoother; below this the regime change is under-determined.
MIN_SAMPLES_FOR_CHANGEPOINT: Final[int] = 6

#: Minimum samples per segment either side of a detected change (a constant mean
#: needs >=2 points to be distinguishable from noise).
MIN_SEG_SAMPLES: Final[int] = 2

#: Fraction of the median sample spacing used as the irreducible cadence floor on
#: the change-point time uncertainty (the change cannot be located more finely
#: than a fraction of the cadence). Documented tunable.
CADENCE_SIGMA_FRACTION: Final[float] = 0.5

#: Absolute floor on the reported ``sigma_t`` (seconds); matches the physics
#: estimators so fused uncertainties are on a common scale.
MIN_SIGMA_T_S: Final[float] = 0.25

#: Floor on the deceleration shift magnitude (m/s^2) in the ``sigma_t`` term, so
#: an almost imperceptible regime change does not yield an absurdly tight sigma
#: (and avoids divide-by-zero).
MIN_DECEL_SHIFT_MPS2: Final[float] = 0.05


@dataclass(frozen=True)
class ChangePointSignalConfig:
    """Minimal :class:`~tdz.config.schema.SignalsConfig`-shaped smoothing config.

    Supplies exactly the attributes :func:`smoothed_derivatives` reads. The
    default uses a Savitzky-Golay-style local polynomial with the smallest
    window the smoother allows (5 samples, Req 16.2) and a quadratic order, which
    keeps the deceleration knee sharp for change-point detection while still
    suppressing single-sample noise. Externalizable: an estimator may be
    constructed with a different config (e.g. the project's resolved
    ``SignalsConfig``) without code changes.
    """

    smoothing_method: str = "savgol"
    savgol_window_samples: int = 5
    savgol_poly_order: int = 2
    gp_length_scale_s: float = 8.0
    gp_noise_variance: float = 0.5


#: Default smoothing configuration for the deceleration/jerk signals.
DEFAULT_SIGNAL_CONFIG: Final[ChangePointSignalConfig] = ChangePointSignalConfig()


@dataclass(frozen=True)
class DecelSignal:
    """Smoothed deceleration (and jerk) sampled on the valid velocity timebase.

    Attributes
    ----------
    times:
        Sorted epoch-second sample times with a finite smoothed deceleration.
    deceleration_mps2:
        Smoothed deceleration at ``times`` (m/s^2; negative while slowing).
    jerk_mps3:
        Smoothed jerk at ``times`` (m/s^3); ``NaN`` where unavailable.
    jerk_uncertainty:
        1-sigma of ``jerk_mps3`` (m/s^3).
    cadence_s:
        Median inter-sample spacing (seconds).
    derivative:
        The full :class:`~tdz.signals.derivatives.DerivativeResult` (carries the
        reliability flag / reported window for diagnostics).
    """

    times: np.ndarray
    deceleration_mps2: np.ndarray
    jerk_mps3: np.ndarray
    jerk_uncertainty: np.ndarray
    cadence_s: float
    derivative: DerivativeResult


def median_cadence(times: np.ndarray) -> float:
    """Median inter-sample spacing of a (possibly unsorted) time array (seconds)."""
    finite = np.asarray(times, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return 0.0
    return float(np.median(np.diff(np.sort(finite))))


def prepare_decel_signal(
    flight, config: ChangePointSignalConfig = DEFAULT_SIGNAL_CONFIG
) -> tuple[Optional[DecelSignal], Optional[FailureReason]]:
    """Build the smoothed-deceleration signal a detector runs on.

    Returns ``(signal, None)`` on success or ``(None, reason)`` when the flight
    cannot support a change-point detector:

    * :attr:`FailureReason.NO_GROUNDSPEED` -- groundspeed is missing entirely.
    * :attr:`FailureReason.INSUFFICIENT_SAMPLES` -- fewer than
      :data:`MIN_SAMPLES_FOR_CHANGEPOINT` valid groundspeed samples, or the
      smoother could not fit a finite deceleration at enough points.
    """
    times = np.asarray(flight.velocity_times, dtype=float)
    gs_kt = np.asarray(flight.groundspeeds, dtype=float)

    if gs_kt.size == 0 or np.all(np.isnan(gs_kt)):
        return None, FailureReason.NO_GROUNDSPEED

    valid = np.isfinite(times) & np.isfinite(gs_kt)
    if int(np.count_nonzero(valid)) < MIN_SAMPLES_FOR_CHANGEPOINT:
        return None, FailureReason.INSUFFICIENT_SAMPLES

    deriv = smoothed_derivatives(times, gs_kt, config)

    finite = np.isfinite(deriv.times) & np.isfinite(deriv.deceleration_mps2)
    if int(np.count_nonzero(finite)) < MIN_SAMPLES_FOR_CHANGEPOINT:
        return None, FailureReason.INSUFFICIENT_SAMPLES

    t = deriv.times[finite]
    d = deriv.deceleration_mps2[finite]
    j = deriv.jerk_mps3[finite]
    ju = deriv.jerk_uncertainty[finite]

    order = np.argsort(t)
    t = t[order]
    d = d[order]
    j = j[order]
    ju = ju[order]

    return (
        DecelSignal(
            times=t,
            deceleration_mps2=d,
            jerk_mps3=j,
            jerk_uncertainty=ju,
            cadence_s=median_cadence(t),
            derivative=deriv,
        ),
        None,
    )


def subsample_transition_time(
    times: np.ndarray,
    signal: np.ndarray,
    change_index: int,
    mean_before: float,
    mean_after: float,
) -> float:
    """Refine a change *index* to a sub-sample transition *time*.

    Finds where ``signal`` crosses the half-level ``(mean_before + mean_after)/2``
    in the interval bracketing ``change_index`` and linearly interpolates the
    crossing time. The crossing nearest ``change_index`` is used. Falls back to
    the midpoint of the bracketing sample times when the two means coincide or no
    sign change is found (e.g. a flat signal).

    ``change_index`` is the first index of the post-change regime, so the
    transition lies in ``[times[change_index - 1], times[change_index]]``.
    """
    t = np.asarray(times, dtype=float)
    x = np.asarray(signal, dtype=float)
    n = t.size
    k = int(np.clip(change_index, 1, n - 1))

    bracket_mid = 0.5 * (t[k - 1] + t[k])
    target = 0.5 * (mean_before + mean_after)
    if not np.isfinite(target) or mean_before == mean_after:
        return float(bracket_mid)

    # Half-level crossings: consecutive samples straddling ``target``.
    g = x - target
    sign_change = np.where(np.sign(g[:-1]) * np.sign(g[1:]) < 0.0)[0]
    if sign_change.size == 0:
        return float(bracket_mid)

    # Choose the crossing whose left index is nearest the detected change.
    i = int(sign_change[np.argmin(np.abs(sign_change - (k - 1)))])
    denom = g[i + 1] - g[i]
    if denom == 0.0:
        return float(0.5 * (t[i] + t[i + 1]))
    frac = float(-g[i] / denom)
    frac = min(max(frac, 0.0), 1.0)
    return float(t[i] + frac * (t[i + 1] - t[i]))


def segment_residual_std(signal: np.ndarray, change_index: int) -> float:
    """Within-segment std of a signal split into two constant-mean regimes.

    The residual after removing each regime's mean is the detector's noise scale;
    it feeds the ``sigma_t`` term in :func:`localization_sigma`. Returns ``0.0``
    for a perfectly clean two-level signal.
    """
    x = np.asarray(signal, dtype=float)
    n = x.size
    k = int(np.clip(change_index, 1, n - 1))
    before = x[:k]
    after = x[k:]
    resid = np.concatenate([before - np.mean(before), after - np.mean(after)])
    if resid.size <= 1:
        return 0.0
    return float(np.sqrt(np.mean(resid ** 2)))


def localization_sigma(
    cadence_s: float,
    residual_std_mps2: float,
    shift_mps2: float,
    *,
    cadence_fraction: float = CADENCE_SIGMA_FRACTION,
    min_sigma_s: float = MIN_SIGMA_T_S,
) -> float:
    """Derive the change-point time uncertainty (seconds); see module docstring."""
    shift = max(abs(shift_mps2), MIN_DECEL_SHIFT_MPS2)
    detector_term = cadence_s * abs(residual_std_mps2) / shift
    cadence_floor = cadence_fraction * cadence_s
    sigma = float(np.hypot(detector_term, cadence_floor))
    return max(sigma, min_sigma_s)
