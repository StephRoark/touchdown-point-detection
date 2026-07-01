"""Split-conformal interval calibration (Task 19).

The fusion layer reports a *model-based* 90 % interval ``t_td +/- z * sigma_t``
under a Gaussian assumption (see :mod:`tdz.fusion.ensemble`). That assumption
is rarely exactly right: the per-estimator ``sigma`` values are themselves
estimates, so the raw intervals can be systematically too tight (unsafe
under-coverage) or too wide (uninformative). This module recalibrates the
interval width against held-out truth so the *empirical* coverage lands in the
required 85-95 % band (Req 4.3, 4.4).

Method: normalized split conformal
----------------------------------
Given a calibration split of flights with point estimates ``p_i``, truths
``y_i`` and reported 1-sigma widths ``sigma_i > 0``, the (normalized)
nonconformity score is

    s_i = |y_i - p_i| / sigma_i .

For a target coverage ``c`` the conformal multiplier is the finite-sample
corrected empirical quantile of the scores,

    q = Quantile_{ level }( {s_i} ),   level = ceil((n + 1) * c) / n ,

capped at ``1.0`` (when ``(n + 1) * c > n`` the guarantee needs the max score,
so ``q`` is the largest score). The calibrated interval for a new flight is then

    [ p - q * sigma ,  p + q * sigma ] .

Split conformal gives the finite-sample marginal-coverage guarantee
``P(y in interval) >= c`` when the calibration and test flights are
exchangeable, which is exactly the property Req 4.3/4.4 asks the harness to
demonstrate. Because the score is *normalized* by ``sigma`` the calibration
multiplies (rather than replaces) the model interval, so per-flight relative
uncertainty structure -- and the Task-19 widening applied on top -- is
preserved.

Distance vs time
----------------
The distance interval's truth is clock-independent (Req 4.4), so a distance
calibrator is fit on distance residuals with **no clock-offset correction**,
while the time calibrator is fit on clock-aligned time residuals. The two are
independent :class:`ConformalCalibrator` instances; this module is agnostic to
which quantity it is calibrating.

Units: this module is unit-agnostic -- points, truths and sigmas must share a
unit (seconds for time; meters or feet for distance). The multiplier is
dimensionless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from scipy.stats import norm

__all__ = ["gaussian_multiplier", "ConformalCalibrator"]


def gaussian_multiplier(coverage_target: float) -> float:
    """Two-sided standard-normal quantile for ``coverage_target``.

    This is the ``z`` that an *uncalibrated* interval uses
    (``~1.645`` for 0.90). It is the fallback multiplier when no calibration
    data is available, and the reference the conformal multiplier refines.
    """
    if not 0.0 < coverage_target < 1.0:
        raise ValueError(
            f"coverage_target must be in (0, 1), got {coverage_target!r}"
        )
    return float(norm.ppf((1.0 + coverage_target) / 2.0))


@dataclass(frozen=True)
class ConformalCalibrator:
    """A fitted (or Gaussian-fallback) interval-width multiplier.

    Attributes
    ----------
    multiplier:
        The dimensionless factor applied to a flight's reported ``sigma`` to
        obtain the calibrated interval half-width (``half_width = multiplier *
        sigma``).
    coverage_target:
        The coverage the multiplier was calibrated (or derived) for.
    n_calibration:
        Number of calibration residuals used. ``0`` marks a Gaussian-fallback
        calibrator (no calibration data), which uses :func:`gaussian_multiplier`.
    """

    multiplier: float
    coverage_target: float
    n_calibration: int = 0

    @classmethod
    def gaussian(cls, coverage_target: float) -> "ConformalCalibrator":
        """A fallback calibrator using the Gaussian ``z`` multiplier.

        Used when no calibration split is available; it reproduces the raw
        model interval so behavior is unchanged until real calibration data is
        supplied.
        """
        return cls(
            multiplier=gaussian_multiplier(coverage_target),
            coverage_target=coverage_target,
            n_calibration=0,
        )

    @classmethod
    def fit(
        cls,
        points: Sequence[float],
        truths: Sequence[float],
        sigmas: Sequence[float],
        *,
        coverage_target: float,
    ) -> "ConformalCalibrator":
        """Fit the conformal multiplier on a calibration split.

        Parameters
        ----------
        points, truths, sigmas:
            Equal-length sequences of point estimates, ground-truth values and
            reported 1-sigma widths (same unit for points/truths; ``sigma`` in
            that unit too). Entries with a non-finite value or ``sigma <= 0``
            are dropped (they carry no usable normalized score).
        coverage_target:
            Target empirical coverage ``c`` in ``(0, 1)``.

        Returns
        -------
        ConformalCalibrator
            A calibrator whose :attr:`multiplier` yields >= ``coverage_target``
            empirical coverage on exchangeable data. Falls back to the Gaussian
            multiplier when no valid calibration residual is available.
        """
        if not 0.0 < coverage_target < 1.0:
            raise ValueError(
                f"coverage_target must be in (0, 1), got {coverage_target!r}"
            )
        if not (len(points) == len(truths) == len(sigmas)):
            raise ValueError(
                "points, truths and sigmas must have equal length "
                f"(got {len(points)}, {len(truths)}, {len(sigmas)})"
            )

        scores: list[float] = []
        for p, y, s in zip(points, truths, sigmas):
            if not (math.isfinite(p) and math.isfinite(y) and math.isfinite(s)):
                continue
            if s <= 0.0:
                continue
            scores.append(abs(y - p) / s)

        if not scores:
            # No usable calibration data: reproduce the raw model interval.
            return cls.gaussian(coverage_target)

        scores.sort()
        n = len(scores)
        # Finite-sample corrected quantile level; ceil((n+1)*c)/n. When this
        # exceeds 1 the guarantee needs the largest score (index n-1).
        rank = math.ceil((n + 1) * coverage_target)
        index = min(rank, n) - 1
        multiplier = scores[index]
        return cls(
            multiplier=float(multiplier),
            coverage_target=coverage_target,
            n_calibration=n,
        )

    def half_width(self, sigma: float) -> float:
        """Calibrated interval half-width for a reported ``sigma``."""
        return self.multiplier * sigma
