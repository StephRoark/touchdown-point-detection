"""Vertical flare-crossing physics estimator (Task 12.2).

Estimates touchdown by fitting a **joint glideslope-plus-flare model over an
extended region** (default ~250 ft above the runway down to the surface) to the
geometric-altitude-vs-time profile and solving for the time the **main gear**
reaches the runway (design "Vertical Flare-Crossing Estimator"; Req 17.1-17.5).

Why an extended region and a curved model (Req 17.1)
----------------------------------------------------
At the 4-5 s ADS-B cadence and typical descent rates only one or two
geometric-altitude samples fall below 50 ft, so a flare-only (sub-50-ft) fit is
almost always under-determined. The extended region supplies 3-5 samples, and a
**curved** flare term (a quadratic in time, which captures the flare flattening)
prevents the long bias that a single straight-line fit through the flare would
produce. The model is

    h(t) = c0 + c1 * t + c2 * t^2     (heights above the runway, metres HAE)

fit by ordinary least squares over the extended-region samples; ``c2`` is the
flare curvature. (An exponential flare is an alternative; the quadratic is used
here because it is numerically stable and needs no nonlinear solve.) The
touchdown time is the model's crossing of the **main-gear target height** ``V``
(see below), taking the physically sensible descending root.

HAE datum and the lever arm (Req 17.2, 17.4)
--------------------------------------------
The estimator works entirely in HAE. The runway threshold elevation is converted
to HAE by the **deterministic** geoid correction
(:func:`tdz.geo.resolve_threshold_elevation_hae`); heights are
``geometric_altitude - threshold_hae``. The antenna-to-main-gear vertical offset
``V`` is **added to the crossing target**: the antenna sits ``V`` above the gear,
so the gear contacts the runway when the antenna height-above-runway equals
``V``. Solving ``h(t) = V`` therefore yields **main-gear** contact, not antenna
contact (Req 17.4). ``V`` comes from the per-type lever arm (Task 6); the
estimator accepts it directly or via a :class:`~tdz.geo.LeverArmCorrection`.

Geoid/datum vs residual sensor bias kept separate (Req 17.3)
------------------------------------------------------------
The deterministic geoid conversion is applied first (above). Only **then**, and
only if the geoid-corrected **high-approach** samples (well above the flare,
where the profile is ~linear glideslope) deviate from the fitted curve by more
than ``residual_bias_trigger_ft``, is a residual static sensor bias estimated
and subtracted. The bias is the **median residual of the high-approach samples
only** about the joint model -- **flare-region samples are never used** for the
bias (they are curving, and would let real flare dynamics leak into the bias
term). This keeps the deterministic datum offset and the empirical bias as two
separate, auditable steps.

Disabling / starvation (Req 17.2, 17.5, 8.8)
--------------------------------------------
* If the source provides no geometric altitude (``geometric_altitudes`` is empty
  or all-NaN), the estimator **self-disables**: it returns a failed estimate
  with :attr:`FailureReason.GEOMETRIC_ALT_UNAVAILABLE`, regardless of upstream
  source gating. Barometric altitude is never substituted (Req 8.8).
* If fewer than ``min_samples_in_fit_region`` (default 3) geometric-altitude
  samples lie in the **extended** fit region, it returns failed with
  :attr:`FailureReason.INSUFFICIENT_FLARE_SAMPLES` and defers to other
  estimators rather than fitting an under-constrained curve (Req 17.5). Note the
  region is the *extended* one: a landing with only one sample below 50 ft still
  fits, as long as >=3 samples fall in the ~0-250 ft band (Task 12.5).

How ``t_td`` and ``sigma_t`` are computed
-----------------------------------------
``t_td`` is the model's crossing of ``h = V`` (the descending real root nearest
the end of the descent). If the fitted profile has no descending crossing (only
an ascending one, e.g. a climbing segment), or the crossing lies more than
:data:`MAX_EXTRAPOLATION_CADENCES` median sample spacings outside the fitted
data, the estimator defers (failed estimate) rather than reporting an
extrapolation artifact. ``sigma_t`` combines the fit's height residual RMS
mapped through the descent rate at the crossing (``residual_rms / |dh/dt|``,
metres / (m/s) = s) with a cadence floor, mirroring the decel-knee derivation.

Diagnostics: flare model parameters, crossing time, geoid undulation applied,
residual sensor-bias estimate (and whether applied), number of samples in the
extended region, and the datum used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

import numpy as np

from tdz.config.schema import VerticalCrossingConfig
from tdz.estimators.physics.base import (
    CONFIDENCE_NORMAL,
    PhysicsEstimator,
    failed_estimate,
    make_estimate,
)
from tdz.geo.datum import resolve_threshold_elevation_hae
from tdz.geo.errors import DatumUnresolvedError
from tdz.geo.lever_arm import LeverArmCorrection
from tdz.models import FailureReason, FlightRecord, TDEstimate

__all__ = [
    "FlareCrossingEstimator",
    "METHOD_NAME",
    "FT_TO_M",
    "FLARE_ONSET_M",
    "CADENCE_SIGMA_FRACTION",
    "MIN_DESCENT_RATE_MPS",
    "MIN_SIGMA_T_S",
    "MAX_EXTRAPOLATION_CADENCES",
    "default_vertical_crossing_config",
]

#: Estimator identifier (matches the ``flare_crossing`` id in ``ALLOWED_ESTIMATORS``).
METHOD_NAME: Final[str] = "flare_crossing"

#: Feet -> metres (exact). The config fit-region bounds and the bias trigger are
#: stated in feet (analyst-facing); they are converted to SI here.
FT_TO_M: Final[float] = 0.3048

#: Height-above-runway (metres) below which a sample is considered to be in the
#: flare and is therefore EXCLUDED from the residual-bias estimate (Req 17.3:
#: bias never uses flare-region samples). 50 ft is the conventional flare onset.
FLARE_ONSET_M: Final[float] = 50.0 * FT_TO_M

#: Fraction of the median sample spacing used as the cadence floor on sigma_t.
CADENCE_SIGMA_FRACTION: Final[float] = 0.5

#: Floor on the descent rate (m/s) used in the sigma_t mapping so a near-zero
#: dh/dt at the crossing does not blow up the time uncertainty.
MIN_DESCENT_RATE_MPS: Final[float] = 0.3

#: Extrapolation horizon for the crossing solution, in multiples of the median
#: sample spacing of the fit-region samples. A crossing farther than this
#: outside the fitted data is an extrapolation artifact (e.g. a near-flat
#: profile whose fitted curve only reaches the target height far beyond the
#: data), not a touchdown observation; the estimator defers instead.
MAX_EXTRAPOLATION_CADENCES: Final[float] = 3.0

#: Absolute floor on the reported sigma_t (seconds).
MIN_SIGMA_T_S: Final[float] = 0.25


def default_vertical_crossing_config() -> VerticalCrossingConfig:
    """Return the design-default vertical-crossing config (Req 17 defaults).

    ``fit_region_upper_ft=250``, ``fit_region_lower_ft=0``,
    ``min_samples_in_fit_region=3``, ``residual_bias_trigger_ft=15``.
    """
    return VerticalCrossingConfig(
        fit_region_upper_ft=250.0,
        fit_region_lower_ft=0.0,
        min_samples_in_fit_region=3,
        residual_bias_trigger_ft=15.0,
    )


@dataclass(frozen=True)
class _CrossingSolution:
    """Internal: the solved crossing plus fit quantities (all SI)."""

    t_cross: float
    coeffs: np.ndarray            # [c0, c1, c2] of c0 + c1 t + c2 t^2 (local time)
    residual_rms_m: float
    descent_rate_mps: float       # |dh/dt| at the crossing


class FlareCrossingEstimator(PhysicsEstimator):
    """Estimate ``t_td`` from the geometric-altitude flare crossing (Req 17).

    Parameters
    ----------
    config:
        Vertical-crossing fit-region / bias-trigger config. Defaults to
        :func:`default_vertical_crossing_config`.
    vertical_offset_m:
        Antenna-to-main-gear vertical offset ``V`` (metres), added to the
        crossing target so the solution is main-gear contact (Req 17.4). Ignored
        when ``lever_arm_correction`` is supplied.
    lever_arm_correction:
        Optional :class:`~tdz.geo.LeverArmCorrection`; its
        ``altitude_target_shift_m`` supplies ``V``.
    geodesy_config:
        Optional geodesy config passed to the datum resolver.
    """

    method_name = METHOD_NAME

    def __init__(
        self,
        config: Optional[VerticalCrossingConfig] = None,
        *,
        vertical_offset_m: float = 0.0,
        lever_arm_correction: Optional[LeverArmCorrection] = None,
        geodesy_config: object = None,
        cadence_sigma_fraction: float = CADENCE_SIGMA_FRACTION,
        min_sigma_t_s: float = MIN_SIGMA_T_S,
    ) -> None:
        self.config = config or default_vertical_crossing_config()
        if lever_arm_correction is not None:
            self.vertical_offset_m = float(lever_arm_correction.altitude_target_shift_m)
        else:
            self.vertical_offset_m = float(vertical_offset_m)
        self.geodesy_config = geodesy_config
        self.cadence_sigma_fraction = float(cadence_sigma_fraction)
        self.min_sigma_t_s = float(min_sigma_t_s)

    def _raw_estimate(self, flight: FlightRecord) -> TDEstimate:
        geo_alt = np.asarray(flight.geometric_altitudes, dtype=float)
        times = np.asarray(flight.position_times, dtype=float)

        # (Req 17.2 / 8.8) Self-disable when the source lacks geometric altitude.
        if geo_alt.size == 0 or np.all(np.isnan(geo_alt)):
            return failed_estimate(
                self.method_name, FailureReason.GEOMETRIC_ALT_UNAVAILABLE
            )

        # (Req 11.2 / 17.2) Deterministic geoid/datum correction to HAE.
        try:
            threshold_hae = resolve_threshold_elevation_hae(
                flight.runway, self.geodesy_config
            )
        except DatumUnresolvedError:
            return failed_estimate(self.method_name, FailureReason.DATUM_UNRESOLVED)

        undulation = float(getattr(flight.runway, "geoid_undulation_m", float("nan")))

        valid = ~(np.isnan(geo_alt) | np.isnan(times))
        t = times[valid]
        heights = geo_alt[valid] - threshold_hae  # height above runway (HAE)

        upper_m = self.config.fit_region_upper_ft * FT_TO_M
        lower_m = self.config.fit_region_lower_ft * FT_TO_M
        in_region = (heights >= lower_m) & (heights <= upper_m)
        n_in_region = int(np.count_nonzero(in_region))

        # (Req 17.5) Under-determined: defer rather than fit a bad curve.
        if n_in_region < self.config.min_samples_in_fit_region:
            return failed_estimate(
                self.method_name,
                FailureReason.INSUFFICIENT_FLARE_SAMPLES,
                diagnostics={
                    "n_samples_in_fit_region": n_in_region,
                    "fit_region_ft": (
                        self.config.fit_region_lower_ft,
                        self.config.fit_region_upper_ft,
                    ),
                    "geoid_undulation_m": undulation,
                    "datum_used": "HAE",
                },
            )

        t_region = t[in_region]
        h_region = heights[in_region]
        order = np.argsort(t_region)
        t_region = t_region[order]
        h_region = h_region[order]

        t0 = float(t_region[0])
        u = t_region - t0  # local time frame for conditioning

        # (Req 17.3) Residual static sensor bias from HIGH-APPROACH samples only.
        bias_m, bias_applied = self._estimate_residual_bias(u, h_region)
        h_corrected = h_region - bias_m if bias_applied else h_region

        solution = self._solve_crossing(u, h_corrected, t0)
        if solution is None:
            # Curve fit succeeded but no physical descending crossing of h=V was
            # found in/after the region -> defer.
            return failed_estimate(
                self.method_name,
                FailureReason.INSUFFICIENT_FLARE_SAMPLES,
                diagnostics={
                    "n_samples_in_fit_region": n_in_region,
                    "geoid_undulation_m": undulation,
                    "datum_used": "HAE",
                    "reason_detail": (
                        "no descending crossing of target height within the "
                        "extrapolation horizon"
                    ),
                },
            )

        sigma_t = self._sigma_t(solution.residual_rms_m, solution.descent_rate_mps, t)

        diagnostics = {
            "crossing_time": solution.t_cross,
            "flare_model": "quadratic",
            "flare_coeffs_local": list(map(float, solution.coeffs)),
            "flare_curvature_c2_mps2": float(2.0 * solution.coeffs[2]),
            "crossing_target_height_m": self.vertical_offset_m,
            "vertical_offset_m": self.vertical_offset_m,
            "threshold_elevation_hae_m": threshold_hae,
            "geoid_undulation_m": undulation,
            "datum_used": "HAE",
            "residual_bias_estimate_m": bias_m,
            "residual_bias_applied": bias_applied,
            "residual_bias_trigger_ft": self.config.residual_bias_trigger_ft,
            "n_samples_in_fit_region": n_in_region,
            "fit_region_ft": (
                self.config.fit_region_lower_ft,
                self.config.fit_region_upper_ft,
            ),
            "fit_residual_rms_m": solution.residual_rms_m,
            "descent_rate_at_crossing_mps": solution.descent_rate_mps,
        }
        return make_estimate(
            t_td=solution.t_cross,
            sigma_t=sigma_t,
            confidence=CONFIDENCE_NORMAL,
            method_name=self.method_name,
            diagnostics=diagnostics,
        )

    def _estimate_residual_bias(
        self, u: np.ndarray, heights: np.ndarray
    ) -> tuple[float, bool]:
        """Estimate a static sensor bias from high-approach samples only.

        Fits the joint quadratic to all region samples, then takes the **median
        residual of the high-approach samples** (height >= :data:`FLARE_ONSET_M`)
        as the candidate bias. It is applied only if its magnitude exceeds
        ``residual_bias_trigger_ft`` (Req 17.3). Flare-region samples (below the
        flare onset) never contribute to the bias. Returns ``(bias_m, applied)``.
        """
        high = heights >= FLARE_ONSET_M
        if np.count_nonzero(high) < 2:
            # Not enough high-approach samples to characterize a bias.
            return 0.0, False

        coeffs = np.polyfit(u, heights, 2)
        model = np.polyval(coeffs, u)
        residuals_high = heights[high] - model[high]
        bias = float(np.median(residuals_high))

        trigger_m = self.config.residual_bias_trigger_ft * FT_TO_M
        return bias, bool(abs(bias) > trigger_m)

    def _solve_crossing(
        self, u: np.ndarray, heights: np.ndarray, t0: float
    ) -> Optional[_CrossingSolution]:
        """Fit the quadratic flare model and solve ``h(t) = V`` (main gear)."""
        coeffs = np.polyfit(u, heights, 2)  # [c2, c1, c0] (numpy order)
        c2, c1, c0 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        target = self.vertical_offset_m

        fitted = np.polyval(coeffs, u)
        residual_rms = float(np.sqrt(np.mean((heights - fitted) ** 2)))

        roots = self._real_roots(c2, c1, c0 - target)
        if not roots:
            return None

        # Take the DESCENDING crossing nearest the end of the fit region. An
        # ascending root (the fitted profile climbing through the target height)
        # is not a touchdown; if no descending root exists the estimator defers
        # (returns None) rather than reporting a physically meaningless time.
        u_start = float(u[0])
        u_end = float(u[-1])
        best_u: Optional[float] = None
        best_key = np.inf
        for r in roots:
            slope = 2.0 * c2 * r + c1  # dh/du
            if slope > 0:
                continue
            key = abs(r - u_end)
            if key < best_key:
                best_key = key
                best_u = r
        if best_u is None:
            return None

        # Reject a crossing outside the extrapolation horizon: the quadratic is
        # only trusted near the data it was fit to (design comment "within a
        # reasonable extrapolation horizon").
        if u.size >= 2:
            median_dt = float(np.median(np.diff(u)))
        else:
            median_dt = 0.0
        horizon = MAX_EXTRAPOLATION_CADENCES * median_dt
        if best_u < u_start - horizon or best_u > u_end + horizon:
            return None

        descent_rate = abs(2.0 * c2 * best_u + c1)
        # store coeffs in ascending order [c0, c1, c2] for readability
        return _CrossingSolution(
            t_cross=t0 + best_u,
            coeffs=np.array([c0, c1, c2], dtype=float),
            residual_rms_m=residual_rms,
            descent_rate_mps=descent_rate,
        )

    @staticmethod
    def _real_roots(a: float, b: float, c: float) -> list[float]:
        """Real roots of ``a x^2 + b x + c`` (handles the linear/degenerate case)."""
        if abs(a) < 1e-12:
            if abs(b) < 1e-12:
                return []
            return [-c / b]
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return []
        sq = float(np.sqrt(disc))
        return [(-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)]

    def _sigma_t(
        self, residual_rms_m: float, descent_rate_mps: float, times: np.ndarray
    ) -> float:
        """Map the height-fit residual through the descent rate to a time sigma."""
        rate = max(abs(descent_rate_mps), MIN_DESCENT_RATE_MPS)
        fit_term = residual_rms_m / rate

        finite = times[np.isfinite(times)]
        if finite.size >= 2:
            median_dt = float(np.median(np.diff(np.sort(finite))))
        else:
            median_dt = 0.0
        cadence_floor = self.cadence_sigma_fraction * median_dt

        sigma = float(np.hypot(fit_term, cadence_floor))
        return max(sigma, self.min_sigma_t_s)
