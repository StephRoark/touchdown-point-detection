"""Deceleration-knee physics estimator (Task 12.1).

Fits a continuous piecewise-linear model to the **raw groundspeed-vs-time**
series and takes the breakpoint -- the transition from gentle approach
deceleration to sharp ground-roll braking -- as the candidate touchdown time
(design "Deceleration-Knee Estimator"; Req 5.1, 6.1, 16.1). The segmentation
itself is **not** re-implemented here: this estimator *wraps*
:func:`tdz.signals.segmented.fit_segmented_groundspeed` (Task 11.1) and adds the
estimator contract, the uncertainty derivation, and aircraft-type priors.

Why the velocity stream
-----------------------
The knee lives entirely in the velocity stream, so this estimator is **immune to
altitude-source issues and async-timestamp problems** (design): it consumes only
:attr:`FlightRecord.velocity_times` and :attr:`FlightRecord.groundspeeds` and
never touches geometric altitude or position. It is therefore the speed/position
anchor that still runs on sources lacking geometric altitude (Task 12.5 FR24
case).

How ``t_td`` is computed
------------------------
``t_td`` is the segmented fit's primary breakpoint (the knot with the steepest
deceleration increase). With ``n_segments=2`` this is the single breakpoint;
with ``n_segments=3`` (approach / transition / rollout) it is the steepest of
the two knots (chosen inside :func:`fit_segmented_groundspeed`).

How ``sigma_t`` is computed
---------------------------
The breakpoint's time uncertainty is derived from the fit, combining two terms
in quadrature:

* a **fit term** ``residual_rms / |slope_drop|`` -- a vertical speed scatter of
  ``residual_rms`` (m/s) around a knee whose slope changes by ``slope_drop``
  (m/s^2) maps to a horizontal (time) ambiguity of ``residual_rms /
  slope_drop`` seconds. A sharp knee (large slope drop) localizes the breakpoint
  tightly; a soft knee or noisy speeds widen it; and
* a **cadence floor** ``CADENCE_SIGMA_FRACTION * median_dt`` -- the breakpoint
  cannot be located more finely than a fraction of the sample spacing.

Aircraft-type priors
--------------------
Aircraft differ in plausible approach speed (Vref scales with weight/type) and
rollout deceleration (braking/thrust-reverser authority). The priors here are a
per-class :class:`DecelPrior` table giving plausible ranges for the approach
speed (m/s) at the knee and the rollout deceleration magnitude (m/s^2). They are
**constraints/plausibility checks**, not hard clamps: the segmented fit is data
driven, and the prior is used to (a) flag the estimate low-confidence when the
fitted approach speed or rollout deceleration falls outside the plausible range
for the type, and (b) record the prior influence in diagnostics (Req 6.1).

Provenance of the prior values: the defaults are broad, physically-grounded
envelopes (approach groundspeeds roughly 50-180 kt across regional->widebody;
rollout decelerations roughly 0.7-4.0 m/s^2) rather than tuned per-tail values.
They are exposed as a constructor parameter / module table so they can be moved
into configuration without code changes (Req 20); the defaults live in
:data:`DEFAULT_DECEL_PRIORS` and are documented there.

Units: SI throughout (m/s, m/s^2, seconds). Groundspeed knots->m/s conversion is
handled inside the segmented fit; the priors are stated in SI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Optional

import numpy as np

from tdz.estimators.physics.base import (
    CONFIDENCE_LOW,
    CONFIDENCE_NORMAL,
    PhysicsEstimator,
    failed_estimate,
    make_estimate,
)
from tdz.models import FailureReason, FlightRecord, TDEstimate
from tdz.signals.segmented import fit_segmented_groundspeed
from tdz.timebase.interpolation import KNOTS_TO_MPS

__all__ = [
    "DecelPrior",
    "DEFAULT_DECEL_PRIORS",
    "GLOBAL_DECEL_PRIOR",
    "resolve_decel_prior",
    "DecelKneeEstimator",
    "METHOD_NAME",
    "CADENCE_SIGMA_FRACTION",
    "MIN_SLOPE_DROP_MPS2",
    "MIN_SIGMA_T_S",
]

#: Estimator identifier (matches the ``decel_knee`` id in ``ALLOWED_ESTIMATORS``).
METHOD_NAME: Final[str] = "decel_knee"

#: Fraction of the median sample spacing used as the irreducible cadence floor
#: on the breakpoint time uncertainty (the knee cannot be located more finely
#: than a fraction of the cadence). Documented tunable; overridable per instance.
CADENCE_SIGMA_FRACTION: Final[float] = 0.5

#: Floor on the slope drop (m/s^2) used in the ``sigma_t`` fit term so an almost
#: imperceptible knee does not produce a divide-by-zero / absurdly tight sigma.
MIN_SLOPE_DROP_MPS2: Final[float] = 0.05

#: Absolute floor on the reported ``sigma_t`` (seconds): the breakpoint is never
#: claimed more precise than this regardless of how clean the fit looks.
MIN_SIGMA_T_S: Final[float] = 0.25


@dataclass(frozen=True)
class DecelPrior:
    """Plausible approach-speed / rollout-deceleration envelope for a class.

    All SI. ``approach_speed_*_mps`` bound the groundspeed at the knee (the end
    of the approach segment); ``rollout_decel_*_mps2`` bound the **magnitude** of
    the ground-roll deceleration (positive numbers; the fitted slope is
    negative, so its magnitude is compared). These are envelopes for
    plausibility, not point estimates.
    """

    approach_speed_min_mps: float
    approach_speed_max_mps: float
    rollout_decel_min_mps2: float
    rollout_decel_max_mps2: float


#: Broad, physically-grounded global envelope used when no class-specific prior
#: applies. Approach groundspeed ~50-185 kt (25.7-95.2 m/s); rollout
#: deceleration magnitude ~0.6-4.5 m/s^2.
GLOBAL_DECEL_PRIOR: Final[DecelPrior] = DecelPrior(
    approach_speed_min_mps=25.0,
    approach_speed_max_mps=95.0,
    rollout_decel_min_mps2=0.6,
    rollout_decel_max_mps2=4.5,
)

#: Per-aircraft-class priors (see :data:`GLOBAL_DECEL_PRIOR` for the fallback).
#: Regional types fly slower approaches with modest braking; widebodies fly
#: faster approaches with strong braking authority. Externalizable to config.
DEFAULT_DECEL_PRIORS: Final[Mapping[str, DecelPrior]] = {
    "regional": DecelPrior(25.0, 75.0, 0.7, 3.5),
    "narrowbody": DecelPrior(30.0, 85.0, 0.8, 4.0),
    "widebody": DecelPrior(33.0, 95.0, 0.8, 4.5),
}


def resolve_decel_prior(
    aircraft_class: Optional[str],
    priors: Mapping[str, DecelPrior] = DEFAULT_DECEL_PRIORS,
    *,
    default: DecelPrior = GLOBAL_DECEL_PRIOR,
) -> DecelPrior:
    """Resolve the deceleration prior for an aircraft class.

    Returns the class-specific prior when ``aircraft_class`` is known and
    present in ``priors``; otherwise the broad ``default`` global envelope. The
    caller (ingest/fusion) supplies the class; the lever-arm table is the
    canonical type->class source, but the prior is intentionally class-grained.
    """
    if aircraft_class is not None:
        prior = priors.get(aircraft_class)
        if prior is not None:
            return prior
    return default


class DecelKneeEstimator(PhysicsEstimator):
    """Estimate ``t_td`` from the groundspeed deceleration knee (Req 5.1, 6.1).

    Parameters
    ----------
    n_segments:
        ``2`` (default) or ``3`` segments for the piecewise fit.
    aircraft_class:
        Optional aircraft class used to select a :class:`DecelPrior`. When
        ``None`` the global envelope is used.
    priors:
        Per-class prior table (defaults to :data:`DEFAULT_DECEL_PRIORS`).
    default_prior:
        Fallback prior when the class is unknown (defaults to
        :data:`GLOBAL_DECEL_PRIOR`).
    cadence_sigma_fraction, min_sigma_t_s:
        Overridable ``sigma_t`` tunables (see module docstring / constants).
    """

    method_name = METHOD_NAME

    def __init__(
        self,
        *,
        n_segments: int = 2,
        aircraft_class: Optional[str] = None,
        priors: Mapping[str, DecelPrior] = DEFAULT_DECEL_PRIORS,
        default_prior: DecelPrior = GLOBAL_DECEL_PRIOR,
        cadence_sigma_fraction: float = CADENCE_SIGMA_FRACTION,
        min_sigma_t_s: float = MIN_SIGMA_T_S,
    ) -> None:
        if n_segments not in (2, 3):
            raise ValueError(f"n_segments must be 2 or 3, got {n_segments}")
        self.n_segments = n_segments
        self.aircraft_class = aircraft_class
        self.priors = priors
        self.default_prior = default_prior
        self.cadence_sigma_fraction = float(cadence_sigma_fraction)
        self.min_sigma_t_s = float(min_sigma_t_s)

    def _raw_estimate(self, flight: FlightRecord) -> TDEstimate:
        times = np.asarray(flight.velocity_times, dtype=float)
        gs_kt = np.asarray(flight.groundspeeds, dtype=float)

        # No groundspeed at all -> cannot run the velocity-stream estimator.
        if gs_kt.size == 0 or np.all(np.isnan(gs_kt)):
            return failed_estimate(self.method_name, FailureReason.NO_GROUNDSPEED)

        try:
            fit = fit_segmented_groundspeed(times, gs_kt, n_segments=self.n_segments)
        except ValueError:
            # Too few valid samples to support the requested segmentation.
            return failed_estimate(self.method_name, FailureReason.INSUFFICIENT_SAMPLES)

        prior = resolve_decel_prior(
            self.aircraft_class, self.priors, default=self.default_prior
        )

        # The steepest slope drop at the primary breakpoint drives both the
        # plausibility check and the sigma_t fit term.
        bp = fit.breakpoint_time
        bp_index = (
            int(np.argmin(np.abs(np.asarray(fit.breakpoint_times) - bp)))
            if fit.breakpoint_times
            else 0
        )
        slope_before = fit.slopes_mps2[bp_index]
        slope_after = fit.slopes_mps2[bp_index + 1]
        slope_drop = slope_before - slope_after  # positive when braking steepens
        rollout_decel_mps2 = abs(slope_after)

        # Approach groundspeed at the knee, reconstructed from the fitted
        # pre-knee segment (intercept + slope * local-time-at-breakpoint).
        t0 = float(np.nanmin(times[~np.isnan(gs_kt)])) if np.any(~np.isnan(gs_kt)) else 0.0
        approach_speed_mps = float(fit.intercepts_mps[bp_index] + slope_before * (bp - t0))

        prior_eval = self._evaluate_prior(prior, approach_speed_mps, rollout_decel_mps2)
        sigma_t = self._sigma_t(fit.residual_rms_mps, slope_drop, times)

        confidence = CONFIDENCE_NORMAL
        reason: Optional[FailureReason] = None
        if not prior_eval["within_prior"]:
            # The fit is physically implausible for the type: keep the estimate
            # (data-driven) but flag low-confidence so fusion down-weights it.
            confidence = CONFIDENCE_LOW
            reason = FailureReason.ESTIMATOR_DISAGREEMENT

        diagnostics = {
            "breakpoint_time": bp,
            "breakpoint_times": list(fit.breakpoint_times),
            "segment_decelerations_mps2": list(fit.slopes_mps2),
            "segment_intercepts_mps": list(fit.intercepts_mps),
            "fit_residual_rms_mps": fit.residual_rms_mps,
            "n_segments": fit.n_segments,
            "n_samples": fit.n_samples,
            "slope_drop_mps2": slope_drop,
            "approach_speed_mps": approach_speed_mps,
            "rollout_decel_mps2": rollout_decel_mps2,
            "prior_influence": prior_eval,
        }
        return make_estimate(
            t_td=bp,
            sigma_t=sigma_t,
            confidence=confidence,
            method_name=self.method_name,
            diagnostics=diagnostics,
            reason=reason,
        )

    def _evaluate_prior(
        self, prior: DecelPrior, approach_speed_mps: float, rollout_decel_mps2: float
    ) -> dict:
        """Check the fitted quantities against the prior; record the influence."""
        speed_ok = (
            prior.approach_speed_min_mps <= approach_speed_mps <= prior.approach_speed_max_mps
        )
        decel_ok = (
            prior.rollout_decel_min_mps2 <= rollout_decel_mps2 <= prior.rollout_decel_max_mps2
        )
        return {
            "within_prior": bool(speed_ok and decel_ok),
            "approach_speed_within_prior": bool(speed_ok),
            "rollout_decel_within_prior": bool(decel_ok),
            "approach_speed_prior_mps": (
                prior.approach_speed_min_mps,
                prior.approach_speed_max_mps,
            ),
            "rollout_decel_prior_mps2": (
                prior.rollout_decel_min_mps2,
                prior.rollout_decel_max_mps2,
            ),
        }

    def _sigma_t(
        self, residual_rms_mps: float, slope_drop_mps2: float, times: np.ndarray
    ) -> float:
        """Derive the breakpoint time uncertainty from the fit (see module docstring)."""
        drop = max(abs(slope_drop_mps2), MIN_SLOPE_DROP_MPS2)
        fit_term = residual_rms_mps / drop

        finite = times[np.isfinite(times)]
        if finite.size >= 2:
            median_dt = float(np.median(np.diff(np.sort(finite))))
        else:
            median_dt = 0.0
        cadence_floor = self.cadence_sigma_fraction * median_dt

        sigma = float(np.hypot(fit_term, cadence_floor))
        return max(sigma, self.min_sigma_t_s)
