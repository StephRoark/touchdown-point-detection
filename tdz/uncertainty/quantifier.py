"""Uncertainty quantification: calibrated, widened time & distance CIs (Task 19).

This module is the single place that turns a fused touchdown estimate into the
reported 90 % confidence intervals for **time** (seconds) and **along-runway
distance** (feet), satisfying Requirement 4 and the gap/starvation widening of
Requirement 9.

Pipeline position
-----------------
The fusion layer (:mod:`tdz.fusion.ensemble`) produces a fused ``t_td`` and a
1-sigma ``sigma_t`` (seconds) with a provisional Gaussian interval. This module
layers on top of that:

1. **Conformal calibration** (:mod:`tdz.uncertainty.conformal`) replaces the
   Gaussian ``z`` with a multiplier fit on the calibration split so empirical
   coverage lands in 85-95 % (Req 4.3, 4.4). Separate calibrators are used for
   time and distance because distance truth is clock-independent (Req 4.4).
2. **Data-driven widening** (:mod:`tdz.uncertainty.widening`) multiplies the
   calibrated width by gap-proportional (Req 9.2), post-transition-starvation
   (Req 9.6) and missing-lever-arm (Req 7.5, distance only) factors.
3. **Low-confidence flagging** (Req 4.5): when a widening condition or a
   degenerate input means a *reliable* interval cannot be produced, the estimate
   is flagged low-confidence and the (widened) interval is still emitted -- never
   suppressed.

Distance interval
-----------------
The distance interval is centered on the along-runway distance point supplied
by the mapping layer (Task 20) and its half-width is obtained by propagating the
time uncertainty through the touchdown groundspeed: ``sigma_distance = v_td *
sigma_t`` (a first-order delta-method propagation of "when" into "where"), then
calibrated and widened. Callers pass SI (meters, m/s); this module performs the
sole SI->feet conversion for the distance fields at its output boundary.

Units: inputs are SI (seconds, meters, m/s). Time CI fields are seconds;
distance CI fields are feet (converted here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Optional

from tdz.config.schema import UncertaintyConfig
from tdz.estimators.physics.base import CONFIDENCE_LOW, CONFIDENCE_NORMAL
from tdz.fusion.ensemble import CONFIDENCE_NO_ESTIMATE
from tdz.models import FailureReason, FlightRecord, FusedEstimate
from tdz.uncertainty.conformal import ConformalCalibrator
from tdz.uncertainty.widening import (
    gap_widening_factor,
    missing_lever_arm_widening_factor,
    post_transition_starved,
    starvation_widening_factor,
)

__all__ = ["UncertaintyResult", "UncertaintyQuantifier"]

#: International-foot to meter conversion (exact). Distance is computed in meters
#: internally and converted to feet at this module's output boundary, matching
#: the ``FT_TO_M`` constant used elsewhere (e.g. :mod:`tdz.geo.gates`).
FT_TO_M: Final[float] = 0.3048
M_TO_FT: Final[float] = 1.0 / FT_TO_M


@dataclass
class UncertaintyResult:
    """Calibrated, widened uncertainty for one flight.

    Time fields are seconds; distance fields are feet. ``confidence`` is one of
    ``"normal"`` / ``"low-confidence"`` (a ``"no-estimate"`` fused input yields
    ``None`` from :meth:`UncertaintyQuantifier.quantify`, never this record).
    ``diagnostics`` records the multipliers/factors applied for traceability.
    """

    t_td: float
    time_ci_90_lower_s: float
    time_ci_90_upper_s: float

    along_runway_distance_ft: float
    distance_ci_90_lower_ft: float
    distance_ci_90_upper_ft: float

    confidence: str
    reason_code: Optional[str]
    diagnostics: dict


class UncertaintyQuantifier:
    """Produce calibrated, widened 90 % time and distance CIs (Task 19).

    Parameters
    ----------
    config:
        The resolved ``uncertainty`` configuration (coverage target, gap and
        starvation widening knobs).
    time_calibrator, distance_calibrator:
        Optional fitted :class:`ConformalCalibrator` instances. When omitted, a
        Gaussian-``z`` fallback calibrator is derived from ``coverage_target``
        (so the reported interval reproduces the raw model interval until real
        calibration data is available).
    """

    def __init__(
        self,
        config: UncertaintyConfig,
        *,
        time_calibrator: Optional[ConformalCalibrator] = None,
        distance_calibrator: Optional[ConformalCalibrator] = None,
    ) -> None:
        self.config = config
        self.time_calibrator = time_calibrator or ConformalCalibrator.gaussian(
            config.coverage_target
        )
        self.distance_calibrator = distance_calibrator or ConformalCalibrator.gaussian(
            config.coverage_target
        )

    def quantify(
        self,
        fused: FusedEstimate,
        flight: FlightRecord,
        *,
        groundspeed_at_td_mps: float,
        along_runway_distance_m: float,
        lever_arm_missing: bool = False,
    ) -> Optional[UncertaintyResult]:
        """Compute calibrated, widened time & distance CIs for one flight.

        Returns ``None`` when the fused estimate is a no-estimate (there is no
        point estimate to bracket). For a ``"normal"`` or ``"low-confidence"``
        fused estimate an :class:`UncertaintyResult` is always returned with a
        valid interval (Property 6) -- when a reliable interval cannot be
        computed the estimate is flagged low-confidence and a widened interval is
        still emitted (Req 4.5), never suppressed.

        Parameters
        ----------
        fused:
            The fused estimate (``t_td``, ``sigma_t`` in seconds; confidence).
        flight:
            The flight record (position timestamps, on-ground transition) driving
            the gap and starvation widening.
        groundspeed_at_td_mps:
            Touchdown groundspeed (m/s) used to propagate time uncertainty into
            distance uncertainty.
        along_runway_distance_m:
            Along-runway distance point (meters) the distance interval centers on
            (from the mapping layer, Task 20).
        lever_arm_missing:
            ``True`` when a class-median lever arm was substituted (widens the
            distance interval; Req 7.5).
        """
        if fused.confidence == CONFIDENCE_NO_ESTIMATE:
            return None

        t_td = float(fused.t_td)
        sigma_t = float(fused.sigma_t)

        # Widening factors (>= 1.0). Gap and starvation widen both intervals;
        # the lever-arm factor widens distance only.
        gap_factor = gap_widening_factor(flight, t_td, self.config)
        starved = post_transition_starved(flight, self.config)
        starv_factor = starvation_widening_factor(flight, self.config)
        lever_factor = missing_lever_arm_widening_factor(
            lever_arm_missing, self.config
        )

        time_widen = gap_factor * starv_factor
        dist_widen = gap_factor * starv_factor * lever_factor

        # --- Confidence / reason resolution (Req 4.5, 7.5, 9.6) --------------
        confidence = fused.confidence
        reason_code = fused.reason_code

        def _downgrade(reason: str) -> None:
            nonlocal confidence, reason_code
            confidence = CONFIDENCE_LOW
            if reason_code is None:
                reason_code = reason

        if lever_arm_missing:
            _downgrade(FailureReason.MISSING_LEVER_ARM.value)
        if starved:
            _downgrade(FailureReason.NO_GROUND_ROLL_CONFIRMATION.value)

        # --- Time interval ---------------------------------------------------
        reliable = True
        if math.isfinite(sigma_t) and sigma_t > 0.0:
            time_half = self.time_calibrator.half_width(sigma_t) * time_widen
        else:
            # No usable sigma: cannot compute a reliable interval. Flag
            # low-confidence and emit a widened fallback rather than suppress
            # (Req 4.5). The fallback half-width is the calibrated width of one
            # nominal cadence -- an honest "at least one sample interval" floor.
            reliable = False
            fallback_sigma = self.config.nominal_cadence_s
            time_half = self.time_calibrator.half_width(fallback_sigma) * time_widen

        # --- Distance interval ----------------------------------------------
        # Propagate time uncertainty through groundspeed: sigma_d = v * sigma_t.
        sigma_used = sigma_t if (math.isfinite(sigma_t) and sigma_t > 0.0) else self.config.nominal_cadence_s
        speed = abs(groundspeed_at_td_mps) if math.isfinite(groundspeed_at_td_mps) else 0.0
        sigma_d_m = speed * sigma_used
        if sigma_d_m > 0.0:
            dist_half_m = self.distance_calibrator.half_width(sigma_d_m) * dist_widen
        else:
            # Degenerate groundspeed -> distance uncertainty cannot be propagated
            # from timing. Fall back to the distance travelled in one nominal
            # cadence at the propagation speed floor, flag low-confidence.
            reliable = False
            fallback_speed = speed if speed > 0.0 else 1.0
            dist_half_m = (
                self.distance_calibrator.half_width(fallback_speed * self.config.nominal_cadence_s)
                * dist_widen
            )

        if not reliable:
            _downgrade(FailureReason.WIDE_CONFIDENCE_INTERVAL.value)

        distance_ft = along_runway_distance_m * M_TO_FT
        dist_half_ft = dist_half_m * M_TO_FT

        diagnostics = {
            "coverage_target": self.config.coverage_target,
            "time_multiplier": self.time_calibrator.multiplier,
            "distance_multiplier": self.distance_calibrator.multiplier,
            "gap_widening_factor": gap_factor,
            "starvation_widening_factor": starv_factor,
            "missing_lever_arm_widening_factor": lever_factor,
            "post_transition_starved": starved,
            "lever_arm_missing": bool(lever_arm_missing),
            "reliable_interval": reliable,
            "time_calibration_n": self.time_calibrator.n_calibration,
            "distance_calibration_n": self.distance_calibrator.n_calibration,
        }

        return UncertaintyResult(
            t_td=t_td,
            time_ci_90_lower_s=t_td - time_half,
            time_ci_90_upper_s=t_td + time_half,
            along_runway_distance_ft=distance_ft,
            distance_ci_90_lower_ft=distance_ft - dist_half_ft,
            distance_ci_90_upper_ft=distance_ft + dist_half_ft,
            confidence=confidence,
            reason_code=reason_code,
            diagnostics=diagnostics,
        )
