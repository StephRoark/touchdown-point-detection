"""Output-record assembly: the single SI->presentation boundary (Task 20).

This module turns a fused touchdown *time* estimate into the final
:class:`~tdz.models.TouchdownResult` output record. It is the one place in the
pipeline where internal SI quantities (meters, m/s, seconds) are converted to
the reported presentation units (feet, knots); everything upstream stays in SI.

Flow (per flight)
-----------------
1. **No-estimate short-circuit.** When the fused estimate is a no-estimate,
   assemble a record carrying the confidence class and a non-null reason code
   (Req 14.1, 14.3, 14.4) plus the diagnostics/provenance -- the primary numeric
   fields are ``NaN`` (there is no touchdown to place).
2. **Map time -> geometry + speed** (:func:`tdz.geo.map_touchdown`): along-runway
   distance and lateral offset (Req 2.1, 2.2), the pitch-resolved lever-arm
   correction (Req 2.3), the interpolated groundspeed (Req 3.2) and its sigma
   propagated from ``sigma_t`` (Req 3.3), and the non-fatal position gates
   (Req 2.4, 2.5).
3. **Quantify uncertainty** (:class:`tdz.uncertainty.UncertaintyQuantifier`): the
   calibrated, widened time (s) and along-runway distance (ft) 90 % intervals,
   centered on the mapped geometry (Req 4.1, 4.2).
4. **Presentation conversion + rounding**: groundspeed to knots at
   ``speed_resolution_kt`` clamped to the plausible band (Req 3.1), lateral
   offset to feet, and the propagated speed 90 % interval to knots (Req 3.3).
5. **Confidence resolution**: fold the speed-, out-of-bounds- and wrong-runway
   flags into the confidence class, always attaching a reason code when the
   class is not ``"normal"`` (Req 3.4, 2.4, 2.5, 14.3, 14.4).
6. **Assemble** the :class:`TouchdownResult` with every diagnostic and
   provenance field populated (Req 14.4, 15.3).

Provenance
----------
:func:`resolve_provenance` derives the ``config_hash`` from the fully-resolved
configuration and the ``code_commit`` from the local git checkout (degrading to
``"unknown"`` when git is unavailable), so every output record carries the
information needed to reproduce the run (Req 15.3).
"""

from __future__ import annotations

import hashlib
import math
import subprocess
from dataclasses import dataclass
from typing import Final, Optional

from tdz.config.schema import TDZConfig
from tdz.estimators.physics.base import CONFIDENCE_LOW, CONFIDENCE_NORMAL
from tdz.fusion.ensemble import CONFIDENCE_NO_ESTIMATE
from tdz.geo import TouchdownMapping, map_touchdown
from tdz.geo.lever_arm import LeverArmCorrection
from tdz.models import FailureReason, FlightRecord, FusedEstimate, TouchdownResult
from tdz.timebase.interpolation import KNOTS_TO_MPS
from tdz.uncertainty import M_TO_FT, UncertaintyQuantifier, gaussian_multiplier

__all__ = [
    "PHYSICS_ANCHOR_PRIORITY",
    "Provenance",
    "compute_config_hash",
    "resolve_provenance",
    "assemble_touchdown_result",
]

#: Preference order for selecting the physics anchor to record on the output
#: (Req 6): the decel-knee anchor first, then the vertical estimators. The first
#: present in ``per_estimator_results`` is recorded.
PHYSICS_ANCHOR_PRIORITY: Final[tuple[str, ...]] = (
    "decel_knee",
    "flare_crossing",
    "imm_rts",
)

#: Placeholder used when a provenance component cannot be resolved.
_UNKNOWN: Final[str] = "unknown"


# ---------------------------------------------------------------------------
# Provenance (Req 15.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """Reproducibility provenance recorded on every output (Req 15.3)."""

    data_version: str
    code_commit: str
    config_hash: str
    model_artifact_hash: Optional[str] = None


def compute_config_hash(config: TDZConfig) -> str:
    """Return a stable SHA-256 hash of the fully-resolved configuration.

    Hashes the canonical (sorted-key) YAML serialization of the resolved config
    so the same parameter set -- including any applied defaults -- always yields
    the same hash (Req 20.5 / 15.3).
    """
    payload = config.to_yaml().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str:
    """Best-effort local git commit hash; ``"unknown"`` when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return _UNKNOWN
    commit = result.stdout.strip()
    return commit if (result.returncode == 0 and commit) else _UNKNOWN


def resolve_provenance(
    config: TDZConfig,
    *,
    data_version: str = _UNKNOWN,
    code_commit: Optional[str] = None,
    model_artifact_hash: Optional[str] = None,
) -> Provenance:
    """Build a :class:`Provenance` for a run (Req 15.3).

    ``config_hash`` is derived from the resolved config; ``code_commit`` is taken
    from the local git checkout unless supplied explicitly (tests pass it to stay
    hermetic). ``data_version`` and ``model_artifact_hash`` are supplied by the
    caller (they are external to the code/config).
    """
    return Provenance(
        data_version=data_version,
        code_commit=code_commit if code_commit is not None else _git_commit(),
        config_hash=compute_config_hash(config),
        model_artifact_hash=model_artifact_hash,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round_to_resolution(value: float, resolution: float) -> float:
    """Round ``value`` to the nearest multiple of ``resolution``."""
    if not math.isfinite(value) or resolution <= 0.0:
        return value
    return round(value / resolution) * resolution


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to ``[low, high]`` (finite inputs only)."""
    return max(low, min(high, value))


def _extract_physics_anchor(
    fused: FusedEstimate,
) -> tuple[Optional[float], Optional[dict]]:
    """Return ``(t_td, diagnostics)`` of the preferred physics anchor (Req 6)."""
    results = fused.per_estimator_results or {}
    for name in PHYSICS_ANCHOR_PRIORITY:
        est = results.get(name)
        if est is not None:
            return float(est.t_td), dict(est.diagnostics or {})
    return None, None


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------


def assemble_touchdown_result(
    flight: FlightRecord,
    fused: FusedEstimate,
    config: TDZConfig,
    *,
    lever_arm: LeverArmCorrection,
    trajectory_type: str,
    provenance: Provenance,
    geometric_altitude_available: bool,
    uncertainty_quantifier: Optional[UncertaintyQuantifier] = None,
    interpolation_method: Optional[str] = None,
) -> TouchdownResult:
    """Assemble the final :class:`TouchdownResult` for one flight (Task 20).

    Parameters
    ----------
    flight:
        The flight record (trajectory + runway geometry).
    fused:
        The fused touchdown-time estimate (Task 18).
    config:
        The resolved configuration (``output`` / ``uncertainty`` / ``validation``
        sections drive the presentation band, the CIs, and the wrong-runway
        margin).
    lever_arm:
        The resolved :class:`LeverArmCorrection` (Task 6); its low-confidence
        flag drives the missing-lever-arm distance-CI widening.
    trajectory_type:
        The classified trajectory (``"completed-landing"`` / ``"go-around"`` /
        ``"touch-and-go"``).
    provenance:
        The reproducibility provenance (see :func:`resolve_provenance`).
    geometric_altitude_available:
        Whether the source provided true HAE altitude (vertical estimators ran).
    uncertainty_quantifier:
        Optional pre-built quantifier (e.g. with fitted calibrators); a Gaussian
        fallback quantifier is built from ``config.uncertainty`` when omitted.
    interpolation_method:
        Position interpolation method; defaults to ``config.timebase.interpolation_method``.

    Returns
    -------
    TouchdownResult
        The complete output record, always carrying a confidence classification
        and (when not ``"normal"``) a reason code, plus provenance (Req 14.3,
        14.4, 15.3).
    """
    output = config.output
    runway_datum = str(flight.runway.elevation_datum)
    physics_anchor_t_td, physics_anchor_diag = _extract_physics_anchor(fused)

    # --- No-estimate: no touchdown to place (Req 14.1, 14.3, 14.4) --------
    if fused.confidence == CONFIDENCE_NO_ESTIMATE:
        reason_code = fused.reason_code or FailureReason.ALL_ESTIMATORS_FAILED.value
        nan = float("nan")
        return TouchdownResult(
            flight_id=flight.flight_id,
            aircraft_type=flight.aircraft_type,
            ads_b_source=flight.ads_b_source,
            touchdown_time=nan,
            along_runway_distance_ft=nan,
            lateral_offset_ft=nan,
            groundspeed_at_touchdown_kt=nan,
            time_ci_90_lower=nan,
            time_ci_90_upper=nan,
            distance_ci_90_lower_ft=nan,
            distance_ci_90_upper_ft=nan,
            speed_ci_90_lower_kt=nan,
            speed_ci_90_upper_kt=nan,
            trajectory_type=trajectory_type,
            confidence=CONFIDENCE_NO_ESTIMATE,
            reason_code=reason_code,
            contributing_estimators=list(fused.contributing_estimators),
            excluded_estimators=list(fused.excluded_estimators),
            physics_anchor_t_td=physics_anchor_t_td,
            physics_anchor_diagnostics=physics_anchor_diag,
            lever_arm_used=lever_arm.lever_arm,
            lever_arm_missing=bool(lever_arm.low_confidence),
            assumed_touchdown_pitch_deg=float(lever_arm.assumed_pitch_deg),
            geometric_altitude_available=bool(geometric_altitude_available),
            runway_elevation_datum=runway_datum,
            suspected_wrong_runway=False,
            out_of_bounds=False,
            data_version=provenance.data_version,
            code_commit=provenance.code_commit,
            config_hash=provenance.config_hash,
            model_artifact_hash=provenance.model_artifact_hash,
        )

    # --- Map fused time -> SI geometry + speed ----------------------------
    method = interpolation_method or config.timebase.interpolation_method
    mapping: TouchdownMapping = map_touchdown(
        flight,
        float(fused.t_td),
        float(fused.sigma_t),
        lever_arm=lever_arm,
        speed_min_mps=output.speed_min_kt * KNOTS_TO_MPS,
        speed_max_mps=output.speed_max_kt * KNOTS_TO_MPS,
        velocity_gap_max_s=output.speed_velocity_gap_max_s,
        validation_config=config.validation,
        interpolation_method=method,
    )

    # --- Uncertainty: calibrated, widened time & distance CIs (SI->ft) ----
    quantifier = uncertainty_quantifier or UncertaintyQuantifier(config.uncertainty)
    unc = quantifier.quantify(
        fused,
        flight,
        groundspeed_at_td_mps=mapping.groundspeed_mps,
        along_runway_distance_m=mapping.along_runway_distance_m,
        lever_arm_missing=bool(lever_arm.low_confidence),
    )
    # ``unc`` is never None here: a no-estimate fused input is handled above.
    assert unc is not None

    # --- Presentation: SI -> feet / knots (single conversion boundary) ----
    resolution = output.speed_resolution_kt
    raw_speed_kt = (
        mapping.groundspeed_mps / KNOTS_TO_MPS
        if math.isfinite(mapping.groundspeed_mps)
        else float("nan")
    )
    if math.isfinite(raw_speed_kt):
        speed_kt = _round_to_resolution(
            _clamp(raw_speed_kt, output.speed_min_kt, output.speed_max_kt), resolution
        )
    else:
        speed_kt = float("nan")

    # Speed 90% CI from propagated sigma_v (Req 3.3), same coverage as time/dist.
    z90 = gaussian_multiplier(config.uncertainty.coverage_target)
    sigma_v_kt = mapping.groundspeed_sigma_mps / KNOTS_TO_MPS
    speed_half_kt = z90 * sigma_v_kt if math.isfinite(sigma_v_kt) else float("nan")
    if math.isfinite(raw_speed_kt) and math.isfinite(speed_half_kt):
        speed_ci_lower_kt = _round_to_resolution(raw_speed_kt - speed_half_kt, resolution)
        speed_ci_upper_kt = _round_to_resolution(raw_speed_kt + speed_half_kt, resolution)
    else:
        speed_ci_lower_kt = speed_kt
        speed_ci_upper_kt = speed_kt

    lateral_offset_ft = mapping.lateral_offset_m * M_TO_FT

    # --- Confidence resolution (Req 3.4, 2.4, 2.5, 14.3, 14.4) ------------
    confidence = unc.confidence
    reason_code = unc.reason_code

    def _downgrade(reason: str) -> None:
        nonlocal confidence, reason_code
        confidence = CONFIDENCE_LOW
        if reason_code is None:
            reason_code = reason

    if mapping.gates.out_of_bounds:
        _downgrade(FailureReason.OUT_OF_BOUNDS_POSITION.value)
    if mapping.gates.suspected_wrong_runway:
        _downgrade(FailureReason.SUSPECTED_WRONG_RUNWAY.value)
    if mapping.speed_low_confidence and mapping.speed_reason_code is not None:
        _downgrade(mapping.speed_reason_code.value)

    return TouchdownResult(
        flight_id=flight.flight_id,
        aircraft_type=flight.aircraft_type,
        ads_b_source=flight.ads_b_source,
        touchdown_time=float(fused.t_td),
        along_runway_distance_ft=unc.along_runway_distance_ft,
        lateral_offset_ft=lateral_offset_ft,
        groundspeed_at_touchdown_kt=speed_kt,
        time_ci_90_lower=unc.time_ci_90_lower_s,
        time_ci_90_upper=unc.time_ci_90_upper_s,
        distance_ci_90_lower_ft=unc.distance_ci_90_lower_ft,
        distance_ci_90_upper_ft=unc.distance_ci_90_upper_ft,
        speed_ci_90_lower_kt=speed_ci_lower_kt,
        speed_ci_90_upper_kt=speed_ci_upper_kt,
        trajectory_type=trajectory_type,
        confidence=confidence,
        reason_code=reason_code,
        contributing_estimators=list(fused.contributing_estimators),
        excluded_estimators=list(fused.excluded_estimators),
        physics_anchor_t_td=physics_anchor_t_td,
        physics_anchor_diagnostics=physics_anchor_diag,
        lever_arm_used=lever_arm.lever_arm,
        lever_arm_missing=bool(lever_arm.low_confidence),
        assumed_touchdown_pitch_deg=float(lever_arm.assumed_pitch_deg),
        geometric_altitude_available=bool(geometric_altitude_available),
        runway_elevation_datum=runway_datum,
        suspected_wrong_runway=bool(mapping.gates.suspected_wrong_runway),
        out_of_bounds=bool(mapping.gates.out_of_bounds),
        data_version=provenance.data_version,
        code_commit=provenance.code_commit,
        config_hash=provenance.config_hash,
        model_artifact_hash=provenance.model_artifact_hash,
    )
