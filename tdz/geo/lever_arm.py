"""Pitch-resolved lever-arm correction (Task 6).

The GNSS antenna sits some distance from the main-gear contact point: a
**vertical** offset ``V`` (antenna height above the gear, meters) and a
**longitudinal** offset ``X`` (antenna forward ``+`` / aft ``-`` of the gear,
meters). Because pitch is *not* observable in ADS-B, the longitudinal arm is
projected onto the ground using the per-type **nominal touchdown pitch** ``θ``
(degrees in config, radians internally). This module turns a
:class:`~tdz.config.models.LeverArm` into the two corrections needed when the
mapping layer (Task 20) converts a touchdown *time* to a touchdown *position*:

* **Along-runway ground-distance correction** ``X·cos θ + V·sin θ`` -- the FULL
  horizontal term, including the height-induced component ``V·sin θ`` (a nose-up
  attitude swings an elevated antenna forward of the gear), not the longitudinal
  term ``X·cos θ`` alone (Req 2.3 / 7.2; Property 2).
* **Altitude-crossing target correction** ``V`` -- added to the crossing-target
  elevation (equivalently subtracted from the antenna's geometric altitude) so
  the altitude-crossing solution corresponds to main-gear contact, not the
  antenna (Req 7.2 / 17.4).

Sign / direction convention
----------------------------
``along_runway_shift_m`` is a **magnitude along the landing direction** using the
same positive sense as :attr:`ProjectedPosition.along_runway_distance_m`
(positive = past the threshold in the landing direction, Req 11.3). With a
forward-mounted antenna (``X > 0``) the antenna *leads* the main gear, so its
runway projection reads a larger along-runway distance than the gear. The
mapping layer therefore **subtracts** ``along_runway_shift_m`` from the
antenna-projected along-runway distance to recover main-gear contact. This
module only *computes* the shift; it does not itself compose it with the
projection (Task 20 does), and it does not build the confidence interval
(Task 19 does) -- it only reports the widening magnitude and the low-confidence
flag/reason.

Missing-type default
--------------------
When the table has no entry for an ICAO type, the **class-median** lever arm is
used (Req 7.4), falling back to the **global median** across all known entries
if the class is unknown. A class/global median is a *central* default: it never
uses a worst-case (largest-offset) value, which would bias the safety metric
(short = false negatives on overruns, long = false positives -- Req 7.5 /
Property 23). A defaulted estimate is marked low-confidence with
:attr:`FailureReason.MISSING_LEVER_ARM`, records the assumed values / class /
pitch for diagnostics (Req 7.6), and reports a distance-CI widening magnitude
spanning the plausible class lever-arm range (Req 7.4b), gated by
:attr:`LeverArmsConfig.class_default_widens_ci`.

Units: SI throughout (meters; pitch in degrees in config, radians internally).
No conversion to feet happens here (that is the output boundary, Task 20).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Optional

from tdz.config.models import LeverArm
from tdz.config.schema import ClassMedian, LeverArmsConfig
from tdz.geo.errors import LeverArmResolutionError
from tdz.models import FailureReason

__all__ = [
    "LeverArmCorrection",
    "horizontal_ground_correction",
    "compute_lever_arm_correction",
    "resolve_lever_arm",
    "compute_lever_arm_range_widening",
    "resolve_lever_arm_correction",
]

#: Label recorded on a lever arm whose class could not be determined and which
#: was filled from the global median across all table entries.
_UNKNOWN_CLASS: str = "unknown"


@dataclass(frozen=True)
class LeverArmCorrection:
    """Corrections to apply when mapping touchdown time -> position (meters).

    Immutable value object. The inputs are never mutated.

    Attributes
    ----------
    along_runway_shift_m:
        ``X·cos θ + V·sin θ`` -- the full horizontal ground-distance correction
        at the assumed pitch, a magnitude in the landing direction (same
        positive sense as :attr:`ProjectedPosition.along_runway_distance_m`).
        The mapping layer subtracts this from the antenna-projected along-runway
        distance so the result corresponds to main-gear contact.
    altitude_target_shift_m:
        ``V`` -- the vertical offset, added to the altitude-crossing target
        elevation (equivalently subtracted from antenna geometric altitude) so
        the crossing solution is at main-gear contact.
    assumed_pitch_deg:
        The per-type nominal touchdown pitch ``θ`` used (degrees).
    aircraft_class:
        Aircraft class of the lever arm used ("regional"/"narrowbody"/
        "widebody", or "unknown" for a global-median default).
    is_class_default:
        ``True`` when the lever arm was filled from a class/global median
        because no type-specific entry existed.
    lever_arm:
        The resolved :class:`LeverArm` actually used (for diagnostics, Req 7.6).
    pitch_assumed:
        Always ``True`` -- pitch is assumed (a per-type nominal), never measured
        (Req 7.3 diagnostics).
    low_confidence:
        ``True`` when a class/global-median default was applied (Req 7.4a).
    reason_code:
        :attr:`FailureReason.MISSING_LEVER_ARM` when defaulted, else ``None``.
    ci_widening_m:
        Magnitude (meters) by which a downstream step should widen the
        along-runway-distance CI to span the plausible class lever-arm range
        (Req 7.4b). ``0.0`` for a type-specific lever arm, or when
        :attr:`LeverArmsConfig.class_default_widens_ci` is ``False``. This
        module does not build the CI itself (Task 19 does).
    """

    along_runway_shift_m: float
    altitude_target_shift_m: float
    assumed_pitch_deg: float
    aircraft_class: str
    is_class_default: bool
    lever_arm: LeverArm
    pitch_assumed: bool = True
    low_confidence: bool = False
    reason_code: Optional[FailureReason] = None
    ci_widening_m: float = 0.0


def horizontal_ground_correction(
    longitudinal_offset_m: float,
    vertical_offset_m: float,
    pitch_deg: float,
) -> float:
    """Return the horizontal ground-distance correction ``X·cos θ + V·sin θ``.

    Parameters
    ----------
    longitudinal_offset_m:
        Antenna forward(+)/aft(-) offset from the main gear, ``X`` (meters).
    vertical_offset_m:
        Antenna height above the main gear, ``V`` (meters).
    pitch_deg:
        Nominal touchdown pitch ``θ`` (degrees); converted to radians here.

    Returns
    -------
    float
        The full horizontal correction, including the height-induced
        ``V·sin θ`` term (Req 2.3 / 7.2). At ``θ = 0`` this reduces to ``X``.
    """
    theta = math.radians(pitch_deg)
    return longitudinal_offset_m * math.cos(theta) + vertical_offset_m * math.sin(theta)


def compute_lever_arm_correction(
    lever_arm: LeverArm,
    *,
    ci_widening_m: float = 0.0,
) -> LeverArmCorrection:
    """Compute the geometric correction for a single :class:`LeverArm`.

    Pure geometry plus confidence bookkeeping; does not consult the config and
    does not mutate ``lever_arm``. When ``lever_arm.is_class_default`` is
    ``True`` the result is marked low-confidence with
    :attr:`FailureReason.MISSING_LEVER_ARM` and carries ``ci_widening_m``;
    otherwise it is normal-confidence with zero widening.

    Parameters
    ----------
    lever_arm:
        The resolved lever arm (type-specific or class/global-median default).
    ci_widening_m:
        Distance-CI widening magnitude to record (meters). Ignored (forced to
        ``0.0``) unless ``lever_arm.is_class_default`` is ``True``.

    Returns
    -------
    LeverArmCorrection
        The along-runway and altitude corrections plus diagnostics flags.
    """
    along = horizontal_ground_correction(
        lever_arm.longitudinal_offset_m,
        lever_arm.vertical_offset_m,
        lever_arm.nominal_touchdown_pitch_deg,
    )
    if lever_arm.is_class_default:
        low_confidence = True
        reason_code: Optional[FailureReason] = FailureReason.MISSING_LEVER_ARM
        widening = float(ci_widening_m)
    else:
        low_confidence = False
        reason_code = None
        widening = 0.0
    return LeverArmCorrection(
        along_runway_shift_m=along,
        altitude_target_shift_m=float(lever_arm.vertical_offset_m),
        assumed_pitch_deg=float(lever_arm.nominal_touchdown_pitch_deg),
        aircraft_class=lever_arm.aircraft_class,
        is_class_default=lever_arm.is_class_default,
        lever_arm=lever_arm,
        pitch_assumed=True,
        low_confidence=low_confidence,
        reason_code=reason_code,
        ci_widening_m=widening,
    )


def _lever_arm_from_class_median(
    icao_type: str, aircraft_class: str, median: ClassMedian
) -> LeverArm:
    """Build a class-median default :class:`LeverArm` (``is_class_default``)."""
    return LeverArm(
        icao_type=icao_type,
        vertical_offset_m=median.vertical_offset_m,
        longitudinal_offset_m=median.longitudinal_offset_m,
        nominal_touchdown_pitch_deg=median.nominal_touchdown_pitch_deg,
        aircraft_class=aircraft_class,
        is_class_default=True,
    )


def _global_median_lever_arm(icao_type: str, config: LeverArmsConfig) -> LeverArm:
    """Build a global-median default lever arm across all known entries.

    The global median is taken component-wise (vertical, longitudinal, pitch)
    across every type-specific entry in :attr:`LeverArmsConfig.arms`. If the
    table has no type-specific entries, it falls back to the median across the
    configured :attr:`LeverArmsConfig.class_medians`. A median is a central
    (never worst-case) default per Req 7.5. Raises if neither source exists.
    """
    arms = list(config.arms.values())
    if arms:
        vertical = statistics.median(a.vertical_offset_m for a in arms)
        longitudinal = statistics.median(a.longitudinal_offset_m for a in arms)
        pitch = statistics.median(a.nominal_touchdown_pitch_deg for a in arms)
    elif config.class_medians:
        medians = list(config.class_medians.values())
        vertical = statistics.median(m.vertical_offset_m for m in medians)
        longitudinal = statistics.median(m.longitudinal_offset_m for m in medians)
        pitch = statistics.median(m.nominal_touchdown_pitch_deg for m in medians)
    else:
        raise LeverArmResolutionError(
            f"no lever arm for type {icao_type!r}: the lever-arm table has no "
            "type entries and no class medians to derive a global default from"
        )
    return LeverArm(
        icao_type=icao_type,
        vertical_offset_m=vertical,
        longitudinal_offset_m=longitudinal,
        nominal_touchdown_pitch_deg=pitch,
        aircraft_class=_UNKNOWN_CLASS,
        is_class_default=True,
    )


def resolve_lever_arm(
    icao_type: str,
    config: LeverArmsConfig,
    aircraft_class: Optional[str] = None,
) -> LeverArm:
    """Resolve the :class:`LeverArm` to use for an ICAO type.

    Resolution order (Req 7.4):

    1. **Type-specific** entry in :attr:`LeverArmsConfig.arms` -> used as-is
       (``is_class_default=False``).
    2. **Class median** from :attr:`LeverArmsConfig.class_medians` for the
       given ``aircraft_class`` -> a default lever arm (``is_class_default=True``).
    3. **Global median** across all known entries when the class is unknown or
       has no configured median -> a default lever arm (``is_class_default=True``,
       ``aircraft_class="unknown"``).

    Contract: the caller (ingest layer) supplies ``aircraft_class`` because the
    class is needed to pick the right class median; the lever-arm table itself
    is keyed by ICAO type, not class. A type-specific entry takes precedence and
    does not require ``aircraft_class``.

    Parameters
    ----------
    icao_type:
        ICAO type designator (e.g. ``"B738"``).
    config:
        The lever-arm table / class medians / CI policy.
    aircraft_class:
        The flight's aircraft class, if known. Used only to select a class
        median when no type-specific entry exists.

    Returns
    -------
    LeverArm
        The lever arm to use. Never a worst-case (largest-offset) default.

    Raises
    ------
    LeverArmResolutionError
        When no type entry, class median, or global median is available
        (carries :attr:`FailureReason.MISSING_LEVER_ARM`).
    """
    type_specific = config.arms.get(icao_type)
    if type_specific is not None:
        return type_specific

    if aircraft_class is not None:
        median = config.class_medians.get(aircraft_class)
        if median is not None:
            return _lever_arm_from_class_median(icao_type, aircraft_class, median)

    # Class unknown or no median for it -> global-median fallback.
    return _global_median_lever_arm(icao_type, config)


def compute_lever_arm_range_widening(lever_arms: Iterable[LeverArm]) -> float:
    """Return the spread (max - min) of horizontal corrections over lever arms.

    The widening magnitude is the range of ``X·cos θ + V·sin θ`` across the
    supplied lever arms, i.e. how far apart the plausible along-runway
    corrections are. A downstream CI step (Task 19) adds this to the distance
    interval so a defaulted estimate honestly reflects lever-arm uncertainty
    (Req 7.4b). Returns ``0.0`` when fewer than two lever arms are supplied
    (no spread is defined).
    """
    corrections = [
        horizontal_ground_correction(
            a.longitudinal_offset_m, a.vertical_offset_m, a.nominal_touchdown_pitch_deg
        )
        for a in lever_arms
    ]
    if len(corrections) < 2:
        return 0.0
    return max(corrections) - min(corrections)


def _class_widening(config: LeverArmsConfig, aircraft_class: Optional[str]) -> float:
    """Compute the distance-CI widening for a defaulted lever arm.

    Spans the plausible lever-arm range for the class using the type entries
    present in :attr:`LeverArmsConfig.arms` for ``aircraft_class``. If the class
    is unknown or has fewer than two type entries, falls back to the spread
    across **all** table entries (the global range) as a conservative proxy so
    the CI is still honestly widened.
    """
    if aircraft_class is not None and aircraft_class != _UNKNOWN_CLASS:
        class_members: Sequence[LeverArm] = [
            a for a in config.arms.values() if a.aircraft_class == aircraft_class
        ]
        if len(class_members) >= 2:
            return compute_lever_arm_range_widening(class_members)
    # Unknown class, or too few class members: use the full-table range.
    return compute_lever_arm_range_widening(config.arms.values())


def resolve_lever_arm_correction(
    icao_type: str,
    config: LeverArmsConfig,
    aircraft_class: Optional[str] = None,
) -> LeverArmCorrection:
    """Resolve the lever arm for a type and compute its full correction.

    Combines :func:`resolve_lever_arm` (default resolution) with
    :func:`compute_lever_arm_correction` (geometry), and -- for a defaulted
    lever arm -- the class/global CI-widening magnitude. Respects
    :attr:`LeverArmsConfig.class_default_widens_ci`: when ``False``, the
    widening is suppressed (``ci_widening_m == 0``) but the estimate is still
    marked low-confidence with :attr:`FailureReason.MISSING_LEVER_ARM`.

    Parameters
    ----------
    icao_type:
        ICAO type designator.
    config:
        The lever-arm table / class medians / CI policy.
    aircraft_class:
        The flight's aircraft class, if known (see :func:`resolve_lever_arm`).

    Returns
    -------
    LeverArmCorrection
        Along-runway and altitude corrections, plus low-confidence flag, reason
        code, and CI-widening magnitude when a default was applied.

    Raises
    ------
    LeverArmResolutionError
        When no lever arm can be resolved.
    """
    lever_arm = resolve_lever_arm(icao_type, config, aircraft_class)
    widening = 0.0
    if lever_arm.is_class_default and config.class_default_widens_ci:
        # For a class-median default the resolved arm carries the class label;
        # for a global-median default it is "unknown" -> full-table range.
        widening_class = (
            aircraft_class
            if lever_arm.aircraft_class == _UNKNOWN_CLASS
            else lever_arm.aircraft_class
        )
        widening = _class_widening(config, widening_class)
    return compute_lever_arm_correction(lever_arm, ci_widening_m=widening)
