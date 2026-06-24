"""Position gates: wrong-runway lateral offset and out-of-bounds (Task 7).

After the mapping layer projects a touchdown onto the runway centerline
(:class:`~tdz.geo.projection.ProjectedPosition`), two sanity gates examine the
projected position and raise **non-fatal flags** for the output diagnostics:

* **Wrong-runway lateral-offset gate** (Req 2.5 / Property 22) -- flags
  ``suspected_wrong_runway`` when the magnitude of the lateral offset exceeds
  half the runway width plus a configurable margin. A large lateral offset
  usually signals runway mis-assignment (e.g. a parallel-runway swap) or
  erroneous geometry rather than a genuine off-centerline touchdown.

* **Out-of-bounds along-runway gate** (Req 2.4) -- flags ``out_of_bounds`` when
  the along-runway distance is past the runway end (``> length_m``) or before
  the threshold (``< 0``).

Both gates are **flags, not failures**: unlike the fatal
:class:`~tdz.geo.errors.InvalidRunwayReferenceError` (which rejects the flight
and produces no estimate), these gates never raise on the geometry they
inspect. The estimate is still produced, the computed value is still reported
for diagnostics, and the gate only annotates the result with a boolean flag and
a :class:`~tdz.models.FailureReason` code. The out-of-bounds gate in particular
does **not** clamp or drop the value -- it reports it verbatim (Req 2.4).

Boundary conventions
--------------------
* Wrong-runway: the comparison is **strictly greater-than**
  (``abs(offset) > half_width + margin``). An offset *exactly* at the threshold
  is **not** flagged. Sign is irrelevant -- left/right offsets are compared by
  magnitude.
* Out-of-bounds: the in-bounds interval is the **inclusive** ``[0, length_m]``.
  A distance of exactly ``0`` (at the threshold) or exactly ``length_m`` (at the
  far end) is in-bounds; only ``< 0`` or ``> length_m`` trips the gate.

Units
-----
SI internally (meters). The margin is the one config value expressed in
**feet** (``ValidationConfig.wrong_runway_lateral_margin_ft``); it is converted
to meters with the documented :data:`FT_TO_M` constant before being compared
against the SI ``lateral_offset_m``. No conversion to feet happens here (that is
the output boundary, Task 20).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

from tdz.config.schema import ValidationConfig
from tdz.geo.projection import ProjectedPosition
from tdz.models import FailureReason, RunwayReference

__all__ = [
    "FT_TO_M",
    "PositionGateResult",
    "wrong_runway_lateral_threshold_m",
    "is_suspected_wrong_runway",
    "is_out_of_bounds",
    "resolve_wrong_runway_margin_m",
    "evaluate_position_gates",
]

#: International-foot to meter conversion factor (exact, by definition).
#: The wrong-runway margin is the one config value carried in feet; everything
#: else is SI, so the margin is converted to meters with this constant before it
#: is compared against ``lateral_offset_m``.
FT_TO_M: Final[float] = 0.3048


@dataclass(frozen=True)
class PositionGateResult:
    """Outcome of the position gates for one projected touchdown (meters).

    Immutable value object. All distances are SI (meters). The flags are
    non-fatal: the estimate is still produced and reported regardless of their
    value; ``reason_codes`` is the subset of :class:`FailureReason` codes the
    output/diagnostics record should carry.

    Attributes
    ----------
    suspected_wrong_runway:
        ``True`` when ``abs(lateral_offset_m) > lateral_threshold_m`` (Req 2.5).
    out_of_bounds:
        ``True`` when ``along_runway_distance_m`` is ``< 0`` or
        ``> runway_length_m`` (Req 2.4).
    lateral_offset_m:
        The signed lateral offset inspected (carried through verbatim).
    along_runway_distance_m:
        The signed along-runway distance inspected (carried through verbatim;
        never clamped or dropped, even when out of bounds).
    lateral_threshold_m:
        The threshold used for the wrong-runway gate: half-width + margin.
    margin_m:
        The wrong-runway margin actually applied (meters).
    runway_length_m:
        The runway length used for the out-of-bounds gate.
    runway_width_m:
        The runway width used for the wrong-runway gate.
    reason_codes:
        Tuple of triggered :class:`FailureReason` codes, in a stable order:
        :attr:`FailureReason.OUT_OF_BOUNDS_POSITION` (if any) then
        :attr:`FailureReason.SUSPECTED_WRONG_RUNWAY` (if any). Empty when both
        gates pass.
    """

    suspected_wrong_runway: bool
    out_of_bounds: bool
    lateral_offset_m: float
    along_runway_distance_m: float
    lateral_threshold_m: float
    margin_m: float
    runway_length_m: float
    runway_width_m: float
    reason_codes: tuple[FailureReason, ...]


def wrong_runway_lateral_threshold_m(width_m: float, margin_m: float) -> float:
    """Return the wrong-runway lateral threshold: ``width_m / 2 + margin_m``.

    Parameters
    ----------
    width_m:
        Runway width (meters).
    margin_m:
        Additional lateral margin beyond the half-width (meters).

    Returns
    -------
    float
        Half the runway width plus the margin, in meters.
    """
    return width_m / 2.0 + margin_m


def is_suspected_wrong_runway(
    lateral_offset_m: float, *, width_m: float, margin_m: float
) -> bool:
    """Return whether the lateral offset trips the wrong-runway gate.

    The test is on **magnitude** and is **strictly greater-than**: an offset of
    either sign whose magnitude exceeds ``width_m / 2 + margin_m`` is flagged; an
    offset exactly at the threshold is not (Req 2.5 / Property 22).
    """
    return abs(lateral_offset_m) > wrong_runway_lateral_threshold_m(width_m, margin_m)


def is_out_of_bounds(along_runway_distance_m: float, *, length_m: float) -> bool:
    """Return whether the along-runway distance is out of bounds.

    Out of bounds means past the runway end (``> length_m``) or before the
    threshold (``< 0``). The in-bounds interval ``[0, length_m]`` is inclusive
    at both ends (Req 2.4).
    """
    return along_runway_distance_m < 0.0 or along_runway_distance_m > length_m


def resolve_wrong_runway_margin_m(
    *,
    validation_config: Optional[ValidationConfig] = None,
    wrong_runway_margin_m: Optional[float] = None,
) -> float:
    """Resolve the wrong-runway margin (meters) from the supplied inputs.

    Exactly one source is required; there is **no** hard-coded margin default in
    this module (the 50 ft default lives in the configuration schema). An
    explicit ``wrong_runway_margin_m`` (already in meters) takes precedence; it
    is useful for overrides and tests. Otherwise the margin is read from
    ``validation_config.wrong_runway_lateral_margin_ft`` -- the one config value
    carried in **feet** -- and converted to meters with :data:`FT_TO_M`.

    Parameters
    ----------
    validation_config:
        The validation config carrying ``wrong_runway_lateral_margin_ft``.
    wrong_runway_margin_m:
        An explicit margin in meters (overrides ``validation_config``).

    Returns
    -------
    float
        The margin in meters.

    Raises
    ------
    ValueError
        When neither a margin nor a config is supplied.
    """
    if wrong_runway_margin_m is not None:
        return float(wrong_runway_margin_m)
    if validation_config is not None:
        return float(validation_config.wrong_runway_lateral_margin_ft) * FT_TO_M
    raise ValueError(
        "a wrong-runway margin is required: pass wrong_runway_margin_m (meters) "
        "or validation_config (wrong_runway_lateral_margin_ft, feet)"
    )


def evaluate_position_gates(
    projected: ProjectedPosition,
    runway: RunwayReference,
    *,
    validation_config: Optional[ValidationConfig] = None,
    wrong_runway_margin_m: Optional[float] = None,
) -> PositionGateResult:
    """Evaluate the wrong-runway and out-of-bounds gates for a projected point.

    Both gates are non-fatal: this function never raises on the projected
    geometry it inspects (it raises only if no margin source is supplied). The
    inspected values are carried through verbatim -- the out-of-bounds gate does
    not clamp or drop the along-runway distance (Req 2.4).

    The wrong-runway margin must be supplied via ``wrong_runway_margin_m``
    (meters) or ``validation_config`` (feet, converted with :data:`FT_TO_M`); an
    explicit meters margin takes precedence. See
    :func:`resolve_wrong_runway_margin_m`.

    Parameters
    ----------
    projected:
        The projected touchdown position (along-runway distance + lateral
        offset, meters).
    runway:
        Runway geometry; ``width_m`` drives the wrong-runway threshold and
        ``length_m`` drives the out-of-bounds gate.
    validation_config:
        Validation config carrying ``wrong_runway_lateral_margin_ft`` (feet).
    wrong_runway_margin_m:
        Explicit margin in meters (overrides ``validation_config``).

    Returns
    -------
    PositionGateResult
        The two flags, the inspected values, the thresholds used, and the tuple
        of triggered :class:`FailureReason` codes for the diagnostics record.

    Raises
    ------
    ValueError
        When neither a margin nor a config is supplied.
    """
    margin_m = resolve_wrong_runway_margin_m(
        validation_config=validation_config,
        wrong_runway_margin_m=wrong_runway_margin_m,
    )

    width_m = float(runway.width_m)
    length_m = float(runway.length_m)
    lateral_offset_m = float(projected.lateral_offset_m)
    along_runway_distance_m = float(projected.along_runway_distance_m)

    lateral_threshold_m = wrong_runway_lateral_threshold_m(width_m, margin_m)
    suspected_wrong_runway = abs(lateral_offset_m) > lateral_threshold_m
    out_of_bounds = is_out_of_bounds(along_runway_distance_m, length_m=length_m)

    codes: list[FailureReason] = []
    if out_of_bounds:
        codes.append(FailureReason.OUT_OF_BOUNDS_POSITION)
    if suspected_wrong_runway:
        codes.append(FailureReason.SUSPECTED_WRONG_RUNWAY)

    return PositionGateResult(
        suspected_wrong_runway=suspected_wrong_runway,
        out_of_bounds=out_of_bounds,
        lateral_offset_m=lateral_offset_m,
        along_runway_distance_m=along_runway_distance_m,
        lateral_threshold_m=lateral_threshold_m,
        margin_m=margin_m,
        runway_length_m=length_m,
        runway_width_m=width_m,
        reason_codes=tuple(codes),
    )
