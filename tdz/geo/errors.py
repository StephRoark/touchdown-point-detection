"""Geometry/geodesy error types.

Runway-reference validation is fail-fast: the first field that is missing,
null, NaN, or out of bounds raises :class:`InvalidRunwayReferenceError`. The
message names the offending field and the violated bound, and the exception
carries the associated :class:`~tdz.models.FailureReason` reason code so the
caller can reject the flight (Requirement 11.5 / Property 18).
"""

from __future__ import annotations

from tdz.models import FailureReason


class GeoError(Exception):
    """Base class for geometry/geodesy problems."""


class InvalidRunwayReferenceError(GeoError):
    """Raised when a :class:`~tdz.models.RunwayReference` field is invalid.

    A field is invalid when it is missing/``None``, ``NaN``, or outside the
    bounds defined by Requirement 11.5. The message identifies the offending
    field and the constraint it violated (e.g.
    ``threshold_lat: must be within [-90, 90], got 95.0``).

    Attributes
    ----------
    field:
        Name of the offending :class:`RunwayReference` field.
    constraint:
        Human-readable description of the violated bound.
    reason_code:
        Always :attr:`FailureReason.INVALID_RUNWAY_REF`; the flight is rejected
        and no touchdown estimate is produced.
    """

    reason_code: FailureReason = FailureReason.INVALID_RUNWAY_REF

    def __init__(self, field: str, constraint: str) -> None:
        self.field = field
        self.constraint = constraint
        self.reason_code = FailureReason.INVALID_RUNWAY_REF
        super().__init__(f"{field}: {constraint}")


class DatumUnresolvedError(GeoError):
    """Raised when a runway's vertical datum cannot be resolved to HAE.

    The deterministic geoid/datum conversion (Task 5, Requirement 11.2) needs a
    recognized ``elevation_datum`` and, for an orthometric (MSL) elevation, a
    finite geoid undulation. The datum is *unresolved* when:

    * ``elevation_datum`` is missing and no configured default is available;
    * ``elevation_datum`` is an unrecognized value (not ``"HAE"`` or ``"MSL"``);
    * the elevation is tagged MSL but ``geoid_undulation_m`` is
      missing/``None``/``NaN``/infinite and no geoid model is available to look
      it up; or
    * the conversion would not produce a finite HAE elevation.

    Rather than silently comparing an MSL elevation against HAE geometric
    altitude (which would inject a tens-of-metres vertical bias), the flight is
    rejected.

    Attributes
    ----------
    detail:
        Human-readable description of what could not be resolved.
    reason_code:
        Always :attr:`FailureReason.DATUM_UNRESOLVED`.
    """

    reason_code: FailureReason = FailureReason.DATUM_UNRESOLVED

    def __init__(self, detail: str) -> None:
        self.detail = detail
        self.reason_code = FailureReason.DATUM_UNRESOLVED
        super().__init__(detail)
