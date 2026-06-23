"""Runway-reference validation (Task 4.3).

Validates the required geometry fields of a :class:`~tdz.models.RunwayReference`
against the bounds defined by Requirement 11.5 / Property 18 before any
projection is attempted. Validation is fail-fast and raises
:class:`InvalidRunwayReferenceError` naming the first offending field; the
exception carries :attr:`FailureReason.INVALID_RUNWAY_REF` so the caller rejects
the flight and produces no estimate.

Only the six fields enumerated by Requirement 11.5 are bounds-checked here:
threshold latitude, threshold longitude, runway heading, threshold elevation,
runway length, and runway width. Datum/geoid resolution is a separate concern
(Task 5) and is not validated in this module.
"""

from __future__ import annotations

import math
from typing import Final

from tdz.geo.errors import InvalidRunwayReferenceError
from tdz.models import RunwayReference

__all__ = ["validate_runway_reference"]

# Inclusive bounds per Requirement 11.5 / Property 18. These are specification
# constants (acceptance-criteria limits), not tunable estimation parameters.
_LAT_MIN: Final[float] = -90.0
_LAT_MAX: Final[float] = 90.0
_LON_MIN: Final[float] = -180.0
_LON_MAX: Final[float] = 180.0
_HEADING_MIN: Final[float] = 0.0
_HEADING_MAX: Final[float] = 360.0
_ELEVATION_MIN_M: Final[float] = -500.0
_ELEVATION_MAX_M: Final[float] = 10000.0
# Length/width are strictly positive (a zero-extent runway is meaningless).
_LENGTH_MAX_M: Final[float] = 6000.0
_WIDTH_MAX_M: Final[float] = 100.0


def _require_real(field: str, value: object) -> float:
    """Return ``value`` as a finite float or raise for None/NaN/non-numeric."""
    if value is None:
        raise InvalidRunwayReferenceError(field, "is required but missing (None)")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidRunwayReferenceError(
            field, f"must be a real number, got {type(value).__name__}"
        )
    numeric = float(value)
    if math.isnan(numeric):
        raise InvalidRunwayReferenceError(field, "must be a real number, got NaN")
    if math.isinf(numeric):
        raise InvalidRunwayReferenceError(field, "must be finite, got infinity")
    return numeric


def _check_closed(field: str, value: float, low: float, high: float) -> None:
    """Validate ``low <= value <= high`` (inclusive)."""
    if not (low <= value <= high):
        raise InvalidRunwayReferenceError(
            field, f"must be within [{low}, {high}], got {value}"
        )


def _check_positive_upper(field: str, value: float, high: float) -> None:
    """Validate ``0 < value <= high`` (strictly positive, inclusive upper)."""
    if not (0.0 < value <= high):
        raise InvalidRunwayReferenceError(
            field, f"must be within (0, {high}], got {value}"
        )


def validate_runway_reference(runway: RunwayReference) -> None:
    """Validate a runway reference, raising on the first invalid field.

    Parameters
    ----------
    runway:
        The :class:`RunwayReference` to validate.

    Raises
    ------
    InvalidRunwayReferenceError
        If any required field is missing/``None``, ``NaN``, infinite, or outside
        its Requirement 11.5 bound. The exception names the field, describes the
        violated bound, and exposes :attr:`FailureReason.INVALID_RUNWAY_REF` via
        its ``reason_code`` attribute.

    Notes
    -----
    Returns ``None`` on success. Validation is *validate-and-raise* rather than
    *return-a-result*: Requirement 11.5 says the flight must be rejected with an
    error identifying the field, so a raised exception carrying both the field
    name and the reason code is the natural contract.
    """
    lat = _require_real("threshold_lat", runway.threshold_lat)
    _check_closed("threshold_lat", lat, _LAT_MIN, _LAT_MAX)

    lon = _require_real("threshold_lon", runway.threshold_lon)
    _check_closed("threshold_lon", lon, _LON_MIN, _LON_MAX)

    heading = _require_real("heading_deg", runway.heading_deg)
    _check_closed("heading_deg", heading, _HEADING_MIN, _HEADING_MAX)

    elevation = _require_real("elevation_m", runway.elevation_m)
    _check_closed("elevation_m", elevation, _ELEVATION_MIN_M, _ELEVATION_MAX_M)

    length = _require_real("length_m", runway.length_m)
    _check_positive_upper("length_m", length, _LENGTH_MAX_M)

    width = _require_real("width_m", runway.width_m)
    _check_positive_upper("width_m", width, _WIDTH_MAX_M)
