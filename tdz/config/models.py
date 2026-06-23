"""Configuration-side data models.

These dataclasses describe configuration/lookup inputs that gate or
parameterize the pipeline. They are deliberately free of any dependency on the
pipeline-internal models (:mod:`tdz.models`) so that importing configuration
types never creates an import cycle.

Units convention: all numeric fields are held in SI units (meters, degrees for
angles as noted). Conversion to presentation units (feet, knots) happens only
at the output boundary (see :class:`tdz.models.TouchdownResult`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LeverArm:
    """Aircraft-type-specific antenna-to-gear offset.

    Horizontal ground-distance correction =
    ``longitudinal_offset_m * cos(pitch) + vertical_offset_m * sin(pitch)``,
    using ``nominal_touchdown_pitch_deg`` (pitch is NOT observable in ADS-B).

    When a type-specific entry is missing, the class-median values are used
    (``is_class_default=True``), the estimate is marked low-confidence, and the
    distance CI is widened to span the class range.
    """

    icao_type: str                          # ICAO type designator (e.g. "B738")
    vertical_offset_m: float                # Antenna height above main gear (meters)
    longitudinal_offset_m: float            # Antenna forward(+)/aft(-) of main gear (meters)
    nominal_touchdown_pitch_deg: float      # Assumed pitch at touchdown for this type (degrees)
    aircraft_class: str                     # "regional" | "narrowbody" | "widebody"
    is_class_default: bool = False          # True if filled from class median (type-specific value absent)


@dataclass
class SourceCapability:
    """Per-source descriptor that gates which estimators may run."""

    source: str                     # "aireon" | "fr24"
    has_geometric_altitude: bool    # True only if true HAE altitude is provided
    samples_are_raw: bool           # False if provider-interpolated/smoothed
    async_timestamps: bool          # True if position/velocity carry separate times
